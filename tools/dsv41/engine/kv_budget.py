#!/usr/bin/env python3
"""KV cache bytes of a DeepSeek-V4.1 GGUF, per layer and per token, as the stock fork allocates them
(src/llama-kv-cache-dsv4.cpp, llama_kv_cache_dsv4::llama_kv_cache_dsv4, and src/llama-kv-cache-iswa.cpp) and as
the engine could store them (SPEC.md, "KV cache").

  kv_budget.py [--ctx N] [--ubatch N] [--type-k f16|bf16|f32|q8_0] HEADER [HEADER ...]

HEADER: the GGUF, or the header of each shard of a split GGUF (the first bytes of every shard: tensor infos are per
shard, so all shards are needed for the layer roles; shard 1 alone carries the metadata). python3 only.

The fork's DeepSeek-V4.1 cache has four parts:
  raw   every layer, K only (MLA style, key_length values per cell), PAD(min(ctx, n_swa + n_ubatch), 256) cells
  csa   the compressed rows, one slot per KV source layer (a layer with attn_compressor_kv), every slot sized for
        the finest ratio: PAD(ceil(ctx / min ratio), 256) cells of key_length values; a coarser source leaves part of
        its slot unused; readers alias their source's slot (no storage)
  lid   the index keys, one slot per key owner (a layer with indexer.attn_k), same cells, indexer.key_length values
  state the pooling carry-over, f32 kv and score rows, ratio rows per source (and per key owner for lid)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
import gguf_header as gh  # noqa: E402

TYPE_BYTES = {"f16": (1, 2), "bf16": (1, 2), "f32": (1, 4), "q8_0": (32, 34)}
MiB = 1024.0 * 1024.0


def pad(n, m):
    return (n + m - 1) // m * m


def row_bytes(type_k, n):
    blck, size = TYPE_BYTES[type_k]
    if n % blck:
        raise gh.GgufError(f"a cache row of {n} values is not whole {type_k} blocks of {blck}")
    return n // blck * size


def read_model(paths):
    md, tensors = {}, set()
    for p in paths:
        h = gh.read_header(p)
        for k, v in h.metadata.items():
            md.setdefault(k, v[1])
        tensors |= {t["name"] for t in h.tensors}
    arch = md.get("general.architecture")
    if arch != "deepseek41":
        raise gh.GgufError(f"{paths[0]}: architecture {arch}, expected deepseek41")
    g = lambda k: md[f"{arch}.{k}"]  # noqa: E731
    n_layer = g("block_count")
    has = lambda i, t: f"blk.{i}.{t}.weight" in tensors  # noqa: E731
    if not has(0, "attn_norm"):
        raise gh.GgufError("blk.0.attn_norm.weight is in none of the headers: pass every shard's header")
    return dict(
        n_layer=n_layer, D=g("attention.key_length"), ID=g("attention.indexer.key_length"),
        n_swa=g("attention.sliding_window"), ratios=g("attention.compress_ratios")[:n_layer],
        top_k=g("attention.indexer.top_k"),
        kv_source=[has(i, "attn_compressor_kv") for i in range(n_layer)],
        key_owner=[has(i, "indexer.attn_k") for i in range(n_layer)],
        index_source=[has(i, "indexer.attn_q_b") for i in range(n_layer)],
        gate=[has(i, "attn_compressor_gate") for i in range(n_layer)])


def budget(m, ctx, ubatch, type_k):
    L, D, ID = m["n_layer"], m["D"], m["ID"]
    ratios = m["ratios"]
    src = [i for i in range(L) if m["kv_source"][i]]
    own = [i for i in range(L) if m["key_owner"][i]]
    used = sorted({ratios[i] for i in range(L) if ratios[i]})
    if len(used) > 2:
        raise gh.GgufError(f"compress ratios {used}: the fork supports at most two")
    r_min = used[0] if used else 1
    # the first ratio met in layer order is the csa state's (llama-kv-cache-dsv4.cpp), which the lid state shares
    r_a = next((ratios[i] for i in range(L) if ratios[i]), 1)

    raw_cells = pad(min(ctx, m["n_swa"] + ubatch), 256)
    comp_cells = pad(max(1, -(-ctx // r_min)), 256)
    rb_d, rb_i = row_bytes(type_k, D), row_bytes(type_k, ID)
    out = dict(raw_cells=raw_cells, comp_cells=comp_cells, r_min=r_min, src=src, own=own,
               raw=L * raw_cells * rb_d, csa=len(src) * comp_cells * rb_d, lid=len(own) * comp_cells * rb_i)
    st = 0
    for i in src:
        r = ratios[i]
        st += 2 * D * r * 4  # kv and score rows, f32, in the csa state (ratio r_a) or the hca state
    st += 2 * ID * r_a * 4 * len(own)
    out["state"] = st
    # bytes the cache grows by per token: every source and key owner adds a row each r_min tokens
    out["fork_per_token"] = (len(src) * rb_d + len(own) * rb_i) / r_min
    # stored at each layer's own ratio (a ratio 2 source fills only every other row of its fork slot)
    out["own_ratio_per_token"] = sum(rb_d / ratios[i] for i in src) + sum(rb_i / ratios[i] for i in own)
    out["dense_per_token"] = L * rb_d
    return out


def checkpoint_bytes(m, positions):
    """The host bytes of one of the engine's conversation checkpoints at `positions` (include/strata/dsv41/engine/
    kv_checkpoint.hpp, kv_checkpoint_bytes, f16 rows): every layer's window cells of the last n_swa - 1 positions,
    each KV source's completed compressed rows and its pooling carry (ratio rows of kv and score, f32), each key
    owner's index keys. A slot is pinned for the --ctx of the engine, its largest checkpoint."""
    D, ID, ratios = m["D"], m["ID"], m["ratios"]
    n = 0
    for i in range(m["n_layer"]):
        n += min(positions, m["n_swa"] - 1) * D * 2
        blocks = positions // ratios[i] if ratios[i] else 0
        if m["kv_source"][i]:
            n += blocks * D * 2 + 2 * ratios[i] * D * 4
        if m["key_owner"][i]:
            n += blocks * ID * 2
    return n


def slot_bytes(m, positions):
    """The device bytes of one of the engine's batch slots, a KV cache of
    `positions` that only decodes (include/strata/dsv41/engine/kv_cache.hpp kv_cache_bytes with max_chunk 1): every
    layer's window ring of n_swa cells, each KV source's rows (ceil(positions / ratio)) and its carry with the in-place
    snapshot (4 x ratio rows f32), each key owner's index keys, each index source's top-k ids of one token."""
    D, ID, ratios = m["D"], m["ID"], m["ratios"]
    n = 0
    for i in range(m["n_layer"]):
        n += m["n_swa"] * D * 2
        cap = -(-positions // ratios[i]) if ratios[i] else 0
        if m["kv_source"][i]:
            n += cap * D * 2 + 4 * ratios[i] * D * 4
        if m["key_owner"][i]:
            n += cap * ID * 2
        if m["index_source"][i]:
            n += m["top_k"] * 4
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--type-k", default="f16", choices=sorted(TYPE_BYTES))
    ap.add_argument("headers", nargs="+")
    args = ap.parse_args()
    try:
        m = read_model(args.headers)
        b = budget(m, args.ctx, args.ubatch, args.type_k)
    except (gh.GgufError, OSError, KeyError) as e:
        print(f"kv_budget.py: {e}", file=sys.stderr)
        return 1
    L = m["n_layer"]
    print(f"model: {L} layers, key_length {m['D']}, indexer key_length {m['ID']}, sliding window {m['n_swa']}, "
          f"ratios used {sorted(set(r for r in m['ratios'] if r))}")
    print(f"roles: kv sources {b['src']}, index key owners {b['own']}, "
          f"index sources {[i for i in range(L) if m['index_source'][i]]}, "
          f"window only {[i for i in range(L) if not m['ratios'][i]]}")
    print(f"ctx {args.ctx}, n_ubatch {args.ubatch}, type_k {args.type_k}")
    print(f"  raw   {L} layers x {b['raw_cells']} cells x {row_bytes(args.type_k, m['D'])} B = {b['raw'] / MiB:9.2f} MiB"
          f"  (fixed: the window does not grow with ctx)")
    print(f"  csa   {len(b['src'])} slots x {b['comp_cells']} cells x {row_bytes(args.type_k, m['D'])} B = "
          f"{b['csa'] / MiB:9.2f} MiB")
    print(f"  lid   {len(b['own'])} slots x {b['comp_cells']} cells x {row_bytes(args.type_k, m['ID'])} B = "
          f"{b['lid'] / MiB:9.2f} MiB")
    print(f"  state {b['state']} B (f32 pooling carry-over, fixed)")
    total = b["raw"] + b["csa"] + b["lid"] + b["state"]
    print(f"  total {total / MiB:9.2f} MiB = {total} B")
    print("per token (growth with ctx):")
    print(f"  fork layout       {b['fork_per_token']:8.1f} B/token = {b['fork_per_token'] / L:6.1f} B/token/layer "
          f"(over {L} layers), {b['fork_per_token'] * args.ctx / MiB:8.2f} MiB at ctx {args.ctx}")
    print(f"  own-ratio layout  {b['own_ratio_per_token']:8.1f} B/token = "
          f"{b['own_ratio_per_token'] / L:6.1f} B/token/layer, {b['own_ratio_per_token'] * args.ctx / MiB:8.2f} MiB")
    print(f"  dense MLA (every layer keeps every position, for scale) {b['dense_per_token']:8.0f} B/token, "
          f"{b['dense_per_token'] * args.ctx / MiB:9.2f} MiB")
    print("per layer, B/token in the fork layout:")
    rb_d, rb_i = row_bytes(args.type_k, m["D"]), row_bytes(args.type_k, m["ID"])
    for i in range(L):
        own = (rb_d if m["kv_source"][i] else 0) + (rb_i if m["key_owner"][i] else 0)
        role = ("window" if not m["ratios"][i] else f"ratio {m['ratios'][i]}") + \
            (" source" if m["kv_source"][i] else "") + (" gate" if m["gate"][i] else "") + \
            (" key-owner" if m["key_owner"][i] else "") + (" index-source" if m["index_source"][i] else "")
        print(f"  L{i:02d} {role:44s} {own / b['r_min']:7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
