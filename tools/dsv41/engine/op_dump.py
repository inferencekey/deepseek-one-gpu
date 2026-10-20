#!/usr/bin/env python3
"""Reads, checks and compares op-dump directories (tools/dsv41/engine/op-dump), with python3 and numpy.

  op_dump.py check DIR                 every file present with its manifest size, every spec op of every layer role
                                       in every ubatch, and every shape equal to its SPEC.md formula
  op_dump.py compare DIR_A DIR_B       same files, same generated tokens; per op the files that are bit-identical,
                                       otherwise the worst relative L2 (idx_top_k and ffn_moe_topk: same index sets)
  op_dump.py map-oracle MANIFEST       the spec op of every source tensor a dsv41-moe-dump manifest (the real model
                                       oracles, tools/dsv41/oracle) recorded
  op_dump.py show DIR OP LAYER UBATCH [OCC]
                                       shape, dtype and statistics of one file
  op_dump.py roles [--ctx N] DIR | GGUF [GGUF ...]
                                       per layer its role (window, index source Full or Reindex, reader), the layers
                                       it reads, and the longest context at which the engine's Reuse Mode and
                                       `--readers dense` (the fork) compute the same thing; for a dump DIR, whether
                                       they coincide over its tokens (SPEC.md, "Reader modes")

The shapes are checked against the model's own GGUF (the manifest's model_path, read with
tools/dsv41/oracle/gguf_header.py): dims come from its metadata, layer roles from which tensors each layer carries
(as llama_model_deepseek41::load_arch_tensors works them out). load(dir, op, layer, ubatch, occ) returns one
file as a numpy array in ggml order reversed ([ne3, ne2, ne1, ne0]) for the engine's parity tests;
Model(...).modes_coincide(n_tokens) tells such a test whether the default Reuse Mode may be checked against the
dump or only `--readers dense`. manifest() and Model need only python3; the commands and load() need numpy.
"""

import json
import math
import os
import sys

try:
    import numpy as np
except ImportError:  # manifest(), Model and the shapes need none (attention_cases.py runs in the builder image)
    np = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
import gguf_header as gh  # noqa: E402

DTYPES = {"f32": "float32", "f16": "float16", "i32": "int32"}
HC = 4


class DumpError(Exception):
    pass


def manifest(d):
    path = os.path.join(d, "dump-manifest.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise DumpError(f"{path}: {e}")


def load(d, op, layer, ubatch, occ=0):
    lay = 999 if layer < 0 else layer
    name = f"{op}.{occ}-L{lay:03d}-u{ubatch:04d}.bin"
    m = manifest(d)
    for f in m["files"]:
        if f["file"] == name:
            a = np.fromfile(os.path.join(d, name), dtype=DTYPES[f["dtype"]])
            return a.reshape(list(reversed(f["ne"])))
    raise DumpError(f"{d}: no {name} in the manifest")


class Model:
    """The dims and layer roles of a DeepSeek-V4.1 GGUF, from its header (or the headers of all its shards)."""

    def __init__(self, *paths):
        md, names = {}, {}
        for p in paths:
            h = gh.read_header(p)
            for k, v in h.metadata.items():
                md.setdefault(k, v[1])
            names.update({t["name"]: t for t in h.tensors})
        a = md.get("general.architecture")
        if a != "deepseek41":
            raise DumpError(f"{paths[0]}: architecture {a}, expected deepseek41")
        g = lambda k: md[f"{a}.{k}"]  # noqa: E731
        if "blk.0.attn_norm.weight" not in names:
            raise DumpError("blk.0.attn_norm.weight is in none of the headers: pass every shard's header")
        self.n_layer = g("block_count")
        self.E, self.H, self.D = g("embedding_length"), g("attention.head_count"), g("attention.key_length")
        self.QL, self.IH, self.ID = g("attention.q_lora_rank"), g("attention.indexer.head_count"), \
            g("attention.indexer.key_length")
        self.top_k = g("attention.indexer.top_k")
        self.OG, self.OL = g("attention.output_group_count"), g("attention.output_lora_rank")
        self.NE, self.K, self.FF = g("expert_count"), g("expert_used_count"), g("expert_feed_forward_length")
        self.NV = names["token_embd.weight"]["ne"][1]
        self.NC = (g("engram.max_ngram_size") - 1) * g("engram.head_count")
        self.KL = g("engram.key_length")
        self.ratios = g("attention.compress_ratios")[: self.n_layer]
        self.engram = set(g("engram.layer_ids"))
        has = lambda i, t: f"blk.{i}.{t}.weight" in names  # noqa: E731
        self.kv_source = [has(i, "attn_compressor_kv") for i in range(self.n_layer)]
        self.key_owner = [has(i, "indexer.attn_k") for i in range(self.n_layer)]
        self.index_source = [has(i, "indexer.attn_q_b") for i in range(self.n_layer)]
        # the layers a compressed layer reads: the latest of each kind at or before it (deepseek41.cpp 209-246)
        self.kv_src, self.key_src, self.topk_src = ([-1] * self.n_layer for _ in range(3))
        last = [-1, -1, -1]
        for i in range(self.n_layer):
            for j, own in enumerate((self.kv_source[i], self.key_owner[i], self.index_source[i])):
                if own:
                    last[j] = i
            if self.ratios[i]:
                self.kv_src[i], self.key_src[i], self.topk_src[i] = last

    def mode(self, il):
        """The layer's tech report mode: window, Full (an index source with its own compressor), Reindex (an index
        source reading another layer's keys) or Reuse (a reader)."""
        if self.ratios[il] == 0:
            return "window"
        if self.index_source[il]:
            return "Full" if self.kv_source[il] else "Reindex"
        return "Reuse"

    def readers(self):
        return [i for i in range(self.n_layer) if self.mode(i) == "Reuse"]

    def coincide_until(self, il):
        """The longest context, in tokens, at which reader il computes the same under Reuse Mode (its index
        source's top-k) and `--readers dense` (every visible compressed row): while the visible rows, (p + 1) // r
        at position p, are at most top_k, the top-k selects them all. None for a layer that is not a reader."""
        if self.mode(il) != "Reuse":
            return None
        return (self.top_k + 1) * self.ratios[il] - 1

    def modes_coincide(self, n_tokens):
        """True when, over a sequence of n_tokens, every reader computes the same in both reader modes, so a parity
        test may run the engine's default against a fork dump; otherwise only `--readers dense` matches the fork."""
        return all(self.coincide_until(i) >= n_tokens for i in self.readers())

    def layer_ops(self, il):
        """The spec ops layer il computes, by role (SPEC.md, "Layer roles")."""
        ops = ["hc_mixes", "hc_pre", "hc_post", "hc_comb", "hc_attn_pre", "attn_norm", "qr", "q", "kv", "kq",
               "kq_soft_max", "kqv", "attn_derope", "attn_wo_a", "attn_out", "hc_attn_post", "hc_ffn_pre",
               "ffn_norm", "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_probs_biased", "ffn_moe_argsort",
               "ffn_moe_topk", "ffn_moe_weights", "ffn_moe_weights_sum", "ffn_moe_weights_sum_clamped",
               "ffn_moe_weights_norm", "ffn_moe_weights_scaled", "ffn_moe_up", "ffn_moe_gate",
               "ffn_moe_swiglu_limited", "ffn_moe_down", "ffn_moe_weighted", "ffn_moe_out", "ffn_up", "ffn_gate",
               "ffn_swiglu_limited", "ffn_shexp", "ffn_out", "l_last"]
        if il in self.engram:
            ops += ["engram_embd", "engram_kv", "engram_gate", "engram_out"]
        if self.ratios[il] == 0:
            ops += ["attn_raw"]
        else:
            ops += ["raw_k", "comp_k", "k_all", "kq_mask", "attn_out_raw"]
        if self.kv_source[il]:
            ops += ["comp_state_score", "comp_kv"]
            # without a pooling gate the score is the state KV itself, whose name the score's callback overwrites
            if self.ratios[il] != 1:
                ops += ["comp_state_kv"]
        if self.key_owner[il]:
            ops += ["idx_k_new"]
        if self.index_source[il]:
            ops += ["idx_q", "idx_q_rot", "idx_weights", "idx_k", "idx_score", "idx_top_k", "comp_top_k_mask"]
        return ops


MODEL_OPS = ["embd", "hc_init", "hc_pre_init", "hc_out", "result_norm", "result_output"]

# SPEC.md shapes, ggml order (ne0 first). T tokens of the ubatch, O outputs (1: the last token), r the layer's
# compress ratio, B its completed blocks (at least 1: a ubatch completing none computes a dummy block), C the
# compressed cells (n_visible padded to 256, at least 256), W the raw window cells (a multiple of 256, bound by the
# first file that has it).
SHAPES = {
    "embd": "E T", "hc_init": "E HC T", "hc_pre_init": "HC T",
    "engram_embd": "NC*KL T", "engram_kv": "E*(HC+1) T", "engram_gate": "1 HC T", "engram_out": "E HC T",
    "hc_mixes": "(2+HC)*HC T", "hc_pre": "HC T", "hc_post": "HC T", "hc_comb": "HC HC T",
    "hc_attn_pre": "E T", "hc_attn_post": "E HC T", "hc_ffn_pre": "E T", "l_last": "E HC T",
    "attn_norm": "E T", "ffn_norm": "E T", "qr": "QL T", "q": "D H T", "kv": "D 1 T",
    "comp_state_kv": "D T", "comp_state_score": "D T",
    "comp_kv.0": "D r B", "comp_kv.1": "D r B", "comp_kv.2": "D 1 B", "comp_kv.3": "D 1 B", "comp_kv.4": "D 1 B",
    "idx_k_new": "ID 1 B", "idx_q": "ID IH T", "idx_q_rot": "ID IH T", "idx_weights": "IH T", "idx_k": "ID 1 C",
    "idx_score": "C T", "idx_top_k": "min(C,TOPK) T", "comp_top_k_mask": "C T",
    "raw_k": "D 1 W", "comp_k": "D 1 C", "k_all": "D 1 W+C", "kq_mask": "W+C T",
    "kq": "N T H", "kq_soft_max": "N T H", "kqv": "D T H", "attn_raw": "D*H T", "attn_out_raw": "D*H T",
    "attn_derope": "D H T", "attn_wo_a": "OL T OG", "attn_out": "E T",
    "ffn_moe_logits": "NE T", "ffn_moe_probs": "NE T", "ffn_moe_probs_biased": "NE T", "ffn_moe_argsort": "NE T",
    "ffn_moe_topk": "K T", "ffn_moe_weights": "1 K T", "ffn_moe_weights_sum": "1 T",
    "ffn_moe_weights_sum_clamped": "1 T", "ffn_moe_weights_norm": "K T", "ffn_moe_weights_scaled": "1 K T",
    "ffn_moe_up": "FF K T", "ffn_moe_gate": "FF K T", "ffn_moe_swiglu_limited": "FF K T", "ffn_moe_down": "E K T",
    "ffn_moe_weighted": "E K T", "ffn_moe_out": "E T", "ffn_up": "FF T", "ffn_gate": "FF T",
    "ffn_swiglu_limited": "FF T", "ffn_shexp": "E T", "ffn_out": "E T",
    "hc_out": "E O", "result_norm": "E O", "result_output": "NV O",
}


def expected_ne(model, f, ub, binds):
    key = f"{f['op']}.{f['occurrence']}" if f"{f['op']}.{f['occurrence']}" in SHAPES else f["op"]
    if key not in SHAPES:
        return None
    il = f["layer"]
    r = model.ratios[il] if il >= 0 else 0
    pos = range(ub["pos0"], ub["pos0"] + ub["n_tokens"])
    env = dict(E=model.E, HC=HC, H=model.H, D=model.D, QL=model.QL, IH=model.IH, ID=model.ID, TOPK=model.top_k,
               OG=model.OG, OL=model.OL, NE=model.NE, K=model.K, FF=model.FF, NV=model.NV, NC=model.NC,
               KL=model.KL, T=ub["n_tokens"], O=1, r=r, min=min)
    if r:
        env["B"] = max(1, sum(1 for p in pos if (p + 1) % r == 0))
        n_visible = (pos[-1] + 1) // r
        env["C"] = max(256, -(-n_visible // 256) * 256)
    if "W" in binds:
        env["W"] = binds["W"]
    # the attention scores span the window, plus the compressed cells where the layer has them
    if "W" in env:
        env["N"] = env["W"] + (env["C"] if r else 0)
    dims = SHAPES[key].split()
    try:
        ne = [int(eval(d, {}, env)) for d in dims]  # noqa: S307 (formulas from this file only)
    except NameError:
        return None
    return ne + [1] * (4 - len(ne))


def cmd_check(d):
    m = manifest(d)
    model = Model(m["model_path"])
    ubs = {u["ubatch"]: u for u in m["ubatches"]}
    problems = []
    by_key = {}
    for f in m["files"]:
        path = os.path.join(d, f["file"])
        size = os.path.getsize(path) if os.path.exists(path) else -1
        want = int(np.prod(f["ne"])) * np.dtype(DTYPES[f["dtype"]]).itemsize
        if size != want:
            problems.append(f"size: {f['file']} is {size} B, its ne {f['ne']} {f['dtype']} make {want} B")
        by_key.setdefault((f["ubatch"], f["layer"]), []).append(f)
    # W: the raw window cells, from raw_k of the layer (or kq of a window only layer: its ne0)
    n_shape = 0
    n_skipped = 0
    for (u, il), files in sorted(by_key.items()):
        binds = {}
        for f in files:
            if f["op"] == "raw_k":
                binds["W"] = f["ne"][2]
            elif f["op"] == "kq" and il >= 0 and model.ratios[il] == 0:
                binds["W"] = f["ne"][0]
        if "W" in binds and binds["W"] % 256:
            problems.append(f"shape: ubatch {u} layer {il}: raw window of {binds['W']} cells is not a multiple of 256")
        seen = {f["op"] for f in files}
        wanted = MODEL_OPS if il < 0 else model.layer_ops(il)
        for op in wanted:
            if op not in seen:
                problems.append(f"missing: op {op} of layer {il} in ubatch {u}")
        for f in files:
            ne = expected_ne(model, f, ubs[u], binds)
            if ne is None:
                n_skipped += 1
                continue
            n_shape += 1
            if ne != f["ne"]:
                problems.append(f"shape: {f['file']} has ne {f['ne']}, SPEC.md says {ne}")
    n_files = len(m["files"])
    print(f"{d}: {n_files} files, {len(ubs)} ubatches, {len(m['files_per_op'])} ops, "
          f"{n_shape} shapes checked against SPEC.md ({n_skipped} without a formula), "
          f"size mismatches {sum(p.startswith('size') for p in problems)}, "
          f"shape mismatches {sum(p.startswith('shape') for p in problems)}, "
          f"missing ops {sum(p.startswith('missing') for p in problems)}")
    for p in problems[:40]:
        print("  " + p)
    return 1 if problems else 0


def rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)) or not np.array_equal(a[~finite], b[~finite]):
        return math.inf
    den = np.linalg.norm(a[finite])
    num = np.linalg.norm(a[finite] - b[finite])
    return float(num / den) if den else float(num)


def cmd_compare(da, db):
    ma, mb = manifest(da), manifest(db)
    rc = 0
    same_tokens = ma["generated_token_ids"] == mb["generated_token_ids"]
    print(f"generated tokens: {'identical' if same_tokens else 'DIFFER'} ({len(ma['generated_token_ids'])}): "
          f"{' '.join(map(str, ma['generated_token_ids']))}")
    print(f"logits fnv1a64: {ma['logits_fnv1a64_all']} vs {mb['logits_fnv1a64_all']}")
    rc |= not same_tokens
    fa = {f["file"]: f for f in ma["files"]}
    fb = {f["file"]: f for f in mb["files"]}
    if set(fa) != set(fb):
        print(f"files only in {da}: {len(set(fa) - set(fb))}, only in {db}: {len(set(fb) - set(fa))}")
        rc = 1
    per_op = {}
    for name in sorted(set(fa) & set(fb)):
        a_m, b_m = fa[name], fb[name]
        op = a_m["op"]
        st = per_op.setdefault(op, {"files": 0, "identical": 0, "worst": 0.0, "sets": 0})
        st["files"] += 1
        if a_m["ne"] != b_m["ne"] or a_m["dtype"] != b_m["dtype"]:
            st["worst"] = math.inf
            continue
        ba = open(os.path.join(da, name), "rb").read()
        bb = open(os.path.join(db, name), "rb").read()
        if ba == bb:
            st["identical"] += 1
            st["sets"] += 1
            continue
        a = np.frombuffer(ba, dtype=DTYPES[a_m["dtype"]]).reshape(list(reversed(a_m["ne"])))
        b = np.frombuffer(bb, dtype=DTYPES[b_m["dtype"]]).reshape(list(reversed(b_m["ne"])))
        if op in ("idx_top_k", "ffn_moe_topk", "ffn_moe_argsort"):
            # the selected index sets per token; ggml_top_k does not order them
            same = all(set(x) == set(y) for x, y in zip(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])))
            st["sets"] += same
            st["worst"] = max(st["worst"], 0.0 if same else math.inf)
        else:
            st["worst"] = max(st["worst"], rel_l2(a, b))
    n_files = sum(s["files"] for s in per_op.values())
    n_identical = sum(s["identical"] for s in per_op.values())
    print(f"{n_files} files compared, {n_identical} bit-identical, {len(per_op)} ops")
    for op, s in per_op.items():
        if s["identical"] != s["files"]:
            print(f"  {op:28s} {s['identical']}/{s['files']} identical, worst rel L2 {s['worst']:.3e}")
            rc = 1
    if n_identical == n_files and not rc:
        print("compare: all files bit-identical")
    return rc


def cmd_map_oracle(path):
    with open(path) as f:
        m = json.load(f)
    spec_ops = {k.split(".")[0] for k in SHAPES}
    seen = {}
    for f in m["files"]:
        base = f["source_tensor"].rsplit("-", 1)[0]
        seen.setdefault((f["kind"], base), set()).add(f["layer"])
    rc = 0
    print(f"{path}: {m['tool']} at fork {m['fork_commit']}, {len(m['files'])} files, {m['n_layer']} layers")
    for (kind, base), layers in sorted(seen.items()):
        ok = base in spec_ops
        rc |= not ok
        print(f"  {kind:8s} <- {base:24s} {'SPEC.md op ' + base if ok else 'NOT A SPEC OP'} "
              f"(layers {min(layers)}..{max(layers)}, {len(layers)})")
    return rc


def cmd_show(d, op, layer, ubatch, occ):
    a = load(d, op, int(layer), int(ubatch), int(occ))
    fin = a[np.isfinite(a)] if a.dtype.kind == "f" else a
    print(f"{op}.{occ} layer {layer} ubatch {ubatch}: shape {list(a.shape)} (ggml ne reversed) {a.dtype}, "
          f"finite {fin.size}/{a.size}, min {fin.min()}, max {fin.max()}, mean {float(np.mean(fin)):.6g}")
    print(a.reshape(-1)[:16])
    return 0


def cmd_roles(argv):
    n_tokens = None
    if len(argv) >= 2 and argv[0] == "--ctx":
        n_tokens, argv = int(argv[1]), argv[2:]
    if not argv:
        raise DumpError("roles: no dump directory or GGUF")
    what = argv[0]
    if len(argv) == 1 and os.path.isdir(argv[0]):
        m = manifest(argv[0])
        model = Model(m["model_path"])
        n_dump = max(u["pos0"] + u["n_tokens"] for u in m["ubatches"])
        n_tokens = n_tokens or n_dump
        what = f"{argv[0]} ({m['model_path']}, {n_dump} tokens dumped)"
    else:
        model = Model(*argv)
    print(f"{what}: {model.n_layer} layers, indexer top_k {model.top_k}")
    print("  layer ratio mode     kv_src key_src topk_src  reuse = dense up to")
    for il in range(model.n_layer):
        src = lambda v: str(v) if v >= 0 else "-"  # noqa: E731
        until = model.coincide_until(il)
        print(f"  {il:5d} {model.ratios[il]:5d} {model.mode(il):8s} {src(model.kv_src[il]):>6s} "
              f"{src(model.key_src[il]):>7s} {src(model.topk_src[il]):>8s}  "
              f"{'-' if until is None else f'{until} tokens'}")
    print(f"readers (Reuse): {model.readers()}")
    print(f"index sources (Full, Reindex): {[i for i in range(model.n_layer) if model.index_source[i]]}")
    if n_tokens:
        if model.modes_coincide(n_tokens):
            print(f"at {n_tokens} tokens: Reuse Mode and --readers dense coincide in every reader, "
                  f"a parity test may use either")
        else:
            differ = [i for i in model.readers() if model.coincide_until(i) < n_tokens]
            print(f"at {n_tokens} tokens: Reuse Mode differs from the fork in readers {differ}, "
                  f"parity tests must run --readers dense")
    return 0


def main(argv):
    if np is None and argv and argv[0] in ("check", "compare", "show"):
        print(f"op_dump.py: {argv[0]} needs numpy", file=sys.stderr)
        return 2
    try:
        if len(argv) == 2 and argv[0] == "check":
            return cmd_check(argv[1])
        if len(argv) == 3 and argv[0] == "compare":
            return cmd_compare(argv[1], argv[2])
        if len(argv) == 2 and argv[0] == "map-oracle":
            return cmd_map_oracle(argv[1])
        if len(argv) in (5, 6) and argv[0] == "show":
            return cmd_show(*argv[1:5], argv[5] if len(argv) == 6 else 0)
        if argv and argv[0] == "roles":
            return cmd_roles(argv[1:])
    except (DumpError, gh.GgufError, OSError, KeyError) as e:
        print(f"op_dump.py: {e}", file=sys.stderr)
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
