#!/usr/bin/env python3
"""Writes a tiny random DeepSeek-V4.1 GGUF that the stock fork ($FORK_DIR, the patched llama.cpp fork with the hook off) loads and runs on the CPU.

  tiny_dsv41.py --config a|b OUT.gguf [--types real|f32] [--seed N]

Same architecture (general.architecture deepseek41), metadata keys and tensor names as the real
vcruz305/DeepSeek-V4.1-Flash-GGUF Q2_K model (tools/dsv41/engine/SPEC.md, "Tensors"), with small dims. Every layer
role of the real model is present in both configs (the layer roles follow from which tensors a layer carries, as the
fork's llama_model_deepseek41::load_arch_tensors works them out):

  config a: 4 layers, n_embd 256, 4 heads of 256, 8 experts (2 used), n_ff_exp 256
            ratios [0, 2, 2, 1]: L0 window only + engram, L1 ratio 2 source with a pooling gate, index key owner and
            index source + engram, L2 ratio 2 reader, L3 ratio 1 source without a gate, key owner, index source
  config b: 6 layers, n_embd 512, 8 heads of 512, 32 experts (6 used), n_ff_exp 512
            ratios [0, 0, 2, 2, 1, 1]: L0 window only, L1 window only + engram, L2 ratio 2 source (gate, key owner,
            index source) + engram, L3 ratio 2 reader, L4 ratio 1 source (no gate, key owner, index source), L5 ratio 1
            index source reading L4's keys (the real layers 24, 28, 32, 36)

Both use a 512 token no_vocab vocabulary (token ids are fed directly), context_length 2048, sliding window 16 and
indexer top-k 8 or 16, so a 64 + 8 token run slides the window and the indexer really selects. The scale invariant
constants (rope and YaRN, epsilons, Sinkhorn iterations, SwiGLU clamp 10, routing scale 1.5, sqrtsoftplus gating,
hc_mult 4, engram n-gram 4 and key length 256) are the real model's.

--types real gives every tensor the real model's type (q2_K for the matrices, q3_K for the down projections, q6_K
for output, bf16 for the router, f32 for norms, biases and hyper-connection scales); --types f32 makes all f32.
Quantized tensors are random blocks, not quantized floats: random scale and quant bytes with the block's f16
super-scale chosen so the dequantized weights have about 1/sqrt(fan_in) standard deviation, which exercises the
fork's real dequantization. Needs python3 and numpy.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
import gguf_header as gh  # noqa: E402

U32, I32, F32, BOOL, STR, U64 = 4, 5, 6, 7, 8, 10
T_F32, T_Q2_K, T_Q3_K, T_Q6_K, T_BF16 = 0, 10, 11, 14, 30
TYPE_IDS = {"f32": T_F32, "q2_K": T_Q2_K, "q3_K": T_Q3_K, "q6_K": T_Q6_K, "bf16": T_BF16}

CONFIGS = {
    "a": dict(n_layer=4, n_embd=256, n_head=4, head_dim=256, q_lora=256, o_groups=4, o_lora=64,
              n_expert=8, n_expert_used=2, n_ff_exp=256, idx_heads=4, idx_dim=128, idx_top_k=8,
              ratios=[0, 2, 2, 1], engram_layers=[0, 1], engram_heads=2,
              # per layer: (compressor, pooling gate, index key owner, index source)
              roles={1: (True, True, True, True), 3: (True, False, True, True)}),
    "b": dict(n_layer=6, n_embd=512, n_head=8, head_dim=512, q_lora=512, o_groups=8, o_lora=64,
              n_expert=32, n_expert_used=6, n_ff_exp=512, idx_heads=8, idx_dim=128, idx_top_k=16,
              ratios=[0, 0, 2, 2, 1, 1], engram_layers=[1, 2], engram_heads=2,
              roles={2: (True, True, True, True), 4: (True, False, True, True), 5: (False, False, False, True)}),
}
N_VOCAB = 512
N_CTX = 2048
N_SWA = 16
N_ROT = 64
HC = 4
NGRAM = 4
ENGRAM_KEY = 256
ENGRAM_PAD = 2
# the real tokenizer folds 129280 ids into 99092 compressed ids; here 512 ids fold into 384
ENGRAM_IDS = 384


def f16_bytes(x):
    return np.asarray(x, dtype=np.float16).tobytes()


def q2_k(rng, n_rows, ne0, std):
    # block_q2_K: scales[16] (low nibble scale, high nibble min), qs[64], d, dmin. y = d*sc*q - dmin*m with q in 0..3:
    # E[sc*q] = 11.25 = 1.5 E[m], so dmin = 1.5 d centres it; its spread is then about 13.9 d
    nb = n_rows * (ne0 // 256)
    d = std / 13.9
    blocks = np.empty((nb, 84), dtype=np.uint8)
    blocks[:, :80] = rng.integers(0, 256, size=(nb, 80), dtype=np.uint8)
    blocks[:, 80:82] = np.frombuffer(f16_bytes(d), dtype=np.uint8)
    blocks[:, 82:84] = np.frombuffer(f16_bytes(1.5 * d), dtype=np.uint8)
    return blocks.tobytes()


def q3_k(rng, n_rows, ne0, std):
    # block_q3_K: hmask[32], qs[64], scales[12] (6 bit, -32..31), d. q in -4..3 and the scale are both about zero
    # mean, spread about 2.29 x 18.5 = 42.4 d
    nb = n_rows * (ne0 // 256)
    blocks = np.empty((nb, 110), dtype=np.uint8)
    blocks[:, :108] = rng.integers(0, 256, size=(nb, 108), dtype=np.uint8)
    blocks[:, 108:110] = np.frombuffer(f16_bytes(std / 42.4), dtype=np.uint8)
    return blocks.tobytes()


def q6_k(rng, n_rows, ne0, std):
    # block_q6_K: ql[128], qh[64], scales[16] (int8), d. q - 32 in -32..31 (spread 18.5) times an int8 scale
    # (spread 74): about 1368 d
    nb = n_rows * (ne0 // 256)
    blocks = np.empty((nb, 210), dtype=np.uint8)
    blocks[:, :208] = rng.integers(0, 256, size=(nb, 208), dtype=np.uint8)
    blocks[:, 208:210] = np.frombuffer(f16_bytes(std / 1368.0), dtype=np.uint8)
    return blocks.tobytes()


def tensor_bytes(rng, ttype, ne, values):
    """values(n) -> float32 array for f32/bf16; quantized types use the std of values' kind (std attribute)."""
    n_rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
    if ttype == T_F32:
        return values(int(np.prod(ne))).astype(np.float32).tobytes()
    if ttype == T_BF16:
        x = values(int(np.prod(ne))).astype(np.float32)
        return (x.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    if ne[0] % 256:
        raise gh.GgufError(f"row of {ne[0]} elements is not whole 256 element K-quant blocks")
    std = values.std
    return {T_Q2_K: q2_k, T_Q3_K: q3_k, T_Q6_K: q6_k}[ttype](rng, n_rows, ne[0], std)


class Values:
    """A random value source with a known spread, for f32 data and the quantized blocks' scale."""

    def __init__(self, rng, std, mean=0.0, positive=False):
        self.rng, self.std, self.mean, self.positive = rng, std, mean, positive

    def __call__(self, n):
        x = self.mean + self.std * self.rng.standard_normal(n)
        return np.abs(x) if self.positive else x


def primes_from(start, count):
    out, p = [], start
    while len(out) < count:
        if p > 1 and all(p % q for q in range(2, int(p ** 0.5) + 1)):
            out.append(p)
        p += 1
    return out


def build(cfg, types, seed):
    rng = np.random.default_rng(seed)
    L, E, hd = cfg["n_layer"], cfg["n_embd"], cfg["head_dim"]
    H, ff, ne = cfg["n_head"], cfg["n_ff_exp"], cfg["n_expert"]
    hc_dim, hc_mix = HC * E, (2 + HC) * HC
    n_cols = (NGRAM - 1) * cfg["engram_heads"]
    n_eg = len(cfg["engram_layers"])

    # engram hash tables: per layer and bucket a prime, offsets cumulative within a layer, as in the real file
    primes = []
    offsets = []
    for e in range(n_eg):
        ps = primes_from(97 + 61 * e, n_cols)
        acc = 0
        for p in ps:
            offsets.append(acc)
            acc += p
        primes += ps
    multipliers = [int(x) | 1 for x in rng.integers(1, 2 ** 37, size=n_eg * NGRAM, dtype=np.int64)]
    token_map = [int(x) for x in rng.integers(0, ENGRAM_IDS, size=N_VOCAB)]

    arch = "deepseek41"
    k = lambda s: f"{arch}.{s}"  # noqa: E731
    md = {
        "general.architecture": (STR, arch, None),
        "general.type": (STR, "model", None),
        "general.name": (STR, f"tiny random DeepSeek V4.1 ({cfg['name']}, seed {seed}, types {types})", None),
        k("block_count"): (U32, L, None),
        k("context_length"): (U32, N_CTX, None),
        k("embedding_length"): (U32, E, None),
        k("vocab_size"): (U32, N_VOCAB, None),
        k("attention.head_count"): (U32, H, None),
        k("attention.head_count_kv"): (U32, 1, None),
        k("rope.scaling.type"): (STR, "yarn", None),
        k("rope.scaling.factor"): (F32, 16.0, None),
        k("rope.scaling.original_context_length"): (U32, 65536, None),
        k("rope.scaling.yarn_beta_fast"): (F32, 32.0, None),
        k("rope.scaling.yarn_beta_slow"): (F32, 1.0, None),
        k("rope.freq_base"): (F32, 10000.0, None),
        k("attention.layer_norm_rms_epsilon"): (F32, 1e-20, None),
        k("expert_count"): (U32, ne, None),
        k("expert_used_count"): (U32, cfg["n_expert_used"], None),
        k("expert_gating_func"): (U32, 4, None),
        k("attention.key_length"): (U32, hd, None),
        k("attention.value_length"): (U32, hd, None),
        k("rope.dimension_count"): (U32, N_ROT, None),
        k("attention.q_lora_rank"): (U32, cfg["q_lora"], None),
        k("attention.sliding_window"): (U32, N_SWA, None),
        k("expert_feed_forward_length"): (U32, ff, None),
        k("expert_shared_count"): (U32, 1, None),
        k("expert_weights_scale"): (F32, 1.5, None),
        k("expert_weights_norm"): (BOOL, True, None),
        k("swiglu_clamp_exp"): (gh.ARRAY, [10.0] * L, F32),
        k("swiglu_clamp_shexp"): (gh.ARRAY, [10.0] * L, F32),
        k("attention.indexer.head_count"): (U32, cfg["idx_heads"], None),
        k("attention.indexer.key_length"): (U32, cfg["idx_dim"], None),
        k("attention.indexer.top_k"): (U32, cfg["idx_top_k"], None),
        k("attention.output_group_count"): (U32, cfg["o_groups"], None),
        k("attention.output_lora_rank"): (U32, cfg["o_lora"], None),
        k("attention.compress_ratios"): (gh.ARRAY, cfg["ratios"], I32),
        k("attention.compress_rope_freq_base"): (F32, 160000.0, None),
        k("hyper_connection.count"): (U32, HC, None),
        k("hyper_connection.sinkhorn_iterations"): (U32, 20, None),
        k("hyper_connection.epsilon"): (F32, 1e-6, None),
        k("hash_layer_count"): (U32, 0, None),
        k("engram.head_count"): (U32, cfg["engram_heads"], None),
        k("engram.key_length"): (U32, ENGRAM_KEY, None),
        k("engram.max_ngram_size"): (U32, NGRAM, None),
        k("engram.layer_ids"): (gh.ARRAY, cfg["engram_layers"], I32),
        k("engram.multipliers"): (gh.ARRAY, multipliers, U64),
        k("engram.primes"): (gh.ARRAY, primes, U64),
        k("engram.offsets"): (gh.ARRAY, offsets, U64),
        k("engram.token_map"): (gh.ARRAY, token_map, I32),
        k("engram.pad_id"): (U32, ENGRAM_PAD, None),
        "general.quantization_version": (U32, 2, None),
        "tokenizer.ggml.model": (STR, "no_vocab", None),
    }

    tensors = []  # (name, ne, real type, Values)

    def add(name, ne_, real_type, values):
        tensors.append((name, list(ne_), real_type, values))

    def lin(fan_in):
        return Values(rng, 1.0 / np.sqrt(fan_in))

    norm = Values(rng, 0.1, mean=1.0)
    add("token_embd.weight", [E, N_VOCAB], "q2_K", Values(rng, 1.0))
    add("output_norm.weight", [E], "f32", norm)
    add("output.weight", [E, N_VOCAB], "q6_K", lin(E))
    for i in range(L):
        b = f"blk.{i}."
        add(b + "attn_norm.weight", [E], "f32", norm)
        add(b + "attn_sinks.weight", [H], "f32", Values(rng, 1.0))
        add(b + "attn_q_a.weight", [E, cfg["q_lora"]], "q2_K", lin(E))
        add(b + "attn_q_a_norm.weight", [cfg["q_lora"]], "f32", norm)
        add(b + "attn_q_b.weight", [cfg["q_lora"], H * hd], "q2_K", lin(cfg["q_lora"]))
        add(b + "attn_kv.weight", [E, hd], "q2_K", lin(E))
        add(b + "attn_kv_a_norm.weight", [hd], "f32", norm)
        # the file keeps wo_a 2D, (n_head * head_dim / o_groups, o_lora_rank * o_groups); the fork reshapes at load
        add(b + "attn_output_a.weight", [H * hd // cfg["o_groups"], cfg["o_lora"] * cfg["o_groups"]], "q2_K",
            lin(H * hd // cfg["o_groups"]))
        add(b + "attn_output_b.weight", [cfg["o_groups"] * cfg["o_lora"], E], "q2_K",
            lin(cfg["o_groups"] * cfg["o_lora"]))
        for sub in ("attn", "ffn"):
            add(b + f"hc_{sub}_fn.weight", [hc_dim, hc_mix], "q2_K", lin(hc_dim))
            add(b + f"hc_{sub}_base.weight", [hc_mix], "f32", Values(rng, 0.5))
            add(b + f"hc_{sub}_scale.weight", [3], "f32", Values(rng, 0.5, mean=1.0, positive=True))
        comp, gate, key_owner, index_source = cfg["roles"].get(i, (False, False, False, False))
        if comp:
            add(b + "attn_compressor_kv.weight", [E, hd], "q2_K", lin(E))
            add(b + "attn_compressor_norm.weight", [hd], "f32", norm)
        if gate:
            add(b + "attn_compressor_gate.weight", [E, hd], "q2_K", lin(E))
        if key_owner:
            add(b + "indexer.attn_k.weight", [hd, cfg["idx_dim"]], "q2_K", lin(hd))
            add(b + "indexer.k_norm.weight", [cfg["idx_dim"]], "f32", norm)
        if index_source:
            add(b + "indexer.attn_q_b.weight", [cfg["q_lora"], cfg["idx_heads"] * cfg["idx_dim"]], "q2_K",
                lin(cfg["q_lora"]))
            add(b + "indexer.proj.weight", [E, cfg["idx_heads"]], "q2_K", lin(E))
        add(b + "ffn_norm.weight", [E], "f32", norm)
        add(b + "ffn_gate_inp.weight", [E, ne], "bf16", lin(E))
        add(b + "exp_probs_b.bias", [ne], "f32", Values(rng, 0.05))
        add(b + "exp_probs_b_vl.bias", [ne], "f32", Values(rng, 0.05))
        add(b + "ffn_gate_exps.weight", [E, ff, ne], "q2_K", lin(E))
        add(b + "ffn_up_exps.weight", [E, ff, ne], "q2_K", lin(E))
        add(b + "ffn_down_exps.weight", [ff, E, ne], "q3_K", lin(ff))
        add(b + "ffn_gate_shexp.weight", [E, ff], "q2_K", lin(E))
        add(b + "ffn_up_shexp.weight", [E, ff], "q2_K", lin(E))
        add(b + "ffn_down_shexp.weight", [ff, E], "q3_K", lin(ff))
        if i in cfg["engram_layers"]:
            e = cfg["engram_layers"].index(i)
            n_rows = max(offsets[e * n_cols + c] + primes[e * n_cols + c] for c in range(n_cols))
            add(b + "engram_embd.weight", [ENGRAM_KEY, n_rows], "q2_K", Values(rng, 1.0))
            add(b + "engram_wkv.weight", [n_cols * ENGRAM_KEY, E * (HC + 1)], "q2_K", lin(n_cols * ENGRAM_KEY))
            add(b + "engram_q.weight", [E, HC], "q2_K", Values(rng, 0.1, mean=1.0))
            add(b + "engram_k.weight", [E, HC], "q2_K", Values(rng, 0.1, mean=1.0))

    header = gh.Header(3, md, [], gh.DEFAULT_ALIGNMENT, 0)
    blobs = {}
    for name, ne_, real_type, values in tensors:
        ttype = T_F32 if types == "f32" else TYPE_IDS[real_type]
        header.tensors.append({"name": name, "ne": ne_, "type": ttype, "offset": 0})
        blobs[name] = tensor_bytes(rng, ttype, ne_, values)
    md["general.file_type"] = (U32, 0 if types == "f32" else 10, None)
    return header, blobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--types", default="real", choices=["real", "f32"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("output")
    args = ap.parse_args()
    cfg = dict(CONFIGS[args.config], name=args.config)
    try:
        header, blobs = build(cfg, args.types, args.seed)
        gh.write_gguf(args.output, header, lambda name: blobs[name])
    except (gh.GgufError, OSError) as e:
        print(f"tiny_dsv41.py: {e}", file=sys.stderr)
        return 1
    n_bytes = sum(len(b) for b in blobs.values())
    print(f"{args.output}: deepseek41 config {args.config}, {cfg['n_layer']} layers, n_embd {cfg['n_embd']}, "
          f"{cfg['n_expert']} experts ({cfg['n_expert_used']} used), types {args.types}, seed {args.seed}, "
          f"{len(header.tensors)} tensors, {n_bytes} B of tensor data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
