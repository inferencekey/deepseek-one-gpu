#!/usr/bin/env python3
"""Lists the dense cases of an op-dump directory for dsv41_engine_dense_test, python3 only.

  dense_cases.py DUMP_DIR [DUMP_DIR ...]   on stdout, per directory: a '# DIR: MODEL' header, then

    model PATH                                  the dump's GGUF (the test reads dims and weights from it)
    ubatch U POS0 N_TOKENS                      every ubatch of the dump, in order
    tokens U ID ...                             the tokens of ubatch U (the prompt, then the generated tokens)
    file OP OCC LAYER U DTYPE NE0 NE1 NE2 NE3 PATH
                                                every dump file of the ops below (ggml ne, raw little endian);
                                                LAYER -1 for the model level ops

Every op of SPEC.md sections 1, 2, 3, 6, 7 (shared expert) and 8 that a layer computes (op_dump.py Model.layer_ops),
the model level ops (MODEL_OPS), and the inputs the test feeds (attn_out, ffn_moe_out) must be present in every
ubatch with the dtype the fork writes, or this exits 1 naming the file.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import op_dump  # noqa: E402

LAYER_OPS = {"hc_mixes", "hc_pre", "hc_post", "hc_comb", "hc_attn_pre", "attn_norm", "attn_out", "hc_attn_post",
             "hc_ffn_pre", "ffn_norm", "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_probs_biased", "ffn_moe_argsort",
             "ffn_moe_topk", "ffn_moe_weights", "ffn_moe_weights_sum", "ffn_moe_weights_sum_clamped",
             "ffn_moe_weights_norm", "ffn_moe_weights_scaled", "ffn_moe_out", "ffn_up", "ffn_gate",
             "ffn_swiglu_limited", "ffn_shexp", "ffn_out", "l_last", "engram_embd", "engram_kv", "engram_gate",
             "engram_out"}
MODEL_OPS = set(op_dump.MODEL_OPS)
TWO_OCC = {"hc_mixes", "hc_pre", "hc_post", "hc_comb"}   # .0 the attention sublayer's, .1 the FFN's
I32_OPS = {"ffn_moe_argsort", "ffn_moe_topk"}


def fail(msg):
    print(f"dense_cases.py: {msg}", file=sys.stderr)
    sys.exit(1)


def cases(d):
    try:
        m = op_dump.manifest(d)
        model = op_dump.Model(m["model_path"])
    except (op_dump.DumpError, OSError, KeyError) as e:
        fail(str(e))
    files = {}
    for f in m["files"]:
        files[(f["op"], f["occurrence"], f["layer"], f["ubatch"])] = f
    seq = list(m["prompt_token_ids"]) + list(m["generated_token_ids"])
    out = [f"model {m['model_path']}"]
    for ub in m["ubatches"]:
        out.append(f"ubatch {ub['ubatch']} {ub['pos0']} {ub['n_tokens']}")
    for ub in m["ubatches"]:
        p0, n = ub["pos0"], ub["n_tokens"]
        if p0 + n > len(seq):
            fail(f"{d}: ubatch {ub['ubatch']} ends at {p0 + n}, past the {len(seq)} known tokens")
        out.append(f"tokens {ub['ubatch']} " + " ".join(str(t) for t in seq[p0:p0 + n]))

    def want(op, occ, il, u):
        f = files.get((op, occ, il, u))
        if f is None:
            fail(f"{d}: no {op}.{occ} of layer {il} in ubatch {u}")
        dt = "i32" if op in I32_OPS else "f32"
        if f["dtype"] != dt:
            fail(f"{d}/{f['file']}: dtype {f['dtype']}, expected {dt}")
        ne = " ".join(str(x) for x in f["ne"])
        out.append(f"file {op} {occ} {il} {u} {f['dtype']} {ne} {os.path.join(os.path.abspath(d), f['file'])}")

    for ub in m["ubatches"]:
        u = ub["ubatch"]
        for op in sorted(MODEL_OPS):
            want(op, 0, -1, u)
        for il in range(model.n_layer):
            for op in [op for op in model.layer_ops(il) if op in LAYER_OPS]:
                for occ in (0, 1) if op in TWO_OCC else (0,):
                    want(op, occ, il, u)
    return m["model_path"], out


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(argv) >= 2 else 2
    for d in argv[1:]:
        model, lines = cases(d)
        n_files = sum(1 for x in lines if x.startswith("file "))
        print(f"# {d}: {model}, {n_files} files")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
