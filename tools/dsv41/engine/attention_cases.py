#!/usr/bin/env python3
"""Lists the attention cases of an op-dump directory for dsv41_engine_attention_test, python3 only.

  attention_cases.py DUMP_DIR [DUMP_DIR ...]   on stdout, per directory: a '# DIR: MODEL' header, then

    model PATH                                  the dump's GGUF (the test reads dims, roles and weights from it)
    ubatch U POS0 N_TOKENS                      every ubatch of the dump, in order
    file OP OCC LAYER U DTYPE NE0 NE1 NE2 NE3 PATH
                                                every dump file of the attention ops below (ggml ne, raw little endian)

Every op of SPEC.md section 5 that a layer's role computes (op_dump.py Model.layer_ops: the window, compressor, index
key and indexer ops, the masks and the attention and output ops) must be present in every ubatch with the dtype the
fork writes, or this exits 1 naming the file. The layer inputs (attn_norm) are listed too.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import op_dump  # noqa: E402

ATTN_OPS = {"attn_norm", "qr", "q", "kv", "kq", "kq_soft_max", "kqv", "attn_raw", "attn_out_raw", "attn_derope",
            "attn_wo_a", "attn_out", "raw_k", "comp_k", "k_all", "kq_mask", "comp_state_kv", "comp_state_score",
            "comp_kv", "idx_k_new", "idx_q", "idx_q_rot", "idx_weights", "idx_k", "idx_score", "idx_top_k",
            "comp_top_k_mask"}
F16_OPS = {"raw_k", "comp_k", "k_all", "idx_k"}
I32_OPS = {"idx_top_k"}


def fail(msg):
    print(f"attention_cases.py: {msg}", file=sys.stderr)
    sys.exit(1)


def cases(d):
    try:
        m = op_dump.manifest(d)
        model = op_dump.Model(m["model_path"])
    except (op_dump.DumpError, OSError, KeyError) as e:
        fail(str(e))
    files = {}
    for f in m["files"]:
        if f["op"] in ATTN_OPS:
            files[(f["op"], f["occurrence"], f["layer"], f["ubatch"])] = f
    out = [f"model {m['model_path']}"]
    for ub in m["ubatches"]:
        out.append(f"ubatch {ub['ubatch']} {ub['pos0']} {ub['n_tokens']}")
    for ub in m["ubatches"]:
        u = ub["ubatch"]
        for il in range(model.n_layer):
            wanted = [op for op in model.layer_ops(il) if op in ATTN_OPS]
            for op in wanted:
                occs = range(5) if op == "comp_kv" else [0]
                for occ in occs:
                    f = files.get((op, occ, il, u))
                    if f is None:
                        fail(f"{d}: no {op}.{occ} of layer {il} in ubatch {u}")
                    want = "f16" if op in F16_OPS else "i32" if op in I32_OPS else "f32"
                    if f["dtype"] != want:
                        fail(f"{d}/{f['file']}: dtype {f['dtype']}, expected {want}")
                    ne = " ".join(str(x) for x in f["ne"])
                    out.append(f"file {op} {occ} {il} {u} {f['dtype']} {ne} {os.path.join(os.path.abspath(d), f['file'])}")
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
