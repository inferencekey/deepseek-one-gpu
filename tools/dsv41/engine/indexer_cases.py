#!/usr/bin/env python3
"""Lists the lightning indexer cases of an op-dump directory for dsv41_engine_indexer_test, python3 only.

  indexer_cases.py DUMP_DIR [DUMP_DIR ...]   one line per (index source layer, ubatch) on stdout

Each line: `case LAYER UBATCH POS0 N_TOKENS RATIO N_HEADS DIM TOP_K N_COMP Q K W SCORE TOPK`, the five paths being the
dump files of idx_q_rot [T, IH, ID] f32, idx_k [C, 1, ID] f16, idx_weights [T, IH] f32, idx_score [T, C] f32 and
idx_top_k [T, k] i32 (ggml ne reversed, raw little endian). Ratio, heads, dim and top-k come from the dump's own GGUF
(the manifest's model_path, read with tools/dsv41/oracle/gguf_header.py), positions from the manifest's ubatches.
Every index source layer of the GGUF (it carries indexer.attn_q_b) must have all five files in every ubatch, with the
shapes the GGUF implies, or this exits 1 naming the file.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
import gguf_header as gh  # noqa: E402

OPS = ("idx_q_rot", "idx_k", "idx_weights", "idx_score", "idx_top_k")
DTYPES = {"idx_q_rot": "f32", "idx_k": "f16", "idx_weights": "f32", "idx_score": "f32", "idx_top_k": "i32"}


def fail(msg):
    print(f"indexer_cases.py: {msg}", file=sys.stderr)
    sys.exit(1)


def cases(d):
    path = os.path.join(d, "dump-manifest.json")
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        fail(f"{path}: {e}")
    h = gh.read_header(m["model_path"])
    md = {k: v[1] for k, v in h.metadata.items()}
    arch = md.get("general.architecture")
    if arch != "deepseek41":
        fail(f"{m['model_path']}: architecture {arch}, expected deepseek41")
    g = lambda k: md[f"{arch}.{k}"]  # noqa: E731
    n_layer = g("block_count")
    ratios = g("attention.compress_ratios")[:n_layer]
    ih, idim, top_k = g("attention.indexer.head_count"), g("attention.indexer.key_length"), g("attention.indexer.top_k")
    names = {t["name"] for t in h.tensors}
    sources = [i for i in range(n_layer) if f"blk.{i}.indexer.attn_q_b.weight" in names]
    if not sources:
        fail(f"{m['model_path']}: no index source layer (indexer.attn_q_b)")
    files = {(f["op"], f["layer"], f["ubatch"], f["occurrence"]): f for f in m["files"]}
    out = []
    for ub in m["ubatches"]:
        u, pos0, nt = ub["ubatch"], ub["pos0"], ub["n_tokens"]
        for il in sources:
            got = {op: files.get((op, il, u, 0)) for op in OPS}
            for op, f in got.items():
                if f is None:
                    fail(f"{d}: no {op} of layer {il} in ubatch {u}")
            n_comp = got["idx_k"]["ne"][2]
            row = []
            for op, f in got.items():
                if f["dtype"] != DTYPES[op]:
                    fail(f"{d}/{f['file']}: dtype {f['dtype']}, expected {DTYPES[op]}")
                ne = f["ne"]
                want = {"idx_q_rot": [idim, ih, nt, 1], "idx_k": [idim, 1, n_comp, 1], "idx_weights": [ih, nt, 1, 1],
                        "idx_score": [n_comp, nt, 1, 1], "idx_top_k": [min(n_comp, top_k), nt, 1, 1]}[op]
                if ne != want:
                    fail(f"{d}/{f['file']}: ne {ne}, expected {want}")
                row.append(os.path.join(os.path.abspath(d), f["file"]))
            out.append(f"case {il} {u} {pos0} {nt} {ratios[il]} {ih} {idim} {top_k} {n_comp} " + " ".join(row))
    return m["model_path"], out


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(argv) >= 2 else 2
    for d in argv[1:]:
        model, lines = cases(d)
        print(f"# {d}: {model}, {len(lines)} cases")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
