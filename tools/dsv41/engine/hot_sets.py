#!/usr/bin/env python3
"""Writes the expert placements of the engine's end-to-end runs as strata-moe hot-set files, python3 only.

  hot_sets.py MODEL.gguf OUT_DIR

writes OUT_DIR/all-vram.json (every routed expert of every layer in VRAM) and OUT_DIR/split.json (per layer the even
experts in VRAM, the odd ones in the warm list except every fourth, which stays cold), in the format
src/dsv41/moe_hot_set.h reads (strata-moe-hot-set/1, the one tools/dsv41/oracle/hot_set.py writes from activation
counts). The third placement, all on the CPU, is no hot set at all. The types and the dims come from the GGUF's
tensor headers (blk.L.ffn_{up,gate,down}_exps.weight); budget_gib and ram_budget_gib are the files' own sizes rounded
up to whole GiB, which dsv41-engine's --vram-gb and --ram-gb must cover.
"""

import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
from gguf_header import GGML_TYPES, GgufError, read_header, tensor_nbytes  # noqa: E402

FORMAT = "strata-moe-hot-set/1"
KINDS = ("up", "gate", "down")


def fail(msg):
    print(f"hot_sets.py: {msg}", file=sys.stderr)
    sys.exit(1)


def layers_of(path):
    try:
        h = read_header(path)
    except (GgufError, OSError) as e:
        fail(str(e))
    out = {}
    for t in h.tensors:
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight", t["name"])
        if m:
            out.setdefault(int(m.group(1)), {})[m.group(2)] = t
    if not out:
        fail(f"{path}: no routed expert tensors")
    return out


def main(argv):
    if len(argv) != 3 or argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(argv) >= 2 and argv[1] in ("-h", "--help") else 2
    model, out_dir = argv[1], argv[2]
    layers = layers_of(model)
    os.makedirs(out_dir, exist_ok=True)
    placements = {
        "all-vram": (lambda e: True, lambda e: False),
        "split": (lambda e: e % 2 == 0, lambda e: e % 2 == 1 and e % 4 != 3),
    }
    for name, (vram, warm) in placements.items():
        doc_layers, vram_bytes, warm_bytes = [], 0, 0
        for il in sorted(layers):
            t = layers[il]
            if set(t) != set(KINDS):
                fail(f"{model}: layer {il} has the expert parts {sorted(t)}")
            n_embd, n_ff, n_expert = t["up"]["ne"][0], t["up"]["ne"][1], t["up"]["ne"][2]
            per = [tensor_nbytes(t[k]["type"], t[k]["ne"]) // n_expert for k in KINDS]
            hot = [e for e in range(n_expert) if vram(e)]
            cool = [e for e in range(n_expert) if not vram(e) and warm(e)]
            vram_bytes += len(hot) * sum(per)
            # a warm slot per part: whole 4096-byte blocks plus one (src/dsv41/moe_residency.cpp direct_slot_bytes)
            warm_bytes += len(cool) * sum((p + 4095) // 4096 * 4096 + 4096 for p in per)
            doc_layers.append({"layer": il, "n_expert": n_expert, "n_embd": n_embd, "n_ff": n_ff,
                               "types": {k: GGML_TYPES[t[k]["type"]][0] for k in KINDS}, "experts": hot,
                               "warm": cool})
        doc = {"format": FORMAT, "budget_gib": max(1, math.ceil(vram_bytes / (1 << 30))),
               "ram_budget_gib": max(1, math.ceil(warm_bytes / (1 << 30))), "layers": doc_layers}
        path = os.path.join(out_dir, name + ".json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        print(f"{path}: {sum(len(x['experts']) for x in doc_layers)} VRAM experts ({vram_bytes / 2**20:.1f} MiB), "
              f"{sum(len(x['warm']) for x in doc_layers)} warm ({warm_bytes / 2**20:.1f} MiB), "
              f"budgets {doc['budget_gib']} / {doc['ram_budget_gib']} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
