#!/usr/bin/env python3
"""Turns the fork's tiny random DeepSeek-V4 MoE (test-llama-archs -a deepseek4 -o DIR writes deepseek4-moe.gguf)
into a model the LLAMA_STRATA_MOE=1 hook runs, with a SwiGLU clamp limit of its own in every layer.

  tiny_moe_model.py IN.gguf OUT.gguf --clamp L0,L1,...

It writes IN with two changes:
  - <arch>.swiglu_clamp_exp becomes the f32 array of the given limits, one per layer (block_count values), so a
    limit read from the wrong layer or from freed memory shows;
  - the routed experts' scale tensors (blk.N.ffn_{up,gate,down}_exps.scale and .input_scale, which test-llama-archs
    creates for every MoE) are dropped: the hook computes unscaled experts only and aborts on them.
Every other key and tensor is copied unchanged. Used by tools/dsv41/oracle/hook-clamp-test.sh. Needs python3 only.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_header  # noqa: E402

SCALE = re.compile(r"blk\.\d+\.ffn_(up|gate|down)_exps\.(scale|input_scale)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--clamp", required=True, help="comma-separated SwiGLU clamp limits, one per layer, each > 1e-6")
    args = ap.parse_args()
    try:
        h = gguf_header.read_header(args.input)
        arch = h.metadata["general.architecture"][1]
        n_layer = gguf_header.integer_value(args.input, h, f"{arch}.block_count")
        limits = [float(x) for x in args.clamp.split(",")]
        if len(limits) != n_layer or not all(x > 1e-6 for x in limits):
            raise gguf_header.GgufError(f"--clamp {args.clamp}: expected {n_layer} limits > 1e-6 ({arch}.block_count)")
        key = f"{arch}.swiglu_clamp_exp"
        if key not in h.metadata:
            raise gguf_header.GgufError(f"{args.input}: no {key}, {arch} has no SwiGLU clamp")
        h.metadata[key] = (gguf_header.ARRAY, limits, gguf_header.FLOAT32)
        dropped = [t["name"] for t in h.tensors if SCALE.fullmatch(t["name"])]
        source = {t["name"]: t for t in h.tensors}
        h.tensors = [t for t in h.tensors if not SCALE.fullmatch(t["name"])]
        with open(args.input, "rb") as f:
            def data(name):
                t = source[name]
                f.seek(h.data_start + t["offset"])
                return f.read(gguf_header.tensor_nbytes(t["type"], t["ne"]))

            gguf_header.write_gguf(args.output, h, data)
    except (gguf_header.GgufError, OSError, ValueError, KeyError) as e:
        print(f"tiny_moe_model.py: {e}", file=sys.stderr)
        return 1
    print(f"{args.output}: {arch}, {n_layer} layers, {key} = {limits}, {len(h.tensors)} tensors "
          f"({len(dropped)} routed expert scale tensors dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
