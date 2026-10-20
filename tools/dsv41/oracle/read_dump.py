#!/usr/bin/env python3
"""Reads the dumps of dsv41-moe-dump (tools/dsv41/oracle/README.md).

  read_dump.py DIR                    checks every file against the manifest, prints one row per MoE layer
  read_dump.py --compare DIR_A DIR_B  per-layer relative L2 of B against A and topk match counts; with
                                      --tokens-must-match it exits 1 unless the generated tokens are identical
  read_dump.py --expert-placement DIR where the run put the routed experts, from its command line (manifest argv):
                                      "gpu_layers A-B" (the layers whose experts went to a CUDA device) and one
                                      "override PATTERN=BUFFER" line per --override-tensor value that places them

Exits 1 when a file is missing, its byte size does not match its manifest shape, or a shape is inconsistent, and
--expert-placement when the command line leaves an expert tensor unplaced or the CUDA layers are not one range.
Needs python3 and numpy only.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

KINDS = ("cur", "topk", "weights", "out")
DTYPES = {"f32": np.dtype("<f4"), "f16": np.dtype("<f2"), "i32": np.dtype("<i4")}


class DumpError(Exception):
    pass


def fmt_ne(ne):
    # trailing 1s dropped, as ggml prints shapes: [n_embd, n_tokens]
    dims = list(ne)
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return "[" + ", ".join(str(d) for d in dims) + "]"


class Dump:
    def __init__(self, path):
        self.path = path
        mpath = os.path.join(path, "dump-manifest.json")
        if not os.path.isfile(mpath):
            raise DumpError(f"{path}: no dump-manifest.json")
        with open(mpath) as f:
            self.manifest = json.load(f)
        self.n_embd = self.manifest["n_embd"]
        self.ubatches = self.manifest["ubatches"]
        self.weights_tensor = {e["layer"]: e["weights_tensor"] for e in self.manifest["moe_layers"]}
        # (kind, layer, ubatch) -> manifest entry
        self.entries = {}
        for e in self.manifest["files"]:
            key = (e["kind"], e["layer"], e["ubatch"])
            if key in self.entries:
                raise DumpError(f"{path}: {e['file']} duplicates {self.entries[key]['file']}")
            self.entries[key] = e
        self.layers = sorted({e["layer"] for e in self.manifest["files"]})

    def size_mismatches(self):
        """(file, expected bytes, actual bytes or None) for every file whose size does not match its shape."""
        bad = []
        for e in self.manifest["files"]:
            dtype = DTYPES.get(e["dtype"])
            if dtype is None:
                raise DumpError(f"{e['file']}: unsupported dtype {e['dtype']!r}")
            expected = int(np.prod(e["ne"])) * dtype.itemsize
            fpath = os.path.join(self.path, e["file"])
            actual = os.path.getsize(fpath) if os.path.isfile(fpath) else None
            if actual != expected:
                bad.append((e["file"], expected, actual))
        return bad

    def n_tokens(self, layer, ubatch):
        """Token rows of one layer in one ubatch: the ubatch's n_tokens, or fewer in a layer the graph computes
        for the output rows only (llama.cpp's inp_out_ids, e.g. OLMoE's last layer)."""
        return self.entries[("cur", layer, ubatch)]["ne"][1]

    def shape_problems(self):
        """Every shape that breaks cur [n_embd, n_tok], topk [n_used, n_tok], weights [1, n_used, n_tok],
        out [n_embd, n_tok], with n_tok the layer's token rows (1 to the ubatch's n_tokens) and n_used the same
        in topk and weights."""
        problems = []
        for layer in self.layers:
            for ub in self.ubatches:
                u = ub["ubatch"]
                ne = {}
                for kind in KINDS:
                    e = self.entries.get((kind, layer, u))
                    if e is None:
                        problems.append(f"layer {layer} ubatch {u}: no {kind} file")
                    else:
                        ne[kind] = e["ne"]
                if len(ne) != len(KINDS):
                    continue
                n_tok, n_used = ne["cur"][1], ne["topk"][0]
                if not 1 <= n_tok <= ub["n_tokens"]:
                    problems.append(f"layer {layer} ubatch {u}: cur has {n_tok} token rows, the ubatch "
                                    f"{ub['n_tokens']}")
                expected = {
                    "cur": [self.n_embd, n_tok, 1, 1],
                    "topk": [n_used, n_tok, 1, 1],
                    "weights": [1, n_used, n_tok, 1],
                    "out": [self.n_embd, n_tok, 1, 1],
                }
                for kind in KINDS:
                    if ne[kind] != expected[kind]:
                        problems.append(f"layer {layer} ubatch {u}: {kind} is {fmt_ne(ne[kind])}, "
                                        f"expected {fmt_ne(expected[kind])}")
        return problems

    def load(self, kind, layer, ubatch):
        """The tensor as a numpy array of shape [n_tok, values per token] (cur and out n_embd, topk and weights
        n_expert_used)."""
        e = self.entries[(kind, layer, ubatch)]
        data = np.fromfile(os.path.join(self.path, e["file"]), dtype=DTYPES[e["dtype"]])
        return data.reshape(self.n_tokens(layer, ubatch), -1)

    def validate(self):
        bad = self.size_mismatches()
        for name, expected, actual in bad:
            print(f"SIZE MISMATCH {name}: expected {expected} bytes, "
                  f"{'missing' if actual is None else f'{actual} bytes'}")
        problems = self.shape_problems()
        for p in problems:
            print(f"SHAPE {p}")
        return len(bad), len(problems)


def show(dump):
    m = dump.manifest
    print(f"dump:       {dump.path}")
    print(f"model:      {m['model_path']}")
    print(f"fork:       {m['fork_commit']} (build {m['fork_build_number']})")
    print(f"tokens:     n_prompt {m['n_prompt']}, n_predict {m['n_predict']}, seed {m['seed']}, "
          f"n_ubatch {m['n_ubatch']}, {len(dump.ubatches)} ubatches")
    print(f"generated:  {' '.join(str(t) for t in m['generated_token_ids'])}")
    print(f"files:      {len(m['files'])}, MoE layers {len(dump.layers)} of n_layer {m['n_layer']}")
    n_bad, n_problems = dump.validate()

    print()
    print(f"{'layer':>5}  {'ubatches':>8}  {'tokens':>6}  {'cur':<12} {'topk':<10} {'weights':<14} {'out':<12} "
          f"weights tensor")
    u0 = dump.ubatches[0]["ubatch"]
    for layer in dump.layers:
        complete = [u["ubatch"] for u in dump.ubatches
                    if all((k, layer, u["ubatch"]) in dump.entries for k in KINDS)]
        n_ub = len(complete)
        total_tokens = sum(dump.n_tokens(layer, u) for u in complete)
        shapes = [fmt_ne(dump.entries[(k, layer, u0)]["ne"]) if (k, layer, u0) in dump.entries else "-"
                  for k in KINDS]
        print(f"{layer:>5}  {n_ub:>8}  {total_tokens:>6}  {shapes[0]:<12} {shapes[1]:<10} {shapes[2]:<14} "
              f"{shapes[3]:<12} {dump.weights_tensor.get(layer, '-')}")
    print(f"(shapes of ubatch {u0}; tokens: token rows over all ubatches, fewer in a layer computed for the "
          f"output rows only)")
    print()
    print(f"size mismatches: {n_bad}, shape problems: {n_problems}")
    return 0 if n_bad == 0 and n_problems == 0 else 1


def rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    den = np.linalg.norm(a)
    num = np.linalg.norm(b - a)
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return float(num / den)


def compare(a, b, tokens_must_match):
    for d in (a, b):
        n_bad, n_problems = d.validate()
        if n_bad or n_problems:
            raise DumpError(f"{d.path}: {n_bad} size mismatches, {n_problems} shape problems")
    if a.layers != b.layers:
        raise DumpError(f"MoE layers differ: {a.layers} vs {b.layers}")

    gen_a = a.manifest["generated_token_ids"]
    gen_b = b.manifest["generated_token_ids"]
    same = 0
    while same < min(len(gen_a), len(gen_b)) and gen_a[same] == gen_b[same]:
        same += 1
    # decode call k feeds generated token k-1: its inputs match while the first k generated tokens do
    valid = [u for u in a.ubatches if u["decode_call"] <= same]
    ub_b = {u["ubatch"]: u for u in b.ubatches}
    for u in valid:
        v = ub_b.get(u["ubatch"])
        if v is None or (v["decode_call"], v["pos0"], v["n_tokens"]) != (u["decode_call"], u["pos0"], u["n_tokens"]):
            raise DumpError(f"ubatch {u['ubatch']} differs: {u} vs {v}")

    print(f"A: {a.path}\nB: {b.path}")
    if same == len(gen_a) == len(gen_b):
        print(f"generated token ids: identical ({same} tokens)")
    else:
        print(f"generated token ids: first difference at generated token {same} "
              f"(A {gen_a[same:same + 1]}, B {gen_b[same:same + 1]})")
    print(f"compared: {len(valid)} of {len(a.ubatches)} ubatches ({sum(u['n_tokens'] for u in valid)} token "
          f"positions), the ones whose inputs match")
    print()
    print(f"{'layer':>5}  {'relL2 out':>10}  {'max/ubatch':>10}  {'relL2 cur':>10}  {'relL2 w':>10}  "
          f"{'topk exact':>16}  {'topk same set':>16}")
    worst = 0.0
    all_exact = True
    for layer in a.layers:
        cat = {k: ([], []) for k in KINDS}
        max_ub = 0.0
        for u in valid:
            for k in KINDS:
                cat[k][0].append(a.load(k, layer, u["ubatch"]))
                cat[k][1].append(b.load(k, layer, u["ubatch"]))
            max_ub = max(max_ub, rel_l2(cat["out"][0][-1], cat["out"][1][-1]))
        x = {k: (np.concatenate(cat[k][0]), np.concatenate(cat[k][1])) for k in KINDS}
        r_out = rel_l2(*x["out"])
        worst = max(worst, r_out)
        ta, tb = x["topk"]
        exact = int((ta == tb).sum())
        sets = int((np.sort(ta, axis=1) == np.sort(tb, axis=1)).all(axis=1).sum())
        all_exact &= exact == ta.size
        print(f"{layer:>5}  {r_out:>10.3e}  {max_ub:>10.3e}  {rel_l2(*x['cur']):>10.3e}  "
              f"{rel_l2(*x['weights']):>10.3e}  {exact:>7}/{ta.size:<5} {100.0 * exact / ta.size:>5.1f}%  "
              f"{sets:>7}/{ta.shape[0]:<5} {100.0 * sets / ta.shape[0]:>5.1f}%")
    print()
    print(f"max relative L2 of out over layers: {worst:.3e}; topk exact match in every layer: "
          f"{'yes' if all_exact else 'no'}")
    if tokens_must_match:
        identical = same == len(gen_a) == len(gen_b)
        print(f"generated tokens identical: {'PASS' if identical else 'FAIL'} ({same} of {len(gen_a)} match)")
        return 0 if identical else 1
    return 0


EXPERT_KINDS = ("up", "gate", "down")


def expert_placement(path):
    """The routed experts' buffer per layer as llama.cpp assigns it: the first --override-tensor pattern (in
    command-line order, comma-separated values split) that regex-searches the tensor name. Prints the overrides
    that match an expert tensor and the contiguous range of layers on a CUDA buffer."""
    with open(os.path.join(path, "dump-manifest.json")) as f:
        manifest = json.load(f)
    argv, n_layer = manifest["argv"], manifest["n_layer"]
    overrides = []
    for i, a in enumerate(argv):
        if a in ("-ncmoe", "--n-cpu-moe", "-cmoe", "--cpu-moe"):
            raise DumpError(f"{path}: the run used {a}, which this placement reader does not model")
        if a in ("-ot", "--override-tensor"):
            for v in argv[i + 1].split(","):
                pattern, _, buft = v.rpartition("=")
                if not pattern:
                    raise DumpError(f"{path}: --override-tensor value {v!r} is not PATTERN=BUFFER")
                overrides.append((pattern, buft))
    used, gpu = set(), []
    for layer in range(n_layer):
        bufts = set()
        for kind in EXPERT_KINDS:
            name = f"blk.{layer}.ffn_{kind}_exps.weight"
            hit = next((k for k, (p, _) in enumerate(overrides) if re.search(p, name)), None)
            if hit is None:
                raise DumpError(f"{path}: no --override-tensor places {name}")
            used.add(hit)
            bufts.add(overrides[hit][1])
        if len(bufts) != 1:
            raise DumpError(f"{path}: layer {layer}'s expert tensors go to {sorted(bufts)}")
        if bufts.pop().startswith("CUDA"):
            gpu.append(layer)
    if not gpu or gpu != list(range(gpu[0], gpu[-1] + 1)):
        raise DumpError(f"{path}: the layers with experts on CUDA are {gpu}, not one contiguous range")
    print(f"gpu_layers {gpu[0]}-{gpu[-1]}")
    for k in sorted(used):
        print(f"override {overrides[k][0]}={overrides[k][1]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", help="dump directory to check and summarize")
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"), help="reference dump A, candidate dump B")
    ap.add_argument("--tokens-must-match", action="store_true", help="with --compare: exit 1 unless the generated "
                    "tokens are identical")
    ap.add_argument("--expert-placement", metavar="DIR", help="the run's routed-expert placement, from its argv")
    args = ap.parse_args()
    if sum(x is not None for x in (args.dir, args.compare, args.expert_placement)) != 1:
        ap.error("give one of DIR, --compare DIR_A DIR_B, --expert-placement DIR")
    if args.tokens_must_match and not args.compare:
        ap.error("--tokens-must-match goes with --compare")
    try:
        if args.expert_placement:
            return expert_placement(args.expert_placement)
        if args.compare:
            return compare(Dump(args.compare[0]), Dump(args.compare[1]), args.tokens_must_match)
        return show(Dump(args.dir))
    except DumpError as e:
        print(f"read_dump.py: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
