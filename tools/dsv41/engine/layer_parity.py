#!/usr/bin/env python3
"""The layer-0 isolation: dsv41-layer-ref's op dumps of the real model's
first layers, in its three numerics, against each other, against the real-model oracle and against the engine's own dumps
of the same call, python3 only.

  layer_parity.py compare --oracle DIR [--engine NAME=DIR ...] [--json FILE] REF_DIR [REF_DIR ...]
      REF_DIR: dsv41-layer-ref outputs (layer-ref-manifest.json, one numerics each, the same prompt). DIR: the oracle,
      a dsv41-moe-dump directory (cur, out, topk, weights of call 0) or a fork op dump (its ffn_norm,
      ffn_moe_out, ffn_moe_topk, ffn_moe_weights_scaled). NAME=DIR: an engine run's --dump-layers directory (its
      call 0 is the same prompt), e.g. dense-forced=out/dense-forced. Prints, per layer:
        oracle   each numerics' cur (ffn_norm) and out (ffn_moe_out with the oracle's expert ids) against the
                 oracle's, the isolated MoE (the oracle's cur, ids and weights through the numerics' experts:
                 ffn_moe_out.1) and the weights (.0 on its own cur, .1 on the oracle's, at the oracle's ids), and
                 how many of the router's own top-k sets are the oracle's (on its own cur, on the oracle's)
        engine   each engine dump's cur, out and l_last against each numerics' (the GPU against its references on
                 the real weights: engine numerics should be within the GPU's rounding, a semantic bug is O(1))
        ops      every op in the order computed, the rel L2 between each pair of numerics, and per pair the first
                 op at or above 1e-3 and the largest one before layer 0's ffn_norm: where the gap enters
      and the reading: whether the oracle's own matmul arithmetic (fork-cuda) explains the engine's cur gap at
      layer 0 (fork-cuda's cur at most half the engine's against the oracle): then a GPU run on MMQ (dsv41-engine
      --prefill-mmq) measures how much of the gap the f16 prefill activations make; and per engine dump whether its
      layer-0 cur is at least half as far from its own engine-numerics reference as from the oracle (then the GPU
      path, not the numerics, carries the gap: a real-model bug). --json writes the numbers and both readings
      (mmq_worth_measuring, gpu_suspect); exits 0 unless an input is missing or malformed.
  layer_parity.py ops REF_DIR DUMP_DIR
      a fork-cpu REF_DIR against the fork op dump it was made from (verify.sh: tiny-b's dump-b-1, its first ubatch):
      every op of the chain (occurrence 0) bit for bit against the dump's same file, and the isolated MoE
      (occurrence 1, from the dump's own cur, ids and weights) against the dump's ffn_moe_out, weights and top-k
      set; exits 1 on any difference.
  layer_parity.py --self-test
      compare's numbers and reading on synthetic directories (verify.sh runs it).
"""

import array
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile

ENTRY_BOUND = 1e-3       # an op "carries the gap" from this rel L2 on
MMQ_FACTOR = 0.5         # fork-cuda's cur at most this share of the engine's: the matmuls' activations explain it
# occurrence 1 of these is the isolated MoE (from the oracle's cur, ids and weights), compared with the dump's occurrence 0
ISOLATED_OPS = {"ffn_moe_out", "ffn_moe_weights_scaled", "ffn_moe_topk"}


def fail(msg):
    print(f"layer_parity.py: {msg}", file=sys.stderr)
    sys.exit(2)


def read_array(p, typecode):
    a = array.array(typecode)
    with open(p, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def rel_l2(a, b):
    if len(a) != len(b):
        fail(f"sizes differ: {len(a)} and {len(b)} values")
    num = den = 0.0
    for x, y in zip(a, b):
        num += (x - y) * (x - y)
        den += y * y
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)


class Ref:
    """A dsv41-layer-ref directory: its manifest's files by (op, occurrence, layer), in the order computed."""

    def __init__(self, d):
        p = os.path.join(d, "layer-ref-manifest.json")
        if not os.path.exists(p):
            fail(f"{d}: no layer-ref-manifest.json")
        with open(p) as f:
            m = json.load(f)
        if m.get("tool") != "dsv41-layer-ref":
            fail(f"{p}: written by {m.get('tool')!r}")
        self.d, self.m = d, m
        self.name = m["numerics"]
        self.order = [(x["op"], x["occurrence"], x["layer"]) for x in m["files"]]
        self.files = {(x["op"], x["occurrence"], x["layer"]): x for x in m["files"]}

    def has(self, op, occ, il):
        return (op, occ, il) in self.files

    def get(self, op, occ, il):
        x = self.files.get((op, occ, il))
        if x is None:
            fail(f"{self.d}: no {op}.{occ} of layer {il}")
        return read_array(os.path.join(self.d, x["file"]), "i" if x["dtype"] == "i32" else "f")


class OracleCall:
    """Call 0 of the oracle: a dsv41-moe-dump (cur-L000-u0000.bin ...) or a fork op dump (ffn_norm.0-L000-...)."""
    NAMES = {"cur": "ffn_norm", "out": "ffn_moe_out", "topk": "ffn_moe_topk", "weights": "ffn_moe_weights_scaled"}

    def __init__(self, d):
        self.d = d
        self.moe_dump = os.path.exists(os.path.join(d, "cur-L000-u0000.bin"))
        if not self.moe_dump and not os.path.exists(os.path.join(d, "ffn_norm.0-L000-u0000.bin")):
            fail(f"{d}: neither a dsv41-moe-dump nor a fork op dump")

    def get(self, kind, il):
        name = f"{kind}-L{il:03d}-u0000.bin" if self.moe_dump else f"{self.NAMES[kind]}.0-L{il:03d}-u0000.bin"
        p = os.path.join(self.d, name)
        if not os.path.exists(p):
            fail(f"no {p}")
        return read_array(p, "i" if kind == "topk" else "f")


def engine_call0(d, op, il, n):
    """The first n values of an engine dump's call 0 (its prompt call: exactly n when the prompt is one chunk)."""
    p = os.path.join(d, f"{op}.0-L{il:03d}-u0000.bin")
    if not os.path.exists(p):
        fail(f"no {p}")
    a = read_array(p, "i" if op == "ffn_moe_topk" else "f")
    if len(a) != n:
        fail(f"{p}: {len(a)} values, the reference's call has {n} (the engine's first call must be the whole prompt)")
    return a


def sets_equal(a, b, k):
    return sum(set(a[t * k:(t + 1) * k]) == set(b[t * k:(t + 1) * k]) for t in range(len(b) // k))


def compare(ref_dirs, oracle_dir, engines, json_path):
    refs = [Ref(d) for d in ref_dirs]
    names = [r.name for r in refs]
    if len(set(names)) != len(names):
        fail(f"two directories of one numerics: {names}")
    prompt = refs[0].m["prompt_token_ids"]
    for r in refs:
        if r.m["prompt_token_ids"] != prompt or r.m["layers"] != refs[0].m["layers"]:
            fail(f"{r.d}: another prompt or layer count than {refs[0].d}")
    n_layers, T = refs[0].m["layers"], len(prompt)
    o = OracleCall(oracle_dir)
    out = {"oracle": oracle_dir, "n_tokens": T, "layers": n_layers, "numerics": names, "oracle_rows": [],
           "engine_rows": [], "pairs": []}
    print(f"layer_parity.py: {', '.join(names)} on {T} tokens, layers 0 .. {n_layers - 1}, against the oracle "
          f"{oracle_dir}")
    for il in range(n_layers):
        ocur, oout, ow, otk = o.get("cur", il), o.get("out", il), o.get("weights", il), o.get("topk", il)
        K = len(otk) // T
        for r in refs:
            row = {"layer": il, "numerics": r.name, "cur": rel_l2(r.get("ffn_norm", 0, il), ocur),
                   "out": rel_l2(r.get("ffn_moe_out", 0, il), oout),
                   "weights": rel_l2(r.get("ffn_moe_weights_scaled", 0, il), ow),
                   "topk_same": sets_equal(r.get("ffn_moe_topk", 0, il), otk, K)}
            if r.has("ffn_moe_out", 1, il):
                row.update({"out_isolated": rel_l2(r.get("ffn_moe_out", 1, il), oout),
                            "weights_isolated": rel_l2(r.get("ffn_moe_weights_scaled", 1, il), ow),
                            "topk_same_isolated": sets_equal(r.get("ffn_moe_topk", 1, il), otk, K)})
            out["oracle_rows"].append(row)
            iso = ("" if "out_isolated" not in row else
                   f"; isolated (the oracle's cur, ids, weights): out {row['out_isolated']:.3g}, weights "
                   f"{row['weights_isolated']:.3g}, own top-k {row['topk_same_isolated']} of {T}")
            print(f"  oracle   layer {il} {r.name:9s}: cur {row['cur']:.3g}, out {row['out']:.3g}, weights "
                  f"{row['weights']:.3g}, own top-k {row['topk_same']} of {T}{iso}")
        for ename, ed in engines:
            E = len(ocur) // T
            ecur = engine_call0(ed, "ffn_norm", il, T * E)
            eout = engine_call0(ed, "ffn_moe_out", il, T * E)
            elast = engine_call0(ed, "l_last", il, T * 4 * E)
            row = {"layer": il, "run": ename, "cur_oracle": rel_l2(ecur, ocur), "out_oracle": rel_l2(eout, oout),
                   "against": {}}
            print(f"  engine   layer {il} {ename}: cur {row['cur_oracle']:.3g} and out {row['out_oracle']:.3g} against "
                  f"the oracle (its call 0)")
            for r in refs:
                a = row["against"][r.name] = {"cur": rel_l2(ecur, r.get("ffn_norm", 0, il)),
                                              "out": rel_l2(eout, r.get("ffn_moe_out", 0, il)),
                                              "l_last": rel_l2(elast, r.get("l_last", 0, il))}
                print(f"  engine   layer {il} {ename} against {r.name:9s}: cur {a['cur']:.3g}, out {a['out']:.3g}, "
                      f"l_last {a['l_last']:.3g}")
            out["engine_rows"].append(row)

    # the ops, pair by pair, in the first directory's order
    # the chain's f32 ops (the hyper-connections' two occurrences), not the isolated MoE nor the expert ids
    order = [k for k in refs[0].order if all(r.has(*k) for r in refs) and refs[0].files[k]["dtype"] == "f32" and
             not (k[0] in ISOLATED_OPS and k[1] == 1)]
    pairs = [(a, b) for i, a in enumerate(refs) for b in refs[i + 1:]]
    if pairs:
        print("  ops      rel L2 between numerics (" + "; ".join(f"{a.name} vs {b.name}" for a, b in pairs) + ")")
    vals = {}
    for k in order:
        op, occ, il = k
        vals[k] = [rel_l2(a.get(*k), b.get(*k)) for a, b in pairs]
        out.setdefault("ops", []).append({"op": op, "occurrence": occ, "layer": il,
                                          "rel_l2": {f"{a.name} vs {b.name}": v for (a, b), v in zip(pairs, vals[k])}})
        if pairs:
            print(f"    L{il if il >= 0 else 999:03d} {op}.{occ:<1} " + "  ".join(f"{v:.3g}" for v in vals[k]))
    for i, (a, b) in enumerate(pairs):
        first = next((k for k in order if vals[k][i] >= ENTRY_BOUND), None)
        before = []
        for k in order:
            before.append(k)
            if k == ("ffn_norm", 0, 0):
                break
        worst = max(before, key=lambda k: vals[k][i]) if before else None
        rec = {"a": a.name, "b": b.name, "first_at_1e-3": None if first is None else {"op": first[0], "layer": first[2],
                                                                                       "rel_l2": vals[first][i]},
               "largest_before_l0_cur": None if worst is None else {"op": worst[0], "layer": worst[2],
                                                                     "rel_l2": vals[worst][i]}}
        out["pairs"].append(rec)
        ftxt = "none" if first is None else f"{first[0]} of layer {first[2]} ({vals[first][i]:.3g})"
        wtxt = "n/a" if worst is None else f"{worst[0]} of layer {worst[2]} ({vals[worst][i]:.3g})"
        print(f"  entry    {a.name} vs {b.name}: the first op at or above {ENTRY_BOUND:g}: {ftxt}; the largest up to "
              f"layer 0's ffn_norm: {wtxt}")

    # the reading: does the oracle's matmul arithmetic explain the engine numerics' cur gap at layer 0?
    l0 = {r["numerics"]: r["cur"] for r in out["oracle_rows"] if r["layer"] == 0}
    worth = None
    if "engine" in l0 and "fork-cuda" in l0:
        worth = l0["fork-cuda"] <= MMQ_FACTOR * l0["engine"]
        print(f"  reading  layer 0 cur against the oracle: engine numerics {l0['engine']:.3g}, the oracle's matmul "
              f"arithmetic (fork-cuda) {l0['fork-cuda']:.3g}"
              f"{', the fork CPU ' + format(l0['fork-cpu'], '.3g') if 'fork-cpu' in l0 else ''}: "
              + ("the quantized matmuls' activations carry most of the gap: measure the GPU on MMQ (--prefill-mmq)"
                 if worth else "the matmuls' activations do not explain the gap: look at the ops above"))
    out["mmq_worth_measuring"] = worth
    # a GPU bug on the real weights: the engine's dump as far from its own numerics' reference as from the oracle
    suspect = []
    for row in out["engine_rows"]:
        if row["layer"] == 0 and "engine" in row["against"]:
            gpu_ref, gpu_oracle = row["against"]["engine"]["cur"], row["cur_oracle"]
            bad = gpu_ref >= 0.5 * gpu_oracle
            suspect.append({"run": row["run"], "gpu_vs_reference": gpu_ref, "gpu_vs_oracle": gpu_oracle, "suspect": bad})
            print(f"  reading  {row['run']} layer 0 cur: {gpu_ref:.3g} from its engine-numerics reference, "
                  f"{gpu_oracle:.3g} from the oracle: "
                  + ("as far from its own reference as from the oracle: suspect the GPU path on the real weights"
                     if bad else "its reference explains it (the gap is the numerics', not the GPU's)"))
    out["gpu_suspect"] = suspect
    if json_path:
        with open(json_path, "w") as f:
            json.dump(out, f, indent=1)
    print("layer_parity.py: done")
    return out


def ops(ref_dir, dump_dir):
    r = Ref(ref_dir)
    ok, n = True, 0
    for k in r.order:
        op, occ, il = k
        name = f"{op}.{0 if op in ISOLATED_OPS else occ}-L{il if il >= 0 else 999:03d}-u0000.bin"
        p = os.path.join(dump_dir, name)
        if not os.path.exists(p):
            print(f"FAIL {op}.{occ} of layer {il}: no {name} in {dump_dir}")
            ok = False
            continue
        x = r.files[k]
        mine, theirs = r.get(*k), read_array(p, "i" if x["dtype"] == "i32" else "f")
        if op == "ffn_moe_topk" and occ == 1:
            w = x["row"]
            same = sets_equal(mine, theirs, w) == len(theirs) // w
        else:
            same = mine.tobytes() == theirs.tobytes()
        n += 1
        if not same:
            ok = False
            diff = "sets differ" if op == "ffn_moe_topk" else f"rel L2 {rel_l2(mine, theirs):.3g}"
            print(f"FAIL {op}.{occ} of layer {il} against {name}: {diff}")
    print(f"layer_parity.py ops: {n} dumps of {ref_dir} ({r.name}) against {dump_dir}: "
          f"{'every one bit for bit' if ok else 'DIFFER'}")
    return ok


# ------------------------------------------------------------------------------------------------ --self-test
def self_test():
    T, E, K, L = 3, 8, 2, 2

    def write(path, typecode, values):
        a = array.array(typecode, values)
        if sys.byteorder != "little":
            a.byteswap()
        with open(path, "wb") as f:
            f.write(a.tobytes())

    def base(i):
        return 1.0 + 0.01 * i

    root = tempfile.mkdtemp(prefix="layer-parity-self-test-")
    good = True
    try:
        od = os.path.join(root, "oracle")
        os.makedirs(od)
        for il in range(L):
            write(os.path.join(od, f"cur-L{il:03d}-u0000.bin"), "f", [base(i) for i in range(T * E)])
            write(os.path.join(od, f"out-L{il:03d}-u0000.bin"), "f", [base(i) + 1 for i in range(T * E)])
            write(os.path.join(od, f"topk-L{il:03d}-u0000.bin"), "i", [0, 1] * T)
            write(os.path.join(od, f"weights-L{il:03d}-u0000.bin"), "f", [0.75, 0.75] * T)

        # numerics off by `cur_off` in cur (relative), engine also off in the q projection of layer 0
        def ref(name, cur_off, q_off):
            d = os.path.join(root, name)
            os.makedirs(d)
            files = []

            def put(op, occ, il, typecode, values, row):
                f = f"{op}.{occ}-L{il if il >= 0 else 999:03d}-u0000.bin"
                write(os.path.join(d, f), typecode, values)
                files.append({"file": f, "op": op, "occurrence": occ, "layer": il,
                              "dtype": "i32" if typecode == "i" else "f32", "row": row, "count": len(values)})
            put("embd", 0, -1, "f", [base(i) for i in range(T * E)], E)
            for il in range(L):
                put("q", 0, il, "f", [base(i) * (1 + (q_off if il == 0 else 0)) for i in range(T * E)], E)
                put("ffn_norm", 0, il, "f", [base(i) * (1 + cur_off) for i in range(T * E)], E)
                put("ffn_moe_topk", 0, il, "i", [0, 1] * T, K)
                put("ffn_moe_weights_scaled", 0, il, "f", [0.75, 0.75] * T, K)
                put("ffn_moe_out", 0, il, "f", [base(i) + 1 for i in range(T * E)], E)
                put("l_last", 0, il, "f", [base(i) for i in range(T * 4 * E)], 4 * E)
                put("ffn_moe_weights_scaled", 1, il, "f", [0.75, 0.75] * T, K)
                put("ffn_moe_topk", 1, il, "i", [0, 1] * T, K)
                put("ffn_moe_out", 1, il, "f", [base(i) + 1 for i in range(T * E)], E)
            with open(os.path.join(d, "layer-ref-manifest.json"), "w") as f:
                json.dump({"tool": "dsv41-layer-ref", "numerics": name, "n_tokens": T, "layers": L,
                           "prompt_token_ids": [1, 2, 3], "files": files}, f)
            return d

        dirs = [ref("fork-cpu", 0.004, 0.0), ref("fork-cuda", 0.001, 0.0), ref("engine", 0.01, 0.02)]
        ed = os.path.join(root, "engine-run")
        os.makedirs(ed)
        for il in range(L):
            write(os.path.join(ed, f"ffn_norm.0-L{il:03d}-u0000.bin"), "f", [base(i) * 1.0101 for i in range(T * E)])
            write(os.path.join(ed, f"ffn_moe_out.0-L{il:03d}-u0000.bin"), "f", [base(i) + 1 for i in range(T * E)])
            write(os.path.join(ed, f"l_last.0-L{il:03d}-u0000.bin"), "f", [base(i) for i in range(T * 4 * E)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = compare(dirs, od, [("dense-forced", ed)], None)
        l0 = {r["numerics"]: r for r in res["oracle_rows"] if r["layer"] == 0}
        eng = res["engine_rows"][0]
        checks = [
            ("cur against the oracle per numerics", abs(l0["engine"]["cur"] - 0.01) < 1e-6 and
             abs(l0["fork-cuda"]["cur"] - 0.001) < 1e-6),
            ("the isolated MoE and the top-k sets", l0["engine"]["out_isolated"] == 0 and l0["engine"]["topk_same"] == T),
            ("the engine dump against its reference", abs(eng["against"]["engine"]["cur"] - 0.0001 / 1.01) < 1e-6),
            ("the GPU is explained by its reference", res["gpu_suspect"] == [
                {"run": "dense-forced", "gpu_vs_reference": eng["against"]["engine"]["cur"],
                 "gpu_vs_oracle": eng["cur_oracle"], "suspect": False}]),
            ("the gap enters at layer 0's q (engine against fork-cuda)",
             res["pairs"][2]["first_at_1e-3"]["op"] == "q" and res["pairs"][2]["first_at_1e-3"]["layer"] == 0),
            ("the reading: the matmuls' activations explain the gap", res["mmq_worth_measuring"] is True),
        ]
        for what, right in checks:
            good = good and right
            print(f"{'ok  ' if right else 'FAIL'} self-test: {what}")
        if not good:
            print(buf.getvalue())
        # the reading the other way: fork-cuda no closer than the engine
        shutil.rmtree(dirs[1])
        dirs[1] = ref("fork-cuda", 0.009, 0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            res = compare(dirs, od, [], None)
        right = res["mmq_worth_measuring"] is False
        good = good and right
        print(f"{'ok  ' if right else 'FAIL'} self-test: the reading: the matmuls' activations do not explain the gap")
    finally:
        shutil.rmtree(root)
    return good


def main(argv):
    args = argv[1:]
    if args == ["--self-test"]:
        ok = self_test()
        print(f"layer_parity.py: self-test {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if len(args) == 3 and args[0] == "ops":
        return 0 if ops(args[1], args[2]) else 1
    if args and args[0] == "compare":
        args = args[1:]
        oracle, json_path, engines, refs = None, None, [], []
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("--oracle", "--json", "--engine"):
                if i + 1 >= len(args):
                    fail(f"{a} needs a value")
                v = args[i + 1]
                if a == "--oracle":
                    oracle = v
                elif a == "--json":
                    json_path = v
                else:
                    if "=" not in v:
                        fail(f"--engine {v}: NAME=DIR")
                    engines.append(tuple(v.split("=", 1)))
                i += 2
                continue
            refs.append(a)
            i += 1
        if not oracle or not refs:
            fail("compare needs --oracle DIR and at least one REF_DIR")
        compare(refs, oracle, engines, json_path)
        return 0
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
