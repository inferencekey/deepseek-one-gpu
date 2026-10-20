#!/usr/bin/env python3
"""The engine against the real-model oracle: a dsv41-engine --dump-layers run of the real DeepSeek-V4.1-Flash
Q2_K against the stock fork's dsv41-moe-dump of the same prompt (on an A100, $BUILD_ROOT/oracle-dsv41/run-1), python3
only.

  oracle_parity.py forced ORACLE_DIR OUT_DIR [--top-k K]
      a --force-selection reference made from the oracle: dump-manifest.json (its prompt, generated ids and calls),
      per call and layer ffn_moe_topk (the oracle's topk files), ffn_norm (its cur files: the MoE input that
      dsv41-engine --forced-moe-input feeds every layer) and idx_top_k. The oracle recorded no indexer rows,
      so each idx_top_k lists every row 0 .. P - 1 of the dump's P positions, of which the engine keeps those visible
      at a token's position (selection.cpp): the indexer's own pick while the visible rows are at most top_k K
      (default 512, SPEC.md); a dump with P > K is refused.
  oracle_parity.py ENGINE_DIR ORACLE_DIR [--bound B] [--json FILE]
      the comparison, per MoE layer and over every position whose inputs are the same in both runs (the prompt's,
      then each decode step up to and including the first whose fed token differs; a forced run feeds the oracle's,
      so all 95), of the oracle's four kinds (SPEC.md "Real-model oracle mapping") with the engine's dumps:
        cur      ffn_norm, the MoE input [T][E]                 rel L2
        out      ffn_moe_out, the routed experts' sum [T][E]    rel L2 (before the shared expert)
        weights  ffn_moe_weights_scaled [T][K]                  rel L2 over the tokens whose expert sets agree,
                                                               matched by expert id
        topk     ffn_moe_topk [T][K]                            the tokens whose expert sets agree, and each flip
                                                               with the engine's k-th - (k+1)-th score
                                                               (ffn_moe_probs_biased)
      and the greedy tokens: how many of the oracle's generated ids the engine's equal (position by position), the
      equal prefix, and at the first divergence the engine's own logits there (its pick against the oracle's: a gap
      of at most 2 B rms(logits) is a near-tie, the bound engine_parity.py uses).
      Verdicts: a forced run (manifest forced_selection) passes
      when every layer's cur and out rel L2 are at most B (default 2e-2) and no expert set differs. An isolated run
      (manifest forced_moe_input) is a forced run whose every layer read the
      oracle's cur: its cur must be the oracle's (rel L2 at most 1e-6, else the run did not take it), its out and
      weights are the router and the routed experts alone on the oracle's input (the per-layer isolated number),
      and the router's own top-k on that input (from ffn_moe_probs_biased) is counted against the oracle's sets,
      each flip with its margin (reported); a free run
      passes when at least 90% of the generated ids are the oracle's and a first divergence, if any, is a near-tie.
      A free run's per-layer numbers are reported (a selection flip moves its token and everything downstream).
      --json writes the numbers and the verdict.
  oracle_parity.py --self-test
      the checks on synthetic dumps (tools/dsv41/verify.sh runs it).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_parity import Run, fmt_set, margin, rel_l2  # noqa: E402

DEFAULT_BOUND = 2e-2
DEFAULT_TOP_K = 512
TOKENS_SHARE = 0.9
# the engine's dump of each oracle kind, and the element type
KINDS = {"cur": ("ffn_norm", "f"), "out": ("ffn_moe_out", "f"), "topk": ("ffn_moe_topk", "i"),
         "weights": ("ffn_moe_weights_scaled", "f")}


def fail(msg):
    print(f"oracle_parity.py: {msg}", file=sys.stderr)
    sys.exit(2)


def read_array(p, typecode):
    a = array.array(typecode)
    with open(p, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


class Oracle:
    """A dsv41-moe-dump directory: its manifest's calls and files, rows by position, as engine_parity.Run."""

    def __init__(self, d):
        self.d = d
        p = os.path.join(d, "dump-manifest.json")
        if not os.path.exists(p):
            fail(f"{d}: no dump-manifest.json")
        with open(p) as f:
            m = json.load(f)
        if m.get("tool") != "dsv41-moe-dump":
            fail(f"{p}: written by {m.get('tool')!r}, not dsv41-moe-dump")
        self.prompt = list(m["prompt_token_ids"])
        self.generated = list(m["generated_token_ids"])
        self.n_layer = m["n_layer"]
        self.ubatches = {u["ubatch"]: (u["pos0"], u["n_tokens"]) for u in m["ubatches"]}
        self.files = {(x["kind"], x["layer"], x["ubatch"]): x["file"] for x in m["files"]}
        self.at = {}
        for u, (p0, n) in self.ubatches.items():
            for k in range(n):
                self.at[p0 + k] = (u, k)
        self.cache = {}

    def row(self, kind, il, pos):
        u, k = self.at[pos]
        key = (kind, il, u)
        if key not in self.cache:
            if key not in self.files:
                fail(f"{self.d}: no {kind} file of layer {il}, ubatch {u}")
            self.cache[key] = read_array(os.path.join(self.d, self.files[key]), KINDS[kind][1])
        a = self.cache[key]
        w = len(a) // self.ubatches[u][1]
        return a[k * w:(k + 1) * w]

    def forget(self):
        self.cache.clear()


# ------------------------------------------------------------------------------------------------ forced reference
def make_forced(oracle_dir, out_dir, top_k):
    o = Oracle(oracle_dir)
    n_pos = len(o.at)
    if n_pos > top_k:
        fail(f"{oracle_dir}: {n_pos} positions, more than top_k {top_k}: the indexer's rows are not every visible row "
             "and the oracle did not record them")
    os.makedirs(out_dir, exist_ok=True)
    man = {"tool": "oracle_parity.py forced", "source": os.path.abspath(oracle_dir), "prompt_token_ids": o.prompt,
           "generated_token_ids": o.generated, "n_layer": o.n_layer,
           "ubatches": [{"ubatch": u, "pos0": p0, "n_tokens": n} for u, (p0, n) in sorted(o.ubatches.items())]}
    with open(os.path.join(out_dir, "dump-manifest.json"), "w") as f:
        json.dump(man, f)
    every_row = array.array("i", range(n_pos))
    if sys.byteorder != "little":
        every_row.byteswap()
    for u, (_, n) in sorted(o.ubatches.items()):
        for il in range(o.n_layer):
            for kind, op in (("topk", "ffn_moe_topk"), ("cur", "ffn_norm")):
                if (kind, il, u) not in o.files:
                    fail(f"{oracle_dir}: no {kind} file of layer {il}, ubatch {u}")
                shutil.copyfile(os.path.join(oracle_dir, o.files[(kind, il, u)]),
                                os.path.join(out_dir, f"{op}.0-L{il:03d}-u{u:04d}.bin"))
            with open(os.path.join(out_dir, f"idx_top_k.0-L{il:03d}-u{u:04d}.bin"), "wb") as f:
                f.write(every_row.tobytes() * n)
    print(f"oracle_parity.py: {out_dir}: the forced reference of {oracle_dir}: {len(o.prompt)} prompt tokens, "
          f"{len(o.generated)} fed tokens, {len(o.ubatches)} calls x {o.n_layer} layers of ffn_moe_topk and ffn_norm, "
          f"idx_top_k every row of the {n_pos} positions (top_k {top_k})")


# ------------------------------------------------------------------------------------------------ comparison
def compare(engine_dir, oracle_dir, bound, json_path=None):
    e, o = Run(engine_dir), Oracle(oracle_dir)
    em = e.man
    if em["prompt"] != o.prompt:
        fail(f"the prompts differ ({len(em['prompt'])} and {len(o.prompt)} tokens)")
    if em["n_layer"] != o.n_layer:
        fail(f"{em['n_layer']} and {o.n_layer} layers")
    forced, isolated = bool(em["forced"]), bool(em["moe_input"])
    if isolated and not forced:
        fail(f"{engine_dir}: the oracle's MoE input without a forced selection")
    mode = "isolated" if isolated else "forced" if forced else "free"
    print(f"oracle_parity.py: {engine_dir} against the oracle {oracle_dir}, "
          f"{'forced selection (' + em['forced'] + ')' if forced else 'free selection'}"
          f"{', every layer fed the oracle cur (isolated MoE)' if isolated else ''}, prefill {em['prefill']}, "
          f"bound {bound:g}")
    n_prompt = len(o.prompt)
    n = min(len(em["fed"]), len(o.generated))
    first_fed = next((i for i in range(n) if em["fed"][i] != o.generated[i]), None)
    last_step = n if first_fed is None else first_fed
    positions = [p for p in range(n_prompt + last_step) if p in e.at and p in o.at]
    if any(p not in e.at or p not in o.at for p in range(n_prompt)):
        fail(f"the prompt's {n_prompt} positions are not all in both runs")
    print(f"  {len(positions)} positions compared ({n_prompt} prompt, {len(positions) - n_prompt} decode steps)")
    ok = True
    layers, flips, own_flips = [], [], []
    for il in range(o.n_layer):
        acc = {k: ([], []) for k in ("cur", "out", "weights")}
        per_call = {k: {} for k in ("cur", "out")}
        same_set = own_same = 0
        for pos in positions:
            for kind in ("cur", "out"):
                a, b = e.row(KINDS[kind][0], il, pos, "f"), o.row(kind, il, pos)
                acc[kind][0].extend(a)
                acc[kind][1].extend(b)
                pa, pb = per_call[kind].setdefault(e.at[pos][0], ([], []))
                pa.extend(a)
                pb.extend(b)
            ti, ri = e.row("ffn_moe_topk", il, pos, "i"), o.row("topk", il, pos)
            if isolated:
                # the router's own pick on the oracle's input: the top K of its biased probs (ties to the lower id)
                probs = e.row("ffn_moe_probs_biased", il, pos, "f")
                own = set(sorted(range(len(probs)), key=lambda x: (-probs[x], x))[:len(ri)])
                if own == set(ri):
                    own_same += 1
                else:
                    kth = sorted(probs, reverse=True)[len(ri) - 1]
                    own_flips.append((il, pos, own, set(ri), margin(probs, len(ri)),
                                      kth - max(probs[x] for x in set(ri) - own)))
            if set(ti) == set(ri):
                same_set += 1
                tw, rw = e.row("ffn_moe_weights_scaled", il, pos, "f"), o.row("weights", il, pos)
                by_id = dict(zip(ti, tw))
                acc["weights"][0].extend(by_id[x] for x in ri)
                acc["weights"][1].extend(rw)
            else:
                probs = e.row("ffn_moe_probs_biased", il, pos, "f", False)
                m = margin(probs, len(ti)) if probs is not None else None
                missed = set(ri) - set(ti)
                gap = None
                if probs is not None and missed:
                    kth = sorted(probs, reverse=True)[len(ti) - 1]
                    gap = kth - max(probs[x] for x in missed)
                flips.append((il, pos, set(ti), set(ri), m, gap))
        e.forget()
        o.forget()
        row = {"layer": il, "positions": len(positions), "topk_same": same_set}
        if isolated:
            row["own_topk_same"] = own_same
        for kind in ("cur", "out"):
            row[kind] = rel_l2(*acc[kind])
            worst = max(((rel_l2(a, b), u) for u, (a, b) in per_call[kind].items()), default=(0.0, None))
            row[kind + "_worst_call"] = {"rel_l2": worst[0], "call": worst[1]}
        row["weights"] = rel_l2(*acc["weights"]) if acc["weights"][0] else None
        over = row["cur"] > bound or row["out"] > bound
        if isolated and row["cur"] > 1e-6:
            print(f"FAIL layer {il:2d}: cur rel L2 {row['cur']:.3g}: the MoE input is not the oracle's")
            ok = False
        layer_flips = sum(f[0] == il for f in flips)
        if forced:
            status = "FAIL" if over or layer_flips else "ok  "
            ok = ok and status == "ok  "
        else:
            status = ("flip" if any(f[0] <= il for f in flips) else "over") if over else "ok  "
        layers.append(row)
        wtxt = "n/a" if row["weights"] is None else f"{row['weights']:.3g}"
        print(f"{status} layer {il:2d}: cur rel L2 {row['cur']:.3g} (worst call {row['cur_worst_call']['call']}: "
              f"{row['cur_worst_call']['rel_l2']:.3g}), out {row['out']:.3g} (worst call "
              f"{row['out_worst_call']['call']}: {row['out_worst_call']['rel_l2']:.3g}), weights {wtxt}, expert sets "
              f"equal at {same_set} of {len(positions)} positions"
              f"{f', its own top-k on the oracle cur at {own_same}' if isolated else ''}")
    worst_cur = max(layers, key=lambda r: r["cur"])
    worst_out = max(layers, key=lambda r: r["out"])
    print(f"  worst layer: cur {worst_cur['cur']:.3g} (layer {worst_cur['layer']}), out {worst_out['out']:.3g} "
          f"(layer {worst_out['layer']}), bound {bound:g}{' (the forced gate)' if forced else ' (reported)'}")
    for il, pos, ts, rs, m, gap in own_flips:
        print(f"  note own top-k on the oracle cur, layer {il}, position {pos}: {fmt_set(ts)} against the oracle's "
              f"{fmt_set(rs)}, the engine's k-th - (k+1)-th score {m:.3g}, the oracle's expert {gap:.3g} below its k-th")
    if isolated:
        n_own = sum(r["own_topk_same"] for r in layers)
        print(f"  the router's own top-k on the oracle cur: {n_own} of {len(positions) * o.n_layer} (layer, position) "
              f"sets are the oracle's (reported)")
    for il, pos, ts, rs, m, gap in flips:
        mtxt = "no score dump" if m is None else f"the engine's k-th - (k+1)-th score {m:.3g}"
        gtxt = "" if gap is None else f", the oracle's expert {gap:.3g} below the engine's k-th"
        print(f"  {'FAIL' if forced else 'flip'} router flip at layer {il}, position {pos}: {fmt_set(ts)} against the "
              f"oracle's {fmt_set(rs)}, {mtxt}{gtxt}")

    # the greedy tokens: generated id i comes from the logits at position n_prompt - 1 + i
    gen, ref = em["generated"], o.generated
    n_gen = min(len(gen), len(ref))
    identical = sum(gen[i] == ref[i] for i in range(n_gen))
    prefix = next((i for i in range(n_gen) if gen[i] != ref[i]), n_gen)
    tokens = {"identical": identical, "of": n_gen, "prefix": prefix, "first_divergence": None}
    if prefix < n_gen:
        last_pos = n_prompt - 1 + prefix
        if last_pos not in e.ends:
            fail(f"no engine call ends at position {last_pos} (the logits of generated id {prefix})")
        logits = read_array(os.path.join(engine_dir, f"result_output.0-L999-u{e.ends[last_pos]:04d}.bin"), "f")
        gap = logits[gen[prefix]] - logits[ref[prefix]]
        rms = math.sqrt(sum(v * v for v in logits) / len(logits))
        tie = 2 * bound * rms
        tokens["first_divergence"] = {"step": prefix, "engine": gen[prefix], "oracle": ref[prefix], "gap": gap,
                                      "tie_bound": tie, "near_tie": gap <= tie,
                                      "flips_before": sum(f[1] <= last_pos for f in flips)}
    share = identical / n_gen if n_gen else 0.0
    if forced:
        tok_ok = True
        print(f"note tokens: {identical} of {n_gen} generated ids are the oracle's (forced: fed the oracle's tokens, "
              "reported)")
    else:
        d = tokens["first_divergence"]
        tok_ok = share >= TOKENS_SHARE and (d is None or d["near_tie"])
        ok = ok and tok_ok
        print(f"{'ok  ' if tok_ok else 'FAIL'} tokens: {identical} of {n_gen} generated ids are the oracle's "
              f"({100 * share:.1f}%, bar {100 * TOKENS_SHARE:.0f}%), the first {prefix} in a row")
    d = tokens["first_divergence"]
    if d:
        print(f"{'ok  ' if d['near_tie'] else 'FAIL' if not forced else 'note'} first divergence: step {d['step']} "
              f"picks {d['engine']} where the oracle picks {d['oracle']}: the engine's logits there differ by "
              f"{d['gap']:.4g}, {'a near-tie within' if d['near_tie'] else 'more than'} 2 B rms = {d['tie_bound']:.4g}; "
              f"{d['flips_before']} router flips at or before its position")
    if forced and flips:
        ok = False
        print(f"FAIL selections: {len(flips)} expert sets differ from the oracle's (a forced run must have none)")
    print(f"oracle_parity.py: {'PASS' if ok else 'FAIL'}")
    if json_path:
        with open(json_path, "w") as f:
            json.dump({"engine_dir": engine_dir, "oracle_dir": oracle_dir, "mode": mode, "prefill": em["prefill"],
                       "readers": json.load(open(os.path.join(engine_dir, "engine-manifest.json"))).get("readers"),
                       "bound": bound, "positions": len(positions), "layers": layers,
                       "worst_cur": {"layer": worst_cur["layer"], "rel_l2": worst_cur["cur"]},
                       "worst_out": {"layer": worst_out["layer"], "rel_l2": worst_out["out"]},
                       "router_flips": len(flips),
                       "own_flips": [{"layer": il, "position": pos, "margin": m, "gap": gap}
                                     for il, pos, _, _, m, gap in own_flips], "tokens": tokens, "pass": ok}, f, indent=1)
    return ok


# ------------------------------------------------------------------------------------------------ --self-test
def self_test():
    n_layer, E, K, NE, n_vocab, n_prompt, n_gen = 2, 8, 2, 4, 16, 3, 20

    def write(path, typecode, values):
        a = array.array(typecode, values)
        if sys.byteorder != "little":
            a.byteswap()
        with open(path, "wb") as f:
            f.write(a.tobytes())

    ubs = [(0, n_prompt)] + [(n_prompt + i, 1) for i in range(n_gen)]
    ref_gen = [(5 + i) % n_vocab for i in range(n_gen)]

    def val(kind, il, pos, i):
        return 1.0 + 0.01 * (i + pos + il) + (0.5 if kind == "out" else 0.0)

    def oracle(d):
        os.makedirs(d)
        files = []
        for u, (p0, n) in enumerate(ubs):
            for il in range(n_layer):
                for kind in ("cur", "out", "topk", "weights"):
                    name = f"{kind}-L{il:03d}-u{u:04d}.bin"
                    if kind == "topk":
                        write(os.path.join(d, name), "i", [0, 2] * n)
                    elif kind == "weights":
                        write(os.path.join(d, name), "f", [0.6, 0.4] * n)
                    else:
                        write(os.path.join(d, name), "f", [val(kind, il, p0 + t, i) for t in range(n) for i in range(E)])
                    files.append({"file": name, "kind": kind, "layer": il, "ubatch": u})
        with open(os.path.join(d, "dump-manifest.json"), "w") as f:
            json.dump({"tool": "dsv41-moe-dump", "prompt_token_ids": [1, 2, 3], "generated_token_ids": ref_gen,
                       "n_layer": n_layer, "ubatches": [{"ubatch": u, "pos0": p, "n_tokens": n}
                                                        for u, (p, n) in enumerate(ubs)], "files": files}, f)

    def engine(d, forced=False, bump=None, diverge=None, gap=0.0, flip=None, moe_input=False, cur_bump=0.0,
               own_flip=False):
        """bump: (layer, size) added to out at every position; diverge: the step whose pick differs (every later
        step picks the oracle's again), gap: its logit margin; flip: (layer, position) whose expert set differs;
        moe_input: an isolated run (its cur the oracle's plus cur_bump); own_flip: the biased probs pick another
        set than the ids (the router's own top-k on the oracle's cur)"""
        os.makedirs(d)
        gen = list(ref_gen)
        if diverge is not None:
            gen[diverge] = (ref_gen[diverge] + 1) % n_vocab
        fed = ref_gen if forced else gen
        for u, (p0, n) in enumerate(ubs):
            for il in range(n_layer):
                b = bump[1] if bump and bump[0] == il else 0.0
                write(os.path.join(d, f"ffn_norm.0-L{il:03d}-u{u:04d}.bin"), "f",
                      [val("cur", il, p0 + t, i) + cur_bump for t in range(n) for i in range(E)])
                write(os.path.join(d, f"ffn_moe_out.0-L{il:03d}-u{u:04d}.bin"), "f",
                      [val("out", il, p0 + t, i) + b for t in range(n) for i in range(E)])
                ids = []
                for t in range(n):
                    ids += [0, 3] if flip == (il, p0 + t) else [2, 0]
                write(os.path.join(d, f"ffn_moe_topk.0-L{il:03d}-u{u:04d}.bin"), "i", ids)
                write(os.path.join(d, f"ffn_moe_weights_scaled.0-L{il:03d}-u{u:04d}.bin"), "f", [0.4, 0.6] * n)
                write(os.path.join(d, f"ffn_moe_probs_biased.0-L{il:03d}-u{u:04d}.bin"), "f",
                      ([1.0, 0.95, 0.9, 0.1] if own_flip else [1.0, 0.5, 0.9, 0.89]) * n)
            # the logits of the call's last token: its pick on top, the oracle's `gap` below at the divergence
            step = p0 + n - n_prompt
            logits = [0.1 * (i % 3) for i in range(n_vocab)]
            if step < n_gen:
                logits[gen[step]] = 3.0
                if step == diverge:
                    logits[ref_gen[step]] = 3.0 - gap
            write(os.path.join(d, f"result_output.0-L999-u{u:04d}.bin"), "f", logits)
        with open(os.path.join(d, "engine-manifest.json"), "w") as f:
            json.dump({"tool": "dsv41-engine", "readers": "dense", "n_layer": n_layer, "prompt_token_ids": [1, 2, 3],
                       "generated_token_ids": gen, "fed_token_ids": fed, "forced_selection": "ref" if forced else "",
                       "forced_moe_input": moe_input,
                       "ubatches": [{"ubatch": u, "pos0": p, "n_tokens": n} for u, (p, n) in enumerate(ubs)]}, f)

    cases = [
        ("a forced run equal to the oracle", {"forced": True}, True),
        ("a forced run whose routed sum is 5% off in layer 1", {"forced": True, "bump": (1, 0.08)}, False),
        ("a forced run with a flipped expert set", {"forced": True, "flip": (1, 4)}, False),
        ("a free run with the oracle's tokens", {}, True),
        ("a free run diverging at the last step on a near-tie", {"diverge": n_gen - 1, "gap": 0.01}, True),
        ("a free run diverging at the last step by a clear margin", {"diverge": n_gen - 1, "gap": 2.0}, False),
        ("a free run diverging at step 1 on a near-tie (19 of 20 equal, its rows past the divergence not compared)",
         {"diverge": 1, "gap": 0.01}, True),
        ("a free run with a router flip and an off routed sum: reported", {"flip": (0, 2), "bump": (0, 0.08)}, True),
        ("an isolated run on the oracle's cur", {"forced": True, "moe_input": True}, True),
        ("an isolated run whose router picks other experts on the oracle's cur: reported",
         {"forced": True, "moe_input": True, "own_flip": True}, True),
        ("an isolated run whose cur is not the oracle's", {"forced": True, "moe_input": True, "cur_bump": 1e-3}, False),
        ("an isolated run whose routed sum is 5% off in layer 1", {"forced": True, "moe_input": True, "bump": (1, 0.08)},
         False),
    ]
    root = tempfile.mkdtemp(prefix="oracle-parity-self-test-")
    good = True
    try:
        od = os.path.join(root, "oracle")
        oracle(od)
        for i, (what, kw, want) in enumerate(cases):
            td = os.path.join(root, f"e{i}")
            engine(td, **kw)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                got = compare(td, od, DEFAULT_BOUND)
            right = got == want
            good = good and right
            print(f"{'ok  ' if right else 'FAIL'} self-test: {what}: {'PASS' if got else 'FAIL'}, expected "
                  f"{'PASS' if want else 'FAIL'}")
            if not right:
                print(buf.getvalue())
        # the forced reference: the oracle's topk as the engine's file names, every row in idx_top_k
        fd = os.path.join(root, "forced")
        with contextlib.redirect_stdout(io.StringIO()):
            make_forced(od, fd, DEFAULT_TOP_K)
        n_pos = n_prompt + n_gen
        rows = read_array(os.path.join(fd, "idx_top_k.0-L001-u0000.bin"), "i")
        ids = read_array(os.path.join(fd, "ffn_moe_topk.0-L001-u0003.bin"), "i")
        cur = read_array(os.path.join(fd, "ffn_norm.0-L001-u0003.bin"), "f")
        with open(os.path.join(fd, "dump-manifest.json")) as f:
            man = json.load(f)
        right = (list(rows) == list(range(n_pos)) * n_prompt and list(ids) == [0, 2] and
                 list(cur) == list(read_array(os.path.join(od, "cur-L001-u0003.bin"), "f")) and
                 man["generated_token_ids"] == ref_gen and len(man["ubatches"]) == len(ubs))
        good = good and right
        print(f"{'ok  ' if right else 'FAIL'} self-test: the forced reference holds the oracle's expert ids, its cur as "
              f"ffn_norm and every row of the {n_pos} positions")
    finally:
        shutil.rmtree(root)
    return good


def main(argv):
    args = argv[1:]
    opts = {"--bound": DEFAULT_BOUND, "--top-k": DEFAULT_TOP_K, "--json": None}
    for k in list(opts):
        if k in args:
            i = args.index(k)
            if i + 1 >= len(args):
                fail(f"{k} needs a value")
            opts[k] = args[i + 1] if k == "--json" else (int if k == "--top-k" else float)(args[i + 1])
            del args[i:i + 2]
    if args == ["--self-test"]:
        ok = self_test()
        print(f"oracle_parity.py: self-test {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if len(args) == 3 and args[0] == "forced":
        make_forced(args[1], args[2], opts["--top-k"])
        return 0
    if len(args) == 2 and not args[0].startswith("-"):
        return 0 if compare(args[0], args[1], opts["--bound"], opts["--json"]) else 1
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
