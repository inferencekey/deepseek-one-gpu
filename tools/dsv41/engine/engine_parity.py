#!/usr/bin/env python3
"""The engine's end-to-end parity: a dsv41-engine --dump-layers directory against a fork op dump, python3 only.

  engine_parity.py ENGINE_DIR FORK_DUMP_DIR [--bound B]   the engine against the stock fork
  engine_parity.py --engines DIR_A DIR_B [--bound B]       two engine runs (two expert placements) against each other
  engine_parity.py --self-test                             the checks on synthetic dumps (verify.sh runs it)

Both directories hold, per call u (the prompt's chunks, then one decode step per generated token) and layer l,
l_last.0-L<l>-u<u>.bin ([T][HC][E] f32, the fork's [E, HC, T]), the router's ffn_moe_topk [T][K] i32 and
ffn_moe_probs_biased [T][NE] f32, an index source's idx_top_k [T][k] i32 and idx_score [T][C] f32 (-inf past the
visible rows), result_output.0-L999-u<u>.bin (the logits of the call's last token), and a manifest
(engine-manifest.json, the fork's dump-manifest.json) with the prompt, the generated and fed ids and each call's first
position and tokens. B defaults to 5e-2: the bound of the engine's op tests against the fork (dense_test_common.hpp
kEngineRelL2, attention 5e-2), whose engine numerics (f16 or int8 per 32 activations, f32 sums) sit 1e-2 to 3e-2 from
the fork's int8 per 256 on these random tiny weights; a wrong row, layer or slot is O(1).

Positions compared: those whose inputs are the same, the prompt's, then each decode step up to and including the
first whose fed token differs (a free run feeds its own picks; a --force-selection run feeds the reference's, so all).
The two runs' calls are matched by position, so either may hold the prompt in several chunks (a --chunk below
the prompt against the fork's one ubatch); a call's logits are compared where both runs have a call ending at the same
position.

  tokens     at every compared step whose pick differs, the reference's logits there must show a near-tie: the gap
             between its pick and the other run's at most 2 B rms(logits) (the per-value error a rel L2 of B allows
             on both picks), else FAIL.
  flips      per call, layer and token, the router's expert set and an index source's visible row set against the
             reference's, with the reference's margin between its k-th and (k+1)-th score (ffn_moe_probs_biased,
             idx_score): a flip there is a near-tie the two numerics resolve differently.
  l_last     per layer and token, rel L2 at most B. Forced run (manifest forced_selection): every token, and no flip
             may remain (FAIL). Free run: a token over B must follow a flip, at its own position in this or an
             earlier layer, or at an earlier position in an earlier layer (the attention reads the earlier
             positions' rows of the layer's input), and is reported with it; a token over B without one is a bug
             (FAIL).
  logits     per call, rel L2 at most B; in a free run a call over B must follow a flip at or before its last position.

Prints one line per layer (ok, flip: over B only after flips, FAIL), each token over B with the flip it follows, the
flips, the token verdict and PASS or FAIL; exits 1 on FAIL.
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


def fail(msg):
    print(f"engine_parity.py: {msg}", file=sys.stderr)
    sys.exit(2)


def manifest(d):
    for name in ("engine-manifest.json", "dump-manifest.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            ub = {u["ubatch"]: (u["pos0"], u["n_tokens"]) for u in m["ubatches"]}
            gen = list(m["generated_token_ids"])
            return {"prompt": list(m["prompt_token_ids"]), "generated": gen, "fed": list(m.get("fed_token_ids", gen)),
                    "forced": m.get("forced_selection", ""), "moe_input": bool(m.get("forced_moe_input", False)),
                    "prefill": m.get("prefill", "dequant-f16"), "ubatches": ub, "n_layer": m["n_layer"], "kind": name}
    fail(f"{d}: no engine-manifest.json or dump-manifest.json")


def path(d, op, layer, u):
    return os.path.join(d, f"{op}.0-L{layer:03d}-u{u:04d}.bin")


def load(d, op, layer, u, typecode="f", required=True):
    p = path(d, op, layer, u)
    if not os.path.exists(p):
        if required:
            fail(f"no {p}")
        return None
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


def margin(scores, k):
    """The k-th largest finite score minus the (k+1)-th, None when at most k are finite (all selected)."""
    v = sorted((x for x in scores if math.isfinite(x)), reverse=True)
    return v[k - 1] - v[k] if len(v) > k else None


def fmt_set(s):
    return "{" + ",".join(map(str, sorted(s))) + "}"


class Run:
    """One dump directory: its manifest, where each position is (call, token), each call's arrays loaded once."""

    def __init__(self, d):
        self.d = d
        self.man = manifest(d)
        self.at = {}
        self.ends = {}
        for u, (p0, n) in self.man["ubatches"].items():
            for k in range(n):
                self.at[p0 + k] = (u, k)
            self.ends[p0 + n - 1] = u
        self.cache = {}

    def call(self, op, il, u, typecode="f", required=True):
        key = (op, il, u)
        if key not in self.cache:
            self.cache[key] = load(self.d, op, il, u, typecode, required)
        return self.cache[key]

    def row(self, op, il, pos, typecode="f", required=True):
        u, k = self.at[pos]
        a = self.call(op, il, u, typecode, required)
        if a is None:
            return None
        w = len(a) // self.man["ubatches"][u][1]
        return a[k * w:(k + 1) * w]

    def forget(self):
        self.cache.clear()


def flips_of(test, ref, n_layer, positions):
    """Every selection that differs: (layer, test call, its token, position, kind, test set, reference set, reference
    margin)."""
    out = []
    for il in range(n_layer):
        for pos in positions:
            u, t = test.at[pos]
            ta, ra = test.row("ffn_moe_topk", il, pos, "i", False), ref.row("ffn_moe_topk", il, pos, "i", False)
            if ta is None or ra is None:
                fail(f"layer {il}, position {pos}: ffn_moe_topk missing in {test.d if ta is None else ref.d}")
            if set(ta) != set(ra):
                probs = ref.row("ffn_moe_probs_biased", il, pos, "f", False)
                m = margin(probs, len(ra)) if probs is not None else None
                out.append((il, u, t, pos, "router", set(ta), set(ra), m))
            ti, ri = test.row("idx_top_k", il, pos, "i", False), ref.row("idx_top_k", il, pos, "i", False)
            if ti is None and ri is None:
                continue
            if ti is None or ri is None:
                fail(f"layer {il}, position {pos}: idx_top_k missing in {test.d if ti is None else ref.d}")
            row = ref.row("idx_score", il, pos)
            vis = {j for j in range(len(row)) if math.isfinite(row[j])}
            ts, rs = {j for j in ti if j in vis}, {j for j in ri if j in vis}
            if ts != rs:
                out.append((il, u, t, pos, "index", ts, rs, margin(row, len(ri))))
        test.forget()
        ref.forget()
    return out


def upstream(flips, il, pos):
    """The flips (layer, token) at layer il, position pos follows: its own position in this or an earlier layer, or
    an earlier position in an earlier layer; the nearest first."""
    hits = [f for f in flips if (f[3] == pos and f[0] <= il) or (f[3] < pos and f[0] < il)]
    return sorted(hits, key=lambda f: (-f[0], -f[3]))


def describe(f, ref_name):
    il, u, t, pos, kind, ts, rs, m = f
    mtxt = "no margin (every row selected)" if m is None else f"{ref_name}'s k-th - (k+1)-th score {m:.3g}"
    return f"the {kind} flip at layer {il}, call {u} token {t} (position {pos}): {fmt_set(ts)} against {fmt_set(rs)}, {mtxt}"


def compare(test_dir, ref_dir, bound, ref_name):
    test, ref = Run(test_dir), Run(ref_dir)
    t, r = test.man, ref.man
    if t["prompt"] != r["prompt"]:
        fail(f"the prompts differ ({len(t['prompt'])} and {len(r['prompt'])} tokens)")
    if t["n_layer"] != r["n_layer"]:
        fail(f"{t['n_layer']} and {r['n_layer']} layers")
    forced = bool(t["forced"])
    print(f"engine_parity.py: {test_dir} against {ref_dir}, "
          f"{'forced selection (' + t['forced'] + ')' if forced else 'free selection'}, bound {bound:g}")
    ok = True
    # the positions whose inputs are the same: the prompt's, then each decode step up to and including the first fed
    # token that differs (decode step s at position n_prompt + s feeds the token of step s - 1). The calls are matched
    # by position, so either run may hold the prompt in several chunks.
    n_prompt = len(t["prompt"])
    n = min(len(t["fed"]), len(r["fed"]))
    first_fed = next((i for i in range(n) if t["fed"][i] != r["fed"][i]), None)
    last_step = n if first_fed is None else first_fed
    positions = [p for p in range(n_prompt + last_step) if p in test.at and p in ref.at]
    if any(p not in test.at or p not in ref.at for p in range(n_prompt)):
        fail(f"the prompt's {n_prompt} positions are not all in both runs' calls")
    for p in range(n_prompt, n_prompt + last_step):
        for run in (test, ref):
            if p in run.at and run.man["ubatches"][run.at[p][0]][1] != 1:
                fail(f"position {p} (a decode step) is in a call of {run.man['ubatches'][run.at[p][0]][1]} tokens in {run.d}")
    calls = sorted({test.at[p][0] for p in positions})
    print(f"  {len(positions)} positions compared: {len(calls)} calls of this run, "
          f"{len({ref.at[p][0] for p in positions})} of {ref_name}'s")
    flips = flips_of(test, ref, t["n_layer"], positions)

    for il in range(t["n_layer"]):
        worst, where = 0.0, None
        tokens = 0
        worst_tok, where_tok = 0.0, None
        over = []   # (u, k, position, rel L2)
        for u in calls:
            p0, n_tok = t["ubatches"][u]
            mine = [p for p in range(p0, p0 + n_tok) if p in ref.at and p < n_prompt + last_step]
            a, b = [], []
            for pos in mine:
                ra, rb = test.row("l_last", il, pos), ref.row("l_last", il, pos)
                a.extend(ra)
                b.extend(rb)
                y = rel_l2(ra, rb)
                tokens += 1
                if y > bound:
                    over.append((u, pos - p0, pos, y))
                if y >= worst_tok:
                    worst_tok, where_tok = y, (u, pos - p0)
            x = rel_l2(a, b)
            if x >= worst:
                worst, where = x, u
        test.forget()
        ref.forget()
        lines, unexplained = [], 0
        for u, k, pos, y in over:
            up = upstream(flips, il, pos)
            if up:
                lines.append(f"       call {u} token {k} (position {pos}): {y:.3g}, after {describe(up[0], ref_name)}")
            else:
                unexplained += 1
                lines.append(f"       call {u} token {k} (position {pos}): {y:.3g}, after no selection flip: a bug")
        if not over:
            status = "ok  "
        elif forced or unexplained:
            status = "FAIL"
        else:
            status = "flip"
        ok = ok and status != "FAIL"
        tail = ""
        if over and not forced:
            tail = f", {len(over) - unexplained} after a selection flip, {unexplained} without one"
        print(f"{status} l_last layer {il}: worst rel L2 {worst:.3g} (call {where}) over {len(calls)} calls, "
              f"bound {bound:g}; per token worst {worst_tok:.3g} (call {where_tok[0]} token {where_tok[1]}), "
              f"{len(over)} of {tokens} tokens over the bound{tail}")
        for line in lines:
            print(line)

    # the logits of every call of this run that ends where one of the reference's ends (a chunk inside the
    # reference's prompt ubatch has none to meet)
    worst, where, bad, n_logits = 0.0, None, [], 0
    for u in calls:
        last_pos = t["ubatches"][u][0] + t["ubatches"][u][1] - 1
        if last_pos not in ref.ends or last_pos >= n_prompt + last_step:
            continue
        n_logits += 1
        x = rel_l2(load(test_dir, "result_output", 999, u), load(ref_dir, "result_output", 999, ref.ends[last_pos]))
        if x >= worst:
            worst, where = x, u
        if x > bound:
            bad.append((u, x, [f for f in flips if f[3] <= last_pos]))
    unexplained = [b for b in bad if not b[2]]
    status = "ok  " if not bad else ("FAIL" if forced or unexplained else "flip")
    ok = ok and status != "FAIL"
    print(f"{status} logits: worst rel L2 {worst:.3g} (call {where}) over {n_logits} calls, bound {bound:g}"
          + (f"; {len(bad)} calls over the bound, {len(bad) - len(unexplained)} after a selection flip" if bad else ""))
    for u, x, fl in bad:
        print(f"       call {u}: {x:.3g}, " + (f"after {len(fl)} flips, the last {describe(fl[-1], ref_name)}" if fl
                                              else "after no selection flip: a bug"))

    n_router = sum(f[4] == "router" for f in flips)
    status = "ok  " if not flips else ("FAIL" if forced else "flip")
    ok = ok and status != "FAIL"
    print(f"{status} selections: {n_router} router and {len(flips) - n_router} index flips against {ref_name}'s over "
          f"{len(positions)} positions" + (" (a forced run must have none)" if forced and flips else ""))
    for f in flips:
        print(f"       {describe(f, ref_name)}")

    # generated id i comes from the logits at position n_prompt - 1 + i
    n_gen = min(len(t["generated"]), len(r["generated"]), last_step + 1)
    differ = [i for i in range(n_gen) if t["generated"][i] != r["generated"][i]]
    if not differ:
        print(f"ok   tokens: the {n_gen} generated ids are {ref_name}'s: {' '.join(map(str, t['generated'][:n_gen]))}")
    for i in differ:
        logits = load(ref_dir, "result_output", 999, ref.ends[n_prompt - 1 + i])
        pick_r, pick_t = r["generated"][i], t["generated"][i]
        gap = logits[pick_r] - logits[pick_t]
        rms = math.sqrt(sum(v * v for v in logits) / len(logits))
        tie = 2 * bound * rms
        good = gap <= tie
        ok = ok and good
        print(f"{'ok  ' if good else 'FAIL'} tokens: {i} of {n_gen} compared ids equal {ref_name}'s before step {i}; "
              f"step {i} picks {pick_t} where {ref_name} picks {pick_r}: {ref_name}'s logits there differ by "
              f"{gap:.4g}, {'a near-tie within' if good else 'more than'} 2 B rms = {tie:.4g}")
    return ok


# ---- --self-test: two synthetic runs of 3 layers (layer 1 an index source), a 4-token prompt and 2 steps ----

def self_test():
    n_layer, hc_e, k, ne, top_k = 3, 8, 2, 4, 2

    def write(d, op, il, u, typecode, values):
        a = array.array(typecode, values)
        if sys.byteorder != "little":
            a.byteswap()
        with open(path(d, op, il, u), "wb") as f:
            f.write(a.tobytes())

    def run(d, forced, router_flip=None, jump=None, chunked=False):
        """router_flip: (layer, u, token) whose expert set differs; jump: (layer, u, token, size) added to l_last there
        and to every downstream token of that position; chunked: the prompt in two calls of 2 tokens"""
        os.makedirs(d)
        ubs = [(0, 2), (2, 2), (4, 1), (5, 1)] if chunked else [(0, 4), (4, 1), (5, 1)]
        man = {"prompt_token_ids": [1, 2, 3, 4], "generated_token_ids": [5, 6], "fed_token_ids": [5, 6],
               "forced_selection": "ref" if forced else "", "n_layer": n_layer,
               "ubatches": [{"ubatch": u, "pos0": p, "n_tokens": n} for u, (p, n) in enumerate(ubs)]}
        with open(os.path.join(d, "engine-manifest.json"), "w") as f:
            json.dump(man, f)
        for u, (p0, n_tok) in enumerate(ubs):
            for il in range(n_layer):
                vals, ids, probs = [], [], []
                for t in range(n_tok):
                    pos = p0 + t
                    bump = 0.0
                    if jump and ((pos == jump[2] + ubs[jump[1]][0] and il >= jump[0]) or
                                 (pos > jump[2] + ubs[jump[1]][0] and il > jump[0])):
                        bump = jump[3]
                    vals += [1.0 + 0.01 * (i + pos + il) + bump for i in range(hc_e)]
                    flip = router_flip == (il, u, t)
                    ids += [0, 3] if flip else [0, 2]
                    probs += [1.0, 0.5, 0.9, 0.8999]
                write(d, "l_last", il, u, "f", vals)
                write(d, "ffn_moe_topk", il, u, "i", ids)
                write(d, "ffn_moe_probs_biased", il, u, "f", probs)
                if il == 1:
                    rows, scores = [], []
                    for t in range(n_tok):
                        vis = (p0 + t + 1) // 2
                        rows += [j for j in range(min(vis, top_k))] + [-1] * (top_k - min(vis, top_k))
                        scores += [1.0 / (j + 1) if j < vis else -math.inf for j in range(4)]
                    write(d, "idx_top_k", il, u, "i", rows)
                    write(d, "idx_score", il, u, "f", scores)
            write(d, "result_output", 999, u, "f", [float(i % 7) for i in range(16)])

    cases = [
        ("identical runs", {}, {}, True),
        ("a router flip at layer 1, call 0 token 2, and the jump it makes there and downstream",
         {"router_flip": (1, 0, 2), "jump": (1, 0, 2, 1.0)}, {}, True),
        ("a jump at layer 1, call 0 token 2 without a flip", {"jump": (1, 0, 2, 1.0)}, {}, False),
        ("a jump in a decode step without a flip", {"jump": (2, 2, 0, 1.0)}, {}, False),
        ("a forced run whose jump follows a flip", {"router_flip": (1, 0, 2), "jump": (1, 0, 2, 1.0), "forced": True}, {},
         False),
        ("a forced run with a flip and no jump", {"router_flip": (0, 1, 0), "forced": True}, {}, False),
        ("a forced run within the bound", {"forced": True}, {}, True),
        ("a run with the prompt in two chunks against one ubatch", {"chunked": True}, {}, True),
        ("a chunked forced run within the bound", {"chunked": True, "forced": True}, {}, True),
        ("a jump in the second chunk without a flip", {"chunked": True, "jump": (1, 1, 0, 1.0)}, {}, False),
        ("a chunked run against a chunked reference", {"chunked": True}, {"chunked": True}, True),
    ]
    root = tempfile.mkdtemp(prefix="engine-parity-self-test-")
    good = True
    try:
        for i, (what, test_kw, ref_kw, want) in enumerate(cases):
            td, rd = os.path.join(root, f"t{i}"), os.path.join(root, f"r{i}")
            forced = test_kw.pop("forced", False)
            run(td, forced, **test_kw)
            run(rd, False, **ref_kw)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                got = compare(td, rd, 5e-2, "the reference")
            right = got == want
            good = good and right
            print(f"{'ok  ' if right else 'FAIL'} self-test: {what}: {'PASS' if got else 'FAIL'}, expected "
                  f"{'PASS' if want else 'FAIL'}")
            if not right:
                print(buf.getvalue())
    finally:
        shutil.rmtree(root)
    return good


def main(argv):
    args = argv[1:]
    bound = 5e-2
    if "--bound" in args:
        i = args.index("--bound")
        bound = float(args[i + 1])
        del args[i:i + 2]
    if args == ["--self-test"]:
        ok = self_test()
    elif len(args) == 3 and args[0] == "--engines":
        ok = compare(args[1], args[2], bound, "the other run")
    elif len(args) == 2 and not args[0].startswith("-"):
        ok = compare(args[0], args[1], bound, "the fork")
    else:
        print(__doc__.strip())
        return 2
    print(f"engine_parity.py: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
