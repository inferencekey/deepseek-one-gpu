#!/usr/bin/env python3
"""The per-request hot set's policies replayed on routing traces, python3 only.

  hot_policy_sim.py replay TRACE --hot-set FILE [--policy P] [KNOBS]    every request with its policy (or P): hits
                                                                       per layer, in dsv41_engine_hot_policy_test
                                                                       --replay's format (verify.sh diffs the two)
  hot_policy_sim.py compare TRACE --hot-set FILE [KNOBS]                static, prompt and adaptive on every request:
                                                                       the hit rate per layer and overall
  hot_policy_sim.py synthetic OUT_DIR [--layers N] [--experts N] [--used K] [--slab N] [--requests N] [--prompt N]
                    [--decode N] [--seed S]                            a skewed synthetic trace and its hot set
  hot_policy_sim.py oracle ORACLE_DIR OUT_TRACE                         a fork op dump's (or the real-model oracle's) topk
                                                                       files as a trace: ubatch 0 the prompt, the
                                                                       others its decode steps
  hot_policy_sim.py two-turn OUT_DIR [--experts N] [--slab N] [--prompt N] [--decode N] [--suffix N] [--seed S]
                                                                       a conversation of two turns (turn 1 a long
                                                                       prompt and its decode, turn 2 a short suffix
                                                                       and its decode) as two traces, turn 2 resumed
                                                                       (trace-resume.txt) and fresh (trace-fresh.txt),
                                                                       and turn 2's verdict (exit 1 on FAIL)
  hot_policy_sim.py batch TRACE --hot-set FILE --rows SPEC[@S],... [--reads TRACE2:REQ@S,...] [--steps N]
                    [--cuts A,B,...] [KNOBS]                          batch slots decoding together (a row: a
                                                                       request's decode, or REQ/pN its prompt's tokens
                                                                       from N, from step S), prompts read beside them
                                                                       at their steps under each read rule (target,
                                                                       none, union: merge_prompt): per window between
                                                                       the cuts the hit rate and per step the distinct
                                                                       experts, CPU jobs and swaps
  hot_policy_sim.py --self-test                                         synthetic: prompt and adaptive above static;
                                                                       two turns: the resumed turn 2 hits at least as
                                                                       often as the fresh one with far fewer swaps
                                                                       (and within 1% of the pairs of the continued run)
KNOBS: --swaps N (2) --spare N (2) --half-life H (128) --margin MUL:ADD (1.5:1) --mapping-pinned, as dsv41-engine's
--hot-* flags.

The trace (dsv41-engine --route-trace FILE; dsv41-route-trace/1), one line each:
  # dsv41-route-trace/1 n_layer L n_expert NE n_used K     the header
  R <request> <policy> reset=<0|1> [resume=<0|1>]          a request starts (its policy; reset: the slab back to the
                                                            hot set's before it; resume: it reuses the sequence's
                                                            prefix without a reset, so the planners keep their
                                                            counts; absent in traces written before it: 0)
  P <layer> <T> <T K ids>                                   a prefill call of T tokens: each token's K routed experts
  D <layer> <K ids>                                         one decode token of a layer
The hot set is a strata-moe-hot-set/1 file (moe_hot_set.h): per layer its VRAM experts (the static slab, in slot
order) and its warm list (the warm tier's experts, each in its own pinned home).

The rules are include/strata/dsv41/engine/hot_policy.hpp's, repeated here operation for operation (the orders, the
ties, the limits, the counts' double arithmetic), per layer: a request's policy, its prompt counted from its P lines,
its target at the prompt's end and the first swaps planned; then per decode token its pairs are hits when the slab
holds their experts, its ids counted (adaptive: every count times 2^(-1 / half_life), then one per use), then the
boundary: the swaps planned at the last one published (their copies taken as done by then: the engine's --hot-check,
which waits for them), the next ones planned. A resumed request (resume=1) starts from the counts the last one left,
its target from them plus its prompt's, ties by the slab's residency before the static order. A layer has spare slots when it has experts in and out of its slab, and
spare homes beside its warm tier when the mapping is not pinned (the real model's). Only an expert in a home (or in
a pinned mapping, --mapping-pinned) is ever promoted.
"""

import json
import math
import os
import random
import struct
import sys
import tempfile

STATIC, PROMPT, ADAPTIVE = "static", "prompt", "adaptive"
POLICIES = (STATIC, PROMPT, ADAPTIVE)


def fail(msg):
    print(f"hot_policy_sim.py: {msg}", file=sys.stderr)
    sys.exit(2)


class Knobs:
    def __init__(self):
        self.swaps, self.spare, self.half_life, self.mul, self.add, self.mapping_pinned = 2, 2, 128.0, 1.5, 1.0, False


class LayerPlan:
    """hot_policy.hpp HotLayerPlan."""

    def __init__(self, n_expert, vram, spare, warm, spare_homes, mapping_pinned):
        self.ne, self.pinned = n_expert, mapping_pinned
        self.slot_of = [-1] * n_expert
        self.home_of = [-1] * n_expert
        self.static = [False] * n_expert
        self.flight = [False] * n_expert
        self.rank = [-1] * n_expert
        self.slot_expert = [-1] * (len(vram) + spare)
        self.home_expert = [-1] * (len(warm) + spare_homes)
        r = 0
        for s, e in enumerate(vram):
            self.rank[e], r = r, r + 1
            self.slot_of[e], self.slot_expert[s], self.static[e] = s, e, True
        for k, e in enumerate(warm):
            if self.rank[e] >= 0:
                fail(f"expert {e} is in the hot set twice")
            self.rank[e], r = r, r + 1
            self.home_of[e], self.home_expert[k] = k, e
        for e in range(n_expert):
            if self.rank[e] < 0:
                self.rank[e], r = r, r + 1
        self.n_static = len(vram)
        self.free_slots = list(range(len(self.slot_expert) - 1, len(vram) - 1, -1))
        self.free_homes = list(range(len(self.home_expert) - 1, len(warm) - 1, -1))
        self.n_flight = 0
        self.score = [0.0] * n_expert
        self.begin_request(STATIC, False)

    def begin_request(self, p, resume):
        self.policy, self.prompt_seen, self.lfu = p, False, False
        self.resume = resume and p != STATIC
        self.prompt = [0.0] * self.ne
        if not self.resume:
            self.score = [0.0] * self.ne
        self.target = list(self.static)

    def count_prompt(self, ids):
        for x in ids:
            self.prompt[x] += 1.0

    def promotable(self, e):
        return self.slot_of[e] < 0 and (self.home_of[e] >= 0 or self.pinned)

    def prompt_done(self, k):
        self.prompt_seen, self.lfu = True, False
        if self.policy == STATIC:
            return
        if self.resume:
            self.score = [c + x for c, x in zip(self.score, self.prompt)]
        else:
            self.score = list(self.prompt)
        self.prompt = [0.0] * self.ne        # taken into the counts (merge_prompt merges a later read alone)
        if self.resume and self.policy == ADAPTIVE:
            # the slab, with the swaps adaptive's margin allows at once
            self.target = [self.slot_of[e] >= 0 for e in range(self.ne)]
            ins = [e for e in range(self.ne) if self.promotable(e)]
            outs = [e for e in range(self.ne) if self.slot_of[e] >= 0]
            ins.sort(key=lambda e: (-self.score[e], self.rank[e]))
            outs.sort(key=lambda e: (self.score[e], -self.rank[e]))
            for i, o in zip(ins, outs):
                if not self.score[i] > k.mul * self.score[o] + k.add:
                    break
                self.target[i], self.target[o] = True, False
            return
        if self.resume:
            key = lambda e: (-self.score[e], self.slot_of[e] < 0, self.rank[e])
        else:
            key = lambda e: (-self.score[e], self.rank[e])
        order = [e for e in range(self.ne) if self.slot_of[e] >= 0 or self.promotable(e)]
        order.sort(key=key)
        self.target = [False] * self.ne
        for e in order[:self.n_static]:
            self.target[e] = True

    def merge_prompt(self, mass):
        total = sum(self.prompt)
        if self.policy == ADAPTIVE and self.prompt_seen and total > 0:
            self.score = [c + x * (mass / total) for c, x in zip(self.score, self.prompt)]
        self.prompt = [0.0] * self.ne

    def count_token(self, ids, decay):
        if self.policy != ADAPTIVE or not self.prompt_seen:
            return
        self.score = [c * decay for c in self.score]
        for x in ids:
            self.score[x] += 1.0

    def start(self, i, o, out):
        s = {"in": i, "out": o, "slot": self.free_slots.pop(), "from_home": self.home_of[i], "to_home": -1}
        if not self.pinned:
            s["to_home"] = self.free_homes.pop()
        self.flight[i] = self.flight[o] = True
        self.n_flight += 1
        out.append(s)

    def can_start(self):
        return self.free_slots and (self.pinned or self.free_homes)

    def plan_target(self, mx, out):
        ins = [e for e in range(self.ne) if not self.flight[e] and self.target[e] and self.promotable(e)]
        outs = [e for e in range(self.ne) if not self.flight[e] and not self.target[e] and self.slot_of[e] >= 0]
        ins.sort(key=lambda e: (-self.score[e], self.rank[e]))
        outs.sort(key=lambda e: (self.score[e], -self.rank[e]))
        n = 0
        for i, o in zip(ins, outs):
            if n >= mx or not self.can_start():
                break
            self.start(i, o, out)
            n += 1
        return n, len(ins) > n

    def plan_lfu(self, k, mx, out):
        n = 0
        while n < mx and self.can_start():
            i = v = -1
            for e in range(self.ne):
                if self.flight[e]:
                    continue
                s = self.score[e]
                if self.slot_of[e] >= 0:
                    if v < 0 or s < self.score[v] or (s == self.score[v] and self.rank[e] > self.rank[v]):
                        v = e
                elif self.promotable(e):
                    if i < 0 or s > self.score[i] or (s == self.score[i] and self.rank[e] < self.rank[i]):
                        i = e
            if i < 0 or v < 0 or not self.score[i] > k.mul * self.score[v] + k.add:
                break
            self.start(i, v, out)
            n += 1

    def plan(self, k, out):
        if k.spare == 0 or k.swaps < 1:
            return
        if self.policy == STATIC:
            self.target = list(self.static)
            self.plan_target(k.swaps, out)
            return
        if not self.prompt_seen:
            return
        n = 0
        if not self.lfu:
            n, left = self.plan_target(k.swaps, out)
            if self.policy == ADAPTIVE and not left:
                self.lfu = True
        if self.policy == ADAPTIVE and self.lfu:
            self.plan_lfu(k, k.swaps - n, out)

    def plan_static(self, mx, out):
        keep = self.target
        self.target = list(self.static)
        self.plan_target(mx, out)
        self.target = keep

    def publish(self, s):
        old = self.slot_of[s["out"]]
        self.slot_of[s["in"]], self.slot_expert[s["slot"]] = s["slot"], s["in"]
        self.slot_of[s["out"]], self.slot_expert[old] = -1, -1
        self.free_slots.append(old)
        if s["from_home"] >= 0:
            self.home_expert[s["from_home"]], self.home_of[s["in"]] = -1, -1
            self.free_homes.append(s["from_home"])
        if s["to_home"] >= 0:
            self.home_expert[s["to_home"]], self.home_of[s["out"]] = s["out"], s["to_home"]
        self.flight[s["in"]] = self.flight[s["out"]] = False
        self.n_flight -= 1

    def is_static(self):
        return self.n_flight == 0 and all((self.slot_of[e] >= 0) == self.static[e] for e in range(self.ne))


def read_trace(path):
    with open(path) as f:
        head = f.readline().split()
        if head[:2] != ["#", "dsv41-route-trace/1"] or len(head) != 8:
            fail(f"{path}: not a dsv41-route-trace/1 file")
        n_layer, n_expert, n_used = int(head[3]), int(head[5]), int(head[7])
        reqs = []
        for line in f:
            w = line.split()
            if not w or w[0].startswith("#"):
                continue
            if w[0] == "R":
                if len(w) not in (4, 5) or (len(w) == 5 and w[4] not in ("resume=0", "resume=1")):
                    fail(f"{path}: an R line {line.strip()!r}")
                reqs.append({"name": w[1], "policy": w[2], "reset": w[3] == "reset=1",
                             "resume": len(w) == 5 and w[4] == "resume=1",
                             "prompt": [[] for _ in range(n_layer)], "decode": [[] for _ in range(n_layer)]})
                continue
            if not reqs:
                fail(f"{path}: a routing line before the first R line")
            layer = int(w[1])
            if w[0] == "P":
                T = int(w[2])
                ids = [int(x) for x in w[3:]]
                if len(ids) != T * n_used:
                    fail(f"{path}: a P line of {len(ids)} ids for {T} tokens")
                reqs[-1]["prompt"][layer].append(ids)
            elif w[0] == "D":
                ids = [int(x) for x in w[2:]]
                if len(ids) != n_used:
                    fail(f"{path}: a D line of {len(ids)} ids")
                reqs[-1]["decode"][layer].append(ids)
            else:
                fail(f"{path}: line kind {w[0]}")
    return n_layer, n_expert, n_used, reqs


def read_hot_set(path):
    with open(path) as f:
        doc = json.load(f)
    return {l["layer"]: (l["experts"], l.get("warm", [])) for l in doc["layers"]}


def plans_for(n_layer, n_expert, hot, k):
    plans = []
    for l in range(n_layer):
        vram, warm = hot.get(l, ([], []))
        mixed = 0 < len(vram) < n_expert
        spare = k.spare if mixed else 0
        homes = k.spare if mixed and not k.mapping_pinned and warm else 0
        plans.append(LayerPlan(n_expert, vram, spare, warm, homes, k.mapping_pinned))
    return plans


def replay(n_layer, n_expert, reqs, hot, k, policy=None, out=None):
    """Per request and layer (hits, pairs, swaps) of the decode tokens (swaps: the ones the request planned, the reset's
    not counted); prints dsv41_engine_hot_policy_test's lines to out."""
    plans = plans_for(n_layer, n_expert, hot, k)
    decay = 2.0 ** (-1.0 / k.half_life)
    res = []
    all_h = all_p = all_s = 0
    for r in reqs:
        p = policy or r["policy"]
        if p not in POLICIES:
            fail(f"request {r['name']}: policy {p}")
        if out:
            print(f"request {r['name']} policy {p}", file=out)
        per = []
        for l in range(n_layer):
            h = plans[l]
            if r["reset"]:
                while not h.is_static():
                    fl = []
                    h.plan_static(k.spare, fl)
                    if not fl:
                        fail(f"layer {l}: the static reset is stuck")
                    for s in fl:
                        h.publish(s)
            h.begin_request(p, r["resume"])
            for ids in r["prompt"][l]:
                h.count_prompt(ids)
            h.prompt_done(k)
            flight = []
            h.plan(k, flight)
            lh = lp = 0
            ls = len(flight)
            for ids in r["decode"][l]:
                lh += sum(1 for x in ids if h.slot_of[x] >= 0)
                lp += len(ids)
                h.count_token(ids, decay)
                for s in flight:
                    h.publish(s)
                flight = []
                h.plan(k, flight)
                ls += len(flight)
            for s in flight:
                h.publish(s)
            if out:
                print(f"layer {l} hits {lh} of {lp} swaps {ls}", file=out)
            per.append((lh, lp, ls))
        th, tp, ts = (sum(x[i] for x in per) for i in range(3))
        if out:
            print(f"request {r['name']} total hits {th} of {tp} swaps {ts}", file=out)
        all_h, all_p, all_s = all_h + th, all_p + tp, all_s + ts
        res.append(per)
    if out:
        print(f"total hits {all_h} of {all_p} swaps {all_s}", file=out)
    return res


def compare(n_layer, n_expert, reqs, hot, k, out=sys.stdout):
    """Each policy on every request: the hit rate per layer and overall; returns {policy: overall}."""
    rates, layers = {}, {}
    for p in POLICIES:
        res = replay(n_layer, n_expert, reqs, hot, k, policy=p)
        lh = [sum(r[l][0] for r in res) for l in range(n_layer)]
        lp = [sum(r[l][1] for r in res) for l in range(n_layer)]
        rates[p] = sum(lh) / max(sum(lp), 1)
        layers[p] = [h / p_ if p_ else 0.0 for h, p_ in zip(lh, lp)]
    print(f"hot_policy_sim.py: {len(reqs)} requests, {n_layer} layers, knobs: {k.swaps} swaps per layer per token, "
          f"{k.spare} spare slots, half-life {k.half_life}, margin {k.mul} x + {k.add}"
          f"{', pinned mapping' if k.mapping_pinned else ''}", file=out)
    print("layer  " + "  ".join(f"{p:>8}" for p in POLICIES), file=out)
    for l in range(n_layer):
        print(f"{l:5d}  " + "  ".join(f"{layers[p][l]:8.4f}" for p in POLICIES), file=out)
    print("all    " + "  ".join(f"{rates[p]:8.4f}" for p in POLICIES), file=out)
    return rates


class Skewed:
    """The synthetic traces' routing: per layer a static profile (Zipf over a random order: the hot set's counts) and
    topics (Zipf over the profile's order with half of it reshuffled), each token's K experts drawn from a topic."""

    def __init__(self, layers, experts, used, slab, seed):
        self.rng = random.Random(seed)
        self.layers, self.experts, self.used = layers, experts, used
        self.base = [self.rng.sample(range(experts), experts) for _ in range(layers)]
        self.hot = {"format": "strata-moe-hot-set/1", "budget_gib": 1, "ram_budget_gib": 1, "layers": []}
        for l in range(layers):
            self.hot["layers"].append({"layer": l, "n_expert": experts, "n_embd": 64, "n_ff": 32,
                                       "types": {"up": "q2_K", "gate": "q2_K", "down": "q3_K"},
                                       "experts": self.base[l][:slab], "warm": self.base[l][slab:]})

    def zipf(self, order):
        w = [0.0] * self.experts
        for i, e in enumerate(order):
            w[e] = 1.0 / (i + 1) ** 1.1
        return w

    def topic(self):
        out = []
        for l in range(self.layers):
            o = list(self.base[l])
            cut = self.rng.sample(range(self.experts), self.experts // 2)   # half the order reshuffled per topic
            vals = [o[i] for i in cut]
            self.rng.shuffle(vals)
            for i, v in zip(cut, vals):
                o[i] = v
            out.append(self.zipf(o))
        return out

    def draw(self, w):
        picked = set()
        while len(picked) < self.used:
            picked.add(self.rng.choices(range(self.experts), weights=w)[0])
        return sorted(picked, key=lambda e: -w[e])

    def lines(self, topic, prompt, decode):
        """A request's P lines (prompt tokens per layer) and D lines (decode tokens), drawn from topic."""
        out = []
        for l in range(self.layers):
            ids = [x for _ in range(prompt) for x in self.draw(topic[l])]
            out.append(f"P {l} {prompt} {' '.join(map(str, ids))}\n")
        for _ in range(decode):
            for l in range(self.layers):
                out.append(f"D {l} {' '.join(map(str, self.draw(topic[l])))}\n")
        return out

    def header(self):
        return f"# dsv41-route-trace/1 n_layer {self.layers} n_expert {self.experts} n_used {self.used}\n"

    def write_hot_set(self, out_dir):
        path = os.path.join(out_dir, "hot-set.json")
        with open(path, "w") as f:
            json.dump(self.hot, f)
        return path


def synthetic(out_dir, layers=4, experts=64, used=6, slab=16, requests=3, prompt=256, decode=480, seed=1):
    """A skewed trace: per request a topic of its own, its prompt and decode drawn from it, so a request's own
    routing predicts its decode better than the profile does."""
    g = Skewed(layers, experts, used, slab, seed)
    os.makedirs(out_dir, exist_ok=True)
    hot = g.write_hot_set(out_dir)
    path = os.path.join(out_dir, "trace.txt")
    with open(path, "w") as f:
        f.write(g.header())
        for r in range(requests):
            f.write(f"R {r + 1} adaptive reset=1\n")
            f.writelines(g.lines(g.topic(), prompt, decode))
    return path, hot


def two_turn(out_dir, layers=4, experts=384, used=6, slab=119, prompt=1024, decode=256, suffix=50, seed=2):
    """One conversation, two turns on one topic (the real model's 384 experts, 6 used, ~119 in each layer's slab:
    the A100 gate's 4756 over 40 layers): turn 1 from the hot set's slab, a long prompt and its decode; turn 2 a short
    suffix and its decode. Written twice, with the same ids: turn 2 resumed (trace-resume.txt, resume=1, as the
    engine's serve loop marks a RESUME) and fresh (trace-fresh.txt, resume=0: the planner before the resume rule)."""
    g = Skewed(layers, experts, used, slab, seed)
    os.makedirs(out_dir, exist_ok=True)
    hot = g.write_hot_set(out_dir)
    t = g.topic()
    turn1, turn2 = g.lines(t, prompt, decode), g.lines(t, suffix, decode)
    paths = []
    for name, resume in (("trace-resume.txt", 1), ("trace-fresh.txt", 0)):
        paths.append(os.path.join(out_dir, name))
        with open(paths[-1], "w") as f:
            f.write(g.header())
            f.write("R 1 adaptive reset=1 resume=0\n")
            f.writelines(turn1)
            f.write(f"R 2 adaptive reset=0 resume={resume}\n")
            f.writelines(turn2)
    return paths[0], paths[1], hot


def continued(trace, hot, k):
    """Turn 2's (hits, pairs, swaps) when no second request starts: turn 1's decode goes on over the suffix's tokens
    (as decode tokens) and then turn 2's decode; the swaps from the suffix's first token on."""
    n_layer, n_expert, n_used, reqs = read_trace(trace)
    r1, r2 = reqs
    plans = plans_for(n_layer, n_expert, hot, k)
    decay = 2.0 ** (-1.0 / k.half_life)
    th = tp = ts = 0
    for l in range(n_layer):
        h = plans[l]
        h.begin_request(ADAPTIVE, False)
        for ids in r1["prompt"][l]:
            h.count_prompt(ids)
        h.prompt_done(k)
        suffix = [ids[i:i + n_used] for ids in r2["prompt"][l] for i in range(0, len(ids), n_used)]
        flight = []
        h.plan(k, flight)
        steps = [(x, 0) for x in r1["decode"][l]] + [(x, 1) for x in suffix] + [(x, 2) for x in r2["decode"][l]]
        for ids, part in steps:
            if part == 2:
                th += sum(1 for x in ids if h.slot_of[x] >= 0)
                tp += len(ids)
            h.count_token(ids, decay)
            for s in flight:
                h.publish(s)
            flight = []
            h.plan(k, flight)
            if part:
                ts += len(flight)
    return th, tp, ts


def two_turn_verdict(resume, fresh, cont, out=sys.stdout):
    """Turn 2's (hits, pairs, swaps) resumed, fresh and continued: the resumed turn hits at least as often as the fresh
    one (the planner before the resume rule) and swaps at most a quarter as much ("far below" the fresh one); it
    follows the continued run but not swap for swap, so it may trail it by 1% of the pairs at most (the same rule as
    dsv41_engine_hot_policy_test's two-turn case)."""
    (rh, rp, rs), (fh, fp, fs), (ch, cp, cs) = resume, fresh, cont
    ok = rh >= fh and 100 * rh >= 100 * ch - rp and fs > 0 and 4 * rs <= fs
    print(f"hot_policy_sim.py: two turns, turn 2: resumed hit rate {rh / max(rp, 1):.4f} with {rs} swaps, fresh "
          f"{fh / max(fp, 1):.4f} with {fs} swaps, continued {ch / max(cp, 1):.4f} with {cs} swaps: resumed hits >= "
          f"fresh and within 1% of continued, swaps <= a quarter of fresh: {'PASS' if ok else 'FAIL'}", file=out)
    return ok


def stream_of(reqs, spec, n_layer, n_used):
    """A batch row's routing, one token a step, per layer: spec REQ (its decode tokens) or REQ/pN (its prompt's tokens
    from token N on: a prompt's tokens stand in for another conversation's decode)."""
    name, _, part = spec.partition("/")
    r = next((x for x in reqs if x["name"] == name), None)
    if r is None:
        fail(f"batch: no request {name} in the trace")
    if not part:
        return [list(r["decode"][l]) for l in range(n_layer)]
    if not part.startswith("p") or not part[1:].isdigit():
        fail(f"batch: row {spec}: REQ or REQ/pN")
    first = int(part[1:])
    out = []
    for l in range(n_layer):
        flat = [x for ids in r["prompt"][l] for x in ids]
        out.append([flat[i:i + n_used] for i in range(first * n_used, len(flat) - n_used + 1, n_used)])
    return out


READ_RULES = ("target", "none", "union")


def batch(n_layer, n_expert, n_used, rows, reads, steps, hot, k, rule):
    """Batch slots decoding one token each a step, as the engine's
    decode_batch does (experts.cpp decode_serve: the step's pairs of every row counted at once with one decay, the
    misses grouped by expert into one CPU job each), from the hot set's slab as the deterministic start leaves it.
    rows: (first step, routing per layer) each, a row decoding from its first step on; reads: (step, prompt ids per
    layer) each, another request's prompt read beside the slots before that step. How a read beside moves the
    planners: "target" (the first batched engine): its counts join the prompt counts, which then replace the batch's (prompt_done:
    the summed prompts' target first, then their counts as the LFU's); "none": it moves nothing; "union" (the engine's,
    HotLayerPlan::merge_prompt): its counts, scaled to the mass of one row's decode counts at equilibrium
    (K / (1 - 2^(-1 / half_life)) a layer), are added to the batch's, no target. Returns per step and layer (hits,
    pairs, distinct experts, distinct missed experts, swaps started)."""
    plans = plans_for(n_layer, n_expert, hot, k)
    decay = 2.0 ** (-1.0 / k.half_life)
    mass = n_used / (1.0 - decay)
    out = [[None] * n_layer for _ in range(steps)]
    for l in range(n_layer):
        h = plans[l]
        h.begin_request(ADAPTIVE, False)
        h.prompt_done(k)
        flight = []
        h.plan(k, flight)
        for t in range(steps):
            for at, read in reads:
                if at != t or rule == "none":
                    continue
                for ids in read[l]:
                    h.count_prompt(ids)
                if rule == "target":
                    for s in flight:
                        h.publish(s)
                    flight = []
                    h.prompt_done(k)
                    h.plan(k, flight)
                else:
                    h.merge_prompt(mass)
            ids = [x for first, r in rows if t >= first for x in r[l][t - first]]
            hits = sum(1 for x in ids if h.slot_of[x] >= 0)
            distinct = set(ids)
            missed = {x for x in distinct if h.slot_of[x] < 0}
            h.count_token(ids, decay)
            for s in flight:
                h.publish(s)
            flight = []
            h.plan(k, flight)
            out[t][l] = (hits, len(ids), len(distinct), len(missed), len(flight))
    return out


def batch_report(res, cuts, label, out=sys.stdout):
    """The batch's steps in windows between the cuts: the hit rate and per step the distinct experts, the CPU jobs
    (distinct missed experts) and the swaps, summed over the layers."""
    for a, b in zip(cuts, cuts[1:]):
        sel = res[a:b]
        if not sel:
            continue
        n = len(sel)
        tot = [sum(x[l][i] for x in sel for l in range(len(x))) for i in range(5)]
        print(f"hot_policy_sim.py: batch {label}: steps {a}-{b}: hit rate {tot[0] / max(tot[1], 1):.4f}, per step "
              f"{tot[2] / n:.1f} distinct experts, {tot[3] / n:.1f} CPU jobs, {tot[4] / n:.1f} swaps", file=out)


def oracle(dump, out_path):
    """A fork op dump's ffn_moe_topk (tools/dsv41/oracle's dump: topk-L<l>-u<u>.bin, [T][K] i32) as a trace."""
    with open(os.path.join(dump, "dump-manifest.json")) as f:
        m = json.load(f)
    topk = [x for x in m["files"] if x["kind"] == "topk"]
    if not topk:
        fail(f"{dump}: no topk files")
    n_layer = max(x["layer"] for x in topk) + 1
    k = topk[0]["ne"][0]
    by = {(x["layer"], x["ubatch"]): x for x in topk}
    ubatches = sorted({x["ubatch"] for x in topk})
    with open(out_path, "w") as f:
        f.write(f"# dsv41-route-trace/1 n_layer {n_layer} n_expert 384 n_used {k}\n")
        f.write("R 1 adaptive reset=0\n")
        for u in ubatches:
            for l in range(n_layer):
                x = by[(l, u)]
                T = x["ne"][1]
                with open(os.path.join(dump, x["file"]), "rb") as g:
                    ids = struct.unpack(f"<{T * k}i", g.read())
                if u == ubatches[0]:
                    f.write(f"P {l} {T} {' '.join(map(str, ids))}\n")
                else:
                    for t in range(T):
                        f.write(f"D {l} {' '.join(map(str, ids[t * k:(t + 1) * k]))}\n")
    print(f"hot_policy_sim.py: {out_path}: {n_layer} layers, ubatch {ubatches[0]} as the prompt, "
          f"{len(ubatches) - 1} decode steps")


def parse_knobs(args):
    k, rest, i = Knobs(), [], 0
    while i < len(args):
        a = args[i]
        if a in ("--swaps", "--spare", "--half-life", "--margin", "--hot-set", "--policy") and i + 1 < len(args):
            v = args[i + 1]
            if a == "--swaps":
                k.swaps = int(v)
            elif a == "--spare":
                k.spare = int(v)
            elif a == "--half-life":
                k.half_life = float(v)
            elif a == "--margin":
                mul, _, add = v.partition(":")
                k.mul, k.add = float(mul), float(add)
            else:
                rest += [a, v]
            i += 2
            continue
        if a == "--mapping-pinned":
            k.mapping_pinned = True
        else:
            rest.append(a)
        i += 1
    if k.spare < 0 or (k.spare > 0 and k.swaps < 1) or not k.half_life > 0 or not k.mul >= 0:
        fail("knobs: --spare >= 0, --swaps >= 1, --half-life > 0, --margin MUL >= 0")
    return k, rest


def opt(rest, name):
    if name in rest:
        i = rest.index(name)
        if i + 1 >= len(rest):
            fail(f"{name} needs a value")
        v = rest[i + 1]
        del rest[i:i + 2]
        return v
    return None


def turn2(trace, hot, k):
    """The second request's (hits, pairs, swaps) of a two-turn trace."""
    n_layer, n_expert, _, reqs = read_trace(trace)
    per = replay(n_layer, n_expert, reqs, hot, k)[1]
    return tuple(sum(x[i] for x in per) for i in range(3))


def self_test():
    d = tempfile.mkdtemp(prefix="hot-policy-sim-")
    trace, hot = synthetic(d)
    n_layer, n_expert, _, reqs = read_trace(trace)
    rates = compare(n_layer, n_expert, reqs, read_hot_set(hot), Knobs())
    ok = rates[PROMPT] > rates[STATIC] and rates[ADAPTIVE] > rates[STATIC]
    print(f"hot_policy_sim.py: self-test (synthetic skewed trace): static {rates[STATIC]:.4f}, prompt "
          f"{rates[PROMPT]:.4f}, adaptive {rates[ADAPTIVE]:.4f}: prompt and adaptive above static: "
          f"{'PASS' if ok else 'FAIL'}")
    resume, fresh, hot2 = two_turn(os.path.join(d, "two-turn"))
    hs, k = read_hot_set(hot2), Knobs()
    ok2 = two_turn_verdict(turn2(resume, hs, k), turn2(fresh, hs, k), continued(resume, hs, k))
    return 0 if ok and ok2 else 1


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 2
    if argv[0] == "--self-test":
        return self_test()
    cmd, k_rest = argv[0], argv[1:]
    if cmd in ("replay", "compare"):
        k, rest = parse_knobs(k_rest)
        hot_path, policy = opt(rest, "--hot-set"), opt(rest, "--policy")
        if len(rest) != 1 or not hot_path:
            fail(f"{cmd} TRACE --hot-set FILE")
        n_layer, n_expert, _, reqs = read_trace(rest[0])
        hot = read_hot_set(hot_path)
        if cmd == "replay":
            if policy and policy not in POLICIES:
                fail(f"--policy {policy}")
            replay(n_layer, n_expert, reqs, hot, k, policy=policy, out=sys.stdout)
        else:
            compare(n_layer, n_expert, reqs, hot, k)
        return 0
    if cmd == "synthetic":
        if not k_rest:
            fail("synthetic OUT_DIR")
        rest = k_rest[1:]
        kw = {}
        for name in ("layers", "experts", "used", "slab", "requests", "prompt", "decode", "seed"):
            v = opt(rest, "--" + name)
            if v is not None:
                kw[name] = int(v)
        if rest:
            fail(f"synthetic: unknown {rest}")
        trace, hot = synthetic(k_rest[0], **kw)
        print(f"hot_policy_sim.py: {trace} and {hot}")
        return 0
    if cmd == "two-turn":
        if not k_rest:
            fail("two-turn OUT_DIR")
        rest = k_rest[1:]
        kw = {}
        for name in ("experts", "slab", "prompt", "decode", "suffix", "seed"):
            v = opt(rest, "--" + name)
            if v is not None:
                kw[name] = int(v)
        if rest:
            fail(f"two-turn: unknown {rest}")
        resume, fresh, hot = two_turn(k_rest[0], **kw)
        print(f"hot_policy_sim.py: {resume}, {fresh} and {hot}")
        hs, k = read_hot_set(hot), Knobs()
        return 0 if two_turn_verdict(turn2(resume, hs, k), turn2(fresh, hs, k), continued(resume, hs, k)) else 1
    if cmd == "batch":
        k, rest = parse_knobs(k_rest)
        hot_path, rows_spec, reads_spec = opt(rest, "--hot-set"), opt(rest, "--rows"), opt(rest, "--reads")
        steps, cuts = opt(rest, "--steps"), opt(rest, "--cuts")
        if len(rest) != 1 or not hot_path or not rows_spec:
            fail("batch TRACE --hot-set FILE --rows SPEC[@S],... [--reads TRACE2:REQ@S,...] [--steps N] [--cuts A,B,...]")
        n_layer, n_expert, n_used, reqs = read_trace(rest[0])
        rows = []
        for spec in rows_spec.split(","):
            spec, _, first = spec.partition("@")
            rows.append((int(first or 0), stream_of(reqs, spec, n_layer, n_used)))
        longest = min(first + len(r[l]) for first, r in rows for l in range(n_layer))
        steps = min(int(steps), longest) if steps else longest
        reads = []
        for spec in (reads_spec.split(",") if reads_spec else []):
            spec, _, at = spec.partition("@")
            path, _, name = spec.rpartition(":")
            n2, e2, u2, reqs2 = read_trace(path)
            if (n2, e2, u2) != (n_layer, n_expert, n_used):
                fail(f"batch: {path}'s shape differs from {rest[0]}'s")
            r2 = next((x for x in reqs2 if x["name"] == name), None)
            if r2 is None:
                fail(f"batch: no request {name} in {path}")
            reads.append((int(at or 0), r2["prompt"]))
        cut = sorted({0, steps, *(int(x) for x in cuts.split(","))} if cuts else {0, steps})
        hot = read_hot_set(hot_path)
        print(f"hot_policy_sim.py: batch of {len(rows)} rows ({rows_spec}), {steps} steps, {len(reads)} reads beside "
              f"them ({reads_spec or 'none'}); knobs: {k.swaps} swaps per layer per step, {k.spare} spare slots, "
              f"half-life {k.half_life}, margin {k.mul} x + {k.add}")
        for rule in (READ_RULES if reads else ("none",)):
            batch_report(batch(n_layer, n_expert, n_used, rows, reads, steps, hot, k, rule), cut, f"read rule {rule}")
        return 0
    if cmd == "oracle" and len(k_rest) == 2:
        oracle(k_rest[0], k_rest[1])
        return 0
    fail(f"unknown command {cmd} (--help)")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
