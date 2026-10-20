#!/usr/bin/env python3
"""tools/dsv41/quality/det_sampling_check.py - the determinism (item 7) and sampling (item 8) checks over HTTP.

Runs against serve/server.py on dsv41-engine (the deepseek-v4.1 profile) right after the server's load, one
non-streaming /v1/chat/completions request at a time, every reply's token ids read from the engine's --log-ids lines
in its log (each request's "generated_token_ids" line, written before its DONE, so it is there when the reply is) and
its decode from the request's "ms per step" line:

  item 7  prompt A greedy (temperature 0, max_tokens 256) twice, the first the server's first request after the
          load (cold), then prompt B greedy, then A again: the three A replies must have the same token ids; on a
          mismatch the first divergent position of each against the first reply, with both tokens, is recorded.
  item 8  A at temperature 0.7, top_p 0.95, top_k 40, seed 1234 twice: the same ids; seed 1235: other ids; the
          engine's line names that sampler and seed; the first 1234 reply's text (max_tokens 256) for the hand check;
          the speed: prompt S (a count that runs past 256 tokens) greedy and at 0.7 (seed 1234), 3 runs each,
          interleaved, every run 256 tokens with no end of reply (finish_reason length; a run that ends early is a
          FAIL, not a shorter sample), the median decode ms per token of each (the engine's per-step figure): within
          5% of each other; and a request with frequency_penalty 0.5 answered 400 with a message naming the field.

Items 7 and 8 run in the engine's default mode, which must be its deterministic
one (each request's engine line names the mode: "deterministic"); a request whose line says otherwise leaves the item
unmeasured. Then the matrix row with the mode off (every request "strata_deterministic": false, its line "deterministic
off"): item 7's four requests and item 8's three seeded ones again, recorded, not part of the verdict (the measure of
what the mode fixes); and the mode's decode cost: prompt S greedy with the mode on and off, 3 runs each, interleaved,
256 tokens each, the median ms per token of each and on / off (over 1.20 it is flagged for review). Every divergence record carries both requests' placement from their engine lines: hits and misses
of the decode's routed pairs, swaps, batches, deferred boundaries and the boundaries that waited. Every request also
records its time to first token (the engine line's "first token at", from session.run's start, so without the hot
set's reset) and the reset before it (its hot set line's "reset before it ... in X ms"); the cost row gives both with
the mode on and off and their sum, the first token the mode's reset delays (0.6-2.0 s measured on
the A100 and the RTX PRO 6000 with hot_reset=1). With several cached conversations the mode resets after
a request, in the idle time, so a request's start resets only after one in the mode off; the hot set line's "idle
reset after the last request ... waited W ms" is the part of the last request's reset this one waited for, and the
reset before its first token is the sum of the two (reset_ms; reset_before_ms and idle_wait_ms apart). In the cost
row's interleaved runs each run in the mode off follows one in the mode and waits for its idle reset, and each run in
the mode follows one off and resets at its start.

    det_sampling_check.py --url http://127.0.0.1:8090 --engine-log out/quality-engine.log --out out/det-sampling.json

The JSON has every request (label, sampling, status, finish, ids, ms per step), each item's verdict and its parts.
Exit 0 when items 7 and 8 both pass, 1 when either fails or a request could not be measured (the reason printed).
section(path) is the markdown e07_report.py puts on the qdet sheet.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MAX_TOKENS = 256
SPEED_RUNS = 3
SPEED_TOLERANCE = 0.05
PROMPT_A = ("Explain how a hash table works: hashing, buckets, collisions and resizing, in about 300 words, then give "
            "a short Python example.")
PROMPT_B = "Name three primary colors, one per line."
PROMPT_S = "Count from 1 to 500, writing every number in words, separated by commas, with nothing else."
SAMPLED = {"temperature": 0.7, "top_p": 0.95, "top_k": 40}
PENALTY = {"temperature": 0.7, "frequency_penalty": 0.5}
IDS_RE = re.compile(r"^dsv41-engine serve: request (\d+): generated_token_ids: ?(.*)$", re.M)
LINE_RE = re.compile(r"^dsv41-engine serve: request (\d+): \d+ prompt tokens, .*; (greedy|sampled: .*)$", re.M)
MODE_RE = re.compile(r"^dsv41-engine serve: request (\d+): \d+ prompt tokens, .*; (deterministic|deterministic off); "
                     r"(?:greedy|sampled: .*)$", re.M)
PAIRS_RE = re.compile(r"^dsv41-engine serve: request (\d+): decode experts: \d+ steps, (\d+) hits and (\d+) misses", re.M)
HOT_RE = re.compile(r"^dsv41-engine serve: request (\d+): decode hot set \w+[^:]*: (\d+) swaps \(.*?, (\d+) batches, "
                    r"(\d+) deferred\).*?; (?:deterministic: (\d+) boundaries waited|deterministic off)", re.M)
OFF = {"strata_deterministic": False}
COST_FLAG = 1.20
FIRST_RE = re.compile(r"^dsv41-engine serve: request (\d+): \d+ prompt tokens, [^;]*, first token at ([0-9.]+) ms;",
                      re.M)
RESET_RE = re.compile(r"^dsv41-engine serve: request (\d+): decode hot set .*?; reset before it (\d+) swaps "
                      r"\(([0-9.]+) MiB\) in ([0-9.]+) ms;", re.M)
IDLE_RE = re.compile(r"^dsv41-engine serve: request (\d+): decode hot set .*; idle reset after the last request (\d+) "
                     r"swaps \(([0-9.]+) MiB\) in ([0-9.]+) ms, waited ([0-9.]+) ms$", re.M)
STEP_RE = re.compile(r"^dsv41-engine serve: request (\d+): decode experts: (\d+) steps, .*; ([0-9.]+) ms per step", re.M)


class Unmeasured(Exception):
    """A request the check cannot read (no answer, no log line): the item it belongs to is not measured."""


def post(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        with e:
            raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw}


def model_of(url: str) -> str:
    with urllib.request.urlopen(url + "/v1/models", timeout=60) as r:
        return json.loads(r.read().decode())["data"][0]["id"]


class Client:
    """One request at a time; each answered one matched to the engine's new request in the log."""

    def __init__(self, url: str, log: Path, timeout: float):
        self.url, self.log, self.timeout = url, log, timeout
        self.model = model_of(url)
        self.seen = self.last_request()
        self.requests: list[dict] = []

    def text(self) -> str:
        return self.log.read_text(errors="replace") if self.log.exists() else ""

    def last_request(self) -> int:
        return max((int(n) for n, _ in IDS_RE.findall(self.text())), default=0)

    def ask(self, label: str, prompt: str, sampling: dict) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
                **sampling}
        t0 = time.time()
        status, j = post(self.url, body, self.timeout)
        r = {"label": label, "sampling": sampling, "status": status, "seconds": round(time.time() - t0, 2)}
        self.requests.append(r)
        if status != 200:
            r["error"] = (j.get("error") or {}).get("message") or j.get("raw", "")
            return r
        c = j["choices"][0]
        r.update(finish=c["finish_reason"], completion_tokens=j["usage"]["completion_tokens"],
                 text=c["message"].get("content") or "")
        log = self.text()
        new = [(int(n), x) for n, x in IDS_RE.findall(log) if int(n) > self.seen]
        if len(new) != 1:
            raise Unmeasured(f"{label}: {len(new)} new generated_token_ids lines in {self.log} (one expected: the "
                             "engine runs with --log-ids?)")
        n = self.seen = new[0][0]
        r["request"], r["ids"] = n, [int(t) for t in new[0][1].split()]
        r["engine_sampling"] = next((x for m, x in LINE_RE.findall(log) if int(m) == n), None)
        step = next(((int(s), float(ms)) for m, s, ms in STEP_RE.findall(log) if int(m) == n), None)
        r["steps"], r["ms_per_step"] = step if step else (None, None)
        r["mode"] = next((x for m, x in MODE_RE.findall(log) if int(m) == n), None)
        pl = {}
        for m, hits, misses in PAIRS_RE.findall(log):
            if int(m) == n:
                pl.update(hits=int(hits), misses=int(misses))
        for m, swaps, batches, deferred, waited in HOT_RE.findall(log):
            if int(m) == n:
                pl.update(swaps=int(swaps), batches=int(batches), deferred=int(deferred),
                          waited=int(waited) if waited else None)
        r["placement"] = pl or None
        r["first_token_ms"] = next((float(ms) for m, ms in FIRST_RE.findall(log) if int(m) == n), None)
        reset = next(((int(sw), float(ms)) for m, sw, _, ms in RESET_RE.findall(log) if int(m) == n), None)
        r["reset_swaps"], r["reset_before_ms"] = reset if reset else (None, None)
        # the last request's idle reset and the part of it this one waited for
        idle = next(((int(sw), float(ms), float(w)) for m, sw, _, ms, w in IDLE_RE.findall(log) if int(m) == n), None)
        r["idle_reset_swaps"], r["idle_reset_ms"], r["idle_wait_ms"] = idle if idle else (None, None, None)
        r["reset_ms"] = (None if reset is None or idle is None else round(r["reset_before_ms"] + r["idle_wait_ms"], 3))
        return r


def first_divergence(a: list[int], b: list[int]) -> dict | None:
    """The first position where b leaves a (None: equal); a token past a list's end is None."""
    for i in range(max(len(a), len(b))):
        x, y = (a[i] if i < len(a) else None), (b[i] if i < len(b) else None)
        if x != y:
            return {"position": i, "first": x, "other": y}
    return None


def check_mode(item: str, rs: list[dict], want: str) -> None:
    """Every request of an item ran in the mode it asked for, by its engine line."""
    bad = [f"{r['label']} ({r.get('mode')})" for r in rs if r.get("mode") != want]
    if bad:
        raise Unmeasured(f"{item}: the engine's line does not say {want!r} for {', '.join(bad)} (an engine without the deterministic mode?)")


def divergence(first: dict, r: dict) -> dict | None:
    """first_divergence of r against first, with the reply's label and both requests' placement."""
    d = first_divergence(first["ids"], r["ids"])
    if d is None:
        return None
    return dict(d, reply=r["label"], request=r["request"], placement=r.get("placement"),
                first_placement=first.get("placement"))


def item7(c: Client, extra: dict | None = None, tag: str = "") -> dict:
    greedy = {"temperature": 0, **(extra or {})}
    a = [c.ask(f"A greedy{tag}, cold (the first request after the load)" if not tag else f"A greedy{tag}, first",
               PROMPT_A, greedy),
         c.ask(f"A greedy{tag}, again", PROMPT_A, greedy)]
    b = c.ask(f"B greedy{tag}", PROMPT_B, greedy)
    a.append(c.ask(f"A greedy{tag}, after B", PROMPT_A, greedy))
    bad = [r["label"] for r in a if r["status"] != 200]
    if bad:
        raise Unmeasured(f"item 7{tag}: {', '.join(bad)} not answered with 200")
    check_mode(f"item 7{tag}", a + [b], "deterministic off" if extra else "deterministic")
    div = [d for d in (divergence(a[0], r) for r in a[1:]) if d]
    return {"pass": not div, "tokens": [len(r["ids"]) for r in a], "divergence": div,
            "requests": [r["request"] for r in a]}


def seeds(c: Client, extra: dict | None = None, tag: str = "") -> tuple[dict, dict]:
    s1 = c.ask(f"A at 0.7, seed 1234{tag}", PROMPT_A, dict(SAMPLED, seed=1234, **(extra or {})))
    s2 = c.ask(f"A at 0.7, seed 1234 again{tag}", PROMPT_A, dict(SAMPLED, seed=1234, **(extra or {})))
    s3 = c.ask(f"A at 0.7, seed 1235{tag}", PROMPT_A, dict(SAMPLED, seed=1235, **(extra or {})))
    bad = [r["label"] for r in (s1, s2, s3) if r["status"] != 200]
    if bad:
        raise Unmeasured(f"item 8{tag}: {', '.join(bad)} not answered with 200")
    check_mode(f"item 8{tag}", [s1, s2, s3], "deterministic off" if extra else "deterministic")
    want = "sampled: temperature 0.7, top_k 40, top_p 0.95, min_p 0, seed {} (given)"
    named = s1["engine_sampling"] == want.format(1234) and s3["engine_sampling"] == want.format(1235)
    same = divergence(s1, s2)
    return {"pass": same is None and s1["ids"] != s3["ids"] and named, "same_seed_identical": same is None,
            "same_seed_divergence": same, "other_seed_differs": s1["ids"] != s3["ids"],
            "engine_names_sampler": named, "engine_sampling": s1["engine_sampling"]}, s1


def item8(c: Client) -> dict:
    seeds_, s1 = seeds(c)
    runs = {"greedy": [], "sampled": []}
    for i in range(SPEED_RUNS):
        for kind, sampling in (("greedy", {"temperature": 0}), ("sampled", dict(SAMPLED, seed=1234))):
            r = c.ask(f"S {kind}, run {i + 1}", PROMPT_S, sampling)
            if r["status"] != 200 or r.get("ms_per_step") is None:
                raise Unmeasured(f"item 8 speed: {r['label']}: status {r['status']}, no decode line")
            runs[kind].append(r)
    short = [f"{r['label']} ended at {r['completion_tokens']} tokens ({r['finish']})"
             for k in runs for r in runs[k] if r["finish"] != "length" or r["completion_tokens"] != MAX_TOKENS]
    g = statistics.median(r["ms_per_step"] for r in runs["greedy"])
    sm = statistics.median(r["ms_per_step"] for r in runs["sampled"])
    ratio = sm / g if g > 0 else None
    speed = {"pass": not short and ratio is not None and abs(ratio - 1) <= SPEED_TOLERANCE,
             "greedy_ms_per_token": [r["ms_per_step"] for r in runs["greedy"]],
             "sampled_ms_per_token": [r["ms_per_step"] for r in runs["sampled"]],
             "greedy_median": g, "sampled_median": sm, "ratio": ratio, "not_256": short}
    p = c.ask("A with frequency_penalty 0.5", PROMPT_A, PENALTY)
    penalty = {"pass": p["status"] == 400 and "frequency_penalty" in (p.get("error") or ""), "status": p["status"],
               "message": p.get("error")}
    return {"pass": seeds_["pass"] and speed["pass"] and penalty["pass"], "seeds": seeds_, "speed": speed,
            "penalty": penalty, "text_256": {"text": s1["text"], "tokens": s1["completion_tokens"],
                                              "finish": s1["finish"], "request": s1["request"]}}


def mode_off(c: Client) -> dict:
    """The matrix row: items 7 and 8's seeds with the mode off, recorded (not part of the verdict)."""
    out = {}
    for name, fn in (("7", lambda: item7(c, OFF, " (mode off)")), ("8_seeds", lambda: seeds(c, OFF, " (mode off)")[0])):
        try:
            out[name] = fn()
        except Unmeasured as e:
            out[name] = {"pass": False, "unmeasured": str(e)}
    return out


def cost(c: Client) -> dict:
    """The mode's decode cost, prompt S greedy on and off, SPEED_RUNS each, interleaved: median ms per token;
    and the first token of each, the engine's "first token at" and the reset before it."""
    runs = {"on": [], "off": []}
    for i in range(SPEED_RUNS):
        for kind, extra in (("on", {}), ("off", OFF)):
            r = c.ask(f"S greedy, mode {kind}, run {i + 1}", PROMPT_S, {"temperature": 0, **extra})
            if r["status"] != 200 or r.get("ms_per_step") is None:
                raise Unmeasured(f"cost: {r['label']}: status {r['status']}, no decode line")
            runs[kind].append(r)
    check_mode("cost (mode on)", runs["on"], "deterministic")
    check_mode("cost (mode off)", runs["off"], "deterministic off")
    short = [f"{r['label']} ended at {r['completion_tokens']} tokens ({r['finish']})"
             for k in runs for r in runs[k] if r["finish"] != "length" or r["completion_tokens"] != MAX_TOKENS]
    on = statistics.median(r["ms_per_step"] for r in runs["on"])
    off = statistics.median(r["ms_per_step"] for r in runs["off"])
    ratio = on / off if off > 0 else None
    waits = [(r.get("placement") or {}).get("waited") for r in runs["on"]]
    first = {}
    for kind in runs:
        if any(r.get("first_token_ms") is None or r.get("reset_ms") is None for r in runs[kind]):
            first = {"unmeasured": f"mode {kind}: a request without its \"first token at\", \"reset before it\" or "
                                   "\"idle reset after the last request\" (an engine without the idle reset?)"}
            break
        ft, rs = [r["first_token_ms"] for r in runs[kind]], [r["reset_ms"] for r in runs[kind]]
        first[kind] = {"first_token_ms": ft, "reset_ms": rs, "first_token_median": statistics.median(ft),
                       "reset_median": statistics.median(rs),
                       "reset_before_median": statistics.median(r["reset_before_ms"] for r in runs[kind]),
                       "idle_wait_median": statistics.median(r["idle_wait_ms"] for r in runs[kind]),
                       "with_reset_median": statistics.median(f + x for f, x in zip(ft, rs))}
    return {"measured": not short and ratio is not None, "on_ms_per_token": [r["ms_per_step"] for r in runs["on"]],
            "off_ms_per_token": [r["ms_per_step"] for r in runs["off"]], "on_median": on, "off_median": off,
            "ratio": ratio, "over_flag": ratio is not None and ratio > COST_FLAG, "on_waited": waits, "not_256": short,
            "first_token": first}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--url", required=True, help="the server, e.g. http://127.0.0.1:8090")
    ap.add_argument("--engine-log", required=True, type=Path, help="the engine's log (the config's \"log\")")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=float, default=600, help="seconds per request")
    a = ap.parse_args()
    out = {"prompts": {"A": PROMPT_A, "B": PROMPT_B, "S": PROMPT_S}, "max_tokens": MAX_TOKENS,
           "speed_runs": SPEED_RUNS, "speed_tolerance": SPEED_TOLERANCE, "cost_flag": COST_FLAG, "items": {}}
    c = None
    try:
        c = Client(a.url, a.engine_log, a.timeout)
        for name, fn in (("7", item7), ("8", item8)):
            try:
                out["items"][name] = fn(c)
            except Unmeasured as e:
                out["items"][name] = {"pass": False, "unmeasured": str(e)}
        out["mode_off"] = mode_off(c)
        try:
            out["cost"] = cost(c)
        except Unmeasured as e:
            out["cost"] = {"measured": False, "unmeasured": str(e)}
    except (OSError, ValueError, KeyError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["requests"] = c.requests if c else []
    out["ok"] = all(out["items"].get(k, {}).get("pass") for k in ("7", "8"))
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for r in out["requests"]:
        print(f"request {r.get('request', '-')}: {r['label']}: {r['status']}, {r.get('completion_tokens', 0)} tokens "
              f"({r.get('finish', r.get('error', ''))}), {r.get('ms_per_step')} ms per step, first token at "
              f"{r.get('first_token_ms')} ms after a reset of {r.get('reset_ms')} ms, {r['seconds']} s")
    print("\n".join(verdict_lines(out)))
    return 0 if out["ok"] else 1


def verdict_lines(j: dict) -> list[str]:
    lines = []
    if j.get("error"):
        lines.append(f"det_sampling_check.py: FAIL: {j['error']}")
    i7, i8 = j["items"].get("7", {}), j["items"].get("8", {})
    if "unmeasured" in i7:
        lines.append(f"item 7: FAIL, not measured: {i7['unmeasured']}")
    elif i7:
        lines.append(f"item 7: {'PASS' if i7['pass'] else 'FAIL'}: the three A replies (greedy, {i7['tokens']} tokens) "
                     + ("have the same token ids" if i7["pass"] else "diverge: " + "; ".join(
                         f"{d['reply']} at position {d['position']} (token {d['other']}, the cold reply's {d['first']})"
                         for d in i7["divergence"])))
    if "unmeasured" in i8:
        lines.append(f"item 8: FAIL, not measured: {i8['unmeasured']}")
    elif i8:
        s, v, p = i8["seeds"], i8["speed"], i8["penalty"]
        lines.append(f"item 8: {'PASS' if i8['pass'] else 'FAIL'}")
        lines.append(f"  seeds: {'PASS' if s['pass'] else 'FAIL'}: seed 1234 twice "
                     f"{'identical' if s['same_seed_identical'] else 'different at ' + str(s['same_seed_divergence'])}, "
                     f"seed 1235 {'differs' if s['other_seed_differs'] else 'gives the same ids'}, the engine's line "
                     f"{'names' if s['engine_names_sampler'] else 'does not name'} the sampler and seed")
        ratio = f"{v['ratio']:.3f}" if v["ratio"] is not None else "n/a"
        lines.append(f"  speed: {'PASS' if v['pass'] else 'FAIL'}: median {v['sampled_median']:.2f} ms per token at "
                     f"0.7 against {v['greedy_median']:.2f} greedy, ratio {ratio} (within {SPEED_TOLERANCE:.0%})"
                     + (f"; {', '.join(v['not_256'])}" if v["not_256"] else ""))
        lines.append(f"  penalty: {'PASS' if p['pass'] else 'FAIL'}: frequency_penalty 0.5 answered {p['status']}: "
                     f"{p['message']}")
    off = j.get("mode_off") or {}
    o7, o8 = off.get("7") or {}, off.get("8_seeds") or {}
    if o7 or o8:
        lines.append("mode off (strata_deterministic false; recorded, not in the verdict):")
    if "unmeasured" in o7:
        lines.append(f"  item 7: not measured: {o7['unmeasured']}")
    elif o7:
        lines.append(f"  item 7: {'PASS' if o7['pass'] else 'FAIL'}: "
                     + ("the same token ids" if o7["pass"] else "; ".join(
                         f"{d['reply']} at position {d['position']}" for d in o7["divergence"])))
    if "unmeasured" in o8:
        lines.append(f"  item 8 seeds: not measured: {o8['unmeasured']}")
    elif o8:
        lines.append(f"  item 8 seeds: {'PASS' if o8['pass'] else 'FAIL'}: seed 1234 twice "
                     f"{'identical' if o8['same_seed_identical'] else 'different at ' + str(o8['same_seed_divergence']['position'])}")
    k = j.get("cost") or {}
    if "unmeasured" in k:
        lines.append(f"cost: not measured: {k['unmeasured']}")
    elif k:
        ratio = f"{k['ratio']:.3f}" if k["ratio"] is not None else "n/a"
        lines.append(f"cost: median {k['on_median']:.2f} ms per token with the mode on against {k['off_median']:.2f} off, "
                     f"ratio {ratio}" + (f" (over {j.get('cost_flag', COST_FLAG):.2f}: flagged for review)"
                                         if k["over_flag"] else "")
                     + (f"; {', '.join(k['not_256'])}" if k["not_256"] else ""))
        f = k.get("first_token") or {}
        if "unmeasured" in f:
            lines.append(f"cost: first token: not measured: {f['unmeasured']}")
        elif f:
            on, off = f["on"], f["off"]
            def split(x):
                return (f"{x['reset_median']:.1f} of reset: {x.get('reset_before_median', x['reset_median']):.1f} before it "
                        f"and {x.get('idle_wait_median', 0.0):.1f} waiting for the last request's idle reset; "
                        f"{x['first_token_median']:.1f} from the request's start")
            lines.append(f"cost: first token: median {on['with_reset_median']:.1f} ms with the mode on ({split(on)}) against "
                         f"{off['with_reset_median']:.1f} off ({split(off)})")
    lines.append(f"det_sampling_check.py: {'PASS' if j.get('ok') else 'FAIL'}")
    return lines


def placement_text(d: dict) -> str:
    """A divergence's placement record: the two requests' hits, misses, swaps, deferred and waited boundaries."""
    def one(p):
        if not p:
            return "no engine line"
        return (f"{p.get('hits')} hits, {p.get('misses')} misses, {p.get('swaps')} swaps in {p.get('batches')} "
                f"batches, {p.get('deferred')} deferred, {p.get('waited')} waited")
    return f" (placement: this reply {one(d.get('placement'))}; the first {one(d.get('first_placement'))})"


def section(path: Path) -> str:
    """The qdet sheet's section from the check's JSON (tools/dsv41/pod/e07_report.py)."""
    try:
        j = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return f"no result ({path} did not come back)"
    out = ["\n".join(f"  - {x.strip()}" if x.startswith("  ") else f"- {x}" for x in verdict_lines(j)), ""]
    rows = [f"| {r.get('request', '-')} | {r['label']} | {r['status']} | {r.get('completion_tokens', '')} | "
            f"{r.get('finish', r.get('error', ''))} | {r.get('ms_per_step', '')} | {r.get('first_token_ms', '')} | "
            f"{r.get('reset_ms', '')} | {r['seconds']} |"
            for r in j.get("requests", [])]
    out += ["| engine request | what | status | tokens | finish | ms per step | first token ms | reset ms | s |",
            "|---|---|---|---|---|---|---|---|---|",
            *rows, ""]
    i7 = j["items"].get("7") or {}
    for d in i7.get("divergence") or []:
        out.append(f"Item 7 divergence: {d['reply']} (engine request {d['request']}) leaves the cold reply at "
                   f"position {d['position']}: token {d['other']} where the cold reply has {d['first']}"
                   f"{placement_text(d)}.")
    for d in ((j.get("mode_off") or {}).get("7") or {}).get("divergence") or []:
        out.append(f"Mode off, item 7 divergence: {d['reply']} (engine request {d['request']}) at position "
                   f"{d['position']}: token {d['other']} against {d['first']}{placement_text(d)}.")
    t = ((j["items"].get("8") or {}).get("text_256") or {})
    if t:
        out += ["", f"Item 8, the reply at temperature 0.7, top_p 0.95, top_k 40, seed 1234 (engine request "
                    f"{t['request']}, {t['tokens']} tokens, {t['finish']}) for the hand check:", "", "```",
                t["text"], "```"]
    out += ["", f"Prompts: A {j['prompts']['A']!r}; B {j['prompts']['B']!r}; S {j['prompts']['S']!r}; max_tokens "
                f"{j['max_tokens']} (logs/det-sampling.json)."]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
