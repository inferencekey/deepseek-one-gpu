#!/usr/bin/env python3
"""tools/dsv41/quality/det_batch_check.py - batch determinism over HTTP (item 14): a batch of temperature-0 requests
returns the tokens each request returns alone.

Runs against serve/server.py on dsv41-engine with batch slots (the config's "parallel"), the engine started with
--log-ids, its deterministic mode on (the default), every request greedy (temperature 0, max_tokens 256):

  alone       prompt A alone, then each of the 4 different prompts D1 .. D4 alone, one request at a time
  identical   A 4 times sent together, non-streamed: each reply's tokens A alone's
  stream      A 4 times sent together, streamed: the same
  different   D1 .. D4 sent together, non-streamed: each reply's tokens its own alone run's

The tokens come from the engine's log: each request's prompt_token_ids and generated_token_ids lines, and a batch
slot's slot_token_ids line (every token from its admission's), so a reply the server moved between the solo path and a
slot (a STOP then a BGEN with the prompt and the tokens so far, or a BSTOP and a GEN) is put back together: the engine
requests of a phase are chained by their prompts (a request whose prompt is the root prompt plus the first k tokens of
a chain continues it from there). A phase passes when its chains are as many as its requests and each chain's tokens
are its prompt's alone tokens, each reply's text is its prompt's alone text, every engine request of the phase and
of its alone runs was in the deterministic mode (each request's line in the engine's log), and the batch decoded them
beside each other (the engine's batch slot lines: at least all but one of the phase's requests decoded steps beside
other slots; otherwise the phase proves nothing and fails as not batched). Item 14 passes when identical, stream and
different all pass.

Each chain's verdict is printed and recorded, with whether it was promoted (it began on the solo path and was moved
into a slot: more than one engine request; on the RTX PRO 6000 cdet session of 2026-10-17 only those chains left
their alone runs, before the promotion fix, their last token read by prefill), whether its one token left was
fed by decode (the engine's line), and the reproducibility its batch slot line claims. The flag is checked per reply
against its own chain: each reply is paired with a chain of its prompt, a reply
whose text is its alone run's with a chain equal to its alone tokens, any other with one that is not (a phase whose
counts of the two kinds disagree fails), and a reply saying "strata_reproducible": true whose chain differs from its
alone run fails the phase as a dishonest flag; so does a chain whose slot line claims reproducible while it differs.

    det_batch_check.py --url http://127.0.0.1:8090 --engine-log out/quality-engine.log --out out/det-batch.json

The JSON has every reply (phase, prompt, text, finish, strata_reproducible), each phase's chains with their first
difference against the alone tokens, whether promoted, their slot line's claim, each reply's chain and verdict, the
batch evidence, and the verdict. Exit 0 when item 14 passes, 1 otherwise.
section(path) is the markdown e07_report.py puts on the cdet sheet.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from det_sampling_check import MAX_TOKENS, PROMPT_A, first_divergence, model_of, post  # noqa: E402

N_BATCH = 4
PROMPTS_D = ["Write a haiku about the sea, then explain its imagery in two sentences.",
             "List the planets of the solar system in order from the sun, with one fact about each.",
             "Describe how a bicycle's gears work, in about 150 words.",
             "Give three tips for writing clear technical documentation, each with a short example."]
REQ_TIMEOUT_S = 900
ENGINE_LOG_WAIT_S = 30
PROMPT_RE = re.compile(r"^dsv41-engine serve: request (\d+): prompt_token_ids: ?(.*)$", re.M)
GEN_RE = re.compile(r"^dsv41-engine serve: request (\d+): generated_token_ids: ?(.*)$", re.M)
SLOT_RE = re.compile(r"^dsv41-engine serve: request (\d+): slot_token_ids: ?(.*)$", re.M)
MODE_RE = re.compile(r"^dsv41-engine serve: request (\d+): \d+ prompt tokens, .*; \d+ tokens held, snapshot at -?\d+; "
                     r"(deterministic(?: off)?); ", re.M)
BESIDE_RE = re.compile(r"^dsv41-engine serve: request (\d+): batch slot \d+: \d+ generated \(\w+\) in [0-9.]+ ms, \d+ steps, "
                       r"(\d+) of them beside other slots.*; (reproducible|not reproducible)\b", re.M)
# a request whose one token left was the last request's unfed token, fed by decode (session.hpp)
DECODED_RE = re.compile(r"^dsv41-engine serve: request (\d+): its 1 token read was fed by decode", re.M)


def ids(text: str) -> list[int]:
    return [int(x) for x in text.split()]


def body(model: str, prompt: str, stream: bool) -> dict:
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": MAX_TOKENS,
         "reasoning_effort": "none"}
    if stream:
        b["stream"] = True
    return b


def stream_post(url: str, b: dict, timeout: float) -> tuple[int, dict]:
    """A streamed /v1/chat/completions: the content of its deltas, its finish and its last chunk's strata fields."""
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    content, last = [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        status = r.status
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            c = json.loads(line[5:])
            if not c.get("choices"):
                continue
            content.append(c["choices"][0].get("delta", {}).get("content") or "")
            if c["choices"][0].get("finish_reason"):
                last = c
    out = {"choices": [{"message": {"content": "".join(content)},
                        "finish_reason": last.get("choices", [{}])[0].get("finish_reason")}]}
    for k in ("strata_reproducible", "usage"):
        if k in last:
            out[k] = last[k]
    return status, out


class Log:
    """The engine's --log-ids lines from a point on."""

    def __init__(self, path: Path):
        self.path = path

    def text(self) -> str:
        return self.path.read_text(errors="replace") if self.path.exists() else ""

    def mark(self) -> int:
        return len(self.text())

    def requests(self, since: int) -> list[dict]:
        """Every engine request since the mark with its prompt and its tokens (a slot's whole run when it had one)."""
        t = self.text()[since:]
        prompts = {int(n): ids(v) for n, v in PROMPT_RE.findall(t)}
        gens = {int(n): ids(v) for n, v in GEN_RE.findall(t)}
        slots = {int(n): ids(v) for n, v in SLOT_RE.findall(t)}
        beside = {int(n): (int(u), r == "reproducible") for n, u, r in BESIDE_RE.findall(t)}
        decoded = {int(n) for n in DECODED_RE.findall(t)}
        mode = {int(n): m == "deterministic" for n, m in MODE_RE.findall(t)}
        out = []
        for n in sorted(prompts):
            if n not in gens:
                continue
            out.append({"request": n, "prompt": prompts[n], "tokens": slots.get(n, gens[n]), "slot": n in slots,
                        "beside_steps": beside.get(n, (0, None))[0], "slot_reproducible": beside.get(n, (0, None))[1],
                        "fed_by_decode": n in decoded, "deterministic": mode.get(n)})
        return out


def chains(reqs: list[dict], root: list[int]) -> list[dict]:
    """The engine requests of one prompt put back together: a request whose prompt is the root continues the chain
    whose tokens its prompt's rest begins (from that point), else it starts a chain."""
    out = []
    for q in reqs:
        p = q["prompt"]
        if p[:len(root)] != root:
            continue
        rest = p[len(root):]
        for c in out:
            if len(rest) <= len(c["tokens"]) and c["tokens"][:len(rest)] == rest and (rest or not c["members"]):
                c["tokens"] = c["tokens"][:len(rest)] + q["tokens"]
                c["members"].append(q["request"])
                c["beside_steps"] += q["beside_steps"]
                c["deterministic"] = c["deterministic"] and q["deterministic"] is True
                c["fed_by_decode"] = c["fed_by_decode"] or q["fed_by_decode"]
                if q["slot_reproducible"] is not None:
                    c["slot_reproducible"] = q["slot_reproducible"]
                break
        else:
            if rest:
                continue   # a continuation of a chain this phase does not have
            out.append({"tokens": list(q["tokens"]), "members": [q["request"]], "beside_steps": q["beside_steps"],
                        "deterministic": q["deterministic"] is True, "fed_by_decode": q["fed_by_decode"],
                        "slot_reproducible": q["slot_reproducible"]})
    return out


def wait_requests(log: Log, since: int, n: int) -> list[dict]:
    """The phase's engine requests once the log has n roots' worth of lines (the slot lines come at the slot's end)."""
    deadline = time.time() + ENGINE_LOG_WAIT_S
    while True:
        reqs = log.requests(since)
        if len(reqs) >= n or time.time() > deadline:
            return reqs
        time.sleep(0.5)


class Check:
    def __init__(self, url: str, log: Path):
        self.url, self.log = url, Log(log)
        self.model = model_of(url)
        self.replies: list[dict] = []

    def one(self, phase: str, prompt: str, stream: bool = False) -> dict:
        b = body(self.model, prompt, stream)
        status, r = (stream_post if stream else post)(self.url, b, REQ_TIMEOUT_S)
        msg = (r.get("choices") or [{}])[0]
        rep = {"i": len(self.replies), "phase": phase, "prompt": prompt, "stream": stream, "status": status,
               "text": (msg.get("message") or {}).get("content"), "finish": msg.get("finish_reason"),
               "strata_reproducible": r.get("strata_reproducible")}
        self.replies.append(rep)
        return rep

    def together(self, phase: str, prompts: list[str], stream: bool) -> list[dict]:
        """The prompts sent at once, one thread each, released together; the replies in prompt order."""
        out: list[dict | None] = [None] * len(prompts)
        gate = threading.Barrier(len(prompts))

        def go(i: int) -> None:
            gate.wait()
            try:
                out[i] = self.one(phase, prompts[i], stream)
            except Exception as e:   # recorded, the phase fails on it
                out[i] = {"i": len(self.replies), "phase": phase, "prompt": prompts[i], "stream": stream,
                          "error": repr(e)}
                self.replies.append(out[i])

        ts = [threading.Thread(target=go, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return [r for r in out if r is not None]


def alone_tokens(log: Log, since: int) -> tuple[list[int], list[int], bool] | None:
    """The one engine request of an alone run: its prompt, its tokens and whether it was in the deterministic mode."""
    reqs = wait_requests(log, since, 1)
    if len(reqs) != 1:
        return None
    return reqs[0]["prompt"], reqs[0]["tokens"], reqs[0]["deterministic"] is True


def judge(phase: str, replies: list[dict], reqs: list[dict], want: dict[str, dict]) -> dict:
    """One batched phase against the alone runs (want: prompt -> {"prompt_ids", "tokens", "text", "deterministic"}):
    every chain's verdict, every reply paired with a chain of its prompt and its flag checked against that chain."""
    why, rows = [], []
    for r in replies:
        if r.get("error") or r.get("status") != 200:
            why.append(f"a request failed: {r.get('error') or r.get('status')}")
    by_prompt: dict[str, list[dict]] = {}
    for r in replies:
        by_prompt.setdefault(r["prompt"], []).append(r)
    beside = 0
    for prompt, rs in by_prompt.items():
        w = want[prompt]
        if not w["deterministic"]:
            why.append("an alone run was not in the deterministic mode")
        cs = chains(reqs, w["prompt_ids"])
        beside += sum(1 for c in cs if c["beside_steps"] > 0)
        if any(not c["deterministic"] for c in cs):
            why.append("an engine request was not in the deterministic mode")
        if len(cs) != len(rs):
            why.append(f"{len(rs)} requests of a prompt but {len(cs)} token chains in the engine's log")
        mine = []
        for c in cs:
            d = first_divergence(w["tokens"], c["tokens"])
            row = {"prompt": prompt, "members": c["members"], "promoted": len(c["members"]) > 1,
                   "fed_by_decode": c["fed_by_decode"], "n_tokens": len(c["tokens"]), "beside_steps": c["beside_steps"],
                   "deterministic": c["deterministic"], "slot_reproducible": c["slot_reproducible"], "divergence": d,
                   "verdict": "PASS" if d is None else "FAIL", "reply": None}
            rows.append(row)
            mine.append(row)
            if d is not None:
                why.append(f"engine requests {c['members']}{' (promoted)' if row['promoted'] else ''}: the tokens differ "
                           f"from the alone run's at {d['position']} ({d['first']} alone, {d['other']} here)")
                if c["slot_reproducible"]:
                    why.append(f"engine requests {c['members']}: the slot line says reproducible but the tokens differ "
                               "(a dishonest flag)")
        # each reply with a chain of its prompt: a text equal to the alone run's with a chain equal to its tokens
        ok = [r for r in rs if not r.get("error")]
        same = [r for r in ok if r.get("text") == w["text"]]
        other = [r for r in ok if r.get("text") != w["text"]]
        if other:
            why.append("a reply's text differs from its alone run's")
        equal = [c for c in mine if c["divergence"] is None]
        apart = [c for c in mine if c["divergence"] is not None]
        if len(same) != len(equal) or len(other) != len(apart):
            why.append(f"the replies and the chains disagree: {len(other)} replies' texts differ from the alone run's, "
                       f"{len(apart)} chains' tokens")
        for r, c in list(zip(same, equal)) + list(zip(other, apart)):
            c["reply"] = r["i"]
            r["chain"] = c["members"]
            r["chain_equal"] = c["divergence"] is None
        for r in ok:
            says = r.get("strata_reproducible") is True
            shown = r.get("chain_equal") is True
            r["flag"] = "true, shown equal" if says and shown else "true, NOT shown equal" if says else "false"
            if says and not shown:
                why.append(f"reply {r['i']} says strata_reproducible true but its chain "
                           f"{r.get('chain') or '(none)'} is not shown equal to its alone run (a dishonest flag)")
    if beside < len(replies) - 1:
        why.append(f"not batched: {beside} of {len(replies)} requests decoded beside other slots (at least "
                   f"{len(replies) - 1} needed)")
    claimed = sum(1 for r in replies if r.get("strata_reproducible") is True)
    return {"phase": phase, "requests": len(replies), "beside": beside, "reproducible_true": claimed, "chains": rows,
            "replies": [{k: r.get(k) for k in ("i", "stream", "strata_reproducible", "chain", "chain_equal", "flag")}
                        for r in replies],
            "verdict": "PASS" if not why else "FAIL", "reasons": sorted(set(why), key=why.index)}


def chain_lines(p: dict) -> list[str]:
    """Each chain's verdict on a line of its own (a FAIL says at once whether only the promoted chains differ)."""
    out = []
    for c in p["chains"]:
        d = c["divergence"]
        what = ("its alone run's tokens" if d is None else
                f"differs from the alone run at {d['position']} ({d['first']} alone, {d['other']} here)")
        how = (f"promoted (began on the solo path, engine requests {c['members']}"
               + (", its last token fed by decode" if c.get("fed_by_decode") else "") + ")" if c["promoted"]
               else f"engine request {c['members'][0]}")
        claim = {True: "reproducible", False: "not reproducible", None: "no slot line"}[c.get("slot_reproducible")]
        out.append(f"  chain {how}: {c['n_tokens']} tokens, {c['beside_steps']} steps beside other slots, {what}; "
                   f"slot line {claim}; reply {c.get('reply') if c.get('reply') is not None else 'none'}: {c['verdict']}")
    for r in p.get("replies", []):
        out.append(f"  reply {r['i']}{' (streamed)' if r.get('stream') else ''}: strata_reproducible "
                   f"{json.dumps(r.get('strata_reproducible'))}, chain {r.get('chain')}: {r.get('flag', 'no reply')}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine-log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = Check(a.url, a.engine_log)
    want: dict[str, dict] = {}
    for p in [PROMPT_A] + PROMPTS_D:
        since = c.log.mark()
        r = c.one("alone", p)
        got = alone_tokens(c.log, since)
        if r.get("status") != 200 or got is None:
            print(f"det_batch_check.py: the alone run of {p[:40]!r} could not be read (status {r.get('status')}, its "
                  "engine lines: none or more than one request): unmeasured", file=sys.stderr)
            a.out.write_text(json.dumps({"replies": c.replies, "item14": {"verdict": "FAIL",
                                         "reasons": ["an alone run could not be read"]}}, indent=1) + "\n")
            return 1
        want[p] = {"prompt_ids": got[0], "tokens": got[1], "text": r["text"], "deterministic": got[2]}
    phases = []
    for name, prompts, stream in (("identical", [PROMPT_A] * N_BATCH, False), ("stream", [PROMPT_A] * N_BATCH, True),
                                  ("different", PROMPTS_D, False)):
        since = c.log.mark()
        replies = c.together(name, prompts, stream)
        reqs = wait_requests(c.log, since, len(prompts))
        time.sleep(1.0)   # the slots' end lines follow their last BT
        phases.append(judge(name, replies, c.log.requests(since), want))
        cs = phases[-1]["chains"]
        print(f"det_batch_check.py: {name}: {phases[-1]['verdict']}: {len(cs)} chains ({sum(c['promoted'] for c in cs)} "
              f"promoted), {sum(c['divergence'] is None for c in cs)} equal to alone"
              + (f" ({'; '.join(phases[-1]['reasons'])})" if phases[-1]["reasons"] else ""), flush=True)
        for line in chain_lines(phases[-1]):
            print(line, flush=True)
    reasons = [f"{p['phase']}: {x}" for p in phases for x in p["reasons"]]
    verdict = "PASS" if all(p["verdict"] == "PASS" for p in phases) else "FAIL"
    out = {"max_tokens": MAX_TOKENS, "batch": N_BATCH, "alone": {p: {"n_tokens": len(w["tokens"])} for p, w in want.items()},
           "replies": c.replies, "phases": phases, "item14": {"verdict": verdict, "reasons": reasons}}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"det_batch_check.py: item 14: {verdict}", flush=True)
    return 0 if verdict == "PASS" else 1


def section(path: Path) -> str:
    """Item 14's part of the session sheet: each phase, then every chain and every reply."""
    j = json.loads(Path(path).read_text())
    item = j["item14"]
    out = [f"Batch determinism (item 14, det_batch_check.py): **{item['verdict']}**"
           + (f" ({'; '.join(item['reasons'])})" if item["reasons"] else "") + ".", "",
           f"Every request greedy, max_tokens {j.get('max_tokens')}; the alone runs one at a time, then each phase's "
           f"{j.get('batch')} requests sent together; tokens from the engine's --log-ids lines (a reply moved between the "
           "solo path and a slot put back together: a promoted chain).", "",
           "| phase | requests | decoded beside other slots | token chains (promoted), their tokens | chains equal to "
           "alone | replies saying strata_reproducible true | verdict |",
           "|---|---|---|---|---|---|---|"]
    for p in j.get("phases", []):
        cs = p["chains"]
        lens = ", ".join(str(c["n_tokens"]) for c in cs) or "none"
        out.append(f"| {p['phase']} | {p['requests']} | {p['beside']} | {len(cs)} ({sum(bool(c.get('promoted')) for c in cs)}): "
                   f"{lens} | {sum(c['divergence'] is None for c in cs)} | {p.get('reproducible_true', 'not recorded')} | "
                   f"{p['verdict']} |")
    out += ["", "Every chain (a FAIL shows which chains differ, promoted or not):", "",
            "| phase | engine requests | promoted | last token fed by decode | tokens | steps beside other slots | first "
            "difference against alone | slot line | its reply's flag | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for p in j.get("phases", []):
        flags = {r.get("i"): r.get("flag") for r in p.get("replies", [])}
        for c in p["chains"]:
            d = c["divergence"]
            claim = {True: "reproducible", False: "not reproducible", None: "none"}[c.get("slot_reproducible")]
            out.append(f"| {p['phase']} | {', '.join(map(str, c['members']))} | {'yes' if c.get('promoted') else 'no'} | "
                       f"{'yes' if c.get('fed_by_decode') else 'no'} | {c['n_tokens']} | {c['beside_steps']} | "
                       f"{'none' if d is None else 'at ' + str(d['position'])} | {claim} | "
                       f"{flags.get(c.get('reply'), 'no reply paired') if c.get('reply') is not None else 'no reply paired'} | "
                       f"{c.get('verdict', 'FAIL' if d else 'PASS')} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
