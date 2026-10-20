#!/usr/bin/env python3
"""tools/dsv41/quality/chat_load.py - N concurrent streaming chat users against one server: the concurrency baseline,
and the load of the cached-conversation, batched-decode, long-prompt and robustness checks.

Each user has a conversation of its own: a synthetic log of numbered entries (random.Random(SEED + its index), a
codename first, so no two users share a prompt token past the chat template's head), sized offline with the
model's tokenizer (tools/strata_tokenizer.py on the GGUF's vocabulary, the deepseek-v4.1 profile's template, thinking
off: needle_check.py's way) to CONTEXT tokens +- 2% for its first user message and a short assistant answer; then
TURNS turns, each a new user message of at most TURN_TOKENS new tokens asking for a count that runs past the reply's
max_tokens (det_sampling_check.py's speed prompt: eos cannot be ignored over HTTP, so a reply that ends early is a
measured short reply, flagged). The users start together; each sends its next turn when its previous reply ends, the
reply appended to its conversation.

  chat_load.py build --vocab GGUF --out conversations.json [--users 8] [--context 16384] [--long-context 0]
      writes the conversations: per user its codename, messages, prompt tokens and the turns' messages; with
      --long-context L one more, the "long" user, of L tokens (the user with a long prompt of `chat_load.py long`). Exit 1 when a
      conversation is not within 2% of its size, a turn has more than TURN_TOKENS new tokens, or two users share more
      than the template's head
  chat_load.py run --url URL --conversations F --users 1,2,4,8 --out results.json [--turns 2] [--max-tokens 256]
                   [--long-user] [--alternate [--think-s S] [--engine-log F]] [--timeout S]
      one run per N of --users (users 0..N-1, plus the long user with --long-user), every request a streaming
      /v1/chat/completions at temperature 0; prints each request's row and the table by N, writes the JSON. Exit 1
      when a request failed (not 200, no first token, no usage) or a turn had more than TURN_TOKENS new tokens.
      --alternate: the users take turns instead of starting together,
      user k's turn t sent when the previous user's reply has ended (user 0 turn 1, user 1 turn 1, .., user N-1 turn
      1, user 0 turn 2, ..), so nobody queues behind another request: the cached turn 2+ first token. --think-s S:
      S seconds between a reply's end and the next user's turn (default 0, the
      strict run), a chat whose users read the reply; the run's label names it. --engine-log F (the engine's log,
      the config's "log"): each request matched to the engine's new request in F (one at a time, as
      det_sampling_check.py's), its row gets the engine's request number and the part of the last request's idle reset
      (the deterministic mode's) it waited for, reset_wait_s; exit 1 when a request has
      not exactly one new hot set line in F
  chat_load.py long --url URL --conversations F --out results.json [--users 4] [--max-tokens 1024]
                    [--long-max-tokens 16] [--after-s 2] [--timeout S]
      Long prompts: the long user's first request alone (its TTFT solo),
      then N users together with long replies (max_tokens past the long prompt's read) and, AFTER_S seconds after
      every one of them has its first token, the long user's same request: each user's largest gap between two
      content chunks over its reply and over the long prompt's read (from its send to its first token), whether every
      user still decoded when the read ended (covered), and the long request's TTFT against its solo run. Both long
      requests go with "strata_checkpoint": false, so the second reads its prompt again rather than resume it. Exit 1
      when a request failed or a user ended before the read did (the run did not measure the read)
  chat_load.py steady --url URL --conversations F --out results.json [--users 4] [--window-s 15] [--repeats 3]
                      [--max-tokens 1024] [--timeout S]
      Batched decode: the users decoding at once in steady
      state, REPEATS times: users 0..N-1 send their first turn together; the window opens once every prompt is read
      and every user decodes (the last user's first token: nothing more is sent, so each engine step in it has exactly
      N rows and no prompt read between them) and lasts WINDOW_S seconds. Per repeat: each user's content chunks in
      the window over WINDOW_S (one a token, checked against its completion tokens), the aggregate their sum, the
      slowest, fastest and spread; then over the repeats (steady_summary). MAX_TOKENS has to outlast the tokens
      decoded before the window (between the prompt reads, which hold the batch) and the window itself (1024: 15 s at
      60 tok/s a user, above the single user's 56.50 on the RTX PRO 6000, and 124 tokens before it). Exit 1 when a
      request failed, a reply ended inside the window or its chunks are not its tokens

Per request (the JSON's rows): user, turn, its start after the run's, prompt tokens and the cached ones the server
reused (usage), its new tokens (turn 1: the prompt past the conversation; later: past the previous request's prompt
and reply), TTFT (the request sent to its first content chunk, at the client), queue wait (the server's
strata_queue_ms: from the request's entry into Service.run, after its prompt is tokenized, to its holding the FIFO),
decode tok/s (the tokens after the
first over the time from the first content chunk to the last), finish and the short flag (a reply that ended before
max_tokens), the server's timings, and whether it was queued behind another prefill (its queue wait over
QUEUED_S, 0.1 s). Per N: TTFT p50 and p95 and queue wait p50 and p95 (over every request of the run, linear between
the closest ranks), per-user decode tok/s (each user's tokens after the first over their time, over its turns: the
median and the slowest user), aggregate tok/s (every generated token over the run's wall time); over the turns 2+
alone, TTFT p50, p95 and max, the max of those not queued (the bar: <= 2 s), how many resumed a cached prefix
(cached tokens > 0), how many requests were queued, and with --engine-log the reset wait's p50 and max.
The streaming client is two_turn_check.py's post_stream, with the reply's last chunk read for the usage and the
queue wait. section(paths) is the markdown e07_report.py puts on the cbase sheet. python3 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SEED = 20261013
TOLERANCE = 0.02
TURN_TOKENS = 512
MAX_TOKENS = 256
# a request queued behind another prefill: its queue wait over this
QUEUED_S = 0.1
# a request's hot set line in the engine's log, with the last request's idle reset and its wait
# (det_sampling_check.py's IDLE_RE)
IDLE_RE = re.compile(r"^dsv41-engine serve: request (\d+): decode hot set .*; idle reset after the last request (\d+) "
                     r"swaps \(([0-9.]+) MiB\) in ([0-9.]+) ms, waited ([0-9.]+) ms$", re.M)
# each user's first word, distinct tokens, so the conversations share nothing past the template's head
CODENAMES = ("Amber", "Basalt", "Cobalt", "Driftwood", "Ember", "Fjord", "Granite", "Harbor", "Indigo", "Juniper",
             "Kestrel", "Lantern", "Meadow", "Nickel", "Orchard", "Pebble")
LONG_CODENAME = "Quartz"
NOUNS = ("crate", "valve", "ledger", "lantern", "bridge", "sensor", "barrel", "cable", "pump", "drone", "satchel",
         "turbine", "beacon", "anchor", "furnace", "compass")
ADJECTIVES = ("rusty", "polished", "heavy", "spare", "cracked", "new", "sealed", "damp", "bright", "worn")
VERBS = ("inspected", "moved", "repaired", "counted", "weighed", "logged", "cleaned", "tested", "stacked", "labelled")
PLACES = ("north dock", "east warehouse", "upper deck", "pump room", "south gate", "workshop", "loading bay",
          "control room")
ACK = "I have read the {name} log and will keep it in mind."
TURNS = ("Count from 1 to 500, writing every number in words, separated by commas, with nothing else (the {name} log "
         "above is only background).",
         "Now count from 501 to 1000 the same way, every number in words, separated by commas, with nothing else.",
         "Now count from 1001 to 1500 the same way, every number in words, separated by commas, with nothing else.",
         "Now count from 1501 to 2000 the same way, every number in words, separated by commas, with nothing else.")


def entry(rng: random.Random, i: int) -> str:
    return (f"Entry {i}: the crew {rng.choice(VERBS)} {rng.randrange(2, 90)} {rng.choice(ADJECTIVES)} "
            f"{rng.choice(NOUNS)}s at the {rng.choice(PLACES)} on day {rng.randrange(1, 366)}. ")


def document(name: str, seed: int, n: int) -> str:
    rng = random.Random(seed)
    return f"{name} log.\n" + "".join(entry(rng, i + 1) for i in range(n))


class Template:
    """The deepseek-v4.1 profile's chat template and tokenizer, thinking off (the profile's default)."""

    def __init__(self, vocab: str):
        from serve.frontend import ChatTemplate
        from serve.model_profile import DeepSeekV41Profile
        self.prof = DeepSeekV41Profile(vocab)
        self.tpl = ChatTemplate(source=self.prof.template_source)
        self.kw = self.prof.template_kwargs({"enable_thinking": False})

    def ids(self, messages: list[dict], generation: bool) -> list[int]:
        text = self.tpl.render(messages, add_generation_prompt=generation, **self.kw)
        return self.prof.tokenizer.encode(text, parse_special=True)


def history(name: str, doc: str) -> list[dict]:
    return [{"role": "user", "content": doc}, {"role": "assistant", "content": ACK.format(name=name)}]


def size_conversation(t: Template, name: str, seed: int, target: int) -> dict:
    """The largest number of entries whose conversation is at most target tokens."""
    def count(n):
        return len(t.ids(history(name, document(name, seed, n)), False))

    per = (count(400) - count(200)) / 200
    n = max(1, int((target - count(0)) / per))
    while n > 1 and count(n) > target:
        n -= 1
    while count(n + 1) <= target:
        n += 1
    msgs = history(name, document(name, seed, n))
    tokens = t.ids(msgs, False)
    turn1 = t.ids(msgs + [{"role": "user", "content": TURNS[0].format(name=name)}], True)
    return {"name": name, "seed": seed, "entries": n, "tokens": len(tokens), "target": target,
            "turn1_prompt_tokens": len(turn1), "turn1_new_tokens": len(turn1) - len(tokens),
            "turns": [x.format(name=name) for x in TURNS], "messages": msgs, "_ids": turn1}


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build(vocab: str, users: int, context: int, long_context: int) -> dict:
    if not 1 <= users <= len(CODENAMES):
        raise SystemExit(f"chat_load.py: --users {users}: 1 to {len(CODENAMES)}")
    t = Template(vocab)
    convs = [size_conversation(t, CODENAMES[u], SEED + u, context) for u in range(users)]
    if long_context:
        convs.append(size_conversation(t, LONG_CODENAME, SEED + 1000, long_context))
    # the template's head: what two one-token user messages share
    head = common_prefix(*(t.ids([{"role": "user", "content": x}], True) for x in ("A", "B")))
    shared = max((common_prefix(a["_ids"], b["_ids"]) for i, a in enumerate(convs) for b in convs[i + 1:]), default=0)
    bad = [f"{c['name']}: {c['tokens']} tokens, not within {TOLERANCE:.0%} of {c['target']}" for c in convs
           if abs(c["tokens"] - c["target"]) > TOLERANCE * c["target"]]
    bad += [f"{c['name']}: turn 1 has {c['turn1_new_tokens']} new tokens (at most {TURN_TOKENS})" for c in convs
            if c["turn1_new_tokens"] > TURN_TOKENS]
    if shared > head:
        bad.append(f"two conversations share {shared} prompt tokens, past the template's head of {head}")
    if bad:
        raise SystemExit("chat_load.py build: " + "; ".join(bad))
    for c in convs:
        del c["_ids"]
    return {"seed": SEED, "vocab": str(vocab), "vocab_sha256": hashlib.sha256(Path(vocab).read_bytes()).hexdigest(),
            "context": context, "long_context": long_context, "tolerance": TOLERANCE, "turn_tokens": TURN_TOKENS,
            "template_head_tokens": head, "shared_prefix_max": shared, "conversations": convs}


def model_of(url: str) -> str:
    with urllib.request.urlopen(url + "/v1/models", timeout=60) as r:
        return json.loads(r.read().decode())["data"][0]["id"]


def post_stream(url: str, body: dict, timeout: float, times: list | None = None, on_first=None) -> dict:
    """One streaming request (two_turn_check.py's post_stream): its status, content, the seconds from sending it to
    its first and last content chunk, and its last chunk (finish_reason, usage, strata_queue_ms, timings). times
    (the steady run): every content chunk's arrival appended, seconds on time.perf_counter()'s clock; also the
    seconds from sending it to every content chunk (chunks_s), and on_first() called at the first."""
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, last_at, last, text, rel = time.perf_counter(), None, None, None, [], []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                c = json.loads(line[len("data: "):])
                piece = c["choices"][0]["delta"].get("content")
                if piece:
                    at = time.perf_counter()
                    now = at - t0
                    if times is not None:
                        times.append(at)
                    if first is None and on_first is not None:
                        on_first()
                    first = now if first is None else first
                    last_at = now
                    text.append(piece)
                    rel.append(now)
                last = c
            return {"status": r.status, "content": "".join(text), "first_s": first, "last_s": last_at, "last": last,
                    "end_s": time.perf_counter() - t0, "chunks_s": rel}
    except urllib.error.HTTPError as e:
        with e:
            return {"status": e.code, "error": e.read().decode(errors="replace")[:500],
                    "end_s": time.perf_counter() - t0}
    except (OSError, ValueError) as e:
        return {"status": None, "error": f"{type(e).__name__}: {e}", "end_s": time.perf_counter() - t0}


class Conversation:
    """One user's turns: its messages and the last request's prompt and reply (the next turn's new tokens)."""

    def __init__(self, user: int, conv: dict, extra: dict | None = None, keep_chunks: bool = False):
        self.user, self.conv, self.msgs = user, conv, list(conv["messages"])
        self.prev_prompt = self.prev_reply = None
        self.failed = False
        self.extra, self.keep_chunks = dict(extra or {}), keep_chunks   # more request fields, chunk times
        self.first = threading.Event()   # set at the current request's first content chunk

    def turn(self, url: str, model: str, k: int, max_tokens: int, timeout: float, t0: float, rows: list,
             lock: threading.Lock) -> bool:
        """Turn k (0-based): its request, its row; False when it failed (the next turn would build on a reply it does
        not have)."""
        conv, msgs, user = self.conv, self.msgs, self.user
        msgs.append({"role": "user", "content": conv["turns"][k]})
        start = time.perf_counter() - t0
        self.first.clear()
        r = post_stream(url, {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, **self.extra},
                        timeout, on_first=self.first.set)
        row = {"user": user, "name": conv["name"], "turn": k + 1, "start_s": round(start, 3), "status": r["status"],
               "end_s": round(start + r["end_s"], 3)}
        last = r.get("last") or {}
        usage = last.get("usage")
        if r["status"] != 200 or r.get("first_s") is None or not usage:
            row["error"] = r.get("error") or ("no usage in the reply's last chunk" if r.get("first_s") is not None
                                              else "no content chunk")
            with lock:
                rows.append(row)
            self.failed = True
            return False
        pt, gen = usage["prompt_tokens"], usage["completion_tokens"]
        new = pt - conv["tokens"] if k == 0 else pt - self.prev_prompt - self.prev_reply
        decode_s = r["last_s"] - r["first_s"]
        q = last.get("strata_queue_ms")
        finish = last["choices"][0]["finish_reason"]
        gaps = [b - a for a, b in zip(r["chunks_s"], r["chunks_s"][1:])]
        row.update(prompt_tokens=pt, cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                   new_tokens=new, ttft_s=round(r["first_s"], 3), queue_s=None if q is None else round(q / 1000, 3),
                   max_gap_s=round(max(gaps), 3) if gaps else None,
                   queued=q is not None and q / 1000 > QUEUED_S,
                   completion_tokens=gen, decode_s=round(decode_s, 3),
                   decode_tok_s=round((gen - 1) / decode_s, 2) if gen > 1 and decode_s > 0 else None, finish=finish,
                   short=finish != "length", timings=last.get("timings"))
        if new > TURN_TOKENS:
            row["error"] = f"{new} new tokens in turn {k + 1}, more than {TURN_TOKENS}"
        if self.keep_chunks:   # every content chunk's time, from the run's start
            row["chunks_s"] = [round(start + x, 3) for x in r["chunks_s"]]
        with lock:
            rows.append(row)
        msgs.append({"role": "assistant", "content": r["content"]})
        self.prev_prompt, self.prev_reply = pt, gen
        return True


def user_run(url: str, model: str, user: int, conv: dict, turns: int, max_tokens: int, timeout: float, t0: float,
             rows: list, lock: threading.Lock, extra: dict | None = None) -> None:
    c = Conversation(user, conv, extra)
    for k in range(turns):
        if not c.turn(url, model, k, max_tokens, timeout, t0, rows, lock):
            return


def percentile(xs: list[float], q: float) -> float | None:
    """Linear between the closest ranks (numpy's default)."""
    if not xs:
        return None
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summary(rows: list[dict], wall_s: float) -> dict:
    ok = [r for r in rows if "error" not in r]
    ttft = [r["ttft_s"] for r in ok]
    queue = [r["queue_s"] for r in ok if r["queue_s"] is not None]
    per_user = []
    for u in sorted({r["user"] for r in ok}):
        mine = [r for r in ok if r["user"] == u and r["completion_tokens"] > 1]
        t = sum(r["decode_s"] for r in mine)
        if mine and t > 0:
            per_user.append(sum(r["completion_tokens"] - 1 for r in mine) / t)
    gen = sum(r["completion_tokens"] for r in ok)
    # the turns 2+ alone (a conversation's prefix there to reuse), and those not queued behind another prefill
    later = [r for r in ok if r["turn"] > 1]
    later_ttft = [r["ttft_s"] for r in later]
    free = [r["ttft_s"] for r in later if not r.get("queued")]
    waits = [r["reset_wait_s"] for r in later if r.get("reset_wait_s") is not None]

    def rnd(v, nd=3):
        return None if v is None else round(v, nd)
    return {"requests": len(rows), "errors": len(rows) - len(ok), "short": sum(r["short"] for r in ok),
            "turn2_requests": len(later), "turn2_ttft_p50_s": rnd(percentile(later_ttft, 0.5)),
            "turn2_ttft_p95_s": rnd(percentile(later_ttft, 0.95)), "turn2_ttft_max_s": rnd(max(later_ttft, default=None)),
            "turn2_unqueued_ttft_max_s": rnd(max(free, default=None)),
            "turn2_resumed": sum(1 for r in later if r.get("cached_tokens", 0) > 0),
            "queued": sum(1 for r in ok if r.get("queued")),
            "turn2_reset_wait_p50_s": rnd(percentile(waits, 0.5)), "turn2_reset_wait_max_s": rnd(max(waits, default=None)),
            "ttft_p50_s": rnd(percentile(ttft, 0.5)), "ttft_p95_s": rnd(percentile(ttft, 0.95)),
            "queue_p50_s": rnd(percentile(queue, 0.5)), "queue_p95_s": rnd(percentile(queue, 0.95)),
            "queue_reported": len(queue),
            "decode_tok_s_user_median": rnd(statistics.median(per_user), 2) if per_user else None,
            "decode_tok_s_user_min": rnd(min(per_user), 2) if per_user else None,
            "generated": gen, "wall_s": round(wall_s, 3),
            "aggregate_tok_s": round(gen / wall_s, 2) if wall_s > 0 else None}


def decode_window(rows: list[dict]) -> dict:
    """The aggregate decode tok/s over the decode window: every request's tokens
    after its first over the time at least one request was decoding (the union of each request's interval from its
    first content chunk to its last, start_s + ttft_s .. + decode_s), so prompt reads with nobody decoding leave it
    out (the baseline's aggregate is summary's aggregate_tok_s, over the run's wall time)."""
    ok = [r for r in rows if "error" not in r and r.get("completion_tokens", 0) > 1 and r.get("decode_s")]
    spans = sorted((r["start_s"] + r["ttft_s"], r["start_s"] + r["ttft_s"] + r["decode_s"]) for r in ok)
    busy, end = 0.0, None
    for a, b in spans:
        if end is None or a > end:
            busy += b - a
            end = b
        elif b > end:
            busy += b - end
            end = b
    tokens = sum(r["completion_tokens"] - 1 for r in ok)
    return {"decode_tokens": tokens, "decode_window_s": round(busy, 3),
            "decode_window_tok_s": round(tokens / busy, 2) if busy > 0 else None}


def idle_lines(log: Path) -> list[tuple[int, float]]:
    """The engine's hot set lines in its log: each request's number and the ms it waited for the last idle reset."""
    text = log.read_text(errors="replace") if log.exists() else ""
    return [(int(n), float(w)) for n, _, _, _, w in IDLE_RE.findall(text)]


def run_alternating(url: str, model: str, convs: list[dict], turns: int, max_tokens: int, timeout: float,
                    think_s: float = 0.0, engine_log: Path | None = None) -> dict:
    """The users take turns, each request sent think_s after the one before it ended (nobody queues); with
    engine_log, each row with the engine's request and the idle reset it waited for."""
    rows, lock = [], threading.Lock()
    users = [Conversation(u, c) for u, c in enumerate(convs)]
    seen = max((n for n, _ in idle_lines(engine_log)), default=0) if engine_log else 0
    t0, sent = time.perf_counter(), 0
    for k in range(turns):
        for c in users:
            if c.failed:
                continue
            if sent and think_s > 0:
                time.sleep(think_s)
            sent += 1
            answered = c.turn(url, model, k, max_tokens, timeout, t0, rows, lock)
            if not engine_log:
                continue
            new = [(n, w) for n, w in idle_lines(engine_log) if n > seen]
            if not answered:
                # a failed request's engine line, if it has one, is not the next request's
                seen = max([seen] + [n for n, _ in new])
                continue
            row = rows[-1]
            if len(new) != 1:
                row["error"] = (f"{len(new)} new hot set lines with the idle reset in {engine_log} after this request "
                                "(one expected: the engine's log, one request at a time?)")
                continue
            seen = new[0][0]
            row.update(engine_request=seen, reset_wait_s=round(new[0][1] / 1000, 3))
    wall = time.perf_counter() - t0
    rows.sort(key=lambda r: (r["user"], r["turn"]))
    return {"users": len(convs), "names": [c["name"] for c in convs], "rows": rows, "summary": summary(rows, wall),
            "alternate": True, "think_s": think_s}


def run_n(url: str, model: str, convs: list[dict], turns: int, max_tokens: int, timeout: float,
          extra: dict | None = None) -> dict:
    rows, lock = [], threading.Lock()
    gate = threading.Barrier(len(convs) + 1)
    t0 = [0.0]

    def go(u, c):
        gate.wait()
        user_run(url, model, u, c, turns, max_tokens, timeout, t0[0], rows, lock, extra)

    threads = [threading.Thread(target=go, args=(u, c), daemon=True) for u, c in enumerate(convs)]
    for th in threads:
        th.start()
    t0[0] = time.perf_counter()
    gate.wait()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0[0]
    rows.sort(key=lambda r: (r["user"], r["turn"]))
    return {"users": len(convs), "names": [c["name"] for c in convs], "rows": rows, "summary": summary(rows, wall)}


# the long-prompt bar: no user waits longer than this between two tokens while another user's long prompt is read
GAP_MAX_S = 5.0


def window_gap(times: list[float], lo: float, hi: float) -> float | None:
    """The largest gap between two consecutive chunk times that overlaps [lo, hi] (None: no such gap)."""
    gaps = [b - a for a, b in zip(times, times[1:]) if b >= lo and a <= hi]
    return max(gaps) if gaps else None


def run_long(url: str, model: str, convs: list[dict], long_conv: dict, max_tokens: int, long_max_tokens: int,
             after_s: float, timeout: float) -> dict:
    """The long user's first request alone, then the users together and
    the same long request after_s seconds after each of them has its first token (see the module's docstring)."""
    lock, off = threading.Lock(), {"strata_checkpoint": False}
    solo_rows = []
    Conversation(len(convs), long_conv, off).turn(url, model, 0, long_max_tokens, timeout, time.perf_counter(),
                                                  solo_rows, lock)
    rows, long_rows = [], []
    users = [Conversation(u, c, keep_chunks=True) for u, c in enumerate(convs)]
    t0 = time.perf_counter()
    threads = [threading.Thread(target=c.turn, args=(url, model, 0, max_tokens, timeout, t0, rows, lock), daemon=True)
               for c in users]
    for th in threads:
        th.start()
    started = all(c.first.wait(timeout) for c in users)
    if started:
        time.sleep(after_s)
        Conversation(len(convs), long_conv, off).turn(url, model, 0, long_max_tokens, timeout, t0, long_rows, lock)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    rows.sort(key=lambda r: r["user"])
    solo = solo_rows[0] if solo_rows else {"error": "no row"}
    lng = long_rows[0] if long_rows else {"error": "not sent: a user had no first token"}
    ok = "error" not in solo and "error" not in lng and all("error" not in r for r in rows) and len(rows) == len(users)
    read = (lng["start_s"], lng["start_s"] + lng["ttft_s"]) if ok else None
    for r in rows:
        if read and r.get("chunks_s"):
            g = window_gap(r["chunks_s"], *read)
            r["read_max_gap_s"] = None if g is None else round(g, 3)
            r["covered"] = r["chunks_s"][0] <= read[0] and r["chunks_s"][-1] >= read[1]
    covered = ok and all(r.get("covered") for r in rows)
    gaps = [r["max_gap_s"] for r in rows if r.get("max_gap_s") is not None]
    read_gaps = [r["read_max_gap_s"] for r in rows if r.get("read_max_gap_s") is not None]
    summ = {"users": len(users), "max_gap_s": max(gaps, default=None), "read_max_gap_s": max(read_gaps, default=None),
            "covered": covered, "long_prompt_tokens": lng.get("prompt_tokens"),
            "long_ttft_solo_s": solo.get("ttft_s"), "long_ttft_s": lng.get("ttft_s"),
            "long_ttft_ratio": round(lng["ttft_s"] / solo["ttft_s"], 3) if ok and solo["ttft_s"] else None,
            "long_sent_s": lng.get("start_s"), "wall_s": round(wall, 3),
            "decode_tok_s_user_median": statistics.median(r["decode_tok_s"] for r in rows if r.get("decode_tok_s"))
            if ok else None}
    summ["gap_pass"] = ok and covered and summ["max_gap_s"] is not None and summ["max_gap_s"] <= GAP_MAX_S
    return {"solo": solo, "rows": rows, "long": lng, "summary": summ, "ok": ok and covered, "max_tokens": max_tokens,
            "long_max_tokens": long_max_tokens, "after_s": after_s, "gap_max_s": GAP_MAX_S}


LONG_HEAD = ["user", "prompt tokens", "TTFT s", "decode tok/s", "generated", "max gap s", "max gap in the read s",
             "decoded through the read", "finish"]


def long_tables(j: dict) -> str:
    """The long run's tables: the users' gaps, the long request alone and beside them."""
    s = j["summary"]
    rows = [[r["user"], r.get("prompt_tokens"), r.get("ttft_s"), r.get("decode_tok_s"), r.get("completion_tokens"),
             r.get("max_gap_s"), r.get("read_max_gap_s"), r.get("covered"), r.get("error") or r.get("finish")]
            for r in j["rows"]]
    lng = [["alone", j["solo"].get("prompt_tokens"), j["solo"].get("ttft_s"), j["solo"].get("error") or "ok"],
           [f"beside {s['users']} decoding users (sent at {s['long_sent_s']} s)", j["long"].get("prompt_tokens"),
            j["long"].get("ttft_s"), j["long"].get("error") or "ok"]]
    return "\n".join([markdown(LONG_HEAD, rows), "", markdown(["long request", "prompt tokens", "TTFT s", "status"], lng),
                      "", f"Largest gap between two tokens of any user: {s['max_gap_s']} s ({s['read_max_gap_s']} s "
                      f"during the long read), at most {j['gap_max_s']} s: {'PASS' if s['gap_pass'] else 'FAIL'}; "
                      f"every user decoded through the read: {s['covered']}; the long TTFT {s['long_ttft_s']} s "
                      f"against {s['long_ttft_solo_s']} s alone ({s['long_ttft_ratio']} x)."])
def run_steady(url: str, model: str, convs: list[dict], window_s: float, max_tokens: int, timeout: float) -> dict:
    """One repeat of the steady state. The N users
    send their first turn together (run_n's start); the server's engine reads their prompts one at a time, each user
    decoding from its own first token on; the window opens at the last user's first token (every prompt read, every
    user in a batch slot and nothing more sent, so each step of the window has exactly N rows and no prompt read
    between them) and lasts window_s seconds. Per user: its content chunks in the window (one a token: checked
    against its completion tokens) over window_s; the aggregate: their sum over window_s. A failed request, a reply
    that ends inside the window or whose content chunks are not its completion tokens (+- 1) is the row's error."""
    n = len(convs)
    res, times = [None] * n, [[] for _ in range(n)]
    gate = threading.Barrier(n + 1)

    def go(u, c):
        msgs = list(c["messages"]) + [{"role": "user", "content": c["turns"][0]}]
        gate.wait()
        res[u] = post_stream(url, {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0},
                             timeout, times[u])

    threads = [threading.Thread(target=go, args=(u, c), daemon=True) for u, c in enumerate(convs)]
    for th in threads:
        th.start()
    gate.wait()
    t0 = time.perf_counter()
    for th in threads:
        th.join()
    rows = []
    for u, c in enumerate(convs):
        r = res[u] or {"status": None, "error": "no reply"}
        last = r.get("last") or {}
        usage = last.get("usage")
        row = {"user": u, "name": c["name"], "status": r["status"]}
        if r["status"] != 200 or not times[u] or not usage:
            row["error"] = r.get("error") or ("no usage in the reply's last chunk" if times[u] else "no content chunk")
        else:
            row.update(prompt_tokens=usage["prompt_tokens"],
                       cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                       first_s=round(times[u][0] - t0, 3), last_s=round(times[u][-1] - t0, 3),
                       completion_tokens=usage["completion_tokens"], chunks=len(times[u]),
                       finish=last["choices"][0]["finish_reason"])
            if abs(len(times[u]) - usage["completion_tokens"]) > 1:
                row["error"] = (f"{len(times[u])} content chunks for {usage['completion_tokens']} completion tokens: "
                                "the window's count (a chunk a token) does not hold")
        rows.append(row)
    out = {"users": n, "names": [c["name"] for c in convs], "window_s": window_s, "max_tokens": max_tokens,
           "rows": rows}
    if any("error" in x for x in rows):
        return out
    start = max(t[0] for t in times)
    end = start + window_s
    for u, x in enumerate(rows):
        x["window_tokens"] = sum(1 for t in times[u] if start < t <= end)
        x["window_tok_s"] = round(x["window_tokens"] / window_s, 2)
        if times[u][-1] < end:
            x["error"] = (f"its reply ended at {times[u][-1] - t0:.3f} s, inside the window ({start - t0:.3f} to "
                          f"{end - t0:.3f} s): more max_tokens")
    if any("error" in x for x in rows):
        return out
    rates = sorted(x["window_tok_s"] for x in rows)
    med = statistics.median(rates)
    out.update(window_start_s=round(start - t0, 3),
               aggregate_tok_s=round(sum(x["window_tokens"] for x in rows) / window_s, 2),
               user_tok_s_min=rates[0], user_tok_s_median=round(med, 2), user_tok_s_max=rates[-1],
               user_spread_pct=round(100 * (rates[-1] - rates[0]) / med, 1) if med > 0 else None)
    return out


def steady_summary(repeats: list[dict]) -> dict:
    """The steady repeats together: the aggregate's mean, lowest and highest, the slowest and fastest user of any
    repeat, the median user over every repeat's users and the widest spread within a repeat."""
    ok = [r for r in repeats if "aggregate_tok_s" in r]
    agg = [r["aggregate_tok_s"] for r in ok]
    users = [x["window_tok_s"] for r in ok for x in r["rows"]]
    return {"repeats": len(repeats), "measured": len(ok),
            "aggregate_tok_s_mean": round(statistics.mean(agg), 2) if agg else None,
            "aggregate_tok_s_min": min(agg, default=None), "aggregate_tok_s_max": max(agg, default=None),
            "user_tok_s_min": min(users, default=None),
            "user_tok_s_median": round(statistics.median(users), 2) if users else None,
            "user_tok_s_max": max(users, default=None),
            "user_spread_pct_max": max((r["user_spread_pct"] for r in ok if r["user_spread_pct"] is not None),
                                       default=None)}


def steady_end_to_end(repeats: list[dict]) -> dict | None:
    """The steady runs' end-to-end figures, the prompt reads included, from each
    repeat's rows: each user's whole reply, (completion tokens - 1) over its first to its last token; the aggregate over
    the decode window, every user's tokens after its first over the first user's first token to the last user's last;
    each user's first token (from the repeat's start). Over the repeats: the aggregate's mean, lowest and highest, the
    slowest and the median whole reply, and the users' first tokens in their order, each its lowest and highest."""
    per, aggs, firsts = [], [], []
    for r in repeats:
        rows = [x for x in r.get("rows", []) if x.get("status") == 200 and x.get("completion_tokens", 0) > 1
                and x.get("last_s") is not None and x["last_s"] > x["first_s"]]
        if not rows or len(rows) != len(r.get("rows", [])):
            continue
        per += [(x["completion_tokens"] - 1) / (x["last_s"] - x["first_s"]) for x in rows]
        a, b = min(x["first_s"] for x in rows), max(x["last_s"] for x in rows)
        aggs.append(sum(x["completion_tokens"] - 1 for x in rows) / (b - a))
        firsts.append(sorted(x["first_s"] for x in rows))
    if not aggs:
        return None
    return {"repeats": len(aggs), "aggregate_tok_s_mean": round(statistics.mean(aggs), 2),
            "aggregate_tok_s_min": round(min(aggs), 2), "aggregate_tok_s_max": round(max(aggs), 2),
            "user_tok_s_min": round(min(per), 2), "user_tok_s_median": round(statistics.median(per), 2),
            "first_token_s": [(round(min(f[k] for f in firsts), 2), round(max(f[k] for f in firsts), 2))
                              for k in range(min(len(f) for f in firsts))]}


def steady_section(path: str) -> str:
    """The cbatch sheet's table of a steady run's file (one row a repeat, every user's rate in the window)."""
    try:
        j = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return f"no steady result ({Path(path).name} missing or unreadable)"
    rows = []
    for k, r in enumerate(j["repeats"], 1):
        bad = [f"user {x['user']}: {x['error']}" for x in r["rows"] if "error" in x]
        rows.append([k, r.get("window_start_s"), r.get("aggregate_tok_s"),
                     " / ".join(str(x.get("window_tok_s")) for x in r["rows"]), r.get("user_tok_s_min"),
                     r.get("user_tok_s_max"), r.get("user_spread_pct"),
                     " / ".join(str(x.get("cached_tokens")) for x in r["rows"]), "; ".join(bad) or "ok"])
    return "\n".join([markdown(["repeat", "window opens at s", "aggregate tok/s", "per user tok/s (users 0..N-1)",
                                "slowest", "fastest", "spread %", "cached prompt tokens", "errors"], rows), "",
                      f"{j['users']} users ({', '.join(j['names'])}), each its first turn of {Path(j['conversations']).name} "
                      f"at temperature 0, max_tokens {j['max_tokens']}, sent together; the window ({j['window_s']:g} s) "
                      "opens at the last user's first token: every prompt read, every user decoding in a batch slot "
                      "and nothing more sent, so each step in it has exactly N rows and no prompt read between them. "
                      "Per user: its content chunks in the window (one a token, checked against its completion "
                      "tokens) over the window; the aggregate: their sum over the window; spread: (fastest - slowest) "
                      "/ median user."])


HEAD = ["N", "requests", "TTFT p50 s", "TTFT p95 s", "queue p50 s", "queue p95 s", "decode tok/s per user (median)",
        "slowest user tok/s", "aggregate tok/s", "short replies", "errors", "turn 2+ TTFT p50 s", "turn 2+ TTFT p95 s",
        "turn 2+ TTFT max s", "turn 2+ max not queued s", "turn 2+ resumed", "queued", "turn 2+ reset wait p50 s",
        "turn 2+ reset wait max s"]


def table_row(run: dict) -> list:
    s = run["summary"]
    return [run["label"], s["requests"], s["ttft_p50_s"], s["ttft_p95_s"], s["queue_p50_s"], s["queue_p95_s"],
            s["decode_tok_s_user_median"], s["decode_tok_s_user_min"], s["aggregate_tok_s"], s["short"], s["errors"],
            s.get("turn2_ttft_p50_s"), s.get("turn2_ttft_p95_s"), s.get("turn2_ttft_max_s"),
            s.get("turn2_unqueued_ttft_max_s"),
            None if "turn2_resumed" not in s else f"{s['turn2_resumed']} of {s['turn2_requests']}", s.get("queued"),
            s.get("turn2_reset_wait_p50_s"), s.get("turn2_reset_wait_max_s")]


def markdown(head: list, rows: list) -> str:
    def cell(v):
        return "n/a" if v is None else str(v)
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows])


def section(paths: list[str], logs: str = "cbase") -> str:
    """The cbase (and ccache) sheet's table by N from the runs' JSON files (one per load step), each file named
    under logs/<logs>/."""
    runs, missing = [], []
    for p in paths:
        try:
            j = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            missing.append(Path(p).name)
            continue
        for r in j["runs"]:
            runs.append(dict(r, file=Path(p).name, max_tokens=j["max_tokens"], turns=j["turns"]))
    if not runs:
        return "no load result came back" + (f" ({', '.join(missing)} missing or unreadable)" if missing else "")
    out = [markdown(HEAD + ["rows"], [table_row(r) + [f"logs/{logs}/{r['file']}"] for r in runs]), "",
           f"Every request: streaming /v1/chat/completions at temperature 0, max_tokens {runs[0]['max_tokens']}, "
           "the turns per user after its own conversation (the run's label says alternating when the users take "
           "turns, else they start together, and its think time between a reply's end and the next turn); TTFT at "
           "the client (the request sent to its first content chunk), queue "
           "wait the server's strata_queue_ms (from the request's entry into Service.run, after its prompt is "
           "tokenized, to its holding the FIFO), p50 and p95 over every request of the run; decode tok/s per user its "
           "tokens after the first over their time; aggregate tok/s every generated token over the run's wall time; "
           f"turn 2+ the requests after each user's first, queued a queue wait over {QUEUED_S} s (behind another "
           "prefill), resumed those with cached tokens, reset wait the part of the last request's idle reset (the "
           "deterministic mode's, after each request) it waited for, from the engine's log (n/a without it). Each file "
           "has every request's row (prompt, cached and new tokens, TTFT, queue wait, queued, reset wait, decode "
           "tok/s, finish, the server's timings)."]
    for r in runs:
        rows = [[x["user"], x["turn"], x.get("prompt_tokens"), x.get("cached_tokens"), x.get("new_tokens"),
                 x.get("ttft_s"), x.get("queue_s"), x.get("queued"), x.get("reset_wait_s"), x.get("decode_tok_s"),
                 x.get("completion_tokens"), x.get("error") or x.get("finish")] for x in r["rows"]]
        out += ["", f"N = {r['label']}, per request ({r['file']}):", "",
                markdown(["user", "turn", "prompt tokens", "cached", "new", "TTFT s", "queue s", "queued",
                          "reset wait s", "decode tok/s", "generated", "finish"], rows)]
    if missing:
        out += ["", f"Missing or unreadable: {', '.join(missing)}"]
    return "\n".join(out)


def steady_main(a, short: list[dict]) -> int:
    """chat_load.py steady (see the module's doc)."""
    if not 2 <= a.users <= len(short):
        raise SystemExit(f"chat_load.py steady: --users {a.users}: 2 to the {len(short)} conversations of "
                         f"{a.conversations}")
    if a.window_s <= 0 or a.repeats < 1 or a.max_tokens < 2:
        raise SystemExit("chat_load.py steady: --window-s > 0, --repeats >= 1 and --max-tokens >= 2")
    model = model_of(a.url)
    repeats = []
    for k in range(a.repeats):
        r = run_steady(a.url, model, short[:a.users], a.window_s, a.max_tokens, a.timeout)
        repeats.append(r)
        for x in r["rows"]:
            print(f"steady N={a.users} repeat {k + 1} user {x['user']}: " + (
                f"FAIL {x['status']}: {x['error']}" if "error" in x and "chunks" not in x else
                f"{x['prompt_tokens']} prompt tokens ({x['cached_tokens']} cached), first token at {x['first_s']} s, "
                f"{x['completion_tokens']} generated ({x['chunks']} chunks, {x['finish']}), "
                f"{x.get('window_tokens')} in the window = {x.get('window_tok_s')} tok/s" +
                (f"; FAIL {x['error']}" if "error" in x else "")), flush=True)
        if "aggregate_tok_s" in r:
            print(f"steady N={a.users} repeat {k + 1}: window {r['window_start_s']} s + {a.window_s:g} s, aggregate "
                  f"{r['aggregate_tok_s']} tok/s, per user {r['user_tok_s_min']} to {r['user_tok_s_max']} (median "
                  f"{r['user_tok_s_median']}, spread {r['user_spread_pct']}%)", flush=True)
    sm = steady_summary(repeats)
    ok = sm["measured"] == a.repeats
    out = {"url": a.url, "conversations": str(a.conversations),
           "conversations_sha256": hashlib.sha256(a.conversations.read_bytes()).hexdigest(), "model": model,
           "mode": "steady", "users": a.users, "names": [c["name"] for c in short[:a.users]], "window_s": a.window_s,
           "max_tokens": a.max_tokens, "repeats": repeats, "summary": sm, "ok": ok}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"steady N={a.users}: aggregate {sm['aggregate_tok_s_mean']} tok/s (mean of {sm['measured']} repeats, "
          f"{sm['aggregate_tok_s_min']} to {sm['aggregate_tok_s_max']}), per user {sm['user_tok_s_min']} to "
          f"{sm['user_tok_s_max']} (median {sm['user_tok_s_median']}), widest spread {sm['user_spread_pct_max']}%")
    print(f"chat_load.py steady: {'PASS' if ok else 'FAIL'}, {a.repeats} repeats of {a.users} users, {a.out}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--vocab", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--users", type=int, default=8)
    b.add_argument("--context", type=int, default=16384)
    b.add_argument("--long-context", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--conversations", type=Path, required=True)
    r.add_argument("--users", default="1,2,4,8", help="comma-separated N, one run each")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--turns", type=int, default=2)
    r.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    r.add_argument("--long-user", action="store_true", help="the conversations' long user too, in every run")
    r.add_argument("--alternate", action="store_true",
                   help="the users take turns (each request when the one before it ended) instead of together")
    r.add_argument("--think-s", type=float, default=0.0,
                   help="with --alternate: seconds between a reply's end and the next user's turn (default 0)")
    r.add_argument("--engine-log", type=Path,
                   help="with --alternate: the engine's log, for each request's wait for the last idle reset")
    r.add_argument("--timeout", type=float, default=1800, help="seconds per request")
    r.add_argument("--deterministic", choices=("on", "off"),
                   help="every request's \"strata_deterministic\" (the speed A/B with the mode off); "
                        "absent: the server's default")
    lp = sub.add_parser("long")
    lp.add_argument("--url", required=True)
    lp.add_argument("--conversations", type=Path, required=True)
    lp.add_argument("--out", type=Path, required=True)
    lp.add_argument("--users", type=int, default=4)
    lp.add_argument("--max-tokens", type=int, default=1024, help="the users' replies, past the long prompt's read")
    lp.add_argument("--long-max-tokens", type=int, default=16)
    lp.add_argument("--after-s", type=float, default=2.0,
                    help="seconds after every user's first token before the long request goes")
    lp.add_argument("--timeout", type=float, default=1800, help="seconds per request")
    st = sub.add_parser("steady")
    st.add_argument("--url", required=True)
    st.add_argument("--conversations", type=Path, required=True)
    st.add_argument("--out", type=Path, required=True)
    st.add_argument("--users", type=int, default=4)
    st.add_argument("--window-s", type=float, default=15.0, help="seconds of the window with every user decoding")
    st.add_argument("--repeats", type=int, default=3)
    st.add_argument("--max-tokens", type=int, default=1024)
    st.add_argument("--timeout", type=float, default=1800, help="seconds per request")
    a = ap.parse_args()
    if a.cmd == "build":
        d = build(a.vocab, a.users, a.context, a.long_context)
        a.out.write_text(json.dumps(d, ensure_ascii=False) + "\n")
        for c in d["conversations"]:
            print(f"conversation {c['name']}: {c['tokens']} tokens (target {c['target']} +- {TOLERANCE:.0%}), "
                  f"{c['entries']} entries; turn 1 {c['turn1_prompt_tokens']} prompt tokens, "
                  f"{c['turn1_new_tokens']} new")
        print(f"shared prompt tokens between any two users: at most {d['shared_prefix_max']} (the template's head "
              f"{d['template_head_tokens']})")
        print(f"chat_load.py build: {a.out} sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
        return 0
    d = json.loads(a.conversations.read_text())
    short = [c for c in d["conversations"] if c["name"] != LONG_CODENAME]
    long_ = [c for c in d["conversations"] if c["name"] == LONG_CODENAME]
    if a.cmd == "long":
        if not long_:
            raise SystemExit(f"chat_load.py long: {a.conversations} has no long user (build --long-context)")
        if not 1 <= a.users <= len(short):
            raise SystemExit(f"chat_load.py long: --users {a.users}: 1 to the {len(short)} conversations of {a.conversations}")
        j = run_long(a.url, model_of(a.url), short[:a.users], long_[0], a.max_tokens, a.long_max_tokens, a.after_s,
                     a.timeout)
        j.update(url=a.url, conversations=str(a.conversations),
                 conversations_sha256=hashlib.sha256(a.conversations.read_bytes()).hexdigest())
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(j, indent=1, ensure_ascii=False) + "\n")
        print(long_tables(j))
        print(f"chat_load.py long: {'PASS' if j['ok'] else 'FAIL'} (the run; item 12's gap "
              f"{'PASS' if j['summary']['gap_pass'] else 'FAIL'}), {a.out}")
        return 0 if j["ok"] else 1
    if a.cmd == "steady":
        return steady_main(a, short)
    try:
        ns = [int(x) for x in a.users.split(",") if x]
    except ValueError:
        raise SystemExit(f"chat_load.py: --users '{a.users}' is not a list of whole numbers")
    if not ns or any(not 1 <= n <= len(short) for n in ns):
        raise SystemExit(f"chat_load.py: --users {a.users}: each N from 1 to the {len(short)} conversations of "
                         f"{a.conversations}")
    if a.long_user and not long_:
        raise SystemExit(f"chat_load.py: --long-user: {a.conversations} has no long user (build --long-context)")
    if not 1 <= a.turns <= len(TURNS):
        raise SystemExit(f"chat_load.py: --turns {a.turns}: 1 to {len(TURNS)}")
    if a.alternate and a.deterministic is not None:
        raise SystemExit("chat_load.py: --deterministic is for the users at once (not --alternate)")
    if not a.alternate and (a.think_s or a.engine_log):
        raise SystemExit("chat_load.py: --think-s and --engine-log need --alternate (one request at a time)")
    if a.think_s < 0:
        raise SystemExit(f"chat_load.py: --think-s {a.think_s}: 0 or more seconds")
    model = model_of(a.url)
    runs = []
    for n in ns:
        convs = short[:n] + (long_ if a.long_user else [])
        res = (run_alternating(a.url, model, convs, a.turns, a.max_tokens, a.timeout, a.think_s, a.engine_log)
               if a.alternate else run_n(a.url, model, convs, a.turns, a.max_tokens, a.timeout,
                                         {} if a.deterministic is None else
                                         {"strata_deterministic": a.deterministic == "on"}))
        res["label"] = (f"{n}" + (" + long" if a.long_user else "") + (" alternating" if a.alternate else "") +
                        (f", think {a.think_s:g} s" if a.think_s else ""))
        for x in res["rows"]:
            print(f"N={res['label']} user {x['user']} turn {x['turn']}: " + (
                f"FAIL {x['status']}: {x['error']}" if "error" in x and "ttft_s" not in x else
                f"{x['prompt_tokens']} prompt tokens ({x['cached_tokens']} cached, {x['new_tokens']} new), TTFT "
                f"{x['ttft_s']} s, queue {x['queue_s']} s{' (queued)' if x['queued'] else ''}" +
                (f", reset wait {x['reset_wait_s']} s" if "reset_wait_s" in x else "") + f", {x['completion_tokens']} "
                f"generated at {x['decode_tok_s']} tok/s, {x['finish']}" + (" (short)" if x["short"] else "") +
                (f"; FAIL {x['error']}" if "error" in x else "")), flush=True)
        runs.append(res)
    ok = all("error" not in x for r in runs for x in r["rows"])
    out = {"url": a.url, "conversations": str(a.conversations),
           "conversations_sha256": hashlib.sha256(a.conversations.read_bytes()).hexdigest(), "model": model,
           "turns": a.turns, "max_tokens": a.max_tokens, "long_user": a.long_user, "alternate": a.alternate,
           "think_s": a.think_s, "engine_log": None if a.engine_log is None else str(a.engine_log),
           "deterministic": a.deterministic,
           "queued_s": QUEUED_S, "runs": runs, "ok": ok}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(markdown(HEAD, [table_row(r) for r in runs]))
    print(f"chat_load.py run: {'PASS' if ok else 'FAIL'}, {sum(len(r['rows']) for r in runs)} requests, {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
