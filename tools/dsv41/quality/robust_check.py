#!/usr/bin/env python3
"""tools/dsv41/quality/robust_check.py - the robustness check (item 13) over HTTP: the server, with the engine's
batch slots, served and robust.

Runs against serve/server.py with "parallel" (the deepseek-v4.1 profile: its queue_max and its batch watchdog), on
the real engine (the clong session) or fakes.py's (c04-test.sh), every request a /v1/chat/completions at temperature 0
on a socket of its own so a client can hang up. The slots P and the queue's bound Q come from /metrics (live.queue).

  stream          one streamed request (max_tokens 32): 200, content, its usage in the last chunk, then [DONE]
  non-stream      the same not streamed: 200, a message and its usage
  batched         two non-streamed requests at once (max_tokens 64): both 200 and both carrying
                  "strata_reproducible" (a reply alone carries none), so each decoded in a batch slot beside the
                  other; true in the deterministic mode, false off, either passes
  disconnect-stream      P streamed runners (a count, max_tokens 4096) decoding in the P slots and one more request
                  waiting; runner 0 hangs up: the waiting request gets its first token within DISCONNECT_MAX_S and
                  the other runners go on
  disconnect-non-stream  the same with a non-streamed runner hanging up (the server's socket check between tokens)
  overflow        the P runners decoding and Q requests waiting (/metrics' live.queue.waiting): one more gets a 503 at
                  once with Retry-After (whole seconds >= 1) and a JSON error naming the queue (code queue_full, its
                  waiting, running and queue_max); then every client hangs up and the server drains (nothing admitted,
                  no slot busy within DRAIN_MAX_S) and answers a last request with 200 within LAST_MAX_S (an
                  earlier clong session: the engine left the cancelled slots' BDONE lines unflushed and the last request
                  waited for a slot until the engine watchdog's 300 s)
  no-max-tokens   min(4, P - 1) streamed runners decoding and one more streamed count with no
                  max_tokens: it is admitted to a slot (min(4, P - 1) + 1 slots decoding within ADMIT_MAX_S), its slot's
                  max_tokens (/metrics) the server's default for it, min(default_max_tokens, batch_ctx - prompt + 1)
                  (8192 on the deepseek-v4.1 profile), and over its read and the NO_MAX_WINDOW_S
                  after its admission every runner keeps receiving tokens with no gap past GAP_MAX_S (the long-prompt
                  bar's 5 s); then everyone hangs up. Then, alone, the NEAR conversation (one user message of filler sized
                  by /v1/messages/count_tokens to about NEAR_MARGIN tokens under a slot) without max_tokens: 200,
                  its reply's strata_max_tokens {"asked": null, "max_tokens": the slot's room, batch_ctx - prompt + 1}
                  (the server's own word, as for slot-room's clamp turn), the room under the default, finish
                  "length" at the room or "stop" before it (near_verdict)
  slot-room       min(4, P - 1) runners decoding; a prompt of LONG_TOKENS (40K) with
                  max_tokens LONG_MAX_TOKENS (32000, past its slot's room) is admitted to a slot with max_tokens cut to
                  the room (/metrics), no runner's gap past GAP_MAX_S over its read and the NO_MAX_WINDOW_S after;
                  NEAR's next turn with max_tokens 32000: 200, its reply's strata_max_tokens the cut to the room
                  (batch_ctx - prompt + 1; the server's own word, as a reply that ends "stop" says
                  nothing of it), finish "length" at the room or "stop" before it (clamp_verdict), its cached tokens
                  (NEAR's checkpoint) not past its prompt; the turn after it, past a slot (PAST_EXTRA
                  more tokens), is refused at once with a 400 naming the slot's size; the runners leave, and that turn
                  alone (streamed, max_tokens 256) is moved into a slot by a short request that comes after its first
                  token, which it does not fit: it runs alone, /metrics' live.ran_alone shows it (held_slots 0) while
                  it runs, its reply says "strata_ran_alone", and the short request is answered after it

    robust_check.py --url http://127.0.0.1:8090 --out out/clong/robust.json [--timeout 600]

The JSON has each check's verdict, its numbers (the seconds a freed slot took, the Retry-After, the error body) and
why it failed; it is written again after every check ("complete": false until the last one), and each check's whole
JSON line is printed as it ends, so a step killed by its timeout keeps the checks that ran. Exit 0 when every check
passes, 1 otherwise. section(path) is the markdown e07_report.py puts on the
clong sheet.
"""
from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

SHORT = "Say hello in five words."
COUNT = "Count from 1 to 3000, writing every number in words, separated by commas, with nothing else."
# a freed slot's next request: the server's socket check (0.5 s), one batch step, the waiting request's short prompt
# read and its first token, on a busy GPU (the RTX PRO 6000's step ~18 ms; the fake's 5 ms)
DISCONNECT_MAX_S = 15.0
DRAIN_MAX_S = 60.0
# overflow's last request (a short prompt, 16 tokens) once the server drained: a slot free at once
LAST_MAX_S = 30.0
# no-max-tokens: a short count prompt read and admitted beside the runners (its read and a step); the window after its
# admission; the long-prompt bar's gap; a max_tokens past the default slot of 69632 positions and within the 262144 context
ADMIT_MAX_S = 120.0
NO_MAX_WINDOW_S = 10.0
GAP_MAX_S = 5.0
# NEAR, a conversation this many tokens under a slot (its room, its
# reply's tokens, a few more than its next turn's ~40 new ones); a long prompt that fits a slot with a max_tokens past its
# room (a 40K prompt and 32000); the tokens a turn past a slot adds (it then passes the slot by ~100)
NEAR_MARGIN = 160
LONG_TOKENS = 40000
LONG_MAX_TOKENS = 32000
PAST_EXTRA = 260
FILLER = " apple"
RUNNER_MAX_TOKENS = 4096     # a runner's reply (a count), past every check's window


class Client:
    """One request on its own socket: its status and headers, every SSE data line with its arrival, or its body."""

    def __init__(self, url: str, text, stream: bool, max_tokens: int | None, timeout: float):
        u = urlsplit(url)
        self.conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
        self.stream, self.status, self.headers, self.body, self.error = stream, None, {}, None, None
        self.events, self.sock, self.done = [], None, threading.Event()
        # text: one user message, or the messages themselves
        messages = [{"role": "user", "content": text}] if isinstance(text, str) else text
        body = {"messages": messages, "temperature": 0, "stream": stream}
        if max_tokens is not None:     # None: no max_tokens in the request (no-max-tokens)
            body["max_tokens"] = max_tokens
        threading.Thread(target=self._go, args=(json.dumps(body),), daemon=True).start()

    def _go(self, body: str):
        try:
            self.conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
            self.sock = self.conn.sock          # the response keeps it after the connection lets go (Connection: close)
            r = self.conn.getresponse()
            self.status, self.headers = r.status, dict(r.getheaders())
            if not self.stream or r.status != 200:
                raw = r.read().decode(errors="replace")
                try:
                    self.body = json.loads(raw)
                except ValueError:
                    self.body = {"raw": raw[:500]}
                return
            while True:
                line = r.fp.readline()
                if not line:
                    return
                line = line.decode(errors="replace").strip()
                if line.startswith("data: "):
                    self.events.append((time.monotonic(), line[6:]))
        except (OSError, ValueError, http.client.HTTPException) as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.done.set()

    def chunks(self) -> list[dict]:
        out = []
        for _, x in self.events:
            if x != "[DONE]":
                try:
                    out.append(json.loads(x))
                except ValueError:
                    pass
        return out

    def last_chunk(self) -> dict:
        """A streamed reply's last chunk (its finish, usage and strata_* fields), {} before it."""
        return next((c for c in reversed(self.chunks()) if c.get("usage")), {})

    def content_times(self) -> list[float]:
        out = []
        for t, x in list(self.events):
            try:
                c = json.loads(x) if x != "[DONE]" else {}
            except ValueError:
                continue
            if ((c.get("choices") or [{}])[0].get("delta") or {}).get("content"):
                out.append(t)
        return out

    def hang_up(self):
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
        self.conn.close()


def metrics(url: str) -> dict:
    with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
        return json.loads(r.read().decode())["live"]


def wait_until(cond, timeout: float, step: float = 0.1) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


def decoding(url: str) -> int:
    return sum(1 for s in metrics(url).get("slots", []) if s["state"] == "decoding")


def cached_ok(usage: dict | None) -> bool:
    """The cached prompt tokens are never past the prompt."""
    u = usage or {}
    return ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0) <= (u.get("prompt_tokens") or 0)


def room_verdict(status, body: dict | None, bctx: int, asked: int | None) -> dict:
    """A turn whose prompt fits a slot and whose max_tokens the server sets to the slot's room (batch_ctx - prompt + 1):
    the limit proven by the reply's strata_max_tokens (the max_tokens it decoded with, the room, and the `asked` it
    says: the request's max_tokens when it cut one, None when it asked for none), and the reply within it: finish
    "length" at the room, or "stop" at or before it (the model ended its answer itself; the 2026-10-16 clong session: 13 tokens of a
    room of 137, which a check wanting "length" took for a failure) -> the turn's figures, `ok` its verdict."""
    b = body or {}
    u = b.get("usage") or {}
    fin = (b.get("choices") or [{}])[0].get("finish_reason")
    n = u.get("completion_tokens")
    room = bctx - (u.get("prompt_tokens") or 0) + 1
    said = b.get("strata_max_tokens") or {}
    proven = room >= 1 and "asked" in said and said["asked"] == asked and said.get("max_tokens") == room
    within = isinstance(n, int) and (fin == "length" and n == room or fin == "stop" and 1 <= n <= room)
    return {"status": status, "prompt_tokens": u.get("prompt_tokens"), "finish": fin, "completion_tokens": n,
            "room": room, "max_tokens": said.get("max_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ok": status == 200 and proven and within and cached_ok(u)}


def clamp_verdict(status, body: dict | None, bctx: int, asked: int) -> dict:
    """slot-room's clamp turn (max_tokens `asked`, past its slot's room): room_verdict -> the check's clamp_* figures."""
    return {"clamp_" + k: v for k, v in room_verdict(status, body, bctx, asked).items()}


def near_verdict(status, body: dict | None, bctx: int, default: int | None) -> dict:
    """no-max-tokens' NEAR turn (no max_tokens, its prompt NEAR_MARGIN under a slot): the server gives it the slot's
    room, which is under the profile's default (proven by strata_max_tokens
    {"asked": null, "max_tokens": the room}, and finish "length" at the room or "stop" within it, as the clamp turn)
    -> the check's near_* figures."""
    turn = room_verdict(status, body, bctx, None)
    turn["ok"] = turn["ok"] and (not default or turn["room"] <= default)
    return {"near_" + k: v for k, v in turn.items()}


def count_tokens(url: str, messages: list[dict]) -> int:
    """The server's own count of a conversation's prompt (/v1/messages/count_tokens, the same template and tokenizer)."""
    req = urllib.request.Request(url + "/v1/messages/count_tokens", data=json.dumps({"messages": messages}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(json.loads(r.read().decode())["input_tokens"])


def sized(url: str, tokens: int, tail: str) -> list[dict]:
    """One user message of FILLER sized to about `tokens` prompt tokens, `tail` (an instruction) last: the count per
    filler unit from two measures, then one correction."""
    def conv(n):
        return [{"role": "user", "content": "Background words:" + FILLER * n + "\n\n" + tail}]
    a, b = count_tokens(url, conv(1000)), count_tokens(url, conv(3000))
    per = (b - a) / 2000
    n = max(1, round(1000 + (tokens - a) / per))
    n = max(1, n + round((tokens - count_tokens(url, conv(n))) / per))
    return conv(n)


def slot_of(url: str, max_tokens_not: int) -> dict | None:
    """/metrics' decoding slot whose max_tokens is not the runners' (the one request among them that is not one)."""
    for s in metrics(url).get("slots", []):
        if s["state"] == "decoding" and s.get("max_tokens") != max_tokens_not:
            return s
    return None


def runner_gaps(rs, t0: float, t1: float) -> list[float]:
    gaps = []
    for r in rs:
        ts = [t for t in r.content_times() if t0 <= t <= t1]
        gaps.append(round(max([b - a for a, b in zip([t0] + ts, ts + [t1])]), 3))
    return gaps


def check_simple(url: str, timeout: float) -> list[dict]:
    out = []
    s = Client(url, SHORT, True, 32, timeout)
    s.done.wait(timeout)
    ch = s.chunks()
    usage = next((c["usage"] for c in reversed(ch) if c.get("usage")), None)
    ok = (s.status == 200 and bool(s.content_times()) and usage is not None and s.events[-1][1] == "[DONE]" and
          cached_ok(usage))
    out.append({"check": "stream", "pass": ok, "status": s.status, "content_chunks": len(s.content_times()),
                "usage": usage, "error": s.error})
    n = Client(url, SHORT, False, 32, timeout)
    n.done.wait(timeout)
    msg = ((n.body or {}).get("choices") or [{}])[0].get("message", {}).get("content")
    ok = n.status == 200 and bool(msg) and bool((n.body or {}).get("usage")) and cached_ok((n.body or {}).get("usage"))
    out.append({"check": "non-stream", "pass": ok, "status": n.status, "content": (msg or "")[:120],
                "usage": (n.body or {}).get("usage"), "error": n.error})
    b = [Client(url, COUNT, False, 64, timeout) for _ in range(2)]
    for c in b:
        c.done.wait(timeout)
    rep = [(c.body or {}).get("strata_reproducible") for c in b]
    ok = all(c.status == 200 for c in b) and all(isinstance(r, bool) for r in rep) and all(cached_ok((c.body or {}).get("usage"))
                                                                         for c in b)
    out.append({"check": "batched", "pass": ok, "status": [c.status for c in b], "strata_reproducible": rep,
                "completion_tokens": [((c.body or {}).get("usage") or {}).get("completion_tokens") for c in b]})
    return out


def runners(url: str, n: int, timeout: float, first_stream: bool = True) -> list[Client]:
    """n count requests decoding in the n slots (runner 0 streamed or not)."""
    rs = [Client(url, f"{COUNT} (runner {k})", first_stream or k > 0, RUNNER_MAX_TOKENS, timeout) for k in range(n)]
    if not wait_until(lambda: decoding(url) == n, timeout):
        raise RuntimeError(f"the {n} runners never decoded together: {metrics(url).get('slots')}")
    return rs


def check_disconnect(url: str, slots: int, stream: bool, timeout: float) -> dict:
    name = "disconnect-stream" if stream else "disconnect-non-stream"
    rs = runners(url, slots, timeout, first_stream=stream)
    try:
        w = Client(url, SHORT, True, 16, timeout)
        queued = wait_until(lambda: metrics(url)["queue"]["waiting"] >= 1, 30)
        time.sleep(2.0)
        still = not w.content_times()
        others0 = [len(r.content_times()) for r in rs[1:]]
        t0 = time.monotonic()
        rs[0].hang_up()
        got = wait_until(lambda: bool(w.content_times()), DISCONNECT_MAX_S * 2, 0.02)
        took = (w.content_times()[0] - t0) if got else None
        time.sleep(1.0)
        on = all(len(r.content_times()) > k for r, k in zip(rs[1:], others0))
        ok = queued and still and got and took is not None and took <= DISCONNECT_MAX_S and on
        res = {"check": name, "pass": ok, "slots": slots, "waiting_seen": queued, "waited_for_a_slot": still,
               "freed_slot_first_token_s": None if took is None else round(took, 3), "max_s": DISCONNECT_MAX_S,
               "other_runners_went_on": on}
        w.done.wait(timeout)
        res["waiting_status"] = w.status
        res["pass"] = res["pass"] and w.status == 200
        return res
    finally:
        for r in rs:
            r.hang_up()
        wait_until(lambda: not metrics(url).get("running") and decoding(url) == 0, DRAIN_MAX_S)


def check_overflow(url: str, slots: int, queue_max: int, timeout: float) -> dict:
    rs = runners(url, slots, timeout)
    ws = []
    try:
        ws = [Client(url, f"{SHORT} ({k})", True, 16, timeout) for k in range(queue_max)]
        full = wait_until(lambda: metrics(url)["queue"]["waiting"] == queue_max, 60)
        one = Client(url, SHORT, True, 16, timeout)
        answered = one.done.wait(10)
        retry = one.headers.get("Retry-After", "")
        err = (one.body or {}).get("error") or {}
        q = err.get("queue") or {}
        # Retry-After at least the server's RETRY_AFTER_MIN_S, 5 s
        ok = (full and answered and one.status == 503 and retry.isdigit() and int(retry) >= 5 and
              err.get("code") == "queue_full" and "queue" in err.get("message", "") and q.get("queue_max") == queue_max
              and q.get("running") == slots and q.get("waiting") == queue_max)
        res = {"check": "overflow", "pass": ok, "slots": slots, "queue_max": queue_max, "queue_full_seen": full,
               "status": one.status, "retry_after": retry, "error": err}
    finally:
        for c in rs + ws:
            c.hang_up()
    drained = wait_until(lambda: metrics(url)["queue"]["admitted"] == 0 and decoding(url) == 0, DRAIN_MAX_S)
    t0 = time.monotonic()
    last = Client(url, SHORT, False, 16, timeout)
    answered = last.done.wait(LAST_MAX_S)
    took = time.monotonic() - t0
    if not answered:
        last.hang_up()
    res.update(drained=drained, last_status=last.status if answered else None, last_answered_s=round(took, 3),
               last_max_s=LAST_MAX_S)
    res["pass"] = res["pass"] and drained and answered and last.status == 200
    return res


def check_no_max_tokens(url: str, slots: int, timeout: float) -> tuple[dict, list[dict]]:
    """-> the check, and NEAR (its conversation, for slot-room)."""
    q = metrics(url).get("queue") or {}
    bctx, default = q.get("batch_ctx"), q.get("default_max_tokens")
    n = min(4, slots - 1)
    rs = runners(url, n, timeout)
    extra = None
    try:
        t0 = time.monotonic()
        extra = Client(url, f"{COUNT} (no max_tokens)", True, None, timeout)
        admitted = wait_until(lambda: decoding(url) == n + 1, ADMIT_MAX_S, 0.05)
        t_admit = time.monotonic()
        s = slot_of(url, RUNNER_MAX_TOKENS) if admitted else None
        want = (min(x for x in (default, bctx - s["prompt_tokens"] + 1 if bctx else None) if x)
                if s and (default or bctx) else None)
        if admitted:
            time.sleep(NO_MAX_WINDOW_S)
        t1 = time.monotonic()
        gaps = runner_gaps(rs, t0, t1)
        got = len([t for t in extra.content_times() if t <= t1])
        max_new_ok = s is not None and want is not None and s.get("max_tokens") == want
        ok = admitted and got > 0 and max(gaps) <= GAP_MAX_S and max_new_ok
        res = {"check": "no-max-tokens", "pass": ok, "runners": n, "admitted_to_a_slot": admitted,
               "admitted_after_s": round(t_admit - t0, 3) if admitted else None, "window_s": NO_MAX_WINDOW_S,
               "its_tokens": got, "runner_gaps_s": gaps, "gap_max_s": GAP_MAX_S, "default_max_tokens": default,
               "batch_ctx": bctx, "its_max_tokens": s and s.get("max_tokens"), "max_tokens_expected": want}
    finally:
        if extra is not None:
            extra.hang_up()
        for r in rs:
            r.hang_up()
        wait_until(lambda: not metrics(url).get("running") and decoding(url) == 0, DRAIN_MAX_S)
    # its limit: NEAR alone without max_tokens gets the slot's room (under the default), said by its reply
    near = sized(url, bctx - NEAR_MARGIN, COUNT) if bctx else []
    if not near:
        res.update(near_ok=False, near_error="no batch_ctx in /metrics' live.queue")
        res["pass"] = False
        return res, near
    c = Client(url, near, False, None, timeout)
    c.done.wait(timeout)
    res.update(near_verdict(c.status, c.body, bctx, default))
    res["pass"] = res["pass"] and res["near_ok"]
    # its next turns (slot-room) start from NEAR's prompt, which the engine keeps as a checkpoint; a short answer in
    # them, so the next user message keeps the turn within a slot
    return res, near + [{"role": "assistant", "content": "One, two, three."}]


def check_slot_room(url: str, slots: int, near: list[dict], timeout: float) -> dict:
    q = metrics(url).get("queue") or {}
    bctx = q.get("batch_ctx")
    if not bctx or not near:
        return {"check": "slot-room", "pass": False, "error": "no batch_ctx or no NEAR conversation"}
    n = min(4, slots - 1)
    rs = runners(url, n, timeout)
    res = {"check": "slot-room", "runners": n, "batch_ctx": bctx}
    long = None
    try:
        # a prompt that fits a slot, a max_tokens past its room: admitted, its max_tokens the room
        long_conv = sized(url, LONG_TOKENS, COUNT)
        t0 = time.monotonic()
        long = Client(url, long_conv, True, LONG_MAX_TOKENS, timeout)
        admitted = wait_until(lambda: decoding(url) == n + 1, ADMIT_MAX_S, 0.05)
        s = slot_of(url, RUNNER_MAX_TOKENS) if admitted else None
        if admitted:
            time.sleep(NO_MAX_WINDOW_S)
        t1 = time.monotonic()
        gaps = runner_gaps(rs, t0, t1)
        cut_ok = s is not None and s.get("max_tokens") == bctx - s["prompt_tokens"] + 1
        res.update(long_admitted=admitted, long_prompt_tokens=s and s["prompt_tokens"], long_max_tokens=s and
                   s.get("max_tokens"), long_cut_ok=cut_ok, runner_gaps_s=gaps)
        long.hang_up()
        wait_until(lambda: decoding(url) == n, DRAIN_MAX_S)
        # NEAR's next turn, max_tokens past its room: cut to it, finish "length"; its cached tokens NEAR's prefix
        turn = near + [{"role": "user", "content": "Now go on counting the same way, from where you stopped."}]
        b = Client(url, turn, False, LONG_MAX_TOKENS, timeout)
        b.done.wait(timeout)
        res.update(clamp_verdict(b.status, b.body, bctx, LONG_MAX_TOKENS))
        # a turn past a slot beside the runners: a 400 naming the slot's size, at once
        # (the clamp turn's checkpoint superseded NEAR's: the turn past a slot goes on from it, so alone it reads only
        # its own new tokens)
        past = turn + [{"role": "assistant", "content": "Four, five."},
                       {"role": "user", "content": "More words:" + FILLER * PAST_EXTRA + "\n\n" + COUNT}]
        c = Client(url, past, False, 64, timeout)
        answered = c.done.wait(30)
        msg = ((c.body or {}).get("error") or {}).get("message", "")
        refused_ok = answered and c.status == 400 and f"past a batch slot ({bctx} positions" in msg
        res.update(past_status=c.status, past_error=msg[:200], refused_ok=refused_ok)
    finally:
        if long is not None:
            long.hang_up()
        for r in rs:
            r.hang_up()
        wait_until(lambda: not metrics(url).get("running") and decoding(url) == 0, DRAIN_MAX_S)
    # the same turn alone, moved into a slot by a short request after its first token: it runs alone, said at its
    # start (/metrics) and in its reply
    d = Client(url, past, True, 256, timeout)
    first = wait_until(lambda: bool(d.content_times()), ADMIT_MAX_S, 0.05)
    short = Client(url, SHORT, False, 16, timeout)
    seen = {}

    def ran_alone_now():
        a = metrics(url).get("ran_alone")
        if a:
            seen.update(a)
        return bool(a) or d.done.is_set()
    wait_until(ran_alone_now, 60, 0.05)
    d.done.wait(timeout)
    short.done.wait(timeout)
    ran = d.last_chunk().get("strata_ran_alone") or {}
    alone_ok = (first and d.status == 200 and seen.get("held_slots") == 0 and ran.get("held_slots") == 0 and
                f"a batch slot holds {bctx}" in ran.get("reason", "") and short.status == 200)
    res.update(alone_first_token=first, alone_status=d.status, alone_seen_in_metrics=seen or None,
               strata_ran_alone=ran or None, short_status=short.status, alone_ok=alone_ok)
    res["pass"] = bool(res.get("long_admitted") and res.get("long_cut_ok") and max(res.get("runner_gaps_s") or [99])
                       <= GAP_MAX_S and res.get("clamp_ok") and res.get("refused_ok") and alone_ok)
    return res


def section(path: str) -> str:
    """The clong sheet's item 13 table from the JSON."""
    try:
        j = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return f"no result came back ({Path(path).name} missing or unreadable)"
    cut = [] if j.get("complete", True) else [
        "", f"The run ended before its last check (killed or failed): the {len(j['checks'])} checks above came back, "
        "the rest did not run or did not finish."]
    rows = []
    for c in j["checks"]:
        what = {k: v for k, v in c.items() if k not in ("check", "pass")}
        rows.append(f"| {c['check']} | {'PASS' if c['pass'] else 'FAIL'} | {json.dumps(what, ensure_ascii=False)[:400]} |")
    return "\n".join(["| check | verdict | figures |", "|---|---|---|"] + rows +
                     ["", f"Item 13: {'PASS' if j['ok'] else 'FAIL'} ({sum(c['pass'] for c in j['checks'])} of "
                      f"{len(j['checks'])} checks; P = {j['slots']} slots, queue_max {j['queue_max']})."] + cut)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=600, help="seconds per request")
    a = ap.parse_args()
    q = metrics(a.url).get("queue") or {}
    slots, queue_max = q.get("running_max"), q.get("queue_max")
    if not slots or slots < 2 or queue_max is None:
        raise SystemExit(f"robust_check.py: {a.url}/metrics: live.queue {q}: needs batch slots (parallel >= 2) and a "
                         "queue_max")
    checks = []
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def record(c: dict):
        """One check's whole line, and the JSON so far (a later check killed by the step's timeout keeps it)."""
        checks.append(c)
        print(f"robust_check.py: {c['check']}: {'PASS' if c['pass'] else 'FAIL'} "
              f"{json.dumps({k: v for k, v in c.items() if k not in ('check', 'pass')}, ensure_ascii=False)}", flush=True)
        write(False)

    def write(complete: bool):
        j = {"url": a.url, "slots": slots, "queue_max": queue_max, "checks": checks, "complete": complete,
             "ok": complete and all(c["pass"] for c in checks)}
        tmp = a.out.with_name(a.out.name + ".tmp")
        tmp.write_text(json.dumps(j, indent=1, ensure_ascii=False) + "\n")
        tmp.replace(a.out)
        return j
    for c in check_simple(a.url, a.timeout):
        record(c)
    near = []

    def no_max():
        nonlocal near
        c, near = check_no_max_tokens(a.url, slots, a.timeout)
        return c
    for fn in (lambda: check_disconnect(a.url, slots, True, a.timeout),
               lambda: check_disconnect(a.url, slots, False, a.timeout),
               lambda: check_overflow(a.url, slots, queue_max, a.timeout),
               no_max, lambda: check_slot_room(a.url, slots, near, a.timeout)):
        try:
            c = fn()
        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as e:
            c = {"check": "setup", "pass": False, "error": f"{type(e).__name__}: {e}"}
        record(c)
    j = write(True)
    print(f"robust_check.py: {'PASS' if j['ok'] else 'FAIL'}, {a.out}")
    return 0 if j["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
