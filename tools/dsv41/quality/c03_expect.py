#!/usr/bin/env python3
"""tools/dsv41/quality/c03_expect.py - what chat_load.py should measure against the fake engine's batch slots, by hand
from its delays (c03-test.sh). c01_expect.py is the FIFO's model (batching off).

The server (its "parallel": N, StrataEngine.generate_batched) and fakes.py's engine with --batch N, as one timeline:

  * the engine does one thing at a time: a prompt read (an admission, or a request alone) or a step of the batch;
    a request waiting for it goes before a step (the engine reads its stdin lines between two steps);
  * a request that arrives with nothing else running or waiting runs alone (GEN): its read at P per prompt token
    read, then a token every D. Once another request waits, the alone one is stopped at its next token (the server's
    STOP rides on a T line, so a prompt read is never cut) and sent again into a slot (BGEN of its prompt and its
    tokens: the engine's live state, all of them but the last, so it reads that one token, P, and picks its next, D);
  * any other request is admitted into a slot in arrival order: its read, then its first token D, while the slots
    wait (the stall);
  * a step: every slot one token, in D + R (B - 1) for B slots (FAKE_ENGINE_BATCH_ROW_MS); a slot ends at its
    max_tokens (its reply runs past them: chat_load's count prompts), and its user's next turn arrives then.

Every request reaches the engine S after it was sent: the server's own work on it (its chat template and the
tokenization of a 16K prompt in Python), taken from the run of 1 user when the results have one (its turns 2+: their
TTFT less their read and first token, which nothing else delays), else 0.

A turn reads its prompt less the previous turn's prompt (the engine's checkpoint at that prompt's end, the fake run with
enough --checkpoints for every user), the first turn all of it. Every user arrives at 0, in the order of their
measured first tokens. The summaries are chat_load.summary's over those rows; the queue wait is the server's
strata_queue_ms, 0 with "parallel" (no FIFO). The server's tokenization of each 16K prompt and the stream are what the
10% covers, as in c01_expect.py.

    c03_expect.py RESULTS.json --prompt-ms P --decode-ms D --row-ms R --slots N
    c03_expect.py STEADY.json --steady --decode-ms D --row-ms R

Each run's measured summary against the model's: the per-user decode tok/s (median and slowest) and the aggregate
tok/s within 10% (the acceptance bound), the TTFT percentiles printed beside them. Exit 1 when one is off.
--steady (chat_load.py steady): in the window every step has the N users' rows, so each user
gets a token every D + R (N - 1) and the aggregate is N times that rate; every repeat's aggregate and slowest and
fastest user within 10%.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_load  # noqa: E402
from c01_expect import near  # noqa: E402

KEYS = (("decode_tok_s_user_median", False), ("decode_tok_s_user_min", False), ("aggregate_tok_s", False))
# printed beside them, not judged (acceptance 2 is the tok/s): a turn 2's TTFT is a few hundred ms of the server's
# threads and the stream that the model leaves out, and the p50 of N first turns and N second turns falls between them
INFO = ("ttft_p50_s", "ttft_p95_s")


def simulate(run: dict, p_s: float, d_s: float, r_s: float, n_slots: int, s_s: float = 0.0) -> tuple[list[dict], float]:
    """The run's rows as the batched model has them, and its wall time (the last reply's end)."""
    rows = {(r["user"], r["turn"]): r for r in run["rows"]}
    turns = max(r["turn"] for r in run["rows"])
    users = sorted({r["user"] for r in run["rows"]}, key=lambda u: rows[(u, 1)]["ttft_s"])

    def read_of(u, k):
        prev = rows[(u, k - 1)]["prompt_tokens"] if k > 1 else 0
        return rows[(u, k)]["prompt_tokens"] - prev

    now = 0.0
    waiting = [(s_s, i, u, 1) for i, u in enumerate(users)]   # (arrival at the engine, order, user, turn)
    slots = {}            # (user, turn) -> {"left": tokens still to generate, "first": its first token's time}
    out = []

    def finish(key, first, end):
        u, k = key
        gen = rows[key]["completion_tokens"]
        out.append({"user": u, "turn": k, "ttft_s": first - arrivals[key], "queue_s": 0.0, "completion_tokens": gen,
                    "decode_s": end - first, "short": rows[key]["short"]})
        if k < turns:
            waiting.append((end + s_s, len(users) + len(out), u, k + 1))
            arrivals[(u, k + 1)] = end

    arrivals = {(u, 1): 0.0 for u in users}   # when each request was sent
    while waiting or slots:
        waiting.sort()
        arrived = [w for w in waiting if w[0] <= now]
        if not slots and len(arrived) == 1:
            # alone: it runs solo until another request waits
            a, _, u, k = arrived[0]
            waiting.remove(arrived[0])
            key = (u, k)
            gen = rows[key]["completion_tokens"]
            t = max(now, a) + read_of(u, k) * p_s
            first = t + d_s
            produced, t = 1, first
            nxt = min((w[0] for w in waiting), default=None)
            while produced < gen and (nxt is None or t < nxt):
                t += d_s
                produced += 1
            now = t
            if produced >= gen:
                finish(key, first, t)
                continue
            # another one waits: the alone one goes into a slot, its next token read back (one token) and picked
            now += p_s + d_s
            produced += 1
            slots[key] = {"left": gen - produced, "first": first}
            if slots[key]["left"] <= 0:
                del slots[key]
                finish(key, first, now)
            continue
        if arrived and len(slots) < n_slots:
            # an admission: the read and the first token, the slots waiting
            a, _, u, k = arrived[0]
            waiting.remove(arrived[0])
            key = (u, k)
            now = max(now, a) + read_of(u, k) * p_s + d_s
            gen = rows[key]["completion_tokens"]
            if gen <= 1:
                finish(key, now, now)
            else:
                slots[key] = {"left": gen - 1, "first": now}
            continue
        if slots:
            now += d_s + r_s * (len(slots) - 1)
            for key in list(slots):
                slots[key]["left"] -= 1
                if slots[key]["left"] <= 0:
                    first = slots.pop(key)["first"]
                    finish(key, first, now)
            continue
        now = min(w[0] for w in waiting)   # idle until the next arrival
    return out, now


def steady(j: dict, d_ms: float, r_ms: float) -> int:
    """--steady: each repeat's window against N rows a step; the number of figures off."""
    n = j["users"]
    user = 1000 / (d_ms + r_ms * (n - 1))
    bad = 0
    print(f"c03_expect.py --steady: {n} users, a step of {n} rows every {d_ms + r_ms * (n - 1):g} ms: {user:.2f} tok/s a "
          f"user, {n * user:.2f} in aggregate")
    for k, r in enumerate(j["repeats"], 1):
        if "aggregate_tok_s" not in r:
            print(f"FAIL repeat {k}: no window ({'; '.join(x.get('error', '') for x in r['rows'] if 'error' in x)})")
            bad += 1
            continue
        for key, want in (("aggregate_tok_s", n * user), ("user_tok_s_min", user), ("user_tok_s_max", user)):
            ok = near(r[key], round(want, 2), False)
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} repeat {k} {key}: measured {r[key]}, expected {want:.2f}")
    print(f"c03_expect.py: {'PASS' if not bad else f'FAIL ({bad})'}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--prompt-ms", type=float)
    ap.add_argument("--decode-ms", type=float, required=True)
    ap.add_argument("--row-ms", type=float, required=True)
    ap.add_argument("--slots", type=int)
    ap.add_argument("--steady", action="store_true", help="a chat_load.py steady file")
    a = ap.parse_args()
    if a.steady:
        return steady(json.loads(a.results.read_text()), a.decode_ms, a.row_ms)
    if a.prompt_ms is None or a.slots is None:
        ap.error("--prompt-ms and --slots (without --steady)")
    runs = json.loads(a.results.read_text())["runs"]
    p_s, d_s = a.prompt_ms / 1000, a.decode_ms / 1000
    one = next((r for r in runs if r["users"] == 1), None)
    s_s = 0.0
    if one:
        by = {r["turn"]: r for r in one["rows"]}
        later = sorted(by[k]["ttft_s"] - (by[k]["prompt_tokens"] - by[k - 1]["prompt_tokens"]) * p_s - d_s
                       for k in by if k > 1)
        s_s = max(0.0, later[len(later) // 2]) if later else 0.0
    print(f"c03_expect.py: the server's work on a request before the engine sees it: {s_s * 1000:.0f} ms (the run of 1 "
          f"user's turns 2+)" if one else "c03_expect.py: no run of 1 user: the server's work on a request taken as 0")
    bad = 0
    for run in runs:
        if any("error" in r for r in run["rows"]):
            print(f"FAIL N={run['label']}: a request failed")
            bad += 1
            continue
        sim, wall = simulate(run, p_s, d_s, a.row_ms / 1000, a.slots, s_s)
        want, got = chat_load.summary(sim, wall), run["summary"]
        for key, seconds in KEYS:
            ok = near(got[key], want[key], seconds)
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} N={run['label']} {key}: measured {got[key]}, expected {want[key]}")
        for key in INFO:
            print(f"info N={run['label']} {key}: measured {got[key]}, the model's {want[key]}")
    print(f"c03_expect.py: {'PASS' if not bad else f'FAIL ({bad})'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
