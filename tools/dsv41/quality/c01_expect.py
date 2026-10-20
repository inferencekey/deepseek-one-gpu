#!/usr/bin/env python3
"""tools/dsv41/quality/c01_expect.py - what chat_load.py should measure against the fake engine, by hand from its
delays (c01-test.sh).

The FIFO server answers one request at a time, in the order they reach it. fakes.py's engine reads a request's new
prompt tokens (its prompt less the cached ones the server reports) at FAKE_ENGINE_PROMPT_MS_PER_TOKEN each, then
writes every token FAKE_ENGINE_DECODE_MS_PER_TOKEN after the one before. So a request that starts at s (the engine
free and the request arrived) has its first token at s + read x P + D and ends at s + read x P + generated x D; its
queue wait is s less its arrival; a user's next turn arrives when its reply ends. Every user arrives at 0, in the
order the server took their first turns (the order of their measured first tokens). The run's summaries are
chat_load.summary's over those rows. Overheads the model leaves out (the server's tokenization of each 16K prompt,
about 0.13 s on the dev machine, and the stream) are what the 10% covers.

    c01_expect.py RESULTS.json --prompt-ms P --decode-ms D [--fifo N]

Each run's measured summary against the model's: every TTFT and queue percentile, the per-user decode tok/s and the
aggregate tok/s within 10% (or 0.05 s for a time under 0.5 s, a queue wait that should be 0). --fifo N: the run of N
users also has each first turn's TTFT, in order, within 10% of k requests ahead (their read and decode) plus its own
read. Exit 1 when one is off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_load  # noqa: E402

TOLERANCE = 0.10
FLOOR_S = 0.05
KEYS = (("ttft_p50_s", True), ("ttft_p95_s", True), ("queue_p50_s", True), ("queue_p95_s", True),
        ("decode_tok_s_user_median", False), ("decode_tok_s_user_min", False), ("aggregate_tok_s", False))


def simulate(run: dict, p_s: float, d_s: float) -> tuple[list[dict], float]:
    """The run's rows as the FIFO model has them, and its wall time (the last reply's end)."""
    rows = {(r["user"], r["turn"]): r for r in run["rows"]}
    turns = max(r["turn"] for r in run["rows"])
    first = sorted({r["user"] for r in run["rows"]}, key=lambda u: rows[(u, 1)]["ttft_s"])
    pending = [(0.0, i, u, 1) for i, u in enumerate(first)]
    free, out = 0.0, []
    while pending:
        pending.sort()
        arrival, i, u, k = pending.pop(0)            # FIFO: the earliest arrival goes next
        r = rows[(u, k)]
        read, gen = r["prompt_tokens"] - r["cached_tokens"], r["completion_tokens"]
        start = max(free, arrival)
        free = start + read * p_s + gen * d_s
        out.append({"user": u, "turn": k, "ttft_s": start + read * p_s + d_s - arrival, "queue_s": start - arrival,
                    "completion_tokens": gen, "decode_s": (gen - 1) * d_s, "short": r["short"], "read": read})
        if k < turns:
            pending.append((free, i, u, k + 1))
    return out, free


def near(measured, expected, seconds: bool) -> bool:
    if measured is None or expected is None:
        return False
    floor = FLOOR_S if seconds and abs(expected) < 0.5 else 0.0
    return abs(measured - expected) <= max(TOLERANCE * abs(expected), floor)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--prompt-ms", type=float, required=True)
    ap.add_argument("--decode-ms", type=float, required=True)
    ap.add_argument("--fifo", type=int, default=0)
    a = ap.parse_args()
    p_s, d_s = a.prompt_ms / 1000, a.decode_ms / 1000
    runs = json.loads(a.results.read_text())["runs"]
    bad = int(bool(a.fifo) and not any(r["users"] == a.fifo for r in runs))
    if bad:
        print(f"FAIL --fifo {a.fifo}: {a.results} has no run of {a.fifo} users")
    for run in runs:
        if any("error" in r for r in run["rows"]):
            print(f"FAIL N={run['label']}: a request failed")
            bad += 1
            continue
        sim, wall = simulate(run, p_s, d_s)
        want, got = chat_load.summary(sim, wall), run["summary"]
        for key, seconds in KEYS:
            ok = near(got[key], want[key], seconds)
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} N={run['label']} {key}: measured {got[key]}, expected {want[key]}")
        if a.fifo and run["users"] == a.fifo:
            mine = sorted(r["ttft_s"] for r in run["rows"] if r["turn"] == 1)
            theirs = sorted((s for s in sim if s["turn"] == 1), key=lambda s: s["ttft_s"])
            for k, (m, s) in enumerate(zip(mine, theirs)):
                ahead, ok = s["queue_s"], near(m, s["ttft_s"], True)
                bad += not ok
                print(f"{'ok  ' if ok else 'FAIL'} N={run['label']} first turn {k + 1}: TTFT {m} s, expected "
                      f"{s['ttft_s']:.3f} s (the {k} ahead in the FIFO, their reads and replies {ahead:.3f} s, + its own "
                      f"read of {s['read']} tokens {s['read'] * p_s:.3f} s and its first token {d_s:.3f} s)")
    print(f"c01_expect.py: {'PASS' if not bad else f'FAIL ({bad})'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
