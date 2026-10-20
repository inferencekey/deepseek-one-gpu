#!/usr/bin/env python3
"""tools/dsv41/quality/c04_expect.py - what chat_load.py long should measure against the fake engine, by hand from its
delays (c04-test.sh). c03_expect.py is the batch's decode model.

fakes.py's engine with --batch and --interleave-chunk K (by default K is its --chunk, the prefill chunk, as the real
engine's; c04-test.sh runs it at 16384, e07-run.sh's) reads a prompt that comes while slots
decode in chunks of K tokens, each K x P (FAKE_ENGINE_PROMPT_MS_PER_TOKEN), with one batch step after each chunk but
the last, D + R (B - 1) for B decoding slots (FAKE_ENGINE_DECODE_MS_PER_TOKEN, FAKE_ENGINE_BATCH_ROW_MS). So:

  * a decoding user's largest gap between two tokens is one chunk and one step: min(K, L) P + D + R (N - 1) with the
    N users decoding (the long prompt's read), whatever the long prompt's length past K; with K 0 (no interleaving)
    it is the whole read, L P for the long prompt's L tokens;
  * the long request's TTFT alone is L P + D (its read whole, then its first token); beside the N users it is its
    chunks' reads, the same L P, plus a step after each chunk but the last, (ceil(L / K) - 1) (D + R (N - 1)), + D.

Every request also takes the server's own work before it reaches the engine (its chat template and the tokenization
of a 64K prompt in Python, about 0.3 s on the 12 vCPU dev machine) and the stream's delivery; the TTFTs are judged within 10%, the
gap within 10% + 50 ms (Python's sleeps and the SSE path on a 12 vCPU VM), and both are printed beside the model.

    c04_expect.py LONG.json --prompt-ms P --decode-ms D --row-ms R --chunk K

Exit 0 when the measured largest gap and both TTFTs are within the model's bounds, 1 otherwise; with --chunk 0 the
model is the whole read's (the check of what interleaving guards against: the gap past chat_load.GAP_MAX_S).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

GAP_TOL, GAP_SLACK_S, TTFT_TOL = 0.10, 0.05, 0.10


def model(n_users: int, long_tokens: int, p_ms: float, d_ms: float, r_ms: float, chunk: int) -> dict:
    step = d_ms + r_ms * (n_users - 1)
    gap = (min(chunk, long_tokens) * p_ms + step) if chunk else long_tokens * p_ms
    steps = math.ceil(long_tokens / chunk) - 1 if chunk else 0
    return {"step_ms": round(step, 3), "max_gap_s": round(gap / 1000, 4),
            "long_ttft_solo_s": round((long_tokens * p_ms + d_ms) / 1000, 4),
            "long_ttft_s": round((long_tokens * p_ms + steps * step + d_ms) / 1000, 4), "steps_between": steps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--prompt-ms", type=float, required=True)
    ap.add_argument("--decode-ms", type=float, required=True)
    ap.add_argument("--row-ms", type=float, required=True)
    ap.add_argument("--chunk", type=int, required=True, help="the engine's interleave chunk (its --chunk by default; 0: none)")
    a = ap.parse_args()
    j = json.loads(a.results.read_text())
    s = j["summary"]
    m = model(s["users"], s["long_prompt_tokens"], a.prompt_ms, a.decode_ms, a.row_ms, a.chunk)
    gap_bound = m["max_gap_s"] * (1 + GAP_TOL) + GAP_SLACK_S
    ok_gap = s["max_gap_s"] is not None and s["max_gap_s"] <= gap_bound
    ok_solo = s["long_ttft_solo_s"] is not None and abs(s["long_ttft_solo_s"] - m["long_ttft_solo_s"]) <= \
        TTFT_TOL * m["long_ttft_solo_s"]
    ok_long = s["long_ttft_s"] is not None and abs(s["long_ttft_s"] - m["long_ttft_s"]) <= TTFT_TOL * m["long_ttft_s"]
    print(f"c04_expect.py: {s['users']} users decoding, a long prompt of {s['long_prompt_tokens']} tokens, chunk "
          f"{a.chunk or 'none (read whole)'}, step {m['step_ms']} ms, {m['steps_between']} steps between the chunks")
    print(f"  largest gap: measured {s['max_gap_s']} s, model {m['max_gap_s']} s (bound {gap_bound:.4f}): "
          f"{'ok' if ok_gap else 'off'}; the bar's {j['gap_max_s']} s: {'PASS' if s['gap_pass'] else 'FAIL'}")
    print(f"  the long TTFT alone: measured {s['long_ttft_solo_s']} s, model {m['long_ttft_solo_s']} s: "
          f"{'ok' if ok_solo else 'off'}; beside the users: measured {s['long_ttft_s']} s, model {m['long_ttft_s']} s: "
          f"{'ok' if ok_long else 'off'} (measured {s['long_ttft_ratio']} x alone, model "
          f"{m['long_ttft_s'] / m['long_ttft_solo_s']:.3f} x)")
    ok = ok_gap and ok_solo and ok_long
    print(f"c04_expect.py: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
