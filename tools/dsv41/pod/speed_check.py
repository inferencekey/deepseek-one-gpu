#!/usr/bin/env python3
"""The qengine speed check and the qfork one (--fork):
right after the continuation, the server's decode speed on this machine from its log, so a session on a machine that
decodes far below the measured speed ends within a minute or two of serving instead of spending its cap for nothing.

  speed_check.py LOG [--fork] [--requests N] [--max-ms X] [--machine M] [--machine-source S]

LOG is the server's engine log; its requests are parsed by qdiag_report.py's engine_requests (the same decode lines
the qdiag report reads). The check takes the first N requests (default: every request in LOG; in the session that is
the continuation's 3) that decoded at least one step, and prints the requests and decode steps it used, the median
over them of ms per decode step and of doorbell wait per step (the host waiting for the GPU's ring: the GPU's front
work), the machine id and where it came from, and the threshold: --max-ms, else the env
Q_SPEED_MAX_MS, else 30 ms per decode step (qdiag's continuation 20.2 on a normal RTX PRO 6000 WS machine, a slow
qengine machine's 44.1-45.8, the e07b plan's short-adaptive 15.9).
Exit 0 at or under the threshold; above it exit 1, the last line
  stop: engine too slow on this machine: X ms per decode step (machine M)
which e07-run.sh writes to out/steps.log as is (a runner greps it); exit 2 when LOG has no decode line to measure.

--fork: LOG is the fork's out/quality-server.log; its replies are llama-server's print_timing lines
  "eval time = T ms / N tokens ( X ms per token, ...)" (not the "prompt eval time" ones), each reply's X with N >= 1.
The check takes the first N replies (default every one; in the session the continuation's 3), prints their tasks,
tokens and ms per token, the median ms per generated token over them, the machine and the threshold: --max-ms, else
the env Q_FORK_MAX_MS, else 80 ms per generated token (12.5 tok/s; t04a's stock baseline on the A100 SXM4 52.2-52.9,
18.92-19.15 tok/s; the 2026-10-11 qfork session at -np 4 on an 18 vCPU A100 SXM4 host: its 3 continuation replies,
one at a time, 69.2-71.4, its lm-eval replies at 4 concurrent 334-1002, median 578 over all 152). Above it exit 1
with the last line
  stop: fork too slow on this machine: X ms per token (machine M)
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qdiag_report import engine_requests, med, read  # noqa: E402

DEFAULT_MAX_MS = 30.0
STOP = "stop: engine too slow on this machine: {:.1f} ms per decode step (machine {})"
FORK_MAX_MS = 80.0
FORK_STOP = "stop: fork too slow on this machine: {:.1f} ms per token (machine {})"
# llama-server's print_timing line of a reply's decode (tools/server/server-context.cpp); "prompt eval time" is the
# prompt's and does not match
FORK_EVAL = re.compile(r"\| task (\d+) \|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(\s*([\d.]+) ms per token")


def fork_replies(text):
    """Each reply's decode in llama-server's log: its task id, generated tokens and ms per token."""
    return [{"task": int(m[1]), "tokens": int(m[3]), "ms_per_token": float(m[4])}
            for m in FORK_EVAL.finditer(text) if int(m[3]) >= 1]


def fork_check(a):
    max_ms = a.max_ms if a.max_ms is not None else float(os.environ.get("Q_FORK_MAX_MS") or FORK_MAX_MS)
    used = fork_replies(read(a.log))
    print(f"fork speed check of {a.log}: {len(used)} replies in the log")
    if a.requests is not None:
        used = used[:a.requests]
    print(f"machine: {a.machine} ({a.machine_source})")
    print(f"threshold: {max_ms:g} ms per generated token")
    if not used:
        print("speed_check.py: no llama-server reply (print_timing 'eval time') to measure", file=sys.stderr)
        return 2
    ms = med(r["ms_per_token"] for r in used)
    print(f"used: replies of tasks {','.join(str(r['task']) for r in used)} ({len(used)}), "
          f"{sum(r['tokens'] for r in used)} generated tokens; ms per token per reply "
          f"{', '.join('%.1f' % r['ms_per_token'] for r in used)}")
    print(f"median: {ms:.1f} ms per generated token ({1000 / ms:.1f} tok/s)")
    if ms > max_ms:
        print(FORK_STOP.format(ms, a.machine))
        return 1
    print(f"pass: {ms:.1f} ms per generated token, at most {max_ms:g}")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("log")
    ap.add_argument("--fork", action="store_true", help="LOG is llama-server's: ms per generated token of its replies")
    ap.add_argument("--requests", type=int, default=None,
                    help="the first N requests with decode steps, or with --fork replies (default all)")
    ap.add_argument("--max-ms", type=float, default=None,
                    help=f"default Q_SPEED_MAX_MS, else {DEFAULT_MAX_MS:g}; with --fork Q_FORK_MAX_MS, else "
                    f"{FORK_MAX_MS:g}")
    ap.add_argument("--machine", default="unknown")
    ap.add_argument("--machine-source", default="not given")
    a = ap.parse_args(argv)
    if a.fork:
        return fork_check(a)
    max_ms = a.max_ms if a.max_ms is not None else float(os.environ.get("Q_SPEED_MAX_MS") or DEFAULT_MAX_MS)
    reqs = engine_requests(read(a.log))
    used = [r for r in reqs if r.get("steps")]
    if a.requests is not None:
        used = used[:a.requests]
    print(f"speed check of {a.log}: {len(reqs)} requests in the log")
    print(f"machine: {a.machine} ({a.machine_source})")
    print(f"threshold: {max_ms:g} ms per decode step")
    if not used:
        print("speed_check.py: no request with decode steps to measure", file=sys.stderr)
        return 2
    ms = med(r["ms_per_step"] for r in used)
    print(f"used: requests {','.join(str(r['n']) for r in used)} ({len(used)}), {sum(r['steps'] for r in used)} "
          f"decode steps; ms per step per request {', '.join('%.1f' % r['ms_per_step'] for r in used)}")
    print(f"median: {ms:.1f} ms per decode step, doorbell waits {med(r['doorbell_ms_per_step'] for r in used):.1f} ms "
          "per step")
    if ms > max_ms:
        print(STOP.format(ms, a.machine))
        return 1
    print(f"pass: {ms:.1f} ms per decode step, at most {max_ms:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
