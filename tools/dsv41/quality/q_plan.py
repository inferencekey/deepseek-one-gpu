#!/usr/bin/env python3
"""tools/dsv41/quality/q_plan.py - the quality steps of a quality pod plan and their estimates (tools/dsv41/pod/e07.sh,
E07_PLAN=qfork and qengine), from the tokens of the 2026-10-06 API run on the same subsets.

The usage log of that run (the vLLM-EXL3 experiment's data/controlled/E11-quality-reference/api/20261006T181137Z/api_usage.jsonl,
464 requests in run_lm_eval.sh's task order: gsm8k_cot_llama 150, ifeval 150, humaneval_instruct 164, the last
through /beta) gives each task's prompt and completion tokens. A task's minutes are its prompt tokens at the plan's
prompt rate plus its completion tokens at its decode rate, one request at a time (lm-eval's --num-concurrent, qengine's
2, may finish sooner; nothing here counts on it; qfork sends 1, the fork's one slot),
plus 0.5 for lm-eval's start; a needle's are its prompt tokens at the
long-prompt rate plus its 32 answer tokens at the decode rate, plus 0.2. Each printed line is e07.sh's
"NAME (what):MINUTES", in the order the pod runs them (e07-run.sh):

  <server>-server   the server's load (fork-server: llama-server, 6 min: the oracle run's cold page cache load; engine-server:
                    serve/server.py on dsv41-engine, 4.5 min: the measured engine load of 250 s and the server's start)
  quality-venv      the lm-eval venv, installed in the background since setup (0.5: the wait for it)
  continuation      the 3 HumanEval items of continuation.py (their mean tokens of the 2026-10-06 run), plus 0.5
  speed-check       tools/dsv41/pod/speed_check.py on the engine's log, its threshold
                    --speed-max-ms in the label; qfork: speed_check.py --fork on
                    llama-server's log, --fork-max-ms in the label; 0.1
  lm-eval-<task>    each task of --tasks, its NUM_CONCURRENT (--num-concurrent) in the label
  needle-<length>   each length of --needles
  server-stop       0.5

    q_plan.py --plan qfork|qengine --usage API_USAGE --needle NEEDLE_JSON --tasks "..." --needles "..." \\
        --prompt-tok-s X --long-prompt-tok-s Y --decode-tok-s Z --num-concurrent N [--speed-max-ms MS] \\
        [--fork-max-ms MS]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ORDER = [("gsm8k_cot_llama", 150), ("ifeval", 150), ("humaneval_instruct", 164)]
SERVER = {"qfork": ("fork-server", "llama-server's load; the oracle run's cold page cache load took 6 min", 6.0),
          "qengine": ("engine-server", "serve/server.py on dsv41-engine; a measured engine load of 250 s", 4.5)}


def task_tokens(usage: Path) -> dict[str, tuple[int, int]]:
    rows = [json.loads(x) for x in usage.read_text().splitlines() if x.strip()]
    if len(rows) != sum(n for _, n in ORDER):
        raise SystemExit(f"q_plan.py: {usage} has {len(rows)} requests, not the 464 of the subsets")
    out, i = {}, 0
    for task, n in ORDER:
        seg = rows[i:i + n]
        i += n
        if (task == "humaneval_instruct") != all("/beta/" in r["url"] for r in seg):
            raise SystemExit(f"q_plan.py: {usage}: the {task} requests are not where run_lm_eval.sh's order puts them")
        out[task] = (sum(r["usage"]["prompt_tokens"] for r in seg), sum(r["usage"]["completion_tokens"] for r in seg))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--plan", choices=sorted(SERVER), required=True)
    ap.add_argument("--usage", type=Path, required=True)
    ap.add_argument("--needle", type=Path, required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--needles", default="")
    ap.add_argument("--prompt-tok-s", type=float, required=True)
    ap.add_argument("--long-prompt-tok-s", type=float, required=True)
    ap.add_argument("--decode-tok-s", type=float, required=True)
    ap.add_argument("--num-concurrent", type=int, required=True, help="lm-eval's concurrent requests")
    ap.add_argument("--speed-max-ms", type=float, default=30.0)
    ap.add_argument("--fork-max-ms", type=float, default=80.0)
    a = ap.parse_args()
    tok = task_tokens(a.usage)
    tasks, needles = a.tasks.split(), a.needles.split()
    bad = [t for t in tasks if t not in tok]
    if bad or not tasks:
        raise SystemExit(f"q_plan.py: tasks {bad or tasks}: one or more of {', '.join(tok)}")
    prompts = {p["name"]: p for p in json.loads(a.needle.read_text())["prompts"]}
    bad = [n for n in needles if n not in prompts]
    if bad:
        raise SystemExit(f"q_plan.py: needles {bad}: {', '.join(prompts)}")
    pr, lr, dr = a.prompt_tok_s, a.long_prompt_tok_s, a.decode_tok_s

    def minutes(p, c, extra):
        return round((p / pr + c / dr) / 60 + extra, 1)

    name, what, load = SERVER[a.plan]
    hp, hc = tok["humaneval_instruct"]
    lines = [f"{name} ({what}):{load}", "quality-venv (the lm-eval venv, installed in the background since setup):0.5",
             f"continuation (3 HumanEval items, the 2026-10-06 mean of {hp // 164} prompt and {hc // 164} completion "
             f"tokens each):{minutes(3 * hp / 164, 3 * hc / 164, 0.5)}"]
    if a.plan == "qengine":
        lines.append(f"speed-check (speed_check.py on the continuation's requests, at most {a.speed_max_ms:g} ms per "
                     "decode step):0.1")
    else:
        lines.append(f"speed-check (speed_check.py --fork on the continuation's replies, at most {a.fork_max_ms:g} ms "
                     "per generated token):0.1")
    for t in tasks:
        p, c = tok[t]
        lines.append(f"lm-eval-{t} ({dict(ORDER)[t]} items, NUM_CONCURRENT {a.num_concurrent}: {p:,} prompt tokens at "
                     f"{pr:g} tok/s, {c:,} completion at {dr:g} tok/s, + 0.5):{minutes(p, c, 0.5)}")
    for n in needles:
        p = prompts[n]["prompt_tokens"]
        lines.append(f"needle-{n} ({p:,} prompt tokens at {lr:g} tok/s, 32 at {dr:g} tok/s, + 0.2):"
                     f"{round((p / lr + 32 / dr) / 60 + 0.2, 1)}")
    lines.append("server-stop:0.5")
    print("\n".join(lines))
    tp, tc = sum(p for p, _ in tok.values()), sum(c for _, c in tok.values())
    sp, sc = sum(tok[t][0] for t in tasks), sum(tok[t][1] for t in tasks)
    print(f"tokens of the 2026-10-06 API run: {tp:,} prompt and {tc:,} completion over 464 requests; this plan's "
          f"tasks {sp:,} and {sc:,}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
