#!/usr/bin/env python3
"""tools/dsv41/quality/q_report.py - the session sheets of the quality runs (every parameter as run, the cost and a
source for every number), in SHEET_ROOT (e07.sh's default: $BUILD_ROOT/sheets).

  q_report.py api RUN_DIR SHEET_ROOT     the API run of api_run.sh: SHEET_ROOT/<UTC date>-qapi/README.md and logs/
                                         (RUN_DIR's files: lm-eval's results and samples, the usage log, the price
                                         page, cost.json, analysis.md, the needle results); prints the sheet's path

results_section(RUN_DIR, COLUMN) is also the quality part of the pod sheets (tools/dsv41/pod/e07_report.py,
E07_PLAN=qfork and qengine; COLUMN is analyze.py's: api_fresh, fork or engine): eval.sh's settings, lm-eval's
version, each task's headline (analyze.py's metrics) and status, the run beside the API on 2026-10-06 (analyze.py),
the needle results, the 3 HumanEval items for the hand check of the continuation (the request's partial message
ends, the reply follows), and the usage totals.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import quality as Q  # noqa: E402


def read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def table(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def cell(s: str, n: int = 60) -> str:
    s = s.replace("\n", "\\n").replace("|", "\\|")
    return s[:n] + ("..." if len(s) > n else "")


def handcheck(run: Path) -> list[str]:
    """The first 3 HumanEval items of the run: the end of the partial message lm-eval sent, then the reply."""
    files = sorted(run.rglob("samples_humaneval_instruct_*.jsonl"))
    if not files:
        return ["No humaneval_instruct samples in this run."]
    rows = sorted((json.loads(x) for x in files[-1].read_text().splitlines()), key=lambda r: r["doc_id"])[:3]
    out = ["The first 3 HumanEval items, for the hand check of the continuation: the reply must continue "
           "the function after its docstring, not start the message or the code block again.", ""]
    for r in rows:
        msgs = json.loads(r["arguments"]["gen_args_0"]["arg_0"][0])
        tail = "\n".join(msgs[-1]["content"].rstrip("\n").splitlines()[-3:])
        reply = r["resps"][0][0]
        out += [f"- {r['doc']['task_id']} (doc_id {r['doc_id']}, pass@1 {r.get('pass@1')}): the partial message ends",
                "", "  ```", "  " + tail.replace("\n", "\n  "), "  ```", "", "  and the reply starts", "",
                "  ```", "  " + "\n".join(reply.splitlines()[:8]).replace("\n", "\n  "), "  ```", ""]
    return out


def results_section(run: Path, column: str) -> str:
    run = Path(run)
    out = []
    settings = read(run / "settings.txt").strip()
    freeze = re.search(r"^lm[-_]eval==(\S+)", read(run / "lm_eval_pip_freeze.txt"), re.M)
    out += ["Settings (eval.sh's settings.txt, one block per eval.sh call):", "", "```", settings or "none", "```", "",
            f"lm-eval {freeze.group(1) if freeze else 'n/a'} (lm_eval_pip_freeze.txt), seeds 1234, temperature 0, "
            "thinking off, --log_samples, max_gen_toks gsm8k 1024, ifeval 1280, humaneval 1024 (run_lm_eval.sh).", ""]
    status = read(run / "lm_eval_status.txt").strip()
    found = Q.find_results(run)
    rows = []
    for task, (metric, filt) in analyze.TASKS.items():
        v = Q.headline(found[task]["metrics"], metric, filt) if task in found else (None, None)
        n = len(Q.load_samples(run, task, metric, filt))
        st = [x for x in status.splitlines() if x.startswith(task + " ")]
        rows.append((task, f"{metric} ({filt})", f"{v[0]:.4f}" if v[0] is not None else "n/a",
                     f"{v[1]:.4f}" if v[1] is not None else "n/a", n, st[-1] if st else "not run",
                     os.path.relpath(found[task]["file"], run) if task in found else ""))
    out += [table(["task", "metric", "value", "stderr", "items", "status", "results file"], rows), ""]
    if found:
        runs = {"api_fresh": None, "fork": None, "engine": None, "api_then": analyze.API_THEN,
                "exl3": analyze.EXL3_THEN, column: run}
        out += [f"Beside the API on 2026-10-06 and the EXL3 stack (analyze.py, this run as "
                f"{dict(analyze.COLUMNS)[column]}; the pairs of runs not in this session are filled in by analyze.py "
                "once all of them exist):", "",
                analyze.markdown(analyze.analyze(runs)), ""]
    needles = sorted(run.glob("needle-*.json"))
    if needles:
        results = sorted((x for f in needles for x in json.loads(f.read_text())["results"]), key=lambda x: x["context"])
        nrows = [(x["name"], x["context"], x["prompt_tokens"], x["status"], x["seconds"], cell(x["reply"], 40),
                  x["answer"], "PASS" if x["exact"] else "FAIL") for x in results]
        out += ["Needle (needle_check.py: the number at depth 0.5, exact match):", "",
                table(["length", "context", "prompt tokens", "status", "s", "reply", "expected", "exact"], nrows), ""]
    usage = [json.loads(x) for x in read(run / "api_usage.jsonl").splitlines() if x.strip()]
    if usage:
        us = [u["usage"] for u in usage if u.get("usage")]
        out += [f"Usage (api_usage.jsonl, the proxy's log): {len(usage)} requests, "
                f"{sum(u['status'] == 200 for u in usage)} answered 200, {sum(u['prompt_tokens'] for u in us)} prompt "
                f"and {sum(u['completion_tokens'] for u in us)} completion tokens.", ""]
    return "\n".join(out + handcheck(run))


def copy_logs(src: Path, logs: Path) -> None:
    for f in src.rglob("*"):
        if f.is_file():
            dst = logs / f.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)


def api_sheet(run: Path, sheet_root: Path) -> Path:
    stamp = run.name
    sheet = sheet_root / f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}-qapi"
    marker = sheet / "logs" / "run-dir.txt"
    if marker.exists() and read(marker).strip() != str(run.resolve()):
        sheet = Path(f"{sheet}-{stamp[9:13]}")
    copy_logs(run, sheet / "logs")
    (sheet / "logs" / "run-dir.txt").write_text(str(run.resolve()) + "\n")
    cost = json.loads(read(run / "cost.json") or "{}")
    page = read(run / "price-page.txt").strip()
    doc = [f"# The DeepSeek API, fresh run ({stamp})", "",
           "`tools/dsv41/quality/api_run.sh` from a dev machine: model deepseek-flash (DeepSeek-V4.1-Flash per the price page), thinking "
           "disabled, HumanEval through chat prefix completion (/beta), 32 concurrent requests.", "",
           "## Cost", "",
           (f"{cost['usd']:.4f} USD measured: {cost['requests']} requests ({cost['peak_requests']} at peak), "
            f"{cost['input_miss']} cache-miss and {cost['input_hit']} cache-hit input tokens, {cost['output']} output "
            f"tokens, priced with the page fetched for the run (api_cost.py; logs/cost.json, logs/price-page.html: "
            f"{page})." if cost else "no cost.json"), "",
           "## Results", "", results_section(run, "api_fresh"), ""]
    (sheet / "README.md").write_text("\n".join(doc))
    return sheet


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != "api":
        print(__doc__.strip(), file=sys.stderr)
        return 2
    print(api_sheet(Path(argv[1]), Path(argv[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
