#!/usr/bin/env python3
"""tools/dsv41/quality/analyze_test.py - the quality bar of analyze.py as amended on 2026-10-12 (engine - API fresh,
one-sided), on small synthetic lm-eval runs written here (no model, no network, a few seconds).

Each run is a RUN_DIR as run_lm_eval.sh writes it: lm_eval/<task>/<model>/results_*.json with the task's headline
and samples_<task>_*.jsonl with one line per item and filter (gsm8k_cot_llama also gets a strict-match line holding
the opposite score, so a filter mix-up shows). The API run scores 135 of 150 items on each task (0.900). Cases:

  a  PASS: the engine loses one item and gains another on each task (diff 0)
  b  FAIL, a loss over 3 points: ifeval loses 12 items and gains 6 (diff -0.040), its interval still includes zero
  c  FAIL, the interval lies below zero: humaneval_instruct loses 4 items (diff -0.027, inside the 3 points)
  d  not evaluated: no engine run, then an engine run without humaneval_instruct
  e  no --fork: the fork column and the pairs that need it read "not run (dropped 2026-10-12)"; with --fork the
     engine - fork pair is computed and called information, while the quality bar stays on engine - API fresh; with --fork
     and no --engine that remark is not printed
  f  PASS with a flag, a significant gain: humaneval_instruct gains 4 items (diff +0.027, interval above zero)
  g  FAIL, a small loss below zero: on a 300-item API run (270 right) ifeval loses 6 items (diff -0.020, interval
     below zero); against case c's run, 3 losses (diff -0.020) give an upper bound of exactly 0.000, which counts as
     including zero, so that one is a PASS
  h  a fork run of HumanEval only: the fork column has humaneval_instruct and plain
     "not run" for gsm8k_cot_llama and ifeval, so do their engine - fork and fork - API fresh pairs; engine - fork on
     humaneval_instruct is read two-sided as information: an interval that includes zero within 3 points (+0.013),
     one below zero with a loss over 3 points (-0.053, with case c's engine), one above zero with a gain over 3
     points (+0.060); the quality bar's verdict and reasons are those of the same engine without --fork
  i  --engine-ref: the batched engine against the single-user engine run per task,
     two-sided: PASS when each task's interval includes zero within 3 points (case a's engine against itself, diff 0;
     case g2's -0.020 touching zero); FAIL on a loss over 3 points (case b's ifeval), an interval below zero (case c's
     humaneval_instruct) or above it (case f's gain, which the API pair only flags); not evaluated without the
     reference's humaneval_instruct; the API pair's quality bar line unchanged beside it

Every case runs the real command line (analyze.py with --out and --json, --api-then and --exl3 pointed at the API
run so the vLLM-EXL3 reference checkout is not needed) and checks the markdown and the JSON.

    python3 tools/dsv41/quality/analyze_test.py

Exits 1 if any check fails.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

N = 150
BASE = [1.0] * 135 + [0.0] * 15
failed = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failed.append(name)


def write_run(d: Path, scores: dict[str, list[float]]) -> Path:
    """A synthetic RUN_DIR with one results and one samples file per task in `scores` (task -> per-item 0/1)."""
    for task, v in scores.items():
        metric, filt = A.TASKS[task]
        t = d / "lm_eval" / task / "synthetic"
        t.mkdir(parents=True)
        (t / "results_2026-10-12T00-00-00.json").write_text(json.dumps(
            {"results": {task: {f"{metric},{filt}": sum(v) / len(v), f"{metric}_stderr,{filt}": 0.01}}}))
        lines = [{"doc_id": i, "filter": filt, metric: x} for i, x in enumerate(v)]
        if task == "gsm8k_cot_llama":
            lines += [{"doc_id": i, "filter": "strict-match", metric: 1.0 - x} for i, x in enumerate(v)]
        (t / f"samples_{task}_2026-10-12T00-00-00.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return d


def moved(losses: int, gains: int) -> list[float]:
    """BASE with its first `losses` passes failed and its first `gains` failures passed."""
    v = list(BASE)
    for i in range(losses):
        v[i] = 0.0
    for i in range(gains):
        v[135 + i] = 1.0
    return v


def run(tmp: Path, name: str, **cols: Path) -> tuple[str, dict, str]:
    """analyze.py's command line on the columns given; returns stdout, the JSON and the --out file."""
    out, js = tmp / f"{name}.md", tmp / f"{name}.json"
    args = [sys.executable, str(HERE / "analyze.py"), "--api-then", str(cols["api_fresh"]), "--exl3",
            str(cols["api_fresh"]), "--out", str(out), "--json", str(js)]
    for k, d in cols.items():
        args += [f"--{k.replace('_', '-')}", str(d)]
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"analyze_test.py: {name}: analyze.py exited {p.returncode}: {p.stderr}")
    return p.stdout, json.loads(js.read_text()), out.read_text()


def item4_line(md: str) -> str:
    return next((x for x in md.splitlines() if x.startswith("Quality bar (")), "")


def pair(res: dict, a: str, b: str, task: str) -> dict | None:
    return next(p["result"] for p in res["pairs"] if (p["a"], p["b"], p["task"]) == (a, b, task))


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        api = write_run(tmp / "api", {task: BASE for task in A.TASKS})
        pair_json = {"a": "api_fresh", "b": "engine", "name": "engine - API fresh"}

        print("--- a. PASS")
        eng = write_run(tmp / "engine-a", {task: moved(1, 1) for task in A.TASKS})
        md, res, out = run(tmp, "a", api_fresh=api, engine=eng)
        line = item4_line(md)
        check("a: stdout and --out are the same table", md == out)
        check("a: the verdict is PASS, from engine - API fresh", res["item4"]["verdict"] == "PASS" and
              res["item4"]["pair"] == pair_json and "**PASS**" in line, line)
        check("a: the verdict line names the pair, the 2026-10-12 amendment and the Q2_K loss it holds",
              line.startswith("Quality bar (engine - API fresh, since 2026-10-12:") and "Q2_K quantization loss" in line
              and "without a fork run" in line, line)
        check("a: each task's engine - API fresh diff is 0 with 2 flips (the strict-match lines ignored)",
              all(pair(res, "api_fresh", "engine", task)["diff"] == 0 and
                  pair(res, "api_fresh", "engine", task)["discordant"] == 2 for task in A.TASKS))

        print("--- b. FAIL, a loss over 3 points")
        eng = write_run(tmp / "engine-b", {"gsm8k_cot_llama": BASE, "ifeval": moved(12, 6), "humaneval_instruct": BASE})
        md, res, _ = run(tmp, "b", api_fresh=api, engine=eng)
        r = pair(res, "api_fresh", "engine", "ifeval")
        check("b: ifeval diff -0.040 with an interval that includes zero (so only the 3 points fail it)",
              round(r["diff"], 3) == -0.04 and r["ci95"][0] <= 0 <= r["ci95"][1], json.dumps(r))
        reasons = res["item4"]["reasons"]
        want = f"ifeval: diff -0.040, 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], a loss over 3 points"
        check("b: FAIL naming ifeval only, with its diff, interval and reason", res["item4"]["verdict"] == "FAIL" and
              reasons == [want] and f"**FAIL** ({want})." in item4_line(md), json.dumps(reasons))

        print("--- c. FAIL, the interval lies below zero")
        eng = write_run(tmp / "engine-c", {"gsm8k_cot_llama": BASE, "ifeval": BASE, "humaneval_instruct": moved(4, 0)})
        md, res, _ = run(tmp, "c", api_fresh=api, engine=eng)
        r = pair(res, "api_fresh", "engine", "humaneval_instruct")
        check("c: humaneval_instruct diff -0.027 (inside the 3 points) with an interval below zero",
              round(r["diff"], 3) == -0.027 and r["diff"] >= -A.MAX_LOSS and r["ci95"][1] < 0, json.dumps(r))
        want = (f"humaneval_instruct: diff -0.027, 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], "
                "the interval lies below zero")
        check("c: FAIL naming humaneval_instruct only, with its diff, interval and reason",
              res["item4"]["verdict"] == "FAIL" and res["item4"]["reasons"] == [want] and want in item4_line(md),
              json.dumps(res["item4"]["reasons"]))

        print("--- d. not evaluated")
        md, res, _ = run(tmp, "d1", api_fresh=api)
        check("d: without the engine: not evaluated, the engine run named as missing",
              res["item4"]["verdict"] == "not evaluated" and res["item4"]["reasons"] == ["no engine run"] and
              res["item4"]["pair"] == pair_json and "**not evaluated** (no engine run)." in item4_line(md),
              json.dumps(res["item4"]))
        eng = write_run(tmp / "engine-d", {"gsm8k_cot_llama": BASE, "ifeval": BASE})
        md, res, _ = run(tmp, "d2", api_fresh=api, engine=eng)
        check("d: an engine run without humaneval_instruct: not evaluated, the task named",
              res["item4"]["verdict"] == "not evaluated" and
              res["item4"]["reasons"] == ["humaneval_instruct: fewer than 2 items answered by both"],
              json.dumps(res["item4"]))

        print("--- e. the fork column")
        eng = write_run(tmp / "engine-e", {task: moved(1, 1) for task in A.TASKS})
        md, res, _ = run(tmp, "e1", api_fresh=api, engine=eng)
        rows = md.splitlines()
        heads = [x for x in rows if x.startswith(tuple(f"| {task} |" for task in A.TASKS))]
        check("e: no --fork: the fork column reads 'not run (dropped 2026-10-12)' on every task",
              rows[0].split(" | ")[3] == "fork" and len(heads) == 3 and
              all(x.split(" | ")[3] == A.FORK_NOT_RUN for x in heads), "\n".join(heads))
        fork_rows = [x for x in rows if x.startswith(("| engine - fork |", "| fork - API fresh |"))]
        check("e: no --fork: engine - fork and fork - API fresh are listed per task as 'not run (dropped 2026-10-12)'",
              len(fork_rows) == 6 and all(x.split(" | ")[2] == A.FORK_NOT_RUN for x in fork_rows), "\n".join(fork_rows))
        check("e: no --fork: the engine column of a missing run still reads plain 'not run'",
              all(x.split(" | ")[4] == "not run" for x in run(tmp, "e2", api_fresh=api)[0].splitlines()
                  if x.startswith("| ifeval |")))
        # a fork the engine beats by 6 points (an interval above zero): engine - fork would fail the bar, the quality bar must not
        fork = write_run(tmp / "fork", {task: moved(9, 0) for task in A.TASKS})
        md, res, _ = run(tmp, "e3", api_fresh=api, fork=fork, engine=eng)
        r = pair(res, "fork", "engine", "ifeval")
        check("e: --fork: the fork column has its headlines and engine - fork is computed",
              "| ifeval | prompt_level_strict_acc (none) | 0.900 | 0.840 | 0.900 |" in md and r is not None and
              round(r["diff"], 3) == 0.06 and pair(res, "api_fresh", "fork", "ifeval") is not None, json.dumps(r))
        check("e: --fork: the quality bar stays PASS on engine - API fresh, engine - fork called information only",
              res["item4"]["verdict"] == "PASS" and res["item4"]["pair"] == pair_json and
              "engine - fork is information only, not the quality bar." in md.splitlines() and A.FORK_NOT_RUN not in md)
        md, res, _ = run(tmp, "e4", api_fresh=api, fork=fork)
        check("e: --fork without --engine: no 'information only' remark (there is no engine - fork pair)",
              "information only" not in md and res["item4"]["verdict"] == "not evaluated", md)

        print("--- f. PASS with a flag, a significant gain")
        eng = write_run(tmp / "engine-f", {"gsm8k_cot_llama": BASE, "ifeval": BASE,
                                           "humaneval_instruct": moved(0, 4)})
        md, res, _ = run(tmp, "f", api_fresh=api, engine=eng)
        r = pair(res, "api_fresh", "engine", "humaneval_instruct")
        check("f: humaneval_instruct diff +0.027 with an interval above zero",
              round(r["diff"], 3) == 0.027 and r["ci95"][0] > 0, json.dumps(r))
        want = (f"humaneval_instruct: diff +0.027, 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], "
                "significant gain, check (API drift or a scoring artifact)")
        check("f: PASS, with the gain flagged in the markdown verdict and in the JSON flags",
              res["item4"]["verdict"] == "PASS" and res["item4"]["flags"] == [want] and
              "**PASS**" in item4_line(md) and want in item4_line(md), json.dumps(res["item4"]))
        check("f: no flag on a PASS without a significant gain (case a's engine)",
              run(tmp, "f2", api_fresh=api, engine=tmp / "engine-a")[1]["item4"]["flags"] == [])

        print("--- g. FAIL, a small loss below zero")
        base300 = [1.0] * 270 + [0.0] * 30
        api300 = write_run(tmp / "api300", {task: base300 for task in A.TASKS})
        eng = write_run(tmp / "engine-g", {"gsm8k_cot_llama": base300, "ifeval": [0.0] * 6 + base300[6:],
                                           "humaneval_instruct": base300})
        md, res, _ = run(tmp, "g", api_fresh=api300, engine=eng)
        r = pair(res, "api_fresh", "engine", "ifeval")
        check("g: ifeval diff -0.020 (inside the 3 points) with an interval below zero",
              round(r["diff"], 3) == -0.02 and r["ci95"][1] < 0, json.dumps(r))
        want = (f"ifeval: diff -0.020, 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], "
                "the interval lies below zero")
        check("g: still FAIL, naming ifeval only, no flag", res["item4"]["verdict"] == "FAIL" and
              res["item4"]["reasons"] == [want] and res["item4"]["flags"] == [] and want in item4_line(md),
              json.dumps(res["item4"]))
        eng = write_run(tmp / "engine-g2", {"gsm8k_cot_llama": BASE, "ifeval": moved(3, 0),
                                            "humaneval_instruct": BASE})
        md, res, _ = run(tmp, "g2", api_fresh=api, engine=eng)
        r = pair(res, "api_fresh", "engine", "ifeval")
        check("g: 3 losses of 150 (diff -0.020) touch zero (upper bound 0.000): counted as including it, PASS",
              round(r["diff"], 3) == -0.02 and r["ci95"][1] == 0 and res["item4"]["verdict"] == "PASS",
              json.dumps(res["item4"]))

        print("--- h. a fork run of HumanEval only")
        he = "humaneval_instruct"
        eng = tmp / "engine-e"
        fork = write_run(tmp / "fork-he", {he: moved(2, 0)})
        md, res, _ = run(tmp, "h1", api_fresh=api, fork=fork, engine=eng)
        rows = md.splitlines()
        heads = {x.split(" | ")[0][2:]: x.split(" | ")[3] for x in rows
                 if x.startswith(tuple(f"| {t} |" for t in A.TASKS))}
        check("h: the fork column has humaneval_instruct (0.887) and plain 'not run' for gsm8k_cot_llama and ifeval",
              heads == {"gsm8k_cot_llama": "not run", "ifeval": "not run", he: "0.887"}, json.dumps(heads))
        fork_rows = {(x.split(" | ")[0][2:], x.split(" | ")[1]): x.split(" | ")[2] for x in rows
                     if x.startswith(("| engine - fork |", "| fork - API fresh |"))}
        check("h: engine - fork and fork - API fresh are computed on humaneval_instruct only, 'not run' on the others",
              len(fork_rows) == 6 and all((v == "150") == (t == he) for (_, t), v in fork_rows.items()) and
              all(v == "not run" for (_, t), v in fork_rows.items() if t != he), json.dumps(list(fork_rows.items())))
        r = pair(res, "fork", "engine", he)
        want = (f"- {he}: diff +0.013, 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]: the interval includes zero, "
                "the difference within 3 points")
        check("h: engine - fork on humaneval_instruct read two-sided: the interval includes zero, within 3 points",
              round(r["diff"], 3) == 0.013 and want in rows and "- gsm8k_cot_llama: not run" in rows and
              "- ifeval: not run" in rows and "engine - fork is information only, not the quality bar." in rows and
              [x["task"] for x in res["fork_reading"]] == list(A.TASKS) and
              res["fork_reading"][2]["reading"] == want[len(f"- {he}: "):],
              "\n".join(x for x in rows if x.startswith("- ")))
        alone = run(tmp, "h1-alone", api_fresh=api, engine=eng)[1]["item4"]
        check("h: the quality bar is the same verdict and reasons as without --fork (PASS on engine - API fresh)",
              res["item4"] == alone and alone["verdict"] == "PASS", json.dumps(res["item4"]))
        for name, engine, fork_he, diff, reading, c in (
                ("h2", tmp / "engine-c", moved(0, 4), -0.053,
                 "the interval lies below zero, the difference a loss over 3 points", "c"),
                ("h3", eng, moved(9, 0), 0.06, "the interval lies above zero, the difference a gain over 3 points",
                 "e")):
            fork = write_run(tmp / f"fork-{name}", {he: fork_he})
            md, res, _ = run(tmp, name, api_fresh=api, fork=fork, engine=engine)
            r = pair(res, "fork", "engine", he)
            alone = run(tmp, f"{name}-alone", api_fresh=api, engine=engine)[1]["item4"]
            check(f"h: engine - fork {diff:+.3f} (case {c}'s engine): '{reading}'; the quality bar as without --fork "
                  f"({alone['verdict']})",
                  round(r["diff"], 3) == diff and f"- {he}: diff {r['diff']:+.3f}, 95% CI [{r['ci95'][0]:+.3f}, "
                  f"{r['ci95'][1]:+.3f}]: {reading}" in md.splitlines() and res["item4"] == alone, json.dumps(r))

        print("--- i. the batched engine against the single-user engine run (--engine-ref)")

        def same(name: str, ref: Path, engine: Path) -> tuple[str, dict]:
            md, res, _ = run(tmp, name, api_fresh=api, engine=engine, engine_ref=ref)
            return md, res["same_engine"]
        line = lambda md: next((x for x in md.splitlines() if x.startswith("Batched quality bar (")), "")
        md, bar = same("i1", tmp / "engine-a", tmp / "engine-a")
        check("i: an engine against itself: PASS, every diff 0, the line names the bar",
              bar["verdict"] == "PASS" and all(x["result"]["diff"] == 0 for x in bar["pairs"]) and
              "**PASS**" in line(md) and "the same weights, so either way counts" in line(md) and
              "| engine batched - engine alone | n | alone | batched |" in md, line(md))
        check("i: the API pair's quality bar line is still there beside it (reported)",
              "**PASS**" in item4_line(md) and item4_line(md).startswith("Quality bar (engine - API fresh"), item4_line(md))
        md, bar = same("i2", api, tmp / "engine-g2")
        check("i: 3 losses of 150 (diff -0.020, the interval touching zero): PASS", bar["verdict"] == "PASS", json.dumps(bar))
        for name, engine, task, why in (("i3", tmp / "engine-b", "ifeval", "a loss over 3 points"),
                                        ("i4", tmp / "engine-c", "humaneval_instruct", "the interval lies below zero"),
                                        ("i5", tmp / "engine-f", "humaneval_instruct", "the interval lies above zero")):
            md, bar = same(name, api, engine)
            check(f"i: {task} {why}: FAIL naming it only", bar["verdict"] == "FAIL" and len(bar["reasons"]) == 1 and
                  bar["reasons"][0].startswith(f"{task}: diff ") and bar["reasons"][0].endswith(why) and
                  "**FAIL**" in line(md), json.dumps(bar["reasons"]))
        md, bar = same("i6", tmp / "engine-d", tmp / "engine-a")
        check("i: a reference without humaneval_instruct: not evaluated, the task named",
              bar["verdict"] == "not evaluated" and any(r.startswith("humaneval_instruct: not evaluated") for r in bar["reasons"]),
              json.dumps(bar))
    print(f"analyze_test.py: {'PASS' if not failed else 'FAIL'} ({len(failed)} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
