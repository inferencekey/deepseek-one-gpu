#!/usr/bin/env python3
"""tools/dsv41/quality/analyze.py - the quality table of the DeepSeek-V4.1-Flash engine against the API and the fork.

Reads lm-eval output directories (each the RUN_DIR that run_lm_eval.sh wrote: lm_eval/<task>/<model>/results_*.json
and samples_*.jsonl) and writes a markdown table: per task the headline metric of each column, then the paired
differences with their 95% bootstrap interval and flips (quality.py's paired_difference: 5000 resamples, seed 0, the
items both runs answered), then the verdict of the quality bar.

Columns, in this order: API fresh (--api-fresh), fork (--fork), engine (--engine), API 2026-10-06 (--api-then) and
EXL3 2026-10-06 (--exl3); the last two default to the earlier vLLM-EXL3 experiment's runs under $VLLM_EXL3 (PROVENANCE.md). A column
without a directory is "not run"; the fork's reads "not run (dropped 2026-10-12)", since the fork run was dropped
that day. Headline metrics: gsm8k_cot_llama exact_match (flexible-extract), ifeval
prompt_level_strict_acc, humaneval_instruct pass@1. Pairs, each "b minus a" on the shared items: engine - API fresh,
engine - fork, fork - API fresh, API fresh - API 2026-10-06; a pair whose runs are missing is listed as not run.
Flips are the items whose outcome differs.

The quality bar (the engine is good enough), as amended on 2026-10-12 and made one-sided against the API the same
day: FAIL when on any task the engine - API fresh interval lies entirely
below zero (its upper bound < 0) or the difference is < -0.03 (a loss of more than 3 points), naming each failing
task; PASS otherwise; "not evaluated" without both runs and the three tasks. An interval touching zero includes it. An
interval entirely above zero (lower bound > 0) still passes, with a flag on that task: "significant gain, check (API
drift or a scoring artifact)", listed in the JSON's item4 flags. The bar was engine - fork until the fork run was
dropped; engine - API fresh also holds the Q2_K quantization loss, so a FAIL cannot be put on the engine or on the
quantization without a fork run. With --fork, engine - fork is still reported, as information only, with its
two-sided reading per task (the same weights, so a difference either way counts): whether the interval includes zero, lies below or above it, and whether the difference
is within 3 points. A fork run of some tasks only (a HumanEval-only session) leaves the other tasks' fork cells
and pairs "not run"; the quality bar is unchanged by it.

Batched: with --engine-ref, the engine's run (with batch slots) is also held to the single-user engine run (the same
weights, so either way counts): per task the paired difference engine - engine ref, PASS when every task's 95%
interval includes zero and the loss is <= 3 points; the quality bar line above is then the API fresh pair, reported
(it holds the Q2_K loss), and the batched bar is the verdict for batching.

    python3 tools/dsv41/quality/analyze.py --api-fresh RUN [--fork RUN] --engine RUN [--api-then RUN] [--exl3 RUN] \\
        [--engine-ref RUN] [--out table.md] [--json table.json]

Each RUN is searched recursively (quality.find_results and load_samples, rglob), so a column split over several pod
sessions takes their parent directory: qfork runs in 2 or 3 sessions split by task,
so --fork is the directory that holds each session's out/quality/; every task must come from one session only (a task
found twice takes the last file in sorted path order).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality as Q  # noqa: E402

# a checkout of the earlier vLLM-EXL3 host-offload experiment, whose reference runs are the last two columns
VLLM_EXL3 = Path(os.environ.get("VLLM_EXL3") or
                 Path(os.environ.get("BUILD_ROOT") or Path.home() / "dsv41-build") / "quality-ref") / "data/controlled"
API_THEN = VLLM_EXL3 / "E11-quality-reference/api/20261006T181137Z"
EXL3_THEN = VLLM_EXL3 / "E10-quality-exl3/exl3-offload-gen30-lean-v2/20261006T211059Z"

# task -> (metric, filter) of the headline
TASKS = {"gsm8k_cot_llama": ("exact_match", "flexible-extract"),
         "ifeval": ("prompt_level_strict_acc", "none"),
         "humaneval_instruct": ("pass@1", "create_test")}
COLUMNS = [("api_fresh", "API fresh"), ("fork", "fork"), ("engine", "engine"), ("api_then", "API 2026-10-06"),
           ("exl3", "EXL3 2026-10-06")]
# the quality bar's pair, b minus a, since 2026-10-12; it was (fork, engine)
ITEM4 = ("api_fresh", "engine")
PAIRS = [ITEM4, ("fork", "engine"), ("api_fresh", "fork"), ("api_then", "api_fresh")]
MAX_LOSS = 0.03
FORK_NOT_RUN = "not run (dropped 2026-10-12)"
GAIN_FLAG = "significant gain, check (API drift or a scoring artifact)"


def pair_name(a: str, b: str) -> str:
    label = dict(COLUMNS)
    return f"{label[b]} - {label[a]}"


def not_run(res: dict, *keys: str) -> str:
    """The cell of a column or a pair without a result; the fork's says its run was dropped."""
    return FORK_NOT_RUN if "fork" in keys and res["runs"]["fork"] is None else "not run"


def two_sided(r: dict) -> str:
    """engine - fork's two-sided reading (the same weights, so a gain counts as much as a loss): the interval
    against zero, the difference against 3 points."""
    lo, hi = r["ci95"]
    where = "lies below zero" if hi < 0 else "lies above zero" if lo > 0 else "includes zero"
    size = ("within 3 points" if abs(r["diff"]) <= MAX_LOSS else
            "a loss over 3 points" if r["diff"] < 0 else "a gain over 3 points")
    return f"diff {r['diff']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]: the interval {where}, the difference {size}"


def analyze(runs: dict[str, Path | None]) -> dict:
    """Headlines, paired differences and the quality bar verdict of the runs given (None: not run)."""
    label = dict(COLUMNS)
    for key, d in runs.items():
        if d is not None and not Path(d).is_dir():
            raise SystemExit(f"analyze.py: {label[key]}: {d} is not a directory")
    heads = {}
    for key, d in runs.items():
        found = Q.find_results(d) if d is not None else {}
        for task, (metric, filt) in TASKS.items():
            v = Q.headline(found[task]["metrics"], metric, filt)[0] if task in found else None
            heads[(key, task)] = v
    pairs = []
    for a, b in PAIRS:
        for task, (metric, filt) in TASKS.items():
            r = None
            if runs.get(a) is not None and runs.get(b) is not None:
                r = Q.paired_difference(Q.load_samples(runs[a], task, metric, filt),
                                        Q.load_samples(runs[b], task, metric, filt))
            pairs.append({"a": a, "b": b, "task": task, "result": r})
    item4 = [p for p in pairs if (p["a"], p["b"]) == ITEM4]
    flags = []
    if any(p["result"] is None for p in item4):
        verdict = "not evaluated"
        reasons = [f"no {label[k]} run" for k in ITEM4 if runs.get(k) is None] or \
            [f"{p['task']}: fewer than 2 items answered by both" for p in item4 if p["result"] is None]
    else:
        reasons = []
        for p in item4:
            r = p["result"]
            stat = f"{p['task']}: diff {r['diff']:+.3f}, 95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
            # one-sided: the bar asks "not worse than the vendor"; a significant gain is implausible for Q2_K, so it
            # is a signal to inspect, not a defect
            why = [w for w, bad in (("the interval lies below zero", r["ci95"][1] < 0),
                                    ("a loss over 3 points", r["diff"] < -MAX_LOSS)) if bad]
            if why:
                reasons.append(f"{stat}, " + " and ".join(why))
            if r["ci95"][0] > 0:
                flags.append(f"{stat}, {GAIN_FLAG}")
        verdict = "FAIL" if reasons else "PASS"
        reasons = reasons or [f"no task's {pair_name(*ITEM4)} interval lies below zero and no loss exceeds 3 points"]
    fork_reading = [{"task": p["task"], "reading": two_sided(p["result"]) if p["result"] else None}
                    for p in pairs if (p["a"], p["b"]) == ("fork", "engine")] \
        if runs.get("fork") is not None and runs.get("engine") is not None else []
    return {"runs": {k: (str(v) if v is not None else None) for k, v in runs.items()},
            "headline": [{"column": k, "task": t, "value": heads[(k, t)]} for k, _ in COLUMNS for t in TASKS],
            "pairs": [dict(p, result=(dict(p["result"], ci95=list(p["result"]["ci95"])) if p["result"] else None))
                      for p in pairs],
            "item4": {"verdict": verdict, "pair": {"a": ITEM4[0], "b": ITEM4[1], "name": pair_name(*ITEM4)},
                      "reasons": reasons, "flags": flags},
            "fork_reading": fork_reading}


def markdown(res: dict) -> str:
    label = dict(COLUMNS)
    item4 = res["item4"]
    head = {(h["column"], h["task"]): h["value"] for h in res["headline"]}
    out = ["| task | metric | " + " | ".join(lbl for _, lbl in COLUMNS) + " |",
           "|---|---|" + "---|" * len(COLUMNS)]
    for task, (metric, filt) in TASKS.items():
        cells = [f"{head[(k, task)]:.3f}" if head[(k, task)] is not None else not_run(res, k)
                 for k, _ in COLUMNS]
        out.append(f"| {task} | {metric} ({filt}) | " + " | ".join(cells) + " |")
    out += ["", "Paired differences, b minus a on the items both answered (95% percentile bootstrap, 5000 resamples, "
            "seed 0; flips: items whose outcome differs):", "",
            "| b - a | task | n | a | b | diff | 95% CI | flips |", "|---|---|---|---|---|---|---|---|"]
    for p in res["pairs"]:
        r, name = p["result"], pair_name(p["a"], p["b"])
        if r is None:
            out.append(f"| {name} | {p['task']} | {not_run(res, p['a'], p['b'])} | | | | | |")
            continue
        flips = f"{r['discordant']} ({100 * r['discordant'] / r['n']:.1f}%)"
        out.append(f"| {name} | {p['task']} | {r['n']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['diff']:+.3f} | "
                   f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {flips} |")
    out += ["", f"Quality bar ({item4['pair']['name']}, since 2026-10-12: no interval lies below zero and the loss "
            "is <= 3 points on every task; the bar holds the Q2_K quantization loss too, so a FAIL cannot be put on "
            f"the engine or on the quantization without a fork run): **{item4['verdict']}** "
            f"({'; '.join(item4['reasons'] + item4['flags'])})."]
    if res["fork_reading"]:
        out += ["", "engine - fork is information only, not the quality bar.", "",
                "engine - fork, two-sided reading (the same Q2_K weights, so a difference either way counts):"]
        out += [f"- {x['task']}: {x['reading'] or 'not run'}" for x in res["fork_reading"]]
    out += ["", "Runs: " + "; ".join(f"{label[k]} `{v}`" for k, v in res["runs"].items() if v is not None) + "."]
    return "\n".join(out) + "\n"


def same_engine_bar(ref: Path | None, run: Path | None) -> dict:
    """The batched quality bar: run (the engine batched) - ref (the single-user engine run) per task, PASS
    when every task's 95% interval includes zero and the loss is <= 3 points; "not evaluated" without both runs."""
    for d in (ref, run):
        if d is not None and not Path(d).is_dir():
            raise SystemExit(f"analyze.py: {d} is not a directory")
    rows, reasons = [], []
    for task, (metric, filt) in TASKS.items():
        r = None
        if ref is not None and run is not None:
            r = Q.paired_difference(Q.load_samples(ref, task, metric, filt), Q.load_samples(run, task, metric, filt))
        rows.append({"task": task, "result": dict(r, ci95=list(r["ci95"])) if r else None})
        if r is None:
            reasons.append(f"{task}: not evaluated (no run, or fewer than 2 items answered by both)")
            continue
        lo, hi = r["ci95"]
        why = [w for w, bad in (("the interval lies below zero", hi < 0), ("the interval lies above zero", lo > 0),
                                ("a loss over 3 points", r["diff"] < -MAX_LOSS)) if bad]
        if why:
            reasons.append(f"{task}: diff {r['diff']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], " + " and ".join(why))
    if any(x["result"] is None for x in rows):
        verdict = "not evaluated"
    else:
        verdict = "FAIL" if reasons else "PASS"
        reasons = reasons or ["every task's interval includes zero and no loss exceeds 3 points"]
    return {"ref": str(ref) if ref else None, "run": str(run) if run else None, "pairs": rows, "verdict": verdict,
            "reasons": reasons}


def same_engine_markdown(bar: dict) -> str:
    out = ["| engine batched - engine alone | n | alone | batched | diff | 95% CI | flips |", "|---|---|---|---|---|---|---|"]
    for x in bar["pairs"]:
        r = x["result"]
        if r is None:
            out.append(f"| {x['task']} | not run | | | | | |")
            continue
        out.append(f"| {x['task']} | {r['n']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['diff']:+.3f} | "
                   f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['discordant']} ({100 * r['discordant'] / r['n']:.1f}%) |")
    out += ["", "Batched quality bar (the batched engine against the single-user engine "
            "run, the same weights, so either way counts; every task's 95% interval includes zero and the loss is <= 3 "
            f"points): **{bar['verdict']}** ({'; '.join(bar['reasons'])}).", "",
            f"Runs: single-user engine `{bar['ref']}`; batched `{bar['run']}`."]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--api-fresh", type=Path)
    ap.add_argument("--fork", type=Path, help="the fork's run, or the parent directory of its split qfork sessions "
                    "(searched recursively)")
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--api-then", type=Path, default=API_THEN)
    ap.add_argument("--exl3", type=Path, default=EXL3_THEN)
    ap.add_argument("--engine-ref", type=Path, help="the single-user engine run, the batched engine's bar")
    ap.add_argument("--out", type=Path, help="the markdown table (default: stdout only)")
    ap.add_argument("--json", type=Path, help="the same as JSON")
    a = ap.parse_args()
    res = analyze({"api_fresh": a.api_fresh, "fork": a.fork, "engine": a.engine, "api_then": a.api_then,
                   "exl3": a.exl3})
    md = markdown(res)
    if a.engine_ref is not None:
        res["same_engine"] = same_engine_bar(a.engine_ref, a.engine)
        md += "\n" + same_engine_markdown(res["same_engine"])
    print(md, end="")
    if a.out:
        a.out.write_text(md)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
