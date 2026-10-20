"""Read lm-evaluation-harness outputs and compare two systems item by item.

Copied from an earlier vLLM-EXL3 host-offload experiment at 40202fd (src/hostoffload/analysis/quality.py, PROVENANCE.md), with
stats.mean inlined and one change: load_samples takes the filter. lm-eval writes one sample line per item and filter
(gsm8k_cot_llama logs strict-match and flexible-extract for every doc_id), and the source kept whichever line came
last; the headline of a task names its filter, so its items are read with that filter.

lm-eval writes, per task, a results_*.json ({"results": {task: {"<metric>,<filter>": value,
"<metric>_stderr,<filter>": se}}}) and, with --log_samples, samples_<task>_*.jsonl with one line per item
(doc_id plus the metric values). The comparison uses the items both systems answered (paired), and a paired
bootstrap over items for the difference in mean score.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def mean(v) -> float | None:
    v = list(v)
    return sum(v) / len(v) if v else None


def find_results(run_dir: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(run_dir).rglob("results_*.json")):
        d = json.loads(f.read_text())
        for task, metrics in d.get("results", {}).items():
            out[task] = {"file": f, "metrics": metrics}
    return out


def headline(metrics: dict, metric: str, filt: str | None = None) -> tuple[float | None, float | None]:
    """Value and stderr of `metric` (with `filt`, that filter's; else the first filter that has it)."""
    for k, v in metrics.items():
        name, _, f = k.partition(",")
        if name == metric and (filt is None or f == filt) and isinstance(v, (int, float)):
            se = metrics.get(f"{metric}_stderr,{f}")
            return float(v), (float(se) if isinstance(se, (int, float)) else None)
    return None, None


def load_samples(run_dir: Path, task: str, metric: str, filt: str | None = None) -> dict[int, float]:
    """doc_id -> the item's `metric` (with `filt`, only that filter's lines)."""
    out = {}
    for f in sorted(Path(run_dir).rglob(f"samples_{task}_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if filt is not None and r.get("filter") != filt:
                continue
            if metric in r and isinstance(r[metric], (int, float, bool)):
                out[int(r["doc_id"])] = float(r[metric])
    return out


def paired_difference(a: dict[int, float], b: dict[int, float], n_boot: int = 5000, seed: int = 0) -> dict | None:
    """Mean(b) - mean(a) over shared items, with a paired percentile bootstrap 95% interval."""
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return {"n": len(ids), "mean_a": mean([a[i] for i in ids]), "mean_b": mean([b[i] for i in ids]),
            "diff": sum(d) / len(d), "ci95": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]),
            "discordant": sum(1 for x in d if x != 0)}
