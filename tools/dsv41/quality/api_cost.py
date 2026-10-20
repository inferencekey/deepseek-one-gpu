#!/usr/bin/env python3
"""tools/dsv41/quality/api_cost.py - the real cost of a DeepSeek API run, from api_proxy.py's usage log and the price
page saved with the run (the budget: <= 0.50 USD per full run).

The page's deepseek-flash column gives the USD per 1M tokens for input cache hits, input cache misses and output, at
peak and off-peak, and its peak hours (2026-10-06: 01:00-04:00 and 06:00-10:00 UTC, Monday through Friday; off-peak
is half). Each logged request is priced at the rate of its own time (the log's unix `t`) with its own usage
(prompt_cache_hit_tokens, prompt_cache_miss_tokens, completion_tokens; a usage without the cache fields is all
misses). The page names Chinese public holidays as off-peak too; they are priced at peak here, so the cost is an
upper bound on those days. A page whose prices or peak hours cannot be read is an error, never a guess.

    python3 tools/dsv41/quality/api_cost.py RUN_DIR/api_usage.jsonl PRICE_PAGE.html [--json cost.json]

--estimate NEEDLE --max-usd X (api_run.sh, before anything is sent): the usage log is a past run's (the 2026-10-06
one), each of its prompt tokens priced as a cache miss and every token at peak, plus the needle prompts of
needle_check.py's NEEDLE file with their max tokens; exits 2 when that is over X.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def prices(page: str) -> dict:
    """{"peak": {...}, "off_peak": {...}, "peak_hours": [(start, end), ...]} of the page's deepseek-flash column."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
    model = re.search(r"MODEL (\S+) \(1\) (\S+)", text)
    if not model or model.group(1) != "deepseek-flash":
        raise SystemExit("api_cost.py: the price page's first model column is not deepseek-flash")
    out = {"peak": {}, "off_peak": {}}
    for key, label in (("input_cache_hit", r"1M INPUT TOKENS \(CACHE HIT\)"),
                       ("input_cache_miss", r"1M INPUT TOKENS \(CACHE MISS\)"), ("output", r"1M OUTPUT TOKENS")):
        m = re.search(label + r" OFF-PEAK \$([\d.]+) \$[\d.]+ PEAK \$([\d.]+) \$[\d.]+", text)
        if not m:
            raise SystemExit(f"api_cost.py: no {label} prices on the page")
        out["off_peak"][key], out["peak"][key] = float(m.group(1)), float(m.group(2))
    hours = re.search(r"Peak hours are (\d\d):00 - (\d\d):00 and (\d\d):00 - (\d\d):00 UTC, Monday through Friday",
                      text)
    if not hours:
        raise SystemExit("api_cost.py: the page's peak hours are not the sentence this reads")
    h = [int(x) for x in hours.groups()]
    out["peak_hours"] = [(h[0], h[1]), (h[2], h[3])]
    return out


def is_peak(t: float, peak_hours) -> bool:
    d = datetime.fromtimestamp(t, timezone.utc)
    return d.weekday() < 5 and any(a <= d.hour < b for a, b in peak_hours)


def cost(usage_file: Path, price: dict) -> dict:
    rows = [json.loads(x) for x in usage_file.read_text().splitlines() if x.strip()]
    tot = {"requests": len(rows), "ok": sum(r.get("status") == 200 for r in rows), "peak_requests": 0,
           "input_hit": 0, "input_miss": 0, "output": 0, "usd": 0.0}
    for r in rows:
        u = r.get("usage")
        if not u:
            continue
        hit = u.get("prompt_cache_hit_tokens", 0)
        miss = u.get("prompt_cache_miss_tokens", u["prompt_tokens"] - hit)
        out = u["completion_tokens"]
        peak = is_peak(r["t"], price["peak_hours"])
        p = price["peak" if peak else "off_peak"]
        tot["peak_requests"] += peak
        tot["input_hit"] += hit
        tot["input_miss"] += miss
        tot["output"] += out
        tot["usd"] += (hit * p["input_cache_hit"] + miss * p["input_cache_miss"] + out * p["output"]) / 1e6
    return tot


def estimate(usage_file: Path, needle_file: Path, price: dict) -> dict:
    """The upper bound before a run: the past run's tokens and the needle prompts, all misses, all at peak."""
    rows = [json.loads(x) for x in usage_file.read_text().splitlines() if x.strip()]
    nd = json.loads(needle_file.read_text())
    pin = sum(r["usage"]["prompt_tokens"] for r in rows) + sum(q["prompt_tokens"] for q in nd["prompts"])
    out = sum(r["usage"]["completion_tokens"] for r in rows) + nd["max_tokens"] * len(nd["prompts"])
    p = price["peak"]
    return {"requests": len(rows) + len(nd["prompts"]), "input": pin, "output": out,
            "usd": (pin * p["input_cache_miss"] + out * p["output"]) / 1e6}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("usage", type=Path)
    ap.add_argument("page", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--estimate", type=Path, metavar="NEEDLE")
    ap.add_argument("--max-usd", type=float)
    a = ap.parse_args()
    price = prices(a.page.read_text(errors="replace"))
    if a.estimate:
        e = estimate(a.usage, a.estimate, price)
        print(f"api_cost.py estimate: {e['requests']} requests ({a.usage} and the needle prompts of {a.estimate}), "
              f"{e['input']} prompt tokens all at the peak cache-miss price, {e['output']} output tokens at peak: "
              f"{e['usd']:.4f} USD (cap {a.max_usd})")
        return 2 if a.max_usd is not None and e["usd"] > a.max_usd else 0
    c = cost(a.usage, price)
    c.update(usage=str(a.usage), price_page=str(a.page), prices={k: price[k] for k in ("peak", "off_peak")},
             peak_hours_utc=price["peak_hours"])
    print(f"api_cost.py: {c['requests']} requests ({c['ok']} answered 200, {c['peak_requests']} at peak), input "
          f"{c['input_miss']} cache-miss and {c['input_hit']} cache-hit tokens, {c['output']} output tokens: "
          f"{c['usd']:.4f} USD at the page's deepseek-flash prices ({a.page})")
    if a.json:
        a.json.write_text(json.dumps(c, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
