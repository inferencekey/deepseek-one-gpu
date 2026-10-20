#!/usr/bin/env python3
"""The qdiag diagnostic report: per phase of an engine quality session (tools/dsv41/pod/
e07-run.sh, E07_PLAN=qdiag or qengine), what the engine did and what the GPU and the host were doing meanwhile, to
separate the four hypotheses behind a qengine session's ~15 tok/s against the e07b bench's 62.8 (a slow or throttled GPU or host,
concurrency in the server path, the adaptive hot set, CPU quota throttling).

  qdiag_report.py OUT        writes OUT/out/qdiag.md and OUT/out/qdiag.json; prints the table

OUT is e07.sh's results directory (out/ extracted from the pod's logs). One row per phase:
  bench-<name> <phase>  each phase of out/bench-*.json (qdiag: bench-qdiag's short-static and short-adaptive), one
                        request on the engine's own protocol, its window the phase's start_unix and end_unix
  continuation, lm-eval-<...>, needle-<n>  the server path's steps (qdiag: lm-eval-gsm8k-c1 and -c2), their windows
                        from out/step-times.log (pod-steps.sh); a run without it (an older session) gets them rebuilt from
                        env.log's `date -u` and out/steps.log's durations in order (seconds off: the gaps between
                        steps). The engine's requests of out/quality-engine.log (no time of their own) are matched in
                        order to the proxy's responses (every api_usage.jsonl under out/quality with usage, any
                        status: the time, the prompt and completion tokens), each to the first unused one of the next
                        LOOKAHEAD with its tokens (concurrent responses land in either order), and take the phase whose
                        window holds that response; a matched response with an error status (the proxy's 502 of a
                        failed echo cut) counts as answered with an error; a request with no response (cancelled: the
                        client gave up, or an error response without usage) takes the phase of the request before it
  all server requests   every request of out/quality-engine.log
Per row: requests, generated tokens, the medians over the requests of ms per generated token (decode ms over the
generated tokens), ms per decode step, doorbell wait per step (the host waiting for the GPU's ring, experts.cpp
decode_serve: the GPU's front work), CPU-path GB/s, hit rate, adaptive swaps per request and prefill tok/s (tokens
read over their ms); from the samplers in the window (pod-samplers.sh: out/gpu-clocks.csv every 1 s, out/host-load.log
every 2 s) the median and minimum SM clock, the median power draw and the enforced power limit, every clock event
(throttle) reason seen with its samples, the cgroup's CPU throttled time accrued over the window and the median and
maximum 1-minute load average. Also each engine load's init rates (PCIe into the ring, the CPU path) and load time,
and the cgroup's CPU limit. Missing inputs give n/a, never an error. python3 only.
"""

import calendar
import glob
import json
import os
import re
import statistics
import sys
import time

# NVML's clocks event reasons (nvmlClocksEventReason*, nvidia-smi's clocks_event_reasons.active bitmask)
REASONS = ((0x1, "GpuIdle"), (0x2, "ApplicationsClocksSetting"), (0x4, "SwPowerCap"), (0x8, "HwSlowdown"),
           (0x10, "SyncBoost"), (0x20, "SwThermalSlowdown"), (0x40, "HwThermalSlowdown"),
           (0x80, "HwPowerBrakeSlowdown"), (0x100, "DisplayClockSetting"))
REQ_RE = re.compile(r"request (\d+): (\d+) prompt tokens, cold resume of (\d+), (\d+) read in ([0-9.]+) ms, (\d+) "
                    r"generated \((\w+)\) in ([0-9.]+) ms")
DECODE_RE = re.compile(
    r"request (\d+): decode experts: (\d+) steps, \d+ hits and \d+ misses \(hit rate ([0-9.]+)\), \d+ CPU jobs "
    r"reading [0-9.]+ MiB in [0-9.]+ ms \([0-9.]+ ms per step, ([0-9.]+) GB/s\), doorbell waits ([0-9.]+) ms over "
    r"\d+ doorbells, plan [0-9.]+ ms; ([0-9.]+) ms per step")
HOT_RE = re.compile(r"request (\d+): decode hot set (\w+)[^:]*: (\d+) swaps \(([0-9.]+) MiB copied")
INIT_RE = re.compile(r"PCIe ([0-9.]+) GB/s into the ring, CPU ([0-9.]+) GB/s")
STEP_RE = re.compile(r"^step (\S+): exit (\d+), (\d+) s", re.M)
# how many unused proxy responses ahead an engine request is looked for: twice the highest lm-eval num_concurrent
# (4, the fork's)
LOOKAHEAD = 8
# the second e07b session (2026-10-11, RTX PRO 6000 Blackwell WS, AMD EPYC 9355P, bench-e07b), the reference
E07B = ("E07b session 2 (2026-10-11, RTX PRO 6000 Blackwell WS, bench-e07b short-adaptive: 128-token prompt, 480 "
        "decode steps): 62.8 tok/s, 15.9 ms per step, doorbell waits 13.0 ms per step, CPU path 123.0 GB/s, hit "
        "rate 0.940, prefill 288.7 tok/s; init rates PCIe 26.68 and CPU 151.13 GB/s")


def read(p):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(v, nd=1):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def windows(o):
    """{step: (start_s, end_s)} and how they were found: out/step-times.log, else rebuilt from env.log's date and
    out/steps.log's durations."""
    w = {}
    for line in read(os.path.join(o, "step-times.log")).splitlines():
        f = line.split()
        if len(f) == 4 and f[1].isdigit() and f[2].isdigit():
            w[f[0]] = (int(f[1]), int(f[2]))
    if w:
        return w, "out/step-times.log"
    m = re.search(r"^\w{3} (\w{3}) +(\d+) (\d\d:\d\d:\d\d) UTC (\d{4})$", read(os.path.join(o, "env.log")), re.M)
    if not m:
        return {}, "none (no out/step-times.log, no date in env.log)"
    t = calendar.timegm(time.strptime(" ".join(m.groups()), "%b %d %H:%M:%S %Y"))
    for name, _, secs in STEP_RE.findall(read(os.path.join(o, "steps.log"))):
        w[name] = (t, t + int(secs))
        t += int(secs)
    return w, "rebuilt from env.log's date and out/steps.log's durations (no out/step-times.log: seconds off)"


def gpu_samples(o):
    """out/gpu-clocks.csv as [{field: value}] with "t" in unix seconds, and its fields."""
    lines = read(os.path.join(o, "gpu-clocks.csv")).splitlines()
    if not lines:
        return [], []
    fields = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        v = [x.strip() for x in line.split(",")]
        if len(v) != len(fields):
            continue
        r = dict(zip(fields, v))
        try:
            ts = r.get("timestamp", "")
            r["t"] = calendar.timegm(time.strptime(ts.split(".")[0], "%Y/%m/%d %H:%M:%S"))
        except ValueError:
            continue
        rows.append(r)
    return rows, fields


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def host_samples(o):
    """out/host-load.log as [{t, load1, stat: {key: int}, limit: str}]."""
    out = []
    for block in read(os.path.join(o, "host-load.log")).split("=== ")[1:]:
        lines = block.splitlines()
        if not lines or not lines[0].strip().isdigit():
            continue
        r = {"t": int(lines[0].strip()), "stat": {}}
        for line in lines[1:]:
            if line.startswith("loadavg "):
                r["load1"] = num(line.split()[1]) if len(line.split()) > 1 else None
            elif line.startswith("cpu-limit "):
                r["limit"] = line[len("cpu-limit "):]
            elif line.startswith("cpu-stat ") and line != "cpu-stat none":
                f = line.split()[1:]
                r["stat"] = {f[i]: int(f[i + 1]) for i in range(0, len(f) - 1, 2) if f[i + 1].isdigit()}
        out.append(r)
    return out


def throttled_ms(stat):
    """The cgroup's cumulative CPU throttled time in ms: v2 throttled_usec, v1 throttled_time (ns)."""
    if "throttled_usec" in stat:
        return stat["throttled_usec"] / 1e3
    if "throttled_time" in stat:
        return stat["throttled_time"] / 1e6
    return None


def cpu_limit(limit):
    """The cgroup's CPU limit in CPUs from a cpu-limit line, None for no limit."""
    m = re.match(r"v2 cpu\.max (\d+) (\d+)", limit or "")
    if m:
        return int(m.group(1)) / int(m.group(2))
    m = re.match(r"v1 cfs_quota_us (\d+) cfs_period_us (\d+)", limit or "")
    if m:
        return int(m.group(1)) / int(m.group(2))
    return None


def engine_requests(text):
    """Each request of an engine log, in order: its prompt, its decode and its hot set."""
    reqs = {int(m.group(1)): {"n": int(m.group(1)), "prompt_tokens": int(m.group(2)), "read": int(m.group(4)),
                              "read_ms": float(m.group(5)), "generated": int(m.group(6)), "finish": m.group(7),
                              "decode_ms": float(m.group(8))} for m in REQ_RE.finditer(text)}
    for m in DECODE_RE.finditer(text):
        r = reqs.get(int(m.group(1)))
        if r is not None:
            steps = int(m.group(2))
            r.update(steps=steps, hit_rate=float(m.group(3)), cpu_gbs=float(m.group(4)),
                     doorbell_ms_per_step=float(m.group(5)) / steps if steps else None, ms_per_step=float(m.group(6)))
    for m in HOT_RE.finditer(text):
        r = reqs.get(int(m.group(1)))
        if r is not None:
            r.update(policy=m.group(2), swaps=int(m.group(3)), swap_mib=float(m.group(4)))
    return [reqs[k] for k in sorted(reqs)]


def engine_init(path):
    text = read(path)
    m, load = INIT_RE.search(text), re.search(r"loaded in (\d+) ms", text)
    if not m and not load:
        return None
    return {"log": os.path.basename(path), "pcie_ring_gbs": float(m.group(1)) if m else None,
            "cpu_gbs": float(m.group(2)) if m else None, "load_s": int(load.group(1)) / 1e3 if load else None}


def proxy_records(o):
    """The proxy's responses that carry usage (any status: the 502 of a failed echo cut keeps the upstream's), by
    time; and the count of error responses without usage, which no engine request can be matched to."""
    recs, bare = [], 0
    for f in glob.glob(os.path.join(o, "quality", "**", "api_usage.jsonl"), recursive=True):
        for line in read(f).splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("usage") and "t" in r:
                recs.append(r)
            elif r.get("status") != 200:
                bare += 1
    return sorted(recs, key=lambda r: r["t"]), bare


def server_phases(o, w):
    """The server path's steps in order, each with its window and its engine requests; the match's counts."""
    names = [n for n, _, _ in STEP_RE.findall(read(os.path.join(o, "steps.log")))
             if n == "continuation" or n.startswith(("lm-eval-", "needle-"))]
    phases = [{"phase": n, "window": w.get(n), "requests": []} for n in names]

    def phase_at(t):
        for p in phases:
            if p["window"] and p["window"][0] <= t <= p["window"][1] + 1:
                return p
        return None

    reqs, (recs, bare) = engine_requests(read(os.path.join(o, "quality-engine.log"))), proxy_records(o)
    used, i = set(), 0
    last = phases[0] if phases else None
    for r in reqs:
        # the first of the next LOOKAHEAD unused responses with the same prompt and completion tokens: concurrent
        # responses reach the proxy's log in either order; the ones skipped stay for the next requests
        j, seen = i, 0
        while j < len(recs) and seen < LOOKAHEAD:
            if j not in used:
                u = recs[j]["usage"]
                if u.get("prompt_tokens") == r["prompt_tokens"] and u.get("completion_tokens") == r["generated"]:
                    break
                seen += 1
            j += 1
        if j < len(recs) and seen < LOOKAHEAD:
            used.add(j)
            while i in used:
                i += 1
            p = phase_at(recs[j]["t"]) or last
            if recs[j].get("status") != 200:
                r["error"] = recs[j].get("status")
        else:
            p, r["cancelled"] = last, True
        if p is not None:
            p["requests"].append(r)
            last = p
    return phases, reqs, {"engine_requests": len(reqs), "proxy_responses": len(recs) + bare, "matched": len(used),
                          "errors": sum("error" in r for r in reqs), "unmatched": len(reqs) - len(used),
                          "bare_errors": bare, "unused_responses": len(recs) - len(used)}


def bench_phases(o):
    out = []
    for f in sorted(glob.glob(os.path.join(o, "bench-*.json"))):
        try:
            j = json.load(open(f))
        except (OSError, ValueError):
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        for p in j.get("phases", []):
            x, hs = p.get("decode_experts") or {}, p.get("hot") or {}
            steps = x.get("steps")
            r = {"generated": p.get("generated", 0), "decode_ms": p.get("decode_ms"), "read": p.get("prompt_tokens", 0)
                 - p.get("reused", 0), "read_ms": p.get("prompt_ms"), "steps": steps, "hit_rate": x.get("hit_rate"),
                 "cpu_gbs": x.get("cpu_gbs"), "ms_per_step": x.get("ms_per_step"), "swaps": hs.get("swaps"),
                 "doorbell_ms_per_step": x["doorbell_wait_ms"] / steps if steps else None, "policy": hs.get("policy")}
            win = (p["start_unix"], p["end_unix"]) if "start_unix" in p and "end_unix" in p else None
            out.append({"phase": f"{name} {p['phase']}", "window": win, "requests": [r]})
    return out


def row(p, gpu, host, throttle_field):
    rs, win = p["requests"], p["window"]
    gen = [r for r in rs if r.get("generated")]
    out = {"phase": p["phase"], "window": list(win) if win else None, "requests": len(rs),
           "cancelled": sum(bool(r.get("cancelled")) for r in rs), "errors": sum("error" in r for r in rs),
           "generated": sum(r.get("generated") or 0 for r in rs),
           "ms_per_token": med(r["decode_ms"] / r["generated"] for r in gen if r.get("decode_ms") is not None),
           "ms_per_step": med(r.get("ms_per_step") for r in rs),
           "doorbell_ms_per_step": med(r.get("doorbell_ms_per_step") for r in rs),
           "cpu_gbs": med(r.get("cpu_gbs") for r in rs), "hit_rate": med(r.get("hit_rate") for r in rs),
           "swaps_per_request": med(r.get("swaps") for r in rs),
           "prefill_tok_s": med(r["read"] / (r["read_ms"] / 1e3) for r in rs if r.get("read") and r.get("read_ms"))}
    g = [s for s in gpu if win and win[0] <= s["t"] <= win[1]]
    sm = [num(s.get("clocks.sm")) for s in g]
    out.update(gpu_samples=len(g), sm_clock_median=med(sm),
               sm_clock_min=min([x for x in sm if x is not None], default=None),
               mem_clock_median=med(num(s.get("clocks.mem")) for s in g),
               power_median=med(num(s.get("power.draw")) for s in g),
               power_limit=med(num(s.get("enforced.power.limit")) for s in g),
               temperature_max=max([x for x in (num(s.get("temperature.gpu")) for s in g) if x is not None],
                                   default=None),
               gpu_util_median=med(num(s.get("utilization.gpu")) for s in g),
               pcie_gen=sorted({s["pcie.link.gen.current"] for s in g if s.get("pcie.link.gen.current")}))
    seen = {}
    for s in g:
        try:
            bits = int(s.get(throttle_field) or "", 16)
        except ValueError:
            continue
        for bit, name in REASONS:
            if bits & bit:
                seen[name] = seen.get(name, 0) + 1
    out["throttle_reasons"] = seen if throttle_field else None
    h = [s for s in host if win and win[0] <= s["t"] <= win[1]]
    th = [throttled_ms(s["stat"]) for s in h if throttled_ms(s["stat"]) is not None]
    nt = [s["stat"]["nr_throttled"] for s in h if "nr_throttled" in s["stat"]]
    out.update(host_samples=len(h), cgroup_throttled_ms=round(th[-1] - th[0], 1) if len(th) > 1 else None,
               cgroup_nr_throttled=nt[-1] - nt[0] if len(nt) > 1 else None,
               load1_median=med(s.get("load1") for s in h),
               load1_max=max([s["load1"] for s in h if s.get("load1") is not None], default=None))
    return out


def requests_cell(r):
    """A row's requests, with how many of them were cancelled or answered with an error."""
    notes = [f"{r['cancelled']} cancelled"] if r["cancelled"] else []
    notes += [f"{r['errors']} answered with an error"] if r["errors"] else []
    return f"{r['requests']}" + (f" ({', '.join(notes)})" if notes else "")


def gpu_table(rows):
    """The samplers' side of the rows: the GPU's clocks, power and throttle reasons, the host's throttling and load."""
    return table(["phase", "GPU samples", "SM MHz median", "SM MHz min", "mem MHz median", "power W median",
                  "power limit W", "temp C max", "GPU util %", "PCIe gen", "throttle reasons (samples)", "host samples",
                  "cgroup throttled ms", "nr_throttled", "load1 median", "load1 max"],
                 [(r["phase"], r["gpu_samples"], fmt(r["sm_clock_median"], 0), fmt(r["sm_clock_min"], 0),
                   fmt(r["mem_clock_median"], 0), fmt(r["power_median"]), fmt(r["power_limit"]),
                   fmt(r["temperature_max"], 0), fmt(r["gpu_util_median"], 0), ",".join(r["pcie_gen"]) or "n/a",
                   ("none" if not r["throttle_reasons"] else
                    ", ".join(f"{k} ({v})" for k, v in r["throttle_reasons"].items()))
                   if r["throttle_reasons"] is not None else "n/a", r["host_samples"], fmt(r["cgroup_throttled_ms"]),
                   fmt(r["cgroup_nr_throttled"]), fmt(r["load1_median"], 2), fmt(r["load1_max"], 2)) for r in rows])


def gpu_fields_line(data):
    return (f"- GPU fields sampled: {','.join(data['gpu_fields']) or 'none (no out/gpu-clocks.csv)'}; throttle reasons "
            f"from {data['throttle_field'] or 'no field'} ({data['gpu_samples']} samples). Host samples: "
            f"{data['host_samples']}.")


def gpu_summary(data):
    """The GPU clock summary of a qengine sheet (report's data): the samplers' table per phase and what was sampled."""
    return "\n".join([f"Per phase, windows {data['windows']} (tools/dsv41/pod/qdiag_report.py):", "",
                      gpu_table(data["rows"]), "", gpu_fields_line(data), ""])


def report(o):
    """The report of the run's out/ directory as (markdown, data)."""
    w, wsrc = windows(o)
    gpu, fields = gpu_samples(o)
    host = host_samples(o)
    throttle = next((f for f in fields if f.startswith("clocks_")), None)
    server, reqs, match = server_phases(o, w)
    phases = bench_phases(o) + server
    if reqs:
        phases.append({"phase": "all server requests", "window": None, "requests": reqs})
    rows = [row(p, gpu, host, throttle) for p in phases]
    engines = [e for e in (engine_init(f) for f in sorted(glob.glob(os.path.join(o, "bench-*-engine.log"))) +
                           [os.path.join(o, "quality-engine.log")]) if e]
    limits = sorted({s["limit"] for s in host if s.get("limit")})
    data = {"windows": wsrc, "gpu_fields": fields, "throttle_field": throttle, "gpu_samples": len(gpu),
            "host_samples": len(host), "cpu_limit": limits, "match": match, "engines": engines, "rows": rows,
            "reference": E07B}
    md = ["Per phase (tools/dsv41/pod/qdiag_report.py): the engine's medians over each "
          "phase's requests, the GPU (out/gpu-clocks.csv, every 1 s) and the host (out/host-load.log, every 2 s) "
          f"over its window; windows {wsrc}.", "",
          table(["phase", "requests", "generated", "ms per token", "ms per decode step", "doorbell ms per step",
                 "CPU GB/s", "hit rate", "swaps per request", "prefill tok/s"],
                [(r["phase"], requests_cell(r),
                  r["generated"], fmt(r["ms_per_token"]), fmt(r["ms_per_step"]), fmt(r["doorbell_ms_per_step"]),
                  fmt(r["cpu_gbs"]), fmt(r["hit_rate"], 3), fmt(r["swaps_per_request"], 0), fmt(r["prefill_tok_s"]))
                 for r in rows]), "",
          gpu_table(rows), "",
          f"- Engine requests matched to the proxy's responses: {match['matched']} of {match['engine_requests']} "
          f"({match['errors']} of them answered with an error status; {match['unmatched']} without a response: "
          f"cancelled, the client gave up" + (f", or among the {match['bare_errors']} error responses without usage "
                                               f"to match" if match["bare_errors"] else "") +
          f"; {match['proxy_responses']} responses, {match['unused_responses']} with usage left unmatched).",
          gpu_fields_line(data),
          "- cgroup CPU limit: " + ("; ".join(f"{x} ({fmt(cpu_limit(x))} CPUs)" if cpu_limit(x) else x for x in limits)
                                    or "n/a") + "."]
    md += [f"- Engine load {e['log']}: init rates PCIe {fmt(e['pcie_ring_gbs'], 2)} GB/s into the ring, CPU "
           f"{fmt(e['cpu_gbs'], 2)} GB/s; loaded in {fmt(e['load_s'])} s." for e in engines]
    md += [f"- Reference: {E07B}.", ""]
    return "\n".join(md), data


def write(o):
    """report(o), written to o/qdiag.md and o/qdiag.json (also e07_report.py's, for the qengine and qdiag sheets)."""
    md, data = report(o)
    with open(os.path.join(o, "qdiag.md"), "w") as f:
        f.write(md)
    with open(os.path.join(o, "qdiag.json"), "w") as f:
        json.dump(data, f, indent=1)
    return md, data


def main(argv):
    if len(argv) != 1:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    o = os.path.join(argv[0], "out")
    md, data = write(o)
    print(md)
    print(f"qdiag_report.py: {os.path.join(o, 'qdiag.md')} and qdiag.json, {len(data['rows'])} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
