#!/usr/bin/env python3
"""The real-model session sheet (tools/dsv41/pod/e07.sh, after the pull): README.md with every table and the session's
logs, in SHEET_ROOT/<UTC date>-e07a/ (the record: host CPU and its measured GB/s, PCIe H2D GB/s, VRAM and RAM
use, every parameter and the cost).

  e07_report.py OUT SHEET_ROOT

OUT is e07.sh's results directory (out/ extracted from the pod's logs archive, cost.txt, pod.json, e07.log). Every
section is written from what came back, so a session that stopped early still gets its sheet, the missing parts
named (a reader mode E07_READERS left out, per out/plan.log, as not run). Targets (the engine's "Done means"): forced parity per
layer <= 2e-2, free greedy >= 90% of 64 ids, 16K TTFT <= 5 s, second turn <= 1 s (accepted at about 2 s), decode
>= 30 tok/s at short context, 256K TTFT <= 180 s without a crash. The isolation runs add the
isolated MoE run (the per-layer isolated rel L2), the MMQ prefill run when it ran, the layer-0 isolation of
layer_parity.py (the three numerics against the oracle, the GPU against its reference, where the gap enters) and each
bench phase's decode experts (hit rate, CPU misses, doorbell waits) with the timed phase's per-stage split.
E07_PLAN=e08 writes SHEET_ROOT/<UTC date>-e08/: bench-e08's phases, the hot set per
phase (policy, hit rate overall and per layer, swaps, the reset before it), the best policy's short decode against
the gate (>= 30 tok/s on the A100), the 16K prompt's per-stage split, the server's chats, and the routing trace
(logs/route-trace.txt.gz, tools/dsv41/engine/hot_policy_sim.py compare replays it). E07_PLAN=e09
 writes SHEET_ROOT/<UTC date>-e09/: bench-e09's 16K prompts with their per-stage split against
the gate (16K TTFT <= 5 s, the engine's prompt ms), the short decode (>= 30 tok/s), the 256K prompt (<= 3 min) and the
two-turn chat (turn 2's cached tokens, the route trace's resume=1 and its decode). E07_PLAN=e07b
 writes SHEET_ROOT/<UTC date>-e07b/ with a goal table first: one row per "Done means" item
1-6 with the measured number, PASS, FAIL or REPORTED (item 1, the free-greedy and isolated MoE parity on this card,
reported) and the cause of every FAIL or missing result, read from out/steps.log ("timeout after N
s in <step>", a failed or skipped step, a step that never ran and the stop before it); the session's GPU named
from env.log (else pod.json); item 3's one rule in the goal and the Speed tables; the host's PCIe link
(generation and width) in the host table; the two-turn chat with turn 2's first token and the HTTP check;
bench-e07b's phases. E07_PLAN=qfork and qengine (Part Q) write SHEET_ROOT/<UTC date>-qfork/ or
-qengine/: the server's command as run (out/quality-server.cmd, the engine's config; qfork: the fork server's peak
GPU memory, out/quality-gpu.csv), the HumanEval continuation
check, the quality results of out/quality/ (tools/dsv41/quality/q_report.py: the settings, each task's headline, the
run beside the API on 2026-10-06 with analyze.py, the needles, the 3 items for the hand check, the usage), the host and
the steps; qengine also the GPU clock summary per phase (pod/qdiag_report.py on the session samplers). E07_PLAN=qdiag
 writes SHEET_ROOT/<UTC date>-qdiag/: out/qdiag.md (pod/qdiag_report.py: per phase, the
engine's decode, doorbell waits, CPU path, hit rate, swaps and prefill beside the GPU's clocks, power and throttle
reasons and the host's cgroup throttling and load), the server as run, the continuation check, the two GSM8K runs
(NUM_CONCURRENT 1 and 2), the host and the steps. E07_PLAN=qdet writes
SHEET_ROOT/<UTC date>-qdet/: items 7 and 8 from out/det-sampling.json (quality/det_sampling_check.py's verdicts,
every request with its ids' count and decode, item 7's divergence record, the 256-token sampled reply for the hand
check; the mode-off row and the deterministic mode's decode cost), the server as run, the host and the steps.
E07_PLAN=cbase writes SHEET_ROOT/<UTC date>-cbase/: item 9's table by N from
out/cbase/load-c<N>.json (quality/chat_load.py's TTFT p50 and p95, queue wait p50 and p95, per-user decode tok/s,
aggregate tok/s; every request's row, the files under logs/cbase/), the server as run, the host and the steps.
E07_PLAN=cbatch writes SHEET_ROOT/<UTC date>-cbatch/: item 11 from out/cbatch/
(the single user at short context with batching off and on; N users together end to end, the user-visible figures with
the prompt reads included: per-user decode tok/s, the aggregate over the decode window and over the wall time; the
steady state, steady-c4.json: the 4 users decoding at once with exactly 4 rows a step, per repeat and against both
single-user baselines; ONE item 11 verdict line: the steady state against the single user
with batching off, the single user's regression and the end-to-end figures beside it), the batch's lines of the engine's log (its
steps, every slot's end with its stall and its reproducibility), items 7 and 8 once, both gates' placements.
E07_PLAN=clong writes SHEET_ROOT/<UTC date>-clong/: item 12 from
out/clong/load-long.json (quality/chat_load.py long: each user's largest gap between two tokens over its reply and over
the 64K read, <= 5 s; the 64K TTFT beside the users against its solo run), item 13 from out/clong/robust.json
(quality/robust_check.py's checks), the prompts read in chunks beside decoding slots from the engine's log (their
chunks, each chunk's read time, the steps between them, the longest hold), the gate, the server as run.
E07_PLAN=cbatch-clong writes SHEET_ROOT/<UTC date>-cbatch-clong/: cbatch's sheet (its batch
server's logs out/cbatch-<file>; no load-c<N> nor det-sampling, measured on 2026-10-14: the
item 11 line with the end-to-end figures of steady-c4's own rows (P1), the 8 users' steady state of steady-c8.json or,
without it, the 2026-10-14 figures labelled stale (P2), the baseline's change since the previous session and its machine
disclosed, the session's machine from pod.json) and then clong's sections,
each item with its own table and verdict; with the second cbatch server (
out/gate-spare, out/cbatch/load-short-spare.json, steady-c4-spare.json, out/spare-<file>) item 11 on each server side
by side against the same baseline, a verdict line for each and the rule over both.
E07_PLAN=cdet writes SHEET_ROOT/<UTC date>-cdet/: item 14 from
out/det-batch.json (quality/det_batch_check.py's phases), items 7 and 8 (det-sampling), the speed bars (the single user
with batching on against the measured 61.55 tok/s and >= 30, beside it the same pod's batching off; steady-c4's
aggregate against 2 x the single user with batching off), item 3 batched (out/quality, q_report's table) and item 4
(analyze.py same_engine_bar: the batched run against the engine quality run, CDET_Q02_ENGINE, per task the
interval including zero and the loss within 3 points) with the API fresh pair reported as in the engine quality run (analyze.py's table
against CDET_Q02_API), the batch's lines from the engine's log, both gates and servers, the host and the steps.
E07_PLAN=cfinal writes SHEET_ROOT/<UTC date>-cfinal/: cdet's sheet
less item 14's det_batch_check.py, items 7 and 8 and the A/B (none run): item 11 (the single user against the measured
61.55 tok/s and >= 30; steady-c4's aggregate >= 2 x the single user and every user >= 20 tok/s), item 3 batched, item 4
against the engine quality run, and item 14 read as no quality loss from batching: PASS when item 3 came back on all
three tasks and item 4 passes.
E07_PLAN=ccache writes SHEET_ROOT/<UTC date>-ccache/: item 10 from
out/ccache/load-alt<N>.json (the users taking turns back to back: every turn 2+ resumed, its first token <= 2 s when
not queued, the engine's own idle reset wait counted in it) beside out/ccache/
load-alt<N>-think<S>.json (the same with S seconds between turns, informational), each run's turns 2+
TTFT and reset wait, the cbase runs again by N against a cbase sheet's (CCACHE_BASELINE, default SHEET_ROOT/2026-10-13-cbase-2320/logs/cbase:
TTFT p50 and p95, the turns 2+ resumed, before and after), the cost per conversation (its checkpoint slot on the host
and what a 16K turn fills of it, the live KV and the carry snapshot on the device), apart from it the host's fixed
costs (the homes, the warm tier, the engram tables, working memory, the margin) and the total host RAM the gate placed
against what homes for every slab expert would need, every request's checkpoints (restored, saved, evicted, MiB, ms,
the idle reset waited for), the hot set's path (host homes or two-way, the gate's reason,
and per request the full reset's ms on the
device and the per-layer waits the next request saw), items 7 and 8 once
(det-sampling), the session's real cost against its cap (cost.txt, OUT's session-cap),
the server as run, the host and the steps. For
qengine and qdiag, out/qdiag.md and qdiag.json are written into
OUT before the logs are copied. Every sheet has the machine qualification of
each pod the session tried, passed or rejected, from OUT's probe.log; a session that never started its run (every
machine rejected) gets its sheet from that alone, its plan from OUT's session-plan. A session whose final pull brought
nothing back (the pod lost mid-run) gets its sheet from the last mid-run pull of out/, with a "Session lost"
section first: OUT's session-lost (e07.sh), the step it was lost after, the steps pulled and the ones lost. python3
only.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quality"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qdiag_report  # noqa: E402

GOALS = {"16k": ("TTFT", 5000, "<= 5 s"), "turn2": ("TTFT", 1000, "<= 1 s (accepted at about 2 s)"),
         "short": ("decode", 30, ">= 30 tok/s"), "256k": ("TTFT", 180000, "<= 3 min, no crash"),
         "short-static": ("decode", 30, ">= 30 tok/s"), "short-prompt": ("decode", 30, ">= 30 tok/s"),
         "short-adaptive": ("decode", 30, ">= 30 tok/s"), "16k-e09": ("TTFT", 5000, "<= 5 s")}
SHORT_POLICIES = ("short-static", "short-prompt", "short-adaptive")
STEP_RE = re.compile(r"^step (\S+): (.*)$", re.M)
# tests/dsv41/engine/machine_probe.cu's lines, after pod-session.sh's "=== pod" line per pod in probe.log, then
# tests/dsv41/engine/cpu_experts_probe.cpp's after its "qualification: ...: CPU experts floor" line
POD_RE = re.compile(r"^=== pod (\S+) machine (\S+) \((.*)\) at (\S+) UTC$")
MEASURE_RE = re.compile(r"^dsv41_(?:machine|cpu_experts)_probe: measure (\S+) (\S+) bound (\S+) (ok|FAIL|info)(?: \((.*)\))?$")
CPU_FLOOR_RE = re.compile(r"^qualification: \S+ \(.*\): CPU experts floor ")
QUALIFY_MEASURES = (("gpu_read_gbs", "GPU read GB/s"), ("h2d_gbs", "H2D GB/s"), ("d2h_gbs", "D2H GB/s"),
                    ("doorbell_median_us", "doorbell median us"), ("doorbell_p99_us", "doorbell p99 us"),
                    ("host_read_gbs", "host read GB/s"), ("cpu_experts_gbs", "CPU experts GB/s"),
                    ("sm_clock_mhz", "SM MHz"), ("sm_clock_max_mhz", "SM max MHz"),
                    ("power_draw_w", "power W"), ("power_limit_w", "power limit W"),
                    ("power_default_limit_w", "default limit W"))
STEP_EXIT_RE = re.compile(r"^exit (\d+), (\d+) s(?: \(timeout after (\d+) s: .*\))?$")


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read(p):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def fmt(v, nd=1):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pcie_link(env):
    """The host's PCIe link from env.log: nvidia-smi's link maximum and current generation and width (the maximum is
    the lower of the GPU's and the host's), and the host's and the GPU's own maxima when the driver reports them."""
    gpu = re.search(r"^name, memory.total.*\n(.+)$", env, re.M)
    f = [x.strip() for x in gpu.group(1).split(",")] if gpu else []
    if len(f) < 6:
        return "n/a (no GPU line in env.log)"
    s = f"Gen {f[2]} x{f[3]} (link maximum; at env's idle moment Gen {f[4]} x{f[5]})"
    host = re.search(r"^pcie.link.gen.hostmax, pcie.link.gen.gpumax.*\n(\d+), (\d+), (\d+)$", env, re.M)
    if host:
        s += f"; the host's maximum Gen {host.group(1)}, the GPU's Gen {host.group(2)}"
    elif "host PCIe link" in env:
        s += "; the host's and the GPU's own maxima not reported by this driver"
    return s


def session_gpu(o, out_dir):
    """The session's GPU by name: env.log's nvidia-smi line (the host table's source), else pod.json's gpu."""
    gpu = re.search(r"^name, memory.total.*\n([^,\n]+)", read(os.path.join(o, "env.log")), re.M)
    if gpu:
        return gpu.group(1).strip()
    return (load_json(os.path.join(out_dir, "pod.json")) or {}).get("gpu") or "GPU not recorded (no env.log GPU line, no pod.json)"


def turn2_verdict(first_s):
    """Item 3's one rule, for the goal table and the E07b Speed table: PASS within the
    accepted bar of about 2 s, with a note when the goal's strict 1 s is not met, FAIL above it."""
    if first_s <= 1.0:
        return "PASS", ""
    if first_s <= 2.0:
        return "PASS", "within the accepted bar of about 2 s, the goal's strict 1 s not met"
    return "FAIL", f"first token {first_s:.2f} s, over the accepted bar of about 2 s"


def host_section(o):
    env = read(os.path.join(o, "env.log"))
    cpu = re.search(r"^Model name:\s*(.+)$", env, re.M)
    gpu = re.search(r"^name, memory.total.*\n(.+)$", env, re.M)
    lim = re.findall(r"^(/sys/fs/cgroup/\S+): (\S+)$", env, re.M)
    mem = re.search(r"^MemTotal:\s*(\d+) kB", env, re.M)
    nproc = re.search(r"^(\d+)$", env, re.M)
    rows = [("CPU", cpu.group(1).strip() if cpu else "n/a"), ("nproc (the pod's CPUs)", nproc.group(1) if nproc else "n/a"),
            ("MemTotal (the host)", f"{int(mem.group(1)) / (1 << 20):.1f} GiB" if mem else "n/a"),
            ("memory cgroup", "; ".join(f"{k} = {v}" for k, v in lim) or "n/a"),
            ("GPU (name, memory, PCIe gen/width max and current, driver)", gpu.group(1).strip() if gpu else "n/a"),
            ("host PCIe link (generation, width)", pcie_link(env))]
    return table(["", ""], rows)


def steps_of(o):
    """out/steps.log as {step: its line's rest} (the last line of each) and the stop lines."""
    text = read(os.path.join(o, "steps.log"))
    return dict(STEP_RE.findall(text)), [ln for ln in text.splitlines() if ln.startswith("stop:")]


def cause(o, steps, what):
    """Why the steps (one name, or the chain that makes WHAT, in order) left no WHAT: the first one's out/steps.log
    line that did not pass (a timeout, a failure, a skip), or the stop before a step that never ran."""
    lines, stops = steps_of(o)
    for step in [steps] if isinstance(steps, str) else steps:
        r = lines.get(step)
        if r is None:
            return f"{step} did not run: {stops[-1] if stops else 'out/steps.log has no line for it'}"
        if r.startswith("skipped"):
            return f"{step} {r} before the deadline"
        m = STEP_EXIT_RE.match(r)
        if not m:
            return f"{step}: {r}"
        rc, secs, budget = int(m.group(1)), m.group(2), m.group(3)
        if budget:
            return f"timeout after {budget} s in {step}"
        if rc == 124:
            return f"timeout after {secs} s in {step} (the session's deadline)"
        if rc:
            return f"{step} failed (exit {rc} after {secs} s, logs/{step}.log)"
    return f"{step} passed but left no {what}"


def parity_section(o, e07b=False):
    # session 1's four runs, then the isolated MoE run and the MMQ prefill run (only when its step ran); E07b:
    # dense free and the isolated run, each missing one with its cause
    runs = [(m, sel, f"parity-{m}-{sel}.json") for m in ("dense", "reuse") for sel in ("forced", "free")]
    if e07b:
        runs = [("dense", "free", "parity-dense-free.json"),
                ("dense", "isolated (the oracle's cur in every layer)", "parity-moe-isolated.json")]
    # E07_READERS skips modes on purpose (e07-run.sh writes them to out/plan.log): those rows say so, not "no result"
    readers = re.search(r"^plan: \S+, readers: (.+)$", read(os.path.join(o, "plan.log")), re.M)
    chosen = readers.group(1).split() if readers else ("dense", "reuse")
    optional = [("dense", "isolated (the oracle's cur in every layer)", "parity-moe-isolated.json"),
                ("dense", "forced, prefill on MMQ", "parity-dense-forced-mmq.json")]
    runs += [] if e07b else [r for r in optional if os.path.exists(os.path.join(o, r[2][:-5] + ".log"))]
    rows, got = [], {}
    for mode, sel, name in runs:
        if mode not in chosen:
            rows.append((mode, sel, f"not run (E07_READERS={' '.join(chosen)})", "", "", "", "", ""))
            continue
        p = load_json(os.path.join(o, name))
        if not p:
            chain = ["dense-free", "parity-dense-free"] if sel == "free" else \
                ["forced-ref", "moe-isolated", "parity-moe-isolated"]
            rows.append((mode, sel, cause(o, chain, name) if e07b else "no result (see the step's log)",
                         "", "", "", "", ""))
            continue
        got[name] = p
        t, d = p["tokens"], p["tokens"]["first_divergence"]
        div = "none" if not d else (f"step {d['step']}: {d['engine']} vs {d['oracle']}, gap {d['gap']:.4g} "
                                    f"(tie bound {d['tie_bound']:.4g}, {'near-tie' if d['near_tie'] else 'NOT a near-tie'})")
        rows.append((mode, sel, f"{p['worst_cur']['rel_l2']:.3g} (L{p['worst_cur']['layer']})",
                     f"{p['worst_out']['rel_l2']:.3g} (L{p['worst_out']['layer']})", p["router_flips"],
                     f"{t['identical']} of {t['of']} (first {t['prefix']} in a row)", div,
                     "PASS" if p["pass"] else "FAIL"))
    s = table(["readers", "selection", "worst cur", "worst out", "router flips", "greedy ids = oracle's",
               "first divergence", "verdict"], rows)
    s += ("\n\nForced selection is the correctness gate (cur and out <= 2e-2 in every layer); free greedy must keep >= 90%"
          " of the 64 ids, a first divergence explained by a near-tie. At 95 positions the"
          " two reader modes coincide (SPEC.md \"Reader modes\"). The isolated run feeds every layer's MoE the oracle's"
          " cur: its out and weights are the router and the routed experts alone, the per-layer isolated rel L2"
          ".")
    base = got.get("parity-dense-forced.json")
    iso = got.get("parity-moe-isolated.json")
    mmq = got.get("parity-dense-forced-mmq.json")
    if base or iso:
        head = ["layer"]
        if base:
            head += ["forced: cur rel L2", "out rel L2", "weights rel L2", "expert sets equal"]
        if iso:
            head += ["isolated: out rel L2", "weights rel L2", "own top-k = oracle's"]
        if mmq:
            head += ["forced on MMQ: cur", "out"]
        n = len((base or iso)["layers"])
        trows = []
        for il in range(n):
            r = [il]
            if base:
                b = base["layers"][il]
                r += [f"{b['cur']:.3g}", f"{b['out']:.3g}", "n/a" if b["weights"] is None else f"{b['weights']:.3g}",
                      f"{b['topk_same']} of {b['positions']}"]
            if iso:
                x = iso["layers"][il]
                r += [f"{x['out']:.3g}", "n/a" if x["weights"] is None else f"{x['weights']:.3g}",
                      f"{x.get('own_topk_same', 'n/a')} of {x['positions']}"]
            if mmq:
                m = mmq["layers"][il]
                r += [f"{m['cur']:.3g}", f"{m['out']:.3g}"]
            trows.append(r)
        s += "\n\nPer layer, readers dense:\n\n" + table(head, trows)
    return s


def isolation_section(o):
    """layer_parity.py's layer-0 isolation (out/layer-parity.json and, after the MMQ run, layer-parity-mmq.json)."""
    j = load_json(os.path.join(o, "layer-parity.json"))
    if not j:
        return "no result (see layer-parity.log)" if os.path.exists(os.path.join(o, "layer-parity.log")) else \
            "not run in this session"
    out = [table(["layer", "numerics", "cur vs oracle", "out vs oracle", "weights", "own top-k", "isolated out",
                  "isolated weights", "isolated own top-k"],
                 [(r["layer"], r["numerics"], f"{r['cur']:.3g}", f"{r['out']:.3g}", f"{r['weights']:.3g}",
                   f"{r['topk_same']} of {j['n_tokens']}", "n/a" if "out_isolated" not in r else f"{r['out_isolated']:.3g}",
                   "n/a" if "weights_isolated" not in r else f"{r['weights_isolated']:.3g}",
                   "n/a" if "topk_same_isolated" not in r else f"{r['topk_same_isolated']} of {j['n_tokens']}")
                  for r in j["oracle_rows"]])]
    for name in ("layer-parity.json", "layer-parity-mmq.json"):
        jj = load_json(os.path.join(o, name))
        if not jj or not jj["engine_rows"]:
            continue
        rows = []
        for r in jj["engine_rows"]:
            for n, a in r["against"].items():
                rows.append((r["layer"], r["run"], f"{r['cur_oracle']:.3g}", n, f"{a['cur']:.3g}", f"{a['out']:.3g}",
                             f"{a['l_last']:.3g}"))
        out.append(f"The engine's call 0 ({name}):\n\n" + table(["layer", "run", "cur vs oracle", "reference",
                                                                   "cur vs reference", "out", "l_last"], rows))
    ops = [x for x in j.get("ops", []) if x["layer"] in (-1, 0)]
    if ops:
        pairs = list(ops[0]["rel_l2"])
        out.append("Layer 0 op by op (rel L2 between numerics, each op from its own numerics' previous ones):\n\n" +
                   table(["op"] + pairs, [(f"{x['op']}.{x['occurrence']}",) + tuple(f"{x['rel_l2'][p]:.3g}" for p in pairs)
                                          for x in ops]))
    notes = []
    for p in j["pairs"]:
        f, w = p["first_at_1e-3"], p["largest_before_l0_cur"]
        notes.append(f"{p['a']} vs {p['b']}: first op at or above 1e-3 "
                     f"{'none' if not f else f['op'] + ' of layer ' + str(f['layer']) + ' (' + format(f['rel_l2'], '.3g') + ')'}"
                     f", the largest up to layer 0's cur "
                     f"{'n/a' if not w else w['op'] + ' (' + format(w['rel_l2'], '.3g') + ')'}")
    notes.append(f"the oracle's matmul arithmetic (fork-cuda) explains the layer-0 gap: {j.get('mmq_worth_measuring')}"
                 f" (then dense-forced-mmq runs; out/plan.log)")
    for g in j.get("gpu_suspect", []):
        notes.append(f"{g['run']}: layer 0 cur {g['gpu_vs_reference']:.3g} from its engine-numerics reference, "
                     f"{g['gpu_vs_oracle']:.3g} from the oracle: {'SUSPECT the GPU path' if g['suspect'] else 'explained by its numerics'}")
    out.append("\n".join(f"- {n}" for n in notes))
    plan = read(os.path.join(o, "plan.log")).strip()
    if plan:
        out.append("out/plan.log: " + "; ".join(plan.splitlines()))
    return "\n\n".join(out)


def hot_section(o):
    """E08: bench-e08's hot set per phase, the hit rate per layer of the short phases, the gate, the 16K split."""
    j = load_json(os.path.join(o, "bench-e08.json"))
    if not j:
        return "no result (see bench-e08.log)"
    rows, per_layer, best = [], {}, None
    for p in j["phases"]:
        h = p.get("hot")
        if not h:
            continue
        st = max(p["decode_steps"], 1)
        gbs = h["swap_mib"] * (1 << 20) / (h["swap_copy_ms"] * 1e6) if h["swap_copy_ms"] > 0 else None
        x = p.get("decode_experts", {})
        # the client's first token with and without the bench's hot_reset (sessions before e07_bench.py wrote
        # ttft_client_no_reset_ms: the same difference from their fields)
        ttft = p.get("ttft_client_ms")
        bare = p.get("ttft_client_no_reset_ms", None if ttft is None else ttft - h["reset_ms"])
        rows.append((p["phase"], h["policy"], fmt(x.get("hit_rate"), 4), f"{h['swaps'] / st:.2f}",
                     f"{h['swap_mib'] / st:.1f}", fmt(gbs, 2), h["batches"], h["deferred"],
                     f"{h['reset_swaps']} in {h['reset_ms']:.0f} ms", fmt(None if ttft is None else ttft / 1e3, 2),
                     fmt(None if bare is None else bare / 1e3, 2), fmt(p["prompt_ms"] / 1e3, 2),
                     fmt(p["decode_tok_s"], 2)))
        if p["phase"] in SHORT_POLICIES:
            per_layer[p["phase"]] = h["layer_hit_rate"]
            if p["decode_tok_s"] is not None and (best is None or p["decode_tok_s"] > best[1]):
                best = (p["phase"], p["decode_tok_s"])
    out = [table(["phase", "policy", "hit rate", "swaps per step", "MiB per step", "copy GB/s", "batches",
                  "deferred boundaries", "reset before it", "first token s (client)",
                  "first token s (client, without the reset)", "TTFT s (engine)", "decode tok/s"], rows),
           "The bench's hot_reset=1 runs before the prompt, so the client's first token includes it; a server's request "
           "carries the slab over and pays no reset: its first token is the column without the reset (the client's "
           "less the reset's ms)."]
    if best:
        out.append(f"Gate: short-context decode over 480 tokens with the best policy "
                   f">= 30 tok/s: {best[0]} {best[1]:.2f} tok/s: {'PASS' if best[1] >= 30 else 'FAIL'}.")
    if per_layer:
        names = [n for n in SHORT_POLICIES if n in per_layer]
        n_layer = max(len(v) for v in per_layer.values())
        out.append("Decode hit rate per layer (short context, 480 steps, each policy from the hot set's slab):\n\n" +
                   table(["layer"] + names, [[l] + [fmt(per_layer[n][l], 3) if l < len(per_layer[n]) else "n/a"
                                                    for n in names] for l in range(n_layer)]))
    for p in j["phases"]:
        f = p.get("prefill_split")
        if f:
            out.append(f"{p['phase']}: the prompt of {p['prompt_tokens']} ids timed per stage (prefill_times=1, TTFT "
                       f"{fmt(p['prompt_ms'] / 1e3, 2)} s timed): {f['line']}")
    if os.path.exists(os.path.join(o, "route-trace.txt.gz")):
        out.append("Routing trace: logs/route-trace.txt.gz (every phase but 64k and 256k; "
                   "`tools/dsv41/engine/hot_policy_sim.py compare <trace> --hot-set logs/gate/hot-set.json`).")
    return "\n\n".join(out)


def prefill_section(o, bench="bench-e09"):
    """E09 (E07b: bench-e07b): the bench's 16K prompts split by stage (the engine's prefill_times line) against the
    gate, then the two-turn chat (E07b: with its HTTP check)."""
    j = load_json(os.path.join(o, f"{bench}.json"))
    out = []
    if not j:
        out.append(f"{bench}: no result ({cause(o, bench, bench + '.json')})" if bench == "bench-e07b" else
                   f"{bench}: no result (see {bench}.log)")
    else:
        rows = []
        for p in j["phases"]:
            f = p.get("prefill_split")
            if p["phase"] != "16k-e09" or not f:
                continue
            g = lambda k: f.get(k)  # noqa: E731
            gaps = None if g("plan_gap_ms") is None or g("stall_ms") is None else g("plan_gap_ms") + g("stall_ms")
            rows.append((p["phase"], p["prompt_tokens"], p.get("reused", 0), fmt(p["prompt_ms"] / 1e3, 2),
                         fmt(None if p.get("ttft_client_no_reset_ms") is None else p["ttft_client_no_reset_ms"] / 1e3, 2),
                         fmt(g("attention_ms")),
                         fmt(None if g("attention_ms") is None else g("attention_ms") / 40, 1), fmt(g("dense_ms")),
                         fmt(g("routed_experts_ms")), fmt(g("experts_compute_ms")), fmt(g("plan_gap_ms")),
                         fmt(g("stall_ms")), fmt(gaps), fmt(g("copy_gb"), 2), fmt(g("copy_ms")),
                         "PASS" if p["prompt_ms"] <= 5000 else "FAIL"))
        out.append(table(["phase", "prompt ids", "reused", "TTFT s (engine)", "first token s (client, without the reset)",
                          "front attention ms", "per layer", "dense ms", "routed experts ms", "expert compute ms",
                          "plan gaps ms", "stalls ms", "gaps + stalls", "copies GB", "copy ms", "16K TTFT <= 5 s"], rows))
        out.append("Targets beside the gate (from the same split): front attention <= 30 ms per "
                   "layer, expert compute <= 2.4 s, plan gaps + stalls <= 0.6 s. Every line in full:")
        for p in j["phases"]:
            if p.get("prefill_split"):
                out.append(f"- {p['phase']} ({p['prompt_tokens']} ids, TTFT {fmt(p['prompt_ms'] / 1e3, 2)} s timed): "
                           f"{p['prefill_split']['line']}")
    t = load_json(os.path.join(o, "two-turn", "two-turn.json"))
    if not t:
        out.append(f"two-turn: no result ({cause(o, 'two-turn', 'two-turn/two-turn.json')})" if bench == "bench-e07b"
                   else "two-turn: no result (see two-turn.log)")
    else:
        rows = []
        for i, x in enumerate(t["turns"]):
            d = x.get("decode", {})
            rows.append((i + 1, x["prompt_tokens"], x["cached_tokens"], x.get("resume_kind", "n/a"), x.get("reused", "n/a"),
                         fmt(x.get("prompt_ms")), x["completion_tokens"], fmt(d.get("tok_s"), 2),
                         fmt(d.get("hit_rate"), 4), fmt(x.get("first_token_s"), 3), x["client_s"]))
        out.append("Two-turn chat through the server (engine/two_turn_check.py; turn 2 streamed since E07b):\n\n" +
                   table(["turn", "prompt tokens", "cached", "engine resume", "reused", "prompt ms (engine)",
                          "generated", "decode tok/s", "hit rate", "first token s (client)", "client s"], rows))
        r = [x["trace"] for x in t["turns"] if "trace" in x] or t.get("trace_requests", [])[-2:]
        out.append(f"Route trace R lines of the two turns: {r}; verdict: {'PASS' if t.get('ok') else 'FAIL: ' + '; '.join(t.get('failed', []))}")
        h = t.get("http")
        if h:
            ns, st = h.get("non_streaming") or {}, h.get("streaming") or {}
            out.append(f"HTTP check on the same server load (non-streaming, streaming, tools refused): "
                       f"{'PASS' if h['ok'] else 'FAIL: ' + ', '.join(h['failed'])}; non-streaming {ns.get('content')!r} "
                       f"({ns.get('usage')}), streaming {st.get('content')!r} (first token {st.get('first_token_s')} s); "
                       f"the two replies' ids equal: {h.get('same_ids', 'n/a')} (reported)")
    return "\n\n".join(out)


def goal_section(o, out_dir, sheet):
    """E07b: one row per goal "Done means" item 1-6, the number, PASS, FAIL or REPORTED,
    and the cause of a FAIL or a missing result."""
    b = load_json(os.path.join(o, "bench-e07b.json")) or {}
    ph = {p["phase"]: p for p in b.get("phases", [])}
    t = load_json(os.path.join(o, "two-turn", "two-turn.json"))

    def missing(phase):
        """A bench phase that did not come back: the bench's own error, else the step's cause."""
        why = b.get("error") or cause(o, "bench-e07b", "bench-e07b.json")
        return f"{phase}: no result ({why})"

    rows = []
    # 1. correctness on this card: reported, not gating (E07a's A100: 7 of 64)
    free, iso = load_json(os.path.join(o, "parity-dense-free.json")), load_json(os.path.join(o, "parity-moe-isolated.json"))
    num, why = [], []
    if free:
        tk = free["tokens"]
        num.append(f"free greedy {tk['identical']} of {tk['of']} ids = the stock fork's (E07a A100: 7 of 64)")
    else:
        why.append(cause(o, ["dense-free", "parity-dense-free"], "parity-dense-free.json"))
    if iso:
        num.append(f"isolated MoE worst out rel L2 {iso['worst_out']['rel_l2']:.3g} (L{iso['worst_out']['layer']})")
    else:
        why.append(cause(o, ["forced-ref", "moe-isolated", "parity-moe-isolated"], "parity-moe-isolated.json"))
    gpu = session_gpu(o, out_dir)
    rows.append((1, f"correctness on the real model ({gpu})", "; ".join(num) or "n/a",
                 "REPORTED" if num else "FAIL", "; ".join(why) or "reported, not gating"))
    # 2. prefill: 16K <= 5 s, 256K <= 3 min, the engine still answering
    p16, p256 = ph.get("16k-e09"), ph.get("256k")
    num, why, ok = [], [], True
    for p, name, lim in ((p16, "16k-e09", 5000), (p256, "256k", 180000)):
        if p:
            num.append(f"{name} TTFT {p['prompt_ms'] / 1e3:.2f} s (engine)")
            if p["prompt_ms"] > lim:
                ok = False
                why.append(f"{name} over {lim / 1e3:.0f} s")
        else:
            ok = False
            why.append(missing(name))
    if p256 and b.get("alive_after") is not True:
        ok = False
        why.append("the engine did not answer after 256k" if "alive_after" in b else missing("the check after 256k"))
    rows.append((2, "prefill: 16K <= 5 s, 256K <= 3 min without a crash", "; ".join(num) or "n/a",
                 "PASS" if ok else "FAIL", "; ".join(why)))
    # 3. second turn of the 16K conversation (+ <= 512 tokens) <= 1 s, accepted at about 2 s
    p2 = ph.get("turn2")
    tt = t["turns"][1] if t and len(t.get("turns", [])) == 2 else None
    # the bench's first token gates (E07a's phase: 16,415 reused + 513 new); without it the server's turn 2 is only
    # reported (never over-claim)
    num, why, notes = [], [], []
    first = None
    if p2:
        first = p2.get("ttft_client_ms", 0) / 1e3
        num.append(f"bench turn2 first token {first:.2f} s at the client ({p2.get('reused')} reused + "
                   f"{p2['prompt_tokens'] - p2.get('reused', 0)}, engine {p2['prompt_ms'] / 1e3:.2f} s)")
        if not p2.get("resume"):
            why.append("bench turn2 was read cold (no RESUME)")
    else:
        notes.append(missing("turn2"))
    if tt:
        num.append(f"server turn 2 first token {fmt(tt.get('first_token_s'), 2)} s at the client ({tt['cached_tokens']} of "
                   f"{tt['prompt_tokens']} cached, R line resume={(tt.get('trace') or {}).get('resume', 'n/a')})")
        if not t.get("turns_ok", t.get("ok")):
            why.append("server two-turn: " + "; ".join(t.get("failed", [])))
    else:
        notes.append(f"server two-turn: no result ({cause(o, 'two-turn', 'two-turn/two-turn.json')})")
    if first is None:
        v = "REPORTED" if tt and not why else "FAIL"
        if tt:
            notes.append(f"server turn only: first token {fmt(tt.get('first_token_s'), 2)} s")
    elif why:
        v = "FAIL"
    else:
        v, note = turn2_verdict(first)
        why += [note] if note else []
    rows.append((3, "prefix cache: second turn of a 16K conversation", "; ".join(num) or "n/a", v,
                 "; ".join(why + notes)))
    # 4. decode >= 30 tok/s over 480 tokens at short context, reported at 64K and 256K
    ps = ph.get("short-adaptive")
    num, why = [], []
    for p, name in ((ps, "short-adaptive"), (ph.get("64k"), "64k"), (p256, "256k")):
        if p and p.get("decode_tok_s") is not None:
            num.append(f"{name} {p['decode_tok_s']:.2f} tok/s over {p['decode_steps']} steps")
        else:
            why.append(missing(name))
    v = "FAIL" if not ps or ps.get("decode_tok_s") is None else "PASS" if ps["decode_tok_s"] >= 30 else "FAIL"
    if ps and ps.get("decode_tok_s") is not None and ps["decode_tok_s"] < 30:
        why.append("short-context decode under 30 tok/s")
    rows.append((4, "decode: >= 30 tok/s at short context, 64K and 256K reported", "; ".join(num) or "n/a", v,
                 "; ".join(why)))
    # 5. the server answers /v1/chat/completions, streaming and not
    h = (t or {}).get("http")
    if h:
        rows.append((5, "served: /v1/chat/completions streaming and non-streaming",
                     "non-streaming and streaming replies the text of the engine's ids, tools refused",
                     "PASS" if h["ok"] else "FAIL", "" if h["ok"] else "failed: " + ", ".join(h["failed"])))
    else:
        rows.append((5, "served: /v1/chat/completions streaming and non-streaming", "n/a", "FAIL",
                     cause(o, "two-turn", "HTTP check (two-turn/two-turn.json)")))
    # 6. recorded, the pod terminated (the session's cost line is written by its delete)
    cost = read(os.path.join(out_dir, "cost.txt")).strip()
    rows.append((6, "recorded: the session sheet, its parameters and cost; pods terminated", f"{sheet}; {cost or 'no cost'}",
                 "PASS" if cost else "FAIL",
                 "" if cost else "no cost.txt: the pod's delete is not recorded (check the RunPod console: no pod left running)"))
    return (table(["item", "goal (Done means)", "measured", "verdict", "cause"], rows) +
            f"\n\nEvery number: the {gpu} session of this sheet (env.log's GPU line, as in the host table); sources "
            "logs/bench-e07b.json, logs/two-turn/two-turn.json, logs/parity-*.json, logs/steps.log.")


def speed_section(o, benches=("bench-a", "bench-b"), e07b=False):
    rows, notes, dec = [], [], []
    for b in benches:
        j = load_json(os.path.join(o, f"{b}.json"))
        if not j:
            notes.append(f"{b}: no result (see {b}.log)")
            continue
        notes.append(f"{b}: load {fmt(j.get('load_s'))} s; the engine's measured rates {j.get('rates', {})}; "
                     f"after the load {j.get('after_load', {})}; still answering after the last phase: "
                     f"{j.get('alive_after', 'n/a')}; {'ok' if j.get('ok') else 'FAIL: ' + str(j.get('error'))}")
        for p in j["phases"]:
            goal, verdict = GOALS.get(p["phase"], (None, None, ""))[0], ""
            if e07b and p["phase"] == "turn2":
                # the goal table's rule and number (the client's first token), so the two tables agree
                verdict, note = turn2_verdict(p.get("ttft_client_ms", 0) / 1e3)
                verdict += f" ({note})" if note else ""
            elif goal == "TTFT":
                verdict = "PASS" if p["prompt_ms"] <= GOALS[p["phase"]][1] else "FAIL"
            elif goal == "decode" and p["decode_tok_s"] is not None:
                verdict = "PASS" if p["decode_tok_s"] >= GOALS[p["phase"]][1] else "FAIL"
            rows.append((p["phase"], p["prompt_tokens"], p.get("reused", 0), fmt(p["prompt_ms"] / 1e3, 2),
                         fmt(p.get("ttft_client_ms", 0) / 1e3, 2), fmt(p["prompt_tok_s"]), fmt(p["decode_tok_s"], 2),
                         p["decode_steps"], fmt(p.get("storage_read_mib")), fmt(p.get("major_faults")),
                         fmt(p.get("vram_used_mib")), fmt(p.get("rss_mib")), fmt(p.get("cgroup_usage_gib")),
                         f"{GOALS[p['phase']][2]}: {verdict}" if p["phase"] in GOALS else ""))
            x = p.get("decode_experts")
            if x:
                st = max(x["steps"], 1)
                dec.append((p["phase"], x["steps"], f"{x['hit_rate']:.4f}", f"{x['misses'] / st:.2f}",
                            f"{x['cpu_jobs'] / st:.2f}", f"{x['cpu_ms_per_step']:.3f}", fmt(x["cpu_gbs"], 2),
                            f"{x['doorbell_wait_ms'] / st:.3f}", f"{x['plan_ms'] / st:.3f}", f"{x['ms_per_step']:.3f}"))
            if p.get("timed"):
                notes.append(f"{b} {p['phase']}: ms per decode step by stage (timed: every op launched, no token graph): "
                             + ", ".join(f"{k} {v}" for k, v in p["timed"]["ms_per_step"].items()))
    s = table(["phase", "prompt ids", "reused", "TTFT s (engine)", "first token s (client)", "prompt tok/s",
               "decode tok/s", "decode steps", "storage read MiB", "major faults", "VRAM MiB", "RSS MiB",
               "cgroup GiB", "goal"], rows)
    if dec:
        s += ("\n\nDecode, the routed experts per phase (the engine's log of each request):\n\n" +
              table(["phase", "steps", "hit rate", "CPU misses per step", "CPU jobs per step", "CPU ms per step",
                     "CPU GB/s", "doorbell wait ms per step", "plan ms per step", "ms per step"], dec))
    return s + "\n\n" + "\n".join(f"- {n}" for n in notes)


def probe_pods(out_dir):
    """Each pod probe.log names, in order: its pod id, machine, offer, start, every machine_probe and cpu_experts_probe
    measure (key -> (value, bound, ok|FAIL|info)), whether the CUDA probe passed, the machine probe's verdict line and,
    when the CPU experts floor was set, the CPU experts probe's (None: it gave none)."""
    pods = []
    for ln in read(os.path.join(out_dir, "probe.log")).splitlines():
        m = POD_RE.match(ln)
        if m:
            pods.append({"pod": m.group(1), "machine": m.group(2), "offer": m.group(3), "at": m.group(4), "m": {},
                         "cuda": False, "verdict": None, "cpu_floor": False, "cpu_verdict": None})
            continue
        if not pods:
            continue
        p = pods[-1]
        if ln.startswith("dsv41_cuda_probe: ok"):
            p["cuda"] = True
        m = MEASURE_RE.match(ln)
        if m:
            p["m"][m.group(1)] = (m.group(2), m.group(3), m.group(4))
        elif ln.startswith(("dsv41_machine_probe: PASS", "dsv41_machine_probe: FAIL")):
            p["verdict"] = ln[len("dsv41_machine_probe: "):]
        elif CPU_FLOOR_RE.match(ln):
            p["cpu_floor"] = True
        elif ln.startswith(("dsv41_cpu_experts_probe: PASS", "dsv41_cpu_experts_probe: FAIL")):
            p["cpu_verdict"] = ln[len("dsv41_cpu_experts_probe: "):]
    return pods


def qualification_section(out_dir):
    """The machine qualification: each pod the session tried, its machine and the probe's every measure
    (a value past its bound marked FAIL), then the verdict: passed, unfit (the measures that failed), the CUDA probe
    failed, or no qualification (QUALIFY=0)."""
    pods = probe_pods(out_dir)
    if not pods:
        if not os.path.exists(os.path.join(out_dir, "probe.log")):
            return "no pod reached SSH (no probe.log)"
        return "probe.log names no pod (a session from before the qualification)"
    rows = []
    for p in pods:
        cells = []
        for key, _ in QUALIFY_MEASURES:
            v = p["m"].get(key)
            cells.append("" if v is None else v[0] if v[2] != "FAIL" else f"**{v[0]} FAIL** ({v[1]})")
        if p["verdict"] and p["verdict"].startswith("PASS") and p["cpu_floor"] and not p["cpu_verdict"]:
            verdict = "rejected: the CPU experts probe gave no verdict"
        elif p["verdict"] and p["verdict"].startswith("PASS") and p["cpu_verdict"] and not p["cpu_verdict"].startswith("PASS"):
            verdict = "rejected: " + p["cpu_verdict"].removeprefix("FAIL, ")
        elif p["verdict"] and p["verdict"].startswith("PASS"):
            verdict = "passed"
        elif p["verdict"]:
            verdict = "rejected: " + p["verdict"].removeprefix("FAIL, ")
        elif not p["cuda"]:
            verdict = "rejected: the CUDA probe failed"
        else:
            verdict = "not qualified (QUALIFY=0, or no verdict came back)"
        rows.append([p["pod"], p["machine"], p["offer"]] + cells + [verdict])
    # the bounds per measure in the order the pods met them: the GPU read's is per model, the SM clock's and the
    # power limit's are ratios of each GPU's own maximum
    bounds = {}
    for p in pods:
        for k, v in p["m"].items():
            if v[1] != "none" and v[1] not in bounds.setdefault(k, []):
                bounds[k].append(v[1])
    return (table(["pod", "machine", "offer"] + [h for _, h in QUALIFY_MEASURES] + ["verdict"], rows) + "\n\n" +
            "Bounds: " + ", ".join(f"{h} {' / '.join(bounds[k])}" for k, h in QUALIFY_MEASURES if bounds.get(k)) +
            " (pod/pod-session.sh runs tests/dsv41/engine/machine_probe.cu after the CUDA probe; the GPU read's bound "
            "is per GPU model, none for a model the plans do not name; the SM clock's and the power limit's are "
            "QUALIFY_MIN_SM_RATIO and QUALIFY_MIN_POWER_RATIO of the GPU's own clocks.max.sm and power.default_limit; "
            "a passing machine then runs tests/dsv41/engine/cpu_experts_probe.cpp, its bound CPU_EXPERTS_FLOOR_GBS "
            "; logs/probe.log has each measure's method).")


def quality_head(o, plan, stamp, build, cost, model):
    """Part Q (qfork, qengine): the server as run, the continuation check and the quality results."""
    import q_report
    fork = plan == "qfork"
    cont = [x for x in read(os.path.join(o, "continuation.log")).splitlines()
            if x.startswith(("ok ", "FAIL", "continuation.py"))]
    server = read(os.path.join(o, "quality-server.cmd")).strip() or "not started"
    cfg = read(os.path.join(o, "quality-server.json")).strip()
    if fork:
        # the slots and lm-eval's concurrency as run (Q_FORK_NP, 1 by default;
        # the 2026-10-11 session ran 4)
        np_ = re.search(r"(?:^| )-np (\d+)", server)
        conc = re.search(r"^num_concurrent (\d+)$", read(os.path.join(o, "quality", "settings.txt")), re.M)
        system = (f"the fork's llama-server (llamacpp backend, {conc.group(1) if conc else 'unknown'} concurrent "
                  f"requests, -np {np_.group(1) if np_ else 'unknown'})")
    else:
        system = "serve/server.py on dsv41-engine (strata backend, 2 concurrent requests, the FIFO server)"
    quality = os.path.join(o, "quality")
    who = "stock fork" if fork else "engine"
    out = [f"# Q: the quality of the {who} on the real DeepSeek-V4.1-Flash Q2_K ({stamp})",
           "", f"Session of `tools/dsv41/pod/e07.sh` with E07_PLAN={plan} (Part Q): lm-eval 0.4.13 on the fixed-seed subsets "
           f"against {system} through tools/dsv41/quality/api_proxy.py on the pod, then the needles; every step with its own "
           "timeout.", "", f"Build: Strata {build} (the tarball's MANIFEST). Cost: {cost}. {model}", "",
           "## Server as run", "", "```", server, "```", ""]
    if cfg:
        out += ["Engine config (out/quality-server.json):", "", "```", cfg, "```", ""]
    if fork:
        out += [fork_gpu_peak(o), ""]
    out += ["## HumanEval continuation (before the full run)", "",
            "\n".join(f"- {x}" for x in cont) or f"no result ({cause(o, 'continuation', 'continuation.log')})", "",
            "## Quality", "",
            q_report.results_section(quality, "fork" if fork else "engine") if os.path.isdir(quality)
            else "no out/quality came back", "",
            "## Host", "", host_section(o), "", monitor_peaks(o), ""]
    return out


def qdiag_head(o, stamp, build, cost, model, diag):
    """qdiag: the per-phase diagnostic, the server as run, the continuation check and
    the two GSM8K runs."""
    import analyze
    import quality as Q
    cont = [x for x in read(os.path.join(o, "continuation.log")).splitlines()
            if x.startswith(("ok ", "FAIL", "continuation.py"))]
    server = read(os.path.join(o, "quality-server.cmd")).strip() or "not started"
    cfg = read(os.path.join(o, "quality-server.json")).strip()
    metric, filt = analyze.TASKS["gsm8k_cot_llama"]
    rows = []
    for c in ("c1", "c2"):
        run = os.path.join(o, "quality", c)
        found = Q.find_results(run) if os.path.isdir(run) else {}
        v = (Q.headline(found["gsm8k_cot_llama"]["metrics"], metric, filt) if "gsm8k_cot_llama" in found
             else (None, None))
        st = read(os.path.join(run, "lm_eval_status.txt")).strip().splitlines()
        conc = re.search(r"^num_concurrent (\d+)", read(os.path.join(run, "settings.txt")), re.M)
        rows.append((f"lm-eval-gsm8k-{c}", conc.group(1) if conc else "n/a", f"{metric} ({filt})",
                     f"{v[0]:.4f}" if v[0] is not None else "n/a", st[-1] if st else "not run"))
    out = [f"# The engine's speed diagnostic on the real DeepSeek-V4.1-Flash Q2_K ({stamp})", "",
           "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=qdiag: bench-qdiag (short-static, short-adaptive on one engine load, no "
           "server), then serve/server.py as qengine runs it: the continuation check and the first 20 GSM8K items of "
           "the fixed-seed subsets at NUM_CONCURRENT 1 and 2; the GPU sampled every 1 s and the host every 2 s for "
           "the whole session.", "", f"Build: Strata {build} (the tarball's MANIFEST). Cost: {cost}. {model}", "",
           "## Diagnostic per phase (out/qdiag.md)", "", diag, "",
           "## Server as run", "", "```", server, "```", ""]
    if cfg:
        out += ["Engine config (out/quality-server.json):", "", "```", cfg, "```", ""]
    out += ["## HumanEval continuation (before the GSM8K runs)", "",
            "\n".join(f"- {x}" for x in cont) or f"no result ({cause(o, 'continuation', 'continuation.log')})", "",
            "## GSM8K, the first 20 items (reported, not gating)", "",
            table(["step", "num_concurrent", "metric", "value", "status"], rows), "",
            "## Host", "", host_section(o), "", monitor_peaks(o), ""]
    return out


def qdet_head(o, stamp, build, cost, model):
    """qdet: items 7 and 8 over HTTP, the server as run."""
    import det_sampling_check
    server = read(os.path.join(o, "quality-server.cmd")).strip() or "not started"
    cfg = read(os.path.join(o, "quality-server.json")).strip()
    out = [f"# Determinism and sampling of the engine on the real DeepSeek-V4.1-Flash Q2_K ({stamp})", "",
           "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=qdet: serve/server.py on dsv41-engine as qengine "
           "runs it (its engine with --log-ids), then tools/dsv41/quality/det_sampling_check.py as the server's first "
           "requests after its load, one at a time: item 7 (greedy determinism) and item 8 (seeded sampling, its "
           "speed, a penalty's 400) in the engine's deterministic mode, then items 7 and 8's seeds with the "
           "mode off and the mode's decode cost; every step with its own timeout.", "",
           f"Build: Strata {build} (the tarball's MANIFEST). Cost: {cost}. {model}", "",
           "## Items 7 and 8 (out/det-sampling.json)", "",
           det_sampling_check.section(os.path.join(o, "det-sampling.json")), "",
           "## Server as run", "", "```", server, "```", ""]
    if cfg:
        out += ["Engine config (out/quality-server.json):", "", "```", cfg, "```", ""]
    out += ["## Host", "", host_section(o), "", monitor_peaks(o), ""]
    return out


def cbase_head(o, stamp, build, cost, model):
    """cbase: N concurrent chat users against today's FIFO server, the table by N."""
    import chat_load
    server = read(os.path.join(o, "quality-server.cmd")).strip() or "not started"
    cfg = read(os.path.join(o, "quality-server.json")).strip()
    found = sorted((f for f in os.listdir(os.path.join(o, "cbase")) if re.fullmatch(r"load-c\d+\.json", f))
                   if os.path.isdir(os.path.join(o, "cbase")) else [], key=lambda f: int(f[6:-5]))
    paths = [os.path.join(o, "cbase", f) for f in found]
    ran = {f[6:-5] for f in found}
    loads = sorted((x for x in steps_of(o)[0] if re.fullmatch(r"load-c\d+", x)), key=lambda x: int(x[6:]))
    out = [f"# N concurrent chat users against today's FIFO server, the real DeepSeek-V4.1-Flash Q2_K ({stamp})", "",
           "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=cbase (item 9): serve/server.py on dsv41-engine as qengine runs "
           "it (the deterministic mode on, one request at a time), then tools/dsv41/quality/chat_load.py per N: N users, "
           "each its own 16K-token conversation, started together, 2 streamed turns of 256-token replies each; "
           "measurement only (no engine or scheduling change); every step with its own timeout.", "",
           f"Build: Strata {build} (the tarball's MANIFEST). Cost: {cost}. {model}", "",
           "## Concurrent users by N (out/cbase/load-c<N>.json)", "", chat_load.section(paths)]
    missing = [f"- N = {x[6:]}: no result ({cause(o, x, f'out/cbase/{x}.json')})" for x in loads if x[6:] not in ran]
    if not loads:
        stops = steps_of(o)[1]
        missing.append(f"- no load step ran: {stops[-1] if stops else 'out/steps.log has no load-c<N> line'}")
    if missing:
        out += [""] + missing
    out += ["", "## Server as run", "", "```", server, "```", ""]
    if cfg:
        out += ["Engine config (out/quality-server.json):", "", "```", cfg, "```", ""]
    out += ["## Host", "", host_section(o), "", monitor_peaks(o), ""]
    return out


CK_REQ_RE = re.compile(r"^dsv41-engine serve: request (\d+): (\d+) prompt tokens, (\S+) resume of (\d+), .*?first token at "
                       r"([0-9.]+) ms;", re.M)
CK_LINE_RE = re.compile(r"^dsv41-engine serve: request (\d+): checkpoints: (.*)$", re.M)
CK_IDLE_RE = re.compile(r"^dsv41-engine serve: request (\d+): decode hot set .*; idle reset after the last request (\d+) "
                        r"swaps \(([0-9.]+) MiB\) in ([0-9.]+) ms, waited ([0-9.]+) ms$", re.M)
CK_COST_RE = re.compile(r"^conversation checkpoints: .*$", re.M)
# the load line's numbers: the slots, each slot's MiB and positions, and the carry snapshot on the device
CK_SLOTS_RE = re.compile(r"^conversation checkpoints: (\d+) slots of ([0-9.]+) MiB pinned host RAM each \(a checkpoint of "
                         r"(\d+) positions\), ([0-9.]+) MiB in all.*?carry snapshot \(([0-9.]+) KiB\)", re.M)
CK_LIVE_KV_RE = re.compile(r"^device memory: .*?KV cache ([0-9.]+) MiB .*? at ctx (\d+)", re.M)
# the path the engine took, the homes' RAM, a request's layered reset and its waits
CK_HOMES_RE = re.compile(r"^experts: hot set homes: .*$", re.M)
# the homes' RAM line: "fixed host RAM beside ..." in current engines, "conversation cost beside ..." in the engines
# before it (the 2026-10-14 ccache sessions' logs), which divided the shared homes per checkpoint
CK_HOMES_COST_RE = re.compile(r"^(?:fixed host RAM|conversation cost) beside the checkpoints: the hot set's host homes, "
                              r"([0-9.]+) MiB pinned host RAM", re.M)
CK_LAYERED_RE = re.compile(r"^dsv41-engine serve: request (\d+): idle reset layer by layer \(host homes\): (\d+) swaps copied "
                           r"up only \(([0-9.]+) MiB, none down\), ([0-9.]+) ms on the device(?: \(not seen done yet\))?, "
                           r"queued in ([0-9.]+) ms; this request waited ([0-9.]+) ms on the host and ([0-9.]+) ms on the "
                           r"device over (\d+) layers \(at most ([0-9.]+) ms, layer (-?\d+)\)$", re.M)
# partial host homes, the rounds of the layers not homed in full and the copies down left
CK_ROUNDS_RE = re.compile(r"^dsv41-engine serve: request (\d+): idle reset in rounds \(partial host homes, (\d+) layers\): "
                          r"(\d+) swaps \(([0-9.]+) MiB up, ([0-9.]+) MiB down: (\d+) victims copied down\) in ([0-9.]+) ms "
                          r"on the host, before this request was read$", re.M)
CK_DOWNS_RE = re.compile(r"^dsv41-engine serve: request (\d+): copies down left \(host homes held (\d+) of (\d+) slab "
                         r"experts\): (\d+) of this request's (\d+) swaps copied their victim down \(([0-9.]+) MiB\), "
                         r"(\d+) of the (\d+) swaps of the idle reset before it \(([0-9.]+) MiB\)$", re.M)
CK_RESTORE_RE = re.compile(r"restored \d+ \(\d+ tokens, (?:in place on the device|([0-9.]+) MiB host to device), ([0-9.]+) ms\)")
CK_SAVE_RE = re.compile(r"saved to \d+ \((\d+) tokens, ([0-9.]+) MiB device to host, ([0-9.]+) ms\)")


def checkpoint_section(o):
    """The cost per conversation, what each further conversation adds (its
    checkpoint slot in pinned host RAM, sized by the load line, and what a 16K turn fills of it by the saves; on the
    device the live KV and the carry snapshot), then the host's fixed costs that do not grow with conversations (the
    hot set's host homes, the warm tier, the engram tables, working memory, the margin; out/gate/gate.json) and the
    total host RAM the gate placed beside what homes for every slab expert would need; then every request's
    checkpoints (restored, saved, evicted: MiB and the copies' ms), the idle reset it waited for and its first token,
    from out/quality-engine.log."""
    log = read(os.path.join(o, "quality-engine.log"))
    cost = CK_COST_RE.search(log)
    lines = {int(n): x for n, x in CK_LINE_RE.findall(log)}
    idle = {int(n): (int(sw), float(ms), float(w)) for n, sw, _, ms, w in CK_IDLE_RE.findall(log)}
    rows, restores, saves, mibs, turns16k = [], [], [], [], []
    for n, pt, kind, reuse, first in CK_REQ_RE.findall(log):
        n = int(n)
        text = lines.get(n, "no checkpoints line")
        r = CK_RESTORE_RE.search(text)
        if r and r.group(1):
            restores.append(float(r.group(2)))
            mibs.append(float(r.group(1)))
        v = CK_SAVE_RE.search(text)
        if v:
            saves.append(float(v.group(3)))
            # a 16K turn's checkpoint: the saves of the prompts of 16384 tokens or more (C01's conversations)
            if int(v.group(1)) >= 16384:
                turns16k.append(float(v.group(2)))
        sw, ms, w = idle.get(n, (None, None, None))
        rows.append([n, pt, kind, reuse, text, "n/a" if ms is None else f"{sw} swaps in {ms:.0f}", fmt(w), first])
    if not rows:
        return "no request in out/quality-engine.log"

    def span(xs, unit):
        xs = sorted(xs)
        return f"{len(xs)}, {xs[0]:.2f} to {xs[-1]:.2f} {unit} (median {xs[len(xs) // 2]:.2f})" if xs else "none"
    slots = CK_SLOTS_RE.search(log)
    live = CK_LIVE_KV_RE.search(log)
    out = [f"At load: `{cost.group(0) if cost else 'no cost line (an engine before C02?)'}`", "",
           "Per conversation (what each further conversation adds):", ""]
    if slots:
        n_slots, slot_mib, ctx, all_mib, carry = slots.groups()
        t = sorted(turns16k)
        filled = (f"; a 16K turn's checkpoint fills {t[0]:.1f} to {t[-1]:.1f} MiB of it (median {t[len(t) // 2]:.1f}, the "
                  f"{len(t)} saves of prompts of 16384 tokens or more)" if t else "; no save of a 16K prompt in the log")
        out += [f"- host: its checkpoint slot, {slot_mib} MiB of pinned host RAM, sized for a full {ctx}-position "
                f"conversation and pinned at load ({n_slots} slots, {all_mib} MiB in all){filled}",
                f"- device: the live KV ({live.group(1) + ' MiB at ctx ' + live.group(2) if live else 'its size not in the log'}"
                f") and the in-place carry snapshot ({carry} KiB); C02 decodes one conversation at a time, so the live "
                "KV is held once and a resumed conversation's checkpoint is restored into it"]
    else:
        out.append("- no checkpoints line with its numbers in out/quality-engine.log")
    out += [f"- restores from the host: {span(restores, 'ms')}; their MiB {span(mibs, 'MiB')}",
            f"- saves at a prompt's end: {span(saves, 'ms')}", ""]
    g = load_json(os.path.join(o, "gate", "gate.json")) or {}
    if g.get("warm_gib") is not None:
        homes_on, held = bool(g.get("host_homes")), g.get("host_homes_held", 0)
        partial = homes_on and g.get("need_with_partial_homes_gib") is not None
        homes_mib = CK_HOMES_COST_RE.search(log)
        fixed = []
        if homes_on:
            fixed.append([f"the hot set's host homes ({held} of the slab's {g.get('host_homes_of')} experts, --hot-homes "
                          f"{g.get('hot_homes')}{'; the engine: ' + homes_mib.group(1) + ' MiB' if homes_mib else ''})",
                          f"{g.get('host_homes_held_bytes', 0) / 2**30:.2f}"])
        if partial:
            fixed.append(["the spare homes of the layers on the two-way path", f"{g.get('partial_spare_bytes', 0) / 2**30:.2f}"])
        elif not homes_on:
            fixed.append(["the spare homes (the two-way path, no host homes)", f"{g.get('spare_gib', 0):.2f}"])
        fixed += [[f"the warm tier ({g.get('warm_experts')} experts, pinned)", f"{g['warm_gib']:.2f}"],
                  ["the engram tables (pinned)", f"{g.get('engram_gib', 0):.2f}"],
                  ["working memory", f"{g.get('working_gib', 0):.2f}"],
                  ["the margin (decision e07 gate)", f"{g.get('ram_margin_gib', 0):.2f}"]]
        placed = (g.get("need_with_partial_homes_gib") if partial else g.get("need_with_homes_gib") if homes_on
                  else g.get("need_gib"))
        margin, limit = g.get("ram_margin_gib"), g.get("limit_gib")
        # the shortfall as the gate computed it (unrounded) when it named one
        short = re.search(r"([0-9.]+) GiB short", g.get("host_homes_reason") or "")
        out += ["The host's fixed costs, which do not grow with conversations (out/gate/gate.json):", "",
                table(["what", "GiB"], fixed), "",
                f"Host RAM the gate placed: {placed} GiB (the fixed costs above without the margin, plus the "
                f"{g.get('checkpoints')} checkpoint slots' {g.get('checkpoints_gib')} GiB) + {margin} margin against "
                f"{limit} GiB, {g.get('limit_source')}; homes for every slab expert ({g.get('host_homes_of')}, "
                f"{g.get('host_homes_gib')} GiB) would need {g.get('need_with_homes_gib')} GiB + {margin} margin"
                + (f", {short.group(1)} GiB more than the limit (the gate's)" if short else ""), ""]
    else:
        out += ["The host's fixed costs: no out/gate/gate.json", ""]
    out.append(table(["request", "prompt tokens", "resume", "reused", "checkpoints", "idle reset before it (ms)",
                      "waited for it (ms)", "first token at (ms)"], rows))
    return "\n".join(out)


def homes_section(o):
    """The path the swaps and the deterministic reset took (the engine's load line, the gate's
    reason from out/placement.txt and its numbers from out/gate/gate.json), the homes' RAM, and per request after a
    layered reset: its swaps and MiB, the full reset on the device, its host queuing, and the waits the request saw.
    With partial host homes: the homes held of the slab's experts (the gate's), the rounds of the layers not homed in full, and per
    request the copies down left (its swaps' and the reset's before it)."""
    log = read(os.path.join(o, "quality-engine.log"))
    path = CK_HOMES_RE.search(log)
    why = next((x for x in read(os.path.join(o, "placement.txt")).splitlines() if x.startswith("host homes")), None)
    g = load_json(os.path.join(o, "gate", "gate.json")) or {}
    out = [f"Path: `{path.group(0) if path else 'no hot set homes line (an older engine?)'}`",
           "", f"Gate: {why or 'no host homes line in out/placement.txt'}"]
    if g.get("host_homes_gib") is not None:
        out.append(f"(the homes {g['host_homes_gib']} GiB; the host RAM with them {g.get('need_with_homes_gib')} GiB + "
                   f"{g.get('ram_margin_gib')} margin against {g.get('limit_gib')} GiB, {g.get('limit_source')})")
    if g.get("host_homes_of"):
        cut = g.get("host_homes_cut")
        out += ["", f"Homes held: {g.get('host_homes_held', 0)} of the slab's {g['host_homes_of']} experts "
                f"({100 * g.get('host_homes_fraction', 0):.1f}%), {g.get('host_homes_held_bytes', 0) / 2**30:.2f} GiB, "
                f"--hot-homes {g.get('hot_homes')}" + (
                    f"; partial: layer {cut[0]} cut at its last {cut[1]} of {cut[2]}, the "
                    f"layers after it on the two-way path, the RAM with them {g.get('need_with_partial_homes_gib')} GiB + "
                    f"{g.get('ram_margin_gib')} margin" if cut else
                    f"; partial, the RAM with them {g.get('need_with_partial_homes_gib')} "
                    f"GiB + {g.get('ram_margin_gib')} margin" if g.get("need_with_partial_homes_gib") else "")]
    rows = []
    for n, sw, mib, dev, queued, host_w, dev_w, layers, wmax, at in CK_LAYERED_RE.findall(log):
        rows.append([n, sw, mib, dev, queued, host_w, dev_w, layers, f"{wmax} (layer {at})"])
    if rows:
        devs = sorted(float(r[3]) for r in rows)
        waits = sorted(float(r[6]) for r in rows)
        out += ["", f"The layered reset: {len(rows)} requests after one, the full reset {devs[0]:.0f} to {devs[-1]:.0f} ms "
                f"on the device (median {devs[len(devs) // 2]:.0f}), the next request's layers waiting {waits[0]:.0f} to "
                f"{waits[-1]:.0f} ms in all (median {waits[len(waits) // 2]:.0f})", "",
                table(["request", "reset swaps", "MiB up", "full reset on the device (ms)", "queued on the host (ms)",
                       "waited on the host (ms)", "waited on the device (ms)", "layers that waited",
                       "longest layer wait (ms)"], rows)]
    else:
        out += ["", "No layered reset in out/quality-engine.log (the two-way path: the reset in rounds, each request's "
                "wait in the table above)"]
    # the layers not homed in full, reset in rounds before the next request was read
    rounds = CK_ROUNDS_RE.findall(log)
    if rounds:
        ms = sorted(float(x[6]) for x in rounds)
        out += ["", f"The rounds (partial host homes, {rounds[0][1]} layers not homed in full): {len(rounds)} requests "
                f"after one, {sum(int(x[2]) for x in rounds)} swaps, {sum(int(x[5]) for x in rounds)} victims copied down "
                f"({sum(float(x[4]) for x in rounds):.1f} MiB), {ms[0]:.0f} to {ms[-1]:.0f} ms on the host each (median "
                f"{ms[len(ms) // 2]:.0f})"]
    downs = CK_DOWNS_RE.findall(log)
    if downs:
        d_sw, d_all = sum(int(x[3]) for x in downs), sum(int(x[4]) for x in downs)
        r_sw, r_all = sum(int(x[6]) for x in downs), sum(int(x[7]) for x in downs)
        out += ["", f"Copies down left (host homes held {downs[0][1]} of {downs[0][2]}): {d_sw} of {d_all} decode swaps "
                f"({100 * d_sw / d_all if d_all else 0:.1f}%) and {r_sw} of {r_all} reset swaps "
                f"({100 * r_sw / r_all if r_all else 0:.1f}%) over {len(downs)} requests", "",
                table(["request", "swaps", "copied down", "MiB down", "reset swaps before it", "reset copied down",
                       "reset MiB down"], [[x[0], x[4], x[3], x[5], x[7], x[6], x[8]] for x in downs])]
    return "\n".join(out)


def ccache_head(o, stamp, build, cost, model, sheet_root):
    """ccache: item 10's alternating users, the C01 runs again against C01's,
    the cost per conversation, items 7 and 8 once."""
    import chat_load
    import det_sampling_check
    server = read(os.path.join(o, "quality-server.cmd")).strip() or "not started"
    cfg = read(os.path.join(o, "quality-server.json")).strip()
    d = os.path.join(o, "ccache")
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    alt = [os.path.join(d, f) for f in files if re.fullmatch(r"load-alt\d+\.json", f)]
    think = [os.path.join(d, f) for f in files if re.fullmatch(r"load-alt\d+-think\d+\.json", f)]
    byn = sorted((f for f in files if re.fullmatch(r"load-c\d+\.json", f)), key=lambda f: int(f[6:-5]))
    # the session's real cost (cost.txt: each pod's seconds from create to delete at its
    # price, an upper bound) against the cap e07.sh ran it with (session-cap, MAX_SESSION_USD: ccache's 2.30)
    usd = [float(u) for u in re.findall(r"^pod \S+: \d+ s at [0-9.]+ USD/h = ([0-9.]+) USD", cost, re.M)]
    cap = read(os.path.join(os.path.dirname(o), "session-cap")).strip()
    spent = (f"Session: {sum(usd):.3f} USD over {len(usd)} pod{'s' if len(usd) != 1 else ''}" if usd else
             "Session: no pod's cost recorded (cost.txt)")
    spent += (f", the session's cap {cap} USD" + (f": {'within' if sum(usd) <= float(cap) else 'OVER'} it" if usd else "")
              if cap else ", no session-cap recorded") + "."
    out = [f"# Several cached conversations on the real DeepSeek-V4.1-Flash Q2_K ({stamp})", "",
           "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=ccache (item 10): serve/server.py "
           "on dsv41-engine as qengine runs it (the deterministic mode on, its reset after each request; 4 conversation "
           "checkpoints, the engine's default; every reply's ids in its log), then tools/dsv41/quality/chat_load.py: the "
           "users taking turns back to back (item 10), again with a think time between turns, the C01 runs again by "
           "N, then det_sampling_check.py's items 7 and 8; every step with its own timeout.", "",
           f"Build: Strata {build} (the tarball's MANIFEST). Cost: {cost}. {spent} {model}", ""]
    out += ["## Item 10: the users taking turns (out/ccache/load-alt<N>.json, load-alt<N>-think<S>.json)", ""]
    if alt or think:
        runs = [(r, p in think) for p in alt + think for r in (load_json(p) or {}).get("runs", [])]
        rows = []
        for r, info in runs:
            s = r["summary"]
            rows.append([r["label"], fmt(float(r.get("think_s", 0)), 1)] +
                        [fmt(s.get(k), 3) for k in ("turn2_ttft_p50_s", "turn2_ttft_p95_s", "turn2_ttft_max_s",
                                                    "turn2_unqueued_ttft_max_s", "turn2_reset_wait_p50_s",
                                                    "turn2_reset_wait_max_s")] +
                        ["informational" if info else "item 10's verdict"])
        out += [table(["run", "think s", "turn 2+ TTFT p50 s", "p95 s", "max s", "max not queued s",
                       "turn 2+ reset wait p50 s", "reset wait max s", "role"], rows), "",
                "The strict run: each request sent when the one before it ended; its first "
                "token includes its wait for the deterministic mode's idle reset after the previous reply (the reset "
                "wait, from the engine's log), which counts against the 2 s (it is not a queue behind another "
                "prefill). The think-time run: the same users and turns, each request sent the think "
                "time after the one before it ended, as in a chat where people read the reply; informational only.",
                "", chat_load.section(alt + think, logs="ccache")]
        for r, info in runs:
            s = r["summary"]
            later = [x for x in r["rows"] if x.get("turn", 1) > 1 and "error" not in x]
            if info:
                out += ["", f"Think-time run ({r['label']}, informational): {s.get('turn2_resumed')} of "
                        f"{s.get('turn2_requests')} turns 2+ resumed, their first token at most "
                        f"{s.get('turn2_unqueued_ttft_max_s')} s when not queued, the reset wait at most "
                        f"{s.get('turn2_reset_wait_max_s')} s"]
                continue
            ok = (s.get("errors") == 0 and s.get("turn2_requests") and s["turn2_resumed"] == s["turn2_requests"] and
                  s.get("turn2_unqueued_ttft_max_s") is not None and s["turn2_unqueued_ttft_max_s"] <= 2.0)
            out += ["", f"Item 10 ({r['label']}): {'PASS' if ok else 'FAIL'}: {s.get('turn2_resumed')} of "
                    f"{s.get('turn2_requests')} turns 2+ resumed a cached prefix, their first token at most "
                    f"{s.get('turn2_unqueued_ttft_max_s')} s when not queued (p50 {s.get('turn2_ttft_p50_s')}, p95 "
                    f"{s.get('turn2_ttft_p95_s')}; goal <= 2 s), {s.get('queued')} requests queued, the reset wait at "
                    f"most {s.get('turn2_reset_wait_max_s')} s"]
            # the reset wait the reason when every slow turn would be in time without it
            slow = [x for x in later if not x.get("queued") and x["ttft_s"] > 2.0]
            if slow and all(x.get("reset_wait_s") is not None and x["ttft_s"] - x["reset_wait_s"] <= 2.0 for x in slow):
                # the follow-up depends on the path the reset took
                layered = bool(CK_LAYERED_RE.search(read(os.path.join(o, "quality-engine.log"))))
                out += ["", f"Every turn 2+ over 2 s ({len(slow)}) is within 2 s without its reset wait: the deterministic "
                        "reset is the reason; " + (
                            "it ran layer by layer with host homes, so the follow-up is the fallback (a "
                            "PCIe 5 host in the qualification), sized from the reset wait column" if layered else
                            "it ran in rounds on the two-way path (no host homes: see the gate's reason below), so the "
                            "follow-up is a host whose RAM holds the homes")]
    else:
        step = next((x for x in steps_of(o)[0] if x.startswith("load-alt")), "load-alt4")
        out.append(f"no result ({cause(o, step, 'out/ccache/load-alt<N>.json')})")
    base = os.environ.get("CCACHE_BASELINE") or os.path.join(sheet_root, "2026-10-13-cbase-2320", "logs", "cbase")

    def resumed(path):
        """The turns 2+ of a load file that resumed a cached prefix (C01's files have no turn2_ fields: their rows)."""
        j = load_json(path) or {}
        rr = [x for x in (j.get("runs") or [{}])[0].get("rows", []) if x.get("turn", 1) > 1 and "error" not in x]
        return f"{sum(1 for x in rr if x.get('cached_tokens'))} of {len(rr)}" if rr else "n/a"
    rows = []
    for f in byn:
        after = ((load_json(os.path.join(d, f)) or {}).get("runs") or [{}])[0].get("summary", {})
        before = ((load_json(os.path.join(base, f)) or {}).get("runs") or [{}])[0].get("summary", {})
        rows.append([f[6:-5], fmt(before.get("ttft_p50_s"), 3), fmt(after.get("ttft_p50_s"), 3),
                     fmt(before.get("ttft_p95_s"), 3), fmt(after.get("ttft_p95_s"), 3), resumed(os.path.join(base, f)),
                     resumed(os.path.join(d, f)), fmt(before.get("queue_p50_s"), 3), fmt(after.get("queue_p50_s"), 3)])
    out += ["", "## Before and after: the C01 runs again (out/ccache/load-c<N>.json)", "",
            f"Before: the C01 sheet's runs ({base}); after: this session's, the same conversations, users and turns.", "",
            table(["N", "TTFT p50 s before", "after", "TTFT p95 s before", "after", "turns 2+ resumed before", "after",
                   "queue p50 s before", "after"], rows) if rows else "no load-c<N> result", "",
            chat_load.section([os.path.join(d, f) for f in byn], logs="ccache") if byn else ""]
    out += ["", "## Cost per conversation (out/quality-engine.log, out/gate/gate.json)", "", checkpoint_section(o), "",
            "## The hot set's homes and the reset", "", homes_section(o), "",
            "## Items 7 and 8 once (out/det-sampling.json)", "",
            det_sampling_check.section(os.path.join(o, "det-sampling.json")), "",
            "## Server as run", "", "```", server, "```", ""]
    if cfg:
        out += ["Engine config (out/quality-server.json):", "", "```", cfg, "```", ""]
    out += ["## Host", "", host_section(o), "", monitor_peaks(o), ""]
    return out


BATCH_RE = re.compile(r"^dsv41-engine serve: batch: .*$", re.M)
BATCH_SLOT_RE = re.compile(r"^dsv41-engine serve: request (\d+): batch slot (\d+): (\d+) generated \((\w+)\) in ([0-9.]+) ms, "
                           r"(\d+) steps, (\d+) of them beside other slots \(([0-9.]+) rows a step\), steps stalled "
                           r"([0-9.]+) ms by other requests' prompts; (.*)$", re.M)
# a prompt read beside decoding slots, in chunks with a step after each but the last (dsv41-engine.cpp's line)
BATCH_READ_RE = re.compile(r"^dsv41-engine serve: request (\d+): read beside (\d+) decoding batch slots, (\d+) tokens in "
                           r"(\d+) chunks of (.+?) with (\d+) batch steps between them(?: \(decision-c04-long-prefill-"
                           r"robustness 1\))?, a chunk read in ([0-9.]+) ms on average and ([0-9.]+) at most, ([0-9.]+) ms "
                           r"from its start to its (?:first|last) token, the slots held at most ([0-9.]+) ms at once; .*$",
                           re.M)


def session_spent(o, cost):
    """The session's real cost against its cap (cost.txt's pods, OUT's session-cap)."""
    usd = [float(u) for u in re.findall(r"^pod \S+: \d+ s at [0-9.]+ USD/h = ([0-9.]+) USD", cost, re.M)]
    cap = read(os.path.join(os.path.dirname(o), "session-cap")).strip()
    spent = (f"Session: {sum(usd):.3f} USD over {len(usd)} pod{'s' if len(usd) != 1 else ''}" if usd else
             "Session: no pod's cost recorded (cost.txt)")
    return spent + ((f", the session's cap {cap} USD" + (f": {'within' if sum(usd) <= float(cap) else 'OVER'} it"
                                                         if usd else "")
                     if cap else ", no session-cap recorded") + ".")


# dsv41-engine's per-batch breakdown of its steps, one line per part of the batch
STEP_PARTS_RE = re.compile(
    r"^dsv41-engine serve: batch steps, (.+?)(?: \(decision-c03b-batched-speed 2\))?: (\d+) steps of ([0-9.]+) rows, ([0-9.]+) ms "
    r"a step from one step's start to the next's \(the prompts read between them apart\): inside the step ([0-9.]+) "
    r"\(graph launch ([0-9.]+), doorbells served ([0-9.]+) of which doorbell waits ([0-9.]+), CPU jobs ([0-9.]+) and their "
    r"plan ([0-9.]+), tail ([0-9.]+), boundary ([0-9.]+) of which swap waits ([0-9.]+) at (\d+) steps; the device's graph "
    r"([0-9.]+); (\d+) graphs captured again\), between steps ([0-9.]+) \(the picks ([0-9.]+), the BT lines and their flush "
    r"([0-9.]+), admissions ([0-9.]+), the loop and its stdin poll ([0-9.]+)\); per step ([0-9.]+) routed pairs to "
    r"([0-9.]+) distinct experts, ([0-9.]+) hits \(hit rate ([0-9.]+)\) and ([0-9.]+) misses in ([0-9.]+) CPU jobs reading "
    r"([0-9.]+) MiB \(([0-9.]+) GB/s\), ([0-9.]+) swaps \(([0-9.]+) MiB\) published, ([0-9.]+) deferred$", re.M)
# C04 audit 2 and 5: a request that ran alone past a slot, and a chunk read beside the slots past 4.5 s
RAN_ALONE_RE = re.compile(r"^dsv41-engine serve: request (\d+): ran alone on the live sequence, not in batch slot (\d+): "
                          r"(.*) \(C04 audit 2\); (\d+) generated \((\w+)\), holding (\d+) decoding batch slots for "
                          r"([0-9.]+) ms(?:; not reproducible \(--cpu-numerics fork: its bits follow the "
                          r"placement\))?$", re.M)
CHUNK_WARN_RE = re.compile(r"^dsv41-engine serve: request (\d+): warning: a chunk read beside (\d+) decoding batch slots "
                           r"took ([0-9.]+) ms", re.M)


def step_parts(log):
    """C03b: every batch's step breakdown in the engine's log, one row per part (all its
    steps, those within 15 s of a prompt read beside the slots, the others): where a step's wall time went inside the
    step and between two, and per step the rows' routed pairs, distinct experts, hits, misses, CPU jobs and swaps."""
    rows = STEP_PARTS_RE.findall(log)
    if not rows:
        return "no breakdown line (an engine before C03b, or no request decoded in a slot)"
    batch = 0
    out = []
    for r in rows:
        batch += r[0] == "all"
        out.append([batch, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], f"{r[12]} ({r[13]})",
                    r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26],
                    f"{r[27]} ({r[28]} GB/s)", f"{r[29]} ({r[30]} MiB)", r[31]])
    return table(["batch", "steps", "n", "rows", "ms a step", "inside", "launch", "doorbells served", "doorbell waits",
                  "CPU jobs ms", "plan", "tail", "boundary", "swap waits ms (steps)", "device graph", "captures",
                  "between", "picks", "BT out", "admissions", "loop", "pairs", "distinct experts", "hits", "hit rate",
                  "misses", "CPU jobs", "CPU MiB", "swaps (MiB)", "deferred"], out)


def ran_alone(logs):
    """C04 audit 2: the requests that ran alone past a batch slot, holding the decoding slots, per engine log."""
    rows = []
    for name, log in logs:
        for r in RAN_ALONE_RE.findall(log):
            rows.append([name, r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
    head = (f"{len(rows)} request{'' if len(rows) == 1 else 's'} ran alone past a batch slot (C04 audit 2: a prompt, or "
            "a max_tokens, past the slot's "
            "positions; the reply says \"strata_ran_alone\"); a request without max_tokens is given its slot's room "
            "instead")
    if not rows:
        return head + "."
    return head + ":\n\n" + table(["log", "request", "slot refused", "why", "generated", "finish", "slots held",
                                     "ms held"], rows)


def interleaved_reads(log):
    """C04: the prompts read beside decoding slots, from the engine's log, as a table: each chunk's read time
    (the session measures it) and the slots' longest hold, the gap the engine gave the
    decoding users."""
    reads = BATCH_READ_RE.findall(log)
    if not reads:
        return "no prompt was read beside decoding slots"
    return table(["request", "beside slots", "tokens read", "chunks", "chunk", "steps between", "chunk read ms (mean)",
                  "chunk read ms (longest)", "ms to its token", "longest hold ms (the slots' gap)"],
                 [list(x) for x in reads])


def homes_against_checkpoints(o):
    """Per gate of the session (out/gate, gate-off, gate-long and
    gate-spare), the slab's host homes held against the conversation
    checkpoints the host also pins: the checkpoints and their GiB, the homes held of the slab's experts and their GiB,
    the host RAM needed with them against the limit."""
    rows = []
    for sub in ("gate", "gate-off", "gate-long", "gate-spare"):
        g = load_json(os.path.join(o, sub, "gate.json")) or {}
        if not g:
            continue
        held = g.get("host_homes_held", g.get("host_homes_of") if g.get("host_homes") else 0)
        rows.append([f"out/{sub}", g.get("checkpoints"), g.get("checkpoints_gib"), g.get("batch", 0),
                     f"{held} of {g.get('host_homes_of', 'n/a')}" + (f" ({100 * g['host_homes_fraction']:.1f}%)"
                                                                       if g.get("host_homes_fraction") is not None else ""),
                     round(g["host_homes_held_bytes"] / 2**30, 2) if g.get("host_homes_held_bytes") else g.get("host_homes_gib"),
                     g.get("hot_homes"), g.get("need_with_partial_homes_gib") or g.get("need_with_homes_gib"),
                     g.get("limit_gib")])
    if not rows:
        return "no gate result"
    return table(["gate", "checkpoints", "checkpoints GiB", "batch slots", "host homes held of the slab's experts",
                  "homes GiB", "--hot-homes", "host RAM needed GiB", "limit GiB"], rows)


def clong_sections(o, prefix=""):
    """clong: item 12's gaps and 64K TTFT, item 13's checks, the interleaved
    reads, the long gate (out/gate, or the combined plan's out/gate-long) and the server as run (prefix: its logs')."""
    import chat_load
    import robust_check
    d = os.path.join(o, "clong")
    j = load_json(os.path.join(d, "load-long.json"))
    out = ["## Item 12: no user frozen by a long prompt (out/clong/load-long.json)", ""]
    if j:
        sm = j["summary"]
        out += [chat_load.long_tables(j), "",
                f"- item 12: {'PASS' if sm['gap_pass'] else 'FAIL'}: the largest gap between two tokens of any of the "
                f"{sm['users']} users {fmt(sm['max_gap_s'], 3)} s ({fmt(sm['read_max_gap_s'], 3)} s during the "
                f"{sm['long_prompt_tokens']}-token read; goal <= {j['gap_max_s']} s), every user decoding through the "
                f"read: {sm['covered']}; the long TTFT {fmt(sm['long_ttft_s'], 3)} s beside them against "
                f"{fmt(sm['long_ttft_solo_s'], 3)} s alone ({sm['long_ttft_ratio']} x, reported)", "",
                "Gaps: between two content chunks of a user's stream at the client (chat_load.py long); the read: from "
                "the long request's send to its first token. The long request alone reads its prompt in the engine's "
                "--chunk; beside decoding slots in chunks of the interleave chunk (the engine's --chunk by default) "
                "with a batch step after each but the last (dsv41-engine --interleave-chunk). Each chunk's read time and the slots' longest hold, the gap "
                "at the engine, are in the table of prompts read beside decoding slots below; if the gap passes 5 s "
                "the follow-up is a cheaper interleaved read sized from them.", ""]
    else:
        out += ["no result (" + cause(o, "load-long", "out/clong/load-long.json") + ")", ""]
    log = read(os.path.join(o, prefix + "quality-engine.log"))
    warns = CHUNK_WARN_RE.findall(log)
    out += ["## Item 13: served and robust (out/clong/robust.json)", "",
            robust_check.section(os.path.join(d, "robust.json")) if os.path.exists(os.path.join(d, "robust.json"))
            else "no result (" + cause(o, "robust", "out/clong/robust.json") + ")", "",
            "Not checked on the real engine here: the per-request deadline (request_deadline_s, "
            "1800 s) and the engine watchdog (engine_silence_s, 300 s) were checked on the fake engine only "
            "(serve/test_robust.py and c04-test.sh: a dead and a hung engine fail every pending request with a 503 that "
            "has a code and Retry-After, a request reaching its deadline ends with its text and finish \"length\"); a "
            "session cannot wait for them.", "",
            f"## Prompts read beside decoding slots (out/{prefix}quality-engine.log)", "",
            interleaved_reads(log), "",
            chunk_time_text(o, log, prefix) + "; past 4.5 s the engine warns" + (f" ({len(warns)} warnings here: requests "
                                        f"{', '.join(sorted({w[0] for w in warns}, key=int))})" if warns else
                                        " (no warning in this log)") + ".", "",
            "### Ran alone", "", ran_alone([(f"out/{prefix}quality-engine.log", log)]), ""]
    sub = "gate-long" if os.path.isdir(os.path.join(o, "gate-long")) else "gate"
    g = load_json(os.path.join(o, sub, "gate.json")) or {}
    if g:
        out += [f"## Placement: {sub} (the long prompt's slots, out/{sub}/gate.json)", "",
                table(["", ""], [(k, v) for k, v in g.items()]), ""]
    out += ["## Server as run, the long prompt's slots", "", "```",
            read(os.path.join(o, prefix + "quality-server.cmd")).strip() or "not started", "```", ""]
    return out


def clong_head(o, stamp, build, cost, model):
    """clong: items 12 and 13 on the real engine."""
    return [f"# A long prompt beside decoding users, and the server robust, on the real DeepSeek-V4.1-Flash Q2_K "
            f"({stamp})", "",
            "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=clong (items 12 and 13): "
            "serve/server.py on dsv41-engine with batch slots that hold the 64K prompt and its reply; "
            "tools/dsv41/quality/chat_load.py long (the 64K prompt alone, then beside decoding users) and "
            "robust_check.py (item 13 over HTTP); every step with its own timeout.", "",
            f"Build: Strata {build} (the tarball's MANIFEST). Cost: {cost}. {session_spent(o, cost)} {model}", ""] + \
        clong_sections(o) + ["## Host homes held against the checkpoints", "", homes_against_checkpoints(o), "",
                             "## Host", "", host_section(o), "", monitor_peaks(o), ""]


# the 8 users' figures before a plan measured steady-c8 (the cbatch session of 2026-10-14): stale, older code
STALE_8_USERS = ("70.42 tok/s in aggregate over the decode window, 11.35 tok/s per user (median), measured on 2026-10-14 "
                 "(the cbatch session, chat_load.py run N = 8, build 90f5fed2, before the long-prompt and batch speed work, slots of "
                 "32768 positions): stale, not this code")
# the single user's baseline of the session before (the cbatch-clong session of 2026-10-15), for the disclosure of
# its change and its machine's
PREVIOUS_BASELINE = {"sheet": "the cbatch-clong session of 2026-10-15", "tok_s": 53.68, "machine": "machine A",
                     "cpu": "AMD EPYC 9335 32-Core Processor", "build": "strata-db3b4f61 (c075c44)"}


def f2_note(steady):
    """A steady run whose replies said more cached tokens than their prompt (the server
    before the fix counted a moved request's own read as cached) -> the note under its table."""
    bad = [(x["cached_tokens"], x["prompt_tokens"]) for r in steady.get("repeats") or [] for x in r.get("rows", [])
           if (x.get("cached_tokens") or 0) > (x.get("prompt_tokens") or 0)]
    if not bad:
        return []
    return [f"Note: {len(bad)} repl{'y' if len(bad) == 1 else 'ies'} above said more cached "
            f"prompt tokens than {'its' if len(bad) == 1 else 'their'} prompt (e.g. {bad[0][0]} of {bad[0][1]}): the "
            "server of this session counted a request's own read as cached once it was moved from the solo path into a "
            "slot; fixed since (the cached tokens are the prefix cached before the request's own read, at most its "
            "prompt).", ""]


def session_machine(o):
    """The session's machine: pod.json's machineId (beside out/) and env.log's CPU model."""
    pod = load_json(os.path.join(os.path.dirname(o), "pod.json")) or {}
    cpu = re.search(r"^Model name:\s*(.+)$", read(os.path.join(o, "env.log")), re.M)
    return pod.get("machineId") or "not recorded (no pod.json)", cpu.group(1).strip() if cpu else "CPU not recorded"


def chunk_times(log):
    """Every prompt read beside decoding slots in an engine log, the users' own 16K prompts
    as much as the long ones (BATCH_READ_RE's groups): (all the reads, those with a full 16K chunk, the long prompts'
    (two chunks of 16384 or more) mean of their chunks' mean read time with the last partial chunk included, the read
    with the longest chunk, the read that held the slots longest), or None without a read."""
    reads = BATCH_READ_RE.findall(log)
    if not reads:
        return None
    full = [x for x in reads if x[4] == "16384" and int(x[2]) >= 16384]
    long = [x for x in full if int(x[2]) >= 2 * 16384]
    mean = sum(float(x[6]) for x in long) / len(long) if long else None
    return (reads, full, long, mean, max(reads, key=lambda x: float(x[7])), max(reads, key=lambda x: float(x[9])))


# the H2D of the two Server machines whose chunk times the note cites (their sheets'
# qualification tables, the 2026-10-15 cbatch-clong sessions)
REFERENCE_H2D = "57.0"


def session_h2d(o):
    """The host's pinned H2D GB/s from the qualification (probe.log beside out/): the session pod's (pod.json's id)
    with its PCIe link from env.log, then every other pod qualified that session, by GPU (its offer's name)."""
    out_dir = os.path.dirname(o)
    pod = (load_json(os.path.join(out_dir, "pod.json")) or {}).get("id")
    pods = probe_pods(out_dir)
    mine = next((p for p in pods if p["pod"] == pod), None)
    if not mine or "h2d_gbs" not in mine["m"]:
        return "the host's H2D not recorded (no qualification measure for the session's pod in logs/probe.log)"
    link = pcie_link(read(os.path.join(o, "env.log")))
    s = (f"the host's H2D {mine['m']['h2d_gbs'][0]} GB/s (pinned, the machine qualification table, pod {pod}"
         + (f", PCIe {link.split(' (')[0]}" if link.startswith("Gen ") else "") + ")")
    others = [f"{p['m']['h2d_gbs'][0]} GB/s on {p['machine']} ({p['offer'].split(',')[0]}, pod {p['pod']})"
              for p in pods if p is not mine and "h2d_gbs" in p["m"]]
    return s + (f", against {', '.join(others)} qualified the same session" if others else "")


def chunk_time_text(o, log, prefix):
    """The chunk read times this session measured over every prompt read beside the
    slots, the read with the longest chunk and the one that held the slots longest, on this session's machine with its
    H2D, then the references (other hosts)."""
    machine, cpu = session_machine(o)
    gpu = session_gpu(o, os.path.dirname(o))
    ct = chunk_times(log)
    src = f"logs/{prefix}quality-engine.log; {gpu}, machine {machine}, {cpu}"
    if ct:
        reads, full, long, mean, top, hold = ct
        plural = lambda n, w: f"{w}{'s' if n != 1 else ''}"
        read_of = lambda x: f"request {x[0]}'s {x[2]}-token read beside {x[1]} {plural(int(x[1]), 'slot')}"
        here = (f"{len(reads)} {plural(len(reads), 'prompt')} read beside decoding slots in this session ({src}), "
                f"{len(full)} of them with a full 16K chunk ({plural(len(full), 'request')} "
                f"{', '.join(x[0] for x in full)}: the users' own 16K prompts and the long ones alike); "
                + (f"the chunks of the long prompts ({plural(len(long), 'request')} {', '.join(x[0] for x in long)}) "
                   f"took {mean / 1e3:.2f} s on average (the last partial chunk included, as in the references); "
                   if long else "no long prompt (two chunks of 16384 or more) was read beside them; ")
                + (f"the longest chunk and the slots' longest hold (the gap at the engine) both "
                   f"{float(top[7]) / 1e3:.2f} s ({top[7]} ms), {read_of(top)}" if top is hold else
                   f"the longest chunk {float(top[7]) / 1e3:.2f} s ({top[7]} ms), {read_of(top)}; the slots' longest "
                   f"hold (the gap at the engine) {float(hold[9]) / 1e3:.2f} s ({hold[9]} ms), {read_of(hold)}"))
    else:
        here = f"no prompt was read beside decoding slots in this session ({src})"
    return (f"A chunk's read time is the host's: {here}; {session_h2d(o)}; for reference, the two "
            f"RTX PRO 6000 Server machines of the 2026-10-15 sessions (H2D {REFERENCE_H2D} "
            "GB/s each) each 2.59 s on average for their long prompts' chunks and 3.29 s at most over every read beside "
            "their slots, the E09 RTX PRO 6000 Workstation Edition read a 16K prompt cold in 4.59 s, E07b's slower "
            "Workstation card in 7.68 s")


# the 4-spare-slot variant left out of this plan (CBATCH_SPARE 0), and why;
# the figures are tools/dsv41/engine/README.md's (hot_policy_sim.py batch, e07.sh --dry-run locally)
VARIANT_NOT_RUN = (
    "- item 11 with 4 spare slab slots per layer: not run in this plan, "
    "\"hot_spare\" stays off (absent: the engine's 2 spare slots per layer). hot_policy_sim.py batch predicts no gain "
    "(the E08 routing trace, A100 2026-10-09, on an RTX PRO 6000 host's gates with 8 slots of 69632: the planner starts "
    "about 1.6 swaps a layer a step, under the 2 slots, and the copy stream keeps up; the slab's 84 fewer experts cost "
    "about 2 CPU jobs a step, 211.8 / 72.1 / 62.9 against 209.4 / 70.3 / 60.8 in steps 0-50 / 50-150 / 150-300), and "
    "with CBATCH_SPARE=4 steady-c4-spare's whole timeout ends at minute 68.0 (74.7 since steady-c8 joined the plan), "
    "past the pod's 67-minute deadline (the dry run's latest end), so the dry run refuses it.")


def spare_variant(o, d, single_off, default):
    """The combined plan's second cbatch server, its engine with more spare
    slab slots per layer (out/gate-spare, out/cbatch/load-short-spare.json and steady-c4-spare.json), judged for item 11
    as the default is and against the same baseline (the single user with batching off on the default config), side by
    side with the default (`default`: its single user with batching on, steady aggregate, lowest repeat, slowest user
    and item 11 verdict, None where missing), then the rule over both. The item's lines and the variant's
    steady-state section; when the session ran no such server, the item's one line saying why."""
    import chat_load
    g = load_json(os.path.join(o, "gate-spare", "gate.json")) or {}
    if not g and "gate-spare" not in steps_of(o)[0]:
        return [VARIANT_NOT_RUN], []
    g0 = load_json(os.path.join(o, "gate", "gate.json")) or {}
    spare = g.get("hot_spare", "n/a")
    on = ((load_json(os.path.join(d, "load-short-spare.json")) or {}).get("runs") or [None])[0]
    steady = load_json(os.path.join(d, "steady-c4-spare.json")) or {}
    sm = steady.get("summary") or {}
    v = {"single": on["summary"].get("decode_tok_s_user_median") if on else None, "agg": sm.get("aggregate_tok_s_mean"),
         "agg_min": sm.get("aggregate_tok_s_min"), "slowest": sm.get("user_tok_s_min"), "ok": None}
    measured = steady.get("ok") and steady.get("users") == 4 and v["agg"] is not None
    single_ok = None
    if v["single"] is not None and single_off:
        reg = (single_off - v["single"]) / single_off
        single_ok = v["single"] >= 30 and reg <= 0.05
        single_text = (f"its single user {fmt(v['single'], 2)} tok/s with batching on against {fmt(single_off, 2)} off "
                       f"on the default config ({100 * reg:+.1f}%; goal >= 30 and within 5%)")
    else:
        single_text = ("its single user: no result ("
                       + cause(o, ["engine-server-spare", "load-short-spare"], "out/cbatch/load-short-spare.json") + ")")
    if measured and single_off and single_ok is not None:
        v["ok"] = v["agg"] >= 2 * single_off and v["slowest"] >= 20 and single_ok
        text = (f"{'PASS' if v['ok'] else 'FAIL'}: 4 users decoding at once (steady-c4-spare: {sm['measured']} repeats "
                f"of a {steady['window_s']:g} s window with exactly 4 rows a step) {fmt(v['agg'], 2)} tok/s in aggregate "
                f"(the mean; repeats {fmt(v['agg_min'], 2)} to {fmt(sm['aggregate_tok_s_max'], 2)}) = "
                f"{v['agg'] / single_off:.2f}x the baseline ({fmt(single_off, 2)} tok/s), the slowest user "
                f"{fmt(v['slowest'], 2)} tok/s (goal: >= 2x and each >= 20); {single_text}")
    else:
        text = "no verdict: " + (single_text if measured else "no 4-user steady result (" + cause(
            o, ["engine-server-spare", "steady-c4-spare"], "out/cbatch/steady-c4-spare.json") + ")")
    slab = (f"{g['vram_experts']} slab experts against the default's {g0['vram_experts']}, "
            f"{g['spare_gib'] - g0['spare_gib']:.2f} GiB more of spare slots" if g and g0 else "no gate result")
    if default["ok"]:
        rule = "PASS on the default config: the default is the verdict"
    elif v["ok"]:
        rule = (f"PASS with {spare} spare slots only: \"hot_spare\": {spare} becomes the deepseek-v4.1 profile's default "
                f"(its slab cost: {slab}) and item 11 is judged on it, {single_text}")
    elif v["ok"] is None or default["ok"] is None:
        rule = "no verdict: a server's result is missing (above)"
    else:
        rule = "FAIL on both servers: item 11 is reported with the step breakdown"

    def cell(x, ratio=False):
        return "" if x is None else f"{fmt(x, 2)} ({x / single_off:.2f}x)" if ratio and single_off else fmt(x, 2)

    def verdict_cell(x):
        return "no verdict" if x is None else "PASS" if x else "FAIL"
    rows = [["spare slab slots per layer (--hot-spare)", g0.get("hot_spare", "n/a"), spare],
            ["slab experts (out/gate, out/gate-spare)", g0.get("vram_experts", "n/a"), g.get("vram_experts", "n/a")],
            ["the single user, batching on, tok/s (x off)", cell(default["single"], True), cell(v["single"], True)],
            ["steady-c4 aggregate, mean of the repeats (x the baseline)", cell(default["agg"], True), cell(v["agg"], True)],
            ["steady-c4 lowest repeat's aggregate", cell(default["agg_min"], True), cell(v["agg_min"], True)],
            ["steady-c4 slowest user, tok/s", cell(default["slowest"]), cell(v["slowest"])],
            ["item 11 on this server", verdict_cell(default["ok"]), verdict_cell(v["ok"])]]
    return ["", f"- item 11 with {spare} spare slab slots per layer: {text} (the second cbatch server, --hot-spare "
            f"{spare}, out/gate-spare: {slab})",
            f"- item 11's verdict over both servers (amendment 1's rule: the default passes -> its verdict; only the "
            f"variant passes -> it becomes the deepseek-v4.1 profile's default and item 11 is judged on it, its single "
            f"user re-checked on it): {rule}", "",
            f"Both servers side by side, the baseline the single user with batching off on the default config "
            f"({fmt(single_off, 2) if single_off else 'no result'} tok/s):", "",
            table(["", "default config", f"{spare} spare slots per layer"], rows)], [
            f"## Item 11's steady state with {spare} spare slots (out/cbatch/steady-c4-spare.json)", "",
            chat_load.steady_section(os.path.join(d, "steady-c4-spare.json")) if steady else
            "no steady result (" + cause(o, ["engine-server-spare", "steady-c4-spare"],
                                         "out/cbatch/steady-c4-spare.json") + ")", ""]


def cbatch_head(o, stamp, build, cost, model, prefix=""):
    """cbatch: item 11's single user with batching off and on, N users end to end,
    the 4 users' steady state and the one item 11 verdict, the batch's own
    figures from the engine's log, items 7 and 8 once. prefix: the batch server's logs' (the combined plan's
    out/cbatch-<file>; that plan runs no load-c<N> nor det-sampling)."""
    combined = prefix == "cbatch-"
    measured_before = ("not in this plan: measured on 2026-10-14 (strata-engine/2026-10-14-cbatch), "
                       "decision-c04-long-prefill-robustness amendment 2")
    import chat_load
    import det_sampling_check
    d = os.path.join(o, "cbatch")
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    byn = sorted((f for f in files if re.fullmatch(r"load-c\d+\.json", f)), key=lambda f: int(f[6:-5]))

    def run_of(name):
        return ((load_json(os.path.join(d, name)) or {}).get("runs") or [None])[0]
    spent = session_spent(o, cost)
    # the intro names the plan that ran and its steps
    runs = ("one user at short context off and on, then the steady state of 4 users decoding at once (item 11's verdict) "
            "and of the reported user counts (steady-c<N>); no load-c<N> nor det-sampling (measured on 2026-10-14)"
            if combined else "one user at short context off and on, then N users together on the C01 16K conversations "
            "and the steady state of 4 users decoding at once; det_sampling_check.py's items 7 and 8")
    out = [f"# Batched decode on the real DeepSeek-V4.1-Flash Q2_K ({stamp})", "",
           f"Session of `tools/dsv41/pod/e07.sh` with E07_PLAN={'cbatch-clong' if combined else 'cbatch'}, its item 11 "
           "part: serve/server.py on dsv41-engine as "
           "qengine runs it, first with batching off (one request at a time, the slab of gate-off) for the single user's "
           "before figure, then with the engine's batch slots (the server's parallel requests: a request alone decodes "
           f"alone, the others one token per slot a step); tools/dsv41/quality/chat_load.py: {runs}; every step with its "
           "own timeout.", "",
           f"Build: Strata {build} (the tarball's MANIFEST). Cost: {cost}. {spent} {model}", "",
           "## Item 11", ""]
    off, on = run_of("load-short-off.json"), run_of("load-short.json")
    single = on["summary"].get("decode_tok_s_user_median") if on else None
    single_off = off["summary"].get("decode_tok_s_user_median") if off else None
    # no batching-off server; steady-c4's base is the single user on
    # the batched server (its solo path: GEN, nothing else decoding), the A/B in its place
    base, base_what = ((single_off, "the single user with batching off on this pod") if single_off else
                       (single, "the single user on the batched server (its solo path; no batching-off server in "
                                "this plan)"))
    rows, info = [], []
    single_ok, single_text = None, "the single user: no result (" + cause(o, "load-short", "out/cbatch/load-short.json") + ")"
    if single is not None and single_off:
        # one sign convention: on against off, + is faster with batching on
        change = (single - single_off) / single_off
        single_ok = single >= 30 and change >= -0.05
        n_off = (off.get("summary") or {}).get("requests")
        rows.append(["1 user, short context", fmt(single_off, 2), fmt(single, 2), f"{100 * change:+.1f}% (on against off)",
                     "", "", "the single user: >= 30 tok/s, within 5% of off"])
        single_text = (f"the single user {fmt(single, 2)} tok/s with batching on against {fmt(single_off, 2)} off "
                       f"({100 * change:+.1f}%, on against off; goal >= 30 and within 5%; the baseline from {n_off} "
                       "requests, load-short-off)")
    # the verdict is the steady state's (every user decoding,
    # exactly N rows a step) against the single user with batching off; the load-c<N> runs are the user-visible figures
    sfile = "steady-c4.json"
    steady = load_json(os.path.join(d, sfile)) or {}
    sm = steady.get("summary") or {}
    # the end-to-end figures of steady-c4's own rows (its prompt reads included), on
    # this code; load-c4's when the plan ran it
    e2e = chat_load.steady_end_to_end(steady.get("repeats") or [])
    e2e_text = ("no end-to-end result ("
                + (measured_before if combined else cause(o, "load-c4", "out/cbatch/load-c4.json")) + ")")
    if e2e:
        e2e_text = (f"steady-c4's own replies end to end ({e2e['repeats']} repeats; each user's whole reply, its tokens "
                    f"after the first over its first to last token, the prompt reads that held it included): the slowest "
                    f"user {fmt(e2e['user_tok_s_min'], 2)} tok/s, the median {fmt(e2e['user_tok_s_median'], 2)}; "
                    f"{fmt(e2e['aggregate_tok_s_mean'], 2)} tok/s in aggregate over the decode window (repeats "
                    f"{fmt(e2e['aggregate_tok_s_min'], 2)} to {fmt(e2e['aggregate_tok_s_max'], 2)}; the first user's "
                    "first token to the last user's last)" +
                    (f" = {e2e['aggregate_tok_s_mean'] / single_off:.2f}x the baseline" if single_off else "") +
                    "; the users' first tokens at " + " / ".join(f"{a:.2f}-{b:.2f}" for a, b in e2e["first_token_s"]) +
                    " s (in their order, lowest to highest over the repeats)")
    for f in byn:
        r = run_of(f)
        if not r:
            continue
        n, sm_n = int(f[6:-5]), r["summary"]
        win = chat_load.decode_window(r["rows"])
        rows.append([f"{n} users at once, end to end", "", fmt(sm_n.get("decode_tok_s_user_median"), 2),
                     fmt(sm_n.get("decode_tok_s_user_min"), 2), fmt(win["decode_window_tok_s"], 2),
                     fmt(sm_n.get("aggregate_tok_s"), 2), "user-visible, the prompt reads included (reported)"])
        if n == 4:                                  # the plan's own end-to-end run, when it ran one
            e2e_text = (f"{fmt(win['decode_window_tok_s'], 2)} tok/s in aggregate over the decode window "
                        f"({fmt(sm_n.get('aggregate_tok_s'), 2)} over the run's wall time), the slowest user "
                        f"{fmt(sm_n.get('decode_tok_s_user_min'), 2)} tok/s (load-c4)")
        else:
            info.append(f"- {n} users end to end (reported): per user median {fmt(sm_n.get('decode_tok_s_user_median'), 2)}, "
                        f"slowest {fmt(sm_n.get('decode_tok_s_user_min'), 2)}, aggregate {fmt(win['decode_window_tok_s'], 2)} "
                        f"over the decode window, {fmt(sm_n.get('aggregate_tok_s'), 2)} over the wall time")
    agg, slowest = sm.get("aggregate_tok_s_mean"), sm.get("user_tok_s_min")
    measured = steady.get("ok") and steady.get("users") == 4 and agg is not None
    # the steady state of more users (steady-c<N>.json, N != 4), reported; without one,
    # the 8 users' figures C03 had, labelled stale
    more = sorted((f for f in files if re.fullmatch(r"steady-c\d+\.json", f) and f != sfile), key=lambda f: int(f[8:-5]))
    more_text = []
    for f in more:
        j = load_json(os.path.join(d, f)) or {}
        m = j.get("summary") or {}
        if j.get("ok") and m.get("aggregate_tok_s_mean") is not None:
            more_text.append(f"{j['users']} users decoding at once ({f}, reported, not a bar: {m['measured']} repeats of "
                             f"a {j['window_s']:g} s window with exactly {j['users']} rows a step) "
                             f"{fmt(m['aggregate_tok_s_mean'], 2)} tok/s in aggregate"
                             + (f" = {m['aggregate_tok_s_mean'] / single_off:.2f}x the baseline" if single_off else "") +
                             f", per user {fmt(m['user_tok_s_median'], 2)} (median), the slowest "
                             f"{fmt(m['user_tok_s_min'], 2)}")
            rows.append([f"{j['users']} users decoding at once, steady ({m['measured']} x {j['window_s']:g} s, reported)",
                         "", fmt(m.get("user_tok_s_median"), 2), fmt(m.get("user_tok_s_min"), 2),
                         fmt(m["aggregate_tok_s_mean"], 2), "", "reported, not a bar"])
        else:
            more_text.append(f"{f[8:-5]} users: no steady result ({cause(o, f[:-5], 'out/cbatch/' + f)})")
    if not more and not byn:
        more_text.append(f"8 users: not measured in this session; the last figures are {STALE_8_USERS}")
    if measured:
        rows.append([f"4 users decoding at once, steady ({sm['measured']} x {steady['window_s']:g} s, 4 rows a step)", "",
                     fmt(sm.get("user_tok_s_median"), 2), fmt(slowest, 2), fmt(agg, 2), "",
                     "item 11: aggregate >= 2 x the single user with batching off, each >= 20 tok/s"])
    if measured and single_off and single_ok is not None:
        ok = agg >= 2 * single_off and slowest >= 20 and single_ok
        verdict = (f"- item 11: {'PASS' if ok else 'FAIL'}: 4 users decoding at "
                   f"once (steady-c4: {sm['measured']} repeats of a {steady['window_s']:g} s window with exactly 4 rows a "
                   f"step) {fmt(agg, 2)} tok/s in aggregate (the mean; repeats {fmt(sm['aggregate_tok_s_min'], 2)} to "
                   f"{fmt(sm['aggregate_tok_s_max'], 2)}) = {agg / single_off:.2f}x the baseline, the single user with "
                   f"batching off ({fmt(single_off, 2)} tok/s; {agg / single:.2f}x the {fmt(single, 2)} with batching on), the "
                   f"slowest user {fmt(slowest, 2)} tok/s (goal: >= 2x and each >= 20); {single_text}. User-visible, the "
                   f"prompt reads included (end to end, this code): {e2e_text}" +
                   (f". More users: {'; '.join(more_text)}" if more_text else ""))
    else:
        why = (f"no 4-user steady result ({cause(o, 'steady-c4', 'out/cbatch/steady-c4.json')})" if not measured else
               single_text)
        verdict = (f"- item 11: no verdict: {why}. User-visible, the prompt "
                   f"reads included (end to end): {e2e_text}" + (f". More users: {'; '.join(more_text)}" if more_text else ""))
    # the baseline's change since the previous combined session, and the machines
    if combined and single_off:
        machine, cpu = session_machine(o)
        p = PREVIOUS_BASELINE
        info.append(f"- the baseline (the single user with batching off): {fmt(single_off, 2)} tok/s here, machine "
                    f"{machine} ({cpu}); {p['tok_s']} tok/s in {p['sheet']}, machine {p['machine']} ({p['cpu']}), build "
                    f"{p['build']}, before the lone request's eight-lane pick: {100 * (single_off - p['tok_s']) / p['tok_s']:+.1f}%, " +
                    ("the same machine: the code's change" if machine == p["machine"] else
                     "another machine: the code's change and the host's together") +
                    f"; item 11's bar moved with it, 2.0x = {fmt(2 * single_off, 2)} tok/s against {fmt(2 * p['tok_s'], 2)}")
    # the combined plan's second cbatch server beside this one
    default = {"single": single, "agg": agg if measured else None, "agg_min": sm.get("aggregate_tok_s_min") if measured
               else None, "slowest": slowest if measured else None,
               "ok": ok if measured and single_off and single_ok is not None else None}
    variant, variant_steady = spare_variant(o, d, single_off, default) if combined else ([], [])
    steady_rows = []
    if measured and single_off and single:
        for label, v in (("aggregate (mean of the repeats)", agg), ("lowest repeat's aggregate", sm["aggregate_tok_s_min"]),
                         ("per user (median)", sm.get("user_tok_s_median")), ("slowest user", slowest)):
            steady_rows.append([label, fmt(v, 2), f"{v / single_off:.2f}x", f"{v / single:.2f}x"])
    out += [table(["run", "per user tok/s off", "per user tok/s (median)", "slowest user tok/s",
                   "aggregate decode tok/s (decode window)", "aggregate tok/s (wall, C01's)", "goal"], rows), "",
            verdict, *info, *variant, "",
            "Per-user decode tok/s end to end: each user's tokens after its first over their time (chat_load.py); the "
            "aggregate over the decode window: every request's tokens after its first over the time at least one request "
            "was decoding (chat_load.py decode_window); the C01-style aggregate: every generated token over the run's wall "
            "time. A prompt read while slots decode holds their steps (C04 reads it in "
            "chunks of its --chunk with a step between two), which both end-to-end aggregates and the per-user rate count; the "
            "steady state leaves them out (its window opens once every prompt is read).", "",
            f"## Item 11's steady state (out/cbatch/{sfile})", "",
            (table(["", "tok/s", "x the single user, batching off (the baseline)", "x batching on"], steady_rows) + "\n\n"
             if steady_rows else "") + (chat_load.steady_section(os.path.join(d, sfile)) if steady else
                                         f"no steady result ({cause(o, 'steady-c4', 'out/cbatch/steady-c4.json')})"), "",
            *f2_note(steady),
            *[x for f in more for x in (f"## The steady state with {f[8:-5]} users, reported (out/cbatch/{f})", "",
                                        chat_load.steady_section(os.path.join(d, f)), "")],
            *variant_steady,
            "## The runs (out/cbatch/)", "",
            chat_load.section([os.path.join(d, f) for f in ["load-short-off.json", "load-short.json", "load-short-spare.json"]
                               + byn if os.path.exists(os.path.join(d, f))], logs="cbatch"), ""]
    log = read(os.path.join(o, prefix + "quality-engine.log"))
    slots = BATCH_SLOT_RE.findall(log)
    rep_n = sum(1 for x in slots if x[9].startswith("reproducible"))
    out += [f"## The batch in the engine's log (out/{prefix}quality-engine.log)", "",
            "\n".join(f"- {x}" for x in BATCH_RE.findall(log)) or "no batch line (no request decoded in a slot)", "",
            f"{len(slots)} requests decoded in batch slots, {rep_n} of them reproducible (the deterministic mode; in older "
            "engines only those decoded alone, the engine of 3a6d6b3 the same, until the promotion fix), the others marked not reproducible (the mode off, or "
            "such an engine; the server's replies carry \"strata_reproducible\": false); "
            f"{len(BATCH_READ_RE.findall(log))} prompts read beside decoding slots; every slot's stalls:", "",
            table(["request", "slot", "generated", "finish", "ms", "steps", "beside others", "rows a step", "stalled ms"],
                  [list(x[:9]) for x in slots]) if slots else "", "",
            "The prompts read beside decoding slots (C04's chunks):", "", interleaved_reads(log), "",
            "### Where a batch step's time goes", "",
            "Per batch (each \"all\" row starts one) and part of it: ms a step from one step's start to the next "
            "(the prompts read between them apart) = inside the step (the backend's batch_decode: the graph's launch "
            "with the rows' inputs, the doorbells served, of them the waits for the device's ring, the CPU jobs and "
            "their plan; the tail to the logits; the boundary and its swap waits; the device's own graph time beside "
            "them) + between steps (the picks of the rows' tokens, their BT lines and the flush, the admissions' "
            "copies, the loop with its stdin poll). Per step: the rows' routed pairs, their distinct experts, hits, "
            "misses and CPU jobs (one per distinct missed expert for every row routed to it) and the swaps. The part "
            "\"within 15 s of a prompt read beside the slots\" is steady-c4's window after the last user's read; the "
            "single user's own figures are its requests' \"decode experts\" lines.", "",
            step_parts(log), "", ran_alone([(f"out/{prefix}quality-engine.log", log)]), "",
            "## Items 7 and 8 once (out/det-sampling.json)", "",
            "Not" + measured_before[3:] + "." if combined
            else det_sampling_check.section(os.path.join(o, "det-sampling.json")),
            ""]
    # the second cbatch server's own lines, its step breakdown beside the default's
    spare_log = read(os.path.join(o, "spare-quality-engine.log"))
    if spare_log:
        out += ["## The batch with more spare slots in the engine's log (out/spare-quality-engine.log)", "",
                "\n".join(f"- {x}" for x in BATCH_RE.findall(spare_log)) or "no batch line (no request decoded in a slot)",
                "", "### Where its batch step's time goes", "", step_parts(spare_log), ""]
    for name, sub in (("gate (the batch)", "gate"), ("gate-off (no batch)", "gate-off"),
                      ("gate-spare (the batch, more spare slots)", "gate-spare")):
        g = load_json(os.path.join(o, sub, "gate.json")) or {}
        if g:
            out += [f"## Placement: {name} (out/{sub}/gate.json)", "", table(["", ""], [(k, v) for k, v in g.items()]), ""]
    for label, cmd in (("off", "off-quality-server.cmd"), ("on", prefix + "quality-server.cmd"),
                       ("on, more spare slots", "spare-quality-server.cmd")):
        if label.startswith("on, ") and not os.path.exists(os.path.join(o, cmd)):
            continue
        out += [f"## Server as run, batching {label}", "", "```", read(os.path.join(o, cmd)).strip() or "not started", "```", ""]
    out += ["## Host", "", host_section(o), "", monitor_peaks(o), ""]
    return out


# the engine quality run (item 4's bar for the batched engine) and its API fresh run (reported as in that run), both
# results directories of earlier sessions under BUILD_ROOT, and session 4's single user with batching on (the
# cbatch-clong session of 2026-10-16), the speed bar's reference
BUILD_ROOT = os.environ.get("BUILD_ROOT", os.path.expanduser("~/dsv41-build"))
CDET_Q02_ENGINE = os.environ.get("CDET_Q02_ENGINE",
                                 os.path.join(BUILD_ROOT, "engine/e07a-pod/20261013T000542Z/out/quality"))
CDET_Q02_API = os.environ.get("CDET_Q02_API", os.path.join(BUILD_ROOT, "quality/api/20261011T113443Z"))
SESSION4_SINGLE = {"tok_s": 61.55, "sheet": "docs/RESULTS.md"}
CBATCH_TURNS_NOTE = "load-short's turns of 256 tokens at a 1K context"


def item3_complete(quality):
    """Item 3 batched came back on every task (each one's last lm_eval_status.txt line "<task> rc=0" and
    its results file); the tasks that did not, else []."""
    import analyze
    import quality as Q
    status = read(os.path.join(quality, "lm_eval_status.txt")).splitlines()
    found = Q.find_results(Path(quality)) if os.path.isdir(quality) else {}
    missing = []
    for task in analyze.TASKS:
        st = [x for x in status if x.startswith(task + " ")]
        if not st or st[-1] != f"{task} rc=0" or task not in found:
            missing.append(f"{task} ({st[-1] if st else 'not run'})")
    return missing


def cdet_head(o, stamp, build, cost, model, plan="cdet"):
    """cdet: item 14 (det_batch_check.py), items 7 and 8, the speed bars (the
    single user with batching on against session 4 and off, steady-c4 against the single user off), item 3 batched
    (lm-eval at 4 concurrent requests on the batch slots) and item 4 against the engine quality run, the API fresh run
    reported as in that run. cfinal: the same less item 14's det_batch_check.py, items 7 and 8 and the A/B,
    steady-c4's every user >= 20 tok/s in its verdict, and item 14 as amended from items 3 and 4."""
    import analyze
    import chat_load
    import det_batch_check
    import det_sampling_check
    import q_report
    d = os.path.join(o, "cbatch")
    spent = session_spent(o, cost)

    def run_of(name):
        return ((load_json(os.path.join(d, name)) or {}).get("runs") or [None])[0]
    final = plan == "cfinal"
    if final:
        out = [f"# C05 cycle 4: the fast path's speed and the quality of batched requests ({stamp})", "",
               "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=cfinal (item 14 read as no quality loss from "
               "batching: item 3 batched on all three tasks and item 4 "
               "against the engine quality run): serve/server.py on dsv41-engine with the engine's batch slots, its deterministic "
               "mode on and the CPU experts in the fork's numerics (the engine's default): the single "
               "user, the steady state of 4 users (item 11), then lm-eval through api_proxy.py at 4 concurrent requests on "
               "the fixed-seed subsets (item 3 batched); every step with its own timeout.", "",
               f"Build: Strata {build} (the tarball's MANIFEST). Cost: {cost}. {spent} {model}", "",
               "## Speed (item 11)", ""]
    else:
        out = [f"# C05: batched requests as alone ones, and their quality ({stamp})", "",
               "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=cdet (items 14, 3 and 4): serve/server.py on "
               "dsv41-engine with the engine's batch slots and its deterministic mode on: the single user, the same user "
               "with the mode off (the speed A/B's gpu arm), the steady state of 4 users, det_batch_check.py (item 14), "
               "det_sampling_check.py (items 7 and 8), then lm-eval through api_proxy.py at 4 concurrent requests on Q02's "
               "fixed-seed subsets (item 3 batched); last the same server with --cpu-numerics fork for the A/B's other arm "
               "(sessions before it ran a batching-off server first); every step with "
               "its own timeout.", "",
               f"Build: Strata {build} (the tarball's MANIFEST). Cost: {cost}. {spent} {model}", "",
               "## Item 14 (out/det-batch.json)", "",
               det_batch_check.section(os.path.join(o, "det-batch.json")) if os.path.exists(os.path.join(o, "det-batch.json"))
               else f"no result ({cause(o, 'det-batch', 'out/det-batch.json')})", "",
               "## Items 7 and 8 (out/det-sampling.json)", "",
               det_sampling_check.section(os.path.join(o, "det-sampling.json"))
               if os.path.exists(os.path.join(o, "det-sampling.json"))
               else f"no result ({cause(o, 'det-sampling', 'out/det-sampling.json')})", "",
               "## Speed", ""]
    off, on = run_of("load-short-off.json"), run_of("load-short.json")
    single = on["summary"].get("decode_tok_s_user_median") if on else None
    single_off = off["summary"].get("decode_tok_s_user_median") if off else None
    # no batching-off server; steady-c4's base is the single user on
    # the batched server (its solo path: GEN, nothing else decoding), the A/B in its place
    base, base_what = ((single_off, "the single user with batching off on this pod") if single_off else
                       (single, "the single user on the batched server (its solo path; no batching-off server in "
                                "this plan)"))
    steady = load_json(os.path.join(d, "steady-c4.json")) or {}
    sm = steady.get("summary") or {}
    agg = sm.get("aggregate_tok_s_mean") if steady.get("ok") and steady.get("users") == 4 else None
    ref = SESSION4_SINGLE["tok_s"]
    lines = []
    if single is not None:
        ok1 = single >= 30 and single >= 0.95 * ref
        lines.append(f"- the single user with batching on: {'PASS' if ok1 else 'FAIL'}: {fmt(single, 2)} tok/s against "
                     f"session 4's {ref} ({100 * (single - ref) / ref:+.1f}%, {SESSION4_SINGLE['sheet']}; goal within 5% "
                     "and >= 30)" + (f"; {fmt(single_off, 2)} with batching off on this pod "
                                     f"({100 * (single - single_off) / single_off:+.1f}%, on against off)"
                                     if single_off else ""))
    else:
        lines.append("- the single user with batching on: no result (" + cause(o, "load-short", "out/cbatch/load-short.json") + ")")
    if agg is not None and base:
        # cfinal: item 11's whole bar, every user >= 20 tok/s too (cdet's verdict was the aggregate's alone)
        slowest = sm.get("user_tok_s_min")
        ok2 = agg >= 2 * base and (not final or (slowest is not None and slowest >= 20))
        lines.append(f"- steady-c4: {'PASS' if ok2 else 'FAIL'}: {fmt(agg, 2)} tok/s in aggregate (the mean of "
                     f"{sm['measured']} repeats of a {steady['window_s']:g} s window with exactly 4 rows a step) = "
                     f"{agg / base:.2f}x {base_what} ({fmt(base, 2)} tok/s; goal >= 2x), the slowest user "
                     f"{fmt(slowest, 2)} tok/s" + (" (goal: each >= 20)" if final else ""))
    else:
        lines.append("- steady-c4: no verdict (" + (cause(o, "steady-c4", "out/cbatch/steady-c4.json") if agg is None
                                                     else "no single user") + ")")
    # the speed A/B: load-short's user with the deterministic mode off, the CPU experts in the GPU's numerics (the
    # default since C05) and in the fork's (--cpu-numerics fork, the engine before C05), on one host
    ab_gpu, ab_fork = run_of("ab-gpu.json"), run_of("ab-fork.json")
    g_ts = ab_gpu["summary"].get("decode_tok_s_user_median") if ab_gpu else None
    f_ts = ab_fork["summary"].get("decode_tok_s_user_median") if ab_fork else None
    if g_ts and f_ts:
        lines.append(f"- the speed A/B (the single user, the deterministic mode off, "
                     f"{CBATCH_TURNS_NOTE}): CPU experts in the GPU's numerics {fmt(g_ts, 2)} tok/s, in the fork's "
                     f"{fmt(f_ts, 2)} tok/s ({100 * (g_ts - f_ts) / f_ts:+.1f}%, gpu against fork); session 4's 61.55 ran "
                     "the fork's numerics in the mode on (before C05)" +
                     (f"; the mode on with the GPU's numerics {fmt(single, 2)} tok/s (load-short)" if single else ""))
    elif not final and (ab_gpu or ab_fork or os.path.exists(os.path.join(o, "ab-gpu.log"))):
        lines.append("- the speed A/B: incomplete (" + ", ".join(
            cause(o, n, f"out/cbatch/{n}.json") for n, r in (("ab-gpu", g_ts), ("ab-fork", f_ts)) if not r) + ")")
    out += lines + ["", (chat_load.steady_section(os.path.join(d, "steady-c4.json")) if steady else ""), "",
                    chat_load.section([os.path.join(d, f) for f in ("load-short-off.json", "load-short.json",
                                                                    "ab-gpu.json", "ab-fork.json")
                                       if os.path.exists(os.path.join(d, f))], logs="cbatch"), ""]
    quality = os.path.join(o, "quality")
    out += ["## Item 3 batched (out/quality)", "",
            q_report.results_section(quality, "engine") if os.path.isdir(quality) else "no out/quality came back", "",
            "## Item 4", ""]
    have = os.path.isdir(quality) and os.path.isdir(CDET_Q02_ENGINE)
    bar = analyze.same_engine_bar(Path(CDET_Q02_ENGINE), Path(quality)) if have else None
    if have:
        out += [analyze.same_engine_markdown(bar), ""]
    else:
        out += [f"not evaluated: {'no out/quality' if not os.path.isdir(quality) else 'no Q02 engine run at ' + CDET_Q02_ENGINE}",
                ""]
    out += ["### Against the API fresh run (reported, as in Q02)", "",
            "The engine column is this session's batched run; the API pair's item 4 line below is Q02's reading "
            "(engine - API fresh), which holds the Q2_K quantization's loss: on HumanEval "
            "the fork ran 0.878 against the engine's 0.890 in Q02 (its 2026-10-13 sessions), so the loss "
            "against the API there is the quantization's; GSM8K and IFEval have no fork run.", ""]
    if os.path.isdir(quality) and os.path.isdir(CDET_Q02_API):
        out += [analyze.markdown(analyze.analyze({"api_fresh": Path(CDET_Q02_API), "fork": None, "engine": Path(quality),
                                                  "api_then": analyze.API_THEN if analyze.API_THEN.is_dir() else None,
                                                  "exl3": analyze.EXL3_THEN if analyze.EXL3_THEN.is_dir() else None})), ""]
    else:
        out += [f"not computed: {'no out/quality' if not os.path.isdir(quality) else 'no API fresh run at ' + CDET_Q02_API}", ""]
    if final:
        # item 14 read as no quality loss from batching: items 3 and 4 together
        missing = item3_complete(quality)
        if missing:
            v14, why = "not evaluated", "item 3 did not come back on " + ", ".join(missing)
        elif bar is None or bar["verdict"] == "not evaluated":
            v14, why = "not evaluated", "item 4 not evaluated" + (f" ({'; '.join(bar['reasons'])})" if bar else "")
        else:
            v14 = bar["verdict"]
            why = (f"item 3 batched on all three tasks; item 4 against Q02's engine run {bar['verdict']}: "
                   + "; ".join(bar["reasons"]))
        out += ["## Item 14 (the owner's amendment of 2026-10-19: no quality loss from batching)", "",
                f"Goal item 14 (item 3 batched on all three tasks and item 4 against Q02's engine run): **{v14}** "
                f"({why}).", ""]
    log = read(os.path.join(o, "quality-engine.log"))
    slots = BATCH_SLOT_RE.findall(log)
    rep_n = sum(1 for x in slots if x[9].startswith("reproducible"))
    out += ["## The batch in the engine's log (out/quality-engine.log)", "",
            "\n".join(f"- {x}" for x in BATCH_RE.findall(log)) or "no batch line (no request decoded in a slot)", "",
            f"{len(slots)} requests decoded in batch slots, {rep_n} of them reproducible (the deterministic mode; "
            "the engine of 3a6d6b3 marked only those decoded alone), the others not; "
            f"{len(BATCH_READ_RE.findall(log))} prompts read beside decoding slots.", "",
            "### Where a batch step's time goes", "", step_parts(log), ""]
    for name, sub in (("gate (the batch)", "gate"), ("gate-off (no batch)", "gate-off")):
        if not os.path.isdir(os.path.join(o, sub)):
            continue
        g = load_json(os.path.join(o, sub, "gate.json")) or {}
        if g:
            out += [f"## Placement: {name} (out/{sub}/gate.json)", "", table(["", ""], [(k, v) for k, v in g.items()]), ""]
    for label, cmd in (("batching off", "off-quality-server.cmd"), ("batching on", "quality-server.cmd"),
                       ("the A/B's fork arm (--cpu-numerics fork)", "fork-quality-server.cmd")):
        if (label == "batching off" or (final and cmd.startswith("fork-"))) and not os.path.exists(os.path.join(o, cmd)):
            continue
        out += [f"## Server as run, {label}", "", "```", read(os.path.join(o, cmd)).strip() or "not started", "```", ""]
    out += ["## Host", "", host_section(o), "", monitor_peaks(o), ""]
    return out


def fork_gpu_peak(o):
    """qfork: the GPU memory used at its peak while the fork's llama-server ran (quality/pod_server.sh's nvidia-smi
    samples, every 1 s from the server's start to server-stop)."""
    rows = [x.split(",") for x in read(os.path.join(o, "quality-gpu.csv")).splitlines()]
    used = [(int(r[1]), int(r[2])) for r in rows if len(r) == 3 and r[1].strip().isdigit() and r[2].strip().isdigit()]
    if not used:
        return "Fork server peak GPU memory: n/a (no samples in out/quality-gpu.csv)"
    peak, total = max(used)
    return (f"Fork server peak GPU memory: {peak} MiB used of {total} MiB, {total - peak} MiB free at the peak "
            f"(out/quality-gpu.csv, nvidia-smi every 1 s, {len(used)} samples)")


def monitor_peaks(o):
    m = read(os.path.join(o, "monitor.log"))
    gpu = [int(x) for x in re.findall(r"GPU memory used, total: (\d+) MiB", m)]
    cg = [int(x) for x in re.findall(r"cgroup memory usage: (\d+) MiB", m)]
    cg_txt = f"{max(cg) / 1024:.1f} GiB" if cg else "n/a"
    return (f"peak GPU memory used {max(gpu) if gpu else 'n/a'} MiB, peak memory cgroup usage {cg_txt} "
            f"(out/monitor.log, every 30 s)")


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    out_dir, sheet_root = argv
    o = os.path.join(out_dir, "out")
    full_stamp = os.path.basename(os.path.normpath(out_dir))
    stamp = full_stamp[:8]
    plan = re.search(r"^plan: (\S+?),", read(os.path.join(o, "plan.log")), re.M)
    # no out/plan.log when the run never started (every machine rejected): e07.sh's own record of the plan
    pname = plan.group(1) if plan else read(os.path.join(out_dir, "session-plan")).strip() or None
    e08 = pname == "e08"
    e09 = pname == "e09"
    e07b = pname == "e07b"
    qplan = pname if pname in ("qfork", "qengine", "qdiag", "qdet", "cbase", "ccache", "cbatch", "clong",
                               "cbatch-clong", "cdet", "cfinal") else None
    # the engine's quality plans: the per-phase diagnostic into OUT/out before the logs are copied
    diag = None
    if qplan in ("qengine", "qdiag") and os.path.isdir(o):
        diag, diag_data = qdiag_report.write(o)
    name = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}-" + (qplan if qplan else "e07b" if e07b else "e09" if e09 else
                                                         "e08" if e08 else
                                                         "e07a" + ("-cycle2" if pname == "cycle2" else ""))
    sheet = os.path.join(sheet_root, name)
    # another session's sheet of the same day and plan is kept: this one takes its start time too
    marker = os.path.join(sheet, "logs", "e07-out.txt")
    if os.path.exists(marker) and read(marker).strip() != os.path.abspath(out_dir):
        sheet += "-" + full_stamp[9:13]
        marker = os.path.join(sheet, "logs", "e07-out.txt")
    logs = os.path.join(sheet, "logs")
    os.makedirs(logs, exist_ok=True)
    with open(marker, "w") as f:
        f.write(os.path.abspath(out_dir) + "\n")
    if os.path.isdir(o):
        for root, _, files in os.walk(o):
            for f in files:
                if f.endswith(".bin"):
                    continue
                rel = os.path.relpath(os.path.join(root, f), o)
                os.makedirs(os.path.dirname(os.path.join(logs, rel)), exist_ok=True)
                shutil.copy2(os.path.join(root, f), os.path.join(logs, rel))
    for f in ("e07.log", "cost.txt", "pod.json", "probe.log", "session-plan", "session-cap", "session-lost"):
        if os.path.exists(os.path.join(out_dir, f)):
            shutil.copy2(os.path.join(out_dir, f), logs)
    gate = load_json(os.path.join(o, "gate", "gate.json")) or {}
    manifest = read(os.path.join(o, "env.log"))
    commit = re.search(r"^commit: (\S+)", manifest, re.M)
    steps = [ln for ln in read(os.path.join(o, "steps.log")).splitlines() if ln]
    serve = read(os.path.join(o, "serve.log"))
    chats = load_json(os.path.join(o, "serve", "chats.json")) or []
    tok = [ln for ln in read(os.path.join(o, "tokenizer.log")).splitlines() if ln.startswith("tokenizer_parity.py")]
    dl = load_json(os.path.join(o, "model-download.json"))
    if dl and dl.get("too_slow"):
        model = f"Model from Hugging Face: stopped by the throughput guard: {dl['guard']} (model-download.json)."
    elif dl:
        # no network volume: the model came from Hugging Face (pod/hf-download.sh)
        model = (f"Model: {dl['repo']}@{dl['revision']} from Hugging Face, {dl['files']} files, {dl['bytes'] / 1e9:.1f} GB "
                 f"in {dl['download_s']} s = {dl['gb_per_s']} GB/s; size and sha256 against the HF API "
                 f"({dl['check_s']} s): {'ok' if dl['ok'] else 'FAIL'} (model-download.json).")
    else:
        vol = "unchanged" if "the volume is unchanged" in read(os.path.join(o, "volume-after.log")) else "NOT shown unchanged"
        model = f"Network volume: {vol} (volume-before.txt against volume-after.txt)."
    cost = read(os.path.join(out_dir, "cost.txt")).strip() or "n/a"
    if qplan == "qdiag":
        head = qdiag_head(o, stamp, commit.group(1) if commit else "n/a", cost, model, diag)
        if gate:
            head += ["## Placement (out/gate/gate.json)", "", table(["", ""], [(k, v) for k, v in gate.items()]), ""]
    elif qplan == "cbatch":
        head = cbatch_head(o, stamp, commit.group(1) if commit else "n/a", cost, model)
    elif qplan == "clong":
        head = clong_head(o, stamp, commit.group(1) if commit else "n/a", cost, model)
    elif qplan in ("cdet", "cfinal"):
        head = cdet_head(o, stamp, commit.group(1) if commit else "n/a", cost, model, plan=qplan)
    elif qplan == "cbatch-clong":
        # one session, each item its own table and verdict
        head = cbatch_head(o, stamp, commit.group(1) if commit else "n/a", cost, model, prefix="cbatch-")
        head[0] = f"# C03 and C04: batched decode, then a long prompt and the server's robustness ({stamp})"
        head[2:2] = ["E07_PLAN=cbatch-clong: cbatch's steps, then clong's on a "
                     "third server (slots that hold the 64K prompt), one model download for both. Item 11 first, then "
                     "items 12 and 13.", ""]
        k = head.index("## Host")
        head[k:k] = clong_sections(o) + ["## Host homes held against the checkpoints (every gate of the session)", "",
                                         homes_against_checkpoints(o), ""]
    elif qplan in ("qdet", "cbase", "ccache"):
        b = commit.group(1) if commit else "n/a"
        head = (qdet_head(o, stamp, b, cost, model) if qplan == "qdet" else cbase_head(o, stamp, b, cost, model)
                if qplan == "cbase" else ccache_head(o, stamp, b, cost, model, sheet_root))
        if gate:
            head += ["## Placement (out/gate/gate.json)", "", table(["", ""], [(k, v) for k, v in gate.items()]), ""]
    elif qplan:
        head = quality_head(o, qplan, stamp, commit.group(1) if commit else "n/a", cost, model)
        if qplan == "qengine" and diag is not None:
            head += ["## GPU clocks and host load (pod/pod-samplers.sh)", "", qdiag_report.gpu_summary(diag_data)]
        if gate:
            head += ["## Placement (out/gate/gate.json)", "", table(["", ""], [(k, v) for k, v in gate.items()]), ""]
    elif e07b:
        head = [f"# The closing measurement on the real DeepSeek-V4.1-Flash Q2_K ({stamp})", "",
                "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=e07b: two-turn through the server (turn 1 the 16K prompt's text, ctx "
                "20480) with the HTTP check on the same load, one bench load (16k-e09, turn2, short-adaptive, 64k, "
                "256k), then the free-greedy and isolated MoE parity (reported); every step with its own timeout.", "",
                f"Build: Strata {commit.group(1) if commit else 'n/a'} (the tarball's MANIFEST). Cost: {cost}. {model}", "",
                "## Goal (Done means, items 1-6)", "", goal_section(o, out_dir, os.path.join(sheet, "README.md")), "",
                "## Host", "", host_section(o), "", monitor_peaks(o), "",
                "## Placement (out/gate/gate.json)", "",
                table(["", ""], [(k, v) for k, v in gate.items()]) if gate else "no gate result", "",
                "## The 16K prompt, the second turn and the server", "", prefill_section(o, "bench-e07b"), "",
                "## Speed", "", speed_section(o, ("bench-e07b",), e07b=True), "",
                "## Correctness on this card (reported, not gating)", "", parity_section(o, e07b=True), ""]
    elif e09:
        head = [f"# The 16K prompt on the real DeepSeek-V4.1-Flash Q2_K ({stamp})", "",
                "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=e09: one engine load for the 16K prompt twice (per-stage split), the short decode and "
                "the 256K prompt, then a two-turn chat through the server; every engine with --prefetch-experts 128.", "",
                f"Build: Strata {commit.group(1) if commit else 'n/a'} (the tarball's MANIFEST). Cost: {cost}. {model}", "",
                "## Host", "", host_section(o), "", monitor_peaks(o), "",
                "## Placement (out/gate/gate.json)", "",
                table(["", ""], [(k, v) for k, v in gate.items()]) if gate else "no gate result", "",
                "## The 16K prompt and the second turn", "", prefill_section(o), "",
                "## Speed", "", speed_section(o, ("bench-e09",)), ""]
    elif e08:
        head = [f"# Decode on the real DeepSeek-V4.1-Flash Q2_K, the per-request hot set ({stamp})", "",
                "Session of `tools/dsv41/pod/e07.sh` with E07_PLAN=e08: one engine load measures every hot-set policy (the serve "
                "protocol's hot= key, each from the hot set's slab with hot_reset=1).", "",
                f"Build: Strata {commit.group(1) if commit else 'n/a'} (the tarball's MANIFEST). Cost: {cost}. {model}", "",
                "## Host", "", host_section(o), "", monitor_peaks(o), "",
                "## Placement (out/gate/gate.json)", "",
                table(["", ""], [(k, v) for k, v in gate.items()]) if gate else "no gate result", "",
                "## Hot set (bench-e08)", "", hot_section(o), "",
                "## Speed", "", speed_section(o, ("bench-e08",)), ""]
    else:
        head = [f"# The real DeepSeek-V4.1-Flash Q2_K on one GPU ({stamp})", "",
                "Session of `tools/dsv41/pod/e07.sh`. Oracle: the stock fork's oracle dump run-1, the stock fork 5210c7c on an A100 SXM4 80 GB (`dsv41-moe-dump ... -n 64 -s 42 "
                "--flash-attn on --ctx-size 4096 --ubatch-size 1024`), prompt "
                "tools/dsv41/oracle/prompt.txt (31 ids, no BOS).", "",
                f"Build: Strata {commit.group(1) if commit else 'n/a'} (the tarball's MANIFEST). Cost: {cost}. {model}", "",
                "## Host", "", host_section(o), "", monitor_peaks(o), "",
                "## Placement (out/gate/gate.json)", "",
                table(["", ""], [(k, v) for k, v in gate.items()]) if gate else "no gate result", "",
                "## Correctness against the oracle", "", parity_section(o), "",
                "## Layer-0 isolation (dsv41-layer-ref, layer_parity.py)", "", isolation_section(o), "",
                "## Tokenizer parity on the real GGUF", "", "\n".join(f"- {x}" for x in tok) or "no result", "",
                "## Speed", "", speed_section(o), ""]
    lost = read(os.path.join(out_dir, "session-lost")).splitlines()
    if lost:
        # the final pull brought nothing back; this sheet is the mid-run pulls' out/ (e07.sh's session_lost)
        head[1:1] = ["", "## Session lost", "", "The final pull brought nothing back: every table here is from the last "
                     "mid-run pull (pod-session.sh's pull_steps; logs/session-lost, logs/e07.log).", ""] + \
                    [f"- {x}" for x in lost]
    elif not os.path.isdir(o):
        head[1:1] = ["", "**No run**: no out/ came back, the run never started (no machine passed the qualification below, "
                     "or the session stopped before its run; logs/e07.log)."]
    head += ["## Machine qualification", "", qualification_section(out_dir), ""]
    # E07b's server check is two-turn's HTTP part (in its section above)
    doc = head + ([] if e07b or qplan else [
           "## Server (serve_check.py --parts profile,http)", "",
           "\n".join(f"- {x}" for x in serve.splitlines() if x.startswith(("ok ", "FAIL", "note", "serve_check.py")))
           or "no result", "", "Chats (reported, not gated):", ""])
    for c in chats:
        doc += [f"- **{c['prompt']}** ({c['usage']}, {c['seconds']} s, {c['finish_reason']}):", "",
                "  " + (c["reply"] or "").replace("\n", "\n  "), ""]
    doc += ["## Steps (out/steps.log)", "", "\n".join(f"- {x}" for x in steps) or "none came back", "",
            "Parameters: every engine run's placement in logs/placement.txt, the benches' engine arguments in "
            "logs/bench-*.json (args), the prompts' sha256 in logs/prompts-manifest.json "
            "(pod/e07_prompts.py), the pod in logs/pod.json, the session log logs/e07.log.", ""]
    if qplan in ("cbatch", "clong", "cbatch-clong", "cdet", "cfinal"):
        # the files the sheet names are under its logs/ (the pod's out/), except in
        # the commands as run (their code blocks keep the paths the engine and the server were given)
        fenced, text = False, []
        for line in "\n".join(doc).split("\n"):
            fenced ^= line.startswith("```")
            text.append(line if fenced or line.startswith("```") else re.sub(r"(?<![\w./-])out/", "logs/", line))
        doc = text
    with open(os.path.join(sheet, "README.md"), "w") as f:
        f.write("\n".join(doc))
    print(f"e07_report.py: {os.path.join(sheet, 'README.md')} and {logs}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
