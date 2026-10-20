#!/usr/bin/env python3
"""The real-model session's speed phases (tools/dsv41/pod/e07-run.sh): one resident `dsv41-engine --serve` on the real model, driven
through its stdin protocol (dsv41-engine.cpp, Strata's `strata --serve`) with the exact token ids of
pod/e07_prompts.py, so every number is the engine's own for a known prompt (no template, no tokenizer in the loop).

  e07_bench.py --engine EXE --engine-args "ARGS" --prompts DIR --phases P,P,... --out OUT.json [--deadline UNIX_S]

The engine starts with ARGS plus --serve --eos-ids none (no id ends a reply, so each request generates exactly its
tokens). Phases, in the order given:
  16k     turn 1 of the 16K conversation: p16k (16,384 ids), 32 generated: its TTFT (target <= 5 s)
  turn2   its second turn, right after 16k (or 16k-e09 in e07b): p16k + the 32 generated + turn2 (512 new ids), 32
          generated: RESUME must reuse the live state (16,415 ids), its TTFT (target <= 1 s; accepted at about 2 s)
  short   short (128 ids), 481 generated: 480 decode steps at short context (target >= 30 tok/s)
  64k     p64k (65,536 ids), 481 generated: TTFT and 480 decode steps at 64K
  256k    p256k (261,664 ids), 481 generated: TTFT (target <= 3 min) and 480 decode steps filling the 262,144
          positions; the engine must still answer afterwards
  short-timed  short again, 65 generated with times=1: 64 decode steps
          timed per stage (every op launched, no token graph, so slower than short's), the split of a decode step
  e08 (the engine's default policy is adaptive):
  short-static, short-prompt, short-adaptive  short, 481 generated, with hot=<policy> hot_reset=1: each policy from
          the same slab (the hot set's), its decode hit rate overall and per layer and its swaps (goal: >= 30 tok/s
          with the best policy)
  16k-e08 p16k, 481 generated, hot_reset=1 prefill_times=1: the 16K TTFT with its per-stage split (the prefill bench's numbers),
          then 480 decode steps after the 16K prompt
  e09:
  16k-e09 p16k, 32 generated, hot_reset=1 prefill_times=1: the 16K TTFT (target <= 5 s) with its per-stage split; E07_PLAN
          e09 runs it twice with short-adaptive between (one cached conversation: the second 16K reads cold)
  e07b runs 16k-e09,turn2,short-adaptive,64k,256k on one load.
  64k and 256k send trace=0 (left out of a --route-trace file: 261,664 x 40 x 6 ids are not worth pulling)
  hot_reset=1 runs before the request's prompt (the engine's begin_request, the host waiting for every round of
  swaps), so the client's first token includes it: each phase with a hot set line also gets ttft_client_no_reset_ms,
  the client's first token less the reset's ms (2026-10-09-e08, A100-SXM4, short-adaptive's 128-id prompt: 2697 ms
  at the client, 1923 of them the reset's 3119 swaps, 774 without; the engine's prompt 773 ms). A server's request,
  which carries the slab over, pays no reset.
Per phase: its start and end in unix seconds (start_unix, end_unix: the window qdiag_report.py joins to the session
samplers), the client's time to the first T line, the engine's PP chunks and DONE fields (prompt ms, decode ms over
generated - 1 decode steps, RESUME), prompt and decode tok/s, and, between the request's start and its DONE, the
engine's bytes read from storage (/proc/PID/io read_bytes) and major page faults (nothing should come from the NVMe
in steady state: every expert is in VRAM or the pinned warm tier), then VRAM, the engine's RSS and the memory
cgroup's usage, and from the engine's log line of the request (written before its DONE) the decode's routed experts:
hits and misses (the hit rate), the CPU's jobs, MiB, ms and GB/s, the doorbell waits (the real model's
20 tok/s against the 30 of the target), and for short-timed the per-stage ms per step. The load (start to READY) and the engine's own lines (device memory, expert tiers, the measured PCIe
and CPU rates) go in too. OUT.json is rewritten after every phase, so a crash keeps what ran. Exits 1 when the engine
died, a phase failed or ran out of time.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

PHASES = {"16k": ("p16k", 32), "turn2": (None, 32), "short": ("short", 481), "64k": ("p64k", 481),
          "256k": ("p256k", 481), "short-timed": ("short", 65), "short-static": ("short", 481),
          "short-prompt": ("short", 481), "short-adaptive": ("short", 481), "16k-e08": ("p16k", 481),
          "16k-e09": ("p16k", 32)}
# each phase's GEN keys: times=1 the decode timed per stage; the hot set policy and reset, the prompt's split, the
# route trace
KEYS = {"short-timed": "times=1", "short-static": "hot=static hot_reset=1", "short-prompt": "hot=prompt hot_reset=1",
        "short-adaptive": "hot=adaptive hot_reset=1", "16k-e08": "hot_reset=1 prefill_times=1",
        "16k-e09": "hot_reset=1 prefill_times=1", "64k": "trace=0",
        "256k": "trace=0"}
DECODE_RE = re.compile(
    r"request (\d+): decode experts: (\d+) steps, (\d+) hits and (\d+) misses \(hit rate ([0-9.]+)\), (\d+) CPU jobs "
    r"reading ([0-9.]+) MiB in ([0-9.]+) ms \(([0-9.]+) ms per step, ([0-9.]+) GB/s\), doorbell waits ([0-9.]+) ms over "
    r"(\d+) doorbells, plan ([0-9.]+) ms; ([0-9.]+) ms per step")
DECODE_KEYS = ("steps", "hits", "misses", "hit_rate", "cpu_jobs", "cpu_mib", "cpu_ms", "cpu_ms_per_step", "cpu_gbs",
               "doorbell_wait_ms", "doorbells", "plan_ms", "ms_per_step")
# dsv41-engine's hot set line of a request (print_decode) and its prompt's split (print_prefill)
HOT_RE = re.compile(
    r"request (\d+): decode hot set (\w+)(?: \(static: no spare slots\))?: (\d+) swaps \(([0-9.]+) MiB copied in "
    r"([0-9.]+) ms on the copy stream, (\d+) batches, (\d+) deferred\), (\d+) checks; hit rate per layer ([0-9.,]+); "
    r"reset before it (\d+) swaps \(([0-9.]+) MiB\) in ([0-9.]+) ms")
PREFILL_RE = re.compile(r"request (\d+): prefill split: (.*)$", re.M)
PREFILL_FIELDS = {"total_ms": r"calls, ([0-9.]+) ms \(timed\)", "attention_ms": r"front attention ([0-9.]+)",
                  "dense_ms": r", dense ([0-9.]+)", "routed_experts_ms": r"routed experts ([0-9.]+);",
                  "experts_compute_ms": r"compute ([0-9.]+) \(gather", "copy_gb": r"copies ([0-9.]+) GB",
                  "copy_ms": r"GB in ([0-9.]+) ms", "copy_hidden_ms": r"of which ([0-9.]+) hidden",
                  "plan_gap_ms": r"plan gaps ([0-9.]+)", "stall_ms": r"stalls ([0-9.]+)",
                  "fetch_ms": r"routing fetch ([0-9.]+)", "head_ms": r"head ([0-9.]+) ms"}
# print_times' stages (dsv41-engine.cpp), in its order
STAGES = ("embed", "engram", "attention", "hc and norms", "router", "shared expert", "routed experts")
# the phases a turn2 may follow: turn 1 of the 16K conversation, 32 generated
TURN1 = ("16k", "16k-e09")
LOAD_S = 1800   # start to READY: 246 GiB from the local disk into VRAM and pinned RAM


def read_ids(d, name):
    with open(os.path.join(d, f"{name}.ids")) as f:
        return [int(x) for x in f.read().split()]


def proc_io(pid):
    """(read_bytes, majflt) of a process, None for what cannot be read."""
    rb = mf = None
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    rb = int(line.split()[1])
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/stat") as f:
            mf = int(f.read().rsplit(")", 1)[1].split()[9])
    except (OSError, IndexError, ValueError):
        pass
    return rb, mf


def memory(pid):
    out = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        used, total = (int(x) for x in r.stdout.split("\n")[0].split(","))
        out["vram_used_mib"], out["vram_total_mib"] = used, total
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_mib"] = int(line.split()[1]) // 1024
    except OSError:
        pass
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(p) as f:
                out["cgroup_usage_gib"] = round(int(f.read().strip()) / (1 << 30), 2)
                break
        except (OSError, ValueError):
            continue
    return out


class Engine:
    def __init__(self, exe, args, log_path):
        self.log_path, self.n_req = log_path, 0
        self.log = open(log_path, "w")
        self.p = subprocess.Popen([exe, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                  text=True, bufsize=1)
        self.lines = Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.lines.put((time.monotonic(), line.rstrip("\n")))
        self.lines.put((time.monotonic(), None))

    def next(self, until):
        """The next stdout line and its time, None at the engine's end; raises TimeoutError past `until`."""
        left = until - time.monotonic()
        if left <= 0:
            raise TimeoutError
        try:
            return self.lines.get(timeout=left)
        except Empty:
            raise TimeoutError from None

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def stop(self):
        try:
            self.send("QUIT")
            self.p.wait(timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()
            self.p.wait()
        self.log.close()


def decode_lines(log_path, n_req):
    """The decode's routed experts and, when timed, its per-stage ms per step from the engine's log of request n_req
    (the lines it writes before the request's DONE)."""
    out = {}
    try:
        with open(log_path, errors="replace") as f:
            text = f.read()
    except OSError:
        return out
    for m in DECODE_RE.finditer(text):
        if int(m.group(1)) == n_req:
            out["decode_experts"] = {k: (float(v) if "." in v else int(v)) for k, v in zip(DECODE_KEYS, m.groups()[1:])}
    for m in HOT_RE.finditer(text):
        if int(m.group(1)) == n_req:
            g = m.groups()
            out["hot"] = {"policy": g[1], "swaps": int(g[2]), "swap_mib": float(g[3]), "swap_copy_ms": float(g[4]),
                          "batches": int(g[5]), "deferred": int(g[6]), "checks": int(g[7]),
                          "layer_hit_rate": [float(x) for x in g[8].split(",")], "reset_swaps": int(g[9]),
                          "reset_mib": float(g[10]), "reset_ms": float(g[11])}
    for m in PREFILL_RE.finditer(text):
        if int(m.group(1)) == n_req:
            f = {"line": m.group(2)}
            for k, rx in PREFILL_FIELDS.items():
                v = re.search(rx, m.group(2))
                f[k] = float(v.group(1)) if v else None
            out["prefill_split"] = f
    m = re.search(rf"^times decode \(request {n_req}\): (\d+) calls, \d+ tokens, ([0-9.]+) ms total: (.*)$", text, re.M)
    if m:
        calls, rest = int(m.group(1)), m.group(3)
        per = {}
        for stage in STAGES:
            v = re.search(rf"(?:^|, ){re.escape(stage)} ([0-9.]+)", rest)
            if v:
                per[stage] = round(float(v.group(1)) / calls, 3) if calls else None
        head = re.search(r"head ([0-9.]+) ms", rest)
        if head:
            per["head"] = round(float(head.group(1)) / calls, 3) if calls else None
        out["timed"] = {"calls": calls, "total_ms": float(m.group(2)), "ms_per_step": per, "line": m.group(0)}
    return out


def run_phase(eng, name, ids, max_new, until):
    io0 = proc_io(eng.p.pid)
    t0 = time.monotonic()
    eng.n_req += 1
    keys = KEYS.get(name, "")
    eng.send(f"GEN {max_new}{' ' + keys if keys else ''} {','.join(map(str, ids))}")
    r = {"phase": name, "start_unix": round(time.time(), 3), "prompt_tokens": len(ids), "max_new": max_new, "pp": [],
         "tokens": []}
    while True:
        t, line = eng.next(until)
        if line is None:
            raise RuntimeError("the engine ended during the request")
        if line.startswith("ERR"):
            raise RuntimeError(f"the engine refused the request: {line}")
        if line.startswith("RESUME "):
            r["resume"] = int(line.split()[1])
        elif line.startswith("PP "):
            f = line.split()
            r["pp"].append({"read": int(f[1]), "total": int(f[2]), "ms": float(f[3])})
        elif line.startswith("T "):
            if not r["tokens"]:
                r["ttft_client_ms"] = round((t - t0) * 1e3, 1)
            r["tokens"].append(int(line.split()[1]))
        elif line.startswith("DONE "):
            f = line.split()
            r.update({"generated": int(f[1]), "prompt_ms": float(f[3]), "decode_ms": float(f[4]), "finish": f[5],
                      "reused": int(f[8]), "wall_ms": round((t - t0) * 1e3, 1), "end_unix": round(time.time(), 3)})
            break
    io1 = proc_io(eng.p.pid)
    read = r["prompt_tokens"] - r["reused"]
    r["prompt_tok_s"] = round(read / (r["prompt_ms"] / 1e3), 1) if r["prompt_ms"] > 0 else None
    steps = r["generated"] - 1
    r["decode_steps"] = steps
    r["decode_tok_s"] = round(steps / (r["decode_ms"] / 1e3), 2) if steps > 0 and r["decode_ms"] > 0 else None
    if io0[0] is not None and io1[0] is not None:
        r["storage_read_mib"] = round((io1[0] - io0[0]) / (1 << 20), 1)
    if io0[1] is not None and io1[1] is not None:
        r["major_faults"] = io1[1] - io0[1]
    r.update(memory(eng.p.pid))
    r.update(decode_lines(eng.log_path, eng.n_req))
    hs = r.get("hot")
    if hs and "ttft_client_ms" in r:
        r["ttft_client_no_reset_ms"] = round(r["ttft_client_ms"] - hs["reset_ms"], 1)
    return r


def engine_lines(log_path):
    """The engine's own lines worth keeping: device memory, the expert tiers, the split's measured rates."""
    keep = []
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                if line.startswith(("device memory:", "experts:", "dsv41-engine serve:")) and "request" not in line:
                    keep.append(line.rstrip())
    except OSError:
        pass
    rates = {}
    for line in keep:
        m = re.search(r"PCIe ([0-9.]+) GB/s into the ring, CPU ([0-9.]+) GB/s", line)
        if m:
            rates = {"pcie_ring_gbs": float(m.group(1)), "cpu_gbs": float(m.group(2))}
        m = re.search(r"cold (\d+) \(the model's mapping\)", line)
        if m:
            rates["cold_experts"] = int(m.group(1))
    return keep, rates


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--engine", required=True)
    ap.add_argument("--engine-args", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--phases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, default=0, help="unix seconds: every phase must end by then")
    a = ap.parse_args()
    phases = a.phases.split(",")
    for p in phases:
        if p not in PHASES:
            sys.exit(f"e07_bench.py: unknown phase {p} (one of {', '.join(PHASES)})")
    if "turn2" in phases and (phases.index("turn2") == 0 or phases[phases.index("turn2") - 1] not in TURN1):
        sys.exit("e07_bench.py: turn2 follows 16k or 16k-e09 (its second turn resumes the first's live state)")
    deadline = time.monotonic() + (a.deadline - time.time() if a.deadline else 1e9)
    args = shlex.split(a.engine_args) + ["--serve", "--eos-ids", "none"]
    log_path = os.path.splitext(a.out)[0] + "-engine.log"
    res = {"engine": a.engine, "args": args, "prompts": os.path.abspath(a.prompts), "phases": [], "ok": False}

    def save():
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)

    t0 = time.monotonic()
    eng = Engine(a.engine, args, log_path)
    res["pid"] = eng.p.pid
    ok = True
    try:
        until = min(deadline, t0 + LOAD_S)
        while True:
            _, line = eng.next(until)
            if line is None:
                raise RuntimeError(f"the engine exited ({eng.p.wait()}) before READY: see {log_path}")
            if line.startswith("INFO "):
                res["info"] = line[5:]
            if line.startswith("READY "):
                break
        res["load_s"] = round(time.monotonic() - t0, 1)
        res["engine_lines"], res["rates"] = engine_lines(log_path)
        res.update({"after_load": memory(eng.p.pid)})
        print(f"e07_bench.py: loaded in {res['load_s']} s; {res['rates']}", flush=True)
        save()
        last16 = None
        for name in phases:
            src, max_new = PHASES[name]
            ids = read_ids(a.prompts, src) if src else last16 + read_ids(a.prompts, "turn2")
            r = run_phase(eng, name, ids, max_new, deadline)
            if name in TURN1:
                last16 = ids + r["tokens"]
            if name == "turn2" and not r.get("resume"):
                r["error"] = "turn 2 was read cold: no RESUME of the live state"
                ok = False
            res["phases"].append(r)
            save()
            print(f"e07_bench.py: {name}: {r['prompt_tokens']} prompt ids ({r['reused']} reused), prompt "
                  f"{r['prompt_ms']:.0f} ms ({r['prompt_tok_s']} tok/s), first token at {r.get('ttft_client_ms')} ms "
                  f"(client), {r['generated']} generated, decode {r['decode_tok_s']} tok/s over {r['decode_steps']} "
                  f"steps, storage read {r.get('storage_read_mib')} MiB, {r.get('major_faults')} major faults, VRAM "
                  f"{r.get('vram_used_mib')} MiB", flush=True)
            x = r.get("decode_experts")
            if x:
                print(f"e07_bench.py: {name}: decode experts: hit rate {x['hit_rate']:.4f} ({x['hits']} hits, {x['misses']} "
                      f"misses over {x['steps']} steps), CPU {x['cpu_ms_per_step']:.3f} ms per step at {x['cpu_gbs']} "
                      f"GB/s, doorbell waits {x['doorbell_wait_ms'] / max(x['steps'], 1):.3f} ms per step, "
                      f"{x['ms_per_step']:.3f} ms per step", flush=True)
            if r.get("timed"):
                print(f"e07_bench.py: {name}: ms per step by stage {r['timed']['ms_per_step']}", flush=True)
            hs = r.get("hot")
            if hs:
                st = max(r["decode_steps"], 1)
                print(f"e07_bench.py: {name}: hot set {hs['policy']}: {hs['swaps'] / st:.2f} swaps and "
                      f"{hs['swap_mib'] / st:.1f} MiB per step, {hs['deferred']} deferred boundaries, reset before it "
                      f"{hs['reset_swaps']} swaps in {hs['reset_ms']:.0f} ms; first token at {r.get('ttft_client_ms')} "
                      f"ms (client), {r.get('ttft_client_no_reset_ms')} ms without the reset", flush=True)
            if r.get("prefill_split"):
                print(f"e07_bench.py: {name}: prefill split {r['prefill_split']['line']}", flush=True)
        # the engine still answers after the last phase (the 256K one must not leave it broken)
        alive = run_phase(eng, "alive", read_ids(a.prompts, "short")[:16], 2, deadline)
        res["alive_after"] = alive["generated"] == 2
        ok = ok and res["alive_after"]
    except TimeoutError:
        res["error"] = "out of time (the session's deadline or the load's limit)"
        ok = False
    except RuntimeError as e:
        res["error"] = str(e)
        ok = False
    finally:
        eng.stop()
        res["engine_exit"] = eng.p.returncode
        res["ok"] = ok
        save()
    print(f"e07_bench.py: {'done' if ok else 'FAIL: ' + res.get('error', 'see ' + a.out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
