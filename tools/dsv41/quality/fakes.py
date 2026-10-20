#!/usr/bin/env python3
"""tools/dsv41/quality/fakes.py - stand-ins for the three systems the quality runs talk to, for the GPU-free tests
(the quality tooling's tests and the API run's dry test): no model, no GPU, no money.

  fakes.py --serve --tokenizer-gguf V [--record F] [--ctx N]
      the engine: dsv41-engine --serve's stdin protocol (tools/dsv41/engine/dsv41-engine.cpp, its header), so
      the real serve/server.py drives it through StrataEngine (the config's "exe" is this file): READY, then per GEN
      line PP, one T per token and DONE; every GEN's keys and prompt ids go to F (JSON lines). A --hot-set file is
      opened relative to the engine's own cwd, as the real engine's load_hot_set opens it, and the engine exits 1
      before READY with the real one's message when it cannot (a qengine session's engine once ran from src/ and
      did not find out/gate/hot-set.json). Its log (stderr) has the real one's
      lines that tools/dsv41/pod/e07_bench.py and qdiag_report.py parse, with fixed numbers: the init rates and the
      load, then per request its prompt and decode, its decode experts (hit rate 0.9, 12.5 ms of doorbell waits and
      15 ms per step, or the env FAKE_ENGINE_STEP_MS: q02-test.sh's slow machine for the speed check) and its hot set
      (the GEN's hot= policy, adaptive by default; 2 swaps per step, none static). Sampling: a GEN at temperature above 0 samples as dsv41-engine does in outline: max_new tokens drawn from a
      small pool of word tokens by random.Random(seed) (the seed=N key, else a random one), no end of reply, so the
      same seed gives the same tokens; the request's line ends in "greedy" or the sampler's parameters and seed, in
      the real one's words; a penalty is refused with the real one's ERR; --log-ids writes each request's
      prompt_token_ids and generated_token_ids lines. FAKE_ENGINE_NONDET=P (q03-test.sh's nondeterministic engine):
      every greedy request of a prompt already answered gets token P of its reply replaced, so the determinism check
      sees the replies diverge at P, whatever the mode. The deterministic mode as the real
      engine's, on unless --deterministic off or the GEN's det=0, named on each request's line and its hot set line;
      FAKE_ENGINE_PLACEMENT=P (q03-test.sh's placement-dependent engine): a request in the mode off starts from the
      slab the requests before it left (as the real engine's adaptive hot set), so every reply after the first has
      token P replaced by one that depends on that history; in the mode a request starts from the hot set's slab and
      its reply is the first one's. In the mode the hot set line's reset before the request is 8 swaps in 5.0 ms (off
      the mode none), outside the request line's first token at 1.0 ms, as the real engine resets before session.run.
      The prefix cache of session.hpp (one sequence: ids that strictly extend the
      live tokens, the last prompt and every generated token but the last, resume them; ids that strictly extend
      the last prompt restore its end; else a cold read), with its RESUME line and the reused count on DONE; the env
      FAKE_ENGINE_PROMPT_MS_PER_TOKEN and FAKE_ENGINE_DECODE_MS_PER_TOKEN (default 0) are wall-clock delays per
      prompt token read and per generated token, and the request's line, PP and DONE give those times (the
      rehearsal's FIFO queue is then predictable: c01-test.sh). Several conversations:
      the cache is session.hpp's N checkpoints (--checkpoints N, default 4): a request resumes from the longest
      strict prefix among the live tokens and the checkpoints (ties: the live tokens, then the checkpoint used last),
      its prompt's end goes to the checkpoint with the same ids, else the first one it supersedes (their ids a
      prefix of its own; all of those dropped), else a free slot, else the least recently used one (evicted); INFO
      checkpoints=N, the cost line at load (805.0 MiB per slot at --ctx 262144, the real layout's) and each request's
      "checkpoints:" line in the real one's words (the bytes 3518 per token, the real 16K checkpoint's 55.1 MiB, the
      copies at 25 GB/s; "in place on the device" for the checkpoint the last request saved while the live tokens are
      at most a chunk of 16384 past it). The deterministic mode's reset runs after the request: a
      request in the mode resets before it only when the request before it was in the mode off (8 swaps in 5.0 ms),
      and a request after one in the mode reports that one's idle reset (8 swaps in 5.0 ms) and a wait of 5.0 ms for
      it (the next request taken to arrive at once, as the rehearsals send them). FAKE_ENGINE_IDLE_RESET_MS=R
      (c02-test.sh's ccache rehearsal): the idle reset in wall-clock time instead, R ms from
      a request's DONE in the mode; the next GEN that arrives before its end waits for the rest, which its hot set
      line reports (a request after the reset's end waits 0.0 ms). --hot-homes all (the
      gate's choice when the host's RAM holds the slab's host homes): the real engine's load lines for the homes
      (72702.2 MiB, the real placement's) and the reset layer by layer, FAKE_ENGINE_RESET_LAYERS (default 40, the real
      model's) layers each done R / layers ms after the last: the next request's prompt read layer by layer at its
      per-token delay, each layer starting once its restore is done, so it waits only where its read overtakes the
      restore; its hot set line's wait is the device's (none on the host) and the next line splits it as the real
      engine's "idle reset layer by layer (host homes)" line does. --hot-homes N (the gate's
      partial host homes): the first N slab experts of --hot-set's layers that swap homed in layer order (the real
      engine's home order; their RAM the full placement's 72702.2 MiB per expert), the layers homed in full restored
      layer by layer as above, the others reset in rounds from DONE for R x (their share of the 40 layers), which a
      GEN arriving earlier waits for on the host (the real engine runs them before it reads the next request); the
      real engine's "idle reset in rounds" line, and with any homes its "copies down left" line (this request's
      swaps copied down in the same share, the reset's one round swap). Batch slots: --batch N
      (--batch-ctx C, default 32768) speaks the real engine's batch protocol: BGEN read as a GEN of one token, then
      BADM <slot> 1 and the slot's BT lines in steps run while no stdin line waits, each step of B slots taking
      FAKE_ENGINE_DECODE_MS_PER_TOKEN + FAKE_ENGINE_BATCH_ROW_MS x (B - 1) of wall-clock time, BDONE with rep=0|1, a
      BSTOP ending its slot before the next step, a request past C answered alone with BADM <slot> 0 (with
      alone=<slots held> why=<the reason> and the real engine's "ran alone" line; its "runs alone" line
      and ALONE <slot> <slots held> why=<the reason> before it runs), rep=1 for a request in the deterministic mode and
      rep=0 in the mode off, beside other slots or not (an older real
      engine said rep=0 for every request decided beside other slots, reads beside them included); a STOP now reaches
      a running request between its tokens (a reader thread, as the real engine's); the real engine's batch lines in
      its log (each slot's end, the batch's totals and the "batch steps, all" line with the fake's step time, a read
      beside decoding slots). Long prompts: --interleave-chunk K (default --chunk, 4096 when absent; 0: the prompt read
      whole) reads
      a prompt that arrives while slots decode in chunks of K tokens, K x FAKE_ENGINE_PROMPT_MS_PER_TOKEN each, a PP
      line after each and one batch step after each but the last, a BSTOP that came meanwhile ending its slot before
      that step and a STOP ending the read (cancel); INFO interleave_chunk=K and the real engine's "read beside" line
      with the chunks, each chunk's read time, the steps between them and the longest hold. FAKE_ENGINE_HANG_AT=N and
      FAKE_ENGINE_DIE_AT=N (c04-test.sh's broken engines): at the N-th request line (GEN or BGEN) the engine stops
      printing for good (alive, its stdin unread) or exits 3. The promotion fix:
      session.hpp's unfed token: a live resume whose one token left is the token the last request emitted and did not
      feed (the server's promotion: a STOP on the solo path, then a BGEN of the prompt and the tokens) is fed by
      decode, the real engine's line "its 1 token read was fed by decode" in its log, and the reply goes on as the
      request alone; FAKE_ENGINE_PROMOTION_PREFILL=P is the engine before the fix (that token read by prefill(T=1)):
      the reply's token P after it differs from the request alone's (what the RTX cdet session measured at tokens 67
      and 91). --cpu-numerics fork (the default, as the real engine's): the real engine's load
      line, and no batch slot ends rep=1; --cpu-numerics gpu: the slots' rep= as above.
  fakes.py llama-server --tokenizer-gguf V [--record F] --port P [--prefill-assistant | --no-prefill-assistant] ...
      the fork's llama-server: /health, /apply-template and /v1/chat/completions, the prompt rendered from V's chat
      template as the fork renders it (with its default --prefill-assistant, a last assistant message is continued:
      the messages before it with the generation prompt, then its text; thinking on unless chat_template_kwargs
      say "enable_thinking": false, the fork's default), and the reply's content of a continued message starts with
      that message's text, as the real fork's does (fork_prompt_test.py measured it); every request and its prompt go
      to F. Each reply logs the real one's print_timing lines (prompt eval, eval, total) on stdout, its eval at 52.5 ms
      per token (the stock fork's baseline on the A100 SXM4, 52.2-52.9) or the env FAKE_FORK_MS_PER_TOKEN (q02-test.sh's
      slow machine for the fork's speed check). Other llama-server flags are taken and ignored.
  fakes.py deepseek-api --key-file K --record F --port P
      DeepSeek's API: /chat/completions and /beta/chat/completions (chat prefix completion: "prefix": true on the
      last message, refused outside /beta as the API does), the bearer key checked against K, the usage with the
      API's cache fields; every request body goes to F (never the key).

Without --tokenizer-gguf the engine and llama-server read the tokenizer from --model / -m (a vocab GGUF in the
place of the model: tools/dsv41/quality/q01-test.sh runs pod_server.sh, the pod's own launcher, on them that way).

Each one's reply: the 7-digit number of a needle prompt ("The pass key is NNNNNNN", needle.py), the numbers 1 to 2000
for a prompt that asks to count ("Count from 1 to": tools/dsv41/quality/det_sampling_check.py's speed prompt, longer
than its 256 tokens), else a fixed short text that continues a code block, cut at the request's stop strings and its
max tokens.
"""
from __future__ import annotations

import json
import os
import queue
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

REPLY = "    return 42\n```\nThe answer is 42."
# a checkpoint slot at --ctx 262144 on the real layout (kv_checkpoint_bytes), MiB
SLOT_MIB = 805.0
NEEDLE = re.compile(r"The pass key is (\d{7})")
COUNT = "Count from 1 to"
# the fake engine's sampled tokens are drawn from these words' tokens
SAMPLED_WORDS = " the quick brown fox jumps over a lazy dog while seven small birds sing in the old garden"


def reply_for(prompt: str, stops: list[str] | None = None) -> str:
    m = NEEDLE.findall(prompt)
    text = m[-1] if m else ", ".join(str(i) for i in range(1, 2001)) if COUNT in prompt else REPLY
    for s in stops or []:
        if s and s in text:
            text = text[:text.index(s)]
    return text


def arg(argv: list[str], name: str, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


class Recorder:
    def __init__(self, path: str | None):
        self.path, self.lock = path, threading.Lock()

    def write(self, rec: dict):
        if self.path is None:
            return
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def engine(argv: list[str]) -> int:
    import strata_tokenizer as ST
    from gguf_reader import GGUFFile
    gguf = arg(argv, "--tokenizer-gguf") or arg(argv, "--model")
    tok = ST.Tokenizer.from_gguf(gguf)
    eos = int(GGUFFile(Path(gguf)).metadata["tokenizer.ggml.eos_token_id"])
    ctx, rec = int(arg(argv, "--ctx", "262144")), Recorder(arg(argv, "--record"))
    # the wall-clock delays, per prompt token read and per generated token (0: none, as before)
    prompt_ms = float(os.environ.get("FAKE_ENGINE_PROMPT_MS_PER_TOKEN") or 0)
    decode_ms = float(os.environ.get("FAKE_ENGINE_DECODE_MS_PER_TOKEN") or 0)
    step_ms = float(os.environ.get("FAKE_ENGINE_STEP_MS") or decode_ms or 15)
    # the idle reset in wall-clock ms (unset: 5.0 ms, the next request taken to arrive at once)
    idle_ms = os.environ.get("FAKE_ENGINE_IDLE_RESET_MS")
    idle_ms = float(idle_ms) if idle_ms else None
    idle_end = idle_start = rounds_end = 0.0
    # the host homes and the reset layer by layer, all of them or N (partial)
    hot_homes = arg(argv, "--hot-homes", "spare")
    homes, held = hot_homes != "spare", None if hot_homes in ("all", "spare") else int(hot_homes)
    reset_layers = int(os.environ.get("FAKE_ENGINE_RESET_LAYERS") or 40)
    nondet = os.environ.get("FAKE_ENGINE_NONDET")
    nondet = int(nondet) if nondet else None
    placement = os.environ.get("FAKE_ENGINE_PLACEMENT")
    placement = int(placement) if placement else None
    # FAKE_ENGINE_BATCH_DRIFT=P (c05-test.sh's engine before batch determinism): a slot's token at position P differs once the
    # slot has decoded a step beside another slot (the placement its neighbours moved), whatever the mode
    drift = os.environ.get("FAKE_ENGINE_BATCH_DRIFT")
    drift = int(drift) if drift else None
    # the engine before the promotion fix (a promoted request's last token read by prefill)
    promo_prefill = os.environ.get("FAKE_ENGINE_PROMOTION_PREFILL")
    promo_prefill = int(promo_prefill) if promo_prefill else None
    # the fork's numerics by default, as dsv41-engine's
    tier_free = arg(argv, "--cpu-numerics", "fork") == "gpu"
    det_default = arg(argv, "--deterministic", "on") == "on"
    slab = 0   # the mode off: the history the slab carries (0: the hot set's, as loaded)
    # session.hpp's cache: the live tokens and the checkpoints (each [ids, last use] or None)
    n_slots = int(arg(argv, "--checkpoints", "4"))
    live, slots, clock, last_saved, evictions = [], [None] * n_slots, 0, -1, 0
    unfed = None   # the token the last request emitted and did not feed (session.hpp)
    dirty, idle = False, None   # the slab moved since the hot set's (a request in the mode off); the last idle reset
    log_ids, answered = "--log-ids" in argv, set()
    # each greedy prompt's whole reply, so a prompt that is an earlier one plus the start of its reply (the
    # server's STOP then BGEN, or BSTOP then GEN, of a moved request) goes on with the rest of that reply, as the real
    # engine's greedy decode does
    replies: dict[tuple, list[int]] = {}
    pool = sorted(set(tok.encode(SAMPLED_WORDS)))
    hot = arg(argv, "--hot-set")
    if hot is not None:
        try:
            open(hot).close()
        except OSError:
            print(f"fakes.py: strata-moe: STRATA_MOE_HOT={hot}: cannot open the file (cwd {Path.cwd()})", file=sys.stderr,
                  flush=True)
            return 1
    # partial homes: the layers that swap (their slab experts), the ones homed in full, the cut one, and those in rounds
    swap_layers, full_layers, cut, homeable = [], 0, None, 5951
    if held is not None:
        if hot is None:
            print("fakes.py: --hot-homes N needs --hot-set (the layers the homes go to)", file=sys.stderr, flush=True)
            return 1
        swap_layers = [(x["layer"], len(x["experts"])) for x in json.load(open(hot))["layers"]
                       if 0 < len(x["experts"]) < x["n_expert"]]
        homeable, left = sum(n for _, n in swap_layers), held
        if held > homeable:
            print(f"dsv41 experts: --hot-homes {held}: the slab has {homeable} experts that can take a host home (in the "
                  "layers that swap)", file=sys.stderr, flush=True)
            return 1
        for layer, n in swap_layers:
            if left >= n:
                full_layers, left = full_layers + 1, left - n
            else:
                cut = (layer, left, n) if left else None
                break
    partial = held is not None and held < homeable
    round_layers = len(swap_layers) - full_layers if partial else 0
    homes_mib = 72702.2 * (held if held is not None else homeable) / 5951
    print("experts: prefill split for chunks of up to 2048 tokens, by the rates measured at init: PCIe 26.00 GB/s into "
          "the ring, CPU 150.00 GB/s on one token per expert + 0.0500 ms per further pair", file=sys.stderr, flush=True)
    print(f"conversation checkpoints: {n_slots} slots of {SLOT_MIB * ctx / 262144:.1f} MiB pinned host RAM each (a "
          f"checkpoint of {ctx} positions), {n_slots * SLOT_MIB * ctx / 262144:.1f} MiB in all; on the device the live KV "
          "and the in-place carry snapshot (12288.0 KiB)", file=sys.stderr, flush=True)
    if homes:
        flag = "--hot-homes all" if held is None else f"--hot-homes {held}: {held} of {homeable} slab experts"
        print(f"fixed host RAM beside the checkpoints: the hot set's host homes, {homes_mib:.1f} MiB pinned host RAM "
              f"({flag}), held once for the engine whatever the number of conversations (not part of a conversation's "
              "cost)", file=sys.stderr, flush=True)
    if partial:
        print(f"experts: hot set homes: partial host homes (--hot-homes {held}): {held} of the slab's {homeable} VRAM "
              f"experts keep a permanent pinned home ({100 * held / homeable:.1f}%), {homes_mib:.1f} MiB: every slab "
              f"expert in {full_layers} of the {len(swap_layers)} layers that swap (the first in layer order)"
              + (f", layer {cut[0]}'s last {cut[1]} of its {cut[2]} by the hot set's order" if cut else "")
              + "; there a swap copies its expert up only and the deterministic reset restores in place layer by layer on "
              "its own stream, the next request waiting per layer; in the others a homed victim is dropped, any other "
              "copied down into a spare home, and the reset runs in rounds of the spare slots, waited for before the next "
              "request", file=sys.stderr, flush=True)
    else:
        print("experts: hot set homes: " + ("host homes (--hot-homes all): every slab expert keeps a permanent pinned "
                                            "home, 72702.2 MiB; a swap copies its expert up only and drops its victim (no "
                                            "copy down, no spare homes); the deterministic reset restores in place layer "
                                            "by layer on its own stream, the next request waiting per layer" if homes else
                                            "two-way (--hot-homes spare): a swap copies its victim down into a spare home; "
                                            "the deterministic reset in rounds of the spare slots, the next request "
                                            "waiting for all of it" + (
                                                "; the deterministic mode is on (--deterministic on), so this idle reset "
                                                "is the slow one: the next request waited 1.67-1.91 s for it on the RTX "
                                                "PRO 6000 (2026-10-14, 2885-3311 swaps after each "
                                                "256-token reply); tools/dsv41/pod/e07_gate.py's gate.json hot_homes is "
                                                "the --hot-homes value this host's RAM holds" if det_default else "")),
          file=sys.stderr, flush=True)
    if not tier_free:
        print("dsv41-engine serve: the CPU experts in the fork's numerics (--cpu-numerics fork, the default): a "
              "routed expert's bits follow its tier, so no batch slot ends reproducible", file=sys.stderr, flush=True)
    print("dsv41-engine serve: loaded in 1000 ms", file=sys.stderr, flush=True)
    # the batch slots (--batch N, --batch-ctx C) and the step's time model: D (the decode delay) a step of one row,
    # FAKE_ENGINE_BATCH_ROW_MS more per further row
    n_batch, batch_ctx = int(arg(argv, "--batch", "0")), int(arg(argv, "--batch-ctx", "32768"))
    row_ms = float(os.environ.get("FAKE_ENGINE_BATCH_ROW_MS") or 0)
    # the chunk of a prompt read while slots decode (0: read whole between two steps; by default the
    # prefill chunk, --chunk, the real engine's 4096 when absent, at most --ctx); the broken engines
    prefill_chunk = min(int(arg(argv, "--chunk", "4096")), ctx)
    interleave = int(arg(argv, "--interleave-chunk", str(prefill_chunk)))
    if interleave > prefill_chunk:
        sys.exit(f"dsv41-engine: --interleave-chunk {interleave} is past the prefill chunk of {prefill_chunk} (--chunk, "
                 "at most --ctx)")
    hang_at, die_at = (int(os.environ.get(k) or 0) for k in ("FAKE_ENGINE_HANG_AT", "FAKE_ENGINE_DIE_AT"))
    print(f"INFO checkpoints={n_slots}" + (f" batch_slots={n_batch} batch_ctx={batch_ctx} slot_cache=0 "
                                           f"interleave_chunk={interleave}" if n_batch else "")
          + " engine=fake-dsv41-engine", flush=True)
    print(f"READY {ctx} stop", flush=True)
    # the stdin lines from a thread, a STOP as a flag (the real engine's reader thread): a running request sees it
    # between its tokens
    inbox, stop = queue.Queue(), threading.Event()

    def reader():
        for x in sys.stdin:
            if x.strip() == "STOP":
                stop.set()
            else:
                inbox.put(x)
        inbox.put(None)
    threading.Thread(target=reader, daemon=True).start()
    # per batch slot: its request, its tokens still to come, its max_new and produced, its mode, its start, its steps
    # (and those beside another slot), a BSTOP; the totals since the slots were last all idle
    active, totals = {}, {"steps": 0, "rows": 0, "ms": 0.0, "admissions": 0, "stalls": 0, "stall_ms": 0.0}

    def batch_step():
        for b in sorted(active):
            if active[b]["stop"]:
                batch_end(b, "cancel")
        rows = sorted(active)
        if not rows:
            return
        step_ms = decode_ms + row_ms * (len(rows) - 1)
        time.sleep(step_ms / 1000)
        totals["steps"] += 1
        totals["rows"] += len(rows)
        totals["ms"] += step_ms
        for b in rows:
            a = active[b]
            a["steps"] += 1
            a["shared_steps"] += len(rows) > 1
            a["rows"] += len(rows)
            t = a["left"].pop(0) if a["left"] else eos
            if drift is not None and a["produced"] == drift and a["shared_steps"] and t != eos:
                t = pool[(pool.index(t) + 1) % len(pool)] if t in pool else pool[0]
            a["produced"] += 1
            a["ids"].append(t)
            print(f"BT {b} {t}", flush=True)
            if t == eos:
                batch_end(b, "stop")
            elif a["produced"] >= a["max_new"]:
                batch_end(b, "length")
        if not active:
            print(f"dsv41-engine serve: batch: {totals['steps']} steps of {totals['rows'] / max(totals['steps'], 1):.2f} "
                  f"rows on average ({totals['rows']} rows) in {totals['ms']:.1f} ms, "
                  f"{totals['ms'] / max(totals['steps'], 1):.2f} ms a step, "
                  f"{totals['rows'] / max(totals['ms'] / 1000, 1e-9):.1f} tok/s over the steps; {totals['admissions']} "
                  f"admissions, {totals['stalls']} prompts read between steps holding the batch {totals['stall_ms']:.1f} ms",
                  file=sys.stderr, flush=True)
            # the real engine's breakdown line for all the steps, the fake's step time inside, nothing else
            n = max(totals["steps"], 1)
            print(f"dsv41-engine serve: batch steps, all: {totals['steps']} steps of "
                  f"{totals['rows'] / n:.2f} rows, {totals['ms'] / n:.2f} ms a step from one step's start to the next's (the "
                  f"prompts read between them apart): inside the step {totals['ms'] / n:.2f} (graph launch 0.00, doorbells "
                  "served 0.00 of which doorbell waits 0.00, CPU jobs 0.00 and their plan 0.00, tail 0.00, boundary 0.00 of "
                  "which swap waits 0.00 at 0 steps; the device's graph 0.00; 0 graphs captured again), between steps 0.000 "
                  "(the picks 0.000, the BT lines and their flush 0.000, admissions 0.000, the loop and its stdin poll "
                  "0.000); per step 0.0 routed pairs to 0.0 distinct experts, 0.0 hits (hit rate 0.0000) and 0.0 misses in "
                  "0.0 CPU jobs reading 0.0 MiB (0.00 GB/s), 0.00 swaps (0.0 MiB) published, 0.00 deferred",
                  file=sys.stderr, flush=True)
            for k in totals:
                totals[k] = 0

    def batch_end(b, finish):
        a = active.pop(b)
        rep = a["det"] and tier_free   # placement-independent bits (not with --cpu-numerics fork)
        ms = (time.monotonic() - a["t0"]) * 1000
        why = ("reproducible (the deterministic mode: its bits do not depend on the expert placement)" if rep else "not reproducible (deterministic off)" if not a["det"]
               else "not reproducible (--cpu-numerics fork: a routed expert's bits follow its tier, so its neighbours' "
               "placement)")
        print(f"dsv41-engine serve: request {a['request']}: batch slot {b}: {a['produced']} generated ({finish}) in "
              f"{ms:.1f} ms, {a['steps']} steps, {a['shared_steps']} of them beside other slots "
              f"({a['rows'] / max(a['steps'], 1):.2f} rows a step), steps stalled {a['stall_ms']:.1f} ms by other "
              f"requests' prompts; {why}", file=sys.stderr, flush=True)
        if log_ids:   # the slot's every token (the real engine's slot_token_ids line)
            print(f"dsv41-engine serve: request {a['request']}: slot_token_ids: {' '.join(map(str, a['ids']))}",
                  file=sys.stderr, flush=True)
        print(f"BDONE {b} {a['produced']} {finish} {ms:.1f} rep={1 if rep else 0}", flush=True)

    n_req = 0
    while True:
        if active:
            try:
                line = inbox.get_nowait()
            except queue.Empty:   # nothing to read: the batch's next step
                batch_step()
                continue
        else:
            line = inbox.get()
        if line is None:
            break
        f = line.split()
        if not f:
            continue
        if f[0] == "QUIT":
            break
        if f[0] == "BSTOP":
            if int(f[1]) in active:
                active[int(f[1])]["stop"] = True
            continue
        if f[0] == "BYIELD":
            continue
        admit = None
        if f[0] == "BGEN":
            b = int(f[1]) if len(f) > 1 and f[1].lstrip("-").isdigit() else -1
            if not 0 <= b < n_batch or b in active:
                print(f"ERR BGEN: no such free slot (--batch {n_batch})", flush=True)
                continue
            admit, f = b, ["GEN"] + f[2:]
        if f[0] != "GEN":
            print(f"ERR unknown command {f[0]}", flush=True)
            continue
        if n_req + 1 in (hang_at, die_at):
            print(f"fakes.py: request {n_req + 1}: FAKE_ENGINE_{'HANG' if n_req + 1 == hang_at else 'DIE'}_AT", file=sys.stderr,
                  flush=True)
            if n_req + 1 == die_at:
                return 3
            while True:   # hung: alive, silent, its stdin no longer read
                time.sleep(3600)
        max_new, keys, ids = int(f[1]), dict(k.split("=", 1) for k in f[2:-1]), [int(x) for x in f[-1].split(",")]
        penalty = next((k for k, off in (("penalty_repeat", 1), ("penalty_freq", 0), ("penalty_present", 0))
                        if float(keys.get(k, off)) != off), None)
        if penalty:
            api = {"penalty_repeat": "repetition_penalty", "penalty_freq": "frequency_penalty"}.get(penalty,
                                                                                                  "presence_penalty")
            print(f"ERR {penalty}={keys[penalty]} (the API's {api}): this engine applies no penalties", flush=True)
            continue
        if len(ids) + max_new + 8 > ctx:
            print(f"ERR the prompt ({len(ids)}) and max_new ({max_new}) exceed the context ({ctx})", flush=True)
            continue
        if keys.get("det", "1") not in ("0", "1"):
            print(f"ERR det={keys['det']}: 1 (the deterministic mode) or 0 (off)", flush=True)
            continue
        rec.write({"max_new": max_new, "keys": keys, "ids": ids} | ({"slot": admit} if admit is not None else {}))
        # a request beside decoding slots starts no reset of its own; a BGEN that fits reads its first token only
        others, t_req, beside = bool(active), time.monotonic(), len(active)
        fits = admit is not None and max_new > 1 and len(ids) + max_new - 1 <= batch_ctx
        # refused for its positions it runs alone (the engine's line and BADM keys, with fits()'s words)
        unfit = (f"{len(ids)} prompt tokens and {max_new} to generate need {len(ids) + max_new - 1} positions, a batch "
                 f"slot holds {batch_ctx}") if admit is not None and not fits and max_new > 1 else ""
        admit_max_new = max_new
        if fits:
            max_new = 1
        # run alone past a slot, said before the request runs (the real engine's line)
        if unfit:
            print(f"dsv41-engine serve: request {n_req + 1}: runs alone on the live sequence, not in batch slot {admit}: "
                  f"{unfit}; it holds {beside} decoding batch slots until its end", file=sys.stderr,
                  flush=True)
            print(f"ALONE {admit} {beside} why={unfit}", flush=True)
        stop.clear()
        # the part of the last request's idle reset still running when this GEN arrived (the layered
        # reset: none on the host, the prompt's read waits per layer below)
        # (homes and no FAKE_ENGINE_IDLE_RESET_MS: its 5.0 ms on the device, below, none on the host)
        wait_ms, arrived = 0.0 if homes else 5.0, time.monotonic()
        if idle_ms is not None:
            # partial host homes: the rounds of the layers not homed in full are the host's wait
            wait_ms = max(0.0, (rounds_end if homes else idle_end) - arrived) * 1000
            time.sleep(wait_ms / 1000)
        det = keys["det"] == "1" if "det" in keys else det_default
        temperature = float(keys.get("temperature", 0))
        if temperature > 0:
            seed = int(keys["seed"]) if "seed" in keys else random.SystemRandom().getrandbits(64)
            draw = random.Random(seed)
            out = [draw.choice(pool) for _ in range(admit_max_new)]
            sampling = (f"sampled: temperature {temperature:g}, top_k {int(keys.get('top_k', 0))}, top_p "
                        f"{float(keys.get('top_p', 1)):g}, min_p {float(keys.get('min_p', 0)):g}, seed {seed} "
                        f"({'given' if 'seed' in keys else 'from std::random_device'})")
        else:
            go_on = next((r[len(ids) - len(p):] for p, r in replies.items() if len(p) < len(ids) <= len(p) + len(r)
                          and tuple(ids[:len(p)]) == p and ids[len(p):] == r[:len(ids) - len(p)]), None)
            whole = tok.encode(reply_for(tok.decode(ids)))
            if go_on is None:
                replies.setdefault(tuple(ids), whole)
            out = (whole if go_on is None else go_on)[:admit_max_new]
            if nondet is not None and tuple(ids) in answered and nondet < len(out):
                out[nondet] = pool[0] if out[nondet] != pool[0] else pool[1]
            answered.add(tuple(ids))
            sampling = "greedy"
        # the placement the request starts from: the hot set's in the mode, else what the requests before left
        start = 0 if det else slab
        if placement is not None and start and placement < len(out):
            moved = 1000 + start % 100000       # a token of its own per history
            out[placement] = moved if out[placement] != moved else moved + 1
        slab = (slab * 31 + sum(ids) + len(ids)) % 1000003 or 1
        # session.hpp's rule: the longest strict prefix among the live tokens and the checkpoints, the live tokens
        # first at equal lengths, then the checkpoint used last; else the prompt is read from position 0
        def extends(held):
            return bool(held) and len(ids) > len(held) and ids[:len(held)] == held
        kind, reuse, slot = ("live", len(live), -1) if extends(live) else ("cold", 0, -1)
        for k in sorted((k for k in range(n_slots) if slots[k]), key=lambda k: -slots[k][1]):
            if extends(slots[k][0]) and len(slots[k][0]) > reuse:
                kind, reuse, slot = "prompt-end", len(slots[k][0]), k
        ckpt = []
        if slot >= 0:
            clock += 1
            slots[slot][1] = clock
            mib = 3518 * reuse / 2**20
            in_place = slot == last_saved and len(live) - reuse <= 16384 and live[:reuse] == slots[slot][0]
            ckpt.append(f"restored {slot} ({reuse} tokens, in place on the device, 0.05 ms)" if in_place else
                        f"restored {slot} ({reuse} tokens, {mib:.1f} MiB host to device, {mib * 2**20 / 25e6:.2f} ms)")
        else:
            ckpt.append(f"none restored ({kind})")
        if reuse:
            print(f"RESUME {reuse}", flush=True)
        read = len(ids) - reuse
        # the last request's unfed token as the one token left: decode (the fix), or prefill(T=1) before
        # it, which moves the reply's token promo_prefill after it (greedy: the request alone's reply goes on)
        fed_unfed = kind == "live" and read == 1 and unfed is not None and ids[-1] == unfed
        unfed = None
        if fed_unfed and promo_prefill is not None and temperature <= 0 and promo_prefill < len(out):
            out[promo_prefill] = pool[0] if out[promo_prefill] != pool[0] else pool[1]
            fed_unfed = False
        # the read: layer by layer behind a layered reset still running (each layer once its restore is done), else
        # at once; the layers' waits on the device
        dev_wait = layer_max = 0.0
        layer_at, layers_waited = -1, 0
        inter = others and interleave > 0   # read in chunks with steps between them (below)
        if homes and idle and idle_ms is not None and not inter:
            t, per = time.monotonic(), read * prompt_ms / 1000 / reset_layers
            for l in range(reset_layers):
                w = max(0.0, idle_start + idle_ms / 1000 * (l + 1) / reset_layers - t)
                if w > 0:
                    dev_wait, layers_waited = dev_wait + w, layers_waited + 1
                    if w > layer_max:
                        layer_max, layer_at = w, l
                t += w + per
            time.sleep(max(0.0, t - time.monotonic()))
            dev_wait, layer_max = dev_wait * 1000, layer_max * 1000
        elif homes and idle and not inter:
            dev_wait, layer_max, layer_at, layers_waited = 5.0, 5.0, 0, 1
            time.sleep(read * prompt_ms / 1000)
        elif not inter:
            time.sleep(read * prompt_ms / 1000)
        read_ms, read_rate = (read * prompt_ms, 1000 / prompt_ms) if prompt_ms else (1.0, 1000.0)
        # beside decoding slots, chunks of `interleave` with a step after each but the last (BSTOPs first)
        chunks = steps_between = 0
        hold_max, mark, cancelled = 0.0, t_req, False
        chunk_ms = []   # each chunk's read time, from the last chunk's or step's end (the first from the request's start)
        if inter:
            pos = reuse
            while pos < len(ids):
                n = min(interleave, len(ids) - pos)
                time.sleep(n * prompt_ms / 1000)
                chunk_ms.append((time.monotonic() - mark) * 1000)
                pos, chunks = pos + n, chunks + 1
                print(f"PP {pos} {len(ids)} {(time.monotonic() - t_req) * 1000:.1f} {read_rate:.1f}", flush=True)
                if pos == len(ids):
                    break
                if stop.is_set():
                    cancelled = True
                    break
                with inbox.mutex:
                    kept = [x for x in inbox.queue if not (x and x.startswith("BSTOP "))]
                    for x in inbox.queue:
                        if x and x.startswith("BSTOP ") and int(x.split()[1]) in active:
                            active[int(x.split()[1])]["stop"] = True
                    inbox.queue.clear()
                    inbox.queue.extend(kept)
                held_now = (time.monotonic() - mark) * 1000
                hold_max = max(hold_max, held_now)
                totals["stalls"] += 1
                totals["stall_ms"] += held_now
                for a in active.values():
                    a["stall_ms"] += held_now
                steps_between += 1
                batch_step()
                mark = time.monotonic()
        else:
            chunk_ms.append((time.monotonic() - t_req) * 1000)
            print(f"PP {len(ids)} {len(ids)} {read_ms:.1f} {read_rate:.1f}", flush=True)
        full = out + [eos] if fits else []   # an admission: the slot goes on with the rest of the reply
        reply = [] if cancelled else out[:max_new] + ([eos] if len(out) < max_new else [])
        sent = 0
        for t in reply:
            if stop.is_set():   # a STOP between two tokens (the real engine's Session), the reply ends here
                break
            time.sleep(decode_ms / 1000)
            print(f"T {t}", flush=True)
            sent += 1
        finish = "cancel" if cancelled or sent < len(reply) else "stop" if len(out) < max_new else "length"
        reply = reply[:sent]
        if reply:
            unfed = reply[-1]
        n_req, gen = n_req + 1, len(reply)
        gen_ms = gen * decode_ms if decode_ms else 1.0
        first_ms = read * prompt_ms + decode_ms if prompt_ms or decode_ms else 1.0
        # every generated token but the last is fed: the live state after the request; the prompt's end saved
        live = ids + reply[:-1]
        save, superseded, evicted = next((k for k in range(n_slots) if slots[k] and slots[k][0] == ids), -1), 0, None
        if save < 0:
            for k in range(n_slots):
                if slots[k] and extends(slots[k][0]):
                    slots[k], superseded = None, superseded + 1
                    save = k if save < 0 else save
        if save < 0 and n_slots:
            save = next((k for k in range(n_slots) if not slots[k]), -1)
            if save < 0:
                save = min(range(n_slots), key=lambda k: slots[k][1])
                evicted, evictions = (save, len(slots[save][0])), evictions + 1
        if keys.get("ckpt", "1") == "0" or not n_slots:
            save = -1
        last_saved = save
        if save >= 0:
            clock += 1
            slots[save] = [list(ids), clock]
            mib = 3518 * len(ids) / 2**20
            ckpt.append(f"saved to {save} ({len(ids)} tokens, {mib:.1f} MiB device to host, {mib * 2**20 / 25e6:.2f} ms)")
        else:
            ckpt.append("none saved")
        ckpt.append(f"{superseded} superseded")
        if evicted:
            ckpt.append(f"evicted {evicted[0]} ({evicted[1]} tokens, the least recently used)")
        ckpt_line = ", ".join(ckpt) + f"; {sum(1 for x in slots if x)} of {n_slots} slots held, {evictions} evictions"
        snapshot_n = len(ids) if save >= 0 else 0
        # the mode's reset at the start only after a request in the mode off; the idle reset after one in it
        reset_before = det and dirty and not others
        # the reset's 8 swaps: all in place with homes (12.2 MiB up each), one of them in rounds with partial homes
        # (12.2 up and 12.2 down), all in rounds without (two-way)
        reset_ms, in_place = 5.0 if idle_ms is None else idle_ms, 7 if partial else 8
        idle_text = (f"idle reset after the last request 8 swaps ({(109.8 if partial else 97.6) if homes else 195.2:.1f} "
                     f"MiB) in {reset_ms:.1f} ms, waited {wait_ms + dev_wait:.1f} ms" if idle else
                     "idle reset after the last request 0 swaps (0.0 MiB) in 0.0 ms, waited 0.0 ms")
        layered = [f"request {n_req}: idle reset layer by layer (host homes): {in_place} swaps copied up only "
                   f"({12.2 * in_place:.1f} MiB, none down), {reset_ms:.1f} ms on the device, queued in 0.4 ms; this "
                   f"request waited {wait_ms:.1f} ms on the host and {dev_wait:.1f} ms on the device over {layers_waited} "
                   f"layers (at most {layer_max:.1f} ms, layer {layer_at})"] if homes and idle else []
        if partial and idle:
            layered.append(f"request {n_req}: idle reset in rounds (partial host homes, {round_layers} layers): 1 swaps "
                           f"(12.2 MiB up, 12.2 MiB down: 1 victims copied down) in {reset_ms * round_layers / reset_layers:.1f} "
                           "ms on the host, before this request was read")
        if not others:
            idle, dirty = det, not det
        # the engine's log lines of the request, written before its DONE as the real engine writes them
        steps, policy = max(gen - 1, 0), keys.get("hot", "adaptive")
        swaps = 0 if policy == "static" else 2 * steps
        lines = [f"request {n_req}: {len(ids)} prompt tokens, {kind} resume of {reuse}, {read} read in {read_ms:.1f} "
                 f"ms, {gen} generated ({finish}) in {gen_ms:.1f} ms, first token at {first_ms:.1f} ms; {len(live)} "
                 f"tokens held, snapshot at {snapshot_n}; {'deterministic' if det else 'deterministic off'}; "
                 f"{sampling}"] + ([
                 f"request {n_req}: its 1 token read was fed by decode: the token the last request emitted and did not "
                 "feed, as the request alone feeds it"] if fed_unfed else []) + [
                 f"request {n_req}: checkpoints: {ckpt_line}",
                 f"request {n_req}: decode experts: {steps} steps, {216 * steps} hits and {24 * steps} misses (hit rate "
                 f"0.9000), {24 * steps} CPU jobs reading {12.0 * steps:.1f} MiB in {1.5 * steps:.2f} ms (1.500 ms per "
                 f"step, 100.00 GB/s), doorbell waits {12.5 * steps:.2f} ms over {40 * steps} doorbells, plan 0.10 ms; "
                 f"{step_ms:.3f} ms per step",
                 f"request {n_req}: decode hot set {policy}: {swaps} swaps ({24.4 * swaps:.1f} MiB copied in "
                 f"{0.6 * swaps:.2f} ms on the copy stream, 1 batches, 0 deferred), 0 checks; hit rate per layer "
                 f"{','.join(['0.900'] * 40)}; reset before it {8 if reset_before else 0} swaps "
                 f"({195.2 if reset_before else 0.0:.1f} MiB) in {5.0 if reset_before else 0.0:.1f} ms; "
                 + (f"deterministic: {steps // 4} boundaries waited for their swaps' copies, {0.5 * (steps // 4):.2f} ms"
                    if det else "deterministic off (det=0)") + f"; {idle_text}"] + layered
        # with any host homes, the copies down left (the layers not homed in full: their share of the
        # swaps, the reset's round swap)
        if homes:
            downs = swaps * round_layers // max(len(swap_layers), 1)
            lines.append(f"request {n_req}: copies down left (host homes held {held if held is not None else homeable} "
                         f"of {homeable} slab experts): {downs} of this request's {swaps} swaps copied their victim down "
                         f"({12.2 * downs:.1f} MiB), {1 if partial and idle else 0} of the {8 if idle else 0} swaps of the "
                         f"idle reset before it ({12.2 if partial and idle else 0.0:.1f} MiB)")
        if log_ids:
            lines += [f"request {n_req}: prompt_token_ids: {' '.join(map(str, ids))}",
                      f"request {n_req}: generated_token_ids: {' '.join(map(str, out + ([eos] if finish == 'stop' else [])))}"]
        for x in lines:
            print(f"dsv41-engine serve: {x}", file=sys.stderr, flush=True)
        if others:
            held = (time.monotonic() - mark) * 1000
            hold_max = max(hold_max, held)
            totals["stalls"] += 1
            totals["stall_ms"] += held
            for a in active.values():
                a["stall_ms"] += held
            how = (f"{chunks} chunks of {interleave}" if interleave else
                   "1 chunks of the engine's --chunk (not interleaved)")
            print(f"dsv41-engine serve: request {n_req}: read beside {beside} decoding batch slots, {read} tokens in {how} "
                  f"with {steps_between} batch steps between them, a chunk read "
                  f"in {sum(chunk_ms) / len(chunk_ms):.1f} ms on average and {max(chunk_ms):.1f} at most, "
                  f"{(time.monotonic() - t_req) * 1000:.1f} ms from its start to its {'last' if gen > 1 else 'first'} "
                  f"token, the slots held at most {hold_max:.1f} ms at once; "
                  + ("not reproducible (deterministic off)" if not det else
                     "reproducible: its expert placement followed theirs, which no longer moves its bits" if tier_free else
                     "not reproducible (--cpu-numerics fork: its bits follow the placement)"),
                  file=sys.stderr, flush=True)
        if unfit:
            held = (time.monotonic() - t_req) * 1000 if beside else 0.0
            print(f"dsv41-engine serve: request {n_req}: ran alone on the live sequence, not in batch slot {admit}: {unfit} "
                  f"(C04 audit 2); {gen} generated ({finish}), holding {beside} decoding batch slots for {held:.1f} ms"
                  + ("; not reproducible (--cpu-numerics fork: its bits follow the placement)"
                     if det and not tier_free and beside else ""),
                  file=sys.stderr, flush=True)
        print(f"DONE {gen} {len(ids)} {read_ms:.1f} {gen_ms:.1f} {finish} 0 0 {reuse}", flush=True)
        if admit is not None:
            if fits and finish == "length" and gen == 1:
                active[admit] = {"request": n_req, "left": full[1:admit_max_new], "max_new": admit_max_new,
                                 "produced": 1, "det": det, "t0": t_req, "steps": 0, "ids": [full[0]],
                                 "shared_steps": 0, "rows": 0, "stall_ms": 0.0, "stop": False}
                totals["admissions"] += 1
                print(f"BADM {admit} 1", flush=True)
            else:
                # as the engine: with --cpu-numerics fork a request beside decoding slots ends rep=0
                rep = det and (tier_free or beside == 0)
                print(f"BADM {admit} 0 rep={1 if rep else 0}" + (f" alone={beside} why={unfit}" if unfit else ""),
                      flush=True)
        if idle_ms is not None and (active or others):
            continue      # the idle reset runs once nothing decodes
        if idle_ms is not None:
            idle_start = time.monotonic()
            idle_end = idle_start + idle_ms / 1000 if det else 0.0
            rounds_end = idle_start + idle_ms / 1000 * round_layers / reset_layers if det else 0.0
    return 0


def serve(port: int, routes: dict) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"status": "ok"})
            self.reply(404, {"error": {"message": "not found"}})

        def do_POST(self):
            fn = routes.get(self.path)
            if fn is None:
                return self.reply(404, {"error": {"message": "not found"}})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self.reply(*fn(body, self.headers))

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def chat_response(model: str, content: str, prompt_tokens: int, completion_tokens: int, usage_extra=None) -> dict:
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens, **(usage_extra or {})}
    return {"id": "fake", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": usage}


def llama_server(argv: list[str]) -> int:
    import strata_tokenizer as ST
    from gguf_reader import GGUFFile
    from serve.frontend import ChatTemplate
    gguf = arg(argv, "--tokenizer-gguf") or arg(argv, "-m")
    md = GGUFFile(Path(gguf)).metadata
    tok, tpl = ST.Tokenizer.from_gguf(gguf), ChatTemplate(source=md["tokenizer.chat_template"])
    bos = md["tokenizer.ggml.tokens"][int(md["tokenizer.ggml.bos_token_id"])]
    prefill = "--no-prefill-assistant" not in argv
    port, rec = int(arg(argv, "--port", "8080")), Recorder(arg(argv, "--record"))
    ms_tok = float(os.environ.get("FAKE_FORK_MS_PER_TOKEN") or 52.5)
    task = iter(range(0, 1 << 30, 7))   # the real server's task ids grow by more than one per reply
    log_lock = threading.Lock()

    def print_timing(n_prompt: int, n_gen: int) -> None:
        """llama-server's per-reply timing lines (tools/server/server-context.cpp print_timing), as the real log has
        them (the 2026-10-11 qfork session's out/quality-server.log)."""
        t, n_gen, pp = next(task), max(n_gen, 1), 8.7   # the prompt at 115 tok/s, the qfork plan's prompt rate
        pre = f"0.00.000.000 I slot print_timing: id  0 | task {t} |"
        with log_lock:
            print(f"{pre} prompt eval time = {n_prompt * pp:10.2f} ms / {n_prompt:6d} tokens ({pp:8.2f} ms per "
                  f"token, {1000 / pp:8.2f} tokens per second)", flush=True)
            print(f"{pre}        eval time = {n_gen * ms_tok:10.2f} ms / {n_gen:6d} tokens ({ms_tok:8.2f} ms per "
                  f"token, {1000 / ms_tok:8.2f} tokens per second)", flush=True)
            print(f"{pre}       total time = {n_prompt * pp + n_gen * ms_tok:10.2f} ms / {n_prompt + n_gen:6d} "
                  "tokens", flush=True)

    def render(body: dict) -> str:
        msgs = body["messages"]
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        if prefill and msgs and msgs[-1].get("role") == "assistant":
            head = tpl.render(msgs[:-1], add_generation_prompt=True, bos_token=bos, enable_thinking=thinking)
            return head + (msgs[-1].get("content") or "")
        return tpl.render(msgs, add_generation_prompt=True, bos_token=bos, enable_thinking=thinking)

    def apply_template(body, headers):
        return 200, {"prompt": render(body)}

    def chat(body, headers):
        prompt = render(body)
        text = reply_for(prompt, body.get("stop"))
        msgs = body["messages"]
        cont = prefill and msgs and msgs[-1].get("role") == "assistant"
        rec.write({"body": body, "prompt": prompt})
        n_prompt, n_gen = len(tok.encode(prompt, parse_special=True)), len(tok.encode(text))
        print_timing(n_prompt, n_gen)
        return 200, chat_response(body.get("model") or "fake", (msgs[-1].get("content") or "") + text if cont else text,
                                  n_prompt, n_gen)

    print(f"main: server is listening on http://127.0.0.1:{port} - starting the main loop", flush=True)
    serve(port, {"/apply-template": apply_template, "/v1/chat/completions": chat})
    return 0


def deepseek_api(argv: list[str]) -> int:
    key = Path(arg(argv, "--key-file")).read_text().strip()
    port, rec = int(arg(argv, "--port", "8200")), Recorder(arg(argv, "--record"))

    def chat(beta: bool):
        def handle(body, headers):
            if headers.get("Authorization") != f"Bearer {key}":
                return 401, {"error": {"message": "Authentication Fails (fake)"}}
            msgs = body.get("messages") or []
            if msgs and msgs[-1].get("prefix") and not beta:
                return 400, {"error": {"message": "prefix completion needs the beta base URL (fake)"}}
            rec.write({"beta": beta, "body": body})
            prompt = "\n".join(str(m.get("content") or "") for m in msgs)
            text = reply_for(prompt, body.get("stop"))
            n = max(1, len(prompt) // 4)
            return 200, chat_response("deepseek-flash", text, n, max(1, len(text) // 4),
                                      {"prompt_cache_hit_tokens": n // 2, "prompt_cache_miss_tokens": n - n // 2})
        return handle

    serve(port, {"/chat/completions": chat(False), "/beta/chat/completions": chat(True)})
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--serve":
        return engine(argv[1:])
    if argv and argv[0] == "llama-server":
        return llama_server(argv[1:])
    if argv and argv[0] == "deepseek-api":
        return deepseek_api(argv[1:])
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
