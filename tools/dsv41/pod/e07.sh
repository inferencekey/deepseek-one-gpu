#!/usr/bin/env bash
# The real-model pod session: the real
# DeepSeek-V4.1-Flash Q2_K on one RTX PRO 6000 Blackwell (Server, else Workstation Edition; the main target), else
# one A100 80GB (SXM4, else PCIe), with at least 250 GB of host RAM. With NETWORK_VOLUME set, each GPU is tried first
# in VOLUME_DC with that RunPod network volume (holding the model) mounted at /workspace, which is only ever read (the
# copy reads it, nothing writes there, and the pod's last step, also run on an early stop, checks its listing is
# unchanged), then in any data center without a network volume, SECURE then COMMUNITY, where the pod
# downloads the model from Hugging Face (HF_REPO at HF_REVISION, the 7 Q2_K shards, every file's
# size and sha256 checked against the HF API's; pod/hf_model.py, pod/hf-download.sh). The session is
# devgpu.sh's (pod/pod-session.sh: price and RAM checks before and after the create, the watchdog at the cap, SSH
# and the CUDA probe, the payload, the detached run polled over fresh SSH connections, the pull before the delete on
# every way out); the pod runs pod/e07-run.sh (its header lists the steps: the gate, the model copy, correctness
# against the stock fork's oracle dump in both reader modes forced and free, the isolation runs (the CPU
# reference of layers 0 and 1 in three numerics, the engine's MoE on the oracle's input, the MMQ prefill when the
# reference says so), tokenizer parity, the 16K TTFT, the second turn and short decode with its experts' split, the
# server, then 64K and 256K; E07_PLAN=cycle2, the default, skips tokenizer, server, 64K and 256K; E07_PLAN=e08 runs
# the adaptive hot-set bench and the server only; E07_PLAN=e09 the prefill bench and a two-turn chat), and back on
# the local machine pod/e07_report.py writes the session sheet into
# $SHEET_ROOT/<date>-e07a/ (-e08/, -e09/ for those plans; README.md with the tables, the logs).
# Run from the local machine (the repo's root), with your RunPod API key in RUNPOD_API_KEY:
#
#   RUNPOD_API_KEY=... E07_PLAN=cycle2 E07_READERS=dense MAX_USD_PER_HR=1.95 bash tools/dsv41/pod/e07.sh
#   RUNPOD_API_KEY=... E07_PLAN=e08 MAX_USD_PER_HR=1.95 bash tools/dsv41/pod/e07.sh    # the adaptive hot-set session
#   RUNPOD_API_KEY=... E07_PLAN=e09 MAX_USD_PER_HR=1.95 bash tools/dsv41/pod/e07.sh    # the prefill session
#   RUNPOD_API_KEY=... E07_PLAN=e07b MAX_USD_PER_HR=1.95 bash tools/dsv41/pod/e07.sh   # the RTX PRO 6000 session
#   RUNPOD_API_KEY=... E07_PLAN=qfork Q_TASKS=gsm8k_cot_llama,humaneval_instruct HF_CLOUDS= NETWORK_VOLUME=<id> \
#     bash tools/dsv41/pod/e07.sh                                                     # qfork on a network volume
#   RUNPOD_API_KEY=... E07_PLAN=qengine bash tools/dsv41/pod/e07.sh    # likewise qdiag, qdet, cbase, ccache
#   RUNPOD_API_KEY=... E07_PLAN=cbatch BUILD_TAG=-c03 bash tools/dsv41/pod/e07.sh   # likewise clong, cbatch-clong
#   RUNPOD_API_KEY=... E07_PLAN=cdet BUILD_TAG=-c04 MAX_SESSION_USD=2.90 NETWORK_VOLUME=<id> bash tools/dsv41/pod/e07.sh
#   RUNPOD_API_KEY=... E07_PLAN=cfinal BUILD_TAG=-c04 bash tools/dsv41/pod/e07.sh
#   bash tools/dsv41/pod/e07.sh --dry-run      # anywhere: the whole plan, no RunPod request, no key
#   OUT=<dir> bash tools/dsv41/pod/e07.sh --payload-only   # the session's payloads in OUT as the run builds them, no key
#
# E07_PLAN=e08 (<= 3 USD at MAX_USD_PER_HR 1.95, any data center with the model from
# HF): the gate (the hot set packed beside 2 spare slots per layer, the spare homes pinned), the model, the dry run,
# bench-e08 on one engine load (480 decode steps at short context per hot-set policy from the same slab, the 16K
# prompt's per-stage split and 480 steps after it, the timed split, 64K and 256K with the default policy; the routing
# trace of all but 64K and 256K), the trace gzipped for the pull, and the server's 3-chat quality check.
#
# E07_PLAN=e09 (one session <= 1.5 USD, so MAX_SESSION_USD defaults to 1.50 for it; at
# MAX_USD_PER_HR 1.95 any data center with the model from HF): the gate with a prefetch region of 128 experts (every
# engine run gets --prefetch-experts 128), the model, the dry run, bench-e09 on one engine load (the 16K prompt twice
# with its per-stage split, short-adaptive's 480 decode steps between, the 256K prompt once), and two-turn (a two-turn
# chat through the server, turn 2 resumed: cached tokens and the route trace's resume=1, its decode reported). The
# dry run exits 2 when the HF plan's estimate costs more than 1.50 USD at MAX_USD_PER_HR.
#
# E07_PLAN=e07b (the closing measurement on the RTX PRO 6000, first in GPU_TYPES;
# MAX_USD_PER_HR defaults to 1.95 for it, the session cap is 3.00 USD): the gate with e09's prefetch region of 128
# experts, the model, the dry run, then the cheapest goal-gating checks first: two-turn through the server (turn 1 the
# 16K prompt's text, e07/turns.json from pod/e07_prompts.py turns, at ctx 20480; turn 2 a short follow-up, streamed,
# its first token at the client) with the HTTP check on the same server load (non-streaming and streaming
# /v1/chat/completions), bench-e07b on one engine load (16k-e09, turn2, short-adaptive, 64k, 256k), then the
# free-greedy oracle parity (dense readers) and the isolated MoE parity, reported, not gating. Each pod step gets its
# own timeout, min(2 x its estimate, the time left: the estimates go up as e07/estimates-volume and -hf); the HF
# download is estimated at HF_GBS (default 0.115 GB/s, the measured 2026-10-10 rate: the pessimistic case) and its
# guard stops the session after 3 min when the measured rate cannot finish the download and the rest of the plan by
# the deadline ("download too slow: X GB/s"). env records the host's PCIe link (generation and width, the host's
# maximum when the driver reports it). The dry run exits 2 when the HF plan costs more than 3.00 USD at MAX_USD_PER_HR.
#
# E07_PLAN=qfork and qengine (the Part Q quality plans):
# the quality of the stock fork and of the engine on the fixed-seed lm-eval subsets, through
# tools/dsv41/quality/api_proxy.py, lm-eval on the pod against localhost (its venv installed in the background since
# setup, while the model downloads and loads). qfork (an A100 80GB, SXM4 then PCIe, under MAX_USD_PER_HR 1.95):
# no gate, no engine; the fork's llama-server (quality/pod_server.sh: t04a.sh's stock
# flags, -np Q_FORK_NP --kv-unified --ctx-size 32768 --prefill-assistant, --threads min(16, cpus) and --threads-batch
# min(32, cpus), cpus nproc capped by the cgroup's CPU quota on the pod), the HumanEval continuation checked on its /apply-template and on 3 replies (the session
# stops there when it fails), then each task of Q_TASKS (llamacpp backend, NUM_CONCURRENT Q_FORK_NP) and each needle of
# Q_NEEDLES (default 16k). Q_FORK_NP is the fork's slots, default 1 (the 2026-10-11
# qfork session's -np 4 with --threads-batch 32 on an 18 vCPU host decoded at 1.6-2.3 tok/s per slot), shipped as
# e07/fork-np. qengine (the RTX PRO 6000, MAX_USD_PER_HR 2.60,
# the standing limit for that card): the gate and the dry run as e07b's (prefetch 128), serve/server.py on the
# engine (strata backend, 2 concurrent), the same continuation check on 3 replies, the tasks, then Q_NEEDLES
# (default 16k 64k 256k). Both: MIN_RAM_GB 256 (the target host), each step's timeout min(2 x its estimate, the time
# left), the HF download's guard, every step's estimate from the 2026-10-06 API run's tokens per task
# (quality/q_plan.py) at Q_PROMPT_TOK_S, Q_LONG_PROMPT_TOK_S (the needles) and Q_DECODE_TOK_S, one request at a time
# (defaults: qfork 115, 115 and 20 tok/s, the fork's measured rates, its decode the stock baseline's 18.9-19.2 on an
# A100; qengine 288.7 (the e07b session's 128-token prompt), 1866.8 (its 256K prompt) and 60 (its 62.8 short, 56.1 at
# 64K and 256K)); the dry run prints them and exits 2 when a model path the offers can take costs more than MAX_SESSION_USD
# (3.00) at MAX_USD_PER_HR. Q_TASKS (default the three tasks) and Q_NEEDLES split a plan into sessions by task (the
# rule when one session would cost more than 3 USD: qengine is one session, qfork
# two with the model from the volume, Q_TASKS=gsm8k_cot_llama,humaneval_instruct with the 16K needle then
# Q_TASKS=ifeval Q_NEEDLES=, and three from HF, gsm8k_cot_llama, humaneval_instruct with the 16K needle, ifeval; the
# whole qfork plan's dry run names them). Their HF download is planned at HF_GBS 0.385. The needle prompts are built locally
# (quality/needle_check.py build, QUALITY_NEEDLE default $BUILD_ROOT/quality/needle.json); the session sheet is
# SHEET_ROOT/<date>-qfork/ or -qengine/ with the quality results (quality/q_report.py), and the run's lm-eval output
# comes back in out/quality/ (analyze.py reads it). Both the dry run and the run (into e07.log) list the payload's
# sources (q_payload_sources: HEAD and the last commit of serve/server.py and the quality tools, the subsets' and the
# needles' sha256, each tarball's commit and its engine's or llama-server's sha256 with the sheets that ran that
# engine) and refuse a working tree whose shipped paths differ from HEAD, a HEAD without the continuation (2c43875)
# or the echo cut (9ce8dc5), stale needle prompts, a qengine engine built before a change to src/, include/ or
# CMakeLists.txt, and a fork tarball that is not the fork repo's HEAD (the quality pre-flight).
#
# E07_PLAN=qdiag (one qengine session decoded at ~15 tok/s under lm-eval against the e07b bench's 62.8 tok/s
# on the same card model; one diagnostic session on the RTX PRO 6000 at MAX_USD_PER_HR 2.60, MAX_SESSION_USD
# 1.50 by default): qengine's gate, payload checks, model paths, HF guard and per-step timeouts, then bench-qdiag
# (e07_bench.py short-static,short-adaptive on one engine load, no server: the static against the adaptive hot set),
# engine-server, quality-venv and continuation as qengine's, lm-eval-gsm8k-c1 and lm-eval-gsm8k-c2 (the first 20
# gsm8k_cot_llama items of the fixed-seed subsets, e07/subsets-qdiag.json, at NUM_CONCURRENT 1 then 2, into
# out/quality/c1 and c2) and server-stop; no IFEval, HumanEval nor needles. Its estimates come from measured numbers
# (the slow qengine session: 4.1 s per 1080-token GSM8K prompt, ~100 generated tokens at the worst 15 tok/s; the e07b
# session: the load 78 s).
# qengine and qdiag run pod/pod-samplers.sh for the whole session (the GPU's clocks, power, temperature, utilization,
# PCIe generation and throttle reasons every 1 s; the host's load average, cgroup CPU limit and throttling and top
# processes every 2 s), and after the pull e07_report.py runs pod/qdiag_report.py, which writes out/qdiag.md and
# qdiag.json (per phase: the decode, doorbell waits, CPU path, hit rate, swaps, prefill, SM clock, power, throttle
# reasons, cgroup throttled time and load); the sheet SHEET_ROOT/<date>-qdiag/ carries it, the qengine sheet the GPU
# clock summary.
# qengine and qdiag run speed-check right after the continuation (one qengine machine
# decoded 4-5x slower than qdiag's and spent 2.889 USD for no result): pod/speed_check.py, the median ms per decode
# step of the continuation's requests against Q_SPEED_MAX_MS (default 30; qdiag's continuation 20.2, the slow machine's 44.2,
# RTX PRO 6000 WS), shipped as e07/speed-max-ms; the pod gets the created pod's machineId as POD_MACHINE
# (pod-session.sh). Above it qengine stops with "stop: engine too slow on this machine: X ms per decode step (machine
# M)" in out/steps.log, the server stopped and the pod pulled and deleted as on any other stop; qdiag carries on.
# qfork runs speed-check there too: pod/speed_check.py --fork on out/quality-server.log,
# the median ms per generated token of llama-server's print_timing lines of the continuation's replies against
# Q_FORK_MAX_MS (default 80, 12.5 tok/s; t04a's stock baseline 52.2-52.9 on the A100 SXM4, the 2026-10-11 session's
# lm-eval replies at -np 4 334-1002), shipped as e07/fork-max-ms; above it the session stops with "stop: fork too slow
# on this machine: X ms per token (machine M)", the same way.
#
# E07_PLAN=qdet (items 7 and 8: greedy determinism and seeded sampling): the RTX PRO 6000 at MAX_USD_PER_HR 2.60, MAX_SESSION_USD 2.50, MAX_PODS 3, the
# model from HF (NETWORK_VOLUME defaults to none for it), qengine's gate, prefetch 128, payload checks and HF guard,
# then engine-server (serve/server.py on dsv41-engine as qengine runs it, plus --log-ids), det-sampling
# (tools/dsv41/quality/det_sampling_check.py over HTTP: item 7, prompt A greedy cold, again, prompt B, A again, the
# same ids or the first divergent position; item 8, seeds 1234 twice and 1235 at 0.7 / 0.95 / 40, the 256-token reply
# for the hand check, decode ms per token at 0.7 against greedy, 3 runs of 256 each, within 5%, frequency_penalty 0.5
# answered 400: in the engine's deterministic mode, then items 7 and 8's seeds
# again with the mode off and the mode's decode cost, prompt S greedy on and off 3 runs each; out/det-sampling.json)
# and server-stop; no lm-eval, no needles. Its estimates are the 2026-10-13
# qengine session's measured rates (sheet 2026-10-13-qengine, RTX PRO 6000): the model download step 309 s (264.5 GB
# at 0.924 GB/s, so HF_GBS defaults to 0.924 for it, plus the sha256 check), the engine's load ~6 min (planned
# with margin; that session's engine-server step took 110 s) and ~55 tok/s of decode under lm-eval; each step's timeout is
# 2 x its estimate. The sheet is SHEET_ROOT/<date>-qdet/.
#
# E07_PLAN=cbase (item 9: the concurrency baseline):
# today's FIFO server under N concurrent chat users, measurement only, on the RTX PRO 6000 only (no A100 fallback),
# MAX_USD_PER_HR 2.60, MAX_SESSION_USD 2.00, MAX_PODS 3, the model from HF (NETWORK_VOLUME defaults to none; HF_GBS
# 0.25, the worst measured rate), qengine's gate, prefetch 128, payload checks and HF guard, then engine-server
# (serve/server.py on dsv41-engine as qengine runs it, its deterministic mode on: the default), load-c<N> for each N of
# CBASE_USERS (default "1 2 4 8": tools/dsv41/quality/chat_load.py run, N users with their own 16K conversations, 2
# turns of 256-token replies each, streamed, into out/cbase/load-c<N>.json) and server-stop. The conversations are built
# locally (chat_load.py build, CBASE_CONVERSATIONS default $BUILD_ROOT/quality/conversations-16k.json) and must be
# what chat_load.py at HEAD builds. Its estimates are the worst measured rates on that card: the engine's load ~6 min
# (the 2026-10-13 qengine session), the 16K prompt at 3571.5 tok/s (the e09 session of 2026-10-10, RTX PRO 6000 WS: 4.59 s),
# the deterministic mode's reset 2.0 s per request (0.6-2.0 s), decode 35.4 tok/s (that session's
# 31 steps after the 16K prompt; qengine's ~55 under lm-eval), every request read whole (no reuse), one at a time;
# each step's timeout is 2 x its estimate. The sheet is SHEET_ROOT/<date>-cbase/ (chat_load.py's table by N).
#
# E07_PLAN=ccache (item 10: several cached conversations): the engine's conversation checkpoints (4, its
# default) and the deterministic mode's reset after each request, measured as cbase is (the RTX PRO 6000 only, 2.60
# USD/h, a 2.30 USD cap (the pod's life follows it, 53 min at 2.60 USD/h), 3 pods, the
# qualification first, the model from HF at HF_GBS 0.25, qengine's gate and prefetch 128) with the engine-server's
# engine also writing every request's ids (--log-ids, for det-sampling), then load-alt<N> (chat_load.py run
# --alternate: CCACHE_ALT_USERS users, default 4, taking CCACHE_ALT_TURNS turns each, default 3, each request sent when the one before it ended, so nobody queues: item 10's turn 2+ first token <= 2 s,
# judged on this strict run, each request's wait for the deterministic mode's idle reset from the engine's log),
# load-alt<N>-think<S> (the same users and turns with CCACHE_THINK_S, default 5, seconds between a reply's end and the
# next user's turn: informational, the first token once the reset has had time to end),
# load-c<N> for each N of CCACHE_USERS (default "1 2 4": cbase's run again, the before/after table against the cbase
# session of 2026-10-13), det-sampling (det_sampling_check.py, items 7 and 8 once: the mode still holds) and
# server-stop, planned at 60 s (the engine's shutdown took 49-52 s). The conversations are cbase's.
# Estimates as cbase's (every request read whole at 14.2 s, the reset 2.0 s in each; the alternating runs' turns 2+
# resumed, at most 512 new tokens read), det-sampling as qdet's; each step's timeout is 2 x its estimate. The dry run
# also prints each model path's plan with no turn resumed (every alternating request read whole), its worst case, and
# refuses one over the cap. The sheet is SHEET_ROOT/<date>-ccache/ (the session's real cost, the item 10 table, the
# runs by N, the engine's cost per conversation and every request's checkpoint).
#
# E07_PLAN=cbatch (item 11: batched decode): batched decode, measured as ccache is (the RTX PRO 6000 only,
# 2.60 USD/h, MAX_SESSION_USD 3.00, the session cap, the plan about 2, 3 pods, the qualification first, the model from
# HF at HF_GBS 0.25, prefetch 128), with two gates: gate (dsv41-engine --batch CBATCH_SLOTS, default 8, of
# CBATCH_CTX positions, default 69632, the deployment's (the deepseek-v4.1 profile's batch_ctx: item 11 is
# measured with it, not the earlier 32768), and CBATCH_CHECKPOINTS conversation checkpoints, default 8:
# every user of N = 8 keeps its prefix) and gate-off (the same without the batch: today's slab, for the single user's before figure).
# Then engine-server-off (the server with batching off: qengine's FIFO server on gate-off's placement and the
# checkpoints), load-short-off (chat_load.py, 1 user of CBATCH_SHORT, a short conversation built locally:
# chat_load.py build --users 1 --context 1024, CBATCH_SHORT_TURNS turns of 256-token replies, default 4 (the
# baseline from at least 4 requests); item 11's single user at short context, its before figure),
# server-stop-off, engine-server (the server's "parallel" as --batch on the engine, gate's placement,
# --log-ids), load-short (the same user with batching on: within 5% of the off run and >= 30 tok/s), load-c<N> for each
# N of CBATCH_USERS (default "4 8": cbase's runs, N users together decoding at once: per-user and aggregate decode
# tok/s, the end-to-end figures: the user-visible ones with the prompt reads included; 8 reported), steady-c<N>
# (chat_load.py steady, CBATCH_STEADY_USERS users, default 4, their
# prompts read, then a window of CBATCH_STEADY_WINDOW_S seconds, default 15, with every user decoding and exactly N rows
# a step, CBATCH_STEADY_REPEATS times, default 3, each reply CBATCH_STEADY_MAX_TOKENS long, default 1024: item 11's
# verdict, aggregate >= 2 x the single user with batching off and each >= 20 tok/s), steady-c<N> for each N of
# CBATCH_STEADY_REPORT (default none here, 8 in cbatch-clong: the same with N users
# decoding at once, aggregate and per user reported, not a bar), det-sampling (items 7
# and 8 alone, which still decode alone: the solo path) and server-stop. Estimates as cbase's for the load steps (a
# request read whole at 14.2 s, one at a time: the batch's worst case, no speed-up at all), the engine's load ~6 min
# twice, the stops at 60 s; each step's timeout 2 x its estimate. The sheet is SHEET_ROOT/<date>-cbatch/ (the item 11
# table, the runs by N, the batch's own figures from the engine's log, the gate's slots and slab).
#
# E07_PLAN=clong (items 12 and 13: long prompts and robustness): no user frozen by a long prompt, and the server
# robust, measured as cbatch is (the RTX PRO 6000 only, 2.60 USD/h, MAX_SESSION_USD 3.00, 3 pods, the qualification
# first, the model from HF at HF_GBS 0.25, prefetch 128), with the gate of CLONG_SLOTS batch slots (default 8) of
# CLONG_CTX positions (default 69632: the 64K prompt and its reply fit a slot, so it decodes beside the others instead
# of alone) and CLONG_CHECKPOINTS checkpoints (default 8). Then engine-server (the server's engine with those slots,
# its default interleave chunk, its --chunk 16384, --log-ids; the deepseek-v4.1 profile's queue_max 8,
# its deadline and its watchdog), load-long (chat_load.py long on CLONG_CONVERSATIONS, built locally: chat_load.py build --users
# CLONG_USERS --long-context 65536; the 64K user alone, then CLONG_USERS users, default 4, with replies of
# CLONG_MAX_TOKENS tokens, default 1024, and the 64K user's request 2 s after each has its first token: each user's
# largest gap between two tokens, item 12's <= 5 s, and the 64K TTFT against its solo run), robust (robust_check.py:
# item 13's HTTP checks on the real engine, streamed and not, two non-streamed requests batched, a streamed and a
# non-streamed client hanging up and its slot going to the waiting request, the 8 slots and 8 waiting answering the
# next one 503 with Retry-After; no max_tokens: a request without max_tokens beside 4
# decoding users takes a slot and no user's gap passes 5 s, and one past a slot says it ran alone) and server-stop. Estimates: the 64K prompt alone 25 s (measured ~20-25
# s on the RTX PRO 6000), each 16K user read at the e09 session's 3571.5 tok/s, the long read again in chunks of 16384 (5 for the
# 65565-token prompt) at a planned 4.65 s each (the warm tier through the ring at 26.12 GB/s; the time is the host's:
# 3.2-3.3 s beside 4 users on the RTX PRO 6000 Server of 2026-10-15, the e09 session's card 4.59 s, the e07b session's slower
# Workstation card 7.68 s, within the step's 2 x timeout) with a 50 ms step between two, the users' replies at 20 tok/s each (item 11's
# floor), robust's 3 min, the engine's load
# ~6 min, its stop 60 s; each step's timeout 2 x its estimate. The sheet is SHEET_ROOT/<date>-clong/ (item 12's table
# and verdict, item 13's checks, the interleaved reads from the engine's log with each chunk's read time and the
# slots' longest hold, the gate).
#
# E07_PLAN=cbatch-clong: cbatch's steps less the ones measured
# on 2026-10-14 (load-c<N> and det-sampling): its two gates, its two servers, the single user with batching off and on
# (the baselines) and steady-c<N>; then clong's (gate-long, engine-server-long, load-long, robust, server-stop-long) in
# one session, so one model download and qualification serve both; each item keeps its own table
# and verdict on the sheet SHEET_ROOT/<date>-cbatch-clong/. The dry run also prints the plan's latest end: the step
# that ends latest when the steps before it take their estimates and it runs to its whole timeout, within the pod's
# minutes, and the plan's USD under the session's cap at HF_GBS 0.25. Last, with CBATCH_SPARE > 0 (default 0: the plan with 4 is past the pod's deadline, above), a second cbatch
# server: gate-spare (gate's batch with --hot-spare CBATCH_SPARE, e.g. 4, the deployment's 2 spare slab slots per
# layer and 2 more), engine-server-spare on its placement with --hot-spare CBATCH_SPARE, load-short-spare (the single
# user with batching on), steady-c<N>-spare and server-stop-spare, with cbatch's estimates. The sheet judges item 11 on
# each server against the same baseline (the single user with batching off on the default config).
#
# E07_PLAN=cdet (item 14, batched replies identical to alone, and item 3 batched): cbatch's session (the RTX PRO 6000 only, 2.60 USD/h, MAX_SESSION_USD 3.00 by default, the dry run asked at
# 2.90, 3 pods, the qualification first, prefetch 128, one gate, the batched server; the model from the network volume
# only: NETWORK_VOLUME must be set (mounted read-only as for every plan), and HF_CLOUDS defaults to none, as the HF plan at HF_GBS 0.25 takes 72.6 min, about 3.15 USD, past the pod's deadline at 2.90)
# less load-c<N> and the batching-off server: load-short (the single user, the speed bar: within 5% of the measured
# 61.55 tok/s and >= 30), ab-gpu (the same user with the deterministic mode off: the speed A/B's CPU numerics gpu arm)
# and steady-c4 (aggregate >= 2 x load-short's single user, the batching-off server's place; CBATCH_STEADY_REPEATS 2),
# then det-batch (quality/det_batch_check.py: item 14, 4 identical temperature-0 requests sent together, non-streamed
# and streamed, each the tokens of the same request alone, and 4 different ones together each its own alone run's,
# every request in the deterministic mode, each chain's verdict printed with the promoted ones marked, each reply's
# "strata_reproducible" checked against its own chain: true only for a chain shown equal), det-sampling (items 7 and
# 8 still pass), quality-venv and continuation (qengine's), and item 3 batched: lm-eval through quality/api_proxy.py
# (strata backend) at NUM_CONCURRENT CDET_CONCURRENT, 4, in the deterministic mode, the fixed-seed subsets, ifeval
# first (its timeout has the other tasks after it), then gsm8k_cot_llama and humaneval_instruct, into out/quality;
# then server-stop, and last the A/B's fork arm: engine-server-fork (the same server with --cpu-numerics fork), ab-fork
# (the same user, the mode off) and server-stop-fork. Estimates from measured numbers where there are some: each
# server's load 2.5 min (85-140 s in the cbatch and cdet sessions), det-sampling 3 min (455 s in the 2026-10-17 cdet
# session, 300 of them its penalty request's wait after the engine's ERR, fixed in serve 09b390e), each lm-eval task
# as it ran on the batch slots in the cdet session (ifeval 1042 s, gsm8k 459 s; humaneval, cut at its deadline, the
# engine quality run's 359 s on the FIFO server, the other two having taken 0.97 and 0.86 x that run's: not the 0.5 x the plan assumed before),
# det-batch from the rates of cbase (the 2.0 s reset, 256 tokens at 35.4 tok/s alone, 20 tok/s a user batched). The
# sheet is SHEET_ROOT/<date>-cdet/ (item 14 with every chain, items 7 and 8, the speed bars and the A/B, item 3's table
# and item 4 against the engine quality run with the API fresh run reported, e07_report.py). The engine's default
# CPU numerics are now the fork's, so cdet's batched server says --cpu-numerics gpu (its A/B's gpu arm as it ran).
#
# E07_PLAN=cfinal (item 14 read as no quality loss from batching, item 3
# batched on all three tasks and item 4 against the engine quality run): cdet's session on the
# engine's default numerics, the fork's (the fast path), less what measured determinism: no ab-gpu, det-batch,
# det-sampling nor the fork server. The RTX PRO 6000 only, 2.60 USD/h, MAX_SESSION_USD 3.35 by default, 3 pods, the
# qualification first, the model from HF in any region by default (NETWORK_VOLUME empty, HF_CLOUDS "SECURE COMMUNITY",
# as the other Part C plans; NETWORK_VOLUME=<id> HF_CLOUDS= for a volume), one gate, the batched server, then
# load-short (item 11's single user: within 5% of the measured 61.55 tok/s and >= 30), steady-c4 (aggregate >= 2 x
# load-short's single user, every user >= 20 tok/s; cbatch's 3 repeats), quality-venv, continuation, the three lm-eval
# tasks at NUM_CONCURRENT 4 as cdet's and server-stop. Each task's estimate is its time in the fork's numerics, the engine
# quality run one at a time on the FIFO server (ifeval 1077 s, gsm8k 531 s; humaneval 359 s x 1.25, 450 s), so each
# timeout (2 x) is above the GPU's numerics' times on the batch slots on 2026-10-19 (ifeval 1521 s, gsm8k 909 s,
# humaneval 162 of 164 items at its 720 s cut). The sheet is SHEET_ROOT/<date>-cfinal/: item 11, item 3 batched, item 4
# against the engine quality run (and the API fresh run reported) and item 14's verdict from those two (e07_report.py).
#
# Every plan qualifies each machine first (a
# slow or bandwidth-starved machine is left before the payload, the download or the load): right after the CUDA probe
# pod-session.sh runs the pod's flavor's dsv41_machine_probe (tests/dsv41/engine/machine_probe.cu, built on its own by
# pod/build-machine-probe.sh into $BUILD_ROOT/<tag>/machine-probe, its MANIFEST's source HEAD's, staged beside the
# payload's pod files; the engine binary is not rebuilt). Its bounds, "clearly unfit" for every GPU model the plans
# take (the good machines: the earlier sessions' A100 and RTX PRO 6000 pods at the engine's init): the GPU read bandwidth per model, RTX PRO 6000 Blackwell (any edition) 800 GB/s, A100 80GB (SXM4 or PCIe) 900
# (about half of their 1935-2039), any other model measured and logged only, QUALIFY_MIN_GPU_READ_GBS one bound for
# every model; QUALIFY_MIN_H2D_GBS and QUALIFY_MIN_D2H_GBS 18 (lowest good 22.5); QUALIFY_MAX_DOORBELL_US 50 (the
# doorbell round trip's median; one slow machine lost ~1 ms per doorbell); the 16-thread host read floor per the
# created pod's GPU: 250 GB/s for an RTX PRO 6000 (and any GPU but
# the A100), from every probe.log of 2026-10-13 to 2026-10-15: the good hosts, RTX PRO 6000 pods on EPYC
# 9355P/9335/9655/9575F, 341.8-545.4 GB/s, decode ~18.6 ms per token; the slow RTX hosts, one EPYC 9575F machine at
# 175.4 on 2026-10-14 (545.4 on 2026-10-15, the same GPU UUID) and an Intel Xeon 6760P host at 109.5, where the
# CPU expert reads ran at 9-11 GB/s, decode at 33-43 ms per token and the engine took 7 min to start, both passed the
# old floor of 40. An A100 keeps 40 (lowest good 52.5): its pods' EPYC 7742 hosts measured 40.9-162.2, all under 250,
# so 250 would leave qfork (and an A100 fallback) no host. QUALIFY_MIN_HOST_READ_GBS set by hand is the floor for
# every GPU. probe.log and the session log name the floor each pod got; QUALIFY_MIN_SM_RATIO 0.70 of the GPU's
# own clocks.max.sm under the read load and
# QUALIFY_MIN_POWER_RATIO 0.70 of its power.default_limit for the enforced limit (the RTX PRO 6000's ~2000 MHz and
# 420 W, an A100's ~990 MHz and 210-280 W; the absolute values are logged). A measure past its bound
# deletes the pod, remembers its machine and takes the next offer, at most MAX_PODS pods (3 for the q plans: the third
# rejected machine ends the session, nothing uploaded). Every machine's values go to probe.log, the session log and the sheet
# (a session that never started its run gets one too). QUALIFY=0 is the only bypass (the CUDA probe alone).
# BAD_MACHINES (space or comma separated machine ids) names machines known bad from earlier sessions: a pod
# landing on one is deleted at once (pod-session.sh) and the next offer taken; the dry run prints the list. A session
# stopped by the HF download guard logs "slow download machine: <id>" for the next session's BAD_MACHINES.
# once dsv41_machine_probe passes, the pod's flavor's dsv41_cpu_experts_probe (from the
# strata tarball, tests/dsv41/engine/cpu_experts_probe.cpp) measures the host's CPU experts rate as decode's misses run
# (one token per expert through the engine's CpuExperts, fork numerics, the engine's 16 pool threads, bound to the
# GPU's NUMA node as e07-run.sh binds the engine) and a rate under CPU_EXPERTS_FLOOR_GBS (default 71) rejects the
# machine the same way: item 11's bar (1020 tokens in 17445 ms) with a measured session's non-CPU time (14505
# ms) for the 204 GB the CPU reads needs 70.7 GB/s; the slow hosts read at 44-45 (EPYC 9575F, 9355P), the good one
# (EPYC 9655) at 100.4.
#
# Every plan pulls the pod's out/ while it runs (on 2026-10-18 RunPod removed a cdet pod
# at 17:36 UTC mid lm-eval, after 7 steps exit 0, and the one pull at the end brought nothing back for
# 1.578 USD): each poll that sees a new last line of out/steps.log, the run still going, pulls out/ into OUT/out over
# one more fresh SSH connection (pod-session.sh's pull_steps: every file but the *.bin dumps and any above 64 MiB, the
# model is not under out/), a failed one a WARNING. The final pull stays as it was and its out/ replaces those. When it
# brings nothing back, OUT/session-lost and e07.log name the step the session was lost after, the steps finished and
# pulled, one a poll saw finish after the last pull, and the plan's steps with no line pulled; the sheet starts with
# them ("Session lost"), its tables from the last mid-run pull. A session that ends before its sheet (the watchdog at
# the cap, a failed command here) still gets one, from what was pulled. The time it adds: one pull per finished step,
# each under timeout POLL_SSH_S (45 s), none started within 45 s of the run's deadline, so the final pull, the delete
# and the pod's deadline (cdet's 75 min) keep their times and no pod step waits for a pull; the end of the run is
# seen at most 45 s later (0.033 USD at 2.60 USD/h). Measured: the 2026-10-17 cdet session's final pull of the same
# files (62 files, an 818 KiB archive, two SSH connections) took 7 s; pull_steps' find and tar of them locally
# 0.10-0.12 s (an 824 KiB stream).
# POLL_S (default 30) is the polls' interval.
#
# The cycle2 plan's knobs: dense readers only,
# and 1.95 USD/h, which leaves the RTX PRO 6000 out and takes the A100 (cap 92 min, the HF plan 57.9 min).
#
# The payload follows the GPU created: one is built per flavor GPU_TYPES needs (gpu_flavor in tools/dsv41/flavors.sh:
# cu130 sm_120 for the RTX PRO 6000, cu128 sm_80/89 for the A100 and L40S, each bundling its CUDA runtime) and the
# created pod's goes up, its CUDA probe first; each GPU's offers ask for hosts whose driver runs its flavor.
# The dry run prints the payloads, the one each GPU takes, every create body (both model paths), the HF manifest
# (built once from the HF API into HF_MANIFEST), the gate's arithmetic for each GPU on the real model's headers
# (HEADERS, the sparse layout of engine/gguf_layout.py: e07_gate.py locally, as the pod runs it; exit 2 when one
# fails), the pod's steps with their commands and time estimates per model path against the cap (exit 2 when one is
# over), the pull, the delete and the report; with DRY_ENGINE set (a dsv41-engine that runs here: verify.sh's
# builder image) also dsv41-engine --dry-run of the correctness, isolation and bench runs and dsv41-layer-ref --dry-run
# of each numerics on that layout. The gate per GPU takes the pod's memory cgroup as MIN_RAM_GB GB unless PROBE_LOG
# names a past session's probe.log: then the last cgroup limit and MemTotal that session's
# pod reported (pod-session.sh's "pod memory cgroup" line, or for a session before it the same directory's
# out/env.log, e07-run.sh's), so the dry run's placement and host homes are a real pod's; it prints the expected host
# homes (how many of the slab's experts, the fraction, the --hot-homes the engine gets).
#
# Money (one session <= 3 USD): MAX_USD_PER_HR (default 2.45: the RTX PRO
# 6000's 2.09-2.19 list price plus the 320 GB container disk, under the 2.5 USD/h approval line; the A100 SXM4 lists
# 1.79), MAX_SESSION_USD 3.00, CAP_MIN 110 (the session cap leaves 73 min at 2.45 USD/h), MAX_PODS 2 (the q plans 3).
# Environment as devgpu.sh's (GPU_TYPES default the two RTX PRO 6000 Blackwell editions, then the two A100 80GB
# variants: the L40S is out of the default list, any override still meets e07_gate.py's 8 GiB of host-RAM margin or
# the pod stops before the copy; IMAGE, DISK_GB default 320, MIN_RAM_GB default 250, SSH_KEY, PULL_S, POLL_FAILS),
# plus NETWORK_VOLUME (a RunPod network volume id holding the model's 7 Q2_K shards; default none: no volume offer;
# cdet needs it or HF_CLOUDS)
# and VOLUME_DC (default EUR-IS-1: the volume's data center; its offer is SECURE: network volumes are secure cloud only), HF_CLOUDS
# (default "SECURE COMMUNITY", none for cdet: the clouds tried in any data center with the model from HF, "" for
# none), HF_REPO (default
# vcruz305/DeepSeek-V4.1-Flash-GGUF), HF_REVISION (default c4a08554), HF_MANIFEST (default
# $BUILD_ROOT/e07/hf-manifest-<HF_REVISION>.json), HF_TOKEN (optional: the repo is public; sent to a pod without the
# volume as /root/.hf-token over SSH, never logged nor in the payload), BUILD_ROOT, COUNTS (default $BUILD_ROOT/counts.json:
# the routing counts of a real-model run, the t04a payload's counts.json), ORACLE (the stock fork's oracle dump of the
# real model, run-1), VOCAB (the real tokenizer, for the prompts), FORK_DIR (the corpus of the
# prompts), PROMPTS (default $BUILD_ROOT/e07/prompts, built by pod/e07_prompts.py when missing), HEADERS (default
# $BUILD_ROOT/engine/headers), SHEET_ROOT (default $BUILD_ROOT/sheets), OUT (default
# $BUILD_ROOT/engine/e07a-pod/<UTC time>), E07_PLAN (cycle2, the default, or full: the first session's steps too; e08, e09,
# e07b, qfork, qengine, qdiag, qdet, cbase, ccache above), HF_GBS (the HF download rate in the plan, default 0.115, the Part Q plans 0.385) and E07_READERS (the correctness runs'
# reader modes: "dense reuse", the default, or "dense"; e07b's default "dense"). PULL_CAP_MB is 0: the
# engine's and the CPU reference's dumps stay on the pod (the parity verdicts and their JSON come back).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
E07_PLAN="${E07_PLAN:-cycle2}"
# the plans served over HTTP with per-step timeouts, MAX_PODS 3 and a 256 GB host: Part Q's (q*) and Part C's
# (cbase, ccache); pc: Part C's load plans
# the plans that run cbatch's steps (item 11) and clong's (items 12 and 13)
# cdet runs cbatch's steps (less load-c<N>, which plain cbatch alone runs) and its own; has_c05: cdet
# and cfinal (the batched server only, item 3 batched on it), cfinal without cdet's determinism steps and A/B
has_cbatch=0 has_clong=0 has_cdet=0 has_c05=0 cbatch_loads=0
[[ "$E07_PLAN" == cbatch || "$E07_PLAN" == cbatch-clong || "$E07_PLAN" == cdet || "$E07_PLAN" == cfinal ]] && has_cbatch=1
[[ "$E07_PLAN" == clong || "$E07_PLAN" == cbatch-clong ]] && has_clong=1
[[ "$E07_PLAN" == cdet ]] && has_cdet=1
[[ "$E07_PLAN" == cdet || "$E07_PLAN" == cfinal ]] && has_c05=1
[[ "$E07_PLAN" == cbatch ]] && cbatch_loads=1
qc=0 pc=0
[[ "$E07_PLAN" == cbase || "$E07_PLAN" == ccache ]] || (( has_cbatch || has_clong )) && pc=1
[[ "$E07_PLAN" == q* ]] || (( pc )) && qc=1
# the RTX PRO 6000 first, then the two A100 80GB variants; not the L40S (an earlier fallback), which passed the
# 250 GB gate by 0.05 GiB. qfork: the A100s only; qengine and qdiag: the RTX PRO 6000 only (the target card)
RTX_PRO="NVIDIA RTX PRO 6000 Blackwell Server Edition;NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
A100="NVIDIA A100-SXM4-80GB;NVIDIA A100 80GB PCIe"
case "$E07_PLAN" in qfork) GPU_DEFAULT="$A100" ;; qengine|qdiag|qdet|cbase|ccache|cbatch|clong|cbatch-clong|cdet|cfinal) GPU_DEFAULT="$RTX_PRO" ;; *) GPU_DEFAULT="$RTX_PRO;$A100" ;; esac
GPU_TYPES="${GPU_TYPES:-$GPU_DEFAULT}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-320}"
# 1.95 USD/h admits the RTX PRO 6000 COMMUNITY (1.69 listed) and the A100 SXM4 fallback;
# the fork's A100 under 1.95; the standing limit for the RTX PRO 6000: 2.60
case "$E07_PLAN" in e07b|qfork) E07B_USD_PER_HR=1.95 ;; qengine|qdiag|qdet|cbase|ccache|cbatch|clong|cbatch-clong|cdet|cfinal) E07B_USD_PER_HR=2.60 ;; *) E07B_USD_PER_HR=2.45 ;; esac
MAX_USD_PER_HR="${MAX_USD_PER_HR:-$E07B_USD_PER_HR}"
# the e09 session <= 1.5 USD: the qdiag session too; the other plans'
# cap is the 3 USD measurement-pod rule, qdet's 2.50, cbase's 2.00
# and ccache's 2.30 (its plan with every request read whole, 46.7 min, fits the pod's 53 min at 2.60 USD/h)
[[ "$E07_PLAN" == e09 || "$E07_PLAN" == qdiag ]] && E09_SESSION_USD=1.50 || E09_SESSION_USD=3.00
[[ "$E07_PLAN" != qdet ]] || E09_SESSION_USD=2.50
(( ! pc )) || E09_SESSION_USD=2.00
[[ "$E07_PLAN" != ccache ]] || E09_SESSION_USD=2.30
# cbatch: the 3 USD cap, the plan about 2; the same 3 USD for clong and the combined plan
(( ! has_cbatch && ! has_clong )) || E09_SESSION_USD=3.00
# cfinal's session from HF in any region, 3.35 USD
[[ "$E07_PLAN" != cfinal ]] || E09_SESSION_USD=3.35
MAX_SESSION_USD="${MAX_SESSION_USD:-$E09_SESSION_USD}"
CAP_MIN="${CAP_MIN:-110}"
# the q plans (and cbase, ccache) try up to 3 pods, then give up for the round
(( qc )) && Q_PODS=3 || Q_PODS=2
MAX_PODS="${MAX_PODS:-$Q_PODS}"
# the machine qualification's bounds, each overridable;
# QUALIFY=0 bypasses it
QUALIFY="${QUALIFY:-1}"
# the GPU read floor per known model, NAME=GB/s with NAME a part of the CUDA device's name (the probe takes the first
# that matches; a GPU none names is measured and logged only); QUALIFY_MIN_GPU_READ_GBS replaces them for every model
QUALIFY_GPU_READ_MODELS=("RTX PRO 6000 Blackwell=800" "A100-SXM4-80GB=900" "A100 80GB PCIe=900")
QUALIFY_MIN_GPU_READ_GBS="${QUALIFY_MIN_GPU_READ_GBS:-}" QUALIFY_MIN_H2D_GBS="${QUALIFY_MIN_H2D_GBS:-18}"
QUALIFY_MIN_D2H_GBS="${QUALIFY_MIN_D2H_GBS:-18}" QUALIFY_MAX_DOORBELL_US="${QUALIFY_MAX_DOORBELL_US:-50}"
QUALIFY_MIN_HOST_READ_GBS="${QUALIFY_MIN_HOST_READ_GBS:-}" QUALIFY_MIN_SM_RATIO="${QUALIFY_MIN_SM_RATIO:-0.70}"
QUALIFY_MIN_POWER_RATIO="${QUALIFY_MIN_POWER_RATIO:-0.70}"
# the CPU experts floor and the pool threads it is measured with (the placements'
# --threads 16, e07-run.sh); pod-session.sh's qualify runs dsv41_cpu_experts_probe when the floor is set
CPU_EXPERTS_FLOOR_GBS="${CPU_EXPERTS_FLOOR_GBS:-71}" CPU_EXPERTS_THREADS=16
# qualify_host_read G (pod-session.sh's qualify, per created pod): "GB/s|why" of the host read floor for GPU G
#: QUALIFY_MIN_HOST_READ_GBS when set, else 40 for an A100, else 250
qualify_host_read() {
  if [[ -n "$QUALIFY_MIN_HOST_READ_GBS" ]]; then echo "$QUALIFY_MIN_HOST_READ_GBS|QUALIFY_MIN_HOST_READ_GBS, set by hand"
  elif [[ "$1" == *A100* ]]; then echo "40|the A100 default, its hosts measured 40.9-162.2"
  else echo "250|the default for any GPU but the A100, the RTX PRO 6000's good hosts measured 341.8-545.4"
  fi
}
if [[ -n "$QUALIFY_MIN_GPU_READ_GBS" ]]; then
  QUALIFY_ARGS="--min-gpu-read-gbs $QUALIFY_MIN_GPU_READ_GBS"
  QUALIFY_GPU_READ_TEXT="GPU read >= $QUALIFY_MIN_GPU_READ_GBS GB/s for every model (QUALIFY_MIN_GPU_READ_GBS)"
else
  QUALIFY_ARGS="" QUALIFY_GPU_READ_TEXT="GPU read per model:"
  for qm in "${QUALIFY_GPU_READ_MODELS[@]}"; do
    QUALIFY_ARGS+="--gpu-read-bound '$qm' "
    QUALIFY_GPU_READ_TEXT+=" ${qm%=*} >= ${qm##*=} GB/s,"
  done
  QUALIFY_ARGS="${QUALIFY_ARGS% }"
  QUALIFY_GPU_READ_TEXT+=" any other model info only (QUALIFY_MIN_GPU_READ_GBS: one bound for all)"
fi
QUALIFY_ARGS+=" --min-h2d-gbs $QUALIFY_MIN_H2D_GBS --min-d2h-gbs $QUALIFY_MIN_D2H_GBS"
QUALIFY_ARGS+=" --max-doorbell-us $QUALIFY_MAX_DOORBELL_US"
QUALIFY_ARGS+=" --min-sm-ratio $QUALIFY_MIN_SM_RATIO --min-power-ratio $QUALIFY_MIN_POWER_RATIO"
# the probe takes ~10 s (3 s of read load, 4 GiB of host reads); a hung driver gets this long
QUALIFY_S=120
# where pod/build-machine-probe.sh put each flavor's probe: MACHINE_PROBE_ROOT/<tag>/machine-probe
MACHINE_PROBE_ROOT="${MACHINE_PROBE_ROOT:-$BUILD_ROOT}"
# Part Q (and Part C): the fork's and the engine's hosts have >= 256 GB of RAM
(( qc )) && Q_RAM_GB=256 || Q_RAM_GB=250
MIN_RAM_GB="${MIN_RAM_GB:-$Q_RAM_GB}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/runpod}"
# NETWORK_VOLUME (a RunPod network volume holding the model, in VOLUME_DC) has no default: no volume offer unless it
# is set. cdet takes the volume only (its HF plan does not fit 2.90 USD): no HF offer unless HF_CLOUDS is set, so
# cdet needs NETWORK_VOLUME or HF_CLOUDS; every other plan downloads from HF in any region by default
VOLUME_DEF="" HF_DEF="SECURE COMMUNITY"
[[ "$E07_PLAN" != cdet ]] || HF_DEF=""
NETWORK_VOLUME="${NETWORK_VOLUME-$VOLUME_DEF}"
VOLUME_DC="${VOLUME_DC:-EUR-IS-1}"
HF_CLOUDS="${HF_CLOUDS-$HF_DEF}"
# the 7 Q2_K shards of 246.3 GiB the volume holds (their headers are byte for byte HEADERS')
HF_REPO="${HF_REPO:-vcruz305/DeepSeek-V4.1-Flash-GGUF}"
HF_REVISION="${HF_REVISION:-c4a08554}"
HF_MANIFEST="${HF_MANIFEST:-$BUILD_ROOT/e07/hf-manifest-$HF_REVISION.json}"
COUNTS="${COUNTS:-$BUILD_ROOT/counts.json}"
ORACLE="${ORACLE:-$BUILD_ROOT/oracle-dsv41/run-1}"
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"
FORK_DIR="${FORK_DIR:-$HOME/llama-dsv41}"
PROMPTS="${PROMPTS:-$BUILD_ROOT/e07/prompts}"
HEADERS="${HEADERS:-$BUILD_ROOT/engine/headers}"
SHEET_ROOT="${SHEET_ROOT:-$BUILD_ROOT/sheets}"
OUT="${OUT:-$BUILD_ROOT/engine/e07a-pod/$(date -u +%Y%m%dT%H%M%SZ)}"
PULL_S="${PULL_S:-300}"
PULL_CAP_MB=0
POLL_FAILS="${POLL_FAILS:-10}"
# the pod's steps (e07-run.sh): cycle2 (the default: correctness, the isolation
# steps and bench-a) or full (session 1's list plus those); the correctness runs' reader modes, dense first
# (e07b runs dense only)
[[ "$E07_PLAN" == e07b ]] && E07B_READERS=dense || E07B_READERS="dense reuse"
E07_READERS="${E07_READERS:-$E07B_READERS}"
# the HF download's rate in the plan, GB/s (the 2026-10-10 session's 0.115: the pessimistic case); the Part Q
# plans: 0.385, the first e07b session's (2026-10-10, the slower of the two latest real-model sessions), the guard still stopping a slower download; qdet: 0.924, the 2026-10-13 qengine session's (264.5 GB in 286.3 s);
# cbase and ccache: 0.25, the worst measured
(( qc )) && Q_HF_GBS=0.385 || Q_HF_GBS=0.115
[[ "$E07_PLAN" != qdet ]] || Q_HF_GBS=0.924
(( ! pc )) || Q_HF_GBS=0.25
HF_GBS="${HF_GBS:-$Q_HF_GBS}"
# Part Q: the tasks and needles of the session (spaces or commas between tasks), the rates of its estimates
# (header), the needle prompts and the 2026-10-06 API run's usage log (each task's tokens)
Q_TASKS="${Q_TASKS:-gsm8k_cot_llama ifeval humaneval_instruct}"
Q_TASKS="${Q_TASKS//,/ }"
[[ "$E07_PLAN" == qfork ]] && Q_DEF="16k|115|115|20" || Q_DEF="16k 64k 256k|288.7|1866.8|60"
IFS='|' read -r Q_NEEDLES_DEF Q_PROMPT_DEF Q_LONG_DEF Q_DECODE_DEF <<<"$Q_DEF"
Q_NEEDLES="${Q_NEEDLES-$Q_NEEDLES_DEF}"
Q_PROMPT_TOK_S="${Q_PROMPT_TOK_S:-$Q_PROMPT_DEF}" Q_LONG_PROMPT_TOK_S="${Q_LONG_PROMPT_TOK_S:-$Q_LONG_DEF}"
Q_DECODE_TOK_S="${Q_DECODE_TOK_S:-$Q_DECODE_DEF}"
# the engine's decode on the continuation, ms per step, above which qengine stops
Q_SPEED_MAX_MS="${Q_SPEED_MAX_MS:-30}"
# the fork's slots, and lm-eval's concurrent requests against it
Q_FORK_NP="${Q_FORK_NP:-1}"
# and its decode on the continuation, ms per generated token, above which qfork stops
Q_FORK_MAX_MS="${Q_FORK_MAX_MS:-80}"
QUALITY_NEEDLE="${QUALITY_NEEDLE:-$BUILD_ROOT/quality/needle.json}"
# Part C item 9: the users of each load step, and their conversations, built locally
# (quality/chat_load.py build: 8 users of 16K tokens)
CBASE_USERS="${CBASE_USERS:-1 2 4 8}"
CBASE_USERS="${CBASE_USERS//,/ }"
CBASE_CONVERSATIONS="${CBASE_CONVERSATIONS:-$BUILD_ROOT/quality/conversations-16k.json}"
# Part C item 10: the alternating users and their turns, then the cbase
# runs again (cbase's conversations)
CCACHE_ALT_USERS="${CCACHE_ALT_USERS:-4}" CCACHE_ALT_TURNS="${CCACHE_ALT_TURNS:-3}"
# the second alternating run's think time between a reply's end and the next turn, seconds
CCACHE_THINK_S="${CCACHE_THINK_S:-5}"
CCACHE_USERS="${CCACHE_USERS:-1 2 4}"
CCACHE_USERS="${CCACHE_USERS//,/ }"
# Part C item 11: the users decoding at once, the batch slots, their positions and
# the conversation checkpoints, and the single user's short conversation (quality/chat_load.py build --users 1
# --context 1024 locally)
CBATCH_USERS="${CBATCH_USERS:-4 8}"
CBATCH_USERS="${CBATCH_USERS//,/ }"
CBATCH_SLOTS="${CBATCH_SLOTS:-8}" CBATCH_CTX="${CBATCH_CTX:-69632}" CBATCH_CHECKPOINTS="${CBATCH_CHECKPOINTS:-8}"
CBATCH_SHORT="${CBATCH_SHORT:-$BUILD_ROOT/quality/conversations-short.json}"
# the steady-state step's users, window (s), repeats and reply length (chat_load.py steady)
CBATCH_STEADY_USERS="${CBATCH_STEADY_USERS:-4}" CBATCH_STEADY_WINDOW_S="${CBATCH_STEADY_WINDOW_S:-15}"
# (cdet's repeats 2, its HF plan's latest end >= 2 min before the pod's deadline)
CBATCH_STEADY_REPEATS="${CBATCH_STEADY_REPEATS:-$([[ "$E07_PLAN" == cdet ]] && echo 2 || echo 3)}"
CBATCH_STEADY_MAX_TOKENS="${CBATCH_STEADY_MAX_TOKENS:-1024}"
# the steady state again with these users decoding at once, reported beside item 11's
# (aggregate and per user, not a bar; "": none; default 8 in cbatch-clong, none in cbatch, whose load-c8 has 8 users),
# with the same window, repeats and reply length; and (minor) the single
# user's turns at short context (chat_load.py --turns, 1 to 4; 4: at least 4 requests for the baseline)
CBATCH_STEADY_REPORT="${CBATCH_STEADY_REPORT-$([[ "$E07_PLAN" == cbatch-clong ]] && echo 8 || true)}"
CBATCH_STEADY_REPORT="${CBATCH_STEADY_REPORT//,/ }"
CBATCH_SHORT_TURNS="${CBATCH_SHORT_TURNS:-4}"
# cbatch-clong's second cbatch server, its engine with CBATCH_SPARE spare
# slab slots per layer (4: the deployment's 2 and 2 more), its gate sized for them. Default 0, no such server: with 4
# the HF plan's estimates take 66.7 min (2.89 USD at 2.60 USD/h, within the 3.00 cap) but steady-c4-spare's whole
# timeout ends at minute 68.0, past the pod's 67-minute deadline, so the dry run refuses it (the variant
# only runs if the dry run passes)
CBATCH_SPARE="${CBATCH_SPARE:-0}"
# cdet's (and cfinal's) lm-eval run on the batched server, its concurrent
# requests and its tasks in the order the pod runs them (ifeval first: the longest, its 2 x timeout with the others
# after it)
CDET_CONCURRENT=4 CDET_TASKS="ifeval gsm8k_cot_llama humaneval_instruct"
# Part C items 12 and 13: the users decoding when the 64K prompt comes,
# their replies (past the long read, at about one token a chunk), the batch slots (each holds the 64K prompt and its
# reply), the checkpoints, the long user's conversation (quality/chat_load.py build --users N --long-context 65536)
CLONG_USERS="${CLONG_USERS:-4}" CLONG_MAX_TOKENS="${CLONG_MAX_TOKENS:-1024}"
CLONG_SLOTS="${CLONG_SLOTS:-8}" CLONG_CTX="${CLONG_CTX:-69632}" CLONG_CHECKPOINTS="${CLONG_CHECKPOINTS:-8}"
CLONG_LONG_CONTEXT=65536
# the interleave chunk is the engine's prefill chunk, pod_server.sh's --chunk 16384
CLONG_CHUNK=16384
CLONG_CONVERSATIONS="${CLONG_CONVERSATIONS:-$BUILD_ROOT/quality/conversations-64k.json}"
# the load steps' users: cbase's, ccache's, cbatch's or clong's
case "$E07_PLAN" in
  ccache) LOAD_USERS="$CCACHE_USERS" LOAD_VAR=CCACHE_USERS ;;
  cbatch|cbatch-clong|cdet|cfinal) LOAD_USERS="$CBATCH_USERS" LOAD_VAR=CBATCH_USERS ;;
  clong) LOAD_USERS="$CLONG_USERS" LOAD_VAR=CLONG_USERS ;;
  *) LOAD_USERS="$CBASE_USERS" LOAD_VAR=CBASE_USERS ;;
esac
API_THEN_USAGE="${API_THEN_USAGE:-$BUILD_ROOT/quality/api_usage.jsonl}"
SSH_WAIT_S=600 PROBE_S=60 PULL_TAR_S=120 START_S=300 POLL_SSH_S=45
POLL_S="${POLL_S:-30}"
# out/ pulled after each poll that sees a new step (pod-session.sh's pull_steps)
PULL_STEPS=1
REMOTE=/root/e07
RUN_SCRIPT=e07-run.sh
SESSION_LOG="$OUT/e07.log"
REST_SHOWN=https://rest.runpod.io/v1
CATALOG_SHOWN=https://api.runpod.io/v2/catalog/gpus

dry=0 payload_only=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry=1 ;;
    --payload-only) payload_only=1 ;;
    -h|--help) sed -n '2,/^set -euo pipefail$/{/^set -euo/!p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "e07.sh: unknown argument '$arg' (--dry-run or --payload-only)" >&2; exit 2 ;;
  esac
done
(( ! (dry && payload_only) )) || { echo "e07.sh: --dry-run or --payload-only, not both" >&2; exit 2; }
# shellcheck source=tools/dsv41/pod/pod-session.sh
source "$here/pod-session.sh"
[[ "$MIN_RAM_GB" =~ ^[0-9]+$ ]] || { echo "e07.sh: MIN_RAM_GB='$MIN_RAM_GB' is not a whole number of GB" >&2; exit 2; }
[[ "$POLL_FAILS" =~ ^[1-9][0-9]*$ ]] || { echo "e07.sh: POLL_FAILS='$POLL_FAILS' is not a whole number >= 1" >&2; exit 2; }
[[ "$POLL_S" =~ ^[1-9][0-9]*$ ]] || { echo "e07.sh: POLL_S='$POLL_S' is not a whole number of seconds >= 1" >&2; exit 2; }
case "$E07_PLAN" in cycle2|full|e08|e09|e07b|qfork|qengine|qdiag|qdet|cbase|ccache|cbatch|clong|cbatch-clong|cdet|cfinal) ;;
  *) echo "e07.sh: E07_PLAN='$E07_PLAN': cycle2, full, e08, e09, e07b, qfork, qengine, qdiag, qdet, cbase, ccache, cbatch, clong, cbatch-clong, cdet or cfinal" >&2; exit 2 ;;
esac
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "$HF_GBS" 2>/dev/null ||
  { echo "e07.sh: HF_GBS='$HF_GBS' is not a rate in GB/s above 0" >&2; exit 2; }
[[ "$QUALIFY" == 0 || "$QUALIFY" == 1 ]] || { echo "e07.sh: QUALIFY='$QUALIFY': 1 (the default) or 0 (no machine qualification)" >&2; exit 2; }
# the absolute bounds of the first qualification, replaced by the ratios
for v in QUALIFY_MIN_SM_MHZ QUALIFY_MIN_POWER_LIMIT_W; do
  [[ -z "${!v:-}" ]] || { echo "e07.sh: $v is gone: the bound is QUALIFY_MIN_SM_RATIO or QUALIFY_MIN_POWER_RATIO, a ratio of the GPU's own maximum" >&2; exit 2; }
done
for v in QUALIFY_MIN_H2D_GBS QUALIFY_MIN_D2H_GBS QUALIFY_MAX_DOORBELL_US CPU_EXPERTS_FLOOR_GBS \
         $([[ -n "$QUALIFY_MIN_GPU_READ_GBS" ]] && echo QUALIFY_MIN_GPU_READ_GBS) \
         $([[ -n "$QUALIFY_MIN_HOST_READ_GBS" ]] && echo QUALIFY_MIN_HOST_READ_GBS); do
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "${!v}" 2>/dev/null ||
    { echo "e07.sh: $v='${!v}' is not a number above 0" >&2; exit 2; }
done
for v in QUALIFY_MIN_SM_RATIO QUALIFY_MIN_POWER_RATIO; do
  python3 -c "import sys; sys.exit(0 if 0 < float(sys.argv[1]) <= 1 else 1)" "${!v}" 2>/dev/null ||
    { echo "e07.sh: $v='${!v}' is not a ratio above 0 and at most 1" >&2; exit 2; }
done
# the plans without the session 1 correctness runs (e08, e09, qfork, qengine, qdiag; e07b runs its own: dense free and
# the isolated MoE)
bench_only=0
[[ "$E07_PLAN" == e08 || "$E07_PLAN" == e09 || "$E07_PLAN" == e07b ]] || (( qc )) && bench_only=1
# the e09 plan's prefetch region of 128 experts in the gate and every engine run (e09, e07b, qengine, qdiag, qdet, cbase, ccache)
prefetch=32
[[ "$E07_PLAN" == e09 || "$E07_PLAN" == e07b || "$E07_PLAN" == qengine || "$E07_PLAN" == qdiag || "$E07_PLAN" == qdet ]] ||
  (( pc )) && prefetch=128
# the combined plan's second cbatch server, CBATCH_SPARE spare slots
has_spare=0
[[ "$E07_PLAN" == cbatch-clong && "$CBATCH_SPARE" != 0 ]] && has_spare=1
# the plans whose every pod step has its own timeout and whose HF download is planned at HF_GBS with its guard
timed=0
[[ "$E07_PLAN" == e07b ]] || (( qc )) && timed=1
if (( pc )); then
  for n in $LOAD_USERS; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "e07.sh: $LOAD_VAR has '$n' (whole numbers of users >= 1)" >&2; exit 2; }
  done
  [[ -n "$LOAD_USERS" ]] || { echo "e07.sh: $LOAD_VAR is empty" >&2; exit 2; }
  [[ "$CCACHE_ALT_USERS" =~ ^[1-9][0-9]*$ && "$CCACHE_ALT_TURNS" =~ ^[1-4]$ ]] ||
    { echo "e07.sh: CCACHE_ALT_USERS='$CCACHE_ALT_USERS' (users >= 1), CCACHE_ALT_TURNS='$CCACHE_ALT_TURNS' (1 to 4: chat_load.py's turns)" >&2; exit 2; }
  [[ "$CCACHE_THINK_S" =~ ^[1-9][0-9]*$ ]] ||
    { echo "e07.sh: CCACHE_THINK_S='$CCACHE_THINK_S' (whole seconds >= 1: the second alternating run's think time)" >&2; exit 2; }
  [[ "$CBATCH_SLOTS" =~ ^[1-8]$ && "$CBATCH_CTX" =~ ^[1-9][0-9]*$ && "$CBATCH_CHECKPOINTS" =~ ^[1-9][0-9]*$ ]] ||
    { echo "e07.sh: CBATCH_SLOTS='$CBATCH_SLOTS' (1 to 8), CBATCH_CTX='$CBATCH_CTX', CBATCH_CHECKPOINTS='$CBATCH_CHECKPOINTS' (whole numbers >= 1)" >&2; exit 2; }
  # robust_check.py's overflow needs 2 slots or more; the slot holds the long prompt and its 16 tokens
  [[ "$CLONG_SLOTS" =~ ^[2-8]$ && "$CLONG_CTX" =~ ^[1-9][0-9]*$ && "$CLONG_CHECKPOINTS" =~ ^[1-9][0-9]*$ &&
     "$CLONG_USERS" =~ ^[1-9][0-9]*$ && "$CLONG_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] && (( CLONG_USERS < CLONG_SLOTS )) ||
    { echo "e07.sh: CLONG_SLOTS='$CLONG_SLOTS' (2 to 8), CLONG_USERS='$CLONG_USERS' (fewer than the slots: the long user takes one), CLONG_CTX='$CLONG_CTX', CLONG_CHECKPOINTS='$CLONG_CHECKPOINTS', CLONG_MAX_TOKENS='$CLONG_MAX_TOKENS' (whole numbers >= 1)" >&2; exit 2; }
  [[ "$CBATCH_STEADY_USERS" =~ ^[2-8]$ && "$CBATCH_STEADY_WINDOW_S" =~ ^[1-9][0-9]*$ && "$CBATCH_STEADY_REPEATS" =~ ^[1-9]$ &&
     "$CBATCH_STEADY_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] && (( CBATCH_STEADY_USERS <= CBATCH_SLOTS )) ||
    { echo "e07.sh: CBATCH_STEADY_USERS='$CBATCH_STEADY_USERS' (2 to CBATCH_SLOTS $CBATCH_SLOTS), CBATCH_STEADY_WINDOW_S='$CBATCH_STEADY_WINDOW_S' (whole seconds >= 1), CBATCH_STEADY_REPEATS='$CBATCH_STEADY_REPEATS' (1 to 9), CBATCH_STEADY_MAX_TOKENS='$CBATCH_STEADY_MAX_TOKENS' (tokens >= 1)" >&2; exit 2; }
  for n in $CBATCH_STEADY_REPORT; do
    [[ "$n" =~ ^[2-8]$ && "$n" != "$CBATCH_STEADY_USERS" ]] && (( n <= CBATCH_SLOTS )) ||
      { echo "e07.sh: CBATCH_STEADY_REPORT has '$n' (2 to CBATCH_SLOTS $CBATCH_SLOTS, not item 11's CBATCH_STEADY_USERS $CBATCH_STEADY_USERS)" >&2; exit 2; }
  done
  [[ "$CBATCH_SHORT_TURNS" =~ ^[1-4]$ ]] ||
    { echo "e07.sh: CBATCH_SHORT_TURNS='$CBATCH_SHORT_TURNS' (1 to 4: chat_load.py's turns)" >&2; exit 2; }
  [[ "$CBATCH_SPARE" == 0 || "$CBATCH_SPARE" =~ ^[3-9]$ ]] ||
    { echo "e07.sh: CBATCH_SPARE='$CBATCH_SPARE' (3 to 9 spare slab slots per layer, more than the deployment's 2; 0: no such server)" >&2; exit 2; }
elif [[ "$E07_PLAN" == q* ]]; then
  for t in $Q_TASKS; do
    case "$t" in gsm8k_cot_llama|ifeval|humaneval_instruct) ;; *) echo "e07.sh: Q_TASKS has '$t'" >&2; exit 2 ;; esac
  done
  for n in $Q_NEEDLES; do
    case "$n" in 16k|64k|256k) ;; *) echo "e07.sh: Q_NEEDLES has '$n' (16k, 64k, 256k)" >&2; exit 2 ;; esac
  done
  python3 -c "import sys; sys.exit(0 if all(float(x) > 0 for x in sys.argv[1:]) else 1)" "$Q_PROMPT_TOK_S" \
    "$Q_LONG_PROMPT_TOK_S" "$Q_DECODE_TOK_S" 2>/dev/null || { echo "e07.sh: the Q_*_TOK_S rates must be above 0" >&2; exit 2; }
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "$Q_SPEED_MAX_MS" 2>/dev/null ||
    { echo "e07.sh: Q_SPEED_MAX_MS='$Q_SPEED_MAX_MS' is not a time in ms above 0" >&2; exit 2; }
  [[ "$Q_FORK_NP" =~ ^[1-9][0-9]*$ ]] || { echo "e07.sh: Q_FORK_NP='$Q_FORK_NP' is not a whole number of slots >= 1" >&2; exit 2; }
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "$Q_FORK_MAX_MS" 2>/dev/null ||
    { echo "e07.sh: Q_FORK_MAX_MS='$Q_FORK_MAX_MS' is not a time in ms above 0" >&2; exit 2; }
  # qdiag's and qdet's estimates are their own (no quality/q_plan.py)
  [[ "$E07_PLAN" == qdiag || "$E07_PLAN" == qdet || -f "$API_THEN_USAGE" ]] ||
    { echo "e07.sh: no $API_THEN_USAGE (the 2026-10-06 API run's tokens per task)" >&2; exit 1; }
fi
case "$E07_READERS" in
  dense|"dense reuse") ;;
  *) echo "e07.sh: E07_READERS='$E07_READERS': 'dense' or 'dense reuse' (layer-parity reads dense-forced)" >&2; exit 2 ;;
esac

IFS=';' read -ra gpus <<<"$GPU_TYPES"
# the inputs, per flavor GPU_TYPES needs (its tag in tags, in GPU order): the strata tarball's engine, the fork
# tarball's libraries for its llama-tokenize (the build dir's: the tarball has none; the same build when
# libllama.so.0 is the tarball's byte for byte); then the oracle, the profile
declare -A strata_tgz fork_tgz tokenize cuda_major
tags=()
for g in "${gpus[@]}"; do
  t="$(gpu_cfg "$g" FLAVOR_TAG)"
  [[ -z "${strata_tgz[$t]:-}" ]] || continue
  flavor_config "$(gpu_flavor "$g")"
  shopt -s nullglob
  ts=("$BUILD_ROOT"/dist"${BUILD_TAG:-}"/strata-*-"$t-$ARCH_TAG".tar.gz) fs=("$BUILD_ROOT"/dist"${BUILD_TAG:-}"/llama-dsv41-*-"$t-$ARCH_TAG".tar.gz)
  shopt -u nullglob
  (( ${#ts[@]} == 1 && ${#fs[@]} == 1 )) ||
    { echo "e07.sh: '$g' takes $t: expected one strata and one llama-dsv41 $t-$ARCH_TAG tarball in $BUILD_ROOT/dist${BUILD_TAG:-} (tools/dsv41/build.sh)" >&2; exit 1; }
  strata_tgz[$t]="${ts[0]}" fork_tgz[$t]="${fs[0]}" tokenize[$t]="$BUILD_ROOT/$t/fork/bin/llama-tokenize" cuda_major[$t]="$CUDA_MAJOR"
  tags+=("$t")
done
# the payload's files of a flavor: the engine with the cuBLAS it loads, the fork's libraries with their cudart
bins_of() { echo dsv41-engine dsv41-layer-ref dsv41_cuda_probe dsv41_cpu_experts_probe "libcublas.so.${cuda_major[$1]}" "libcublasLt.so.${cuda_major[$1]}"; }
# (one line: the callers read it with read -ra)
fork_libs_of() {
  local l="libllama.so.0 libllama-common.so.0 libggml.so.0 libggml-base.so.0 libggml-cpu.so.0 libggml-cuda.so.0"
  l+=" libcudart.so.${cuda_major[$1]} libgomp.so.1"
  # qfork: the fork's server
  [[ "$E07_PLAN" != qfork ]] || l+=" llama-server libllama-server-impl.so libmtmd.so.0"
  echo "$l"
}
for f in "$COUNTS" "$ORACLE/dump-manifest.json" "$VOCAB" "${tokenize[@]}" "$HEADERS/head-1.bin"; do
  [[ -f "$f" ]] || { echo "e07.sh: no $f" >&2; exit 1; }
done
# the pod's sources, at their repo paths under src/ (serve_check.py and the tools find the repo from their own)
src=(serve tools/strata_tokenizer.py tools/gguf_reader.py tools/llama_unicode_flags.txt
     tools/dsv41/engine/serve_check.py tools/dsv41/engine/oracle_parity.py tools/dsv41/engine/engine_parity.py
     tools/dsv41/engine/layer_parity.py tools/dsv41/engine/two_turn_check.py
     tools/dsv41/engine/gguf_layout.py tools/dsv41/engine/tokenizer_parity.py tools/dsv41/engine/tokenizer_corpus.txt
     tools/dsv41/engine/kv_budget.py
     tools/dsv41/oracle/hot_set.py tools/dsv41/oracle/gguf_header.py tools/dsv41/oracle/layout-dsv41-q2k.json
     tools/dsv41/pod/e07_bench.py tools/dsv41/pod/e07_gate.py tools/dsv41/pod/e07_prompts.py
     tools/dsv41/pod/prefill_prompts.py)
# Part Q: the quality tooling (eval.sh, the proxy, the subsets, the needle check, pod_server.sh; Part C's
# chat_load.py)
(( ! qc )) || src+=(tools/dsv41/quality)
# the Part Q plans: the speed check (qfork's too) and the report's parser it reads the
# engine's log with
(( ! qc )) || src+=(tools/dsv41/pod/speed_check.py tools/dsv41/pod/qdiag_report.py)
# the pod scripts at the payload's top level
pod_files=(e07-run.sh pod-steps.sh pod-samplers.sh model-copy.sh t04a-lib.sh hf-download.sh hf_model.py)
# the offers per GPU (pod-session.sh): the volume's data center with the volume, then any data center without one
[[ -z "$NETWORK_VOLUME" ]] || offers+=("SECURE|$VOLUME_DC|$NETWORK_VOLUME")
for c in $HF_CLOUDS; do
  [[ "$c" == SECURE || "$c" == COMMUNITY ]] || { echo "e07.sh: HF_CLOUDS='$HF_CLOUDS': SECURE and/or COMMUNITY" >&2; exit 2; }
  offers+=("$c|any|")
done
(( ${#offers[@]} )) || { echo "e07.sh: no offer: NETWORK_VOLUME and HF_CLOUDS are both empty" >&2; exit 2; }

# e07b's dry runs with DRY_ENGINE on the real layout M and placement PLACE (dry_e07b M PLACE): the free-greedy and the
# isolated MoE runs, two-turn with its HTTP check (turn 1 the 16K prompt's text at ctx 20480) and the bench's phases
dry_e07b() {
  local m="$1" place="$2" d="$BUILD_ROOT/e07/e07b-dry"
  echo "--- e07b: dsv41-engine --dry-run of dense-free and moe-isolated (the oracle's prompt, 64 greedy steps)"
  # shellcheck disable=SC2086  # the placement is whole words
  "$DRY_ENGINE" --model "$m" --tokens-file "$BUILD_ROOT/e07/oracle-prompt.txt" --n-predict 64 --ctx 4096 --readers dense \
    $place --dry-run | grep -E "^(prompt:|dry run: [0-9]+ layers)"
  # shellcheck disable=SC2086
  "$DRY_ENGINE" --model "$m" --tokens-file "$BUILD_ROOT/e07/oracle-prompt.txt" --n-predict 64 --ctx 4096 --readers dense \
    $place --force-selection "$BUILD_ROOT/e07/oracle-forced" --forced-moe-input --dry-run | grep -E "^(prompt:|dry run: forced)"
  echo "--- e07b: engine/two_turn_check.py --dry-run --http (turn 1 the 16K prompt's text, ctx 20480, --chunk 16384)"
  mkdir -p "$d"
  python3 "$here/e07_prompts.py" turns "$PROMPTS" --vocab "$VOCAB" --out "$d/turns.json" | tail -n 1
  PYTHONDONTWRITEBYTECODE=1 python3 "$STRATA_DIR/tools/dsv41/engine/two_turn_check.py" --engine "$DRY_ENGINE" \
    --model "$m" --tokenizer-gguf "$VOCAB" --ctx 20480 --load-s 600 --engine-args "$place --chunk 16384" \
    --turns "$d/turns.json" --max-tokens 128 --http --out "$d/two-turn" --dry-run > "$d/two-turn.log" 2>&1 ||
    { tail -n 20 "$d/two-turn.log"; echo "e07.sh: the e07b two-turn dry run failed"; exit 1; }
  grep -E "^(two_turn_check.py|note http)" "$d/two-turn.log"
  python3 -c "import json, sys
j = json.load(open(sys.argv[1]))
t1, t2 = j['turns']
assert j['ok'] and j['http']['ok'] and t1['prompt_tokens'] >= 16384 and t2['cached_tokens'] > 0, j['failed']
assert t2['prompt_tokens'] + t2['completion_tokens'] <= 20480, j['turns']   # the dry stand-in replies are longer than 512
print(f\"two-turn dry: turn 1 {t1['prompt_tokens']} prompt tokens at ctx 20480, turn 2 {t2['cached_tokens']} cached of \"
      f\"{t2['prompt_tokens']}, its first token at {t2['first_token_s']} s; http non-streaming and streaming ok\")" \
    "$d/two-turn/two-turn.json"
  echo "--- e07b: pod/e07_bench.py --phases 16k-e09,turn2,short-adaptive,64k,256k against the serve dry run"
  python3 "$here/e07_bench.py" --engine "$DRY_ENGINE" --engine-args "--model $m --ctx 262144 --chunk 16384 $place --dry-run" \
    --prompts "$PROMPTS" --phases 16k-e09,turn2,short-adaptive,64k,256k --out "$d/bench-e07b.json" |
    grep -E "^e07_bench.py: (16k-e09|turn2|64k|256k|done|FAIL)"
  python3 -c "import json, sys
j = json.load(open(sys.argv[1]))
p = {x['phase']: x for x in j['phases']}
assert j['ok'] and list(p) == ['16k-e09', 'turn2', 'short-adaptive', '64k', '256k'], j.get('error')
assert p['turn2']['resume'] == 16415 and p['64k']['generated'] == 481 and p['256k']['generated'] == 481, p
print('bench-e07b dry: turn2 resumed 16415 ids after 16k-e09, 64k and 256k generated 481 each, the engine still answers')" \
    "$d/bench-e07b.json"
}

# the prompts: built once from the corpus and the real tokenizer, checked on every run
prompts_ready() {
  [[ -f "$PROMPTS/manifest.json" ]] ||
    python3 "$here/e07_prompts.py" build --fork "$FORK_DIR" --vocab "$VOCAB" --out "$PROMPTS" > /dev/null
  python3 "$here/e07_prompts.py" check "$PROMPTS"
}
# Part C: the users' conversations, built once locally from the real tokenizer
# (quality/chat_load.py build: 8 users of 16K tokens), and each N of the load steps (and ccache's alternating users)
# within them
cbase_ready() {
  [[ -f "$CBASE_CONVERSATIONS" ]] || { mkdir -p "$(dirname "$CBASE_CONVERSATIONS")" &&
    python3 "$STRATA_DIR/tools/dsv41/quality/chat_load.py" build --vocab "$VOCAB" --out "$CBASE_CONVERSATIONS" > /dev/null; }
  local have n
  have="$(python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['conversations']))" "$CBASE_CONVERSATIONS")"
  for n in $LOAD_USERS $([[ "$E07_PLAN" == ccache ]] && echo "$CCACHE_ALT_USERS"); do
    (( n <= have )) || { echo "e07.sh: $E07_PLAN has $n users, $CBASE_CONVERSATIONS has $have conversations" >&2; exit 2; }
  done
  # cbatch: the single user's short conversation
  (( ! has_cbatch )) || [[ -f "$CBATCH_SHORT" ]] || { mkdir -p "$(dirname "$CBATCH_SHORT")" &&
    python3 "$STRATA_DIR/tools/dsv41/quality/chat_load.py" build --vocab "$VOCAB" --users 1 --context 1024 --out "$CBATCH_SHORT" > /dev/null; }
}
# the plan's file of its load steps' users (e07/<name>-users): cbatch's in the combined plan
plan_users() { (( has_cbatch )) && echo cbatch || echo "$E07_PLAN"; }
# the users and the 64K user's conversations, built once locally
clong_ready() {
  [[ -f "$CLONG_CONVERSATIONS" ]] || { mkdir -p "$(dirname "$CLONG_CONVERSATIONS")" &&
    python3 "$STRATA_DIR/tools/dsv41/quality/chat_load.py" build --vocab "$VOCAB" --users "$CLONG_USERS" \
      --long-context "$CLONG_LONG_CONTEXT" --out "$CLONG_CONVERSATIONS" > /dev/null; }
  python3 -c "import json, sys; d = json.load(open(sys.argv[1]))
sys.exit(0 if d['long_context'] == int(sys.argv[3]) and len(d['conversations']) - 1 >= int(sys.argv[2]) else 1)" \
    "$CLONG_CONVERSATIONS" "$CLONG_USERS" "$CLONG_LONG_CONTEXT" ||
    { echo "e07.sh: $CLONG_CONVERSATIONS has no $CLONG_LONG_CONTEXT-token long user beside $CLONG_USERS users" >&2; exit 2; }
}
# Part Q: the needle prompts, built once locally from the real tokenizer (quality/needle_check.py)
needle_ready() {
  [[ -f "$QUALITY_NEEDLE" ]] || { mkdir -p "$(dirname "$QUALITY_NEEDLE")" &&
    python3 "$STRATA_DIR/tools/dsv41/quality/needle_check.py" build --vocab "$VOCAB" --out "$QUALITY_NEEDLE" > /dev/null; }
  mkdir -p "$BUILD_ROOT/quality"
}
# qdiag: the first QDIAG_ITEMS gsm8k_cot_llama items of the fixed-seed subsets, the same
# items in lm-eval-gsm8k-c1 and -c2 (qdiag_subsets FILE writes them)
QDIAG_ITEMS=20 QDIAG_SUBSETS_SRC="$STRATA_DIR/tools/dsv41/quality/subsets_lean2h_seed20261006.json"
qdiag_subsets() {
  python3 -c "import json, sys
d = json.load(open(sys.argv[1]))
json.dump({'gsm8k_cot_llama': d['gsm8k_cot_llama'][:int(sys.argv[3])]}, open(sys.argv[2], 'w'))" \
    "$QDIAG_SUBSETS_SRC" "$1" "$QDIAG_ITEMS"
}
# Part Q (the quality pre-flight): the commits HEAD must hold, serve/server.py's continuation of a final assistant
# message on the deepseek-v4.1 profile and api_proxy.py's cut of the fork's echo of it
Q_CONTINUATION_COMMIT=2c43875 Q_ECHO_CUT_COMMIT=9ce8dc5
# q_payload_sources: each Part Q payload item with its source commit or sha256. Exit 1 when a shipped path of the
# working tree (src/, the pod scripts) is not HEAD's, HEAD lacks one of those commits, the needle prompts are not what
# needle_check.py at HEAD builds (qdiag runs no needle: its derived GSM8K subsets instead; qdet neither; cbase and
# ccache their conversations, which must be what chat_load.py at HEAD builds), qengine's, qdiag's, qdet's, cbase's or
# ccache's engine sources
# (src/, include/, CMakeLists.txt and the engine binaries' own sources under tools/dsv41/engine/) changed since its
# tarball's commit, or a fork tarball is not the fork repo's HEAD
Q_ENGINE_SRC=(src include CMakeLists.txt tools/dsv41/engine/dsv41-engine.cpp tools/dsv41/engine/dsv41-layer-ref.cpp)
q_payload_sources() {
  local g=(git -C "$STRATA_DIR") dirty p c n top m commit sha ran
  dirty="$("${g[@]}" status --porcelain --ignored --untracked-files=all -- "${src[@]}" "${pod_files[@]/#/tools/dsv41/pod/}" \
    tools/dsv41/pod/e07_chats.json | grep -vE '(^|/)__pycache__(/|$)|(^|/)test_[^/]*\.py$' || true)"
  [[ -z "$dirty" ]] || { printf 'e07.sh: the payload ships these paths from the working tree, which differ from HEAD (commit or stash them):\n%s\n' "$dirty" >&2; exit 1; }
  for c in "$Q_CONTINUATION_COMMIT:serve/server.py's continuation" "$Q_ECHO_CUT_COMMIT:api_proxy.py's echo cut"; do
    # a history that starts after these commits (a squashed or public copy) has them in its first tree: no check
    if ! "${g[@]}" cat-file -e "${c%%:*}^{commit}" 2>/dev/null; then
      echo "  (${c%%:*}, ${c#*:}, is older than this repository's history: its change is in the tree)"
      continue
    fi
    "${g[@]}" merge-base --is-ancestor "${c%%:*}" HEAD 2>/dev/null ||
      { echo "e07.sh: HEAD does not contain ${c%%:*} (${c#*:})" >&2; exit 1; }
  done
  echo "  src/ and ${pod_files[*]} from Strata HEAD $("${g[@]}" rev-parse HEAD) ($("${g[@]}" rev-parse --abbrev-ref HEAD)),"
  echo "    their working tree paths clean; HEAD contains $Q_CONTINUATION_COMMIT (the continuation) and $Q_ECHO_CUT_COMMIT (the echo cut)"
  for p in serve/server.py tools/dsv41/quality tools/dsv41/quality/api_proxy.py tools/dsv41/quality/eval.sh \
           tools/dsv41/quality/pod_server.sh tools/dsv41/quality/subsets_lean2h_seed20261006.json; do
    sha=""
    [[ -d "$STRATA_DIR/$p" ]] || sha=", sha256 $(sha256sum "$STRATA_DIR/$p" | cut -d' ' -f1)"
    echo "  src/$p: last changed in $("${g[@]}" log -1 --format='%h %s' -- "$p")$sha"
  done
  if [[ "$E07_PLAN" == qdiag ]]; then
    n="$(mktemp)"
    qdiag_subsets "$n"
    echo "  e07/subsets-qdiag.json: sha256 $(sha256sum "$n" | cut -d' ' -f1), the first $QDIAG_ITEMS gsm8k_cot_llama items of"
    echo "    src/tools/dsv41/quality/subsets_lean2h_seed20261006.json (sha256 $(sha256sum "$QDIAG_SUBSETS_SRC" | cut -d' ' -f1))"
    rm -f "$n"
  elif [[ "$E07_PLAN" == qdet ]]; then
    echo "  src/tools/dsv41/quality/det_sampling_check.py: last changed in $("${g[@]}" log -1 --format='%h %s' -- \
      tools/dsv41/quality/det_sampling_check.py), sha256 $(sha256sum "$STRATA_DIR/tools/dsv41/quality/det_sampling_check.py" | cut -d' ' -f1)"
    echo "  (qdet ships no needle prompts nor lm-eval subsets)"
  elif (( pc )); then
    for p in tools/dsv41/quality/chat_load.py \
             $([[ "$E07_PLAN" == ccache ]] || (( has_cbatch )) && [[ "$E07_PLAN" != cfinal ]] && echo tools/dsv41/quality/det_sampling_check.py) \
             $( (( has_clong )) && echo tools/dsv41/quality/robust_check.py) \
             $( (( has_cdet )) && echo tools/dsv41/quality/det_batch_check.py) $( (( has_c05 )) && echo tools/dsv41/quality/run_lm_eval.sh); do
      echo "  src/$p: last changed in $("${g[@]}" log -1 --format='%h %s' -- "$p"), sha256 $(sha256sum "$STRATA_DIR/$p" | cut -d' ' -f1)"
    done
    n="$(mktemp)"
    python3 "$STRATA_DIR/tools/dsv41/quality/chat_load.py" build --vocab "$VOCAB" --out "$n" > /dev/null
    sha="$(sha256sum "$CBASE_CONVERSATIONS" | cut -d' ' -f1)"
    [[ "$(sha256sum "$n" | cut -d' ' -f1)" == "$sha" ]] ||
      { rm -f "$n"; echo "e07.sh: $CBASE_CONVERSATIONS is not what chat_load.py at HEAD builds from $VOCAB: remove it to rebuild" >&2; exit 1; }
    rm -f "$n"
    echo "  e07/conversations.json: $CBASE_CONVERSATIONS sha256 $sha, what chat_load.py at HEAD builds from $VOCAB"
    if (( has_cbatch )); then
      n="$(mktemp)"
      python3 "$STRATA_DIR/tools/dsv41/quality/chat_load.py" build --vocab "$VOCAB" --users 1 --context 1024 --out "$n" > /dev/null
      sha="$(sha256sum "$CBATCH_SHORT" | cut -d' ' -f1)"
      [[ "$(sha256sum "$n" | cut -d' ' -f1)" == "$sha" ]] ||
        { rm -f "$n"; echo "e07.sh: $CBATCH_SHORT is not what chat_load.py at HEAD builds (--users 1 --context 1024) from $VOCAB: remove it to rebuild" >&2; exit 1; }
      rm -f "$n"
      echo "  e07/conversations-short.json: $CBATCH_SHORT sha256 $sha, chat_load.py build --users 1 --context 1024 at HEAD"
    fi
    if (( has_clong )); then
      n="$(mktemp)"
      python3 "$STRATA_DIR/tools/dsv41/quality/chat_load.py" build --vocab "$VOCAB" --users "$CLONG_USERS" \
        --long-context "$CLONG_LONG_CONTEXT" --out "$n" > /dev/null
      sha="$(sha256sum "$CLONG_CONVERSATIONS" | cut -d' ' -f1)"
      [[ "$(sha256sum "$n" | cut -d' ' -f1)" == "$sha" ]] ||
        { rm -f "$n"; echo "e07.sh: $CLONG_CONVERSATIONS is not what chat_load.py at HEAD builds (--users $CLONG_USERS --long-context $CLONG_LONG_CONTEXT) from $VOCAB: remove it to rebuild" >&2; exit 1; }
      rm -f "$n"
      echo "  e07/conversations-long.json: $CLONG_CONVERSATIONS sha256 $sha, chat_load.py build --users $CLONG_USERS --long-context $CLONG_LONG_CONTEXT at HEAD"
    fi
    (( has_c05 )) && echo "  ($E07_PLAN ships no needle prompts; its lm-eval subsets are src/tools/dsv41/quality/subsets_lean2h_seed20261006.json, the reference run's)" ||
      echo "  ($E07_PLAN ships no needle prompts nor lm-eval subsets)"
  else
    n="$(mktemp)"
    python3 "$STRATA_DIR/tools/dsv41/quality/needle_check.py" build --vocab "$VOCAB" --out "$n" > /dev/null
    sha="$(sha256sum "$QUALITY_NEEDLE" | cut -d' ' -f1)"
    [[ "$(sha256sum "$n" | cut -d' ' -f1)" == "$sha" ]] ||
      { rm -f "$n"; echo "e07.sh: $QUALITY_NEEDLE is not what needle_check.py at HEAD builds from $VOCAB: remove it to rebuild" >&2; exit 1; }
    rm -f "$n"
    echo "  e07/needle.json: $QUALITY_NEEDLE sha256 $sha, what needle_check.py at HEAD builds from $VOCAB"
  fi
  for t in "${tags[@]}"; do
    top="$(basename "${strata_tgz[$t]}" .tar.gz)"
    m="$(tar -xzOf "${strata_tgz[$t]}" "$top/MANIFEST")"
    commit="$(awk '$1 == "commit:" { print $2 }' <<<"$m")" sha="$(awk '$2 == "bin/dsv41-engine" { print $1 }' <<<"$m")"
    # the session sheets whose pod ran this binary (their env.log holds the payload's MANIFEST)
    ran="$(grep -lF "$sha  bin/dsv41-engine" "$SHEET_ROOT"/*/logs/env.log 2>/dev/null | awk -F/ '{ print $(NF - 2) }' | paste -sd' ' || true)"
    echo "  payload-$t dsv41-engine: sha256 $sha from $top (Strata $commit, $(awk '$1 == "built_utc:" { print $2 }' <<<"$m"),"
    echo "    tracked_changes $(awk '$1 == "tracked_changes:" { print $2 }' <<<"$m")); run by the sessions ${ran:-none under $SHEET_ROOT}"
    if "${g[@]}" diff --quiet "$commit" HEAD -- "${Q_ENGINE_SRC[@]}"; then
      echo "    no change in ${Q_ENGINE_SRC[*]} from $commit to HEAD: the tarball's engine is HEAD's"
    elif [[ "$E07_PLAN" == qengine || "$E07_PLAN" == qdiag || "$E07_PLAN" == qdet ]] || (( pc )); then
      echo "e07.sh: the engine's sources changed from $commit to HEAD (git diff --stat $commit HEAD -- ${Q_ENGINE_SRC[*]}): $top is not HEAD's engine: rebuild it (tools/dsv41/build.sh)" >&2
      exit 1
    else
      echo "    the engine's sources changed from $commit to HEAD (qfork runs no engine: only its CUDA probe is used)"
    fi
    top="$(basename "${fork_tgz[$t]}" .tar.gz)"
    m="$(tar -xzOf "${fork_tgz[$t]}" "$top/MANIFEST")"
    commit="$(awk '$1 == "commit:" { print $2 }' <<<"$m")"
    [[ "$commit" == "$(git -C "$FORK_DIR" rev-parse HEAD)" ]] ||
      { echo "e07.sh: $top is llama-dsv41 $commit, not $FORK_DIR's HEAD $(git -C "$FORK_DIR" rev-parse HEAD): rebuild it (tools/dsv41/build.sh)" >&2; exit 1; }
    echo "  payload-$t fork/: from $top (llama-dsv41 $commit, $FORK_DIR's HEAD, tracked_changes $(awk '$1 == "tracked_changes:" { print $2 }' <<<"$m"))"
    for p in $(fork_libs_of "$t"); do
      case "$p" in llama-server|libllama.so.0) echo "    $p: sha256 $(awk -v f="bin/$p" '$2 == f { print $1 }' <<<"$m")" ;; esac
    done
  done
}
# the HF manifest (the shards' names, sizes and LFS sha256 at the pinned commit): from the HF API once, then its
# names and sizes checked against the volume's (gguf_layout.py's SHARD_BYTES, measured on a volume copy)
hf_manifest_ready() {
  [[ -f "$HF_MANIFEST" ]] || python3 "$here/hf_model.py" manifest --repo "$HF_REPO" --revision "$HF_REVISION" \
    --pattern "DeepSeek-V4.1-Flash-Q2_K-*-of-00007.gguf" --count 7 --out "$HF_MANIFEST" > /dev/null
  python3 - "$HF_MANIFEST" "$STRATA_DIR/tools/dsv41/engine" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[2])
from gguf_layout import SHARD_BYTES, SHARD_NAME
m = json.load(open(sys.argv[1]))
want = [(SHARD_NAME.format(i + 1), n) for i, n in enumerate(SHARD_BYTES)]
if [(f["name"], f["size"]) for f in m["files"]] != want:
    sys.exit(f"e07.sh: {sys.argv[1]} does not list the volume's 7 shards at their sizes")
for f in m["files"]:
    print(f"  {f['name']} {f['size']} sha256 {f['sha256']}")
print(f"  {m['repo']}@{m['revision']}: {sum(f['size'] for f in m['files']) / (1 << 30):.1f} GiB, the volume's names and sizes")
EOF
}

# the pod's steps and their estimates in minutes, per model path, from session 1's steps.log (2026-10-08, A100
# SXM4, the model from HF): the create to the run's start
# 1.2, an engine load 4.0-4.7 (257-283 s), each correctness run 4.9-5.8, the HF download 17.4 with its checks (264.5
# GB at 0.263 GB/s), tokenizer 5.0, bench-a 6.1, serve 10.4, bench-b 8.7; the volume's copy 4.5 (4 min 17 s measured).
# Unmeasured: dsv41-layer-ref (three CPU runs of two layers, guessed 1 min each), the timed decode phase (64 steps,
# 0.2) and the MMQ run, which only runs when layer-parity's reading asks for it (counted, the worst case)
plan_head=("create, SSH, CUDA probe, qualification (session 1: 0.2):1" "payload upload (the pod's flavor) and start (session 1: 1.0):1.5"
      "env:0.1")
plan_tail=()
[[ "$E07_PLAN" == qfork ]] || plan_tail+=("dry-run:0.1")
(( bench_only )) || plan_tail+=("forced-ref:0.1")
(( bench_only )) && E07_RUN_READERS="" || E07_RUN_READERS="$E07_READERS"
for mode in $E07_RUN_READERS; do
  plan_tail+=("$mode-forced (load + 65 calls with dumps; session 1 $([[ $mode == dense ]] && echo 5.0 || echo 5.8)):5.8"
              "parity-$mode-forced:0.3" "$mode-free:5.5" "parity-$mode-free:0.1")
done
if [[ "$E07_PLAN" == qdiag ]]; then
  # from measured numbers: the engine's load 78 s (the second e07b session, RTX PRO 6000 WS); the slow qengine
  # session (2026-10-12, RTX PRO 6000 WS): the server's start 70 s, the continuation step 16 s, a GSM8K item 1080 prompt
  # tokens read in 4.1 s and ~100 generated at its worst 15 tok/s; the bench's 2 x 480 decode steps at that rate too,
  # plus the hot set's resets, the 128-token prompts and the engine's stop (0.6, allowed); the FIFO server does the
  # same work at NUM_CONCURRENT 2 as at 1
  qdiag_gsm8k="$(python3 -c "print(round($QDIAG_ITEMS * (4.1 + 100 / 15) / 60 + 0.5, 1))")"
  plan_tail+=("bench-qdiag (load 1.3, short-static and short-adaptive: 480 decode steps each at 15 tok/s):3"
              "engine-server (serve/server.py on dsv41-engine; e07b's load 78 s, the slow qengine session's step 70 s):1.5"
              "quality-venv (the lm-eval venv, installed in the background since setup):0.5"
              "continuation (3 HumanEval items; the slow qengine session's step 16 s):0.5"
              "speed-check (speed_check.py on the continuation's requests, at most $Q_SPEED_MAX_MS ms per decode step):0.1"
              "lm-eval-gsm8k-c1 ($QDIAG_ITEMS items, NUM_CONCURRENT 1: 4.1 s prompt + 100 at 15 tok/s each, + 0.5):$qdiag_gsm8k"
              "lm-eval-gsm8k-c2 (the same $QDIAG_ITEMS items, NUM_CONCURRENT 2, the FIFO server's same work):$qdiag_gsm8k"
              "server-stop:0.5")
elif [[ "$E07_PLAN" == qdet ]]; then
  # from measured numbers (the 2026-10-13 qengine session, RTX PRO 6000): the
  # engine's load ~6 min (planned with margin; that session's engine-server step 110 s, its engine "loaded in 108929
  # ms"), ~55 tok/s of decode under lm-eval; det-sampling's 13 answered requests of at most 256 tokens are 3328 tokens,
  # 61 s at 55 tok/s, their prompts under 100 tokens each
  # det-sampling from the slower measured run, the 2026-10-13 qdet session's on the A100 (14 requests in 115 s,
  # 8.2 s each): 26 answered requests (13 in the mode, 7 with it off, 6 for its cost) are 213 s, plus the mode's resets
  # (the e08 and e09 benches' hot_reset=1, A100-SXM4 and RTX PRO 6000: 0.6 to 2.0 s each, 2.0 s for 19 requests in the
  # mode: 38 s) and its waits (20% of the decode: 43 s), 294 s
  plan_tail+=("engine-server (serve/server.py on dsv41-engine with --log-ids; the engine's load ~6 min, measured 110 s):6"
              "det-sampling (det_sampling_check.py: 26 requests at 8.2 s, the A100's, 213 s, + resets 38 + waits 43 s):5"
              "server-stop:0.5")
elif [[ "$E07_PLAN" == cbase ]]; then
  # the worst measured rates on the RTX PRO 6000: the engine's load ~6 min (the 2026-10-13
  # qengine session); a request reads at most 16384 + 2 x 512 + 256 = 17664 prompt tokens (no reuse) at 3571.5 tok/s
  # (the e09 session's slower 16K prompt, 4.59 s), the deterministic mode's reset 2.0 s and 256 tokens
  # at 35.4 tok/s (the e09 session's decode after the 16K prompt): 14.2 s; the FIFO server answers the N users' 2 turns one at a
  # time, plus 0.2 min for the client's start and its stragglers
  cbase_ready
  cbase_req_s="$(python3 -c "print(round(17664 / 3571.5 + 2.0 + 256 / 35.4, 1))")"
  plan_tail+=("engine-server (serve/server.py on dsv41-engine; the engine's load ~6 min, measured 110 s):6")
  for n in $CBASE_USERS; do
    plan_tail+=("load-c$n (chat_load.py, N = $n: 2 turns each, $(( 2 * n )) requests one at a time at $cbase_req_s s, + 0.2):$(
      python3 -c "print(round(2 * $n * $cbase_req_s / 60 + 0.2, 1))")")
  done
  plan_tail+=("server-stop:0.5")
elif [[ "$E07_PLAN" == ccache ]]; then
  # cbase's rates (a request read whole at 14.2 s, the deterministic mode's
  # reset 2.0 s in each), the alternating users' requests one after the other: each user's turn 1 read whole, its
  # turns 2+ resumed from its checkpoint (what the engine's checkpoints do, c02-test.sh), reading at most chat_load.py's
  # 512 new tokens at the same 3571.5 tok/s with the same reset and decode (the 16K checkpoint's 55.1 MiB restore left
  # out: milliseconds at PCIe rates, not measured yet); the second alternating run the same
  # with CCACHE_THINK_S between its requests. Every request read whole (no resume at all) still ends within each
  # alternating step's timeout (2 x the estimate), and the dry run reports that worst case's plan against the cap
  # (ccache_whole_min, the minutes it adds). det-sampling as qdet's (26 requests, the
  # A100's 8.2 s each, its resets and waits), the server's stop at 60 s (the engine's shutdown 49-52 s in qdet and cbase,
  # cbase's 30 s plan timed out)
  cbase_ready
  cbase_req_s="$(python3 -c "print(round(17664 / 3571.5 + 2.0 + 256 / 35.4, 1))")"
  ccache_resume_s="$(python3 -c "print(round(512 / 3571.5 + 2.0 + 256 / 35.4, 1))")"
  alt_n=$(( CCACHE_ALT_USERS * CCACHE_ALT_TURNS ))
  alt_s="$(python3 -c "print($CCACHE_ALT_USERS * $cbase_req_s + ($alt_n - $CCACHE_ALT_USERS) * $ccache_resume_s)")"
  # the two alternating steps' minutes with every request read whole, less their resumed estimates
  ccache_whole_min="$(python3 -c "w, r, g = $alt_n * $cbase_req_s, $alt_s, ($alt_n - 1) * $CCACHE_THINK_S
print(round(round(w / 60 + 0.2, 1) - round(r / 60 + 0.2, 1) + round((w + g) / 60 + 0.2, 1) - round((r + g) / 60 + 0.2, 1), 1))")"
  alt_what="$CCACHE_ALT_USERS users x $CCACHE_ALT_TURNS turns: $alt_n requests one after the other, $CCACHE_ALT_USERS turns 1 at $cbase_req_s s, $(( alt_n - CCACHE_ALT_USERS )) resumed at $ccache_resume_s s"
  plan_tail+=("engine-server (serve/server.py on dsv41-engine with --log-ids; the engine's load ~6 min, measured 110 s):6"
              "load-alt$CCACHE_ALT_USERS (chat_load.py --alternate, $alt_what, + 0.2):$(python3 -c "print(round($alt_s / 60 + 0.2, 1))")"
              "load-alt$CCACHE_ALT_USERS-think$CCACHE_THINK_S (the same with $CCACHE_THINK_S s between requests, $(( alt_n - 1 )) x $CCACHE_THINK_S s more, + 0.2):$(
                python3 -c "print(round(($alt_s + ($alt_n - 1) * $CCACHE_THINK_S) / 60 + 0.2, 1))")")
  for n in $CCACHE_USERS; do
    plan_tail+=("load-c$n (chat_load.py, N = $n: 2 turns each, $(( 2 * n )) requests one at a time at $cbase_req_s s, + 0.2):$(
      python3 -c "print(round(2 * $n * $cbase_req_s / 60 + 0.2, 1))")")
  done
  plan_tail+=("det-sampling (det_sampling_check.py: 26 requests at 8.2 s, the A100's, 213 s, + resets 38 + waits 43 s):5"
              "server-stop (the engine's shutdown, 49-52 s measured):1")
elif (( has_cbatch )); then
  # cbase's rates for the 16K load steps, a request read whole at 14.2 s one at a
  # time (the batch's worst case: no speed-up at all; the timeouts 2 x that); the short user's 2 turns of a prompt of
  # about 1K tokens (0.3 s at 3571.5 tok/s) and 256 tokens at 35.4 tok/s with the 2.0 s reset, + 0.2 min; the engine's load ~6 min for each
  # server, its stop 60 s; det-sampling as qdet's
  cbase_ready
  cbase_req_s="$(python3 -c "print(round(17664 / 3571.5 + 2.0 + 256 / 35.4, 1))")"
  short_min="$(python3 -c "print(round($CBATCH_SHORT_TURNS * (1024 / 3571.5 + 2.0 + 256 / 35.4) / 60 + 0.2, 1))")"
  # cdet's (and cfinal's) server loads from the cbatch sessions' measured steps (85-140 s), the others' the
  # planned ~6 min
  (( has_c05 )) && srv="85-140 s measured in the cbatch sessions:2.5" || srv="load ~6 min:6"
  # cdet's speed A/B replaces the batching-off server; cfinal has
  # neither (steady-c<N>'s base is the single user on the batched server, as in cdet)
  (( has_c05 )) ||
    plan_tail+=("engine-server-off (the FIFO server, gate-off's placement; the engine's ${srv%:*}):${srv##*:}"
                "load-short-off (chat_load.py, 1 user, a 1K prompt, $CBATCH_SHORT_TURNS turns of 256 at 35.4 tok/s, + 0.2):$short_min"
                "server-stop-off (the engine's shutdown, 49-52 s measured):1")
  plan_tail+=("engine-server (--batch $CBATCH_SLOTS --batch-ctx $CBATCH_CTX --checkpoints $CBATCH_CHECKPOINTS, gate's placement, --log-ids; ${srv%:*}):${srv##*:}")
  if (( has_c05 )); then
    plan_tail+=("load-short (chat_load.py, 1 user, a 1K prompt, $CBATCH_SHORT_TURNS turns of 256 at 35.4 tok/s, + 0.2: item 11's single user):$short_min")
    (( ! has_cdet )) ||
      plan_tail+=("ab-gpu (the same user, the deterministic mode off: the speed A/B's CPU numerics gpu):$short_min")
  else
    plan_tail+=("load-short (the same user with batching on):$short_min")
  fi
  # the combined plan leaves out the cbatch steps measured on 2026-10-14 (load-c<N>,
  # det-sampling) and keeps the baselines and steady-c<N>;
  # cdet leaves out load-c<N> too
  (( ! cbatch_loads )) || for n in $CBATCH_USERS; do
    plan_tail+=("load-c$n (chat_load.py, N = $n decoding at once: 2 turns each, at worst $(( 2 * n )) requests one at a time at $cbase_req_s s, + 0.2):$(
      python3 -c "print(round(2 * $n * $cbase_req_s / 60 + 0.2, 1))")")
  done
  # each repeat N 16K prompts read one at a time (4.9 s each at 3571.5 tok/s), then the replies of
  # CBATCH_STEADY_MAX_TOKENS at 20 tok/s a user (item 11's floor: the window outlasts slower users), the 2.0 s reset
  sn=$CBATCH_STEADY_USERS
  steady_min="$(python3 -c "print(round($CBATCH_STEADY_REPEATS * ($sn * 17664 / 3571.5 + $CBATCH_STEADY_MAX_TOKENS / 20 + 2.0) / 60 + 0.2, 1))")"
  plan_tail+=("steady-c$sn (chat_load.py steady, $CBATCH_STEADY_REPEATS x: $sn prompts at 4.9 s, $CBATCH_STEADY_MAX_TOKENS tokens at 20 tok/s, the reset 2.0 s; the $CBATCH_STEADY_WINDOW_S s window inside, + 0.2):$steady_min")
  # the same with CBATCH_STEADY_REPORT's users, reported
  for n in $CBATCH_STEADY_REPORT; do
    plan_tail+=("steady-c$n (the same with $n users, reported: $CBATCH_STEADY_REPEATS x $n prompts at 4.9 s, $CBATCH_STEADY_MAX_TOKENS tokens at 20 tok/s, the reset 2.0 s, + 0.2):$(
      python3 -c "print(round($CBATCH_STEADY_REPEATS * ($n * 17664 / 3571.5 + $CBATCH_STEADY_MAX_TOKENS / 20 + 2.0) / 60 + 0.2, 1))")")
  done
  # item 14 on the batched server; its alone runs (A and the 4 different
  # prompts) each the 2.0 s reset and 256 tokens at 35.4 tok/s, its 3 batched phases each 256 tokens at item 11's
  # floor of 20 tok/s a user with the reset, + 0.2
  (( ! has_cdet )) ||
    plan_tail+=("det-batch (det_batch_check.py: 5 requests alone, then 4 identical together twice, streamed and not, and 4 different together):$(
      python3 -c "print(round((5 * (2.0 + 256 / 35.4) + 3 * (2.0 + 256 / 20)) / 60 + 0.2, 1))")")
  if (( has_cdet )); then
    # 455 s in the 2026-10-17 cdet session, 300 of them its frequency_penalty request's wait for a DONE
    # after the engine's ERR (fixed in serve 09b390e): 155 s
    plan_tail+=("det-sampling (det_sampling_check.py; 155 s measured in the 2026-10-17 cdet session less the 300 s ERR wait fixed since):3")
  elif (( ! has_clong && ! has_c05 )); then
    plan_tail+=("det-sampling (det_sampling_check.py: 26 requests at 8.2 s, the A100's, 213 s, + resets 38 + waits 43 s):5")
  fi
  # item 3 batched: each task as measured on the batch slots in the 2026-10-17 cdet session (header);
  # cfinal's from the fork's numerics' own runs (header)
  if (( has_c05 )); then
    plan_tail+=("quality-venv (the lm-eval venv, installed in the background since setup):0.5"
                "continuation (3 HumanEval items; the engine quality run's step 7 s):0.5")
    for t in $CDET_TASKS; do
      if (( has_cdet )); then
        case "$t" in
          gsm8k_cot_llama) m=459 how="459 s measured in the 2026-10-17 cdet session" ;;
          ifeval) m=1042 how="1042 s measured in the 2026-10-17 cdet session" ;;
          humaneval_instruct) m=359 how="the reference run's 359 s one at a time (cut at the cdet session's deadline; ifeval and gsm8k took 0.97 and 0.86 x the reference run's there)" ;;
        esac
      else
        case "$t" in
          gsm8k_cot_llama) m=531 how="the reference run's 531 s in the fork's numerics one at a time; the GPU's 909 s on 2026-10-19 within its timeout" ;;
          ifeval) m=1077 how="the reference run's 1077 s in the fork's numerics one at a time; the GPU's 1521 s on 2026-10-19 within its timeout" ;;
          humaneval_instruct) m=450 how="the reference run's 359 s in the fork's numerics one at a time x 1.25; the GPU's had 162 of 164 at its 720 s cut on 2026-10-19" ;;
        esac
      fi
      plan_tail+=("lm-eval-$t (NUM_CONCURRENT $CDET_CONCURRENT on the batch slots, $how):$(python3 -c "print(round($m / 60, 1))")")
    done
  fi
  plan_tail+=("server-stop (the engine's shutdown, 49-52 s measured):1")
  # the speed A/B's fork arm, last (the step cut first if time runs out)
  (( ! has_cdet )) ||
    plan_tail+=("engine-server-fork (the same server with --cpu-numerics fork; ${srv%:*}):${srv##*:}"
                "ab-fork (load-short's user, the deterministic mode off, the CPU experts in the fork's numerics):$short_min"
                "server-stop-fork (the engine's shutdown, 49-52 s measured):1")
elif [[ "$E07_PLAN" == clong ]]; then
  :   # its steps below, as the combined plan's after cbatch's
elif [[ "$E07_PLAN" == q* ]]; then
  # Part Q: the server, the venv, the continuation check, the tasks and the needles (quality/q_plan.py)
  needle_ready
  # each run's own file: two e07.sh at once (a dry run beside a session's payload) read each other's steps from a
  # shared one (q02-test.sh caught a dry run planning the other plan's needles)
  q_file="$(mktemp)"
  q_tokens="$(python3 "$STRATA_DIR/tools/dsv41/quality/q_plan.py" --plan "$E07_PLAN" --usage "$API_THEN_USAGE" \
    --speed-max-ms "$Q_SPEED_MAX_MS" --fork-max-ms "$Q_FORK_MAX_MS" --num-concurrent "$([[ "$E07_PLAN" == qfork ]] && echo "$Q_FORK_NP" || echo 2)" \
    --needle "$QUALITY_NEEDLE" --tasks "$Q_TASKS" --needles "$Q_NEEDLES" --prompt-tok-s "$Q_PROMPT_TOK_S" \
    --long-prompt-tok-s "$Q_LONG_PROMPT_TOK_S" --decode-tok-s "$Q_DECODE_TOK_S" 2>&1 > "$q_file")" ||
    { rm -f "$q_file"; echo "$q_tokens" >&2; exit 2; }
  mapfile -t q_steps < "$q_file"
  rm -f "$q_file"
  plan_tail+=("${q_steps[@]}")
elif [[ "$E07_PLAN" == e07b ]]; then
  # from the e09 session (2026-10-10, RTX PRO 6000 Blackwell WS): the engine's load 250 s, bench-e09 396 s (16K
  # twice, 480 short steps, the 256K prompt 79 s), the server's load ~150 s; turn2, 64K (16K's rate: ~20 s prompt,
  # 480 steps at ~50 tok/s) added to the bench; the correctness runs as the e07a session's (A100: dense-free 331 and 677 s,
  # moe-isolated 344 s, their parity 4 and 16 s)
  plan_tail+=("two-turn (the server's load, the 16K turn 1 and turn 2 of up to 128 tokens, then the HTTP check):5.5"
              "bench-e07b (load 4.2, 16K, turn 2, 480 short steps, 64K and 256K with 480 steps each):7.5"
              "dense-free (load + 65 calls, the oracle's prompt, free greedy, reported):6" "parity-dense-free:0.2"
              "forced-ref:0.1" "moe-isolated (load + 65 calls, the oracle's cur in every layer, reported):6"
              "parity-moe-isolated:0.3")
elif [[ "$E07_PLAN" == e09 ]]; then
  # from the e08 session (2026-10-09, A100-SXM4): the load 237 s, the 16K prompt 9.0 s (25 s with 481 steps), 480
  # short steps 15 s, the 256K prompt 148 s and 480 steps 15 s; the server's load as the bench's
  plan_tail+=("bench-e09 (load 4.0, the 16K prompt twice, 480 short steps, the 256K prompt):7.5"
              "two-turn (the server's load 4.0, two turns of up to 128 tokens):5.5")
elif [[ "$E07_PLAN" == e08 ]]; then
  # bench-a (load + 16K + turn 2 + 480 short steps + 64 timed) took 6.1 min and bench-b (load + 64K + 256K with 480
  # steps each) 8.7 in session 1: one load (4.7, the spare homes pinned too), 3 x 480 short steps (~0.35 each at the
  # measured 20-25 tok/s, with a reset before each), the 16K prompt and 480 steps (0.6), 64 timed (0.1), 64K and 256K
  # (8.7 - its load 4.5 = 4.2)
  plan_tail+=("bench-e08 (load, 3 x 480 short steps, 16K + 480, 64 timed, 64K and 256K + 480 each):11.5"
              "route-trace (gzip):0.2" "serve (a reference load, then the server's, 3 chats):10.4")
else
  plan_tail+=("layer-ref fork-cpu, fork-cuda, engine (CPU only, unmeasured):3" "layer-parity:0.5"
        "moe-isolated (load + 65 calls, the oracle's cur in every layer):5.8" "parity-moe-isolated:0.3"
        "dense-forced-mmq, its parity and layer-parity-mmq (only when layer-parity reads so):6.6")
  [[ "$E07_PLAN" == cycle2 ]] || plan_tail+=("tokenizer:5")
  plan_tail+=("bench-a (load, 16K, turn 2, 480 decode steps, then 64 timed):6.3")
  [[ "$E07_PLAN" == cycle2 ]] || plan_tail+=("serve (a reference load, then the server's, 3 chats):10.4"
        "bench-b (load, 64K, 256K with 480 decode steps each):8.7")
fi
# clong's steps (the combined plan's after cbatch's, their
# names with -long where cbatch's have the same). load-long: the 64K prompt alone at 25 s (measured
# ~20-25 s on the RTX PRO 6000), the 2.0 s reset and 16 tokens at 35.4 tok/s; then each user's 16K read one at a time at
# the e09 session's 3571.5 tok/s, 2 s, the 64K read again in chunks of the engine's prefill chunk (CLONG_CHUNK, 16384:
# the interleave chunk, pod_server.sh's --chunk) at a planned 4.65 s each: the warm tier streamed whole through the ring
# (a prefill chunk copies every expert outside the slab it touches; the gate's 113.13 GiB warm tier at the
# measured 26.12 GB/s host to device). The time is the host's: 3.2-3.3 s a 16K chunk beside 4
# users on the RTX PRO 6000 Server of 2026-10-15, the e09 session's card 4.59 s cold, the e07b session's slower Workstation card 7.68 s. Item
# 12's margin follows the host's pinned H2D: 4.02 s a 16K chunk and a 4.048 s gap at 28.6
# GB/s (PCIe Gen 4 x16, the clong session of 2026-10-16) against 3.29-3.31 s and 3.325-3.350 s at 55.1-57.7 GB/s on
# the Server machines; QUALIFY_MIN_H2D_GBS (18) does not bound it (not raised). The step's timeout
# is 2 x the estimate. Then a step of 50 ms after each chunk but the last (4 users at item 11's floor of 20 tok/s each),
# and the users' CLONG_MAX_TOKENS at 20 tok/s each, + 0.2 min. robust: robust_check.py's eight checks, 4 min (the
# 2026-10-15 sessions' seven took 20 s and 41 s; the later checks add a 69.5K-token prompt read alone, about 19
# s at the 3571.5 tok/s above, its ~160 tokens, a 40K prompt read beside 4 runners in 3 chunks of 16384 at 4.65 s and
# the 10 s after it, two turns reading ~300 new tokens each and 256 tokens alone: about 1.5 min more)
if (( has_clong )); then
  clong_ready
  sfx=""
  (( has_cbatch )) && sfx="-long"
  # the long prompt's tokens (its turn 1 with the chat template's, e.g. 65565) and the chunks they take
  long_tokens="$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['conversations'][-1]['turn1_prompt_tokens'])" \
    "$CLONG_CONVERSATIONS")"
  long_chunks=$(( (long_tokens + CLONG_CHUNK - 1) / CLONG_CHUNK ))
  long_min="$(python3 -c "solo = 25 + 2.0 + 16 / 35.4
users = $CLONG_USERS * (16414 / 3571.5) + 2 + $long_chunks * 113.13 * 2**30 / 26.12e9 + ($long_chunks - 1) * 0.05 + $CLONG_MAX_TOKENS / 20
print(round((solo + users) / 60 + 0.2, 1))")"
  plan_tail+=("engine-server$sfx (--batch $CLONG_SLOTS --batch-ctx $CLONG_CTX --checkpoints $CLONG_CHECKPOINTS, the long gate's placement, --log-ids; load ~6 min):6"
              "load-long (chat_load.py long: the 64K prompt alone 25 s, then $CLONG_USERS users and the $long_tokens-token prompt beside them in $long_chunks chunks of $CLONG_CHUNK at a planned 4.65 s, $CLONG_MAX_TOKENS tokens each at 20 tok/s, + 0.2):$long_min"
              "robust (robust_check.py: item 13's eight checks over HTTP, no-max-tokens and slot-room with them):4"
              "server-stop$sfx (the engine's shutdown, 49-52 s measured):1")
fi
# the second cbatch server last, after items 12 and 13, with cbatch's
# estimates: the single user with batching on and steady-c<N> on the engine with CBATCH_SPARE spare slots per layer
if (( has_spare )); then
  plan_tail+=("engine-server-spare (--hot-spare $CBATCH_SPARE, gate-spare's placement, --batch $CBATCH_SLOTS --batch-ctx $CBATCH_CTX --checkpoints $CBATCH_CHECKPOINTS, --log-ids; load ~6 min):6"
              "load-short-spare (the single user with batching on, on this engine):$short_min"
              "steady-c$sn-spare (steady-c$sn on this engine):$steady_min"
              "server-stop-spare (the engine's shutdown, 49-52 s measured):1")
fi
download="model-download (264.5 GB, then size and sha256 per file; session 1: 17.4):18"
# e07b and the Part Q plans: the download at HF_GBS, the sha256 check 0.6 min (2026-10-10:
# 32.4 s)
(( timed )) && download="model-download (264.5 GB at $HF_GBS GB/s, then size and sha256 per file):$(
  python3 -c "import sys; print(round(264.5 / float(sys.argv[1]) / 60 + 0.6, 1))" "$HF_GBS")"
# the steps the pod runs per model path (e07-run.sh's names first in each label), between the session's own
# volume-before, volume-after and setup take at least 0.5 min, so their timeout (2 x) is at least 60 s
listing=0.1
(( timed )) && listing=0.5
# qfork runs no engine: no gate, so no model headers before the download either
gate_step=("gate:0.2") headers_step=("model-headers (7 heads by HTTP range, 112 MiB):0.5")
(( ! has_cbatch || has_c05 )) || gate_step+=("gate-off (the same placement without the batch: today's slab):0.2")
[[ "$E07_PLAN" != cbatch-clong ]] || gate_step+=("gate-long (the long prompt's slots: --batch $CLONG_SLOTS --batch-ctx $CLONG_CTX):0.2")
(( ! has_spare )) || gate_step+=("gate-spare (gate's batch with --hot-spare $CBATCH_SPARE: the slab leaves their room):0.2")
[[ "$E07_PLAN" == qfork ]] && gate_step=() headers_step=()
pod_volume=("volume-before:$listing" "${gate_step[@]}" "setup:0.5" "model-copy:4.5" "${plan_tail[@]}" "volume-after:$listing")
pod_hf=("${headers_step[@]}" "${gate_step[@]}" "setup (and aria2):0.7" "$download" "${plan_tail[@]}")
plan_volume=("${plan_head[@]}" "${pod_volume[@]}" "pull and delete:1.5")
plan_hf=("${plan_head[@]}" "${pod_hf[@]}" "pull and delete:1.5")
plan_total() { local s t=0; for s in "$@"; do t="$(python3 -c "print(round($t + ${s##*:}, 1))")"; done; echo "$t"; }
# estimates PATH: the pod's "NAME SECONDS" lines of a model path (pod-steps.sh's STEP_ESTIMATES: each step's timeout
# is min(2 x its estimate, the time left))
estimates() {
  local -n st="pod_$1"
  local e
  for e in "${st[@]}"; do
    python3 -c "import sys; print(sys.argv[1], round(float(sys.argv[2]) * 60))" "${e%%[ :]*}" "${e##*:}"
  done
}

if (( dry )); then
  echo "e07.sh --dry-run: nothing is created, uploaded or run"
  echo "payloads, one per flavor, built before the create; the created pod's GPU takes its flavor's:"
  for t in "${tags[@]}"; do
    read -ra bins <<<"$(bins_of "$t")"
    read -ra fork_libs <<<"$(fork_libs_of "$t")"
    # every file the payload takes is in its tarball (the session checks each one's sha256 against the MANIFEST)
    for pair in "${strata_tgz[$t]}:${bins[*]}" "${fork_tgz[$t]}:${fork_libs[*]}"; do
      tgz="${pair%%:*}" top="$(basename "${pair%%:*}" .tar.gz)" listing="$(tar -tzf "${pair%%:*}")"
      for b in ${pair#*:} MANIFEST; do
        grep -qx "$top/\(bin/\)\?$b" <<<"$listing" || { echo "e07.sh: $tgz has no $b" >&2; exit 1; }
      done
    done
    echo "  payload-$t: ${bins[*]} and MANIFEST from ${strata_tgz[$t]} (sha256 checked against its MANIFEST);"
    echo "    fork/: ${fork_libs[*]} from ${fork_tgz[$t]} (sha256 checked) and llama-tokenize from ${tokenize[$t]}"
    echo "    (the same build: its libllama.so.0.4.0 must be the tarball's libllama.so.0), libcublas.so.${cuda_major[$t]} and"
    echo "    libcublasLt.so.${cuda_major[$t]} linked to the engine's"
  done
  echo "  each with pod/e07-run.sh, pod/pod-steps.sh, pod/model-copy.sh, pod/t04a-lib.sh, pod/hf-download.sh,"
  echo "  pod/hf_model.py; e07/hf-manifest.json from $HF_MANIFEST; e07/counts.json from $COUNTS;"
  echo "  e07/oracle/ (the stock fork's oracle dump: $ORACLE, its manifest and cur/out/topk/weights); e07/prompts/ from $PROMPTS;"
  echo "  e07/plan ($E07_PLAN) and e07/readers ($E07_READERS);"
  echo "  e07/chats.json (pod/e07_chats.json); src/: ${src[*]} (no test_*.py, no __pycache__)"
  echo "--- the prompts (pod/e07_prompts.py check $PROMPTS)"
  prompts_ready
  echo "--- the HF manifest ($HF_MANIFEST: $HF_REPO at $HF_REVISION, from the HF API's tree; the pod without the volume checks every file against it)"
  hf_manifest_ready
  [[ "$E07_PLAN" != e07b ]] ||
    echo "  e07b only: e07/turns.json (pod/e07_prompts.py turns: the 16K prompt's text, a follow-up), e07/estimates-volume and -hf"
  if [[ "$E07_PLAN" == qdiag || "$E07_PLAN" == qdet ]] || (( pc )); then
    if [[ "$E07_PLAN" == qdiag ]]; then
      echo "  qdiag only: e07/subsets-qdiag.json (its sha256 and source below), e07/estimates-volume and -hf,"
      echo "    e07/speed-max-ms ($Q_SPEED_MAX_MS)"
    elif (( pc )); then
      echo "  $E07_PLAN only: e07/conversations.json from $CBASE_CONVERSATIONS, e07/$(plan_users)-users ($LOAD_USERS),$(
        [[ "$E07_PLAN" == ccache ]] && echo " e07/ccache-alt ($CCACHE_ALT_USERS users, $CCACHE_ALT_TURNS turns, think $CCACHE_THINK_S s),")$(
        (( has_cbatch )) && echo " e07/conversations-short.json from $CBATCH_SHORT, e07/cbatch-batch ($CBATCH_SLOTS slots of $CBATCH_CTX positions, $CBATCH_CHECKPOINTS checkpoints), e07/cbatch-steady ($CBATCH_STEADY_USERS users, $CBATCH_STEADY_WINDOW_S s, $CBATCH_STEADY_REPEATS repeats, $CBATCH_STEADY_MAX_TOKENS tokens), e07/cbatch-steady-report (${CBATCH_STEADY_REPORT:-none}), e07/cbatch-short ($CBATCH_SHORT_TURNS turns),")$(
        (( has_clong )) && echo " e07/conversations-long.json from $CLONG_CONVERSATIONS, e07/clong ($CLONG_SLOTS slots of $CLONG_CTX positions, $CLONG_CHECKPOINTS checkpoints, $CLONG_USERS users of $CLONG_MAX_TOKENS tokens),")$(
        (( has_spare )) && echo " e07/cbatch-spare ($CBATCH_SPARE spare slab slots per layer),")$(
        (( has_c05 )) && echo " e07/cdet-quality ($CDET_CONCURRENT concurrent requests, tasks $CDET_TASKS),")"
      echo "    e07/estimates-volume and -hf; src/tools/dsv41/quality/chat_load.py$( (( has_cdet )) && echo ", det_batch_check.py,")$( (( has_c05 )) &&
        echo " and lm-eval's run_lm_eval.sh, eval.sh, api_proxy.py and the fixed-seed subsets (no needle)" ||
        echo " (no needle, no subsets)")"
      python3 -c "import json, sys
for c in json.load(open(sys.argv[1]))['conversations']:
    print(f\"    conversation {c['name']}: {c['tokens']} tokens, turn 1 {c['turn1_prompt_tokens']} prompt tokens\")" \
        "$CBASE_CONVERSATIONS"
    else
      echo "  qdet only: e07/estimates-volume and -hf; src/tools/dsv41/quality/det_sampling_check.py (no needle, no subsets)"
    fi
    echo "--- the $E07_PLAN payload's sources (each item's commit or sha256; the run refuses what this refuses)"
    q_payload_sources
  elif [[ "$E07_PLAN" == q* ]]; then
    echo "  $E07_PLAN only: e07/needle.json from $QUALITY_NEEDLE (sha256 $(sha256sum "$QUALITY_NEEDLE" | cut -d' ' -f1)),"
    echo "    e07/qtasks ($Q_TASKS), e07/qneedles (${Q_NEEDLES:-none}), e07/estimates-volume and -hf$([[ "$E07_PLAN" == qengine ]] &&
      echo ", e07/speed-max-ms ($Q_SPEED_MAX_MS)" || echo ", e07/fork-np ($Q_FORK_NP), e07/fork-max-ms ($Q_FORK_MAX_MS)")"
    python3 -c "import json, sys
for p in json.load(open(sys.argv[1]))['prompts']:
    print(f\"    needle {p['name']}: {p['prompt_tokens']} prompt tokens, the number at depth {p['depth']}\")" "$QUALITY_NEEDLE"
    echo "--- the $E07_PLAN payload's sources (each item's commit or sha256; the run refuses what this refuses)"
    q_payload_sources
  fi
  case "$E07_PLAN" in e07b|qfork|qengine|qdiag|qdet|cbase|ccache|cbatch|clong|cbatch-clong|cdet|cfinal) sheet_name="$E07_PLAN" ;; *) sheet_name=e07a ;; esac
  echo "results: $OUT; session sheet: $SHEET_ROOT/<UTC date>-$sheet_name/"
  echo "GPUs, in order: $GPU_TYPES; host RAM at least $MIN_RAM_GB GB (minRAMPerGPU; a pod reporting less, or no memoryInGb,"
  echo "  deleted at once); per GPU the offers in order:"
  for offer in "${offers[@]}"; do
    IFS='|' read -r c dc vol <<<"$offer"
    if [[ -n "$vol" ]]; then echo "  $c in $dc with network volume $vol at /workspace (read only): the model copied from it"
    else echo "  $c in any data center, no network volume: the model downloaded from HF ($HF_REPO at $HF_REVISION)"; fi
  done
  echo "machines excluded by BAD_MACHINES (a pod landing on one is deleted at once):${env_bad% }$([[ "$env_bad" == " " ]] && echo " none")"
  echo "price limit: $MAX_USD_PER_HR USD/h; session cap: $MAX_SESSION_USD USD over at most $MAX_PODS pods, each min($CAP_MIN min, what is left / price), at the limit price $(cap_minutes "$MAX_USD_PER_HR") min"
  for g in "${gpus[@]}"; do
    echo "  '$g' takes payload-$(gpu_cfg "$g" FLAVOR_TAG) (hosts whose driver runs CUDA $(gpu_cfg "$g" POD_CUDA_VERSIONS))"
    priced=" "
    for offer in "${offers[@]}"; do
      IFS='|' read -r c dc vol <<<"$offer"
      [[ "$dc" == any ]] && dc=""
      [[ "$priced" == *" $c "* ]] ||
        echo "    GET $CATALOG_SHOWN?include=AVAILABILITY&product=POD&cloud=$c&count=1&cudaVersions=$(gpu_cfg "$g" POD_CUDA_VERSIONS) -> its price"
      priced+="$c "
      # shellcheck disable=SC2046  # create_opts prints whole words
      echo "    POST $REST_SHOWN/pods $(api body --gpu "$g" --cloud "$c" --dc "$dc" $(create_opts "$g" "$vol"))"
    done
  done
  # a past session's real cgroup and MemTotal (PROBE_LOG), else MIN_RAM_GB's
  gate_cg="$(( MIN_RAM_GB * 1000000000 ))" gate_mt="$(( MIN_RAM_GB * 1000000 ))" gate_src="the pod's cgroup taken as $MIN_RAM_GB GB"
  if [[ -n "${PROBE_LOG:-}" ]]; then
    probe_mem="$(python3 - "$PROBE_LOG" <<'PY'
import os, re, sys
log = sys.argv[1]
texts = [open(f, errors="replace").read() for f in (log, os.path.join(os.path.dirname(log), "out", "env.log"))
         if os.path.isfile(f)]
cg = [m for t in texts for m in re.findall(r"(?:/sys/fs/cgroup/memory\.max|memory\.limit_in_bytes): (\d+|max)\s*$", t, re.M)]
mt = [m for t in texts for m in re.findall(r"MemTotal:?\s+(\d+) kB", t)]
if not cg or not mt:
    sys.exit(f"e07.sh: PROBE_LOG={log}: no memory cgroup limit or MemTotal in it (nor in its out/env.log)")
print("unlimited" if cg[-1] == "max" else cg[-1], mt[-1])
PY
)" || exit 2
    read -r gate_cg gate_mt <<<"$probe_mem"
    gate_src="the pod's cgroup $gate_cg bytes and MemTotal $gate_mt kB from PROBE_LOG=$PROBE_LOG"
  fi
  echo "--- the gate per GPU (pod/e07_gate.py on the real headers $HEADERS, $gate_src)"
  # cbatch: the batch's slots and checkpoints in the gate (gate-off, the pod's second, is the same without them); clong:
  # the long prompt's slots (the combined plan's gate-long)
  CBATCH_GATE="" CLONG_GATE="--batch $CLONG_SLOTS --batch-ctx $CLONG_CTX --checkpoints $CLONG_CHECKPOINTS"
  (( ! has_cbatch )) || CBATCH_GATE="--batch $CBATCH_SLOTS --batch-ctx $CBATCH_CTX --checkpoints $CBATCH_CHECKPOINTS"
  [[ "$E07_PLAN" != clong ]] || CBATCH_GATE="$CLONG_GATE"
  layout="$BUILD_ROOT/e07/real-layout"
  [[ -f "$layout/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf" ]] ||
    python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" sparse "$HEADERS" "$layout" > /dev/null
  gate_failed=() first_gate="" gate_gpus=("${gpus[@]}")
  # qfork runs no engine, so no gate
  [[ "$E07_PLAN" != qfork ]] || { gate_gpus=(); echo "  none: E07_PLAN=qfork runs the fork's llama-server only"; }
  for g in "${gate_gpus[@]}"; do
    # nvidia-smi's memory.total: the RTX PRO 6000's 96 GB leave 97887 MiB usable
    case "$g" in *"RTX PRO 6000"*) mib=97887 ;; *A100*80GB*) mib=81920 ;; *L40S*) mib=46068 ;; *) mib=0 ;; esac
    (( mib )) || { echo "  $g: no memory.total known here; the pod's gate computes it"; continue; }
    echo "  $g ($mib MiB):"
    # shellcheck disable=SC2086  # the batch's gate arguments are whole words
    python3 "$here/e07_gate.py" --vram-mib "$mib" --prefetch-experts "$prefetch" --cgroup-bytes "$gate_cg" \
      --mem-total-kib "$gate_mt" --counts "$COUNTS" --layout "$STRATA_DIR/tools/dsv41/oracle/layout-dsv41-q2k.json" \
      --model-headers "$layout/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf" --out "$BUILD_ROOT/e07/gate-dry-$mib" \
      $CBATCH_GATE | sed 's/^/    /' || gate_failed+=("$g")
    # the combined plan: the long prompt's gate too (gate-long)
    if [[ "$E07_PLAN" == cbatch-clong ]]; then
      echo "    gate-long:"
      # shellcheck disable=SC2086
      python3 "$here/e07_gate.py" --vram-mib "$mib" --prefetch-experts "$prefetch" --cgroup-bytes "$gate_cg" \
        --mem-total-kib "$gate_mt" --counts "$COUNTS" --layout "$STRATA_DIR/tools/dsv41/oracle/layout-dsv41-q2k.json" \
        --model-headers "$layout/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf" --out "$BUILD_ROOT/e07/gate-dry-long-$mib" \
        $CLONG_GATE | sed 's/^/      /' || gate_failed+=("$g (gate-long)")
    fi
    # the second cbatch server's gate, its spare slots' slab cost
    if (( has_spare )); then
      echo "    gate-spare:"
      # shellcheck disable=SC2086
      python3 "$here/e07_gate.py" --vram-mib "$mib" --prefetch-experts "$prefetch" --cgroup-bytes "$gate_cg" \
        --mem-total-kib "$gate_mt" --counts "$COUNTS" --layout "$STRATA_DIR/tools/dsv41/oracle/layout-dsv41-q2k.json" \
        --model-headers "$layout/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf" --out "$BUILD_ROOT/e07/gate-dry-spare-$mib" \
        $CBATCH_GATE --hot-spare "$CBATCH_SPARE" | sed 's/^/      /' || gate_failed+=("$g (gate-spare)")
      python3 -c "import json, sys; a, b = (json.load(open(f)) for f in sys.argv[1:3])
print(f\"      the spare slots' cost: --hot-spare {b['hot_spare']} against {a['hot_spare']}, {b['spare_gib'] - a['spare_gib']:.2f} GiB \"
      f\"more of slab, {b['vram_experts']} slab experts against {a['vram_experts']} ({b['vram_experts'] - a['vram_experts']})\")" \
        "$BUILD_ROOT/e07/gate-dry-$mib/gate.json" "$BUILD_ROOT/e07/gate-dry-spare-$mib/gate.json" || gate_failed+=("$g (gate-spare)")
    fi
    # the host homes the engine would get there
    python3 -c "import json, sys; g = json.load(open(sys.argv[1]))
if g.get('host_homes_of'):
    print(f\"    expected host homes: {g['host_homes_held']} of the slab's {g['host_homes_of']} experts \"
          f\"({100 * g['host_homes_fraction']:.1f}%), dsv41-engine --hot-homes {g['hot_homes']}: {g['host_homes_reason']}\")" \
      "$BUILD_ROOT/e07/gate-dry-$mib/gate.json" 2>/dev/null || true
    [[ -n "$first_gate" ]] || first_gate="$g:$mib"
  done
  if [[ -n "${DRY_ENGINE:-}" && -n "$first_gate" ]]; then
    gd="$BUILD_ROOT/e07/gate-dry-${first_gate##*:}" g="$BUILD_ROOT/e07/gate-dry-${first_gate##*:}/gate.json"
    place="--hot-set $gd/hot-set.json --vram-gb $(python3 -c "import json; print(json.load(open('$g'))['vram_gb'])") --ram-gb 190 --threads 16"
    # the gate's choice of the slab's host homes at the plan's host RAM
    place+=" --hot-homes $(python3 -c "import json; print(json.load(open('$g'))['hot_homes'])")"
    (( prefetch == 128 )) && place+=" --prefetch-experts 128"
    # cbatch, clong: the serve dry run with the batch's slots and checkpoints (its "batch slots:" line, the gate's slot
    # bytes; clong's INFO with interleave_chunk=16384, the engine's --chunk)
    serve_extra="$CBATCH_GATE"
    m="$layout/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf"
    python3 -c "import json, sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['prompt_token_ids'])))" \
      "$ORACLE/dump-manifest.json" > "$BUILD_ROOT/e07/oracle-prompt.txt"
    python3 "$STRATA_DIR/tools/dsv41/engine/oracle_parity.py" forced "$ORACLE" "$BUILD_ROOT/e07/oracle-forced" > /dev/null
    echo "--- dsv41-engine --dry-run on the real layout (the placement of ${first_gate%:*}): the correctness runs, then the benches' engine"
    forced="--force-selection $BUILD_ROOT/e07/oracle-forced"
    for mode in $E07_RUN_READERS; do
      for force in "" "$forced"; do
        # shellcheck disable=SC2086  # the placement and the forced reference are whole words
        "$DRY_ENGINE" --model "$m" --tokens-file "$BUILD_ROOT/e07/oracle-prompt.txt" --n-predict 64 --ctx 4096 \
          --readers "$mode" $place $force --dry-run | grep -E "^(dsv41-engine:|prompt:|dry run: [0-9]+ layers|dry run: forced|experts: VRAM)"
      done
    done
    (( bench_only )) ||
      echo "--- the isolation runs: moe-isolated (the oracle's cur in every layer), dense-forced-mmq, dsv41-layer-ref per numerics"
    for extra in $( (( bench_only )) || echo --forced-moe-input --prefill-mmq); do
      # shellcheck disable=SC2086
      "$DRY_ENGINE" --model "$m" --tokens-file "$BUILD_ROOT/e07/oracle-prompt.txt" --n-predict 64 --ctx 4096 \
        --readers dense $place $forced "$extra" --dry-run | grep -E "^(prompt:|dry run: forced|dry run: prefill)"
    done
    for n in $( (( bench_only )) || echo fork-cpu fork-cuda engine); do
      "$(dirname "$DRY_ENGINE")/dsv41-layer-ref" --model "$m" --tokens-file "$BUILD_ROOT/e07/oracle-prompt.txt" \
        --numerics "$n" --out "$BUILD_ROOT/e07/layer-ref-dry/$n" --oracle "$ORACLE" --threads 32 --dry-run | sed "s|$m|MODEL|"
    done
    # shellcheck disable=SC2086
    "$DRY_ENGINE" --serve --model "$m" --ctx 262144 --chunk 16384 $place $serve_extra --dry-run < /dev/null 2>&1 | grep -E "^(READY|INFO|experts: VRAM|experts: hot set|dsv41-engine serve:|conversation cost|batch slots:|dry run: hot set reset)"
    [[ "$E07_PLAN" != e07b ]] || dry_e07b "$m" "$place"
  fi
  echo "then: watchdog deletes the pod at its cap; wait for publicIp + portMappings[22] and an SSH login (up to $SSH_WAIT_S s),"
  echo "  the CUDA probe (the pod's flavor's dsv41_cuda_probe), on failure DELETE, remember the machine, the next offer (at most $MAX_PODS pods)"
  # every line of this step starts with "  qualification"
  if (( QUALIFY )); then
    echo "  qualification, when the CUDA probe passed and before anything else goes up:"
    echo "  qualification:   scp the pod's flavor's dsv41_machine_probe to /root, then"
    echo "  qualification:   ssh \"timeout $QUALIFY_S /root/dsv41_machine_probe $QUALIFY_ARGS --min-host-read-gbs <the pod's GPU's>\""
    echo "  qualification:   bounds: $QUALIFY_GPU_READ_TEXT, 2 GiB streamed;"
    echo "  qualification:   pinned H2D >= $QUALIFY_MIN_H2D_GBS and D2H >= $QUALIFY_MIN_D2H_GBS GB/s (QUALIFY_MIN_H2D_GBS, QUALIFY_MIN_D2H_GBS, 1 GiB),"
    echo "  qualification:   mapped-memory doorbell round trip median <= $QUALIFY_MAX_DOORBELL_US us (QUALIFY_MAX_DOORBELL_US, 2000 round trips, p99 reported),"
    for g in "${gpus[@]}"; do
      hr="$(qualify_host_read "$g")"
      echo "  qualification:   16-thread host read >= ${hr%%|*} GB/s on $g (${hr#*|}),"
    done
    echo "  qualification:   SM clock under the read load >= $QUALIFY_MIN_SM_RATIO x the GPU's"
    echo "  qualification:   clocks.max.sm (QUALIFY_MIN_SM_RATIO, nvidia-smi; power draw reported), enforced power limit >= $QUALIFY_MIN_POWER_RATIO x its"
    echo "  qualification:   power.default_limit (QUALIFY_MIN_POWER_RATIO); the absolute clock, limit and maximums logged"
    echo "  qualification:   then scp the pod's flavor's dsv41_cpu_experts_probe (the strata tarball's) to /root and ssh"
    echo "  qualification:   \"timeout $QUALIFY_S /root/dsv41_cpu_experts_probe --threads $CPU_EXPERTS_THREADS --min-gbs $CPU_EXPERTS_FLOOR_GBS [--numa-node <the GPU's>]\":"
    echo "  qualification:   CPU experts >= $CPU_EXPERTS_FLOOR_GBS GB/s (CPU_EXPERTS_FLOOR_GBS; one token per expert, 192"
    echo "  qualification:   synthetic q2_K/q3_K experts of 12.81 MB, 2.46 GB a run, the median of 7; ~6 s, 2.4 GB of RAM),"
    echo "  qualification:   every line into probe.log and the session log, each machine's values on the sheet; a FAIL line:"
    echo "  qualification:   DELETE, remember the machine, the next offer; MAX_PODS $MAX_PODS$( (( qc )) &&
      echo " (the q plans' default: the third rejected machine ends the session, nothing uploaded)")"
    for t in "${tags[@]}"; do mp="$(machine_probe_ready "$t")" || exit 1; echo "  qualification:   $mp"; done
  else
    echo "  qualification: none (QUALIFY=0): the CUDA probe alone, MAX_PODS $MAX_PODS"
  fi
  echo "  without the volume and with HF_TOKEN set: ssh 'umask 077 && cat > /root/.hf-token' (the token on stdin, not logged)"
  echo "  scp the pod's flavor's payload-<tag>.tar.gz to $REMOTE.tar.gz, then detached:"
  echo "  DEADLINE=<cap - 120 s> POD_VOLUME=<the pod's volume, or empty> nohup setsid bash -c 'bash $RUN_SCRIPT; ...'"
  [[ "$E07_PLAN" != qengine && "$E07_PLAN" != qdiag ]] ||
    echo "  (and POD_MACHINE=<the create's machineId>, which the speed check names)"
  echo "  every $POLL_S s a poll over a fresh SSH connection ($POLL_FAILS failed in a row: stop polling, pull)"
  echo "  a poll that sees a new last line of out/steps.log pulls out/ into OUT/out over one more fresh connection"
  echo "    (timeout $POLL_SSH_S ssh ... 'cd $REMOTE && find out -type f ! -name \"*.bin\" ! -size +${PULL_STEPS_MAX_MB}M -print0 | tar --null -czf - -T -',"
  echo "    a failure a WARNING; none within $POLL_SSH_S s of the run's deadline); when the final pull brings nothing back,"
  echo "    OUT/session-lost, e07.log and the sheet name the step the session was lost after and the steps lost"
  echo "on the pod, pod/e07-run.sh (every engine run with the gate's placement, bound to the GPU's NUMA node by numactl, else numactl --cpunodebind alone, else taskset):"
  echo "  env; then the model's path:"
  if [[ "$E07_PLAN" == qengine || "$E07_PLAN" == qdiag ]]; then
    echo "  right after env, pod/pod-samplers.sh start, in their own sessions until e07-run.sh exits (stop): nvidia-smi"
    echo "    --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,utilization.gpu,"
    echo "    utilization.memory,pcie.link.gen.current,clocks_event_reasons.active (else clocks_throttle_reasons.active; each"
    echo "    field probed, a refused one left out and named in out/samplers.log) -lms 1000 into out/gpu-clocks.csv; every 2 s"
    echo "    /proc/loadavg, the cgroup's CPU limit and cpu.stat (nr_throttled, throttled time) and the top 8 processes by"
    echo "    CPU into out/host-load.log; out/step-times.log holds each step's window"
  fi
  echo "  with the volume: volume-before (find /workspace/models/q2k -maxdepth 1 -printf '%f %s %T@ %m');"
  echo "    gate (e07_gate.py on /workspace/models/q2k/<shard 1>, exit: stop); setup (numactl, jinja2);"
  echo "    model-copy (pod/model-copy.sh /workspace/models/q2k /root/models/q2k, gguf_layout.py sizes)"
  echo "  without it: model-headers (hf_model.py headers e07/hf-manifest.json e07/headers, gguf_layout.py sparse e07/layout);"
  echo "    gate (e07_gate.py on e07/layout/<shard 1>, exit: stop); setup (numactl, jinja2, aria2);"
  echo "    model-download (pod/hf-download.sh e07/hf-manifest.json /root/models/q2k out/model-download.json: aria2c, the 7"
  echo "    files at once, 16 range connections each, then hf_model.py check: each file's size and sha256 against the"
  echo "    manifest, the download's seconds and GB/s in out/model-download.json; then gguf_layout.py sizes)"
  echo "  dry-run: dsv41-engine --serve --model /root/models/q2k/<shard 1> --ctx 262144 --chunk 16384 <placement> --dry-run"
  if [[ "$E07_PLAN" == qdet ]]; then
    echo "  (the gate counts a prefetch region of 128 experts; every step's timeout is min(2 x its estimate below, the"
    echo "  time left before the deadline), pod-steps.sh with e07/estimates-<path>; no lm-eval venv, no samplers)"
    echo "  engine-server: quality/pod_server.sh engine <shard 1> 8090 '<placement> --log-ids' <numactl>: serve/server.py,"
    echo "    the deepseek-v4.1 profile, ./dsv41-engine <placement> --log-ids --chunk 16384 --ctx 262144 --eos-ids <the"
    echo "    GGUF's> run from the payload directory, repeat_stop_tokens 0 (qengine's server, every reply's ids in"
    echo "    out/quality-engine.log); in its own session, ready at its 'ready: http' line"
    echo "  det-sampling: quality/det_sampling_check.py --url http://127.0.0.1:8090 --engine-log out/quality-engine.log"
    echo "    --out out/det-sampling.json, the server's first requests after its load, one at a time, max_tokens 256:"
    echo "    item 7: prompt A greedy (cold), A again, prompt B, A again: the three A replies' ids equal, else the first"
    echo "      divergent position and both tokens; item 8: A at temperature 0.7, top_p 0.95, top_k 40, seed 1234 twice"
    echo "      (the same ids) and seed 1235 (other ids), the engine's line naming the sampler and seed, the 1234 reply's"
    echo "      text for the hand check; prompt S (a count past 256 tokens) greedy and at 0.7, 3 interleaved runs each,"
    echo "      every run 256 tokens without an end of reply, the median decode ms per token within 5%; frequency_penalty"
    echo "      0.5 answered 400 naming it. Exit 1 when an item fails (the step's line), the session goes on"
    echo "  server-stop: pod_server.sh stop"
  elif [[ "$E07_PLAN" == cbase ]]; then
    echo "  (the gate counts a prefetch region of 128 experts; every step's timeout is min(2 x its estimate below, the"
    echo "  time left before the deadline), pod-steps.sh with e07/estimates-<path>; no lm-eval venv, no samplers)"
    echo "  engine-server: quality/pod_server.sh engine <shard 1> 8090 '<placement>' <numactl>: serve/server.py, the"
    echo "    deepseek-v4.1 profile, ./dsv41-engine <placement> --chunk 16384 --ctx 262144 --eos-ids <the GGUF's> run from"
    echo "    the payload directory, repeat_stop_tokens 0, the deterministic mode on (the engine's default): qengine's"
    echo "    server, the FIFO one (one request at a time); in its own session, ready at its 'ready: http' line"
    echo "  load-c<N> for each N of $CBASE_USERS (CBASE_USERS, e07/cbase-users): quality/chat_load.py run --url"
    echo "    http://127.0.0.1:8090 --conversations e07/conversations.json --users <N> --out out/cbase/load-c<N>.json: the"
    echo "    first N conversations, their users started together, each 2 turns of a count past 256 tokens (max_tokens"
    echo "    256, temperature 0, streamed), its next turn when its reply ends; every request's row (prompt, cached and"
    echo "    new tokens, TTFT at the client, the server's queue wait, decode tok/s, finish) and the run's TTFT p50/p95,"
    echo "    queue wait p50/p95, per-user decode tok/s and aggregate tok/s. Exit 1 when a request failed (the step's"
    echo "    line), the session goes on"
    echo "  server-stop: pod_server.sh stop"
  elif [[ "$E07_PLAN" == ccache ]]; then
    echo "  (the gate counts a prefetch region of 128 experts and the engine's 4 conversation checkpoints; every step's"
    echo "  timeout is min(2 x its estimate below, the time left before the deadline), pod-steps.sh with"
    echo "  e07/estimates-<path>; no lm-eval venv, no samplers)"
    echo "  engine-server: cbase's (quality/pod_server.sh engine, qengine's server, the deterministic mode on) with"
    echo "    --log-ids: dsv41-engine keeps 4 conversation checkpoints (its default; the longest prefix resumes, the least"
    echo "    recently used goes) and resets the hot set after each request"
    echo "  load-alt$CCACHE_ALT_USERS: quality/chat_load.py run --url http://127.0.0.1:8090 --conversations e07/conversations.json"
    echo "    --users $CCACHE_ALT_USERS --turns $CCACHE_ALT_TURNS --alternate --engine-log out/quality-engine.log"
    echo "    --out out/ccache/load-alt$CCACHE_ALT_USERS.json: the first $CCACHE_ALT_USERS conversations taking turns (user k's turn t"
    echo "    when the previous user's reply ended, nobody queued), every turn 2+ resumed from its checkpoint; TTFT"
    echo "    p50/p95/max of the turns 2+, the queued flag (queue wait > 0.1 s), each request's wait for the idle reset"
    echo "    (the engine's log): item 10's verdict"
    echo "  load-alt$CCACHE_ALT_USERS-think$CCACHE_THINK_S: the same with --think-s $CCACHE_THINK_S (CCACHE_THINK_S: seconds between a reply's"
    echo "    end and the next user's turn) into out/ccache/load-alt$CCACHE_ALT_USERS-think$CCACHE_THINK_S.json, informational"
    echo "   "
    echo "  load-c<N> for each N of $CCACHE_USERS (CCACHE_USERS, e07/ccache-users): cbase's load step into"
    echo "    out/ccache/load-c<N>.json (the users together, 2 turns each), for the before/after table"
    echo "  det-sampling: qdet's (items 7 and 8 once, the mode with its reset after each request; out/det-sampling.json)"
    echo "  server-stop: pod_server.sh stop (planned at 60 s)"
  elif (( has_cbatch || has_clong )); then
    (( ! has_cbatch )) || {
    if (( has_c05 )); then
    echo "  (one gate: gate with --batch $CBATCH_SLOTS --batch-ctx $CBATCH_CTX --checkpoints $CBATCH_CHECKPOINTS, the batch's slots"
    echo "  taken from the slab, a prefetch region of 128 experts; no batching-off server$( (( has_cdet )) &&
      echo ": the speed A/B replaces it" || echo ", nor an A/B: the fork's numerics only"); every"
    echo "  step's timeout is min(2 x its estimate below, the time left before the deadline); no samplers)"
    else
    echo "  (two gates: gate with --batch $CBATCH_SLOTS --batch-ctx $CBATCH_CTX --checkpoints $CBATCH_CHECKPOINTS, the batch's slots"
    echo "  taken from the slab, and gate-off without them, today's slab; a prefetch region of 128 experts in both; every"
    echo "  step's timeout is min(2 x its estimate below, the time left before the deadline); no lm-eval venv, no samplers)"
    echo "  engine-server-off: quality/pod_server.sh engine <shard 1> 8090 '<gate-off's placement> --checkpoints"
    echo "    $CBATCH_CHECKPOINTS' <numactl>: cbase's FIFO server (batching off: no --batch, one request at a time)"
    echo "  load-short-off: quality/chat_load.py run --url http://127.0.0.1:8090 --conversations e07/conversations-short.json"
    echo "    --users 1 --turns $CBATCH_SHORT_TURNS --out out/cbatch/load-short-off.json: the single user's $CBATCH_SHORT_TURNS turns at short"
    echo "    context (item 11's before; e07/cbatch-short)"
    echo "  server-stop-off: pod_server.sh stop"
    fi
    echo "  engine-server: quality/pod_server.sh engine <shard 1> 8090 '<gate's placement> --batch $CBATCH_SLOTS --batch-ctx"
    echo "    $CBATCH_CTX --checkpoints $CBATCH_CHECKPOINTS --log-ids' <numactl>: the server drives the engine's batch slots"
    echo "    (StrataEngine.generate_batched: a request alone decodes alone, the others in slots, one token per slot a step)"
    (( ! has_cdet )) || echo "    and --cpu-numerics gpu (the A/B's gpu arm; the engine's default is the fork's)"
    [[ "$E07_PLAN" != cfinal ]] || echo "    in the engine's default numerics, the fork's"
    if (( has_c05 )); then
    echo "  load-short: quality/chat_load.py run --url http://127.0.0.1:8090 --conversations e07/conversations-short.json"
    echo "    --users 1 --turns $CBATCH_SHORT_TURNS --out out/cbatch/load-short.json: the single user's $CBATCH_SHORT_TURNS turns at short"
    echo "    context, the deterministic mode on (item 11: >= 30 tok/s, within 5% of session 4's 61.55; steady-c4's 2 x base)"
    (( ! has_cdet )) ||
    echo "  ab-gpu: the same with --deterministic off into out/cbatch/ab-gpu.json: the speed A/B's CPU numerics gpu arm"
    else
    echo "  load-short: the same as load-short-off into out/cbatch/load-short.json (item 11: >= 30 tok/s, within 5% of off)"
    fi
    if (( has_clong )); then
      echo "  (no load-c<N> nor det-sampling: measured on 2026-10-14, the cbatch session)"
    elif (( has_c05 )); then
      echo "  (no load-c<N>: measured on 2026-10-14, the cbatch session)"
    else
      echo "  load-c<N> for each N of $CBATCH_USERS (CBATCH_USERS, e07/cbatch-users): cbase's load step into"
      echo "    out/cbatch/load-c<N>.json, the N users together decoding at once: per-user and aggregate decode tok/s (item 11:"
      echo "    the end-to-end figures, user-visible, the prompt reads included; 8 reported)"
    fi
    echo "  steady-c$CBATCH_STEADY_USERS: quality/chat_load.py steady --url http://127.0.0.1:8090 --conversations e07/conversations.json"
    echo "    --users $CBATCH_STEADY_USERS --window-s $CBATCH_STEADY_WINDOW_S --repeats $CBATCH_STEADY_REPEATS --max-tokens $CBATCH_STEADY_MAX_TOKENS"
    echo "    --out out/cbatch/steady-c$CBATCH_STEADY_USERS.json (e07/cbatch-steady): the users' prompts read, then"
    echo "    $CBATCH_STEADY_WINDOW_S s with every user decoding, exactly $CBATCH_STEADY_USERS rows a step: item 11's verdict"
    echo "    (aggregate >= 2 x the single user with batching off, each >= 20 tok/s)"
    for n in $CBATCH_STEADY_REPORT; do
      echo "  steady-c$n: the same with --users $n into out/cbatch/steady-c$n.json (e07/cbatch-steady-report):"
      echo "    $n users decoding at once, aggregate and per user reported, not a bar"
    done
    if (( has_cdet )); then
      echo "  det-batch: quality/det_batch_check.py --url http://127.0.0.1:8090 --engine-log out/quality-engine.log --out"
      echo "    out/det-batch.json (item 14): prompt A and 4 other prompts alone, one at"
      echo "    a time, then A 4 times sent together non-streamed, 4 times streamed, and the 4 others together, every request"
      echo "    greedy with max_tokens 256: each reply's tokens (the engine's --log-ids lines, a reply moved between the solo"
      echo "    path and a slot put back together: a promoted chain) and text its prompt's alone run's, every request in the"
      echo "    deterministic mode, all but one of each phase's requests decoded beside other slots, each chain's verdict"
      echo "    printed (promoted or not), and each reply saying \"strata_reproducible\": true paired with a chain shown equal"
      echo "    to its alone run. Exit 1 when item 14 fails, the session goes on"
    fi
    (( has_clong )) || [[ "$E07_PLAN" == cfinal ]] ||
      echo "  det-sampling: qdet's (items 7 and 8, each request alone: the solo path, reproducible; out/det-sampling.json)"
    if (( has_c05 )); then
      echo "  quality-venv: the lm-eval venv installed in the background since setup (quality/run_lm_eval.sh INSTALL_ONLY=1)"
      echo "  continuation: quality/eval.sh strata http://127.0.0.1:8090/v1/chat/completions out/quality continuation (qengine's"
      echo "    HumanEval continuation check before the full run; exit: stop)"
      echo "  lm-eval-<task> for each of $CDET_TASKS: NUM_CONCURRENT=$CDET_CONCURRENT quality/eval.sh strata"
      echo "    http://127.0.0.1:8090/v1/chat/completions out/quality <task>: item 3 batched, run_lm_eval.sh through"
      echo "    api_proxy.py on the fixed-seed subsets (quality/subsets_lean2h_seed20261006.json), $CDET_CONCURRENT requests at"
      echo "    once on the batch slots, the deterministic mode on (the engine's default)"
    fi
    echo "  server-stop: pod_server.sh stop (planned at 60 s)"
    if (( has_cdet )); then
      echo "  engine-server-fork: engine-server's command with --cpu-numerics fork (the CPU experts in the fork's numerics,"
      echo "    the engine's earlier default), the batched server's logs set aside meanwhile; ab-fork: ab-gpu's user into"
      echo "    out/cbatch/ab-fork.json; server-stop-fork, its logs kept as out/fork-<file>: the speed A/B on one host"
      echo "    (the 50.7 against 61.55 tok/s), last, so the first cut if time runs out"
    fi
    }
    if (( has_clong )); then
      sfx=""
      (( has_cbatch )) && sfx="-long"
      if (( has_cbatch )); then
        echo "  then clong's steps, their own table and verdict on the sheet:"
        echo "  gate-long (after gate-off): e07_gate.py with $CLONG_GATE into out/gate-long"
      else
        echo "  (the gate with $CLONG_GATE: each slot holds the 64K prompt and its reply; a prefetch region of"
        echo "  128 experts; every step's timeout is min(2 x its estimate below, the time left before the deadline))"
      fi
      echo "  engine-server$sfx: quality/pod_server.sh engine <shard 1> 8090 '<the long gate's placement> --batch $CLONG_SLOTS"
      echo "    --batch-ctx $CLONG_CTX --checkpoints $CLONG_CHECKPOINTS --log-ids' <numactl>: the engine's default interleave"
      echo "    chunk, its --chunk $CLONG_CHUNK (a prompt read beside decoding slots in chunks of $CLONG_CHUNK, a step between two),"
      echo "    the deepseek-v4.1 profile's queue_max 8, request_deadline_s 1800 and engine"
      echo "    watchdog (engine_silence_s 300); the engine's log gives each chunk's read time and the slots' longest hold"
      echo "  load-long: quality/chat_load.py long --url http://127.0.0.1:8090 --conversations e07/conversations-long.json"
      echo "    --users $CLONG_USERS --max-tokens $CLONG_MAX_TOKENS --out out/clong/load-long.json: the 64K user alone (its TTFT solo),"
      echo "    then $CLONG_USERS users of 16K together and, 2 s after each has its first token, the same 64K request: each user's"
      echo "    largest gap between two tokens (item 12: <= 5 s) and over the read, the 64K TTFT against solo"
      echo "  robust: quality/robust_check.py --url http://127.0.0.1:8090 --out out/clong/robust.json: item 13 on the real"
      echo "    engine (streamed and not, two non-streamed batched, a streamed and a non-streamed client hanging up while the"
      echo "    $CLONG_SLOTS slots decode and the waiting request getting its slot within 15 s, the $CLONG_SLOTS slots and 8 waiting:"
      echo "    the next request 503 with Retry-After and the queue in its JSON error, then the server drains; no-max-tokens:"
      echo "    a request without max_tokens beside 4 users takes a slot with the profile's default_max_tokens, and one under a"
      echo "    slot ends \"length\" at its room; slot-room: a 40K prompt with max_tokens 32000 cut to its slot's room beside 4"
      echo "    users, a turn past a slot refused with a 400 beside them and run alone after them, said by the engine's ALONE"
      echo "    line, /metrics and strata_ran_alone)"
      echo "  server-stop$sfx: pod_server.sh stop (planned at 60 s)"
    fi
    if (( has_spare )); then
      echo "  then the second cbatch server, item 11 judged on each server:"
      echo "  gate-spare (after gate-long): e07_gate.py with gate's --batch $CBATCH_SLOTS --batch-ctx $CBATCH_CTX --checkpoints"
      echo "    $CBATCH_CHECKPOINTS and --hot-spare $CBATCH_SPARE into out/gate-spare (the slab less the spare slots' room)"
      echo "  engine-server-spare: quality/pod_server.sh engine <shard 1> 8090 '<gate-spare's placement> --hot-spare $CBATCH_SPARE"
      echo "    --batch $CBATCH_SLOTS --batch-ctx $CBATCH_CTX --checkpoints $CBATCH_CHECKPOINTS --log-ids' <numactl>"
      echo "  load-short-spare: load-short on this server into out/cbatch/load-short-spare.json (its single user re-checked)"
      echo "  steady-c$CBATCH_STEADY_USERS-spare: steady-c$CBATCH_STEADY_USERS on this server into out/cbatch/steady-c$CBATCH_STEADY_USERS-spare.json"
      echo "  server-stop-spare: pod_server.sh stop; its logs kept as out/spare-<file>"
    fi
  elif [[ "$E07_PLAN" == q* ]]; then
    [[ "$E07_PLAN" != qfork ]] ||
      echo "  (qfork runs no engine: no model-headers, gate nor dry-run; setup as above)"
    echo "  setup also makes python3's venv work (virtualenv when the image's python3 has no ensurepip); right after it, in"
    echo "  the background: run_lm_eval.sh INSTALL_ONLY=1 (python3 -m venv /root/venv-quality, pip install"
    echo "  lm-eval[api,ifeval,ruler,math]==0.4.13), out/quality-venv.log, while the model downloads and loads"
    echo "  (every step's timeout is min(2 x its estimate below, the time left before the deadline), as e07b's)"
    if [[ "$E07_PLAN" == qdiag ]]; then
      echo "  bench-qdiag: e07_bench.py --engine ./dsv41-engine --engine-args '--model <shard 1> --ctx 262144 --chunk 16384"
      echo "    <placement>' --prompts e07/prompts --phases short-static,short-adaptive --out out/bench-qdiag.json (one engine"
      echo "    load, no server: hot=static and hot=adaptive, hot_reset=1, 481 generated each; out/bench-qdiag-engine.log);"
      echo "    a failure leaves the server path to run"
    fi
    if [[ "$E07_PLAN" == qfork ]]; then
      echo "  fork-server: quality/pod_server.sh fork <shard 1> 8080 <numactl>: fork/llama-server with t04a.sh's stock flags"
      echo "    (14 expert layers in VRAM, the other routed experts on the CPU, -lm mmap, --flash-attn on, -b 2048 -ub 1024,"
      echo "    --numa numactl) and --ctx-size 32768 -np $Q_FORK_NP (Q_FORK_NP, e07/fork-np) --kv-unified --prefill-assistant;"
      echo "    --threads min(16, cpus) and --threads-batch min(32, cpus), cpus = min(nproc under the NUMA binding,"
      echo "    floor(cgroup CPU quota / period)), at least 1, on the pod; the values in out/quality-server.cmd and their"
      echo "    source (nproc, the cgroup's cpu.max or cfs_quota_us) in out/fork-server.log;"
      echo "    in its own session, ready at /health; nvidia-smi samples the GPU's memory used every 1 s into"
      echo "    out/quality-gpu.csv until server-stop (the fork server's peak on the qfork sheet)"
    else
      echo "  engine-server: quality/pod_server.sh engine <shard 1> 8090 '<placement>' <numactl>: serve/server.py, the"
      echo "    deepseek-v4.1 profile, ./dsv41-engine <placement> --chunk 16384 --ctx 262144 --eos-ids <the GGUF's>"
      echo "    run from the payload directory (the placement's relative --hot-set resolves there), repeat_stop_tokens 0;"
      echo "    in its own session, ready at its 'ready: http' line"
    fi
    backend="$([[ "$E07_PLAN" == qfork ]] && echo "llamacpp, NUM_CONCURRENT $Q_FORK_NP" || echo "strata, 2 concurrent")"
    echo "  quality-venv: the background install's exit status (0, or the session stops)"
    echo "  continuation: $([[ "$E07_PLAN" == qfork ]] && echo "continuation.py apply-template (llama-server's prompt of 3 HumanEval items) and ")eval.sh"
    echo "    continuation (their replies through the proxy, out/quality/continuation.json, for the hand check); a failure"
    echo "    stops the server and the session before the full run"
    if [[ "$E07_PLAN" == qfork ]]; then
      echo "  speed-check: pod/speed_check.py out/quality-server.log --fork --max-ms $Q_FORK_MAX_MS (Q_FORK_MAX_MS,"
      echo "    e07/fork-max-ms) --machine <POD_MACHINE>: the continuation's replies (llama-server's print_timing 'eval"
      echo "    time' lines), their ms per generated token and median, the machine id and the threshold; above"
      echo "    $Q_FORK_MAX_MS ms per token the server and the session stop with out/steps.log's 'stop: fork too slow on"
      echo "    this machine: X ms per token (machine M)', pulled and deleted"
    fi
    [[ "$E07_PLAN" == qfork ]] || {
      echo "  speed-check: pod/speed_check.py out/quality-engine.log --max-ms $Q_SPEED_MAX_MS (Q_SPEED_MAX_MS, e07/speed-max-ms)"
      echo "    --machine <POD_MACHINE>: the continuation's requests and decode steps, the median ms per decode step and"
      echo "    doorbell wait per step, the machine id and the threshold; above $Q_SPEED_MAX_MS ms per step $([[ "$E07_PLAN" == qdiag ]] &&
        echo "the step fails and qdiag carries on" || echo "the server and the session stop with")"
      [[ "$E07_PLAN" == qdiag ]] ||
        echo "    out/steps.log's 'stop: engine too slow on this machine: X ms per decode step (machine M)', pulled and deleted"
    }
    if [[ "$E07_PLAN" == qdiag ]]; then
      echo "  lm-eval-gsm8k-c1: SAMPLES=e07/subsets-qdiag.json NUM_CONCURRENT=1 eval.sh strata ... out/quality/c1"
      echo "    gsm8k_cot_llama (the first $QDIAG_ITEMS items; the proxy's usage in out/quality/c1/api_usage.jsonl)"
      echo "  lm-eval-gsm8k-c2: the same with NUM_CONCURRENT=2 into out/quality/c2; no IFEval, HumanEval nor needles"
    else
      echo "  lm-eval-<task> for each of $Q_TASKS: eval.sh ($backend, the proxy's"
      echo "    usage in out/quality/api_usage.jsonl) on the fixed-seed subsets, lm-eval's output in out/quality/lm_eval/<task>/"
      echo "  needle-<length> for each of ${Q_NEEDLES:-none}: eval.sh needle-<length> (needle_check.py on e07/needle.json)"
    fi
    echo "  server-stop: pod_server.sh stop"
  elif [[ "$E07_PLAN" == e07b ]]; then
    echo "  (the gate counts a prefetch region of 128 experts; every engine run gets --prefetch-experts 128; every step's"
    echo "  timeout is min(2 x its estimate below, the time left before the deadline), pod-steps.sh with e07/estimates-<path>:"
    echo "  a step killed at it says 'step NAME killed: timeout after N s (...)' in its log and on its out/steps.log line,"
    echo "  and the next step runs)"
    echo "  two-turn: engine/two_turn_check.py --engine ./dsv41-engine --model <shard 1> --tokenizer-gguf <shard 1> --ctx 20480"
    echo "    --load-s 1500 --engine-args '<placement> --chunk 16384' --turns e07/turns.json --max-tokens 128 --http"
    echo "    --out out/two-turn (serve/server.py; turn 1 the 16K prompt's text, turn 2 a short follow-up streamed: its"
    echo "    cached tokens, its R line resume=1 (turn 1's resume=0), its decode and its first token at the client; then"
    echo "    on the same server load /v1/chat/completions non-streaming and streaming, each reply the text of the"
    echo "    engine's ids, tools refused)"
    echo "  bench-e07b: e07_bench.py --engine ./dsv41-engine --engine-args '--model <shard 1> --ctx 262144 --chunk 16384"
    echo "    <placement>' --prompts e07/prompts --phases 16k-e09,turn2,short-adaptive,64k,256k --out out/bench-e07b.json:"
    echo "    16k-e09 (hot_reset=1 prefill_times=1, 32 generated), turn2 (16,415 reused + 513, 32 generated), short-adaptive"
    echo "    (481 generated), 64k and 256k (trace=0, 481 generated each); the engine must answer afterwards"
    echo "  dense-free: dsv41-engine --model <shard 1> --tokens-file e07/oracle-prompt.txt --n-predict 64 --ctx 4096 --readers"
    echo "    dense <placement> --dump-layers out/dense-free, then parity-dense-free: oracle_parity.py out/dense-free e07/oracle"
    echo "    (the free-greedy ids against the stock fork's: reported, not gating; the e07a session's A100 7 of 64)"
    echo "  forced-ref, moe-isolated (--force-selection e07/oracle-forced --forced-moe-input) and parity-moe-isolated"
    echo "    (the isolated MoE per layer, reported) in the time left"
  elif [[ "$E07_PLAN" == e09 ]]; then
    echo "  (the gate counts a prefetch region of 128 experts: e07_gate.py --prefetch-experts 128; every engine run gets"
    echo "  --prefetch-experts 128 in its placement)"
    echo "  bench-e09: e07_bench.py --engine ./dsv41-engine --engine-args '--model <shard 1> --ctx 262144 --chunk 16384 <placement>'"
    echo "    --prompts e07/prompts --phases 16k-e09,short-adaptive,16k-e09,256k --out out/bench-e09.json: GEN keys"
    echo "    hot_reset=1 prefill_times=1 for 16k-e09 (32 generated; the second reads cold after short-adaptive),"
    echo "    hot=adaptive hot_reset=1 for short-adaptive (481 generated), trace=0 for 256k (481 generated)"
    echo "  two-turn: engine/two_turn_check.py --engine ./dsv41-engine --model <shard 1> --tokenizer-gguf <shard 1> --ctx 4096"
    echo "    --load-s 1500 --engine-args '<placement>' --out out/two-turn (serve/server.py, --route-trace out/two-turn/"
    echo "    route-trace.txt: turn 2's cached tokens, its R line resume=1, its decode from the engine's log)"
  elif [[ "$E07_PLAN" == e08 ]]; then
    echo "  (the gate packs the hot set beside 2 spare slots per layer: the slab's --vram-gb holds both, the spare homes pinned)"
    echo "  bench-e08: e07_bench.py --engine ./dsv41-engine --engine-args '--model <shard 1> --ctx 262144 --chunk 16384 <placement>"
    echo "    --route-trace out/route-trace.txt' --prompts e07/prompts --phases short-static,short-prompt,short-adaptive,16k-e08,"
    echo "    short-timed,64k,256k --out out/bench-e08.json: GEN keys hot=<policy> hot_reset=1 per short phase, hot_reset=1"
    echo "    prefill_times=1 for 16k-e08 (481 generated), times=1 for short-timed, trace=0 for 64k and 256k; each request's"
    echo "    decode experts, hot set (policy, swaps, reset, hit rate per layer) and the 16K prompt's split from the log"
    echo "  route-trace: gzip -f out/route-trace.txt (pulled with the logs; hot_policy_sim.py compare replays it here)"
    echo "  serve: serve_check.py --model <shard 1> --tokenizer-gguf <shard 1> --vocab <shard 1> --reference cli --parts profile,http"
    echo "    --one-engine --ctx 4096 --load-s 1500 --engine-args '<placement>' --chats e07/chats.json --out out/serve"
    echo "    (the 3-chat quality check; the reference CLI run first, then the server)"
  else
  echo "  forced-ref: oracle_parity.py forced e07/oracle e07/oracle-forced"
  echo "  per readers ($E07_READERS) and selection forced, free: dsv41-engine --model <shard 1> --tokens-file e07/oracle-prompt.txt"
  echo "    --n-predict 64 --ctx 4096 --readers <mode> <placement> [--force-selection e07/oracle-forced] --dump-layers out/<mode>-<sel>,"
  echo "    then oracle_parity.py out/<mode>-<sel> e07/oracle --json out/parity-<mode>-<sel>.json"
  echo "  (readers: $E07_READERS)"
  echo "  per numerics fork-cpu, fork-cuda, engine: layer-ref-<n>: dsv41-layer-ref --model <shard 1> --tokens-file"
  echo "    e07/oracle-prompt.txt --numerics <n> --out out/layer-ref/<n> --oracle e07/oracle --threads 32 (the CPU only)"
  echo "  layer-parity: layer_parity.py compare --oracle e07/oracle --engine dense-forced=out/dense-forced"
  echo "    --json out/layer-parity.json out/layer-ref/fork-cpu out/layer-ref/fork-cuda out/layer-ref/engine"
  echo "  moe-isolated: dsv41-engine ... --readers dense <placement> --force-selection e07/oracle-forced --forced-moe-input"
  echo "    --dump-layers out/moe-isolated, then oracle_parity.py out/moe-isolated e07/oracle --json out/parity-moe-isolated.json"
  echo "  when out/layer-parity.json has mmq_worth_measuring true (the decision in out/plan.log): dense-forced-mmq (the"
  echo "    dense-forced run with --prefill-mmq), its parity, and layer-parity-mmq against it"
  [[ "$E07_PLAN" == cycle2 ]] ||
    echo "  tokenizer: tokenizer_parity.py --llama-tokenize fork/llama-tokenize --corpus tokenizer_corpus.txt <shard 1>"
  echo "  bench-a: e07_bench.py --engine ./dsv41-engine --engine-args '--model <shard 1> --ctx 262144 --chunk 16384 <placement>'"
  echo "    --prompts e07/prompts --phases 16k,turn2,short,short-timed --out out/bench-a.json (each request's decode"
  echo "    experts from the engine's log; short-timed: 64 decode steps timed per stage)"
  if [[ "$E07_PLAN" == full ]]; then
    echo "  serve: serve_check.py --model <shard 1> --tokenizer-gguf <shard 1> --vocab <shard 1> --reference cli --parts profile,http"
    echo "    --one-engine --ctx 4096 --load-s 1500 --engine-args '<placement>' --chats e07/chats.json --out out/serve"
    echo "    (the reference CLI run first, then the server: one engine on the GPU at a time)"
    echo "  bench-b: e07_bench.py ... --phases 64k,256k --out out/bench-b.json"
  else
    echo "  (E07_PLAN=cycle2: no tokenizer, serve nor bench-b, measured in session 1)"
  fi
  fi
  echo "  with the volume: volume-after (the listing again, diffed; also on an early stop)"
  echo "the pull (the logs, the parity and bench JSON; the dumps stay on the pod), DELETE, cost in $OUT/cost.txt, then"
  echo "  pod/e07_report.py $OUT $SHEET_ROOT: the session sheet (README.md tables, the logs)"
  [[ "$E07_PLAN" != qengine && "$E07_PLAN" != qdiag ]] ||
    echo "  (its pod/qdiag_report.py first writes out/qdiag.md and qdiag.json: per phase, joined to the samplers by out/step-times.log)"
  # the cap a pod at the limit price gets: CAP_MIN, or less when the session's USD run out first
  cap="$(cap_minutes "$MAX_USD_PER_HR")" over=()
  if [[ "$E07_PLAN" == qdiag ]]; then
    echo "qdiag estimates: the engine's load 78 s; the slow qengine session's server start 70 s,"
    echo "  continuation 16 s, GSM8K 4.1 s per 1080-token prompt and ~100 generated tokens at 15 tok/s per item"
    echo "  the session's cap: $MAX_SESSION_USD USD at $MAX_USD_PER_HR USD/h on each model path the offers take"
  elif [[ "$E07_PLAN" == qdet ]]; then
    echo "qdet estimates, the 2026-10-13 qengine session's rates (RTX PRO 6000):"
    echo "  the model download at HF_GBS $HF_GBS GB/s (that session: 264.5 GB in 286.3 s, its step 309 s with the sha256"
    echo "  check), the engine's load ~6 min (its step 110 s), decode ~55 tok/s under lm-eval; every timeout 2 x the estimate"
    echo "  the session's cap: $MAX_SESSION_USD USD at $MAX_USD_PER_HR USD/h on each model path the offers take"
  elif [[ "$E07_PLAN" == cbase ]]; then
    echo "cbase estimates, the worst measured rates on the RTX PRO 6000: the model download at"
    echo "  HF_GBS $HF_GBS GB/s, the engine's load ~6 min (the 2026-10-13 qengine session), a request of at most 17664"
    echo "  prompt tokens read whole at 3571.5 tok/s (the e09 session, 2026-10-10: the 16K prompt in 4.59 s), the deterministic mode's"
    echo "  reset 2.0 s, 256 tokens at 35.4 tok/s (the e09 session's decode after the 16K prompt; qengine's"
    echo "  ~55 under lm-eval): $cbase_req_s s each, one at a time; every timeout 2 x the estimate"
    echo "  the session's cap: $MAX_SESSION_USD USD at $MAX_USD_PER_HR USD/h on each model path the offers take"
  elif [[ "$E07_PLAN" == ccache ]]; then
    echo "ccache estimates: cbase's (the model download at HF_GBS $HF_GBS GB/s,"
    echo "  the engine's load ~6 min, a request read whole $cbase_req_s s with the 2.0 s reset, one at a time), the"
    echo "  alternating runs' turns 2+ resumed ($ccache_resume_s s: at most 512 new tokens read, the reset, the decode; the"
    echo "  think run $CCACHE_THINK_S s more between requests), det-sampling as qdet's, the server's stop 60 s;"
    echo "  every timeout 2 x the estimate; the worst case, no turn resumed, adds $ccache_whole_min min (every alternating"
    echo "  request read whole at $cbase_req_s s)"
    echo "  the session's cap: $MAX_SESSION_USD USD at $MAX_USD_PER_HR USD/h on each model path the offers take"
  elif (( has_cbatch || has_clong )); then
    if (( has_cbatch )); then
      echo "cbatch estimates: cbase's (the model download at HF_GBS $HF_GBS GB/s, the"
      echo "  engine's load ~6 min, here twice; a 16K request read whole $cbase_req_s s with the 2.0 s reset, one at a time:"
      echo "  the batch's worst case, no speed-up), the short user's turn the reset and 256 tokens at 35.4 tok/s,$( (( has_clong || has_c05 )) ||
        echo " det-sampling as qdet's,")"
      echo "  each server's stop 60 s; every timeout 2 x the estimate"
      (( ! has_cdet )) || {
        echo "cdet estimates, measured where there are"
        echo "  numbers: each server's load 2.5 min (85-140 s in the cbatch and cdet sessions), det-sampling 3 min (the 2026-10-17"
        echo "  cdet session's 455 s less the 300 s its penalty request waited after the engine's ERR, fixed), each lm-eval task"
        echo "  as it ran on the batch slots there (ifeval 1042 s, gsm8k 459 s; humaneval, cut at the deadline, the reference run's 359 s on"
        echo "  the FIFO server: the other two took 0.97 and 0.86 x the reference run's), det-batch from cbase's rates (the 2.0 s reset, 256"
        echo "  tokens at 35.4 tok/s alone, 20 a user batched)"
      }
      [[ "$E07_PLAN" != cfinal ]] || {
        echo "cfinal estimates, measured where there are numbers: the"
        echo "  server's load 2.5 min (85-140 s in the cbatch and cdet sessions), each lm-eval task at the fork's numerics' own"
        echo "  time, the engine quality run one at a time on the FIFO server (ifeval 1077 s, gsm8k 531 s, humaneval 359 s x 1.25):"
        echo "  every timeout (2 x) above the GPU's numerics' 2026-10-19 times on the batch slots (1521 s, 909 s, humaneval at"
        echo "  162 of 164 items when cut at 720 s), so the slower path still ends inside it"
      }
    fi
    if (( has_clong )); then
      echo "clong estimates: the model download at HF_GBS $HF_GBS GB/s, the engine's"
      echo "  load ~6 min$( (( has_cbatch )) && echo " (a third time)"); load-long: the 64K prompt alone 25 s (~20-25 s measured on the RTX PRO 6000) + the"
      echo "  2.0 s reset + 16 tokens at 35.4 tok/s, then $CLONG_USERS 16K reads at 3571.5 tok/s, 2 s, the $long_tokens-token read again in"
      echo "  $long_chunks chunks of $CLONG_CHUNK at a planned 4.65 s each (the warm tier's 113.13 GiB through the ring at 26.12 GB/s;"
      echo "  host dependent: 3.2-3.3 s on the RTX PRO 6000 Server of 2026-10-15, the e09 session's 4.59 s, the e07b session's slower card 7.68 s) with a"
      echo "  50 ms step between two, $CLONG_MAX_TOKENS tokens per user at 20 tok/s: $long_min min; robust 3 min; the stop 60 s; every"
      echo "  timeout 2 x the estimate"
    fi
    echo "  the session's cap: $MAX_SESSION_USD USD at $MAX_USD_PER_HR USD/h on each model path the offers take"
  elif [[ "$E07_PLAN" == q* ]]; then
    echo "$E07_PLAN estimates (quality/q_plan.py), one request at a time:"
    echo "  $q_tokens"
    echo "  rates (Q_PROMPT_TOK_S, Q_LONG_PROMPT_TOK_S, Q_DECODE_TOK_S): prompts $Q_PROMPT_TOK_S tok/s, the needles'"
    echo "  $Q_LONG_PROMPT_TOK_S tok/s, decode $Q_DECODE_TOK_S tok/s; $([[ "$E07_PLAN" == qfork ]] &&
      echo "the fork's planned rates (decode: the stock baseline 18.9-19.2 tok/s, A100 SXM4)" ||
      echo "the e07b session, RTX PRO 6000 WS: the 128-token prompt, the 256K prompt, decode 62.8 short and 56.1 at 64K and 256K")"
    echo "  the session's cap: $MAX_SESSION_USD USD at $MAX_USD_PER_HR USD/h on each model path the offers take"
  fi
  for path in volume hf; do
    declare -n steps="plan_$path"
    echo "plan with the model from $([[ $path == volume ]] && echo "the volume" || echo "HF") (minutes, estimates):"
    t=0 i=0
    declare -n pod_steps="pod_$path"
    for s in "${steps[@]}"; do
      t="$(python3 -c "print(round($t + ${s##*:}, 1))")"
      # e07b and the Part Q plans: each pod step's timeout (the session's own steps, the head and the pull, have none)
      timeout_note=""
      if (( timed )) && (( i >= ${#plan_head[@]} && i < ${#plan_head[@]} + ${#pod_steps[@]} )); then
        timeout_note="; timeout min($(python3 -c "print(round(2 * ${s##*:}, 1))") min, the time left)"
      fi
      printf '  %-62s %5s  (total %s%s)\n' "${s%:*}" "${s##*:}" "$t" "$timeout_note"
      i=$(( i + 1 ))
    done
    unset -n pod_steps
    total="$(plan_total "${steps[@]}")"
    echo "plan ($path): total $total min of the cap $cap min at the limit price (about $(python3 -c "print(f'{$total / 60 * $MAX_USD_PER_HR:.2f}')") USD at $MAX_USD_PER_HR USD/h)"
    # the Part Q plans: only the model paths the offers can take (NETWORK_VOLUME, HF_CLOUDS)
    offered=1
    (( qc )) && ! { [[ $path == volume && -n "$NETWORK_VOLUME" ]] || [[ $path == hf && -n "$HF_CLOUDS" ]]; } && offered=0
    (( offered )) || echo "plan ($path): not offered ($([[ $path == hf ]] && echo HF_CLOUDS || echo NETWORK_VOLUME) is empty): shown, not checked"
    (( ! offered )) || python3 -c "import sys; sys.exit(0 if $total <= $cap else 2)" || over+=("$path")
    # ccache's worst case, no turn resumed, within the pod's cap and the session's USD too
    if [[ "$E07_PLAN" == ccache ]]; then
      worst="$(python3 -c "print(round($total + $ccache_whole_min, 1))")"
      worst_usd="$(python3 -c "print(f'{$worst / 60 * $MAX_USD_PER_HR:.2f}')")"
      echo "plan ($path) with no turn resumed: total $worst min of the cap $cap min at the limit price (about $worst_usd USD at $MAX_USD_PER_HR USD/h)"
      (( ! offered )) || python3 -c "import sys; sys.exit(0 if $worst <= $cap and $worst / 60 * $MAX_USD_PER_HR <= $MAX_SESSION_USD else 2)" ||
        { echo "e07.sh: E07_PLAN=ccache's $path plan with no turn resumed is over the cap: $worst min, $worst_usd USD"
          over+=("$path (no turn resumed)"); }
    fi
    # the Part C batch plans' worst case: the plan's USD within the session's cap at the
    # limit price (over is refused above), and each pod step's whole timeout (2 x its estimate) from the minute the
    # estimates before it end, within the pod's deadline (cap - 2 min), so no step is cut short by the time left
    if (( has_cbatch || has_clong )); then
      declare -n w_pod="pod_$path"
      read -r w_step w_start w_end <<<"$(python3 - "${plan_head[@]}" "@" "${w_pod[@]}" <<'EOF2'
import sys
a = sys.argv[1:]
k = a.index("@")
head, pod = a[:k], a[k + 1:]
t = sum(float(s.rsplit(":", 1)[1]) for s in head)
worst = ("none", 0.0, 0.0)
for s in pod:
    est = float(s.rsplit(":", 1)[1])
    if t + 2 * est > worst[2]:
        worst = (s.split()[0].split(":")[0], round(t, 1), round(t + 2 * est, 1))
    t += est
print(*worst)
EOF2
)"
      echo "plan ($path) latest end: $w_step, started at minute $w_start after the estimates before it and run to its whole timeout, ends at minute $w_end, the latest of any step,"
      echo "  of the pod's $(( cap - 2 ))-minute deadline: $(python3 -c "print('every step covered' if $w_end <= $cap - 2 else 'NOT covered')");" \
        "the estimates' $total min about $(python3 -c "print(f'{$total / 60 * $MAX_USD_PER_HR:.2f}')") USD, the deadline's at most $(python3 -c "print(f'{($cap - 2) / 60 * $MAX_USD_PER_HR:.2f}')") USD, of the $MAX_SESSION_USD USD cap"
      # the model paths the offers can take only (as the cap check above)
      (( ! offered )) || python3 -c "import sys; sys.exit(0 if $w_end <= $cap - 2 else 2)" ||
        { echo "e07.sh: E07_PLAN=$E07_PLAN's $path plan: $w_step's whole timeout ends at minute $w_end, past the pod's deadline"
          over+=("$path (a step's timeout past the deadline)"); }
      unset -n w_pod
    fi
    # the HF plan at the pessimistic rate within 3.00 USD at the limit price; the
    # download's guard and what it does at that rate
    if [[ "$E07_PLAN" == e07b && "$path" == hf ]]; then
      python3 -c "import sys; sys.exit(0 if $total / 60 * $MAX_USD_PER_HR <= 3.00 else 2)" ||
        { echo "e07.sh: E07_PLAN=e07b's HF plan at $HF_GBS GB/s costs more than 3.00 USD at $MAX_USD_PER_HR USD/h"; over+=("hf (3.00 USD)"); }
    fi
    # Part Q: each model path the offers can take within the session's USD at the limit price (the rule:
    # a plan over 3 USD is split into sessions by task, Q_TASKS)
    if (( qc && offered )); then
      python3 -c "import sys; sys.exit(0 if $total / 60 * $MAX_USD_PER_HR <= $MAX_SESSION_USD else 2)" ||
        { echo "e07.sh: E07_PLAN=$E07_PLAN's $path plan costs more than $MAX_SESSION_USD USD at $MAX_USD_PER_HR USD/h: split it by task (Q_TASKS, Q_NEEDLES)"
          # qfork's sessions per model path
          if [[ "$E07_PLAN" == qfork && $path == volume ]]; then
            echo "  qfork's split with the model from the volume, 2 sessions: Q_TASKS=gsm8k_cot_llama,humaneval_instruct (Q_NEEDLES 16k),"
            echo "  then Q_TASKS=ifeval Q_NEEDLES="
          elif [[ "$E07_PLAN" == qfork ]]; then
            echo "  qfork's split with the model from HF (NETWORK_VOLUME=), 3 sessions: Q_TASKS=gsm8k_cot_llama Q_NEEDLES=,"
            echo "  Q_TASKS=humaneval_instruct (Q_NEEDLES 16k), Q_TASKS=ifeval Q_NEEDLES="
          fi
          over+=("$path ($MAX_SESSION_USD USD)"); }
    fi
    if (( timed )) && [[ "$path" == hf ]]; then
      read -r before dl rest <<<"$(python3 - "$HF_GBS" "${plan_head[@]}" "${pod_hf[@]}" <<'EOF'
import sys
steps = [(s.rsplit(":", 1)[0].split()[0], float(s.rsplit(":", 1)[1])) for s in sys.argv[2:]]
k = [n for n, _ in steps].index("model-download")
print(round(sum(m for _, m in steps[:k]), 1), steps[k][1], round(sum(m for _, m in steps[k + 1:]), 1))
EOF
)"
      echo "HF download guard (pod/hf-download.sh): 180 s into model-download its rate is the bytes"
      echo "  on disk over the time; when now + (264.5 GB - those bytes) / rate passes the deadline less 30 s less the rest of"
      echo "  the plan after it ($rest min of estimates), aria2c stops, the partial files go, out/steps.log says 'stop: download"
      echo "  too slow: X GB/s (...)' and the session pulls and deletes the pod. At $HF_GBS GB/s the download ends at minute"
      echo "  $(python3 -c "print(round($before + $dl, 1))") of the pod's $(( cap - 2 ))-minute deadline, $(python3 -c "print(round($cap - 2 - $before - $dl, 1))") min before it, for the $rest min after it:" \
        "$(python3 -c "print('the guard lets it run' if $cap - 2 - 0.5 - $before - $dl >= $rest else 'the guard stops it')")"
    fi
    # the e09 session's HF plan within 1.50 USD at the limit price
    if [[ "$E07_PLAN" == e09 && "$path" == hf ]]; then
      python3 -c "import sys; sys.exit(0 if $total / 60 * $MAX_USD_PER_HR <= 1.50 else 2)" ||
        { echo "e07.sh: E07_PLAN=e09's HF plan costs more than 1.50 USD at $MAX_USD_PER_HR USD/h"; over+=("hf (1.50 USD)"); }
    fi
    unset -n steps
  done
  # each path once with its reasons: "volume" (over the pod's cap) and "volume (3.00 USD)" -> "volume (the 92 min cap,
  # 3.00 USD)"; a path over the cap alone stays bare, as before
  declare -A why=()
  paths=()
  for o in "${over[@]}"; do
    p="${o%% *}" r="the $cap min cap"
    [[ "$o" == *"("* ]] && { r="${o#*(}"; r="${r%)}"; }
    [[ -v "why[$p]" ]] || paths+=("$p")
    why[$p]+="${why[$p]:+, }$r"
  done
  listed=()
  for p in "${paths[@]}"; do
    [[ "${why[$p]}" == "the $cap min cap" ]] && listed+=("$p") || listed+=("$p (${why[$p]})")
  done
  (( ${#over[@]} == 0 )) || { echo "e07.sh: the plan is over the cap with the model from: ${listed[*]}"; exit 2; }
  # a GPU in the list that fails the gate here would stop its pod after the create: refuse the list
  (( ${#gate_failed[@]} == 0 )) || { echo "e07.sh: the gate fails on $(IFS=';'; echo "${gate_failed[*]}"): drop it from GPU_TYPES"; exit 2; }
  exit 0
fi

(( payload_only )) || [[ -n "${RUNPOD_API_KEY:-}" ]] || { echo "e07.sh: RUNPOD_API_KEY is not set: export your RunPod API key" >&2; exit 1; }
(( payload_only )) || [[ -f "$SSH_KEY" && -f "$SSH_KEY.pub" ]] || { echo "e07.sh: no SSH key pair $SSH_KEY(.pub)" >&2; exit 1; }
mkdir -p "$OUT"
log "results in $OUT"
# e07_report.py's plan when no out/plan.log comes back (a session whose every machine was rejected)
echo "$E07_PLAN" > "$OUT/session-plan"
# and the session's cap, for the sheet's real cost against it (ccache's 2.30)
echo "$MAX_SESSION_USD" > "$OUT/session-cap"
prompts_ready >> "$SESSION_LOG"
hf_manifest_ready >> "$SESSION_LOG"
if (( qc )); then
  q_payload_sources | tee -a "$SESSION_LOG" >&2 || { log "FAIL: the $E07_PLAN payload's sources (above)"; exit 1; }
fi

# the payloads, one per flavor (pod-session.sh uploads the created pod's: stages[pod_tag])
declare -A stages=()
for t in "${tags[@]}"; do
  stage="$OUT/payload-$t"
  read -ra bins <<<"$(bins_of "$t")"
  read -ra fork_libs <<<"$(fork_libs_of "$t")"
  stage_bins "${strata_tgz[$t]}" "$stage" "${bins[@]}"
  stage_bins "${fork_tgz[$t]}" "$stage/fork" "${fork_libs[@]}"
  want="$(awk '$2 == "bin/libllama.so.0" { print $1 }' "$stage/fork/MANIFEST")"
  got="$(sha256sum "$(dirname "${tokenize[$t]}")/libllama.so.0.4.0" | cut -d' ' -f1)"
  [[ "$want" == "$got" ]] || { log "FAIL: ${tokenize[$t]} is not the build of ${fork_tgz[$t]} (libllama.so.0 $got, MANIFEST $want)"; exit 1; }
  cp "${tokenize[$t]}" "$stage/fork/"
  ln -sf "../libcublas.so.${cuda_major[$t]}" "$stage/fork/libcublas.so.${cuda_major[$t]}"
  ln -sf "../libcublasLt.so.${cuda_major[$t]}" "$stage/fork/libcublasLt.so.${cuda_major[$t]}"
  cp "${pod_files[@]/#/$here/}" "$stage/"
  # the probe beside the pod files, uploaded on its own before the payload
  if (( QUALIFY )); then
    mp="$(machine_probe_ready "$t")" || exit 1
    log "$mp"
    cp "$MACHINE_PROBE_ROOT/$t/machine-probe/dsv41_machine_probe" "$stage/"
  fi
  mkdir -p "$stage/e07/oracle" "$stage/src"
  cp "$COUNTS" "$stage/e07/counts.json"
  cp "$here/e07_chats.json" "$stage/e07/chats.json"
  echo "$E07_PLAN" > "$stage/e07/plan"
  echo "$E07_READERS" > "$stage/e07/readers"
  cp "$HF_MANIFEST" "$stage/e07/hf-manifest.json"
  [[ "$E07_PLAN" != e07b ]] ||
    python3 "$here/e07_prompts.py" turns "$PROMPTS" --vocab "$VOCAB" --out "$stage/e07/turns.json" >> "$SESSION_LOG"
  if (( timed )); then
    estimates volume > "$stage/e07/estimates-volume"
    estimates hf > "$stage/e07/estimates-hf"
  fi
  [[ "$E07_PLAN" != qengine && "$E07_PLAN" != qdiag ]] || echo "$Q_SPEED_MAX_MS" > "$stage/e07/speed-max-ms"
  if [[ "$E07_PLAN" == qdiag ]]; then
    qdiag_subsets "$stage/e07/subsets-qdiag.json"
  elif (( pc )); then
    cp "$CBASE_CONVERSATIONS" "$stage/e07/conversations.json"
    echo "$LOAD_USERS" > "$stage/e07/$(plan_users)-users"
    [[ "$E07_PLAN" != ccache ]] || echo "$CCACHE_ALT_USERS $CCACHE_ALT_TURNS $CCACHE_THINK_S" > "$stage/e07/ccache-alt"
    if (( has_cbatch )); then
      cp "$CBATCH_SHORT" "$stage/e07/conversations-short.json"
      echo "$CBATCH_SLOTS $CBATCH_CTX $CBATCH_CHECKPOINTS" > "$stage/e07/cbatch-batch"
      echo "$CBATCH_STEADY_USERS $CBATCH_STEADY_WINDOW_S $CBATCH_STEADY_REPEATS $CBATCH_STEADY_MAX_TOKENS" > "$stage/e07/cbatch-steady"
      echo "$CBATCH_STEADY_REPORT" > "$stage/e07/cbatch-steady-report"
      echo "$CBATCH_SHORT_TURNS" > "$stage/e07/cbatch-short"
    fi
    if (( has_clong )); then
      cp "$CLONG_CONVERSATIONS" "$stage/e07/conversations-long.json"
      echo "$CLONG_SLOTS $CLONG_CTX $CLONG_CHECKPOINTS $CLONG_USERS $CLONG_MAX_TOKENS" > "$stage/e07/clong"
    fi
    (( ! has_spare )) || echo "$CBATCH_SPARE" > "$stage/e07/cbatch-spare"
    # item 3 batched, its concurrent requests and tasks in order
    (( ! has_c05 )) || echo "$CDET_CONCURRENT $CDET_TASKS" > "$stage/e07/cdet-quality"
  elif [[ "$E07_PLAN" == q* && "$E07_PLAN" != qdet ]]; then
    echo "$Q_TASKS" > "$stage/e07/qtasks"
    echo "$Q_NEEDLES" > "$stage/e07/qneedles"
    [[ "$E07_PLAN" != qfork ]] || { echo "$Q_FORK_NP" > "$stage/e07/fork-np"; echo "$Q_FORK_MAX_MS" > "$stage/e07/fork-max-ms"; }
    cp "$QUALITY_NEEDLE" "$stage/e07/needle.json"
  fi
  cp -r "$PROMPTS" "$stage/e07/prompts"
  cp "$ORACLE/dump-manifest.json" "$ORACLE"/{cur,out,topk,weights}-L*.bin "$stage/e07/oracle/"
  tar -C "$STRATA_DIR" -cf - --exclude="test_*.py" --exclude=__pycache__ "${src[@]}" | tar -C "$stage/src" -xf -
  tar -C "$stage" -czf "$OUT/payload-$t.tar.gz" MANIFEST "${pod_files[@]}" fork e07 src "${bins[@]}"
  stages[$t]="$stage"
  log "payload-$t $(du -h "$OUT/payload-$t.tar.gz" | cut -f1): ${bins[*]} from $(basename "${strata_tgz[$t]}"), fork/ from $(basename "${fork_tgz[$t]}"), the oracle from $ORACLE, prompts from $PROMPTS"
done
# --payload-only: the payloads as the session would upload them, for inspection; no pod
(( ! payload_only )) || { log "payload only: $(cd "$OUT" && echo payload-*.tar.gz) in $OUT, no pod"; exit 0; }

# when the final pull brought nothing back (the pod lost, or gone at the cap), what the mid-run pulls brought
# (pull_steps) is the session's record: OUT/session-lost and e07.log name the step the session was lost after, the
# steps finished and pulled, one a poll saw finish after the last pull, and the plan's steps (the payload's
# e07/estimates-<path> of the pod's model path, the timed plans') with no line pulled; the sheet's "Session lost"
# section is that file
session_lost() {
  local f="$OUT/out/steps.log" est names=() left=() n l last
  [[ ! -f "$OUT/out-logs.tar.gz" ]] || return 0
  [[ -f "$f" ]] && mapfile -t names < <(sed -n 's/^step \([^:]*\): .*/\1/p' "$f")
  last="$(tail -n 1 "$f" 2>/dev/null || true)"
  # pod_volume here is pod-session.sh's (the created pod's volume, "" without one)
  est="${stages[$pod_tag]:-}/e07/estimates-$([[ -n "$pod_volume" ]] && echo volume || echo hf)"
  if [[ -f "$est" ]]; then
    while read -r n _; do [[ " ${names[*]} " == *" $n "* ]] || left+=("$n"); done < "$est"
  fi
  {
    if (( ${#names[@]} )); then
      echo "lost after step ${names[-1]}: the final pull brought nothing back, out/ is the last mid-run pull's (out/steps.log's last line: $last)"
    else
      echo "lost before any step was pulled: the final pull brought nothing back$([[ -n "$last" ]] && echo ", out/steps.log's last line: $last")"
    fi
    echo "finished and pulled (out/steps.log): ${names[*]:-none}"
    [[ -z "$polled_step" || "$polled_step" == "$last" ]] ||
      echo "seen by a poll after the last pull, not pulled: $polled_step"
    if [[ -f "$est" ]]; then echo "the plan's steps with no line pulled (lost or never run): ${left[*]:-none}"
    else echo "the plan's steps: no ${est#"${stages[$pod_tag]:-}/"} in the payload (E07_PLAN=$E07_PLAN is not a timed plan)"
    fi
  } > "$OUT/session-lost"
  while IFS= read -r l; do log "$l"; done < "$OUT/session-lost"
}

# after the delete: the final pull's out/ over the mid-run pulls' (pull_steps), session_lost when it brought nothing,
# out/steps.log and the verdict lines into the session log, then the sheet
reported=0
session_sheet() {
  reported=1
  [[ -f "$OUT/out-logs.tar.gz" ]] && { tar -xzf "$OUT/out-logs.tar.gz" -C "$OUT" || log "WARNING: out-logs.tar.gz does not extract"; }
  session_lost
  [[ -f "$OUT/out/steps.log" ]] && tee -a "$SESSION_LOG" < "$OUT/out/steps.log" >&2
  # the machine a slow HF download stopped, for the runner to grep into the next session's BAD_MACHINES
  ! grep -q "^stop: download too slow" "$OUT/out/steps.log" 2>/dev/null || log "slow download machine: ${machine:-?}"
  grep -h "oracle_parity.py: \(PASS\|FAIL\)\|tokenizer_parity.py: \(PASS\|FAIL\)\|serve_check.py:\|e07_bench.py:\|e07_gate.py:" \
    "$OUT/out/"*.log 2>/dev/null | tee -a "$SESSION_LOG" >&2 || true
  python3 "$here/e07_report.py" "$OUT" "$SHEET_ROOT" 2>&1 | tee -a "$SESSION_LOG" >&2 || log "WARNING: e07_report.py failed"
}

# a session that never started its run still gets its sheet when a machine
# was tried, with each one's qualification values (the EXIT trap: cleanup first, the pods gone); so does one
# that started its run and ended before its sheet (the watchdog at the cap, a failed step of this script), from what
# was pulled
session_exit() {
  local rc=$?
  # cleanup reads the exit status from $?
  set +e
  (exit "$rc"); cleanup
  if (( started && !reported )); then
    session_sheet
  elif (( !started )) && grep -q "^=== pod " "$OUT/probe.log" 2>/dev/null; then
    python3 "$here/e07_report.py" "$OUT" "$SHEET_ROOT" 2>&1 | tee -a "$SESSION_LOG" >&2 || log "WARNING: e07_report.py failed"
  fi
  exit "$rc"
}
trap session_exit EXIT
trap 'exit 130' INT TERM

pod_up
# the optional token goes on the pod's stdin, never on a command line or into the payload
if [[ -z "$pod_volume" ]]; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    ssh "${ssh_opts[@]}" -p "$port" "root@$ip" 'umask 077 && cat > /root/.hf-token' <<<"$HF_TOKEN" ||
      { log "FAIL: sending HF_TOKEN to the pod"; exit 1; }
    log "no network volume: the model comes from HF, HF_TOKEN in the pod's /root/.hf-token"
  else
    log "no network volume: the model comes from HF, anonymously (no HF_TOKEN)"
  fi
fi
run_detached "$OUT/payload-$pod_tag.tar.gz"
pull_results
end_pod   # the pod goes as soon as its results are here, before anything else runs locally
session_sheet
# the session's verdict: the remote script's exit and every step's line (the EXIT trap still deletes the pod)
steps_verdict || exit 1
log "every step passed"
