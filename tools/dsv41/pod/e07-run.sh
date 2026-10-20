#!/usr/bin/env bash
# Runs on the pod of a real-model session (tools/dsv41/pod/e07.sh uploads and starts it):
# the real DeepSeek-V4.1-Flash Q2_K on one GPU, from the payload directory, each step under pod-steps.sh's `run`
# (timeout to DEADLINE, out/NAME.log, out/steps.log, the 30 s monitor, leftovers reaped), the cheapest and most
# important results first:
# The model comes from the network volume when the pod has it (POD_VOLUME, set by pod-session.sh: its id), else from
# Hugging Face (e07/hf-manifest.json, the repo at a pinned commit):
#   env           GPU (name, memory, PCIe link, driver), topology, CPU (lscpu), RAM, the memory cgroup, disks, MANIFEST
#   volume-before with the volume: the model's files on it (name, size, mtime, mode): the volume is only read, and
#                 volume-after must list the same at the end
#   model-headers without the volume: hf_model.py headers (each shard's head by HTTP range requests), then
#                 gguf_layout.py sparse: the real layout, the gate's model headers before the download
#   gate          e07_gate.py: the VRAM slab and the pinned warm tier of every engine run and the memory they need
#                 against the cgroup's limit (exit: the session stops here, before the copy)
#   setup         numactl and jinja2 when the image lacks them, and aria2c without the volume
#   model-copy    with the volume: model-copy.sh (t04a.sh's 28 dd streams) from /workspace/models/q2k to
#                 /root/models/q2k; without it, model-download: hf-download.sh (aria2c, the 7 files at once with 16
#                 HTTP range connections each, then each file's size and sha256 against the HF API's, the download's
#                 time and GB/s in out/model-download.json) into /root/models/q2k; then gguf_layout.py sizes: the 7
#                 shards byte for byte the sizes measured on the volume
#   dry-run       dsv41-engine --dry-run on the copy with the bench's arguments (every view and tier, no CUDA call)
#   forced-ref    oracle_parity.py forced: the stock fork's oracle dump (e07/oracle, run-1) as a --force-selection reference
#   <mode>-<sel>  correctness, per reader mode (dense: the fork-parity mode, then reuse, the default; at 95 positions
#                 they coincide, SPEC.md "Reader modes") and selection (forced, then free): dsv41-engine on the
#                 oracle's 31-token prompt (no BOS) and 64 greedy steps with --dump-layers, then oracle_parity.py
#                 against the oracle (forced: every layer's cur and out rel L2 <= 2e-2; free: >= 90% of the 64 ids,
#                 a first divergence a near-tie), its numbers in out/parity-<mode>-<sel>.json
#   layer-ref-<n> dsv41-layer-ref on the CPU, the oracle's prompt through
#                 layers 0 and 1 (the engram before layer 1) in numerics n = fork-cpu, fork-cuda and engine, every op
#                 dumped (out/layer-ref/<n>), with the oracle's expert ids and the isolated MoE of each layer
#   layer-parity  layer_parity.py compare: the three numerics against the oracle, against the dense-forced run's
#                 call 0 (the GPU against its references on the real weights) and op by op against each other, where
#                 the gap enters, and whether the oracle's matmul arithmetic explains it (out/layer-parity.json)
#   moe-isolated  dsv41-engine --readers dense with the oracle's selections and every
#                 layer's MoE input the oracle's cur (--forced-moe-input), then parity-moe-isolated: per layer the
#                 router's weights, its own top-k and the routed sum on the oracle's input (the per-layer isolated
#                 rel L2, out/parity-moe-isolated.json)
#   dense-forced-mmq  only when layer-parity reads that the oracle's matmul arithmetic explains the layer-0 gap
#                 (the decision in out/plan.log): dense-forced again with the prefill
#                 matmuls on the fork's MMQ (--prefill-mmq), then its parity and layer-parity-mmq
#   tokenizer     tokenizer_parity.py: tools/strata_tokenizer.py on the real GGUF against the fork's llama-tokenize on
#                 the 226-line corpus
#   bench-a       e07_bench.py 16k,turn2,short,short-timed: the 16K TTFT, the second turn of the 16K conversation
#                 (+512, the prefix cache: it resumes turn 1's live state, so it runs in turn 1's process), 480 decode
#                 steps at short context, then 64 timed ones (every phase's decode
#                 hit rate, CPU misses and doorbell waits from the engine's log, the timed phase's per-stage split)
#   serve         serve_check.py --parts profile,http on the real model: the profile on the real GGUF's template and
#                 tokenizer, /v1/chat/completions non-streaming and streaming (ids against a reference CLI run made
#                 before the server starts, --one-engine: one engine on the GPU and in pinned RAM at a time), the
#                 resumed turn (its ids reported), tools refused, then the 3 chat prompts of e07/chats.json with
#                 their replies saved
#   bench-b       e07_bench.py 64k,256k: TTFT and 480 decode steps at 64K, then the 256K prompt (261,664 ids at ctx
#                 262,144) and 480 decode steps; the engine must still answer afterwards
#   volume-after  with the volume: its listing again, equal to volume-before; it also runs when a step stops the
#                 session early (volume-before, gate, model-copy, dry-run)
#   bench-e08     e07_bench.py short-static,short-prompt,short-adaptive,16k-e08,
#                 short-timed,64k,256k on one engine load with --route-trace out/route-trace.txt: 480 decode steps at
#                 short context per hot-set policy (each from the hot set's slab: hot_reset=1; the hit rate per layer,
#                 the swaps), the 16K prompt's per-stage split (prefill_times=1) and 480 decode steps after it, the timed
#                 split of 64 steps, then 64K and 256K with the default policy (adaptive; left out of the trace)
#   route-trace   gzip of the trace (out/route-trace.txt.gz, pulled with the logs: tools/dsv41/engine/hot_policy_sim.py
#                 replays it locally)
#   bench-e09     e07_bench.py 16k-e09,short-adaptive,16k-e09,256k on one engine load: the 16K
#                 prompt twice with its per-stage split (prefill_times=1; short-adaptive between, so the second reads
#                 cold), 480 decode steps at short context with the default policy, the 256K prompt once
#   two-turn      engine/two_turn_check.py: a two-turn chat through serve/server.py with --route-trace: turn 2 resumed
#                 (cached tokens, the trace's R line resume=1) with its decode reported; turn 1 the 16K prompt's
#                 text (e07/turns.json, ctx 20480), turn 2 streamed (its first token at the client), then --http on the
#                 same server load: /v1/chat/completions non-streaming and streaming, each reply the text of the
#                 engine's ids, tools refused
#   bench-e07b    e07_bench.py 16k-e09,turn2,short-adaptive,64k,256k on one engine load
#   dense-free    the free-greedy oracle parity on this card (reported, not gating),
#                 then forced-ref, moe-isolated and their parity when the time allows
#   fork-server   qfork (Part Q): the fork's llama-server (quality/pod_server.sh fork: t04a.sh's stock flags, -np
#                 e07/fork-np (e07.sh's Q_FORK_NP, 1 by default) --kv-unified --ctx-size
#                 32768 --prefill-assistant, the threads from nproc and the cgroup's CPU quota) in its own session,
#                 ready at /health; the lm-eval steps after it send NUM_CONCURRENT of the same number
#   engine-server qengine: serve/server.py on ./dsv41-engine with the gate's placement (quality/pod_server.sh engine)
#   quality-venv  the lm-eval venv, installed in the background right after setup (run_lm_eval.sh INSTALL_ONLY=1:
#                 out/quality-venv.log), while the model downloads and loads; the step waits for its exit status
#   continuation  the HumanEval continuation on 3 items (quality/continuation.py; qfork: llama-server's /apply-template
#                 prompt, then both: the replies through the proxy, out/quality/continuation.json): a failure stops
#                 the server and the session before the full run
#   speed-check   qengine and qdiag: pod/speed_check.py on out/quality-engine.log (the
#                 continuation's requests): the median ms per decode step against e07/speed-max-ms (e07.sh's
#                 Q_SPEED_MAX_MS, 30 by default), with the machine id POD_MACHINE (RunPod's machineId of the create,
#                 pod-session.sh); above it qengine stops the server and the session with out/steps.log's line
#                 "stop: engine too slow on this machine: X ms per decode step (machine M)"; qdiag carries on (it
#                 measures the machine whatever its speed); qfork: speed_check.py --fork
#                 on out/quality-server.log (llama-server's print_timing lines of the continuation's replies), the
#                 median ms per generated token against e07/fork-max-ms (e07.sh's Q_FORK_MAX_MS, 80 by default),
#                 above it the same stop with "stop: fork too slow on this machine: X ms per token (machine M)"
#   lm-eval-<t>   quality/eval.sh on each task of e07/qtasks (llamacpp or strata backend), out/quality/
#   needle-<n>    quality/eval.sh needle-<n> for each length of e07/qneedles (e07/needle.json)
#   server-stop   quality/pod_server.sh stop
#   bench-qdiag   qdiag: e07_bench.py short-static,short-adaptive on one engine load, no
#                 server (the static against the adaptive hot set on the e07b session's comparable case), before engine-server
#   lm-eval-gsm8k-c1, lm-eval-gsm8k-c2  qdiag: quality/eval.sh gsm8k_cot_llama on the first 20 items of the fixed-seed
#                 subsets (e07/subsets-qdiag.json) with NUM_CONCURRENT 1, then 2, into out/quality/c1 and c2
#   det-sampling  qdet: quality/det_sampling_check.py over HTTP on the server
#                 qengine's engine-server starts (its engine with --log-ids too), the server's first requests after
#                 its load: items 7 (greedy determinism, the first divergent position on a mismatch) and 8 (seeded
#                 sampling, its speed against greedy, a penalty's 400) into out/det-sampling.json; in the
#                 engine's deterministic mode, then the matrix row with it off and the mode's decode cost
#   load-c<N>     cbase: quality/chat_load.py run on the server engine-server starts, for each N
#                 of e07/cbase-users (e07.sh's CBASE_USERS, "1 2 4 8" by default): the first N users of
#                 e07/conversations.json (16K tokens each, built locally) started together, 2 streamed turns of
#                 256 tokens each, every request's TTFT, the server's queue wait and decode into
#                 out/cbase/load-c<N>.json
#   load-alt<N>   ccache: quality/chat_load.py run --alternate on that server,
#                 the first N users of e07/conversations.json (e07/ccache-alt: N and the turns, e07.sh's
#                 CCACHE_ALT_USERS 4 and CCACHE_ALT_TURNS 3) taking turns, each request sent when the one before it
#                 ended, into out/ccache/load-alt<N>.json (item 10: the turns 2+ resumed and their first token, each
#                 request's wait for the idle reset from out/quality-engine.log)
#   load-alt<N>-think<S>  ccache: the same with S seconds (e07/ccache-alt's third field,
#                 e07.sh's CCACHE_THINK_S 5) between a reply's end and the next turn, into out/ccache/
#                 load-alt<N>-think<S>.json (informational)
# e07/plan (e07.sh's E07_PLAN) picks the steps: full runs every one; cycle2 (the second
# session) skips tokenizer, serve and bench-b, which session 1 measured (PASS, PASS, 64K and 256K); e08 runs the gate,
# the model, dry-run, bench-e08, route-trace and serve (the 3-chat quality check) only; e09 the gate, the model,
# dry-run, bench-e09 and two-turn, every engine with a prefetch of E09_PREFETCH (128) experts (the gate counts the
# region). e07b the gate, the model, dry-run, two-turn (with the HTTP check), bench-e07b, then
# dense-free with its parity, forced-ref, moe-isolated and its parity, every engine with that prefetch too; each step
# gets its own timeout, min(2 x its estimate in e07/estimates-<volume|hf>, the time left: pod-steps.sh), and the HF
# download stops early when its throughput after 3 min cannot finish it in time to run the rest of the plan
# (hf-download.sh's guard: out/steps.log then says "stop: download too slow: X GB/s"). e07/readers (E07_READERS)
# lists the correctness runs' reader modes, dense first (default "dense reuse"; e08 runs none). qfork and qengine
# (Part Q): the model, setup, the venv in the background, then fork-server or (after the
# gate and the dry run) engine-server, quality-venv, continuation, speed-check, lm-eval-<task> per
# e07/qtasks, needle-<n> per e07/qneedles and server-stop; qfork has no model-headers, gate nor dry-run (no engine);
# each step's timeout as e07b's.
# qdiag runs qengine's steps up to the dry run, then bench-qdiag, engine-server,
# quality-venv, continuation, speed-check, lm-eval-gsm8k-c1, lm-eval-gsm8k-c2 and server-stop. qdet
# runs qengine's steps up to the dry run (no lm-eval venv), then engine-server,
# det-sampling and server-stop, every engine with the e09 session's prefetch. cbase runs qdet's steps up to
# the dry run, then engine-server (qengine's server), load-c<N> per e07/cbase-users and server-stop. ccache
# runs cbase's steps up to the dry run, then engine-server (its engine with
# --log-ids), load-alt<N>, load-alt<N>-think<S>, load-c<N> per e07/ccache-users into out/ccache/, det-sampling (qdet's) and server-stop.
# cbatch runs cbase's steps up to the dry run with a second gate (gate-off: the
# placement without the batch's slots), then engine-server-off (the FIFO server on gate-off's placement), load-short-off,
# server-stop-off, engine-server (the engine with --batch, --batch-ctx, --checkpoints from e07/cbatch-batch and
# --log-ids), load-short, load-c<N> per e07/cbatch-users into out/cbatch/, steady-c<N> (chat_load.py steady with
# e07/cbatch-steady's users, window, repeats and reply length), steady-c<N> per
# e07/cbatch-steady-report (the same, reported), det-sampling and server-stop; load-short-off and load-short take
# e07/cbatch-short's turns.
# clong runs cbase's steps up to the dry run with the gate of e07/clong's batch
# slots and checkpoints, then engine-server (the engine with those slots, --log-ids), load-long (chat_load.py long on
# e07/conversations-long.json: e07/clong's users and their max_tokens), robust (robust_check.py) into out/clong/ and
# server-stop. cbatch-clong runs cbatch's steps less load-c<N> and det-sampling (measured on
# 2026-10-14) with a third gate (gate-long: clong's slots) after gate-off, the batch server's logs kept as
# out/cbatch-<file> at its stop, then clong's steps as engine-server-long, load-long, robust and server-stop-long;
# with e07/cbatch-spare a fourth gate (gate-spare: gate's batch with
# --hot-spare N) after gate-long, and last a second cbatch server on it, engine-server-spare (--hot-spare N),
# load-short-spare, steady-c<N>-spare into out/cbatch/ and server-stop-spare, its logs kept as out/spare-<file>.
# cdet runs cbatch's steps less load-c<N>, then on the batched server det-batch
# (det_batch_check.py: item 14 into out/det-batch.json), det-sampling, quality-venv, continuation (qengine's: exit
# stops the session) and lm-eval-<task> per e07/cdet-quality (its NUM_CONCURRENT and tasks in order, eval.sh's strata
# backend into out/quality), then server-stop. cdet has no
# batching-off server (nor gate-off): the single user's speed A/B replaces it, ab-gpu (load-short's user with the
# deterministic mode off, chat_load.py --deterministic off) on the batched server after load-short, and after
# server-stop engine-server-fork (the same server with --cpu-numerics fork), ab-fork (the same user) and
# server-stop-fork, its logs kept as out/fork-<file> (the batched server's put back as they were). The
# fork's numerics are the engine's default, so cdet's batched server says --cpu-numerics gpu (the A/B's gpu arm as it
# ran). cfinal (item 14 is no quality loss from batching) runs cdet's steps less
# ab-gpu, det-batch, det-sampling and the fork server: engine-server (the default numerics, the fork's), load-short,
# steady-c<N>, quality-venv, continuation, lm-eval-<task> per e07/cdet-quality and server-stop.
# qengine and qdiag
# start pod-samplers.sh right after env (the GPU's clocks, power and throttle reasons every 1 s into out/gpu-clocks.csv, the
# host's load and cgroup CPU throttling every 2 s into out/host-load.log) and stop it on exit; pod-steps.sh writes each
# step's window to out/step-times.log, which tools/dsv41/pod/qdiag_report.py joins to them.
# Every engine run takes the gate's placement (the hot set from the real counts profile in VRAM, every other expert
# in the pinned warm tier, 16 CPU threads; --hot-homes all when the gate found the RAM for the
# slab's host homes, --hot-homes N when it holds N of them, else --hot-homes spare, the reason in
# out/placement.txt), bound to the GPU's NUMA node (numactl with its memory policy, else its CPUs only, else
# taskset; the forms tried and why each failed in out/placement.txt and out/env.log; unbound on one node). Exits 1
# when a step failed or was skipped; the last line of out/steps.log is "done".
set -uo pipefail
: "${DEADLINE:?}"
POD_VOLUME="${POD_VOLUME:-}"
cd "$(dirname "${BASH_SOURCE[0]}")"
PLAN="$(cat e07/plan)" READERS="$(cat e07/readers)"
case "$PLAN" in full|cycle2|e08|e09|e07b|qfork|qengine|qdiag|qdet|cbase|ccache|cbatch|clong|cbatch-clong|cdet|cfinal) ;;
  *) echo "e07-run.sh: e07/plan is '$PLAN' (full, cycle2, e08, e09, e07b, qfork, qengine, qdiag, qdet, cbase, ccache, cbatch, clong, cbatch-clong, cdet or cfinal)" >&2; exit 2 ;;
esac
# Part C's load plans (cbase, ccache, cbatch, clong, cbatch-clong, cdet, cfinal); the plans with cbatch's steps,
# with clong's; cdet's own (and plain cbatch alone runs load-c<N>); has_c05: cdet and cfinal, the
# batched server only and item 3 batched on it
has_cbatch=0 has_clong=0 has_cdet=0 has_c05=0 cbatch_loads=0
[[ "$PLAN" == cbatch || "$PLAN" == cbatch-clong || "$PLAN" == cdet || "$PLAN" == cfinal ]] && has_cbatch=1
[[ "$PLAN" == clong || "$PLAN" == cbatch-clong ]] && has_clong=1
[[ "$PLAN" == cdet ]] && has_cdet=1
[[ "$PLAN" == cdet || "$PLAN" == cfinal ]] && has_c05=1
[[ "$PLAN" == cbatch ]] && cbatch_loads=1
pc=0
[[ "$PLAN" == cbase || "$PLAN" == ccache ]] || (( has_cbatch || has_clong )) && pc=1
[[ "$PLAN" == e08 || "$PLAN" == e09 || "$PLAN" == q* || $pc == 1 || " $READERS " == *" dense "* ]] || { echo "e07-run.sh: e07/readers is '$READERS': dense is needed (layer-parity reads dense-forced)" >&2; exit 2; }
# cgroup_limit_bytes from t04a.sh's helpers; pod-steps.sh after it, so `run` is the step runner's
# shellcheck source=tools/dsv41/pod/t04a-lib.sh
source t04a-lib.sh
# shellcheck source=tools/dsv41/pod/pod-steps.sh
source pod-steps.sh
# every step's estimate for this pod's model path (e07.sh's plan), so each gets its own timeout
ESTIMATES="e07/estimates-$([[ -n "$POD_VOLUME" ]] && echo volume || echo hf)"
[[ ( "$PLAN" == e07b || "$PLAN" == q* || $pc == 1 ) && -f "$ESTIMATES" ]] && STEP_ESTIMATES="$ESTIMATES"
steps_begin

# the rehearsal of the q plans locally (quality/q02-test.sh): E07_POD_ROOT stands in for the pod's / (the model's
# and the venv's paths) and E07_Q_PORT for the Part Q server's port; both unset on a pod
POD_ROOT="${E07_POD_ROOT:-}"
MODEL_SRC=$POD_ROOT/workspace/models/q2k
MODEL_DIR=$POD_ROOT/root/models/q2k
SHARD1=DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf
SRC=src/tools/dsv41
Q=src/tools/dsv41/quality
# the prompts' manifest (their sha256) comes back with the logs, for the session sheet
cp e07/prompts/manifest.json out/prompts-manifest.json

run env bash -c 'date -u; nvidia-smi; nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,driver_version --format=csv
  nvidia-smi topo -m; lscpu; nproc; free -g; grep MemTotal /proc/meminfo
  for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do [ -r "$f" ] && echo "$f: $(cat "$f")"; done
  df -h /root; [ -d /workspace ] && df -h /workspace; mount | grep -w /workspace; echo "RLIMIT_MEMLOCK $(ulimit -l)"; cat MANIFEST
  echo "host PCIe link (the host side and the GPU side, when the driver reports them):"
  nvidia-smi --query-gpu=pcie.link.gen.hostmax,pcie.link.gen.gpumax,pcie.link.width.max --format=csv || true
  echo "CPU placement (the CPU experts ran at 44-45 GB/s on some hosts, 100.4 on others):"
  cpuset="" quota=""
  for f in /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/cpuset/cpuset.effective_cpus; do
    [ -r "$f" ] && cpuset="$cpuset $f: $(cat "$f")"; done
  for f in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us; do
    [ -r "$f" ] && quota="$quota $f: $(cat "$f")"; done
  echo "cgroup CPU set:${cpuset:- none (no cpuset.cpus.effective, v2 or v1)}"
  echo "cgroup CPU quota:${quota:- none (no cpu.max, v2, nor cpu.cfs_quota_us, v1)}"
  mhz="$(lscpu | sed -n "s/^CPU max MHz: *//p")" tpc="$(lscpu | sed -n "s/^Thread(s) per core: *//p")"
  echo "lscpu max MHz: ${mhz:-none (lscpu has no CPU max MHz)}, threads per core: ${tpc:-unknown}, SMT control: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo none)"
  echo "SMT siblings of the CPUs this step may run on ($(grep Cpus_allowed_list /proc/self/status | cut -f2)): $(for c in $(python3 -c "import os; print(*sorted(os.sched_getaffinity(0)))"); do
    cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list; done | sort -un | paste -sd" ")"'
# the engine's quality plans : the GPU's clocks, power and throttle reasons
# every 1 s and the host's load and CPU throttling every 2 s for the whole session (pod-samplers.sh), stopped on exit
if [[ "$PLAN" == qengine || "$PLAN" == qdiag ]]; then
  bash pod-samplers.sh start
  trap 'kill "$monitor_pid" 2>/dev/null; bash pod-samplers.sh stop' EXIT
fi
listing() { find "$MODEL_SRC" -maxdepth 1 -type f -printf '%f %s %T@ %m\n' | sort; }
# the read-only proof, with the volume: volume-after runs on every way out, an early stop() included
volume_after() {
  [[ -n "$POD_VOLUME" ]] || return 0
  run volume-after bash -c "$(declare -f listing); MODEL_SRC=$MODEL_SRC; listing > out/volume-after.txt;
    diff out/volume-before.txt out/volume-after.txt && echo 'the volume is unchanged'"
}
stop() { echo "$1" >> out/steps.log; volume_after; echo done >> out/steps.log; exit 1; }
if [[ -n "$POD_VOLUME" ]]; then
  run volume-before bash -c "$(declare -f listing); MODEL_SRC=$MODEL_SRC; listing > out/volume-before.txt; cat out/volume-before.txt;
    [ \$(grep -c '\.gguf ' out/volume-before.txt) -eq 7 ]" || stop "stop: the volume $POD_VOLUME does not hold the 7 shards at $MODEL_SRC"
  HEADERS_SHARD1="$MODEL_SRC/$SHARD1"
elif [[ "$PLAN" != qfork ]]; then
  run model-headers bash -c "python3 hf_model.py headers e07/hf-manifest.json e07/headers &&
    python3 $SRC/engine/gguf_layout.py sparse e07/headers e07/layout" ||
    stop "stop: the model's headers from Hugging Face failed (out/model-headers.log)"
  HEADERS_SHARD1="e07/layout/$SHARD1"
fi

# no readable limit: MemTotal is the bound (e07_gate.py)
cgroup_limit() { local v; v="$(cgroup_limit_bytes)"; [[ "$v" == absent ]] && echo unlimited || echo "$v"; }
vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
# the next layer's experts copied during its front (experts.hpp prefetch_experts; the default 32 otherwise)
E09_PREFETCH=128
PREFETCH=32
[[ "$PLAN" == e09 || "$PLAN" == e07b || "$PLAN" == qengine || "$PLAN" == qdiag || "$PLAN" == qdet || $pc == 1 ]] &&
  PREFETCH=$E09_PREFETCH
# qfork runs no engine: no gate, no placement
# cbatch: the batch's slots, their positions and the conversation checkpoints
# clong: its slots (each holds the 64K prompt and its reply), checkpoints,
# users and their max_tokens; its gate is the plan's gate alone, the combined plan's third (gate-long)
BATCH_GATE=() BATCH_ARGS=""
if (( has_cbatch )); then
  read -r b_slots b_ctx b_ckpt < e07/cbatch-batch
  BATCH_GATE=(--checkpoints "$b_ckpt") BATCH_ARGS="--batch $b_slots --batch-ctx $b_ctx"
fi
if (( has_clong )); then
  read -r l_slots l_ctx l_ckpt l_users l_max < e07/clong
  LONG_ARGS="--checkpoints $l_ckpt --batch $l_slots --batch-ctx $l_ctx"
  (( has_cbatch )) || BATCH_GATE=(--checkpoints "$l_ckpt") BATCH_ARGS="--batch $l_slots --batch-ctx $l_ctx"
fi
if [[ "$PLAN" != qfork ]]; then
  # shellcheck disable=SC2086  # the batch's arguments are whole words
  run gate python3 src/tools/dsv41/pod/e07_gate.py --vram-mib "$vram_mib" --cgroup-bytes "$(cgroup_limit)" --prefetch-experts "$PREFETCH" \
    --mem-total-kib "$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)" --counts e07/counts.json \
    --layout "$SRC/oracle/layout-dsv41-q2k.json" --model-headers "$HEADERS_SHARD1" --out out/gate "${BATCH_GATE[@]}" $BATCH_ARGS ||
    stop "stop: the gate failed (out/gate.log): this pod cannot hold the placement"
  gate() { python3 -c "import json, sys; print(json.load(open('out/gate/gate.json'))[sys.argv[1]])" "$1"; }
  PLACE="--hot-set out/gate/hot-set.json --vram-gb $(gate vram_gb) --ram-gb $(gate ram_gb) --threads 16 --hot-homes $(gate hot_homes)"
  HOMES_WHY="$(gate host_homes_reason)"
  [[ "$PLAN" == e09 || "$PLAN" == e07b || "$PLAN" == qengine || "$PLAN" == qdiag || "$PLAN" == qdet || $pc == 1 ]] &&
    PLACE+=" --prefetch-experts $PREFETCH"
fi
# cbatch: the same placement without the batch (today's slab), for the single user's before figure (not cdet: its A/B
# replaces the batching-off server; nor cfinal: steady-c<N>'s base is the single user on the batched server too)
if (( has_cbatch && ! has_c05 )); then
  run gate-off python3 src/tools/dsv41/pod/e07_gate.py --vram-mib "$vram_mib" --cgroup-bytes "$(cgroup_limit)" --prefetch-experts "$PREFETCH" \
    --mem-total-kib "$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)" --counts e07/counts.json \
    --layout "$SRC/oracle/layout-dsv41-q2k.json" --model-headers "$HEADERS_SHARD1" --out out/gate-off "${BATCH_GATE[@]}" ||
    stop "stop: the gate without the batch failed (out/gate-off.log)"
  goff() { python3 -c "import json, sys; print(json.load(open('out/gate-off/gate.json'))[sys.argv[1]])" "$1"; }
  PLACE_OFF="--hot-set out/gate-off/hot-set.json --vram-gb $(goff vram_gb) --ram-gb $(goff ram_gb) --threads 16 --hot-homes $(goff hot_homes)"
  PLACE_OFF+=" --prefetch-experts $PREFETCH --checkpoints $b_ckpt"
fi
# clong: its placement (the plan's gate), or in the combined plan gate-long's
if [[ "$PLAN" == cbatch-clong ]]; then
  # shellcheck disable=SC2086  # the batch's arguments are whole words
  run gate-long python3 src/tools/dsv41/pod/e07_gate.py --vram-mib "$vram_mib" --cgroup-bytes "$(cgroup_limit)" --prefetch-experts "$PREFETCH" \
    --mem-total-kib "$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)" --counts e07/counts.json \
    --layout "$SRC/oracle/layout-dsv41-q2k.json" --model-headers "$HEADERS_SHARD1" --out out/gate-long \
    --checkpoints "$l_ckpt" --batch "$l_slots" --batch-ctx "$l_ctx" ||
    stop "stop: the gate of the long prompt's slots failed (out/gate-long.log)"
  glong() { python3 -c "import json, sys; print(json.load(open('out/gate-long/gate.json'))[sys.argv[1]])" "$1"; }
  PLACE_LONG="--hot-set out/gate-long/hot-set.json --vram-gb $(glong vram_gb) --ram-gb $(glong ram_gb) --threads 16"
  PLACE_LONG+=" --hot-homes $(glong hot_homes) --prefetch-experts $PREFETCH"
elif (( has_clong )); then
  PLACE_LONG="$PLACE"
fi
# the combined plan's second cbatch server (e07/cbatch-spare: its spare
# slab slots per layer), gate's batch with the slab leaving their room
SPARE=""
[[ "$PLAN" == cbatch-clong && -f e07/cbatch-spare ]] && SPARE="$(cat e07/cbatch-spare)"
if [[ -n "$SPARE" ]]; then
  # shellcheck disable=SC2086  # the batch's arguments are whole words
  run gate-spare python3 src/tools/dsv41/pod/e07_gate.py --vram-mib "$vram_mib" --cgroup-bytes "$(cgroup_limit)" --prefetch-experts "$PREFETCH" \
    --mem-total-kib "$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)" --counts e07/counts.json \
    --layout "$SRC/oracle/layout-dsv41-q2k.json" --model-headers "$HEADERS_SHARD1" --out out/gate-spare \
    "${BATCH_GATE[@]}" $BATCH_ARGS --hot-spare "$SPARE" ||
    stop "stop: the gate with $SPARE spare slots failed (out/gate-spare.log)"
  gspare() { python3 -c "import json, sys; print(json.load(open('out/gate-spare/gate.json'))[sys.argv[1]])" "$1"; }
  PLACE_SPARE="--hot-set out/gate-spare/hot-set.json --vram-gb $(gspare vram_gb) --ram-gb $(gspare ram_gb) --threads 16"
  PLACE_SPARE+=" --hot-homes $(gspare hot_homes) --prefetch-experts $PREFETCH --hot-spare $SPARE"
fi

# aria2c only for the download (without the volume); the Part Q plans: a python3 that makes venvs (virtualenv when
# the image's has no ensurepip)
run setup bash -c 'command -v numactl || { apt-get update -qq && apt-get install -y -qq numactl; }
  [ -n "$1" ] || command -v aria2c || { apt-get update -qq && apt-get install -y -qq aria2; }
  python3 -c "import jinja2" 2>/dev/null || pip install -q jinja2==3.1.6
  case "$2" in q*|cdet|cfinal) python3 -c "import ensurepip" 2>/dev/null || pip install -q virtualenv ;; esac' setup "$POD_VOLUME" "$PLAN"
# Part Q: the lm-eval venv in the background, while the model downloads and loads (quality-venv waits for it);
# qdet runs no lm-eval; cdet's and cfinal's item 3 does
if [[ "$PLAN" == q* && "$PLAN" != qdet ]] || (( has_c05 )); then
  ( INSTALL_ONLY=1 QUALITY_VENV="$POD_ROOT/root/venv-quality" bash "$Q/run_lm_eval.sh" out/quality > out/quality-venv.log 2>&1
    echo $? > out/quality-venv.rc ) &
fi
NODE="$(nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/ { print $(NF-1) }')"
# numactl's memory policy, else its CPUs only, else taskset on the node's CPU list (t04a-lib.sh numa_binding;
# the 20261019T203344Z session ran unbound at 52-56 GB/s against the probe's 106.4 on node 0)
numa_binding "$NODE"
NUMA=("${NUMA_BIND[@]}")
# qfork has no gate, so no PLACE (`set -u` ended its session here)
[[ "$PLAN" == qfork ]] && shown="none (qfork runs the fork's stock flags)" || shown="$PLACE"
echo "engine runs bound with: ${NUMA[*]:-nothing}; placement: $shown" > out/placement.txt
printf '%s\n' "${NUMA_BIND_LOG[@]}" | tee -a out/env.log >> out/placement.txt
# the CPUs the engine's 16 pool threads run on, the affinity it inherits under the same binding
echo "engine pool threads (--threads 16) run on CPUs $("${NUMA[@]}" grep Cpus_allowed_list /proc/self/status | cut -f2) (bound with: ${NUMA[*]:-nothing})" >> out/env.log
[[ -z "${HOMES_WHY:-}" ]] || echo "host homes (e07_gate.py, decision-c02 amendments 3c and 3d): $HOMES_WHY" >> out/placement.txt
echo "plan: $PLAN, readers: $READERS" > out/plan.log

if [[ -n "$POD_VOLUME" ]]; then
  run model-copy bash -c "bash model-copy.sh $MODEL_SRC $MODEL_DIR && python3 $SRC/engine/gguf_layout.py sizes $MODEL_DIR" ||
    stop "stop: the model copy failed (out/model-copy.log)"
else
  # the e07b session's guard: the download must end by the deadline less the rest of the plan
  guard=""
  if [[ -n "${STEP_ESTIMATES:-}" ]]; then
    rest="$(awk 'found { s += $2 } $1 == "model-download" { found = 1 } END { print s + 0 }' "$STEP_ESTIMATES")"
    guard="HF_GUARD_END_BY=$(( DEADLINE - 30 - rest )) HF_GUARD_REST_S=$rest"
  fi
  run model-download bash -c "$guard bash hf-download.sh e07/hf-manifest.json $MODEL_DIR out/model-download.json &&
    python3 $SRC/engine/gguf_layout.py sizes $MODEL_DIR" ||
    stop "stop: $(grep -h -o 'download too slow: .*' out/model-download.log | tail -n 1 | grep . || echo "the model download failed (out/model-download.log)")"
fi
M="$MODEL_DIR/$SHARD1"
BENCH_ARGS="--model $M --ctx 262144 --chunk 16384 ${PLACE:-}"
[[ "$PLAN" == qfork ]] ||
  run dry-run ./dsv41-engine --serve $BENCH_ARGS --dry-run < /dev/null || stop "stop: the engine's dry run failed on the copy"
if [[ "$PLAN" == qdet ]]; then
  # qengine's server with every reply's ids in its engine's log, items 7 and 8
  # as its first requests, then the server stops; a failed item is its step's exit 1, the session goes on to the stop
  port="${E07_Q_PORT:-8090}"
  mkdir -p out/quality
  run engine-server bash "$Q/pod_server.sh" engine "$M" "$port" "$PLACE --log-ids" "${NUMA[@]}" ||
    { bash "$Q/pod_server.sh" stop >> out/quality-server.log 2>&1
      stop "stop: serve/server.py did not get ready (out/engine-server.log, out/quality-server.log)"; }
  run det-sampling python3 "$Q/det_sampling_check.py" --url "http://127.0.0.1:$port" --engine-log out/quality-engine.log \
    --out out/det-sampling.json
  run server-stop bash "$Q/pod_server.sh" stop
  volume_after
  echo done >> out/steps.log
  exit "$failed"
fi
if [[ "$PLAN" == cbase ]]; then
  # qengine's server, the FIFO one, then N concurrent chat users per load step (measurement
  # only); a failed request is its step's exit 1, the session goes on to the next N and the stop
  port="${E07_Q_PORT:-8090}"
  mkdir -p out/quality out/cbase
  run engine-server bash "$Q/pod_server.sh" engine "$M" "$port" "$PLACE" "${NUMA[@]}" ||
    { bash "$Q/pod_server.sh" stop >> out/quality-server.log 2>&1
      stop "stop: serve/server.py did not get ready (out/engine-server.log, out/quality-server.log)"; }
  for n in $(cat e07/cbase-users); do
    run "load-c$n" python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" --conversations e07/conversations.json \
      --users "$n" --out "out/cbase/load-c$n.json"
  done
  run server-stop bash "$Q/pod_server.sh" stop
  volume_after
  echo done >> out/steps.log
  exit "$failed"
fi
if (( has_cbatch || has_clong )); then
  port="${E07_Q_PORT:-8090}"
  mkdir -p out/quality
fi
if (( has_cbatch )); then
  # the FIFO server on today's slab for the single user's before figure, then the
  # server with the engine's batch slots: the single user again, N users decoding at once, items 7 and 8 once; a failed
  # request is its step's exit 1, the session goes on to the next step and the stop
  mkdir -p out/cbatch
  read -r short_turns < e07/cbatch-short
  if (( ! has_c05 )); then
    run engine-server-off bash "$Q/pod_server.sh" engine "$M" "$port" "$PLACE_OFF" "${NUMA[@]}" ||
      { bash "$Q/pod_server.sh" stop >> out/quality-server.log 2>&1
        stop "stop: serve/server.py did not get ready (out/engine-server-off.log, out/quality-server.log)"; }
    run load-short-off python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" --conversations e07/conversations-short.json \
      --users 1 --turns "$short_turns" --out out/cbatch/load-short-off.json
    run server-stop-off bash "$Q/pod_server.sh" stop
    # the off server's logs kept beside the batch's
    for f in quality-server.log quality-engine.log quality-server.cmd quality-server.json; do
      [[ -f "out/$f" ]] && mv "out/$f" "out/off-$f"
    done
  fi
  # cdet's batched server in the GPU's numerics as it ran (the A/B's gpu arm); every other plan's in the
  # engine's default, the fork's
  BATCHED="$PLACE --checkpoints $b_ckpt $BATCH_ARGS --log-ids"
  (( ! has_cdet )) || BATCHED+=" --cpu-numerics gpu"
  run engine-server bash "$Q/pod_server.sh" engine "$M" "$port" "$BATCHED" "${NUMA[@]}" ||
    { bash "$Q/pod_server.sh" stop >> out/quality-server.log 2>&1
      stop "stop: serve/server.py did not get ready (out/engine-server.log, out/quality-server.log)"; }
  run load-short python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" --conversations e07/conversations-short.json \
    --users 1 --turns "$short_turns" --out out/cbatch/load-short.json
  # the speed A/B's gpu arm, the same user with the mode off
  (( ! has_cdet )) || run ab-gpu python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" \
    --conversations e07/conversations-short.json --users 1 --turns "$short_turns" --deterministic off --out out/cbatch/ab-gpu.json
  # the combined plan leaves out load-c<N> and det-sampling (measured on 2026-10-14);
  # cdet leaves out load-c<N>
  (( ! cbatch_loads )) || for n in $(cat e07/cbatch-users); do
    run "load-c$n" python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" --conversations e07/conversations.json \
      --users "$n" --out "out/cbatch/load-c$n.json"
  done
  # item 11's verdict, the users decoding at once with exactly N rows a
  # step, their prompts read before the window
  read -r s_users s_window s_repeats s_tokens < e07/cbatch-steady
  run "steady-c$s_users" python3 "$Q/chat_load.py" steady --url "http://127.0.0.1:$port" --conversations e07/conversations.json \
    --users "$s_users" --window-s "$s_window" --repeats "$s_repeats" --max-tokens "$s_tokens" \
    --out "out/cbatch/steady-c$s_users.json"
  # the steady state with more users, reported beside item 11's
  for n in $(cat e07/cbatch-steady-report); do
    run "steady-c$n" python3 "$Q/chat_load.py" steady --url "http://127.0.0.1:$port" --conversations e07/conversations.json \
      --users "$n" --window-s "$s_window" --repeats "$s_repeats" --max-tokens "$s_tokens" --out "out/cbatch/steady-c$n.json"
  done
  # item 14 on the batched server; a failed phase is its step's exit 1
  (( ! has_cdet )) || run det-batch python3 "$Q/det_batch_check.py" --url "http://127.0.0.1:$port" \
    --engine-log out/quality-engine.log --out out/det-batch.json
  # (not cfinal: its item 14 is no quality loss from batching only
  (( has_clong )) || [[ "$PLAN" == cfinal ]] || run det-sampling python3 "$Q/det_sampling_check.py" --url "http://127.0.0.1:$port" \
    --engine-log out/quality-engine.log --out out/det-sampling.json
  if (( has_c05 )); then
    # item 3 batched, qengine's lm-eval through api_proxy.py on the batch slots, CDET_CONCURRENT requests at once
    export QUALITY_VENV="$POD_ROOT/root/venv-quality"
    read -r cdet_conc cdet_tasks < e07/cdet-quality
    url="http://127.0.0.1:$port/v1/chat/completions"
    qstop() { bash "$Q/pod_server.sh" stop >> out/quality-server.log 2>&1; stop "$1"; }
    run quality-venv bash -c 'until [ -f out/quality-venv.rc ]; do sleep 5; done; tail -n 20 out/quality-venv.log
      [ "$(cat out/quality-venv.rc)" = 0 ]' || qstop "stop: the lm-eval venv failed (out/quality-venv.log)"
    run continuation bash "$Q/eval.sh" strata "$url" out/quality continuation ||
      qstop "stop: the HumanEval continuation failed (out/continuation.log)"
    for t in $cdet_tasks; do
      run "lm-eval-$t" env NUM_CONCURRENT="$cdet_conc" bash "$Q/eval.sh" strata "$url" out/quality "$t"
    done
  fi
  run server-stop bash "$Q/pod_server.sh" stop
  if (( has_cdet )); then
    # the speed A/B's fork arm on the same server with the CPU experts in the fork's numerics; the batched
    # server's logs set aside meanwhile (the sheet reads them as out/quality-*), the fork server's kept as out/fork-*
    for f in quality-server.log quality-engine.log quality-server.cmd quality-server.json; do
      [[ -f "out/$f" ]] && mv "out/$f" "out/batch-$f"
    done
    if run engine-server-fork bash "$Q/pod_server.sh" engine "$M" "$port" "${BATCHED% gpu} fork" "${NUMA[@]}"; then
      run ab-fork python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" --conversations e07/conversations-short.json \
        --users 1 --turns "$short_turns" --deterministic off --out out/cbatch/ab-fork.json
    else
      echo "step ab-fork: skipped, the fork server did not get ready (out/engine-server-fork.log)" >> out/steps.log
      failed=1
    fi
    run server-stop-fork bash "$Q/pod_server.sh" stop
    for f in quality-server.log quality-engine.log quality-server.cmd quality-server.json; do
      [[ -f "out/$f" ]] && mv "out/$f" "out/fork-$f"
      [[ -f "out/batch-$f" ]] && mv "out/batch-$f" "out/$f"
    done
  fi
  # the combined plan: the batch server's logs kept beside clong's
  if (( has_clong )); then
    for f in quality-server.log quality-engine.log quality-server.cmd quality-server.json; do
      [[ -f "out/$f" ]] && mv "out/$f" "out/cbatch-$f"
    done
  fi
fi
if (( has_clong )); then
  # the server with slots that hold the 64K prompt and its reply, item 12's
  # long prompt beside decoding users, item 13's HTTP checks; a failed check is its step's exit 1, the session goes on
  sfx=""
  (( has_cbatch )) && sfx="-long"
  mkdir -p out/clong
  run "engine-server$sfx" bash "$Q/pod_server.sh" engine "$M" "$port" "$PLACE_LONG $LONG_ARGS --log-ids" "${NUMA[@]}" ||
    { bash "$Q/pod_server.sh" stop >> out/quality-server.log 2>&1
      stop "stop: serve/server.py did not get ready (out/engine-server$sfx.log, out/quality-server.log)"; }
  run load-long python3 "$Q/chat_load.py" long --url "http://127.0.0.1:$port" --conversations e07/conversations-long.json \
    --users "$l_users" --max-tokens "$l_max" --out out/clong/load-long.json
  run robust python3 "$Q/robust_check.py" --url "http://127.0.0.1:$port" --out out/clong/robust.json
  run "server-stop$sfx" bash "$Q/pod_server.sh" stop
fi
if [[ -n "$SPARE" ]]; then
  # the single user with batching on and the steady state again on the engine with more spare slots,
  # last so the items before it keep their time; its logs kept as out/spare-<file>
  for f in quality-server.log quality-engine.log quality-server.cmd quality-server.json; do
    [[ -f "out/$f" ]] && mv "out/$f" "out/long-$f"
  done
  run engine-server-spare bash "$Q/pod_server.sh" engine "$M" "$port" "$PLACE_SPARE --checkpoints $b_ckpt $BATCH_ARGS --log-ids" "${NUMA[@]}" ||
    { bash "$Q/pod_server.sh" stop >> out/quality-server.log 2>&1
      stop "stop: serve/server.py did not get ready (out/engine-server-spare.log, out/quality-server.log)"; }
  run load-short-spare python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" --conversations e07/conversations-short.json \
    --users 1 --turns "$short_turns" --out out/cbatch/load-short-spare.json
  run "steady-c$s_users-spare" python3 "$Q/chat_load.py" steady --url "http://127.0.0.1:$port" --conversations e07/conversations.json \
    --users "$s_users" --window-s "$s_window" --repeats "$s_repeats" --max-tokens "$s_tokens" \
    --out "out/cbatch/steady-c$s_users-spare.json"
  run server-stop-spare bash "$Q/pod_server.sh" stop
  for f in quality-server.log quality-engine.log quality-server.cmd quality-server.json; do
    [[ -f "out/$f" ]] && mv "out/$f" "out/spare-$f"
    [[ -f "out/long-$f" ]] && mv "out/long-$f" "out/$f"
  done
fi
if (( has_cbatch || has_clong )); then
  volume_after
  echo done >> out/steps.log
  exit "$failed"
fi
if [[ "$PLAN" == ccache ]]; then
  # qengine's server with every reply's ids in its engine's log (for
  # det-sampling), item 10's alternating users, the cbase runs again, items 7 and 8 once, then the server stops; a failed
  # request is its step's exit 1, the session goes on to the next step and the stop
  port="${E07_Q_PORT:-8090}"
  mkdir -p out/quality out/ccache
  run engine-server bash "$Q/pod_server.sh" engine "$M" "$port" "$PLACE --log-ids" "${NUMA[@]}" ||
    { bash "$Q/pod_server.sh" stop >> out/quality-server.log 2>&1
      stop "stop: serve/server.py did not get ready (out/engine-server.log, out/quality-server.log)"; }
  read -r alt_users alt_turns alt_think < e07/ccache-alt
  run "load-alt$alt_users" python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" --conversations e07/conversations.json \
    --users "$alt_users" --turns "$alt_turns" --alternate --engine-log out/quality-engine.log \
    --out "out/ccache/load-alt$alt_users.json"
  run "load-alt$alt_users-think$alt_think" python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" \
    --conversations e07/conversations.json --users "$alt_users" --turns "$alt_turns" --alternate --think-s "$alt_think" \
    --engine-log out/quality-engine.log --out "out/ccache/load-alt$alt_users-think$alt_think.json"
  for n in $(cat e07/ccache-users); do
    run "load-c$n" python3 "$Q/chat_load.py" run --url "http://127.0.0.1:$port" --conversations e07/conversations.json \
      --users "$n" --out "out/ccache/load-c$n.json"
  done
  run det-sampling python3 "$Q/det_sampling_check.py" --url "http://127.0.0.1:$port" --engine-log out/quality-engine.log \
    --out out/det-sampling.json
  run server-stop bash "$Q/pod_server.sh" stop
  volume_after
  echo done >> out/steps.log
  exit "$failed"
fi
if [[ "$PLAN" == q* ]]; then
  # Part Q: the server, the venv, the HumanEval continuation before the full run, the
  # tasks, the needles, then the server stops; lm-eval runs here against localhost through api_proxy.py
  export QUALITY_VENV="$POD_ROOT/root/venv-quality"
  # qdiag runs no needle (its payload has no e07/needle.json)
  [[ "$PLAN" == qdiag ]] || export NEEDLE=e07/needle.json
  mkdir -p out/quality
  qstop() { bash "$Q/pod_server.sh" stop >> out/quality-server.log 2>&1; stop "$1"; }
  # qdiag: the engine's own decode first, one load, no server (a failure still leaves the server path to measure)
  [[ "$PLAN" != qdiag ]] ||
    run bench-qdiag "${NUMA[@]}" python3 "$SRC/pod/e07_bench.py" --engine ./dsv41-engine --engine-args "$BENCH_ARGS" \
      --prompts e07/prompts --phases short-static,short-adaptive --out out/bench-qdiag.json --deadline "$DEADLINE"
  if [[ "$PLAN" == qfork ]]; then
    # lm-eval sends as many requests at once as the server has slots
    backend=llamacpp port="${E07_Q_PORT:-8080}" Q_FORK_NP="$(cat e07/fork-np)" NUM_CONCURRENT="$(cat e07/fork-np)"
    export Q_FORK_NP NUM_CONCURRENT
    run fork-server bash "$Q/pod_server.sh" fork "$M" "$port" "${NUMA[@]}" ||
      qstop "stop: llama-server did not get ready (out/fork-server.log, out/quality-server.log)"
  else
    backend=strata port="${E07_Q_PORT:-8090}"
    run engine-server bash "$Q/pod_server.sh" engine "$M" "$port" "$PLACE" "${NUMA[@]}" ||
      qstop "stop: serve/server.py did not get ready (out/engine-server.log, out/quality-server.log)"
  fi
  url="http://127.0.0.1:$port/v1/chat/completions"
  run quality-venv bash -c 'until [ -f out/quality-venv.rc ]; do sleep 5; done; tail -n 20 out/quality-venv.log
    [ "$(cat out/quality-venv.rc)" = 0 ]' || qstop "stop: the lm-eval venv failed (out/quality-venv.log)"
  cont="bash $Q/eval.sh $backend $url out/quality continuation"
  [[ "$PLAN" == qfork ]] &&
    cont="python3 $Q/continuation.py apply-template http://127.0.0.1:$port --out out/quality/continuation-prompts.json && $cont"
  run continuation bash -c "$cont" || qstop "stop: the HumanEval continuation failed (out/continuation.log)"
  # a machine whose engine decodes far below the measured speed ends the session here, for
  # cents (the slow qengine machine ran 44.2 ms per step on the continuation, qdiag's 20.2); qfork's server the same way
  # (t04a's stock baseline 52-53 ms per token, the 2026-10-11 session's slots 334-1002)
  machine_src="RunPod's machineId of the create (POD_MACHINE, pod-session.sh)"
  [[ -n "${POD_MACHINE:-}" ]] || machine_src="POD_MACHINE unset; pod ${RUNPOD_POD_ID:-unknown} (RUNPOD_POD_ID), host $(hostname)"
  if [[ "$PLAN" == qfork ]]; then
    speed=(out/quality-server.log --fork --max-ms "$(cat e07/fork-max-ms)")
  else
    speed=(out/quality-engine.log --max-ms "$(cat e07/speed-max-ms)")
  fi
  run speed-check python3 "$SRC/pod/speed_check.py" "${speed[@]}" --machine "${POD_MACHINE:-unknown}" \
    --machine-source "$machine_src" || [[ "$PLAN" == qdiag ]] ||
    qstop "$(grep -h -o -E '^stop: (engine|fork) too slow on this machine: .*' out/speed-check.log | tail -n 1 | grep . ||
      echo "stop: the speed check failed (out/speed-check.log)")"
  if [[ "$PLAN" == qdiag ]]; then
    # the same items one request at a time, then two (lm-eval's qengine setting): the server path's concurrency
    for c in 1 2; do
      run "lm-eval-gsm8k-c$c" env SAMPLES=e07/subsets-qdiag.json NUM_CONCURRENT=$c \
        bash "$Q/eval.sh" "$backend" "$url" "out/quality/c$c" gsm8k_cot_llama
    done
  else
    for t in $(cat e07/qtasks); do
      run "lm-eval-$t" bash "$Q/eval.sh" "$backend" "$url" out/quality "$t"
    done
    for n in $(cat e07/qneedles); do
      run "needle-$n" bash "$Q/eval.sh" "$backend" "$url" out/quality "needle-$n"
    done
  fi
  run server-stop bash "$Q/pod_server.sh" stop
  volume_after
  echo done >> out/steps.log
  exit "$failed"
fi
if [[ "$PLAN" == e08 ]]; then
  run bench-e08 "${NUMA[@]}" python3 "$SRC/pod/e07_bench.py" --engine ./dsv41-engine \
    --engine-args "$BENCH_ARGS --route-trace out/route-trace.txt" --prompts e07/prompts \
    --phases short-static,short-prompt,short-adaptive,16k-e08,short-timed,64k,256k --out out/bench-e08.json --deadline "$DEADLINE"
  run route-trace bash -c 'ls -l out/route-trace.txt && gzip -f out/route-trace.txt && ls -l out/route-trace.txt.gz'
  run serve "${NUMA[@]}" python3 "$SRC/engine/serve_check.py" --engine ./dsv41-engine --model "$M" --tokenizer-gguf "$M" \
    --vocab "$M" --reference cli --parts profile,http --one-engine --ctx 4096 --load-s 1500 --engine-args "$PLACE" \
    --chats e07/chats.json --out out/serve
  volume_after
  echo done >> out/steps.log
  exit "$failed"
fi
if [[ "$PLAN" == e07b ]]; then
  # the cheapest gating checks first, on the server: turn 1 the 16K prompt's text at
  # ctx 20480 (one prefill chunk of 16384), turn 2 a short follow-up, then the HTTP check on the same server load
  run two-turn "${NUMA[@]}" python3 "$SRC/engine/two_turn_check.py" --engine ./dsv41-engine --model "$M" \
    --tokenizer-gguf "$M" --ctx 20480 --load-s 1500 --engine-args "$PLACE --chunk 16384" --turns e07/turns.json \
    --max-tokens 128 --http --out out/two-turn
  run bench-e07b "${NUMA[@]}" python3 "$SRC/pod/e07_bench.py" --engine ./dsv41-engine --engine-args "$BENCH_ARGS" \
    --prompts e07/prompts --phases 16k-e09,turn2,short-adaptive,64k,256k --out out/bench-e07b.json --deadline "$DEADLINE"
  # free-greedy parity on this card: reported, not gating (the e07a session's A100: 7 of 64)
  python3 -c "import json; print(' '.join(map(str, json.load(open('e07/oracle/dump-manifest.json'))['prompt_token_ids'])))" \
    > e07/oracle-prompt.txt
  run dense-free "${NUMA[@]}" ./dsv41-engine --model "$M" --tokens-file e07/oracle-prompt.txt --n-predict 64 --ctx 4096 \
    --readers dense $PLACE --dump-layers out/dense-free
  run parity-dense-free python3 "$SRC/engine/oracle_parity.py" out/dense-free e07/oracle --json out/parity-dense-free.json
  run forced-ref python3 "$SRC/engine/oracle_parity.py" forced e07/oracle e07/oracle-forced
  run moe-isolated "${NUMA[@]}" ./dsv41-engine --model "$M" --tokens-file e07/oracle-prompt.txt --n-predict 64 --ctx 4096 \
    --readers dense $PLACE --force-selection e07/oracle-forced --forced-moe-input --dump-layers out/moe-isolated
  run parity-moe-isolated python3 "$SRC/engine/oracle_parity.py" out/moe-isolated e07/oracle --json out/parity-moe-isolated.json
  volume_after
  echo done >> out/steps.log
  exit "$failed"
fi
if [[ "$PLAN" == e09 ]]; then
  run bench-e09 "${NUMA[@]}" python3 "$SRC/pod/e07_bench.py" --engine ./dsv41-engine --engine-args "$BENCH_ARGS" \
    --prompts e07/prompts --phases 16k-e09,short-adaptive,16k-e09,256k --out out/bench-e09.json --deadline "$DEADLINE"
  run two-turn "${NUMA[@]}" python3 "$SRC/engine/two_turn_check.py" --engine ./dsv41-engine --model "$M" \
    --tokenizer-gguf "$M" --ctx 4096 --load-s 1500 --engine-args "$PLACE" --out out/two-turn
  volume_after
  echo done >> out/steps.log
  exit "$failed"
fi
run forced-ref python3 "$SRC/engine/oracle_parity.py" forced e07/oracle e07/oracle-forced
python3 -c "import json; print(' '.join(map(str, json.load(open('e07/oracle/dump-manifest.json'))['prompt_token_ids'])))" \
  > e07/oracle-prompt.txt

# correctness: the oracle's prompt and 64 greedy steps (1 + 64 calls, the oracle's 65 ubatches, 95 positions)
for mode in $READERS; do
  for sel in forced free; do
    name="$mode-$sel" force=""
    [[ "$sel" == forced ]] && force="--force-selection e07/oracle-forced"
    run "$name" "${NUMA[@]}" ./dsv41-engine --model "$M" --tokens-file e07/oracle-prompt.txt --n-predict 64 --ctx 4096 \
      --readers "$mode" $PLACE $force --dump-layers "out/$name"
    run "parity-$name" python3 "$SRC/engine/oracle_parity.py" "out/$name" e07/oracle --json "out/parity-$name.json"
  done
done
# where the forced runs' gap enters (the CPU reference of layers 0 and 1), then the MoE alone on the oracle's
# input; the GPU on MMQ only when the CPU's reading says the matmuls' activations carry the gap
for n in fork-cpu fork-cuda engine; do
  run "layer-ref-$n" ./dsv41-layer-ref --model "$M" --tokens-file e07/oracle-prompt.txt --numerics "$n" \
    --out "out/layer-ref/$n" --oracle e07/oracle --threads 32
done
run layer-parity python3 "$SRC/engine/layer_parity.py" compare --oracle e07/oracle --engine dense-forced=out/dense-forced \
  --json out/layer-parity.json out/layer-ref/fork-cpu out/layer-ref/fork-cuda out/layer-ref/engine
run moe-isolated "${NUMA[@]}" ./dsv41-engine --model "$M" --tokens-file e07/oracle-prompt.txt --n-predict 64 --ctx 4096 \
  --readers dense $PLACE --force-selection e07/oracle-forced --forced-moe-input --dump-layers out/moe-isolated
run parity-moe-isolated python3 "$SRC/engine/oracle_parity.py" out/moe-isolated e07/oracle --json out/parity-moe-isolated.json
mmq="$(python3 -c "import json; print(json.load(open('out/layer-parity.json')).get('mmq_worth_measuring'))" 2>/dev/null || echo none)"
echo "layer-parity's reading mmq_worth_measuring: $mmq; dense-forced-mmq $([[ "$mmq" == True ]] && echo runs || echo "does not run")" >> out/plan.log
if [[ "$mmq" == True ]]; then
  run dense-forced-mmq "${NUMA[@]}" ./dsv41-engine --model "$M" --tokens-file e07/oracle-prompt.txt --n-predict 64 \
    --ctx 4096 --readers dense $PLACE --force-selection e07/oracle-forced --prefill-mmq --dump-layers out/dense-forced-mmq
  run parity-dense-forced-mmq python3 "$SRC/engine/oracle_parity.py" out/dense-forced-mmq e07/oracle \
    --json out/parity-dense-forced-mmq.json
  run layer-parity-mmq python3 "$SRC/engine/layer_parity.py" compare --oracle e07/oracle \
    --engine dense-forced-mmq=out/dense-forced-mmq --json out/layer-parity-mmq.json out/layer-ref/fork-cpu \
    out/layer-ref/fork-cuda out/layer-ref/engine
fi
[[ "$PLAN" == cycle2 ]] || run tokenizer python3 "$SRC/engine/tokenizer_parity.py" --llama-tokenize "fork/llama-tokenize" \
  --corpus "$SRC/engine/tokenizer_corpus.txt" "$M"
run bench-a "${NUMA[@]}" python3 "$SRC/pod/e07_bench.py" --engine ./dsv41-engine --engine-args "$BENCH_ARGS" \
  --prompts e07/prompts --phases 16k,turn2,short,short-timed --out out/bench-a.json --deadline "$DEADLINE"
if [[ "$PLAN" == full ]]; then
  run serve "${NUMA[@]}" python3 "$SRC/engine/serve_check.py" --engine ./dsv41-engine --model "$M" --tokenizer-gguf "$M" \
    --vocab "$M" --reference cli --parts profile,http --one-engine --ctx 4096 --load-s 1500 --engine-args "$PLACE" \
    --chats e07/chats.json --out out/serve
  run bench-b "${NUMA[@]}" python3 "$SRC/pod/e07_bench.py" --engine ./dsv41-engine --engine-args "$BENCH_ARGS" \
    --prompts e07/prompts --phases 64k,256k --out out/bench-b.json --deadline "$DEADLINE"
fi
volume_after
echo done >> out/steps.log
exit "$failed"
