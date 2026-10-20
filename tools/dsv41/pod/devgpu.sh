#!/usr/bin/env bash
# The engine's dev pod session (the lightning indexer, sparse attention and KV, the quantized matmuls, the
# hyper-connections, engram, router and dense layers, the expert tiers, prefill overlap, expert and decode speed, the
# prefix cache and server): creates one cheap
# GPU pod on RunPod (RTX 4090 sm_89 preferred; any GPU the cu128 sm_80/sm_89 build runs on), checks that its CUDA
# works (dsv41_cuda_probe: a broken host is deleted, its machine remembered, and the next offer tried), uploads the
# engine tests, benches and dsv41-engine from the cu128 strata tarball with the cuBLAS libraries they link (bundled in
# that tarball) and the engine's test data (the tiny GGUFs, their prompts, the fork's dumps of them, the hot sets of the three
# expert placements; the hybrid's decode counts profile), runs pod/devgpu-run.sh there (the indexer, qmatmul,
# attention, dense and experts tests --gpu, the grouped expert GEMM's test, their benches, the engine end to end on
# tiny-a and tiny-b against the fork, in one chunk and in chunks of 24, the real-dims smoke and the measurement smoke of
# 8 layers x 384 experts, with the synthetic router's routing and with the profile's; dsv41-engine --serve behind
# serve/server.py on tiny-b with tools/dsv41/engine/serve_check.py, and the smoke's prefix cache at real dims) detached
# from the SSH session (setsid, nohup, its output in out/; out/monitor.log: free -m, the running step's RSS and dmesg's
# tail every 30 s) and polls it over fresh SSH connections, retried, within the run's deadline, pulls the results
# (two archives made on the pod, the logs first, each copy under `timeout`, the dumps only below PULL_CAP_MB; tried on
# every way out once the run started, a failed poll too) and deletes the pod right after the pull. It exits non-zero
# when a step on the pod failed, was skipped for lack of time, or did not report (out/steps.log). Run from the local machine (the repo's root), with your RunPod API key:
#
#   RUNPOD_API_KEY=... bash tools/dsv41/pod/devgpu.sh
#   bash tools/dsv41/pod/devgpu.sh --dry-run      # anywhere: the plan and every command, no request, no key
#
# DEVGPU_PLAN=c03 runs the batched decode's CUDA checks only (pod/devgpu-run.sh's
# c03 steps: dsv41_engine_batch_test --kernels and --gpu, then the expert tiers, qmatmul and two forced e2e runs whose
# paths the batch's code touched), on the default cheap GPUs, MAX_USD_PER_HR 0.70, MAX_SESSION_USD 1.50, its dry run
# printing each step's estimate and the session's cost at the limit price (refused over 1.50 USD):
#
#   BUILD_TAG=-c03 DEVGPU_PLAN=c03 bash tools/dsv41/pod/devgpu.sh --dry-run
#   RUNPOD_API_KEY=... BUILD_TAG=-c03 DEVGPU_PLAN=c03 bash tools/dsv41/pod/devgpu.sh
#
# DEVGPU_PLAN=c05 runs the CUDA expert kernels against the CPU experts in the
# GPU's numerics bit for bit on synthetic tensors (dsv41_engine_expert_bits_test --gpu: every matvec, the FFN, the
# weighted sum), the CPU side's own checks and speed table on the pod's CPU, the expert tiers' decode bit for bit in
# every placement (experts_test --gpu), the engine's batch against each request alone with the hot set swapping at
# every token (batch_test --gpu-swaps) and without (--gpu), the batched kernels again (--kernels), and the tests
# whose numerics the deterministic CPU experts changed (qmatmul_test and dense_test --gpu: the pinned MMVQ traits, the shared expert's clamp),
# then two forced end-to-end runs against the fork; on the default cheap GPUs, MAX_USD_PER_HR 0.70, MAX_SESSION_USD
# 1.50, the machine qualified first (QUALIFY=1, pod-session.sh: the cu128 dsv41_machine_probe of
# pod/build-machine-probe.sh in MACHINE_PROBE_ROOT/cu128/machine-probe, with the qualification's bounds for any GPU model:
# GPU read measured and logged, H2D and D2H >= 18 GB/s, the doorbell's median <= 50 us, SM clock and power limit >=
# 0.70 of the GPU's own, and its host read floor of 40 GB/s, raised to 250 only
# for the RTX PRO 6000's real-model sessions; QUALIFY_MIN_* override them, QUALIFY=0 is the CUDA probe alone); its dry
# run prints each step's estimate and the session's cost at the limit price (refused over 1.50 USD):
#
#   BUILD_TAG=-c04 DEVGPU_PLAN=c05 bash tools/dsv41/pod/devgpu.sh --dry-run
#   RUNPOD_API_KEY=... BUILD_TAG=-c04 DEVGPU_PLAN=c05 bash tools/dsv41/pod/devgpu.sh
#
# DEVGPU_PLAN=c05b runs the promotion's parity on the engine itself,
# tiny-a and tiny-b (dsv41_engine_batch_test --gpu and --gpu-swaps, whose check 12 stops a request on the solo path
# after 1, 2 and 5 tokens, sends it as a BGEN of its prompt and tokens into slot 0 with 3 requests admitted beside it
# and compares every token and every call's logits bytes with its alone run, and whose check 13 reports how many
# logits a token read by prefill(T=1) changes against decode on the same state), then the two forced end-to-end runs
# against the fork (the alone path); c05's qualification and limits (MAX_USD_PER_HR 0.70, MAX_SESSION_USD 1.50):
#
#   BUILD_TAG=-c04 DEVGPU_PLAN=c05b bash tools/dsv41/pod/devgpu.sh --dry-run
#
# Money (the dev pod limits): before any create, the GPU's list price in that cloud (runpod_api.py price, REST
# v2's catalog) must be at or under MAX_USD_PER_HR (default 0.70), else that GPU and cloud are skipped unasked (once
# a 0.74 USD/h 4090 was created and deleted within a second); a pod whose costPerHr (the list price plus its
# disk) is still above the limit is deleted at once and the next choice tried; each pod's hard cap is
# min(CAP_MIN, what is left of MAX_SESSION_USD / costPerHr) minutes from its create (defaults 40 min, 1.50 USD for the
# session, every pod deleted on the way counted at its full lifetime). At the cap a watchdog deletes the pod whatever
# the script is doing, and the EXIT trap deletes it on every other way out; the pod's last state is checked after the
# delete. Broken hosts (once a COMMUNITY RTX 4090 answered nvidia-smi while every CUDA call failed with "unknown
# error"): the probe runs right after the first SSH login, before the payload goes up; on failure, or when no SSH login
# comes within SSH_WAIT_S, the pod is deleted, its machineId remembered (a later create on it is deleted at once) and the next offer tried, at most MAX_PODS pods.
#
# Environment: GPU_TYPES (';'-separated RunPod gpuTypeIds, in order of preference), CLOUD_TYPES (default "SECURE
# COMMUNITY"), DATA_CENTERS (default "EUR-IS-1 any"), IMAGE (a RunPod image with sshd and python3), DISK_GB (default
# 20), MAX_USD_PER_HR, MAX_SESSION_USD, CAP_MIN, MAX_PODS (default 3), BUILD_ROOT (default ~/dsv41-build), E01 (the
# tiny models and dumps of tools/dsv41/engine/run-oracles.sh, default $BUILD_ROOT/engine/e01), OUT (default
# $BUILD_ROOT/engine/$SESSION-pod/<UTC time>, SESSION default e06), COUNTS (the strata-moe-counts/1 profile of the
# expert speed smoke, default $BUILD_ROOT/counts.json: the routing counts of a real-model run, the t04a payload's), SSH_KEY (default ~/.ssh/runpod), PULL_S (the
# results pull's limit, default 300 s), PULL_CAP_MB (the dumps archive's cap, default 512), MIN_RAM_GB (default 0:
# none; above 0 the create asks RunPod for at least that much host RAM, minRAMPerGPU, and a pod whose memoryInGb is
# below it, or not reported, is deleted at once and the next offer tried), POLL_FAILS (failed polls of the detached
# run in a row before the session gives up on the host and pulls what it can, default 10).
#
# The prefill overlap measurement session (smoke-e05b pins 37 GiB of experts, so a 48 GB GPU
# with at least 64 GB of host RAM; the 24 GB default list stays for the other sessions):
#
#   RUNPOD_API_KEY=... GPU_TYPES='NVIDIA RTX A6000;NVIDIA A40;NVIDIA L40' MIN_RAM_GB=64 bash tools/dsv41/pod/devgpu.sh
#
# The expert speed session runs the same smoke twice (synthetic and --counts routing), the second pinning the same
# 37 GiB; with no 48 GB GPU in stock, an A100 80GB under the measurement-pod rule:
#
#   RUNPOD_API_KEY=... GPU_TYPES='NVIDIA A100 80GB PCIe;NVIDIA A100-SXM4-80GB' MIN_RAM_GB=64 MAX_USD_PER_HR=1.50 \
#     bash tools/dsv41/pod/devgpu.sh
#
# The decode speed session measures decode on the same A100 (the same overrides): both smokes' decode phases, now from the token
# graph, and the e2e runs decoding through it (plus --no-graphs forced runs of the split placement).
#
# The prefix cache session adds, on the same A100: the server on tiny-b (the payload's serve/, tools/strata_tokenizer.py and
# serve_check.py, a 512-token cut of $VOCAB, the vocab-only GGUF of the real tokenizer, by engine/tiny_vocab.py; the
# pod's python3 with jinja2), and dsv41_engine_smoke --prefix-cache 512 (turn 1 = 16,384 + 32 tokens, turn 2 + 512).
# An early A100 pod dropped off SSH 13 min into a blocking `ssh ... devgpu-run.sh` and nothing
# came back: the run is now detached on the pod and each poll is its own connection, so a dropped one loses nothing.
# The session itself (offers, watchdog, SSH, probe, detached run, polls, pull, delete) is pod/pod-session.sh, which
# pod/e07.sh (the real model) shares.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
E01="${E01:-$BUILD_ROOT/engine/e01}"
GPU_TYPES="${GPU_TYPES:-NVIDIA GeForce RTX 4090;NVIDIA L4;NVIDIA RTX A5000;NVIDIA GeForce RTX 3090}"
CLOUD_TYPES="${CLOUD_TYPES:-SECURE COMMUNITY}"
DATA_CENTERS="${DATA_CENTERS:-EUR-IS-1 any}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-20}"
MAX_USD_PER_HR="${MAX_USD_PER_HR:-0.70}"
MAX_SESSION_USD="${MAX_SESSION_USD:-1.50}"
CAP_MIN="${CAP_MIN:-40}"
MAX_PODS="${MAX_PODS:-3}"
MIN_RAM_GB="${MIN_RAM_GB:-0}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/runpod}"
# the batched decode's checks only (c03); the expert bits' (c05); the promotion's parity
# (c05b); or every check (all, the default)
DEVGPU_PLAN="${DEVGPU_PLAN:-all}"
case "$DEVGPU_PLAN" in
  all) SESSION_DEF=e06 ;;
  c03|c05|c05b) SESSION_DEF="$DEVGPU_PLAN" ;;
  *) echo "devgpu.sh: DEVGPU_PLAN='$DEVGPU_PLAN': all, c03, c05 or c05b" >&2; exit 2 ;;
esac
# the machine qualification first (pod-session.sh's QUALIFY=1, the header), the qualification's bounds for any GPU
# model and its host read floor (c05b the same)
if [[ "$DEVGPU_PLAN" == c05 || "$DEVGPU_PLAN" == c05b ]]; then
  QUALIFY="${QUALIFY:-1}"
  QUALIFY_MIN_H2D_GBS="${QUALIFY_MIN_H2D_GBS:-18}" QUALIFY_MIN_D2H_GBS="${QUALIFY_MIN_D2H_GBS:-18}"
  QUALIFY_MAX_DOORBELL_US="${QUALIFY_MAX_DOORBELL_US:-50}" QUALIFY_MIN_HOST_READ_GBS="${QUALIFY_MIN_HOST_READ_GBS:-40}"
  QUALIFY_MIN_SM_RATIO="${QUALIFY_MIN_SM_RATIO:-0.70}" QUALIFY_MIN_POWER_RATIO="${QUALIFY_MIN_POWER_RATIO:-0.70}"
  QUALIFY_ARGS="${QUALIFY_MIN_GPU_READ_GBS:+--min-gpu-read-gbs $QUALIFY_MIN_GPU_READ_GBS }--min-h2d-gbs $QUALIFY_MIN_H2D_GBS"
  QUALIFY_ARGS+=" --min-d2h-gbs $QUALIFY_MIN_D2H_GBS --max-doorbell-us $QUALIFY_MAX_DOORBELL_US"
  QUALIFY_ARGS+=" --min-sm-ratio $QUALIFY_MIN_SM_RATIO --min-power-ratio $QUALIFY_MIN_POWER_RATIO"
  QUALIFY_S=120   # e07.sh's: the probe takes ~10 s, a hung driver gets this long
  MACHINE_PROBE_ROOT="${MACHINE_PROBE_ROOT:-$BUILD_ROOT}"
  # 40 GB/s, not the RTX sessions' 250, since this pod
  # reads synthetic tensors, not the 246 GiB model
  qualify_host_read() {
    echo "$QUALIFY_MIN_HOST_READ_GBS|the dev pod floor, QUALIFY_MIN_HOST_READ_GBS; not the RTX sessions' 250 GB/s: synthetic tensors, not the model"
  }
fi
SESSION="${SESSION:-$SESSION_DEF}"
COUNTS="${COUNTS:-$BUILD_ROOT/counts.json}"
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"   # the real tokenizer (tools/dsv41/README.md)
STRATA_DIR="$(cd "$here/../../.." && pwd)"
OUT="${OUT:-$BUILD_ROOT/engine/$SESSION-pod/$(date -u +%Y%m%dT%H%M%SZ)}"
SSH_WAIT_S=600      # create to a working SSH login; RunPod pods took 1 to 5 minutes in the T04 sessions
PROBE_S=60          # the CUDA probe takes about a second; a hung driver gets this long
PULL_S="${PULL_S:-300}"            # the whole results pull, archives made on the pod to both copies done
PULL_TAR_S=120                     # the pod's two tar -czf
PULL_CAP_MB="${PULL_CAP_MB:-512}"  # out-dumps.tar.gz above this stays on the pod
START_S=300         # the payload's extract on the pod (746 MB in E06) and the detached start
POLL_S=30           # between two polls of the detached run, each a fresh SSH connection
POLL_SSH_S=45       # one poll, connect (ConnectTimeout 15) to answer
POLL_FAILS="${POLL_FAILS:-10}"     # failed polls in a row (7 to 13 min of an unreachable host) before giving up on it
REMOTE=/root/devgpu
RUN_SCRIPT=devgpu-run.sh   # the payload's script on the pod (pod-session.sh)
NETWORK_VOLUME=""          # no volume: the payload is a few hundred MB
SESSION_LOG="$OUT/devgpu.log"
REST_SHOWN=https://rest.runpod.io/v1   # runpod_api.py's bases, for the dry run's plan
CATALOG_SHOWN=https://api.runpod.io/v2/catalog/gpus

dry=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry=1 ;;
    -h|--help) sed -n '2,103p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "devgpu.sh: unknown argument '$arg' (only --dry-run)" >&2; exit 2 ;;
  esac
done

# shellcheck source=tools/dsv41/pod/pod-session.sh
source "$here/pod-session.sh"
[[ "$MIN_RAM_GB" =~ ^[0-9]+$ ]] || { echo "devgpu.sh: MIN_RAM_GB='$MIN_RAM_GB' is not a whole number of GB" >&2; exit 2; }
[[ "$POLL_FAILS" =~ ^[1-9][0-9]*$ ]] || { echo "devgpu.sh: POLL_FAILS='$POLL_FAILS' is not a whole number >= 1" >&2; exit 2; }

# the cu128 strata tarball (sm_80, sm_89), the engine binaries in it and the cuBLAS libraries they load
shopt -s nullglob
tarballs=("$BUILD_ROOT"/dist"${BUILD_TAG:-}"/strata-*-cu128-sm80-89.tar.gz)
shopt -u nullglob
(( ${#tarballs[@]} == 1 )) || { echo "devgpu.sh: expected one $BUILD_ROOT/dist${BUILD_TAG:-}/strata-*-cu128-sm80-89.tar.gz, found ${#tarballs[@]} (tools/dsv41/build.sh)" >&2; exit 1; }
tarball="${tarballs[0]}"
top="$(basename "$tarball" .tar.gz)"
bins=(dsv41_engine_indexer_test dsv41_engine_indexer_bench dsv41_engine_qmatmul_test dsv41_engine_qmatmul_bench
      dsv41_engine_attention_test dsv41_engine_attention_bench dsv41_engine_dense_test dsv41_engine_dense_bench
      dsv41_engine_experts_test dsv41_engine_smoke dsv41-engine dsv41_cuda_probe libcublas.so.12 libcublasLt.so.12)
# the batched decode's test; and the expert bits' test
[[ "$DEVGPU_PLAN" != c03 ]] || bins+=(dsv41_engine_batch_test)
[[ "$DEVGPU_PLAN" != c05 ]] || bins+=(dsv41_engine_batch_test dsv41_engine_expert_bits_test)
[[ "$DEVGPU_PLAN" != c05b ]] || bins+=(dsv41_engine_batch_test)
listing="$(tar -tzf "$tarball")"
for b in "${bins[@]}" MANIFEST; do
  grep -qx "$top/\(bin/\)\?$b" <<<"$listing" || { echo "devgpu.sh: $tarball has no $b" >&2; exit 1; }
done
# the tiny models and the fork's dumps the engine is compared with (run-oracles.sh)
dumps=(dump-a-16 dump-b-16 dump-a-short dump-b-short)
for f in tiny-a.gguf tiny-b.gguf "${dumps[@]/%//dump-manifest.json}"; do
  [[ -f "$E01/$f" ]] || { echo "devgpu.sh: no $E01/$f (tools/dsv41/engine/run-oracles.sh $E01)" >&2; exit 1; }
done
# the real model's routing for the measurement smoke (dsv41_engine_smoke --counts)
[[ -f "$COUNTS" ]] || { echo "devgpu.sh: no COUNTS=$COUNTS (a strata-moe-counts/1 profile, the hybrid's T04d c7 run)" >&2; exit 1; }
[[ -f "$VOCAB" ]] || { echo "devgpu.sh: no VOCAB=$VOCAB (the vocab-only DeepSeek-V4.1 GGUF, tools/dsv41/README.md T06)" >&2; exit 1; }
# the server's sources on the pod, at their repo paths under src/ (serve_check.py finds the repo from its own)
server_src=(serve tools/strata_tokenizer.py tools/gguf_reader.py tools/llama_unicode_flags.txt tools/dsv41/engine/serve_check.py)

IFS=';' read -ra gpus <<<"$GPU_TYPES"
read -ra clouds <<<"$CLOUD_TYPES"
read -ra dcs <<<"$DATA_CENTERS"

if (( dry )) && [[ "$DEVGPU_PLAN" == c03 ]]; then
  # the steps, their estimates and the session's cost at the limit price; the
  # kernels' and the e2e's minutes are guesses (nothing of them ran on a GPU yet), the tests' from their E05 sessions
  # (A100: experts-test ~2 min, qmatmul-test ~1.5 min, a forced e2e on a tiny model under 0.5 min)
  echo "devgpu.sh --dry-run (DEVGPU_PLAN=c03): nothing is created, uploaded or run"
  echo "payload: ${bins[*]} and MANIFEST from $tarball (sha256 checked against its MANIFEST), pod/devgpu-run.sh,"
  echo "  pod/pod-steps.sh, engine/engine_parity.py, e05a/ (tiny-a.gguf and tiny-b.gguf from $E01, their hot sets, the"
  echo "  short prompts and the fork's dumps), plan (c03); results: $OUT"
  echo "GPUs, in order: $GPU_TYPES; price limit $MAX_USD_PER_HR USD/h; session cap $MAX_SESSION_USD USD over at most $MAX_PODS pods,"
  echo "  each min($CAP_MIN min, what is left / price), at the limit price $(cap_minutes "$MAX_USD_PER_HR") min"
  c03_steps=("create, SSH, CUDA probe (the E0x sessions: 1-5 min):5" "payload upload and start:1" "env:0.1"
             "batch-kernels (dsv41_engine_batch_test --kernels: (g1) 14 real shapes, (g2) 7 expert calls, (g3) 3 dims):4"
             "batch-e2e (dsv41_engine_batch_test --gpu, tiny-a and tiny-b, the split placement, no swaps):3"
             "experts-test (dsv41_engine_experts_test --gpu, three placements, calls of 1, 4 and 64 tokens):2"
             "qmatmul-test (dsv41_engine_qmatmul_test --gpu):1.5"
             "e2e-a-reuse-split-forced:0.5" "e2e-b-reuse-split-forced:0.5" "pull and delete:1.5")
  t=0
  echo "plan (minutes, estimates; each step's limit the time left before the pod's deadline):"
  for st in "${c03_steps[@]}"; do
    t="$(python3 -c "print(round($t + ${st##*:}, 1))")"
    printf '  %-100s %4s  (total %s)
' "${st%:*}" "${st##*:}" "$t"
  done
  usd="$(python3 -c "print(f'{$t / 60 * $MAX_USD_PER_HR:.2f}')")"
  echo "plan: total $t min, about $usd USD at $MAX_USD_PER_HR USD/h (the session cap 1.50 USD; the pod's hard cap $(cap_minutes "$MAX_USD_PER_HR") min)"
  python3 -c "import sys; sys.exit(0 if $usd <= 1.50 and $t <= $(cap_minutes "$MAX_USD_PER_HR") else 2)" ||
    { echo "devgpu.sh: the C03 plan is over 1.50 USD or the pod's cap"; exit 2; }
  for g in "${gpus[@]}"; do for c in "${clouds[@]}"; do
    echo "  GET $CATALOG_SHOWN?include=AVAILABILITY&product=POD&cloud=$c&count=1&cudaVersions=12.8,12.9,13.0 -> price of '$g'"
    for dc in "${dcs[@]}"; do
      [[ "$dc" == any ]] && dc=""
      echo "    POST $REST_SHOWN/pods $(api body --gpu "$g" --cloud "$c" --dc "$dc" --image "$IMAGE" --disk "$DISK_GB" --min-ram "$MIN_RAM_GB")"
    done
  done; done
  echo "then pod-session.sh as for every dev pod: the watchdog at the cap, the CUDA probe, the detached run, the polls,"
  echo "  the pull, the delete (the RunPod console must list no pod after it)"
  exit 0
fi
if (( dry )) && [[ "$DEVGPU_PLAN" == c05 || "$DEVGPU_PLAN" == c05b ]]; then
  # the steps' minutes from C03's dev pod (2026-10-14, RTX A5000 SECURE at
  # 0.27 USD/h, 214 s from create to delete: experts-test 54 s, qmatmul-test 20 s, batch-kernels and batch-e2e 2 s
  # each, a forced e2e 1 s) with a margin, the expert bits' from the build machine (its CPU part 5 s on the Ryzen 7 5700X;
  # the GPU part's CPU side is 4 FFN calls of up to 48 pairs, its kernels synthetic), dense-test from E04b's dev pods
  echo "devgpu.sh --dry-run (DEVGPU_PLAN=$DEVGPU_PLAN): nothing is created, uploaded or run"
  echo "payload: ${bins[*]} and MANIFEST from $tarball (sha256 checked against its MANIFEST), pod/devgpu-run.sh,"
  echo "  pod/pod-steps.sh, engine/engine_parity.py, e05a/ (tiny-a.gguf and tiny-b.gguf from $E01, their hot sets, the"
  echo "  short prompts and the fork's dumps), plan ($DEVGPU_PLAN); results: $OUT"
  if (( ${QUALIFY:-0} )); then
    mp="$(machine_probe_ready cu128)" || exit 1
    echo "qualification first (QUALIFY=1, right after the CUDA probe, before the payload): $mp"
    echo "  bounds: $QUALIFY_ARGS --min-host-read-gbs $QUALIFY_MIN_HOST_READ_GBS (GPU read measured and logged for these models)"
    hr="$(qualify_host_read)"
    echo "  host read floor ${hr%%|*} GB/s (${hr#*|})"
  else
    echo "qualification: off (QUALIFY=0: the CUDA probe alone)"
  fi
  echo "GPUs, in order: $GPU_TYPES; price limit $MAX_USD_PER_HR USD/h; session cap $MAX_SESSION_USD USD over at most $MAX_PODS pods,"
  echo "  each min($CAP_MIN min, what is left / price), at the limit price $(cap_minutes "$MAX_USD_PER_HR") min"
  c05_steps=("create, SSH, CUDA probe, machine qualification (the E0x sessions: 1-5 min, the probe ~10 s):5"
             "payload upload and start:1" "env:0.1"
             "expert-bits-gpu (dsv41_engine_expert_bits_test --gpu: (g1) gate/up [2304][5120], down [5120][2304], [64][256] x 1, 2, 4, 8 activations, (g2) 4 FFN calls at those shapes, (g3) their sums, (g0) round 1's upload order, info):1"
             "expert-bits-cpu (dsv41_engine_expert_bits_test: the CPU against the reference, the speed table on this CPU):1"
             "experts-test (dsv41_engine_experts_test --gpu: decode bit for bit in all five placements):2"
             "batch-kernels (dsv41_engine_batch_test --kernels):0.5"
             "batch-e2e (dsv41_engine_batch_test --gpu, tiny-a and tiny-b, no swaps):0.5"
             "batch-swaps (dsv41_engine_batch_test --gpu-swaps, the hot set swapping at every token):1"
             "qmatmul-test (dsv41_engine_qmatmul_test --gpu):1" "dense-test (dsv41_engine_dense_test --gpu):1.5"
             "e2e-a-reuse-split-forced:0.5" "e2e-b-reuse-split-forced:0.5" "pull and delete:1.5")
  # the promotion's parity, checks 12 and 13 of batch_test --gpu and
  # --gpu-swaps (each a few seconds more than C03's 2 s batch-e2e: 3 promotions and 3 diagnostics per model)
  [[ "$DEVGPU_PLAN" != c05b ]] ||
    c05_steps=("create, SSH, CUDA probe, machine qualification (the E0x sessions: 1-5 min, the probe ~10 s):5"
               "payload upload and start:1" "env:0.1"
               "batch-e2e (dsv41_engine_batch_test --gpu, tiny-a and tiny-b, no swaps: checks 1 to 4, check 12 the promotion after 1, 2 and 5 tokens into slot 0 with 3 beside, every call's logits bytes against alone, check 13 prefill(T=1) against decode, info):1"
               "batch-swaps (dsv41_engine_batch_test --gpu-swaps: the same with the hot set swapping at every token):1"
               "e2e-a-reuse-split-forced:0.5" "e2e-b-reuse-split-forced:0.5" "pull and delete:1.5")
  t=0
  echo "plan (minutes, estimates; each step's limit the time left before the pod's deadline):"
  for st in "${c05_steps[@]}"; do
    t="$(python3 -c "print(round($t + ${st##*:}, 1))")"
    printf '  %-118s %4s  (total %s)\n' "${st%:*}" "${st##*:}" "$t"
  done
  usd="$(python3 -c "print(f'{$t / 60 * $MAX_USD_PER_HR:.2f}')")"
  echo "plan: total $t min, about $usd USD at $MAX_USD_PER_HR USD/h (the session cap 1.50 USD; the pod's hard cap $(cap_minutes "$MAX_USD_PER_HR") min)"
  python3 -c "import sys; sys.exit(0 if $usd <= 1.50 and $t <= $(cap_minutes "$MAX_USD_PER_HR") else 2)" ||
    { echo "devgpu.sh: the $DEVGPU_PLAN plan is over 1.50 USD or the pod's cap"; exit 2; }
  for g in "${gpus[@]}"; do for c in "${clouds[@]}"; do
    echo "  GET $CATALOG_SHOWN?include=AVAILABILITY&product=POD&cloud=$c&count=1&cudaVersions=12.8,12.9,13.0 -> price of '$g'"
    for dc in "${dcs[@]}"; do
      [[ "$dc" == any ]] && dc=""
      echo "    POST $REST_SHOWN/pods $(api body --gpu "$g" --cloud "$c" --dc "$dc" --image "$IMAGE" --disk "$DISK_GB" --min-ram "$MIN_RAM_GB")"
    done
  done; done
  echo "then pod-session.sh as for every dev pod: the watchdog at the cap, the CUDA probe, the qualification, the"
  echo "  detached run, the polls, the pull, the delete (the RunPod console must list no pod after it)"
  exit 0
fi
if (( dry )); then
  echo "devgpu.sh --dry-run: nothing is created, uploaded or run"
  echo "payload: ${bins[*]} and MANIFEST from $tarball (sha256 checked against its MANIFEST), pod/devgpu-run.sh and"
  echo "  pod/pod-steps.sh,"
  echo "  engine/engine_parity.py, and e05a/: tiny-a.gguf and tiny-b.gguf from $E01, the hot sets of engine/hot_sets.py"
  echo "  (all-vram, split), the prompts and the fork's l_last, result_output, router and indexer selections and scores"
  echo "  (ffn_moe_topk, ffn_moe_probs_biased, idx_top_k, idx_score) and manifest of ${dumps[*]}; e05c/counts.json from"
  echo "  $COUNTS; E06: e06/tiny-vocab.gguf (engine/tiny_vocab.py: the first 512 tokens of $VOCAB) and src/: ${server_src[*]}"
  echo "  (no test_*.py, no __pycache__)"
  echo "results: $OUT"
  echo "GPUs, in order: $GPU_TYPES; host RAM: $( (( MIN_RAM_GB > 0 )) && echo "at least $MIN_RAM_GB GB (minRAMPerGPU; a pod reporting less, or no memoryInGb, deleted at once)" || echo "no minimum")"
  echo "price limit: $MAX_USD_PER_HR USD/h; session cap: $MAX_SESSION_USD USD over at most $MAX_PODS pods, each min($CAP_MIN min, what is left / price), at the limit price $(cap_minutes "$MAX_USD_PER_HR") min"
  echo "create attempts, in order: per GPU and cloud first the list price (skipped unasked when not offered, none"
  echo "available, or above $MAX_USD_PER_HR USD/h), then the creates (the first pod at or under the limit and on no broken"
  echo "machine is kept, a dearer one, one below the RAM minimum or one on a broken machine deleted at once):"
  for g in "${gpus[@]}"; do for c in "${clouds[@]}"; do
    echo "  GET $CATALOG_SHOWN?include=AVAILABILITY&product=POD&cloud=$c&count=1&cudaVersions=12.8,12.9,13.0 -> price of '$g'"
    for dc in "${dcs[@]}"; do
      [[ "$dc" == any ]] && dc=""
      echo "    POST $REST_SHOWN/pods $(api body --gpu "$g" --cloud "$c" --dc "$dc" --image "$IMAGE" --disk "$DISK_GB" --min-ram "$MIN_RAM_GB")"
    done
  done; done
  echo "then: watchdog deletes the pod at its cap; wait for publicIp + portMappings[22] and an SSH login (up to $SSH_WAIT_S s)"
  echo "    none within $SSH_WAIT_S s: DELETE the pod, remember its machineId, the next offer (at most $MAX_PODS pods)"
  echo "  scp -P <port> -i $SSH_KEY dsv41_cuda_probe root@<ip>:/root/dsv41_cuda_probe"
  echo "  ssh -p <port> -i $SSH_KEY root@<ip> 'timeout $PROBE_S /root/dsv41_cuda_probe'"
  echo "    on failure: DELETE the pod, remember its machineId, the next offer (at most $MAX_PODS pods)"
  echo "  scp -P <port> -i $SSH_KEY <payload>.tar.gz root@<ip>:$REMOTE.tar.gz"
  echo "  timeout $START_S ssh -p <port> -i $SSH_KEY root@<ip> 'mkdir -p $REMOTE/out && tar -xzf $REMOTE.tar.gz -C $REMOTE && cd $REMOTE &&"
  echo "    { DEADLINE=<cap - 120 s> nohup setsid bash -c \"bash devgpu-run.sh; echo \$? > out/devgpu-run.exit\" > out/devgpu-run.out 2>&1 < /dev/null & }'"
  echo "    (detached: the SSH session returns at once; on the pod a monitor appends free -m, the running step's processes"
  echo "    and RSS and dmesg | tail every 30 s to out/monitor.log; once a step returns, what is left of its process group"
  echo "    gets SIGTERM, SIGKILL 10 s later, before the next step starts, each kill in out/leftovers.log)"
  echo "  every $POLL_S s, a fresh connection each: timeout $POLL_SSH_S ssh -p <port> -i $SSH_KEY root@<ip>"
  echo "    'cat $REMOTE/out/devgpu-run.exit 2>/dev/null || echo running; tail -n 1 $REMOTE/out/steps.log'"
  echo "    a failed poll is retried; $POLL_FAILS failed in a row, or the run still going 30 s after its deadline: stop polling, pull"
  echo "    on the pod: env (nvidia-smi, MANIFEST), dsv41_engine_indexer_test --gpu, dsv41_engine_indexer_bench,"
  echo "    dsv41_engine_indexer_bench --ratio 1 --nk 262144 --nq 1,4096 --reps 5, dsv41_engine_qmatmul_test --gpu,"
  echo "    dsv41_engine_qmatmul_bench, dsv41_engine_attention_test --gpu (q2_K weights), --gpu --f16 and"
  echo "    --gpu --quick --mmq (the MMQ prefill option), dsv41_engine_attention_bench (q2_K) and --f16,"
  echo "    dsv41_engine_dense_test --gpu and --gpu --quick --mmq, dsv41_engine_dense_bench,"
  echo "    E05a: dsv41_engine_experts_test --gpu (tiny-a, tiny-b, three placements), E05c: dsv41_engine_experts_test"
  echo "    --gemm (the grouped prefill GEMM at real dims, one-hot rows bit for bit, its timing); per tiny model and placement"
  echo "    (all-vram, all-cpu, split) dsv41-engine --readers dense on the 64-token prompt and 16 greedy steps against"
  echo "    dump-<c>-16, --readers reuse on the short prompt against dump-<c>-short (engine_parity.py), the placements"
  echo "    against each other (engine_parity.py --engines), --f16 on tiny-a; each of these again with"
  echo "    --force-selection (the fork's router ids, index rows and tokens) against the fork and across placements;"
  echo "    E05b: per tiny model and placement, forced against the fork, the 64-token prompt in chunks of 24 with a ring"
  echo "    of 2 sets of 2 and 2 prefetched experts (--chunk 24 --ring-set 2 --ring-sets 2 --prefetch-experts 2"
  echo "    --prefill-split off: every expert outside the slab through the ring; the one-chunk runs read through the"
  echo "    prefill split, E06 cycle 2), and"
  echo "    those placements against each other; E05d: every e2e run's decode steps from the token graph (one CUDA graph"
  echo "    per token, the experts' doorbell), and per tiny model the split placement forced again with --no-graphs (every"
  echo "    op launched) against the fork and against the graph's run;"
  echo "    dsv41_engine_smoke (real dims, 40 layers, 8 experts per layer, a 4096-token chunk at ctx 16384, 32 decode"
  echo "    steps x 3, per-stage times), then E05b's: dsv41_engine_smoke --layers 8 --experts 384 --vram-experts-gb 16.12"
  echo "    --chunk 16384 --memory-table --pod-link <nvidia-smi pcie.link.gen.max:width.max> --pod-tflops <the GPU's"
  echo "    published dense FP16 tensor TFLOPS: A6000 154.8, A40 149.7, L40 181.05, A100 312, else 0> --project 'RTX PRO"
  echo "    6000:5:16:96:503.8' --project 'L40S:4:16:48:362.05' (the target's memory per chunk with the real 40 layers at"
  echo "    ctx 262144 first; 37 GiB pinned: the pod needs about 45 GB of RAM, MIN_RAM_GB=64 for this session; the"
  echo "    gates' numbers, per-layer copies and overlap, decode host syncs, the 40-layer projection at compute scales"
  echo "    1 and the published ratio), and E05c's: the same with --counts e05c/counts.json (routing and"
  echo "    hot set from the profile; the routed-expert compute against 35% of the published peak, decode hit rate, CPU"
  echo "    GB/s per thread, the 40-layer ms per token); E05d: in both, decode's split per layer per token (timed), the"
  echo "    token graph's ms per token, host syncs, graph launches and doorbells per token, and the 40-layer projection"
  echo "    against 30 tok/s);"
  echo "    E06: python3 src/tools/dsv41/engine/serve_check.py --engine ./dsv41-engine --model e05a/tiny-b.gguf"
  echo "    --tokenizer-gguf e06/tiny-vocab.gguf --reference cli (jinja2 installed first if the image lacks it; every path"
  echo "    it hands the server absolute, every process it started stopped on its way out, a failed check too): the serve"
  echo "    protocol with the prefix cache's paths, then serve/server.py's /v1/chat/completions non-streaming and streaming,"
  echo "    the generated ids against dsv41-engine --tokens-file <the rendered prompt> --n-predict 12; then"
  echo "    dsv41_engine_smoke --layers 8 --experts 384 --vram-experts-gb 16.12 --chunk 16384 --counts e05c/counts.json"
  echo "    --prefix-cache 512 --pod-link <the link> --floor-target 'RTX PRO 6000:5:16' (turn 2 RESUME and TTFT against a"
  echo "    cold read, the 40-layer projection, turn 2's prefill split (slab, ring, CPU) and its physical floor on the RTX PRO"
  echo "    6000 (the touched experts outside the slab over PCIe + CPU GB/s), forced rel L2 of the"
  echo "    live resume and the prompt-end restore <= 5e-2);"
  echo "    E08: per tiny model the split placement forced with --hot-policy static (no swap), then with --hot-policy"
  echo "    adaptive --hot-margin 0:-1 --hot-check (2 swaps per layer at every token, each token's swaps waited for and"
  echo "    the residency rows, slot map and every filled slot and home checked after it), its swaps and checks counted,"
  echo "    against the fork and against the static run (engine_parity.py --engines: logits within 5e-2, the same greedy"
  echo "    ids); every smoke above also prints E08's lines (hc and norms per layer from graphs and the steps' phase"
  echo "    split, a decode with the swaps' copies in flight and none published (copy only, margin 0:-1) against the"
  echo "    static one, copy only at the default margin, then the adaptive decode with its swaps published for"
  echo "    comparison, each with where a token's time goes; the targets <= 0.10 ms and copy only <= 1.1x with its"
  echo "    batches deferred when they outlast a token); the smokes' targets are PASS/FAIL lines, not exit codes;"
  echo "    E09: dsv41_engine_attention_bench --chunk 16384 --prefill-ctx 16384 --decode-ctx 16384 --reps 3"
  echo "    --peak-tflops <the GPU's published dense FP16 tensor TFLOPS> (a 16K chunk per role split by op, the real 40"
  echo "    layers' mean against 30 ms per layer, the attend roofline against the peak and the select scorer's floor,"
  echo "    attend <= 12 ms and select <= 4 ms per layer), every smoke's E09 targets line"
  echo "    (attention per layer, routed-expert compute <= 2.4 s and plan gaps + stalls <= 0.6 s on 40 layers), and"
  echo "    smoke-e09: smoke-e05c's run with --prefetch-experts 128 (the real-model session's prefetch)"
  echo "  the pull, at most $PULL_S s in all (E05a audit: scp -r of the out dir hung 30 min), then the pod is deleted at once;"
  echo "  tried on every way out once the run started (a failed start or poll, an error, Ctrl-C), not after the watchdog:"
  echo "    timeout $PULL_TAR_S ssh -p <port> -i $SSH_KEY root@<ip> 'cd $REMOTE && { find out -type f ! -name \"*.bin\" -print0 |"
  echo "      tar --null -czf out-logs.tar.gz -T - || [ \$? -eq 1 ]; } && { find out -type f -name \"*.bin\" -print0 |"
  echo "      tar --null -czf out-dumps.tar.gz -T - || [ \$? -eq 1 ]; } &&   (tar's 1: a file changed, the archive is whole)"
  echo "      stat -c %s out-logs.tar.gz out-dumps.tar.gz'"
  echo "    timeout <left of $PULL_S s> scp -P <port> -i $SSH_KEY root@<ip>:$REMOTE/out-logs.tar.gz $OUT/   (the logs first)"
  echo "    timeout <left of $PULL_S s> scp -P <port> -i $SSH_KEY root@<ip>:$REMOTE/out-dumps.tar.gz $OUT/  (only when <= PULL_CAP_MB=$PULL_CAP_MB MiB)"
  echo "    log: pull <s> s, logs <KiB> KiB, dumps <MiB> MiB"
  echo "  DELETE $REST_SHOWN/pods/<id> right after the pull, then GET it: gone or not running; cost = costPerHr x (delete - create) in $OUT/cost.txt"
  echo "  then locally: out-logs.tar.gz and out-dumps.tar.gz extracted into $OUT"
  echo "  exit 1 unless devgpu-run.sh exited 0 (out/devgpu-run.exit) and out/steps.log reports every step with exit 0 and ends with done"
  exit 0
fi

[[ -n "${RUNPOD_API_KEY:-}" ]] || { echo "devgpu.sh: RUNPOD_API_KEY is not set: export your RunPod API key" >&2; exit 1; }
[[ -f "$SSH_KEY" && -f "$SSH_KEY.pub" ]] || { echo "devgpu.sh: no SSH key pair $SSH_KEY(.pub)" >&2; exit 1; }
mkdir -p "$OUT"
log "results in $OUT"

# payload: the binaries checked against the tarball's MANIFEST, then the E05a data
stage="$OUT/payload"
# the cu128 payload only: pod-session.sh skips a GPU of GPU_TYPES that takes another flavor
declare -A stages=([cu128]="$stage")
stage_bins "$tarball" "$stage" "${bins[@]}"
# the qualification's probe beside the CUDA probe (pod-session.sh scps it from the stage before the payload)
if (( ${QUALIFY:-0} )); then
  log "$(machine_probe_ready cu128)" || exit 1
  cp "$MACHINE_PROBE_ROOT/cu128/machine-probe/dsv41_machine_probe" "$stage/"
fi
cp "$here/devgpu-run.sh" "$here/pod-steps.sh" "$here/../engine/engine_parity.py" "$stage/"
mkdir -p "$stage/e05c"
cp "$COUNTS" "$stage/e05c/counts.json"
mkdir -p "$stage/e05a/fork"
for c in a b; do
  cp "$E01/tiny-$c.gguf" "$stage/e05a/"
  python3 "$here/../engine/hot_sets.py" "$E01/tiny-$c.gguf" "$stage/e05a/hot-$c" >> "$OUT/devgpu.log"
done
for d in "${dumps[@]}"; do
  mkdir -p "$stage/e05a/fork/$d"
  # the selections and their scores: the forced runs read the first and third, engine_parity.py all four
  cp "$E01/$d/dump-manifest.json" "$E01/$d"/l_last.* "$E01/$d"/result_output.* "$E01/$d"/ffn_moe_topk.* \
    "$E01/$d"/ffn_moe_probs_biased.* "$E01/$d"/idx_top_k.* "$E01/$d"/idx_score.* "$stage/e05a/fork/$d/"
  python3 -c "import json, sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['prompt_token_ids'])))" \
    "$E01/$d/dump-manifest.json" > "$stage/e05a/prompt-${d#dump-}.txt"
done
mkdir -p "$stage/e06" "$stage/src"
python3 "$here/../engine/tiny_vocab.py" "$VOCAB" "$stage/e06/tiny-vocab.gguf" >> "$OUT/devgpu.log"
tar -C "$STRATA_DIR" -cf - --exclude="test_*.py" --exclude=__pycache__ "${server_src[@]}" | tar -C "$stage/src" -xf -
echo "$DEVGPU_PLAN" > "$stage/plan"
tar -C "$stage" -czf "$OUT/payload.tar.gz" MANIFEST plan devgpu-run.sh pod-steps.sh engine_parity.py e05a e05c e06 src "${bins[@]}"
log "payload $(du -h "$OUT/payload.tar.gz" | cut -f1): ${bins[*]} from $(basename "$tarball"), E05a data from $E01, E05c counts from $COUNTS, E06 tokenizer from $VOCAB"

trap cleanup EXIT
trap 'exit 130' INT TERM

pod_up
run_detached "$OUT/payload.tar.gz"
pull_results
end_pod   # the pod goes as soon as its results are here, before anything else runs locally
for t in out-logs out-dumps; do
  [[ -f "$OUT/$t.tar.gz" ]] && { tar -xzf "$OUT/$t.tar.gz" -C "$OUT" || log "WARNING: $t.tar.gz does not extract"; }
done
[[ -f "$OUT/out/steps.log" ]] && cat "$OUT/out/steps.log" | tee -a "$OUT/devgpu.log" >&2
grep -h "dsv41_engine_indexer_test (b)\|dsv41_engine_qmatmul_test (b)\|dsv41_engine_attention_test (b)\|dsv41_engine_dense_test (b)\|dsv41_engine_experts_test (b)\|engine_parity.py:\|dsv41_engine_smoke:\|serve_check.py:" "$OUT/out/"*.log 2>/dev/null | tee -a "$OUT/devgpu.log" >&2 || true
# the session's verdict: the remote script's exit and every step's line (the EXIT trap still deletes the pod)
steps_verdict || exit 1
log "every step passed"
