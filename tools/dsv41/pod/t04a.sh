#!/usr/bin/env bash
# The GPU-resident experts pod phase (t04a): runs on one A100 80 GB pod, writes everything
# under $OUT, and stops by itself within the wall-time cap (CAP_MIN, default 110 min): every command runs under
# `timeout` with the time left, a step that would not fit is skipped and recorded, and the servers it starts are
# stopped on exit. It never starts or stops the pod. Every step writes a start and an end line with its exit status
# to $OUT/steps.log (and t04a.log), and the last line of every run, however it ends, is its exit status.
#
#   bash /root/t04a/dsv41/pod/t04a.sh            # on the pod
#   bash /root/t04a/dsv41/pod/t04a.sh --speed    # on the pod: the speed steps only (below)
#   bash /root/t04a/dsv41/pod/t04a.sh --speed --profile    # on the pod: the speed steps on profile-driven hot sets
#   bash /root/t04a/dsv41/pod/t04a.sh --prefill  # on the pod: time to first token of 4K-32K prompts (below)
#   bash tools/dsv41/pod/t04a.sh --dry-run       # anywhere: prints the plan and every command, runs nothing (with
#                                                # --speed, --profile or --prefill too); exits 2 when the plan is over
#                                                # CAP_MIN (and with --prefill when the prompts fail their check)
#
# Upload first (the payload, $PAYLOAD, default /root/t04a):
#   $PAYLOAD/llama-dsv41-<commit8>-cu128-sm80-89.tar.gz   the fork tarball (llama-server, dsv41-moe-dump,
#                                                         dsv41-moe-replay, libstrata-moe.so)
#   $PAYLOAD/dsv41/                                       a copy of Strata's tools/dsv41 (this script included)
#   $PAYLOAD/run-1/                                       $BUILD_ROOT/oracle-dsv41/run-1, the stock fork's oracle dump
#   $PAYLOAD/counts.json, $PAYLOAD/prefill-prompts/       --prefill only: a real-model run's STRATA_MOE_COUNTS
#                                                         file (its results/t04a-results/counts.json) and the
#                                                         output of pod/prefill_prompts.py build
# and the model on the network volume ($MODEL_SRC, default /workspace/models/q2k).
#
# Parity is measured against stock with the same expert placement: run-1 computed
# the experts of some layers on CUDA0 and the rest on the CPU, and each backend matches only its own layers
# (about 5e-8 against 2e-2 across backends). read_dump.py --expert-placement reads that placement from run-1's
# command line; a hot set with every expert of exactly those layers (hot_set.py --layers-full) reproduces it.
#
# Steps, in order of priority:
#   setup     environment snapshot, flavor, tarball, numactl, model copy to local NVMe with parallel dd
#             (pod/model-copy.sh) and size check
#   hot sets  run-1's expert placement; hot_set.py on run-1 with the model's own tensor headers: the oracle
#             placement (--layers-full) and 40, 60 and 70 GiB by activation count
#   parity    dsv41-moe-replay of the 40 layers against run-1 with the oracle placement: max rel L2 <= 1e-6
#   greedy    64-token dsv41-moe-dump, stock with run-1's expert overrides and --no-repack against the hook with
#             every routed expert in host memory, --no-repack and the oracle placement hot set: identical tokens
#   baseline  stock llama-server with the stock reference flags (14 expert layers in VRAM): 120- and 480-token decode
#   hook      llama-server with the hook at 40, 60 and 70 GiB: 120- and 480-token decode, profile at exit
#   gaps      information, no bound: the replay with the 70 GiB hot set (every expert run-1 uses on the GPU) and on
#             the CPU only, each against run-1, which shows the cross-backend gap of the other backend's layers
#   extra     stock llama-server with every routed expert in host memory (the hook's flags without the hook)
# Each server run and the replays sample nvidia-smi every 5 s; free and the cgroup memory are snapshot per step.
# The hook's CPU misses run the library's default kernel (STRATA_MOE_CPU unset: fast).
#
# --speed (T04b, the speed table of the CPU kernels against the same pod's stock baseline): a NUMA gate right after
# the environment snapshot (exit 3 when the GPU's NUMA node has less MemFree than the largest warm tier of RAM_RUNS
# plus 8 GiB: one earlier host had 0.67 GB free on the GPU's node and every run there was slow), setup, the hot sets
# of SPEED_BUDGETS, a health line (uptime, nproc, loadavg, RLIMIT_MEMLOCK, NUMA MemFree) in verdicts.txt, the
# baseline with a health gate (exit 3 when its measured 120-token decode is under HEALTH_MIN tok/s or it does not
# start: a slow host, as one earlier pod was, would make every number meaningless), then per hot set of
# SPEED_BUDGETS the hook once per CPU kernel in CPU_KERNELS (hook-<GiB>-<kernel>, STRATA_MOE_CPU set), then the
# residency runs, then the extra (EXTRA=0 skips it). No placement, parity, greedy or gap steps. A 75-minute session
# of stock, hook-70 and hook-70-ram40 only:
#   SPEED_BUDGETS=70 RAM_RUNS=70:40 CPU_KERNELS=fast EXTRA=0 CAP_MIN=60 bash t04a.sh --speed
# The residency runs (T04c, --speed): per VRAM:RAM pair of RAM_RUNS, a hot set with that VRAM budget and a warm tier
# of that RAM budget (hot_set.py --ram-budget-gib) and the hook with STRATA_MOE_RAM_GB (hook-<VRAM>-ram<RAM>, fast
# kernel): every expert in neither tier is read from the local NVMe with O_DIRECT. 70:40 leaves the dense part, the
# engram tables' page cache and the staging about 36 GiB of the 116.4 GiB cgroup. The warm tier is CUDA pinned host
# memory (STRATA_MOE_RAM_MODE unset: the library's default with a GPU), which needs no RLIMIT_MEMLOCK (the RunPod
# containers have 8 MiB and no CAP_IPC_LOCK); RAM_MODE sets STRATA_MOE_RAM_MODE (pinned, locked or plain) for these
# runs. The mode each run used, from its log, goes to verdicts.txt; RLIMIT_MEMLOCK goes to env.txt and the health
# line.
# Every bench line in bench.tsv also has the server's major page faults during the request and their rate per
# second (/proc/<pid>/stat), the page-cache thrash T04c removes (the c3 run had 40k-60k per second).
#
# T04d (the c5 pod: the warm list had no signal and O_DIRECT for every other expert threw the page cache away):
# - COLD sets STRATA_MOE_COLD for the residency runs; unset leaves the library default, pagecache (the cold experts
#   from the fork's mapping through the page cache), direct is the T04c behaviour. Each run's line goes to
#   verdicts.txt.
# - RAM_RUNS takes `auto` for the RAM side (VRAM:auto, or a bare `auto` for one run per SPEED_BUDGETS): at the NUMA
#   gate hot_set.py --propose-ram-budget sizes the warm tier from the cgroup's limit minus WORKING_SET_GIB (default
#   4.5: during the c5 hook-70-ram40 benches the memory cgroup held 47.75 GB, io.log 00:15:18, of which 42.95 GB
#   the pinned tier, so 4.80 GB = 4.47 GiB of page cache of the dense part and the engram rows plus anonymous
#   memory, with every routed expert's page dropped) minus MARGIN_GIB (default 8, the NUMA gate's margin), at most the GPU node's MemFree minus the same margin; the
#   arithmetic goes to verdicts.txt and the run is named hook-<VRAM>-ram<GiB>. A proposal under 1 GiB exits 3.
# - --profile (with --speed): after the baseline, the hook with STRATA_MOE_COUNTS on the prompts of pod/prompts/
#   (code, prose, chat; pod/prompts/README.md), up to PROFILE_TOKENS (default 256) tokens each, starting from a hot
#   set of the largest VRAM budget built from run-1; the hot sets of SPEED_BUDGETS and RAM_RUNS are then built from
#   its counts file (hot_set.py --counts, decode calls) instead of run-1's 95 tokens. A profile that fails falls back
#   to run-1 with a FAIL line in verdicts.txt, so the speed runs still happen.
# - The plan: every run first prints its steps with time estimates (from the c5 pod's step times) and the total
#   against CAP_MIN, plus the steps outside the script; a dry run over the cap exits 2. A 70-minute session:
#   SPEED_BUDGETS=70 RAM_RUNS=auto CPU_KERNELS=fast EXTRA=0 CAP_MIN=60 bash t04a.sh --speed --profile
#
# --prefill: time to first
# token (TTFT) of the 4K, 8K, 16K and 32K code-corpus prompts of pod/prefill_prompts.py ($PREFILL_PROMPTS, default
# $PAYLOAD/prefill-prompts, checked first: sha256 and token counts within 2% of the targets), one run each. The
# indexer's scores take 2 x 32 x ctx x ubatch x 4 bytes of VRAM, so the prompts run on two server configurations:
# A, --ctx-size PREFILL_CTX_A (default 16896) and batch = ubatch = PREFILL_UB_A (default 4096), for 4K, 8K and 16K
# (16.5 GiB of scores); B, PREFILL_CTX_B (default 33280) and PREFILL_UB_B (default 2048), for 32K (16.25 GiB). Per
# 4096-token ubatch nearly every expert is used, so every expert outside the slab is streamed once per ubatch (a
# smaller ubatch streams the same bytes more often). Steps: env, the NUMA gate and the auto warm tier of --speed,
# setup, prompts, one hot set for both (hot_set.py --counts $COUNTS, default $PAYLOAD/counts.json: the T04d pod's
# profile, at PREFILL_VRAM GiB of VRAM and the auto warm tier), then the arms in this order: hook-prefill-a,
# hook-prefill-b, stock-prefill-a, stock-prefill-b, and hook-prefill-a-direct when it runs (below). Every arm runs
# llama-server with every routed expert in host memory and -nr (the hook's flags, so hook and stock of a
# configuration differ only in LLAMA_STRATA_MOE=1: the same-flags parity of T03), -np 1 and its configuration's ctx
# and ubatch. Each prompt is one /completion request, raw text (the GGUF adds no BOS), cache_prompt false,
# temperature 0; its prompt_n must equal the manifest's count; the 4K one (configuration A) takes 64 greedy tokens
# for the token agreement of the hook against stock (identical tokens out of 64, first divergence). Results go to
# prefill.tsv (arm, prompt, tokens, prompt_n, prompt ms, prefill tok/s, predicted_n, client s), the 4K tokens to
# <arm>-4k.tokens.json; after each request the server must still run and its log hold no CUDA error or GGML_ASSERT
# (the int32 overflow of the indexer scores crashed 131072 x 512 on a B300). The goal's bar is hook-prefill-a's 16K
# TTFT <= 60 s.
# - The hot set: PREFILL_VRAM (default computed) is the VRAM the fork leaves for the slab in the worse of the two
#   configurations, less PREFILL_RESERVE_GIB (default 8); the plan prints the arithmetic of both.
# - Dense against expert time: after each hook arm stops (STRATA_MOE_PROFILE=1, its summary at exit), verdicts.txt
#   gets per prompt the server's ms per ubatch (prompt ms / ubatches) and once per arm the same over all its prompts,
#   split into the MoE op's ms per ubatch and the rest, the dense part. The op time is the profile's prefill line:
#   the prefill calls only, without the server's 2-token warmup and each layer's first call (the slab upload and the
#   warm-tier fill), the arm's mean over its prompts, so the split is per arm, not per prompt. Then the prefill
#   profile per ubatch (bytes copied and staged, copy, compute and GPU ms, overlap, staging ms). For stock, the ms
#   per ubatch.
# - STRATA_MOE_PREFILL_COLD: hook-prefill-a runs with the library default (pagecache). When its profile shows
#   staging-bound passes (the host's staging ms per ubatch over the copy stream's busy ms, so the PCIe copies waited
#   for the staging) and the time left holds its estimate, hook-prefill-a-direct reruns the 16K prompt with
#   STRATA_MOE_PREFILL_COLD=direct; otherwise verdicts.txt says why not.
# - The plan must fit a 70-minute session with the 6.3 minutes outside the script and CAP_MIN; when it does not,
#   stock-prefill-b (the 32K stock arm) is dropped first and the plan says so. hook-prefill-a-direct is conditional
#   and outside the total; the plan prints its estimate. A 70-minute session: CAP_MIN=63 bash t04a.sh --prefill
#
# Environment: PAYLOAD, OUT (default /root/t04a-results), MODEL_SRC, MODEL_DIR (default /root/models/q2k),
# CAP_MIN (default 110), THREADS (default 16), PORT (default 18000); with --speed also HEALTH_MIN (default 15),
# SPEED_BUDGETS (default "40 60 70", the hook runs' hot-set GiB, "" for none), CPU_KERNELS (default "fast ref", each
# fast or ref), RAM_RUNS (default "70:40", VRAM:RAM GiB pairs, RAM a whole number or auto, or a bare auto; "" for
# none), RAM_MODE (default unset: the library's default), COLD (default unset: the library's default), WORKING_SET_GIB
# (default 4.5), MARGIN_GIB (default 8), PROFILE_TOKENS (default 256) and EXTRA (default 1; 0 skips stock-hostexps);
# with --prefill also PREFILL_PROMPTS, COUNTS, PREFILL_CTX_A, PREFILL_UB_A, PREFILL_CTX_B, PREFILL_UB_B,
# PREFILL_VRAM (default computed, or whole GiB), PREFILL_RESERVE_GIB, PREFILL_RAM (default auto, or whole GiB),
# WORKING_SET_GIB and MARGIN_GIB.
set -Eeuo pipefail

DRY=0
SPEED=0
PROFILE=0
PREFILL=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --speed) SPEED=1 ;;
    --profile) PROFILE=1 ;;
    --prefill) PREFILL=1 ;;
    -h|--help) sed -n '2,129p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "t04a.sh: unknown argument '$arg' (only --dry-run, --speed, --profile and --prefill)" >&2; exit 2 ;;
  esac
done
(( ! PROFILE || SPEED )) || { echo "t04a.sh: --profile needs --speed" >&2; exit 2; }
(( ! PREFILL || ! SPEED )) || { echo "t04a.sh: --prefill is a session of its own, without --speed" >&2; exit 2; }

PAYLOAD="${PAYLOAD:-/root/t04a}"
OUT="${OUT:-/root/t04a-results}"
MODEL_SRC="${MODEL_SRC:-/workspace/models/q2k}"
MODEL_DIR="${MODEL_DIR:-/root/models/q2k}"
CAP_MIN="${CAP_MIN:-110}"
THREADS="${THREADS:-16}"
PORT="${PORT:-18000}"
HEALTH_MIN="${HEALTH_MIN:-15}"
CPU_KERNELS="${CPU_KERNELS:-fast ref}"
for k in $CPU_KERNELS; do
  [[ "$k" == fast || "$k" == ref ]] || { echo "t04a.sh: CPU_KERNELS has '$k' (each must be fast or ref)" >&2; exit 2; }
done
SPEED_BUDGETS="${SPEED_BUDGETS-40 60 70}"
for b in $SPEED_BUDGETS; do
  [[ "$b" =~ ^[0-9]+$ ]] || { echo "t04a.sh: SPEED_BUDGETS has '$b' (each must be whole GiB)" >&2; exit 2; }
done
PREFILL_PROMPTS="${PREFILL_PROMPTS:-${PAYLOAD:-/root/t04a}/prefill-prompts}"
COUNTS="${COUNTS:-${PAYLOAD:-/root/t04a}/counts.json}"
# the two server configurations: A for the 4K, 8K and 16K prompts, B for 32K
PREFILL_CTX_A="${PREFILL_CTX_A:-16896}"
PREFILL_UB_A="${PREFILL_UB_A:-4096}"
PREFILL_CTX_B="${PREFILL_CTX_B:-33280}"
PREFILL_UB_B="${PREFILL_UB_B:-2048}"
PREFILL_VRAM="${PREFILL_VRAM:-}"
PREFILL_RESERVE_GIB="${PREFILL_RESERVE_GIB:-8}"
PREFILL_RAM="${PREFILL_RAM:-auto}"
for v in "$PREFILL_CTX_A" "$PREFILL_UB_A" "$PREFILL_CTX_B" "$PREFILL_UB_B" ${PREFILL_VRAM:+"$PREFILL_VRAM"}; do
  [[ "$v" =~ ^[1-9][0-9]*$ ]] ||
    { echo "t04a.sh: PREFILL_CTX_A=$PREFILL_CTX_A PREFILL_UB_A=$PREFILL_UB_A PREFILL_CTX_B=$PREFILL_CTX_B PREFILL_UB_B=$PREFILL_UB_B PREFILL_VRAM=$PREFILL_VRAM (whole numbers; PREFILL_VRAM may be unset)" >&2; exit 2; }
done
[[ "$PREFILL_RESERVE_GIB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "t04a.sh: PREFILL_RESERVE_GIB=$PREFILL_RESERVE_GIB (GiB, >= 0)" >&2; exit 2; }
[[ "$PREFILL_RAM" == auto || "$PREFILL_RAM" =~ ^[0-9]+$ ]] || { echo "t04a.sh: PREFILL_RAM=$PREFILL_RAM (auto or whole GiB)" >&2; exit 2; }
# --prefill: the VRAM the hook server takes besides its slab, per configuration, from the c7 pod (T04d, A100 SXM 80 GB:
# the fork took 79.250 - 74.302 = 4.948 GiB before the slab at ctx 4096 and ubatch 1024, its indexer scores of
# 2 x 32 x 4096 x 1024 x 4 B = 1 GiB among them) plus what the configuration adds that is known: the scores' growth
# (the mul_mat output and its cont, live together: 2 x 32 x ctx x ubatch x 4 B, less c7's 1 GiB), the ring (3 sets x
# 16 experts x 12,810,240 B), a prefill pass's buffers (~350 MiB, moe_prefill.h) and the input (ubatch x 5120 x 4 B).
# Prints "GiB text".
prefill_fixed_gib() {
  awk -v c="$1" -v u="$2" 'BEGIN {
    g = 1024 * 1024 * 1024; idx = 2 * 32 * c * u * 4 / g; base = 2 * 32 * 4096 * 1024 * 4 / g
    ring = 3 * 16 * 12810240 / g; pass = 350 / 1024; x = u * 5120 * 4 / g; t = 4.948 + idx - base + ring + pass + x
    printf "%.2f fork at c7 ctx 4096 x ub 1024 4.95 + indexer scores %.2f - %.2f in it + ring %.2f + pass %.2f + input %.2f = %.2f GiB", t, idx, base, ring, pass, x, t }'
}
# The hot set (PREFILL_VRAM unset): 79.25 GiB of usable VRAM (c7) less the worse configuration less
# PREFILL_RESERVE_GIB, rounded down. The reserve holds what nothing has measured: the KV cache at the larger ctx and
# the fork's other compute buffers at the larger ubatch; its default, 8 GiB, is twice c7's 3.95 GiB outside the
# scores (the dense weights, the KV cache at 4096 and the compute buffers at 1024 together), so all of that could
# double. The pod logs each server's peak VRAM.
PREFILL_FIXED_A="$(prefill_fixed_gib "$PREFILL_CTX_A" "$PREFILL_UB_A")"
PREFILL_FIXED_B="$(prefill_fixed_gib "$PREFILL_CTX_B" "$PREFILL_UB_B")"
PREFILL_WORSE="$(awk -v a="${PREFILL_FIXED_A%% *}" -v b="${PREFILL_FIXED_B%% *}" 'BEGIN { print (a + 0 >= b + 0) ? "A" : "B" }')"
PREFILL_WORSE_GIB="${PREFILL_FIXED_A%% *}"
[[ "$PREFILL_WORSE" == B ]] && PREFILL_WORSE_GIB="${PREFILL_FIXED_B%% *}"
PREFILL_VRAM_SOURCE="set"
if [[ -z "$PREFILL_VRAM" ]]; then
  PREFILL_VRAM="$(awk -v w="$PREFILL_WORSE_GIB" -v r="$PREFILL_RESERVE_GIB" 'BEGIN { v = int(79.25 - w - r); print (v > 0) ? v : 0 }')"
  PREFILL_VRAM_SOURCE="computed"
  (( PREFILL_VRAM >= 1 )) || { echo "t04a.sh: --prefill: no hot set fits: 79.25 - $PREFILL_WORSE_GIB (configuration $PREFILL_WORSE) - $PREFILL_RESERVE_GIB < 1 GiB" >&2; exit 2; }
fi
RAM_RUNS="${RAM_RUNS-70:40}"
# --prefill: one residency run, the prefill hot set at PREFILL_VRAM with the PREFILL_RAM warm tier
(( PREFILL )) && RAM_RUNS="$PREFILL_VRAM:$PREFILL_RAM"
RAM_PAIRS=()  # VRAM:RAM, RAM a whole number of GiB or auto (sized at the NUMA gate)
for p in $RAM_RUNS; do
  if [[ "$p" == auto ]]; then
    [[ -n "$SPEED_BUDGETS" ]] || { echo "t04a.sh: RAM_RUNS=auto takes its VRAM budgets from SPEED_BUDGETS, which is empty" >&2; exit 2; }
    for b in $SPEED_BUDGETS; do RAM_PAIRS+=("$b:auto"); done
    continue
  fi
  [[ "$p" =~ ^[0-9]+:([0-9]+|auto)$ ]] ||
    { echo "t04a.sh: RAM_RUNS has '$p' (each must be VRAM:RAM in whole GiB, RAM or the whole entry may be auto)" >&2; exit 2; }
  RAM_PAIRS+=("$p")
done
WARM_MAX=0  # the largest fixed warm tier of RAM_RUNS, GiB: the NUMA gate's need
RAM_AUTO=0  # 1: a run's warm tier is sized at the gate
for p in "${RAM_PAIRS[@]}"; do
  if [[ "${p##*:}" == auto ]]; then RAM_AUTO=1; continue; fi
  (( 10#${p##*:} <= WARM_MAX )) || WARM_MAX=$(( 10#${p##*:} ))
done
WORKING_SET_GIB="${WORKING_SET_GIB:-4.5}"
MARGIN_GIB="${MARGIN_GIB:-8}"
for v in "$WORKING_SET_GIB" "$MARGIN_GIB"; do
  [[ "$v" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "t04a.sh: WORKING_SET_GIB=$WORKING_SET_GIB MARGIN_GIB=$MARGIN_GIB (GiB, >= 0)" >&2; exit 2; }
done
PROFILE_TOKENS="${PROFILE_TOKENS:-256}"
[[ "$PROFILE_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "t04a.sh: PROFILE_TOKENS=$PROFILE_TOKENS (a whole number >= 1)" >&2; exit 2; }
COLD="${COLD:-}"
[[ -z "$COLD" || "$COLD" == pagecache || "$COLD" == direct ]] ||
  { echo "t04a.sh: COLD=$COLD (pagecache, direct or unset)" >&2; exit 2; }
COLD_ENV=()
[[ -z "$COLD" ]] || COLD_ENV=(STRATA_MOE_COLD="$COLD")
RAM_MODE="${RAM_MODE:-}"
[[ -z "$RAM_MODE" || "$RAM_MODE" == pinned || "$RAM_MODE" == locked || "$RAM_MODE" == plain ]] ||
  { echo "t04a.sh: RAM_MODE=$RAM_MODE (pinned, locked, plain or unset)" >&2; exit 2; }
RAM_MODE_ENV=()
[[ -z "$RAM_MODE" ]] || RAM_MODE_ENV=(STRATA_MOE_RAM_MODE="$RAM_MODE")
EXTRA="${EXTRA:-1}"
[[ "$EXTRA" == 0 || "$EXTRA" == 1 ]] || { echo "t04a.sh: EXTRA=$EXTRA (0 or 1)" >&2; exit 2; }
# the NUMA gate: the GPU's node must have this much MemFree at start (the warm tier is allocated there, numactl
# --cpunodebind keeps the server's memory local)
NUMA_NEED_GIB=$(( WARM_MAX + 8 ))
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORACLE="$PAYLOAD/run-1"
MODEL="$MODEL_DIR/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf"
BUDGETS=(40 60 70)
# shellcheck disable=SC2206  # whole numbers, checked above
(( SPEED )) && BUDGETS=($SPEED_BUDGETS)
# the oracle placement: 14 full layers of 384 experts take 64.14 GiB of slab (hot_set.py on the layout)
ORACLE_BUDGET=66
INFO_BUDGET=70
START=$(date +%s)
DEADLINE=$(( START + CAP_MIN * 60 ))

# shellcheck source=tools/dsv41/pod/t04a-lib.sh
source "$TOOLS/pod/t04a-lib.sh"

# --profile: the profiling server starts from run-1's hot set of the largest VRAM budget of the speed runs
PROMPTS=("$TOOLS"/pod/prompts/*.txt)
PROFILE_VRAM=0
for b in "${BUDGETS[@]}"; do (( b <= PROFILE_VRAM )) || PROFILE_VRAM=$b; done
for p in "${RAM_PAIRS[@]}"; do (( 10#${p%%:*} <= PROFILE_VRAM )) || PROFILE_VRAM=$(( 10#${p%%:*} )); done
if (( PROFILE )); then
  (( PROFILE_VRAM > 0 )) || { echo "t04a.sh: --profile needs a VRAM budget in SPEED_BUDGETS or RAM_RUNS" >&2; exit 2; }
  [[ -f "${PROMPTS[0]}" ]] || { echo "t04a.sh: --profile: no prompts in $TOOLS/pod/prompts" >&2; exit 2; }
fi

# The plan, in minutes, from a measured pod's step times (A100 SXM, its steps.log):
# setup 4.3 (model copy 236 s), stock baseline 8.4 (listening after 335 s), hook-70-fast 6.0 (245 s), hook-70-ram40
# 9.9 (285 s with the 40 GiB pinned in 34.9 s and filled at 5.3 GB/s, and 4.6 min of benches at 4.3 tok/s). A
# residency run is estimated as 4.0 + its warm tier's allocation (0.87 s per GiB) and fill (5.3 GB/s) + 4.6 + 0.4
# to stop, with an auto tier at the cgroup's 116 GiB as the worst case; a profile prompt as PROFILE_TOKENS at 18
# tok/s (the c5 hooks' decode) plus 10 s. A ref-kernel or extra run has no --speed measurement yet: as hook-70-ram40.
plan_ram_min() {
  local r="$1"
  [[ "$r" == auto ]] && r=116
  awk -v r="$r" 'BEGIN { printf "%.1f", 4.0 + r * 0.87 / 60 + r * 1.0737 / 5.3 / 60 + 4.6 + 0.4 }'
}
# --prefill, from the c7 pod (T04d, steps.log): the hook with the 103 GiB auto warm tier listening after 636 s
# (10.6 min, the pinned allocation's 104.5 s in it), its warm fill at the layers' first calls at 5.4 GB/s (0.34 min
# for 103 GiB), stock listening after 345 s (5.75 min), setup 4.2 min; then the prompts at an assumed rate, since no
# long prompt has been measured on the A100: stock at the goal's 45 tok/s reference for short prompts, the hook at
# 100 tok/s (a third of its 270 tok/s bar, so a slow hook still fits); the 64 greedy tokens at 18 tok/s (c5's
# decode); 0.4 min to stop. The tokens are the targets: A 4096 + 8192 + 16384, B 32704.
PREFILL_TOKENS_A=$(( 4096 + 8192 + 16384 ))
PREFILL_TOKENS_B=32704
PREFILL_SESSION_MIN=70
PREFILL_OUTSIDE_MIN=6.3
# plan_prefill_min hook|stock TOKENS GREEDY_TOKENS: one arm's minutes
plan_prefill_min() {
  awk -v k="$1" -v n="$2" -v d="$3" 'BEGIN {
    s = (k == "hook") ? 10.6 + 0.34 : 5.75; r = (k == "hook") ? 100 : 45
    printf "%.1f", s + n / r / 60 + d / 18 / 60 + 0.4 }'
}
PREFILL_STOCK_B=1      # 0: stock-prefill-b dropped by the plan
PREFILL_DIRECT_MIN=""  # the estimate of hook-prefill-a-direct, the 16K prompt
PLAN_NOTES=()
make_plan() {
  plan_add env 0.1
  (( SPEED || PREFILL )) && plan_add numa-gate 0.1
  plan_add setup 4.5
  if (( PREFILL )); then
    plan_add "prompts + hot set" 0.3
    plan_add hook-prefill-a "$(plan_prefill_min hook "$PREFILL_TOKENS_A" 64)"
    plan_add hook-prefill-b "$(plan_prefill_min hook "$PREFILL_TOKENS_B" 0)"
    plan_add stock-prefill-a "$(plan_prefill_min stock "$PREFILL_TOKENS_A" 64)"
    local stock_b sum limit
    stock_b="$(plan_prefill_min stock "$PREFILL_TOKENS_B" 0)"
    sum="$(printf '%s\n' "${PLAN_MINUTES[@]}" "$stock_b" 0.2 | awk '{ s += $1 } END { printf "%.1f", s }')"
    # the script's limit: CAP_MIN, and the session's 70 minutes less the 6.3 outside the script
    limit="$(awk -v c="$CAP_MIN" -v s="$PREFILL_SESSION_MIN" -v o="$PREFILL_OUTSIDE_MIN" 'BEGIN { l = s - o; printf "%.1f", (c + 0 < l) ? c : l }')"
    if awk -v t="$sum" -v l="$limit" 'BEGIN { exit !(t + 0 <= l + 0) }'; then
      plan_add stock-prefill-b "$stock_b"
    else
      PREFILL_STOCK_B=0
      PLAN_NOTES+=("stock-prefill-b ($stock_b min) dropped: with it the script takes $sum min, over $limit (CAP_MIN $CAP_MIN, and the $PREFILL_SESSION_MIN-minute session less $PREFILL_OUTSIDE_MIN outside); the 32K prompt runs on the hook only")
    fi
    plan_add "snapshot-end + pack" 0.2
    PREFILL_DIRECT_MIN="$(plan_prefill_min hook 16384 0)"
    sum="$(printf '%s\n' "${PLAN_MINUTES[@]}" "$PREFILL_DIRECT_MIN" | awk '{ s += $1 } END { printf "%.1f", s }')"
    PLAN_NOTES+=("hook-prefill-a-direct (the 16K prompt with STRATA_MOE_PREFILL_COLD=direct, $PREFILL_DIRECT_MIN min) is conditional and not in the total: only when hook-prefill-a's profile is staging-bound and the time left holds it (with it the script would take $sum of $limit min)")
    return 0
  fi
  if (( ! SPEED )); then
    plan_add "placement + hot sets" 0.5
    plan_add replay-oracle 15
    plan_add greedy 25
  fi
  local b k p
  if (( PROFILE )); then plan_add hot-set-profile 0.1; elif (( SPEED )); then plan_add "hot sets" 0.2; fi
  plan_add baseline 8.5
  if (( PROFILE )); then
    plan_add "profile (${#PROMPTS[@]} prompts)" "$(awk -v n="${#PROMPTS[@]}" -v t="$PROFILE_TOKENS" \
      'BEGIN { printf "%.1f", 4.1 + n * (t / 18 + 10) / 60 + 0.5 }')"
    plan_add "hot sets from the counts" 0.2
  fi
  for b in "${BUDGETS[@]}"; do
    if (( ! SPEED )); then plan_add "hook-$b" 6.0; continue; fi
    for k in $CPU_KERNELS; do
      if [[ "$k" == fast ]]; then plan_add "hook-$b-fast" 6.0; else plan_add "hook-$b-ref" 9.9; fi
    done
  done
  if (( SPEED )); then
    for p in "${RAM_PAIRS[@]}"; do plan_add "hook-${p%%:*}-ram${p##*:}" "$(plan_ram_min "${p##*:}")"; done
  fi
  (( ! SPEED )) && plan_add "replay gaps" 20
  (( EXTRA )) && plan_add stock-hostexps 9.9
  plan_add "snapshot-end + pack" 0.2
}
make_plan

# the flags of the stock reference run (14 expert layers, blk 26-39, in VRAM)
L14='blk\.(2[6-9]|3[0-9])\.ffn_(down|gate|up)_exps\.weight=CUDA0'
CPU_EXPS='ffn_(down|gate|up)_exps\.weight=CPU'
COMMON=(-lm mmap --n-gpu-layers 99 --override-tensor 'engram_embd\.weight=CPU' --flash-attn on --ctx-size 4096
        --batch-size 2048 --ubatch-size 1024 --threads "$THREADS" --threads-batch 32 --numa numactl)
STOCK14=(--override-tensor "$L14" --override-tensor "$CPU_EXPS" "${COMMON[@]}")
# the hook needs every routed expert in host memory in ggml's layout: all of them on the CPU, --no-repack
HOSTEXPS=(--override-tensor "$CPU_EXPS" -nr "${COMMON[@]}")
SERVER_ONLY=(--host 127.0.0.1 --port "$PORT" --jinja --no-reasoning-preserve)

LOG="$OUT/t04a.log"
# a dry run writes nothing
if (( DRY )); then
  LOG=/dev/null
else
  mkdir -p "$OUT"
  [[ -z "$(ls -A "$OUT")" ]] || { echo "t04a.sh: OUT=$OUT is not empty" >&2; exit 1; }
fi

VRAM_PID=""
vram_start() {
  (( DRY )) && return 0
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits || true; sleep 5; done ) \
    > "$OUT/$1.vram" 2>&1 &
  VRAM_PID=$!
}
vram_stop() {
  (( DRY )) && return 0
  if [[ -n "$VRAM_PID" ]]; then kill "$VRAM_PID" 2>/dev/null || true; wait "$VRAM_PID" 2>/dev/null || true; VRAM_PID=""; fi
  if [[ -s "$OUT/$1.vram" ]]; then log "$1: peak VRAM $(sort -n "$OUT/$1.vram" | tail -n 1) MiB"; fi
  return 0
}

SERVER_PID=""
# server_start NAME ENV_ASSIGNMENTS... -- LLAMA_SERVER_ARGS...: starts llama-server in the background on the GPU's
# NUMA node and waits until it listens (up to 15 min: a cold page cache load took 6 min in the oracle dump session)
server_start() {
  local name="$1"; shift
  local envs=()
  while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  local cmd=(env -u LD_LIBRARY_PATH "${envs[@]}" numactl --cpunodebind="$NODE" "$R/bin/llama-server"
             -m "$MODEL" "$@" "${SERVER_ONLY[@]}")
  if (( DRY )); then echo "+ [$name] $(printf '%q ' "${cmd[@]}") &"; return 0; fi
  step_start "$name"
  printf '%q ' "${cmd[@]}" > "$OUT/$name.cmd"
  vram_start "$name"
  "${cmd[@]}" > "$OUT/$name.log" 2>&1 &
  SERVER_PID=$!
  local waited=0 limit=$(( 15 * 60 ))
  (( limit < $(left) )) || limit=$(left)
  while ! grep -qa "listening on" "$OUT/$name.log"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      local rc=0
      wait "$SERVER_PID" || rc=$?
      log "$name: server died (exit $rc): $(tail -n 3 "$OUT/$name.log" | tr '\n' ' ')"
      SERVER_PID=""; vram_stop "$name"; step_end "$name" "$rc"; return 1
    fi
    if (( waited >= limit )); then log "$name: not listening after $waited s"; server_stop "$name"; return 1; fi
    sleep 5; waited=$(( waited + 5 ))
  done
  log "$name: listening after $waited s"
}

# SIGINT first: llama-server then exits normally, so libstrata-moe's profile summary (atexit) reaches its log
server_stop() {
  local name="$1"
  if (( DRY )); then echo "+ [$name] kill -INT <llama-server>"; return 0; fi
  local rc=0
  if [[ -n "$SERVER_PID" ]]; then
    kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || rc=$?
    SERVER_PID=""
  fi
  vram_stop "$name"
  step_end "$name" "$rc"
  snapshot "$name"
}

cleanup() {
  local rc=$?
  if [[ -n "$SERVER_PID" ]]; then kill -KILL "$SERVER_PID" 2>/dev/null || true; fi
  if [[ -n "$VRAM_PID" ]]; then kill "$VRAM_PID" 2>/dev/null || true; fi
  on_exit "$rc"
}
trap cleanup EXIT
trap 'on_err "$LINENO" "$BASH_COMMAND" "$?"' ERR

# one decode request as the stock reference sweep (temperature 0); ignore_eos makes the answer exactly MAXTOK tokens.
# Prints "predicted_n predicted_per_second prompt_n prompt_per_second"; bench.tsv gets that and the server's major
# page faults over the request (fault_rate).
# shellcheck disable=SC2016  # python source, expanded by nobody
BENCH_PY='import json, sys
t = json.load(sys.stdin)["timings"]
print(t["predicted_n"], round(t["predicted_per_second"], 2), t["prompt_n"], round(t["prompt_per_second"], 2))'
bench() {
  local name="$1" maxtok="$2" kind="$3"
  local body="{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function to reverse a linked list. Only the function.\"}],\"max_tokens\":$maxtok,\"temperature\":0,\"ignore_eos\":true,\"stream\":false}"
  local r rc=0
  if (( DRY )); then
    echo "+ [$name-$maxtok-$kind] curl -s --max-time 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '$body' | python3 -c <timings>; majflt of llama-server before and after"
    return 0
  fi
  step_start "bench-$name-$maxtok-$kind"
  local budget; budget=$(left); (( budget < 600 )) || budget=600
  local f0 f1 t0 t1
  f0="$(majflt "$SERVER_PID")"; t0="$(date +%s.%N)"
  r="$(curl -s --max-time "$budget" "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
       -d "$body" | python3 -c "$BENCH_PY" 2>&1)" || rc=$?
  f1="$(majflt "$SERVER_PID")"; t1="$(date +%s.%N)"
  (( rc == 0 )) || r="fail ($r)"
  local faults; faults="$(fault_rate "$f0" "$f1" "$t0" "$t1")"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$maxtok" "$kind" "$r" "$faults" >> "$OUT/bench.tsv"
  log "bench $name $maxtok $kind: predicted_n tok/s prompt_n prompt_tok/s = $r; $faults"
  step_end "bench-$name-$maxtok-$kind" "$rc"
}

# the two lengths, each once to warm the page cache and once measured
bench_all() {
  local name="$1"
  for n in 120 480; do bench "$name" "$n" warm; bench "$name" "$n" measured; done
}

# profile_request PROMPT_FILE (--profile): one request with the file as the user message, up to PROFILE_TOKENS tokens,
# temperature 0 and no ignore_eos (the routing of a natural answer); its timings go to profile.tsv
PROFILE_BODY_PY='import json, sys
print(json.dumps({"model": "default", "messages": [{"role": "user", "content": open(sys.argv[1]).read().strip()}],
                  "max_tokens": int(sys.argv[2]), "temperature": 0, "stream": False}))'
profile_request() {
  local name; name="profile-$(basename "$1" .txt)"
  if (( DRY )); then
    echo "+ [$name] curl -s --max-time 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d <${1#"$TOOLS/"} as the user message, max_tokens $PROFILE_TOKENS, temperature 0> | python3 -c <timings>"
    return 0
  fi
  step_start "$name"
  local budget r rc=0; budget=$(left); (( budget < 600 )) || budget=600
  r="$(python3 -c "$PROFILE_BODY_PY" "$1" "$PROFILE_TOKENS" | curl -s --max-time "$budget" \
       "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @- |
       python3 -c "$BENCH_PY" 2>&1)" || rc=$?
  (( rc == 0 )) || r="fail ($r)"
  printf '%s\t%s\n' "$name" "$r" >> "$OUT/profile.tsv"
  log "$name: predicted_n tok/s prompt_n prompt_tok/s = $r"
  step_end "$name" "$rc"
}

# replay NAME MAX_REL_L2 ENV_ASSIGNMENTS...: dsv41-moe-replay of the 40 layers against run-1 with the profile on
replay() {
  local name="$1" bound="$2"; shift 2
  local rc=0
  vram_start "$name"
  run "$name" env -u LD_LIBRARY_PATH "$@" STRATA_MOE_PROFILE=1 numactl --cpunodebind="$NODE" "$R/bin/dsv41-moe-replay" \
    -m "$MODEL" --dump-dir "$ORACLE" -t "$THREADS" --max-rel-l2 "$bound" || rc=$?
  vram_stop "$name"
  snapshot "$name"
  return "$rc"
}

# ---------------------------------------------------------------- setup
(( PREFILL )) && log "T06 --prefill: prompts $PREFILL_PROMPTS, counts $COUNTS, configuration A ctx $PREFILL_CTX_A x ubatch $PREFILL_UB_A (4K, 8K, 16K), B ctx $PREFILL_CTX_B x ubatch $PREFILL_UB_B (32K), hot set $PREFILL_VRAM GiB ($PREFILL_VRAM_SOURCE), warm tier $PREFILL_RAM, RAM mode: ${RAM_MODE:-the library default}, cold path: ${COLD:-the library default}"
log "T04a pod script$( (( SPEED )) && echo " (--speed$( (( PROFILE )) && echo " --profile, $PROFILE_TOKENS tokens per prompt"), hot sets: ${SPEED_BUDGETS:-none}, CPU kernels: $CPU_KERNELS, RAM runs: ${RAM_PAIRS[*]:-none}, RAM mode: ${RAM_MODE:-the library default}, cold path: ${COLD:-the library default}, extra: $EXTRA)"), cap $CAP_MIN min, OUT=$OUT, PAYLOAD=$PAYLOAD"
# outside the script, on the c5 pod: create to SSH 0.3 min, upload 5.4 (23:45:33 to 23:50:56), pull and delete 0.6
PLAN_FITS=1
plan_print "$CAP_MIN" | tee -a "$LOG" || PLAN_FITS=0
for note in "${PLAN_NOTES[@]}"; do echo "plan: $note" | tee -a "$LOG"; done
plan_total="$(awk -v t="$(printf '%s\n' "${PLAN_MINUTES[@]}" | awk '{ s += $1 } END { print s }')" 'BEGIN { printf "%.1f", t + 6.3 }')"
log "plan: plus 6.3 min outside the script (c5: create 0.3, upload 5.4, pull and delete 0.6): a session of $plan_total min"
if (( PREFILL )) && ! awk -v t="$plan_total" -v s="$PREFILL_SESSION_MIN" 'BEGIN { exit !(t + 0 <= s + 0) }'; then
  log "plan: the session of $plan_total min is over the $PREFILL_SESSION_MIN minutes of --prefill"
  PLAN_FITS=0
fi
if (( DRY && ! PLAN_FITS )); then echo "t04a.sh: the plan is over CAP_MIN=$CAP_MIN" >&2; exit 2; fi
# --prefill: the VRAM arithmetic of both configurations (prefill_fixed_gib above) and the hot set; the KV cache
# and the other compute buffers at the larger ctx and ubatch are not in it, PREFILL_RESERVE_GIB is kept for them
if (( PREFILL )); then
  log "prefill VRAM, configuration A (ctx $PREFILL_CTX_A x ubatch $PREFILL_UB_A, the 4K, 8K and 16K prompts): ${PREFILL_FIXED_A#* }"
  log "prefill VRAM, configuration B (ctx $PREFILL_CTX_B x ubatch $PREFILL_UB_B, the 32K prompt): ${PREFILL_FIXED_B#* }"
  VRAM_EST="$(awk -v w="$PREFILL_WORSE_GIB" -v h="$PREFILL_VRAM" -v r="$PREFILL_RESERVE_GIB" 'BEGIN {
    m = 79.25 - w - h
    printf "%.2f (configuration the worse) + hot set %d = %.2f of 79.25 GiB, margin %.2f for the KV cache and the other compute buffers (reserve %g)", w, h, w + h, m, r }')"
  VRAM_EST="${VRAM_EST/the worse/$PREFILL_WORSE, the worse}"
  log "prefill VRAM estimate (hook servers, one hot set of $PREFILL_VRAM GiB, $PREFILL_VRAM_SOURCE): $VRAM_EST"
  if awk -v w="$PREFILL_WORSE_GIB" -v h="$PREFILL_VRAM" 'BEGIN { exit !(79.25 - w - h < 2) }'; then
    echo "t04a.sh: --prefill: the estimate leaves under 2 GiB of VRAM for the unestimated buffers; lower PREFILL_VRAM" >&2
    exit 2
  fi
fi
if (( DRY )); then
  NODE="<GPU NUMA node>"; R="/root/llama-dsv41-<commit8>-cu128-sm80-89"; TGZ="$PAYLOAD/llama-dsv41-<commit8>-cu128-sm80-89.tar.gz"
  echo "+ [env] nvidia-smi; nvidia-smi topo -m; free -g; cgroup memory files; lscpu; df -h /root /workspace; nproc"
else
  step_start env
  {
    echo "== nvidia-smi"; show_cmd nvidia-smi
    echo "== nvidia-smi topo -m"; show_cmd nvidia-smi topo -m
    echo "== free -g"; show_cmd free -g
    echo "== cgroup"; cgroup_memory
    echo "== lscpu"; show_cmd lscpu
    echo "== df"; show_cmd df -h /root /workspace
    echo "== nproc"; show_cmd nproc
    echo "== RLIMIT_MEMLOCK"; memlock_line
    echo "== NUMA MemFree"; numa_free_line
  } > "$OUT/env.txt" 2>&1 || true
  step_end env 0
  # the GPU's NUMA node, as the stock reference run reads it (node 1 on the oracle dump pod, 0 on the first t04a one; never assumed)
  NODE="$(nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/{print $(NF-1)}')" || true
  [[ "$NODE" =~ ^[0-9]+$ ]] || { log "cannot read the GPU NUMA node from nvidia-smi topo -m (got '$NODE')"; exit 1; }
  log "GPU NUMA node $NODE"
fi

# the NUMA gate (--speed, --prefill): the GPU's node needs room for the warm tier and the server's own memory
if (( SPEED || PREFILL )); then
  if (( DRY )); then
    echo "+ [numa-gate] MemFree of node <GPU NUMA node> ($NUMA_ROOT/node<N>/meminfo) >= $NUMA_NEED_GIB GiB (warm tier $WARM_MAX GiB + 8), else exit 3"
  else
    free_kib="$(numa_free_kib "$NODE")"
    verdict "numa: GPU NUMA node $NODE; $(numa_free_line); $(memlock_line)"
    if [[ "$free_kib" == absent ]] || (( free_kib < NUMA_NEED_GIB * 1048576 )); then
      verdict "health: FAIL, the GPU's NUMA node $NODE has $( [[ "$free_kib" == absent ]] && echo "no readable MemFree" ||
        awk -v k="$free_kib" 'BEGIN { printf "%.2f GiB", k / 1048576 }') free, less than the largest warm tier of RAM_RUNS ($WARM_MAX GiB) plus 8 GiB = $NUMA_NEED_GIB GiB: the warm tier and the server's memory would land on the other node or not fit"
      exit 3
    fi
    verdict "health: PASS, the GPU's NUMA node $NODE has $(awk -v k="$free_kib" 'BEGIN { printf "%.2f", k / 1048576 }') GiB free >= $NUMA_NEED_GIB GiB (warm tier $WARM_MAX + 8)"
  fi
fi

# RAM_RUNS auto: the warm tier from the cgroup's limit, the working set and the GPU node's MemFree at the gate
AUTO_GIB=auto
(( DRY )) && AUTO_GIB="<auto>"
if (( (SPEED || PREFILL) && RAM_AUTO )); then
  PROPOSE=(python3 "$TOOLS/oracle/hot_set.py" --propose-ram-budget --working-set-gib "$WORKING_SET_GIB" --margin-gib "$MARGIN_GIB")
  if (( DRY )); then
    echo "+ [ram-auto] $(printf '%q ' "${PROPOSE[@]}")--cgroup-limit-bytes <the memory cgroup's limit> --numa-free-kib <the GPU node's MemFree>"
  else
    limit="$(cgroup_limit_bytes)"
    if [[ ! "$limit" =~ ^[0-9]+$ ]]; then
      verdict "health: FAIL, RAM_RUNS auto sizes the warm tier from the memory cgroup's limit, which is $limit"
      exit 3
    fi
    if ! run ram-auto "${PROPOSE[@]}" --cgroup-limit-bytes "$limit" --numa-free-kib "$free_kib"; then
      verdict "health: FAIL, no warm tier fits: $(tr '\n' ' ' < "$OUT/ram-auto.log")"
      exit 3
    fi
    AUTO_GIB="$(sed -n 's/^proposed_ram_budget_gib //p' "$OUT/ram-auto.log")"
    verdict "ram-auto: $AUTO_GIB GiB warm tier ($(grep -v '^proposed_ram_budget_gib' "$OUT/ram-auto.log" | tr '\n' ';' | sed 's/;$//'))"
  fi
fi
# the residency runs with the auto size filled in
RAM_RUN_PAIRS=()
for p in "${RAM_PAIRS[@]}"; do
  r="${p##*:}"
  [[ "$r" == auto ]] && r="$AUTO_GIB"
  RAM_RUN_PAIRS+=("${p%%:*}:$r")
done

run pick-flavor bash "$TOOLS/pick-flavor.sh" --fork-only "$PAYLOAD" || { log "no flavor for this pod"; exit 1; }
if (( ! DRY )); then
  TGZ="$PAYLOAD/$(sed -n 's/^fork_tarball=//p' "$OUT/pick-flavor.log")"
  R="/root/$(basename "$TGZ" .tar.gz)"
fi
run unpack tar -xzf "$TGZ" -C /root || { log "cannot unpack $TGZ"; exit 1; }
# shellcheck disable=SC2016  # the bash -c scripts take their arguments as $1, $2
run manifest-sha256 bash -c 'cd "$1" && sed -n "/^sha256:/,\$p" MANIFEST | tail -n +2 | sha256sum -c --quiet' _ "$R" \
  || { log "tarball MANIFEST sha256 mismatch"; exit 1; }
run numactl bash -c 'command -v numactl || { apt-get update -qq && apt-get install -y -qq numactl; }' \
  || { log "no numactl"; exit 1; }

# 4 dd streams per shard, then every shard's size against the volume (pod/model-copy.sh)
run model-copy bash "$TOOLS/pod/model-copy.sh" "$MODEL_SRC" "$MODEL_DIR" || { log "model copy failed"; exit 1; }
snapshot setup

# ---------------------------------------------------------------- T06: --prefill, TTFT of 4K-32K prompts
# PREFILL_PY PROMPT_FILE MAX_TOKENS OUT_JSON: one /completion request, raw text, no prompt cache, temperature 0,
# ignore_eos (MAX_TOKENS exactly); writes the server's answer to OUT_JSON and prints "prompt_n prompt_ms
# prompt_per_second predicted_n client_s" (client_s: from sending the request to the whole answer, the TTFT when
# MAX_TOKENS is 1)
# shellcheck disable=SC2016  # python source, expanded by nobody
PREFILL_PY='import json, sys, time, urllib.request
prompt, n, out, port, timeout = open(sys.argv[1]).read(), int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5])
body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False, "ignore_eos": True,
        "return_tokens": True, "stream": False}
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
d = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
client_s = time.time() - t0
json.dump(d, open(out, "w"))
t = d["timings"]
print(t["prompt_n"], round(t["prompt_ms"], 1), round(t["prompt_per_second"], 2), t["predicted_n"], round(client_s, 2))'
# AGREE_PY STOCK_JSON HOOK_JSON: the token agreement of two greedy answers (identical tokens out of the shorter, the
# first divergence)
AGREE_PY='import json, sys
a, b = (json.load(open(f))["tokens"] for f in sys.argv[1:3])
n = min(len(a), len(b))
same = sum(1 for i in range(n) if a[i] == b[i])
first = next((i for i in range(n) if a[i] != b[i]), None)
print(f"{same} of {n} identical, first divergence at token {first if first is not None else \"none\"}")'

# prefill_arm NAME RATE CTX UB PROMPTS ENV_ASSIGNMENTS...: the server with the prefill flags at CTX and UB, then the
# prompts named in PROMPTS (space separated, manifest order; the 4K one with 64 greedy tokens); a prompt whose
# estimate (its tokens at RATE tok/s) would not fit the time left is skipped and recorded
prefill_arm() {
  local name="$1" rate="$2" ctx="$3" ub="$4" want=" $5 "; shift 5
  local flags=(--override-tensor "$CPU_EXPS" -nr -lm mmap --n-gpu-layers 99 --override-tensor 'engram_embd\.weight=CPU'
               --flash-attn on --ctx-size "$ctx" --batch-size "$ub" --ubatch-size "$ub"
               --threads "$THREADS" --threads-batch 32 --numa numactl -np 1)
  server_start "$name" "$@" -- "${flags[@]}" || { verdict "$name: FAIL, the server did not start (see $name.log)"; return 0; }
  local i
  for i in "${!PROMPT_NAMES[@]}"; do
    local pn="${PROMPT_NAMES[$i]}" tokens="${PROMPT_TOKENS[$i]}" n_out=1
    [[ "$want" == *" $pn "* ]] || continue
    [[ "$pn" == 4k ]] && n_out=64
    local need=$(( tokens / rate + 60 ))
    if (( DRY )); then
      echo "+ [$name-$pn] python3 -c <PREFILL_PY> $PREFILL_PROMPTS/${PROMPT_FILES[$i]} $n_out $OUT/$name-$pn.json $PORT <timeout>  # $tokens tokens, n_predict $n_out"
      continue
    fi
    if (( $(left) < need + 30 )); then
      log "$name-$pn: skipped, needs about $need s at $rate tok/s and $(left) s are left"
      printf '%s\t%s\t%s\tskipped-time\n' "$name" "$pn" "$tokens" >> "$OUT/prefill.tsv"
      continue
    fi
    step_start "$name-$pn"
    local r rc=0 timeout=$(( $(left) - 30 ))
    r="$(python3 -c "$PREFILL_PY" "$PREFILL_PROMPTS/${PROMPT_FILES[$i]}" "$n_out" "$OUT/$name-$pn.json" "$PORT" \
         "$timeout" 2>&1)" || rc=$?
    step_end "$name-$pn" "$rc"
    (( rc == 0 )) || r="fail ($(tail -n 1 <<<"$r"))"
    printf '%s\t%s\t%s\t%s\n' "$name" "$pn" "$tokens" "$r" >> "$OUT/prefill.tsv"
    log "$name $pn ($tokens tokens): prompt_n prompt_ms prompt_tok/s predicted_n client_s = $r"
    local prompt_n="${r%% *}"
    if (( rc == 0 )) && [[ "$prompt_n" != "$tokens" ]]; then
      verdict "$name $pn: FAIL, the server evaluated $prompt_n prompt tokens, the manifest counts $tokens"
    fi
    [[ "$pn" == 4k ]] && (( rc == 0 )) && cp "$OUT/$name-$pn.json" "$OUT/$name-4k.tokens.json"
    # no crash: the server still runs and its log has no CUDA error or failed assert
    if ! kill -0 "$SERVER_PID" 2>/dev/null || grep -qaE "CUDA error|GGML_ASSERT|GGML_ABORT" "$OUT/$name.log"; then
      verdict "$name $pn: FAIL, the server crashed: $(grep -aE "CUDA error|GGML_ASSERT|GGML_ABORT|error" "$OUT/$name.log" | tail -n 2 | tr '\n' ' ')"
      server_stop "$name"
      return 0
    fi
    if (( rc == 0 )); then
      local ttft; ttft="$(awk '{ print $5 }' <<<"$r")"
      verdict "$name $pn: no crash, TTFT $ttft s for $tokens tokens (client, n_predict $n_out), $(awk '{ print $3 }' <<<"$r") tok/s prefill (server), ctx $ctx x ubatch $ub"
      # the bar is the client's time to the first token, which includes the server's own overhead
      if [[ "$name" == hook-prefill-a && "$pn" == 16k ]]; then
        local eval_s; eval_s="$(awk '{ printf "%.1f", $2 / 1000 }' <<<"$r")"
        if awk -v t="$ttft" 'BEGIN { exit !(t + 0 <= 60) }'; then
          verdict "goal 3: PASS, the hook's 16K prompt reached its first token in $ttft s (<= 60 s; prompt evaluation $eval_s s)"
        else
          verdict "goal 3: FAIL, the hook's 16K prompt reached its first token in $ttft s (> 60 s; prompt evaluation $eval_s s)"
        fi
      fi
    fi
  done
  server_stop "$name"
}

if (( PREFILL )); then
  # the check reads files only, so a dry run runs it too: the prompts' counts are part of the plan
  if (( DRY )); then
    echo "+ [prompts] python3 $TOOLS/pod/prefill_prompts.py check $PREFILL_PROMPTS"
    python3 "$TOOLS/pod/prefill_prompts.py" check "$PREFILL_PROMPTS" ||
      { echo "t04a.sh: the prefill prompts in $PREFILL_PROMPTS fail their check" >&2; exit 2; }
  else
    run prompts python3 "$TOOLS/pod/prefill_prompts.py" check "$PREFILL_PROMPTS" ||
      { log "the prefill prompts in $PREFILL_PROMPTS fail their check: $(tr '\n' ' ' < "$OUT/prompts.log")"; exit 1; }
  fi
  # the prompts, in manifest order
  PROMPT_NAMES=() PROMPT_TOKENS=() PROMPT_FILES=()
  while read -r pn pt pf; do PROMPT_NAMES+=("$pn"); PROMPT_TOKENS+=("$pt"); PROMPT_FILES+=("$pf"); done < <(
    python3 -c 'import json, sys
for p in json.load(open(sys.argv[1]))["prompts"]: print(p["name"], p["tokens"], p["file"])' "$PREFILL_PROMPTS/manifest.json")
  # each prompt and its answer must fit its configuration's context
  for i in "${!PROMPT_NAMES[@]}"; do
    pn="${PROMPT_NAMES[$i]}" c="$PREFILL_CTX_A" n_out=1
    [[ "$pn" == 32k ]] && c="$PREFILL_CTX_B"
    [[ "$pn" == 4k ]] && n_out=64
    if (( PROMPT_TOKENS[i] + n_out > c )); then
      echo "t04a.sh: --prefill: the $pn prompt (${PROMPT_TOKENS[$i]} tokens + $n_out) does not fit its ctx $c" >&2
      exit 2
    fi
  done
  v="$PREFILL_VRAM" r="${RAM_RUN_PAIRS[0]##*:}"
  run "hot-set-$v-ram$r" python3 "$TOOLS/oracle/hot_set.py" --counts "$COUNTS" --gguf "$MODEL" --budget-gib "$v" \
    --ram-budget-gib "$r" -o "$OUT/hot-$v-ram$r.json" || { log "hot_set.py failed for $v GiB + $r GiB RAM"; exit 1; }
  HOOK_ENV=(LLAMA_STRATA_MOE=1 STRATA_MOE_HOT="$OUT/hot-$v-ram$r.json" STRATA_MOE_VRAM_GB="$v" STRATA_MOE_RAM_GB="$r"
            "${RAM_MODE_ENV[@]}" "${COLD_ENV[@]}" STRATA_MOE_PROFILE=1 STRATA_MOE_CPU=fast)
  prefill_arm hook-prefill-a 100 "$PREFILL_CTX_A" "$PREFILL_UB_A" "4k 8k 16k" "${HOOK_ENV[@]}"
  (( DRY )) || { ram_mode_verdict hook-prefill-a; cold_path_verdict hook-prefill-a; }
  ubatch_verdicts hook-prefill-a "$PREFILL_UB_A" "$OUT/hook-prefill-a.log"
  A_STAGING_BOUND="${STAGING_BOUND:-unknown}"
  prefill_arm hook-prefill-b 100 "$PREFILL_CTX_B" "$PREFILL_UB_B" "32k" "${HOOK_ENV[@]}"
  ubatch_verdicts hook-prefill-b "$PREFILL_UB_B" "$OUT/hook-prefill-b.log"
  prefill_arm stock-prefill-a 45 "$PREFILL_CTX_A" "$PREFILL_UB_A" "4k 8k 16k"
  ubatch_verdicts stock-prefill-a "$PREFILL_UB_A"
  if (( PREFILL_STOCK_B )); then
    prefill_arm stock-prefill-b 45 "$PREFILL_CTX_B" "$PREFILL_UB_B" "32k"
    ubatch_verdicts stock-prefill-b "$PREFILL_UB_B"
  else
    (( DRY )) && echo "+ [stock-prefill-b] dropped by the plan"
    (( DRY )) || verdict "stock-prefill-b: not run, the plan dropped it to fit the $PREFILL_SESSION_MIN-minute session"
  fi
  if (( DRY )); then
    echo "+ [hook-prefill-a-direct] only when hook-prefill-a is staging-bound and $PREFILL_DIRECT_MIN min are left:"
    prefill_arm hook-prefill-a-direct 100 "$PREFILL_CTX_A" "$PREFILL_UB_A" "16k" "${HOOK_ENV[@]}" STRATA_MOE_PREFILL_COLD=direct
  else
    direct_s="$(awk -v m="$PREFILL_DIRECT_MIN" 'BEGIN { printf "%d", m * 60 }')"
    if [[ "$A_STAGING_BOUND" != yes ]]; then
      verdict "hook-prefill-a-direct: not run, hook-prefill-a's profile is not staging-bound (staging-bound: $A_STAGING_BOUND)"
    elif (( $(left) < direct_s )); then
      verdict "hook-prefill-a-direct: not run, hook-prefill-a is staging-bound but it needs about $direct_s s and $(left) s are left"
    else
      prefill_arm hook-prefill-a-direct 100 "$PREFILL_CTX_A" "$PREFILL_UB_A" "16k" "${HOOK_ENV[@]}" STRATA_MOE_PREFILL_COLD=direct
      grep -a '^strata-moe: STRATA_MOE_PREFILL_COLD=' "$OUT/hook-prefill-a-direct.log" | while IFS= read -r line; do verdict "hook-prefill-a-direct: $line"; done
      ubatch_verdicts hook-prefill-a-direct "$PREFILL_UB_A" "$OUT/hook-prefill-a-direct.log"
    fi
  fi
  if (( ! DRY )); then
    if [[ -s "$OUT/stock-prefill-a-4k.tokens.json" && -s "$OUT/hook-prefill-a-4k.tokens.json" ]]; then
      verdict "greedy 4K: $(python3 -c "$AGREE_PY" "$OUT/stock-prefill-a-4k.tokens.json" "$OUT/hook-prefill-a-4k.tokens.json") (hook against stock, same flags, configuration A)"
    else
      verdict "greedy 4K: no token agreement, an arm has no 4K answer"
    fi
  else
    echo "+ [agreement] python3 -c <AGREE_PY> $OUT/stock-prefill-a-4k.tokens.json $OUT/hook-prefill-a-4k.tokens.json"
  fi
  snapshot end
  if (( ! DRY )); then
    step_start pack
    rc=0
    tar -C "$(dirname "$OUT")" -cf "$OUT.tar" "$(basename "$OUT")" || rc=$?
    step_end pack "$rc"
    log "done: $OUT.tar, TTFT in $OUT/prefill.tsv, verdicts in $OUT/verdicts.txt"
  fi
  exit 0
fi

# ---------------------------------------------------------------- run-1's placement and the hot sets
# (--speed needs neither the placement nor the oracle hot set, only the budget hot sets)
if (( ! SPEED )); then
  run placement python3 "$TOOLS/oracle/read_dump.py" --expert-placement "$ORACLE" || { log "no expert placement in run-1"; exit 1; }
  if (( DRY )); then
    ORACLE_LAYERS="<A-B>"; ORACLE_OT=(--override-tensor "<run-1 expert override 1>" --override-tensor "<... N>")
  else
    ORACLE_LAYERS="$(sed -n 's/^gpu_layers //p' "$OUT/placement.log")"
    ORACLE_OT=()
    while read -r ot; do ORACLE_OT+=(--override-tensor "$ot"); done < <(sed -n 's/^override //p' "$OUT/placement.log")
    log "run-1 expert placement: layers $ORACLE_LAYERS on CUDA, ${#ORACLE_OT[@]} override flags"
  fi
  # stock as run-1 placed the experts, with the hook's --no-repack
  STOCK_ORACLE_NR=("${ORACLE_OT[@]}" -nr "${COMMON[@]}")

  run hot-set-oracle python3 "$TOOLS/oracle/hot_set.py" --dump "$ORACLE" --gguf "$MODEL" --budget-gib "$ORACLE_BUDGET" \
    --layers-full "$ORACLE_LAYERS" -o "$OUT/hot-oracle.json" || { log "hot_set.py failed for the oracle placement"; exit 1; }
fi

# build_hot_sets SOURCE...: the hot sets of BUDGETS and of the residency runs from hot_set.py's sources (--dump
# run-1, or --counts the profile's file); returns non-zero at the first that fails
build_hot_sets() {
  local b p v r
  for b in "${BUDGETS[@]}"; do
    run "hot-set-$b" python3 "$TOOLS/oracle/hot_set.py" "$@" --gguf "$MODEL" --budget-gib "$b" \
      -o "$OUT/hot-$b.json" || { log "hot_set.py failed for $b GiB"; return 1; }
  done
  (( SPEED )) || return 0
  for p in "${RAM_RUN_PAIRS[@]}"; do
    v="${p%%:*}" r="${p##*:}"
    run "hot-set-$v-ram$r" python3 "$TOOLS/oracle/hot_set.py" "$@" --gguf "$MODEL" --budget-gib "$v" \
      --ram-budget-gib "$r" -o "$OUT/hot-$v-ram$r.json" || { log "hot_set.py failed for $v GiB + $r GiB RAM"; return 1; }
  done
}
if (( PROFILE )); then
  # the profiling server's VRAM tier, from run-1; the speed runs' hot sets come from its counts, after the baseline
  run hot-set-profile python3 "$TOOLS/oracle/hot_set.py" --dump "$ORACLE" --gguf "$MODEL" --budget-gib "$PROFILE_VRAM" \
    -o "$OUT/hot-profile.json" || { log "hot_set.py failed for the profiling server's $PROFILE_VRAM GiB"; exit 1; }
else
  build_hot_sets --dump "$ORACLE" || exit 1
fi

if (( ! SPEED )); then
  # ---------------------------------------------------------------- parity: replay with run-1's placement
  if fits replay-oracle 15; then
    if replay replay-oracle 1e-6 STRATA_MOE_HOT="$OUT/hot-oracle.json" STRATA_MOE_VRAM_GB="$ORACLE_BUDGET"; then
      verdict "replay-oracle: PASS, max rel L2 <= 1e-6 with run-1's placement"
    else
      verdict "replay-oracle: FAIL (bound 1e-6, see replay-oracle.log)"
    fi
  fi

  # ---------------------------------------------------------------- 64 greedy tokens, hook vs stock, same placement
  DUMP=(-f "$TOOLS/oracle/prompt.txt" -n 64 -s 42)
  if fits greedy 25; then
    vram_start greedy-stock
    run greedy-stock env -u LD_LIBRARY_PATH numactl --cpunodebind="$NODE" "$R/bin/dsv41-moe-dump" \
      --dump-dir "$OUT/dump-stock-oracle-nr" -m "$MODEL" "${DUMP[@]}" "${STOCK_ORACLE_NR[@]}" || true
    vram_stop greedy-stock
    vram_start greedy-hook
    run greedy-hook env -u LD_LIBRARY_PATH LLAMA_STRATA_MOE=1 STRATA_MOE_HOT="$OUT/hot-oracle.json" \
      STRATA_MOE_VRAM_GB="$ORACLE_BUDGET" STRATA_MOE_PROFILE=1 numactl --cpunodebind="$NODE" "$R/bin/dsv41-moe-dump" \
      --dump-dir "$OUT/dump-hook-oracle" -m "$MODEL" "${DUMP[@]}" "${HOSTEXPS[@]}" || true
    vram_stop greedy-hook
    if run compare-hook-vs-stock python3 "$TOOLS/oracle/read_dump.py" --tokens-must-match --compare \
         "$OUT/dump-stock-oracle-nr" "$OUT/dump-hook-oracle"; then
      verdict "greedy: PASS, the 64 tokens of the hook equal stock's with the same placement"
    else
      verdict "greedy: FAIL (see compare-hook-vs-stock.log)"
    fi
    run compare-stock-vs-oracle python3 "$TOOLS/oracle/read_dump.py" --compare "$ORACLE" "$OUT/dump-stock-oracle-nr" || true
    run compare-hook-vs-oracle python3 "$TOOLS/oracle/read_dump.py" --compare "$ORACLE" "$OUT/dump-hook-oracle" || true
    snapshot greedy
  fi
fi

# ---------------------------------------------------------------- stock baseline, same pod
if (( ! SPEED )); then
  if fits baseline 20 && server_start baseline -- "${STOCK14[@]}"; then
    bench_all baseline
    server_stop baseline
  fi
else
  # the health gate: the stock 120-token decode must reach HEALTH_MIN tok/s on this host
  (( DRY )) || verdict "health: uptime $(uptime); nproc $(nproc); loadavg $(cat /proc/loadavg); $(memlock_line); $(numa_free_line)"
  if fits baseline 20 && server_start baseline -- "${STOCK14[@]}"; then
    bench baseline 120 warm
    bench baseline 120 measured
    if (( ! DRY )); then
      tps="$(awk -F'\t' '$1 == "baseline" && $2 == "120" && $3 == "measured" { split($4, a, " "); print a[2] }' "$OUT/bench.tsv")"
      verdict "health: loadavg $(cat /proc/loadavg); GPU util $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader || echo absent)"
      if ! awk -v t="$tps" -v m="$HEALTH_MIN" 'BEGIN { exit !(t + 0 >= m + 0) }'; then
        verdict "health: FAIL, stock 120-token decode ${tps:-none} tok/s < $HEALTH_MIN"
        server_stop baseline
        exit 3
      fi
      verdict "health: PASS, stock 120-token decode $tps tok/s >= $HEALTH_MIN"
    fi
    bench baseline 480 warm
    bench baseline 480 measured
    server_stop baseline
  elif (( ! DRY )); then
    verdict "health: FAIL, the stock baseline did not run"
    exit 3
  fi
fi

# ---------------------------------------------------------------- T04d: the serving profile and its hot sets
if (( PROFILE )); then
  profile_ok=0
  if fits profile 12 && server_start profile LLAMA_STRATA_MOE=1 STRATA_MOE_HOT="$OUT/hot-profile.json" \
       STRATA_MOE_VRAM_GB="$PROFILE_VRAM" STRATA_MOE_PROFILE=1 STRATA_MOE_CPU=fast STRATA_MOE_COUNTS="$OUT/counts.json" \
       -- "${HOSTEXPS[@]}"; then
    for f in "${PROMPTS[@]}"; do profile_request "$f"; done
    server_stop profile
    (( DRY )) || [[ ! -s "$OUT/counts.json" ]] || profile_ok=1
  fi
  if (( DRY )); then
    build_hot_sets --counts "$OUT/counts.json"
  elif (( profile_ok )) && build_hot_sets --counts "$OUT/counts.json"; then
    verdict "profile: PASS, $(grep -vc 'fail (' "$OUT/profile.tsv") of ${#PROMPTS[@]} prompts answered; counts.json: $(python3 -c 'import json, sys
d = json.load(open(sys.argv[1]))
print(sum(l["decode_calls"] for l in d["layers"]), "decode calls over", len(d["layers"]), "layers")' "$OUT/counts.json"); the hot sets built from it"
  else
    verdict "profile: FAIL (server, prompts or counts.json: see profile.log, profile.tsv, hot-set-*.log); the hot sets from run-1 instead"
    build_hot_sets --dump "$ORACLE" || exit 1
  fi
fi

# ---------------------------------------------------------------- decode with the hook per hot-set size
for b in "${BUDGETS[@]}"; do
  if (( ! SPEED )); then
    if fits "hook-$b" 15 && server_start "hook-$b" LLAMA_STRATA_MOE=1 STRATA_MOE_HOT="$OUT/hot-$b.json" \
         STRATA_MOE_VRAM_GB="$b" STRATA_MOE_PROFILE=1 -- "${HOSTEXPS[@]}"; then
      bench_all "hook-$b"
      server_stop "hook-$b"
    fi
    continue
  fi
  for k in $CPU_KERNELS; do
    if fits "hook-$b-$k" 15 && server_start "hook-$b-$k" LLAMA_STRATA_MOE=1 STRATA_MOE_HOT="$OUT/hot-$b.json" \
         STRATA_MOE_VRAM_GB="$b" STRATA_MOE_PROFILE=1 STRATA_MOE_CPU="$k" -- "${HOSTEXPS[@]}"; then
      bench_all "hook-$b-$k"
      server_stop "hook-$b-$k"
    fi
  done
done

# ---------------------------------------------------------------- T04c: the residency, VRAM + RAM + O_DIRECT
if (( SPEED )); then
  for p in "${RAM_RUN_PAIRS[@]}"; do
    v="${p%%:*}" r="${p##*:}"
    fits "hook-$v-ram$r" 15 || continue
    if server_start "hook-$v-ram$r" LLAMA_STRATA_MOE=1 STRATA_MOE_HOT="$OUT/hot-$v-ram$r.json" STRATA_MOE_VRAM_GB="$v" \
         STRATA_MOE_RAM_GB="$r" "${RAM_MODE_ENV[@]}" "${COLD_ENV[@]}" STRATA_MOE_PROFILE=1 STRATA_MOE_CPU=fast \
         -- "${HOSTEXPS[@]}"; then
      ram_mode_verdict "hook-$v-ram$r"
      cold_path_verdict "hook-$v-ram$r"
      bench_all "hook-$v-ram$r"
      server_stop "hook-$v-ram$r"
    else
      ram_mode_verdict "hook-$v-ram$r"
      cold_path_verdict "hook-$v-ram$r"
    fi
  done
fi

# ---------------------------------------------------------------- the cross-backend gaps, information only
# bound 1: these compare a backend with run-1's layers of the other backend, so they report and do not gate
if (( ! SPEED )) && fits replay-gpu-gap 10; then
  replay replay-gpu-gap 1 STRATA_MOE_HOT="$OUT/hot-$INFO_BUDGET.json" STRATA_MOE_VRAM_GB="$INFO_BUDGET" || true
fi
if (( ! SPEED )) && fits replay-cpu-gap 10; then
  replay replay-cpu-gap 1 || true
fi

# ---------------------------------------------------------------- extras, if time is left
if (( EXTRA )) && fits stock-hostexps 15 && server_start stock-hostexps -- "${HOSTEXPS[@]}"; then
  bench_all stock-hostexps
  server_stop stock-hostexps
fi

snapshot end
if (( ! DRY )); then
  step_start pack
  rc=0
  tar -C "$(dirname "$OUT")" -cf "$OUT.tar" --exclude='dump-*' "$(basename "$OUT")" || rc=$?
  step_end pack "$rc"
  log "done: $OUT.tar (dumps stay in $OUT/dump-*), steps in $OUT/steps.log and steps.tsv, decode in $OUT/bench.tsv"
fi
