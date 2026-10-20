#!/usr/bin/env bash
# The OLMoE smoke under memory pressure, without a GPU. The model (4.2 GB) runs inside a
# container whose memory limit (MEM_LIMIT, default 3g, no swap) is smaller than it, as the DeepSeek-V4.1 pods run
# 246 GiB in a 116.4 GiB cgroup. First a profile: the hook with STRATA_MOE_COUNTS on each of the pod's profiling
# prompts (tools/dsv41/pod/prompts/*.txt, none of them the measured prompt), PROFILE_TOKENS (default 128) tokens each,
# one counts file per prompt; hot_set.py --counts merges them into the hot set: no VRAM experts, a RAM_GB warm tier.
# (One 64-token prose prompt alone gave a warm list no better than chance on the measured prompt: 23.6% of its decode
# pairs for 27% of the experts, against 51.8% from the six.) Then the hook runs on the measured prompt (dsv41-moe-dump,
# 16 greedy tokens, -nr, the hook's smoke flags, oracle/prompt.txt), each after the model file is dropped from the page
# cache:
#   mmap-only         the misses read the fork's mmapped experts (STRATA_MOE_RAM_GB unset), as before the residency tiers
#   tiered-direct     the residency: the warm list in RAM_GB GiB of mlock'd RAM (STRATA_MOE_RAM_MODE=locked,
#                     --ulimit memlock=-1), every other expert cold, read with O_DIRECT (STRATA_MOE_COLD=direct)
#   tiered-pagecache  the same warm tier, the cold experts from the fork's mapping through the page cache
#                     (STRATA_MOE_COLD=pagecache, the default)
#   tiered-plain      the warm tier in STRATA_MOE_RAM_MODE=plain (pre-faulted, not locked), cold path unset
#                     (pagecache), under --ulimit memlock=8388608, the RLIMIT_MEMLOCK of the RunPod containers
#   locked-memlock-8m  STRATA_MOE_RAM_MODE unset (no CUDA device here: the default is locked) under the same 8 MiB
#                     limit: it must abort, naming the mode, the warm tier's size and RLIMIT_MEMLOCK
# (pinned, the default with a CUDA device, needs a GPU: the pod run checks it.) Each run's profile
# (STRATA_MOE_PROFILE=1) gives its decode op per forward pass, its major page faults per token and, tiered, the tier
# tables; the container's memory.peak and the wall time are printed too. The dumps of the hook runs are compared with
# stock -nr outside any limit (STOCK_DUMP, the stock dump t03r2-offnr): identical tokens and relative L2 0 in every
# layer are required. A table of the runs follows, and the bar: tiered-pagecache's decode op per pass at most
# 5% over mmap-only's. Its major faults per decode token are printed against mmap-only's without a bar: the page cache
# still serves the cold experts in pagecache mode, so they stay at the mmap level by design.
# Last, the O_DIRECT read baseline of the same file with dd (1 MiB blocks, about one OLMoE expert part): one stream
# and READERS parallel streams (the cold tier's default reader count), 1 GiB each, for the tier table's GB/s.
#
#   tools/dsv41/oracle/residency-pressure-test.sh OUT_DIR
#
# Needs the cu128 build (tools/dsv41/build.sh), docker, python3, and numpy for read_dump.py (PYTHON, default
# $BUILD_ROOT/venv/bin/python). OUT_DIR must not exist. Environment: MEM_LIMIT, RAM_GB (default 1),
# THREADS (default 8), READERS (default 8), BUILD_ROOT, MODEL, STOCK_DUMP, PYTHON, PROFILE_TOKENS. Exits non-zero when
# a run, a compare or the bar fails.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
source "$here/../flavors.sh"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
MODEL="${MODEL:-$BUILD_ROOT/testmodels/olmoe-1b-7b-0924-q4_k_m.gguf}"
STOCK_DUMP="${STOCK_DUMP:-$BUILD_ROOT/oracle-smoke/t03r2-offnr}"
PYTHON="${PYTHON:-$BUILD_ROOT/venv/bin/python}"
MEM_LIMIT="${MEM_LIMIT:-3g}"
RAM_GB="${RAM_GB:-1}"
THREADS="${THREADS:-8}"
READERS="${READERS:-8}"
PROFILE_TOKENS="${PROFILE_TOKENS:-128}"

[[ $# -eq 1 ]] || { sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 2; }
OUT="$1"
[[ ! -e "$OUT" ]] || { echo "residency-pressure-test.sh: $OUT exists" >&2; exit 1; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
flavor_config 12.8
FORK_BIN="$BUILD_ROOT/$FLAVOR_TAG/fork/bin"
image="$(builder_image)"

# run NAME MEMLOCK PROMPT N_TOKENS ENV...: one hook dump of N_TOKENS greedy tokens on PROMPT in the limited container
# with RLIMIT_MEMLOCK MEMLOCK (docker --ulimit memlock, bytes or -1), after dropping the model from the page cache
run() {
  local name="$1" memlock="$2" prompt="$3" n_tok="$4"; shift 4
  python3 -c 'import os, sys; fd = os.open(sys.argv[1], os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)' \
    "$MODEL"
  echo "=== $name: docker --memory $MEM_LIMIT (no swap) --ulimit memlock=$memlock, model dropped from the page cache, $*"
  local t0 rc=0
  t0=$(date +%s.%N)
  docker run --rm --memory "$MEM_LIMIT" --memory-swap "$MEM_LIMIT" --ulimit "memlock=$memlock:$memlock" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e OUT="$OUT" -e NAME="$name" -e MODEL="$MODEL" \
    -e FORK_BIN="$FORK_BIN" -e THREADS="$THREADS" -e PROMPT="$prompt" -e N_TOK="$n_tok" \
    -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$BUILD_ROOT:$BUILD_ROOT" -v "$OUT:$OUT" \
    "$image" env "$@" bash -euo pipefail -c '
      mkdir -p /tmp/cudastub
      ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1
      export LD_LIBRARY_PATH="/tmp/cudastub:$FORK_BIN"
      rc=0
      LLAMA_STRATA_MOE=1 STRATA_MOE_PROFILE=1 "$FORK_BIN/dsv41-moe-dump" --dump-dir "$OUT/$NAME" -m "$MODEL" \
        -f "$PROMPT" -n "$N_TOK" -ngl 0 -c 256 -t "$THREADS" -s 42 -nr > "$OUT/$NAME.log" 2>&1 || rc=$?
      echo "container memory.peak $(cat /sys/fs/cgroup/memory.peak) bytes, memory.max $(cat /sys/fs/cgroup/memory.max)" \
        >> "$OUT/$NAME.log"
      exit $rc
    ' > /dev/null 2>&1 || rc=$?
  echo "exit $rc, wall $(python3 -c "import sys; print(round(float(sys.argv[2]) - float(sys.argv[1]), 1))" "$t0" "$(date +%s.%N)") s"
  grep -E "^strata-moe: (residency|STRATA_MOE)|: strata-moe: STRATA_MOE_RAM_MODE|^generated_token_ids|^major page faults|^all layers|^tiers over|^container" \
    "$OUT/$name.log" || true
  return "$rc"
}

status=0
COUNTS=()
for prompt in "$here"/../pod/prompts/*.txt; do
  name="profile-$(basename "$prompt" .txt)"
  echo "=== $name: the hook with STRATA_MOE_COUNTS, $PROFILE_TOKENS tokens"
  run "$name" -1 "$prompt" "$PROFILE_TOKENS" STRATA_MOE_COUNTS="$OUT/$name.counts.json" > /dev/null || status=1
  COUNTS+=(--counts "$OUT/$name.counts.json")
done
echo "=== hot set: no VRAM experts, a $RAM_GB GiB warm tier, from the merged decode counts of the profile"
python3 "$here/hot_set.py" "${COUNTS[@]}" --gguf "$MODEL" --budget-gib 0 --ram-budget-gib "$RAM_GB" \
  -o "$OUT/hot.json" | tail -n 3

MEASURED=("$here/prompt.txt" 16)
TIERED=(STRATA_MOE_HOT="$OUT/hot.json" STRATA_MOE_RAM_GB="$RAM_GB" STRATA_MOE_COLD_THREADS="$READERS")
run mmap-only -1 "${MEASURED[@]}" || status=1
run tiered-direct -1 "${MEASURED[@]}" "${TIERED[@]}" STRATA_MOE_RAM_MODE=locked STRATA_MOE_COLD=direct || status=1
run tiered-pagecache -1 "${MEASURED[@]}" "${TIERED[@]}" STRATA_MOE_RAM_MODE=locked STRATA_MOE_COLD=pagecache || status=1
run tiered-plain 8388608 "${MEASURED[@]}" "${TIERED[@]}" STRATA_MOE_RAM_MODE=plain || status=1
# the expected failure: the default mode without a GPU is locked, which the 8 MiB limit refuses
rc=0
run locked-memlock-8m 8388608 "${MEASURED[@]}" "${TIERED[@]}" || rc=$?
log="$OUT/locked-memlock-8m.log"
if (( rc != 0 )) && grep -q "^strata-moe: STRATA_MOE_RAM_MODE=locked (the default without a CUDA device)" "$log" &&
   grep -q ": strata-moe: STRATA_MOE_RAM_MODE=locked: mlock of the .* GiB warm tier (.* bytes) failed: .*; RLIMIT_MEMLOCK is 0.008 GiB (8388608 bytes)" "$log"; then
  echo "locked-memlock-8m: aborted as required (exit $rc), naming the mode, the size and RLIMIT_MEMLOCK"
else
  echo "FAIL: locked-memlock-8m: exit $rc, without the default-mode line and the mlock abort naming RLIMIT_MEMLOCK"
  status=1
fi

RUNS=(mmap-only tiered-direct tiered-pagecache tiered-plain)
for name in "${RUNS[@]}"; do
  echo "=== compare: stock -nr without a limit ($STOCK_DUMP) against $name"
  "$PYTHON" "$here/read_dump.py" --tokens-must-match --compare "$STOCK_DUMP" "$OUT/$name" \
    > "$OUT/compare-$name.log" 2>&1 || status=1
  grep -E "^generated token ids|^max relative L2" "$OUT/compare-$name.log" || tail -n 5 "$OUT/compare-$name.log"
  grep -q "^max relative L2 of out over layers: 0.000e+00;" "$OUT/compare-$name.log" ||
    { echo "FAIL: $name: relative L2 of out is not 0 in every layer"; status=1; }
done
echo "=== the runs under --memory $MEM_LIMIT (decode: 1-token calls; per pass: the sum of the layer means)"
# shellcheck disable=SC2016  # python source
TABLE_PY='import re, sys
out, runs = sys.argv[1], sys.argv[2:]
PASS_OP = r"per forward pass \(sum of the layer means\): submit \S+ gpu \S+ cpu \S+ sync \S+ wait \S+ op ([0-9.]+) ms"
ROW = "%-17s %24s %10s %12s %17s %17s %6s %6s"
print(ROW % ("run", "tokens vs stock", "max rel L2", "decode op ms", "majflt/decode tok", "majflt/prompt tok", "warm %",
             "cold %"))
rows = {}
for r in runs:
    log = open("%s/%s.log" % (out, r)).read()
    cmp = open("%s/compare-%s.log" % (out, r)).read()
    ops = re.findall(PASS_OP, log)  # decode first, then batch
    faults = re.findall(r"^major page faults: \d+ over \d+ tokens, ([0-9.]+) per token", log, re.M)
    tiers = re.findall(r"^tiers over all layers: .*\(([0-9.]+) / ([0-9.]+) / ([0-9.]+)%\)", log, re.M)
    tok = re.findall(r"^generated token ids: (.*)$", cmp, re.M)
    l2 = re.findall(r"^max relative L2 of out over layers: ([0-9.e+-]+);", cmp, re.M)
    rows[r] = (float(ops[0]) if ops else None, float(faults[0]) if faults else None)
    print(ROW % (r, tok[0] if tok else "-", l2[0] if l2 else "-", ops[0] if ops else "-", faults[0] if faults else "-",
                 faults[1] if len(faults) > 1 else "-", tiers[0][1] if tiers else "-", tiers[0][2] if tiers else "-"))
(m_op, m_f), (p_op, p_f) = rows["mmap-only"], rows["tiered-pagecache"]
if None in (m_op, m_f, p_op, p_f):
    sys.exit("decode bar: FAIL, a profile line is missing")
ok = p_op <= 1.05 * m_op
print("decode bar: tiered-pagecache decode op %.3f ms per pass against mmap-only %.3f (%+.1f%%, at most +5%%): %s; "
      "major faults per decode token %g against %g (reported, no bar)" % (p_op, m_op, 100.0 * (p_op / m_op - 1.0),
                                                                          "PASS" if ok else "FAIL", p_f, m_f))
sys.exit(0 if ok else 1)'
python3 -c "$TABLE_PY" "$OUT" "${RUNS[@]}" || status=1

# dd_streams N: N parallel O_DIRECT dd streams of 1 GiB / N each at spread offsets of the model; prints GB/s
dd_streams() {
  local n="$1" per=$(( 1024 / $1 )) i t0 t1
  t0=$(date +%s.%N)
  for (( i = 0; i < n; i++ )); do
    dd if="$MODEL" of=/dev/null bs=1M count="$per" skip=$(( i * 3800 / n )) iflag=direct status=none &
  done
  wait
  t1=$(date +%s.%N)
  python3 -c "import sys; print(f'{sys.argv[1]} streams: {1024 * 2**20 / (float(sys.argv[3]) - float(sys.argv[2])) / 1e9:.2f} GB/s')" \
    "$n" "$t0" "$t1"
}
echo "=== O_DIRECT read baseline of $MODEL with dd, 1 MiB blocks, 1 GiB per run (load average $(cut -d' ' -f1-3 /proc/loadavg))"
dd_streams 1
dd_streams "$READERS"
echo "residency-pressure-test.sh: $( (( status )) && echo FAIL || echo PASS ), logs in $OUT"
exit "$status"
