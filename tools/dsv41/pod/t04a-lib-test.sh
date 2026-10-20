#!/usr/bin/env bash
# Tests tools/dsv41/pod/t04a-lib.sh without a pod: snapshot() on a fake cgroup v1 tree (no memory.peak, the T04a
# pod's layout), a fake cgroup v2 tree and an empty one, with nvidia-smi failing and missing, under t04a.sh's
# `set -Eeuo pipefail`; run() on a failing command; and that a script dying on an unhandled error still writes
# its exit status and last step; majflt and fault_rate (the major page faults per second of the T04c runs); the
# NUMA free memory, RLIMIT_MEMLOCK and RAM mode lines of the T04c gate and runs; the T04d cold path line, the cgroup
# limit the warm tier sizing reads and the session plan; the T06 dense against expert time per ubatch and the
# staging-bound word of --prefill (ubatch_verdicts). Exits 1 if any check fails.
#
#   bash tools/dsv41/pod/t04a-lib-test.sh
set -Eeuo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
failed=0
check() {
  if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; failed=$((failed + 1)); fi
}

# fake cgroup trees
mkdir -p "$tmp/v1/memory" "$tmp/v2" "$tmp/none"
echo 124999999488 > "$tmp/v1/memory/memory.limit_in_bytes"
echo 123800000000 > "$tmp/v1/memory/memory.usage_in_bytes"
echo 124999999488 > "$tmp/v1/memory/memory.max_usage_in_bytes"
echo max > "$tmp/v2/memory.max"
echo 4096 > "$tmp/v2/memory.current"
# a failing nvidia-smi, and a PATH without one (free and the coreutils stay)
mkdir -p "$tmp/bin-fail" "$tmp/bin-none"
printf '#!/bin/sh\necho "NVIDIA-SMI has failed" >&2\nexit 9\n' > "$tmp/bin-fail/nvidia-smi"
chmod +x "$tmp/bin-fail/nvidia-smi"
for tool in cat date tee free; do
  if command -v "$tool" >/dev/null; then ln -s "$(command -v "$tool")" "$tmp/bin-none/$tool"; fi
done

# snapshot_case NAME CGROUP_TREE PATH: snapshot() in a fresh shell with t04a.sh's options and traps
snapshot_case() {
  local name="$1" tree="$2" path="$3" out="$tmp/out-$1"
  mkdir -p "$out"
  local rc=0
  env -i PATH="$path" HOME="$tmp" "$BASH" -c '
    set -Eeuo pipefail
    OUT="$1" DRY=0 START=$(date +%s) DEADLINE=$(( $(date +%s) + 60 )) LOG="$1/t04a.log" CGROUP_ROOT="$2"
    source "$3"
    trap "on_err \$LINENO \"\$BASH_COMMAND\" \$?" ERR
    snapshot "$4"
    echo after-snapshot >> "$1/t04a.log"
  ' _ "$out" "$tree" "$here/t04a-lib.sh" "$name" > "$out/stdout" 2>&1 || rc=$?
  check "snapshot $name: exit 0 and the script goes on (rc $rc)" "(( rc == 0 )) && grep -q after-snapshot '$out/t04a.log'"
}

snapshot_case v1 "$tmp/v1" "$tmp/bin-fail:/usr/bin:/bin"
check "v1: memory.limit_in_bytes read" "grep -q 'memory/memory.limit_in_bytes = 124999999488' '$tmp/out-v1/snapshots.txt'"
check "v1: memory.peak absent, written as absent" "grep -q 'memory.peak: absent' '$tmp/out-v1/snapshots.txt'"
check "v1: the failing nvidia-smi recorded with its exit" "grep -q 'nvidia-smi .*: exit 9' '$tmp/out-v1/snapshots.txt'"
check "v1: start and end in steps.log" \
  "grep -q 'start snapshot-v1' '$tmp/out-v1/steps.log' && grep -q 'end snapshot-v1 exit 0' '$tmp/out-v1/steps.log'"

snapshot_case v2 "$tmp/v2" "$tmp/bin-none"
check "v2: memory.current read" "grep -q 'memory.current = 4096' '$tmp/out-v2/snapshots.txt'"
check "v2: the v1 files absent, written as absent" "grep -q 'memory/memory.limit_in_bytes: absent' '$tmp/out-v2/snapshots.txt'"
check "v2: nvidia-smi missing, written as not installed" "grep -q 'nvidia-smi: not installed' '$tmp/out-v2/snapshots.txt'"

snapshot_case none "$tmp/none" "$tmp/bin-none"
check "no cgroup files: every one absent" "[[ \$(grep -c ': absent' '$tmp/out-none/snapshots.txt') -eq 6 ]]"

# run() with a failing command, and a script that dies on an unhandled error after it
out="$tmp/out-die"
mkdir -p "$out"
rc=0
bash -c '
  set -Eeuo pipefail
  OUT="$1" DRY=0 START=$(date +%s) DEADLINE=$(( $(date +%s) + 60 )) LOG="$1/t04a.log"
  source "$2"
  trap "on_exit \$?" EXIT
  trap "on_err \$LINENO \"\$BASH_COMMAND\" \$?" ERR
  run failing bash -c "exit 3" || true
  step_start doomed
  false
  echo not-reached >> "$1/t04a.log"
' _ "$out" "$here/t04a-lib.sh" > "$out/stdout" 2>&1 || rc=$?
check "run failing: start, end exit 3, steps.tsv row" \
  "grep -q 'start failing' '$out/steps.log' && grep -q 'end failing exit 3' '$out/steps.log' && grep -q '^failing 3 ' '$out/steps.tsv'"
check "unhandled error: the script exits 1 (rc $rc) and stops there" "(( rc == 1 )) && ! grep -q not-reached '$out/t04a.log'"
check "unhandled error: the failing line is logged" "grep -q 'failed (exit 1): false' '$out/t04a.log'"
check "unhandled error: exit status and last step are the last line of steps.log" \
  "tail -n 1 '$out/steps.log' | grep -q 'exit 1 (last step doomed)'"

# majflt and fault_rate: a live process, a gone one, and the rate between two counts
source "$here/t04a-lib.sh"
n="$(majflt $$)"
check "majflt of this shell: a count ($n)" "[[ '$n' =~ ^[0-9]+$ ]]"
check "majflt of a process that is gone: absent" "[[ \$(majflt 999999999) == absent ]]"
check "fault_rate 100 1600 10 12.5: 1500 faults, 600 per second" "[[ \"\$(fault_rate 100 1600 10 12.5)\" == 'majflt 1500 majflt/s 600' ]]"
check "fault_rate with an absent count: absent" "[[ \"\$(fault_rate absent 5 1 2)\" == 'majflt absent' ]]"

# numa_free_kib, numa_free_line, memlock_line, ram_mode_verdict: a fake two-node tree, a missing node, a log
mkdir -p "$tmp/numa/node0" "$tmp/numa/node1" "$tmp/out-ram"
printf 'Node 0 MemTotal:       263904844 kB\nNode 0 MemFree:        198000000 kB\n' > "$tmp/numa/node0/meminfo"
printf 'Node 1 MemTotal:       264225776 kB\nNode 1 MemFree:           654321 kB\n' > "$tmp/numa/node1/meminfo"
NUMA_ROOT="$tmp/numa"
check "numa_free_kib 1: MemFree of node 1 (654321)" "[[ \$(numa_free_kib 1) == 654321 ]]"
check "numa_free_kib 7, no such node: absent" "[[ \$(numa_free_kib 7) == absent ]]"
check "numa_free_line: both nodes in GiB" \
  "[[ \"\$(numa_free_line)\" == 'node 0 MemFree 188.83 GiB; node 1 MemFree 0.62 GiB' ]]"
check "memlock_line: soft, hard and CapEff ($(memlock_line))" "memlock_line | grep -Eq '^RLIMIT_MEMLOCK soft [0-9a-z]+ hard [0-9a-z]+ KiB; CapEff'"
OUT="$tmp/out-ram" DRY=0 START=$(date +%s) LOG="$tmp/out-ram/t04a.log"
printf 'x\nstrata-moe: STRATA_MOE_RAM_MODE=pinned (the default with a CUDA device)\ny\n' > "$OUT/hook-70-ram40.log"
ram_mode_verdict hook-70-ram40
check "ram_mode_verdict: the mode line in verdicts.txt" \
  "grep -q '^hook-70-ram40: RAM mode: strata-moe: STRATA_MOE_RAM_MODE=pinned (the default with a CUDA device)' '$OUT/verdicts.txt'"
: > "$OUT/none.log"
ram_mode_verdict none
check "ram_mode_verdict without a mode line: says so" "grep -q '^none: RAM mode: no STRATA_MOE_RAM_MODE line' '$OUT/verdicts.txt'"

# cold_path_verdict, cgroup_limit_bytes on the fake trees, the plan
printf 'strata-moe: STRATA_MOE_COLD=pagecache (the default)\n' > "$OUT/hook-70-ram78.log"
cold_path_verdict hook-70-ram78
check "cold_path_verdict: the cold path line in verdicts.txt" \
  "grep -q '^hook-70-ram78: cold path: strata-moe: STRATA_MOE_COLD=pagecache (the default)' '$OUT/verdicts.txt'"
cold_path_verdict none
check "cold_path_verdict without a line: says so" "grep -q '^none: cold path: no STRATA_MOE_COLD line' '$OUT/verdicts.txt'"
mkdir -p "$tmp/v1-nolimit/memory"
echo 9223372036854771712 > "$tmp/v1-nolimit/memory/memory.limit_in_bytes"
mkdir -p "$tmp/v2-limit"
echo 3221225472 > "$tmp/v2-limit/memory.max"
check "cgroup_limit_bytes, v1 (the pods): 124999999488" "[[ \$(CGROUP_ROOT='$tmp/v1' cgroup_limit_bytes) == 124999999488 ]]"
check "cgroup_limit_bytes, v1 without a limit: unlimited" "[[ \$(CGROUP_ROOT='$tmp/v1-nolimit' cgroup_limit_bytes) == unlimited ]]"
check "cgroup_limit_bytes, v2 max: unlimited" "[[ \$(CGROUP_ROOT='$tmp/v2' cgroup_limit_bytes) == unlimited ]]"
check "cgroup_limit_bytes, v2 3 GiB: 3221225472" "[[ \$(CGROUP_ROOT='$tmp/v2-limit' cgroup_limit_bytes) == 3221225472 ]]"
check "cgroup_limit_bytes, no files: absent" "[[ \$(CGROUP_ROOT='$tmp/none' cgroup_limit_bytes) == absent ]]"
plan_add setup 4.5
plan_add baseline 8.5
check "plan_print: 13.0 min fits a 20-minute cap" "plan_print 20 | tail -n 1 | grep -q 'total 13.0 min, fits the 20-minute cap'"
check "plan_print: 13.0 min is over a 10-minute cap, returns 1" "! plan_print 10 >/dev/null && { plan_print 10 || true; } | tail -n 1 | grep -q 'over the 10-minute cap'"

# the dense against expert time per ubatch and the staging-bound word, from a hook log in the library's profile
# format and a stock arm without one. The batch-call table holds the server's 2-token warmup (layer 0's first call,
# the slab upload and the warm-tier fill, 120 s) and so a mean op far above the prefill calls'; the split must take
# the prefill line's op, the arm's mean over its prefill calls
mkdir -p "$tmp/out-ub"
OUT="$tmp/out-ub" LOG="$tmp/out-ub/t04a.log"
cat > "$OUT/hook.log" <<'LOGEOF'
strata-moe profile, decode calls (1 token) (per call: mean ms)
all layers: 2600 calls, hit rate 50.0%; per forward pass (sum of the layer means): submit 0.100 gpu 1.000 cpu 2.000 sync 0.100 wait 0.200 op 40.000 ms
strata-moe profile, batch calls (more than 1 token) (per call: mean ms)
    0       6    20480      122880         0  100.0    1.000  225.000    0.000    0.000  225.000 20000.000
all layers: 240 calls, hit rate 100.0%; per forward pass (sum of the layer means): submit 1.000 gpu 9000.000 cpu 0.000 sync 0.000 wait 9000.000 op 24000.000 ms
strata-moe prefill, batch calls on the GPU (per call: mean)
    0      5   20478     4.0    1797.1     770.6    85.00   22.17    50.00   225.00    0.10   200.00     1.00      5    237.50
prefill per ubatch (sum of the layer means): 70.2 GiB copied (30.1 GiB staged) in 3400.0 ms of copy stream (22.17 GB/s), 2000.0 ms of compute, 9000.0 ms of GPU time (overlap 0.10), 8000.0 ms filling the staging on the host, 9500.0 ms of MoE op over 5 op calls per layer (a layer's first call left out)
LOGEOF
printf 'hook-prefill-a\t16k\t16378\t16378 48000.0 341.2 1 48.5\nhook-prefill-a\t4k\t4094\t4094 11000.0 372.2 64 15.1\nhook-prefill-a\t8k\t8178\tskipped-time\nstock-prefill-a\t16k\t16378\t16378 360000.0 45.5 1 361.0\n' > "$OUT/prefill.tsv"
ubatch_verdicts hook-prefill-a 4096 "$OUT/hook.log"
check "ubatch_verdicts: 16K = 4 ubatches, 12000 ms each, no split per prompt" \
  "grep -q '^hook-prefill-a 16k: 4 ubatches of <= 4096 tokens, 12000 ms per ubatch (server prompt ms / ubatches)\$' '$OUT/verdicts.txt'"
check "ubatch_verdicts: 4K = 1 ubatch of 11000 ms" \
  "grep -q '^hook-prefill-a 4k: 1 ubatches of <= 4096 tokens, 11000 ms per ubatch' '$OUT/verdicts.txt'"
check "ubatch_verdicts: the arm, 5 ubatches at 11800 ms, experts 9500 (the prefill op, not the warmup's batch mean), dense 2300" \
  "grep -q \"^hook-prefill-a: 5 ubatches over 2 prompts, 11800 ms per ubatch (server prompt ms / ubatches), experts 9500 ms (the prefill calls' MoE op, profile, the arm's mean over 5 calls per layer) and dense 2300 ms (the rest)\" '$OUT/verdicts.txt'"
check "ubatch_verdicts: the warmup's batch-call op (24000) is not the expert time" "! grep -q 'experts 24000' '$OUT/verdicts.txt'"
check "ubatch_verdicts: a skipped prompt has no timings" "grep -q '^hook-prefill-a 8k: no timings (skipped-time)' '$OUT/verdicts.txt'"
check "ubatch_verdicts: the prefill profile line per ubatch" \
  "grep -q '^hook-prefill-a prefill profile per ubatch: 70.2 GiB copied (30.1 GiB staged), copy stream 3400 ms (22.17 GB/s), compute 2000 ms, GPU 9000 ms, overlap 0.10, staging 8000 ms on the host' '$OUT/verdicts.txt'"
check "ubatch_verdicts: staging 8000 ms over the copy stream's 3400: staging-bound yes" "[[ \$STAGING_BOUND == yes ]]"
check "ubatch_verdicts: only its own arm's rows" "! grep -q '^hook-prefill-a.*360000' '$OUT/verdicts.txt'"
sed 's/8000.0 ms filling/3000.0 ms filling/' "$OUT/hook.log" > "$OUT/hook-copy.log"
ubatch_verdicts hook-prefill-a 4096 "$OUT/hook-copy.log"
check "ubatch_verdicts: staging 3000 ms under the copy stream's 3400: staging-bound no" "[[ \$STAGING_BOUND == no ]]"
ubatch_verdicts stock-prefill-a 4096
check "ubatch_verdicts, stock (no log): 90000 ms per ubatch, no expert split" \
  "grep -q '^stock-prefill-a 16k: 4 ubatches of <= 4096 tokens, 90000 ms per ubatch (server prompt ms / ubatches)\$' '$OUT/verdicts.txt' && grep -q '^stock-prefill-a: 4 ubatches over 1 prompts, 90000 ms per ubatch (server prompt ms / ubatches)\$' '$OUT/verdicts.txt'"
check "ubatch_verdicts, stock (no log): staging-bound unknown" "[[ \$STAGING_BOUND == unknown ]]"

echo "t04a-lib-test: $( (( failed )) && echo FAIL || echo PASS) ($failed failed)"
(( failed == 0 ))
