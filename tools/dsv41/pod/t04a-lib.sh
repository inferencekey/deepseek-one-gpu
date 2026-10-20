# Helpers of tools/dsv41/pod/t04a.sh, sourced by it and by t04a-lib-test.sh. They never fail the caller's
# `set -euo pipefail` on a missing file or tool: a diagnostic that cannot be read is written as absent.
#
# The caller sets: OUT (results dir), DRY (1: print, run nothing), START and DEADLINE (epoch seconds), LOG (the
# main log, /dev/null in a dry run), CGROUP_ROOT (default /sys/fs/cgroup) and NUMA_ROOT (default
# /sys/devices/system/node), fake trees in the test.
#
# Logs: $LOG gets every line; $OUT/steps.log gets one "start" and one "end ... exit RC" line per step;
# $OUT/steps.tsv gets "NAME RC SECONDS" per command step (RC skipped-cap / skipped-time when not run).

CGROUP_ROOT="${CGROUP_ROOT:-/sys/fs/cgroup}"
NUMA_ROOT="${NUMA_ROOT:-/sys/devices/system/node}"
CURRENT_STEP="(none)"

log() { echo "[t04a $(date -u +%H:%M:%S) +$(( ($(date +%s) - START) / 60 ))m] $*" | tee -a "$LOG"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
# verdict MSG: a pass/fail line of the run, also in $OUT/verdicts.txt; a dry run has none
verdict() {
  (( DRY )) && return 0
  log "$*"
  echo "$*" >> "$OUT/verdicts.txt"
}

# step_start NAME / step_end NAME RC: the step log; a dry run writes nothing
step_start() {
  CURRENT_STEP="$1"
  (( DRY )) && return 0
  echo "$(date -u +%FT%TZ) start $1" >> "$OUT/steps.log"
  log "$1: start"
}
step_end() {
  (( DRY )) && return 0
  echo "$(date -u +%FT%TZ) end $1 exit $2" >> "$OUT/steps.log"
  log "$1: end, exit $2"
}

# show_file PATH: "PATH = CONTENT", or "PATH: absent"; never fails
show_file() {
  if [[ -r "$1" ]]; then
    echo "$1 = $(cat "$1" 2>&1 || true)"
  else
    echo "$1: absent"
  fi
}

# show_cmd CMD...: the command's output, or a line saying it is missing or how it failed; never fails
show_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then echo "$1: not installed"; return 0; fi
  local rc=0
  "$@" 2>&1 || rc=$?
  (( rc == 0 )) || echo "$*: exit $rc"
  return 0
}

# the memory files of cgroup v1 (memory/) and v2 (the root): the pods so far are v1, whose usage includes the
# page cache; whichever version is absent shows as absent
cgroup_memory() {
  local f
  for f in "$CGROUP_ROOT"/memory/memory.{limit_in_bytes,usage_in_bytes,max_usage_in_bytes} \
           "$CGROUP_ROOT"/memory.{max,current,peak}; do
    show_file "$f"
  done
}

# snapshot NAME: GPU memory, free and the cgroup memory, appended to $OUT/snapshots.txt; never fails
snapshot() {
  local name="$1"
  if (( DRY )); then echo "+ [snapshot-$name] nvidia-smi; free -g; cgroup memory files"; return 0; fi
  step_start "snapshot-$name"
  {
    echo "== $(date -u +%FT%TZ) $name"
    show_cmd nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
    show_cmd free -g
    cgroup_memory
  } >> "$OUT/snapshots.txt" 2>&1 || true
  step_end "snapshot-$name" 0
}

# majflt PID: the process's major page faults so far (all its threads, field 12 of /proc/PID/stat), or "absent";
# never fails
majflt() {
  local stat f
  stat="$(cat "/proc/$1/stat" 2>/dev/null)" || { echo absent; return 0; }
  # the fields after "PID (comm) " start at field 3; comm may hold spaces and parentheses
  read -ra f <<<"${stat##*) }"
  echo "${f[9]:-absent}"
}

# fault_rate N0 N1 T0 T1: "majflt N majflt/s R" between two majflt counts at times T0 and T1 (seconds, may be
# fractional), or "majflt absent" when a count is absent; never fails
fault_rate() {
  awk -v a="$1" -v b="$2" -v t0="$3" -v t1="$4" 'BEGIN {
    if (a !~ /^[0-9]+$/ || b !~ /^[0-9]+$/ || t1 <= t0) { print "majflt absent"; exit }
    printf "majflt %d majflt/s %.0f\n", b - a, (b - a) / (t1 - t0) }'
}

# numa_free_kib NODE: MemFree of NUMA node NODE in KiB (its meminfo), or "absent"; never fails
numa_free_kib() {
  local kib
  kib="$(awk -v n="$1" '$1 == "Node" && $2 == n && $3 == "MemFree:" { print $4 }' "$NUMA_ROOT/node$1/meminfo" 2>/dev/null)" || true
  [[ "$kib" =~ ^[0-9]+$ ]] && echo "$kib" || echo absent
}

# numa_free_line: "node N MemFree X GiB" for every NUMA node, or "no NUMA nodes"; never fails
numa_free_line() {
  local d n kib out=""
  for d in "$NUMA_ROOT"/node[0-9]*; do
    [[ -d "$d" ]] || continue
    n="${d##*/node}"; kib="$(numa_free_kib "$n")"
    if [[ "$kib" == absent ]]; then out+="node $n MemFree absent; "
    else out+="node $n MemFree $(awk -v k="$kib" 'BEGIN { printf "%.2f", k / 1048576 }') GiB; "; fi
  done
  out="${out%; }"
  echo "${out:-no NUMA nodes under $NUMA_ROOT}"
}

# numa_binding NODE: the command prefix that runs a process on NUMA node NODE's CPUs with its memory
# first-touched there, into the array NUMA_BIND (empty: unbound), and one "binding: ..." line per form tried and the
# outcome into the array NUMA_BIND_LOG. The forms, each tried on `true`, in order:
#   numactl --cpunodebind=NODE --preferred=NODE  the CPUs and the memory policy; Docker's default seccomp profile
#                                                refuses set_mempolicy (EPERM) without CAP_SYS_NICE (the build machine's
#                                                ubuntu:22.04 container, numactl 2.0.14, 2026-10-19)
#   numactl --cpunodebind=NODE                   the CPUs only: the default local policy first-touches on NODE
#   taskset -c <NUMA_ROOT/nodeNODE/cpulist>      the same CPUs without libnuma, the list dsv41_cpu_experts_probe binds to
# NODE not a number or fewer than two nodes under NUMA_ROOT: unbound. Never fails.
numa_binding() {
  local node="$1" d nodes=0 cpus="" f rc err status seccomp capeff nice
  local -a forms cmd
  NUMA_BIND=() NUMA_BIND_LOG=()
  if [[ ! "$node" =~ ^[0-9]+$ ]]; then
    NUMA_BIND_LOG=("binding: unbound, no NUMA node of the GPU (nvidia-smi topo -m's GPU0 row: '${node:-nothing}')")
    return 0
  fi
  for d in "$NUMA_ROOT"/node[0-9]*; do
    if [[ -d "$d" ]]; then nodes=$(( nodes + 1 )); fi
  done
  if (( nodes < 2 )); then
    NUMA_BIND_LOG=("binding: unbound, $nodes NUMA node(s) under $NUMA_ROOT: node $node holds every CPU and all the memory")
    return 0
  fi
  cpus="$(cat "$NUMA_ROOT/node$node/cpulist" 2>/dev/null)" || true
  forms=("numactl --cpunodebind=$node --preferred=$node" "numactl --cpunodebind=$node")
  [[ -z "$cpus" ]] || forms+=("taskset -c $cpus")
  for f in "${forms[@]}"; do
    read -ra cmd <<<"$f"
    rc=0; err="$("${cmd[@]}" true 2>&1 >/dev/null)" || rc=$?
    if (( rc == 0 )); then
      NUMA_BIND=("${cmd[@]}")
      case "$f" in
        *--preferred*) NUMA_BIND_LOG+=("binding: $f worked (node $node's CPUs, its memory preferred)") ;;
        numactl*) NUMA_BIND_LOG+=("binding: $f worked (node $node's CPUs; the default local policy first-touches the memory there)") ;;
        *) NUMA_BIND_LOG+=("binding: $f worked (node $node's CPUs from $NUMA_ROOT/node$node/cpulist; the default local policy first-touches the memory there)") ;;
      esac
      return 0
    fi
    NUMA_BIND_LOG+=("binding: $f refused, exit $rc: ${err//$'\n'/; }")
    if [[ "$f" == *--preferred* ]]; then
      # whether the seccomp filter refused set_mempolicy: CAP_SYS_NICE is capability bit 23
      status="$(cat /proc/self/status 2>/dev/null)" || true
      seccomp="$(awk '$1 == "Seccomp:" { print $2 }' <<<"$status")"
      capeff="$(awk '$1 == "CapEff:" { print $2 }' <<<"$status")"
      nice=unknown
      [[ "$capeff" =~ ^[0-9a-fA-F]+$ ]] && { (( 16#$capeff >> 23 & 1 )) && nice=present || nice=absent; }
      NUMA_BIND_LOG+=("binding: this process: Seccomp ${seccomp:-absent} (2: a filter), CapEff ${capeff:-absent}, CAP_SYS_NICE $nice")
      if [[ "$err" == *"Operation not permitted"* && "$seccomp" == 2 && "$nice" == absent ]]; then
        NUMA_BIND_LOG+=("binding: the memory policy is the seccomp filter's refusal: Docker's default profile allows set_mempolicy and mbind only with CAP_SYS_NICE")
      fi
    fi
  done
  [[ -n "$cpus" ]] || NUMA_BIND_LOG+=("binding: taskset not tried, no CPU list in $NUMA_ROOT/node$node/cpulist")
  NUMA_BIND_LOG+=("binding: unbound, every form refused")
}

# memlock_line: RLIMIT_MEMLOCK (soft and hard, KiB) and CapEff of this shell; never fails
memlock_line() {
  echo "RLIMIT_MEMLOCK soft $(ulimit -l 2>&1 || true) hard $(ulimit -Hl 2>&1 || true) KiB;" \
       "$(grep '^CapEff' /proc/self/status 2>/dev/null | tr -s '\t' ' ' || echo 'CapEff absent')"
}

# ram_mode_verdict NAME: the warm tier's STRATA_MOE_RAM_MODE lines of residency run NAME (its log) in verdicts.txt
ram_mode_verdict() {
  (( DRY )) && return 0
  local lines
  lines="$(grep -a 'strata-moe: STRATA_MOE_RAM_MODE=' "$OUT/$1.log" 2>/dev/null | tr '\n' ' ')" || true
  verdict "$1: RAM mode: ${lines:-no STRATA_MOE_RAM_MODE line in $1.log}"
}

# cold_path_verdict NAME: the STRATA_MOE_COLD line of residency run NAME (its log) in verdicts.txt
cold_path_verdict() {
  (( DRY )) && return 0
  local lines
  lines="$(grep -a '^strata-moe: STRATA_MOE_COLD=' "$OUT/$1.log" 2>/dev/null | tr '\n' ' ')" || true
  verdict "$1: cold path: ${lines:-no STRATA_MOE_COLD line in $1.log}"
}

# T06 --prefill, UBATCH_PY ARM UB TSV [LOG]: per prompt of ARM in prefill.tsv, the server's ms per ubatch (prompt
# ms / ubatches of at most UB tokens), and once per arm the same over all its prompts; with the arm's log (a hook
# arm, STRATA_MOE_PROFILE=1) the arm line splits it into the MoE op's ms per ubatch and the rest, the dense part.
# The op time is the prefill line's: the prefill calls only (not the batch-call table, which also holds the server's
# 2-token warmup) and no layer's first call (it uploads the slab and fills the warm tier), the mean over every
# prompt of the arm, so it is set against the arm's own mean and not a prompt's. Then the prefill profile per
# ubatch; the last line is "staging-bound yes|no|unknown": yes when the host's staging ms per ubatch exceed the copy
# stream's busy ms (the PCIe copies waited for the staging)
# shellcheck disable=SC2016  # python source, expanded by nobody
UBATCH_PY='import math, re, sys
arm, ub, tsv = sys.argv[1], int(sys.argv[2]), sys.argv[3]
text = open(sys.argv[4], errors="replace").read() if len(sys.argv) > 4 else ""
m = re.search(r"prefill per ubatch \(sum of the layer means\): ([0-9.]+) GiB copied \(([0-9.]+) GiB staged\) in "
              r"([0-9.]+) ms of copy stream \(([0-9.]+) GB/s\), ([0-9.]+) ms of compute, ([0-9.]+) ms of GPU time "
              r"\(overlap ([0-9.]+)\), ([0-9.]+) ms filling the staging on the host, ([0-9.]+) ms of MoE op over "
              r"([0-9]+) op calls", text)
prompts, ubatches, total_ms = 0, 0, 0.0
for line in open(tsv):
    f = line.rstrip("\n").split("\t")
    if f[0] != arm or len(f) < 4: continue
    v = f[3].split()
    if len(v) != 5 or not v[0].isdigit():
        print(f"{arm} {f[1]}: no timings ({f[3]})"); continue
    n, ms = int(v[0]), float(v[1])
    k = math.ceil(n / ub)
    prompts, ubatches, total_ms = prompts + 1, ubatches + k, total_ms + ms
    print(f"{arm} {f[1]}: {k} ubatches of <= {ub} tokens, {ms / k:.0f} ms per ubatch (server prompt ms / ubatches)")
if ubatches:
    per = total_ms / ubatches
    out = f"{arm}: {ubatches} ubatches over {prompts} prompts, {per:.0f} ms per ubatch (server prompt ms / ubatches)"
    if m:
        op, calls = float(m.group(9)), int(m.group(10))
        out += (f", experts {op:.0f} ms (the prefill calls\x27 MoE op, profile, the arm\x27s mean over {calls} calls "
                f"per layer) and dense {per - op:.0f} ms (the rest)")
    print(out)
if m:
    g, st, copy, gbs, comp, gpu, ov, stage = (float(x) for x in m.groups()[:8])
    print(f"{arm} prefill profile per ubatch: {g:.1f} GiB copied ({st:.1f} GiB staged), copy stream {copy:.0f} ms "
          f"({gbs:.2f} GB/s), compute {comp:.0f} ms, GPU {gpu:.0f} ms, overlap {ov:.2f}, staging {stage:.0f} ms on the host")
    print("staging-bound " + ("yes" if stage > copy else "no"))
else:
    if text: print(f"{arm}: no prefill profile line in its log")
    print("staging-bound unknown")'
# ubatch_verdicts ARM UB [LOG]: UBATCH_PY's lines in verdicts.txt; STAGING_BOUND gets its last word
STAGING_BOUND=unknown
ubatch_verdicts() {
  local arm="$1" ub="$2" log="${3:-}"
  if (( DRY )); then echo "+ [$arm-ubatch] python3 -c <UBATCH_PY> $arm $ub $OUT/prefill.tsv${log:+ $log}"; return 0; fi
  local args=("$arm" "$ub" "$OUT/prefill.tsv") lines line
  [[ -n "$log" && -f "$log" ]] && args+=("$log")
  lines="$(python3 -c "$UBATCH_PY" "${args[@]}" 2>&1)" || lines="$arm: the ubatch analysis failed: $lines"
  while IFS= read -r line; do
    [[ "$line" == staging-bound* ]] || verdict "$line"
  done <<<"$lines"
  STAGING_BOUND="$(sed -n 's/^staging-bound //p' <<<"$lines" | tail -n 1)"
}

# cgroup_limit_bytes: the memory cgroup's limit in bytes (v2 memory.max, else v1 memory.limit_in_bytes), "unlimited"
# (v2 max, or v1's near-2^63 no-limit value) or "absent"; never fails
cgroup_limit_bytes() {
  local v
  for f in "$CGROUP_ROOT/memory.max" "$CGROUP_ROOT/memory/memory.limit_in_bytes"; do
    [[ -r "$f" ]] || continue
    v="$(head -n 1 "$f" 2>/dev/null)" || true
    if [[ "$v" == max ]] || { [[ "$v" =~ ^[0-9]+$ ]] && (( ${#v} > 18 || v >= 4611686018427387904 )); }; then
      echo unlimited; return 0
    fi
    [[ "$v" =~ ^[0-9]+$ ]] && { echo "$v"; return 0; }
  done
  echo absent
}

# plan_add NAME MINUTES / plan_print CAP_MIN: the session plan, the steps in order with an estimate each; plan_print
# prints it with the running total and returns 1 when the total is over CAP_MIN
PLAN_NAMES=()
PLAN_MINUTES=()
plan_add() { PLAN_NAMES+=("$1"); PLAN_MINUTES+=("$2"); }
plan_print() {
  local i total=0
  echo "plan: step, estimate min, cumulative min (cap $1 min)"
  for i in "${!PLAN_NAMES[@]}"; do
    total="$(awk -v a="$total" -v b="${PLAN_MINUTES[$i]}" 'BEGIN { printf "%.1f", a + b }')"
    printf 'plan: %-28s %5.1f %6.1f\n' "${PLAN_NAMES[$i]}" "${PLAN_MINUTES[$i]}" "$total"
  done
  if awk -v t="$total" -v c="$1" 'BEGIN { exit !(t + 0 <= c + 0) }'; then
    echo "plan: total $total min, fits the $1-minute cap"
  else
    echo "plan: total $total min, over the $1-minute cap"
    return 1
  fi
}

# run NAME CMD...: runs CMD with the time left under the cap, output in $OUT/NAME.log, the command in NAME.cmd;
# appends "NAME rc seconds" to steps.tsv and returns CMD's exit code (124 when the cap stopped it)
run() {
  local name="$1"; shift
  local cmd
  cmd="$(printf '%q ' "$@")"
  if (( DRY )); then echo "+ [$name] $cmd"; return 0; fi
  echo "$cmd" > "$OUT/$name.cmd"
  step_start "$name"
  local t0 rc=0 budget
  t0=$(date +%s)
  budget=$(left)
  if (( budget <= 0 )); then
    log "$name: wall-time cap reached, not run"; echo "$name skipped-cap 0" >> "$OUT/steps.tsv"
    step_end "$name" 124
    return 124
  fi
  timeout --kill-after=30 "$budget" "$@" > "$OUT/$name.log" 2>&1 || rc=$?
  echo "$name $rc $(( $(date +%s) - t0 ))" >> "$OUT/steps.tsv"
  log "$name: exit $rc in $(( $(date +%s) - t0 )) s"
  step_end "$name" "$rc"
  return "$rc"
}

# fits NAME MINUTES: true when the step's estimated minutes still fit under the cap, else records the skip
fits() {
  (( DRY )) && return 0
  if (( $(left) < $2 * 60 )); then
    log "$1: skipped, needs about $2 min and $(( $(left) / 60 )) min are left"
    echo "$1 skipped-time 0" >> "$OUT/steps.tsv"
    echo "$(date -u +%FT%TZ) skip $1 (time)" >> "$OUT/steps.log"
    return 1
  fi
}

# on_exit RC: the last line of every run, whatever ended it (t04a.sh's EXIT trap passes the script's status)
on_exit() {
  local rc="$1"
  (( DRY )) && return 0
  echo "$(date -u +%FT%TZ) exit $rc (last step $CURRENT_STEP)" >> "$OUT/steps.log" 2>/dev/null || true
  log "t04a.sh: exit $rc, last step $CURRENT_STEP" || true
}

# on_err LINE CMD: a failing command that set -e turns into an exit is named before it happens (the ERR trap)
on_err() {
  (( DRY )) && return 0
  log "t04a.sh: line $1 failed (exit $3): $2" || true
}
