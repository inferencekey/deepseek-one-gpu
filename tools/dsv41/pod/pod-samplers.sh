#!/usr/bin/env bash
# The session samplers of the engine's quality plans (tools/dsv41/pod/e07-run.sh, E07_PLAN=qengine and qdiag:
# one qengine session decoded at ~15 tok/s against 62.8 on the same card model, and no GPU clock, power or throttle
# sample existed to tell why, so the GPU clocks are sampled during the run). Started right
# after env, stopped on e07-run.sh's exit; run from the payload directory. Each sampler runs in its own session
# (setsid, as pod_server.sh's GPU memory sampler), so pod-steps.sh's reap of a step leaves it running; a sampler that
# cannot read something records that and goes on, and this script always exits 0 (it must never fail a session).
#
#   bash pod-samplers.sh start   the GPU every 1 s into out/gpu-clocks.csv: its first line the fields queried, then
#                                nvidia-smi's rows (csv, no units, the time in UTC): timestamp, clocks.sm, clocks.mem,
#                                power.draw, enforced.power.limit, temperature.gpu, utilization.gpu,
#                                utilization.memory, pcie.link.gen.current and the active clock event (throttle)
#                                reasons bitmask, clocks_event_reasons.active, or clocks_throttle_reasons.active on a
#                                driver without it; each field is probed first and one the driver refuses is left
#                                out, named in out/samplers.log. The host every 2 s into out/host-load.log, one block
#                                per sample: "=== UNIX_S", loadavg, the cgroup's CPU limit (v2 cpu.max, else v1
#                                cpu.cfs_quota_us and cpu.cfs_period_us), its cpu.stat (v2: nr_throttled,
#                                throttled_usec; v1: nr_throttled, throttled_time in ns) and the top 8 processes by
#                                ps's %CPU (CPU time over the process's life). Pids in out/samplers.pid.
#   bash pod-samplers.sh stop    both stopped (SIGTERM to their sessions)
set -u
mode="${1:-}"
cg="${CGROUP_ROOT:-/sys/fs/cgroup}"

host_loop() {
  local f
  while :; do
    {
      echo "=== $(date +%s)"
      echo "loadavg $(cat /proc/loadavg 2>&1)"
      if [[ -r "$cg/cpu.max" ]]; then
        echo "cpu-limit v2 cpu.max $(cat "$cg/cpu.max" 2>&1)"
        f="$cg/cpu.stat"
      elif [[ -r "$cg/cpu/cpu.cfs_quota_us" ]]; then
        echo "cpu-limit v1 cfs_quota_us $(cat "$cg/cpu/cpu.cfs_quota_us" 2>&1) cfs_period_us $(cat "$cg/cpu/cpu.cfs_period_us" 2>&1)"
        f="$cg/cpu/cpu.stat"
      else
        echo "cpu-limit none (no cpu.max nor cpu/cpu.cfs_quota_us under $cg)"
        f=""
      fi
      [[ -n "$f" && -r "$f" ]] && echo "cpu-stat $(tr '\n' ' ' < "$f")" || echo "cpu-stat none"
      ps -e -o pid=,pcpu=,rss=,comm= --sort=-pcpu 2>&1 | head -n 8 | sed 's/^/top /'
    } >> out/host-load.log 2>&1
    sleep 2
  done
}

case "$mode" in
  start)
    mkdir -p out
    fields=()
    missing=()
    for f in timestamp clocks.sm clocks.mem power.draw enforced.power.limit temperature.gpu utilization.gpu \
             utilization.memory pcie.link.gen.current; do
      nvidia-smi --query-gpu="$f" --format=csv,noheader > /dev/null 2>&1 && fields+=("$f") || missing+=("$f")
    done
    reason=""
    for f in clocks_event_reasons.active clocks_throttle_reasons.active; do
      nvidia-smi --query-gpu="$f" --format=csv,noheader > /dev/null 2>&1 && { reason="$f"; break; }
    done
    [[ -n "$reason" ]] && fields+=("$reason") || missing+=("clocks_event_reasons.active clocks_throttle_reasons.active")
    q="$(IFS=,; echo "${fields[*]}")"
    {
      echo "pod-samplers.sh: the GPU every 1 s into out/gpu-clocks.csv: $q"
      echo "pod-samplers.sh: the throttle reasons field: ${reason:-none (the driver knows neither)}"
      echo "pod-samplers.sh: fields the driver refused: ${missing[*]:-none}"
    } >> out/samplers.log
    echo "$q" > out/gpu-clocks.csv
    if (( ${#fields[@]} )); then
      TZ=UTC setsid nvidia-smi --query-gpu="$q" --format=csv,noheader,nounits -lms 1000 >> out/gpu-clocks.csv \
        2>> out/samplers.log < /dev/null &
      echo "gpu $!" > out/samplers.pid
    fi
    export -f host_loop
    cg="$cg" setsid bash -c host_loop < /dev/null > /dev/null 2>&1 &
    echo "host $!" >> out/samplers.pid
    echo "pod-samplers.sh: started ($(tr '\n' ' ' < out/samplers.pid))" >> out/samplers.log
    ;;
  stop)
    while read -r what pid; do
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      echo "pod-samplers.sh: the $what sampler $pid stopped" >> out/samplers.log
    done < out/samplers.pid 2>/dev/null
    rm -f out/samplers.pid
    ;;
  *) echo "pod-samplers.sh: '$mode': start or stop" >&2 ;;
esac
exit 0
