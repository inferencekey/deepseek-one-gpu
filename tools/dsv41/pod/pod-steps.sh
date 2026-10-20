# The step runner of the engine's pod-side scripts, sourced by tools/dsv41/pod/devgpu-run.sh and e07-run.sh (moved
# out of devgpu-run.sh): run NAME CMD... runs one step under `timeout` with the time left before DEADLINE
# (unix seconds), its output in out/NAME.log, and writes "step NAME: exit N, S s" (or "step NAME: skipped, ...")
# to out/steps.log; once a step returns, whatever is left of its process group (a server's engine after a failed
# check) gets SIGTERM, then SIGKILL 10 s later, before the next step starts, each such kill in out/leftovers.log
# (a failed serve step's leftover engine was once the likely contention of a later decode). run returns the step's exit status (124 when skipped). steps_begin starts the monitor: every 30 s it
# appends free -m, the GPU's memory, the memory cgroup's usage, the running step's processes with their RSS and
# dmesg's tail to out/monitor.log (an early A100 pod went silent mid-session, host OOM suspected, and nothing
# told what the host was doing). `failed` is 1 once a step failed or was skipped. The caller
# runs from its payload directory with DEADLINE set.
# Per-step budgets: with STEP_ESTIMATES set (a file of "NAME SECONDS" lines, the plan's
# estimate per step), a listed step's timeout is min(2 x its estimate, the time left), so one slow step does not use
# the next one's time; a step killed at its timeout gets "step NAME killed: timeout after B s (...)" at the end of
# its log and the same cause on its steps.log line.
# Each step that ran also gets "NAME START_S END_S EXIT" (unix seconds) in out/step-times.log: its time window, which
# tools/dsv41/pod/qdiag_report.py joins to the session samplers (pod-samplers.sh) and the proxy's requests.

# the running step's name and process group (timeout leads its own group, so the group is the whole step)
monitor() {
  while :; do
    {
      echo "=== $(date -u +%H:%M:%S)"
      free -m
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | sed 's/^/GPU memory used, total: /'
      local f
      for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.usage_in_bytes; do
        [[ -r "$f" ]] && echo "cgroup memory usage: $(( $(cat "$f") >> 20 )) MiB ($f)" && break
      done
      local step="" pgid=""
      if read -r step pgid 2>/dev/null < out/current-step; then
        echo "step $step (process group $pgid):"
        ps -e -o pgid=,pid=,rss=,comm= | awk -v g="$pgid" '$1 == g { kb += $3; printf "  pid %s %s: RSS %d MiB\n", $2, $4, $3 / 1024 }
          END { printf "  RSS of the step: %d MiB\n", kb / 1024 }'
      else
        echo "no step running"
      fi
      dmesg 2>&1 | tail -n 10
    } >> out/monitor.log
    sleep 30
  done
}

steps_begin() {
  mkdir -p out
  : > out/steps.log
  : > out/step-times.log
  monitor &
  monitor_pid=$!
  trap 'kill "$monitor_pid" 2>/dev/null' EXIT
}

# the pids still in process group $1
group_pids() { ps -e -o pgid=,pid= | awk -v g="$1" '$1 == g { print $2 }'; }

# reap STEP PGID: the step's processes still running once it returned, stopped and logged to out/leftovers.log
reap() {
  local left
  left="$(group_pids "$2" | tr '\n' ' ')"
  [[ -n "$left" ]] || return 0
  kill -TERM -- "-$2" 2>/dev/null
  for _ in $(seq 20); do [[ -z "$(group_pids "$2")" ]] && break; sleep 0.5; done
  local still
  still="$(group_pids "$2" | tr '\n' ' ')"
  [[ -n "$still" ]] && kill -KILL -- "-$2" 2>/dev/null
  echo "step $1: left running: ${left% }; SIGTERM$([[ -n "$still" ]] && echo ", then SIGKILL to ${still% }")" >> out/leftovers.log
}

# step_estimate NAME: the step's estimate in seconds from STEP_ESTIMATES, empty when it has none
step_estimate() {
  [[ -n "${STEP_ESTIMATES:-}" ]] || return 0
  awk -v n="$1" '$1 == n { print $2; exit }' "$STEP_ESTIMATES"
}

failed=0
run() {   # run NAME CMD...: stdout and stderr to out/NAME.log
  local name="$1"; shift
  local left=$(( DEADLINE - $(date +%s) - 30 ))
  if (( left < 30 )); then echo "step $name: skipped, ${left} s left" >> out/steps.log; failed=1; return 124; fi
  local budget="$left" why="the time left before the deadline" est
  est="$(step_estimate "$name")"
  if [[ -n "$est" ]] && (( 2 * est < left )); then budget=$(( 2 * est )) why="2 x its plan estimate of $est s"; fi
  [[ -z "$est" ]] || why+="; min(2 x $est s, the $left s left)"
  local t0=$(date +%s) rc=0 pid cause=""
  timeout "$budget" "$@" > "out/$name.log" 2>&1 &
  pid=$!
  echo "$name $pid" > out/current-step
  wait "$pid" || rc=$?
  reap "$name" "$pid"
  rm -f out/current-step
  if (( rc == 124 )) && (( $(date +%s) - t0 >= budget )); then
    cause=" (timeout after $budget s: $why)"
    echo "step $name killed: timeout after $budget s ($why)" >> "out/$name.log"
  fi
  echo "step $name: exit $rc, $(( $(date +%s) - t0 )) s$cause" >> out/steps.log
  echo "$name $t0 $(date +%s) $rc" >> out/step-times.log
  (( rc == 0 )) || failed=1
  return "$rc"
}
