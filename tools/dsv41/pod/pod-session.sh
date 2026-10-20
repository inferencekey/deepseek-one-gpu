# The RunPod session of the engine's pod scripts, sourced by tools/dsv41/pod/devgpu.sh (the dev pod) and
# tools/dsv41/pod/e07.sh (the real model): the offers and their price and RAM checks, the create, the watchdog
# at the cap, the SSH wait and the CUDA probe (pod_up), the payload's upload and the detached run with its polls over
# fresh SSH connections (run_detached) and, with PULL_STEPS=1, out/ pulled after each poll that sees a new step
# (pull_steps), the results' pull as two archives made on the pod (pull_results), the delete and the cost (end_pod),
# and the EXIT trap's pull-then-delete (cleanup, set by the caller). e07.sh runs the same session as devgpu.sh on another
# payload.
#
# The caller sets, before sourcing: OUT, SESSION_LOG (the session's log), GPU_TYPES, CLOUD_TYPES, DATA_CENTERS, IMAGE,
# DISK_GB, MAX_USD_PER_HR, MAX_SESSION_USD, CAP_MIN, MAX_PODS, MIN_RAM_GB, SSH_KEY, PULL_S, PULL_CAP_MB (0: no dumps
# archive), PULL_STEPS (1: pull_steps mid-run; unset or 0: devgpu.sh's one pull at the end), POLL_FAILS, SSH_WAIT_S,
# PROBE_S, PULL_TAR_S, START_S, POLL_S, POLL_SSH_S, REMOTE (the payload directory on the pod), RUN_SCRIPT (the
# payload's script that runs there), NETWORK_VOLUME (a RunPod network volume id mounted at /workspace, "" for none)
# and stages (an associative array: a flavor's tag, cu128 or cu130, to the local payload directory of that flavor,
# which holds its dsv41_cuda_probe). Each GPU of GPU_TYPES takes the flavor
# built for it (gpu_flavor, tools/dsv41/flavors.sh): its offers ask for hosts whose driver runs that flavor, a GPU
# whose flavor has no payload is skipped, and the created pod's flavor is pod_tag (its payload: stages[pod_tag]).
# The offers tried per GPU, in order, are offers ("CLOUD|DC|VOLUME": DC "any" for any data center, VOLUME a network
# volume id or "" for none), filled by the caller after sourcing, else every cloud of clouds times every data center
# of dcs with NETWORK_VOLUME; the created pod's volume is pod_volume, which RUN_SCRIPT gets as POD_VOLUME, and its
# machineId (the create's) as POD_MACHINE (e07-run.sh's speed check names the machine).
# QUALIFY=1 (e07.sh's sessions; unset or 0: devgpu.sh's) adds the machine qualification
# right after a passing CUDA probe and before anything else goes up: the pod's flavor's
# dsv41_machine_probe (stages[pod_tag]) with the bounds QUALIFY_ARGS and --min-host-read-gbs from the caller's
# qualify_host_read on the created pod's GPU (pod_gpu; it prints "GB/s|why", logged with the pod) under timeout
# QUALIFY_S; a FAIL line deletes the pod and remembers its machine as a failed CUDA probe does. probe.log gets a "=== pod" line per pod (its id, machine,
# GPU and offer) before its probe's output, then (a passing CUDA probe) the pod's memory cgroup limit and MemTotal
# ("pod memory cgroup: ..." lines: e07.sh's dry run reads them with PROBE_LOG), and the
# qualification's lines also go to SESSION_LOG. With CPU_EXPERTS_FLOOR_GBS set (e07.sh's sessions) a passing
# machine probe is followed by dsv41_cpu_experts_probe (stages[pod_tag], from the strata tarball) with
# CPU_EXPERTS_THREADS threads and that floor; a rate under it ("qualification: CPU experts X GB/s < FLOOR ...") deletes
# the pod and remembers its machine the same way.

pod_lib="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/dsv41/flavors.sh
source "$pod_lib/../flavors.sh"
# RUNPOD_API: q02-test.sh's rehearsal of the session points it at a fake RunPod
api() { python3 "${RUNPOD_API:-$pod_lib/runpod_api.py}" "$@"; }
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$SESSION_LOG" >&2; }
json_field() { python3 -c 'import json, sys; print(json.loads(sys.argv[1]).get(sys.argv[2]) or "")' "$1" "$2"; }
# minutes of session the price allows: floor((MAX_SESSION_USD - spent) / price * 60), capped at CAP_MIN
cap_minutes() { python3 -c "import sys; p, s, c, u = map(float, sys.argv[1:]); print(max(0, int(min(c, (s - u) / p * 60))))" "$1" "$MAX_SESSION_USD" "$CAP_MIN" "${2:-0}"; }
price_ok() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$1" "$MAX_USD_PER_HR"; }
# a pod's host RAM against MIN_RAM_GB: always fine without a minimum, else its memoryInGb must be reported and enough
ram_ok() { (( MIN_RAM_GB == 0 )) || python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$1" "$MIN_RAM_GB" 2>/dev/null; }
# gpu_cfg G NAME: flavor_config's NAME (FLAVOR_TAG, POD_CUDA_VERSIONS, ...) for the flavor built for GPU G
gpu_cfg() { local FLAVOR_TAG CUDA_ARCHS ARCH_TAG MIN_DRIVER CUDA_MAJOR POD_CUDA_VERSIONS; flavor_config "$(gpu_flavor "$1")"; echo "${!2}"; }
# create_opts G [VOLUME]: the create's options beyond the GPU, cloud and data center (the dry runs print the same body)
create_opts() {
  echo --image "$IMAGE" --disk "$DISK_GB" --min-ram "$MIN_RAM_GB" --cuda-versions "$(gpu_cfg "$1" POD_CUDA_VERSIONS)"
  [[ -z "${2:-}" ]] || echo --network-volume "$2"
}
offers=()

# machine_probe_ready TAG (QUALIFY=1's callers, e07.sh and devgpu.sh's c05 plan; they set STRATA_DIR and
# MACHINE_PROBE_ROOT): the flavor's dsv41_machine_probe (pod/build-machine-probe.sh) described in one line; exit 1 when
# it is missing, its MANIFEST's source is not HEAD's tests/dsv41/engine/machine_probe.cu or the binary is not the
# MANIFEST's
MACHINE_PROBE_SRC=tests/dsv41/engine/machine_probe.cu
machine_probe_ready() {
  local d="$MACHINE_PROBE_ROOT/$1/machine-probe" me="${BASH_SOURCE[-1]##*/}" m want head got
  m="$(cat "$d/MANIFEST" 2>/dev/null)" ||
    { echo "$me: no $d/MANIFEST: build the probe (tools/dsv41/pod/build-machine-probe.sh)" >&2; exit 1; }
  want="$(awk -v s="$MACHINE_PROBE_SRC" '$1 == "source:" && $2 == s { print $4 }' <<<"$m")"
  head="$(git -C "$STRATA_DIR" show "HEAD:$MACHINE_PROBE_SRC" 2>/dev/null | sha256sum | cut -d' ' -f1)"
  [[ -n "$want" && "$want" == "$head" ]] ||
    { echo "$me: $d/dsv41_machine_probe was built from $MACHINE_PROBE_SRC sha256 '${want}', not HEAD's $head: rebuild it (tools/dsv41/pod/build-machine-probe.sh)" >&2; exit 1; }
  got="$(sha256sum "$d/dsv41_machine_probe" 2>/dev/null | cut -d' ' -f1)"
  [[ "$got" == "$(awk '$2 == "dsv41_machine_probe" { print $1 }' <<<"$m")" ]] ||
    { echo "$me: $d/dsv41_machine_probe sha256 '${got}' is not its MANIFEST's" >&2; exit 1; }
  echo "payload-$1's dsv41_machine_probe: sha256 $got from $MACHINE_PROBE_SRC (HEAD's, sha256 $head), nvcc$(
    awk '$1 == "nvcc:" { $1 = ""; print }' <<<"$m"), archs $(awk '$1 == "archs:" { print $2 }' <<<"$m")"
}

# stage_bins TARBALL DIR NAME...: each NAME (a file in the tarball's bin/ or at its top) and its MANIFEST extracted
# into DIR, flat, each checked against the MANIFEST's sha256; exits on a mismatch
stage_bins() {
  local tgz="$1" dir="$2" top listing b f want got
  shift 2
  top="$(basename "$tgz" .tar.gz)"
  listing="$(tar -tzf "$tgz")"
  mkdir -p "$dir"
  tar -xzf "$tgz" -C "$dir" --strip-components=1 "$top/MANIFEST" $(for b in "$@"; do grep -x "$top/\(bin/\)\?$b" <<<"$listing"; done)
  for b in "$@"; do
    f="$(cd "$dir" && find . -name "$b" -type f | head -n 1)"
    want="$(awk -v n="${f#./}" '$2 == n { print $1 }' "$dir/MANIFEST")"
    got="$(sha256sum "$dir/$f" | cut -d' ' -f1)"
    [[ -n "$want" && "$want" == "$got" ]] || { log "FAIL: $b sha256 $got, MANIFEST says '${want}'"; exit 1; }
    [[ "$f" == "./$b" ]] || mv "$dir/$f" "$dir/$b"
  done
}

# BAD_MACHINES (space or comma separated machine ids, e07.sh's env): machines known bad from earlier sessions (one
# machine's HF download at 0.13 GB/s stopped two sessions), seeded into bad_machines so a pod landing there goes at once
env_bad="${BAD_MACHINES:-}"
read -ra env_bad_ids <<<"${env_bad//,/ }"
env_bad=" "
for bad_id in "${env_bad_ids[@]}"; do
  [[ "$bad_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "pod-session.sh: BAD_MACHINES has '$bad_id', not a machine id" >&2; exit 2; }
  env_bad+="$bad_id "
done
pod="" cost="" machine="" t_create=0 watchdog="" spent=0 bad_machines="$env_bad"
# deletes the current pod (if any), checks its state, adds its cost to the session's
end_pod() {
  [[ -n "$watchdog" ]] && kill "$watchdog" 2>/dev/null || true
  watchdog=""
  [[ -n "$pod" ]] || return 0
  api delete "$pod" 2>&1 | tee -a "$SESSION_LOG" >&2 || log "WARNING: delete of $pod failed: check the RunPod console"
  local after; after="$(api get "$pod" 2>&1 || true)"
  log "after delete: $after"
  local secs=$(( $(date +%s) - t_create ))
  python3 -c "import sys; p, s = float(sys.argv[1]), int(sys.argv[2]); print(f'pod {sys.argv[3]}: {s} s at {p} USD/h = {p * s / 3600:.3f} USD (upper bound: billing rounds up)')" \
    "$cost" "$secs" "$pod" | tee -a "$OUT/cost.txt" >&2
  spent="$(python3 -c "import sys; print(float(sys.argv[1]) + float(sys.argv[2]) * int(sys.argv[3]) / 3600)" "$spent" "$cost" "$secs")"
  pod=""
}
started=0 pulled=0   # the detached run was started on the pod; its results were pulled
cleanup() {
  local rc=$?
  # once the run started the results are pulled before the pod goes, whatever ended the session (the watchdog
  # deleted the pod already)
  if (( started && !pulled )) && [[ -n "$pod" && ! -f "$OUT/watchdog-fired" ]]; then
    log "pulling the results before the pod goes (exit $rc)"
    pull_results
  fi
  end_pod
  log "session spent at most $(python3 -c "import sys; print(f'{float(sys.argv[1]):.3f}')" "$spent") USD"
  log "exit $rc"
}

# the offers, in order: per GPU each offer, the list price of its GPU and cloud first (nothing is created above the
# limit, asked once per GPU and cloud), then the create in its data center
listed=" " pod_tag="" pod_volume="" pod_offer="" pod_gpu=""
create_pod() {
  local g c dc vol offer rc out id price quote list_price ram tag opts model
  if (( ${#offers[@]} == 0 )); then
    for c in "${clouds[@]}"; do for dc in "${dcs[@]}"; do offers+=("$c|$dc|$NETWORK_VOLUME"); done; done
  fi
  for g in "${gpus[@]}"; do
    tag="$(gpu_cfg "$g" FLAVOR_TAG)"
    if [[ -z "${stages[$tag]:-}" ]]; then log "$g takes the $tag flavor, which this session has no payload of: skipped"; continue; fi
    for offer in "${offers[@]}"; do
      IFS='|' read -r c dc vol <<<"$offer"
      [[ "$dc" == any ]] && dc=""
      read -ra opts <<<"$(create_opts "$g" "$vol" | tr '\n' ' ')"
      model="${vol:+network volume $vol}" model="${model:-no network volume}"
      case "$listed" in *" $g|$c|none "*) continue ;; esac
      if [[ "$listed" != *" $g|$c|"* ]]; then
        rc=0; quote="$(api price --gpu "$g" --cloud "$c" --cuda-versions "$(gpu_cfg "$g" POD_CUDA_VERSIONS)" 2>>"$SESSION_LOG")" || rc=$?
        if (( rc == 3 )); then log "not offered or none available now: $g, $c ($quote)"; listed+="$g|$c|none "; continue; fi
        (( rc == 0 )) || { log "FAIL: price of $g ($c), see $SESSION_LOG: nothing is created without it"; exit 1; }
        list_price="$(json_field "$quote" price)"
        log "$g ($c) lists $list_price USD/h"
        listed+="$g|$c|$list_price "
      fi
      list_price="${listed##*" $g|$c|"}" list_price="${list_price%% *}"
      if ! price_ok "$list_price"; then log "$g ($c) lists $list_price USD/h > $MAX_USD_PER_HR: not created ($model)"; continue; fi
      rc=0; out="$(api create --gpu "$g" --cloud "$c" --dc "$dc" "${opts[@]}" 2>>"$SESSION_LOG")" || rc=$?
      if (( rc == 3 )); then log "no stock: $g, $c, ${dc:-any DC}, $model"; continue; fi
      (( rc == 0 )) || { log "FAIL: create $g ($c, ${dc:-any DC}), see $SESSION_LOG"; exit 1; }
      id="$(json_field "$out" id)"
      price="$(json_field "$out" costPerHr)"
      pod="$id" cost="$price" machine="$(json_field "$out" machineId)" t_create=$(date +%s) pod_tag="$tag" pod_volume="$vol" pod_gpu="$g"
      pod_offer="$g, $c, ${dc:-any DC}, $model"
      if ! price_ok "$price"; then
        log "$g ($c, ${dc:-any DC}) costs $price USD/h > $MAX_USD_PER_HR: deleting $id"
        end_pod
        continue
      fi
      # the create's memoryInGb, else the pod's own (RunPod may fill it in after the create)
      ram="$(json_field "$out" memoryInGb)"
      [[ -n "$ram" ]] || ram="$(json_field "$(api get "$id" 2>>"$SESSION_LOG" || echo '{}')" memoryInGb)"
      if ! ram_ok "$ram"; then
        log "$g ($c, ${dc:-any DC}) has ${ram:-unreported} GB of RAM < MIN_RAM_GB=$MIN_RAM_GB: deleting $id"
        end_pod
        continue
      fi
      if [[ "$env_bad" == *" $machine "* ]]; then
        log "$id landed on machine $machine, excluded by BAD_MACHINES: deleting it"
        end_pod
        continue
      fi
      if [[ "$bad_machines" == *" $machine "* ]]; then
        log "$id landed on machine $machine, which failed the CUDA probe or the qualification before: deleting it"
        end_pod
        continue
      fi
      echo "$out" > "$OUT/pod.json"
      log "created $pod on machine ${machine:-?} (the $pod_tag payload, $model): $out"
      return 0
    done
  done
  return 1
}

ssh_opts=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15
          -o ServerAliveInterval=15 -o LogLevel=ERROR)

# The results as two archives made on the pod (once `scp -r` of the out dir's thousands of dump files hung 30
# minutes until the watchdog): out-logs.tar.gz (everything but the *.bin dumps), pulled first, then out-dumps.tar.gz
# only when it is at most PULL_CAP_MB (PULL_CAP_MB=0: an empty one, not pulled, the dumps stay on the pod); each
# step under `timeout`, all of them within PULL_S. Logs the time and size. tar's exit 1 (a file changed while it was read)
# still leaves a whole archive: a run still going is pulled as it is.
pull_results() {
  local t0 rc=0 sizes logs_b dumps_b left dumps_find="find out -type f -name '*.bin' -print0"
  pulled=1
  t0=$(date +%s)
  (( PULL_CAP_MB > 0 )) || dumps_find="printf ''"
  sizes="$(timeout "$PULL_TAR_S" ssh "${ssh_opts[@]}" -p "$port" "root@$ip" \
    "cd $REMOTE && { find out -type f ! -name '*.bin' -print0 | tar --null -czf out-logs.tar.gz -T - || [ \$? -eq 1 ]; } &&
     { $dumps_find | tar --null -czf out-dumps.tar.gz -T - || [ \$? -eq 1 ]; } &&
     stat -c %s out-logs.tar.gz out-dumps.tar.gz")" || rc=$?
  if (( rc != 0 )); then log "WARNING: archiving $REMOTE/out on the pod failed (exit $rc): nothing pulled"; return 0; fi
  read -r logs_b dumps_b <<<"$(tr '\n' ' ' <<<"$sizes")"
  [[ "$logs_b" =~ ^[0-9]+$ && "$dumps_b" =~ ^[0-9]+$ ]] || { log "WARNING: the pod's archive sizes are '$sizes': nothing pulled"; return 0; }
  left=$(( PULL_S - ($(date +%s) - t0) ))
  rc=0; timeout "$left" scp "${ssh_opts[@]}" -P "$port" "root@$ip:$REMOTE/out-logs.tar.gz" "$OUT/" || rc=$?
  # a partial archive would overwrite the mid-run pulls' files with cut ones (pull_steps)
  (( rc == 0 )) || { log "WARNING: pulling out-logs.tar.gz failed (exit $rc)"; rm -f "$OUT/out-logs.tar.gz"; }
  if (( PULL_CAP_MB == 0 )); then
    # the empty archive is not worth an scp (and its time out of PULL_S)
    log "PULL_CAP_MB=0: the dumps stay on the pod, out-dumps.tar.gz not pulled"
    dumps_b=0
  elif (( dumps_b > PULL_CAP_MB * 1024 * 1024 )); then
    log "out-dumps.tar.gz is $(( dumps_b >> 20 )) MiB, above PULL_CAP_MB=$PULL_CAP_MB: left on the pod (the parity verdicts are in the logs)"
    dumps_b=0
  else
    left=$(( PULL_S - ($(date +%s) - t0) ))
    rc=0; (( left > 0 )) && timeout "$left" scp "${ssh_opts[@]}" -P "$port" "root@$ip:$REMOTE/out-dumps.tar.gz" "$OUT/" || rc=$?
    (( rc == 0 && left > 0 )) || { log "WARNING: pulling out-dumps.tar.gz failed or ran out of time (exit $rc, $left s left)"; dumps_b=0; }
  fi
  log "pull: $(( $(date +%s) - t0 )) s, logs $(( logs_b >> 10 )) KiB, dumps $(( dumps_b >> 20 )) MiB (cap $PULL_CAP_MB MiB, limit $PULL_S s)"
}

# pull_steps (run_detached with PULL_STEPS=1): the pod's out/ as it is mid-run, extracted into OUT/out, so a pod
# lost later keeps locally every step that finished (the 2026-10-18 cdet pod went mid lm-eval, after 7 steps
# exit 0, and the one pull at the end brought nothing back). One fresh SSH connection, as a poll's, under the poll's
# timeout POLL_SSH_S, streams a tar of every file but the *.bin dumps and any file above PULL_STEPS_MAX_MB (named in
# the log; the final pull still takes it). Its cost: the 2026-10-17 cdet session's final pull of the same files (818
# KiB, two connections) took 7 s; POLL_SSH_S bounds it. A failure is a WARNING. Sets pulled_step (the last line of
# the copy's out/steps.log).
PULL_STEPS_MAX_MB=64
pulled_step="" pull_skipped=""
pull_steps() {
  local t0 rc=0 part="$OUT/out-steps.tar.gz.part" err l
  t0=$(date +%s)
  err="$(timeout "$POLL_SSH_S" ssh "${ssh_opts[@]}" -p "$port" "root@$ip" \
    "cd $REMOTE && find out -type f ! -name '*.bin' -size +${PULL_STEPS_MAX_MB}M -printf 'above $PULL_STEPS_MAX_MB MiB, left for the final pull: %p\n' >&2 &&
     { find out -type f ! -name '*.bin' ! -size +${PULL_STEPS_MAX_MB}M -print0 | tar --null -czf - -T - || [ \$? -eq 1 ]; }" \
    2>&1 > "$part")" || rc=$?
  if (( rc != 0 )); then
    log "WARNING: pulling out/ mid-run failed (exit $rc, $(( $(date +%s) - t0 )) s): ${err:-no message}; the run goes on"
    rm -f "$part"
    return 0
  fi
  [[ "$err" == "$pull_skipped" ]] || while IFS= read -r l; do [[ -z "$l" ]] || log "pull mid-run: $l"; done <<<"$err"
  pull_skipped="$err"
  rc=0; tar -xzf "$part" -C "$OUT" || rc=$?
  if (( rc != 0 )); then log "WARNING: the mid-run pull's archive does not extract (exit $rc); the run goes on"; rm -f "$part"; return 0; fi
  pulled_step="$(tail -n 1 "$OUT/out/steps.log" 2>/dev/null || true)"
  log "pulled out/ mid-run after '${pulled_step:-no steps.log line}': $(( $(date +%s) - t0 )) s, $(( $(stat -c %s "$part") >> 10 )) KiB"
  rm -f "$part"
}

# qualify: the machine qualification of the pod up with SSH (QUALIFY=1), its lines into probe.log and the session
# log; returns 0 when its last line is the PASS
qualify() {
  local rc=0 res fails l hr
  hr="$(qualify_host_read "$pod_gpu")"
  echo "qualification: $pod ($pod_gpu): host read floor ${hr%%|*} GB/s (${hr#*|})" >> "$OUT/probe.log"
  log "qualification: $pod ($pod_gpu): host read floor ${hr%%|*} GB/s (${hr#*|})"
  scp "${ssh_opts[@]}" -P "$port" "${stages[$pod_tag]}/dsv41_machine_probe" "root@$ip:/root/dsv41_machine_probe" || rc=$?
  if (( rc != 0 )); then log "the machine qualification: scp of dsv41_machine_probe failed (exit $rc)"; return 1; fi
  res="$(ssh "${ssh_opts[@]}" -p "$port" "root@$ip" "timeout $QUALIFY_S /root/dsv41_machine_probe $QUALIFY_ARGS --min-host-read-gbs ${hr%%|*}" 2>&1)" || rc=$?
  printf '%s\n' "$res" >> "$OUT/probe.log"
  while IFS= read -r l; do log "$l"; done <<<"$res"
  if ! (( rc == 0 )) || [[ "$(tail -n 1 <<<"$res")" != "dsv41_machine_probe: PASS"* ]]; then
    # the measures past their bounds, or the CUDA error that stopped the probe
    fails="$(grep -E '^dsv41_machine_probe: (measure .* FAIL|FAIL [^,])' <<<"$res" | sed 's/^dsv41_machine_probe: //' |
      paste -sd';' | sed 's/;/; /g' || true)"
    log "the machine qualification failed on $pod (machine ${machine:-?}, exit $rc): ${fails:-no FAIL line, its last: $(tail -n 1 <<<"$res")}"
    return 1
  fi
  [[ -z "${CPU_EXPERTS_FLOOR_GBS:-}" ]] || qualify_cpu_experts
}

# qualify_cpu_experts (qualify, CPU_EXPERTS_FLOOR_GBS set): the strata tarball's
# dsv41_cpu_experts_probe (stages[pod_tag]) with CPU_EXPERTS_THREADS pool threads, bound to the GPU's NUMA node
# (nvidia-smi topo -m's GPU0 row; unbound without one), its lines into probe.log and the session log; returns 0 when
# its last line is the PASS. The probe's CPUs are /sys/devices/system/node/nodeN/cpulist within its inherited affinity
# (sched_setaffinity), its memory first-touched under the default local policy: the CPU set every form of e07-run.sh's
# numa_binding (t04a-lib.sh) gives the engine, since numactl --cpunodebind takes the same sysfs node and the
# kernel narrows taskset's list to the cgroup's cpuset; on one node both are every allowed CPU.
qualify_cpu_experts() {
  local rc=0 res gbs l why
  # e07_report.py reads this line as "the CPU experts probe was due" (a pod whose probe printed nothing is rejected)
  echo "qualification: $pod ($pod_gpu): CPU experts floor $CPU_EXPERTS_FLOOR_GBS GB/s (CPU_EXPERTS_FLOOR_GBS), $CPU_EXPERTS_THREADS pool threads" >> "$OUT/probe.log"
  log "qualification: $pod ($pod_gpu): CPU experts floor $CPU_EXPERTS_FLOOR_GBS GB/s (CPU_EXPERTS_FLOOR_GBS), $CPU_EXPERTS_THREADS pool threads"
  scp "${ssh_opts[@]}" -P "$port" "${stages[$pod_tag]}/dsv41_cpu_experts_probe" "root@$ip:/root/dsv41_cpu_experts_probe" || rc=$?
  if (( rc != 0 )); then log "the machine qualification: scp of dsv41_cpu_experts_probe failed (exit $rc)"; return 1; fi
  res="$(ssh "${ssh_opts[@]}" -p "$port" "root@$ip" "node=\$(nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/ { print \$(NF-1) }')
    case \"\$node\" in ''|*[!0-9]*) bind='' ;; *) bind=\"--numa-node \$node\" ;; esac
    timeout $QUALIFY_S /root/dsv41_cpu_experts_probe --threads $CPU_EXPERTS_THREADS --min-gbs $CPU_EXPERTS_FLOOR_GBS \$bind" 2>&1)" || rc=$?
  printf '%s\n' "$res" >> "$OUT/probe.log"
  while IFS= read -r l; do log "$l"; done <<<"$res"
  (( rc == 0 )) && [[ "$(tail -n 1 <<<"$res")" == "dsv41_cpu_experts_probe: PASS"* ]] && return 0
  gbs="$(sed -n 's/^dsv41_cpu_experts_probe: measure cpu_experts_gbs \([0-9.]*\) .*/\1/p' <<<"$res")"
  if [[ -n "$gbs" ]]; then
    why="qualification: CPU experts $gbs GB/s < $CPU_EXPERTS_FLOOR_GBS on $pod (machine ${machine:-?}, exit $rc; CPU_EXPERTS_FLOOR_GBS): the machine qualification failed"
  else
    why="qualification: CPU experts not measured on $pod (machine ${machine:-?}, exit $rc): $(tail -n 1 <<<"$res"): the machine qualification failed"
  fi
  echo "$why" >> "$OUT/probe.log"
  log "$why"
  return 1
}

# pod_up: a pod at or under the price, with SSH, a passing CUDA probe and (QUALIFY=1) a passing machine
# qualification, its watchdog armed at the cap; sets ip, port and deadline (the cap, unix seconds). Up to MAX_PODS
# pods: a broken or unfit one is deleted and its machine remembered.
ip="" port="" deadline=0
pod_up() {
  local attempt cap_min t_wait ssh_up info rc
  for (( attempt = 1; attempt <= MAX_PODS; ++attempt )); do
    create_pod || { log "FAIL: no GPU of '$GPU_TYPES' under $MAX_USD_PER_HR USD/h on a working machine"; exit 1; }
    cap_min="$(cap_minutes "$cost" "$spent")"
    (( cap_min >= 5 )) || { log "FAIL: $spent USD spent, $cap_min min left of $MAX_SESSION_USD USD at $cost USD/h"; exit 1; }
    deadline=$(( t_create + cap_min * 60 ))
    log "pod $attempt of at most $MAX_PODS: hard cap $cap_min min at $cost USD/h (until $(date -u -d "@$deadline" +%H:%M:%S) UTC)"
    # the watchdog: at the cap the pod goes, then this script
    ( sleep $(( deadline - $(date +%s) )); touch "$OUT/watchdog-fired"; echo "[watchdog] cap reached: deleting $pod" >&2
      api delete "$pod" >&2; kill -TERM $$ ) &
    watchdog=$!

    # a host that never boots sshd is as broken as one whose CUDA probe fails: both cost a pod, not the session
    t_wait=$(date +%s) ssh_up=0
    while (( $(date +%s) - t_wait < SSH_WAIT_S )); do
      info="$(api get "$pod" 2>>"$SESSION_LOG" || true)"
      ip="$(json_field "$info" publicIp 2>/dev/null || true)"
      port="$(json_field "$info" port22 2>/dev/null || true)"
      if [[ -n "$ip" && -n "$port" ]] && ssh "${ssh_opts[@]}" -p "$port" "root@$ip" true 2>/dev/null; then ssh_up=1; break; fi
      sleep 10
    done
    if (( ssh_up )); then
      log "ssh root@$ip -p $port ready after $(( $(date +%s) - t_create )) s"
      # the CUDA probe before anything else goes up, then the machine qualification
      echo "=== pod $pod machine ${machine:-?} ($pod_offer) at $(date -u +%H:%M:%S) UTC" >> "$OUT/probe.log"
      rc=0
      scp "${ssh_opts[@]}" -P "$port" "${stages[$pod_tag]}/dsv41_cuda_probe" "root@$ip:/root/dsv41_cuda_probe" || rc=$?
      (( rc != 0 )) || ssh "${ssh_opts[@]}" -p "$port" "root@$ip" "timeout $PROBE_S /root/dsv41_cuda_probe" 2>&1 | tee -a "$OUT/probe.log" >&2 || rc=$?
      if (( rc == 0 )) && [[ "$(tail -n 1 "$OUT/probe.log")" == "dsv41_cuda_probe: ok"* ]]; then
        # the pod's memory cgroup and MemTotal, which a later dry run's gate takes
        # (e07.sh PROBE_LOG); nothing decides on them here
        ssh "${ssh_opts[@]}" -p "$port" "root@$ip" 'for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
          [ -r "$f" ] && echo "pod memory cgroup: $f: $(cat "$f")"; done; grep "^MemTotal:" /proc/meminfo' >> "$OUT/probe.log" 2>/dev/null || true
        (( ${QUALIFY:-0} )) || return 0
        qualify && return 0
        log "$pod (machine ${machine:-?}) is unfit: deleting it, remembering the machine and trying the next offer"
      else
        log "the CUDA probe failed on $pod (machine ${machine:-?}): deleting it and trying the next offer"
      fi
    else
      log "no SSH login within $SSH_WAIT_S s on $pod (machine ${machine:-?}): deleting it and trying the next offer"
    fi
    bad_machines+="$machine "
    end_pod
    (( attempt < MAX_PODS )) || { log "FAIL: $MAX_PODS pods and none came up with SSH and a passing CUDA probe$( (( ${QUALIFY:-0} )) &&
      echo " and machine qualification") (machines tried: $(b="${bad_machines#"$env_bad"}"; echo "${b% }")): nothing uploaded"; exit 1; }
  done
}

# run_detached PAYLOAD: the payload up and RUN_SCRIPT started on the pod, detached: setsid and nohup keep it alive
# when this SSH session (or a later one) drops, every output in out/, its exit status in out/<RUN_SCRIPT>.exit; only
# the launch is in the background (the braces), so a mkdir or tar failure still fails this SSH command. Then a poll
# every POLL_S s over a fresh connection until it exits, POLL_FAILS polls fail in a row, or 30 s past its deadline
# (the cap less 120 s, which it gets as DEADLINE). With PULL_STEPS=1 (e07.sh) a poll that sees a new last line of
# out/steps.log, the run still going, pulls out/ (pull_steps) unless the pull's timeout would end past the run's
# deadline (the final pull follows then). Sets run_rc (its exit status, "" when none came) and polled_step (the last
# line of out/steps.log a poll saw).
run_rc="" polled_step=""
run_detached() {
  local run_deadline=$(( deadline - 120 )) rc=0 fails=0 state step exit_file="out/${RUN_SCRIPT%.sh}.exit"
  scp "${ssh_opts[@]}" -P "$port" "$1" "root@$ip:$REMOTE.tar.gz"
  started=1
  timeout "$START_S" ssh "${ssh_opts[@]}" -p "$port" "root@$ip" \
    "mkdir -p $REMOTE/out && tar -xzf $REMOTE.tar.gz -C $REMOTE && cd $REMOTE &&
     { DEADLINE=$run_deadline POD_VOLUME=$pod_volume POD_MACHINE=$machine nohup setsid bash -c 'bash $RUN_SCRIPT; echo \$? > $exit_file' \
       > out/${RUN_SCRIPT%.sh}.out 2>&1 < /dev/null & }" || rc=$?
  if (( rc != 0 )); then
    log "FAIL: starting $RUN_SCRIPT on the pod (exit $rc)"
  else
    log "$RUN_SCRIPT started on the pod, detached (deadline $(date -u -d "@$run_deadline" +%H:%M:%S) UTC)"
    # its exit status or "running", then out/steps.log's last line; every step ends by the deadline
    while :; do
      sleep "$POLL_S"
      rc=0; state="$(timeout "$POLL_SSH_S" ssh "${ssh_opts[@]}" -p "$port" "root@$ip" \
        "cat $REMOTE/$exit_file 2>/dev/null || echo running; tail -n 1 $REMOTE/out/steps.log 2>/dev/null")" || rc=$?
      if (( rc != 0 )); then
        fails=$(( fails + 1 ))
        log "WARNING: poll failed (exit $rc), $fails in a row of at most $POLL_FAILS"
        (( fails < POLL_FAILS )) || { log "FAIL: the pod did not answer $POLL_FAILS polls in a row"; break; }
      else
        fails=0
        step="$(sed -n 2p <<<"$state")"
        [[ -z "$step" || "$step" == "$polled_step" ]] || log "on the pod: $step"
        if [[ "$(head -n 1 <<<"$state")" =~ ^[0-9]+$ ]]; then polled_step="$step" run_rc="$(head -n 1 <<<"$state")"; break; fi
        if (( ${PULL_STEPS:-0} )) && [[ -n "$step" && "$step" != "$polled_step" ]]; then
          if (( $(date +%s) + POLL_SSH_S < run_deadline )); then pull_steps
          else log "out/ not pulled after '$step': its $POLL_SSH_S s would pass the run's deadline (the final pull follows)"
          fi
        fi
        polled_step="$step"
      fi
      (( $(date +%s) < run_deadline + 30 )) || { log "FAIL: $RUN_SCRIPT still not done 30 s after its deadline"; break; }
    done
  fi
  log "$RUN_SCRIPT: $([[ -n "$run_rc" ]] && echo "exit $run_rc" || echo "no exit status")"
}

# steps_verdict: the session's verdict from the remote script's exit and out/steps.log (every step exit 0, then
# "done"; "note: " lines, a run script's policy notes, pass), each failure logged; returns 1 on any
steps_verdict() {
  local fail=0 bad
  [[ "$run_rc" == 0 ]] || { log "FAIL: $RUN_SCRIPT $([[ -n "$run_rc" ]] && echo "exited $run_rc" || echo "reported no exit status")"; fail=1; }
  if [[ ! -f "$OUT/out/steps.log" ]]; then
    log "FAIL: no out/steps.log came back"; fail=1
  else
    bad="$(grep -v -E '^(step [^:]+: exit 0, [0-9]+ s|note: .*|done)$' "$OUT/out/steps.log" || true)"
    [[ -z "$bad" ]] || { log "FAIL: steps that did not pass: $(tr '\n' ';' <<<"$bad")"; fail=1; }
    [[ "$(tail -n 1 "$OUT/out/steps.log")" == done ]] || { log "FAIL: out/steps.log does not end with done"; fail=1; }
  fi
  return "$fail"
}
