#!/usr/bin/env bash
# Rehearses the quality pod sessions without a GPU (the first qfork session died after setup on `PLACE: unbound
# variable`, a path of e07-run.sh no test had run): the real tools/dsv41/pod/e07-run.sh from
# the real --payload-only payloads of tools/dsv41/pod/e07.sh, end to end, with stubs only where a pod has hardware,
# the model or the network. No GPU, no model, no RunPod, no HF, no money.
#   1. e07.sh --payload-only builds the qfork payload (A100's flavor), the qengine one and the qdiag one (the RTX PRO
#      6000's), and e07.sh --dry-run prints each session's plan (qfork from the volume
#      [gsm8k_cot_llama,humaneval_instruct + 16K needle] [ifeval], from HF [gsm8k_cot_llama]
#      [humaneval_instruct + 16K needle] [ifeval]; qengine one session, volume and HF; qdiag one session, volume and
#      HF): its pod steps are the steps the rehearsal expects
#   2. each session in its own directory: the payload unpacked (less the CUDA libraries nothing loads here), its
#      e07/qtasks and e07/qneedles written as e07.sh writes them (the plan's estimates kept: they set the timeouts),
#      E07_POD_ROOT in place of the pod's / (the volume's 7 shards under workspace/, the model's copy and the venv
#      under root/), then e07-run.sh as pod-session.sh starts it (DEADLINE, POD_VOLUME, POD_MACHINE,
#      out/e07-run.out); the 12 sessions run at once, each on its own server and proxy ports; qengine-slow is
#      qengine-volume's plan with the fake engine made slow (FAKE_ENGINE_STEP_MS=45, try 2's 44.2 ms per step);
#      qfork-he is the HumanEval-only session (HF, Q_TASKS=humaneval_instruct, Q_NEEDLES empty) and
#      qfork-he-slow the same with the fake fork made slow (FAKE_FORK_MS_PER_TOKEN=400, the 2026-10-11 session's slots
#      ran 334-1002 ms per token)
#   3. each session: exit 0, out/steps.log has every planned step in the plan's order, each "exit 0", then "done";
#      qengine-slow: exit 1, the planned steps up to speed-check, speed-check exit 1,
#      then "stop: engine too slow on this machine: 45.0 ms per decode step (machine q02-qengine-slow)", volume-after and
#      "done", the server stopped and no lm-eval step run; qfork-he-slow the same with "stop: fork too slow on this
#      machine: 400.0 ms per token (machine q02-qfork-he-slow)" (HF: no volume-after); every qfork session's
#      speed-check passed at the fake fork's 52.5 ms per token, its server ran -np 1 with --threads min(16, cpus)
#      and --threads-batch min(32, cpus), and its lm-eval runs NUM_CONCURRENT 1; cpus
#      is nproc capped by the cgroup's CPU quota, each qfork session on its own fake cgroup root and
#      nproc (fork_cpus: qfork-he is the 2026-10-11 host, an 18 CPU quota on nproc 128, so 16 and 18; v1 and v2
#      files, no quota, a quota under one CPU); a cpu.max that is not a quota stops pod_server.sh fork with exit 1;
#      placement.txt says qfork has no placement and gives qengine and qdiag the gate's; qengine and qdiag have the
#      samplers' rows (pod-samplers.sh: out/gpu-clocks.csv past its header with the throttle field the stub's driver
#      knows, out/host-load.log's blocks) and stopped them; qdiag ran bench-qdiag's two phases with their windows and
#      the same 20 GSM8K items at NUM_CONCURRENT 1 and 2
#   4. pod/qdiag_report.py on each qdiag session writes every row (the two bench phases, continuation, c1, c2, all
#      server requests), every engine request matched to a proxy response and the stub GPU's clocks, power and
#      throttle reason in the server phases; e07_report.py puts it on the qdiag sheet and the GPU clock summary on
#      the qengine one; then the report on qengine try 2's real logs (TRY2_RUN's out-logs.tar.gz, skipped when
#      missing) reproduces that session's numbers: 201 requests, 66.9 ms per token, 60.0 ms of doorbell waits
#      per step, hit rate 0.917
#   5. pod/speed_check.py on the real logs, the first 3 requests (the continuation's):
#      qengine try 2's (TRY2_RUN) exits 1 with the stop line at 44.2 ms per decode step;
#      qdiag's (QDIAG_RUN) exits 0 at 20.2; each skipped when its log is missing. Then
#      speed_check.py --fork on the 2026-10-11 qfork session's llama-server log
#      (FORK_RUN, -np 4 on an 18 vCPU A100 SXM4 host): its first 3 replies (the continuation's, one request at a time)
#      pass at 70.8 ms per token; its first 3 lm-eval replies (4 concurrent) exit 1 with the stop line at 444.2; a log
#      of t04a's stock baseline replies (18.92 and 19.15 tok/s: 52.85 and 52.22 ms per token) exits 0 at 52.5. So
#      the threshold stays 80 ms per token, the
#      check is there for machines several times slower, and that machine at one request at a time passes it
#   6. the machine qualification: the real e07.sh session (pod-session.sh's
#      offers, create, SSH wait, CUDA probe, qualification, delete, payload upload, polls, pull, sheet) against a fake
#      RunPod (RUNPOD_API), ssh and scp stubs that run the pod's commands in a directory per pod, and a stub
#      dsv41_machine_probe that prints the real one's lines with each fake machine's numbers against the bounds it is
#      given, resolved as the real one does (the GPU read's per model, the SM clock's and the power limit's as ratios
#      of the GPU's own maximums). "rejected": three unfit machines (doorbell; GPU read, SM clock and power limit;
#      host memory) end a qengine session at MAX_PODS 3 with the FAIL line, each deleted and remembered, the log naming
#      the measure and its value, no payload uploaded, the sheet carrying the three; "passes": an unfit machine, the
#      same machine again (deleted at the create, no probe), then a good one, whose session proceeds as before (the
#      payload up, the run, the pull, exit 0), the sheet carrying both; "a100": an A100 SXM4 machine with an A100's
#      numbers (1410 MHz, 400 W: the first absolute bounds rejected every A100) passes on the relative bounds and its
#      model's 900 GB/s read bound and runs the cu128 payload
# The stubs: on PATH nvidia-smi (Q02_VRAM_MIB of memory, the NUMA node Q02_NUMA: 0 for the volume sessions, so their
# servers run under numactl on a fake two-node NUMA_ROOT, none for HF; the samplers' fields with fixed values, its throttle field Q02_THROTTLE:
# clocks_event_reasons.active on the volume sessions, the older clocks_throttle_reasons.active on HF), numactl (runs its command), pip (records its call), aria2c and apt-get
# (fail: nothing may download or install); in the payload dsv41-engine (--dry-run exits 0, --serve is fakes.py's
# engine, which opens the gate's --hot-set from its own cwd as the real one does), fork/llama-server (fakes.py llama-server), hf_model.py headers (the real heads from HEADERS),
# hf-download.sh and model-copy.sh (the 7 shards of gguf_layout.py sparse: the real headers and sizes, holes for the
# weights, so gguf_layout.py sizes and the eos lookup run for real) and e07_gate.py (the payload's own gate with a
# 256 GB host's memory, as e07.sh's dry run takes it, instead of this machine's). lm-eval runs for real from
# QUALITY_VENV, offline (HF_HUB_OFFLINE: the datasets from its cache), on 3 items per task (eval.sh's SAMPLES).
# Exits 1 if any check fails.
#
#   QUALITY_VENV=<a venv with lm-eval 0.4.13> bash tools/dsv41/quality/q02-test.sh
#
# q01-test.sh runs it (its section 9). QUALITY_VENV defaults to $BUILD_ROOT/quality/venv, HEADERS to
# $BUILD_ROOT/engine/headers (e07.sh's), TRY2_RUN to $BUILD_ROOT/engine/e07a-pod/20261012T112049Z (qengine try 2),
# QDIAG_RUN to $BUILD_ROOT/engine/e07a-pod/20261012T190817Z (the qdiag session), FORK_RUN to
# $BUILD_ROOT/engine/e07a-pod/20261011T151030Z (the 2026-10-11 qfork session);
# QUALITY_NEEDLE (the needle prompts) goes through to e07.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
# e07.sh has no default network volume: its dry runs and rehearsals here plan one by this placeholder id
export NETWORK_VOLUME="${NETWORK_VOLUME-testvolume01}"
QUALITY_VENV="${QUALITY_VENV:-$BUILD_ROOT/quality/venv}"
HEADERS="${HEADERS:-$BUILD_ROOT/engine/headers}"
TRY2_RUN="${TRY2_RUN:-$BUILD_ROOT/engine/e07a-pod/20261012T112049Z}"
QDIAG_RUN="${QDIAG_RUN:-$BUILD_ROOT/engine/e07a-pod/20261012T190817Z}"
FORK_RUN="${FORK_RUN:-$BUILD_ROOT/engine/e07a-pod/20261011T151030Z}"
[[ -x "$QUALITY_VENV/bin/lm_eval" ]] || { echo "q02-test.sh: no lm_eval in $QUALITY_VENV (QUALITY_VENV)" >&2; exit 2; }
tmp="$(mktemp -d)"
jobs_=()
cleanup() {
  local f
  for f in "${jobs_[@]}"; do kill "$f" 2>/dev/null || true; done
  # a session that died with its server up leaves it in its own session (pod_server.sh's setsid)
  for f in "$tmp"/s-*/pod/out/quality-server.pid "$tmp"/s-*/pod/out/quality-gpu.pid; do
    [[ -f "$f" ]] && { kill -TERM -- "-$(cat "$f")" 2>/dev/null || kill -TERM "$(cat "$f")" 2>/dev/null || true; }
  done
  wait 2>/dev/null || true
  rm -rf -- "${tmp:?}"
}
trap cleanup EXIT
failed=0
check() {
  if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; failed=$((failed + 1)); fi
}
ports=()
free_port() {   # free_port VAR: a free port none of this test's other servers got, into VAR
  local p
  while :; do
    p="$(python3 -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")"
    [[ " ${ports[*]} " == *" $p "* ]] || break
  done
  ports+=("$p")
  printf -v "$1" %s "$p"
}

# name|model path|plan|Q_TASKS|Q_NEEDLES[|slow: the fake engine's ms per decode step (default 15), or for qfork the fake
# fork's ms per token (default 52.5)]
sessions=("qfork-volume-1|volume|qfork|gsm8k_cot_llama,humaneval_instruct|16k" "qfork-volume-2|volume|qfork|ifeval|"
  "qfork-hf-1|hf|qfork|gsm8k_cot_llama|" "qfork-hf-2|hf|qfork|humaneval_instruct|16k" "qfork-hf-3|hf|qfork|ifeval|"
  "qengine-volume|volume|qengine|gsm8k_cot_llama,ifeval,humaneval_instruct|16k 64k 256k"
  "qengine-hf|hf|qengine|gsm8k_cot_llama,ifeval,humaneval_instruct|16k 64k 256k"
  "qdiag-volume|volume|qdiag||" "qdiag-hf|hf|qdiag||"
  "qengine-slow|volume|qengine|gsm8k_cot_llama,ifeval,humaneval_instruct|16k 64k 256k|45"
  "qfork-he|hf|qfork|humaneval_instruct|" "qfork-he-slow|hf|qfork|humaneval_instruct||400")
path_vars() { [[ "$1" == volume ]] && echo "HF_CLOUDS=" || echo "NETWORK_VOLUME="; }
# the fork's cpus are nproc capped by the cgroup's CPU quota. Each qfork
# session has its own cgroup root (pod_server.sh's Q_FORK_CGROUP, empty when not listed here) and, where nproc is
# given, an nproc stub (else this machine's):
# name|nproc|the cgroup's files (path=content, ';' between)|the cpus that follow|their source
fork_cpus=("qfork-he|128|cpu.max=1800000 100000|18|the cgroup quota"   # the 2026-10-11 host: --threads 16 --threads-batch 18
  "qfork-hf-2|128|cpu/cpu.cfs_quota_us=1800000;cpu/cpu.cfs_period_us=100000|18|the cgroup quota"
  "qfork-he-slow|128|cpu.max=max 100000|128|nproc" "qfork-hf-1||cpu/cpu.cfs_quota_us=-1;cpu/cpu.cfs_period_us=100000|$(nproc)|nproc"
  "qfork-hf-3|128|cpu.max=50000 100000|1|the cgroup quota")
fork_cpu_case() {   # fork_cpu_case NAME: its fork_cpus row, or the default one (no cgroup files, this machine's nproc)
  local c
  for c in "${fork_cpus[@]}"; do [[ "${c%%|*}" == "$1" ]] && { echo "$c"; return; }; done
  echo "$1|||$(nproc)|nproc"
}

echo "--- 1. the payloads (e07.sh --payload-only) and each session's plan (e07.sh --dry-run)"
for p in qfork qengine qdiag; do
  ( E07_PLAN=$p OUT="$tmp/payload-$p" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --payload-only > "$tmp/payload-$p.log" 2>&1
    echo $? > "$tmp/payload-$p.rc"; rm -rf -- "$tmp/payload-$p"/payload-*/ ) &
  jobs_+=($!)
done
for s in "${sessions[@]}"; do
  IFS='|' read -r name path plan tasks needles slow <<<"$s"
  rc=0; env E07_PLAN="$plan" Q_TASKS="$tasks" Q_NEEDLES="$needles" "$(path_vars "$path")" OUT="$tmp/dry-out" \
    bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry-$name.txt" 2>&1 || rc=$?
  # the pod's steps of the plan for that model path, without the session's own (create, upload, pull)
  head="$([[ $path == volume ]] && echo "the volume" || echo HF)"
  sed -n "/^plan with the model from $head /,/^plan ($path)/p" "$tmp/dry-$name.txt" | awk '/^  / { print $1 }' |
    grep -vxE 'create,|payload|pull' > "$tmp/planned-$name.txt" || true
  before=$failed
  check "$name: the dry run exits 0 and plans $(wc -l < "$tmp/planned-$name.txt") pod steps" '(( rc == 0 )) && [[ -s "$tmp/planned-$name.txt" ]]'
  (( failed == before )) || { echo "  $name: the dry run's exit $rc, its plan's steps $(tr '\n' ' ' < "$tmp/planned-$name.txt"), its tail:"; tail -n 5 "$tmp/dry-$name.txt" | sed 's/^/    /'; }
done
# the host read floor per GPU, 40 for the A100 (qfork's only), 250 for the
# RTX PRO 6000 (qengine's), the one QUALIFY_MIN_HOST_READ_GBS set by hand for every GPU of a mixed plan
rc=0; env E07_PLAN=cycle2 QUALIFY_MIN_HOST_READ_GBS=120 OUT="$tmp/dry-out" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run \
  > "$tmp/dry-hostread-override.txt" 2>&1 || rc=$?
grep -h "qualification:   16-thread host read" "$tmp/dry-qfork-hf-1.txt" "$tmp/dry-qengine-hf.txt" "$tmp/dry-hostread-override.txt" |
  sed 's/^ */  host read: /'
check "the dry runs print the host read floor per GPU: qfork's A100s 40, qengine's RTX PRO 6000 250, 120 by hand for all" \
  'hr="qualification:   16-thread host read >=" am="),"
   grep -qx "  $hr 40 GB/s on NVIDIA A100-SXM4-80GB (the A100 default, its hosts measured 40.9-162.2$am" "$tmp/dry-qfork-hf-1.txt" &&
   grep -qx "  $hr 40 GB/s on NVIDIA A100 80GB PCIe (the A100 default, its hosts measured 40.9-162.2$am" "$tmp/dry-qfork-hf-1.txt" &&
   (( $(grep -c "$hr" "$tmp/dry-qfork-hf-1.txt") == 2 )) &&
   grep -q "^  $hr 250 GB/s on NVIDIA RTX PRO 6000 Blackwell .*(the default for any GPU but the A100, the RTX PRO 6000.s good hosts measured 341.8-545.4$am$" "$tmp/dry-qengine-hf.txt" &&
   ! grep "$hr" "$tmp/dry-qengine-hf.txt" | grep -qv " 250 GB/s on NVIDIA RTX PRO 6000 " &&
   (( rc == 0 )) && (( $(grep -c "$hr 120 GB/s on .*(QUALIFY_MIN_HOST_READ_GBS, set by hand$am" "$tmp/dry-hostread-override.txt") == 4 )) &&
   ! grep "$hr" "$tmp/dry-hostread-override.txt" | grep -qv " 120 GB/s "'
wait "${jobs_[@]}"
jobs_=()
for p in qfork qengine qdiag; do
  grep -E "payload-[a-z0-9]+ [0-9.]+[MG]: |payload only" "$tmp/payload-$p.log" | sed "s|$tmp|TMP|g; s/^/  $p: /"
  check "$p: e07.sh --payload-only exits 0 with one payload" \
    '[[ "$(cat "$tmp/payload-$p.rc")" == 0 ]] && (( $(ls "$tmp/payload-$p"/payload-*.tar.gz | wc -l) == 1 ))'
done
(( failed == 0 )) || { echo "q02-test.sh: FAIL ($failed), no session rehearsed"; tail -n 5 "$tmp"/payload-*.log; exit 1; }

echo "--- 2. the sessions: e07-run.sh from the unpacked payload, with the stubs"
bin="$tmp/bin" stubs="$tmp/stubs"
mkdir -p "$bin" "$stubs" "$tmp/numa/node0" "$tmp/numa/node1"
# e07-run.sh binds only on a host of two nodes or more; node 0's CPUs the ones this shell may run on
grep Cpus_allowed_list /proc/self/status | cut -f2 > "$tmp/numa/node0/cpulist"
echo 4095 > "$tmp/numa/node1/cpulist"
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# q02-test.sh's GPU: Q02_VRAM_MIB of memory, GPU0's NUMA node Q02_NUMA (N/A: none), as nvidia-smi prints them; the
# samplers' fields (pod-samplers.sh) with fixed values, SwPowerCap active, the throttle field Q02_THROTTLE's only
value() {
  case "$1" in
    timestamp) date -u +'%Y/%m/%d %H:%M:%S.000' ;; clocks.sm) echo 2400 ;; clocks.mem) echo 14001 ;;
    power.draw) echo 300.50 ;; enforced.power.limit) echo 600.00 ;; temperature.gpu) echo 55 ;;
    utilization.gpu) echo 90 ;; utilization.memory) echo 40 ;; pcie.link.gen.current) echo 5 ;;
    "$Q02_THROTTLE") echo 0x0000000000000004 ;;
    *) echo "Field \"$1\" is not a valid field to query." >&2; return 2 ;;
  esac
}
case " $* " in
  *" topo "*) printf '\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t0-11\t%s\t\tN/A\n' "$Q02_NUMA" ;;
  *" --query-gpu=timestamp,clocks.sm,"*" -lms "*)
    IFS=, read -ra fields <<<"${1#--query-gpu=}"
    while :; do row=(); for f in "${fields[@]}"; do row+=("$(value "$f")"); done; (IFS=,; echo "${row[*]}") | sed 's/,/, /g'; sleep 1; done ;;
  *" -lms "*) while :; do echo "$(date +'%Y/%m/%d %H:%M:%S.000'), 1024, $Q02_VRAM_MIB"; sleep 1; done ;;
  *" --query-gpu=memory.total --format=csv,noheader,nounits "*) echo "$Q02_VRAM_MIB" ;;
  *" --query-gpu=memory.used,memory.total --format=csv,noheader "*) echo "1024 MiB, $Q02_VRAM_MIB MiB" ;;
  " --query-gpu="*" --format=csv,noheader ") value "${1#--query-gpu=}" ;;
  *) echo "nvidia-smi (q02-test.sh stub): GPU 0, $Q02_VRAM_MIB MiB" ;;
esac
EOF
cat > "$bin/numactl" <<'EOF'
#!/usr/bin/env bash
# q02-test.sh's numactl: the binding options taken, the command run
while [[ "${1:-}" == --* ]]; do shift; done
exec "$@"
EOF
cat > "$bin/pip" <<'EOF'
#!/usr/bin/env bash
echo "pip (q02-test.sh stub, nothing installed): $*" | tee -a "$Q02_LOG"
EOF
for t in aria2c apt-get; do
  printf '#!/usr/bin/env bash\necho "%s (q02-test.sh stub): nothing downloads or installs here: $*" >&2\nexit 1\n' "$t" > "$bin/$t"
done
cat > "$stubs/dsv41-engine" <<'EOF'
#!/usr/bin/env bash
# q02-test.sh's engine: --dry-run exits 0 (the real one runs in verify.sh's image only), --serve (first,
# as serve/server.py passes it, or after the arguments, as e07_bench.py does) is fakes.py's engine (the tokenizer from
# --model)
case " $* " in *" --dry-run "*) echo "dsv41-engine (q02-test.sh stub): dry run of $*"; exit 0 ;; esac
[[ " $* " == *" --serve "* ]] || { echo "dsv41-engine (q02-test.sh stub): --serve or --dry-run, not: $*" >&2; exit 2; }
args=()
for a in "$@"; do [[ "$a" == --serve ]] || args+=("$a"); done
exec python3 "$(dirname "$0")/src/tools/dsv41/quality/fakes.py" --serve "${args[@]}"
EOF
cat > "$stubs/llama-server" <<'EOF'
#!/usr/bin/env bash
# q02-test.sh's fork server: fakes.py llama-server (the tokenizer and template from -m)
exec python3 "$(dirname "$0")/../src/tools/dsv41/quality/fakes.py" llama-server "$@"
EOF
cat > "$stubs/hf_model.py" <<'EOF'
#!/usr/bin/env python3
"""q02-test.sh's hf_model.py: `headers MANIFEST DIR` copies the real heads (Q02_HEADERS/head-N.bin, what the real
one fetches by HTTP range) into DIR; nothing else runs here (hf-download.sh is a stub too)."""
import os
import shutil
import sys

if len(sys.argv) != 4 or sys.argv[1] != "headers":
    sys.exit(f"hf_model.py (q02-test.sh stub): headers MANIFEST DIR only, not {sys.argv[1:]}")
os.makedirs(sys.argv[3], exist_ok=True)
for i in range(1, 8):
    shutil.copy(os.path.join(os.environ["Q02_HEADERS"], f"head-{i}.bin"), sys.argv[3])
    print(f"{sys.argv[3]}/head-{i}.bin: copied from {os.environ['Q02_HEADERS']} (q02-test.sh stub)")
EOF
cat > "$stubs/hf-download.sh" <<'EOF'
#!/usr/bin/env bash
# q02-test.sh's download: hf-download.sh MANIFEST DST JSON, the sparse shards (Q02_MODEL) copied into DST
set -euo pipefail
mkdir -p "$2"
cp --sparse=always "$Q02_MODEL"/*.gguf "$2"/
echo '{"stub": "q02-test.sh: the sparse shards copied, no download"}' > "$3"
echo "hf-download.sh (q02-test.sh stub): $(ls "$2" | wc -l) sparse shards in $2"
EOF
cat > "$stubs/model-copy.sh" <<'EOF'
#!/usr/bin/env bash
# q02-test.sh's copy: model-copy.sh SRC DST, the shards copied sparse (SRC only read)
set -euo pipefail
mkdir -p "$2"
cp --sparse=always "$1"/*.gguf "$2"/
echo "model-copy.sh (q02-test.sh stub): $(ls "$2" | wc -l) shards from $1"
EOF
cat > "$stubs/e07_gate.py" <<'EOF'
#!/usr/bin/env python3
"""q02-test.sh's gate: the payload's own e07_gate.py (e07_gate_real.py beside this file) with a 256 GB host's cgroup
and MemTotal (e07.sh's dry run takes the pod's memory so) in place of this machine's."""
import os
import sys

a = sys.argv[1:]
for key, value in (("--cgroup-bytes", str(256 * 10**9)), ("--mem-total-kib", str(256 * 10**6))):
    a[a.index(key) + 1] = value
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e07_gate_real.py"), *a])
EOF
chmod +x "$bin"/* "$stubs"/*
python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" sparse "$HEADERS" "$tmp/model" > /dev/null
python3 -c "import json, sys; d = json.load(open(sys.argv[1])); json.dump({k: v[:3] for k, v in d.items()}, open(sys.argv[2], 'w'))" \
  "$here/subsets_lean2h_seed20261006.json" "$tmp/samples3.json"
t0=$(date +%s)
for s in "${sessions[@]}"; do
  IFS='|' read -r name path plan tasks needles slow <<<"$s"
  d="$tmp/s-$name"
  mkdir -p "$d/pod/out" "$d/root/root"
  tar -xzf "$tmp/payload-$plan"/payload-*.tar.gz -C "$d/pod" --exclude='libcublas*' --exclude='libggml-cuda.so*'
  if [[ "$plan" != qdiag ]]; then
    echo "${tasks//,/ }" > "$d/pod/e07/qtasks"
    echo "$needles" > "$d/pod/e07/qneedles"
  fi
  cp "$stubs/dsv41-engine" "$stubs/hf_model.py" "$stubs/hf-download.sh" "$stubs/model-copy.sh" "$d/pod/"
  cp "$stubs/llama-server" "$d/pod/fork/"
  mv "$d/pod/src/tools/dsv41/pod/e07_gate.py" "$d/pod/src/tools/dsv41/pod/e07_gate_real.py"
  cp "$stubs/e07_gate.py" "$d/pod/src/tools/dsv41/pod/"
  ln -s "$QUALITY_VENV" "$d/root/root/venv-quality"
  volume="" numa=N/A throttle=clocks_throttle_reasons.active
  if [[ "$path" == volume ]]; then
    volume=q02-rehearsal numa=0 throttle=clocks_event_reasons.active
    mkdir -p "$d/root/workspace/models/q2k"
    cp --sparse=always "$tmp/model"/*.gguf "$d/root/workspace/models/q2k/"
  fi
  vram=81920; [[ "$plan" == qfork ]] || vram=97887
  mkdir -p "$d/cgroup" "$d/bin"
  if [[ "$plan" == qfork ]]; then
    IFS='|' read -r _ ncpu files _ _ <<<"$(fork_cpu_case "$name")"
    IFS=';' read -ra files <<<"$files"
    for f in "${files[@]}"; do mkdir -p "$(dirname "$d/cgroup/${f%%=*}")"; echo "${f#*=}" > "$d/cgroup/${f%%=*}"; done
    [[ -z "$ncpu" ]] || { printf '#!/usr/bin/env bash\necho %s\n' "$ncpu" > "$d/bin/nproc"; chmod +x "$d/bin/nproc"; }
  fi
  free_port server_port; free_port proxy_port
  ( cd "$d/pod" && export PATH="$d/bin:$bin:$PATH" Q_FORK_CGROUP="$d/cgroup" E07_POD_ROOT="$d/root" E07_Q_PORT="$server_port" PROXY_PORT="$proxy_port" \
      NUMA_ROOT="$tmp/numa" SAMPLES="$tmp/samples3.json" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 Q02_VRAM_MIB=$vram Q02_NUMA=$numa \
      Q02_THROTTLE=$throttle Q02_MODEL="$tmp/model" Q02_HEADERS="$HEADERS" Q02_LOG="$d/stubs.log" \
      POD_MACHINE="q02-$name"
    [[ -z "$slow" ]] || { [[ "$plan" == qfork ]] && export FAKE_FORK_MS_PER_TOKEN="$slow" || export FAKE_ENGINE_STEP_MS="$slow"; }
    rc=0; DEADLINE=$(( $(date +%s) + 1200 )) POD_VOLUME=$volume timeout 1500 bash e07-run.sh > out/e07-run.out 2>&1 < /dev/null || rc=$?
    echo "$rc" > out/e07-run.exit ) &
  jobs_+=($!)
done
wait "${jobs_[@]}"
jobs_=()
echo "  the ${#sessions[@]} sessions took $(( $(date +%s) - t0 )) s"

echo "--- 3. every session: exit 0, every planned step in order with exit 0, then done"
for s in "${sessions[@]}"; do
  IFS='|' read -r name path plan tasks needles slow <<<"$s"
  o="$tmp/s-$name/pod/out" before=$failed
  awk '/^step / { sub(":$", "", $2); print $2 }' "$o/steps.log" > "$tmp/ran-$name.txt"
  echo "  $name: $(tr '\n' ' ' < "$tmp/ran-$name.txt")"
  if [[ -n "$slow" ]]; then
    # the slow machine's session ends at speed-check, the
    # server stopped, as any other stop
    if [[ "$plan" == qfork ]]; then
      stop_line="stop: fork too slow on this machine: $slow.0 ms per token (machine q02-$name)" what="the fake fork at $slow ms per token"
      used="^used: replies of tasks 0,7,14 (3), " threshold="threshold: 80 ms per generated token"
    else
      stop_line="stop: engine too slow on this machine: $slow.0 ms per decode step (machine q02-$name)" what="the fake engine at $slow ms per step"
      used="^used: requests 1,2,3 (3), " threshold="threshold: 30 ms per decode step"
    fi
    after=""   # the step after the stop: volume-after when the session has the volume
    [[ $path != volume ]] || after=volume-after
    check "$name ($plan, $path, $what): e07-run.sh exits 1" '[[ "$(cat "$o/e07-run.exit")" == 1 ]]'
    check "$name: steps.log has the planned steps up to speed-check (exit 1), then the stop line, ${after:+$after and }done" \
      'sed "/^speed-check$/q" "$tmp/planned-$name.txt" > "$tmp/planned-$name-stop.txt" &&
       diff <(cat "$tmp/planned-$name-stop.txt"; [[ -z "$after" ]] || echo "$after") "$tmp/ran-$name.txt" > /dev/null &&
       grep -q "^step speed-check: exit 1, " "$o/steps.log" && ! grep "^step " "$o/steps.log" | grep -v "^step speed-check: " | grep -qv ": exit 0, " &&
       [[ "$(grep -v "^step " "$o/steps.log")" == "$(printf "%s\ndone" "$stop_line")" ]]'
    check "$name: speed-check.log names the replies or requests, the machine and the threshold; the server was stopped" \
      'grep -q "$used" "$o/speed-check.log" && grep -q "^machine: q02-$name (RunPod" "$o/speed-check.log" &&
       grep -qx "$threshold" "$o/speed-check.log" && grep -q "^server [0-9]* stopped$" "$o/quality-server.log" &&
       ! kill -0 "$(cat "$o/quality-server.pid")" 2>/dev/null'
  else
    check "$name ($plan, $path, Q_TASKS=$tasks Q_NEEDLES=$needles): e07-run.sh exits 0" '[[ "$(cat "$o/e07-run.exit")" == 0 ]]'
    check "$name: steps.log has every planned step in order, each exit 0, then done" \
      'diff "$tmp/planned-$name.txt" "$tmp/ran-$name.txt" > /dev/null && ! grep "^step " "$o/steps.log" | grep -qv ": exit 0, " &&
       [[ "$(tail -n 1 "$o/steps.log")" == done ]]'
    if [[ "$plan" == qfork ]]; then
      check "$name: speed-check passed at the fake fork's 52.5 ms per token" \
        'grep -qx "pass: 52.5 ms per generated token, at most 80" "$o/speed-check.log"'
    else
      check "$name: speed-check passed at the fake engine's 15 ms per decode step" 'grep -qx "pass: 15.0 ms per decode step, at most 30" "$o/speed-check.log"'
    fi
  fi
  want="nothing" why="binding: unbound, no NUMA node of the GPU (nvidia-smi topo -m's GPU0 row: 'N/A')"
  [[ "$path" == volume ]] && want="numactl --cpunodebind=0 --preferred=0" \
    why="binding: numactl --cpunodebind=0 --preferred=0 worked (node 0's CPUs, its memory preferred)"
  check "$name: placement.txt: bound with $want, the binding's line" \
    'grep -q "^engine runs bound with: $want; placement: " "$o/placement.txt" && grep -qxF "$why" "$o/placement.txt"'
  if [[ "$plan" == qfork ]]; then
    check "$name: placement.txt says qfork has no placement" \
      'grep -qs "; placement: none (qfork runs the fork'"'"'s stock flags)$" "$o/placement.txt"'
    # one slot, the threads from the cpus (nproc capped by the cgroup quota), lm-eval one request at a time
    IFS='|' read -r _ ncpu files cpus source <<<"$(fork_cpu_case "$name")"
    ncpu="${ncpu:-$(nproc)}" threads=$(( cpus < 16 ? cpus : 16 )) threads_batch=$(( cpus < 32 ? cpus : 32 ))
    check "$name: the fork ran -np 1, --threads $threads --threads-batch $threads_batch (nproc $ncpu, cgroup '${files:-none}': cpus $cpus from $source); eval.sh NUM_CONCURRENT 1" \
      'grep -q -- " --threads $threads --threads-batch $threads_batch " "$o/quality-server.cmd" &&
       grep -q -- " -np 1 --kv-unified " "$o/quality-server.cmd" &&
       grep -qE "^fork threads: nproc $ncpu( under [^,]*)?, cgroup CPU limit: .*; cpus $cpus \(from $source\); --threads $threads --threads-batch $threads_batch -np 1$" "$o/fork-server.log" &&
       [[ "$(grep "^num_concurrent " "$o/quality/settings.txt" | sort -u)" == "num_concurrent 1" ]]'
  else
    check "$name: placement.txt gives the gate's placement" 'grep -qs "; placement: --hot-set out/gate/hot-set.json --vram-gb " "$o/placement.txt"'
    # the samplers for the whole session, stopped on exit
    want="$([[ $path == volume ]] && echo clocks_event_reasons.active || echo clocks_throttle_reasons.active)"
    check "$name: out/gpu-clocks.csv has rows of every field, the throttle field $want; out/host-load.log has blocks" \
      '[[ "$(head -n 1 "$o/gpu-clocks.csv" 2>/dev/null)" == "timestamp,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,utilization.gpu,utilization.memory,pcie.link.gen.current,$want" ]] &&
       (( $(grep -c ", 0x0000000000000004$" "$o/gpu-clocks.csv") >= 2 )) && (( $(grep -c "^=== [0-9]" "$o/host-load.log" 2>/dev/null) >= 2 )) &&
       grep -q "^loadavg [0-9]" "$o/host-load.log" && grep -q "^cpu-limit " "$o/host-load.log"'
    check "$name: the samplers stopped on exit" \
      '[[ ! -f "$o/samplers.pid" ]] && grep -q "the gpu sampler .* stopped" "$o/samplers.log" && grep -q "the host sampler .* stopped" "$o/samplers.log"'
  fi
  if [[ "$plan" == qdiag ]]; then
    check "$name: bench-qdiag ran short-static and short-adaptive, each with its window" \
      'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); p = j[\"phases\"]
sys.exit(0 if j[\"ok\"] and [x[\"phase\"] for x in p] == [\"short-static\", \"short-adaptive\"] and
         all(x[\"start_unix\"] <= x[\"end_unix\"] and x[\"decode_experts\"][\"steps\"] > 0 for x in p) else 1)" "$o/bench-qdiag.json"'
    check "$name: the same 20 GSM8K items at NUM_CONCURRENT 1 into out/quality/c1 and 2 into c2" \
      '(for c in 1 2; do grep -qx "num_concurrent $c" "$o/quality/c$c/settings.txt" && grep -qx "gsm8k_cot_llama rc=0" "$o/quality/c$c/lm_eval_status.txt" &&
         python3 -c "import glob, json, sys
ids = {json.loads(x)[\"doc_id\"] for f in glob.glob(sys.argv[1]) for x in open(f)}
sys.exit(0 if len(ids) == 20 else 1)" "$o/quality/c$c/lm_eval/gsm8k_cot_llama/*/samples_gsm8k_cot_llama_*.jsonl" || exit 1; done)'
  fi
  if (( failed > before )); then
    # and the engine's own log, when the session started one
    logs=("$o/e07-run.out" "$o/steps.log")
    [[ -f "$o/quality-engine.log" ]] && logs+=("$o/quality-engine.log")
    echo "  $name: the tail of ${logs[*]##*/pod/}:"
    tail -n 5 "${logs[@]}" | sed 's/^/    /'
  fi
done
# a cgroup CPU limit that is neither a quota nor "max" stops the fork server before it starts (fail loudly)
d="$tmp/fork-cgroup-bad"
mkdir -p "$d/cgroup" "$d/out"
echo "1.5 100000" > "$d/cgroup/cpu.max"
rc=0; ( cd "$d" && Q_FORK_CGROUP="$d/cgroup" bash "$here/pod_server.sh" fork model 1 ) > "$d/out.txt" 2>&1 || rc=$?
check "pod_server.sh fork on cpu.max '1.5 100000': exit 1 naming the limit, no server started" \
  '(( rc == 1 )) && grep -q "cgroup CPU limit .v2 $d/cgroup/cpu.max 1.5 100000. is not a quota and a period" "$d/out.txt" &&
   [[ ! -e "$d/out/quality-server.pid" ]]'

echo "--- 4.pod/qdiag_report.py on the qdiag sessions and on qengine try 2's real logs; e07_report.py's sheets"
# row FILE PHASE EXPR: EXPR (Python, on the row r of PHASE in qdiag.json FILE) holds
row() {
  python3 -c "import json, sys
r = {x['phase']: x for x in json.load(open(sys.argv[1]))['rows']}.get(sys.argv[2])
sys.exit(0 if r is not None and eval('(' + sys.argv[3] + ')') else 1)" "$@"
}
for name in qdiag-volume qdiag-hf; do
  o="$tmp/s-$name/pod/out"
  rc=0; python3 "$STRATA_DIR/tools/dsv41/pod/qdiag_report.py" "$tmp/s-$name/pod" > "$tmp/qdiag-$name.txt" 2>&1 || rc=$?
  j="$o/qdiag.json"
  check "$name: qdiag_report.py exits 0 and writes out/qdiag.md with the 6 rows in order" \
    '(( rc == 0 )) && [[ -s "$o/qdiag.md" ]] && python3 -c "import json, sys
sys.exit(0 if [r[\"phase\"] for r in json.load(open(sys.argv[1]))[\"rows\"]] == [\"bench-qdiag short-static\",
  \"bench-qdiag short-adaptive\", \"continuation\", \"lm-eval-gsm8k-c1\", \"lm-eval-gsm8k-c2\", \"all server requests\"] else 1)" "$j"'
  check "$name: every row has requests and the fake engine's decode (15 ms per step, 12.5 of doorbell waits, hit rate 0.9)" \
    '(for p in "bench-qdiag short-static" "bench-qdiag short-adaptive" continuation lm-eval-gsm8k-c1 lm-eval-gsm8k-c2 "all server requests"; do
       row "$j" "$p" "r[\"requests\"] > 0 and r[\"generated\"] > 0 and r[\"ms_per_step\"] == 15.0 and r[\"doorbell_ms_per_step\"] == 12.5 and r[\"hit_rate\"] == 0.9" || exit 1; done)'
  check "$name: the static bench phase has no swaps, the adaptive one has; c1 and c2 have 20 requests each, all matched" \
    'row "$j" "bench-qdiag short-static" "r[\"swaps_per_request\"] == 0" && row "$j" "bench-qdiag short-adaptive" "r[\"swaps_per_request\"] > 0" &&
     row "$j" lm-eval-gsm8k-c1 "r[\"requests\"] == 20" && row "$j" lm-eval-gsm8k-c2 "r[\"requests\"] == 20" &&
     python3 -c "import json, sys; m = json.load(open(sys.argv[1]))[\"match\"]; sys.exit(0 if m[\"unmatched\"] == 0 and m[\"matched\"] == m[\"engine_requests\"] > 0 else 1)" "$j"'
  check "$name: c1 and c2 carry the stub GPU's samples (SM 2400 MHz, 300.5 W of 600, SwPowerCap) and the host's" \
    '(for p in lm-eval-gsm8k-c1 lm-eval-gsm8k-c2; do row "$j" "$p" "r[\"gpu_samples\"] > 0 and r[\"sm_clock_median\"] == 2400 and r[\"power_median\"] == 300.5 and
       r[\"power_limit\"] == 600 and list(r[\"throttle_reasons\"]) == [\"SwPowerCap\"] and r[\"host_samples\"] > 0" || exit 1; done)'
  check "$name: the engines' init rates (the bench's load and the server's)" \
    'python3 -c "import json, sys; e = json.load(open(sys.argv[1]))[\"engines\"]
sys.exit(0 if [x[\"log\"] for x in e] == [\"bench-qdiag-engine.log\", \"quality-engine.log\"] and all(x[\"cpu_gbs\"] == 150 for x in e) else 1)" "$j"'
  (( failed == 0 )) || tail -n 20 "$tmp/qdiag-$name.txt" | sed 's/^/    /'
done
# the match on a fixture: NUM_CONCURRENT 2's responses landing in the proxy's log in the other order (requests 1 and
# 2, 3 and 4), a 502 with usage (the echo cut's), a request with no response and a 500 without usage
f="$tmp/match/out"
mkdir -p "$f/quality/c2"
python3 - "$f" <<'EOF'
import json, os, sys
o = sys.argv[1]
with open(os.path.join(o, "quality-engine.log"), "w") as f:
    for n in range(1, 7):
        f.write(f"request {n}: {100 + n} prompt tokens, cold resume of 0, {100 + n} read in 10.0 ms, {10 + n} "
                f"generated (stop) in 100.0 ms\n")
with open(os.path.join(o, "steps.log"), "w") as f:
    f.write("step lm-eval-swap: exit 0, 100 s\n")
with open(os.path.join(o, "step-times.log"), "w") as f:
    f.write("lm-eval-swap 1000 1100 0\n")
with open(os.path.join(o, "quality", "c2", "api_usage.jsonl"), "w") as f:
    for t, n, status in ((1010, 2, 200), (1011, 1, 200), (1020, 4, 200), (1021, 3, 200), (1030, 5, 502)):
        f.write(json.dumps({"t": t, "status": status, "usage": {"prompt_tokens": 100 + n, "completion_tokens": 10 + n}})
                + "\n")
    f.write(json.dumps({"t": 1040, "status": 500, "usage": None}) + "\n")
EOF
rc=0; python3 "$STRATA_DIR/tools/dsv41/pod/qdiag_report.py" "$tmp/match" > "$tmp/qdiag-match.txt" 2>&1 || rc=$?
check "qdiag_report.py: swapped concurrent responses still match; a 502 is an error, not a cancel; the 500 is named" \
  '(( rc == 0 )) && row "$f/qdiag.json" lm-eval-swap "r[\"requests\"] == 6 and r[\"cancelled\"] == 1 and r[\"errors\"] == 1" &&
   python3 -c "import json, sys; m = json.load(open(sys.argv[1]))[\"match\"]
sys.exit(0 if (m[\"matched\"], m[\"errors\"], m[\"unmatched\"], m[\"bare_errors\"], m[\"proxy_responses\"]) == (5, 1, 1, 1, 6) else 1)" "$f/qdiag.json" &&
   grep -q "^| lm-eval-swap | 6 (1 cancelled, 1 answered with an error) |" "$f/qdiag.md" &&
   grep -q "or among the 1 error responses without usage to match" "$f/qdiag.md"'
f="$tmp/report/20261012T120000Z"
mkdir -p "$f"
# the volume sessions: the HF ones' model-download.json is the download stub's, not one the sheet can read
cp -r "$tmp/s-qdiag-volume/pod/out" "$f/out"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/report/sheets" > /dev/null
r="$tmp/report/sheets/2026-10-12-qdiag/README.md"
check "e07_report.py: the qdiag sheet has the diagnostic per phase, the server as run and both GSM8K runs" \
  'grep -q "^## Diagnostic per phase (out/qdiag.md)" "$r" && grep -q "^| lm-eval-gsm8k-c2 | 20 |" "$r" && grep -q "^## Server as run" "$r" &&
   grep -q "^| lm-eval-gsm8k-c1 | 1 | " "$r" && grep -q "^| lm-eval-gsm8k-c2 | 2 | " "$r"'
f="$tmp/report/20261012T130000Z"
mkdir -p "$f"
cp -r "$tmp/s-qengine-volume/pod/out" "$f/out"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/report/sheets" > /dev/null
r="$tmp/report/sheets/2026-10-12-qengine/README.md"
check "e07_report.py: the qengine sheet has the GPU clock summary per phase" \
  'grep -q "^## GPU clocks and host load (pod/pod-samplers.sh)" "$r" && grep -qE "^\| lm-eval-gsm8k_cot_llama \| [1-9][0-9]* \| 2400 \| 2400 \|" "$r"'
if [[ -f "$TRY2_RUN/out-logs.tar.gz" ]]; then
  mkdir -p "$tmp/try2"
  tar -xzf "$TRY2_RUN/out-logs.tar.gz" -C "$tmp/try2"
  rc=0; python3 "$STRATA_DIR/tools/dsv41/pod/qdiag_report.py" "$tmp/try2" > "$tmp/qdiag-try2.txt" 2>&1 || rc=$?
  grep "^| all server requests | 201" "$tmp/qdiag-try2.txt" | head -n 1 | sed 's/^/  try 2: /'
  check "try 2's real logs: 201 requests (4 cancelled), 66.9 ms per token, 67.5 per step, 60.0 of doorbell waits, hit rate 0.917" \
    '(( rc == 0 )) && row "$tmp/try2/out/qdiag.json" "all server requests" "r[\"requests\"] == 201 and r[\"cancelled\"] == 4 and
       round(r[\"ms_per_token\"], 1) == 66.9 and round(r[\"ms_per_step\"], 1) == 67.5 and round(r[\"doorbell_ms_per_step\"], 1) == 60.0 and
       round(r[\"hit_rate\"], 3) == 0.917 and r[\"swaps_per_request\"] == 521"'
  check "try 2: the continuation and the two lm-eval steps get their requests (3, then the rest)" \
    'row "$tmp/try2/out/qdiag.json" continuation "r[\"requests\"] == 3" && row "$tmp/try2/out/qdiag.json" lm-eval-gsm8k_cot_llama "r[\"requests\"] > 100" &&
     row "$tmp/try2/out/qdiag.json" lm-eval-ifeval "r[\"requests\"] > 50"'
else
  echo "SKIP try 2: no $TRY2_RUN/out-logs.tar.gz (TRY2_RUN)"
fi

echo "--- 5. pod/speed_check.py on the real logs: qengine try 2's machine fails, qdiag's passes; the fork's check"
# name|run|machine|exit|the last line
for c in "try 2|$TRY2_RUN|try2-host|1|stop: engine too slow on this machine: 44.2 ms per decode step (machine try2-host)" \
         "qdiag|$QDIAG_RUN|qdiag-host|0|pass: 20.2 ms per decode step, at most 30"; do
  IFS='|' read -r name run machine want last <<<"$c"
  if [[ -f "$run/out/quality-engine.log" ]]; then
    rc=0; Q_SPEED_MAX_MS= python3 "$STRATA_DIR/tools/dsv41/pod/speed_check.py" "$run/out/quality-engine.log" --requests 3 \
      --machine "$machine" --machine-source "$run/pod.json" > "$tmp/speed-$name.txt" 2>&1 || rc=$?
    sed 's/^/  /' "$tmp/speed-$name.txt"
    check "$name's real log ($machine): speed_check.py exits $want, the last line '$last'" \
      '(( rc == want )) && [[ "$(tail -n 1 "$tmp/speed-$name.txt")" == "$last" ]]'
  else
    echo "SKIP $name: no $run/out/quality-engine.log"
  fi
done

# the fork's check on llama-server's print_timing lines. The 2026-10-11 qfork session
# (A100 SXM4, 18 vCPU, -np 4 --threads-batch 32): its continuation's 3 replies ran one request
# at a time, the lm-eval replies after them 4 at once. The reading at the threshold of 80: the continuation
# (70.8) passes, the 4 concurrent lm-eval replies (444.2) stop, t04a's baseline (52.5) passes
f="$tmp/fork-speed"
mkdir -p "$f"
fork_cases=()
if [[ -f "$FORK_RUN/out/quality-server.log" ]]; then
  grep -E "\|\s+eval time = " "$FORK_RUN/out/quality-server.log" | tail -n +4 > "$f/lm-eval.log"
  fork_cases+=("2026-10-11 continuation|$FORK_RUN/out/quality-server.log|qfork-host|0|pass: 70.8 ms per generated token, at most 80"
               "2026-10-11 lm-eval|$f/lm-eval.log|qfork-host|1|stop: fork too slow on this machine: 444.2 ms per token (machine qfork-host)")
else
  echo "SKIP the 2026-10-11 qfork session: no $FORK_RUN/out/quality-server.log (FORK_RUN)"
fi
# t04a's stock baseline (pod 1 in tools/dsv41/README.md, A100 SXM4): 18.92 and 19.15 tok/s decode, as llama-server logs a reply
for x in "0 52.85 128" "7 52.22 128"; do
  read -r t ms n <<<"$x"
  printf 'I slot print_timing: id  0 | task %s |        eval time = %10.2f ms / %6d tokens (%8.2f ms per token, %8.2f tokens per second)\n' \
    "$t" "$(python3 -c "print($ms * $n)")" "$n" "$ms" "$(python3 -c "print(1000 / $ms)")"
done > "$f/t04a.log"
fork_cases+=("t04a stock baseline|$f/t04a.log|t04a|0|pass: 52.5 ms per generated token, at most 80")
for c in "${fork_cases[@]}"; do
  IFS='|' read -r name log machine want last <<<"$c"
  rc=0; Q_FORK_MAX_MS= python3 "$STRATA_DIR/tools/dsv41/pod/speed_check.py" "$log" --fork --requests 3 \
    --machine "$machine" --machine-source "test case" > "$tmp/speed-$name.txt" 2>&1 || rc=$?
  sed 's/^/  /' "$tmp/speed-$name.txt"
  check "$name (speed_check.py --fork, the first 3 replies): exits $want, the last line '$last'" \
    '(( rc == want )) && [[ "$(tail -n 1 "$tmp/speed-$name.txt")" == "$last" ]]'
done

echo "--- 6. the machine qualification: e07.sh's session against a fake RunPod, ssh and scp"
# section 1's payloads are not read again: their room (/tmp may be a tmpfs) for these two sessions' own
rm -rf -- "$tmp"/payload-*/
q="$tmp/q02f"
mkdir -p "$q/bin" "$q/probe/cu130/machine-probe"
cat > "$q/runpod_api.py" <<'EOF'
#!/usr/bin/env python3
"""q02-test.sh's RunPod (pod-session.sh's RUNPOD_API): price lists 2.09 USD/h; create takes the next machine of
Q02F_STATE/machines (none left: exit 3, no stock), its pod q02f-pod-N on SSH port 22000 + N; get answers with
127.0.0.1 and that port; delete records the pod. Q02F_STATE/created and deleted keep the calls."""
import json
import os
import sys

st = os.environ["Q02F_STATE"]
cmd, args = sys.argv[1], sys.argv[2:]
opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
if cmd == "price":
    print(json.dumps({"gpu": opt["--gpu"], "cloud": opt["--cloud"], "price": 2.09, "availability": "High"}))
elif cmd == "create":
    machines = open(os.path.join(st, "machines")).read().split()
    created = os.path.join(st, "created")
    n = len(open(created).read().splitlines()) if os.path.exists(created) else 0
    if n >= len(machines):
        print("no stock (q02-test.sh fake)", file=sys.stderr)
        sys.exit(3)
    with open(created, "a") as f:
        f.write(f"q02f-pod-{n + 1} {machines[n]}\n")
    with open(os.path.join(st, f"port-{22001 + n}"), "w") as f:
        f.write(machines[n])
    print(json.dumps({"id": f"q02f-pod-{n + 1}", "costPerHr": 2.59, "gpu": opt["--gpu"], "cloud": opt["--cloud"],
                      "dc": opt.get("--dc") or "any", "machineId": machines[n], "memoryInGb": 377}))
elif cmd == "get":
    n = int(args[0].rsplit("-", 1)[1])
    print(json.dumps({"id": args[0], "desiredStatus": "RUNNING", "publicIp": "127.0.0.1", "port22": 22000 + n,
                      "memoryInGb": 377}))
elif cmd == "delete":
    with open(os.path.join(st, "deleted"), "a") as f:
        f.write(args[0] + "\n")
    print(f"deleted {args[0]} (HTTP 204, q02-test.sh fake)")
else:
    sys.exit(f"runpod_api.py (q02-test.sh fake): {cmd}?")
EOF
cat > "$q/bin/ssh" <<'EOF'
#!/usr/bin/env bash
# q02-test.sh's ssh: the pod is Q02F_STATE/root-<port> (its /root/), its machine Q02F_STATE/port-<port>. `true`
# answers; the CUDA probe (a real binary, no GPU here) says ok; the run's detached start writes the finished run's
# exit status and steps.log instead of running it; every other command runs locally with /root/ in the pod's dir
while [[ "${1:-}" == -* ]]; do
  case "$1" in -p) port="$2"; shift 2 ;; -i|-o) shift 2 ;; *) shift ;; esac
done
shift   # root@<ip>
cmd="$*" r="$Q02F_STATE/root-$port"
mkdir -p "$r"
echo "ssh $port: $cmd" >> "$Q02F_STATE/ssh.log"
case "$cmd" in
  true) exit 0 ;;
  *"/root/dsv41_cuda_probe"*) echo "dsv41_cuda_probe: ok, the q02-test.sh stub (sm_120), 1 device(s)"; exit 0 ;;
  # the CPU experts probe of a machine the machine probe passed, at a measured cbatch session's rate (pod/c05f-test.sh
  # checks its floor)
  *"/root/dsv41_cpu_experts_probe"*)
    echo "dsv41_cpu_experts_probe: measure cpu_experts_gbs 100.4 bound >=71 ok (the q02-test.sh stub)"
    echo "dsv41_cpu_experts_probe: PASS, the CPU experts rate within its bound"; exit 0 ;;
  *"nohup setsid"*)
    mkdir -p "$r/e07/out"
    echo 0 > "$r/e07/out/e07-run.exit"
    echo done > "$r/e07/out/steps.log"
    echo "plan: qengine, readers: dense" > "$r/e07/out/plan.log"
    echo "$port" >> "$Q02F_STATE/started"
    exit 0 ;;
esac
FAKE_MACHINE="$(cat "$Q02F_STATE/port-$port")" exec bash -c "${cmd//\/root\//$r/}"
EOF
cat > "$q/bin/scp" <<'EOF'
#!/usr/bin/env bash
# q02-test.sh's scp: into or out of the pod's dir (q02-test.sh's ssh); a payload tarball is recorded, not copied
while [[ "${1:-}" == -* ]]; do
  case "$1" in -P) port="$2"; shift 2 ;; -i|-o) shift 2 ;; *) shift ;; esac
done
r="$Q02F_STATE/root-$port"
mkdir -p "$r"
case "$2" in
  root@*:*) echo "$port ${2#*:} $(basename "$1")" >> "$Q02F_STATE/uploads"
    [[ "$1" == *.tar.gz ]] || cp "$1" "$r/${2#*:/root/}" ;;
  *) cp "$r/${1#*:/root/}" "$2" ;;
esac
EOF
cat > "$q/probe/cu130/machine-probe/dsv41_machine_probe" <<'EOF'
#!/usr/bin/env python3
"""q02-test.sh's dsv41_machine_probe: tests/dsv41/engine/machine_probe.cu's lines with FAKE_MACHINE's numbers against
the bounds it is given, resolved as the real one resolves them (the GPU read's per model from --gpu-read-bound, else
--min-gpu-read-gbs, else info; the SM clock's and the power limit's as ratios of the GPU's own maximums). A machine
named a100-* is an A100 SXM4 80GB with an A100 host's numbers (measured: PCIe 22.5-25, host 52.5-83 GB/s), any
other an RTX PRO 6000 Blackwell Server Edition with a good machine's (host read 375.2, the 2026-10-13 EPYC 9355P
pod's), or one measure made unfit by the machine's name."""
import os
import sys

need = ("--min-h2d-gbs", "--min-d2h-gbs", "--max-doorbell-us", "--min-host-read-gbs", "--min-sm-ratio",
        "--min-power-ratio")
b, reads, args = {}, [], sys.argv[1:]
while args:
    k, x = args[0], args[1] if len(args) > 1 else ""
    args = args[2:]
    if k == "--gpu-read-bound":
        name, _, gbs = x.rpartition("=")
        reads.append((name, float(gbs)))
    elif k in need or k == "--min-gpu-read-gbs":
        b[k] = float(x)
    else:
        b = None
        break
if b is None or any(k not in b for k in need):
    print(f"dsv41_machine_probe: the q02-test.sh stub got {sys.argv[1:]}")
    sys.exit(2)
m = os.environ["FAKE_MACHINE"]
if m.startswith("a100"):
    gpu = "NVIDIA A100-SXM4-80GB"
    v = {"gpu_read_gbs": 1712.6, "h2d_gbs": 22.6, "d2h_gbs": 23.1, "doorbell_median_us": 9.6, "doorbell_p99_us": 17.3,
         "host_read_gbs": 55.0, "sm_clock_mhz": 1410.0, "sm_clock_max_mhz": 1410.0, "power_draw_w": 238.0,
         "power_limit_w": 400.0, "power_default_limit_w": 400.0}
else:
    gpu = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    v = {"gpu_read_gbs": 1520.3, "h2d_gbs": 26.5, "d2h_gbs": 26.4, "doorbell_median_us": 8.2, "doorbell_p99_us": 14.9,
         "host_read_gbs": 375.2, "sm_clock_mhz": 2790.0, "sm_clock_max_mhz": 2850.0, "power_draw_w": 251.0,
         "power_limit_w": 600.0, "power_default_limit_w": 600.0}
if m.startswith("slow-doorbell"):
    v["doorbell_median_us"], v["doorbell_p99_us"] = 1003.4, 1210.0
elif m.startswith("slow-gpu"):
    # a power-capped card: the read and the clock low, the limit under 0.70 of its default
    v["gpu_read_gbs"], v["sm_clock_mhz"], v["power_limit_w"] = 412.7, 1650.0, 350.0
elif m.startswith("slow-host"):
    v["host_read_gbs"] = 109.5   # session 3's Intel Xeon 6760P (2026-10-15)
read_bound = b.get("--min-gpu-read-gbs") or next((g for n, g in reads if n in gpu), None)
bound = {"gpu_read_gbs": (">=", read_bound), "h2d_gbs": (">=", b["--min-h2d-gbs"]),
         "d2h_gbs": (">=", b["--min-d2h-gbs"]), "doorbell_median_us": ("<=", b["--max-doorbell-us"]),
         "host_read_gbs": (">=", b["--min-host-read-gbs"]),
         "sm_clock_mhz": (">=", b["--min-sm-ratio"] * v["sm_clock_max_mhz"]),
         "power_limit_w": (">=", b["--min-power-ratio"] * v["power_default_limit_w"])}
failed = []
for k, x in v.items():
    op, y = bound.get(k, (None, None))
    if y is None:
        print(f"dsv41_machine_probe: measure {k} {x:.1f} bound none info (the q02-test.sh stub)")
        continue
    ok = x >= y if op == ">=" else x <= y
    print(f"dsv41_machine_probe: measure {k} {x:.1f} bound {op}{y:g} {'ok' if ok else 'FAIL'} (the q02-test.sh stub)")
    if not ok:
        failed.append(f"{k} {x:.1f} (bound {op}{y:g})")
print(f"dsv41_machine_probe: machine {gpu}, the q02-test.sh stub of {m}")
print("dsv41_machine_probe: PASS, every measure within its bound" if not failed else "dsv41_machine_probe: FAIL, " + "; ".join(failed))
sys.exit(1 if failed else 0)
EOF
chmod +x "$q/bin"/* "$q/probe/cu130/machine-probe/dsv41_machine_probe"
# the same stub for the A100's flavor
mkdir -p "$q/probe/cu128/machine-probe"
cp "$q/probe/cu130/machine-probe/dsv41_machine_probe" "$q/probe/cu128/machine-probe/"
src_sha="$(git -C "$STRATA_DIR" show HEAD:tests/dsv41/engine/machine_probe.cu | sha256sum | cut -d' ' -f1)"
for t in "cu130|120" "cu128|80;89"; do
  printf 'source: tests/dsv41/engine/machine_probe.cu sha256 %s\ncommit: q02-test.sh stub\nnvcc: none (a stub)\narchs: %s\n%s  dsv41_machine_probe\n' \
    "$src_sha" "${t#*|}" "$(sha256sum "$q/probe/${t%|*}/machine-probe/dsv41_machine_probe" | cut -d' ' -f1)" > "$q/probe/${t%|*}/machine-probe/MANIFEST"
done
: > "$q/key"; : > "$q/key.pub"
# name|the machines the fake RunPod hands out, in order|GPU_TYPES (empty: qengine's default, the RTX PRO 6000)
for c in "rejected|slow-doorbell-1 slow-gpu-2 slow-host-3 good-4|" "passes|slow-doorbell-1 slow-doorbell-1 good-3|" \
         "a100|a100-good-1|NVIDIA A100-SXM4-80GB"; do
  IFS='|' read -r name machines gpus <<<"$c"
  mkdir -p "$q/$name"
  echo "$machines" > "$q/$name/machines"
  touch "$q/$name/uploads"
  ( export PATH="$q/bin:$PATH" Q02F_STATE="$q/$name"
    [[ -z "$gpus" ]] || export GPU_TYPES="$gpus"
    rc=0; E07_PLAN=qengine HF_CLOUDS= CAP_MIN=5 RUNPOD_API="$q/runpod_api.py" RUNPOD_API_KEY=q02-test-fake SSH_KEY="$q/key" \
      MACHINE_PROBE_ROOT="$q/probe" OUT="$q/$name/20261012T220000Z" SHEET_ROOT="$q/$name/sheets" \
      timeout 900 bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" > "$q/$name/e07.out" 2>&1 < /dev/null || rc=$?
    echo "$rc" > "$q/$name/rc" ) &
  jobs_+=($!)
done
wait "${jobs_[@]}"
jobs_=()
r="$q/rejected" l="$q/rejected/20261012T220000Z/e07.log"
grep -E "qualification failed|FAIL: 3 pods" "$l" | sed 's/^/  rejected: /' || true
check "rejected: e07.sh exits 1 with MAX_PODS 3's FAIL line naming the three machines, nothing uploaded" \
  '[[ "$(cat "$r/rc")" == 1 ]] &&
   grep -q "FAIL: 3 pods and none came up with SSH and a passing CUDA probe and machine qualification (machines tried: slow-doorbell-1 slow-gpu-2 slow-host-3): nothing uploaded$" "$l"'
check "rejected: each failure names its measure and value; each pod deleted; the fourth machine never created" \
  'grep -q "the machine qualification failed on q02f-pod-1 (machine slow-doorbell-1, exit 1): measure doorbell_median_us 1003.4 bound <=50 FAIL (the q02-test.sh stub)$" "$l" &&
   grep -q "the machine qualification failed on q02f-pod-2 (machine slow-gpu-2, exit 1): measure gpu_read_gbs 412.7 bound >=800 FAIL (the q02-test.sh stub); measure sm_clock_mhz 1650.0 bound >=1995 FAIL (the q02-test.sh stub); measure power_limit_w 350.0 bound >=420 FAIL (the q02-test.sh stub)$" "$l" &&
   grep -q "the machine qualification failed on q02f-pod-3 (machine slow-host-3, exit 1): measure host_read_gbs 109.5 bound >=250 FAIL (the q02-test.sh stub)$" "$l" &&
   [[ "$(cat "$r/deleted")" == "$(printf "q02f-pod-%s\n" 1 2 3)" ]] && (( $(wc -l < "$r/created") == 3 )) &&
   ! grep -q "tar.gz" "$r/uploads" && [[ ! -e "$r/started" ]]'
check "rejected: probe.log and the session log carry every machine's values" \
  '(( $(grep -c "^=== pod q02f-pod-[123] machine " "$r/20261012T220000Z/probe.log") == 3 && $(grep -c "^dsv41_machine_probe: measure " "$r/20261012T220000Z/probe.log") == 33 )) &&
   (( $(grep -c "\] dsv41_machine_probe: measure " "$l") == 33 ))'
sheet="$r/sheets/2026-10-12-qengine"
check "rejected: the qengine sheet has the three rejected machines with their failed measures" \
  'grep -q "^## Machine qualification" "$sheet/README.md" &&
   grep -q "^| q02f-pod-1 | slow-doorbell-1 | .*\*\*1003.4 FAIL\*\* (<=50) .*rejected: doorbell_median_us 1003.4 (bound <=50) |$" "$sheet/README.md" &&
   grep -q "^| q02f-pod-2 | slow-gpu-2 | .*rejected: gpu_read_gbs 412.7 (bound >=800); sm_clock_mhz 1650.0 (bound >=1995); power_limit_w 350.0 (bound >=420) |$" "$sheet/README.md" &&
   grep -q "^| q02f-pod-3 | slow-host-3 | .*rejected: host_read_gbs 109.5 (bound >=250) |$" "$sheet/README.md"'
check "rejected: probe.log and the session log name each RTX PRO 6000 pod's host read floor, the default 250" \
  '(( $(grep -c "^qualification: q02f-pod-[123] (NVIDIA RTX PRO 6000 Blackwell [^)]*): host read floor 250 GB/s (the default for any GPU but the A100, " "$r/20261012T220000Z/probe.log") == 3 )) &&
   (( $(grep -c "\] qualification: q02f-pod-[123] (NVIDIA RTX PRO 6000 Blackwell [^)]*): host read floor 250 GB/s " "$l") == 3 ))'
r="$q/passes" l="$q/passes/20261012T220000Z/e07.log"
check "passes: the unfit machine rejected, the same machine deleted at the create without a probe, the good one passes" \
  'grep -q "the machine qualification failed on q02f-pod-1 (machine slow-doorbell-1, exit 1)" "$l" &&
   grep -q "q02f-pod-2 landed on machine slow-doorbell-1, which failed the CUDA probe or the qualification before: deleting it" "$l" &&
   ! grep -q "^=== pod q02f-pod-2 " "$r/20261012T220000Z/probe.log" && grep -q "\] dsv41_machine_probe: PASS, every measure within its bound" "$l"'
check "passes: the session proceeds as before: the payload up to the good pod only, the run, the pull, exit 0" \
  '[[ "$(cat "$r/rc")" == 0 ]] && [[ "$(grep tar.gz "$r/uploads")" == "22003 /root/e07.tar.gz payload-cu130.tar.gz" ]] &&
   [[ "$(cat "$r/started")" == 22003 ]] && grep -q "every step passed" "$l" &&
   [[ "$(cat "$r/deleted")" == "$(printf "q02f-pod-%s\n" 1 2 3)" ]]'
sheet="$r/sheets/2026-10-12-qengine"
check "passes: the sheet has the rejected machine and the passed one" \
  'grep -q "^| q02f-pod-1 | slow-doorbell-1 | .*rejected: doorbell_median_us" "$sheet/README.md" &&
   grep -q "^| q02f-pod-3 | good-3 | .* | 1520.3 | 26.5 | 26.4 | 8.2 | 14.9 | 375.2 | 100.4 | 2790.0 | 2850.0 | 251.0 | 600.0 | 600.0 | passed |$" "$sheet/README.md"'
# an A100's 1410 MHz and 400 W are its own maximums (the first 2000 MHz bound rejected
# every A100), its read bound is the A100's 900 GB/s, its PCIe numbers pass the universal 18 GB/s and its host read
# the A100's own floor of 40
r="$q/a100" pl="$q/a100/20261012T220000Z/probe.log"
grep "^dsv41_machine_probe: measure \(gpu_read\|sm_clock_mhz\|power_limit\)" "$pl" 2>/dev/null | sed 's/^/  a100: /' || true
check "a100: the A100 machine passes on the relative bounds and its model's GPU read bound" \
  'grep -q "^dsv41_machine_probe: measure gpu_read_gbs 1712.6 bound >=900 ok " "$pl" &&
   grep -q "^dsv41_machine_probe: measure h2d_gbs 22.6 bound >=18 ok " "$pl" && grep -q "^dsv41_machine_probe: measure host_read_gbs 55.0 bound >=40 ok " "$pl" &&
   grep -q "^dsv41_machine_probe: measure sm_clock_mhz 1410.0 bound >=987 ok " "$pl" &&
   grep -q "^dsv41_machine_probe: measure power_limit_w 400.0 bound >=280 ok " "$pl" &&
   grep -q "^dsv41_machine_probe: machine NVIDIA A100-SXM4-80GB, " "$pl" && grep -q "^dsv41_machine_probe: PASS" "$pl" &&
   grep -qx "qualification: q02f-pod-1 (NVIDIA A100-SXM4-80GB): host read floor 40 GB/s (the A100 default, its hosts measured 40.9-162.2)" "$pl"'
check "a100: the session proceeds with the payload of the A100 flavor, exit 0, the sheet row passed" \
  '[[ "$(cat "$r/rc")" == 0 ]] && [[ "$(grep tar.gz "$r/uploads")" == "22001 /root/e07.tar.gz payload-cu128.tar.gz" ]] &&
   [[ "$(cat "$r/started")" == 22001 ]] &&
   grep -q "^| q02f-pod-1 | a100-good-1 | NVIDIA A100-SXM4-80GB, .* | 1712.6 | 22.6 | 23.1 | 9.6 | 17.3 | 55.0 | 100.4 | 1410.0 | 1410.0 | 238.0 | 400.0 | 400.0 | passed |$" "$r/sheets/2026-10-12-qengine/README.md" &&
   grep -q "^Bounds: GPU read GB/s >=900, .*SM MHz >=987, power limit W >=280 " "$r/sheets/2026-10-12-qengine/README.md"'
if (( failed > 0 )); then
  for n in rejected passes a100; do echo "  $n: the tail of e07.out:"; tail -n 8 "$q/$n/e07.out" | sed 's/^/    /'; done
fi

echo "q02-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
