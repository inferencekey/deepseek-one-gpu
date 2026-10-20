#!/usr/bin/env bash
# The cfinal plan's checks without a GPU (batch quality: item 14 is no quality loss from batching, item 3 batched on
# all three tasks and item 4 against the single-user engine run; the fast path by default): no model, no RunPod, no HF,
# no money.
#   1. in the cu130 builder image (tools/dsv41/build.sh's; no GPU): dsv41-engine --serve --dry-run --batch 4 on tiny-b
#      starts in the fork's CPU numerics by default (its load line) and in the GPU's with --cpu-numerics gpu (no load
#      line), its --help names fork the default, a third value refused; fakes.py's engine the same (as the engine's)
#   2. pod/e07.sh --dry-run with E07_PLAN=cfinal and its defaults (the model from HF in any region, MAX_SESSION_USD
#      3.35, MAX_USD_PER_HR 2.60): exit 0, the RTX PRO 6000 only, the two HF offers (SECURE and COMMUNITY in any data
#      center), the volume not offered, the qualification first, the HF plan's steps (no batching-off server, no
#      ab-gpu, det-batch, det-sampling nor fork server), each with its timeout, the lm-eval tasks from the fork's
#      numerics' own times (the single-user run's ifeval 1077 s, gsm8k 531 s, humaneval 359 s x 1.25), steady-c4 3
#      repeats, the plan under 3.35 USD at 2.60 USD/h, its latest end >= 2 min before the pod's deadline, the
#      payload's sources (the tarball's engine HEAD's, run_lm_eval.sh, no det_batch_check.py); then the same with a
#      network volume (NETWORK_VOLUME=testvolume01 HF_CLOUDS=, a placeholder id: nothing is created): exit 0, the
#      volume's offer only, its plan's steps, USD and latest end the same way
#   3. the cfinal rehearsal on the volume's path: e07-run.sh from the payload with c05-test.sh's stubs and fakes.py as
#      the engine, lm-eval for real from QUALITY_VENV offline on 3 items per task: every planned step exit 0, then
#      done, the batched server as planned with the engine's default numerics (no --cpu-numerics: the fake's log says
#      the fork's), no det-batch, det-sampling nor A/B output, out/quality's three tasks at 4 concurrent requests
#   4. e07_report.py's cfinal sheet: item 11 (the single user against session 4's 61.55, steady-c4 against the single
#      user with every user >= 20), item 3 batched, item 4 against the single-user engine run, item 14 (items 3 and 4
#      together: its verdict item 4's when item 3 came back whole), no det-batch, items 7 and 8 nor fork arm
#      sections; with humaneval's status cut, item 14 not evaluated naming it
#   5. pod/sheet_banners_test.py: items 11 and 14 on the final batch determinism sheet, a banner on every earlier one
# Exits 1 if any check fails. c05-test.sh, c04-test.sh, c03-test.sh, e07-lost-test.sh and verify.sh are run on their
# own.
#
#   BUILD_TAG=-c04 bash tools/dsv41/quality/cfinal-test.sh
#
# BUILD_ROOT (default $HOME/dsv41-build) holds the build (cu130${BUILD_TAG}/strata), the tiny models (engine/e01), the
# vocabulary, the real headers (engine/headers), the conversations, the tarballs (dist${BUILD_TAG}) the dry runs and
# the payload read, and the lm-eval venv (QUALITY_VENV, default quality/venv). TMPDIR on disk is advised (/tmp is a
# tmpfs; the rehearsal copies the sparse shards).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
HEADERS="${HEADERS:-$BUILD_ROOT/engine/headers}"
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"
E01="${E01:-$BUILD_ROOT/engine/e01}"
QUALITY_VENV="${QUALITY_VENV:-$BUILD_ROOT/quality/venv}"
STRATA_BUILD="$BUILD_ROOT/cu130${BUILD_TAG:-}/strata"
for f in "$HEADERS/head-7.bin" "$VOCAB" "$E01/tiny-b.gguf" "$STRATA_BUILD/dsv41-engine" "$QUALITY_VENV/bin/lm_eval"; do
  [[ -e "$f" ]] || { echo "cfinal-test.sh: no $f" >&2; exit 2; }
done
# the fake engine's delays: c04-test.sh's (a prompt token 0.28 ms, 5 ms a step alone and 1.5 ms per further row)
P_MS=0.28 D_MS=5 R_MS=1.5
tmp="$(mktemp -d)"
jobs_=()
cleanup() {
  local f
  for f in "${jobs_[@]}"; do kill "$f" 2>/dev/null || true; done
  for f in "$tmp"/s-*/pod/out/quality-server.pid; do
    [[ -f "$f" ]] && { kill -TERM -- "-$(cat "$f")" 2>/dev/null || kill -TERM "$(cat "$f")" 2>/dev/null || true; }
  done
  wait 2>/dev/null || true
  # CFINAL_KEEP_TMP=1 keeps the test's files (to look at a failed check)
  [[ -n "${CFINAL_KEEP_TMP:-}" ]] && echo "cfinal-test.sh: kept $tmp" || rm -rf -- "${tmp:?}"
}
trap cleanup EXIT
failed=0
check() {
  if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; failed=$((failed + 1)); fi
}
ports=()
free_port() {   # free_port VAR: a free port none of this test's other servers got
  local p
  while :; do
    p="$(python3 -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")"
    [[ " ${ports[*]} " == *" $p "* ]] || break
  done
  ports+=("$p")
  printf -v "$1" %s "$p"
}
export PYTHONDONTWRITEBYTECODE=1
python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" sparse "$HEADERS" "$tmp/model" > /dev/null

source "$STRATA_DIR/tools/dsv41/flavors.sh"
flavor_config 13.0
image="$(builder_image)"
docker image inspect "$image" > /dev/null 2>&1 || { echo "cfinal-test.sh: no $image (tools/dsv41/build.sh)" >&2; exit 2; }
d="$tmp/docker"
mkdir -p "$d"

echo "--- 1. the engine's default CPU numerics (the serve dry run, in the cu130 builder image) and the fake's"
printf 'BGEN 0 8 10,11,12,13\nBGEN 1 8 20,21,22\n' > "$d/two.txt"
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_BUILD="$STRATA_BUILD" -e E01="$E01" -e D="$d" \
  -v "$BUILD_ROOT:$BUILD_ROOT:ro" -v "$tmp:$tmp" "$image" bash -euo pipefail -c '
    mkdir -p /tmp/cudastub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1
    export LD_LIBRARY_PATH=/tmp/cudastub
    for n in default gpu fork; do
      a=()
      [ "$n" = default ] || a=(--cpu-numerics "$n")
      "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
        "${a[@]}" --dry-run < "$D/two.txt" > "$D/$n.out" 2> "$D/$n.err" || true
    done
    "$STRATA_BUILD/dsv41-engine" --help > "$D/help.txt" 2>&1 || true
    rc_=0
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --cpu-numerics cpu --dry-run < /dev/null \
      > "$D/bad.txt" 2>&1 || rc_=$?
    echo "$rc_" > "$D/bad.rc"' > "$tmp/docker.log" 2>&1 || rc=$?
check "the builder image ran the serve dry runs (exit $rc)" '(( rc == 0 )) && [[ -s "$d/default.out" ]]' || tail -n 20 "$tmp/docker.log"
load_line="^dsv41-engine serve: the CPU experts in the fork.s numerics (--cpu-numerics fork, the default)"
check "dsv41-engine: the fork's CPU numerics by default (its load line, as with --cpu-numerics fork), the GPU's with --cpu-numerics gpu (no load line); both answer the two requests in slots" \
  'grep -q "$load_line" "$d/default.err" && grep -q "$load_line" "$d/fork.err" && ! grep -q "$load_line" "$d/gpu.err" &&
   for n in default gpu fork; do (( $(grep -cE "^BDONE [01] 8 length " "$d/$n.out") == 2 )) || exit 1; done'
check "dsv41-engine --help: fork the default, gpu an option; --cpu-numerics cpu refused (exit $(cat "$d/bad.rc" 2>/dev/null))" \
  'grep -qE "^  --cpu-numerics M    the CPU experts. arithmetic: fork \(default( since|:)" "$d/help.txt" &&
   grep -q "or gpu$" "$d/help.txt" && [[ "$(cat "$d/bad.rc")" != 0 ]] && grep -q -- "--cpu-numerics cpu: expected gpu or fork" "$d/bad.txt"'
for n in default gpu; do
  a=()
  [[ "$n" == default ]] || a=(--cpu-numerics "$n")
  timeout 60 python3 "$here/fakes.py" --serve --tokenizer-gguf "$VOCAB" --batch 4 "${a[@]}" < /dev/null > /dev/null 2> "$tmp/fake-$n.err" || true
done
check "fakes.py's engine: the fork's numerics by default (the engine's load line), none with --cpu-numerics gpu" \
  'grep -q "$load_line" "$tmp/fake-default.err" && ! grep -q "$load_line" "$tmp/fake-gpu.err" && grep -q "loaded in" "$tmp/fake-gpu.err"'

echo "--- 2. pod/e07.sh --dry-run with E07_PLAN=cfinal: its defaults (the model from HF), then the network volume"
plan=cfinal
# path: the dry run's model path; its environment: cfinal's defaults, or the volume only
for path in hf volume; do
  env_=()
  [[ "$path" == hf ]] || env_=(NETWORK_VOLUME=testvolume01 HF_CLOUDS=)
  rc=0; env "${env_[@]}" E07_PLAN=$plan OUT="$tmp/dry-$path" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run \
    > "$tmp/dry-$path.txt" 2>&1 || rc=$?
  echo "$rc" > "$tmp/dry-$path.rc"
  grep -E "^plan \((volume|hf)\)|^price limit|^GPUs, in order|^e07.sh: " "$tmp/dry-$path.txt" | sed "s/^/  $path: /" | cut -c1-220 || true
  grep -A1 -E "^plan \($path\) latest end: " "$tmp/dry-$path.txt" | tail -n 1 | sed "s/^/  $path: /" | cut -c1-220 || true
done
f="$tmp/dry-hf.txt" v="$tmp/dry-volume.txt"
check "the $plan dry run with its defaults exits 0 (exit $(cat "$tmp/dry-hf.rc")), with the volume only exits 0 (exit $(cat "$tmp/dry-volume.rc"))" \
  '[[ "$(cat "$tmp/dry-hf.rc")" == 0 && "$(cat "$tmp/dry-volume.rc")" == 0 ]]' || tail -n 10 "$f" "$v"
check "$plan's defaults: the RTX PRO 6000 only, the two HF offers in any data center (SECURE, COMMUNITY), the volume not offered, 2.60 USD/h, the session cap 3.35 USD over at most 3 pods, the qualification first" \
  'grep -q "^GPUs, in order: NVIDIA RTX PRO 6000 Blackwell Server Edition;NVIDIA RTX PRO 6000 Blackwell Workstation Edition; host RAM at least 256 GB" "$f" &&
   [[ "$(grep -A2 "per GPU the offers in order:$" "$f" | tail -n 2 | tr "\n" "|")" == "  SECURE in any data center, no network volume: the model downloaded from HF (vcruz305/DeepSeek-V4.1-Flash-GGUF at c4a08554)|  COMMUNITY in any data center, no network volume: the model downloaded from HF (vcruz305/DeepSeek-V4.1-Flash-GGUF at c4a08554)|" ]] &&
   grep -qx "plan (volume): not offered (NETWORK_VOLUME is empty): shown, not checked" "$f" &&
   grep -q "^price limit: 2.60 USD/h; session cap: 3.35 USD over at most 3 pods" "$f" &&
   [[ "$(grep -A1 "^plan with the model from HF" "$f" | tail -n 1)" == "  create, SSH, CUDA probe, qualification"* ]]'
check "$plan with NETWORK_VOLUME=testvolume01 HF_CLOUDS=: the volume's offer only (read only), the HF plan not offered" \
  '[[ "$(grep -A1 "per GPU the offers in order:$" "$v" | tail -n 1)" == "  SECURE in EUR-IS-1 with network volume testvolume01 at /workspace (read only): the model copied from it" ]] &&
   [[ "$(grep -A2 "per GPU the offers in order:$" "$v" | tail -n 1)" == "machines excluded by BAD_MACHINES "* ]] &&
   [[ "$(grep -A3 "per GPU the offers in order:$" "$v" | tail -n 1)" == "price limit: "* ]] &&
   grep -qx "plan (hf): not offered (HF_CLOUDS is empty): shown, not checked" "$v"'
sed -n "/^plan with the model from HF /,/^plan (hf)/p" "$f" | awk '/^  / { print $1 }' > "$tmp/planned-hf.txt"
sed -n "/^plan with the model from the volume /,/^plan (volume)/p" "$v" | awk '/^  / { print $1 }' > "$tmp/planned-volume.txt"
tail_="engine-server load-short steady-c4 quality-venv continuation lm-eval-ifeval lm-eval-gsm8k_cot_llama lm-eval-humaneval_instruct server-stop"
want_hf="create, payload env model-headers gate setup model-download dry-run $tail_ pull"
want_volume="create, payload env volume-before gate setup model-copy dry-run $tail_ volume-after pull"
check "$plan: the HF plan's steps (no batching-off server, no ab-gpu, det-batch, det-sampling nor fork server): $want_hf" \
  '[[ "$(tr "\n" " " < "$tmp/planned-hf.txt")" == "$want_hf " ]]' || echo "  planned: $(tr "\n" " " < "$tmp/planned-hf.txt")"
check "$plan: the volume plan's steps: $want_volume" \
  '[[ "$(tr "\n" " " < "$tmp/planned-volume.txt")" == "$want_volume " ]]' || echo "  planned: $(tr "\n" " " < "$tmp/planned-volume.txt")"
check "$plan: every pod step of both model paths with its timeout, 2 x its estimate; the lm-eval tasks from the fork's numerics' own times, steady-c4 3 repeats, the server's load measured" \
  'for x in "$f:hf:HF" "$v:volume:the volume"; do
     IFS=: read -r file p name <<<"$x"
     w="want_$p"
     (( $(sed -n "/^plan with the model from $name /,/^plan ($p)/p" "$file" | grep -c "; timeout min(") == $(wc -w <<<"${!w}") - 4 )) || exit 1
     grep -qE "^  lm-eval-ifeval \(NUM_CONCURRENT 4 on the batch slots, the reference run.s 1077 s in the fork.s numerics one at a time; the GPU.s 1521 s on 2026-10-19 within its timeout\) +17.9  \(total [0-9.]+; timeout min\(35.8 min, the time left\)\)$" "$file" || exit 1
     grep -qE "^  lm-eval-gsm8k_cot_llama \(NUM_CONCURRENT 4 on the batch slots, the reference run.s 531 s in the fork.s numerics one at a time; the GPU.s 909 s on 2026-10-19 within its timeout\) +8.8  \(total [0-9.]+; timeout min\(17.6 min" "$file" || exit 1
     grep -qE "^  lm-eval-humaneval_instruct \(NUM_CONCURRENT 4 on the batch slots, the reference run.s 359 s in the fork.s numerics one at a time x 1.25; the GPU.s had 162 of 164 at its 720 s cut on 2026-10-19\) +7.5  \(total [0-9.]+; timeout min\(15.0 min" "$file" || exit 1
     grep -qE "^  steady-c4 \(chat_load.py steady, 3 x: " "$file" || exit 1
     grep -qE "^  engine-server \(--batch 8 .*85-140 s measured in the cbatch sessions\) +2.5  \(total [0-9.]+; timeout min\(5.0 min" "$file" || exit 1
   done'
for p in hf volume; do
  file="$tmp/dry-$p.txt"
  usd="$(sed -n "s/^plan ($p): total .* (about \([0-9.]*\) USD at 2.60 USD\/h)$/\1/p" "$file")"
  mins="$(sed -n "s/^plan ($p): total \([0-9.]*\) min .*/\1/p" "$file")"
  end="$(sed -n "s/^plan ($p) latest end: .*, ends at minute \([0-9.]*\), the latest of any step,$/\1/p" "$file")"
  dl="$(grep -A1 "^plan ($p) latest end: " "$file" | sed -n "s/^  of the pod.s \([0-9]*\)-minute deadline: .*/\1/p")"
  echo "  $p: ${mins:-none} min, ${usd:-none} USD at 2.60 USD/h; latest end minute ${end:-none} of the ${dl:-none}-minute deadline"
  check "$plan ($p): the plan under 3.35 USD at 2.60 USD/h (${usd:-none} USD, ${mins:-none} min), its latest end (minute ${end:-none}) covered and >= 2 min before the pod's ${dl:-none}-minute deadline (HumanEval's whole timeout inside)" \
    '[[ -n "$usd" && -n "$end" && -n "$dl" ]] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 3.35 else 1)" "$usd" &&
     grep -A1 "^plan ($p) latest end: " "$file" | grep -q ": every step covered;" &&
     python3 -c "import sys; sys.exit(0 if float(sys.argv[2]) - float(sys.argv[1]) >= 2.0 else 1)" "$end" "$dl"'
done
check "$plan: the payload's sources: the tarball's engine HEAD's, run_lm_eval.sh, the quality runs' subsets; no det_batch_check.py nor det_sampling_check.py" \
  'grep -q "the tarball.s engine is HEAD.s$" "$f" &&
   grep -q "src/tools/dsv41/quality/run_lm_eval.sh: last changed in " "$f" &&
   ! grep -qE "src/tools/dsv41/quality/det_(batch|sampling)_check.py: last changed in " "$f" &&
   grep -q "its lm-eval subsets are src/tools/dsv41/quality/subsets_lean2h_seed20261006.json, the reference run.s" "$f"'

echo "--- 3. the $plan rehearsal from its payload on the volume's path"
bin="$tmp/bin" stubs="$tmp/stubs"
mkdir -p "$bin" "$stubs"
# c05-test.sh's stubs: the GPU, numactl, pip, the download tools, the engine, the volume copy, the gate at a 300 GB host
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# cfinal-test.sh's GPU: an RTX PRO 6000's 97887 MiB, GPU0 on NUMA node 1 (a fake two-node host)
case " $* " in
  *" topo "*) printf '\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t0-11\t1\t\tN/A\n' ;;
  *" --query-gpu=memory.total --format=csv,noheader,nounits "*) echo 97887 ;;
  *" --query-gpu=memory.used,memory.total --format=csv,noheader "*) echo "1024 MiB, 97887 MiB" ;;
  *) echo "nvidia-smi (cfinal-test.sh stub): GPU 0, 97887 MiB" ;;
esac
EOF
# numactl refuses the memory policy as Docker's default seccomp profile does without CAP_SYS_NICE (the
# 20261019T203344Z session's EPYC 9655 pod ran unbound)
cat > "$bin/numactl" <<'EOF'
#!/usr/bin/env bash
[[ " $* " != *" --preferred="* ]] ||
  { printf 'set_mempolicy: Operation not permitted\nsetting preferred node: Operation not permitted\n' >&2; exit 1; }
while [[ "${1:-}" == --* ]]; do shift; done
exec "$@"
EOF
# two NUMA nodes, node 1's CPUs the ones this shell may run on
mkdir -p "$tmp/numa/node0" "$tmp/numa/node1"
echo 4095 > "$tmp/numa/node0/cpulist"
grep Cpus_allowed_list /proc/self/status | cut -f2 > "$tmp/numa/node1/cpulist"
printf '#!/usr/bin/env bash\necho "pip (cfinal-test.sh stub, nothing installed): $*"\n' > "$bin/pip"
for t in aria2c apt-get; do
  printf '#!/usr/bin/env bash\necho "%s (cfinal-test.sh stub): nothing downloads or installs here: $*" >&2\nexit 1\n' "$t" > "$bin/$t"
done
cat > "$stubs/dsv41-engine" <<'EOF'
#!/usr/bin/env bash
# cfinal-test.sh's engine: --dry-run exits 0, --serve is fakes.py's engine (the tokenizer from --model)
case " $* " in *" --dry-run "*) echo "dsv41-engine (cfinal-test.sh stub): dry run of $*"; exit 0 ;; esac
args=()
for a in "$@"; do [[ "$a" == --serve ]] || args+=("$a"); done
exec python3 "$(dirname "$0")/src/tools/dsv41/quality/fakes.py" --serve "${args[@]}"
EOF
cat > "$stubs/model-copy.sh" <<'EOF'
#!/usr/bin/env bash
# cfinal-test.sh's copy (q02-test.sh's): model-copy.sh SRC DST, the shards copied sparse (SRC only read)
set -euo pipefail
mkdir -p "$2"
cp --sparse=always "$1"/*.gguf "$2"/
echo "model-copy.sh (cfinal-test.sh stub): $(ls "$2" | wc -l) shards from $1"
EOF
cat > "$stubs/e07_gate.py" <<'EOF'
#!/usr/bin/env python3
"""cfinal-test.sh's gate: the payload's own e07_gate.py (e07_gate_real.py) at a 300 GB host."""
import os
import sys

a = sys.argv[1:]
for key, value in (("--cgroup-bytes", "300000000000"), ("--mem-total-kib", "300000000")):
    a[a.index(key) + 1] = value
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e07_gate_real.py"), *a])
EOF
chmod +x "$bin"/* "$stubs"/*
rc=0; NETWORK_VOLUME=testvolume01 HF_CLOUDS= CBATCH_STEADY_WINDOW_S=4 E07_PLAN=$plan OUT="$tmp/payload" \
  bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --payload-only > "$tmp/payload.log" 2>&1 || rc=$?
check "e07.sh --payload-only builds the $plan payload, e07/plan cfinal, e07/cdet-quality: 4 concurrent requests, ifeval first" \
  '(( rc == 0 )) && [[ -f "$tmp/payload/payload-cu130.tar.gz" ]] &&
   [[ "$(tar -xzOf "$tmp/payload/payload-cu130.tar.gz" e07/plan)" == cfinal ]] &&
   [[ "$(tar -xzOf "$tmp/payload/payload-cu130.tar.gz" e07/cdet-quality)" == "4 ifeval gsm8k_cot_llama humaneval_instruct" ]]' ||
  tail -n 5 "$tmp/payload.log"
s="$tmp/s-$plan"
mkdir -p "$s/pod/out" "$s/root/root"
tar -xzf "$tmp/payload/payload-cu130.tar.gz" -C "$s/pod" --exclude='libcublas*' --exclude='libggml-cuda.so*'
rm -rf -- "$tmp/payload"
cp "$stubs/dsv41-engine" "$stubs/model-copy.sh" "$s/pod/"
# the network volume at the pod's /workspace: the sparse shards, only read (volume-before against volume-after)
mkdir -p "$s/root/workspace/models/q2k"
cp --sparse=always "$tmp/model"/*.gguf "$s/root/workspace/models/q2k/"
mv "$s/pod/src/tools/dsv41/pod/e07_gate.py" "$s/pod/src/tools/dsv41/pod/e07_gate_real.py"
cp "$stubs/e07_gate.py" "$s/pod/src/tools/dsv41/pod/"
ln -s "$QUALITY_VENV" "$s/root/root/venv-quality"
# q02-test.sh's 3 items per task (eval.sh's SAMPLES)
python3 -c "import json, sys; d = json.load(open(sys.argv[1])); json.dump({k: v[:3] for k, v in d.items()}, open(sys.argv[2], 'w'))" \
  "$here/subsets_lean2h_seed20261006.json" "$tmp/samples3.json"
free_port port; free_port proxy_port
t0=$(date +%s)
( cd "$s/pod" && export PATH="$bin:$PATH" E07_POD_ROOT="$s/root" E07_Q_PORT="$port" PROXY_PORT="$proxy_port" \
    NUMA_ROOT="$tmp/numa" POD_MACHINE="cfinal-test" FAKE_ENGINE_PROMPT_MS_PER_TOKEN="$P_MS" \
    FAKE_ENGINE_DECODE_MS_PER_TOKEN="$D_MS" FAKE_ENGINE_BATCH_ROW_MS="$R_MS" SAMPLES="$tmp/samples3.json" \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  rc=0; DEADLINE=$(( $(date +%s) + 3000 )) POD_VOLUME=cfinal-rehearsal timeout 3300 bash e07-run.sh > out/e07-run.out 2>&1 < /dev/null || rc=$?
  echo "$rc" > out/e07-run.exit ) &
jobs_+=($!)
wait "${jobs_[@]}"
jobs_=()
echo "  the rehearsal took $(( $(date +%s) - t0 )) s"
o="$s/pod/out"
awk '/^step / { sub(":$", "", $2); print $2 }' "$o/steps.log" > "$tmp/ran.txt"
echo "  $plan: $(tr '\n' ' ' < "$tmp/ran.txt")"
grep -vxE 'create,|payload|pull' "$tmp/planned-volume.txt" > "$tmp/planned-pod.txt" || true
check "$plan: e07-run.sh exits 0, every step exit 0" \
  '[[ "$(cat "$o/e07-run.exit")" == 0 ]] && ! grep "^step " "$o/steps.log" | grep -qv ": exit 0, "' ||
  { grep "^step " "$o/steps.log" | grep -v ": exit 0, " | sed 's/^/  /'; tail -n 5 "$o/e07-run.out" | sed 's/^/  /'; }
check "$plan: steps.log has every planned step in order, then done" \
  'diff "$tmp/planned-pod.txt" "$tmp/ran.txt" > /dev/null && [[ "$(tail -n 1 "$o/steps.log")" == done ]]'
check "$plan: the batched server as planned: gate's placement, --batch 8, --log-ids, the deterministic mode on, the engine's default numerics (no --cpu-numerics; the fake's log names the fork's)" \
  'python3 -c "import json, sys; a = json.load(open(sys.argv[1]))[\"args\"]; g = json.load(open(sys.argv[2]))
v = lambda k: a[a.index(k) + 1]
sys.exit(0 if v(\"--batch\") == \"8\" and \"--log-ids\" in a and \"--deterministic\" not in a and \"--cpu-numerics\" not in a and
         v(\"--vram-gb\") == str(g[\"vram_gb\"]) else 1)" "$o/quality-server.json" "$o/gate/gate.json" &&
   grep -q "$load_line" "$o/quality-engine.log" && grep -qE "^dsv41-engine serve: request [0-9]+: batch slot [0-9]+: " "$o/quality-engine.log"'
check "$plan: --preferred refused (its stderr on one line), the engine bound with numactl --cpunodebind=1, the lines in placement.txt and env.log, the server under it" \
  'refused="binding: numactl --cpunodebind=1 --preferred=1 refused, exit 1: set_mempolicy: Operation not permitted; setting preferred node: Operation not permitted"
   worked="binding: numactl --cpunodebind=1 worked (node 1'"'"'s CPUs; the default local policy first-touches the memory there)"
   grep -q "^engine runs bound with: numactl --cpunodebind=1; placement: --hot-set out/gate/hot-set.json " "$o/placement.txt" &&
   for x in placement.txt env.log; do grep -qxF "$refused" "$o/$x" && grep -qxF "$worked" "$o/$x" &&
     grep -q "^binding: this process: Seccomp " "$o/$x" || exit 1; done &&
   grep -q "^engine pool threads (--threads 16) run on CPUs $(cat "$tmp/numa/node1/cpulist") (bound with: numactl --cpunodebind=1)$" "$o/env.log" &&
   grep -q "numactl --cpunodebind=1 " "$o/quality-server.cmd"' ||
  sed 's/^/  /' "$o/placement.txt"
check "$plan: load-short and steady-c4 (3 repeats) came back; no det-batch, det-sampling, ab-gpu, ab-fork nor fork server output" \
  'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[2]))
sys.exit(0 if j[\"ok\"] and s[\"ok\"] and s[\"users\"] == 4 and s[\"summary\"][\"measured\"] == 3 else 1)" "$o/cbatch/load-short.json" "$o/cbatch/steady-c4.json" &&
   for x in det-batch.json det-sampling.json cbatch/ab-gpu.json cbatch/ab-fork.json fork-quality-server.cmd gate-off; do [[ ! -e "$o/$x" ]] || exit 1; done'
check "$plan: lm-eval's three tasks at NUM_CONCURRENT 4 on 3 items each, each rc=0" \
  '(for t in ifeval gsm8k_cot_llama humaneval_instruct; do ls "$o"/quality/lm_eval/"$t"/*/samples_"$t"_*.jsonl > /dev/null 2>&1 &&
      grep -qx "$t rc=0" "$o/quality/lm_eval_status.txt" || exit 1; done) &&
   (( $(grep -cx "num_concurrent 4" "$o/quality/settings.txt") == 3 ))'

echo "--- 4. e07_report.py's $plan sheet"
f="$tmp/report/20261020T090000Z"
mkdir -p "$f"
cp -r "$o" "$f/out"
echo 3.35 > "$f/session-cap"
echo "pod cfinalstub: 3000 s at 2.6 USD/h = 2.167 USD (upper bound: billing rounds up)" > "$f/cost.txt"
echo "{\"id\": \"cfinalstub\", \"machineId\": \"cfinal-test\"}" > "$f/pod.json"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/sheets" > "$tmp/report.log" 2>&1 || true
r="$tmp/sheets/2026-10-20-$plan/README.md"
v4="$(sed -n 's/^Batched quality bar (.*: \*\*\([A-Za-z ]*\)\*\* .*/\1/p' "$r" 2>/dev/null)"
check "the $plan sheet: item 11 (the single user against session 4's 61.55, steady-c4 against the single user with each user >= 20), item 3 batched, item 4 against the single-user engine run (${v4:-none}), item 14 from them (item 4's verdict, item 3 whole), no det-batch, items 7 and 8 nor fork arm" \
  'grep -qE "^# Final session: the fast path.s speed and the quality of batched requests \(20261020[0-9TZ]*\)$" "$r" &&
   grep -q "^## Speed (item 11)$" "$r" &&
   grep -qE "^- the single user with batching on: (PASS|FAIL): [0-9.]+ tok/s against session 4.s 61.55 " "$r" &&
   grep -qE "^- steady-c4: (PASS|FAIL): [0-9.]+ tok/s in aggregate .* the single user on the batched server \(its solo path.* the slowest user [0-9.]+ tok/s \(goal: each >= 20\)$" "$r" &&
   grep -q "^## Item 3 batched (logs/quality)$" "$r" && grep -q "^## Item 4$" "$r" &&
   grep -q "^| engine batched - engine alone | n | alone | batched | diff | 95% CI | flips |$" "$r" &&
   [[ -n "$v4" && "$v4" != "not evaluated" ]] &&
   grep -q "^## Item 14 (no quality loss from batching)$" "$r" &&
   grep -qE "^Item 14 \(item 3 batched on all three tasks and item 4 against the reference engine run\): \*\*$v4\*\* \(item 3 batched on all three tasks; item 4 against the reference engine run $v4: .*\)\.$" "$r" &&
   grep -q "^## Server as run, batching on$" "$r" &&
   ! grep -qE "^## (Item 14 \(logs/det-batch.json\)|Items 7 and 8|Server as run, the A/B)" "$r" && ! grep -q "the speed A/B" "$r"' ||
  { tail -n 5 "$tmp/report.log"; grep -E "^(- |Goal item|Batched quality bar)" "$r" 2>/dev/null | sed 's/^/  /' | cut -c1-220; }
grep -E "^(Item 14|Batched quality bar) " "$r" 2>/dev/null | sed 's/^/  /' | cut -c1-240 || true
# item 3 not whole: humaneval's status line a timeout, as the 2026-10-19 cdet session's
f2="$tmp/report/20261021T090000Z"
cp -r "$f" "$f2"
sed -i 's/^humaneval_instruct rc=0$/humaneval_instruct rc=124/' "$f2/out/quality/lm_eval_status.txt"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f2" "$tmp/sheets" > "$tmp/report2.log" 2>&1 || true
r2="$tmp/sheets/2026-10-21-$plan/README.md"
check "the $plan sheet with humaneval cut (rc=124): item 14 not evaluated, naming item 3's missing task" \
  'grep -qx "Item 14 (item 3 batched on all three tasks and item 4 against the reference engine run): \*\*not evaluated\*\* (item 3 did not come back on humaneval_instruct (humaneval_instruct rc=124))." "$r2"' ||
  grep "^Item 14" "$r2" 2>/dev/null | sed 's/^/  /'
if (( failed )); then
  echo "  the tail of e07-run.out and steps.log:"
  tail -n 5 "$o/e07-run.out" "$o/steps.log" 2>/dev/null | sed 's/^/    /' || true
fi

echo "--- 5. pod/sheet_banners_test.py (items 11 and 14 on the final batch determinism sheet)"
python3 "$STRATA_DIR/tools/dsv41/pod/sheet_banners_test.py" -v > "$tmp/sheet-banners.out" 2> "$tmp/sheet-banners.log" || true
check "sheet_banners_test.py: OK (items 11 and 14 on the final batch determinism sheet, every earlier one bannered)" \
  '[[ "$(tail -n 1 "$tmp/sheet-banners.log")" == OK ]]' || tail -n 5 "$tmp/sheet-banners.log"

echo "cfinal-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
