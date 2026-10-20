#!/usr/bin/env bash
# The concurrency baseline's checks without a GPU: no model, no RunPod, no HF, no money.
#   1. quality/chat_load.py build on the real vocabulary: 8 conversations of 16384 tokens +- 2%, each turn 1 at most
#      512 new tokens, no two sharing a prompt token past the chat template's head (checked again here by tokenizing
#      every pair's first request), the same file twice
#   2. the server's tests (python3 -m unittest discover serve, in the builder image as verify.sh runs them): QueueWait
#      (a request's strata_queue_ms and its done line's "queued", alone and behind another) and every other test
#   3. chat_load.py run against serve/server.py on fakes.py's engine with its delays (FAKE_ENGINE_PROMPT_MS_PER_TOKEN
#      0.3, FAKE_ENGINE_DECODE_MS_PER_TOKEN 5) for N = 1 and 2: the table by N printed, the JSON with every request's
#      row, every reply 256 tokens, the queue wait reported for each, the server's done lines with it, and each
#      summary within 10% of quality/c01_expect.py's FIFO model of the fake's delays
#   4. pod/e07.sh --dry-run with E07_PLAN=cbase: exit 0, the RTX PRO 6000 only, the machine qualification listed
#      before the payload goes up and as the plan's first step, every pod step of both model paths with its timeout
#      (2 x its estimate), the HF plan under 2.00 USD at 2.60 USD/h, the payload's sources (chat_load.py and the
#      conversations)
#   5. the cbase rehearsal: pod/e07.sh --payload-only's cbase payload unpacked, then pod/e07-run.sh as pod-session.sh
#      starts it (DEADLINE, no volume), with q03-test.sh's stubs where a pod has hardware, the model or the network, and
#      dsv41-engine --serve as fakes.py's engine with the delays above: e07-run.sh exits 0, every planned step in order
#      with exit 0, then done; each load-c<N>.json within 10% of the FIFO model; at N = 4 the first turns queue one
#      behind the other (the 4th user's first TTFT 3 requests' reads and replies plus its own read); the server
#      stopped; pod/e07_report.py's cbase sheet has the table by N
# Exits 1 if any check fails.
#
#   bash tools/dsv41/quality/c01-test.sh
#
# BUILD_ROOT (default $HOME/dsv41-build) holds the vocabulary (vocab/deepseek41-vocab.gguf), the real headers
# (engine/headers) and the builds and tarballs the dry run and the payload read.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
# e07.sh has no default network volume: its dry runs and rehearsals here plan one by this placeholder id
export NETWORK_VOLUME="${NETWORK_VOLUME-testvolume01}"
HEADERS="${HEADERS:-$BUILD_ROOT/engine/headers}"
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"
for f in "$HEADERS/head-7.bin" "$VOCAB"; do
  [[ -f "$f" ]] || { echo "c01-test.sh: no $f" >&2; exit 2; }
done
# the fake engine's delays: a 16K prompt read in ~4.9 s (the RTX PRO 6000's measured 4.3-4.6 s), 200 tok/s of decode
P_MS=0.3 D_MS=5
tmp="$(mktemp -d)"
jobs_=()
cleanup() {
  local f
  for f in "${jobs_[@]}"; do kill "$f" 2>/dev/null || true; done
  for f in "$tmp"/server.pid "$tmp"/s-*/pod/out/quality-server.pid; do
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

echo "--- 1. chat_load.py build on the real vocabulary ($VOCAB)"
rc=0
python3 "$here/chat_load.py" build --vocab "$VOCAB" --out "$tmp/conv.json" > "$tmp/build.txt" 2>&1 || rc=$?
python3 "$here/chat_load.py" build --vocab "$VOCAB" --out "$tmp/conv-again.json" > /dev/null 2>&1 || rc=$?
sed 's/^/  /' "$tmp/build.txt"
check "the build exits 0 and writes the same file twice" \
  '(( rc == 0 )) && cmp -s "$tmp/conv.json" "$tmp/conv-again.json"'
check "8 conversations of 16384 tokens +- 2%, each turn 1 at most 512 new tokens" \
  'python3 -c "import json, sys; c = json.load(open(sys.argv[1]))[\"conversations\"]
sys.exit(0 if len(c) == 8 and all(abs(x[\"tokens\"] - 16384) <= 0.02 * 16384 and 0 < x[\"turn1_new_tokens\"] <= 512
                                  for x in c) else 1)" "$tmp/conv.json"'
# every pair's first request tokenized again: what they share is the template's head, no more
python3 - "$tmp/conv.json" "$VOCAB" "$here" > "$tmp/distinct.txt" 2>&1 <<'EOF' || true
import json
import sys
sys.path.insert(0, sys.argv[3])
import chat_load
d = json.load(open(sys.argv[1]))
t = chat_load.Template(sys.argv[2])
head = chat_load.common_prefix(*(t.ids([{"role": "user", "content": x}], True) for x in ("A", "B")))
ids = [t.ids(c["messages"] + [{"role": "user", "content": c["turns"][0]}], True) for c in d["conversations"]]
shared = max(chat_load.common_prefix(a, b) for i, a in enumerate(ids) for b in ids[i + 1:])
names = len({c["name"] for c in d["conversations"]})
print(f"distinct: {names} names, {len(ids)} first requests of {min(map(len, ids))}-{max(map(len, ids))} tokens, at most "
      f"{shared} shared by two of them, the template's head {head}")
sys.exit(0 if shared <= head and names == len(ids) == 8 else 1)
EOF
sed 's/^/  /' "$tmp/distinct.txt"
check "distinct per user: no two first requests share more than the chat template's head" \
  'grep -q "^distinct: 8 names, 8 first requests of .* the template.s head [0-9]*$" "$tmp/distinct.txt" &&
   python3 -c "import re, sys; m = re.search(r\"at most (\d+) shared .* head (\d+)$\", open(sys.argv[1]).read().strip())
sys.exit(0 if m and int(m.group(1)) <= int(m.group(2)) else 1)" "$tmp/distinct.txt"'

source "$STRATA_DIR/tools/dsv41/flavors.sh"
flavor_config 13.0
image="$(builder_image)"
docker image inspect "$image" > /dev/null 2>&1 || { echo "c01-test.sh: no $image (tools/dsv41/build.sh)" >&2; exit 2; }
mkdir -p "$tmp/docker"
echo "--- 2. the server's tests (python3 -m unittest discover serve in the builder image, as verify.sh runs them)"
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e D="$tmp/docker" \
  -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$tmp/docker:$tmp/docker" "$image" bash -euo pipefail -c '
    mkdir -p /tmp/serve-tests && cd /tmp/serve-tests
    export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STRATA_DIR"
    python3 -m unittest discover -s "$STRATA_DIR/serve" -t "$STRATA_DIR" -p "test_*.py" > "$D/serve-tests.log" 2>&1 || true
    python3 -m unittest -v serve.test_server.QueueWait > "$D/serve-c01.log" 2>&1 || true' \
  > "$tmp/docker-serve.log" 2>&1 || rc=$?
grep -E "^(FAIL|ERROR):" "$tmp/docker/serve-tests.log" | sed 's/^/  /' || true
tail -n 3 "$tmp/docker/serve-tests.log" | sed 's/^/  /' || true
check "serve/test_*.py pass (docker exit $rc)" '(( rc == 0 )) && [[ "$(tail -n 1 "$tmp/docker/serve-tests.log")" == "OK"* ]]'
grep -oE "^test_[a-z0-9_]+ \(serve\.test_server\.QueueWait[^)]*\)" "$tmp/docker/serve-c01.log" | sed 's/^/  /' || true
check "the queue wait server tests: the queue wait alone (none) and behind another (the first reply's length), streamed too" \
  'grep -q "^Ran 2 tests in " "$tmp/docker/serve-c01.log" && [[ "$(tail -n 1 "$tmp/docker/serve-c01.log")" == OK ]]'

echo "--- 3. chat_load.py run against serve/server.py on the fake engine (${P_MS} ms per prompt token, ${D_MS} ms per token)"
mkdir -p "$tmp/direct"
cat > "$tmp/direct/engine" <<EOF
#!/usr/bin/env bash
exec python3 "$here/fakes.py" --serve "\$@"
EOF
chmod +x "$tmp/direct/engine"
python3 -c "import json, sys; d = sys.argv[1]
print(json.dumps({'exe': d + '/engine', 'cwd': d, 'args': ['--model', sys.argv[2], '--ctx', '262144'],
  'model_profile': 'deepseek-v4.1', 'tokenizer_gguf': sys.argv[2], 'log': d + '/engine.log', 'repeat_stop_tokens': 0}))" \
  "$tmp/direct" "$VOCAB" > "$tmp/direct/config.json"
free_port port
( cd "$STRATA_DIR" && FAKE_ENGINE_PROMPT_MS_PER_TOKEN=$P_MS FAKE_ENGINE_DECODE_MS_PER_TOKEN=$D_MS exec setsid python3 -m serve.server \
    --engine strata --config "$tmp/direct/config.json" --port "$port" ) > "$tmp/direct/server.log" 2>&1 < /dev/null &
echo $! > "$tmp/server.pid"
for _ in $(seq 120); do grep -q "ready: http" "$tmp/direct/server.log" && break; sleep 1; done
rc=0
timeout 600 python3 "$here/chat_load.py" run --url "http://127.0.0.1:$port" --conversations "$tmp/conv.json" --users 1,2 \
  --out "$tmp/direct/load.json" > "$tmp/direct/load.txt" 2>&1 || rc=$?
sed 's/^/  /' "$tmp/direct/load.txt" | cut -c1-200
check "chat_load.py run exits 0, prints the table by N (1 and 2) and writes the JSON" \
  '(( rc == 0 )) && grep -q "^| N | requests | TTFT p50 s | TTFT p95 s | queue p50 s | queue p95 s |" "$tmp/direct/load.txt" &&
   grep -q "^| 1 | 2 | " "$tmp/direct/load.txt" && grep -q "^| 2 | 4 | " "$tmp/direct/load.txt" && [[ -s "$tmp/direct/load.json" ]]'
check "every request's row: 256 tokens (length), its queue wait, TTFT, decode tok/s, prompt and cached tokens; N = 1's turn 2 cached" \
  'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); rows = [r for x in j[\"runs\"] for r in x[\"rows\"]]
one = {r[\"turn\"]: r for r in j[\"runs\"][0][\"rows\"]}
sys.exit(0 if j[\"ok\"] and len(rows) == 6 and all(r[\"completion_tokens\"] == 256 and r[\"finish\"] == \"length\" and
         r[\"queue_s\"] is not None and r[\"ttft_s\"] > 0 and r[\"decode_tok_s\"] and r[\"new_tokens\"] <= 512 for r in rows)
         and one[2][\"cached_tokens\"] > 16384 else 1)" "$tmp/direct/load.json"'
check "the server's done lines say each request's queue wait" \
  '(( $(grep -cE "^\[strata\] done: 256 tokens in .*\(length, cancel=False\), queued [0-9]+\.[0-9]{2} s" "$tmp/direct/server.log") == 6 ))'
python3 "$here/c01_expect.py" "$tmp/direct/load.json" --prompt-ms "$P_MS" --decode-ms "$D_MS" > "$tmp/direct/expect.txt" 2>&1 || true
sed 's/^/  /' "$tmp/direct/expect.txt" | cut -c1-200
check "every summary within 10% of the FIFO model of the fake's delays (c01_expect.py)" \
  'grep -qx "c01_expect.py: PASS" "$tmp/direct/expect.txt"'
kill -TERM -- "-$(cat "$tmp/server.pid")" 2>/dev/null || true
rm -f "$tmp/server.pid"

echo "--- 4. pod/e07.sh --dry-run, E07_PLAN=cbase"
rc=0; E07_PLAN=cbase OUT="$tmp/dry-out" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry.txt" 2>&1 || rc=$?
grep -E "^plan \((volume|hf)\): total|^price limit|^GPUs, in order" "$tmp/dry.txt" | sed 's/^/  /' | cut -c1-200 || true
check "the cbase dry run exits 0 (exit $rc)" '(( rc == 0 ))' || tail -n 10 "$tmp/dry.txt"
check "the RTX PRO 6000 only (both editions, no A100)" \
  'grep -q "^GPUs, in order: NVIDIA RTX PRO 6000 Blackwell Server Edition;NVIDIA RTX PRO 6000 Blackwell Workstation Edition; host RAM at least 256 GB" "$tmp/dry.txt" &&
   ! grep -q "A100.* takes payload" "$tmp/dry.txt"'
q_line="$(grep -n "^  qualification, when" "$tmp/dry.txt" | cut -d: -f1 || true)"
up_line="$(grep -n "payload-<tag>.tar.gz to /root/e07.tar.gz" "$tmp/dry.txt" | cut -d: -f1 || true)"
check "the machine qualification is listed before the payload goes up, and is each plan's first step" \
  '[[ -n "$q_line" && -n "$up_line" ]] && (( q_line < up_line )) &&
   [[ "$(grep -A1 "^plan with the model from HF" "$tmp/dry.txt" | tail -n 1)" == "  create, SSH, CUDA probe, qualification"* ]]'
sed -n "/^plan with the model from HF /,/^plan (hf)/p" "$tmp/dry.txt" | awk '/^  / { print $1 }' > "$tmp/planned.txt"
want_hf="create, payload env model-headers gate setup model-download dry-run engine-server load-c1 load-c2 load-c4 load-c8 server-stop pull"
check "the HF plan's steps: $want_hf" '[[ "$(tr "\n" " " < "$tmp/planned.txt")" == "$want_hf " ]]'
check "every pod step of both model paths (12 from the volume, 11 from HF) has its timeout, 2 x its estimate" \
  '(( $(sed -n "/^plan with the model from /,/^plan (hf)/p" "$tmp/dry.txt" | grep -c "; timeout min(") == 23 )) &&
   grep -q "^  load-c8 (chat_load.py, N = 8: 2 turns each, 16 requests one at a time at 14.2 s, + 0.2) *4.0  (total [0-9.]*; timeout min(8.0 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  engine-server (.*  *6  (total [0-9.]*; timeout min(12 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  model-download (264.5 GB at 0.25 GB/s, .*18.2  (total [0-9.]*; timeout min(36.4 min, the time left))$" "$tmp/dry.txt"'
usd="$(sed -n "s/^plan (hf): total .* (about \([0-9.]*\) USD at 2.60 USD\/h)$/\1/p" "$tmp/dry.txt")"
check "the HF plan's estimate under 2.00 USD at 2.60 USD/h (${usd:-none} USD), the session cap 2.00 over at most 3 pods" \
  '[[ -n "$usd" ]] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 2.0 else 1)" "$usd" &&
   grep -q "^price limit: 2.60 USD/h; session cap: 2.00 USD over at most 3 pods" "$tmp/dry.txt"'
check "the payload's sources pass: the tarball's engine is HEAD's, chat_load.py shipped, the conversations HEAD's build" \
  'grep -q "the tarball.s engine is HEAD.s$" "$tmp/dry.txt" && grep -q "src/tools/dsv41/quality/chat_load.py: last changed in " "$tmp/dry.txt" &&
   grep -q "^  e07/conversations.json: .*, what chat_load.py at HEAD builds from " "$tmp/dry.txt"'

echo "--- 5. the cbase rehearsal: e07-run.sh from the payload, with the stubs; e07_report.py's sheet"
rc=0; E07_PLAN=cbase OUT="$tmp/payload" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --payload-only > "$tmp/payload.log" 2>&1 || rc=$?
check "e07.sh --payload-only builds the cbase payload" '(( rc == 0 )) && [[ -f "$tmp/payload/payload-cu130.tar.gz" ]]' ||
  tail -n 5 "$tmp/payload.log"
rm -rf -- "$tmp/payload/payload-cu130/"
bin="$tmp/bin" stubs="$tmp/stubs"
mkdir -p "$bin" "$stubs"
# q03-test.sh's stubs: the GPU, numactl, pip, the download tools, the engine, the HF headers and download, the gate
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# c01-test.sh's GPU: an RTX PRO 6000's 97887 MiB, no NUMA node for GPU0
case " $* " in
  *" topo "*) printf '\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t0-11\tN/A\t\tN/A\n' ;;
  *" --query-gpu=memory.total --format=csv,noheader,nounits "*) echo 97887 ;;
  *" --query-gpu=memory.used,memory.total --format=csv,noheader "*) echo "1024 MiB, 97887 MiB" ;;
  *) echo "nvidia-smi (c01-test.sh stub): GPU 0, 97887 MiB" ;;
esac
EOF
cat > "$bin/numactl" <<'EOF'
#!/usr/bin/env bash
while [[ "${1:-}" == --* ]]; do shift; done
exec "$@"
EOF
printf '#!/usr/bin/env bash\necho "pip (c01-test.sh stub, nothing installed): $*"\n' > "$bin/pip"
for t in aria2c apt-get; do
  printf '#!/usr/bin/env bash\necho "%s (c01-test.sh stub): nothing downloads or installs here: $*" >&2\nexit 1\n' "$t" > "$bin/$t"
done
cat > "$stubs/dsv41-engine" <<'EOF'
#!/usr/bin/env bash
# c01-test.sh's engine: --dry-run exits 0, --serve is fakes.py's engine (the tokenizer from --model)
case " $* " in *" --dry-run "*) echo "dsv41-engine (c01-test.sh stub): dry run of $*"; exit 0 ;; esac
args=()
for a in "$@"; do [[ "$a" == --serve ]] || args+=("$a"); done
exec python3 "$(dirname "$0")/src/tools/dsv41/quality/fakes.py" --serve "${args[@]}"
EOF
cat > "$stubs/hf_model.py" <<'EOF'
#!/usr/bin/env python3
"""c01-test.sh's hf_model.py: `headers MANIFEST DIR` copies the real heads (C01_HEADERS/head-N.bin) into DIR."""
import os
import shutil
import sys

if len(sys.argv) != 4 or sys.argv[1] != "headers":
    sys.exit(f"hf_model.py (c01-test.sh stub): headers MANIFEST DIR only, not {sys.argv[1:]}")
os.makedirs(sys.argv[3], exist_ok=True)
for i in range(1, 8):
    shutil.copy(os.path.join(os.environ["C01_HEADERS"], f"head-{i}.bin"), sys.argv[3])
EOF
cat > "$stubs/hf-download.sh" <<'EOF'
#!/usr/bin/env bash
# c01-test.sh's download: the sparse shards (C01_MODEL) copied into DST, the record the sheet reads, marked a stub
set -euo pipefail
mkdir -p "$2"
cp --sparse=always "$C01_MODEL"/*.gguf "$2"/
python3 -c "import json, sys; json.dump({'stub': 'c01-test.sh: the sparse shards copied, no download', 'repo': 'c01-test-stub',
  'revision': 'none', 'files': 7, 'bytes': 0, 'download_s': 0, 'gb_per_s': 0, 'check_s': 0, 'ok': True}, open(sys.argv[1], 'w'))" "$3"
EOF
cat > "$stubs/e07_gate.py" <<'EOF'
#!/usr/bin/env python3
"""c01-test.sh's gate: the payload's own e07_gate.py (e07_gate_real.py) with a 256 GB host's cgroup and MemTotal."""
import os
import sys

a = sys.argv[1:]
for key, value in (("--cgroup-bytes", str(256 * 10**9)), ("--mem-total-kib", str(256 * 10**6))):
    a[a.index(key) + 1] = value
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e07_gate_real.py"), *a])
EOF
chmod +x "$bin"/* "$stubs"/*
python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" sparse "$HEADERS" "$tmp/model" > /dev/null
d="$tmp/s-cbase"
mkdir -p "$d/pod/out" "$d/root/root"
tar -xzf "$tmp/payload/payload-cu130.tar.gz" -C "$d/pod" --exclude='libcublas*' --exclude='libggml-cuda.so*'
cp "$stubs/dsv41-engine" "$stubs/hf_model.py" "$stubs/hf-download.sh" "$d/pod/"
mv "$d/pod/src/tools/dsv41/pod/e07_gate.py" "$d/pod/src/tools/dsv41/pod/e07_gate_real.py"
cp "$stubs/e07_gate.py" "$d/pod/src/tools/dsv41/pod/"
free_port port
( cd "$d/pod" && export PATH="$bin:$PATH" E07_POD_ROOT="$d/root" E07_Q_PORT="$port" C01_MODEL="$tmp/model" \
    C01_HEADERS="$HEADERS" POD_MACHINE="c01-cbase" FAKE_ENGINE_PROMPT_MS_PER_TOKEN="$P_MS" FAKE_ENGINE_DECODE_MS_PER_TOKEN="$D_MS"
  rc=0; DEADLINE=$(( $(date +%s) + 1800 )) POD_VOLUME= timeout 2100 bash e07-run.sh > out/e07-run.out 2>&1 < /dev/null || rc=$?
  echo "$rc" > out/e07-run.exit ) &
jobs_+=($!)
wait "${jobs_[@]}"
jobs_=()
o="$d/pod/out"
awk '/^step / { sub(":$", "", $2); print $2 }' "$o/steps.log" > "$tmp/ran.txt"
echo "  cbase: $(tr '\n' ' ' < "$tmp/ran.txt")"
grep -vxE 'create,|payload|pull' "$tmp/planned.txt" > "$tmp/planned-pod.txt" || true
check "e07-run.sh exits 0, every step exit 0" \
  '[[ "$(cat "$o/e07-run.exit")" == 0 ]] && ! grep "^step " "$o/steps.log" | grep -qv ": exit 0, "'
check "steps.log has every planned step in order, then done" \
  'diff "$tmp/planned-pod.txt" "$tmp/ran.txt" > /dev/null && [[ "$(tail -n 1 "$o/steps.log")" == done ]]'
check "the engine ran with the gate's placement and prefetch 128, every one of the 30 requests in the deterministic mode" \
  'python3 -c "import json, sys; a = json.load(open(sys.argv[1]))[\"args\"]
sys.exit(0 if a[a.index(\"--prefetch-experts\") + 1] == \"128\" and \"--hot-set\" in a and \"--deterministic\" not in a else 1)" "$o/quality-server.json" &&
   (( $(grep -c "; deterministic; greedy$" "$o/quality-engine.log") == 30 ))'
check "the server was stopped" \
  'grep -q "^server [0-9]* stopped$" "$o/server-stop.log" && ! kill -0 "$(cat "$o/quality-server.pid")" 2>/dev/null'
: > "$tmp/expect-all.txt"
for n in 1 2 4 8; do
  fifo=()
  [[ $n != 4 ]] || fifo=(--fifo 4)
  python3 "$here/c01_expect.py" "$o/cbase/load-c$n.json" --prompt-ms "$P_MS" --decode-ms "$D_MS" "${fifo[@]}" \
    > "$tmp/expect-c$n.txt" 2>&1 || true
  sed "s/^/  load-c$n: /" "$tmp/expect-c$n.txt" | grep -vE "c01_expect.py: PASS" | grep -E "FAIL|first turn|ttft_p95|queue_p95|aggregate" |
    cut -c1-220 || true
  cat "$tmp/expect-c$n.txt" >> "$tmp/expect-all.txt"
done
check "every load-c<N>.json (N = 1, 2, 4, 8) within 10% of the FIFO model of the fake's delays" \
  '(( $(grep -cx "c01_expect.py: PASS" "$tmp/expect-all.txt") == 4 ))'
check "N = 4 queues FIFO: each first turn's TTFT k requests' reads and replies after the one before, the 4th user's 3 plus its own read" \
  '(( $(grep -c "^ok   N=4 first turn [1-4]: " "$tmp/expect-c4.txt") == 4 )) && grep -q "^ok   N=4 first turn 4: .*(the 3 ahead in the FIFO" "$tmp/expect-c4.txt"'
f="$tmp/report/20261014T090000Z"
mkdir -p "$f"
cp -r "$o" "$f/out"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/sheets" > "$tmp/report.log" 2>&1 || true
r="$tmp/sheets/2026-10-14-cbase/README.md"
check "the cbase sheet has the table by N, every N's request rows, the server as run and the qualification" \
  'grep -q "^## Concurrent users by N (out/cbase/load-c<N>.json)" "$r" && grep -q "^| 1 | 2 | .* | logs/cbase/load-c1.json |$" "$r" &&
   grep -q "^| 8 | 16 | .* | logs/cbase/load-c8.json |$" "$r" && grep -q "^N = 4, per request (load-c4.json):" "$r" &&
   [[ -f "$tmp/sheets/2026-10-14-cbase/logs/cbase/load-c8.json" ]] && grep -q "^## Server as run" "$r" &&
   grep -q "^## Machine qualification" "$r"' || tail -n 5 "$tmp/report.log"
if (( failed )); then
  echo "  the tail of e07-run.out and steps.log:"
  tail -n 5 "$o/e07-run.out" "$o/steps.log" 2>/dev/null | sed 's/^/    /' || true
fi

echo "c01-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
