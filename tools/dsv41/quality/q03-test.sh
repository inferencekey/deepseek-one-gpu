#!/usr/bin/env bash
# The sampler's and the deterministic mode's checks without a GPU: no model, no RunPod, no HF, no money.
#   1. tools/dsv41/engine/sampler_test.sh: the host sampler's unit test with the host compiler passes, and fails on a
#      copy of the sampler whose generator ignores the seed
#   2. the server's tests (python3 -m unittest discover serve, in the builder image as verify.sh runs them): the
#      deepseek-v4.1 profile's seed 0 and top_k as given, frequency_penalty 0.5 answered 400 naming it, the Qwen
#      profile's keys as before, strata_deterministic and "deterministic" (Dsv41Deterministic), every other test
#   3. in the cu130 builder image (tools/dsv41/build.sh's; no GPU: the toolkit's libcuda stub): the build's
#      dsv41_engine_sampler_test (the CMake target); `dsv41-engine --serve --dry-run` on tiny-b fed GEN lines on stdin:
#      `GEN 16 temperature=0.7 top_k=40 top_p=0.95 seed=1234 10,11,12` twice gives the same T lines, seed=1235 other
#      ones, the stderr lines name the sampler and seed, bad values and a penalty get ERR naming the key; greedy
#      requests (cold, live resume, prompt-end restore, a cold read in 3 chunks, temperature=0 with sampling keys)
#      answer line for line as the engine built before the sampler (Q03_HEAD_ENGINE, default $BUILD_ROOT/q03/
#      engine-2869996c/dsv41-engine: strata-2869996c-cu130-sm120's, sha256 89fe7525...), the timings aside; then
#      engine/serve_check.py --reference dry (the protocol's and the server's sampled requests);
#      the deterministic mode: det=0 and det=1 taken and named on each request's line, det=2 an ERR, --deterministic off the default off,
#      and dsv41_engine_experts_test --determinism (the mode on the CPU reference path: bit for bit with it, apart
#      without it)
#   4. pod/e07.sh --dry-run with E07_PLAN=qdet: exit 0, the machine qualification listed before the payload goes up
#      and as the plan's first step, every pod step of both model paths with its timeout (2 x its estimate), the HF
#      plan under 2.50 USD at 2.60 USD/h
#   5. the qdet rehearsal: pod/e07.sh --payload-only's qdet payload unpacked, then pod/e07-run.sh as pod-session.sh
#      starts it (DEADLINE, no volume: the model from HF), with stubs only where a pod has hardware, the model or the
#      network (q02-test.sh's: nvidia-smi, numactl, pip, aria2c and apt-get that fail, hf_model.py headers from
#      HEADERS, hf-download.sh copying gguf_layout.py's sparse shards, e07_gate.py with a 256 GB host, dsv41-engine
#      --serve as fakes.py's engine, which samples by seed). Two sessions at once: "qdet" exits 0, every planned step
#      in order with exit 0, then done, det-sampling.json with items 7 and 8 PASS (the 256-token sampled reply in it),
#      the server stopped; "qdet-nondet" (FAKE_ENGINE_NONDET=5: every repeated greedy reply changed at token 5) has
#      item 7 FAIL with the first divergent position 5 and both tokens, det-sampling exit 1, server-stop exit 0, done
#      and the server stopped. pod/e07_report.py writes each one's qdet sheet. "qdet" runs on the
#      placement-dependent fake (FAKE_ENGINE_PLACEMENT=5: off the mode, a reply follows the slab the requests before it
#      left), so its default mode passes items 7 and 8 while its mode-off row fails item 7 at 5 with the placement
#      records, and the cost row is measured, with the first token of each request with the mode on (after the
#      fake's 5.0 ms reset) and off (the reset after the request, each run off
#      waiting for the idle reset of the run in the mode before it); the session ends with done either way
# Exits 1 if any check fails.
#
#   bash tools/dsv41/quality/q03-test.sh
#
# BUILD_ROOT (default $HOME/dsv41-build) holds the build (cu130/strata), the tiny models (engine/e01), the vocabulary
# (vocab/deepseek41-vocab.gguf) and the real headers (engine/headers); Q03_HEAD_ENGINE the engine before the sampler.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
# e07.sh has no default network volume: its dry runs and rehearsals here plan one by this placeholder id
export NETWORK_VOLUME="${NETWORK_VOLUME-testvolume01}"
HEADERS="${HEADERS:-$BUILD_ROOT/engine/headers}"
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"
E01="${E01:-$BUILD_ROOT/engine/e01}"
Q03_HEAD_ENGINE="${Q03_HEAD_ENGINE:-$BUILD_ROOT/q03/engine-2869996c/dsv41-engine}"
for f in "$HEADERS/head-7.bin" "$VOCAB" "$E01/tiny-b.gguf" "$Q03_HEAD_ENGINE" "$BUILD_ROOT/cu130${BUILD_TAG:-}/strata/dsv41-engine"; do
  [[ -f "$f" ]] || { echo "q03-test.sh: no $f" >&2; exit 2; }
done
tmp="$(mktemp -d)"
jobs_=()
cleanup() {
  local f
  for f in "${jobs_[@]}"; do kill "$f" 2>/dev/null || true; done
  for f in "$tmp"/s-*/pod/out/quality-server.pid; do
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

echo "--- 1. tools/dsv41/engine/sampler_test.sh (the host compiler)"
rc=0; bash "$STRATA_DIR/tools/dsv41/engine/sampler_test.sh" > "$tmp/sampler.txt" 2>&1 || rc=$?
grep -E "^(ok|FAIL|  broken)" "$tmp/sampler.txt" | grep -E "seed|FAIL" | sed 's/^/  /' || true
check "sampler_test.sh: the unit test passes, and fails with the seeding broken on purpose" \
  '(( rc == 0 )) && grep -qx "sampler_test.sh: PASS (the test passes on the source and fails with the seed ignored)" "$tmp/sampler.txt"'

source "$STRATA_DIR/tools/dsv41/flavors.sh"
flavor_config 13.0
image="$(builder_image)"
docker image inspect "$image" > /dev/null 2>&1 || { echo "q03-test.sh: no $image (tools/dsv41/build.sh)" >&2; exit 2; }
mkdir -p "$tmp/docker"

echo "--- 2. the server's tests (python3 -m unittest discover serve in the builder image, as verify.sh runs them: the"
echo "    tokenizer tests need python3's regex module, which the image has)"
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e D="$tmp/docker" \
  -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$tmp/docker:$tmp/docker" "$image" bash -euo pipefail -c '
    mkdir -p /tmp/serve-tests && cd /tmp/serve-tests
    export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STRATA_DIR"
    python3 -m unittest discover -s "$STRATA_DIR/serve" -t "$STRATA_DIR" -p "test_*.py" > "$D/serve-tests.log" 2>&1 || true
    python3 -m unittest -v serve.test_server.Dsv41Sampling serve.test_server.SamplingKeys \
      serve.test_server.Dsv41Deterministic > "$D/serve-q03.log" 2>&1 || true' \
  > "$tmp/docker-serve.log" 2>&1 || rc=$?
tail -n 3 "$tmp/docker/serve-tests.log" | sed 's/^/  /' || true
check "serve/test_*.py pass, Dsv41Sampling and SamplingKeys among them (docker exit $rc)" \
  '(( rc == 0 )) && [[ "$(tail -n 1 "$tmp/docker/serve-tests.log")" == "OK"* ]]'
grep -oE "^test_[a-z0-9_]+ \(serve\.test_server\.[A-Za-z0-9]+\)" "$tmp/docker/serve-q03.log" | sed 's/^/  /' || true
check "the sampling server tests: seed 0 and top_k 100 forwarded, frequency_penalty 0.5 a 400 naming it, Qwen unchanged; det= and the config flag" \
  'grep -q "^Ran 10 tests in " "$tmp/docker/serve-q03.log" && [[ "$(tail -n 1 "$tmp/docker/serve-q03.log")" == OK ]]'

echo "--- 3. dsv41-engine --serve --dry-run in the cu130 builder image"
# the GEN lines: sampled (seed 1234 twice, 1235), the refusals, then greedy ones through the prefix cache's paths
cat > "$tmp/docker/sampled.txt" <<'EOF'
GEN 16 temperature=0.7 top_k=40 top_p=0.95 seed=1234 10,11,12
GEN 16 temperature=0.7 top_k=40 top_p=0.95 seed=1234 10,11,12
GEN 16 temperature=0.7 top_k=40 top_p=0.95 seed=1235 10,11,12
GEN 2 temperature=0.7 top_p=1.5 1,2
GEN 2 temperature=0.7 top_k=-3 1,2
GEN 2 temperature=0.7 seed=x 1,2
GEN 2 temperature=-1 1,2
GEN 2 penalty_freq=0.5 1,2
QUIT
EOF
cat > "$tmp/docker/greedy.txt" <<'EOF'
GEN 16 10,11,12
GEN 4 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
GEN 3 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,100,101
GEN 3 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,100,101,200
GEN 5 temperature=0 top_k=40 top_p=0.95 seed=7 10,11,12
GEN 5 temperature=0.0 10,11,99
QUIT
EOF
# the mode's key per request, a bad value, and the engine's default off
cat > "$tmp/docker/det.txt" <<'EOF'
GEN 4 10,11,12
GEN 4 det=0 10,11,12
GEN 4 det=1 temperature=0.7 seed=3 10,11,12
GEN 2 det=2 1,2
QUIT
EOF
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e BUILD_ROOT="$BUILD_ROOT" \
  -e BUILD_TAG="${BUILD_TAG:-}" \
  -e E01="$E01" -e VOCAB="$VOCAB" -e HEAD_ENGINE="$Q03_HEAD_ENGINE" -e D="$tmp/docker" \
  -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$BUILD_ROOT:$BUILD_ROOT" -v "$tmp/docker:$tmp/docker" "$image" bash -euo pipefail -c '
    mkdir -p /tmp/cudastub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1
    export LD_LIBRARY_PATH=/tmp/cudastub
    strata="$BUILD_ROOT/cu130${BUILD_TAG:-}/strata"
    "$strata/dsv41_engine_sampler_test" > "$D/cmake-sampler.txt" || true
    serve() { "$1" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 8 --dry-run < "$2" > "$3.out" 2> "$3.err"; }
    serve "$strata/dsv41-engine" "$D/sampled.txt" "$D/new-sampled"
    serve "$strata/dsv41-engine" "$D/greedy.txt" "$D/new-greedy"
    serve "$HEAD_ENGINE" "$D/greedy.txt" "$D/head-greedy"
    serve "$strata/dsv41-engine" "$D/det.txt" "$D/det"
    "$strata/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 8 --deterministic off --dry-run \
      < "$D/det.txt" > "$D/det-off.out" 2> "$D/det-off.err"
    "$strata/dsv41_engine_experts_test" --determinism "$E01/tiny-b.gguf" > "$D/determinism.txt" 2>&1 || true
    python3 "$STRATA_DIR/tools/dsv41/engine/tiny_vocab.py" "$VOCAB" /tmp/tiny-vocab.gguf > /dev/null
    PYTHONDONTWRITEBYTECODE=1 python3 "$STRATA_DIR/tools/dsv41/engine/serve_check.py" --engine "$strata/dsv41-engine" \
      --model "$E01/tiny-b.gguf" --tokenizer-gguf /tmp/tiny-vocab.gguf --vocab "$VOCAB" --reference dry \
      --out /tmp/serve-check > "$D/serve-check.log" 2>&1 || true' > "$tmp/docker.log" 2>&1 || rc=$?
d="$tmp/docker"
check "the builder image ran the engine (exit $rc)" '(( rc == 0 )) && [[ -s "$d/new-sampled.out" && -s "$d/head-greedy.out" ]]' ||
  tail -n 20 "$tmp/docker.log"
tail -n 1 "$d/cmake-sampler.txt" 2>/dev/null | sed 's/^/  /' || true
check "the build's dsv41_engine_sampler_test (CMake) passes" 'grep -qx "dsv41_engine_sampler_test: PASS" "$d/cmake-sampler.txt"'
# each request's T lines, one line per request (DONE ends it, ERR is its own)
replies() { awk '/^T / { t = t " " $2 } /^DONE / { print "T" t; t = "" } /^ERR / { print }' "$1"; }
replies "$d/new-sampled.out" > "$d/sampled-replies.txt"
sed 's/^/  /' "$d/sampled-replies.txt"
r1="$(sed -n 1p "$d/sampled-replies.txt")" r2="$(sed -n 2p "$d/sampled-replies.txt")" r3="$(sed -n 3p "$d/sampled-replies.txt")"
check "GEN ... temperature=0.7 top_k=40 top_p=0.95 seed=1234 twice: the same 16 tokens; seed=1235: others" \
  '[[ "$r1" == "$r2" && "$r1" != "$r3" && $(wc -w <<<"$r1") == 17 && "$r1" != "T 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28" ]]'
check "the stderr lines name the sampler and the seed (given)" \
  '(( $(grep -c "; sampled: temperature 0.7, top_k 40, top_p 0.95, min_p 0, seed 1234 (given)$" "$d/new-sampled.err") == 2 )) &&
   (( $(grep -c "; sampled: temperature 0.7, top_k 40, top_p 0.95, min_p 0, seed 1235 (given)$" "$d/new-sampled.err") == 1 ))'
check "bad values and a penalty: ERR naming the key (top_p, top_k, seed, temperature, penalty_freq as frequency_penalty)" \
  '[[ "$(sed -n 4,8p "$d/sampled-replies.txt")" == "$(printf "%s\n" "ERR top_p=1.5: a number in [0, 1]" \
     "ERR top_k=-3: a whole number >= 0 (0: off)" "ERR seed=x: a whole number in [0, 2^64 - 1]" \
     "ERR temperature=-1: a number >= 0" \
     "ERR penalty_freq=0.5 (the API'"'"'s frequency_penalty): this engine applies no penalties")" ]]'
# greedy: every protocol line as the engine before the sampler, PP's and DONE's milliseconds and rates aside
# (the INFO line's checkpoints=N, a key the engine before it did not have, left out too)
norm() { awk '$1 == "PP" { $4 = "-"; $5 = "-" } $1 == "DONE" { $4 = "-"; $5 = "-" } $1 == "INFO" { sub(/ checkpoints=[0-9]+/, "") } { print }' "$1"; }
diff <(norm "$d/head-greedy.out") <(norm "$d/new-greedy.out") > "$d/greedy.diff" || true
replies "$d/new-greedy.out" | sed 's/^/  greedy: /' || true
check "greedy requests (cold, live resume, prompt-end restore, 3 chunks, temperature=0 with keys): every line as $(basename "$(dirname "$Q03_HEAD_ENGINE")")'s" \
  '[[ ! -s "$d/greedy.diff" ]] && (( $(grep -c "^DONE " "$d/new-greedy.out") == 6 )) && grep -q "^RESUME " "$d/new-greedy.out"' ||
  head -n 20 "$d/greedy.diff"
check "the new engine's greedy requests say greedy on stderr" '(( $(grep -c "; greedy$" "$d/new-greedy.err") == 6 ))'
grep -E "^ok   protocol: (a sampled|the same sampled|seed 1235|two requests|seed [0-9]+ \(the first)|^ok   http: (seed 0|the engine.s log has both|frequency)" \
  "$d/serve-check.log" | sed 's/^/  /' | cut -c1-200 || true
check "engine/serve_check.py --reference dry: PASS (the sampled protocol and HTTP checks among its own)" \
  'grep -q "^serve_check.py: PASS (dry: profile, protocol, http)$" "$d/serve-check.log"' || tail -n 15 "$d/serve-check.log"
# each request's number and mode, from its stderr line
modes() {
  sed -n 's/^dsv41-engine serve: request \([0-9]*\): .*; \(deterministic\|deterministic off\); \(greedy\|sampled\).*/\1 \2/p' "$1" |
    tr '\n' ','
}
check "det=0 and det=1 taken, each request's line names its mode (default on), det=2 an ERR naming the key" \
  '[[ "$(modes "$d/det.err")" == "1 deterministic,2 deterministic off,3 deterministic," ]] &&
   grep -qx "ERR det=2: 1 (the deterministic mode) or 0 (off)" "$d/det.out" && (( $(grep -c "^DONE " "$d/det.out") == 3 ))'
check "--deterministic off makes the mode off the default, det=1 still turns it on" \
  '[[ "$(modes "$d/det-off.err")" == "1 deterministic off,2 deterministic off,3 deterministic," ]]'
grep -E "^(ok|FAIL) " "$d/determinism.txt" | sed 's/^/  /' | cut -c1-200 || true
check "dsv41_engine_experts_test --determinism: bit for bit in the mode (other timings, orders, threads, after B), apart off it" \
  'grep -qx "dsv41_engine_experts_test (d): PASS" "$d/determinism.txt" && (( $(grep -c "^ok   deterministic " "$d/determinism.txt") == 8 ))' ||
  tail -n 20 "$d/determinism.txt"

echo "--- 4. pod/e07.sh --dry-run, E07_PLAN=qdet"
rc=0; E07_PLAN=qdet OUT="$tmp/dry-out" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry.txt" 2>&1 || rc=$?
grep -E "^plan \((volume|hf)\): total|^price limit" "$tmp/dry.txt" | sed 's/^/  /' || true
check "the qdet dry run exits 0 (exit $rc)" '(( rc == 0 ))' || tail -n 10 "$tmp/dry.txt"
q_line="$(grep -n "^  qualification, when" "$tmp/dry.txt" | cut -d: -f1 || true)"
up_line="$(grep -n "payload-<tag>.tar.gz to /root/e07.tar.gz" "$tmp/dry.txt" | cut -d: -f1 || true)"
check "the machine qualification is listed before the payload goes up, and is each plan's first step" \
  '[[ -n "$q_line" && -n "$up_line" ]] && (( q_line < up_line )) &&
   [[ "$(grep -A1 "^plan with the model from HF" "$tmp/dry.txt" | tail -n 1)" == "  create, SSH, CUDA probe, qualification"* ]]'
sed -n "/^plan with the model from HF /,/^plan (hf)/p" "$tmp/dry.txt" | awk '/^  / { print $1 }' > "$tmp/planned.txt"
want_hf="create, payload env model-headers gate setup model-download dry-run engine-server det-sampling server-stop pull"
check "the HF plan's steps: $want_hf" '[[ "$(tr "\n" " " < "$tmp/planned.txt")" == "$want_hf " ]]'
check "every pod step of both model paths (9 from the volume, 8 from HF) has its timeout, 2 x its estimate" \
  '(( $(sed -n "/^plan with the model from /,/^plan (hf)/p" "$tmp/dry.txt" | grep -c "; timeout min(") == 17 )) &&
   grep -q "^  det-sampling (.*  *5  (total [0-9.]*; timeout min(10 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  engine-server (.*  *6  (total [0-9.]*; timeout min(12 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  model-download (264.5 GB at 0.924 GB/s, .*5.4  (total [0-9.]*; timeout min(10.8 min, the time left))$" "$tmp/dry.txt"'
usd="$(sed -n "s/^plan (hf): total .* (about \([0-9.]*\) USD at 2.60 USD\/h)$/\1/p" "$tmp/dry.txt")"
check "the HF plan's estimate under 2.50 USD at 2.60 USD/h (${usd:-none} USD), the session cap 2.50 over at most 3 pods" \
  '[[ -n "$usd" ]] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 2.5 else 1)" "$usd" &&
   grep -q "^price limit: 2.60 USD/h; session cap: 2.50 USD over at most 3 pods" "$tmp/dry.txt"'
check "the payload's sources pass: the tarball's engine is HEAD's (no engine source changed), det_sampling_check.py shipped" \
  'grep -q "the tarball.s engine is HEAD.s$" "$tmp/dry.txt" && grep -q "src/tools/dsv41/quality/det_sampling_check.py: last changed in " "$tmp/dry.txt"'

echo "--- 5. the qdet rehearsal: e07-run.sh from the payload, with the stubs; e07_report.py's sheets"
rc=0; E07_PLAN=qdet OUT="$tmp/payload" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --payload-only > "$tmp/payload.log" 2>&1 || rc=$?
check "e07.sh --payload-only builds the qdet payload" '(( rc == 0 )) && [[ -f "$tmp/payload/payload-cu130.tar.gz" ]]' ||
  tail -n 5 "$tmp/payload.log"
rm -rf -- "$tmp/payload/payload-cu130/"
bin="$tmp/bin" stubs="$tmp/stubs"
mkdir -p "$bin" "$stubs"
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# q03-test.sh's GPU: an RTX PRO 6000's 97887 MiB, no NUMA node for GPU0
case " $* " in
  *" topo "*) printf '\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t0-11\tN/A\t\tN/A\n' ;;
  *" --query-gpu=memory.total --format=csv,noheader,nounits "*) echo 97887 ;;
  *" --query-gpu=memory.used,memory.total --format=csv,noheader "*) echo "1024 MiB, 97887 MiB" ;;
  *) echo "nvidia-smi (q03-test.sh stub): GPU 0, 97887 MiB" ;;
esac
EOF
cat > "$bin/numactl" <<'EOF'
#!/usr/bin/env bash
while [[ "${1:-}" == --* ]]; do shift; done
exec "$@"
EOF
printf '#!/usr/bin/env bash\necho "pip (q03-test.sh stub, nothing installed): $*"\n' > "$bin/pip"
for t in aria2c apt-get; do
  printf '#!/usr/bin/env bash\necho "%s (q03-test.sh stub): nothing downloads or installs here: $*" >&2\nexit 1\n' "$t" > "$bin/$t"
done
cat > "$stubs/dsv41-engine" <<'EOF'
#!/usr/bin/env bash
# q03-test.sh's engine: --dry-run exits 0, --serve is fakes.py's engine (the tokenizer from --model)
case " $* " in *" --dry-run "*) echo "dsv41-engine (q03-test.sh stub): dry run of $*"; exit 0 ;; esac
args=()
for a in "$@"; do [[ "$a" == --serve ]] || args+=("$a"); done
exec python3 "$(dirname "$0")/src/tools/dsv41/quality/fakes.py" --serve "${args[@]}"
EOF
cat > "$stubs/hf_model.py" <<'EOF'
#!/usr/bin/env python3
"""q03-test.sh's hf_model.py: `headers MANIFEST DIR` copies the real heads (Q03_HEADERS/head-N.bin) into DIR."""
import os
import shutil
import sys

if len(sys.argv) != 4 or sys.argv[1] != "headers":
    sys.exit(f"hf_model.py (q03-test.sh stub): headers MANIFEST DIR only, not {sys.argv[1:]}")
os.makedirs(sys.argv[3], exist_ok=True)
for i in range(1, 8):
    shutil.copy(os.path.join(os.environ["Q03_HEADERS"], f"head-{i}.bin"), sys.argv[3])
EOF
cat > "$stubs/hf-download.sh" <<'EOF'
#!/usr/bin/env bash
# q03-test.sh's download: the sparse shards (Q03_MODEL) copied into DST, the record the sheet reads, marked a stub
set -euo pipefail
mkdir -p "$2"
cp --sparse=always "$Q03_MODEL"/*.gguf "$2"/
python3 -c "import json, sys; json.dump({'stub': 'q03-test.sh: the sparse shards copied, no download', 'repo': 'q03-test-stub',
  'revision': 'none', 'files': 7, 'bytes': 0, 'download_s': 0, 'gb_per_s': 0, 'check_s': 0, 'ok': True}, open(sys.argv[1], 'w'))" "$3"
EOF
cat > "$stubs/e07_gate.py" <<'EOF'
#!/usr/bin/env python3
"""q03-test.sh's gate: the payload's own e07_gate.py (e07_gate_real.py) with a 256 GB host's cgroup and MemTotal."""
import os
import sys

a = sys.argv[1:]
for key, value in (("--cgroup-bytes", str(256 * 10**9)), ("--mem-total-kib", str(256 * 10**6))):
    a[a.index(key) + 1] = value
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e07_gate_real.py"), *a])
EOF
chmod +x "$bin"/* "$stubs"/*
python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" sparse "$HEADERS" "$tmp/model" > /dev/null
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
# name|FAKE_ENGINE_NONDET (empty: deterministic)|FAKE_ENGINE_PLACEMENT (off the mode a reply follows the slab)
sessions=("qdet||5" "qdet-nondet|5|")
for s in "${sessions[@]}"; do
  IFS='|' read -r name nondet placement <<<"$s"
  d="$tmp/s-$name"
  mkdir -p "$d/pod/out" "$d/root/root"
  tar -xzf "$tmp/payload/payload-cu130.tar.gz" -C "$d/pod" --exclude='libcublas*' --exclude='libggml-cuda.so*'
  cp "$stubs/dsv41-engine" "$stubs/hf_model.py" "$stubs/hf-download.sh" "$d/pod/"
  mv "$d/pod/src/tools/dsv41/pod/e07_gate.py" "$d/pod/src/tools/dsv41/pod/e07_gate_real.py"
  cp "$stubs/e07_gate.py" "$d/pod/src/tools/dsv41/pod/"
  free_port port
  ( cd "$d/pod" && export PATH="$bin:$PATH" E07_POD_ROOT="$d/root" E07_Q_PORT="$port" Q03_MODEL="$tmp/model" \
      Q03_HEADERS="$HEADERS" POD_MACHINE="q03-$name"
    [[ -z "$nondet" ]] || export FAKE_ENGINE_NONDET="$nondet"
    [[ -z "$placement" ]] || export FAKE_ENGINE_PLACEMENT="$placement"
    rc=0; DEADLINE=$(( $(date +%s) + 1200 )) POD_VOLUME= timeout 1500 bash e07-run.sh > out/e07-run.out 2>&1 < /dev/null || rc=$?
    echo "$rc" > out/e07-run.exit ) &
  jobs_+=($!)
done
wait "${jobs_[@]}"
jobs_=()
for s in "${sessions[@]}"; do
  IFS='|' read -r name nondet placement <<<"$s"
  o="$tmp/s-$name/pod/out" before=$failed
  awk '/^step / { sub(":$", "", $2); print $2 }' "$o/steps.log" > "$tmp/ran-$name.txt"
  echo "  $name: $(tr '\n' ' ' < "$tmp/ran-$name.txt")"
  grep -E "^(item|  |det_sampling)" "$o/det-sampling.log" 2>/dev/null | sed "s/^/  $name: /" | cut -c1-220 || true
  # the planned pod steps of the HF path, less the session's own (create, payload, pull)
  grep -vxE 'create,|payload|pull' "$tmp/planned.txt" > "$tmp/planned-pod.txt" || true
  check "$name: steps.log has every planned step in order, then done" \
    'diff "$tmp/planned-pod.txt" "$tmp/ran-$name.txt" > /dev/null && [[ "$(tail -n 1 "$o/steps.log")" == done ]]'
  check "$name: the engine ran with the gate's placement, prefetch 128 and --log-ids" \
    'python3 -c "import json, sys; a = json.load(open(sys.argv[1]))[\"args\"]
sys.exit(0 if \"--log-ids\" in a and a[a.index(\"--prefetch-experts\") + 1] == \"128\" and \"--hot-set\" in a else 1)" "$o/quality-server.json"'
  check "$name: the server was stopped" \
    'grep -q "^server [0-9]* stopped$" "$o/server-stop.log" && ! kill -0 "$(cat "$o/quality-server.pid")" 2>/dev/null'
  j="$o/det-sampling.json"
  if [[ -z "$nondet" ]]; then
    check "$name: e07-run.sh exits 0, every step exit 0" \
      '[[ "$(cat "$o/e07-run.exit")" == 0 ]] && ! grep "^step " "$o/steps.log" | grep -qv ": exit 0, "'
    check "$name: items 7 and 8 PASS (the same greedy ids, seeds, speed within 5%, the 400), the 256-token reply in the JSON" \
      'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); i7, i8 = j[\"items\"][\"7\"], j[\"items\"][\"8\"]
sys.exit(0 if j[\"ok\"] and i7[\"pass\"] and not i7[\"divergence\"] and i8[\"seeds\"][\"pass\"] and i8[\"speed\"][\"pass\"] and
         i8[\"penalty\"][\"pass\"] and i8[\"text_256\"][\"tokens\"] == 256 and len(j[\"requests\"]) == 27 else 1)" "$j"'
    check "$name: the mode-off row on the placement-dependent fake: item 7 FAIL at $placement for both repeats with their placement, seed 1234 apart; the cost measured" \
      'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); o = j[\"mode_off\"]; d = o[\"7\"][\"divergence\"]; k = j[\"cost\"]
sys.exit(0 if not o[\"7\"][\"pass\"] and [x[\"position\"] for x in d] == [int(sys.argv[2])] * 2 and
         all(x[\"placement\"][\"hits\"] > 0 and x[\"first_placement\"][\"swaps\"] >= 0 for x in d) and
         not o[\"8_seeds\"][\"same_seed_identical\"] and k[\"measured\"] and len(k[\"on_ms_per_token\"]) == 3 and
         len(k[\"off_ms_per_token\"]) == 3 and k[\"ratio\"] is not None else 1)" "$j" "$placement" &&
       grep -q "^mode off (strata_deterministic false; recorded, not in the verdict):" "$o/det-sampling.log" &&
       grep -q "^cost: median .* ms per token with the mode on against .* off, ratio " "$o/det-sampling.log"'
    # the mode resets after its request, so in the interleaved cost runs each run in the mode resets at its start
    # (the one before it was off) and each run off waits for the idle reset of the one before it (in the mode)
    check "$name: the first token with the idle reset: every answered request has its first token and reset; the cost row 6.0 ms to the first token in the mode (5.0 of reset before it) and 6.0 off (5.0 waiting for the last request's idle reset)" \
      'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); f = j[\"cost\"][\"first_token\"]
ok = all(r[\"first_token_ms\"] == 1.0 and r[\"reset_ms\"] is not None for r in j[\"requests\"] if r[\"status\"] == 200)
sys.exit(0 if ok and f[\"on\"][\"reset_ms\"] == [5.0] * 3 and f[\"off\"][\"reset_ms\"] == [5.0] * 3 and
         f[\"on\"][\"reset_before_median\"] == 5.0 and f[\"on\"][\"idle_wait_median\"] == 0.0 and
         f[\"off\"][\"reset_before_median\"] == 0.0 and f[\"off\"][\"idle_wait_median\"] == 5.0 and
         f[\"on\"][\"with_reset_median\"] == 6.0 and f[\"off\"][\"with_reset_median\"] == 6.0 else 1)" "$j" &&
       grep -q "^cost: first token: median 6.0 ms with the mode on (5.0 of reset: 5.0 before it and 0.0 waiting for the last request.s idle reset; 1.0 from the request.s start) against 6.0 off (5.0 of reset: 0.0 before it and 5.0 waiting for the last request.s idle reset; 1.0 from the request.s start)$" "$o/det-sampling.log"'
  else
    check "$name: e07-run.sh exits 1; only det-sampling failed (exit 1), server-stop ran after it" \
      '[[ "$(cat "$o/e07-run.exit")" == 1 ]] && grep -q "^step det-sampling: exit 1, " "$o/steps.log" &&
       ! grep "^step " "$o/steps.log" | grep -v "^step det-sampling: " | grep -qv ": exit 0, " &&
       grep -q "^step server-stop: exit 0, " "$o/steps.log"'
    check "$name: item 7 FAIL with the first divergent position $nondet and both tokens, for both repeats; item 8 PASS" \
      'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); i7 = j[\"items\"][\"7\"]; d = i7[\"divergence\"]
sys.exit(0 if not j[\"ok\"] and not i7[\"pass\"] and [x[\"position\"] for x in d] == [int(sys.argv[2])] * 2 and
         all(x[\"first\"] != x[\"other\"] and x[\"other\"] is not None for x in d) and j[\"items\"][\"8\"][\"pass\"] else 1)" "$j" "$nondet" &&
       grep -q "^item 7: FAIL: .* diverge: A greedy, again at position $nondet (token " "$o/det-sampling.log"'
  fi
  # the sheet, as e07.sh writes it after the pull
  f="$tmp/report-$name/20261013T150000Z"
  mkdir -p "$f"
  cp -r "$o" "$f/out"
  python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/sheets-$name" > "$tmp/report-$name.log" 2>&1 || true
  r="$tmp/sheets-$name/2026-10-13-qdet/README.md"
  if [[ -z "$nondet" ]]; then
    check "$name: the qdet sheet has items 7 and 8 PASS, the requests, the 256-token reply and the server as run" \
      'grep -q "^## Items 7 and 8 (out/det-sampling.json)" "$r" && grep -q "^- item 7: PASS" "$r" && grep -q "^- item 8: PASS" "$r" &&
       grep -q "^| 5 | A at 0.7, seed 1234 | 200 | 256 | length |" "$r" && grep -q "^Item 8, the reply at temperature 0.7, top_p 0.95, top_k 40, seed 1234 " "$r" &&
       grep -q "^## Server as run" "$r" && grep -q "^## Machine qualification" "$r"' || tail -n 5 "$tmp/report-$name.log"
  else
    check "$name: the qdet sheet has item 7 FAIL and its divergence record" \
      'grep -q "^- item 7: FAIL" "$r" && (( $(grep -c "^Item 7 divergence: .* at position $nondet: token " "$r") == 2 ))' ||
      tail -n 5 "$tmp/report-$name.log"
  fi
  if (( failed > before )); then
    echo "  $name: the tail of e07-run.out, steps.log, det-sampling.log:"
    tail -n 5 "$o/e07-run.out" "$o/steps.log" "$o/det-sampling.log" 2>/dev/null | sed 's/^/    /' || true
  fi
done

echo "q03-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
