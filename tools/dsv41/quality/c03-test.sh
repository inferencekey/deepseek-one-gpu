#!/usr/bin/env bash
# The batched decode's checks without a GPU: no model, no RunPod, no HF, no money.
#   1. in the cu130 builder image (tools/dsv41/build.sh's; no GPU): dsv41_engine_batch_test on tiny-b (the CPU
#      reference: 4 requests of different lengths decoding in a batch, each bit for bit its alone run; slots ending
#      mid-batch; a prefill between steps; B = 1; BSTOP; the slot's ring and the admission's copies, their negative
#      controls), dsv41-engine --serve --dry-run --batch 4 on tiny-b: the batch protocol (BGEN, BADM 1, every slot's BT
#      ids its own predicted ones, BDONE with rep=1 beside others in the deterministic mode and rep=0 with det=0,
#      a BSTOP ending its slot "cancel", a request past --batch-ctx answered alone and BADM 0, a BGEN to a busy
#      slot refused, the INFO line's batch_slots), and on the real layout with --batch 8: its "batch slots:" line's KV
#      per slot the gate's; the dev pod's forced end-to-end runs (ctx 2048, no --batch) as dry runs
#      and the serve dry run at ctx 2048 without --batch accepted, with --batch 4 and the default --batch-ctx refused
#   2. the server's tests (python3 -m unittest discover serve in the builder image, as verify.sh runs them): the
#      batched reply's "strata_reproducible" (true in the deterministic mode, false off), batch_ctx's config,
#      --parallel over the config's; "strata_reproducible": "required" with 3 slots decoding waits and
#      decodes alone, its reply true, a normal request still batched, the refused values
#   3. quality/chat_load.py against serve/server.py on fakes.py's engine with --batch 8 (0.3 ms per prompt token, 5 ms
#      a step of one row and 1.5 ms per further row): 1, 4 and 8 users, every request answered, each run's per-user and
#      aggregate tok/s within 10% of quality/c03_expect.py's model of the fake's batch, the server's done lines marking
#      the batched replies reproducible (the deterministic mode); the same 4 users with batching off (the
#      server's --parallel 1) within 10% of c01_expect.py's FIFO model; chat_load.py steady with 4
#      users: every repeat's window with every user decoding within 10% of c03_expect.py
#      --steady (4 rows a step)
#   4. pod/e07_gate.py with --batch: the live slots, their KV and the slab's loss on an RTX PRO 6000 host; a placement
#      the batch's slots leave without a hot set refused naming them
#   5. pod/e07.sh --dry-run with E07_PLAN=cbatch: exit 0, the RTX PRO 6000 only, the qualification first, every step
#      with its timeout, the plan under 3 USD at 2.60 USD/h, both gates, the payload's sources; pod/devgpu.sh --dry-run
#      with DEVGPU_PLAN=c03: the dev pod's batch steps and their cost under 1.50 USD; then the
#      cbatch rehearsal: e07-run.sh from the payload (its steady window 4 s: the fake decodes ~105 tok/s a user) with
#      c02-test.sh's stubs and fakes.py as the engine: every planned step exit 0, then done, both servers as planned,
#      det-sampling PASS, e07_report.py's cbatch sheet (item 11's table, its one verdict line from the steady state
#      against batching off with the end-to-end figures beside it, the steady table, the batch's lines, both gates)
# Exits 1 if any check fails. c02-test.sh, c01-test.sh, q03-test.sh, q02-test.sh, analyze_test.py and verify.sh are run
# on their own.
#
#   BUILD_TAG=-c03 bash tools/dsv41/quality/c03-test.sh
#
# BUILD_ROOT (default $HOME/dsv41-build) holds the build (cu130${BUILD_TAG}/strata), the tiny models (engine/e01), the
# vocabulary, the real headers (engine/headers), the conversations and the tarballs the dry runs and the payload read.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
# e07.sh has no default network volume: its dry runs and rehearsals here plan one by this placeholder id
export NETWORK_VOLUME="${NETWORK_VOLUME-testvolume01}"
HEADERS="${HEADERS:-$BUILD_ROOT/engine/headers}"
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"
E01="${E01:-$BUILD_ROOT/engine/e01}"
CONVERSATIONS="${CONVERSATIONS:-$BUILD_ROOT/quality/conversations-16k.json}"
COUNTS="${COUNTS:-$BUILD_ROOT/counts.json}"
STRATA_BUILD="$BUILD_ROOT/cu130${BUILD_TAG:-}/strata"
for f in "$HEADERS/head-7.bin" "$VOCAB" "$E01/tiny-b.gguf" "$STRATA_BUILD/dsv41_engine_batch_test" "$CONVERSATIONS" "$COUNTS"; do
  [[ -f "$f" ]] || { echo "c03-test.sh: no $f" >&2; exit 2; }
done
# the fake engine's delays: c01-test.sh's (a 16K prompt read in ~4.9 s, 5 ms a token alone), 1.5 ms per further row
P_MS=0.3 D_MS=5 R_MS=1.5
tmp="$(mktemp -d)"
jobs_=()
cleanup() {
  local f
  for f in "${jobs_[@]}"; do kill "$f" 2>/dev/null || true; done
  for f in "$tmp"/server*.pid "$tmp"/s-*/pod/out/quality-server.pid; do
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
python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" sparse "$HEADERS" "$tmp/model" > /dev/null
SHARD1="$tmp/model/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf"

source "$STRATA_DIR/tools/dsv41/flavors.sh"
flavor_config 13.0
image="$(builder_image)"
docker image inspect "$image" > /dev/null 2>&1 || { echo "c03-test.sh: no $image (tools/dsv41/build.sh)" >&2; exit 2; }
d="$tmp/docker"
mkdir -p "$d"

echo "--- 1. the engine's CPU reference and the batch protocol of the dry run, in the cu130 builder image"
# BGEN 0 and 1 decode together; BSTOP 1 ends slot 1 early; a BGEN to slot 0 while it decodes is refused (the loop reads
# every waiting line before a step, and slot 0 needs 5 steps); slot 2's request, 120 prompt tokens and 20 to generate,
# is past --batch-ctx 128 (alone, BADM 0); slot 3 joins. No QUIT: stdin's end drains the batch. Slots 1 and 2
# with det=0 (the mode off: rep=0), slots 0 and 3 in the deterministic mode (rep=1 beside other slots, in the GPU's
# numerics: --cpu-numerics gpu; the fork's numerics are the default)
python3 -c "print('BGEN 0 6 10,11,12,13'); print('BGEN 1 40 det=0 20,21,22'); print('BSTOP 1'); print('BGEN 0 4 30,31')
print('BGEN 2 20 det=0 ' + ','.join(str(i) for i in range(40, 160))); print('BGEN 3 5 200,201')" > "$d/batch.txt"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_gate.py" --vram-mib 97887 --prefetch-experts 128 --cgroup-bytes 300000000000 \
  --mem-total-kib 300000000 --counts "$COUNTS" --layout "$STRATA_DIR/tools/dsv41/oracle/layout-dsv41-q2k.json" \
  --model-headers "$SHARD1" --out "$d/gate-batch" --batch 8 --checkpoints 8 > "$d/gate-batch.log" 2>&1 || true
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e STRATA_BUILD="$STRATA_BUILD" \
  -e E01="$E01" -e SHARD1="$SHARD1" -e D="$d" -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$BUILD_ROOT:$BUILD_ROOT:ro" \
  -v "$tmp:$tmp" "$image" bash -euo pipefail -c '
    mkdir -p /tmp/cudastub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1
    export LD_LIBRARY_PATH=/tmp/cudastub
    "$STRATA_BUILD/dsv41_engine_batch_test" "$E01/tiny-b.gguf" > "$D/batch-test.txt" 2>&1 || true
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 8 --batch 4 --batch-ctx 128 \
      --cpu-numerics gpu --dry-run < "$D/batch.txt" > "$D/batch.out" 2> "$D/batch.err" || true
    "$STRATA_BUILD/dsv41-engine" --serve --model "$SHARD1" --ctx 262144 --chunk 16384 --batch 8 --dry-run < /dev/null \
      > "$D/real.out" 2> "$D/real.err" || true
    # the dev pod'"'"'s forced end-to-end runs (pod/devgpu-run.sh: ctx 2048, no --batch, so the
    # default --batch-ctx 32768) as dry runs, which check the batch config as Forward::init does; then the serve dry
    # run at ctx 2048 without --batch, and with --batch 4 and the default --batch-ctx (past --ctx: refused)
    for c in a b; do
      case "$c" in a) steps=5 ;; b) steps=9 ;; esac
      python3 "$STRATA_DIR/tools/dsv41/engine/hot_sets.py" "$E01/tiny-$c.gguf" "$D/hot-$c" > /dev/null
      python3 -c "import json, sys; print(\" \".join(map(str, json.load(open(sys.argv[1]))[\"prompt_token_ids\"])))" \
        "$E01/dump-$c-short/dump-manifest.json" > "$D/prompt-$c-short.txt"
      rc_=0
      "$STRATA_BUILD/dsv41-engine" --model "$E01/tiny-$c.gguf" --tokens-file "$D/prompt-$c-short.txt" --n-predict "$steps" \
        --ctx 2048 --readers reuse --hot-set "$D/hot-$c/split.json" --vram-gb 1 --ram-gb 1 \
        --force-selection "$E01/dump-$c-short" --dry-run > "$D/forced-$c.out" 2>&1 || rc_=$?
      echo "$rc_" > "$D/forced-$c.rc"
    done
    rc_=0
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 2048 --dry-run < /dev/null \
      > "$D/serve-2048.out" 2> "$D/serve-2048.err" || rc_=$?
    echo "$rc_" > "$D/serve-2048.rc"
    rc_=0
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 2048 --batch 4 --dry-run < /dev/null \
      > "$D/serve-2048-batch.out" 2> "$D/serve-2048-batch.err" || rc_=$?
    echo "$rc_" > "$D/serve-2048-batch.rc"' > "$tmp/docker.log" 2>&1 || rc=$?
check "the builder image ran the tests (exit $rc)" '(( rc == 0 )) && [[ -s "$d/batch-test.txt" && -s "$d/batch.out" ]]' ||
  tail -n 20 "$tmp/docker.log"
grep -E "^(ok|FAIL) " "$d/batch-test.txt" | sed 's/^/  /' | cut -c1-200 || true
check "dsv41_engine_batch_test on tiny-b: PASS (batch = alone bit for bit, lengths, a slot ending, a prefill between steps, B = 1)" \
  '[[ "$(tail -n 1 "$d/batch-test.txt")" == "dsv41_engine_batch_test: PASS" ]] && ! grep -q "^FAIL " "$d/batch-test.txt" &&
   (( $(grep -c "^ok   " "$d/batch-test.txt") >= 15 ))'
python3 - "$d/batch.out" > "$d/batch-check.txt" 2>&1 <<'EOF' || true
import re, sys
lines = open(sys.argv[1]).read().splitlines()
bt, done, badm, errs = {}, {}, [], []
for x in lines:
    f = x.split()
    if f[0] == "BT":
        bt.setdefault(int(f[1]), []).append(int(f[2]))
    elif f[0] == "BDONE":
        done.setdefault(int(f[1]), []).append(f[2:])
    elif f[0] == "BADM":
        badm.append(f[1:])
    elif f[0] == "ERR":
        errs.append(x)
info = next((x for x in lines if x.startswith("INFO ")), "")
ok = "batch_slots=4 batch_ctx=128 slot_cache=0" in info
# the dry run's token is the last one read plus one: slot 0 picks 14 at its admission (a T line), then 15 .. 19 in
# the steps; slot 1 23, then 24 .. until its BSTOP; slot 3 202, then 203 .. 206
ok = ok and bt.get(0) == [15, 16, 17, 18, 19] and done.get(0) == [["6", "length", done[0][0][2], "rep=1"]]
s1 = bt.get(1, [])
ok = ok and done.get(1, [[0, ""]])[0][1] == "cancel" and int(done[1][0][0]) < 40 and s1 == list(range(24, 24 + len(s1)))
ok = ok and done.get(3, [[0, ""]])[0][:2] == ["5", "length"] and bt.get(3) == [203, 204, 205, 206]
# the deterministic mode's slots rep=1 beside other slots, the mode off's (det=0) rep=0
ok = ok and done[3][0][-1] == "rep=1" and done[1][0][-1] == "rep=0"
# the request past --batch-ctx says it ran alone, holding the 2 decoding slots, and why
ok = ok and any(b[:4] == ["2", "0", "rep=0", "alone=2"] and b[4].startswith("why=") and "a batch slot holds 128" in " ".join(b)
                for b in badm) and 2 not in bt
ok = ok and sum(1 for b in badm if b[1] == "1") == 3 and len(errs) == 1 and "no such free slot" in errs[0]
reps = [d[-1] for v in done.values() for d in v]
ok = ok and all(r in ("rep=0", "rep=1") for r in reps) and "rep=0" in reps
print(f"batch protocol: INFO {info[5:]!r}; BT per slot {dict((k, len(v)) for k, v in bt.items())}; BDONE {done}; BADM {badm};"
      f" ERR {errs}: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
sed 's/^/  /' "$d/batch-check.txt" | cut -c1-240
check "the dry run's batch protocol: BGEN, BADM 1, each slot's BT ids its own, BDONE rep=1 (deterministic) and rep=0 (det=0), BSTOP cancel, past --batch-ctx alone and BADM 0, a busy slot refused" \
  'grep -q ": PASS$" "$d/batch-check.txt"'
check "the stderr lines: each slot's end (det=0: not reproducible; batch determinism, the deterministic mode: reproducible), the batch's totals, a read beside decoding slots" \
  'grep -qE "^dsv41-engine serve: request [0-9]+: batch slot 1: [0-9]+ generated \(cancel\) in .* not reproducible \(deterministic off\)$" "$d/batch.err" &&
   grep -qE "^dsv41-engine serve: request [0-9]+: batch slot 0: 6 generated \(length\) in .*, [0-9]+ of them beside other slots .*; reproducible \(the deterministic mode" "$d/batch.err" &&
   grep -qE "^dsv41-engine serve: batch: [0-9]+ steps of [0-9.]+ rows on average" "$d/batch.err" &&
   grep -qE "^dsv41-engine serve: request [0-9]+: read beside [0-9]+ decoding batch slots" "$d/batch.err"'
slot="$(python3 -c "import json, sys; print(round(json.load(open(sys.argv[1]))['batch_slot_bytes'] / 2**20, 1))" "$d/gate-batch/gate.json" 2>/dev/null || echo none)"
grep -E "^batch slots:" "$d/real.err" | sed 's/^/  /' | cut -c1-200 || true
check "the real layout's dry run with --batch 8: a slot's device KV the gate's ($slot MiB)" \
  'grep -q "^batch slots: 8 live sequences of $slot MiB device KV each (a slot of 32768 positions" "$d/real.err" &&
   grep -q "^INFO .* batch_slots=8 batch_ctx=32768 slot_cache=0 interleave_chunk=16384 engine=dsv41-engine$" "$d/real.out"'

for c in a b; do
  tail -n 1 "$d/forced-$c.out" | sed "s/^/  tiny-$c forced: /" | cut -c1-200 || true
done
check "the dev pod's forced end-to-end runs (ctx 2048, no --batch) as dry runs on tiny-a and tiny-b: exit 0, the default --batch-ctx not read" \
  '(for c in a b; do
     [[ "$(cat "$d/forced-$c.rc")" == 0 ]] && grep -q "^dsv41-engine: dry run done, no CUDA call$" "$d/forced-$c.out" &&
       grep -q "selections and decode tokens forced from" "$d/forced-$c.out" && ! grep -q "batch-ctx" "$d/forced-$c.out" || exit 1
   done)'
check "the serve dry run at ctx 2048: no --batch taken (exit 0, no batch_slots), --batch 4 with the default --batch-ctx 32768 refused naming it" \
  '[[ "$(cat "$d/serve-2048.rc")" == 0 ]] && grep -q "^INFO .* checkpoints=4 engine=dsv41-engine$" "$d/serve-2048.out" &&
   [[ "$(cat "$d/serve-2048-batch.rc")" != 0 ]] && grep -q -- "--batch-ctx 32768 is past --ctx 2048" "$d/serve-2048-batch.err"'

echo "--- 2. the server's tests (python3 -m unittest discover serve in the builder image, as verify.sh runs them)"
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e D="$d" \
  -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$d:$d" "$image" bash -euo pipefail -c '
    mkdir -p /tmp/serve-tests && cd /tmp/serve-tests
    export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STRATA_DIR"
    python3 -m unittest discover -s "$STRATA_DIR/serve" -t "$STRATA_DIR" -p "test_*.py" > "$D/serve-tests.log" 2>&1 || true
    python3 -m unittest -v serve.test_server.Dsv41Batch \
      serve.test_parallel.ParallelService.test_a_batched_reply_says_whether_it_is_reproducible \
      serve.test_parallel.ParallelService.test_a_required_reproducible_reply_waits_and_decodes_alone \
      serve.test_parallel.ParallelService.test_a_normal_request_still_batches \
      serve.test_parallel.ParallelService.test_required_reproducible_refused_values > "$D/serve-c03.log" 2>&1 || true' \
  > "$tmp/docker-serve.log" 2>&1 || rc=$?
grep -E "^(FAIL|ERROR):" "$d/serve-tests.log" | sed 's/^/  /' || true
tail -n 3 "$d/serve-tests.log" | sed 's/^/  /' || true
check "serve/test_*.py pass (docker exit $rc)" '(( rc == 0 )) && [[ "$(tail -n 1 "$d/serve-tests.log")" == "OK"* ]]'
check "the batching server tests: the batched reply marked, batch_ctx's config, --parallel over the config, \"strata_reproducible\": \"required\" alone after 3 decoding slots (true), a normal request batched, the refused values" \
  'grep -q "^Ran 6 tests in " "$d/serve-c03.log" && [[ "$(tail -n 1 "$d/serve-c03.log")" == OK ]]'

echo "--- 3. chat_load.py against serve/server.py on the fake engine's batch slots (P ${P_MS}, D ${D_MS}, ${R_MS} ms a further row)"
# (in the GPU's numerics, so the done lines' reproducible marks below hold; the fork's are the default)
serve_fake() {   # serve_fake NAME PARALLEL: the server on fakes.py's engine with --batch 8 and 8 checkpoints
  local dir="$tmp/$1" port
  mkdir -p "$dir"
  cat > "$dir/engine" <<EOF
#!/usr/bin/env bash
exec python3 "$here/fakes.py" --serve "\$@"
EOF
  chmod +x "$dir/engine"
  python3 -c "import json, sys; d = sys.argv[1]
print(json.dumps({'exe': d + '/engine', 'cwd': d, 'args': ['--model', sys.argv[2], '--ctx', '262144', '--checkpoints', '8',
  '--batch-ctx', '32768', '--cpu-numerics', 'gpu'], 'model_profile': 'deepseek-v4.1', 'tokenizer_gguf': sys.argv[2], 'log': d + '/engine.log',
  'repeat_stop_tokens': 0, 'parallel': 8}))" "$dir" "$VOCAB" > "$dir/config.json"
  free_port port
  ( cd "$STRATA_DIR" && FAKE_ENGINE_PROMPT_MS_PER_TOKEN=$P_MS FAKE_ENGINE_DECODE_MS_PER_TOKEN=$D_MS \
      FAKE_ENGINE_BATCH_ROW_MS=$R_MS exec setsid python3 -m serve.server --engine strata --config "$dir/config.json" \
      --port "$port" --parallel "$2" ) > "$dir/server.log" 2>&1 < /dev/null &
  echo $! > "$tmp/server-$1.pid"
  for _ in $(seq 120); do grep -q "ready: http" "$dir/server.log" && break; sleep 1; done
  echo "$port" > "$dir/port"
}
serve_fake on 8
rc=0
timeout 900 python3 "$here/chat_load.py" run --url "http://127.0.0.1:$(cat "$tmp/on/port")" --conversations "$CONVERSATIONS" \
  --users 1,4,8 --out "$tmp/on/load.json" > "$tmp/on/load.txt" 2>&1 || rc=$?
grep "^| " "$tmp/on/load.txt" | sed 's/^/  /' | cut -c1-200
check "batching on: 1, 4 and 8 users, every request answered (exit $rc), 256 tokens each" \
  '(( rc == 0 )) && python3 -c "import json, sys; j = json.load(open(sys.argv[1])); rows = [r for x in j[\"runs\"] for r in x[\"rows\"]]
sys.exit(0 if j[\"ok\"] and len(rows) == 26 and all(r[\"completion_tokens\"] == 256 for r in rows) else 1)" "$tmp/on/load.json"'
python3 "$here/c03_expect.py" "$tmp/on/load.json" --prompt-ms "$P_MS" --decode-ms "$D_MS" --row-ms "$R_MS" --slots 8 \
  > "$tmp/on/expect.txt" 2>&1 || true
sed 's/^/  /' "$tmp/on/expect.txt" | cut -c1-200
check "per-user and aggregate tok/s within 10% of the fake's batch model (c03_expect.py)" 'grep -qx "c03_expect.py: PASS" "$tmp/on/expect.txt"'
check "the batched replies marked reproducible on the server's done lines (batch determinism: the deterministic mode), the batch's lines in the engine's log" \
  '(( $(grep -c "^\[strata\] done: 256 tokens in .*, batched: reproducible$" "$tmp/on/server.log") >= 20 )) &&
   ! grep -q "batched: not reproducible" "$tmp/on/server.log" &&
   (( $(grep -cE "^dsv41-engine serve: request [0-9]+: batch slot [0-9]+: 256 generated \(length\)" "$tmp/on/engine.log") >= 20 ))'
rc=0
timeout 900 python3 "$here/chat_load.py" steady --url "http://127.0.0.1:$(cat "$tmp/on/port")" --conversations "$CONVERSATIONS" \
  --users 4 --window-s 4 --repeats 2 --max-tokens 1024 --out "$tmp/on/steady.json" > "$tmp/on/steady.txt" 2>&1 || rc=$?
grep -E "^steady N=4( repeat [0-9]+)?: " "$tmp/on/steady.txt" | sed 's/^/  /' | cut -c1-200
python3 "$here/c03_expect.py" "$tmp/on/steady.json" --steady --decode-ms "$D_MS" --row-ms "$R_MS" > "$tmp/on/steady-expect.txt" 2>&1 || true
sed 's/^/  /' "$tmp/on/steady-expect.txt" | cut -c1-200
check "chat_load.py steady: 4 users, 2 repeats of a 4 s window with every user decoding (exit $rc), each repeat within 10% of 4 rows a step (c03_expect.py --steady)" \
  '(( rc == 0 )) && grep -qx "c03_expect.py: PASS" "$tmp/on/steady-expect.txt" &&
   python3 -c "import json, sys; j = json.load(open(sys.argv[1])); sys.exit(0 if j[\"ok\"] and j[\"summary\"][\"measured\"] == 2 and all(x[\"chunks\"] == x[\"completion_tokens\"] == 1024 for r in j[\"repeats\"] for x in r[\"rows\"]) else 1)" "$tmp/on/steady.json"'
kill -TERM -- "-$(cat "$tmp/server-on.pid")" 2>/dev/null || true
serve_fake off 1
rc=0
timeout 900 python3 "$here/chat_load.py" run --url "http://127.0.0.1:$(cat "$tmp/off/port")" --conversations "$CONVERSATIONS" \
  --users 4 --out "$tmp/off/load.json" > "$tmp/off/load.txt" 2>&1 || rc=$?
grep "^| " "$tmp/off/load.txt" | sed 's/^/  /' | cut -c1-200
python3 "$here/c01_expect.py" "$tmp/off/load.json" --prompt-ms "$P_MS" --decode-ms "$D_MS" > "$tmp/off/expect.txt" 2>&1 || true
sed 's/^/  /' "$tmp/off/expect.txt" | cut -c1-200
check "batching off (--parallel 1): the engine started without --batch, 4 users within 10% of the FIFO model (c01_expect.py)" \
  '(( rc == 0 )) && grep -qx "c01_expect.py: PASS" "$tmp/off/expect.txt" && ! grep -q "batch_slots" "$tmp/off/engine.log" &&
   ! grep -q -- "--batch " "$tmp/off/engine.log" && ! grep -q ", batched: " "$tmp/off/server.log"'
kill -TERM -- "-$(cat "$tmp/server-off.pid")" 2>/dev/null || true

echo "--- 4. pod/e07_gate.py with the batch's slots"
for case in "rtx|97887|8|32768|0" "small|25300|0|32768|0" "small-batch|25300|8|262144|1"; do
  IFS='|' read -r name mib b ctx want <<<"$case"
  rc=0
  python3 "$STRATA_DIR/tools/dsv41/pod/e07_gate.py" --vram-mib "$mib" --prefetch-experts 128 --cgroup-bytes 300000000000 \
    --mem-total-kib 300000000 --counts "$COUNTS" --layout "$STRATA_DIR/tools/dsv41/oracle/layout-dsv41-q2k.json" \
    --model-headers "$SHARD1" --out "$tmp/gate-$name" --batch "$b" --batch-ctx "$ctx" --checkpoints 8 \
    > "$tmp/gate-$name.log" 2>&1 || rc=$?
  echo "$rc" > "$tmp/gate-$name.rc"
  grep -E "^e07_gate.py: (batch|refused|VRAM)" "$tmp/gate-$name.log" | sed 's/^/  /' | cut -c1-220 || true
done
check "the RTX PRO 6000 with 8 slots of 32768: the live slots, their KV (105.1 MiB each) and the slab's loss reported, it fits" \
  '[[ "$(cat "$tmp/gate-rtx.rc")" == 0 ]] &&
   grep -q "^e07_gate.py: batch: 8 live slots of 32768 positions, 105.1 MiB device KV each, 840.6 MiB in all + the rows. logits 3.9 MiB = 0.82 GiB, taken from the slab (--batch 8 --batch-ctx 32768): it loses about [0-9]* experts" "$tmp/gate-rtx.log" &&
   python3 -c "import json, sys; g = json.load(open(sys.argv[1])); sys.exit(0 if g[\"batch\"] == 8 and g[\"batch_slot_bytes\"] == 110174208 and g[\"batch_slab_experts_lost\"] > 0 else 1)" "$tmp/gate-rtx/gate.json"'
check "a 25300 MiB GPU: a slab without the batch, refused with 8 slots of 262144 (their 6.3 GiB of KV named)" \
  '[[ "$(cat "$tmp/gate-small.rc")" == 0 && "$(cat "$tmp/gate-small-batch.rc")" == 1 ]] &&
   grep -q "^e07_gate.py: refused: .* the batch.s 8 live slots of 262144 positions 8 x [0-9.]* MiB KV + their logits = [0-9.]* GiB of it$" "$tmp/gate-small-batch.log"'

echo "--- 5. pod/e07.sh --dry-run, E07_PLAN=cbatch; pod/devgpu.sh --dry-run, DEVGPU_PLAN=c03; the cbatch rehearsal"
rc=0; E07_PLAN=cbatch OUT="$tmp/dry-out" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry.txt" 2>&1 || rc=$?
grep -E "^plan \((volume|hf)\): total|^price limit|^GPUs, in order" "$tmp/dry.txt" | sed 's/^/  /' | cut -c1-200 || true
check "the cbatch dry run exits 0 (exit $rc)" '(( rc == 0 ))' || tail -n 10 "$tmp/dry.txt"
check "the RTX PRO 6000 only, the machine qualification first" \
  'grep -q "^GPUs, in order: NVIDIA RTX PRO 6000 Blackwell Server Edition;NVIDIA RTX PRO 6000 Blackwell Workstation Edition; host RAM at least 256 GB" "$tmp/dry.txt" &&
   ! grep -q "A100.* takes payload" "$tmp/dry.txt" &&
   [[ "$(grep -A1 "^plan with the model from HF" "$tmp/dry.txt" | tail -n 1)" == "  create, SSH, CUDA probe, qualification"* ]]'
sed -n "/^plan with the model from HF /,/^plan (hf)/p" "$tmp/dry.txt" | awk '/^  / { print $1 }' > "$tmp/planned.txt"
want_hf="create, payload env model-headers gate gate-off setup model-download dry-run engine-server-off load-short-off server-stop-off engine-server load-short load-c4 load-c8 steady-c4 det-sampling server-stop pull"
check "the HF plan's steps: $want_hf" '[[ "$(tr "\n" " " < "$tmp/planned.txt")" == "$want_hf " ]]'
check "every pod step of both model paths with its timeout, 2 x its estimate" \
  '(( $(sed -n "/^plan with the model from /,/^plan (hf)/p" "$tmp/dry.txt" | grep -c "; timeout min(") == 33 )) &&
   grep -q "^  steady-c4 (chat_load.py steady, 3 x: 4 prompts at 4.9 s, 1024 tokens at 20 tok/s, the reset 2.0 s; the 15 s window inside, + 0.2) *3.8  (total [0-9.]*; timeout min(7.6 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  load-c8 (chat_load.py, N = 8 decoding at once: 2 turns each, at worst 16 requests one at a time at 14.2 s, + 0.2) *4.0  (total [0-9.]*; timeout min(8.0 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  model-download (264.5 GB at 0.25 GB/s, .*18.2  (total [0-9.]*; timeout min(36.4 min, the time left))$" "$tmp/dry.txt"'
usd="$(sed -n "s/^plan (hf): total .* (about \([0-9.]*\) USD at 2.60 USD\/h)$/\1/p" "$tmp/dry.txt")"
check "the HF plan's estimate under 3.00 USD at 2.60 USD/h (${usd:-none} USD), the session cap 3.00 over at most 3 pods" \
  '[[ -n "$usd" ]] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 3.0 else 1)" "$usd" &&
   grep -q "^price limit: 2.60 USD/h; session cap: 3.00 USD over at most 3 pods" "$tmp/dry.txt"'
check "the gate per GPU with the batch's slots and checkpoints, the payload's sources (the engine HEAD's, the short conversation HEAD's build)" \
  'grep -q "^    e07_gate.py: batch: 8 live slots of 69632 positions" "$tmp/dry.txt" &&
   grep -q "the tarball.s engine is HEAD.s$" "$tmp/dry.txt" &&
   grep -q "src/tools/dsv41/quality/det_sampling_check.py: last changed in " "$tmp/dry.txt" &&
   grep -q "^  e07/conversations-short.json: .*, chat_load.py build --users 1 --context 1024 at HEAD$" "$tmp/dry.txt"'
rc=0; DEVGPU_PLAN=c03 OUT="$tmp/devgpu-dry" bash "$STRATA_DIR/tools/dsv41/pod/devgpu.sh" --dry-run > "$tmp/devgpu.txt" 2>&1 || rc=$?
grep -E "^plan: total|^price limit|^GPUs" "$tmp/devgpu.txt" | sed 's/^/  /' | cut -c1-200 || true
dusd="$(sed -n "s/^plan: total .* min, about \([0-9.]*\) USD at 0.70 USD\/h .*/\1/p" "$tmp/devgpu.txt")"
check "the batch dev pod's plan: exit $rc, its steps, about ${dusd:-?} USD at 0.70 USD/h, under 1.50" \
  '(( rc == 0 )) && [[ -n "$dusd" ]] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 1.5 else 1)" "$dusd" &&
   grep -q "^  batch-kernels (dsv41_engine_batch_test --kernels" "$tmp/devgpu.txt" && grep -q "^  batch-e2e " "$tmp/devgpu.txt" &&
   grep -q "^payload: .*dsv41_engine_batch_test and MANIFEST from " "$tmp/devgpu.txt"'

rc=0; CBATCH_STEADY_WINDOW_S=4 E07_PLAN=cbatch OUT="$tmp/payload" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --payload-only > "$tmp/payload.log" 2>&1 || rc=$?
check "e07.sh --payload-only builds the cbatch payload" '(( rc == 0 )) && [[ -f "$tmp/payload/payload-cu130.tar.gz" ]]' ||
  tail -n 5 "$tmp/payload.log"
rm -rf -- "$tmp/payload/payload-cu130/"
bin="$tmp/bin" stubs="$tmp/stubs"
mkdir -p "$bin" "$stubs"
# c02-test.sh's stubs: the GPU, numactl, pip, the download tools, the engine, the HF headers and download; the gate the
# payload's own at a 300 GB host
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# c03-test.sh's GPU: an RTX PRO 6000's 97887 MiB, no NUMA node for GPU0
case " $* " in
  *" topo "*) printf '\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t0-11\tN/A\t\tN/A\n' ;;
  *" --query-gpu=memory.total --format=csv,noheader,nounits "*) echo 97887 ;;
  *" --query-gpu=memory.used,memory.total --format=csv,noheader "*) echo "1024 MiB, 97887 MiB" ;;
  *) echo "nvidia-smi (c03-test.sh stub): GPU 0, 97887 MiB" ;;
esac
EOF
cat > "$bin/numactl" <<'EOF'
#!/usr/bin/env bash
while [[ "${1:-}" == --* ]]; do shift; done
exec "$@"
EOF
printf '#!/usr/bin/env bash\necho "pip (c03-test.sh stub, nothing installed): $*"\n' > "$bin/pip"
for t in aria2c apt-get; do
  printf '#!/usr/bin/env bash\necho "%s (c03-test.sh stub): nothing downloads or installs here: $*" >&2\nexit 1\n' "$t" > "$bin/$t"
done
cat > "$stubs/dsv41-engine" <<'EOF'
#!/usr/bin/env bash
# c03-test.sh's engine: --dry-run exits 0, --serve is fakes.py's engine (the tokenizer from --model)
case " $* " in *" --dry-run "*) echo "dsv41-engine (c03-test.sh stub): dry run of $*"; exit 0 ;; esac
args=()
for a in "$@"; do [[ "$a" == --serve ]] || args+=("$a"); done
exec python3 "$(dirname "$0")/src/tools/dsv41/quality/fakes.py" --serve "${args[@]}"
EOF
cat > "$stubs/hf_model.py" <<'EOF'
#!/usr/bin/env python3
"""c03-test.sh's hf_model.py: `headers MANIFEST DIR` copies the real heads (C03_HEADERS/head-N.bin) into DIR."""
import os
import shutil
import sys

if len(sys.argv) != 4 or sys.argv[1] != "headers":
    sys.exit(f"hf_model.py (c03-test.sh stub): headers MANIFEST DIR only, not {sys.argv[1:]}")
os.makedirs(sys.argv[3], exist_ok=True)
for i in range(1, 8):
    shutil.copy(os.path.join(os.environ["C03_HEADERS"], f"head-{i}.bin"), sys.argv[3])
EOF
cat > "$stubs/hf-download.sh" <<'EOF'
#!/usr/bin/env bash
# c03-test.sh's download: the sparse shards (C03_MODEL) copied into DST, the record the sheet reads, marked a stub
set -euo pipefail
mkdir -p "$2"
cp --sparse=always "$C03_MODEL"/*.gguf "$2"/
python3 -c "import json, sys; json.dump({'stub': 'c03-test.sh: the sparse shards copied, no download', 'repo': 'c03-test-stub',
  'revision': 'none', 'files': 7, 'bytes': 0, 'download_s': 0, 'gb_per_s': 0, 'check_s': 0, 'ok': True}, open(sys.argv[1], 'w'))" "$3"
EOF
cat > "$stubs/e07_gate.py" <<'EOF'
#!/usr/bin/env python3
"""c03-test.sh's gate: the payload's own e07_gate.py (e07_gate_real.py) at a 300 GB host."""
import os
import sys

a = sys.argv[1:]
for key, value in (("--cgroup-bytes", "300000000000"), ("--mem-total-kib", "300000000")):
    a[a.index(key) + 1] = value
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e07_gate_real.py"), *a])
EOF
chmod +x "$bin"/* "$stubs"/*
s="$tmp/s-cbatch"
mkdir -p "$s/pod/out" "$s/root/root"
tar -xzf "$tmp/payload/payload-cu130.tar.gz" -C "$s/pod" --exclude='libcublas*' --exclude='libggml-cuda.so*'
cp "$stubs/dsv41-engine" "$stubs/hf_model.py" "$stubs/hf-download.sh" "$s/pod/"
mv "$s/pod/src/tools/dsv41/pod/e07_gate.py" "$s/pod/src/tools/dsv41/pod/e07_gate_real.py"
cp "$stubs/e07_gate.py" "$s/pod/src/tools/dsv41/pod/"
free_port port
( cd "$s/pod" && export PATH="$bin:$PATH" E07_POD_ROOT="$s/root" E07_Q_PORT="$port" C03_MODEL="$tmp/model" \
    C03_HEADERS="$HEADERS" POD_MACHINE="c03-cbatch" FAKE_ENGINE_PROMPT_MS_PER_TOKEN="$P_MS" \
    FAKE_ENGINE_DECODE_MS_PER_TOKEN="$D_MS" FAKE_ENGINE_BATCH_ROW_MS="$R_MS"
  rc=0; DEADLINE=$(( $(date +%s) + 2400 )) POD_VOLUME= timeout 2700 bash e07-run.sh > out/e07-run.out 2>&1 < /dev/null || rc=$?
  echo "$rc" > out/e07-run.exit ) &
jobs_+=($!)
wait "${jobs_[@]}"
jobs_=()
o="$s/pod/out"
awk '/^step / { sub(":$", "", $2); print $2 }' "$o/steps.log" > "$tmp/ran.txt"
echo "  cbatch: $(tr '\n' ' ' < "$tmp/ran.txt")"
grep -vxE 'create,|payload|pull' "$tmp/planned.txt" > "$tmp/planned-pod.txt" || true
check "e07-run.sh exits 0, every step exit 0" \
  '[[ "$(cat "$o/e07-run.exit")" == 0 ]] && ! grep "^step " "$o/steps.log" | grep -qv ": exit 0, "'
check "steps.log has every planned step in order, then done" \
  'diff "$tmp/planned-pod.txt" "$tmp/ran.txt" > /dev/null && [[ "$(tail -n 1 "$o/steps.log")" == done ]]'
check "both servers as planned: off with gate-off's placement and no --batch, on with gate's, --batch 8 --batch-ctx 69632 (the deployment's) --checkpoints 8 --log-ids" \
  'python3 -c "import json, sys; off, on = (json.load(open(f))[\"args\"] for f in sys.argv[1:3])
g, goff = (json.load(open(f)) for f in sys.argv[3:5])
v = lambda a, k: a[a.index(k) + 1]
sys.exit(0 if \"--batch\" not in off and v(off, \"--hot-set\") == \"out/gate-off/hot-set.json\" and v(off, \"--checkpoints\") == \"8\" and
         v(off, \"--vram-gb\") == str(goff[\"vram_gb\"]) and v(on, \"--batch\") == \"8\" and v(on, \"--batch-ctx\") == \"69632\" and
         v(on, \"--checkpoints\") == \"8\" and \"--log-ids\" in on and v(on, \"--vram-gb\") == str(g[\"vram_gb\"]) and
         g[\"batch\"] == 8 and goff[\"batch\"] == 0 and g[\"vram_gb\"] < goff[\"vram_gb\"] else 1)" \
     "$o/off-quality-server.json" "$o/quality-server.json" "$o/gate/gate.json" "$o/gate-off/gate.json"'
check "load-short-off and load-short (4 turns: e07/cbatch-short), load-c4 and load-c8 came back, every request answered; steady-c4: 3 repeats of a 4 s window (e07/cbatch-steady), no steady-c8 in this plan" \
  'for f in load-short-off:4 load-short:4 load-c4:8 load-c8:16; do python3 -c "import json, sys; j = json.load(open(sys.argv[1])); sys.exit(0 if j[\"ok\"] and len(j[\"runs\"][0][\"rows\"]) == int(sys.argv[2]) else 1)" "$o/cbatch/${f%:*}.json" "${f#*:}" || exit 1; done
   [[ "$(cat "$s/pod/e07/cbatch-short")" == 4 && -z "$(cat "$s/pod/e07/cbatch-steady-report")" && ! -e "$o/cbatch/steady-c8.json" ]] &&
   [[ "$(cat "$s/pod/e07/cbatch-steady")" == "4 4 3 1024" ]] &&
   python3 -c "import json, sys; j = json.load(open(sys.argv[1])); sys.exit(0 if j[\"ok\"] and j[\"users\"] == 4 and j[\"window_s\"] == 4 and j[\"summary\"][\"measured\"] == 3 else 1)" "$o/cbatch/steady-c4.json"'
check "det-sampling: items 7 and 8 PASS, each request alone (the solo path) with the batch slots on" \
  'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); i7, i8 = j[\"items\"][\"7\"], j[\"items\"][\"8\"]
sys.exit(0 if j[\"ok\"] and i7[\"pass\"] and i8[\"seeds\"][\"pass\"] and i8[\"speed\"][\"pass\"] and i8[\"penalty\"][\"pass\"] else 1)" "$o/det-sampling.json"'
check "the server was stopped" \
  'grep -q "^server [0-9]* stopped$" "$o/server-stop.log" && grep -q "^server [0-9]* stopped$" "$o/server-stop-off.log"'
f="$tmp/report/20261015T090000Z"
mkdir -p "$f"
cp -r "$o" "$f/out"
echo 3.00 > "$f/session-cap"
echo "pod c03stub: 3000 s at 2.6 USD/h = 2.167 USD (upper bound: billing rounds up)" > "$f/cost.txt"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/sheets" > "$tmp/report.log" 2>&1 || true
r="$tmp/sheets/2026-10-15-cbatch/README.md"
check "the cbatch sheet: the session's cost against its cap, item 11's table and verdicts, the runs, the batch's lines, both gates" \
  'grep -q "^Build: Strata .*Session: 2.167 USD over 1 pod, the session.s cap 3.00 USD: within it. " "$r" &&
   grep -q "^## Item 11$" "$r" &&
   grep -q "^| 1 user, short context | [0-9.]* | [0-9.]* | .* | the single user: >= 30 tok/s, within 5% of off |$" "$r" &&
   grep -q "^| 4 users at once, end to end |  | [0-9.]* | [0-9.]* | [0-9.]* | [0-9.]* | user-visible, the prompt reads included (reported) |$" "$r" &&
   grep -q "^| 8 users at once, end to end |  | [0-9.]* | [0-9.]* | [0-9.]* | [0-9.]* | user-visible, the prompt reads included (reported) |$" "$r" &&
   grep -q "^| 4 users decoding at once, steady (3 x 4 s, 4 rows a step) |  | [0-9.]* | [0-9.]* | [0-9.]* |  | item 11: aggregate >= 2 x the single user with batching off, each >= 20 tok/s |$" "$r" &&
   (( $(grep -c "^- item 11" "$r") == 1 )) && ! grep -qE "^- (single user|4 users): (PASS|FAIL)" "$r" &&
   grep -qE "^- item 11: (PASS|FAIL): 4 users decoding at once \(steady-c4: 3 repeats of a 4 s window with exactly 4 rows a step\) [0-9.]* tok/s in aggregate \(the mean; repeats [0-9.]* to [0-9.]*\) = [0-9.]*x the baseline, the single user with batching off \([0-9.]* tok/s; [0-9.]*x the [0-9.]* with batching on\), the slowest user [0-9.]* tok/s \(goal: >= 2x and each >= 20\); the single user [0-9.]* tok/s with batching on against [0-9.]* off .*\. User-visible, the prompt reads included \(end to end, this code\): [0-9.]* tok/s in aggregate over the decode window \([0-9.]* over the run.s wall time\), the slowest user [0-9.]* tok/s \(load-c4\)$" "$r" &&
   grep -q "^## Item 11.s steady state (logs/cbatch/steady-c4.json)$" "$r" &&
   grep -q "^| aggregate (mean of the repeats) | [0-9.]* | [0-9.]*x | [0-9.]*x |$" "$r" &&
   grep -q "^## The batch in the engine.s log (logs/quality-engine.log)" "$r" &&
   grep -qE "^[0-9]+ requests decoded in batch slots, [0-9]+ of them reproducible" "$r" &&
   grep -q "^## Placement: gate (the batch) (logs/gate/gate.json)" "$r" && grep -q "^## Placement: gate-off (no batch) (logs/gate-off/gate.json)" "$r" &&
   grep -q "^## Items 7 and 8 once (logs/det-sampling.json)" "$r" && grep -q "^## Server as run, batching off" "$r" &&
   grep -q "^## Machine qualification" "$r"' || tail -n 5 "$tmp/report.log"
if (( failed )); then
  echo "  the tail of e07-run.out and steps.log:"
  tail -n 5 "$o/e07-run.out" "$o/steps.log" 2>/dev/null | sed 's/^/    /' || true
fi

echo "c03-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
