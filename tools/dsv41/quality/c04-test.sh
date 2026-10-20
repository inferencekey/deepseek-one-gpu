#!/usr/bin/env bash
# The long-prompt and robustness checks without a GPU: no model, no RunPod, no HF, no money.
#   1. in the cu130 builder image (tools/dsv41/build.sh's; no GPU): dsv41_engine_batch_test on tiny-b (the CPU
#      reference; check 6: a 2600-token prompt read in chunks of 1024 with a step between two beside 3 decoding slots,
#      bit for bit its alone run with the same chunks, the slots bit for bit theirs), dsv41-engine --serve --dry-run
#      --batch 4 --interleave-chunk 8 on tiny-b: a 40-token prompt read in 5 chunks with a slot's BT line after each
#      but the last, a BSTOP that came during the read ending its slot before the next step, INFO interleave_chunk=8,
#      the "read beside" line's chunks, chunk read times and steps; --interleave-chunk past --chunk and without --batch
#      refused; the real layout's dry run with clong's 8 slots of 69632 positions: its "batch slots:" line the gate's,
#      INFO interleave_chunk=16384, its --chunk; batch_test's check 7 (the slots' lane argmax picks the plain loop's
#      id), its check 8 (a request alone picks with it, the plain loop's id on every row) and the dry run's "batch
#      steps" lines (all the steps, those within 15 s of the read beside the slots, the others); batch_test's check 9
#      and the dry run's lone slot with a one-token request read beside it (rep=1 as alone, rep=0 with det=0), a
#      request past --batch-ctx saying ALONE before its read; the overflow hang seen in a clong session: a slot
#      stopped with nothing else decoding has its BDONE on stdout while the engine's stdin is still open (polled up to
#      30 s), and batch_test's check 4 flushes the BDONE lines of every slot stopped at once
#   2. the server's tests (python3 -m unittest discover serve in the builder image, as verify.sh runs them), and
#      serve/test_robust.py's: a client that leaves (streamed or not) frees its slot for the waiting one, the ninth
#      waiting request's 503 with Retry-After and its JSON error, a dead and a hung engine failing every pending
#      request (now with a code and Retry-After), non-streamed replies batched; a moved request's cached tokens, the
#      cut to a slot's room and the 400 past a slot, ALONE on /metrics and strata_ran_alone on the three routes, the
#      default max_tokens and the deadline that keeps the text; the cut to a slot's room said by the reply's
#      strata_max_tokens on the three routes, and quality/robust_check_test.py: slot-room's clamp acceptance on the
#      fake server (a reply that stops before the room with the cut proven PASS, no cut FAIL) and on clong session 5's
#      figures; a request without max_tokens told its limit by strata_max_tokens {"asked": null, ...} on the three
#      routes, and no-max-tokens' NEAR acceptance the same way (an early stop with the limit said PASS, nothing said
#      FAIL, sessions 4 and 5's NEAR figures)
#   3. quality/chat_load.py long against serve/server.py on fakes.py's engine (--chunk 16384, --batch 8 and, from the
#      deepseek-v4.1 profile's default, --batch-ctx 69632; 0.28 ms per prompt token, 5 ms a step of one row and 1.5 ms
#      per further row): 4 users decoding and the 64K user's request, read in the default interleave chunk (16384,
#      the --chunk): every user's largest gap within c04_expect.py's model (one chunk and one step) and under 5 s, the
#      64K TTFT against its solo run within 10% of the model; the same with --interleave-chunk 0 (the prompt read
#      whole): the gap is the whole 64K read, past 5 s (what interleaving guards against); quality/robust_check.py
#      PASS on the same server, with its no-max-tokens check (4 users decoding and a count without max_tokens admitted
#      to a slot with the default 8192, no user's gap past 5 s; one under a slot told its room by strata_max_tokens
#      and ending "length" there) and its slot-room check (4 users and a 40K prompt with max_tokens 32000 cut to its
#      slot's room, every gap <= 5 s; a turn past a slot refused with a 400, then run alone, said by /metrics,
#      strata_ran_alone and the engine's and the server's logs); overflow's last request answered within 5 s once
#      every client hung up, robust.json complete and each check's whole line printed
#   4. pod/e07_gate.py with clong's slots: their KV and the slab's loss on an RTX PRO 6000 host
#   5. pod/e07.sh --dry-run with E07_PLAN=clong and cbatch-clong: exit 0, the RTX PRO 6000 only, the qualification
#      first, every step with its timeout, the plan under 3 USD at 2.60 USD/h, its latest end (each step's whole
#      timeout) within the pod's deadline, the combined plan without load-c<N> and det-sampling and with steady-c8,
#      the single user's 4 turns, the payload's sources; then each plan's rehearsal: e07-run.sh from its payload with
#      c03-test.sh's stubs and fakes.py as the engine, every planned step exit 0, then done, the servers as planned,
#      the sheet (items 12 and 13; the combined one with item 11 too, on both cbatch servers side by side,
#      CBATCH_SPARE=4); the combined plan with CBATCH_SPARE=4 refused by its dry run (the pod's deadline), so
#      CBATCH_SPARE defaults to 0, and its sheet without the second server says the variant was not run and why; the
#      chunk note over every prompt read beside the slots, naming the longest chunk's read and the qualification's H2D
#   6. quality/pod_server.sh stop: a server that ignores SIGTERM and whose listening
#      socket outlives its SIGKILL by 4 s: the stop waits until the port is free, then the next server (serve/server.py
#      on fakes.py's engine) starts on the same port; a holder that never lets go: exit 1 naming its pid and command
#   7. pod/sheet_banners_test.py: the superseded banners and VERDICTS.md on a temporary
#      sheet root, idempotent (a second run changes no byte), and the table's errors refused
# Exits 1 if any check fails. c03-test.sh, c02-test.sh, c01-test.sh, q03-test.sh, q02-test.sh, analyze_test.py and
# verify.sh are run on their own.
#
#   BUILD_TAG=-c04 bash tools/dsv41/quality/c04-test.sh
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
CONVERSATIONS_LONG="${CONVERSATIONS_LONG:-$BUILD_ROOT/quality/conversations-64k.json}"
COUNTS="${COUNTS:-$BUILD_ROOT/counts.json}"
STRATA_BUILD="$BUILD_ROOT/cu130${BUILD_TAG:-}/strata"
for f in "$HEADERS/head-7.bin" "$VOCAB" "$E01/tiny-b.gguf" "$STRATA_BUILD/dsv41_engine_batch_test" "$CONVERSATIONS_LONG" "$COUNTS"; do
  [[ -f "$f" ]] || { echo "c04-test.sh: no $f" >&2; exit 2; }
done
# the fake engine's delays: a prompt token at the measured 3,572 tok/s (a 16K read in 4.59 s cold on the RTX PRO 6000, so a
# chunk of 16384 takes 4.59 s), c03-test.sh's 5 ms a step alone and 1.5 ms per further row
P_MS=0.28 D_MS=5 R_MS=1.5
tmp="$(mktemp -d)"
jobs_=()
cleanup() {
  local f
  for f in "${jobs_[@]}"; do kill "$f" 2>/dev/null || true; done
  for f in "$tmp"/server*.pid "$tmp"/s-*/pod/out/quality-server.pid "$tmp"/s-*/pod/out/holder.pid; do
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
docker image inspect "$image" > /dev/null 2>&1 || { echo "c04-test.sh: no $image (tools/dsv41/build.sh)" >&2; exit 2; }
d="$tmp/docker"
mkdir -p "$d"

echo "--- 1. the engine's CPU reference and the interleaved read of the dry run, in the cu130 builder image"
# slots 0 and 1 decode (30 tokens each); slot 2's 40-token prompt is read in chunks of 8 beside them; the BSTOP of slot 1
# comes after that BGEN, so the loop takes it during the read, before the first step between two chunks
python3 -c "print('BGEN 0 30 10,11,12,13'); print('BGEN 1 30 20,21')
print('BGEN 2 4 ' + ','.join(str(i) for i in range(40, 80))); print('BSTOP 1')" > "$d/interleave.txt"
# a lone slot (nothing decoding at its admission) and a request of one token read beside
# it; the same lone slot with nothing read beside it; a request past --batch-ctx 128 beside
# a decoding slot says ALONE before its read
# the lone slot rep=1 in the deterministic mode whatever is read beside it, rep=0 with det=0 (f1off); these dry
# runs and the interleaved read's take the GPU's numerics (--cpu-numerics gpu: the fork's are the
# default)
printf 'BGEN 0 12 10,11,12\nBGEN 1 1 20,21,22,23\n' > "$d/f1.txt"
printf 'BGEN 0 12 det=0 10,11,12\nBGEN 1 1 det=0 20,21,22,23\n' > "$d/f1off.txt"
printf 'BGEN 0 12 10,11,12\n' > "$d/lone.txt"
printf 'BGEN 0 30 10,11\nBGEN 1 200 20,21,22\n' > "$d/alone.txt"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_gate.py" --vram-mib 97887 --prefetch-experts 128 --cgroup-bytes 300000000000 \
  --mem-total-kib 300000000 --counts "$COUNTS" --layout "$STRATA_DIR/tools/dsv41/oracle/layout-dsv41-q2k.json" \
  --model-headers "$SHARD1" --out "$d/gate-long" --batch 8 --batch-ctx 69632 --checkpoints 8 > "$d/gate-long.log" 2>&1 || true
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e STRATA_BUILD="$STRATA_BUILD" \
  -e E01="$E01" -e SHARD1="$SHARD1" -e D="$d" -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$BUILD_ROOT:$BUILD_ROOT:ro" \
  -v "$tmp:$tmp" "$image" bash -euo pipefail -c '
    mkdir -p /tmp/cudastub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1
    export LD_LIBRARY_PATH=/tmp/cudastub
    "$STRATA_BUILD/dsv41_engine_batch_test" "$E01/tiny-b.gguf" > "$D/batch-test.txt" 2>&1 || true
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
      --interleave-chunk 8 --cpu-numerics gpu --dry-run < "$D/interleave.txt" > "$D/interleave.out" 2> "$D/interleave.err" || true
    for t in f1 f1off lone alone; do
      "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
        --cpu-numerics gpu --dry-run < "$D/$t.txt" > "$D/$t.out" 2> "$D/$t.err" || true
    done
    # the overflow hang of a clong session: a slot stopped with nothing else decoding, stdin left open (the server never closes it): its BDONE
    # must reach stdout then, not at the engine'"'"'s exit (the step that ends it has no row, and flushed nothing)
    # (polled up to 30 s; stdin stays open until the poll ends, BDONE seen or not)
    { printf "BGEN 0 120 10,11\nBSTOP 0\n"; while [ ! -e "$D/flush.polled" ]; do sleep 0.1; done; } |
      "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
      --dry-run > "$D/flush.out" 2> "$D/flush.err" &
    for _ in $(seq 300); do grep -q "^BDONE 0 " "$D/flush.out" && break; sleep 0.1; done
    cp "$D/flush.out" "$D/flush-open.out"
    touch "$D/flush.polled"
    wait $! || true
    for a in "--batch 4 --batch-ctx 128 --interleave-chunk 32:past" "--interleave-chunk 8:nobatch"; do
      rc_=0
      "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 ${a%:*} --dry-run < /dev/null \
        > "$D/refused-${a#*:}.out" 2>&1 || rc_=$?
      echo "$rc_" > "$D/refused-${a#*:}.rc"
    done
    "$STRATA_BUILD/dsv41-engine" --serve --model "$SHARD1" --ctx 262144 --chunk 16384 --batch 8 --batch-ctx 69632 \
      --dry-run < /dev/null > "$D/real.out" 2> "$D/real.err" || true' > "$tmp/docker.log" 2>&1 || rc=$?
check "the builder image ran the tests (exit $rc)" '(( rc == 0 )) && [[ -s "$d/batch-test.txt" && -s "$d/interleave.out" ]]' ||
  tail -n 20 "$tmp/docker.log"
grep -E "^(ok|FAIL) " "$d/batch-test.txt" | tail -n 2 | sed 's/^/  /' | cut -c1-220 || true
check "dsv41_engine_batch_test on tiny-b: PASS, check 6's interleaved read bit for bit its alone run with the same chunks, the 3 slots theirs" \
  '[[ "$(tail -n 1 "$d/batch-test.txt")" == "dsv41_engine_batch_test: PASS" ]] && ! grep -q "^FAIL " "$d/batch-test.txt" &&
   grep -q "^ok   a 2600-token prompt read in 3 chunks of 1024 (prefill chunk 2048) with 2 steps of 6 rows between them, beside 3 decoding slots: its tokens and logits its alone run.s with the same chunks bit for bit$" "$d/batch-test.txt" &&
   grep -q "^ok   the 3 slots decoding through the interleaved read: each its alone run.s (C03.s invariance): slot 0 bit for bit; slot 1 bit for bit; slot 2 bit for bit$" "$d/batch-test.txt"'
python3 - "$d/interleave.out" > "$d/interleave-check.txt" 2>&1 <<'EOF' || true
import sys
lines = open(sys.argv[1]).read().splitlines()
# slot 2's read: the PP lines from its BGEN to its BADM, and what came between them
start = next(i for i, x in enumerate(lines) if x.startswith("PP ") and x.split()[2] == "40")
end = next(i for i, x in enumerate(lines) if x == "BADM 2 1")
pp = [x for x in lines[start:end] if x.startswith("PP ")]
between = [x for x in lines[start:end] if x.startswith(("BT ", "BDONE "))]
info = next((x for x in lines if x.startswith("INFO ")), "")
ok = [x.split()[1] for x in pp] == ["8", "16", "24", "32", "40"]
# one step between two chunks (4 steps): slot 0's token each; slot 1's BSTOP before the first step
ok = ok and [x for x in between if x.startswith("BT ")] == [f"BT 0 {t}" for t in range(15, 19)]
# slot 1 in the deterministic mode beside slot 0 and the read: rep=1
ok = ok and [x for x in between if x.startswith("BDONE ")] == ["BDONE 1 1 cancel " + between[0].split()[4] + " rep=1"]
ok = ok and "batch_slots=4 batch_ctx=128 slot_cache=0 interleave_chunk=8" in info
print(f"interleaved read: PP {[x.split()[1] for x in pp]}, between them {between}; INFO {info[5:]!r}: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
sed 's/^/  /' "$d/interleave-check.txt" | cut -c1-240
check "the dry run's interleaved read: 5 chunks of 8, one step after each but the last (slot 0's tokens), the BSTOP sent during the read ending slot 1 before the first step, INFO interleave_chunk=8" \
  'grep -q ": PASS$" "$d/interleave-check.txt"'
check "batch_test check 7, the lanes' argmax the plain loop's on 2002 rows, and check 8, a request alone picking with it the same 2002 ids; the dry run's batch steps lines: all, within 15 s of the read beside the slots, the others" \
  'grep -q "^ok   batch_argmax picks the plain loop.s id on 2002 rows (ties, NaN, -inf, +inf, lengths 1 to 40 and 129280): 0 differ$" "$d/batch-test.txt" &&
   grep -q "^ok   a request alone (the Session, greedy) picks the plain loop.s id on each of the 2002 rows, 2002 tokens: 0 differ$" "$d/batch-test.txt" &&
   grep -qE "^dsv41-engine serve: batch steps, all: [0-9]+ steps of [0-9.]+ rows, [0-9.]+ ms a step from one step.s start to the next.s .*between steps [0-9.]+ \(the picks [0-9.]+, the BT lines and their flush [0-9.]+, admissions [0-9.]+, the loop and its stdin poll [0-9.]+\); per step " "$d/interleave.err" &&
   grep -qE "^dsv41-engine serve: batch steps, within 15 s of a prompt read beside the slots: [0-9]+ steps " "$d/interleave.err"'
check "the request's line: read beside 2 decoding slots, 40 tokens in 5 chunks of 8 with 4 batch steps between them, each chunk's read time, its longest hold" \
  'grep -qE "^dsv41-engine serve: request 3: read beside 2 decoding batch slots, 40 tokens in 5 chunks of 8 with 4 batch steps between them, a chunk read in [0-9.]+ ms on average and [0-9.]+ at most, [0-9.]+ ms from its start to its first token, the slots held at most [0-9.]+ ms at once; reproducible: its expert placement followed theirs, which no longer moves its bits$" "$d/interleave.err"'
grep -hE "^(BDONE|ALONE) " "$d/f1.out" "$d/f1off.out" "$d/lone.out" "$d/alone.out" | sed 's/^/  /' | cut -c1-200 || true
check "batch_test check 9 (the read beside moves the slab, the slot reproducible); the dry run's lone slot with a one-token request read beside it ends rep=1 (det=0: rep=0), alone with nothing read beside it rep=1" \
  'grep -q "^ok   a lone slot with a max_new 1 request read beside it after 3 steps (and the slab moved to the odd experts from then on): marked reproducible (alone, nothing read beside it: marked reproducible); its tokens its alone run.s bit for bit$" "$d/batch-test.txt" &&
   grep -qE "^BDONE 0 12 length [0-9.]+ rep=1$" "$d/f1.out" && grep -qE "^BADM 1 0 rep=1$" "$d/f1.out" &&
   grep -qE "^BDONE 0 12 length [0-9.]+ rep=0$" "$d/f1off.out" && grep -qE "^BADM 1 0 rep=0$" "$d/f1off.out" &&
   grep -qE "^BDONE 0 12 length [0-9.]+ rep=1$" "$d/lone.out"'
check "a request past --batch-ctx beside a decoding slot says ALONE before its read (stdout) and runs alone on stderr first, then its end line; the slot beside it rep=1" \
  'python3 -c "import sys; l = open(sys.argv[1]).read().splitlines()
a = next(i for i, x in enumerate(l) if x.startswith(\"ALONE \"))
p = next(i for i, x in enumerate(l) if x.startswith(\"PP \") and x.split()[2] == \"3\")
sys.exit(0 if l[a] == \"ALONE 1 1 why=3 prompt tokens and 200 to generate need 202 positions, a batch slot holds 128\" and a < p else 1)" "$d/alone.out" &&
   grep -q "^dsv41-engine serve: request 2: runs alone on the live sequence, not in batch slot 1: 3 prompt tokens and 200 to generate need 202 positions, a batch slot holds 128 (C04 audit 2); it holds 1 decoding batch slots until its end$" "$d/alone.err" &&
   grep -qE "^dsv41-engine serve: request 2: ran alone on the live sequence, not in batch slot 1: .* holding 1 decoding batch slots for [0-9.]+ ms$" "$d/alone.err" &&
   grep -qE "^BDONE 0 30 length [0-9.]+ rep=1$" "$d/alone.out"'
grep -E "^(BADM|BDONE) " "$d/flush-open.out" | sed 's/^/  stdin open: /' | cut -c1-200 || true
check "the overflow hang: a slot stopped with nothing else decoding: its BDONE (cancel) on stdout while the engine's stdin is still open, not at its exit (the server frees a slot only at its BDONE)" \
  'grep -q "^BADM 0 1$" "$d/flush-open.out" && grep -qE "^BDONE 0 [0-9]+ cancel [0-9.]+ rep=[01]$" "$d/flush-open.out"'
check "batch_test: BSTOP of every slot at once ends them in a step with no row, their BDONE lines flushed" \
  'grep -q "^ok   BSTOP of every slot at once: both end (cancel) in a step of 0 rows, their BDONE lines flushed (0 left unflushed)$" "$d/batch-test.txt"'
check "--interleave-chunk past --chunk and without --batch refused, naming the flag" \
  '[[ "$(cat "$d/refused-past.rc")" != 0 && "$(cat "$d/refused-nobatch.rc")" != 0 ]] &&
   grep -q -- "--interleave-chunk 32 is past the prefill chunk of 16" "$d/refused-past.out" &&
   grep -q -- "--interleave-chunk is --batch.s" "$d/refused-nobatch.out"'
slot="$(python3 -c "import json, sys; print(round(json.load(open(sys.argv[1]))['batch_slot_bytes'] / 2**20, 1))" "$d/gate-long/gate.json" 2>/dev/null || echo none)"
grep -E "^batch slots:" "$d/real.err" | sed 's/^/  /' | cut -c1-200 || true
check "the real layout's dry run with clong's 8 slots of 69632: a slot's device KV the gate's ($slot MiB), INFO interleave_chunk=16384 (its --chunk, the default)" \
  'grep -q "^batch slots: 8 live sequences of $slot MiB device KV each (a slot of 69632 positions" "$d/real.err" &&
   grep -q "^INFO .* batch_slots=8 batch_ctx=69632 slot_cache=0 interleave_chunk=16384 engine=dsv41-engine$" "$d/real.out"'

echo "--- 2. the server's tests (python3 -m unittest discover serve in the builder image, as verify.sh runs them)"
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e D="$d" \
  -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$d:$d" "$image" bash -euo pipefail -c '
    mkdir -p /tmp/serve-tests && cd /tmp/serve-tests
    export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STRATA_DIR"
    python3 -m unittest discover -s "$STRATA_DIR/serve" -t "$STRATA_DIR" -p "test_*.py" > "$D/serve-tests.log" 2>&1 || true
    python3 -m unittest -v serve.test_robust > "$D/serve-c04.log" 2>&1 || true
    # unittest writes to stderr, the server its log lines to stdout: apart, so each test'"'"'s line stays whole
    python3 "$STRATA_DIR/tools/dsv41/quality/robust_check_test.py" -v > "$D/robust-check-test.out" \
      2> "$D/robust-check-test.log" || true' \
  > "$tmp/docker-serve.log" 2>&1 || rc=$?
grep -E "^(FAIL|ERROR):" "$d/serve-tests.log" | sed 's/^/  /' || true
tail -n 3 "$d/serve-tests.log" | sed 's/^/  /' || true
check "serve/test_*.py pass (docker exit $rc)" '(( rc == 0 )) && [[ "$(tail -n 1 "$d/serve-tests.log")" == "OK"* ]]'
grep -E " \.\.\. (ok|FAIL|ERROR)$" "$d/serve-c04.log" | sed 's/^/  /' | cut -c1-200 || true
check "serve/test_robust.py: freed slots (streamed and not), the ninth waiting 503, a dead and a hung engine, non-streamed batched; the slot's room for a request without max_tokens, BADM alone= read, --ctx checked (C04 audit 1, 2, 6); audit cycle 2: a moved request's cached tokens (F2), max_tokens cut to a slot's room and a prompt past a slot refused beside decoding requests (M1), ALONE at its start on /metrics and strata_ran_alone on the 3 routes (M2), the default max_tokens and the deadline ending with its text (M3), Retry-After's floor and the alone hold, the engine-down 503s' code; the overflow hang: after 8 runners and 8 waiting requests (streamed and not) hang up at once the next request is answered within 5 s; C04 session 5: the cut said by strata_max_tokens on the 3 routes; C04d: the limit given to a request without max_tokens said by strata_max_tokens (asked null) on the 3 routes" \
  'grep -q "^Ran 19 tests in " "$d/serve-c04.log" && [[ "$(tail -n 1 "$d/serve-c04.log")" == OK ]] &&
   for t in test_a_moved_request_counts_only_its_cached_prefix test_max_tokens_past_a_slot_is_cut_to_its_room \
            test_a_prompt_past_a_slot_is_refused_beside_decoding_requests test_ran_alone_is_said_at_its_start_and_on_every_route \
            test_no_max_tokens_takes_the_profile_default test_a_request_past_its_deadline_ends_with_its_text \
            test_retry_after_floor_and_alone_hold test_after_a_burst_of_cancelled_waiters_the_next_request_is_answered \
            test_the_cut_is_said_in_the_reply_on_every_route test_the_limit_without_max_tokens_is_said_on_every_route; do grep -q "^$t (serve.test_robust" "$d/serve-c04.log" || echo "no $t"; done |
     { ! grep -q .; }'
grep -E "^test_.* \.\.\. |^(ok|FAIL|ERROR)$|^Ran " "$d/robust-check-test.log" | sed 's/^/  /' | cut -c1-200 || true
check "robust_check_test.py: slot-room's clamp turn on the fake server, a reply that stops before the room with the cut proven by strata_max_tokens PASS, the same with the cut turned off (no strata_max_tokens) or the reply saying max_tokens 2000 FAIL; session 5's figures (room 137, stop after 13) PASS with the field, FAIL without it; C04d: NEAR's turn without max_tokens, an early stop with its limit said (asked null, the room) PASS, nothing said FAIL; sessions 4 and 5's NEAR figures (room 160, length at 160) PASS with the field, FAIL without it" \
  'grep -q "^Ran 6 tests in " "$d/robust-check-test.log" && [[ "$(tail -n 1 "$d/robust-check-test.log")" == OK ]] &&
   for t in Clamp:test_early_stop_with_the_cut_proven_passes Clamp:test_no_cut_fails Clamp:test_session5 \
            Near:test_near_early_stop_with_the_limit_said_passes Near:test_near_without_the_limit_said_fails \
            Near:test_sessions_4_and_5; do      # class prefix:test name
     grep -q "^${t#*:} (__main__\.${t%%:*}" "$d/robust-check-test.log" || echo "no $t"; done | { ! grep -q .; }'

echo "--- 3. chat_load.py long and robust_check.py against serve/server.py on the fake engine (P ${P_MS}, D ${D_MS}, ${R_MS} ms a further row)"
serve_fake() {   # serve_fake NAME INTERLEAVE: the server on fakes.py's engine with --chunk 16384 and --batch 8 of the
  # profile's default slots (69632 positions); INTERLEAVE "default" leaves "interleave_chunk" out (the engine's --chunk)
  local dir="$tmp/$1" port
  mkdir -p "$dir"
  cat > "$dir/engine" <<EOF
#!/usr/bin/env bash
exec python3 "$here/fakes.py" --serve "\$@"
EOF
  chmod +x "$dir/engine"
  python3 -c "import json, sys; d = sys.argv[1]
c = {'exe': d + '/engine', 'cwd': d, 'args': ['--model', sys.argv[2], '--chunk', '16384', '--ctx', '262144',
  '--checkpoints', '8'], 'model_profile': 'deepseek-v4.1', 'tokenizer_gguf': sys.argv[2], 'log': d + '/engine.log',
  'repeat_stop_tokens': 0, 'parallel': 8}
if sys.argv[3] != 'default':
    c['interleave_chunk'] = int(sys.argv[3])
print(json.dumps(c))" "$dir" "$VOCAB" "$2" > "$dir/config.json"
  free_port port
  ( cd "$STRATA_DIR" && FAKE_ENGINE_PROMPT_MS_PER_TOKEN=$P_MS FAKE_ENGINE_DECODE_MS_PER_TOKEN=$D_MS \
      FAKE_ENGINE_BATCH_ROW_MS=$R_MS exec setsid python3 -m serve.server --engine strata --config "$dir/config.json" \
      --port "$port" ) > "$dir/server.log" 2>&1 < /dev/null &
  echo $! > "$tmp/server-$1.pid"
  for _ in $(seq 120); do grep -q "ready: http" "$dir/server.log" && break; sleep 1; done
  echo "$port" > "$dir/port"
}
for k in default 0; do
  serve_fake "chunk$k" "$k"
  rc=0
  timeout 900 python3 "$here/chat_load.py" long --url "http://127.0.0.1:$(cat "$tmp/chunk$k/port")" \
    --conversations "$CONVERSATIONS_LONG" --users 4 --out "$tmp/chunk$k/long.json" > "$tmp/chunk$k/long.txt" 2>&1 || rc=$?
  echo "$rc" > "$tmp/chunk$k/long.rc"
  grep -E "^\| (0|1|2|3|alone|beside) |^Largest gap" "$tmp/chunk$k/long.txt" | sed "s/^/  chunk $k: /" | cut -c1-220 || true
  python3 "$here/c04_expect.py" "$tmp/chunk$k/long.json" --prompt-ms "$P_MS" --decode-ms "$D_MS" --row-ms "$R_MS" \
    --chunk "$([[ $k == default ]] && echo 16384 || echo "$k")" > "$tmp/chunk$k/expect.txt" 2>&1 || true
  sed "s/^/  chunk $k: /" "$tmp/chunk$k/expect.txt" | cut -c1-220
  grep -E "read beside 4 decoding" "$tmp/chunk$k/engine.log" | sed "s/^.*: read beside/  chunk $k: read beside/" | cut -c1-260 || true
  if [[ "$k" == default ]]; then
    rc=0
    timeout 600 python3 "$here/robust_check.py" --url "http://127.0.0.1:$(cat "$tmp/chunk$k/port")" \
      --out "$tmp/chunk$k/robust.json" > "$tmp/chunk$k/robust.txt" 2>&1 || rc=$?
    echo "$rc" > "$tmp/chunk$k/robust.rc"
    grep -E "^robust_check.py: " "$tmp/chunk$k/robust.txt" | sed 's/^/  /' | cut -c1-200 || true
  fi
  kill -TERM -- "-$(cat "$tmp/server-chunk$k.pid")" 2>/dev/null || true
done
check "the default config (no interleave_chunk, no batch_ctx): the engine started with --batch 8 --batch-ctx 69632, the profile's default, and no --interleave-chunk" \
  'grep -qE "engine started: .* --chunk 16384 .* --batch 8 --batch-ctx 69632$" "$tmp/chunkdefault/engine.log" &&
   ! grep -q -- "--interleave-chunk" "$tmp/chunkdefault/engine.log"'
check "chunk 16384 (the default): 4 users decoding through the 64K read, every request answered (exit $(cat "$tmp/chunkdefault/long.rc")), item 12's gap PASS" \
  '[[ "$(cat "$tmp/chunkdefault/long.rc")" == 0 ]] && python3 -c "import json, sys; j = json.load(open(sys.argv[1])); s = j[\"summary\"]
sys.exit(0 if j[\"ok\"] and s[\"covered\"] and s[\"gap_pass\"] and s[\"long_prompt_tokens\"] == 65565 and s[\"users\"] == 4 else 1)" "$tmp/chunkdefault/long.json"'
check "chunk 16384 (the default): every user's largest gap within c04_expect.py's model (one chunk and one step), the 64K TTFT alone and beside them within 10%" \
  'grep -qx "c04_expect.py: PASS" "$tmp/chunkdefault/expect.txt"'
check "chunk 16384 (the default): the engine's log has the 64K read in 5 chunks of 16384 with 4 steps beside the 4 users, each chunk's read time" \
  'grep -qE "^dsv41-engine serve: request [0-9]+: read beside 4 decoding batch slots, 65565 tokens in 5 chunks of 16384 with 4 batch steps between them, a chunk read in [0-9.]+ ms on average and [0-9.]+ at most" "$tmp/chunkdefault/engine.log"'
check "chunk 0 (the prompt read whole): the gap is the 64K read (c04_expect.py's model), past 5 s: item 12 FAIL, the check sees what interleaving guards against" \
  'grep -qx "c04_expect.py: PASS" "$tmp/chunk0/expect.txt" && python3 -c "import json, sys; s = json.load(open(sys.argv[1]))[\"summary\"]
sys.exit(0 if s[\"covered\"] and not s[\"gap_pass\"] and s[\"max_gap_s\"] > 5 else 1)" "$tmp/chunk0/long.json"'
check "robust_check.py on the fake server: every item 13 check PASS (exit $(cat "$tmp/chunkdefault/robust.rc" 2>/dev/null))" \
  '[[ "$(cat "$tmp/chunkdefault/robust.rc")" == 0 ]] && python3 -c "import json, sys; j = json.load(open(sys.argv[1]))
sys.exit(0 if j[\"ok\"] and [c[\"check\"] for c in j[\"checks\"]] == [\"stream\", \"non-stream\", \"batched\", \"disconnect-stream\", \"disconnect-non-stream\", \"overflow\", \"no-max-tokens\", \"slot-room\"] else 1)" "$tmp/chunkdefault/robust.json"'
check "overflow (the overflow hang): once every client hung up and the server drained, the last request answered 200 within 5 s" \
  'python3 -c "import json, sys; c = [x for x in json.load(open(sys.argv[1]))[\"checks\"] if x[\"check\"] == \"overflow\"][0]
sys.exit(0 if c[\"pass\"] and c[\"drained\"] and c[\"last_status\"] == 200 and c[\"last_answered_s\"] <= 5.0 else 1)" "$tmp/chunkdefault/robust.json"'
check "robust_check.py: the JSON written after every check (complete at the end), each check's whole JSON line printed" \
  'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); sys.exit(0 if j[\"complete\"] and j[\"ok\"] else 1)" "$tmp/chunkdefault/robust.json" &&
   [[ "$(grep -cE "^robust_check.py: [a-z-]+: PASS \{.*\}$" "$tmp/chunkdefault/robust.txt")" == 8 ]]'
check "no max_tokens: 4 users decoding and a count without max_tokens, admitted to a slot with the profile's default_max_tokens 8192 (/metrics), no user's gap past 5 s; one 160 tokens under a slot told its room by strata_max_tokens (asked null, C04d) and ending \"length\" there" \
  'python3 -c "import json, sys; c = [x for x in json.load(open(sys.argv[1]))[\"checks\"] if x[\"check\"] == \"no-max-tokens\"][0]
sys.exit(0 if c[\"runners\"] == 4 and c[\"admitted_to_a_slot\"] and c[\"its_tokens\"] > 0 and max(c[\"runner_gaps_s\"]) <= 5.0 and
         c[\"its_max_tokens\"] == 8192 == c[\"max_tokens_expected\"] and c[\"near_ok\"] and c[\"near_finish\"] == \"length\" and c[\"near_max_tokens\"] == c[\"near_room\"] and
         c[\"near_completion_tokens\"] == 69632 - c[\"near_prompt_tokens\"] + 1 else 1)" "$tmp/chunkdefault/robust.json"'
check "slot room: 4 users and a 40K prompt with max_tokens 32000 cut to its slot's room, every user's gap <= 5 s; a turn with max_tokens 32000 cut to its room (its reply's strata_max_tokens) and ending within it, its cached tokens under its prompt (F2); a turn past a slot refused with a 400 naming the slot" \
  'python3 -c "import json, sys; c = [x for x in json.load(open(sys.argv[1]))[\"checks\"] if x[\"check\"] == \"slot-room\"][0]
sys.exit(0 if c[\"pass\"] and c[\"long_admitted\"] and c[\"long_cut_ok\"] and 39000 < c[\"long_prompt_tokens\"] < 41000 and
         max(c[\"runner_gaps_s\"]) <= 5.0 and c[\"clamp_ok\"] and c[\"clamp_max_tokens\"] == c[\"clamp_room\"] and
         c[\"clamp_cached_tokens\"] <= c[\"clamp_prompt_tokens\"] and
         c[\"refused_ok\"] and c[\"past_status\"] == 400 else 1)" "$tmp/chunkdefault/robust.json" &&
   grep -q "^\[strata\] max_tokens 32000 past a batch slot.s room: answered up to [0-9]* tokens" "$tmp/chunkdefault/server.log" &&
   grep -q "^\[strata\] 400 invalid request: the prompt ([0-9]* tokens) is past a batch slot (69632 positions" "$tmp/chunkdefault/server.log"'
check "ran alone: the turn past a slot, alone and moved into a slot by a short request, runs alone: ALONE seen on /metrics while it runs (held_slots 0), strata_ran_alone in its reply, the engine's lines at its start and end, the server's log at its start" \
  'python3 -c "import json, sys; c = [x for x in json.load(open(sys.argv[1]))[\"checks\"] if x[\"check\"] == \"slot-room\"][0]
r, m = c[\"strata_ran_alone\"] or {}, c[\"alone_seen_in_metrics\"] or {}
sys.exit(0 if c[\"alone_ok\"] and r.get(\"held_slots\") == 0 == m.get(\"held_slots\") and \"a batch slot holds 69632\" in r.get(\"reason\", \"\") and c[\"short_status\"] == 200 else 1)" "$tmp/chunkdefault/robust.json" &&
   grep -qE "^dsv41-engine serve: request [0-9]+: runs alone on the live sequence, not in batch slot [0-9]: [0-9]+ prompt tokens and [0-9]+ to generate need [0-9]+ positions, a batch slot holds 69632 \(C04 audit 2\); it holds 0 decoding batch slots until its end$" "$tmp/chunkdefault/engine.log" &&
   grep -qE "^dsv41-engine serve: request [0-9]+: ran alone on the live sequence, not in batch slot [0-9]: .*, a batch slot holds 69632 \(C04 audit 2\); [0-9]+ generated \((length|stop)\), holding 0 decoding batch slots for [0-9.]+ ms$" "$tmp/chunkdefault/engine.log" &&
   grep -q "^\[strata\] batch slot [0-9]: this request runs alone, not in a slot, holding 0 decoding slot(s) until its end: " "$tmp/chunkdefault/server.log"'
check "the server's log: the cancelled slots freed, the 503 said" \
  'grep -q "^\[strata\] batch slot [0-9]*: its request was cancelled after [0-9]* tokens" "$tmp/chunkdefault/server.log" &&
   grep -q "^\[strata\] 503: the server.s queue is full: 8 requests are waiting for the 8 running (queue_max 8)" "$tmp/chunkdefault/server.log"'

echo "--- 4. pod/e07_gate.py with clong's slots"
grep -E "^e07_gate.py: (batch|refused|VRAM)" "$d/gate-long.log" | sed 's/^/  /' | cut -c1-220 || true
check "the RTX PRO 6000 with 8 slots of 69632: their KV (217.6 MiB each) and the slab's loss reported, it fits" \
  'grep -q "^e07_gate.py: batch: 8 live slots of 69632 positions, 217.6 MiB device KV each, 1740.6 MiB in all + the rows. logits 3.9 MiB = 1.70 GiB, taken from the slab (--batch 8 --batch-ctx 69632): it loses about [0-9]* experts" "$d/gate-long.log" &&
   python3 -c "import json, sys; g = json.load(open(sys.argv[1])); sys.exit(0 if g[\"batch\"] == 8 and g[\"batch_slab_experts_lost\"] > 0 else 1)" "$d/gate-long/gate.json"'

echo "--- 5. pod/e07.sh --dry-run with E07_PLAN=clong and cbatch-clong, and each plan's rehearsal from its payload"
bin="$tmp/bin" stubs="$tmp/stubs"
mkdir -p "$bin" "$stubs"
# c03-test.sh's stubs: the GPU, numactl, pip, the download tools, the engine, the HF headers and download; the gate the
# payload's own at a 300 GB host
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# c04-test.sh's GPU: an RTX PRO 6000's 97887 MiB, no NUMA node for GPU0
case " $* " in
  *" topo "*) printf '\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t0-11\tN/A\t\tN/A\n' ;;
  *" --query-gpu=memory.total --format=csv,noheader,nounits "*) echo 97887 ;;
  *" --query-gpu=memory.used,memory.total --format=csv,noheader "*) echo "1024 MiB, 97887 MiB" ;;
  *) echo "nvidia-smi (c04-test.sh stub): GPU 0, 97887 MiB" ;;
esac
EOF
cat > "$bin/numactl" <<'EOF'
#!/usr/bin/env bash
while [[ "${1:-}" == --* ]]; do shift; done
exec "$@"
EOF
printf '#!/usr/bin/env bash\necho "pip (c04-test.sh stub, nothing installed): $*"\n' > "$bin/pip"
for t in aria2c apt-get; do
  printf '#!/usr/bin/env bash\necho "%s (c04-test.sh stub): nothing downloads or installs here: $*" >&2\nexit 1\n' "$t" > "$bin/$t"
done
cat > "$stubs/dsv41-engine" <<'EOF'
#!/usr/bin/env bash
# c04-test.sh's engine: --dry-run exits 0, --serve is fakes.py's engine (the tokenizer from --model)
case " $* " in *" --dry-run "*) echo "dsv41-engine (c04-test.sh stub): dry run of $*"; exit 0 ;; esac
args=()
for a in "$@"; do [[ "$a" == --serve ]] || args+=("$a"); done
exec python3 "$(dirname "$0")/src/tools/dsv41/quality/fakes.py" --serve "${args[@]}"
EOF
cat > "$stubs/hf_model.py" <<'EOF'
#!/usr/bin/env python3
"""c04-test.sh's hf_model.py: `headers MANIFEST DIR` copies the real heads (C04_HEADERS/head-N.bin) into DIR."""
import os
import shutil
import sys

if len(sys.argv) != 4 or sys.argv[1] != "headers":
    sys.exit(f"hf_model.py (c04-test.sh stub): headers MANIFEST DIR only, not {sys.argv[1:]}")
os.makedirs(sys.argv[3], exist_ok=True)
for i in range(1, 8):
    shutil.copy(os.path.join(os.environ["C04_HEADERS"], f"head-{i}.bin"), sys.argv[3])
EOF
cat > "$stubs/hf-download.sh" <<'EOF'
#!/usr/bin/env bash
# c04-test.sh's download: the sparse shards (C04_MODEL) copied into DST, the record the sheet reads, marked a stub
set -euo pipefail
mkdir -p "$2"
cp --sparse=always "$C04_MODEL"/*.gguf "$2"/
python3 -c "import json, sys; json.dump({'stub': 'c04-test.sh: the sparse shards copied, no download', 'repo': 'c04-test-stub',
  'revision': 'none', 'files': 7, 'bytes': 0, 'download_s': 0, 'gb_per_s': 0, 'check_s': 0, 'ok': True}, open(sys.argv[1], 'w'))" "$3"
EOF
cat > "$stubs/e07_gate.py" <<'EOF'
#!/usr/bin/env python3
"""c04-test.sh's gate: the payload's own e07_gate.py (e07_gate_real.py) at a 300 GB host."""
import os
import sys

a = sys.argv[1:]
for key, value in (("--cgroup-bytes", "300000000000"), ("--mem-total-kib", "300000000")):
    a[a.index(key) + 1] = value
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e07_gate_real.py"), *a])
EOF
chmod +x "$bin"/* "$stubs"/*

for plan in clong cbatch-clong; do
  rc=0; E07_PLAN=$plan OUT="$tmp/dry-$plan" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry-$plan.txt" 2>&1 || rc=$?
  grep -E "^plan \((volume|hf)\): total|^price limit|^GPUs, in order" "$tmp/dry-$plan.txt" | sed "s/^/  $plan: /" | cut -c1-200 || true
  check "the $plan dry run exits 0 (exit $rc)" '(( rc == 0 ))' || tail -n 10 "$tmp/dry-$plan.txt"
  check "$plan: the RTX PRO 6000 only, the machine qualification first" \
    'grep -q "^GPUs, in order: NVIDIA RTX PRO 6000 Blackwell Server Edition;NVIDIA RTX PRO 6000 Blackwell Workstation Edition; host RAM at least 256 GB" "$tmp/dry-$plan.txt" &&
     [[ "$(grep -A1 "^plan with the model from HF" "$tmp/dry-$plan.txt" | tail -n 1)" == "  create, SSH, CUDA probe, qualification"* ]]'
  sed -n "/^plan with the model from HF /,/^plan (hf)/p" "$tmp/dry-$plan.txt" | awk '/^  / { print $1 }' > "$tmp/planned-$plan.txt"
  case "$plan" in
    clong) want="create, payload env model-headers gate setup model-download dry-run engine-server load-long robust server-stop pull" ;;
    # no load-c<N> nor det-sampling in the combined plan (measured on 2026-10-14);
    # steady-c8 after steady-c4, reported
    *) want="create, payload env model-headers gate gate-off gate-long setup model-download dry-run engine-server-off load-short-off server-stop-off engine-server load-short steady-c4 steady-c8 server-stop engine-server-long load-long robust server-stop-long pull" ;;
  esac
  check "$plan: the HF plan's steps: $want" '[[ "$(tr "\n" " " < "$tmp/planned-$plan.txt")" == "$want " ]]'
  # the HF path's pod steps: the plan's words less create, payload, env and pull (the session's own steps)
  n_pod="$(( $(wc -w <<<"$want") - 4 ))"
  check "$plan: every pod step of both model paths with its timeout, 2 x its estimate" \
    '(( $(sed -n "/^plan with the model from /,/^plan (hf)/p" "$tmp/dry-$plan.txt" | grep -c "; timeout min(") == 2 * n_pod + 1 )) &&
     grep -qE "^  load-long \(chat_load.py long: the 64K prompt alone 25 s, then 4 users and the 65565-token prompt beside them in 5 chunks of 16384 at a planned 4.65 s, 1024 tokens each at 20 tok/s, \+ 0.2\) +2.2  \(total [0-9.]+; timeout min\(4.4 min, the time left\)\)$" "$tmp/dry-$plan.txt" &&
     grep -qE "^  robust \(robust_check.py: item 13.s eight checks over HTTP, no-max-tokens and slot-room with them\) +4  \(total [0-9.]+; timeout min\(8 min, the time left\)\)$" "$tmp/dry-$plan.txt" &&
     { [[ "$plan" == clong ]] || grep -qE "^  load-short-off \(chat_load.py, 1 user, a 1K prompt, 4 turns of 256 at 35.4 tok/s, \+ 0.2\) +" "$tmp/dry-$plan.txt"; }'
  usd="$(sed -n "s/^plan (hf): total .* (about \([0-9.]*\) USD at 2.60 USD\/h)$/\1/p" "$tmp/dry-$plan.txt")"
  echo "$usd" > "$tmp/usd-$plan"
  check "$plan: the HF plan's estimate under 3.00 USD at 2.60 USD/h (${usd:-none} USD), the session cap 3.00 over at most 3 pods" \
    '[[ -n "$usd" ]] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 3.0 else 1)" "$usd" &&
     grep -q "^price limit: 2.60 USD/h; session cap: 3.00 USD over at most 3 pods" "$tmp/dry-$plan.txt"'
  # the latest end, each step's whole timeout after the estimates before it, within the
  # pod's deadline on both model paths
  grep -A1 -E "^plan \((volume|hf)\) latest end: " "$tmp/dry-$plan.txt" | sed "s/^/  $plan: /" | cut -c1-200 || true
  check "$plan: the latest end on both model paths: every step's whole timeout within the pod's deadline" \
    '(( $(grep -cE "^plan \((volume|hf)\) latest end: [a-z0-9-]+, started at minute [0-9.]+ after the estimates before it and run to its whole timeout, ends at minute [0-9.]+, the latest of any step,$" "$tmp/dry-$plan.txt") == 2 )) &&
     (( $(grep -cE "^  of the pod.s [0-9]+-minute deadline: every step covered; the estimates. [0-9.]+ min about [0-9.]+ USD, the deadline.s at most [0-9.]+ USD, of the 3.00 USD cap$" "$tmp/dry-$plan.txt") == 2 ))'
  check "$plan: the gate with the long prompt's slots, the payload's sources (the engine HEAD's, the 64K conversation HEAD's build)" \
    'grep -qE "^ +e07_gate.py: batch: 8 live slots of 69632 positions" "$tmp/dry-$plan.txt" &&
     grep -q "the tarball.s engine is HEAD.s$" "$tmp/dry-$plan.txt" &&
     grep -q "src/tools/dsv41/quality/robust_check.py: last changed in " "$tmp/dry-$plan.txt" &&
     grep -q "^  e07/conversations-long.json: .*, chat_load.py build --users 4 --long-context 65536 at HEAD$" "$tmp/dry-$plan.txt"'
  # the second cbatch server (CBATCH_SPARE=4) last, its gate after
  # gate-long; the plan's estimates within the cap but steady-c4-spare's whole timeout past the pod's deadline, so the
  # dry run refuses it (CBATCH_SPARE defaults to 0); its rehearsal below still runs it
  if [[ "$plan" == cbatch-clong ]]; then
    rc=0; CBATCH_SPARE=4 E07_PLAN=$plan OUT="$tmp/dry-spare" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry-spare.txt" 2>&1 || rc=$?
    grep -E "^plan \(hf\)|^  of the pod|^e07.sh: E07_PLAN|the spare slots. cost" "$tmp/dry-spare.txt" | sed "s/^/  CBATCH_SPARE=4: /" | cut -c1-200 || true
    sed -n "/^plan with the model from HF /,/^plan (hf)/p" "$tmp/dry-spare.txt" | awk '/^  / { print $1 }' > "$tmp/planned-$plan.txt"
    want="${want/gate-long setup/gate-long gate-spare setup}"
    want="${want/server-stop-long pull/server-stop-long engine-server-spare load-short-spare steady-c4-spare server-stop-spare pull}"
    check "CBATCH_SPARE=4: the second cbatch server last, its gate after gate-long ($want), its slab cost per GPU; steady-c4-spare's whole timeout past the pod's deadline (and, with steady-c8 since its addition, the HF plan past 3.00 USD): refused (exit $rc)" \
      '(( rc == 2 )) && [[ "$(tr "\n" " " < "$tmp/planned-$plan.txt")" == "$want " ]] &&
       grep -qE "^      the spare slots. cost: --hot-spare 4 against 2, 0.9[56] GiB more of slab, [0-9]+ slab experts against [0-9]+ \(-[0-9]+\)$" "$tmp/dry-spare.txt" &&
       grep -qE "^plan \(hf\): total [0-9.]+ min of the cap 69 min at the limit price \(about [0-9.]+ USD at 2.60 USD/h\)$" "$tmp/dry-spare.txt" &&
       grep -q "^e07.sh: E07_PLAN=cbatch-clong.s hf plan: steady-c4-spare.s whole timeout ends at minute [0-9.]*, past the pod.s deadline$" "$tmp/dry-spare.txt"'
  fi

  # c03-test.sh's steady window of 4 s for the combined plan's steady-c4: the fake decodes ~105 tok/s a user
  spare=0
  [[ "$plan" != cbatch-clong ]] || spare=4
  rc=0; CBATCH_SPARE=$spare CBATCH_STEADY_WINDOW_S=4 E07_PLAN=$plan OUT="$tmp/payload-$plan" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --payload-only \
    > "$tmp/payload-$plan.log" 2>&1 || rc=$?
  check "e07.sh --payload-only builds the $plan payload" '(( rc == 0 )) && [[ -f "$tmp/payload-$plan/payload-cu130.tar.gz" ]]' ||
    tail -n 5 "$tmp/payload-$plan.log"
  s="$tmp/s-$plan"
  mkdir -p "$s/pod/out" "$s/root/root"
  tar -xzf "$tmp/payload-$plan/payload-cu130.tar.gz" -C "$s/pod" --exclude='libcublas*' --exclude='libggml-cuda.so*'
  rm -rf -- "$tmp/payload-$plan"
  cp "$stubs/dsv41-engine" "$stubs/hf_model.py" "$stubs/hf-download.sh" "$s/pod/"
  mv "$s/pod/src/tools/dsv41/pod/e07_gate.py" "$s/pod/src/tools/dsv41/pod/e07_gate_real.py"
  cp "$stubs/e07_gate.py" "$s/pod/src/tools/dsv41/pod/"
  free_port port
  ( cd "$s/pod" && export PATH="$bin:$PATH" E07_POD_ROOT="$s/root" E07_Q_PORT="$port" C04_MODEL="$tmp/model" \
      C04_HEADERS="$HEADERS" POD_MACHINE="c04-$plan" FAKE_ENGINE_PROMPT_MS_PER_TOKEN="$P_MS" \
      FAKE_ENGINE_DECODE_MS_PER_TOKEN="$D_MS" FAKE_ENGINE_BATCH_ROW_MS="$R_MS"
    rc=0; DEADLINE=$(( $(date +%s) + 3000 )) POD_VOLUME= timeout 3300 bash e07-run.sh > out/e07-run.out 2>&1 < /dev/null || rc=$?
    echo "$rc" > out/e07-run.exit ) &
  jobs_+=($!)
  wait "${jobs_[@]}"
  jobs_=()
  o="$s/pod/out"
  awk '/^step / { sub(":$", "", $2); print $2 }' "$o/steps.log" > "$tmp/ran-$plan.txt"
  echo "  $plan: $(tr '\n' ' ' < "$tmp/ran-$plan.txt")"
  grep -vxE 'create,|payload|pull' "$tmp/planned-$plan.txt" > "$tmp/planned-pod-$plan.txt" || true
  check "$plan: e07-run.sh exits 0, every step exit 0" \
    '[[ "$(cat "$o/e07-run.exit")" == 0 ]] && ! grep "^step " "$o/steps.log" | grep -qv ": exit 0, "'
  check "$plan: steps.log has every planned step in order, then done" \
    'diff "$tmp/planned-pod-$plan.txt" "$tmp/ran-$plan.txt" > /dev/null && [[ "$(tail -n 1 "$o/steps.log")" == done ]]'
  long_gate="$o/gate/gate.json"
  [[ "$plan" == clong ]] || long_gate="$o/gate-long/gate.json"
  check "$plan: the long prompt's server as planned: its gate's placement, --batch 8 --batch-ctx 69632 --checkpoints 8 --log-ids" \
    'python3 -c "import json, sys; a = json.load(open(sys.argv[1]))[\"args\"]; g = json.load(open(sys.argv[2]))
v = lambda k: a[a.index(k) + 1]
sys.exit(0 if v(\"--batch\") == \"8\" and v(\"--batch-ctx\") == \"69632\" and v(\"--checkpoints\") == \"8\" and \"--log-ids\" in a and
         v(\"--vram-gb\") == str(g[\"vram_gb\"]) and g[\"batch\"] == 8 and g[\"batch_ctx\"] == 69632 else 1)" "$o/quality-server.json" "$long_gate"'
  [[ "$plan" != cbatch-clong ]] ||
    check "$plan: the second cbatch server as planned: gate-spare's placement with --hot-spare 4, gate's batch, --log-ids; its logs as out/spare-*, clong's back in place" \
      'python3 -c "import json, sys; a = json.load(open(sys.argv[1]))[\"args\"]; g, g0 = (json.load(open(f)) for f in sys.argv[2:4])
v = lambda k: a[a.index(k) + 1]
sys.exit(0 if v(\"--hot-spare\") == \"4\" and v(\"--batch\") == \"8\" and v(\"--batch-ctx\") == \"69632\" and \"--log-ids\" in a and
         v(\"--vram-gb\") == str(g[\"vram_gb\"]) and v(\"--hot-set\") == \"out/gate-spare/hot-set.json\" and g[\"hot_spare\"] == 4 and
         g0[\"hot_spare\"] == 2 and g[\"vram_experts\"] < g0[\"vram_experts\"] else 1)" "$o/spare-quality-server.json" "$o/gate-spare/gate.json" "$o/gate/gate.json" &&
       [[ -s "$o/spare-quality-engine.log" && -s "$o/cbatch/steady-c4-spare.json" && -s "$o/cbatch/load-short-spare.json" ]] &&
       ! grep -q -- "--hot-spare" "$o/quality-server.json"'
  check "$plan: load-long and robust came back, item 12's gap and item 13 PASS on the fake" \
    'python3 -c "import json, sys; a, b = (json.load(open(f)) for f in sys.argv[1:3])
sys.exit(0 if a[\"ok\"] and a[\"summary\"][\"gap_pass\"] and b[\"ok\"] else 1)" "$o/clong/load-long.json" "$o/clong/robust.json"'
  f="$tmp/report-$plan/20261015T090000Z"
  mkdir -p "$f"
  cp -r "$o" "$f/out"
  echo 3.00 > "$f/session-cap"
  echo "pod c04stub: 3000 s at 2.6 USD/h = 2.167 USD (upper bound: billing rounds up)" > "$f/cost.txt"
  echo "{\"id\": \"c04stub\", \"machineId\": \"c04-$plan\"}" > "$f/pod.json"   # the sheet's machine (pod.json's)
  # the qualification's H2D the chunk note cites (the session pod's, then another pod's)
  printf '%s\n' "=== pod c04other machine c04-other (NVIDIA RTX PRO 6000 Blackwell Server Edition, SECURE) at 2026-10-15T08:58:00 UTC" \
    "dsv41_machine_probe: measure h2d_gbs 56.7 bound >=18 ok (1 GiB pinned, median of 5 after a warm-up)" \
    "=== pod c04stub machine c04-$plan (NVIDIA RTX PRO 6000 Blackwell Workstation Edition, SECURE) at 2026-10-15T09:00:00 UTC" \
    "dsv41_machine_probe: measure h2d_gbs 28.6 bound >=18 ok (1 GiB pinned, median of 5 after a warm-up)" > "$f/probe.log"
  python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/sheets" > "$tmp/report-$plan.log" 2>&1 || true
  r="$tmp/sheets/2026-10-15-$plan/README.md"
  check "the $plan sheet: the cost against its cap, item 12's table and verdict, item 13's checks, the interleaved reads, the long gate" \
    'grep -q "^Build: Strata .*Session: 2.167 USD over 1 pod, the session.s cap 3.00 USD: within it. " "$r" &&
     grep -q "^## Item 12: no user frozen by a long prompt (logs/clong/load-long.json)$" "$r" &&
     grep -qE "^- item 12: PASS: the largest gap between two tokens of any of the 4 users [0-9.]+ s" "$r" &&
     grep -q "^## Item 13: served and robust (logs/clong/robust.json)$" "$r" &&
     grep -q "^Item 13: PASS (8 of 8 checks; P = 8 slots, queue_max 8).$" "$r" &&
     grep -q "^Not checked on the real engine here: the per-request deadline" "$r" &&
     grep -q "^A chunk.s read time is the host.s: " "$r" &&
     grep -qE "^1 request ran alone past a batch slot \(C04 audit 2" "$r" &&
     grep -qE "^\| [0-9]+ \| 4 \| 65565 \| 5 \| 16384 \| 4 \| [0-9.]+ \| [0-9.]+ \| [0-9.]+ \| [0-9.]+ \|$" "$r" &&
     grep -q "^| request | beside slots | tokens read | chunks | chunk | steps between | chunk read ms (mean) | chunk read ms (longest) | ms to its token | longest hold ms (the slots. gap) |$" "$r" &&
     grep -qE "^## Placement: gate(-long)? \(the long prompt.s slots" "$r" && grep -q "^## Machine qualification" "$r" &&
     grep -q "^## Host homes held against the checkpoints" "$r" &&
     grep -qE "^\| logs/gate(-long)? \| 8 \| 6.289 \| 8 \| [0-9]+ of [0-9]+ \([0-9.]+%\) \| " "$r" &&
     grep -qE "^A chunk.s read time is the host.s: [0-9]+ prompts? read beside decoding slots in this session \(logs/quality-engine.log; .*, machine c04-$plan, [^)]*\), [0-9]+ of them with a full 16K chunk \(requests? [0-9, ]+: the users. own 16K prompts and the long ones alike\); the chunks of the long prompts \(requests? [0-9, ]+\) took [0-9.]+ s on average \(the last partial chunk included, as in the references\); the longest chunk .*[0-9.]+ s \([0-9.]+ ms\), request [0-9]+.s [0-9]+-token read beside [0-9]+ slots?; the host.s H2D 28.6 GB/s \(pinned, the machine qualification table, pod c04stub[,)].*, against 56.7 GB/s on c04-other \(NVIDIA RTX PRO 6000 Blackwell Server Edition, pod c04other\) qualified the same session; for reference, " "$r"' ||
    tail -n 5 "$tmp/report-$plan.log"
  if [[ "$plan" == cbatch-clong ]]; then
    check "the combined sheet keeps item 11 apart: its one verdict line and steady state from the batch server's own logs (out/cbatch-*), the homes of all three gates, no load-c4 nor det-sampling" \
      'grep -q "^# C03 and C04: batched decode, then a long prompt" "$r" && grep -q "^## Item 11$" "$r" &&
       (( $(grep -c "^- item 11" "$r") == 3 )) && grep -q "^## Item 11.s steady state (logs/cbatch/steady-c4.json)$" "$r" &&
       grep -qE "^- item 11: (PASS|FAIL): .*; the baseline from 4 requests, load-short-off\)\. User-visible, the prompt reads included \(end to end, this code\): steady-c4.s own replies end to end \(3 repeats; .*\) = [0-9.]+x the baseline; the users. first tokens at [0-9./ -]+ s \(in their order, lowest to highest over the repeats\)\. More users: 8 users decoding at once \(steady-c8.json, reported, not a bar: 3 repeats of a 4 s window with exactly 8 rows a step\) [0-9.]+ tok/s in aggregate = [0-9.]+x the baseline, per user [0-9.]+ \(median\), the slowest [0-9.]+$" "$r" &&
       grep -q "^## The steady state with 8 users, reported (logs/cbatch/steady-c8.json)$" "$r" &&
       grep -qE "^- the baseline \(the single user with batching off\): [0-9.]+ tok/s here, machine c04-$plan \(" "$r" &&
       (( $(grep -c "^- item 11: " "$r") == 1 )) &&
       grep -qE "^- item 11 with 4 spare slab slots per layer: (PASS|FAIL): 4 users decoding at once \(steady-c4-spare: 3 repeats " "$r" &&
       grep -qE "^- item 11.s verdict over both servers \(amendment 1.s rule: the default passes -> its verdict; only the variant passes -> it becomes the deepseek-v4.1 profile.s default and item 11 is judged on it, its single user re-checked on it\): (PASS on the default config|PASS with 4 spare slots only|FAIL on both servers)" "$r" &&
       grep -q "^|  | default config | 4 spare slots per layer |$" "$r" && grep -q "^| spare slab slots per layer (--hot-spare) | 2 | 4 |$" "$r" &&
       grep -qE "^\| item 11 on this server \| (PASS|FAIL) \| (PASS|FAIL) \|$" "$r" &&
       grep -q "^## Item 11.s steady state with 4 spare slots (logs/cbatch/steady-c4-spare.json)$" "$r" &&
       grep -q "^## The batch with more spare slots in the engine.s log (logs/spare-quality-engine.log)$" "$r" &&
       grep -q "^## Placement: gate-spare (the batch, more spare slots) (logs/gate-spare/gate.json)$" "$r" &&
       (( $(grep -cE "^\| logs/gate(-off|-long|-spare)? \| 8 \| " "$r") == 4 )) &&
       grep -q "^## The batch in the engine.s log (logs/cbatch-quality-engine.log)" "$r" &&
       grep -q "^### Where a batch step.s time goes$" "$r" &&
       grep -qE "^\| 1 \| all \| [0-9]+ \| [0-9.]+ \| " "$r" &&
       grep -q "^Not in this plan: measured on 2026-10-14 (strata-engine/2026-10-14-cbatch), decision-c04-long-prefill-robustness amendment 2.$" "$r" &&
       [[ ! -e "$o/det-sampling.json" && ! -e "$o/cbatch/load-c4.json" ]] &&
       [[ -s "$o/cbatch-quality-engine.log" && -s "$o/quality-engine.log" ]] &&
       (( $(grep -n "^## Item 11$" "$r" | cut -d: -f1) < $(grep -n "^## Item 12" "$r" | cut -d: -f1) ))'
    # the plan as it runs (CBATCH_SPARE 0), the same session without the
    # second server's steps and files
    f2="$tmp/report-$plan-nospare/20261015T090000Z"
    mkdir -p "$f2"
    cp -r "$f/." "$f2"
    rm -rf "$f2/out/gate-spare" "$f2"/out/spare-* "$f2"/out/cbatch/*-spare.json
    sed -i "/spare/d" "$f2/out/steps.log"
    python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f2" "$tmp/sheets-nospare" > "$tmp/report-$plan-nospare.log" 2>&1 || true
    r2="$tmp/sheets-nospare/2026-10-15-$plan/README.md"
    check "the combined sheet without the second server: item 11 says the 4-spare-slot variant was not run and why, hot_spare off, no side-by-side table" \
      'grep -qE "^- item 11 with 4 spare slab slots per layer: not run in this plan, \"hot_spare\" stays off \(absent: the engine.s 2 spare slots per layer\)\. hot_policy_sim\.py batch predicts no gain .* past the pod.s 67-minute deadline \(the dry run.s latest end\), so the dry run refuses it\.$" "$r2" &&
       (( $(grep -c "^- item 11" "$r2") == 2 )) && ! grep -q "spare slots per layer |$" "$r2" &&
       ! grep -q "^## Item 11.s steady state with 4 spare slots" "$r2"' ||
      tail -n 5 "$tmp/report-$plan-nospare.log"
  fi
  if (( failed )); then
    echo "  the tail of e07-run.out and steps.log:"
    tail -n 5 "$o/e07-run.out" "$o/steps.log" 2>/dev/null | sed 's/^/    /' || true
  fi
done
echo "the dry runs' HF plans: clong about $(cat "$tmp/usd-clong") USD, cbatch-clong about $(cat "$tmp/usd-cbatch-clong") USD at 2.60 USD/h"

echo "--- 6. pod_server.sh stop frees the port before the next server"
# clong session 3 (2026-10-15): server-stop's SIGKILL at 60 s, then engine-server-long died in 6 s on
# "port 8090 is already in use". The stand-in server ignores SIGTERM and listens on the port, and a process in its own
# session keeps the listening socket HOLD_S seconds after the server is gone (a holder the SIGKILL of the server's
# session does not reach). The working tree's pod_server.sh, from a payload directory whose src/ is this checkout and
# whose engine is section 5's stub (fakes.py)
cat > "$stubs/stubborn.py" <<'EOF'
#!/usr/bin/env python3
"""c04-test.sh's server that ignores SIGTERM: python3 stubborn.py PORT HOLD_S HOLDER_PIDF listens on PORT; a forked
child in a new session (its pid in HOLDER_PIDF) holds the same listening socket until HOLD_S seconds after the server
is gone."""
import os
import signal
import socket
import sys
import time

port, hold_s, holder_pidf = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port))
s.listen()
parent = os.getpid()
if os.fork() == 0:
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    with open(holder_pidf, "w") as f:
        f.write(f"{os.getpid()}\n")
    while os.getppid() == parent:   # reparented once the server is gone
        time.sleep(0.1)
    time.sleep(hold_s)
    os._exit(0)
print("stubborn server: listening", flush=True)
while True:
    time.sleep(1)
EOF
p6="$tmp/s-port/pod" ps="$STRATA_DIR/tools/dsv41/quality/pod_server.sh"
mkdir -p "$p6/out"
ln -s "$STRATA_DIR" "$p6/src"
cp "$stubs/dsv41-engine" "$p6/"
free_port port6
stubborn() {   # stubborn HOLD_S: the stand-in server on port6, its pid and port recorded as pod_server.sh records its own
  rm -f "$p6/out/holder.pid"
  (cd "$p6" && exec setsid python3 "$stubs/stubborn.py" "$port6" "$1" out/holder.pid) > "$p6/out/stubborn.log" 2>&1 < /dev/null &
  echo $! > "$p6/out/quality-server.pid"
  disown   # its SIGKILL is the point: no job notice; cleanup ends it through the pid file
  echo "$port6" > "$p6/out/quality-server.port"
  for _ in $(seq 50); do [[ -s "$p6/out/holder.pid" ]] && grep -q listening "$p6/out/stubborn.log" && break; sleep 0.1; done
}
stubborn 4
rc=0; ( cd "$p6" && POD_SERVER_TERM_S=2 POD_SERVER_PORT_FREE_S=15 bash "$ps" stop ) > "$tmp/stop6a.log" 2>&1 || rc=$?
sed 's/^/  stop: /' "$tmp/stop6a.log"
check "a server that ignores SIGTERM: SIGKILL after the TERM wait, then the stop waits until its holder lets the port go (exit $rc)" \
  '(( rc == 0 )) && grep -qx "pod_server.sh: SIGKILL after 2 s" "$tmp/stop6a.log" &&
   n="$(sed -n "s/^port $port6 free after \([0-9]*\) s more$/\1/p" "$tmp/stop6a.log")" && [[ -n "$n" ]] && (( n >= 3 )) &&
   grep -q "^server [0-9]* stopped$" "$tmp/stop6a.log" && ! kill -0 "$(cat "$p6/out/holder.pid")" 2>/dev/null'
rc=0; ( cd "$p6" && timeout 120 bash "$ps" engine "$SHARD1" "$port6" "--log-ids" ) > "$tmp/start6.log" 2>&1 || rc=$?
check "the next server starts on the same port right after that stop (exit $rc)" \
  '(( rc == 0 )) && grep -qx "serve/server.py ready on $port6" "$tmp/start6.log"' ||
  tail -n 5 "$tmp/start6.log" "$p6/out/quality-server.log"
rc=0; ( cd "$p6" && bash "$ps" stop ) > "$tmp/stop6b.log" 2>&1 || rc=$?
check "that server's own stop: the port free at once (exit $rc)" \
  '(( rc == 0 )) && grep -qx "port $port6 free" "$tmp/stop6b.log" && grep -q "^server [0-9]* stopped$" "$tmp/stop6b.log"'
stubborn 600
holder="$(cat "$p6/out/holder.pid")"
rc=0; ( cd "$p6" && POD_SERVER_TERM_S=1 POD_SERVER_PORT_FREE_S=3 bash "$ps" stop ) > "$tmp/stop6c.log" 2>&1 || rc=$?
sed 's/^/  stop: /' "$tmp/stop6c.log"
check "a port that never frees: exit 1 naming the holder's pid and command line, no \"stopped\" line (exit $rc)" \
  '(( rc == 1 )) && grep -qxF "pod_server.sh: FAIL: port $port6 is in use, held by pid $holder (python3 $stubs/stubborn.py $port6 600 out/holder.pid) 3 s after the stop'"'"'s last signal, the next server cannot start on port $port6" "$tmp/stop6c.log" &&
   ! grep -q "stopped$" "$tmp/stop6c.log"'
kill "$holder" 2>/dev/null || true

echo "--- 7. pod/sheet_banners.py on a temporary sheet root"
python3 "$STRATA_DIR/tools/dsv41/pod/sheet_banners_test.py" -v > "$tmp/sheet-banners.out" 2> "$tmp/sheet-banners.log" || true
grep -E "^test_.* \.\.\. |^Ran " "$tmp/sheet-banners.log" | sed 's/^/  /' | cut -c1-200 || true
check "sheet_banners_test.py: every banner under its sheet's title between the markers, its links to existing sheets, the hand-written banner gone, VERDICTS.md with items 9-14; a second run changes no byte (--check 0 after, 1 before); an edited banner replaced, not stacked; a missing sheet or a current sheet without its verdict's figures exit 2, nothing written" \
  'grep -q "^Ran 4 tests in " "$tmp/sheet-banners.log" && [[ "$(tail -n 1 "$tmp/sheet-banners.log")" == OK ]]'

echo "c04-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
