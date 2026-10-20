#!/usr/bin/env bash
# The batch determinism checks without a GPU: no model, no RunPod, no HF, no money.
#   1. in the cu130 builder image (tools/dsv41/build.sh's; no GPU): dsv41_engine_expert_bits_test (the CPU experts in
#      the GPU's numerics, mmvq_ref.hpp's bits on every check, and the fork's numerics differing, its negative control;
#      then the speed table of both CPU numerics on this host), dsv41_engine_batch_test on tiny-b (the CPU reference
#      with the device's tiers: checks 9 to 11, batched requests in the deterministic mode their alone run's bit for
#      bit under other placements, a request joining mid-batch, a prompt read beside, and the negative controls; the
#      promotion fix: check 12, a request stopped on the solo path after 1, 2 and 5 tokens and promoted into slot 0
#      with 3 beside, its alone run's bit for bit, and check 13's prefill(T=1) against decode lines),
#      dsv41_engine_session_test (the CMake target), and dsv41-engine --serve --dry-run --batch 4 --log-ids on tiny-b:
#      a slot in the deterministic mode ends rep=1 beside a det=0 one that ends rep=0, each slot's slot_token_ids line
#      its admission token and its BT ids; a GEN of 2 tokens then a BGEN of its prompt and both (the promotion's
#      shape): the engine's line "its 1 token read was fed by decode", a live resume of the prompt and the first
#      token; --cpu-numerics fork: the load line and its deterministic slot rep=0, and a request that ends at its
#      admission (max_new 1) read beside a decoding slot BADM rep=0 (rep=1 in the GPU's numerics), one with nothing
#      beside it rep=1; the fork's numerics are the engine's default, so every run here that checks the GPU's says
#      --cpu-numerics gpu (cfinal-test.sh checks the default)
#   1b. tools/dsv41/engine/session_test.sh (the host compiler): the Session's unit test passes on the source and fails
#      with the fix taken out (the promotions read by prefill(T=1))
#   1c. tools/dsv41/engine/hot_policy_test.sh (the host compiler): the planner's unit test passes on the source and
#      fails with the fix taken out on its promoted batch alone (a batch opened by a promoted request: no swap, the
#      static slab's misses a row, the 2026-10-19 cfinal session's frozen hot set)
#   2. the server's tests (in the builder image, as verify.sh runs them): a batched reply's "strata_reproducible"
#      true in the deterministic mode and false off (the request's "strata_deterministic": false, or an engine started
#      with --deterministic off); an engine ERR on the solo path or an admission a 400 at once
#   3. quality/det_batch_check.py against serve/server.py on fakes.py's engine with --batch 8, --log-ids and
#      --cpu-numerics gpu: PASS (item 14's three phases, every chain printed, the promoted ones marked and fed by
#      decode, every reply's flag true and shown equal); FAIL with FAKE_ENGINE_PROMOTION_PREFILL=3 (the engine before
#      the promotion fix: only the promoted chains apart from alone, their replies' true a dishonest flag), FAIL with
#      FAKE_ENGINE_BATCH_DRIFT=3 (the engine before batch determinism: a slot's token moves once it decoded beside
#      another), FAIL with batching off (no phase batched), FAIL with the mode off (replies false); the per-reply flag
#      rule on made-up phases
#   4. pod/e07.sh --dry-run with E07_PLAN=cdet, NETWORK_VOLUME=testvolume01 (a placeholder: nothing is created) and
#      MAX_SESSION_USD C05_CDET_USD (default 2.90), cdet's defaults otherwise: exit 0, the RTX PRO 6000 only, the
#      network volume's offer only (read only; the HF plan shown as not offered), the qualification first, the planned
#      steps (cbatch's less load-c<N> and the batching-off server: load-short, ab-gpu, steady-c4, det-batch,
#      det-sampling, quality-venv, continuation, the three lm-eval tasks, server-stop, then the A/B's fork server,
#      ab-fork and its stop, between volume-before and volume-after), each with its timeout, the estimates measured in
#      the 2026-10-17 cdet session, the volume plan under the cap at 2.60 USD/h, its latest end within the pod's
#      deadline and >= 2 min before it (HumanEval's whole timeout inside), steady-c4 2 repeats, the payload's sources
#      (the tarball's engine HEAD's, det_batch_check.py, run_lm_eval.sh); with HF_CLOUDS set the same dry run exits 2
#      (the HF plan past the deadline); then the cdet rehearsal on the volume's path: e07-run.sh from the payload with
#      c04-test.sh's stubs and fakes.py as the engine, lm-eval for real from QUALITY_VENV offline on 3 items per task:
#      every planned step exit 0, then done, the batched server with --cpu-numerics gpu and the fork server with
#      --cpu-numerics fork, det-batch PASS, det-sampling's file, out/quality's three tasks at 4 concurrent requests,
#      e07_report.py's cdet sheet (item 14 with every chain, items 7 and 8, the speed bars and the A/B, item 3
#      batched, item 4 against the single-user engine run and the API fresh run reported)
#   5. pod/devgpu.sh --dry-run with DEVGPU_PLAN=c05: exit 0, the batch determinism steps, the plan under 1.50 USD at
#      0.70 USD/h, the probe's bounds and the qualification first, the 40 GB/s host read floor and why; and with
#      DEVGPU_PLAN=c05b: the promotion's parity steps under 1.50 USD
#   6. pod/sheet_banners_test.py: items 11 and 14 on the final batch determinism sheet, a banner on every earlier one
# Exits 1 if any check fails. c04-test.sh, c03-test.sh, c02-test.sh, q03-test.sh, the serve suite and verify.sh are
# run on their own.
#
#   BUILD_TAG=-c04 bash tools/dsv41/quality/c05-test.sh
#
# BUILD_ROOT (default $HOME/dsv41-build) holds the build (cu130${BUILD_TAG}/strata), the tiny models (engine/e01), the
# vocabulary, the real headers (engine/headers), the conversations, the tarballs (dist${BUILD_TAG}) the dry runs and
# the payload read, and the lm-eval venv (QUALITY_VENV, default quality/venv). TMPDIR on disk is advised (/tmp is a
# tmpfs; the rehearsal copies the sparse shards).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
# e07.sh has no default network volume: its dry runs and rehearsals here plan one by this placeholder id
export NETWORK_VOLUME="${NETWORK_VOLUME-testvolume01}"
HEADERS="${HEADERS:-$BUILD_ROOT/engine/headers}"
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"
E01="${E01:-$BUILD_ROOT/engine/e01}"
COUNTS="${COUNTS:-$BUILD_ROOT/counts.json}"
QUALITY_VENV="${QUALITY_VENV:-$BUILD_ROOT/quality/venv}"
STRATA_BUILD="$BUILD_ROOT/cu130${BUILD_TAG:-}/strata"
for f in "$HEADERS/head-7.bin" "$VOCAB" "$E01/tiny-b.gguf" "$STRATA_BUILD/dsv41_engine_batch_test" \
         "$STRATA_BUILD/dsv41_engine_expert_bits_test" "$COUNTS" "$QUALITY_VENV/bin/lm_eval"; do
  [[ -e "$f" ]] || { echo "c05-test.sh: no $f" >&2; exit 2; }
done
# the fake engine's delays: c04-test.sh's (a prompt token 0.28 ms, 5 ms a step alone and 1.5 ms per further row)
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
  # C05_KEEP_TMP=1 keeps the test's files (to look at a failed check)
  [[ -n "${C05_KEEP_TMP:-}" ]] && echo "c05-test.sh: kept $tmp" || rm -rf -- "${tmp:?}"
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
docker image inspect "$image" > /dev/null 2>&1 || { echo "c05-test.sh: no $image (tools/dsv41/build.sh)" >&2; exit 2; }
d="$tmp/docker"
mkdir -p "$d"

echo "--- 1. the expert bits, the batch test and the serve dry run's rep=, in the cu130 builder image"
# slot 0 in the deterministic mode (the default), slot 1 with det=0, both decoding together; stdin's end drains them
# (in the GPU's numerics, --cpu-numerics gpu, where a slot beside others ends rep=1)
printf 'BGEN 0 8 10,11,12,13\nBGEN 1 8 det=0 20,21,22\n' > "$d/rep.txt"
# the promotion's shape (the dry run's tokens are the last one read plus one: GEN 2 of 10,11,12 gives 13
# and 14, the live state 10 .. 13, 14 emitted and not fed), then the BGEN of its prompt and both tokens
printf 'GEN 2 10,11,12\nBGEN 0 6 10,11,12,13,14\n' > "$d/promo.txt"
# requests that end at their admission (max_new 1), slot 3 with nothing beside it, slots 1 and 2
# read beside decoding slot 0; with --cpu-numerics fork only slot 3's BADM may say rep=1, with the GPU's all do
printf 'BGEN 3 1 40,41\nBGEN 0 8 10,11,12,13\nBGEN 1 1 20,21,22\nBGEN 2 1 30,31\n' > "$d/badm.txt"
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e STRATA_BUILD="$STRATA_BUILD" \
  -e E01="$E01" -e D="$d" -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$BUILD_ROOT:$BUILD_ROOT:ro" -v "$tmp:$tmp" "$image" \
  bash -euo pipefail -c '
    mkdir -p /tmp/cudastub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1
    export LD_LIBRARY_PATH=/tmp/cudastub
    "$STRATA_BUILD/dsv41_engine_expert_bits_test" > "$D/expert-bits.txt" 2>&1 || true
    "$STRATA_BUILD/dsv41_engine_batch_test" "$E01/tiny-b.gguf" > "$D/batch-test.txt" 2>&1 || true
    "$STRATA_BUILD/dsv41_engine_session_test" > "$D/session-test.txt" 2>&1 || true
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
      --log-ids --cpu-numerics gpu --dry-run < "$D/rep.txt" > "$D/rep.out" 2> "$D/rep.err" || true
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
      --log-ids --cpu-numerics gpu --dry-run < "$D/promo.txt" > "$D/promo.out" 2> "$D/promo.err" || true
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
      --cpu-numerics fork --dry-run < "$D/rep.txt" > "$D/fork.out" 2> "$D/fork.err" || true
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
      --cpu-numerics gpu --dry-run < "$D/badm.txt" > "$D/badm-gpu.out" 2> "$D/badm-gpu.err" || true
    "$STRATA_BUILD/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
      --cpu-numerics fork --dry-run < "$D/badm.txt" > "$D/badm-fork.out" 2> "$D/badm-fork.err" || true' \
  > "$tmp/docker.log" 2>&1 || rc=$?
check "the builder image ran the tests (exit $rc)" \
  '(( rc == 0 )) && [[ -s "$d/expert-bits.txt" && -s "$d/batch-test.txt" && -s "$d/rep.out" ]]' || tail -n 20 "$tmp/docker.log"
grep -E "^(ok|FAIL) " "$d/expert-bits.txt" | sed 's/^/  /' | cut -c1-220 || true
check "dsv41_engine_expert_bits_test: PASS, every check (exp and the clamp, q8_1 also on the --gpu test's inputs against qref_quantize, the matvecs at the real gate/up and down shapes to 8 activations, the FFN at 1 and 3 threads, the weighted sum under 16 placements) and the fork's numerics differing (its negative control)" \
  '[[ "$(tail -n 1 "$d/expert-bits.txt")" == "dsv41_engine_expert_bits_test: PASS" ]] && ! grep -q "^FAIL " "$d/expert-bits.txt" &&
   (( $(grep -c "^ok   " "$d/expert-bits.txt") >= 7 )) &&
   grep -qE "^ok   q8_1 on the --gpu test.s inputs .*: mmvq_cpu::quantize 0 differ from mmvq_ref_quantize, 0 from qref_quantize$" "$d/expert-bits.txt" &&
   (( $(grep -cE "^ok   matvec (gate/up q2_K \[2304\]\[5120\]|down q3_K \[5120\]\[2304\]), 1 to 8 activations at once: [0-9]+ values against the reference, 0 differ$" "$d/expert-bits.txt") == 2 )) &&
   grep -qE "^ok   the weighted sum of [0-9]+ tokens under 16 placements .*: 0 values differ from the all-hit sum; with the fork.s CPU numerics for the misses [1-9][0-9]* differ$" "$d/expert-bits.txt"'
grep -A8 "^CPU experts. speed on this host" "$d/expert-bits.txt" | sed 's/^/  /' | cut -c1-220 || true
grep -E "^(ok|FAIL|info) " "$d/batch-test.txt" | tail -n 12 | sed 's/^/  /' | cut -c1-220 || true
check "dsv41_engine_batch_test on tiny-b: PASS, check 9 (a read beside moving the slab: reproducible), checks 10 and 11 (identical and different requests batched under moved placements, each its alone run's) and their negative controls" \
  '[[ "$(tail -n 1 "$d/batch-test.txt")" == "dsv41_engine_batch_test: PASS" ]] && ! grep -q "^FAIL " "$d/batch-test.txt" &&
   grep -q "^ok   a lone slot with a max_new 1 request read beside it after 3 steps (and the slab moved to the odd experts from then on): marked reproducible" "$d/batch-test.txt" &&
   (( $(grep -cE "^ok   4 identical requests \(30 prompt tokens, 12 generated\) in a batch on (the odd experts in the slab|a slab drawn anew at every call): 4 of 4 their alone run.s bit for bit, 4 marked reproducible" "$d/batch-test.txt") == 2 )) &&
   grep -q "^ok   4 different requests (one sampled, one joining mid-batch beside 3 decoding slots, a one-token request read beside them) under a slab drawn anew at every call: " "$d/batch-test.txt" &&
   grep -qE "^ok   the same with the fork.s CPU numerics for the misses \(the default\): [1-4] of 4 requests differ from their alone runs \(seen\)$" "$d/batch-test.txt" &&
   grep -q "^ok   the request read beside the slots in the mode off (its prefill split from the slab as it is): differs from its alone run (seen), marked not reproducible$" "$d/batch-test.txt"'
python3 - "$d/rep.out" "$d/rep.err" > "$d/rep-check.txt" 2>&1 <<'EOF' || true
import re, sys
out, err = (open(f).read().splitlines() for f in sys.argv[1:3])
bt, done, first = {}, {}, {}
for x in out:
    f = x.split()
    if f and f[0] == "BT":
        bt.setdefault(int(f[1]), []).append(int(f[2]))
    elif f and f[0] == "BDONE":
        done[int(f[1])] = f[-1]
    elif f and f[0] == "T" and len(first) < 2:
        first[len(first)] = int(f[1])
slot = {int(m.group(1)): [int(t) for t in m.group(2).split()]
        for m in (re.match(r"^dsv41-engine serve: request (\d+): slot_token_ids: (.*)$", x) for x in err) if m}
ok = done == {0: "rep=1", 1: "rep=0"} and len(slot) == 2
# each slot's line: its admission's token (the request's T line) then its BT ids, in the order the slots end
ok = ok and sorted(v[1:] for v in slot.values()) == sorted(bt.values()) and all(len(v) == 8 for v in slot.values())
print(f"rep: BDONE {done}, BT per slot {dict((k, len(v)) for k, v in bt.items())}, slot_token_ids {slot}: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
sed 's/^/  /' "$d/rep-check.txt" | cut -c1-240
check "batch_test check 12: a request stopped on the solo path after 1, 2 and 5 tokens, promoted into slot 0 with 3 beside, its alone run's bit for bit and reproducible; check 13's prefill(T=1) against decode lines" \
  '(( $(grep -cE "^ok   a request stopped on the solo path after [125] tokens and promoted into slot 0 with its prompt and tokens, 3 requests beside it: every token and every call.s logits from its first decode on its alone run.s bit for bit, marked reproducible$" "$d/batch-test.txt") == 3 )) &&
   (( $(grep -cE "^info the token after [125] \(position [0-9]+\): read by prefill\(T=1\) against decode, [0-9]+ of [0-9]+ logits differ" "$d/batch-test.txt") == 3 ))'
check "dsv41_engine_session_test (the CMake target): PASS, the promotions fed by decode" \
  '[[ "$(tail -n 1 "$d/session-test.txt")" == "dsv41_engine_session_test: PASS" ]] && ! grep -q "^FAIL " "$d/session-test.txt" &&
   (( $(grep -cE "^ok   promoted after [125] \(the BGEN of its prompt and tokens\): live resume of [0-9]+, its 1 token left fed by decode [0-9]+, no prefill call" "$d/session-test.txt") == 3 ))'
check "the serve dry run, the promotion's shape: GEN 2 then BGEN of its prompt and both tokens: a live resume of 4, 1 read, the engine's line that it was fed by decode, the slot's tokens on from 15" \
  'grep -qE "^dsv41-engine serve: request 2: 5 prompt tokens, live resume of 4, 1 read in " "$d/promo.err" &&
   grep -q "^dsv41-engine serve: request 2: its 1 token read was fed by decode: the token the last request emitted and did not feed, as the request alone feeds it$" "$d/promo.err" &&
   ! grep -q "^dsv41-engine serve: request 1: its 1 token read" "$d/promo.err" &&
   grep -q "^BADM 0 1$" "$d/promo.out" && [[ "$(grep "^BT 0 " "$d/promo.out" | head -n 1)" == "BT 0 16" ]] &&
   grep -qE "^BDONE 0 6 length [0-9.]+ rep=1$" "$d/promo.out"'
check "--cpu-numerics fork: the load line, the deterministic slot rep=0 beside the det=0 one, the slot line saying why" \
  'grep -q "^dsv41-engine serve: the CPU experts in the fork.s numerics (--cpu-numerics fork, the default): a routed expert.s bits follow its tier, so no batch slot ends reproducible$" "$d/fork.err" &&
   grep -qE "^BDONE 0 8 length [0-9.]+ rep=0$" "$d/fork.out" && grep -qE "^BDONE 1 8 length [0-9.]+ rep=0$" "$d/fork.out" &&
   grep -qE "^dsv41-engine serve: request [0-9]+: batch slot 0: 8 generated \(length\) in .*; not reproducible \(--cpu-numerics fork: a routed expert.s bits follow its tier, so its neighbours. placement\)$" "$d/fork.err"'
check "--cpu-numerics fork, a request that ends at its admission: BADM rep=0 for the two read beside decoding slot 0 (rep=1 in the GPU.s numerics), rep=1 for the one with nothing beside it, the read beside line saying why" \
  'grep -qx "BADM 3 0 rep=1" "$d/badm-fork.out" && grep -qx "BADM 1 0 rep=0" "$d/badm-fork.out" &&
   grep -qx "BADM 2 0 rep=0" "$d/badm-fork.out" && ! grep -q "^BADM .* rep=1" <(grep -v "^BADM 3 " "$d/badm-fork.out") &&
   grep -qx "BADM 3 0 rep=1" "$d/badm-gpu.out" && grep -qx "BADM 1 0 rep=1" "$d/badm-gpu.out" &&
   grep -qx "BADM 2 0 rep=1" "$d/badm-gpu.out" &&
   (( $(grep -cE "^dsv41-engine serve: request [0-9]+: read beside 1 decoding batch slots, .*; not reproducible \(--cpu-numerics fork: its bits follow the placement\)$" "$d/badm-fork.err") == 2 ))'
echo "--- 1b. tools/dsv41/engine/session_test.sh (the host compiler, with and without the fix)"
rc=0; bash "$STRATA_DIR/tools/dsv41/engine/session_test.sh" > "$tmp/session-sh.txt" 2>&1 || rc=$?
grep -E "^(  without the fix: FAIL|session_test.sh: )" "$tmp/session-sh.txt" | sed 's/^/  /' | cut -c1-220 || true
check "session_test.sh: PASS on the source, the three promotions FAIL with the fix taken out (prefill 1 seen) (exit $rc)" \
  '(( rc == 0 )) && grep -qx "session_test.sh: PASS (the test passes on the source and fails with the promotion.s token read by prefill)" "$tmp/session-sh.txt"'
echo "--- 1c. tools/dsv41/engine/hot_policy_test.sh (the host compiler, with and without the fix)"
rc=0; bash "$STRATA_DIR/tools/dsv41/engine/hot_policy_test.sh" > "$tmp/hot-policy-sh.txt" 2>&1 || rc=$?
grep -E "^(ok   promoted batch|  without the fix: FAIL|hot_policy_test.sh: )" "$tmp/hot-policy-sh.txt" | sed 's/^/  /' | cut -c1-220 || true
check "hot_policy_test.sh: PASS on the source, the promoted batch alone FAIL with the fix taken out (0 swaps) (exit $rc)" \
  '(( rc == 0 )) && grep -qx "hot_policy_test.sh: PASS (the test passes on the source and fails with the promoted request.s prompt left open)" "$tmp/hot-policy-sh.txt"'
check "the serve dry run: the deterministic slot rep=1 beside the det=0 one (rep=0), each slot's slot_token_ids its BT ids after its first token (--log-ids)" \
  'grep -q ": PASS$" "$d/rep-check.txt" &&
   grep -qE "^dsv41-engine serve: request [0-9]+: batch slot 0: 8 generated \(length\) in .*, [1-9][0-9]* of them beside other slots .*; reproducible \(the deterministic mode" "$d/rep.err" &&
   grep -qE "^dsv41-engine serve: request [0-9]+: batch slot 1: 8 generated \(length\) in .*; not reproducible \(deterministic off\)$" "$d/rep.err"'

echo "--- 2. the server's batch determinism test (in the builder image)"
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e D="$d" \
  -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$d:$d" "$image" bash -euo pipefail -c '
    mkdir -p /tmp/serve-tests && cd /tmp/serve-tests
    export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STRATA_DIR"
    python3 -m unittest -v serve.test_parallel.ParallelService.test_a_batched_reply_says_whether_it_is_reproducible \
      serve.test_parallel.ParallelService.test_a_required_reproducible_reply_waits_and_decodes_alone \
      serve.test_parallel.ParallelService.test_an_engine_error_on_the_solo_path_answers_at_once \
      > "$D/serve-c05.log" 2>&1 || true' > "$tmp/docker-serve.log" 2>&1 || rc=$?
tail -n 3 "$d/serve-c05.log" | sed 's/^/  /' || true
check "serve/test_parallel.py: a batched reply true in the deterministic mode, false off (the request's or the engine's), \"required\" still alone, an engine ERR a 400 at once (docker exit $rc)" \
  '(( rc == 0 )) && grep -q "^Ran 3 tests in " "$d/serve-c05.log" && [[ "$(tail -n 1 "$d/serve-c05.log")" == OK ]]'

echo "--- 3. det_batch_check.py against serve/server.py on the fake engine (P ${P_MS}, D ${D_MS}, ${R_MS} ms a further row)"
serve_fake() {   # serve_fake NAME PARALLEL ENGINE_ARGS...: the server on fakes.py's engine with --batch-ctx 32768, --log-ids
  local name="$1" par="$2" dir="$tmp/$1" port
  shift 2
  mkdir -p "$dir"
  cat > "$dir/engine" <<EOF
#!/usr/bin/env bash
exec python3 "$here/fakes.py" --serve "\$@"
EOF
  chmod +x "$dir/engine"
  python3 -c "import json, sys; d = sys.argv[1]
print(json.dumps({'exe': d + '/engine', 'cwd': d, 'args': ['--model', sys.argv[2], '--ctx', '262144', '--checkpoints', '8',
  '--batch-ctx', '32768', '--log-ids', *sys.argv[4:]], 'model_profile': 'deepseek-v4.1', 'tokenizer_gguf': sys.argv[2],
  'log': d + '/engine.log', 'repeat_stop_tokens': 0, 'parallel': int(sys.argv[3])}))" "$dir" "$VOCAB" "$par" "$@" > "$dir/config.json"
  free_port port
  ( cd "$STRATA_DIR" && FAKE_ENGINE_PROMPT_MS_PER_TOKEN=$P_MS FAKE_ENGINE_DECODE_MS_PER_TOKEN=$D_MS \
      FAKE_ENGINE_BATCH_ROW_MS=$R_MS exec setsid python3 -m serve.server --engine strata --config "$dir/config.json" \
      --port "$port" ) > "$dir/server.log" 2>&1 < /dev/null &
  echo $! > "$tmp/server-$name.pid"
  for _ in $(seq 120); do grep -q "ready: http" "$dir/server.log" && break; sleep 1; done
  echo "$port" > "$dir/port"
}
for case in "pass|8||0|" "promo|8||1|3" "drift|8|3|1|" "alone|1||1|" "off|8||1|"; do
  IFS='|' read -r name par drift want promo <<<"$case"
  # the GPU's numerics (the fork's, the default, end no slot beside others reproducible)
  extra=(--cpu-numerics gpu)
  [[ "$name" != off ]] || extra+=(--deterministic off)
  FAKE_ENGINE_BATCH_DRIFT="$drift" FAKE_ENGINE_PROMOTION_PREFILL="$promo" serve_fake "dbc-$name" "$par" "${extra[@]}"
  rc=0
  timeout 600 python3 "$here/det_batch_check.py" --url "http://127.0.0.1:$(cat "$tmp/dbc-$name/port")" \
    --engine-log "$tmp/dbc-$name/engine.log" --out "$tmp/dbc-$name/det-batch.json" > "$tmp/dbc-$name/out.txt" 2>&1 || rc=$?
  echo "$rc" > "$tmp/dbc-$name/rc"
  sed "s/^/  $name: /" "$tmp/dbc-$name/out.txt" | cut -c1-240
  kill -TERM -- "-$(cat "$tmp/server-dbc-$name.pid")" 2>/dev/null || true
done
check "det_batch_check.py on the fake's batch slots: identical, stream and different PASS, item 14 PASS (exit 0), every chain printed, a promoted chain fed by decode, every batched reply true and shown equal" \
  '[[ "$(cat "$tmp/dbc-pass/rc")" == 0 ]] && grep -qx "det_batch_check.py: item 14: PASS" "$tmp/dbc-pass/out.txt" &&
   (( $(grep -cE "^det_batch_check.py: (identical|stream|different): PASS: 4 chains \([0-4] promoted\), 4 equal to alone$" "$tmp/dbc-pass/out.txt") == 3 )) &&
   (( $(grep -cE "^  chain .*: PASS$" "$tmp/dbc-pass/out.txt") == 12 )) &&
   python3 -c "import json, sys; j = json.load(open(sys.argv[1])); b = [r for r in j[\"replies\"] if r[\"phase\"] != \"alone\"]
cs = [c for p in j[\"phases\"] for c in p[\"chains\"]]
fl = [r[\"flag\"] for p in j[\"phases\"] for r in p[\"replies\"]]
sys.exit(0 if j[\"item14\"][\"verdict\"] == \"PASS\" and len(j[\"replies\"]) == 17 and len(b) == 12 and
         all(r[\"strata_reproducible\"] is True for r in b) and fl == [\"true, shown equal\"] * 12 and
         any(c[\"promoted\"] and c[\"fed_by_decode\"] for c in cs) else 1)" "$tmp/dbc-pass/det-batch.json"'
check "the engine before the promotion fix (FAKE_ENGINE_PROMOTION_PREFILL=3): item 14 FAIL (exit 1), only promoted chains apart from alone, their replies' true named a dishonest flag" \
  '[[ "$(cat "$tmp/dbc-promo/rc")" == 1 ]] && grep -qx "det_batch_check.py: item 14: FAIL" "$tmp/dbc-promo/out.txt" &&
   grep -qE "^  chain promoted \(began on the solo path, engine requests \[[0-9, ]+\]\): [0-9]+ tokens, .*differs from the alone run at [0-9]+ .*: FAIL$" "$tmp/dbc-promo/out.txt" &&
   python3 -c "import json, sys; j = json.load(open(sys.argv[1])); cs = [c for p in j[\"phases\"] for c in p[\"chains\"]]
bad = [c for c in cs if c[\"divergence\"]]
sys.exit(0 if bad and all(c[\"promoted\"] and not c[\"fed_by_decode\"] for c in bad) and
         all(c[\"divergence\"] is None for c in cs if not c[\"promoted\"]) and
         any(\"a dishonest flag\" in x for x in j[\"item14\"][\"reasons\"]) else 1)" "$tmp/dbc-promo/det-batch.json"'
check "the engine before batch determinism (FAKE_ENGINE_BATCH_DRIFT=3): every phase FAIL naming the token where it differs (exit 1)" \
  '[[ "$(cat "$tmp/dbc-drift/rc")" == 1 ]] && grep -qx "det_batch_check.py: item 14: FAIL" "$tmp/dbc-drift/out.txt" &&
   (( $(grep -cE "^det_batch_check.py: (identical|stream|different): FAIL: .*the tokens differ from the alone run.s at [0-9]+ " "$tmp/dbc-drift/out.txt") == 3 ))'
check "batching off (parallel 1): every phase FAIL as not batched (exit 1)" \
  '[[ "$(cat "$tmp/dbc-alone/rc")" == 1 ]] &&
   (( $(grep -cE "^det_batch_check.py: (identical|stream|different): FAIL: .*not batched: 0 of 4 requests decoded beside other slots" "$tmp/dbc-alone/out.txt") == 3 ))'
check "the engine with --deterministic off: every phase FAIL, its requests not in the deterministic mode, its replies false (exit 1)" \
  '[[ "$(cat "$tmp/dbc-off/rc")" == 1 ]] &&
   (( $(grep -cE "^det_batch_check.py: (identical|stream|different): FAIL: .*an engine request was not in the deterministic mode" "$tmp/dbc-off/out.txt") == 3 )) &&
   ! grep -q "strata_reproducible true" "$tmp/dbc-off/out.txt"'
# det_batch_check's per-reply flag rule on made-up phases: each reply paired with a
# chain of its prompt by whether its text and the chain's tokens are the alone run's
python3 - "$here" > "$tmp/dbc-honesty.txt" 2>&1 <<'PY' || true
import sys
sys.path.insert(0, sys.argv[1])
from det_batch_check import judge
want = {"A": {"prompt_ids": [1, 2], "tokens": [5, 6, 7], "text": "x", "deterministic": True}}
def phase(rows):   # rows: (its chain's tokens, its reply's text, its reply's flag, its slot line's claim)
    reqs = [{"request": k, "prompt": [1, 2], "tokens": t, "slot": True, "beside_steps": 3, "deterministic": True,
             "fed_by_decode": False, "slot_reproducible": c} for k, (t, _, _, c) in enumerate(rows)]
    replies = [{"i": k, "phase": "identical", "prompt": "A", "status": 200, "text": x, "strata_reproducible": f}
               for k, (_, x, f, _) in enumerate(rows)]
    return judge("identical", replies, reqs, want)
same, apart = [5, 6, 7], [5, 9, 7]
cases = [("4 true, 4 equal", phase([(same, "x", True, True)] * 4), "PASS", False),
         ("4 false, 4 equal", phase([(same, "x", False, False)] * 4), "PASS", False),
         ("1 apart, its reply true", phase([(same, "x", True, True)] * 3 + [(apart, "y", True, False)]), "FAIL", True),
         ("1 apart, its reply false", phase([(same, "x", True, True)] * 3 + [(apart, "y", False, False)]), "FAIL", False),
         ("1 apart, its slot line reproducible", phase([(same, "x", True, True)] * 3 + [(apart, "y", False, True)]), "FAIL", True),
         ("1 apart, every text the alone run's", phase([(same, "x", True, True)] * 3 + [(apart, "x", True, True)]), "FAIL", True)]
ok = True
for name, got, verdict, dishonest in cases:
    said = any("a dishonest flag" in x for x in got["reasons"])
    print(f"{name}: {got['verdict']}, dishonest {said}: {got['reasons']}")
    ok = ok and got["verdict"] == verdict and said == dishonest
print("honesty:", "PASS" if ok else "FAIL")
PY
sed 's/^/  /' "$tmp/dbc-honesty.txt" | cut -c1-240
check "det_batch_check.py's flag rule per reply: true with a chain shown equal PASS, a true reply or slot line on a chain that differs FAIL as dishonest, a false one on it FAIL on the tokens only" \
  'grep -qx "honesty: PASS" "$tmp/dbc-honesty.txt"'

echo "--- 4. pod/e07.sh --dry-run with E07_PLAN=cdet MAX_SESSION_USD=2.90 (the volume only), and the cdet rehearsal from its payload"
bin="$tmp/bin" stubs="$tmp/stubs"
mkdir -p "$bin" "$stubs"
# c04-test.sh's stubs: the GPU, numactl, pip, the download tools, the engine; q02-test.sh's volume copy; the gate the
# payload's own at a 300 GB host
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# c05-test.sh's GPU: an RTX PRO 6000's 97887 MiB, no NUMA node for GPU0
case " $* " in
  *" topo "*) printf '\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t0-11\tN/A\t\tN/A\n' ;;
  *" --query-gpu=memory.total --format=csv,noheader,nounits "*) echo 97887 ;;
  *" --query-gpu=memory.used,memory.total --format=csv,noheader "*) echo "1024 MiB, 97887 MiB" ;;
  *) echo "nvidia-smi (c05-test.sh stub): GPU 0, 97887 MiB" ;;
esac
EOF
cat > "$bin/numactl" <<'EOF'
#!/usr/bin/env bash
while [[ "${1:-}" == --* ]]; do shift; done
exec "$@"
EOF
printf '#!/usr/bin/env bash\necho "pip (c05-test.sh stub, nothing installed): $*"\n' > "$bin/pip"
for t in aria2c apt-get; do
  printf '#!/usr/bin/env bash\necho "%s (c05-test.sh stub): nothing downloads or installs here: $*" >&2\nexit 1\n' "$t" > "$bin/$t"
done
cat > "$stubs/dsv41-engine" <<'EOF'
#!/usr/bin/env bash
# c05-test.sh's engine: --dry-run exits 0, --serve is fakes.py's engine (the tokenizer from --model)
case " $* " in *" --dry-run "*) echo "dsv41-engine (c05-test.sh stub): dry run of $*"; exit 0 ;; esac
args=()
for a in "$@"; do [[ "$a" == --serve ]] || args+=("$a"); done
exec python3 "$(dirname "$0")/src/tools/dsv41/quality/fakes.py" --serve "${args[@]}"
EOF
cat > "$stubs/model-copy.sh" <<'EOF'
#!/usr/bin/env bash
# c05-test.sh's copy (q02-test.sh's): model-copy.sh SRC DST, the shards copied sparse (SRC only read)
set -euo pipefail
mkdir -p "$2"
cp --sparse=always "$1"/*.gguf "$2"/
echo "model-copy.sh (c05-test.sh stub): $(ls "$2" | wc -l) shards from $1"
EOF
cat > "$stubs/e07_gate.py" <<'EOF'
#!/usr/bin/env python3
"""c05-test.sh's gate: the payload's own e07_gate.py (e07_gate_real.py) at a 300 GB host."""
import os
import sys

a = sys.argv[1:]
for key, value in (("--cgroup-bytes", "300000000000"), ("--mem-total-kib", "300000000")):
    a[a.index(key) + 1] = value
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e07_gate_real.py"), *a])
EOF
chmod +x "$bin"/* "$stubs"/*

plan=cdet
# the session's cap C05_CDET_USD
# (default 2.90), cdet's defaults otherwise: the network volume only
usd_cap="${C05_CDET_USD:-2.90}"
rc=0; E07_PLAN=$plan MAX_SESSION_USD="$usd_cap" OUT="$tmp/dry-$plan" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry-$plan.txt" 2>&1 || rc=$?
grep -E "^plan \((volume|hf)\): |^price limit|^GPUs, in order|^e07.sh: " "$tmp/dry-$plan.txt" | sed "s/^/  $plan: /" | cut -c1-200 || true
check "the $plan dry run at MAX_SESSION_USD $usd_cap exits 0 (exit $rc)" '(( rc == 0 ))' || tail -n 10 "$tmp/dry-$plan.txt"
check "$plan: the RTX PRO 6000 only, the network volume's offer only (read only), the HF plan not offered, the machine qualification first" \
  'grep -q "^GPUs, in order: NVIDIA RTX PRO 6000 Blackwell Server Edition;NVIDIA RTX PRO 6000 Blackwell Workstation Edition; host RAM at least 256 GB" "$tmp/dry-$plan.txt" &&
   [[ "$(grep -A1 "per GPU the offers in order:$" "$tmp/dry-$plan.txt" | tail -n 1)" == "  SECURE in EUR-IS-1 with network volume testvolume01 at /workspace (read only): the model copied from it" ]] &&
   [[ "$(grep -A2 "per GPU the offers in order:$" "$tmp/dry-$plan.txt" | tail -n 1)" == "machines excluded by BAD_MACHINES "* ]] &&
   [[ "$(grep -A3 "per GPU the offers in order:$" "$tmp/dry-$plan.txt" | tail -n 1)" == "price limit: "* ]] &&
   grep -qx "plan (hf): not offered (HF_CLOUDS is empty): shown, not checked" "$tmp/dry-$plan.txt" &&
   [[ "$(grep -A1 "^plan with the model from the volume" "$tmp/dry-$plan.txt" | tail -n 1)" == "  create, SSH, CUDA probe, qualification"* ]]'
sed -n "/^plan with the model from the volume /,/^plan (volume)/p" "$tmp/dry-$plan.txt" | awk '/^  / { print $1 }' > "$tmp/planned-$plan.txt"
want="create, payload env volume-before gate setup model-copy dry-run engine-server load-short ab-gpu steady-c4 det-batch det-sampling quality-venv continuation lm-eval-ifeval lm-eval-gsm8k_cot_llama lm-eval-humaneval_instruct server-stop engine-server-fork ab-fork server-stop-fork volume-after pull"
check "$plan: the volume plan's steps (no batching-off server, the A/B's arms): $want" '[[ "$(tr "\n" " " < "$tmp/planned-$plan.txt")" == "$want " ]]' ||
  echo "  planned: $(tr "\n" " " < "$tmp/planned-$plan.txt")"
check "$plan: every pod step of both model paths with its timeout, 2 x its estimate; det-batch, det-sampling, the lm-eval tasks and the servers' loads from measured numbers" \
  '(( $(sed -n "/^plan with the model from the volume /,/^plan (volume)/p" "$tmp/dry-$plan.txt" | grep -c "; timeout min(") == $(wc -w <<<"$want") - 4 )) &&
   ! sed -n "/^plan with the model from /,/^plan (hf)/p" "$tmp/dry-$plan.txt" | grep "^  [a-z]" | grep -v "; timeout min(" |
     grep -vE "^  (create, SSH|payload upload|env |pull and delete|of the pod)" &&
   grep -qE "^  det-batch \(det_batch_check.py: 5 requests alone, then 4 identical together twice, streamed and not, and 4 different together\) +1.7  \(total [0-9.]+; timeout min\(" "$tmp/dry-$plan.txt" &&
   grep -qE "^  det-sampling \(det_sampling_check.py; 155 s measured in the 2026-10-17 cdet session less the 300 s ERR wait fixed since\) +3  \(total " "$tmp/dry-$plan.txt" &&
   grep -qE "^  lm-eval-ifeval \(NUM_CONCURRENT 4 on the batch slots, 1042 s measured in the 2026-10-17 cdet session\) +17.4  \(total [0-9.]+; timeout min\(34.8 min, the time left\)\)$" "$tmp/dry-$plan.txt" &&
   grep -qE "^  lm-eval-gsm8k_cot_llama \(NUM_CONCURRENT 4 on the batch slots, 459 s measured in the 2026-10-17 cdet session\) +7.7  " "$tmp/dry-$plan.txt" &&
   grep -qE "^  lm-eval-humaneval_instruct \(NUM_CONCURRENT 4 on the batch slots, the reference run.s 359 s one at a time .*\) +6.0  \(total [0-9.]+; timeout min\(12.0 min" "$tmp/dry-$plan.txt" &&
   grep -qE "^  engine-server \(--batch 8 .*85-140 s measured in the cbatch sessions\) +2.5  \(total [0-9.]+; timeout min\(5.0 min" "$tmp/dry-$plan.txt"'
usd="$(sed -n "s/^plan (volume): total .* (about \([0-9.]*\) USD at 2.60 USD\/h)$/\1/p" "$tmp/dry-$plan.txt")"
check "$plan: the volume plan's estimate under $usd_cap USD at 2.60 USD/h (${usd:-none} USD), the session cap $usd_cap over at most 3 pods" \
  '[[ -n "$usd" ]] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$usd" "$usd_cap" &&
   grep -q "^price limit: 2.60 USD/h; session cap: $usd_cap USD over at most 3 pods" "$tmp/dry-$plan.txt"'
grep -A1 -E "^plan \((volume|hf)\) latest end: " "$tmp/dry-$plan.txt" | sed "s/^/  $plan: /" | cut -c1-200 || true
# steady-c4 2 repeats, the plan's latest end >= 2 min before the pod's deadline
vol_end="$(sed -n "s/^plan (volume) latest end: .*, ends at minute \([0-9.]*\), the latest of any step,$/\1/p" "$tmp/dry-$plan.txt")"
vol_dl="$(grep -A1 "^plan (volume) latest end: " "$tmp/dry-$plan.txt" | sed -n "s/^  of the pod.s \([0-9]*\)-minute deadline: .*/\1/p")"
check "$plan: steady-c4 2 repeats, the volume plan's latest end (minute ${vol_end:-none}) covered and >= 2 min before its ${vol_dl:-none}-minute deadline (HumanEval's whole timeout inside)" \
  '(( $(grep -cE "^  steady-c4 \(chat_load.py steady, 2 x: " "$tmp/dry-$plan.txt") == 2 )) && [[ -n "$vol_end" && -n "$vol_dl" ]] &&
   grep -A1 "^plan (volume) latest end: " "$tmp/dry-$plan.txt" | grep -q ": every step covered;" &&
   python3 -c "import sys; sys.exit(0 if float(sys.argv[2]) - float(sys.argv[1]) >= 2.0 else 1)" "$vol_end" "$vol_dl"'
# the HF offer refused at 2.90 when asked for (its plan past the pod's deadline)
rc=0; E07_PLAN=$plan MAX_SESSION_USD=2.90 HF_CLOUDS="SECURE COMMUNITY" OUT="$tmp/dry-$plan-hf" \
  bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry-$plan-hf.txt" 2>&1 || rc=$?
grep -A1 -E "^plan \(hf\) latest end: " "$tmp/dry-$plan-hf.txt" | sed "s/^/  $plan with HF_CLOUDS: /" | cut -c1-200 || true
check "$plan with HF_CLOUDS set at 2.90 USD: exit $rc (2), its HF plan's last step past the pod's deadline" \
  '(( rc == 2 )) && grep -q "^e07.sh: E07_PLAN=cdet.s hf plan: .* past the pod.s deadline$" "$tmp/dry-$plan-hf.txt"'
check "$plan: the payload's sources: the tarball's engine HEAD's, det_batch_check.py and run_lm_eval.sh, the quality runs' subsets" \
  'grep -q "the tarball.s engine is HEAD.s$" "$tmp/dry-$plan.txt" &&
   grep -q "src/tools/dsv41/quality/det_batch_check.py: last changed in " "$tmp/dry-$plan.txt" &&
   grep -q "src/tools/dsv41/quality/run_lm_eval.sh: last changed in " "$tmp/dry-$plan.txt" &&
   grep -q "its lm-eval subsets are src/tools/dsv41/quality/subsets_lean2h_seed20261006.json, the reference run.s" "$tmp/dry-$plan.txt"'

rc=0; CBATCH_STEADY_WINDOW_S=4 E07_PLAN=$plan MAX_SESSION_USD="$usd_cap" OUT="$tmp/payload-$plan" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --payload-only \
  > "$tmp/payload-$plan.log" 2>&1 || rc=$?
check "e07.sh --payload-only builds the $plan payload, e07/cdet-quality: 4 concurrent requests, ifeval first" \
  '(( rc == 0 )) && [[ -f "$tmp/payload-$plan/payload-cu130.tar.gz" ]] &&
   [[ "$(tar -xzOf "$tmp/payload-$plan/payload-cu130.tar.gz" e07/cdet-quality)" == "4 ifeval gsm8k_cot_llama humaneval_instruct" ]]' ||
  tail -n 5 "$tmp/payload-$plan.log"
s="$tmp/s-$plan"
mkdir -p "$s/pod/out" "$s/root/root"
tar -xzf "$tmp/payload-$plan/payload-cu130.tar.gz" -C "$s/pod" --exclude='libcublas*' --exclude='libggml-cuda.so*'
rm -rf -- "$tmp/payload-$plan"
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
    POD_MACHINE="c05-$plan" FAKE_ENGINE_PROMPT_MS_PER_TOKEN="$P_MS" \
    FAKE_ENGINE_DECODE_MS_PER_TOKEN="$D_MS" FAKE_ENGINE_BATCH_ROW_MS="$R_MS" SAMPLES="$tmp/samples3.json" \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  rc=0; DEADLINE=$(( $(date +%s) + 3000 )) POD_VOLUME=c05-rehearsal timeout 3300 bash e07-run.sh > out/e07-run.out 2>&1 < /dev/null || rc=$?
  echo "$rc" > out/e07-run.exit ) &
jobs_+=($!)
wait "${jobs_[@]}"
jobs_=()
echo "  the rehearsal took $(( $(date +%s) - t0 )) s"
o="$s/pod/out"
awk '/^step / { sub(":$", "", $2); print $2 }' "$o/steps.log" > "$tmp/ran-$plan.txt"
echo "  $plan: $(tr '\n' ' ' < "$tmp/ran-$plan.txt")"
grep -vxE 'create,|payload|pull' "$tmp/planned-$plan.txt" > "$tmp/planned-pod-$plan.txt" || true
check "$plan: e07-run.sh exits 0, every step exit 0" \
  '[[ "$(cat "$o/e07-run.exit")" == 0 ]] && ! grep "^step " "$o/steps.log" | grep -qv ": exit 0, "' ||
  { grep "^step " "$o/steps.log" | grep -v ": exit 0, " | sed 's/^/  /'; tail -n 5 "$o/e07-run.out" | sed 's/^/  /'; }
check "$plan: steps.log has every planned step in order, then done" \
  'diff "$tmp/planned-pod-$plan.txt" "$tmp/ran-$plan.txt" > /dev/null && [[ "$(tail -n 1 "$o/steps.log")" == done ]]'
check "$plan: the batched server as planned: gate's placement, --batch 8, --log-ids, the deterministic mode on (no --deterministic off)" \
  'python3 -c "import json, sys; a = json.load(open(sys.argv[1]))[\"args\"]; g = json.load(open(sys.argv[2]))
v = lambda k: a[a.index(k) + 1]
sys.exit(0 if v(\"--batch\") == \"8\" and \"--log-ids\" in a and \"--deterministic\" not in a and v(\"--vram-gb\") == str(g[\"vram_gb\"]) else 1)" "$o/quality-server.json" "$o/gate/gate.json"'
check "$plan: the speed A/B's fork arm on the same server with --cpu-numerics fork (its logs out/fork-*, the batched server's back as out/quality-*), the batched server with --cpu-numerics gpu (the fork's are the default), ab-gpu and ab-fork with the deterministic mode off" \
  'python3 -c "import json, sys; a = json.load(open(sys.argv[1]))[\"args\"]; b = json.load(open(sys.argv[2]))[\"args\"]
g, f = (json.load(open(x)) for x in sys.argv[3:5])
k = a.index(\"--cpu-numerics\") if \"--cpu-numerics\" in a else -2
sys.exit(0 if a[k + 1] == \"fork\" and b[k:k + 2] == [\"--cpu-numerics\", \"gpu\"] and a.count(\"--cpu-numerics\") == b.count(\"--cpu-numerics\") == 1 and
         a[:k + 1] + a[k + 2:] == b[:k + 1] + b[k + 2:] and g[\"deterministic\"] == f[\"deterministic\"] == \"off\"
         and g[\"ok\"] and f[\"ok\"] else 1)" "$o/fork-quality-server.json" "$o/quality-server.json" "$o/cbatch/ab-gpu.json" "$o/cbatch/ab-fork.json" &&
   grep -q "^dsv41-engine serve: the CPU experts in the fork.s numerics (--cpu-numerics fork" "$o/fork-quality-engine.log" &&
   grep -qE "^dsv41-engine serve: request [0-9]+: batch slot [0-9]+: " "$o/quality-engine.log"'
check "$plan: det-batch PASS on the fake (item 14), det-sampling's file, lm-eval's three tasks at NUM_CONCURRENT 4 on 3 items each" \
  'python3 -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))[\"item14\"][\"verdict\"] == \"PASS\" else 1)" "$o/det-batch.json" &&
   [[ -s "$o/det-sampling.json" ]] &&
   (for t in ifeval gsm8k_cot_llama humaneval_instruct; do ls "$o"/quality/lm_eval/"$t"/*/samples_"$t"_*.jsonl > /dev/null 2>&1 || exit 1; done) &&
   (( $(grep -cx "num_concurrent 4" "$o/quality/settings.txt") == 3 ))'
f="$tmp/report-$plan/20261016T090000Z"
mkdir -p "$f"
cp -r "$o" "$f/out"
echo 2.90 > "$f/session-cap"
echo "pod c05stub: 3000 s at 2.6 USD/h = 2.167 USD (upper bound: billing rounds up)" > "$f/cost.txt"
echo "{\"id\": \"c05stub\", \"machineId\": \"c05-$plan\"}" > "$f/pod.json"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/sheets" > "$tmp/report-$plan.log" 2>&1 || true
r="$tmp/sheets/2026-10-16-$plan/README.md"
check "the $plan sheet: item 14 PASS with every chain's row, items 7 and 8, the speed bars against session 4's 61.55 and the single user on the batched server, the speed A/B, item 3 batched, item 4 against the single-user engine run and the API fresh run reported, the batch's lines" \
  'grep -qE "^# C05: batched requests as alone ones, and their quality \(20261016[0-9TZ]*\)$" "$r" &&
   grep -q "^## Item 14 (logs/det-batch.json)$" "$r" && grep -q "^Batch determinism (item 14, det_batch_check.py): \*\*PASS\*\*.$" "$r" &&
   (( $(grep -cE "^\| (identical|stream|different) \| [0-9, ]+ \| (yes|no) \| (yes|no) \| [0-9]+ \| [0-9]+ \| none \| (reproducible|none) \| true, shown equal \| PASS \|$" "$r") == 12 )) &&
   grep -qE "^- steady-c4: (PASS|FAIL): [0-9.]+ tok/s in aggregate .* the single user on the batched server \(its solo path" "$r" &&
   grep -qE "^- the speed A/B \(the single user, the deterministic mode off, .*\): CPU experts in the GPU.s numerics [0-9.]+ tok/s, in the fork.s [0-9.]+ tok/s " "$r" &&
   grep -q "^## Server as run, the A/B.s fork arm (--cpu-numerics fork)$" "$r" &&
   grep -q "^## Items 7 and 8 (logs/det-sampling.json)$" "$r" &&
   grep -qE "^- the single user with batching on: (PASS|FAIL): [0-9.]+ tok/s against session 4.s 61.55 " "$r" &&
   grep -qE "^- steady-c4: (PASS|FAIL): [0-9.]+ tok/s in aggregate " "$r" &&
   grep -q "^## Item 3 batched (logs/quality)$" "$r" && grep -q "^## Item 4$" "$r" &&
   grep -q "^| engine batched - engine alone | n | alone | batched | diff | 95% CI | flips |$" "$r" &&
   grep -q "^### Against the API fresh run (reported, as in the reference run)$" "$r" &&
   grep -q "^## The batch in the engine.s log (logs/quality-engine.log)$" "$r" &&
   grep -qE "^[0-9]+ requests decoded in batch slots, [0-9]+ of them reproducible" "$r"' ||
  tail -n 5 "$tmp/report-$plan.log"
grep -A8 "^## Item 4" "$r" 2>/dev/null | sed 's/^/  /' | cut -c1-220 || true
if (( failed )); then
  echo "  the tail of e07-run.out and steps.log:"
  tail -n 5 "$o/e07-run.out" "$o/steps.log" 2>/dev/null | sed 's/^/    /' || true
fi

echo "--- 5. pod/devgpu.sh --dry-run with DEVGPU_PLAN=c05 and c05b"
rc=0; DEVGPU_PLAN=c05b bash "$STRATA_DIR/tools/dsv41/pod/devgpu.sh" --dry-run > "$tmp/devgpu-c05b.txt" 2>&1 || rc=$?
grep -E "^plan: total|^  batch-" "$tmp/devgpu-c05b.txt" | sed 's/^/  c05b: /' | cut -c1-200 || true
usd_b="$(sed -n "s/^plan: total [0-9.]* min, about \([0-9.]*\) USD at 0.70 USD\/h .*/\1/p" "$tmp/devgpu-c05b.txt")"
check "devgpu.sh c05b dry run: exit $rc, batch_test --gpu and --gpu-swaps with the promotion's parity (check 12) and the prefill(T=1) against decode lines (check 13), the forced e2e runs, the qualification first, ${usd_b:-none} USD at 0.70 USD/h (cap 1.50)" \
  '(( rc == 0 )) && [[ -n "$usd_b" ]] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 1.50 else 1)" "$usd_b" &&
   grep -qE "^  batch-e2e \(dsv41_engine_batch_test --gpu, tiny-a and tiny-b, no swaps: checks 1 to 4, check 12 the promotion after 1, 2 and 5 tokens into slot 0 with 3 beside, every call.s logits bytes against alone, check 13 prefill\(T=1\) against decode, info\)" "$tmp/devgpu-c05b.txt" &&
   grep -q "^  batch-swaps " "$tmp/devgpu-c05b.txt" && grep -q "^  e2e-b-reuse-split-forced" "$tmp/devgpu-c05b.txt" &&
   ! grep -q "expert-bits-gpu" "$tmp/devgpu-c05b.txt" &&
   grep -q "^qualification first (QUALIFY=1" "$tmp/devgpu-c05b.txt" && grep -q "^  host read floor 40 GB/s" "$tmp/devgpu-c05b.txt"' ||
  tail -n 10 "$tmp/devgpu-c05b.txt"
rc=0; DEVGPU_PLAN=c05 bash "$STRATA_DIR/tools/dsv41/pod/devgpu.sh" --dry-run > "$tmp/devgpu-c05.txt" 2>&1 || rc=$?
grep -E "USD|^plan|qualif" "$tmp/devgpu-c05.txt" | head -n 8 | sed 's/^/  /' | cut -c1-200 || true
check "devgpu.sh c05 dry run: exit $rc, the expert bits on the GPU and the CPU, experts_test --gpu, the batch's kernels and its end-to-end and swap runs, under 1.50 USD" \
  '(( rc == 0 )) && grep -q "expert-bits-gpu" "$tmp/devgpu-c05.txt" && grep -q "expert-bits-cpu" "$tmp/devgpu-c05.txt" &&
   grep -q "batch-swaps" "$tmp/devgpu-c05.txt" && grep -q "experts-test" "$tmp/devgpu-c05.txt"' || tail -n 10 "$tmp/devgpu-c05.txt"
check "devgpu.sh c05 dry run: the host read floor 40 GB/s and why (synthetic tensors, not the model)" \
  'grep -qE "^  host read floor 40 GB/s \(.*not the RTX sessions. 250 GB/s: synthetic tensors, not the model\)$" "$tmp/devgpu-c05.txt"' ||
  grep "floor" "$tmp/devgpu-c05.txt"

echo "--- 6. pod/sheet_banners_test.py (items 11 and 14 on the final batch determinism sheet)"
python3 "$STRATA_DIR/tools/dsv41/pod/sheet_banners_test.py" -v > "$tmp/sheet-banners.out" 2> "$tmp/sheet-banners.log" || true
check "sheet_banners_test.py: OK (items 11 and 14 on the final batch determinism sheet, every earlier one bannered)" \
  '[[ "$(tail -n 1 "$tmp/sheet-banners.log")" == OK ]]' || tail -n 5 "$tmp/sheet-banners.log"

echo "c05-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
