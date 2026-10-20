#!/usr/bin/env bash
# The checks of several cached conversations (the engine's checkpoints) without a GPU: no model, no RunPod, no HF,
# no money.
#   1. in the cu130 builder image (tools/dsv41/build.sh's; no GPU): dsv41_engine_checkpoint_test on tiny-b (the
#      Session's checkpoints on the CPU reference: 4 conversations alternating, each turn 2+ resumed from its own
#      checkpoint, the restored state and the outputs bit for bit the straight read's; a 5th conversation evicting the
#      least recently used; the in-place restore and its ring limit; a missing span seen), engine/serve_check.py
#      --reference dry (its checkpoint protocol checks among the rest: 4 conversations over --serve --dry-run, the eviction,
#      --checkpoints 1), and dsv41-engine --serve --dry-run on the real layout: its cost line's slot bytes are the
#      gate's
#   2. the deterministic mode with its reset after the request: dsv41_engine_experts_test --determinism (the A's bit for
#      bit, every reset at a request's end, none at its start; with host homes and the reset
#      layer by layer every request bit for bit the two-way run's, no layer computing with a slot not yet restored,
#      every home intact, and without the per-layer wait the stale slots seen), dsv41_engine_hot_policy_test (the host
#      homes' swaps the spare homes', no copy down, the in-place restore, HotRestore's order and its refusal), the
#      protocol's hot set line naming the idle reset and its wait (q03-test.sh's det_sampling rehearsal
#      in section 5 too); dsv41-engine --serve --dry-run on the real layout with --hot-homes all and spare: the dry
#      reset (every layer back at the slab, copies up only with homes, in rounds both ways without) and the homes' RAM,
#      the gate's; --determinism with partial host homes (some layers homed in full, one in part, the rest
#      none) bit for bit the two-way run's with victims copied down, hot_policy_test's partial homes (the spare homes'
#      swaps, homed victims dropped, the reset in rounds, restore_in_place refused), and --hot-homes 5612 (the gate's
#      count on the ccache pod) on the real layout: 37 of 40 layers homed in full, the dry reset in place there and in
#      rounds in the other 3, the homes' RAM the gate's; a count over the slab's refused by name
#   3. pod/e07_gate.py on the real headers: a 256 GB host holds the placement with the 4 checkpoints; at 209 GB the
#      placement fits only without them and is refused naming them, their slot size and every other part; with
#      --checkpoints 0 it passes there; the host homes on at 300 GB, off with --host-homes off;
#      the partial homes, the worked example: the ccache pod's 282 GB host (cgroup 262.63 GiB) cannot hold
#      all 5951 homes (3.97 GiB short with the 8 GiB margin) and gets 5612 (94.3%: 37 of 40 layers in full, layer 37's
#      last 116 of 142), 254.63 GiB + 8.0 margin; a 256 GB host 3552 (59.7%)
#   4. quality/chat_load.py run --alternate against serve/server.py on fakes.py's engine (its 4 checkpoints, 0.3 ms per
#      prompt token, 5 ms per token): 4 users taking 3 turns, every turn 2+ resumed (cached tokens), none queued, the
#      table with the turns 2+ columns, each request's reset wait from the engine's log (--engine-log); --think-s and
#      --engine-log refused without --alternate; then 5 users over the 4 checkpoints: the 5th evicts the least recently
#      used one (the engine's log), whose next turn is read cold
#   5. pod/e07.sh --dry-run with E07_PLAN=ccache: exit 0, the RTX PRO 6000 only, the qualification first, every step
#      with its timeout (2 x its estimate; server-stop planned at 60 s), the HF plan under 2.30 USD at 2.60 USD/h and
#      its worst case (no turn resumed) under it too, within the pod's 53 min (at the old
#      2.00 USD cap the dry run refuses the worst case), the payload's sources; the gate on the
#      ccache pod's real cgroup through PROBE_LOG (a session before the probe's cgroup line: its out/env.log; the
#      second run a probe.log with that line), the expected host homes printed: 5612 of 5951 (94.3%); then the ccache
#      rehearsal: e07-run.sh from the payload with c01-test.sh's stubs (the gate's host the ccache pod's: 282 GB, its
#      cgroup 262.63 GiB, so partial host homes) and
#      fakes.py as the engine, its idle reset 1.41 s of wall-clock time over 40 layers (FAKE_ENGINE_IDLE_RESET_MS,
#      the host homes' estimated reset for 2900 swaps at 0.49 ms; the reset layer by layer, the next request's read waiting
#      per layer; the 3 layers not homed in full in rounds the host waits for): every planned step exit 0, then done;
#      the engine with --hot-homes 5612; load-alt4 (the strict run, back to back) with every turn 2+ resumed, none
#      queued, each turn 2+ waiting for the reset and under 2 s with it; load-alt4-think5 (5 s between turns) the same
#      but waiting for none; load-c1/c2/c4,
#      det-sampling PASS (items 7 and 8 with the reset after each request), the engine with --log-ids, the server
#      stopped; pod/e07_report.py's ccache sheet (item 10, the before/after table against the cbase sheet, the cost per
#      conversation, the session's cost against its 2.30 USD cap, items 7 and 8; both alternating runs' turns 2+ TTFT
#      and reset wait, item 10's verdict on the strict run, here a PASS; the path taken, the gate's reason, the homes'
#      RAM, each request's full reset and its per-layer waits; the homes held of the slab's experts, the rounds and the
#      copies down left per request)
#   6. pod/runpod_api.py body (no key, no request): the cu130 flavor's 13.0,13.1,13.2 leaves allowedCudaVersions out
#      (REST v1 answers HTTP 400 above 13.0), 12.8,12.9,13.0 keeps it as given
# Exits 1 if any check fails. c01-test.sh, q03-test.sh, q02-test.sh, analyze_test.py and verify.sh are run on their own.
#
#   bash tools/dsv41/quality/c02-test.sh
#
# BUILD_ROOT (default $HOME/dsv41-build) holds the build (cu130/strata), the tiny models (engine/e01), the vocabulary,
# the real headers (engine/headers), the conversations and the tarballs the dry run and the payload read; COUNTS (default
# $BUILD_ROOT/counts.json) a real-model run's routing counts; CCACHE_BASELINE (default $BUILD_ROOT/baselines/cbase,
# the cbase sheet's logs/cbase) the before/after table's baseline runs.
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
CCACHE_BASELINE="${CCACHE_BASELINE:-$BUILD_ROOT/baselines/cbase}"
for f in "$HEADERS/head-7.bin" "$VOCAB" "$E01/tiny-b.gguf" "$BUILD_ROOT/cu130${BUILD_TAG:-}/strata/dsv41_engine_checkpoint_test" \
         "$CONVERSATIONS" "$COUNTS" "$CCACHE_BASELINE/load-c4.json"; do
  [[ -f "$f" ]] || { echo "c02-test.sh: no $f" >&2; exit 2; }
done
# the fake engine's delays, c01-test.sh's: a 16K prompt read in ~4.9 s, 200 tok/s of decode
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
python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" sparse "$HEADERS" "$tmp/model" > /dev/null
SHARD1="$tmp/model/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf"

source "$STRATA_DIR/tools/dsv41/flavors.sh"
flavor_config 13.0
image="$(builder_image)"
docker image inspect "$image" > /dev/null 2>&1 || { echo "c02-test.sh: no $image (tools/dsv41/build.sh)" >&2; exit 2; }
d="$tmp/docker"
mkdir -p "$d"
echo "--- 1 and 2. the engine's CPU and dry-run tests in the cu130 builder image"
# the protocol's hot set lines need a device; the dry run's checkpoints lines are what the serve loop prints
cat > "$d/alternate.txt" <<'EOF'
GEN 3 100,101,102,103,104,105,106,107,108,109
GEN 3 120,121,122,123,124,125,126,127,128,129,130
GEN 3 100,101,102,103,104,105,106,107,108,109,110,111,112,7
GEN 3 120,121,122,123,124,125,126,127,128,129,130,131,132,133,8
QUIT
EOF
# the gate's placement on an RTX PRO 6000 host whose RAM holds the host homes (section 3's at 300 GB), for the dry runs
python3 "$STRATA_DIR/tools/dsv41/pod/e07_gate.py" --vram-mib 97887 --prefetch-experts 128 --cgroup-bytes 300000000000 \
  --mem-total-kib 300000000 --counts "$COUNTS" --layout "$STRATA_DIR/tools/dsv41/oracle/layout-dsv41-q2k.json" \
  --model-headers "$SHARD1" --out "$d/gate-300" > "$d/gate-300.log" 2>&1 || true
rc=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e STRATA_DIR="$STRATA_DIR" -e BUILD_ROOT="$BUILD_ROOT" \
  -e BUILD_TAG="${BUILD_TAG:-}" \
  -e E01="$E01" -e VOCAB="$VOCAB" -e SHARD1="$SHARD1" -e D="$d" -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$BUILD_ROOT:$BUILD_ROOT:ro" \
  -v "$tmp:$tmp" "$image" bash -euo pipefail -c '
    strata="$BUILD_ROOT/cu130${BUILD_TAG:-}/strata"
    "$strata/dsv41_engine_checkpoint_test" "$E01/tiny-b.gguf" > "$D/checkpoint.txt" 2>&1 || true
    "$strata/dsv41_engine_experts_test" --determinism "$E01/tiny-b.gguf" > "$D/determinism.txt" 2>&1 || true
    "$strata/dsv41_engine_hot_policy_test" > "$D/hot-policy.txt" 2>&1 || true
    "$strata/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 8 --dry-run < "$D/alternate.txt" \
      > "$D/alternate.out" 2> "$D/alternate.err" || true
    "$strata/dsv41-engine" --serve --model "$SHARD1" --ctx 262144 --chunk 16384 --dry-run < /dev/null \
      > "$D/real.out" 2> "$D/real.err" || true
    for h in all spare 5612 5952; do
      "$strata/dsv41-engine" --serve --model "$SHARD1" --ctx 262144 --chunk 16384 --hot-set "$D/gate-300/hot-set.json" \
        --vram-gb 72 --ram-gb 190 --threads 16 --prefetch-experts 128 --hot-homes "$h" --dry-run < /dev/null \
        > "$D/homes-$h.out" 2> "$D/homes-$h.err" || true
    done
    "$strata/dsv41-engine" --serve --model "$SHARD1" --ctx 262144 --chunk 16384 --hot-set "$D/gate-300/hot-set.json" \
      --vram-gb 72 --ram-gb 190 --threads 16 --prefetch-experts 128 --hot-homes spare --deterministic off --dry-run \
      < /dev/null > "$D/homes-spare-off.out" 2> "$D/homes-spare-off.err" || true
    python3 "$STRATA_DIR/tools/dsv41/engine/tiny_vocab.py" "$VOCAB" /tmp/tiny-vocab.gguf > /dev/null
    python3 "$STRATA_DIR/tools/dsv41/engine/serve_check.py" --engine "$strata/dsv41-engine" --model "$E01/tiny-b.gguf" \
      --tokenizer-gguf /tmp/tiny-vocab.gguf --vocab "$VOCAB" --reference dry --out /tmp/serve-check > "$D/serve-check.log" 2>&1 || true' \
  > "$tmp/docker.log" 2>&1 || rc=$?
check "the builder image ran the tests (exit $rc)" '(( rc == 0 )) && [[ -s "$d/checkpoint.txt" && -s "$d/alternate.out" ]]' ||
  tail -n 20 "$tmp/docker.log"
grep -E "^(ok|FAIL) " "$d/checkpoint.txt" | sed 's/^/  /' | cut -c1-200 || true
check "dsv41_engine_checkpoint_test: 4 conversations alternating, every turn 2+ resumed from its own checkpoint, bit for bit the straight read" \
  'grep -qx "dsv41_engine_checkpoint_test: PASS" "$d/checkpoint.txt" &&
   (( $(grep -cE "^ok   user [0-3] turn [23]: [0-9]+ prompt tokens, prompt-end resume of [0-9]+ \(checkpoint [0-3]\), saved to [0-3], 1 superseded; against the straight read bit for bit$" "$d/checkpoint.txt") == 8 ))'
check "a 5th conversation evicts the least recently used checkpoint (user 0's), whose next turn is read cold bit for bit" \
  'grep -q "^ok   user 4 (a fifth conversation): cold, evicts checkpoint 0 (.* user 0.s, the least recently used), saved to 0, bit for bit$" "$d/checkpoint.txt" &&
   grep -q "^ok   user 0 turn 4 after the eviction: cold .*bit for bit as a cold read$" "$d/checkpoint.txt"'
check "the in-place restore at a chunk past the checkpoint, the host one a token further; one slot resumes nothing; every missing span seen" \
  'grep -q "^ok   16 tokens fed past the checkpoint: restored in place" "$d/checkpoint.txt" &&
   grep -q "^ok   17 tokens fed past the checkpoint: restored from the host" "$d/checkpoint.txt" &&
   grep -q "^ok   one slot, 2 users alternating: 4 of 4 turns cold" "$d/checkpoint.txt" &&
   (( $(grep -c "^ok   a restore without one row of the .*: seen: " "$d/checkpoint.txt") == 4 ))'
sed 's/^/  /' "$d/alternate.out" | grep -E "RESUME|DONE" | tr '\n' ' ' | cut -c1-200; echo
check "--serve --dry-run: two conversations alternating, each second turn RESUMEs its own prompt (10 and 11 tokens)" \
  '[[ "$(grep -E "^(RESUME|DONE)" "$d/alternate.out" | awk "{ print \$1 \" \" \$2 }" | tr "\n" ",")" == "DONE 3,DONE 3,RESUME 10,DONE 3,RESUME 11,DONE 3," ]] &&
   grep -q "^dsv41-engine serve: request 4: checkpoints: restored 1 (11 tokens, dry run), saved to 1 (15 tokens, dry run), 1 superseded; 2 of 4 slots held, 0 evictions$" "$d/alternate.err" &&
   grep -q "^INFO .*checkpoints=4 " "$d/alternate.out"'
grep -E "^ok   protocol(: (a 5th|conversation 0.s next)| \([14] checkpoints\): (INFO|conversation 1 turn 3))" "$d/serve-check.log" |
  sed 's/^/  /' | cut -c1-200 || true
check "engine/serve_check.py --reference dry: PASS, its checkpoints checks among them (4 resumed over 3 turns, the eviction, one slot)" \
  'grep -q "^serve_check.py: PASS (dry: profile, protocol, http)$" "$d/serve-check.log" &&
   (( $(grep -c "^ok   protocol (4 checkpoints): conversation [0-3] turn [23] resumes its own checkpoint" "$d/serve-check.log") == 8 )) &&
   grep -q "^ok   protocol: a 5th conversation is read cold and evicts conversation 0.s checkpoint" "$d/serve-check.log" &&
   (( $(grep -c "^ok   protocol (1 checkpoints): conversation [01] turn [1-3] read cold" "$d/serve-check.log") == 6 ))' ||
  tail -n 15 "$d/serve-check.log"
grep -E "^(ok|FAIL) |reset .* before it" "$d/determinism.txt" | sed 's/^/  /' | cut -c1-200 || true
check "the deterministic mode with the reset after the request: --determinism bit for bit in the mode, every reset after a request, none before" \
  'grep -qx "dsv41_engine_experts_test (d): PASS" "$d/determinism.txt" && (( $(grep -c "^ok   deterministic " "$d/determinism.txt") == 8 )) &&
   grep -q "^ok   deterministic on: every reset after its request (C02): none before one, each after one$" "$d/determinism.txt"' ||
  tail -n 20 "$d/determinism.txt"
check "host homes on the CPU reference: host homes and the reset layer by layer bit for bit the two-way runs, no layer before its restore, every home intact; the missing wait seen" \
  'grep -q "^ok   host homes, layer by layer: every request bit for bit the two-way run.s, [1-9][0-9]* layer waits, 0 layers computed with a slot not yet restored, every home its expert.s bytes: yes$" "$d/determinism.txt" &&
   grep -q "^ok   deterministic on, host homes, the reset layer by layer: A again bit-identical to A cold, A after B bit-identical to A cold$" "$d/determinism.txt" &&
   grep -q "^ok   host homes, layer by layer without the per-layer wait (negative control): [1-9][0-9]* hits on a slot still holding its victim, .* differs from the two-way run, as expected$" "$d/determinism.txt"' ||
  tail -n 8 "$d/determinism.txt"
check "partial host homes on the CPU reference: partial host homes (layers homed in full, in part, none) bit for bit the two-way runs, victims copied down into homes the misses read" \
  'grep -q "^ok   partial host homes ([1-9] layers homed in full, [1-9] in part, [0-9] none): every request bit for bit the two-way run.s, [1-9][0-9]* victims copied down, [1-9][0-9]* layer waits, 0 layers computed with a slot not yet restored, every home its expert.s bytes: yes$" "$d/determinism.txt" &&
   grep -q "^ok   deterministic on, partial host homes, the reset layer by layer and in rounds: A again bit-identical to A cold, A after B bit-identical to A cold$" "$d/determinism.txt"' ||
  tail -n 8 "$d/determinism.txt"
check "dsv41_engine_hot_policy_test: partial homes swap as the spare homes, homed victims dropped and the others copied down, the reset in rounds, restore_in_place refused" \
  'grep -q "^ok   partial homes: 200 of 200 tokens swap the same experts in and out as with spare homes" "$d/hot-policy.txt" &&
   grep -q "^ok   partial homes: [1-9][0-9]* homed victims dropped, [1-9][0-9]* others copied down into a free home, the homed experts. homes their own throughout$" "$d/hot-policy.txt" &&
   grep -q "^ok   partial homes: the reset in [1-9][0-9]* rounds back at the hot set.s slab, every home consistent$" "$d/hot-policy.txt" &&
   grep -q "^ok   restore order: an in-place restore of a layer with partial host homes exits 1 (a child run)$" "$d/hot-policy.txt"' ||
  tail -n 6 "$d/hot-policy.txt"
check "dsv41_engine_hot_policy_test: the mode's rules, the reset at the end among them" \
  'grep -q "^ok   deterministic mode: every request reset and fresh, every boundary waits, every end resets" "$d/hot-policy.txt" &&
   ! grep -q "^FAIL" "$d/hot-policy.txt"' || tail -n 5 "$d/hot-policy.txt"
check "dsv41_engine_hot_policy_test: host homes swap as the spare homes, nothing copied down, the in-place restore, the restore order and its refusal" \
  'grep -q "^ok   host homes: 200 of 200 tokens swap the same experts in and out as with spare homes" "$d/hot-policy.txt" &&
   grep -q "^ok   host homes: no victim copied down, every expert.s home the one init gave it" "$d/hot-policy.txt" &&
   grep -q "^ok   host homes: restore_in_place puts the [1-9] displaced experts" "$d/hot-policy.txt" &&
   grep -q "^ok   restore order: each layer.s computation waits for its own restore once" "$d/hot-policy.txt" &&
   grep -q "^ok   restore order: a layer computing before its restore is queued exits 1" "$d/hot-policy.txt" &&
   grep -qx "dsv41_engine_hot_policy_test: PASS" "$d/hot-policy.txt"' || tail -n 6 "$d/hot-policy.txt"
grep -hE "^(dry run: hot set reset|fixed host RAM beside)" "$d/homes-all.err" "$d/homes-spare.err" | sed 's/^/  /' | cut -c1-200 || true
homes_engine="$(sed -n 's/^fixed host RAM beside the checkpoints: the hot set.s host homes, \([0-9.]*\) MiB pinned host RAM .*/\1/p' "$d/homes-all.err")"
homes_gate="$(python3 -c "import json, sys; print(f\"{json.load(open(sys.argv[1]))['host_homes_bytes'] / 2**20:.1f}\")" "$d/gate-300/gate.json" 2>/dev/null || true)"
check "--serve --dry-run, --hot-homes all: the homes' RAM the gate's (${homes_engine:-none} and ${homes_gate:-none} MiB), the dry reset in place layer by layer, copies up only, every home its own" \
  '[[ -n "$homes_engine" && "$homes_engine" == "$homes_gate" ]] &&
   grep -q "^experts: hot set homes: host homes (--hot-homes all): every slab expert keeps a permanent pinned home, " "$d/homes-all.err" &&
   grep -q "^dry run: hot set reset (--hot-homes all): 40 layers displaced by [0-9]* experts, restored in place layer by layer: [0-9]* swaps, [0-9.]* MiB up and 0.0 down, every layer back at the hot set.s slab with every home its own; 40 events in layer order" "$d/homes-all.err" &&
   grep -q "^READY 262144 stop$" "$d/homes-all.out"'
check "--serve --dry-run, --hot-homes spare: the two-way path named, its slow idle reset named with the deterministic mode on (not off), the dry reset in rounds with the victims copied down" \
  'grep -q "^experts: hot set homes: two-way (--hot-homes spare): .*; the deterministic mode is on (--deterministic on), so this idle reset is the slow one: the next request waited 1.67-1.91 s for it on the RTX PRO 6000 (ccache cycle 1, 2026-10-14, 2885-3311 swaps after each 256-token reply); tools/dsv41/pod/e07_gate.py.s gate.json hot_homes is the --hot-homes value this host.s RAM holds$" "$d/homes-spare.err" &&
   grep -q "^experts: hot set homes: two-way (--hot-homes spare): " "$d/homes-spare-off.err" &&
   ! grep -q "this idle reset is the slow one" "$d/homes-spare-off.err" &&
   ! grep -q "^fixed host RAM beside the checkpoints" "$d/homes-spare.err" &&
   grep -q "^dry run: hot set reset (--hot-homes spare): 40 layers displaced by [0-9]* experts, reset in at most [0-9]* rounds of 2 per layer: [0-9]* swaps, \([0-9.]*\) MiB up and \1 down" "$d/homes-spare.err"'
grep -hE "^(experts: hot set homes|dry run: hot set reset|fixed host RAM beside)" "$d/homes-5612.err" | sed 's/^/  /' | cut -c1-240 || true
# the gate on the ccache pod's 282 GB host (section 3 checks its arithmetic): its held homes' bytes
python3 "$STRATA_DIR/tools/dsv41/pod/e07_gate.py" --vram-mib 97887 --prefetch-experts 128 --cgroup-bytes 281999998976 \
  --mem-total-kib 1188134272 --counts "$COUNTS" --layout "$STRATA_DIR/tools/dsv41/oracle/layout-dsv41-q2k.json" \
  --model-headers "$SHARD1" --out "$d/gate-282" > "$d/gate-282.log" 2>&1 || true
part_engine="$(sed -n 's/^fixed host RAM beside the checkpoints: the hot set.s host homes, \([0-9.]*\) MiB pinned host RAM (--hot-homes 5612: 5612 of 5951 slab experts), held once for the engine whatever the number of conversations (not part of a conversation.s cost)$/\1/p' "$d/homes-5612.err")"
part_gate="$(python3 -c "import json, sys; print(f\"{json.load(open(sys.argv[1]))['host_homes_held_bytes'] / 2**20:.1f}\")" "$d/gate-282/gate.json" 2>/dev/null || true)"
check "partial host homes, --serve --dry-run --hot-homes 5612 (the ccache pod's count): 5612 of 5951 homed (94.3%), 37 of 40 layers in full and layer 37's last 116, the dry reset in place there and in rounds in the other 3, the RAM the gate's (${part_engine:-none} and ${part_gate:-none} MiB)" \
  '[[ -n "$part_engine" && "$part_engine" == "$part_gate" ]] &&
   grep -q "^experts: hot set homes: partial host homes (--hot-homes 5612): 5612 of the slab.s 5951 VRAM experts keep a permanent pinned home (94.3%), [0-9.]* MiB (dry run: as if no mapping were pinned): every slab expert in 37 of the 40 layers that swap (the first in layer order), layer 37.s last 116 of its 142 by the hot set.s order; " "$d/homes-5612.err" &&
   grep -q "^dry run: hot set reset (--hot-homes 5612, partial host homes: 5612 of 5951 slab experts homed, 94.3%): 40 layers displaced by [0-9]* experts; 37 layers with every slab expert homed restored in place layer by layer: [0-9]* swaps, [0-9.]* MiB up and 0.0 down (about [0-9.]* s at 26.12 GB/s up), the next request.s layer l waiting for layer l.s alone; the other 3 in at most [0-9]* rounds of 2 per layer, waited for before the next request: [1-9][0-9]* swaps, \([0-9.]*\) MiB up and \1 down, [1-9][0-9]* victims copied down" "$d/homes-5612.err" &&
   grep -q "^READY 262144 stop$" "$d/homes-5612.out"'
check "partial host homes: --hot-homes 5952, one more than the slab's experts, exits naming both" \
  'grep -q "^dsv41 experts: --hot-homes 5952: the slab has 5951 experts that can take a host home (in the layers that swap)$" "$d/homes-5952.err" &&
   ! grep -q "^READY" "$d/homes-5952.out"'
cost="$(grep -m1 "^conversation checkpoints: " "$d/real.err" || true)"
echo "  real layout: ${cost:-no cost line}" | cut -c1-200
slot_engine="$(sed -n 's/^conversation checkpoints: 4 slots of .* (\([0-9]*\) bytes each); dry run: nothing pinned$/\1/p' "$d/real.err")"

echo "--- 3. pod/e07_gate.py on the real headers: the checkpoints counted, a host without room for them refused"
gate() {   # gate GB OUT [extra args]: GB in bytes when it has 12 digits (a real pod's cgroup)
  local gb="$1" out="$2" cg; shift 2
  (( ${#gb} == 12 )) && cg="$gb" || cg="$(( gb * 1000000000 ))"
  python3 "$STRATA_DIR/tools/dsv41/pod/e07_gate.py" --vram-mib 97887 --prefetch-experts 128 --cgroup-bytes "$cg" \
    --mem-total-kib "$(( cg / 1000 ))" --counts "$COUNTS" --layout "$STRATA_DIR/tools/dsv41/oracle/layout-dsv41-q2k.json" \
    --model-headers "$SHARD1" --out "$tmp/$out" "$@" > "$tmp/$out.log" 2>&1
}
rc256=0 rc209=0 rc209_0=0
gate 256 gate-256 || rc256=$?
gate 209 gate-209 || rc209=$?
gate 209 gate-209-none --checkpoints 0 || rc209_0=$?
tail -n 1 "$tmp/gate-209.log" | sed 's/^/  /' | cut -c1-240
slot_gate="$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['checkpoint_slot_bytes'])" "$tmp/gate-256/gate.json" 2>/dev/null || true)"
check "the engine's dry run and the gate agree on a checkpoint slot of 262144 positions (${slot_engine:-none} and ${slot_gate:-none} bytes)" \
  '[[ -n "$slot_engine" && "$slot_engine" == "$slot_gate" && "$slot_gate" == 844091392 ]]'
check "a 256 GB host: PASS with 4 checkpoints of 0.786 GiB (3.14 GiB pinned) in the RAM the placement needs" \
  '(( rc256 == 0 )) && grep -q "conversation checkpoints 4 x 0.786 GiB = 3.14 GiB pinned (--checkpoints 4, a checkpoint of 262144 positions each)" "$tmp/gate-256.log" &&
   python3 -c "import json, sys; g = json.load(open(sys.argv[1])); sys.exit(0 if g[\"checkpoints\"] == 4 and g[\"checkpoints_gib\"] == 3.144 and g[\"fits\"] else 1)" "$tmp/gate-256/gate.json"'
rc282=0 rc300=0 rc300_off=0
gate 281999998976 gate-282 || rc282=$?
gate 300 gate-300 || rc300=$?
gate 300 gate-300-off --host-homes off || rc300_off=$?
grep -h "host homes" "$tmp/gate-282.log" "$tmp/gate-300.log" | sed 's/^/  /' | cut -c1-240
homes() { python3 -c "import json, sys; g = json.load(open(sys.argv[1])); print(g['hot_homes'], g['host_homes'], g['fits'])" "$1"; }
# the partial homes' worked example: the ccache pod's 282 GB host, cgroup 281999998976 bytes = 262.63 GiB; all 5951 homes
# (71.00 GiB) need 258.61 + 8.0 margin, 3.97 GiB short; 5612 of them (66.95 GiB, the 3 layers left keep 0.07 GiB of
# spare homes) need 254.63 + 8.0, inside it
check "partial host homes, the worked example: the ccache pod's 282 GB (262.63 GiB cgroup) holds 5612 of the 5951 homes (94.3%, 37 of 40 layers and layer 37's last 116) with the 8 GiB margin, all of them 3.97 GiB short; 256 GB holds 3552 (59.7%)" \
  '(( rc256 == 0 && rc282 == 0 )) && [[ "$(homes "$tmp/gate-256/gate.json")" == "3552 True True" && "$(homes "$tmp/gate-282/gate.json")" == "5612 True True" ]] &&
   grep -q "^e07_gate.py: host homes: PARTIAL (--hot-homes 5612): 5612 of the slab.s 5951 experts homed, 94.3% (the first 37 of the 40 layers that swap in full, layer 37.s last 116 of its 142), 66.95 GiB pinned, the other layers. spare homes 0.07: 254.63 GiB + 8.0 margin against 262.63 GiB; all 5951 would need 71.00 GiB (258.61 GiB + 8.0 margin against 262.63 GiB, 3.97 GiB short); " "$tmp/gate-282.log" &&
   grep -q "^e07_gate.py: host homes: PARTIAL (--hot-homes 3552): 3552 of the slab.s 5951 experts homed, 59.7% " "$tmp/gate-256.log" &&
   python3 -c "import json, sys; g = json.load(open(sys.argv[1]))
sys.exit(0 if g[\"host_homes_held\"] == 5612 and g[\"host_homes_of\"] == 5951 and g[\"host_homes_fraction\"] == 0.943 and
         g[\"host_homes_cut\"] == [37, 116, 142] and g[\"need_with_partial_homes_gib\"] + g[\"ram_margin_gib\"] <= g[\"limit_gib\"] else 1)" "$tmp/gate-282/gate.json"'
check "host homes: 300 GB turns the host homes on (71.00 GiB in place of the spare homes), --host-homes off keeps them off" \
  '(( rc300 == 0 && rc300_off == 0 )) && [[ "$(homes "$tmp/gate-300/gate.json")" == "all True True" && "$(homes "$tmp/gate-300-off/gate.json")" == "spare False True" ]] &&
   grep -q "^e07_gate.py: host homes: the slab.s 5951 experts. homes, 71.00 GiB pinned in place of the spare homes. 0.95: 258.61 GiB + 8.0 margin against 279.40 GiB: ON (--hot-homes all)" "$tmp/gate-300.log" &&
   grep -q "refused, the two-way path (--hot-homes spare): --host-homes off$" "$tmp/gate-300-off.log"'
check "209 GB: refused for the checkpoints, naming their count and slot size and every other part; with --checkpoints 0 it fits" \
  '(( rc209 == 1 && rc209_0 == 0 )) &&
   grep -q "^e07_gate.py: refused: the host RAM cannot hold 4 conversation checkpoints of 0.786 GiB (3.14 GiB) beside the warm tier.s 112.38, the engram tables. 60.08, the spare homes. 0.95 and 12.0 GiB working, with 8.0 GiB to spare, in 194.65 GiB$" "$tmp/gate-209.log"'

echo "--- 4. chat_load.py run --alternate against serve/server.py on the fake engine (${P_MS} ms per prompt token, ${D_MS} ms per token)"
mkdir -p "$tmp/direct"
cat > "$tmp/direct/engine" <<EOF
#!/usr/bin/env bash
exec python3 "$here/fakes.py" --serve "\$@"
EOF
chmod +x "$tmp/direct/engine"
python3 -c "import json, sys; d = sys.argv[1]
print(json.dumps({'exe': d + '/engine', 'cwd': d, 'args': ['--model', sys.argv[2], '--ctx', '262144'],
  'model_profile': 'deepseek-v4.1', 'tokenizer_gguf': sys.argv[2], 'log': d + '/engine.log', 'repeat_stop_tokens': 0,
  'checkpoints': 4}))" "$tmp/direct" "$VOCAB" > "$tmp/direct/config.json"
free_port port
( cd "$STRATA_DIR" && FAKE_ENGINE_PROMPT_MS_PER_TOKEN=$P_MS FAKE_ENGINE_DECODE_MS_PER_TOKEN=$D_MS exec setsid python3 -m serve.server \
    --engine strata --config "$tmp/direct/config.json" --port "$port" ) > "$tmp/direct/server.log" 2>&1 < /dev/null &
echo $! > "$tmp/server.pid"
for _ in $(seq 120); do grep -q "ready: http" "$tmp/direct/server.log" && break; sleep 1; done
rc=0
timeout 600 python3 "$here/chat_load.py" run --url "http://127.0.0.1:$port" --conversations "$CONVERSATIONS" --users 4 --turns 3 \
  --alternate --engine-log "$tmp/direct/engine.log" --out "$tmp/direct/alt4.json" > "$tmp/direct/alt4.txt" 2>&1 || rc=$?
sed 's/^/  /' "$tmp/direct/alt4.txt" | tail -n 4 | cut -c1-240
check "the server started the engine with --checkpoints 4 (the config's \"checkpoints\")" \
  'grep -q "^conversation checkpoints: 4 slots of 805.0 MiB" "$tmp/direct/engine.log"'
check "4 users alternating 3 turns: exit 0, every turn 2+ resumed its own prefix (cached = the last prompt), none queued, the turns 2+ columns" \
  '(( rc == 0 )) && grep -q "^| N | requests | .* | turn 2+ TTFT p50 s | turn 2+ TTFT p95 s | turn 2+ TTFT max s | turn 2+ max not queued s | turn 2+ resumed | queued | turn 2+ reset wait p50 s | turn 2+ reset wait max s |$" "$tmp/direct/alt4.txt" &&
   python3 -c "import json, sys; r = json.load(open(sys.argv[1]))[\"runs\"][0]; s = r[\"summary\"]; rows = r[\"rows\"]
last = {}
ok = r[\"label\"] == \"4 alternating\" and len(rows) == 12 and s[\"turn2_requests\"] == 8 and s[\"turn2_resumed\"] == 8 and s[\"queued\"] == 0
for x in sorted(rows, key=lambda x: x[\"start_s\"]):
    ok = ok and not x[\"queued\"] and (x[\"turn\"] == 1 and x[\"cached_tokens\"] == 0 or x[\"cached_tokens\"] == last.get(x[\"user\"]))
    last[x[\"user\"]] = x[\"prompt_tokens\"]
starts = sorted((x[\"start_s\"], x[\"end_s\"]) for x in rows)
ok = ok and all(b[0] >= a[1] - 0.01 for a, b in zip(starts, starts[1:]))
sys.exit(0 if ok and s[\"turn2_unqueued_ttft_max_s\"] < 1.0 else 1)" "$tmp/direct/alt4.json"'
check "--engine-log: each request matched to the engine's (1 to 12), its reset wait the fake's 5.0 ms after a request in the mode" \
  'python3 -c "import json, sys; r = json.load(open(sys.argv[1]))[\"runs\"][0]; rows = sorted(r[\"rows\"], key=lambda x: x[\"start_s\"])
s = r[\"summary\"]
sys.exit(0 if [x[\"engine_request\"] for x in rows] == list(range(1, 13)) and [x[\"reset_wait_s\"] for x in rows] == [0.0] + [0.005] * 11 and
         s[\"turn2_reset_wait_max_s\"] == 0.005 and r[\"think_s\"] == 0 else 1)" "$tmp/direct/alt4.json"'
rc_think=0 rc_log=0
python3 "$here/chat_load.py" run --url "http://127.0.0.1:$port" --conversations "$CONVERSATIONS" --users 1 --think-s 5 \
  --out "$tmp/direct/bad.json" > "$tmp/direct/bad-think.txt" 2>&1 || rc_think=$?
python3 "$here/chat_load.py" run --url "http://127.0.0.1:$port" --conversations "$CONVERSATIONS" --users 1 \
  --engine-log "$tmp/direct/engine.log" --out "$tmp/direct/bad.json" > "$tmp/direct/bad-log.txt" 2>&1 || rc_log=$?
check "--think-s and --engine-log without --alternate: refused by name, nothing sent" \
  '(( rc_think != 0 && rc_log != 0 )) && grep -q "need --alternate" "$tmp/direct/bad-think.txt" && grep -q "need --alternate" "$tmp/direct/bad-log.txt" &&
   [[ ! -e "$tmp/direct/bad.json" ]]'
rc=0
timeout 600 python3 "$here/chat_load.py" run --url "http://127.0.0.1:$port" --conversations "$CONVERSATIONS" --users 5 --turns 2 \
  --alternate --out "$tmp/direct/alt5.json" > "$tmp/direct/alt5.txt" 2>&1 || rc=$?
grep -E "checkpoints: .*evicted" "$tmp/direct/engine.log" | head -n 2 | sed 's/^/  /' | cut -c1-200 || true
# the same engine: alt4's 12 requests, then alt5's; its 5th request (17, Ember's first turn) finds every slot held by
# alt5's first four users and evicts the least recently used (slot 0, Amber's 16413 tokens), so Amber's second turn
# (18) is read cold and evicts the next one (slot 1, Basalt's)
check "5 users over the 4 checkpoints: the 5th's first turn evicts the least recently used (Amber's), whose next turn is read cold" \
  '(( rc == 0 )) && grep -q "^dsv41-engine serve: request 17: checkpoints: none restored (cold), saved to 0 (16407 tokens, .*), 0 superseded, evicted 0 (16413 tokens, the least recently used); 4 of 4 slots held, " "$tmp/direct/engine.log" &&
   grep -q "^dsv41-engine serve: request 18: checkpoints: none restored (cold), saved to 1 (.*, evicted 1 (16414 tokens, the least recently used); " "$tmp/direct/engine.log" &&
   python3 -c "import json, sys; r = json.load(open(sys.argv[1]))[\"runs\"][0][\"rows\"]
sys.exit(0 if [x[\"cached_tokens\"] for x in r if x[\"user\"] == 0] == [0, 0] else 1)" "$tmp/direct/alt5.json"'
kill -TERM -- "-$(cat "$tmp/server.pid")" 2>/dev/null || true
rm -f "$tmp/server.pid"

echo "--- 5. pod/e07.sh --dry-run, E07_PLAN=ccache"
# partial host homes: the ccache pod's records (20261014T044315Z: cgroup 281999998976 bytes, MemTotal
# 1188134272 kB), as that session kept them (no cgroup line in probe.log, e07-run.sh's out/env.log) and as
# pod-session.sh now writes them into probe.log
mkdir -p "$tmp/pod-old/out" "$tmp/pod-new"
printf '%s\n' "=== pod test-pod-1 machine test-machine-1 (NVIDIA RTX PRO 6000 Blackwell Workstation Edition) at 04:44:45 UTC" \
  "dsv41_cuda_probe: ok, NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120), 1 device(s)" > "$tmp/pod-old/probe.log"
printf '%s\n' "MemTotal:       1188134272 kB" "/sys/fs/cgroup/memory.max: 281999998976" > "$tmp/pod-old/out/env.log"
{ cat "$tmp/pod-old/probe.log"; printf '%s\n' "pod memory cgroup: /sys/fs/cgroup/memory.max: 281999998976" \
    "MemTotal:       1188134272 kB"; } > "$tmp/pod-new/probe.log"
rc=0; E07_PLAN=ccache PROBE_LOG="$tmp/pod-old/probe.log" OUT="$tmp/dry-out" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry.txt" 2>&1 || rc=$?
# the old 2.00 USD cap (46 min a pod): the resumed plan fits, its worst case does not, and the dry run says so
rc2=0; E07_PLAN=ccache MAX_SESSION_USD=2.00 PROBE_LOG="$tmp/pod-new/probe.log" OUT="$tmp/dry-out" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry-2.txt" 2>&1 || rc2=$?
grep -m1 "^    expected host homes: " "$tmp/dry.txt" | sed 's/^ */  /' | cut -c1-200 || true
check "partial host homes: the dry run's gate on the ccache pod's real cgroup (PROBE_LOG: its out/env.log, then a probe.log with the cgroup line), the expected host homes 5612 of 5951 (94.3%)" \
  'for f in "$tmp/dry.txt" "$tmp/dry-2.txt"; do
     grep -q "^--- the gate per GPU (pod/e07_gate.py on the real headers .*, the pod.s cgroup 281999998976 bytes and MemTotal 1188134272 kB from PROBE_LOG=" "$f" &&
     (( $(grep -c "^    expected host homes: 5612 of the slab.s 5951 experts (94.3%), dsv41-engine --hot-homes 5612: the RAM holds 5612 of the slab.s 5951 experts. homes with the margin (94.3%; all of them: 258.61 GiB + 8.0 margin against 262.63 GiB, 3.97 GiB short)$" "$f") == 2 )) || exit 1
   done'
grep -E "^plan \((volume|hf)\)( with no turn resumed)?: total|^price limit|^GPUs, in order" "$tmp/dry.txt" | sed 's/^/  /' | cut -c1-200 || true
check "the ccache dry run exits 0 (exit $rc)" '(( rc == 0 ))' || tail -n 10 "$tmp/dry.txt"
check "at MAX_SESSION_USD 2.00 the dry run exits 2 (exit $rc2) on the HF plan with no turn resumed, naming its minutes and USD" \
  '(( rc2 == 2 )) && grep -q "^e07.sh: E07_PLAN=ccache.s hf plan with no turn resumed is over the cap: 46.7 min, 2.02 USD$" "$tmp/dry-2.txt" &&
   grep -q "^e07.sh: the plan is over the cap with the model from: hf (no turn resumed)$" "$tmp/dry-2.txt"' || tail -n 5 "$tmp/dry-2.txt"
check "the RTX PRO 6000 only (both editions, no A100)" \
  'grep -q "^GPUs, in order: NVIDIA RTX PRO 6000 Blackwell Server Edition;NVIDIA RTX PRO 6000 Blackwell Workstation Edition; host RAM at least 256 GB" "$tmp/dry.txt" &&
   ! grep -q "A100.* takes payload" "$tmp/dry.txt"'
q_line="$(grep -n "^  qualification, when" "$tmp/dry.txt" | cut -d: -f1 || true)"
up_line="$(grep -n "payload-<tag>.tar.gz to /root/e07.tar.gz" "$tmp/dry.txt" | cut -d: -f1 || true)"
check "the machine qualification is listed before the payload goes up, and is each plan's first step" \
  '[[ -n "$q_line" && -n "$up_line" ]] && (( q_line < up_line )) &&
   [[ "$(grep -A1 "^plan with the model from HF" "$tmp/dry.txt" | tail -n 1)" == "  create, SSH, CUDA probe, qualification"* ]]'
sed -n "/^plan with the model from HF /,/^plan (hf)/p" "$tmp/dry.txt" | awk '/^  / { print $1 }' > "$tmp/planned.txt"
want_hf="create, payload env model-headers gate setup model-download dry-run engine-server load-alt4 load-alt4-think5 load-c1 load-c2 load-c4 det-sampling server-stop pull"
check "the HF plan's steps: $want_hf" '[[ "$(tr "\n" " " < "$tmp/planned.txt")" == "$want_hf " ]]'
check "every pod step of both model paths (14 from the volume, 13 from HF) has its timeout, 2 x its estimate; server-stop planned at 60 s" \
  '(( $(sed -n "/^plan with the model from /,/^plan (hf)/p" "$tmp/dry.txt" | grep -c "; timeout min(") == 27 )) &&
   grep -q "^  load-alt4 (chat_load.py --alternate, 4 users x 3 turns: 12 requests one after the other, 4 turns 1 at 14.2 s, 8 resumed at 9.4 s, + 0.2) *2.4  (total [0-9.]*; timeout min(4.8 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  load-alt4-think5 (the same with 5 s between requests, 11 x 5 s more, + 0.2) *3.3  (total [0-9.]*; timeout min(6.6 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  load-c4 (chat_load.py, N = 4: 2 turns each, 8 requests one at a time at 14.2 s, + 0.2) *2.1  (total [0-9.]*; timeout min(4.2 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  det-sampling (.*  *5  (total [0-9.]*; timeout min(10 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  server-stop (the engine.s shutdown, 49-52 s measured: followup-q 18) *1  (total [0-9.]*; timeout min(2 min, the time left))$" "$tmp/dry.txt" &&
   grep -q "^  model-download (264.5 GB at 0.25 GB/s, .*18.2  (total [0-9.]*; timeout min(36.4 min, the time left))$" "$tmp/dry.txt"'
usd="$(sed -n "s/^plan (hf): total .* (about \([0-9.]*\) USD at 2.60 USD\/h)$/\1/p" "$tmp/dry.txt")"
worst="$(sed -n "s/^plan (hf) with no turn resumed: total \([0-9.]*\) min of the cap 53 min at the limit price (about \([0-9.]*\) USD at 2.60 USD\/h)$/\1 \2/p" "$tmp/dry.txt")"
check "the HF plan's estimate under 2.30 USD at 2.60 USD/h (${usd:-none} USD), its worst case with no turn resumed under it too (${worst:-none}: min, USD), the session cap 2.30 over at most 3 pods, a pod's 53 min" \
  '[[ -n "$usd" && -n "$worst" ]] && python3 -c "import sys; u, m, w = map(float, sys.argv[1:]); sys.exit(0 if u < 2.3 and w < 2.3 and m < 53 and w > u else 1)" "$usd" $worst &&
   grep -q "^price limit: 2.60 USD/h; session cap: 2.30 USD over at most 3 pods, each min(110 min, what is left / price), at the limit price 53 min$" "$tmp/dry.txt" &&
   grep -q "^  every timeout 2 x the estimate; the worst case, no turn resumed, adds 1.3 min (every alternating$" "$tmp/dry.txt"'
check "the payload's sources pass: the tarball's engine is HEAD's, chat_load.py and det_sampling_check.py shipped, the conversations HEAD's build" \
  'grep -q "the tarball.s engine is HEAD.s$" "$tmp/dry.txt" && grep -q "src/tools/dsv41/quality/chat_load.py: last changed in " "$tmp/dry.txt" &&
   grep -q "src/tools/dsv41/quality/det_sampling_check.py: last changed in " "$tmp/dry.txt" &&
   grep -q "^  e07/conversations.json: .*, what chat_load.py at HEAD builds from " "$tmp/dry.txt"'

echo "--- 5. the ccache rehearsal: e07-run.sh from the payload, with the stubs; e07_report.py's sheet"
rc=0; E07_PLAN=ccache OUT="$tmp/payload" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --payload-only > "$tmp/payload.log" 2>&1 || rc=$?
check "e07.sh --payload-only builds the ccache payload" '(( rc == 0 )) && [[ -f "$tmp/payload/payload-cu130.tar.gz" ]]' ||
  tail -n 5 "$tmp/payload.log"
rm -rf -- "$tmp/payload/payload-cu130/"
bin="$tmp/bin" stubs="$tmp/stubs"
mkdir -p "$bin" "$stubs"
# c01-test.sh's stubs: the GPU, numactl, pip, the download tools, the engine, the HF headers and download, the gate
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# c02-test.sh's GPU: an RTX PRO 6000's 97887 MiB, no NUMA node for GPU0
case " $* " in
  *" topo "*) printf '\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t0-11\tN/A\t\tN/A\n' ;;
  *" --query-gpu=memory.total --format=csv,noheader,nounits "*) echo 97887 ;;
  *" --query-gpu=memory.used,memory.total --format=csv,noheader "*) echo "1024 MiB, 97887 MiB" ;;
  *) echo "nvidia-smi (c02-test.sh stub): GPU 0, 97887 MiB" ;;
esac
EOF
cat > "$bin/numactl" <<'EOF'
#!/usr/bin/env bash
while [[ "${1:-}" == --* ]]; do shift; done
exec "$@"
EOF
printf '#!/usr/bin/env bash\necho "pip (c02-test.sh stub, nothing installed): $*"\n' > "$bin/pip"
for t in aria2c apt-get; do
  printf '#!/usr/bin/env bash\necho "%s (c02-test.sh stub): nothing downloads or installs here: $*" >&2\nexit 1\n' "$t" > "$bin/$t"
done
cat > "$stubs/dsv41-engine" <<'EOF'
#!/usr/bin/env bash
# c02-test.sh's engine: --dry-run exits 0, --serve is fakes.py's engine (the tokenizer from --model)
case " $* " in *" --dry-run "*) echo "dsv41-engine (c02-test.sh stub): dry run of $*"; exit 0 ;; esac
args=()
for a in "$@"; do [[ "$a" == --serve ]] || args+=("$a"); done
exec python3 "$(dirname "$0")/src/tools/dsv41/quality/fakes.py" --serve "${args[@]}"
EOF
cat > "$stubs/hf_model.py" <<'EOF'
#!/usr/bin/env python3
"""c02-test.sh's hf_model.py: `headers MANIFEST DIR` copies the real heads (C02_HEADERS/head-N.bin) into DIR."""
import os
import shutil
import sys

if len(sys.argv) != 4 or sys.argv[1] != "headers":
    sys.exit(f"hf_model.py (c02-test.sh stub): headers MANIFEST DIR only, not {sys.argv[1:]}")
os.makedirs(sys.argv[3], exist_ok=True)
for i in range(1, 8):
    shutil.copy(os.path.join(os.environ["C02_HEADERS"], f"head-{i}.bin"), sys.argv[3])
EOF
cat > "$stubs/hf-download.sh" <<'EOF'
#!/usr/bin/env bash
# c02-test.sh's download: the sparse shards (C02_MODEL) copied into DST, the record the sheet reads, marked a stub
set -euo pipefail
mkdir -p "$2"
cp --sparse=always "$C02_MODEL"/*.gguf "$2"/
python3 -c "import json, sys; json.dump({'stub': 'c02-test.sh: the sparse shards copied, no download', 'repo': 'c02-test-stub',
  'revision': 'none', 'files': 7, 'bytes': 0, 'download_s': 0, 'gb_per_s': 0, 'check_s': 0, 'ok': True}, open(sys.argv[1], 'w'))" "$3"
EOF
cat > "$stubs/e07_gate.py" <<'EOF'
#!/usr/bin/env python3
"""c02-test.sh's gate: the payload's own e07_gate.py (e07_gate_real.py) with the ccache pod's cgroup and MemTotal
(20261014T044315Z's 282 GB host: 262.63 GiB, partial host homes)."""
import os
import sys

a = sys.argv[1:]
for key, value in (("--cgroup-bytes", "281999998976"), ("--mem-total-kib", "1188134272")):
    a[a.index(key) + 1] = value
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e07_gate_real.py"), *a])
EOF
chmod +x "$bin"/* "$stubs"/*
s="$tmp/s-ccache"
mkdir -p "$s/pod/out" "$s/root/root"
tar -xzf "$tmp/payload/payload-cu130.tar.gz" -C "$s/pod" --exclude='libcublas*' --exclude='libggml-cuda.so*'
cp "$stubs/dsv41-engine" "$stubs/hf_model.py" "$stubs/hf-download.sh" "$s/pod/"
mv "$s/pod/src/tools/dsv41/pod/e07_gate.py" "$s/pod/src/tools/dsv41/pod/e07_gate_real.py"
cp "$stubs/e07_gate.py" "$s/pod/src/tools/dsv41/pod/"
free_port port
( cd "$s/pod" && export PATH="$bin:$PATH" E07_POD_ROOT="$s/root" E07_Q_PORT="$port" C02_MODEL="$tmp/model" \
    C02_HEADERS="$HEADERS" POD_MACHINE="c02-ccache" FAKE_ENGINE_PROMPT_MS_PER_TOKEN="$P_MS" FAKE_ENGINE_DECODE_MS_PER_TOKEN="$D_MS" \
    FAKE_ENGINE_IDLE_RESET_MS=1410
  rc=0; DEADLINE=$(( $(date +%s) + 2400 )) POD_VOLUME= timeout 2700 bash e07-run.sh > out/e07-run.out 2>&1 < /dev/null || rc=$?
  echo "$rc" > out/e07-run.exit ) &
jobs_+=($!)
wait "${jobs_[@]}"
jobs_=()
o="$s/pod/out"
awk '/^step / { sub(":$", "", $2); print $2 }' "$o/steps.log" > "$tmp/ran.txt"
echo "  ccache: $(tr '\n' ' ' < "$tmp/ran.txt")"
grep -vxE 'create,|payload|pull' "$tmp/planned.txt" > "$tmp/planned-pod.txt" || true
check "e07-run.sh exits 0, every step exit 0" \
  '[[ "$(cat "$o/e07-run.exit")" == 0 ]] && ! grep "^step " "$o/steps.log" | grep -qv ": exit 0, "'
check "steps.log has every planned step in order, then done" \
  'diff "$tmp/planned-pod.txt" "$tmp/ran.txt" > /dev/null && [[ "$(tail -n 1 "$o/steps.log")" == done ]]'
# the load steps' 38 requests in the deterministic mode (det-sampling's mode-off row after them is its own)
check "the engine ran with the gate's placement, prefetch 128, the partial host homes (5612) and --log-ids, its 4 checkpoints, the 38 load requests in the deterministic mode" \
  'python3 -c "import json, sys; a = json.load(open(sys.argv[1]))[\"args\"]
sys.exit(0 if a[a.index(\"--prefetch-experts\") + 1] == \"128\" and \"--hot-set\" in a and \"--log-ids\" in a and \"--deterministic\" not in a and
         a[a.index(\"--hot-homes\") + 1] == \"5612\" else 1)" "$o/quality-server.json" &&
   grep -q "^host homes (e07_gate.py, decision-c02 amendments 3c and 3d): the RAM holds 5612 of the slab.s 5951 experts. homes with the margin (94.3%; " "$o/placement.txt" &&
   grep -q "^experts: hot set homes: partial host homes (--hot-homes 5612): 5612 of the slab.s 5951 VRAM experts keep a permanent pinned home (94.3%), 68560.7 MiB: every slab expert in 37 of the 40 layers that swap" "$o/quality-engine.log" &&
   grep -q "^conversation checkpoints: 4 slots of 805.0 MiB" "$o/quality-engine.log" &&
   [[ "$(grep -E "^dsv41-engine serve: request [0-9]+: [0-9]+ prompt tokens, " "$o/quality-engine.log" | head -n 38 | grep -c "; deterministic; greedy$")" == 38 ]]'
check "load-alt4: 4 users alternating 3 turns, every turn 2+ resumed (8 of 8), none queued" \
  'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); r = j[\"runs\"][0]; s = r[\"summary\"]
sys.exit(0 if j[\"ok\"] and j[\"alternate\"] and r[\"label\"] == \"4 alternating\" and s[\"turn2_requests\"] == 8 and
         s[\"turn2_resumed\"] == 8 and s[\"queued\"] == 0 else 1)" "$o/ccache/load-alt4.json"'
# the fake's idle reset runs 1.41 s from each DONE over 40 layers, the 3 not homed in full in rounds for their share
# (106 ms) the host waits for: back to back, every turn 2+ arrives inside it, waits on the host for the rounds' rest and
# its read per layer for the layered rest (1.41 s less the client's and the server's own time in all); 5 s later, after
# its end
check "load-alt4, the strict run: every turn 2+ waited for the reset (0.5 to 1.41 s, from the engine's log: the rounds on the host, the layered part on the device), its TTFT under 2 s with it" \
  'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); r = j[\"runs\"][0]; later = [x for x in r[\"rows\"] if x[\"turn\"] > 1]
sys.exit(0 if j[\"think_s\"] == 0 and len(later) == 8 and all(0.5 <= x[\"reset_wait_s\"] <= 1.41 and x[\"ttft_s\"] <= 2.0
         for x in later) else 1)" "$o/ccache/load-alt4.json" &&
   (( $(grep -cE "^dsv41-engine serve: request [0-9]+: idle reset layer by layer \(host homes\): 7 swaps copied up only \(85.4 MiB, none down\), 1410.0 ms on the device, queued in 0.4 ms; this request waited [0-9.]+ ms on the host and [0-9.]+ ms on the device over [0-9]+ layers" "$o/quality-engine.log") >= 11 )) &&
   (( $(grep -cE "^dsv41-engine serve: request [0-9]+: idle reset in rounds \(partial host homes, 3 layers\): 1 swaps \(12.2 MiB up, 12.2 MiB down: 1 victims copied down\) in 105.8 ms on the host, before this request was read$" "$o/quality-engine.log") >= 11 )) &&
   (( $(grep -cE "^dsv41-engine serve: request [0-9]+: copies down left \(host homes held 5612 of 5951 slab experts\): [0-9]+ of this request.s [0-9]+ swaps copied their victim down " "$o/quality-engine.log") >= 38 ))'
check "load-alt4-think5: 4 users alternating 3 turns 5 s apart, every turn 2+ resumed, none queued, no reset wait, TTFT under 2 s" \
  'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); r = j[\"runs\"][0]; s = r[\"summary\"]
st = sorted((x[\"start_s\"], x[\"end_s\"]) for x in r[\"rows\"])
sys.exit(0 if j[\"ok\"] and j[\"think_s\"] == 5 and r[\"label\"] == \"4 alternating, think 5 s\" and s[\"turn2_resumed\"] == 8 and
         s[\"queued\"] == 0 and s[\"turn2_reset_wait_max_s\"] == 0.0 and s[\"turn2_unqueued_ttft_max_s\"] < 2.0 and
         all(b[0] - a[1] >= 4.99 for a, b in zip(st, st[1:])) else 1)" "$o/ccache/load-alt4-think5.json"'
check "load-c1, load-c2 and load-c4 came back, every request answered" \
  'for n in 1 2 4; do python3 -c "import json, sys; j = json.load(open(sys.argv[1])); sys.exit(0 if j[\"ok\"] and len(j[\"runs\"][0][\"rows\"]) == 2 * int(sys.argv[2]) else 1)" "$o/ccache/load-c$n.json" "$n" || exit 1; done'
grep -E "^(item|  item|cost: first)" "$o/det-sampling.log" 2>/dev/null | sed 's/^/  /' | cut -c1-200 || true
check "det-sampling: items 7 and 8 PASS with the reset after each request (the det_sampling rehearsal still passing)" \
  'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); i7, i8 = j[\"items\"][\"7\"], j[\"items\"][\"8\"]
sys.exit(0 if j[\"ok\"] and i7[\"pass\"] and i8[\"seeds\"][\"pass\"] and i8[\"speed\"][\"pass\"] and i8[\"penalty\"][\"pass\"] else 1)" "$o/det-sampling.json"'
check "the server was stopped" \
  'grep -q "^server [0-9]* stopped$" "$o/server-stop.log" && ! kill -0 "$(cat "$o/quality-server.pid")" 2>/dev/null'
f="$tmp/report/20261014T090000Z"
mkdir -p "$f"
cp -r "$o" "$f/out"
# e07.sh's records of the session: its cap and a pod's cost line as pod-session.sh writes it
echo 2.30 > "$f/session-cap"
echo "pod c02stub: 2700 s at 2.6 USD/h = 1.950 USD (upper bound: billing rounds up)" > "$f/cost.txt"
CCACHE_BASELINE="$CCACHE_BASELINE" python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/sheets" > "$tmp/report.log" 2>&1 || true
r="$tmp/sheets/2026-10-14-ccache/README.md"
check "the ccache sheet: the session's cost against its cap, item 10's table and verdict, the before/after table by N against the baseline's, the cost per conversation, items 7 and 8" \
  'grep -q "^Build: Strata .*Cost: pod c02stub: 2700 s at 2.6 USD/h = 1.950 USD (upper bound: billing rounds up). Session: 1.950 USD over 1 pod, the session.s cap 2.30 USD: within it. " "$r" &&
   grep -q "^## Item 10: the users taking turns (out/ccache/load-alt<N>.json, load-alt<N>-think<S>.json)" "$r" &&
   grep -q "^| 4 alternating | 12 | .* | 8 of 8 | 0 | [0-9.]* | [0-9.]* | logs/ccache/load-alt4.json |$" "$r" &&
   grep -q "^| 4 alternating, think 5 s | 12 | .* | 8 of 8 | 0 | 0.0 | 0.0 | logs/ccache/load-alt4-think5.json |$" "$r" &&
   grep -q "^| 4 alternating | 0.0 | [0-9.]* | [0-9.]* | [0-9.]* | [0-9.]* | [0-9.]* | [0-9.]* | item 10.s verdict |$" "$r" &&
   grep -q "^| 4 alternating, think 5 s | 5.0 | [0-9.]* | [0-9.]* | [0-9.]* | [0-9.]* | 0.000 | 0.000 | informational |$" "$r" &&
   grep -q "^Item 10 (4 alternating): PASS: 8 of 8 turns 2+ resumed a cached prefix, .* the reset wait at most [0-9.]* s$" "$r" &&
   ! grep -q "^Every turn 2+ over 2 s" "$r" &&
   grep -q "^## The hot set.s homes and the reset" "$r" &&
   grep -q "^Path: .experts: hot set homes: partial host homes (--hot-homes 5612): 5612 of the slab.s 5951 VRAM experts" "$r" &&
   grep -q "^Gate: host homes (e07_gate.py, decision-c02 amendments 3c and 3d): the RAM holds 5612 of the slab.s 5951 experts. homes with the margin (94.3%; all of them: 258.61 GiB + 8.0 margin against 262.63 GiB, 3.97 GiB short)$" "$r" &&
   grep -q "^(the homes 70.998 GiB; the host RAM with them 258.61 GiB + 8.0 margin against 262.63 GiB, the memory cgroup.s limit)$" "$r" &&
   grep -q "^Homes held: 5612 of the slab.s 5951 experts (94.3%), 66.95 GiB, --hot-homes 5612; partial: layer 37 cut at its last 116 of 142, the layers after it on the two-way path, the RAM with them 254.63 GiB + 8.0 margin$" "$r" &&
   ! grep -q "each at [0-9]* checkpoints" "$r" &&
   grep -q "^- host: its checkpoint slot, 805.0 MiB of pinned host RAM, sized for a full 262144-position conversation and pinned at load (4 slots, 3220.0 MiB in all); a 16K turn.s checkpoint fills [0-9.]* to [0-9.]* MiB of it (median [0-9.]*, the [1-9][0-9]* saves of prompts of 16384 tokens or more)$" "$r" &&
   grep -q "^- device: the live KV (.*) and the in-place carry snapshot ([0-9.]* KiB); " "$r" &&
   grep -q "^| the hot set.s host homes (5612 of the slab.s 5951 experts, --hot-homes 5612; the engine: 68560.7 MiB) | 66.95 |$" "$r" &&
   grep -q "^| the warm tier ([0-9]* experts, pinned) | [0-9.]* |$" "$r" && grep -q "^| the margin (decision e07 gate) | 8.00 |$" "$r" &&
   grep -q "^Host RAM the gate placed: 254.63 GiB (the fixed costs above without the margin, plus the 4 checkpoint slots. 3.144 GiB) + 8.0 margin against 262.63 GiB, the memory cgroup.s limit; homes for every slab expert (5951, 70.998 GiB) would need 258.61 GiB + 8.0 margin, 3.97 GiB more than the limit (the gate.s)$" "$r" &&
   grep -q "^The rounds (partial host homes, 3 layers not homed in full): [0-9]* requests after one, [0-9]* swaps, [0-9]* victims copied down" "$r" &&
   grep -q "^Copies down left (host homes held 5612 of 5951): [0-9]* of [0-9]* decode swaps ([0-9.]*%) and [0-9]* of [0-9]* reset swaps ([0-9.]*%) over [0-9]* requests$" "$r" &&
   grep -q "^| request | swaps | copied down | MiB down | reset swaps before it | reset copied down | reset MiB down |$" "$r" &&
   grep -q "^The layered reset: [0-9]* requests after one, the full reset 1410 to 1410 ms on the device" "$r" &&
   grep -q "^| request | reset swaps | MiB up | full reset on the device (ms) | queued on the host (ms) | waited on the host (ms) | waited on the device (ms) | layers that waited | longest layer wait (ms) |$" "$r" &&
   grep -q "^Think-time run (4 alternating, think 5 s, informational): 8 of 8 turns 2+ resumed, " "$r" &&
   grep -q "^## Before and after: the C01 runs again (out/ccache/load-c<N>.json)" "$r" &&
   grep -q "^| 4 | 40.415 | [0-9.]* | 40.895 | [0-9.]* | 0 of 4 | [0-9] of 4 | 33.962 | " "$r" &&
   grep -q "^At load: .conversation checkpoints: 4 slots of 805.0 MiB pinned host RAM each" "$r" &&
   grep -q "^- restores from the host: [0-9]*, " "$r" && grep -q "^## Items 7 and 8 once (out/det-sampling.json)" "$r" &&
   grep -q "^## Machine qualification" "$r"' || tail -n 5 "$tmp/report.log"
if (( failed )); then
  echo "  the tail of e07-run.out and steps.log:"
  tail -n 5 "$o/e07-run.out" "$o/steps.log" 2>/dev/null | sed 's/^/    /' || true
fi

echo "--- 6. pod/runpod_api.py body: allowedCudaVersions left out with a minor above 13.0, kept otherwise (no key, no request)"
cuda_body() {   # cuda_body VERSIONS: the create body's allowedCudaVersions, or "absent"
  python3 "$STRATA_DIR/tools/dsv41/pod/runpod_api.py" body --gpu "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
    --cloud SECURE --image runpod/base:0.6.2-cuda12.4.1 --disk 20 --cuda-versions "$1" |
    python3 -c 'import json, sys; b = json.load(sys.stdin); print(",".join(b["allowedCudaVersions"]) if "allowedCudaVersions" in b else "absent")'
}
new_minor="$(cuda_body 13.0,13.1,13.2 2>&1 || true)" old_minor="$(cuda_body 12.8,12.9,13.0 2>&1 || true)"
check "runpod_api.py body: 13.0,13.1,13.2 leaves the field out (${new_minor:-none}), 12.8,12.9,13.0 keeps it (${old_minor:-none})" \
  '[[ "$new_minor" == absent && "$old_minor" == 12.8,12.9,13.0 ]]'

echo "c02-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
