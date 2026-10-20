#!/usr/bin/env bash
# Tests the e07b plan's session tooling without a pod: pod-steps.sh's per-step budgets
# (a step over its own timeout, min(2 x its estimate, the time left), is killed there with the cause at the end of its
# log and on its out/steps.log line, and the next step runs; a step without an estimate keeps the deadline),
# hf-download.sh's throughput guard on a fake aria2c (a slow download stopped after HF_GUARD_AFTER_S with exit 3,
# "download too slow: X GB/s" printed, too_slow in its JSON and the partial file deleted; a fast one left to finish
# and checked), and pod/e07_report.py on a fixture E07_PLAN=e07b session (two-turn passed, bench-e07b killed at its
# timeout, the rest never ran): the goal table's six rows with a verdict each and a cause for every FAIL ("timeout
# after N s in bench-e07b", never only "no result"), the host PCIe row and the GPU named from env.log; then a fixture
# on an A100 whose bench turn2 took 1.06 s: PASS in both the goal and the Speed table (one rule, the strict 1 s noted)
# and the A100 named, never the RTX PRO 6000. Exits 1 if any check fails.
#
#   bash tools/dsv41/pod/e07b-test.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf -- "${tmp:?}"' EXIT
failed=0
check() {
  if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; failed=$((failed + 1)); fi
}

echo "--- pod-steps.sh: per-step budgets from STEP_ESTIMATES"
mkdir -p "$tmp/steps"
printf 'slow 2\nquick 5\n' > "$tmp/steps/estimates"
(
  # e07-run.sh's options: no errexit (pod-steps.sh's functions return a status, the runner goes on)
  set +e
  cd "$tmp/steps"
  DEADLINE=$(( $(date +%s) + 600 )) STEP_ESTIMATES="$tmp/steps/estimates"
  # shellcheck source=tools/dsv41/pod/pod-steps.sh
  source "$here/pod-steps.sh"
  steps_begin
  run slow sleep 60 || echo "slow returned $?" > out/slow-rc
  run quick echo "the next step ran"
  run unlisted true
  echo "$failed" > out/failed
)
o="$tmp/steps/out"
check "the overrunning step is killed at its own timeout (2 x 2 s), exit 124" \
  'grep -qx "slow returned 124" "$o/slow-rc"'
check "its steps.log line names the cause" \
  'grep -qE "^step slow: exit 124, [45] s \(timeout after 4 s: 2 x its plan estimate of 2 s; min\(2 x 2 s, the [0-9]+ s left\)\)$" "$o/steps.log"'
check "its log ends with the cause" 'tail -n 1 "$o/slow.log" | grep -qE "^step slow killed: timeout after 4 s \(2 x its plan estimate of 2 s"'
check "the next step still runs and passes" \
  'grep -qx "step quick: exit 0, [0-9] s" "$o/steps.log" && grep -qx "the next step ran" "$o/quick.log"'
check "a step without an estimate passes with the deadline as its budget" 'grep -qx "step unlisted: exit 0, 0 s" "$o/steps.log"'
check "failed is 1 after the killed step" 'grep -qx 1 "$o/failed"'
sed 's/^/  /' "$o/steps.log"

echo "--- hf-download.sh: the throughput guard on a fake aria2c"
mkdir -p "$tmp/bin"
# the fake writes RATE bytes every 0.2 s into each out= file of its input file until it holds SIZE bytes
cat > "$tmp/bin/aria2c" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in --input-file=*) input="${a#*=}" ;; --dir=*) dir="${a#*=}" ;; esac; done
name="$(sed -n 's/^  out=//p' "$input" | head -n 1)"
: > "$dir/$name"
while (( $(stat -c %s "$dir/$name") < FAKE_SIZE )); do head -c "$FAKE_RATE" /dev/zero >> "$dir/$name"; sleep 0.2; done
EOF
chmod +x "$tmp/bin/aria2c"
manifest() {   # manifest SIZE: one file of SIZE zero bytes
  python3 -c "import hashlib, json, sys; n = int(sys.argv[1])
json.dump({'repo': 'fake/repo', 'revision': 'f' * 40, 'files': [{'name': 'x.gguf', 'size': n,
  'sha256': hashlib.sha256(bytes(n)).hexdigest(), 'url': 'https://example.invalid/x.gguf'}]}, open(sys.argv[2], 'w'))" "$1" "$2"
}
manifest 10000000000 "$tmp/slow.json"
rc=0
PATH="$tmp/bin:$PATH" FAKE_SIZE=10000000000 FAKE_RATE=1048576 HF_GUARD_AFTER_S=2 HF_GUARD_END_BY=$(( $(date +%s) + 60 )) \
  HF_GUARD_REST_S=1500 bash "$here/hf-download.sh" "$tmp/slow.json" "$tmp/dl-slow" "$tmp/dl-slow.json" > "$tmp/dl-slow.log" 2>&1 || rc=$?
sed 's/^/  /' "$tmp/dl-slow.log"
check "a slow download stops with exit 3" '(( rc == 3 ))'
check "it says download too slow with its rate" 'grep -qE "^hf-download.sh: download too slow: 0\.[0-9]{3} GB/s \(" "$tmp/dl-slow.log"'
check "its JSON records too_slow" 'python3 -c "import json, sys; j = json.load(open(sys.argv[1])); sys.exit(0 if j[\"too_slow\"] and j[\"gb_per_s\"] < 0.1 else 1)" "$tmp/dl-slow.json"'
check "the partial file is deleted" '[[ ! -e "$tmp/dl-slow/x.gguf" ]]'
manifest 6000000 "$tmp/fast.json"
rc=0
PATH="$tmp/bin:$PATH" FAKE_SIZE=6000000 FAKE_RATE=150000 HF_GUARD_AFTER_S=2 HF_GUARD_END_BY=$(( $(date +%s) + 60 )) \
  HF_GUARD_REST_S=0 bash "$here/hf-download.sh" "$tmp/fast.json" "$tmp/dl-fast" "$tmp/dl-fast.json" > "$tmp/dl-fast.log" 2>&1 || rc=$?
sed 's/^/  /' "$tmp/dl-fast.log"
check "a download fast enough passes the guard and finishes, checked (exit 0)" \
  '(( rc == 0 )) && grep -q "^hf-download.sh: the guard after 2 s: ok, " "$tmp/dl-fast.log"'

echo "--- e07_report.py on a fixture E07_PLAN=e07b session (two-turn passed, bench-e07b timed out)"
f="$tmp/session/20261011T080000Z"
mkdir -p "$f/out/two-turn"
echo "plan: e07b, readers: dense" > "$f/out/plan.log"
cat > "$f/out/steps.log" <<'EOF'
step env: exit 0, 0 s
step gate: exit 0, 0 s
step dry-run: exit 0, 0 s
step two-turn: exit 0, 212 s
step bench-e07b: exit 124, 900 s (timeout after 900 s: 2 x its plan estimate of 450 s; min(2 x 450 s, the 2310 s left))
step dense-free: skipped, 12 s left
done
EOF
cat > "$f/out/env.log" <<'EOF'
name, memory.total [MiB], pcie.link.gen.max, pcie.link.width.max, pcie.link.gen.current, pcie.link.width.current, driver_version
NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, 4, 16, 1, 16, 580.65.06
host PCIe link (the host side and the GPU side, when the driver reports them):
pcie.link.gen.hostmax, pcie.link.gen.gpumax, pcie.link.width.max
4, 5, 16
EOF
cat > "$f/out/two-turn/two-turn.json" <<'EOF'
{"turns": [{"prompt_tokens": 16388, "completion_tokens": 128, "cached_tokens": 0, "client_s": 9.1, "first_token_s": null,
  "reply": "a", "engine_request": 1, "resume_kind": "cold", "reused": 0, "read": 16388, "prompt_ms": 4600.0,
  "trace": {"request": 1, "policy": "adaptive", "reset": 0, "resume": 0}},
 {"prompt_tokens": 16560, "completion_tokens": 128, "cached_tokens": 16515, "client_s": 3.4, "first_token_s": 0.9,
  "reply": "b", "engine_request": 2, "resume_kind": "live", "reused": 16515, "read": 45, "prompt_ms": 410.0,
  "decode": {"steps": 127, "hit_rate": 0.93, "ms_per_step": 16.0, "tok_s": 62.5},
  "trace": {"request": 2, "policy": "adaptive", "reset": 0, "resume": 1}}],
 "load_s": 151.0, "http": {"ok": true, "failed": [], "non_streaming": {"content": "Hi.", "usage": {}},
  "streaming": {"content": "Hi.", "usage": {}, "first_token_s": 0.2}, "same_ids": true},
 "failed": [], "turns_ok": true, "ok": true, "trace_requests": []}
EOF
python3 "$here/e07_report.py" "$f" "$tmp/sheets" | sed 's/^/  /'
sheet="$tmp/sheets/2026-10-11-e07b/README.md"
goal="$(sed -n '/^## Goal/,/^## Host/p' "$sheet" | grep -E '^\| [1-6] \|' || true)"
echo "$goal" | sed 's/^/  /'
check "the goal table has the rows 1 to 6" '[[ "$(cut -d"|" -f2 <<<"$goal" | tr -d " " | tr "\n" " ")" == "1 2 3 4 5 6 " ]]'
check "every row has PASS, FAIL or REPORTED" '! awk -F" [|] " "\$4 !~ /^(PASS|FAIL|REPORTED)$/" <<<"$goal" | grep -q .'
check "every FAIL row has a cause" '! awk -F" [|] " "\$4 == \"FAIL\" && \$5 ~ /^ *[|]? *$/" <<<"$goal" | grep -q .'
check "no row says only no result" '! grep -qE "\| no result \|" <<<"$goal"'
check "the bench rows say timeout after 900 s in bench-e07b" '[[ $(grep -c "timeout after 900 s in bench-e07b" <<<"$goal") -ge 2 ]]'
check "item 1 names the skipped dense-free step" 'grep -E "^\| 1 \|" <<<"$goal" | grep -q "dense-free skipped, 12 s left"'
check "item 3 without the bench's turn2 is REPORTED with its cause, the server turn as a note" \
  'grep -E "^\| 3 \|" <<<"$goal" | grep "| REPORTED |" | grep "turn2: no result (timeout after 900 s in bench-e07b" | grep -q "server turn only: first token 0.90 s"'
check "item 5 passes on the HTTP check" 'grep -E "^\| 5 \|" <<<"$goal" | grep -q "| PASS |"'
check "item 6 fails without cost.txt, with its cause" 'grep -E "^\| 6 \|" <<<"$goal" | grep -q "| FAIL | no cost.txt"'
check "the host table has the PCIe link row" \
  'grep -q "^| host PCIe link (generation, width) | Gen 4 x16 (link maximum; at env.s idle moment Gen 1 x16); the host.s maximum Gen 4, the GPU.s Gen 5 |$" "$sheet"'

check "item 1 and the sources line name this fixture's RTX PRO 6000" \
  'grep -E "^\| 1 \|" <<<"$goal" | grep -q "(NVIDIA RTX PRO 6000 Blackwell Workstation Edition)" &&
   grep -q "^Every number: the NVIDIA RTX PRO 6000 Blackwell Workstation Edition session of this sheet" "$sheet"'

echo "--- e07_report.py on a fixture E07_PLAN=e07b session on an A100 with a bench turn2 of 1.06 s"
g="$tmp/session/20261012T080000Z"
mkdir -p "$g/out"
cp -r "$f/out/two-turn" "$f/out/plan.log" "$g/out/"
cat > "$g/out/steps.log" <<'EOF2'
step env: exit 0, 0 s
step two-turn: exit 0, 212 s
step bench-e07b: exit 0, 700 s
done
EOF2
cat > "$g/out/env.log" <<'EOF2'
name, memory.total [MiB], pcie.link.gen.max, pcie.link.width.max, pcie.link.gen.current, pcie.link.width.current, driver_version
NVIDIA A100-SXM4-80GB, 81920 MiB, 4, 16, 4, 16, 580.126.16
EOF2
cat > "$g/out/bench-e07b.json" <<'EOF2'
{"load_s": 150.0, "ok": true, "alive_after": true, "phases": [
 {"phase": "turn2", "prompt_tokens": 16928, "reused": 16415, "resume": true, "prompt_ms": 1050.0,
  "ttft_client_ms": 1060.0, "prompt_tok_s": 488.5, "decode_tok_s": 30.6, "decode_steps": 31}]}
EOF2
python3 "$here/e07_report.py" "$g" "$tmp/sheets" | sed 's/^/  /'
sheet="$tmp/sheets/2026-10-12-e07b/README.md"
goal="$(sed -n '/^## Goal/,/^## Host/p' "$sheet" | grep -E '^\| [1-6] \|' || true)"
speed="$(sed -n '/^## Speed/,/^## /p' "$sheet" | grep -E '^\| turn2 \|' || true)"
printf '%s\n%s\n' "$goal" "$speed" | sed 's/^/  /'
check "a bench turn2 of 1.06 s is PASS in the goal table, the strict 1 s noted" \
  'grep -E "^\| 3 \|" <<<"$goal" | grep "| PASS |" | grep -q "the goal.s strict 1 s not met"'
check "the same turn2 is PASS in the Speed table, the strict 1 s noted" \
  'grep -q "| <= 1 s (accepted at about 2 s): PASS (within the accepted bar of about 2 s.*strict 1 s not met) |$" <<<"$speed"'
check "item 1 and the sources line name the A100, not the RTX PRO 6000" \
  'grep -E "^\| 1 \|" <<<"$goal" | grep -q "(NVIDIA A100-SXM4-80GB)" &&
   grep -q "^Every number: the NVIDIA A100-SXM4-80GB session of this sheet" "$sheet" && ! grep -q "RTX PRO 6000" "$sheet"'

(( failed == 0 )) || { echo "e07b-test.sh: $failed checks failed"; exit 1; }
echo "e07b-test.sh: all checks passed"
