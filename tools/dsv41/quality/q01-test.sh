#!/usr/bin/env bash
# Tests the quality tooling without a GPU: no
# pod, no model, no money, every remote system a fakes.py stand-in.
#   1. analyze.py on the two vLLM-EXL3 experiment runs (the API as the API fresh column, EXL3 as the engine): the headline numbers
#      (API 0.960 / 0.853 / 0.982, EXL3 0.960 / 0.867 / 0.963) and the paired intervals and flips of that repo's
#      paper table paper/generated/quality_paired.tex, to its rounding, on engine - API fresh, the quality
#      bar's pair since 2026-10-12, PASS with the fork column "not run (dropped
#      2026-10-12)"; the same runs as --fork and --engine still give that table on engine - fork; then
#      analyze_test.py, the quality bar's PASS, FAIL, not evaluated and significant gain cases on synthetic runs
#   2. needle_check.py build: the 16K, 64K and 256K prompts within their contexts less 64 tokens
#   3. a 3-item lm-eval run per task through the proxy, llamacpp on fakes.py llama-server and strata on the real
#      serve/server.py with fakes.py's engine, --log_samples written for each task, then the needles (16K on the
#      fork, 16K, 64K and 256K on the engine), each answered
#   4. continuation_test.py: the HumanEval continuation through llamacpp, strata and deepseek, with its negative
#      control
#   5. api_run.sh against fakes.py deepseek-api with a fake key: refused without DEEPSEEK_KEY_FILE and with a key file
#      others can read; then the full run (464 requests, the needles, the cost, the analysis, the sheet) and no output
#      file holding the key's first 8 characters
#   6. tools/dsv41/pod/e07.sh --dry-run with E07_PLAN=qfork and qengine: qengine on either model path at 2.60 USD/h and qfork's sessions (two from the
#      volume, three from HF) at 1.95 USD/h exit 0 and print every step in the pod's order with its timeout, the
#      2026-10-06 tokens, the rates, the HF download at 0.385 GB/s and the USD; the
#      HumanEval-only qfork session plans -np 1, NUM_CONCURRENT 1 and the fork's speed-check after the continuation at
#      80 ms per token, the qualification first; the whole qfork plan exits 2 and names
#      the split; a slower forced rate (decode, or HF_GBS 0.115) that takes one over 3.00 USD exits 2; every other
#      E07_PLAN dry-runs byte for byte as before the quality tooling (commit 32f51ec, skipped when that commit is not
#      in this repository's history) but for the machine qualification's lines; qfork's payload takes the fork's llama-server and
#      the payload sources (the pre-flight) name serve/server.py's commit and the engine's and llama-server's sha256
#   7. e07_report.py on a qengine fixture: the sheet's quality sections, the qfork sheet's fork server peak GPU memory
#   8. fork_prompt_test.py on the REAL fork: a CPU-only llama-server ($FORK_CPU) on tiny-a.gguf with the real tokenizer
#      gets byte for byte the prompt text serve/server.py renders, for the 3 HumanEval continuations and a plain chat,
#      thinking off, and the llamacpp proxy cuts the fork's echo of the continued message;
#      skipped, said clearly, when that build is missing
#   9. q02-test.sh: the q plans' pod sessions rehearsed end to end (the real e07-run.sh from the real payloads, with
#      stubs for the GPU, the model, the network and the servers), qfork's 5 split sessions, its HumanEval-only one
#      and qengine on both model paths, each to steps.log's "done" with every planned step at exit 0
#     , and the slow engine's and slow fork's sessions stopped at speed-check
# Exits 1 if any check fails.
#
#   QUALITY_VENV=<a venv with lm-eval 0.4.13> bash tools/dsv41/quality/q01-test.sh
#
# QUALITY_VENV defaults to $BUILD_ROOT/quality/venv (run_lm_eval.sh creates it when missing: python3 -m venv and pip),
# VOCAB to $BUILD_ROOT/vocab/deepseek41-vocab.gguf, VLLM_EXL3 to $BUILD_ROOT/quality-ref (a checkout of the vLLM-EXL3
# experiment's repository, PROVENANCE.md), FORK_CPU to
# $BUILD_ROOT/fork-cpu/build/bin/llama-server (its build: tools/dsv41/engine/README.md), TINY_A to
# $BUILD_ROOT/engine/models/tiny-a.gguf.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
# e07.sh has no default network volume: its dry runs and rehearsals here plan one by this placeholder id
export NETWORK_VOLUME="${NETWORK_VOLUME-testvolume01}"
export QUALITY_VENV="${QUALITY_VENV:-$BUILD_ROOT/quality/venv}"
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"
VLLM_EXL3="${VLLM_EXL3:-$BUILD_ROOT/quality-ref}"
export VLLM_EXL3   # analyze.py reads its reference runs there too
FORK_CPU="${FORK_CPU:-$BUILD_ROOT/fork-cpu/build/bin/llama-server}"
TINY_A="${TINY_A:-$BUILD_ROOT/engine/models/tiny-a.gguf}"
tmp="$(mktemp -d)"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; rm -rf -- "${tmp:?}"; }
trap cleanup EXIT
failed=0
check() {
  if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; failed=$((failed + 1)); fi
}
free_port() { python3 -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])"; }
wait_port() { for _ in $(seq 300); do (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null && return 0; sleep 0.2; done; return 1; }
[[ -x "$QUALITY_VENV/bin/lm_eval" ]] || INSTALL_ONLY=1 bash "$here/run_lm_eval.sh" "$tmp/install"

echo "--- 1. analyze.py on the vLLM-EXL3 experiment runs against its paper table"
d="$VLLM_EXL3/data/controlled"
e11="$d/E11-quality-reference/api/20261006T181137Z" e10="$d/E10-quality-exl3/exl3-offload-gen30-lean-v2/20261006T211059Z"
python3 "$here/analyze.py" --api-fresh "$e11" --engine "$e10" --json "$tmp/analysis.json" > "$tmp/analysis.md"
python3 "$here/analyze.py" --fork "$e11" --engine "$e10" --json "$tmp/analysis-fork.json" > "$tmp/analysis-fork.md"
check "headlines: API 0.960 / 0.853 / 0.982 (API fresh column), EXL3 0.960 / 0.867 / 0.963 (engine column)" \
  'python3 - "$tmp/analysis.json" <<EOF
import json, sys
h = {(x["column"], x["task"]): x["value"] for x in json.load(open(sys.argv[1]))["headline"]}
want = {"api_fresh": (0.960, 0.853, 0.982), "engine": (0.960, 0.867, 0.963)}
tasks = ("gsm8k_cot_llama", "ifeval", "humaneval_instruct")
sys.exit(0 if all(round(h[(c, t)], 3) == v for c, vs in want.items() for t, v in zip(tasks, vs)) else 1)
EOF'
for s in "analysis|api_fresh|engine - API fresh" "analysis-fork|fork|engine - fork"; do
  IFS='|' read -r f a name <<<"$s"
  check "$name: paired intervals and flips equal quality_paired.tex (E10 minus E11) to its rounding" \
    'python3 - "$tmp/$f.json" "$VLLM_EXL3/paper/generated/quality_paired.tex" "$a" <<EOF
import json, re, sys
pairs = {p["task"]: p["result"] for p in json.load(open(sys.argv[1]))["pairs"] if (p["a"], p["b"]) == (sys.argv[3], "engine")}
rows = re.findall(r"^(\S+) & \S+ & (\d+) & ([\d.]+) & ([\d.]+) & ([+-][\d.]+) & \[([+-][\d.]+), ([+-][\d.]+)\] & ([\d.]+) ",
                  open(sys.argv[2]).read(), re.M)
assert len(rows) == 3, rows
for task, n, e10, e11, diff, lo, hi, flips in rows:
    r = pairs[task.replace("\\\\_", "_")]
    got = (str(r["n"]), "{:.3f}".format(r["mean_b"]), "{:.3f}".format(r["mean_a"]), "{:+.3f}".format(r["diff"]),
           "{:+.3f}".format(r["ci95"][0]), "{:+.3f}".format(r["ci95"][1]), "{:.1f}".format(100 * r["discordant"] / r["n"]))
    print("  ", task, got)
    assert got == (n, e10, e11, diff, lo, hi, flips), (task, got)
EOF'
done
check "the quality bar on engine - API fresh is PASS, the fork column reads 'not run (dropped 2026-10-12)'" \
  'grep -q "^Quality bar (engine - API fresh, since 2026-10-12: .*\*\*PASS\*\*" "$tmp/analysis.md" &&
   grep -q "^| ifeval | prompt_level_strict_acc (none) | 0.853 | not run (dropped 2026-10-12) | 0.867 |" "$tmp/analysis.md"'
check "with --fork and no API fresh: the quality bar not evaluated, engine - fork called information only" \
  'grep -q "\*\*not evaluated\*\* (no API fresh run)" "$tmp/analysis-fork.md" &&
   grep -qx "engine - fork is information only, not the quality bar." "$tmp/analysis-fork.md"'
python3 "$here/analyze_test.py" > "$tmp/analyze-test.log" 2>&1 && rc=0 || rc=$?
grep -E "^(FAIL|analyze_test.py)" "$tmp/analyze-test.log" | sed 's/^/  /'
check "analyze_test.py: the quality bar's PASS, FAIL, not evaluated and gain cases and the fork column ($(grep -c "^ok" "$tmp/analyze-test.log") checks)" \
  '(( rc == 0 ))'

echo "--- 2. needle_check.py build"
python3 "$here/needle_check.py" build --vocab "$VOCAB" --out "$tmp/needle.json" | sed 's/^/  /'
check "16K, 64K and 256K prompts within their context less 64 tokens, one 7-digit number each" \
  'python3 -c "import json, sys; p = json.load(open(sys.argv[1]))[\"prompts\"]
sys.exit(0 if [x[\"name\"] for x in p] == [\"16k\", \"64k\", \"256k\"] and all(x[\"context\"] - 64 - 40 < x[\"prompt_tokens\"] <= x[\"context\"] - 64 and len(x[\"answer\"]) == 7 for x in p) else 1)" "$tmp/needle.json"'

echo "--- 3. 3-item lm-eval runs through the proxy on the fakes"
python3 -c "import json, sys; d = json.load(open(sys.argv[1])); json.dump({k: v[:3] for k, v in d.items()}, open(sys.argv[2], 'w'))" \
  "$here/subsets_lean2h_seed20261006.json" "$tmp/samples3.json"
lp="$(free_port)"
python3 "$here/fakes.py" llama-server --tokenizer-gguf "$VOCAB" --record "$tmp/llama.jsonl" --port "$lp" -np 4 \
  --prefill-assistant > "$tmp/llama.log" 2>&1 &
pids+=($!)
sp="$(free_port)"
python3 -c "import json, sys; json.dump({'exe': sys.argv[1], 'args': ['--tokenizer-gguf', sys.argv[2], '--record', sys.argv[3],
  '--ctx', '262144'], 'model_profile': 'deepseek-v4.1', 'tokenizer_gguf': sys.argv[2], 'log': sys.argv[4],
  'repeat_stop_tokens': 0}, open(sys.argv[5], 'w'))" "$here/fakes.py" "$VOCAB" "$tmp/engine.jsonl" "$tmp/engine.log" "$tmp/strata.json"
(cd "$STRATA_DIR" && exec python3 -m serve.server --engine strata --config "$tmp/strata.json" --port "$sp") > "$tmp/strata.log" 2>&1 &
pids+=($!)
wait_port "$lp" && wait_port "$sp" || { echo "FAIL the fake servers did not start"; tail -n 5 "$tmp"/*.log; exit 1; }
for b in llamacpp strata; do
  port="$([[ $b == llamacpp ]] && echo "$lp" || echo "$sp")"
  needles="needle-16k"; [[ $b == strata ]] && needles="needle-16k needle-64k needle-256k"
  # shellcheck disable=SC2086  # the needles are whole words
  SAMPLES="$tmp/samples3.json" NEEDLE="$tmp/needle.json" PROXY_PORT="$(free_port)" bash "$here/eval.sh" "$b" \
    "http://127.0.0.1:$port/v1/chat/completions" "$tmp/run-$b" gsm8k_cot_llama ifeval humaneval_instruct $needles \
    > "$tmp/eval-$b.log" 2>&1 && rc=0 || rc=$?
  grep -E "^(eval.sh|ok|FAIL|needle_check)" "$tmp/eval-$b.log" | sed 's/^/  /'
  check "$b: eval.sh exits 0" '(( rc == 0 ))'
  for t in gsm8k_cot_llama ifeval humaneval_instruct; do
    check "$b: $t answered for its 3 items, --log_samples written" \
      'python3 -c "import json, sys, glob
f = glob.glob(sys.argv[1] + \"/lm_eval/\" + sys.argv[2] + \"/*/samples_\" + sys.argv[2] + \"_*.jsonl\")
ids = {json.loads(x)[\"doc_id\"] for x in open(f[0])} if len(f) == 1 else set()
sys.exit(0 if ids == set(json.load(open(sys.argv[3]))[sys.argv[2]]) else 1)" "$tmp/run-$b" "$t" "$tmp/samples3.json"'
  done
  for n in $needles; do
    check "$b: $n answered with the number" 'python3 -c "import json, sys; r = json.load(open(sys.argv[1]))[\"results\"]
sys.exit(0 if r and all(x[\"exact\"] for x in r) else 1)" "$tmp/run-$b/$n.json"'
  done
done
check "strata: every request reached the engine as GEN lines (3 tasks x 3 items + 3 needles)" \
  '(( $(wc -l < "$tmp/engine.jsonl") >= 12 ))'

echo "--- 4. continuation_test.py"
python3 "$here/continuation_test.py" --vocab "$VOCAB" --venv "$QUALITY_VENV" --out "$tmp/continuation" > "$tmp/cont.log" 2>&1 && rc=0 || rc=$?
grep -E "^(FAIL|continuation_test.py|  the new turn)" "$tmp/cont.log" | sed 's/^/  /'
check "continuation through llamacpp, strata and deepseek, the negative control caught ($(grep -c "^ok" "$tmp/cont.log") checks)" '(( rc == 0 ))'

echo "--- 5. api_run.sh against fakes.py deepseek-api"
printf 'sk-q01fake%s\n' "$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')" > "$tmp/key"
chmod 600 "$tmp/key"
ap="$(free_port)"
python3 "$here/fakes.py" deepseek-api --key-file "$tmp/key" --record "$tmp/api.jsonl" --port "$ap" > "$tmp/api.log" 2>&1 &
pids+=($!)
wait_port "$ap"
api_env=(API_BASE="http://127.0.0.1:$ap" PRICE_URL="file://$here/deepseek-api-pricing-2026-10-06.html" NEEDLE="$tmp/needle.json"
         SHEET_ROOT="$tmp/sheets" PROXY_PORT="$(free_port)")
rc=0; env -u DEEPSEEK_KEY_FILE "${api_env[@]}" RUN="$tmp/api-nokey" bash "$here/api_run.sh" > "$tmp/api-nokey.log" 2>&1 || rc=$?
check "without DEEPSEEK_KEY_FILE: exit 2, said clearly" '(( rc == 2 )) && grep -q "DEEPSEEK_KEY_FILE is not set" "$tmp/api-nokey.log"'
cp "$tmp/key" "$tmp/key-open"; chmod 644 "$tmp/key-open"
rc=0; env "${api_env[@]}" DEEPSEEK_KEY_FILE="$tmp/key-open" RUN="$tmp/api-open" bash "$here/api_run.sh" > "$tmp/api-open.log" 2>&1 || rc=$?
check "with a key file others can read: exit 2, said clearly" '(( rc == 2 )) && grep -q "readable by group or others" "$tmp/api-open.log"'
run="$tmp/api/20261011T120000Z"
rc=0; env "${api_env[@]}" DEEPSEEK_KEY_FILE="$tmp/key" RUN="$run" bash "$here/api_run.sh" > "$tmp/api-run.log" 2>&1 || rc=$?
grep -E "^(api_run.sh|api_cost.py|eval.sh|ok|FAIL|key scan)" "$tmp/api-run.log" | sed 's/^/  /'
check "the run exits 0" '(( rc == 0 ))'
check "464 lm-eval requests and 3 needles reached the fake API, HumanEval's 164 through /beta" \
  'python3 -c "import json, sys; r = [json.loads(x) for x in open(sys.argv[1])]
sys.exit(0 if len(r) == 467 and sum(x[\"beta\"] for x in r) == 164 else 1)" "$tmp/api.jsonl"'
check "cost.json, analysis.md and the sheet written" \
  '[[ -s "$run/cost.json" && -s "$run/analysis.md" && -s "$tmp/sheets/2026-10-11-qapi/README.md" ]]'
check "the key scan passed" 'grep -q "^key scan: no file of the run or the sheet holds the key" "$tmp/api-run.log"'
head8="$(head -c 8 "$tmp/key")"
check "no file of the test (logs, runs, sheets) holds the key's first 8 characters but the key file" \
  '! grep -rlF -- "$head8" "$tmp" | grep -vx -e "$tmp/key" -e "$tmp/key-open" | grep -q .'

echo "--- 6. e07.sh --dry-run, E07_PLAN=qfork and qengine"
dry() { env "$@" OUT=/tmp/e07-dry-out QUALITY_NEEDLE="$tmp/needle.json" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run; }
# qengine one session at 2.60 USD/h on either model path; qfork's sessions at 1.95 USD/h,
# two from the volume (HF_CLOUDS empty) and three from HF (NETWORK_VOLUME empty)
sessions=("qengine-volume|E07_PLAN=qengine HF_CLOUDS=|2.60" "qengine-hf|E07_PLAN=qengine NETWORK_VOLUME=|2.60"
  "qfork-volume-1|E07_PLAN=qfork Q_TASKS=gsm8k_cot_llama,humaneval_instruct HF_CLOUDS=|1.95"
  "qfork-volume-2|E07_PLAN=qfork Q_TASKS=ifeval Q_NEEDLES= HF_CLOUDS=|1.95"
  "qfork-hf-1|E07_PLAN=qfork Q_TASKS=gsm8k_cot_llama Q_NEEDLES= NETWORK_VOLUME=|1.95"
  "qfork-hf-2|E07_PLAN=qfork Q_TASKS=humaneval_instruct NETWORK_VOLUME=|1.95"
  "qfork-hf-3|E07_PLAN=qfork Q_TASKS=ifeval Q_NEEDLES= NETWORK_VOLUME=|1.95"
  "qfork-he|E07_PLAN=qfork Q_TASKS=humaneval_instruct Q_NEEDLES= NETWORK_VOLUME=|1.95")
for s in "${sessions[@]}"; do
  IFS='|' read -r name vars usd <<<"$s"
  path="$([[ $name == *-hf* ]] && echo hf || echo volume)"
  # shellcheck disable=SC2086  # the session's variables are whole words
  rc=0; dry $vars > "$tmp/$name.txt" 2>&1 || rc=$?
  grep -E "^plan \($path\)" "$tmp/$name.txt" | sed "s/^/  $name: /"
  check "$name ($vars) exits 0 at $usd USD/h" '(( rc == 0 )) && grep -qE "^plan \($path\): .* USD at $usd USD/h\)" "$tmp/$name.txt"'
done
check "qfork prints the 2026-10-06 tokens (219,916 prompt, 80,741 completion) and the rates (115, 20 tok/s)" \
  'grep -q "219,916 prompt and 80,741 completion" "$tmp/qfork-hf-1.txt" && grep -q "prompts 115 tok/s" "$tmp/qfork-hf-1.txt" &&
   grep -q "decode 20 tok/s" "$tmp/qfork-hf-1.txt"'
check "qengine prints the engine's 60 tok/s decode" 'grep -q "decode 60 tok/s" "$tmp/qengine-volume.txt"'
check "the Part Q plans plan the HF download at 0.385 GB/s, and the guard lets it run" \
  '(for n in qengine-hf qfork-hf-1 qfork-hf-2 qfork-hf-3; do grep -q "model-download (264.5 GB at 0.385 GB/s" "$tmp/$n.txt" &&
   grep -q "the guard lets it run" "$tmp/$n.txt" || exit 1; done)'
check "qfork's HF steps in the pod's order, each with its timeout" \
  'sed -n "/^plan with the model from HF/,/^plan (hf)/p" "$tmp/qfork-hf-2.txt" | grep -oE "^  (setup|model-download|fork-server|quality-venv|continuation|speed-check|lm-eval-[a-z0-9_]+|needle-16k|server-stop)\b.*timeout min" |
   sed -E "s/^  ([^ ]+).*/\1/" | tr "\n" " " | grep -qx "setup model-download fork-server quality-venv continuation speed-check lm-eval-humaneval_instruct needle-16k server-stop "'
# the HumanEval-only session at one slot, with the fork's speed check after the
# continuation
check "qfork-he: -np 1, NUM_CONCURRENT 1, speed-check after continuation at 80 ms per token, the qualification first" \
  'grep -q "^    --numa numactl) and --ctx-size 32768 -np 1 (Q_FORK_NP, e07/fork-np) " "$tmp/qfork-he.txt" &&
   grep -q "^  lm-eval-<task> for each of humaneval_instruct: eval.sh (llamacpp, NUM_CONCURRENT 1, " "$tmp/qfork-he.txt" &&
   grep -q "^  speed-check: pod/speed_check.py out/quality-server.log --fork --max-ms 80 " "$tmp/qfork-he.txt" &&
   sed -n "/^plan with the model from HF/,/^plan (hf)/p" "$tmp/qfork-he.txt" | awk "/^  / { print \$1 }" | tr "\n" " " |
     grep -q "^create, payload env setup model-download fork-server quality-venv continuation speed-check lm-eval-humaneval_instruct server-stop pull $" &&
   grep -q "^  speed-check (speed_check.py --fork on the continuation.s replies, at most 80 ms per generated token) " "$tmp/qfork-he.txt" &&
   grep -q "^  lm-eval-humaneval_instruct (164 items, NUM_CONCURRENT 1: " "$tmp/qfork-he.txt"'
check "qengine's volume steps in the pod's order, each with its timeout" \
  'sed -n "/^plan with the model from the volume/,/^plan (volume)/p" "$tmp/qengine-volume.txt" | grep -oE "^  (gate|dry-run|engine-server|quality-venv|continuation|lm-eval-[a-z0-9_]+|needle-[0-9]+k|server-stop)\b.*timeout min" |
   sed -E "s/^  ([^ ]+).*/\1/" | tr "\n" " " | grep -qx "gate dry-run engine-server quality-venv continuation lm-eval-gsm8k_cot_llama lm-eval-ifeval lm-eval-humaneval_instruct needle-16k needle-64k needle-256k server-stop "'
check "qfork's payload takes the fork's llama-server; the payload sources name server.py's commit, the engine's and the server's sha256" \
  'grep -qE "^    fork/: .* llama-server libllama-server-impl.so libmtmd.so.0 from " "$tmp/qfork-hf-2.txt" &&
   grep -q "HEAD contains 2c43875 (the continuation) and 9ce8dc5 (the echo cut)" "$tmp/qengine-volume.txt" &&
   grep -qE "^  src/serve/server.py: last changed in [0-9a-f]+ " "$tmp/qengine-volume.txt" &&
   grep -qE "^  payload-cu130 dsv41-engine: sha256 [0-9a-f]{64} " "$tmp/qengine-volume.txt" &&
   grep -qE "^    llama-server: sha256 [0-9a-f]{64}$" "$tmp/qfork-hf-2.txt"'
rc=0; dry E07_PLAN=qfork > "$tmp/qfork-whole.txt" 2>&1 || rc=$?
grep -E "^plan \((volume|hf)\)" "$tmp/qfork-whole.txt" | sed 's/^/  qfork, the whole plan: /'
check "the whole qfork plan: over 3.00 USD on both paths, exit 2, the split of each path named" \
  '(( rc == 2 )) && grep -q "from the volume, 2 sessions: Q_TASKS=gsm8k_cot_llama,humaneval_instruct" "$tmp/qfork-whole.txt" &&
   grep -q "from HF (NETWORK_VOLUME=), 3 sessions: Q_TASKS=gsm8k_cot_llama Q_NEEDLES=," "$tmp/qfork-whole.txt"'
check "the over-cap line names each path once with its reasons" \
  'grep -qx "e07.sh: the plan is over the cap with the model from: volume (the 92 min cap, 3.00 USD) hf (the 92 min cap, 3.00 USD)" "$tmp/qfork-whole.txt"'
for s in "qfork-slow|E07_PLAN=qfork Q_TASKS=gsm8k_cot_llama Q_DECODE_TOK_S=5" "qengine-slow|E07_PLAN=qengine HF_CLOUDS= Q_DECODE_TOK_S=20" \
         "qengine-slow-hf|E07_PLAN=qengine NETWORK_VOLUME= HF_GBS=0.115"; do
  IFS='|' read -r name vars <<<"$s"
  # shellcheck disable=SC2086
  rc=0; dry $vars > "$tmp/$name.txt" 2>&1 || rc=$?
  check "$name ($vars): over 3.00 USD, exit 2" '(( rc == 2 )) && grep -q "costs more than 3.00 USD" "$tmp/$name.txt"'
done
# e07.sh before the quality tooling; a repository without that commit in its history skips the comparison
if git -C "$STRATA_DIR" cat-file -e "32f51ec^{commit}" 2>/dev/null; then
git -C "$STRATA_DIR" archive 32f51ec tools | tar -x -C "$tmp" --one-top-level=base
# less the machine qualification, which every plan runs since: its lines
# ("  qualification") and the first step's label, which names it
for p in cycle2 full e08 e09 e07b; do
  a=0 b=0
  E07_PLAN=$p OUT=/tmp/e07-dry-out bash "$tmp/base/tools/dsv41/pod/e07.sh" --dry-run 2>&1 | sed "s|$tmp/base|STRATA|g" |
    grep -v "^  create, SSH and the CUDA probe (session 1: 0.2) " > "$tmp/before-$p.txt" || a=$?
  E07_PLAN=$p OUT=/tmp/e07-dry-out bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run 2>&1 | sed "s|$STRATA_DIR|STRATA|g" \
    > "$tmp/after-$p-all.txt" || b=$?
  grep -v -e "^  qualification" -e "^  create, SSH, CUDA probe, qualification (session 1: 0.2) " "$tmp/after-$p-all.txt" > "$tmp/after-$p.txt" || true
  check "E07_PLAN=$p dry-runs as before the quality tooling but for the qualification ($(wc -l < "$tmp/after-$p.txt") lines)" \
    'diff -q "$tmp/before-$p.txt" "$tmp/after-$p.txt" > /dev/null && grep -q "^  qualification, when" "$tmp/after-$p-all.txt"'
done
else
  echo "SKIP: commit 32f51ec (e07.sh before the quality tooling) is not in this repository's history"
fi

echo "--- 7. e07_report.py on a qengine fixture"
f="$tmp/report/20261012T080000Z"
mkdir -p "$f/out"
cp -r "$tmp/run-strata" "$f/out/quality"
echo "plan: qengine, readers: dense" > "$f/out/plan.log"
printf 'step engine-server: exit 0, 250 s\nstep lm-eval-ifeval: exit 0, 900 s\ndone\n' > "$f/out/steps.log"
printf 'ok   HumanEval/0: the reply continues the code\ncontinuation.py chat: PASS, 3 of 3 items\n' > "$f/out/continuation.log"
echo "python3 -m serve.server --engine strata --config out/quality-server.json --port 8090" > "$f/out/quality-server.cmd"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/report/sheets" > /dev/null
r="$tmp/report/sheets/2026-10-12-qengine/README.md"
check "the qengine sheet: the server as run, the continuation, each task's headline, the needles, the hand check" \
  'grep -q "^## Server as run" "$r" && grep -q "continuation.py chat: PASS" "$r" && grep -q "^| ifeval | prompt_level_strict_acc" "$r" &&
   grep -q "^| 256k | 262144 |" "$r" && grep -q "for the hand check of the continuation" "$r"'
f="$tmp/report/20261012T090000Z"
mkdir -p "$f/out"
cp -r "$tmp/run-llamacpp" "$f/out/quality"
echo "plan: qfork, readers: " > "$f/out/plan.log"
printf 'step fork-server: exit 0, 400 s\ndone\n' > "$f/out/steps.log"
printf '2026/10/12 09:01:00.000, 61234, 81920\n2026/10/12 09:01:01.000, 70411, 81920\n2026/10/12 09:01:02.000, 65000, 81920\n' \
  > "$f/out/quality-gpu.csv"
python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" "$f" "$tmp/report/sheets" > /dev/null
check "the qfork sheet: the fork server's peak GPU memory from its 1 s samples" \
  'grep -q "^Fork server peak GPU memory: 70411 MiB used of 81920 MiB, 11509 MiB free at the peak (out/quality-gpu.csv, nvidia-smi every 1 s, 3 samples)" \
   "$tmp/report/sheets/2026-10-12-qfork/README.md"'

echo "--- 8. fork_prompt_test.py on the real CPU llama-server"
if [[ -x "$FORK_CPU" ]]; then
  python3 "$here/fork_prompt_test.py" --llama-server "$FORK_CPU" --model "$TINY_A" --vocab "$VOCAB" --venv "$QUALITY_VENV" \
    --out "$tmp/fork-prompt" > "$tmp/fork-prompt.log" 2>&1 && rc=0 || rc=$?
  grep -E "^(FAIL|fork_prompt_test.py|  )" "$tmp/fork-prompt.log" | sed 's/^/  /'
  check "the fork and Strata's server get byte-identical prompts, the proxy cuts the echo ($(grep -c "^ok" "$tmp/fork-prompt.log") checks)" \
    '(( rc == 0 ))'
else
  echo "SKIP 8: no CPU-only fork build at $FORK_CPU (build it as tools/dsv41/engine/README.md says, or set FORK_CPU)"
fi

echo "--- 9. q02-test.sh: the q plans' pod sessions rehearsed"
QUALITY_NEEDLE="$tmp/needle.json" bash "$here/q02-test.sh" > "$tmp/q02.log" 2>&1 && rc=0 || rc=$?
grep -E "^(FAIL|q02-test.sh|  the 7 sessions)" "$tmp/q02.log" | sed 's/^/  /'
check "the 7 rehearsed sessions reach done with every planned step at exit 0 ($(grep -c "^ok" "$tmp/q02.log") checks)" '(( rc == 0 ))'

echo "q01-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
