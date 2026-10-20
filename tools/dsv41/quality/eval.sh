#!/usr/bin/env bash
# One system's quality run through api_proxy.py: the proxy on BACKEND in front of UPSTREAM, then each
# WHAT in order, then the proxy stops. Every request's usage goes to RUN_DIR/api_usage.jsonl.
#
#   bash tools/dsv41/quality/eval.sh BACKEND UPSTREAM RUN_DIR WHAT...
#
# BACKEND deepseek (UPSTREAM is the API's base URL, default https://api.deepseek.com, and DEEPSEEK_KEY_FILE the file
# with the key), llamacpp or strata (UPSTREAM the server's /v1/chat/completions). WHAT is an lm-eval task
# (gsm8k_cot_llama, ifeval, humaneval_instruct: run_lm_eval.sh on the fixed-seed subsets, its output in
# RUN_DIR/lm_eval/<task>/, the task's "rc=0" line in RUN_DIR/lm_eval_status.txt required) or needle-<length>
# (needle_check.py run on NEEDLE's prompt of that length, 16k, 64k or 256k, its result in
# RUN_DIR/needle-<length>.json) or continuation (continuation.py chat: the 3 HumanEval items' replies in
# RUN_DIR/continuation.json, for the hand check before the full run). Exits 1 when any WHAT fails, after running the
# others.
#
# Environment: QUALITY_VENV (the lm-eval venv, which also runs the proxy: it has aiohttp), NUM_CONCURRENT (default
# per backend: deepseek 32, strata 2; llamacpp 1, the fork's one slot
# by default, e07-run.sh passes its -np), SAMPLES (default the subsets file here),
# PROXY_PORT (default 8100), MODEL_NAME (default deepseek-flash for deepseek, deepseek-v4.1-flash otherwise), NEEDLE
# (the prompts of needle_check.py build). RUN_DIR/settings.txt records them.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend="${1:?backend}" upstream="${2:?upstream}" run_dir="${3:?run dir}"
shift 3
(( $# )) || { echo "eval.sh: nothing to run (lm-eval tasks or needle-<length>)" >&2; exit 2; }
VENV="${QUALITY_VENV:-/root/venv-quality}"
SAMPLES="${SAMPLES:-$here/subsets_lean2h_seed20261006.json}"
PROXY_PORT="${PROXY_PORT:-8100}"
case "$backend" in
  deepseek) conc=32 model=deepseek-flash ;;
  llamacpp) conc=1 model=deepseek-v4.1-flash ;;
  strata) conc=2 model=deepseek-v4.1-flash ;;
  *) echo "eval.sh: backend '$backend': deepseek, llamacpp or strata" >&2; exit 2 ;;
esac
NUM_CONCURRENT="${NUM_CONCURRENT:-$conc}"
MODEL_NAME="${MODEL_NAME:-$model}"
[[ -x "$VENV/bin/python" ]] || { echo "eval.sh: no venv at $VENV (QUALITY_VENV)" >&2; exit 2; }
for w in "$@"; do
  case "$w" in
    gsm8k_cot_llama|ifeval|humaneval_instruct) ;;
    continuation) ;;
    needle-16k|needle-64k|needle-256k) [[ -f "${NEEDLE:-}" ]] || { echo "eval.sh: $w needs NEEDLE (needle_check.py build)" >&2; exit 2; } ;;
    *) echo "eval.sh: '$w': gsm8k_cot_llama, ifeval, humaneval_instruct, continuation or needle-16k|64k|256k" >&2; exit 2 ;;
  esac
done
proxy=(--backend "$backend")
if [[ "$backend" == deepseek ]]; then
  # the key stays in its file, read by the proxy; never in an environment or a log
  [[ -n "${DEEPSEEK_KEY_FILE:-}" && -r "$DEEPSEEK_KEY_FILE" && -s "$DEEPSEEK_KEY_FILE" ]] ||
    { echo "eval.sh: the deepseek backend needs DEEPSEEK_KEY_FILE, a readable non-empty file with the key" >&2; exit 2; }
  proxy+=(--key-file "$DEEPSEEK_KEY_FILE" --api-base "$upstream")
else
  proxy+=(--upstream "$upstream")
fi
mkdir -p "$run_dir"
{
  echo "backend $backend"; echo "upstream $upstream"; echo "num_concurrent $NUM_CONCURRENT"; echo "model_name $MODEL_NAME"
  echo "samples $SAMPLES sha256 $(sha256sum "$SAMPLES" | cut -d' ' -f1)"
  [[ -z "${NEEDLE:-}" ]] || echo "needle $NEEDLE sha256 $(sha256sum "$NEEDLE" | cut -d' ' -f1)"
  echo "what $*"
} >> "$run_dir/settings.txt"
"$VENV/bin/python" "$here/api_proxy.py" "${proxy[@]}" --port "$PROXY_PORT" --usage "$run_dir/api_usage.jsonl" \
  >> "$run_dir/api_proxy.log" 2>&1 &
proxy_pid=$!
trap 'kill "$proxy_pid" 2>/dev/null; wait "$proxy_pid" 2>/dev/null' EXIT
for _ in $(seq 100); do
  kill -0 "$proxy_pid" 2>/dev/null || { echo "eval.sh: the proxy exited: $(tail -n 3 "$run_dir/api_proxy.log")" >&2; exit 1; }
  (echo > "/dev/tcp/127.0.0.1/$PROXY_PORT") 2>/dev/null && break
  sleep 0.2
done
url="http://127.0.0.1:$PROXY_PORT/v1/chat/completions"
rc=0
for w in "$@"; do
  if [[ "$w" == continuation ]]; then
    "$VENV/bin/python" "$here/continuation.py" chat "$url" --out "$run_dir/continuation.json" || rc=1
    continue
  fi
  if [[ "$w" == needle-* ]]; then
    "$VENV/bin/python" "$here/needle_check.py" run "$url" --prompts "$NEEDLE" --lengths "${w#needle-}" \
      --out "$run_dir/$w.json" || rc=1
    continue
  fi
  QUALITY_VENV="$VENV" SAMPLES="$SAMPLES" TASKS_OVERRIDE="$w" CHAT_URL="$url" MODEL_NAME="$MODEL_NAME" \
    NUM_CONCURRENT="$NUM_CONCURRENT" bash "$here/run_lm_eval.sh" "$run_dir"
  if [[ "$(grep "^$w rc=" "$run_dir/lm_eval_status.txt" 2>/dev/null | tail -n 1)" == "$w rc=0" ]]; then
    echo "eval.sh: $w done ($(find "$run_dir/lm_eval/$w" -name "samples_*.jsonl" | wc -l) samples file)"
  else
    echo "eval.sh: $w FAILED ($(grep "^$w " "$run_dir/lm_eval_status.txt" 2>/dev/null | tail -n 1 || echo "no status"); $run_dir/lm_eval_$w.log)"
    rc=1
  fi
done
exit "$rc"
