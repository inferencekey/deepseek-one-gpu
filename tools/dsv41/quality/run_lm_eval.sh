#!/usr/bin/env bash
# Run the pinned lm-evaluation-harness tasks against an OpenAI-compatible chat endpoint. Default: the server started by
# experiments/lib.sh on 127.0.0.1:$PORT. Output: <run_dir>/lm_eval/<task>/ (results JSON and per-sample logs).
#
#   TASKS="gsm8k_cot_llama ifeval humaneval_instruct" bash quality/run_lm_eval.sh <run_dir>
#   SAMPLES=quality/subsets_lean2h_seed20261006.json   fixed-seed item subsets (lean plan); incompatible with LIMIT
#   TIME_CAP_S=900                                    wall-clock cap for all tasks together (lean plan)
#   CHAT_URL=http://127.0.0.1:8100/v1/chat/completions MODEL_NAME=deepseek-flash   e.g. the API proxy (E11 lean)
#
# Strata (tools/dsv41/quality/PROVENANCE.md, the vLLM-EXL3 experiment at 40202fd): lm-eval's version is pinned here (the source read
# it from its stack/pins.env); QUALITY_VENV is the venv (default /root/venv-quality, created with lm-eval when it has
# none, with virtualenv when python3's venv has no ensurepip); INSTALL_ONLY=1 creates it and exits (the pod plans install it while the model loads); the venv's packages
# go to <run_dir>/lm_eval_pip_freeze.txt with pip, or uv for a venv made by uv (no pip in it).
set -uo pipefail
D=${1:?run dir}
PORT=${PORT:-8000}
LM_EVAL_VERSION=0.4.13
TASKS=${TASKS:-"gsm8k_cot_llama ifeval humaneval_instruct mbpp_instruct minerva_math500"}
[[ ${MINIMAL:-0} == 1 || -n ${SAMPLES:-} ]] && TASKS=${TASKS_OVERRIDE:-"gsm8k_cot_llama ifeval humaneval_instruct"}
CHAT_URL=${CHAT_URL:-http://127.0.0.1:$PORT/v1/chat/completions}
MODEL_NAME=${MODEL_NAME:-${SERVED:-deepseek-v4.1-flash}}
VENV=${QUALITY_VENV:-/root/venv-quality}
[[ -x $VENV/bin/lm_eval ]] || { { python3 -m venv "$VENV" || python3 -m virtualenv "$VENV"; } &&
  "$VENV/bin/pip" install -q "lm-eval[api,ifeval,ruler,math]==$LM_EVAL_VERSION"; }
[[ -x $VENV/bin/lm_eval ]] || { echo "run_lm_eval.sh: no lm_eval in $VENV" >&2; exit 1; }
[[ ${INSTALL_ONLY:-0} == 1 ]] && exit 0
mkdir -p "$D"
if [[ -x $VENV/bin/pip ]]; then "$VENV/bin/pip" freeze; else uv pip freeze --python "$VENV/bin/python"; fi > "$D/lm_eval_pip_freeze.txt"
export OPENAI_API_KEY=${OPENAI_API_KEY:-${API_KEY:-none}} HF_ALLOW_CODE_EVAL=1
t_end=$(( $(date +%s) + ${TIME_CAP_S:-86400} ))
for t in $TASKS; do
  left=$(( t_end - $(date +%s) ))
  (( left > 60 )) || { echo "$t skipped: time cap" >> "$D/lm_eval_status.txt"; continue; }
  maxtok=1024; case $t in ifeval) maxtok=1280;; minerva_math500) maxtok=2048;; esac
  extra=(); case $t in humaneval_instruct|mbpp_instruct) extra+=(--confirm_run_unsafe_code);; esac
  if [[ -n ${SAMPLES:-} ]]; then
    extra+=(--samples "$(python3 -c "import json,sys; d=json.load(open('$SAMPLES')); print(json.dumps({'$t': d['$t']}))")")
  elif [[ -n ${LIMIT:-} ]]; then extra+=(--limit "$LIMIT"); fi
  timeout "$left" "$VENV/bin/lm_eval" run --model local-chat-completions \
    --model_args "model=$MODEL_NAME,base_url=$CHAT_URL,num_concurrent=${NUM_CONCURRENT:-32},max_retries=3,tokenized_requests=False" \
    --tasks "$t" --apply_chat_template --seed 1234,1234,1234,1234 --log_samples \
    --gen_kwargs "temperature=0,max_gen_toks=$maxtok" "${extra[@]}" \
    --output_path "$D/lm_eval/$t" > "$D/lm_eval_$t.log" 2>&1
  echo "$t rc=$?" >> "$D/lm_eval_status.txt"
done
