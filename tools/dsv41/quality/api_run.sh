#!/usr/bin/env bash
# The fresh DeepSeek API run, from a dev machine (no GPU needed). It costs money (the
# 2026-10-06 run's tokens plus the needle prompts: about 0.1-0.25 USD); run it only with the approved budget.
#
#   DEEPSEEK_KEY_FILE=<the 0600 key file> bash tools/dsv41/quality/api_run.sh
#
# In order: the key file is checked (readable, not empty, no access for group or others;
# the key is only ever read by api_proxy.py, never echoed, logged, written or exported); the price page is fetched
# (PRICE_URL, default https://api-docs.deepseek.com/quick_start/pricing) into RUN/price-page.html, and the run is
# estimated with its peak prices on the 2026-10-06 run's tokens (464 requests, 219,916 prompt and 80,741 completion
# tokens, every prompt token a cache miss) plus the needle prompts' (exit 2 when over MAX_USD before anything is
# sent); the needle prompts (NEEDLE, built by needle_check.py from VOCAB when missing); eval.sh with the deepseek
# backend: gsm8k_cot_llama, ifeval and humaneval_instruct (32 concurrent, as the vLLM-EXL3 experiment's API run), then needle-16k, -64k and
# -256k; api_cost.py on RUN/api_usage.jsonl and the page (RUN/cost.json); analyze.py, API fresh against the API on
# 2026-10-06 (RUN/analysis.md); q_report.py's sheet in SHEET_ROOT/<UTC date>-qapi/; and a scan of RUN and the sheet
# for the key's first 8 characters (exit 1 if any file holds them). Exits 1 when a task or a needle failed or the
# measured cost is over MAX_USD.
#
# Environment: DEEPSEEK_KEY_FILE (required), API_BASE (default https://api.deepseek.com; the tests point it at
# fakes.py deepseek-api), PRICE_URL, MAX_USD (default 0.50), QUALITY_VENV (default $BUILD_ROOT/quality/venv, made by
# run_lm_eval.sh when missing), BUILD_ROOT (default $HOME/dsv41-build), VOCAB (default
# $BUILD_ROOT/vocab/deepseek41-vocab.gguf), NEEDLE (default $BUILD_ROOT/quality/needle.json), RUN (default
# $BUILD_ROOT/quality/api/<UTC time>), SHEET_ROOT (default $BUILD_ROOT/sheets), PROXY_PORT,
# API_THEN_USAGE (default $BUILD_ROOT/quality/api_usage.jsonl: the 2026-10-06 API run's usage log, the estimate's
# tokens; data/controlled/E11-quality-reference/api/20261006T181137Z/api_usage.jsonl of the vLLM-EXL3 experiment).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
API_BASE="${API_BASE:-https://api.deepseek.com}"
PRICE_URL="${PRICE_URL:-https://api-docs.deepseek.com/quick_start/pricing}"
MAX_USD="${MAX_USD:-0.50}"
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"
NEEDLE="${NEEDLE:-$BUILD_ROOT/quality/needle.json}"
RUN="${RUN:-$BUILD_ROOT/quality/api/$(date -u +%Y%m%dT%H%M%SZ)}"
SHEET_ROOT="${SHEET_ROOT:-$BUILD_ROOT/sheets}"
export QUALITY_VENV="${QUALITY_VENV:-$BUILD_ROOT/quality/venv}"
API_THEN_USAGE="${API_THEN_USAGE:-$BUILD_ROOT/quality/api_usage.jsonl}"

key="${DEEPSEEK_KEY_FILE:-}"
[[ -n "$key" ]] || { echo "api_run.sh: DEEPSEEK_KEY_FILE is not set (the path of the file that holds the key)" >&2; exit 2; }
[[ -f "$key" && -r "$key" && -s "$key" ]] || { echo "api_run.sh: DEEPSEEK_KEY_FILE=$key is not a readable, non-empty file" >&2; exit 2; }
[[ "$(stat -c %a "$key")" =~ ^[4-7]00$ ]] ||
  { echo "api_run.sh: $key is readable by group or others (mode $(stat -c %a "$key")): chmod 600 it" >&2; exit 2; }
export DEEPSEEK_KEY_FILE="$key"
mkdir -p "$RUN"
echo "api_run.sh: run directory $RUN"
curl -sSfL --max-time 60 "$PRICE_URL" -o "$RUN/price-page.html" ||
  { echo "api_run.sh: fetching the price page $PRICE_URL failed" >&2; exit 1; }
echo "price page $PRICE_URL fetched $(date -u +%FT%TZ) sha256 $(sha256sum "$RUN/price-page.html" | cut -d' ' -f1)" > "$RUN/price-page.txt"
[[ -f "$NEEDLE" ]] || { mkdir -p "$(dirname "$NEEDLE")" && python3 "$here/needle_check.py" build --vocab "$VOCAB" --out "$NEEDLE"; }
# the estimate before anything is sent: the 2026-10-06 run's tokens and the needle prompts, every prompt token a
# cache miss, at the page's peak prices
python3 "$here/api_cost.py" "$API_THEN_USAGE" "$RUN/price-page.html" --estimate "$NEEDLE" --max-usd "$MAX_USD" ||
  { echo "api_run.sh: the estimate is over MAX_USD=$MAX_USD: nothing was sent" >&2; exit 2; }
rc=0
NEEDLE="$NEEDLE" bash "$here/eval.sh" deepseek "$API_BASE" "$RUN" gsm8k_cot_llama ifeval humaneval_instruct \
  needle-16k needle-64k needle-256k || rc=1
python3 "$here/api_cost.py" "$RUN/api_usage.jsonl" "$RUN/price-page.html" --json "$RUN/cost.json"
python3 -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['usd'] <= float(sys.argv[2]) else 1)" \
  "$RUN/cost.json" "$MAX_USD" || { echo "api_run.sh: the measured cost is over MAX_USD=$MAX_USD"; rc=1; }
python3 "$here/analyze.py" --api-fresh "$RUN" --out "$RUN/analysis.md" --json "$RUN/analysis.json" > /dev/null
sheet="$(python3 "$here/q_report.py" api "$RUN" "$SHEET_ROOT")"
echo "api_run.sh: sheet $sheet"
# acceptance 6: no file of the run or the sheet holds the key's first 8 characters (the key itself is never printed)
python3 - "$key" "$RUN" "$sheet" <<'EOF' || rc=1
import sys
from pathlib import Path
head = Path(sys.argv[1]).read_text().strip()[:8].encode()
hits = [str(f) for d in sys.argv[2:] for f in Path(d).rglob("*") if f.is_file() and head in f.read_bytes()]
print(f"key scan: {'FAIL, the key is in ' + ', '.join(hits) if hits else 'no file of the run or the sheet holds the key'}")
sys.exit(1 if hits else 0)
EOF
exit "$rc"
