#!/usr/bin/env bash
# Starts DeepSeek-V4.1-Flash on this machine in one command: checks the machine, downloads the model at the pinned
# revision when it is missing, plans the placement, writes deepseek.json, binds the server to the GPU's NUMA node,
# starts it and waits until it answers. The same steps as the README's Quickstart (steps 3 to 6), automated.
#
#   ./run.sh                          # engine from $ENGINE or ~/dsv41/strata-*-cu130-sm120, model in ~/models/q2k
#   ./run.sh --engine PATH            # the dsv41-engine binary (README, step 2)
#   ./run.sh --model-dir DIR          # where the seven GGUF files are, or go (264.5 GB)
#   ./run.sh --port N                 # the server's port (default 8090, on 127.0.0.1)
#   ./run.sh --dry-run                # print every command it would run, run none of them
#
# Environment: ENGINE, MODEL_DIR and PORT are the defaults of the three options. The placement goes to placement/,
# the configuration to deepseek.json and the engine's log to engine.log, all in this folder. Ctrl+C stops the server.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

HF_REPO=vcruz305/DeepSeek-V4.1-Flash-GGUF
HF_REVISION=c4a085541cb53f67ee5e57b63d255e80cef286e7
SHARD1=DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf
NEED_VRAM_MIB=$((90 * 1024))   # the RTX PRO 6000 reports 97887 MiB; e07_gate.py does the exact check
WANT_RAM_KIB=$((280 * 1000 * 1000))   # about 280 GB; e07_gate.py refuses a host below its minimum
NUMA_ROOT="${NUMA_ROOT:-/sys/devices/system/node}"   # overridable for tests, as in tools/dsv41/pod/t04a-lib.sh

engine="${ENGINE:-}"
model_dir="${MODEL_DIR:-$HOME/models/q2k}"
port="${PORT:-8090}"
dry=0

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
while (( $# )); do
  case "$1" in
    --engine) engine="${2:?--engine needs a path}"; shift 2 ;;
    --engine=*) engine="${1#*=}"; shift ;;
    --model-dir) model_dir="${2:?--model-dir needs a directory}"; shift 2 ;;
    --model-dir=*) model_dir="${1#*=}"; shift ;;
    --port) port="${2:?--port needs a number}"; shift 2 ;;
    --port=*) port="${1#*=}"; shift ;;
    --dry-run) dry=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run.sh: unknown argument '$1' (see ./run.sh --help)" >&2; exit 2 ;;
  esac
done
[[ "$port" =~ ^[0-9]+$ ]] || { echo "run.sh: --port '$port' is not a number" >&2; exit 2; }

say() { printf '[run] %s\n' "$*"; }
fail() {
  if (( dry )); then printf '[run] error (a real run stops here): %s\n' "$*" >&2; return 0; fi
  printf '[run] error: %s\n' "$*" >&2
  exit 1
}
warn() { printf '[run] warning: %s\n' "$*" >&2; }
# show: print a command, each argument single-quoted when the shell needs it, so it can be copied; run: print it and
# run it (only print it with --dry-run)
show() {
  local a out="+"
  for a in "$@"; do
    if [[ "$a" =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then out+=" $a"; else out+=" '${a//\'/\'\\\'\'}'"; fi
  done
  printf '%s\n' "$out"
}
run() { show "$@"; if (( dry )); then return 0; fi; "$@"; }
(( dry )) && say "dry run: every command is printed, none is run; read-only checks of this machine still run"

# --- 1. the machine ----------------------------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [[ ! "$vram_mib" =~ ^[0-9]+$ ]]; then
    vram_mib="<vram-mib>"
    fail "nvidia-smi does not answer (driver not loaded?): run nvidia-smi to see why"
  else
    say "GPU 0: ${gpu_name:-unknown}, $vram_mib MiB"
    (( vram_mib >= NEED_VRAM_MIB )) || fail "GPU 0 has $vram_mib MiB; this engine needs an RTX PRO 6000 with 96 GB"
  fi
else
  vram_mib="<vram-mib>"
  fail "nvidia-smi not found: an NVIDIA GPU with its driver (580.65 or newer) is needed"
fi

mem_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
say "RAM: $(( mem_kib / 1000 / 1000 )) GB"
(( mem_kib >= WANT_RAM_KIB )) || warn "about 280 GB of RAM is recommended; the placement below decides whether this host is enough"

# the memory cgroup's limit, as the pod scripts read it (tools/dsv41/pod/t04a-lib.sh): bytes, or unlimited
cgroup=unlimited
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [[ -r "$f" ]] || continue
  v="$(head -n 1 "$f")"
  if [[ "$v" =~ ^[0-9]+$ ]] && (( ${#v} <= 18 )) && (( v < 4611686018427387904 )); then cgroup="$v"; fi
  break
done
[[ "$cgroup" == unlimited ]] || say "memory cgroup limit: $(( cgroup / 1000 / 1000 / 1000 )) GB"

py="$(command -v python3 || true)"
if [[ -z "$py" ]] || ! "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  fail "Python 3.10 or newer is needed (python3)"
elif ! "$py" -c 'import jinja2' 2>/dev/null; then
  fail "the server's Python packages are missing: pip install -r requirements.txt"
fi

if [[ -z "$engine" ]]; then
  for e in "$HOME"/dsv41/strata-*-cu130-sm120/bin/dsv41-engine; do   # the newest build sorts last
    if [[ -x "$e" ]]; then engine="$e"; fi
  done
fi
if [[ -z "$engine" ]]; then
  engine="<dsv41-engine>"
  fail "no engine: build it (README, step 2) and pass --engine PATH or set ENGINE"
elif [[ ! -x "$engine" ]]; then
  fail "the engine $engine is not an executable file"
else
  engine="$(cd "$(dirname "$engine")" && pwd)/$(basename "$engine")"
  say "engine: $engine"
fi

# --- 2. the model ------------------------------------------------------------------------------------------------
model="$model_dir/$SHARD1"
missing=0
for i in 1 2 3 4 5 6 7; do
  [[ -s "$model_dir/DeepSeek-V4.1-Flash-Q2_K-0000$i-of-00007.gguf" ]] || missing=$(( missing + 1 ))
done
if (( missing )); then
  say "model: $missing of 7 files missing in $model_dir, downloading (264.5 GB in all)"
  command -v hf >/dev/null 2>&1 || fail "the hf command is missing: pip install -U \"huggingface_hub[cli]\""
  run mkdir -p "$model_dir"
  free_kb="$(df -Pk "$(dirname "$model_dir")" 2>/dev/null | awk 'NR == 2 {print $4}' || true)"
  [[ -z "$free_kb" ]] || (( free_kb >= 270 * 1000 * 1000 )) ||
    warn "$(( free_kb / 1000 / 1000 )) GB free under $(dirname "$model_dir"); the model needs 264.5 GB"
  run hf download "$HF_REPO" --revision "$HF_REVISION" --include "DeepSeek-V4.1-Flash-Q2_K-*" --local-dir "$model_dir"
else
  say "model: all 7 files in $model_dir"
fi
model="$(cd "$model_dir" 2>/dev/null && pwd || echo "$model_dir")/$SHARD1"

# --- 3. the placement (README, step 4) ---------------------------------------------------------------------------
run "${py:-python3}" tools/dsv41/pod/e07_gate.py \
  --vram-mib "$vram_mib" --cgroup-bytes "$cgroup" --mem-total-kib "$mem_kib" \
  --counts data/dsv41-expert-counts.json --layout tools/dsv41/oracle/layout-dsv41-q2k.json --model-headers "$model" \
  --prefetch-experts 128 --checkpoints 8 --batch 8 --batch-ctx 69632 --out placement

# --- 4. the configuration (README, step 5) -----------------------------------------------------------------------
write_config() {
  "${py:-python3}" - "$engine" "$model" <<'EOF'
import json, sys
engine, model = sys.argv[1], sys.argv[2]
gate = json.load(open("placement/gate.json"))
args = ["--model", model, "--hot-set", "placement/hot-set.json",
        "--vram-gb", str(gate["vram_gb"]), "--ram-gb", str(gate["ram_gb"]), "--threads", "16",
        "--hot-homes", str(gate["hot_homes"]), "--prefetch-experts", "128", "--checkpoints", "8",
        "--batch", "8", "--batch-ctx", "69632", "--chunk", "16384", "--ctx", "262144", "--eos-ids", "1"]
json.dump({"exe": engine, "args": args, "model_profile": "deepseek-v4.1", "tokenizer_gguf": model,
           "log": "engine.log", "repeat_stop_tokens": 0}, open("deepseek.json", "w"), indent=1)
EOF
}
say "writing deepseek.json from placement/gate.json (the README's step 5)"
(( dry )) || write_config

# --- 5. the NUMA node of the GPU ---------------------------------------------------------------------------------
# "NUMA Affinity" in `nvidia-smi topo -m` (GPU0's row, the column before the last); then the forms the pod scripts
# try, in order (tools/dsv41/pod/t04a-lib.sh numa_binding): numactl with the node's memory preferred, numactl on
# its CPUs only (a container's seccomp profile can refuse the memory policy), taskset on the node's CPU list.
bind=()
node="$(nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/ {print $(NF-1)}' || true)"
nodes=0
for d in "$NUMA_ROOT"/node[0-9]*; do
  if [[ -d "$d" ]]; then nodes=$(( nodes + 1 )); fi
done
if [[ ! "$node" =~ ^[0-9]+$ ]]; then
  warn "no NUMA node for GPU 0 in nvidia-smi topo -m ('${node:-nothing}'): the server runs unbound, expect lower speed"
elif (( nodes < 2 )); then
  say "NUMA: one node on this host, no binding needed"
else
  forms=("numactl --cpunodebind=$node --preferred=$node" "numactl --cpunodebind=$node")
  cpus="$(cat "$NUMA_ROOT/node$node/cpulist" 2>/dev/null || true)"
  [[ -z "$cpus" ]] || forms+=("taskset -c $cpus")
  for f in "${forms[@]}"; do
    read -ra cmd <<<"$f"
    if "${cmd[@]}" true >/dev/null 2>&1; then
      bind=("${cmd[@]}"); say "NUMA: GPU 0 is on node $node, binding with: $f"; break
    fi
    warn "NUMA: '$f' was refused here"
  done
  (( ${#bind[@]} )) || warn "NUMA: every binding was refused; the server runs unbound, expect lower speed"
fi

# --- 6. the server -----------------------------------------------------------------------------------------------
server=(${bind[@]+"${bind[@]}"} "${py:-python3}" -m serve.server --engine strata --config deepseek.json --port "$port")
curl_example="curl http://localhost:$port/v1/chat/completions -H 'Content-Type: application/json' -d '{
  \"model\": \"deepseek-v4.1-flash\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Explain mixture-of-experts models in two sentences.\"}],
  \"max_tokens\": 200
}'"
if (( dry )); then
  show "${server[@]}"
  show curl -s "http://localhost:$port/health"
  say "(repeated until it answers, then) ready on http://localhost:$port/v1; it prints this request to try:"
  printf '%s\n' "$curl_example"
  exit 0
fi

show "${server[@]}"
"${server[@]}" &
pid=$!
trap 'kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; exit 130' INT TERM
say "loading the model (a few minutes); the engine writes to engine.log"
# /health answers once the server listens; "loaded" turns true when the engine has the model
until h="$(curl -sf "http://localhost:$port/health")" && [[ "$h" != *'"loaded": false'* ]]; do
  kill -0 "$pid" 2>/dev/null || { wait "$pid" || true; fail "the server stopped before it was ready: see engine.log"; }
  sleep 5
done
say "ready on http://localhost:$port/v1 (OpenAI compatible). Try:"
printf '\n%s\n\n' "$curl_example"
say "Ctrl+C stops the server"
wait "$pid"
