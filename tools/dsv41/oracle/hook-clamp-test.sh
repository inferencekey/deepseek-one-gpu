#!/usr/bin/env bash
# The fork's LLAMA_STRATA_MOE=1 hook path with a SwiGLU clamp, end to end on the CPU (no GPU, no real model):
#   1. the fork's own test-llama-archs writes its tiny random DeepSeek-V4 MoE (4 layers, 2 experts, 1 used, f32);
#   2. tiny_moe_model.py gives every layer its own clamp limit and drops the routed expert scale tensors;
#   3. dsv41-moe-clamp-check decodes it without the hook and with it: the op must have computed with the GGUF
#      limit of its layer in every call (reused graphs included), and the logits must be bit-identical.
# DeepSeek-V4.1 itself has no tiny random fixture in test-llama-archs; deepseek4 goes through the same
# build_moe_ffn hook with the same clamp (both are swiglu_clamp_arch in the fork's llama-graph.cpp).
#
#   tools/dsv41/oracle/hook-clamp-test.sh FORK_BIN WORK_DIR
#
# FORK_BIN: the fork build's bin/ with test-llama-archs, dsv41-moe-clamp-check and libstrata-moe.so. Exits 1 on any
# failure. verify.sh runs it in the builder image.
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 FORK_BIN WORK_DIR" >&2; exit 2; }
bin="$1" work="$2"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIMITS=0.05,0.06,0.07,0.08
mkdir -p "$work"

"$bin/test-llama-archs" -a deepseek4 -s 1 -o "$work" > "$work/test-llama-archs.log" 2>&1 ||
  { tail -n 5 "$work/test-llama-archs.log"; echo "FAIL: test-llama-archs -a deepseek4 -o"; exit 1; }
python3 "$here/tiny_moe_model.py" "$work/deepseek4-moe.gguf" "$work/tiny-clamp.gguf" --clamp "$LIMITS"

check() {
  local name="$1"; shift
  local rc=0
  "$@" "$bin/dsv41-moe-clamp-check" -m "$work/tiny-clamp.gguf" -n 8 > "$work/$name.out" 2> "$work/$name.err" || rc=$?
  echo "--- dsv41-moe-clamp-check, $name (exit $rc)"
  grep -v "^$" "$work/$name.out" || true
  if (( rc != 0 )); then tail -n 5 "$work/$name.err"; echo "FAIL: $name"; exit 1; fi
}
check stock env -u LLAMA_STRATA_MOE
# the tiny model's experts are f32, which the fast CPU kernel does not compute (it aborts naming the tensor)
check hook env LLAMA_STRATA_MOE=1 STRATA_MOE_CPU=ref

stock="$(sed -n 's/^logits: *\(fnv1a64 [0-9a-f]*\).*/\1/p' "$work/stock.out")"
hook="$(sed -n 's/^logits: *\(fnv1a64 [0-9a-f]*\).*/\1/p' "$work/hook.out")"
[[ -n "$stock" && "$stock" == "$hook" ]] ||
  { echo "FAIL: logits differ with the hook ($hook) and without ($stock)"; exit 1; }
echo "hook-clamp-test: PASS (clamp limits $LIMITS seen in every layer and call; logits $hook with and without the hook)"
