#!/usr/bin/env bash
# On a CPU-only machine, no pods: the tiny random DeepSeek-V4.1 models and the per-op oracles of the stock fork.
#   1. tiny_dsv41.py writes configs a and b with the real model's tensor types, and a with all f32 (SPEC.md,
#      "Tiny models");
#   2. op-dump is built in the cu128 builder image against $BUILD_ROOT/cu128/fork, whose libllama, libggml*
#      and libllama-common must be byte-identical to the cu128 tarball's (checked against its MANIFEST sha256);
#   3. per config, two 16 token greedy runs with the tarball's libraries (-ngl 0): same tokens and logits hashes
#      (one run of the f32 variant);
#   4. per config, two op-dump runs (64 prompt tokens, 8 generated): op_dump.py check (sizes, every spec op of every
#      layer role, SPEC.md shapes) on each, then op_dump.py compare (bit-identical) and op_dump.py roles (which
#      layers are readers and index sources, and whether the two reader modes coincide over the dump); then one
#      short op-dump per config over which Reuse Mode and --readers dense coincide (SPEC.md "Reader modes": 12 + 5
#      tokens for a, 24 + 9 for b, positions up to 16 and 32), checked, its roles confirming the modes coincide;
#      then one op-dump per config over the 16 greedy tokens of step 3 (dump-a-16, dump-b-16), checked, its tokens
#      the greedy runs' (the engine's end-to-end parity reads every step's logits and l_last from it);
#   5. kv_budget.py on tiny-b against the fork's own KV buffer sizes for it, then on the real model's headers;
#   6. op_dump.py map-oracle on the real-model oracle's manifest (the stock fork's dsv41-moe-dump on an A100): every recorded tensor is a SPEC.md op;
#      op_dump.py roles on the real model's headers at 262,144 tokens.
#
#   tools/dsv41/engine/run-oracles.sh WORK_DIR
#
# Environment: BUILD_ROOT ($HOME/dsv41-build), FORK_DIR ($HOME/llama-dsv41, the patched fork's checkout), PYTHON (python3 with numpy, default the
# build root's venv), TARBALL (the cu128 fork tarball), HEADERS (the real model's shard headers, SPEC.md "Sources"),
# ORACLE_MANIFEST (the real-model oracle's run-1 manifest), THREADS (4). Exits 1 on the first failed check.
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 WORK_DIR" >&2; exit 2; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../flavors.sh"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
FORK_DIR="${FORK_DIR:-$HOME/llama-dsv41}"
PYTHON="${PYTHON:-$BUILD_ROOT/venv/bin/python}"
TARBALL="${TARBALL:-$BUILD_ROOT/dist/llama-dsv41-b4d9cb12-cu128-sm80-89.tar.gz}"
HEADERS="${HEADERS:-$BUILD_ROOT/engine/headers}"
ORACLE_MANIFEST="${ORACLE_MANIFEST:-$BUILD_ROOT/oracle-dsv41/run-1/dump-manifest.json}"
THREADS="${THREADS:-4}"
mkdir -p "$1"
work="$(cd "$1" && pwd)"
case "$work/" in
  "$STRATA_DIR"/*|"$FORK_DIR"/*) echo "run-oracles.sh: WORK_DIR must be outside both repos" >&2; exit 2 ;;
esac
"$PYTHON" -c "import numpy" || { echo "run-oracles.sh: $PYTHON has no numpy (set PYTHON)" >&2; exit 1; }

flavor_config 12.8
image="$(builder_image)"
docker image inspect "$image" >/dev/null 2>&1 || { echo "run-oracles.sh: no $image, run build.sh first" >&2; exit 1; }
fork_build="$BUILD_ROOT/$FLAVOR_TAG/fork"

# runs a command in the builder image as the invoking user, without the CUDA image's banner entrypoint; the
# toolkit's libcuda stub stands in for the driver, so ggml-cuda finds no device and everything runs on the CPU
in_image() {
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home --entrypoint bash \
    -v "$FORK_DIR:$FORK_DIR:ro" -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$BUILD_ROOT:$BUILD_ROOT" -v "$work:$work" \
    "$image" -euo pipefail -c "mkdir -p /tmp/cs; ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cs/libcuda.so.1; $1"
}

echo "=== 1. tiny models"
for c in a b; do
  "$PYTHON" "$here/tiny_dsv41.py" --config "$c" "$work/tiny-$c.gguf"
done
"$PYTHON" "$here/tiny_dsv41.py" --config a --types f32 "$work/tiny-a-f32.gguf"

echo "=== 2. op-dump (builder $image, fork build tree $fork_build)"
tar xzf "$TARBALL" -C "$work"
fork_bin="$work/$(basename "$TARBALL" .tar.gz)/bin"
for lib in libllama.so.0 libllama-common.so.0 libggml.so.0 libggml-base.so.0 libggml-cpu.so.0; do
  want="$(awk -v f="bin/$lib" '$2 == f { print $1 }' "$fork_bin/../MANIFEST")"
  have="$(sha256sum "$fork_build/bin/$lib" | cut -d' ' -f1)"
  [[ -n "$want" && "$want" == "$have" ]] || { echo "FAIL: $lib of $fork_build differs from the tarball's ($have vs $want)"; exit 1; }
done
echo "the build tree's libllama, libllama-common, libggml, libggml-base, libggml-cpu match the tarball MANIFEST sha256"
grep -E '^(commit|cuda_flavor|cmake_flags):' "$fork_bin/../MANIFEST"
in_image "cmake -S $here/op-dump -B $work/build-op-dump -G Ninja -DCMAKE_BUILD_TYPE=Release -DFORK_DIR=$FORK_DIR \
  -DFORK_BUILD=$fork_build > $work/build-op-dump.log && cmake --build $work/build-op-dump >> $work/build-op-dump.log" ||
  { tail -n 20 "$work/build-op-dump.log"; echo "FAIL: op-dump build"; exit 1; }
op_dump="$work/build-op-dump/op-dump"
run_fork() {   # run_fork MODEL [ARGS...], 64 prompt tokens unless PROMPT_TOKENS says otherwise
  in_image "LD_LIBRARY_PATH=$fork_bin:/tmp/cs $op_dump -m $1 --prompt-tokens ${PROMPT_TOKENS:-64} --prompt-seed 1 \
    -c 2048 -ngl 0 -fa off -t $THREADS ${*:2}"
}

echo "=== 3. 16 token greedy runs, twice per config (CPU, -ngl 0, tarball libraries)"
for c in a b; do
  for r in 1 2; do
    run_fork "$work/tiny-$c.gguf" -n 16 > "$work/gen-$c-$r.out" 2> "$work/gen-$c-$r.err" ||
      { tail -n 5 "$work/gen-$c-$r.err"; echo "FAIL: config $c run $r"; exit 1; }
    echo "config $c run $r: $(tr '\n' ' ' < "$work/gen-$c-$r.out")"
  done
  cmp -s "$work/gen-$c-1.out" "$work/gen-$c-2.out" || { echo "FAIL: config $c, the two runs differ"; exit 1; }
  echo "config $c: tokens and logits hash identical in both runs"
done

run_fork "$work/tiny-a-f32.gguf" -n 16 > "$work/gen-a-f32.out" 2> "$work/gen-a-f32.err" ||
  { tail -n 5 "$work/gen-a-f32.err"; echo "FAIL: config a, f32"; exit 1; }
echo "config a, all f32: $(tr '\n' ' ' < "$work/gen-a-f32.out")"

echo "=== 4. op-dump, 64 prompt tokens and 8 generated, twice per config"
for c in a b; do
  for r in 1 2; do
    rm -rf "$work/dump-$c-$r"
    run_fork "$work/tiny-$c.gguf" -n 8 --dump-dir "$work/dump-$c-$r" > "$work/dump-$c-$r.out" 2> "$work/dump-$c-$r.err" ||
      { tail -n 5 "$work/dump-$c-$r.err"; echo "FAIL: op-dump config $c run $r"; exit 1; }
    grep -v '^$' "$work/dump-$c-$r.out"
    if grep -q "spec ops never computed" "$work/dump-$c-$r.err"; then
      grep "spec ops never computed" "$work/dump-$c-$r.err"; echo "FAIL: config $c run $r"; exit 1
    fi
    "$PYTHON" "$here/op_dump.py" check "$work/dump-$c-$r"
  done
  "$PYTHON" "$here/op_dump.py" compare "$work/dump-$c-1" "$work/dump-$c-2"
  "$PYTHON" "$here/op_dump.py" roles "$work/dump-$c-1"
  echo "dump size: $(du -sh "$work/dump-$c-1" | cut -f1)"
done
for spec in a:12:5 b:24:9; do
  IFS=: read -r c n_prompt n_gen <<<"$spec"
  rm -rf "$work/dump-$c-short"
  PROMPT_TOKENS=$n_prompt run_fork "$work/tiny-$c.gguf" -n "$n_gen" --dump-dir "$work/dump-$c-short" \
    > "$work/dump-$c-short.out" 2> "$work/dump-$c-short.err" ||
    { tail -n 5 "$work/dump-$c-short.err"; echo "FAIL: short op-dump config $c"; exit 1; }
  grep -v '^$' "$work/dump-$c-short.out"
  "$PYTHON" "$here/op_dump.py" check "$work/dump-$c-short"
  "$PYTHON" "$here/op_dump.py" roles "$work/dump-$c-short" | tee "$work/dump-$c-short.roles" | tail -n 1
  grep -q "coincide in every reader" "$work/dump-$c-short.roles" ||
    { echo "FAIL: the short dump of config $c is past the reader modes' common range"; exit 1; }
done

# The end-to-end parity of the engine compares every generated step's logits and every layer's l_last, so it
# needs a dump over the 16 greedy tokens of step 3 (the 8-token dumps above cover half of them)
for c in a b; do
  rm -rf "$work/dump-$c-16"
  run_fork "$work/tiny-$c.gguf" -n 16 --dump-dir "$work/dump-$c-16" > "$work/dump-$c-16.out" 2> "$work/dump-$c-16.err" ||
    { tail -n 5 "$work/dump-$c-16.err"; echo "FAIL: 16-token op-dump config $c"; exit 1; }
  grep -v '^$' "$work/dump-$c-16.out"
  "$PYTHON" "$here/op_dump.py" check "$work/dump-$c-16"
  [[ "$(grep '^generated_token_ids:' "$work/dump-$c-16.out")" == "$(grep '^generated_token_ids:' "$work/gen-$c-1.out")" ]] ||
    { echo "FAIL: config $c, the 16-token dump generated other tokens than the greedy runs"; exit 1; }
  echo "config $c: the 16-token dump generated the greedy runs' tokens"
done

echo "=== 5. KV bytes"
# the fork's own buffer sizes for tiny-b (raw window, csa and lid caches, in its log order) against kv_budget.py's
in_image "LD_LIBRARY_PATH=$fork_bin:/tmp/cs $op_dump -m $work/tiny-b.gguf --prompt-ids 1 -n 1 -c 2048 -ngl 0 -fa off \
  -t $THREADS -lv 4" 2> "$work/kv-tiny-b.log" > /dev/null
grep -E 'llama_kv_cache: size|comp_state: .* ratio' "$work/kv-tiny-b.log" | sed 's/^[0-9.]* I //'
fork_mib="$(sed -n 's/.*llama_kv_cache: size = *\([0-9.]*\) MiB.*/\1/p' "$work/kv-tiny-b.log" | awk 'NR == 2 || NR == 3 || NR == 5' | xargs)"
python3 "$here/kv_budget.py" --ctx 2048 "$work/tiny-b.gguf" | sed -n '4,8p' | tee "$work/kv-tiny-b.budget"
budget_mib="$(awk '$1 == "raw" || $1 == "csa" || $1 == "lid" { for (i = 1; i <= NF; i++) if ($i == "MiB") print $(i - 1) }' \
  "$work/kv-tiny-b.budget" | xargs)"
[[ -n "$fork_mib" && "$fork_mib" == "$budget_mib" ]] ||
  { echo "FAIL: kv_budget.py says raw/csa/lid $budget_mib MiB, the fork allocated $fork_mib"; exit 1; }
echo "tiny-b at ctx 2048: raw/csa/lid $fork_mib MiB, kv_budget.py and the fork agree"
python3 "$here/kv_budget.py" --ctx 262144 "$HEADERS"/head-*.bin

echo "=== 6. the real model's oracle manifest against the spec ops"
"$PYTHON" "$here/op_dump.py" map-oracle "$ORACLE_MANIFEST"
"$PYTHON" "$here/op_dump.py" roles --ctx 262144 "$HEADERS"/head-*.bin
echo "run-oracles.sh: PASS"
