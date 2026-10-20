#!/usr/bin/env bash
# Checks the products of tools/dsv41/build.sh without a GPU (T01, T02a, T03, T04a, T04b, T04c, T04d, T06 and the
# engine's E02, E03, E04a, E04b, E05a, E05b, E05c and E05d acceptance),
# per flavor: version/help output (dsv41-moe-dump and dsv41-moe-replay included), the fork's DeepSeek-V4.1 ggml
# symbols, libstrata-moe.so's exports, the flavor's CUDA archs in the device code, Strata's GPU-free tests
# (dsv41_moe_ref_test, dsv41_moe_split_test, dsv41_moe_cpu_fast_test, dsv41_moe_residency_test,
# dsv41_moe_counts_test, dsv41_moe_prefill_test and dsv41_indexer_index_test included), a short dsv41_moe_cpu_bench run,
# the fork's hook path with a SwiGLU clamp on a tiny random model (oracle/hook-clamp-test.sh), the tool tests
# (oracle/hot_set_test.py, pod/t04a-lib-test.sh) and the pod script's dry runs (full, --speed, the --speed
# selection SPEED_BUDGETS=70 RAM_RUNS=70:40 CPU_KERNELS=fast EXTRA=0 and the T04d session --speed --profile
# SPEED_BUDGETS=70 RAM_RUNS=auto CPU_KERNELS=fast EXTRA=0 CAP_MIN=60, whose plan must fit), the T06 prefill prompts
# (pod/prefill_prompts.py build with the fork's llama-tokenize and $VOCAB, a vocab-only DeepSeek-V4.1 GGUF, then
# check: token counts within 2%) and the dry run of the T06 session --prefill CAP_MIN=63 on them (the plan in
# full), RUNPATH $ORIGIN on every built binary of both tarballs, and that both tarballs resolve their libraries in a
# bare ubuntu:22.04 without LD_LIBRARY_PATH and run there (each flavor bundles its CUDA runtime). E02: the engine's
# lightning indexer reference against the E01 op dumps of tiny-a and tiny-b (dsv41_engine_indexer_test --cases on
# the lists of engine/indexer_cases.py; $E01, default $BUILD_ROOT/engine/e01, from engine/run-oracles.sh), E03: the
# attention reference against the same dumps (dsv41_engine_attention_test --cases) and the host side of its GPU test
# (--gpu --dry-run, plain, --f16 and --mmq: every case's weight checks and reference numerics, no CUDA call), E04a: the
# qmatmul reference on hand-quantized blocks and on the tiny GGUFs' matrices against the fork's CPU
# (dsv41_engine_qmatmul_test --cpu $E01/tiny-a.gguf $E01/tiny-b.gguf), E04b: the dense ops' reference (hyper-
# connections, engram, router, norms, shared expert, embedding, head) against the same four dumps bit for bit
# (dsv41_engine_dense_test --cases on the lists of engine/dense_cases.py) and the host side of its GPU test (--gpu
# --dry-run, plain and --mmq), E05a: the CPU experts against the six dumps' ffn_moe_out bit for bit
# (dsv41_engine_experts_test --cases), the host side of the expert tiers' GPU test in three placements (--gpu
# --dry-run on the hot sets of engine/hot_sets.py), dsv41-engine --dry-run with the arguments of every end-to-end run
# of the dev pod, free and with --force-selection on the fork's dumps (every view, tier and call planned, every
# forced selection read and checked, no CUDA call), the forced mode's prompt guard, the architecture guard on a
# non-deepseek41 GGUF ($GUARD_MODEL), engine/engine_parity.py on a fork dump against itself and its --self-test (flip
# attribution and the forced rules on synthetic dumps), the engine binaries' device code for the
# flavor's archs and in the strata tarball, and pod/devgpu.sh --dry-run; E05b: dsv41-engine --dry-run of the
# multi-chunk forced runs (--chunk 24 against the fork's one ubatch, a small ring and prefetch), engine_parity.py's
# chunked self-test cases, dsv41_engine_smoke --dry-run with the pod's arguments, and devgpu.sh's pull in its plan;
# the prefill GEMM's numerics on the host against the six dumps' ffn_moe_out (in dsv41_engine_experts_test
# --cases), the host side of its GPU test (--gemm --dry-run), and dsv41_engine_smoke --dry-run with --counts $COUNTS
# (the hybrid's decode profile: the hot set it picks and its expected hit rate); E05d: the decode path each placement's
# dry plan gives (all-vram: grouped hits in every layer and no doorbell, all-cpu: a doorbell in every layer and no
# hits, split: both), the host side of the attention's and the experts' GPU tests again (the decode token's checks and
# its slab layout), and the smokes' and devgpu.sh's dry runs with the token graph's decode phases; E06: the
# tokenizer's codepoint table against the fork's (engine/llama_unicode.py check), the server's DeepSeek-V4.1 tokenizer
# against the fork's llama-tokenize on engine/tokenizer_corpus.txt (226 lines, specials and unicode) with $VOCAB and
# with engine/tiny_vocab.py's 512-token cut of it (engine/tokenizer_parity.py), the server's own tests (serve/test_*.py),
# engine/serve_check.py on dsv41-engine --serve --dry-run (the deepseek-v4.1 profile, the stdin protocol with the
# prefix cache's paths, the server's /v1/chat/completions streaming and not), and dsv41_engine_smoke --prefix-cache
# --dry-run with the pod's arguments; engine/oracle_parity.py --self-test (the real-model oracle's comparison
# and its forced reference), a split copy of tiny-b (engine/gguf_layout.py split, 3 shards) whose dsv41-engine
# --dry-run plans every run as the whole file's, the real model's 7 shards as sparse files from their headers
# (gguf_layout.py sparse, $HEADERS) at the sizes measured on the volume, the speed prompts (pod/e07_prompts.py check, built
# once from the corpus and $VOCAB), pod/e07_bench.py's phases against dsv41-engine --serve --dry-run (tiny-b, the
# prompts' ids mod 512, --eos-ids none), pod/e07_report.py on an empty session, and pod/e07.sh --dry-run (the whole
# plan, the gate per GPU on the real headers, and dsv41-engine --dry-run of every correctness run, free and forced
# on the oracle's reference, and of the benches' engine on the real layout); the isolation runs:
# engine/layer_parity.py --self-test, dsv41-layer-ref on tiny-b's layers 0 and 1 in its three numerics with the fork's
# dump as the oracle (layer_parity.py ops: fork-cpu bit for bit against every op of the dump; compare: its cur, out
# and isolated out 0), dsv41-engine --dry-run with --forced-moe-input (every call's MoE input read and checked) and
# --prefill-mmq and the guard of the first, the bench's short-timed phase (times=1) against the serve dry run, and
# pod/e07.sh --dry-run of the cycle2 plan (with dsv41-layer-ref --dry-run of each numerics on the real layout) and of
# the full and dense-only plans' totals; dsv41_engine_hot_policy_test (the
# planner's rules and invariants, two turns of a conversation), engine/hot_policy_sim.py --self-test (prompt and
# adaptive above static on a skewed synthetic trace; two turns) and two-turn (the resumed turn 2 against fresh and
# continued), the simulator against the planner's --replay on that trace, the two-turn traces and the stock fork's oracle dump's
# topk (the same hits and swaps per layer for five knob settings) and its compare there (prompt and adaptive above
# static),
# dsv41-engine --dry-run of the dev pod's forced-swap runs and its refusals of bad --hot-* flags, the serve protocol's
# hot set keys, pod/e07_bench.py's e08 phases against the serve dry run, pod/e07_report.py on an empty e08 session,
# and pod/e07.sh --dry-run with E07_PLAN=e08 at 1.95 USD/h (the HF plan under 3 USD); the prefill work:
# dsv41_engine_qdequant_test (the grouped GEMM's table decoder against the dequant reference, bit for bit), the
# attention bench's prefill options in devgpu.sh's plan, pod/e07_bench.py's e09 phases against the serve dry run,
# engine/two_turn_check.py --dry-run (tiny-b behind serve/server.py, turn 2 resumed), pod/e07_report.py on an empty
# e09 session, and pod/e07.sh --dry-run with E07_PLAN=e09 at 1.95 USD/h (the HF plan within 1.50 USD, the gate with a
# prefetch region of 128 experts); pod/e07b-test.sh (pod-steps.sh's per-step timeouts,
# hf-download.sh's throughput guard on a fake aria2c, pod/e07_report.py's goal table on a fixture session),
# pod/e07_bench.py's e07b phases against the serve dry run, engine/two_turn_check.py --dry-run --http (tiny-b: the HTTP
# check on the same server load), pod/e07_report.py on an empty e07b session (every goal row with its cause), and
# pod/e07.sh --dry-run with E07_PLAN=e07b at 1.95 USD/h (the HF plan at 0.115 GB/s within 3.00 USD; with DRY_ENGINE the
# two-turn dry run with the 16K prompt's text at ctx 20480 and the bench's phases on the real layout) and with
# HF_GBS=0.06 (exit 2: over 3.00 USD). serve_check.py's sampled requests (the
# protocol's seeds and the server's seed 0, top_k 40 and frequency_penalty 400) and dsv41_engine_sampler_test;
# dsv41_engine_experts_test --determinism on tiny-b (the deterministic mode: the same
# requests bit for bit under other swap timings, arrival orders and thread counts, and after another request; with the
# mode off they differ). dsv41_engine_checkpoint_test on tiny-b (the
# Session's conversation checkpoints on the CPU reference: save, restore and evict bit for bit against a straight
# read), serve_check.py's conversations over the protocol, and --determinism with the reset after each request;
# dsv41_engine_batch_test on tiny-b (the batch slots on the CPU reference: every
# batched request bit for bit its alone run) and --serve --dry-run --batch 4 answering a request in a slot;
# the batch test's check 6 (a prompt read in chunks with steps between them,
# bit for bit) and --serve --dry-run --interleave-chunk 8 reading a prompt beside a decoding slot in chunks, a step
# between two. dsv41_engine_expert_bits_test (the CPU experts in the GPU's
# numerics, mmvq_ref.hpp's bits) and the batch test's checks 10 and 11 (batched requests in the deterministic mode their
# alone run's bit for bit under other placements), a deterministic slot rep=1 beside a det=0 one;
# dsv41_engine_session_test (a promoted request's one token left fed by decode)
# and the batch test's check 12 (a promoted request its alone run's bit for bit beside 3 others); for the batched
# decode's regression, dsv41_engine_hot_policy_test's promoted batch (a batch opened by a promoted request swaps).
# Exits non-zero on the first failed check.
#
#   tools/dsv41/verify.sh                     # both flavors
#   CUDA_FLAVOR=13.0 tools/dsv41/verify.sh    # one flavor
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/flavors.sh"
STRATA_DIR="$(cd "$here/../.." && pwd)"
FORK_DIR="${FORK_DIR:-$HOME/llama-dsv41}"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
# the prefill prompts' tokenizer: made once from the first 64 MiB of DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf
# (tools/dsv41/README.md), 5.6 MiB, kept out of the repo
VOCAB="${VOCAB:-$BUILD_ROOT/vocab/deepseek41-vocab.gguf}"
[[ -f "$VOCAB" ]] || { echo "verify.sh: no $VOCAB (tools/dsv41/README.md: prefill_prompts.py vocab)" >&2; exit 1; }
# the op dumps the indexer reference is checked against (tools/dsv41/engine/run-oracles.sh writes them)
E01="${E01:-$BUILD_ROOT/engine/e01}"
for m in a-1 b-1 a-short b-short a-16 b-16; do
  [[ -f "$E01/dump-$m/dump-manifest.json" ]] || { echo "verify.sh: no $E01/dump-$m (tools/dsv41/engine/run-oracles.sh $E01)" >&2; exit 1; }
done
# a GGUF of another architecture, which dsv41-engine must refuse (any OLMoE GGUF: only its header is read)
GUARD_MODEL="${GUARD_MODEL:-$BUILD_ROOT/testmodels/olmoe-1b-7b-0924-q4_k_m.gguf}"
[[ -f "$GUARD_MODEL" ]] || { echo "verify.sh: no $GUARD_MODEL (a non-deepseek41 GGUF for the architecture guard)" >&2; exit 1; }
for m in tiny-a tiny-b; do
  [[ -f "$E01/$m.gguf" ]] || { echo "verify.sh: no $E01/$m.gguf (tools/dsv41/engine/run-oracles.sh $E01)" >&2; exit 1; }
done
# the real model's shard headers (SPEC.md "Real model header") and the stock fork's oracle dump
HEADERS="${HEADERS:-$BUILD_ROOT/engine/headers}"
[[ -f "$HEADERS/head-7.bin" ]] || { echo "verify.sh: no $HEADERS/head-1.bin .. head-7.bin (SPEC.md, Real model header)" >&2; exit 1; }
[[ -f "$BUILD_ROOT/oracle-dsv41/run-1/dump-manifest.json" ]] || { echo "verify.sh: no $BUILD_ROOT/oracle-dsv41/run-1" >&2; exit 1; }
# the decode counts profile the measurement smoke routes by (a real-model run's counts.json, as pod/devgpu.sh
# uploads it)
COUNTS="${COUNTS:-$BUILD_ROOT/counts.json}"
[[ -f "$COUNTS" ]] || { echo "verify.sh: no COUNTS=$COUNTS (a strata-moe-counts/1 profile)" >&2; exit 1; }
# the bare image the pods' extraction is checked in, pinned like the builder bases
BARE_IMAGE="ubuntu:22.04@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401"

flavors="$(selected_flavors)" || exit 2
fork_short="$(git -C "$FORK_DIR" rev-parse --short=7 HEAD)"
# a git worktree's .git file points into its main repository's .git, which the dry runs' git reads too (e07.sh's
# payload sources, the machine probe's HEAD source): mounted read only beside the worktree when it lies outside it, as
# build.sh mounts it
git_mounts=()
git_common="$(git -C "$STRATA_DIR" rev-parse --path-format=absolute --git-common-dir)"
case "$git_common/" in "$STRATA_DIR"/*) ;; *) git_mounts=(-v "$git_common:$git_common:ro") ;; esac
start=$(date +%s)

for flavor in $flavors; do
  flavor_config "$flavor"
  image="$(builder_image)"
  docker image inspect "$image" >/dev/null 2>&1 || { echo "verify.sh: no $image, run build.sh first" >&2; exit 1; }
  flavor_root="$BUILD_ROOT/$FLAVOR_TAG${BUILD_TAG:-}"
  archs=""
  IFS=';' read -ra arch_list <<<"$CUDA_ARCHS"
  for a in "${arch_list[@]}"; do archs+=" sm_$a"; done

  echo "=== $FLAVOR_TAG: checks in the builder image $image (no GPU)"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e FLAVOR_ROOT="$flavor_root" \
    -e FORK_SHORT="$fork_short" -e ARCHS="$archs" -e DIST="$BUILD_ROOT/dist${BUILD_TAG:-}" -e STRATA_DIR="$STRATA_DIR" \
    -e PATTERN="*-$FLAVOR_TAG-$ARCH_TAG.tar.gz" -e FORK_DIR="$FORK_DIR" -e VOCAB="$VOCAB" -e E01="$E01" \
    -e BUILD_ROOT="$BUILD_ROOT" -e BUILD_TAG="${BUILD_TAG:-}" -e GUARD_MODEL="$GUARD_MODEL" -e COUNTS="$COUNTS" \
    -e HEADERS="$HEADERS" \
    -v "$FORK_DIR:$FORK_DIR:ro" -v "$STRATA_DIR:$STRATA_DIR:ro" "${git_mounts[@]}" -v "$BUILD_ROOT:$BUILD_ROOT" \
    -v "$GUARD_MODEL:$GUARD_MODEL:ro" -v "$COUNTS:$COUNTS:ro" \
    "$image" bash -euo pipefail -c '
      fork="$FLAVOR_ROOT/fork/bin" strata="$FLAVOR_ROOT/strata"
      echo "--- nvcc: $(nvcc --version | tail -n 2 | head -n 1)"
      # ggml-cuda links the driver library (VMM); without a GPU the toolkit stub stands in for libcuda.so.1.
      # A copy in the build dir is what the bare ubuntu:22.04 check below installs as its system libcuda.so.1.
      mkdir -p /tmp/cudastub "$FLAVOR_ROOT/cudastub"
      ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1
      cp /usr/local/cuda/lib64/stubs/libcuda.so "$FLAVOR_ROOT/cudastub/libcuda.so.1"
      export LD_LIBRARY_PATH=/tmp/cudastub

      for exe in llama-server llama-cli; do
        echo "--- $exe --version"
        out="$("$fork/$exe" --version 2>&1)"; echo "$out"
        grep -q "$FORK_SHORT" <<<"$out" || { echo "FAIL: $exe --version lacks $FORK_SHORT"; exit 1; }
      done

      echo "--- dsv41-moe-dump --help"
      rc=0; "$fork/dsv41-moe-dump" --help > /tmp/dump-help.txt 2>&1 || rc=$?
      tail -n 11 /tmp/dump-help.txt; echo "(exit $rc)"
      [[ $rc -eq 0 ]] && grep -q -- "--dump-dir DIR" /tmp/dump-help.txt || { echo "FAIL: dsv41-moe-dump --help"; exit 1; }

      echo "--- dsv41-moe-replay --help"
      rc=0; "$fork/dsv41-moe-replay" --help > /tmp/replay-help.txt 2>&1 || rc=$?
      head -n 1 /tmp/replay-help.txt; echo "(exit $rc)"
      [[ $rc -eq 0 ]] && grep -q -- "--dump-dir DIR" /tmp/replay-help.txt || { echo "FAIL: dsv41-moe-replay --help"; exit 1; }

      echo "--- strata --help"
      rc=0; "$strata/strata" --help > /tmp/help.txt 2>&1 || rc=$?
      head -n 15 /tmp/help.txt; echo "(exit $rc)"
      # strata has no "usage:" header: its usage text starts with the command line it documents
      [[ $rc -eq 0 ]] && grep -q "^strata generate --pack" /tmp/help.txt || { echo "FAIL: strata --help"; exit 1; }

      echo "--- nm -D: DeepSeek-V4.1 ops in the fork ggml"
      for sym in ggml_dsv4_hc_comb ggml_dsv4_hc_pre ggml_dsv4_hc_post ggml_lightning_indexer; do
        hit="$(for so in "$fork"/libggml*.so; do nm -D --defined-only "$so" | awk -v s="$sym" -v f="${so##*/}" "\$3 == s { print f\": \"\$0 }"; done)"
        [[ -n "$hit" ]] || { echo "FAIL: $sym not exported"; exit 1; }
        echo "$hit"
      done

      echo "--- nm -D: the hook op and the layer counts in libstrata-moe.so (next to libllama, which dlopens it)"
      for sym in strata_moe_forward strata_moe_layer_counts strata_moe_layer_clamp; do
        nm -D --defined-only "$fork/libstrata-moe.so" | grep -w "$sym" || { echo "FAIL: $sym not exported"; exit 1; }
      done
      echo "--- NEEDED of libstrata-moe.so"
      readelf -d "$fork/libstrata-moe.so" | grep NEEDED | grep -E "libggml|libcudart" || { echo "FAIL: libstrata-moe.so NEEDED"; exit 1; }
      echo "--- the pinned warm tier (STRATA_MOE_RAM_MODE=pinned, run on the pod only) is linked: cudaHostAlloc, cudaFreeHost"
      for sym in cudaHostAlloc cudaFreeHost; do
        nm -D --undefined-only "$fork/libstrata-moe.so" | grep -w "$sym" || { echo "FAIL: libstrata-moe.so does not import $sym"; exit 1; }
      done

      for bin in "$fork/libggml-cuda.so" "$strata/strata" "$strata/dsv41_engine_indexer_test" "$strata/dsv41_engine_indexer_bench" \
                 "$strata/dsv41_engine_qmatmul_test" "$strata/dsv41_engine_qmatmul_bench" \
                 "$strata/dsv41_engine_attention_test" "$strata/dsv41_engine_attention_bench" \
                 "$strata/dsv41_engine_dense_test" "$strata/dsv41_engine_dense_bench" "$strata/dsv41_engine_experts_test" \
                 "$strata/dsv41_engine_smoke" "$strata/dsv41-engine" "$strata/dsv41_cuda_probe"; do
        echo "--- cuobjdump --list-elf ${bin#$FLAVOR_ROOT/} (distinct archs, expected:$ARCHS)"
        found="$(cuobjdump --list-elf "$bin" | grep -o "sm_[0-9]*a\?" | sort | uniq -c)"; echo "$found"
        for a in $ARCHS; do grep -qw "${a}a\?" <<<"$found" || { echo "FAIL: $a missing"; exit 1; }; done
      done

      # every binary the build made carries RUNPATH $ORIGIN; the bundled third-party libraries keep their own
      echo "--- RUNPATH of the tarballs binaries (expected \$ORIGIN)"
      mkdir -p /tmp/rp && cd /tmp/rp
      shopt -s nullglob
      tarballs=("$DIST"/$PATTERN)
      (( ${#tarballs[@]} == 2 )) || { echo "FAIL: expected 2 tarballs $DIST/$PATTERN, found ${#tarballs[@]}"; exit 1; }
      for tgz in "${tarballs[@]}"; do
        tar -xzf "$tgz"
        dir="$(basename "$tgz" .tar.gz)"
        n=0
        for f in "$dir"/bin/*; do
          case "${f##*/}" in libcudart.so.*|libcublas.so.*|libcublasLt.so.*|libgomp.so.*) continue ;; esac
          [[ -L "$f" ]] && continue
          rp="$(readelf -d "$f" | sed -n "s/.*(RUNPATH).*\[\(.*\)\]/\1/p")"
          [[ "$rp" == "\$ORIGIN" ]] || { echo "FAIL: ${f}: RUNPATH [${rp}], expected [\$ORIGIN]"; exit 1; }
          n=$((n + 1))
        done
        echo "$dir: $n binaries with RUNPATH [\$ORIGIN], among them:"
        case "$dir" in llama-dsv41-*) shown=(llama-server dsv41-moe-dump dsv41-moe-replay) ;; strata-*) shown=(strata dsv41_engine_indexer_test dsv41_engine_indexer_bench dsv41_engine_qmatmul_test dsv41_engine_qmatmul_bench dsv41_engine_attention_test dsv41_engine_attention_bench dsv41_engine_dense_test dsv41_engine_dense_bench dsv41_engine_experts_test dsv41_engine_smoke dsv41-engine dsv41_cuda_probe dsv41_cpu_experts_probe) ;; esac
        for exe in "${shown[@]}"; do echo "  bin/$exe: $(readelf -d "$dir/bin/$exe" | grep RUNPATH | tr -s " ")"; done
      done
      cd /

      echo "--- ctest (GPU-free tests)"
      cd "$strata"
      ctest -R "^(conversation_|gguf_reader_test|gguf_split_test|tuning_header_arch_family_test|dsv41_moe_ref_test|dsv41_moe_split_test|dsv41_moe_cpu_fast_test|dsv41_moe_residency_test|dsv41_moe_counts_test|dsv41_moe_prefill_test|dsv41_indexer_index_test)" --output-on-failure
      echo "--- dsv41_moe_ref_test, dsv41_moe_split_test, dsv41_moe_cpu_fast_test, dsv41_moe_residency_test and dsv41_moe_counts_test output"
      sed -n "/^dsv41_moe_ref_test: n_embd/,/^dsv41_moe_ref_test: PASS/p;/^dsv41_moe_split_test: n_embd/,/^dsv41_moe_split_test: PASS/p;/^dsv41_moe_cpu_fast_test: rows/,/^dsv41_moe_cpu_fast_test: PASS/p" Testing/Temporary/LastTest.log
      sed -n "/^Command: .*dsv41_moe_residency_test/,/^dsv41_moe_residency_test: PASS/p" Testing/Temporary/LastTest.log |
        grep -E "^(ok|FAIL|skip|tier map|       |dsv41_moe_residency_test)"
      sed -n "/^Command: .*dsv41_moe_counts_test/,/^dsv41_moe_counts_test: PASS/p" Testing/Temporary/LastTest.log |
        grep -E "^(ok|FAIL|       |dsv41_moe_counts_test)"
      echo "--- dsv41_moe_prefill_test and dsv41_indexer_index_test output (the plan lines: their count and the 4096-token ones)"
      sed -n "/^Command: .*dsv41_moe_prefill_test/,/^dsv41_moe_prefill_test: PASS/p" Testing/Temporary/LastTest.log > /tmp/prefill-test.txt
      echo "$(grep -c "^ok   plan:" /tmp/prefill-test.txt) plan cases ok, $(grep -c "^FAIL" /tmp/prefill-test.txt) failed; those of 4096 tokens:"
      grep -E "^ok   plan: 4096 tokens, slab +(0|147) " /tmp/prefill-test.txt
      grep -E "^(FAIL|ok   (schedule|staging|prefill op)|prefill op:|dsv41_moe_prefill_test)" /tmp/prefill-test.txt
      sed -n "/^dsv41_indexer_index_test: score/,/^dsv41_indexer_index_test: PASS/p" Testing/Temporary/LastTest.log
      # E02 (a): the indexer reference of the engine against the op dumps of the stock fork, bit for bit (GPU free)
      echo "--- engine/indexer_cases.py on $E01/dump-a-1 and dump-b-1, then dsv41_engine_indexer_test --cases"
      for m in a b; do python3 "$STRATA_DIR/tools/dsv41/engine/indexer_cases.py" "$E01/dump-$m-1" > "/tmp/indexer-cases-$m.txt"; done
      grep -h "^# " /tmp/indexer-cases-a.txt /tmp/indexer-cases-b.txt
      "$strata/dsv41_engine_indexer_test" --cases /tmp/indexer-cases-a.txt /tmp/indexer-cases-b.txt
      # E03 (a): the attention reference against the op dumps, bit for bit, both reader modes (GPU free)
      echo "--- engine/attention_cases.py on $E01/dump-{a-1,b-1,a-short,b-short}, then dsv41_engine_attention_test --cases"
      lists=()
      for m in a-1 b-1 a-short b-short; do
        python3 "$STRATA_DIR/tools/dsv41/engine/attention_cases.py" "$E01/dump-$m" > "/tmp/attention-cases-$m.txt"
        lists+=("/tmp/attention-cases-$m.txt")
      done
      grep -h "^# " "${lists[@]}"
      "$strata/dsv41_engine_attention_test" --cases "${lists[@]}"
      # E03/E04a (b) without a GPU: every case of the pod test built on the host, each call through the engine weight
      # checks and the reference numerics it would be compared in (a wrong view type fails here, not on a pod)
      for mode in "" --f16 --mmq; do
        echo "--- dsv41_engine_attention_test --gpu --dry-run $mode"
        "$strata/dsv41_engine_attention_test" --gpu --dry-run $mode | grep -E "^(dry run|dsv41_engine_attention_test)"
      done
      # E04b (a): the dense ops reference against the op dumps, bit for bit (GPU free)
      echo "--- engine/dense_cases.py on $E01/dump-{a-1,b-1,a-short,b-short}, then dsv41_engine_dense_test --cases"
      lists=()
      for m in a-1 b-1 a-short b-short; do
        python3 "$STRATA_DIR/tools/dsv41/engine/dense_cases.py" "$E01/dump-$m" > "/tmp/dense-cases-$m.txt"
        lists+=("/tmp/dense-cases-$m.txt")
      done
      grep -h "^# " "${lists[@]}"
      "$strata/dsv41_engine_dense_test" --cases "${lists[@]}"
      # E04b (b) without a GPU (the E04a lesson): every case of the pod test built on the host, each call through the
      # ops weight checks, the engram hash and the reference numerics of its path
      for mode in "" --mmq; do
        echo "--- dsv41_engine_dense_test --gpu --dry-run $mode"
        "$strata/dsv41_engine_dense_test" --gpu --dry-run $mode | grep -E "^(dry run|dsv41_engine_dense_test)"
      done
      # E05a (a): the CPU experts (strata-moe fast kernel behind plain pointers) against the op dumps, bit for bit
      echo "--- engine/dense_cases.py on $E01/dump-{a-16,b-16} too, then dsv41_engine_experts_test --cases on all six lists"
      for m in a-16 b-16; do
        python3 "$STRATA_DIR/tools/dsv41/engine/dense_cases.py" "$E01/dump-$m" > "/tmp/dense-cases-$m.txt"
        lists+=("/tmp/dense-cases-$m.txt")
      done
      "$strata/dsv41_engine_experts_test" --cases "${lists[@]}"
      # E05a (b) without a GPU: the placements of the pod test (the hot sets the dev pod session uploads), every
      # tier and expert view checked, the prefill plans, the references of every call on the host
      echo "--- engine/hot_sets.py on tiny-a and tiny-b, then dsv41_engine_experts_test --gpu --dry-run"
      for c in a b; do python3 "$STRATA_DIR/tools/dsv41/engine/hot_sets.py" "$E01/tiny-$c.gguf" "/tmp/e05a-hot-$c"; done
      "$strata/dsv41_engine_experts_test" --gpu --dry-run "$E01/tiny-a.gguf" /tmp/e05a-hot-a "$E01/tiny-b.gguf" /tmp/e05a-hot-b |
        grep -E "^(dry run: experts|dry run .*references|dsv41_engine_experts_test)"
      # E05c without a GPU: the grouped prefill GEMM test at the real dims, its shapes, tiles and one-hot references
      echo "--- dsv41_engine_experts_test --gemm --dry-run"
      "$strata/dsv41_engine_experts_test" --gemm --dry-run
      # dsv41-engine --dry-run with the arguments of every end-to-end run of pod/devgpu-run.sh (no CUDA call)
      echo "--- dsv41-engine --dry-run: tiny-a and tiny-b x all-vram, all-cpu, split x (dense 64 + 16, reuse short), and --f16,"
      echo "    each free and with --force-selection on the fork dump (the router ids and index rows of every call read and checked)"
      for d in a-16 b-16 a-short b-short; do
        python3 -c "import json, sys; print(\" \".join(map(str, json.load(open(sys.argv[1]))[\"prompt_token_ids\"])))" \
          "$E01/dump-$d/dump-manifest.json" > "/tmp/e05a-prompt-$d.txt"
      done
      placement_args() {
        case "$2" in
          all-vram) echo "--hot-set /tmp/e05a-hot-$1/all-vram.json --vram-gb 1" ;;
          all-cpu) echo "" ;;
          split) echo "--hot-set /tmp/e05a-hot-$1/split.json --vram-gb 1 --ram-gb 1" ;;
        esac
      }
      for c in a b; do
        case "$c" in a) short_steps=5 ;; b) short_steps=9 ;; esac
        for pl in all-vram all-cpu split; do
          "$strata/dsv41-engine" --model "$E01/tiny-$c.gguf" --tokens-file "/tmp/e05a-prompt-$c-16.txt" --n-predict 16 \
            --ctx 2048 --readers dense $(placement_args "$c" "$pl") --dry-run > "/tmp/e05a-dry-$c-dense-$pl.txt"
          "$strata/dsv41-engine" --model "$E01/tiny-$c.gguf" --tokens-file "/tmp/e05a-prompt-$c-short.txt" \
            --n-predict "$short_steps" --ctx 2048 --readers reuse $(placement_args "$c" "$pl") --dry-run > "/tmp/e05a-dry-$c-reuse-$pl.txt"
          echo "tiny-$c $pl: $(grep -c "^dry run:   layer" "/tmp/e05a-dry-$c-dense-$pl.txt") layers planned; $(grep "^dry run: experts" "/tmp/e05a-dry-$c-dense-$pl.txt")"
          # the path of a decode step in this placement, from the plan ("N layers with a slab", "a doorbell in M of L")
          plan="$(grep "^dry run: experts" "/tmp/e05a-dry-$c-dense-$pl.txt")"
          n_slab="$(grep -oE "each of [0-9]+ layers with a slab" <<<"$plan" | grep -oE "[0-9]+")"
          nums="$(grep -oE "a doorbell in [0-9]+ of [0-9]+ layers" <<<"$plan" | grep -oE "[0-9]+" | xargs)"
          n_ring="${nums% *}" n_all="${nums#* }"
          case "$pl" in
            all-vram) want="$n_all 0" ;;
            all-cpu) want="0 $n_all" ;;
            split) want="$n_all $n_all" ;;
          esac
          [[ "$n_slab $n_ring" == "$want" ]] ||
            { echo "FAIL: tiny-$c $pl decode: $n_slab layers with grouped hits and $n_ring with a doorbell, want $want of $n_all"; exit 1; }
          echo "tiny-$c $pl decode: grouped hits in $n_slab and a doorbell in $n_ring of $n_all layers"
          # the forced runs of devgpu-run.sh (e2e NAME-forced)
          "$strata/dsv41-engine" --model "$E01/tiny-$c.gguf" --tokens-file "/tmp/e05a-prompt-$c-16.txt" --n-predict 16 \
            --ctx 2048 --readers dense $(placement_args "$c" "$pl") --force-selection "$E01/dump-$c-16" --dry-run \
            > "/tmp/e05a-dry-$c-dense-$pl-forced.txt"
          "$strata/dsv41-engine" --model "$E01/tiny-$c.gguf" --tokens-file "/tmp/e05a-prompt-$c-short.txt" \
            --n-predict "$short_steps" --ctx 2048 --readers reuse $(placement_args "$c" "$pl") \
            --force-selection "$E01/dump-$c-short" --dry-run > "/tmp/e05a-dry-$c-reuse-$pl-forced.txt"
          echo "tiny-$c $pl forced: dense $(grep "^dry run: forced" "/tmp/e05a-dry-$c-dense-$pl-forced.txt" | sed "s/.*: //");" \
            "reuse $(grep "^dry run: forced" "/tmp/e05a-dry-$c-reuse-$pl-forced.txt" | sed "s/.*: //")"
        done
      done
      for force in "" "--force-selection $E01/dump-a-16"; do
        "$strata/dsv41-engine" --model "$E01/tiny-a.gguf" --tokens-file /tmp/e05a-prompt-a-16.txt --n-predict 16 --ctx 2048 \
          --readers dense --f16 $force --dry-run | grep -E "^(dsv41-engine|dry run: [0-9]+ layers|dry run: forced)"
      done
      # the multi-chunk forced runs of devgpu-run.sh (a 64-token prompt in chunks of 24 against the one ubatch of
      # the fork, a ring of 2 sets of 2, 2 prefetched experts): the selections of every chunk read from its part of it
      echo "--- dsv41-engine --dry-run, E05b: tiny-a and tiny-b x all-vram, all-cpu, split, --chunk 24 forced on dump-<c>-16"
      for c in a b; do
        for pl in all-vram all-cpu split; do
          "$strata/dsv41-engine" --model "$E01/tiny-$c.gguf" --tokens-file "/tmp/e05a-prompt-$c-16.txt" --n-predict 16 \
            --ctx 2048 --readers dense --chunk 24 --ring-set 2 --ring-sets 2 --prefetch-experts 2 --prefill-split off \
            $(placement_args "$c" "$pl") \
            --force-selection "$E01/dump-$c-16" --dry-run > "/tmp/e05b-dry-$c-chunk24-$pl.txt"
          echo "tiny-$c $pl chunk 24: $(grep "^dry run: [0-9]* layers" "/tmp/e05b-dry-$c-chunk24-$pl.txt" | sed "s/, positions.*//");" \
            "$(grep "^dry run: forced" "/tmp/e05b-dry-$c-chunk24-$pl.txt" | sed "s/.*: //")"
          grep "^dry run: experts" "/tmp/e05b-dry-$c-chunk24-$pl.txt"
        done
      done
      # the smokes of devgpu-run.sh, their arguments and plans (the model is made on the pod only)
      echo "--- dsv41_engine_smoke --dry-run (the E05a smoke, then the E05b measurement)"
      "$strata/dsv41_engine_smoke" --dry-run
      "$strata/dsv41_engine_smoke" --layers 8 --experts 384 --vram-experts-gb 16.12 --chunk 16384 --memory-table \
        --pod-link 4:16 --pod-tflops 154.8 --project "RTX PRO 6000:5:16:96:503.8" --project "L40S:4:16:48:362.05" --dry-run
      # the second measurement smoke of devgpu-run.sh, on an A100 (its link and 312 TFLOPS) with the profile
      echo "--- dsv41_engine_smoke --dry-run --counts (the E05c measurement)"
      "$strata/dsv41_engine_smoke" --layers 8 --experts 384 --vram-experts-gb 16.12 --chunk 16384 --memory-table \
        --pod-link 4:16 --pod-tflops 312 --project "RTX PRO 6000:5:16:96:503.8" --project "L40S:4:16:48:362.05" \
        --counts "$COUNTS" --dry-run
      echo "--- dsv41-engine --force-selection refuses a reference of another prompt"
      rc=0; "$strata/dsv41-engine" --model "$E01/tiny-a.gguf" --tokens-file /tmp/e05a-prompt-a-short.txt --n-predict 5 \
        --ctx 2048 --force-selection "$E01/dump-a-16" --dry-run > /tmp/e05a-forced-guard.txt 2>&1 || rc=$?
      cat /tmp/e05a-forced-guard.txt | tail -n 1; echo "(exit $rc)"
      [[ $rc -ne 0 ]] && grep -q "its prompt is not the one given" /tmp/e05a-forced-guard.txt ||
        { echo "FAIL: --force-selection took the reference of another prompt"; exit 1; }
      echo "--- dsv41-engine --dry-run on $GUARD_MODEL (must be refused)"
      rc=0; "$strata/dsv41-engine" --model "$GUARD_MODEL" --tokens 1 --dry-run > /tmp/e05a-guard.txt 2>&1 || rc=$?
      cat /tmp/e05a-guard.txt; echo "(exit $rc)"
      [[ $rc -ne 0 ]] && grep -q "requires .deepseek41." /tmp/e05a-guard.txt || { echo "FAIL: the architecture guard"; exit 1; }
      echo "--- engine/engine_parity.py on $E01/dump-a-16 against itself, then its self-test (flip attribution, forced runs)"
      python3 "$STRATA_DIR/tools/dsv41/engine/engine_parity.py" "$E01/dump-a-16" "$E01/dump-a-16" | tail -n 4
      python3 "$STRATA_DIR/tools/dsv41/engine/engine_parity.py" --self-test
      # E04a (a): the qmatmul reference on hand-quantized blocks and against the CPU dots of the fork on the tiny GGUFs
      echo "--- dsv41_engine_qmatmul_test --cpu $E01/tiny-a.gguf $E01/tiny-b.gguf"
      "$strata/dsv41_engine_qmatmul_test" --cpu "$E01/tiny-a.gguf" "$E01/tiny-b.gguf"
      # a smoke run only (16 experts, 0.2 GB): the measured table is in tools/dsv41/README.md
      echo "--- dsv41_moe_cpu_bench --experts 16 --threads 2 --tokens 1,4 --reps 2"
      LD_LIBRARY_PATH="$fork:$LD_LIBRARY_PATH" "$strata/dsv41/dsv41_moe_cpu_bench" --experts 16 --threads 2 --tokens 1,4 --reps 2

      echo "--- oracle/hook-clamp-test.sh (the fork hook with a SwiGLU clamp, tiny random DeepSeek-V4, CPU)"
      bash "$STRATA_DIR/tools/dsv41/oracle/hook-clamp-test.sh" "$fork" /tmp/hook-clamp
      echo "--- oracle/hot_set_test.py"
      python3 "$STRATA_DIR/tools/dsv41/oracle/hot_set_test.py"
      echo "--- pod/t04a-lib-test.sh"
      bash "$STRATA_DIR/tools/dsv41/pod/t04a-lib-test.sh"
      # then the T04c 75-minute selection (stock + hook-70 + hook-70-ram40) and the T04d session (profile, auto warm
      # tier); a dry run whose plan is over CAP_MIN exits 2
      for mode in "" --speed "--speed SPEED_BUDGETS=70 RAM_RUNS=70:40 CPU_KERNELS=fast EXTRA=0" \
                  "--speed --profile SPEED_BUDGETS=70 RAM_RUNS=auto CPU_KERNELS=fast EXTRA=0 CAP_MIN=60"; do
        echo "--- pod/t04a.sh --dry-run $mode"
        read -r -a words <<< "$mode"
        flags=() envs=()
        for w in "${words[@]}"; do if [[ "$w" == --* ]]; then flags+=("$w"); else envs+=("$w"); fi; done
        env "${envs[@]}" bash "$STRATA_DIR/tools/dsv41/pod/t04a.sh" --dry-run "${flags[@]}" > /tmp/t04a-dry.txt
        echo "$(wc -l < /tmp/t04a-dry.txt) lines, exit 0; the steps: $(sed -n "s/^+ \[\([^]]*\)\].*/\1/p" /tmp/t04a-dry.txt | grep -v -- "-[0-9]*-\(warm\|measured\)$" | tr "\n" " ")"
        grep -E "^plan: total|plus .* outside the script" /tmp/t04a-dry.txt
      done
      # the tokenizer (bit for bit the fork tokenizer), the server, the serve protocol
      echo "--- E06: tools/llama_unicode_flags.txt against src/unicode-data.cpp of the fork"
      python3 "$STRATA_DIR/tools/dsv41/engine/llama_unicode.py" check "$FORK_DIR" "$STRATA_DIR/tools/llama_unicode_flags.txt"
      echo "--- E06: engine/tiny_vocab.py (the first 512 tokens of $VOCAB), then engine/tokenizer_parity.py against llama-tokenize"
      python3 "$STRATA_DIR/tools/dsv41/engine/tiny_vocab.py" "$VOCAB" /tmp/tiny-vocab.gguf
      python3 "$STRATA_DIR/tools/dsv41/engine/tokenizer_parity.py" \
        --llama-tokenize "env LD_LIBRARY_PATH=$fork:$LD_LIBRARY_PATH $fork/llama-tokenize" \
        --corpus "$STRATA_DIR/tools/dsv41/engine/tokenizer_corpus.txt" "$VOCAB" /tmp/tiny-vocab.gguf
      echo "--- E06: the tests of the server (python3 -m unittest discover serve, from a writable directory)"
      mkdir -p /tmp/serve-tests
      (cd /tmp/serve-tests && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$STRATA_DIR/serve" -t "$STRATA_DIR" \
        -p "test_*.py" > /tmp/serve-tests.log 2>&1) || { tail -n 60 /tmp/serve-tests.log; echo "FAIL: the server tests"; exit 1; }
      tail -n 3 /tmp/serve-tests.log
      echo "--- E06: engine/serve_check.py --reference dry (dsv41-engine --serve --dry-run on tiny-b behind serve/server.py)"
      PYTHONDONTWRITEBYTECODE=1 python3 "$STRATA_DIR/tools/dsv41/engine/serve_check.py" --engine "$strata/dsv41-engine" \
        --model "$E01/tiny-b.gguf" --tokenizer-gguf /tmp/tiny-vocab.gguf --vocab "$VOCAB" --reference dry \
        --out /tmp/serve-check > /tmp/serve-check.log 2>&1 || { tail -n 40 /tmp/serve-check.log; exit 1; }
      echo "$(grep -c "^ok " /tmp/serve-check.log) checks ok, among them:"
      grep -E "^ok   (protocol: (live resume|prompt-end restore|a diverging|[0-9]+ tokens fed|STOP|ERR|a sampled|the same sampled|seed 1235|two requests without)|http: (the content|the streamed|the second turn resumes|tools|seed 0 twice|frequency_penalty))" /tmp/serve-check.log
      echo "--- dsv41_engine_sampler_test (the host sampler: masks, draws, seeds, the lazy sort, the Session pick)"
      "$strata/dsv41_engine_sampler_test" > /tmp/q03-sampler.txt || { cat /tmp/q03-sampler.txt; echo "FAIL: the sampler test"; exit 1; }
      tail -n 1 /tmp/q03-sampler.txt
      tail -n 1 /tmp/serve-check.log
      echo "--- dsv41_engine_experts_test --determinism (the deterministic mode on the CPU reference path, tiny-b)"
      "$strata/dsv41_engine_experts_test" --determinism "$E01/tiny-b.gguf" > /tmp/q03b-det.txt ||
        { cat /tmp/q03b-det.txt; echo "FAIL: the determinism test"; exit 1; }
      grep -E "^(ok|FAIL) " /tmp/q03b-det.txt
      tail -n 1 /tmp/q03b-det.txt
      echo "--- dsv41_engine_session_test (a promoted request: its last token fed by decode, as alone)"
      "$strata/dsv41_engine_session_test" > /tmp/c05-session.txt || { cat /tmp/c05-session.txt; echo "FAIL: the session test"; exit 1; }
      echo "$(grep -c "^ok " /tmp/c05-session.txt) checks ok"
      tail -n 1 /tmp/c05-session.txt
      echo "--- dsv41_engine_checkpoint_test (the conversation checkpoints on the CPU reference path, tiny-b)"
      "$strata/dsv41_engine_checkpoint_test" "$E01/tiny-b.gguf" > /tmp/c02-checkpoint.txt ||
        { cat /tmp/c02-checkpoint.txt; echo "FAIL: the checkpoint test"; exit 1; }
      echo "$(grep -c "^ok " /tmp/c02-checkpoint.txt) checks ok"
      tail -n 1 /tmp/c02-checkpoint.txt
      echo "--- dsv41_engine_batch_test (the batch slots on the CPU reference path, tiny-b)"
      "$strata/dsv41_engine_batch_test" "$E01/tiny-b.gguf" > /tmp/c03-batch.txt ||
        { cat /tmp/c03-batch.txt; echo "FAIL: the batch test"; exit 1; }
      echo "$(grep -c "^ok " /tmp/c03-batch.txt) checks ok"
      tail -n 1 /tmp/c03-batch.txt
      echo "--- dsv41_engine_expert_bits_test (the CPU experts in the GPU numerics against mmvq_ref.hpp)"
      "$strata/dsv41_engine_expert_bits_test" > /tmp/c05-expert-bits.txt ||
        { cat /tmp/c05-expert-bits.txt; echo "FAIL: the expert bits test"; exit 1; }
      echo "$(grep -c "^ok " /tmp/c05-expert-bits.txt) checks ok"
      tail -n 1 /tmp/c05-expert-bits.txt
      echo "--- dsv41-engine --serve --dry-run --batch 4 (a request in a batch slot over the protocol)"
      # slot 0 in the deterministic mode (rep=1 beside slot 1), slot 1 with det=0 (rep=0), in the GPU numerics
      # (the fork numerics are the default now)
      printf "BGEN 0 6 10,11,12,13\nBGEN 1 4 det=0 20,21,22\n" |
        "$strata/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 8 --batch 4 --batch-ctx 128 \
        --cpu-numerics gpu --dry-run \
        > /tmp/c03-batch-serve.txt 2> /tmp/c03-batch-serve.err ||
        { cat /tmp/c03-batch-serve.txt /tmp/c03-batch-serve.err; echo "FAIL: the batch protocol"; exit 1; }
      grep -q "^BADM 0 1" /tmp/c03-batch-serve.txt && grep -q "^BADM 1 1" /tmp/c03-batch-serve.txt &&
        grep -q "^BDONE 0 6 length .* rep=1$" /tmp/c03-batch-serve.txt && grep -q "^BDONE 1 4 length .* rep=0$" /tmp/c03-batch-serve.txt ||
        { cat /tmp/c03-batch-serve.txt /tmp/c03-batch-serve.err; echo "FAIL: the batch protocol"; exit 1; }
      grep -E "^(BADM|BDONE) " /tmp/c03-batch-serve.txt
      echo "--- dsv41-engine --serve --dry-run --interleave-chunk 8 (a 40-token prompt beside a decoding slot)"
      printf "BGEN 0 30 10,11,12,13\nBGEN 1 4 %s\n" "$(seq -s, 40 79)" |
        "$strata/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 16 --batch 4 --batch-ctx 128 \
        --interleave-chunk 8 --dry-run > /tmp/c04-interleave.txt 2> /tmp/c04-interleave.err ||
        { cat /tmp/c04-interleave.txt /tmp/c04-interleave.err; echo "FAIL: the interleaved read"; exit 1; }
      [[ "$(sed -n "/^PP 8 40 /,/^BADM 1 1\$/p" /tmp/c04-interleave.txt | grep -cE "^(PP [0-9]+ 40|BT 0) ")" == 9 ]] &&
        grep -q "read beside 1 decoding batch slots, 40 tokens in 5 chunks of 8 with 4 batch steps between them" /tmp/c04-interleave.err ||
        { cat /tmp/c04-interleave.txt /tmp/c04-interleave.err; echo "FAIL: the interleaved read"; exit 1; }
      grep -o "read beside .* between them" /tmp/c04-interleave.err
      echo "--- E06: dsv41-engine --serve refuses --tokens (its prompts come on stdin)"
      rc=0; "$strata/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --tokens 1 --dry-run > /tmp/e06-serve-guard.txt 2>&1 || rc=$?
      tail -n 1 /tmp/e06-serve-guard.txt; echo "(exit $rc)"
      [[ $rc -ne 0 ]] && grep -q "takes its prompts on stdin" /tmp/e06-serve-guard.txt || { echo "FAIL: --serve with --tokens"; exit 1; }
      echo "--- E06: dsv41_engine_smoke --prefix-cache 512 --dry-run (the prefix-cache measurement of the pod)"
      "$strata/dsv41_engine_smoke" --layers 8 --experts 384 --vram-experts-gb 16.12 --chunk 16384 --counts "$COUNTS" \
        --prefix-cache 512 --pod-link 4:16 --floor-target "RTX PRO 6000:5:16" --dry-run | tail -n 3
      # the prefill prompts from the fork at its base commit, counted with its llama-tokenize, then the session
      echo "--- pod/prefill_prompts.py build (llama-dsv41 5210c7c5, $VOCAB) and check"
      python3 "$STRATA_DIR/tools/dsv41/pod/prefill_prompts.py" build --fork "$FORK_DIR" --out /tmp/prefill-prompts \
        --tokenizer "env LD_LIBRARY_PATH=$fork:$LD_LIBRARY_PATH $fork/llama-tokenize -m $VOCAB --show-count --log-disable -f"
      python3 "$STRATA_DIR/tools/dsv41/pod/prefill_prompts.py" check /tmp/prefill-prompts
      # the real-model session, everything of it a machine without a GPU can run
      echo "--- engine/oracle_parity.py --self-test"
      python3 "$STRATA_DIR/tools/dsv41/engine/oracle_parity.py" --self-test
      echo "--- engine/gguf_layout.py split (tiny-b in 3 shards), then dsv41-engine --dry-run on it against the whole file"
      python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" split "$E01/tiny-b.gguf" 3 /tmp/e07-split
      for args in "--tokens-file /tmp/e05a-prompt-b-16.txt --n-predict 16 --ctx 2048 --readers dense $(placement_args b split)" \
                  "--tokens-file /tmp/e05a-prompt-b-16.txt --n-predict 16 --ctx 2048 --readers dense $(placement_args b split) --force-selection $E01/dump-b-16"; do
        for m in "$E01/tiny-b.gguf" /tmp/e07-split/tiny-b-00001-of-00003.gguf; do
          # shellcheck disable=SC2086
          "$strata/dsv41-engine" --model "$m" $args --dry-run | sed "s|$m|MODEL|g" > "/tmp/e07-dry-$(basename "$m").txt"
        done
        diff /tmp/e07-dry-tiny-b.gguf.txt /tmp/e07-dry-tiny-b-00001-of-00003.gguf.txt > /dev/null ||
          { diff /tmp/e07-dry-tiny-b.gguf.txt /tmp/e07-dry-tiny-b-00001-of-00003.gguf.txt | head -n 20; echo "FAIL: the split model plans differently"; exit 1; }
        echo "the split tiny-b plans as the whole: $(wc -l < /tmp/e07-dry-tiny-b.gguf.txt) dry-run lines equal ($(grep "^dry run: forced" /tmp/e07-dry-tiny-b.gguf.txt | sed "s/.*: //" || echo free))"
      done
      echo "--- engine/gguf_layout.py sparse ($HEADERS: the real model at its 7 shard sizes, sparse) and sizes"
      python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" sparse "$HEADERS" /tmp/e07-real | tail -n 1
      python3 "$STRATA_DIR/tools/dsv41/engine/gguf_layout.py" sizes /tmp/e07-real | tail -n 1
      echo "--- pod/e07_prompts.py (built once into $BUILD_ROOT/e07/prompts) check"
      [[ -f "$BUILD_ROOT/e07/prompts/manifest.json" ]] || python3 "$STRATA_DIR/tools/dsv41/pod/e07_prompts.py" build \
        --fork "$FORK_DIR" --vocab "$VOCAB" --out "$BUILD_ROOT/e07/prompts"
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_prompts.py" check "$BUILD_ROOT/e07/prompts"
      echo "--- pod/e07_bench.py, every phase, against dsv41-engine --serve --dry-run (tiny-b, the ids mod 512)"
      mkdir -p /tmp/e07-tiny-prompts
      for n in short p16k turn2 p64k p256k; do
        python3 -c "import sys; print(\" \".join(str(int(x) % 512) for x in open(sys.argv[1]).read().split()))" \
          "$BUILD_ROOT/e07/prompts/$n.ids" > "/tmp/e07-tiny-prompts/$n.ids"
      done
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_bench.py" --engine "$strata/dsv41-engine" \
        --engine-args "--model $E01/tiny-b.gguf --ctx 262144 --chunk 16384 --dry-run" --prompts /tmp/e07-tiny-prompts \
        --phases 16k,turn2,short,64k,256k,short-timed --out /tmp/e07-bench-dry.json | grep -E "^e07_bench.py: (16k|turn2|256k|short-timed|done|FAIL)"
      python3 -c "import json, sys; j = json.load(open(sys.argv[1])); p = {x[\"phase\"]: x for x in j[\"phases\"]}
assert j[\"ok\"] and j[\"alive_after\"] and p[\"turn2\"][\"resume\"] == 16415 and p[\"256k\"][\"generated\"] == 481, j
assert p[\"short-timed\"][\"generated\"] == 65, j
print(\"e07_bench.py dry: turn 2 resumed 16415 ids, 256k generated 481 at ctx 262144, short-timed (times=1) 65, the engine still answers\")" /tmp/e07-bench-dry.json
      # the isolation tools on tiny-b, whose layers 0 and 1 are window-only with
      # the engram before layer 1, as in the real model
      echo "--- engine/layer_parity.py --self-test"
      python3 "$STRATA_DIR/tools/dsv41/engine/layer_parity.py" --self-test
      echo "--- dsv41-layer-ref on tiny-b (dump-b-1: 64 tokens, layers 0 and 1, its expert ids and the isolated MoE) per numerics,"
      echo "    then layer_parity.py ops: fork-cpu bit for bit against the op dump of the fork, and compare"
      python3 -c "import json, sys; print(\" \".join(map(str, json.load(open(sys.argv[1]))[\"prompt_token_ids\"])))" \
        "$E01/dump-b-1/dump-manifest.json" > /tmp/e07c2-prompt-b1.txt
      for n in fork-cpu fork-cuda engine; do
        "$strata/dsv41-layer-ref" --model "$E01/tiny-b.gguf" --tokens-file /tmp/e07c2-prompt-b1.txt --numerics "$n" \
          --out "/tmp/e07c2-lr/$n" --oracle "$E01/dump-b-1" | tail -n 1 | sed "s/ in [0-9.]* s$//"
      done
      python3 "$STRATA_DIR/tools/dsv41/engine/layer_parity.py" ops /tmp/e07c2-lr/fork-cpu "$E01/dump-b-1"
      python3 "$STRATA_DIR/tools/dsv41/engine/layer_parity.py" compare --oracle "$E01/dump-b-1" --json /tmp/e07c2-lp.json \
        /tmp/e07c2-lr/fork-cpu /tmp/e07c2-lr/fork-cuda /tmp/e07c2-lr/engine | grep -E "^  (oracle|entry|reading)"
      python3 -c "import json, sys; j = json.load(open(sys.argv[1]))
fc = [r for r in j[\"oracle_rows\"] if r[\"numerics\"] == \"fork-cpu\"]
assert len(fc) == 2 and all(r[\"cur\"] == 0 and r[\"out\"] == 0 and r[\"out_isolated\"] == 0 for r in fc), fc
print(\"layer_parity.py compare: fork-cpu cur, out and isolated out equal the fork dump (0) at layers 0 and 1\")" /tmp/e07c2-lp.json
      echo "--- dsv41-engine --dry-run with --forced-moe-input and with --prefill-mmq (tiny-b forced on dump-b-16)"
      for extra in --forced-moe-input --prefill-mmq; do
        "$strata/dsv41-engine" --model "$E01/tiny-b.gguf" --tokens-file /tmp/e05a-prompt-b-16.txt --n-predict 16 --ctx 2048 \
          --readers dense $(placement_args b split) --force-selection "$E01/dump-b-16" "$extra" --dry-run |
          grep -E "^(prompt:|dry run: forced|dry run: prefill)"
      done
      echo "--- dsv41-engine refuses --forced-moe-input without --force-selection"
      rc=0; "$strata/dsv41-engine" --model "$E01/tiny-b.gguf" --tokens-file /tmp/e05a-prompt-b-16.txt --n-predict 16 \
        --ctx 2048 --forced-moe-input --dry-run > /tmp/e07c2-guard.txt 2>&1 || rc=$?
      tail -n 1 /tmp/e07c2-guard.txt; echo "(exit $rc)"
      [[ $rc -ne 0 ]] && grep -q "takes the reference of --force-selection" /tmp/e07c2-guard.txt ||
        { echo "FAIL: --forced-moe-input ran without a reference"; exit 1; }
      echo "--- pod/e07_report.py on an empty session"
      mkdir -p /tmp/e07-empty/20261008T000000Z/out
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" /tmp/e07-empty/20261008T000000Z /tmp/e07-sheets
      grep -c "no result" /tmp/e07-sheets/2026-10-08-e07a/README.md | sed "s/$/ sections name a missing result/"
      echo "--- pod/e07_report.py on an empty session whose out/plan.log says readers: dense"
      mkdir -p /tmp/e07-empty-dense/20261008T000000Z/out
      echo "plan: cycle2, readers: dense" > /tmp/e07-empty-dense/20261008T000000Z/out/plan.log
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" /tmp/e07-empty-dense/20261008T000000Z /tmp/e07-sheets-dense
      sheet=/tmp/e07-sheets-dense/2026-10-08-e07a-cycle2/README.md
      ! grep -E "^\| reuse \| [a-z]+ \| no result" "$sheet" && [[ $(grep -c "^| reuse | .* | not run (E07_READERS=dense)" "$sheet") -eq 2 ]] ||
        { echo "FAIL: the sheet names the skipped reuse runs as missing results"; exit 1; }
      grep "^| reuse " "$sheet"
      echo "--- pod/e07.sh --dry-run (the E07a session, E07_PLAN=cycle2: the plan, the gate per GPU, the engine and the CPU reference on the real layout)"
      DRY_ENGINE="$strata/dsv41-engine" BUILD_ROOT="$BUILD_ROOT" COUNTS="$COUNTS" HEADERS="$HEADERS" VOCAB="$VOCAB" \
        FORK_DIR="$FORK_DIR" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run
      echo "--- pod/e07.sh --dry-run with E07_PLAN=full and with E07_READERS=dense: their plans against the cap (reported:"
      echo "    exit 2 is a plan over the cap at the limit price)"
      for knob in E07_PLAN=full E07_READERS=dense; do
        rc=0; env "$knob" BUILD_ROOT="$BUILD_ROOT" COUNTS="$COUNTS" HEADERS="$HEADERS" VOCAB="$VOCAB" FORK_DIR="$FORK_DIR" \
          bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > /tmp/e07c2-knob.txt 2>&1 || rc=$?
        (( rc == 0 || rc == 2 )) || { tail -n 5 /tmp/e07c2-knob.txt; echo "FAIL: e07.sh --dry-run with $knob exited $rc"; exit 1; }
        grep -E "^plan \((volume|hf)\): total" /tmp/e07c2-knob.txt | sed "s/^/$knob: /"; echo "$knob: (exit $rc)"
      done
      # the per-request hot set, everything of it a machine without a GPU can run
      echo "--- E08: dsv41_engine_hot_policy_test (the planner: swaps at every token, the prompt target, hysteresis, the static reset,"
      echo "    two turns of a conversation: turn 2 resumed against fresh and continued; a batch opened by a promoted"
      echo "    request swaps and misses about as often a row as the request alone)"
      "$strata/dsv41_engine_hot_policy_test"
      sim="$STRATA_DIR/tools/dsv41/engine/hot_policy_sim.py"
      echo "--- E08: engine/hot_policy_sim.py --self-test (a skewed synthetic trace: prompt and adaptive above static; two turns)"
      python3 "$sim" --self-test | tail -n 2
      echo "--- E08: engine/hot_policy_sim.py two-turn (4 layers of 384 experts, 119 in the slab; turn 1 1024 prompt + 256 decode"
      echo "    tokens, turn 2 a 50-token suffix + 256): turn 2 resumed hits >= fresh (within 1% of continued), swaps <= 1/4 of fresh"
      python3 "$sim" two-turn /tmp/e08-two | tail -n 1
      echo "--- E08: the simulator against the planner (dsv41_engine_hot_policy_test --replay): the same hits and swaps per"
      echo "    layer on a synthetic trace, the two-turn traces (resumed and fresh) and the T02b oracle (31 prompt ids, 64"
      echo "    decode steps; the A100 gate hot set of the e07.sh dry run)"
      python3 "$sim" synthetic /tmp/e08-syn > /dev/null
      python3 "$sim" oracle "$BUILD_ROOT/oracle-dsv41/run-1" /tmp/e08-oracle-trace.txt
      oracle_hot="$BUILD_ROOT/e07/gate-dry-81920/hot-set.json"
      for t in "/tmp/e08-syn/trace.txt /tmp/e08-syn/hot-set.json" "/tmp/e08-two/trace-resume.txt /tmp/e08-two/hot-set.json" \
               "/tmp/e08-two/trace-fresh.txt /tmp/e08-two/hot-set.json" "/tmp/e08-oracle-trace.txt $oracle_hot"; do
        read -r tr hs <<<"$t"
        for knobs in "" "--policy static" "--policy prompt --swaps 3 --spare 3" "--margin 0:-1" \
                     "--half-life 16 --margin 1:0 --mapping-pinned"; do
          # shellcheck disable=SC2086  # the knobs are whole words
          python3 "$sim" replay "$tr" --hot-set "$hs" $knobs > /tmp/e08-py.txt
          # shellcheck disable=SC2086
          "$strata/dsv41_engine_hot_policy_test" --replay "$tr" --hot-set "$hs" $knobs > /tmp/e08-cc.txt
          diff -q /tmp/e08-py.txt /tmp/e08-cc.txt > /dev/null ||
            { diff /tmp/e08-py.txt /tmp/e08-cc.txt | head -n 10; echo "FAIL: the simulator and the planner differ ($tr $knobs)"; exit 1; }
          echo "$(basename "$tr") ${knobs:-(the defaults)}: $(tail -n 1 /tmp/e08-cc.txt), the same in both"
        done
      done
      echo "--- E08: engine/hot_policy_sim.py compare on the T02b oracle trace: prompt and adaptive above static"
      python3 "$sim" compare /tmp/e08-oracle-trace.txt --hot-set "$oracle_hot" > /tmp/e08-oracle-compare.txt
      python3 -c "import sys; w = open(sys.argv[1]).read().splitlines()[-1].split(); s, p, a = map(float, w[1:4])
assert w[0] == \"all\" and p > s and a > s, w
print(f\"oracle trace, the share of decode pairs in VRAM: static {s:.4f}, prompt {p:.4f}, adaptive {a:.4f}\")" /tmp/e08-oracle-compare.txt
      echo "--- E08: dsv41-engine --dry-run of the forced-swap runs of devgpu-run.sh (tiny-a and tiny-b, split, forced)"
      for c in a b; do
        for hot in "--hot-policy static" "--hot-policy adaptive --hot-margin 0:-1 --hot-swaps 2 --hot-spare 2 --hot-check"; do
          # shellcheck disable=SC2086
          "$strata/dsv41-engine" --model "$E01/tiny-$c.gguf" --tokens-file "/tmp/e05a-prompt-$c-16.txt" --n-predict 16 \
            --ctx 2048 --readers dense $(placement_args "$c" split) $hot --force-selection "$E01/dump-$c-16" \
            --route-trace /tmp/e08-trace-dry.txt --dry-run | grep -E "^(experts: hot set|dry run: forced)"
        done
      done
      echo "--- E08: dsv41-engine refuses an unknown policy, a margin without its two numbers, a half-life of 0, no swaps with spare slots"
      for bad in "--hot-policy lru" "--hot-margin 1.5" "--hot-half-life 0" "--hot-spare 2 --hot-swaps 0"; do
        # shellcheck disable=SC2086
        rc=0; "$strata/dsv41-engine" --model "$E01/tiny-b.gguf" --tokens 1 $(placement_args b split) $bad --dry-run \
          > /tmp/e08-guard.txt 2>&1 || rc=$?
        [[ $rc -ne 0 ]] || { echo "FAIL: dsv41-engine took $bad"; exit 1; }
        echo "$bad: refused (exit $rc: $(tail -n 1 /tmp/e08-guard.txt))"
      done
      echo "--- E08: dsv41-engine --serve --dry-run takes hot=, hot_reset=1, trace=0, prefill_times=1 and refuses hot=lru"
      printf "GEN 3 hot=prompt hot_reset=1 trace=0 prefill_times=1 5,6,7\nGEN 2 hot=lru 1,2\nQUIT\n" |
        "$strata/dsv41-engine" --serve --model "$E01/tiny-b.gguf" --ctx 256 --chunk 64 $(placement_args b split) --dry-run \
        2> /dev/null > /tmp/e08-serve.txt
      grep -E "^(DONE|ERR)" /tmp/e08-serve.txt
      grep -q "^DONE 3 " /tmp/e08-serve.txt && grep -q "^ERR hot=lru" /tmp/e08-serve.txt ||
        { cat /tmp/e08-serve.txt; echo "FAIL: the serve protocol hot set keys"; exit 1; }
      echo "--- E08: pod/e07_bench.py, the e08 phases, against dsv41-engine --serve --dry-run (tiny-b, the ids mod 512)"
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_bench.py" --engine "$strata/dsv41-engine" \
        --engine-args "--model $E01/tiny-b.gguf --ctx 262144 --chunk 16384 --dry-run" --prompts /tmp/e07-tiny-prompts \
        --phases short-static,short-prompt,short-adaptive,16k-e08,short-timed,64k,256k --out /tmp/e08-bench-dry.json |
        grep -E "^e07_bench.py: (short-adaptive|16k-e08|256k|done|FAIL)"
      python3 -c "import json, sys; j = json.load(open(sys.argv[1])); p = {x[\"phase\"]: x for x in j[\"phases\"]}
assert j[\"ok\"] and all(p[n][\"generated\"] == 481 for n in (\"short-static\", \"short-prompt\", \"short-adaptive\", \"16k-e08\", \"256k\")), j
print(\"e07_bench.py dry: the three policies, 16k-e08 and 256k generated 481 each, the engine still answers\")" /tmp/e08-bench-dry.json
      echo "--- E08: pod/e07_report.py on an empty E07_PLAN=e08 session"
      mkdir -p /tmp/e08-empty/20261009T000000Z/out
      echo "plan: e08, readers: dense reuse" > /tmp/e08-empty/20261009T000000Z/out/plan.log
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" /tmp/e08-empty/20261009T000000Z /tmp/e08-sheets
      grep -q "^## Hot set (bench-e08)" /tmp/e08-sheets/2026-10-09-e08/README.md || { echo "FAIL: no e08 sheet"; exit 1; }
      echo "--- E08: pod/e07.sh --dry-run with E07_PLAN=e08 MAX_USD_PER_HR=1.95 (the session: under 3 USD with the model from HF)"
      E07_PLAN=e08 MAX_USD_PER_HR=1.95 DRY_ENGINE="$strata/dsv41-engine" BUILD_ROOT="$BUILD_ROOT" COUNTS="$COUNTS" \
        HEADERS="$HEADERS" VOCAB="$VOCAB" FORK_DIR="$FORK_DIR" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > /tmp/e08-dry.txt
      grep -E "^plan \((volume|hf)\): total|^    e07_gate.py: VRAM|^experts: hot set" /tmp/e08-dry.txt
      python3 -c "import re, sys; m = re.search(r\"^plan \(hf\): total ([0-9.]+) min .*about ([0-9.]+) USD at 1.95 USD/h\", open(sys.argv[1]).read(), re.M)
assert m and float(m.group(2)) < 3.0, m
print(f\"E07_PLAN=e08 with the model from HF: {m.group(1)} min, {m.group(2)} USD at 1.95 USD/h: under 3 USD\")" /tmp/e08-dry.txt
      # the 16K prompt, everything of it a machine without a GPU can run
      echo "--- E09: dsv41_engine_qdequant_test (the grouped GEMM table decoder of q2_K and q3_K against the dequant reference)"
      "$strata/dsv41_engine_qdequant_test"
      echo "--- E09: pod/e07_bench.py, the e09 phases, against dsv41-engine --serve --dry-run (tiny-b, the ids mod 512)"
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_bench.py" --engine "$strata/dsv41-engine" \
        --engine-args "--model $E01/tiny-b.gguf --ctx 262144 --chunk 16384 --prefetch-experts 128 --dry-run" \
        --prompts /tmp/e07-tiny-prompts --phases 16k-e09,short-adaptive,16k-e09,256k --out /tmp/e09-bench-dry.json |
        grep -E "^e07_bench.py: (16k-e09|256k|done|FAIL)"
      python3 -c "import json, sys; j = json.load(open(sys.argv[1])); p = j[\"phases\"]
assert j[\"ok\"] and [x[\"phase\"] for x in p] == [\"16k-e09\", \"short-adaptive\", \"16k-e09\", \"256k\"], j
assert p[0][\"generated\"] == 32 and p[2][\"generated\"] == 32 and p[2][\"reused\"] < 64, p[2]
r = p[2][\"reused\"]
print(f\"e07_bench.py dry: 16k-e09 twice (32 generated each, the second reusing {r} ids), short-adaptive and 256k\")" /tmp/e09-bench-dry.json
      echo "--- E09: engine/two_turn_check.py --dry-run (tiny-b behind serve/server.py): turn 2 resumes"
      PYTHONDONTWRITEBYTECODE=1 python3 "$STRATA_DIR/tools/dsv41/engine/two_turn_check.py" --engine "$strata/dsv41-engine" \
        --model "$E01/tiny-b.gguf" --tokenizer-gguf /tmp/tiny-vocab.gguf --ctx 2048 --max-tokens 12 \
        --engine-args "--prefetch-experts 128" --out /tmp/e09-two-turn --dry-run
      echo "--- E09: pod/e07_report.py on an empty E07_PLAN=e09 session"
      mkdir -p /tmp/e09-empty/20261009T000000Z/out
      echo "plan: e09, readers: dense reuse" > /tmp/e09-empty/20261009T000000Z/out/plan.log
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" /tmp/e09-empty/20261009T000000Z /tmp/e09-sheets
      grep -q "^## The 16K prompt and the second turn" /tmp/e09-sheets/2026-10-09-e09/README.md || { echo "FAIL: no e09 sheet"; exit 1; }
      echo "--- E09: pod/e07.sh --dry-run with E07_PLAN=e09 MAX_USD_PER_HR=1.95 (the session: within 1.50 USD with the model from HF)"
      E07_PLAN=e09 MAX_USD_PER_HR=1.95 DRY_ENGINE="$strata/dsv41-engine" BUILD_ROOT="$BUILD_ROOT" COUNTS="$COUNTS" \
        HEADERS="$HEADERS" VOCAB="$VOCAB" FORK_DIR="$FORK_DIR" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > /tmp/e09-dry.txt
      grep -E "^plan \((volume|hf)\): total|^    e07_gate.py: VRAM|^experts: (hot set|prefill ring)" /tmp/e09-dry.txt
      python3 -c "import re, sys; t = open(sys.argv[1]).read(); m = re.search(r\"^plan \(hf\): total ([0-9.]+) min .*about ([0-9.]+) USD at 1.95 USD/h\", t, re.M)
assert m and float(m.group(2)) <= 1.5, m
assert \"a prefetch region of 128 experts\" in t, \"the gate without the E09 prefetch\"
print(f\"E07_PLAN=e09 with the model from HF: {m.group(1)} min, {m.group(2)} USD at 1.95 USD/h: within 1.50 USD\")" /tmp/e09-dry.txt
      # the tooling of the closing session, everything of it a machine without a GPU can run
      echo "--- E07b: pod/e07b-test.sh (per-step timeouts, the HF download guard, the goal table on a fixture session)"
      bash "$STRATA_DIR/tools/dsv41/pod/e07b-test.sh" > /tmp/e07b-test.log 2>&1 || { cat /tmp/e07b-test.log; exit 1; }
      grep -E "^(ok|FAIL) " /tmp/e07b-test.log; tail -n 1 /tmp/e07b-test.log
      echo "--- E07b: pod/e07_bench.py, the e07b phases, against dsv41-engine --serve --dry-run (tiny-b, the ids mod 512)"
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_bench.py" --engine "$strata/dsv41-engine" \
        --engine-args "--model $E01/tiny-b.gguf --ctx 262144 --chunk 16384 --prefetch-experts 128 --dry-run" \
        --prompts /tmp/e07-tiny-prompts --phases 16k-e09,turn2,short-adaptive,64k,256k --out /tmp/e07b-bench-dry.json |
        grep -E "^e07_bench.py: (turn2|64k|done|FAIL)"
      python3 -c "import json, sys; j = json.load(open(sys.argv[1])); p = {x[\"phase\"]: x for x in j[\"phases\"]}
assert j[\"ok\"] and list(p) == [\"16k-e09\", \"turn2\", \"short-adaptive\", \"64k\", \"256k\"] and p[\"turn2\"][\"resume\"] == 16415, j
print(\"e07_bench.py dry: turn2 resumed 16415 ids after 16k-e09, then short-adaptive, 64k and 256k\")" /tmp/e07b-bench-dry.json
      echo "--- E07b: engine/two_turn_check.py --dry-run --http (tiny-b behind serve/server.py: the turns, then the HTTP check)"
      PYTHONDONTWRITEBYTECODE=1 python3 "$STRATA_DIR/tools/dsv41/engine/two_turn_check.py" --engine "$strata/dsv41-engine" \
        --model "$E01/tiny-b.gguf" --tokenizer-gguf /tmp/tiny-vocab.gguf --ctx 2048 --max-tokens 12 \
        --engine-args "--prefetch-experts 128" --http --out /tmp/e07b-two-turn --dry-run > /tmp/e07b-two-turn.log 2>&1 ||
        { tail -n 30 /tmp/e07b-two-turn.log; exit 1; }
      grep -E "^(ok   http|two_turn_check.py: (turn 2|http|PASS))" /tmp/e07b-two-turn.log
      echo "--- E07b: pod/e07_report.py on an empty E07_PLAN=e07b session: every goal row has a verdict and a cause"
      mkdir -p /tmp/e07b-empty/20261011T000000Z/out
      echo "plan: e07b, readers: dense" > /tmp/e07b-empty/20261011T000000Z/out/plan.log
      python3 "$STRATA_DIR/tools/dsv41/pod/e07_report.py" /tmp/e07b-empty/20261011T000000Z /tmp/e07b-sheets
      goal="$(grep -E "^\| [1-6] \|" /tmp/e07b-sheets/2026-10-11-e07b/README.md)"
      [[ $(wc -l <<<"$goal") -eq 6 ]] && ! awk -F" [|] " "\$4 == \"FAIL\" && \$5 ~ /^ *[|]? *$/" <<<"$goal" | grep -q . ||
        { echo "$goal"; echo "FAIL: the e07b goal table"; exit 1; }
      cut -d"|" -f2,5,6 <<<"$goal"
      echo "--- E07b: pod/e07.sh --dry-run with E07_PLAN=e07b MAX_USD_PER_HR=1.95 (the HF plan at 0.115 GB/s within 3.00 USD)"
      E07_PLAN=e07b MAX_USD_PER_HR=1.95 DRY_ENGINE="$strata/dsv41-engine" BUILD_ROOT="$BUILD_ROOT" COUNTS="$COUNTS" \
        HEADERS="$HEADERS" VOCAB="$VOCAB" FORK_DIR="$FORK_DIR" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > /tmp/e07b-dry.txt
      grep -E "^plan \((volume|hf)\): total|^(two-turn|bench-e07b) dry:|guard (lets|stops)" /tmp/e07b-dry.txt
      python3 -c "import re, sys; t = open(sys.argv[1]).read(); m = re.search(r\"^plan \(hf\): total ([0-9.]+) min .*about ([0-9.]+) USD at 1.95 USD/h\", t, re.M)
assert m and float(m.group(2)) <= 3.0, m
order = [re.search(rf\"^  {s}\\b.*timeout min\", t, re.M) for s in (\"dry-run\", \"two-turn\", \"bench-e07b\", \"dense-free\")]
assert all(order) and [x.start() for x in order] == sorted(x.start() for x in order), \"the e07b steps out of order or without timeouts\"
print(f\"E07_PLAN=e07b with the model from HF at 0.115 GB/s: {m.group(1)} min, {m.group(2)} USD at 1.95 USD/h: within 3.00 USD\")" /tmp/e07b-dry.txt
      echo "--- E07b: pod/e07.sh --dry-run with E07_PLAN=e07b HF_GBS=0.06 (a slower download: over 3.00 USD, exit 2)"
      rc=0; E07_PLAN=e07b MAX_USD_PER_HR=1.95 HF_GBS=0.06 BUILD_ROOT="$BUILD_ROOT" COUNTS="$COUNTS" HEADERS="$HEADERS" \
        VOCAB="$VOCAB" FORK_DIR="$FORK_DIR" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > /tmp/e07b-slow.txt 2>&1 || rc=$?
      grep -E "^e07.sh: E07_PLAN=e07b" /tmp/e07b-slow.txt; echo "(exit $rc)"
      [[ $rc -eq 2 ]] || { tail -n 5 /tmp/e07b-slow.txt; echo "FAIL: e07.sh --dry-run with HF_GBS=0.06 exited $rc, not 2"; exit 1; }
      echo "--- pod/devgpu.sh --dry-run (the E02, E03, E04a, E04b, E05a, E05b and E05c dev pod session)"
      BUILD_ROOT="$BUILD_ROOT" COUNTS="$COUNTS" bash "$STRATA_DIR/tools/dsv41/pod/devgpu.sh" --dry-run
            echo "--- pod/t04a.sh --dry-run --prefill CAP_MIN=63 (the prompts above)"
      PREFILL_PROMPTS=/tmp/prefill-prompts CAP_MIN=63 bash "$STRATA_DIR/tools/dsv41/pod/t04a.sh" --dry-run --prefill > /tmp/t04a-dry.txt
      echo "$(wc -l < /tmp/t04a-dry.txt) lines, exit 0; the steps: $(sed -n "s/^+ \[\([^]]*\)\].*/\1/p" /tmp/t04a-dry.txt | tr "\n" " ")"
      grep -E "^plan: |\] plan: |prefill VRAM|^ok [0-9]+k:" /tmp/t04a-dry.txt
    '

  # allowed as not found: the driver library only (both flavors bundle their CUDA runtime)
  allowed='libcuda\.so\.1'

  # no LD_LIBRARY_PATH: the binaries find bin/ through RUNPATH $ORIGIN, and the toolkit's libcuda stub is
  # installed as the system libcuda.so.1, where a pod's driver puts the real one
  echo "=== $FLAVOR_TAG: tarballs in a bare ubuntu:22.04, no LD_LIBRARY_PATH (allowed not found: $allowed)"
  docker run --rm -e PATTERN="*-$FLAVOR_TAG-$ARCH_TAG.tar.gz" -e ALLOWED="$allowed" \
    -v "$BUILD_ROOT/dist${BUILD_TAG:-}:/dist:ro" -v "$flavor_root/cudastub:/cudastub:ro" "$BARE_IMAGE" bash -euo pipefail -c '
    cp /cudastub/libcuda.so.1 /usr/local/lib/ && ldconfig
    unset LD_LIBRARY_PATH
    cd /tmp
    shopt -s nullglob
    tarballs=(/dist/$PATTERN)
    (( ${#tarballs[@]} == 2 )) || { echo "FAIL: expected 2 tarballs /dist/$PATTERN, found ${#tarballs[@]}"; exit 1; }
    status=0
    for tgz in "${tarballs[@]}"; do
      echo "--- ${tgz##*/}"
      tar -xzf "$tgz"
      dir="$(basename "$tgz" .tar.gz)"
      (cd "$dir" && sed -n "/^sha256:/,\$p" MANIFEST | tail -n +2 | sha256sum -c --quiet) && echo "MANIFEST sha256: OK"
      case "$dir" in llama-dsv41-*) exes=(llama-server dsv41-moe-replay libstrata-moe.so) ;; strata-*) exes=(strata dsv41-engine dsv41_cuda_probe) ;; esac
      echo "ldd bin/{${exes[*]}}, not found:"
      missing="$(for exe in "${exes[@]}"; do ldd "$dir/bin/$exe"; done | grep "not found" | sort -u || true)"
      echo "${missing:-  (none)}"
      rest="$(grep -v -E "$ALLOWED" <<<"$missing" || true)"
      [[ -z "$rest" ]] || { echo "FAIL: unexpected missing libraries"; status=1; }
      case "$dir" in
        llama-dsv41-*) runs=("llama-server --version" "dsv41-moe-dump --help" "dsv41-moe-replay --help") ;;
        strata-*) runs=("strata --help") ;;
      esac
      for r in "${runs[@]}"; do
        read -ra cmd <<<"$r"
        rc=0; out="$("$dir/bin/${cmd[0]}" "${cmd[@]:1}" 2>&1)" || rc=$?
        echo "bin/$r: exit $rc, $(grep -m 1 -E "^version:|^usage:|^strata generate" <<<"$out" || echo "no version/usage line")"
        [[ $rc -eq 0 ]] || { echo "$out" | tail -n 5; echo "FAIL: bin/$r"; status=1; }
      done
      # the qualification uploads dsv41_cpu_experts_probe alone, before the payload: copied out of bin/, it
      # loads with no library of the tarball (no cuBLAS, no CUDA runtime) and answers bad arguments with its usage
      if [[ "$dir" == strata-* ]]; then
        mkdir -p /tmp/alone && cp "$dir/bin/dsv41_cpu_experts_probe" /tmp/alone/
        rc=0; out="$(/tmp/alone/dsv41_cpu_experts_probe 2>&1)" || rc=$?
        libs="$(ldd /tmp/alone/dsv41_cpu_experts_probe | awk "{ print \$1 }" | paste -sd" ")"
        echo "dsv41_cpu_experts_probe alone: exit $rc, $(grep -m 1 "^usage:" <<<"$out" || echo "no usage line"); ldd: $libs"
        [[ $rc -eq 2 ]] && grep -q "^usage:" <<<"$out" && ! grep -q "libcu" <<<"$libs" ||
          { echo "$out" | tail -n 5; echo "FAIL: dsv41_cpu_experts_probe does not run alone"; status=1; }
      fi
    done
    exit $status
  '
done

elapsed=$(( $(date +%s) - start ))
printf 'verify.sh: all checks passed, total wall time %dm%02ds (%d s)\n' $((elapsed / 60)) $((elapsed % 60)) "$elapsed"
