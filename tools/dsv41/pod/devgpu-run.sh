#!/usr/bin/env bash
# Runs on the dev pod (tools/dsv41/pod/devgpu.sh uploads and starts it): the engine's GPU checks and
# benches (the indexer, attention, quantized matmuls, dense layers, expert tiers, prefill, decode, prefix cache, hot
# set and prefill speed; DEVGPU_PLAN c03, c05 and c05b have their own steps, below) from
# the payload directory, each under `timeout` with the time left before DEADLINE (unix seconds), output in ./out.
# Every step writes "step NAME: exit N, S s" (or "step NAME: skipped, ..." when the time is gone) to out/steps.log;
# the last line of out/steps.log is "done", the one before it the smokes' exit code policy (a
# smoke's speed targets are PASS/FAIL lines in its log, never its exit code, as the expert speed targets always were; a smoke
# exits non-zero only when a call failed, a logit was not finite or a copy-only check failed). Exits 1 when a step
# failed or was skipped. The binaries find the
# payload's cuBLAS through their RUNPATH ($ORIGIN). The step runner is pod-steps.sh's: every 30 s a background monitor
# appends free -m, the GPU's and the cgroup's memory, the running step's processes with their RSS and dmesg's tail to
# out/monitor.log (an early A100 pod went
# silent mid-session, host OOM suspected, and nothing told what the host was doing). Once a step returns, whatever is
# left of its process group (a server's engine after a failed check) gets SIGTERM, then SIGKILL 10 s later, before the
# next step starts, each such kill in out/leftovers.log (the failed serve step's leftover
# engine was the likely contention of a later decode).
set -uo pipefail
: "${DEADLINE:?}"
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=tools/dsv41/pod/pod-steps.sh
source pod-steps.sh
steps_begin

# the GPU's published dense FP16 tensor TFLOPS (FP32 accumulate, without sparsity; NVIDIA's datasheets, the list in
# the smokes' comment below): the attention bench's roofline and the smokes' expert compute are read against it
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" in
  "NVIDIA RTX A6000") pod_tflops=154.8 ;;
  "NVIDIA A40") pod_tflops=149.7 ;;
  "NVIDIA L40") pod_tflops=181.05 ;;
  "NVIDIA A100"*) pod_tflops=312 ;;
  *) pod_tflops=0 ;;
esac
run env bash -c 'date -u; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm --format=csv; nproc; free -g; cat MANIFEST'
# devgpu.sh DEVGPU_PLAN=c03 (the payload's plan file): the batched decode's CUDA
# side only. The kernels on synthetic tensors (qmatmul's MMVQ at 2 to 8 columns against one, the hits grouped by expert
# against one token at a time, attn_decode_batch against attn_decode alone at tiny and real dims, every value bit for
# bit), the engine's batch end to end on tiny-a and tiny-b against each request alone (the per-B graphs, their update
# when the slots change), then the tests whose paths the batch's code touched: the expert tiers (their calls of 4
# tokens now take the grouped hits), qmatmul against its reference, and the alone path end to end against the fork
# (Reuse Mode, the split placement, forced)
if [[ "$(cat plan 2>/dev/null)" == c03 ]]; then
  run batch-kernels ./dsv41_engine_batch_test --kernels
  run batch-e2e ./dsv41_engine_batch_test --gpu e05a/tiny-a.gguf e05a/hot-a/split.json e05a/tiny-b.gguf e05a/hot-b/split.json
  run experts-test ./dsv41_engine_experts_test --gpu e05a/tiny-a.gguf e05a/hot-a e05a/tiny-b.gguf e05a/hot-b
  run qmatmul-test ./dsv41_engine_qmatmul_test --gpu
  for c in a b; do
    case "$c" in a) short_steps=5 ;; b) short_steps=9 ;; esac
    run "e2e-$c-reuse-split-forced" bash -c "./dsv41-engine --model e05a/tiny-$c.gguf --tokens-file e05a/prompt-$c-short.txt \
      --n-predict $short_steps --ctx 2048 --readers reuse --hot-set e05a/hot-$c/split.json --vram-gb 1 --ram-gb 1 \
      --force-selection e05a/fork/dump-$c-short --dump-layers out/e2e-$c-reuse-split-forced &&
      python3 engine_parity.py out/e2e-$c-reuse-split-forced e05a/fork/dump-$c-short"
  done
  echo done >> out/steps.log
  exit "$failed"
fi
# devgpu.sh DEVGPU_PLAN=c05b: the promotion's parity on the engine,
# batch_test --gpu and --gpu-swaps on tiny-a and tiny-b (check 12: a request stopped on the solo path after 1, 2 and 5
# tokens, promoted into slot 0 with 3 requests beside it, every call's logits bytes its alone run's; check 13: a token
# read by prefill(T=1) against decode on the same state, the logits that differ, reported), then the alone path end to
# end against the fork
if [[ "$(cat plan 2>/dev/null)" == c05b ]]; then
  run batch-e2e ./dsv41_engine_batch_test --gpu e05a/tiny-a.gguf e05a/hot-a/split.json e05a/tiny-b.gguf e05a/hot-b/split.json
  run batch-swaps ./dsv41_engine_batch_test --gpu-swaps e05a/tiny-a.gguf e05a/hot-a/split.json e05a/tiny-b.gguf \
    e05a/hot-b/split.json
  for c in a b; do
    case "$c" in a) short_steps=5 ;; b) short_steps=9 ;; esac
    run "e2e-$c-reuse-split-forced" bash -c "./dsv41-engine --model e05a/tiny-$c.gguf --tokens-file e05a/prompt-$c-short.txt \
      --n-predict $short_steps --ctx 2048 --readers reuse --hot-set e05a/hot-$c/split.json --vram-gb 1 --ram-gb 1 \
      --force-selection e05a/fork/dump-$c-short --dump-layers out/e2e-$c-reuse-split-forced &&
      python3 engine_parity.py out/e2e-$c-reuse-split-forced e05a/fork/dump-$c-short"
  done
  echo done >> out/steps.log
  exit "$failed"
fi
# devgpu.sh DEVGPU_PLAN=c05: the CUDA expert kernels against the CPU experts in
# the GPU's numerics bit for bit on synthetic tensors (every matvec at 1 to 8 activations, the FFN through
# qmatmul_experts at 1 to 8 tokens, the weighted sum with some rows from the CPU), the CPU side's own checks against the
# line-for-line reference and its speed table on this pod's CPU, the expert tiers' decode bit for bit in all five
# placements, the engine's batch against each request alone with the hot set swapping at every token and without,
# the batched kernels again, the tests whose numerics the deterministic CPU experts changed (the pinned MMVQ traits: qmatmul; the shared
# expert's clamp: dense), and the alone path end to end against the fork
if [[ "$(cat plan 2>/dev/null)" == c05 ]]; then
  run expert-bits-gpu ./dsv41_engine_expert_bits_test --gpu
  run expert-bits-cpu ./dsv41_engine_expert_bits_test
  run experts-test ./dsv41_engine_experts_test --gpu e05a/tiny-a.gguf e05a/hot-a e05a/tiny-b.gguf e05a/hot-b
  run batch-kernels ./dsv41_engine_batch_test --kernels
  run batch-e2e ./dsv41_engine_batch_test --gpu e05a/tiny-a.gguf e05a/hot-a/split.json e05a/tiny-b.gguf e05a/hot-b/split.json
  run batch-swaps ./dsv41_engine_batch_test --gpu-swaps e05a/tiny-a.gguf e05a/hot-a/split.json e05a/tiny-b.gguf \
    e05a/hot-b/split.json
  run qmatmul-test ./dsv41_engine_qmatmul_test --gpu
  run dense-test ./dsv41_engine_dense_test --gpu
  for c in a b; do
    case "$c" in a) short_steps=5 ;; b) short_steps=9 ;; esac
    run "e2e-$c-reuse-split-forced" bash -c "./dsv41-engine --model e05a/tiny-$c.gguf --tokens-file e05a/prompt-$c-short.txt \
      --n-predict $short_steps --ctx 2048 --readers reuse --hot-set e05a/hot-$c/split.json --vram-gb 1 --ram-gb 1 \
      --force-selection e05a/fork/dump-$c-short --dump-layers out/e2e-$c-reuse-split-forced &&
      python3 engine_parity.py out/e2e-$c-reuse-split-forced e05a/fork/dump-$c-short"
  done
  echo done >> out/steps.log
  exit "$failed"
fi
run test ./dsv41_engine_indexer_test --gpu
run bench ./dsv41_engine_indexer_bench
# the ratio-1 index sources (layers 20, 24, 28, 32, 36) see every position: 262,144 rows at the 262K target
run bench-ratio1 ./dsv41_engine_indexer_bench --ratio 1 --nk 262144 --nq 1,4096 --reps 5
# qmatmul against its CPU reference at every real dense shape on every path (MMVQ, dequant + cuBLAS, MMQ), then
# its bench against cuBLAS f16
run qmatmul-test ./dsv41_engine_qmatmul_test --gpu
run qmatmul-bench ./dsv41_engine_qmatmul_bench
# the attention layer against its CPU reference (tiny and real dims) on q2_K weights (qmatmul, E04a), on f16
# weights (the cuBLAS test path) and, quick (the tiny cases' prefill chunks), on the MMQ option; then its bench on q2_K
# and f16 weights (E04a acceptance: q2_K prefill per role within 1.1x of f16). Each test's "attention margin" line
# gives prefill_attn_kernel's (E09, p in f16) and attn_kernel's worst ratio against the 2^-11 tolerance, measured
run attention-test ./dsv41_engine_attention_test --gpu
run attention-test-f16 ./dsv41_engine_attention_test --gpu --f16
run attention-test-mmq ./dsv41_engine_attention_test --gpu --quick --mmq
run attention-bench ./dsv41_engine_attention_bench
run attention-bench-f16 ./dsv41_engine_attention_bench --f16
# one 16384-token chunk from position 0 per role, split by op, the real 40
# layers' mean per layer and its target line (<= 30 ms at 16K); the attend roofline
# against the peak, the select scorer's floor, and the attend (<= 12 ms) and select (<= 4 ms) lines
run attention-bench-16k ./dsv41_engine_attention_bench --chunk 16384 --prefill-ctx 16384 --decode-ctx 16384 --reps 3 \
  --peak-tflops "$pod_tflops"
# the dense ops (hyper-connections, engram, router, norms, shared expert, embedding, head) against their CPU
# reference at tiny and real dims (a 4M-row engram table in pinned RAM), quick on the MMQ option, then their bench
run dense-test ./dsv41_engine_dense_test --gpu
run dense-test-mmq ./dsv41_engine_dense_test --gpu --quick --mmq
run dense-bench ./dsv41_engine_dense_bench
# the expert tiers in three placements against their CPU and engine-numerics references
run experts-test ./dsv41_engine_experts_test --gpu e05a/tiny-a.gguf e05a/hot-a e05a/tiny-b.gguf e05a/hot-b
# the grouped prefill GEMM at the real dims against its reference, one-hot rows bit for bit against the dequant
# path's f16 weights (q2_K and q3_K), and its timing (48 experts x 256 and x 32 rows)
run experts-gemm ./dsv41_engine_experts_test --gemm
# the engine end to end. Per tiny model and expert placement: --readers dense (the fork-parity mode) on the
# 64-token prompt with the fork's 16 greedy steps, then the default Reuse Mode on the short prompt, where both modes
# coincide (SPEC.md "Reader modes"), each against the fork's dump; then the placements against each other, and the
# attention on f16 weights once. Every run twice: free (its own selections and greedy tokens; engine_parity.py ties
# each token over the bound to a router or indexer flip) and with --force-selection (the fork's router ids, index rows
# and tokens: every layer, token and call within the bound, the placements too)
placement_args() {   # placement_args MODEL PLACEMENT
  case "$2" in
    all-vram) echo "--hot-set e05a/hot-$1/all-vram.json --vram-gb 1" ;;
    all-cpu) echo "" ;;
    split) echo "--hot-set e05a/hot-$1/split.json --vram-gb 1 --ram-gb 1" ;;
  esac
}
e2e() {   # e2e NAME DUMP ENGINE_ARGS...: one run against the fork's DUMP, free and then forced (NAME-forced)
  local name="$1" dump="$2"; shift 2
  run "$name" bash -c "./dsv41-engine $* --dump-layers out/$name && python3 engine_parity.py out/$name e05a/fork/$dump"
  run "$name-forced" bash -c "./dsv41-engine $* --force-selection e05a/fork/$dump --dump-layers out/$name-forced &&
    python3 engine_parity.py out/$name-forced e05a/fork/$dump"
}
for c in a b; do
  case "$c" in a) short_steps=5 ;; b) short_steps=9 ;; esac
  for pl in all-vram all-cpu split; do
    e2e "e2e-$c-dense-$pl" "dump-$c-16" --model "e05a/tiny-$c.gguf" --tokens-file "e05a/prompt-$c-16.txt" --n-predict 16 \
      --ctx 2048 --readers dense $(placement_args "$c" "$pl")
    e2e "e2e-$c-reuse-$pl" "dump-$c-short" --model "e05a/tiny-$c.gguf" --tokens-file "e05a/prompt-$c-short.txt" \
      --n-predict "$short_steps" --ctx 2048 --readers reuse $(placement_args "$c" "$pl")
  done
  for pl in all-cpu split; do
    run "e2e-$c-placements-$pl" python3 engine_parity.py --engines "out/e2e-$c-dense-$pl" "out/e2e-$c-dense-all-vram"
    for mode in dense reuse; do
      run "e2e-$c-placements-$mode-$pl-forced" python3 engine_parity.py --engines "out/e2e-$c-$mode-$pl-forced" \
        "out/e2e-$c-$mode-all-vram-forced"
    done
  done
done
e2e e2e-a-dense-f16 dump-a-16 --model e05a/tiny-a.gguf --tokens-file e05a/prompt-a-16.txt --n-predict 16 --ctx 2048 \
  --readers dense --f16
# every run above decodes from the token graph (one CUDA graph per token, the experts' doorbell); the split
# placement again with every op launched from the host (--no-graphs: the same device work layer by layer, as the
# smoke's timed phase), forced against the fork and against the graph's run
for c in a b; do
  run "e2e-$c-dense-split-nographs-forced" bash -c "./dsv41-engine --model e05a/tiny-$c.gguf --tokens-file \
    e05a/prompt-$c-16.txt --n-predict 16 --ctx 2048 --readers dense $(placement_args "$c" split) --no-graphs \
    --force-selection e05a/fork/dump-$c-16 --dump-layers out/e2e-$c-dense-split-nographs-forced &&
    python3 engine_parity.py out/e2e-$c-dense-split-nographs-forced e05a/fork/dump-$c-16"
  run "e2e-$c-placements-nographs-forced" python3 engine_parity.py --engines "out/e2e-$c-dense-split-nographs-forced" \
    "out/e2e-$c-dense-split-forced"
done
# the per-request hot set's swaps under the fork's parity. Per tiny model the split placement forced (every call
# the fork's selections and tokens): without swaps (--hot-policy static), then with a swap whenever a layer can start
# one (adaptive, --hot-margin 0:-1: count(in) > 0 x count(out) - 1 always holds; 2 per layer per token into its 2 spare
# slots) and --hot-check (each token's swaps waited for, so every batch is published at the next token, then the
# device's residency rows, the host's slot and tier maps and every filled slot and home checked byte for byte). Its
# line must show a check after every token and at least 2 swaps per token; against the fork, and against the static run:
# the logits within 5e-2 rel L2 (hits on the GPU, misses on the CPU: not bit for bit) and the same greedy ids
hot_swap_check() {   # hot_swap_check LOG N_PREDICT
  python3 - "$1" "$2" <<'PYEOF'
import re, sys
text = open(sys.argv[1]).read()
n = int(sys.argv[2])
m = re.search(r"hot set adaptive: (\d+) swaps .* (\d+) checks;", text)
ok = bool(m) and int(m.group(1)) >= 2 * n and int(m.group(2)) == n
print(f"hot swaps: {m.group(1) if m else '?'} swaps, {m.group(2) if m else '?'} checks over {n} decode steps (at least "
      f"{2 * n} swaps, a check after every token): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF
}
same_ids() {   # same_ids DIR_A DIR_B: the two runs' greedy ids
  python3 -c "import json, sys; a, b = (json.load(open(d + '/engine-manifest.json'))['generated_token_ids'] for d in sys.argv[1:]);
print(f'greedy ids: {len(a)} and {len(b)}, equal: {a == b}'); sys.exit(0 if a == b else 1)" "$1" "$2"
}
for c in a b; do
  run "e2e-$c-hot-static-forced" bash -c "./dsv41-engine --model e05a/tiny-$c.gguf --tokens-file e05a/prompt-$c-16.txt \
    --n-predict 16 --ctx 2048 --readers dense $(placement_args "$c" split) --hot-policy static \
    --force-selection e05a/fork/dump-$c-16 --dump-layers out/e2e-$c-hot-static-forced &&
    python3 engine_parity.py out/e2e-$c-hot-static-forced e05a/fork/dump-$c-16"
  run "e2e-$c-hot-swaps-forced" bash -c "set -o pipefail; ./dsv41-engine --model e05a/tiny-$c.gguf --tokens-file e05a/prompt-$c-16.txt \
    --n-predict 16 --ctx 2048 --readers dense $(placement_args "$c" split) --hot-policy adaptive --hot-margin 0:-1 \
    --hot-swaps 2 --hot-spare 2 --hot-check --force-selection e05a/fork/dump-$c-16 --dump-layers out/e2e-$c-hot-swaps-forced \
    | tee out/e2e-$c-hot-swaps.txt && python3 engine_parity.py out/e2e-$c-hot-swaps-forced e05a/fork/dump-$c-16"
  run "e2e-$c-hot-swaps-count" bash -c "$(declare -f hot_swap_check); hot_swap_check out/e2e-$c-hot-swaps.txt 16"
  run "e2e-$c-hot-swaps-vs-static" bash -c "python3 engine_parity.py --engines out/e2e-$c-hot-swaps-forced \
    out/e2e-$c-hot-static-forced && $(declare -f same_ids); same_ids out/e2e-$c-hot-swaps-forced out/e2e-$c-hot-static-forced"
done
# the multi-chunk prefill under the fork's parity: the 64-token prompt in chunks of 24 (24, 24, 16: the
# compressor's carry and the readers' sources across chunk edges), a ring of 2 sets of 2 experts and 2 prefetched ones
# (many passes, every set and staging set reused, the copy stream's waits on every path), forced against the fork in
# each placement (all-cpu: every expert staged from the mapping; split: the slab, the pinned warm tier copied directly
# and prefetched, the cold ones staged) and across placements. --prefill-split off: chunks of 24 are short reads, whose
# experts outside the slab the prefill split would partly take to the CPU; these runs keep every one on
# the ring (the runs above, in one chunk, read through the split)
for c in a b; do
  for pl in all-vram all-cpu split; do
    run "e2e-$c-chunk24-$pl-forced" bash -c "./dsv41-engine --model e05a/tiny-$c.gguf --tokens-file e05a/prompt-$c-16.txt \
      --n-predict 16 --ctx 2048 --readers dense --chunk 24 --ring-set 2 --ring-sets 2 --prefetch-experts 2 \
      --prefill-split off $(placement_args "$c" "$pl") --force-selection e05a/fork/dump-$c-16 --dump-layers out/e2e-$c-chunk24-$pl-forced &&
      python3 engine_parity.py out/e2e-$c-chunk24-$pl-forced e05a/fork/dump-$c-16"
  done
  for pl in all-cpu split; do
    run "e2e-$c-chunk24-placements-$pl-forced" python3 engine_parity.py --engines "out/e2e-$c-chunk24-$pl-forced" \
      "out/e2e-$c-chunk24-all-vram-forced"
  done
done
# the real-dims smoke (8 experts per layer, half in VRAM): a 4096-token chunk at ctx 16384, 32 decode steps,
# per-stage times (reported, not gated)
run smoke ./dsv41_engine_smoke
# the measurement smoke: first the target's non-expert memory per chunk size (the real 40 layers at ctx 262144),
# then 8 layers x the real 384 experts (37 GiB of pinned RAM), 16.12 GiB of them in VRAM (44%), a 16384-token chunk (or
# the largest that fits), per layer the copies, compute, overlap and idle gaps, the gates' numbers, decode with its host
# syncs and pool handoffs, and the 40-layer projection for 1x RTX PRO 6000 (PCIe 5 x16, 96 GiB) and 1x L40S (PCIe 4
# x16, 48 GiB) on this pod's link, each target's hot fraction from that memory,
# at compute scale 1 and at the published dense FP16 tensor TFLOPS ratio (FP32 accumulate, without sparsity; NVIDIA's
# datasheets: RTX PRO 6000 Blackwell 503.8, L40S 362.05, L40 181.05, RTX A6000 154.8, A40 149.7, A100 312 (PCIe and
# SXM4); another pod GPU: 0, scale 1 only). E05c: the routed experts' compute against that peak (gate >= 35%), and
# the same run again with the real model's routing and hot set (--counts: a real-model run's decode profile).
# both runs' decode phases (timed with the split per layer per token, then the token graph with and without the
# pool spinning) with host syncs, doorbells and the 40-layer projection against the goal's 30 tok/s
link="$(nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits | head -n 1 | tr -d ' ' | tr ',' ':')"
run smoke-e05b ./dsv41_engine_smoke --layers 8 --experts 384 --vram-experts-gb 16.12 --chunk 16384 --memory-table \
  --pod-link "$link" --pod-tflops "$pod_tflops" --project "RTX PRO 6000:5:16:96:503.8" --project "L40S:4:16:48:362.05"
run smoke-e05c ./dsv41_engine_smoke --layers 8 --experts 384 --vram-experts-gb 16.12 --chunk 16384 --memory-table \
  --pod-link "$link" --pod-tflops "$pod_tflops" --project "RTX PRO 6000:5:16:96:503.8" --project "L40S:4:16:48:362.05" \
  --counts e05c/counts.json
# smoke-e05c's run with the prefetch of the real-model session (E07_PLAN=e09: 128 experts of the next layer
# copied during its front, so the ring's copies start earlier): its line of the e09 targets (attention per layer at
# 16K, the routed-expert compute and the plan gaps + stalls, on 40 layers) beside smoke-e05c's at the default 32
run smoke-e09 ./dsv41_engine_smoke --layers 8 --experts 384 --vram-experts-gb 16.12 --chunk 16384 --memory-table \
  --pod-link "$link" --pod-tflops "$pod_tflops" --project "RTX PRO 6000:5:16:96:503.8" --project "L40S:4:16:48:362.05" \
  --counts e05c/counts.json --prefetch-experts 128
# dsv41-engine --serve behind serve/server.py on tiny-b, the tokenizer the first 512 tokens of the real one: the
# stdin protocol with the prefix cache's paths (live resume, prompt-end restore, cold, the window's limit, ERR), then
# /v1/chat/completions non-streaming and streaming, the generated ids against the engine's own greedy run of the
# rendered prompt (--reference cli: dsv41-engine --tokens-file, the same --ctx and --chunk)
run serve-tiny-b bash -c 'python3 -c "import jinja2" 2>/dev/null || pip install -q jinja2==3.1.6;
  python3 src/tools/dsv41/engine/serve_check.py --engine ./dsv41-engine --model e05a/tiny-b.gguf \
    --tokenizer-gguf e06/tiny-vocab.gguf --reference cli --out out/serve-tiny-b'
# the prefix cache at real dims (8 layers x 384 experts, --counts routing): turn 1 = 16,384 prompt tokens + 32
# generated, turn 2 = turn 1 + 512 new: RESUME of the live state, its TTFT next to a cold read of the same tokens and
# its 40-layer projection (goal <= 1 s), turn 2's prefill split (slab, ring, CPU) and its physical floor on the RTX PRO
# 6000 (PCIe 5 x16 at this pod's copy efficiency, from the link), then forced from cold references (selections only):
# the live resume's and the prompt-end restore's logits within 5e-2 rel L2
run smoke-e06-prefix ./dsv41_engine_smoke --layers 8 --experts 384 --vram-experts-gb 16.12 --chunk 16384 \
  --counts e05c/counts.json --prefix-cache 512 --pod-link "$link" --floor-target "RTX PRO 6000:5:16"
echo "note: the smokes' targets (E05b, E05c, E05d, E08, E09) are reported as PASS/FAIL lines in their logs, not as exit codes; a smoke's non-zero exit is a failed call, a non-finite logit or a failed copy-only check" >> out/steps.log
echo done >> out/steps.log
exit "$failed"
