# Tools for the dedicated DeepSeek-V4.1-Flash engine

The engine is checked op by op against the stock fork (the patched llama.cpp fork of `third_party/llama-dsv41/`,
checked out at `$FORK_DIR`, commit `b4d9cb1`, the Strata hook off).
These tools make that check free on the dev VM (CPU only, no pods, no 246 GiB model).

| File | What |
|---|---|
| `SPEC.md` | the forward pass as the fork computes it: every op with shapes, constants, fork source lines and dump name; the KV bytes per token; the divergences from the tech report |
| `tiny_dsv41.py` | a tiny random DeepSeek-V4.1 GGUF (configs a and b, every layer role of the real model, its tensor names and types, or all f32), python3 + numpy |
| `op-dump/` | `op-dump`: greedy run of the fork through its eval callback, one file per spec op, occurrence, layer and ubatch, plus `dump-manifest.json` |
| `op_dump.py` | `check` (sizes, every op of every role, SPEC.md shapes from the GGUF), `compare` (bit-identical or worst rel L2, index sets for top-k), `map-oracle` (a real-model dsv41-moe-dump manifest against the spec ops), `roles` (readers and index sources, and where Reuse Mode and `--readers dense` coincide), `show`, and `load()` / `Model.modes_coincide()` for parity tests |
| `kv_budget.py` | KV cache bytes per part, per layer and per token from a GGUF or its shard headers, the fork's layout and a per-ratio layout, python3 only |
| `run-oracles.sh` | the fork's oracles on the dev VM: models, op-dump build, determinism runs, dumps, checks, KV bytes, oracle mapping |
| `attention_cases.py` | every attention op file of a dump (the layer inputs, projections, compressor, index keys and queries, masks, attention and output, the cache views), the input of `dsv41_engine_attention_test --cases`, python3 only |
| `dense_cases.py` | every dense op file of a dump (sections 1, 2, 3, 6, 7 shared expert and 8, the inputs `attn_out` and `ffn_moe_out`, the model level ops) and each ubatch's tokens, the input of `dsv41_engine_dense_test --cases`, python3 only |
| `indexer_cases.py` | the lightning indexer cases of a dump (every index source layer and ubatch: positions, ratio, heads, top-k and the five `idx_*` files), the input of `dsv41_engine_indexer_test --cases`, python3 only |
| `dsv41-engine.cpp` | `dsv41-engine`, the engine end to end: a deepseek41 GGUF, token ids in, greedy token ids out, the expert placement from a hot set, `--dump-layers` for the parity below, `--force-selection` (a reference's router ids, index rows and tokens), `--dry-run` without CUDA; `--serve`, resident behind Strata's stdin protocol with the prefix cache |
| `hot_sets.py` | the hot sets of the end-to-end runs' expert placements (`all-vram.json`, `split.json`; all-CPU is no hot set) from a GGUF's tensor headers, python3 only |
| `llama_unicode.py` | writes and checks `tools/llama_unicode_flags.txt`, llama.cpp's codepoint categories and whitespace from the fork's `src/unicode-data.cpp`, which `tools/strata_tokenizer.py` reads; python3 only |
| `tokenizer_parity.py`, `tokenizer_corpus.txt` | the server's tokenizer against the fork's `llama-tokenize` on a fixed 226-line corpus, per vocab-only GGUF; python3 only |
| `tiny_vocab.py` | the first N tokens of the real tokenizer (`$VOCAB`) as a vocab-only GGUF, the tokenizer of the tiny models' 512 ids; python3 only |
| `serve_check.py` | `dsv41-engine --serve` behind `serve/server.py`: the deepseek-v4.1 profile, the stdin protocol with the prefix cache's paths, the HTTP server's chat completions; `--reference dry` (dev VM) or `cli` (dev pod) |
| `engine_parity.py` | a `dsv41-engine --dump-layers` directory against a fork dump (per-layer and per-token `l_last` and per-step logits rel L2, the router and indexer flips with the reference's margins, each token over the bound tied to a flip, the greedy tokens with a near-tie check where they differ) or against another engine run (`--engines`); `--self-test`; python3 only |
| `oracle_parity.py` | a `dsv41-engine --dump-layers` run of the real model against the real-model oracle (the stock fork's `dsv41-moe-dump` run-1 on an A100): per layer cur and out rel L2, weights, expert sets, the greedy ids with the first divergence's near-tie; `forced` makes the oracle a `--force-selection` reference; `--self-test`; python3 only |
| `gguf_layout.py` | `sparse` (the real model's 7 shards from their headers as sparse files, for host dry runs), `sizes` (a copy against the measured shard sizes), `split` (a GGUF in N shards as llama-gguf-split writes them); python3 only |

## Running

```
tools/dsv41/engine/run-oracles.sh $BUILD_ROOT/engine/e01
```

Needs the cu128 builder image and fork build (`tools/dsv41/build.sh`), the cu128 fork tarball in
`$BUILD_ROOT/dist`, a python3 with numpy (`PYTHON`, default `$BUILD_ROOT/venv/bin/python`), the real model's
shard headers and the real-model oracle's manifest. The headers (about 112 MiB, the tensor infos of all 7 shards and shard 1's
metadata) come from HTTP range requests, no model download:

```
mkdir -p $BUILD_ROOT/engine/headers && cd $BUILD_ROOT/engine/headers
for i in 1 2 3 4 5 6 7; do
  n=$([ $i = 1 ] && echo 67108863 || echo 8388607)
  curl -sfL -r 0-$n -o head-$i.bin "https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/resolve/c4a085541cb53f67ee5e57b63d255e80cef286e7/DeepSeek-V4.1-Flash-Q2_K-0000$i-of-00007.gguf"
done
```

Measured on the dev VM (12 vCPU, `-t 4`, cu128 tarball libraries, no GPU): the whole script takes 17 to 23 s; the dumps
of 64 prompt + 8 generated tokens are 2115 files / 43 MiB (config a) and 3024 files / 136 MiB (config b), and two
runs are bit-identical. Numbers and the checks' output: `SPEC.md`, "Tiny models and op-dump".

## Using a dump in a parity test

```
import op_dump
q = op_dump.load(dump_dir, "q", layer=1, ubatch=0)          # [T, H, D] (ggml ne reversed)
score = op_dump.load(dump_dir, "idx_score", 1, 0)            # [T, C]
pooled = op_dump.load(dump_dir, "comp_kv", 1, 0, occ=2)      # occurrences: SPEC.md section 5
```

A test of the engine's default Reuse Mode may compare against a dump only when
`op_dump.Model(model_path).modes_coincide(n_tokens)` is true; otherwise it runs the engine with `--readers dense`
(SPEC.md, "Reader modes").

The inputs of an op are the outputs of the op before it in SPEC.md order (for example `idx_score` from `idx_q_rot`,
`idx_k` and `idx_weights`), and the weights come from the same tiny GGUF.

## The fused lightning indexer

The engine's indexer (`include/strata/dsv41/engine/indexer.hpp`, `src/dsv41/engine/indexer.cu`) computes, for every
query of a decode token or a prefill ubatch, the index scores of the visible compressed rows and their top-512, without
the fork's [n_comp, n_tokens, 32] score tensor: the scores, summed over the heads, go to a [q_chunk, n_k] f32 scratch
(`indexer_plan`: at most 256 MiB by default, 512 queries per pass at 131,072 keys), then a per-query top-k reads them.

| Piece | What | Strata origin |
|---|---|---|
| `scores_simt_kernel` | FP32 FMA, one thread per (query, key), the f16-rounded query in shared memory read as broadcasts: decode | `qsa_block_scores` (warp per block); one thread per key because at 32 heads the warp form needs 160 shuffles per key |
| `scores_tc_kernel` | mma.sync m16n8k16 f16 x f16 -> f32, 8 queries per CTA (one per warp, its 32 heads as A fragments), 32-key shared tiles, relu x weight in the epilogue and a reduce-scatter over the lanes: prefill (`Auto` from 8 queries, sm_80+) | `qsa_block_scores_tc` (3xTF32 m16n8k8): f16 because the fork's inputs are f16 and one f16 product is exact |
| `topk_kernel` | one CTA per query, 4-pass 8-bit radix select on per-warp histograms, ties to the lowest index, ids ascending, -1 past min(top_k, visible) | `qsa_block_topk` wide kernel, every row a cell of weight 1 (no block-to-cell expansion) |
| `indexer_ref.cpp` | CPU reference, the fork's CPU order bit for bit (SPEC.md "Numerics") | new: the fork's op order, which no Strata code has |

Numerics: every path rounds the query to f16 (round to nearest even) as the fork's mul_mat does against the f16 key
cache, so the products are exact and the paths differ only in summation order. Visibility is `(pos + 1) / ratio` rows
(the compressed causal mask), from a device array of positions (a captured decode graph can move it).

Tests: `dsv41_engine_indexer_test --cases` (a) runs in `verify.sh` on the tiny models' dumps; `--gpu` (b) on a dev pod:
both scorers on sampled queries against the reference with |err| <= 2e-5 of the absolute-term scale (2 x (128 + 32)
x 2^-24 = 1.9e-5 is the worst case of two summation orders), the top-k kernel on its own scores equal to the
reference rule, `indexer_select`'s ids against the reference's (a difference accepted only as a near-tie within that
tolerance), and the top-k on tied and equal scores. Cases: tiny-a/tiny-b dims (4 and 8 heads), and real dims at 1K,
16K and 131K keys with 1 and 4096 queries (the 1K prefill case is 2048 queries from position 0: 4096 tokens make at
least 2048 rows at ratio 2), plus a 4K fresh prompt (causal triangle) and a 16K case with 1000 queries per pass.

Measured on the dev VM (2026-10-07, 12 vCPU, no GPU, `verify.sh` in the cu128 and cu130 builder images):

| Check | tiny-a | tiny-b |
|---|---|---|
| cases (index source layer x ubatch) | 18 (layers 1, 3; 9 ubatches) | 27 (layers 2, 4, 5) |
| queries (decode ubatches) | 144 (16) | 216 (24) |
| hidden rows -inf exactly where `(pos + 1) / ratio` says | 32,940 | 48,744 |
| visible scores bit-identical | 3924/3924 (max abs err 0) | 6552/6552 (0) |
| top-k sets identical / ties at the threshold / other | 143 / 1 / 0 | 216 / 0 / 0 |

The device code is in `dsv41_engine_indexer_test` and `_bench` for sm_80 and sm_89 (cu128) and sm_120 (cu130)
(`cuobjdump`); ptxas: the tensor-core kernel 127 registers at 32 heads, no spills.

The dev pod session (`pod/devgpu.sh`, run after the dev VM checks pass): one RTX 4090 (or an L4, RTX A5000 or RTX 3090:
the cu128 build's sm_80/sm_89 code runs on all of them) at <= 0.70 USD/h, a hard cap of min(40 min, 1.50 USD / price),
the two binaries from the cu128 strata tarball (sha256 checked against its MANIFEST, 1.2 MB with the static CUDA
runtime), `pod/devgpu-run.sh` on the pod (test (b), the bench, and the bench at 262,144 keys and ratio 1, the ratio-1
sources at the 262K target), results in `$BUILD_ROOT/engine/e02-pod/<UTC time>/`, the pod deleted on every exit
and by a watchdog at the cap.

## The attention layer and its KV cache

The engine's attention (`include/strata/dsv41/engine/attention.hpp`, `src/dsv41/engine/attention.cu`) runs one layer
over a prefill chunk (`attn_prefill`) or a decode token (`attn_decode`) for every role (window, Full, Reindex,
Reuse), against one sequence's cache (`kv_cache.hpp`, `kv_cache.cpp`), in Reuse Mode or with `--readers dense`.

| Piece | What | Strata origin |
|---|---|---|
| projections | cuBLAS f16 x f16 -> f32 on f16 weights dequantized from the GGUF; the input of each converted to f16 once | `Gemm::f16` (src/prefill/gemm.cu), wrapped in the engine (strata_prefill would bring the shared CUDA runtime into the static-runtime binaries) |
| norms, rope | `rms_norm_kernel` (squares in double), `rope_kernel` in the fork's float order (YaRN ramp, its two FMAs) with sincosf | `rope_scaling.hpp`'s YaRN arithmetic |
| window | a ring of n_swa - 1 + max_chunk f16 cells per layer, position p at p % cells | `qsa.hpp`'s cache, contiguous (the target holds the whole cache in VRAM) |
| compressor | `pool_kernel` over the block's r positions from the carried f32 state and the chunk, then norm, rope at the block's first position, f16 rows at the source's own ratio; `carry_kernel` keeps the incomplete block | new: DeepSeek's gated pooling |
| index keys, queries | the key projection of the f16 latent, norm, rope, `fwht_kernel` (the fork's FWHT order), f16 keys; query projection, rope, FWHT, scaled head weights; then the indexer's `indexer_select` | the indexer above |
| attention | `attn_kernel`: a block per (token, 32 heads, split of the cells), 32-cell chunks of the window positions then the selected rows to shared memory, q k and p v on f16 m16n8k16 MMAs (q rounded to f16 as the fork; p split hi + lo), online softmax in base 2, the sink in the denominator; decode splits the cells and `merge_kernel` combines them | `qsa_prompt_attn.cu` (online softmax, MMA, hi/lo), `qsa_decode_attn.cu` (split-K, log-sum-exp merge); MQA: one latent K = V row read once for all heads |
| output | `rope_kernel` inverted into f16, wo_a as one strided batched GEMM over the groups, wo_b | |
| `attention_ref.cpp` | the CPU reference: the fork's numerics (bit for bit on the fork's cell layout) or the engine's (f16 weights and activations, double sums), stage by stage for the GPU test | new: the fork's op order (`fork_cpu.hpp`, SPEC.md "Numerics") |

KV bytes at the real dims, 262,144 positions, max_chunk 4096 (by the layout above; the bench prints the allocation):
window 40 x 4,223 cells x 1 KiB = 165.0 MiB, compressed rows 3 x 131,072 + 262,144 rows x 1 KiB = 640 MiB, index keys
160 MiB, top-k ids of the 8 index sources 64 MiB, pooling state 28 KiB: 1029 MiB, against the fork's 1310 MiB
(SPEC.md "KV cache"). The workspace for 4096-token chunks is about 2 GiB (q, attention output and de-rope at [4096,
32768] dominate, plus the 256 MiB indexer scratch).

Tests: `dsv41_engine_attention_test --cases` (a) runs in `verify.sh` on the tiny models' dumps (64 + 8 tokens, and the short
dumps of `run-oracles.sh`, 17 tokens of tiny-a and 33 of tiny-b, over which Reuse Mode and `--readers dense`
coincide); `--gpu` (b) on a dev pod: tiny-a and tiny-b dims over a 64-token prompt in chunks of 39 + 25 then 8 decode
steps in both reader modes, and the real dims on synthetic histories (decode at 4K, 16K, 64K and 262,144 positions,
dense decode at 16K, 4096-token prefill at offsets 0 and 16K), each stage from the GPU's own input against the
reference, tolerances in `attention_test_gpu.cu`. `dsv41_engine_attention_bench`: per role, ms of a 4096-token chunk at
16K and 262K, of a decode token at 4K, 64K and 262K, by stage.

Measured on the dev VM (2026-10-07, 12 vCPU, no GPU, `verify.sh` in the cu128 and cu130 builder images), (a), the
fork's numerics, tolerance 0:

| Dump | layers (roles) | ubatches, tokens (decode) | ops compared bit for bit | top-k sets identical / ties / other |
|---|---|---|---|---|
| tiny-a 64 + 8 | 4 (window, Full, Reuse, Full) | 9, 72 (8) | 29 ops, every value identical, in both modes | 143 / 1 / 0 |
| tiny-b 64 + 8 | 6 (window x2, Full, Reuse, Full, Reindex) | 9, 72 (8) | 29 ops, every value identical, in both modes | 216 / 0 / 0 |
| tiny-a short | as tiny-a | 6, 17 (5) | every value identical; Reuse Mode on every reader query | 34 / 0 / 0 |
| tiny-b short | as tiny-b | 10, 33 (9) | every value identical; Reuse Mode on every reader query | 99 / 0 / 0 |

On the 64 + 8 dumps Reuse Mode is compared on the reader queries where the modes coincide (17 of tiny-a's 72, 33 of
tiny-b's), every other layer in full. The engine's numerics against the same dumps, attn_out per layer and ubatch:
worst 1.9% to 2.5% rel L2 per dump with the fork's top-k (bound 5e-2), and with their own top-k 0 to 24 of the
index-source queries per dump (24 of 216 on tiny-b) select other rows than the fork (near-ties), up to 0.71 rel L2 on
those. That is the decided price: the GPU path keeps f16 activations with f32 or double sums and does not emulate the
fork's Q8_K, the fork's numerics stay in the CPU reference, and (b) asserts the engine-numerics tolerances, not
bit-exactness with the fork; the real-model parity decides whether the indexer's query needs Q8_K (SPEC.md "Engine
numerics and fork numerics"). The device code
is in `dsv41_engine_attention_test` and `_bench` for sm_80 and sm_89 (cu128) and sm_120 (cu130); ptxas (sm_80):
`attn_kernel<512>` 147 registers, `attn_kernel<256>` 72, no spills, 71,424 bytes of shared memory at head dim 512.

The dev pod session (`pod/devgpu.sh`): the indexer steps, then `dsv41_engine_attention_test --gpu`
and `dsv41_engine_attention_bench`, with the tarball's libcublas and libcublasLt in the payload; results in
`$BUILD_ROOT/engine/e03-pod/<UTC time>/`. The script now exits 1 when a step fails, is skipped or does not report.


## Matmul on the quantized weights

`include/strata/dsv41/engine/qmatmul.hpp`, `src/dsv41/engine/qmatmul.cu`: y = x W^T for a GGUF matrix kept on the
device in its own q2_K, q3_K or q6_K blocks, uploaded as stored, for every dense matrix of the model (q2_K for the
attention, compressor, indexer, HC, engram and shared-expert gate/up matrices, q3_K for the shared expert's down, q6_K
for the output head). Decode (up to 8 columns) uses ggml-cuda's integer dots. Prefill (more than 8 columns)
dequantizes the matrix to f16 and runs cuBLAS by default (`QPrefill::DequantF16`, chosen after the first dev pod's
numbers below). The fork's MMQ stays as the `QPrefill::Mmq` option:

| Piece | What | Strata origin |
|---|---|---|
| `quantize_q8_1_kernel` | decode activations to q8_1 blocks of 32 (half d = amax / 127, half sum, q = round(x / d)), f32 or f16 input with a row stride | `native_quantize_q8_1_kernel` (native_mmvq.cu), `q8_1_finite.hpp` |
| `mmvq_kernel` | decode GEMV, 1 to 8 columns: four warps, two rows per block, a weight block decoded once and applied to every column; `groups` (blockIdx.y) for attn_output_a | `native_mmvq_multi_kernel` with its Q3_K and Q6_K traits; the Q2_K trait new in their pattern from the fork's `vec_dot_q2_K_q8_1` |
| `quantize_mmq_kernel` | prefill activations to MMQ's 128-value tiles: one f32 scale per 32 (D4: q3_K, q6_K) or one f16 scale per 64 plus f16 sums of 16 raw values (D2S6: q2_K), k padded to 512 | the fork's `quantize_mmq_q8_1` (quantize.cu) without fast math |
| `dequant_f16_kernel`, `qmatmul_f16` | prefill default: the matrix dequantized to f16 (ggml's `dequantize_row_*` order, round to nearest even; a thread per 8 values, one 16-byte store) into the context's scratch, whole or in row slices of each group, then `cublasGemmEx` (`cublasGemmStridedBatchedEx` over attn_output_a's groups) on f16 activations | `src/kernels/cuda/dequant_bf16.cu` (its q3_K and q6_K arithmetic; q2_K added from ggml-quants.c), `Gemm::native` and `Gemm::f16` (`src/prefill/gemm.cu`) |
| `qmatmul_mmq.cu` | prefill option (`QPrefill::Mmq`): the fork's `mul_mat_q_case` (mmq.cuh, the q2_K, q3_K, q6_K instances), groups as MMQ channels, in `strata-dsv41-mmq` against the static CUDA runtime | `src/prefill/moe_mmq.cu` (`Context::run`, `fits`) and `ggml_cuda_host.cu` |
| `qmatmul_ref.cpp` | CPU reference: the quantizers bit for bit, the matmul as the exact value of what the kernels compute (MMQ's f16 d * scale and dmin * min of q2_K included), in double; the dequant path as f16 weights times f16 activations summed in double (`qref_matmul_f16`) | new |

Attention (`attention.cu`) takes `QWeightView`s and runs every projection through qmatmul. On MMVQ and MMQ, x is
quantized once for wq_a, wkv, the compressor and the indexer weights, qr once for wq_b and the indexer query, and the
f16 de-rope for attn_output_a's eight groups. On the dequant path the same inputs go in as their f16 copies, which the
f16 test path already keeps. A view of type f16 takes the attention's original cuBLAS path, kept for the tests (`--f16`). `AttnEngine::init`
takes the prefill choice, and the qmatmul scratch is the dims' largest matrix (`attn_max_matrix_elems`: 80 MiB at the
real dims, attn_q_b and attn_output_b). The engine's weight checks are one host function, `attn_check_weights`.

Tests: `dsv41_engine_qmatmul_test --cpu` (a) runs in `verify.sh`. `--gpu` (b) runs on a dev pod: MMVQ, the dequant
path (its f16 weights bit for bit, then the matmul with a whole-matrix scratch and a sliced one) and MMQ; tolerances
and cases are in `qmatmul_test.cpp` and `qmatmul_test_gpu.cu`. `dsv41_engine_qmatmul_bench` reports the decode GEMV
(MMVQ against the dequant path, with the crossover T) and the prefill GEMM (dequant path and MMQ), both against
cuBLAS f16 on a ready f16 copy. `dsv41_engine_attention_test --gpu` and `dsv41_engine_attention_bench` run on q2_K
weights, `--f16` on the f16 path, `--mmq` on the MMQ option. `dsv41_engine_attention_test --gpu --dry-run` (plain,
`--f16`, `--mmq`, in `verify.sh`, no CUDA call) builds every case of (b) on the host and runs each call through the
engine's weight checks and the reference numerics it would be compared in. The first dev pod died there on an empty
view of default type 0 (F32) that the test handed to `qact_layout`.

Measured on the dev VM (2026-10-07, 12 vCPU, no GPU, `verify.sh`), (a):

| Check | Result |
|---|---|
| hand-quantized blocks (every weight 1.5, -0.5 of the min term alone, 0.5, -0.5), every layout and the fork's Q8_K dot | the literal value, bit for bit |
| random blocks with power-of-two d against integer activations, 16 x 512, 3 rows | 144 of 144 outputs exact per type (both layouts and the fork) |
| dequant of every row of the tiny GGUFs' 2D matrices against ggml's to_float | 18,796 rows (tiny-a, 59 matrices) and 59,100 (tiny-b, 81), bit for bit |
| against the fork's CPU (Q8_K, ggml's vec_dot), N(0, 1) activations, worst rel L2 per matrix (bound 2^-5) | q2_K q8_1 1.06% / 0.99%, D2S6 1.39% / 1.05%; q3_K 0.88% / 0.85%; q6_K 0.87% / 0.86% (tiny-a / tiny-b) |
| the same, per output, against the rounding bound sum \|w\| \|x^ - x^fork\| + 2^-16 sum \|w x^fork\| (q8_1, D4) | worst 0.45 of the bound |
| attention (a), engine numerics with q2_K weights on qmatmul's default paths (MMVQ up to 8 tokens, the dequant path above), with the fork's top-k (bound 5e-2) | worst rel L2 2.85% tiny-a, 2.12% tiny-b, 2.40% / 2.44% short (f16 test path: 2.54%, 1.91%, 2.08%, 2.22%; with MMQ prefill, before the dequant path: 3.12%, 2.93%, 2.59%, 2.83%) |
| the same with its own top-k: index-source queries selecting other rows than the fork | 10 of 144, 24 of 216, 0 of 34, 1 of 99 (f16: 11, 24, 0, 2; MMQ prefill: 12, 31, 0, 2) |
| attention (b) `--gpu --dry-run`, `--f16`, `--mmq`: every case's weight checks and reference numerics on the host | pass in 2-6 s each; per mode 1911 matrix runs (1501 MMVQ and 410 dequant, or 410 MMQ, or 1911 f16); with the absent-view guard removed, `--mmq` fails with the pod's message, `no activation layout for weight type 0` |

The int8 activations move attn_out further from the fork than f16 does (two different int8 roundings, per 32 or 64
values here and per 256 in the fork, against one). With the dequant path only decode and calls of up to 8 tokens
take them, which leaves the q2_K path between the f16 one and the MMQ one: within the attention test's bounds, and the
real-model parity decides (SPEC.md "Engine numerics
and fork numerics"). VRAM: q2_K is 84 bytes per 256 values, 0.164 of f16, so one layer's attention weights take about
a sixth of the f16 copy (`dsv41_engine_attention_bench` prints it). The dequant path adds one scratch per engine
(80 MiB at the real dims) and a 32 MiB cuBLAS workspace, and saves the int8 prefill buffers MMQ needs (q8_1 for 8
columns only). The device code is in the four engine binaries
for sm_80 and sm_89 (cu128) and sm_120 (cu130); the MMVQ kernels use up to 128 registers (the launch bound), no spills.

First dev pod run (2026-10-07, RTX A5000 sm_86, 139 s, 0.010 USD,
`$BUILD_ROOT/engine/e04a-pod/20261007T142350Z/`, MMQ still the prefill path then): the qmatmul test (b) and both
benches passed. The attention test (b) exited at once on the empty view above. Measured by `dsv41_engine_qmatmul_bench`
against cuBLAS f16 on a ready f16 copy:

| Shape | MMVQ + quantize, T = 1 / 4 / 8 (us) | cuBLAS f16, T = 1 / 4 / 8 (us) | MMQ + quantize, T = 4096 (ms, TFLOP/s) | cuBLAS f16, T = 4096 (ms, TFLOP/s) |
|---|---|---|---|---|
| attn_q_b (32768 x 1280) | 60.1 / 155.0 / 243.3 | 127.4 / 132.2 / 133.7 | 10.703, 32.1 | 3.981, 86.3 |
| attn_output_b (5120 x 8192) | 42.9 / 90.8 / 169.2 | 137.2 / 147.1 / 151.0 | 10.546, 32.6 | 3.629, 94.7 |
| attn_output_a (8 groups) | 32.1 / 78.4 / 133.2 | 102.0 / 105.2 / 106.2 | 9.458, 29.1 | 3.176, 86.5 |
| output (q6_K, 129280 x 5120) | 862.0 / 1769.0 / 2964.8 | 1867.8 / 1889.5 / 1899.3 | 108.346, 50.0 | 76.895, 70.5 |

A 4096-token chunk at context 16K took 360.8 ms for the six roles on q2_K weights, against 255.0 ms on f16
(`dsv41_engine_attention_bench`): projections 14.7-16.8 ms against 7.2-8.6 ms per layer, output 19.9-20.2 ms against
8.4-8.5. Hence the dequant path for prefill. It pays one dequantization of the matrix per call (writing its f16 bytes,
80 MiB for attn_q_b) on top of cuBLAS. At T <= 8 that write costs more than MMVQ's margin over cuBLAS for the large
matrices, so MMVQ keeps T <= 8 (its own limit). The bench's dequant + cuBLAS column (next pod) measures the crossover,
and the target is q2_K prefill per role within 1.1x of `--f16` in `dsv41_engine_attention_bench`.

The dev pod session (`pod/devgpu.sh`): before any create it reads the GPU's list price
(`runpod_api.py price`, REST v2's catalog) and skips a GPU above `MAX_USD_PER_HR` unasked; the indexer and attention steps plus
`dsv41_engine_qmatmul_test --gpu`, `dsv41_engine_qmatmul_bench`, the attention test on q2_K, with `--f16` and with
`--quick --mmq`, and the attention bench on q2_K and with `--f16`; results in
`$BUILD_ROOT/engine/e04a-pod/<UTC time>/`.

## Hyper-connections, engram, router, norms, shared expert, embedding and head

`include/strata/dsv41/engine/dense.hpp` (GPU, `DenseEngine`), `engram.hpp` (the host half: the n-gram row hash, the
tables in pinned RAM and their gather) and `dense_ref.hpp` (CPU reference): every op of SPEC.md sections 1, 2, 3, 6,
7 (shared expert) and 8, for a prefill chunk or a decode token, plain device pointers on one stream, so the layer loop only
assembles layers and plugs in the routed experts. Matrices go through qmatmul (hc_fn, engram_wkv and the shared
gate/up in q2_K, the shared down in q3_K, the output head in q6_K); the router's bf16 ffn_gate_inp has its own GEMV
and cuBLAS bf16 path.

| Piece | What | Strata origin |
|---|---|---|
| `dense.cu` | `DenseEngine` (qmatmul's context with a dequant scratch of one shared-expert matrix, 22.5 MiB at the real dims; engram_wkv and the output head go in row slices of it), the weighted rms norm (squares in double), `dense_embed` (q2_K rows of token_embd into the four copies, the one-hot pre_init), `dense_head` (the unfused collapse, the norm, the output) | `native_gr_norm.cu` weighted_rms_norm; the NativeEmbed / NativeHead pattern |
| `hc.cu` | `hc_mixes` (rms over HC * E, written f32 for MMVQ or f16 for the prefill paths, hc_fn, then pre, post and the 20-round Sinkhorn per token), `hc_collapse_norm` (the DSV4_HC_PRE FMA chain and the norm), `hc_expand` (DSV4_HC_POST, in place allowed) | `native_gr_postops.cu` pre_gated / post (shapes), the fork's ggml-cuda `dsv4-hc.cu` comb kernel; the fork's math in place of the gated residual's |
| `engram.cu`, `engram_hash.cpp` | the hash (`engram_rows`), `EngramTable` (pinned host RAM), `EngramGather` (host threads into a pinned staging buffer, one copy to the device), `engram_forward` (dequant, wkv, the per-copy rms gates, the signed-sqrt sigmoid, x += value * gate, in sub-chunks of `engram_chunk` tokens) | `PleTable::gather_batch` (src/kernels/ngram.cpp) with the decode on the GPU; the PLE gate's signed sqrt (`native_ple_postops.cu`); the hash is the fork's (Strata's `ngram_mixed` mixes differently) |
| `router.cu` | input to bf16, logits by a warp-per-row GEMV (up to 8 tokens) or `cublasGemmEx` bf16, then a warp per token: sqrt(softplus), the biased top-6 of 384 (ties to the lower expert), the unbiased weights, their sum in double, the clamp, the scale 1.5 | `native_router.cu` (the warp top-k), its softmax replaced |
| `dense_ffn.cu` | the shared expert: up and gate on qmatmul, the clamped SwiGLU, down | `shared_expert.cu` without its sigmoid gate |
| `dense_ref.cpp` | the reference: the fork's op by op order with its FMA contractions; matrices in fork numerics (Q8_K into the K-quant dots, bf16 into tinyBLAS or `ggml_vec_dot_bf16`, `fork_cpu.hpp`) or the engine's (qmatmul's paths, the bf16 router with exact products) | new; the q3_K / q6_K dots and the bf16 dot added to `fork_cpu.hpp`, `ref_matmul` extended to them |

The fork's numerics that (a) pins down, op by op on the dumps (SPEC.md "Numerics on the fork's CPU path"): the fused
hc ops are contracted by GCC (DSV4_HC_PRE `sum = fma(x, w, sum)` from `fma(x0, w0, 0)`, DSV4_HC_POST
`fma(x_s, comb, y post)`, DSV4_HC_COMB `fma(m, scale, base)` and `fma(e, 1 / sum, eps)`; with a plain product and sum
for the collapse, the expansion and comb's affine, 21% to 56% of those values differ on tiny-a, by up to 3.8e-6);
the head's collapse is the unfused form (each
product rounded, then added); the q3_K and q6_K dots accumulate 8 int32 lanes of 4 neighbouring bytes per 32-value
chunk (q6_K subtracts its 32 offset per lane from the bsums), one FMA per block and lane; the router's argsort is
`std::sort` descending, and the dumps' every position matches it; the token map, multipliers, primes and offsets are
read in full (the Strata GGUF reader kept 64 items of an array; `meta_int_array` now reads all of them from the mapped
file).

Tests: `dsv41_engine_dense_test --cases` (a), `--gpu` (b), `--gpu --dry-run` (the host side of (b): every call's
weight checks, the engram rows of its tokens against the table, the reference numerics of its path), and
`dsv41_engine_dense_bench`; tolerances and cases are in `dense_test.cpp`, `dense_test_gpu.cu` and `dense_bench.cu`.
The GPU cases use a synthetic model (`dense_synth.cuh`) and an engram table of the real row width (84 bytes) with
`--table-rows` rows (default 4M, 336 MiB pinned); (b) runs every op of a layer in order at tiny-a, tiny-b and real
dims, T = 64, 4, 1 (tiny) and 1, 4, 4096 (real), each op checked from the GPU's own inputs.

Measured on the dev VM (2026-10-07, 12 vCPU, no GPU, `verify.sh`, both flavors), (a):

| Check | tiny-a | tiny-b | tiny-a short | tiny-b short |
|---|---|---|---|---|
| fork numerics, every op of sections 1, 2, 3, 6, 7 (shared), 8 from the dump's inputs (35 ops, prompt and decode ubatches) | 1,948,320 values bit-identical | 4,951,872 | 463,988 | 2,278,632 |
| router top-k sets and the full argsort | 288 sets, identical | 432 | 68 | 198 |
| engine numerics, worst rel L2 per ubatch (bound 5e-2): hc_mixes / engram_kv / shared up, gate, down / output | 1.33% / 0.62% / 1.00%, 1.05%, 2.83% / 0.85% | 1.23% / 0.68% / 0.92%, 0.87%, 2.01% / 0.93% | 1.09% / 0.62% / 1.04%, 0.99%, 1.99% / 0.96% | 1.35% / 0.63% / 0.94%, 0.94%, 2.48% / 0.92% |
| engine numerics: router logits (bf16, exact products) and top-k sets differing from the fork's | 9.2e-8, 0 of 288 | 9.4e-8, 0 of 432 | 1.2e-7, 0 of 68 | 1.0e-7, 0 of 198 |
| (b) `--gpu --dry-run`, plain and `--mmq` | pass in 1.8 s; 72 matrix runs (45 MMVQ, 18 dequant or MMQ, 6 bf16 GEMV, 3 bf16 cuBLAS) | | | |

The engine's numerics differ from the fork's only in the matrices (int8 per 32 or f16 activations where the fork
rounds to Q8_K, as in the attention and the quantized matmul) and in the router's order among exactly tied experts; every other op is the same
f32 arithmetic, which the GPU kernels write with explicit roundings (`__fmul_rn`, `__fadd_rn`, `__fmaf_rn`) so nvcc's
contraction cannot reorder it. On MMQ (the option) the hc and engram inputs reach the int8 quantizer through their
f16 copies, one rounding more than on MMVQ.

Dev pod (`pod/devgpu.sh`: the indexer, attention and quantized matmul steps plus `dsv41_engine_dense_test --gpu`, `--gpu
--quick --mmq` and `dsv41_engine_dense_bench`; results in `$BUILD_ROOT/engine/e04b-pod/<UTC time>/`): not run
yet. The bench reports ms per op for a 4096-token chunk and a decode token, the engram gather (host copy GB/s from
pinned RAM and the copy to the device), the head, the router with its GEMV and with cuBLAS bf16 at 1 to 8 tokens, and
the decode sum of the non-attention, non-routed-expert ops over 40 layers plus embedding and head.

## The model loader, the routed expert tier and the layer loop

`include/strata/dsv41/engine/model.hpp` (the GGUF loader and the device model), `experts.hpp` (the routed experts in
VRAM, warm and cold tiers) and `forward.hpp` (the layer loop) put the pieces above together: `dsv41-engine` loads a deepseek41
GGUF, keeps the dense part in VRAM, places the routed experts by a hot set and runs a prompt of token ids through every
layer in chunks, then greedy decode, on one stream with every buffer allocated at init (the KV cache from `--ctx`).
Correctness first: copy/compute overlap, decode CUDA graphs and the speed numbers come in the next sections; the
tokenizer is the server's.

| Piece | What | Strata origin |
|---|---|---|
| `model.cpp` | `load_gguf_model` (the deepseek41 guard, AttnDims, DenseDims, the roles from `attn_layers`, the engram arrays in full through `meta_int_array`, every tensor of every shard), `Model` (every dense weight in one device arena, 256-byte aligned; the attention matrices as stored or, with `--f16`, dequantized to f16; engram q and k to f32; the engram tables in pinned RAM; the routed experts left on the host), `Model::check` (every view through the ops' host checks) | `gguf_reader.hpp` (GgufModel, mmap, no ggml) and its `check_architecture` pattern; the tests' `load_model` |
| `experts.cpp` | `ExpertEngine`: the tiers from a hot set, the VRAM slab, the warm tier, the decode split (hits gathered by expert on qmatmul, misses on the CPU at the same time, their rows copied up, one combine) and the prefill passes (every pair on the GPU, the experts outside the slab copied into a ring of device slots) | strata-moe behind plain pointers: `moe_hot_set.cpp` (the loader, moved out of `moe_gpu.cpp`), `moe_residency.cpp` (the warm tier, bound through the new `ExpertTensors`; the Call-side methods moved to `moe_residency_call.cpp`), `moe_prefill.cpp` (`plan_prefill`, `run_prefill` over a one-stream `PrefillEngine`), `moe_counts.cpp` (a plain-pointer `record`) |
| `experts_cpu.cpp` | `CpuExperts`: Q8_K inputs, gate and up, the clamped SwiGLU, Q8_K h, down, on a thread pool; `ref_expert_combine` | strata-moe's CPU fast path (`moe_ref.cpp` run, `cpu_kernel_avx2.cpp` compiled into the engine, `fast_kernel_cpu_ok` moved to `cpu_kernel.cpp`) |
| `experts_kernels.cu`, `ffn_kernels.hpp` | the row gather by expert and the slot-order combine (`__fmul_rn`, `__fadd_rn`, plus the shared expert's output); the clamped SwiGLU shared with `dense_ffn.cu` | moe_gpu.cpp's combine moved to the device |
| `selection.cpp` | `SelectionDump`: a fork op dump's or an engine run's manifest (prompt, calls, fed tokens) and its `ffn_moe_topk` and `idx_top_k` per call and layer, checked, as `--force-selection` takes them | new; parses with the vendored nlohmann json as `moe_hot_set.cpp` does |
| `forward.cpp` | `Forward`: embedding, the engram before its layers, the V4.1 lag of the hyper-connection mixes, attention (`attn_prefill` per chunk, `attn_decode` per step), router, shared expert, routed experts, expansion, the head on the last token; `--dump-layers`; the forced selections; `forward_dry_run` | `src/prefill/prefill.cpp` (init, carve, run_impl) and `layer.cpp` (the decode step), with DeepSeek-V4.1's sublayers |

The routed experts' numerics: the misses are the fork's CPU experts bit for bit (int8 per 256 activations, ggml-cpu's
AVX2 dots, the scalar clamp), the hits the engine's (qmatmul: MMVQ on int8 per 32 activations up to 8 rows of an
expert, the dequant path above), and the combine adds the slots in the fork's order whichever tier made them. A
prefill chunk computes every pair on the GPU, so its result does not depend on the placement (one expert's pairs are
one qmatmul call over the same gathered rows in the slab or in the ring). The end-to-end bound is 5e-2 rel L2 per
layer's `l_last` per token and per step's logits against the fork (`engine_parity.py`; the op tests' kEngineRelL2),
and at a greedy step that differs the fork's logits must show a near-tie (gap at most 2 x 5e-2 x their rms).

### Selection flips and `--force-selection`

The router (top 2 of 8 experts on tiny-a, 6 of 32 on tiny-b) and the indexer (top 8 and 16 rows) are discrete: where
the k-th and (k+1)-th scores nearly tie, the engine's numerics (f16 or int8 per 32 activations) and the fork's (int8
per 256) can pick different sides, and one other expert or row moves that token by 0.2 to 0.7 rel L2, and through
the attention every later position of the later layers. Two modes separate that from a bug:

- **Free** (the default): the engine's own selections and greedy tokens. `--dump-layers` also writes each layer's
  `ffn_moe_topk`, `ffn_moe_probs_biased` and, on an index source, `idx_top_k` and `idx_score` in the fork's names and
  layouts (the scores recomputed from the layer's queries with `indexer_scores`). `engine_parity.py` compares every
  call's sets with the reference's and prints each flip with the reference's k-th minus (k+1)-th score; a token over
  the bound must follow a flip (at its own position in this or an earlier layer, or at an earlier position in an
  earlier layer), else it is a bug and the run fails. `engine_parity.py --self-test` checks these rules on synthetic
  dumps.
- **Forced** (`dsv41-engine --force-selection DIR`, `selection.hpp`): every call takes the router ids
  (`ffn_moe_topk`, the slot order; the weights are still this call's probs at those ids) and the index sources' rows
  (`idx_top_k`, the visible ones) of the reference in DIR, a fork op dump or another engine run's `--dump-layers`, and
  the decode steps feed the reference's tokens. Every call then computes the reference's function, so every layer,
  token and call must land within the bound, with no flip left, and so must two placements forced from the same dump.
  `--dry-run` reads and checks every selection the run will take.

The fork's dumps index both selections as the engine does (checked on the dev VM, 2026-10-07, `dump-a-16`,
`dump-b-16`, `dump-a-short`, `dump-b-short`): in all 254 `ffn_moe_topk` files the ids are the argsort's first K, in
descending `ffn_moe_probs_biased` order (the engine's rank order); in all 533 index-source rows `idx_top_k` restricted
to the visible rows (a prefix of the finite `idx_score` entries) is the top k of those scores (2 rows at an exact tie).

First dev pod (2026-10-07, L4, `e05a-pod/20261007T181553Z`): 3 of 17 end-to-end checks passed (tiny-a Reuse Mode in all
three placements). Every jump it saw sits on a router near-tie in the fork's own scores (`ffn_moe_probs_biased`,
k-th 0.90 to 1.26; layer 0 is window-only in both tiny models, so only the router can flip there):

| Jump on the pod | Runs | Fork's router margin there |
|---|---|---|
| tiny-a call 0 token 24, layer 1: 0.25 | dense, all three placements | 1.58e-3 (the smallest of layer 1's 64 tokens) |
| tiny-a call 2 (position 65), layer 2: 0.27; all-VRAM vs all-CPU 0.277 | dense all-VRAM and split | 1.68e-5 |
| tiny-a `--f16` call 5, layer 0: 0.68 | dense f16 | 1.41e-3 |
| tiny-b call 0 token 48, layer 0: 0.30 | dense, all three placements | 3.92e-3 |
| tiny-b call 0 token 21, layer 0: 0.23 | Reuse all-VRAM | 1.08e-4 (the smallest of layer 0's 24 tokens) |
| tiny-b call 9, layer 0: 0.31 | Reuse all-CPU and split | 4.88e-4 |
| tiny-b call 11, layer 1: 0.11 | dense all-VRAM and split | 2.54e-3 |

The placements part at the 1.7e-5 tie because the hit path (int8 per 32) and the miss path (the fork's int8 per 256)
differ by up to 2.2% on the same pairs (the experts test's all-VRAM vs all-CPU decode, bound 3e-2): all-CPU keeps the
fork's expert there, all-VRAM and split take the other. That pod's dumps had no selections, so whether the engine
flipped at each of these tokens is what the forced and free runs of the next pod show.

Tests: `dsv41_engine_experts_test --cases` (a, GPU free: the CPU experts against the dumps' `ffn_moe_out`), `--gpu
GGUF HOT_DIR ...` (b: three placements, decode calls of 1 and 4 tokens and a 64-token prefill per layer, against the
CPU and the engine-numerics references; `--gpu --dry-run` its host side); `dsv41-engine --dry-run` (verify.sh runs it
with the arguments of every pod run); `dsv41_engine_smoke` (a bench: the real shapes on a synthetic model with 8
experts per layer, half in VRAM, a 4096-token chunk at ctx 16384 and 32 decode steps, per-stage times);
`dsv41_cuda_probe` (devgpu.sh's first step on a new pod). The fork's 16-token dumps for the parity
(`dump-a-16`, `dump-b-16`) come from `run-oracles.sh`, beside the short dumps over which both reader modes coincide.

Measured on the dev VM (2026-10-07, 12 vCPU, no GPU, `verify.sh`, both flavors, 6m42s):

| Check | tiny-a | tiny-b | tiny-a short | tiny-b short | tiny-a 16 | tiny-b 16 |
|---|---|---|---|---|---|---|
| (a) `ffn_moe_out`, every layer and ubatch, 1 and 3 threads | 147,456 values bit-identical | 442,368 | 34,816 | 202,752 | 163,840 | 491,520 |
| (b) `--dry-run`: engine numerics against the fork's CPU experts, worst rel L2 over every call (bound 3e-2) | 2.24% | 2.19% | | | | |

`dsv41-engine --dry-run` plans every pod run (tiny-a and tiny-b, the three placements, both reader modes, `--f16`):
the prefill chunk of 64 tokens is 4 slab passes on tiny-a all-VRAM, 4 ring passes of 32 experts all-CPU and 8 passes
(256 of 512 pairs on the slab, 16 experts through the ring) split; a decode step of tiny-b split computes 18 pairs on
the GPU and 18 on the CPU. A GGUF of another architecture (OLMoE) is refused: "architecture is 'olmoe', this engine
requires 'deepseek41'".

Dev pod (`pod/devgpu.sh`, which starts with `dsv41_cuda_probe` and leaves a host whose CUDA fails
it, or that gives no SSH login within `SSH_WAIT_S`, for the next offer; results in
`$BUILD_ROOT/engine/e05a-pod/<UTC time>/`; the first run is above). After the steps above it runs
`dsv41_engine_experts_test --gpu`, then per tiny model and placement `dsv41-engine --readers dense` on the 64-token
prompt with 16 greedy steps against `dump-<c>-16` and the default Reuse Mode on the short prompt against
`dump-<c>-short`, `--f16` once, each free and with `--force-selection` on the same dump (`<name>-forced`), the
placements against each other (free dense, and forced in both modes), and last `dsv41_engine_smoke`. The payload
carries the dumps' selections and scores besides `l_last` and the logits.

## Prefill copies overlapped, chunks to 16384, decode with one wait per layer

On the target (one 96 GB GPU, PCIe 5 x16) about 103 GiB of routed experts cross PCIe for every chunk that touches
every expert, so a 16K prompt goes in one chunk and the copies run beside the compute. Correctness stays the forced
parity of the previous section, now also with the prompt in several chunks.

| Piece | What | Strata origin |
|---|---|---|
| `experts.cpp` prefill | the ring's copies on a copy stream of their own; per ring set a `copied` event (the pass waits for it) and a `consumed` one (the set's next copy waits for the last pass that read it, in this call or the one before); a source in pinned host memory (the pinned warm tier, a mapping the caller pinned) copied in place, any other staged first into pinned staging (`stage_layout`, 1 MiB pieces over the CPU pool) whose set is refilled after a `stage_free` event; the slab's passes first, so the ring fills while they run; ring 3 sets of 16, staging 2 sets | `moe_gpu.cpp` `GpuTier::Prefill` and `alloc_prefill` (the events, the staging, the defaults of `STRATA_MOE_PREFILL_SETS`, `_GROUP`, `_STAGE`), `moe_prefill.cpp` `plan_prefill` / `run_prefill` unchanged |
| `experts.cpp` prefetch | once a layer's passes are queued, the next layer's first `--prefetch-experts` (32) non-resident experts with a pinned source (the hot set's warm order, then the cold ones by id) into a prefetch region on the copy stream, while that layer's attention and dense ops run; its call gives them slots after the slab's, so the slab's passes read them in place; only for chunks with T x 6 >= 4 x n_expert | new |
| `experts.cpp` decode | the one host wait is the routing (and the input rows) into pinned buffers; nothing waits after the combine; the CPU jobs are reused (no allocation per call) | the first loop's split |
| `experts_cpu.cpp` | the pool's job behind a function pointer (no `std::function` allocation per phase), its handoffs measured (wake: post to the last worker's start; return: the last worker's end to the caller), `--pool-spin-us`: workers and caller poll before they block | the first loop's pool |
| `forward.cpp` | one `QmmContext` for the stream (attention, dense ops and experts share its 80 MiB dequant scratch); the stage times from event pairs read once per call (the first loop waited after every stage); decode's dense segments as CUDA graphs per layer (the attention's input: mixes, collapse, attn_norm; the FFN's front: the attention's expansion, mixes, collapse, ffn_norm, router, shared expert; the head), captured on the first untimed decode step and replayed; the host waits it makes counted (`ForwardTimes::host_syncs`) | `include/strata/core/session.hpp`'s per-layer graphs and its fixed-address rule (the collapse weights' two buffers reset at each call so every graph's pointers hold) |
| `selection.cpp`, `engine_parity.py` | a forced chunk may be part of a reference ubatch (`call_of`); the parity matches the two runs' calls by position, the logits where both runs end a call | the first loop's |
| `engine_smoke.cpp` | `--layers`, `--experts`, `--vram-experts-gb`, the experts made in pinned RAM, the pinned H2D bench, `--memory-table` (the target's memory: the real 40 layers at ctx 262,144), the chunk fallback from `--chunk` down to 2048, per layer its front and its expert phase, the gates' numbers, three decode phases, the 40-layer projection (`--pod-link`, `--pod-tflops`, `--project`), `--dry-run` | the first smoke |
| `pod/devgpu.sh` | the results as two archives made on the pod (logs first, dumps only up to `PULL_CAP_MB`), each copy under `timeout` within `PULL_S`, the pod deleted right after the pull, the pull's time and sizes logged | an earlier pull with `scp -r` hung for 30 min |

The expert phase of a prefill layer, per layer from CUDA events (`ExpertLayerStats`): `copy_ms` (the ring's copies
and the layer's prefetch on the copy stream), `compute_ms` (the passes), `overlap_ms` (the copies' time while a pass
ran, plus the prefetch's time before the routing's fetch, when the stream runs the layer's front), the plan gap (the
stream idle from the routing's fetch to the first pass: the host plans) and the stalls (idle between the first pass's
start and the last's end: waits for copies). The gates: copy GB/s = bytes / `copy_ms` at least 80% of the pod's
pinned H2D (256 MiB copies), overlap = `overlap_ms` / `copy_ms` at least 70%.

The memory table (`--memory-table`, before the measured run): per chunk of
2048 up to `--chunk`, the device memory a target needs besides its routed experts, measured with `cudaMemGetInfo` on
a synthetic model of the real 40 layers at ctx 262,144: the dense model (measured with max(6, `--prefetch-experts`)
experts per layer, so the prefetch region is the target's, plus the router rows of the other experts up to 384, 40 x
10,244 B each: 138 MiB at 32), the KV cache and the indexer's rows at 262K, the attention, dense and forward
workspaces, the experts' call buffers, the ring and the prefetch region. Its model (15 GiB of pinned experts at 32 per
layer) is freed before the 8-layer run pins its own.

The projection to 40 layers (`dsv41_engine_smoke --project NAME:GEN:WIDTH:VRAM_GIB:TFLOPS`, the formula in the
source's header): the target's copy bandwidth is its raw PCIe link times this run's ring GB/s over the pod's raw link
(the copy efficiency carried over); its hot fraction is (VRAM_GIB - the memory table's GiB at the chunk - 2 GiB
margin) / 183.25 GiB (all 40 x 384 experts), within 0..1, printed for every chunk the table measured with the copy
bound of a 16K prompt at it (16384 / chunk x 40 layers of copies), so a chunk that buys a larger hot set shows; the
copies per layer and chunk are the non-resident (1 - HOT) x 384 x 12,810,240 B, the front (attention and dense ops)
per attention role and the expert compute are this pod's divided by the compute scale, the prefetch hides up to
`--prefetch-experts` experts behind the front, and a layer costs front + max(compute, copies - prefetch). Two compute
scales: 1 (this pod's compute: pessimistic) and the target's published dense FP16 tensor TFLOPS over the pod's
(`--pod-tflops`; FP32 accumulate, without sparsity, NVIDIA's datasheets and the RTX Blackwell architecture
whitepaper):

| GPU | Dense FP16 tensor TFLOPS | Role |
|---|---|---|
| RTX PRO 6000 Blackwell | 503.8 | target, PCIe 5 x16, 96 GiB |
| L40S | 362.05 | target, PCIe 4 x16, 48 GiB |
| L40 | 181.05 | dev pod |
| RTX A6000 | 154.8 | dev pod |
| A40 | 149.7 | dev pod |
| A100 80GB (PCIe, SXM4) | 312 | dev pod (no 48 GB GPU in stock) |

`pod/devgpu-run.sh` passes the pod's number by its `nvidia-smi` name (0 for another GPU: scale 1 only). The ratio is
a proxy: the engine's matmuls are q2_K and q3_K dequantization and MMQ, not FP16 tensor GEMMs only, so the two
scales bracket the target's compute rather than predict it.

Decode: one host wait per layer (the experts' routing) and one per token (the logits): 41 per token at 40 layers,
1.025 per layer, printed by the smoke. The engram gathers wait only when their staging's last copy is not done (never
in the layer loop: a routing wait came between). The attention's decode launches stay direct.

Pod runs (`pod/devgpu-run.sh`, started by `pod/devgpu.sh`): every run of the previous section as before (the prompt in one chunk
of 2048, `--chunk` capped by `--ctx`), plus per tiny model and placement the 64-token prompt in chunks of 24 (24, 24,
16: the compressor's carry and the readers' sources across chunk edges) with a ring of 2 sets of 2 and 2 prefetched
experts, forced against the fork's one ubatch and across placements (`e2e-<c>-chunk24-<placement>-forced`), then
`dsv41_engine_smoke` as before and the measurement smoke `smoke-e05b`: `--layers 8 --experts 384 --vram-experts-gb
16.12 --chunk 16384 --memory-table --pod-link <nvidia-smi> --pod-tflops <by name> --project "RTX PRO
6000:5:16:96:503.8" --project "L40S:4:16:48:362.05"`. That smoke pins 36.65 GiB of experts (8 x 384 x 12,810,240 B);
its slab is 168 experts per layer (43.8%, 16.03 GiB), and the chunk falls back from 16384 to the largest that fits
beside it. Its session runs on a 48 GB GPU with at least 64 GB of host RAM
(`MIN_RAM_GB` goes to the create as RunPod's `minRAMPerGPU`, a pod reporting less is deleted at once):

    export RUNPOD_API_KEY=...
    GPU_TYPES='NVIDIA RTX A6000;NVIDIA A40;NVIDIA L40' MIN_RAM_GB=64 bash tools/dsv41/pod/devgpu.sh

Measured on the dev VM (2026-10-07, 12 vCPU, no GPU, `verify.sh`, both flavors, 6m49s): every check
passes. `dsv41-engine --dry-run` of the chunked forced runs plans 3 chunks and 19 calls read from the fork's 17 (76 and
114 router, 38 and 57 index selections on tiny-a and tiny-b). Their expert plans per 24-token chunk over every layer:

| Placement | tiny-a | tiny-b |
|---|---|---|
| all-VRAM | 4 slab passes, 192 of 192 pairs | 6 slab passes, 864 of 864 pairs |
| all-CPU | 16 ring passes, 32 experts through the ring, all staged (mapped GGUF), prefetch on | 96 ring passes, 192 experts, all staged, prefetch on |
| split | 12 passes, 96 slab pairs, 16 ring experts (the cold ones staged, the warm ones copied in place and prefetched) | 54 passes, 432 slab pairs, 96 ring experts |

`engine_parity.py --self-test` passes its 11 cases (4 new: a prompt in two chunks against one ubatch, forced and free,
a jump without a flip in the second chunk, two chunked runs). `dsv41_engine_smoke --dry-run` with the pod's arguments
plans 8 layers x 384 experts, 168 in VRAM per layer, chunks from 16384 down to 2048, ctx 16480, three decode phases,
the projection on a PCIe 4 x16 link of 31.5 GB/s for the two targets. `pod/devgpu.sh --dry-run` shows the pull: two
`tar -czf` on the pod under `timeout 120`, the logs pulled first, the dumps only up to 512 MiB, all within 300 s, the
pod deleted right after, the pull's seconds and sizes logged.

Again after the projection changes (2026-10-07, `verify.sh`, both flavors, 7m34s): every
check passes. The smoke's dry run with the pod's arguments (`--pod-link 4:16 --pod-tflops 154.8`) plans the memory
table first (the real 40 layers, 32 experts per layer and the router of 384, at ctx 262,144, chunks 2048 up to
16384), then the same run, and the projections: RTX PRO 6000 on 63.0 GB/s with 96 GiB, L40S on 31.5 GB/s with 48 GiB,
each hot fraction (VRAM - the measured GiB - 2) / 183.25 GiB, compute scales 1 and 3.25 (RTX PRO 6000) or 2.34
(L40S) against an RTX A6000. `GPU_TYPES='NVIDIA RTX A6000;NVIDIA A40;NVIDIA L40' MIN_RAM_GB=64 pod/devgpu.sh
--dry-run` lists the three GPUs, the 64 GB minimum and `"minRAMPerGPU": 64` in all 12 create bodies.

## Routed experts at tensor-core speed in prefill, the real routing in the smoke

The previous section's
A100 run spent 588-673 ms per layer in the routed experts of a 16384-token chunk (98,304 pairs, 6.96 TFLOP, ~11
TFLOP/s), so the 16K TTFT projection was 29 s against the 5 s target.

Where the ~620 ms went (from that run's counts and the same pod's `qmatmul_bench`): `plan_prefill` (strata-moe's,
shared with `moe_gpu.cpp`) cuts the slab's pairs in pair order every 4096, so with 50,621 slab pairs per layer each of
the 168 slab experts came back in each of 13 slab passes with ~23 rows. That is 2184 expert calls per layer, each three
qmatmuls that dequantize the whole matrix into the f16 scratch (39-52 us at any row count: dequant + cuBLAS minus
cuBLAS at 16 rows) and run cuBLAS at 16-row shapes (6 TFLOP/s): 2184 x (0.063 + 0.063 + 0.073) ms = ~435 ms, plus
the 216 ring experts' ~256-row calls (~0.27 ms each, ~58 ms), plus the clamp, gather and conversion launches. About
170 GB of f16 weights per layer were written and read again (7200 matrices of 23.6 MB).

| Piece | What | Strata origin |
|---|---|---|
| `experts.cpp` `regroup_slab` | the plan's slab passes regrouped by slot into passes of whole experts (at most max(4096, chunk) rows; an expert is cut only when it alone has more), the rows still 0 .. n_slab - 1, so the ring passes keep theirs; each slab expert's weights are read once per call | `moe_prefill.cpp` `plan_prefill` unchanged (the fork hook's GPU tier still uses it as it is) |
| `experts_gemm.cu` | one pass = the gather of its rows into f16, one launch of gate, up and the clamp for every expert (h in f16), one launch of down: tiles of 128 rows of one expert x 128 weight rows (gate/up: 64 features with their gate and up rows), k in steps of 32 through two shared buffers (activations by `cp.async`, weights decoded by the block from the q2_K / q3_K blocks where the slab, ring or prefetch region holds them, the next step's loads issued before the MMAs), `mma.sync m16n8k16` f16 x f16 -> f32, 8 warps of 32 x 64 accumulators, 2 blocks per SM (128 registers, 40 KiB of shared memory) | `src/prefill/moe_fused.cu` (#136): the grouping into tiles of one expert, a block per (tile, weight slice), weights decoded in the kernel, the cp.async pipeline, gate/up with the activation then down; its int8 m16n8k32 does not carry over (a q2_K / q3_K block scales every 16 values, Strata's Q2_0 every 64: an int8 MMA over 32 values would need its int32 rescaled per 16 values and row, the ALU work that keeps the fork's q2_K MMQ at 45-52 TFLOP/s on the A100 against cuBLAS f16's 197-210) |
| `qdequant.cuh` | the dequant path's block decoder in one place, split into `fetch8` (the loads) and `decode8` (the arithmetic), used by `qmatmul.cu`'s dequant kernel and the grouped GEMM alike; the q3_K scale read straight from its two bytes instead of ggml's kmask unpacking into a local array | `qmatmul.cu`'s `dequant8`, from `dequant_bf16.cu` |
| `ffn_device.cuh` | the fork's SwiGLU clamp as one device function, for the shared expert, decode's hits and the grouped GEMM's epilogue | `dense_ffn.cu`'s kernel |
| `forward.cpp` `routing` | a synthetic routing: every call's router ids from a function (the router's weights for them, as `--force-selection`) | the forced ids |
| `engine_smoke.cpp` `--counts` | the routing and the hot set from a strata-moe-counts/1 profile (below), the routed-expert TFLOP/s against `--pod-tflops` (gate 35%), per decode phase the hit rate, the CPU experts per layer, the CPU's ms and GB/s (per thread too) and the 40-layer ms per token | the previous smoke |

Numerics: the dequant path's (`QPrefill::DequantF16`) for every prefill row count, also for an expert with 8 rows or
fewer, which the per-expert path sent through MMVQ on q8_1: each weight is the f16 value `qref_dequant_row` gives
rounded to nearest even, the activations are rounded to f16 the same way (`cvt.rn.f16x2.f32`), the products are
exact and the tensor cores sum them in f32; the clamp runs on the f32 gate and up and h is rounded to f16 for down,
as `qmatmul_f16` rounded it before. `qref_matmul_f16` is the reference. A row's values do not depend on the other
rows of its tile or pass, so the three placements stay bit-identical. Decode's hits keep qmatmul (MMVQ, at most 8
rows per expert); their f32 buffers shrank to 48 rows, and the prefill pass buffers are f16 now: 14.8 KB per pass
row instead of 48 KB (at chunk 16384, 232 MiB instead of 752 MiB, which the memory table gives back to the hot set).

Tests (`dsv41_engine_experts_test`): `--cases` also runs the prefill numerics on the host on each dump's own inputs
and selections against the fork's `ffn_moe_out` (bound 3e-2); `--gpu` checks prefill pairs against the f16 reference
at every row count; `--gemm` (new, the pod) runs the two kernels at the real dims on six random experts of 1, 7, 64,
128, 129 and 300 rows (gathered from a shuffled x) against `qref_matmul_f16` and the host clamp (bound 1e-3 per
expert), then one-hot rows (a single 1 at k = 0 .. 511 and 64 random k) whose outputs are single weights and must
equal the dequant path's f16 exactly, q2_K through the gate as a plain grouped GEMM and q3_K through down, then times
48 experts x 256 rows (a 16K chunk's mean per expert) and x 32 rows: gather, gate/up and down in ms and TFLOP/s.

`--counts FILE` in the smoke: FILE is the hybrid's serving profile (a strata-moe-counts/1 file the fork's Strata hook
recorded on the real model, tools/dsv41/README.md; 1530 decode calls per layer; `COUNTS` in `verify.sh` and
`pod/devgpu.sh`). Smoke layer l routes by its layer l * 40 / --layers (8 layers: 0,
5, .., 35); each token's 6 distinct experts are drawn from that layer's decode counts plus one (an expert the profile
never saw keeps a chance below every seen one: the real model routes it a few times in a 16K prompt), seeded by
layer and position. The slab keeps the same number of experts as the uniform run (168 x 8), chosen by
`hot_set.py`'s rule (a global greedy by count over the layers, ties to the better rank in the layer, then the lower
layer), so the per-layer slab sizes differ (129 to 204 at 16.12 GiB).

Pod runs (`pod/devgpu-run.sh`): every earlier step (the forced e2e runs now take the grouped GEMM in every prefill
chunk, chunk 2048 and 24), `experts-gemm` (`--gemm`), `smoke-e05b` as before (the synthetic router's routing) and
`smoke-e05c` (the same with `--counts e05c/counts.json`). The A100's published dense FP16 tensor 312 TFLOPS is now
passed by name, so both smokes print the gate (routed-expert compute >= 35% of it: <= ~64 ms per layer for 6.96
TFLOP) and the projection at both compute scales. Session (an A100 80GB when no 48 GB GPU is in stock):

    export RUNPOD_API_KEY=...
    GPU_TYPES='NVIDIA A100 80GB PCIe;NVIDIA A100-SXM4-80GB' MIN_RAM_GB=64 MAX_USD_PER_HR=1.50 \
      bash tools/dsv41/pod/devgpu.sh

Measured on the dev VM (2026-10-07, 12 vCPU, no GPU; `build.sh` both flavors, `verify.sh` both flavors 7m32s): every check passes. ptxas (sm_80 and sm_120): 127-128 registers and 40 KiB of shared memory for
every grouped kernel, no spill in the gate/up kernels of q2_K, 24 B spilled by the q3_K down kernel on sm_80.
`experts_test --cases`: the prefill numerics against the fork's `ffn_moe_out` at worst 2.01% to 2.51% rel L2 per dump
(bound 3%; the per-expert path's engine numerics gave 2.19% to 2.24% on the GPU test's random calls).
`--gpu --dry-run`: tiny-a 32 and tiny-b 192 tiles per 64-token chunk over every layer in each placement.
`--gemm --dry-run`: six experts, 629 rows in 9 tiles, the one-hot references of 576 q2_K and 576 q3_K rows. The smoke's
`--counts` dry run (`--pod-tflops 312`): per-layer slabs 129 to 204 experts (profile layers 0, 5, .., 35), 90.2% to
92.5% of each layer's decode counts on the slab, 91.4% on average (in-sample), compute scales 1.61 (RTX PRO 6000) and
1.16 (L40S) against the A100. `GPU_TYPES='NVIDIA A100 80GB PCIe;NVIDIA A100-SXM4-80GB' MIN_RAM_GB=64
MAX_USD_PER_HR=1.50 pod/devgpu.sh --dry-run` lists the two GPUs, the 1.50 USD/h limit, `"minRAMPerGPU": 64` in all 8
create bodies and `e05c/counts.json` in the payload. The dry run reads only the tarball's listing: it needs exactly one
cu128 strata tarball in `$BUILD_ROOT/dist` naming every payload binary; the sha256 of each against its MANIFEST is
checked only when a real run stages the payload.

### After the first A100 run: wide passes, no per-pass upload, no spill

The first pod run (A100 80GB PCIe) was correct
(every forced e2e run, the `--gemm` checks) and 3.2x faster than the previous section's, but the routed-expert compute was 191.5 ms per
layer with the synthetic routing and 214.5 with `--counts` (11.6% and 10.4% of 312 TFLOPS; gate 35%, <= ~64 ms),
while `--gemm` alone ran 48 experts x 256 rows at 109.6 TFLOP/s (gate/up 139.1, down 83.1 with the 24 B spill). By
those rates a layer's 956 tiles are ~80 ms of kernels. The `--counts` layers say where the rest went: 16 to 20 passes
per layer (ring sets of 16) and 180 to 238 ms of compute, 10 to 14 ms more per extra pass, about one ring set's copy
(16 x 12.8 MB at 20 GB/s); each pass uploaded its pairs and tiles on the engine's stream, most likely queued on the
copy engine behind the set copies the copy stream had in flight.

| Piece | What |
|---|---|
| `experts.cpp` `prepare_passes` | every pass's pairs grouped by slot, the output rows and the tiles planned on the host before the first pass and uploaded once per layer (three copies, ahead of the layer's ring copies); a pass queues its gather and two GEMM launches only |
| ring sets of 48 (`ExpertConfig::ring_set`, the smoke's and `dsv41-engine`'s `--ring-set` default) | a 16384-token chunk's ~184 ring experts per layer (216 outside a 44% slab, 32 prefetched) in 4 passes instead of 12: ~12,300 rows per pass with uniform routing, ~2,200 with the real one; with the slab's passes of up to 16384 rows, ~8 to 11 passes per layer instead of 16 to 20 |
| `moe_prefill.cpp` `plan_prefill(..., max_pass_pairs)` | the pass cap as a parameter: the engine passes its buffers' rows (up to the chunk), the fork hook keeps the default 4096 (`moe_gpu.cpp` unchanged); without it a set of 48 still closed at 4096 pairs (16 experts of 256) |
| `experts_gemm.cu` | B fragments loaded one n8 pair at a time between the MMAs (4 registers live, not 16): ptxas gives 128 registers, 40 KiB of shared memory and no spill for every grouped kernel on sm_80, sm_89 and sm_120 (the q3_K down kernel spilled 24 B, the q2_K plain one 8 B); a warp's 16-row group with none of the tile's rows skips its ldmatrix and MMAs, so a ring expert's ~45-row tile under the real routing costs the tensor cores its rows, not 128 |
| `engine_smoke.cpp` | per layer and in the gate line the compute split into gather, gate/up and down (with their TFLOP/s), the tiles per pass, the gate/up (tiles x 36) and down (tiles x 40) blocks per launch and the GPU's SMs; the memory table prints the ring's share |

The wider ring's cost (12,810,240 B per slot, 3 sets): 1.72 GiB of VRAM instead of 0.57 (+1.15 GiB), and, for a source
that is not pinned (the real model's mapped GGUF), pinned staging of 2 sets x 48 slots, 1.15 GiB of host RAM instead
of 0.38. The memory table measures the ring with the rest of the non-expert memory, so the projection's hot fraction
pays for it: at chunk 16384, by the first run's table plus 1.15 GiB, RTX PRO 6000 40.9% -> 40.3% hot (2.91 -> 2.94 GB
of copies per layer per chunk), L40S 14.7% -> 14.1%; the pod's table replaces these.

Not done: slab and ring experts in one launch. With sets of 48 a ring pass of the real routing has ~48 tiles, 1,728
gate/up and 1,920 down blocks (16 to 18 per SM on 108 SMs, 2 resident per SM), a uniform one ~100 tiles, a slab pass
up to 128 tiles; a launch over the slab and the ring would also wait for the last ring copy. 64-row tiles are not
done either: the skipped MMAs cover a partial tile's tensor work, its decode of the weights stays per tile.

Measured on the dev VM (2026-10-07, 12 vCPU, no GPU; `build.sh` both flavors 2m45s, `verify.sh` both flavors 7m28s): every check passes. `dsv41_moe_prefill_test` 181 cases, among them every plan case again with ring
48 x 3 at cap 16384 and, at 16384 tokens with the test's skewed routing, 26 passes of at most 32 experts at the
hook's cap 4096 against 9 of up to 48 at 16384, each closed only when full or when the next expert would pass the
cap. The tiny models' forced and free dry runs plan their 64-token chunks on 3 sets of 48 (tiny-a all-CPU: 4 ring
passes instead of 16); the chunk-24 runs keep their ring of 2 sets of 2, which still cycles the sets within a call.
The smoke's dry runs print `ring 3 sets of 48 (1.72 GiB of VRAM)`. `GPU_TYPES='NVIDIA A100 80GB PCIe;NVIDIA
A100-SXM4-80GB' MIN_RAM_GB=64 MAX_USD_PER_HR=1.50 pod/devgpu.sh --dry-run` takes `strata-1087c2a4-cu128-sm80-89.tar.gz`,
lists the two GPUs, the 1.50 USD/h limit, `"minRAMPerGPU": 64` in all 8 create bodies and `e05c/counts.json`.

## Decode as one CUDA graph per token, the experts' hits grouped, the misses through a doorbell

The decode target is >= 30 tok/s. The previous section's A100 smoke (8 layers x 384
experts, `--counts` routing, 88.9% hits, graphs + pool spin) ran 9.72 ms per token, a 40-layer projection of 48.6 ms
(20.6 tok/s); per layer per token (its timed phase) attention ~0.40 ms, routed experts ~0.84 ms although the hits read
only ~68 MB (~36 us at the A100's bandwidth) and the CPU's misses took 0.25-0.4 ms, hyper-connections and norms ~0.13,
shared expert ~0.07, router ~0.03; head 0.58 ms per token; 9 host syncs per token (one per layer for the routing, one
for the logits) and ~25 direct launches per layer for the attention. Correctness stays the forced parity.

| Piece | What | Strata origin |
|---|---|---|
| `qmatmul.cu` `qmatmul_experts` | every hit of a decode call in four launches whatever the routing: x to q8_1 once per token, gate, up and the SwiGLU clamp in one kernel (two matrices per block, the clamp in the epilogue), h to q8_1, down; block row p is pair p, its slot from the layer's residency row on the device (a miss, slot -1, returns at once); each row is `mmvq_kernel`'s sum at one column, so a hit is qmatmul's value bit for bit | `iq_kernels.hpp` `native_expert_grouped` (residency on the device, a block row per entry, fused SwiGLU), `session.hpp` `TokenHits`; the quantized matmul's MMVQ traits |
| `experts.cpp` decode | `decode_publish`: one kernel copies the input rows and the router's ids to mapped pinned memory, a system fence, then the ring (a counter incremented in memory); `decode_finish`: the hits, a one-thread kernel waiting for the host's flag, the CPU rows and each pair's row read from mapped memory, the combine; `decode_serve` (host): polls the ring (a stream query every 2 ms notices a fault or a stream that ended, 20 s bound), splits the ids, runs the misses on the CPU pool from the warm tier or the mapping into mapped memory, writes the flag. A layer whose every expert is in the slab rings nothing (unless counts are recorded). `forward()` decode is the three in a row | `core/layer.hpp` `Doorbell` and `layer.cpp` `doorbell_init` (mapped pinned regions), `elementwise.cu` `doorbell_publish_kernel` / `doorbell_wait_kernel` / `copy_from_mapped_kernel` (the fence, the volatile store, the in-memory increment against the cloned-literal bug), `session.cpp` `session_run_token`'s host loop |
| `attention.cu` `attn_decode` | the position read on the device (`scratch().pos`): the compressor's block completes on the device (`compress_decode_kernel`: pool or carry, the block's first position for the rope; `store_block_kernel` stores the row and the index key only when complete), the indexer over its key source's capacity (each kernel bounds it by the position: `indexer.hpp`), the attention with a split count fixed by the layer's largest cell count and the cells per split computed per token; `attn_decode_advance` is the host's checks and bookkeeping of a replayed token | `qsa.hpp`'s step buffer |
| `forward.cpp` token graph | every layer's engram, hyper-connections, attention, router, doorbell, shared expert (after the ring, so the CPU's misses overlap it), hits, wait, combine, then the head, captured at the first untimed decode call; per token the host writes the token, the position, the engram's rows (hashed and copied from the pinned tables before the launch) and, forced or synthetic, every layer's router ids and index rows into pinned memory the graph's first nodes copy, launches once, serves the doorbells in order, waits once for the logits; `--dump-layers` takes per-layer copies the graph makes. The per-layer dense graphs of the overlap section are gone; `--no-graphs` and timing queue the same device work layer by layer | `session.hpp` `session_capture_token` / `session_run_token` |
| `ExpertConfig::pool_spin_us` | 50 by default (the spin cut the pool's wake 1.386 -> 0.814 ms and its return 0.310 -> 0.063 ms per token) | `kernels/cpu/pool.hpp`'s spin |
| `engine_smoke.cpp` | the timed phase's split per layer per token; every phase's doorbells, host ring wait, host syncs and graph launches per token; the 40-layer projection (ms per token - head) x 40 / layers + head (the timed phase's head once, not per layer) against 30 tok/s | the previous smoke |
| `pod/devgpu-run.sh` | every e2e run decodes from the token graph; per tiny model the split placement forced again with `--no-graphs` against the fork and against the graph's run | the earlier e2e runs |

What the token graph removes, from the previous split: the host's sync and its ~4 launches per hit per layer (the routed
experts' 0.84 ms against ~36 us of reads and the CPU's 0.25-0.4 ms), the attention's ~25 host launches per layer, and
8 of the 9 host syncs per token. What it does not remove: the CPU's misses stay on each layer's path (the combine
needs them; only the shared expert and the hits overlap them), and the head is unchanged (0.58 ms
per token, 1.7% of a 33 ms token; changed only if the token graph's profile shows it). The pod's run prints both
routings' split, the projection and its verdict; if it misses 30 tok/s, the floor per stage is that split.

Measured on the dev VM (2026-10-08, 12 vCPU, no GPU; `build.sh` both flavors 2m46s, `verify.sh` both flavors 7m39s): every check passes. The tiny models' dry plans give the decode path per placement, checked by
`verify.sh`: all-VRAM grouped hits in every layer and no doorbell (tiny-a 4 of 4 layers, tiny-b 6 of 6), all-CPU a
doorbell in every layer and no hits, split both. The host side of the attention's and the experts' GPU tests passes
(the slab's layout checked by `qmatmul_experts_check` in every layer with a slab). The smokes' dry runs list the
decode phases `timed; token graph; token graph with the pool spinning 50 us`. `GPU_TYPES='NVIDIA A100 80GB
PCIe;NVIDIA A100-SXM4-80GB' MIN_RAM_GB=64 MAX_USD_PER_HR=1.50 pod/devgpu.sh --dry-run` takes
`strata-2ed9b488-cu128-sm80-89.tar.gz`, lists the two GPUs, the 1.50 USD/h limit, `"minRAMPerGPU": 64` in all 8
create bodies, `e05c/counts.json` and the session directory `e05d-pod`. Nothing of the new path has run on a GPU yet.

### Fused decode launches, staged single-column MMVQ, the per-kernel split

The first A100 run (A100-SXM4-80GB) was correct
(every forced run, graph and `--no-graphs`) with 1 host sync per token, but projected 36.4 ms per token (27.5 tok/s,
`--counts`, pool spin) against the target's 30. Per layer per token (timed): attention 0.404 ms for ~40 MB of q2_K
(~21 us at the A100's bandwidth), routed experts 0.390, hyper-connections and norms 0.128, shared expert 0.068, router
0.029; the head 0.581 ms per token for 543 MB of q6_K. The attention bench split a Reuse layer's 0.41 ms at 4K into
projections 0.17, attention 0.07 and output 0.17: about 17 launches, the norms, ropes and quantizers one block each
(the q rope one block of 128 threads over 64 heads), and MMVQ at 190 GB/s (q_a) to 540 GB/s (wo_b); qmatmul_bench put
attn_q_b at T = 1 at 355 GB/s (80 of 128 threads busy on its 5-block rows).

| Piece | What | Strata origin |
|---|---|---|
| `mmvq.cuh` `mmvq1_kernel` | MMVQ at one column: a block of 4 warps copies its R rows (R * row bytes contiguous, up to 12 KiB: R = 16 for attn_q_b, 4 for the x projections and attn_output_b, 8 for attn_output_a, 2 for the head) to shared memory with 16-byte `cp.async`, then runs `mmvq_kernel`'s per-row sums from there in its order (the 4 warps' partials added in warp order, the xor sum), so every value is the old kernel's bit for bit; several matrices on one input in one launch (segments) and an epilogue functor per row (the row pair of a rope). `qmatmul_q` takes it at T = 1 for every matrix whose rows stage (16-byte aligned data), `qmatmul_experts` stages its gate/up pairs and downs the same way | `native_mmvq.cu`'s multi-column kernel and traits (moved here from `qmatmul.cu`) |
| `attention.cu` fused decode token | with q2_K weights a decode token is 9 launches (Reuse, window), 10 to 12 (Full, Reindex): x's q8_1 (skipped when the caller hands it), the x projections in one MMVQ launch (q_a, kv, the compressor's kv and gate, indexer.proj with its scale), `decode_qkv_kernel` (the q_a norm with qr's f16 and q8_1; the kv norm, rope and window cell), the qr projections in one launch (q_b with its rope and q's f16, indexer.attn_q_b with its rope), `decode_book_kernel` (the compressor's pool or carry, norm, rope and row; the index key from indexer.attn_k staged in shared memory, its norm, rope, FWHT and key; the index queries' FWHT), the selection, `attn_kernel<D, true>` (q read as f16, an empty split returns at once, splits of 32 cells: 20 at 640 cells instead of 10), `merge_derope_kernel` (merge, de-rope, f16, q8_1), wo_a, oa's q8_1, wo_b. Each stores the unfused kernels' values bit for bit (their device functions shared: `rope_pair`, `rms_scale`, `fwht_row`, `q8_1_warp`), except that the attention's normalization is always the split merge's. f16 weights (the test path) and matrices that do not stage keep the unfused launches. Kernel marks (`AttnEngine::set_kernel_marks`, `cudaEventRecordExternal` so they become graph nodes) give the split | the attention's kernels; `fused_gr.cu`'s q8_1 images written by the producer (S26) for the quantizers |
| `hc.cu` `hc_decode_step` | a sublayer's hyper-connections in one launch of 6 blocks of 512 threads: the expansion of the previous sublayer's output (when given), the squares of the stream in `hc_norm_kernel`'s order, rms(x)'s q8_1 in shared memory, one hc_fn row per 128 threads on its staged row in `mmvq_kernel`'s order; the last block (a ticket in device memory) writes the expanded stream, runs `collapse_norm_kernel`'s 256 threads (named barrier) with the normed row's q8_1, and `hc_coeffs_kernel`'s thread beside them. The dense test checks every value against the unfused ops bit for bit | `fused_gr.cu` (the previous half's write folded into the next read, S26's q8_1 image of the output for the next matmul, the last-arriving block by an atomic counter after a fence) on `native_gr_postops.cu`'s math |
| `dense_ffn.cu` decode | the shared expert at T = 1: up and gate in one MMVQ launch on x's q8_1 (the caller's), the SwiGLU with act's q8_1, down: 3 launches instead of 7, the same values | `shared_expert.cu` |
| `forward.cpp` | both decode paths (token graph, timed / `--no-graphs`) on `hc_decode_step`; the attention's, the shared expert's and the hits' input is the step's q8_1 of h; the FFN output's expansion is folded into the next layer's attention step (not before an engram layer, nor with the dumps, which read the stream after each layer). `profile_decode_attention` captures each layer's attention alone with its kernel marks and replays it (the cache's bookkeeping restored; what the replays write the next token writes again), `profile_head` the head | `session.hpp`'s graphs |
| `engine_smoke.cpp` | after the decode phases: each layer's attention in a graph (20 replays) with its split by fused launch, the mean per layer by role and over the real 40 layers' roles; the head's GB/s against the device-to-device copy and the published HBM bandwidth (memory clock x bus width); one line of the targets (attention <= 0.15 ms per layer, hyper-connections and norms <= 0.06 ms per layer timed, head >= 70% of HBM, >= 30 tok/s with the pool spin) | the previous smoke |

Launches per layer of a decode token, timed or in the graph: the attention 17 to 25 -> 9 to 12, the hyper-connections
and norms 7 -> 2 (3 before an engram layer), the shared expert 7 -> 3, the hits 4 -> 3. None of it has run on a GPU:
the pod's forced e2e runs (graph and `--no-graphs`), `attention-test` (the fused token against the CPU reference at
every decode call), `dense-test` (the step bit for bit against the unfused ops at T = 1) and `qmatmul-test` (every real
shape at T = 1 on the staged kernel) prove it, the smokes measure it.

Measured on the dev VM (2026-10-08, 12 vCPU, no GPU; `build.sh` both flavors 3m27s, `verify.sh` both flavors 7m39s):
every check passes. cuobjdump (sm_80): no local memory in any new kernel; `hc_step_kernel` 64 registers at 512
threads (24 B of stack), `attn_kernel<512, true>` 147 registers as the prefill instance, `mmvq1_kernel` 40 to 80, the
staged expert kernels 40 to 96, `decode_book_kernel` 64. The smokes' dry runs list `then each layer's decode attention
and the head alone in graphs (20 replays)` and the measurement smoke's context 16481 (the profile's position).
`GPU_TYPES='NVIDIA A100 80GB PCIe;NVIDIA A100-SXM4-80GB' MIN_RAM_GB=64 MAX_USD_PER_HR=1.50 pod/devgpu.sh --dry-run`
lists the two GPUs, the 1.50 USD/h limit, `"minRAMPerGPU": 64` in all 8 create bodies and `e05c/counts.json`.

## The prefix cache across turns and the OpenAI-compatible server

A fast second turn and an OpenAI server. The engine stays token ids only; text lives in `serve/server.py`. One cached
sequence with two resume points, the end of the last request and the end of its prompt.

| Piece | What | Strata origin |
|---|---|---|
| `kv_cache.hpp` snapshot | one snapshot of the running state: every layer's next position and a KV source's pooling carry (the open block's kv and score, ratio x head_dim f32 each), copied device to device on the stream; the window cells, compressed rows and index keys stay in place and are written again past the snapshot. An index source's top-k ids are not kept: each call's index source writes them before its readers read them. Restorable while the sequence is at most `max_chunk` positions past it: the window ring (`n_swa - 1 + max_chunk` cells) reuses the cells of the `n_swa - 1` positions before it after that | `core/conversation_cache.hpp` `ConversationCheckpoint` (the running state only; the identity-layout KV pages stay where they are) |
| `forward.hpp` | `reset`, `save_snapshot`, `can_restore`, `restore_snapshot` (the token history cut back for the engram's n-grams; the token graph replays unchanged: its buffers are the same, the position comes through pinned memory), `set_forced` (a second reference for the parity mode), `dump_values = false` (dumps of the selections only: a 16K prompt's `l_last` and `idx_score` are GiBs), `ForwardBackend` | the layer loop's Forward |
| `session.hpp` | the policy, host only: a request whose ids extend the live tokens resumes from them; one that extends only the last prompt restores the snapshot; else position 0. Then the rest of the prompt in chunks, the snapshot at its end, greedy decode where each token is printed and fed only when another one is wanted (the live state is the prompt and every generated token but the last), STOP between chunks and tokens. `check` refuses a request that cannot run (empty, an id outside the vocabulary, more positions than ctx) | `src/program/generate.cpp`'s `--serve` block: the live session or the longest checkpoint the prompt starts with, the decode loop |
| `dsv41-engine --serve` | `INFO`, `READY <ctx> stop`, `GEN <max_new> [key=value ...] <ids>` with `RESUME`, `PP`, `T`, `DONE` (9 fields: drafts 0 0, then the reused tokens), `ERR`, `STOP`, `QUIT`; a reader thread so a STOP reaches a running request; greedy at temperature 0 or none (top_k, top_p, min_p, seed accepted and unused); temperature > 0 samples on the host (sampler.cpp: top_k, top_p, min_p, temperature, a seeded std::mt19937_64 draw; a bad value is an ERR naming the key), active penalties are an ERR naming the OpenAI field, each request's stderr line ends in `greedy` or its sampler and seed; `det=0|1` and `--deterministic on|off` (the deterministic mode, on by default, [DETERMINISM.md](DETERMINISM.md)), the mode named before `greedy` or the sampler; `--eos-ids` (default the GGUF's eos id), `--log-ids`; `--serve --dry-run` the same protocol on the dev VM with a predictable stand-in (the last token read plus one) | `generate.cpp`'s protocol (5032-5048, the READY and DONE lines, the stdin thread) |
| `engine_smoke.cpp --prefix-cache N` | (A) free: turn 1 (16,384 + 32), turn 2 (+ N new) timed per stage with its RESUME and TTFT, the same tokens read cold (a re-send no snapshot covers) and its TTFT, the 40-layer projection `(TTFT - embed - head) x 40 / layers + embed + head`; (B), (D) cold references with their selections dumped; (C) forced: the live resume against B, then a cold turn 1 and the prompt-end restore against D, both within 5e-2 rel L2 | the earlier smoke |
| `tools/strata_tokenizer.py` `joyai-llm` | the fork's three regexes through llama.cpp's `unicode_regex_split` (the collapsed text, the `\p{..}` classes as category bytes plus their ASCII members, std::regex's ASCII `\s`), the categories from `tools/llama_unicode_flags.txt` (the fork's `src/unicode-data.cpp`, `llama_unicode.py`), special tokens split in `tokenizer_st_partition`'s order (longest text first), the BPE byte fallback; a special token's text decodes to its UTF-8 | Strata's own BPE (the qwen35 path, unchanged) |
| `serve/model_profile.py` | the `deepseek-v4.1` profile (`"model_profile"` and `"tokenizer_gguf"` in the server config): the tokenizer and chat template from the GGUF (`ChatTemplate(source=...)`, `bos_token`, `from_json`), the eos id as the stop, thinking off unless asked (the template ends with `</think>`), tools refused with 400 (the model's DSML calls are not parsed yet), Qwen-only options refused at start | `serve/server.py` as is otherwise (reused behind the same protocol) |
| `serve_check.py` | the profile, the protocol through the server's own `StrataEngine` (live resume, prompt-end restore, cold, the window limit, PP, the end of a reply, STOP, ERR), the HTTP server's chat completions streaming and not and a resumed second turn; `--reference dry` predicts every id, `--reference cli` compares with `dsv41-engine --tokens-file` | `serve/test_server.py`'s HTTP pattern |

The tokenizer is Strata's own BPE rather than HF `tokenizers`: the fork's pre-tokenizer is std::regex on a collapsed
text with llama.cpp's own category table, which a Unicode-category regex of another library (or of another Python's
`unicodedata`) does not reproduce for every codepoint; emulating the collapse reads the fork's table, and
`tokenizer_parity.py` proves the result. The parity runs on `$VOCAB` (the real GGUF's tokenizer metadata, tools/dsv41/README.md) and on
its 512-token cut: the tiny models are `no_vocab`, so the tiny-b server uses that cut (the specials, the 256 byte
characters and the first 253 merges).

Measured on the dev VM (2026-10-08, 12 vCPU, no GPU; `build.sh` both flavors 2m49s, `verify.sh` both
flavors 16m42s, of which the tokenizer parity takes ~112 s and the server's tests ~154 s per flavor): every check
passes. Tokenizer parity: 227 texts (226 lines and the whole corpus) with specials and 29 without, 4033 ids (80
special) on the real tokenizer and 9900 on the cut, every one equal to `llama-tokenize`, every round trip exact. The
server's own tests: 420, 10 skipped, all pass with the profile code in. `serve_check.py --reference dry`: 47 checks,
among them RESUME 24 (a live resume of 20 prompt + 4 fed tokens), RESUME 27 (the prompt-end restore), 8 fed tokens
past the snapshot restore and 9 read cold at `--chunk 8`, STOP after 3 read tokens ending the reply at 52 of 1999000,
the three ERRs, `/v1/chat/completions` JSON and SSE equal to the predicted ids' text, and a second turn with 62 of 101
prompt tokens cached. `pod/devgpu.sh --dry-run` (the A100 overrides) lists the server's payload and steps and the session
directory `e06-pod`. The pod's `serve-tiny-b` and `smoke-e06-prefix` are the device-side gates.

### Fixes after the first verification: the prefill split, the server check's paths, a step's leftovers

The A100 pod's second session (0.310 USD) passed the prefix cache's correctness
(forced: the live resume at rel L2 0.00228 against the cold read, the prompt-end restore at 0) and missed its speed:
turn 2 (16,415 cached + 513 new tokens) took 777.5 ms on 8 layers against 2367.9 ms cold, 3885 ms projected to 40
layers against the target's 1000; 720 ms of the 777 were routed experts, 11.7 GiB of experts outside the slab through
the ring for one 513-token read. The HTTP half of `serve_check.py` never ran (a config path relative to another cwd).

| Piece | What | Strata origin |
|---|---|---|
| `experts.hpp` prefill split | a prefill chunk of at most `split_max_t` (2048) tokens: the slab's pairs on the GPU; of the experts outside it, the ones with few pairs on the CPU pool from where they are (warm tier or mapping, decode's CPU kernel and numerics) while the others go through the ring, at once. Per layer and call the experts by pair count, fewest first, go to the CPU while that lowers max(ring, CPU) + the CPU rows' upload, at rates measured in `init` (`PrefillSplitRates`: up to 16 experts copied to the ring on the copy stream; the same on the CPU pool at 1 and 16 tokens each). The call's input rows come to the host with its routing (one wait), the CPU runs while the passes run, its rows go up on the copy stream behind the ring's copies into the output rows past the GPU's, and the combine waits for them. No prefetch for a split chunk. `--prefill-split measured|cpu|off`, `--split-max-t N` | the hybrid's hit/miss split (decode's), Strata's `--short-read` idea of reading short prompts the decode way |
| `experts_test --gpu` | two more placements, all-CPU and split with `PrefillSplit::Cpu`: their 64-token prefill bit for bit the CPU reference (all-CPU) or within kTolEngine of the slab pairs' engine reference and the others' CPU reference (split); the first three keep the split off, bit-identical across placements | the earlier test |
| `engine_smoke.cpp --prefix-cache` | turn 2's split (pairs on the slab, the ring and the CPU; the ring's copy ms from events, the CPU's host ms; the split's estimates and the measured rates) and its physical floor on `--floor-target` (RTX PRO 6000:5:16): the experts outside the slab it touches x 40 / layers over PCIe (the target's link x this pod's pinned-H2D efficiency against `--pod-link`) + the CPU pool's GB/s measured here, with the rest of turn 2 projected, against the 1 s target | the previous smoke |
| `serve_check.py` | every path it hands on absolute (`--out`, the engine, the model, the tokenizer, engine arguments that name files); every process it started stopped on its way out (the engines, the server and the server's engine, taken while the server runs), a failed check and SIGTERM included | |
| `pod/devgpu-run.sh` | once a step returns, what is left of its process group (timeout leads it) gets SIGTERM, SIGKILL 10 s later, before the next step, logged to `out/leftovers.log`; the chunk-24 forced runs with `--prefill-split off` (they test the ring), the one-chunk e2e runs read through the split | |

The split's bound of 2048 tokens leaves the 16K chunks on the ring, where CPU prefill was discarded. The CPU rates stand in for the target's host in the floor (it is not measured).

## The real model on one 80 GB GPU, the session script

`tools/dsv41/pod/e07.sh` takes one A100-SXM4-80GB (L40S fallback), optionally in the data center of a RunPod network
volume holding the model (`NETWORK_VOLUME`, mounted read only at /workspace), copies the model to the local disk and
runs, in this order, correctness against the real-model oracle, tokenizer parity on the real GGUF, the 16K TTFT with its second turn and short
decode, the server, then 64K and 256K (`pod/e07-run.sh`'s header has every step), then writes the session sheet
(`pod/e07_report.py`). Every engine run takes `pod/e07_gate.py`'s placement: the slab VRAM - 20.22 GiB beside the
experts (the measured memory table at chunk 16384, ctx 262144) - 2 GiB, filled from the real counts profile, every
other expert pinned in the warm tier (none cold), the gate checking warm + engram + working memory against the cgroup
before the copy.

| Piece | What | Strata origin |
|---|---|---|
| `forward.cpp` dumps | with `--dump-layers` (values on) every call and layer also writes `ffn_norm` [T][E] (the MoE input), `ffn_moe_weights_scaled` [T][K] and `ffn_moe_out` [T][E], the routed sum before the shared expert: the experts run with the shared expert null and `expert_add_shared` adds it after (one rounding, the combine's bits); the token graph copies the same into per-layer buffers | the earlier dumps |
| `dsv41-engine --eos-ids none` | `--serve` with no end-of-reply id: every request generates its max_new tokens (the benches' fixed decodes) | |
| `oracle_parity.py`, `gguf_layout.py` | above | `engine_parity.py`'s Run, rel L2 and margins; `prefill_prompts.py vocab`'s header copy |
| `serve_check.py` | `--parts`, `--ctx`, `--load-s`, `--chats` (3 chat prompts, replies saved), the first request's reference before the server starts, `--one-engine` (no reference beside the server's engine: the real model's GPU and pinned RAM hold one) | the server check above |
| `pod/pod-session.sh`, `pod/pod-steps.sh` | the session (offers, watchdog, SSH, probe, detached run, polls, pull, delete, the binaries' staging, the verdict) and the step runner (with the GPU's and the cgroup's memory in the monitor) moved out of `devgpu.sh` and `devgpu-run.sh`, which source them; `runpod_api.py --network-volume` | devgpu.sh, devgpu-run.sh |
| `pod/model-copy.sh` | t04a.sh's 28 dd streams, moved out of it | t04a.sh |
| `pod/e07_gate.py`, `pod/e07_prompts.py`, `pod/e07_bench.py`, `pod/e07_report.py`, `pod/e07_chats.json` | the placement and memory gate; the speed prompts (exact ids from the prefill corpus by the server's tokenizer: 128, 16,384, 512, 65,536, 261,664); the benches through the engine's stdin protocol (TTFT, decode tok/s, RESUME, storage bytes read and major faults per phase, VRAM, RSS, cgroup); the session sheet | `hot_set.py`, `prefill_prompts.py`, Strata's serve protocol |

Measured on the dev VM (2026-10-08, no GPU; `build.sh` both flavors, `verify.sh` both flavors 16m42s): every check passes. The split tiny-b (3 shards) plans as the whole file,
free and forced; the 7 sparse shards of the real headers are 246.35 GiB at the measured sizes; `dsv41-engine
--dry-run` on that layout with the A100's placement (slab 57.75 GiB: 4840 experts, 83.4% in-sample decode hit rate;
warm 10520 experts, 125.65 GiB; cold 0) reads and checks the oracle's 2600 router and 520 index selections in 65
calls, both reader modes; the gate needs 197.73 GiB of a 250 GB cgroup on the A100 and 232.78 of 232.83 GiB on the
L40S; `e07_bench.py` on the dry engine resumes turn 2 at 16,415 ids and generates 481 tokens after the 261,664-id
prompt at ctx 262,144; the plan is 55.4 min of the 110 min cap.

RTX PRO 6000 first: `GPU_TYPES` defaults to the RTX PRO
6000 Blackwell Server and Workstation Editions, then the two A100 80GB variants, at `MAX_USD_PER_HR` 2.30 (the 3.00
USD session cap leaves 78 min at that price). `e07.sh` builds one payload per flavor the list needs and uploads the
created pod's (`pod-session.sh`'s `stages` and `pod_tag`): cu130 sm_120 for the RTX PRO 6000 (its offers ask for
hosts running CUDA 13.0, driver >= 580.65), cu128 for the A100. The cu130 tarballs now bundle `libcudart.so.13`,
`libcublas.so.13` and `libcublasLt.so.13` as the cu128 ones bundle CUDA 12.8's (the pod image has CUDA 12.4); the
payloads are 591 MB (cu130) and 797 MB (cu128). `dsv41_cuda_probe` links cudart statically and its cu130 build holds
sm_120 code (cuobjdump); in a bare ubuntu:22.04 it starts and reports the stub driver. Dev VM, 2026-10-08:
`build.sh` both flavors 3m26s, `verify.sh` both flavors 17m56s, exit 0
(both flavors' binaries run in the bare image); the RTX PRO 6000's gate at 97887 MiB: slab
73.25 GiB, 6139 experts, 89.6% in-sample decode hit rate, warm 9221 experts 110.13 GiB, needs 182.22 GiB of 232.83;
the dry engine runs on that placement; the plan is 55.4 min of the 78 min cap, about 2.12 USD at 2.30 USD/h.


Any data center, the model from Hugging Face (the volume's data center may have no A100 or RTX PRO 6000 stock): per
GPU, `e07.sh` tries the volume's data center (`VOLUME_DC`) with `NETWORK_VOLUME` first when one is set, then any data
center without a network volume, SECURE then COMMUNITY (`pod-session.sh`'s `offers`; `MAX_USD_PER_HR` 2.45, the 3.00 USD cap leaves
73 min at that price). Without the volume the pod fetches the 7 shards' heads by HTTP range (`pod/hf_model.py
headers`, then `gguf_layout.py sparse`) for the gate, then `pod/hf-download.sh` downloads
vcruz305/DeepSeek-V4.1-Flash-GGUF at c4a085541cb5 (the 7 Q2_K shards, 264,515,279,456 bytes, the names and sizes a
volume copy holds; their heads equal `$HEADERS` byte for byte) with aria2c, every file at once with 16 range connections
each, then checks each file's size and LFS sha256 against the manifest made from the HF API's tree
(`$BUILD_ROOT/e07/hf-manifest-c4a08554.json`, in the payload); the download's seconds and GB/s go into
`out/model-download.json` and the session sheet. `HF_TOKEN` is optional (the repo is public): sent to the pod on SSH
stdin into `/root/.hf-token`, then only into aria2c's mode 600 input file. Dev VM, 2026-10-08: the headers from HF
(26 s) build the sparse layout and the gate on a container without a volume; `hf-download.sh` downloads and checks
two small files of the repo in ubuntu:22.04, a wrong sha256 fails it; `verify.sh` both flavors 18m10s,
exit 0; the dry run's plans are 55.4 min with the volume and 72.4 min
with the download (21 min estimated at 0.25 GB/s, unmeasured) of the 73 min cap at 2.45 USD/h, about 2.26 and 2.96
USD (the A100 at 1.79 USD/h gets a 100 min cap).

### Where the real-model gap enters

Session 1 (2026-10-08, A100-SXM4-80GB, 2.14 USD) ran
the real model end to end with good answers, but with the oracle's selections forced the MoE input `cur` is already
1.06% rel L2 off at layer 0 (worst call, the 31-token prompt: 1.73%) and about 6% by layer 39, against the 2e-2 bound.
The cumulative comparison cannot tell an engine-numerics gap from a real-model bug; session 2 adds the isolation:

| Piece | What |
|---|---|
| `dsv41-engine --forced-moe-input` | with `--force-selection`: every layer's MoE input h replaced by the reference's `ffn_norm` (`SelectionDump::moe_input`, ForwardConfig::forced_moe_input) right after the FFN's collapse and norm; the router, shared expert and routed experts then read the oracle's cur, so the dumps' `ffn_moe_out` and weights measure the MoE path alone; decode steps launch every op (no token graph) and the hits and the shared expert quantize the replaced h themselves. `oracle_parity.py forced` copies the oracle's cur files as `ffn_norm`; `oracle_parity.py` reports such a run as isolated (its cur must be the oracle's, rel L2 <= 1e-6), with the router's own top-k on the oracle's cur from its biased probs (flips and margins reported): the per-layer isolated rel L2 |
| `dsv41-engine --prefill-mmq` | ForwardConfig::prefill = QPrefill::Mmq: the prefill matmuls of the attention and the dense ops on the fork's MMQ (int8 activations, the oracle's own GPU arithmetic) instead of the dequant path's f16 activations |
| `dsv41-layer-ref` | `tools/dsv41/engine/dsv41-layer-ref.cpp`, CPU only: the prompt through layers 0 and 1 (the engram before layer 1) on the CPU reference, every op dumped under the fork's op-dump names, in three numerics: fork-cpu (the fork's CPU backend, bit for bit), fork-cuda (the oracle's matmul arithmetic: MMQ and MMVQ int8 activations, the oracle's CPU experts), engine (the engine's GPU numerics); with `--oracle` the chain's experts take the oracle's ids and each layer also gets the isolated MoE (the oracle's cur, ids and weights through the numerics' router and experts). The oracle ran its dense part on CUDA (SPEC.md "Real-model oracle mapping"), so fork-cpu is not expected at ~0 from it |
| `layer_parity.py` | `compare`: the numerics against the oracle (cur, out, weights, own top-k, isolated MoE), each engine dump's call 0 against them (the GPU against its references on the real weights), every op between each pair of numerics with the first op at or above 1e-3 and the largest before layer 0's cur, and two readings: MMQ worth measuring (fork-cuda's layer-0 cur at most half the engine's) and a GPU suspect (its layer-0 cur at least half as far from its engine-numerics reference as from the oracle); `ops`: a fork-cpu run against the fork op dump it came from, bit for bit |
| serve's decode line, `times=1` | before each DONE the engine writes the request's decode experts to stderr (hits, misses, hit rate, the CPU's jobs, MiB, ms and GB/s, the doorbell waits, the plan, ms per step); the GEN key `times=1` times the request's decode per stage (every op launched) and prints `print_times`' split |
| `e07_bench.py`, `e07_report.py` | the decode line parsed into every phase, the phase `short-timed` (65 tokens with `times=1`), the session sheet's decode table, the isolated and MMQ parity columns and the layer-0 isolation section (`out/layer-parity.json`); a session 2 sheet is `<date>-e07a-cycle2`, a second sheet of the same day and plan takes its start time |
| `pod/e07-run.sh`, `pod/e07.sh` | after the correctness runs: `layer-ref-<numerics>` (3), `layer-parity` (with the dense-forced run's dumps), `moe-isolated` and its parity, then `dense-forced-mmq`, its parity and `layer-parity-mmq` only when layer-parity's reading asks for it (out/plan.log); bench-a with `short-timed`. `E07_PLAN=cycle2` (the default) skips tokenizer, serve and bench-b (measured in session 1), `E07_PLAN=full` keeps them; `E07_READERS` ("dense reuse" or "dense") picks the correctness runs' reader modes. The step estimates are session 1's measured times |

Measured on the dev VM (2026-10-08, no GPU): `dsv41-layer-ref` on tiny-b's 64-token dump-b-1 prompt, layers 0 and 1
(window-only with the engram before layer 1, as the real model's), against the fork's CPU op dump as the oracle
(`layer_parity.py`, 0.4 s fork-cpu, 0.3 s fork-cuda, 3.7 s engine numerics on the builder image):

| numerics | layer 0 cur | layer 0 out | layer 1 cur | layer 1 out | isolated out (0 / 1) | own top-k (0 / 1, of 64) |
|---|---|---|---|---|---|---|
| fork-cpu | 0 | 0 | 0 | 0 | 0 / 0 | 64 / 64 |
| fork-cuda | 0.00859 | 0.0154 | 0.0174 | 0.0324 | 0 / 0 | 62 / 59 |
| engine | 0.00508 | 0.0161 | 0.0152 | 0.0266 | 0.0156 / 0.018 | 62 / 62 |

fork-cpu gives all 69 dumps bit for bit (`layer_parity.py ops`: every op of both layers, the engram, the router, the
shared and routed experts, the isolated MoE). Against the fork CPU the first op past 1e-3 is layer 0's `hc_mixes`
(0.0049 engine numerics: the hyper-connection matmul's activations), the largest before layer 0's cur `attn_out`
(0.0134): on tiny random weights the int8-per-256 against f16 (or int8-per-32) activations alone move layer 0's cur by
0.5-0.9%, the order of the real model's 1.06% (its 31-token call 1.73%). Which numerics the real oracle is closest to,
and whether the GPU follows its own reference on the real weights, is what session 2 measures.

Dev VM, 2026-10-08: `build.sh` both flavors 3m31s, `verify.sh` both flavors
20m05s, exit 0: the checks above, `dsv41-engine --dry-run` with `--forced-moe-input` reading
and checking all 2600 MoE inputs of the oracle's 65 calls on the real layout, `dsv41-layer-ref --dry-run` in each
numerics on it, and the plans (session 1's measured step times; the MMQ run counted): `E07_PLAN=cycle2` 69.6 min with the HF
download, 55.6 with the volume, 57.9 and 43.9 with `E07_READERS=dense`, of the 73 min cap at 2.45 USD/h (the A100 at
1.79 USD/h: 2.08, 1.66, 1.73 and 1.31 USD); `E07_PLAN=full` is over the cap (93.7 and 79.7 min).

Session 2 ran with dense readers only, the MMQ run allowed, at most 1.95 USD/h (so the RTX PRO 6000's 2.09-2.19 list
price is skipped and the A100 is taken):

    export RUNPOD_API_KEY=...
    E07_PLAN=cycle2 E07_READERS=dense MAX_USD_PER_HR=1.95 bash tools/dsv41/pod/e07.sh

Its dry run (`E07_PLAN=cycle2 E07_READERS=dense MAX_USD_PER_HR=1.95 bash tools/dsv41/pod/e07.sh --dry-run`, dev VM,
2026-10-08, exit 0): the cap at 1.95 USD/h is 92 min (min(110, 3.00 / 1.95 x 60)); the plans are 43.9 min with the
volume and 57.9 min with the HF download, the MMQ run counted, about 1.43 and 1.88 USD at the limit price (1.31 and
1.73 USD at the A100's 1.79).

## The per-request hot set and the decode HC step

Real-model session 2 (A100-SXM4 80GB, 250 GB RAM, EPYC 7742, the real Q2_K) decoded at 20.2 tok/s short (target 30): the static hot set, packed from the hybrid's counts on one prompt, held only
34-44% of a request's expert uses, so the CPU computed ~60% of them (24-28 ms of a ~50 ms token), and the
hyper-connections and norms took 9.3 ms (0.23 ms per layer, timed). The per-request hot set makes the VRAM slab follow each
request (the Strata method: the prompt's own experts, then a decayed LFU over the decode,
swapped in from pinned RAM between tokens) and shortens the HC step:

| Piece | What |
|---|---|
| `hot_policy.hpp` / `hot_policy.cpp` | the planner, host only: per layer the slab's slots, the pinned homes and the counts; policies static (the hot set's experts), prompt (the prompt's N most used experts, ties by the hot set's order) and adaptive (prompt's target, then an LFU with half-life 128 decode tokens and a 1.5 x + 1 hysteresis); at most `--hot-swaps` (2) swaps per layer per token, each into one of `--hot-spare` (2) free slots; a resumed request (the serve loop's RESUME without `hot_reset=1`: a conversation's next turn) keeps the counts the last request left and adds its suffix's, its target the slab plus the swaps adaptive's margin allows at once (prompt: the N most counted, the slab's experts first among equal counts); a fresh request starts from zero counts |
| swaps (`experts.cpp`) | make-before-break: the slab's victims have no RAM copy (the warm tier holds the others), so the promoted expert is copied from its pinned home (or a pinned mapping) into a free slot on the copy stream and the victim's bytes into a spare pinned home on a second stream; a batch's copies are queued right after a token's graph launch (`decode_started`) and run beside it; at the boundary after the token an event query (never a wait) publishes the batch if both its events are done, or leaves it open (`deferred`): the host's tier map, the residency row mirror (mapped pinned memory) and its upload by a kernel on the engine's stream before the next token's graph (the graph reads the rows through a pointer). One batch at a time; a prefill and a request's start publish everything first; `hot_reset=1` restores the hot set's slab in rounds of the spare slots |
| copy only (`experts.cpp`, `HotConfig::copy_only`) | the smoke's copy-cost run: batches planned and copied as above, then dropped at the boundary that finds them done (`HotLayerPlan::abandon`: the slot and the home free again, nothing published); a victim without a home (a pinned mapping) goes down into one of `spare` scratch homes per layer, made at the first copy-only config; counted as `swaps_dropped`, their bytes and copy ms with the others |
| `--hot-check` | the host waits for each token's batch (every swap published at the next token), then checks every layer's device residency row, the host's slot and tier maps and the planner's invariants, and every filled slot and home byte for byte against the model's mapping |
| `--route-trace F` | every call's routed ids per layer (dsv41-route-trace/1: R, P and D lines, the R line with `resume=`, absent read as 0; every layer rings the doorbell for its ids); `tools/dsv41/engine/hot_policy_sim.py` replays static, prompt and adaptive on it with the planner's rules, and `dsv41_engine_hot_policy_test --replay` gives the same hits and swaps per layer (verify.sh diffs the two) |
| serve protocol | GEN keys `hot=static|prompt|adaptive` (the request's policy; default `--hot-policy`, adaptive), `hot_reset=1`, `trace=0` (the request left out of the trace), `prefill_times=1` (the prompt timed per stage: the `prefill split` line, front attention and dense ops, the experts' compute, copies and how much of them the passes hid, the idle gaps, the host's parts); each request's decode line adds `decode hot set <policy>: swaps, MiB, copy ms, batches, deferred, checks; hit rate per layer; reset before it` |
| HC step (`hc.cu`) | one launch per sublayer step of 7 blocks: six row blocks (the stream, its rms and q8_1, four hc_fn rows each), and a norm block that computes the collapse, the norm and its q8_1 at once beside them (it needs pre_c and the expanded stream, not the step's mixes; it was the last block's serial tail); the last block to take the ticket only writes the stream back and runs the coefficients' Sinkhorn on 16 lanes of a warp (each sum gathered by shuffles in hc_coeffs_one's order). The dims are template arguments (n_embd 5120, 256, 512: the loops fully unrolled with no exit, so a thread's loads and q8_1 shuffles issue together); `collapse_norm_kernel` sums on 512 threads, the norm block's order, so every value stays the unfused ops' bit for bit. `Forward::profile_decode_hc` times each layer's two steps from graph replays (the token graph's cost, no launch gaps), then each step alone with %globaltimer marks at its phase ends (`hc_step_split`) |
| decode transfers (`forward.cpp`, `experts_kernels.cu`) | a decode token's inputs (token, position, engram rows, synthetic or forced router ids and index rows) and its logits cross through mapped pinned memory, moved by a kernel (`copy_from_mapped`, `copy_to_mapped`), and so do the residency rows: the first verification found every token waiting for the whole open batch of swap copies, because its cudaMemcpyAsync calls ran on the copy engines after the batch's copies queued before them in the same direction |
| `dsv41_engine_smoke` | the HC per layer from graphs and the steps' phase split, then three requests on the adaptive hot set, each from the static slab, against the static pool-spin phase (wall ms per token, both hit rates, and where a token's time goes: the graph on the device, the host's launch, swap queueing, doorbells, stream end and boundary): copy only with a swap whenever a layer can start one (margin 0:-1; `HotConfig::copy_only`: every swap's copies made, the promoted expert up and the victim down, this pinned synthetic model's victims into scratch homes as the real model's go into theirs, none published, so the slab and the hit rate stay the static phase's; the run fails if a swap is published, no batch runs or a slab moves), copy only at the default margin 1.5 x + 1 (reference), then the adaptive request with its swaps published (comparison only); the targets: HC <= 0.10 ms per layer, the first copy-only run's token <= 1.1x the static one, with deferred boundaries whenever its batches outlast a token; the earlier phases keep the hot set static. Its targets are PASS/FAIL lines, never its exit code (0 unless a call fails, a logit is not finite or a copy-only check fails); `devgpu-run.sh` writes the same policy as a `note:` line in `steps.log` |
| `pod/e07_gate.py` | the hot set packed beside the spare slots (`--hot-spare 2`: 0.95 GiB of the A100's 57.75 GiB slab, 4756 experts instead of 4840) and the spare homes counted in the pinned RAM (199.7 of 232.8 GiB with the margin: PASS) |
| `E07_PLAN=e08` | `pod/e07.sh`, `pod/e07-run.sh`: the gate, the model, the dry run, bench-e08 (one engine load: short-static, short-prompt, short-adaptive with 480 decode steps each from the hot set's slab; 16k-e08 with the prompt's split and 480 steps after it; short-timed; 64k and 256k with the default policy, trace=0), the trace gzipped for the pull, and serve_check's 3 chats; `pod/e07_report.py` writes `<date>-e08` with each phase's hot set, the hit rate per layer per policy and the gate (best policy >= 30 tok/s), and each hot set phase's first token at the client with and without the bench's `hot_reset=1` (the reset runs before the prompt; a server's request pays none) |
| `pod/devgpu-run.sh` | per tiny model the split placement forced without swaps (`--hot-policy static`) and with a swap whenever a layer can start one (`--hot-margin 0:-1 --hot-check`): a check after every token, >= 2 swaps per token, against the fork, and against the static run the logits within 5e-2 and the same greedy ids |

Measured on the dev VM (2026-10-09, no GPU), `hot_policy_sim.py` with every batch published at the next token: on
its synthetic skewed trace (4 layers x 64 experts, a 16-expert slab, 3 requests of 256 prompt tokens and 480 decode
steps whose routing follows a per-request reshuffle of the profile) the decode pairs in VRAM are static 0.5125,
prompt 0.6913, adaptive 0.6914; on the real-model oracle's routing (31 prompt ids, 64 decode steps, 40 layers) with the
A100 gate's hot set (4756 experts beside the spare slots), static 0.470 (the real session measured 0.34-0.44 on its
prompts), prompt 0.768, adaptive 0.859.
There the prompt's target differs from the static slab by 19-52 experts per layer (mean 37.4), so prompt and adaptive
reach it over the first ~20-40 decode tokens at the swap rate.

A conversation's second turn (`hot_policy_sim.py two-turn`, dev VM, 2026-10-09: 4 layers x 384 experts
with 119 in each slab, 6 used, turn 1 a 1024-token prompt and 256 decode tokens, turn 2 a 50-token suffix and 256
decode tokens, one topic): resumed, turn 2 hits 0.8299 of its pairs with 65 swaps; fresh (the planner before the
resume rule: the target from the suffix alone) 0.7974 with 399; turn 1's decode run on over the suffix 0.8293 with 83.
With exact ties only (the slab's experts first among equal counts, no margin) the resumed turn made 106 swaps: the
members adaptive's margin had kept left again. On the 2026-10-09 bench (A100-SXM4-80GB), short-adaptive's first token took 2697 ms
at the client, 1923 ms of it the `hot_reset=1` rounds (3119 swaps), so 774 ms without (the engine's prompt 773 ms).

`E07_PLAN=e08 MAX_USD_PER_HR=1.95 bash tools/dsv41/pod/e07.sh --dry-run` (dev VM, 2026-10-09, estimates from session
1's measured steps): 45.7 min with the HF download, about 1.49 USD at the limit price; 31.7 min (1.03 USD) with the
volume. The session:

    export RUNPOD_API_KEY=...
    E07_PLAN=e08 MAX_USD_PER_HR=1.95 bash tools/dsv41/pod/e07.sh

Dev VM, 2026-10-09: `build.sh` both flavors exit 0 with no warning; `verify.sh` both flavors 21m10s, exit 0, with the
hot set checks (the planner's test, the simulator's self-test and its replay equal to the planner's on five knob settings
of the synthetic and the oracle traces, the forced-swap runs' dry runs, the serve keys, the e08 bench and sheet, the
e08 plan under 3 USD).

The first verification (dev pod A100-SXM4-80GB, 250 GB RAM, 16 vCPU, 0.289 USD): the swaps
correct (16 greedy ids equal, logits within 0.022 of the static run, 128 swaps checked), but the copy-only token took
2.917x the static one on the 8-layer x 384 smoke (20.43 against 7.00 ms, 403.2 MiB of copies per token at 29.1 GB/s)
and 2.695x on the 40-layer one, every token its static time plus its batch's bytes at 29 GB/s, no boundary finding a
batch still copying: the token graph's first nodes were cudaMemcpyAsync calls (token, position, engram rows, the
synthetic router ids) and its logits a pageable copy, which the copy engines ran after the batch's copies queued before
them in the same direction. HC and norms 0.146 ms per layer (graph) against 0.10. The fixes (dev VM only, no GPU run):
the transfers above through mapped memory, the batch queued after the graph's launch, the HC step's norm block and
unrolled dims, the smoke's context (its 16K runs died past `--ctx` in the second request: their exit 1, not the
FAIL lines).

## The 16K prompt's attention, expert compute and idle gaps

The hot set's real-model session (A100-SXM4 80GB, a 16,384-token prompt in one chunk, timed) read it in 8955 ms
against the 5 s target: front attention 2947 (74 ms per layer), dense 859 (hc and norms
457), routed experts 5129 (device compute 3351, plan gaps 600, stalls 1099; 123.3 GB of copies in 4891 ms, 3793
hidden). The same period's dev pod attention bench (same GPU, a 4096-token chunk at 16K) split a Reuse layer's 18.9 ms into
projections 2.8, attention kernel 12.7 and output 3.4: the kernel ran its MMAs at ~41 TFLOP/s with the hi/lo split.
The changes, measured on the first dev pod (the follow-up below):

| Piece | What |
|---|---|
| `prefill_attn_kernel` (`attention.cu`) | a prefill chunk's attention: the same block (a token, 32 heads, a split of its cells, 8 warps, chunks of 32 cells) and online softmax as `attn_kernel`, with the K rows by cp.async (zero-filled for a skipped cell), q.k on tensor cores with p rounded once to f16 (no hi/lo split; p.v's MMAs halve. One f16 rounding is up to 2^-11 per cell, the GPU test's whole tolerance of 2^-11 of sum p \|v\| in the worst case: it holds by the errors' independence across cells and the running maximum's p being exactly 1, not by a worst-case margin, and `dsv41_engine_attention_test --gpu` ends with an "attention margin" line, this kernel's worst ratio against the tolerance, measured on the dev pod) with p.v's A by ldmatrix and B (V = K) by ldmatrix.trans, `__launch_bounds__(256, 2)`, and with one split the de-rope written by its epilogue (the same f16 as `rope_kernel`'s for every finite value). Decode keeps `attn_kernel` and its split. Its later form is below |
| `rope_kernel` in place | only the rotated last n_rot values of each head are read and written (q's rope at 16K moved 4 GiB for the 1/8 that rotates) |
| `AttnOp`, `AttnTimes::op_ms` | a prefill call's split by op (q_a, q_b, q rope, kv, the indexer's query and weights, the compressor's projections, pooling, index keys, the selection, the attention, the de-rope, wo_a, wo_b); `dsv41_engine_attention_bench` prints it per role and the real 40 layers' mean per layer (each layer its role's row); at a 16384-token chunk its target line (<= 30 ms per layer) |
| `decode8_f16x2` (`qdequant.cuh`) | the grouped GEMM's q2_K and q3_K weights from a per-group table: the 4 or 8 values a 16-value group can take, computed as `decode8` does and rounded to f16, each pair picked by `prmt` from the 2-bit codes (q3_K: the hmask bit selects the table's half through a sign-replicated mask). `dsv41_engine_qdequant_test` (host, a prmt emulation) checks it against `qref_dequant_row` rounded to f16 on 4096 random blocks of each type: 0 differ |
| `ExpertEngine::prefill_fetch` | the routing goes to the host right after the router, before the shared expert, and `forward` waits on its event: the host's plan runs while the device computes the shared expert; the plan gap is measured from the shared expert's end |
| uploads from mapped memory | a layer's pairs, rows and tiles (`h_perm`, `h_row`, `h_tiles` mapped pinned) go up by one `copy_from_mapped` launch: the three cudaMemcpyAsync queued behind the ring's and the prefetch's copies on the copy engine (the same lesson as the decode transfers above) |
| prefetch wait per pass | only the first slab pass that reads the prefetch region waits for it, so a larger `--prefetch-experts` (the next layer's experts copied during its front, when the copy engine was otherwise idle) does not hold the slab's own passes |
| `hc_prefill_step` (`hc.cu`) | a prefill chunk's HC step: a block per token row does the expansion, the stream's rms (hc_norm_kernel's order) and the collapse with its norm (collapse_norm_kernel's order) from one read of x, then the hc_fn matmul and the coefficients; each layer's last expansion is deferred into the next layer's attention step (not before an engram layer, not with dumps). `dense_test_gpu` checks it against the unfused ops bit for bit at T > 1 |
| `dsv41_engine_smoke` | one line of the targets on the real 40 layers: attention per layer by role (<= 30 ms at 16K), routed-expert compute (<= 2.4 s) and plan gaps + stalls (<= 0.6 s), scaled by 40 / --layers; `devgpu-run.sh` adds `attention-bench-16k` and `smoke-e09` (smoke-e05c with `--prefetch-experts 128`) |
| `E07_PLAN=e09` | `pod/e07.sh`, `pod/e07-run.sh`: the gate with a prefetch region of 128 experts (`e07_gate.py --prefetch-experts`: 21.37 GiB beside the experts, a 56.5 GiB slab of 4651 experts on the A100 instead of 4756), the model from HF, the dry run, bench-e09 (16k-e09 twice with `prefill_times=1`, short-adaptive's 480 steps between, so the second reads cold; the 256K prompt once) and `engine/two_turn_check.py` (a two-turn chat through `serve/server.py` with `--route-trace`: turn 2's cached tokens, its R line `resume=1`, its decode); session cap 1.50 USD, the HF plan 36.6 min, about 1.19 USD at 1.95 USD/h; `pod/e07_report.py` writes `<date>-e09` |

Expected, not measured: with the hot set session's per-layer numbers, the copy engine was busy ~122 ms of each layer's
~224 ms, idle during the next layer's front once the 32-expert prefetch (~16 ms of copies) was done; 123 GB at the
measured 25.2 GB/s is 4.9 s on the A100's PCIe 4 link, so the 16K TTFT there cannot fall below about 5 s unless the
copies overlap the fronts as well as the passes, which is what the larger prefetch is for.

### The attention's fixed cost, the selection, the q3_K down kernel's loads

The first dev pod run (A100-SXM4, `attention-bench-16k`, a 16384-token chunk at positions 0..16383) measured 45.8 ms
per layer against 30: attend 25.5-26.2 ms in the Full, Reuse and Reindex layers but 15.5 in a window layer, which
attends a fifth of the cells (128 against 640 per query: a fixed cost per block about as large as the chunks' work),
select 7.6 ms (ratio 2) and 12.9 (ratio 1); and the smoke's routed-expert compute 3.1-3.3 s against 2.4, down at 67.5
TFLOP/s against gate/up's 102.6. The SASS of the three kernels (sm_80, CUDA 12.8, dev VM) showed the causes; changed,
not yet measured on a GPU:

| Piece | Cause in the SASS | Now |
|---|---|---|
| `prefill_attn_kernel` | q loaded by a loop of scalar loads, each one's branchy `f16_from_f32` waiting on its load before the next was issued; the epilogue inlined 32 copies of `rope_cs` with `sincosf` (24,600 SASS lines); the f32 output written beside the de-rope (2 GiB per 16K layer) | q by 16-byte loads, four in flight, the hardware's round to nearest even, staged in the K buffer and held as A fragments (each warp 16 heads over a quarter of D, the quarters' sums meeting in the softmax: q.k reads only K from shared memory); the next chunk's top-k ids loaded at the chunk's top; the de-rope's cos and sin once per block (`rope_cs`, the same bits); the f32 output only with `AttnEngine::set_attn_output(true)` (the GPU test). 5,237 SASS lines; ptxas -v: 128 registers at D 512 and 92 at D 256 on sm_80 (no spill), 128 on sm_89 (an 8-byte spill) and on sm_120 (none); 54.4 KiB of shared memory at D 512: two blocks per A100 SM (one on sm_89 and sm_120, as before) |
| `scores_tc_kernel` (`indexer.cu`) | the A fragments from 128 scalar loads per lane, each behind the previous one's software conversion; one key tile copied synchronously between two barriers | float2 loads all in flight, the hardware's conversion; the key tiles double-buffered by cp.async with one barrier per tile; B by ldmatrix. The same products and sums in the same order: the same scores |
| `topk_kernel` | every pass read the row's scores from memory again; thread 0 walked the 256 bins one by one, four times | up to 16,384 visible rows (a 16K prompt at ratio 1) kept in registers, read once (`topk_kernel<true>`, 55 registers on sm_80: one CTA per SM); each pass's digit by warp 0's scan of the bins (a prefix over the lanes, then one lane's eight): the same digit and count, so the same ids |
| `grouped_kernel` (`experts_gemm.cu`) | 22 loads per thread per k step in the q3_K down kernel, 18 of them 2 bytes wide (110-byte blocks sit at 2-byte alignment), each warp's load touching 8 rows' lines: ~1,700 L1 wavefronts per block and step against 512 cycles of MMAs (q2_K gate/up: 8 loads of 4 bytes) | each k-block (256 values) of the block's 128 weight rows staged once by cp.async in 16-byte pieces from the block's address rounded down to 16 (8 threads per row, one line each), into 144-byte shared rows; `fetch8` and `decode8_f16x2` read the same bytes from there (the same values; `experts_test --gemm`'s one-hot rows check them bit for bit). 48 KiB of shared memory (activations double, weights single with two barriers per step, the raw k-block 18 KiB): two blocks per SM on sm_80, sm_89 and sm_120; 127-128 registers, no spill on sm_80 and sm_89 (a 4-byte one in sm_120's q3_K kernel). Parts must start on 16-byte boundaries (the engine's slots are 256-aligned; `experts.cpp` checks, the kernel traps). q6_K experts (none in the real or tiny models) are refused by `expert_gemm_check` |

`dsv41_engine_attention_bench --peak-tflops F` (devgpu-run.sh passes the GPU's published dense FP16 tensor TFLOPS)
adds each prefill table's attend roofline per role (queries x heads x keys attended x D, 4 FLOPs per product, the
attend ms, TFLOP/s and its share of the peak) and the 40 layers' together, the select scorer's FLOPs per index source
with their time at the peak, and at 16K the lines `attend per layer ... (<= 12)` (the slowest role) and `select per
index source layer ... (<= 4)`. The scorer's floor at 16K on the A100 (312 TFLOP/s): about 1.1 TFLOP for a ratio-1
source (16384 queries, 8192 visible keys each on average, 32 heads of 128), 3.5 ms at the whole peak, so select <= 4
ms at ratio 1 needs the scorer above ~88% of peak with the top-k free; at ratio 2 it is half of that.

## The session on the RTX PRO 6000 and its tooling

| piece | what it does |
|---|---|
| `E07_PLAN=e07b` | `pod/e07.sh`, `pod/e07-run.sh`: the RTX PRO 6000 first, MAX_USD_PER_HR 1.95 by default, session cap 3.00 USD; the gate and every engine with the prefetch of 128 experts; after the dry run, two-turn, bench-e07b (16k-e09, turn2, short-adaptive, 64k, 256k on one load), then dense-free with its parity and the isolated MoE run (reported, not gating); `pod/e07_report.py` writes `<date>-e07b` |
| per-step timeouts | `pod/pod-steps.sh` with STEP_ESTIMATES (e07/estimates-volume or -hf, the dry run's plan in seconds): a step's timeout is min(2 x its estimate, the time left); a step killed there gets `step NAME killed: timeout after B s (...)` in its log and the same cause on its steps.log line, and the next step runs |
| HF throughput guard | `pod/hf-download.sh` with HF_GUARD_END_BY: 180 s in, the bytes on disk give the rate; when the projected end passes the deadline less the rest of the plan, the download stops (exit 3) and the session stops with `stop: download too slow: X GB/s (...)`, then pulls and deletes the pod; the plan estimates the download at HF_GBS (default 0.115 GB/s, measured on 2026-10-10) and the dry run exits 2 when that HF plan costs more than 3.00 USD |
| two-turn at 16K | `engine/two_turn_check.py --turns e07/turns.json` (`pod/e07_prompts.py turns`: the text of p16k, which re-encodes to its 16,384 ids, and a short follow-up) at ctx 20480 and `--chunk 16384`; turn 2 streamed, its first content chunk timed at the client; `--http` then posts a new conversation non-streaming and streaming on the same server load (each reply the text of the engine's `--log-ids` ids, the two replies' ids compared and reported) and checks tools are refused |
| target table | `pod/e07_report.py`: one row per target 1-6 with the number, PASS, FAIL or REPORTED and the cause of a FAIL or a missing result from out/steps.log (`timeout after N s in <step>`, a failed or skipped step, a step that never ran and the stop before it); the host table's PCIe row from env.log (the link maximum and the current generation and width, the host's and the GPU's maxima when the driver reports them) |

Checked without a pod by `pod/e07b-test.sh` (a fake step that overruns its 4 s budget, a fake aria2c below and above
the guard's rate, the report on a fixture session) and by verify.sh's e07b block.

## The quality tooling

The method of an earlier vLLM-EXL3 host-offload experiment's quality harness, copied into `tools/dsv41/quality/`
(`PROVENANCE.md`): lm-eval 0.4.13, the fixed-seed subsets (GSM8K CoT 150, IFEval 150, HumanEval instruct 164),
greedy, thinking off, `--log_samples`, every request through `api_proxy.py`.

| piece | what it does |
|---|---|
| proxy backends | `api_proxy.py --backend llamacpp` (the fork's llama-server) and `--backend strata` (serve/server.py): thinking off through chat_template_kwargs, the messages as lm-eval sent them; llamacpp cuts the continued message's text off the start of the fork's reply (below); `deepseek` and `vllm` unchanged, `--key-file` for the DeepSeek key |
| continuation | `serve/server.py`: a chat request whose last message is the assistant's is continued on the deepseek-v4.1 profile (the template without the generation prompt, the end-of-sentence text that closes the message cut off); requests that do not end with the assistant's message render as before, and the Qwen profile keeps its behaviour. llama-server does the same with `--prefill-assistant`, to the byte (below) |
| eval.sh | one system through the proxy: the lm-eval tasks (32, 4 or 2 concurrent for deepseek, llamacpp, strata), `continuation` (3 HumanEval replies for the hand check) and `needle-<length>` |
| needle | `needle_check.py`: passkey prompts (seed 20261011, depth 0.5) of 16,317, 65,469 and 262,077 tokens with the GGUF tokenizer (the 16K, 64K and 256K contexts less 64), exact-match scoring |
| analyze.py | `--fork` takes the directory that holds each qfork session's `out/quality/` (each RUN is searched recursively: `quality.find_results` and `load_samples` use rglob; every task from one session only). The quality table: API fresh, fork, engine, API 2026-10-06, EXL3 2026-10-06; the paired differences engine - fork, fork - API, engine - API, API fresh - API then (95% bootstrap, flips) and the quality verdict; on the two vllm-exl3 runs it gives that experiment's published table to the third decimal |
| api_run.sh | the fresh API run from the dev VM: the key from DEEPSEEK_KEY_FILE (0600), the price page fetched, an upper-bound estimate before anything is sent (0.266 USD at peak prices for the 2026-10-06 tokens plus the needles), the tasks and needles, `api_cost.py` per request at its hour's price, the sheet `<date>-qapi`, a scan of every output for the key |
| `E07_PLAN=qfork`, `qengine` | `pod/e07.sh`, `pod/e07-run.sh`: the fork's llama-server (A100, 1.95 USD/h, no gate) or the engine's server (RTX PRO 6000, 2.60 USD/h), `quality/pod_server.sh` in its own session across the steps, the lm-eval venv installed in the background while the model downloads and loads, the continuation checked before the full run, then each task of Q_TASKS and each needle of Q_NEEDLES; every step's estimate from the 2026-10-06 tokens at Q_PROMPT_TOK_S, Q_LONG_PROMPT_TOK_S and Q_DECODE_TOK_S (`quality/q_plan.py`), the dry run exits 2 when a model path the offers take costs more than 3.00 USD (the over-cap line names each path once with its reasons); qfork samples the GPU's memory used every 1 s while the fork server runs (`out/quality-gpu.csv`, nvidia-smi) and the sheet gives its peak and the free memory at it (the headroom of `--ctx-size 32768 -np 4` on the A100 80GB, not measured yet); `pod/e07_report.py` writes `<date>-qfork` or `-qengine` |

Checked without a pod by `quality/q01-test.sh` (fakes.py for the engine, llama-server and DeepSeek's API, and the
real fork's CPU build below).

### The real llama-server's prompt and reply

A review read `common/chat-auto-parser-generator.cpp:54-66` of the fork as adding `<think></think>` after the
generation prompt's `</think>` when it continues a message, which would make the fork's HumanEval prompt differ from
Strata's. `quality/fork_prompt_test.py` (q01-test.sh section 8) checks it on the real llama-server: a CPU-only build of
`$FORK_DIR` at b4d9cb1 (`feat/strata-moe-hook`, LLAMA_STRATA_MOE unset: the stock path) on `tiny-a.gguf` with
the real tokenizer's first 512 tokens (`engine/tiny_vocab.py --model`, so the fork renders the real
`<｜begin▁of▁sentence｜>`), `--jinja --chat-template-file` set to the `tokenizer.chat_template` of
`deepseek41-vocab.gguf`, and the pod's chat flags (`--prefill-assistant --no-reasoning-preserve -np 4 --kv-unified`).
For each request, the body `api_proxy.py --backend llamacpp` forwards goes to the fork's `/apply-template`, and the
result is compared with the prompt ids serve/server.py (deepseek-v4.1 profile) hands its engine, decoded.

Measured on the dev VM (12 CPUs, 2026-10-11), thinking off:

| request | fork against Strata | prompt characters | fork chat route prompt_tokens (512-token vocabulary) |
|---|---|---|---|
| HumanEval/0 (continued) | byte-identical | 886 | 571 |
| HumanEval/1 (continued) | byte-identical | 1202 | 703 |
| HumanEval/2 (continued) | byte-identical | 852 | 495 |
| plain chat ("What is 6 times 7?") | byte-identical | 68 | 43 |

Each prompt from its last `<｜Assistant｜>` (the start, then the end; the same text on both sides):

    HumanEval/0: '<｜Assistant｜></think>Here is the completed function:\n```pyth' ... ' 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n\n'
    HumanEval/1: '<｜Assistant｜></think>Here is the completed function:\n```pyth' ... ')\n    [\'()\', \'(())\', \'(()())\']\n    """\n\n'
    HumanEval/2: '<｜Assistant｜></think>Here is the completed function:\n```pyth' ... '> truncate_number(3.5)\n    0.5\n    """\n\n'
    plain chat:  '<｜Assistant｜></think>'

The fork's chat route reads that same prompt (its prompt_tokens equal its `/tokenize` of the `/apply-template` text),
and with thinking on the plain prompt ends `<｜Assistant｜><think>` instead (the test's negative control). Verdict: the
prompts are the same, no `<think></think>` is added. The cause: this template defines `dsml_token`, so
`common/chat.cpp:1164-1172` hands it to the DeepSeek V3.2/V4 handler (`common/parsers/deepseek.cpp:69`), which never
reaches the auto-parser's continuation code.

What the run found instead: the fork's reply to a continued message starts with that message's text (the partial
code, then the new tokens), where Strata's server and DeepSeek's API return the new text only. lm-eval's
`build_predictions_instruct` keeps the reply up to its first code fence and puts it after the prompt, so every fork
HumanEval item would have been the signature plus "Here is the completed function:". The llamacpp backend now cuts
that text off each reply (a reply that does not start with it is a 502), `fakes.py llama-server` echoes it as the
real one does, and continuation_test.py checks the cut.

The CPU build, outside the repos (cmake and ninja from a venv, as the dev VM has no cmake; about 2 minutes on its 12
CPUs):

    B=$BUILD_ROOT/fork-cpu
    python3 -m venv $B/cmake-venv && $B/cmake-venv/bin/pip install cmake ninja
    PATH=$B/cmake-venv/bin:$PATH cmake -S $FORK_DIR -B $B/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    PATH=$B/cmake-venv/bin:$PATH cmake --build $B/build --target llama-server -j 10

q01-test.sh reads it from `FORK_CPU` (default `$B/build/bin/llama-server`) and skips section 8, saying so, without it.

## Deterministic decode

The determinism session of 2026-10-13 (A100-SXM4-80GB, the real Q2_K) found identical greedy requests apart at token 14:
the adaptive hot set's swaps were published when the copy streams finished (59 of 196 batches later than planned)
and each request kept the slab the last one left, and an expert computed in the slab (q8_1, MMVQ) and on the CPU
(Q8_K, the fork's vec_dot) differs in its last bits. [DETERMINISM.md](DETERMINISM.md) lists every source of
run-to-run variation in the decode path with its evidence (placement: yes; GPU against CPU numerics: yes, by design;
the combine's order, atomics and reductions, CPU threads: no; the prefix cache: no for identical prompts).

The deterministic mode, on by default for every request (`--deterministic off` and the GEN key `det=0` turn it off,
the server's `"strata_deterministic": false` and the config's `"deterministic": false` too): a request starts from the
hot set's slab with zero counts, and a decode boundary waits for a batch still copying instead of deferring it, so a
request's placement follows from its own prompt and tokens. The adaptive swaps stay. Each request's hot set line ends
in the mode with the boundaries that waited and their ms. Its cost is measured on a GPU
(`tools/dsv41/quality/det_sampling_check.py`: the mode on against off, prompt S, 3 runs each); on the dev VM
`dsv41_engine_experts_test --determinism` checks the mode on the CPU reference path (bit for bit with it under other
swap timings, arrival orders and thread counts, and after another request; apart without it).

## Several cached conversations

The baseline session (2026-10-13, RTX PRO 6000 Blackwell WS, the real Q2_K, 16K conversations) found no turn
reusing its prefix from two users on: the engine kept one live sequence and one prompt-end snapshot. Now it keeps N
conversation checkpoints (`dsv41-engine --serve --checkpoints N`, default 4; the server config's `"checkpoints": N`):

| Part | What | Strata's counterpart |
|---|---|---|
| `kv_checkpoint.hpp` spans | what a checkpoint at position P holds of the KV cache: every layer's window cells of the last `n_swa - 1` positions (one or two runs of the ring), each KV source's completed compressed rows and its pooling carry, each key owner's index keys; packed layer by layer into a host buffer. 55.1 MiB for the baseline's 16,413-token first turn, 805.0 MiB for a slot of 262,144 positions (the real layout) | `core/conversation_cache.hpp` `ConversationCheckpoint` |
| `KvCache` slots | N pinned host slots, each sized for `max_ctx`; a save copies the spans device to host on the stream, a restore host to device in one hop, or in place (the single snapshot's way: the carry only) when the device still holds that conversation within `max_chunk` positions; every copy timed by events | the parked conversations' host images |
| `Session` | the live state and N `ConversationSlot`s (a slot id, the ids whose end it holds: batched decode runs over them); the longest prefix resumes (`core::conversation_prefix`, ties: the live state, then the checkpoint used last); a prompt's end goes to the slot it supersedes, else a free one, else the least recently used is evicted | `ConversationCache::best`, `drop_superseded`, `make_room` |

The cost per conversation is on the engine's log: at load `conversation checkpoints: N slots of X MiB pinned host RAM
each ...` (the dry run too, from the same spans), per request `checkpoints: restored k (T tokens, X MiB host to
device, Y ms), saved to k (...), S superseded, evicted k (...)`. `tools/dsv41/pod/e07_gate.py` counts the N slots in
the host RAM a placement needs (`kv_budget.py checkpoint_bytes`, equal to the engine's). The deterministic mode's hot
set reset moves to the end of a request (its idle time, `hot_policy.hpp` `hot_request_end_resets`): the hot set line
ends in the idle reset before the request and how long the request waited for it. On the dev VM
`dsv41_engine_checkpoint_test tiny-b.gguf` checks save, restore and evict on the CPU reference against a straight read
bit for bit, and `tools/dsv41/quality/c02-test.sh` runs these checks and the conversation-cache session's rehearsal.

### Host homes and the reset layer by layer

The conversation-cache session (2026-10-14, RTX PRO 6000 Blackwell WS) missed its resumed-turn target only through
the reset: back to back, each
turn 2+ waited 1.67-1.91 s for the last reply's idle reset (2885-3311 swaps in 1750-2030 ms). The reset is PCIe-copy
bound: 24.4 MiB a swap, 12.2 up for the promoted expert and 12.2 down for the victim into a spare home, ~20.4 GB/s each
way at once, 0.6 ms a swap. `dsv41-engine --hot-homes all` removes the copy down and hides the rest:

| Part | What |
|---|---|
| host homes (`HotLayerPlan` `Layout::homed`) | every expert of the hot set's slab gets a permanent pinned home at init (after the warm tier's homes; the slab is filled from them), so every expert that can be in the slab always has a valid host copy: a swap copies up only and drops its victim, a promoted expert keeps its home. The planner's choices are the two-way path's swap for swap (`hot_policy_test`), so the slab and the tokens are the same. 72,702 MiB on the real placement (5951 experts), in place of the spare homes' 977 MiB |
| the reset in place, layer by layer (`restore_in_place`, `HotRestore`) | at a request's end in the deterministic mode the planners go to the hot set's slab at once; each layer's copies (each hot-set expert outside the slab into a non-member's slot, from its home) go on a restore stream of their own with an event after them, and `end_request` returns without waiting. The next request's prefill passes and eager decode hits of layer l wait for layer l's event only, the token graph (no event wait inside it) for the last one before its launch; nothing waits once a call has seen the last event done |
| the choice | `tools/dsv41/pod/e07_gate.py` turns the homes on (`hot_homes: all` in gate.json, e07-run.sh passes `--hot-homes`) only when the host's RAM holds them beside everything else with its 8 GiB margin; else `--hot-homes spare`, the two-way path and the reset in rounds as before, and the gate names the homes' size and the shortfall. The engine's load lines name the path (`experts: hot set homes: ...`) and the homes' RAM (`fixed host RAM beside the checkpoints: ...`, held once for the engine: not a conversation's cost, which is its checkpoint slot); with `--hot-homes spare` and the deterministic mode on, the two-way line says its idle reset is the slow one (the next request waited 1.67-1.91 s on the RTX PRO 6000 in the first conversation-cache session) and points to the gate's `hot_homes` |

Expected on the RTX PRO 6000 (host homes, not measured yet): 0.49 ms a swap up only at ~26 GB/s, ~1.41 s for 2900
swaps, a resumed turn's short prefill trailing the restore by about one layer. The hot set line's idle reset is then
the copies' ms on the device, its wait the host's plus the request's layers' waits on the device, and the next line
(`idle reset layer by layer (host homes): ...`) splits them. The gate's arithmetic on the real headers: the homes need
258.61 GiB + 8 margin, so a 282 GB host (that session pod's 262.63 GiB cgroup) is 3.97 GiB short and a 300 GB one fits.
On the dev VM: `dsv41_engine_hot_policy_test` (the swaps against the spare homes', no copy down, the in-place
restore, the restore order and its refusal of a layer computing before its restore is queued),
`dsv41_engine_experts_test --determinism` (the mode with host homes and the reset layer by layer bit for bit the
two-way runs, with an injected restore progress per request, every slot as the planner says when its layer computes,
every home intact; and without the per-layer wait the stale slots seen), and `--serve --dry-run` /
`--dry-run`'s dry reset of the placement (`dry run: hot set reset ...`). The CUDA parts (the restore stream and its
events, the per-layer stream waits, the token graph's wait, the pinned homes' allocation and fill, the overlap of the
restore's copies with the next request's) run only on a pod: the conversation-cache session is their check.

### Partial host homes

The conversation-cache pod's host (282 GB, cgroup 262.63 GiB) cannot hold a home for every slab expert with the gate's 8 GiB margin
(3.97 GiB short), and the all-or-nothing gate would have fallen back to the two-way path. `--hot-homes N` homes the
first N slab experts of a fixed home order and leaves the rest on the two-way path:

| Part | What |
|---|---|
| the home order (`experts.hpp`) | the layers that swap (and whose mapping is not pinned) in layer order, every slab expert of a layer before the next layer's; the layer the limit cuts through takes them from the end of its VRAM order (the least counted first, the planner's victims among equal counts). It depends on no request. Whole layers, not the most evicted experts across all layers: a layer with any expert left without a home recycles its promoted warm experts' homes, so it cannot restore in place |
| a layer homed in full | as with `--hot-homes all`: copies up only, the reset in place layer by layer |
| any other layer that swaps (`Layout::homed` a subset, or empty) | a homed victim is dropped, any other is copied down into a spare home (it keeps its spare homes); the planner's choices are the two-way path's (`hot_policy_test`). Its reset runs in rounds of the spare slots right after `end_request` queued the layered part, and `end_request` waits for them, so the next request finds them done |
| the choice | `e07_gate.py` gives homes to as many slab experts as the cgroup holds with the margin (not lowered), counting the spare homes the layers not homed in full keep; gate.json has the homes held, of how many, the fraction, the cut layer; e07-run.sh passes `--hot-homes N`. `e07.sh --dry-run` with `PROBE_LOG` (a past session's probe.log; pod-session.sh now records the pod's cgroup there, an older session's out/env.log is read instead) takes that pod's real cgroup and prints the expected homes |

The gate on the real headers: that pod's 262.63 GiB holds 5612 of the 5951 slab experts' homes (94.3%, 66.95 GiB:
37 of the 40 layers in full, layer 37's last 116 of 142, the 3 layers left keeping 0.07 GiB of spare homes), 254.63 GiB
+ 8.0 margin; a 256 GB host 3552 (59.7%). The dry reset of that placement (`--serve --dry-run --hot-homes 5612`, every
promotable expert displaced, the worst case): 5490 swaps in place in the 37 layers (67,070 MiB up, about 2.69 s at
26.12 GB/s) and 455 in rounds in the other 3 (455 victims copied down, about 0.29 s at 20.4 GB/s each way). The
conversation-cache session's reset (about 2900 swaps) would put about 7.5% of them in rounds, waited for on the host; not measured
yet. The load line says `partial host homes (--hot-homes N): H of the slab's V VRAM experts ...`, a request's report
adds `idle reset in rounds (partial host homes, L layers): ...` and `copies down left (host homes held H of V slab
experts): ...` (its swaps' copies down and the reset's before it). On the dev VM: `dsv41_engine_hot_policy_test`'s
partial homes (the spare homes' swaps, homed victims dropped, the reset in rounds, `restore_in_place` refusing such a
layer) and `dsv41_engine_experts_test --determinism`'s partial mode (tiny-b: 3 layers homed in full, 1 in part, 2 none;
a victim copied down writes its slot's bytes into the home a miss then reads; every request bit for bit the two-way
run's). The CUDA side of the rounds next to the restore stream (their copies sharing the PCIe link, the host's wait for
them) runs only on a pod.

## Batched decode

The engine decoded one sequence at a time (the 2026-10-13 baseline, RTX PRO 6000 Blackwell WS: 54 tok/s for
one user, 22-23 tok/s in aggregate at N = 2 to 8). `dsv41-engine --serve --batch N` (1 to 8) keeps N batch slots
beside the live sequence and decodes every active one together, one token per slot per step, in Strata's batch
protocol, which `serve/server.py`'s `StrataEngine.generate_batched` drives (the server config's `"parallel": N`, or the
server's `--parallel N`; 0 or 1 is today's FIFO):

| Part | What | Strata's counterpart |
|---|---|---|
| the slots (`batch.hpp` `BatchSlots`, host only) | a request (`BGEN <slot> <max_new> ...`) reads its prompt and first token on the live sequence as a GEN of one token, then the live state is copied into its slot (`BADM <slot> 1`); every step takes the active slots in slot order (`BT <slot> <id>`), a slot ends at an end of reply, its max_new or a `BSTOP` (`BDONE <slot> <generated> <finish> <ms> rep=0\|1`); a prompt read between steps holds them (its stall logged per request); a request that cannot go on in a slot (past `--batch-ctx`, one token, an end at once) is answered alone (`BADM <slot> 0`) | `src/program/generate.cpp`'s batch windows (`BSlot`, `batch_step`, the BGEN admission as a GEN 1) |
| a slot's KV (`kv_cache.hpp`) | a KvCache of `--batch-ctx` positions (default 32768) with a ring of n_swa cells (it only decodes), filled at the admission by `copy_state_from` (`kv_state_copies`: the window's positions cut where either ring wraps, rows, keys and carry row for row). 105.1 MiB a slot of 32768 on the real model (`kv_budget.py slot_bytes`, the engine's `kv_cache_bytes`) | the slots' sessions |
| a step (`forward.hpp` `decode_batch`) | one CUDA graph per batch size B: B = 1 is the token graph's own launches on the slot's cache; B > 1 runs every op over the B rows in one launch where its per-row values do not depend on the row count (the embedding, the engram, `hc_prefill_step` with the rows' q8_1, the router's GEMV, the shared expert, the head, qmatmul's MMVQ at B columns), `attn_decode_batch` (the projections batched, the cache's stages per row with attn_decode's own kernels), and one experts call of B tokens. A graph whose slots change is captured again and swapped into its instance (`cudaGraphExecUpdate`) | `run_slot_rows`, the per-B verify graphs |
| the experts of a step | the hits grouped by expert (`qmatmul.cu` `mmvq1_experts_grouped_kernel`: the expert's first pair, `expert_hit_leader`, computes every pair of that expert from one staged read of its rows), the misses as one CPU job per expert for all its tokens (`decode_miss_jobs`): an expert routed by several rows is read once. One hot set, its swaps on, driven by the union of the batch's routes | the slots' shared expert cache |

Batch invariance: with the same placement, a row's token is the one it gets decoding alone. Every
batched kernel computes each row's values in the one-row order (MMVQ's per-column sums, the grouped hits' per-pair
sums, the per-row norms and ropes, the per-row caches); `dsv41_engine_batch_test` checks it on the CPU reference
(below), the dev pod's `--kernels` (MMVQ at 2 to 8 columns, the grouped hits at every T of 2 to 8 tokens since each T
is its own kernel, the batched attention over 4 sequences) and `--gpu` on CUDA. With more than one slot
the placement a slot sees follows its neighbours' routes too, which at first made a batched request not reproducible
across batch compositions. With `--cpu-numerics gpu` ([DETERMINISM.md](DETERMINISM.md)) a routed expert's bits do not
depend on its tier (the CPU experts compute in the GPU's numerics), so a batched request in the deterministic mode is
its alone run's bit for bit: its BDONE says `rep=1`, the server's reply `"strata_reproducible": true`; in the mode off
`rep=0` and false. An RTX PRO 6000 session of 2026-10-17 (AMD EPYC 9335, the real Q2_K, greedy, 256 tokens) measured
it: the 9 batched requests that went into a slot at once matched their alone runs bit for bit, the 3 promoted ones
(started on the solo path, STOPped after 2 tokens, sent again as a BGEN of the prompt and the tokens) left them at
tokens 67 and 91, their last token read by `prefill(T=1)` where alone it is decoded. The fix
([DETERMINISM.md](DETERMINISM.md)): the Session feeds that token by decode (its line "its 1 token read was fed by
decode"). The same session's single user decoded 50.71 tok/s with batching off and 50.15 on, against 61.55 on an AMD
EPYC 9655 host of 2026-10-16 with the fork's CPU numerics. `--cpu-numerics fork` (the CPU experts in the fork's
numerics; no batch slot then ends `rep=1`, nor a request that ends at its admission beside decoding slots, its
`BADM <slot> 0 rep=0`) is now the default and `--cpu-numerics gpu` an explicit option: on 2026-10-19 (RTX PRO 6000
Blackwell Server Edition, AMD EPYC 9575F, the real Q2_K, the single user with the deterministic mode off, 4 turns of 256
tokens at a 1K context) the single user decoded 47.04 tok/s in the GPU's numerics against 63.34 in the fork's (-25.7%).
So under the default a batched reply is not its alone run's tokens and says `"strata_reproducible": false`, even in
the deterministic mode. A request alone takes the solo path (GEN), unchanged. A client that needs a reproducible reply
sends `"strata_reproducible": "required"` ([docs/DETAILS.md](../../../docs/DETAILS.md), Strata's own keys
`strata_*`): it waits, next in line, until no slot decodes, then decodes alone in the deterministic mode, and its reply
says `"strata_reproducible": true`.

The gate (`tools/dsv41/pod/e07_gate.py --batch N --batch-ctx C`) takes the slots' KV and the rows' logits from the
expert slab: on the RTX PRO 6000 (97887 MiB, prefetch 128) 8 slots of 32768 cost 0.82 GiB, the slab 72.0 -> 71.25 GiB,
5951 -> 5888 VRAM experts; a placement the slots leave without a hot set is refused naming them.

On the dev VM (2026-10-14, 12 vCPU, no GPU; the cu130 builder image): `dsv41_engine_batch_test tiny-b.gguf` (the CPU
reference: attention_ref, the router, the frozen placement's hits by their leader, the misses as the engine's CPU jobs,
images of the live ring and of the slots' rings read back before every step, driven by the engine's Session and
BatchSlots): 4 requests of 21 to 64 prompt tokens and 6 to 15 to generate, each bit for bit its alone run (tokens and
every step's logits), 17 steps (38 rows) of 4 rows shrinking as slots end, 101 hit pairs computed by another row's
leader and 101 CPU jobs of two rows or more (counted over the steps); a prompt read between steps; B = 1 greedy and sampled marked
reproducible; BSTOP; a checkpoint across the two rings; an admission missing a window run or the carry seen. The
`--serve --dry-run --batch 4` protocol; the rehearsal on `fakes.py --batch 8` (16K conversations, 0.3 ms a prompt
token, 5 ms a step of one row and 1.5 ms per further row) within 10% of `quality/c03_expect.py` at 1, 4 and 8 users.
The CUDA side has run since: `tools/dsv41/pod/devgpu.sh` with `DEVGPU_PLAN=c03` on an NVIDIA L4 (2026-10-14: the
batched kernels and the engine's batch on tiny models, every request bit for bit its alone run), and the aggregate
throughput numbers on the RTX PRO 6000 Server (`E07_PLAN=cbatch` 2026-10-14, `cbatch-clong` 2026-10-15; see "The
batched steps' time" below; results in docs/RESULTS.md).

## No user frozen by a long prompt, and the server robust

At first a prompt read while batch slots decode held every slot for the whole read: a 64K prompt (~20-25 s alone on the
RTX PRO 6000) froze every decoding user for that long. `dsv41-engine --serve --batch N --interleave-chunk K` (default
`--chunk`, the prefill chunk: 16384 as `e07-run.sh` and `pod_server.sh` start it; at most `--chunk`; 0 is the first version's read
whole) reads such a prompt in fixed chunks of K tokens with one batch step after each chunk but the last
(`SessionRequest::chunk`, `SessionEvents::between`), so a decoding user waits for one chunk and one step at a time; a
`BSTOP` that comes during the read ends its slot before the next of those steps (a client that left frees its slot
mid-read). A prompt read with no slot decoding is read in `--chunk`, as before. `Forward::hold_prompt` keeps the
experts' prompt target for the prompt's end, so the steps between its chunks do not end it early. The request's "read
beside" log line gives its chunks, each chunk's read time (mean and longest), the steps between them and the longest
hold.

Why the prefill chunk and not a small one: a prefill chunk on the real model copies nearly every warm expert it
touches, whatever its size (A100, chunk 16384: ~103 GiB a chunk; the 113.13 GiB warm tier at the measured 26.12 GB/s
would take 4.65 s), so a 1024-token chunk costs about what a 16384 one does: a 64K prompt in 1024-token chunks would
take ~64 x 4.6 s, about 5 min, beside the users, with no gain in the gap. At 16384 the gap is one chunk and one step,
and the 64K TTFT beside the users stays close to its solo run (4 chunks and a short fifth for the 65565-token prompt).
A chunk's time is the host's: beside 4 decoding users a 16K chunk took 3.2-3.3 s on an RTX PRO 6000 Server Edition
host of 2026-10-15 (PCIe Gen 5: the 64K read's longest chunk 3.29 s, the users' largest gap 3.35 s), an RTX PRO 6000
Workstation card read a 16K prompt cold in 4.59 s, and a slower Workstation card in 7.68 s, past the 5 s gap target.
The gap's margin follows the host's pinned H2D: on an RTX PRO 6000 Workstation Edition host of 2026-10-16 (AMD EPYC
9355P, PCIe Gen 4 x16, H2D 28.6 GB/s in the machine qualification) the longest chunk beside the slots was 4.02 s
(4016.6 ms, a user's own 16K prompt read beside 1 slot) and the users' largest gap 4.048 s (target 5 s), against
3.29-3.31 s at most and gaps of 3.325-3.350 s on the Server Edition machines of 2026-10-15 and 2026-10-16 (PCIe Gen 5,
H2D 55.1-57.7 GB/s). The qualification's H2D floor (`QUALIFY_MIN_H2D_GBS`, 18 GB/s) admits hosts slower than that Gen 4
one, so it does not bound the gap's margin; each sheet's chunk note names its host's H2D beside its chunk times. The
engine warns on stderr when a chunk read beside decoding slots passes 4.5 s ("warning: a chunk read beside N decoding
batch slots took X ms"). Smaller chunks stay a flag or config value (`"interleave_chunk"`). On a host whose gap passes
5 s the way forward is a cheaper interleaved read, sized from that host's chunk times.

The batch slots' positions: the server's deepseek-v4.1 profile starts the engine with `--batch-ctx 69632` when the
config has `"parallel"` and no `"batch_ctx"` (at most its `--ctx`), so a 64K prompt and its reply fit a slot and the
long-prompt case decodes beside the other users on a default deployment. The trade-off, `e07_gate.py` on the real
headers for an RTX PRO 6000 host: 8 slots of 69632 take 217.6 MiB of device KV each, 1.70 GiB with the rows' logits,
and the slab holds about 142 experts fewer (2.4% of its 5825); 8 slots of 32768 (the engine's own default) take 105.1
MiB each, 0.82 GiB, about 69 fewer. `"batch_ctx"` in the config sets another size.

A request without `max_tokens` answers at most the deepseek-v4.1 profile's `default_max_tokens`, 8192 (finish
"length"); with batch slots an explicit `max_tokens` past a slot's room is cut to the room when the prompt fits a slot
(finish "length"), so only a prompt past a slot can run alone, and only while nothing else decodes: beside decoding
requests it gets a 400 naming the slot's size. A request that does run alone says so at its start: the engine's line
"request N: runs alone on the live sequence, not in batch slot S: <why>; it holds K decoding batch slots until its
end" and the control line `ALONE <slot> <K> why=<why>` before its read, which the server logs at once and shows on
`/metrics` (`live.ran_alone`) until its `BADM`; `strata_ran_alone` is on the `/v1/messages` and `/v1/responses`
replies too. A request reaching `request_deadline_s` ends with its text and finish "length" (an error only with no
token yet). Details and the 503s' codes and `Retry-After`: [docs/DETAILS.md](../../../docs/DETAILS.md).

The cut is said in the reply too. A request whose `max_tokens` the server shortened (a slot's room, or
`--fit-max-tokens`' context room) carries `"strata_max_tokens": {"asked": A, "max_tokens": M}` (the last streamed chunk
and the non-streamed body of `/v1/chat/completions`, the `message_delta` event and the message of `/v1/messages`, the
response object of `/v1/responses`); absent when it got what it asked (`"asked": null` when it asked for none, below).
A reply that ends "stop" before M says nothing of the cut by itself: in a session of 2026-10-16 (RTX PRO 6000) the
server cut 32000 to a room of 137 (a 69496-token prompt in a 69632 slot) and the model ended after 13 tokens, so
`robust_check.py`'s slot-room check, which wanted finish "length" at the room, failed on a correct server. It now takes
the cut from this field and accepts "length" at the room or "stop" within it (`clamp_verdict`, tested by
`quality/robust_check_test.py`).

A request that asks for no `max_tokens` gets the server's limit said the same way, `"strata_max_tokens": {"asked":
null, "max_tokens": M}` on the same three routes, M the profile's `default_max_tokens` (8192 on deepseek-v4.1) within
the context's room and a slot's. `robust_check.py`'s no-max-tokens check had the same weakness in its NEAR turn (a
prompt 160 tokens under a slot, no `max_tokens`: it wanted finish "length" at the room, and passed in two sessions only
because the prompt asks for a count past the room: 160 of 160 tokens, "length"). It now wants this field's M equal to
the room (and the room under the default) and accepts "length" at M or "stop" within it (`near_verdict`, with
`clamp_verdict` one `room_verdict`). The field is absent only when the `max_tokens` asked was kept.

Before the default reply length, a request without `max_tokens` got the rest of the context, which never fits a slot,
so it ran alone and held every decoding user until its end of reply (OpenAI clients leave `max_tokens` out by
default). With batch slots the server gave such a request, when its prompt fits a slot, the slot's room instead
(`Service.slot_room`: `batch_ctx - prompt + 1`, about 53K tokens for a 16K prompt in a 69632 slot), ending "length" at
the slot's end, decoding beside the others ([docs/DETAILS.md](../../../docs/DETAILS.md), `max_tokens` with batch
slots). A request longer than a slot (its prompt, or an explicit `max_tokens`, past the slot's positions) still runs
alone on the live sequence and holds the batch while it does: its read is interleaved as any other, but from its first
token to its last no slot steps. It is visible: the engine's stderr line "request N: ran alone on the live sequence,
not in batch slot S: <why>; G generated (finish), holding K decoding batch slots for X ms" and its
`BADM <slot> 0 rep=R alone=K why=<why>`; the server logs it and adds `"strata_ran_alone": {"reason": ...,
"held_slots": K}` to the reply (the last streamed chunk, the non-streamed body); `e07_report.py` lists them on the
sheet. `quality/robust_check.py`'s no-max-tokens check runs both on the real engine (4 users decoding and a count
without `max_tokens`: it takes a slot and no user's gap passes 5 s; then a short request past a slot that says it ran
alone holding the 4).

| Part | What | Where |
|---|---|---|
| interleaved read | fixed chunks (deterministic boundaries), a step between two, BSTOPs taken between chunks, no idle reset under the read | `session.cpp`, `dsv41-engine.cpp` serve loop, `forward.hpp` `hold_prompt` |
| bounded queue | `"queue_max"` (deepseek-v4.1 profile: 8): past it a 503 at once, `Retry-After` from the queue (the requests ahead over the running places, each the median of the last finished requests), a JSON error naming the queue; `/metrics` `live.queue` | `serve/server.py` `Service.admission` |
| freed slots | a streamed client's failed write or a non-streamed one's socket check (every 0.5 s) cancels it: `BSTOP`, the slot freed at its `BDONE` for the next waiting request, logged | `StrataEngine.generate_batched` |
| nothing hangs | the engine watchdog with `"parallel"` (no engine line for `"engine_silence_s"` while a request is in it: the engine is ended, every running and waiting request fails with that reason, 503 or the stream's error event) and `"request_deadline_s"` (1800: an error, 504 or the stream's error event) | `StrataEngine._watchdog`, the handler's watch |
| non-streamed replies | the same batched path as streamed ones (they always took it; `serve/test_robust.py` checks it) | `openai_collect` over `openai_chunks` |

The Qwen3.8 profile keeps its behaviour (no queue bound, no deadline, no watchdog in batch mode; the engine's
`--interleave-chunk` is dsv41-engine's only).

On the dev VM (2026-10-14, 12 vCPU, no GPU; the cu130 builder image): `dsv41_engine_batch_test` check 6 (tiny-b, the CPU
reference): a 2600-token prompt read in 3 chunks of 1024 under a prefill chunk of 2048, 2 steps between them beside 3
decoding slots, bit for bit its alone run with the same chunks, the slots bit for bit theirs; the serve dry run's
interleaved read (`tools/dsv41/quality/c04-test.sh`, `verify.sh`); `serve/test_robust.py`; 2026-10-15, with the
prefill chunk as the interleave default, on `fakes.py --chunk 16384 --batch 8` with the profile's default `--batch-ctx
69632` and the default interleave chunk (0.28 ms a prompt token, the measured 16K rate, 5 ms a step, 1.5 ms per further
row) `quality/chat_load.py long`: the 65565-token prompt in 5 chunks of 16384 with 4 steps between them, 4 users'
largest gap 4.636 s against `quality/c04_expect.py`'s 4.597 s (one chunk and one step; under 5 s), the 64K TTFT 18.78 s
beside them against 18.68 s alone (1.006 x), 18.41 s with `--interleave-chunk 0`; `quality/robust_check.py` 6 of 6.
Those are the fake's numbers. The CUDA side, each real chunk's read time, the real gap and the 64K TTFT are
`E07_PLAN=clong`'s (or `cbatch-clong`'s, with the throughput steps first) checks; the RTX PRO 6000 Server session of
2026-10-15 passed the gap and long-prompt TTFT targets (the gap 3.35 s, the 64K TTFT 0.996 x its solo run, 6 of 6
checks). The request deadline and the engine watchdog were checked on the fake engine only (`serve/test_robust.py`,
c04-test.sh). With the slot-room rule (2026-10-15, the same fake): the no-max-tokens check, the count admitted to a
slot within 0.06 s, the 4 users' largest gap 0.045 s through it and the 10 s after it, the request past a slot answered
with `strata_ran_alone` holding 4; `robust_check.py` 7 of 7.

## The batched steps' time

The RTX PRO 6000 Server session of 2026-10-15 (cbatch-clong, steady-c4 with `--batch-ctx 32768`) missed the
throughput target: 4 users decoding at once 71.64 tok/s in aggregate, 1.33x the single user with batching off
(53.68), every user 16.5 to 18.9 tok/s. The engine's own steps: 1029 steps of 3.97 rows, 38.96 ms a step inside
`batch_decode`. From the same log, a slot that no prompt read held (request 12, slot 3) took 45966.6 ms from its
request's start to its 1023rd step, 3411.6 of them its own prompt read: 41.6 ms a step of wall time, so about 3.2 ms a
step went outside the step. And the 15 s window that opens at the last user's first token ran at 18.3 tok/s a user
against about 26.6 later in the same replies (the server log's per-second counts climb for about 30 s after the last
read).

What the engine says now (`dsv41-engine.cpp` `StepParts`, `BatchSlots::Totals`): once the slots are all idle, one
"batch steps, <part>" line each for all the steps, those within 15 s of a prompt read beside the slots and the others:
the ms a step from one step's start to the next, inside the step (the graph's launch with the rows' inputs, the
doorbells served and of them the waits for the device's ring, the CPU jobs and their plan, the tail to the logits, the
boundary and its swap waits, the device's own graph time, the graphs captured again) and between two steps (the picks
of the rows' tokens, the BT lines and their flush, the admissions' copies, the loop with its stdin poll), and per step
the rows' routed pairs, their distinct experts (`ExpertTimes::distinct`), hits, misses, CPU jobs (one per distinct
missed expert for every row routed to it) and swaps. `e07_report.py` puts them in a table on the sheet beside the
steady-c4 result.

What changed:

| | Change | Measured on the dev VM | Expected on the RTX PRO 6000 |
|---|---|---|---|
| between steps | the slots' greedy pick `batch_argmax` in eight lanes (the plain loop's pick, checked on 2002 rows by `dsv41_engine_batch_test` check 7) | 4 rows of 129280 logits: 0.15 ms against 1.50 (Ryzen 7 5700X, -O3); the serve dry run with the real vocabulary, 3 rows a step: 0.13 ms between two steps, 0.129 of them the picks, the BT lines and the loop 0.003 | the plain loop's ~2.3 ms a step at the EPYC 9335's 3.0 GHz (the dev VM's 1.50 ms at 4.6 GHz scaled) goes: about 3.2 -> 1 ms a step, the rest for the breakdown to name |
| after a prompt read beside the slots | the read's routing joins the batch's counts at one row's decode mass (`HotLayerPlan::merge_prompt`) instead of replacing them with its 16K tokens' and taking a target of its own | `hot_policy_sim.py batch` on the real model's routing trace (A100, 2026-10-09) with this session's RTX hot set: a row joining after its own 16K prompt, the CPU jobs a step 49.5 / 26.3 / 19.1 / 16.3 in steps 0-25 / 25-50 / 50-100 / 100-200 after the read, against 52.0 / 48.6 / 50.8 / 52.7 before; 4 rows and an unrelated 16K read, 92.1 / 64.2 / 57.1 / 58.3 against 99.4 / 91.6 / 82.4 / 89.8 (steps 0-25 / 25-50 / 50-100 / 100-200) | at the single row's 0.289 ms a CPU job (request 1 of that log: 5188 jobs in 1497 ms), about 1 to 10 ms less a step in the 200 steps after a read |
| the boundary of a batched step | a step of 2 rows or more no longer waits, in the deterministic mode, for its open batch of swaps (with `--cpu-numerics gpu` a row's bits do not depend on where its experts are, so the swaps' timing moves none; a 1-row step still waits) | (CUDA side: the breakdown's swap waits) | whatever the session's "swap waits" column shows in the window, where the swaps run at up to 80 a step |
| the slots' positions | the cbatch plan's slots at the deployment's 69632 (`CBATCH_CTX`; first measured at 32768) | e07_gate.py on the real headers: about 142 slab experts fewer (69 at 32768) | the hit rate a little lower than 2026-10-15's |

A request alone keeps its hot set: it starts from the slab with its own prompt target and its boundaries wait in the
deterministic mode ([DETERMINISM.md](DETERMINISM.md); q03-test.sh, c02-test.sh). Batched replies stay
`"strata_reproducible": false` under the default `--cpu-numerics fork`.

Two more changes:

- A request alone picks its greedy token with `batch_argmax` too (`Session::run`): the same token as the plain loop
  (`dsv41_engine_batch_test` check 8: the Session on a backend returning check 7's 2002 rows, every pick the plain
  loop's), in about 0.04 ms a row of 129280 logits against 0.375 (dev VM, Ryzen 7 5700X, -O3; check 7's four-row
  figures over one row). The single user with batching off gains it as well as the slots; the deterministic mode's
  tokens do not change (q03-test.sh, c02-test.sh).
- `"hot_spare": N` in a deepseek-v4.1 config starts the engine with `--hot-spare N`, the free slab slots per layer the
  hot set's swaps copy into; absent keeps the engine's 2. On an RTX PRO 6000 host with 8 slots of 69632 positions
  (`e07_gate.py`, the real headers), 4 spare slots take 0.95 GiB more of slab than 2: 5741 slab experts against 5825,
  the single request's expected hit rate 0.8794 against 0.8831. The hot set must leave their room
  (`e07_gate.py --hot-spare N`), else the engine refuses to start.

`E07_PLAN=cbatch-clong` with `CBATCH_SPARE=4` runs a second cbatch server with 4 spare slots (gate-spare,
engine-server-spare, load-short-spare, steady-c4-spare, server-stop-spare, last in the plan); the sheet then judges the
throughput target on each server side by side against the same baseline (the single user with batching off on the
default config). `CBATCH_SPARE` defaults to 0: with 4 the HF plan's estimates take 66.7 min, 2.89 USD at 2.60 USD/h,
but steady-c4-spare's whole timeout ends at minute 68.0, past the pod's 67-minute deadline, and the dry run refuses
it. What the variant should change, by `hot_policy_sim.py batch` on the real routing trace (A100, 2026-10-09; 4 rows,
request 4's decode and three windows of its 16K prompt's tokens, from the deterministic slab, the gates' hot sets of
an RTX PRO 6000 host with 8 slots of 69632): CPU jobs a step 209.4 / 70.3 / 60.8 with 2 spare slots and 211.8 / 72.1 /
62.9 with 4 (steps 0-50 / 50-150 / 150-300), 4 swaps a layer a step as well 191.7 / 72.0 / 63.0. The planner starts
about 66 swaps a step in the warm-up (1.6 a layer, under its 2), so the spare slots do not bound it; the slab's 84
fewer experts cost about 2 jobs a step. The sim publishes every swap at the next step; on the RTX the copies ran at 44
GB/s (2026-10-15: 45446.5 MiB in 1081.5 ms, request 1), about 21 ms for 66 swaps of 13 MiB against a 35-39 ms step,
so they keep up there too. So the variant was not run: the simulation predicts no gain and its plan ends past the
pod's latest-end rule. `"hot_spare"` stays off (absent, the engine's 2) and `CBATCH_SPARE` stays 0.

The result (the combined RTX session of 2026-10-15 15:43, RTX PRO 6000 Blackwell Server Edition, AMD EPYC 9575F, 283
GB, the default config: slots of 69632, 2 spare slab slots a layer): the throughput target passed. steady-c4 122.13
tok/s in aggregate (repeats 121.07 to 122.93) = 2.21x the single user with batching off (55.24 tok/s; it rose from
53.68 with the lone request's pick and the other machine), the slowest user 30.27 tok/s (>= 20); the single user with
batching on 56.16 tok/s (+1.7% against off, >= 30). End to end, the prompt reads that held them included, the same
replies: the slowest user 24.00 tok/s over its whole reply, 95.86 tok/s in aggregate over the decode window (1.74x),
the users' first tokens at about 4.1 / 7.6 / 11.0 / 14.5 s. The 8 users' figures are still 2026-10-14's (70.42 tok/s in
aggregate, before the interleaved read and these step changes): `CBATCH_STEADY_REPORT` measures steady-c8 on this code
(reported, not a target). Results: docs/RESULTS.md.
