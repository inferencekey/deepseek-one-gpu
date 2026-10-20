# DeepSeek-V4.1-Flash forward pass, as the stock fork computes it

The correctness reference for the dedicated engine is the stock fork (the patched llama.cpp fork of
`third_party/llama-dsv41/`, checked out at `$FORK_DIR`) at `b4d9cb1` (base `5210c7c5` plus the Strata MoE hook, which is off unless
`LLAMA_STRATA_MOE=1`; nothing below runs it). This file specifies every op the fork computes for
`general.architecture = deepseek41`, in forward order, with the shapes, types and constants it uses, the fork source
lines that compute it, and the name under which `op-dump` records it. Line numbers are at `b4d9cb1`. Where the
DeepSeek tech report (`DeepSeek_V41_Tech_Report.pdf`) or the official `config.json` says something else, "Divergences"
records it: the engine reproduces the fork (it is the oracle) unless a listed item says otherwise.

Conventions: shapes are ggml `ne` order, fastest first (`[n_embd, n_tokens]` is one row of `n_embd` per token).
`T` tokens in the ubatch, `E` n_embd, `HC` = 4 hyper-connection copies, `H` heads, `D` key_length, `R` rope dims,
`QL` q_lora_rank, `IH`/`ID` indexer heads/key length, `NE` experts, `K` experts used, `FF` expert FFN width, `OG`/`OL`
output groups/lora rank, `NC` engram buckets, `KL` engram key length, `r` the layer's compress ratio. rms(x) is
`x / sqrt(mean(x^2) + eps)` (ggml_rms_norm). Dump names are `op.occurrence-Llayer-uubatch.bin`, layer 999 for
model level tensors (`op-dump/op-dump.cpp`, `op_dump.py`).

## Sources

| What | Where |
|---|---|
| Fork graph | `src/models/deepseek41.cpp` (948 lines), `src/models/deepseek4.cpp` (shared helpers), `src/llama-graph.cpp` (`build_norm`, `build_ffn`, `build_moe_ffn`, `build_attn_mha`, DeepSeek-V4 inputs) |
| Fork KV cache | `src/llama-kv-cache-dsv4.cpp` (plan 427-743, cache 1222-1447), `src/llama-kv-cache-iswa.cpp` 69-80, `src/llama-kv-cache.cpp` 324-335 and 1438-1459 |
| Fork CPU kernels | `ggml/src/ggml-cpu/ops.cpp`: swiglu_clamp 3408-3465, soft_max with sinks 5584-5694, rope with `n_offs` 6100-6131, DSV4 hc comb/pre/post 11110-11380, lightning indexer (V4 only) 12082-12164 |
| Real model header | `vcruz305/DeepSeek-V4.1-Flash-GGUF` Q2_K revision `c4a08554`, the first 64 MiB of shard 1 and 8 MiB of shards 2-7 (HTTP range requests, every tensor info of all 1046 tensors), kept in `$BUILD_ROOT/engine/headers/` |
| Reference | `deepseek-ai/DeepSeek-V4.1-Flash` `config.json` and the tech report, fetched 2026-10-07 |

The fork keeps V4.1's three changes against V4 in its file header (deepseek41.cpp 14-34): the hyper-connection mix
lags by one sublayer, there is no hyper-connection head, and the engram tables.

## Constants (real GGUF header)

| Key | Value | Used as |
|---|---:|---|
| block_count | 40 | layers (compress_ratios has 43 entries: the 3 DSpark/MTP layers, unused here) |
| embedding_length / vocab | 5120 / 129280 | E, token_embd and output rows |
| attention.head_count / head_count_kv | 64 / 1 | H; one shared latent K (MQA, V = K) |
| attention.key_length = value_length | 512 | D, of which rope.dimension_count 64 = R are rotated |
| attention.q_lora_rank | 1280 | QL |
| attention.output_group_count / output_lora_rank | 8 / 1024 | OG, OL |
| attention.sliding_window | 128 | n_swa |
| attention.compress_ratios | 0,0, 2 x18 (L2-19), 1 x20 (L20-39) | r per layer |
| attention.compress_rope_freq_base | 160000 | rope base of compressed layers |
| attention.indexer.head_count / key_length / top_k | 32 / 128 / 512 | IH, ID, top-k |
| rope.freq_base, rope.scaling yarn factor 16, orig ctx 65536, beta_fast 32, beta_slow 1 | | see "Rope" |
| attention.layer_norm_rms_epsilon | 1e-20 (f32 9.9999997e-21) | every rms norm |
| hyper_connection.count / sinkhorn_iterations / epsilon | 4 / 20 / 1e-6 | HC |
| expert_count / expert_used_count / expert_feed_forward_length / expert_shared_count | 384 / 6 / 2304 / 1 | NE, K, FF |
| expert_gating_func | 4 (sqrtsoftplus) | router |
| expert_weights_norm / expert_weights_scale | true / 1.5 | router |
| swiglu_clamp_exp / swiglu_clamp_shexp | 10.0 for every layer | expert SwiGLU limit |
| engram.layer_ids / head_count / key_length / max_ngram_size / pad_id | [1, 14] / 8 / 256 / 4 / 2 | NC = 3 x 8 = 24, KL |
| engram.multipliers / primes / offsets | 8 / 48 / 48 u64 | the n-gram hash (primes about 16.0M, offsets cumulative per layer, from 0) |
| engram.token_map | 129280 i32, 99092 distinct ids | token -> compressed id |
| hash_layer_count | 0 | every layer routes through the gate |

## Tensors (real model, 1046)

| Tensor | Shape | Type | Layers |
|---|---|---|---|
| token_embd / output | [5120, 129280] | q2_K / q6_K | |
| output_norm | [5120] | f32 | |
| attn_norm, ffn_norm | [5120] | f32 | all 40 |
| attn_sinks | [64] | f32 | all |
| attn_q_a / attn_q_a_norm / attn_q_b | [5120, 1280] / [1280] / [1280, 32768] | q2_K / f32 / q2_K | all |
| attn_kv / attn_kv_a_norm | [5120, 512] / [512] | q2_K / f32 | all |
| attn_output_a / attn_output_b | [4096, 8192] (loaded as [4096, 1024, 8]) / [8192, 5120] | q2_K | all |
| hc_attn_fn, hc_ffn_fn / _base / _scale | [20480, 24] / [24] / [3] | q2_K / f32 / f32 | all |
| attn_compressor_kv / attn_compressor_norm | [5120, 512] / [512] | q2_K / f32 | 2, 8, 14, 20 |
| attn_compressor_gate | [5120, 512] | q2_K | 2, 8, 14 (ratio 2 only) |
| indexer.attn_k / indexer.k_norm | [512, 128] / [128] | q2_K / f32 | 2, 8, 14, 20 |
| indexer.attn_q_b / indexer.proj | [1280, 4096] / [5120, 32] | q2_K | 2, 8, 14, 20, 24, 28, 32, 36 |
| ffn_gate_inp | [5120, 384] | bf16 | all |
| exp_probs_b, exp_probs_b_vl (bias) | [384] | f32 | all (vl: image tokens only) |
| ffn_gate_exps, ffn_up_exps / ffn_down_exps | [5120, 2304, 384] / [2304, 5120, 384] | q2_K / q3_K | all |
| ffn_gate_shexp, ffn_up_shexp / ffn_down_shexp | [5120, 2304] / [2304, 5120] | q2_K / q3_K | all |
| engram_embd | [256, 384006168] (L1), [256, 384016682] (L14) | q2_K (about 30 GiB each) | 1, 14 |
| engram_wkv / engram_q, engram_k | [6144, 25600] / [5120, 4] | q2_K | 1, 14 |

## Layer roles

The file says which layer plays which role (deepseek41.cpp 209-256): a **KV source** carries attn_compressor_kv, a
**key owner** indexer.attn_k, an **index source** indexer.attn_q_b. Every compressed layer reads the latest source,
key owner and index source at or before it (`dsv41_kv_source`, `dsv41_index_key_source`, `dsv41_topk_source`); the KV
cache aliases a reader's rows onto its source's slot (reuse callbacks, llama-kv-cache-dsv4.cpp 1390-1396).

| Real layers | Role (tech report mode) | Computes |
|---|---|---|
| 0, 1 | window only (SWA) | raw attention over the last 128 positions |
| 2, 8, 14 | ratio 2 source, gate, key owner, index source (Full) | compressor, index keys, indexer top-k, attention |
| 3-7, 9-13, 15-19 | ratio 2 reader (Reuse) | attention over the source's compressed rows (see Divergences 2) |
| 20 | ratio 1 source without gate, key owner, index source (Full, candidate source) | as 2, pooling over 1 token |
| 24, 28, 32, 36 | ratio 1 index source reading layer 20's keys (Reindex) | own indexer query, top-k |
| 21-23, 25-27, 29-31, 33-35, 37-39 | ratio 1 reader (Reuse) | attention over layer 20's rows |

The engram adds into the stream before layers 1 and 14.

### Reader modes (engine decision)

The fork's readers attend to every visible compressed row of their source, not to the index source's top-k
(Divergences 2). Decided for the engine:

- **Reuse Mode is the default.** A reader attends with the top-512 of its `topk_src` (the latest Full or Reindex layer
  at or before it), as the tech report specifies; its attention then costs the same at any context.
- **`--readers dense` is the fork-parity mode.** A reader attends to every visible compressed row, exactly as the fork.
  Parity against a fork dump is required in this mode at every tested length.
- **Where they coincide.** At position p a reader sees (p + 1) // r compressed rows; while those are at most top_k the
  top-k selects them all and both modes compute the same. That holds for contexts up to (top_k + 1) x r - 1 tokens:
  1025 at ratio 2 and 512 at ratio 1 for the real model (top_k 512), so up to 512 tokens in every reader; 17 for
  tiny-a and 33 for tiny-b (top_k 8 and 16). Only there is the default mode checked against the fork.

`op_dump.py roles` prints each layer's mode, the layers it reads and that bound, and for a dump whether the modes
coincide over its tokens; `Model.modes_coincide(n_tokens)` gives a parity test the same answer. The 64 + 8 token dumps
of `run-oracles.sh` exceed both tiny bounds, so they check the default mode only through `--readers dense`; a dump of
at most 17 tokens (tiny-a) or 33 (tiny-b) checks both. The missing candidate indexer (Divergences 3) follows the same
rule in the indexer and attention tests.

## Forward pass

### 1. Embedding and the stream (deepseek41.cpp 812-829, llama-graph.cpp 2438-2525)

- `embd` [E, T] f32 = get_rows(token_embd, tokens): rows dequantized from q2_K.
- `hc_init` [E, HC, T]: the embedding repeated into the 4 copies.
- `hc_pre_init` [HC, T]: the collapse weights layer 0's attention uses, one-hot [1, 0, 0, 0] (no previous sublayer).

### 2. Engram (before layers 1 and 14; deepseek41.cpp 262-438)

- Rows, host side (262-355): per token i the n-gram `t_i, t_{i-1}, t_{i-2}, t_{i-3}` through engram.token_map; a
  position before the sequence start, and everything older than it, reads as pad_id 2 (the look-back stops at the
  first missing token). `rolling_1 = c_0*m_0 ^ c_1*m_1`, `rolling_s = rolling_{s-1} ^ c_s*m_s` (u64, wrapping), with
  m the layer's 4 multipliers; for s = 1..3 and head h, bucket b = (s-1)*8 + h: `row = rolling_s % prime[b] +
  offset[b]`. The predecessors come from the raw KV cells (`ext.tok`), so a token shared by several sequences aborts.
- `engram_embd` [KL*NC, T] = get_rows(engram_embd, rows) flattened bucket-major (`emb[KL*b + j]`), dequantized.
- `engram_kv` [E*(HC+1), T] = engram_wkv x emb; key = the first HC*E values, value = the next E.
- gate (382-431): `k` = rms per (token, copy) of key over E, times engram_k; `q` = the same of the stream x, times
  engram_q (both [E, HC], dequantized through get_rows when quantized, 398-406); `s = sum_e k*q / sqrt(E)`;
  `engram_gate` [1, HC, T] = sigmoid(sgn(s) * sqrt(clamp(|s|, 1e-6, inf))).
- `engram_out` [E, HC, T] = x + value * gate (value shared by the copies).

### 3. Hyper-connection mixes, collapse and expansion (deepseek4.cpp 286-457, ops.cpp 11110-11380)

Per sublayer, from the stream x [E, HC, T] (deepseek4.cpp 352-406):
- `hc_mixes` [24, T] = hc_fn x rms(x flattened to [HC*E, T]) (no weight; eps 1e-20). Rows: 0-3 pre, 4-7 post, 8-23
  comb (dst fastest).
- `hc_pre` [HC, T] = sigmoid(mixes[0:4] * scale[0] + base[0:4]) + 1e-6.
- `hc_post` [HC, T] = 2 * sigmoid(mixes[4:8] * scale[1] + base[4:8]).
- `hc_comb` [HC(dst), HC(src), T] (fused `GGML_OP_DSV4_HC_COMB` on the CPU, enabled by resolve_fused_ops,
  llama-context.cpp 573-579): `c = mixes[8:24] * scale[2] + base[8:24]`; per src a softmax over dst, plus 1e-6;
  then a column normalization (per dst over src: `c /= (sum + 1e-6)`), then 19 more rounds of row normalization
  (per src over dst) followed by column normalization (ops.cpp 11110-11226, the unfused form deepseek4.cpp 313-348).
- collapse `hc_pre` op (fused `DSV4_HC_PRE`, ops.cpp 11248-11300): `out[e] = sum_s x[e, s] * w[s]`.
- expansion (fused `DSV4_HC_POST`, ops.cpp 11313-11370): `out[e, d] = y[e] * post[d] + sum_s comb[d, s] * x[e, s]`.

The V4.1 lag (deepseek41.cpp 850-919): in layer l the attention mixes are computed from the incoming stream, but the
attention input collapses with the FFN mix of layer l-1 (`hc_pre_init` for l = 0); the FFN input collapses with the
attention `hc_pre` of the same layer. Dump occurrences per layer: `hc_mixes.0`/`hc_pre.0`/`hc_post.0`/`hc_comb.0`
are the attention sublayer's, `.1` the FFN's.
- `hc_attn_pre` [E, T] = collapse(x, pre of layer l-1's FFN mixes); `attn_norm` = rms * attn_norm.
- `hc_attn_post` [E, HC, T] = expansion(attn_out, x, hc_post.0, hc_comb.0).
- `hc_ffn_pre` [E, T] = collapse(hc_attn_post, hc_pre.0); `ffn_norm` = rms * ffn_norm.
- `l_last` [E, HC, T] = expansion(ffn_out, hc_attn_post, hc_post.1, hc_comb.1).

### 4. Rope (deepseek41.cpp 440-461, deepseek4.cpp 11-17, ops.cpp 6100-6131)

Mode NORM (adjacent pairs), applied to the last R = 64 values of each head (`ggml_rope_set_offset(t, dim - R)`, the
first `dim - R` pass through). Window-only layers: base 10000, freq_scale 1, no YaRN (ext_factor 0, attn_factor 1).
Compressed layers, the indexer and the compressed rows: base 160000, freq_scale 1/16, ext_factor 1 (llama-context.cpp
172-173), beta 32/1, n_ctx_orig 65536, attn_factor `1 / (1 + 0.1 ln 16)` = 0.78293, which cancels YaRN's magnitude
factor, so YaRN only reshapes the frequencies. Queries and window keys rotate at the token position, compressed rows
and index keys at the first position of their block.

### 5. Attention (deepseek41.cpp 592-804)

- `qr` [QL, T] = rms(wq_a x cur) * attn_q_a_norm (616-619; V4.1 normalizes only the low rank part).
- `q` [D, H, T] = rope(wq_b x qr) (621-626).
- `kv` [D, 1, T] = rope(rms(wkv x cur) * attn_kv_norm) (628-635), stored as f16 in the layer's raw window cell.
- **Window only (r = 0)** (641-646, deepseek4.cpp 843-877): `kq` [W, T, H] = K^T q over the raw cells, `kq_soft_max`
  = softmax(kq / sqrt(D) + mask) with the head's sink in the denominator (ops.cpp 5625-5680: the max includes the sink
  and `exp(sink - max)` is added to the sum), `kqv` [D, T, H] = V x p with V = K (the same latent), `attn_raw` [D*H, T].
  The window mask hides keys with `p_q - p_k >= 128` (llama-hparams.h 484-496) and future keys.
- **Compressor (KV sources)** (661-746, deepseek4.cpp 481-534): `comp_state_kv` [D, T] = attn_compressor_kv x cur;
  `comp_state_score` [D, T] = attn_compressor_gate x cur, or the state KV itself at ratio 1 (no gate; the callback
  then names that one tensor `comp_state_score`). Each completed block of r consecutive positions `[j*r, (j+1)*r)`
  pools per channel: `latent_j[d] = sum_i softmax_i(score[d, i]) kv[d, i]` (softmax over the r tokens; exactly the KV
  at r = 1). A block that the ubatch does not complete is carried in the f32 state (`dsv4_<csa|hca>_state_*`, restored
  and persisted by the plan, llama-kv-cache-dsv4.cpp 427-743); a ubatch that completes none computes one dummy block
  into the last cache cell, masked out (605-633). Then rms * attn_compressor_norm (pre-rope latent), rope at `j*r`,
  stored f16 in compressed row j of the source's slot. Dump: `comp_kv.0` gathered KV [D, r, B], `.1` gathered score,
  `.2` pooled [D, 1, B], `.3` normed (the index keys' input), `.4` roped (what the cache stores).
- **Index keys (key owners)** (698-716): `idx_k_new` [ID, 1, B] = rope(rms(indexer.attn_k x latent_pre) *
  indexer.k_norm) at `j*r`, then multiplied by a 128 x 128 Hadamard matrix (the lid cache always has one for DeepSeek
  indexers, llama-kv-cache.cpp 330-335) and stored f16 in the lid slot.
- **Indexer (index sources)** (503-590): `idx_q` [ID, IH, T] = rope(indexer.attn_q_b x qr) at the token position,
  `idx_q_rot` = idx_q x Hadamard; `idx_weights` [IH, T] = indexer.proj x cur / sqrt(ID * IH); `idx_k` [ID, 1, C] the
  key owner's rows; `idx_score` [C, T] = `sum_h relu(idx_q_rot[:, h, t] . idx_k[:, j]) * w[h, t]` plus the compressed
  causal mask (0 for `j < (pos_t + 1) / r`, else -inf; llama-graph.cpp 764-806). The scores are materialized (mul_mat,
  permute, relu, mul, sum_rows), C x T x IH floats. `idx_top_k` [min(C, top_k), T] i32 = ggml_top_k(score), an
  unordered set; where fewer rows are visible than k, the set includes -inf rows, which the masks below keep hidden.
- **Masks** (785-790, deepseek4.cpp 697-724): `comp_top_k_mask` [C, T] = (0 on the top-k rows, -inf elsewhere) +
  the compressed causal mask, for index sources; readers use the compressed causal mask alone. `kq_mask` [W + C, T] =
  concat(window mask, compressed mask).
- **Attention over window + compressed rows** (754-801): `raw_k` [D, 1, W] (the window cells), `comp_k` [D, 1, C] (the
  source's compressed cells), `k_all` [D, 1, W + C] f16; `kq` [W + C, T, H], `kq_soft_max`, `kqv` [D, T, H] as in the
  window case with V = K; `attn_out_raw` [D*H, T]. C is the visible compressed count padded to 256, at least 256
  (llama-kv-cache-dsv4.cpp 647).
- **Output** (463-501): `attn_derope` [D, H, T] = inverse rope of the last R values at the token position (the output
  is a mix of rotated keys); `attn_wo_a` [OL, T, OG] = per group g of H/OG heads, `wo_a[:, :, g]^T x out_g`;
  `attn_out` [E, T] = attn_output_b x (the groups flattened to [OL*OG, T]).

### 6. Router (llama-graph.cpp 2056-2187)

- `ffn_moe_logits` [NE, T] = ffn_gate_inp x ffn_norm (bf16 weight, F32 accumulation requested).
- `ffn_moe_probs` = sqrt(softplus(logits)) (2088-2091); `ffn_moe_probs_biased` = probs + exp_probs_b (selection
  only; exp_probs_b_vl for an embd batch, deepseek41.cpp 887-892).
- `ffn_moe_argsort` [NE, T] i32, `ffn_moe_topk` [K, T] = its first K (the view keeps the argsort's row stride).
- `ffn_moe_weights` [1, K, T] = probs at the selected experts (unbiased); `ffn_moe_weights_sum` [1, T];
  `ffn_moe_weights_sum_clamped` = max(sum, 6.1035156e-5); `ffn_moe_weights_norm` [K, T] = weights / sum;
  `ffn_moe_weights_scaled` [1, K, T] = norm * 1.5.

### 7. Experts (llama-graph.cpp 2237-2435, 1783-1982)

- Routed: `ffn_moe_up`, `ffn_moe_gate` [FF, K, T] = mul_mat_id of the selected experts; `ffn_moe_swiglu_limited` =
  `silu(min(gate, 10)) * clamp(up, -10, 10)` (ggml_swiglu_clamp, ops.cpp 3408-3465); `ffn_moe_down` [E, K, T];
  `ffn_moe_weighted` = down * weights; `ffn_moe_out` [E, T] = the K slots added in slot order 0..K-1.
- Shared: `ffn_up`, `ffn_gate` [FF, T], `ffn_swiglu_limited` with swiglu_clamp_shexp (1864-1880), `ffn_shexp` [E, T]
  = ffn_down_shexp x it.
- `ffn_out` [E, T] = ffn_moe_out + ffn_shexp, then `l_last` (section 3).

### 8. Head (deepseek41.cpp 928-947)

Only the output rows (the last token here): `hc_out` [E, 1] = collapse(stream, pre of the last layer's FFN mixes),
with the unfused form (il = -1, deepseek4.cpp 296-310); `result_norm` = rms * output_norm; `result_output` [NV, 1] =
output x result_norm (q6_K).

## Numerics on the fork's CPU path

Matmuls convert the activation to the weight type's `vec_dot_type` (ggml-cpu.c type traits): q2_K, q3_K and q6_K
weights take Q8_K activations, bf16 (the router) bf16, f16 (the KV cache in `kq` and `kqv`) f16, f32 f32. So the
query is rounded to f16 against the cached keys and the attention probabilities to f16 against V. The f16 x f16
products of `idx_score` (and `kq`) then sum in an order that depends on the ubatch (found on the dumps): with 2 or
more tokens llamafile's tinyBLAS runs (one 8-lane FMA accumulator per output, then hsum; sgemm.cpp 3966-3973), with a
single token `llamafile_sgemm` refuses (`n < 2`, sgemm.cpp 3818-3822) and `ggml_vec_dot_f16` runs (four 8-lane
accumulators over steps of 32; vec.cpp 345-362). `sum_rows` over the indexer heads adds in double
(`ggml_vec_sum_f32`). The engine's CPU reference (`src/dsv41/engine/indexer_ref.cpp`) follows both orders and gives
the dumps' `idx_score` bit for bit. The engine's
kernels will not reproduce these roundings bit for bit; each parity test states its tolerance (the aim: bit-exact or a
documented tolerance per op). The tiny models' dumps were taken with `-fa off` (flash attention shows no `kq` and
`kq_soft_max`); the real-model oracle (the stock fork's dsv41-moe-dump on an A100) ran with flash attention on the GPU.

The rest of the attention, as the attention's CPU reference reproduces it bit for bit (`src/dsv41/engine/fork_cpu.hpp`,
`attention_ref.cpp`; `dsv41_engine_attention_test --cases`):
- q2_K matmuls: each activation row is quantized to Q8_K by `quantize_row_q8_K_ref` (ggml-quants.c, built without
  -mfma), then `ggml_vec_dot_q2_K_q8_K`'s AVX2 branch (arch/x86/quants.c 1574-1650) accumulates per block, in 8 float
  lanes, first `fma(-d8 * dmin, mins . bsums, .)` then `fma(d8 * d, sumi, .)` (sumi exact in int32), and ends with
  `hsum_float_8`. No tinyBLAS, no repack (q2_K repacks only on AVX512), no IQ panel (IQ types only) on this CPU.
- Rope (ops.cpp 5951-6235): ggml-cpu is built gnu++17 with -mfma, so GCC contracts. Its disassembly
  (`rope_flt<float>` in libggml-cpu.so) and the dumps give: `theta = fma(theta_interp, 1 - ramp_mix, theta_extrap *
  ramp_mix)`, the YaRN magnitude `mscale * fma(0.1f, logf(1 / freq_scale), 1)`, cos and sin from one `sincosf`, and
  per pair `x0' = fma(x0, cos, -(x1 sin))`, `x1' = fma(x0, sin, x1 cos)`. libllama (`dsv4_rope_attn_factor`) has no
  -mfma and fuses nothing. The cache's cos/sin run over the row's full width but only the first n_rot / 2 pairs are
  used, which equals a cache of n_rot values.
- `llama_mul_mat_hadamard` never multiplies on the CPU: the hint routes it to `ggml_compute_forward_fwht` (x / sqrt(n),
  then butterflies u + v, u - v; the SIMD passes' `fma(v, -1, u)` rounds as u - v).
- soft_max (5584-5694): `x * scale + mask`, the max with the sink, `ggml_v_expf` over groups of 8 each summed by the
  hsum into a double (a -inf input takes its special path and gives 0), `expf` for a tail and for the sink, then
  `* (float) (1 / sum)`. The compressor's softmax over r = 2 rows is all tail (libm `expf`); its weighted sum is
  `sum_rows`, a double sum.
- `kqv` runs over the fork's whole k_all (W window cells then C compressed cells) with masked cells at probability 0,
  so the lane a cell lands in depends on its index: the reference needs the fork's layout (in these dumps cell =
  position: 72 positions never evict from the 256 cells) to be bit for bit, and is exact with it.
- A ubatch that completes no compressor block computes a dummy block into the last cache cell, masked (605-633):
  nothing of the engine's corresponds to it.

The dense ops of sections 1, 2, 3, 6, 7 (shared) and 8, as the dense ops' reference reproduces them bit for bit
(`src/dsv41/engine/dense_ref.cpp`, `fork_cpu.hpp`; `dsv41_engine_dense_test --cases`, every op from the dump's inputs):
- The fused hc ops are contracted by GCC: DSV4_HC_PRE sums `fma(x_s, w_s, sum)` from `fma(x_0, w_0, 0)`, DSV4_HC_POST
  `fma(x_s, comb[d, s], y post[d])` for s = 0..3, DSV4_HC_COMB starts from `fma(m, scale[2], base)`, normalizes the
  softmax with `fma(e, 1 / sum, eps)`, and its row and column sums start at eps. pre and post are separate ggml ops
  (mul, add, sigmoid `1 / (1 + expf(-x))`, then + eps or x 2), each rounded. The head's collapse (il = -1) is the
  unfused form: each product rounded, then added in copy order.
- rms norms: the squares summed in double, `1 / sqrtf(mean + eps)` in float, `(x * scale) * w`; the engram's two
  grouped norms the same per copy, its dot a `sum_rows` (double), then `* (1 / sqrtf(E))`, clamp, sqrt, sgn, sigmoid,
  and `x + value * gate` as a mul then an add.
- q3_K and q6_K weights take Q8_K activations; `ggml_vec_dot_q3_K_q8_K` and `_q6_K_q8_K` (AVX2) accumulate per block
  8 int32 lanes, lane k holding bytes 4k..4k+3 of every 32-value chunk times its 16-value scale (q6_K subtracts
  `32 (bsums[2k] sc[2k] + bsums[2k+1] sc[2k+1])` from lane k), then `fma(y.d * d, lane, acc)`, then hsum_float_8.
- The router's bf16 matmul rounds the activation to bf16 (GGML_FP32_TO_BF16) and runs tinyBLAS for 2 or more tokens
  (one 8-lane accumulator per output) and `ggml_vec_dot_bf16` for one (four 8-lane accumulators, `(0 + 2) + (1 + 3)`,
  hsum); a bf16 product is exact in f32. softplus is `x > 20 ? x : logf(1 + expf(x))`; the argsort is `std::sort`
  with `a > b` (its order among exact ties is std::sort's); `sum_rows` and the clamp, div and scale follow.
- engram.token_map (129,280 entries) and the other engram arrays must be read in full; the rows are i32
  (`rolling % prime + offset`), dequantized by `dequantize_row_q2_K` like token_embd's.

### Engine numerics and fork numerics

Two numerics, two places. The fork's numerics (Q8_K activations into every q2_K matmul, the orders above) live only
in the CPU reference (`RefNumerics::Fork`), for the bit-exact op tests of (a). The engine's GPU path keeps f16
activations with f32 or double sums (`RefNumerics::Engine`, f16 weights dequantized from the same q2_K): it is closer
to the reference model, which is not the fork, and it suits the tensor cores. The GPU path does not emulate Q8_K, so
the CUDA test (b) checks the GPU against the reference in engine numerics, stage by stage with stated tolerances, and
never against the fork bit for bit.

What that costs against the fork, measured on tiny-a and tiny-b (`dsv41_engine_attention_test --cases`, --readers
dense, 2026-10-07):
- with the fork's top-k, attn_out per layer and ubatch lands within 1.9% to 2.5% rel L2 of the fork (worst per dump:
  2.54% tiny-a, 1.91% tiny-b, 2.08% tiny-a short, 2.22% tiny-b short; bound 5e-2 in the test, a semantic error is
  O(1));
- with its own top-k, near-ties in the index scores pick other rows than the fork for 11 of 144 index-source queries
  on tiny-a, 24 of 216 (11%) on tiny-b, 0 of 34 and 2 of 99 on the short dumps, which moves those queries by up to
  0.71 rel L2 (reported, not bounded). Tiny random weights make many near-ties that real weights do not.

Since the quantized matmul work, the projections themselves read the GGUF's q2_K blocks: with int8 activations per 32 values for calls of
up to 8 tokens (MMVQ, ggml-cuda's own arithmetic), and
above that as f16 weights dequantized from the blocks against f16 activations (the default prefill path, amended after
the first dev pod; the fork's MMQ on int8 tiles stays as an option). The f16 rule above holds for every other
activation; `RefNumerics::Engine` with `RefActs::Auto` reproduces the default paths (README.md, "Matmul on the quantized weights"). On the same dumps
attn_out lands within 2.1% to 2.9% rel L2 of the fork with its top-k, and 10 of 144 and 24 of 216 index-source queries
select other rows with their own (the f16 path's 2.54%, 11 and 24 above; MMQ prefill 2.6% to 3.1%, 12 and 31).

Revisited on the real model: its per-layer rel L2 (<= 2e-2) and greedy agreement (>= 90% of 64 tokens).
If that fails because of top-k divergence, the fallback is a Q8_K-emulating indexer-query path (only the indexer's
query projection, cheap) and a remeasure; Q8_K on every projection was discarded (slower, further from the reference
model).

## KV cache (what is stored per token)

`kv_budget.py` (checked against the fork's own allocation for tiny-b: raw/csa/lid 4.50/4.00/1.00 MiB at ctx 2048,
equal) on the real header, f16 cache, ctx 262144, n_ubatch 512:

| Part | Per layer | Bytes |
|---|---|---:|
| raw window, all 40 layers | PAD(128 + 512, 256) = 768 cells x 1024 B, fixed | 30.00 MiB |
| csa: compressed rows, sources 2, 8, 14, 20 | 262144 cells (sized for the finest ratio, 1) x 1024 B each | 1024.00 MiB |
| lid: index keys, owners 2, 8, 14, 20 | 262144 cells x 256 B each | 256.00 MiB |
| pooling state, f32 | ratio rows of kv and score per source and owner | 36,864 B |
| total |  | 1310.04 MiB |

Growth per token: 5120 B (1280 B on each of the 4 source layers, 0 on the other 36), 128 B/token/layer averaged over
40 layers, 1280 MiB at 262,144. The fork gives the ratio 2 sources a full-ratio slot and fills every other row; stored
at each source's own ratio it is 3200 B/token (80 B/token/layer, 800 MiB at 262,144). For scale, dense MLA (512 values
per position per layer) would be 40 KiB/token, 10 GiB. Readers own no storage. The 262K target is therefore bound by
compute (the indexer scores and the readers' attention), not by KV memory.

## Divergences and notes

1. Routing is sqrt(softplus) + bias (gating func 4; `config.json` scoring_func "sqrtsoftplus", topk_method
   "noaux_tc"), not sigmoid + bias as the engine memory summarizes it.
2. **Reuse Mode is not reproduced.** The tech report (section 2.3.1) says a Reuse layer attends with the latest
   top-512 indices of a Full or Reindex layer. The fork computes `top_k` only in index sources and drops it after the
   layer (deepseek41.cpp 748-752, 785-787); readers attend to every visible compressed row of their source, and
   `dsv41_topk_source` is computed (deepseek41.cpp 214-246) but never read. Below 512 visible rows (1024 positions
   at ratio 2, 512 at ratio 1) both select every row and agree; beyond that the fork differs from the reference and
   its readers' cost grows with the context (layers 21-39 except the index sources attend to every position at
   ratio 1). The tiny models' top-k of 8 and 16 make this visible at 64 tokens. The engine implements Reuse Mode by
   default and the fork's dense readers behind `--readers dense` ("Reader modes").
3. The hierarchical candidate indexer (`candidate_source_layer_id` 20, 2048 blocks of 8) is not implemented
   (deepseek41.cpp 32-34). It changes nothing below 16,384 compressed rows; the fork comment says the context is
   capped there instead, but no cap is enforced in the fork's code.
4. CED prefill (report section 2.2): the decoder's global KV comes from layer 20's input only (the encoder's
   output), so a prompt token's decoder computation reaches later tokens only through the decoder layers' window KV.
   Under the fork's math, layer 20 + k (k = 0..19) of a prompt is needed only for its last (19 - k) x 127 + 1
   positions (2414 at layer 20, 1 at layer 39: a query sees 127 earlier window keys, and each decoder layer's keys
   come from the previous layer's output), which also leaves every decoder layer's window KV of the last 127
   positions in place for the next token; for every other position, layers 0-19 plus layer 20's window KV,
   compressor and index keys are exact. The report uses a bounded, approximate replay; the fork computes everything.
5. engram.multipliers reach 2^47 (the real file), not "under 2^37" as the fork comment says (deepseek41.cpp 342); with
   compressed ids under 2^17 the products still fit in u64.
6. On the CPU the fork runs the fused hc pre/comb/post ops; the head's collapse (il = -1) is the unfused mul/add form.
7. ggml_top_k does not order its output: `op_dump.py compare` compares `idx_top_k` and `ffn_moe_topk` as index sets.

## Real-model oracle mapping

`op_dump.py map-oracle $BUILD_ROOT/oracle-dsv41/run-1/dump-manifest.json` (the real-model oracle, the real Q2_K model on an A100
SXM 80 GB, flash attention on): `cur` <- `ffn_norm` (section 3), `topk` <- `ffn_moe_topk`, `weights` <-
`ffn_moe_weights_scaled` (section 6), `out` <- `ffn_moe_out` (section 7), in all 40 layers. `dsv41-engine
--dump-layers` writes the same four under these names (`ffn_norm`, `ffn_moe_topk`, `ffn_moe_weights_scaled`,
`ffn_moe_out`, the routed sum before the shared expert), and `oracle_parity.py` compares them with the oracle per
layer.

The oracle's own numerics (its run's command): `--n-gpu-layers 99` puts every op but the routed experts
of layers 0-25 and the engram table's get_rows on the A100, so its dense part is ggml-cuda's, not the CPU's above:
the q2_K, q3_K and q6_K matmuls on MMQ's int8 tiles for its 31-token prompt ubatch (`ggml_cuda_should_use_mmq` is true
for them on every device with Turing tensor cores, the fork's mmq.cu 266-320) and on MMVQ's q8_1 for each decode
token, the attention on CUDA flash attention (f16 K and V, the mma kernel for D = 512), the router's bf16 matmul on
cuBLAS. The experts of layers 0-25 are the fork's CPU experts (Q8_K), those of 26-39 MMQ and MMVQ. So the CPU
reference's fork numerics are not expected to land on the oracle's cur; `dsv41-layer-ref --numerics fork-cuda`
emulates the oracle's matmul arithmetic (MMQ and MMVQ int8 activations, exact integer sums, qmatmul_ref.hpp) with the
engine's f32 ops elsewhere. Only the MoE path can be fed the oracle's own input per layer (`dsv41-engine
--forced-moe-input`: every layer's MoE input replaced by the oracle's cur): the oracle recorded no stream, so a
layer's attention and hyper-connections are compared only through the next cur, cumulatively.

## Tiny models and op-dump (dev VM, measured)

`tiny_dsv41.py` configs (docstring): a = 4 layers, n_embd 256, 8 experts (2 used), heads 4 x 256; b = 6 layers,
n_embd 512, 32 experts (6 used), heads 8 x 512; both ctx 2048, vocab 512 (no_vocab), window 16, every role of the
table above, the real tensor types (q2_K, q3_K, q6_K, bf16, f32) or `--types f32`. `run-oracles.sh WORK_DIR` on the
dev VM (12 vCPU, CPU only, `-t 4`, cu128 tarball `llama-dsv41-b4d9cb12-cu128-sm80-89` libraries in the cu128 builder
image, no GPU), 17 to 23 s wall over three runs:

| Check | a | b |
|---|---|---|
| 16 greedy tokens, 2 runs | identical, logits fnv1a64 `ff8a07c6ed718701` | identical, `07eb9145cf8a50ae` |
| op-dump 64 + 8 tokens (9 ubatches) | 2115 files, 43 MiB | 3024 files, 136 MiB |
| op_dump.py check | 0 size, 0 shape, 0 missing | 0 size, 0 shape, 0 missing |
| op_dump.py compare, 2 runs | 2115/2115 bit-identical | 3024/3024 bit-identical |

The all-f32 variant of a also loads and generates 16 tokens.
