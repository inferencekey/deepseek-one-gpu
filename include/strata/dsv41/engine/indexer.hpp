// include/strata/dsv41/engine/indexer.hpp - the DeepSeek-V4.1-Flash lightning indexer: for every query of a
// decode token or a prefill ubatch, the index score of each visible compressed index key and the top-k rows, without
// the fork's materialized [n_comp, n_tokens, heads] score tensor.
//
// The math is the fork's (tools/dsv41/engine/SPEC.md, section 5, "Indexer"; deepseek41.cpp 503-590): for query t at
// position p and compressed row j,
//
//   score[t, j] = sum_h relu(q[t, h, :] . k[j, :]) * w[t, h],   visible when j < (p + 1) / ratio,
//
// with q the rotated indexer query (`idx_q_rot`, the Hadamard already applied), k the key owner's cached index keys
// (`idx_k`, f16, rotated before they were stored), w the per-head weights (`idx_weights`, already scaled by
// 1 / sqrt(dim * heads)). The fork's mul_mat rounds q to f16 against the f16 key cache (vec_dot_type of f16); every
// path here does the same (round to nearest even), so the products are exact in f32 and the paths differ from the
// fork only in their summation order. The top-k is a set (the fork's ggml_top_k is unordered): here the visible rows
// with the largest scores, ties to the lowest index, written ascending.
//
// Starting point: Strata's QSA select (src/kernels/cuda/qsa_select.cu). The
// top-k is its radix select (block_topk_wide_kernel: four 8-bit passes, per-warp histograms, ties to the lowest index)
// with every row a cell of its own. The tensor-core scorer keeps block_scores_tc_kernel's structure (a query tile per
// CTA, a padded key tile in shared memory, mma.sync, relu per head summed in the epilogue) on f16 m16n8k16 instead of
// 3xTF32 m16n8k8, because the fork's inputs are f16 and one f16 product is already exact. The FP32 scorer is one
// thread per key instead of qsa_block_scores' warp per key: at 32 heads the warp form spends 5 shuffles per head and
// key (160 per key against 128 FMAs per lane), the thread form reads the query from shared memory as broadcasts.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::dsv41::engine {

/// The real model's indexer (GGUF attention.indexer.*, SPEC.md "Constants").
constexpr int kIndexerHeads = 32;
constexpr int kIndexerDim = 128;
constexpr int kIndexerTopK = 512;
/// Largest top-k the selection kernel takes (shared counters are int; this bounds the output rows).
constexpr int kIndexerMaxTopK = 4096;
/// Default cap of the score scratch: 256 query rows at 262,144 keys (ratio 1 at the 262K target).
constexpr size_t kIndexerScratchCap = size_t(256) << 20;

/// One index source layer's geometry. The key dim is always kIndexerDim (128 in the real model and both tiny ones).
struct IndexerDims {
    int n_heads = kIndexerHeads;   ///< 1..32 (tiny-a 4, tiny-b 8, real 32)
    int top_k = kIndexerTopK;      ///< 1..kIndexerMaxTopK (tiny-a 8, tiny-b 16, real 512)
    int ratio = 1;                 ///< the layer's compress ratio, >= 1 (the real model: 2 or 1)
};

enum class IndexerScorer {
    Auto,         ///< tensor cores from kIndexerTcMinQueries queries on a sm_80+ device, otherwise Simt
    Simt,         ///< FP32 FMA, one thread per (query, key): the decode path
    TensorCore,   ///< f16 mma.sync m16n8k16 with FP32 accumulation, 8 queries per CTA: the prefill path
};
/// Below this many queries the tensor-core CTA (one query per warp, 8 warps) would leave warps idle.
constexpr int64_t kIndexerTcMinQueries = 8;

/// Rows query position `pos` sees: min(n_k, (pos + 1) / ratio), the compressed causal mask of llama-graph.cpp 764-806.
inline int64_t indexer_visible(int64_t pos, int ratio, int64_t n_k) {
    const int64_t v = pos < 0 ? 0 : (pos + 1) / ratio;
    return v < n_k ? v : n_k;
}

/// How indexer_select splits nq queries over n_k keys: q_chunk queries per pass through a [q_chunk, ld] f32 scratch.
struct IndexerPlan {
    int64_t q_chunk = 0;       ///< queries per pass, >= 1
    int64_t ld = 0;            ///< floats per scratch row: n_k rounded up to 32 (128-byte rows)
    size_t scratch_bytes = 0;  ///< q_chunk * ld * 4, at most the cap
};
/// Exits naming the values when a single query's row (ld * 4 bytes) does not fit `scratch_cap`.
IndexerPlan indexer_plan(int64_t nq, int64_t n_k, size_t scratch_cap = kIndexerScratchCap);

// ---- CUDA (device pointers, `stream` a cudaStream_t or nullptr) ----
// q [nq, n_heads, 128] f32; w [nq, n_heads] f32; pos [nq] i32 (device, so a captured decode graph can move it);
// keys [n_k, 128] f16 bits, 16-byte aligned; scores [nq, ld] f32; ids [nq, top_k] i32.

/// Scores of nq queries: scores[t * ld + j] for j < indexer_visible(pos[t]); the rest of the row is not written.
void indexer_scores(const float* q, const float* w, const int32_t* pos, int64_t nq, const uint16_t* keys, int64_t n_k,
                    const IndexerDims& d, float* scores, int64_t ld, IndexerScorer which, void* stream);

/// Top-k of scored rows: ids[t * top_k + i] for i < min(top_k, visible) are the selected rows ascending, the rest -1.
void indexer_topk(const float* scores, int64_t ld, const int32_t* pos, int64_t nq, int64_t n_k, const IndexerDims& d,
                  int32_t* ids, void* stream);

/// Scores and top-k of nq queries in plan.q_chunk passes through `scratch` (plan.scratch_bytes, from indexer_plan with
/// the same n_k and nq >= this call's).
void indexer_select(const float* q, const float* w, const int32_t* pos, int64_t nq, const uint16_t* keys, int64_t n_k,
                    const IndexerDims& d, const IndexerPlan& plan, float* scratch, IndexerScorer which, int32_t* ids,
                    void* stream);

/// The scorer `which` resolves to for nq queries on the current device.
IndexerScorer indexer_resolve(IndexerScorer which, int64_t nq);

// ---- CPU reference (src/dsv41/engine/indexer_ref.cpp): the fork's CPU backend, bit for bit ----
// The fork runs idx_score as mul_mat (q rounded to f16; in a ubatch of >= 2 tokens llamafile's tinyBLAS, one 8-lane
// FMA accumulator per output, in a single-token ubatch ggml_vec_dot_f16, four), relu, mul by the weights, and
// sum_rows over the heads in double (ggml_vec_sum_f32). This reproduces that order exactly; the dump check compares
// it bit for bit.

/// scores[j], j < n_rows, of one query in a fork ubatch of `ubatch_tokens` tokens (which picks the fork's dot):
/// q [n_heads, 128], w [n_heads], keys [n_rows, 128] f16 bits.
void indexer_ref_scores(const float* q, const float* w, int n_heads, const uint16_t* keys, int64_t n_rows,
                        int64_t ubatch_tokens, float* scores);
/// The same rule as indexer_topk on one query's visible scores: min(top_k, visible) ids ascending, the rest of
/// ids[0, top_k) set to -1. Ties at the threshold go to the lowest index; NaN orders below every number.
void indexer_ref_topk(const float* scores, int64_t visible, int top_k, int32_t* ids);
/// Per (query, row) bound scale of the scores' rounding: sum_h |w_h| sum_d |q_d k_d| with q rounded to f16 (the sum
/// of the absolute terms, in double). Two paths that only order the sums differently differ by at most about
/// 2 * (dims + heads) * 2^-24 of this.
void indexer_ref_abs_scale(const float* q, const float* w, int n_heads, const uint16_t* keys, int64_t n_rows,
                           double* scale);

}  // namespace strata::dsv41::engine
