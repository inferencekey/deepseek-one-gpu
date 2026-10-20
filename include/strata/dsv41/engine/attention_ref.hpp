// include/strata/dsv41/engine/attention_ref.hpp - the CPU reference of one attention layer
// (src/dsv41/engine/attention_ref.cpp): the engine's semantics (attention.hpp: the window, the compressor and its
// carried state, the index keys and the indexer, both reader modes, the grouped output) in one of two numerics.
//
//   RefNumerics::Fork    the stock fork's CPU backend, bit for bit where the inputs and the cell layout are the
//                        fork's: q2_K weights against Q8_K activations (ggml_vec_dot_q2_K_q8_K), queries and
//                        probabilities rounded to f16 against the f16 cache in tinyBLAS (ubatch >= 2) or
//                        ggml_vec_dot_f16 (one token) order, ggml's soft_max (ggml_v_expf, sinks), rms_norm, rope
//                        with the fork's FMA contractions, the FWHT of llama_mul_mat_hadamard (fork_cpu.hpp). Checked
//                        against the op dumps by dsv41_engine_attention_test --cases.
//   RefNumerics::Engine  the GPU path's math with one rounding fewer per stage: a quantized weight (RefActs other
//                        than Fork) against the engine's int8 activations as qmatmul quantizes them (qmatmul_ref.hpp,
//                        exact products), an f16 weight (the cuBLAS test path) against f16 activations, sums in
//                        double; attention in double over the cells the engine attends (no f16 rounding of the
//                        probabilities). The GPU differs from it by summation order and the device's sincosf/exp2f;
//                        dsv41_engine_attention_test --gpu bounds that per stage.
//
// Where the fork's order needs the fork's cell layout (the attention sums run over its k_all: W window cells, cell
// = position while the sequence fits, then C compressed cells), RefForkLayout gives it; without one the reference
// attends the engine's cell list (the window positions, then the compressed rows ascending).
#pragma once

#include "strata/dsv41/engine/attention.hpp"

#include <cstdint>
#include <vector>

namespace strata::dsv41::engine {

enum class RefNumerics { Fork, Engine };

/// How a quantized weight's activations are quantized: Fork, the fork's Q8_K per 256 (ggml_vec_dot_q2_K_q8_K, q2_K
/// only); otherwise the engine's (qmatmul_ref.hpp): Auto the path qmatmul takes for the matmul's row count under the
/// default QPrefill::DequantF16, the others that layout (a GPU test recomputing sampled rows of a larger call), F16
/// the dequant path (qref_matmul_f16: f16 weights and activations).
enum class RefActs { Fork, Auto, Q8_1, MmqD4, MmqD2S6, F16 };

/// A weight as the reference reads it: ne [k, rows] (k fastest), q2_K / q3_K / q6_K blocks (the fork's Q8_K dot with
/// RefActs::Fork; the engine's with any other), f16 bits (type 1, the engine's f16 test path) or bf16 bits (type 30,
/// the router's ffn_gate_inp: with RefActs::Fork the fork's bf16 activations in tinyBLAS (2 or more tokens)
/// or ggml_vec_dot_bf16 (one token) order; with any other the engine's, bf16 activations, exact products, a double
/// sum).
struct RefMatrix {
    int type = 0;
    int64_t k = 0, rows = 0;
    const void* data = nullptr;
    RefActs acts = RefActs::Fork;
};
constexpr int kRefTypeF16 = 1;
constexpr int kRefTypeQ2K = 10;
constexpr int kRefTypeQ3K = 11;
constexpr int kRefTypeQ6K = 14;
constexpr int kRefTypeBF16 = 30;

struct RefLayerWeights {
    RefMatrix wq_a, wq_b, wkv, wo_a, wo_b, comp_kv, comp_gate, idx_k, idx_q_b, idx_proj;
    const float* q_a_norm = nullptr;
    const float* kv_norm = nullptr;
    const float* comp_norm = nullptr;
    const float* idx_k_norm = nullptr;
    const float* sinks = nullptr;
};

/// y [T][rows] = x [T][k] times w^T in w's numerics: RefActs::Fork the fork's (each row of x quantized to Q8_K for
/// a K-quant, rounded to bf16 for bf16), the other RefActs the engine's qmatmul (qref_quantize, qref_matmul) or bf16
/// path, f16 the engine's test path (x rounded to f16, double sums). T is the call's token count, which picks the
/// fork's bf16 kernel.
void ref_matmul(const RefMatrix& w, const float* x, int64_t T, float* y);
/// The rows of a q2_K matrix as f32 (dequantize_row_q2_K), then as f16 bits: the engine's device weights.
std::vector<float> ref_dequant(const RefMatrix& w);
std::vector<uint16_t> ref_to_f16(const std::vector<float>& v);

/// One sequence's KV state as the engine keeps it, per layer (host). The reference keeps every window position.
struct RefKv {
    struct Layer {
        int64_t next_pos = 0;               ///< the position the layer's next chunk starts at
        std::vector<uint16_t> window;       ///< [positions][head_dim] f16, position p at row p
        std::vector<uint16_t> rows;         ///< KV sources: [rows][head_dim] f16 compressed rows, row j at j
        std::vector<uint16_t> keys;         ///< key owners: [rows][idx_dim] f16 index keys (rotated)
        std::vector<float> carry_kv, carry_score;   ///< KV sources: [n][head_dim], positions floor(p/r) r .. p - 1
        std::vector<int32_t> ids;           ///< index sources: the last chunk's top-k [T][top_k], -1 padded
        int64_t ids_p0 = -1, ids_T = 0;
    };
    std::vector<Layer> layers;
    explicit RefKv(size_t n_layers = 0) : layers(n_layers) {}
};

/// The fork's k_all layout of one call: W window cells (cell c holds position c) then C compressed cells.
struct RefForkLayout {
    int64_t W = 0, C = 0;
};

struct RefOptions {
    RefNumerics numerics = RefNumerics::Fork;
    ReaderMode mode = ReaderMode::Dense;
    const RefForkLayout* layout = nullptr;   ///< the fork's cells (Fork numerics); null: the engine's cell list
    /// index sources: attend (and hand readers) this top-k [T][topk_override_k] instead of the reference's own, in
    /// the fork's form (any order, rows at or past the visible count ignored); null: the own one
    const int32_t* topk_override = nullptr;
    int64_t topk_override_k = 0;
};

/// Every stage of one call, row-major, for the parity tests.
struct RefTrace {
    std::vector<float> qr, q, kv, ckv, cscore;            ///< [T][..] as attention.hpp's AttnScratch
    std::vector<float> gathered_kv, gathered_score;      ///< [B][r][head_dim] the completed blocks' inputs
    std::vector<float> pooled, latent, rows;             ///< [B][head_dim]: pooled, normed, roped (pre-f16)
    std::vector<float> idx_k_new, idx_kf;                ///< [B][idx_dim] roped; then rotated (pre-f16)
    std::vector<float> iq, iq_rot, iw;                   ///< [T][idx_heads][idx_dim], [T][idx_heads]
    std::vector<float> scores;                           ///< [T][n_k] index scores of the visible rows (else -inf)
    std::vector<int32_t> topk;                           ///< [T][top_k] the reference's own selection
    int64_t n_k = 0;                                     ///< the key source's rows at the end of the chunk
    std::vector<float> mask, kq, kq_soft_max;            ///< fork layout only: [T][W + C], [H][T][W + C]
    std::vector<float> attn, derope, oa, out;            ///< [T][H][D], [T][H][D], [T][OG * OL], [T][E]
    int64_t B = 0, first_block = 0;
};

/// Layer l over T tokens at positions p0.. (x [T][n_embd] the attention input), updating kv; out [T][n_embd].
void ref_attn_forward(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, RefKv& kv, const float* x,
                      int64_t T, int64_t p0, const RefOptions& o, float* out, RefTrace* trace);

// ---- the stages, for the GPU parity test (each from its own inputs) ----

/// qr [T][q_lora] = rms(x wq_a) * q_a_norm, q [T][H][D] = rope(qr wq_b) at pos.
void ref_stage_q(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* x, int64_t T,
                 const int32_t* pos, float* qr, float* q);
/// q [T][H][D] = rope(qr wq_b) from a given qr.
void ref_stage_q_from_qr(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* qr, int64_t T,
                         const int32_t* pos, float* q);
/// kv [T][D] = rope(rms(x wkv) * kv_norm).
void ref_stage_kv(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* x, int64_t T,
                  const int32_t* pos, float* kv);
/// pooled, latent, rows [B][D] from the blocks' inputs [B][r][D] (block b is block first_block + b).
void ref_stage_pool(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, RefNumerics n, const float* gkv,
                    const float* gscore, int64_t B, int64_t first_block, float* pooled, float* latent, float* rows);
/// index keys [B][idx_dim]: rope(rms(latent attn_k) * k_norm) (idx_k_new) and its Hadamard (idx_kf).
void ref_stage_index_keys(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* latent,
                          int64_t B, int64_t first_block, float* idx_k_new, float* idx_kf);
/// iq [T][IH][128] roped (iq) and rotated (iq_rot), iw [T][IH] from qr and x.
void ref_stage_index_query(const AttnDims& d, const RefLayerWeights& w, const float* qr, const float* x, int64_t T,
                           const int32_t* pos, float* iq, float* iq_rot, float* iw);
/// One token's attention over n cells (keys f16 [n][D] by pointer, mask[c] 0 or -inf or null): out [H][D]. Fork
/// numerics take the fork's cell order and ubatch size (the dot it uses); Engine numerics sum in double. kq and sm,
/// when not null, get [H][n] scores and probabilities (Fork numerics).
void ref_stage_attend(const AttnDims& d, RefNumerics n, const float* q, const uint16_t* const* keys, const float* mask,
                      int64_t n_cells, const float* sinks, int64_t ubatch_tokens, float* out, float* kq, float* sm);
/// The absolute-term scale of ref_stage_attend's Engine sums: scale[h][dim] = sum_c p_c |k_c[dim]| (the bound the GPU's
/// rounding is measured against).
void ref_stage_attend_scale(const AttnDims& d, const float* q, const uint16_t* const* keys, int64_t n_cells,
                            const float* sinks, double* scale);
/// derope [T][H][D] (de-roped attn), oa [T][OG * OL], out [T][E] from attn [T][H][D].
void ref_stage_output(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* attn, int64_t T,
                      const int32_t* pos, float* derope, float* oa, float* out);
/// oa from a given derope (f16 [T][H*D] as the GPU feeds wo_a), out from a given oa.
void ref_stage_wo_a(const AttnDims& d, const RefLayerWeights& w, const float* derope, int64_t T, float* oa);

/// The absolute-term scale of an Engine-numerics matmul: f16 scale[t][r] = sum_k |f16(x[t][k]) w[r][k]|, quantized the
/// sum of the absolute values of qref_matmul's terms.
void ref_matmul_abs_scale(const RefMatrix& w, const float* x, int64_t T, double* scale);

}  // namespace strata::dsv41::engine
