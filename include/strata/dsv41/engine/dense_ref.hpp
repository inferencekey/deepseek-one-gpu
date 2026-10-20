// include/strata/dsv41/engine/dense_ref.hpp - the CPU reference of dense.hpp (src/dsv41/engine/dense_ref.cpp).
//
// Every matrix is a RefMatrix (attention_ref.hpp) whose RefActs chooses the numerics of its products: RefActs::Fork
// the stock fork's CPU backend (Q8_K activations into ggml_vec_dot_q2_K / q3_K / q6_K_q8_K, bf16 activations into
// tinyBLAS or ggml_vec_dot_bf16, fork_cpu.hpp), the others the engine's qmatmul paths and bf16 router (exact
// products, double sums). The rest is one f32 arithmetic for both: the fork's own op by op order (ops.cpp, its FMA
// contractions as GCC -mfma makes them; the rms norms' squares in double), which the GPU kernels also follow, so
// the GPU differs from the reference only by its matmul sums, the norms' sum order and its expf / logf / sqrtf.
// The one semantic difference is the router's order among exactly tied experts: RefNumerics::Fork sorts as the
// fork's argsort (std::sort, descending), RefNumerics::Engine takes the lower expert first as the GPU does.
//
// Shapes are row-major with the token slowest: the stream [T][HC][E], mixes [T][24], pre and post [T][HC], comb
// [T][HC][HC] in the fork's order (src slowest: comb[t][src][dst]).
#pragma once

#include "strata/dsv41/engine/attention_ref.hpp"
#include "strata/dsv41/engine/dense.hpp"

#include <cstdint>
#include <vector>

namespace strata::dsv41::engine {

struct RefHcWeights {
    RefMatrix fn;                  ///< [kHcMix][HC * E]
    const float* base = nullptr;   ///< [kHcMix]
    const float* scale = nullptr;  ///< [3]
};
struct RefEngramWeights {
    RefMatrix wkv;                 ///< [E (HC + 1)][cols * key]
    const float* q = nullptr;      ///< [HC][E] f32 (dequantized)
    const float* k = nullptr;
};
struct RefRouterWeights {
    RefMatrix gate_inp;            ///< [n_expert][E] bf16
    const float* bias = nullptr;   ///< exp_probs_b [n_expert]
};
struct RefSharedWeights {
    RefMatrix gate, up, down;
};

// ---- 1. embedding ----
/// embd [T][E]: the token_embd rows (q2_K) dequantized (ggml's get_rows).
void ref_embed(const RefMatrix& token_embd, const int32_t* tokens, int64_t T, float* embd);

// ---- 3. hyper-connections ----
/// hc_in [T][HC * E] = rms(x) without a weight (ggml_rms_norm over the flattened copies), then mixes = fn . hc_in.
void ref_hc_norm(const DenseDims& d, const float* x, int64_t T, float* hc_in);
void ref_hc_mixes(const DenseDims& d, const RefHcWeights& w, const float* x, int64_t T, float* mixes);
/// pre, post, comb from given mixes: pre = sigmoid(m s0 + b) + eps, post = 2 sigmoid(m s1 + b) (separate ggml ops:
/// each product, sum and sigmoid rounded), comb = the fused DSV4_HC_COMB (ops.cpp 11110-11226: fma(m, s2, b), the
/// softmax over dst with expf and fma(v, 1 / sum, eps), then the Sinkhorn rounds, each sum starting at eps).
void ref_hc_coeffs(const DenseDims& d, const RefHcWeights& w, const float* mixes, int64_t T, float* pre, float* post,
                   float* comb);
/// out [T][E] = sum_s x[t][s][e] pre[t][s]: fused (DSV4_HC_PRE, the sublayers: an FMA chain from x_0 pre_0) or not
/// (the head, il = -1: each product rounded, then added in copy order).
void ref_hc_collapse(const float* x, const float* pre, int64_t T, int E, bool fused, float* out);
/// out [T][HC][E] = y[t][e] post[t][d] then + comb[t][s][d] x[t][s][e] for s = 0..3 by FMA (DSV4_HC_POST).
void ref_hc_expand(const float* y, const float* x, const float* post, const float* comb, int64_t T, int E, float* out);
/// y [rows][n] = rms(x) * w (ggml_rms_norm then ggml_mul).
void ref_rms_norm(const float* x, const float* w, int64_t rows, int n, float eps, float* y);

// ---- 2. engram ----
/// emb [n][key] = the rows (q2_K, key / 256 blocks each) of a table, dequantized.
void ref_engram_emb(const uint8_t* table, int key, const int32_t* rows, int64_t n, float* emb);
/// kv [T][(HC + 1) E] = wkv . emb, then gate and out from it (ref_engram_gate).
void ref_engram(const DenseDims& d, const RefEngramWeights& w, const float* emb, const float* x, int64_t T, float* kv,
                float* gate, float* out);
/// gate [T][HC] and out [T][HC][E] = x + value gate from a given kv (build_engram's ops one by one: rms then times
/// the weight, the products summed in double, times 1 / sqrtf(E), the clamp, sqrt, sign, sigmoid, then mul and add).
void ref_engram_gate(const DenseDims& d, const RefEngramWeights& w, const float* kv, const float* x, int64_t T,
                     float* gate, float* out);

// ---- 6. router ----
struct RefRouterTrace {
    std::vector<float> logits, probs, biased;        ///< [T][n_expert]
    std::vector<int32_t> order;                      ///< [T][n_expert] the descending order of biased (the argsort)
    std::vector<int32_t> ids;                        ///< [T][K] its first K
    std::vector<float> weights, norm, scaled;        ///< [T][K] unbiased probs, normalized, scaled
    std::vector<float> sum, sum_clamped;             ///< [T]
};
void ref_router(RefNumerics n, const DenseDims& d, const RefRouterWeights& w, const float* xn, int64_t T,
                RefRouterTrace& tr);
/// The same from given logits [T][n_expert] (tr.logits).
void ref_router_select(RefNumerics n, const DenseDims& d, const float* bias, int64_t T, RefRouterTrace& tr);

// ---- 7. shared expert ----
struct RefSharedTrace {
    std::vector<float> gate, up, act, out;   ///< [T][n_ff] x3, [T][E]
};
void ref_shared_expert(const DenseDims& d, const RefSharedWeights& w, const float* xn, int64_t T, RefSharedTrace& tr);
/// act = silu(min(gate, L)) clamp(up, -L, L), ops.cpp 3408-3465 (gate / (1 + expf(-gate)) * up).
void ref_swiglu_clamp(const float* gate, const float* up, int64_t n, float limit, float* act);

// ---- 8. head ----
/// hc_out [n][E] (unfused collapse), norm [n][E] = rms * output_norm, logits [n][n_vocab] = output . norm.
void ref_head(const DenseDims& d, const float* x, const float* pre, int64_t n, const float* output_norm,
              const RefMatrix& output, float* hc_out, float* norm, float* logits);

}  // namespace strata::dsv41::engine
