// include/strata/dsv41/engine/dense.hpp - every DeepSeek-V4.1-Flash op outside the attention and the routed experts
// on the GPU: the embedding and the stream, the hyper-connection mixes, collapse and expansion of each
// sublayer, the rms norms, the engram, the router, the shared expert and the head, for a prefill chunk of T tokens or
// a decode token (T = 1), on one stream, with plain device pointers like attention.hpp.
//
// The math is the fork's (tools/dsv41/engine/SPEC.md sections 1, 2, 3, 6, 7 shared part, 8). The stream x of a
// token is HC = 4 copies of E values, row-major [T][HC][E] (ggml's [E, HC, T]). Per layer l (forward.hpp assembles them):
//
//   engram (layers in engram.layer_ids, before the block): x += value * gate, engram.hpp
//   attention sublayer:  (pre_a, post, comb) = hc_mixes(x, hc_attn);  h = rms(collapse(x, pre_f of layer l - 1)) *
//                        attn_norm  (hc_pre_init, one-hot copy 0, for l = 0);  x = expand(attn(h), x, post, comb)
//   FFN sublayer:        (pre_f, post, comb) = hc_mixes(x, hc_ffn);  h = rms(collapse(x, pre_a)) * ffn_norm;
//                        x = expand(moe(h) + shared(h), x, post, comb)
//   head:                rms(collapse(x, pre_f of the last layer)) * output_norm, then the output matrix
//
// where hc_mixes is the hc_fn matmul of rms(x over HC * E) (no weight), then pre = sigmoid(m[0:4] s0 + b) + eps,
// post = 2 sigmoid(m[4:8] s1 + b), comb = Sinkhorn(m[8:24] s2 + b) (a softmax over dst per src plus eps, a column
// normalization, then sinkhorn_iters - 1 rounds of row and column normalizations, each sum plus eps), collapse
// out[e] = sum_s x[s][e] pre[s], expansion out[d][e] = y[e] post[d] + sum_s comb[d][s] x[s][e].
//
// Numerics: every quantized matrix (hc_fn, the
// engram's wkv, the shared expert, the output head; q2_K, q3_K, q6_K as stored) goes through qmatmul.hpp, so MMVQ on
// int8 activations for up to 8 tokens and the dequant path (f16 weights and activations, cuBLAS) above; the router's
// bf16 ffn_gate_inp takes its input rounded to bf16 as the fork does (exact products, f32 sums) in a GEMV kernel for
// up to kRouterGemvMaxCols tokens and cuBLAS bf16 above; everything else is f32 with the norms' squares summed in
// double. dense_ref.hpp is the CPU reference in these numerics and in the fork's.
//
// Starting points in Strata: the rms norm is native_gr_norm.cu's weighted kernel
// (with attention.cu's double sum); collapse and expansion are native_gr_postops.cu's pre_gated / post kernels
// rewritten to the fork's DSV4_HC_PRE / DSV4_HC_POST math, and the Sinkhorn is the fork's ggml-cuda dsv4-hc.cu
// comb kernel (whose CPU form the reference follows); the router is native_router.cu's warp-per-token top-k with
// sqrt(softplus), the selection bias and 384 / 6 in place of softmax and 512 / 10; the shared expert is
// shared_expert.cu's gate / up / down sequence without its sigmoid gate and with the SwiGLU clamp; the embedding and
// the head follow the NativeEmbed / NativeHead pattern (a dequantized row per token, a matmul for the output rows).
#pragma once

#include "strata/dsv41/engine/qmatmul.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

constexpr int kHc = 4;                      ///< hyper_connection.count (the fork asserts 4)
constexpr int kHcMix = (2 + kHc) * kHc;     ///< 24 mixes: pre 0-3, post 4-7, comb 8-23 (dst fastest)
constexpr int64_t kRouterGemvMaxCols = 8;   ///< the router's GEMV up to this many tokens, cuBLAS bf16 above

/// The geometry of everything here (GGUF deepseek41.* keys, SPEC.md "Constants").
struct DenseDims {
    int n_embd = 5120;
    int n_vocab = 129280;
    int n_expert = 384;
    int n_expert_used = 6;
    int n_ff = 2304;               ///< expert_feed_forward_length (the shared expert's width, one shared expert)
    float rms_eps = 1e-20f;        ///< attention.layer_norm_rms_epsilon (every rms norm)
    float hc_eps = 1e-6f;          ///< hyper_connection.epsilon
    int sinkhorn_iters = 20;       ///< hyper_connection.sinkhorn_iterations
    float swiglu_limit = 10.f;     ///< swiglu_clamp_shexp (10 on every layer of the real model)
    bool weights_norm = true;      ///< expert_weights_norm
    float weights_scale = 1.5f;    ///< expert_weights_scale
    int engram_heads = 8;          ///< engram.head_count
    int engram_key = 256;          ///< engram.key_length
    int engram_ngram = 4;          ///< engram.max_ngram_size
    int engram_cols() const { return (engram_ngram - 1) * engram_heads; }   ///< rows gathered per token (24)
};

/// The real DeepSeek-V4.1-Flash (GGUF header, vcruz305 Q2_K c4a08554).
DenseDims dense_dims_real();

/// One sublayer's hyper-connection weights: hc_fn [kHcMix][HC * E] (q2_K in the real model), base [kHcMix] and
/// scale [3] f32 on the device.
struct HcWeights {
    QWeightView fn;
    const float* base = nullptr;
    const float* scale = nullptr;
};

/// The engram of one layer: wkv [E * (HC + 1)][cols * key] (q2_K), engram_q and engram_k as f32 [HC][E] on the
/// device (q2_K in the file, dequantized at load: the fork's get_rows does the same per call).
struct EngramWeights {
    QWeightView wkv;
    const float* q = nullptr;
    const float* k = nullptr;
};

/// The router of one layer: ffn_gate_inp [n_expert][E] bf16 bits, exp_probs_b [n_expert] f32, on the device.
struct RouterWeights {
    const uint16_t* gate_inp = nullptr;
    const float* bias = nullptr;
};

/// The shared expert of one layer: gate and up [n_ff][E] (q2_K), down [E][n_ff] (q3_K).
struct SharedExpertWeights {
    QWeightView gate, up, down;
};

/// The buffers of the last calls, kept so a test can read every stage (row-major, the call's T rows).
struct DenseScratch {
    float* mixes = nullptr;       ///< [T][kHcMix] the last hc_mixes call's matmul output
    float* hc_in = nullptr;       ///< [T][HC * E] rms(x) as f32 (calls on MMVQ: T <= kMmvqMaxCols)
    uint16_t* hc_inh = nullptr;   ///< [T][HC * E] rms(x) as f16 (calls on the dequant path)
    float* collapsed = nullptr;   ///< [T][E] the last hc_collapse_norm's collapse (before its norm)
    float* emb = nullptr;         ///< [Te][cols * key] the engram's dequantized rows (MMVQ calls)
    uint16_t* embh = nullptr;     ///< [Te][cols * key] their f16 (dequant path calls)
    float* engram_kv = nullptr;   ///< [Te][(HC + 1) E] engram_wkv's output
    float* engram_gate = nullptr; ///< [T][HC]
    uint16_t* router_in = nullptr;///< [T][E] the router's input in bf16
    float* logits = nullptr;      ///< [T][n_expert] ffn_moe_logits
    float* probs = nullptr;       ///< [T][n_expert] sqrt(softplus(logits))
    float* sh_gate = nullptr;     ///< [T][n_ff] shared expert gate, up and the clamped SwiGLU
    float* sh_up = nullptr;
    float* sh_act = nullptr;
    float* head_in = nullptr;     ///< [n][E] the head's collapse and its norm
    float* head_norm = nullptr;
    void* qact = nullptr;         ///< qmatmul's activation scratch (int8 layouts or f16)
    uint32_t* hc_ticket = nullptr;///< hc_decode_step's count of finished blocks (0 between launches)
    unsigned long long* hc_marks = nullptr;   ///< hc_decode_step's phase marks [kHcStepGrid][kHcStepMarks]
    int64_t engram_T = 0;         ///< tokens of the engram's last pass (its sub-chunk: Te <= engram_chunk)
};

/// Stage times of the last call of each op family, from CUDA events (DenseEngine::set_timing(true)).
struct DenseTimes {
    float embed_ms = 0, mixes_ms = 0, collapse_ms = 0, expand_ms = 0, step_ms = 0, engram_ms = 0, router_ms = 0,
          shared_ms = 0, head_ms = 0, norm_ms = 0;
};

/// The context of the dense ops for one model: qmatmul's context (its own or the stream's shared one: an f16 weight
/// scratch of at least the shared expert's n_ff x E matrix, the engram's wkv and the output head dequantized in row
/// slices of it), a cuBLAS
/// handle for the router, the scratch for chunks of up to max_T tokens (the engram in sub-chunks of at most
/// engram_chunk tokens), one stream.
class DenseEngine {
public:
    DenseEngine() = default;
    ~DenseEngine();
    DenseEngine(const DenseEngine&) = delete;
    DenseEngine& operator=(const DenseEngine&) = delete;

    /// False with `err` set when the device or a buffer is missing. max_head: output rows per dense_head call.
    /// shared: a qmatmul context of the same prefill path, on the same stream, whose f16 scratch holds the shared
    /// expert's n_ff x E matrix (one dequant scratch per stream); null: the engine makes its own.
    bool init(const DenseDims& d, int64_t max_T, int64_t max_head, int64_t engram_chunk, void* stream, std::string& err,
              QPrefill prefill = QPrefill::DequantF16, QmmContext* shared = nullptr);

    const DenseDims& dims() const { return d_; }
    const DenseScratch& scratch() const { return s_; }
    size_t workspace_bytes() const { return ws_bytes_; }
    void* stream() const { return stream_; }
    int64_t max_T() const { return max_T_; }
    int64_t max_head() const { return max_head_; }
    int64_t engram_chunk() const { return engram_chunk_; }
    void set_timing(bool on) { timing_ = on; }
    const DenseTimes& times() const { return times_; }
    /// hc_decode_step's launches from now on write their phase marks (hc_step_split; a profile's own graphs)
    void set_step_marks(bool on) { step_marks_ = on; }
    bool step_marks() const { return step_marks_; }

    // internals used by the .cu files
    QmmContext& qmm() { return *qmm_; }
    void* blas() const { return blas_; }
    DenseScratch& scratch_mut() { return s_; }
    DenseTimes& times_mut() { return times_; }
    bool timing() const { return timing_; }
    void* event(int i) const { return events_[i]; }

private:
    DenseDims d_{};
    DenseScratch s_{};
    QmmContext own_qmm_;
    QmmContext* qmm_ = &own_qmm_;
    void* blas_ = nullptr;
    void* blas_ws_ = nullptr;
    void* stream_ = nullptr;
    int64_t max_T_ = 0, max_head_ = 0, engram_chunk_ = 0;
    size_t ws_bytes_ = 0;
    bool timing_ = false, step_marks_ = false;
    DenseTimes times_{};
    void* events_[2] = {};
    std::vector<void*> allocs_;
};

// ---- host-only checks (no device access): the call's weight views, as each op checks them before its first launch;
// dsv41_engine_dense_test --gpu --dry-run runs them on the dev VM ----
void hc_check_weights(const DenseDims& d, const HcWeights& w, const char* what);
void engram_check_weights(const DenseDims& d, const EngramWeights& w, const char* what);
void shared_check_weights(const DenseDims& d, const SharedExpertWeights& w, const char* what);
void head_check_weights(const DenseDims& d, const QWeightView& output, const char* what);
void router_check_weights(const DenseDims& d, const RouterWeights& w, const char* what);
void embed_check_weights(const DenseDims& d, const QWeightView& token_embd, const char* what);

// ---- 1. embedding ----
/// x [T][HC][E] = the token_embd row (q2_K, dequantized as ggml's get_rows) of each token in every copy, and
/// pre_init [T][HC] = the one-hot [1, 0, 0, 0] layer 0's attention collapses with. tokens: device [T].
void dense_embed(DenseEngine& e, const QWeightView& token_embd, const int32_t* tokens, int64_t T, float* x,
                 float* pre_init);

// ---- 3. hyper-connections ----
/// pre, post [T][HC] and comb [T][HC (src)][HC (dst)] (the fork's memory order: dst fastest) of the stream x.
void hc_mixes(DenseEngine& e, const HcWeights& w, const float* x, int64_t T, float* pre, float* post, float* comb);
/// normed [T][E] = rms(collapse(x, pre)) * norm_w (attn_norm, ffn_norm); the collapse itself stays in the scratch.
void hc_collapse_norm(DenseEngine& e, const float* x, const float* pre, const float* norm_w, int64_t T, float* normed);
/// out [T][HC][E] = expansion of the sublayer output y [T][E] over the stream x with post and comb. out may be x.
void hc_expand(DenseEngine& e, const float* y, const float* x, const float* post, const float* comb, int64_t T,
               float* out);

/// a decode token's (T = 1) hyper-connection step before a sublayer in one launch: with y, x =
/// hc_expand(y, x, post, comb) first (the previous sublayer's output, written back into x), then hc_mixes of w on x
/// into pre, post and comb (which may be the buffers the expansion read: they are read first) and normed =
/// hc_collapse_norm(x, pre_c, norm_w), and with normed_q its q8_1 (qact_quantize's Q8_1 layout of normed, for the
/// sublayer's matmuls). Every value equals the unfused ops' bit for bit (the same arithmetic in the same order, the
/// scratch's hc_in, mixes and collapsed included); dims the one launch does not take (n_embd other than 5120, 256 and
/// 512, or hc_fn not 16-byte aligned) run those ops. y null: no expansion.
void hc_decode_step(DenseEngine& e, const HcWeights& w, const float* y, float* x, const float* pre_c, const float* norm_w,
                    float* pre, float* post, float* comb, float* normed, void* normed_q);

/// a prefill chunk's step before a sublayer, hc_decode_step's math for T tokens: with y, x = hc_expand(y, x,
/// post, comb) first, then hc_mixes of w on x into pre, post and comb (the expansion's post and comb are read first)
/// and normed = hc_collapse_norm(x, pre_c, norm_w). One launch per token row for the expansion, the stream's norm and
/// the collapse and norm (one read of x), then the hc_fn matmul and the coefficients; every value, the scratch's
/// included, equals the unfused ops' bit for bit. n_embd other than 5120 and 512 runs those ops.
void hc_prefill_step(DenseEngine& e, const HcWeights& w, const float* y, float* x, const float* pre_c, const float* norm_w,
                     int64_t T, float* pre, float* post, float* comb, float* normed);

/// hc_decode_step's grid: six row blocks of four hc_fn rows each, then the norm block (hc.cu); the phase marks per block.
constexpr int kHcStepGrid = 7;
constexpr int kHcStepMarks = 6;

/// where an hc_decode_step's device time goes, from the %globaltimer marks of the last
/// launch made with DenseEngine::set_step_marks on, in microseconds from its first block's start: the row blocks'
/// stream and rms (rms_us), their q8_1 and the hc_fn rows staged (q8_us), the mixes written (mixes_us) and the ticket
/// taken (ticket_us), each at the latest row block; the norm block's collapse, norm and q8_1 written (norm_us); the
/// kernel's end, the last block's expanded stream and coefficients (end_us). The timer ticks every microsecond on
/// some GPUs: a mean over many launches is what reads.
struct HcStepSplit {
    double rms_us = 0, q8_us = 0, mixes_us = 0, ticket_us = 0, norm_us = 0, end_us = 0;
};
/// The last marked launch's split (waits for the engine's stream first).
HcStepSplit hc_step_split(DenseEngine& e);

/// y [rows][n] = rms(x) * w, the squares summed in double. In place is allowed.
void dense_rms_norm(DenseEngine& e, const float* x, const float* w, int64_t rows, int n, float* y);

// ---- 2. engram (the hash and the table gather on the host: engram.hpp) ----
/// x [T][HC][E] += value * gate, from the gathered q2_K rows [T][cols] of key / 256 blocks (rows_q2k, device, the
/// bucket-major order of the fork's get_rows): emb = their dequantized values, kv = wkv emb, key = kv[0 : HC E],
/// value = kv[HC E : (HC + 1) E], gate[c] = sigmoid(sgn(s) sqrt(max(|s|, 1e-6))) with s = (rms(key_c) k_c) .
/// (rms(x_c) q_c) / sqrt(E). Tokens run in sub-chunks of engram_chunk.
void engram_forward(DenseEngine& e, const EngramWeights& w, const void* rows_q2k, int64_t T, float* x);

// ---- 6. router ----
/// The router's matmul: Auto is the GEMV up to kRouterGemvMaxCols tokens and cuBLAS above; the others force one (the
/// GEMV takes at most kRouterGemvMaxCols), for dsv41_engine_dense_bench's comparison.
enum class RouterMatmul { Auto, Gemv, Cublas };
/// ids, weights [T][n_expert_used]: logits = gate_inp . bf16(xn), probs = sqrt(softplus(logits)), the top
/// n_expert_used of probs + bias (ties: the lower expert first), weights = their unbiased probs, divided by
/// max(sum, 6.1035156e-5) when weights_norm, times weights_scale. Ids in descending biased order. forced (device
/// [T][n_expert_used], the parity mode of selection.hpp): these ids in this order instead of the top n_expert_used,
/// their weights computed the same way from this call's probs.
void dense_router(DenseEngine& e, const RouterWeights& w, const float* xn, int64_t T, int32_t* ids, float* weights,
                  RouterMatmul mm = RouterMatmul::Auto, const int32_t* forced = nullptr);

// ---- 7. shared expert ----
/// out [T][E] = down (silu(min(gate xn, L)) clamp(up xn, -L, L)), L = swiglu_limit. A decode token (T = 1)
/// runs up and gate in one launch and the SwiGLU with its q8_1 in another (the same values); xq: xn's q8_1
/// (qact_quantize's Q8_1 layout, e.g. hc_decode_step's) when the caller has it.
void dense_shared_expert(DenseEngine& e, const SharedExpertWeights& w, const float* xn, int64_t T, float* out,
                         const void* xq = nullptr);

// ---- 8. head ----
/// logits [n][n_vocab] for n output rows: x [n][HC][E] and pre [n][HC] (the last layer's FFN mix of those tokens),
/// collapsed, rms * output_norm, times output (q6_K). n <= max_head.
void dense_head(DenseEngine& e, const float* x, const float* pre, int64_t n, const float* output_norm,
                const QWeightView& output, float* logits);

}  // namespace strata::dsv41::engine
