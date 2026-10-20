// include/strata/dsv41/engine/attention.hpp - one DeepSeek-V4.1-Flash attention layer on the GPU: query and
// latent KV projections, rope, the sliding window, the compressor with its carried state, the lightning indexer
// and Reuse Mode, attention over window + selected compressed rows with sinks, de-rope and the grouped output
// projection, for a prefill chunk or a decode token, against the KV cache of kv_cache.hpp.
//
// The math is the fork's (tools/dsv41/engine/SPEC.md, sections 4 and 5; deepseek41.cpp 440-804). Per token t at
// position p, with x the attention input (attn_norm):
//
//   qr = rms(x wq_a) * q_a_norm;  q = rope(qr wq_b);  kv = rope(rms(x wkv) * kv_norm)  -> window cell p
//   KV source:   ckv = x comp_kv, cs = x comp_gate (or ckv at ratio 1); each completed block j of r positions
//                pools latent_j = sum_i softmax_i(cs) ckv; pre = rms(latent_j) * comp_norm; row j = rope(pre, j r)
//   key owner:   index key j = hadamard(rope(rms(pre attn_k) * k_norm, j r))
//   index source: q_idx = hadamard(rope(qr idx_q_b)), w = x proj / sqrt(128 heads); top-k of the key source's keys
//   attention:   the window (positions p - 127 .. p) plus the compressed rows of the KV source: an index source's own
//                top-k; a reader's (Reuse) the top-k of its topk_src (ReaderMode::Reuse, the default) or every
//                visible row (ReaderMode::Dense, the fork); softmax(q k / sqrt(D)) with the head's sink in the
//                denominator, V = K;  out = (derope(attn) grouped by wo_a) wo_b
//
// Positions see rope at their own position, compressed rows and index keys at their block's first position. Window
// layers rope with rope.freq_base and no YaRN, compressed layers with compress_rope_freq_base and YaRN (SPEC.md 4).
//
// Starting point: Strata's QSA attention. The attention kernel is
// qsa_prompt_attn.cu's online softmax on m16n8k16 f16 MMAs with the hi/lo split of p, regrown for one latent KV head
// shared by every query head (MQA: a block holds 32 heads and reads each key once for all of them, V = K), head dim
// 256 or 512, and a cell list of window positions plus selected rows; decode splits the cells over blocks and merges
// with the log-sum-exp rescale of qsa_decode_attn.cu (sinks enter there, as a cell with no value). A prefill
// chunk takes a kernel of its own on the same math (attention.cu, "prefill attention"): p rounded once to f16 instead
// of the hi/lo split (one f16 rounding is up to 2^-11 per cell, the GPU test's tolerance of sum p |v| in the worst
// case; it holds by the rounding errors' independence across cells, and the test prints the measured margin), the
// q.k and p.v fragments by ldmatrix, the K rows by cp.async, and the de-rope in its epilogue; decode keeps the split.
// The step position lives in device memory as in qsa.hpp's step buffer. Rope is rope_scaling.hpp's YaRN arithmetic
// in the fork's order.
// The dense projections run on qmatmul.hpp over the GGUF's q2_K blocks as stored: MMVQ on int8 activations
// for up to 8 tokens (decode), and above that the matrix dequantized into qmatmul's f16 scratch for cuBLAS f16 on the
// f16 activations (QPrefill::DequantF16, the default) or the fork's MMQ on int8 ones (QPrefill::Mmq, the option). A
// weight view of type f16 takes the cuBLAS path instead (f16 weights dequantized from the GGUF, f16 activations,
// f32 accumulation, as Strata's prefill Gemm::f16; wrapped here because strata_prefill would bring the shared CUDA
// runtime into binaries that link the static one), kept for the tests.
#pragma once

#include "strata/dsv41/engine/qmatmul.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::dsv41::engine {

/// One model's attention geometry (GGUF deepseek41.* keys, SPEC.md "Constants").
struct AttnDims {
    int n_embd = 5120;
    int n_head = 64;
    int head_dim = 512;       ///< key_length = value_length; 256 or 512 (the kernels' instances)
    int n_rot = 64;           ///< rope.dimension_count: the last n_rot values of a head (and of an index head) rotate
    int q_lora = 1280;
    int o_groups = 8;
    int o_lora = 1024;
    int n_swa = 128;          ///< attention.sliding_window
    int idx_heads = 32;
    int idx_dim = 128;        ///< the indexer's key length; the indexer has only 128
    int top_k = 512;
    float rms_eps = 1e-20f;
    float rope_base = 10000.f;        ///< rope.freq_base: window only layers
    float comp_rope_base = 160000.f;  ///< attention.compress_rope_freq_base: compressed layers, indexer, rows
    float yarn_factor = 16.f;         ///< rope.scaling.factor (freq_scale = 1 / factor)
    int yarn_orig_ctx = 65536;
    float yarn_beta_fast = 32.f;
    float yarn_beta_slow = 1.f;
};

/// The real DeepSeek-V4.1-Flash (GGUF header, vcruz305 Q2_K c4a08554).
AttnDims attn_dims_real();

enum class AttnRole {
    Window,    ///< ratio 0: the window only
    Full,      ///< an index source with its own compressor (real 2, 8, 14, 20)
    Reindex,   ///< an index source reading another layer's keys (real 24, 28, 32, 36)
    Reuse,     ///< a reader of a source's rows (real 3-7, 9-13, 15-19, 21-23, ...)
};

/// One layer's role, from which tensors the file gives it (deepseek41.cpp 209-256).
struct AttnLayer {
    int index = 0;
    int ratio = 0;              ///< compress ratio, 0 = window only
    bool kv_source = false;     ///< carries attn_compressor_kv
    bool gate = false;          ///< carries attn_compressor_gate (ratio > 1 sources)
    bool key_owner = false;     ///< carries indexer.attn_k
    bool index_source = false;  ///< carries indexer.attn_q_b
    int kv_src = -1, key_src = -1, topk_src = -1;   ///< the latest of each kind at or before this layer
    AttnRole role() const;
};

/// The roles of every layer from per-layer flags (ratio and the four tensors), the latest-source rule of the fork.
/// Exits naming the layer when a compressed layer has no source before it or its ratio differs from its source's.
std::vector<AttnLayer> attn_layers(const std::vector<int>& ratios, const std::vector<int>& flags);
/// attn_layers' flags: bit 0 kv_source, 1 gate, 2 key_owner, 3 index_source
enum AttnLayerFlag : int { kKvSource = 1, kGate = 2, kKeyOwner = 4, kIndexSource = 8 };
/// The real model's 40 layers.
std::vector<AttnLayer> attn_layers_real();

enum class ReaderMode {
    Reuse,   ///< default: a reader attends with the top-k of its topk_src (tech report 2.3.1)
    Dense,   ///< `--readers dense`: every visible compressed row, as the fork (parity mode)
};

/// Rope arguments of one layer kind, in the fork's float arithmetic (SPEC.md 4).
struct RopeArgs {
    float theta_scale = 0.f;   ///< powf(base, -2 / n_rot)
    float freq_scale = 1.f;
    float ext_factor = 0.f;
    float attn_factor = 1.f;   ///< dsv4_rope_attn_factor
    float mscale = 1.f;        ///< attn_factor times rope_yarn's log term when ext_factor != 0 (host logf, as the CPU)
    float corr0 = 0.f, corr1 = 0.f;
    int n_rot = 64;
};
/// compressed = false: rope.freq_base, no YaRN (window only layers); true: compress base with YaRN.
RopeArgs attn_rope_args(const AttnDims& d, bool compressed);

/// Weights of one layer on the device: each matrix a view of a GGUF matrix of ne [K, rows] as stored ([rows] rows of
/// K values; q2_K in the real model, any qmatmul type, or f16 bits for the cuBLAS test path), norms and sinks f32.
/// Empty (null data) where the role has none; the call checks every view's type and shape.
struct AttnLayerWeights {
    QWeightView wq_a;        ///< [q_lora][n_embd]
    QWeightView wq_b;        ///< [n_head * head_dim][q_lora]
    QWeightView wkv;         ///< [head_dim][n_embd]
    QWeightView wo_a;        ///< [o_groups * o_lora][n_head / o_groups * head_dim]; group g rows g*o_lora..
    QWeightView wo_b;        ///< [n_embd][o_groups * o_lora]
    QWeightView comp_kv;     ///< [head_dim][n_embd]
    QWeightView comp_gate;   ///< [head_dim][n_embd]
    QWeightView idx_k;       ///< indexer.attn_k [idx_dim][head_dim]
    QWeightView idx_q_b;     ///< [idx_heads * idx_dim][q_lora]
    QWeightView idx_proj;    ///< [idx_heads][n_embd]
    const float* q_a_norm = nullptr;     ///< [q_lora]
    const float* kv_norm = nullptr;      ///< [head_dim]
    const float* comp_norm = nullptr;    ///< [head_dim]
    const float* idx_k_norm = nullptr;   ///< [idx_dim]
    const float* sinks = nullptr;        ///< [n_head]
};

class KvCache;   // kv_cache.hpp

/// A prefill call's ops in call order (the per-op split of a 16K chunk), each
/// with its norms, ropes and stores. A decode call marks its stages only, at the stage's first op (QA, CompProj,
/// Select, Attend, WoA).
enum class AttnOp : int {
    QA,         ///< attn_q_a and the qr norm
    QB,         ///< attn_q_b
    QRope,      ///< q's rope
    Kv,         ///< attn_kv, its norm and rope, the window cells
    IdxQ,       ///< indexer.attn_q_b, its rope and rotation
    IdxW,       ///< indexer.proj, the head weights
    CompProj,   ///< attn_compressor_kv and attn_compressor_gate
    Pool,       ///< the pooled rows, their norm, rope and store, the carried state
    IdxKeys,    ///< indexer.attn_k of the new rows, its norm, rope, rotation and store
    Select,     ///< indexer_select (or the forced rows)
    Attend,     ///< the attention kernel and its merge
    Derope,     ///< the de-rope (0 when the attention's epilogue writes it)
    WoA,        ///< attn_output_a
    WoB,        ///< attn_output_b
    Count
};
constexpr int kAttnOps = (int) AttnOp::Count;
const char* attn_op_name(AttnOp op);

/// Stage times of the last call, from CUDA events (AttnEngine::set_timing(true)): op_ms per AttnOp, and the five
/// stages as their sums.
struct AttnTimes {
    float op_ms[kAttnOps] = {};
    float proj_ms = 0;        ///< QA .. IdxW: q, kv and indexer query projections and their norms and ropes
    float compress_ms = 0;    ///< CompProj .. IdxKeys: the compressor's projections, pooling, rows, index keys
    float indexer_ms = 0;     ///< Select: indexer_select
    float attention_ms = 0;   ///< Attend: the attention kernel and its merge
    float output_ms = 0;      ///< Derope .. WoB: de-rope, wo_a, wo_b
    float total_ms = 0;
};

/// The device buffers of one call, kept after it so a test can read every stage (row-major, the call's T rows). The
/// f16 copies of projection inputs (xh, qrh, latenth, oah) are what the f16 test path and qmatmul's dequant path read
/// (the norms write qrh and latenth on every call); de-rope is f16 on every path (MMVQ and MMQ quantize it from
/// there).
struct AttnScratch {
    uint16_t* xh = nullptr;      ///< [T][n_embd] f16 of the input
    float* qr = nullptr;         ///< [T][q_lora] after its norm
    uint16_t* qrh = nullptr;     ///< [T][q_lora] f16
    float* q = nullptr;          ///< [T][n_head][head_dim] roped
    float* kv = nullptr;         ///< [T][head_dim] normed and roped (its f16 is the window cell)
    float* ckv = nullptr;        ///< [T][head_dim] compressor kv projection
    float* cscore = nullptr;     ///< [T][head_dim] compressor gate projection (ckv itself at ratio 1)
    float* pooled = nullptr;     ///< [B][head_dim] pooled rows of the blocks the call completed
    float* latent = nullptr;     ///< [B][head_dim] normed (pre-rope) rows
    uint16_t* latenth = nullptr; ///< [B][head_dim] f16 of latent (the index key projection's input)
    float* rowsf = nullptr;      ///< [B][head_dim] roped rows before their f16 store
    float* idx_kf = nullptr;     ///< [B][idx_dim] index keys before their f16 store (roped, rotated)
    float* iq = nullptr;         ///< [T][idx_heads][idx_dim] indexer queries, roped and rotated
    float* iw = nullptr;         ///< [T][idx_heads] indexer head weights (scaled)
    int32_t* pos = nullptr;      ///< [T] positions (decode with device_pos: the caller's, written before the call runs)
    int32_t* blk_pos = nullptr;  ///< [1] decode: the first position of the token's compressed block (the rows' rope)
    uint16_t* qh = nullptr;      ///< [n_head][head_dim] decode with q2_K weights: f16 of q, the attention's input
    float* attn = nullptr;       ///< [T][n_head][head_dim] attention output before de-rope (a prefill chunk of one
                                 ///< split: only with AttnEngine::set_attn_output(true))
    uint16_t* derope = nullptr;  ///< [T][n_head * head_dim] f16, de-roped
    float* oa = nullptr;         ///< [T][o_groups * o_lora] wo_a output
    uint16_t* oah = nullptr;     ///< [T][o_groups * o_lora] f16
    float* idx_scratch = nullptr;
    size_t idx_scratch_bytes = 0;
    float* part = nullptr;       ///< split-K partials
    size_t part_floats = 0;
    void* xq = nullptr;          ///< qmatmul's int8 activations of x, qr, the latent, de-rope and oa (qact_bytes_max each)
    void* qrq = nullptr;
    void* latq = nullptr;
    void* drq = nullptr;
    void* oaq = nullptr;
    uint16_t* qhb = nullptr;     ///< [max_batch][n_head][head_dim] f16 of each batch row's q (attn_decode_batch)
    int32_t* blk_posb = nullptr; ///< [max_batch] each batch row's compressed block's first position
    int64_t T = 0, B = 0, first_block = 0;   ///< the last call's tokens, completed blocks and the first block's index
    int n_splits = 1;                        ///< the last call's attention splits
    // (host values of the last call made from the host: a replayed decode graph does not update them)
};

/// The launches of a fused decode token (attn_decode with q2_K weights), for its per-kernel split:
/// the x quantizer (skipped when the caller hands x's q8_1), the x projections in one MMVQ launch, the qr and kv norms
/// (with kv's rope and window cell), the qr projections with their ropes, the compressor and index key bookkeeping,
/// the index selection, the attention, the merge with the de-rope, wo_a, oa's quantizer, wo_b.
enum class AttnKernel : int { XQuant, XProj, QkvNorm, QProj, Book, Select, Attend, Merge, WoA, OaQuant, WoB, Count };
constexpr int kAttnKernels = (int) AttnKernel::Count;
const char* attn_kernel_name(AttnKernel k);

/// The engine's attention for one model: qmatmul's context (its own or the stream's shared one, an f16 scratch of the
/// largest matrix of the dims, attn_max_matrix_elems), cuBLAS (f16 test path), the workspace for chunks of up to max_chunk tokens, one stream.
class AttnEngine {
public:
    AttnEngine() = default;
    ~AttnEngine();
    AttnEngine(const AttnEngine&) = delete;
    AttnEngine& operator=(const AttnEngine&) = delete;

    /// Allocates the workspace for chunks of up to max_chunk tokens against index sources of up to max_rows keys, with
    /// qmatmul's prefill path `prefill`. shared: a qmatmul context of that path on the same stream whose f16 scratch
    /// holds attn_max_matrix_elems values (one dequant scratch per stream); null: the engine makes its own.
    /// False with `err` set when the device or a buffer is missing.
    bool init(const AttnDims& d, int64_t max_chunk, int64_t max_rows, void* stream, std::string& err,
              QPrefill prefill = QPrefill::DequantF16, QmmContext* shared = nullptr);

    /// the buffers attn_decode_batch adds for up to max_batch rows (each row's q in f16, its block position).
    bool init_batch(int max_batch, std::string& err);
    int max_batch() const { return max_batch_; }

    const AttnDims& dims() const { return d_; }
    const AttnScratch& scratch() const { return s_; }
    size_t workspace_bytes() const { return ws_bytes_; }
    void* stream() const { return stream_; }
    void set_timing(bool on) { timing_ = on; }
    const AttnTimes& times() const { return times_; }
    /// Keep a prefill chunk's f32 attention output in scratch().attn (off by default). The layer reads only the f16
    /// de-rope, which a chunk of one split writes in the attention's epilogue; the f32 copy is for the GPU test's
    /// attention stage (2 GiB of writes per layer at a 16384-token chunk). Several splits always write it (the
    /// merge's output, the de-rope's input), and so does a decode token.
    void set_attn_output(bool on) { attn_output_ = on; }
    bool attn_output() const { return attn_output_; }
    /// The fused decode's per-kernel marks (off by default): an event on the stream before each launch and after
    /// the last; inside a captured graph they become its nodes and mark every replay. read_kernel_marks adds each
    /// marked launch's time since clear_kernel_marks to ms[kAttnKernels] once the stream is past them (false when an
    /// event's time cannot be read); one call marks at most kAttnKernels + 1 points, two calls fit between clears.
    void set_kernel_marks(bool on);
    bool read_kernel_marks(double* ms) const;
    void clear_kernel_marks();
    void mark_kernel(AttnKernel k);

    // internals used by attention.cu's free functions
    void* blas() const { return blas_; }
    QmmContext& qmm() { return *qmm_; }
    int64_t max_chunk() const { return max_chunk_; }
    AttnScratch& scratch_mut() { return s_; }
    AttnTimes& times_mut() { return times_; }
    bool timing() const { return timing_; }
    void* event(int i) const { return events_[i]; }
    int sm_count() const { return sm_count_; }

private:
    AttnDims d_{};
    AttnScratch s_{};
    QmmContext own_qmm_;
    QmmContext* qmm_ = &own_qmm_;
    void* blas_ = nullptr;
    void* stream_ = nullptr;
    void* blas_ws_ = nullptr;
    int64_t max_chunk_ = 0;
    int max_batch_ = 0;
    size_t ws_bytes_ = 0;
    bool timing_ = false;
    bool attn_output_ = false;
    AttnTimes times_{};
    void* events_[kAttnOps + 1] = {};
    bool kmarks_on_ = false;
    std::vector<void*> kevents_;
    std::vector<std::pair<int, void*>> kmarks_;
    int sm_count_ = 0;
    std::vector<void*> allocs_;
};

/// The checks attn_prefill makes of a layer's weights before its first launch, host only (no device access): every
/// view the role reads is present, of the dims' shape, and f16 or a matrix qmatmul takes (qmatmul_check). Exits naming
/// the layer and the tensor. dsv41_engine_attention_test --gpu --dry-run runs it on the dev VM.
void attn_check_weights(const AttnDims& d, const AttnLayer& l, const AttnLayerWeights& w);
/// Values of the largest projection matrix of the dims (attn_q_b and attn_output_b at the real dims, 80 MiB as f16).
int64_t attn_max_matrix_elems(const AttnDims& d);

/// Layer `l` over a chunk of T tokens at positions p0 .. p0 + T - 1 of the cache's sequence: x [T][n_embd] f32
/// (device), out [T][n_embd] f32 (device). The layer's previous chunk must have ended at p0 (the cache checks it), and
/// a reader's or Reindex layer's sources must have run this chunk already. T <= the engine's max_chunk.
/// forced_rows (device [T][top_k], an index source only; the parity mode of selection.hpp): the layer attends, and
/// hands its readers, these rows (per token the visible ones ascending, then -1) instead of the indexer's top-k; the
/// indexer's query side still runs.
void attn_prefill(AttnEngine& e, const AttnLayer& l, const AttnLayerWeights& w, KvCache& kv, const float* x,
                  int64_t T, int64_t p0, float* out, ReaderMode mode = ReaderMode::Reuse,
                  const int32_t* forced_rows = nullptr);

/// One decode token at position p: attn_prefill's math with T = 1 in launches of a fixed shape that read the
/// position on the device (scratch().pos[0]), so a CUDA graph captured at one position replays at every later one
/// (forward.hpp's token graph, Strata's qsa.hpp step buffer): the compressor's block completes on
/// the device ((p + 1) % ratio == 0: its row and index key stored, else the position carried), the indexer runs over
/// its key source's whole capacity and each kernel bounds it by the position (indexer.hpp), the attention takes a
/// split count fixed by the layer's largest cell count and divides the cells on the device. The host's checks and
/// the cache's bookkeeping are attn_prefill's (attn_decode_advance). device_pos: the caller wrote p into
/// scratch().pos[0] before these launches run (a graph's memcpy node); false: the call writes it.
/// with q2_K weights whose rows stage (attention.cu, the fused decode token) the token takes 9 to 12
/// fused launches (AttnKernel) that store attn_prefill's values bit for bit, the attention's normalization aside
/// (always the split merge's); other weights take attn_prefill's launches at T = 1. xq: x's q8_1 (qact_quantize's
/// Q8_1 layout of x, e.g. hc_decode_step's) when the caller has it, so the fused token skips its quantizer.
void attn_decode(AttnEngine& e, const AttnLayer& l, const AttnLayerWeights& w, KvCache& kv, const float* x, int64_t p,
                 float* out, ReaderMode mode = ReaderMode::Reuse, const int32_t* forced_rows = nullptr,
                 bool device_pos = false, const void* xq = nullptr);
/// The host's half of a replayed attn_decode at position p: its checks (the layer at p, its sources at p + 1, the
/// cache's capacity) and the bookkeeping (next position, an index source's top-k now the token's), no launch.
void attn_decode_advance(const AttnLayer& l, KvCache& kv, int64_t p, ReaderMode mode, bool forced_rows);

/// one decode token of each of B sequences (batch rows, 1 <= B <=
/// AttnEngine::max_batch()), row i against its own cache kv[i] at its own position: pos [B] in device memory (what a
/// captured graph reads), p [B] the host's values for the checks. x [B][n_embd] f32 with xq its q8_1 rows
/// (qact_quantize's Q8_1 layout, hc's step), out [B][n_embd]. The projections take every row in one launch each
/// (qmatmul_q's MMVQ at B columns, each column the one-column kernel's value bit for bit: the weights read once for
/// the batch), their ropes and norms run per row in attn_decode's arithmetic, and what reads a sequence's cache runs
/// per row with attn_decode's own kernels on its buffers (the window cell, the compressor and the index key, the
/// selection, the attention and its merge with the de-rope), so every value a row stores is attn_decode's for that
/// sequence alone (no value of a row depends on the others). Takes the fused token's weights (q2_K whose
/// rows stage) and ReaderMode::Reuse (ReaderMode::Dense's split count follows a cache's capacity), no forced rows;
/// exits naming the reason otherwise. The host's checks and bookkeeping are attn_decode_advance's for every row.
void attn_decode_batch(AttnEngine& e, const AttnLayer& l, const AttnLayerWeights& w, KvCache* const* kv, int B,
                       const float* x, const void* xq, const int32_t* pos, const int64_t* p, float* out,
                       ReaderMode mode = ReaderMode::Reuse);
/// The host's half of a replayed attn_decode_batch: attn_decode_advance for every row.
void attn_decode_batch_advance(const AttnLayer& l, KvCache* const* kv, int B, const int64_t* p, ReaderMode mode);

}  // namespace strata::dsv41::engine
