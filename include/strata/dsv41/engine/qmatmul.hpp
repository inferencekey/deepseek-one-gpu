// include/strata/dsv41/engine/qmatmul.hpp - the engine's matmul on quantized weights: y = x W^T for a GGUF
// matrix kept on the device in its own q2_K, q3_K or q6_K blocks, uploaded as stored (no f16 copy). The real model's
// dense matrices are all of these (tools/dsv41/engine/SPEC.md "Tensors"): q2_K for every attention, compressor,
// indexer, HC, engram and shared-expert gate/up matrix, q3_K for the shared expert's down, q6_K for the output head.
//
// Numerics: ggml-cuda's integer dots. The activations
// are quantized to int8, the weight blocks are read as stored, the products of a sub-block are summed in int32 (dp4a
// or int8 MMA) and scaled in f32:
//
//   decode, T <= kMmvqMaxCols columns: MMVQ, Strata's native_mmvq.cu multi-column kernel (native_mmvq_multi_kernel,
//       its Q3_K and Q6_K traits as there, Q2_K added in the same pattern from the fork's vec_dot_q2_K_q8_1), on
//       activations in q8_1 blocks of 32 (QActLayout::Q8_1: half d = amax / 127, half sum, q = round(x / d));
//   prefill, T > kMmvqMaxCols, QPrefill::Mmq (the option): MMQ, the fork's ggml-cuda mul_mat_q (mmq.cuh with its
//       q2_K, q3_K and q6_K instances, run as Strata's src/prefill/moe_mmq.cu runs it), on activations in
//       block_q8_1_mmq tiles of 128 values: q3_K and q6_K one f32 scale per 32 (MmqD4), q2_K one f16 scale per 64
//       plus f16 sums of 16 raw values that its min term reads for the first 96 of every 128 (MmqD2S6). MMQ rounds a
//       q2_K block's d * scale and dmin * min to f16.
//
// The default prefill path, QPrefill::DequantF16 (chosen over MMQ, which ran at 0.33-0.39x of cuBLAS f16 on the
// A5000), is not an integer dot: the matrix is dequantized
// (ggml's dequantize_row_*, then rounded to f16) into the context's f16 scratch, in row slices when it is larger, and
// multiplied by cuBLAS f16 x f16 -> f32 on the f16 activations, as Strata's prefill Gemm::native does
// (src/prefill/gemm.cu). The scratch is the largest matrix the caller multiplies at once (80 MiB for attn_q_b).
// MMVQ stays below it: on the A5000 MMVQ took 155-243 us for attn_q_b at T = 4..8 against cuBLAS f16's 133 us on a
// ready f16 copy, which the dequant path must first write (80 MiB), so the threshold is MMVQ's own limit;
// qmatmul_bench's dequant + cuBLAS column measures the crossover.
//
// (one decode column ran at 190-540 GB/s, the q6_K head at 936): at one column
// MMVQ stages each block's rows in shared memory with 16-byte copies first (mmvq.cuh mmvq1_kernel; R rows of up to
// 12 KiB per block of 4 warps) and then runs the same sums in the same order, so its values are the multi-column
// kernel's bit for bit; a matrix whose rows fit no block size or whose data is not 16-byte aligned keeps that kernel.
//
// The quantizers are the engine's own kernels, the formulas of Strata's native_quantize_q8_1_kernel and of the fork's
// quantize_mmq_q8_1 without fast math, so qmatmul_ref.hpp reproduces them bit for bit and every integer product
// exactly: the GPU differs from the reference only in the order and rounding of its f32 sums. The dequant path's f16
// weights and activations are the reference's bit for bit too, and f16 products are exact in f32: only cuBLAS's f32
// sums differ.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::dsv41::engine {

/// GGUF type ids: f16 (attention's cuBLAS test path, not qmatmul's) and the three K-quants qmatmul reads.
constexpr int kQTypeF16 = 1;
constexpr int kQTypeQ2K = 10;
constexpr int kQTypeQ3K = 11;
constexpr int kQTypeQ6K = 14;
constexpr int64_t kQK = 256;          ///< values per K-quant block
constexpr int64_t kMmvqMaxCols = 8;   ///< MMVQ up to this many columns, the prefill path above (ggml's MMVQ_MAX_BATCH_SIZE)

/// The path for T > kMmvqMaxCols columns.
enum class QPrefill {
    DequantF16,   ///< default: dequantize into the f16 scratch, cuBLAS f16 x f16 -> f32
    Mmq,          ///< the fork's MMQ on int8 tiles, an option
};
enum class QPath { Mmvq, DequantF16, Mmq };
const char* qpath_name(QPath p);
/// The path qmatmul takes at T columns under a prefill choice (every weight type alike).
QPath qmatmul_path(QPrefill p, int64_t T);

/// A GGUF matrix of ne [k, rows] (k fastest) as stored: rows of k / 256 blocks of `type` (or k f16 values for
/// kQTypeF16). qmatmul reads device memory; qmatmul_ref.hpp takes the same view of host memory.
struct QWeightView {
    int type = 0;
    int64_t k = 0, rows = 0;
    const void* data = nullptr;
};

/// Bytes of one block (84 q2_K, 110 q3_K, 210 q6_K), 0 for any other type. Inline, with qtype_name, so the CPU experts
/// (experts_cpu*.cpp, mmvq_ref.cpp) do not link qmatmul.cu and cuBLAS (dsv41_cpu_experts_probe runs alone on a
/// pod before the payload); qmatmul.cu checks the sizes against mmvq.cuh's blocks.
constexpr size_t qtype_block_bytes(int type) {
    return type == kQTypeQ2K ? 84 : type == kQTypeQ3K ? 110 : type == kQTypeQ6K ? 210 : 0;
}
constexpr const char* qtype_name(int type) {
    return type == kQTypeF16 ? "f16" : type == kQTypeQ2K ? "q2_K" : type == kQTypeQ3K ? "q3_K"
           : type == kQTypeQ6K ? "q6_K" : "unsupported";
}
/// Bytes of the whole view (f16: 2 per value).
size_t qweight_bytes(const QWeightView& w);
/// The view of rows [r0, r0 + n).
QWeightView qweight_rows(const QWeightView& w, int64_t r0, int64_t n);

enum class QActLayout {
    Q8_1,      ///< MMVQ: [T][k / 32] blocks {half d, half sum; int8 q[32]}
    MmqD4,     ///< MMQ q3_K, q6_K: [k_pad / 128][T] blocks {float d[4]; int8 q[128]}
    MmqD2S6,   ///< MMQ q2_K: [k_pad / 128][T] blocks {half d[2] per 64, half sum[6] per 16 of values 0..95; q[128]}
};
const char* qact_layout_name(QActLayout l);
/// The int8 layout MMVQ (T <= kMmvqMaxCols) or MMQ (above) reads for a weight type at T columns.
QActLayout qact_layout(int type, int64_t T);
/// Device bytes of T rows of k values in a layout. MMQ: k padded to 512 (MATRIX_ROW_PADDING) plus 128 trailing
/// blocks that its last column tile may read (src/prefill/moe_mmq.cu q8_bytes). The largest of the int8 layouts qmatmul
/// reads at any T up to max_T under a prefill choice (DequantF16: q8_1 up to kMmvqMaxCols only), for a buffer that
/// serves them all: qact_bytes_max.
size_t qact_bytes(QActLayout l, int64_t T, int64_t k);
size_t qact_bytes_max(QPrefill p, int64_t max_T, int64_t k);

/// Quantized activations on the device: T rows of k values in one layout.
struct QActs {
    QActLayout layout = QActLayout::Q8_1;
    int64_t T = 0, k = 0;
    void* data = nullptr;
};

/// The prefill path's runtime: for DequantF16 a cuBLAS handle (with a fixed workspace, as Strata's Gemm::init) and the
/// f16 weight scratch of f16_scratch_elems values; for Mmq the fork's ggml_backend_cuda_context (its buffer pool for
/// MMQ's stream-k fixup) and the device limits. One per engine. MMQ calls on several streams may share it (the pool
/// is locked); dequant calls share the one scratch, so they must be ordered (one stream).
class QmmContext {
public:
    QmmContext() = default;
    ~QmmContext();
    QmmContext(const QmmContext&) = delete;
    QmmContext& operator=(const QmmContext&) = delete;
    /// False with `err` set when there is no CUDA device, MMQ cannot run the three types on it, or the scratch or the
    /// cuBLAS handle cannot be made. f16_scratch_elems: DequantF16 only, at least one row of the widest matrix (a
    /// larger matrix is dequantized in row slices).
    bool init(std::string& err, QPrefill prefill, int64_t f16_scratch_elems);
    QPrefill prefill() const { return prefill_; }
    QPath path(int64_t T) const { return qmatmul_path(prefill_, T); }
    void* mmq() const { return mmq_; }
    /// The cuBLAS handle on `stream` with the context's workspace (DequantF16 only).
    void* blas_for(void* stream);
    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    /// Device bytes the context holds (scratch and cuBLAS workspace).
    size_t device_bytes() const { return device_bytes_; }

private:
    QPrefill prefill_ = QPrefill::DequantF16;
    void* mmq_ = nullptr;
    void* blas_ = nullptr;
    void* blas_ws_ = nullptr;
    void* blas_stream_ = nullptr;
    bool blas_bound_ = false;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    size_t device_bytes_ = 0;
};

/// T rows of k values (f32 x, or f16 bits xh: exactly one of them) at row stride ld into `dst` (qact_bytes(l, T, k)
/// bytes) in layout l; `out` describes the result. k % 256 == 0.
void qact_quantize(const float* x, const uint16_t* xh, int64_t ld, int64_t T, int64_t k, QActLayout l, void* dst,
                   void* stream, QActs& out);

/// Exits naming `what` unless qmatmul takes w with `groups`: a q2_K, q3_K or q6_K matrix present, of whole 256-value
/// rows, rows a multiple of groups, within the kernels' int range, q2_K data 4-byte aligned. Host only (no device
/// access): attention's weight checks run it on the dev VM too.
void qmatmul_check(const QWeightView& w, int groups, const char* what);

/// y [T][ldy] f32 = a . w^T for a's T rows (MMVQ or MMQ). groups > 1 splits w's rows and each row of a into `groups`
/// equal parts (a.k = groups * w.k): y's columns g * rows / groups .. come from part g of both (attn_output_a's eight
/// groups). a.layout must be qact_layout(w.type, a.T) and the context's path at a.T MMVQ or MMQ. Exits naming `what`
/// on a shape, type, layout or path mismatch.
void qmatmul_q(QmmContext& c, const QWeightView& w, const QActs& a, float* y, int64_t ldy, int groups, void* stream,
               const char* what);

/// The DequantF16 path: y [T][ldy] f32 = xh . f16(w)^T, xh [T][ldx] f16 bits (groups as qmatmul_q: part g of each
/// row at xh + g * w.k), w dequantized into the context's scratch (whole, or row slices of each group when it is
/// larger), then cublasGemmEx (cublasGemmStridedBatchedEx over the groups when the whole matrix fits). Any T >= 1.
void qmatmul_f16(QmmContext& c, const QWeightView& w, const uint16_t* xh, int64_t ldx, int64_t T, float* y, int64_t ldy,
                 int groups, void* stream, const char* what);
/// Rows [r0, r0 + n) of w as the dequant path's f16 weights (ggml's dequantize_row_*, rounded to nearest even) into
/// out [n][w.k] on the device.
void qweight_dequant_f16(const QWeightView& w, int64_t r0, int64_t n, uint16_t* out, void* stream);

/// Device bytes of qmatmul's scratch for T rows of k values on a path: its int8 layout, or T k f16 values.
size_t qmatmul_scratch_bytes(QPath p, int type, int64_t T, int64_t k);
/// x [T][ldx] f32 on the context's path at T: quantized into `scratch` then qmatmul_q, or converted to f16 there
/// (round to nearest even) then qmatmul_f16. scratch: at least qmatmul_scratch_bytes(c.path(T), w.type, T,
/// groups * w.k) bytes.
void qmatmul(QmmContext& c, const QWeightView& w, const float* x, int64_t ldx, int64_t T, float* y, int64_t ldy,
             void* scratch, int groups, void* stream, const char* what);

/// decode's routed experts in VRAM (experts.hpp): one layer's experts at slots of a slab, each slot the three
/// parts of one expert (up [n_ff][n_embd], gate [n_ff][n_embd], down [n_embd][n_ff] as stored), and the layer's
/// residency row on the device. Strata's native_expert_grouped pattern (iq_kernels.hpp) on the MMVQ traits above.
struct QExpertSlab {
    const void* base = nullptr;          ///< the layer's slot 0 (device)
    const int32_t* slot_of = nullptr;    ///< device [n_expert]: the expert's slot, or -1 outside the slab
    size_t slot_bytes = 0;               ///< bytes from one slot to the next
    size_t part_offset[3] = {0, 0, 0};   ///< up, gate, down inside a slot
    int type[3] = {0, 0, 0};             ///< up, gate, down (gate and up of one type: one decoder reads both)
    int64_t n_embd = 0, n_ff = 0;
};
/// Exits naming `what` unless qmatmul_experts takes the slab: q2_K, q3_K or q6_K parts, gate's type up's, whole
/// 256-value rows, 4-byte aligned slots and parts. Host only.
void qmatmul_experts_check(const QExpertSlab& s, const char* what);
/// Device bytes of qmatmul_experts' scratch for T tokens of n_pairs pairs: the q8_1 rows of x and of h.
size_t qmatmul_experts_scratch_bytes(const QExpertSlab& s, int64_t T, int64_t n_pairs);
/// For the T * n_used pairs of x [T][n_embd] (f32, device; pair p is token p / n_used's, routed to expert ids[p],
/// device) whose expert has a slot: h [p][n_ff] = silu(min(gate x, L)) clamp(up x, -L, L) and y [p][n_embd] = down h;
/// the rows of the other pairs (the CPU's) are left as they are. Four launches whatever the routing (graph
/// capturable): x to q8_1, gate, up and the clamp of every pair (a block row per pair, a miss returns at once), h to
/// q8_1, down. Every value is qmatmul's (MMVQ, one column) and ffn_swiglu_clamp's bit for bit. T <= kMmvqMaxCols.
/// xq: x's q8_1 rows (qact_quantize's Q8_1 layout) when the caller has them (the first launch is skipped).
/// Each of the two matmul launches stages its rows in shared memory when they fit (mmvq.cuh), the same sums.
/// with T > 1 (a batch's tokens) the pairs that share an expert are computed
/// from one staged read of its rows (the expert's first pair in pair order is its leader, expert_hit_leader), each
/// pair's row still the one-pair kernel's bit for bit.
void qmatmul_experts(const QExpertSlab& s, const float* x, int64_t T, const int32_t* ids, int n_used, float limit,
                     float* h, void* scratch, float* y, void* stream, const void* xq = nullptr);
/// the grouped hits' rule on the host: the leader of pair p of a call (pairs ids[0 .. P)) is the first pair q <= p
/// routed to the same expert; -1 when that expert has no slot (slot_of[e] < 0, a miss).
int64_t expert_hit_leader(const int32_t* ids, int64_t P, int64_t p, const int32_t* slot_of);

}  // namespace strata::dsv41::engine
