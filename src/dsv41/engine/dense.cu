// src/dsv41/engine/dense.cu - DenseEngine, the weight checks, the rms norm, the embedding and the head of
// include/strata/dsv41/engine/dense.hpp; hc.cu, engram.cu, router.cu and dense_ffn.cu hold the other ops.
//
//   rms norm     rms_norm_kernel: a block per row (native_gr_norm.cu's weighted_rms_norm), the squares summed in
//                double as the fork and attention.cu, y = (x * scale) * w with each product rounded (no FMA);
//   embedding    embed_kernel: a thread per value of the token's q2_K row (dequantize_row_q2_K, q2k_value), written
//                into the four copies, and the one-hot pre_init;
//   head         head_collapse_kernel (the fork's unfused collapse of il = -1: products rounded, then added in copy
//                order), the norm, then the output matrix on qmatmul (MMVQ for up to 8 rows, its dequant path in
//                row slices of the context's scratch above).
#include "dense_internal.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine {
namespace dense_detail {

void failf(const char* fmt, ...) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 dense: %s\n", msg);
    std::exit(1);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) failf("%s: %s", what, cudaGetErrorString(e));
}

void project(DenseEngine& e, const QWeightView& w, const float* xf, const uint16_t* xh, int64_t T, float* y, int64_t ldy,
             const char* what) {
    DenseScratch& s = e.scratch_mut();
    const QPath p = e.qmm().path(T);
    if (!f16_input(T)) {
        if (!xf) failf("%s: %lld rows take f32 input", what, (long long) T);
        qmatmul(e.qmm(), w, xf, w.k, T, y, ldy, s.qact, 1, e.stream(), what);
        return;
    }
    if (!xh) failf("%s: %lld rows take f16 input", what, (long long) T);
    if (p == QPath::DequantF16) {
        qmatmul_f16(e.qmm(), w, xh, w.k, T, y, ldy, 1, e.stream(), what);
        return;
    }
    QActs a;
    qact_quantize(nullptr, xh, w.k, T, w.k, qact_layout(w.type, T), s.qact, e.stream(), a);
    qmatmul_q(e.qmm(), w, a, y, ldy, 1, e.stream(), what);
}

}  // namespace dense_detail

using namespace dense_detail;

namespace {

constexpr int NORM_T = 256;

// rows of n values: y = (x * (1 / sqrt(mean(x^2) + eps))) * w; in place is allowed (the row is read before written)
__global__ void __launch_bounds__(NORM_T) rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, int n,
                                                          float eps, float* y) {
    __shared__ double sh[32];
    const float* xr = x + (int64_t) blockIdx.x * n;
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += NORM_T) s += (double) __fmul_rn(xr[i], xr[i]);
    s = block_sum_d(s, sh);
    const float mean = (float) (s / (double) n);
    const float scale = 1.0f / sqrtf(mean + eps);
    float* yr = y + (int64_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += NORM_T) yr[i] = __fmul_rn(__fmul_rn(xr[i], scale), w[i]);
}

__global__ void embed_kernel(const uint8_t* __restrict__ table, int64_t row_bytes, const int32_t* __restrict__ tokens, int E,
                             float* __restrict__ x, float* __restrict__ pre_init) {
    const int64_t t = blockIdx.y;
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < kHc && pre_init) pre_init[t * kHc + v] = v == 0 ? 1.f : 0.f;
    if (v >= E) return;
    const uint8_t* row = table + (int64_t) tokens[t] * row_bytes;
    const float val = q2k_value(row + (int64_t) (v / 256) * 84, v % 256);
    float* xt = x + t * kHc * E;
    for (int c = 0; c < kHc; ++c) xt[(int64_t) c * E + v] = val;
}

// the head's collapse: out[e] = ((x0 w0 + x1 w1) + x2 w2) + x3 w3, each product rounded (deepseek4.cpp 296-310)
__global__ void head_collapse_kernel(const float* __restrict__ x, const float* __restrict__ pre, int E, float* __restrict__ out) {
    const int64_t t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const float* xt = x + t * kHc * E;
    const float* w = pre + t * kHc;
    float s = __fmul_rn(xt[e], w[0]);
    for (int c = 1; c < kHc; ++c) s = __fadd_rn(s, __fmul_rn(xt[(int64_t) c * E + e], w[c]));
    out[t * E + e] = s;
}

void* dalloc(size_t bytes, std::vector<void*>& allocs, size_t& total, const char* what, std::string& err) {
    void* p = nullptr;
    if (const cudaError_t e = cudaMalloc(&p, bytes ? bytes : 16); e != cudaSuccess) {
        char msg[192];
        std::snprintf(msg, sizeof msg, "dense workspace %s: %zu bytes: %s", what, bytes, cudaGetErrorString(e));
        err = msg;
        return nullptr;
    }
    allocs.push_back(p);
    total += bytes;
    return p;
}

void check_view(const QWeightView& w, int type, int64_t k, int64_t rows, const char* what, const char* name) {
    if (!w.data) failf("%s: %s is missing", what, name);
    if (w.type != type) failf("%s: %s has type %d (%s), expected %s", what, name, w.type, qtype_name(w.type), qtype_name(type));
    if (w.k != k || w.rows != rows)
        failf("%s: %s is [%lld][%lld], expected [%lld][%lld]", what, name, (long long) w.rows, (long long) w.k, (long long) rows, (long long) k);
    qmatmul_check(w, 1, name);
}

}  // namespace

DenseDims dense_dims_real() { return DenseDims{}; }

void hc_check_weights(const DenseDims& d, const HcWeights& w, const char* what) {
    check_view(w.fn, kQTypeQ2K, (int64_t) kHc * d.n_embd, kHcMix, what, "hc_fn");
    if (!w.base || !w.scale) failf("%s: hc base or scale is missing", what);
}
void engram_check_weights(const DenseDims& d, const EngramWeights& w, const char* what) {
    check_view(w.wkv, kQTypeQ2K, (int64_t) d.engram_cols() * d.engram_key, (int64_t) d.n_embd * (kHc + 1), what, "engram_wkv");
    if (!w.q || !w.k) failf("%s: engram_q or engram_k is missing", what);
    if (d.engram_key % 256) failf("%s: engram rows of %d values (q2_K blocks of 256)", what, d.engram_key);
}
void shared_check_weights(const DenseDims& d, const SharedExpertWeights& w, const char* what) {
    check_view(w.gate, kQTypeQ2K, d.n_embd, d.n_ff, what, "ffn_gate_shexp");
    check_view(w.up, kQTypeQ2K, d.n_embd, d.n_ff, what, "ffn_up_shexp");
    check_view(w.down, kQTypeQ3K, d.n_ff, d.n_embd, what, "ffn_down_shexp");
}
void head_check_weights(const DenseDims& d, const QWeightView& output, const char* what) {
    check_view(output, kQTypeQ6K, d.n_embd, d.n_vocab, what, "output");
}
void embed_check_weights(const DenseDims& d, const QWeightView& token_embd, const char* what) {
    if (!token_embd.data) failf("%s: token_embd is missing", what);
    if (token_embd.type != kQTypeQ2K || token_embd.k != d.n_embd || token_embd.rows < 1)
        failf("%s: token_embd is %s [%lld][%lld], expected q2_K rows of %d", what, qtype_name(token_embd.type),
              (long long) token_embd.rows, (long long) token_embd.k, d.n_embd);
}

DenseEngine::~DenseEngine() {
    for (void* p : allocs_) cudaFree(p);
    for (void* ev : events_)
        if (ev) cudaEventDestroy((cudaEvent_t) ev);
    if (blas_) cublasDestroy((cublasHandle_t) blas_);
}

bool DenseEngine::init(const DenseDims& d, int64_t max_T, int64_t max_head, int64_t engram_chunk, void* stream,
                       std::string& err, QPrefill prefill, QmmContext* shared) {
    if (blas_) failf("DenseEngine::init called twice");
    if (max_T < 1 || max_T > 65535 || max_head < 1 || max_head > 65535 || engram_chunk < 1)
        failf("max_T %lld, max_head %lld, engram_chunk %lld (1..65535)", (long long) max_T, (long long) max_head, (long long) engram_chunk);
    if (d.n_embd % 256 || d.n_ff % 256 || d.n_expert > 512 || d.n_expert_used > 32 || d.n_expert_used > d.n_expert)
        failf("dims: E %d, FF %d (multiples of 256), %d experts (<= 512), %d used (<= 32)", d.n_embd, d.n_ff, d.n_expert, d.n_expert_used);
    // the f16 weight scratch of the dequant path: the shared expert's matrices whole, the rest in row slices
    const int64_t scratch_elems = (int64_t) d.n_ff * d.n_embd;
    if (shared) {
        if (shared->prefill() != prefill || (prefill == QPrefill::DequantF16 && shared->scratch_elems() < scratch_elems))
            failf("a shared qmatmul context of another prefill path or of %lld scratch values (the dense ops need %lld)",
                  (long long) shared->scratch_elems(), (long long) scratch_elems);
        qmm_ = shared;
    } else {
        if (!own_qmm_.init(err, prefill, scratch_elems)) return false;
        ws_bytes_ += own_qmm_.device_bytes();
    }
    d_ = d;
    max_T_ = max_T;
    max_head_ = max_head;
    engram_chunk_ = std::min(engram_chunk, max_T);
    stream_ = stream;
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
        err = "cublasCreate failed";
        return false;
    }
    blas_ = h;
    if (cublasSetStream(h, (cudaStream_t) stream) != CUBLAS_STATUS_SUCCESS) failf("cublasSetStream");
    const size_t ws = 8u << 20;   // the router's GEMM only
    if (!(blas_ws_ = dalloc(ws, allocs_, ws_bytes_, "cuBLAS", err))) return false;
    if (cublasSetWorkspace(h, blas_ws_, ws) != CUBLAS_STATUS_SUCCESS) failf("cublasSetWorkspace");

    const int64_t T = max_T, Te = engram_chunk_, Tv = std::min<int64_t>(T, kMmvqMaxCols), Tev = std::min<int64_t>(Te, kMmvqMaxCols);
    const int64_t E = d.n_embd, X = (int64_t) kHc * E, EK = (int64_t) d.engram_cols() * d.engram_key, NE = d.n_expert, F = d.n_ff;
    // the activation scratch: the largest any qmatmul call here quantizes or converts into. The calls on f32 rows
    // (shared expert, head) take it on every path; the hc and engram calls write f16 rows of their own above 8 tokens,
    // which only MMQ quantizes into it.
    auto need = [&](int type, int64_t rows, int64_t k, bool f16_rows) {
        size_t b = qmatmul_scratch_bytes(QPath::Mmvq, type, std::min<int64_t>(rows, kMmvqMaxCols), k);
        const QPath p = qmatmul_path(prefill, rows);
        if (rows > kMmvqMaxCols && (!f16_rows || p == QPath::Mmq)) b = std::max(b, qmatmul_scratch_bytes(p, type, rows, k));
        return b;
    };
    size_t qact = std::max({need(kQTypeQ2K, T, X, true), need(kQTypeQ2K, Te, EK, true), need(kQTypeQ2K, T, E, false),
                            need(kQTypeQ3K, T, F, false), need(kQTypeQ6K, max_head, E, false)});
    DenseScratch& s = s_;
    struct Buf { void** p; size_t bytes; const char* what; };
    const Buf bufs[] = {
        {(void**) &s.mixes, (size_t) (T * kHcMix) * 4, "mixes"},
        {(void**) &s.hc_in, (size_t) (Tv * X) * 4, "hc_in"},
        {(void**) &s.hc_inh, f16_input(T) ? (size_t) (T * X) * 2 : 0, "hc_inh"},
        {(void**) &s.collapsed, (size_t) (T * E) * 4, "collapsed"},
        {(void**) &s.emb, (size_t) (Tev * EK) * 4, "engram emb"},
        {(void**) &s.embh, f16_input(Te) ? (size_t) (Te * EK) * 2 : 0, "engram emb f16"},
        {(void**) &s.engram_kv, (size_t) (Te * E * (kHc + 1)) * 4, "engram kv"},
        {(void**) &s.engram_gate, (size_t) (T * kHc) * 4, "engram gate"},
        {(void**) &s.router_in, (size_t) (T * E) * 2, "router input"},
        {(void**) &s.logits, (size_t) (T * NE) * 4, "router logits"},
        {(void**) &s.probs, (size_t) (T * NE) * 4, "router probs"},
        {(void**) &s.sh_gate, (size_t) (T * F) * 4, "shared gate"},
        {(void**) &s.sh_up, (size_t) (T * F) * 4, "shared up"},
        {(void**) &s.sh_act, (size_t) (T * F) * 4, "shared act"},
        {(void**) &s.head_in, (size_t) (max_head * E) * 4, "head collapse"},
        {(void**) &s.head_norm, (size_t) (max_head * E) * 4, "head norm"},
        {&s.qact, qact, "activation scratch"},
        {(void**) &s.hc_ticket, 4, "hc_decode_step ticket"},
        {(void**) &s.hc_marks, (size_t) kHcStepGrid * kHcStepMarks * 8, "hc_decode_step marks"},
    };
    for (const Buf& b : bufs)
        if (!(*b.p = dalloc(b.bytes, allocs_, ws_bytes_, b.what, err))) return false;
    if (cudaMemset(s.hc_ticket, 0, 4) != cudaSuccess) {
        err = "cudaMemset (hc_decode_step ticket) failed";
        return false;
    }
    hc_step_attrs(d.n_embd);
    for (auto& ev : events_) {
        cudaEvent_t e = nullptr;
        if (cudaEventCreate(&e) != cudaSuccess) {
            err = "cudaEventCreate failed";
            return false;
        }
        ev = e;
    }
    return true;
}

void dense_rms_norm(DenseEngine& e, const float* x, const float* w, int64_t rows, int n, float* y) {
    OpTimer tm(e, &e.times_mut().norm_ms);
    if (rows < 1 || n < 1) failf("dense_rms_norm: %lld rows of %d", (long long) rows, n);
    rms_norm_kernel<<<(unsigned) rows, NORM_T, 0, stream_of(e)>>>(x, w, n, e.dims().rms_eps, y);
    ck(cudaGetLastError(), "rms_norm_kernel");
}

void dense_embed(DenseEngine& e, const QWeightView& token_embd, const int32_t* tokens, int64_t T, float* x,
                 float* pre_init) {
    OpTimer tm(e, &e.times_mut().embed_ms);
    const DenseDims& d = e.dims();
    embed_check_weights(d, token_embd, "dense_embed");
    if (T < 1 || T > e.max_T()) failf("dense_embed: %lld tokens (1..%lld)", (long long) T, (long long) e.max_T());
    const dim3 grid((unsigned) ((d.n_embd + 255) / 256), (unsigned) T);
    embed_kernel<<<grid, 256, 0, stream_of(e)>>>(static_cast<const uint8_t*>(token_embd.data), (int64_t) (d.n_embd / 256) * 84,
                                                 tokens, d.n_embd, x, pre_init);
    ck(cudaGetLastError(), "embed_kernel");
}

void dense_head(DenseEngine& e, const float* x, const float* pre, int64_t n, const float* output_norm,
                const QWeightView& output, float* logits) {
    OpTimer tm(e, &e.times_mut().head_ms);
    const DenseDims& d = e.dims();
    head_check_weights(d, output, "dense_head");
    if (n < 1 || n > e.max_head()) failf("dense_head: %lld rows (1..%lld)", (long long) n, (long long) e.max_head());
    DenseScratch& s = e.scratch_mut();
    const dim3 grid((unsigned) ((d.n_embd + 255) / 256), (unsigned) n);
    head_collapse_kernel<<<grid, 256, 0, stream_of(e)>>>(x, pre, d.n_embd, s.head_in);
    ck(cudaGetLastError(), "head_collapse_kernel");
    rms_norm_kernel<<<(unsigned) n, NORM_T, 0, stream_of(e)>>>(s.head_in, output_norm, d.n_embd, d.rms_eps, s.head_norm);
    ck(cudaGetLastError(), "rms_norm_kernel");
    qmatmul(e.qmm(), output, s.head_norm, d.n_embd, n, logits, d.n_vocab, s.qact, 1, e.stream(), "output");
}

}  // namespace strata::dsv41::engine
