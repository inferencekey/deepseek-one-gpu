// src/dsv41/engine/engram.cu - the engram tables and gather of include/strata/dsv41/engine/engram.hpp, and
// engram_forward of dense.hpp (SPEC.md section 2, deepseek41.cpp 357-438).
//
//   gather          EngramGather: the chunk's rows memcpy'd by host threads (Strata's PleTable::gather_batch, the
//                   decode moved to the GPU) into a pinned staging buffer, one cudaMemcpyAsync to the device;
//   rows            engram_dequant_kernel: each gathered q2_K row dequantized (dequantize_row_q2_K) in the
//                   bucket-major order of the fork's get_rows, as f32 for MMVQ calls or f16 for the dequant and MMQ
//                   paths, then engram_wkv on qmatmul;
//   gate            engram_gate_kernel (a block per token): for each copy the rms of the key and of the stream copy
//                   (squares in double) times engram_k and engram_q, their products summed in double, times
//                   1 / sqrtf(E), then sigmoid(sgn(s) sqrt(max(|s|, 1e-6))), and x += value * gate (product rounded,
//                   then added: the fork's ggml_mul and ggml_add) in place. Native_ple_postops.cu's PLE gate has the
//                   same signed square root (deepseek41.cpp 420-423 says the two models build it alike).
// Tokens run in sub-chunks of the engine's engram_chunk, so the [Te][(HC + 1) E] projection (100 MiB at 1024 tokens
// of the real dims) does not grow with the prefill chunk.
#include "strata/dsv41/engine/engram.hpp"

#include "dense_internal.cuh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::dsv41::engine {

using namespace dense_detail;

// ---------------------------------------------------------------- tables and the gather
EngramTable::~EngramTable() {
    if (owned_) cudaFreeHost(owned_);
}

bool EngramTable::alloc(int64_t n_rows, int row_bytes, std::string& err) {
    if (owned_ || rows_) failf("EngramTable::alloc on a table already set");
    if (n_rows < 1 || row_bytes < 1) failf("an engram table of %lld rows of %d bytes", (long long) n_rows, row_bytes);
    void* p = nullptr;
    const size_t bytes = (size_t) n_rows * (size_t) row_bytes;
    if (const cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocDefault); e != cudaSuccess) {
        err = "pinned engram table of " + std::to_string(bytes >> 20) + " MiB: " + cudaGetErrorString(e);
        return false;
    }
    owned_ = static_cast<uint8_t*>(p);
    rows_ = owned_;
    n_rows_ = n_rows;
    row_bytes_ = row_bytes;
    return true;
}

void EngramTable::view(const uint8_t* rows, int64_t n_rows, int row_bytes) {
    if (owned_ || rows_) failf("EngramTable::view on a table already set");
    if (!rows || n_rows < 1 || row_bytes < 1) failf("an engram table view of %lld rows of %d bytes", (long long) n_rows, row_bytes);
    rows_ = rows;
    n_rows_ = n_rows;
    row_bytes_ = row_bytes;
}

EngramGather::~EngramGather() {
    if (staging_) cudaFreeHost(staging_);
    cudaFree(dev_);
    if (done_) cudaEventDestroy((cudaEvent_t) done_);
}

bool EngramGather::init(int64_t max_rows, int row_bytes, int threads, std::string& err) {
    if (staging_) failf("EngramGather::init called twice");
    if (max_rows < 1 || row_bytes < 1) failf("an engram gather of %lld rows of %d bytes", (long long) max_rows, row_bytes);
    const size_t bytes = (size_t) max_rows * (size_t) row_bytes;
    // each buffer is the object's as soon as it exists, so the destructor frees it on every way out (the
    // pinned staging leaked when the device buffer failed)
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess) staging_ = static_cast<uint8_t*>(p);
    if (!staging_ || cudaMalloc(&dev_, bytes) != cudaSuccess) {
        err = "the engram gather buffers (" + std::to_string(bytes >> 10) + " KiB pinned and on the device) do not fit";
        return false;
    }
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        err = "cudaEventCreate failed";
        return false;
    }
    done_ = ev;
    max_rows_ = max_rows;
    row_bytes_ = row_bytes;
    threads_ = std::max(1, threads);
    return true;
}

void engram_copy_rows(const EngramTable& t, const int32_t* rows, int64_t n, uint8_t* dst) {
    const size_t rb = (size_t) t.row_bytes();
    for (int64_t i = 0; i < n; ++i) {
        if (rows[i] < 0 || rows[i] >= t.n_rows()) failf("engram row %d outside the table's %lld", rows[i], (long long) t.n_rows());
        std::memcpy(dst + (size_t) i * rb, t.rows() + (size_t) rows[i] * rb, rb);
    }
}

const void* EngramGather::gather(const EngramTable& t, const int32_t* rows, int64_t n, void* stream) {
    if (!staging_) failf("EngramGather::gather before init");
    if (n < 1 || n > max_rows_) failf("an engram gather of %lld rows (1..%lld)", (long long) n, (long long) max_rows_);
    if (t.row_bytes() != row_bytes_) failf("an engram table of %d-byte rows into a gather of %d", t.row_bytes(), row_bytes_);
    for (int64_t i = 0; i < n; ++i)
        if (rows[i] < 0 || rows[i] >= t.n_rows()) failf("engram row %d outside the table's %lld", rows[i], (long long) t.n_rows());
    // in the layer loop the last copy is long done (an expert call's routing fetch waited for the stream since)
    if (const cudaError_t q = cudaEventQuery((cudaEvent_t) done_); q == cudaErrorNotReady) {
        ck(cudaEventSynchronize((cudaEvent_t) done_), "the engram staging buffer's last copy");
        ++syncs_;
    } else {
        ck(q, "the engram staging buffer's last copy");
    }
    const auto t0 = std::chrono::steady_clock::now();
    const size_t rb = (size_t) row_bytes_;
    auto copy = [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; ++i) std::memcpy(staging_ + (size_t) i * rb, t.rows() + (size_t) rows[i] * rb, rb);
    };
    const int nt = (size_t) n * rb > (1u << 20) ? threads_ : 1;
    if (nt == 1) {
        copy(0, n);
    } else {
        std::vector<std::thread> th;
        for (int k = 0; k < nt; ++k) th.emplace_back(copy, n * k / nt, n * (k + 1) / nt);
        for (auto& x : th) x.join();
    }
    last_host_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ck(cudaMemcpyAsync(dev_, staging_, (size_t) n * rb, cudaMemcpyHostToDevice, (cudaStream_t) stream), "engram rows to the device");
    ck(cudaEventRecord((cudaEvent_t) done_, (cudaStream_t) stream), "cudaEventRecord");
    return dev_;
}

// ---------------------------------------------------------------- the device half
namespace {

// emb[r][v] for the n rows of key values (key / 256 blocks of 84 bytes each)
__global__ void engram_dequant_kernel(const uint8_t* __restrict__ rows, int64_t n, int key, float* __restrict__ yf,
                                      uint16_t* __restrict__ yh) {
    const int64_t total = n * key;
    const int64_t row_bytes = (int64_t) (key / 256) * 84;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = i / key;
        const int v = (int) (i % key);
        const float val = q2k_value(rows + r * row_bytes + (int64_t) (v / 256) * 84, v % 256);
        if (yf) yf[i] = val;
        else yh[i] = kernels::f16_from_f32(val);
    }
}

constexpr int GATE_T = 256;

__global__ void __launch_bounds__(GATE_T) engram_gate_kernel(const float* __restrict__ kv, const float* __restrict__ wq,
                                                             const float* __restrict__ wk, int E, float eps, float inv_sqrt,
                                                             float* x, float* __restrict__ gate) {
    __shared__ double sh[32];
    __shared__ float g_sh[kHc];
    const int64_t t = blockIdx.x;
    const float* key = kv + t * (int64_t) E * (kHc + 1);
    const float* value = key + (int64_t) kHc * E;
    float* xt = x + t * kHc * E;
    for (int c = 0; c < kHc; ++c) {
        const float* kc = key + (int64_t) c * E;
        const float* xc = xt + (int64_t) c * E;
        double sk = 0.0, sx = 0.0;
        for (int e = threadIdx.x; e < E; e += GATE_T) {
            sk += (double) __fmul_rn(kc[e], kc[e]);
            sx += (double) __fmul_rn(xc[e], xc[e]);
        }
        sk = block_sum_d(sk, sh);
        sx = block_sum_d(sx, sh);
        const float scale_k = 1.0f / sqrtf((float) (sk / (double) E) + eps);
        const float scale_x = 1.0f / sqrtf((float) (sx / (double) E) + eps);
        double dot = 0.0;
        for (int e = threadIdx.x; e < E; e += GATE_T) {
            const float kn = __fmul_rn(__fmul_rn(kc[e], scale_k), wk[(int64_t) c * E + e]);
            const float qn = __fmul_rn(__fmul_rn(xc[e], scale_x), wq[(int64_t) c * E + e]);
            dot += (double) __fmul_rn(kn, qn);
        }
        dot = block_sum_d(dot, sh);
        if (threadIdx.x == 0) {
            const float s = __fmul_rn((float) dot, inv_sqrt);
            const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
            const float sg = s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f);
            const float g = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-__fmul_rn(sg, mag))));
            g_sh[c] = g;
            gate[t * kHc + c] = g;
        }
    }
    __syncthreads();
    // every value of x was read above, by the thread that now writes it
    for (int c = 0; c < kHc; ++c)
        for (int e = threadIdx.x; e < E; e += GATE_T)
            xt[(int64_t) c * E + e] = __fadd_rn(xt[(int64_t) c * E + e], __fmul_rn(value[e], g_sh[c]));
}

}  // namespace

void engram_forward(DenseEngine& e, const EngramWeights& w, const void* rows_q2k, int64_t T, float* x) {
    OpTimer tm(e, &e.times_mut().engram_ms);
    const DenseDims& d = e.dims();
    engram_check_weights(d, w, "engram_forward");
    if (T < 1 || T > e.max_T()) failf("engram_forward: %lld tokens (1..%lld)", (long long) T, (long long) e.max_T());
    DenseScratch& s = e.scratch_mut();
    const cudaStream_t st = stream_of(e);
    const int C = d.engram_cols(), E = d.n_embd;
    const int64_t EK = (int64_t) C * d.engram_key, KV = (int64_t) E * (kHc + 1);
    const int64_t row_bytes = (int64_t) (d.engram_key / 256) * 84;
    const float inv_sqrt = 1.0f / std::sqrt((float) E);
    for (int64_t t0 = 0; t0 < T; t0 += e.engram_chunk()) {
        const int64_t n = std::min(e.engram_chunk(), T - t0);
        const bool h = f16_input(n);
        const int64_t total = n * EK;
        engram_dequant_kernel<<<(unsigned) std::min<int64_t>((total + 255) / 256, 8192), 256, 0, st>>>(
            static_cast<const uint8_t*>(rows_q2k) + t0 * C * row_bytes, n * C, d.engram_key, h ? nullptr : s.emb, h ? s.embh : nullptr);
        ck(cudaGetLastError(), "engram_dequant_kernel");
        project(e, w.wkv, s.emb, s.embh, n, s.engram_kv, KV, "engram_wkv");
        engram_gate_kernel<<<(unsigned) n, GATE_T, 0, st>>>(s.engram_kv, w.q, w.k, E, d.rms_eps, inv_sqrt, x + t0 * kHc * E,
                                                           s.engram_gate + t0 * kHc);
        ck(cudaGetLastError(), "engram_gate_kernel");
        s.engram_T = n;
    }
}

}  // namespace strata::dsv41::engine
