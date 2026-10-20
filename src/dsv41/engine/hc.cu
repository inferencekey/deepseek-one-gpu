// src/dsv41/engine/hc.cu - the hyper-connection ops of include/strata/dsv41/engine/dense.hpp (SPEC.md section 3).
//
//   hc_mixes          hc_norm_kernel (a block per token: rms over the HC * E values, squares in double, no weight;
//                     written as f32 for MMVQ calls, f16 for the dequant and MMQ paths), the hc_fn matmul on
//                     qmatmul, then hc_coeffs_kernel (a thread per token): pre and post as the fork's separate ggml
//                     ops (each product, sum and sigmoid rounded on its own) and the Sinkhorn of the fork's fused
//                     DSV4_HC_COMB, written after ggml-cuda dsv4-hc.cu's dsv4_hc_comb_f32 with the CPU's two FMAs;
//   hc_collapse_norm  collapse_norm_kernel (a block of 512 threads per token: the DSV4_HC_PRE FMA chain from copy 0,
//                     then the rms norm times its weight, the collapse kept in the scratch); native_gr_postops.cu
//                     pre_gated's shape, the fork's math;
//   hc_expand         expand_kernel (a thread per (token, value) for all four copies, so out may alias x): the
//                     DSV4_HC_POST order, y post[d] then + comb[d][s] x[s] by FMA for s = 0..3; native_gr_postops.cu
//                     post's shape.
// Every product and sum is written with explicit rounding (__fmul_rn, __fadd_rn, __fmaf_rn) so nvcc's contraction
// cannot move it away from the reference's order (dense_ref.cpp); expf is the device's (within 2 ulp of glibc's).
#include "dense_internal.cuh"
#include "mmvq.cuh"

#include <algorithm>

namespace strata::dsv41::engine {

using namespace dense_detail;

namespace {

constexpr int HC_T = 512;

__global__ void __launch_bounds__(HC_T) hc_norm_kernel(const float* __restrict__ x, int n, float eps, float* __restrict__ yf,
                                                       uint16_t* __restrict__ yh) {
    __shared__ double sh[32];
    const float* xr = x + (int64_t) blockIdx.x * n;
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += HC_T) s += (double) __fmul_rn(xr[i], xr[i]);
    s = block_sum_d(s, sh);
    const float mean = (float) (s / (double) n);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int i = threadIdx.x; i < n; i += HC_T) {
        const float v = __fmul_rn(xr[i], scale);
        if (yf) yf[(int64_t) blockIdx.x * n + i] = v;
        else yh[(int64_t) blockIdx.x * n + i] = kernels::f16_from_f32(v);
    }
}

__device__ __forceinline__ float sigmoidf_rn(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x))); }

// one token's pre, post and comb from its kHcMix mixes m (hc_coeffs_kernel's thread; hc_step_kernel's last block)
__device__ void hc_coeffs_one(const float* m, const float* __restrict__ scale, const float* __restrict__ base, float eps,
                              int iters, float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb) {
    for (int i = 0; i < kHc; ++i) {
        pre[i] = __fadd_rn(sigmoidf_rn(__fadd_rn(__fmul_rn(m[i], scale[0]), base[i])), eps);
        post[i] = __fmul_rn(2.0f, sigmoidf_rn(__fadd_rn(__fmul_rn(m[kHc + i], scale[1]), base[kHc + i])));
    }
    float c[kHc * kHc];
    for (int src = 0; src < kHc; ++src) {
        float mx = -INFINITY;
        for (int dst = 0; dst < kHc; ++dst) {
            const int idx = dst + kHc * src;
            c[idx] = __fmaf_rn(m[2 * kHc + idx], scale[2], base[2 * kHc + idx]);
            mx = fmaxf(mx, c[idx]);
        }
        float sum = 0.0f;
        for (int dst = 0; dst < kHc; ++dst) {
            const int idx = dst + kHc * src;
            c[idx] = expf(__fsub_rn(c[idx], mx));
            sum = __fadd_rn(sum, c[idx]);
        }
        const float inv = __fdiv_rn(1.0f, sum);
        for (int dst = 0; dst < kHc; ++dst) c[dst + kHc * src] = __fmaf_rn(c[dst + kHc * src], inv, eps);
    }
    for (int it = 0; it < iters; ++it) {
        if (it > 0)
            for (int src = 0; src < kHc; ++src) {   // rows: per src over dst
                float sum = eps;
                for (int dst = 0; dst < kHc; ++dst) sum = __fadd_rn(sum, c[dst + kHc * src]);
                const float inv = __fdiv_rn(1.0f, sum);
                for (int dst = 0; dst < kHc; ++dst) c[dst + kHc * src] = __fmul_rn(c[dst + kHc * src], inv);
            }
        for (int dst = 0; dst < kHc; ++dst) {       // columns: per dst over src
            float sum = eps;
            for (int src = 0; src < kHc; ++src) sum = __fadd_rn(sum, c[dst + kHc * src]);
            const float inv = __fdiv_rn(1.0f, sum);
            for (int src = 0; src < kHc; ++src) c[dst + kHc * src] = __fmul_rn(c[dst + kHc * src], inv);
        }
    }
    for (int i = 0; i < kHc * kHc; ++i) comb[i] = c[i];
}

// hc_coeffs_one on the 16 lanes of a warp, one comb value each (idx = dst + kHc src; lanes 16-31 repeat 0-15 so the
// whole warp takes part in the shuffles): every sum gathers its four values by shuffles and adds them in
// hc_coeffs_one's order, every lane of a row or column computing the same sum, so each value is hc_coeffs_one's bit
// for bit (the Sinkhorn's 20 iterations off one thread's serial chain). Lanes 0-3 write pre, 4-7 post.
__device__ void hc_coeffs_warp(const float* m, const float* __restrict__ scale, const float* __restrict__ base, float eps,
                               int iters, float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
                               int lane) {
    constexpr unsigned kAll = 0xffffffffu;
    const int idx = lane & 15, dst = idx % kHc, src = idx / kHc, half = lane & 16;
    if (lane < kHc) pre[lane] = __fadd_rn(sigmoidf_rn(__fadd_rn(__fmul_rn(m[lane], scale[0]), base[lane])), eps);
    else if (lane < 2 * kHc)
        post[lane - kHc] = __fmul_rn(2.0f, sigmoidf_rn(__fadd_rn(__fmul_rn(m[lane], scale[1]), base[lane])));
    float c = __fmaf_rn(m[2 * kHc + idx], scale[2], base[2 * kHc + idx]);
    // the value at (d, src) of this lane's row, at (dst, s) of its column
    auto row = [&](int d) { return __shfl_sync(kAll, c, half + d + kHc * src); };
    auto col = [&](int s2) { return __shfl_sync(kAll, c, half + dst + kHc * s2); };
    float mx = -INFINITY;
#pragma unroll
    for (int d = 0; d < kHc; ++d) mx = fmaxf(mx, row(d));
    c = expf(__fsub_rn(c, mx));
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < kHc; ++d) sum = __fadd_rn(sum, row(d));
    c = __fmaf_rn(c, __fdiv_rn(1.0f, sum), eps);
    for (int it = 0; it < iters; ++it) {
        if (it > 0) {
            float r = eps;
#pragma unroll
            for (int d = 0; d < kHc; ++d) r = __fadd_rn(r, row(d));
            c = __fmul_rn(c, __fdiv_rn(1.0f, r));
        }
        float k = eps;
#pragma unroll
        for (int s2 = 0; s2 < kHc; ++s2) k = __fadd_rn(k, col(s2));
        c = __fmul_rn(c, __fdiv_rn(1.0f, k));
    }
    if (lane < 16) comb[idx] = c;
}

__global__ void hc_coeffs_kernel(const float* __restrict__ mixes, const float* __restrict__ scale, const float* __restrict__ base,
                                 int64_t T, float eps, int iters, float* __restrict__ pre, float* __restrict__ post,
                                 float* __restrict__ comb) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    hc_coeffs_one(mixes + t * kHcMix, scale, base, eps, iters, pre + t * kHc, post + t * kHc, comb + t * kHc * kHc);
}

// 512 threads (256 before), the order hc_decode_step's norm block sums in
constexpr int COL_T = HC_T;

__global__ void __launch_bounds__(COL_T) collapse_norm_kernel(const float* __restrict__ x, const float* __restrict__ pre,
                                                              const float* __restrict__ w, int E, float eps,
                                                              float* __restrict__ col, float* __restrict__ out) {
    __shared__ double sh[32];
    const int64_t t = blockIdx.x;
    const float* xt = x + t * kHc * E;
    const float* p = pre + t * kHc;
    float* ct = col + t * E;
    double s = 0.0;
    for (int e = threadIdx.x; e < E; e += COL_T) {
        float v = __fmaf_rn(xt[e], p[0], 0.0f);
        for (int c = 1; c < kHc; ++c) v = __fmaf_rn(xt[(int64_t) c * E + e], p[c], v);
        ct[e] = v;
        s += (double) __fmul_rn(v, v);
    }
    s = block_sum_d(s, sh);
    const float mean = (float) (s / (double) E);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int e = threadIdx.x; e < E; e += COL_T) out[t * E + e] = __fmul_rn(__fmul_rn(ct[e], scale), w[e]);
}

__global__ void expand_kernel(const float* __restrict__ y, const float* x, const float* __restrict__ post,
                              const float* __restrict__ comb, int E, float* out) {
    const int64_t t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const float* xt = x + t * kHc * E;
    float xs[kHc];
    for (int s = 0; s < kHc; ++s) xs[s] = xt[(int64_t) s * E + e];
    const float yv = y[t * E + e];
    const float* p = post + t * kHc;
    const float* c = comb + t * kHc * kHc;
    float r[kHc];
    for (int d = 0; d < kHc; ++d) {
        float v = __fmul_rn(yv, p[d]);
        for (int s = 0; s < kHc; ++s) v = __fmaf_rn(xs[s], c[d + kHc * s], v);
        r[d] = v;
    }
    float* ot = out + t * kHc * E;
    for (int d = 0; d < kHc; ++d) ot[(int64_t) d * E + e] = r[d];
}

// ---------------------------------------------------------------- hc_decode_step
// One decode token's step before a sublayer in one launch of kStepGrid blocks of HC_T threads (before it,
// the hyper-connections and norms took 0.128 ms per layer in seven launches per sublayer pair, two of them a
// single thread or block). Strata's fused_gr.cu is the pattern (plan v0.3 P3: the previous half's write folded into
// the next read; S26: the q8_1 image of the output written for the next matmul; the last block by an atomic counter
// after a fence).
//   row blocks (kStepBlocks): the stream values v_i (the expansion of the previous sublayer, as expand_kernel
//       computes them, or x as is) for i = tid + k HC_T, their squares summed in double in hc_norm_kernel's order,
//       rms(x) = v_i scale, its q8_1 into shared memory (a warp per 32 consecutive values, qact_quantize's), then four
//       rows of hc_fn, one per quarter of the block, in mmvq_kernel's per-row arithmetic on the staged rows (mmvq.cuh).
//   the norm block (the grid's last): the collapse and norm, which need pre_c and the expanded stream but not this
//       step's mixes, at once beside the row blocks: per e = tid + j HC_T its four expanded values (the row blocks'
//       arithmetic), the collapse, its square summed in double in collapse_norm_kernel's order (HC_T threads), the
//       norm times its weight and its q8_1.
//   the last block to take the ticket (a count in device memory, reset by it; every block takes it once it has read
//       x and post/comb, the row blocks after writing their mixes): the expanded x written back, then
//       hc_coeffs_kernel's values on the mixes on one warp (hc_coeffs_warp).
// (0.146 ms per layer for the two steps, A100 graph replays, against <= 0.10; one launch per
// step already): the step's serial path shortened, the same values. The dims are template arguments (the loops fully
// unrolled with no exit, so each thread's loads and q8_1 shuffles are issued together instead of one value's chain at
// a time), and the collapse and norm left the last block's tail for a block of their own (before: after every row
// block's ticket, on 256 threads of the last block). With marks (DenseEngine::set_step_marks) thread 0 of every block
// writes %globaltimer at its phase ends (hc_step_split).
constexpr int kStepRows = HC_T / mmvq::THREADS;   // hc_fn rows per block
constexpr int kStepBlocks = kHcMix / kStepRows;
constexpr int kStepGrid = kStepBlocks + 1;
static_assert(kHcMix % kStepRows == 0);
static_assert(kStepGrid == kHcStepGrid, "dense.hpp's kHcStepGrid");

// the marks' slots per block (hc_step_split reads them)
enum StepMark : int { kMarkStart = 0, kMarkRms = 1, kMarkQ8 = 2, kMarkMixes = 3, kMarkTicket = 4, kMarkEnd = 5 };
static_assert(kHcStepMarks == 6, "dense.hpp's kHcStepMarks");

struct HcStepArgs {
    const uint8_t* fn;   // hc_fn [kHcMix][HC E] q2_K, 16-byte aligned
    const float* scale;
    const float* base;
    const float* y;      // null: no expansion
    float* x;
    const float* pre_c;
    const float* norm_w;
    float* pre;
    float* post;         // the expansion's input, then this step's output
    float* comb;
    float* normed;
    mmvq::Q81Block* normed_q;   // null: none
    float* hc_in;
    float* mixes;
    float* collapsed;
    unsigned* ticket;
    unsigned long long* marks;  // null: none; [kStepGrid][kHcStepMarks]
    float rms_eps, hc_eps;
    int iters;
};

size_t step_smem(int E) {
    const size_t n = (size_t) kHc * E, row_bytes = n / 256 * sizeof(mmvq::Q2KBlock);
    return kStepRows * row_bytes + n / mmvq::Q8K * sizeof(mmvq::Q81Block) + (size_t) (HC_T / 32) * 32 * 4;
}

__device__ __forceinline__ void step_mark(const HcStepArgs& a, int k) {
    if (!a.marks || threadIdx.x != 0) return;
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    a.marks[blockIdx.x * kHcStepMarks + k] = t;
}

// the last block's tail: the mixes in, hc_coeffs_warp on warp 0, the ticket reset
__device__ __forceinline__ void step_coeffs(const HcStepArgs& a, float* smix, int tid, int warp, int lane) {
    __threadfence();
    if (tid < kHcMix) smix[tid] = __ldcg(a.mixes + tid);
    __syncthreads();
    if (warp == 0) hc_coeffs_warp(smix, a.scale, a.base, a.hc_eps, a.iters, a.pre, a.post, a.comb, lane);
    if (tid == 0) *a.ticket = 0u;
}

template <int E> __global__ void __launch_bounds__(HC_T) hc_step_kernel(const HcStepArgs a) {
    constexpr int n = kHc * E, NK = n / HC_T, NJ = (E + HC_T - 1) / HC_T;
    constexpr int row_bytes = n / 256 * (int) sizeof(mmvq::Q2KBlock);
    static_assert(n % HC_T == 0 && E % 32 == 0, "hc_step_kernel's dims");
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ double sh[32];
    __shared__ float spc[kHc + kHc * kHc], smix[kHcMix];
    __shared__ int last;
    const int tid = (int) threadIdx.x, warp = tid / 32, lane = tid % 32;
    const bool norm_block = blockIdx.x == kStepBlocks;
    // a row block's hc_fn rows on their way
    uint8_t* wrows = smem;
    const int row0 = (int) blockIdx.x * kStepRows;
    if (!norm_block)
        for (int o = tid * 16; o < kStepRows * row_bytes; o += HC_T * 16) mmvq::cp16(wrows + o, a.fn + (size_t) row0 * row_bytes + o);
    if (a.y && tid < kHc + kHc * kHc) spc[tid] = tid < kHc ? a.post[tid] : a.comb[tid - kHc];
    __syncthreads();
    step_mark(a, kMarkStart);
    // value (d, e) of the expanded stream (expand_kernel's order), or x's
    auto expanded = [&](int d, int e) {
        if (!a.y) return a.x[d * E + e];
        float t = __fmul_rn(a.y[e], spc[d]);
#pragma unroll
        for (int c = 0; c < kHc; ++c) t = __fmaf_rn(a.x[c * E + e], spc[kHc + d + kHc * c], t);
        return t;
    };

    if (norm_block) {
        // the collapse and norm (collapse_norm_kernel on HC_T threads: e = tid + j HC_T)
        float xv[NJ][kHc], col[NJ], pc[kHc];
#pragma unroll
        for (int c = 0; c < kHc; ++c) pc[c] = a.pre_c[c];
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int e = tid + j * HC_T;
            if (E % HC_T == 0 || e < E)
#pragma unroll
                for (int d = 0; d < kHc; ++d) xv[j][d] = expanded(d, e);
        }
        __syncthreads();   // every read of x, post and comb done
        if (tid == 0) last = atomicAdd(a.ticket, 1u) == gridDim.x - 1;
        step_mark(a, kMarkTicket);
        double c2 = 0.0;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int e = tid + j * HC_T;
            col[j] = 0.0f;
            if (E % HC_T != 0 && e >= E) continue;
            float t = __fmaf_rn(xv[j][0], pc[0], 0.0f);
#pragma unroll
            for (int c = 1; c < kHc; ++c) t = __fmaf_rn(xv[j][c], pc[c], t);
            col[j] = t;
            a.collapsed[e] = t;
            c2 += (double) __fmul_rn(t, t);
        }
        c2 = block_sum_d(c2, sh);
        const float cs = 1.0f / sqrtf((float) (c2 / (double) E) + a.rms_eps);
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int e = tid + j * HC_T;
            if (E % HC_T != 0 && e >= E) continue;   // whole warps: E is a multiple of 32
            const float o = __fmul_rn(__fmul_rn(col[j], cs), a.norm_w[e]);
            a.normed[e] = o;
            if (a.normed_q) mmvq::q8_1_warp(o, a.normed_q[e / mmvq::Q8K], lane);
        }
        __syncthreads();
        step_mark(a, kMarkQ8);
        if (!last) {
            step_mark(a, kMarkEnd);
            return;
        }
        if (a.y)
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const int e = tid + j * HC_T;
                if (E % HC_T == 0 || e < E)
#pragma unroll
                    for (int d = 0; d < kHc; ++d) a.x[d * E + e] = xv[j][d];
            }
        step_coeffs(a, smix, tid, warp, lane);
        step_mark(a, kMarkEnd);
        return;
    }

    // a row block: the stream values and their squares (hc_norm_kernel)
    auto* sq = reinterpret_cast<mmvq::Q81Block*>(smem + kStepRows * row_bytes);
    auto* part = reinterpret_cast<float*>(sq + n / mmvq::Q8K);
    float v[NK];
#pragma unroll
    for (int k = 0; k < NK; ++k) {
        const int i = tid + k * HC_T;
        v[k] = expanded(i / E, i % E);
    }
    double s = 0.0;
#pragma unroll
    for (int k = 0; k < NK; ++k) s += (double) __fmul_rn(v[k], v[k]);
    s = block_sum_d(s, sh);
    const float scale = 1.0f / sqrtf((float) (s / (double) n) + a.rms_eps);
    step_mark(a, kMarkRms);
#pragma unroll
    for (int k = 0; k < NK; ++k) {
        const int i = tid + k * HC_T;
        const float r = __fmul_rn(v[k], scale);
        if (blockIdx.x == 0) a.hc_in[i] = r;
        mmvq::q8_1_warp(r, sq[i / mmvq::Q8K], lane);
    }
    mmvq::cp16_wait();
    step_mark(a, kMarkQ8);
    // row row0 + quarter of hc_fn: mmvq_kernel's per-row sums (its 128 threads, warp order, xor sum)
    {
        using F = mmvq::Q2KTraits;
        const int q = tid / mmvq::THREADS, t = tid % mmvq::THREADS;
        const auto* wr = reinterpret_cast<const mmvq::Q2KBlock*>(wrows + q * row_bytes);
        float acc = 0.0f;
        for (int kbx = t / F::T; kbx < n / F::DIV; kbx += F::BPI) acc += F::apply(F::load(wr + kbx, F::kqs(t)), sq + kbx * F::KBY, F::kqs(t));
        part[warp * 32 + lane] = acc;
        __syncthreads();
        if (warp % mmvq::WARPS == 0) {
            float m = part[warp * 32 + lane];
#pragma unroll
            for (int l = 1; l < mmvq::WARPS; ++l) m += part[(warp + l) * 32 + lane];
            m = mmvq::warp_sum(m);
            if (lane == 0) a.mixes[row0 + q] = m;
        }
    }
    __threadfence();
    __syncthreads();
    step_mark(a, kMarkMixes);
    if (tid == 0) last = atomicAdd(a.ticket, 1u) == gridDim.x - 1;
    __syncthreads();
    step_mark(a, kMarkTicket);
    if (!last) {
        step_mark(a, kMarkEnd);
        return;
    }
    // the last block: the expanded stream, then the coefficients
    if (a.y)
#pragma unroll
        for (int k = 0; k < NK; ++k) a.x[tid + k * HC_T] = v[k];
    step_coeffs(a, smix, tid, warp, lane);
    step_mark(a, kMarkEnd);
}

// ---------------------------------------------------------------- hc_prefill_step
// hc_decode_step's fusion for a prefill chunk (hc and norms took 457 ms of the 16K
// prompt on the A100 real model, ~11 ms per layer, each sublayer reading the 1.3 GiB stream three times): a block
// per token, HC_T threads, each owning the values e = tid + j HC_T of all four copies. From one read of x (and y):
// the expansion (expand_kernel's arithmetic), written back into x; the squares of the expanded stream in
// hc_norm_kernel's order (value i = tid + k HC_T is copy k / NJ, e = tid + (k % NJ) HC_T) and its normed row in f32
// or f16 for the hc_fn matmul; the collapse with pre_c, its squares in collapse_norm_kernel's order and the norm.
// The same values as the unfused ops, bit for bit. E a multiple of HC_T (the real 5120; 512).
template <int E>
__global__ void __launch_bounds__(HC_T) hc_prefill_kernel(const float* __restrict__ y, float* x, const float* __restrict__ post,
                                                          const float* __restrict__ comb, const float* __restrict__ pre_c,
                                                          const float* __restrict__ norm_w, float eps, float* __restrict__ hc_in,
                                                          uint16_t* __restrict__ hc_inh, float* __restrict__ collapsed,
                                                          float* __restrict__ normed) {
    constexpr int n = kHc * E, NJ = E / HC_T, NK = kHc * NJ;
    static_assert(E % HC_T == 0, "hc_prefill_kernel's dims");
    __shared__ double sh[32];
    __shared__ float spc[kHc + kHc * kHc];
    const int64_t t = blockIdx.x;
    const int tid = (int) threadIdx.x;
    float* xt = x + t * n;
    if (y && tid < kHc + kHc * kHc) spc[tid] = tid < kHc ? post[t * kHc + tid] : comb[t * kHc * kHc + tid - kHc];
    __syncthreads();
    float v[NK];
#pragma unroll
    for (int k = 0; k < NK; ++k) v[k] = xt[(k / NJ) * E + tid + (k % NJ) * HC_T];
    if (y) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const float yv = y[t * E + tid + j * HC_T];
            float r[kHc];
#pragma unroll
            for (int d = 0; d < kHc; ++d) {
                float u = __fmul_rn(yv, spc[d]);
#pragma unroll
                for (int c = 0; c < kHc; ++c) u = __fmaf_rn(v[c * NJ + j], spc[kHc + d + kHc * c], u);
                r[d] = u;
            }
#pragma unroll
            for (int d = 0; d < kHc; ++d) v[d * NJ + j] = r[d];
        }
#pragma unroll
        for (int k = 0; k < NK; ++k) xt[(k / NJ) * E + tid + (k % NJ) * HC_T] = v[k];
    }
    double s = 0.0;
#pragma unroll
    for (int k = 0; k < NK; ++k) s += (double) __fmul_rn(v[k], v[k]);
    s = block_sum_d(s, sh);
    const float scale = 1.0f / sqrtf((float) (s / (double) n) + eps);
#pragma unroll
    for (int k = 0; k < NK; ++k) {
        const int64_t i = t * n + (k / NJ) * E + tid + (k % NJ) * HC_T;
        const float r = __fmul_rn(v[k], scale);
        if (hc_in) hc_in[i] = r;
        else hc_inh[i] = kernels::f16_from_f32(r);
    }
    float pc[kHc], col[NJ];
#pragma unroll
    for (int c = 0; c < kHc; ++c) pc[c] = pre_c[t * kHc + c];
    double c2 = 0.0;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        float u = __fmaf_rn(v[j], pc[0], 0.0f);
#pragma unroll
        for (int c = 1; c < kHc; ++c) u = __fmaf_rn(v[c * NJ + j], pc[c], u);
        col[j] = u;
        collapsed[t * E + tid + j * HC_T] = u;
        c2 += (double) __fmul_rn(u, u);
    }
    c2 = block_sum_d(c2, sh);
    const float cs = 1.0f / sqrtf((float) (c2 / (double) E) + eps);
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int e = tid + j * HC_T;
        normed[t * E + e] = __fmul_rn(__fmul_rn(col[j], cs), norm_w[e]);
    }
}

void check_T(const DenseEngine& e, int64_t T, const char* what) {
    if (T < 1 || T > e.max_T()) failf("%s: %lld tokens (1..%lld)", what, (long long) T, (long long) e.max_T());
}

}  // namespace

void hc_mixes(DenseEngine& e, const HcWeights& w, const float* x, int64_t T, float* pre, float* post, float* comb) {
    OpTimer tm(e, &e.times_mut().mixes_ms);
    const DenseDims& d = e.dims();
    hc_check_weights(d, w, "hc_mixes");
    check_T(e, T, "hc_mixes");
    DenseScratch& s = e.scratch_mut();
    const int n = kHc * d.n_embd;
    const bool h = f16_input(T);
    hc_norm_kernel<<<(unsigned) T, HC_T, 0, stream_of(e)>>>(x, n, d.rms_eps, h ? nullptr : s.hc_in, h ? s.hc_inh : nullptr);
    ck(cudaGetLastError(), "hc_norm_kernel");
    project(e, w.fn, s.hc_in, s.hc_inh, T, s.mixes, kHcMix, "hc_fn");
    hc_coeffs_kernel<<<(unsigned) ((T + 127) / 128), 128, 0, stream_of(e)>>>(s.mixes, w.scale, w.base, T, d.hc_eps,
                                                                           d.sinkhorn_iters, pre, post, comb);
    ck(cudaGetLastError(), "hc_coeffs_kernel");
}

void hc_collapse_norm(DenseEngine& e, const float* x, const float* pre, const float* norm_w, int64_t T, float* normed) {
    OpTimer tm(e, &e.times_mut().collapse_ms);
    check_T(e, T, "hc_collapse_norm");
    collapse_norm_kernel<<<(unsigned) T, COL_T, 0, stream_of(e)>>>(x, pre, norm_w, e.dims().n_embd, e.dims().rms_eps,
                                                                  e.scratch_mut().collapsed, normed);
    ck(cudaGetLastError(), "collapse_norm_kernel");
}

void hc_prefill_step(DenseEngine& e, const HcWeights& w, const float* y, float* x, const float* pre_c, const float* norm_w,
                     int64_t T, float* pre, float* post, float* comb, float* normed) {
    const DenseDims& d = e.dims();
    const int E = d.n_embd;
    if (E != 5120 && E != 512) {   // the fused kernel's dims only; any other runs the ops it fuses
        if (y) hc_expand(e, y, x, post, comb, T, x);
        hc_mixes(e, w, x, T, pre, post, comb);
        hc_collapse_norm(e, x, pre_c, norm_w, T, normed);
        return;
    }
    hc_check_weights(d, w, "hc_prefill_step");
    check_T(e, T, "hc_prefill_step");
    DenseScratch& s = e.scratch_mut();
    {
        OpTimer tm(e, &e.times_mut().step_ms);
        const bool h = f16_input(T);
        auto k = E == 5120 ? hc_prefill_kernel<5120> : hc_prefill_kernel<512>;
        k<<<(unsigned) T, HC_T, 0, stream_of(e)>>>(y, x, post, comb, pre_c, norm_w, d.rms_eps, h ? nullptr : s.hc_in,
                                                  h ? s.hc_inh : nullptr, s.collapsed, normed);
        ck(cudaGetLastError(), "hc_prefill_kernel");
    }
    OpTimer tm(e, &e.times_mut().mixes_ms);
    project(e, w.fn, s.hc_in, s.hc_inh, T, s.mixes, kHcMix, "hc_fn");
    hc_coeffs_kernel<<<(unsigned) ((T + 127) / 128), 128, 0, stream_of(e)>>>(s.mixes, w.scale, w.base, T, d.hc_eps,
                                                                           d.sinkhorn_iters, pre, post, comb);
    ck(cudaGetLastError(), "hc_coeffs_kernel");
}

void hc_expand(DenseEngine& e, const float* y, const float* x, const float* post, const float* comb, int64_t T,
               float* out) {
    OpTimer tm(e, &e.times_mut().expand_ms);
    check_T(e, T, "hc_expand");
    const int E = e.dims().n_embd;
    expand_kernel<<<dim3((unsigned) ((E + 255) / 256), (unsigned) T), 256, 0, stream_of(e)>>>(y, x, post, comb, E, out);
    ck(cudaGetLastError(), "expand_kernel");
}


namespace {

// hc_step_kernel's instance for n_embd: the real model's 5120 and the tiny models' 256 and 512 (null: another, which
// takes the unfused ops)
using StepKernel = void (*)(HcStepArgs);
StepKernel step_kernel(int E) {
    switch (E) {
        case 256: return hc_step_kernel<256>;
        case 512: return hc_step_kernel<512>;
        case 5120: return hc_step_kernel<5120>;
        default: return nullptr;
    }
}

}  // namespace

namespace dense_detail {

void hc_step_attrs(int n_embd) {
    if (const StepKernel k = step_kernel(n_embd))
        ck(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) step_smem(n_embd)),
           "cudaFuncSetAttribute(hc_step_kernel)");
}

}  // namespace dense_detail

void hc_decode_step(DenseEngine& e, const HcWeights& w, const float* y, float* x, const float* pre_c, const float* norm_w,
                    float* pre, float* post, float* comb, float* normed, void* normed_q) {
    const DenseDims& d = e.dims();
    hc_check_weights(d, w, "hc_decode_step");
    const int E = d.n_embd;
    const StepKernel kernel = step_kernel(E);
    if (!kernel || (uintptr_t) w.fn.data % 16) {
        // dims the one launch does not take: the unfused ops in the same order
        if (y) hc_expand(e, y, x, post, comb, 1, x);
        hc_mixes(e, w, x, 1, pre, post, comb);
        hc_collapse_norm(e, x, pre_c, norm_w, 1, normed);
        if (normed_q) {
            QActs q;
            qact_quantize(normed, nullptr, E, 1, E, QActLayout::Q8_1, normed_q, e.stream(), q);
        }
        return;
    }
    OpTimer tm(e, &e.times_mut().step_ms);
    DenseScratch& s = e.scratch_mut();
    HcStepArgs a{};
    a.fn = static_cast<const uint8_t*>(w.fn.data);
    a.scale = w.scale;
    a.base = w.base;
    a.y = y;
    a.x = x;
    a.pre_c = pre_c;
    a.norm_w = norm_w;
    a.pre = pre;
    a.post = post;
    a.comb = comb;
    a.normed = normed;
    a.normed_q = static_cast<mmvq::Q81Block*>(normed_q);
    a.hc_in = s.hc_in;
    a.mixes = s.mixes;
    a.collapsed = s.collapsed;
    a.ticket = s.hc_ticket;
    a.marks = e.step_marks() ? s.hc_marks : nullptr;
    a.rms_eps = d.rms_eps;
    a.hc_eps = d.hc_eps;
    a.iters = d.sinkhorn_iters;
    kernel<<<kStepGrid, HC_T, step_smem(E), stream_of(e)>>>(a);
    ck(cudaGetLastError(), "hc_step_kernel");
}

HcStepSplit hc_step_split(DenseEngine& e) {
    unsigned long long m[kStepGrid * kHcStepMarks];
    ck(cudaStreamSynchronize(stream_of(e)), "hc_step_split");
    ck(cudaMemcpy(m, e.scratch().hc_marks, sizeof m, cudaMemcpyDeviceToHost), "hc_step_split");
    auto at = [&](int b, int k) { return m[b * kHcStepMarks + k]; };
    unsigned long long t0 = at(0, kMarkStart), rms = 0, q8 = 0, mixes = 0, ticket = 0, end = 0;
    for (int b = 0; b < kStepGrid; ++b) {
        t0 = std::min(t0, at(b, kMarkStart));
        end = std::max(end, at(b, kMarkEnd));
        if (b == kStepBlocks) continue;
        rms = std::max(rms, at(b, kMarkRms));
        q8 = std::max(q8, at(b, kMarkQ8));
        mixes = std::max(mixes, at(b, kMarkMixes));
        ticket = std::max(ticket, at(b, kMarkTicket));
    }
    auto us = [&](unsigned long long t) { return t >= t0 ? (double) (t - t0) / 1e3 : 0.0; };
    HcStepSplit r;
    r.rms_us = us(rms);
    r.q8_us = us(q8);
    r.mixes_us = us(mixes);
    r.ticket_us = us(ticket);
    r.norm_us = us(at(kStepBlocks, kMarkQ8));
    r.end_us = us(end);
    return r;
}
}  // namespace strata::dsv41::engine
