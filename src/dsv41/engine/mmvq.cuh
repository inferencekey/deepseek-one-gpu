// src/dsv41/engine/mmvq.cuh - the MMVQ arithmetic of qmatmul.cu as device code the engine's fused decode
// kernels share (qmatmul.cu, attention.cu, hc.cu): the q8_1 and K-quant block layouts, the Q2_K, Q3_K and Q6_K traits
// (Strata's native_mmvq.cu, itself from llama.cpp's ggml-cuda vecdotq.cuh and mmvq.cuh, MIT License, Copyright (c)
// 2023-2026 The ggml authors), the q8_1 quantizer of one warp, and mmvq1_kernel, the single-column MMVQ.
// Internal to src/dsv41/engine.
//
// mmvq1_kernel (one decode column: before it, attn_q_b ran at 355 GB/s, the q6_K head at 936 GB/s): a
// block of 4 warps takes R consecutive rows of one matrix, copies their R * row_bytes contiguous bytes to shared memory
// with 16-byte cp.async (every weight load of the block in flight at once, instead of the 4-byte loads of each
// iteration of mmvq_kernel), then runs mmvq_kernel's arithmetic on them: thread tid takes the blocks kbx = tid / T +
// i BPI of every row, the four warps' partial sums are added in warp order and the warp sums them by xor shuffles. Each
// row is therefore mmvq_kernel's value at one column bit for bit, whatever R. One launch may cover several matrices
// that read the same activations (MmvqSegs: the attention's projections of x, of qr), and an epilogue functor writes
// each row (Epi::row(args, segment, group, row, value, partner): partner is the value of row ^ 1, the other row of
// the block's pair, for the attention's fused rope).
#pragma once

#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace strata::dsv41::engine::mmvq {

constexpr int Q8K = 32;      // values per q8_1 block
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int THREADS = WARPS * WARP;

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q2KBlock {
    uint8_t scales[16];
    uint8_t qs[64];
    half2 dm;
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q2KBlock) == 84 && alignof(Q2KBlock) == 4 && offsetof(Q2KBlock, qs) == 16 &&
              offsetof(Q2KBlock, dm) == 80);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 && offsetof(Q6KBlock, qh) == 128 &&
              offsetof(Q6KBlock, scales) == 192 && offsetof(Q6KBlock, d) == 208);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    return x;
}

/// One q8_1 block from a warp, lane l holding value l: qmatmul.cu's quantize_q8_1_kernel for that block (the same
/// shuffles, scale, rounding and stored sum), so a fused kernel's activations are the quantizer's bit for bit.
__device__ __forceinline__ void q8_1_warp(float xi, Q81Block& b, int lane) {
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = kernels::q8_1_finite(amax / 127.0f);
    b.qs[lane] = kernels::q8_1_quant(xi, d, amax);
    if (lane == 0)
        b.ds = __halves2half2(__ushort_as_half(kernels::f16_from_f32(d)),
                              __ushort_as_half(kernels::f16_from_f32(kernels::q8_1_finite(sum))));
}

// Q3_K's and Q6_K's 110- and 210-byte strides give alternate blocks only two-byte alignment (native_mmvq.cu)
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

// every float operation of the traits below is an _rn intrinsic, so nvcc
// contracts none of them on its own and the CPU experts in the GPU's numerics (experts_cpu_mmvq.cpp, the reference
// mmvq_ref.cpp) repeat them one by one: per q8_1 block d8 times the block's exact integer dot, fused into the running
// sum (__fmaf_rn), then the block row's scales. The integer dots (dp4a) are exact. The kernels add the traits' values
// with plain adds, which nothing can contract (no product beside them).

// the fork's vec_dot_q2_K_q8_1_impl_mmvq: per q8_1 block i the 4 values of the thread times their 16-group's
// scale, minus the group's min times the sum of the 4 activations, scaled by d8[i]; then dm
struct Q2KTraits {
    using Block = Q2KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int v; int sc[4]; float2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.v = reinterpret_cast<const int*>(w->qs)[iqs];
#pragma unroll
        for (int i = 0; i < 4; ++i) r.sc[i] = w->scales[scale_offset + 2 * i];
        r.dm = __half22float2(w->dm);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int u = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            const float d8 = __low2float(x[r.bq8_offset + i].ds);
            const int vi = (r.v >> (2 * i)) & 0x03030303;
            sumf_d = __fmaf_rn(d8, __int2float_rn(STRATA_DP4A(vi, u, 0) * (r.sc[i] & 0xF)), sumf_d);
            int m = r.sc[i] >> 4;
            m |= m << 8;
            m |= m << 16;
            sumf_m = __fmaf_rn(d8, __int2float_rn(STRATA_DP4A(m, u, 0)), sumf_m);
        }
        return __fmaf_rn(r.dm.x, sumf_d, -__fmul_rn(r.dm.y, sumf_m));
    }
};

// native_mmvq.cu q3_q8_dot_impl and Q3KTraits
__device__ __forceinline__ float q3_q8_dot_impl(int vl, int vh, const int* __restrict__ u, const uint8_t* __restrict__ scales,
                                              int scale_offset, float d3, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf = __fmaf_rn(d8[i], __int2float_rn(STRATA_DP4A(vi, u[i], 0) * sc), sumf);
    }
    return __fmul_rn(d3, sumf);
}
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + i].ds);
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};

// native_mmvq.cu q6_q8_dot_impl and Q6KTraits
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u, const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf = __fmaf_rn(d8[i], __int2float_rn(STRATA_DP4A(vi, u[i], 0) * sc), sumf);
    }
    return __fmul_rn(d, sumf);
}
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};

// ---------------------------------------------------------------- mmvq1_kernel
constexpr int kMaxSegs = 6;
/// Shared memory staged per block at most (the weights of its R rows); mmvq1_plan picks R under it.
constexpr size_t kStageBytes = 12288;

/// One matrix of a launch: rows [n_out] of the launch's row bytes, its blocks first .. first + n_out / R - 1 of
/// blockIdx.x, y its output (the plain epilogue: y[row] = alpha * value), tag for the caller's epilogue.
struct MmvqSeg {
    const uint8_t* w = nullptr;
    float* y = nullptr;
    int n_out = 0;
    int first = 0;
    float alpha = 1.0f;
    int tag = 0;
};
/// The launch: x in q8_1 [groups][n_in / 32] (group g, blockIdx.y, reads rows g n_out .. of each matrix, activation
/// blocks g n_in / 32 .. and writes y + g n_out: attn_output_a's groups), n_seg matrices of n_in columns.
struct MmvqSegs {
    const Q81Block* x = nullptr;
    int n_in = 0;
    int n_seg = 0;
    MmvqSeg seg[kMaxSegs];
};

/// The plain epilogue: y[row] = value * alpha (qmatmul's scale_kernel after the matmul, the same product).
struct PlainEpi {
    struct Args {};
    __device__ static void row(const Args&, const MmvqSeg& s, int g, int r, float v, float) {
        s.y[(size_t) g * s.n_out + r] = v * s.alpha;
    }
};

/// 16 bytes from global to shared memory: cp.async on sm_80 and newer (completed by cp16_wait), a plain copy before
__device__ __forceinline__ void cp16(uint8_t* dst, const uint8_t* __restrict__ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
#else
    *reinterpret_cast<uint4*>(dst) = __ldg(reinterpret_cast<const uint4*>(src));
#endif
}
/// The block's cp16 copies done and visible to every thread.
__device__ __forceinline__ void cp16_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
    __syncthreads();
}
/// `bytes` (a multiple of 16) from 16-byte aligned src to dst, every thread of the block taking part.
__device__ __forceinline__ void stage16(uint8_t* dst, const uint8_t* __restrict__ src, int bytes) {
    for (int o = (int) threadIdx.x * 16; o < bytes; o += THREADS * 16) cp16(dst + o, src + o);
    cp16_wait();
}

/// Dynamic shared memory of a block of R rows of row_bytes: the staged rows, then the 4 warps' partial sums, which
/// reuse the rows' bytes once the products are done.
inline size_t mmvq1_smem(int R, size_t row_bytes) {
    const size_t stage = (size_t) R * row_bytes, part = (size_t) WARPS * R * WARP * sizeof(float);
    return stage > part ? stage : part;
}

template <typename F, int R, typename Epi>
__global__ void __launch_bounds__(THREADS) mmvq1_kernel(const __grid_constant__ MmvqSegs L,
                                                        const __grid_constant__ typename Epi::Args ea) {
    extern __shared__ __align__(16) uint8_t smem[];
    int si = 0;
    while (si + 1 < L.n_seg && (int) blockIdx.x >= L.seg[si + 1].first) ++si;
    const MmvqSeg& S = L.seg[si];
    const int g = (int) blockIdx.y;
    const int bpr = L.n_in / F::DIV;
    const int row_bytes = bpr * (int) sizeof(typename F::Block);
    const int row0 = ((int) blockIdx.x - S.first) * R;
    stage16(smem, S.w + ((size_t) g * S.n_out + row0) * (size_t) row_bytes, R * row_bytes);
    const Q81Block* __restrict__ x = L.x + (size_t) g * (L.n_in / Q8K);
    const int tid = (int) threadIdx.x, warp = tid / WARP, lane = tid % WARP;
    float tmp[R] = {};
    for (int kbx = tid / F::T; kbx < bpr; kbx += F::BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const auto* wr = reinterpret_cast<const typename F::Block*>(smem + i * row_bytes);
            tmp[i] += F::apply(F::load(wr + kbx, kqs), x + kby, kqs);
        }
    }
    __syncthreads();   // the rows are read: their bytes take the partials
    float* part = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int i = 0; i < R; ++i) part[(warp * R + i) * WARP + lane] = tmp[i];
    __syncthreads();
    // mmvq_kernel's order: warp 0's partial, then warps 1, 2, 3 added in turn, then the warp's xor sum (every lane
    // holds the same total); a warp takes a row pair (rope partners) at a time
    constexpr int P = R > 1 ? R / 2 : 1, PR = R > 1 ? 2 : 1;
    for (int pi = warp; pi < P; pi += WARPS) {
        float v[PR];
#pragma unroll
        for (int k = 0; k < PR; ++k) {
            const int i = pi * PR + k;
            float s = part[i * WARP + lane];
#pragma unroll
            for (int l = 1; l < WARPS; ++l) s += part[(l * R + i) * WARP + lane];
            v[k] = warp_sum(s);
        }
        if (lane < PR) {
            const float mine = PR == 1 || lane == 0 ? v[0] : v[PR - 1], other = PR == 1 || lane == 1 ? v[0] : v[PR - 1];
            Epi::row(ea, S, g, row0 + pi * PR + lane, mine, other);
        }
    }
}

/// The R a launch of matrices of `n_out` rows each (rows of row_bytes, data 16-byte aligned) takes, 0 when none fits:
/// the largest of 16, 8, 4, 2 (1 unless even is set) whose rows fit kStageBytes, divide every n_out and span a
/// multiple of 16 bytes (each block's rows start 16-byte aligned).
inline int mmvq1_plan(size_t row_bytes, const int* n_out, int n, bool even) {
    for (int R : {16, 8, 4, 2, 1}) {
        if (even && R == 1) continue;
        if ((size_t) R * row_bytes > kStageBytes || ((size_t) R * row_bytes) % 16) continue;
        bool ok = true;
        for (int i = 0; i < n; ++i) ok = ok && n_out[i] % R == 0;
        if (ok) return R;
    }
    return 0;
}

/// Queues mmvq1_kernel<F, R, Epi> over `groups` groups: L's segments with their `first` blocks set here.
template <typename F, typename Epi>
void mmvq1_launch(MmvqSegs L, int R, int groups, const typename Epi::Args& ea, cudaStream_t st) {
    int blocks = 0;
    for (int i = 0; i < L.n_seg; ++i) {
        L.seg[i].first = blocks;
        blocks += L.seg[i].n_out / R;
    }
    const size_t row_bytes = (size_t) (L.n_in / F::DIV) * sizeof(typename F::Block);
    const size_t smem = mmvq1_smem(R, row_bytes);
    const dim3 grid((unsigned) blocks, (unsigned) groups);
    switch (R) {
        case 16: mmvq1_kernel<F, 16, Epi><<<grid, THREADS, smem, st>>>(L, ea); break;
        case 8: mmvq1_kernel<F, 8, Epi><<<grid, THREADS, smem, st>>>(L, ea); break;
        case 4: mmvq1_kernel<F, 4, Epi><<<grid, THREADS, smem, st>>>(L, ea); break;
        case 2: mmvq1_kernel<F, 2, Epi><<<grid, THREADS, smem, st>>>(L, ea); break;
        default: mmvq1_kernel<F, 1, Epi><<<grid, THREADS, smem, st>>>(L, ea); break;
    }
}

}  // namespace strata::dsv41::engine::mmvq
