// src/dsv41/engine/qmatmul.cu - matmul on q2_K / q3_K / q6_K weights, see include/strata/dsv41/engine/qmatmul.hpp.
//
// Adapted from Strata's src/kernels/cuda/native_mmvq.cu (itself adapted from llama.cpp's ggml-cuda quantize.cu,
// vecdotq.cuh and mmvq.cu, MIT License, Copyright (c) 2023-2026 The ggml authors): the q8_1 quantizer
// (native_quantize_q8_1_kernel), the multi-column MMVQ kernel (native_mmvq_multi_kernel, its exact layout of four
// warps and two rows per block) and its Q3_K and Q6_K traits, with a Q2_K trait written in their pattern from the
// fork's vec_dot_q2_K_q8_1 (vecdotq.cuh). Changes for the engine: a row stride and f16 input for the quantizer, a
// column stride and groups (blockIdx.y) for the MMVQ kernel, and the MMQ tile quantizer of the fork's
// quantize_mmq_q8_1 (quantize.cu, the D4 and D2S6 layouts) without fast math, so the CPU reference repeats it. The
// MMQ prefill option is the fork's MMQ itself (qmatmul_mmq.cu). The default prefill path dequantizes as Strata's
// src/kernels/cuda/dequant_bf16.cu (its q3_K and q6_K arithmetic in its order, q2_K added from ggml-quants.c's
// dequantize_row_q2_K) and multiplies with cublasGemmEx as Strata's Gemm::native and Gemm::f16 (src/prefill/gemm.cu).
#include "strata/dsv41/engine/qmatmul.hpp"

#include "ffn_device.cuh"
#include "mmvq.cuh"
#include "qdequant.cuh"
#include "qmatmul_mmq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 qmatmul: %s\n", msg);
    std::exit(1);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) failf("%s: %s", what, cudaGetErrorString(e));
}
void ckb(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) failf("%s: cuBLAS status %d", what, (int) s);
}

using mmvq::Q81Block;
using mmvq::Q8K;
using mmvq::Q2KTraits;
using mmvq::Q3KTraits;
using mmvq::Q6KTraits;
using mmvq::WARP;
using mmvq::WARPS;
using mmvq::warp_sum;
constexpr int ROWS = 2;      // rows per MMVQ block (native_mmvq.cu #783: two rows for every K)
constexpr int MMQ_TILE = 128;
constexpr int64_t kMmqPad = 512;   // MATRIX_ROW_PADDING
constexpr int64_t kMmqTailBlocks = 128;

/// block_q8_1_mmq (mmq.cuh): 16 bytes of scales (d4: 4 floats; d2s6: 8 halves), then 128 int8 values
struct MmqTile {
    union {
        float d4[4];
        half d2s6[8];
    };
    int8_t qs[MMQ_TILE];
};
static_assert(sizeof(MmqTile) == 144);

int64_t pad512(int64_t n) { return (n + kMmqPad - 1) / kMmqPad * kMmqPad; }

__device__ __forceinline__ float load_in(const float* p) { return *p; }
__device__ __forceinline__ float load_in(const uint16_t* p) { return __half2float(__ushort_as_half(*p)); }

// ---------------------------------------------------------------- quantizers
// q8_1 (native_quantize_q8_1_kernel): a thread per value, a warp per block of 32, row t of x at x + t ld
constexpr int QUANT_THREADS = 256;
template <typename In>
__global__ void __launch_bounds__(QUANT_THREADS) quantize_q8_1_kernel(const In* __restrict__ x, int64_t ld, int64_t k,
                                                                      Q81Block* __restrict__ y) {
    const int64_t t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * QUANT_THREADS + threadIdx.x;
    if (i >= k) return;   // k is a multiple of 256: whole blocks return
    // q8_1_ds's finite d and sum, rounded by f16_bits.hpp's converter (the one the attention and the MMQ tiles use)
    mmvq::q8_1_warp(load_in(x + t * ld + i), y[t * (k / Q8K) + i / Q8K], (int) (i % Q8K));
}

// MMQ tiles (the fork's quantize_mmq_q8_1, not scattered): a thread per 4 values over k padded to 512 (zeros past
// k), tile (k_tile, t) at k_tile * T + t. The scale is 1 / (127 / amax) over 32 values (D4) or 64 (D2S6, stored as
// f16), q = round(x * 127 / amax); D2S6 also keeps the f16 sum of each 16 raw values among the tile's first 96.
// An all-zero group gets q = 0 and d = 0 (what the fork's 0 * inf conversion yields).
constexpr int MMQ_QUANT_THREADS = 128;
template <typename In, bool D2S6>
__global__ void __launch_bounds__(MMQ_QUANT_THREADS) quantize_mmq_kernel(const In* __restrict__ x, int64_t ld, int64_t k,
                                                                         int64_t k_pad, int64_t T, MmqTile* __restrict__ y) {
    constexpr int vals_per_scale = D2S6 ? 64 : 32;
    const int64_t t = blockIdx.x;
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= k_pad) return;   // the grid covers k_pad (a multiple of 512) exactly: no thread leaves a shuffle
    const In* xr = x + t * ld;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = i0 < k ? load_in(xr + i0 + j) : 0.0f;
    float amax = fmaxf(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fabsf(v[2])), fabsf(v[3]));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, WARP));
    float sum = 0.0f;
    if (D2S6) {
        sum = v[0] + v[1] + v[2] + v[3];
#pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, offset, WARP);
    }
    const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
    const float d = amax == 0.0f ? 0.0f : 1.0f / d_inv;
    MmqTile& b = y[(i0 / MMQ_TILE) * T + t];
    const int64_t iqs = i0 % MMQ_TILE;
    char4 q;
    q.x = (signed char) roundf(v[0] * d_inv);
    q.y = (signed char) roundf(v[1] * d_inv);
    q.z = (signed char) roundf(v[2] * d_inv);
    q.w = (signed char) roundf(v[3] * d_inv);
    reinterpret_cast<char4*>(b.qs)[iqs / 4] = q;
    if (D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            b.d2s6[2 + iqs / 16] = __ushort_as_half(kernels::f16_from_f32(sum));
            if (iqs % 64 == 0) b.d2s6[iqs / 64] = __ushort_as_half(kernels::f16_from_f32(d));
        }
    } else if (iqs % 32 == 0) {
        b.d4[iqs / 32] = d;
    }
}

// ---------------------------------------------------------------- MMVQ (the traits: mmvq.cuh)
// native_mmvq_multi_kernel's exact layout (WARPS warps, ROWS rows per block): a weight block is decoded once per
// (row, block) and applied to every column. Group g = blockIdx.y reads rows g * n_out .., activation blocks
// g * n_in / 32 .. of each column and writes y columns g * n_out ..
template <typename F, int NCOLS>
__global__ void __launch_bounds__(WARPS * WARP, 4) mmvq_kernel(const typename F::Block* __restrict__ w,
                                                               const Q81Block* __restrict__ x, float* __restrict__ y,
                                                               int n_in, int n_out, int64_t x_stride, int64_t ldy) {
    const int g = (int) blockIdx.y;
    const int blocks_per_row = n_in / F::DIV;
    w += (size_t) g * n_out * blocks_per_row;
    x += (size_t) g * (n_in / Q8K);
    y += (size_t) g * n_out;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + (size_t) (row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + (size_t) j * x_stride + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < WARPS - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[(size_t) j * ldy + row0 + i] = tmp[j][i];
        }
    }
}

template <typename F, int NCOLS>
void mmvq_launch_n(const void* w, const void* x, float* y, int n_in, int n_out, int groups, int64_t x_stride,
                   int64_t ldy, cudaStream_t s) {
    const dim3 grid((unsigned) ((n_out + ROWS - 1) / ROWS), (unsigned) groups);
    mmvq_kernel<F, NCOLS><<<grid, dim3(WARP, WARPS), 0, s>>>(static_cast<const typename F::Block*>(w),
                                                            static_cast<const Q81Block*>(x), y, n_in, n_out, x_stride, ldy);
}

template <typename F>
void mmvq_launch(const void* w, const void* x, float* y, int n_in, int n_out, int groups, int ncols, int64_t x_stride,
                 int64_t ldy, cudaStream_t s) {
    switch (ncols) {
        case 1: mmvq_launch_n<F, 1>(w, x, y, n_in, n_out, groups, x_stride, ldy, s); break;
        case 2: mmvq_launch_n<F, 2>(w, x, y, n_in, n_out, groups, x_stride, ldy, s); break;
        case 3: mmvq_launch_n<F, 3>(w, x, y, n_in, n_out, groups, x_stride, ldy, s); break;
        case 4: mmvq_launch_n<F, 4>(w, x, y, n_in, n_out, groups, x_stride, ldy, s); break;
        case 5: mmvq_launch_n<F, 5>(w, x, y, n_in, n_out, groups, x_stride, ldy, s); break;
        case 6: mmvq_launch_n<F, 6>(w, x, y, n_in, n_out, groups, x_stride, ldy, s); break;
        case 7: mmvq_launch_n<F, 7>(w, x, y, n_in, n_out, groups, x_stride, ldy, s); break;
        case 8: mmvq_launch_n<F, 8>(w, x, y, n_in, n_out, groups, x_stride, ldy, s); break;
        default: failf("MMVQ takes 1..%lld columns, not %d", (long long) kMmvqMaxCols, ncols);
    }
}

// one column on mmvq1_kernel (mmvq.cuh, the same sums as mmvq_kernel's), when its rows stage: false
// (mmvq_kernel then) for a matrix whose rows fit no block size or whose data is not 16-byte aligned
template <typename F>
bool mmvq1_try(const void* w, const void* x, float* y, int n_in, int n_out, int groups, cudaStream_t s) {
    const size_t row_bytes = (size_t) (n_in / F::DIV) * sizeof(typename F::Block);
    const int R = mmvq::mmvq1_plan(row_bytes, &n_out, 1, false);
    if (!R || (uintptr_t) w % 16) return false;
    mmvq::MmvqSegs L;
    L.x = static_cast<const Q81Block*>(x);
    L.n_in = n_in;
    L.n_seg = 1;
    L.seg[0].w = static_cast<const uint8_t*>(w);
    L.seg[0].y = y;
    L.seg[0].n_out = n_out;
    mmvq::mmvq1_launch<F, mmvq::PlainEpi>(L, R, groups, {}, s);
    return true;
}

// ---------------------------------------------------------------- decode's routed experts, grouped
// Strata's native_expert_grouped pattern (iq_kernels.hpp: a residency row on the device, an expert's parts at its slot
// of a VRAM arena, one block row per routed entry) on mmvq_kernel's layout: block row p is pair p of the call, its
// expert ids[p] at slot slot_of[ids[p]]; a pair outside the slab (slot -1, the CPU's) returns at once. Each output
// row is mmvq_kernel's sum at NCOLS = 1 in its order, so a hit is qmatmul's value bit for bit (a column of a
// multi-column call sums alone). GLU: the block reads two matrices of the slot (gate at off_a, up at off_b) and
// writes swiglu_clamp(gate, up) (ffn_device.cuh), the decode path's clamp kernel fused into the epilogue.
template <typename F, bool GLU>
__global__ void __launch_bounds__(WARPS * WARP, 4) mmvq_experts_kernel(const uint8_t* __restrict__ base,
                                                                       const int32_t* __restrict__ slot_of,
                                                                       const int32_t* __restrict__ ids, size_t slot_bytes,
                                                                       size_t off_a, size_t off_b,
                                                                       const Q81Block* __restrict__ x, int x_div,
                                                                       int n_in, int n_out, float limit,
                                                                       float* __restrict__ y) {
    const int p = (int) blockIdx.y;
    const int32_t slot = slot_of[ids[p]];
    if (slot < 0) return;
    const int blocks_per_row = n_in / F::DIV;
    const uint8_t* ex = base + (size_t) slot * slot_bytes;
    const auto* wa = reinterpret_cast<const typename F::Block*>(ex + off_a);
    const auto* wb = reinterpret_cast<const typename F::Block*>(ex + off_b);
    x += (size_t) (p / x_div) * (size_t) (n_in / Q8K);
    y += (size_t) p * n_out;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    float ta[ROWS] = {}, tb[ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const size_t at = (size_t) (row0 + i) * blocks_per_row + kbx;
                ta[i] += F::apply(F::load(wa + at, kqs), x + kby, kqs);
                if constexpr (GLU) tb[i] += F::apply(F::load(wb + at, kqs), x + kby, kqs);
            }
        }
    }
    __shared__ float partial[GLU ? 2 : 1][WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            partial[0][threadIdx.y - 1][i][threadIdx.x] = ta[i];
            if constexpr (GLU) partial[1][threadIdx.y - 1][i][threadIdx.x] = tb[i];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) {
            ta[i] += partial[0][l][i][threadIdx.x];
            if constexpr (GLU) tb[i] += partial[1][l][i][threadIdx.x];
        }
        ta[i] = warp_sum(ta[i]);
        if constexpr (GLU) tb[i] = warp_sum(tb[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = GLU ? swiglu_clamp(ta[i], tb[i], limit) : ta[i];
    }
}

// mmvq_experts_kernel on staged rows (mmvq.cuh mmvq1_kernel's pattern): R rows of the pair's matrix
// (of both with GLU) copied to shared memory at once, then the same per-row sums in the same order, so each row is
// mmvq_experts_kernel's bit for bit
template <typename F, bool GLU, int R>
__global__ void __launch_bounds__(mmvq::THREADS) mmvq1_experts_kernel(const uint8_t* __restrict__ base,
                                                                      const int32_t* __restrict__ slot_of,
                                                                      const int32_t* __restrict__ ids, size_t slot_bytes,
                                                                      size_t off_a, size_t off_b,
                                                                      const Q81Block* __restrict__ x, int x_div,
                                                                      int n_in, int n_out, float limit,
                                                                      float* __restrict__ y) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int p = (int) blockIdx.y;
    const int32_t slot = slot_of[ids[p]];
    if (slot < 0) return;
    const int bpr = n_in / F::DIV, row_bytes = bpr * (int) sizeof(typename F::Block);
    const int row0 = R * (int) blockIdx.x;
    const uint8_t* ex = base + (size_t) slot * slot_bytes + (size_t) row0 * row_bytes;
    // both matrices' rows before the one wait
    for (int o = (int) threadIdx.x * 16; o < R * row_bytes; o += mmvq::THREADS * 16) {
        mmvq::cp16(smem + o, ex + off_a + o);
        if constexpr (GLU) mmvq::cp16(smem + R * row_bytes + o, ex + off_b + o);
    }
    mmvq::cp16_wait();
    x += (size_t) (p / x_div) * (size_t) (n_in / Q8K);
    y += (size_t) p * n_out;
    const int tid = (int) threadIdx.x, warp = tid / WARP, lane = tid % WARP;
    float ta[R] = {}, tb[R] = {};
    for (int kbx = tid / F::T; kbx < bpr; kbx += F::BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const auto* wa = reinterpret_cast<const typename F::Block*>(smem + i * row_bytes);
            ta[i] += F::apply(F::load(wa + kbx, kqs), x + kby, kqs);
            if constexpr (GLU) {
                const auto* wb = reinterpret_cast<const typename F::Block*>(smem + (R + i) * row_bytes);
                tb[i] += F::apply(F::load(wb + kbx, kqs), x + kby, kqs);
            }
        }
    }
    __syncthreads();   // the rows are read: their bytes take the partials
    float* part = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int i = 0; i < R; ++i) {
        part[(warp * R + i) * WARP + lane] = ta[i];
        if constexpr (GLU) part[((WARPS + warp) * R + i) * WARP + lane] = tb[i];
    }
    __syncthreads();
    for (int i = warp; i < R; i += WARPS) {
        float a = part[i * WARP + lane], b = 0.f;
        if constexpr (GLU) b = part[(WARPS * R + i) * WARP + lane];
#pragma unroll
        for (int l = 1; l < WARPS; ++l) {
            a += part[(l * R + i) * WARP + lane];
            if constexpr (GLU) b += part[((WARPS + l) * R + i) * WARP + lane];
        }
        a = warp_sum(a);
        if constexpr (GLU) b = warp_sum(b);
        if (lane == 0) y[row0 + i] = GLU ? swiglu_clamp(a, b, limit) : a;
    }
}

template <typename F, bool GLU>
bool mmvq1_experts_try(const QExpertSlab& s, const int32_t* ids, size_t off_a, size_t off_b, const void* x, int x_div,
                       int n_in, int n_out, int64_t n_pairs, float limit, float* y, cudaStream_t st) {
    constexpr int NM = GLU ? 2 : 1;
    const size_t row_bytes = (size_t) (n_in / F::DIV) * sizeof(typename F::Block);
    // each matrix's rows under half the budget with GLU
    const int R = mmvq::mmvq1_plan(row_bytes * NM, &n_out, 1, false);
    if (!R || ((size_t) R * row_bytes) % 16 || (uintptr_t) s.base % 16 || s.slot_bytes % 16 || off_a % 16 || off_b % 16)
        return false;
    const size_t smem = std::max((size_t) NM * R * row_bytes, (size_t) NM * WARPS * R * WARP * sizeof(float));
    const dim3 grid((unsigned) (n_out / R), (unsigned) n_pairs);
    const auto* base = static_cast<const uint8_t*>(s.base);
    const auto* xq = static_cast<const Q81Block*>(x);
    switch (R) {
        case 16: mmvq1_experts_kernel<F, GLU, 16><<<grid, mmvq::THREADS, smem, st>>>(base, s.slot_of, ids, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
        case 8: mmvq1_experts_kernel<F, GLU, 8><<<grid, mmvq::THREADS, smem, st>>>(base, s.slot_of, ids, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
        case 4: mmvq1_experts_kernel<F, GLU, 4><<<grid, mmvq::THREADS, smem, st>>>(base, s.slot_of, ids, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
        case 2: mmvq1_experts_kernel<F, GLU, 2><<<grid, mmvq::THREADS, smem, st>>>(base, s.slot_of, ids, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
        default: mmvq1_experts_kernel<F, GLU, 1><<<grid, mmvq::THREADS, smem, st>>>(base, s.slot_of, ids, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
    }
    return true;
}

// a batch's hits grouped by expert. The pairs of a call of T tokens that route
// to the same slab expert (at most one per token: a token's experts are distinct) are computed together from one
// staged read of the expert's rows: the block of pair p returns at once unless p is its expert's leader, the first
// pair of the call in pair order with that expert (expert_hit_leader, the host's statement of the rule); the leader's
// block takes the pairs from p on with the same expert as its columns, up to NC, and runs mmvq1_experts_kernel's
// per-row sums for each column in its order (the four warps' partials added in warp order, then the xor sum), so every
// pair's row is the per-pair kernel's bit for bit: a token's values do not depend on the others of the batch.
// NC is the call's T (the most pairs one expert can have).
template <typename F, bool GLU, int R, int NC>
__global__ void __launch_bounds__(mmvq::THREADS) mmvq1_experts_grouped_kernel(const uint8_t* __restrict__ base,
                                                                              const int32_t* __restrict__ slot_of,
                                                                              const int32_t* __restrict__ ids, int n_pairs,
                                                                              size_t slot_bytes, size_t off_a, size_t off_b,
                                                                              const Q81Block* __restrict__ x, int x_div,
                                                                              int n_in, int n_out, float limit,
                                                                              float* __restrict__ y) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int p = (int) blockIdx.y;
    const int32_t ex = ids[p];
    const int32_t slot = slot_of[ex];
    if (slot < 0) return;
    for (int q = 0; q < p; ++q)
        if (ids[q] == ex) return;   // not the leader: its expert's leader computes it
    int cols[NC], nc = 0;
    for (int q = p; q < n_pairs && nc < NC; ++q)
        if (ids[q] == ex) cols[nc++] = q;
    const int bpr = n_in / F::DIV, row_bytes = bpr * (int) sizeof(typename F::Block);
    const int row0 = R * (int) blockIdx.x;
    const uint8_t* exb = base + (size_t) slot * slot_bytes + (size_t) row0 * row_bytes;
    for (int o = (int) threadIdx.x * 16; o < R * row_bytes; o += mmvq::THREADS * 16) {
        mmvq::cp16(smem + o, exb + off_a + o);
        if constexpr (GLU) mmvq::cp16(smem + R * row_bytes + o, exb + off_b + o);
    }
    mmvq::cp16_wait();
    const int tid = (int) threadIdx.x, warp = tid / WARP, lane = tid % WARP;
    float ta[NC][R] = {}, tb[NC][R] = {};
    for (int kbx = tid / F::T; kbx < bpr; kbx += F::BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const auto* wa = reinterpret_cast<const typename F::Block*>(smem + i * row_bytes);
            const typename F::W va = F::load(wa + kbx, kqs);
            typename F::W vb{};
            if constexpr (GLU) vb = F::load(reinterpret_cast<const typename F::Block*>(smem + (R + i) * row_bytes) + kbx, kqs);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                if (c >= nc) break;
                const Q81Block* xc = x + (size_t) (cols[c] / x_div) * (size_t) (n_in / Q8K);
                ta[c][i] += F::apply(va, xc + kby, kqs);
                if constexpr (GLU) tb[c][i] += F::apply(vb, xc + kby, kqs);
            }
        }
    }
    __syncthreads();   // the rows are read: their bytes take the partials
    float* part = reinterpret_cast<float*>(smem);
    constexpr int NM = GLU ? 2 : 1;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (c >= nc) break;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            part[(((c * NM) * WARPS + warp) * R + i) * WARP + lane] = ta[c][i];
            if constexpr (GLU) part[(((c * NM + 1) * WARPS + warp) * R + i) * WARP + lane] = tb[c][i];
        }
    }
    __syncthreads();
    for (int c = 0; c < nc; ++c) {
        const float* pa = part + (size_t) (c * NM) * WARPS * R * WARP;
        const float* pb = part + (size_t) (c * NM + 1) * WARPS * R * WARP;
        for (int i = warp; i < R; i += WARPS) {
            float a = pa[i * WARP + lane], b = 0.f;
            if constexpr (GLU) b = pb[i * WARP + lane];
#pragma unroll
            for (int l = 1; l < WARPS; ++l) {
                a += pa[(l * R + i) * WARP + lane];
                if constexpr (GLU) b += pb[(l * R + i) * WARP + lane];
            }
            a = warp_sum(a);
            if constexpr (GLU) b = warp_sum(b);
            if (lane == 0) y[(size_t) cols[c] * n_out + row0 + i] = GLU ? swiglu_clamp(a, b, limit) : a;
        }
    }
}

// the R of a grouped launch: mmvq1_plan's for the rows (of both matrices with GLU), at most 32 / NC rows so a
// column's sums fit the registers (R changes no value: each row is summed alone)
int grouped_rows(size_t row_bytes, int NM, int n_out, int NC) {
    for (int R : {16, 8, 4, 2, 1})
        if (R * NC <= 32 && (size_t) NM * R * row_bytes <= mmvq::kStageBytes && ((size_t) R * row_bytes) % 16 == 0 && n_out % R == 0)
            return R;
    return 0;
}

template <typename F, bool GLU, int NC>
bool mmvq_experts_grouped_nc(const QExpertSlab& s, const int32_t* ids, size_t off_a, size_t off_b, const void* x,
                             int x_div, int n_in, int n_out, int64_t n_pairs, float limit, float* y, cudaStream_t st) {
    constexpr int NM = GLU ? 2 : 1;
    const size_t row_bytes = (size_t) (n_in / F::DIV) * sizeof(typename F::Block);
    const int R = grouped_rows(row_bytes, NM, n_out, NC);
    if (!R || (uintptr_t) s.base % 16 || s.slot_bytes % 16 || off_a % 16 || off_b % 16) return false;
    const size_t smem = std::max((size_t) NM * R * row_bytes, (size_t) NC * NM * WARPS * R * WARP * sizeof(float));
    const dim3 grid((unsigned) (n_out / R), (unsigned) n_pairs);
    const auto* b = static_cast<const uint8_t*>(s.base);
    const auto* xq = static_cast<const Q81Block*>(x);
    const int np = (int) n_pairs;
    switch (R) {
        case 16: mmvq1_experts_grouped_kernel<F, GLU, 16, NC><<<grid, mmvq::THREADS, smem, st>>>(b, s.slot_of, ids, np, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
        case 8: mmvq1_experts_grouped_kernel<F, GLU, 8, NC><<<grid, mmvq::THREADS, smem, st>>>(b, s.slot_of, ids, np, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
        case 4: mmvq1_experts_grouped_kernel<F, GLU, 4, NC><<<grid, mmvq::THREADS, smem, st>>>(b, s.slot_of, ids, np, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
        case 2: mmvq1_experts_grouped_kernel<F, GLU, 2, NC><<<grid, mmvq::THREADS, smem, st>>>(b, s.slot_of, ids, np, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
        default: mmvq1_experts_grouped_kernel<F, GLU, 1, NC><<<grid, mmvq::THREADS, smem, st>>>(b, s.slot_of, ids, np, s.slot_bytes, off_a, off_b, xq, x_div, n_in, n_out, limit, y); break;
    }
    return true;
}

// a call of T > 1 tokens grouped by expert (false: its rows do not stage, the per-pair launch takes it)
template <typename F, bool GLU>
bool mmvq_experts_grouped_try(const QExpertSlab& s, const int32_t* ids, size_t off_a, size_t off_b, const void* x,
                              int x_div, int n_in, int n_out, int64_t n_pairs, int64_t T, float limit, float* y,
                              cudaStream_t st) {
    switch (T) {
        case 2: return mmvq_experts_grouped_nc<F, GLU, 2>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, limit, y, st);
        case 3: return mmvq_experts_grouped_nc<F, GLU, 3>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, limit, y, st);
        case 4: return mmvq_experts_grouped_nc<F, GLU, 4>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, limit, y, st);
        case 5: return mmvq_experts_grouped_nc<F, GLU, 5>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, limit, y, st);
        case 6: return mmvq_experts_grouped_nc<F, GLU, 6>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, limit, y, st);
        case 7: return mmvq_experts_grouped_nc<F, GLU, 7>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, limit, y, st);
        case 8: return mmvq_experts_grouped_nc<F, GLU, 8>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, limit, y, st);
        default: return false;
    }
}

template <typename F, bool GLU>
void mmvq_experts_launch(const QExpertSlab& s, const int32_t* ids, size_t off_a, size_t off_b, const void* x, int x_div,
                         int n_in, int n_out, int64_t n_pairs, int64_t T, float limit, float* y, cudaStream_t st) {
    if (T > 1 && mmvq_experts_grouped_try<F, GLU>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, T, limit, y, st)) return;
    if (mmvq1_experts_try<F, GLU>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, limit, y, st)) return;
    const dim3 grid((unsigned) ((n_out + ROWS - 1) / ROWS), (unsigned) n_pairs);
    mmvq_experts_kernel<F, GLU><<<grid, dim3(WARP, WARPS), 0, st>>>(static_cast<const uint8_t*>(s.base), s.slot_of, ids,
                                                                    s.slot_bytes, off_a, off_b,
                                                                    static_cast<const Q81Block*>(x), x_div, n_in, n_out,
                                                                    limit, y);
}

template <bool GLU>
void mmvq_experts_type(int type, const QExpertSlab& s, const int32_t* ids, size_t off_a, size_t off_b, const void* x,
                       int x_div, int n_in, int n_out, int64_t n_pairs, int64_t T, float limit, float* y, cudaStream_t st) {
    switch (type) {
        case kQTypeQ2K: mmvq_experts_launch<Q2KTraits, GLU>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, T, limit, y, st); break;
        case kQTypeQ3K: mmvq_experts_launch<Q3KTraits, GLU>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, T, limit, y, st); break;
        default: mmvq_experts_launch<Q6KTraits, GLU>(s, ids, off_a, off_b, x, x_div, n_in, n_out, n_pairs, T, limit, y, st); break;
    }
}

bool quant_type(int type) { return type == kQTypeQ2K || type == kQTypeQ3K || type == kQTypeQ6K; }

// ---------------------------------------------------------------- dequant to f16 (QPrefill::DequantF16)
// the block decoder: qdequant.cuh
using qdequant::dequant8;

// a thread per 8 consecutive outputs, one 16-byte store (out is a row-major [rows][k] f16 matrix, k % 256 == 0)
constexpr int DQ_THREADS = 256;
template <int TYPE>
__global__ void __launch_bounds__(DQ_THREADS) dequant_f16_kernel(const uint8_t* __restrict__ w, int64_t row_bytes,
                                                                  int64_t k, int64_t n8, uint4* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * DQ_THREADS + threadIdx.x;
    if (i >= n8) return;
    const int64_t r = i * 8 / k, c = i * 8 % k;
    constexpr int bb = qdequant::block_bytes<TYPE>();
    float v[8];
    dequant8<TYPE>(w + r * row_bytes + (c / 256) * bb, (int) (c % 256), v);
    uint32_t h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) h[j] = (uint32_t) kernels::f16_from_f32(v[2 * j]) | ((uint32_t) kernels::f16_from_f32(v[2 * j + 1]) << 16);
    out[i] = make_uint4(h[0], h[1], h[2], h[3]);
}

void dequant_launch(const QWeightView& w, int64_t r0, int64_t n, uint16_t* out, cudaStream_t s) {
    const int64_t rb = (int64_t) (w.k / kQK) * (int64_t) qtype_block_bytes(w.type), n8 = n * w.k / 8;
    const uint8_t* src = static_cast<const uint8_t*>(w.data) + r0 * rb;
    const unsigned grid = (unsigned) ((n8 + DQ_THREADS - 1) / DQ_THREADS);
    uint4* o = reinterpret_cast<uint4*>(out);
    switch (w.type) {
        case kQTypeQ2K: dequant_f16_kernel<kQTypeQ2K><<<grid, DQ_THREADS, 0, s>>>(src, rb, w.k, n8, o); break;
        case kQTypeQ3K: dequant_f16_kernel<kQTypeQ3K><<<grid, DQ_THREADS, 0, s>>>(src, rb, w.k, n8, o); break;
        default: dequant_f16_kernel<kQTypeQ6K><<<grid, DQ_THREADS, 0, s>>>(src, rb, w.k, n8, o); break;
    }
    ck(cudaGetLastError(), "dequant_f16_kernel");
}

// T rows of k f32 values at row stride ld into contiguous f16 (round to nearest even, f16_bits.hpp's converter)
__global__ void to_f16_rows_kernel(const float* __restrict__ x, int64_t ld, int64_t k, uint16_t* __restrict__ y) {
    const int64_t t = blockIdx.y;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < k; i += (int64_t) gridDim.x * blockDim.x)
        y[t * k + i] = kernels::f16_from_f32(x[t * ld + i]);
}

// the cuBLAS workspace the context owns (a fixed one, so the handle never allocates; Strata's Gemm::init)
constexpr size_t kBlasWorkspace = 32u << 20;

}  // namespace

const char* qpath_name(QPath p) {
    switch (p) {
        case QPath::Mmvq: return "mmvq";
        case QPath::DequantF16: return "dequant+cublas";
        case QPath::Mmq: return "mmq";
    }
    return "?";
}

QPath qmatmul_path(QPrefill p, int64_t T) {
    if (T <= kMmvqMaxCols) return QPath::Mmvq;
    return p == QPrefill::Mmq ? QPath::Mmq : QPath::DequantF16;
}

static_assert(qtype_block_bytes(kQTypeQ2K) == sizeof(mmvq::Q2KBlock) && qtype_block_bytes(kQTypeQ3K) == sizeof(mmvq::Q3KBlock) &&
              qtype_block_bytes(kQTypeQ6K) == sizeof(mmvq::Q6KBlock), "qmatmul.hpp's block bytes are mmvq.cuh's blocks");

size_t qweight_bytes(const QWeightView& w) {
    if (w.type == kQTypeF16) return (size_t) (w.k * w.rows) * 2;
    return (size_t) w.rows * (size_t) (w.k / kQK) * qtype_block_bytes(w.type);
}

QWeightView qweight_rows(const QWeightView& w, int64_t r0, int64_t n) {
    if (r0 < 0 || n < 0 || r0 + n > w.rows) failf("rows %lld..%lld of a %lld-row matrix", (long long) r0, (long long) (r0 + n), (long long) w.rows);
    QWeightView v = w;
    v.rows = n;
    const size_t row_bytes = w.type == kQTypeF16 ? (size_t) w.k * 2 : (size_t) (w.k / kQK) * qtype_block_bytes(w.type);
    v.data = w.data ? static_cast<const uint8_t*>(w.data) + (size_t) r0 * row_bytes : nullptr;
    return v;
}

const char* qact_layout_name(QActLayout l) {
    switch (l) {
        case QActLayout::Q8_1: return "q8_1";
        case QActLayout::MmqD4: return "mmq-d4";
        case QActLayout::MmqD2S6: return "mmq-d2s6";
    }
    return "?";
}

QActLayout qact_layout(int type, int64_t T) {
    if (!quant_type(type)) failf("no activation layout for weight type %d (%s)", type, qtype_name(type));
    if (T <= kMmvqMaxCols) return QActLayout::Q8_1;
    return type == kQTypeQ2K ? QActLayout::MmqD2S6 : QActLayout::MmqD4;
}

size_t qact_bytes(QActLayout l, int64_t T, int64_t k) {
    if (l == QActLayout::Q8_1) return (size_t) T * (size_t) (k / Q8K) * sizeof(Q81Block);
    return ((size_t) T * (size_t) (pad512(k) / MMQ_TILE) + kMmqTailBlocks) * sizeof(MmqTile);
}

size_t qact_bytes_max(QPrefill p, int64_t max_T, int64_t k) {
    const size_t a = qact_bytes(QActLayout::Q8_1, max_T < kMmvqMaxCols ? max_T : kMmvqMaxCols, k);
    // D2S6 tiles are D4's size
    const size_t b = p == QPrefill::Mmq && max_T > kMmvqMaxCols ? qact_bytes(QActLayout::MmqD4, max_T, k) : 0;
    return a > b ? a : b;
}

QmmContext::~QmmContext() {
    if (mmq_) mmq::context_delete(mmq_);
    if (blas_) cublasDestroy((cublasHandle_t) blas_);
    cudaFree(blas_ws_);
    cudaFree(scratch_);
}

void* QmmContext::blas_for(void* stream) {
    if (!blas_) failf("the QmmContext has no dequant path (it was initialized for MMQ)");
    // cublasSetStream resets the handle to cuBLAS's own workspace pool: rebind the fixed one with every change
    if (!blas_bound_ || stream != blas_stream_) {
        ckb(cublasSetStream((cublasHandle_t) blas_, (cudaStream_t) stream), "cublasSetStream");
        ckb(cublasSetWorkspace((cublasHandle_t) blas_, blas_ws_, kBlasWorkspace), "cublasSetWorkspace");
        blas_stream_ = stream;
        blas_bound_ = true;
    }
    return blas_;
}

bool QmmContext::init(std::string& err, QPrefill prefill, int64_t f16_scratch_elems) {
    if (mmq_ || blas_) failf("QmmContext::init called twice");
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        err = "no CUDA device";
        return false;
    }
    prefill_ = prefill;
    if (prefill == QPrefill::DequantF16) {
        if (f16_scratch_elems < kQK) failf("a dequant scratch of %lld values (at least one row of %lld)", (long long) f16_scratch_elems, (long long) kQK);
        const size_t bytes = (size_t) f16_scratch_elems * 2;
        if (cudaMalloc(&scratch_, bytes) != cudaSuccess || cudaMalloc(&blas_ws_, kBlasWorkspace) != cudaSuccess) {
            err = "the dequant scratch (" + std::to_string(bytes >> 20) + " MiB) or the cuBLAS workspace does not fit";
            return false;
        }
        scratch_elems_ = f16_scratch_elems;
        device_bytes_ = bytes + kBlasWorkspace;
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            err = "cublasCreate failed";
            return false;
        }
        blas_ = h;
        return true;
    }
    // the tile layouts this file writes must be the ones the fork's MMQ reads
    if (mmq::tile_bytes() != (int) sizeof(MmqTile) || mmq::ds_layout(kQTypeQ2K) != 2 || mmq::ds_layout(kQTypeQ3K) != 0 ||
        mmq::ds_layout(kQTypeQ6K) != 0)
        failf("the fork's MMQ tile layout changed (block_q8_1_mmq %d bytes, layouts q2_K %d q3_K %d q6_K %d)", mmq::tile_bytes(),
              mmq::ds_layout(kQTypeQ2K), mmq::ds_layout(kQTypeQ3K), mmq::ds_layout(kQTypeQ6K));
    for (int t : {kQTypeQ2K, kQTypeQ3K, kQTypeQ6K})
        for (int64_t rows : {128, 24})
            if (!mmq::fits(t, rows)) {
                err = std::string("the fork's MMQ has no tile for ") + qtype_name(t) + " on this device";
                return false;
            }
    mmq_ = mmq::context_new();
    if (!mmq_) {
        err = "no CUDA device for the MMQ context";
        return false;
    }
    return true;
}

void qact_quantize(const float* x, const uint16_t* xh, int64_t ld, int64_t T, int64_t k, QActLayout l, void* dst,
                   void* stream, QActs& out) {
    if ((x == nullptr) == (xh == nullptr)) failf("qact_quantize takes f32 or f16 input, exactly one");
    if (k <= 0 || k % kQK) failf("rows of %lld values (not whole blocks of 256)", (long long) k);
    if (T < 1 || T > 65535) failf("%lld rows (1..65535)", (long long) T);
    if (ld < k) failf("row stride %lld below the row length %lld", (long long) ld, (long long) k);
    if (l == QActLayout::Q8_1 && T > kMmvqMaxCols) failf("q8_1 activations for %lld columns: MMVQ takes up to %lld", (long long) T, (long long) kMmvqMaxCols);
    const cudaStream_t s = (cudaStream_t) stream;
    if (l == QActLayout::Q8_1) {
        const dim3 grid((unsigned) (k / QUANT_THREADS), (unsigned) T);
        if (x) quantize_q8_1_kernel<float><<<grid, QUANT_THREADS, 0, s>>>(x, ld, k, static_cast<Q81Block*>(dst));
        else quantize_q8_1_kernel<uint16_t><<<grid, QUANT_THREADS, 0, s>>>(xh, ld, k, static_cast<Q81Block*>(dst));
    } else {
        const int64_t k_pad = pad512(k);
        const dim3 grid((unsigned) T, (unsigned) (k_pad / (4 * MMQ_QUANT_THREADS)));
        MmqTile* y = static_cast<MmqTile*>(dst);
        if (l == QActLayout::MmqD2S6) {
            if (x) quantize_mmq_kernel<float, true><<<grid, MMQ_QUANT_THREADS, 0, s>>>(x, ld, k, k_pad, T, y);
            else quantize_mmq_kernel<uint16_t, true><<<grid, MMQ_QUANT_THREADS, 0, s>>>(xh, ld, k, k_pad, T, y);
        } else {
            if (x) quantize_mmq_kernel<float, false><<<grid, MMQ_QUANT_THREADS, 0, s>>>(x, ld, k, k_pad, T, y);
            else quantize_mmq_kernel<uint16_t, false><<<grid, MMQ_QUANT_THREADS, 0, s>>>(xh, ld, k, k_pad, T, y);
        }
    }
    ck(cudaGetLastError(), "qact_quantize");
    out = QActs{l, T, k, dst};
}

void qmatmul_check(const QWeightView& w, int groups, const char* what) {
    if (!w.data) failf("%s: the weight is missing", what);
    if (!quant_type(w.type)) failf("%s: weight type %d (%s): qmatmul reads q2_K, q3_K and q6_K", what, w.type, qtype_name(w.type));
    if (w.k <= 0 || w.k % kQK || w.rows <= 0) failf("%s: a %lld x %lld matrix (rows of whole 256-value blocks)", what, (long long) w.rows, (long long) w.k);
    if (groups < 1 || w.rows % groups) failf("%s: %lld rows in %d groups", what, (long long) w.rows, groups);
    if (w.rows / groups > (1 << 30) || w.k * groups > (1 << 30)) failf("%s: a %lld x %lld matrix is past the kernels' int range", what, (long long) w.rows, (long long) w.k);
    if (w.type == kQTypeQ2K && (uintptr_t) w.data % 4) failf("%s: q2_K data at a %p: the kernels read it in 4-byte words", what, w.data);
    if (w.type == kQTypeQ3K && (uintptr_t) w.data % 2) failf("%s: q3_K data at a %p: the dequant reads it in 2-byte words", what, w.data);
}

void qmatmul_q(QmmContext& c, const QWeightView& w, const QActs& a, float* y, int64_t ldy, int groups, void* stream,
               const char* what) {
    qmatmul_check(w, groups, what);
    if (a.k != (int64_t) groups * w.k) failf("%s: activations of %lld values, the matrix takes %d x %lld", what, (long long) a.k, groups, (long long) w.k);
    if (c.path(a.T) == QPath::DequantF16)
        failf("%s: %lld columns take the dequant path in this context (qmatmul_f16), not int8 activations", what, (long long) a.T);
    if (a.layout != qact_layout(w.type, a.T))
        failf("%s: %s activations for %s at %lld columns, which reads %s", what, qact_layout_name(a.layout), qtype_name(w.type),
              (long long) a.T, qact_layout_name(qact_layout(w.type, a.T)));
    if (ldy < w.rows) failf("%s: output stride %lld below its %lld columns", what, (long long) ldy, (long long) w.rows);
    const int64_t rows_g = w.rows / groups;
    const cudaStream_t s = (cudaStream_t) stream;
    if (a.layout == QActLayout::Q8_1) {
        const int64_t x_stride = a.k / Q8K;
        const int n_in = (int) w.k, n_out = (int) rows_g, nc = (int) a.T;
        bool one = false;
        if (nc == 1) {
            switch (w.type) {
                case kQTypeQ2K: one = mmvq1_try<Q2KTraits>(w.data, a.data, y, n_in, n_out, groups, s); break;
                case kQTypeQ3K: one = mmvq1_try<Q3KTraits>(w.data, a.data, y, n_in, n_out, groups, s); break;
                default: one = mmvq1_try<Q6KTraits>(w.data, a.data, y, n_in, n_out, groups, s); break;
            }
        }
        if (!one) {
            switch (w.type) {
                case kQTypeQ2K: mmvq_launch<Q2KTraits>(w.data, a.data, y, n_in, n_out, groups, nc, x_stride, ldy, s); break;
                case kQTypeQ3K: mmvq_launch<Q3KTraits>(w.data, a.data, y, n_in, n_out, groups, nc, x_stride, ldy, s); break;
                default: mmvq_launch<Q6KTraits>(w.data, a.data, y, n_in, n_out, groups, nc, x_stride, ldy, s); break;
            }
        }
    } else {
        if (!c.mmq()) failf("%s: the QmmContext is not initialized", what);
        mmq::run(c.mmq(), w.type, w.data, w.k, rows_g, groups, a.data, a.T, y, ldy, s);
    }
    ck(cudaGetLastError(), what);
}

void qweight_dequant_f16(const QWeightView& w, int64_t r0, int64_t n, uint16_t* out, void* stream) {
    qmatmul_check(w, 1, "qweight_dequant_f16");
    if (r0 < 0 || n < 1 || r0 + n > w.rows) failf("qweight_dequant_f16: rows %lld..%lld of %lld", (long long) r0, (long long) (r0 + n), (long long) w.rows);
    if ((uintptr_t) out % 16) failf("qweight_dequant_f16: output at %p (the kernel stores 16 bytes at a time)", (void*) out);
    dequant_launch(w, r0, n, out, (cudaStream_t) stream);
}

void qmatmul_f16(QmmContext& c, const QWeightView& w, const uint16_t* xh, int64_t ldx, int64_t T, float* y, int64_t ldy,
                 int groups, void* stream, const char* what) {
    qmatmul_check(w, groups, what);
    if (!xh || T < 1 || T > (1 << 30)) failf("%s: f16 activations of %lld rows", what, (long long) T);
    if (ldx < (int64_t) groups * w.k) failf("%s: activation stride %lld below its %d x %lld values", what, (long long) ldx, groups, (long long) w.k);
    if (ldy < w.rows) failf("%s: output stride %lld below its %lld columns", what, (long long) ldy, (long long) w.rows);
    if (!c.scratch()) failf("%s: the QmmContext has no dequant scratch (it was initialized for MMQ)", what);
    const cudaStream_t s = (cudaStream_t) stream;
    const cublasHandle_t h = (cublasHandle_t) c.blas_for(stream);
    const int64_t rows_g = w.rows / groups, K = w.k;
    const float one = 1.0f, zero = 0.0f;
    if (w.rows * K <= c.scratch_elems()) {
        dequant_launch(w, 0, w.rows, c.scratch(), s);
        if (groups == 1)
            ckb(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) w.rows, (int) T, (int) K, &one, c.scratch(), CUDA_R_16F, (int) K,
                             xh, CUDA_R_16F, (int) ldx, &zero, y, CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                what);
        else
            ckb(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) rows_g, (int) T, (int) K, &one, c.scratch(), CUDA_R_16F,
                                           (int) K, (long long) rows_g * K, xh, CUDA_R_16F, (int) ldx, (long long) K, &zero, y,
                                           CUDA_R_32F, (int) ldy, (long long) rows_g, groups, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                what);
        return;
    }
    // row slices of each group, as Gemm::native's "too large for the scratch at once" (the slices reuse the scratch
    // in stream order)
    const int64_t slice = c.scratch_elems() / K;
    if (slice < 1) failf("%s: a dequant scratch of %lld values holds no row of %lld", what, (long long) c.scratch_elems(), (long long) K);
    for (int g = 0; g < groups; ++g)
        for (int64_t r0 = 0; r0 < rows_g; r0 += slice) {
            const int64_t n = std::min(slice, rows_g - r0);
            dequant_launch(w, (int64_t) g * rows_g + r0, n, c.scratch(), s);
            ckb(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) n, (int) T, (int) K, &one, c.scratch(), CUDA_R_16F, (int) K,
                             xh + (int64_t) g * K, CUDA_R_16F, (int) ldx, &zero, y + (int64_t) g * rows_g + r0, CUDA_R_32F, (int) ldy,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                what);
        }
}

size_t qmatmul_scratch_bytes(QPath p, int type, int64_t T, int64_t k) {
    switch (p) {
        case QPath::Mmvq: return qact_bytes(QActLayout::Q8_1, T, k);
        case QPath::Mmq: return qact_bytes(qact_layout(type, T), T, k);
        case QPath::DequantF16: return (size_t) (T * k) * 2;
    }
    return 0;
}

void qmatmul(QmmContext& c, const QWeightView& w, const float* x, int64_t ldx, int64_t T, float* y, int64_t ldy,
             void* scratch, int groups, void* stream, const char* what) {
    const int64_t K = (int64_t) groups * w.k;
    if (c.path(T) == QPath::DequantF16) {
        if (T > 65535 || ldx < K) failf("%s: %lld rows of %lld values at stride %lld", what, (long long) T, (long long) K, (long long) ldx);
        const dim3 grid((unsigned) std::min<int64_t>((K + 255) / 256, 64), (unsigned) T);
        to_f16_rows_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(x, ldx, K, static_cast<uint16_t*>(scratch));
        ck(cudaGetLastError(), "to_f16_rows_kernel");
        qmatmul_f16(c, w, static_cast<const uint16_t*>(scratch), K, T, y, ldy, groups, stream, what);
        return;
    }
    QActs a;
    qact_quantize(x, nullptr, ldx, T, K, qact_layout(w.type, T), scratch, stream, a);
    qmatmul_q(c, w, a, y, ldy, groups, stream, what);
}

void qmatmul_experts_check(const QExpertSlab& s, const char* what) {
    if (!quant_type(s.type[0]) || !quant_type(s.type[2]))
        failf("%s: expert types up %s, gate %s, down %s: the grouped kernels read q2_K, q3_K and q6_K", what,
              qtype_name(s.type[0]), qtype_name(s.type[1]), qtype_name(s.type[2]));
    if (s.type[1] != s.type[0])
        failf("%s: gate %s and up %s: the fused gate/up kernel reads both with one decoder", what, qtype_name(s.type[1]),
              qtype_name(s.type[0]));
    if (s.n_embd <= 0 || s.n_embd % kQK || s.n_ff <= 0 || s.n_ff % kQK || s.n_embd > (1 << 30) || s.n_ff > (1 << 30))
        failf("%s: experts of %lld x %lld (whole 256-value blocks)", what, (long long) s.n_embd, (long long) s.n_ff);
    // every slot's parts at the kernels' word alignment (the slab aligns each part to 256 bytes)
    if (s.slot_bytes % 4 || s.part_offset[0] % 4 || s.part_offset[1] % 4 || s.part_offset[2] % 4)
        failf("%s: slot %zu bytes, parts at %zu, %zu, %zu (4-byte aligned)", what, s.slot_bytes, s.part_offset[0],
              s.part_offset[1], s.part_offset[2]);
}

int64_t expert_hit_leader(const int32_t* ids, int64_t P, int64_t p, const int32_t* slot_of) {
    if (p < 0 || p >= P) failf("expert_hit_leader: pair %lld of %lld", (long long) p, (long long) P);
    if (slot_of[ids[p]] < 0) return -1;
    int64_t q = 0;
    while (ids[q] != ids[p]) ++q;
    return q;
}

size_t qmatmul_experts_scratch_bytes(const QExpertSlab& s, int64_t T, int64_t n_pairs) {
    const size_t a = (qact_bytes(QActLayout::Q8_1, T, s.n_embd) + 255) / 256 * 256;
    return a + qact_bytes(QActLayout::Q8_1, n_pairs, s.n_ff);
}

void qmatmul_experts(const QExpertSlab& s, const float* x, int64_t T, const int32_t* ids, int n_used, float limit,
                     float* h, void* scratch, float* y, void* stream, const void* xq) {
    qmatmul_experts_check(s, "qmatmul_experts");
    const int64_t P = T * n_used;
    if (T < 1 || T > kMmvqMaxCols || n_used < 1 || P > 65535 || !s.base || !s.slot_of)
        failf("qmatmul_experts: %lld tokens of %d pairs (1..%lld tokens), slab %p, slots %p", (long long) T, n_used,
              (long long) kMmvqMaxCols, s.base, (const void*) s.slot_of);
    const cudaStream_t st = (cudaStream_t) stream;
    const int E = (int) s.n_embd, F = (int) s.n_ff;
    uint8_t* xs = static_cast<uint8_t*>(scratch);
    uint8_t* hq = xs + (qact_bytes(QActLayout::Q8_1, T, s.n_embd) + 255) / 256 * 256;
    // x's rows once for every pair of their token, h's rows for down (the q8_1 quantizer of qact_quantize, whose
    // MMVQ limit of 8 columns the pairs pass: each pair is a column of its own here)
    if (!xq) {
        quantize_q8_1_kernel<float><<<dim3((unsigned) (E / QUANT_THREADS), (unsigned) T), QUANT_THREADS, 0, st>>>(
            x, E, E, reinterpret_cast<Q81Block*>(xs));
        ck(cudaGetLastError(), "quantize_q8_1_kernel (expert inputs)");
        xq = xs;
    }
    mmvq_experts_type<true>(s.type[1], s, ids, s.part_offset[1], s.part_offset[0], xq, n_used, E, F, P, T, limit, h, st);
    ck(cudaGetLastError(), "mmvq_experts_kernel (gate, up)");
    quantize_q8_1_kernel<float><<<dim3((unsigned) (F / QUANT_THREADS), (unsigned) P), QUANT_THREADS, 0, st>>>(
        h, F, F, reinterpret_cast<Q81Block*>(hq));
    ck(cudaGetLastError(), "quantize_q8_1_kernel (expert activations)");
    mmvq_experts_type<false>(s.type[2], s, ids, s.part_offset[2], s.part_offset[2], hq, 1, F, E, P, T, 0.f, y, st);
    ck(cudaGetLastError(), "mmvq_experts_kernel (down)");
}

}  // namespace strata::dsv41::engine
