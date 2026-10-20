// src/dsv41/engine/dense_internal.cuh - helpers shared by the .cu files of the dense ops (dense.cu, hc.cu, engram.cu,
// router.cu, dense_ffn.cu); internal to src/dsv41/engine.
#pragma once

#include "strata/dsv41/engine/dense.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace strata::dsv41::engine::dense_detail {

[[noreturn]] void failf(const char* fmt, ...);
void ck(cudaError_t e, const char* what);
inline cudaStream_t stream_of(const DenseEngine& e) { return (cudaStream_t) e.stream(); }

/// T rows that qmatmul reads as f32 (MMVQ: T <= kMmvqMaxCols) or f16 (the dequant and MMQ paths): the norm and the
/// engram's dequant kernels write the one the call's path takes.
inline bool f16_input(int64_t T) { return T > kMmvqMaxCols; }

/// y [T][ldy] = w . x for x [T][k] given as f32 xf (f16_input(T) false) or f16 xh (true), on the engine's qmatmul
/// path at T (MMQ quantizes the f16 rows into the activation scratch).
void project(DenseEngine& e, const QWeightView& w, const float* xf, const uint16_t* xh, int64_t T, float* y, int64_t ldy,
             const char* what);

/// hc_decode_step's shared memory limit at n_embd, set by DenseEngine::init (outside any stream capture: the decode
/// token's first launch may be inside one).
void hc_step_attrs(int n_embd);

/// CUDA event timing of one op family: start() then stop(ms) adds nothing when the engine's timing is off.
struct OpTimer {
    DenseEngine& e;
    float* ms;
    explicit OpTimer(DenseEngine& eng, float* out) : e(eng), ms(out) {
        if (e.timing()) ck(cudaEventRecord((cudaEvent_t) e.event(0), stream_of(e)), "cudaEventRecord");
    }
    ~OpTimer() {
        if (!e.timing()) return;
        ck(cudaEventRecord((cudaEvent_t) e.event(1), stream_of(e)), "cudaEventRecord");
        ck(cudaEventSynchronize((cudaEvent_t) e.event(1)), "cudaEventSynchronize");
        ck(cudaEventElapsedTime(ms, (cudaEvent_t) e.event(0), (cudaEvent_t) e.event(1)), "cudaEventElapsedTime");
    }
};

/// Element v (0..255) of one q2_K block as dequantize_row_q2_K computes it, d sc q - dmin m without contraction
/// (ggml-quants.c rounds the product, then subtracts).
__device__ __forceinline__ float q2k_value(const uint8_t* __restrict__ b, int v) {
    const int n = v >> 7, r = v & 127, j = r >> 5, h = (r >> 4) & 1, l = r & 15;
    const uint8_t sc = b[8 * n + 2 * j + h];
    const uint8_t q = (uint8_t) ((b[16 + 32 * n + 16 * h + l] >> (2 * j)) & 3);
    uint16_t dh, mh;
    dh = (uint16_t) (b[80] | (b[81] << 8));
    mh = (uint16_t) (b[82] | (b[83] << 8));
    const float d = kernels::f32_from_f16(dh), dmin = kernels::f32_from_f16(mh);
    const float dl = __fmul_rn(d, (float) (sc & 0xF)), ml = __fmul_rn(dmin, (float) (sc >> 4));
    return __fsub_rn(__fmul_rn(dl, (float) q), ml);
}

/// The sum of a block's threads (blockDim.x a multiple of 32, at most 1024), in double; every thread gets it.
__device__ __forceinline__ double block_sum_d(double v, double* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = lane < nw ? sh[lane] : 0.0;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

}  // namespace strata::dsv41::engine::dense_detail
