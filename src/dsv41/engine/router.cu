// src/dsv41/engine/router.cu - the router of include/strata/dsv41/engine/dense.hpp (SPEC.md section 6).
//
//   input    to_bf16_kernel: ffn_norm rounded to bf16 (round to nearest even, GGML_FP32_TO_BF16), as the fork feeds
//            its bf16 ffn_gate_inp (the matmul's vec_dot_type), so the products are exact and only the sums differ;
//   logits   up to kRouterGemvMaxCols tokens router_gemv_kernel (a warp per expert row, 16-byte loads of 8 bf16
//            values, one f32 accumulator per token, then the warp's butterfly); above, cublasGemmEx bf16 x bf16 ->
//            f32 with f32 compute (the decode case is a 3.9 MB weight read, which a GEMV streams at bandwidth;
//            dsv41_engine_dense_bench measures both);
//   select   router_select_kernel: native_router.cu's warp-per-token top-k (values spread over the lanes, K rounds of
//            a butterfly argmax, ties to the lower expert), with sqrt(softplus) in place of the softmax, the selection
//            on probs + exp_probs_b, the unbiased probs as weights, their sum in double in rank order, the clamp at
//            6.1035156e-5, the division and the scale (llama-graph.cpp 2056-2187), for any expert count up to 512;
//            with forced ids (the parity mode, selection.hpp) the ranks are those ids and the rest is the same.
#include "dense_internal.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <climits>

namespace strata::dsv41::engine {

using namespace dense_detail;

namespace {

__device__ __forceinline__ float bf16_to_f32(uint32_t h) { return __uint_as_float(h << 16); }

__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const uint32_t u = __float_as_uint(x[i]);
        y[i] = (u & 0x7fffffffu) > 0x7f800000u ? (uint16_t) ((u >> 16) | 64) : (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
    }
}

constexpr int GEMV_WARPS = 8;

template <int NC>
__global__ void __launch_bounds__(32 * GEMV_WARPS) router_gemv_kernel(const uint16_t* __restrict__ w, const uint16_t* __restrict__ x,
                                                                      int E, int NE, float* __restrict__ y) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);
    if (row >= NE) return;
    const uint4* wr = reinterpret_cast<const uint4*>(w + (int64_t) row * E);
    float acc[NC];
    for (int c = 0; c < NC; ++c) acc[c] = 0.f;
    for (int i = lane; i < E / 8; i += 32) {
        const uint4 wv = wr[i];
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
        for (int c = 0; c < NC; ++c) {
            const uint4 xv = reinterpret_cast<const uint4*>(x + (int64_t) c * E)[i];
            const uint32_t xx[4] = {xv.x, xv.y, xv.z, xv.w};
            for (int j = 0; j < 4; ++j) {
                acc[c] = __fmaf_rn(bf16_to_f32(ww[j] & 0xffffu), bf16_to_f32(xx[j] & 0xffffu), acc[c]);
                acc[c] = __fmaf_rn(bf16_to_f32(ww[j] >> 16), bf16_to_f32(xx[j] >> 16), acc[c]);
            }
        }
    }
    for (int c = 0; c < NC; ++c) {
        float v = acc[c];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[(int64_t) c * NE + row] = v;
    }
}

void gemv_launch(int nc, const uint16_t* w, const uint16_t* x, int E, int NE, float* y, cudaStream_t s) {
    const unsigned blocks = (unsigned) ((NE + GEMV_WARPS - 1) / GEMV_WARPS);
    switch (nc) {
        case 1: router_gemv_kernel<1><<<blocks, 32 * GEMV_WARPS, 0, s>>>(w, x, E, NE, y); break;
        case 2: router_gemv_kernel<2><<<blocks, 32 * GEMV_WARPS, 0, s>>>(w, x, E, NE, y); break;
        case 3: router_gemv_kernel<3><<<blocks, 32 * GEMV_WARPS, 0, s>>>(w, x, E, NE, y); break;
        case 4: router_gemv_kernel<4><<<blocks, 32 * GEMV_WARPS, 0, s>>>(w, x, E, NE, y); break;
        case 5: router_gemv_kernel<5><<<blocks, 32 * GEMV_WARPS, 0, s>>>(w, x, E, NE, y); break;
        case 6: router_gemv_kernel<6><<<blocks, 32 * GEMV_WARPS, 0, s>>>(w, x, E, NE, y); break;
        case 7: router_gemv_kernel<7><<<blocks, 32 * GEMV_WARPS, 0, s>>>(w, x, E, NE, y); break;
        case 8: router_gemv_kernel<8><<<blocks, 32 * GEMV_WARPS, 0, s>>>(w, x, E, NE, y); break;
        default: failf("router GEMV takes 1..%lld tokens, not %d", (long long) kRouterGemvMaxCols, nc);
    }
    ck(cudaGetLastError(), "router_gemv_kernel");
}

constexpr int SEL_WARPS = 8;
constexpr int SEL_PER_LANE = 16;   // up to 512 experts

__global__ void __launch_bounds__(32 * SEL_WARPS) router_select_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                                       int NE, int K, int64_t T, bool norm, float scale,
                                                                       float* probs, const int32_t* __restrict__ forced,
                                                                       int32_t* __restrict__ ids, float* __restrict__ weights) {
    const int lane = threadIdx.x & 31;
    const int64_t t = (int64_t) blockIdx.x * SEL_WARPS + (threadIdx.x >> 5);
    if (t >= T) return;
    const float* lt = logits + t * NE;
    float* pt = probs + t * NE;
    float b[SEL_PER_LANE];
    for (int i = 0; i < SEL_PER_LANE; ++i) {
        const int idx = lane + 32 * i;
        b[i] = -INFINITY;
        if (idx < NE) {
            const float l = lt[idx];
            const float sp = l > 20.0f ? l : logf(__fadd_rn(1.0f, expf(l)));
            const float p = sqrtf(sp);
            pt[idx] = p;
            b[i] = __fadd_rn(p, bias[idx]);
        }
    }
    __syncwarp();
    double wsum = 0.0;
    float mine = 0.f;
    for (int rank = 0; rank < K; ++rank) {
        int bid = INT_MAX;
        if (forced) {
            bid = forced[t * K + rank];
        } else {
            float best = -INFINITY;
            for (int i = 0; i < SEL_PER_LANE; ++i) {
                const int idx = lane + 32 * i;
                if (idx < NE && (b[i] > best || (b[i] == best && idx < bid))) { best = b[i]; bid = idx; }
            }
            for (int o = 16; o > 0; o >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffffu, best, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bid, o);
                if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; }
            }
            if ((bid & 31) == lane)
                for (int i = 0; i < SEL_PER_LANE; ++i)
                    if (lane + 32 * i == bid) b[i] = -INFINITY;
        }
        const float p = pt[bid];   // written by its lane before the __syncwarp above
        if (lane == 0) ids[t * K + rank] = bid;
        wsum += (double) p;
        if (lane == rank) mine = p;
    }
    if (lane < K) {
        float v = mine;
        if (norm) v = __fdiv_rn(v, fmaxf((float) wsum, 6.103515625e-5f));
        if (scale != 0.0f && scale != 1.0f) v = __fmul_rn(v, scale);
        weights[t * K + lane] = v;
    }
}

}  // namespace

void router_check_weights(const DenseDims& d, const RouterWeights& w, const char* what) {
    if (!w.gate_inp || !w.bias) failf("%s: ffn_gate_inp or exp_probs_b is missing", what);
    if ((uintptr_t) w.gate_inp % 16) failf("%s: ffn_gate_inp at %p (the GEMV reads 16 bytes at a time)", what, (const void*) w.gate_inp);
    if (d.n_embd % 8 || d.n_expert < d.n_expert_used || d.n_expert > 32 * SEL_PER_LANE)
        failf("%s: %d experts of %d values (%d used): the selection takes up to %d", what, d.n_expert, d.n_embd, d.n_expert_used, 32 * SEL_PER_LANE);
}

void dense_router(DenseEngine& e, const RouterWeights& w, const float* xn, int64_t T, int32_t* ids, float* weights,
                  RouterMatmul mm, const int32_t* forced) {
    OpTimer tm(e, &e.times_mut().router_ms);
    const DenseDims& d = e.dims();
    router_check_weights(d, w, "dense_router");
    if (T < 1 || T > e.max_T()) failf("dense_router: %lld tokens (1..%lld)", (long long) T, (long long) e.max_T());
    DenseScratch& s = e.scratch_mut();
    const cudaStream_t st = stream_of(e);
    const int64_t n = T * d.n_embd;
    to_bf16_kernel<<<(unsigned) std::min<int64_t>((n + 255) / 256, 4096), 256, 0, st>>>(xn, s.router_in, n);
    ck(cudaGetLastError(), "to_bf16_kernel");
    const bool gemv = mm == RouterMatmul::Gemv || (mm == RouterMatmul::Auto && T <= kRouterGemvMaxCols);
    if (gemv) {
        gemv_launch((int) T, w.gate_inp, s.router_in, d.n_embd, d.n_expert, s.logits, st);
    } else {
        const float one = 1.f, zero = 0.f;
        const cublasStatus_t r = cublasGemmEx((cublasHandle_t) e.blas(), CUBLAS_OP_T, CUBLAS_OP_N, d.n_expert, (int) T, d.n_embd, &one,
                                              w.gate_inp, CUDA_R_16BF, d.n_embd, s.router_in, CUDA_R_16BF, d.n_embd, &zero, s.logits,
                                              CUDA_R_32F, d.n_expert, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (r != CUBLAS_STATUS_SUCCESS) failf("ffn_gate_inp: cublasGemmEx status %d", (int) r);
    }
    router_select_kernel<<<(unsigned) ((T + SEL_WARPS - 1) / SEL_WARPS), 32 * SEL_WARPS, 0, st>>>(
        s.logits, w.bias, d.n_expert, d.n_expert_used, T, d.weights_norm, d.weights_scale, s.probs, forced, ids, weights);
    ck(cudaGetLastError(), "router_select_kernel");
}

}  // namespace strata::dsv41::engine
