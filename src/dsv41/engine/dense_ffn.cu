// src/dsv41/engine/dense_ffn.cu - the shared expert of include/strata/dsv41/engine/dense.hpp (SPEC.md section 7):
// Strata's shared_expert.cu sequence (gate and up on the normed input, the activation, down) on qmatmul, without its
// sigmoid expert gate (DeepSeek-V4.1 has none) and with the fork's SwiGLU clamp (ffn_device.cuh). The routed experts
// take the same function in the epilogues of their decode hits (qmatmul_experts) and prefill passes (experts_gemm.cu).
#include "dense_internal.cuh"
#include "ffn_device.cuh"
#include "ffn_kernels.hpp"
#include "mmvq.cuh"

#include <algorithm>

namespace strata::dsv41::engine {

using namespace dense_detail;

namespace {

__global__ void swiglu_clamp_kernel(const float* __restrict__ gate, const float* __restrict__ up, int64_t n, float limit,
                                    float* __restrict__ act) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        act[i] = swiglu_clamp(gate[i], up[i], limit);
}

// a decode token: act = swiglu_clamp(gate, up) and its q8_1 (qact_quantize's), a warp per 32 values
__global__ void swiglu_q8_kernel(const float* __restrict__ gate, const float* __restrict__ up, int n, float limit,
                                 float* __restrict__ act, mmvq::Q81Block* __restrict__ q) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;   // n % 256 == 0: whole warps return
    const float a = swiglu_clamp(gate[i], up[i], limit);
    act[i] = a;
    mmvq::q8_1_warp(a, q[i / mmvq::Q8K], (int) (threadIdx.x & 31));
}

}  // namespace

void ffn_swiglu_clamp(const float* gate, const float* up, int64_t n, float limit, float* act, void* stream) {
    if (n < 1) return;
    swiglu_clamp_kernel<<<(unsigned) std::min<int64_t>((n + 255) / 256, 4096), 256, 0, (cudaStream_t) stream>>>(gate, up, n,
                                                                                                         limit, act);
    ck(cudaGetLastError(), "swiglu_clamp_kernel");
}

void dense_shared_expert(DenseEngine& e, const SharedExpertWeights& w, const float* xn, int64_t T, float* out,
                         const void* xq) {
    OpTimer tm(e, &e.times_mut().shared_ms);
    const DenseDims& d = e.dims();
    shared_check_weights(d, w, "dense_shared_expert");
    if (T < 1 || T > e.max_T()) failf("dense_shared_expert: %lld tokens (1..%lld)", (long long) T, (long long) e.max_T());
    DenseScratch& s = e.scratch_mut();
    // a decode token: x's q8_1 once (or the caller's), up and gate in one mmvq1 launch, the SwiGLU with
    // act's q8_1, down on it; the values of the calls below bit for bit
    const size_t rb = (size_t) (d.n_embd / kQK) * sizeof(mmvq::Q2KBlock);
    const int rows[2] = {d.n_ff, d.n_ff};
    const int R = T == 1 && (uintptr_t) w.up.data % 16 == 0 && (uintptr_t) w.gate.data % 16 == 0 ? mmvq::mmvq1_plan(rb, rows, 2, false) : 0;
    if (R) {
        cudaStream_t st = stream_of(e);
        if (!xq) {
            QActs q;
            qact_quantize(xn, nullptr, d.n_embd, 1, d.n_embd, QActLayout::Q8_1, s.qact, st, q);
            xq = s.qact;
        }
        mmvq::MmvqSegs L;
        L.x = static_cast<const mmvq::Q81Block*>(xq);
        L.n_in = d.n_embd;
        L.n_seg = 2;
        L.seg[0].w = static_cast<const uint8_t*>(w.up.data);
        L.seg[0].y = s.sh_up;
        L.seg[0].n_out = d.n_ff;
        L.seg[1].w = static_cast<const uint8_t*>(w.gate.data);
        L.seg[1].y = s.sh_gate;
        L.seg[1].n_out = d.n_ff;
        mmvq::mmvq1_launch<mmvq::Q2KTraits, mmvq::PlainEpi>(L, R, 1, {}, st);
        ck(cudaGetLastError(), "mmvq1_kernel (ffn_up_shexp, ffn_gate_shexp)");
        // act's q8_1 over x's (the stream has read it)
        swiglu_q8_kernel<<<(unsigned) ((d.n_ff + 255) / 256), 256, 0, st>>>(s.sh_gate, s.sh_up, d.n_ff, d.swiglu_limit,
                                                                              s.sh_act, static_cast<mmvq::Q81Block*>(s.qact));
        ck(cudaGetLastError(), "swiglu_q8_kernel");
        qmatmul_q(e.qmm(), w.down, QActs{QActLayout::Q8_1, 1, d.n_ff, s.qact}, out, d.n_embd, 1, st, "ffn_down_shexp");
        return;
    }
    qmatmul(e.qmm(), w.up, xn, d.n_embd, T, s.sh_up, d.n_ff, s.qact, 1, e.stream(), "ffn_up_shexp");
    qmatmul(e.qmm(), w.gate, xn, d.n_embd, T, s.sh_gate, d.n_ff, s.qact, 1, e.stream(), "ffn_gate_shexp");
    ffn_swiglu_clamp(s.sh_gate, s.sh_up, T * d.n_ff, d.swiglu_limit, s.sh_act, e.stream());
    qmatmul(e.qmm(), w.down, s.sh_act, d.n_ff, T, out, d.n_embd, s.qact, 1, e.stream(), "ffn_down_shexp");
}

}  // namespace strata::dsv41::engine
