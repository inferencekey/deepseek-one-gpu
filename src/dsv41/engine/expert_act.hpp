// src/dsv41/engine/expert_act.hpp - the SwiGLU clamp of the engine's FFNs as
// one implementation the CUDA kernels (ffn_device.cuh: the decode hits' epilogue, the shared expert, the prefill GEMM's
// epilogue) and the CPU experts in the GPU's numerics (experts_cpu_mmvq.cpp, mmvq_ref.cpp) both compile, so a routed
// expert's h has the same bits whichever side computes it. Internal to src/dsv41/engine.
//
// Before this, the device called CUDA's expf (2 ulp in the CUDA Math API's table) and the CPU glibc's expf: two
// implementations whose last bits differ for some inputs, one of the reasons a hit and a miss of the same expert
// differed (DETERMINISM.md row b). expf_det is Cephes' expf (Cody-Waite range reduction by ln 2 in two parts, a degree
// 5 polynomial, 2^n built from its exponent bits) written with one rounding per operation: on the device the _rn
// intrinsics, which nvcc never contracts, on the host plain operators and std::fma, in files built with
// -ffp-contract=off (CMakeLists.txt). dsv41_engine_expert_bits_test measures its error against double's exp on a
// sweep of every input the clamp can give it.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define DSV41_ACT_FN __host__ __device__ __forceinline__
#else
#define DSV41_ACT_FN inline
#endif

namespace strata::dsv41::engine::act {

DSV41_ACT_FN float mul(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
DSV41_ACT_FN float add(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
DSV41_ACT_FN float sub(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}
DSV41_ACT_FN float div(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
/// a b + c rounded once
DSV41_ACT_FN float fma(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
DSV41_ACT_FN float from_bits(uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}

/// e^x: NaN stays NaN, x past ln(FLT_MAX) (88.72284) gives +inf, x below -87.33654 (where 2^n leaves the normal range)
/// gives 0; between them Cephes' expf.
DSV41_ACT_FN float expf_det(float x) {
    if (x != x) return x;
    if (x > 88.72284f) return from_bits(0x7f800000u);
    if (!(x > -87.33654f)) return 0.0f;
    // n = x log2(e) to the nearest integer (ties to even): the fma rounds x log2(e) + 1.5 2^23 once to a whole number
    const float t = fma(x, 1.44269504f, 12582912.0f);
    const float n = sub(t, 12582912.0f);
    // r = x - n ln 2, ln 2 = 0.693359375 - 2.12194440e-4 (the first part's product is exact for |n| <= 128)
    float r = fma(n, -0.693359375f, x);
    r = fma(n, 2.12194440e-4f, r);
    const float z = mul(r, r);
    float p = 1.9875691500e-4f;
    p = fma(p, r, 1.3981999507e-3f);
    p = fma(p, r, 8.3334519073e-3f);
    p = fma(p, r, 4.1665795894e-2f);
    p = fma(p, r, 1.6666665459e-1f);
    p = fma(p, r, 5.0000001201e-1f);
    float y = add(fma(p, z, r), 1.0f);
    // y 2^n, n in -126 .. 128: 2^127 at most from the exponent bits, the last factor 2 apart
    int ni = (int) n;
    if (ni > 127) {
        y = mul(y, 2.0f);
        ni = 127;
    }
    return mul(y, from_bits((uint32_t) (ni + 127) << 23));
}

/// silu(min(gate, L)) clamp(up, -L, L), written gate / (1 + e^-gate) * up with each operation rounded (the fork's
/// ggml_swiglu_clamp order, ops.cpp 3408-3465, with expf_det for its expf). A NaN gate or up stays NaN.
DSV41_ACT_FN float swiglu_clamp(float gate, float up, float limit) {
    const float g = gate > limit ? limit : gate;
    const float u = up < -limit ? -limit : up > limit ? limit : up;
    return mul(div(g, add(1.0f, expf_det(-g))), u);
}

}  // namespace strata::dsv41::engine::act

#undef DSV41_ACT_FN
