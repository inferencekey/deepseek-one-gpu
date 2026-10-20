// src/dsv41/engine/ffn_device.cuh - the fork's SwiGLU clamp as one device function, shared by the activation kernel
// of the shared expert (dense_ffn.cu), the epilogue of decode's grouped hits (qmatmul.cu qmatmul_experts) and
// that of the routed experts' grouped GEMM (experts_gemm.cu), so every path computes the same bits from the
// same gate and up. It is expert_act.hpp's, which the CPU experts in the GPU's
// numerics compile too (before: fminf, fmaxf and CUDA's expf). Internal to src/dsv41/engine.
#pragma once

#include "expert_act.hpp"

#include <cuda_runtime.h>

namespace strata::dsv41::engine {

/// silu(min(gate, L)) clamp(up, -L, L), written gate / (1 + e^-gate) * up as the fork's CPU does
/// (ggml_swiglu_clamp, ops.cpp 3408-3465), e^x by expert_act.hpp's expf_det.
__device__ __forceinline__ float swiglu_clamp(float gate, float up, float limit) { return act::swiglu_clamp(gate, up, limit); }

}  // namespace strata::dsv41::engine
