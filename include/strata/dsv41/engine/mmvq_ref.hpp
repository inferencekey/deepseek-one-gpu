// include/strata/dsv41/engine/mmvq_ref.hpp - the GPU-order reference of
// decode's routed experts, plain scalar C++ written to mirror the CUDA code line for line (src/dsv41/engine/mmvq_ref.cpp):
// quantize_q8_1_kernel and mmvq.cuh's q8_1_warp for the activations, mmvq.cuh's Q2KTraits and Q3KTraits for one weight
// block against them, mmvq_kernel's one-column order for a row (the 128 threads' running sums, warps 1 to 3 added to
// warp 0's in turn, the xor butterfly), and qmatmul_experts for an expert on one token (gate and up, the SwiGLU clamp
// of expert_act.hpp, h quantized, down). Every float operation is the one the kernel rounds, in its order, so its
// values are the CUDA kernels' bit for bit (a dev GPU pod checks the kernels against the CPU experts that this
// reference checks on the dev VM: dsv41_engine_expert_bits_test).
#pragma once

#include "strata/dsv41/engine/dense.hpp"
#include "strata/dsv41/engine/model.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::dsv41::engine {

/// One q8_1 block on the host as the device stores it (mmvq.cuh Q81Block): half2 ds (the f16 scale d, then the f16 sum
/// of the 32 raw values), then 32 int8.
constexpr size_t kQ81BlockBytes = 36;

/// x [k] (k a multiple of 32) into k / 32 q8_1 blocks at out, quantize_q8_1_kernel's bits: per block amax (fmaxf over
/// |x|), the sum by warp_sum's butterfly, d = amax / 127 kept finite, q = roundf(x / d) held to +-127 (0 for an
/// all-zero block), d and the sum stored as f16.
void mmvq_ref_quantize(const float* x, int64_t k, uint8_t* out);

/// Row `row` (n_in / 256 blocks of `type`: q2_K or q3_K; exits otherwise) against the q8_1 activations xq of n_in
/// values: mmvq_kernel's value at one column (and mmvq1_kernel's, mmvq_experts_kernel's, the grouped kernel's).
float mmvq_ref_dot(int type, const uint8_t* row, const uint8_t* xq, int64_t n_in);

/// Expert `expert` of layer L on the token x [n_embd]: y [n_embd] as qmatmul_experts computes a hit (x quantized,
/// gate and up rows with the clamp fused, h [n_ff] quantized, down rows).
void mmvq_ref_expert(const ExpertLayerHost& L, const DenseDims& d, int32_t expert, const float* x, float* y);

}  // namespace strata::dsv41::engine
