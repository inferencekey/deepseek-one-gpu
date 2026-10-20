// src/dsv41/engine/experts_cpu_mmvq.hpp - the CPU experts in the GPU's
// numerics (experts_cpu_mmvq.cpp, AVX2): decode's hit arithmetic (q8_1 activations, mmvq.cuh's traits, mmvq_kernel's
// order per row, expert_act.hpp's SwiGLU clamp) on the CPU, so a routed expert's output has the same bits on the CPU
// as on the GPU, whichever tier holds it. Every value is mmvq_ref.hpp's (the line-for-line reference) bit for bit:
// dsv41_engine_expert_bits_test. Internal to src/dsv41/engine (experts_cpu.cpp, CpuNumerics::Gpu).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::dsv41::engine::mmvq_cpu {

/// The weight types computed: q2_K and q3_K (DeepSeek-V4.1's experts: q2_K gate and up, q3_K down).
bool supports(int type);

/// x [k] (k a multiple of 32) into k / 32 q8_1 blocks of 36 bytes at out (mmvq_ref_quantize's bits).
void quantize(const float* x, int64_t k, uint8_t* out);

/// h[i] = act::swiglu_clamp(gate[i], up[i], limit) for i < n.
void swiglu(const float* gate, const float* up, int64_t n, float limit, float* h);

/// One activation as rows() reads it: its q8_1 blocks, per block its d as f32 and the sums of its values 4 by 4 as f32
/// (q2_K's min term: dot lane k's 4 values), so a call's activations are prepared once for all of its rows.
struct Act {
    const uint8_t* x = nullptr;
    const float* d8 = nullptr;
    const float* s4 = nullptr;
};
/// The f32 room prepare_act takes for an activation of n_in values.
constexpr int64_t act_floats(int64_t n_in) { return n_in / 32 + n_in / 4; }
/// The q8_1 blocks xq of n_in values (quantize's) as rows() reads them, their f32 data written into f.
Act prepare_act(const uint8_t* xq, int64_t n_in, float* f);

/// For every row r in [r0, r1) of w (row r at w + r * row_bytes, n_in values of `type`) and every activation t < n_acts
/// (prepare_act's of n_in values): outs[t][r] = mmvq_ref_dot(type, row r, acts[t].x, n_in).
void rows(int type, const uint8_t* w, size_t row_bytes, int64_t r0, int64_t r1, int64_t n_in, const Act* acts,
          float* const* outs, int64_t n_acts);

}  // namespace strata::dsv41::engine::mmvq_cpu
