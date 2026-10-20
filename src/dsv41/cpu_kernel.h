// src/dsv41/cpu_kernel.h - strata-moe's own CPU expert kernel (STRATA_MOE_CPU=fast): K-quant weight rows against
// Q8_K activations, every token routed to the expert in one pass over each row (src/dsv41/cpu_kernel_avx2.cpp).
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace strata::dsv41 {

// The weight types the kernel computes: q2_K, q3_K (DeepSeek-V4.1), q4_K, q6_K (the OLMoE smoke model). Their
// ggml-cpu activation type is Q8_K for all four.
bool fast_kernel_supports(ggml_type type);

// Whether this CPU has what the kernel is compiled for (AVX2, FMA, F16C), asked at run time. Defined in
// cpu_kernel.cpp, which is built without those flags, so it runs on any x86-64.
bool fast_kernel_cpu_ok();

// For every row r in [r0, r1) of `w` (row r at w + r * row_bytes, n_per_row weights of `type`) and every
// activation t in [0, n_acts) (a Q8_K row of n_per_row values): outs[t][r] = row r . acts[t]. Each result is
// ggml-cpu's AVX2 vec_dot of that row and activation to the bit, whatever the number of rows and tokens.
void fast_kernel_rows(ggml_type type, const char* w, size_t row_bytes, int64_t r0, int64_t r1, int64_t n_per_row,
                      const void* const* acts, float* const* outs, int64_t n_acts);

}  // namespace strata::dsv41
