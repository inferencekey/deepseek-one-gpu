// src/dsv41/ggml_cpu_vec.h - the one ggml-cpu internal function strata-moe calls.
//
// The stock graph's plain SwiGLU (ggml_swiglu_split, GGML_GLU_OP_SWIGLU on f32) runs ggml_vec_swiglu_f32, whose SIMD
// silu has its own exp approximation, so the reference matches the stock output bit for bit only by calling it too.
// libggml-cpu exports it but ships no public header for it: the declaration is copied from
// ggml/src/ggml-cpu/vec.h at the fork's base 5210c7c5, and src/dsv41/CMakeLists.txt checks at configure time that
// the fork's libggml-cpu exports it.
#pragma once

extern "C" {

// y[i] = silu(x[i]) * g[i] for i in [0, n): x is the gate, g the up projection
void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g);
}
