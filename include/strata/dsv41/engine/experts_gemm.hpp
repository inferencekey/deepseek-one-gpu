// include/strata/dsv41/engine/experts_gemm.hpp - the routed experts' prefill computation, as plain host calls
// (experts.cpp's prefill passes; experts_test --gemm checks and times them): one grouped tensor-core computation per
// pass for all of its experts, gate and up with the SwiGLU clamp in one kernel, down in a second, reading the experts'
// q2_K / q3_K blocks where the slab, the ring or the prefetch region holds them (src/dsv41/engine/experts_gemm.cu has
// the design and its measured reasons).
//
// Numerics: the dequant path's (QPrefill::DequantF16, qmatmul.hpp): every weight decoded to the f16 value
// qref_dequant_row gives rounded to nearest even (qdequant.cuh, the dequant kernel's own decoder), the activations
// rounded to f16 the same way, the exact f16 products summed in f32 by the tensor cores; the clamp on the f32 gate and
// up (ffn_device.cuh), h rounded to f16 for down. qref_matmul_f16 is its reference; a row's values do not depend on
// the other rows of its tile or pass. Every pointer is device memory, `stream` a cudaStream_t.
#pragma once

#include "strata/dsv41/engine/qmatmul.hpp"

#include <cstdint>

namespace strata::dsv41::engine {

/// Routed rows per tile: the token side of one block of the grouped GEMM.
constexpr int kExpertTileRows = 128;
/// Output columns per block: gate/up features (each with its gate and its up row), down's output rows.
constexpr int kExpertGateUpCols = 64;
constexpr int kExpertDownCols = 128;

/// One tile of a pass: up to kExpertTileRows consecutive rows of the pass's buffers, all routed to one expert, whose
/// parts (0 up, 1 gate, 2 down) are at w, each 16-byte aligned (the kernels stage the blocks by 16-byte pieces from
/// 16-byte boundaries; a misaligned part traps).
struct ExpertTile {
    const void* w[3];
    int32_t row0, rows;
};
static_assert(sizeof(ExpertTile) == 32, "ExpertTile is uploaded as an array of 32-byte records");

/// Exits naming the problem unless the grouped GEMM takes these experts: gate and up of one type, every type q2_K or
/// q3_K (the experts' types in the real model and the tiny ones), n_ff a multiple of kExpertGateUpCols, n_embd of
/// kExpertDownCols, both of 256. Host only.
void expert_gemm_check(int type_up, int type_gate, int type_down, int64_t n_embd, int64_t n_ff);

/// xg [i][E] f16 = f16(x [pairs[i] / n_used][E]) for the n pairs of a pass grouped by expert (round to nearest even).
void expert_gather_rows_f16(const float* x, const int32_t* pairs, int n_used, int64_t n, int E, uint16_t* xg,
                            void* stream);

/// For each of the n_tiles tiles (device array) and its rows r: h [r][F] f16 = f16(clamp(x W_gate^T, x W_up^T)), x the
/// rows of xg [.][E] f16, W_gate and W_up [F][E] of type `type` (the tile's w[1] and w[0]).
void expert_gate_up(const ExpertTile* tiles, int64_t n_tiles, int type, const uint16_t* xg, int64_t E, int64_t F,
                    float limit, uint16_t* h, void* stream);

/// For each tile and its rows r: y [r][ldy] f32 = x W^T over x [.][K] f16, W [N][K] of type `type` (the tile's
/// w[part]). The routed experts' down (part 2, K = n_ff, N = n_embd); any part for the tests (N a multiple of
/// kExpertDownCols, K of 256).
void expert_gemm(const ExpertTile* tiles, int64_t n_tiles, int part, int type, const uint16_t* x, int64_t K, int64_t N,
                 float* y, int64_t ldy, void* stream);

}  // namespace strata::dsv41::engine
