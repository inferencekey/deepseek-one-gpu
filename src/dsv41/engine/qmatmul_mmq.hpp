// src/dsv41/engine/qmatmul_mmq.hpp - the bridge to the fork's ggml-cuda MMQ (qmatmul_mmq.cu, the only engine file
// compiled against ggml-cuda's headers), internal to src/dsv41/engine.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace strata::dsv41::engine::mmq {

/// A ggml_backend_cuda_context for the current device (its pool serves MMQ's stream-k fixup buffers).
void* context_new();
void context_delete(void* ctx);
/// The block_q8_1_mmq scale layout MMQ expects for a type: 0 D4, 1 DS4, 2 D2S6 (mmq_get_q8_1_ds_layout).
int ds_layout(int type);
/// sizeof(block_q8_1_mmq), checked against the engine's own tile struct.
int tile_bytes();
/// The largest tile the fork's MMQ has for this type on every device at w_rows rows, within its shared memory.
bool fits(int type, int64_t w_rows);
/// y [T][ldy] = acts . w^T on MMQ: w `groups` channels of rows_g rows of k values (channel g at row g * rows_g),
/// acts in the type's tile layout for T columns of groups * k values.
void run(void* ctx, int type, const void* w, int64_t k, int64_t rows_g, int groups, const void* acts, int64_t T,
         float* y, int64_t ldy, cudaStream_t stream);

}  // namespace strata::dsv41::engine::mmq
