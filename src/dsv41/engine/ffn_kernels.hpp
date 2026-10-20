// src/dsv41/engine/ffn_kernels.hpp - the FFN kernels the dense ops and the routed experts share, as plain host calls
// (internal to src/dsv41/engine; experts.cpp is host C++): the fork's SwiGLU clamp (dense_ffn.cu), the experts'
// slot combine and decode's doorbell (experts_kernels.cu). Every pointer is device memory, `stream` a
// cudaStream_t.
#pragma once

#include <cstdint>

namespace strata::dsv41::engine {

/// act[i] = silu(min(gate[i], L)) clamp(up[i], -L, L), written gate / (1 + expf(-gate)) * up as the fork's CPU does
/// (ggml_swiglu_clamp, ops.cpp 3408-3465).
void ffn_swiglu_clamp(const float* gate, const float* up, int64_t n, float limit, float* act, void* stream);

/// out [T][E] = (((y_0 w_0) + y_1 w_1) + ... + y_{K-1} w_{K-1}) + shared, y_s the row of pair t * K + s: row r =
/// row_of_pair[p] >= 0 in yg, else row -r - 1 in ycpu; w [T][K]; each product and sum rounded (ggml_mul, the view
/// adds of build_moe_ffn, ffn_out's add). shared may be null.
void expert_combine(const float* yg, const float* ycpu, const int32_t* row_of_pair, const float* w, const float* shared,
                    int64_t T, int n_used, int E, float* out, void* stream);
/// out[i] = out[i] + shared[i], one rounding (forward.cpp's MoE dumps run expert_combine with shared null and
/// add it after; the same bits as the combine's own add).
void expert_add_shared(float* out, const float* shared, int64_t n, void* stream);

// ---- decode's doorbell (Strata's layer.hpp Doorbell, elementwise.hpp doorbell_publish / doorbell_wait). x_out,
// ids_out, seq, flag, ycpu_mapped and miss_row are the device views of mapped pinned host memory.

/// x_out [n_x] = x, ids_out [n_ids] = ids, then *seq += 1 behind a system fence (the host polls seq). One kernel.
void expert_doorbell_publish(const float* x, const int32_t* ids, int64_t n_x, int64_t n_ids, float* x_out,
                             int32_t* ids_out, uint32_t* seq, void* stream);
/// One thread spins until *flag equals *seq (the host writes the ring's value once the CPU rows are in place).
void expert_doorbell_wait(const uint32_t* flag, const uint32_t* seq, void* stream);
/// After the wait, per pair p < P: r = miss_row[p] >= 0 (the CPU's row) copies ycpu_mapped row r to ycpu row r and
/// sets row_of_pair[p] = -r - 1; r < 0 (a GPU hit) sets row_of_pair[p] = p (the hits' rows are pair-indexed).
void expert_misses_in(const float* ycpu_mapped, const int32_t* miss_row, int64_t P, int E, float* ycpu,
                      int32_t* row_of_pair, void* stream);

// ---- a decode token's small transfers through mapped pinned memory, moved by the
// device's threads instead of a copy engine. Copies in one direction run in the order they were queued, whatever
// their stream, so a token's cudaMemcpyAsync of its inputs (and of its logits) waited behind the whole open batch of
// the hot set's swap copies on the copy streams (Strata's copy_from_mapped_multi and the doorbell's mapped rows are
// the pattern, src/kernels/cuda/elementwise.cu).

constexpr int kMaxMappedCopies = 8;

/// One copy of `words` 32-bit words from src (the device view of mapped pinned host memory) into dst (device).
struct MappedCopy {
    void* dst = nullptr;
    const void* src = nullptr;
    int64_t words = 0;
};
/// The n copies (1..kMaxMappedCopies, exits otherwise) in one launch, read when the launch runs.
void copy_from_mapped(const MappedCopy* copies, int n, void* stream);
/// dst [n] (the device view of mapped pinned host memory) = src [n] (device), written by the device's threads.
void copy_to_mapped(float* dst, const float* src, int64_t n, void* stream);

}  // namespace strata::dsv41::engine
