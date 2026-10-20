// src/dsv41/moe_gpu.h - the GPU tier of strata-moe: a fixed hot set of experts per layer resident in VRAM, their
// hits computed with the fork's own ggml-cuda kernels (src/dsv41/moe_gpu.cpp), and the prefill path that
// streams the other experts of a batch call to the GPU (moe_prefill.h).
#pragma once

#include "moe_hot_set.h"
#include "moe_op.h"

#include <memory>

namespace strata::dsv41 {

// The prefill path (STRATA_MOE_PREFILL_*, src/dsv41/moe.cpp): batch calls of at least min_tokens tokens (0: never)
// run every pair on the GPU through a ring of n_sets sets of set_size expert slots, staged experts through
// n_stage_sets sets of pinned staging, filled by `cold` (with kDirect the call's residency reads them).
struct PrefillConfig {
    int64_t min_tokens = 0;
    int set_size = 0, n_sets = 0, n_stage_sets = 0;
    PrefillCold cold = PrefillCold::kPageCache;
    bool cold_is_default = true;
};

// The GPU tier on CUDA device 0, with the slab of every layer of `hot` allocated in one VRAM buffer at once
// (aborts when it exceeds vram_budget_gib or the device's free memory), and with prefill.min_tokens > 0 the ring
// and the pinned staging of the prefill path beside it (aborts when they do not fit in what the slab leaves free).
// Each layer's slab is filled from the fork's host expert tensors at its first call.
std::unique_ptr<HitPath> make_gpu_tier(const HotSet& hot, double vram_budget_gib, const PrefillConfig& prefill);

}  // namespace strata::dsv41
