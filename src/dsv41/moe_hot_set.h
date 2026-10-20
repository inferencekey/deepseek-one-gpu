// src/dsv41/moe_hot_set.h - a hot-set file (STRATA_MOE_HOT; tools/dsv41/oracle/hot_set.py writes them, its README
// has the format): per layer, the experts the GPU tier keeps in VRAM (moe_gpu.h) and the ones the warm tier keeps
// in RAM (moe_residency.h). load_hot_set is in moe_hot_set.cpp.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::dsv41 {

struct HotLayer {
    int layer = -1;
    int64_t n_expert = 0, n_embd = 0, n_ff = 0;
    ggml_type up_type = GGML_TYPE_COUNT, gate_type = GGML_TYPE_COUNT, down_type = GGML_TYPE_COUNT;
    std::vector<int32_t> experts;  // resident in VRAM, in slab slot order
    std::vector<int32_t> warm;     // resident in the warm tier, in slot order (none in a file without a warm list)
};

struct HotSet {
    std::string path;
    double budget_gib = 0.0;      // the VRAM budget hot_set.py packed it for
    double ram_budget_gib = -1.0; // the RAM budget of its warm lists, -1: the file has none
    std::vector<HotLayer> layers;
};

// Parses a hot-set file; aborts naming the file and the field on anything it cannot use.
HotSet load_hot_set(const char* path);

}  // namespace strata::dsv41
