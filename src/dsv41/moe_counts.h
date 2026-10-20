// src/dsv41/moe_counts.h - per-(layer, expert) activation counts of strata_moe_forward while it serves
// (STRATA_MOE_COUNTS=<path>, src/dsv41/moe.cpp), so the hot set and the warm lists (tools/dsv41/oracle/hot_set.py
// --counts) come from a real decode profile over many prompts instead of the 95-token oracle dump.
//
// Every finished call adds each (token, slot) pair's expert to its layer, whichever tier computed it, decode calls
// (1 token) and batch calls (more) apart. The file is json, written whole to <path>.tmp and renamed over <path>,
// so a reader never sees half a file: once at the first call (a path that cannot be written aborts then, not at
// exit), then every `every` calls and at exit. Format (strata-moe-counts/1; hot_set.py reads and merges files):
//   {"format": "strata-moe-counts/1", "layers": [
//    {"layer": L, "n_expert": N, "decode_calls": C, "batch_calls": B, "batch_tokens": T,
//     "decode": [count per expert id], "batch": [count per expert id]}, ...]}
// with the layers in ascending order.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace strata::dsv41 {

class ActivationCounts {
public:
    // Counts written to `path` every `every` calls (>= 1) and by write().
    ActivationCounts(std::string path, int64_t every);

    // One finished call of `layer` (n_expert experts): its selected_experts tensor [n_used, n_tokens]. Aborts naming
    // the layer when n_expert differs from the layer's earlier calls, or an id is outside [0, n_expert).
    void record(int layer, int64_t n_expert, const ggml_tensor* ids);
    // The same from plain memory (the dedicated engine, src/dsv41/engine/experts.cpp): ids [n_tokens][n_used] i32.
    void record(int layer, int64_t n_expert, const int32_t* ids, int64_t n_used, int64_t n_tokens);
    // The file's text now.
    std::string json() const;
    // Writes the file now (<path>.tmp, then rename); aborts naming the path and errno when it cannot.
    void write() const;

    const std::string& path() const { return path_; }

private:
    struct Layer {
        int64_t decode_calls = 0, batch_calls = 0, batch_tokens = 0;
        std::vector<int64_t> decode, batch;
    };
    // both record forms: id_at(t, s) is token t's slot s
    template <class IdAt>
    void record_ids(int layer, int64_t n_expert, int64_t n_used, int64_t n_tokens, IdAt id_at);
    std::string json_locked() const;
    void write_locked() const;

    std::string path_;
    int64_t every_ = 0;
    int64_t calls_ = 0;  // since the start
    std::map<int, Layer> layers_;
    mutable std::mutex mutex_;
};

}  // namespace strata::dsv41
