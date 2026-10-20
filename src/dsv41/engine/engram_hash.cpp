// src/dsv41/engine/engram_hash.cpp - the engram n-gram row hash of include/strata/dsv41/engine/engram.hpp, written
// after llm_graph_input_engram::set_input (deepseek41.cpp 309-356). Host only: the GPU path and the CPU reference
// both take its rows.
#include "strata/dsv41/engine/engram.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 engram: %s\n", msg);
    std::exit(1);
}

}  // namespace

int EngramHash::index_of(int il) const {
    for (size_t e = 0; e < layer_ids.size(); ++e)
        if (layer_ids[e] == il) return (int) e;
    return -1;
}

int64_t EngramHash::rows_needed(int e) const {
    uint64_t mx = 0;
    for (int b = 0; b < cols(); ++b) {
        const size_t k = (size_t) e * cols() + b;
        mx = std::max(mx, offsets[k] + primes[k]);
    }
    return (int64_t) mx;
}

void EngramHash::check() const {
    const size_t L = layer_ids.size();
    if (L == 0 || n_gram < 2 || heads < 1) failf("%zu engram layers, n-gram %d, %d heads", L, n_gram, heads);
    if (multipliers.size() != L * (size_t) n_gram) failf("%zu multipliers for %zu layers of %d-grams", multipliers.size(), L, n_gram);
    if (primes.size() != L * (size_t) cols() || offsets.size() != primes.size())
        failf("%zu primes and %zu offsets for %zu layers of %d buckets", primes.size(), offsets.size(), L, cols());
    for (uint64_t p : primes)
        if (p == 0) failf("a prime of zero");
    if (token_map.empty()) failf("no token map");
}

void engram_rows(const EngramHash& h, int e, const int32_t* tokens, int64_t T, const int32_t* prev, int n_prev,
                 int32_t* rows) {
    const int G = h.n_gram, H = h.heads, C = h.cols();
    if (e < 0 || (size_t) e >= h.layer_ids.size()) failf("engram index %d of %zu", e, h.layer_ids.size());
    if (n_prev < 0 || n_prev > G - 1) failf("%d previous tokens (0 .. %d)", n_prev, G - 1);
    const uint64_t* mult = h.multipliers.data() + (size_t) e * G;
    const uint64_t* prime = h.primes.data() + (size_t) e * C;
    const uint64_t* offset = h.offsets.data() + (size_t) e * C;
    // an id outside the map (an image position in the fork) reads as the pad id, as map_of does
    auto map_of = [&](int32_t t) -> uint64_t {
        if (t < 0 || (size_t) t >= h.token_map.size()) return (uint64_t) h.pad;
        return (uint64_t) h.token_map[(size_t) t];
    };
    // token i of the chunk sits at hist[n_prev + i]; hist[j] for j < 0 is before the sequence start
    auto at = [&](int64_t j) -> int32_t { return j < n_prev ? prev[j] : tokens[j - n_prev]; };
    std::vector<uint64_t> ctx((size_t) G);
    for (int64_t i = 0; i < T; ++i) {
        ctx[0] = map_of(tokens[i]);
        bool blocked = false;
        for (int s = 1; s < G; ++s) {
            const int64_t j = n_prev + i - s;
            blocked = blocked || j < 0;
            ctx[(size_t) s] = blocked ? (uint64_t) h.pad : map_of(at(j));
        }
        uint64_t rolling = ctx[0] * mult[0];
        for (int s = 1; s < G; ++s) {
            rolling ^= ctx[(size_t) s] * mult[s];
            for (int hh = 0; hh < H; ++hh) {
                const int b = (s - 1) * H + hh;
                rows[i * C + b] = (int32_t) (rolling % prime[b] + offset[b]);
            }
        }
    }
}

}  // namespace strata::dsv41::engine
