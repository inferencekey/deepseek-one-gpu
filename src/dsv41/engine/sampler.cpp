// src/dsv41/engine/sampler.cpp - the host sampler of include/strata/dsv41/engine/sampler.hpp.
//
// The candidates are sorted lazily: only as far as top_k, top_p's cut, min_p's count and the draw's walk need (in
// growing steps), so a request with top_k 40 sorts 64 of the 129,280 ids, and top_k 0 with top_p the head that holds
// the mass. A NaN logit sorts as -inf: it is never drawn. On the dev VM (AMD Ryzen 7 5700X, g++ -O2; sampler_test.cpp,
// 129,280 logits, a dozen strong ones over a normal spread of sigma 2): ~0.2 ms per pick at top_k 40, ~0.3 at min_p
// 0.05 alone, ~1.7 with every filter off; top_k 0 with top_p 0.95 keeps 21,665 ids of that flat spread, ~7.6 ms.
#include "strata/dsv41/engine/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace strata::dsv41::engine {
namespace {

float key(const float* l, int32_t i) {
    const float v = l[i];
    return std::isnan(v) ? -std::numeric_limits<float>::infinity() : v;
}

}  // namespace

Sampler::Sampler(const SamplerParams& p)
    : p_(p),
      seed_(p.has_seed ? p.seed : (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}()),
      rng_(seed_) {}

// [0, m) of the pool sorted best first (ties to the lower id); the sorted head grows at least to double, and to 64
void Sampler::sort_to(int64_t m) {
    m = std::min(m, pool_);
    if (m <= sorted_) return;
    m = std::min(pool_, std::max({m, 2 * sorted_, (int64_t) 64}));
    const float* l = l_;
    auto better = [l](int32_t a, int32_t b) {
        const float ka = key(l, a), kb = key(l, b);
        return ka > kb || (ka == kb && a < b);
    };
    // the next best (m - sorted_) of the rest in order: a strict order, so the head is the full sort's. A short head
    // by a heap (most ids fail one comparison with its top), a long one by a selection, then its sort
    if (m - sorted_ <= 1024) {
        std::partial_sort(idx_.begin() + sorted_, idx_.begin() + m, idx_.begin() + pool_, better);
    } else {
        if (m < pool_) std::nth_element(idx_.begin() + sorted_, idx_.begin() + m, idx_.begin() + pool_, better);
        std::sort(idx_.begin() + sorted_, idx_.begin() + m, better);
    }
    sorted_ = m;
}

int64_t Sampler::cut(const std::vector<float>& logits) {
    const int64_t n = (int64_t) logits.size();
    l_ = logits.data();
    idx_.resize((size_t) n);
    std::iota(idx_.begin(), idx_.end(), 0);
    pool_ = n, sorted_ = 0;
    if (p_.top_k > 0 && p_.top_k < n) {
        sort_to(p_.top_k);
        pool_ = p_.top_k;
        sorted_ = std::min(sorted_, pool_);
    }
    sort_to(1);
    const float lmax = key(l_, idx_[0]);
    if (!std::isfinite(lmax)) return 1;   // every logit -inf or NaN: the first id, as the argmax takes it
    int64_t keep = pool_;
    if (p_.top_p < 1.0) {
        // top_p at temperature 1 over the top_k pool, in double (Strata's sampled_reference)
        double sum = 0.0;
        for (int64_t i = 0; i < pool_; ++i) sum += std::exp((double) key(l_, idx_[(size_t) i]) - (double) lmax);
        double cum = 0.0;
        for (int64_t i = 0; i < pool_; ++i) {
            sort_to(i + 1);
            cum += std::exp((double) key(l_, idx_[(size_t) i]) - (double) lmax) / sum;
            if (cum >= p_.top_p) { keep = i + 1; break; }
        }
    }
    if (p_.min_p > 0.0) {
        // the survivors of min_p are a prefix of the sorted candidates: those at or above the threshold
        const float thresh = lmax + std::log((float) p_.min_p);
        int64_t above = 0;
        for (int64_t i = 0; i < pool_; ++i) above += key(l_, idx_[(size_t) i]) >= thresh;
        keep = std::min(keep, std::max<int64_t>(above, 1));
    }
    if (keep < pool_) sort_to(keep);   // a cut keeps a sorted prefix; the whole pool needs no order
    return keep;
}

std::vector<int32_t> Sampler::survivors(const std::vector<float>& logits) {
    const int64_t keep = cut(logits);
    sort_to(keep);
    return std::vector<int32_t>(idx_.begin(), idx_.begin() + (std::ptrdiff_t) keep);
}

int32_t Sampler::pick(const std::vector<float>& logits) {
    const int64_t keep = cut(logits);
    const double u = (double) (rng_() >> 11) * 0x1.0p-53;   // the request's draw for this token, in [0, 1)
    const double lmax = key(l_, idx_[0]), inv_t = 1.0 / p_.temperature;
    if (!std::isfinite(lmax) || keep == 1) return idx_[0];
    // the softmax of the survivors at the temperature, summed in their order (all of the pool: its array order)
    double sum = 0.0;
    for (int64_t i = 0; i < keep; ++i) sum += std::exp(((double) key(l_, idx_[(size_t) i]) - lmax) * inv_t);
    const double target = u * sum;
    double cum = 0.0;
    for (int64_t i = 0; i < keep; ++i) {
        sort_to(i + 1);
        cum += std::exp(((double) key(l_, idx_[(size_t) i]) - lmax) * inv_t);
        if (target < cum) return idx_[(size_t) i];
    }
    return idx_[(size_t) (keep - 1)];
}

}  // namespace strata::dsv41::engine
