// include/strata/dsv41/engine/sampler.hpp - the host sampler of
// `dsv41-engine --serve`, for the requests with a temperature above 0 (temperature, top_p, top_k and seed
// over the OpenAI API). A request at temperature 0, or without one, has no Sampler: the Session takes the argmax as
// before, so the greedy path is the same code.
//
// The chain is llama.cpp's default order, as Strata's own sampler (include/strata/kernels/sampler.hpp) runs it:
//
//     top_k -> top_p -> min_p -> temperature -> draw
//
// with Strata's arithmetic (src/kernels/sampler_parity.cpp, sampled_reference): the candidates best first (ties to
// the lower id), top_p's cumulative mass in double over the top_k list at temperature 1 (the first candidates whose
// mass reaches top_p, at least one), min_p's cut at max logit + log(min_p) on its survivors (a prefix of them), then
// the temperature and one draw over the softmax of the survivors. Two differences, both on purpose: top_k is
// any k up to the vocabulary (Strata's CUDA chain keeps at most 64; 0 is off, the whole vocabulary), and the draw is
// a std::mt19937_64 per request, seeded with the request's seed or, without one, from std::random_device (the seed
// used is reported, so any reply can be drawn again). One uniform per generated token: the 53 high bits of the
// generator's next value, in [0, 1). The logits are already on the host (the Session's), so no GPU code is involved.
// Penalties are not part of it: dsv41-engine refuses them with ERR.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace strata::dsv41::engine {

struct SamplerParams {
    double temperature = 1.0;   ///< > 0 (a request at 0 gets no Sampler)
    int64_t top_k = 0;          ///< 0: off; k: the k highest logits (k >= the vocabulary: all of them)
    double top_p = 1.0;         ///< (0, 1]; 1: off
    double min_p = 0.0;         ///< [0, 1]; 0: off; keeps the logits >= max + log(min_p)
    bool has_seed = false;      ///< false: seeded from std::random_device
    uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(const SamplerParams& p);

    /// One token from logits [n_vocab] (n_vocab >= 1); advances the generator by one draw.
    int32_t pick(const std::vector<float>& logits);
    /// The candidates the draw is made over, best first: top_k, top_p and min_p applied (no draw, no state change).
    std::vector<int32_t> survivors(const std::vector<float>& logits);

    const SamplerParams& params() const { return p_; }
    uint64_t seed() const { return seed_; }   ///< the request's seed, or the one std::random_device gave

private:
    void sort_to(int64_t m);
    int64_t cut(const std::vector<float>& logits);   // top_k, top_p, min_p over logits_; -> the survivors' count

    SamplerParams p_;
    uint64_t seed_;
    std::mt19937_64 rng_;
    // the call's candidate ids: [0, pool_) the top_k pool, [0, sorted_) of it sorted best first
    std::vector<int32_t> idx_;
    const float* l_ = nullptr;
    int64_t pool_ = 0, sorted_ = 0;
};

}  // namespace strata::dsv41::engine
