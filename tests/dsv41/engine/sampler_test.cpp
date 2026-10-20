// tests/dsv41/engine/sampler_test.cpp - the host sampler of
// include/strata/dsv41/engine/sampler.hpp and the Session's pick, GPU free (plain C++: verify.sh runs the CMake target,
// tools/dsv41/engine/sampler_test.sh builds it with the host compiler and checks that a broken seeding fails it).
//
// Checked, on fixed logits:
//   masks     top_k, top_p and min_p against hand-computed survivors on five logits ln(1/2), ln(1/4), ln(1/8),
//             ln(1/16), ln(1/16) placed at shuffled ids (top_p's cumulative mass 0.5, 0.75, 0.875, 0.9375, 1; min_p's
//             threshold p >= min_p / 2), ties to the lower id, top_k 0 and top_k above the vocabulary as off, the chain
//             order (top_p before min_p), a NaN and a -inf logit never drawn;
//   draws     the frequencies of 40,000 draws at temperature 1 (the five probabilities) and at 0.5 (their squares,
//             normalized) within 0.01;
//   seeds     the same seed draws the same 100 tokens; seeds 1234 and 1235 differ in at least 20 of 100 draws (about
//             66 expected: 1 - sum p^2); without a seed two samplers report different seeds, and a sampler given a
//             reported seed draws that sampler's tokens again;
//   lazy sort the survivors and the picks on 129,280 random logits (the real vocabulary) equal a full sort's, for
//             top_k 40, top_k 0 with top_p 0.95, min_p 0.05 alone, and the three together;
//   greedy    the Session without a sampler takes the argmax (ties to the lower id), a sampler at top_k 1 the same
//             ids, and a sampled request's ids are the sampler's draws.
#include "strata/dsv41/engine/sampler.hpp"
#include "strata/dsv41/engine/session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

std::string ids_text(const std::vector<int32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

e::SamplerParams params(double t, int64_t k, double p, double m, uint64_t seed) {
    e::SamplerParams sp;
    sp.temperature = t, sp.top_k = k, sp.top_p = p, sp.min_p = m, sp.has_seed = true, sp.seed = seed;
    return sp;
}

// the five probabilities at ids 7, 2, 9, 0, 4 of ten (the others -1e30, below any cut), plus a NaN at 5 and -inf at 6
const std::vector<int32_t> kIds = {7, 2, 9, 0, 4};
const double kProb[5] = {0.5, 0.25, 0.125, 0.0625, 0.0625};

std::vector<float> five() {
    std::vector<float> l(10, -1e30f);
    for (int i = 0; i < 5; ++i) l[(size_t) kIds[(size_t) i]] = (float) std::log(kProb[i]);
    l[5] = std::numeric_limits<float>::quiet_NaN();
    l[6] = -std::numeric_limits<float>::infinity();
    return l;
}

std::vector<int32_t> first(size_t n) { return std::vector<int32_t>(kIds.begin(), kIds.begin() + (std::ptrdiff_t) n); }

void test_masks() {
    const std::vector<float> l = five();
    auto surv = [&](int64_t k, double p, double m) { return e::Sampler(params(1.0, k, p, m, 1)).survivors(l); };
    expect(surv(3, 1, 0) == first(3), "top_k 3 keeps ids 7,2,9 (got " + ids_text(surv(3, 1, 0)) + ")");
    expect(surv(5, 1, 0) == first(5), "top_k 5 keeps the five, best first, ties (0 and 4) to the lower id");
    expect(surv(0, 1, 0).size() == 10 && surv(1000000, 1, 0).size() == 10, "top_k 0 and top_k above the vocabulary: off");
    expect(surv(0, 0.7, 0) == first(2), "top_p 0.7: the mass reaches 0.75 at the second (got " + ids_text(surv(0, 0.7, 0)) + ")");
    expect(surv(0, 0.8, 0) == first(3), "top_p 0.8: 0.875 at the third");
    expect(surv(0, 0.9, 0) == first(4), "top_p 0.9: 0.9375 at the fourth");
    expect(surv(0, 0.0, 0) == first(1), "top_p 0: at least one candidate");
    expect(surv(0, 1, 0.2) == first(3), "min_p 0.2: p >= 0.1 keeps three (got " + ids_text(surv(0, 1, 0.2)) + ")");
    expect(surv(0, 1, 0.3) == first(2), "min_p 0.3: p >= 0.15 keeps two");
    expect(surv(0, 1, 0.6) == first(1), "min_p 0.6: p >= 0.3 keeps one");
    expect(surv(2, 1, 0.2) == first(2), "top_k 2 then min_p 0.2: two");
    expect(surv(0, 0.9, 0.2) == first(3), "top_p 0.9 (four) then min_p 0.2 (three): three");
    expect(surv(0, 0.7, 0.2) == first(2), "top_p 0.7 (two) then min_p 0.2: two");
    e::Sampler all(params(1.0, 0, 1, 0, 7));
    bool never = true;
    for (int i = 0; i < 2000; ++i) {
        const int32_t t = all.pick(l);
        never = never && t != 5 && t != 6 && std::find(kIds.begin(), kIds.end(), t) != kIds.end();
    }
    expect(never, "2000 draws with every filter off: never the NaN (id 5), the -inf (id 6) nor the -1e30 ids");
}

void test_draws() {
    const std::vector<float> l = five();
    for (double t : {1.0, 0.5}) {
        e::Sampler s(params(t, 0, 1, 0, 20261013));
        std::vector<int> n(10, 0);
        const int draws = 40000;
        for (int i = 0; i < draws; ++i) ++n[(size_t) s.pick(l)];
        double z = 0;
        for (double p : kProb) z += std::pow(p, 1.0 / t);
        double worst = 0;
        for (int i = 0; i < 5; ++i)
            worst = std::max(worst, std::fabs((double) n[(size_t) kIds[(size_t) i]] / draws - std::pow(kProb[i], 1.0 / t) / z));
        char what[160];
        std::snprintf(what, sizeof what, "temperature %.1f: %d draws within 0.01 of p^(1/T) normalized (worst %.4f)", t, draws, worst);
        expect(worst < 0.01, what);
    }
}

void test_seeds() {
    const std::vector<float> l = five();
    auto draws = [&](const e::SamplerParams& p) {
        e::Sampler s(p);
        std::vector<int32_t> v;
        for (int i = 0; i < 100; ++i) v.push_back(s.pick(l));
        return v;
    };
    const std::vector<int32_t> a = draws(params(1.0, 0, 1, 0, 1234)), b = draws(params(1.0, 0, 1, 0, 1234)),
                               c = draws(params(1.0, 0, 1, 0, 1235));
    int differ = 0;
    for (size_t i = 0; i < a.size(); ++i) differ += a[i] != c[i];
    expect(a == b, "seed 1234 twice: the same 100 draws");
    expect(differ >= 20, "seeds 1234 and 1235: " + std::to_string(differ) + " of 100 draws differ (at least 20)");
    e::SamplerParams none = params(1.0, 0, 1, 0, 0);
    none.has_seed = false;
    e::Sampler r1(none), r2(none);
    expect(r1.seed() != r2.seed(), "no seed: two samplers seeded from std::random_device report different seeds (" +
           std::to_string(r1.seed()) + ", " + std::to_string(r2.seed()) + ")");
    std::vector<int32_t> d1;
    for (int i = 0; i < 100; ++i) d1.push_back(r1.pick(l));
    expect(draws(params(1.0, 0, 1, 0, r1.seed())) == d1, "the reported seed draws that sampler's 100 tokens again");
    expect(e::Sampler(params(1.0, 0, 1, 0, 0)).seed() == 0, "seed 0 is a seed (reported as 0)");
}

// the full-sort reference of one draw (the sampler's arithmetic on a fully sorted copy)
struct Reference {
    std::vector<int32_t> surv;
    int32_t pick;
};

Reference reference(const std::vector<float>& l, const e::SamplerParams& p, double u) {
    std::vector<int32_t> idx(l.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return l[(size_t) a] > l[(size_t) b]; });
    if (p.top_k > 0 && (size_t) p.top_k < idx.size()) idx.resize((size_t) p.top_k);
    const double lmax = l[(size_t) idx[0]];
    size_t keep = idx.size();
    if (p.top_p < 1) {
        double sum = 0, cum = 0;
        for (int32_t i : idx) sum += std::exp((double) l[(size_t) i] - lmax);
        for (size_t i = 0; i < idx.size(); ++i) {
            cum += std::exp((double) l[(size_t) idx[i]] - lmax) / sum;
            if (cum >= p.top_p) { keep = i + 1; break; }
        }
    }
    if (p.min_p > 0) {
        const float thresh = (float) lmax + std::log((float) p.min_p);
        size_t above = 0;
        for (int32_t i : idx) above += l[(size_t) i] >= thresh;
        keep = std::min(keep, std::max<size_t>(above, 1));
    }
    idx.resize(keep);
    double sum = 0, cum = 0;
    for (int32_t i : idx) sum += std::exp(((double) l[(size_t) i] - lmax) / p.temperature);
    Reference r{idx, idx.back()};
    for (int32_t i : idx) {
        cum += std::exp(((double) l[(size_t) i] - lmax) / p.temperature);
        if (u * sum < cum) { r.pick = i; break; }
    }
    return r;
}

void test_lazy_sort() {
    // 129,280 logits: a few strong ones over a normal spread, as a model's head gives them
    std::mt19937 g(20261013);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> l(129280);
    for (float& x : l) x = nd(g);
    for (int i = 0; i < 12; ++i) l[(size_t) (g() % l.size())] = 14.f - (float) i;
    struct Case { const char* name; e::SamplerParams p; bool full_survivors; };
    const Case cases[] = {{"top_k 40, top_p 0.95, temperature 0.7", params(0.7, 40, 0.95, 0, 1234), true},
                          {"top_k 0, top_p 0.95, temperature 0.7", params(0.7, 0, 0.95, 0, 1234), true},
                          {"min_p 0.05 alone, temperature 1", params(1.0, 0, 1, 0.05, 99), true},
                          {"top_k 1000, top_p 0.9, min_p 0.02, temperature 1.3", params(1.3, 1000, 0.9, 0.02, 5), true},
                          {"every filter off, temperature 1", params(1.0, 0, 1, 0, 42), false}};
    for (const Case& c : cases) {
        e::Sampler s(c.p);
        std::mt19937_64 rng(c.p.seed);
        bool same_pick = true, same_surv = true;
        size_t n_surv = 0;
        double ms = 0;
        for (int d = 0; d < 20; ++d) {
            // a different row per draw: the logits shifted by one id each time
            std::rotate(l.begin(), l.begin() + 1, l.end());
            const double u = (double) (rng() >> 11) * 0x1.0p-53;
            const Reference r = reference(l, c.p, u);
            const auto t0 = std::chrono::steady_clock::now();
            const int32_t got = s.pick(l);
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            same_pick = same_pick && got == r.pick;
            if (c.full_survivors) {
                e::Sampler v(c.p);
                const std::vector<int32_t> sv = v.survivors(l);
                same_surv = same_surv && sv == r.surv;
                n_surv = sv.size();
            }
        }
        char what[256];
        std::snprintf(what, sizeof what, "129,280 logits, %s: 20 picks equal the full sort's%s (%s%.3f ms per pick on "
                      "this CPU, information only)", c.name, c.full_survivors ? " and so do the survivors" : "",
                      c.full_survivors ? (std::to_string(n_surv) + " survivors, ").c_str() : "", ms / 20);
        expect(same_pick && same_surv, what);
    }
}

// the Session's pick: a backend whose logits are a fixed row per position, so the argmax and the draws are known
class RowBackend final : public e::SessionBackend {
public:
    explicit RowBackend(std::vector<std::vector<float>> rows) : rows_(std::move(rows)) {}
    int64_t ctx() const override { return 64; }
    int64_t chunk() const override { return 64; }
    int64_t n_vocab() const override { return (int64_t) rows_[0].size(); }
    void reset() override { pos_ = 0; }
    int checkpoints() const override { return 0; }
    void save_checkpoint(int) override {}
    void restore_checkpoint(int) override {}
    void prefill(const int32_t*, int64_t n, std::vector<float>& logits) override { pos_ += n; logits = row(); }
    void decode(int32_t, std::vector<float>& logits) override { ++pos_; logits = row(); }

private:
    const std::vector<float>& row() const { return rows_[(size_t) pos_ % rows_.size()]; }
    std::vector<std::vector<float>> rows_;
    int64_t pos_ = 0;
};

void test_session() {
    std::mt19937 g(7);
    std::uniform_real_distribution<float> ud(-3.f, 3.f);
    std::vector<std::vector<float>> rows(8, std::vector<float>(64));
    for (auto& r : rows)
        for (float& x : r) x = ud(g);
    rows[3][10] = rows[3][20] = 9.f;   // a tie: the argmax takes the lower id, 10
    std::vector<int32_t> want;
    for (int s = 0; s < 8; ++s) {
        const std::vector<float>& r = rows[(size_t) (2 + s) % rows.size()];   // the prompt of 2 reads position 2 first
        want.push_back((int32_t) (std::max_element(r.begin(), r.end()) - r.begin()));
    }
    auto run = [&](e::Sampler* s) {
        RowBackend b(rows);
        e::Session session(b, {});
        e::SessionRequest req;
        req.ids = {1, 2};
        req.max_new = 8;
        req.sampler = s;
        return session.run(req, {}).tokens;
    };
    expect(run(nullptr) == want, "Session without a sampler: the argmax per step, the tie to the lower id (" + ids_text(want) + ")");
    e::Sampler k1(params(0.8, 1, 1, 0, 3));
    expect(run(&k1) == want, "a sampler at top_k 1: the argmax ids too");
    e::Sampler sampled(params(1.0, 0, 1, 0, 1234)), again(params(1.0, 0, 1, 0, 1234));
    std::vector<int32_t> draws;
    for (int s = 0; s < 8; ++s) draws.push_back(again.pick(rows[(size_t) (2 + s) % rows.size()]));
    const std::vector<int32_t> got = run(&sampled);
    expect(got == draws && got != want, "a sampled request: the sampler's draws (" + ids_text(got) + "), not the argmax");
}

}  // namespace

int main() {
    test_masks();
    test_draws();
    test_seeds();
    test_lazy_sort();
    test_session();
    std::printf("dsv41_engine_sampler_test: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
