// tests/dsv41/engine/expert_bits_test.cpp - the routed experts' bits do not depend on the tier that computes them.
//
//   dsv41_engine_expert_bits_test            (the dev VM, GPU free: checks 1 to 5, then the CPU speed table)
//   dsv41_engine_expert_bits_test --gpu      (a dev pod: expert_bits_test_gpu.cu, the CUDA kernels against the CPU)
//
// The reference is mmvq_ref.hpp: plain C++ written to mirror the CUDA kernels line for line (quantize_q8_1_kernel,
// mmvq.cuh's traits, mmvq_kernel's order, qmatmul_experts). The CPU experts the engine runs (CpuExperts in
// CpuNumerics::Gpu, experts_cpu_mmvq.cpp's AVX2) must give its values bit for bit:
//   1. exp and the clamp: expert_act.hpp's expf_det within 2 ulp of double's exp on a sweep of [-87.3, 88.7] (every
//      61st float), its edges (NaN, +-inf, past both ends); the AVX2 clamp (mmvq_cpu::swiglu) the scalar one's bits
//      on every 257th float as the gate (NaN, inf, subnormals, both sides of the limit) and random ups;
//   2. q8_1: mmvq_cpu::quantize the reference's bytes on random rows at scales 1e-3 to 1e6 and on the edge blocks (all
//      zero, a value whose block scale passes f16's range, an inf, a NaN, subnormals, exact halves); and on the --gpu
//      test's own inputs (every K the experts quantize: 5120, gate/up's input, 2304, down's, and 256) both against
//      qref_quantize too, the quantizer qmatmul_test --gpu holds to the device's bytes (an earlier (g1)
//      failure was not the CPU side's, this is the CPU half of that proof);
//   3. one matvec: mmvq_cpu::rows the reference's mmvq_kernel value for every row of random q2_K [2304][5120] and q3_K
//      [5120][2304] matrices (the real gate/up and down) and small ones, against 1 to 8 activations at once (MMVQ's
//      column limit);
//   4. the FFN: CpuExperts (GPU numerics, 1 and 3 threads) the reference's qmatmul_experts value for every pair of 5
//      tokens routed to 3 of 8 random real-size experts each (gate, up, the clamp, h quantized, down), the experts
//      several tokens route to read once (decode_miss_jobs' grouping);
//   5. the weighted sum: each token's slot-order combine (ref_expert_combine, the device's combine_kernel order) the
//      same bits under 16 random placements (each pair a hit, the reference, or a miss, the CPU experts); with the
//      fork's CPU numerics for the misses (CpuNumerics::Fork, the earlier engine) the same placements differ (the
//      check sees what it guards against).
// Then the speed of both CPU numerics on this host (one real-size expert, 1 and 4 tokens, the median of 7 runs): the
// GPU numerics' cost against the fork's, what decode's misses pay for the placement-independent bits.
// PASS or FAIL lines, exit 1 on any FAIL.
#include "qmatmul_synth.hpp"

#include "../../../src/dsv41/engine/expert_act.hpp"
#include "../../../src/dsv41/engine/experts_cpu_mmvq.hpp"
#include "strata/dsv41/engine/experts.hpp"
#include "strata/dsv41/engine/mmvq_ref.hpp"
#include "strata/dsv41/engine/qmatmul_ref.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace e = strata::dsv41::engine;
namespace act = strata::dsv41::engine::act;

int run_gpu();   // expert_bits_test_gpu.cu

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

uint32_t fbits(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    return b;
}
float from_bits(uint32_t b) {
    float x;
    std::memcpy(&x, &b, 4);
    return x;
}
// equal bits, or both NaN (a NaN's payload is the hardware's: the device's is canonical)
bool same(float a, float b) { return fbits(a) == fbits(b) || (std::isnan(a) && std::isnan(b)); }
int64_t differ(const float* a, const float* b, size_t n) {
    int64_t d = 0;
    for (size_t i = 0; i < n; ++i) d += !same(a[i], b[i]);
    return d;
}

constexpr int64_t kE = 5120, kF = 2304;   // the real n_embd and expert width
constexpr float kLimit = 10.f;            // swiglu_clamp_exp of the real model

std::vector<float> normal(size_t n, float scale, std::mt19937& g) {
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(g);
    return v;
}

// 1
void exp_and_clamp() {
    // the error of expf_det in ulps of the float nearest the exact value (double's exp is within 1 ulp of a double)
    double worst = 0;
    float worst_x = 0;
    int64_t n = 0;
    const uint32_t lo_neg = fbits(-87.33f), hi_pos = fbits(88.72f);
    auto probe = [&](float x) {
        const double ex = std::exp((double) x);
        const float got = act::expf_det(x);
        const float near = (float) ex;
        const double ulp = (double) std::nextafter(near, std::numeric_limits<float>::infinity()) - (double) near;
        const double err = std::fabs((double) got - ex) / ulp;
        if (err > worst) {
            worst = err;
            worst_x = x;
        }
        ++n;
    };
    for (uint32_t b = 0x80000000u; b <= lo_neg; b += 61) probe(from_bits(b));   // -0 .. -87.33
    for (uint32_t b = 0; b <= hi_pos; b += 61) probe(from_bits(b));             // 0 .. 88.72
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const bool edges = std::isnan(act::expf_det(nan)) && act::expf_det(inf) == inf && act::expf_det(-inf) == 0.f &&
                       act::expf_det(88.73f) == inf && act::expf_det(-87.4f) == 0.f && act::expf_det(0.f) == 1.f &&
                       std::isfinite(act::expf_det(88.72f)) && act::expf_det(-87.33f) > 0.f;
    char what[320];
    std::snprintf(what, sizeof what, "expf_det on %lld inputs of [-87.33, 88.72] (every 61st float): at most %.3f ulp from "
                  "the exact exp (at x = %.9g); NaN, +-inf, 88.73 -> inf, -87.4 -> 0, exp(0) = 1 %s", (long long) n, worst,
                  (double) worst_x, edges ? "as stated" : "WRONG");
    expect(worst <= 2.0 && edges, what);

    // the AVX2 clamp against the scalar one
    std::mt19937 g(5u);
    std::uniform_real_distribution<float> uu(-30.f, 30.f);
    const int64_t chunk = 4096;
    std::vector<float> gate((size_t) chunk), up((size_t) chunk), hv((size_t) chunk);
    int64_t total = 0, bad = 0, nans = 0;
    uint64_t b = 0;
    while (b <= 0xffffffffull) {
        int64_t k = 0;
        for (; k < chunk && b <= 0xffffffffull; ++k, b += 257) {
            gate[(size_t) k] = from_bits((uint32_t) b);
            up[(size_t) k] = (k % 17 == 0) ? (k % 2 ? inf : nan) : uu(g);
        }
        e::mmvq_cpu::swiglu(gate.data(), up.data(), k, kLimit, hv.data());
        for (int64_t i = 0; i < k; ++i) {
            const float want = act::swiglu_clamp(gate[(size_t) i], up[(size_t) i], kLimit);
            bad += !same(hv[(size_t) i], want);
            nans += std::isnan(want);
        }
        total += k;
    }
    std::snprintf(what, sizeof what, "the AVX2 clamp (mmvq_cpu::swiglu) on %lld gates (every 257th float, %lld NaN "
                  "results) and random ups: %lld differ from expert_act.hpp's scalar clamp", (long long) total,
                  (long long) nans, (long long) bad);
    expect(bad == 0 && total > 16000000, what);
}

// 2
void quantizer() {
    std::mt19937 g(11u);
    int64_t blocks = 0, bad = 0;
    auto run = [&](const std::vector<float>& x) {
        const int64_t k = (int64_t) x.size();
        std::vector<uint8_t> a((size_t) (k / 32) * e::kQ81BlockBytes), b(a.size());
        e::mmvq_ref_quantize(x.data(), k, a.data());
        e::mmvq_cpu::quantize(x.data(), k, b.data());
        for (int64_t q = 0; q < k / 32; ++q)
            bad += std::memcmp(a.data() + q * e::kQ81BlockBytes, b.data() + q * e::kQ81BlockBytes, e::kQ81BlockBytes) != 0;
        blocks += k / 32;
    };
    for (float scale : {1e-3f, 0.1f, 1.f, 30.f, 1e3f, 1e6f})
        for (int r = 0; r < 8; ++r) run(normal((size_t) kE, scale, g));
    // the edges, one block each
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> edge(32 * 8, 0.f);
    for (int l = 0; l < 32; ++l) {
        edge[32 + l] = (float) (l - 16) * 0.37f;                               // ordinary
        edge[64 + l] = l == 3 ? 9.0e6f : (float) l;                            // amax / 127 past f16's 65504
        edge[96 + l] = l == 5 ? inf : (float) l;                               // an inf
        edge[128 + l] = l == 7 ? nan : (float) l;                              // a NaN
        edge[160 + l] = (float) l * 1e-45f;                                    // subnormals (the scale underflows)
        edge[192 + l] = l == 0 ? 127.f : (float) (l % 5) + 0.5f * (l % 2);     // d = 1: exact halves round away from zero
        edge[224 + l] = l == 1 ? -127.f : -((float) (l % 7) + 0.5f);
    }
    run(edge);
    char what[200];
    std::snprintf(what, sizeof what, "q8_1 of %lld blocks (random rows at scales 1e-3 to 1e6, 8 edge blocks): %lld differ "
                  "from the reference's bytes", (long long) blocks, (long long) bad);
    expect(bad == 0, what);
}

// 2, the --gpu test's inputs (expert_bits_test_gpu.cu: its normal() of the same seeds), T rows of k values each
std::vector<float> gpu_test_input(size_t n, uint32_t seed) {
    std::mt19937 g(seed);
    return normal(n, 1.f, g);
}
void quantizer_gpu_inputs() {
    int64_t blocks = 0, bad_ref = 0, bad_qref = 0;
    auto run = [&](const std::vector<float>& x, int64_t T, int64_t k) {
        const size_t row_q = (size_t) (k / 32) * e::kQ81BlockBytes;
        std::vector<uint8_t> cpu((size_t) T * row_q), ref(cpu.size());
        for (int64_t t = 0; t < T; ++t) {
            e::mmvq_cpu::quantize(x.data() + t * k, k, cpu.data() + (size_t) t * row_q);
            e::mmvq_ref_quantize(x.data() + t * k, k, ref.data() + (size_t) t * row_q);
        }
        e::QActsHost got, want;
        e::qref_unpack(cpu.data(), e::QActLayout::Q8_1, T, k, got);
        e::qref_quantize(x.data(), k, T, k, e::QActLayout::Q8_1, want);
        for (int64_t b = 0; b < T * k / 32; ++b) {
            bad_ref += std::memcmp(cpu.data() + (size_t) b * e::kQ81BlockBytes, ref.data() + (size_t) b * e::kQ81BlockBytes,
                                   e::kQ81BlockBytes) != 0;
            bad_qref += std::memcmp(&got.d[(size_t) b], &want.d[(size_t) b], 4) != 0 ||
                        std::memcmp(&got.s[(size_t) b], &want.s[(size_t) b], 4) != 0 ||
                        std::memcmp(got.q.data() + b * 32, want.q.data() + b * 32, 32) != 0;
        }
        blocks += T * k / 32;
    };
    // (g1): gate/up's input 5120 (2304 rows), down's 2304 (5120 rows), 256 (64 rows), at 1, 2, 4 and 8 activations
    const int64_t g1[3][2] = {{kF, kE}, {kE, kF}, {64, 256}};
    for (const auto& rk : g1)
        for (int64_t T : {1, 2, 4, 8}) run(gpu_test_input((size_t) (T * rk[1]), (uint32_t) (T * 7 + rk[0])), T, rk[1]);
    // (g2): the tokens' x [T][5120]; (g0): 32 trials of [8][5120]
    for (int64_t T : {1, 2, 4, 8}) run(gpu_test_input((size_t) (T * kE), (uint32_t) (100 + T)), T, kE);
    for (int i = 0; i < 32; ++i) run(gpu_test_input((size_t) (8 * kE), (uint32_t) (1000 + i)), 8, kE);
    char what[320];
    std::snprintf(what, sizeof what, "q8_1 on the --gpu test's inputs (K 5120, 2304 and 256 at 1 to 8 rows, (g2)'s and (g0)'s "
                  "[T][5120], %lld blocks): mmvq_cpu::quantize %lld differ from mmvq_ref_quantize, %lld from qref_quantize",
                  (long long) blocks, (long long) bad_ref, (long long) bad_qref);
    expect(bad_ref == 0 && bad_qref == 0, what);
}

// 3
void matvec() {
    std::mt19937_64 rng(13u);
    std::mt19937 g(17u);
    struct Shape {
        const char* name;
        int type;
        int64_t rows, k;
    };
    const Shape shapes[] = {{"gate/up q2_K [2304][5120]", e::kQTypeQ2K, kF, kE},
                            {"down q3_K [5120][2304]", e::kQTypeQ3K, kE, kF},
                            {"q2_K [64][256]", e::kQTypeQ2K, 64, 256},
                            {"q3_K [96][768]", e::kQTypeQ3K, 96, 768}};
    for (const Shape& sh : shapes) {
        const std::vector<uint8_t> w = qsynth::random_blocks(sh.type, sh.rows, sh.k, 1.f / std::sqrt((float) sh.k), rng);
        const size_t row_bytes = w.size() / (size_t) sh.rows;
        int64_t bad = 0, values = 0;
        for (int n = 1; n <= (int) e::kMmvqMaxCols; ++n) {
            std::vector<std::vector<uint8_t>> q((size_t) n);
            std::vector<std::vector<float>> f((size_t) n, std::vector<float>((size_t) e::mmvq_cpu::act_floats(sh.k)));
            std::vector<e::mmvq_cpu::Act> acts;
            std::vector<std::vector<float>> out((size_t) n, std::vector<float>((size_t) sh.rows));
            std::vector<float*> outs;
            for (int t = 0; t < n; ++t) {
                const std::vector<float> x = normal((size_t) sh.k, 1.f + (float) t, g);
                q[(size_t) t].resize((size_t) (sh.k / 32) * e::kQ81BlockBytes);
                e::mmvq_cpu::quantize(x.data(), sh.k, q[(size_t) t].data());
                acts.push_back(e::mmvq_cpu::prepare_act(q[(size_t) t].data(), sh.k, f[(size_t) t].data()));
                outs.push_back(out[(size_t) t].data());
            }
            e::mmvq_cpu::rows(sh.type, w.data(), row_bytes, 0, sh.rows, sh.k, acts.data(), outs.data(), n);
            // the reference on every row when the matrix is small, on a stride of them at the real sizes (its scalar
            // threads take ~10 us a row)
            const int64_t stride = sh.rows > 1024 ? 7 : 1;
            for (int t = 0; t < n; ++t)
                for (int64_t r = t % stride; r < sh.rows; r += stride) {
                    bad += !same(out[(size_t) t][(size_t) r], e::mmvq_ref_dot(sh.type, w.data() + (size_t) r * row_bytes,
                                                                               q[(size_t) t].data(), sh.k));
                    ++values;
                }
        }
        char what[200];
        std::snprintf(what, sizeof what, "matvec %s, 1 to 8 activations at once: %lld values against the reference, %lld "
                      "differ", sh.name, (long long) values, (long long) bad);
        expect(bad == 0, what);
    }
}

// a synthetic routed layer of n experts at the real dims (q2_K gate and up, q3_K down)
struct Layer {
    std::vector<uint8_t> part[3];
    e::ExpertLayerHost L;
    e::DenseDims d;
};
void make_layer(Layer& l, int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int types[3] = {e::kQTypeQ2K, e::kQTypeQ2K, e::kQTypeQ3K};
    for (int p = 0; p < 3; ++p) {
        const int64_t rows = p == 2 ? kE : kF, k = p == 2 ? kF : kE;
        l.part[p] = qsynth::random_blocks(types[p], rows * n, k, p == 2 ? 1.f / 48.f : 1.f / 72.f, rng);
        l.L.type[p] = types[p];
        l.L.data[p] = l.part[p].data();
        l.L.bytes[p] = l.part[p].size() / (size_t) n;
        l.L.name[p] = p == 0 ? "up" : p == 1 ? "gate" : "down";
    }
    l.L.layer = 0;
    l.L.limit = kLimit;
    l.d.n_embd = (int) kE;
    l.d.n_ff = (int) kF;
    l.d.n_expert = n;
}

// every pair's row through the CPU experts, grouped as decode_miss_jobs groups a call's misses
std::vector<float> cpu_pairs(e::CpuExperts& cpu, const Layer& l, const std::vector<float>& x, int64_t T, int K,
                             const std::vector<int32_t>& ids) {
    const int64_t P = T * K;
    std::vector<int32_t> slot((size_t) l.d.n_expert, -1), job_of((size_t) l.d.n_expert, -1), job_expert, miss((size_t) P);
    std::vector<e::CpuExpertJob> jobs((size_t) l.d.n_expert);
    int64_t hits = 0;
    const size_t n = e::decode_miss_jobs(ids.data(), P, K, slot.data(), job_of.data(), job_expert, jobs.data(), miss.data(), hits);
    int64_t rows = 0;
    for (size_t j = 0; j < n; ++j) {
        for (int p = 0; p < 3; ++p) jobs[j].rows.part[p] = l.L.data[p] + (size_t) job_expert[j] * l.L.bytes[p];
        rows += (int64_t) jobs[j].tokens.size();
    }
    std::vector<float> yc((size_t) (rows * kE)), y((size_t) (P * kE));
    cpu.run(l.L, l.d, x.data(), T, jobs.data(), n, yc.data());
    for (int64_t p = 0; p < P; ++p) std::memcpy(y.data() + p * kE, yc.data() + (size_t) miss[(size_t) p] * kE, (size_t) kE * 4);
    return y;
}

// 4 and 5
void ffn_and_sum() {
    constexpr int NE = 8, K = 3;
    constexpr int64_t T = 5;
    Layer l;
    make_layer(l, NE, 23u);
    std::mt19937 g(29u);
    const std::vector<float> x = normal((size_t) (T * kE), 1.f, g);
    std::vector<int32_t> ids;
    for (int64_t t = 0; t < T; ++t) {
        std::vector<int32_t> all(NE);
        for (int i = 0; i < NE; ++i) all[(size_t) i] = i;
        std::shuffle(all.begin(), all.end(), g);
        ids.insert(ids.end(), all.begin(), all.begin() + K);
    }
    const int64_t P = T * K;
    std::vector<float> ref((size_t) (P * kE));
    for (int64_t p = 0; p < P; ++p) e::mmvq_ref_expert(l.L, l.d, ids[(size_t) p], x.data() + (p / K) * kE, ref.data() + p * kE);
    int shared_experts = 0;
    for (int ex = 0; ex < NE; ++ex) shared_experts += std::count(ids.begin(), ids.end(), ex) > 1;
    std::vector<float> fork;
    for (int threads : {1, 3}) {
        e::CpuExperts cpu(threads);
        const std::vector<float> y = cpu_pairs(cpu, l, x, T, K, ids);
        char what[240];
        std::snprintf(what, sizeof what, "the FFN, CpuExperts in the GPU numerics on %d thread%s: %lld pairs of %lld tokens "
                      "(%d experts routed by several), %lld values differ from the reference's qmatmul_experts", threads,
                      threads > 1 ? "s" : "", (long long) P, (long long) T, shared_experts,
                      (long long) differ(y.data(), ref.data(), y.size()));
        expect(differ(y.data(), ref.data(), y.size()) == 0 && shared_experts > 0, what);
    }
    {
        e::CpuExperts cpu(3, e::CpuNumerics::Fork);
        fork = cpu_pairs(cpu, l, x, T, K, ids);
    }
    // 5: the slot-order combine under random placements (hit: the reference's row, miss: the CPU's)
    e::CpuExperts cpu(3);
    const std::vector<float> miss = cpu_pairs(cpu, l, x, T, K, ids);
    std::uniform_real_distribution<float> wd(0.05f, 0.6f);
    std::vector<float> w((size_t) P);
    for (float& v : w) v = wd(g);
    std::vector<float> first, y((size_t) (P * kE)), out((size_t) (T * kE));
    int64_t moved = 0, fork_moved = 0;
    for (int pl = 0; pl < 16; ++pl) {
        const uint32_t mask = pl == 0 ? 0u : pl == 1 ? 0xffu : (uint32_t) g();   // all hits, all misses, then random
        for (bool fork_misses : {false, true}) {
            for (int64_t p = 0; p < P; ++p) {
                const bool hit = (mask >> ids[(size_t) p]) & 1u;
                const std::vector<float>& src = hit ? ref : fork_misses ? fork : miss;
                std::memcpy(y.data() + p * kE, src.data() + p * kE, (size_t) kE * 4);
            }
            e::ref_expert_combine(y.data(), w.data(), T, K, (int) kE, nullptr, out.data());
            if (first.empty()) first = out;
            (fork_misses ? fork_moved : moved) += differ(out.data(), first.data(), out.size());
        }
    }
    char what[260];
    std::snprintf(what, sizeof what, "the weighted sum of %lld tokens under 16 placements (all hits, all misses, 14 random): "
                  "%lld values differ from the all-hit sum; with the fork's CPU numerics for the misses %lld differ", (long long) T,
                  (long long) moved, (long long) fork_moved);
    expect(moved == 0 && fork_moved > 0, what);
}

// the speed of both numerics on this host
void speed() {
    Layer l;
    make_layer(l, 1, 31u);
    const int most = (int) std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    std::mt19937 g(37u);
    const double bytes = (double) (l.L.bytes[0] + l.L.bytes[1] + l.L.bytes[2]);
    std::printf("CPU experts' speed on this host (one expert of %.2f MB, the median of 7 runs after one):\n", bytes / 1e6);
    for (int threads : {1, most})
    for (int64_t T : {1, 4}) {
        const std::vector<float> x = normal((size_t) (T * kE), 1.f, g);
        std::vector<e::CpuExpertJob> job(1);
        for (int p = 0; p < 3; ++p) job[0].rows.part[p] = l.L.data[p];
        for (int64_t t = 0; t < T; ++t) job[0].tokens.push_back((int32_t) t);
        std::vector<float> y((size_t) (T * kE));
        double ms[2];
        for (int k = 0; k < 2; ++k) {
            e::CpuExperts cpu(threads, k == 0 ? e::CpuNumerics::Fork : e::CpuNumerics::Gpu);
            std::vector<double> runs;
            for (int r = 0; r < 8; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                cpu.run(l.L, l.d, x.data(), T, job.data(), 1, y.data());
                const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (r) runs.push_back(t);
            }
            std::sort(runs.begin(), runs.end());
            ms[k] = runs[runs.size() / 2];
        }
        std::printf("  %d thread%s, %lld token%s: fork numerics %.3f ms (%.1f GB/s), GPU numerics %.3f ms (%.1f GB/s): %.2fx\n",
                    threads, threads > 1 ? "s" : "", (long long) T, T > 1 ? "s" : "", ms[0], bytes / (ms[0] * 1e6), ms[1],
                    bytes / (ms[1] * 1e6), ms[1] / ms[0]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--gpu") return run_gpu();
    if (argc != 1) {
        std::fprintf(stderr, "usage: dsv41_engine_expert_bits_test          (the dev VM: the CPU against the GPU-order "
                             "reference)\n       dsv41_engine_expert_bits_test --gpu    (a dev pod: the CUDA kernels against "
                             "the CPU)\n");
        return 2;
    }
    {
        Layer l;   // exits naming what this CPU lacks (AVX2, FMA, F16C) before any kernel runs
        make_layer(l, 1, 3u);
        e::cpu_experts_check(l.L, l.d, e::CpuNumerics::Gpu);
        e::cpu_experts_check(l.L, l.d, e::CpuNumerics::Fork);
    }
    exp_and_clamp();
    quantizer();
    quantizer_gpu_inputs();
    matvec();
    ffn_and_sum();
    speed();
    std::printf("dsv41_engine_expert_bits_test: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
