// tests/dsv41/engine/qmatmul_test.cpp - parity of the quantized-weight matmul (include/strata/dsv41/engine/
// qmatmul.hpp, the reference qmatmul_ref.hpp).
//
//   dsv41_engine_qmatmul_test --cpu GGUF [GGUF ...]   (a) GPU free: the reference on hand-quantized blocks and on the
//                                                     tiny GGUFs' matrices against the fork's CPU (ggml-cpu)
//   dsv41_engine_qmatmul_test --gpu [--quick]         (b) the GPU against the reference, qmatmul_test_gpu.cu
//
// (a), for q2_K, q3_K and q6_K and each activation layout the type is read in (q8_1, and MMQ's D2S6 for q2_K or D4):
//   hand-quantized: blocks whose value is known by hand (every weight 1.5, -0.5, 0.5, -0.5) and blocks of random
//     bytes with power-of-two d and dmin, against integer activations with a +-127 in every 32 values, so every
//     scale is 1 and every path is exact: the reference equals the literal value (or the exact dot of the dequantized
//     weights), bit for bit, and so does the fork's own dot (ggml-cpu's Q8_K vec_dot);
//   the tiny GGUFs (tools/dsv41/engine/tiny_dsv41.py, every 2D q2_K / q3_K / q6_K matrix): qref_dequant_row equals
//     ggml's to_float bit for bit on every row; on four rows of N(0, 1) activations the reference against the fork's
//     CPU result (each row quantized to Q8_K, ggml's vec_dot): in q8_1 and D4 |y - y_fork| <= sum_k |w_k| |x^_k -
//     x^fork_k| + 2^-16 sum_k |w_k x^fork_k| (the two activation roundings, then the fork's f32 sums), and in every
//     layout the rel L2 over the matrix <= 2^-5 (both are int8 activations: about 1% each on N(0, 1) inputs, while a
//     wrong scale, sign or layout is O(1)).
#include "qmatmul_synth.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/dsv41/engine/qmatmul_ref.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

int run_gpu(bool quick);   // qmatmul_test_gpu.cu

namespace {

namespace e = strata::dsv41::engine;

/// block_q8_K (ggml-common.h), the fork's CPU activation block
struct BlockQ8K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(BlockQ8K) == 292, "block_q8_K is 292 bytes");

std::vector<e::QActLayout> layouts_of(int type) {
    return {e::QActLayout::Q8_1, type == e::kQTypeQ2K ? e::QActLayout::MmqD2S6 : e::QActLayout::MmqD4};
}

// the fork's CPU product of one row of w with each row of x: quantize_row_q8_K then the type's vec_dot (AVX2 build)
struct ForkCpu {
    std::vector<std::vector<BlockQ8K>> xq;   // per row of x
    void quantize(const float* x, int64_t T, int64_t k) {
        // the vec_dot_type of all three K-quants
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        xq.assign((size_t) T, std::vector<BlockQ8K>((size_t) (k / 256)));
        for (int64_t t = 0; t < T; ++t) tr->from_float(x + t * k, xq[(size_t) t].data(), k);
    }
    float dot(const e::QWeightView& w, int64_t r, int64_t t) const {
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) w.type);
        const size_t row = (size_t) (w.k / 256) * e::qtype_block_bytes(w.type);
        float s = 0.f;
        tr->vec_dot((int) w.k, &s, 0, static_cast<const uint8_t*>(w.data) + (size_t) r * row, 0, xq[(size_t) t].data(), 0, 1);
        return s;
    }
    float act(int64_t t, int64_t i) const {
        const BlockQ8K& b = xq[(size_t) t][(size_t) (i / 256)];
        return b.d * (float) b.qs[i % 256];
    }
};

struct Tally {
    int64_t checks = 0, failed = 0;
    void expect(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failed;
            if (failed <= 20) std::printf("  FAIL %s\n", what.c_str());
        }
    }
};

void put16(uint8_t* p, float v) {
    const uint16_t h = strata::kernels::f16_from_f32(v);
    std::memcpy(p, &h, 2);
}

// a block whose every weight is the same known value
std::vector<uint8_t> uniform_block(int type, float& value) {
    std::vector<uint8_t> b(e::qtype_block_bytes(type), 0);
    if (type == e::kQTypeQ2K) {
        // scale 1, min 0, q = 3, d = 0.5: 1.5
        std::fill(b.begin(), b.begin() + 16, 0x01);
        std::fill(b.begin() + 16, b.begin() + 80, 0xFF);
        put16(b.data() + 80, 0.5f);
        put16(b.data() + 82, 0.25f);
        value = 1.5f;
    } else if (type == e::kQTypeQ3K) {
        // hmask set (no -4), every 2-bit q = 1, every 6-bit scale 33 (sc - 32 = 1: low nibbles 1, high pairs 2), d 0.5
        std::fill(b.begin(), b.begin() + 32, 0xFF);
        std::fill(b.begin() + 32, b.begin() + 96, 0x55);
        std::fill(b.begin() + 96, b.begin() + 104, 0x11);
        std::fill(b.begin() + 104, b.begin() + 108, 0xAA);
        put16(b.data() + 108, 0.5f);
        value = 0.5f;
    } else {
        // ql = qh = 0 (q - 32 = -32), int8 scales 1, d = 1/64: -0.5
        std::fill(b.begin() + 192, b.begin() + 208, 0x01);
        put16(b.data() + 208, 1.0f / 64);
        value = -0.5f;
    }
    return b;
}

// q2_K scale 3 min 2 with q = 0, d 0.5, dmin 0.25: every weight -0.5 (the min term alone)
std::vector<uint8_t> min_only_block(float& value) {
    std::vector<uint8_t> b(84, 0);
    std::fill(b.begin(), b.begin() + 16, 0x23);
    put16(b.data() + 80, 0.5f);
    put16(b.data() + 82, 0.25f);
    value = -0.5f;
    return b;
}

// activations with +127 or -127 first in every 32 values (the first of a block of 256 +127) and -1, 0 or 1
// elsewhere: every quantizer's scale is 1 (the fork's Q8_K -1), x^ = x, and the fork's f32 sums stay exact
std::vector<float> integer_acts(int64_t T, int64_t k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> u(-1, 1);
    std::vector<float> x((size_t) (T * k));
    for (int64_t i = 0; i < T * k; ++i) x[(size_t) i] = i % 32 ? (float) u(rng) : (i / 32) % 2 ? -127.f : 127.f;
    return x;
}

void hand_cases(Tally& tally) {
    std::printf("hand-quantized blocks (integer activations, every scale 1):\n");
    ForkCpu fork;
    for (int type : {e::kQTypeQ2K, e::kQTypeQ3K, e::kQTypeQ6K}) {
        // 1. one block of a known value against 127 everywhere: y = value * 127 * 256
        std::vector<std::pair<std::vector<uint8_t>, float>> blocks;
        float v = 0;
        auto ub = uniform_block(type, v);
        blocks.push_back({ub, v});
        if (type == e::kQTypeQ2K) {
            auto mb = min_only_block(v);
            blocks.push_back({mb, v});
        }
        const int64_t k = 256;
        const std::vector<float> x127((size_t) k, 127.f);
        fork.quantize(x127.data(), 1, k);
        for (const auto& [blk, val] : blocks) {
            const e::QWeightView w{type, k, 1, blk.data()};
            const double want = (double) val * 127.0 * 256.0;
            for (e::QActLayout l : layouts_of(type)) {
                e::QActsHost a;
                e::qref_quantize(x127.data(), k, 1, k, l, a);
                float y = 0;
                e::qref_matmul(w, a, 1, &y, 1, nullptr);
                char what[160];
                std::snprintf(what, sizeof what, "%s weights %g, %s: %.9g, want %.9g", e::qtype_name(type), val, e::qact_layout_name(l), y, want);
                tally.expect((double) y == want, what);
                std::printf("  %-5s every weight %-5g %-9s y = %.9g (want %.9g)\n", e::qtype_name(type), val, e::qact_layout_name(l), y, want);
            }
            const float yf = fork.dot(w, 0, 0);
            char what[160];
            std::snprintf(what, sizeof what, "%s weights %g, the fork's Q8_K dot: %.9g, want %.9g", e::qtype_name(type), val, yf, want);
            tally.expect((double) yf == want, what);
            std::printf("  %-5s every weight %-5g %-9s y = %.9g (want %.9g)\n", e::qtype_name(type), val, "fork q8_K", yf, want);
        }
        // 2. random bytes with power-of-two block scales against integer activations: the exact dot of the
        // dequantized weights (every product and sum is exact in double, and in the fork's f32 at 512 values)
        std::mt19937_64 rng(7 + (uint64_t) type);
        const int64_t rows = 16, kk = 512, T = 3;
        std::vector<uint8_t> wb = qsynth::random_blocks(type, rows, kk, 1.f, rng);
        const size_t bb = e::qtype_block_bytes(type);
        for (size_t b = 0; b < wb.size() / bb; ++b) {
            uint8_t* blk = wb.data() + b * bb;
            if (type == e::kQTypeQ2K) { put16(blk + 80, 1.0f / 64); put16(blk + 82, 1.0f / 128); }
            else if (type == e::kQTypeQ3K) put16(blk + 108, 1.0f / 32);
            else put16(blk + 208, 1.0f / 1024);
        }
        const e::QWeightView w{type, kk, rows, wb.data()};
        const std::vector<float> x = integer_acts(T, kk, 3);
        fork.quantize(x.data(), T, kk);
        std::vector<float> deq((size_t) kk);
        int64_t exact = 0, total = 0;
        for (e::QActLayout l : layouts_of(type)) {
            e::QActsHost a;
            e::qref_quantize(x.data(), kk, T, kk, l, a);
            std::vector<float> y((size_t) (T * rows));
            e::qref_matmul(w, a, 1, y.data(), rows, nullptr);
            for (int64_t r = 0; r < rows; ++r) {
                e::qref_dequant_row(w, r, deq.data());
                for (int64_t t = 0; t < T; ++t) {
                    double want = 0;
                    for (int64_t i = 0; i < kk; ++i) want += (double) deq[(size_t) i] * x[(size_t) (t * kk + i)];
                    ++total;
                    exact += (double) y[(size_t) (t * rows + r)] == want;
                    if (l == e::QActLayout::Q8_1) {
                        ++total;
                        exact += (double) fork.dot(w, r, t) == want;
                    }
                }
            }
        }
        char what[160];
        std::snprintf(what, sizeof what, "%s random blocks: %lld of %lld outputs exact", e::qtype_name(type), (long long) exact, (long long) total);
        tally.expect(exact == total, what);
        std::printf("  %-5s random blocks (power-of-two d), %lld x %lld, 3 rows, both layouts and the fork: %lld of %lld outputs exact\n",
                    e::qtype_name(type), (long long) rows, (long long) kk, (long long) exact, (long long) total);
    }
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

constexpr double kRelL2Fork = 1.0 / 32;
constexpr double kForkSum = 1.0 / 65536;

void gguf_cases(const std::string& path, Tally& tally) {
    strata::GgufFile g(path);
    std::printf("%s: the 2D q2_K / q3_K / q6_K matrices, four rows of N(0, 1) activations:\n", path.c_str());
    ForkCpu fork;
    struct Worst { double rel = 0, bound = 0; int64_t n = 0; };
    std::vector<std::pair<std::string, Worst>> per;   // per type and layout
    auto worst = [&](const std::string& key) -> Worst& {
        for (auto& p : per)
            if (p.first == key) return p.second;
        per.push_back({key, Worst{}});
        return per.back().second;
    };
    int64_t matrices = 0, rows_checked = 0;
    for (const strata::TensorInfo& t : g.tensors()) {
        if (t.shape.size() != 2 || (t.type != 10 && t.type != 11 && t.type != 14)) continue;
        const int type = (int) t.type;
        const int64_t k = (int64_t) t.shape[0], rows = (int64_t) t.shape[1];
        const e::QWeightView w{type, k, rows, g.tensor_data(t)};
        ++matrices;
        // dequant: ggml's to_float, bit for bit
        const ggml_type_traits* gt = ggml_get_type_traits((ggml_type) type);
        std::vector<float> mine((size_t) k), theirs((size_t) k);
        const size_t row_bytes = (size_t) (k / 256) * e::qtype_block_bytes(type);
        int64_t bad_rows = 0;
        std::vector<std::vector<float>> deq((size_t) rows);
        for (int64_t r = 0; r < rows; ++r) {
            e::qref_dequant_row(w, r, mine.data());
            gt->to_float(static_cast<const uint8_t*>(w.data) + (size_t) r * row_bytes, theirs.data(), k);
            bad_rows += std::memcmp(mine.data(), theirs.data(), sizeof(float) * (size_t) k) != 0;
            deq[(size_t) r] = mine;
        }
        rows_checked += rows;
        tally.expect(bad_rows == 0, t.name + ": " + std::to_string(bad_rows) + " rows dequantize unlike ggml's to_float");
        // the products against the fork's CPU
        const int64_t T = 4;
        std::mt19937 rng(std::hash<std::string>{}(t.name) & 0xffffffffu);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) (T * k));
        for (auto& v : x) v = nd(rng);
        fork.quantize(x.data(), T, k);
        std::vector<float> yf((size_t) (T * rows));
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t tt = 0; tt < T; ++tt) yf[(size_t) (tt * rows + r)] = fork.dot(w, r, tt);
        for (e::QActLayout l : layouts_of(type)) {
            e::QActsHost a;
            e::qref_quantize(x.data(), k, T, k, l, a);
            std::vector<float> y((size_t) (T * rows));
            e::qref_matmul(w, a, 1, y.data(), rows, nullptr);
            const double rl = rel_l2(y, yf);
            Worst& wst = worst(std::string(e::qtype_name(type)) + " " + e::qact_layout_name(l));
            wst.rel = std::max(wst.rel, rl);
            ++wst.n;
            tally.expect(rl <= kRelL2Fork, t.name + " " + e::qact_layout_name(l) + ": rel L2 " + std::to_string(rl) + " against the fork");
            if (l == e::QActLayout::MmqD2S6) continue;
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t tt = 0; tt < T; ++tt) {
                    double bound = 0, scale = 0;
                    for (int64_t i = 0; i < k; ++i) {
                        const double wk = deq[(size_t) r][(size_t) i], xf = fork.act(tt, i);
                        bound += std::fabs(wk) * std::fabs((double) e::qref_act(a, tt, i) - xf);
                        scale += std::fabs(wk * xf);
                    }
                    bound += kForkSum * scale;
                    const double d = std::fabs((double) y[(size_t) (tt * rows + r)] - yf[(size_t) (tt * rows + r)]);
                    wst.bound = std::max(wst.bound, bound > 0 ? d / bound : 0);
                    if (!(d <= bound))
                        tally.expect(false, t.name + " " + e::qact_layout_name(l) + " row " + std::to_string(r) + ": |y - y_fork| " +
                                                std::to_string(d) + " over the rounding bound " + std::to_string(bound));
                    else
                        ++tally.checks;
                }
        }
    }
    std::printf("  %lld matrices, %lld rows dequantized as ggml does\n", (long long) matrices, (long long) rows_checked);
    for (const auto& [key, wst] : per)
        std::printf("  %-14s %3lld matrices: worst rel L2 against the fork %.4f (bound %.4f)%s\n", key.c_str(), (long long) wst.n,
                    wst.rel, kRelL2Fork,
                    key.find("d2s6") != std::string::npos ? "" : (", worst |y - y_fork| " + std::to_string(wst.bound) + " of its rounding bound").c_str());
    if (matrices == 0) tally.expect(false, path + ": no 2D q2_K, q3_K or q6_K matrix");
}

int run_cpu(const std::vector<std::string>& ggufs) {
    ggml_cpu_init();
    Tally tally;
    hand_cases(tally);
    for (const std::string& p : ggufs) gguf_cases(p, tally);
    const bool ok = tally.failed == 0;
    std::printf("dsv41_engine_qmatmul_test (a): %s (%lld checks, %lld failed)\n", ok ? "PASS" : "FAIL", (long long) tally.checks,
                (long long) tally.failed);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--cpu") == 0) return run_cpu(std::vector<std::string>(argv + 2, argv + argc));
    if (argc >= 2 && std::strcmp(argv[1], "--gpu") == 0) return run_gpu(argc >= 3 && std::strcmp(argv[2], "--quick") == 0);
    std::fprintf(stderr, "usage: dsv41_engine_qmatmul_test --cpu GGUF [GGUF ...] | --gpu [--quick]\n");
    return 2;
}
