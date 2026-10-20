// tests/dsv41/engine/qdequant_test.cu - dsv41_engine_qdequant_test (GPU free; verify.sh runs it): the grouped
// GEMM's table decoder (qdequant.cuh decode8_f16x2, compiled for the host with its prmt emulation) against
// qmatmul_ref.cpp's qref_dequant_row rounded to f16, bit for bit, on random q2_K and q3_K blocks at every 8-value
// position of a block. The blocks' f16 scales are random finite values of both signs, subnormals and zeros included;
// the other bytes uniform. The device's prmt is the GPU tests' to prove (experts_test --gemm on a dev pod: the
// in-kernel decode against the dequant path).
#include "../../../src/dsv41/engine/qdequant.cuh"
#include "strata/dsv41/engine/qmatmul_ref.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace e = strata::dsv41::engine;
namespace kn = strata::kernels;

namespace {

// a finite f16: random bits with the exponent kept below 31; every tenth a zero or a subnormal
uint16_t random_f16(std::mt19937& rng) {
    const uint32_t b = rng();
    uint16_t h = (uint16_t) (b & 0xFFFF);
    if ((b >> 16) % 10 == 0) return (uint16_t) (h & 0x83FF);   // zero or subnormal, either sign
    if (((h >> 10) & 0x1F) == 0x1F) h = (uint16_t) (h & ~0x4000);
    return h;
}

template <int TYPE>
int64_t check(int blocks, uint32_t seed) {
    constexpr int BB = e::qdequant::block_bytes<TYPE>();
    std::mt19937 rng(seed);
    std::vector<uint8_t> bytes((size_t) blocks * BB);
    for (auto& b : bytes) b = (uint8_t) rng();
    for (int k = 0; k < blocks; ++k) {
        uint8_t* blk = bytes.data() + (size_t) k * BB;
        const int at = TYPE == e::kQTypeQ2K ? 80 : 108;
        for (int f = 0; f < (TYPE == e::kQTypeQ2K ? 2 : 1); ++f) {
            const uint16_t h = random_f16(rng);
            std::memcpy(blk + at + 2 * f, &h, 2);
        }
    }
    // one row of `blocks` blocks: the reference's values in f32, then f16
    const e::QWeightView w{TYPE, (int64_t) blocks * 256, 1, bytes.data()};
    std::vector<float> ref((size_t) blocks * 256);
    e::qref_dequant_row(w, 0, ref.data());
    int64_t bad = 0;
    for (int k = 0; k < blocks; ++k)
        for (int v0 = 0; v0 < 256; v0 += 8) {
            const auto raw = e::qdequant::fetch8<TYPE>(bytes.data() + (size_t) k * BB, v0);
            uint32_t got[4];
            e::qdequant::decode8_f16x2<TYPE>(raw, v0, got);
            for (int i = 0; i < 8; ++i) {
                const uint16_t g = (uint16_t) (got[i / 2] >> (16 * (i % 2)));
                const uint16_t want = kn::f16_from_f32(ref[(size_t) k * 256 + v0 + i]);
                if (g != want && bad++ < 8)
                    std::printf("  %s block %d value %d: 0x%04x, the reference 0x%04x (%g)\n", e::qtype_name(TYPE), k,
                                v0 + i, g, want, (double) ref[(size_t) k * 256 + v0 + i]);
            }
        }
    std::printf("%s %s: %d random blocks, %d values: %lld differ\n", bad ? "FAIL" : "ok  ", e::qtype_name(TYPE), blocks,
                blocks * 256, (long long) bad);
    return bad;
}

}  // namespace

int main() {
    const int64_t bad = check<e::kQTypeQ2K>(4096, 11) + check<e::kQTypeQ3K>(4096, 12);
    std::printf("dsv41_engine_qdequant_test: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
