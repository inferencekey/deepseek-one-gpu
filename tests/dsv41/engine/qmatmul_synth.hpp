// tests/dsv41/engine/qmatmul_synth.hpp - random q2_K, q3_K and q6_K matrices for the engine's tests and benches
// (no real model is loaded): the generators of tools/dsv41/engine/tiny_dsv41.py, whose tiny GGUFs the
// fork runs. Every field is uniform random bytes except the block scale, set so the values spread about `std`:
//   q2_K y = d sc q - dmin m, q in 0..3: E[sc q] = 11.25 = 1.5 E[m], so dmin = 1.5 d centres it, spread about 13.9 d
//   q3_K q in -4..3 and its 6-bit scale (-32..31) both about zero mean, spread about 42.4 d
//   q6_K q - 32 in -32..31 times an int8 scale: about 1368 d
#pragma once

#include "strata/dsv41/engine/qmatmul.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace qsynth {

namespace e = strata::dsv41::engine;

/// Bytes of rows x k values of `type` (exits on a type or k the generators do not make).
inline size_t random_blocks_bytes(int type, int64_t rows, int64_t k) {
    const size_t bb = e::qtype_block_bytes(type);
    if (!bb || k % e::kQK) {
        std::fprintf(stderr, "qsynth: type %d with rows of %lld values\n", type, (long long) k);
        std::exit(2);
    }
    return (size_t) (rows * (k / e::kQK)) * bb;
}

/// random_blocks into out (random_blocks_bytes of room), for a buffer the caller owns (dsv41_engine_smoke's pinned
/// experts).
inline void random_blocks_into(uint8_t* out, int type, int64_t rows, int64_t k, float std, std::mt19937_64& rng) {
    const size_t bytes = random_blocks_bytes(type, rows, k), bb = e::qtype_block_bytes(type);
    const int64_t nb = (int64_t) (bytes / bb);
    for (size_t i = 0; i + 8 <= bytes; i += 8) {
        const uint64_t v = rng();
        std::memcpy(out + i, &v, 8);
    }
    for (size_t i = bytes / 8 * 8; i < bytes; ++i) out[i] = (uint8_t) rng();
    auto put16 = [](uint8_t* p, float v) {
        const uint16_t h = strata::kernels::f16_from_f32(v);
        std::memcpy(p, &h, 2);
    };
    for (int64_t b = 0; b < nb; ++b) {
        uint8_t* blk = out + (size_t) b * bb;
        if (type == e::kQTypeQ2K) {
            put16(blk + 80, std / 13.9f);
            put16(blk + 82, 1.5f * std / 13.9f);
        } else if (type == e::kQTypeQ3K) {
            put16(blk + 108, std / 42.4f);
        } else {
            put16(blk + 208, std / 1368.0f);
        }
    }
}

inline std::vector<uint8_t> random_blocks(int type, int64_t rows, int64_t k, float std, std::mt19937_64& rng) {
    std::vector<uint8_t> out(random_blocks_bytes(type, rows, k));
    random_blocks_into(out.data(), type, rows, k, std, rng);
    return out;
}

/// One dense matrix kind of the real model (tools/dsv41/engine/SPEC.md "Tensors"): rows of k values, `groups` as
/// qmatmul's groups (attn_output_a).
struct Shape {
    const char* name;
    int type;
    int64_t k, rows;
    int groups;
};

/// Every dense matrix shape of the real DeepSeek-V4.1-Flash (the routed experts are experts.hpp's).
inline std::vector<Shape> real_shapes() {
    return {
        {"attn_q_a", e::kQTypeQ2K, 5120, 1280, 1},
        {"attn_q_b", e::kQTypeQ2K, 1280, 32768, 1},
        {"attn_kv, attn_compressor_kv / gate", e::kQTypeQ2K, 5120, 512, 1},
        {"attn_output_a (8 groups)", e::kQTypeQ2K, 4096, 8192, 8},
        {"attn_output_b", e::kQTypeQ2K, 8192, 5120, 1},
        {"indexer.attn_k", e::kQTypeQ2K, 512, 128, 1},
        {"indexer.attn_q_b", e::kQTypeQ2K, 1280, 4096, 1},
        {"indexer.proj", e::kQTypeQ2K, 5120, 32, 1},
        {"hc_attn_fn, hc_ffn_fn", e::kQTypeQ2K, 20480, 24, 1},
        {"ffn_gate_shexp, ffn_up_shexp", e::kQTypeQ2K, 5120, 2304, 1},
        {"engram_wkv", e::kQTypeQ2K, 6144, 25600, 1},
        {"engram_q, engram_k", e::kQTypeQ2K, 5120, 4, 1},
        {"ffn_down_shexp", e::kQTypeQ3K, 2304, 5120, 1},
        {"output", e::kQTypeQ6K, 5120, 129280, 1},
    };
}

}  // namespace qsynth
