// tests/dsv41/engine/dense_synth.cuh - a synthetic model for dsv41_engine_dense_test --gpu and
// dsv41_engine_dense_bench (no real model is loaded): one layer's dense weights at given dims, random bytes
// as tiny_dsv41.py makes them (q2_K / q3_K / q6_K blocks with about 1 / sqrt(fan_in) spread through
// qmatmul_synth.hpp, the router's bf16 N(0, 1 / sqrt(E)), norms about 1, hc base about N(0, 0.5), hc scales about 1,
// engram_q and engram_k about 1, the selection bias about N(0, 0.05)), the token embedding and the output head, and
// one engram table in pinned host RAM of the real row width (256 q2_K values, 84 bytes) with a row count the caller
// picks (the real tables are 384M rows, about 30 GiB each) and hash metadata whose
// buckets fill it. The same bytes are on the device and on the host for the CPU reference; without upload (the dry
// run, no CUDA call) the device views point at the host bytes as stand-ins for the host-only weight checks, which
// never read them.
#pragma once

#include "attention_synth.cuh"

#include "strata/dsv41/engine/dense.hpp"
#include "strata/dsv41/engine/dense_ref.hpp"
#include "strata/dsv41/engine/engram.hpp"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace dsynth {

namespace e = strata::dsv41::engine;
using synth::Matrix;
using synth::Vector;

/// tiny-a and tiny-b of tools/dsv41/engine/tiny_dsv41.py (their dense geometry)
inline e::DenseDims dims_tiny_a() {
    e::DenseDims d;
    d.n_embd = 256; d.n_vocab = 512; d.n_expert = 8; d.n_expert_used = 2; d.n_ff = 256; d.engram_heads = 2;
    return d;
}
inline e::DenseDims dims_tiny_b() {
    e::DenseDims d;
    d.n_embd = 512; d.n_vocab = 512; d.n_expert = 32; d.n_expert_used = 6; d.n_ff = 512; d.engram_heads = 2;
    return d;
}

inline bool is_prime(uint64_t p) {
    if (p < 2) return false;
    for (uint64_t q = 2; q * q <= p; ++q)
        if (p % q == 0) return false;
    return true;
}

struct Model {
    e::DenseDims d;
    Matrix hc_fn[2], wkv, sh_gate, sh_up, sh_down, token_embd, output;
    Vector hc_base[2], hc_scale[2], attn_norm, ffn_norm, eng_q, eng_k, bias, output_norm;
    std::vector<uint16_t> gate_inp;   // [n_expert][E] bf16
    uint16_t* gate_inp_dev = nullptr;
    e::EngramHash hash;
    e::EngramTable table;
    // the engine's and the reference's views
    e::HcWeights hc[2];
    e::RefHcWeights rhc[2];
    e::EngramWeights eng;
    e::RefEngramWeights reng;
    e::RouterWeights router;
    e::RefRouterWeights rrouter;
    e::SharedExpertWeights sh;
    e::RefSharedWeights rsh;
};

/// the device view of a vector, or its host bytes as the dry run's stand-in
inline const float* dev_or_host(const Vector& v) { return v.dev ? v.dev : v.host.data(); }

/// table_rows: the engram table's row count (pinned host RAM, alloc only when up)
inline void make_model(Model& m, const e::DenseDims& d, uint32_t seed, int64_t table_rows, bool up) {
    std::mt19937 rng(seed);
    m.d = d;
    const int64_t E = d.n_embd, X = (int64_t) e::kHc * E, EK = (int64_t) d.engram_cols() * d.engram_key;
    for (int s = 0; s < 2; ++s) {
        m.hc_fn[s] = synth::make_matrix(rng, e::kHcMix, X, e::kQTypeQ2K, up);
        m.hc_base[s] = synth::make_vector(rng, e::kHcMix, 0.f, 0.5f, up);
        m.hc_scale[s] = synth::make_vector(rng, 3, 1.f, 0.25f, up);
        m.hc[s] = {m.hc_fn[s].view(), dev_or_host(m.hc_base[s]), dev_or_host(m.hc_scale[s])};
        m.rhc[s] = {m.hc_fn[s].ref(), m.hc_base[s].host.data(), m.hc_scale[s].host.data()};
    }
    m.attn_norm = synth::make_vector(rng, E, 1.f, 0.1f, up);
    m.ffn_norm = synth::make_vector(rng, E, 1.f, 0.1f, up);
    m.output_norm = synth::make_vector(rng, E, 1.f, 0.1f, up);
    m.wkv = synth::make_matrix(rng, E * (e::kHc + 1), EK, e::kQTypeQ2K, up);
    m.eng_q = synth::make_vector(rng, X, 1.f, 0.1f, up);
    m.eng_k = synth::make_vector(rng, X, 1.f, 0.1f, up);
    m.eng = {m.wkv.view(), dev_or_host(m.eng_q), dev_or_host(m.eng_k)};
    m.reng = {m.wkv.ref(), m.eng_q.host.data(), m.eng_k.host.data()};
    m.gate_inp.resize((size_t) (d.n_expert * E));
    {
        std::normal_distribution<float> nd(0.f, 1.f / std::sqrt((float) E));
        for (auto& h : m.gate_inp) {
            const float v = nd(rng);
            uint32_t u;
            std::memcpy(&u, &v, 4);
            h = (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
        }
    }
    if (up) m.gate_inp_dev = synth::upload(m.gate_inp);
    m.bias = synth::make_vector(rng, d.n_expert, 0.f, 0.05f, up);
    m.router = {m.gate_inp_dev ? m.gate_inp_dev : m.gate_inp.data(), dev_or_host(m.bias)};
    m.rrouter = {e::RefMatrix{e::kRefTypeBF16, E, d.n_expert, m.gate_inp.data(), e::RefActs::Auto}, m.bias.host.data()};
    m.sh_gate = synth::make_matrix(rng, d.n_ff, E, e::kQTypeQ2K, up);
    m.sh_up = synth::make_matrix(rng, d.n_ff, E, e::kQTypeQ2K, up);
    m.sh_down = synth::make_matrix(rng, E, d.n_ff, e::kQTypeQ3K, up);
    m.sh = {m.sh_gate.view(), m.sh_up.view(), m.sh_down.view()};
    m.rsh = {m.sh_gate.ref(), m.sh_up.ref(), m.sh_down.ref()};
    // the embedding rows spread about 1 (tiny_dsv41.py), the head about 1 / sqrt(E)
    {
        m.token_embd.type = e::kQTypeQ2K;
        m.token_embd.k = E;
        m.token_embd.rows = d.n_vocab;
        std::mt19937_64 r64(rng());
        m.token_embd.host = qsynth::random_blocks(e::kQTypeQ2K, d.n_vocab, E, 1.f, r64);
        if (up) m.token_embd.dev = synth::upload(m.token_embd.host);
    }
    m.output = synth::make_matrix(rng, d.n_vocab, E, e::kQTypeQ6K, up);
    // the engram hash: one engram layer, primes just below table_rows / cols each, offsets cumulative
    e::EngramHash& h = m.hash;
    h.n_gram = d.engram_ngram;
    h.heads = d.engram_heads;
    h.pad = 2;
    h.layer_ids = {0};
    const int C = d.engram_cols();
    uint64_t acc = 0;
    for (int b = 0; b < C; ++b) {
        uint64_t p = (uint64_t) (table_rows / C) - (uint64_t) b;
        while (!is_prime(p)) --p;
        h.primes.push_back(p);
        h.offsets.push_back(acc);
        acc += p;
    }
    std::uniform_int_distribution<int32_t> ids(0, (int32_t) (d.n_vocab * 3 / 4));
    h.token_map.resize((size_t) d.n_vocab);
    for (auto& t : h.token_map) t = ids(rng);
    for (int i = 0; i < h.n_gram; ++i) h.multipliers.push_back(((uint64_t) rng() << 20 | rng()) | 1u);
    h.check();
    if (up) {
        std::string err;
        if (!m.table.alloc(table_rows, 84, err)) synth::ck(cudaErrorMemoryAllocation, err.c_str());
        std::mt19937_64 r64(rng());
        const int64_t slab = 1 << 16;
        for (int64_t r0 = 0; r0 < table_rows; r0 += slab) {
            const int64_t n = std::min(slab, table_rows - r0);
            const std::vector<uint8_t> b = qsynth::random_blocks(e::kQTypeQ2K, n, 256, 1.f, r64);
            std::memcpy(m.table.data() + r0 * 84, b.data(), b.size());
        }
    }
}

}  // namespace dsynth
