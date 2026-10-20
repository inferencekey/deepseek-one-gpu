// tests/dsv41/engine/attention_synth.cuh - synthetic models and KV histories for dsv41_engine_attention_test --gpu and
// dsv41_engine_attention_bench (no real model is loaded).
//
// A synthetic model is one layer per attention role at given dims (synth_layers: window, Full ratio 2 with a gate,
// Reuse ratio 2, Full ratio 1 without a gate, Reindex ratio 1, Reuse ratio 1, the real model's roles in its order),
// random weights with about 1 / sqrt(fan_in) spread, q2_K as the real model's (qmatmul_synth.hpp) or f16 (the
// cuBLAS test path), the same bytes on the device and on the host for the CPU reference, norms about 1, sinks
// about N(0, 1). Without upload (dsv41_engine_attention_test --gpu --dry-run, no CUDA call) the device views point at
// the host bytes as stand-ins for the host-only weight checks, which never read them. A synthetic history writes the
// cache of a context of n positions directly on the device (counter-based random values, so it costs no host memory):
// every window cell, the rows and index keys of the sources up to the context's blocks, the carried pooling state;
// keys share one direction plus noise so the index scores spread as a trained indexer's (the inputs), rows and
// window cells are about unit values.
#pragma once

#include "strata/dsv41/engine/attention.hpp"
#include "strata/dsv41/engine/attention_ref.hpp"
#include "strata/dsv41/engine/kv_cache.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "qmatmul_synth.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace synth {

namespace e = strata::dsv41::engine;
namespace kn = strata::kernels;

inline void ck(cudaError_t r, const char* w) {
    if (r != cudaSuccess) {
        std::fprintf(stderr, "dsv41 attention synth: %s: %s\n", w, cudaGetErrorString(r));
        std::exit(2);
    }
}

template <typename T> T* dev_alloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, (n ? n : 1) * sizeof(T)), "cudaMalloc");
    return p;
}
// landed before it returns: a pageable cudaMemcpy may return before its DMA does, and a kernel on a non-blocking
// stream (batch_test_gpu.cu's) is not ordered after it (expert_bits_test_gpu.cu (g0))
template <typename T> T* upload(const std::vector<T>& h) {
    T* p = dev_alloc<T>(h.size());
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    ck(cudaDeviceSynchronize(), "upload");
    return p;
}
template <typename T> std::vector<T> download(const T* p, size_t n) {
    std::vector<T> h(n);
    if (n) ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

/// The six roles in the real model's order: 0 window, 1 Full r2 (gate), 2 Reuse r2, 3 Full r1, 4 Reindex r1, 5 Reuse r1.
inline std::vector<e::AttnLayer> synth_layers() {
    return e::attn_layers({0, 2, 2, 1, 1, 1}, {0, e::kKvSource | e::kGate | e::kKeyOwner | e::kIndexSource, 0,
                                               e::kKvSource | e::kKeyOwner | e::kIndexSource, e::kIndexSource, 0});
}

/// tiny-a and tiny-b of tools/dsv41/engine/tiny_dsv41.py, and the real model
inline e::AttnDims dims_tiny_a() {
    e::AttnDims d;
    d.n_embd = 256; d.n_head = 4; d.head_dim = 256; d.q_lora = 256; d.o_groups = 4; d.o_lora = 64; d.n_swa = 16;
    d.idx_heads = 4; d.top_k = 8;
    return d;
}
inline e::AttnDims dims_tiny_b() {
    e::AttnDims d;
    d.n_embd = 512; d.n_head = 8; d.head_dim = 512; d.q_lora = 512; d.o_groups = 8; d.o_lora = 64; d.n_swa = 16;
    d.idx_heads = 8; d.top_k = 16;
    return d;
}

struct Matrix {
    int type = 0;                 // e::kQTypeQ2K or e::kQTypeF16
    std::vector<uint8_t> host;    // [rows][k] as the GGUF stores it: q2_K blocks or f16 bits
    void* dev = nullptr;
    int64_t k = 0, rows = 0;
    /// the reference's view: a q2_K matrix in the engine's numerics (the layout of each call; a test sets it per call)
    e::RefMatrix ref() const {
        return e::RefMatrix{type, k, rows, host.empty() ? nullptr : host.data(),
                            type == e::kQTypeF16 ? e::RefActs::Fork : e::RefActs::Auto};
    }
    e::QWeightView view() const { return e::QWeightView{type, k, rows, dev ? dev : (host.empty() ? nullptr : host.data())}; }
    size_t f16_bytes() const { return (size_t) (k * rows) * 2; }
};
struct Vector {
    std::vector<float> host;
    float* dev = nullptr;
};

inline Matrix make_matrix(std::mt19937& rng, int64_t rows, int64_t k, int type, bool up) {
    Matrix m;
    m.type = type;
    m.k = k;
    m.rows = rows;
    const float std = 1.f / std::sqrt((float) k);
    if (type == e::kQTypeF16) {
        m.host.resize((size_t) (rows * k) * 2);
        std::normal_distribution<float> nd(0.f, std);
        for (int64_t i = 0; i < rows * k; ++i) {
            const uint16_t h = kn::f16_from_f32(nd(rng));
            std::memcpy(m.host.data() + i * 2, &h, 2);
        }
    } else {
        std::mt19937_64 r64(rng());
        m.host = qsynth::random_blocks(type, rows, k, std, r64);
    }
    if (up) m.dev = upload(m.host);
    return m;
}
inline Vector make_vector(std::mt19937& rng, int64_t n, float mean, float std, bool up) {
    Vector v;
    v.host.resize((size_t) n);
    std::normal_distribution<float> nd(mean, std);
    for (auto& x : v.host) x = nd(rng);
    if (up) v.dev = upload(v.host);
    return v;
}

/// The weights of the six roles. The projections every layer has (q, kv, output) are shared by all six layers to
/// save memory at the real dims (each layer still has its own norms and sinks); the role tensors are per layer.
struct Model {
    e::AttnDims d;
    std::vector<e::AttnLayer> layers;
    Matrix wq_a, wq_b, wkv, wo_a, wo_b;
    std::vector<Matrix> comp_kv, comp_gate, idx_k, idx_q_b, idx_proj;
    std::vector<Vector> q_a_norm, kv_norm, comp_norm, idx_k_norm, sinks;
    std::vector<e::AttnLayerWeights> dev;
    std::vector<e::RefLayerWeights> ref;
};

/// type: e::kQTypeQ2K (the real model's) or e::kQTypeF16 (the cuBLAS test path); up = false keeps it on the host
inline void make_model(Model& m, const e::AttnDims& d, uint32_t seed, int type, bool up = true) {
    std::mt19937 rng(seed);
    m.d = d;
    m.layers = synth_layers();
    const int64_t HD = (int64_t) d.n_head * d.head_dim, gd = HD / d.o_groups;
    m.wq_a = make_matrix(rng, d.q_lora, d.n_embd, type, up);
    m.wq_b = make_matrix(rng, HD, d.q_lora, type, up);
    m.wkv = make_matrix(rng, d.head_dim, d.n_embd, type, up);
    m.wo_a = make_matrix(rng, (int64_t) d.o_groups * d.o_lora, gd, type, up);
    m.wo_b = make_matrix(rng, d.n_embd, (int64_t) d.o_groups * d.o_lora, type, up);
    const size_t n = m.layers.size();
    m.comp_kv.resize(n); m.comp_gate.resize(n); m.idx_k.resize(n); m.idx_q_b.resize(n); m.idx_proj.resize(n);
    m.q_a_norm.resize(n); m.kv_norm.resize(n); m.comp_norm.resize(n); m.idx_k_norm.resize(n); m.sinks.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const e::AttnLayer& l = m.layers[i];
        m.q_a_norm[i] = make_vector(rng, d.q_lora, 1.f, 0.1f, up);
        m.kv_norm[i] = make_vector(rng, d.head_dim, 1.f, 0.1f, up);
        m.sinks[i] = make_vector(rng, d.n_head, 0.f, 1.f, up);
        if (l.kv_source) {
            m.comp_kv[i] = make_matrix(rng, d.head_dim, d.n_embd, type, up);
            m.comp_norm[i] = make_vector(rng, d.head_dim, 1.f, 0.1f, up);
        }
        if (l.gate) m.comp_gate[i] = make_matrix(rng, d.head_dim, d.n_embd, type, up);
        if (l.key_owner) {
            m.idx_k[i] = make_matrix(rng, d.idx_dim, d.head_dim, type, up);
            m.idx_k_norm[i] = make_vector(rng, d.idx_dim, 1.f, 0.1f, up);
        }
        if (l.index_source) {
            m.idx_q_b[i] = make_matrix(rng, (int64_t) d.idx_heads * d.idx_dim, d.q_lora, type, up);
            m.idx_proj[i] = make_matrix(rng, d.idx_heads, d.n_embd, type, up);
        }
        e::AttnLayerWeights w;
        e::RefLayerWeights r;
        w.wq_a = m.wq_a.view(); r.wq_a = m.wq_a.ref();
        w.wq_b = m.wq_b.view(); r.wq_b = m.wq_b.ref();
        w.wkv = m.wkv.view(); r.wkv = m.wkv.ref();
        w.wo_a = m.wo_a.view(); r.wo_a = m.wo_a.ref();
        w.wo_b = m.wo_b.view(); r.wo_b = m.wo_b.ref();
        w.comp_kv = m.comp_kv[i].view(); r.comp_kv = m.comp_kv[i].ref();
        w.comp_gate = m.comp_gate[i].view(); r.comp_gate = m.comp_gate[i].ref();
        w.idx_k = m.idx_k[i].view(); r.idx_k = m.idx_k[i].ref();
        w.idx_q_b = m.idx_q_b[i].view(); r.idx_q_b = m.idx_q_b[i].ref();
        w.idx_proj = m.idx_proj[i].view(); r.idx_proj = m.idx_proj[i].ref();
        w.q_a_norm = m.q_a_norm[i].dev; r.q_a_norm = m.q_a_norm[i].host.data();
        w.kv_norm = m.kv_norm[i].dev; r.kv_norm = m.kv_norm[i].host.data();
        w.comp_norm = m.comp_norm[i].dev; r.comp_norm = l.kv_source ? m.comp_norm[i].host.data() : nullptr;
        w.idx_k_norm = m.idx_k_norm[i].dev; r.idx_k_norm = l.key_owner ? m.idx_k_norm[i].host.data() : nullptr;
        w.sinks = m.sinks[i].dev; r.sinks = m.sinks[i].host.data();
        m.dev.push_back(w);
        m.ref.push_back(r);
    }
}

// ---- synthetic histories
__device__ __forceinline__ uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
// uniform with unit variance: [-sqrt(3), sqrt(3))
__device__ __forceinline__ float unit(uint64_t seed, uint64_t i) {
    return ((float) (mix64(seed * 0x2545f4914f6cdd1dull + i) >> 40) * (1.0f / 16777216.0f) * 2.f - 1.f) * 1.7320508f;
}

// rows [r0, r1) of `cols` f16: scale * (dir_mix * a_row * dir_col + noise)
__global__ void fill_rows_kernel(uint16_t* dst, int64_t r0, int64_t r1, int cols, uint64_t seed, float scale, float dir_mix) {
    const int64_t row = r0 + blockIdx.x;
    if (row >= r1) return;
    const float a = unit(seed, 1ull << 62 | (uint64_t) row);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = scale * (dir_mix * a * unit(seed ^ 0x5bd1e995ull, (uint64_t) c) + unit(seed, (uint64_t) row * cols + c));
        dst[row * cols + c] = kn::f16_from_f32(v);
    }
}
__global__ void fill_f32_kernel(float* dst, int64_t n, uint64_t seed, float scale) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = scale * unit(seed, (uint64_t) i);
}

inline void fill_rows(uint16_t* dst, int64_t r0, int64_t r1, int cols, uint64_t seed, float scale, float dir_mix) {
    for (int64_t a = r0; a < r1; a += 65535) {
        const int64_t b = a + 65535 < r1 ? a + 65535 : r1;
        fill_rows_kernel<<<(unsigned) (b - a), 128>>>(dst, a, b, cols, seed, scale, dir_mix);
        ck(cudaGetLastError(), "fill_rows_kernel");
    }
}

/// The cache of every layer as after a context of n positions: window cells, rows and keys of the blocks
/// 0 .. n / r - 1, the pooling state of positions floor(n / r) r .. n - 1.
inline void write_history(e::KvCache& kv, int64_t n, uint64_t seed) {
    const e::AttnDims& d = kv.dims();
    for (const e::AttnLayer& l : kv.layers()) {
        e::KvLayerBuffers& b = kv.layer(l.index);
        const uint64_t s = seed * 131 + (uint64_t) l.index;
        fill_rows(b.window, 0, b.window_cells, d.head_dim, s ^ 1, 1.f, 0.f);
        if (l.kv_source && n / l.ratio > 0) fill_rows(b.rows, 0, n / l.ratio, d.head_dim, s ^ 2, 1.f, 0.f);
        if (l.key_owner && n / l.ratio > 0) fill_rows(b.keys, 0, n / l.ratio, d.idx_dim, s ^ 3, 0.25f, 1.f);
        if (l.kv_source) {
            fill_f32_kernel<<<16, 256>>>(b.carry_kv, (int64_t) l.ratio * d.head_dim, s ^ 4, 0.5f);
            fill_f32_kernel<<<16, 256>>>(b.carry_score, (int64_t) l.ratio * d.head_dim, s ^ 5, 0.5f);
            ck(cudaGetLastError(), "fill_f32_kernel");
        }
        kv.set_position(l.index, n);
    }
    ck(cudaDeviceSynchronize(), "write_history");
}

/// x [T][n_embd] about N(0, 1), the scale of attn_norm's output
inline std::vector<float> make_input(int64_t T, int n_embd, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) (T * n_embd));
    for (auto& v : x) v = nd(rng);
    return x;
}

}  // namespace synth
