// src/dsv41/engine/indexer_ref.cpp - the CPU reference of the lightning indexer, see
// include/strata/dsv41/engine/indexer.hpp.
//
// The fork's CPU backend computes idx_score (deepseek41.cpp 558-577) as:
//   mul_mat(idx_k f16, idx_q_rot f32): the query is converted to the key's vec_dot_type, f16 (ggml-cpu.c
//     ggml_compute_forward_mul_mat, from_float = ggml_cpu_fp32_to_fp16, F16C round to nearest even), then
//     - a ubatch of 2 or more tokens: llamafile_sgemm runs tinyBLAS<8, __m256, ...> (sgemm.cpp 3966-3973): per
//       output one __m256 accumulator, _mm256_fmadd_ps over k in steps of 8 (gemm_bloc), then hsum: (hi128 +
//       lo128), + movehl, + movehdup;
//     - a single-token ubatch: llamafile_sgemm refuses n < 2 ("only enable sgemm for prompt processing",
//       sgemm.cpp 3818-3822) and ggml_vec_dot_f16 runs (vec.cpp 345-362, simd-mappings.h AVX F16): four __m256
//       accumulators over steps of 32, reduced (0 + 2, 1 + 3, then 0 + 1), then (lo128 + hi128) and two hadds;
//   relu (ggml_vec_relu_f32: x > 0 ? x : 0), mul by idx_weights (f32), sum_rows over the heads (ggml_vec_sum_f32:
//     a double sum from 0.0, rounded to float), then + the causal mask (0 or -inf).
// The two dot orders are fork_cpu.hpp's (shared with attention_ref.cpp); this file is built with -mfma and
// -ffp-contract=off (CMakeLists.txt) so nothing else is fused.
#include "strata/dsv41/engine/indexer.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "fork_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace strata::dsv41::engine {
namespace {

constexpr int D = kIndexerDim;

void check_cpu() {
    static const bool ok = __builtin_cpu_supports("fma");
    if (!ok) {
        std::fprintf(stderr, "indexer_ref: this CPU has no FMA (the reference reproduces _mm256_fmadd_ps)\n");
        std::exit(1);
    }
}

void check_heads(int n_heads) {
    if (n_heads < 1 || n_heads > kIndexerHeads) {
        std::fprintf(stderr, "indexer_ref: n_heads %d outside 1..%d\n", n_heads, kIndexerHeads);
        std::exit(1);
    }
}

// q rounded to f16 and back, as the mul_mat's conversion leaves it
void round_query(const float* q, int n_heads, float* qh) {
    for (int i = 0; i < n_heads * D; ++i) qh[i] = kernels::f32_from_f16(kernels::f16_from_f32(q[i]));
}

void key_row(const uint16_t* key, float* kf) {
    for (int i = 0; i < D; ++i) kf[i] = kernels::f32_from_f16(key[i]);
}

}  // namespace

void indexer_ref_scores(const float* q, const float* w, int n_heads, const uint16_t* keys, int64_t n_rows,
                        int64_t ubatch_tokens, float* scores) {
    check_cpu();
    check_heads(n_heads);
    std::vector<float> qh((size_t) n_heads * D);
    round_query(q, n_heads, qh.data());
    float kf[D];
    for (int64_t j = 0; j < n_rows; ++j) {
        key_row(keys + j * D, kf);
        double sum = 0.0;
        for (int h = 0; h < n_heads; ++h) {
            const float* qr = qh.data() + (size_t) h * D;
            const float dot = fork_cpu::f16_dot(kf, qr, D, ubatch_tokens);
            const float r = dot > 0.f ? dot : 0.f;
            sum += (double) (r * w[h]);
        }
        scores[j] = (float) sum;
    }
}

void indexer_ref_abs_scale(const float* q, const float* w, int n_heads, const uint16_t* keys, int64_t n_rows,
                           double* scale) {
    check_heads(n_heads);
    std::vector<float> qh((size_t) n_heads * D);
    round_query(q, n_heads, qh.data());
    float kf[D];
    for (int64_t j = 0; j < n_rows; ++j) {
        key_row(keys + j * D, kf);
        double s = 0.0;
        for (int h = 0; h < n_heads; ++h) {
            double a = 0.0;
            for (int i = 0; i < D; ++i) a += std::fabs((double) qh[(size_t) h * D + i] * (double) kf[i]);
            s += std::fabs((double) w[h]) * a;
        }
        scale[j] = s;
    }
}

void indexer_ref_topk(const float* scores, int64_t visible, int top_k, int32_t* ids) {
    if (top_k < 1 || top_k > kIndexerMaxTopK || visible < 0) {
        std::fprintf(stderr, "indexer_ref_topk: top_k %d (1..%d), visible %lld\n", top_k, kIndexerMaxTopK,
                     (long long) visible);
        std::exit(1);
    }
    const int64_t n = std::min<int64_t>(top_k, visible);
    std::vector<int32_t> idx((size_t) visible);
    std::iota(idx.begin(), idx.end(), 0);
    // the strict order of indexer_topk: larger score first, NaN last, equal scores (+0 == -0) by lower index
    auto before = [&](int32_t a, int32_t b) {
        const float x = scores[a], y = scores[b];
        const bool nx = std::isnan(x), ny = std::isnan(y);
        if (nx || ny) return nx == ny ? a < b : ny;   // a number before a NaN
        if (x != y) return x > y;
        return a < b;
    };
    std::nth_element(idx.begin(), idx.begin() + n, idx.end(), before);
    std::sort(idx.begin(), idx.begin() + n);
    for (int64_t i = 0; i < top_k; ++i) ids[i] = i < n ? idx[(size_t) i] : -1;
}

}  // namespace strata::dsv41::engine
