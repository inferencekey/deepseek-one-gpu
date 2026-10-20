// src/dsv41/engine/attention_ref.cpp - the CPU reference of one attention layer, see
// include/strata/dsv41/engine/attention_ref.hpp. The fork's arithmetic lives in fork_cpu.hpp (one function per ggml
// routine); this file composes it in the fork's op order (deepseek41.cpp 440-804, deepseek4.cpp 481-534) and holds the
// engine's state rules (window, carried pooling state, rows and keys at their own ratio, reader modes).
#include "strata/dsv41/engine/attention_ref.hpp"
#include "strata/dsv41/engine/indexer.hpp"
#include "strata/dsv41/engine/qmatmul_ref.hpp"

#include "fork_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace strata::dsv41::engine {
namespace {

using fork_cpu::BlockQ2K;
using fork_cpu::BlockQ8K;
using fork_cpu::QK_K;

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fork_cpu::fail("attention_ref", msg);
}

// f16 bits -> f32 by table: the engine numerics read every weight through it
const float* f16_table() {
    static const std::vector<float> t = [] {
        std::vector<float> v(65536);
        for (uint32_t i = 0; i < 65536; ++i) v[i] = kernels::f32_from_f16((uint16_t) i);
        return v;
    }();
    return t.data();
}

void parallel_for(int64_t n, int64_t work, const std::function<void(int64_t)>& f) {
    const int hw = (int) std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    const int nt = work < (int64_t) 1 << 22 ? 1 : (int) std::min<int64_t>(hw, n);
    if (nt <= 1) {
        for (int64_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < nt; ++i)
        th.emplace_back([&] { for (int64_t k; (k = next++) < n;) f(k); });
    for (auto& t : th) t.join();
}

fork_cpu::RopeCfg rope_cfg(const AttnDims& d, bool compressed) {
    const RopeArgs a = attn_rope_args(d, compressed);
    fork_cpu::RopeCfg c;
    c.freq_scale = a.freq_scale;
    c.ext_factor = a.ext_factor;
    c.attn_factor = a.attn_factor;
    c.n_rot = a.n_rot;
    c.theta_scale = a.theta_scale;
    c.corr0 = a.corr0;
    c.corr1 = a.corr1;
    return c;
}

// rope of the last n_rot values of each of `heads` heads of `hd` values at x, at position pos
void rope_heads(const fork_cpu::RopeCfg& c, int64_t pos, float sin_sign, float* x, int heads, int hd) {
    float cache[128];
    if (c.n_rot > 128 || c.n_rot > hd) failf("n_rot %d is above 128 or the head dim %d", c.n_rot, hd);
    fork_cpu::rope_cache(c, pos, sin_sign, cache);
    for (int h = 0; h < heads; ++h) fork_cpu::rope_rotate(cache, c.n_rot, x + (size_t) h * hd + (hd - c.n_rot));
}

void check_matrix(const RefMatrix& w, const char* name) {
    if (!w.data) failf("weight %s is missing", name);
    if (w.type == kRefTypeF16 || w.type == kRefTypeBF16) return;
    if (!qtype_block_bytes(w.type)) failf("weight %s has type %d (q2_K, q3_K, q6_K, f16 or bf16)", name, w.type);
}

// bf16 bits -> f32 (exact), and x rounded to bf16 as the fork's from_float (and the engine's router input) does
std::vector<float> bf16_acts(const float* x, int64_t n) {
    std::vector<float> v((size_t) n);
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = fork_cpu::f32_from_bf16(fork_cpu::bf16_from_f32(x[i]));
    return v;
}
std::vector<float> bf16_weights(const RefMatrix& w) {
    const uint16_t* h = static_cast<const uint16_t*>(w.data);
    std::vector<float> v((size_t) (w.k * w.rows));
    for (size_t i = 0; i < v.size(); ++i) v[i] = fork_cpu::f32_from_bf16(h[i]);
    return v;
}

// a bf16 matrix (the router): the fork's dot (tinyBLAS for 2 or more tokens, ggml_vec_dot_bf16 for one), or the
// engine's exact products in double; abs_scale (engine only, may be null) the sum of their absolute values
void bf16_matmul(const RefMatrix& w, const float* x, int64_t T, float* y, double* abs_scale) {
    const int64_t K = w.k, R = w.rows;
    if (K % 32) failf("bf16 matrix with rows of %lld values (the fork's dots take multiples of 32)", (long long) K);
    const std::vector<float> wf = bf16_weights(w), xb = bf16_acts(x, T * K);
    const bool fork = w.acts == RefActs::Fork;
    parallel_for(R, T * R * K, [&](int64_t r) {
        const float* wr = wf.data() + r * K;
        for (int64_t t = 0; t < T; ++t) {
            const float* xr = xb.data() + t * K;
            if (fork) {
                y[t * R + r] = T >= 2 ? fork_cpu::tinyblas_dot(wr, xr, K) : fork_cpu::vecdot_bf16(wr, xr, K);
                continue;
            }
            double s = 0.0, a = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                const double p = (double) (wr[k] * xr[k]);   // bf16 x bf16 is exact in f32
                s += p;
                a += std::fabs(p);
            }
            if (y) y[t * R + r] = (float) s;
            if (abs_scale) abs_scale[t * R + r] = a;
        }
    });
}

// the engine's quantized matmul of x [T][k] (qmatmul_ref.hpp), its terms' absolute sum when scale is not null
void engine_qmatmul(const RefMatrix& w, const float* x, int64_t T, float* y, double* scale) {
    const QWeightView v{w.type, w.k, w.rows, w.data};
    QActLayout l = QActLayout::Q8_1;
    switch (w.acts) {
        case RefActs::Auto:
            if (qmatmul_path(QPrefill::DequantF16, T) == QPath::DequantF16) {
                qref_matmul_f16(v, x, w.k, T, 1, y, w.rows, scale);
                return;
            }
            l = qact_layout(w.type, T);
            break;
        case RefActs::Q8_1: l = QActLayout::Q8_1; break;
        case RefActs::MmqD4: l = QActLayout::MmqD4; break;
        case RefActs::MmqD2S6: l = QActLayout::MmqD2S6; break;
        case RefActs::F16: qref_matmul_f16(v, x, w.k, T, 1, y, w.rows, scale); return;
        case RefActs::Fork: failf("engine_qmatmul: a Fork-numerics weight");
    }
    QActsHost a;
    qref_quantize(x, w.k, T, w.k, l, a);
    qref_matmul(v, a, 1, y, w.rows, scale);
}

void matmul_named(const RefMatrix& w, const float* x, int64_t T, float* y, const char* name) {
    check_matrix(w, name);
    ref_matmul(w, x, T, y);
}

}  // namespace

void ref_matmul(const RefMatrix& w, const float* x, int64_t T, float* y) {
    const int64_t K = w.k, R = w.rows;
    if (w.type == kRefTypeBF16) {
        bf16_matmul(w, x, T, y, nullptr);
        return;
    }
    if (w.type != kRefTypeF16 && w.acts != RefActs::Fork) {
        engine_qmatmul(w, x, T, y, nullptr);
        return;
    }
    if (w.type == kRefTypeQ2K || w.type == kRefTypeQ3K || w.type == kRefTypeQ6K) {
        if (K % QK_K) failf("%s matrix with rows of %lld values (not whole blocks)", qtype_name(w.type), (long long) K);
        const uint8_t* blocks = static_cast<const uint8_t*>(w.data);
        const int64_t nb = K / QK_K;
        const size_t row_bytes = (size_t) nb * qtype_block_bytes(w.type);
        std::vector<BlockQ8K> q((size_t) (T * nb));
        for (int64_t t = 0; t < T; ++t) fork_cpu::quantize_q8_K(x + t * K, K, q.data() + t * nb);
        parallel_for(R, T * R * K, [&](int64_t r) {
            const uint8_t* row = blocks + (size_t) r * row_bytes;
            for (int64_t t = 0; t < T; ++t)
                y[t * R + r] = w.type == kRefTypeQ2K
                                   ? fork_cpu::vec_dot_q2_K_q8_K(reinterpret_cast<const BlockQ2K*>(row), q.data() + t * nb, K)
                                   : fork_cpu::vec_dot_qk_q8_K(w.type, row, q.data() + t * nb, K);
        });
        return;
    }
    if (w.type != kRefTypeF16) failf("ref_matmul: weight type %d", w.type);
    const uint16_t* wh = static_cast<const uint16_t*>(w.data);
    const float* tab = f16_table();
    std::vector<float> xh((size_t) (T * K));
    for (int64_t i = 0; i < T * K; ++i) xh[(size_t) i] = fork_cpu::f16_round(x[i]);
    parallel_for(R, T * R * K, [&](int64_t r) {
        const uint16_t* wr = wh + r * K;
        for (int64_t t = 0; t < T; ++t) {
            const float* xr = xh.data() + t * K;
            double s = 0.0;
            for (int64_t k = 0; k < K; ++k) s += (double) (xr[k] * tab[wr[k]]);   // f16 x f16 is exact in f32
            y[t * R + r] = (float) s;
        }
    });
}

void ref_matmul_abs_scale(const RefMatrix& w, const float* x, int64_t T, double* scale) {
    if (w.type == kRefTypeBF16) {
        if (w.acts == RefActs::Fork) failf("ref_matmul_abs_scale takes engine-numerics weights");
        bf16_matmul(w, x, T, nullptr, scale);
        return;
    }
    if (w.type != kRefTypeF16) {
        if (w.acts == RefActs::Fork) failf("ref_matmul_abs_scale takes engine-numerics weights");
        std::vector<float> y((size_t) (T * w.rows));
        engine_qmatmul(w, x, T, y.data(), scale);
        return;
    }
    const int64_t K = w.k, R = w.rows;
    const uint16_t* wh = static_cast<const uint16_t*>(w.data);
    const float* tab = f16_table();
    parallel_for(R, T * R * K, [&](int64_t r) {
        for (int64_t t = 0; t < T; ++t) {
            double s = 0.0;
            for (int64_t k = 0; k < K; ++k) s += std::fabs((double) fork_cpu::f16_round(x[t * K + k]) * tab[wh[r * K + k]]);
            scale[t * R + r] = s;
        }
    });
}

std::vector<float> ref_dequant(const RefMatrix& w) {
    std::vector<float> out((size_t) (w.k * w.rows));
    if (w.type == kRefTypeQ2K) {
        fork_cpu::dequant_q2_K(static_cast<const BlockQ2K*>(w.data), w.k * w.rows, out.data());
    } else if (w.type == kRefTypeF16) {
        const uint16_t* h = static_cast<const uint16_t*>(w.data);
        for (size_t i = 0; i < out.size(); ++i) out[i] = kernels::f32_from_f16(h[i]);
    } else if (w.type == kRefTypeBF16) {
        out = bf16_weights(w);
    } else if (qtype_block_bytes(w.type)) {
        const QWeightView v{w.type, w.k, w.rows, w.data};
        for (int64_t r = 0; r < w.rows; ++r) qref_dequant_row(v, r, out.data() + r * w.k);
    } else {
        failf("ref_dequant: weight type %d", w.type);
    }
    return out;
}

std::vector<uint16_t> ref_to_f16(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = kernels::f16_from_f32(v[i]);
    return h;
}

// ---------------------------------------------------------------- stages

void ref_stage_q_from_qr(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* qr, int64_t T,
                         const int32_t* pos, float* q) {
    const int64_t HD = (int64_t) d.n_head * d.head_dim;
    matmul_named(w.wq_b, qr, T, q, "attn_q_b");
    const fork_cpu::RopeCfg c = rope_cfg(d, l.ratio != 0);
    for (int64_t t = 0; t < T; ++t) rope_heads(c, pos[t], 1.f, q + t * HD, d.n_head, d.head_dim);
}

void ref_stage_q(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* x, int64_t T,
                 const int32_t* pos, float* qr, float* q) {
    std::vector<float> raw((size_t) (T * d.q_lora));
    matmul_named(w.wq_a, x, T, raw.data(), "attn_q_a");
    for (int64_t t = 0; t < T; ++t)
        fork_cpu::rms_norm_mul(raw.data() + t * d.q_lora, d.q_lora, w.q_a_norm, d.rms_eps, qr + t * d.q_lora);
    ref_stage_q_from_qr(d, l, w, qr, T, pos, q);
}

void ref_stage_kv(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* x, int64_t T,
                  const int32_t* pos, float* kv) {
    std::vector<float> raw((size_t) (T * d.head_dim));
    matmul_named(w.wkv, x, T, raw.data(), "attn_kv");
    const fork_cpu::RopeCfg c = rope_cfg(d, l.ratio != 0);
    for (int64_t t = 0; t < T; ++t) {
        fork_cpu::rms_norm_mul(raw.data() + t * d.head_dim, d.head_dim, w.kv_norm, d.rms_eps, kv + t * d.head_dim);
        rope_heads(c, pos[t], 1.f, kv + t * d.head_dim, 1, d.head_dim);
    }
}

void ref_stage_pool(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, RefNumerics n, const float* gkv,
                    const float* gscore, int64_t B, int64_t first_block, float* pooled, float* latent, float* rows) {
    const int r = l.ratio, D = d.head_dim;
    std::vector<float> s((size_t) r), wt((size_t) r), tmp((size_t) r);
    for (int64_t b = 0; b < B; ++b) {
        for (int dd = 0; dd < D; ++dd) {
            for (int i = 0; i < r; ++i) s[(size_t) i] = gscore[(b * r + i) * D + dd];
            if (n == RefNumerics::Fork) {
                // ggml_soft_max over the r tokens, ggml_mul by the values, ggml_sum_rows (a double sum)
                fork_cpu::soft_max_row(s.data(), r, 1.0f, nullptr, nullptr, wt.data(), tmp.data());
                double acc = 0.0;
                for (int i = 0; i < r; ++i) acc += (double) (gkv[(b * r + i) * D + dd] * wt[(size_t) i]);
                pooled[b * D + dd] = (float) acc;
            } else {
                double mx = -INFINITY, sum = 0.0, acc = 0.0;
                for (int i = 0; i < r; ++i) mx = std::max(mx, (double) s[(size_t) i]);
                for (int i = 0; i < r; ++i) sum += std::exp((double) s[(size_t) i] - mx);
                for (int i = 0; i < r; ++i)
                    acc += (double) gkv[(b * r + i) * D + dd] * (std::exp((double) s[(size_t) i] - mx) / sum);
                pooled[b * D + dd] = (float) acc;
            }
        }
        fork_cpu::rms_norm_mul(pooled + b * D, D, w.comp_norm, d.rms_eps, latent + b * D);
        std::memcpy(rows + b * D, latent + b * D, sizeof(float) * (size_t) D);
        rope_heads(rope_cfg(d, true), (first_block + b) * r, 1.f, rows + b * D, 1, D);
    }
}

void ref_stage_index_keys(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* latent,
                          int64_t B, int64_t first_block, float* idx_k_new, float* idx_kf) {
    const int ID = d.idx_dim;
    std::vector<float> raw((size_t) (B * ID));
    matmul_named(w.idx_k, latent, B, raw.data(), "indexer.attn_k");
    const fork_cpu::RopeCfg c = rope_cfg(d, true);
    for (int64_t b = 0; b < B; ++b) {
        fork_cpu::rms_norm_mul(raw.data() + b * ID, ID, w.idx_k_norm, d.rms_eps, idx_k_new + b * ID);
        rope_heads(c, (first_block + b) * l.ratio, 1.f, idx_k_new + b * ID, 1, ID);
        std::memcpy(idx_kf + b * ID, idx_k_new + b * ID, sizeof(float) * (size_t) ID);
        fork_cpu::fwht(idx_kf + b * ID, ID);
    }
}

void ref_stage_index_query(const AttnDims& d, const RefLayerWeights& w, const float* qr, const float* x, int64_t T,
                           const int32_t* pos, float* iq, float* iq_rot, float* iw) {
    const int IH = d.idx_heads, ID = d.idx_dim;
    matmul_named(w.idx_q_b, qr, T, iq, "indexer.attn_q_b");
    const fork_cpu::RopeCfg c = rope_cfg(d, true);
    for (int64_t t = 0; t < T; ++t) rope_heads(c, pos[t], 1.f, iq + t * IH * ID, IH, ID);
    std::memcpy(iq_rot, iq, sizeof(float) * (size_t) (T * IH * ID));
    for (int64_t i = 0; i < T * IH; ++i) fork_cpu::fwht(iq_rot + i * ID, ID);
    matmul_named(w.idx_proj, x, T, iw, "indexer.proj");
    const float s = 1.0f / std::sqrt((float) (ID * IH));
    for (int64_t i = 0; i < T * IH; ++i) iw[i] = iw[i] * s;
}

void ref_stage_attend(const AttnDims& d, RefNumerics n, const float* q, const uint16_t* const* keys, const float* mask,
                      int64_t n_cells, const float* sinks, int64_t ubatch_tokens, float* out, float* kq, float* sm) {
    const int H = d.n_head, D = d.head_dim;
    const float* tab = f16_table();
    const float kq_scale = 1.0f / std::sqrt((float) D);
    std::vector<float> qh((size_t) H * D);
    for (size_t i = 0; i < qh.size(); ++i) qh[i] = fork_cpu::f16_round(q[i]);
    if (n == RefNumerics::Fork) {
        if (n_cells % 32) failf("the fork's attention runs over cells padded to 256; got %lld", (long long) n_cells);
        // K (and V = K) as f32, absent cells zero; V transposed for the kqv dot over cells
        std::vector<float> kf((size_t) (n_cells * D), 0.f), vt((size_t) (D * n_cells), 0.f);
        for (int64_t c = 0; c < n_cells; ++c)
            if (keys[c])
                for (int k = 0; k < D; ++k) {
                    kf[(size_t) (c * D + k)] = tab[keys[c][k]];
                    vt[(size_t) (k * n_cells + c)] = tab[keys[c][k]];
                }
        std::vector<float> s((size_t) n_cells), p((size_t) n_cells), wp((size_t) n_cells), ph((size_t) n_cells);
        for (int h = 0; h < H; ++h) {
            for (int64_t c = 0; c < n_cells; ++c)
                s[(size_t) c] = fork_cpu::f16_dot(kf.data() + c * D, qh.data() + (size_t) h * D, D, ubatch_tokens);
            fork_cpu::soft_max_row(s.data(), n_cells, kq_scale, mask, sinks + h, p.data(), wp.data());
            if (kq) std::memcpy(kq + (size_t) h * n_cells, s.data(), sizeof(float) * (size_t) n_cells);
            if (sm) std::memcpy(sm + (size_t) h * n_cells, p.data(), sizeof(float) * (size_t) n_cells);
            for (int64_t c = 0; c < n_cells; ++c) ph[(size_t) c] = fork_cpu::f16_round(p[(size_t) c]);
            for (int k = 0; k < D; ++k)
                out[(size_t) h * D + k] = fork_cpu::f16_dot(vt.data() + (size_t) k * n_cells, ph.data(), n_cells, ubatch_tokens);
        }
        return;
    }
    // engine numerics: the cells the engine attends, in double
    std::vector<double> s((size_t) n_cells), acc((size_t) D);
    for (int h = 0; h < H; ++h) {
        double mx = sinks[h];
        for (int64_t c = 0; c < n_cells; ++c) {
            if (!keys[c] || (mask && mask[c] != 0.f)) { s[(size_t) c] = -INFINITY; continue; }
            double a = 0.0;
            for (int k = 0; k < D; ++k) a += (double) (qh[(size_t) h * D + k] * tab[keys[c][k]]);
            s[(size_t) c] = a * (double) kq_scale;
            mx = std::max(mx, s[(size_t) c]);
        }
        double den = std::exp((double) sinks[h] - mx);
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int64_t c = 0; c < n_cells; ++c) {
            if (s[(size_t) c] == -INFINITY) continue;
            const double e = std::exp(s[(size_t) c] - mx);
            den += e;
            for (int k = 0; k < D; ++k) acc[(size_t) k] += e * tab[keys[c][k]];
        }
        for (int k = 0; k < D; ++k) out[(size_t) h * D + k] = (float) (acc[(size_t) k] / den);
    }
}

void ref_stage_attend_scale(const AttnDims& d, const float* q, const uint16_t* const* keys, int64_t n_cells,
                            const float* sinks, double* scale) {
    const int H = d.n_head, D = d.head_dim;
    const float* tab = f16_table();
    const double kq_scale = 1.0 / std::sqrt((double) D);
    std::vector<double> s((size_t) n_cells);
    for (int h = 0; h < H; ++h) {
        double mx = sinks[h];
        for (int64_t c = 0; c < n_cells; ++c) {
            if (!keys[c]) { s[(size_t) c] = -INFINITY; continue; }
            double a = 0.0;
            for (int k = 0; k < D; ++k) a += (double) fork_cpu::f16_round(q[(size_t) h * D + k]) * tab[keys[c][k]];
            s[(size_t) c] = a * kq_scale;
            mx = std::max(mx, s[(size_t) c]);
        }
        double den = std::exp((double) sinks[h] - mx);
        for (int64_t c = 0; c < n_cells; ++c)
            if (s[(size_t) c] != -INFINITY) den += std::exp(s[(size_t) c] - mx);
        for (int k = 0; k < D; ++k) scale[(size_t) h * D + k] = 0.0;
        for (int64_t c = 0; c < n_cells; ++c) {
            if (s[(size_t) c] == -INFINITY) continue;
            const double p = std::exp(s[(size_t) c] - mx) / den;
            for (int k = 0; k < D; ++k) scale[(size_t) h * D + k] += p * std::fabs((double) tab[keys[c][k]]);
        }
    }
}

void ref_stage_wo_a(const AttnDims& d, const RefLayerWeights& w, const float* derope, int64_t T, float* oa) {
    check_matrix(w.wo_a, "attn_output_a");
    const int OG = d.o_groups, OL = d.o_lora;
    const int64_t gd = (int64_t) d.n_head / OG * d.head_dim, HD = (int64_t) d.n_head * d.head_dim;
    if (w.wo_a.k != gd || w.wo_a.rows != (int64_t) OG * OL)
        failf("attn_output_a is [%lld, %lld], the dims say [%lld, %d]", (long long) w.wo_a.k, (long long) w.wo_a.rows,
              (long long) gd, OG * OL);
    // the fork's batched mul_mat over the groups: group g's rows g*OL.. of wo_a against its heads' values
    const size_t row_bytes = w.wo_a.type == kRefTypeF16 ? (size_t) gd * 2 : (size_t) (gd / QK_K) * qtype_block_bytes(w.wo_a.type);
    std::vector<float> xg((size_t) (T * gd)), yg((size_t) (T * OL));
    for (int g = 0; g < OG; ++g) {
        for (int64_t t = 0; t < T; ++t) std::memcpy(xg.data() + t * gd, derope + t * HD + g * gd, sizeof(float) * (size_t) gd);
        RefMatrix wg = w.wo_a;
        wg.rows = OL;
        wg.data = static_cast<const uint8_t*>(w.wo_a.data) + (size_t) g * OL * row_bytes;
        ref_matmul(wg, xg.data(), T, yg.data());
        for (int64_t t = 0; t < T; ++t) std::memcpy(oa + t * OG * OL + (int64_t) g * OL, yg.data() + t * OL, sizeof(float) * (size_t) OL);
    }
}

void ref_stage_output(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, const float* attn, int64_t T,
                      const int32_t* pos, float* derope, float* oa, float* out) {
    const int64_t HD = (int64_t) d.n_head * d.head_dim;
    std::memcpy(derope, attn, sizeof(float) * (size_t) (T * HD));
    const fork_cpu::RopeCfg c = rope_cfg(d, l.ratio != 0);
    for (int64_t t = 0; t < T; ++t) rope_heads(c, pos[t], -1.f, derope + t * HD, d.n_head, d.head_dim);
    ref_stage_wo_a(d, w, derope, T, oa);
    matmul_named(w.wo_b, oa, T, out, "attn_output_b");
}

// ---------------------------------------------------------------- the layer

void ref_attn_forward(const AttnDims& d, const AttnLayer& l, const RefLayerWeights& w, RefKv& kv, const float* x,
                      int64_t T, int64_t p0, const RefOptions& o, float* out, RefTrace* trace) {
    if ((size_t) l.index >= kv.layers.size()) failf("layer %d is outside the state's %zu layers", l.index, kv.layers.size());
    RefKv::Layer& L = kv.layers[(size_t) l.index];
    if (L.next_pos != p0) failf("layer %d: chunk at position %lld, the layer is at %lld", l.index, (long long) p0, (long long) L.next_pos);
    if (T < 1) failf("layer %d: chunk of %lld tokens", l.index, (long long) T);
    const int D = d.head_dim, H = d.n_head, r = l.ratio, ID = d.idx_dim;
    const int64_t HD = (int64_t) H * D;
    RefTrace local;
    RefTrace& tr = trace ? *trace : local;
    tr = RefTrace{};
    std::vector<int32_t> pos((size_t) T);
    for (int64_t t = 0; t < T; ++t) pos[(size_t) t] = (int32_t) (p0 + t);

    // query and window
    tr.qr.resize((size_t) (T * d.q_lora));
    tr.q.resize((size_t) (T * HD));
    ref_stage_q(d, l, w, x, T, pos.data(), tr.qr.data(), tr.q.data());
    tr.kv.resize((size_t) (T * D));
    ref_stage_kv(d, l, w, x, T, pos.data(), tr.kv.data());
    if ((int64_t) L.window.size() != p0 * D) failf("layer %d: window holds %zu values, expected %lld positions", l.index, L.window.size() / D, (long long) p0);
    for (int64_t i = 0; i < T * D; ++i) L.window.push_back(kernels::f16_from_f32(tr.kv[(size_t) i]));

    // compressor: the blocks this chunk completes, from the carried positions and the chunk
    if (l.kv_source) {
        tr.ckv.resize((size_t) (T * D));
        matmul_named(w.comp_kv, x, T, tr.ckv.data(), "attn_compressor_kv");
        if (l.gate) {
            tr.cscore.resize((size_t) (T * D));
            matmul_named(w.comp_gate, x, T, tr.cscore.data(), "attn_compressor_gate");
        } else {
            tr.cscore = tr.ckv;
        }
        const int64_t n_carry = p0 % r, first = p0 / r, end = (p0 + T) / r, B = end - first;
        if ((int64_t) L.carry_kv.size() != n_carry * D || (int64_t) L.rows.size() != first * D)
            failf("layer %d at position %lld: %zu carried positions and %zu rows, expected %lld and %lld", l.index,
                  (long long) p0, L.carry_kv.size() / D, L.rows.size() / D, (long long) n_carry, (long long) first);
        auto row_of = [&](int64_t p, bool score) -> const float* {
            if (p < p0) return (score ? L.carry_score : L.carry_kv).data() + (p - (p0 - n_carry)) * D;
            return (score ? tr.cscore : tr.ckv).data() + (p - p0) * D;
        };
        tr.B = B;
        tr.first_block = first;
        tr.gathered_kv.resize((size_t) (B * r * D));
        tr.gathered_score.resize((size_t) (B * r * D));
        for (int64_t b = 0; b < B; ++b)
            for (int i = 0; i < r; ++i) {
                const int64_t p = (first + b) * r + i;
                std::memcpy(tr.gathered_kv.data() + (b * r + i) * D, row_of(p, false), sizeof(float) * (size_t) D);
                std::memcpy(tr.gathered_score.data() + (b * r + i) * D, row_of(p, true), sizeof(float) * (size_t) D);
            }
        tr.pooled.resize((size_t) (B * D));
        tr.latent.resize((size_t) (B * D));
        tr.rows.resize((size_t) (B * D));
        ref_stage_pool(d, l, w, o.numerics, tr.gathered_kv.data(), tr.gathered_score.data(), B, first, tr.pooled.data(),
                       tr.latent.data(), tr.rows.data());
        for (int64_t i = 0; i < B * D; ++i) L.rows.push_back(kernels::f16_from_f32(tr.rows[(size_t) i]));
        if (l.key_owner) {
            tr.idx_k_new.resize((size_t) (B * ID));
            tr.idx_kf.resize((size_t) (B * ID));
            ref_stage_index_keys(d, l, w, tr.latent.data(), B, first, tr.idx_k_new.data(), tr.idx_kf.data());
            for (int64_t i = 0; i < B * ID; ++i) L.keys.push_back(kernels::f16_from_f32(tr.idx_kf[(size_t) i]));
        }
        // the incomplete block's positions end * r .. p0 + T - 1 carry over (from the old carry when none completed)
        std::vector<float> ck, cs;
        for (int64_t p = end * r; p < p0 + T; ++p) {
            ck.insert(ck.end(), row_of(p, false), row_of(p, false) + D);
            cs.insert(cs.end(), row_of(p, true), row_of(p, true) + D);
        }
        L.carry_kv.swap(ck);
        L.carry_score.swap(cs);
    }

    // index source: its own selection over the key source's keys
    std::vector<std::vector<int32_t>> sel((size_t) T);
    if (l.index_source) {
        const int IH = d.idx_heads, K = d.top_k;
        tr.iq.resize((size_t) (T * IH * ID));
        tr.iq_rot.resize(tr.iq.size());
        tr.iw.resize((size_t) (T * IH));
        ref_stage_index_query(d, w, tr.qr.data(), x, T, pos.data(), tr.iq.data(), tr.iq_rot.data(), tr.iw.data());
        const RefKv::Layer& ks = kv.layers[(size_t) l.key_src];
        if (l.key_src != l.index && ks.next_pos < p0 + T)
            failf("layer %d reads layer %d's index keys, which has reached position %lld, not %lld", l.index, l.key_src,
                  (long long) ks.next_pos, (long long) (p0 + T));
        const int64_t n_k = (int64_t) ks.keys.size() / ID;
        tr.n_k = n_k;
        tr.scores.assign((size_t) (T * n_k), -INFINITY);
        tr.topk.assign((size_t) (T * K), -1);
        for (int64_t t = 0; t < T; ++t) {
            const int64_t vis = indexer_visible(pos[(size_t) t], r, n_k);
            indexer_ref_scores(tr.iq_rot.data() + t * IH * ID, tr.iw.data() + t * IH, IH, ks.keys.data(), vis, T,
                               tr.scores.data() + t * n_k);
            indexer_ref_topk(tr.scores.data() + t * n_k, vis, K, tr.topk.data() + t * K);
            if (o.topk_override) {
                for (int64_t i = 0; i < o.topk_override_k; ++i) {
                    const int32_t j = o.topk_override[t * o.topk_override_k + i];
                    if (j >= 0 && j < vis) sel[(size_t) t].push_back(j);
                }
                std::sort(sel[(size_t) t].begin(), sel[(size_t) t].end());
                sel[(size_t) t].erase(std::unique(sel[(size_t) t].begin(), sel[(size_t) t].end()), sel[(size_t) t].end());
                if ((int64_t) sel[(size_t) t].size() > K) failf("layer %d: an override selection of %zu rows, top-k %d", l.index, sel[(size_t) t].size(), K);
            } else {
                for (int i = 0; i < K; ++i)
                    if (tr.topk[(size_t) (t * K + i)] >= 0) sel[(size_t) t].push_back(tr.topk[(size_t) (t * K + i)]);
            }
        }
        L.ids.assign((size_t) (T * K), -1);
        for (int64_t t = 0; t < T; ++t) std::copy(sel[(size_t) t].begin(), sel[(size_t) t].end(), L.ids.begin() + t * K);
        L.ids_p0 = p0;
        L.ids_T = T;
    } else if (r != 0 && o.mode == ReaderMode::Reuse) {
        const RefKv::Layer& src = kv.layers[(size_t) l.topk_src];
        if (src.ids_p0 != p0 || src.ids_T != T)
            failf("reader %d (Reuse) needs layer %d's top-k of the chunk at %lld (%lld tokens); it holds %lld (%lld)",
                  l.index, l.topk_src, (long long) p0, (long long) T, (long long) src.ids_p0, (long long) src.ids_T);
        for (int64_t t = 0; t < T; ++t)
            for (int i = 0; i < d.top_k; ++i)
                if (src.ids[(size_t) (t * d.top_k + i)] >= 0) sel[(size_t) t].push_back(src.ids[(size_t) (t * d.top_k + i)]);
    }

    // attention over the window and the compressed rows of the KV source
    const RefKv::Layer* vsrc = r ? &kv.layers[(size_t) l.kv_src] : nullptr;
    const int64_t n_rows = vsrc ? (int64_t) vsrc->rows.size() / D : 0;
    const bool dense = r != 0 && !l.index_source && o.mode == ReaderMode::Dense;
    tr.attn.resize((size_t) (T * HD));
    if (o.numerics == RefNumerics::Fork) {
        if (!o.layout) failf("layer %d: the fork's numerics need its cell layout (RefForkLayout)", l.index);
        const int64_t W = o.layout->W, C = r ? o.layout->C : 0, N = W + C;
        if (p0 + T > W) failf("layer %d: %lld positions do not fit the fork's %lld window cells (cell = position)", l.index, (long long) (p0 + T), (long long) W);
        if (r && n_rows > C) failf("layer %d: %lld compressed rows, the fork's layout has %lld cells", l.index, (long long) n_rows, (long long) C);
        tr.mask.assign((size_t) (T * N), -INFINITY);
        tr.kq.resize((size_t) (H * T * N));
        tr.kq_soft_max.resize((size_t) (H * T * N));
        std::vector<const uint16_t*> keys((size_t) N, nullptr);
        for (int64_t c = 0; c < p0 + T; ++c) keys[(size_t) c] = L.window.data() + c * D;
        for (int64_t j = 0; j < n_rows; ++j) keys[(size_t) (W + j)] = vsrc->rows.data() + j * D;
        std::vector<float> kq((size_t) (H * N)), sm((size_t) (H * N));
        for (int64_t t = 0; t < T; ++t) {
            const int64_t p = pos[(size_t) t];
            float* m = tr.mask.data() + t * N;
            for (int64_t c = 0; c <= p; ++c)
                if (p - c < d.n_swa) m[c] = 0.f;
            if (r) {
                const int64_t vis = std::min<int64_t>(n_rows, (p + 1) / r);
                if (dense) {
                    for (int64_t j = 0; j < vis; ++j) m[W + j] = 0.f;
                } else {
                    for (int32_t j : sel[(size_t) t])
                        if (j < vis) m[W + j] = 0.f;
                }
            }
            ref_stage_attend(d, RefNumerics::Fork, tr.q.data() + t * HD, keys.data(), m, N, w.sinks, T, tr.attn.data() + t * HD,
                             kq.data(), sm.data());
            for (int h = 0; h < H; ++h) {
                std::memcpy(tr.kq.data() + ((size_t) h * T + t) * N, kq.data() + (size_t) h * N, sizeof(float) * (size_t) N);
                std::memcpy(tr.kq_soft_max.data() + ((size_t) h * T + t) * N, sm.data() + (size_t) h * N, sizeof(float) * (size_t) N);
            }
        }
    } else {
        for (int64_t t = 0; t < T; ++t) {
            const int64_t p = pos[(size_t) t];
            std::vector<const uint16_t*> keys;
            for (int64_t q = std::max<int64_t>(0, p - d.n_swa + 1); q <= p; ++q) keys.push_back(L.window.data() + q * D);
            if (r) {
                const int64_t vis = std::min<int64_t>(n_rows, (p + 1) / r);
                if (dense) {
                    for (int64_t j = 0; j < vis; ++j) keys.push_back(vsrc->rows.data() + j * D);
                } else {
                    for (int32_t j : sel[(size_t) t])
                        if (j < vis) keys.push_back(vsrc->rows.data() + (int64_t) j * D);
                }
            }
            ref_stage_attend(d, RefNumerics::Engine, tr.q.data() + t * HD, keys.data(), nullptr, (int64_t) keys.size(),
                             w.sinks, T, tr.attn.data() + t * HD, nullptr, nullptr);
        }
    }

    tr.derope.resize((size_t) (T * HD));
    tr.oa.resize((size_t) (T * d.o_groups * d.o_lora));
    tr.out.resize((size_t) (T * d.n_embd));
    ref_stage_output(d, l, w, tr.attn.data(), T, pos.data(), tr.derope.data(), tr.oa.data(), tr.out.data());
    std::memcpy(out, tr.out.data(), sizeof(float) * tr.out.size());
    L.next_pos = p0 + T;
}

}  // namespace strata::dsv41::engine
