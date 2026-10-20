// src/dsv41/engine/dense_ref.cpp - the CPU reference of the dense ops, see include/strata/dsv41/engine/dense_ref.hpp.
// Compiled with -mfma -ffp-contract=off (CMakeLists.txt): the fork's contractions are written out as std::fma, every
// other product and sum rounds on its own, as the fork's separate ggml ops do (deepseek4.cpp 286-457, deepseek41.cpp
// 357-438 and 805-947, llama-graph.cpp 1783-1982 and 2056-2187).
#include "strata/dsv41/engine/dense_ref.hpp"

#include "fork_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fork_cpu::fail("dense_ref", msg);
}

void check(const RefMatrix& w, int64_t k, int64_t rows, const char* name) {
    if (!w.data) failf("%s is missing", name);
    if (w.k != k || w.rows != rows)
        failf("%s is [%lld][%lld], expected [%lld][%lld]", name, (long long) w.rows, (long long) w.k, (long long) rows, (long long) k);
}

}  // namespace

void ref_embed(const RefMatrix& token_embd, const int32_t* tokens, int64_t T, float* embd) {
    if (token_embd.type != kRefTypeQ2K) failf("token_embd has type %d, the dequant here is q2_K's", token_embd.type);
    const int64_t E = token_embd.k, nb = E / fork_cpu::QK_K;
    const auto* blocks = static_cast<const fork_cpu::BlockQ2K*>(token_embd.data);
    for (int64_t t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= token_embd.rows) failf("token %d of %lld", tokens[t], (long long) token_embd.rows);
        fork_cpu::dequant_q2_K(blocks + (int64_t) tokens[t] * nb, E, embd + t * E);
    }
}

void ref_hc_norm(const DenseDims& d, const float* x, int64_t T, float* hc_in) {
    const int64_t n = (int64_t) kHc * d.n_embd;
    for (int64_t t = 0; t < T; ++t) fork_cpu::rms_norm(x + t * n, n, d.rms_eps, hc_in + t * n);
}

void ref_hc_mixes(const DenseDims& d, const RefHcWeights& w, const float* x, int64_t T, float* mixes) {
    const int64_t n = (int64_t) kHc * d.n_embd;
    check(w.fn, n, kHcMix, "hc_fn");
    std::vector<float> xn((size_t) (T * n));
    ref_hc_norm(d, x, T, xn.data());
    ref_matmul(w.fn, xn.data(), T, mixes);
}

void ref_hc_coeffs(const DenseDims& d, const RefHcWeights& w, const float* mixes, int64_t T, float* pre, float* post,
                   float* comb) {
    const float eps = d.hc_eps;
    const float* s = w.scale;
    const float* b = w.base;
    for (int64_t t = 0; t < T; ++t) {
        const float* m = mixes + t * kHcMix;
        for (int i = 0; i < kHc; ++i) {
            // dsv4_hc_affine is ggml_mul then ggml_add, the sigmoid its own op, then scale_bias (x + eps) or x 2
            const float a = m[i] * s[0];
            pre[t * kHc + i] = fork_cpu::sigmoid(a + b[i]) + eps;
            const float c = m[kHc + i] * s[1];
            post[t * kHc + i] = 2.0f * fork_cpu::sigmoid(c + b[kHc + i]);
        }
        // DSV4_HC_COMB: per src a softmax over dst, + eps, then column / row normalizations (idx = dst + 4 src)
        float cb[kHc * kHc];
        for (int src = 0; src < kHc; ++src) {
            float mx = -INFINITY;
            for (int dst = 0; dst < kHc; ++dst) {
                const int idx = dst + kHc * src;
                cb[idx] = std::fma(m[2 * kHc + idx], s[2], b[2 * kHc + idx]);
                mx = std::max(mx, cb[idx]);
            }
            float sum = 0.0f;
            for (int dst = 0; dst < kHc; ++dst) {
                const int idx = dst + kHc * src;
                cb[idx] = std::exp(cb[idx] - mx);
                sum += cb[idx];
            }
            const float inv = 1.0f / sum;
            for (int dst = 0; dst < kHc; ++dst) cb[dst + kHc * src] = std::fma(cb[dst + kHc * src], inv, eps);
        }
        auto norm_cols = [&] {
            for (int dst = 0; dst < kHc; ++dst) {
                float sum = eps;
                for (int src = 0; src < kHc; ++src) sum += cb[dst + kHc * src];
                const float inv = 1.0f / sum;
                for (int src = 0; src < kHc; ++src) cb[dst + kHc * src] *= inv;
            }
        };
        auto norm_rows = [&] {
            for (int src = 0; src < kHc; ++src) {
                float sum = eps;
                for (int dst = 0; dst < kHc; ++dst) sum += cb[dst + kHc * src];
                const float inv = 1.0f / sum;
                for (int dst = 0; dst < kHc; ++dst) cb[dst + kHc * src] *= inv;
            }
        };
        norm_cols();
        for (int it = 1; it < d.sinkhorn_iters; ++it) {
            norm_rows();
            norm_cols();
        }
        std::memcpy(comb + t * kHc * kHc, cb, sizeof cb);
    }
}

void ref_hc_collapse(const float* x, const float* pre, int64_t T, int E, bool fused, float* out) {
    for (int64_t t = 0; t < T; ++t) {
        const float* xt = x + t * kHc * E;
        const float* w = pre + t * kHc;
        for (int e = 0; e < E; ++e) {
            float sum;
            if (fused) {
                sum = std::fma(xt[e], w[0], 0.0f);   // the fork's `sum = 0; sum += x * w`, contracted
                for (int s = 1; s < kHc; ++s) sum = std::fma(xt[s * E + e], w[s], sum);
            } else {
                sum = xt[e] * w[0];
                for (int s = 1; s < kHc; ++s) {
                    const float p = xt[s * E + e] * w[s];
                    sum = sum + p;
                }
            }
            out[t * E + e] = sum;
        }
    }
}

void ref_hc_expand(const float* y, const float* x, const float* post, const float* comb, int64_t T, int E, float* out) {
    std::vector<float> tmp((size_t) kHc * E);
    for (int64_t t = 0; t < T; ++t) {
        const float* xt = x + t * kHc * E;
        const float* c = comb + t * kHc * kHc;
        for (int dst = 0; dst < kHc; ++dst)
            for (int e = 0; e < E; ++e) {
                float sum = y[t * E + e] * post[t * kHc + dst];
                for (int src = 0; src < kHc; ++src) sum = std::fma(xt[src * E + e], c[dst + kHc * src], sum);
                tmp[(size_t) dst * E + e] = sum;
            }
        std::memcpy(out + t * kHc * E, tmp.data(), sizeof(float) * tmp.size());   // out may be x
    }
}

void ref_rms_norm(const float* x, const float* w, int64_t rows, int n, float eps, float* y) {
    for (int64_t r = 0; r < rows; ++r) fork_cpu::rms_norm_mul(x + r * n, n, w, eps, y + r * n);
}

void ref_engram_emb(const uint8_t* table, int key, const int32_t* rows, int64_t n, float* emb) {
    const int64_t nb = key / fork_cpu::QK_K;
    const auto* blocks = reinterpret_cast<const fork_cpu::BlockQ2K*>(table);
    for (int64_t i = 0; i < n; ++i) fork_cpu::dequant_q2_K(blocks + (int64_t) rows[i] * nb, key, emb + i * key);
}

void ref_engram(const DenseDims& d, const RefEngramWeights& w, const float* emb, const float* x, int64_t T, float* kv,
                float* gate, float* out) {
    const int64_t E = d.n_embd;
    check(w.wkv, (int64_t) d.engram_cols() * d.engram_key, E * (kHc + 1), "engram_wkv");
    ref_matmul(w.wkv, emb, T, kv);
    ref_engram_gate(d, w, kv, x, T, gate, out);
}

void ref_engram_gate(const DenseDims& d, const RefEngramWeights& w, const float* kv, const float* x, int64_t T,
                     float* gate, float* out) {
    const int E = d.n_embd;
    const int64_t KV = (int64_t) E * (kHc + 1);
    const float inv_sqrt = 1.0f / std::sqrt((float) E);   // libllama's constant (no -mfma there; nothing to fuse)
    std::vector<float> kn((size_t) E), qn((size_t) E), prod((size_t) E);
    for (int64_t t = 0; t < T; ++t) {
        const float* key = kv + t * KV;
        const float* value = key + (int64_t) kHc * E;
        for (int c = 0; c < kHc; ++c) {
            fork_cpu::rms_norm_mul(key + (int64_t) c * E, E, w.k + (int64_t) c * E, d.rms_eps, kn.data());
            fork_cpu::rms_norm_mul(x + (t * kHc + c) * E, E, w.q + (int64_t) c * E, d.rms_eps, qn.data());
            for (int e = 0; e < E; ++e) prod[(size_t) e] = kn[(size_t) e] * qn[(size_t) e];
            const float s = fork_cpu::sum_f32(prod.data(), E) * inv_sqrt;
            const float a = std::max(std::min(std::fabs(s), INFINITY), 1e-6f);   // ggml_clamp(|s|, 1e-6, inf)
            const float mag = std::sqrt(a);
            const float sg = s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f);              // ggml_sgn
            gate[t * kHc + c] = fork_cpu::sigmoid(sg * mag);
        }
        for (int c = 0; c < kHc; ++c)
            for (int e = 0; e < E; ++e) {
                const float p = value[e] * gate[t * kHc + c];
                out[(t * kHc + c) * E + e] = x[(t * kHc + c) * E + e] + p;
            }
    }
}

void ref_router(RefNumerics n, const DenseDims& d, const RefRouterWeights& w, const float* xn, int64_t T,
                RefRouterTrace& tr) {
    check(w.gate_inp, d.n_embd, d.n_expert, "ffn_gate_inp");
    if (w.gate_inp.type != kRefTypeBF16) failf("ffn_gate_inp has type %d, expected bf16", w.gate_inp.type);
    tr.logits.assign((size_t) (T * d.n_expert), 0.f);
    ref_matmul(w.gate_inp, xn, T, tr.logits.data());
    ref_router_select(n, d, w.bias, T, tr);
}

void ref_router_select(RefNumerics n, const DenseDims& d, const float* bias, int64_t T, RefRouterTrace& tr) {
    const int NE = d.n_expert, K = d.n_expert_used;
    tr.probs.resize((size_t) (T * NE));
    tr.biased.resize((size_t) (T * NE));
    tr.order.resize((size_t) (T * NE));
    tr.ids.resize((size_t) (T * K));
    tr.weights.resize((size_t) (T * K));
    tr.norm.resize((size_t) (T * K));
    tr.scaled.resize((size_t) (T * K));
    tr.sum.resize((size_t) T);
    tr.sum_clamped.resize((size_t) T);
    for (int64_t t = 0; t < T; ++t) {
        float* p = tr.probs.data() + t * NE;
        float* bz = tr.biased.data() + t * NE;
        for (int i = 0; i < NE; ++i) {
            p[i] = std::sqrt(fork_cpu::softplus(tr.logits[(size_t) (t * NE + i)]));
            bz[i] = p[i] + bias[i];
        }
        int32_t* o = tr.order.data() + t * NE;
        std::iota(o, o + NE, 0);
        // the fork's argsort: std::sort with data[a] > data[b] (ops.cpp 8475-8528); the engine: ties to the lower id
        if (n == RefNumerics::Fork) std::sort(o, o + NE, [bz](int32_t a, int32_t b) { return bz[a] > bz[b]; });
        else std::stable_sort(o, o + NE, [bz](int32_t a, int32_t b) { return bz[a] > bz[b]; });
        float* wt = tr.weights.data() + t * K;
        for (int k = 0; k < K; ++k) {
            tr.ids[(size_t) (t * K + k)] = o[k];
            wt[k] = p[o[k]];
        }
        float* nm = tr.norm.data() + t * K;
        std::memcpy(nm, wt, sizeof(float) * (size_t) K);
        if (d.weights_norm) {
            tr.sum[(size_t) t] = fork_cpu::sum_f32(wt, K);
            const float c = std::max(std::min(tr.sum[(size_t) t], INFINITY), 6.103515625e-5f);
            tr.sum_clamped[(size_t) t] = c;
            for (int k = 0; k < K; ++k) nm[k] = wt[k] / c;
        }
        const bool scaled = d.weights_scale != 0.0f && d.weights_scale != 1.0f;
        for (int k = 0; k < K; ++k) tr.scaled[(size_t) (t * K + k)] = scaled ? nm[k] * d.weights_scale : nm[k];
    }
}

void ref_swiglu_clamp(const float* gate, const float* up, int64_t n, float limit, float* act) {
    for (int64_t i = 0; i < n; ++i) {
        const float g = std::min(gate[i], limit);
        const float u = std::clamp(up[i], -limit, limit);
        act[i] = g / (1.f + std::exp(-g)) * u;
    }
}

void ref_shared_expert(const DenseDims& d, const RefSharedWeights& w, const float* xn, int64_t T, RefSharedTrace& tr) {
    const int64_t E = d.n_embd, F = d.n_ff;
    check(w.gate, E, F, "ffn_gate_shexp");
    check(w.up, E, F, "ffn_up_shexp");
    check(w.down, F, E, "ffn_down_shexp");
    tr.gate.resize((size_t) (T * F));
    tr.up.resize((size_t) (T * F));
    tr.act.resize((size_t) (T * F));
    tr.out.resize((size_t) (T * E));
    ref_matmul(w.up, xn, T, tr.up.data());
    ref_matmul(w.gate, xn, T, tr.gate.data());
    ref_swiglu_clamp(tr.gate.data(), tr.up.data(), T * F, d.swiglu_limit, tr.act.data());
    ref_matmul(w.down, tr.act.data(), T, tr.out.data());
}

void ref_head(const DenseDims& d, const float* x, const float* pre, int64_t n, const float* output_norm,
              const RefMatrix& output, float* hc_out, float* norm, float* logits) {
    check(output, d.n_embd, d.n_vocab, "output");
    ref_hc_collapse(x, pre, n, d.n_embd, false, hc_out);
    ref_rms_norm(hc_out, output_norm, n, d.n_embd, d.rms_eps, norm);
    ref_matmul(output, norm, n, logits);
}

}  // namespace strata::dsv41::engine
