// src/dsv41/engine/qmatmul_ref.cpp - the CPU reference of qmatmul, see include/strata/dsv41/engine/qmatmul_ref.hpp.
// Compiled with -ffp-contract=off (CMakeLists.txt): the quantizers repeat the kernels' f32 operations one by one, and
// qref_dequant_row ggml's dequantize_row_* (ggml-quants.c, built without FMA).
#include "strata/dsv41/engine/qmatmul_ref.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "qmatmul_ref: %s\n", msg);
    std::exit(1);
}

float f16_round(float x) { return kernels::f32_from_f16(kernels::f16_from_f32(x)); }
float f16_val(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return kernels::f32_from_f16(h);
}

// a shuffle-xor butterfly over n lanes as warp_sum computes it: lane i adds lane i ^ o for o = n / 2 .. 1 (addition
// commutes, so every lane ends with lane 0's value)
float butterfly_sum(const float* v, int n) {
    float a[32], b[32];
    std::copy(v, v + n, a);
    for (int o = n / 2; o > 0; o >>= 1) {
        for (int i = 0; i < n; ++i) b[i] = a[i] + a[i ^ o];
        std::copy(b, b + n, a);
    }
    return a[0];
}

constexpr float kHalfMax = 65504.0f;
float finite_half(float v) { return std::fabs(v) > kHalfMax ? std::copysign(kHalfMax, v) : v; }

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

// One K-quant block in integers: value v = d * sc[v / 16] * q[v] - dmin * mn[v / 16] (q3_K, q6_K: no min)
struct Decoded {
    float d = 0, dmin = 0;
    int q[256];
    int sc[16];
    int mn[16];
};

void decode(int type, const uint8_t* b, Decoded& o) {
    std::memset(o.mn, 0, sizeof o.mn);
    o.dmin = 0;
    if (type == kQTypeQ2K) {
        // block_q2_K: scales[16] (low nibble scale, high nibble min), qs[64], d, dmin (dequantize_row_q2_K's order)
        const uint8_t* scales = b;
        const uint8_t* qs = b + 16;
        o.d = f16_val(b + 80);
        o.dmin = f16_val(b + 82);
        for (int g = 0; g < 16; ++g) {
            o.sc[g] = scales[g] & 0xF;
            o.mn[g] = scales[g] >> 4;
        }
        for (int n = 0; n < 2; ++n)
            for (int j = 0; j < 4; ++j)
                for (int l = 0; l < 32; ++l) o.q[128 * n + 32 * j + l] = (qs[32 * n + l] >> (2 * j)) & 3;
    } else if (type == kQTypeQ3K) {
        // block_q3_K: hmask[32], qs[64], scales[12] (6-bit, the kmask unpacking of dequantize_row_q3_K), d
        const uint8_t* hm = b;
        const uint8_t* qs = b + 32;
        uint32_t aux[4];
        std::memcpy(aux, b + 96, 12);
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* s8 = reinterpret_cast<const int8_t*>(aux);
        for (int g = 0; g < 16; ++g) o.sc[g] = s8[g] - 32;
        o.d = f16_val(b + 108);
        for (int n = 0; n < 2; ++n)
            for (int j = 0; j < 4; ++j)
                for (int l = 0; l < 32; ++l) {
                    const int v = 128 * n + 32 * j + l;
                    o.q[v] = ((qs[32 * n + l] >> (2 * j)) & 3) - ((hm[l] & (1 << (4 * n + j))) ? 0 : 4);
                }
    } else if (type == kQTypeQ6K) {
        // block_q6_K: ql[128], qh[64], scales[16] (int8), d
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
        o.d = f16_val(b + 208);
        for (int g = 0; g < 16; ++g) o.sc[g] = sc[g];
        for (int n = 0; n < 2; ++n)
            for (int l = 0; l < 32; ++l) {
                const uint8_t* L = ql + 64 * n;
                const uint8_t* H = qh + 32 * n;
                o.q[128 * n + l + 0] = (int) ((L[l + 0] & 0xF) | (((H[l] >> 0) & 3) << 4)) - 32;
                o.q[128 * n + l + 32] = (int) ((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4)) - 32;
                o.q[128 * n + l + 64] = (int) ((L[l + 0] >> 4) | (((H[l] >> 4) & 3) << 4)) - 32;
                o.q[128 * n + l + 96] = (int) ((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4)) - 32;
            }
    } else {
        failf("weight type %d (%s): the reference reads q2_K, q3_K and q6_K", type, qtype_name(type));
    }
}

size_t row_bytes(const QWeightView& w) { return (size_t) (w.k / kQK) * qtype_block_bytes(w.type); }

void check_weight(const QWeightView& w) {
    if (!w.data) failf("the weight is missing");
    if (!qtype_block_bytes(w.type)) failf("weight type %d (%s)", w.type, qtype_name(w.type));
    if (w.k <= 0 || w.k % kQK || w.rows <= 0) failf("a %lld x %lld matrix", (long long) w.rows, (long long) w.k);
}

// one output: row r of w (in its group) against row t of a, values a0 .. a0 + w.k of a's row
void dot_row(const QWeightView& w, const QActsHost& a, int64_t r, int64_t t, int64_t a0, double& y, double& scale) {
    const uint8_t* row = static_cast<const uint8_t*>(w.data) + (size_t) r * row_bytes(w);
    const int64_t nb = w.k / kQK, G = a.group();
    const int8_t* q8 = a.q.data() + t * a.k + a0;
    const float* dg = a.d.data() + (t * a.k + a0) / G;
    const bool d2s6 = a.layout == QActLayout::MmqD2S6;
    const float* s16 = d2s6 ? a.s.data() + (t * a.k + a0) / 16 : nullptr;
    double acc = 0, abs = 0;
    Decoded b;
    for (int64_t ib = 0; ib < nb; ++ib) {
        decode(w.type, row + (size_t) ib * qtype_block_bytes(w.type), b);
        for (int g = 0; g < 16; ++g) {
            const int64_t v0 = ib * kQK + 16 * g;
            const double d8 = dg[v0 / G];
            if (d2s6) {
                // MMQ's tiles: f16(d sc) and f16(dmin m); the min term from the stored raw sum for groups 0..5 of each
                // 128 values, from d64 sum(q8) for groups 6 and 7
                const double A = f16_round(b.d * (float) b.sc[g]), B = f16_round(b.dmin * (float) b.mn[g]);
                int64_t si = 0, sq = 0;
                for (int l = 0; l < 16; ++l) {
                    si += (int64_t) b.q[16 * g + l] * q8[v0 + l];
                    sq += q8[v0 + l];
                }
                const double ts = A * d8 * (double) si;
                const double tm = (v0 % 128) / 16 < 6 ? B * (double) s16[v0 / 16] : B * d8 * (double) sq;
                acc += ts - tm;
                abs += std::fabs(ts) + std::fabs(tm);
                continue;
            }
            // MMVQ (q8_1) and MMQ D4: d sc sum(q q8) d8 - dmin m sum(q8) d8, the kernels' terms per 4 values
            const double ds = (double) b.d * b.sc[g], dm = (double) b.dmin * b.mn[g];
            for (int c = 0; c < 16; c += 4) {
                int64_t si = 0, sq = 0;
                for (int l = c; l < c + 4; ++l) {
                    si += (int64_t) b.q[16 * g + l] * q8[v0 + l];
                    sq += q8[v0 + l];
                }
                const double ts = ds * d8 * (double) si, tm = dm * d8 * (double) sq;
                acc += ts - tm;
                abs += std::fabs(ts) + std::fabs(tm);
            }
        }
    }
    y = acc;
    scale = abs;
}

void check_acts(const QWeightView& w, const QActsHost& a, int groups) {
    check_weight(w);
    if (groups < 1 || w.rows % groups) failf("%lld rows in %d groups", (long long) w.rows, groups);
    if (a.k != (int64_t) groups * w.k) failf("activations of %lld values for %d x %lld", (long long) a.k, groups, (long long) w.k);
    if ((int64_t) a.q.size() != a.T * a.k || (int64_t) a.d.size() != a.T * a.k / a.group()) failf("malformed activations");
    if (a.layout == QActLayout::MmqD2S6 && (int64_t) a.s.size() != a.T * a.k / 16) failf("D2S6 activations without their sums");
}

}  // namespace

void qref_quantize(const float* x, int64_t ld, int64_t T, int64_t k, QActLayout l, QActsHost& out) {
    if (k <= 0 || k % kQK) failf("rows of %lld values", (long long) k);
    out = QActsHost{};
    out.layout = l;
    out.T = T;
    out.k = k;
    out.q.assign((size_t) (T * k), 0);
    const int64_t G = out.group();
    out.d.assign((size_t) (T * k / G), 0.f);
    if (l == QActLayout::Q8_1) out.s.assign((size_t) (T * k / 32), 0.f);
    if (l == QActLayout::MmqD2S6) out.s.assign((size_t) (T * k / 16), 0.f);
    for (int64_t t = 0; t < T; ++t) {
        const float* xr = x + t * ld;
        for (int64_t g0 = 0; g0 < k; g0 += G) {
            float amax = 0.f;
            for (int64_t i = g0; i < g0 + G; ++i) amax = std::max(amax, std::fabs(xr[i]));
            if (l == QActLayout::Q8_1) {
                // native_quantize_q8_1_kernel: d = amax / 127 (kept finite), q = round(x / d), half d and half sum
                const float d = finite_half(amax / 127.0f);
                for (int64_t i = g0; i < g0 + G; ++i) {
                    int8_t q = 0;
                    if (amax != 0.0f) {
                        const float r = std::round(xr[i] / d);
                        q = (int8_t) (r > 127.0f ? 127.0f : r < -127.0f ? -127.0f : r);
                    }
                    out.q[(size_t) (t * k + i)] = q;
                }
                out.d[(size_t) ((t * k + g0) / G)] = f16_round(d);
                out.s[(size_t) ((t * k + g0) / 32)] = f16_round(finite_half(butterfly_sum(xr + g0, 32)));
            } else {
                // quantize_mmq_kernel: d_inv = 127 / amax, q = round(x d_inv), d = 1 / d_inv (f16 in D2S6)
                const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
                const float d = amax == 0.0f ? 0.0f : 1.0f / d_inv;
                for (int64_t i = g0; i < g0 + G; ++i) out.q[(size_t) (t * k + i)] = (int8_t) std::round(xr[i] * d_inv);
                out.d[(size_t) ((t * k + g0) / G)] = l == QActLayout::MmqD2S6 ? f16_round(d) : d;
            }
        }
        if (l == QActLayout::MmqD2S6)
            for (int64_t s0 = 0; s0 < k; s0 += 16) {
                if ((s0 % 128) / 16 >= 6) continue;
                // a thread's 4 values left to right, then the butterfly over the 4 threads of the 16
                float s4[4];
                for (int j = 0; j < 4; ++j) {
                    const float* v = xr + s0 + 4 * j;
                    s4[j] = ((v[0] + v[1]) + v[2]) + v[3];
                }
                out.s[(size_t) ((t * k + s0) / 16)] = f16_round(butterfly_sum(s4, 4));
            }
    }
}

void qref_unpack(const void* bytes, QActLayout l, int64_t T, int64_t k, QActsHost& out) {
    const uint8_t* p = static_cast<const uint8_t*>(bytes);
    out = QActsHost{};
    out.layout = l;
    out.T = T;
    out.k = k;
    out.q.assign((size_t) (T * k), 0);
    const int64_t G = out.group();
    out.d.assign((size_t) (T * k / G), 0.f);
    if (l == QActLayout::Q8_1) {
        out.s.assign((size_t) (T * k / 32), 0.f);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t b = 0; b < k / 32; ++b) {
                const uint8_t* blk = p + (size_t) ((t * (k / 32) + b) * 36);
                out.d[(size_t) (t * k / 32 + b)] = f16_val(blk);
                out.s[(size_t) (t * k / 32 + b)] = f16_val(blk + 2);
                std::memcpy(out.q.data() + t * k + b * 32, blk + 4, 32);
            }
        return;
    }
    if (l == QActLayout::MmqD2S6) out.s.assign((size_t) (T * k / 16), 0.f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t kt = 0; kt < k / 128; ++kt) {
            const uint8_t* tile = p + (size_t) ((kt * T + t) * 144);
            std::memcpy(out.q.data() + t * k + kt * 128, tile + 16, 128);
            if (l == QActLayout::MmqD4) {
                for (int j = 0; j < 4; ++j) std::memcpy(&out.d[(size_t) ((t * k + kt * 128) / 32 + j)], tile + 4 * j, 4);
            } else {
                for (int j = 0; j < 2; ++j) out.d[(size_t) ((t * k + kt * 128) / 64 + j)] = f16_val(tile + 2 * j);
                for (int j = 0; j < 6; ++j) out.s[(size_t) ((t * k + kt * 128) / 16 + j)] = f16_val(tile + 4 + 2 * j);
            }
        }
}

float qref_act(const QActsHost& a, int64_t t, int64_t i) {
    return a.d[(size_t) ((t * a.k + i) / a.group())] * (float) a.q[(size_t) (t * a.k + i)];
}

void qref_dequant_row(const QWeightView& w, int64_t r, float* out) {
    check_weight(w);
    const uint8_t* row = static_cast<const uint8_t*>(w.data) + (size_t) r * row_bytes(w);
    Decoded b;
    for (int64_t ib = 0; ib < w.k / kQK; ++ib) {
        decode(w.type, row + (size_t) ib * qtype_block_bytes(w.type), b);
        for (int v = 0; v < 256; ++v) {
            const int g = v / 16;
            // ggml's order: q2_K dl * q - ml with dl = d * sc, ml = dmin * m; q3_K and q6_K (d * sc) * q
            const float dl = b.d * (float) b.sc[g];
            out[ib * kQK + v] = w.type == kQTypeQ2K ? dl * (float) b.q[v] - b.dmin * (float) b.mn[g] : dl * (float) b.q[v];
        }
    }
}

void qref_matmul(const QWeightView& w, const QActsHost& a, int groups, float* y, int64_t ldy, double* abs_scale) {
    check_acts(w, a, groups);
    const int64_t rows_g = w.rows / groups;
    parallel_for(w.rows, a.T * w.rows * w.k, [&](int64_t r) {
        for (int64_t t = 0; t < a.T; ++t) {
            double v, s;
            dot_row(w, a, r, t, (r / rows_g) * w.k, v, s);
            y[t * ldy + r] = (float) v;
            if (abs_scale) abs_scale[t * ldy + r] = s;
        }
    });
}

namespace {

// column r of the dequant path: row r of w as f16 values against f16(x), output i of each row of y and abs_scale
void f16_column(const QWeightView& w, const float* x, int64_t ldx, int64_t T, int64_t a0, int64_t r, float* y,
                double* abs_scale, int64_t ldy, int64_t i) {
    std::vector<float> wr((size_t) w.k);
    qref_dequant_row(w, r, wr.data());
    for (float& v : wr) v = f16_round(v);
    for (int64_t t = 0; t < T; ++t) {
        const float* xr = x + t * ldx + a0;
        double acc = 0, abs = 0;
        for (int64_t k = 0; k < w.k; ++k) {
            const double p = (double) wr[(size_t) k] * (double) f16_round(xr[k]);
            acc += p;
            abs += std::fabs(p);
        }
        y[t * ldy + i] = (float) acc;
        if (abs_scale) abs_scale[t * ldy + i] = abs;
    }
}

void check_f16_call(const QWeightView& w, int64_t ldx, int groups) {
    check_weight(w);
    if (groups < 1 || w.rows % groups) failf("%lld rows in %d groups", (long long) w.rows, groups);
    if (ldx < (int64_t) groups * w.k) failf("activation stride %lld below %d x %lld", (long long) ldx, groups, (long long) w.k);
}

}  // namespace

void qref_matmul_f16(const QWeightView& w, const float* x, int64_t ldx, int64_t T, int groups, float* y, int64_t ldy,
                     double* abs_scale) {
    check_f16_call(w, ldx, groups);
    const int64_t rows_g = w.rows / groups;
    parallel_for(w.rows, T * w.rows * w.k, [&](int64_t r) { f16_column(w, x, ldx, T, (r / rows_g) * w.k, r, y, abs_scale, ldy, r); });
}

void qref_matmul_f16_cols(const QWeightView& w, const float* x, int64_t ldx, int64_t T, int groups,
                          const std::vector<int64_t>& cols, float* y, double* abs_scale) {
    check_f16_call(w, ldx, groups);
    const int64_t rows_g = w.rows / groups, n = (int64_t) cols.size();
    parallel_for(n, T * n * w.k, [&](int64_t i) {
        const int64_t r = cols[(size_t) i];
        if (r < 0 || r >= w.rows) failf("column %lld of %lld", (long long) r, (long long) w.rows);
        f16_column(w, x, ldx, T, (r / rows_g) * w.k, r, y, abs_scale, n, i);
    });
}

void qref_matmul_cols(const QWeightView& w, const QActsHost& a, int groups, const std::vector<int64_t>& cols, float* y,
                      double* abs_scale) {
    check_acts(w, a, groups);
    const int64_t rows_g = w.rows / groups, n = (int64_t) cols.size();
    parallel_for(n, a.T * n * w.k, [&](int64_t i) {
        const int64_t r = cols[(size_t) i];
        if (r < 0 || r >= w.rows) failf("column %lld of %lld", (long long) r, (long long) w.rows);
        for (int64_t t = 0; t < a.T; ++t) {
            double v, s;
            dot_row(w, a, r, t, (r / rows_g) * w.k, v, s);
            y[t * n + i] = (float) v;
            if (abs_scale) abs_scale[t * n + i] = s;
        }
    });
}

}  // namespace strata::dsv41::engine
