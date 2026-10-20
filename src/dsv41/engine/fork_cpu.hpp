// src/dsv41/engine/fork_cpu.hpp - the stock fork's CPU arithmetic, op by op, for the engine's CPU references
// (indexer_ref.cpp, attention_ref.cpp). Internal to src/dsv41/engine; every function here reproduces one ggml-cpu
// routine of the fork (b4d9cb1, built with GGML_NATIVE=OFF: AVX2 + FMA + F16C, tools/dsv41/build-inside.sh) in the
// order it rounds, so a reference built on them gives the fork's op dumps bit for bit.
//
// One __m256 lane is one array element here and std::fma is _mm256_fmadd_ps on that lane. The files that include this
// are compiled with -mfma -ffp-contract=off (CMakeLists.txt), so nothing is fused unless written as std::fma: the
// fork's own scalar code (gnu++17, -mfma) is contracted by GCC where its disassembly shows it, and those places are
// written out as std::fma below with the instruction they stand for.
#pragma once

#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::dsv41::engine::fork_cpu {

constexpr int QK_K = 256;

/// block_q2_K (ggml-common.h): 16 scale bytes (low nibble scale, high nibble min), 64 bytes of 2-bit quants, d, dmin
struct BlockQ2K {
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
};
static_assert(sizeof(BlockQ2K) == 84, "block_q2_K is 84 bytes");

/// block_q8_K: the activation side of a K-quant dot (vec_dot_type of q2_K)
struct BlockQ8K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};

[[noreturn]] inline void fail(const char* who, const char* what) {
    std::fprintf(stderr, "%s: %s\n", who, what);
    std::exit(1);
}

inline float f16_round(float x) { return kernels::f32_from_f16(kernels::f16_from_f32(x)); }

/// The horizontal sum of one __m256 that tinyBLAS (hsum), hsum_float_8 and ggml_vec_soft_max_f32 use:
/// (hi128 + lo128), then + movehl, then + movehdup.
inline float hsum8(const float* lane) {
    const float x0 = lane[4] + lane[0], x1 = lane[5] + lane[1], x2 = lane[6] + lane[2], x3 = lane[7] + lane[3];
    const float y0 = x0 + x2, y1 = x1 + x3;
    return y0 + y1;
}

/// tinyBLAS<8, __m256> f16 x f16 (sgemm.cpp, gemm_bloc): one accumulator per output, lane m over k = m, m + 8, ...,
/// then hsum. The mul_mat path of a ubatch of 2 or more tokens (llamafile_sgemm refuses n < 2). n % 8 == 0.
inline float tinyblas_dot(const float* a, const float* b, int64_t n) {
    float lane[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int64_t l = 0; l < n; l += 8)
        for (int m = 0; m < 8; ++m) lane[m] = std::fma(a[l + m], b[l + m], lane[m]);
    return hsum8(lane);
}

/// ggml_vec_dot_f16 on AVX2 (vec.cpp): four 8-lane accumulators over steps of 32, reduced 0 + 2 and 1 + 3, then
/// 0 + 1, then (lo128 + hi128) and two hadds. The single-token ubatch path. n % 32 == 0 (no leftovers: every caller
/// in the engine has rows of 128, 256 or 512 values, or cells padded to 256).
inline float vecdot_f16(const float* a, const float* b, int64_t n) {
    float acc[4][8] = {};
    for (int64_t i = 0; i < n; i += 32)
        for (int j = 0; j < 4; ++j)
            for (int m = 0; m < 8; ++m) acc[j][m] = std::fma(a[i + j * 8 + m], b[i + j * 8 + m], acc[j][m]);
    float x[8];
    for (int m = 0; m < 8; ++m) {
        const float s02 = acc[0][m] + acc[2][m], s13 = acc[1][m] + acc[3][m];
        x[m] = s02 + s13;
    }
    const float t0 = x[0] + x[4], t1 = x[1] + x[5], t2 = x[2] + x[6], t3 = x[3] + x[7];
    return (t0 + t1) + (t2 + t3);
}

/// The f16 x f16 dot of the fork's mul_mat for a ubatch of `ubatch_tokens` tokens (which picks the kernel).
inline float f16_dot(const float* a, const float* b, int64_t n, int64_t ubatch_tokens) {
    return ubatch_tokens >= 2 ? tinyblas_dot(a, b, n) : vecdot_f16(a, b, n);
}

/// ggml_v_expf for one __m256 lane (vec.h, AVX2 + FMA branch), including the special-case path for |n| > 126 that a
/// -inf input (a masked score) takes: it returns 0 there.
inline float v_expf(float x) {
    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    auto flt = [](uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; };
    const float r = 0x1.8p23f;
    const float z = std::fma(x, 0x1.715476p+0f, r);
    const float n = z - r;
    const float b = std::fma(-n, 0x1.7f7d1cp-20f, std::fma(-n, 0x1.62e4p-1f, x));
    const uint32_t e = bits(z) << 23;
    const float k = flt(e + bits(1.0f));
    const bool c = std::fabs(n) > 126.f;
    const float u = b * b;
    const float j = std::fma(std::fma(std::fma(0x1.0e4020p-7f, b, 0x1.573e2ep-5f), u, std::fma(0x1.555e66p-3f, b, 0x1.fffdb6p-2f)),
                             u, 0x1.ffffecp-1f * b);
    if (!c) return std::fma(j, k, k);
    const uint32_t g = n <= 0.f ? 0x82000000u : 0u;
    const float s1 = flt(g + 0x7f000000u);
    const float s2 = flt(e - g);
    const bool d = std::fabs(n) > 192.f;
    if (d) return s1 * s1;
    return std::fma(s2, j, s2) * s1;
}

/// ggml_compute_forward_soft_max_f32 on one row of n values (ops.cpp 5584-5694): wp = x * scale (+ mask), max (and
/// the sink), the exps in groups of 8 by ggml_v_expf each summed by hsum8 into a double, the tail by expf, plus
/// expf(sink - max), then y *= (float) (1 / sum). `mask` and `sink` may be null.
inline void soft_max_row(const float* x, int64_t n, float scale, const float* mask, const float* sink, float* y,
                         float* wp) {
    for (int64_t i = 0; i < n; ++i) wp[i] = x[i] * scale;
    if (mask)
        for (int64_t i = 0; i < n; ++i) wp[i] += mask[i];   // slope 1: slope*mask is exact, fused or not
    float max = -INFINITY;
    for (int64_t i = 0; i < n; ++i) max = max > wp[i] ? max : wp[i];
    if (sink) max = max > *sink ? max : *sink;
    double sum = 0.0;
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        float v[8];
        for (int m = 0; m < 8; ++m) {
            v[m] = v_expf(wp[i + m] - max);
            y[i + m] = v[m];
        }
        sum += (double) hsum8(v);
    }
    for (; i < n; ++i) {
        const float v = std::exp(wp[i] - max);
        sum += (double) v;
        y[i] = v;
    }
    if (sink) sum += (double) std::exp(*sink - max);
    const float inv = (float) (1.0 / sum);
    for (int64_t k = 0; k < n; ++k) y[k] *= inv;
}

/// ggml_compute_forward_rms_norm_f32 followed by the weight multiply (build_norm, LLM_NORM_RMS): the squares summed
/// in double, mean and scale in float, y = x * scale * w.
inline void rms_norm_mul(const float* x, int64_t n, const float* w, float eps, float* y) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += (double) (x[i] * x[i]);
    const float mean = (float) (sum / (double) n);
    const float scale = 1.0f / std::sqrt(mean + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];
}

/// The fork's rope configuration of one layer kind (deepseek41.cpp rope_cfg, llama-context.cpp ext_factor), with the
/// per-call constants ggml derives from it (ops.cpp ggml_compute_forward_rope_flt): theta_scale = powf(base, -2/R)
/// and the YaRN correction dims of ggml_rope_yarn_corr_dims, all in float as ggml computes them.
struct RopeCfg {
    float base = 10000.f;
    float freq_scale = 1.f;
    float ext_factor = 0.f;
    float attn_factor = 1.f;
    float beta_fast = 0.f;
    float beta_slow = 0.f;
    int n_ctx_orig = 0;
    int n_rot = 64;
    // derived
    float theta_scale = 0.f;
    float corr0 = 0.f, corr1 = 0.f;
};

/// ggml_rope_yarn_corr_dim (ggml.c), float
inline float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return (float) n_dims * logf((float) n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base));
}

inline RopeCfg rope_cfg_finish(RopeCfg c) {
    c.theta_scale = powf(c.base, -2.0f / (float) c.n_rot);
    const float start = floorf(yarn_corr_dim(c.n_rot, c.n_ctx_orig, c.beta_fast, c.base));
    const float end = ceilf(yarn_corr_dim(c.n_rot, c.n_ctx_orig, c.beta_slow, c.base));
    c.corr0 = start > 0.f ? start : 0.f;
    c.corr1 = end < (float) (c.n_rot - 1) ? end : (float) (c.n_rot - 1);
    return c;
}

/// dsv4_rope_attn_factor (deepseek4.cpp; libllama is built without -mfma, so nothing is fused)
inline float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) return 1.0f;
    const float t = 0.1f * logf(1.0f / freq_scale);
    return 1.0f / (1.0f + t);
}

/// ggml_rope_cache_init + rope_yarn for one position: cache[2i], cache[2i + 1] = cos, sin of pair i (sin times
/// sin_sign: -1 is ggml_rope_ext_back). The disassembly of the fork's libggml-cpu.so (rope_flt<float>) shows the
/// contractions: theta = fma(theta_interp, 1 - ramp_mix, theta_extrap * ramp_mix) (vfmadd132ss) and the log term
/// fma(0.1f, logf(1 / freq_scale), 1) (vfmadd231ss); cos and sin come from one sincosf.
inline void rope_cache(const RopeCfg& c, int64_t pos, float sin_sign, float* cache) {
    float theta = (float) pos;
    for (int i0 = 0; i0 < c.n_rot; i0 += 2) {
        const float theta_extrap = theta;
        const float theta_interp = c.freq_scale * theta_extrap;
        float th = theta_interp;
        float mscale = c.attn_factor;
        if (c.ext_factor != 0.0f) {
            const float y = ((float) (i0 / 2) - c.corr0) / (0.001f > c.corr1 - c.corr0 ? 0.001f : c.corr1 - c.corr0);
            const float ramp = 1.f - (1.f < (0.f > y ? 0.f : y) ? 1.f : (0.f > y ? 0.f : y));
            const float ramp_mix = ramp * c.ext_factor;
            th = std::fma(theta_interp, 1.f - ramp_mix, theta_extrap * ramp_mix);
            mscale *= std::fma(0.1f, logf(1.0f / c.freq_scale), 1.0f);
        }
        float s, co;
        sincosf(th, &s, &co);
        cache[i0] = co * mscale;
        cache[i0 + 1] = s * mscale * sin_sign;
        theta *= c.theta_scale;
    }
}

/// rotate_pairs, GGML_ROPE_TYPE_NORMAL (adjacent pairs), on the n_rot values at x: per pair (x0, x1)
/// x0' = fma(x0, cos, -(x1 * sin)) (vfmsub231ss), x1' = fma(x0, sin, x1 * cos) (vfmadd132ss), as the fork's
/// disassembly computes them (the op dumps decide between the operand orders it admits).
inline void rope_rotate(const float* cache, int n_rot, float* x) {
    for (int i0 = 0; i0 < n_rot; i0 += 2) {
        const float co = cache[i0], si = cache[i0 + 1];
        const float x0 = x[i0], x1 = x[i0 + 1];
        x[i0] = std::fma(x0, co, -(x1 * si));
        x[i0 + 1] = std::fma(x0, si, x1 * co);
    }
}

/// ggml_compute_forward_fwht_f32 (ops.cpp 11990-12064), the CPU form of llama_mul_mat_hadamard: x * (1 / sqrtf(n)),
/// then the butterflies (u + v, u - v) for len 1, 2, 4 scalar and 8.. n/2 as fma(v, -1, u), which rounds as u - v.
inline void fwht(float* x, int n) {
    const float scale = 1.0f / std::sqrt((float) n);
    for (int j = 0; j < n; ++j) x[j] *= scale;
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += 2 * len)
            for (int j = 0; j < len; ++j) {
                const float u = x[i + j], v = x[i + len + j];
                x[i + j] = u + v;
                x[i + len + j] = u - v;
            }
}

/// quantize_row_q8_K_ref (ggml-quants.c, built without -mfma): per block of 256 the value of largest magnitude,
/// iscale = -127 / max, round to nearest (the 12582912 trick), the 16-value sums, d = 1 / iscale.
inline void quantize_q8_K(const float* x, int64_t n, BlockQ8K* y) {
    if (n % QK_K) fail("fork_cpu", "quantize_q8_K: a row is not whole blocks of 256");
    for (int64_t b = 0; b < n / QK_K; ++b, x += QK_K) {
        float max = 0, amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            const float ax = std::fabs(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (!amax) {
            y[b].d = 0;
            std::memset(y[b].qs, 0, QK_K);
            std::memset(y[b].bsums, 0, sizeof y[b].bsums);
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            const float val = iscale * x[j] + 12582912.f;
            int32_t i;
            std::memcpy(&i, &val, 4);
            const int v = (i & 0x007fffff) - 0x00400000;
            y[b].qs[j] = (int8_t) (v < 127 ? v : 127);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[b].qs[j * 16 + ii];
            y[b].bsums[j] = (int16_t) sum;
        }
        y[b].d = 1 / iscale;
    }
}

/// ggml_vec_dot_q2_K_q8_K, AVX2 branch (arch/x86/quants.c 1574-1650): 8 float lanes; per block lane k first takes
/// fma(-y.d * dmin, mins[2k] * bsums[2k] + mins[2k + 1] * bsums[2k + 1], .) and then fma(y.d * d, sumi_k, .), where
/// sumi_k is the integer sum of the block's products of bytes 4k..4k+3 of every 32-byte quant chunk (scale of its
/// 16-value group applied); then hsum_float_8.
inline float vec_dot_q2_K_q8_K(const BlockQ2K* x, const BlockQ8K* y, int64_t n) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = y[i].d * kernels::f32_from_f16(x[i].d);
        const float dmin = -y[i].d * kernels::f32_from_f16(x[i].dmin);
        int32_t prod[8], sumi[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < 8; ++k)
            prod[k] = (int32_t) (x[i].scales[2 * k] >> 4) * y[i].bsums[2 * k] +
                      (int32_t) (x[i].scales[2 * k + 1] >> 4) * y[i].bsums[2 * k + 1];
        for (int j = 0; j < QK_K / 128; ++j)
            for (int l = 0; l < 4; ++l)
                for (int byte = 0; byte < 32; ++byte) {
                    const int e = 128 * j + 32 * l + byte;
                    const int q2 = (x[i].qs[32 * j + byte] >> (2 * l)) & 3;
                    const int sc = x[i].scales[8 * j + 2 * l + (byte >= 16)] & 0xF;
                    sumi[byte / 4] += sc * q2 * y[i].qs[e];
                }
        for (int k = 0; k < 8; ++k) acc[k] = std::fma(dmin, (float) prod[k], acc[k]);
        for (int k = 0; k < 8; ++k) acc[k] = std::fma(d, (float) sumi[k], acc[k]);
    }
    return hsum8(acc);
}

/// The value and scale of every element of one q3_K or q6_K block (ggml-common.h layouts, dequantize_row_q3_K /
/// _q6_K order): q3_K q in -4..3 and its 6-bit scale minus 32; q6_K q in 0..63 unsigned (the - 32 is the dot's bsums
/// term) and its int8 scale.
struct KBlock {
    int q[QK_K];
    int sc[QK_K / 16];
    float d;
};
inline void decode_q3_K(const uint8_t* b, KBlock& o) {
    const uint8_t* hm = b;        // hmask[32]
    const uint8_t* qs = b + 32;   // qs[64]
    uint32_t aux[4];
    std::memcpy(aux, b + 96, 12);
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* s8 = reinterpret_cast<const int8_t*>(aux);
    for (int g = 0; g < QK_K / 16; ++g) o.sc[g] = s8[g] - 32;
    uint16_t dh;
    std::memcpy(&dh, b + 108, 2);
    o.d = kernels::f32_from_f16(dh);
    for (int n = 0; n < 2; ++n)
        for (int j = 0; j < 4; ++j)
            for (int l = 0; l < 32; ++l)
                o.q[128 * n + 32 * j + l] = ((qs[32 * n + l] >> (2 * j)) & 3) - ((hm[l] & (1 << (4 * n + j))) ? 0 : 4);
}
inline void decode_q6_K(const uint8_t* b, KBlock& o) {
    const uint8_t* ql = b;          // ql[128]
    const uint8_t* qh = b + 128;    // qh[64]
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
    for (int g = 0; g < QK_K / 16; ++g) o.sc[g] = sc[g];
    uint16_t dh;
    std::memcpy(&dh, b + 208, 2);
    o.d = kernels::f32_from_f16(dh);
    for (int n = 0; n < 2; ++n)
        for (int l = 0; l < 32; ++l) {
            const uint8_t* L = ql + 64 * n;
            const uint8_t h = qh[32 * n + l];
            o.q[128 * n + l] = (L[l] & 0xF) | ((h >> 0) & 3) << 4;
            o.q[128 * n + l + 32] = (L[l + 32] & 0xF) | ((h >> 2) & 3) << 4;
            o.q[128 * n + l + 64] = (L[l] >> 4) | ((h >> 4) & 3) << 4;
            o.q[128 * n + l + 96] = (L[l + 32] >> 4) | ((h >> 6) & 3) << 4;
        }
}

/// ggml_vec_dot_q3_K_q8_K and ggml_vec_dot_q6_K_q8_K, AVX2 branches (arch/x86/quants.c 1766-1884, 2426-2502): 8
/// int32 lanes, lane k holding bytes 4k..4k+3 of every 32-value chunk (maddubs then madd pair neighbours), each
/// product times its 16-value scale; q6_K then subtracts 32 (bsums[2k] sc[2k] + bsums[2k+1] sc[2k+1]) from lane k
/// (q8sclsub). Per block acc_k = fma(y.d * d, (float) lane_k, acc_k), then hsum_float_8. q3_K: type 11, q6_K: 14.
inline float vec_dot_qk_q8_K(int type, const uint8_t* x, const BlockQ8K* y, int64_t n) {
    const size_t bb = type == 11 ? 110 : 210;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    KBlock blk;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        if (type == 11) decode_q3_K(x + (size_t) i * bb, blk);
        else decode_q6_K(x + (size_t) i * bb, blk);
        const float d = y[i].d * blk.d;
        int32_t lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int e = 0; e < QK_K; ++e) lane[(e % 32) / 4] += blk.sc[e / 16] * blk.q[e] * y[i].qs[e];
        if (type == 14)
            for (int k = 0; k < 8; ++k)
                lane[k] -= 32 * (y[i].bsums[2 * k] * blk.sc[2 * k] + y[i].bsums[2 * k + 1] * blk.sc[2 * k + 1]);
        for (int k = 0; k < 8; ++k) acc[k] = std::fma(d, (float) lane[k], acc[k]);
    }
    return hsum8(acc);
}

/// GGML_FP32_TO_BF16 (ggml-impl.h): round to nearest even, a NaN kept quiet. bf16 bits to f32: the high half.
inline uint16_t bf16_from_f32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t) ((u >> 16) | 64);
    return (uint16_t) ((u + (0x7fff + ((u >> 16) & 1))) >> 16);
}
inline float f32_from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// ggml_vec_dot_bf16, AVX2 branch (vec.cpp 172-196): four 8-lane accumulators over steps of 32, each step
/// acc = x y + acc (a bf16 x bf16 product is exact in f32, so a contracted FMA rounds the same), reduced
/// (0 + 2) + (1 + 3), then hsum_float_8. The single-token path (tinyBLAS takes n >= 2, as tinyblas_dot).
/// n % 32 == 0.
inline float vecdot_bf16(const float* a, const float* b, int64_t n) {
    float acc[4][8] = {};
    for (int64_t i = 0; i < n; i += 32)
        for (int j = 0; j < 4; ++j)
            for (int m = 0; m < 8; ++m) acc[j][m] = std::fma(a[i + j * 8 + m], b[i + j * 8 + m], acc[j][m]);
    float x[8];
    for (int m = 0; m < 8; ++m) x[m] = (acc[0][m] + acc[2][m]) + (acc[1][m] + acc[3][m]);
    return hsum8(x);
}

/// ggml's unary sigmoid and softplus on the CPU (unary-ops.cpp op_sigmoid, op_softplus)
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : std::log(1.0f + std::exp(x)); }

/// ggml_compute_forward_rms_norm_f32 without the weight: squares in double, y = x * scale
inline void rms_norm(const float* x, int64_t n, float eps, float* y) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += (double) (x[i] * x[i]);
    const float mean = (float) (sum / (double) n);
    const float scale = 1.0f / std::sqrt(mean + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale;
}

/// ggml_vec_sum_f32 (sum_rows): a double sum, rounded to f32
inline float sum_f32(const float* x, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += (double) x[i];
    return (float) s;
}

/// dequantize_row_q2_K (ggml-quants.c): y = d * sc * q - dmin * m per 16-value group
inline void dequant_q2_K(const BlockQ2K* x, int64_t n, float* y) {
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = kernels::f32_from_f16(x[i].d), mn = kernels::f32_from_f16(x[i].dmin);
        const uint8_t* q = x[i].qs;
        int is = 0;
        for (int nn = 0; nn < QK_K; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (float) (sc & 0xF), ml = mn * (float) (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (float) ((q[l] >> shift) & 3) - ml;
                sc = x[i].scales[is++];
                dl = d * (float) (sc & 0xF);
                ml = mn * (float) (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (float) ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

}  // namespace strata::dsv41::engine::fork_cpu
