// src/dsv41/cpu_kernel_avx2.cpp - strata-moe's own CPU expert kernel (cpu_kernel.h): q2_K, q3_K, q4_K and q6_K
// weight rows against Q8_K activations, AVX2 + FMA + F16C (the dev VM's Ryzen 7 5700X and the A100 pods' EPYC
// 7713 are both Zen 3, no AVX-512). Built with those flags for this file only; strata-moe checks the CPU at run
// time (fast_kernel_cpu_ok) before it calls anything here.
//
// The fork's ggml-cpu is built for AVX2 (GGML_NATIVE=OFF, GGML_AVX2=ON), so its vec_dot for these types is the
// __AVX2__ branch of ggml/src/ggml-cpu/arch/x86/quants.c at 5210c7c5. Each kernel below is that branch with the
// loop over tokens moved inside the loop over blocks, as Strata's src/kernels/cpu/kq_avx2.cpp does for Q4_K:
// everything that depends only on the weights (the bit unpacking, the 4- and 6-bit scales and mins, the fp16
// scales) is done once per block, and every token routed to the expert (up to kTokens per pass) then runs ggml's
// integer chain and ggml's float operations in ggml's order, so each (row, token) is ggml's vec_dot to the bit and
// a row is read once for all of them. The tests (tests/dsv41/moe_cpu_fast_test.cpp) require the fast path to equal
// the reference bit for bit at 1 to 32 tokens.
//
// One row per pass, in memory order. Tiles of 4 rows through the same block index (one Q8_K block shared by 4
// weight streams) measured slower with dsv41_moe_cpu_bench on the dev VM's Ryzen 7 5700X, 1 token, 1 thread: 20.2
// ms per call against ref's 8.2 (16 experts), 14.7 against 7.1 with a software prefetch 6 blocks ahead in each
// stream (48 experts), where one row per pass gives 8.1 against 7.9; 32 tokens, 8 threads, 48 experts: 74.4 ms
// with 4-row tiles, 45.7 with one row (ref 58).
#include "cpu_kernel.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <algorithm>
#include <cstring>

namespace strata::dsv41 {
namespace {

// tokens per pass over a row (at most), as kq_avx2.cpp's verify window
constexpr int kTokens = 8;

// ggml_half is uint16_t in ggml-common.h's C++ declarations; GGML_CPU_FP16_TO_FP32 is this F16C conversion
inline float fp16(ggml_half h) { return _cvtsh_ss(h); }

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c, the AVX2 forms), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

alignas(32) const uint8_t k_shuffle_q3k[128] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
};
inline __m256i get_scale_shuffle_q3k(int i) { return _mm256_load_si256((const __m256i*) k_shuffle_q3k + i); }

alignas(32) const uint8_t k_shuffle_k4[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_load_si256((const __m256i*) k_shuffle_k4 + i); }

alignas(16) const uint8_t k_shuffle_q6k[128] = {
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
     8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,
    14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,
};
inline __m128i get_scale_shuffle(int i) { return _mm_load_si128((const __m128i*) k_shuffle_q6k + i); }

// Each kernel: Weights, the weight-side registers of one block of one row; prep, which fills them as ggml's loop
// body does before it touches the activation; Acc and dot, ggml's accumulation of one block for one token; finish,
// ggml's final reduction.

// ---- q2_K x Q8_K (ggml_vec_dot_q2_K_q8_K, __AVX2__)
struct Q2K {
    using Block = block_q2_K;
    struct Weights {
        __m256i q2[QK_K / 128][4];
        __m256i scale[QK_K / 128][4];
        __m256i mins;
        float xd, xdmin;
    };
    struct Acc {
        __m256 acc = _mm256_setzero_ps();
    };

    static void prep(const Block& x, Weights& w) {
        const __m256i m3 = _mm256_set1_epi8(3);
        const __m128i m4 = _mm_set1_epi8(0xF);
        // {d, dmin} is a union member named data.data in the C++ declarations
        w.xd = fp16(x.data.data.d);
        w.xdmin = fp16(x.data.data.dmin);
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*) x.scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        w.mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};
        const uint8_t* q2 = x.qs;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q2bits = _mm256_loadu_si256((const __m256i*) q2);
            q2 += 32;
            w.q2[j][0] = _mm256_and_si256(q2bits, m3);
            w.q2[j][1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            w.q2[j][2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            w.q2[j][3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);
            for (int k = 0; k < 4; ++k) w.scale[j][k] = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(k));
        }
    }

    static void dot(const Weights& w, const block_q8_K& y, Acc& a) {
        const float d = y.d * w.xd;
        const float dmin = -y.d * w.xdmin;
        const __m256i prod = _mm256_madd_epi16(w.mins, _mm256_loadu_si256((const __m256i*) y.bsums));
        a.acc = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), a.acc);
        const int8_t* q8 = y.qs;
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 128; ++j) {
            __m256i p[4];
            for (int k = 0; k < 4; ++k) {
                const __m256i q8k = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                p[k] = _mm256_madd_epi16(w.scale[j][k], _mm256_maddubs_epi16(w.q2[j][k], q8k));
            }
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), _mm256_add_epi32(p[2], p[3])));
        }
        a.acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), a.acc);
    }

    static float finish(const Acc& a) { return hsum_float_8(a.acc); }
};

// ---- q3_K x Q8_K (ggml_vec_dot_q3_K_q8_K, __AVX2__)
struct Q3K {
    using Block = block_q3_K;
    struct Weights {
        __m256i q3l[QK_K / 128][4];
        __m256i q3h[QK_K / 128][4];
        __m256i scale[QK_K / 128][4];
        float xd;
    };
    struct Acc {
        __m256 acc = _mm256_setzero_ps();
    };

    static void prep(const Block& x, Weights& w) {
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;
        const __m256i m3 = _mm256_set1_epi8(3);
        const __m256i mone = _mm256_set1_epi8(1);
        const __m128i m32 = _mm_set1_epi8(32);
        w.xd = fp16(x.d);
        uint32_t aux[3];
        std::memcpy(aux, x.scales, 12);
        __m128i scales128 = _mm_set_epi32(
            (int) (((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4)),
            (int) (((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4)),
            (int) ((aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4)),
            (int) ((aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4)));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x.hmask);
        const uint8_t* q3 = x.qs;
        int bit = 0;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*) q3);
            q3 += 32;
            for (int k = 0; k < 4; ++k) {
                w.q3l[j][k] = _mm256_and_si256(k == 0 ? q3bits : _mm256_srli_epi16(q3bits, 2 * k), m3);
                w.q3h[j][k] = _mm256_slli_epi16(
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                ++bit;
                // ggml's is stays 0: scales[j] holds the 8 scales of this half
                w.scale[j][k] = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(k));
            }
        }
    }

    static void dot(const Weights& w, const block_q8_K& y, Acc& a) {
        const float d = y.d * w.xd;
        const int8_t* q8 = y.qs;
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 128; ++j) {
            __m256i p[4];
            for (int k = 0; k < 4; ++k) {
                const __m256i q8k = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8s = _mm256_maddubs_epi16(w.q3h[j][k], q8k);
                const __m256i p16 = _mm256_sub_epi16(_mm256_maddubs_epi16(w.q3l[j][k], q8k), q8s);
                p[k] = _mm256_madd_epi16(w.scale[j][k], p16);
            }
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), _mm256_add_epi32(p[2], p[3])));
        }
        a.acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), a.acc);
    }

    static float finish(const Acc& a) { return hsum_float_8(a.acc); }
};

// ---- q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__)
struct Q4K {
    using Block = block_q4_K;
    struct Weights {
        __m256i q4l[QK_K / 64], q4h[QK_K / 64];
        __m256i scale_l[QK_K / 64], scale_h[QK_K / 64];
        __m128i mins;
        float xd, xdmin;
    };
    struct Acc {
        __m256 acc = _mm256_setzero_ps();
        __m128 acc_m = _mm_setzero_ps();
    };

    static void prep(const Block& x, Weights& w) {
        static const uint32_t kmask1 = 0x3f3f3f3f;
        static const uint32_t kmask2 = 0x0f0f0f0f;
        static const uint32_t kmask3 = 0x03030303;
        const __m256i m4 = _mm256_set1_epi8(0xF);
        w.xd = fp16(x.data.data.d);
        w.xdmin = fp16(x.data.data.dmin);
        uint32_t utmp[4];
        std::memcpy(utmp, x.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
        w.mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);
        const uint8_t* q4 = x.qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            w.scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            w.scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            w.q4l[j] = _mm256_and_si256(q4bits, m4);
            w.q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
    }

    static void dot(const Weights& w, const block_q8_K& y, Acc& a) {
        const float d = y.d * w.xd;
        const float dmin = -y.d * w.xdmin;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y.bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(w.mins, q8s);
        a.acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), a.acc_m);
        const int8_t* q8 = y.qs;
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
            q8 += 32;
            const __m256i p16l = _mm256_madd_epi16(w.scale_l[j], _mm256_maddubs_epi16(w.q4l[j], q8l));
            const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
            q8 += 32;
            const __m256i p16h = _mm256_madd_epi16(w.scale_h[j], _mm256_maddubs_epi16(w.q4h[j], q8h));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }
        a.acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), a.acc);
    }

    static float finish(const Acc& a) {
        __m128 acc_m = a.acc_m;
        acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
        return hsum_float_8(a.acc) + _mm_cvtss_f32(acc_m);
    }
};

// ---- q6_K x Q8_K (ggml_vec_dot_q6_K_q8_K, __AVX2__)
struct Q6K {
    using Block = block_q6_K;
    struct Weights {
        __m256i q4[QK_K / 128][4];
        __m256i scale[QK_K / 128][4];
        __m256i scales_16;
        float xd;
    };
    struct Acc {
        __m256 acc = _mm256_setzero_ps();
    };

    static void prep(const Block& x, Weights& w) {
        const __m256i m3 = _mm256_set1_epi8(3);
        const __m256i m15 = _mm256_set1_epi8(15);
        w.xd = fp16(x.d);
        const __m128i scales = _mm_loadu_si128((const __m128i*) x.scales);
        w.scales_16 = _mm256_cvtepi8_epi16(scales);
        const uint8_t* q4 = x.ql;
        const uint8_t* qh = x.qh;
        int is = 0;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) qh);
            qh += 32;
            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);
            w.q4[j][0] = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            w.q4[j][1] = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            w.q4[j][2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            w.q4[j][3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
            for (int k = 0; k < 4; ++k) {
                w.scale[j][k] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + k)));
            }
            is += 4;
        }
    }

    static void dot(const Weights& w, const block_q8_K& y, Acc& a) {
        const float d = y.d * w.xd;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y.bsums);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, w.scales_16), 5);
        const int8_t* q8 = y.qs;
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 128; ++j) {
            __m256i p[4];
            for (int k = 0; k < 4; ++k) {
                const __m256i q8k = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                p[k] = _mm256_madd_epi16(w.scale[j][k], _mm256_maddubs_epi16(w.q4[j][k], q8k));
            }
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[0], p[1]));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[2], p[3]));
        }
        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        a.acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), a.acc);
    }

    static float finish(const Acc& a) { return hsum_float_8(a.acc); }
};

static_assert(sizeof(block_q8_K) == 292, "block_q8_K is not ggml's");

// Row r for m <= NT tokens from t0: one pass over the row's blocks, every token per block
template <class K, int NT>
void row_tokens(const char* w, size_t row_bytes, int64_t r, int64_t nb, const void* const* acts, float* const* outs,
                int64_t t0, int m) {
    typename K::Acc acc[NT];
    typename K::Weights wt;
    const auto* x = reinterpret_cast<const typename K::Block*>(w + (size_t) r * row_bytes);
    for (int64_t i = 0; i < nb; ++i) {
        K::prep(x[i], wt);
        for (int t = 0; t < m; ++t) K::dot(wt, static_cast<const block_q8_K*>(acts[t0 + t])[i], acc[t]);
    }
    for (int t = 0; t < m; ++t) outs[t0 + t][r] = K::finish(acc[t]);
}

template <class K>
void rows(const char* w, size_t row_bytes, int64_t r0, int64_t r1, int64_t nb, const void* const* acts,
          float* const* outs, int64_t n_acts) {
    for (int64_t t0 = 0; t0 < n_acts; t0 += kTokens) {
        const int m = (int) std::min<int64_t>(kTokens, n_acts - t0);
        for (int64_t r = r0; r < r1; ++r) {
            // decode, one token: its accumulator stays in registers
            if (m == 1) {
                row_tokens<K, 1>(w, row_bytes, r, nb, acts, outs, t0, 1);
            } else {
                row_tokens<K, kTokens>(w, row_bytes, r, nb, acts, outs, t0, m);
            }
        }
    }
}

}  // namespace

bool fast_kernel_supports(ggml_type type) {
    return type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K;
}

void fast_kernel_rows(ggml_type type, const char* w, size_t row_bytes, int64_t r0, int64_t r1, int64_t n_per_row,
                      const void* const* acts, float* const* outs, int64_t n_acts) {
    const int64_t nb = n_per_row / QK_K;
    switch (type) {
        case GGML_TYPE_Q2_K: rows<Q2K>(w, row_bytes, r0, r1, nb, acts, outs, n_acts); break;
        case GGML_TYPE_Q3_K: rows<Q3K>(w, row_bytes, r0, r1, nb, acts, outs, n_acts); break;
        case GGML_TYPE_Q4_K: rows<Q4K>(w, row_bytes, r0, r1, nb, acts, outs, n_acts); break;
        case GGML_TYPE_Q6_K: rows<Q6K>(w, row_bytes, r0, r1, nb, acts, outs, n_acts); break;
        default: GGML_ABORT("strata-moe: the fast CPU kernel has no %s rows", ggml_type_name(type));
    }
}

}  // namespace strata::dsv41
