// src/dsv41/engine/experts_cpu_mmvq.cpp - the CPU experts in the GPU's numerics of experts_cpu_mmvq.hpp, on AVX2.
// Built with -mavx2 -mfma -mf16c -ffp-contract=off (CMakeLists.txt); experts_cpu.cpp asks fast_kernel_cpu_ok first.
//
// A row's value is mmvq_kernel's at one column (mmvq_ref.cpp dot): 128 threads, thread tid = 16 j + 8 n + k (j = tid /
// 16 the superblocks kbx = j, j + 8, ...; n = iqs / 8, k = iqs % 8) running tmp += apply(superblock kbx). Here thread
// tid is lane k of the accumulator vector 2 j + n, so one superblock's two vectors (n = 0, 1) hold its 16 threads'
// apply() values, lane k of vector n being thread iqs = 8 n + k's:
//   q2_K apply: for i = 0 .. 3 (q8_1 block 4 n + i of the superblock) sumf_d = fma(d8, dp * sc, sumf_d), sumf_m =
//      fma(d8, m * sum4(x), sumf_m), dp the dot of the thread's 4 weights and 4 activations (values 32 (4 n + i) + 4 k
//      .. + 3); then fma(dm.x, sumf_d, -(dm.y sumf_m)). The 8 lanes' dp are one maddubs and one madd (by the lane's
//      scale: sc (dp_lo + dp_hi) is sc dp, exactly) over the block's 32 bytes, in lane order k;
//   q3_K apply: the same with signed weights (maddubs of |w| and sign(x, w)), sumf = fma(d8, dp * sc, sumf), then
//      d3 sumf.
// The integer sums are exact and their float conversions too (|dp sc| < 2^24), so every float operation is the
// kernel's, in its order; then the warps' partials ((w0 + w1) + w2) + w3 and the xor butterfly, lane 0's value.
#include "experts_cpu_mmvq.hpp"

#include "expert_act.hpp"
#include "strata/dsv41/engine/mmvq_ref.hpp"
#include "strata/dsv41/engine/qmatmul.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::dsv41::engine::mmvq_cpu {
namespace {

constexpr size_t kQ81 = kQ81BlockBytes;
constexpr int kGroup = 4;   // activations per pass over a row's superblocks
#define DSV41_INLINE inline __attribute__((always_inline))
constexpr float kHalfMax = 65504.0f;

// an f16 to f32, exact (F16C; kernels::f32_from_f16's value for every finite input)
float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _cvtsh_ss(h);
}
float q8_1_finite(float v) { return std::fabs(v) > kHalfMax ? std::copysign(kHalfMax, v) : v; }

// lane 0 of mmvq.cuh warp_sum over the 32 lanes of v0 .. v3 (lanes 8 j .. 8 j + 7 in vj): lane l adds lane l ^ 16
// (v0 + v2, v1 + v3), then l ^ 8, l ^ 4, l ^ 2, l ^ 1 (addition commutes, so each pair's order is either lane's)
float butterfly(__m256 v0, __m256 v1, __m256 v2, __m256 v3) {
    const __m256 b = _mm256_add_ps(_mm256_add_ps(v0, v2), _mm256_add_ps(v1, v3));
    __m128 c = _mm_add_ps(_mm256_castps256_ps128(b), _mm256_extractf128_ps(b, 1));
    c = _mm_add_ps(c, _mm_movehl_ps(c, c));
    c = _mm_add_ss(c, _mm_movehdup_ps(c));
    return _mm_cvtss_f32(c);
}

// mmvq_kernel's end for one row and activation: acc[v] lane k is thread 8 v + k's sum; warp w is acc[4 w .. 4 w + 3]
float reduce(const __m256* acc) {
    __m256 s[4];
    for (int j = 0; j < 4; ++j) s[j] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(acc[j], acc[4 + j]), acc[8 + j]), acc[12 + j]);
    return butterfly(s[0], s[1], s[2], s[3]);
}

// one q8_1 block of 32 values: q8_1_warp's (mmvq_ref_quantize); false for a block the vector path does not take (a
// NaN among the values, or a scale that underflows to 0), which mmvq_ref_quantize then quantizes
bool quantize_block(const float* x, uint8_t* out) {
    __m256 v[4];
    for (int j = 0; j < 4; ++j) v[j] = _mm256_loadu_ps(x + 8 * j);
    const __m256 nan = _mm256_or_ps(_mm256_or_ps(_mm256_cmp_ps(v[0], v[0], _CMP_UNORD_Q), _mm256_cmp_ps(v[1], v[1], _CMP_UNORD_Q)),
                                    _mm256_or_ps(_mm256_cmp_ps(v[2], v[2], _CMP_UNORD_Q), _mm256_cmp_ps(v[3], v[3], _CMP_UNORD_Q)));
    if (_mm256_movemask_ps(nan)) return false;
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v[0]), _mm256_andnot_ps(sign, v[1])),
                             _mm256_max_ps(_mm256_andnot_ps(sign, v[2]), _mm256_andnot_ps(sign, v[3])));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    const float amax = _mm_cvtss_f32(m4);
    const float sum = butterfly(v[0], v[1], v[2], v[3]);
    const float d = q8_1_finite(amax / 127.0f);
    if (amax != 0.0f && !(d > 0.0f)) return false;
    alignas(32) int8_t qs[32];
    if (amax == 0.0f) {
        std::memset(qs, 0, sizeof qs);
    } else {
        const __m256 dv = _mm256_set1_ps(d), half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
        const __m256 lo = _mm256_set1_ps(-127.0f), hi = _mm256_set1_ps(127.0f);
        __m256i q[4];
        for (int j = 0; j < 4; ++j) {
            // roundf(x / d): the quotient's integer part, one away from zero when the rest is half or more
            const __m256 f = _mm256_div_ps(v[j], dv);
            const __m256 t = _mm256_round_ps(f, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            const __m256 rest = _mm256_andnot_ps(sign, _mm256_sub_ps(f, t));
            const __m256 away = _mm256_and_ps(_mm256_cmp_ps(rest, half, _CMP_GE_OQ), _mm256_or_ps(_mm256_and_ps(f, sign), one));
            const __m256 r = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(t, away), lo), hi);
            q[j] = _mm256_cvtps_epi32(r);
        }
        const __m256i b = _mm256_packs_epi16(_mm256_packs_epi32(q[0], q[1]), _mm256_packs_epi32(q[2], q[3]));
        _mm256_store_si256(reinterpret_cast<__m256i*>(qs), _mm256_permutevar8x32_epi32(b, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
    }
    const uint16_t dh = kernels::f16_from_f32(d), sh = kernels::f16_from_f32(q8_1_finite(sum));
    std::memcpy(out, &dh, 2);
    std::memcpy(out + 2, &sh, 2);
    std::memcpy(out + 4, qs, 32);
    return true;
}

// expert_act.hpp expf_det on 8 lanes, operation for operation
__m256 expf_det8(__m256 x) {
    const __m256 magic = _mm256_set1_ps(12582912.0f);
    const __m256 n = _mm256_sub_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504f), magic), magic);
    __m256 r = _mm256_fmadd_ps(n, _mm256_set1_ps(-0.693359375f), x);
    r = _mm256_fmadd_ps(n, _mm256_set1_ps(2.12194440e-4f), r);
    const __m256 z = _mm256_mul_ps(r, r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    __m256 y = _mm256_add_ps(_mm256_fmadd_ps(p, z, r), _mm256_set1_ps(1.0f));
    __m256i ni = _mm256_cvtps_epi32(n);   // n is a whole number (garbage in the lanes the blends below replace)
    const __m256i big = _mm256_cmpgt_epi32(ni, _mm256_set1_epi32(127));
    y = _mm256_blendv_ps(y, _mm256_mul_ps(y, _mm256_set1_ps(2.0f)), _mm256_castsi256_ps(big));
    ni = _mm256_min_epi32(ni, _mm256_set1_epi32(127));
    __m256 e = _mm256_mul_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23)));
    e = _mm256_blendv_ps(e, _mm256_setzero_ps(), _mm256_cmp_ps(x, _mm256_set1_ps(-87.33654f), _CMP_NGT_UQ));
    e = _mm256_blendv_ps(e, _mm256_set1_ps(INFINITY), _mm256_cmp_ps(x, _mm256_set1_ps(88.72284f), _CMP_GT_OQ));
    return _mm256_blendv_ps(e, x, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
}

// The shuffle masks of a superblock's scale bytes (16 of them, in both 128-bit halves) for q8_1 block b = 4 n + i:
// the i16 lanes 0 .. 7 (dot lanes k 0 .. 3) take byte 8 n + 2 i, lanes 8 .. 15 (k 4 .. 7) byte 8 n + 2 i + 1, each
// zero-extended (q2_32: as i32 lanes). q3_K's: the same i16 lanes from a half's 8 i16 scales (bytes 4 i, 4 i + 1,
// then 4 i + 2, 4 i + 3).
struct Masks {
    alignas(32) uint8_t q2[8][32];
    alignas(32) uint8_t q2_32[8][32];   // the same bytes as i32 lanes (the min term, converted to f32)
    alignas(32) uint8_t q3[4][32];
    Masks() {
        for (int b = 0; b < 8; ++b) {
            const int idx = 8 * (b / 4) + 2 * (b % 4);
            for (int l = 0; l < 32; ++l) {
                q2[b][l] = l % 2 ? 0x80 : (uint8_t) (idx + (l >= 16));
                q2_32[b][l] = l % 4 ? 0x80 : (uint8_t) (idx + (l >= 16));
            }
        }
        for (int i = 0; i < 4; ++i)
            for (int l = 0; l < 32; ++l) q3[i][l] = (uint8_t) (4 * i + 2 * (l >= 16) + l % 2);
    }
};
const Masks kMasks;
__m256i mask(const uint8_t* m) { return _mm256_load_si256(reinterpret_cast<const __m256i*>(m)); }

__m256i load32(const void* p) { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); }

// q2_K, q8_1 blocks I and 4 + I of superblock kbx (step i of both halves' threads, interleaved for the scheduler) against
// G activations: sd[t][n] = fma(d8, dp sc, sd[t][n]), sm[t][n] = fma(d8, m sum4(x), sm[t][n]) (Q2KTraits::apply's step
// i; m sum4(x) < 2^24 and dp sc too, so the f32 product and conversion are exact)
template <int I, int G>
DSV41_INLINE void q2_step(const __m256i* q, __m256i scb, __m256i mb, int64_t qb, const Act* a, __m256 (*sd)[2],
                          __m256 (*sm)[2]) {
    __m256i w[2], sc[2];
    __m256 mn[2];
    for (int n = 0; n < 2; ++n) {
        w[n] = _mm256_and_si256(_mm256_srli_epi16(q[n], 2 * I), _mm256_set1_epi8(3));
        sc[n] = _mm256_shuffle_epi8(scb, mask(kMasks.q2[4 * n + I]));
        mn[n] = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(mb, mask(kMasks.q2_32[4 * n + I])));
    }
    for (int t = 0; t < G; ++t)
        for (int n = 0; n < 2; ++n) {
            const int64_t b = qb + 4 * n + I;
            const __m256i xb = load32(a[t].x + (size_t) b * kQ81 + 4);
            const __m256 d = _mm256_broadcast_ss(a[t].d8 + b);
            const __m256i dp = _mm256_madd_epi16(_mm256_maddubs_epi16(w[n], xb), sc[n]);
            sd[t][n] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(dp), sd[t][n]);
            sm[t][n] = _mm256_fmadd_ps(d, _mm256_mul_ps(mn[n], _mm256_loadu_ps(a[t].s4 + (size_t) b * 8)), sm[t][n]);
        }
}

// a q2_K superblock: acc[t][n] += apply() of the threads iqs = 8 n + k
template <int G> DSV41_INLINE void q2_superblock(const uint8_t* blk, int64_t kbx, const Act* a, __m256 (*acc)[2]) {
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i s = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(blk)));
    const __m256i scb = _mm256_and_si256(s, m15), mb = _mm256_and_si256(_mm256_srli_epi16(s, 4), m15);
    const __m256i q[2] = {load32(blk + 16), load32(blk + 48)};
    __m256 sd[G][2], sm[G][2];
    for (int t = 0; t < G; ++t) sd[t][0] = sd[t][1] = sm[t][0] = sm[t][1] = _mm256_setzero_ps();
    const int64_t qb = kbx * 8;
    q2_step<0, G>(q, scb, mb, qb, a, sd, sm);
    q2_step<1, G>(q, scb, mb, qb, a, sd, sm);
    q2_step<2, G>(q, scb, mb, qb, a, sd, sm);
    q2_step<3, G>(q, scb, mb, qb, a, sd, sm);
    const __m256 sign = _mm256_set1_ps(-0.0f), dmx = _mm256_set1_ps(h2f(blk + 80)), dmy = _mm256_set1_ps(h2f(blk + 82));
    for (int t = 0; t < G; ++t)
        for (int n = 0; n < 2; ++n)
            acc[t][n] = _mm256_add_ps(acc[t][n], _mm256_fmadd_ps(dmx, sd[t][n], _mm256_xor_ps(_mm256_mul_ps(dmy, sm[t][n]), sign)));
}

// q3_K's 16 six-bit scales minus 32 as i16 (dequantize_row_q3_K's kmask unpacking: q3_q8_dot_impl's values)
DSV41_INLINE __m256i q3_scales(const uint8_t* scales) {
    uint32_t aux[3];
    std::memcpy(aux, scales, 12);
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    // the four words straight into a vector (ggml's AVX2 q3_K: no store and reload of them)
    const __m128i v = _mm_set_epi32((int) (((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4)),
                                    (int) (((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4)),
                                    (int) ((aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4)),
                                    (int) ((aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4)));
    return _mm256_sub_epi16(_mm256_cvtepi8_epi16(v), _mm256_set1_epi16(32));
}

// hmask's bit H at bit 2 of each byte
template <int H> DSV41_INLINE __m256i hbit2(__m256i hm) {
    if constexpr (H >= 2) return _mm256_srli_epi16(hm, H - 2);
    else return _mm256_slli_epi16(hm, 2 - H);
}

// q3_K, q8_1 blocks I and 4 + I: s[t][n] = fma(d8, dp sc, s[t][n]) (q3_q8_dot_impl's step i); the weights
// ((qs >> 2 I) & 3) minus 4 where hmask's bit 4 n + I is clear
template <int I, int G>
DSV41_INLINE void q3_step(const __m256i* q, __m256i hm, const __m256i* sv, int64_t qb, const Act* a, __m256 (*s)[2]) {
    __m256i w[2], aw[2], sc[2];
    const __m256i h[2] = {hbit2<I>(hm), hbit2<4 + I>(hm)};
    for (int n = 0; n < 2; ++n) {
        const __m256i lo = _mm256_and_si256(_mm256_srli_epi16(q[n], 2 * I), _mm256_set1_epi8(3));
        w[n] = _mm256_sub_epi8(lo, _mm256_andnot_si256(h[n], _mm256_set1_epi8(4)));
        aw[n] = _mm256_sign_epi8(w[n], w[n]);
        sc[n] = _mm256_shuffle_epi8(sv[n], mask(kMasks.q3[I]));
    }
    for (int t = 0; t < G; ++t)
        for (int n = 0; n < 2; ++n) {
            const int64_t b = qb + 4 * n + I;
            const __m256i xb = load32(a[t].x + (size_t) b * kQ81 + 4);
            const __m256i dp = _mm256_madd_epi16(_mm256_maddubs_epi16(aw[n], _mm256_sign_epi8(xb, w[n])), sc[n]);
            s[t][n] = _mm256_fmadd_ps(_mm256_broadcast_ss(a[t].d8 + b), _mm256_cvtepi32_ps(dp), s[t][n]);
        }
}

template <int G> DSV41_INLINE void q3_superblock(const uint8_t* blk, int64_t kbx, const Act* a, __m256 (*acc)[2]) {
    const __m256i hm = load32(blk);
    const __m256i q[2] = {load32(blk + 32), load32(blk + 64)};
    const __m256i scales = q3_scales(blk + 96);
    // each half's 8 scales in both 128-bit halves
    const __m256i sv[2] = {_mm256_permute2x128_si256(scales, scales, 0x00), _mm256_permute2x128_si256(scales, scales, 0x11)};
    __m256 s[G][2];
    for (int t = 0; t < G; ++t) s[t][0] = s[t][1] = _mm256_setzero_ps();
    const int64_t qb = kbx * 8;
    q3_step<0, G>(q, hm, sv, qb, a, s);
    q3_step<1, G>(q, hm, sv, qb, a, s);
    q3_step<2, G>(q, hm, sv, qb, a, s);
    q3_step<3, G>(q, hm, sv, qb, a, s);
    const __m256 d3 = _mm256_set1_ps(h2f(blk + 108));
    for (int t = 0; t < G; ++t)
        for (int n = 0; n < 2; ++n) acc[t][n] = _mm256_add_ps(acc[t][n], _mm256_mul_ps(d3, s[t][n]));
}

// row r against G activations: thread tid = 16 j + 8 n + k sums superblocks kbx = j, j + 8, ... in order, its value
// in lane k of acc[t][n] while j runs, then part[t][tid]; then mmvq_kernel's end
template <int G, bool Q2>
void row_values(const uint8_t* row, int64_t nb, size_t bb, const Act* a, float* const* outs, int64_t r) {
    alignas(32) float part[G][128];
    for (int j = 0; j < 8; ++j) {
        __m256 acc[G][2];
        for (int t = 0; t < G; ++t) acc[t][0] = acc[t][1] = _mm256_setzero_ps();
        for (int64_t kbx = j; kbx < nb; kbx += 8) {
            if constexpr (Q2) q2_superblock<G>(row + (size_t) kbx * bb, kbx, a, acc);
            else q3_superblock<G>(row + (size_t) kbx * bb, kbx, a, acc);
        }
        for (int t = 0; t < G; ++t) {
            _mm256_store_ps(part[t] + 16 * j, acc[t][0]);
            _mm256_store_ps(part[t] + 16 * j + 8, acc[t][1]);
        }
    }
    for (int t = 0; t < G; ++t) {
        __m256 v[16];
        for (int k = 0; k < 16; ++k) v[k] = _mm256_load_ps(part[t] + 8 * k);
        outs[t][r] = reduce(v);
    }
}

template <bool Q2>
void rows_of(const uint8_t* w, size_t row_bytes, int64_t r0, int64_t r1, int64_t nb, size_t bb, const Act* a,
             float* const* outs, int64_t n_acts) {
    for (int64_t r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        for (int64_t t0 = 0; t0 < n_acts;) {
            switch (std::min<int64_t>(n_acts - t0, kGroup)) {
                case 1: row_values<1, Q2>(row, nb, bb, a + t0, outs + t0, r); t0 += 1; break;
                case 2: row_values<2, Q2>(row, nb, bb, a + t0, outs + t0, r); t0 += 2; break;
                case 3: row_values<3, Q2>(row, nb, bb, a + t0, outs + t0, r); t0 += 3; break;
                default: row_values<4, Q2>(row, nb, bb, a + t0, outs + t0, r); t0 += 4; break;
            }
        }
    }
}

void check_rows(int type, int64_t n_in) {
    if (!supports(type) || n_in <= 0 || n_in % 256) {
        std::fprintf(stderr, "dsv41 experts (CPU, GPU numerics): %s rows of %lld values (q2_K or q3_K, whole blocks of 256)\n",
                     qtype_name(type), (long long) n_in);
        std::exit(1);
    }
}

}  // namespace

bool supports(int type) { return type == kQTypeQ2K || type == kQTypeQ3K; }

void quantize(const float* x, int64_t k, uint8_t* out) {
    if (k <= 0 || k % 32) {
        std::fprintf(stderr, "dsv41 experts (CPU, GPU numerics): q8_1 rows of %lld values (whole blocks of 32)\n", (long long) k);
        std::exit(1);
    }
    for (int64_t b = 0; b < k / 32; ++b)
        if (!quantize_block(x + 32 * b, out + (size_t) b * kQ81)) mmvq_ref_quantize(x + 32 * b, 32, out + (size_t) b * kQ81);
}

void swiglu(const float* gate, const float* up, int64_t n, float limit, float* h) {
    const __m256 L = _mm256_set1_ps(limit), nL = _mm256_set1_ps(-limit), one = _mm256_set1_ps(1.0f), sign = _mm256_set1_ps(-0.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 gv = _mm256_loadu_ps(gate + i), uv = _mm256_loadu_ps(up + i);
        const __m256 g = _mm256_blendv_ps(gv, L, _mm256_cmp_ps(gv, L, _CMP_GT_OQ));
        __m256 u = _mm256_blendv_ps(uv, L, _mm256_cmp_ps(uv, L, _CMP_GT_OQ));
        u = _mm256_blendv_ps(u, nL, _mm256_cmp_ps(uv, nL, _CMP_LT_OQ));
        const __m256 e = expf_det8(_mm256_xor_ps(g, sign));
        _mm256_storeu_ps(h + i, _mm256_mul_ps(_mm256_div_ps(g, _mm256_add_ps(one, e)), u));
    }
    for (; i < n; ++i) h[i] = act::swiglu_clamp(gate[i], up[i], limit);
}

Act prepare_act(const uint8_t* xq, int64_t n_in, float* f) {
    const int64_t nq = n_in / 32;
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    float* s4 = f + nq;
    for (int64_t q = 0; q < nq; ++q) {
        const uint8_t* blk = xq + (size_t) q * kQ81;
        f[q] = h2f(blk);
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 4));
        _mm256_storeu_ps(s4 + 8 * q, _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(ones8, x), ones16)));
    }
    return Act{xq, f, s4};
}

void rows(int type, const uint8_t* w, size_t row_bytes, int64_t r0, int64_t r1, int64_t n_in, const Act* acts,
          float* const* outs, int64_t n_acts) {
    check_rows(type, n_in);
    const int64_t nb = n_in / 256;
    if (type == kQTypeQ2K) rows_of<true>(w, row_bytes, r0, r1, nb, 84, acts, outs, n_acts);
    else rows_of<false>(w, row_bytes, r0, r1, nb, 110, acts, outs, n_acts);
}

}  // namespace strata::dsv41::engine::mmvq_cpu
