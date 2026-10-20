// src/dsv41/engine/qdequant.cuh - the K-quant block decoder of the dequant path (QPrefill::DequantF16), shared by
// qmatmul.cu's dequant kernel and the routed experts' grouped GEMM (experts_gemm.cu), so both read the weights as
// the same f16 values: qmatmul_ref.cpp's qref_dequant_row rounded to f16, bit for bit. Internal to src/dsv41/engine.
//
// Split in two for the grouped GEMM: fetch8 issues the loads of eight values' bytes into registers, decode8 does the
// arithmetic, so a kernel can fetch the next step's weights before its tensor-core work and decode them after it (the
// loads' latency hidden behind the MMAs). dequant8 is the two in a row. Alignment: q2_K blocks are read at 4-byte
// words (qmatmul_check), q3_K at 2-byte ones.
//
// decode8_f16x2 gives the same eight values as
// the f16 pairs the grouped GEMM stores, from a table instead of one multiply per value: the 4 (q2_K) or 8 (q3_K)
// values a 16-value group can take are computed once as decode8 computes them (dl q - ml, dl (q - 4 + 4 h)) and
// rounded to f16, then each pair of values is picked from the table by a byte permute (prmt) whose selector is built
// from the 2-bit codes; q3_K's hmask bit picks between the table's halves by a mask (prmt's sign replicate). fetch8
// and decode8_f16x2 are host and device code: dsv41_engine_qdequant_test checks every group position of random
// blocks against qmatmul_ref.cpp's dequant rounded to f16, bit for bit, on the dev VM (the host's prmt is an
// emulation of PTX prmt.b32's default mode; experts_test --gemm checks the device's on a dev pod).
#pragma once

#include "strata/dsv41/engine/qmatmul.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_fp16.h>

#include <cstdint>

namespace strata::dsv41::engine::qdequant {

/// Bytes of one block of TYPE (qtype_block_bytes as a constant).
template <int TYPE> __host__ __device__ constexpr int block_bytes() { return TYPE == kQTypeQ2K ? 84 : TYPE == kQTypeQ3K ? 110 : 210; }

__device__ __forceinline__ float f16_at(const uint8_t* p) { return __half2float(__ushort_as_half((uint16_t) (p[0] | (p[1] << 8)))); }
__device__ __forceinline__ float f16_bits(uint32_t h) { return __half2float(__ushort_as_half((uint16_t) h)); }

/// The bytes eight values of a block decode from (fetch8), per type.
template <int TYPE> struct Raw;
template <> struct Raw<kQTypeQ2K> {
    uint32_t q[2];   // qs bytes l0 .. l0 + 7 of the value's 128-half, two 2-bit fields of each the j-th
    uint32_t dd;     // d (low half), dmin (high half)
    uint32_t sc;     // the 16-value group's scale (low nibble) and min (high nibble)
};
template <> struct Raw<kQTypeQ3K> {
    uint16_t hm[4], q[4];   // hmask bytes l0 .. l0 + 7, qs bytes of the 128-half
    uint16_t d;
    uint8_t s_lo, s_hi;     // the scale's low 4 bits' byte and its high 2 bits' byte (ggml's kmask unpacking)
};
template <> struct Raw<kQTypeQ6K> {
    const uint8_t* b;       // q6_K is not split (the head's type, no grouped GEMM): decode8 reads the block
};

// Values v0 .. v0 + 7 (v0 % 8 == 0, one 16-value scale group) of block b.
template <int TYPE> __host__ __device__ __forceinline__ Raw<TYPE> fetch8(const uint8_t* __restrict__ b, int v0) {
    const int n = v0 / 128, l0 = v0 % 32, j = (v0 % 128) / 32;
    Raw<TYPE> r;
    if constexpr (TYPE == kQTypeQ2K) {   // scales[16] (scale low nibble, min high), qs[64], d, dmin
        const uint32_t* q = reinterpret_cast<const uint32_t*>(b + 16 + 32 * n + l0);
        r.q[0] = q[0];
        r.q[1] = q[1];
        r.dd = *reinterpret_cast<const uint32_t*>(b + 80);
        r.sc = b[8 * n + 2 * j + l0 / 16];
    } else if constexpr (TYPE == kQTypeQ3K) {   // hmask[32], qs[64], scales[12] (6-bit), d
        const uint16_t* hm = reinterpret_cast<const uint16_t*>(b + l0);
        const uint16_t* q = reinterpret_cast<const uint16_t*>(b + 32 + 32 * n + l0);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            r.hm[i] = hm[i];
            r.q[i] = q[i];
        }
        r.d = *reinterpret_cast<const uint16_t*>(b + 108);
        // scale `is` of the 16: its low 4 bits in scales[is % 8] (the high nibble from 8 on), its high 2 bits in
        // scales[8 + is % 4] at bit 2 (is / 4)
        const int is = 8 * n + 2 * j + l0 / 16;
        r.s_lo = b[96 + (is & 7)];
        r.s_hi = b[104 + (is & 3)];
    } else {
        r.b = b;
    }
    return r;
}

// ggml's f32 order: q2_K dl * q - ml with dl = d * sc, ml = dmin * m (dl * q is exact, so a fused multiply-add gives
// the same bits); q3_K (d * (sc - 32)) * q; q6_K (d * sc) * (q - 32), the one product that can round, in the same
// order as ggml and qmatmul_ref.cpp.
template <int TYPE> __device__ __forceinline__ void decode8(const Raw<TYPE>& r, int v0, float* o) {
    const int n = v0 / 128, l0 = v0 % 32, j = (v0 % 128) / 32;
    if constexpr (TYPE == kQTypeQ2K) {
        const float dl = f16_bits(r.dd) * (float) (r.sc & 0xF), ml = f16_bits(r.dd >> 16) * (float) (r.sc >> 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = dl * (float) ((r.q[i / 4] >> (8 * (i % 4) + 2 * j)) & 3) - ml;
    } else if constexpr (TYPE == kQTypeQ3K) {
        const int is = 8 * n + 2 * j + l0 / 16;
        const int lo = (r.s_lo >> (4 * (is >> 3))) & 0xF, hi = (r.s_hi >> (2 * (is >> 2))) & 3;
        const float dl = f16_bits(r.d) * (float) ((lo | (hi << 4)) - 32);
        const uint32_t m = 1u << (4 * n + j);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = (r.q[i / 2] >> (8 * (i % 2))) & 0xFF, h = (r.hm[i / 2] >> (8 * (i % 2))) & 0xFF;
            o[i] = dl * (float) ((int) ((q >> (2 * j)) & 3) - ((h & m) ? 0 : 4));
        }
    } else {   // q6_K: ql[128], qh[64], scales[16] (int8), d
        const uint8_t* b = r.b;
        const float dl = f16_at(b + 208) * (float) reinterpret_cast<const int8_t*>(b + 192)[8 * n + 2 * j + l0 / 16];
        const uint8_t* L = b + 64 * n + l0 + (j % 2) * 32;
        const uint8_t* H = b + 128 + 32 * n + l0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int lo = j < 2 ? (L[i] & 0xF) : (L[i] >> 4);
            o[i] = dl * (float) ((lo | (((H[i] >> (2 * j)) & 3) << 4)) - 32);
        }
    }
}

// PTX prmt.b32 (default mode): byte k of the result is byte (s >> 4k) & 7 of {b, a} (a the low four), or with bit
// (s >> 4k) & 8 that byte's top bit replicated over all eight
__host__ __device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
#ifdef __CUDA_ARCH__
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
#else
    const uint64_t v = (uint64_t) a | ((uint64_t) b << 32);
    uint32_t r = 0;
    for (int k = 0; k < 4; ++k) {
        const uint32_t n = (s >> (4 * k)) & 0xF;
        uint32_t byte = (uint32_t) (v >> (8 * (n & 7))) & 0xFF;
        if (n & 8) byte = (byte & 0x80) ? 0xFF : 0;
        r |= byte << (8 * k);
    }
    return r;
#endif
}
// f16 bits to f32, and two f32 to an f16 pair (round to nearest even, lo in the low half)
__host__ __device__ __forceinline__ float h2f(uint32_t h) {
#ifdef __CUDA_ARCH__
    return f16_bits(h);
#else
    return kernels::f32_from_f16((uint16_t) h);
#endif
}
__host__ __device__ __forceinline__ uint32_t f16x2_rn(float lo, float hi) {
#ifdef __CUDA_ARCH__
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#else
    return (uint32_t) kernels::f16_from_f32(lo) | ((uint32_t) kernels::f16_from_f32(hi) << 16);
#endif
}
// the selectors of two values' f16 from a table of four (bytes 2 c, 2 c + 1 of value code c), for the codes in the
// low two bytes of c4 (each 0..3)
__host__ __device__ __forceinline__ uint32_t pair_sel(uint32_t c2) { return (c2 & 0xFFFF) * 0x22 + 0x1010; }

/// decode8's values v0 .. v0 + 7, each rounded to f16, as four pairs (o[i] = values 2 i, 2 i + 1); q2_K and q3_K.
template <int TYPE> __host__ __device__ __forceinline__ void decode8_f16x2(const Raw<TYPE>& r, int v0, uint32_t* o) {
    static_assert(TYPE == kQTypeQ2K || TYPE == kQTypeQ3K, "decode8_f16x2: q2_K and q3_K");
    const int n = v0 / 128, l0 = v0 % 32, j = (v0 % 128) / 32;
    if constexpr (TYPE == kQTypeQ2K) {
        const float dl = h2f(r.dd & 0xFFFF) * (float) (r.sc & 0xF), ml = h2f(r.dd >> 16) * (float) (r.sc >> 4);
        const uint32_t t01 = f16x2_rn(dl * 0.0f - ml, dl * 1.0f - ml), t23 = f16x2_rn(dl * 2.0f - ml, dl * 3.0f - ml);
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const uint32_t c4 = (r.q[w] >> (2 * j)) & 0x03030303u;
            o[2 * w] = prmt(t01, t23, pair_sel(c4));
            o[2 * w + 1] = prmt(t01, t23, pair_sel(c4 >> 16));
        }
    } else {
        const int is = 8 * n + 2 * j + l0 / 16;
        const int lo = (r.s_lo >> (4 * (is >> 3))) & 0xF, hi = (r.s_hi >> (2 * (is >> 2))) & 3;
        const float dl = h2f(r.d) * (float) ((lo | (hi << 4)) - 32);
        // value code q + 4 h (h the hmask bit) is the value q - 4 + 4 h
        const uint32_t t01 = f16x2_rn(dl * -4.0f, dl * -3.0f), t23 = f16x2_rn(dl * -2.0f, dl * -1.0f);
        const uint32_t t45 = f16x2_rn(dl * 0.0f, dl * 1.0f), t67 = f16x2_rn(dl * 2.0f, dl * 3.0f);
        const int bit = 4 * n + j;
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const uint32_t q = (uint32_t) r.q[2 * w] | ((uint32_t) r.q[2 * w + 1] << 16);
            const uint32_t h = (uint32_t) r.hm[2 * w] | ((uint32_t) r.hm[2 * w + 1] << 16);
            const uint32_t c4 = (q >> (2 * j)) & 0x03030303u, hb = h << (7 - bit);   // each byte's bit to its top
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const uint32_t sel = pair_sel(c4 >> (16 * p));
                const uint32_t mask = prmt(hb, 0, p ? 0xBBAAu : 0x9988u);
                o[2 * w + p] = (prmt(t01, t23, sel) & ~mask) | (prmt(t45, t67, sel) & mask);
            }
        }
    }
}

template <int TYPE> __device__ __forceinline__ void dequant8(const uint8_t* __restrict__ b, int v0, float* o) {
    decode8<TYPE>(fetch8<TYPE>(b, v0), v0, o);
}

}  // namespace strata::dsv41::engine::qdequant
