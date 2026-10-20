// src/dsv41/engine/mmvq_ref.cpp - the GPU-order reference of include/strata/dsv41/engine/mmvq_ref.hpp. Compiled with
// -mfma -ffp-contract=off (CMakeLists.txt): each line below is the device line named beside it, one rounding per
// float operation (std::fma for __fmaf_rn), the integer dots exact.
#include "strata/dsv41/engine/mmvq_ref.hpp"

#include "expert_act.hpp"
#include "strata/dsv41/engine/qmatmul.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "mmvq_ref: %s\n", msg);
    std::exit(1);
}

constexpr int kWarp = 32, kWarps = 4, kThreads = kWarp * kWarps;
constexpr float kHalfMax = 65504.0f;

// mmvq.cuh Q81Block, read where the device reads it
struct Q81 {
    uint16_t d, s;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == kQ81BlockBytes, "a q8_1 block is 36 bytes");

float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return kernels::f32_from_f16(h);
}
// reinterpret_cast<const int*>(p)[i] and mmvq.cuh load_int_b2 (two 16-bit loads): the same four bytes
int load_int(const void* p, int i) {
    int v;
    std::memcpy(&v, static_cast<const uint8_t*>(p) + 4 * i, 4);
    return v;
}
int8_t byte_of(int v, int b) { return (int8_t) (uint8_t) ((uint32_t) v >> (8 * b)); }
// __dp4a(a, b, c): the four signed byte products added to c (dp4a.hpp's fallback)
int dp4a(int a, int b, int c) {
    for (int k = 0; k < 4; ++k) c += byte_of(a, k) * byte_of(b, k);
    return c;
}
// __vsubss4(a, b): bytewise signed subtraction, saturated
int vsubss4(int a, int b) {
    uint32_t r = 0;
    for (int k = 0; k < 4; ++k) {
        int v = byte_of(a, k) - byte_of(b, k);
        v = v > 127 ? 127 : v < -128 ? -128 : v;
        r |= (uint32_t) (uint8_t) (int8_t) v << (8 * k);
    }
    return (int) r;
}
// q8_1_finite.hpp
float q8_1_finite(float v) { return std::fabs(v) > kHalfMax ? std::copysign(kHalfMax, v) : v; }
int8_t q8_1_quant(float xi, float d, float amax) {
    if (amax == 0.0f) return 0;
    const float q = std::round(xi / d);   // roundf: halves away from zero, as std::round
    if (std::isnan(q)) return 0;          // the device's float-to-int conversion of a NaN
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}
// mmvq.cuh warp_sum and warp_max: lane l combines lane l ^ o for o = 16 .. 1; lane 0's value
float warp_sum(const float* v) {
    float a[kWarp], b[kWarp];
    std::memcpy(a, v, sizeof a);
    for (int o = kWarp / 2; o > 0; o >>= 1) {
        for (int l = 0; l < kWarp; ++l) b[l] = a[l] + a[l ^ o];
        std::memcpy(a, b, sizeof a);
    }
    return a[0];
}
float warp_max(const float* v) {
    float a[kWarp], b[kWarp];
    std::memcpy(a, v, sizeof a);
    for (int o = kWarp / 2; o > 0; o >>= 1) {
        for (int l = 0; l < kWarp; ++l) b[l] = std::fmax(a[l], a[l ^ o]);
        std::memcpy(a, b, sizeof a);
    }
    return a[0];
}

// mmvq.cuh Q2KTraits
struct Q2K {
    static constexpr int kBlock = 84, T = 16, KBY = 8, BPI = kThreads / 16;
    static int kqs(int tid) { return tid % 16; }
    struct W {
        int v;
        int sc[4];
        float dmx, dmy;
        int bq8_offset;
    };
    static W load(const uint8_t* w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.v = load_int(w + 16, iqs);
        for (int i = 0; i < 4; ++i) r.sc[i] = w[scale_offset + 2 * i];
        r.dmx = h2f(w + 80);
        r.dmy = h2f(w + 82);
        return r;
    }
    static float apply(const W& r, const Q81* x, int iqs) {
        float sumf_d = 0.0f, sumf_m = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int u = load_int(x[r.bq8_offset + i].qs, iqs % 8);
            const float d8 = kernels::f32_from_f16(x[r.bq8_offset + i].d);
            const int vi = (r.v >> (2 * i)) & 0x03030303;
            sumf_d = std::fma(d8, (float) (dp4a(vi, u, 0) * (r.sc[i] & 0xF)), sumf_d);
            int m = r.sc[i] >> 4;
            m |= m << 8;
            m |= m << 16;
            sumf_m = std::fma(d8, (float) dp4a(m, u, 0), sumf_m);
        }
        return std::fma(r.dmx, sumf_d, -(r.dmy * sumf_m));
    }
};

// mmvq.cuh q3_q8_dot_impl and Q3KTraits
float q3_q8_dot_impl(int vl, int vh, const int* u, const uint8_t* scales, int scale_offset, float d3, const float* d8) {
    float sumf = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = vsubss4(vil, vih);
        sumf = std::fma(d8[i], (float) (dp4a(vi, u[i], 0) * sc), sumf);
    }
    return d3 * sumf;
}
struct Q3K {
    static constexpr int kBlock = 110, T = 16, KBY = 8, BPI = kThreads / 16;
    static int kqs(int tid) { return tid % 16; }
    struct W {
        int vl, vh;
        float d;
        const uint8_t* scales;
        int scale_offset, bq8_offset;
    };
    static W load(const uint8_t* w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = h2f(w + 108);
        r.vl = load_int(w + 32, iqs);
        r.vh = ~load_int(w, iqs % 8) >> r.bq8_offset;
        r.scales = w + 96;
        return r;
    }
    static float apply(const W& r, const Q81* x, int iqs) {
        int u[4];
        float d8[4];
        for (int i = 0; i < 4; ++i) {
            u[i] = load_int(x[r.bq8_offset + i].qs, iqs % 8);
            d8[i] = kernels::f32_from_f16(x[r.bq8_offset + i].d);
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};

// qmatmul.cu mmvq_kernel at NCOLS = 1 (one row of a block's ROWS): thread tid = 32 threadIdx.y + threadIdx.x
template <typename F> float dot(const uint8_t* row, const Q81* x, int64_t n_in) {
    const int blocks_per_row = (int) (n_in / 256);
    float tmp[kThreads];
    for (int tid = 0; tid < kThreads; ++tid) {
        tmp[tid] = 0.0f;
        for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
            const int kby = kbx * F::KBY;
            const int kqs = F::kqs(tid);
            tmp[tid] += F::apply(F::load(row + (size_t) kbx * F::kBlock, kqs), x + kby, kqs);
        }
    }
    // warps 1 .. 3 into shared memory, warp 0 adds them in turn (tmp += partial[l]), then its xor sum
    float lane[kWarp];
    for (int l = 0; l < kWarp; ++l) {
        float s = tmp[l];
        for (int w = 1; w < kWarps; ++w) s += tmp[w * kWarp + l];
        lane[l] = s;
    }
    return warp_sum(lane);
}

}  // namespace

void mmvq_ref_quantize(const float* x, int64_t k, uint8_t* out) {
    if (k <= 0 || k % 32) failf("q8_1 rows of %lld values (whole blocks of 32)", (long long) k);
    for (int64_t b = 0; b < k / 32; ++b) {
        // mmvq.cuh q8_1_warp, lane l holding x[32 b + l]
        const float* xi = x + b * 32;
        float ax[kWarp];
        for (int l = 0; l < kWarp; ++l) ax[l] = std::fabs(xi[l]);
        const float amax = warp_max(ax);
        const float sum = warp_sum(xi);
        const float d = q8_1_finite(amax / 127.0f);
        Q81 q;
        for (int l = 0; l < kWarp; ++l) q.qs[l] = q8_1_quant(xi[l], d, amax);
        q.d = kernels::f16_from_f32(d);
        q.s = kernels::f16_from_f32(q8_1_finite(sum));
        std::memcpy(out + (size_t) b * kQ81BlockBytes, &q, kQ81BlockBytes);
    }
}

float mmvq_ref_dot(int type, const uint8_t* row, const uint8_t* xq, int64_t n_in) {
    if (n_in <= 0 || n_in % 256) failf("rows of %lld values (whole blocks of 256)", (long long) n_in);
    const Q81* x = reinterpret_cast<const Q81*>(xq);
    if (type == kQTypeQ2K) return dot<Q2K>(row, x, n_in);
    if (type == kQTypeQ3K) return dot<Q3K>(row, x, n_in);
    failf("weight type %d (%s): the GPU-order reference computes q2_K and q3_K", type, qtype_name(type));
}

void mmvq_ref_expert(const ExpertLayerHost& L, const DenseDims& d, int32_t expert, const float* x, float* y) {
    const int64_t E = d.n_embd, F = d.n_ff;
    if (L.type[0] != L.type[1]) failf("layer %d: gate %s and up %s", L.layer, qtype_name(L.type[1]), qtype_name(L.type[0]));
    std::vector<uint8_t> xq((size_t) (E / 32) * kQ81BlockBytes), hq((size_t) (F / 32) * kQ81BlockBytes);
    std::vector<float> h((size_t) F);
    mmvq_ref_quantize(x, E, xq.data());
    // qmatmul_experts: gate (part 1) and up (part 0) by one GLU launch, the clamp in its epilogue
    const size_t up_row = L.bytes[0] / (size_t) F, gate_row = L.bytes[1] / (size_t) F, down_row = L.bytes[2] / (size_t) E;
    const uint8_t* up = L.data[0] + (size_t) expert * L.bytes[0];
    const uint8_t* gate = L.data[1] + (size_t) expert * L.bytes[1];
    const uint8_t* down = L.data[2] + (size_t) expert * L.bytes[2];
    for (int64_t r = 0; r < F; ++r) {
        const float a = mmvq_ref_dot(L.type[1], gate + (size_t) r * gate_row, xq.data(), E);
        const float b = mmvq_ref_dot(L.type[0], up + (size_t) r * up_row, xq.data(), E);
        h[(size_t) r] = act::swiglu_clamp(a, b, L.limit);
    }
    mmvq_ref_quantize(h.data(), F, hq.data());
    for (int64_t r = 0; r < E; ++r) y[r] = mmvq_ref_dot(L.type[2], down + (size_t) r * down_row, hq.data(), F);
}

}  // namespace strata::dsv41::engine
