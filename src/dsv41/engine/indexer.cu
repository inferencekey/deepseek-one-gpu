// src/dsv41/engine/indexer.cu - the lightning indexer's scorers and top-k, see include/strata/dsv41/engine/indexer.hpp.
#include "strata/dsv41/engine/indexer.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine {
namespace {

constexpr int D = kIndexerDim;
constexpr int HMAX = kIndexerHeads;

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "dsv41 indexer: %s\n", what);
    std::exit(1);
}

void check_launch(const char* kernel) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dsv41 indexer: %s: %s\n", kernel, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ __forceinline__ int64_t visible_rows(int32_t pos, int ratio, int64_t n_k) {
    const int64_t v = pos < 0 ? 0 : ((int64_t) pos + 1) / ratio;
    return v < n_k ? v : n_k;
}

// the query element as the fork's mul_mat feeds it: rounded to f16 (to nearest even), as f16 bits
__device__ __forceinline__ uint32_t q_f16(const float* q, int64_t qi, int h, int k, int n_heads) {
    return h < n_heads ? kernels::f16_from_f32(q[(qi * n_heads + h) * D + k]) : 0u;
}

// ---- FP32 scorer: one thread per (query, key), the query in shared memory read as broadcasts. Per key 32 dot
// products of 128 dims, each accumulated in order with fmaf; then relu, times the head's weight, heads added in order.
constexpr int SIMT_T = 128;   // keys per CTA

__global__ void __launch_bounds__(SIMT_T) scores_simt_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                             const int32_t* __restrict__ pos, const uint16_t* __restrict__ keys,
                                                             int64_t n_k, int n_heads, int ratio,
                                                             float* __restrict__ scores, int64_t ld) {
    __shared__ __align__(16) float sQ[HMAX * D];
    __shared__ float sW[HMAX];
    const int64_t qi = blockIdx.y;
    const int64_t vis = visible_rows(pos[qi], ratio, n_k);
    const int64_t j0 = (int64_t) blockIdx.x * SIMT_T;
    if (j0 >= vis) return;                            // the whole CTA is past this query's visible rows
    for (int i = threadIdx.x; i < HMAX * D; i += SIMT_T) {
        const int h = i / D, k = i % D;
        sQ[i] = kernels::f32_from_f16((uint16_t) q_f16(q, qi, h, k, n_heads));
    }
    if (threadIdx.x < HMAX) sW[threadIdx.x] = (int) threadIdx.x < n_heads ? w[qi * n_heads + threadIdx.x] : 0.f;
    __syncthreads();
    const int64_t j = j0 + threadIdx.x;
    if (j >= vis) return;
    float acc[HMAX];
#pragma unroll
    for (int h = 0; h < HMAX; ++h) acc[h] = 0.f;
    const uint4* kr = reinterpret_cast<const uint4*>(keys + j * D);
#pragma unroll 2
    for (int c = 0; c < D / 8; ++c) {
        const uint4 raw = kr[c];
        const uint32_t wd[4] = {raw.x, raw.y, raw.z, raw.w};
        float kf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kf[2 * i] = kernels::f32_from_f16((uint16_t) (wd[i] & 0xffffu));
            kf[2 * i + 1] = kernels::f32_from_f16((uint16_t) (wd[i] >> 16));
        }
#pragma unroll
        for (int h = 0; h < HMAX; ++h) {
            const float4 qa = *reinterpret_cast<const float4*>(sQ + h * D + c * 8);
            const float4 qb = *reinterpret_cast<const float4*>(sQ + h * D + c * 8 + 4);
            float a = acc[h];
            a = fmaf(kf[0], qa.x, a); a = fmaf(kf[1], qa.y, a); a = fmaf(kf[2], qa.z, a); a = fmaf(kf[3], qa.w, a);
            a = fmaf(kf[4], qb.x, a); a = fmaf(kf[5], qb.y, a); a = fmaf(kf[6], qb.z, a); a = fmaf(kf[7], qb.w, a);
            acc[h] = a;
        }
    }
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < HMAX; ++h) s += (acc[h] > 0.f ? acc[h] : 0.f) * sW[h];
    scores[qi * ld + j] = s;
}

// ---- tensor-core scorer (from qsa_select.cu block_scores_tc_kernel): the scores of a CTA's 8 queries are one GEMM,
// rows (query, head) x columns (keys), K = 128, on mma.sync m16n8k16 f16 x f16 -> f32. Warp w owns query q0 + w: its
// heads are MT m16 tiles held as A fragments for the whole kernel (the query rounded to f16, as the fork's mul_mat).
// The CTA walks a range of keys in tiles of 32 (one padded shared tile, every warp reads it); per tile a warp's 8
// partial sums per lane (4 n8 tiles x 2 columns, over the lane's heads) are reduced over the 8 lanes of a column group
// by a reduce-scatter, so each lane ends with one key's score and the warp stores 32 scores at once.
//
// (select 7.6 ms per ratio-2 and 12.9 per ratio-1 index source at a 16K chunk on the
// A100-SXM4; the bench had this scorer at ~150 TFLOP/s): the A fragments from float2 loads all in flight, rounded by
// the hardware's conversion (f16_from_f32's bits for every finite value; the scalar loads each waited on the previous
// one's software conversion); the key tiles double-buffered by cp.async (the next tile's copy under this one's MMAs, one
// barrier per tile instead of two around a synchronous copy); the B fragments by ldmatrix (two per k step instead of
// eight 32-bit loads). Every product and every sum is the same in the same order: the scores are the same bits.
constexpr int TC_WARPS = 8;                  // queries per CTA
constexpr int TC_NT = 32;                    // keys per tile (4 n8 tiles)
constexpr int TC_TILES = 64;                 // tiles per CTA: 2048 keys
constexpr int TC_KS = D + 8;                 // f16 per shared key row: 272 bytes, the 8 rows of an ldmatrix on distinct banks

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define DSV41_IDX_MMA 1
#else
#define DSV41_IDX_MMA 0   // f16 m16n8k16 needs sm_80: an older target's code is a trap, the host refuses it there
#endif

__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, const uint32_t* b) {
#if DSV41_IDX_MMA
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
#if DSV41_IDX_MMA
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
#else
    __trap();
#endif
}
// 16 bytes from src, or zeros when !valid (src is then not read but must be a mapped address); then the group's commit
__device__ __forceinline__ void cp16_or_zero(void* dst, const void* src, bool valid) {
#if DSV41_IDX_MMA
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
#else
    __trap();
#endif
}
__device__ __forceinline__ void cp_commit() {
#if DSV41_IDX_MMA
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_wait_all() {
#if DSV41_IDX_MMA
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}
// two f32 as an f16 pair, lo in the low half, each rounded to nearest even
__device__ __forceinline__ uint32_t f16x2_rn(float2 v) {
    const __half2 h = __floats2half2_rn(v.x, v.y);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ float relu(float x) { return x > 0.f ? x : 0.f; }

template <int MT>
__global__ void __launch_bounds__(TC_WARPS * 32) scores_tc_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                                  const int32_t* __restrict__ pos, int64_t nq,
                                                                  const uint16_t* __restrict__ keys, int64_t n_k,
                                                                  int n_heads, int ratio, float* __restrict__ scores,
                                                                  int64_t ld) {
    __shared__ __align__(16) uint16_t sK[2][TC_NT * TC_KS];
    __shared__ int64_t s_vis[TC_WARPS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t qi = (int64_t) blockIdx.y * TC_WARPS + warp;
    const int64_t vis = qi < nq ? visible_rows(pos[qi], ratio, n_k) : 0;
    if (lane == 0) s_vis[warp] = vis;
    // A fragments (m16n8k16 row-major A): reg 0 = (row gid, k 2*tig..+1), 1 = (row gid + 8, same k), 2 = (row gid,
    // k + 8), 3 = (row gid + 8, k + 8); the lower half holds the lower k
    uint32_t a[MT][D / 16][4];
    float wr[MT][2];
    const float* qq = q + (qi < nq ? qi : 0) * n_heads * D;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int h0 = mt * 16 + gid, h1 = h0 + 8;
        const bool r0 = qi < nq && h0 < n_heads, r1 = qi < nq && h1 < n_heads;
        wr[mt][0] = r0 ? w[qi * n_heads + h0] : 0.f;
        wr[mt][1] = r1 ? w[qi * n_heads + h1] : 0.f;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
            const int k0 = ks * 16 + tig * 2;
            const float2 z = make_float2(0.f, 0.f);
            const float2 v00 = r0 ? __ldg(reinterpret_cast<const float2*>(qq + h0 * D + k0)) : z;
            const float2 v10 = r1 ? __ldg(reinterpret_cast<const float2*>(qq + h1 * D + k0)) : z;
            const float2 v01 = r0 ? __ldg(reinterpret_cast<const float2*>(qq + h0 * D + k0 + 8)) : z;
            const float2 v11 = r1 ? __ldg(reinterpret_cast<const float2*>(qq + h1 * D + k0 + 8)) : z;
            a[mt][ks][0] = f16x2_rn(v00);
            a[mt][ks][1] = f16x2_rn(v10);
            a[mt][ks][2] = f16x2_rn(v01);
            a[mt][ks][3] = f16x2_rn(v11);
        }
    }
    __syncthreads();
    int64_t vmax = 0;
#pragma unroll
    for (int i = 0; i < TC_WARPS; ++i) vmax = s_vis[i] > vmax ? s_vis[i] : vmax;
    const int64_t k_begin = (int64_t) blockIdx.x * TC_TILES * TC_NT;
    const int64_t k_end = k_begin + TC_TILES * TC_NT < vmax ? k_begin + TC_TILES * TC_NT : vmax;
    // a tile's 32 keys of 16 pieces: two per thread, a key's pieces on consecutive threads
    auto load_tile = [&](int b, int64_t k0) {
#pragma unroll
        for (int u = 0; u < TC_NT * D / 8 / (TC_WARPS * 32); ++u) {
            const int i = u * TC_WARPS * 32 + t, r = i / (D / 8), c = i % (D / 8);
            const bool ok = k0 + r < n_k;
            cp16_or_zero(&sK[b][r * TC_KS + c * 8], ok ? keys + (k0 + r) * D + c * 8 : keys, ok);
        }
        cp_commit();
    };
    if (k_begin < k_end) load_tile(0, k_begin);
    // the key of this lane after the reduce-scatter: n8 tile (gid bit 0) * 2 + (gid bit 1), column gid bit 2
    const int my_key = (((gid & 1) << 1) | ((gid >> 1) & 1)) * 8 + tig * 2 + (gid >> 2);
    int b = 0;
    for (int64_t k0 = k_begin; k0 < k_end; k0 += TC_NT, b ^= 1) {
        cp_wait_all();
        __syncthreads();                              // tile b landed; every warp is done with the other buffer
        if (k0 + TC_NT < k_end) load_tile(b ^ 1, k0 + TC_NT);
        if (k0 >= vis) continue;                      // this warp's query sees none of the tile (uniform per warp)
        float acc[MT][4][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                // B fragments (col-major, a key's 16 k contiguous) of n8 tiles 2 np and 2 np + 1: regs 0, 1 = (k 2*tig..+1
                // and k + 8, key gid) of the first, 2, 3 of the second
                uint32_t bf[4];
                ldsm_x4(bf, &sK[b][(np * 16 + (lane & 7) + 8 * (lane >> 4)) * TC_KS + ks * 16 + 8 * ((lane >> 3) & 1)]);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma_f16(acc[mt][2 * np], a[mt][ks], bf);
                    mma_f16(acc[mt][2 * np + 1], a[mt][ks], bf + 2);
                }
            }
        }
        // C fragment: c0, c1 = (row gid, key 2*tig + 0/1), c2, c3 = (row gid + 8, same keys)
        float v[8];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                float s = 0.f;
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) s += relu(acc[mt][nt][c]) * wr[mt][0] + relu(acc[mt][nt][2 + c]) * wr[mt][1];
                v[nt * 2 + c] = s;
            }
        // reduce-scatter over gid (lane bits 2, 3, 4): each step a lane keeps the half its bit selects
#pragma unroll
        for (int step = 0, half = 4; step < 3; ++step, half >>= 1) {
            const bool hi = (lane >> (2 + step)) & 1;
#pragma unroll
            for (int i = 0; i < half; ++i) {
                const float keep = hi ? v[i + half] : v[i], send = hi ? v[i] : v[i + half];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, send, 4 << step);
            }
        }
        const int64_t j = k0 + my_key;
        if (j < vis) scores[qi * ld + j] = v[0];
    }
}

// ---- top-k (from qsa_select.cu block_topk_wide_kernel, every row a cell of weight 1): one CTA per query, a 4-pass
// radix select of the largest key thr with (rows >= thr) >= k, 8 bits per pass on per-warp histograms; then the rows
// above thr and the lowest-index rows equal to thr, emitted ascending in the order (warp, row of 32, lane): warp w
// holds the rows [w * 32 * per, (w + 1) * 32 * per).
//
// with HELD (a query's visible rows at most TK_T * TK_PER, 16,384: a 16K prompt at ratio 1) each thread
// keeps its rows' keys in registers, read from memory once instead of once per pass and twice more for the output;
// otherwise every pass reads them again as before. Each pass's digit is found by warp 0's scan of the 256 bins (eight
// per lane, a prefix over the lanes, then the one lane's eight) instead of thread 0 walking them one by one: the same
// digit and count.
constexpr int TK_T = 1024;
constexpr int TK_PER = 16;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;                         // -0 and +0 are one key
    if (!(v == v)) return 0u;                         // NaN below every number
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// one row of 32 of a warp's range (row r0 + lane, its key k, in: a row of the range below n): its selected rows written
// at their places in the ascending output; run_sel and run_eq count the warp's selected and tied rows so far
__device__ __forceinline__ void emit_row(int64_t r0, uint32_t k, bool in, uint32_t thr, int eq_budget, int lane,
                                         int32_t* out, int& run_sel, int& run_eq) {
    const int my_gt = (in && k > thr) ? 1 : 0, my_eq = (in && k == thr) ? 1 : 0;
    if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) return;   // a row of 32 without a selected one
    int pe = my_eq;                                   // tied rows up to and with this lane
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, pe, o);
        if (lane >= o) pe += y;
    }
    const int left = eq_budget - (run_eq + pe - my_eq);
    const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
    const int my_sel = my_gt + take;
    int ps = my_sel;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, ps, o);
        if (lane >= o) ps += y;
    }
    if (my_sel) out[run_sel + ps - 1] = (int32_t) (r0 + lane);
    run_eq += __shfl_sync(0xffffffffu, pe, 31);
    run_sel += __shfl_sync(0xffffffffu, ps, 31);
}

template <bool HELD>
__global__ void __launch_bounds__(TK_T) topk_kernel(const float* __restrict__ scores, int64_t ld,
                                                    const int32_t* __restrict__ pos, int64_t n_k, int ratio, int top_k,
                                                    int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_gt[TK_T / 32], s_eq[TK_T / 32];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int64_t n = visible_rows(pos[qi], ratio, n_k);
    int32_t* out = ids + qi * top_k;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n <= top_k) {                                 // every visible row is selected: the identity, then -1
        for (int64_t j = t; j < top_k; j += TK_T) out[j] = j < n ? (int32_t) j : -1;
        return;
    }
    const float* sc = scores + qi * ld;
    const int64_t per = (n + TK_T - 1) / TK_T;        // rows of 32 per warp
    const int64_t w0 = (int64_t) warp * 32 * per;
    // HELD: the key of row w0 + lane + 32 i in kr[i], for i < per and the row below n
    uint32_t kr[HELD ? TK_PER : 1];
    if constexpr (HELD) {
#pragma unroll
        for (int i = 0; i < TK_PER; ++i) {
            const int64_t b = w0 + lane + 32 * i;
            kr[i] = i < per && b < n ? order_key(sc[b]) : 0u;
        }
    }
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        if constexpr (HELD) {
#pragma unroll
            for (int i = 0; i < TK_PER; ++i) {
                const uint32_t k = kr[i];
                if (i < per && w0 + lane + 32 * i < n && (k & hi_mask) == (prefix & hi_mask))
                    atomicAdd(&hist[warp][(k >> shift) & 255], 1);
            }
        } else {
            for (int64_t b = t; b < n; b += TK_T) {
                const uint32_t k = order_key(sc[b]);
                if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(k >> shift) & 255], 1);
            }
        }
        __syncthreads();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int sum = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2) sum += hist[w2][t];
            hist[0][t] = sum;
        }
        __syncthreads();
        if (warp == 0) {
            // the largest digit d whose count, with the bins above it, reaches top_k (d = 0 when no digit above 0
            // does), and the rows above d: lane L takes bins 255 - 8 L .. 248 - 8 L, highest first
            int hb[8], sum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { hb[j] = hist[0][255 - 8 * lane - j]; sum += hb[j]; }
            int incl = sum;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += y;
            }
            const uint32_t hit = __ballot_sync(0xffffffffu, above + incl >= top_k);
            const int first = hit ? __ffs((int) hit) - 1 : 31;
            if (lane == first) {
                int cum = above + incl - sum, d = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int dd = 255 - 8 * lane - j;
                    if (dd == 0 || cum + hb[j] >= top_k) { d = dd; break; }
                    cum += hb[j];
                }
                s_digit = d;
                s_above = cum;
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int eq_budget = top_k - above;              // rows equal to thr that fit, lowest index first
    int gt = 0, eq = 0;
    if constexpr (HELD) {
#pragma unroll
        for (int i = 0; i < TK_PER; ++i) {
            if (i < per && w0 + lane + 32 * i < n) {
                gt += kr[i] > thr;
                eq += kr[i] == thr;
            }
        }
    } else {
        for (int64_t b = w0 + lane; b < n && b < w0 + 32 * per; b += 32) {
            const uint32_t k = order_key(sc[b]);
            if (k > thr) ++gt;
            else if (k == thr) ++eq;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    __syncthreads();
    if (t == 0) {                                     // per warp: the rows selected before it and the tied rows before it
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < TK_T / 32; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    __syncthreads();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    if constexpr (HELD) {
#pragma unroll
        for (int i = 0; i < TK_PER; ++i) {
            const int64_t r0 = w0 + 32 * i;
            if (i >= per || r0 >= n) break;           // uniform per warp
            emit_row(r0, kr[i], r0 + lane < n, thr, eq_budget, lane, out, run_sel, run_eq);
        }
    } else {
        for (int64_t r0 = w0; r0 < n && r0 < w0 + 32 * per; r0 += 32) {
            const int64_t b = r0 + lane;
            const bool in = b < n && b < w0 + 32 * per;
            emit_row(r0, in ? order_key(sc[b]) : 0u, in, thr, eq_budget, lane, out, run_sel, run_eq);
        }
    }
}

void check_dims(const IndexerDims& d, int64_t nq, int64_t n_k, int64_t ld) {
    char msg[256];
    if (d.n_heads < 1 || d.n_heads > HMAX) {
        std::snprintf(msg, sizeof msg, "n_heads %d outside 1..%d", d.n_heads, HMAX);
        fail(msg);
    }
    if (d.top_k < 1 || d.top_k > kIndexerMaxTopK) {
        std::snprintf(msg, sizeof msg, "top_k %d outside 1..%d", d.top_k, kIndexerMaxTopK);
        fail(msg);
    }
    if (d.ratio < 1 || nq < 0 || nq > 65535 || n_k < 0 || n_k > INT32_MAX || ld < n_k) {
        std::snprintf(msg, sizeof msg, "ratio %d (>= 1), nq %lld (0..65535), n_k %lld (0..2^31-1), ld %lld (>= n_k)",
                      d.ratio, (long long) nq, (long long) n_k, (long long) ld);
        fail(msg);
    }
}

int device_cc() {
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess)
        fail("no CUDA device");
    return major * 10 + minor;
}

}  // namespace

IndexerPlan indexer_plan(int64_t nq, int64_t n_k, size_t scratch_cap) {
    IndexerPlan p;
    p.ld = (n_k + 31) / 32 * 32;
    const size_t row = (size_t) (p.ld > 0 ? p.ld : 32) * sizeof(float);
    if (row > scratch_cap) {
        char msg[192];
        std::snprintf(msg, sizeof msg, "scratch cap %zu bytes holds no query row of %lld keys (%zu bytes)", scratch_cap,
                      (long long) n_k, row);
        fail(msg);
    }
    int64_t c = (int64_t) (scratch_cap / row);
    if (c > nq) c = nq;
    if (c > 65535) c = 65535;
    p.q_chunk = c < 1 ? 1 : c;
    p.scratch_bytes = (size_t) p.q_chunk * row;
    return p;
}

IndexerScorer indexer_resolve(IndexerScorer which, int64_t nq) {
    if (which == IndexerScorer::Auto)
        return nq >= kIndexerTcMinQueries && device_cc() >= 80 ? IndexerScorer::TensorCore : IndexerScorer::Simt;
    if (which == IndexerScorer::TensorCore && device_cc() < 80) fail("the tensor-core scorer needs sm_80 or newer");
    return which;
}

void indexer_scores(const float* q, const float* w, const int32_t* pos, int64_t nq, const uint16_t* keys, int64_t n_k,
                    const IndexerDims& d, float* scores, int64_t ld, IndexerScorer which, void* stream) {
    check_dims(d, nq, n_k, ld);
    if (reinterpret_cast<uintptr_t>(keys) % 16 != 0) fail("keys are not 16-byte aligned");
    if (nq == 0 || n_k == 0) return;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    if (indexer_resolve(which, nq) == IndexerScorer::TensorCore) {
        const dim3 grid((unsigned) ((n_k + TC_TILES * TC_NT - 1) / (TC_TILES * TC_NT)),
                        (unsigned) ((nq + TC_WARPS - 1) / TC_WARPS));
        if (d.n_heads <= 16)
            scores_tc_kernel<1><<<grid, TC_WARPS * 32, 0, st>>>(q, w, pos, nq, keys, n_k, d.n_heads, d.ratio, scores, ld);
        else
            scores_tc_kernel<2><<<grid, TC_WARPS * 32, 0, st>>>(q, w, pos, nq, keys, n_k, d.n_heads, d.ratio, scores, ld);
        check_launch("scores_tc_kernel");
        return;
    }
    const dim3 grid((unsigned) ((n_k + SIMT_T - 1) / SIMT_T), (unsigned) nq);
    scores_simt_kernel<<<grid, SIMT_T, 0, st>>>(q, w, pos, keys, n_k, d.n_heads, d.ratio, scores, ld);
    check_launch("scores_simt_kernel");
}

void indexer_topk(const float* scores, int64_t ld, const int32_t* pos, int64_t nq, int64_t n_k, const IndexerDims& d,
                  int32_t* ids, void* stream) {
    check_dims(d, nq, n_k, ld);
    if (nq == 0) return;
    // a query sees at most n_k rows: they fit the registers' TK_T * TK_PER or are read again each pass
    if (n_k <= (int64_t) TK_T * TK_PER)
        topk_kernel<true><<<(unsigned) nq, TK_T, 0, static_cast<cudaStream_t>(stream)>>>(scores, ld, pos, n_k, d.ratio, d.top_k, ids);
    else
        topk_kernel<false><<<(unsigned) nq, TK_T, 0, static_cast<cudaStream_t>(stream)>>>(scores, ld, pos, n_k, d.ratio, d.top_k, ids);
    check_launch("topk_kernel");
}

void indexer_select(const float* q, const float* w, const int32_t* pos, int64_t nq, const uint16_t* keys, int64_t n_k,
                    const IndexerDims& d, const IndexerPlan& plan, float* scratch, IndexerScorer which, int32_t* ids,
                    void* stream) {
    if (plan.q_chunk < 1 || plan.ld < n_k) {
        char msg[160];
        std::snprintf(msg, sizeof msg, "plan q_chunk %lld ld %lld does not fit n_k %lld", (long long) plan.q_chunk,
                      (long long) plan.ld, (long long) n_k);
        fail(msg);
    }
    for (int64_t c0 = 0; c0 < nq; c0 += plan.q_chunk) {
        const int64_t n = nq - c0 < plan.q_chunk ? nq - c0 : plan.q_chunk;
        indexer_scores(q + c0 * d.n_heads * D, w + c0 * d.n_heads, pos + c0, n, keys, n_k, d, scratch, plan.ld, which,
                       stream);
        indexer_topk(scratch, plan.ld, pos + c0, n, n_k, d, ids + c0 * d.top_k, stream);
    }
}

}  // namespace strata::dsv41::engine
