// src/dsv41/engine/attention.cu - one attention layer on the GPU, see include/strata/dsv41/engine/attention.hpp.
//
// A call is a fixed sequence of launches on the engine's stream (no host synchronization):
//   projections   qmatmul (qmatmul.hpp) on the layer's quantized views, each input quantized once per layout its
//                 weights read (project below); an f16 view takes cuBLAS f16 x f16 -> f32 (gemm below, the test
//                 path) on the input converted to f16 (round to nearest even) once;
//   norms         rms_norm_kernel: a block per row, the squares summed in double, y = x * scale * w as the fork;
//   rope          rope_kernel: a thread per pair, the fork's float order (theta by repeated multiplication, the
//                 YaRN ramp, the two FMAs of its rotate), cos and sin from sincosf times the host's mscale;
//   window        window_store_kernel: kv rows to f16 at cell pos % window_cells;
//   compressor    pool_kernel (softmax over the block's r positions from the carried state and the chunk), the
//                 norm, rope at the block's first position, the f16 rows; carry_kernel keeps the incomplete block;
//   index keys    the key projection of the f16 latent, norm, rope, fwht_kernel (the fork's FWHT order), f16 keys;
//   indexer       the query projection, rope, FWHT, the scaled head weights, then indexer_select;
//   attention     prefill_attn_kernel for a prefill chunk and attn_kernel for a decode token (below), plus
//                 merge_kernel when the cells are split over blocks;
//   output        rope_kernel inverted (sin_sign -1) into f16 (a prefill chunk of one split: prefill_attn_kernel's
//                 epilogue instead), wo_a as one strided batched GEMM over the groups, wo_b.
// A decode token (attn_decode) runs the same kernels on the position in device memory, with the compressor's
// block decided on the device (compress_decode_kernel, store_block_kernel) and launch shapes that do not depend on
// the position, so a graph captured once serves every later token. With q2_K weights it takes the fused decode
// token's launches instead (the section of that name below), which store the same values.
#include "strata/dsv41/engine/attention.hpp"

#include "mmvq.cuh"
#include "strata/dsv41/engine/indexer.hpp"
#include "strata/dsv41/engine/kv_cache.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 attention: %s\n", msg);
    std::exit(1);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) failf("%s: %s", what, cudaGetErrorString(e));
}
void ckb(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) failf("%s: cuBLAS status %d", what, (int) s);
}
void check_launch(const char* kernel) { ck(cudaGetLastError(), kernel); }

constexpr float kLog2e = 1.4426950408889634f;

// ---------------------------------------------------------------- elementwise
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = kernels::f16_from_f32(x[i]);
}

void to_f16(const float* x, uint16_t* y, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    const int64_t blocks = std::min<int64_t>((n + 255) / 256, 4096);
    to_f16_kernel<<<(unsigned) blocks, 256, 0, st>>>(x, y, n);
    check_launch("to_f16_kernel");
}

__global__ void fill_pos_kernel(int32_t* pos, int64_t p0, int64_t T) {
    const int64_t t = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (t < T) pos[t] = (int32_t) (p0 + t);
}

// rows of n values: y = x * (1 / sqrt(mean(x^2) + eps)) * w, the squares summed in double (ggml's rms_norm), and
// optionally its f16 (yh). In place is allowed (every value is read before the block writes).
constexpr int NORM_T = 256;
// 1 / sqrt(mean(xr^2) + eps) over n values by a block of NORM_T threads (every thread gets it); red is reused by
// the next call once every thread has read it
__device__ __forceinline__ float rms_scale(const float* xr, int n, float eps, double* red) {
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += NORM_T) s += (double) (xr[i] * xr[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    double tot = 0.0;
#pragma unroll
    for (int i = 0; i < NORM_T / 32; ++i) tot += red[i];
    const float mean = (float) (tot / (double) n);
    return 1.0f / sqrtf(mean + eps);
}

__global__ void __launch_bounds__(NORM_T) rms_norm_kernel(const float* __restrict__ x, int64_t ldx, const float* __restrict__ w,
                                                          int n, float eps, float* y, int64_t ldy, uint16_t* yh, int64_t ldyh) {
    const int64_t row = blockIdx.x;
    const float* xr = x + row * ldx;
    __shared__ double red[NORM_T / 32];
    const float scale = rms_scale(xr, n, eps, red);
    for (int i = threadIdx.x; i < n; i += NORM_T) {
        const float v = xr[i] * scale * w[i];
        if (y) y[row * ldy + i] = v;
        if (yh) yh[row * ldyh + i] = kernels::f16_from_f32(v);
    }
}

void rms_norm(const float* x, int64_t ldx, const float* w, int n, float eps, float* y, int64_t ldy, uint16_t* yh,
              int64_t ldyh, int64_t rows, cudaStream_t st) {
    if (rows <= 0) return;
    rms_norm_kernel<<<(unsigned) rows, NORM_T, 0, st>>>(x, ldx, w, n, eps, y, ldy, yh, ldyh);
    check_launch("rms_norm_kernel");
}

// cos and sin of rotated pair `pair` at position p, in the fork's float order (ggml_rope_cache_init, rope_yarn)
__device__ __forceinline__ void rope_cs(const RopeArgs& a, int64_t p, int pair, float sin_sign, float& co, float& si) {
    float theta = (float) p;
    for (int i = 0; i < pair; ++i) theta *= a.theta_scale;
    const float theta_interp = a.freq_scale * theta;
    float th = theta_interp;
    if (a.ext_factor != 0.0f) {
        const float den = 0.001f > a.corr1 - a.corr0 ? 0.001f : a.corr1 - a.corr0;
        const float y = ((float) pair - a.corr0) / den;
        const float ramp = 1.f - fminf(1.f, fmaxf(0.f, y));
        const float rm = ramp * a.ext_factor;
        th = fmaf(theta_interp, 1.f - rm, theta * rm);
    }
    float s, c;
    sincosf(th, &s, &c);
    co = c * a.mscale;
    si = s * a.mscale * sin_sign;
}

// the pair (x0, x1) at value e of a head of hd values: rotated when e is among the last n_rot (the fork's
// x0' = fma(x0, cos, -(x1 sin)), x1' = fma(x0, sin, x1 cos)), else kept. rope_kernel and the fused decode kernels.
__device__ __forceinline__ void rope_pair(const RopeArgs& a, int64_t p, int e, int hd, float sin_sign, float& x0, float& x1) {
    const int nope = hd - a.n_rot;
    if (e < nope) return;
    float co, si;
    rope_cs(a, p, (e - nope) / 2, sin_sign, co, si);
    const float y0 = fmaf(x0, co, -(x1 * si));
    const float y1 = fmaf(x0, si, x1 * co);
    x0 = y0;
    x1 = y1;
}

// rows of `heads` heads of hd values: rope_pair on each adjacent pair, out to dst32 (may be src) or dst16. The
// position of row r is pos[r], or (pos_base + r) * pos_mul when pos is null (a compressed block's first position).
// in_place (dst32 is src at its stride, no dst16): only the last n_rot values of each head are read and written, the
// others stay as they are (q's rope at 16K moved 4 GiB for the 1/8 of it that rotates).
__global__ void rope_kernel(const float* __restrict__ src, int64_t ld_src, float* dst32, uint16_t* dst16, int64_t ld_dst,
                            int heads, int hd, const int32_t* __restrict__ pos, int64_t pos_base, int64_t pos_mul,
                            RopeArgs a, float sin_sign, int64_t rows, bool in_place) {
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    const int64_t p = pos ? (int64_t) pos[row] : (pos_base + row) * pos_mul;
    if (in_place) {
        const int rp = a.n_rot / 2, nope = hd - a.n_rot;
        for (int i = threadIdx.x; i < heads * rp; i += blockDim.x) {
            const int e = nope + 2 * (i % rp);
            float* s = dst32 + row * ld_dst + (int64_t) (i / rp) * hd + e;
            float x0 = s[0], x1 = s[1];
            rope_pair(a, p, e, hd, sin_sign, x0, x1);
            s[0] = x0;
            s[1] = x1;
        }
        return;
    }
    const int pairs = heads * hd / 2;
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        const int h = (2 * i) / hd, e = (2 * i) % hd;
        const float* s = src + row * ld_src + (int64_t) h * hd + e;
        float x0 = s[0], x1 = s[1];
        rope_pair(a, p, e, hd, sin_sign, x0, x1);
        const int64_t o = row * ld_dst + (int64_t) h * hd + e;
        if (dst32) { dst32[o] = x0; dst32[o + 1] = x1; }
        if (dst16) { dst16[o] = kernels::f16_from_f32(x0); dst16[o + 1] = kernels::f16_from_f32(x1); }
    }
}

void rope(const float* src, int64_t ld_src, float* dst32, uint16_t* dst16, int64_t ld_dst, int heads, int hd,
          const int32_t* pos, int64_t pos_base, int64_t pos_mul, const RopeArgs& a, float sin_sign, int64_t rows,
          cudaStream_t st) {
    if (rows <= 0) return;
    const bool in_place = dst32 == src && !dst16 && ld_src == ld_dst;
    rope_kernel<<<(unsigned) rows, 128, 0, st>>>(src, ld_src, dst32, dst16, ld_dst, heads, hd, pos, pos_base, pos_mul, a,
                                                 sin_sign, rows, in_place);
    check_launch("rope_kernel");
}

// one row of 128 by a warp: x * (1 / sqrt(128)), then the butterflies len 1 .. 64 (u + v, u - v), as
// ggml_compute_forward_fwht; sm is the warp's 128 floats of shared memory, xr may be in global or shared memory
__device__ __forceinline__ void fwht_row(float* xr, float* sm, int lane) {
    const float scale = 1.0f / sqrtf(128.0f);
    for (int j = lane; j < 128; j += 32) sm[j] = xr[j] * scale;
    __syncwarp();
    for (int len = 1; len < 128; len <<= 1) {
        for (int k = lane; k < 64; k += 32) {
            const int i = (k / len) * 2 * len + (k % len);
            const float u = sm[i], v = sm[i + len];
            sm[i] = u + v;
            sm[i + len] = u - v;
        }
        __syncwarp();
    }
    for (int j = lane; j < 128; j += 32) xr[j] = sm[j];
    __syncwarp();
}

// rows of 128, a warp per row
__global__ void fwht_kernel(float* x, int64_t rows) {
    __shared__ float s[4][128];
    const int r = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = blockIdx.x * 4 + r;
    if (row >= rows) return;
    fwht_row(x + row * 128, s[r], lane);
}

void fwht(float* x, int64_t rows, cudaStream_t st) {
    if (rows <= 0) return;
    fwht_kernel<<<(unsigned) ((rows + 3) / 4), 128, 0, st>>>(x, rows);
    check_launch("fwht_kernel");
}

__global__ void window_store_kernel(const float* __restrict__ kv, int64_t T, const int32_t* __restrict__ pos,
                                    uint16_t* __restrict__ ring, int64_t cells, int D) {
    const int64_t t = blockIdx.x;
    if (t >= T) return;
    uint16_t* dst = ring + ((int64_t) pos[t] % cells) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) dst[i] = kernels::f16_from_f32(kv[t * D + i]);
}

// block b (block first + b, positions (first + b) r ..) pools channel d over its r positions: softmax over the
// scores, the kv weighted by it. Positions before p0 come from the carried state (row p - (p0 - n_carry)).
__global__ void pool_kernel(const float* __restrict__ ckv, const float* __restrict__ cs, const float* __restrict__ carry_kv,
                            const float* __restrict__ carry_score, int64_t p0, int64_t n_carry, int r, int64_t first, int D,
                            float* __restrict__ pooled) {
    const int64_t b = blockIdx.x;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float mx = -INFINITY;
        for (int i = 0; i < r; ++i) {
            const int64_t p = (first + b) * r + i;
            const float s = p < p0 ? carry_score[(p - (p0 - n_carry)) * D + d] : cs[(p - p0) * D + d];
            mx = fmaxf(mx, s);
        }
        float sum = 0.f, acc = 0.f;
        for (int i = 0; i < r; ++i) {
            const int64_t p = (first + b) * r + i;
            const float s = p < p0 ? carry_score[(p - (p0 - n_carry)) * D + d] : cs[(p - p0) * D + d];
            const float v = p < p0 ? carry_kv[(p - (p0 - n_carry)) * D + d] : ckv[(p - p0) * D + d];
            const float e = expf(s - mx);
            sum += e;
            acc = fmaf(e, v, acc);
        }
        pooled[b * D + d] = acc / sum;
    }
}

// the incomplete block's positions end r .. p0 + T - 1 into the state: a position of this chunk is copied to row
// p - end r; one from the old state (no block completed) is already at that row
__global__ void carry_kernel(const float* __restrict__ ckv, const float* __restrict__ cs, float* carry_kv, float* carry_score,
                             int64_t p0, int64_t T, int64_t base, int D) {
    const int64_t p = base + blockIdx.x;
    if (p < p0 || p >= p0 + T) return;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        carry_kv[(p - base) * D + d] = ckv[(p - p0) * D + d];
        carry_score[(p - base) * D + d] = cs[(p - p0) * D + d];
    }
}

// decode: pool_kernel's block and carry_kernel's row for one token at p = pos[0], read on the device. The
// token completes block p / r when (p + 1) % r == 0: pooled from the carried positions 0 .. r - 2 and this one (the
// same sums in the same order as pool_kernel); otherwise it is carried at row p % r. blk_pos[0] = (p / r) r, the
// block's first position (its rows' rope).
__global__ void compress_decode_kernel(const float* __restrict__ ckv, const float* __restrict__ cs, float* carry_kv,
                                       float* carry_score, const int32_t* __restrict__ pos, int r, int D,
                                       float* __restrict__ pooled, int32_t* __restrict__ blk_pos) {
    const int64_t p = pos[0];
    if (threadIdx.x == 0) blk_pos[0] = (int32_t) (p / r * r);
    const bool done = (p + 1) % r == 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        if (!done) {
            carry_kv[(p % r) * D + d] = ckv[d];
            carry_score[(p % r) * D + d] = cs[d];
            continue;
        }
        float mx = -INFINITY;
        for (int i = 0; i < r; ++i) mx = fmaxf(mx, i < r - 1 ? carry_score[i * D + d] : cs[d]);
        float sum = 0.f, acc = 0.f;
        for (int i = 0; i < r; ++i) {
            const float s = i < r - 1 ? carry_score[i * D + d] : cs[d];
            const float v = i < r - 1 ? carry_kv[i * D + d] : ckv[d];
            const float e = expf(s - mx);
            sum += e;
            acc = fmaf(e, v, acc);
        }
        pooled[d] = acc / sum;
    }
}

// decode: f16(src [n]) into row p / r of dst when the token at p = pos[0] completes its block (the rows' and the
// index keys' to_f16 of attn_prefill)
__global__ void store_block_kernel(const float* __restrict__ src, uint16_t* __restrict__ dst, const int32_t* __restrict__ pos,
                                   int r, int n) {
    const int64_t p = pos[0];
    if ((p + 1) % r != 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[(p / r) * n + i] = kernels::f16_from_f32(src[i]);
}

// ---------------------------------------------------------------- attention
// One block per (token, group of 32 heads, split of the token's cells); 8 warps. The token's cells are its window
// positions (max(0, p - n_swa + 1) .. p, ring cells) then its compressed rows (the top-k ids, -1 entries skipped, or
// every visible row in dense mode). Per chunk of 32 cells: the f16 K rows to shared memory; s = q k on m16n8k16 (q
// rounded to f16 as the fork's mul_mat does: every product is exact, the sum in f32); the online softmax in base 2;
// p v with p split into f16 hi + lo (scaled by 2^14 so the lo half stays normal), V = K. One split writes the
// normalized output with the head's sink in the denominator; several write (m, l, acc) for merge_kernel.
constexpr int HG = 32;         // heads per block: two m16 tiles
constexpr int CH = 32;         // cells per chunk
constexpr int AT = 256;        // 8 warps
constexpr float kPScale = 16384.f;
constexpr int kMaxSplits = 256;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define DSV41_ATTN_MMA 1
#else
#define DSV41_ATTN_MMA 0   // f16 m16n8k16 needs sm_80: older targets get a trap, the host refuses them
#endif

__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, const uint32_t* b) {
#if DSV41_ATTN_MMA
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

__device__ __forceinline__ uint32_t pack2(uint16_t lo, uint16_t hi) { return (uint32_t) lo | ((uint32_t) hi << 16); }

template <int D>
struct AttnSmem {
    static constexpr int KS = D + 8;   // halves per row: 16-byte rows, conflict-free fragment reads
    uint16_t q[HG][KS];
    uint16_t k[CH][KS];
    float s[HG][CH + 1];
    float m[HG], l[HG], alpha[HG];
    const uint16_t* row[CH];
};

struct AttnArgs {
    const float* q;            // [T][n_head][D]
    const uint16_t* qh;        // [T][n_head][D] f16 of q (decode: the q_b epilogue's)
    const uint16_t* win;       // ring [win_cells][D]
    int64_t win_cells;
    const uint16_t* rows;      // the KV source's rows [n_rows][D]
    int64_t n_rows;
    const int32_t* ids;        // [T][top_k] or null (dense)
    int top_k;
    const int32_t* pos;        // [T]
    int n_swa, ratio, n_head;
    float scale_log2;          // log2(e) / sqrt(D)
    const float* sinks;        // [n_head]
    float* out;                // [T][n_head][D] (one split; prefill_attn_kernel: or null, the f32 output not kept)
    float* part;               // [T][splits][n_head][D + 2] (several)
    int n_splits;
    int64_t cps;               // cells per split; 0: a multiple of CH from the token's own cells and n_splits (decode)
    uint16_t* derope;          // prefill_attn_kernel with one split: [T][n_head * D] f16 of the de-roped output, or null
    RopeArgs rope;             // the layer's rope (the de-rope's)
};

// DEC (the fused decode): q read as f16 (qh, the same f16 the f32 path converts to), a split past the token's
// cells returns after marking itself empty (m = -inf, which merge skips), and every split writes partials
template <int D, bool DEC>
__global__ void __launch_bounds__(AT) attn_kernel(AttnArgs a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    AttnSmem<D>& S = *reinterpret_cast<AttnSmem<D>*>(smem_raw);
    constexpr int NT = D / 64;   // n8 tiles of p.v per warp (D / 8 dims each)
    const int split = blockIdx.x, hg = blockIdx.y;
    const int64_t tok = blockIdx.z;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t p = a.pos[tok];
    const int64_t n_win = p + 1 < a.n_swa ? p + 1 : a.n_swa, win0 = p - n_win + 1;
    int64_t vis = 0, n_comp = 0;
    if (a.ratio) {
        vis = (p + 1) / a.ratio < a.n_rows ? (p + 1) / a.ratio : a.n_rows;
        n_comp = a.ids ? a.top_k : vis;
    }
    const int64_t n_cells = n_win + n_comp;
    const int64_t cps = a.cps > 0 ? a.cps : ((n_cells + a.n_splits - 1) / a.n_splits + CH - 1) / CH * CH;
    const int64_t c_begin = (int64_t) split * cps;
    const int64_t c_end = c_begin + cps < n_cells ? c_begin + cps : n_cells;
    if (DEC && c_begin >= n_cells) {
        if (t < HG && hg * HG + t < a.n_head) {
            float* o = a.part + ((tok * a.n_splits + split) * a.n_head + hg * HG + t) * (int64_t) (D + 2);
            o[D] = -INFINITY;
            o[D + 1] = 0.f;
        }
        return;
    }

    if (DEC) {
        for (int i = t; i < HG * D / 8; i += AT) {
            const int h = i / (D / 8), c = i % (D / 8), hh = hg * HG + h;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (hh < a.n_head) v = __ldg(reinterpret_cast<const uint4*>(a.qh + (tok * a.n_head + hh) * D) + c);
            *reinterpret_cast<uint4*>(&S.q[h][c * 8]) = v;
        }
    } else {
        for (int i = t; i < HG * D; i += AT) {
            const int h = i / D, d = i % D, hh = hg * HG + h;
            S.q[h][d] = hh < a.n_head ? kernels::f16_from_f32(a.q[(tok * a.n_head + hh) * D + d]) : (uint16_t) 0;
        }
    }
    if (t < HG) { S.m[t] = -INFINITY; S.l[t] = 0.f; }

    float acc[2][NT][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0.f;

    for (int64_t c0 = c_begin; c0 < c_end; c0 += CH) {
        __syncthreads();   // q and the state are ready; the previous chunk is done with k and s
        if (t < CH) {
            const int64_t idx = c0 + t;
            const uint16_t* ptr = nullptr;
            if (idx < c_end) {
                if (idx < n_win) {
                    ptr = a.win + ((win0 + idx) % a.win_cells) * D;
                } else {
                    const int64_t j = idx - n_win;
                    const int64_t rr = a.ids ? (int64_t) a.ids[tok * a.top_k + j] : j;
                    if (rr >= 0 && rr < vis) ptr = a.rows + rr * D;
                }
            }
            S.row[t] = ptr;
        }
        __syncthreads();
        constexpr int PIECES = D / 8;
        for (int i = t; i < CH * PIECES; i += AT) {
            const int c = i / PIECES, pc = i % PIECES;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (S.row[c]) v = __ldg(reinterpret_cast<const uint4*>(S.row[c]) + pc);
            *reinterpret_cast<uint4*>(&S.k[c][pc * 8]) = v;
        }
        __syncthreads();
        // scores: warp w takes m tile w / 4 (16 heads) and n tile w % 4 (8 cells) over all D
        {
            const int mt = warp >> 2, nt = warp & 3;
            float c4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
            for (int k0 = 0; k0 < D; k0 += 16) {
                uint32_t af[4], bf[2];
                af[0] = *reinterpret_cast<const uint32_t*>(&S.q[mt * 16 + gid][k0 + 2 * tig]);
                af[1] = *reinterpret_cast<const uint32_t*>(&S.q[mt * 16 + gid + 8][k0 + 2 * tig]);
                af[2] = *reinterpret_cast<const uint32_t*>(&S.q[mt * 16 + gid][k0 + 2 * tig + 8]);
                af[3] = *reinterpret_cast<const uint32_t*>(&S.q[mt * 16 + gid + 8][k0 + 2 * tig + 8]);
                bf[0] = *reinterpret_cast<const uint32_t*>(&S.k[nt * 8 + gid][k0 + 2 * tig]);
                bf[1] = *reinterpret_cast<const uint32_t*>(&S.k[nt * 8 + gid][k0 + 2 * tig + 8]);
                mma_f16(c4, af, bf);
            }
            const int r0 = mt * 16 + gid, cc = nt * 8 + 2 * tig;
            const bool v0 = S.row[cc] != nullptr, v1 = S.row[cc + 1] != nullptr;
            S.s[r0][cc] = v0 ? c4[0] * a.scale_log2 : -INFINITY;
            S.s[r0][cc + 1] = v1 ? c4[1] * a.scale_log2 : -INFINITY;
            S.s[r0 + 8][cc] = v0 ? c4[2] * a.scale_log2 : -INFINITY;
            S.s[r0 + 8][cc + 1] = v1 ? c4[3] * a.scale_log2 : -INFINITY;
        }
        __syncthreads();
        // online softmax: row t / 8, cells 4 (t % 8) .. +3; the 8 threads of a row are 8 lanes of one warp
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) { x[j] = S.s[r][sub * 4 + j]; mx = fmaxf(mx, x[j]); }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.m[r], m_new = fmaxf(m_old, mx);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = (x[j] == -INFINITY || m_new == -INFINITY) ? 0.f : exp2f(x[j] - m_new);
                S.s[r][sub * 4 + j] = e * kPScale;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float al = m_old == -INFINITY ? (m_new == -INFINITY ? 1.f : 0.f) : exp2f(m_old - m_new);
                S.alpha[r] = al;
                S.l[r] = fmaf(S.l[r], al, sum);
                S.m[r] = m_new;
            }
        }
        __syncthreads();
        // p.v: warp w owns dims [w D / 8, (w + 1) D / 8): NT n8 tiles, both m tiles, two k16 steps of cells
        {
            float al[2][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) { al[mt][0] = S.alpha[mt * 16 + gid]; al[mt][1] = S.alpha[mt * 16 + gid + 8]; }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[mt][j][0] *= al[mt][0]; acc[mt][j][1] *= al[mt][0];
                    acc[mt][j][2] *= al[mt][1]; acc[mt][j][3] *= al[mt][1];
                }
#pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                const int k0 = ks * 16;
                uint32_t ah[2][4], alo[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const int ra = mt * 16 + gid, rb = ra + 8;
                    const float pv[8] = {S.s[ra][k0 + 2 * tig], S.s[ra][k0 + 2 * tig + 1], S.s[rb][k0 + 2 * tig],
                                         S.s[rb][k0 + 2 * tig + 1], S.s[ra][k0 + 2 * tig + 8], S.s[ra][k0 + 2 * tig + 9],
                                         S.s[rb][k0 + 2 * tig + 8], S.s[rb][k0 + 2 * tig + 9]};
                    // the hardware conversion (round to nearest): any rounding of p is valid, the lo half carries
                    // what the hi half lost
#pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        const __half h0 = __float2half_rn(pv[2 * f]), h1 = __float2half_rn(pv[2 * f + 1]);
                        ah[mt][f] = pack2(__half_as_ushort(h0), __half_as_ushort(h1));
                        alo[mt][f] = pack2(__half_as_ushort(__float2half_rn(pv[2 * f] - __half2float(h0))),
                                           __half_as_ushort(__float2half_rn(pv[2 * f + 1] - __half2float(h1))));
                    }
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int n = warp * (D / 8) + j * 8 + gid;
                    uint32_t bf[2];
                    bf[0] = pack2(S.k[k0 + 2 * tig][n], S.k[k0 + 2 * tig + 1][n]);
                    bf[1] = pack2(S.k[k0 + 2 * tig + 8][n], S.k[k0 + 2 * tig + 9][n]);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        mma_f16(acc[mt][j], ah[mt], bf);
                        mma_f16(acc[mt][j], alo[mt], bf);
                    }
                }
            }
        }
    }
    __syncthreads();
    // epilogue: C fragment rows mt * 16 + gid (+ 8), dims warp D / 8 + j 8 + 2 tig (+ 1)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int r = mt * 16 + gid + half * 8, h = hg * HG + r;
            if (h >= a.n_head) continue;
            const float m = S.m[r], l = S.l[r];
            if (!DEC && a.n_splits == 1) {
                const float sink = a.sinks[h] * kLog2e, M = fmaxf(m, sink);
                const float em = m == -INFINITY ? 0.f : exp2f(m - M);
                const float f = em / (l * em + exp2f(sink - M)) * (1.0f / kPScale);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int n = warp * (D / 8) + j * 8 + 2 * tig;
                    float* o = a.out + (tok * a.n_head + h) * D + n;
                    o[0] = acc[mt][j][2 * half] * f;
                    o[1] = acc[mt][j][2 * half + 1] * f;
                }
            } else {
                float* o = a.part + ((tok * a.n_splits + split) * a.n_head + h) * (int64_t) (D + 2);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int n = warp * (D / 8) + j * 8 + 2 * tig;
                    o[n] = acc[mt][j][2 * half] * (1.0f / kPScale);
                    o[n + 1] = acc[mt][j][2 * half + 1] * (1.0f / kPScale);
                }
                if (warp == 0 && tig == 0) { o[D] = m; o[D + 1] = l; }
            }
        }
}

// the splits of one (token, head): out = sum_s acc_s 2^(m_s - M) / (sum_s l_s 2^(m_s - M) + 2^(sink - M))
__global__ void merge_kernel(const float* __restrict__ part, int n_splits, int n_head, int D, const float* __restrict__ sinks,
                             float* __restrict__ out) {
    const int64_t tok = blockIdx.x;
    const int h = blockIdx.y;
    const float sink = sinks[h] * kLog2e;
    const float* base = part + (tok * n_splits * n_head + h) * (int64_t) (D + 2);
    const int64_t stride = (int64_t) n_head * (D + 2);
    float M = sink;
    for (int s = 0; s < n_splits; ++s) M = fmaxf(M, base[s * stride + D]);
    float den = exp2f(sink - M);
    for (int s = 0; s < n_splits; ++s) {
        const float m = base[s * stride + D];
        if (m != -INFINITY) den += base[s * stride + D + 1] * exp2f(m - M);
    }
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < n_splits; ++s) {
            const float m = base[s * stride + D];
            if (m != -INFINITY) acc += base[s * stride + d] * exp2f(m - M);
        }
        out[(tok * n_head + h) * D + d] = acc / den;
    }
}

// ---------------------------------------------------------------- prefill attention
// attn_kernel's math for a prefill chunk at the tensor cores' rate: the same block
// (a token, HG heads, a split of its cells; 8 warps, chunks of CH cells) and the same online softmax, with p rounded
// once to f16 (no hi/lo split), so p.v takes half the MMAs, its A by ldmatrix from p in f16 and its B (V = K) by
// ldmatrix.trans. One f16 rounding is a relative error of up to 2^-11 per cell, the GPU test's whole tolerance (2^-11
// of sum p |v|) in the worst case: there is no margin by the worst-case bound. It holds by the errors' independence
// across cells (round to nearest, signs mixed: the sum grows as sqrt(n) of a typical 2^-13, not as n 2^-11) and by the
// running maximum's cell entering with p exactly 1; the test prints this kernel's worst ratio against the tolerance
// ("attention margin"), so the margin is the dev pod's measurement (85-92% on the A100-SXM4, 2026-10-09).
//
// (A100-SXM4, a 16384-token chunk: 25.5 ms per Full layer, and 15.5 ms per window
// layer, which has a fifth of the cells: a fixed cost per block about the size of the chunks' work). The SASS of the
// first version (sm_80, CUDA 12.8, dev VM) showed where: the q load was a loop of scalar loads, each one's branchy
// f16 conversion (f16_from_f32) waiting on its load before the next was issued, and the epilogue inlined 32 copies of
// rope_cs with sincosf (24,600 SASS lines). Now:
//   - the block's q: 16-byte loads, four in flight before their conversions, rounded by the hardware's conversion
//     (round to nearest even: f16_from_f32's bits for every finite value), staged in the K buffer and taken into
//     registers by ldmatrix: each warp keeps one m16 tile (16 heads) over a quarter of D as A fragments for the
//     whole call, so q.k reads only K from shared memory (the first version re-read q's 32 KiB in every chunk); the
//     four quarters' sums meet in the softmax;
//   - the chunk's K rows by cp.async (zero-filled for a skipped cell); the next chunk's top-k ids loaded at the top
//     of a chunk and made into row pointers after its q.k, so their latency is not in front of a barrier;
//   - the de-rope's cos and sin at the token's position computed once per block into shared memory (rope_cs, so the
//     same bits as rope_pair), and the f32 output written only when the engine keeps it (AttnEngine::set_attn_output,
//     the GPU test's attention stage): the layer reads the f16 de-rope, and the f32 copy was 2 GiB of writes per 16K
//     layer;
//   - two blocks per SM (__launch_bounds__(AT, 2), a cap of 128 registers): ptxas -v (dev VM, the CUDA 12.8 and 13.0
//     builder images) gives 128 registers at D 512 and 92 at D 256 on sm_80, no spill (sm_89: an 8-byte spill;
//     sm_120: 128 and 108, none); 54.4 KiB of shared memory at D 512 (one block per SM on sm_89 and sm_120).
template <int D>
struct PrefillSmem {
    static constexpr int KS = D + 8;    // 16-byte rows: the 8 rows of an ldmatrix on distinct banks
    static constexpr int PS = CH + 8;   // p's rows, 80 bytes
    uint16_t k[CH][KS];                 // the chunk's K rows; before the first chunk, the block's q ([HG][KS])
    float s[4][HG][CH + 1];             // q.k of each quarter of D
    uint16_t p[HG][PS];                 // exp2(s - m) kPScale in f16
    float m[HG], l[HG], alpha[HG];
    float2 cs[D / 2];                   // the de-rope's (cos, sin) of each rotated pair at the token's position
    const uint16_t* row[2][CH];         // the cells' rows (null: skipped) of this chunk and the next
};
static_assert(HG == CH, "prefill_attn_kernel stages the block's q in its K buffer");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
#if DSV41_ATTN_MMA
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
#else
    __trap();
#endif
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void* p) {
#if DSV41_ATTN_MMA
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
#else
    __trap();
#endif
}
// 16 bytes from src, or zeros when !valid (src is then not read but must be a mapped address)
__device__ __forceinline__ void cp16_or_zero(void* dst, const void* src, bool valid) {
#if DSV41_ATTN_MMA
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
#else
    __trap();
#endif
}
__device__ __forceinline__ void cp_async_wait_all() {
#if DSV41_ATTN_MMA
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
#endif
}
// two f32 as an f16 pair, lo in the low half, each rounded to nearest even (cvt.rn.f16x2.f32)
__device__ __forceinline__ uint32_t f16x2_rn(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// attn_kernel's cell list, in two halves so prefill_attn_kernel can load a chunk's ids ahead of making its pointers:
// the compressed row cell idx reads (its top-k id, or idx - n_win in dense mode), -1 for a window cell or one at or
// past c_end; then the cell's row: the window position's ring cell, row rr when it is visible, else null (skipped)
// (positions, cells and rows in 32 bits: the cache holds fewer than 2^31 positions)
__device__ __forceinline__ int cell_rowid(const AttnArgs& a, int tok, int idx, int c_end, int n_win) {
    if (idx >= c_end || idx < n_win) return -1;
    const int j = idx - n_win;
    return a.ids ? a.ids[(int64_t) tok * a.top_k + j] : j;
}
__device__ __forceinline__ const uint16_t* cell_ptr(const AttnArgs& a, int idx, int c_end, int n_win, int win0, int vis,
                                                    int rr, int D) {
    if (idx >= c_end) return nullptr;
    if (idx < n_win) return a.win + (int64_t) ((win0 + idx) % (int) a.win_cells) * D;
    return rr >= 0 && rr < vis ? a.rows + (int64_t) rr * D : nullptr;
}

template <int D>
__global__ void __launch_bounds__(AT, 2) prefill_attn_kernel(AttnArgs a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    PrefillSmem<D>& S = *reinterpret_cast<PrefillSmem<D>*>(smem_raw);
    constexpr int NT = D / 64;      // n8 tiles of p.v per warp (D / 8 dims each)
    constexpr int PIECES = D / 8;   // 16-byte pieces of a row
    constexpr int DQ = D / 4;       // q.k: the values of a warp's quarter of D
    constexpr int QKS = DQ / 16;    // and its k16 steps
    const int split = blockIdx.x, hg = blockIdx.y, tok = blockIdx.z;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int p = a.pos[tok];
    const int n_win = p + 1 < a.n_swa ? p + 1 : a.n_swa, win0 = p - n_win + 1;
    int vis = 0, n_comp = 0;
    if (a.ratio) {
        vis = (p + 1) / a.ratio < a.n_rows ? (p + 1) / a.ratio : (int) a.n_rows;
        n_comp = a.ids ? a.top_k : vis;
    }
    const int n_cells = n_win + n_comp, cps = (int) a.cps;
    const int c_begin = split * cps;
    const int c_end = c_begin + cps < n_cells ? c_begin + cps : n_cells;

    // the block's q in f16 into the K buffer ([HG][KS]): 16-byte loads, a batch of four in flight before its
    // conversions; the group's missing heads (the tiny dims) are zero
    {
        constexpr int ROW4 = D / 4, PER = HG * ROW4 / AT, BATCH = 4;
        static_assert(PER % BATCH == 0, "q's loads per thread");
        const float4* qb = reinterpret_cast<const float4*>(a.q + ((int64_t) tok * a.n_head + hg * HG) * D);
        const int heads = a.n_head - hg * HG;
#pragma unroll
        for (int b0 = 0; b0 < PER; b0 += BATCH) {
            float4 v[BATCH];
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const int i = (b0 + u) * AT + t;
                v[u] = i / ROW4 < heads ? __ldg(qb + i) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
#pragma unroll
            for (int u = 0; u < BATCH; ++u) {
                const int i = (b0 + u) * AT + t;
                *reinterpret_cast<uint2*>(&S.k[i / ROW4][(i % ROW4) * 4]) =
                    make_uint2(f16x2_rn(v[u].x, v[u].y), f16x2_rn(v[u].z, v[u].w));
            }
        }
    }
    if (t < HG) { S.m[t] = -INFINITY; S.l[t] = 0.f; }
    if (t < CH) S.row[0][t] = cell_ptr(a, c_begin + t, c_end, n_win, win0, vis, cell_rowid(a, tok, c_begin + t, c_end, n_win), D);
    const int nope = D - a.rope.n_rot;
    if (a.derope)
        for (int i = t; i < a.rope.n_rot / 2; i += AT) {
            float co, si;
            rope_cs(a.rope, p, i, -1.f, co, si);
            S.cs[i] = make_float2(co, si);
        }
    __syncthreads();
    // q's A fragments: warp w keeps m16 tile w % 2 (16 heads) over quarter w / 2 of D
    const int qmt = warp & 1, kq = warp >> 1;
    uint32_t qf[QKS][4];
#pragma unroll
    for (int ks = 0; ks < QKS; ++ks) ldsm_x4(qf[ks], &S.k[qmt * 16 + (lane & 15)][kq * DQ + ks * 16 + 8 * (lane >> 4)]);

    float acc[2][NT][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0.f;

    int buf = 0;
    for (int c0 = c_begin; c0 < c_end; c0 += CH, buf ^= 1) {
        __syncthreads();   // q's fragments are taken, the previous chunk is done with k and p, this chunk's rows are ready
        // the next chunk's row ids, in flight during this chunk's copy and q.k
        int rr_next = -1;
        if (t < CH) rr_next = cell_rowid(a, tok, c0 + CH + t, c_end, n_win);
        static_assert(AT % PIECES == 0, "a thread's pieces of the chunk");
#pragma unroll
        for (int u = 0; u < CH * PIECES / AT; ++u) {
            const int c = u * (AT / PIECES) + t / PIECES, pc = t % PIECES;
            const uint16_t* r = S.row[buf][c];
            cp16_or_zero(&S.k[c][pc * 8], r ? r + pc * 8 : a.win, r != nullptr);
        }
        cp_async_wait_all();
        __syncthreads();
        // scores: warp w's quarter of D for its 16 heads against the 32 cells, 16 at a time
#pragma unroll
        for (int np = 0; np < 2; ++np) {
            float c[2][4] = {};
#pragma unroll
            for (int ks = 0; ks < QKS; ++ks) {
                uint32_t bf[4];
                ldsm_x4(bf, &S.k[np * 16 + (lane & 7) + 8 * (lane >> 4)][kq * DQ + ks * 16 + 8 * ((lane >> 3) & 1)]);
                mma_f16(c[0], qf[ks], bf);
                mma_f16(c[1], qf[ks], bf + 2);
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int r0 = qmt * 16 + gid, cc = np * 16 + nt * 8 + 2 * tig;
                S.s[kq][r0][cc] = c[nt][0];
                S.s[kq][r0][cc + 1] = c[nt][1];
                S.s[kq][r0 + 8][cc] = c[nt][2];
                S.s[kq][r0 + 8][cc + 1] = c[nt][3];
            }
        }
        // the next chunk's rows (read after the next top barrier)
        if (t < CH) S.row[buf ^ 1][t] = cell_ptr(a, c0 + CH + t, c_end, n_win, win0, vis, rr_next, D);
        __syncthreads();
        // online softmax: row t / 8, cells 4 (t % 8) .. +3, the 8 threads of a row on 8 lanes of one warp
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int cc = sub * 4 + j;
                x[j] = S.row[buf][cc] ? ((S.s[0][r][cc] + S.s[1][r][cc]) + (S.s[2][r][cc] + S.s[3][r][cc])) * a.scale_log2
                                      : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.m[r], m_new = fmaxf(m_old, mx);
            float sum = 0.f, e[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                e[j] = (x[j] == -INFINITY || m_new == -INFINITY) ? 0.f : exp2f(x[j] - m_new);
                sum += e[j];
            }
            *reinterpret_cast<uint2*>(&S.p[r][sub * 4]) =
                make_uint2(f16x2_rn(e[0] * kPScale, e[1] * kPScale), f16x2_rn(e[2] * kPScale, e[3] * kPScale));
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float al = m_old == -INFINITY ? (m_new == -INFINITY ? 1.f : 0.f) : exp2f(m_old - m_new);
                S.alpha[r] = al;
                S.l[r] = fmaf(S.l[r], al, sum);
                S.m[r] = m_new;
            }
        }
        __syncthreads();
        // p.v: warp w owns dims [w D / 8, (w + 1) D / 8): NT n8 tiles, both m tiles, two k16 steps of cells
        {
            const int n0 = warp * (D / 8);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const float a0 = S.alpha[mt * 16 + gid], a1 = S.alpha[mt * 16 + gid + 8];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[mt][j][0] *= a0; acc[mt][j][1] *= a0;
                    acc[mt][j][2] *= a1; acc[mt][j][3] *= a1;
                }
            }
#pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                uint32_t pa[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) ldsm_x4(pa[mt], &S.p[mt * 16 + (lane & 15)][ks * 16 + 8 * (lane >> 4)]);
#pragma unroll
                for (int jp = 0; jp < NT / 2; ++jp) {
                    uint32_t bf[4];
                    ldsm_x4_trans(bf, &S.k[ks * 16 + (lane & 7) + 8 * ((lane >> 3) & 1)][n0 + jp * 16 + 8 * (lane >> 4)]);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        mma_f16(acc[mt][2 * jp], pa[mt], bf);
                        mma_f16(acc[mt][2 * jp + 1], pa[mt], bf + 2);
                    }
                }
            }
        }
    }
    __syncthreads();
    // epilogue: attn_kernel's (C fragment rows mt * 16 + gid (+ 8), dims warp D / 8 + j 8 + 2 tig (+ 1)); with one
    // split the normalized output and its de-rope (rope_pair's arithmetic on the table's cos and sin)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int r = mt * 16 + gid + half * 8, h = hg * HG + r;
            if (h >= a.n_head) continue;
            const float m = S.m[r], l = S.l[r];
            if (a.n_splits == 1) {
                const float sink = a.sinks[h] * kLog2e, M = fmaxf(m, sink);
                const float em = m == -INFINITY ? 0.f : exp2f(m - M);
                const float f = em / (l * em + exp2f(sink - M)) * (1.0f / kPScale);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int n = warp * (D / 8) + j * 8 + 2 * tig;
                    float x0 = acc[mt][j][2 * half] * f, x1 = acc[mt][j][2 * half + 1] * f;
                    const int64_t o = ((int64_t) tok * a.n_head + h) * D + n;
                    if (a.out) *reinterpret_cast<float2*>(a.out + o) = make_float2(x0, x1);
                    if (a.derope) {
                        if (n >= nope) {
                            const float2 c = S.cs[(n - nope) / 2];
                            const float y0 = fmaf(x0, c.x, -(x1 * c.y));
                            const float y1 = fmaf(x0, c.y, x1 * c.x);
                            x0 = y0;
                            x1 = y1;
                        }
                        *reinterpret_cast<uint32_t*>(a.derope + o) = f16x2_rn(x0, x1);
                    }
                }
            } else {
                float* o = a.part + (((int64_t) tok * a.n_splits + split) * a.n_head + h) * (int64_t) (D + 2);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int n = warp * (D / 8) + j * 8 + 2 * tig;
                    o[n] = acc[mt][j][2 * half] * (1.0f / kPScale);
                    o[n + 1] = acc[mt][j][2 * half + 1] * (1.0f / kPScale);
                }
                if (warp == 0 && tig == 0) { o[D] = m; o[D + 1] = l; }
            }
        }
}

// the kernels' shared memory limit, set by AttnEngine::init (outside any stream capture: the decode token's first
// launch may be inside one)
template <int D>
void attn_attrs() {
    const int smem = (int) sizeof(AttnSmem<D>);
    ck(cudaFuncSetAttribute(attn_kernel<D, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "cudaFuncSetAttribute(attn_kernel)");
    ck(cudaFuncSetAttribute(attn_kernel<D, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "cudaFuncSetAttribute(attn_kernel)");
    ck(cudaFuncSetAttribute(prefill_attn_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(PrefillSmem<D>)),
       "cudaFuncSetAttribute(prefill_attn_kernel)");
}

template <int D, bool DEC = false>
void launch_attn(const AttnArgs& a, int64_t T, int n_hg, cudaStream_t st) {
    const dim3 grid((unsigned) a.n_splits, (unsigned) n_hg, (unsigned) T);
    attn_kernel<D, DEC><<<grid, AT, sizeof(AttnSmem<D>), st>>>(a);
    check_launch("attn_kernel");
}

template <int D>
void launch_prefill_attn(const AttnArgs& a, int64_t T, int n_hg, cudaStream_t st) {
    const dim3 grid((unsigned) a.n_splits, (unsigned) n_hg, (unsigned) T);
    prefill_attn_kernel<D><<<grid, AT, sizeof(PrefillSmem<D>), st>>>(a);
    check_launch("prefill_attn_kernel");
}

// ---------------------------------------------------------------- projections
// Y[T][N] (f32, ldy) = alpha X[T][K] (f16, ldx) . W[N][K]^T (f16): Strata's Gemm::f16 call (src/prefill/gemm.cu)
void gemm(void* blas, const uint16_t* X, int64_t ldx, const void* W, float* Y, int64_t ldy, int64_t T, int64_t N,
          int64_t K, float alpha, const char* what) {
    if (T <= 0) return;
    const float beta = 0.0f;
    ckb(cublasGemmEx((cublasHandle_t) blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W, CUDA_R_16F,
                     (int) K, X, CUDA_R_16F, (int) ldx, &beta, Y, CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT),
        what);
}

__global__ void scale_kernel(float* y, int64_t ldy, int64_t n, float a) {
    const int64_t t = blockIdx.x;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) y[t * ldy + i] *= a;
}

// a projection weight of the layer: present, of the expected shape, f16 (the test path) or a matrix qmatmul takes
// with `groups`
void check_w(const QWeightView& w, int64_t k, int64_t rows, int groups, int layer, const char* what) {
    if (!w.data) failf("layer %d: %s is missing", layer, what);
    if (w.k != k || w.rows != rows)
        failf("layer %d: %s is %lld x %lld (rows x K), the dims want %lld x %lld", layer, what, (long long) w.rows,
              (long long) w.k, (long long) rows, (long long) k);
    if (w.type == kQTypeF16) return;
    if (!qtype_block_bytes(w.type)) failf("layer %d: %s has type %d (%s)", layer, what, w.type, qtype_name(w.type));
    char name[96];
    std::snprintf(name, sizeof name, "layer %d: %s", layer, what);
    qmatmul_check(w, groups, name);
}

// One projection input: T rows of k values, f32 (x, row stride ld) or f16 only (xh, the de-rope). It is converted at
// most once per form its weights ask for: the f16 copy h (contiguous; have_h when the norm already wrote it) for an
// f16 view, quantized activations in qbuf for a qmatmul view (requantized only if a weight reads another layout).
struct ProjInput {
    const float* x = nullptr;
    const uint16_t* xh = nullptr;
    int64_t ld = 0, T = 0, k = 0;
    uint16_t* h = nullptr;
    bool have_h = false;
    void* qbuf = nullptr;
    QActs q{};
    bool have_q = false;
};

// y[T][ldy] = alpha in . w^T; groups > 1: the strided per-group product of attn_output_a. f16 activations for an f16
// view and for qmatmul's dequant path (T above MMVQ's columns), int8 ones for MMVQ and MMQ.
void project(AttnEngine& e, ProjInput& in, const QWeightView& w, float* y, int64_t ldy, float alpha, int groups,
             const char* what) {
    cudaStream_t st = (cudaStream_t) e.stream();
    if (w.type == kQTypeF16 || e.qmm().path(in.T) == QPath::DequantF16) {
        const uint16_t* xh = in.xh;
        int64_t ldh = in.ld;
        if (!xh) {
            if (!in.have_h) {
                if (in.ld != in.k) failf("%s: an f16 copy of rows %lld apart (the inputs are contiguous)", what, (long long) in.ld);
                to_f16(in.x, in.h, in.T * in.k, st);
                in.have_h = true;
            }
            xh = in.h;
            ldh = in.k;
        }
        if (w.type != kQTypeF16) {
            qmatmul_f16(e.qmm(), w, xh, ldh, in.T, y, ldy, groups, st, what);
            if (alpha != 1.0f) {
                scale_kernel<<<(unsigned) in.T, 128, 0, st>>>(y, ldy, w.rows, alpha);
                check_launch("scale_kernel");
            }
            return;
        }
        if (groups == 1) {
            gemm(e.blas(), xh, ldh, w.data, y, ldy, in.T, w.rows, w.k, alpha, what);
            return;
        }
        const int64_t rg = w.rows / groups;
        const float zero = 0.f;
        ckb(cublasGemmStridedBatchedEx((cublasHandle_t) e.blas(), CUBLAS_OP_T, CUBLAS_OP_N, (int) rg, (int) in.T, (int) w.k, &alpha,
                                       w.data, CUDA_R_16F, (int) w.k, (long long) rg * w.k, xh, CUDA_R_16F, (int) ldh,
                                       (long long) w.k, &zero, y, CUDA_R_32F, (int) ldy, (long long) rg, groups,
                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
            what);
        return;
    }
    const QActLayout l = qact_layout(w.type, in.T);
    if (!in.have_q || in.q.layout != l) {
        qact_quantize(in.x, in.x ? nullptr : in.xh, in.ld, in.T, in.k, l, in.qbuf, st, in.q);
        in.have_q = true;
    }
    qmatmul_q(e.qmm(), w, in.q, y, ldy, groups, st, what);
    if (alpha != 1.0f) {
        scale_kernel<<<(unsigned) in.T, 128, 0, st>>>(y, ldy, w.rows, alpha);
        check_launch("scale_kernel");
    }
}

void* dalloc(size_t bytes, std::vector<void*>& allocs, size_t& total, const char* what, std::string& err) {
    void* p = nullptr;
    if (const cudaError_t e = cudaMalloc(&p, bytes ? bytes : 16); e != cudaSuccess) {
        char msg[192];
        std::snprintf(msg, sizeof msg, "workspace %s: %zu bytes: %s", what, bytes, cudaGetErrorString(e));
        err = msg;
        return nullptr;
    }
    allocs.push_back(p);
    total += bytes;
    return p;
}

void check_dims(const AttnDims& d) {
    if (d.head_dim != 256 && d.head_dim != 512) failf("head_dim %d: the attention kernel has 256 and 512", d.head_dim);
    if (d.idx_dim != kIndexerDim) failf("indexer key length %d: the indexer has %d", d.idx_dim, kIndexerDim);
    if (d.n_rot % 2 || d.n_rot > d.idx_dim || d.n_rot > d.head_dim) failf("n_rot %d", d.n_rot);
    if (d.n_head % d.o_groups) failf("n_head %d is not a multiple of o_groups %d", d.n_head, d.o_groups);
    if (d.idx_heads < 1 || d.idx_heads > kIndexerHeads || d.top_k < 1 || d.top_k > kIndexerMaxTopK)
        failf("indexer heads %d (1..%d), top_k %d (1..%d)", d.idx_heads, kIndexerHeads, d.top_k, kIndexerMaxTopK);
}

// A call's marks: an event at the start of each op it runs (in AttnOp order, each at most once) and one at its end
// (timing on; off, nothing is recorded)
struct Timer {
    AttnEngine& e;
    cudaStream_t st;
    int n = 0;
    int ops[kAttnOps] = {};
    void mark(AttnOp op) {
        if (n == kAttnOps) failf("more than %d attention marks in a call", kAttnOps);
        if (e.timing()) ck(cudaEventRecord((cudaEvent_t) e.event(n), st), "cudaEventRecord");
        ops[n++] = (int) op;
    }
    void end() {
        if (e.timing()) ck(cudaEventRecord((cudaEvent_t) e.event(n), st), "cudaEventRecord");
    }
};

// the op and stage times of a call's marks (timing on: the host waits for the last)
void read_times(AttnEngine& e, const Timer& tm) {
    if (!e.timing()) return;
    ck(cudaEventSynchronize((cudaEvent_t) e.event(tm.n)), "cudaEventSynchronize");
    AttnTimes t;
    for (int i = 0; i < tm.n; ++i) {
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, (cudaEvent_t) e.event(i), (cudaEvent_t) e.event(i + 1)), "cudaEventElapsedTime");
        t.op_ms[tm.ops[i]] += ms;
    }
    auto sum = [&](AttnOp a, AttnOp b) {
        float s = 0;
        for (int i = (int) a; i <= (int) b; ++i) s += t.op_ms[i];
        return s;
    };
    t.proj_ms = sum(AttnOp::QA, AttnOp::IdxW);
    t.compress_ms = sum(AttnOp::CompProj, AttnOp::IdxKeys);
    t.indexer_ms = sum(AttnOp::Select, AttnOp::Select);
    t.attention_ms = sum(AttnOp::Attend, AttnOp::Attend);
    t.output_ms = sum(AttnOp::Derope, AttnOp::WoB);
    t.total_ms = t.proj_ms + t.compress_ms + t.indexer_ms + t.attention_ms + t.output_ms;
    e.times_mut() = t;
}

// decode's split count, fixed per layer so a captured graph serves every position: the layer's largest cell count
// (the window, plus top_k selected rows or every row of its KV source's capacity) in splits of at least CH cells,
// while the token's blocks leave the device idle, within the partials' room
int decode_splits(const AttnEngine& e, const KvCache& kv, const AttnLayer& l, bool has_ids) {
    const AttnDims& d = e.dims();
    const int64_t rows = l.ratio ? kv.layer(l.kv_src).rows_cap : 0;
    const int64_t max_cells = std::min<int64_t>(kv.config().max_ctx, d.n_swa) + (l.ratio ? (has_ids ? d.top_k : rows) : 0);
    const int n_hg = (d.n_head + HG - 1) / HG;
    const int64_t target = 2 * (int64_t) e.sm_count();
    int64_t splits = std::min<int64_t>((max_cells + CH - 1) / CH, std::max<int64_t>(1, target / n_hg));
    splits = std::min<int64_t>(splits, (int64_t) (e.scratch().part_floats / ((size_t) d.n_head * (d.head_dim + 2))));
    return (int) std::max<int64_t>(1, std::min<int64_t>(splits, kMaxSplits));
}

// ---------------------------------------------------------------- the fused decode token
// a decode token's attention took 0.40 ms per layer for ~40 MB of q2_K weights (~21 us at the
// A100's bandwidth): about 25 launches, most of them one block (norms, ropes, quantizers, the compressor's
// bookkeeping) and MMVQ at 190-540 GB/s. With q2_K weights attn_decode runs these launches instead, each one the
// arithmetic of the kernels above in their order, so every value it stores is theirs bit for bit (the one exception:
// the attention's normalization is always merge's, which the split decode already took):
//   x to q8_1        qact_quantize, skipped when the caller hands x's q8_1 (hc_decode_step writes it)
//   x projections    one mmvq1 launch (mmvq.cuh): attn_q_a, attn_kv and the role's compressor kv and gate and
//                    indexer.proj (its scale in the epilogue)
//   qr and kv        decode_qkv_kernel: block 0 the q_a norm with qr's f16 and q8_1, block 1 the kv norm, its rope
//                    and the window cell
//   qr projections   one mmvq1 launch: attn_q_b with its rope and q's f16 (the attention's input) and the index
//                    source's indexer.attn_q_b with its rope (DecEpi)
//   bookkeeping      decode_book_kernel, a kv source or an index source only: the compressor's block (carried or
//                    pooled, normed, roped, stored), the key owner's index key (indexer.attn_k staged in shared
//                    memory, its norm, rope, FWHT and store), the index queries' FWHT
//   selection        indexer_select (or the forced rows)
//   attention        attn_kernel<D, true>: splits of CH cells (the decode's split count from the layer's largest
//                    cell count at CH per split, from checking the split count for T = 1: 20 at
//                    640 cells instead of 10), an empty split returns at once
//   merge            merge_derope_kernel: the splits merged with the sink, the de-rope, its f16 and q8_1
//   output           attn_output_a (grouped) and attn_output_b on qmatmul_q (mmvq1 at one column), oa's q8_1 between

enum DecTag : int { kTagPlain = 0, kTagQ = 1, kTagIq = 2 };

// the qr projections' epilogue: rope at pos[0] on the row pair (rope_kernel's arithmetic), q's f16 too
struct DecEpi {
    struct Args {
        const int32_t* pos;
        RopeArgs ra, rc;
        int D, ID;
        uint16_t* qh;
    };
    __device__ static void row(const Args& a, const mmvq::MmvqSeg& s, int g, int r, float v, float partner) {
        if (s.tag == kTagPlain) {
            s.y[(size_t) g * s.n_out + r] = v * s.alpha;
            return;
        }
        const bool q = s.tag == kTagQ;
        const int hd = q ? a.D : a.ID, e = r % hd;
        float x0 = (r & 1) ? partner : v, x1 = (r & 1) ? v : partner;
        rope_pair(q ? a.ra : a.rc, (int64_t) a.pos[0], e & ~1, hd, 1.f, x0, x1);
        const float out = (r & 1) ? x1 : x0;
        s.y[r] = out;
        if (q) a.qh[r] = kernels::f16_from_f32(out);
    }
};

// block 0: qr = rms(qr) * q_a_norm in place with its f16 and q8_1 (rms_norm_kernel, then qact_quantize from the f32);
// block 1: kv = rope(rms(kv) * kv_norm) in place and its f16 at the window cell of pos[0] (rms_norm_kernel,
// rope_kernel, window_store_kernel)
__global__ void __launch_bounds__(NORM_T) decode_qkv_kernel(float* qr, uint16_t* qrh, const float* __restrict__ q_a_norm,
                                                            int q_lora, mmvq::Q81Block* __restrict__ qrq, float* kv,
                                                            const float* __restrict__ kv_norm, int D,
                                                            const int32_t* __restrict__ pos, RopeArgs ra,
                                                            uint16_t* __restrict__ ring, int64_t cells, float eps) {
    __shared__ double red[NORM_T / 32];
    if (blockIdx.x == 0) {
        const float scale = rms_scale(qr, q_lora, eps, red);
        for (int i = threadIdx.x; i < q_lora; i += NORM_T) {   // q_lora % 256 == 0: whole warps per q8_1 block
            const float v = qr[i] * scale * q_a_norm[i];
            qr[i] = v;
            qrh[i] = kernels::f16_from_f32(v);
            mmvq::q8_1_warp(v, qrq[i / mmvq::Q8K], (int) (threadIdx.x & 31));
        }
        return;
    }
    const float scale = rms_scale(kv, D, eps, red);
    const int64_t p = pos[0];
    uint16_t* cell = ring + (p % cells) * D;
    for (int j = threadIdx.x; j < D / 2; j += NORM_T) {
        const int e = 2 * j;
        float x0 = kv[e] * scale * kv_norm[e], x1 = kv[e + 1] * scale * kv_norm[e + 1];
        rope_pair(ra, p, e, D, 1.f, x0, x1);
        kv[e] = x0;
        kv[e + 1] = x1;
        cell[e] = kernels::f16_from_f32(x0);
        cell[e + 1] = kernels::f16_from_f32(x1);
    }
}

constexpr int kKeyRows = 16;   // index key rows per pass of decode_book_kernel's in-block MMVQ

struct BookArgs {
    // compressor (a kv source; block 0)
    bool compress;
    const float* ckv;
    const float* cs;
    float* carry_kv;
    float* carry_score;
    int r, D;
    float* pooled;
    float* latent;
    uint16_t* latenth;
    float* rowsf;
    const float* comp_norm;
    uint16_t* rows;
    int32_t* blk_pos;
    // the key owner's index key
    const uint8_t* idx_k;   // q2_K [ID][D], 16-byte aligned
    const float* idx_k_norm;
    float* idx_kf;
    uint16_t* keys;
    int ID;
    // the index queries' FWHT (an index source; the last block): iq_rows rows of 128, 0 for none
    float* iq;
    int iq_rows;
    const int32_t* pos;
    RopeArgs rc;
    float eps;
};

// Shared memory of decode_book_kernel's compressor block: the staged index key matrix, the partial sums, the pooled
// and normed rows, the latent's q8_1, the key row.
size_t book_smem(const AttnDims& d) {
    const size_t D = (size_t) d.head_dim, ID = (size_t) d.idx_dim;
    return ID * (D / kQK) * sizeof(mmvq::Q2KBlock) + (size_t) mmvq::WARPS * kKeyRows * mmvq::WARP * 4 + 2 * D * 4 +
           (D / mmvq::Q8K) * sizeof(mmvq::Q81Block) + ID * 4;
}

__global__ void __launch_bounds__(NORM_T) decode_book_kernel(const BookArgs a) {
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ double red[NORM_T / 32];
    const int tid = (int) threadIdx.x, warp = tid / 32, lane = tid % 32;
    if (!a.compress || blockIdx.x > 0) {
        // the index queries: fwht_kernel's rows, a warp per row
        auto* sm = reinterpret_cast<float*>(smem);
        for (int row = warp; row < a.iq_rows; row += NORM_T / 32) fwht_row(a.iq + (int64_t) row * 128, sm + warp * 128, lane);
        return;
    }
    const int D = a.D, r = a.r;
    const int64_t p = a.pos[0];
    const bool done = (p + 1) % r == 0;
    if (tid == 0) a.blk_pos[0] = (int32_t) (p / r * r);
    const int bpr = D / (int) kQK;
    const int key_bytes = a.idx_k ? a.ID * bpr * (int) sizeof(mmvq::Q2KBlock) : 0;
    uint8_t* wk = smem;
    float* part = reinterpret_cast<float*>(smem + key_bytes);
    float* sp = part + mmvq::WARPS * kKeyRows * mmvq::WARP;
    float* sl = sp + D;
    auto* sq = reinterpret_cast<mmvq::Q81Block*>(sl + D);
    float* sk = reinterpret_cast<float*>(sq + D / mmvq::Q8K);
    // the index key matrix on its way while the block pools
    if (done && a.idx_k)
        for (int o = tid * 16; o < key_bytes; o += NORM_T * 16) mmvq::cp16(wk + o, a.idx_k + o);
    // compress_decode_kernel's block (or the carried row), then the norm, the rope and the row store
    for (int d = tid; d < D; d += NORM_T) {
        if (!done) {
            a.carry_kv[(p % r) * D + d] = a.ckv[d];
            a.carry_score[(p % r) * D + d] = a.cs[d];
            continue;
        }
        float mx = -INFINITY;
        for (int i = 0; i < r; ++i) mx = fmaxf(mx, i < r - 1 ? a.carry_score[i * D + d] : a.cs[d]);
        float sum = 0.f, acc = 0.f;
        for (int i = 0; i < r; ++i) {
            const float s = i < r - 1 ? a.carry_score[i * D + d] : a.cs[d];
            const float v = i < r - 1 ? a.carry_kv[i * D + d] : a.ckv[d];
            const float e = expf(s - mx);
            sum += e;
            acc = fmaf(e, v, acc);
        }
        sp[d] = acc / sum;
        a.pooled[d] = sp[d];
    }
    if (!done) return;
    __syncthreads();
    const float scale = rms_scale(sp, D, a.eps, red);
    for (int i = tid; i < D; i += NORM_T) {
        const float v = sp[i] * scale * a.comp_norm[i];
        sl[i] = v;
        a.latent[i] = v;
        a.latenth[i] = kernels::f16_from_f32(v);
    }
    __syncthreads();
    const int64_t blk = p / r * r, row = p / r;
    for (int j = tid; j < D / 2; j += NORM_T) {
        const int e = 2 * j;
        float x0 = sl[e], x1 = sl[e + 1];
        rope_pair(a.rc, blk, e, D, 1.f, x0, x1);
        a.rowsf[e] = x0;
        a.rowsf[e + 1] = x1;
        a.rows[row * D + e] = kernels::f16_from_f32(x0);
        a.rows[row * D + e + 1] = kernels::f16_from_f32(x1);
    }
    if (!a.idx_k) return;
    // the key projection of the f32 latent: its q8_1 (qact_quantize), then mmvq_kernel's sums per row on warps 0-3
    for (int b = warp; b < D / mmvq::Q8K; b += NORM_T / 32) mmvq::q8_1_warp(sl[b * mmvq::Q8K + lane], sq[b], lane);
    mmvq::cp16_wait();
    using F = mmvq::Q2KTraits;
    for (int r0 = 0; r0 < a.ID; r0 += kKeyRows) {
        if (tid < mmvq::THREADS) {
            float tmp[kKeyRows] = {};
            for (int kbx = tid / F::T; kbx < bpr; kbx += F::BPI) {
#pragma unroll
                for (int i = 0; i < kKeyRows; ++i) {
                    const auto* wr = reinterpret_cast<const mmvq::Q2KBlock*>(wk) + (size_t) (r0 + i) * bpr;
                    tmp[i] += F::apply(F::load(wr + kbx, F::kqs(tid)), sq + kbx * F::KBY, F::kqs(tid));
                }
            }
#pragma unroll
            for (int i = 0; i < kKeyRows; ++i) part[(warp * kKeyRows + i) * mmvq::WARP + lane] = tmp[i];
        }
        __syncthreads();
        for (int i = warp; i < kKeyRows; i += NORM_T / 32) {
            float s = part[i * mmvq::WARP + lane];
#pragma unroll
            for (int l = 1; l < mmvq::WARPS; ++l) s += part[(l * kKeyRows + i) * mmvq::WARP + lane];
            s = mmvq::warp_sum(s);
            if (lane == 0) sk[r0 + i] = s;
        }
        __syncthreads();
    }
    // its norm, rope at the block's first position, FWHT (warp 0), the f16 key
    const float ks = rms_scale(sk, a.ID, a.eps, red);
    for (int i = tid; i < a.ID; i += NORM_T) sk[i] = sk[i] * ks * a.idx_k_norm[i];
    __syncthreads();
    for (int j = tid; j < a.ID / 2; j += NORM_T) {
        const int e = 2 * j;
        rope_pair(a.rc, blk, e, a.ID, 1.f, sk[e], sk[e + 1]);
    }
    __syncthreads();
    if (warp == 0) fwht_row(sk, sp, lane);
    __syncthreads();
    for (int i = tid; i < a.ID; i += NORM_T) {
        a.idx_kf[i] = sk[i];
        a.keys[row * a.ID + i] = kernels::f16_from_f32(sk[i]);
    }
}

// head h, value d of thread d: merge_kernel's sums for the token's splits, the de-rope (rope_pair with sin -1 on the
// pair of d and d ^ 1), its f16 and the q8_1 of that f16 (qact_quantize from the de-rope's f16)
template <int D>
__global__ void __launch_bounds__(D) merge_derope_kernel(const float* __restrict__ part, int n_splits, int n_head,
                                                         const float* __restrict__ sinks, const int32_t* __restrict__ pos,
                                                         RopeArgs ra, float* __restrict__ attn, uint16_t* __restrict__ derope,
                                                         mmvq::Q81Block* __restrict__ drq) {
    const int h = (int) blockIdx.x, d = (int) threadIdx.x;
    const float sink = sinks[h] * kLog2e;
    const float* base = part + (int64_t) h * (D + 2);
    const int64_t stride = (int64_t) n_head * (D + 2);
    float M = sink;
    for (int s = 0; s < n_splits; ++s) M = fmaxf(M, base[s * stride + D]);
    float den = exp2f(sink - M);
    for (int s = 0; s < n_splits; ++s) {
        const float m = base[s * stride + D];
        if (m != -INFINITY) den += base[s * stride + D + 1] * exp2f(m - M);
    }
    float acc = 0.f;
    for (int s = 0; s < n_splits; ++s) {
        const float m = base[s * stride + D];
        if (m != -INFINITY) acc += base[s * stride + d] * exp2f(m - M);
    }
    const float v = acc / den;
    attn[(int64_t) h * D + d] = v;
    const float pv = __shfl_xor_sync(0xffffffffu, v, 1);
    float x0 = (d & 1) ? pv : v, x1 = (d & 1) ? v : pv;
    rope_pair(ra, (int64_t) pos[0], d & ~1, D, -1.f, x0, x1);
    const uint16_t f = kernels::f16_from_f32((d & 1) ? x1 : x0);
    derope[(int64_t) h * D + d] = f;
    mmvq::q8_1_warp(__half2float(__ushort_as_half(f)), drq[((int64_t) h * D + d) / mmvq::Q8K], d & 31);
}

// the fused token's mmvq1 plans (0: the matrices do not stage, the call takes the launches above instead)
struct DecPlan {
    int r_x = 0, r_q = 0;
};

DecPlan dec_plan(const AttnDims& d, const AttnLayer& l, const AttnLayerWeights& w) {
    const QWeightView* all[] = {&w.wq_a, &w.wq_b, &w.wkv, &w.wo_a, &w.wo_b, &w.comp_kv, &w.comp_gate, &w.idx_k, &w.idx_q_b, &w.idx_proj};
    for (const QWeightView* v : all)
        if (v->data && (v->type != kQTypeQ2K || (uintptr_t) v->data % 16)) return {};
    int nx[5], n = 0;
    nx[n++] = d.q_lora;
    nx[n++] = d.head_dim;
    if (l.kv_source) nx[n++] = d.head_dim;
    if (l.gate) nx[n++] = d.head_dim;
    if (l.index_source) nx[n++] = d.idx_heads;
    const size_t rb_x = (size_t) (d.n_embd / kQK) * sizeof(mmvq::Q2KBlock), rb_q = (size_t) (d.q_lora / kQK) * sizeof(mmvq::Q2KBlock);
    int nq[2] = {d.n_head * d.head_dim, d.idx_heads * d.idx_dim};
    DecPlan p;
    p.r_x = mmvq::mmvq1_plan(rb_x, nx, n, false);
    p.r_q = mmvq::mmvq1_plan(rb_q, nq, l.index_source ? 2 : 1, true);
    if (!p.r_x || !p.r_q) return {};
    return p;
}

// a matrix of a fused launch (DecTag tag, the plain epilogue's alpha)
void add_seg(mmvq::MmvqSegs& L, const QWeightView& w, float* y, float alpha, int tag) {
    mmvq::MmvqSeg& g = L.seg[L.n_seg++];
    g.w = static_cast<const uint8_t*>(w.data);
    g.y = y;
    g.n_out = (int) w.rows;
    g.alpha = alpha;
    g.tag = tag;
}

// attn_decode's launches with q2_K weights (the list above); false, before any launch, when the layer's matrices do
// not stage (dec_plan), for attn_decode's unfused launches
bool attn_decode_fused(AttnEngine& e, const AttnLayer& l, const AttnLayerWeights& w, KvCache& kv, const float* x,
                       const void* xq, int64_t p, float* out, ReaderMode mode, const int32_t* forced_rows, bool device_pos) {
    const AttnDims& d = e.dims();
    const DecPlan plan = dec_plan(d, l, w);
    if (!plan.r_x) return false;
    AttnScratch& s = e.scratch_mut();
    cudaStream_t st = (cudaStream_t) e.stream();
    KvLayerBuffers& L = kv.layer(l.index);
    const int D = d.head_dim, H = d.n_head, r = l.ratio, ID = d.idx_dim, IH = d.idx_heads;
    const int64_t E = d.n_embd, HD = (int64_t) H * D, OO = (int64_t) d.o_groups * d.o_lora;
    const RopeArgs ra = attn_rope_args(d, r != 0), rc = attn_rope_args(d, true);
    Timer tm{e, st};
    tm.mark(AttnOp::QA);
    s.T = 1;
    s.B = r && (p + 1) % r == 0 ? 1 : 0;
    s.first_block = r ? p / r : 0;
    if (!device_pos) {
        fill_pos_kernel<<<1, 32, 0, st>>>(s.pos, p, 1);
        check_launch("fill_pos_kernel");
    }
    e.mark_kernel(AttnKernel::XQuant);
    if (!xq) {
        QActs qa;
        qact_quantize(x, nullptr, E, 1, E, QActLayout::Q8_1, s.xq, st, qa);
        xq = s.xq;
    }
    e.mark_kernel(AttnKernel::XProj);
    const DecEpi::Args ea{s.pos, ra, rc, D, ID, s.qh};
    {
        mmvq::MmvqSegs X;
        X.x = static_cast<const mmvq::Q81Block*>(xq);
        X.n_in = (int) E;
        add_seg(X, w.wq_a, s.qr, 1.f, kTagPlain);
        add_seg(X, w.wkv, s.kv, 1.f, kTagPlain);
        if (l.kv_source) add_seg(X, w.comp_kv, s.ckv, 1.f, kTagPlain);
        if (l.gate) add_seg(X, w.comp_gate, s.cscore, 1.f, kTagPlain);
        if (l.index_source) add_seg(X, w.idx_proj, s.iw, 1.0f / sqrtf((float) (ID * IH)), kTagPlain);
        mmvq::mmvq1_launch<mmvq::Q2KTraits, DecEpi>(X, plan.r_x, 1, ea, st);
        check_launch("mmvq1_kernel (x projections)");
    }
    e.mark_kernel(AttnKernel::QkvNorm);
    decode_qkv_kernel<<<2, NORM_T, 0, st>>>(s.qr, s.qrh, w.q_a_norm, d.q_lora, static_cast<mmvq::Q81Block*>(s.qrq), s.kv,
                                            w.kv_norm, D, s.pos, ra, L.window, L.window_cells, d.rms_eps);
    check_launch("decode_qkv_kernel");
    e.mark_kernel(AttnKernel::QProj);
    {
        mmvq::MmvqSegs Q;
        Q.x = static_cast<const mmvq::Q81Block*>(s.qrq);
        Q.n_in = d.q_lora;
        add_seg(Q, w.wq_b, s.q, 1.f, kTagQ);
        if (l.index_source) add_seg(Q, w.idx_q_b, s.iq, 1.f, kTagIq);
        mmvq::mmvq1_launch<mmvq::Q2KTraits, DecEpi>(Q, plan.r_q, 1, ea, st);
        check_launch("mmvq1_kernel (qr projections)");
    }
    tm.mark(AttnOp::CompProj);

    if (l.kv_source || l.index_source) {
        e.mark_kernel(AttnKernel::Book);
        BookArgs b{};
        b.compress = l.kv_source;
        if (l.kv_source) {
            b.ckv = s.ckv;
            b.cs = l.gate ? s.cscore : s.ckv;
            b.carry_kv = L.carry_kv;
            b.carry_score = L.carry_score;
            b.r = r;
            b.D = D;
            b.pooled = s.pooled;
            b.latent = s.latent;
            b.latenth = s.latenth;
            b.rowsf = s.rowsf;
            b.comp_norm = w.comp_norm;
            b.rows = L.rows;
            b.blk_pos = s.blk_pos;
            if (l.key_owner) {
                b.idx_k = static_cast<const uint8_t*>(w.idx_k.data);
                b.idx_k_norm = w.idx_k_norm;
                b.idx_kf = s.idx_kf;
                b.keys = L.keys;
            }
            b.ID = ID;
        }
        if (l.index_source) {
            b.iq = s.iq;
            b.iq_rows = IH;
        }
        b.pos = s.pos;
        b.rc = rc;
        b.eps = d.rms_eps;
        decode_book_kernel<<<(l.kv_source ? 1 : 0) + (l.index_source ? 1 : 0), NORM_T, book_smem(d), st>>>(b);
        check_launch("decode_book_kernel");
    }
    tm.mark(AttnOp::Select);

    const int32_t* ids = nullptr;
    if (l.index_source && forced_rows) {
        e.mark_kernel(AttnKernel::Select);
        ck(cudaMemcpyAsync(L.ids, forced_rows, (size_t) d.top_k * 4, cudaMemcpyDeviceToDevice, st), "the forced rows");
        ids = L.ids;
    } else if (l.index_source) {
        e.mark_kernel(AttnKernel::Select);
        const KvLayerBuffers& K = kv.layer(l.key_src);
        const IndexerDims id{IH, d.top_k, r};
        const IndexerPlan ip = indexer_plan(1, K.rows_cap, s.idx_scratch_bytes);
        indexer_select(s.iq, s.iw, s.pos, 1, K.keys, K.rows_cap, id, ip, s.idx_scratch, IndexerScorer::Auto, L.ids, st);
        ids = L.ids;
    } else if (r && mode == ReaderMode::Reuse) {
        ids = kv.layer(l.topk_src).ids;
    }
    tm.mark(AttnOp::Attend);

    e.mark_kernel(AttnKernel::Attend);
    {
        const KvLayerBuffers* V = r ? &kv.layer(l.kv_src) : nullptr;
        AttnArgs a{};
        a.qh = s.qh;
        a.win = L.window;
        a.win_cells = L.window_cells;
        a.rows = V ? V->rows : nullptr;
        a.n_rows = V ? V->rows_cap : 0;
        a.ids = ids;
        a.top_k = d.top_k;
        a.pos = s.pos;
        a.n_swa = d.n_swa;
        a.ratio = r;
        a.n_head = H;
        a.scale_log2 = kLog2e / sqrtf((float) D);
        a.sinks = w.sinks;
        a.part = s.part;
        a.cps = 0;
        a.n_splits = decode_splits(e, kv, l, ids != nullptr);
        s.n_splits = a.n_splits;
        const int n_hg = (H + HG - 1) / HG;
        if (D == 256) launch_attn<256, true>(a, 1, n_hg, st);
        else launch_attn<512, true>(a, 1, n_hg, st);
    }
    e.mark_kernel(AttnKernel::Merge);
    auto* drq = static_cast<mmvq::Q81Block*>(s.drq);
    if (D == 256) merge_derope_kernel<256><<<H, 256, 0, st>>>(s.part, s.n_splits, H, w.sinks, s.pos, ra, s.attn, s.derope, drq);
    else merge_derope_kernel<512><<<H, 512, 0, st>>>(s.part, s.n_splits, H, w.sinks, s.pos, ra, s.attn, s.derope, drq);
    check_launch("merge_derope_kernel");
    tm.mark(AttnOp::WoA);

    e.mark_kernel(AttnKernel::WoA);
    qmatmul_q(e.qmm(), w.wo_a, QActs{QActLayout::Q8_1, 1, HD, s.drq}, s.oa, OO, d.o_groups, st, "attn_output_a");
    e.mark_kernel(AttnKernel::OaQuant);
    QActs oq;
    qact_quantize(s.oa, nullptr, OO, 1, OO, QActLayout::Q8_1, s.oaq, st, oq);
    e.mark_kernel(AttnKernel::WoB);
    qmatmul_q(e.qmm(), w.wo_b, oq, out, E, 1, st, "attn_output_b");
    e.mark_kernel(AttnKernel::Count);
    tm.end();
    read_times(e, tm);
    return true;
}

// ---------------------------------------------------------------- the batch's decode token
// The qr projections' epilogue for B rows (attn_decode_batch): DecEpi's rope at each row's own position on the row
// pairs of q (q's f16 too) and of the index queries, after qmatmul_q wrote the raw rows; a thread per pair.
__global__ void batch_qrope_kernel(float* __restrict__ q, uint16_t* __restrict__ qh, int64_t HD, int D, float* __restrict__ iq,
                                   int64_t IQ, int ID, const int32_t* __restrict__ pos, RopeArgs ra, RopeArgs rc) {
    const int b = (int) blockIdx.y;
    const int64_t pair = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t pos_b = pos[b];
    if (pair < HD / 2) {
        const int64_t r = 2 * pair;
        float* row = q + (int64_t) b * HD;
        float x0 = row[r], x1 = row[r + 1];
        rope_pair(ra, pos_b, (int) (r % D), D, 1.f, x0, x1);
        row[r] = x0;
        row[r + 1] = x1;
        qh[(int64_t) b * HD + r] = kernels::f16_from_f32(x0);
        qh[(int64_t) b * HD + r + 1] = kernels::f16_from_f32(x1);
        return;
    }
    const int64_t k = pair - HD / 2;
    if (!iq || k >= IQ / 2) return;
    const int64_t r = 2 * k;
    float* row = iq + (int64_t) b * IQ;
    float x0 = row[r], x1 = row[r + 1];
    rope_pair(rc, pos_b, (int) (r % ID), ID, 1.f, x0, x1);
    row[r] = x0;
    row[r + 1] = x1;
}

}  // namespace

const char* attn_kernel_name(AttnKernel k) {
    switch (k) {
        case AttnKernel::XQuant: return "x to q8_1";
        case AttnKernel::XProj: return "x projections";
        case AttnKernel::QkvNorm: return "qr, kv norms";
        case AttnKernel::QProj: return "qr projections";
        case AttnKernel::Book: return "compressor, keys";
        case AttnKernel::Select: return "selection";
        case AttnKernel::Attend: return "attention";
        case AttnKernel::Merge: return "merge, de-rope";
        case AttnKernel::WoA: return "wo_a";
        case AttnKernel::OaQuant: return "oa to q8_1";
        case AttnKernel::WoB: return "wo_b";
        case AttnKernel::Count: break;
    }
    return "?";
}

const char* attn_op_name(AttnOp op) {
    switch (op) {
        case AttnOp::QA: return "q_a";
        case AttnOp::QB: return "q_b";
        case AttnOp::QRope: return "q rope";
        case AttnOp::Kv: return "kv";
        case AttnOp::IdxQ: return "idx q";
        case AttnOp::IdxW: return "idx w";
        case AttnOp::CompProj: return "comp proj";
        case AttnOp::Pool: return "pool";
        case AttnOp::IdxKeys: return "idx keys";
        case AttnOp::Select: return "select";
        case AttnOp::Attend: return "attend";
        case AttnOp::Derope: return "de-rope";
        case AttnOp::WoA: return "wo_a";
        case AttnOp::WoB: return "wo_b";
        case AttnOp::Count: break;
    }
    return "?";
}

void AttnEngine::set_kernel_marks(bool on) {
    kmarks_on_ = on;
    // every event made here, outside any capture: a call marks at most kAttnKernels + 1 points
    while (on && kevents_.size() < (size_t) (2 * (kAttnKernels + 1))) {
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreate(&ev), "cudaEventCreate (kernel marks)");
        kevents_.push_back(ev);
    }
}

void AttnEngine::mark_kernel(AttnKernel k) {
    if (!kmarks_on_) return;
    if (kmarks_.size() == kevents_.size())
        failf("%zu kernel marks without clear_kernel_marks (one call marks at most %d)", kmarks_.size(), kAttnKernels + 1);
    void* ev = kevents_[kmarks_.size()];
    // external: inside a stream capture the record becomes a graph node that every replay runs (a plain record
    // would only order the capture)
    ck(cudaEventRecordWithFlags((cudaEvent_t) ev, (cudaStream_t) stream_, cudaEventRecordExternal), "cudaEventRecord (kernel mark)");
    kmarks_.push_back({(int) k, ev});
}

bool AttnEngine::read_kernel_marks(double* ms) const {
    for (size_t i = 0; i + 1 < kmarks_.size(); ++i) {
        if (kmarks_[i].first == (int) AttnKernel::Count) continue;
        float t = 0;
        if (cudaEventElapsedTime(&t, (cudaEvent_t) kmarks_[i].second, (cudaEvent_t) kmarks_[i + 1].second) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        ms[kmarks_[i].first] += t;
    }
    return true;
}

void AttnEngine::clear_kernel_marks() { kmarks_.clear(); }

AttnEngine::~AttnEngine() {
    for (void* p : allocs_) cudaFree(p);
    for (void* ev : events_)
        if (ev) cudaEventDestroy((cudaEvent_t) ev);
    for (void* ev : kevents_) cudaEventDestroy((cudaEvent_t) ev);
    if (blas_) cublasDestroy((cublasHandle_t) blas_);
}

bool AttnEngine::init(const AttnDims& d, int64_t max_chunk, int64_t max_rows, void* stream, std::string& err,
                      QPrefill prefill, QmmContext* shared) {
    if (blas_) failf("init called twice");
    check_dims(d);
    if (max_chunk < 1 || max_chunk > 65535) failf("max_chunk %lld outside 1..65535", (long long) max_chunk);
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        err = "no CUDA device";
        return false;
    }
    if (major < 8) {
        err = "the attention kernel needs sm_80 or newer (f16 m16n8k16)";
        return false;
    }
    ck(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
    if (d.head_dim == 256) attn_attrs<256>();
    else attn_attrs<512>();
    if (shared) {
        if (shared->prefill() != prefill ||
            (prefill == QPrefill::DequantF16 && shared->scratch_elems() < attn_max_matrix_elems(d)))
            failf("a shared qmatmul context of another prefill path or of %lld scratch values (the attention needs %lld)",
                  (long long) shared->scratch_elems(), (long long) attn_max_matrix_elems(d));
        qmm_ = shared;
    } else {
        if (!own_qmm_.init(err, prefill, attn_max_matrix_elems(d))) return false;
        ws_bytes_ += own_qmm_.device_bytes();
    }
    d_ = d;
    max_chunk_ = max_chunk;
    stream_ = stream;
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
        err = "cublasCreate failed";
        return false;
    }
    blas_ = h;
    ckb(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // a fixed workspace, so the handle never allocates on the way (as Strata's Gemm::init)
    const size_t ws = 32u << 20;
    if (!(blas_ws_ = dalloc(ws, allocs_, ws_bytes_, "cuBLAS", err))) return false;
    ckb(cublasSetWorkspace(h, blas_ws_, ws), "cublasSetWorkspace");
    const int64_t Tm = max_chunk, Bm = max_chunk + 1, E = d.n_embd, D = d.head_dim;
    const int64_t HD = (int64_t) d.n_head * D, IQ = (int64_t) d.idx_heads * d.idx_dim, OO = (int64_t) d.o_groups * d.o_lora;
    AttnScratch& s = s_;
    struct Buf { void** p; size_t bytes; const char* what; };
    const Buf bufs[] = {
        {(void**) &s.xh, (size_t) (Tm * E) * 2, "xh"},          {(void**) &s.qr, (size_t) (Tm * d.q_lora) * 4, "qr"},
        {(void**) &s.qrh, (size_t) (Tm * d.q_lora) * 2, "qrh"}, {(void**) &s.q, (size_t) (Tm * HD) * 4, "q"},
        {(void**) &s.kv, (size_t) (Tm * D) * 4, "kv"},          {(void**) &s.ckv, (size_t) (Tm * D) * 4, "ckv"},
        {(void**) &s.cscore, (size_t) (Tm * D) * 4, "cscore"},  {(void**) &s.pooled, (size_t) (Bm * D) * 4, "pooled"},
        {(void**) &s.latent, (size_t) (Bm * D) * 4, "latent"},  {(void**) &s.latenth, (size_t) (Bm * D) * 2, "latenth"},
        {(void**) &s.rowsf, (size_t) (Bm * D) * 4, "rowsf"},    {(void**) &s.idx_kf, (size_t) (Bm * d.idx_dim) * 4, "idx_kf"},
        {(void**) &s.iq, (size_t) (Tm * IQ) * 4, "iq"},         {(void**) &s.iw, (size_t) (Tm * d.idx_heads) * 4, "iw"},
        {(void**) &s.pos, (size_t) Tm * 4, "pos"},              {(void**) &s.attn, (size_t) (Tm * HD) * 4, "attn"},
        {(void**) &s.blk_pos, 4, "block position"},
        {(void**) &s.derope, (size_t) (Tm * HD) * 2, "derope"}, {(void**) &s.oa, (size_t) (Tm * OO) * 4, "oa"},
        {(void**) &s.oah, (size_t) (Tm * OO) * 2, "oah"},       {(void**) &s.qh, (size_t) HD * 2, "q (f16, decode)"},
        {&s.xq, qact_bytes_max(prefill, Tm, E), "x (quantized)"},
        {&s.qrq, qact_bytes_max(prefill, Tm, d.q_lora), "qr (quantized)"},
        {&s.latq, qact_bytes_max(prefill, Bm, D), "latent (quantized)"},
        {&s.drq, qact_bytes_max(prefill, Tm, HD), "de-rope (quantized)"},
        {&s.oaq, qact_bytes_max(prefill, Tm, OO), "oa (quantized)"},
    };
    for (const Buf& b : bufs)
        if (!(*b.p = dalloc(b.bytes, allocs_, ws_bytes_, b.what, err))) return false;
    s.idx_scratch_bytes = std::min<size_t>(kIndexerScratchCap, (size_t) Tm * (size_t) ((max_rows + 31) / 32 * 32) * 4);
    s.idx_scratch_bytes = std::max<size_t>(s.idx_scratch_bytes, (size_t) ((max_rows + 31) / 32 * 32) * 4);
    if (!(s.idx_scratch = (float*) dalloc(s.idx_scratch_bytes, allocs_, ws_bytes_, "indexer scratch", err))) return false;
    s.part_floats = (size_t) kMaxSplits * (size_t) d.n_head * (size_t) (D + 2);
    if (!(s.part = (float*) dalloc(s.part_floats * 4, allocs_, ws_bytes_, "split-K partials", err))) return false;
    for (auto& ev : events_) {
        cudaEvent_t e = nullptr;
        ck(cudaEventCreate(&e), "cudaEventCreate");
        ev = e;
    }
    return true;
}

bool AttnEngine::init_batch(int max_batch, std::string& err) {
    if (!blas_ || max_batch_ || max_batch < 1 || max_batch > max_chunk_)
        failf("init_batch(%d): before init, twice, or outside 1..%lld rows", max_batch, (long long) max_chunk_);
    const int64_t HD = (int64_t) d_.n_head * d_.head_dim;
    if (!(s_.qhb = (uint16_t*) dalloc((size_t) max_batch * HD * 2, allocs_, ws_bytes_, "batch q (f16)", err)) ||
        !(s_.blk_posb = (int32_t*) dalloc((size_t) max_batch * 4, allocs_, ws_bytes_, "batch block positions", err)))
        return false;
    max_batch_ = max_batch;
    return true;
}

void attn_check_weights(const AttnDims& d, const AttnLayer& l, const AttnLayerWeights& w) {
    const int64_t E = d.n_embd, D = d.head_dim, HD = (int64_t) d.n_head * D, OO = (int64_t) d.o_groups * d.o_lora;
    check_w(w.wq_a, E, d.q_lora, 1, l.index, "attn_q_a");
    check_w(w.wq_b, d.q_lora, HD, 1, l.index, "attn_q_b");
    check_w(w.wkv, E, D, 1, l.index, "attn_kv");
    check_w(w.wo_a, HD / d.o_groups, OO, d.o_groups, l.index, "attn_output_a");
    check_w(w.wo_b, OO, E, 1, l.index, "attn_output_b");
    if (l.kv_source) check_w(w.comp_kv, E, D, 1, l.index, "attn_compressor_kv");
    if (l.gate) check_w(w.comp_gate, E, D, 1, l.index, "attn_compressor_gate");
    if (l.key_owner) check_w(w.idx_k, D, d.idx_dim, 1, l.index, "indexer.attn_k");
    if (l.index_source) {
        check_w(w.idx_q_b, d.q_lora, (int64_t) d.idx_heads * d.idx_dim, 1, l.index, "indexer.attn_q_b");
        check_w(w.idx_proj, E, d.idx_heads, 1, l.index, "indexer.proj");
    }
}

int64_t attn_max_matrix_elems(const AttnDims& d) {
    const int64_t E = d.n_embd, D = d.head_dim, HD = (int64_t) d.n_head * D, OO = (int64_t) d.o_groups * d.o_lora;
    const int64_t IQ = (int64_t) d.idx_heads * d.idx_dim;
    return std::max({(int64_t) d.q_lora * E, HD * d.q_lora, D * E, OO * (HD / d.o_groups), E * OO, (int64_t) d.idx_dim * D,
                     IQ * d.q_lora, (int64_t) d.idx_heads * E});
}

void attn_prefill(AttnEngine& e, const AttnLayer& l, const AttnLayerWeights& w, KvCache& kv, const float* x, int64_t T,
                  int64_t p0, float* out, ReaderMode mode, const int32_t* forced_rows) {
    const AttnDims& d = e.dims();
    AttnScratch& s = e.scratch_mut();
    cudaStream_t st = (cudaStream_t) e.stream();
    if (T < 1 || T > e.max_chunk()) failf("layer %d: chunk of %lld tokens (1..%lld)", l.index, (long long) T, (long long) e.max_chunk());
    if (T > kv.config().max_chunk) failf("layer %d: chunk of %lld tokens, the cache's window takes %lld", l.index, (long long) T, (long long) kv.config().max_chunk);
    KvLayerBuffers& L = kv.layer(l.index);
    if (L.next_pos != p0) failf("layer %d: chunk at position %lld, the layer is at %lld", l.index, (long long) p0, (long long) L.next_pos);
    if (p0 + T > kv.config().max_ctx) failf("layer %d: positions up to %lld, the cache holds %lld", l.index, (long long) (p0 + T), (long long) kv.config().max_ctx);
    const int D = d.head_dim, H = d.n_head, r = l.ratio, ID = d.idx_dim, IH = d.idx_heads;
    const int64_t E = d.n_embd, HD = (int64_t) H * D, OO = (int64_t) d.o_groups * d.o_lora;
    const RopeArgs ra = attn_rope_args(d, r != 0), rc = attn_rope_args(d, true);
    attn_check_weights(d, l, w);
    Timer tm{e, st};
    tm.mark(AttnOp::QA);
    s.T = T;
    s.B = 0;
    s.first_block = 0;
    fill_pos_kernel<<<(unsigned) ((T + 255) / 256), 256, 0, st>>>(s.pos, p0, T);
    check_launch("fill_pos_kernel");
    ProjInput px{x, nullptr, E, T, E, s.xh, false, s.xq};
    ProjInput pqr{s.qr, nullptr, d.q_lora, T, d.q_lora, s.qrh, true, s.qrq};

    // query: qr = rms(x wq_a) * q_a_norm, q = rope(qr wq_b)
    project(e, px, w.wq_a, s.qr, d.q_lora, 1.f, 1, "attn_q_a");
    rms_norm(s.qr, d.q_lora, w.q_a_norm, d.q_lora, d.rms_eps, s.qr, d.q_lora, s.qrh, d.q_lora, T, st);
    tm.mark(AttnOp::QB);
    project(e, pqr, w.wq_b, s.q, HD, 1.f, 1, "attn_q_b");
    tm.mark(AttnOp::QRope);
    rope(s.q, HD, s.q, nullptr, HD, H, D, s.pos, 0, 0, ra, 1.f, T, st);
    // window: kv = rope(rms(x wkv) * kv_norm) into the ring
    tm.mark(AttnOp::Kv);
    project(e, px, w.wkv, s.kv, D, 1.f, 1, "attn_kv");
    rms_norm(s.kv, D, w.kv_norm, D, d.rms_eps, s.kv, D, nullptr, 0, T, st);
    rope(s.kv, D, s.kv, nullptr, D, 1, D, s.pos, 0, 0, ra, 1.f, T, st);
    window_store_kernel<<<(unsigned) T, 128, 0, st>>>(s.kv, T, s.pos, L.window, L.window_cells, D);
    check_launch("window_store_kernel");
    // index source: its query side (the keys come below or from its key source)
    if (l.index_source) {
        tm.mark(AttnOp::IdxQ);
        project(e, pqr, w.idx_q_b, s.iq, (int64_t) IH * ID, 1.f, 1, "indexer.attn_q_b");
        rope(s.iq, (int64_t) IH * ID, s.iq, nullptr, (int64_t) IH * ID, IH, ID, s.pos, 0, 0, rc, 1.f, T, st);
        fwht(s.iq, T * IH, st);
        tm.mark(AttnOp::IdxW);
        project(e, px, w.idx_proj, s.iw, IH, 1.0f / sqrtf((float) (ID * IH)), 1, "indexer.proj");
    }

    // compressor: the blocks the chunk completes, from the carried positions and the chunk
    if (l.kv_source) {
        tm.mark(AttnOp::CompProj);
        project(e, px, w.comp_kv, s.ckv, D, 1.f, 1, "attn_compressor_kv");
        const float* cs = s.ckv;
        if (l.gate) {
            project(e, px, w.comp_gate, s.cscore, D, 1.f, 1, "attn_compressor_gate");
            cs = s.cscore;
        }
        const int64_t n_carry = p0 % r, first = p0 / r, end = (p0 + T) / r, B = end - first;
        if (end > L.rows_cap) failf("layer %d: %lld rows, the cache holds %lld", l.index, (long long) end, (long long) L.rows_cap);
        s.B = B;
        s.first_block = first;
        tm.mark(AttnOp::Pool);
        if (B > 0) {
            pool_kernel<<<(unsigned) B, 128, 0, st>>>(s.ckv, cs, L.carry_kv, L.carry_score, p0, n_carry, r, first, D, s.pooled);
            check_launch("pool_kernel");
            rms_norm(s.pooled, D, w.comp_norm, D, d.rms_eps, s.latent, D, s.latenth, D, B, st);
            rope(s.latent, D, s.rowsf, nullptr, D, 1, D, nullptr, first, r, rc, 1.f, B, st);
            to_f16(s.rowsf, L.rows + first * D, B * D, st);
            if (l.key_owner) {
                tm.mark(AttnOp::IdxKeys);
                ProjInput plat{s.latent, nullptr, D, B, D, s.latenth, true, s.latq};
                project(e, plat, w.idx_k, s.idx_kf, ID, 1.f, 1, "indexer.attn_k");
                rms_norm(s.idx_kf, ID, w.idx_k_norm, ID, d.rms_eps, s.idx_kf, ID, nullptr, 0, B, st);
                rope(s.idx_kf, ID, s.idx_kf, nullptr, ID, 1, ID, nullptr, first, r, rc, 1.f, B, st);
                fwht(s.idx_kf, B, st);
                to_f16(s.idx_kf, L.keys + first * ID, B * ID, st);
            }
        }
        const int64_t base = end * r, n_new = p0 + T - base;
        if (n_new > 0) {
            carry_kernel<<<(unsigned) n_new, 128, 0, st>>>(s.ckv, cs, L.carry_kv, L.carry_score, p0, T, base, D);
            check_launch("carry_kernel");
        }
    }

    // the selection: an index source's own top-k over its key source's keys (or the forced rows); a Reuse reader's
    // from its topk_src
    const int32_t* ids = nullptr;
    if (l.index_source) tm.mark(AttnOp::Select);
    if (l.index_source && forced_rows) {
        ck(cudaMemcpyAsync(L.ids, forced_rows, (size_t) (T * d.top_k) * 4, cudaMemcpyDeviceToDevice, st), "the forced rows");
        L.ids_p0 = p0;
        L.ids_T = T;
        ids = L.ids;
    } else if (l.index_source) {
        const KvLayerBuffers& K = kv.layer(l.key_src);
        if (l.key_src != l.index && K.next_pos < p0 + T)
            failf("layer %d reads layer %d's index keys, which has reached position %lld, not %lld", l.index, l.key_src,
                  (long long) K.next_pos, (long long) (p0 + T));
        const int64_t n_k = (p0 + T) / r;
        if (n_k > 0) {
            const IndexerDims id{IH, d.top_k, r};
            const IndexerPlan plan = indexer_plan(T, n_k, s.idx_scratch_bytes);
            indexer_select(s.iq, s.iw, s.pos, T, K.keys, n_k, id, plan, s.idx_scratch, IndexerScorer::Auto, L.ids, st);
        } else {
            ck(cudaMemsetAsync(L.ids, 0xff, (size_t) (T * d.top_k) * 4, st), "cudaMemsetAsync(ids)");
        }
        L.ids_p0 = p0;
        L.ids_T = T;
        ids = L.ids;
    } else if (r && mode == ReaderMode::Reuse) {
        const KvLayerBuffers& S = kv.layer(l.topk_src);
        if (S.ids_p0 != p0 || S.ids_T != T)
            failf("reader %d (Reuse) needs layer %d's top-k of the chunk at %lld (%lld tokens); it holds %lld (%lld)", l.index,
                  l.topk_src, (long long) p0, (long long) T, (long long) S.ids_p0, (long long) S.ids_T);
        ids = S.ids;
    }

    // attention over the window and the KV source's rows (prefill_attn_kernel, the de-rope in its epilogue
    // when one split writes the output)
    tm.mark(AttnOp::Attend);
    {
        const KvLayerBuffers* V = r ? &kv.layer(l.kv_src) : nullptr;
        if (V && l.kv_src != l.index && V->next_pos < p0 + T)
            failf("layer %d reads layer %d's rows, which has reached position %lld, not %lld", l.index, l.kv_src,
                  (long long) V->next_pos, (long long) (p0 + T));
        AttnArgs a{};
        a.q = s.q;
        a.win = L.window;
        a.win_cells = L.window_cells;
        a.rows = V ? V->rows : nullptr;
        a.n_rows = V ? (p0 + T) / r : 0;
        a.ids = ids;
        a.top_k = d.top_k;
        a.pos = s.pos;
        a.n_swa = d.n_swa;
        a.ratio = r;
        a.n_head = H;
        a.scale_log2 = kLog2e / sqrtf((float) D);
        a.sinks = w.sinks;
        a.out = s.attn;
        a.part = s.part;
        const int64_t p_last = p0 + T - 1;
        const int64_t n_cells = std::min<int64_t>(p_last + 1, d.n_swa) +
                                (r ? (ids ? d.top_k : std::min<int64_t>(a.n_rows, (p_last + 1) / r)) : 0);
        const int n_hg = (H + HG - 1) / HG;
        // split the cells over blocks while the chunk alone leaves the device idle (decode)
        int64_t splits = 1;
        const int64_t base = T * n_hg, target = 2 * (int64_t) e.sm_count();
        if (base < target) {
            splits = std::min<int64_t>((n_cells + 63) / 64, target / base);
            splits = std::min<int64_t>(splits, (int64_t) (s.part_floats / ((size_t) T * H * (D + 2))));
            splits = std::max<int64_t>(1, std::min<int64_t>(splits, kMaxSplits));
        }
        a.cps = ((n_cells + splits - 1) / splits + CH - 1) / CH * CH;
        a.n_splits = (int) ((n_cells + a.cps - 1) / a.cps);
        if (a.n_splits < 1) a.n_splits = 1;
        s.n_splits = a.n_splits;
        a.derope = a.n_splits == 1 ? s.derope : nullptr;
        // one split: the f32 output only when the engine keeps it (the de-rope is what wo_a reads)
        if (a.n_splits == 1 && !e.attn_output()) a.out = nullptr;
        a.rope = ra;
        if (D == 256) launch_prefill_attn<256>(a, T, n_hg, st);
        else launch_prefill_attn<512>(a, T, n_hg, st);
        if (a.n_splits > 1) {
            merge_kernel<<<dim3((unsigned) T, (unsigned) H), 128, 0, st>>>(s.part, a.n_splits, H, D, w.sinks, s.attn);
            check_launch("merge_kernel");
            // output: de-rope into f16 (one split: the attention's epilogue wrote it)
            tm.mark(AttnOp::Derope);
            rope(s.attn, HD, nullptr, s.derope, HD, H, D, s.pos, 0, 0, ra, -1.f, T, st);
        }
    }

    // wo_a per group (qmatmul's groups, or one strided batched GEMM), wo_b
    {
        tm.mark(AttnOp::WoA);
        ProjInput pdr{nullptr, s.derope, HD, T, HD, nullptr, false, s.drq};
        project(e, pdr, w.wo_a, s.oa, OO, 1.f, d.o_groups, "attn_output_a");
        tm.mark(AttnOp::WoB);
        ProjInput poa{s.oa, nullptr, OO, T, OO, s.oah, false, s.oaq};
        project(e, poa, w.wo_b, out, E, 1.f, 1, "attn_output_b");
    }
    tm.end();
    L.next_pos = p0 + T;

    read_times(e, tm);
}

void attn_decode_advance(const AttnLayer& l, KvCache& kv, int64_t p, ReaderMode mode, bool forced_rows) {
    KvLayerBuffers& L = kv.layer(l.index);
    const int r = l.ratio;
    if (L.next_pos != p) failf("layer %d: a decode token at position %lld, the layer is at %lld", l.index, (long long) p, (long long) L.next_pos);
    if (p + 1 > kv.config().max_ctx) failf("layer %d: position %lld, the cache holds %lld", l.index, (long long) p, (long long) kv.config().max_ctx);
    if (l.index_source && !forced_rows && l.key_src != l.index && kv.layer(l.key_src).next_pos < p + 1)
        failf("layer %d reads layer %d's index keys, which has reached position %lld, not %lld", l.index, l.key_src,
              (long long) kv.layer(l.key_src).next_pos, (long long) (p + 1));
    if (r && !l.index_source && mode == ReaderMode::Reuse) {
        const KvLayerBuffers& S = kv.layer(l.topk_src);
        if (S.ids_p0 != p || S.ids_T != 1)
            failf("reader %d (Reuse) needs layer %d's top-k of the token at %lld; it holds %lld (%lld tokens)", l.index,
                  l.topk_src, (long long) p, (long long) S.ids_p0, (long long) S.ids_T);
    }
    if (r && l.kv_src != l.index && kv.layer(l.kv_src).next_pos < p + 1)
        failf("layer %d reads layer %d's rows, which has reached position %lld, not %lld", l.index, l.kv_src,
              (long long) kv.layer(l.kv_src).next_pos, (long long) (p + 1));
    if (l.kv_source && (p + 1) / r > L.rows_cap)
        failf("layer %d: %lld rows, the cache holds %lld", l.index, (long long) ((p + 1) / r), (long long) L.rows_cap);
    L.next_pos = p + 1;
    if (l.index_source) {
        L.ids_p0 = p;
        L.ids_T = 1;
    }
}

void attn_decode(AttnEngine& e, const AttnLayer& l, const AttnLayerWeights& w, KvCache& kv, const float* x, int64_t p,
                 float* out, ReaderMode mode, const int32_t* forced_rows, bool device_pos, const void* xq) {
    const AttnDims& d = e.dims();
    AttnScratch& s = e.scratch_mut();
    cudaStream_t st = (cudaStream_t) e.stream();
    attn_check_weights(d, l, w);
    attn_decode_advance(l, kv, p, mode, forced_rows != nullptr);
    if (attn_decode_fused(e, l, w, kv, x, xq, p, out, mode, forced_rows, device_pos)) return;
    // other weights (the f16 test path) or matrices that do not stage: attn_prefill's launches at T = 1
    KvLayerBuffers& L = kv.layer(l.index);
    const int D = d.head_dim, H = d.n_head, r = l.ratio, ID = d.idx_dim, IH = d.idx_heads;
    const int64_t E = d.n_embd, HD = (int64_t) H * D, OO = (int64_t) d.o_groups * d.o_lora;
    const RopeArgs ra = attn_rope_args(d, r != 0), rc = attn_rope_args(d, true);
    Timer tm{e, st};
    tm.mark(AttnOp::QA);
    s.T = 1;
    s.B = r && (p + 1) % r == 0 ? 1 : 0;
    s.first_block = r ? p / r : 0;
    if (!device_pos) {
        fill_pos_kernel<<<1, 32, 0, st>>>(s.pos, p, 1);
        check_launch("fill_pos_kernel");
    }
    ProjInput px{x, nullptr, E, 1, E, s.xh, false, s.xq};
    ProjInput pqr{s.qr, nullptr, d.q_lora, 1, d.q_lora, s.qrh, true, s.qrq};

    // query, window and the index source's query side: attn_prefill's, every rope at s.pos
    project(e, px, w.wq_a, s.qr, d.q_lora, 1.f, 1, "attn_q_a");
    rms_norm(s.qr, d.q_lora, w.q_a_norm, d.q_lora, d.rms_eps, s.qr, d.q_lora, s.qrh, d.q_lora, 1, st);
    project(e, pqr, w.wq_b, s.q, HD, 1.f, 1, "attn_q_b");
    rope(s.q, HD, s.q, nullptr, HD, H, D, s.pos, 0, 0, ra, 1.f, 1, st);
    project(e, px, w.wkv, s.kv, D, 1.f, 1, "attn_kv");
    rms_norm(s.kv, D, w.kv_norm, D, d.rms_eps, s.kv, D, nullptr, 0, 1, st);
    rope(s.kv, D, s.kv, nullptr, D, 1, D, s.pos, 0, 0, ra, 1.f, 1, st);
    window_store_kernel<<<1, 128, 0, st>>>(s.kv, 1, s.pos, L.window, L.window_cells, D);
    check_launch("window_store_kernel");
    if (l.index_source) {
        project(e, pqr, w.idx_q_b, s.iq, (int64_t) IH * ID, 1.f, 1, "indexer.attn_q_b");
        rope(s.iq, (int64_t) IH * ID, s.iq, nullptr, (int64_t) IH * ID, IH, ID, s.pos, 0, 0, rc, 1.f, 1, st);
        fwht(s.iq, IH, st);
        project(e, px, w.idx_proj, s.iw, IH, 1.0f / sqrtf((float) (ID * IH)), 1, "indexer.proj");
    }
    tm.mark(AttnOp::CompProj);

    // compressor: the token's block decided on the device. The norm, the rope and the key projection always run on
    // the one row (on the last pooled row when the block is not complete); only a complete block is stored
    if (l.kv_source) {
        project(e, px, w.comp_kv, s.ckv, D, 1.f, 1, "attn_compressor_kv");
        const float* cs = s.ckv;
        if (l.gate) {
            project(e, px, w.comp_gate, s.cscore, D, 1.f, 1, "attn_compressor_gate");
            cs = s.cscore;
        }
        compress_decode_kernel<<<1, 128, 0, st>>>(s.ckv, cs, L.carry_kv, L.carry_score, s.pos, r, D, s.pooled, s.blk_pos);
        check_launch("compress_decode_kernel");
        rms_norm(s.pooled, D, w.comp_norm, D, d.rms_eps, s.latent, D, s.latenth, D, 1, st);
        rope(s.latent, D, s.rowsf, nullptr, D, 1, D, s.blk_pos, 0, 0, rc, 1.f, 1, st);
        store_block_kernel<<<1, 128, 0, st>>>(s.rowsf, L.rows, s.pos, r, D);
        check_launch("store_block_kernel");
        if (l.key_owner) {
            ProjInput plat{s.latent, nullptr, D, 1, D, s.latenth, true, s.latq};
            project(e, plat, w.idx_k, s.idx_kf, ID, 1.f, 1, "indexer.attn_k");
            rms_norm(s.idx_kf, ID, w.idx_k_norm, ID, d.rms_eps, s.idx_kf, ID, nullptr, 0, 1, st);
            rope(s.idx_kf, ID, s.idx_kf, nullptr, ID, 1, ID, s.blk_pos, 0, 0, rc, 1.f, 1, st);
            fwht(s.idx_kf, 1, st);
            store_block_kernel<<<1, 128, 0, st>>>(s.idx_kf, L.keys, s.pos, r, ID);
            check_launch("store_block_kernel");
        }
    }
    tm.mark(AttnOp::Select);

    // the selection: over the key source's capacity, each kernel bounded by the token's visible rows (indexer.hpp
    // reads them from pos), so the launches do not move with the position
    const int32_t* ids = nullptr;
    if (l.index_source && forced_rows) {
        ck(cudaMemcpyAsync(L.ids, forced_rows, (size_t) d.top_k * 4, cudaMemcpyDeviceToDevice, st), "the forced rows");
        ids = L.ids;
    } else if (l.index_source) {
        const KvLayerBuffers& K = kv.layer(l.key_src);
        const IndexerDims id{IH, d.top_k, r};
        const IndexerPlan plan = indexer_plan(1, K.rows_cap, s.idx_scratch_bytes);
        indexer_select(s.iq, s.iw, s.pos, 1, K.keys, K.rows_cap, id, plan, s.idx_scratch, IndexerScorer::Auto, L.ids, st);
        ids = L.ids;
    } else if (r && mode == ReaderMode::Reuse) {
        ids = kv.layer(l.topk_src).ids;
    }
    tm.mark(AttnOp::Attend);

    // attention: the KV source's rows up to its capacity (the kernel bounds them by the position), a split count
    // from the layer's largest cell count, the cells per split on the device
    {
        const KvLayerBuffers* V = r ? &kv.layer(l.kv_src) : nullptr;
        AttnArgs a{};
        a.q = s.q;
        a.win = L.window;
        a.win_cells = L.window_cells;
        a.rows = V ? V->rows : nullptr;
        a.n_rows = V ? V->rows_cap : 0;
        a.ids = ids;
        a.top_k = d.top_k;
        a.pos = s.pos;
        a.n_swa = d.n_swa;
        a.ratio = r;
        a.n_head = H;
        a.scale_log2 = kLog2e / sqrtf((float) D);
        a.sinks = w.sinks;
        a.out = s.attn;
        a.part = s.part;
        a.cps = 0;
        a.n_splits = decode_splits(e, kv, l, ids != nullptr);
        s.n_splits = a.n_splits;
        const int n_hg = (H + HG - 1) / HG;
        if (D == 256) launch_attn<256>(a, 1, n_hg, st);
        else launch_attn<512>(a, 1, n_hg, st);
        if (a.n_splits > 1) {
            merge_kernel<<<dim3(1, (unsigned) H), 128, 0, st>>>(s.part, a.n_splits, H, D, w.sinks, s.attn);
            check_launch("merge_kernel");
        }
    }
    tm.mark(AttnOp::WoA);

    // output: attn_prefill's
    rope(s.attn, HD, nullptr, s.derope, HD, H, D, s.pos, 0, 0, ra, -1.f, 1, st);
    {
        ProjInput pdr{nullptr, s.derope, HD, 1, HD, nullptr, false, s.drq};
        project(e, pdr, w.wo_a, s.oa, OO, 1.f, d.o_groups, "attn_output_a");
        ProjInput poa{s.oa, nullptr, OO, 1, OO, s.oah, false, s.oaq};
        project(e, poa, w.wo_b, out, E, 1.f, 1, "attn_output_b");
    }
    tm.end();
    read_times(e, tm);
}

void attn_decode_batch_advance(const AttnLayer& l, KvCache* const* kv, int B, const int64_t* p, ReaderMode mode) {
    for (int i = 0; i < B; ++i) attn_decode_advance(l, *kv[i], p[i], mode, false);
}

void attn_decode_batch(AttnEngine& e, const AttnLayer& l, const AttnLayerWeights& w, KvCache* const* kv, int B,
                       const float* x, const void* xq, const int32_t* pos, const int64_t* p, float* out, ReaderMode mode) {
    const AttnDims& d = e.dims();
    attn_check_weights(d, l, w);
    if (B < 1 || B > e.max_batch() || B > kMmvqMaxCols)
        failf("layer %d: a batch of %d rows (1..%d, AttnEngine::init_batch)", l.index, B, std::min<int>(e.max_batch(), (int) kMmvqMaxCols));
    if (!dec_plan(d, l, w).r_x)
        failf("layer %d: the batched decode takes the fused token's weights (q2_K, 16-byte aligned rows)", l.index);
    if (mode != ReaderMode::Reuse)
        failf("layer %d: the batched decode reads with Reuse Mode (--readers dense: the split count follows each cache's "
              "capacity)", l.index);
    if (!xq) failf("layer %d: the batched decode takes x's q8_1 rows from the caller", l.index);
    attn_decode_batch_advance(l, kv, B, p, mode);
    AttnScratch& s = e.scratch_mut();
    cudaStream_t st = (cudaStream_t) e.stream();
    const int D = d.head_dim, H = d.n_head, r = l.ratio, ID = d.idx_dim, IH = d.idx_heads;
    const int64_t E = d.n_embd, HD = (int64_t) H * D, OO = (int64_t) d.o_groups * d.o_lora, IQ = (int64_t) IH * ID;
    const RopeArgs ra = attn_rope_args(d, r != 0), rc = attn_rope_args(d, true);
    s.T = B;
    s.B = 0;
    auto proj = [&](const QWeightView& wv, const void* a, int64_t k, float* y, int64_t ldy, int groups, const char* what) {
        qmatmul_q(e.qmm(), wv, QActs{QActLayout::Q8_1, B, k, const_cast<void*>(a)}, y, ldy, groups, st, what);
    };
    // the x projections, every row at once (the fused token's x launch: attn_q_a, attn_kv, the compressor's kv and
    // gate, indexer.proj with its scale after the sum, as the plain epilogue)
    proj(w.wq_a, xq, E, s.qr, d.q_lora, 1, "attn_q_a");
    proj(w.wkv, xq, E, s.kv, D, 1, "attn_kv");
    if (l.kv_source) proj(w.comp_kv, xq, E, s.ckv, D, 1, "attn_compressor_kv");
    if (l.gate) proj(w.comp_gate, xq, E, s.cscore, D, 1, "attn_compressor_gate");
    if (l.index_source) {
        proj(w.idx_proj, xq, E, s.iw, IH, 1, "indexer.proj");
        scale_kernel<<<(unsigned) B, 128, 0, st>>>(s.iw, IH, IH, 1.0f / sqrtf((float) (ID * IH)));
        check_launch("scale_kernel");
    }
    // per row: the qr norm with its f16 and q8_1, the kv norm, rope and window cell (decode_qkv_kernel on the row's
    // buffers and its cache's ring)
    auto* qrq = static_cast<mmvq::Q81Block*>(s.qrq);
    for (int i = 0; i < B; ++i) {
        KvLayerBuffers& L = kv[i]->layer(l.index);
        decode_qkv_kernel<<<2, NORM_T, 0, st>>>(s.qr + (int64_t) i * d.q_lora, s.qrh + (int64_t) i * d.q_lora, w.q_a_norm,
                                                d.q_lora, qrq + (int64_t) i * (d.q_lora / mmvq::Q8K), s.kv + (int64_t) i * D,
                                                w.kv_norm, D, pos + i, ra, L.window, L.window_cells, d.rms_eps);
        check_launch("decode_qkv_kernel");
    }
    // the qr projections, every row at once, then their ropes at each row's position (DecEpi's)
    proj(w.wq_b, s.qrq, d.q_lora, s.q, HD, 1, "attn_q_b");
    if (l.index_source) proj(w.idx_q_b, s.qrq, d.q_lora, s.iq, IQ, 1, "indexer.attn_q_b");
    {
        const int64_t pairs = HD / 2 + (l.index_source ? IQ / 2 : 0);
        const dim3 grid((unsigned) ((pairs + 255) / 256), (unsigned) B);
        batch_qrope_kernel<<<grid, 256, 0, st>>>(s.q, s.qhb, HD, D, l.index_source ? s.iq : nullptr, IQ, ID, pos, ra, rc);
        check_launch("batch_qrope_kernel");
    }
    auto* drq = static_cast<mmvq::Q81Block*>(s.drq);
    for (int i = 0; i < B; ++i) {
        KvCache& K = *kv[i];
        KvLayerBuffers& L = K.layer(l.index);
        const int32_t* pi = pos + i;
        // the compressor's block and the key owner's index key, the index queries' FWHT (decode_book_kernel)
        if (l.kv_source || l.index_source) {
            BookArgs b{};
            b.compress = l.kv_source;
            if (l.kv_source) {
                b.ckv = s.ckv + (int64_t) i * D;
                b.cs = l.gate ? s.cscore + (int64_t) i * D : b.ckv;
                b.carry_kv = L.carry_kv;
                b.carry_score = L.carry_score;
                b.r = r;
                b.D = D;
                b.pooled = s.pooled + (int64_t) i * D;
                b.latent = s.latent + (int64_t) i * D;
                b.latenth = s.latenth + (int64_t) i * D;
                b.rowsf = s.rowsf + (int64_t) i * D;
                b.comp_norm = w.comp_norm;
                b.rows = L.rows;
                b.blk_pos = s.blk_posb + i;
                if (l.key_owner) {
                    b.idx_k = static_cast<const uint8_t*>(w.idx_k.data);
                    b.idx_k_norm = w.idx_k_norm;
                    b.idx_kf = s.idx_kf + (int64_t) i * ID;
                    b.keys = L.keys;
                }
                b.ID = ID;
            }
            if (l.index_source) {
                b.iq = s.iq + (int64_t) i * IQ;
                b.iq_rows = IH;
            }
            b.pos = pi;
            b.rc = rc;
            b.eps = d.rms_eps;
            decode_book_kernel<<<(l.kv_source ? 1 : 0) + (l.index_source ? 1 : 0), NORM_T, book_smem(d), st>>>(b);
            check_launch("decode_book_kernel");
        }
        // the selection over the row's own keys
        const int32_t* ids = nullptr;
        if (l.index_source) {
            const KvLayerBuffers& Kk = K.layer(l.key_src);
            const IndexerDims id{IH, d.top_k, r};
            const IndexerPlan ip = indexer_plan(1, Kk.rows_cap, s.idx_scratch_bytes);
            indexer_select(s.iq + (int64_t) i * IQ, s.iw + (int64_t) i * IH, pi, 1, Kk.keys, Kk.rows_cap, id, ip,
                           s.idx_scratch, IndexerScorer::Auto, L.ids, st);
            ids = L.ids;
        } else if (r) {
            ids = K.layer(l.topk_src).ids;
        }
        // the attention over the row's window and rows, its merge with the sink, the de-rope and its q8_1
        const KvLayerBuffers* V = r ? &K.layer(l.kv_src) : nullptr;
        AttnArgs a{};
        a.qh = s.qhb + (int64_t) i * HD;
        a.win = L.window;
        a.win_cells = L.window_cells;
        a.rows = V ? V->rows : nullptr;
        a.n_rows = V ? V->rows_cap : 0;
        a.ids = ids;
        a.top_k = d.top_k;
        a.pos = pi;
        a.n_swa = d.n_swa;
        a.ratio = r;
        a.n_head = H;
        a.scale_log2 = kLog2e / sqrtf((float) D);
        a.sinks = w.sinks;
        a.part = s.part;
        a.cps = 0;
        a.n_splits = decode_splits(e, K, l, ids != nullptr);
        s.n_splits = a.n_splits;
        const int n_hg = (H + HG - 1) / HG;
        if (D == 256) launch_attn<256, true>(a, 1, n_hg, st);
        else launch_attn<512, true>(a, 1, n_hg, st);
        float* attn_i = s.attn + (int64_t) i * HD;
        uint16_t* derope_i = s.derope + (int64_t) i * HD;
        mmvq::Q81Block* drq_i = drq + (int64_t) i * (HD / mmvq::Q8K);
        if (D == 256) merge_derope_kernel<256><<<H, 256, 0, st>>>(s.part, a.n_splits, H, w.sinks, pi, ra, attn_i, derope_i, drq_i);
        else merge_derope_kernel<512><<<H, 512, 0, st>>>(s.part, a.n_splits, H, w.sinks, pi, ra, attn_i, derope_i, drq_i);
        check_launch("merge_derope_kernel");
    }
    // the output, every row at once: attn_output_a (grouped), oa's q8_1, attn_output_b
    proj(w.wo_a, s.drq, HD, s.oa, OO, d.o_groups, "attn_output_a");
    QActs oq;
    qact_quantize(s.oa, nullptr, OO, B, OO, QActLayout::Q8_1, s.oaq, st, oq);
    qmatmul_q(e.qmm(), w.wo_b, oq, out, E, 1, st, "attn_output_b");
}

}  // namespace strata::dsv41::engine
