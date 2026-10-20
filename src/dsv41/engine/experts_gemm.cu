// src/dsv41/engine/experts_gemm.cu - the routed experts' grouped prefill GEMM of experts_gemm.hpp.
//
// Why (the A100 run): a 16384-token chunk spent 588-673 ms per layer in the
// routed experts, about 11 TFLOP/s. The plan's slab passes cut the slab's pairs every 4096 in pair order, so every one
// of the 168 slab experts came back in each of 13 passes with ~23 rows: 2184 expert calls per layer of three qmatmuls
// each, every one dequantizing its whole matrix into the f16 scratch (39-52 us at any row count, qmatmul_bench) for
// cuBLAS at 16-row shapes (6 TFLOP/s): ~435 ms, and the 216 ring experts' ~256-row calls ~58 ms more. Now the slab's
// pairs are regrouped into whole experts (experts.cpp) and each pass is two launches over all of its experts.
//
// Strata first: the structure of Strata's src/prefill/moe_fused.cu (#136: the Q2_0 pack's prompt experts on its own
// tensor-core kernels): the routed rows grouped by expert into tiles, one block per (tile, slice of weight rows), the
// weights read as stored and decoded inside the kernel, a cp.async pipeline, gate and up with the activation in one
// kernel and down in a second, which reads its h. What changes, and why moe_fused's int8 m16n8k32 does not carry over:
// a q2_K or q3_K block scales every 16 values (and q2_K subtracts a min per 16), where Strata's Q2_0 has one scale per
// 64 values, so an int8 MMA over 32 values would need its int32 result rescaled in f32 per 16 values and per row, the
// ALU work that keeps the fork's q2_K MMQ at 45-52 TFLOP/s on the A100 against cuBLAS f16's 197-210 at 4096 rows
// (qmatmul_bench, A100 pod). Here the decoder writes each weight once per tile as the dequant path's f16 value
// (qdequant.cuh) into shared memory and the tensor cores run f16 x f16 -> f32 (mma.sync m16n8k16): the q2_K / q3_K
// bytes are read, never an f16 copy of the matrix (the dequant path wrote and read about 170 GB of them per layer), and the
// numerics are the engine's dequant path, whose reference qref_matmul_f16 already is.
//
// The kernel: a block is 8 warps and computes kExpertTileRows (128) routed rows against 128 weight rows (gate/up: 64
// features with their gate and up rows, so the epilogue has both; down: 128 output rows), over k in steps of 32: the
// step's activations by cp.async into the other of two shared buffers while the tensor cores take this one, and the
// next step's weights decoded by the block's threads (each 2 x 8 values) into the shared weight tile once every warp
// has read it. Each warp holds 32 rows x 64 weight rows of f32 accumulators. The grid's x is the weight slice, so the
// blocks of one tile run together and its activations stay in L2; the tiles of one expert follow each other, so its
// weights do too. Rows past a tile's last are loaded as a copy of its last row and not stored; a warp's 16-row group
// with none of the tile's rows skips its MMAs.
//
// (down 67.5 TFLOP/s against gate/up's 102.6 in the A100-SXM4 smoke).
// The SASS of the q3_K down kernel (sm_80, CUDA 12.8, dev VM) had 22 loads per thread per k step, 18 of them
// 2-byte (a q3_K block is 110 bytes, so its rows sit at 2-byte alignment and decode8's reads can be no wider), each
// warp's load touching 8 rows' lines: about 1,700 L1 wavefronts per block and step against 512 cycles of the tensor
// cores' work, where q2_K gate/up issued 8 loads of 4 bytes. Now a k-block (256 values, 8 steps) of the block's 128
// weight rows comes in once per block by cp.async: each row's bytes from its block's address rounded down to 16, in
// 16-byte pieces on consecutive threads (one 128-byte line per 8 threads), into a shared row of 144 bytes (the
// rounding's offset, at most 14, plus 110: 8 pieces; 144 bytes apart, the rows' 2-byte reads of a step meet in about
// 1.7 wavefronts, a simulation of the real dims' offsets on the dev VM); the decoder reads fetch8's bytes from there.
// The same values: fetch8 and decode8_f16x2 on the same bytes. Shared memory: 20 KiB of activations (two steps), 10
// KiB of weights (one step: two barriers per step, so 48 KiB in all and two blocks per SM on the 100 KiB SMs of
// sm_89 and sm_120 too) and 18 KiB of raw k-block.
#include "strata/dsv41/engine/experts_gemm.hpp"

#include "ffn_device.cuh"
#include "qdequant.cuh"

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "experts_gemm.cu: cp.async, ldmatrix and mma.sync m16n8k16 need sm_80 or newer (the flavors build 80, 89, 120)"
#endif

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 experts gemm: %s\n", msg);
    std::exit(1);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) failf("%s: %s", what, cudaGetErrorString(e));
}

constexpr int BM = kExpertTileRows;   // routed rows per block
constexpr int BN = 128;               // weight rows per block
constexpr int BK = 32;                // k per pipeline step
constexpr int LDS = BK + 8;           // a shared row of 80 bytes: the 8 rows of an ldmatrix hit distinct banks
constexpr int THREADS = 256;
constexpr int RAW_PIECES = 8;         // 16-byte pieces of a staged row: a q3_K block (110 bytes) from any 2-byte offset
constexpr int RAW_S = 144;            // bytes per staged row: 16 more than the pieces, so 8 rows' reads spread over banks
static_assert(BN == kExpertDownCols && BN == 2 * kExpertGateUpCols, "a block's weight rows");

bool quant_type(int t) { return t == kQTypeQ2K || t == kQTypeQ3K; }

// two f32 to f16 bits (round to nearest even, the IEEE conversion f16_bits.hpp writes out), lo in the low half
__device__ __forceinline__ uint32_t f16x2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ uint32_t smem(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}
// d += A (16 x 16 f16, row) B (16 x 8 f16, col), f32
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One block: tile blockIdx.y against weight slice blockIdx.x. GU: rows 0..63 of the shared weights are the gate rows
// of features 64 x .. 64 x + 63, rows 64..127 their up rows, and h gets the clamp of each pair; else rows 0..127 are
// output rows 128 x .. of part `part`, y gets the sums.
template <int TYPE, bool GU>
__global__ void __launch_bounds__(THREADS, 2)
    grouped_kernel(const ExpertTile* __restrict__ tiles, int part, const uint16_t* __restrict__ x, int K, int N,
                   float limit, float* __restrict__ y, int64_t ldy, uint16_t* __restrict__ h) {
    static_assert(TYPE == kQTypeQ2K || TYPE == kQTypeQ3K, "grouped_kernel: q2_K and q3_K");
    __shared__ __align__(16) uint16_t xs[2][BM][LDS];
    __shared__ __align__(16) uint16_t ws[BN][LDS];
    __shared__ __align__(16) uint8_t raw[BN][RAW_S];
    constexpr int BB = qdequant::block_bytes<TYPE>();
    static_assert(15 + BB <= 16 * RAW_PIECES && 16 * RAW_PIECES <= RAW_S, "a staged row holds a block from any 2-byte offset");
    const ExpertTile tile = tiles[blockIdx.y];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;   // the warp's 32 rows and its half of the weight rows
    const int64_t row_bytes = (int64_t) (K / 256) * BB;

    // weight row R (0..127) of the block: GU rows 0..63 are the gate rows of features 64 x .. 64 x + 63, rows 64..127
    // their up rows; else output rows 128 x + R of part `part`
    const uint8_t* wbase[2];
    if (GU) {
        wbase[0] = static_cast<const uint8_t*>(tile.w[1]) + (int64_t) blockIdx.x * (BN / 2) * row_bytes;   // gate
        wbase[1] = static_cast<const uint8_t*>(tile.w[0]) + (int64_t) blockIdx.x * (BN / 2) * row_bytes;   // up
    } else {
        // read from the array: an index into the copy in registers would put the copy in local memory
        wbase[0] = static_cast<const uint8_t*>(tiles[blockIdx.y].w[part]) + (int64_t) blockIdx.x * BN * row_bytes;
        wbase[1] = wbase[0] + 64 * row_bytes;
    }
    // the part's blocks are read by 16-byte pieces from 16-byte boundaries: the part must start on one (the slab,
    // ring and prefetch slots are 256-byte aligned) and, a multiple of 16 bytes long, no piece leaves it
    if ((reinterpret_cast<uintptr_t>(wbase[0]) | reinterpret_cast<uintptr_t>(wbase[1])) & 15) __trap();

    // this thread's share of a step: shared rows r and r + 64 at values 8 g .. 8 g + 7, weights and activations alike
    const int r = tid >> 2, g = tid & 3;
    const uint16_t* xrow[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) xrow[i] = x + (int64_t) (tile.row0 + min(r + 64 * i, tile.rows - 1)) * K + 8 * g;

    auto load_x = [&](int s, int kt) {
#pragma unroll
        for (int i = 0; i < 2; ++i) cp16(&xs[s][r + 64 * i][8 * g], xrow[i] + kt * BK);
    };
    // k-block kb of the 128 weight rows into raw: row R's pieces from its block's address rounded down to 16, the
    // pieces that start before the block's end (8 threads per row, 4 pieces per thread)
    auto stage_raw = [&](int kb) {
#pragma unroll
        for (int u = 0; u < BN * RAW_PIECES / THREADS; ++u) {
            const int slot = u * THREADS + tid, R = slot / RAW_PIECES, j = slot % RAW_PIECES;
            const uint8_t* A = (R < 64 ? wbase[0] : wbase[1]) + (int64_t) (R & 63) * row_bytes + (int64_t) kb * BB;
            const uint8_t* A0 = reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(A) & ~(uintptr_t) 15);
            if (A0 + 16 * j < A + BB) cp16(&raw[R][16 * j], A0 + 16 * j);
        }
    };
    // step kt's weights from the staged k-block into ws: the dequant path's f16 values (qdequant.cuh)
    auto store_w = [&](int kt) {
        const int kk = kt * BK + 8 * g;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int off = (int) ((reinterpret_cast<uintptr_t>(wbase[i]) + (uintptr_t) (r * row_bytes + (kk >> 8) * BB)) & 15);
            uint32_t hv[4];
            qdequant::decode8_f16x2<TYPE>(qdequant::fetch8<TYPE>(&raw[r + 64 * i][off], kk & 255), kk & 255, hv);
            *reinterpret_cast<uint4*>(&ws[r + 64 * i][8 * g]) = make_uint4(hv[0], hv[1], hv[2], hv[3]);
        }
    };
    // the shared row of the warp's n8 tile j (GU: tiles 0..3 gate, 4..7 the same features' up)
    auto wrow = [&](int j) { return GU ? (j < 4 ? 32 * wn + 8 * j : 64 + 32 * wn + 8 * (j - 4)) : 64 * wn + 8 * j; };

    float acc[2][8][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mi][j][c] = 0.f;

    // the warp's 16-row groups that hold rows of the tile (warp-uniform): the others skip their MMAs, so a partial
    // tile (a ring expert's ~45 rows under the real routing, A100 pod) costs the tensor cores its rows only
    const int n_mi = min(2, max(0, (tile.rows - 32 * wm + 15) / 16));
    const int nk = K / BK;
    constexpr int STEPS = 256 / BK;   // steps per k-block
    stage_raw(0);
    load_x(0, 0);
    cp_commit();
    cp_wait_all();
    __syncthreads();
    store_w(0);
    __syncthreads();
    for (int kt = 0; kt < nk; ++kt) {
        const int s = kt & 1;
        // the next k-block replaces this one once its last step is decoded (at the end of the step before this one)
        const bool next_block = (kt + 1) % STEPS == 0 && kt + 1 < nk;
        if (kt + 1 < nk) load_x(s ^ 1, kt + 1);
        if (next_block) stage_raw((kt + 1) / STEPS);
        cp_commit();
#pragma unroll
        for (int ks = 0; ks < BK / 16 && n_mi > 0; ++ks) {
            uint32_t a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                if (mi < n_mi) ldmatrix_x4(a[mi], &xs[s][32 * wm + 16 * mi + (lane & 15)][16 * ks + 8 * (lane >> 4)]);
            // one pair of n8 tiles' weights at a time: 4 registers live instead of all 16
#pragma unroll
            for (int jp = 0; jp < 4; ++jp) {
                uint32_t q[4];
                ldmatrix_x4(q, &ws[wrow(2 * jp) + (lane & 7) + 8 * (lane >> 4)][16 * ks + 8 * ((lane >> 3) & 1)]);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    if (mi >= n_mi) continue;
                    mma_f16(acc[mi][2 * jp], a[mi], q[0], q[1]);
                    mma_f16(acc[mi][2 * jp + 1], a[mi], q[2], q[3]);
                }
            }
        }
        if (kt + 1 < nk) {
            if (next_block) cp_wait_all();   // the new k-block (and the next activations) landed
            __syncthreads();                 // every warp has read ws; the k-block is everyone's
            store_w(kt + 1);
            cp_wait_all();
            __syncthreads();                 // ws and the next activations are ready
        }
    }

    // accumulator c of an m16n8 tile: row lane / 4 (+ 8 for c >= 2), columns 2 (lane % 4) and the next
    const int lc = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int lr = 32 * wm + 16 * mi + 8 * hh + (lane >> 2);
            if (lr >= tile.rows) continue;
            const int64_t row = tile.row0 + lr;
            if (GU) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int f = blockIdx.x * (BN / 2) + 32 * wn + 8 * j + lc;
                    const float* gt = &acc[mi][j][2 * hh];
                    const float* up = &acc[mi][j + 4][2 * hh];
                    *reinterpret_cast<uint32_t*>(h + row * N + f) =
                        f16x2(swiglu_clamp(gt[0], up[0], limit), swiglu_clamp(gt[1], up[1], limit));
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int n = blockIdx.x * BN + 64 * wn + 8 * j + lc;
                    *reinterpret_cast<float2*>(y + row * ldy + n) = make_float2(acc[mi][j][2 * hh], acc[mi][j][2 * hh + 1]);
                }
            }
        }
}

__global__ void gather_f16_kernel(const float* __restrict__ x, const int32_t* __restrict__ pairs, int n_used, int E,
                                  uint16_t* __restrict__ xg) {
    const int64_t i = blockIdx.y;
    const float2* src = reinterpret_cast<const float2*>(x + (int64_t) (pairs[i] / n_used) * E);
    uint32_t* dst = reinterpret_cast<uint32_t*>(xg + i * E);
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < E / 2; e += gridDim.x * blockDim.x) {
        const float2 v = src[e];
        dst[e] = f16x2(v.x, v.y);
    }
}

template <bool GU>
void launch(int type, const ExpertTile* tiles, int64_t n_tiles, int part, const uint16_t* x, int64_t K, int64_t N,
            float limit, float* y, int64_t ldy, uint16_t* h, cudaStream_t s) {
    if (n_tiles < 1) return;
    const int cols = GU ? kExpertGateUpCols : kExpertDownCols;
    if (n_tiles > 65535 || K % 256 || N % cols || K > (1 << 20) || N > (1 << 20))
        failf("%lld tiles (at most 65535) of k %lld (whole 256-value blocks) against %lld columns (a multiple of %d)",
              (long long) n_tiles, (long long) K, (long long) N, cols);
    if (reinterpret_cast<uintptr_t>(x) % 16) failf("activations at %p (cp.async reads 16 bytes)", (const void*) x);
    const dim3 grid((unsigned) (N / cols), (unsigned) n_tiles);
    switch (type) {
        case kQTypeQ2K: grouped_kernel<kQTypeQ2K, GU><<<grid, THREADS, 0, s>>>(tiles, part, x, (int) K, (int) N, limit, y, ldy, h); break;
        case kQTypeQ3K: grouped_kernel<kQTypeQ3K, GU><<<grid, THREADS, 0, s>>>(tiles, part, x, (int) K, (int) N, limit, y, ldy, h); break;
        default: failf("weight type %d (q2_K or q3_K)", type);
    }
    ck(cudaGetLastError(), GU ? "grouped_kernel (gate, up)" : "grouped_kernel");
}

}  // namespace

void expert_gemm_check(int type_up, int type_gate, int type_down, int64_t n_embd, int64_t n_ff) {
    if (!quant_type(type_up) || !quant_type(type_gate) || !quant_type(type_down) || type_up != type_gate)
        failf("expert types up %s, gate %s, down %s: the grouped GEMM takes q2_K or q3_K, gate and up alike",
              qtype_name(type_up), qtype_name(type_gate), qtype_name(type_down));
    if (n_embd % 256 || n_ff % 256 || n_ff % kExpertGateUpCols || n_embd % kExpertDownCols)
        failf("experts of [%lld, %lld]: n_embd a multiple of 256 and %d, n_ff of 256 and %d", (long long) n_embd,
              (long long) n_ff, kExpertDownCols, kExpertGateUpCols);
}

void expert_gather_rows_f16(const float* x, const int32_t* pairs, int n_used, int64_t n, int E, uint16_t* xg,
                            void* stream) {
    if (n < 1) return;
    if (n > 65535 || E % 2) failf("a gather of %lld rows of %d values (at most 65535 rows, an even width)", (long long) n, E);
    const dim3 grid((unsigned) ((E / 2 + 255) / 256), (unsigned) n);
    gather_f16_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(x, pairs, n_used, E, xg);
    ck(cudaGetLastError(), "gather_f16_kernel");
}

void expert_gate_up(const ExpertTile* tiles, int64_t n_tiles, int type, const uint16_t* xg, int64_t E, int64_t F,
                    float limit, uint16_t* h, void* stream) {
    launch<true>(type, tiles, n_tiles, 0, xg, E, F, limit, nullptr, 0, h, (cudaStream_t) stream);
}

void expert_gemm(const ExpertTile* tiles, int64_t n_tiles, int part, int type, const uint16_t* x, int64_t K, int64_t N,
                 float* y, int64_t ldy, void* stream) {
    if (part < 0 || part > 2 || ldy < N || ldy % 2) failf("part %d, output stride %lld for %lld columns", part, (long long) ldy, (long long) N);
    launch<false>(type, tiles, n_tiles, part, x, K, N, 0.f, y, ldy, nullptr, (cudaStream_t) stream);
}

}  // namespace strata::dsv41::engine
