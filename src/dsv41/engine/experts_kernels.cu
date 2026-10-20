// src/dsv41/engine/experts_kernels.cu - the device kernels of the routed experts (ffn_kernels.hpp): the combine of the
// slots in the fork's order, decode's doorbell (Strata's doorbell_publish_kernel, doorbell_wait_kernel and
// copy_from_mapped_kernel, src/kernels/cuda/elementwise.cu) and the token's copies through mapped memory.
#include "ffn_kernels.hpp"

#include "strata/kernels/dp4a.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dsv41 experts: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void combine_kernel(const float* __restrict__ yg, const float* __restrict__ ycpu,
                               const int32_t* __restrict__ row_of_pair, const float* __restrict__ w,
                               const float* __restrict__ shared, int n_used, int E, float* __restrict__ out) {
    const int64_t t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    float acc = 0.f;
    for (int s = 0; s < n_used; ++s) {
        const int64_t p = t * n_used + s;
        const int32_t r = row_of_pair[p];
        const float y = r >= 0 ? yg[(int64_t) r * E + e] : ycpu[(int64_t) (-r - 1) * E + e];
        const float v = __fmul_rn(y, w[p]);
        acc = s == 0 ? v : __fadd_rn(acc, v);
    }
    if (shared) acc = __fadd_rn(acc, shared[t * E + e]);
    out[t * E + e] = acc;
}

__global__ void add_shared_kernel(float* __restrict__ out, const float* __restrict__ shared, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fadd_rn(out[i], shared[i]);
}

// The payload into mapped pinned memory, then the ring. The ring is incremented in memory, never a value computed on
// the host (Strata's cloned-literal bug: a captured graph would ring the capture's value on every replay); the system
// fence orders the payload's stores before it for the host, and the store is volatile so it leaves the GPU's cache.
__global__ void doorbell_publish_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids, int64_t n_x,
                                        int64_t n_ids, float* x_out, int32_t* ids_out, uint32_t* seq) {
    for (int64_t i = threadIdx.x; i < n_x; i += blockDim.x) x_out[i] = x[i];
    for (int64_t i = threadIdx.x; i < n_ids; i += blockDim.x) ids_out[i] = ids[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
}

__global__ void doorbell_wait_kernel(const volatile uint32_t* flag, const volatile uint32_t* seq) {
    const uint32_t want = *seq;
    while (*flag != want) strata_spin_pause();
    __threadfence_system();
}

// block p: pair p's CPU row r = miss_row[p] (mapped memory, read volatile: the host wrote it before the flag) copied
// to ycpu row r, and the combine's row of the pair: -r - 1 for a CPU row, p for a hit (yg is pair-indexed)
__global__ void misses_in_kernel(const volatile float* ycpu_mapped, const volatile int32_t* miss_row, int E,
                                 float* __restrict__ ycpu, int32_t* __restrict__ row_of_pair) {
    const int64_t p = blockIdx.x;
    const int32_t r = miss_row[p];
    if (threadIdx.x == 0) row_of_pair[p] = r >= 0 ? -r - 1 : (int32_t) p;
    if (r < 0) return;
    const volatile float* src = ycpu_mapped + (int64_t) r * E;
    float* dst = ycpu + (int64_t) r * E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) dst[e] = src[e];
}

struct MappedCopies {
    MappedCopy c[kMaxMappedCopies];
};

// blockIdx.y: the copy; 16-byte words when both ends and the length allow, else 4-byte ones (Strata's
// copy_from_mapped_multi_kernel); the source read volatile, so every launch reads the host's current bytes
__global__ void copy_from_mapped_kernel(MappedCopies a) {
    const MappedCopy c = a.c[blockIdx.y];
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, st = (int64_t) gridDim.x * blockDim.x;
    if ((c.words & 3) == 0 && ((uintptr_t) c.dst & 15) == 0 && ((uintptr_t) c.src & 15) == 0) {
        const volatile float4* s = (const volatile float4*) c.src;
        for (int64_t i = i0; i < c.words / 4; i += st) ((float4*) c.dst)[i] = const_cast<const float4*>(s)[i];
    } else {
        const volatile int32_t* s = (const volatile int32_t*) c.src;
        for (int64_t i = i0; i < c.words; i += st) ((int32_t*) c.dst)[i] = s[i];
    }
}

// 16-byte stores, then block 0 the last n % 4 values
__global__ void copy_to_mapped_kernel(float* dst, const float* __restrict__ src, int64_t n) {
    const int64_t n4 = n / 4;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x)
        reinterpret_cast<float4*>(dst)[i] = reinterpret_cast<const float4*>(src)[i];
    if (blockIdx.x == 0 && threadIdx.x < n - 4 * n4) dst[4 * n4 + threadIdx.x] = src[4 * n4 + threadIdx.x];
}

}  // namespace

void copy_from_mapped(const MappedCopy* copies, int n, void* stream) {
    if (n < 1 || n > kMaxMappedCopies) {
        std::fprintf(stderr, "dsv41 experts: copy_from_mapped of %d copies (1..%d)\n", n, kMaxMappedCopies);
        std::exit(1);
    }
    MappedCopies a;
    int64_t most = 0;
    for (int i = 0; i < n; ++i) {
        a.c[i] = copies[i];
        most = copies[i].words > most ? copies[i].words : most;
    }
    const int64_t units = (most + 3) / 4;
    const unsigned bx = (unsigned) std::min<int64_t>(std::max<int64_t>((units + 255) / 256, 1), 32);
    copy_from_mapped_kernel<<<dim3(bx, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(a);
    ck(cudaGetLastError(), "copy_from_mapped_kernel");
}

void copy_to_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n < 1) return;
    if (((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "dsv41 experts: copy_to_mapped needs both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const unsigned blocks = (unsigned) std::min<int64_t>(std::max<int64_t>((n / 4 + 255) / 256, 1), 128);
    copy_to_mapped_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, src, n);
    ck(cudaGetLastError(), "copy_to_mapped_kernel");
}

void expert_doorbell_publish(const float* x, const int32_t* ids, int64_t n_x, int64_t n_ids, float* x_out,
                             int32_t* ids_out, uint32_t* seq, void* stream) {
    doorbell_publish_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(x, ids, n_x, n_ids, x_out, ids_out, seq);
    ck(cudaGetLastError(), "doorbell_publish_kernel");
}

void expert_doorbell_wait(const uint32_t* flag, const uint32_t* seq, void* stream) {
    doorbell_wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, seq);
    ck(cudaGetLastError(), "doorbell_wait_kernel");
}

void expert_misses_in(const float* ycpu_mapped, const int32_t* miss_row, int64_t P, int E, float* ycpu,
                      int32_t* row_of_pair, void* stream) {
    if (P < 1) return;
    misses_in_kernel<<<(unsigned) P, 256, 0, (cudaStream_t) stream>>>(ycpu_mapped, miss_row, E, ycpu, row_of_pair);
    ck(cudaGetLastError(), "misses_in_kernel");
}

void expert_combine(const float* yg, const float* ycpu, const int32_t* row_of_pair, const float* w, const float* shared,
                    int64_t T, int n_used, int E, float* out, void* stream) {
    if (T < 1) return;
    const dim3 grid((unsigned) ((E + 255) / 256), (unsigned) T);
    combine_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(yg, ycpu, row_of_pair, w, shared, n_used, E, out);
    ck(cudaGetLastError(), "combine_kernel");
}

void expert_add_shared(float* out, const float* shared, int64_t n, void* stream) {
    if (n < 1) return;
    add_shared_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(out, shared, n);
    ck(cudaGetLastError(), "add_shared_kernel");
}

}  // namespace strata::dsv41::engine
