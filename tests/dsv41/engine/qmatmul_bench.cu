// tests/dsv41/engine/qmatmul_bench.cu - bench of qmatmul at the real dense shapes (qmatmul_synth.hpp
// real_shapes) on random blocks (GPU, no model), against cuBLAS f16 (cublasGemmEx f16 x f16 -> f32, the projection
// path) on a ready f16 copy of the same shapes (the floor the dequant path pays its dequantization over):
//   decode, T = 1, 2, 4, 8: MMVQ's us alone and with its quantizer, the weight GB/s (the matrix's bytes over the
//       matmul's time; it reads them once for all T), the dequant path's us (qmatmul_f16 on ready f16 activations:
//       dequant into the scratch, then cuBLAS), cuBLAS f16's us and GB/s, and MMVQ's speedup over the dequant path;
//       then the crossover: the first T where the dequant path beats MMVQ, if any (qmatmul switches above T = 8);
//   prefill, T = 16, 64, 512 and 4096: ms and TFLOP/s (2 rows K T) of qmatmul's default dequant path (f16
//       conversion, dequant and cuBLAS), of the MMQ option (quantizer and MMQ), and of cuBLAS f16, and the default
//       path's time over cuBLAS f16's.
// The dequant scratch holds 64 Mi values (128 MiB): the output head and engram_wkv run in row slices.
// Medians of --reps runs (default 20 decode, 5 prefill) after two warm-up runs.
//
//   dsv41_engine_qmatmul_bench [--reps N] [--only NAME]
#include "qmatmul_synth.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace e = strata::dsv41::engine;

void ck(cudaError_t r, const char* w) {
    if (r != cudaSuccess) {
        std::fprintf(stderr, "dsv41_engine_qmatmul_bench: %s: %s\n", w, cudaGetErrorString(r));
        std::exit(2);
    }
}
void ckb(cublasStatus_t s, const char* w) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "dsv41_engine_qmatmul_bench: %s: cuBLAS status %d\n", w, (int) s);
        std::exit(2);
    }
}

__global__ void fill_f16_kernel(uint16_t* p, int64_t n, uint32_t seed, float scale) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t) i * 2654435761u ^ seed;
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        const float u = ((float) (h >> 8) * (1.0f / 16777216.0f) * 2.f - 1.f) * scale;
        p[i] = __half_as_ushort(__float2half(u));
    }
}
__global__ void fill_f32_kernel(float* p, int64_t n, uint32_t seed) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t) i * 2246822519u ^ seed;
        h ^= h >> 13;
        h *= 3266489917u;
        h ^= h >> 16;
        p[i] = (float) (h >> 8) * (1.0f / 16777216.0f) * 3.4641016f - 1.7320508f;
    }
}

template <typename F> float median_ms(int reps, cudaEvent_t a, cudaEvent_t b, F&& f) {
    for (int i = 0; i < 2; ++i) f();
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        ck(cudaEventRecord(a), "cudaEventRecord");
        f();
        ck(cudaEventRecord(b), "cudaEventRecord");
        ck(cudaEventSynchronize(b), "cudaEventSynchronize");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int reps_decode = 20, reps_prefill = 5;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "dsv41_engine_qmatmul_bench: %s needs a value\n", a.c_str()); return 2; }
        const char* v = argv[++i];
        if (a == "--reps") reps_decode = reps_prefill = std::max(1, std::atoi(v));
        else if (a == "--only") only = v;
        else { std::fprintf(stderr, "dsv41_engine_qmatmul_bench: unknown option %s\n", a.c_str()); return 2; }
    }
    int dev = 0;
    cudaDeviceProp p{};
    ck(cudaGetDevice(&dev), "cudaGetDevice");
    ck(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
    std::printf("device: %s, sm_%d%d, %d SMs, %.1f GiB\n", p.name, p.major, p.minor, p.multiProcessorCount,
                (double) p.totalGlobalMem / (1 << 30));
    constexpr int64_t kScratch = 64ll << 20;
    e::QmmContext qf16, qmmq;
    std::string err;
    if (!qf16.init(err, e::QPrefill::DequantF16, kScratch) || !qmmq.init(err, e::QPrefill::Mmq, 0)) {
        std::fprintf(stderr, "qmatmul: %s\n", err.c_str());
        return 1;
    }
    cublasHandle_t blas = nullptr;
    ckb(cublasCreate(&blas), "cublasCreate");
    cudaEvent_t ev0, ev1;
    ck(cudaEventCreate(&ev0), "cudaEventCreate");
    ck(cudaEventCreate(&ev1), "cudaEventCreate");

    const std::vector<int64_t> decode_T = {1, 2, 4, 8}, prefill_T = {16, 64, 512, 4096};
    const int64_t Tmax = 4096;
    double sum_q_us = 0, sum_f_us = 0;   // the attention matrices of one layer at T = 1 (q_a, q_b, kv, output_a, output_b)
    for (const qsynth::Shape& sh : qsynth::real_shapes()) {
        if (!only.empty() && only != sh.name) continue;
        const int64_t K = sh.k * sh.groups;
        std::mt19937_64 rng(1);
        const std::vector<uint8_t> wb = qsynth::random_blocks(sh.type, sh.rows, sh.k, 1.f / std::sqrt((float) sh.k), rng);
        void* w = nullptr;
        uint16_t *wf = nullptr, *xh = nullptr;
        float *x = nullptr, *y = nullptr;
        void* acts = nullptr;
        const size_t f16_bytes = (size_t) (sh.rows * sh.k) * 2;
        ck(cudaMalloc(&w, wb.size()), "cudaMalloc");
        ck(cudaMemcpy(w, wb.data(), wb.size(), cudaMemcpyHostToDevice), "upload");
        ck(cudaMalloc(&wf, f16_bytes), "cudaMalloc");
        ck(cudaMalloc(&x, sizeof(float) * (size_t) (Tmax * K)), "cudaMalloc");
        ck(cudaMalloc(&xh, 2 * (size_t) (Tmax * K)), "cudaMalloc");
        ck(cudaMalloc(&y, sizeof(float) * (size_t) (Tmax * sh.rows)), "cudaMalloc");
        ck(cudaMalloc(&acts, std::max(e::qact_bytes_max(e::QPrefill::Mmq, Tmax, K), 2 * (size_t) (Tmax * K))), "cudaMalloc");
        fill_f16_kernel<<<1024, 256>>>(wf, (int64_t) (sh.rows * sh.k), 7, 1.7f / std::sqrt((float) sh.k));
        fill_f32_kernel<<<1024, 256>>>(x, Tmax * K, 9);
        fill_f16_kernel<<<1024, 256>>>(xh, Tmax * K, 11, 1.7f);
        ck(cudaDeviceSynchronize(), "fill");
        const e::QWeightView wv{sh.type, sh.k, sh.rows, w};
        const int64_t rg = sh.rows / sh.groups;
        auto blas_run = [&](int64_t T) {
            const float one = 1.f, zero = 0.f;
            if (sh.groups == 1) {
                ckb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) sh.rows, (int) T, (int) sh.k, &one, wf, CUDA_R_16F, (int) sh.k,
                                 xh, CUDA_R_16F, (int) K, &zero, y, CUDA_R_32F, (int) sh.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                    "cublasGemmEx");
            } else {
                ckb(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) rg, (int) T, (int) sh.k, &one, wf, CUDA_R_16F,
                                               (int) sh.k, (long long) rg * sh.k, xh, CUDA_R_16F, (int) K, (long long) sh.k, &zero, y,
                                               CUDA_R_32F, (int) sh.rows, (long long) rg, sh.groups, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                    "cublasGemmStridedBatchedEx");
            }
        };
        std::printf("\n%s: %s, %lld rows x %lld%s, %.2f MiB (f16 %.2f MiB)\n", sh.name, e::qtype_name(sh.type), (long long) sh.rows,
                    (long long) sh.k, sh.groups > 1 ? (" in " + std::to_string(sh.groups) + " groups").c_str() : "",
                    (double) wb.size() / (1 << 20), (double) f16_bytes / (1 << 20));
        std::printf("  %-6s %10s %12s %9s %14s %12s %9s %8s\n", "T", "mmvq us", "+quantize us", "GB/s", "dq+cuBLAS us", "cuBLAS f16 us",
                    "GB/s", "speedup");
        int64_t crossover = 0;
        for (int64_t T : decode_T) {
            e::QActs a;
            e::qact_quantize(x, nullptr, K, T, K, e::qact_layout(sh.type, T), acts, nullptr, a);
            const float mm = median_ms(reps_decode, ev0, ev1, [&] { e::qmatmul_q(qf16, wv, a, y, sh.rows, sh.groups, nullptr, sh.name); });
            const float qm = median_ms(reps_decode, ev0, ev1, [&] {
                e::qmatmul(qf16, wv, x, K, T, y, sh.rows, acts, sh.groups, nullptr, sh.name);
            });
            const float dq = median_ms(reps_decode, ev0, ev1, [&] { e::qmatmul_f16(qf16, wv, xh, K, T, y, sh.rows, sh.groups, nullptr, sh.name); });
            const float fb = median_ms(reps_decode, ev0, ev1, [&] { blas_run(T); });
            if (!crossover && dq < qm) crossover = T;
            std::printf("  %-6lld %10.1f %12.1f %9.0f %14.1f %12.1f %9.0f %7.2fx\n", (long long) T, mm * 1e3, qm * 1e3,
                        (double) wb.size() / (mm * 1e-3) / 1e9, dq * 1e3, fb * 1e3, (double) f16_bytes / (fb * 1e-3) / 1e9, dq / qm);
            if (T == 1 && sh.type == e::kQTypeQ2K &&
                (std::strncmp(sh.name, "attn_q", 6) == 0 || std::strncmp(sh.name, "attn_kv", 7) == 0 ||
                 std::strncmp(sh.name, "attn_output", 11) == 0)) {
                sum_q_us += qm * 1e3;
                sum_f_us += fb * 1e3;
            }
        }
        if (crossover) std::printf("  crossover: the dequant path beats MMVQ from T = %lld\n", (long long) crossover);
        else std::printf("  crossover: MMVQ is faster at every T up to %lld\n", (long long) decode_T.back());
        std::printf("  %-6s %14s %9s %10s %9s %13s %9s %9s\n", "T", "dq+cuBLAS ms", "TFLOP/s", "mmq ms", "TFLOP/s", "cuBLAS f16 ms",
                    "TFLOP/s", "dq/f16");
        for (int64_t T : prefill_T) {
            const double flop = 2.0 * (double) sh.rows * (double) sh.k * (double) T;
            const float dq = median_ms(reps_prefill, ev0, ev1, [&] {
                e::qmatmul(qf16, wv, x, K, T, y, sh.rows, acts, sh.groups, nullptr, sh.name);
            });
            const float mq = median_ms(reps_prefill, ev0, ev1, [&] {
                e::qmatmul(qmmq, wv, x, K, T, y, sh.rows, acts, sh.groups, nullptr, sh.name);
            });
            const float fb = median_ms(reps_prefill, ev0, ev1, [&] { blas_run(T); });
            std::printf("  %-6lld %14.3f %9.1f %10.3f %9.1f %13.3f %9.1f %8.2fx\n", (long long) T, dq, flop / (dq * 1e-3) / 1e12, mq,
                        flop / (mq * 1e-3) / 1e12, fb, flop / (fb * 1e-3) / 1e12, dq / fb);
        }
        cudaFree(w);
        cudaFree(wf);
        cudaFree(x);
        cudaFree(xh);
        cudaFree(y);
        cudaFree(acts);
    }
    if (only.empty())
        std::printf("\none layer's q, kv and output projections at T = 1 (attn_q_a, attn_q_b, attn_kv, attn_output_a, "
                    "attn_output_b, quantize included): %.1f us on q2_K, %.1f us on cuBLAS f16\n", sum_q_us, sum_f_us);
    cublasDestroy(blas);
    return 0;
}
