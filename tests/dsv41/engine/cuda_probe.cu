// tests/dsv41/engine/cuda_probe.cu - dsv41_cuda_probe: a one-second check that a dev pod's GPU really runs CUDA, before
// tools/dsv41/pod/devgpu.sh uploads the payload (a RunPod RTX 4090 host answered nvidia-smi while every CUDA call
// failed with "unknown error"). It allocates, writes with a kernel, copies back and
// checks the values; any failure exits 1 naming the call.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

__global__ void fill_kernel(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.5f * (float) i;
}

int fail(const char* what, cudaError_t e) {
    std::printf("dsv41_cuda_probe: FAIL %s: %s\n", what, cudaGetErrorString(e));
    return 1;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (const cudaError_t e = cudaGetDeviceCount(&n_dev); e != cudaSuccess) return fail("cudaGetDeviceCount", e);
    if (n_dev < 1) {
        std::printf("dsv41_cuda_probe: FAIL no CUDA device\n");
        return 1;
    }
    cudaDeviceProp prop{};
    if (const cudaError_t e = cudaGetDeviceProperties(&prop, 0); e != cudaSuccess) return fail("cudaGetDeviceProperties", e);
    const int n = 1 << 20;
    float* d = nullptr;
    if (const cudaError_t e = cudaMalloc(&d, n * sizeof(float)); e != cudaSuccess) return fail("cudaMalloc", e);
    fill_kernel<<<(n + 255) / 256, 256>>>(d, n);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return fail("a kernel launch", e);
    std::vector<float> h(n);
    if (const cudaError_t e = cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost); e != cudaSuccess)
        return fail("cudaMemcpy", e);
    cudaFree(d);
    for (int i = 0; i < n; ++i)
        if (h[i] != 0.5f * (float) i) {
            std::printf("dsv41_cuda_probe: FAIL value %d is %g, expected %g\n", i, h[i], 0.5f * (float) i);
            return 1;
        }
    std::printf("dsv41_cuda_probe: ok, %s (sm_%d%d), %d device(s), a kernel and a copy of 4 MiB\n", prop.name, prop.major,
                prop.minor, n_dev);
    return 0;
}
