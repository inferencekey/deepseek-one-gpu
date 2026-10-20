// tests/dsv41/engine/machine_probe.cu - dsv41_machine_probe: the machine qualification of a pod session.
// tools/dsv41/pod/pod-session.sh runs it right after dsv41_cuda_probe, before
// the payload, the download or the load, and a machine that fails it is deleted, remembered and replaced by the next
// offer: one RTX PRO 6000 WS machine passed every check that existed, then ran the GPU's
// front work 4-5x slower than normal and cost 2.889 USD for nothing. In a few seconds it
// measures what such a machine lacks: the GPU's memory read bandwidth, pinned host-to-device and device-to-host copy
// bandwidth, the mapped-memory doorbell round trip of the engine's decode (experts.cpp decode_serve with
// experts_kernels.cu's doorbell kernels: the GPU publishes into mapped pinned memory, a host thread spins and
// answers, the GPU spins on the answer), the host's memory read bandwidth with 16 threads (the engine's --threads),
// and the SM clock, power draw and enforced power limit that nvidia-smi reports while the read kernel runs.
//
// It is built on its own (tools/dsv41/pod/build-machine-probe.sh: nvcc in each flavor's builder image), not by the
// engine's CMake, so the payload's dsv41-engine stays the measured binary. Every bound is an argument (e07.sh holds
// the defaults and their env overrides). The bounds hold for every GPU model:
// the SM clock under load and the enforced power limit are ratios of the GPU's own nvidia-smi clocks.max.sm and
// power.default_limit (an A100's 1410 MHz and 300-400 W are its maximum, not a slow machine), and the GPU read
// bandwidth's floor is the --gpu-read-bound whose NAME is part of the GPU's name (one per known model), or
// --min-gpu-read-gbs for every model; a GPU no bound names has its read bandwidth reported only. Each measure prints
// one line
//   dsv41_machine_probe: measure NAME VALUE bound OP BOUND ok|FAIL (how it was measured)
// (bound "none" and status "info" for the reported-only measures; a ratio's BOUND is its absolute value, the ratio
// and the GPU's maximum in the parentheses), then the machine line and the verdict:
// "dsv41_machine_probe: PASS, ..." or "dsv41_machine_probe: FAIL, " and each measure past its bound. Exit 1 on a FAIL
// or a CUDA error (named), 2 on bad arguments.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr size_t kGpuReadBytes = size_t(2) << 30;    // >= 2 GiB streamed
constexpr size_t kCopyBytes = size_t(1) << 30;       // >= 1 GiB per copy
constexpr int kCopyReps = 5;
constexpr double kLoadMinS = 3.0, kLoadMaxS = 10.0;  // the read kernel's load while nvidia-smi samples
constexpr int kMinSamples = 3;
constexpr int kDoorbellWarm = 100, kDoorbellTimed = 2000;
constexpr int kRowFloats = 4096;                     // one token's row (n_embd) each way, as decode's doorbell
constexpr int kHostThreads = 16;
constexpr size_t kHostBytesPerThread = size_t(256) << 20;
constexpr int kHostReps = 3;

[[noreturn]] void die(const char* what, cudaError_t e) {
    std::printf("dsv41_machine_probe: FAIL %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) die(what, e);
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// the nearest-rank percentile p (0-1)
double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const size_t k = (size_t) std::max(1.0, std::ceil(p * (double) v.size()));
    return v[k - 1];
}

std::vector<std::string> failed;

// one measure's line against its bound: op ">=" (a floor) or "<=" (a ceiling)
void measure(const char* name, double v, const char* op, double bound, const std::string& how) {
    const bool ok = op[0] == '>' ? v >= bound : v <= bound;
    std::printf("dsv41_machine_probe: measure %s %.1f bound %s%g %s (%s)\n", name, v, op, bound, ok ? "ok" : "FAIL",
                how.c_str());
    if (!ok) {
        char b[160];
        std::snprintf(b, sizeof b, "%s %.1f (bound %s%g)", name, v, op, bound);
        failed.push_back(b);
    }
}

void info(const char* name, double v, const std::string& how) {
    std::printf("dsv41_machine_probe: measure %s %.1f bound none info (%s)\n", name, v, how.c_str());
}

// a measure with no value (its source failed): a FAIL naming why; bound is the bound's text ("<=50", ">=0.7xmax")
void missing(const char* name, const std::string& bound, const std::string& why) {
    std::printf("dsv41_machine_probe: measure %s n/a bound %s FAIL (%s)\n", name, bound.c_str(), why.c_str());
    failed.push_back(std::string(name) + " n/a (" + why + ")");
}

std::string op_bound(const char* op, double bound) {
    char b[64];
    std::snprintf(b, sizeof b, "%s%g", op, bound);
    return b;
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// a streaming read of n uint4, four independent loads in flight per thread; the sink is written only on a value the
// memset pattern never xors to, so the loads stay
__global__ void read_kernel(const uint4* __restrict__ x, int64_t n, uint32_t* sink) {
    const int64_t s = (int64_t) gridDim.x * blockDim.x;
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (; i + 3 * s < n; i += 4 * s) {
        const uint4 a = x[i], b = x[i + s], c = x[i + 2 * s], d = x[i + 3 * s];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += s) acc ^= x[i].x ^ x[i].y ^ x[i].z ^ x[i].w;
    if (acc == 0x9e3779b9u) *sink = acc;
}

// The engine's doorbell, n times in one launch: the block writes one row into mapped memory, the system fence, then
// the ring (a volatile store, as doorbell_publish_kernel's); thread 0 spins with a pause on the host's answer (as
// doorbell_wait_kernel's) and the block reads the host's row back (as misses_in_kernel). ns[i]: the round trip by
// the GPU's global timer; a host that never answers within 1 s stops it with *timed_out set to the ring.
__global__ void doorbell_kernel(float* x, const volatile float* y, volatile uint32_t* ring, const volatile uint32_t* flag,
                                int n, uint32_t* ns, uint32_t* timed_out, float* sink) {
    __shared__ int stop;
    float acc = 0.f;
    for (int r = 1; r <= n; ++r) {
        const uint64_t t0 = globaltimer_ns();
        for (int k = threadIdx.x; k < kRowFloats; k += blockDim.x) x[k] = (float) (r + k);
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            stop = 0;
            *ring = (uint32_t) r;
            while (*flag != (uint32_t) r) {
                if (globaltimer_ns() - t0 > 1000000000ull) {
                    *timed_out = (uint32_t) r;
                    stop = 1;
                    break;
                }
                __nanosleep(100);
            }
            __threadfence_system();
        }
        __syncthreads();
        if (stop) return;
        for (int k = threadIdx.x; k < kRowFloats; k += blockDim.x) acc += y[k];
        __syncthreads();
        if (threadIdx.x == 0) {
            const uint64_t dt = globaltimer_ns() - t0;
            ns[r - 1] = dt > 0xffffffffull ? 0xffffffffu : (uint32_t) dt;
        }
    }
    if (acc == -1.f) *sink = acc;
}

// the doorbell poll's pause (experts.cpp cpu_relax)
void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

std::string first_line(const char* path, const char* key) {
    FILE* f = std::fopen(path, "r");
    if (!f) return "";
    char b[512];
    std::string out;
    while (std::fgets(b, sizeof b, f))
        if (std::strncmp(b, key, std::strlen(key)) == 0) {
            out = b;
            break;
        }
    std::fclose(f);
    const size_t c = out.find(':');
    out = c == std::string::npos ? out : out.substr(c + 1);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    while (!out.empty() && (out.front() == ' ' || out.front() == '\t')) out.erase(out.begin());
    return out;
}

// nvidia-smi's SM clock, power draw, enforced power limit, maximum SM clock and default power limit of the GPU uuid,
// every call until stop
struct Sample {
    double sm_mhz, power_w, limit_w, max_sm_mhz, default_limit_w;
};
// (err, the first refusal, is read only after the join; bad says one came)
void sample_gpu(const std::string& uuid, const std::atomic<bool>& stop, std::vector<Sample>& out,
                std::atomic<int>& n, std::atomic<bool>& bad, std::string& err) {
    const std::string cmd = "timeout 5 nvidia-smi --query-gpu=clocks.sm,power.draw,enforced.power.limit,clocks.max.sm,"
                            "power.default_limit --format=csv,noheader,nounits -i " + uuid + " 2>&1";
    while (!stop.load()) {
        FILE* p = popen(cmd.c_str(), "r");
        if (!p) {
            err = "popen of nvidia-smi failed";
            bad = true;
            return;
        }
        char b[256] = {0};
        const bool got = std::fgets(b, sizeof b, p) != nullptr;
        pclose(p);
        Sample s{};
        if (got && std::sscanf(b, "%lf , %lf , %lf , %lf , %lf", &s.sm_mhz, &s.power_w, &s.limit_w, &s.max_sm_mhz,
                               &s.default_limit_w) == 5) {
            out.push_back(s);
            ++n;
        } else {
            if (err.empty()) {
                err = std::string("nvidia-smi said: ") + (got ? b : "nothing");
                while (!err.empty() && err.back() == '\n') err.pop_back();
            }
            bad = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

int usage(const char* why) {
    std::printf("dsv41_machine_probe: %s\nusage: dsv41_machine_probe [--gpu-read-bound NAME=GBS ...] "
                "[--min-gpu-read-gbs X] --min-h2d-gbs X --min-d2h-gbs X --max-doorbell-us X --min-host-read-gbs X "
                "--min-sm-ratio R --min-power-ratio R\n",
                why);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    // the bounds (e07.sh passes the defaults or their overrides): every one required but the GPU read's
    // (--min-gpu-read-gbs, for every model, or the per-model --gpu-read-bound list); the ratios at most 1
    const char* names[] = {"--min-gpu-read-gbs", "--min-h2d-gbs", "--min-d2h-gbs", "--max-doorbell-us",
                           "--min-host-read-gbs", "--min-sm-ratio", "--min-power-ratio"};
    constexpr int kBounds = 7;
    double bound[kBounds];
    bool set[kBounds] = {};
    std::vector<std::pair<std::string, double>> model_reads;   // --gpu-read-bound NAME=GBS, in order
    for (int a = 1; a < argc; ++a) {
        char* end = nullptr;
        if (std::strcmp(argv[a], "--gpu-read-bound") == 0) {
            const char* eq = a + 1 < argc ? std::strrchr(argv[a + 1], '=') : nullptr;
            double gbs = 0;
            if (!eq || eq == argv[a + 1] || (gbs = std::strtod(eq + 1, &end), *end != '\0' || end == eq + 1) ||
                !(gbs > 0))
                return usage("--gpu-read-bound needs NAME=GBS, GBS a number above 0");
            model_reads.emplace_back(std::string(argv[a + 1], (size_t) (eq - argv[a + 1])), gbs);
            ++a;
            continue;
        }
        int k = 0;
        while (k < kBounds && std::strcmp(argv[a], names[k]) != 0) ++k;
        if (k == kBounds) return usage((std::string("unknown argument ") + argv[a]).c_str());
        if (a + 1 >= argc || (bound[k] = std::strtod(argv[a + 1], &end), *end != '\0') || !(bound[k] > 0))
            return usage((std::string(names[k]) + " needs a number above 0").c_str());
        if (k >= 5 && bound[k] > 1) return usage((std::string(names[k]) + " is a ratio: at most 1").c_str());
        set[k] = true;
        ++a;
    }
    for (int k = 1; k < kBounds; ++k)
        if (!set[k]) return usage((std::string("no ") + names[k]).c_str());
    const double min_h2d = bound[1], min_d2h = bound[2], max_doorbell = bound[3], min_host_read = bound[4],
                 min_sm_ratio = bound[5], min_power_ratio = bound[6];

    int n_dev = 0;
    ck(cudaGetDeviceCount(&n_dev), "cudaGetDeviceCount");
    if (n_dev < 1) {
        std::printf("dsv41_machine_probe: FAIL no CUDA device\n");
        return 1;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    char uuid[48];
    const unsigned char* u = (const unsigned char*) prop.uuid.bytes;
    std::snprintf(uuid, sizeof uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0],
                  u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    // the GPU read's floor: --min-gpu-read-gbs, else the first --gpu-read-bound whose NAME is in the GPU's name
    double min_gpu_read = 0;
    std::string read_bound_src;
    if (set[0]) {
        min_gpu_read = bound[0];
        read_bound_src = "--min-gpu-read-gbs, every model";
    } else
        for (const auto& mb : model_reads)
            if (std::strstr(prop.name, mb.first.c_str())) {
                min_gpu_read = mb.second;
                read_bound_src = "the bound of \"" + mb.first + "\"";
                break;
            }
    cudaStream_t stream;
    ck(cudaStreamCreate(&stream), "cudaStreamCreate");

    // ---- GPU read bandwidth, nvidia-smi sampling while the kernel runs back to back
    uint4* d_buf = nullptr;
    uint32_t* d_sink = nullptr;
    ck(cudaMalloc(&d_buf, kGpuReadBytes), "cudaMalloc (2 GiB)");
    ck(cudaMalloc(&d_sink, 64), "cudaMalloc (sink)");
    ck(cudaMemset(d_buf, 0x5a, kGpuReadBytes), "cudaMemset");
    const int64_t n_vec = (int64_t) (kGpuReadBytes / sizeof(uint4));
    const int blocks = prop.multiProcessorCount * 8;
    constexpr int kBatch = 32;
    cudaEvent_t ev[kBatch + 1];
    for (cudaEvent_t& e : ev) ck(cudaEventCreate(&e), "cudaEventCreate");
    for (int w = 0; w < 3; ++w) read_kernel<<<blocks, 256, 0, stream>>>(d_buf, n_vec, d_sink);
    ck(cudaGetLastError(), "read_kernel launch");
    ck(cudaStreamSynchronize(stream), "read_kernel warm-up");
    std::atomic<bool> stop{false};
    std::atomic<int> n_samples{0};
    std::atomic<bool> smi_bad{false};
    std::vector<Sample> samples;
    std::string smi_err;
    std::thread sampler(sample_gpu, std::string(uuid), std::cref(stop), std::ref(samples), std::ref(n_samples),
                        std::ref(smi_bad), std::ref(smi_err));
    std::vector<double> rep_ms;
    const double t_load = now_s();
    // at least kLoadMinS, longer (up to kLoadMaxS) while nvidia-smi has given fewer than kMinSamples samples
    while (now_s() - t_load < kLoadMinS ||
           (n_samples.load() < kMinSamples && !smi_bad.load() && now_s() - t_load < kLoadMaxS)) {
        ck(cudaEventRecord(ev[0], stream), "cudaEventRecord");
        for (int b = 0; b < kBatch; ++b) {
            read_kernel<<<blocks, 256, 0, stream>>>(d_buf, n_vec, d_sink);
            ck(cudaEventRecord(ev[b + 1], stream), "cudaEventRecord");
        }
        ck(cudaGetLastError(), "read_kernel launch");
        ck(cudaEventSynchronize(ev[kBatch]), "read_kernel");
        for (int b = 0; b < kBatch; ++b) {
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, ev[b], ev[b + 1]), "cudaEventElapsedTime");
            rep_ms.push_back(ms);
        }
    }
    const double load_s = now_s() - t_load;
    stop = true;
    sampler.join();
    {
        char how[160];
        std::snprintf(how, sizeof how, "2 GiB streamed, median of %zu reps over %.1f s, %d blocks of 256",
                      rep_ms.size(), load_s, blocks);
        const double gbs = (double) kGpuReadBytes / (median(rep_ms) * 1e6);
        if (min_gpu_read > 0)
            measure("gpu_read_gbs", gbs, ">=", min_gpu_read, std::string(how) + ", " + read_bound_src);
        else
            info("gpu_read_gbs", gbs, std::string(how) + ", no bound for " + prop.name);
    }

    // ---- pinned copies, 1 GiB each way
    void* h_pin = nullptr;
    ck(cudaMallocHost(&h_pin, kCopyBytes), "cudaMallocHost (1 GiB)");
    std::memset(h_pin, 1, kCopyBytes);
    for (int dir = 0; dir < 2; ++dir) {
        const bool h2d = dir == 0;
        std::vector<double> ms_v;
        for (int r = 0; r <= kCopyReps; ++r) {
            ck(cudaEventRecord(ev[0], stream), "cudaEventRecord");
            ck(h2d ? cudaMemcpyAsync(d_buf, h_pin, kCopyBytes, cudaMemcpyHostToDevice, stream)
                   : cudaMemcpyAsync(h_pin, d_buf, kCopyBytes, cudaMemcpyDeviceToHost, stream),
               h2d ? "cudaMemcpyAsync H2D" : "cudaMemcpyAsync D2H");
            ck(cudaEventRecord(ev[1], stream), "cudaEventRecord");
            ck(cudaEventSynchronize(ev[1]), "the pinned copy");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, ev[0], ev[1]), "cudaEventElapsedTime");
            if (r > 0) ms_v.push_back(ms);   // the first is a warm-up
        }
        measure(h2d ? "h2d_gbs" : "d2h_gbs", (double) kCopyBytes / (median(ms_v) * 1e6), ">=", h2d ? min_h2d : min_d2h,
                "1 GiB pinned, median of 5 after a warm-up");
    }
    ck(cudaFreeHost(h_pin), "cudaFreeHost");

    // ---- the doorbell round trip in mapped pinned memory (the ring and the flag on their own 64-byte lines)
    float *h_x = nullptr, *h_y = nullptr, *d_x = nullptr, *d_y = nullptr;
    uint32_t *h_bell = nullptr, *d_bell = nullptr, *d_ns = nullptr, *d_timeout = nullptr;
    ck(cudaHostAlloc(&h_x, kRowFloats * sizeof(float), cudaHostAllocMapped), "cudaHostAlloc (doorbell row)");
    ck(cudaHostAlloc(&h_y, kRowFloats * sizeof(float), cudaHostAllocMapped), "cudaHostAlloc (answer row)");
    ck(cudaHostAlloc(&h_bell, 128, cudaHostAllocMapped), "cudaHostAlloc (ring and flag)");
    std::memset(h_bell, 0, 128);
    std::memset(h_y, 0, kRowFloats * sizeof(float));
    ck(cudaHostGetDevicePointer(&d_x, h_x, 0), "cudaHostGetDevicePointer (row)");
    ck(cudaHostGetDevicePointer(&d_y, h_y, 0), "cudaHostGetDevicePointer (answer)");
    ck(cudaHostGetDevicePointer(&d_bell, h_bell, 0), "cudaHostGetDevicePointer (ring)");
    const int n_rings = kDoorbellWarm + kDoorbellTimed;
    ck(cudaMalloc(&d_ns, n_rings * sizeof(uint32_t)), "cudaMalloc (round trips)");
    ck(cudaMalloc(&d_timeout, sizeof(uint32_t)), "cudaMalloc (timeout)");
    ck(cudaMemset(d_timeout, 0, sizeof(uint32_t)), "cudaMemset (timeout)");
    volatile uint32_t* ring = h_bell;
    volatile uint32_t* flag = h_bell + 16;
    std::atomic<int> host_timeout{0};
    // the host's side, as decode_serve's: spin with a pause on the ring, read the row, write the answer, then the flag
    std::thread host([&] {
        volatile const float* x = h_x;
        volatile float* y = h_y;
        for (int r = 1; r <= n_rings; ++r) {
            const double t0 = now_s();
            for (uint32_t spins = 0; *ring != (uint32_t) r; ++spins) {
                cpu_relax();
                if ((spins & 1023u) == 0 && now_s() - t0 > 2.0) {
                    host_timeout = r;
                    return;
                }
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            float s = 0.f;
            for (int k = 0; k < kRowFloats; ++k) s += x[k];
            for (int k = 0; k < kRowFloats; ++k) y[k] = s;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *flag = (uint32_t) r;
        }
    });
    doorbell_kernel<<<1, 256, 0, stream>>>(d_x, d_y, d_bell, d_bell + 16, n_rings, d_ns, d_timeout, (float*) d_sink);
    const cudaError_t launch = cudaGetLastError();
    const cudaError_t done = launch == cudaSuccess ? cudaStreamSynchronize(stream) : launch;
    if (done != cudaSuccess) {
        host.join();   // the host thread gives up within 2 s of its last ring
        die("doorbell_kernel", done);
    }
    host.join();
    uint32_t timed_out = 0;
    std::vector<uint32_t> ns(n_rings);
    ck(cudaMemcpy(&timed_out, d_timeout, sizeof timed_out, cudaMemcpyDeviceToHost), "cudaMemcpy (timeout)");
    ck(cudaMemcpy(ns.data(), d_ns, n_rings * sizeof(uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy (round trips)");
    if (timed_out || host_timeout.load()) {
        char why[160];
        std::snprintf(why, sizeof why, "the doorbell stopped at ring %u of %d: no answer within 1 s (host side %d)",
                      timed_out, n_rings, host_timeout.load());
        missing("doorbell_median_us", op_bound("<=", max_doorbell), why);
    } else {
        std::vector<double> us;
        for (int r = kDoorbellWarm; r < n_rings; ++r) us.push_back(ns[(size_t) r] * 1e-3);
        measure("doorbell_median_us", median(us), "<=", max_doorbell,
                "2000 round trips after 100, one 4096-float row each way, the GPU's global timer");
        info("doorbell_p99_us", percentile(us, 0.99), "the same round trips");
    }

    // ---- host memory read with 16 threads, each over its own 256 MiB (first touched by itself)
    std::atomic<int> ready{0}, finished{0}, go{0};
    std::atomic<uint64_t> sink{0};
    std::atomic<bool> alloc_failed{false};
    std::vector<std::thread> readers;
    for (int t = 0; t < kHostThreads; ++t)
        readers.emplace_back([&] {
            const size_t n = kHostBytesPerThread / sizeof(uint64_t);
            uint64_t* p = (uint64_t*) std::aligned_alloc(64, kHostBytesPerThread);
            if (!p) alloc_failed = true;
            else std::memset(p, 1, kHostBytesPerThread);
            ++ready;
            for (int rep = 1; rep <= kHostReps; ++rep) {
                while (go.load() < rep) std::this_thread::yield();
                uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                if (p)
                    for (size_t i = 0; i < n; i += 4) {
                        a0 += p[i];
                        a1 += p[i + 1];
                        a2 += p[i + 2];
                        a3 += p[i + 3];
                    }
                sink += a0 + a1 + a2 + a3;
                ++finished;
            }
            std::free(p);
        });
    while (ready.load() < kHostThreads) std::this_thread::yield();
    double best_s = 0;
    for (int rep = 1; rep <= kHostReps; ++rep) {
        const double t0 = now_s();
        go = rep;
        while (finished.load() < rep * kHostThreads) std::this_thread::yield();
        const double s = now_s() - t0;
        best_s = rep == 1 ? s : std::min(best_s, s);
    }
    for (std::thread& t : readers) t.join();
    if (alloc_failed.load())
        missing("host_read_gbs", op_bound(">=", min_host_read), "a reader's 256 MiB allocation failed");
    else
        measure("host_read_gbs", (double) kHostThreads * (double) kHostBytesPerThread / (best_s * 1e9), ">=",
                min_host_read, "16 threads over 256 MiB each, the best of 3, as the engine's init rate");

    // ---- the GPU under the read kernel's load, from nvidia-smi; the bounds are ratios of the GPU's own maximums
    char sm_ratio[64], power_ratio[64];
    std::snprintf(sm_ratio, sizeof sm_ratio, ">=%gxclocks.max.sm", min_sm_ratio);
    std::snprintf(power_ratio, sizeof power_ratio, ">=%gxpower.default_limit", min_power_ratio);
    if (samples.empty()) {
        const std::string why = "no nvidia-smi sample during the read load" + (smi_err.empty() ? "" : ": " + smi_err);
        missing("sm_clock_mhz", sm_ratio, why);
        missing("power_limit_w", power_ratio, why);
    } else {
        std::vector<double> sm, pw, lim, max_sm, def_lim;
        for (const Sample& s : samples) {
            sm.push_back(s.sm_mhz);
            pw.push_back(s.power_w);
            lim.push_back(s.limit_w);
            max_sm.push_back(s.max_sm_mhz);
            def_lim.push_back(s.default_limit_w);
        }
        const double sm_max = median(max_sm), lim_default = median(def_lim);
        char how[200];
        std::snprintf(how, sizeof how, "median of %zu nvidia-smi samples during the read load, min %.0f, max %.0f; "
                      "%g x clocks.max.sm %.0f", sm.size(), *std::min_element(sm.begin(), sm.end()),
                      *std::max_element(sm.begin(), sm.end()), min_sm_ratio, sm_max);
        measure("sm_clock_mhz", median(sm), ">=", min_sm_ratio * sm_max, how);
        info("sm_clock_max_mhz", sm_max, "nvidia-smi clocks.max.sm");
        info("power_draw_w", median(pw), "median of the same samples");
        std::snprintf(how, sizeof how, "nvidia-smi enforced.power.limit; %g x power.default_limit %.0f",
                      min_power_ratio, lim_default);
        measure("power_limit_w", median(lim), ">=", min_power_ratio * lim_default, how);
        info("power_default_limit_w", lim_default, "nvidia-smi power.default_limit");
    }

    const std::string cpu = first_line("/proc/cpuinfo", "model name");
    const std::string mem = first_line("/proc/meminfo", "MemTotal");
    std::printf("dsv41_machine_probe: machine %s (sm_%d%d, %d SMs, %.1f GiB, %s), CPU %s, %u CPUs online, MemTotal %s\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, (double) prop.totalGlobalMem / (1 << 30),
                uuid, cpu.empty() ? "unknown" : cpu.c_str(), std::thread::hardware_concurrency(),
                mem.empty() ? "unknown" : mem.c_str());
    if (failed.empty()) {
        std::printf("dsv41_machine_probe: PASS, every measure within its bound\n");
        return 0;
    }
    std::string f;
    for (const std::string& s : failed) f += (f.empty() ? "" : "; ") + s;
    std::printf("dsv41_machine_probe: FAIL, %s\n", f.c_str());
    return 1;
}
