// src/dsv41/moe.cpp - the exported functions of libstrata-moe.so (include/strata/dsv41/moe.h) and its
// configuration, read once from the environment at the first call:
//   STRATA_MOE_HOT=<hot-set file>    the experts per layer to keep in VRAM (tools/dsv41/oracle/hot_set.py)
//   STRATA_MOE_VRAM_GB=<GiB>         the VRAM the slab may take; required when the hot set lists VRAM experts
//   STRATA_MOE_RAM_GB=<GiB>          the residency (moe_residency.h): the hot set's warm lists in that much
//                                    RAM and every other expert not in VRAM cold; needs a hot set with warm lists
//                                    (hot_set.py --ram-budget-gib), and 0 makes every expert outside VRAM cold.
//                                    Unset: the misses read the fork's mmapped tensors, as without the residency
//   STRATA_MOE_COLD=pagecache|direct how the cold experts reach the CPU (moe_residency.h, ColdPath):
//                                    pagecache (the default) computes them from the fork's mmapped tensors and
//                                    leaves its pages alone, direct reads them from their GGUF files with O_DIRECT
//                                    and drops the fork's pages; needs STRATA_MOE_RAM_GB
//   STRATA_MOE_RAM_MODE=pinned|locked|plain  how the warm tier holds its RAM (moe_residency.h, RamMode): pinned
//                                    (CUDA pinned host memory, the default with a CUDA device), locked (mlock,
//                                    the default without one) or plain (pre-faulted, not locked); needs
//                                    STRATA_MOE_RAM_GB
//   STRATA_MOE_COLD_THREADS=<n>      the reader threads of the warm fill and the direct cold tier (default 8)
//   STRATA_MOE_COLD_ROUND=<n>        direct: cold experts per round (default 16, more than one decode step
//                                    routes); the staging holds two rounds
//   STRATA_MOE_COUNTS=<path>         per-(layer, expert) activation counts written to that json file
//                                    (moe_counts.h; STRATA_MOE_COUNTS_EVERY=<n> calls between writes, default 4096)
//   STRATA_MOE_PROFILE=1             per-layer times, printed at exit (moe_profile.cpp)
//   STRATA_MOE_CPU=fast|ref          the CPU kernel of the misses: fast (the default) is strata-moe's own AVX2
//                                    kernel (cpu_kernel.h), ref the reference on ggml-cpu's vec_dot; both
//                                    give the same bits on an AVX2 ggml-cpu, fast only computes q2_K, q3_K, q4_K
//                                    and q6_K experts and aborts naming any other tensor type
//   STRATA_MOE_PREFILL_MIN, _GROUP, _SETS, _STAGE, _COLD  the prefill path of the GPU tier (prefill_config)
// With no hot set, or with one and no CUDA device, every pair runs on the CPU: that mode is logged once, and so is
// the CPU kernel. A hot set without VRAM experts needs no STRATA_MOE_VRAM_GB and makes no GPU tier.
#include "strata/dsv41/moe.h"

#include "cpu_kernel.h"
#include "moe_gpu.h"
#include "moe_op.h"
#include "moe_residency.h"

#include "ggml-cpu.h"
#include "ggml-cuda.h"
// struct ggml_custom_op_params, for the static_assert on STRATA_MOE_CLAMP_PARAM; the internal header's static
// helpers are unused here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "ggml-impl.h"
#pragma GCC diagnostic pop

#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <unistd.h>

namespace strata::dsv41 {
namespace {

// A GiB amount from the environment: positive, or also 0 with allow_zero.
double env_gib(const char* name, const char* env, bool allow_zero) {
    char* end = nullptr;
    errno = 0;
    const double gib = std::strtod(env, &end);
    if (errno != 0 || *end != '\0' || !(gib > 0.0 || (allow_zero && gib == 0.0))) {
        GGML_ABORT("strata-moe: %s=%s: expected a %s number of GiB", name, env, allow_zero ? "non-negative" : "positive");
    }
    return gib;
}

double vram_budget_gib() {
    const char* env = std::getenv("STRATA_MOE_VRAM_GB");
    if (env == nullptr || *env == '\0') {
        GGML_ABORT("strata-moe: STRATA_MOE_HOT lists VRAM experts but STRATA_MOE_VRAM_GB is not set (the VRAM budget "
                   "in GiB)");
    }
    return env_gib("STRATA_MOE_VRAM_GB", env, false);
}

// A whole number >= min from the environment, or fallback when unset.
int64_t env_whole(const char* name, int64_t fallback, int64_t min) {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') return fallback;
    char* end = nullptr;
    errno = 0;
    const long long n = std::strtoll(env, &end, 10);
    if (errno != 0 || *end != '\0' || n < min || n > INT32_MAX) {
        GGML_ABORT("strata-moe: %s=%s: expected a whole number >= %lld", name, env, (long long) min);
    }
    return n;
}

int64_t env_count(const char* name, int64_t fallback) { return env_whole(name, fallback, 1); }

// the prefill path (moe_gpu.h, PrefillConfig), logged when the GPU tier takes it:
//   STRATA_MOE_PREFILL_MIN=<n>    batch calls of at least n tokens stream every expert outside the slab to the GPU
//                                 (default 64: a 64-token ubatch already routes 384 pairs, about one per expert;
//                                 0: off)
//   STRATA_MOE_PREFILL_GROUP=<n>  experts per ring set (default 16: 196 MiB of DeepSeek-V4.1 Q2_K/Q3_K per set, a
//                                 pass of ~1024 pairs at a 4096-token ubatch)
//   STRATA_MOE_PREFILL_SETS=<n>   ring sets (default 3: one being computed, one being copied, one queued)
//   STRATA_MOE_PREFILL_STAGE=<n>  pinned staging sets for the staged experts (default 2: one being filled by the CPU
//                                 threads while the copy stream reads the other)
//   STRATA_MOE_PREFILL_COLD=pagecache|direct  how the staged experts reach the staging (moe_prefill.h, PrefillCold):
//                                 pagecache (the default) copies them from the fork's mapping on the op's threads,
//                                 direct reads them with O_DIRECT on the residency's reader threads; direct needs
//                                 STRATA_MOE_RAM_GB. Logged with the prefill path
PrefillConfig prefill_config() {
    PrefillConfig p;
    p.min_tokens = env_whole("STRATA_MOE_PREFILL_MIN", 64, 0);
    p.set_size = (int) env_count("STRATA_MOE_PREFILL_GROUP", 16);
    p.n_sets = (int) env_count("STRATA_MOE_PREFILL_SETS", 3);
    p.n_stage_sets = (int) env_count("STRATA_MOE_PREFILL_STAGE", 2);
    const PrefillColdChoice cold = choose_prefill_cold(std::getenv("STRATA_MOE_PREFILL_COLD"));
    if (!cold.error.empty()) GGML_ABORT("strata-moe: %s", cold.error.c_str());
    p.cold = cold.cold;
    p.cold_is_default = cold.is_default;
    return p;
}

CpuKernel cpu_kernel() {
    static const CpuKernel kernel = [] {
        const char* env = std::getenv("STRATA_MOE_CPU");
        CpuKernel k = CpuKernel::kFast;
        if (env != nullptr && std::strcmp(env, "ref") == 0) {
            k = CpuKernel::kRef;
        } else if (env != nullptr && *env != '\0' && std::strcmp(env, "fast") != 0) {
            GGML_ABORT("strata-moe: STRATA_MOE_CPU=%s: expected fast (the default) or ref", env);
        }
        if (k == CpuKernel::kFast && !fast_kernel_cpu_ok()) {
            GGML_ABORT("strata-moe: STRATA_MOE_CPU=fast (the default) needs AVX2, FMA and F16C, which this CPU "
                       "lacks; set STRATA_MOE_CPU=ref for ggml-cpu's own kernels");
        }
        const bool fast = k == CpuKernel::kFast;
        std::fprintf(stderr, "strata-moe: STRATA_MOE_CPU=%s: CPU misses on %s\n", fast ? "fast" : "ref",
                     fast ? "strata-moe's AVX2 kernel" : "the reference (ggml-cpu vec_dot)");
        return k;
    }();
    return kernel;
}

// The warm tier's pinned host memory (STRATA_MOE_RAM_MODE=pinned): the driver pins the pages, so RLIMIT_MEMLOCK does
// not apply. Portable is not needed: the one device the GPU tier uses is device 0.
void* pinned_alloc(size_t bytes, std::string* error) {
    void* p = nullptr;
    const cudaError_t err = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        *error = std::string(cudaGetErrorName(err)) + " (" + cudaGetErrorString(err) + ")";
        return nullptr;
    }
    return p;
}

void pinned_free(void* p) { cudaFreeHost(p); }

// The warm tier's mode, logged once with what chose it; aborts on a mode that cannot be used.
RamMode ram_mode(bool cuda_device) {
    const char* env = std::getenv("STRATA_MOE_RAM_MODE");
    const RamModeChoice c = choose_ram_mode(env, cuda_device);
    if (!c.error.empty()) GGML_ABORT("strata-moe: %s", c.error.c_str());
    std::fprintf(stderr, "strata-moe: STRATA_MOE_RAM_MODE=%s%s\n", ram_mode_name(c.mode),
                 !c.is_default ? ""
                 : cuda_device ? " (the default with a CUDA device)"
                               : " (the default without a CUDA device)");
    return c.mode;
}

// The cold path, logged once with what chose it; aborts on an unknown value.
ColdPath cold_path() {
    const ColdPathChoice c = choose_cold_path(std::getenv("STRATA_MOE_COLD"));
    if (!c.error.empty()) GGML_ABORT("strata-moe: %s", c.error.c_str());
    std::fprintf(stderr, "strata-moe: STRATA_MOE_COLD=%s%s\n", cold_path_name(c.path),
                 c.is_default ? " (the default)" : "");
    return c.path;
}

// The tiers of the process: the hit path (null: none, the hot experts if any are misses) and the residency (null:
// the misses read the fork's tensors). Never freed: tearing down CUDA objects from a static destructor at exit can
// run after the CUDA runtime has gone, and the reader threads may still be blocked in a read.
struct Tiers {
    HitPath* hits = nullptr;
    Residency* residency = nullptr;
};

const Tiers& tiers() {
    static const Tiers tiers = [] {
        Tiers t;
        const char* hot = std::getenv("STRATA_MOE_HOT");
        const char* ram = std::getenv("STRATA_MOE_RAM_GB");
        const bool resident = ram != nullptr && *ram != '\0';
        const char* mode = std::getenv("STRATA_MOE_RAM_MODE");
        if (!resident && mode != nullptr && *mode != '\0') {
            GGML_ABORT("strata-moe: STRATA_MOE_RAM_MODE=%s needs STRATA_MOE_RAM_GB, the warm tier it holds", mode);
        }
        const char* cold = std::getenv("STRATA_MOE_COLD");
        if (!resident && cold != nullptr && *cold != '\0') {
            GGML_ABORT("strata-moe: STRATA_MOE_COLD=%s needs STRATA_MOE_RAM_GB, the residency whose cold tier it "
                       "selects", cold);
        }
        // parsed whatever the tiers, so a bad value aborts on a CPU-only run too
        const PrefillConfig prefill = prefill_config();
        if (!resident && prefill.cold == PrefillCold::kDirect) {
            GGML_ABORT("strata-moe: STRATA_MOE_PREFILL_COLD=direct needs STRATA_MOE_RAM_GB, the residency whose reader "
                       "threads read the staged experts");
        }
        const char* no_prefill = "batch calls (prefill) run on the CPU path too, as without the prefill path";
        if (hot == nullptr || *hot == '\0') {
            if (resident) {
                GGML_ABORT("strata-moe: STRATA_MOE_RAM_GB=%s needs STRATA_MOE_HOT, a hot set with warm lists "
                           "(hot_set.py --ram-budget-gib)", ram);
            }
            std::fprintf(stderr, "strata-moe: STRATA_MOE_HOT unset: every routed expert on the CPU; %s\n", no_prefill);
            return t;
        }
        const HotSet set = load_hot_set(hot);
        size_t n_vram = 0;
        for (const HotLayer& l : set.layers) n_vram += l.experts.size();
        if (n_vram == 0) {
            std::fprintf(stderr, "strata-moe: hot set %s lists no VRAM experts: no GPU tier; %s\n", hot, no_prefill);
        } else {
            const double budget = vram_budget_gib();
            if (ggml_backend_cuda_get_device_count() < 1) {
                std::fprintf(stderr, "strata-moe: STRATA_MOE_HOT=%s but no CUDA device: no GPU tier, its VRAM "
                             "experts run on the CPU; %s\n", hot, no_prefill);
            } else {
                t.hits = make_gpu_tier(set, budget, prefill).release();
                if (prefill.min_tokens == 0) {
                    std::fprintf(stderr, "strata-moe: STRATA_MOE_PREFILL_MIN=0: no prefill path, batch calls split "
                                 "into hits and misses as decode calls do\n");
                }
            }
        }
        if (resident) {
            const int64_t readers = env_count("STRATA_MOE_COLD_THREADS", 8);
            const int64_t round = env_count("STRATA_MOE_COLD_ROUND", 16);
            static const PinnedHost pinned{pinned_alloc, pinned_free};
            t.residency = new Residency(set, env_gib("STRATA_MOE_RAM_GB", ram, true), t.hits != nullptr,
                                        (int) readers, round, ::pread,
                                        ram_mode(ggml_backend_cuda_get_device_count() >= 1), &pinned, cold_path());
        }
        return t;
    }();
    return tiers;
}

}  // namespace
}  // namespace strata::dsv41

// the clamp slot must not overlap what ggml_custom_4d writes into op_params
static_assert(sizeof(ggml_custom_op_params) <= STRATA_MOE_CLAMP_PARAM * sizeof(int32_t),
              "STRATA_MOE_CLAMP_PARAM overlaps struct ggml_custom_op_params");

extern "C" STRATA_MOE_API void strata_moe_forward(ggml_tensor* dst, int ith, int nth, void* userdata) {
    using namespace strata::dsv41;
    (void) ith;
    if (userdata != nullptr) {
        GGML_ABORT("strata-moe: %s: userdata %p is set: this fork hook passes the clamp limit as a pointer ("
                   "b1e6002), which dangles once llama.cpp reuses the graph; rebuild the fork with the current "
                   "third_party/llama-dsv41 patch, which writes it in op_params", dst->name, userdata);
    }
    static std::once_flag cpu_init;
    std::call_once(cpu_init, [] {
        ggml_cpu_init();
        profiling();
        cpu_kernel();
    });
    const Tiers& t = tiers();
    forward(dst, nth, strata_moe_clamp(dst), t.hits, t.residency, cpu_kernel());
}

extern "C" STRATA_MOE_API int strata_moe_layer_counts(int layer, int64_t* hits, int64_t* misses) {
    return strata::dsv41::profile_layer_counts(layer, hits, misses) ? 1 : 0;
}

extern "C" STRATA_MOE_API int strata_moe_layer_clamp(int layer, float* min_limit, float* max_limit, int64_t* calls) {
    return strata::dsv41::profile_layer_clamp(layer, min_limit, max_limit, calls) ? 1 : 0;
}
