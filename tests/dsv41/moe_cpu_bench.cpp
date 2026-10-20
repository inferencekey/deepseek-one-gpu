// tests/dsv41/moe_cpu_bench.cpp - microbenchmark of strata-moe's CPU miss path: the fast kernel
// (STRATA_MOE_CPU=fast) and the reference (ref) on the same inputs, at DeepSeek-V4.1-Flash dims: n_embd 5120,
// n_ff 2304, gate/up q2_K, down q3_K (12,810,240 B per expert), 6 used, SwiGLU clamp 10, every pair a miss.
//
// The weights are synthetic: random bytes in every block, with the fp16 scales set to a range that keeps the
// activations finite (values do not change the speed, and the two kernels are compared bit for bit anyway). One
// layer of --experts experts (default 192, 2.46 GB) is far larger than the last-level cache, and each call routes
// every token to 6 distinct experts drawn uniformly, so the weights come from DRAM as they do in a decode step.
// The op runs as llama.cpp runs it: a ggml_custom_4d in a graph computed by ggml-cpu (a persistent threadpool, the
// graph reused, the routing rewritten before each call).
//
// Prints the machine's read bandwidth (every thread summing its share of the weight buffers with 32-byte loads, best
// of 3), then per tokens, threads and kernel: ms per call (mean over --reps calls, after 2 warm-up calls), the
// distinct experts a call reads, the effective GB/s of expert bytes, its ratio to the read bandwidth (at that thread
// count and at the best thread count), fast against ref, and whether fast gave ref's output bit for bit (exit 1 if
// not).
//
//   dsv41_moe_cpu_bench [--experts N] [--threads 1,2,4,8] [--tokens 1,32] [--reps R]
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "cpu_kernel.h"
#include "moe_op.h"
#include "strata/dsv41/moe.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

using strata::dsv41::CpuKernel;

constexpr int64_t kEmbd = 5120;
constexpr int64_t kFf = 2304;
constexpr int64_t kUsed = 6;
constexpr float kLimit = 10.0f;
constexpr int kWarmup = 2;

struct Rng {
    uint64_t s;
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * (float) (next() >> 40) * (1.0f / 16777216.0f); }
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

[[noreturn]] void usage(const char* why) {
    std::fprintf(stderr, "dsv41_moe_cpu_bench: %s\nusage: dsv41_moe_cpu_bench [--experts N] [--threads 1,2,4,8] "
                 "[--tokens 1,32] [--reps R]\n", why);
    std::exit(2);
}

std::vector<int> int_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p != '\0';) {
        char* end = nullptr;
        const long x = std::strtol(p, &end, 10);
        if (end == p || x <= 0 || (*end != ',' && *end != '\0')) {
            usage("expected a comma-separated list of positive integers");
        }
        v.push_back((int) x);
        p = *end == ',' ? end + 1 : end;
    }
    return v;
}

// One expert tensor of random blocks; every block's fp16 scales (the fields at `half_offsets`) drawn from
// [lo, hi]. Filled by `threads` threads, each its own range of blocks.
ggml_tensor* make_experts(ggml_context* ctx, ggml_type type, int64_t n_per_row, int64_t n_rows, int64_t n_expert,
                          const char* name, const std::vector<size_t>& half_offsets, float lo, float hi, uint64_t seed,
                          int threads) {
    ggml_tensor* t = ggml_new_tensor_3d(ctx, type, n_per_row, n_rows, n_expert);
    ggml_set_name(t, name);
    const size_t bytes = ggml_nbytes(t);
    t->data = std::aligned_alloc(4096, (bytes + 4095) / 4096 * 4096);
    if (t->data == nullptr) {
        std::fprintf(stderr, "dsv41_moe_cpu_bench: cannot allocate %zu bytes for %s\n", bytes, name);
        std::exit(1);
    }
    const size_t block = ggml_type_size(type), n_blocks = bytes / block;
    std::vector<std::thread> workers;
    for (int w = 0; w < threads; ++w) {
        workers.emplace_back([&, w] {
            Rng rng{seed * 7919u + (uint64_t) w};
            auto* base = static_cast<uint8_t*>(t->data);
            const size_t first = n_blocks * (size_t) w / (size_t) threads;
            const size_t last = n_blocks * (size_t) (w + 1) / (size_t) threads;
            for (size_t b = first; b < last; ++b) {
                uint8_t* p = base + b * block;
                for (size_t i = 0; i + 8 <= block; i += 8) {
                    const uint64_t r = rng.next();
                    std::memcpy(p + i, &r, 8);
                }
                for (size_t i = block / 8 * 8; i < block; ++i) p[i] = (uint8_t) rng.next();
                for (const size_t off : half_offsets) {
                    const ggml_fp16_t h = ggml_fp32_to_fp16(rng.uniform(lo, hi));
                    std::memcpy(p + off, &h, sizeof(h));
                }
            }
        });
    }
    for (std::thread& w : workers) w.join();
    return t;
}

// One thread's share of the read bandwidth test: 32-byte loads summed in four chains. AVX2 for this function only;
// main calls it after fast_kernel_cpu_ok.
__attribute__((target("avx2"))) uint64_t sum_lines(const uint8_t* p, size_t first, size_t last) {
    __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
    for (size_t l = first; l + 2 <= last; l += 2) {  // two cache lines per step
        const __m256i* q = reinterpret_cast<const __m256i*>(p + l * 64);
        s0 = _mm256_add_epi64(s0, _mm256_load_si256(q + 0));
        s1 = _mm256_add_epi64(s1, _mm256_load_si256(q + 1));
        s2 = _mm256_add_epi64(s2, _mm256_load_si256(q + 2));
        s3 = _mm256_add_epi64(s3, _mm256_load_si256(q + 3));
    }
    const __m256i s = _mm256_add_epi64(_mm256_add_epi64(s0, s1), _mm256_add_epi64(s2, s3));
    return (uint64_t) _mm256_extract_epi64(s, 0) + (uint64_t) _mm256_extract_epi64(s, 3);
}

// GB/s of `threads` threads each reading its share of the expert buffers, best of 3
double read_bandwidth(const std::vector<const ggml_tensor*>& tensors, int threads) {
    size_t total = 0;
    for (const ggml_tensor* t : tensors) total += ggml_nbytes(t) / 128 * 128;
    double best = 0.0;
    std::vector<uint64_t> sink((size_t) threads * 8);
    for (int rep = 0; rep < 3; ++rep) {
        std::vector<std::thread> workers;
        const auto t0 = std::chrono::steady_clock::now();
        for (int w = 0; w < threads; ++w) {
            workers.emplace_back([&, w] {
                uint64_t s = 0;
                for (const ggml_tensor* t : tensors) {
                    const size_t n = ggml_nbytes(t) / 128 * 2;  // whole pairs of cache lines
                    const size_t first = n / 2 * (size_t) w / (size_t) threads * 2;
                    const size_t last = n / 2 * (size_t) (w + 1) / (size_t) threads * 2;
                    s += sum_lines(static_cast<const uint8_t*>(t->data), first, last);
                }
                sink[(size_t) w * 8] = s;
            });
        }
        for (std::thread& w : workers) w.join();
        best = std::max(best, (double) total / seconds_since(t0) / 1e9);
    }
    return best;
}

CpuKernel g_kernel = CpuKernel::kRef;

void op(ggml_tensor* dst, int, int nth, void*) {
    strata::dsv41::forward(dst, nth, strata_moe_clamp(dst), nullptr, nullptr, g_kernel);
}

// The routing of one call: per token, kUsed distinct experts and their weights
struct Routing {
    std::vector<int32_t> ids;
    std::vector<float> weights;
    int64_t distinct = 0;
};

Routing make_routing(int64_t n_tok, int64_t n_expert, Rng& rng) {
    Routing r;
    std::vector<uint8_t> seen((size_t) n_expert);
    for (int64_t t = 0; t < n_tok; ++t) {
        int32_t picked[kUsed];
        for (int64_t s = 0; s < kUsed; ++s) {
            int32_t e;
            do {
                e = (int32_t) (rng.next() % (uint64_t) n_expert);
            } while (std::find(picked, picked + s, e) != picked + s);
            picked[s] = e;
            r.ids.push_back(e);
            r.weights.push_back(rng.uniform(0.05f, 0.5f));
            r.distinct += !seen[(size_t) e];
            seen[(size_t) e] = 1;
        }
    }
    return r;
}

struct Result {
    double ms = 0.0, gbps = 0.0;
    int64_t distinct = 0;  // per call, mean
    std::vector<float> last;  // the output of the last timed call
};

Result run(ggml_tensor* const* experts, int64_t n_tok, int threads, CpuKernel kernel,
           const std::vector<Routing>& routes, size_t expert_bytes) {
    ggml_init_params params = {/*.mem_size =*/ (size_t) 64 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    ggml_context* ctx = ggml_init(params);
    Rng rng{1234 + (uint64_t) n_tok};
    ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbd, n_tok);
    for (int64_t i = 0; i < kEmbd * n_tok; ++i) ggml_get_data_f32(cur)[i] = rng.uniform(-1.0f, 1.0f);
    ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kUsed, n_tok);
    ggml_tensor* weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kUsed, n_tok);
    ggml_tensor* args[] = {cur, ids, weights, experts[0], experts[1], experts[2]};
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, kEmbd, n_tok, 1, 1, args, 6, op, GGML_N_TASKS_MAX, nullptr);
    strata_moe_set_clamp(out, kLimit);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    ggml_threadpool* tp = ggml_threadpool_new(&tpp);
    ggml_cplan plan = ggml_graph_plan(gf, threads, tp);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    g_kernel = kernel;

    Result res;
    double timed = 0.0;
    int64_t bytes = 0;
    for (size_t call = 0; call < routes.size(); ++call) {
        const Routing& r = routes[call];
        std::memcpy(ids->data, r.ids.data(), r.ids.size() * sizeof(int32_t));
        std::memcpy(weights->data, r.weights.data(), r.weights.size() * sizeof(float));
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "dsv41_moe_cpu_bench: graph compute failed\n");
            std::exit(1);
        }
        const double s = seconds_since(t0);
        if (call < (size_t) kWarmup) continue;
        timed += s;
        bytes += r.distinct * (int64_t) expert_bytes;
        res.distinct += r.distinct;
    }
    const int64_t n_timed = (int64_t) routes.size() - kWarmup;
    res.ms = 1e3 * timed / (double) n_timed;
    res.gbps = (double) bytes / timed / 1e9;
    res.distinct /= n_timed;
    res.last.assign(ggml_get_data_f32(out), ggml_get_data_f32(out) + kEmbd * n_tok);
    ggml_threadpool_free(tp);
    ggml_free(ctx);
    return res;
}

}  // namespace

int main(int argc, char** argv) {
    int64_t n_expert = 192;
    std::vector<int> thread_list = {1, 2, 4, 6, 8, 10, 12};
    std::vector<int> token_list = {1, 32};
    int reps = 10;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") usage("the CPU miss-path benchmark");
        if (i + 1 >= argc) usage(("missing the value of " + a).c_str());
        if (a == "--experts") {
            n_expert = int_list(argv[++i]).at(0);
        } else if (a == "--threads") {
            thread_list = int_list(argv[++i]);
        } else if (a == "--tokens") {
            token_list = int_list(argv[++i]);
        } else if (a == "--reps") {
            reps = int_list(argv[++i]).at(0);
        } else {
            usage(("unknown argument " + a).c_str());
        }
    }
    if (n_expert < kUsed) usage("--experts must be at least 6");
    ggml_cpu_init();
    if (!strata::dsv41::fast_kernel_cpu_ok()) {
        std::fprintf(stderr, "dsv41_moe_cpu_bench: this CPU lacks AVX2, FMA or F16C, the fast kernel cannot run\n");
        return 1;
    }
    const int max_threads = *std::max_element(thread_list.begin(), thread_list.end());

    ggml_init_params params = {/*.mem_size =*/ 4 * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr,
                               /*.no_alloc =*/ true};
    ggml_context* wctx = ggml_init(params);
    const auto t0 = std::chrono::steady_clock::now();
    // fp16 ranges: gate/up pre-activations with a std of a few units against inputs in [-1, 1] (so the clamp at 10
    // is active for some), outputs of a few units
    const std::vector<size_t> q2k_halves = {offsetof(block_q2_K, data.data.d), offsetof(block_q2_K, data.data.dmin)};
    const std::vector<size_t> q3k_halves = {offsetof(block_q3_K, d)};
    const int fill_threads = std::max(max_threads, 4);
    ggml_tensor* up = make_experts(wctx, GGML_TYPE_Q2_K, kEmbd, kFf, n_expert, "blk.0.ffn_up_exps.weight", q2k_halves,
                                   0.002f, 0.01f, 1, fill_threads);
    ggml_tensor* gate = make_experts(wctx, GGML_TYPE_Q2_K, kEmbd, kFf, n_expert, "blk.0.ffn_gate_exps.weight",
                                     q2k_halves, 0.002f, 0.01f, 2, fill_threads);
    ggml_tensor* down = make_experts(wctx, GGML_TYPE_Q3_K, kFf, kEmbd, n_expert, "blk.0.ffn_down_exps.weight",
                                     q3k_halves, 0.0005f, 0.002f, 3, fill_threads);
    const size_t expert_bytes = up->nb[2] + gate->nb[2] + down->nb[2];
    const double total_gb = (double) (expert_bytes * (size_t) n_expert) / 1e9;
    std::printf("dsv41_moe_cpu_bench: n_embd %lld n_ff %lld, q2_K gate/up + q3_K down = %zu B per expert, %lld "
                "experts (%.2f GB, filled in %.1f s), %lld used, clamp %g, %d warm-up + %d timed calls per row\n",
                (long long) kEmbd, (long long) kFf, expert_bytes, (long long) n_expert, total_gb,
                seconds_since(t0), (long long) kUsed, kLimit, kWarmup, reps);

    std::map<int, double> bw;
    double best_bw = 0.0;
    for (const int th : thread_list) {
        bw[th] = read_bandwidth({up, gate, down}, th);
        best_bw = std::max(best_bw, bw[th]);
        std::printf("read bandwidth, %2d threads: %6.2f GB/s\n", th, bw[th]);
    }
    std::printf("read bandwidth of this machine (best over the thread counts): %.2f GB/s\n\n", best_bw);

    ggml_tensor* experts[] = {up, gate, down};
    int mismatched = 0;
    std::printf("%6s %7s %-4s %10s %12s %8s %10s %10s %10s %9s\n", "tokens", "threads", "cpu", "ms/call",
                "experts/call", "GB/s", "%read@thr", "%read@max", "fast/ref", "same bits");
    for (const int n_tok : token_list) {
        Rng rng{99 + (uint64_t) n_tok};
        std::vector<Routing> routes;
        for (int i = 0; i < kWarmup + reps; ++i) routes.push_back(make_routing(n_tok, n_expert, rng));
        for (const int th : thread_list) {
            const Result ref = run(experts, n_tok, th, CpuKernel::kRef, routes, expert_bytes);
            const Result fast = run(experts, n_tok, th, CpuKernel::kFast, routes, expert_bytes);
            const bool same = fast.last.size() == ref.last.size() &&
                              std::memcmp(fast.last.data(), ref.last.data(), ref.last.size() * sizeof(float)) == 0;
            mismatched += !same;
            for (const Result* r : {&ref, &fast}) {
                const bool is_fast = r == &fast;
                char speedup[16] = "";
                if (is_fast) std::snprintf(speedup, sizeof(speedup), "%.2fx", ref.ms / fast.ms);
                std::printf("%6d %7d %-4s %10.3f %12lld %8.2f %9.1f%% %9.1f%% %10s %9s\n", n_tok, th,
                            is_fast ? "fast" : "ref", r->ms, (long long) r->distinct, r->gbps,
                            100.0 * r->gbps / bw[th], 100.0 * r->gbps / best_bw, speedup,
                            is_fast ? (same ? "yes" : "NO") : "");
            }
        }
    }
    for (ggml_tensor* t : experts) std::free(t->data);
    ggml_free(wctx);
    if (mismatched > 0) {
        std::printf("dsv41_moe_cpu_bench: FAIL: %d rows where fast did not give ref's output bit for bit\n",
                    mismatched);
        return 1;
    }
    return 0;
}
