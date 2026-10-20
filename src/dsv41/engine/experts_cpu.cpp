// src/dsv41/engine/experts_cpu.cpp - the CPU experts of include/strata/dsv41/engine/experts.hpp in two numerics
// (CpuNumerics), GPU free.
//
// Gpu (the engine's): decode's hit arithmetic, experts_cpu_mmvq.hpp's kernels
//   0  every token's input row quantized to q8_1 (quantize_q8_1_kernel's bits);
//   1  gate . x and up . x, mmvq_kernel's value at one column for each row (a tile of kRowsPerItem rows of one expert
//      for all of that expert's tokens per work item);
//   2  per output row h = expert_act.hpp's swiglu_clamp(gate, up, L), then h quantized to q8_1;
//   3  down . h, a tile of rows per item.
// Fork: strata-moe's fast path (src/dsv41/moe_ref.cpp, run(), CpuKernel::kFast) behind plain pointers
//   0  every token's input row quantized to Q8_K (fork_cpu.hpp quantize_q8_K: quantize_row_q8_K_ref, which ggml-cpu's
//      x86 quantize_row_q8_K calls; the vec_dot_type of q2_K, q3_K, q4_K and q6_K alike);
//   1  gate . x and up . x (cpu_kernel.h fast_kernel_rows: ggml-cpu's AVX2 vec_dot of each row and activation to the
//      bit);
//   2  per output row h = silu(min(gate, L)) clamp(up, -L, L), ggml_compute_forward_swiglu_clamp_f32's scalar loop
//      (gate / (1 + expf(-gate)) * up, each operation rounded: this file is built with -ffp-contract=off), then h
//      quantized to Q8_K (down's vec_dot_type);
//   3  down . h.
// Work items are handed out through an atomic counter, so which thread computes a value never changes it.
#include "strata/dsv41/engine/experts.hpp"

#include "cpu_kernel.h"
#include "experts_cpu_mmvq.hpp"
#include "fork_cpu.hpp"
#include "strata/dsv41/engine/mmvq_ref.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 experts (CPU): %s\n", msg);
    std::exit(1);
}

// moe_ref.cpp's kFastRowsPerItem: one item is a tile of 64 rows of one expert
constexpr int64_t kRowsPerItem = 64;
constexpr size_t kQ8KBytes = sizeof(fork_cpu::BlockQ8K);
static_assert(kQ8KBytes == 292, "block_q8_K is 292 bytes");

// CpuNumerics::Fork: ggml_compute_forward_swiglu_clamp_f32's scalar loop, element for element (moe_ref.cpp swiglu_clamp)
float swiglu_clamp(float g, float u, float limit) {
    g = std::min(g, limit);
    u = std::clamp(u, -limit, limit);
    return g / (1.0f + std::exp(-g)) * u;
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void atomic_max(std::atomic<int64_t>& a, int64_t v) {
    for (int64_t cur = a.load(std::memory_order_relaxed); cur < v && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed);) {
    }
}

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

}  // namespace

// The expert pool, with two changes for decode, where a token makes 4 phases x 40 layers of handoffs:
// a job is the caller's callable behind a function pointer (libstdc++'s std::function heap-allocates a lambda
// that captures more than two pointers, once per phase), and with spin_us > 0 the workers and the caller poll the
// generation and the busy count that long before they block, so a phase that follows within the spin skips the
// condition variables' wake. The handoffs are measured either way: wake from the post to the last worker's start,
// return from the last worker's end to the caller running again.
struct CpuExperts::Pool {
    explicit Pool(int n) {
        for (int i = 1; i < n; ++i) workers.emplace_back([this] { work(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(m);
            stop.store(true, std::memory_order_release);
        }
        cv.notify_all();
        for (std::thread& t : workers) t.join();
    }
    // fn(item) for item in [0, n), on the workers and the caller; returns when all are done
    template <typename F> void run(int64_t n, F&& fn) {
        if (n <= 0) return;
        using Fn = std::remove_reference_t<F>;
        job_ctx = (void*) &fn;
        job_fn = [](void* c, int64_t i) { (*static_cast<Fn*>(c))(i); };
        n_items = n;
        next.store(0, std::memory_order_relaxed);
        busy.store((int) workers.size(), std::memory_order_relaxed);
        last_start.store(0, std::memory_order_relaxed);
        last_done.store(0, std::memory_order_relaxed);
        const int64_t t_post = now_ns();
        {
            std::lock_guard<std::mutex> lock(m);
            gen.fetch_add(1, std::memory_order_release);
        }
        cv.notify_all();
        drain();
        if (workers.empty()) return;
        wait_for([this] { return busy.load(std::memory_order_acquire) == 0; }, done);
        const int64_t t_back = now_ns();
        ++times.phases;
        times.wake_ms += (double) (last_start.load(std::memory_order_relaxed) - t_post) * 1e-6;
        times.return_ms += (double) std::max<int64_t>(0, t_back - last_done.load(std::memory_order_relaxed)) * 1e-6;
    }
    void drain() {
        for (int64_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n_items;) job_fn(job_ctx, i);
    }
    // pred() true, polled for spin_us, then under the mutex on `c` (every change pred reads is notified under it)
    template <typename P> void wait_for(P pred, std::condition_variable& c) {
        const int us = spin_us.load(std::memory_order_relaxed);
        if (us > 0) {
            for (const int64_t until = now_ns() + (int64_t) us * 1000; !pred() && now_ns() < until;) cpu_relax();
            if (pred()) return;
        }
        std::unique_lock<std::mutex> lock(m);
        c.wait(lock, pred);
    }
    void work() {
        uint64_t seen = 0;
        for (;;) {
            wait_for([&] { return stop.load(std::memory_order_acquire) || gen.load(std::memory_order_acquire) != seen; }, cv);
            if (stop.load(std::memory_order_acquire)) return;
            seen = gen.load(std::memory_order_acquire);
            atomic_max(last_start, now_ns());
            drain();
            atomic_max(last_done, now_ns());
            if (busy.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(m);
                done.notify_one();
            }
        }
    }

    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable cv, done;
    void* job_ctx = nullptr;
    void (*job_fn)(void*, int64_t) = nullptr;
    int64_t n_items = 0;
    std::atomic<int64_t> next{0};
    std::atomic<int> busy{0};
    std::atomic<uint64_t> gen{0};
    std::atomic<bool> stop{false};
    std::atomic<int> spin_us{0};
    std::atomic<int64_t> last_start{0}, last_done{0};
    CpuPoolTimes times;
};

// CpuNumerics::Gpu: each token's and each output row's activations prepared once per call (mmvq_cpu::prepare_act), and
// per output row the views its kernel reads (its token's input, its h)
struct CpuExperts::GpuActs {
    std::vector<float> xf, hf;
    std::vector<mmvq_cpu::Act> x_tok, x_row, h_row;
};

CpuExperts::CpuExperts(int threads, CpuNumerics numerics) : numerics_(numerics), gpu_(std::make_unique<GpuActs>()) {
    if (threads <= 0) threads = (int) std::min(16u, std::max(1u, std::thread::hardware_concurrency()));
    threads_ = threads;
    pool_ = std::make_unique<Pool>(threads);
}

CpuExperts::~CpuExperts() = default;

void cpu_experts_check(const ExpertLayerHost& L, const DenseDims& d, CpuNumerics numerics) {
    if (!dsv41::fast_kernel_cpu_ok()) failf("the CPU experts need AVX2, FMA and F16C, which this CPU does not have");
    for (int p = 0; p < 3; ++p) {
        if (numerics == CpuNumerics::Gpu && !mmvq_cpu::supports(L.type[p]))
            failf("%s: type %d, the CPU experts in the GPU's numerics compute q2_K and q3_K", L.name[p].c_str(), L.type[p]);
        if (numerics == CpuNumerics::Fork && !dsv41::fast_kernel_supports((ggml_type) L.type[p]))
            failf("%s: type %d, the CPU kernel computes q2_K, q3_K, q4_K and q6_K", L.name[p].c_str(), L.type[p]);
    }
    if (numerics == CpuNumerics::Gpu && L.type[0] != L.type[1])
        failf("layer %d: gate %d and up %d (decode's hits read both with one decoder)", L.layer, L.type[1], L.type[0]);
    if (d.n_embd % fork_cpu::QK_K || d.n_ff % fork_cpu::QK_K)
        failf("layer %d: E %d and n_ff %d must be whole Q8_K blocks of 256", L.layer, d.n_embd, d.n_ff);
    if (!(L.limit > 0.f)) failf("layer %d: SwiGLU limit %g (the CPU experts are the clamped SwiGLU)", L.layer, L.limit);
}

void CpuExperts::set_spin_us(int us) { pool_->spin_us.store(std::max(0, us), std::memory_order_relaxed); }
CpuPoolTimes CpuExperts::pool_times() const { return pool_->times; }
void CpuExperts::reset_pool_times() { pool_->times = CpuPoolTimes{}; }

void CpuExperts::copy(const CopyChunk* chunks, size_t n) {
    pool_->run((int64_t) n, [chunks](int64_t i) { std::memcpy(chunks[i].dst, chunks[i].src, chunks[i].bytes); });
}

void CpuExperts::run(const ExpertLayerHost& L, const DenseDims& d, const float* x, int64_t T, const CpuExpertJob* jobs,
                     size_t n_jobs, float* y) {
    const int64_t E = d.n_embd, F = d.n_ff;
    const bool gpu = numerics_ == CpuNumerics::Gpu;
    // a row's quantized bytes: q8_1 blocks of 32 values (36 bytes) or Q8_K blocks of 256 (292)
    const size_t xb = gpu ? (size_t) (E / 32) * kQ81BlockBytes : (size_t) (E / fork_cpu::QK_K) * kQ8KBytes;
    const size_t hb = gpu ? (size_t) (F / 32) * kQ81BlockBytes : (size_t) (F / fork_cpu::QK_K) * kQ8KBytes;
    int64_t n_rows = 0;
    for (size_t k = 0; k < n_jobs; ++k) {
        const CpuExpertJob& j = jobs[k];
        for (int32_t t : j.tokens)
            if (t < 0 || t >= T) failf("layer %d: token %d of a call of %lld", L.layer, t, (long long) T);
        n_rows = std::max<int64_t>(n_rows, j.y0 + (int64_t) j.tokens.size());
    }
    if (n_rows == 0) return;
    qx_.resize((size_t) T * xb);
    qh_.resize((size_t) n_rows * hb);
    g_.resize((size_t) (n_rows * F));
    u_.resize((size_t) (n_rows * F));
    // the kernel's operands per output row: its token's Q8_K input, its h, its gate, up and y rows
    std::vector<const void*>& act_x = act_x_;
    std::vector<const void*>& act_h = act_h_;
    std::vector<float*>& out_g = out_g_;
    std::vector<float*>& out_u = out_u_;
    std::vector<float*>& out_y = out_y_;
    for (auto* v : {&act_x, &act_h}) v->resize((size_t) n_rows);
    for (auto* v : {&out_g, &out_u, &out_y}) v->resize((size_t) n_rows);
    for (size_t k = 0; k < n_jobs; ++k) {
        const CpuExpertJob& j = jobs[k];
        for (size_t i = 0; i < j.tokens.size(); ++i) {
            const size_t r = (size_t) j.y0 + i;
            act_x[r] = qx_.data() + (size_t) j.tokens[i] * xb;
            act_h[r] = qh_.data() + r * hb;
            out_g[r] = g_.data() + r * (size_t) F;
            out_u[r] = u_.data() + r * (size_t) F;
            out_y[r] = y + r * (size_t) E;
        }
    }

    GpuActs& ga = *gpu_;
    if (gpu) {
        ga.xf.resize((size_t) (T * mmvq_cpu::act_floats(E)));
        ga.hf.resize((size_t) (n_rows * mmvq_cpu::act_floats(F)));
        ga.x_tok.resize((size_t) T);
        ga.x_row.resize((size_t) n_rows);
        ga.h_row.resize((size_t) n_rows);
    }
    // rows [r0, r1) of part p (of n_in values a row) against the n activations of output rows y0 ..: the numerics'
    // kernel (h: down's activations, else the tokens' inputs)
    auto rows = [&](int p, const uint8_t* w, size_t row_bytes, int64_t r0, int64_t r1, int64_t n_in, int64_t y0, bool h,
                    float* const* outs, int64_t n) {
        if (gpu)
            mmvq_cpu::rows(L.type[p], w, row_bytes, r0, r1, n_in, (h ? ga.h_row : ga.x_row).data() + y0, outs, n);
        else
            dsv41::fast_kernel_rows((ggml_type) L.type[p], reinterpret_cast<const char*>(w), row_bytes, r0, r1, n_in,
                                    (h ? act_h : act_x).data() + y0, outs, n);
    };

    // 0: the inputs
    pool_->run(T, [&](int64_t t) {
        uint8_t* q = qx_.data() + (size_t) t * xb;
        if (!gpu) {
            fork_cpu::quantize_q8_K(x + t * E, E, reinterpret_cast<fork_cpu::BlockQ8K*>(q));
            return;
        }
        mmvq_cpu::quantize(x + t * E, E, q);
        ga.x_tok[(size_t) t] = mmvq_cpu::prepare_act(q, E, ga.xf.data() + (size_t) t * (size_t) mmvq_cpu::act_floats(E));
    });
    if (gpu)
        for (size_t k = 0; k < n_jobs; ++k)
            for (size_t i = 0; i < jobs[k].tokens.size(); ++i)
                ga.x_row[(size_t) jobs[k].y0 + i] = ga.x_tok[(size_t) jobs[k].tokens[i]];

    // 1: gate . x and up . x
    const size_t up_row = L.bytes[0] / (size_t) F, gate_row = L.bytes[1] / (size_t) F, down_row = L.bytes[2] / (size_t) E;
    const int64_t ff_tiles = (F + kRowsPerItem - 1) / kRowsPerItem, e_tiles = (E + kRowsPerItem - 1) / kRowsPerItem;
    pool_->run((int64_t) n_jobs * ff_tiles, [&](int64_t item) {
        const CpuExpertJob& j = jobs[(size_t) (item / ff_tiles)];
        if (j.tokens.empty()) return;
        const int64_t r0 = (item % ff_tiles) * kRowsPerItem, r1 = std::min(r0 + kRowsPerItem, F);
        const int64_t n = (int64_t) j.tokens.size();
        rows(1, j.rows.part[1], gate_row, r0, r1, E, j.y0, false, &out_g[(size_t) j.y0], n);
        rows(0, j.rows.part[0], up_row, r0, r1, E, j.y0, false, &out_u[(size_t) j.y0], n);
    });

    // 2: h, quantized for down
    pool_->run(n_rows, [&](int64_t r) {
        thread_local std::vector<float> h;
        h.resize((size_t) F);
        const float* g = out_g[(size_t) r];
        const float* u = out_u[(size_t) r];
        uint8_t* q = qh_.data() + (size_t) r * hb;
        if (!gpu) {
            for (int64_t k = 0; k < F; ++k) h[(size_t) k] = swiglu_clamp(g[k], u[k], L.limit);
            fork_cpu::quantize_q8_K(h.data(), F, reinterpret_cast<fork_cpu::BlockQ8K*>(q));
            return;
        }
        mmvq_cpu::swiglu(g, u, F, L.limit, h.data());
        mmvq_cpu::quantize(h.data(), F, q);
        ga.h_row[(size_t) r] = mmvq_cpu::prepare_act(q, F, ga.hf.data() + (size_t) r * (size_t) mmvq_cpu::act_floats(F));
    });

    // 3: y = down . h
    pool_->run((int64_t) n_jobs * e_tiles, [&](int64_t item) {
        const CpuExpertJob& j = jobs[(size_t) (item / e_tiles)];
        if (j.tokens.empty()) return;
        const int64_t r0 = (item % e_tiles) * kRowsPerItem, r1 = std::min(r0 + kRowsPerItem, E);
        rows(2, j.rows.part[2], down_row, r0, r1, F, j.y0, true, &out_y[(size_t) j.y0], (int64_t) j.tokens.size());
    });
}

size_t decode_miss_jobs(const int32_t* ids, int64_t P, int K, const int32_t* slot, int32_t* job_of,
                        std::vector<int32_t>& job_expert, CpuExpertJob* jobs, int32_t* miss, int64_t& hits) {
    size_t n = 0;
    job_expert.clear();
    hits = 0;
    for (int64_t p = 0; p < P; ++p) {
        const int32_t e = ids[p];
        if (slot[e] >= 0) {
            ++hits;
            continue;
        }
        if (job_of[e] < 0) {
            job_of[e] = (int32_t) n;
            job_expert.push_back(e);
            jobs[n++].tokens.clear();
        }
        jobs[(size_t) job_of[e]].tokens.push_back((int32_t) (p / K));
    }
    int64_t y0 = 0;
    for (size_t k = 0; k < n; ++k) {
        jobs[k].y0 = y0;
        y0 += (int64_t) jobs[k].tokens.size();
    }
    // each pair's row: its job's first row plus the pairs of that job before it, in pair order
    for (int64_t p = 0; p < P; ++p) {
        const int32_t e = ids[p];
        if (slot[e] >= 0) {
            miss[p] = -1;
            continue;
        }
        CpuExpertJob& j = jobs[(size_t) job_of[e]];
        int32_t before = 0;
        for (int64_t q = 0; q < p; ++q) before += ids[q] == e;
        miss[p] = (int32_t) (j.y0 + before);
    }
    return n;
}

void ref_expert_combine(const float* y, const float* w, int64_t T, int n_used, int E, const float* shared, float* out) {
    for (int64_t t = 0; t < T; ++t) {
        float* o = out + t * E;
        for (int s = 0; s < n_used; ++s) {
            const int64_t p = t * n_used + s;
            const float* yp = y + p * E;
            const float wp = w[p];
            for (int e = 0; e < E; ++e) {
                const float v = yp[e] * wp;
                o[e] = s == 0 ? v : o[e] + v;
            }
        }
        if (shared)
            for (int e = 0; e < E; ++e) o[e] = o[e] + shared[t * E + e];
    }
}

}  // namespace strata::dsv41::engine
