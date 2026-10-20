// src/dsv41/moe_op.h - the internals of strata_moe_forward (include/strata/dsv41/moe.h) shared by its sources and
// tests/dsv41/moe_split_test.cpp, moe_cpu_fast_test.cpp and moe_cpu_bench.cpp.
//
// One op call splits its (token, slot) pairs by expert: a pair whose expert is resident in VRAM (the hot set of
// the layer, src/dsv41/moe_gpu.cpp) is a hit and goes to the HitPath, every other pair is a miss and the CPU
// computes it on the op's ggml-cpu threads, at the same time, with the kernel STRATA_MOE_CPU selects: the CPU
// reference (ggml-cpu's one-row vec_dot) or strata-moe's own multi-row, multi-token kernel (cpu_kernel.h), both in
// src/dsv41/moe_ref.cpp. The combine adds the slots of each token left to right as the stock graph does, taking
// y * w from the hit path or the CPU per slot, so which slots were hits does not change the rounding of the sum.
// With the residency (moe_residency.h) the CPU reads each missed expert from the warm tier or, for a cold one,
// from the fork's tensors (STRATA_MOE_COLD=pagecache) or from the staging its O_DIRECT reads land in, in rounds
// that bound the staging (direct); without it, from the fork's tensors.
// a batch call of at least STRATA_MOE_PREFILL_MIN tokens with a GPU tier is a prefill call (moe_prefill.h):
// the hit path takes every pair, streaming the experts outside its slab to the GPU, the CPU computes none and only
// combines, and the op's other threads copy the hit path's staged experts while it schedules.
#pragma once

#include "moe_prefill.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <thread>
#include <vector>

namespace strata::dsv41 {

// A weight type's ggml-cpu dot product and the activation type it takes.
struct RowFormat {
    ggml_vec_dot_t vec_dot = nullptr;
    ggml_type act_type = GGML_TYPE_COUNT;
    ggml_from_float_t act_quantize = nullptr;
    size_t act_bytes = 0;  // one quantized activation row
};

// The CPU kernel of the misses (STRATA_MOE_CPU): the reference, bit-identical to the stock graph on ggml's
// CPU backend, or strata-moe's own AVX2 kernel (cpu_kernel.h), bit-identical to the reference on an AVX2 ggml-cpu.
enum class CpuKernel { kRef, kFast };

// Destructive interference: the counters the op's threads write to stay one per cache line (64 B on Zen 3), so a
// thread spinning on one does not pull the line another thread is incrementing.
constexpr size_t kCacheLine = 64;

// The allocator of the per-call float buffers: rows start on a cache line, so two threads writing neighbouring row
// tiles of one output (tiles of 16 floats or more) never share a line.
template <class T>
struct CacheLineAllocator {
    using value_type = T;
    CacheLineAllocator() = default;
    template <class U>
    CacheLineAllocator(const CacheLineAllocator<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(kCacheLine))); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(kCacheLine)); }
    template <class U>
    bool operator==(const CacheLineAllocator<U>&) const { return true; }
    template <class U>
    bool operator!=(const CacheLineAllocator<U>&) const { return false; }
};
using AlignedFloats = std::vector<float, CacheLineAllocator<float>>;

struct alignas(kCacheLine) PaddedCounter {
    std::atomic<int64_t> v{0};
};

class SpinBarrier {
public:
    void reset(int n) {
        n_ = n;
        count_.store(0, std::memory_order_relaxed);
    }

    void wait() {
        const int gen = gen_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            count_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
            return;
        }
        while (gen_.load(std::memory_order_acquire) == gen) std::this_thread::yield();
    }

private:
    int n_ = 0;
    alignas(kCacheLine) std::atomic<int> count_{0};
    alignas(kCacheLine) std::atomic<int> gen_{0};  // the waiters spin on this line only
};

class HitPath;
class Residency;

// One evaluation of the op, shared by the nth threads that run it. Calls are pooled (moe_ref.cpp): reset keeps
// every vector's capacity, so once the calls of a process have seen their largest shape a call allocates nothing.
struct Call {
    void reset(int n_threads, CpuKernel cpu_kernel);
    // once round_end is set: room for its rounds' counters, all zero
    void prepare_rounds();

    int nth = 0;
    CpuKernel kernel = CpuKernel::kRef;
    SpinBarrier barrier;
    alignas(kCacheLine) std::atomic<int> left{0};
    PaddedCounter next_quantize, next_combine;  // the next work item of phases 0 and 4
    alignas(kCacheLine) std::atomic<bool> submitted{false};  // a thread has taken the hit path's submit
    std::atomic<bool> cpu_timed{false};

    const ggml_tensor *cur = nullptr, *ids = nullptr, *weights = nullptr;
    const ggml_tensor *up = nullptr, *gate = nullptr, *down = nullptr;
    float limit = 0.0f;  // SwiGLU clamp limit, 0: plain SwiGLU
    int layer = -1;  // from the expert tensors' names, blk.<layer>.ffn_*_exps.weight
    int64_t n_embd = 0, n_ff = 0, n_used = 0, n_tokens = 0, n_expert = 0;
    RowFormat up_fmt, gate_fmt, down_fmt;

    // misses, for the CPU: the distinct experts with a miss (first use first) and their missed pairs
    // (token * n_used + slot), the pairs of experts[k] at expert_pairs[pair_begin[k] .. pair_begin[k + 1])
    std::vector<int32_t> experts;
    std::vector<int32_t> pair_begin;
    std::vector<int32_t> expert_pairs;
    std::vector<int32_t> miss_pairs;          // every missed pair, in pair order
    std::vector<uint8_t> qx_up, qx_gate;      // [n_tokens] rows of the input in up's / gate's activation type
    AlignedFloats g, u;                       // [pair, n_ff], gate . x and up . x
    std::vector<uint8_t> qh;                  // [pair] rows of h in down's activation type
    AlignedFloats y;                          // [pair, n_embd]
    // the fast kernel's operands, per entry of expert_pairs: its input row (qx), its h row, its output rows
    std::vector<const void*> act_x, act_h;
    std::vector<float*> out_g, out_u, out_y;
    std::vector<int32_t> index_of;            // scratch of the split: per expert id, its entry in `experts`
    // the first row of experts[k]'s up, gate and down weights: the fork's tensors, the warm tier or staging
    std::vector<const char*> base_up, base_gate, base_down;

    // rounds of the CPU phases 1 to 3: round r computes experts [round_end[r - 1], round_end[r]) (from 0 for r 0).
    // One round without the residency; with it, experts[0, n_ready) are warm and the cold ones after them come in
    // rounds of the residency's round size, so the staging holds two rounds (moe_residency.cpp)
    Residency* residency = nullptr;
    int64_t n_ready = 0;
    std::vector<int64_t> round_end;
    std::unique_ptr<PaddedCounter[]> round_next;      // [round, phase 1..3]: the next work item
    std::unique_ptr<std::atomic<bool>[]> round_claimed;  // [round]: a thread has taken round_done
    size_t round_capacity = 0;
    std::unique_ptr<std::atomic<uint8_t>[]> cold_ready;  // [cold expert, part]: its bytes are in staging
    size_t cold_capacity = 0;
    alignas(kCacheLine) std::atomic<bool> cold_issued{false};  // a thread has taken the residency's issue
    alignas(kCacheLine) std::atomic<int64_t> cold_left{0};     // cold reads still held by a reader thread
    std::atomic<int64_t> cold_end_ns{0};                        // the last cold read's end, steady clock ns

    // hits, for the hit path: the tokens with at least one hit, each with hit_width slots (its hits in slot
    // order, then padding with weight 0 on resident slots the token does not use, so a token's slots are distinct
    // as mul_mat_id requires); the hit path returns y * w for each of them, [n_hit_tokens, hit_width, n_embd]
    HitPath* hits = nullptr;
    int64_t n_hits = 0, hit_width = 0;
    std::vector<int32_t> hit_tokens;
    std::vector<int32_t> hit_slots;    // [hit token, hit_width]: the expert's slot in the layer's VRAM slab
    std::vector<int32_t> hit_experts;  // the same, as expert ids
    std::vector<float> hit_weights;    // the same, the routing weight (0 for padding)
    std::vector<int32_t> hit_row;      // per pair: its row in the hit path's output, -1 for a miss
    std::vector<int32_t> expert_of_slot;  // scratch of the padding: the inverse of the slots
    std::vector<uint8_t> slot_taken;      // scratch of the padding: the slots a hit token uses
    const float* hit_yw = nullptr;     // the hit path's output, once it is waited for

    // a prefill call (HitPath::prefill took it): every pair is a hit, hit_row its row in the plan's passes
    bool prefill = false;
    PrefillPlan plan;
    std::vector<int32_t> expert_of_pair;              // [pair]
    std::vector<uint8_t> staged;                      // [expert]: its bytes go through pinned staging
    std::vector<std::vector<StageChunk>> stage_table; // the staging jobs, one per staged pass, in plan order
    StageJobs stage;
    // its profile (STRATA_MOE_PROFILE=1): bytes copied to the ring and, of those, staged by the CPU threads; the
    // copy stream's and the compute stream's busy time (CUDA events around each pass's copies and graph), the GPU
    // time from the first to the last of them, and the scheduling thread's time copying into staging and waiting
    // for a staging set to be free
    int64_t prefill_passes = 0;
    uint64_t prefill_bytes = 0, prefill_staged_bytes = 0;
    double prefill_copy_ms = 0.0, prefill_compute_ms = 0.0, prefill_gpu_ms = 0.0;
    double prefill_stage_ms = 0.0, prefill_stage_wait_ms = 0.0;

    // timing, in ms (STRATA_MOE_PROFILE=1); sync_ns: the time all threads waited at the barriers between the CPU
    // phases, summed
    std::chrono::steady_clock::time_point t0;
    double submit_ms = 0.0, gpu_ms = 0.0, cpu_ms = 0.0, wait_ms = 0.0;
    PaddedCounter sync_ns;
    // the tiers of the misses (the residency only): pairs and expert reads from the warm tier and from the cold
    // one, the bytes the cold reads moved, the time from issue to the last read's end, and the time all threads
    // waited for cold bytes, summed
    // (cold_mapped: STRATA_MOE_COLD=pagecache, the cold experts computed from the fork's mapping, so cold_bytes are
    // the experts' bytes and no read was timed)
    int64_t warm_pairs = 0, cold_pairs = 0, warm_reads = 0, cold_reads = 0;
    uint64_t warm_bytes = 0, cold_bytes = 0;
    bool cold_mapped = false;
    std::chrono::steady_clock::time_point cold_t0;
    double cold_ms = 0.0;
    PaddedCounter cold_wait_ns;
};

// Where the hits of a layer are computed: the GPU tier in libstrata-moe, a CPU stand-in in the split test.
class HitPath {
public:
    virtual ~HitPath() = default;
    // per expert of the call's layer: its slot in the layer's VRAM slab, or -1; null when none of the layer is
    // resident. Called once per call, by the thread that creates it, before the split: the GPU tier fills the
    // layer's slab from the call's host expert tensors at the layer's first call, hits or not.
    virtual const std::vector<int32_t>* slots(const Call& c) = 0;
    // per call, right after slots and by the same thread: true when the hit path takes every pair of the call
    // (a prefill call, moe_prefill.h); it then fills c.hit_row for every pair and sets c.n_hits. `residency` (null:
    // none) tells it which experts are in pinned memory.
    virtual bool prefill(Call& c, const Residency* residency) {
        (void) c;
        (void) residency;
        return false;
    }
    // called by one of the op's threads, before the CPU phases; returns once the work is queued. For a prefill call
    // the op's other threads take part in its c.stage jobs meanwhile, until it closes them.
    virtual void submit(Call& c) = 0;
    // called by the same thread after the CPU phases; blocks until the output of submit is ready and returns it
    virtual const float* wait(Call& c) = 0;
};

// Runs the op on one of the nth ggml-cpu threads that call it at once, with SwiGLU clamp `limit` (0: none), the
// misses on CPU kernel `kernel`. `hits` null: every pair on the CPU. `residency` null: the misses read the fork's
// expert tensors.
// kFast aborts, naming the tensor and its type, on weights the fast kernel has no rows for, and on a CPU without
// AVX2, FMA and F16C.
void forward(ggml_tensor* dst, int nth, float limit, HitPath* hits, Residency* residency, CpuKernel kernel);

// Profile (src/dsv41/moe_profile.cpp): every finished call counts its hits and misses per layer; with
// STRATA_MOE_PROFILE=1 it also adds the call's times, and the per-layer summary is printed at exit.
bool profiling();
void profile_record(const Call& c, double op_ms);
bool profile_layer_counts(int layer, int64_t* hits, int64_t* misses);
bool profile_layer_clamp(int layer, float* min_limit, float* max_limit, int64_t* calls);

// Activation counts (src/dsv41/moe_counts.cpp): with STRATA_MOE_COUNTS set, every finished call adds its
// routed experts per layer to the counts file.
void counts_record(const Call& c);

}  // namespace strata::dsv41
