// src/dsv41/moe_prefill.h - the prefill path of strata-moe, GPU free (moe_prefill.cpp); the GPU tier
// (moe_gpu.cpp) runs it on CUDA, tests/dsv41/moe_prefill_test.cpp on a fake engine.
//
// A batch call of at least STRATA_MOE_PREFILL_MIN tokens (src/dsv41/moe.cpp) touches nearly every expert of its
// layer: a 4096-token ubatch routes 24576 (token, slot) pairs over 384 experts, 64 per expert on average. The CPU
// would read ~113 GiB per ubatch (every expert outside the VRAM slab at hot-70) at ~22 GB/s and then compute them;
// instead every pair of the call runs on the GPU. The experts in the layer's VRAM slab are computed there; the
// others are copied into a ring of device slots, n_sets sets of set_size experts, on a copy stream, while the
// compute stream runs the passes whose experts are already in. A pass is one ggml mul_mat_id graph over the slab or
// one set, every pair a column of its own (width 1, so no padding rows); the copy of a set waits on the GPU for the
// last pass that read it, and its pass waits for its copy, through events, with no host sync per expert.
//
// Where an expert's bytes come from: the warm tier when it is CUDA pinned memory (moe_residency.h), copied
// straight from there; anything else (a cold expert in the fork's mapping, a warm tier in locked or plain memory, no
// residency at all) is "staged": it goes into pinned staging first (by STRATA_MOE_PREFILL_COLD, PrefillCold below),
// the pass's copy reads it from there, and a staging set is refilled only once the copies out of it are done (a host
// wait on an event, per pass, not per expert).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::dsv41 {

// How the staged experts reach the pinned staging (STRATA_MOE_PREFILL_COLD):
//   pagecache  the op's ggml-cpu threads copy them from the fork's mapping (StageJobs below), through the page
//              cache; the default, as STRATA_MOE_COLD's (on an A100 SXM pod the page cache served hook-70's decode
//              misses in ~59 ms per pass against direct's 168 ms of waiting)
//   direct     the residency's reader threads read them from their GGUF files with O_DIRECT
//              (Residency::read_expert), never through the page cache (c5: 4.75 GB/s cold); needs the residency
//              (STRATA_MOE_RAM_GB)
enum class PrefillCold { kPageCache, kDirect };
const char* prefill_cold_name(PrefillCold c);

// The path STRATA_MOE_PREFILL_COLD=`env` selects (null or empty: pagecache, the default), or, in `error`, why it
// cannot.
struct PrefillColdChoice {
    PrefillCold cold = PrefillCold::kPageCache;
    bool is_default = false;
    std::string error;
};
PrefillColdChoice choose_prefill_cold(const char* env);

// One slot of the pinned staging, for experts of at most part_bytes[part] per part: each part starts at a
// kDirectAlign-aligned offset with direct_slot_bytes of room (moe_residency.h), so an O_DIRECT read of it at any file
// offset fits and both paths fill the same layout (pagecache copies to the part's start, direct reads land
// DirectSpan::lead bytes after it). The staging's base must be kDirectAlign-aligned too.
struct StageLayout {
    size_t part_offset[3] = {};
    size_t slot_bytes = 0;
};
StageLayout stage_layout(const size_t part_bytes[3]);

// Pairs per pass at most (the slab pass is split, a ring pass closes early): the pass's buffers on the GPU are its
// gathered input rows, y and y * w (pairs x n_embd f32, 80 MiB each at DeepSeek-V4.1's 5120 and 4096 pairs) plus up,
// gate and the activation (pairs x n_ff f32, 36 MiB each at 2304), ~350 MiB before ggml-alloc reuses any, against
// ~2 GiB for all 24576 pairs of a 4096-token ubatch in one pass.
constexpr int64_t kPrefillMaxPassPairs = 4096;

// One GPU pass: a mul_mat_id graph over the layer's VRAM slab (set -1) or one set of the ring.
struct PrefillPass {
    int set = -1;                  // the ring's slot set; -1: the slab, nothing to copy
    std::vector<int32_t> experts;  // ring: experts[k] is copied into slot k of the set
    std::vector<uint8_t> staged;   // ring, per experts[k]: 1 when its bytes go through pinned staging
    int stage = -1;                // the staging set of its staged experts; -1: none staged
    std::vector<int32_t> pairs;    // the call's pairs (token * n_used + slot) it computes: a slab pass's in pair
                                   // order, a ring pass's by expert (its experts' order), each expert's in pair
                                   // order; the dedicated engine regroups the slab passes by slot (experts.cpp)
    std::vector<int32_t> slots;    // per pair: its expert's slot, in the slab or in the set
    int64_t row0 = 0;              // pairs[i]'s output row is row0 + i
};

struct PrefillPlan {
    int n_sets = 0, set_size = 0, n_stage_sets = 0;
    std::vector<PrefillPass> passes;  // the schedule: the slab passes first, then the ring passes
    int64_t n_rows = 0;               // every pair of the call: the output rows
    int64_t ring_experts = 0, staged_experts = 0;
};

// The plan of one call: expert_of_pair [n_pairs]; slab_slot per expert id its slot in the layer's slab or -1 (null:
// no slab); staged per expert id. The slab's pairs go into passes of at most max_pass_pairs pairs; every other
// expert is copied once, in first-use order, the experts with and without staging each grouped into ring passes of
// at most set_size experts and max_pass_pairs pairs (one expert's pairs are never split), the staged passes
// spread evenly among the others so the staging copies overlap the PCIe stream. Ring pass i takes set i % n_sets,
// staged ring pass j takes staging set j % n_stage_sets. hit_row[p] is pair p's output row. Reuses plan's capacity.
// max_pass_pairs: kPrefillMaxPassPairs for the fork hook's ggml graphs (moe_gpu.cpp); the dedicated engine passes its
// own buffers' rows (experts.cpp: up to a chunk, so a ring pass of its wide sets is not cut at 4096 pairs).
void plan_prefill(const std::vector<int32_t>& expert_of_pair, int64_t n_expert, const std::vector<int32_t>* slab_slot,
                  const std::vector<uint8_t>& staged, int set_size, int n_sets, int n_stage_sets, PrefillPlan& plan,
                  std::vector<int32_t>& hit_row, int64_t max_pass_pairs = kPrefillMaxPassPairs);

// What the schedule drives: two GPU streams (copy and compute) and the host's staging, as CUDA semantics: an op
// queued on a stream runs after the ones queued before it; a wait queued on a stream holds it until the matching
// done mark queued *before the wait was queued* has run.
class PrefillEngine {
public:
    virtual ~PrefillEngine() = default;
    // host, blocking: until the copies out of staging set `stage` queued so far have run
    virtual void stage_wait(int stage) = 0;
    // host, blocking: the staged experts of pass p into its staging set
    virtual void stage(const PrefillPass& p) = 0;
    // copy stream: hold until the passes queued so far on set `set` have run
    virtual void copy_wait_compute(int set) = 0;
    // copy stream: the experts of pass p into its set (from the pinned warm tier or its staging set), then the
    // pass's done mark of the set and of its staging set
    virtual void copy(const PrefillPass& p) = 0;
    // compute stream: hold until the copy queued last into set `set` has run
    virtual void compute_wait_copy(int set) = 0;
    // compute stream: pass p's graph and its output rows, then (a ring pass) its done mark of the set
    virtual void compute(const PrefillPass& p) = 0;
};

// Queues the plan's passes on the engine, in plan order. The order of the calls is what keeps the ring safe:
//   ring pass on set s, staged through h:  [stage_wait(h) if h was used] stage(p)  [copy_wait_compute(s) if s was
//   used] copy(p)  compute_wait_copy(s)  compute(p)
// so a set's copy runs after the last pass on that set, a pass after its own copy, and staging is refilled after
// the copy that read it.
void run_prefill(const PrefillPlan& plan, PrefillEngine& engine);

// The staging copies of a prefill call (STRATA_MOE_PREFILL_COLD=pagecache), shared out over the op's ggml-cpu threads:
// the thread that runs the schedule posts each staged pass's copies as one job and takes part in it; the others take
// chunks until it closes (several threads fault the fork's mapping in at once, which one memcpy could not). Every job's
// chunks are fixed before the first is posted (`table`), and a chunk is claimed with a CAS on (job, index), so a thread
// still holding one job's count can claim nothing of the next.
struct StageChunk {
    const char* src = nullptr;
    char* dst = nullptr;
    size_t bytes = 0;
};

class StageJobs {
public:
    // Before any thread of the call uses it.
    void reset();
    // The scheduling thread, before its first run: the call's jobs, unchanged until close.
    void begin(const std::vector<std::vector<StageChunk>>* table);
    // The scheduling thread: copies the chunks of job `job` (an index of the table) with the helpers and returns once
    // all are done.
    void run(size_t job);
    // The scheduling thread, once no job follows: the helpers return.
    void close();
    // Every other thread of the call: helps with each job until close.
    void help();
    // the bytes the call's jobs copied
    uint64_t bytes() const { return bytes_.load(std::memory_order_relaxed); }

private:
    void take(uint64_t job);

    // (job + 1) << 32 | the job's next chunk; 0 before the first job
    alignas(64) std::atomic<uint64_t> next_{0};
    alignas(64) std::atomic<int64_t> done_{0};
    alignas(64) std::atomic<bool> closed_{false};
    alignas(64) std::atomic<uint64_t> bytes_{0};
    const std::vector<std::vector<StageChunk>>* table_ = nullptr;  // written before next_ first names a job
};

}  // namespace strata::dsv41
