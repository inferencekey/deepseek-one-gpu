// src/dsv41/moe_prefill.cpp - the plan, the schedule and the staging jobs of the prefill path (moe_prefill.h).
#include "moe_prefill.h"

#include "moe_residency.h"

#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <thread>

namespace strata::dsv41 {
namespace {

// An expert's pairs, for grouping: its id and where its pairs start in the call's by-expert order.
struct ExpertPairs {
    int32_t expert = -1;
    int32_t begin = 0, count = 0;
};

// Groups `experts` (first-use order) into ring passes of at most set_size experts and max_pairs pairs; an expert with
// more pairs than that gets a pass of its own.
void group_ring(const std::vector<ExpertPairs>& experts, int set_size, int64_t max_pairs,
                std::vector<std::vector<ExpertPairs>>& out) {
    out.clear();
    int64_t pairs = 0;
    for (const ExpertPairs& e : experts) {
        if (out.empty() || (int) out.back().size() == set_size || pairs + e.count > max_pairs) {
            out.emplace_back();
            pairs = 0;
        }
        out.back().push_back(e);
        pairs += e.count;
    }
}

PrefillPass& next_pass(PrefillPlan& plan, size_t& used) {
    if (used == plan.passes.size()) plan.passes.emplace_back();
    PrefillPass& p = plan.passes[used++];
    p.set = -1;
    p.stage = -1;
    p.experts.clear();
    p.staged.clear();
    p.pairs.clear();
    p.slots.clear();
    p.row0 = 0;
    return p;
}

}  // namespace

const char* prefill_cold_name(PrefillCold c) {
    switch (c) {
        case PrefillCold::kPageCache: return "pagecache";
        case PrefillCold::kDirect: return "direct";
    }
    return "?";
}

PrefillColdChoice choose_prefill_cold(const char* env) {
    PrefillColdChoice c;
    if (env == nullptr || *env == '\0') {
        c.is_default = true;
    } else if (std::strcmp(env, "pagecache") == 0) {
        c.cold = PrefillCold::kPageCache;
    } else if (std::strcmp(env, "direct") == 0) {
        c.cold = PrefillCold::kDirect;
    } else {
        c.error = std::string("STRATA_MOE_PREFILL_COLD=") + env + ": expected pagecache (the default) or direct";
    }
    return c;
}

StageLayout stage_layout(const size_t part_bytes[3]) {
    StageLayout l;
    for (int p = 0; p < 3; ++p) {
        l.part_offset[p] = l.slot_bytes;
        l.slot_bytes += direct_slot_bytes(part_bytes[p]);
    }
    return l;
}

void plan_prefill(const std::vector<int32_t>& expert_of_pair, int64_t n_expert, const std::vector<int32_t>* slab_slot,
                  const std::vector<uint8_t>& staged, int set_size, int n_sets, int n_stage_sets, PrefillPlan& plan,
                  std::vector<int32_t>& hit_row, int64_t max_pass_pairs) {
    if (set_size < 1 || n_sets < 1 || n_stage_sets < 1 || max_pass_pairs < 1 || (int64_t) staged.size() != n_expert ||
        (slab_slot != nullptr && (int64_t) slab_slot->size() != n_expert)) {
        GGML_ABORT("strata-moe: prefill plan: %d sets of %d experts, %d staging sets, passes of %lld pairs, %zu staged "
                   "flags and %zu slab slots for %lld experts", n_sets, set_size, n_stage_sets, (long long) max_pass_pairs,
                   staged.size(), slab_slot != nullptr ? slab_slot->size() : (size_t) 0, (long long) n_expert);
    }
    plan.n_sets = n_sets;
    plan.set_size = set_size;
    plan.n_stage_sets = n_stage_sets;
    plan.ring_experts = plan.staged_experts = 0;
    const int64_t n_pairs = (int64_t) expert_of_pair.size();
    hit_row.assign((size_t) n_pairs, -1);
    size_t used = 0;
    auto in_slab = [slab_slot](int32_t e) { return slab_slot != nullptr && (*slab_slot)[(size_t) e] >= 0; };

    // the slab's pairs in pair order, in passes of at most max_pass_pairs
    PrefillPass* slab = nullptr;
    for (int64_t p = 0; p < n_pairs; ++p) {
        const int32_t e = expert_of_pair[(size_t) p];
        if (e < 0 || e >= n_expert) {
            GGML_ABORT("strata-moe: prefill plan: pair %lld is expert %d, outside [0, %lld)", (long long) p, e,
                       (long long) n_expert);
        }
        if (!in_slab(e)) continue;
        if (slab == nullptr || (int64_t) slab->pairs.size() == max_pass_pairs) slab = &next_pass(plan, used);
        slab->pairs.push_back((int32_t) p);
        slab->slots.push_back((*slab_slot)[(size_t) e]);
    }

    // every other expert's pairs, by expert in first-use order: counts, offsets, then the pairs
    std::vector<int32_t> index_of((size_t) n_expert, -1);
    std::vector<ExpertPairs> ring;
    for (int64_t p = 0; p < n_pairs; ++p) {
        const int32_t e = expert_of_pair[(size_t) p];
        if (in_slab(e)) continue;
        if (index_of[(size_t) e] < 0) {
            index_of[(size_t) e] = (int32_t) ring.size();
            ring.push_back({e, 0, 0});
        }
        ++ring[(size_t) index_of[(size_t) e]].count;
    }
    int32_t offset = 0;
    for (ExpertPairs& e : ring) {
        e.begin = offset;
        offset += e.count;
    }
    std::vector<int32_t> by_expert((size_t) offset);
    std::vector<int32_t> cursor((size_t) ring.size());
    for (size_t k = 0; k < ring.size(); ++k) cursor[k] = ring[k].begin;
    for (int64_t p = 0; p < n_pairs; ++p) {
        const int32_t e = expert_of_pair[(size_t) p];
        if (!in_slab(e)) by_expert[(size_t) cursor[(size_t) index_of[(size_t) e]]++] = (int32_t) p;
    }

    std::vector<ExpertPairs> direct, through_staging;
    for (const ExpertPairs& e : ring) (staged[(size_t) e.expert] ? through_staging : direct).push_back(e);
    std::vector<std::vector<ExpertPairs>> direct_passes, staged_passes;
    group_ring(direct, set_size, max_pass_pairs, direct_passes);
    group_ring(through_staging, set_size, max_pass_pairs, staged_passes);

    // the staged passes spread evenly: after i + 1 ring passes, round((i + 1/2) * C / total) of them are staged
    const size_t n_direct = direct_passes.size(), n_staged = staged_passes.size(), total = n_direct + n_staged;
    size_t taken_direct = 0, taken_staged = 0;
    for (size_t i = 0; i < total; ++i) {
        const size_t due = ((2 * i + 1) * n_staged + total) / (2 * total);
        const bool take_staged = taken_direct == n_direct || (taken_staged < n_staged && taken_staged < due);
        const std::vector<ExpertPairs>& group = take_staged ? staged_passes[taken_staged] : direct_passes[taken_direct];
        PrefillPass& pass = next_pass(plan, used);
        pass.set = (int) (i % (size_t) n_sets);
        if (take_staged) pass.stage = (int) (taken_staged % (size_t) n_stage_sets);
        for (size_t k = 0; k < group.size(); ++k) {
            const ExpertPairs& e = group[k];
            pass.experts.push_back(e.expert);
            pass.staged.push_back(take_staged ? 1 : 0);
            for (int32_t j = e.begin; j < e.begin + e.count; ++j) {
                pass.pairs.push_back(by_expert[(size_t) j]);
                pass.slots.push_back((int32_t) k);
            }
        }
        plan.ring_experts += (int64_t) group.size();
        if (take_staged) plan.staged_experts += (int64_t) group.size();
        ++(take_staged ? taken_staged : taken_direct);
    }

    plan.passes.resize(used);
    int64_t row = 0;
    for (PrefillPass& pass : plan.passes) {
        pass.row0 = row;
        for (const int32_t p : pass.pairs) hit_row[(size_t) p] = (int32_t) row++;
    }
    plan.n_rows = row;
}

void run_prefill(const PrefillPlan& plan, PrefillEngine& engine) {
    std::vector<uint8_t> set_used((size_t) plan.n_sets, 0), stage_used((size_t) plan.n_stage_sets, 0);
    for (const PrefillPass& p : plan.passes) {
        if (p.set < 0) {
            engine.compute(p);
            continue;
        }
        if (p.stage >= 0) {
            if (stage_used[(size_t) p.stage]) engine.stage_wait(p.stage);
            engine.stage(p);
            stage_used[(size_t) p.stage] = 1;
        }
        if (set_used[(size_t) p.set]) engine.copy_wait_compute(p.set);
        engine.copy(p);
        engine.compute_wait_copy(p.set);
        engine.compute(p);
        set_used[(size_t) p.set] = 1;
    }
}

void StageJobs::reset() {
    next_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    closed_.store(false, std::memory_order_relaxed);
    bytes_.store(0, std::memory_order_relaxed);
    table_ = nullptr;
}

void StageJobs::begin(const std::vector<std::vector<StageChunk>>* table) { table_ = table; }

// Claims and copies chunks of `job` while it is the posted one and has chunks left.
void StageJobs::take(uint64_t job) {
    const std::vector<StageChunk>& chunks = (*table_)[(size_t) job];
    for (uint64_t v = next_.load(std::memory_order_acquire);;) {
        if ((v >> 32) != job + 1 || (v & 0xffffffffu) >= chunks.size()) return;
        if (!next_.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel, std::memory_order_acquire)) continue;
        const StageChunk& c = chunks[(size_t) (v & 0xffffffffu)];
        std::memcpy(c.dst, c.src, c.bytes);
        bytes_.fetch_add(c.bytes, std::memory_order_relaxed);
        done_.fetch_add(1, std::memory_order_release);
        v = next_.load(std::memory_order_acquire);
    }
}

void StageJobs::run(size_t job) {
    const std::vector<StageChunk>& chunks = (*table_)[job];
    if (chunks.size() >= (1ull << 32)) GGML_ABORT("strata-moe: prefill staging job of %zu chunks", chunks.size());
    // the previous job is done (done_ reached its size), so no thread can still add to done_
    done_.store(0, std::memory_order_relaxed);
    next_.store((uint64_t) (job + 1) << 32, std::memory_order_release);
    take(job);
    while (done_.load(std::memory_order_acquire) < (int64_t) chunks.size()) std::this_thread::yield();
}

void StageJobs::close() { closed_.store(true, std::memory_order_release); }

void StageJobs::help() {
    while (!closed_.load(std::memory_order_acquire)) {
        const uint64_t v = next_.load(std::memory_order_acquire);
        if ((v >> 32) != 0 && (v & 0xffffffffu) < (*table_)[(size_t) (v >> 32) - 1].size()) {
            take((v >> 32) - 1);
        } else {
            std::this_thread::yield();
        }
    }
}

}  // namespace strata::dsv41
