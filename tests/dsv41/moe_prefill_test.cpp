// tests/dsv41/moe_prefill_test.cpp - the prefill path of strata-moe (moe_prefill.h), GPU free.
//
// 1. plan: random routings at DeepSeek-V4.1's 384 experts, 6 used, 64 to 4096 tokens, with slabs and staged sets of
//    several sizes: every pair has one output row, rows are the passes' in order, a pass's slots name its experts,
//    every expert outside the slab is in exactly one ring pass (copied once per ubatch) and slab experts in none,
//    the ring and staging sets go round in turn, and the staged passes are spread among the others.
// 2. schedule: run_prefill on a fake engine that keeps the copy and compute streams as queues with CUDA's event
//    semantics (a wait holds its stream until the done mark queued before it has run) and runs them in a random
//    interleaving after every call, each run with its own GPU speed and bias between the streams: no ring slot is
//    overwritten before the last pass that reads it has run, every pass finds its experts in its set, staging is
//    never refilled while a queued copy still reads it, passes run in plan order, each expert is copied once, and
//    the host waits at most once per staged pass. The same checker run on a schedule without the ring's wait, and
//    on one without the staging wait, must catch each (so the checks can fail).
// 3. staging jobs: StageJobs over 8 threads, random chunk lists: each job's bytes are all in when run returns.
// 4. a prefill call through the op: a CPU engine in place of the GPU tier (plan, staging, a host "ring" filled by the
//    copies, each pass's rows computed from the ring's bytes by a one-slot call of the CPU reference), every output
//    bit equal to the all-CPU output, with the staging filled both ways (STRATA_MOE_PREFILL_COLD): pagecache, the
//    op's threads copying from the mapped tensors, and direct, the residency's reader threads reading the experts'
//    file with O_DIRECT into the shared StageLayout (each staged expert read once, no StageJobs copy); calls under
//    STRATA_MOE_PREFILL_MIN stay off the path. choose_prefill_cold's default and error. The expert tensors are mapped
//    from a file in the working directory (ctest: the build tree), since O_DIRECT needs a real filesystem.
#include "cpu_kernel.h"
#include "moe_op.h"
#include "moe_prefill.h"
#include "moe_residency.h"
#include "strata/dsv41/moe.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace strata::dsv41;

struct Rng {
    uint64_t s;
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint64_t below(uint64_t n) { return next() % n; }
    float uniform(float lo, float hi) { return lo + (hi - lo) * (float) (next() >> 40) * (1.0f / 16777216.0f); }
};

int g_failed = 0, g_cases = 0;

void report(bool pass, const std::string& what) {
    g_failed += !pass;
    ++g_cases;
    std::printf("%-4s %s\n", pass ? "ok" : "FAIL", what.c_str());
}

// n_used distinct experts per token among `pool`, half the picks from its first eighth (a skew like the profiles')
std::vector<int32_t> routing(Rng& rng, int64_t n_tok, int64_t n_used, const std::vector<int32_t>& pool) {
    std::vector<int32_t> out;
    const size_t hot = std::max<size_t>(pool.size() / 8, (size_t) n_used);
    for (int64_t t = 0; t < n_tok; ++t) {
        std::set<int32_t> picked;
        while ((int64_t) picked.size() < n_used) {
            const size_t range = rng.below(2) ? hot : pool.size();
            picked.insert(pool[(size_t) rng.below(range)]);
        }
        std::vector<int32_t> v(picked.begin(), picked.end());
        for (size_t i = v.size() - 1; i > 0; --i) std::swap(v[i], v[(size_t) rng.below(i + 1)]);
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

// ---------------------------------------------------------------- 1. the plan
std::string check_plan(const std::vector<int32_t>& eop, int64_t n_expert, const std::vector<int32_t>* slab,
                       const std::vector<uint8_t>& staged, int set_size, int n_sets, int n_stage_sets,
                       const PrefillPlan& plan, const std::vector<int32_t>& hit_row, size_t* max_direct_run,
                       int64_t cap = kPrefillMaxPassPairs) {
    const int64_t n_pairs = (int64_t) eop.size();
    if (plan.n_rows != n_pairs) return "n_rows " + std::to_string(plan.n_rows) + " != pairs " + std::to_string(n_pairs);
    std::vector<int64_t> row_seen((size_t) n_pairs, 0);
    for (int64_t p = 0; p < n_pairs; ++p) {
        if (hit_row[(size_t) p] < 0 || hit_row[(size_t) p] >= n_pairs) return "pair without a row";
        if (row_seen[(size_t) hit_row[(size_t) p]]++) return "two pairs share a row";
    }
    std::vector<int> ring_passes_of((size_t) n_expert, 0);
    int64_t row = 0, ring_index = 0, staged_index = 0, ring_experts = 0, staged_experts = 0;
    bool ring_started = false;
    size_t run = 0;
    *max_direct_run = 0;
    for (const PrefillPass& pass : plan.passes) {
        if (pass.row0 != row) return "pass rows out of order";
        if (pass.pairs.size() != pass.slots.size() || pass.pairs.empty()) return "pass pairs and slots";
        for (size_t i = 0; i < pass.pairs.size(); ++i) {
            if (hit_row[(size_t) pass.pairs[i]] != row + (int64_t) i) return "a pair's row is not its pass place";
        }
        row += (int64_t) pass.pairs.size();
        if (pass.set < 0) {
            if (ring_started) return "a slab pass after a ring pass";
            if ((int64_t) pass.pairs.size() > cap) return "slab pass over the pair cap";
            for (size_t i = 0; i < pass.pairs.size(); ++i) {
                const int32_t e = eop[(size_t) pass.pairs[i]];
                if (slab == nullptr || (*slab)[(size_t) e] < 0 || pass.slots[i] != (*slab)[(size_t) e]) {
                    return "slab pass slot is not the expert's slab slot";
                }
            }
            continue;
        }
        ring_started = true;
        if (pass.experts.empty() || (int) pass.experts.size() > set_size) return "ring pass expert count";
        if (pass.staged.size() != pass.experts.size()) return "staged flags";
        if (pass.set != (int) (ring_index % n_sets)) return "ring sets not in turn";
        const bool st = pass.stage >= 0;
        if (st && pass.stage != (int) (staged_index % n_stage_sets)) return "staging sets not in turn";
        std::set<int32_t> distinct(pass.experts.begin(), pass.experts.end());
        if (distinct.size() != pass.experts.size()) return "an expert twice in one pass";
        for (size_t k = 0; k < pass.experts.size(); ++k) {
            const int32_t e = pass.experts[k];
            if (slab != nullptr && (*slab)[(size_t) e] >= 0) return "a slab expert in a ring pass";
            if ((bool) pass.staged[k] != (bool) staged[(size_t) e] || (bool) pass.staged[k] != st) {
                return "staged flag differs from the expert's";
            }
            ++ring_passes_of[(size_t) e];
        }
        if (pass.experts.size() > 1 && (int64_t) pass.pairs.size() > cap) {
            return "ring pass over the cap";
        }
        for (size_t i = 0; i < pass.pairs.size(); ++i) {
            if (pass.slots[i] < 0 || (size_t) pass.slots[i] >= pass.experts.size() ||
                pass.experts[(size_t) pass.slots[i]] != eop[(size_t) pass.pairs[i]]) {
                return "ring pass slot does not name the pair's expert";
            }
        }
        ring_experts += (int64_t) pass.experts.size();
        if (st) {
            staged_experts += (int64_t) pass.experts.size();
            ++staged_index;
            run = 0;
        } else {
            *max_direct_run = std::max(*max_direct_run, ++run);
        }
        ++ring_index;
    }
    if (row != n_pairs) return "rows do not cover the pairs";
    std::vector<uint8_t> used((size_t) n_expert, 0);
    for (const int32_t e : eop) used[(size_t) e] = 1;
    for (int64_t e = 0; e < n_expert; ++e) {
        const bool in_slab = slab != nullptr && (*slab)[(size_t) e] >= 0;
        const int want = used[(size_t) e] && !in_slab ? 1 : 0;
        if (ring_passes_of[(size_t) e] != want) {
            return "expert " + std::to_string(e) + " in " + std::to_string(ring_passes_of[(size_t) e]) +
                   " ring passes, expected " + std::to_string(want);
        }
    }
    if (ring_experts != plan.ring_experts || staged_experts != plan.staged_experts) return "expert counts";
    return "";
}

void test_plan() {
    Rng rng{101};
    std::vector<int32_t> all(384);
    for (int32_t e = 0; e < 384; ++e) all[(size_t) e] = e;
    PrefillPlan plan;  // reused across cases, as a pooled call reuses it
    std::vector<int32_t> hit_row;
    for (const int64_t n_tok : {64, 512, 4096}) {
        for (const int n_slab : {0, 1, 147, 384}) {
            for (const double staged_share : {0.0, 0.1, 1.0}) {
                // set size, sets, staging sets, the pass cap (the dedicated engine's: up to its chunk's pairs)
                for (const auto& ring : {std::vector<int>{16, 3, 2, (int) kPrefillMaxPassPairs},
                                         {1, 1, 1, (int) kPrefillMaxPassPairs}, {5, 2, 3, (int) kPrefillMaxPassPairs},
                                         {48, 3, 2, 16384}}) {
                    const std::vector<int32_t> eop = routing(rng, n_tok, 6, all);
                    std::vector<int32_t> slab(384, -1);
                    std::vector<int32_t> order = all;
                    for (size_t i = order.size() - 1; i > 0; --i) std::swap(order[i], order[(size_t) rng.below(i + 1)]);
                    for (int k = 0; k < n_slab; ++k) slab[(size_t) order[(size_t) k]] = n_slab - 1 - k;
                    std::vector<uint8_t> staged(384);
                    for (uint8_t& s : staged) s = rng.uniform(0.0f, 1.0f) < staged_share;
                    plan_prefill(eop, 384, n_slab > 0 ? &slab : nullptr, staged, ring[0], ring[1], ring[2], plan,
                                 hit_row, ring[3]);
                    size_t max_run = 0;
                    std::string err = check_plan(eop, 384, n_slab > 0 ? &slab : nullptr, staged, ring[0], ring[1],
                                                 ring[2], plan, hit_row, &max_run, ring[3]);
                    // spread: no longer run of unstaged passes than the even share, rounded up, plus one
                    int64_t n_ring = 0, n_staged = 0;
                    for (const PrefillPass& p : plan.passes) {
                        n_ring += p.set >= 0;
                        n_staged += p.stage >= 0;
                    }
                    if (err.empty() && n_staged > 0 &&
                        (int64_t) max_run > (n_ring - n_staged + n_staged - 1) / n_staged + 1) {
                        err = "staged passes not spread: a run of " + std::to_string(max_run) + " unstaged passes";
                    }
                    char what[256];
                    std::snprintf(what, sizeof(what),
                                  "plan: %4lld tokens, slab %3d experts, %3.0f%% staged, ring %2d x %d, staging %d, "
                                  "cap %5d: %3zu passes, %3lld experts copied (%3lld staged), %5lld rows%s%s",
                                  (long long) n_tok, n_slab, staged_share * 100, ring[0], ring[1], ring[2], ring[3],
                                  plan.passes.size(), (long long) plan.ring_experts, (long long) plan.staged_experts,
                                  (long long) plan.n_rows, err.empty() ? "" : ": ", err.c_str());
                    report(err.empty(), what);
                }
            }
        }
    }
    // the dedicated engine's wide ring: at a 16384-token chunk a ring pass of 48 experts holds thousands of
    // pairs; with either cap a pass closes only when its set is full or the next expert would take it over the cap
    {
        const std::vector<int32_t> eop = routing(rng, 16384, 6, all);
        const std::vector<uint8_t> staged(384, 0);
        std::vector<int64_t> count(384, 0);
        for (int32_t e : eop) ++count[(size_t) e];
        for (const int64_t cap : {kPrefillMaxPassPairs, (int64_t) 16384}) {
            plan_prefill(eop, 384, nullptr, staged, 48, 3, 2, plan, hit_row, cap);
            size_t max_run = 0, widest = 0;
            std::string err = check_plan(eop, 384, nullptr, staged, 48, 3, 2, plan, hit_row, &max_run, cap);
            for (size_t i = 0; i < plan.passes.size(); ++i) {
                const PrefillPass& p = plan.passes[i];
                widest = std::max(widest, p.experts.size());
                if (err.empty() && i + 1 < plan.passes.size() && p.experts.size() < 48 &&
                    (int64_t) p.pairs.size() + count[(size_t) plan.passes[i + 1].experts[0]] <= cap)
                    err = "pass " + std::to_string(i) + " closed with room for the next expert";
            }
            char what[200];
            std::snprintf(what, sizeof(what), "plan: 16384 tokens, ring 48 x 3, cap %5lld: %3zu passes of at most %2zu "
                          "experts%s%s", (long long) cap, plan.passes.size(), widest, err.empty() ? "" : ": ", err.c_str());
            report(err.empty(), what);
        }
    }
}

// ---------------------------------------------------------------- 2. the schedule on a fake two-stream engine
class FakeEngine final : public PrefillEngine {
public:
    FakeEngine(const PrefillPlan& plan, int64_t n_expert, uint64_t seed)
        : plan_(plan), rng_{seed}, ring_((size_t) plan.n_sets * plan.set_size, -1),
          ring_owner_(ring_.size(), -1), stage_((size_t) plan.n_stage_sets * plan.set_size, -1),
          copied_q_((size_t) plan.n_sets), copied_done_((size_t) plan.n_sets), consumed_q_((size_t) plan.n_sets),
          consumed_done_((size_t) plan.n_sets), free_q_((size_t) plan.n_stage_sets),
          free_done_((size_t) plan.n_stage_sets), readers_((size_t) plan.n_stage_sets),
          ran_(plan.passes.size(), 0), copies_((size_t) n_expert, 0) {}

    std::string error;
    int64_t host_waits = 0;

    void stage_wait(int h) override {
        ++host_waits;
        while (free_done_[(size_t) h] < free_q_[(size_t) h]) {
            if (!step_any()) return fail("stage_wait would never return (deadlock)");
        }
        progress();
    }
    void stage(const PrefillPass& p) override {
        if (readers_[(size_t) p.stage] > 0) fail("staging set refilled while a queued copy still reads it");
        for (size_t k = 0; k < p.experts.size(); ++k) stage_[(size_t) p.stage * plan_.set_size + k] = p.experts[k];
        progress();
    }
    void copy_wait_compute(int s) override {
        copy_q_.push_back({Op::kWait, s, consumed_q_[(size_t) s], nullptr});
        progress();
    }
    void copy(const PrefillPass& p) override {
        copy_q_.push_back({Op::kRun, p.set, 0, &p});
        ++copied_q_[(size_t) p.set];
        if (p.stage >= 0) {
            ++free_q_[(size_t) p.stage];
            ++readers_[(size_t) p.stage];
        }
        progress();
    }
    void compute_wait_copy(int s) override {
        compute_q_.push_back({Op::kWait, s, copied_q_[(size_t) s], nullptr});
        progress();
    }
    void compute(const PrefillPass& p) override {
        compute_q_.push_back({Op::kRun, p.set, 0, &p});
        if (p.set >= 0) ++consumed_q_[(size_t) p.set];
        progress();
    }

    // runs both streams to the end; the checks that need the whole run
    void finish() {
        while (step_any()) {}
        if (!copy_q_.empty() || !compute_q_.empty()) fail("the streams stopped with work queued (deadlock)");
        for (size_t i = 0; i < ran_.size(); ++i) {
            if (!ran_[i]) fail("pass " + std::to_string(i) + " never ran");
        }
        for (const PrefillPass& p : plan_.passes) {
            for (const int32_t e : p.experts) {
                if (copies_[(size_t) e] != 1) fail("expert " + std::to_string(e) + " copied " +
                                                   std::to_string(copies_[(size_t) e]) + " times");
            }
        }
    }

private:
    struct Op {
        enum Kind { kWait, kRun } kind;
        int set;
        int64_t target;  // kWait: the done count it waits for
        const PrefillPass* pass;
    };

    void fail(const std::string& e) {
        if (error.empty()) error = e;
    }
    size_t index(const PrefillPass* p) const { return (size_t) (p - plan_.passes.data()); }

    bool step_copy() {
        if (copy_q_.empty()) return false;
        const Op op = copy_q_.front();
        if (op.kind == Op::kWait) {
            if (consumed_done_[(size_t) op.set] < op.target) return false;
            copy_q_.pop_front();
            return true;
        }
        const PrefillPass& p = *op.pass;
        for (size_t k = 0; k < p.experts.size(); ++k) {
            const size_t slot = (size_t) p.set * plan_.set_size + k;
            if (ring_owner_[slot] >= 0 && !ran_[(size_t) ring_owner_[slot]]) {
                fail("ring set " + std::to_string(p.set) + " slot " + std::to_string(k) + " overwritten by pass " +
                     std::to_string(index(&p)) + " before pass " + std::to_string(ring_owner_[slot]) + " read it");
            }
            if (p.staged[k] && stage_[(size_t) p.stage * plan_.set_size + k] != p.experts[k]) {
                fail("pass " + std::to_string(index(&p)) + " copied staging that does not hold its expert");
            }
            ring_[slot] = p.experts[k];
            ring_owner_[slot] = (int64_t) index(&p);
            ++copies_[(size_t) p.experts[k]];
        }
        ++copied_done_[(size_t) p.set];
        if (p.stage >= 0) {
            ++free_done_[(size_t) p.stage];
            --readers_[(size_t) p.stage];
        }
        copy_q_.pop_front();
        return true;
    }

    bool step_compute() {
        if (compute_q_.empty()) return false;
        const Op op = compute_q_.front();
        if (op.kind == Op::kWait) {
            if (copied_done_[(size_t) op.set] < op.target) return false;
            compute_q_.pop_front();
            return true;
        }
        const PrefillPass& p = *op.pass;
        const size_t i = index(&p);
        if (i != next_pass_) fail("pass " + std::to_string(i) + " ran before pass " + std::to_string(next_pass_));
        next_pass_ = i + 1;
        for (size_t k = 0; p.set >= 0 && k < p.experts.size(); ++k) {
            if (ring_[(size_t) p.set * plan_.set_size + k] != p.experts[k]) {
                fail("pass " + std::to_string(i) + " ran before its expert " + std::to_string(p.experts[k]) +
                     " was in set " + std::to_string(p.set));
            }
        }
        ran_[i] = 1;
        if (p.set >= 0) ++consumed_done_[(size_t) p.set];
        compute_q_.pop_front();
        return true;
    }

    // which stream goes first, with this run's own bias: some runs let the copies race ahead of the passes, some
    // hold them back
    bool step_any() {
        if (rng_.uniform(0.0f, 1.0f) < copy_bias_) return step_copy() || step_compute();
        return step_compute() || step_copy();
    }

    // GPU progress after each call: a geometric number of steps with this run's mean, so a slow GPU (mean 0.2) lets
    // the host queue many passes ahead and a fast one (4) keeps the streams nearly empty
    void progress() {
        while (rng_.uniform(0.0f, 1.0f) < rate_ / (1.0f + rate_)) {
            if (!step_any()) break;
        }
    }

    const PrefillPlan& plan_;
    Rng rng_;
    float copy_bias_ = rng_.uniform(0.05f, 0.95f);
    float rate_ = rng_.uniform(0.2f, 4.0f);
    std::deque<Op> copy_q_, compute_q_;
    std::vector<int32_t> ring_;       // per slot: the expert in it
    std::vector<int64_t> ring_owner_; // per slot: the pass whose copy wrote it
    std::vector<int32_t> stage_;      // per staging slot: the expert in it
    std::vector<int64_t> copied_q_, copied_done_, consumed_q_, consumed_done_, free_q_, free_done_, readers_;
    std::vector<uint8_t> ran_;
    std::vector<int64_t> copies_;
    size_t next_pass_ = 0;
};

// run_prefill with the ring's wait or the staging wait left out: the checker must catch each in some interleaving
void broken_schedule(const PrefillPlan& plan, PrefillEngine& e, bool ring_wait, bool stage_wait) {
    std::vector<uint8_t> set_used((size_t) plan.n_sets, 0), stage_used((size_t) plan.n_stage_sets, 0);
    for (const PrefillPass& p : plan.passes) {
        if (p.set < 0) {
            e.compute(p);
            continue;
        }
        if (p.stage >= 0) {
            if (stage_used[(size_t) p.stage] && stage_wait) e.stage_wait(p.stage);
            e.stage(p);
            stage_used[(size_t) p.stage] = 1;
        }
        if (set_used[(size_t) p.set] && ring_wait) e.copy_wait_compute(p.set);
        set_used[(size_t) p.set] = 1;
        e.copy(p);
        e.compute_wait_copy(p.set);
        e.compute(p);
    }
}

// whether the checker catches the broken schedule in one of `runs` interleavings
bool caught(const PrefillPlan& plan, Rng& rng, bool ring_wait, bool stage_wait, int runs) {
    for (int run = 0; run < runs; ++run) {
        FakeEngine e(plan, 384, rng.next());
        broken_schedule(plan, e, ring_wait, stage_wait);
        e.finish();
        if (!e.error.empty()) return true;
    }
    return false;
}

void test_schedule() {
    Rng rng{202};
    std::vector<int32_t> all(384);
    for (int32_t e = 0; e < 384; ++e) all[(size_t) e] = e;
    PrefillPlan plan;
    std::vector<int32_t> hit_row;
    for (const auto& ring : {std::vector<int>{16, 3, 2}, {1, 1, 1}, {4, 2, 1}, {8, 4, 3}, {16, 1, 2}}) {
        for (const double staged_share : {0.0, 0.3, 1.0}) {
            const std::vector<int32_t> eop = routing(rng, 1024, 6, all);
            std::vector<int32_t> slab(384, -1);
            for (int k = 0; k < 64; ++k) slab[(size_t) k * 5] = k;
            std::vector<uint8_t> staged(384);
            for (uint8_t& s : staged) s = rng.uniform(0.0f, 1.0f) < staged_share;
            plan_prefill(eop, 384, &slab, staged, ring[0], ring[1], ring[2], plan, hit_row);
            int64_t n_staged_passes = 0;
            for (const PrefillPass& p : plan.passes) n_staged_passes += p.stage >= 0;
            std::string err;
            int64_t waits = 0;
            const int kInterleavings = 200;
            for (int run = 0; run < kInterleavings && err.empty(); ++run) {
                FakeEngine e(plan, 384, rng.next());
                run_prefill(plan, e);
                e.finish();
                err = e.error;
                waits = std::max(waits, e.host_waits);
                if (err.empty() && e.host_waits > n_staged_passes) err = "more host waits than staged passes";
            }
            // the same checker on the schedule without the ring's wait, then without the staging wait: some
            // interleaving must break each (with no set or staging set used twice there is nothing to catch)
            int64_t n_ring = 0;
            for (const PrefillPass& p : plan.passes) n_ring += p.set >= 0;
            const bool reuses_ring = n_ring > ring[1], reuses_stage = n_staged_passes > ring[2];
            const bool ring_caught = reuses_ring && caught(plan, rng, false, true, kInterleavings);
            const bool stage_caught = reuses_stage && caught(plan, rng, true, false, kInterleavings);
            const auto verdict = [](bool reuses, bool got) {
                return !reuses ? "nothing reused" : got ? "caught" : "NOT CAUGHT";
            };
            char what[360];
            std::snprintf(what, sizeof(what),
                          "schedule: ring %2d x %d, staging %d, %3.0f%% staged: %3zu passes (%3lld staged), %d random "
                          "interleavings, host waits <= %lld; without the ring wait: %s; without the staging wait: "
                          "%s%s%s",
                          ring[0], ring[1], ring[2], staged_share * 100, plan.passes.size(),
                          (long long) n_staged_passes, kInterleavings, (long long) waits,
                          verdict(reuses_ring, ring_caught), verdict(reuses_stage, stage_caught),
                          err.empty() ? "" : ": ", err.c_str());
            report(err.empty() && (ring_caught || !reuses_ring) && (stage_caught || !reuses_stage), what);
        }
    }
}

// ---------------------------------------------------------------- 3. the staging jobs over threads
void test_stage_jobs() {
    Rng rng{303};
    constexpr int kThreads = 8;
    std::vector<char> src(4 << 20);
    for (char& b : src) b = (char) rng.next();
    bool all_ok = true;
    uint64_t bytes = 0;
    int jobs_run = 0;
    for (int iter = 0; iter < 40; ++iter) {
        std::vector<char> dst(src.size(), 0);
        std::vector<std::vector<StageChunk>> table((size_t) (1 + rng.below(12)));
        uint64_t want = 0;
        size_t off = 0;
        for (auto& job : table) {
            const size_t n_chunks = (size_t) rng.below(40);  // a job may be empty
            for (size_t i = 0; i < n_chunks && off < src.size(); ++i) {
                const size_t len = std::min<size_t>(1 + rng.below(20000), src.size() - off);
                job.push_back({src.data() + off, dst.data() + off, len});
                off += len;
                want += len;
            }
        }
        StageJobs jobs;
        jobs.reset();
        bool ok = true;
        std::vector<std::thread> helpers;
        for (int t = 1; t < kThreads; ++t) helpers.emplace_back([&jobs] { jobs.help(); });
        jobs.begin(&table);
        for (size_t j = 0; j < table.size(); ++j) {
            jobs.run(j);
            for (const StageChunk& c : table[j]) ok &= std::memcmp(c.dst, c.src, c.bytes) == 0;
            ++jobs_run;
        }
        jobs.close();
        for (std::thread& t : helpers) t.join();
        ok &= jobs.bytes() == want;
        all_ok &= ok;
        bytes += want;
    }
    char what[200];
    std::snprintf(what, sizeof(what),
                  "staging jobs: %d threads, 40 rounds, %d jobs of 0 to 39 random chunks, %.1f MiB: every job's bytes "
                  "in when run returns", kThreads, jobs_run, (double) bytes / (1024.0 * 1024.0));
    report(all_ok, what);
}

// ---------------------------------------------------------------- 4. a prefill call through the op, CPU engine
constexpr int64_t kEmbd = 2048;
constexpr int64_t kFf = 1024;
constexpr int64_t kExperts = 64;
constexpr int64_t kUsed = 6;
constexpr float kLimit = 10.0f;
constexpr int kOpThreads = 8;
constexpr int64_t kPrefillMin = 64;

// The expert tensors as the fork has them: up, gate and down at unaligned offsets of one file, mapped read-only and
// shared as llama.cpp maps a GGUF; only the pool's experts are filled.
struct Experts {
    ggml_context* ctx = nullptr;
    ggml_tensor *up = nullptr, *gate = nullptr, *down = nullptr;
    std::string path;
    int fd = -1;
    char* map = nullptr;
    size_t size = 0;
};

Experts make_experts(const std::vector<int32_t>& pool) {
    const struct {
        ggml_type type;
        int64_t n_per_row, n_rows;
        const char* name;
        float amplitude;
        uint64_t seed;
    } parts[3] = {{GGML_TYPE_Q2_K, kEmbd, kFf, "blk.0.ffn_up_exps.weight", 0.25f, 11},
                  {GGML_TYPE_Q2_K, kEmbd, kFf, "blk.0.ffn_gate_exps.weight", 0.25f, 12},
                  {GGML_TYPE_Q3_K, kFf, kEmbd, "blk.0.ffn_down_exps.weight", 0.05f, 13}};
    // a 160-byte head and 96 bytes between the tensors, so no expert starts on a block boundary
    std::vector<char> bytes(160, 0x5a);
    size_t offsets[3];
    for (int p = 0; p < 3; ++p) {
        offsets[p] = bytes.size();
        const size_t expert_bytes = ggml_row_size(parts[p].type, parts[p].n_per_row) * (size_t) parts[p].n_rows;
        bytes.resize(offsets[p] + expert_bytes * (size_t) kExperts, 0);
        std::vector<float> src((size_t) (parts[p].n_per_row * parts[p].n_rows));
        for (const int32_t e : pool) {
            Rng rng{parts[p].seed * 1000003u + (uint64_t) e};
            for (float& v : src) v = rng.uniform(-parts[p].amplitude, parts[p].amplitude);
            ggml_quantize_chunk(parts[p].type, src.data(), bytes.data() + offsets[p] + (size_t) e * expert_bytes, 0,
                                parts[p].n_rows, parts[p].n_per_row, nullptr);
        }
        if (p < 2) bytes.resize(bytes.size() + 96, 0x5a);
    }
    Experts x;
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        std::fprintf(stderr, "FAIL: getcwd: %s\n", std::strerror(errno));
        std::exit(1);
    }
    x.path = std::string(cwd) + "/dsv41_moe_prefill_test.model.bin";
    FILE* out = std::fopen(x.path.c_str(), "wb");
    if (out == nullptr || std::fwrite(bytes.data(), 1, bytes.size(), out) != bytes.size() || std::fclose(out) != 0) {
        std::fprintf(stderr, "FAIL: cannot write %s\n", x.path.c_str());
        std::exit(1);
    }
    x.size = bytes.size();
    x.fd = open(x.path.c_str(), O_RDONLY);
    x.map = static_cast<char*>(mmap(nullptr, x.size, PROT_READ, MAP_SHARED, x.fd, 0));
    if (x.fd < 0 || x.map == MAP_FAILED) {
        std::fprintf(stderr, "FAIL: open and mmap %s: %s\n", x.path.c_str(), std::strerror(errno));
        std::exit(1);
    }
    ggml_init_params wp = {/*.mem_size =*/ 4 * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true};
    x.ctx = ggml_init(wp);
    ggml_tensor** dst[3] = {&x.up, &x.gate, &x.down};
    for (int p = 0; p < 3; ++p) {
        ggml_tensor* t = ggml_new_tensor_3d(x.ctx, parts[p].type, parts[p].n_per_row, parts[p].n_rows, kExperts);
        ggml_set_name(t, parts[p].name);
        t->data = x.map + offsets[p];
        *dst[p] = t;
    }
    return x;
}

void free_experts(Experts& x) {
    munmap(x.map, x.size);
    close(x.fd);
    unlink(x.path.c_str());
    ggml_free(x.ctx);
}

// The residency of the direct staging: layer 0 with no VRAM and no warm experts, so it only binds the file and lends
// its reader threads (the misses of a call under the minimum are cold, computed from the mapping).
HotSet prefill_hot_set() {
    HotSet set;
    set.path = "(prefill test hot set)";
    set.budget_gib = 0.0;
    set.ram_budget_gib = 0.0;
    HotLayer l;
    l.layer = 0;
    l.n_expert = kExperts;
    l.n_embd = kEmbd;
    l.n_ff = kFf;
    l.up_type = GGML_TYPE_Q2_K;
    l.gate_type = GGML_TYPE_Q2_K;
    l.down_type = GGML_TYPE_Q3_K;
    set.layers.push_back(l);
    return set;
}

// The GPU tier's place: the plan of moe_gpu.cpp, the staging filled as moe_gpu.cpp fills it (pagecache: StageJobs
// on the op's threads; direct: Residency::read_expert into the same StageLayout), the copies into a host ring and
// each pass computed from the bytes there (one-slot calls of the CPU reference on a tensor over the ring set, the
// expert being its slot), so a pair computed from the wrong bytes changes the output.
class CpuPrefill final : public HitPath {
public:
    std::vector<int32_t> slab_slot = std::vector<int32_t>((size_t) kExperts, -1);
    std::vector<uint8_t> staged = std::vector<uint8_t>((size_t) kExperts, 0);
    int set_size = 4, n_sets = 2, n_stage_sets = 1;
    PrefillCold cold = PrefillCold::kPageCache;
    CpuKernel kernel = CpuKernel::kRef;
    int64_t prefill_calls = 0, rows = 0;
    // the bytes the op's threads copied into staging and the residency read into it, the staged expert reads
    uint64_t staged_bytes = 0, direct_bytes = 0, want_staged_bytes = 0;
    int64_t direct_experts = 0, want_staged_experts = 0;

    ~CpuPrefill() override { std::free(stage_); }

    const std::vector<int32_t>* slots(const Call&) override { return nullptr; }

    bool prefill(Call& c, const Residency*) override {
        if (c.n_tokens < kPrefillMin) return false;
        const int64_t n_pairs = c.n_tokens * c.n_used;
        c.expert_of_pair.assign(static_cast<const int32_t*>(c.ids->data),
                                static_cast<const int32_t*>(c.ids->data) + n_pairs);
        c.staged = staged;
        const bool any_slab = std::any_of(slab_slot.begin(), slab_slot.end(), [](int32_t s) { return s >= 0; });
        plan_prefill(c.expert_of_pair, c.n_expert, any_slab ? &slab_slot : nullptr, c.staged, set_size, n_sets,
                     n_stage_sets, c.plan, c.hit_row);
        c.n_hits = n_pairs;
        return true;
    }

    void submit(Call& c) override {
        ++prefill_calls;
        const ggml_tensor* t[3] = {c.up, c.gate, c.down};
        for (int p = 0; p < 3; ++p) {
            part_[p] = t[p]->nb[2];
            ring_[p].assign((size_t) (n_sets * set_size) * part_[p], 0);
        }
        layout_ = stage_layout(part_);
        std::free(stage_);
        const size_t stage_bytes = (size_t) (n_stage_sets * set_size) * layout_.slot_bytes;
        stage_ = static_cast<char*>(std::aligned_alloc(kDirectAlign, stage_bytes));
        if (stage_ == nullptr) {
            std::fprintf(stderr, "FAIL: aligned_alloc %zu bytes of staging\n", stage_bytes);
            std::exit(1);
        }
        std::memset(stage_, 0, stage_bytes);
        c.stage_table.clear();
        for (const PrefillPass& p : c.plan.passes) {
            if (p.stage < 0) continue;
            want_staged_experts += (int64_t) p.experts.size();
            for (int part = 0; part < 3; ++part) want_staged_bytes += part_[part] * p.experts.size();
            if (cold != PrefillCold::kPageCache) continue;
            c.stage_table.emplace_back();
            for (size_t k = 0; k < p.experts.size(); ++k) {
                for (int part = 0; part < 3; ++part) {
                    const char* src = static_cast<const char*>(t[part]->data) + (size_t) p.experts[k] * part_[part];
                    char* dst = stage_slot(p.stage, k, part);
                    for (size_t off = 0; off < part_[part]; off += 65536) {
                        const size_t n = std::min<size_t>(65536, part_[part] - off);
                        c.stage_table.back().push_back({src + off, dst + off, n});
                    }
                }
            }
        }
        c.stage.begin(&c.stage_table);
        out_.assign((size_t) (c.plan.n_rows * c.n_embd), 0.0f);
        Engine e(*this, c);
        run_prefill(c.plan, e);
        c.stage.close();
        staged_bytes += c.stage.bytes();
    }

    const float* wait(Call&) override { return out_.data(); }

private:
    // runs every op at once, in call order: one stream for both, which run_prefill's order must make correct
    class Engine final : public PrefillEngine {
    public:
        Engine(CpuPrefill& t, Call& c) : t_(t), c_(c) {}
        void stage_wait(int) override {}
        void stage(const PrefillPass& p) override {
            rows_.assign(p.experts.size() * 3, nullptr);
            if (t_.cold == PrefillCold::kPageCache) {
                c_.stage.run(job_++);
                for (size_t k = 0; k < p.experts.size(); ++k) {
                    for (int part = 0; part < 3; ++part) rows_[k * 3 + (size_t) part] = t_.stage_slot(p.stage, k, part);
                }
                return;
            }
            std::atomic<int64_t> left{0};
            for (size_t k = 0; k < p.experts.size(); ++k) {
                t_.direct_bytes += c_.residency->read_expert(c_.layer, p.experts[k], t_.stage_base(p.stage, k),
                                                             t_.layout_.part_offset, left, &rows_[k * 3]);
                ++t_.direct_experts;
            }
            while (left.load(std::memory_order_acquire) != 0) std::this_thread::yield();
        }
        void copy_wait_compute(int) override {}
        void copy(const PrefillPass& p) override {
            const ggml_tensor* src_t[3] = {c_.up, c_.gate, c_.down};
            for (size_t k = 0; k < p.experts.size(); ++k) {
                for (int part = 0; part < 3; ++part) {
                    // unstaged experts stand for the pinned warm tier: straight from the fork's tensor
                    const char* src = p.staged[k]
                        ? rows_[k * 3 + (size_t) part]
                        : static_cast<const char*>(src_t[part]->data) + (size_t) p.experts[k] * t_.part_[part];
                    std::memcpy(t_.ring_[part].data() + ((size_t) p.set * t_.set_size + k) * t_.part_[part], src,
                                t_.part_[part]);
                }
            }
        }
        void compute_wait_copy(int) override {}
        void compute(const PrefillPass& p) override {
            // the weights: the fork's tensors for the slab (the expert is its id), the ring set otherwise (its slot)
            ggml_init_params wp = {/*.mem_size =*/ 4 * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr,
                                   /*.no_alloc =*/ true};
            ggml_context* wctx = ggml_init(wp);
            ggml_tensor* w[3] = {const_cast<ggml_tensor*>(c_.up), const_cast<ggml_tensor*>(c_.gate),
                                 const_cast<ggml_tensor*>(c_.down)};
            if (p.set >= 0) {
                const char* names[3] = {"blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight",
                                        "blk.0.ffn_down_exps.weight"};
                for (int part = 0; part < 3; ++part) {
                    const ggml_tensor* src = w[part];
                    w[part] = ggml_new_tensor_3d(wctx, src->type, src->ne[0], src->ne[1], t_.set_size);
                    ggml_set_name(w[part], names[part]);
                    w[part]->data = t_.ring_[part].data() + (size_t) p.set * t_.set_size * t_.part_[part];
                }
            }
            ggml_init_params params = {/*.mem_size =*/ 8 * ggml_tensor_overhead() + 4 * (size_t) c_.n_embd * 4,
                                       /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
            for (size_t i = 0; i < p.pairs.size(); ++i) {
                const int64_t pair = p.pairs[i], tok = pair / c_.n_used, slot = pair % c_.n_used;
                ggml_context* ctx = ggml_init(params);
                ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c_.n_embd, 1);
                std::memcpy(cur->data, static_cast<const char*>(c_.cur->data) + tok * c_.cur->nb[1],
                            (size_t) c_.n_embd * sizeof(float));
                ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
                static_cast<int32_t*>(ids->data)[0] = p.set >= 0 ? p.slots[i] : c_.expert_of_pair[(size_t) pair];
                ggml_tensor* wt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1);
                ggml_get_data_f32(wt)[0] = *reinterpret_cast<const float*>(
                    static_cast<const char*>(c_.weights->data) + tok * c_.weights->nb[2] + slot * c_.weights->nb[1]);
                ggml_tensor* args[] = {cur, ids, wt, w[0], w[1], w[2]};
                ggml_tensor* one = ggml_custom_4d(ctx, GGML_TYPE_F32, c_.n_embd, 1, 1, 1, args, 6, nullptr, 1, nullptr);
                strata::dsv41::forward(one, 1, c_.limit, nullptr, nullptr, t_.kernel);
                std::memcpy(t_.out_.data() + (size_t) (p.row0 + (int64_t) i) * c_.n_embd, one->data,
                            (size_t) c_.n_embd * sizeof(float));
                ggml_free(ctx);
            }
            t_.rows += (int64_t) p.pairs.size();
            ggml_free(wctx);
        }

    private:
        CpuPrefill& t_;
        Call& c_;
        size_t job_ = 0;
        std::vector<const char*> rows_;  // [k, part]: the last staged pass's experts in staging
    };

    char* stage_base(int h, size_t k) const { return stage_ + ((size_t) h * set_size + k) * layout_.slot_bytes; }
    char* stage_slot(int h, size_t k, int part) const { return stage_base(h, k) + layout_.part_offset[part]; }

    size_t part_[3] = {};
    std::vector<char> ring_[3];
    StageLayout layout_;
    char* stage_ = nullptr;  // kDirectAlign-aligned, as the GPU tier's pinned staging
    std::vector<float> out_;
};

HitPath* g_path = nullptr;
Residency* g_residency = nullptr;
CpuKernel g_kernel = CpuKernel::kRef;

void prefill_op(ggml_tensor* dst, int, int nth, void*) {
    strata::dsv41::forward(dst, nth, strata_moe_clamp(dst), g_path, g_residency, g_kernel);
}

std::vector<float> compute(ggml_tensor* cur, ggml_tensor* ids, ggml_tensor* weights, ggml_tensor* up,
                           ggml_tensor* gate, ggml_tensor* down, HitPath* path, Residency* residency) {
    const int64_t n_tok = cur->ne[1];
    ggml_init_params params = {/*.mem_size =*/ (size_t) 16 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* args[] = {cur, ids, weights, up, gate, down};
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, kEmbd, n_tok, 1, 1, args, 6, prefill_op, GGML_N_TASKS_MAX,
                                      nullptr);
    strata_moe_set_clamp(out, kLimit);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    g_path = path;
    g_residency = residency;
    if (ggml_graph_compute_with_ctx(ctx, gf, kOpThreads) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: graph compute\n");
        std::exit(1);
    }
    std::vector<float> v(ggml_get_data_f32(out), ggml_get_data_f32(out) + kEmbd * n_tok);
    ggml_free(ctx);
    return v;
}

void test_prefill_cold_choice() {
    const PrefillColdChoice unset = choose_prefill_cold(nullptr), empty = choose_prefill_cold("");
    const PrefillColdChoice page = choose_prefill_cold("pagecache"), direct = choose_prefill_cold("direct");
    const PrefillColdChoice bad = choose_prefill_cold("odirect");
    const bool ok = unset.is_default && unset.cold == PrefillCold::kPageCache && unset.error.empty() &&
                    empty.is_default && empty.cold == PrefillCold::kPageCache && !page.is_default &&
                    page.cold == PrefillCold::kPageCache && !direct.is_default && direct.cold == PrefillCold::kDirect &&
                    direct.error.empty() && !bad.error.empty();
    report(ok, "STRATA_MOE_PREFILL_COLD: unset and empty are pagecache (the default), pagecache and direct parse, "
               "'odirect' is refused: " + bad.error);
}

void test_op() {
    const std::vector<int32_t> pool = {0, 63, 7, 12, 20, 21, 29, 33, 34, 40, 41, 45, 50, 51, 52, 55, 58, 60};
    Experts x = make_experts(pool);
    ggml_tensor *up = x.up, *gate = x.gate, *down = x.down;
    Residency residency(prefill_hot_set(), 0.0, false, 4, 16, ::pread, RamMode::kLocked, nullptr,
                        ColdPath::kPageCache);
    g_kernel = fast_kernel_cpu_ok() ? CpuKernel::kFast : CpuKernel::kRef;
    std::printf("prefill op: n_embd %lld n_ff %lld, %lld experts (%zu filled), %lld used, q2_K gate/up, q3_K down, "
                "clamp %g, %d op threads, STRATA_MOE_CPU=%s, prefill from %lld tokens\n",
                (long long) kEmbd, (long long) kFf, (long long) kExperts, pool.size(), (long long) kUsed, kLimit,
                kOpThreads, g_kernel == CpuKernel::kFast ? "fast" : "ref", (long long) kPrefillMin);

    Rng rng{404};
    struct Case {
        int64_t n_tok;
        int n_slab;
        double staged_share;
        int set_size, n_sets, n_stage_sets;
    } cases[] = {
        {64, 0, 1.0, 4, 2, 1}, {64, 5, 0.5, 4, 2, 1}, {96, 3, 0.0, 16, 3, 2}, {80, 0, 0.5, 1, 1, 1},
        {72, 18, 0.0, 4, 2, 1}, {32, 5, 0.5, 4, 2, 1},
    };
    for (const Case& k : cases) {
        ggml_init_params ip = {/*.mem_size =*/ (size_t) 8 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
        ggml_context* ictx = ggml_init(ip);
        ggml_tensor* cur = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, kEmbd, k.n_tok);
        for (int64_t i = 0; i < kEmbd * k.n_tok; ++i) ggml_get_data_f32(cur)[i] = rng.uniform(-1.0f, 1.0f);
        ggml_tensor* ids = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, kUsed, k.n_tok);
        ggml_tensor* weights = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, 1, kUsed, k.n_tok);
        const std::vector<int32_t> eop = routing(rng, k.n_tok, kUsed, pool);
        std::copy(eop.begin(), eop.end(), static_cast<int32_t*>(ids->data));
        for (int64_t i = 0; i < kUsed * k.n_tok; ++i) ggml_get_data_f32(weights)[i] = rng.uniform(0.05f, 0.5f);

        const std::vector<float> all_cpu = compute(cur, ids, weights, up, gate, down, nullptr, nullptr);
        std::vector<int32_t> order = pool;
        for (size_t i = order.size() - 1; i > 0; --i) std::swap(order[i], order[(size_t) rng.below(i + 1)]);
        std::vector<uint8_t> staged((size_t) kExperts);
        for (uint8_t& s : staged) s = rng.uniform(0.0f, 1.0f) < k.staged_share;
        // the same call with the staging filled both ways
        std::vector<float> got_by[2];
        for (const PrefillCold cold : {PrefillCold::kPageCache, PrefillCold::kDirect}) {
            CpuPrefill path;
            path.kernel = g_kernel;
            path.set_size = k.set_size;
            path.n_sets = k.n_sets;
            path.n_stage_sets = k.n_stage_sets;
            path.cold = cold;
            for (int s = 0; s < k.n_slab; ++s) path.slab_slot[(size_t) order[(size_t) s]] = k.n_slab - 1 - s;
            path.staged = staged;
            const std::vector<float> got = compute(cur, ids, weights, up, gate, down, &path, &residency);
            got_by[cold == PrefillCold::kDirect] = got;
            size_t differ = 0;
            for (size_t i = 0; i < got.size(); ++i) differ += std::memcmp(&got[i], &all_cpu[i], sizeof(float)) != 0;

            const bool want_prefill = k.n_tok >= kPrefillMin;
            const bool path_ok = want_prefill ? path.prefill_calls == 1 && path.rows == k.n_tok * kUsed
                                              : path.prefill_calls == 0 && path.rows == 0;
            // pagecache: the op's threads copy every staged byte; direct: each staged expert read once, whole
            // blocks around its parts, and nothing copied by the op's threads
            const bool stage_ok = cold == PrefillCold::kPageCache
                                      ? path.staged_bytes == path.want_staged_bytes && path.direct_experts == 0
                                      : path.staged_bytes == 0 && path.direct_experts == path.want_staged_experts &&
                                            path.direct_bytes >= path.want_staged_bytes;
            const bool direct = cold == PrefillCold::kDirect;
            char what[400];
            std::snprintf(what, sizeof(what),
                          "prefill op, %-9s: %3lld tokens, slab %2d, %3.0f%% staged, ring %2d x %d, staging %d: %s, "
                          "%4lld rows, %.1f MiB staged by %s (%lld experts), %zu of %zu output floats differ from "
                          "all-CPU", prefill_cold_name(cold), (long long) k.n_tok, k.n_slab, k.staged_share * 100,
                          k.set_size, k.n_sets, k.n_stage_sets,
                          want_prefill        ? (path.prefill_calls == 1 ? "prefill call" : "NOT A PREFILL CALL")
                          : path.prefill_calls == 0 ? "under the minimum, CPU path"
                                                    : "PREFILL UNDER THE MINIMUM",
                          (long long) path.rows,
                          (double) (direct ? path.direct_bytes : path.staged_bytes) / (1024.0 * 1024.0),
                          direct ? "O_DIRECT reads" : "the op threads",
                          (long long) (direct ? path.direct_experts : path.want_staged_experts), differ, got.size());
            report(differ == 0 && path_ok && stage_ok, what);
        }
        size_t between = 0;
        for (size_t i = 0; i < got_by[0].size(); ++i) {
            between += std::memcmp(&got_by[0][i], &got_by[1][i], sizeof(float)) != 0;
        }
        report(between == 0, "prefill op: " + std::to_string(k.n_tok) + " tokens: pagecache and direct staging give " +
                                 std::to_string(between) + " differing output floats");
        ggml_free(ictx);
    }
    free_experts(x);
}

}  // namespace

int main() {
    ggml_cpu_init();
    std::printf("dsv41_moe_prefill_test: the prefill plan, the ring schedule, the staging jobs and a prefill call\n");
    test_plan();
    test_schedule();
    test_stage_jobs();
    test_prefill_cold_choice();
    test_op();
    std::printf("dsv41_moe_prefill_test: %s (%d of %d cases failed)\n", g_failed ? "FAIL" : "PASS", g_failed, g_cases);
    return g_failed ? 1 : 0;
}
