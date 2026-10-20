// src/dsv41/engine/hot_policy.cpp - the per-request hot set of include/strata/dsv41/engine/hot_policy.hpp.
//
// tools/dsv41/engine/hot_policy_sim.py repeats these rules (the orders, ties, limits and the decay's arithmetic), and
// verify.sh holds the two to the same hits per layer on one trace, so a change here goes there too. The file is built
// with -ffp-contract=off (CMakeLists.txt): the counts' products and sums are Python's double operations one by one.
#include "strata/dsv41/engine/hot_policy.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 hot set: %s\n", msg);
    std::exit(1);
}

std::string fmt(const char* f, long long a, long long b = 0, long long c = 0) {
    char msg[256];
    std::snprintf(msg, sizeof msg, f, a, b, c);
    return msg;
}

}  // namespace

const char* hot_policy_name(HotPolicy p) {
    switch (p) {
        case HotPolicy::Static: return "static";
        case HotPolicy::Prompt: return "prompt";
        case HotPolicy::Adaptive: return "adaptive";
    }
    return "?";
}

bool parse_hot_policy(const std::string& s, HotPolicy& p) {
    if (s == "static") p = HotPolicy::Static;
    else if (s == "prompt") p = HotPolicy::Prompt;
    else if (s == "adaptive") p = HotPolicy::Adaptive;
    else return false;
    return true;
}

void hot_config_check(const HotConfig& c) {
    if (c.spare < 0 || (c.spare > 0 && c.max_swaps < 1))
        failf("--hot-spare %d, --hot-swaps %d: spare slots >= 0, and at least one swap per token with any", c.spare, c.max_swaps);
    if (!(c.half_life > 0)) failf("--hot-half-life %g: a positive number of decode tokens", c.half_life);
    if (!(c.margin_mul >= 0)) failf("--hot-margin %g:%g: the factor must be >= 0", c.margin_mul, c.margin_add);
}

HotRequestStart hot_request_start(bool deterministic, bool reset, bool resume) {
    HotRequestStart s;
    s.reset = deterministic || reset;
    s.resume = !s.reset && resume;
    return s;
}

bool hot_boundary_waits(bool deterministic, bool check) { return deterministic || check; }

bool hot_request_end_resets(bool deterministic) { return deterministic; }

void HotRestore::begin(int n_layer) {
    if (open_) failf("a layer by layer restore started while another is open");
    if (n_layer < 1) failf("a layer by layer restore of %d layers", n_layer);
    open_ = true;
    n_layer_ = n_layer;
    recorded_ = 0;
    waited_.assign((size_t) n_layer, 0);
}

void HotRestore::record(int l) {
    if (!open_ || l != recorded_ || l >= n_layer_)
        failf("the restore recorded layer %d, the next in order is %d of %d", l, recorded_, n_layer_);
    ++recorded_;
}

bool HotRestore::wait_before(int l) {
    if (!open_) return false;
    if (l < 0 || l >= n_layer_) failf("a computation of layer %d during a restore of %d layers", l, n_layer_);
    if (l >= recorded_) failf("layer %d computes before its restore is queued (%d of %d layers recorded)", l, recorded_, n_layer_);
    if (waited_[(size_t) l]) return false;
    waited_[(size_t) l] = 1;
    return true;
}

void HotRestore::done() {
    if (open_ && recorded_ != n_layer_) failf("the restore done with %d of %d layers recorded", recorded_, n_layer_);
    open_ = false;
}

void HotPrompt::begin_request() {
    open_ = true;
    beside_ = false;
}

void HotPrompt::prefill(bool beside) {
    open_ = true;
    beside_ = beside;
}

HotPromptEnd HotPrompt::end() {
    const HotPromptEnd e = !open_ ? HotPromptEnd::None : beside_ ? HotPromptEnd::Merge : HotPromptEnd::Done;
    open_ = beside_ = false;
    return e;
}

void HotLayerPlan::init(const Layout& l) {
    n_expert_ = l.n_expert;
    mapping_pinned_ = l.mapping_pinned;
    const size_t W = l.warm.size(), H = l.homed.size();
    if (H > l.vram.size()) failf("%zu homed experts for a slab of %zu", H, l.vram.size());
    host_homes_ = H > 0 && H == l.vram.size();
    if (host_homes_ && l.spare_homes) failf("host homes and %d spare homes: a victim keeps its home", l.spare_homes);
    const size_t NE = (size_t) l.n_expert;
    slot_of_.assign(NE, -1);
    home_of_.assign(NE, -1);
    static_.assign(NE, 0);
    target_.assign(NE, 0);
    flight_.assign(NE, 0);
    prompt_.assign(NE, 0.0);
    score_.assign(NE, 0.0);
    rank_.assign(NE, -1);
    fixed_.assign(NE, 0);
    if (l.spare < 0 || l.spare_homes < 0) failf("%d spare slots, %d spare homes", l.spare, l.spare_homes);
    slot_expert_.assign(l.vram.size() + (size_t) l.spare, -1);
    home_expert_.assign(W + H + (size_t) l.spare_homes, -1);
    int32_t r = 0;
    auto place = [&](int32_t e, const char* what) {
        if (e < 0 || e >= l.n_expert) failf("%s expert %d of %d", what, e, l.n_expert);
        if (rank_[(size_t) e] >= 0) failf("expert %d is in the hot set twice (%s)", e, what);
        rank_[(size_t) e] = r++;
    };
    for (size_t s = 0; s < l.vram.size(); ++s) {
        const int32_t e = l.vram[s];
        place(e, "VRAM");
        slot_of_[(size_t) e] = (int32_t) s;
        slot_expert_[s] = e;
        static_[(size_t) e] = 1;
    }
    for (size_t k = 0; k < l.warm.size(); ++k) {
        const int32_t e = l.warm[k];
        place(e, "warm");
        home_of_[(size_t) e] = (int32_t) k;
        home_expert_[k] = e;
    }
    for (int32_t e = 0; e < l.n_expert; ++e)
        if (rank_[(size_t) e] < 0) rank_[(size_t) e] = r++;
    // the homed slab experts' permanent homes after the warm tier's
    for (size_t k = 0; k < H; ++k) {
        const int32_t e = l.homed[k];
        if (e < 0 || e >= l.n_expert || !static_[(size_t) e]) failf("homed expert %d is not in the hot set's slab", e);
        if (fixed_[(size_t) e]) failf("expert %d is homed twice", e);
        fixed_[(size_t) e] = 1;
        home_of_[(size_t) e] = (int32_t) (W + k);
        home_expert_[W + k] = e;
    }
    n_homed_ = (int) H;
    n_reserved_ = 0;
    // check() holds every home in full, the homed ones' when partial (-2: a home that changes)
    if (host_homes_) {
        fixed_homes_ = home_expert_;
    } else if (H) {
        fixed_homes_.assign(home_expert_.size(), -2);
        for (size_t k = 0; k < H; ++k) fixed_homes_[W + k] = l.homed[k];
    } else {
        fixed_homes_.clear();
    }
    n_static_ = (int) l.vram.size();
    n_resident_ = n_static_;
    n_flight_ = 0;
    // popped from the back: the lowest free slot and home first
    free_slots_.clear();
    for (int s = (int) slot_expert_.size() - 1; s >= n_static_; --s) free_slots_.push_back(s);
    free_homes_.clear();
    for (int k = (int) home_expert_.size() - 1; k >= (int) (W + H); --k) free_homes_.push_back(k);
    begin_request(HotPolicy::Static, false);
}

void HotLayerPlan::begin_request(HotPolicy p, bool resume) {
    policy_ = p;
    prompt_seen_ = false;
    lfu_ = false;
    resume_ = resume && p != HotPolicy::Static;
    std::fill(prompt_.begin(), prompt_.end(), 0.0);
    // a resumed turn keeps the last request's counts until prompt_done adds its suffix's
    if (!resume_) std::fill(score_.begin(), score_.end(), 0.0);
    target_ = static_;
}

void HotLayerPlan::count_prompt(const int32_t* ids, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= n_expert_) failf("a prompt pair routed to expert %d of %d", ids[i], n_expert_);
        prompt_[(size_t) ids[i]] += 1.0;
    }
}

void HotLayerPlan::prompt_done(const HotConfig& c) {
    prompt_seen_ = true;
    lfu_ = false;
    if (policy_ == HotPolicy::Static) return;
    if (resume_)
        for (int32_t e = 0; e < n_expert_; ++e) score_[(size_t) e] = score_[(size_t) e] + prompt_[(size_t) e];
    else
        score_ = prompt_;
    // taken into the counts; a prompt read beside the batch later merges its own alone (merge_prompt)
    std::fill(prompt_.begin(), prompt_.end(), 0.0);
    if (resume_ && policy_ == HotPolicy::Adaptive) {
        // a resumed adaptive turn: an expert outside the slab ties with the slab's unless its count beats theirs by
        // adaptive's hysteresis margin (count(in) > margin_mul count(out) + margin_add), so the target is the slab with
        // the pairs plan_lfu would swap at once: the best counted outside for the least counted inside, while the
        // margin holds. With exact ties only, the slab's members the margin kept would all leave again (measured with
        // hot_policy_sim.py two-turn, 4 layers of 384 experts: 106 swaps in turn 2 against 65 with the margin).
        ins_.clear();
        outs_.clear();
        for (int32_t e = 0; e < n_expert_; ++e) {
            target_[(size_t) e] = slot_of_[(size_t) e] >= 0;
            if (slot_of_[(size_t) e] >= 0) outs_.push_back(e);
            else if (promotable(e)) ins_.push_back(e);
        }
        std::sort(ins_.begin(), ins_.end(), [&](int32_t a, int32_t b) {
            if (score_[(size_t) a] != score_[(size_t) b]) return score_[(size_t) a] > score_[(size_t) b];
            return rank_[(size_t) a] < rank_[(size_t) b];
        });
        std::sort(outs_.begin(), outs_.end(), [&](int32_t a, int32_t b) {
            if (score_[(size_t) a] != score_[(size_t) b]) return score_[(size_t) a] < score_[(size_t) b];
            return rank_[(size_t) a] > rank_[(size_t) b];
        });
        for (size_t i = 0; i < ins_.size() && i < outs_.size(); ++i) {
            const int32_t in = ins_[i], out = outs_[i];
            if (!(score_[(size_t) in] > c.margin_mul * score_[(size_t) out] + c.margin_add)) break;
            target_[(size_t) in] = 1;
            target_[(size_t) out] = 0;
        }
        return;
    }
    // the N most counted experts that can be in the slab (resident now, or promotable), ties by the static order
    // (a resumed prompt policy, which has no margin: the slab's experts first among equal counts)
    std::vector<int32_t> order;
    for (int32_t e = 0; e < n_expert_; ++e)
        if (slot_of_[(size_t) e] >= 0 || promotable(e)) order.push_back(e);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        if (score_[(size_t) a] != score_[(size_t) b]) return score_[(size_t) a] > score_[(size_t) b];
        if (resume_) {
            const bool ra = slot_of_[(size_t) a] >= 0, rb = slot_of_[(size_t) b] >= 0;
            if (ra != rb) return ra;
        }
        return rank_[(size_t) a] < rank_[(size_t) b];
    });
    std::fill(target_.begin(), target_.end(), 0);
    for (size_t i = 0; i < order.size() && (int) i < n_static_; ++i) target_[(size_t) order[i]] = 1;
}

void HotLayerPlan::merge_prompt(double mass) {
    double total = 0;
    for (double c : prompt_) total += c;
    if (policy_ == HotPolicy::Adaptive && prompt_seen_ && total > 0)
        for (int32_t e = 0; e < n_expert_; ++e) score_[(size_t) e] += prompt_[(size_t) e] * (mass / total);
    std::fill(prompt_.begin(), prompt_.end(), 0.0);
}

void HotLayerPlan::count_token(const int32_t* ids, int n, double decay) {
    if (policy_ != HotPolicy::Adaptive || !prompt_seen_) return;
    for (double& c : score_) c = c * decay;
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= n_expert_) failf("a decode pair routed to expert %d of %d", ids[i], n_expert_);
        score_[(size_t) ids[i]] += 1.0;
    }
}

HotSource HotLayerPlan::source(int32_t e) const {
    if (slot_of_[(size_t) e] >= 0) return HotSource::Slot;
    if (home_of_[(size_t) e] >= 0) return HotSource::Home;
    return mapping_pinned_ ? HotSource::Mapping : HotSource::Fixed;
}

bool HotLayerPlan::promotable(int32_t e) const {
    const HotSource s = source(e);
    return s == HotSource::Home || s == HotSource::Mapping;
}

void HotLayerPlan::start(int32_t in, int32_t out, std::vector<HotSwap>& sw) {
    HotSwap s;
    s.in = in;
    s.out = out;
    s.slot = free_slots_.back();
    free_slots_.pop_back();
    s.from_home = home_of_[(size_t) in];
    // a homed victim keeps its home (partial host homes); any other needs a free one off a pinned mapping
    if (needs_home() && !fixed_[(size_t) out]) {
        if (free_homes_.empty()) failf("swap %d in, %d out: no free home for the victim", in, out);
        s.to_home = free_homes_.back();
        free_homes_.pop_back();
        ++n_reserved_;
    }
    flight_[(size_t) in] = flight_[(size_t) out] = 1;
    ++n_flight_;
    sw.push_back(s);
}

int HotLayerPlan::plan_target(int max, std::vector<HotSwap>& out, bool* ins_left) {
    ins_.clear();
    outs_.clear();
    for (int32_t e = 0; e < n_expert_; ++e) {
        if (flight_[(size_t) e]) continue;
        const bool in_slab = slot_of_[(size_t) e] >= 0;
        if (target_[(size_t) e] && !in_slab && promotable(e)) ins_.push_back(e);
        if (!target_[(size_t) e] && in_slab) outs_.push_back(e);
    }
    std::sort(ins_.begin(), ins_.end(), [&](int32_t a, int32_t b) {
        if (score_[(size_t) a] != score_[(size_t) b]) return score_[(size_t) a] > score_[(size_t) b];
        return rank_[(size_t) a] < rank_[(size_t) b];
    });
    std::sort(outs_.begin(), outs_.end(), [&](int32_t a, int32_t b) {
        if (score_[(size_t) a] != score_[(size_t) b]) return score_[(size_t) a] < score_[(size_t) b];
        return rank_[(size_t) a] > rank_[(size_t) b];
    });
    int n = 0;
    for (size_t i = 0; i < ins_.size() && i < outs_.size() && n < max; ++i) {
        if (free_slots_.empty() || (needs_home() && free_homes_.empty())) break;
        start(ins_[i], outs_[i], out);
        ++n;
    }
    if (ins_left) *ins_left = (int) ins_.size() > n;
    return n;
}

int HotLayerPlan::plan_lfu(const HotConfig& c, int max, std::vector<HotSwap>& out) {
    int n = 0;
    while (n < max && !free_slots_.empty() && !(needs_home() && free_homes_.empty())) {
        int32_t in = -1, victim = -1;
        for (int32_t e = 0; e < n_expert_; ++e) {
            if (flight_[(size_t) e]) continue;
            const double s = score_[(size_t) e];
            if (slot_of_[(size_t) e] >= 0) {
                if (victim < 0 || s < score_[(size_t) victim] ||
                    (s == score_[(size_t) victim] && rank_[(size_t) e] > rank_[(size_t) victim]))
                    victim = e;
            } else if (promotable(e)) {
                if (in < 0 || s > score_[(size_t) in] || (s == score_[(size_t) in] && rank_[(size_t) e] < rank_[(size_t) in]))
                    in = e;
            }
        }
        if (in < 0 || victim < 0) break;
        if (!(score_[(size_t) in] > c.margin_mul * score_[(size_t) victim] + c.margin_add)) break;
        start(in, victim, out);
        ++n;
    }
    return n;
}

void HotLayerPlan::plan(const HotConfig& c, std::vector<HotSwap>& out) {
    if (c.spare == 0 || c.max_swaps < 1) return;
    if (policy_ == HotPolicy::Static) {
        target_ = static_;
        plan_target(c.max_swaps, out, nullptr);
        return;
    }
    if (!prompt_seen_) return;
    int n = 0;
    if (!lfu_) {
        bool ins_left = false;
        n = plan_target(c.max_swaps, out, &ins_left);
        // adaptive's LFU phase once every member of the prompt's target is resident or on its way (this call's
        // swaps included)
        if (policy_ == HotPolicy::Adaptive && !ins_left) lfu_ = true;
    }
    if (policy_ == HotPolicy::Adaptive && lfu_) plan_lfu(c, c.max_swaps - n, out);
}

void HotLayerPlan::plan_static(int max, std::vector<HotSwap>& out) {
    const std::vector<uint8_t> keep = target_;
    target_ = static_;
    plan_target(max, out, nullptr);
    target_ = keep;
}

int HotLayerPlan::restore_in_place(std::vector<HotSwap>& out) {
    if (needs_home()) failf("an in-place restore with victims that need a copy down (no host homes, the mapping not pinned)");
    if (n_flight_) failf("an in-place restore with %d swaps in flight", n_flight_);
    ins_.clear();
    outs_.clear();
    for (int32_t e = 0; e < n_expert_; ++e) {
        const bool in_slab = slot_of_[(size_t) e] >= 0;
        if (static_[(size_t) e] && !in_slab) {
            if (!promotable(e)) failf("the hot set's expert %d is outside the slab with no host copy", e);
            ins_.push_back(e);
        }
        if (!static_[(size_t) e] && in_slab) outs_.push_back(e);
    }
    if (ins_.size() != outs_.size())
        failf("an in-place restore: %zu of the hot set's experts outside the slab, %zu others in it", ins_.size(), outs_.size());
    std::sort(ins_.begin(), ins_.end(), [&](int32_t a, int32_t b) { return rank_[(size_t) a] < rank_[(size_t) b]; });
    std::sort(outs_.begin(), outs_.end(), [&](int32_t a, int32_t b) { return slot_of_[(size_t) a] < slot_of_[(size_t) b]; });
    for (size_t i = 0; i < ins_.size(); ++i) {
        HotSwap s;
        s.in = ins_[i];
        s.out = outs_[i];
        s.slot = slot_of_[(size_t) s.out];
        s.from_home = home_of_[(size_t) s.in];
        slot_of_[(size_t) s.in] = s.slot;
        slot_expert_[(size_t) s.slot] = s.in;
        slot_of_[(size_t) s.out] = -1;
        // the two-way bookkeeping with a pinned mapping: a promoted expert's home is freed, as publish frees it
        if (s.from_home >= 0 && !host_homes_ && !fixed_[(size_t) s.in]) {
            home_expert_[(size_t) s.from_home] = -1;
            home_of_[(size_t) s.in] = -1;
            free_homes_.push_back(s.from_home);
        }
        out.push_back(s);
    }
    return (int) ins_.size();
}

void HotLayerPlan::publish(const HotSwap& s) {
    if (s.in < 0 || s.in >= n_expert_ || s.out < 0 || s.out >= n_expert_ || !flight_[(size_t) s.in] || !flight_[(size_t) s.out])
        failf("publishing a swap (%d in, %d out) that is not in flight", s.in, s.out);
    const int32_t old = slot_of_[(size_t) s.out];
    if (old < 0 || slot_expert_[(size_t) s.slot] != -1) failf("swap %d in, %d out: slot %d or %d not as planned", s.in, s.out, s.slot, old);
    slot_of_[(size_t) s.in] = s.slot;
    slot_expert_[(size_t) s.slot] = s.in;
    slot_of_[(size_t) s.out] = -1;
    slot_expert_[(size_t) old] = -1;
    free_slots_.push_back(old);
    // host homes: the promoted expert keeps its home (its bytes stay valid there), the victim has its own; partial
    // host homes: a homed one keeps its home, any other frees it as the two-way path does
    if (s.from_home >= 0 && !host_homes_ && !fixed_[(size_t) s.in]) {
        home_expert_[(size_t) s.from_home] = -1;
        home_of_[(size_t) s.in] = -1;
        free_homes_.push_back(s.from_home);
    }
    if (s.to_home >= 0) {
        home_expert_[(size_t) s.to_home] = s.out;
        home_of_[(size_t) s.out] = s.to_home;
        --n_reserved_;
    }
    flight_[(size_t) s.in] = flight_[(size_t) s.out] = 0;
    --n_flight_;
}

void HotLayerPlan::abandon(const HotSwap& s) {
    if (s.in < 0 || s.in >= n_expert_ || s.out < 0 || s.out >= n_expert_ || !flight_[(size_t) s.in] || !flight_[(size_t) s.out])
        failf("abandoning a swap (%d in, %d out) that is not in flight", s.in, s.out);
    if (s.slot < 0 || s.slot >= n_slots() || slot_expert_[(size_t) s.slot] != -1)
        failf("abandoning swap %d in, %d out: slot %d not as planned", s.in, s.out, s.slot);
    free_slots_.push_back(s.slot);
    if (s.to_home >= 0) {
        if (s.to_home >= n_homes() || home_expert_[(size_t) s.to_home] != -1)
            failf("abandoning swap %d in, %d out: home %d not as planned", s.in, s.out, s.to_home);
        free_homes_.push_back(s.to_home);
        --n_reserved_;
    }
    flight_[(size_t) s.in] = flight_[(size_t) s.out] = 0;
    --n_flight_;
}

bool HotLayerPlan::is_static() const {
    if (n_flight_) return false;
    for (int32_t e = 0; e < n_expert_; ++e)
        if ((slot_of_[(size_t) e] >= 0) != (static_[(size_t) e] != 0)) return false;
    return true;
}

std::string HotLayerPlan::check() const {
    int resident = 0, flight = 0;
    for (int32_t e = 0; e < n_expert_; ++e) {
        const int32_t s = slot_of_[(size_t) e], h = home_of_[(size_t) e];
        if (s >= 0 && h >= 0 && !host_homes_ && !fixed_[(size_t) e]) return fmt("expert %lld is in slot %lld and in home %lld", e, s, h);
        if (s >= 0 && (s >= n_slots() || slot_expert_[(size_t) s] != e))
            return fmt("expert %lld says slot %lld, which holds %lld", e, s, s < n_slots() ? slot_expert_[(size_t) s] : -2);
        if (h >= 0 && (h >= n_homes() || home_expert_[(size_t) h] != e))
            return fmt("expert %lld says home %lld, which holds %lld", e, h, h < n_homes() ? home_expert_[(size_t) h] : -2);
        resident += s >= 0;
        flight += flight_[(size_t) e];
    }
    for (int s = 0; s < n_slots(); ++s) {
        const int32_t e = slot_expert_[(size_t) s];
        if (e >= 0 && slot_of_[(size_t) e] != s) return fmt("slot %lld holds expert %lld, which says %lld", s, e, slot_of_[(size_t) e]);
    }
    for (int k = 0; k < n_homes(); ++k) {
        const int32_t e = home_expert_[(size_t) k];
        if (e >= 0 && home_of_[(size_t) e] != k) return fmt("home %lld holds expert %lld, which says %lld", k, e, home_of_[(size_t) e]);
    }
    for (size_t k = 0; k < fixed_homes_.size(); ++k)
        if (fixed_homes_[k] != -2 && home_expert_[k] != fixed_homes_[k])
            return fmt("host home %lld holds expert %lld, init gave it %lld", (long long) k, home_expert_[k], fixed_homes_[k]);
    if (resident != n_static_) return fmt("%lld experts in the slab, the hot set has %lld", resident, n_static_);
    if (flight != 2 * n_flight_) return fmt("%lld experts marked in flight for %lld swaps", flight, n_flight_);
    // every free slot and home empty and listed once; the in-flight swaps hold the others
    std::vector<uint8_t> seen((size_t) n_slots(), 0);
    for (int32_t s : free_slots_) {
        if (s < 0 || s >= n_slots() || slot_expert_[(size_t) s] != -1 || seen[(size_t) s]) return fmt("free slot %lld is not free", s);
        seen[(size_t) s] = 1;
    }
    int empty_slots = 0;
    for (int32_t e : slot_expert_) empty_slots += e < 0;
    if (empty_slots != (int) free_slots_.size() + n_flight_)
        return fmt("%lld empty slots, %lld free and %lld in flight", empty_slots, (long long) free_slots_.size(), n_flight_);
    std::vector<uint8_t> seen_h((size_t) n_homes(), 0);
    for (int32_t k : free_homes_) {
        if (k < 0 || k >= n_homes() || home_expert_[(size_t) k] != -1 || seen_h[(size_t) k]) return fmt("free home %lld is not free", k);
        seen_h[(size_t) k] = 1;
    }
    int empty_homes = 0;
    for (int32_t e : home_expert_) empty_homes += e < 0;
    if (empty_homes != (int) free_homes_.size() + n_reserved_)
        return fmt("%lld empty homes, %lld free and %lld held for an in-flight victim", empty_homes,
                   (long long) free_homes_.size(), n_reserved_);
    return "";
}

}  // namespace strata::dsv41::engine
