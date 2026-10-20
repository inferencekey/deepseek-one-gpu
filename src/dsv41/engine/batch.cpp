// src/dsv41/engine/batch.cpp - the batch slots of include/strata/dsv41/engine/batch.hpp.
#include "strata/dsv41/engine/batch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <utility>

namespace strata::dsv41::engine {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

[[noreturn]] void failf(const char* what, int b) {
    std::fprintf(stderr, "dsv41 batch: slot %d: %s\n", b, what);
    std::exit(1);
}

}  // namespace

int32_t batch_argmax(const std::vector<float>& v) {
    const size_t n = v.size();
    if (n == 0 || std::isnan(v[0])) return 0;
    // lane j takes the ids j, j + 8, ...: its first id holding its largest value (strict >, so a NaN never enters
    // and a later equal value never replaces an earlier one); -1 while it holds none above -inf
    constexpr size_t kLanes = 8;
    float lv[kLanes];
    int64_t li[kLanes];
    for (size_t j = 0; j < kLanes; ++j) {
        lv[j] = -std::numeric_limits<float>::infinity();
        li[j] = -1;
    }
    size_t i = 1;
    for (; i + kLanes <= n; i += kLanes)
        for (size_t j = 0; j < kLanes; ++j)
            if (v[i + j] > lv[j]) {
                lv[j] = v[i + j];
                li[j] = (int64_t) (i + j);
            }
    for (size_t j = 0; i < n; ++i, ++j)
        if (v[i] > lv[j]) {
            lv[j] = v[i];
            li[j] = (int64_t) i;
        }
    // id 0 first, then each lane's best: the largest value, ties to the lower id (the plain loop's first maximum)
    int32_t best = 0;
    float bv = v[0];
    for (size_t j = 0; j < kLanes; ++j)
        if (li[j] >= 0 && (lv[j] > bv || (lv[j] == bv && li[j] < best))) {
            bv = lv[j];
            best = (int32_t) li[j];
        }
    return best;
}

BatchSlots::BatchSlots(BatchBackend& b, std::vector<int32_t> eos) : b_(b), eos_(std::move(eos)) {
    s_.resize((size_t) std::max(0, b_.batch_slots()));
}

int BatchSlots::active() const {
    int n = 0;
    for (const Slot& s : s_) n += s.on;
    return n;
}

std::string BatchSlots::fits(int64_t prompt, int64_t max_new) const {
    if (max_new < 2) return "max_new " + std::to_string(max_new) + ": nothing to decode after the first token";
    // the prompt and every generated token but the last take a position, as Session::check counts them
    const int64_t need = prompt + max_new - 1;
    if (need > b_.batch_ctx())
        return std::to_string(prompt) + " prompt tokens and " + std::to_string(max_new) + " to generate need " +
               std::to_string(need) + " positions, a batch slot holds " + std::to_string(b_.batch_ctx());
    return "";
}

void BatchSlots::admit(int b, Admission a) {
    if (b < 0 || b >= slots()) failf("admit: no such slot", b);
    Slot& s = s_[(size_t) b];
    if (s.on) failf("admit: the slot is busy", b);
    if (!fits(a.position, a.max_new).empty()) failf(("admit: " + fits(a.position, a.max_new)).c_str(), b);
    const auto t0 = Clock::now();
    b_.batch_admit(b);
    const double admit_ms = ms_since(t0);
    totals_.admit_ms += admit_ms;
    accounted_ms_ += admit_ms;
    s = Slot{};
    s.on = true;
    s.next = a.first;
    s.produced = 1;
    s.pos = a.position;
    s.a = std::move(a);
    ++totals_.admissions;
}

void BatchSlots::stop(int b) {
    if (b >= 0 && b < slots() && s_[(size_t) b].on) s_[(size_t) b].stop = true;
}

void BatchSlots::stalled(double ms) {
    if (!active()) return;
    accounted_ms_ += ms;
    ++totals_.stalls;
    totals_.stall_ms += ms;
    for (Slot& s : s_)
        if (s.on) s.stall_ms += ms;
}

void BatchSlots::end(int b, const char* finish, const Events& ev) {
    Slot& s = s_[(size_t) b];
    BatchDone d;
    d.slot = b;
    d.request = s.a.request;
    d.generated = s.produced;
    d.finish = finish;
    d.ms = ms_since(s.a.t0);
    d.steps = s.steps;
    d.shared_steps = s.shared_steps;
    d.rows = s.rows;
    d.stall_ms = s.stall_ms;
    d.deterministic = s.a.deterministic;
    d.reproducible = s.a.deterministic;   // placement-independent bits (batch.hpp)
    s = Slot{};
    if (ev.done) ev.done(d);
}

int BatchSlots::step(const Events& ev) {
    bool ended = false;
    for (int b = 0; b < slots(); ++b)
        if (s_[(size_t) b].on && s_[(size_t) b].stop) {
            end(b, "cancel", ev);
            ended = true;
        }
    std::vector<int> rows;
    std::vector<int32_t> tokens;
    for (int b = 0; b < slots(); ++b) {
        if (!s_[(size_t) b].on) continue;
        rows.push_back(b);
        tokens.push_back(s_[(size_t) b].next);
    }
    const int B = (int) rows.size();
    if (B == 0) {
        // the stopped slots' BDONE lines out now: with nothing left to decode no later step flushes them, and the
        // server frees a slot only at its BDONE (a measured session: eight cancelled at once held every slot for 300 s)
        if (ended && ev.flush) ev.flush();
        return 0;
    }
    const auto t0 = Clock::now();
    // the interval since the last step, less what the prompts read and the admissions in it took
    if (stepped_) {
        totals_.loop_ms += std::max(0.0, std::chrono::duration<double, std::milli>(t0 - last_end_).count() - accounted_ms_);
        ++totals_.gaps;
    }
    accounted_ms_ = 0;
    b_.batch_decode(rows.data(), tokens.data(), B, logits_);
    auto t1 = Clock::now();
    totals_.ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ++totals_.steps;
    totals_.rows += B;
    // the picks first, then the lines (a pick's time is the host's, the lines' the output's)
    std::vector<int32_t> picked((size_t) B);
    for (int i = 0; i < B; ++i) {
        const Slot& s = s_[(size_t) rows[(size_t) i]];
        const std::vector<float>& l = logits_[(size_t) i];
        picked[(size_t) i] = s.a.sampler ? s.a.sampler->pick(l) : batch_argmax(l);
    }
    auto t2 = Clock::now();
    totals_.pick_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    for (int i = 0; i < B; ++i) {
        const int b = rows[(size_t) i];
        Slot& s = s_[(size_t) b];
        ++s.steps;
        s.rows += B;
        s.shared_steps += B > 1;
        ++s.pos;
        const int32_t tok = picked[(size_t) i];
        ++s.produced;
        if (ev.token) ev.token(b, tok);
        if (std::find(eos_.begin(), eos_.end(), tok) != eos_.end()) end(b, "stop", ev);
        else if (s.produced >= s.a.max_new) end(b, "length", ev);
        else s.next = tok;
    }
    if (ev.flush) ev.flush();
    last_end_ = Clock::now();
    stepped_ = active() > 0;
    totals_.out_ms += std::chrono::duration<double, std::milli>(last_end_ - t2).count();
    return B;
}

}  // namespace strata::dsv41::engine
