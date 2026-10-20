// src/dsv41/engine/session.cpp - the resident conversation of include/strata/dsv41/engine/session.hpp.
#include "strata/dsv41/engine/session.hpp"

#include "strata/dsv41/engine/batch.hpp"
#include "strata/dsv41/engine/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace strata::dsv41::engine {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

bool starts_with(const std::vector<int32_t>& ids, const std::vector<int32_t>& prefix, size_t n) {
    return n <= ids.size() && n <= prefix.size() && std::equal(prefix.begin(), prefix.begin() + (std::ptrdiff_t) n, ids.begin());
}

// core/conversation_cache.hpp's match: the checkpoint's ids a strict prefix of the prompt's (no images here)
int64_t prefix_of(const strata::core::ConversationCheckpoint& c, const std::vector<int32_t>& ids) {
    static const std::vector<strata::core::ConversationImageKey> no_images;
    return strata::core::conversation_prefix(c, ids, no_images);
}

}  // namespace

const char* resume_kind_name(ResumeKind k) {
    switch (k) {
        case ResumeKind::Cold: return "cold";
        case ResumeKind::Live: return "live";
        case ResumeKind::PromptEnd: return "prompt-end";
    }
    return "?";
}

Session::Session(SessionBackend& b, std::vector<int32_t> eos) : b_(b), eos_(std::move(eos)) {
    slots_.resize((size_t) std::max(0, b_.checkpoints()));
    for (size_t k = 0; k < slots_.size(); ++k) slots_[k].id = (int) k;
}

std::string Session::check(const SessionRequest& r) const {
    if (r.ids.empty()) return "no prompt token ids";
    if (r.max_new < 1) return "max_new " + std::to_string(r.max_new) + ": at least 1 token must be generated";
    const int64_t nv = b_.n_vocab();
    auto bad_id = [&](const std::vector<int32_t>& v, const char* what) -> std::string {
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i] < 0 || v[i] >= nv)
                return std::string(what) + " id " + std::to_string(v[i]) + " at " + std::to_string(i) +
                       " is outside the vocabulary of " + std::to_string(nv);
        return "";
    };
    if (std::string e = bad_id(r.ids, "token"); !e.empty()) return e;
    // the prompt and every generated token but the last take a position
    const int64_t need = (int64_t) r.ids.size() + r.max_new - 1;
    if (need > b_.ctx())
        return std::to_string(r.ids.size()) + " prompt tokens and " + std::to_string(r.max_new) + " to generate need " +
               std::to_string(need) + " positions, the context is " + std::to_string(b_.ctx());
    if (r.chunk < 0 || r.chunk > b_.chunk())
        return "a prompt chunk of " + std::to_string(r.chunk) + " tokens: 1 to the engine's chunk of " +
               std::to_string(b_.chunk()) + " (0: the engine's)";
    if (!r.feed.empty()) {
        if ((int64_t) r.feed.size() < r.max_new - 1)
            return "feed: " + std::to_string(r.feed.size()) + " tokens for " + std::to_string(r.max_new - 1) + " decode steps";
        if (std::string e = bad_id(r.feed, "feed"); !e.empty()) return e;
    }
    return "";
}

ResumePlan Session::plan(const std::vector<int32_t>& ids) const {
    const size_t n = ids.size(), L = live_.size();
    ResumePlan best;
    if (L > 0 && n > L && starts_with(ids, live_, L)) best = {ResumeKind::Live, (int64_t) L, -1};
    // the checkpoints from the one used last: a longer prefix wins, an equal one keeps the live state or the later use
    std::vector<const ConversationSlot*> order;
    for (const ConversationSlot& s : slots_)
        if (s.held()) order.push_back(&s);
    std::sort(order.begin(), order.end(), [](const ConversationSlot* a, const ConversationSlot* b) { return a->ckpt.used > b->ckpt.used; });
    for (const ConversationSlot* s : order)
        if (const int64_t t = prefix_of(s->ckpt, ids); t > best.reuse) best = {ResumeKind::PromptEnd, t, s->id};
    return best;
}

bool Session::continues_last(const ResumePlan& p) const {
    return p.kind == ResumeKind::Live || (p.kind == ResumeKind::PromptEnd && p.slot == last_saved_);
}

int Session::pick_slot(const std::vector<int32_t>& ids, SessionResult& r) {
    int pick = -1;
    for (ConversationSlot& s : slots_)
        if (s.held() && s.ckpt.ids == ids) return s.id;
    for (ConversationSlot& s : slots_) {
        if (!s.held() || prefix_of(s.ckpt, ids) == 0) continue;
        s.ckpt.ids.clear();   // superseded: the new checkpoint holds this one's conversation further on
        ++r.superseded;
        if (pick < 0) pick = s.id;
    }
    if (pick >= 0) return pick;
    for (const ConversationSlot& s : slots_)
        if (!s.held()) return s.id;
    const ConversationSlot* lru = nullptr;
    for (const ConversationSlot& s : slots_)
        if (!lru || s.ckpt.used < lru->ckpt.used) lru = &s;
    r.evicted = lru->id;
    r.evicted_tokens = lru->position();
    ++evictions_;
    return lru->id;
}

SessionResult Session::run(const SessionRequest& r, const SessionEvents& ev) {
    SessionResult out;
    const auto t0 = Clock::now();
    out.plan = plan(r.ids);
    if (ev.resume) ev.resume(out.plan);
    switch (out.plan.kind) {
        case ResumeKind::Cold:
            b_.reset();
            live_.clear();
            break;
        case ResumeKind::PromptEnd: {
            ConversationSlot& s = slots_[(size_t) out.plan.slot];
            b_.restore_checkpoint(s.id);
            live_ = s.ckpt.ids;
            touch(s);
            break;
        }
        case ResumeKind::Live:
            break;
    }
    last_saved_ = -1;   // this request is the last one now, whatever it saves
    const int64_t n = (int64_t) r.ids.size();
    // a live resume whose one token left is the token the last
    // request emitted and never fed (a request stopped on the solo path and promoted into a batch slot with its prompt
    // and its tokens) feeds it with decode, as the request alone would have at that position; prefill(T=1) wrote that
    // position with the prefill path's numerics (RTX PRO 6000, 2026-10-17: the promoted chain left its alone run at
    // token 67). Any longer tail is read as a prompt, as alone.
    const int32_t unfed = unfed_;
    unfed_ = -1;
    const bool feed_unfed = out.plan.kind == ResumeKind::Live && n - out.plan.reuse == 1 && unfed >= 0 &&
                            r.ids[(size_t) out.plan.reuse] == unfed;
    std::vector<float> logits;
    int64_t pos = out.plan.reuse;
    const int64_t chunk = r.chunk > 0 ? r.chunk : b_.chunk();
    while (pos < n) {
        const int64_t T = std::min(chunk, n - pos);
        if (feed_unfed) b_.decode(unfed, logits);
        else b_.prefill(r.ids.data() + pos, T, logits);
        live_.insert(live_.end(), r.ids.begin() + pos, r.ids.begin() + pos + T);
        pos += T;
        if (ev.chunk) ev.chunk(pos, n, ms_since(t0));
        if (pos < n && ev.stop && ev.stop()) break;
        if (pos < n && ev.between) ev.between();
    }
    out.read = pos - out.plan.reuse;
    out.fed_unfed = feed_unfed;
    out.prompt_ms = ms_since(t0);
    if (pos < n) {   // stopped between two prompt chunks: the live state is the part read
        out.finish = "cancel";
        return out;
    }
    if (r.snapshot && !slots_.empty()) {
        const int k = pick_slot(r.ids, out);
        ConversationSlot& s = slots_[(size_t) k];
        b_.save_checkpoint(k);
        s.ckpt.ids = r.ids;
        touch(s);
        out.saved = last_saved_ = k;
    }
    if (ev.prompt_logits) ev.prompt_logits(logits);
    const auto t1 = Clock::now();
    for (int64_t s = 0;; ++s) {
        // greedy: the slots' eight-lane pick, the plain loop's token
        const int32_t tok = r.sampler ? r.sampler->pick(logits) : batch_argmax(logits);
        out.tokens.push_back(tok);
        if (++out.generated == 1) out.first_token_ms = ms_since(t0);
        if (ev.token) ev.token(tok);
        // the reply ends with tok emitted, not fed: the live state's next token (a forced run feeds the reference's,
        // so it keeps none)
        if (r.feed.empty()) unfed_ = tok;
        if (std::find(eos_.begin(), eos_.end(), tok) != eos_.end()) { out.finish = "stop"; break; }
        if (out.generated == r.max_new) { out.finish = "length"; break; }
        if (ev.stop && ev.stop()) { out.finish = "cancel"; break; }
        unfed_ = -1;
        const int32_t fed = r.feed.empty() ? tok : r.feed[(size_t) s];
        b_.decode(fed, logits);
        live_.push_back(fed);
    }
    out.decode_ms = ms_since(t1);
    return out;
}

}  // namespace strata::dsv41::engine
