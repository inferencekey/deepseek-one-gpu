// tests/dsv41/engine/session_test.cpp - how the Session feeds
// a promoted request's one token left (session.hpp), GPU free, no model.
//
//   dsv41_engine_session_test          (tools/dsv41/engine/session_test.sh builds it with the host compiler too)
//
// The backend records every call (prefill of n tokens, decode of a token) and its numerics depend on the path: each
// position folds its token and the call that wrote it (prefill or decode) into a running hash, and the logits are
// one-hot at the hash's id. So a position written by prefill(T=1) where the request alone wrote it by decode changes
// every token after it, as an RTX PRO 6000 session's promoted chain did (2026-10-17, token 67), only at once.
//
// Checks (PASS or FAIL lines, exit 1 on any FAIL):
//   1. the server's promotion (serve/server.py generate_batched): a request stopped after k tokens (k = 1, 2, 5) on
//      the solo path, then run again with its prompt and its tokens (BGEN's GEN 1, then on alone for the rest): the
//      second run is a live resume whose one token left is fed by decode, no prefill call at all, and its tokens are
//      the request alone's from token k + 1 on;
//   2. the token picked by the promotion's own run (BGEN's first token, emitted, not fed) is the next one fed by
//      decode when the request goes on alone with it (the same rule after a "length" end);
//   3. a tail of more than one token (a conversation's next turn) is read by prefill, as alone;
//   4. a one-token tail that is not the token the last request emitted, and one after a run that ended inside its
//      prompt (it emitted nothing, so an earlier request's unfed token is no longer the live state's), are read by
//      prefill.
#include "strata/dsv41/engine/session.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

constexpr int64_t kVocab = 97;

std::string ids_text(const std::vector<int32_t>& v) {
    std::string s;
    for (int32_t t : v) s += (s.empty() ? "" : " ") + std::to_string(t);
    return s;
}

// Records the calls; the state of a position is a hash of every token before it and of the call that wrote it.
class RecordingBackend final : public e::SessionBackend {
public:
    std::vector<std::string> calls;

    int64_t ctx() const override { return 256; }
    int64_t chunk() const override { return 64; }
    int64_t n_vocab() const override { return kVocab; }
    void reset() override { hash_.assign(1, 1469598103u); }
    int checkpoints() const override { return 0; }
    void save_checkpoint(int) override {}
    void restore_checkpoint(int) override {}
    void prefill(const int32_t* t, int64_t n, std::vector<float>& logits) override {
        calls.push_back("prefill " + std::to_string(n));
        for (int64_t i = 0; i < n; ++i) push(t[i], 'p');
        one_hot(logits);
    }
    void decode(int32_t t, std::vector<float>& logits) override {
        calls.push_back("decode " + std::to_string(t));
        push(t, 'd');
        one_hot(logits);
    }

private:
    std::vector<uint32_t> hash_{1469598103u};
    void push(int32_t t, char path) {
        uint32_t h = hash_.back();
        for (uint32_t x : {(uint32_t) t, (uint32_t) path}) h = (h ^ x) * 16777619u;
        hash_.push_back(h);
    }
    void one_hot(std::vector<float>& logits) const {
        logits.assign((size_t) kVocab, 0.f);
        logits[(size_t) ((hash_.back() >> 7) % (uint32_t) kVocab)] = 1.f;
    }
};

const std::vector<int32_t> kPrompt = {11, 12, 13, 14, 15, 16, 17};

struct Run {
    e::SessionResult res;
    std::vector<std::string> calls;
};

// one request on the session; stop_after > 0: STOP once that many tokens were emitted (the server's STOP to promote);
// chunk > 0 with stop_chunks > 0: STOP after that many prompt chunks
Run run(e::Session& s, RecordingBackend& b, const std::vector<int32_t>& ids, int64_t max_new, int64_t stop_after = 0,
        int64_t chunk = 0, int64_t stop_chunks = 0) {
    e::SessionRequest r;
    r.ids = ids;
    r.max_new = max_new;
    r.snapshot = false;
    r.chunk = chunk;
    const std::string why = s.check(r);
    if (!why.empty()) {
        std::printf("FAIL request refused: %s\n", why.c_str());
        ++failures;
        return {};
    }
    int64_t emitted = 0, chunks = 0;
    e::SessionEvents ev;
    ev.token = [&](int32_t) { ++emitted; };
    ev.chunk = [&](int64_t, int64_t, double) { ++chunks; };
    ev.stop = [&] { return (stop_after > 0 && emitted >= stop_after) || (stop_chunks > 0 && chunks >= stop_chunks); };
    b.calls.clear();
    Run out;
    out.res = s.run(r, ev);
    out.calls = b.calls;
    return out;
}

std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

std::vector<int32_t> slice(const std::vector<int32_t>& v, size_t from, size_t to) {
    return std::vector<int32_t>(v.begin() + (std::ptrdiff_t) from, v.begin() + (std::ptrdiff_t) to);
}

bool no_prefill(const std::vector<std::string>& calls) {
    for (const std::string& c : calls)
        if (c.rfind("prefill", 0) == 0) return false;
    return true;
}

}  // namespace

int main() {
    constexpr int64_t kMax = 12;
    RecordingBackend alone_b;
    e::Session alone_s(alone_b, {});
    const std::vector<int32_t> alone = run(alone_s, alone_b, kPrompt, kMax).res.tokens;
    expect(alone.size() == (size_t) kMax, "the request alone: " + std::to_string(kMax) + " tokens (" + ids_text(alone) + ")");

    // 1, 2: stopped after k on the solo path, promoted with its prompt and its tokens
    for (int64_t k : {1, 2, 5}) {
        RecordingBackend b;
        e::Session s(b, {});
        const Run solo = run(s, b, kPrompt, kMax, k);
        const std::vector<int32_t> out = solo.res.tokens;
        expect(solo.res.finish == "cancel" && out == slice(alone, 0, (size_t) k),
               "stopped after " + std::to_string(k) + ": the alone run's first " + std::to_string(k) + " tokens");
        const std::vector<int32_t> ids = cat(kPrompt, out);
        const Run bgen = run(s, b, ids, 1);
        const std::string want = "decode " + std::to_string(out.back());
        expect(bgen.res.plan.kind == e::ResumeKind::Live && bgen.res.plan.reuse == (int64_t) ids.size() - 1 &&
                   bgen.res.read == 1 && bgen.calls.size() == 1 && bgen.calls[0] == want && no_prefill(bgen.calls) &&
                   bgen.res.fed_unfed,
               "promoted after " + std::to_string(k) + " (the BGEN of its prompt and tokens): live resume of " +
                   std::to_string(ids.size() - 1) + ", its 1 token left fed by " + want + ", no prefill call (saw: " +
                   (bgen.calls.empty() ? std::string("none") : bgen.calls[0]) + ")");
        expect(bgen.res.tokens.size() == 1 && bgen.res.tokens[0] == alone[(size_t) k],
               "promoted after " + std::to_string(k) + ": its first token in the slot is the alone run's token " +
                   std::to_string(k + 1));
        // on with the BGEN's own token (emitted, not fed: a "length" end), as the slot goes on from it
        const std::vector<int32_t> ids2 = cat(ids, bgen.res.tokens);
        const Run rest = run(s, b, ids2, kMax - (int64_t) ids2.size() + (int64_t) kPrompt.size());
        expect(!rest.calls.empty() && rest.calls[0] == "decode " + std::to_string(bgen.res.tokens[0]) &&
                   no_prefill(rest.calls) && rest.res.fed_unfed &&
                   cat(ids2, rest.res.tokens) == cat(kPrompt, alone),
               "promoted after " + std::to_string(k) + ", then on: the token picked last fed by decode, every token the "
               "alone run's (" + ids_text(slice(cat(ids2, rest.res.tokens), kPrompt.size(), kPrompt.size() + (size_t) kMax)) + ")");
    }

    // 3: a conversation's next turn (a tail of 3: the last reply's last token and two of the user's) is read by prefill
    {
        RecordingBackend b;
        e::Session s(b, {});
        const Run first = run(s, b, kPrompt, 4);
        const Run next = run(s, b, cat(cat(kPrompt, first.res.tokens), {40, 41}), 3);
        expect(first.res.finish == "length" && next.res.plan.kind == e::ResumeKind::Live && !next.calls.empty() &&
                   next.calls[0] == "prefill 3" && !next.res.fed_unfed,
               "a next turn (3 tokens left after a live resume) is read by one prefill of 3, as alone");
    }

    // 4: one token left that is not the last request's unfed token; one after a run that ended inside its prompt
    {
        RecordingBackend b;
        e::Session s(b, {});
        const Run solo = run(s, b, kPrompt, kMax, 2);
        const int32_t other = (solo.res.tokens[1] + 1) % (int32_t) kVocab;
        const Run next = run(s, b, cat(cat(kPrompt, {solo.res.tokens[0]}), {other}), 2);
        expect(next.res.plan.kind == e::ResumeKind::Live && !next.calls.empty() && next.calls[0] == "prefill 1" &&
                   !next.res.fed_unfed,
               "one token left that the last request did not emit (" + std::to_string(other) + "): read by prefill");
    }
    {
        RecordingBackend b;
        e::Session s(b, {});
        const Run solo = run(s, b, kPrompt, kMax, 2);
        // the next request reads the unfed token as the first of its prompt's chunks and is stopped there
        const std::vector<int32_t> ids = cat(cat(kPrompt, solo.res.tokens), {50, 51});
        const Run cut = run(s, b, ids, 2, 0, 1, 1);
        // one token left equal to the solo run's unfed token: that run is no longer the last, so it is read by prefill
        const Run next = run(s, b, cat(cat(kPrompt, solo.res.tokens), {solo.res.tokens[1]}), 2);
        expect(cut.res.finish == "cancel" && cut.res.tokens.empty() && next.res.plan.kind == e::ResumeKind::Live &&
                   !next.calls.empty() && next.calls[0] == "prefill 1" && !next.res.fed_unfed,
               "after a request stopped inside its prompt (nothing emitted), one token left is read by prefill, even "
               "the token an earlier request left unfed");
    }

    std::printf("dsv41_engine_session_test: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
