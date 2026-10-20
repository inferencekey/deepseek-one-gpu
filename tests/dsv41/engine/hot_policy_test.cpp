// tests/dsv41/engine/hot_policy_test.cpp - the per-request hot set's planner (include/strata/dsv41/engine/
// hot_policy.hpp), GPU free.
//
//   dsv41_engine_hot_policy_test                  (a) the planner's rules and invariants on small layers
//   dsv41_engine_hot_policy_test --replay TRACE --hot-set FILE [--policy P] [--swaps N] [--spare N] [--half-life H]
//                                [--margin MUL:ADD] [--mapping-pinned]
//                                                 (b) a routing trace (dsv41-engine --route-trace) replayed per layer
//
// (a) Layers of 16 experts (4 in the slab, 2 spare slots; 8 warm in homes, 2 spare homes; the last 4 cold in an
// unpinned mapping, or the mapping pinned), each boundary as the engine's with --hot-check: the swaps planned at the
// last boundary published, the planner's check(), then the next swaps. Checked: swaps at every token with the churn
// margin (count(in) > 0 x count(out) - 1: always), at most --hot-swaps per token and `spare` in flight; a cold expert of
// an unpinned mapping never promoted; a victim going back to a pinned mapping without a home; the prompt policy's
// target (the prompt's most used experts, ties by the static order) reached and kept; adaptive's hysteresis (a count
// of 1 against 0 swaps nothing at 1.5 x + 1, 3 against 1 does); the static reset in rounds of the spare slots; copy
// only (HotConfig::copy_only, the engine smoke's copy-cost run): every swap abandoned at the next boundary, the slab
// the hot set's at every boundary and --hot-swaps swaps planned at every token, homes or a pinned mapping; two turns
// of one conversation (test_two_turn): turn 2 resumed against fresh and against turn 1's decode run on; the
// deterministic mode's start and boundary rules (test_deterministic_rules). Host homes (test_host_homes,
// test_restore_order): with host homes the same churn as the spare homes' swap for swap (the same experts in and out
// at every token), no victim copied down, every home holding the expert init gave it, a promoted expert keeping its
// home; restore_in_place back to the hot set's slab at once with copies up only from each expert's own home; HotRestore's
// order (each layer waited for once, after its record) and its refusal of a layer computing before its restore is
// queued (a child run of this test with --restore-before-record must exit 1 naming it).
// (test_partial_homes): with only some slab experts homed the same churn as the spare homes' swap for swap, a homed
// victim dropped and any other copied down into a free home, the homed experts' homes never moving, the reset in rounds
// back at the slab; restore_in_place refusing such a layer (a child run with --partial-restore must exit 1).
// (test_merge_prompt): a prompt read beside the batch merged into its counts at one row's decode mass, once.
// (test_promoted_batch): a batch opened by a promoted request (its one token fed by decode, no prefill) swaps and
// misses about as often a row as the request alone, through Forward's HotPrompt (tools/dsv41/engine/
// hot_policy_test.sh builds this test with the host compiler too, with and without the fix).
//
// (b) Each request of the trace from its R line: begin_request with its policy (or --policy), the reset back to the
// static slab when the line asks for it (in rounds of the spare slots, as the engine's), its P lines counted, then per
// layer its D lines token by token: the token's pairs are hits when the slab holds their experts, the counts take the
// token, then the boundary (the last swaps published, the next planned: copies are taken as done by the next token,
// the engine's --hot-check). Prints per request and layer the hits of its decode pairs, then the totals, in the format
// tools/dsv41/engine/hot_policy_sim.py prints, which verify.sh diffs against this.
#include "strata/dsv41/engine/hot_policy.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>

namespace e = strata::dsv41::engine;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "dsv41_engine_hot_policy_test: %s\n", m.c_str());
    std::exit(2);
}

e::HotLayerPlan::Layout small_layout(bool mapping_pinned) {
    e::HotLayerPlan::Layout l;
    l.n_expert = 16;
    l.vram = {0, 1, 2, 3};
    l.spare = 2;
    if (!mapping_pinned) {
        l.warm = {4, 5, 6, 7, 8, 9, 10, 11};
        l.spare_homes = 2;
    }
    l.mapping_pinned = mapping_pinned;
    return l;
}

// one boundary: the swaps in flight published, the check, the next ones planned
void boundary(e::HotLayerPlan& p, const e::HotConfig& c, std::vector<e::HotSwap>& flight, const std::string& tag) {
    for (const e::HotSwap& s : flight) p.publish(s);
    flight.clear();
    const std::string why = p.check();
    if (!why.empty()) die(tag + ": " + why);
    p.plan(c, flight);
    const std::string why2 = p.check();
    if (!why2.empty()) die(tag + " (planned): " + why2);
}

void test_churn(bool mapping_pinned) {
    const std::string tag = mapping_pinned ? "churn, pinned mapping" : "churn, homes";
    e::HotLayerPlan p;
    p.init(small_layout(mapping_pinned));
    e::HotConfig c;
    c.margin_mul = 0;
    c.margin_add = -1;
    p.begin_request(e::HotPolicy::Adaptive, false);
    std::mt19937 rng(1);
    for (int i = 0; i < 40; ++i) {
        const int32_t id = (int32_t) (rng() % 16);
        p.count_prompt(&id, 1);
    }
    p.prompt_done(c);
    std::vector<e::HotSwap> flight;
    p.plan(c, flight);
    int64_t swaps = 0, tokens = 0, cold_promoted = 0, homeless = 0, every = 0;
    const double decay = std::pow(2.0, -1.0 / c.half_life);
    for (int t = 0; t < 200; ++t) {
        const int32_t ids[3] = {(int32_t) (rng() % 16), (int32_t) (rng() % 16), (int32_t) (rng() % 16)};
        p.count_token(ids, 3, decay);
        for (const e::HotSwap& s : flight) {
            if (!mapping_pinned && s.in >= 12) ++cold_promoted;
            if (mapping_pinned != (s.to_home < 0)) ++homeless;
        }
        swaps += (int64_t) flight.size();
        every += flight.size() == (size_t) c.max_swaps;
        ++tokens;
        boundary(p, c, flight, tag);
    }
    expect(every == tokens, tag + ": " + std::to_string(every) + " of " + std::to_string(tokens) +
                                " tokens with --hot-swaps " + std::to_string(c.max_swaps) + " swaps in flight (" +
                                std::to_string(swaps) + " swaps)");
    expect(cold_promoted == 0, tag + ": no cold expert of an unpinned mapping promoted (" + std::to_string(cold_promoted) + ")");
    expect(homeless == 0, tag + (mapping_pinned ? ": every victim back to its pinned mapping, no home" : ": every victim into a home"));
}

void test_prompt() {
    e::HotLayerPlan p;
    p.init(small_layout(false));
    e::HotConfig c;
    p.begin_request(e::HotPolicy::Prompt, false);
    // the prompt uses 9 most, then 5 and 6 (once each) and 1: the target is {9, 1, 5, 6} (1 first by count 2, then 5
    // and 6 by count, the ties broken by the static order: 5 before 6 before the unused ones)
    const int32_t ids[] = {9, 9, 9, 1, 1, 5, 6, 13};
    p.count_prompt(ids, 8);
    p.prompt_done(c);
    std::vector<e::HotSwap> flight;
    p.plan(c, flight);
    for (int t = 0; t < 4; ++t) boundary(p, c, flight, "prompt");
    bool want = true;
    for (int32_t x : {9, 1, 5, 6}) want = want && p.slot_of(x) >= 0;
    expect(want && p.resident() == 4, "prompt: the slab holds the prompt's target {9, 1, 5, 6} (13 is cold: not promoted)");
    expect(p.in_flight() == 0, "prompt: no swap once the target is resident");
}

void test_hysteresis() {
    e::HotLayerPlan p;
    p.init(small_layout(false));
    e::HotConfig c;   // 1.5 x + 1
    p.begin_request(e::HotPolicy::Adaptive, false);
    const int32_t one = 3;   // the prompt's target: {3, 0, 1, 2} (3 by count, then the static order): no target swap
    p.count_prompt(&one, 1);
    p.prompt_done(c);
    std::vector<e::HotSwap> flight;
    p.plan(c, flight);
    expect(flight.empty(), "adaptive: the prompt's target is the slab already, no swap");
    const double decay = 1.0;   // no decay: the counts are exact
    const int32_t a[1] = {8};
    p.count_token(a, 1, decay);   // 8: 1 against the slab's least (0, 1, 2: 0)
    boundary(p, c, flight, "hysteresis");
    expect(flight.empty(), "adaptive: count 1 against 0 swaps nothing (1 > 1.5 x 0 + 1 is false)");
    p.count_token(a, 1, decay);
    boundary(p, c, flight, "hysteresis");
    expect(flight.size() == 1 && flight[0].in == 8 && flight[0].out == 2,
           "adaptive: count 2 against 0 swaps 8 in for 2 (the least counted, the latest in the static order)");
}

// a prompt read beside the batch merges its counts into the batch's at one row's
// decode mass, keeping the target and the phase; the batch's own prompt is not merged again
void test_merge_prompt() {
    e::HotLayerPlan p;
    p.init(small_layout(false));
    e::HotConfig c;
    p.begin_request(e::HotPolicy::Adaptive, false);
    const int32_t own = 3;
    p.count_prompt(&own, 1);
    p.prompt_done(c);
    std::vector<e::HotSwap> flight;
    p.plan(c, flight);
    const double decay = 1.0;
    const int32_t a[2] = {0, 1};
    p.count_token(a, 2, decay);   // the batch's counts: 3 (its prompt) 1, 0 and 1 one each
    const int32_t read[4] = {8, 8, 8, 9};   // the read beside: 8 three times, 9 once, merged at a mass of 8
    p.count_prompt(read, 4);
    p.merge_prompt(8.0);
    const bool counts = p.score(8) == 6.0 && p.score(9) == 2.0 && p.score(3) == 1.0 && p.score(0) == 1.0 &&
                        p.score(1) == 1.0 && p.score(2) == 0.0;
    p.merge_prompt(8.0);   // nothing read since: nothing merged
    const bool once = p.score(8) == 6.0 && p.score(9) == 2.0 && p.score(3) == 1.0;
    boundary(p, c, flight, "merge");
    // 8 (6) beats the least counted slab expert, 2 (0), by 1.5 x + 1; 9 (2) beats nothing else but 2 is taken: 9 > 1.5
    // x 1 + 1 is false
    expect(counts && once && flight.size() == 1 && flight[0].in == 8 && flight[0].out == 2,
           "merge_prompt: the read's counts scaled to the mass (8: 6, 9: 2) added to the batch's, the batch's own prompt "
           "not again, then the LFU swaps 8 in for 2 at once");
    e::HotLayerPlan q;
    q.init(small_layout(false));
    q.begin_request(e::HotPolicy::Static, false);
    q.count_prompt(read, 4);
    q.prompt_done(c);
    q.merge_prompt(8.0);
    expect(q.score(8) == 0.0, "merge_prompt: nothing for the static policy");
}

void test_static_reset() {
    e::HotLayerPlan p;
    p.init(small_layout(false));
    e::HotConfig c;
    c.margin_mul = 0;
    c.margin_add = -1;
    p.begin_request(e::HotPolicy::Adaptive, false);
    p.prompt_done(c);
    std::vector<e::HotSwap> flight;
    std::mt19937 rng(3);
    for (int t = 0; t < 30; ++t) {
        const int32_t ids[2] = {(int32_t) (4 + rng() % 8), (int32_t) (4 + rng() % 8)};
        p.count_token(ids, 2, 0.9);
        boundary(p, c, flight, "reset");
    }
    for (const e::HotSwap& s : flight) p.publish(s);
    flight.clear();
    int moved = 0;
    for (int32_t x = 0; x < 4; ++x) moved += p.slot_of(x) < 0;
    int rounds = 0;
    while (!p.is_static() && rounds < 10) {
        p.plan_static(c.spare, flight);
        for (const e::HotSwap& s : flight) p.publish(s);
        flight.clear();
        ++rounds;
        if (!p.check().empty()) die("reset: " + p.check());
    }
    expect(moved > 0 && p.is_static() && rounds == (moved + c.spare - 1) / c.spare,
           "static reset: " + std::to_string(moved) + " of the hot set's 4 experts back in " + std::to_string(rounds) +
               " rounds of at most " + std::to_string(c.spare) + " swaps");
}

void test_copy_only(bool mapping_pinned) {
    const std::string tag = mapping_pinned ? "copy only, pinned mapping" : "copy only, homes";
    e::HotLayerPlan p;
    p.init(small_layout(mapping_pinned));
    e::HotConfig c;
    c.margin_mul = 0;
    c.margin_add = -1;
    c.copy_only = true;
    p.begin_request(e::HotPolicy::Adaptive, false);
    std::mt19937 rng(5);
    for (int i = 0; i < 40; ++i) {
        const int32_t id = (int32_t) (rng() % 16);
        p.count_prompt(&id, 1);
    }
    p.prompt_done(c);
    std::vector<e::HotSwap> flight;
    p.plan(c, flight);
    int64_t every = 0, still = 0, tokens = 0;
    const double decay = std::pow(2.0, -1.0 / c.half_life);
    for (int t = 0; t < 100; ++t) {
        const int32_t ids[3] = {(int32_t) (rng() % 16), (int32_t) (rng() % 16), (int32_t) (rng() % 16)};
        p.count_token(ids, 3, decay);
        every += flight.size() == (size_t) c.max_swaps;
        for (const e::HotSwap& x : flight) p.abandon(x);
        flight.clear();
        if (const std::string why = p.check(); !why.empty()) die(tag + ": " + why);
        still += p.is_static();
        ++tokens;
        p.plan(c, flight);
        if (const std::string why = p.check(); !why.empty()) die(tag + " (planned): " + why);
    }
    expect(every == tokens, tag + ": " + std::to_string(every) + " of " + std::to_string(tokens) + " tokens with --hot-swaps " +
                                std::to_string(c.max_swaps) + " swaps in flight");
    expect(still == tokens, tag + ": the slab the hot set's at " + std::to_string(still) + " of " + std::to_string(tokens) +
                                " boundaries (nothing published)");
}

// Two turns of one conversation (a resumed turn keeps the counts its conversation left): a
// layer of 64 experts (16 in the slab, 2 spare slots; 48 warm in homes, 2 spare homes), the routing of both turns from
// one topic (Zipf 1.1 over an order that puts 24 warm experts first, 3 picks per token); turn 1 a 400-token prompt
// and 200 decode tokens from the hot set's slab, turn 2 a 17-token suffix (51 pairs) and 200 decode tokens. Turn 2
// three ways from turn 1's end: resumed (begin_request with resume), fresh (without: the planner before the resume
// rule) and continued (no new request: turn 1's decode goes on over the suffix's tokens). The resumed turn must hit at
// least as often as the fresh one and swap at most a quarter of its swaps; against continued, which it follows but
// not swap for swap (the suffix counted undecayed, its swaps planned at the prompt's end), it may trail by 1% of the
// pairs at most (here 1 hit of 600).
struct TurnStats {
    int64_t hits = 0, pairs = 0, swaps = 0;
};

TurnStats decode_turn(e::HotLayerPlan& p, const e::HotConfig& c, std::vector<e::HotSwap>& flight,
                      const std::vector<std::vector<int32_t>>& tokens, double decay, bool count, const std::string& tag) {
    TurnStats t;
    t.swaps = (int64_t) flight.size();
    for (const std::vector<int32_t>& ids : tokens) {
        if (count)
            for (int32_t x : ids) t.hits += p.slot_of(x) >= 0;
        t.pairs += (int64_t) ids.size();
        p.count_token(ids.data(), (int) ids.size(), decay);
        boundary(p, c, flight, tag);
        t.swaps += (int64_t) flight.size();
    }
    return t;
}

void test_two_turn() {
    e::HotConfig c;
    const double decay = std::pow(2.0, -1.0 / c.half_life);
    e::HotLayerPlan::Layout lay;
    lay.n_expert = 64;
    for (int32_t x = 0; x < 16; ++x) lay.vram.push_back(x);
    for (int32_t x = 16; x < 64; ++x) lay.warm.push_back(x);
    lay.spare = lay.spare_homes = 2;
    std::vector<int32_t> order;
    for (int32_t x = 20; x < 44; ++x) order.push_back(x);
    for (int32_t x = 0; x < 64; ++x)
        if (x < 20 || x >= 44) order.push_back(x);
    std::vector<double> w(64, 0.0);
    for (size_t i = 0; i < order.size(); ++i) w[(size_t) order[i]] = 1.0 / std::pow((double) i + 1, 1.1);
    std::mt19937 rng(7);
    std::discrete_distribution<int32_t> pick(w.begin(), w.end());
    auto tokens = [&](int n) {
        std::vector<std::vector<int32_t>> out((size_t) n);
        for (std::vector<int32_t>& ids : out)
            while (ids.size() < 3) {
                const int32_t x = pick(rng);
                if (std::find(ids.begin(), ids.end(), x) == ids.end()) ids.push_back(x);
            }
        return out;
    };
    const auto prompt1 = tokens(400), decode1 = tokens(200), suffix = tokens(17), decode2 = tokens(200);
    e::HotLayerPlan p;
    p.init(lay);
    p.begin_request(e::HotPolicy::Adaptive, false);
    for (const std::vector<int32_t>& ids : prompt1) p.count_prompt(ids.data(), (int64_t) ids.size());
    p.prompt_done(c);
    std::vector<e::HotSwap> flight;
    p.plan(c, flight);
    decode_turn(p, c, flight, decode1, decay, false, "two turns, turn 1");
    for (const e::HotSwap& s : flight) p.publish(s);   // the engine drains the batch at the next prefill
    flight.clear();
    TurnStats got[2];
    for (int resume = 0; resume < 2; ++resume) {
        e::HotLayerPlan q = p;
        q.begin_request(e::HotPolicy::Adaptive, resume == 1);
        for (const std::vector<int32_t>& ids : suffix) q.count_prompt(ids.data(), (int64_t) ids.size());
        q.prompt_done(c);
        std::vector<e::HotSwap> f;
        q.plan(c, f);
        got[resume] = decode_turn(q, c, f, decode2, decay, true, resume ? "two turns, resumed" : "two turns, fresh");
    }
    e::HotLayerPlan q = p;
    std::vector<e::HotSwap> f;
    q.plan(c, f);
    const TurnStats over = decode_turn(q, c, f, suffix, decay, false, "two turns, continued over the suffix");
    TurnStats cont = decode_turn(q, c, f, decode2, decay, true, "two turns, continued");
    cont.swaps += over.swaps;
    const TurnStats &fresh = got[0], &res = got[1];
    auto line = [](const char* n, const TurnStats& t) {
        char b[160];
        std::snprintf(b, sizeof b, "%s %lld of %lld hits, %lld swaps", n, (long long) t.hits, (long long) t.pairs,
                      (long long) t.swaps);
        return std::string(b);
    };
    const std::string all = line("resumed", res) + "; " + line("fresh", fresh) + "; " + line("continued", cont);
    expect(res.hits >= fresh.hits && 100 * res.hits >= 100 * cont.hits - res.pairs,
           "two turns: turn 2 resumed hits at least as often as fresh, within 1% of the pairs of continued (" + all + ")");
    expect(fresh.swaps > 0 && 4 * res.swaps <= fresh.swaps,
           "two turns: turn 2 resumed swaps at most a quarter of fresh's (" + std::to_string(res.swaps) + " against " +
               std::to_string(fresh.swaps) + ")");
}

// (a batched decode regression, hot_policy.hpp HotPrompt): a batch opened by a promoted request
// against the same topic's request alone, through the calls the engine makes and Forward's HotPrompt between them, on
// test_two_turn's layer and topic. Alone (B = 1): begin_request in the deterministic mode's start (reset, zero
// counts), a 400-token prompt read by prefill, then 300 decode tokens. Promoted (B = 4): begin_request the same, its
// one token left fed by decode (no prefill call), the decode call that ends its
// prompt, 20 steps of that one row; then three 400-token prompts read beside the slots, each followed by 20 steps of
// one row more (the first merges the read); then 300 steps of 4 rows, measured. Static: the 4-row steps' rows against
// the hot set's slab. The work of a step is its misses (the CPU jobs' rows, decode_miss_jobs' pairs outside the slab).
// Checked: the batch swaps, its 4-row steps' misses a row are at most 1.25 x the alone run's, and at most half the
// static slab's. Before this fix the promoted decode left the prompt open: no swap, the static slab's misses (the
// 2026-10-19 session's 0.00 swaps and 0.4031 hit rate a step).
struct BatchWork {
    int64_t steps = 0, rows = 0, misses = 0, swaps = 0;
};

void test_promoted_batch() {
    e::HotConfig c;
    const double decay = std::pow(2.0, -1.0 / c.half_life);
    constexpr int K = 3;
    const double mass = (double) K / (1.0 - decay);   // ExpertEngine::merge_prompt's
    e::HotLayerPlan::Layout lay;
    lay.n_expert = 64;
    for (int32_t x = 0; x < 16; ++x) lay.vram.push_back(x);
    for (int32_t x = 16; x < 64; ++x) lay.warm.push_back(x);
    lay.spare = lay.spare_homes = 2;
    std::vector<int32_t> order;
    for (int32_t x = 20; x < 44; ++x) order.push_back(x);
    for (int32_t x = 0; x < 64; ++x)
        if (x < 20 || x >= 44) order.push_back(x);
    std::vector<double> w(64, 0.0);
    for (size_t i = 0; i < order.size(); ++i) w[(size_t) order[i]] = 1.0 / std::pow((double) i + 1, 1.1);
    std::mt19937 rng(11);
    std::discrete_distribution<int32_t> pick(w.begin(), w.end());
    auto row = [&]() {
        std::vector<int32_t> ids;
        while (ids.size() < (size_t) K) {
            const int32_t x = pick(rng);
            if (std::find(ids.begin(), ids.end(), x) == ids.end()) ids.push_back(x);
        }
        return ids;
    };
    auto prompt = [&](e::HotLayerPlan& p, e::HotPrompt& hp, bool beside) {
        for (int t = 0; t < 400; ++t) {
            const std::vector<int32_t> ids = row();
            p.count_prompt(ids.data(), K);
        }
        hp.prefill(beside);
    };
    // one decode call of B rows (Forward::decode or decode_batch): the prompt's end first, the rows' hits on the slab,
    // their pairs counted at once (one decay a step), then the boundary
    auto step = [&](e::HotLayerPlan& p, e::HotPrompt& hp, std::vector<e::HotSwap>& flight, int B, BatchWork* work,
                    const std::string& tag) {
        const e::HotPromptEnd end = hp.end();
        if (end == e::HotPromptEnd::Done) {
            p.prompt_done(c);
            p.plan(c, flight);
        } else if (end == e::HotPromptEnd::Merge) {
            p.merge_prompt(mass);
        }
        std::vector<int32_t> ids;
        for (int b = 0; b < B; ++b) {
            const std::vector<int32_t> r = row();
            ids.insert(ids.end(), r.begin(), r.end());
        }
        if (work) {
            for (int32_t x : ids) work->misses += p.slot_of(x) < 0;
            ++work->steps;
            work->rows += B;
        }
        p.count_token(ids.data(), (int) ids.size(), decay);
        boundary(p, c, flight, tag);
        if (work) work->swaps += (int64_t) flight.size();
    };
    // alone
    BatchWork alone;
    {
        e::HotLayerPlan p;
        p.init(lay);
        e::HotPrompt hp;
        p.begin_request(e::HotPolicy::Adaptive, false);
        hp.begin_request();
        prompt(p, hp, false);
        std::vector<e::HotSwap> flight;
        for (int t = 0; t < 300; ++t) step(p, hp, flight, 1, &alone, "promoted batch, alone");
    }
    // promoted, then the batch
    BatchWork opening, batch, stat;   // opening: the steps before the measured ones
    {
        e::HotLayerPlan p;
        p.init(lay);
        e::HotPrompt hp;
        p.begin_request(e::HotPolicy::Adaptive, false);
        hp.begin_request();
        std::vector<e::HotSwap> flight;
        for (int b = 1; b <= 4; ++b) {
            if (b > 1) prompt(p, hp, true);
            const std::string tag = "promoted batch, " + std::to_string(b) + " rows";
            for (int t = 0; t < 20; ++t) step(p, hp, flight, b, &opening, tag);
        }
        const std::mt19937 at = rng;
        for (int t = 0; t < 300; ++t) step(p, hp, flight, 4, &batch, "promoted batch, 4 rows");
        // the same rows against the hot set's slab
        rng = at;
        e::HotLayerPlan s;
        s.init(lay);
        for (int t = 0; t < 300 * 4; ++t)
            for (int32_t x : row()) stat.misses += s.slot_of(x) < 0;
        stat.rows = 300 * 4;
    }
    auto per_row = [](const BatchWork& b) { return (double) b.misses / (double) b.rows; };
    char what[400];
    std::snprintf(what, sizeof what, "promoted batch: B = 4 after a promoted request %.3f misses a row (%lld swaps in "
                  "the %lld steps before, %lld in its %lld), B = 1 alone %.3f (%lld swaps), the static slab %.3f: the "
                  "batch swaps, its 4-row steps within 1.25 x the alone run's misses a row and at most half the static "
                  "slab's",
                  per_row(batch), (long long) opening.swaps, (long long) opening.steps, (long long) batch.swaps,
                  (long long) batch.steps, per_row(alone), (long long) alone.swaps, per_row(stat));
    expect(opening.swaps + batch.swaps > 0 && per_row(batch) <= 1.25 * per_row(alone) &&
               per_row(batch) <= 0.5 * per_row(stat),
           what);
}

// ---- (b) the trace replay ----

struct HotSetLayer {
    std::vector<int32_t> vram, warm;
};

std::map<int, HotSetLayer> read_hot_set(const std::string& path) {
    std::ifstream f(path);
    if (!f) die("cannot open " + path);
    std::map<int, HotSetLayer> out;
    try {
        const nlohmann::json j = nlohmann::json::parse(f);
        for (const auto& l : j.at("layers")) {
            HotSetLayer h;
            h.vram = l.at("experts").get<std::vector<int32_t>>();
            if (l.contains("warm")) h.warm = l.at("warm").get<std::vector<int32_t>>();
            out[l.at("layer").get<int>()] = h;
        }
    } catch (const std::exception& ex) {
        die(path + ": " + ex.what());
    }
    return out;
}

int replay(int argc, char** argv) {
    std::string trace, hot_path, policy_text;
    e::HotConfig c;
    bool mapping_pinned = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--hot-set") hot_path = val();
        else if (a == "--policy") policy_text = val();
        else if (a == "--swaps") c.max_swaps = std::atoi(val().c_str());
        else if (a == "--spare") c.spare = std::atoi(val().c_str());
        else if (a == "--half-life") c.half_life = std::atof(val().c_str());
        else if (a == "--margin") {
            const std::string m = val();
            const size_t k = m.find(':');
            if (k == std::string::npos) die("--margin MUL:ADD");
            c.margin_mul = std::atof(m.substr(0, k).c_str());
            c.margin_add = std::atof(m.substr(k + 1).c_str());
        } else if (a == "--mapping-pinned") mapping_pinned = true;
        else if (trace.empty() && a[0] != '-') trace = a;
        else die("unknown argument " + a);
    }
    if (trace.empty() || hot_path.empty()) die("--replay TRACE --hot-set FILE");
    e::HotPolicy forced{};
    if (!policy_text.empty() && !e::parse_hot_policy(policy_text, forced)) die("--policy " + policy_text);
    e::hot_config_check(c);
    const std::map<int, HotSetLayer> hot = read_hot_set(hot_path);
    std::ifstream f(trace);
    if (!f) die("cannot open " + trace);
    std::string line;
    int n_layer = 0, n_expert = 0, n_used = 0;
    if (!std::getline(f, line) ||
        std::sscanf(line.c_str(), "# dsv41-route-trace/1 n_layer %d n_expert %d n_used %d", &n_layer, &n_expert, &n_used) != 3)
        die(trace + ": not a dsv41-route-trace/1 file");
    std::vector<e::HotLayerPlan> plans((size_t) n_layer);
    for (int l = 0; l < n_layer; ++l) {
        e::HotLayerPlan::Layout lay;
        lay.n_expert = n_expert;
        auto it = hot.find(l);
        if (it != hot.end()) {
            lay.vram = it->second.vram;
            lay.warm = it->second.warm;
        }
        // the engine's layout: spare slots in a layer with experts in and out of the slab, spare homes when the
        // mapping is not pinned and the layer has a warm tier
        const bool mixed = !lay.vram.empty() && (int) lay.vram.size() < n_expert;
        lay.spare = mixed ? c.spare : 0;
        lay.spare_homes = mixed && !mapping_pinned && !lay.warm.empty() ? c.spare : 0;
        lay.mapping_pinned = mapping_pinned;
        plans[(size_t) l].init(lay);
    }
    const double decay = std::pow(2.0, -1.0 / c.half_life);
    // per request: its policy, then per layer the decode tokens (the D lines in order)
    struct Request {
        std::string name, policy;
        bool reset = false, resume = false;
        std::vector<std::vector<std::vector<int32_t>>> prompt, decode;   // [layer][call or token][ids]
    };
    std::vector<Request> reqs;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        std::string kind;
        in >> kind;
        if (kind == "R") {
            Request r;
            std::string reset, resume, extra;
            in >> r.name >> r.policy >> reset;
            r.reset = reset == "reset=1";
            // resume= came after the first traces (absent: 0)
            if (in >> resume) {
                if (resume != "resume=0" && resume != "resume=1") die(trace + ": an R line " + line);
                r.resume = resume == "resume=1";
            }
            if (in >> extra) die(trace + ": an R line " + line);
            r.prompt.assign((size_t) n_layer, {});
            r.decode.assign((size_t) n_layer, {});
            reqs.push_back(std::move(r));
            continue;
        }
        if (reqs.empty()) die(trace + ": a routing line before the first R line");
        int layer = -1;
        int64_t T = 1;
        in >> layer;
        if (kind == "P") in >> T;
        else if (kind != "D") die(trace + ": line kind " + kind);
        if (layer < 0 || layer >= n_layer || T < 1) die(trace + ": " + line.substr(0, 40));
        std::vector<int32_t> ids((size_t) (T * n_used));
        for (int32_t& x : ids)
            if (!(in >> x)) die(trace + ": short line " + line.substr(0, 40));
        (kind == "P" ? reqs.back().prompt : reqs.back().decode)[(size_t) layer].push_back(std::move(ids));
    }
    int64_t all_hits = 0, all_pairs = 0, all_swaps = 0;
    for (const Request& r : reqs) {
        e::HotPolicy p{};
        if (!policy_text.empty()) p = forced;
        else if (!e::parse_hot_policy(r.policy, p)) die(trace + ": request " + r.name + " policy " + r.policy);
        int64_t hits = 0, pairs = 0, swaps = 0;
        std::printf("request %s policy %s\n", r.name.c_str(), e::hot_policy_name(p));
        for (int l = 0; l < n_layer; ++l) {
            e::HotLayerPlan& h = plans[(size_t) l];
            std::vector<e::HotSwap> flight;
            if (r.reset)
                while (!h.is_static()) {
                    h.plan_static(c.spare, flight);
                    if (flight.empty()) die("layer " + std::to_string(l) + ": the static reset is stuck");
                    for (const e::HotSwap& s : flight) h.publish(s);
                    flight.clear();
                }
            h.begin_request(p, r.resume);
            for (const std::vector<int32_t>& ids : r.prompt[(size_t) l]) h.count_prompt(ids.data(), (int64_t) ids.size());
            h.prompt_done(c);
            h.plan(c, flight);
            int64_t lh = 0, lp = 0, ls = (int64_t) flight.size();
            for (const std::vector<int32_t>& ids : r.decode[(size_t) l]) {
                for (int32_t x : ids) lh += h.slot_of(x) >= 0;
                lp += (int64_t) ids.size();
                h.count_token(ids.data(), (int) ids.size(), decay);
                for (const e::HotSwap& s : flight) h.publish(s);
                flight.clear();
                h.plan(c, flight);
                ls += (int64_t) flight.size();
            }
            // the request's last swaps are published before the next one starts (the engine drains at a prefill)
            for (const e::HotSwap& s : flight) h.publish(s);
            const std::string why = h.check();
            if (!why.empty()) die("layer " + std::to_string(l) + ": " + why);
            std::printf("layer %d hits %lld of %lld swaps %lld\n", l, (long long) lh, (long long) lp, (long long) ls);
            hits += lh;
            pairs += lp;
            swaps += ls;
        }
        std::printf("request %s total hits %lld of %lld swaps %lld\n", r.name.c_str(), (long long) hits, (long long) pairs,
                    (long long) swaps);
        all_hits += hits;
        all_pairs += pairs;
        all_swaps += swaps;
    }
    std::printf("total hits %lld of %lld swaps %lld\n", (long long) all_hits, (long long) all_pairs, (long long) all_swaps);
    return 0;
}

}  // namespace

// the deterministic mode's rules (hot_request_start, hot_boundary_waits, hot_request_end_resets): it resets
// and never resumes, whatever the request's keys, and resets again at the request's end; off, the keys as before (a
// reset drops the counts) and no reset at the end; a boundary waits in the mode or with --hot-check only
void test_deterministic_rules() {
    bool ok = true;
    for (int reset = 0; reset < 2; ++reset)
        for (int resume = 0; resume < 2; ++resume) {
            const e::HotRequestStart on = e::hot_request_start(true, reset, resume);
            const e::HotRequestStart off = e::hot_request_start(false, reset, resume);
            ok = ok && on.reset && !on.resume && off.reset == (reset != 0) && off.resume == (resume && !reset);
        }
    ok = ok && e::hot_boundary_waits(true, false) && e::hot_boundary_waits(false, true) && !e::hot_boundary_waits(false, false);
    // the mode's reset also runs at a request's end (the idle time); off, never
    ok = ok && e::hot_request_end_resets(true) && !e::hot_request_end_resets(false);
    expect(ok, "deterministic mode: every request reset and fresh, every boundary waits, every end resets; off: the "
               "request's keys, no wait, no reset at the end");
}

// host homes against the spare homes on the same counts, token by token
void test_host_homes() {
    e::HotLayerPlan two, host;
    two.init(small_layout(false));
    e::HotLayerPlan::Layout l = small_layout(false);
    l.spare_homes = 0;
    l.homed = l.vram;
    host.init(l);
    // the slab's experts 0..3 keep homes 8..11 after the warm tier's 8
    bool homes_ok = host.n_homes() == 12;
    for (int32_t x = 0; x < 4; ++x) homes_ok = homes_ok && host.home_of(x) == 8 + x && host.slot_of(x) == x;
    e::HotConfig c;
    c.margin_mul = 0;
    c.margin_add = -1;
    for (e::HotLayerPlan* p : {&two, &host}) p->begin_request(e::HotPolicy::Adaptive, false);
    std::mt19937 rng(7);
    for (int i = 0; i < 40; ++i) {
        const int32_t id = (int32_t) (rng() % 16);
        two.count_prompt(&id, 1);
        host.count_prompt(&id, 1);
    }
    two.prompt_done(c);
    host.prompt_done(c);
    std::vector<e::HotSwap> f2, fh;
    two.plan(c, f2);
    host.plan(c, fh);
    int64_t same = 0, tokens = 0, down = 0, swaps = 0;
    std::vector<int32_t> home0(16);
    for (int32_t x = 0; x < 16; ++x) home0[(size_t) x] = host.home_of(x);
    const double decay = std::pow(2.0, -1.0 / c.half_life);
    for (int t = 0; t < 200; ++t) {
        const int32_t ids[3] = {(int32_t) (rng() % 16), (int32_t) (rng() % 16), (int32_t) (rng() % 16)};
        two.count_token(ids, 3, decay);
        host.count_token(ids, 3, decay);
        bool eq = f2.size() == fh.size();
        for (size_t i = 0; eq && i < f2.size(); ++i) eq = f2[i].in == fh[i].in && f2[i].out == fh[i].out;
        for (const e::HotSwap& x : fh) {
            down += x.to_home >= 0;
            // the promoted expert copied up from its own home (the warm tier's or its host home), a cold one never
            homes_ok = homes_ok && x.from_home == home0[(size_t) x.in] && x.from_home >= 0;
        }
        same += eq;
        swaps += (int64_t) fh.size();
        ++tokens;
        boundary(two, c, f2, "host homes (spare homes side)");
        boundary(host, c, fh, "host homes");
        for (int32_t x = 0; x < 16; ++x) homes_ok = homes_ok && host.home_of(x) == home0[(size_t) x];
    }
    expect(same == tokens && swaps > 0, "host homes: " + std::to_string(same) + " of " + std::to_string(tokens) +
                                            " tokens swap the same experts in and out as with spare homes (" +
                                            std::to_string(swaps) + " swaps)");
    expect(down == 0 && homes_ok, "host homes: no victim copied down, every expert's home the one init gave it, each "
                                  "promoted expert copied up from it");
    for (const e::HotSwap& x : f2) two.publish(x);
    for (const e::HotSwap& x : fh) host.publish(x);
    int moved = 0;
    for (int32_t x = 0; x < 4; ++x) moved += host.slot_of(x) < 0;
    std::vector<e::HotSwap> sw;
    const int n = host.restore_in_place(sw);
    bool up_only = (int) sw.size() == n;
    for (const e::HotSwap& x : sw) up_only = up_only && x.to_home < 0 && x.from_home == home0[(size_t) x.in] && x.in < 4;
    expect(moved > 0 && n == moved && host.is_static() && host.check().empty() && up_only,
           "host homes: restore_in_place puts the " + std::to_string(moved) + " displaced experts of the hot set back at "
           "once, each into a non-member's slot from its own home, the slab the hot set's");
}

// partial host homes (the slab's experts 2 and 3 homed, as the engine's home order takes a boundary
// layer's from the end of its VRAM order) against the spare homes on the same counts, token by token
void test_partial_homes() {
    e::HotLayerPlan two, part;
    two.init(small_layout(false));
    e::HotLayerPlan::Layout l = small_layout(false);
    l.homed = {2, 3};
    part.init(l);
    // homes: the warm tier's 0..7, the homed experts' 8 and 9, the spare 10 and 11
    bool homes_ok = part.n_homes() == 12 && part.homed() == 2 && !part.host_homes() && !part.restores_in_place() &&
                    part.home_of(2) == 8 && part.home_of(3) == 9 && part.home_of(0) < 0 && part.check().empty();
    e::HotConfig c;
    c.margin_mul = 0;
    c.margin_add = -1;
    for (e::HotLayerPlan* p : {&two, &part}) p->begin_request(e::HotPolicy::Adaptive, false);
    std::mt19937 rng(11);
    for (int i = 0; i < 40; ++i) {
        const int32_t id = (int32_t) (rng() % 16);
        two.count_prompt(&id, 1);
        part.count_prompt(&id, 1);
    }
    two.prompt_done(c);
    part.prompt_done(c);
    std::vector<e::HotSwap> f2, fp;
    two.plan(c, f2);
    part.plan(c, fp);
    int64_t same = 0, tokens = 0, dropped = 0, down = 0, wrong = 0, swaps = 0;
    const double decay = std::pow(2.0, -1.0 / c.half_life);
    for (int t = 0; t < 200; ++t) {
        const int32_t ids[3] = {(int32_t) (rng() % 16), (int32_t) (rng() % 16), (int32_t) (rng() % 16)};
        two.count_token(ids, 3, decay);
        part.count_token(ids, 3, decay);
        bool eq = f2.size() == fp.size();
        for (size_t i = 0; eq && i < f2.size(); ++i) eq = f2[i].in == fp[i].in && f2[i].out == fp[i].out;
        for (const e::HotSwap& x : fp) {
            // a homed victim dropped, any other copied down into a free home; a homed expert promoted from its own
            const bool homed_out = x.out == 2 || x.out == 3;
            dropped += homed_out && x.to_home < 0;
            down += !homed_out && x.to_home >= 0;
            wrong += homed_out != (x.to_home < 0) || x.to_home == 8 || x.to_home == 9 ||
                     ((x.in == 2 || x.in == 3) && x.from_home != 6 + x.in);
        }
        same += eq;
        swaps += (int64_t) fp.size();
        ++tokens;
        boundary(two, c, f2, "partial homes (spare homes side)");
        boundary(part, c, fp, "partial homes");
        homes_ok = homes_ok && part.home_of(2) == 8 && part.home_of(3) == 9;
    }
    expect(same == tokens && swaps > 0, "partial homes: " + std::to_string(same) + " of " + std::to_string(tokens) +
                                            " tokens swap the same experts in and out as with spare homes (" +
                                            std::to_string(swaps) + " swaps)");
    expect(dropped > 0 && wrong == 0 && homes_ok, "partial homes: " + std::to_string(dropped) + " homed victims dropped, " +
                                                      std::to_string(down) + " others copied down into a free home, the "
                                                      "homed experts' homes their own throughout");
    // the reset in rounds of the spare slots, as the two-way path's, back to the hot set's slab
    for (const e::HotSwap& x : f2) two.publish(x);
    for (const e::HotSwap& x : fp) part.publish(x);
    int rounds = 0;
    while (!part.is_static() && rounds < 100) {
        std::vector<e::HotSwap> sw;
        part.plan_static(c.spare, sw);
        for (const e::HotSwap& x : sw) part.publish(x);
        ++rounds;
    }
    expect(part.is_static() && part.check().empty() && part.home_of(2) == 8 && part.home_of(3) == 9,
           "partial homes: the reset in " + std::to_string(rounds) + " rounds back at the hot set's slab, every home "
           "consistent");
}

// a child run: restore_in_place on a layer with partial host homes (some victim would need a copy down) must exit 1
int partial_restore_child() {
    e::HotLayerPlan p;
    e::HotLayerPlan::Layout l = small_layout(false);
    l.homed = {3};
    p.init(l);
    std::vector<e::HotSwap> sw;
    p.restore_in_place(sw);
    return 0;
}

// HotRestore's order over 4 layers
void test_restore_order(const char* self) {
    e::HotRestore r;
    r.begin(4);
    bool ok = r.open() && r.recorded() == 0;
    for (int l = 0; l < 4; ++l) r.record(l);
    // every layer waited for once, in any order of the calls (a prefill chunk, then decode steps)
    ok = ok && r.wait_before(2) && !r.wait_before(2) && r.wait_before(0) && r.wait_before(1) && r.wait_before(3) &&
         !r.wait_before(0) && r.waited(3);
    r.done();
    ok = ok && !r.open() && !r.wait_before(1);
    expect(ok, "restore order: each layer's computation waits for its own restore once; none once it is done");
    const std::string cmd = std::string(self) + " --restore-before-record > /dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    expect(rc != -1 && WIFEXITED(rc) && WEXITSTATUS(rc) == 1,
           "restore order: a layer computing before its restore is queued exits 1 (a child run)");
    const std::string cmd2 = std::string(self) + " --partial-restore > /dev/null 2>&1";
    const int rc2 = std::system(cmd2.c_str());
    expect(rc2 != -1 && WIFEXITED(rc2) && WEXITSTATUS(rc2) == 1,
           "restore order: an in-place restore of a layer with partial host homes exits 1 (a child run)");
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--replay") return replay(argc, argv);
    if (argc == 2 && std::string(argv[1]) == "--restore-before-record") {
        // test_restore_order's child: layer 1 computes while only layer 0's restore is recorded
        e::HotRestore r;
        r.begin(4);
        r.record(0);
        r.wait_before(1);
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--partial-restore") return partial_restore_child();
    if (argc != 1) die("usage: dsv41_engine_hot_policy_test [--replay TRACE --hot-set FILE ...]");
    test_churn(false);
    test_churn(true);
    test_prompt();
    test_hysteresis();
    test_merge_prompt();
    test_static_reset();
    test_copy_only(false);
    test_copy_only(true);
    test_two_turn();
    test_promoted_batch();
    test_deterministic_rules();
    test_host_homes();
    test_partial_homes();
    test_restore_order(argv[0]);
    std::printf("dsv41_engine_hot_policy_test: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
