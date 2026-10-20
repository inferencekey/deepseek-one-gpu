// include/strata/dsv41/engine/hot_policy.hpp - the per-request VRAM hot set of the decode:
// which routed experts each layer's slab holds while a request decodes, and the swaps
// that change it. Host only, no CUDA: experts.cpp runs the swaps' copies and publishes them, the GPU-free test
// (tests/dsv41/engine/hot_policy_test.cpp) and tools/dsv41/engine/hot_policy_sim.py replay the same rules on a
// routing trace.
//
// Why (a measured session, A100-SXM4, the real Q2_K): the static hot set, packed from the hybrid's counts on one prompt,
// held 34-44% of a request's expert uses, so the CPU computed ~60% of them (24-28 ms of a ~50 ms token). Strata's
// method (memory: strata-moe-offload-metodo) keeps a hot cache in VRAM that follows the routing: the experts a
// request's own prompt used, then a decayed LFU over its decode.
//
// Policies (HotPolicy, per request: dsv41-engine --hot-policy, the serve protocol's hot= key):
//   static    the hot-set file's experts, as before (a request whose slab differs moves back to them)
//   prompt    once the request's prompt is read, each layer's target is its N most used experts in that prompt
//             (N the hot set's count), ties and the experts the prompt did not use by the static order (the hot set's
//             VRAM list, then its warm list, then the others by id)
//   adaptive  prompt's target first; once every member of it is resident or on its way, a decayed LFU over the
//             decode's own routing: per layer each expert's count, the prompt's counts at the prompt's end, times
//             2^(-1 / half_life) per decode token plus one per use; the most counted expert outside the slab replaces
//             the least counted one inside when its count exceeds margin_mul x the victim's + margin_add (hysteresis,
//             default 1.5 x + 1). The default (the Strata method).
// A target swap (static, prompt and adaptive's first phase) pairs the target's best member outside the slab with
// the slab's worst non-member (by the scores, ties by the static order); no hysteresis there.
//
// A prompt read beside decoding batch slots (batch.hpp): no request starts, the
// slots' hot set goes on, and the read's routing joins the batch's decayed counts at the read's end (merge_prompt),
// weighed as one more row's decode at equilibrium. Before, the read's prompt_done replaced the batch's counts with the
// prompt's and took its target: the RTX PRO 6000 Server session of 2026-10-15 (4 steady users) decoded its
// first 15 s after the last read at 18.3 tok/s a user against about 26.6 later in the same replies, the slab held to
// the prompts' 16K-token counts for hundreds of steps. hot_policy_sim.py batch on the A100 trace of the real model
// (2026-10-09; the RTX gate's hot set, 147 slab experts a layer on average): a row joining after its own 16K prompt
// read beside another, the batch's CPU jobs a step 49.5 / 26.3 / 19.1 / 16.3 in steps 0-25 / 25-50 / 50-100 /
// 100-200 after the read with the merge, against 52.0 / 48.6 / 50.8 / 52.7 with the old target (and 68.8 / 45.4 /
// 24.7 / 13.9 with the read ignored); four rows with an unrelated 16K read, 92.1 / 64.2 / 57.1 / 58.3 against 99.4 /
// 91.6 / 82.4 / 89.8.
//
// A resumed conversation (the request reuses the engine's prefix: a Live resume or the prompt-end snapshot, and no
// hot_reset; a conversation's next turn reuses its
// experts): prompt and adaptive keep the counts the last request left (adaptive's decayed LFU, prompt's prompt counts)
// and add the suffix's to them, and the target's ties go to the experts the slab holds before the static order. For
// adaptive a tie is anything within its hysteresis margin: the target is the slab plus the swaps plan_lfu would make
// at once (with exact ties only, the members the margin kept during the last decode would leave and come back). A
// suffix of ~50 tokens alone is a thin sample: as a fresh request's target it pulls the slab toward the suffix's
// noise and the static order. hot_policy_sim.py two-turn (4 layers of 384 experts, 119 in the slab, turn 1 a
// 1024-token prompt and 256 decode tokens, turn 2 a 50-token suffix and 256): turn 2 fresh 0.797 of its pairs hit
// with 399 swaps, resumed 0.830 with 65, turn 1's decode run on uninterrupted over the suffix 0.829 with 83. A fresh
// request (nothing reused) and a static one start from zero counts as before.
//
// A swap and where every expert lives (HotSource): an expert inside the slab (Slot) is computed by the GPU; outside,
// by the CPU from its Home (a pinned host slot: the warm tier's, or a spare home allocated for the swaps) or from the
// model's mapping. Only an expert in pinned host memory is ever promoted (a Home, or a Mapping the caller pinned, as
// the smoke's synthetic model; cold experts never are, Fixed). A swap (in, out) copies in's bytes from
// where it is into a free slot (the slab has `spare` slots per layer beyond the hot set's) and, when out's mapping is
// not pinned, out's bytes from its slot into a free home, both on the engine's copy stream; once the copies' event
// is done the swap is published between two tokens: in takes the slot, out takes the home (or goes back to its
// pinned mapping), in's old home and out's old slot become free. Until then both stay where they were, so no token
// reads a half-copied expert and every expert always has a complete copy where the token reads it. At most
// max_swaps swaps start per layer per token, and at most `spare` are in flight per layer (one free slot each).
//
// Host homes (HotConfig::host_homes, dsv41-engine
// --hot-homes all): the swaps and the deterministic reset were PCIe-copy bound, 24.4 MiB a swap at ~20.4 GB/s each way
// on the RTX PRO 6000 (12.2 MiB up for the promoted expert, 12.2 down for the victim; 0.6 ms a swap, one
// session's 2885-3311 reset swaps in 1750-2030 ms). With host homes every expert of the hot set's slab also has a
// permanent pinned home of its own (filled at init, after the warm tier's homes), so every expert that can be in the
// slab always has a valid host copy: a swap copies its promoted expert up only, its victim is dropped (it keeps its
// home, no copy down, no spare homes), and a promoted expert's home stays its own. The planner's choices are the
// two-way path's (the spare homes never ran out there: one per swap in flight), so the slab, the hits and the
// tokens are the same; only the copies go.
//
// Partial host homes (HotConfig::host_homes_max, dsv41-engine --hot-homes N): when the host's RAM
// holds homes for only some of the slab's experts (a pod's 262.63 GiB cgroup: 3.97 GiB short of all 5951 with
// the gate's 8 GiB margin), a layer whose slab experts all have homes works as above, and a layer where only some do
// keeps the two-way path for the rest: Layout::homed names the experts with a permanent home, a swap whose victim is
// one of them drops it (no copy down), any other victim goes down into a free home as before, a promoted homed expert
// keeps its home and any other promoted expert frees its own. The spare homes stay in such a layer, and they cannot
// run out where the two-way path's did not: its homes hold fewer experts (the homed ones outside the slab are in
// their own), so the planner's choices are the two-way path's there too. Its reset stays the reset in rounds (a victim
// promoted from the warm tier has no host copy left), restore_in_place refuses it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

enum class HotPolicy : uint8_t { Static, Prompt, Adaptive };
const char* hot_policy_name(HotPolicy p);
/// false when s is none of static, prompt, adaptive
bool parse_hot_policy(const std::string& s, HotPolicy& p);

/// The hot set's knobs (dsv41-engine --hot-*): constants are flags, so a session can try a second setting
struct HotConfig {
    HotPolicy policy = HotPolicy::Adaptive;   ///< a request's policy unless it names one
    bool deterministic = true;  ///< a request's mode unless it names one (hot_request_start, hot_boundary_waits)
    int max_swaps = 2;        ///< swaps started per layer per token
    int spare = 2;            ///< free slab slots per layer beyond the hot set's (0: no swaps, every request static)
    double half_life = 128;   ///< adaptive: decode tokens over which a count halves
    double margin_mul = 1.5;  ///< adaptive: a swap needs count(in) > margin_mul count(out) + margin_add; a negative
    double margin_add = 1.0;  ///< margin_add swaps even at equal counts (the tests' swaps at every token)
    /// after every decode token: wait for the swaps' copies (each published at the next token), then check the
    /// device's residency rows against the host's slot map and every newly filled slot and home byte for byte
    bool check = false;
    /// the smoke's copy-cost run: every batch's copies run as
    /// planned, the promoted experts up and the victims down (a victim whose mapping is pinned into a scratch home of
    /// the engine's, as the real model's victims go into their homes), but are never published: the boundary that
    /// finds them done gives their slots and homes back (HotLayerPlan::abandon), so the residency rows and the slab do
    /// not move and the hit rate is the static slab's; the copies' cost alone
    bool copy_only = false;
    /// every expert of the hot set's slab gets a permanent pinned home (the top of this file), a swap
    /// copies up only, and the deterministic reset restores layer by layer in place (HotRestore); e07_gate.py turns it
    /// on only when the host's RAM holds the homes with its margin. false: the two-way path (spare homes, the victims
    /// copied down) and the reset in rounds, as before
    bool host_homes = false;
    /// partial host homes: with host_homes, at most this many of the slab's experts get a home, the
    /// first ones of the engine's home order (experts.hpp); -1: every one (--hot-homes all), N: --hot-homes N
    int64_t host_homes_max = -1;
};

/// Exits naming the field unless the knobs are usable (max_swaps >= 1 when spare > 0, half_life > 0, margin_mul >= 0).
void hot_config_check(const HotConfig& c);

// the deterministic mode (on by default for every request, dsv41-engine
// --deterministic and the serve protocol's det= key). Before the CPU experts took the GPU's numerics, an expert computed in the slab (a hit: q8_1
// activations, MMVQ) and on the CPU (a miss: Q8_K, the fork's vec_dot) differed in its last bits, so the tokens
// followed the placement, and the placement followed the copy stream's speed: a batch still copying at a boundary was published at a later one
// (measured: 59 of 196 batches deferred, two identical greedy requests apart at token 14). In the
// mode, a request's placement is a function of its own prompt and tokens: it starts from the hot set's slab with zero
// counts (the reset of hot_reset=1, nothing carried from the last request, whatever it was), and every boundary waits
// for the open batch, so the swaps planned at a token are published right after it. The adaptive swaps stay
// (1.5x the static slab's speed on short prompts). The CPU experts now compute a
// hit's arithmetic (experts.hpp CpuNumerics::Gpu), so a token's bits no longer follow the placement; the mode's reset
// and waits stay as they are, and they are what keeps a request's placement, and so its
// speed, a function of its own prompt and tokens.

/// How a request starts its hot set: the slab back to the hot set's first (reset), the planners' counts kept (resume).
struct HotRequestStart {
    bool reset = false, resume = false;
};
/// The deterministic mode resets and never resumes; otherwise the request's own keys, a reset dropping the counts.
HotRequestStart hot_request_start(bool deterministic, bool reset, bool resume);
/// Whether a decode boundary waits for the open batch's copies (and publishes it) instead of leaving it open while it
/// still copies: in the deterministic mode, and with HotConfig::check.
bool hot_boundary_waits(bool deterministic, bool check);
// the mode's reset to the hot set's slab moves to the end
// of a request, the engine's idle time before the next one (0.6-2.0 s before a first token at a request's start:
// A100, RTX PRO 6000 with hot_reset=1). A request in the mode then starts from the slab the last one's end
// left, the hot set's, and its own start's reset finds nothing to move; that reset stays, for a request after one in
// the mode off, which does not reset at its end. The next request waits only for what of the end's reset is still
// running when it arrives. The determinism rule is the same: every request in the mode starts from the hot set's
// slab with zero counts.
/// Whether a request's end resets the slab to the hot set's: in the deterministic mode.
bool hot_request_end_resets(bool deterministic);

// with host homes the reset needs no copy down and no make-before-break (no token runs while it
// is queued), so it restores every layer in place (HotLayerPlan::restore_in_place: each hot-set expert outside the
// slab copied from its home into the slot of a non-member, which is dropped) in layer order, on a stream of its own
// with an event after each layer's copies. The planners are the hot set's slab at once, so the placement every layer
// computes with is exactly the slab's, but a layer's slots are complete only at its event: the next
// request's work that reads layer l's slab (a prefill call's passes, an eager decode step's hits) waits for layer l's
// event alone, and the token graph (which holds no event wait) for the last one before its launch. Expected on the RTX
// PRO 6000 (up only, ~26 GB/s): 0.49 ms a swap, ~1.41 s for 2900 swaps, a resumed turn's short prefill trailing the
// restore by about one layer. HotRestore is that order's host side: ExpertEngine keeps one beside its events, the
// tests replay it.
class HotRestore {
public:
    /// A restore of n_layer layers starts (none recorded); exits while another is open.
    void begin(int n_layer);
    /// Layer l's copies are queued and its event recorded; exits unless l is the next layer in order.
    void record(int l);
    /// Before a computation of layer l reads the slab: true when its stream must wait for layer l's event first (a
    /// restore is open and no earlier work on that stream has waited for l); exits when l's event is not recorded
    /// yet, so no layer ever computes before its restore is queued.
    bool wait_before(int l);
    /// Every layer's event seen complete: nothing is waited for any more.
    void done();
    bool open() const { return open_; }
    int recorded() const { return recorded_; }
    int n_layer() const { return n_layer_; }
    bool waited(int l) const { return open_ && waited_[(size_t) l] != 0; }

private:
    bool open_ = false;
    int n_layer_ = 0, recorded_ = 0;
    std::vector<uint8_t> waited_;
};

// When a request's prompt ends for the planners. A prompt or
// adaptive HotLayerPlan plans and counts nothing from its begin_request until its prompt_done, which the engine calls
// at the first decode call after the prompt. Before 6033a20 only a prefill call opened the prompt, so a request
// that read no token by prefill never ended it: once a promoted request's one
// token left is fed by decode, its begin_request cleared the planners and nothing called prompt_done, so the batch
// slot it went into, and every slot that joined it (merge_prompt waits for prompt_done too), decoded on the idle
// reset's slab with no swap until the slots were all idle. The 2026-10-19 session (RTX PRO 6000, AMD EPYC
// 9575F, before the fix): its 4-row steps 0.00 swaps and a hit rate of
// 0.4031, 70.5 ms a step, 45.33 tok/s for 4 steady users; an earlier session on the same GPU: 4.80
// swaps, 0.9864, 32.0 ms, 123.82 tok/s. A request's start now opens its prompt whatever reads it: the first decode
// call ends it with prompt_done, an empty prompt when no token was read by prefill (the counts then start from the
// decode tokens, as hot_policy_sim.py and the test's replay end every request's prompt, P lines or none). Forward
// keeps one, the tests replay it.
enum class HotPromptEnd : uint8_t { None, Done, Merge };
class HotPrompt {
public:
    /// A request starts (Forward::begin_request, after the planners' begin_request): its prompt is open, read by
    /// prefill or not.
    void begin_request();
    /// A prefill call: the prompt is open; beside: read beside decoding batch slots, so merged into their counts.
    void prefill(bool beside);
    /// The first decode call after it: Done (HotLayerPlan::prompt_done), Merge (merge_prompt), or None when no prompt
    /// is open; the prompt is closed.
    HotPromptEnd end();

private:
    bool open_ = false, beside_ = false;
};

enum class HotSource : uint8_t { Slot, Home, Mapping, Fixed };

/// One swap of a layer: in is copied into the free slot `slot` from its home from_home (-1: its pinned mapping); out
/// leaves its slot for the free home to_home (-1: back to its pinned mapping, no copy).
struct HotSwap {
    int32_t in = -1, out = -1, slot = -1, from_home = -1, to_home = -1;
};

/// One layer's slab, homes and counts, and the swaps that move them.
class HotLayerPlan {
public:
    struct Layout {
        int32_t n_expert = 0;
        std::vector<int32_t> vram;   ///< the hot set's experts: slots 0 .. vram.size() - 1, in order
        int spare = 0;               ///< free slots after them
        std::vector<int32_t> warm;   ///< the warm tier's experts in the hot set's order, warm[k] in home k
        int spare_homes = 0;         ///< free homes after the warm ones
        bool mapping_pinned = false; ///< the other experts' mapping is pinned: promotable, and a victim goes back
        /// the slab experts with a permanent host home, a subset of vram in its order, homed[k]
        /// in home warm.size() + k (the spare homes after them). Every one of vram: host homes in full (no spare homes:
        /// spare_homes 0); some: the two-way path for the others (spare homes needed); none: the two-way path
        std::vector<int32_t> homed;
    };
    /// Exits naming the problem when an expert is in two places or out of range.
    void init(const Layout& l);

    /// A request starts: its policy, its counts cleared unless `resume` (the request continues the engine's sequence,
    /// no reset) and p is not static; static targets the hot set from now on, prompt and adaptive plan nothing until
    /// prompt_done.
    void begin_request(HotPolicy p, bool resume);
    /// The prompt's routed pairs (a prefill call's ids, n of them).
    void count_prompt(const int32_t* ids, int64_t n);
    /// The prompt is read: prompt and adaptive take its target (adaptive: its counts start from the prompt's). A
    /// resumed request counts from the carried counts plus the prompt's; its target: adaptive, the slab with the
    /// swaps c's margin allows; prompt, the N most counted, the slab's experts first among equal counts.
    void prompt_done(const HotConfig& c);
    /// a prompt read beside decoding batch slots ends. Its counts (count_prompt
    /// since the last call) join adaptive's decayed counts scaled to `mass` in all (the counts one row's decode
    /// holds at equilibrium: n_used / (1 - 2^(-1 / half_life)), 1110.5 for 6 used and 128), then are cleared; the
    /// target, the phase and the policy stay as they are, so the swaps follow the union of the rows' routes at once.
    /// Nothing for a static or prompt policy, or before the batch's own prompt_done.
    void merge_prompt(double mass);
    /// One decode token's n ids: adaptive's counts decay by `decay` (2^(-1 / half_life)), then one per use.
    void count_token(const int32_t* ids, int n, double decay);
    /// The swaps to start now (appended to out, at most c.max_swaps, each with a free slot and, when needed, a free
    /// home), marked in flight.
    void plan(const HotConfig& c, std::vector<HotSwap>& out);
    /// At most `max` swaps toward the hot set's experts, whatever the policy (a reset to the static slab).
    void plan_static(int max, std::vector<HotSwap>& out);
    /// the slab back to the hot set's at once, in place: each hot-set expert outside it (by the
    /// static order) takes the slot of a non-member (by slot), copied from its home (from_home; -1: its pinned
    /// mapping), the non-member dropped (to_home -1: it keeps its home or goes back to its pinned mapping). Published
    /// here (appended to out), so the caller must complete the copies before anything reads those slots (HotRestore).
    /// Exits unless every victim keeps a host copy (restores_in_place) and no swap is in flight. Returns the swaps.
    int restore_in_place(std::vector<HotSwap>& out);
    /// Whether a victim needs no copy down: host homes in full, or the mapping pinned (restore_in_place's condition).
    bool restores_in_place() const { return !needs_home(); }
    /// Host homes in full: every slab expert has one (Layout::homed is vram).
    bool host_homes() const { return host_homes_; }
    /// the slab experts with a permanent host home (all of them with host_homes(); some: partial).
    int homed() const { return n_homed_; }
    bool has_fixed_home(int32_t e) const { return fixed_[(size_t) e] != 0; }
    /// A planned swap's copies are done: in takes its slot, out its home or its mapping.
    void publish(const HotSwap& s);
    /// A planned swap's copies are done and dropped (HotConfig::copy_only): its slot and home free again, both experts
    /// where they were, so the next plan may start the same swap.
    void abandon(const HotSwap& s);

    bool is_static() const;               ///< the slab holds the hot set's experts, nothing in flight
    int in_flight() const { return n_flight_; }
    HotPolicy policy() const { return policy_; }
    int32_t n_expert() const { return n_expert_; }
    int n_slots() const { return (int) slot_expert_.size(); }
    int n_homes() const { return (int) home_expert_.size(); }
    int resident() const { return n_resident_; }
    int32_t slot_of(int32_t e) const { return slot_of_[(size_t) e]; }
    int32_t slot_expert(int32_t s) const { return slot_expert_[(size_t) s]; }
    int32_t home_of(int32_t e) const { return home_of_[(size_t) e]; }
    HotSource source(int32_t e) const;
    double score(int32_t e) const { return score_[(size_t) e]; }
    /// "" when every invariant holds, else the first broken one: each expert in exactly one place (host homes: a
    /// slab expert in its home too, and every home still holding the expert init gave it; partial: a homed expert in
    /// its home too, its home still its own), slots and homes owned by whom they say, the free lists and the in-flight
    /// marks consistent, the resident count the hot set's.
    std::string check() const;

private:
    bool promotable(int32_t e) const;
    bool needs_home() const { return !mapping_pinned_ && !host_homes_; }
    void start(int32_t in, int32_t out, std::vector<HotSwap>& sw);
    /// target swaps toward target_ (no hysteresis) up to `max`; returns how many were started; *ins_left: a target
    /// member is still outside the slab and not on its way after them
    int plan_target(int max, std::vector<HotSwap>& out, bool* ins_left);
    int plan_lfu(const HotConfig& c, int max, std::vector<HotSwap>& out);

    int32_t n_expert_ = 0;
    int n_static_ = 0;
    bool mapping_pinned_ = false, host_homes_ = false;
    // host homes: every home's expert as init placed it (check() holds them; partial: the homed experts' homes, -2 for
    // the others, which change); per expert whether it has a permanent home
    std::vector<int32_t> fixed_homes_;
    std::vector<uint8_t> fixed_;
    int n_homed_ = 0, n_reserved_ = 0;    // n_reserved_: the in-flight swaps holding a home for their victim
    HotPolicy policy_ = HotPolicy::Static;
    bool prompt_seen_ = false, lfu_ = false, resume_ = false;
    std::vector<int32_t> rank_;           // the static order: VRAM list, warm list, the others by id
    std::vector<uint8_t> static_, target_, flight_;
    std::vector<int32_t> slot_of_, slot_expert_, home_of_, home_expert_;
    std::vector<int32_t> free_slots_, free_homes_;   // popped from the back
    std::vector<double> prompt_, score_;
    int n_flight_ = 0, n_resident_ = 0;
    std::vector<int32_t> ins_, outs_;     // plan's scratch
};

}  // namespace strata::dsv41::engine
