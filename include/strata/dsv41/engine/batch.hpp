// include/strata/dsv41/engine/batch.hpp - several conversations decoding at
// once over the engine's batch slots, one token per active slot per step, for `dsv41-engine --serve --batch N`
// (Strata's batch protocol: `BGEN`, `BADM`, `BT`, `BDONE`, `BSTOP`, src/program/generate.cpp's --serve block, which
// serve/server.py's StrataEngine.generate_batched already drives). Host only: this file holds the slots' policy and
// makes no CUDA call; the device side is a BatchBackend (Forward, forward.hpp; the dry run's stand-in; the CPU
// reference of tests/dsv41/engine/batch_test.cpp).
//
// A request is admitted as Strata's: its prompt is read and its first token picked through the Session (session.hpp)
// on the live sequence, as a request of one token (`BGEN` reads it as a `GEN 1`), then the live sequence's state is
// copied into its batch slot (BatchBackend::batch_admit) and the slot goes on with the rest of its tokens in the
// steps. A slot holds a sequence of at most batch_ctx() positions (the gate sizes it): a request whose
// prompt and tokens do not fit runs alone with the live sequence's full context (BatchSlots::fits says it, the serve
// loop then reads it as a GEN), so the batch waits for it.
//
// A step (BatchSlots::step) takes every active slot in slot order, one row each: the token each one picked last, fed
// at its own next position, through BatchBackend::batch_decode; each row's logits give its next token (greedy, ties to
// the lower id, or the request's own Sampler, which goes on drawing from the generator its first token was drawn
// with), which is printed, and the slot ends (BDONE) at an end-of-sequence id, at its max_new, or once a BSTOP asked:
// a stopped slot ends before the step, its next token not computed. A new request's prompt is read between two steps
// the steps waiting for it; a long prompt is read in chunks between steps (forward.hpp).
//
// Reproducibility: before the CPU experts took the GPU's numerics, the deterministic mode held a
// request's placement fixed by its own prompt and tokens only while it decoded alone, and a request that decoded
// beside another slot, or that a prompt was read beside, was marked not reproducible (the batch's hot set moved with
// the union of its rows' routes, and a GPU hit and a CPU miss of the same expert differed in their last bits). Now
// an expert's output has the same bits on either tier (the CPU experts in the GPU's numerics, experts.hpp
// CpuNumerics::Gpu) and a prompt read in the mode picks its prefill split from the hot set's slab
// (ExpertEngine::prompt_deterministic), so a token's bits do not depend on the placement: a request in the
// deterministic mode is reproducible (BatchDone::reproducible, the serve loop's rep=1) whatever its neighbours, and
// one in the mode off is not (rep=0). A session on 2026-10-17 (RTX PRO 6000, the real Q2_K;
// docs/RESULTS.md) measured it: the 9 batched requests that went into a slot at once matched their alone
// runs bit for bit (about 1,950 tokens, 4-row steps, prompts read beside); the 3 that left them (at tokens 67 and 91)
// were each a request stopped on the solo path and promoted into a slot, whose one token left the Session read with
// prefill(T=1) where alone it is decoded. Session now feeds it by decode (session.hpp).
#pragma once

#include "strata/dsv41/engine/sampler.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

/// The device side of the batch: its slots, each a sequence of at most batch_ctx() positions.
class BatchBackend {
public:
    virtual ~BatchBackend() = default;
    virtual int batch_slots() const = 0;
    virtual int64_t batch_ctx() const = 0;
    /// The live sequence's state (its position, its KV, the tokens its engram reads) copied into slot b; the live
    /// sequence stays as it is.
    virtual void batch_admit(int b) = 0;
    /// One step: row i feeds tokens[i] at slot slots[i]'s next position (B rows, distinct slots, 1 <= B <=
    /// batch_slots()); logits[i] [n_vocab] are that row's.
    virtual void batch_decode(const int* slots, const int32_t* tokens, int B, std::vector<std::vector<float>>& logits) = 0;
};

/// A slot's end, for its BDONE and log line.
struct BatchDone {
    int slot = -1;
    int64_t request = 0;               ///< the serve loop's request number
    int64_t generated = 0;             ///< every token of the request, its first (the admission's) included
    std::string finish;                ///< "stop", "length" or "cancel"
    double ms = 0;                     ///< from the admission to the end
    int64_t steps = 0, shared_steps = 0, rows = 0;   ///< steps it decoded in, of them beside another slot, their rows
    double stall_ms = 0;               ///< its steps' wait for other requests' prompts read between them
    bool deterministic = true;         ///< the request's mode
    bool reproducible = true;          ///< the mode's guarantee holds (the request is in the mode)
};

class BatchSlots {
public:
    /// eos: the ids that end a reply (Session's)
    BatchSlots(BatchBackend& b, std::vector<int32_t> eos);

    int slots() const { return (int) s_.size(); }
    int active() const;                                  ///< slots decoding
    bool busy(int b) const { return s_.at((size_t) b).on; }
    /// "" when a request of `prompt` tokens and max_new can go on in a slot after its first token, else why not
    /// (max_new 1: nothing to go on with; its positions past batch_ctx()).
    std::string fits(int64_t prompt, int64_t max_new) const;

    struct Admission {
        int64_t request = 0;
        int64_t position = 0;          ///< the live sequence's, which batch_admit copies
        int32_t first = 0;             ///< the token the admission picked (fed at the slot's first step)
        int64_t max_new = 0;
        std::unique_ptr<Sampler> sampler;   ///< the request's, null for greedy
        bool deterministic = true;
        std::chrono::steady_clock::time_point t0;   ///< its start (its BDONE's ms run from here)
    };
    /// Slot b (free) takes the request: the backend copies the live sequence into it. Exits naming the slot when it is
    /// busy, out of range, or the request does not fit.
    void admit(int b, Admission a);
    /// BSTOP: slot b ends before the next step (nothing when it is free).
    void stop(int b);
    /// Another request's prompt was read while the slots decoded: its ms count against each active slot's steps.
    void stalled(double ms);
    struct Events {
        std::function<void(int slot, int32_t token)> token;   ///< each generated token (BT)
        std::function<void(const BatchDone&)> done;           ///< each slot's end (BDONE), after its last token
        std::function<void()> flush;                          ///< the step's lines out (once, after its last token;
                                                              ///< after the stopped slots' ends when none is left)
    };
    /// The stopped slots end, then one step over the active ones (none: nothing); returns the step's rows.
    int step(const Events& ev);

    /// Since the slots were last all idle: the steps, their rows, their ms and the prompts read between them.
    /// where a step's wall time goes beyond the backend's step (ms): the picks of the
    /// rows' tokens (pick_ms), their lines and the flush (out_ms), the admissions' copies into their slots (admit_ms),
    /// and from one step's end to the next one's start whatever the prompts read between them did not hold (loop_ms:
    /// the serve loop, its stdin poll, a request's own lines and admission), over `gaps` such intervals.
    struct Totals {
        int64_t steps = 0, rows = 0, admissions = 0, stalls = 0, gaps = 0;
        double ms = 0, stall_ms = 0, pick_ms = 0, out_ms = 0, admit_ms = 0, loop_ms = 0;
    };
    const Totals& totals() const { return totals_; }
    void reset_totals() { totals_ = Totals{}; }

private:
    struct Slot {
        bool on = false, stop = false;
        Admission a;
        int32_t next = 0;
        int64_t produced = 0, pos = 0, steps = 0, shared_steps = 0, rows = 0;
        double stall_ms = 0;
    };
    BatchBackend& b_;
    std::vector<int32_t> eos_;
    std::vector<Slot> s_;
    std::vector<std::vector<float>> logits_;
    Totals totals_;
    // the last step's end (none before the first), and the ms since then the prompts read and the admissions
    // account for (not the loop's)
    std::chrono::steady_clock::time_point last_end_{};
    bool stepped_ = false;
    double accounted_ms_ = 0;
    void end(int b, const char* finish, const Events& ev);
};

/// The greedy pick of the engine, the slots' and a request alone's (Session's): the argmax, ties to the lower id (a NaN
/// never wins; a NaN at id 0 is kept, as the plain loop keeps it). Eight lanes,
/// each the plain loop's rule over its ids, then the lanes' best by value and id: the plain loop's pick, a step of four
/// rows of 129280 logits in 0.15 ms against its 1.50 (dev VM, Ryzen 7 5700X, -O3; the loop's dependence on the last
/// best kept it serial). Amendment 1 (2): the Session's pick too, the same token, one row in about 0.04 ms against
/// 0.375 (the same figures over one row).
int32_t batch_argmax(const std::vector<float>& logits);

}  // namespace strata::dsv41::engine
