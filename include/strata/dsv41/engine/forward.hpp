// include/strata/dsv41/engine/forward.hpp - the layer loop of the engine: a prompt of token ids through every
// layer in chunks, then greedy decode, on one stream, with every buffer allocated at init (the KV cache from ctx).
//
// Per chunk of T tokens at positions p0 .. p0 + T - 1 (SPEC.md sections 1-8, dense.hpp's order):
//   x = embed(tokens), pre_prev = the one-hot hc_pre_init
//   for l in 0 .. n_layer - 1:
//     engram layers (1 and 14):  x += value * gate (the rows hashed and gathered on the host for the chunk)
//     (pre_a, post, comb) = hc_mixes(x, hc_attn);  h = rms(collapse(x, pre_prev)) * attn_norm   (the V4.1 lag)
//     x = expand(attn_prefill or attn_decode(h), x, post, comb)
//     (pre_f, post, comb) = hc_mixes(x, hc_ffn);  h = rms(collapse(x, pre_a)) * ffn_norm
//     (ids, w) = router(h);  x = expand(experts(h, ids, w) + shared(h), x, post, comb);  pre_prev = pre_f
//   logits = head(x of the last token, pre_prev of the last token)
// A prefill chunk runs every expert pair on the GPU, a decode step splits the hits from the misses (experts.hpp).
//
// Starting point: Strata's prefill skeleton (src/prefill/prefill.cpp init, carve
// and run_impl: every buffer sized once from the largest chunk, the chunks run layer by layer against the KV cache)
// and its decode step (layer.cpp), with DeepSeek-V4.1's sublayers from attention.hpp, dense.hpp and experts.hpp.
//
// chunks up to 16384 tokens (every buffer from cfg.chunk), one qmatmul context
// for the stream (attention, dense ops and experts share its dequant scratch), the expert copies overlapped
// (experts.hpp).
//
// a decode token is one CUDA graph, Strata's session.hpp token graph (session_capture_token /
// session_run_token): every layer's engram, hyper-connections, attention (attn_decode on the position in device
// memory), router, the experts' doorbell, shared expert, hits, wait and combine, then the head, captured at the first
// untimed decode call and replayed at every later one. The pointers it bakes in are fixed (every buffer, and the
// collapse weights' two buffers reset at each call); what changes per token reaches it through mapped pinned host
// memory its first launch reads: the token, the position, the engram's gathered rows (hashed and copied on the host
// before the launch) and, in the parity mode or a synthetic routing, every layer's router ids and index rows. While it
// runs the host serves each layer's doorbell in order (ExpertEngine::decode_serve: the misses on the CPU pool), then
// waits once, for the logits, which its last launch writes into mapped memory. With --dump-layers the graph also
// copies each layer's dumped values into per-layer buffers, written once the token is done.
//
// Decode (27.5 tok/s projected against the 30 targeted; attention 0.40, hyper-
// connections and norms 0.13 ms per layer, the head 0.58 ms per token): a decode token's sublayers take their fused
// launches, the timed and --no-graphs paths the same ones as the graph. Each sublayer's hyper-connections are one
// hc_decode_step (the mixes, the collapse and norm, and the q8_1 of the normed row that the attention, the shared
// expert and the routed hits read instead of quantizing it again), with the expansion of the previous sublayer's
// output folded in: the attention's into the FFN's step, the FFN's into the next layer's attention step (except
// before an engram layer, which adds to the expanded stream, and with the dumps, which read the stream after every
// layer). The attention is attn_decode's fused token (attention.hpp), the shared expert dense_shared_expert's decode
// path. profile_decode_attention and profile_head time each layer's attention and the head alone in graphs (the
// smoke's per-kernel split).
//
// the per-request hot set (experts.hpp, hot_policy.hpp): begin_request starts a request's policy (before its
// prefill), the first decode call after a prefill tells the experts the prompt is read, every decode token starts the
// next batch of swap copies right after its own device work is queued (ExpertEngine::decode_started: the copies run
// on the copy streams beside the token) and ends with the experts' boundary (the swaps whose copies are done
// published for the next token, a batch still copying left open), after the call's wait for the stream. A first
// measurement (copy only 2.7-2.9x the static token, no boundary ever found a batch copying): the
// token's inputs and logits were cudaMemcpyAsync calls, which the copy engines ran after the batch's copies queued
// before them in the same direction; they now cross through mapped memory, moved by the device's threads.
// the token graph's launch waits first for an open layered reset of the hot set (ExpertEngine::
// wait_restore_all), since the graph itself holds no event wait.
//
// the prefix cache (session.hpp drives it): reset() starts the sequence again at
// position 0, save_checkpoint(slot) keeps the running state at the current position, restore_checkpoint(slot)
// continues from it: the engram's n-gram history is the slot's token history, and the token graph replays unchanged
// (its pointers are the same buffers, the position reaches it through pinned memory).
// cfg.checkpoints slots of KvCache's in pinned host RAM (kv_cache.hpp: one
// restore copy host to device, or in place for the conversation the device still holds).
//
// batched decode (batch.hpp drives it): cfg.batch slots, each a sequence of
// cfg.batch_ctx positions with a KV cache of its own (a ring of n_swa cells: a slot only decodes) and its engram
// history. batch_admit(b) copies the live sequence into slot b (KvCache::copy_state_from, device to device);
// decode_batch takes one token per row of B distinct slots in one CUDA graph per batch size B (1 to 8): B = 1 is the
// token graph's own sequence on the slot's cache (token_body, the same launches as a request decoding alone), B > 1
// the batch's (batch_body): every row through each op in one launch where the op takes T rows with each row's value
// its value at one row (the embedding, the engram, hc_prefill_step with the rows' q8_1, the router's GEMV, the shared
// expert, the head: their per-row arithmetic and qmatmul's MMVQ, whose column sums do not depend on the column count),
// the attention as attn_decode_batch (the projections batched, each row's cache stages on its own buffers), the routed
// experts as one call of B tokens (the hits grouped by expert, the misses by expert on the CPU). So a
// row's token is the one it would get decoding alone under the same placement. A graph bakes in its
// slots' cache pointers: when the slots of a batch size change it is captured again and cudaGraphExecUpdate swaps the
// launches' arguments into the instantiated graph (the same launches in the same order), or it is instantiated anew.
//
// a prompt read in chunks between batch steps (session.hpp's between):
// the live sequence's prefill and the slots' steps share no state but the expert slab (each call starts from its
// embedding, the live sequence's KV is not a slot's), so a step between two chunks leaves the chunks' numerics as they
// are alone. hold_prompt(true) keeps the experts' prompt target (ExpertEngine::prompt_done) for the prompt's end: the
// steps between its chunks do not end it, so a prompt read in chunks gets one prompt target, as one read whole.
// a prompt read beside decoding slots (read_beside(true)) takes no target of its
// own at its end: its counts join the batch's (ExpertEngine::merge_prompt, hot_policy.hpp), so the slab keeps
// following the rows' routes and adds the new row's.
#pragma once

#include "strata/dsv41/engine/attention.hpp"
#include "strata/dsv41/engine/batch.hpp"
#include "strata/dsv41/engine/dense.hpp"
#include "strata/dsv41/engine/engram.hpp"
#include "strata/dsv41/engine/experts.hpp"
#include "strata/dsv41/engine/kv_cache.hpp"
#include "strata/dsv41/engine/model.hpp"
#include "strata/dsv41/engine/selection.hpp"
#include "strata/dsv41/engine/session.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

struct ForwardConfig {
    int64_t ctx = 4096;                     ///< positions of the sequence (the KV cache's size)
    int64_t chunk = 4096;                   ///< the largest prefill chunk
    int64_t engram_chunk = 1024;            ///< the engram projection's sub-chunk (dense.hpp)
    /// conversation checkpoint slots, each pinned for a checkpoint of ctx positions (kv_cache.hpp); the one by
    /// default, dsv41-engine --serve's --checkpoints (4), 0 for a run that keeps none
    int checkpoints = 1;
    /// batch slots (0: none; dsv41-engine --serve --batch, at most kMmvqMaxCols) and the positions each holds
    /// (--batch-ctx; 32768 by default: the 16K conversations with their 2 turns of 512 + 256 tokens fit with room)
    int batch = 0;
    int64_t batch_ctx = 32768;
    ReaderMode readers = ReaderMode::Reuse; ///< --readers dense is the fork-parity mode
    bool timing = false;                    ///< per-stage times from CUDA events, read once the call is done
    bool graphs = true;                     ///< decode replays the token graph (not while timing)
    /// per chunk and step: every layer's l_last, its router's ffn_moe_topk and ffn_moe_probs_biased, an index
    /// source's idx_top_k and idx_score, and the logits ("": none), in the fork's op-dump names; also the MoE
    /// layer's input ffn_norm [T][E], its weights ffn_moe_weights_scaled [T][K] and the routed experts' sum
    /// ffn_moe_out [T][E] before the shared expert (the real-model oracle's cur, weights and out)
    std::string dump_dir;
    /// false: the dumps hold only what the parity mode reads back (the router's and the index sources' selections,
    /// ffn_moe_topk and idx_top_k) and the logits; l_last, the biased probs, idx_score and the MoE values (a 16K
    /// prompt's are GiBs)
    /// are left out (the prefix-cache smoke)
    bool dump_values = true;
    /// the parity mode (--force-selection): every call's router ids and index rows from this reference instead of
    /// the engine's own (selection.hpp); its calls must be the engine's chunks and steps (null: none)
    const SelectionDump* forced = nullptr;
    /// the isolated MoE parity (needs forced): every layer's MoE input h
    /// (ffn_norm) is replaced by the reference's (SelectionDump::moe_input) before the router, so the router, the
    /// shared expert and the routed experts read the reference's input and their outputs (the dumps' ffn_moe_out and
    /// weights) measure that path alone. Decode steps then launch every op (no token graph); the hits and the
    /// shared expert quantize the replaced h themselves.
    bool forced_moe_input = false;
    /// The prefill matmuls' path above kMmvqMaxCols tokens (attention and dense ops): the default dequant path
    /// (f16 activations) or QPrefill::Mmq, the fork's own ggml-cuda arithmetic on int8
    /// activations (how much of the real-model parity gap the prefill's activations make)
    QPrefill prefill = QPrefill::DequantF16;
    /// a synthetic routing (dsv41_engine_smoke --counts): every call's router ids [T][K] of layer il at
    /// positions p0 .. from this function instead of the router's top-k (distinct within a token); the weights are the
    /// router's for those experts, as with forced (empty: none; not with forced)
    std::function<void(int il, int64_t p0, int64_t T, std::vector<int32_t>& ids)> routing;
};

/// Per-stage times of the calls since the last reset (ms; with timing on), the host's waits on the device (the
/// forward's own for the logits, the experts' and the engram gathers'; the dumps' are not counted) and the graphs.
struct ForwardTimes {
    double embed_ms = 0, engram_ms = 0, attention_ms = 0, hc_ms = 0, router_ms = 0, experts_ms = 0, shared_ms = 0,
           head_ms = 0, total_ms = 0;
    int64_t calls = 0, tokens = 0, host_syncs = 0, graph_launches = 0;
    /// where a token graph's wall time goes, summed over its calls. Host: the inputs
    /// filled and the graph launched (launch_ms), the doorbells served (serve_ms: their polls and the CPU's misses,
    /// ExpertTimes ring_wait_ms and cpu_ms), from the last doorbell to the stream's end and the logits read
    /// (tail_ms); the next batch of swap copies queued between the launch and the first doorbell is ExpertTimes
    /// swap_queue_ms. Every decode call's experts' boundary (boundary_ms, host). Device: the graph from an event
    /// queued right before its launch to one right after it (graph_ms).
    double launch_ms = 0, serve_ms = 0, tail_ms = 0, boundary_ms = 0, graph_ms = 0;
};

/// Forward::profile_decode_attention: per layer, the device time of its decode attention per replay of
/// a graph of it alone, and the split by fused launch (AttnKernel) when the graph's event nodes could be read.
struct ForwardAttnProfile {
    int reps = 0;
    std::vector<double> layer_ms;
    std::vector<std::vector<double>> kernel_ms;   ///< [layer][kAttnKernels]
    bool split = false;
};

/// Forward::profile_decode_hc: per layer the device ms per replay of its two hc_decode_step in a graph of their
/// own, and the steps' phase split (hc_step_split) averaged over every layer's two steps, each step alone in a graph
/// with its marks on, replayed reps times.
struct ForwardHcProfile {
    std::vector<double> layer_ms;
    HcStepSplit split;
    int64_t marked = 0;   ///< marked launches averaged into split
};

/// One layer's device times summed over the calls since the last reset (timing on): its attention, the other
/// sublayer ops before its routed experts (engram, hyper-connections and norms, router, shared expert), the experts.
struct ForwardLayerTimes {
    double attention_ms = 0, dense_ms = 0, experts_ms = 0;
};

class Forward {
public:
    Forward();
    ~Forward();
    Forward(const Forward&) = delete;
    Forward& operator=(const Forward&) = delete;

    /// The engines (attention, dense, experts), the KV cache and every buffer on a stream of its own. False with
    /// `err` set when the device has no room or a size is out of range.
    bool init(const Model& m, const ExpertConfig& ecfg, const ForwardConfig& cfg, std::string& err);

    /// n tokens at the sequence's next positions, in chunks of up to cfg.chunk; logits [n_vocab] of the last one.
    void prefill(const int32_t* tokens, int64_t n, std::vector<float>& logits);
    /// One token at the next position; logits [n_vocab].
    void decode(int32_t token, std::vector<float>& logits);

    int64_t position() const { return pos_; }
    /// the sequence back to position 0 (the KV cache's contents stay; nothing reads them before they are written).
    void reset();
    /// the running state at the current position into checkpoint slot `slot` (KvCache::save_checkpoint), with the
    /// token history the engram reads.
    void save_checkpoint(int slot);
    /// The sequence back to slot `slot`'s state (KvCache::restore_checkpoint); exits when it was never saved.
    void restore_checkpoint(int slot);
    int checkpoints() const;
    /// a slot's cost (KvCache::checkpoint_times: bytes, copy ms, in place) and its tokens.
    KvCache::CheckpointTimes checkpoint_times(int slot) const;
    /// One slot's pinned bytes, and the device bytes the in-place restore keeps (kv_cache.hpp).
    size_t checkpoint_slot_bytes() const;
    size_t checkpoint_device_bytes() const;
    /// the batch slots (ForwardConfig::batch) and the positions each holds.
    int batch_slots() const;
    int64_t batch_ctx() const;
    /// The live sequence's state (position, KV, engram history) into slot b, device to device on the stream.
    void batch_admit(int b);
    /// One step: row i feeds tokens[i] at slot slots[i]'s next position (B distinct slots, 1 <= B <= batch_slots());
    /// logits[i] [n_vocab] are row i's.
    void decode_batch(const int* slots, const int32_t* tokens, int B, std::vector<std::vector<float>>& logits);
    int64_t batch_position(int b) const;
    /// One slot's device bytes (its KV cache) and the batch's other device buffers (its logits, the attention's rows).
    size_t batch_slot_bytes() const;
    /// The batch graphs captured anew (instantiated, or updated in place) since init.
    int64_t batch_captures() const { return batch_captures_; }
    int64_t batch_updates() const { return batch_updates_; }
    /// The parity mode's reference from the next call on (init must have had one: its buffers; not null).
    void set_forced(const SelectionDump* forced);
    /// a request starts with hot-set policy p (ExpertEngine::begin_request; reset: the slab back to the hot set's;
    /// resume: the request reuses the sequence's prefix, so the planners keep their counts; deterministic: the
    /// request's placement fixed by its own prompt and tokens, hot_policy.hpp). It opens its prompt, so the first
    /// decode call ends it even when no prefill reads a token of it (hot_policy.hpp HotPrompt).
    void begin_request(HotPolicy p, bool reset, bool resume, bool deterministic);
    /// a request ended (ExpertEngine::end_request: the deterministic mode's reset, in the idle time after it).
    void end_request();
    /// on while a prompt is read in chunks between batch steps: decode_batch leaves its prompt target for later.
    void hold_prompt(bool on) { prompt_held_ = on; }
    /// on while a prompt is read beside decoding batch slots: its end (the first
    /// decode call after it, in a slot or alone) merges its counts into the batch's (ExpertEngine::merge_prompt)
    /// instead of taking its own target (prompt_done). Taken at each prefill call.
    void read_beside(bool on) { read_beside_ = on; }
    /// the next prompts' mode for the experts' prefill split (ExpertEngine::prompt_deterministic).
    void prompt_deterministic(bool on);
    /// the hot set's knobs from now on (ExpertEngine::set_hot).
    void set_hot(const HotConfig& c);
    /// --route-trace: false leaves the next calls out of the trace (the serve protocol's trace=0).
    void set_trace(bool on);
    /// The dump files written so far: one per chunk or step, its first position and tokens.
    struct DumpStep {
        int64_t pos0 = 0, n_tokens = 0;
    };
    const std::vector<DumpStep>& dump_steps() const { return dump_steps_; }
    /// times_ with host_syncs counting the experts' and the gathers' too.
    ForwardTimes times() const;
    const std::vector<ForwardLayerTimes>& layer_times() const { return layer_times_; }
    void reset_times();
    /// Timing on or off from the next call (decode's token graph runs only with it off).
    void set_timing(bool on);
    /// The experts' CPU pool spin from the next call (ExpertConfig::pool_spin_us).
    void set_pool_spin_us(int us);
    const ExpertEngine& experts() const;
    /// between decode tokens (not with a forced selection): every layer's attn_decode at the next
    /// position, in layer order, captured with the per-kernel marks into a graph of its own and replayed reps times
    /// after one warm-up. The layers' cache bookkeeping is restored afterwards, and what the replays write (the
    /// position's window cell, its pooling row or block, an index source's top-k) the next token writes again.
    ForwardAttnProfile profile_decode_attention(int reps);
    /// The head's device ms per call (dense_head on the stream's last x, captured and replayed reps times).
    double profile_head(int reps);
    /// between decode tokens: per layer the device ms of its hyper-connections and norms as the token graph runs
    /// them (the attention's hc_decode_step, with the previous FFN's expansion folded in where the graph folds it, and
    /// the FFN's), the two captured into a graph of the layer's own and replayed reps times after one warm-up. What
    /// they write (the stream, the coefficients, the normed rows) the next token writes again from its embedding.
    /// Then each step alone, its phase marks on (ForwardHcProfile::split).
    ForwardHcProfile profile_decode_hc(int reps);
    /// Device bytes by part: KV cache, attention, dense and expert workspaces, the stream's buffers.
    void print_memory(std::FILE* out) const;

    struct Impl;

private:
    void run(const int32_t* tokens, int64_t T, std::vector<float>& logits);
    void run_graph(int32_t token, std::vector<float>& logits);
    /// the experts' prompt target once a request began or a prefill ran (decode's and decode_batch's first token
    /// after it)
    void prompt_read();
    /// snapshot: the token graph's per-layer copies instead of the live buffers
    void dump_layer(int il, int u, int64_t p0, int64_t T, bool snapshot);

    std::unique_ptr<Impl> impl_;
    int64_t pos_ = 0;
    std::vector<int32_t> history_;
    std::vector<std::vector<int32_t>> slot_history_;   // each checkpoint's token history
    std::vector<DumpStep> dump_steps_;
    HotPrompt prompt_;            // a prefill ran (or a request began) since the last decode call
    bool prompt_held_ = false;    // a prompt read in chunks between batch steps (hold_prompt)
    bool read_beside_ = false;    // prompts read now are beside decoding batch slots (read_beside)
    ForwardTimes times_;
    std::vector<ForwardLayerTimes> layer_times_;
    int64_t gather_syncs0_ = 0;   // the engram gather's waits at the last reset
    // each batch slot's position and engram history
    std::vector<int64_t> batch_pos_;
    std::vector<std::vector<int32_t>> batch_history_;
    int64_t batch_captures_ = 0, batch_updates_ = 0;
};

/// a Session's device (session.hpp) on a Forward: its context, chunk and vocabulary, its prefix-cache calls;
/// the batch slots' device (batch.hpp).
class ForwardBackend final : public SessionBackend, public BatchBackend {
public:
    ForwardBackend(Forward& fw, const ForwardConfig& cfg, int64_t n_vocab)
        : fw_(fw), ctx_(cfg.ctx), chunk_(std::min(cfg.chunk, cfg.ctx)), n_vocab_(n_vocab) {}
    int64_t ctx() const override { return ctx_; }
    int64_t chunk() const override { return chunk_; }
    int64_t n_vocab() const override { return n_vocab_; }
    void reset() override { fw_.reset(); }
    int checkpoints() const override { return fw_.checkpoints(); }
    void save_checkpoint(int slot) override { fw_.save_checkpoint(slot); }
    void restore_checkpoint(int slot) override { fw_.restore_checkpoint(slot); }
    void prefill(const int32_t* t, int64_t n, std::vector<float>& logits) override { fw_.prefill(t, n, logits); }
    void decode(int32_t t, std::vector<float>& logits) override { fw_.decode(t, logits); }
    int batch_slots() const override { return fw_.batch_slots(); }
    int64_t batch_ctx() const override { return fw_.batch_ctx(); }
    void batch_admit(int b) override { fw_.batch_admit(b); }
    void batch_decode(const int* slots, const int32_t* tokens, int B, std::vector<std::vector<float>>& logits) override {
        fw_.decode_batch(slots, tokens, B, logits);
    }

private:
    Forward& fw_;
    int64_t ctx_, chunk_, n_vocab_;
};

/// The dry run (--dry-run, the dev VM: no CUDA call): the model's views and the experts' tiers were checked by their
/// dry inits; this plans every call of a run of n_prompt tokens and n_predict decode steps: the chunks with their
/// positions against the KV cache's size, each layer's attention call (role, reader mode, rows it reads), the engram
/// rows of the prompt hashed and checked against the tables, the experts' prefill and decode plans
/// (ExpertEngine::dry_plan) and the hot set's reset (ExpertEngine::dry_restore) and, with cfg.forced, every call's reference selections read and checked against the
/// call and the layer. Exits naming the problem.
void forward_dry_run(const Model& m, const ExpertEngine& experts, const ForwardConfig& cfg, const int32_t* tokens,
                     int64_t n_prompt, int64_t n_predict, std::FILE* out);

/// the batch slots' device KV (cfg.batch slots of cfg.batch_ctx positions, a ring of n_swa cells; kv_cache_bytes,
/// no CUDA call, the dry run's figure too), one line beside the memory lines; nothing without slots.
void print_batch_cost(std::FILE* out, const Model& m, const ForwardConfig& cfg);

/// the fixed host RAM beside the checkpoints, the hot set's host homes (pinned host RAM held once
/// for the engine, not a conversation's cost: a further conversation adds its checkpoint slot only; nothing with
/// --hot-homes spare; --hot-homes N names the homes held of the slab's experts), one line after the
/// checkpoints' cost line.
void print_host_homes_cost(std::FILE* out, const ExpertEngine& experts);

}  // namespace strata::dsv41::engine
