// include/strata/dsv41/engine/selection.hpp - a reference run's discrete selections for the engine's forced-selection
// parity mode (`dsv41-engine --force-selection DIR`): the router's top-k experts and the index sources' top-k
// rows of every call and layer, read from a fork op dump (tools/dsv41/engine/op-dump, dump-manifest.json) or from
// another engine run's --dump-layers directory (engine-manifest.json).
//
// Why: on the tiny random models the router and the indexer meet near-ties (the fork's k-th and (k+1)-th scores
// 1.7e-5 to 4e-3 apart at the tokens the first dev pod saw jump, README.md), and the engine's numerics (f16 or
// int8 per 32 activations, SPEC.md "Engine numerics and fork numerics") pick the other side of some of them; one
// flipped expert or row moves that token by 0.2 to 0.7 rel L2 and everything downstream of it. With the reference's selections every call computes the same function as the reference, so every
// layer and token must land within the engine's numerics bound; what remains above it is a bug.
//
// Files, per call u and layer l, as the fork's op-dump names them (ggml ne reversed, raw little endian):
//   ffn_moe_topk.0-L<l>-u<u>.bin   [T][K] i32, the argsort's first K: descending by probs + bias (the slot order)
//   idx_top_k.0-L<l>-u<u>.bin      index sources: [T][k] i32, the selected compressed rows in any order; entries at
//                                  or past the position's visible count (and the engine's -1 padding) are ignored
//   ffn_norm.0-L<l>-u<u>.bin       [T][E] f32, the MoE layer's input (--forced-moe-input only: the
//                                  real-model oracle's `cur`, which oracle_parity.py forced copies under this name)
// and from the manifest: the prompt, each call's first position and tokens, and the token fed at each decode step
// (fed_token_ids of an engine manifest, else generated_token_ids: the fork feeds what it generates).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

class SelectionDump {
public:
    /// Reads DIR's manifest (dump-manifest.json, else engine-manifest.json). Exits naming the problem.
    static SelectionDump open(const std::string& dir);

    const std::string& dir() const { return dir_; }
    const std::vector<int32_t>& prompt() const { return prompt_; }
    /// The token the reference fed at decode step s (call s + 1).
    const std::vector<int32_t>& fed() const { return fed_; }
    int64_t n_calls() const { return (int64_t) calls_.size(); }

    /// The reference's call that holds positions p0 .. p0 + T - 1, so an engine chunk may be part of a reference
    /// ubatch (a prompt in several chunks against the fork's one ubatch); exits when none does.
    int64_t call_of(int64_t p0, int64_t T) const;
    /// ids [T][K] of positions p0 .. in reference call u (call_of), layer il: every id in 0 .. n_expert - 1 and
    /// distinct within a token, else exits.
    void router(int64_t u, int il, int64_t p0, int64_t T, int K, int n_expert, std::vector<int32_t>& ids) const;
    /// rows [T][top_k] of positions p0 .. in reference call u, layer il (an index source of compress ratio `ratio`):
    /// per token the reference's rows visible at its position (at most top_k), ascending, then -1, as the engine's
    /// indexer writes them (indexer.hpp). n_k: the rows the engine's call's key source holds.
    void index_rows(int64_t u, int il, int64_t p0, int64_t T, int ratio, int top_k, int64_t n_k,
                    std::vector<int32_t>& rows) const;
    /// rows [T][E] of positions p0 .. in reference call u, layer il: the reference's MoE input (ffn_norm), every
    /// value finite, else exits (the isolated MoE parity, forward.hpp ForwardConfig::forced_moe_input).
    void moe_input(int64_t u, int il, int64_t p0, int64_t T, int64_t E, std::vector<float>& rows) const;

private:
    std::vector<int32_t> read_i32(const char* op, int64_t u, int il) const;
    std::vector<float> read_f32(const char* op, int64_t u, int il) const;

    std::string dir_, manifest_;
    std::vector<int32_t> prompt_, fed_;
    struct Call {
        int64_t pos0 = 0, n_tokens = 0;
    };
    std::vector<Call> calls_;
};

}  // namespace strata::dsv41::engine
