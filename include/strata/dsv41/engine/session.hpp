// include/strata/dsv41/engine/session.hpp - the resident conversations of the engine: a request of token
// ids against the sequence the engine holds and the conversations it keeps, the prefix it can reuse, the prompt's
// remaining tokens read in chunks and the decode (greedy, or the host sampler for a sampled request), for
// `dsv41-engine --serve` (Strata's serve protocol) and the prefix-cache smoke.
//
// The prefix cache: the engine keeps the
// live state after a request (its prompt and every generated token it fed) and N conversation checkpoints, each the
// state at the end of a request's prompt (backend.checkpoints() slots, `dsv41-engine --checkpoints`, default 4). A
// request resumes from the candidate with the longest prefix of its ids: the live tokens (RESUME of all of them), or a
// checkpoint (RESUME of its prompt: the backend restores it), else it starts from position 0. Each candidate's match
// is Strata's rule, core::conversation_prefix (core/conversation_cache.hpp, the Qwen serve loop's parked
// conversations): a checkpoint's ids a strict prefix of the request's, since the request's next token's logits come
// from reading at least one token; equal lengths prefer the live state (no copy), then the checkpoint used last.
// After the prompt is read its end is saved into a slot (unless the request says ckpt=0): the slot of a checkpoint
// with the same ids, else every checkpoint the new one supersedes (one whose ids its own extend: the same
// conversation a turn back, as Strata's ConversationCache::drop_superseded) is dropped and the first one's slot taken,
// else a free slot, else the least recently used checkpoint's (evicted). A checkpoint is used when it is restored or
// saved. With one slot this is the single-conversation cache: the live state and the last prompt's end.
// Strata's serve loop (src/program/generate.cpp, the --serve block) decides the same way between its live session and
// its conversation checkpoints. Its decode is the same too: each generated token is printed, and fed only when another
// token is wanted, so the live state after a request is its prompt and every generated token but the last.
//
// The batch decodes several conversations at once over the same slots: a ConversationSlot names one (its slot id, the ids
// whose end it holds, so its length and position), the backend's buffers of that id hold its state.
//
// a request may read its prompt in chunks of a fixed size of its own
// (SessionRequest::chunk, at most the backend's), with SessionEvents::between called after every chunk but the last:
// the serve loop runs one batch step there, so the slots decoding beside a long prompt are not held for all of it.
// The chunk boundaries depend on the request alone, so the request reads as it would alone with the same chunks.
//
// the live state keeps the token the last request emitted and did
// not feed (its last token, whatever ended it). A request whose live resume leaves exactly that one token to read (a
// request stopped on the solo path and promoted into a batch slot, its prompt and its tokens) feeds it with decode, the
// call the request alone would have made at that position, not prefill(T=1), whose prefill expert path and
// hyper-connection step write the position with other last bits (an RTX PRO 6000 session of 2026-10-17: the promoted
// chain left its alone run at token 67, the other 9 batched chains matched theirs). The dsv41 engine has no copy of a
// slot back to the live sequence (INFO slot_cache=0: the server never sends a slot back to the solo path), so a
// request goes back to the live sequence only as a new read; were a copy-back added, it would set the live tokens and
// this token from the slot so that the same rule feeds its next token. Any longer tail (a conversation's next turn)
// is read as a prompt, as alone.
//
// The backend is the device side (Forward, forward.hpp) or the dry run's stand-in (dsv41-engine --serve --dry-run),
// behind SessionBackend; this file holds the policy only and makes no CUDA call.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

class Sampler;

/// The device side of a session: the sequence's state, each call's logits of its last token.
class SessionBackend {
public:
    virtual ~SessionBackend() = default;
    virtual int64_t ctx() const = 0;      ///< positions a sequence may reach
    virtual int64_t chunk() const = 0;    ///< the largest prefill call
    virtual int64_t n_vocab() const = 0;
    virtual void reset() = 0;             ///< back to position 0
    virtual int checkpoints() const = 0;  ///< the conversation checkpoint slots (0: none kept)
    /// the running state at the current position into slot (0 .. checkpoints() - 1), replacing what it held
    virtual void save_checkpoint(int slot) = 0;
    /// the running state back to what slot holds (saved before)
    virtual void restore_checkpoint(int slot) = 0;
    /// n tokens (1 .. chunk()) at the next positions; logits [n_vocab] of the last one
    virtual void prefill(const int32_t* tokens, int64_t n, std::vector<float>& logits) = 0;
    /// one token at the next position; logits [n_vocab]
    virtual void decode(int32_t token, std::vector<float>& logits) = 0;
};

enum class ResumeKind { Cold, Live, PromptEnd };
const char* resume_kind_name(ResumeKind k);

struct ResumePlan {
    ResumeKind kind = ResumeKind::Cold;
    int64_t reuse = 0;   ///< the request's first tokens not read again
    int slot = -1;       ///< PromptEnd: the checkpoint restored
};

/// one cached conversation: the backend's slot `id` holds the state at the end of `ckpt.ids` (its length, the
/// position it continues at); ckpt.used orders the slots (the least recently used is evicted first).
struct ConversationSlot {
    int id = -1;
    strata::core::ConversationCheckpoint ckpt;
    int64_t position() const { return (int64_t) ckpt.ids.size(); }
    bool held() const { return !ckpt.ids.empty(); }
};

struct SessionRequest {
    std::vector<int32_t> ids;
    int64_t max_new = 0;         ///< tokens to generate (>= 1)
    bool snapshot = true;        ///< keep the prompt's end in a checkpoint for a later request (Strata's ckpt=0: false)
    /// the parity mode (the smoke's forced runs): decode step s feeds feed[s] instead of the token it picked, as
    /// dsv41-engine --force-selection feeds the reference's tokens (empty: the picks; else at least max_new - 1)
    std::vector<int32_t> feed;
    /// a sampled request's sampler (sampler.hpp), which picks every generated token; null: greedy, the argmax
    Sampler* sampler = nullptr;
    /// the prompt's chunks (1 .. backend.chunk()); 0: the backend's chunk, today's read
    int64_t chunk = 0;
};

/// What the serve loop prints as a request runs (each may be empty).
struct SessionEvents {
    std::function<void(const ResumePlan&)> resume;                           ///< before the first read
    std::function<void(int64_t read, int64_t total, double ms)> chunk;       ///< after each prompt chunk (PP)
    std::function<void(const std::vector<float>& logits)> prompt_logits;     ///< the prompt's last token's logits
    std::function<void(int32_t token)> token;                                ///< each generated token (T)
    std::function<bool()> stop;                                              ///< true: end the request (STOP)
    std::function<void()> between;   ///< after each prompt chunk but the last (the serve loop's batch step)
};

struct SessionResult {
    ResumePlan plan;
    /// the slot the prompt's end was saved to (-1: none), the checkpoints it dropped as superseded, and the one it
    /// evicted (the least recently used, -1: none) with its tokens
    int saved = -1, superseded = 0, evicted = -1;
    int64_t evicted_tokens = 0;
    int64_t generated = 0, read = 0;   ///< tokens generated; prompt tokens read (fewer than ids - reuse when stopped)
    double prompt_ms = 0, decode_ms = 0, first_token_ms = 0;   ///< first_token_ms: the request's start to its first token
    std::string finish;                ///< "stop" (an end-of-sequence id), "length" or "cancel"
    /// the one token read was the last request's emitted and unfed token, fed with decode (above)
    bool fed_unfed = false;
    std::vector<int32_t> tokens;
};

class Session {
public:
    /// eos: the ids that end a reply (the GGUF's tokenizer.ggml.eos_token_id; dsv41-engine --eos-ids)
    Session(SessionBackend& b, std::vector<int32_t> eos);

    /// "" when the request can run, else why not (Strata's ERR line): empty ids, max_new < 1, an id outside the
    /// vocabulary, more positions than ctx (the prompt and every generated token but the last), a feed too short, a
    /// chunk past the backend's.
    std::string check(const SessionRequest& r) const;
    /// The prefix a request of these ids reuses (no state change).
    ResumePlan plan(const std::vector<int32_t>& ids) const;
    /// Runs a request that passed check(): resume, read, decode, the checkpoint at the prompt's end.
    SessionResult run(const SessionRequest& r, const SessionEvents& ev);
    /// Whether a plan continues the conversation of the last request run (a live resume, or the checkpoint it saved):
    /// the planners keep their counts only then (hot_policy.hpp).
    bool continues_last(const ResumePlan& p) const;

    /// The tokens the backend's state holds (positions 0 ..).
    const std::vector<int32_t>& live() const { return live_; }
    /// The tokens of the checkpoint the last request saved (0: it saved none).
    int64_t snapshot_tokens() const { return last_saved_ < 0 ? 0 : slots_[(size_t) last_saved_].position(); }
    /// every slot, held or free (backend.checkpoints() of them).
    const std::vector<ConversationSlot>& slots() const { return slots_; }
    /// Checkpoints evicted (the least recently used, to make room) since the start.
    int64_t evictions() const { return evictions_; }

private:
    SessionBackend& b_;
    std::vector<int32_t> eos_, live_;
    std::vector<ConversationSlot> slots_;
    uint64_t clock_ = 0;
    int last_saved_ = -1;
    int32_t unfed_ = -1;   ///< the token the last request emitted and did not feed (-1: none, or another run since)
    int64_t evictions_ = 0;
    void touch(ConversationSlot& s) { s.ckpt.used = ++clock_; }
    /// the slot a prompt's end goes to, the superseded dropped and an eviction counted in r
    int pick_slot(const std::vector<int32_t>& ids, SessionResult& r);
};

}  // namespace strata::dsv41::engine
