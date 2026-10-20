// tests/dsv41/engine/checkpoint_test.cpp - the
// Session's conversation checkpoints (session.hpp) on the CPU reference path, GPU free.
//
//   dsv41_engine_checkpoint_test GGUF      (tiny-b: window, ratio 2 and ratio 1 layers, n_swa 16)
//
// The backend runs the GGUF's attention layers on the CPU reference (attention_ref.hpp, the engine's numerics and the
// Reuse readers): per call x = the tokens' embedding, then per layer x += attention(rms(x) * attn_norm), the logits
// the output matrix on rms(x of the last token) * output_norm; a call's output is its x rows after the last layer.
// Beside the reference's own state (RefKv) it keeps an image of the device's buffers (kv_cache.hpp: the window ring
// of n_swa - 1 + chunk cells, cell p % cells; the compressed rows and index keys at their row; the pooling carry of
// ratio rows), written from RefKv after every call as the device writes them. A checkpoint is what KvCache copies:
// kv_checkpoint_spans of the image at the prompt's end into a host buffer; a restore copies them back into the image
// (or, KvCache's in-place rule, only the carry when the slot is the one the image last saved or restored and the ring
// still holds its cells) and then rebuilds RefKv from the image alone: every window position before the last n_swa - 1
// poisoned with NaN, the ids of the index sources dropped. So a value a later call reads that no span restored makes
// its output differ from the straight read (or NaN).
//
// Checks (PASS or FAIL lines, exit 1 on any FAIL):
//   1. 4 conversations take 3 turns each, alternating (user 0 1 2 3, then again): each turn 2+ resumes from its own
//      checkpoint (RESUME of its last prompt, the slot it saved) though another user ran between, its state right
//      after the restore equals the straight read's (the same prompt's calls with nothing between them) bit for bit in
//      everything the next call reads, and its prompt's outputs and logits, so its first token, equal the straight
//      read's bit for bit;
//   2. a 5th conversation evicts the least recently used one (user 0's, the one used longest ago), whose next turn is
//      then read cold, bit for bit as a cold read of the same prompt;
//   3. the in-place restore: a turn that extends only the prompt of the request just run restores in place at most
//      a chunk past it (as the device's ring allows) and from the host one token further, both bit for bit;
//   4. one slot (--checkpoints 1, a single cache slot): the same alternation resumes nothing (every turn cold, the
//      baseline), a user alone resumes its live state;
//   5. the checks see what they guard against: a restore that leaves out one span's row (the oldest window cell, the
//      last compressed row, the last index key, a carry row) makes the resumed turn differ from the straight read.
#include "strata/dsv41/engine/attention_ref.hpp"
#include "strata/dsv41/engine/dense_ref.hpp"
#include "strata/dsv41/engine/kv_checkpoint.hpp"
#include "strata/dsv41/engine/model.hpp"
#include "strata/dsv41/engine/session.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "dsv41_engine_checkpoint_test: %s\n", m.c_str());
    std::exit(2);
}

int failures = 0;
void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

constexpr int64_t kChunk = 16;
constexpr uint16_t kPoison = 0x7e00;   // an f16 NaN

bool same_bits(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }

// The device's buffers of one layer as KvCache lays them out (kv_cache.hpp), on the host.
struct ImageLayer {
    std::vector<uint16_t> window, rows, keys;
    std::vector<float> carry_kv, carry_score;
};

class RefBackend final : public e::SessionBackend {
public:
    RefBackend(const e::Model& m, int checkpoints) : m_(m), kv_((size_t) m.n_layer()) {
        const e::AttnDims& ad = m.attn_dims();
        geom_ = e::KvCheckpointGeometry{ad, m.roles(), ad.n_swa - 1 + kChunk};
        const size_t D = (size_t) ad.head_dim;
        img_.resize((size_t) m.n_layer());
        for (int il = 0; il < m.n_layer(); ++il) {
            const e::AttnLayer& l = m.roles()[(size_t) il];
            ImageLayer& g = img_[(size_t) il];
            g.window.assign((size_t) geom_.window_cells * D, kPoison);
            const int64_t cap = l.ratio ? (ctx() + l.ratio - 1) / l.ratio : 0;
            if (l.kv_source) {
                g.rows.assign((size_t) cap * D, kPoison);
                g.carry_kv.assign((size_t) l.ratio * D, NAN);
                g.carry_score.assign((size_t) l.ratio * D, NAN);
            }
            if (l.key_owner) g.keys.assign((size_t) cap * ad.idx_dim, kPoison);
            const e::LayerWeights& L = m.layer(il);
            e::RefLayerWeights w;
            auto mat = [](const e::QWeightView& v) { return e::RefMatrix{v.type, v.k, v.rows, v.data, e::RefActs::Auto}; };
            w.wq_a = mat(L.attn.wq_a);
            w.wq_b = mat(L.attn.wq_b);
            w.wkv = mat(L.attn.wkv);
            w.wo_a = mat(L.attn.wo_a);
            w.wo_b = mat(L.attn.wo_b);
            w.comp_kv = mat(L.attn.comp_kv);
            w.comp_gate = mat(L.attn.comp_gate);
            w.idx_k = mat(L.attn.idx_k);
            w.idx_q_b = mat(L.attn.idx_q_b);
            w.idx_proj = mat(L.attn.idx_proj);
            w.q_a_norm = L.attn.q_a_norm;
            w.kv_norm = L.attn.kv_norm;
            w.comp_norm = L.attn.comp_norm;
            w.idx_k_norm = L.attn.idx_k_norm;
            w.sinks = L.attn.sinks;
            weights_.push_back(w);
        }
        slots_.resize((size_t) checkpoints);
    }

    int64_t ctx() const override { return 512; }
    int64_t chunk() const override { return kChunk; }
    int64_t n_vocab() const override { return m_.dense_dims().n_vocab; }
    void reset() override {
        kv_ = e::RefKv((size_t) m_.n_layer());
        inplace_slot_ = -1;
        resumed = kv_;
        resumed_set = true;
    }
    int checkpoints() const override { return (int) slots_.size(); }
    void save_checkpoint(int k) override {
        Slot& s = slots_.at((size_t) k);
        s.pos = pos();
        s.host.clear();
        for (const e::KvSpan& sp : e::kv_checkpoint_spans(geom_, s.pos)) {
            const uint8_t* src = span(sp);
            s.host.insert(s.host.end(), src, src + sp.bytes());
        }
        if (s.host.size() != e::kv_checkpoint_bytes(geom_, s.pos)) die("a checkpoint's bytes differ from its spans'");
        carry_snap_ = img_;   // the in-place snapshot (only the carry is read back from it)
        snap_pos_ = s.pos;
        inplace_slot_ = k;
    }
    void restore_checkpoint(int k) override {
        Slot& s = slots_.at((size_t) k);
        if (s.pos < 0) die("restore of a checkpoint never saved");
        // KvCache::restore_checkpoint's rule: in place for the slot the image last saved or restored while the ring
        // holds its cells, else every span from the host
        last_in_place_ = k == inplace_slot_ && pos() >= snap_pos_ && pos() - snap_pos_ <= kChunk;
        if (last_in_place_) {
            for (size_t il = 0; il < img_.size(); ++il) {
                img_[il].carry_kv = carry_snap_[il].carry_kv;
                img_[il].carry_score = carry_snap_[il].carry_score;
            }
        } else {
            bool window_spoiled = false;
            for (e::KvSpan sp : e::kv_checkpoint_spans(geom_, s.pos)) {
                size_t skip = 0;
                if (spoil && sp.part == spoil_part && !(sp.part == e::KvPart::Window && window_spoiled)) {
                    if (sp.part == e::KvPart::Window || sp.part == e::KvPart::CarryKv) {   // the first row
                        skip = sp.row_bytes;
                        window_spoiled = true;
                    }
                    --sp.count;                                                     // else the last row
                }
                std::memcpy(span(sp) + skip, s.host.data() + sp.offset + skip, sp.bytes());
            }
            carry_snap_ = img_;
        }
        snap_pos_ = s.pos;
        inplace_slot_ = k;
        rebuild(s.pos);
        resumed = kv_;
        resumed_set = true;
    }
    void prefill(const int32_t* t, int64_t n, std::vector<float>& logits) override { call(t, n, logits); }
    void decode(int32_t t, std::vector<float>& logits) override { call(&t, 1, logits); }

    /// check 5: the host restore leaves out one row of every span of this part (the window's oldest cell, the last
    /// compressed row or index key, the carry's first row)
    bool spoil = false;
    e::KvPart spoil_part = e::KvPart::Window;

    const e::RefKv& kv() const { return kv_; }
    int64_t pos() const { return kv_.layers.empty() ? 0 : kv_.layers[0].next_pos; }
    bool last_in_place() const { return last_in_place_; }
    /// every call's output rows since the last clear, in order
    std::vector<std::vector<float>> outputs;
    /// the state right after the last reset or restore (resumed_set: one happened since it was cleared)
    e::RefKv resumed;
    bool resumed_set = false;

private:
    struct Slot {
        int64_t pos = -1;
        std::vector<uint8_t> host;
    };

    uint8_t* span(const e::KvSpan& sp) {
        ImageLayer& g = img_[(size_t) sp.layer];
        uint8_t* base = nullptr;
        switch (sp.part) {
            case e::KvPart::Window: base = (uint8_t*) g.window.data(); break;
            case e::KvPart::Rows: base = (uint8_t*) g.rows.data(); break;
            case e::KvPart::Keys: base = (uint8_t*) g.keys.data(); break;
            case e::KvPart::CarryKv: base = (uint8_t*) g.carry_kv.data(); break;
            case e::KvPart::CarryScore: base = (uint8_t*) g.carry_score.data(); break;
        }
        return base + (size_t) sp.first * sp.row_bytes;
    }

    // RefKv at position p from the image alone (the device's view of the sequence)
    void rebuild(int64_t p) {
        const e::AttnDims& ad = m_.attn_dims();
        const size_t D = (size_t) ad.head_dim, ID = (size_t) ad.idx_dim;
        const int64_t W = geom_.window_cells, lo = std::max<int64_t>(0, p - (ad.n_swa - 1));
        kv_ = e::RefKv((size_t) m_.n_layer());
        for (int il = 0; il < m_.n_layer(); ++il) {
            const e::AttnLayer& l = m_.roles()[(size_t) il];
            const ImageLayer& g = img_[(size_t) il];
            e::RefKv::Layer& k = kv_.layers[(size_t) il];
            k.next_pos = p;
            k.window.assign((size_t) p * D, kPoison);
            for (int64_t q = lo; q < p; ++q)
                std::memcpy(k.window.data() + (size_t) q * D, g.window.data() + (size_t) (q % W) * D, D * 2);
            const int64_t blocks = l.ratio ? p / l.ratio : 0, open = l.ratio ? p % l.ratio : 0;
            if (l.kv_source) {
                k.rows.assign(g.rows.begin(), g.rows.begin() + (std::ptrdiff_t) ((size_t) blocks * D));
                k.carry_kv.assign(g.carry_kv.begin(), g.carry_kv.begin() + (std::ptrdiff_t) ((size_t) open * D));
                k.carry_score.assign(g.carry_score.begin(), g.carry_score.begin() + (std::ptrdiff_t) ((size_t) open * D));
            }
            if (l.key_owner) k.keys.assign(g.keys.begin(), g.keys.begin() + (std::ptrdiff_t) ((size_t) blocks * ID));
        }
    }

    // the image after a call of T tokens at p0: what the device's kernels write (attention.cu)
    void write_image(int64_t p0, int64_t T) {
        const e::AttnDims& ad = m_.attn_dims();
        const size_t D = (size_t) ad.head_dim, ID = (size_t) ad.idx_dim;
        const int64_t W = geom_.window_cells, p1 = p0 + T;
        for (int il = 0; il < m_.n_layer(); ++il) {
            const e::AttnLayer& l = m_.roles()[(size_t) il];
            ImageLayer& g = img_[(size_t) il];
            const e::RefKv::Layer& k = kv_.layers[(size_t) il];
            for (int64_t q = p0; q < p1; ++q)
                std::memcpy(g.window.data() + (size_t) (q % W) * D, k.window.data() + (size_t) q * D, D * 2);
            if (!l.ratio) continue;
            const int64_t b0 = p0 / l.ratio, b1 = p1 / l.ratio, open = p1 % l.ratio;
            if (l.kv_source) {
                for (int64_t b = b0; b < b1; ++b)
                    std::memcpy(g.rows.data() + (size_t) b * D, k.rows.data() + (size_t) b * D, D * 2);
                std::memcpy(g.carry_kv.data(), k.carry_kv.data(), (size_t) open * D * 4);
                std::memcpy(g.carry_score.data(), k.carry_score.data(), (size_t) open * D * 4);
            }
            if (l.key_owner)
                for (int64_t b = b0; b < b1; ++b)
                    std::memcpy(g.keys.data() + (size_t) b * ID, k.keys.data() + (size_t) b * ID, ID * 2);
        }
    }

    void call(const int32_t* tokens, int64_t T, std::vector<float>& logits) {
        const e::DenseDims& d = m_.dense_dims();
        const e::AttnDims& ad = m_.attn_dims();
        const int64_t E = d.n_embd, p0 = pos();
        std::vector<float> x((size_t) (T * E)), h(x.size()), a(x.size());
        const e::QWeightView& te = m_.token_embd();
        e::ref_embed(e::RefMatrix{te.type, te.k, te.rows, te.data, e::RefActs::Fork}, tokens, T, x.data());
        e::RefOptions o;
        o.numerics = e::RefNumerics::Engine;
        o.mode = e::ReaderMode::Reuse;
        for (int il = 0; il < m_.n_layer(); ++il) {
            e::ref_rms_norm(x.data(), m_.layer(il).attn_norm, T, (int) E, d.rms_eps, h.data());
            e::ref_attn_forward(ad, m_.roles()[(size_t) il], weights_[(size_t) il], kv_, h.data(), T, p0, o, a.data(), nullptr);
            for (size_t i = 0; i < x.size(); ++i) x[i] += a[i];
        }
        write_image(p0, T);
        outputs.push_back(x);
        std::vector<float> last((size_t) E);
        e::ref_rms_norm(x.data() + (T - 1) * E, m_.output_norm(), 1, (int) E, d.rms_eps, last.data());
        logits.assign((size_t) d.n_vocab, 0.f);
        const e::QWeightView& out = m_.output();
        e::ref_matmul(e::RefMatrix{out.type, out.k, out.rows, out.data, e::RefActs::Auto}, last.data(), 1, logits.data());
    }

    const e::Model& m_;
    e::RefKv kv_;
    std::vector<e::RefLayerWeights> weights_;
    e::KvCheckpointGeometry geom_;
    std::vector<ImageLayer> img_, carry_snap_;
    std::vector<Slot> slots_;
    int64_t snap_pos_ = -1;
    int inplace_slot_ = -1;
    bool last_in_place_ = false;
};

// what the next call reads of two states at the same position: every layer's window positions of the last n_swa - 1,
// its compressed rows, index keys and carry, bit for bit ("" when equal, else the first difference)
std::string state_diff(const e::Model& m, const e::RefKv& a, const e::RefKv& b) {
    const e::AttnDims& ad = m.attn_dims();
    const size_t D = (size_t) ad.head_dim;
    for (int il = 0; il < m.n_layer(); ++il) {
        const e::RefKv::Layer& x = a.layers[(size_t) il];
        const e::RefKv::Layer& y = b.layers[(size_t) il];
        const std::string at = "layer " + std::to_string(il) + ": ";
        if (x.next_pos != y.next_pos)
            return at + "position " + std::to_string(x.next_pos) + " against " + std::to_string(y.next_pos);
        const int64_t p = x.next_pos, lo = std::max<int64_t>(0, p - (ad.n_swa - 1));
        if (!same_bits(x.window.data() + (size_t) lo * D, y.window.data() + (size_t) lo * D, (size_t) (p - lo) * D * 2))
            return at + "the window's last positions";
        if (x.rows != y.rows) return at + "the compressed rows";
        if (x.keys != y.keys) return at + "the index keys";
        if (x.carry_kv.size() != y.carry_kv.size() || !same_bits(x.carry_kv.data(), y.carry_kv.data(), x.carry_kv.size() * 4) ||
            !same_bits(x.carry_score.data(), y.carry_score.data(), x.carry_score.size() * 4))
            return at + "the pooling carry";
    }
    return "";
}

bool same_outputs(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].size() != b[i].size() || !same_bits(a[i].data(), b[i].data(), a[i].size() * 4)) return false;
    return true;
}

// The straight read of one conversation: its prompts' calls in the same chunks, with nothing between them (no other
// conversation, no restore, no decode): after each prompt, its state, its calls' outputs and its logits.
struct Straight {
    std::vector<e::RefKv> state_before;                  // the state each turn's read starts from
    std::vector<std::vector<std::vector<float>>> outs;   // each turn's prompt calls' outputs
    std::vector<std::vector<float>> logits;              // each turn's last logits
};

Straight straight(const e::Model& m, const std::vector<std::vector<int32_t>>& prompts, const std::vector<int64_t>& from) {
    RefBackend b(m, 0);
    Straight s;
    for (size_t t = 0; t < prompts.size(); ++t) {
        if (from[t] == 0) b.reset();
        if (b.pos() != from[t]) die("the straight read is at " + std::to_string(b.pos()) + ", the turn resumes at " + std::to_string(from[t]));
        s.state_before.push_back(b.kv());
        b.outputs.clear();
        std::vector<float> logits;
        for (int64_t p = from[t]; p < (int64_t) prompts[t].size(); p += kChunk)
            b.prefill(prompts[t].data() + p, std::min<int64_t>(kChunk, (int64_t) prompts[t].size() - p), logits);
        s.outs.push_back(b.outputs);
        s.logits.push_back(logits);
    }
    return s;
}

std::vector<int32_t> random_ids(std::mt19937& g, int n, int32_t vocab) {
    std::uniform_int_distribution<int32_t> u(0, vocab - 1);
    std::vector<int32_t> v((size_t) n);
    for (int32_t& x : v) x = u(g);
    return v;
}

// One request through the session: its result, the state its read started from, its prompt calls' outputs and logits.
struct Turn {
    e::SessionResult res;
    e::RefKv start;
    bool in_place = false;
    std::vector<std::vector<float>> outs;
    std::vector<float> logits;
};

Turn run_turn(e::Session& s, RefBackend& b, const std::vector<int32_t>& ids, int max_new) {
    Turn t;
    e::SessionRequest r;
    r.ids = ids;
    r.max_new = max_new;
    const std::string why = s.check(r);
    if (!why.empty()) die("a request the session refuses: " + why);
    const e::RefKv live = b.kv();
    b.resumed_set = false;
    e::SessionEvents ev;
    ev.resume = [&](const e::ResumePlan&) { b.outputs.clear(); };
    ev.prompt_logits = [&](const std::vector<float>& l) {
        t.logits = l;
        t.outs = b.outputs;
    };
    t.res = s.run(r, ev);
    t.start = b.resumed_set ? b.resumed : live;
    t.in_place = t.res.plan.kind == e::ResumeKind::PromptEnd && b.last_in_place();
    return t;
}

// a conversation: each turn's request, where its read started, its last reply
struct User {
    std::vector<std::vector<int32_t>> prompts;
    std::vector<int64_t> from;
    std::vector<int32_t> reply;
};

// the turn just run against the straight read of the user's turns so far ("" when every bit matches)
std::string against_straight(const e::Model& m, const User& u, const Turn& t) {
    const Straight st = straight(m, u.prompts, u.from);
    const size_t k = u.prompts.size() - 1;
    if (const std::string d = state_diff(m, t.start, st.state_before[k]); !d.empty()) return "the state after the resume: " + d;
    if (!same_outputs(t.outs, st.outs[k])) return "the prompt calls' outputs";
    if (!same_bits(t.logits.data(), st.logits[k].data(), t.logits.size() * 4) || t.logits.size() != st.logits[k].size())
        return "the logits";
    return "";
}

constexpr int kMaxNew = 6;

// user u's next request: turn 1 its own random prompt (37 to 61 tokens: several chunks, an open block at ratio 2),
// later the last request, its reply and 5 new tokens
std::vector<int32_t> next_prompt(const User& u, int uid, std::mt19937& g, int32_t vocab) {
    if (u.prompts.empty()) {
        std::vector<int32_t> p = random_ids(g, 37 + 6 * uid, vocab);
        p[0] = (int32_t) (uid % vocab);   // distinct from the first token on
        return p;
    }
    std::vector<int32_t> p = u.prompts.back();
    p.insert(p.end(), u.reply.begin(), u.reply.end());
    const std::vector<int32_t> extra = random_ids(g, 5, vocab);
    p.insert(p.end(), extra.begin(), extra.end());
    return p;
}

void record(User& u, const std::vector<int32_t>& ids, const Turn& t) {
    u.prompts.push_back(ids);
    // the straight read starts where this turn's read did: a resume continues the user's last prompt (the state the
    // straight read is at), a cold read from 0
    u.from.push_back(t.res.plan.reuse);
    u.reply = t.res.tokens;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: dsv41_engine_checkpoint_test GGUF (tiny-b)\n");
        return 2;
    }
    const e::HostModel host = e::load_gguf_model(argv[1]);
    e::Model m;
    std::string err;
    if (!m.init(host, true, false, err)) die(err);
    const int32_t vocab = m.dense_dims().n_vocab;
    int ratio2 = 0, ratio1 = 0;
    for (const e::AttnLayer& l : m.roles()) {
        ratio2 += l.kv_source && l.ratio == 2;
        ratio1 += l.kv_source && l.ratio == 1;
    }
    std::printf("%s: %d layers, n_swa %d, head_dim %d, chunk %lld, %d KV sources at ratio 2 and %d at ratio 1\n", argv[1],
                m.n_layer(), m.attn_dims().n_swa, m.attn_dims().head_dim, (long long) kChunk, ratio2, ratio1);

    // 1 and 2: four users alternating over 4 slots, then a fifth
    {
        RefBackend b(m, 4);
        e::Session s(b, {});
        std::mt19937 g(20261013u);
        std::vector<User> users(5);
        for (int t = 1; t <= 3; ++t)
            for (int u = 0; u < 4; ++u) {
                const std::vector<int32_t> ids = next_prompt(users[(size_t) u], u, g, vocab);
                const int64_t last = users[(size_t) u].prompts.empty() ? 0 : (int64_t) users[(size_t) u].prompts.back().size();
                const Turn r = run_turn(s, b, ids, kMaxNew);
                record(users[(size_t) u], ids, r);
                const std::string d = against_straight(m, users[(size_t) u], r);
                const bool kind = t == 1 ? r.res.plan.kind == e::ResumeKind::Cold
                                         : r.res.plan.kind == e::ResumeKind::PromptEnd && r.res.plan.reuse == last &&
                                               r.res.plan.slot == u && !r.in_place;
                char what[256];
                std::snprintf(what, sizeof what, "user %d turn %d: %zu prompt tokens, %s resume of %lld (checkpoint %d%s), "
                              "saved to %d, %d superseded; against the straight read %s", u, t, ids.size(),
                              e::resume_kind_name(r.res.plan.kind), (long long) r.res.plan.reuse, r.res.plan.slot,
                              r.in_place ? ", in place" : "", r.res.saved, r.res.superseded,
                              d.empty() ? "bit for bit" : ("differs: " + d).c_str());
                expect(kind && d.empty() && r.res.saved == u && r.res.superseded == (t > 1 ? 1 : 0) && r.res.evicted < 0, what);
            }
        // a fifth user: every slot held, the least recently used (user 0's, its turn 3 the oldest) goes
        const std::vector<int32_t> ids4 = next_prompt(users[4], 4, g, vocab);
        const Turn r4 = run_turn(s, b, ids4, kMaxNew);
        record(users[4], ids4, r4);
        const std::string d4 = against_straight(m, users[4], r4);
        expect(r4.res.plan.kind == e::ResumeKind::Cold && r4.res.evicted == 0 &&
                   r4.res.evicted_tokens == (int64_t) users[0].prompts.back().size() && r4.res.saved == 0 && d4.empty() &&
                   s.evictions() == 1,
               "user 4 (a fifth conversation): cold, evicts checkpoint " + std::to_string(r4.res.evicted) + " (" +
                   std::to_string(r4.res.evicted_tokens) + " tokens, user 0's, the least recently used), saved to " +
                   std::to_string(r4.res.saved) + (d4.empty() ? ", bit for bit" : ", differs: " + d4));
        // user 0's next turn has lost its checkpoint: read cold, as a cold read of the same prompt
        const std::vector<int32_t> ids0 = next_prompt(users[0], 0, g, vocab);
        const Turn r0 = run_turn(s, b, ids0, kMaxNew);
        record(users[0], ids0, r0);
        const std::string d0 = against_straight(m, users[0], r0);
        expect(r0.res.plan.kind == e::ResumeKind::Cold && d0.empty() && r0.res.evicted == 1,
               "user 0 turn 4 after the eviction: " + std::string(e::resume_kind_name(r0.res.plan.kind)) + " (" +
                   std::to_string(ids0.size()) + " tokens), evicts checkpoint " + std::to_string(r0.res.evicted) +
                   " (user 1's)" + (d0.empty() ? ", bit for bit as a cold read" : ", differs: " + d0));
        // and the users still held resume: user 2 and 3
        for (int u : {2, 3}) {
            const std::vector<int32_t> ids = next_prompt(users[(size_t) u], u, g, vocab);
            const int64_t last = (int64_t) users[(size_t) u].prompts.back().size();
            const Turn r = run_turn(s, b, ids, kMaxNew);
            record(users[(size_t) u], ids, r);
            const std::string d = against_straight(m, users[(size_t) u], r);
            expect(r.res.plan.kind == e::ResumeKind::PromptEnd && r.res.plan.reuse == last && d.empty(),
                   "user " + std::to_string(u) + " turn 4: " + e::resume_kind_name(r.res.plan.kind) + " resume of " +
                       std::to_string(r.res.plan.reuse) + (d.empty() ? ", bit for bit" : ", differs: " + d));
        }
    }

    // 3: the in-place restore and the ring's limit (a chunk of positions past the checkpoint)
    for (const int fed : {(int) kChunk, (int) kChunk + 1}) {
        RefBackend b(m, 4);
        e::Session s(b, {});
        std::mt19937 g(7u + (uint32_t) fed);
        User u;
        const std::vector<int32_t> ids = next_prompt(u, 1, g, vocab);
        const Turn r1 = run_turn(s, b, ids, fed + 1);   // fed tokens past the prompt's end (the last one is not fed)
        record(u, ids, r1);
        // the next request extends the prompt but not the reply (the client re-rendered it): the checkpoint, not live
        std::vector<int32_t> ids2 = ids;
        ids2.push_back((int32_t) ((r1.res.tokens[0] + 1) % vocab));
        const Turn r2 = run_turn(s, b, ids2, kMaxNew);
        u.prompts.push_back(ids2);
        u.from.push_back(r2.res.plan.reuse);
        const std::string d = against_straight(m, u, r2);
        const bool want_in_place = fed <= kChunk;
        expect(r2.res.plan.kind == e::ResumeKind::PromptEnd && r2.res.plan.reuse == (int64_t) ids.size() &&
                   r2.in_place == want_in_place && d.empty(),
               std::to_string(fed) + " tokens fed past the checkpoint: restored " +
                   (r2.in_place ? "in place" : "from the host") + " (expected " + (want_in_place ? "in place" : "from the host") +
                   ")" + (d.empty() ? ", bit for bit" : ", differs: " + d));
    }

    // 4: one slot, a single cached conversation
    {
        RefBackend b(m, 1);
        e::Session s(b, {});
        std::mt19937 g(11u);
        std::vector<User> users(2);
        int cold = 0, turns = 0;
        for (int t = 1; t <= 2; ++t)
            for (int u = 0; u < 2; ++u) {
                const std::vector<int32_t> ids = next_prompt(users[(size_t) u], u, g, vocab);
                const Turn r = run_turn(s, b, ids, kMaxNew);
                record(users[(size_t) u], ids, r);
                cold += r.res.plan.kind == e::ResumeKind::Cold && against_straight(m, users[(size_t) u], r).empty();
                ++turns;
            }
        expect(cold == turns, "one slot, 2 users alternating: " + std::to_string(cold) + " of " + std::to_string(turns) +
                                  " turns cold (C01's baseline), bit for bit");
        const std::vector<int32_t> ids = next_prompt(users[1], 1, g, vocab);
        const int64_t live = (int64_t) (users[1].prompts.back().size() + users[1].reply.size() - 1);
        const Turn r = run_turn(s, b, ids, kMaxNew);
        expect(r.res.plan.kind == e::ResumeKind::Live && r.res.plan.reuse == live,
               "one slot, the same user again: " + std::string(e::resume_kind_name(r.res.plan.kind)) + " resume of " +
                   std::to_string(r.res.plan.reuse) + " (its prompt and fed tokens, " + std::to_string(live) + ")");
    }

    // 5: a restore without one row of a part: user 1 runs between user 0's two turns
    for (const e::KvPart part : {e::KvPart::Window, e::KvPart::Rows, e::KvPart::Keys, e::KvPart::CarryKv}) {
        RefBackend b(m, 4);
        e::Session s(b, {});
        std::mt19937 g(5u);
        std::vector<User> users(2);
        for (int u = 0; u < 2; ++u) {
            const std::vector<int32_t> ids = next_prompt(users[(size_t) u], u, g, vocab);
            record(users[(size_t) u], ids, run_turn(s, b, ids, kMaxNew));
        }
        b.spoil = true;
        b.spoil_part = part;
        const std::vector<int32_t> ids = next_prompt(users[0], 0, g, vocab);
        const Turn r = run_turn(s, b, ids, kMaxNew);
        record(users[0], ids, r);
        const std::string d = against_straight(m, users[0], r);
        // the state's difference is the gate; whether it reaches the outputs depends on what the readers select
        const Straight st = straight(m, users[0].prompts, users[0].from);
        const bool outs = !same_outputs(r.outs, st.outs.back());
        expect(r.res.plan.kind == e::ResumeKind::PromptEnd && !d.empty(),
               std::string("a restore without one row of the ") + e::kv_part_name(part) + ": " +
                   (d.empty() ? "not seen (the resumed turn is bit for bit the straight read's)"
                              : "seen: " + d + (outs ? "; the prompt calls' outputs differ too" : "; the outputs equal")));
    }

    std::printf("dsv41_engine_checkpoint_test: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
