// tests/dsv41/engine/batch_test.cpp - batched decode on
// the CPU reference path, GPU free.
//
//   dsv41_engine_batch_test GGUF                        (tiny-b: window, ratio 2 and ratio 1 layers, n_swa 16)
//   dsv41_engine_batch_test --gpu GGUF HOT_SET [...]    (a dev pod: checks 1 to 4, 12 and 13 on the engine itself)
//   dsv41_engine_batch_test --gpu-swaps GGUF HOT_SET [...]   (a dev pod: the same with the hot set swapping)
//   dsv41_engine_batch_test --kernels                   (a dev pod: batch_test_gpu.cu)
//
// The device is the CPU reference (RefDevice), built as the engine's two halves are: the live sequence and the batch
// slots, each an image of the device buffers kv_cache.hpp lays out (the live sequence's ring of n_swa - 1 + chunk
// cells, a slot's of n_swa: KvCache with max_chunk 1), written after every call as the device writes them, and read
// back only from that image: before every call that decodes a slot, the slot's reference state (RefKv) is rebuilt
// from its image alone, every window position before the last n_swa - 1 poisoned with NaN and the index sources'
// ids dropped, so a value a step reads that the slot's buffers do not hold makes it differ (or NaN). An admission is
// the engine's: the live image's state at the prompt's end copied into the slot's image by kv_state_copies (the
// copies KvCache::copy_state_from queues). A layer: x += attention(rms(x) * attn_norm) (attention_ref's engine
// numerics, Reuse readers), then the routed experts on rms(x) * ffn_norm: the router (dense_ref's engine numerics),
// a placement (Placement: the hot set's slab, the even experts, unless a check gives the device another, which may
// change at every call), the hits by their group's leader (expert_hit_leader, the grouped kernel's rule) on decode's
// hit numerics (mmvq_ref.hpp, the CUDA kernels' line-for-line reference), the misses as the engine's CPU jobs
// (decode_miss_jobs over every row of the call: an expert routed by several rows read once) on CpuExperts (3 threads
// batched, 1 alone; the GPU's numerics, or the fork's for a negative control), the slot-order combine; x += it. A
// prefill chunk takes a model of the engine's split (experts.cpp split_plan): the experts outside its basis (the hot
// set's slab in the deterministic mode; the slab as it is for a prompt read beside the slots in the mode off)
// with at most kSplitPairs pairs on CpuExperts, every other pair on the prefill GEMM's numerics (f16 weights and
// activations, qref_matmul_f16). The logits are the output matrix on rms(x of the last row) * output_norm.
//
// Requests run through the engine's own host code: Session (session.hpp) for the prompt and the first token, BatchSlots
// (batch.hpp) for the slots and the steps, in the serve loop's order (a BGEN read as a GEN of one token, the admission,
// the steps between "lines"). Each request is checked against the same request run alone through the Session on a
// fresh device (a GEN of its whole max_new: today's path, live sequence, no batch), its every generated token and the
// logits of every one of its decode calls bit for bit.
//
// Checks (PASS or FAIL lines, exit 1 on any FAIL):
//   1. four requests of different prompt lengths (21, 37, 50 and 64 tokens: one to four chunks of 16) and max_new
//      (6, 9, 12 and 15), admitted one after the other, decode in a batch of up to 4 rows: each request's tokens and
//      logits are its alone run's bit for bit; slots ending mid-batch shrink it (the other rows unchanged); the
//      steps grouped work (some hit expert computed for two rows from one leader, some CPU job of two rows or more);
//      each request is marked reproducible (before the CPU experts took the GPU's numerics, not, having decoded beside another slot);
//   2. a fifth request admitted while three slots decode: its prompt read between two steps (the slots hold), then
//      it joins the batch; every request bit for bit its alone run's;
//   3. B = 1: a request admitted alone decodes in its slot one row a step, bit for bit its alone run's and marked
//      reproducible (deterministic and alone); a sampled request (seed 7) the same draws as alone;
//   4. BSTOP: a stopped slot ends ("cancel") before the next step, its tokens so far its alone run's, the others
//      unchanged; a request whose prompt and max_new pass the slot's positions (fits refuses it) runs alone;
//   --gpu: checks 1 to 4 with the engine's Forward on CUDA as the device (the live sequence's token graph alone, the
//      batch's graphs per B batched), the placement a hot set file's with no swaps (the same placement), on
//      tiny-a and tiny-b: each request's tokens and logits its alone run's bit for bit; and a batch size's graph
//      captured again for other slots and updated in place (cudaGraphExecUpdate) at least once.
//   5. kv_state_copies' geometries: a checkpoint packed from a slot's ring restores into the live ring (and back)
//      as the same state; the checks see what they guard against: an admission that leaves out one window run, or
//      the carry, makes the request differ from its alone run.
//   6. a long prompt (2600 tokens) read in
//      chunks of 1024 (a --interleave-chunk under a prefill chunk of 2048, the task's chunk; the engine's default
//      is the prefill chunk, whose chunks are the alone read's) with one batch step after
//      each chunk but the last, while three slots decode (the serve loop's interleaved read): its tokens and
//      logits are bit for bit its alone run with the same chunks, and the three slots' are bit for bit their alone
//      runs (the invariance through the interleaving); the steps between the chunks are counted.
//   7. batch_argmax, the slots' greedy pick in eight lanes, picks what the plain
//      loop (the Session's argmax: the first maximum, a NaN never winning, a NaN at id 0 kept) picks on 2000 rows:
//      129280 logits of random values, rows with ties of the maximum (two to five ids), NaNs, -inf and +inf among
//      them, every length from 1 to 40, a row of -inf only and one starting with a NaN.
//   8. a request alone (the Session, no sampler) picks with
//      batch_argmax too: run through the Session on a backend whose every call's logits are check 7's next row, its
//      2002 tokens are the plain loop's picks on those rows (the Session's argmax before the change), one by one.
//   9. a slot admitted alone (B = 1, check 3) with a prompt
//      read beside it whose request never joins a slot (max_new 1), the slab moved to the odd experts from the read
//      on: its tokens its alone run's and marked reproducible (it used to end not reproducible);
//  10. 4 identical requests in a batch whose slab is not the alone run's (the
//      odd experts; a slab drawn anew at every call), each its alone run bit for bit and reproducible;
//  11. 4 different requests under a slab drawn anew at every call (one sampled, one joining mid-batch beside 3
//      decoding slots, a one-token request read beside them), each its alone run bit for bit and reproducible, some
//      split CPU expert held by the slab at its call; the negative controls: the fork's CPU numerics for the misses
//      (the earlier engine) differ from their alone runs, and a prompt read beside the slots in the mode off (its
//      split from the slab as it is) differs from its alone run and ends not reproducible.
//  12. a request stopped on the solo path after 1, 2 and 5
//      tokens and promoted into slot 0 as the server does (a BGEN of its prompt and its tokens), 3 requests admitted
//      beside it: every token and every call's logits from its first decode on its alone run's bit for bit (the
//      Session feeds its last token by decode, as alone), marked reproducible;
//  13. a diagnostic (info lines, no check): on the same state, the next token read by prefill(T=1)
//      against decode, the logits that differ.
//   --gpu, --gpu-swaps: checks 1 to 4, 12 and 13 on the engine; --gpu-swaps with the adaptive hot set swapping at
//      every token (the placement that moves under the batch), each request its alone run's bit for bit.
#include "../../../src/dsv41/engine/expert_act.hpp"
#include "strata/dsv41/engine/attention_ref.hpp"
#include "strata/dsv41/engine/batch.hpp"
#include "strata/dsv41/engine/dense_ref.hpp"
#include "strata/dsv41/engine/experts.hpp"
#include "strata/dsv41/engine/forward.hpp"
#include "strata/dsv41/engine/kv_checkpoint.hpp"
#include "strata/dsv41/engine/mmvq_ref.hpp"
#include "strata/dsv41/engine/model.hpp"
#include "strata/dsv41/engine/qmatmul_ref.hpp"
#include "strata/dsv41/engine/sampler.hpp"
#include "strata/dsv41/engine/session.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "dsv41_engine_batch_test: %s\n", m.c_str());
    std::exit(2);
}

int failures = 0;
void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

constexpr int64_t kChunk = 16;
constexpr int64_t kCtx = 512;
constexpr int64_t kSlotCtx = 160;
constexpr int kSlots = 4;
// check 6: the engine's interleaved chunk (dsv41-engine --interleave-chunk) under a larger prefill chunk, the context
// (live and slots) for a 2600-token prompt
constexpr int64_t kInterleave = 1024;
constexpr int64_t kLongChunk = 2048;
constexpr int64_t kLongCtx = 3072;
constexpr uint16_t kPoison = 0x7e00;   // an f16 NaN

bool same_bits(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }

// The device's buffers of one layer as KvCache lays them out (kv_cache.hpp), on the host.
struct ImageLayer {
    std::vector<uint16_t> window, rows, keys;
    std::vector<float> carry_kv, carry_score;
};

// One sequence on the device: its image of a ring of `cells`, its reference state and its position.
struct Seq {
    e::KvCheckpointGeometry geom;
    std::vector<ImageLayer> img;
    e::RefKv kv;
};

// The parts an admission's copy may leave out (check 5's negative control)
enum class Spoil { None, WindowRun, Carry };

// A device the checks drive (the CPU reference here, the engine's Forward with --gpu): what each call gave.
class Device : public e::SessionBackend, public e::BatchBackend {
public:
    /// the live sequence's every call's logits, in order (the alone runs' record)
    std::vector<std::vector<float>> calls;
    /// each slot's step logits since it was last cleared
    std::vector<std::vector<float>> slot_calls[kSlots];
    int64_t steps = 0, rows_total = 0;
    /// a request starts alone in the deterministic mode (the serve loop's begin_request: the slab back to the hot
    /// set's); a batch starts (the first request's begin_request, then the slab follows the rows' routes)
    virtual void begin_request() {}
    /// the next prompts are read beside decoding slots (the serve loop's read_beside), in the deterministic mode
    /// or not (prompt_deterministic: the prefill split's basis)
    virtual void read_beside(bool, bool) {}
};

// which experts a call of the CPU reference finds in the slab: (layer, expert, the device's call number) -> in
// it. The hot set's slab (the alone runs', and the prefill split's basis in the deterministic mode) is the even
// experts.
using Placement = std::function<bool(int, int32_t, int64_t)>;
bool hot_slab(int, int32_t x, int64_t) { return x % 2 == 0; }
// a check's batch: the odd experts, or a placement drawn anew for every call (a swap at every call and layer)
bool odd_slab(int, int32_t x, int64_t) { return x % 2 == 1; }
bool moving_slab(int il, int32_t x, int64_t call) {
    uint64_t h = (uint64_t) (call * 1000003 + il * 7919 + x) * 0x9e3779b97f4a7c15ull;
    h ^= h >> 29;
    return (h & 3) != 0;   // three experts in four on average
}
// the model of the engine's prefill split (experts.cpp split_plan): an expert outside the basis with at most this many
// pairs in a prefill chunk runs on the CPU experts, every other pair on the prefill GEMM
constexpr int64_t kSplitPairs = 2;

class RefDevice final : public Device {
public:
    RefDevice(const e::Model& m, int threads, int64_t chunk = kChunk, int64_t ctx = kCtx, int64_t slot_ctx = kSlotCtx,
              Placement place = hot_slab, e::CpuNumerics numerics = e::CpuNumerics::Gpu)
        : m_(m), cpu_(threads, numerics), chunk_(chunk), ctx_(ctx), slot_ctx_(slot_ctx), place_(std::move(place)) {
        const e::AttnDims& ad = m.attn_dims();
        for (int il = 0; il < m.n_layer(); ++il) {
            const e::LayerWeights& L = m.layer(il);
            auto mat = [](const e::QWeightView& v) { return e::RefMatrix{v.type, v.k, v.rows, v.data, e::RefActs::Auto}; };
            e::RefLayerWeights w;
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
        live_ = make_seq(ad.n_swa - 1 + chunk_, ctx_);
        for (int b = 0; b < kSlots; ++b) slots_.push_back(make_seq(ad.n_swa, slot_ctx_));
    }

    // SessionBackend: the live sequence
    int64_t ctx() const override { return ctx_; }
    int64_t chunk() const override { return chunk_; }
    int64_t n_vocab() const override { return m_.dense_dims().n_vocab; }
    void reset() override { live_.kv = e::RefKv((size_t) m_.n_layer()); }
    int checkpoints() const override { return 0; }
    void save_checkpoint(int) override { die("no checkpoints here"); }
    void restore_checkpoint(int) override { die("no checkpoints here"); }
    void prefill(const int32_t* t, int64_t n, std::vector<float>& logits) override {
        std::vector<Seq*> s{&live_};
        std::vector<std::vector<float>> l;
        call(s, t, n, l, true);
        logits = l[0];
        calls.push_back(logits);
    }
    void decode(int32_t t, std::vector<float>& logits) override {
        std::vector<Seq*> s{&live_};
        std::vector<std::vector<float>> l;
        call(s, &t, 1, l, false);
        logits = l[0];
        calls.push_back(logits);
    }
    void begin_request() override { split_hot_ = true; }
    void read_beside(bool on, bool deterministic) override { split_hot_ = !on || deterministic; }

    // BatchBackend: the slots
    int batch_slots() const override { return kSlots; }
    int64_t batch_ctx() const override { return slot_ctx_; }
    void batch_admit(int b) override {
        Seq& to = slots_.at((size_t) b);
        const int64_t p = pos(live_);
        bool window_left = false;
        for (const e::KvCopy& c : e::kv_state_copies(live_.geom, to.geom, p)) {
            if (spoil == Spoil::WindowRun && c.part == e::KvPart::Window && !window_left && c.count > 0) {
                window_left = true;   // one run left out
                continue;
            }
            if (spoil == Spoil::Carry && (c.part == e::KvPart::CarryKv || c.part == e::KvPart::CarryScore)) continue;
            std::memcpy(part(to, c.layer, c.part) + (size_t) c.dst * c.row_bytes,
                        part(live_, c.layer, c.part) + (size_t) c.src * c.row_bytes, (size_t) c.count * c.row_bytes);
        }
        rebuild(to, p);
    }
    void batch_decode(const int* rows, const int32_t* tokens, int B, std::vector<std::vector<float>>& logits) override {
        std::vector<Seq*> s;
        for (int i = 0; i < B; ++i) {
            Seq& q = slots_.at((size_t) rows[i]);
            rebuild(q, pos(q));   // the step reads only what the slot's buffers hold
            s.push_back(&q);
        }
        in_step_ = true;
        call(s, tokens, 1, logits, false);
        in_step_ = false;
        for (int i = 0; i < B; ++i) slot_calls[(size_t) rows[i]].push_back(logits[(size_t) i]);
        ++steps;
        rows_total += B;
    }

    Spoil spoil = Spoil::None;
    /// the batch steps' grouped work: hit pairs computed by another row's leader, CPU jobs of two rows or more
    int64_t grouped_hits = 0, shared_jobs = 0;
    /// decode pairs on the slab and on the CPU, prefill pairs on the CPU (the split) and on the prefill GEMM, and
    /// the prefill split's CPU experts that the slab held at that call (the placement-independent split's case)
    int64_t hit_pairs = 0, miss_pairs = 0, split_cpu_pairs = 0, gemm_pairs = 0, split_in_slab = 0;

    /// check 5: the state of `from` at its position packed by its geometry's spans, unpacked by `to`'s into a fresh
    /// image of `to`'s ring, rebuilt there: "" when every value the next call reads is equal
    std::string checkpoint_round_trip(int slot) {
        Seq& from = slots_.at((size_t) slot);
        const int64_t p = pos(from);
        std::vector<uint8_t> host;
        for (const e::KvSpan& sp : e::kv_checkpoint_spans(from.geom, p)) {
            const uint8_t* src = part(from, sp.layer, sp.part) + (size_t) sp.first * sp.row_bytes;
            host.insert(host.end(), src, src + sp.bytes());
        }
        Seq to = make_seq(live_.geom.window_cells, ctx_);
        for (const e::KvSpan& sp : e::kv_checkpoint_spans(to.geom, p))
            std::memcpy(part(to, sp.layer, sp.part) + (size_t) sp.first * sp.row_bytes, host.data() + sp.offset, sp.bytes());
        rebuild(to, p);
        return state_diff(from.kv, to.kv);
    }

private:
    Seq make_seq(int64_t cells, int64_t cap_ctx) const {
        const e::AttnDims& ad = m_.attn_dims();
        const size_t D = (size_t) ad.head_dim;
        Seq s;
        s.geom = e::KvCheckpointGeometry{ad, m_.roles(), cells};
        s.img.resize((size_t) m_.n_layer());
        for (int il = 0; il < m_.n_layer(); ++il) {
            const e::AttnLayer& l = m_.roles()[(size_t) il];
            ImageLayer& g = s.img[(size_t) il];
            g.window.assign((size_t) cells * D, kPoison);
            const int64_t cap = l.ratio ? (cap_ctx + l.ratio - 1) / l.ratio : 0;
            if (l.kv_source) {
                g.rows.assign((size_t) cap * D, kPoison);
                g.carry_kv.assign((size_t) l.ratio * D, NAN);
                g.carry_score.assign((size_t) l.ratio * D, NAN);
            }
            if (l.key_owner) g.keys.assign((size_t) cap * ad.idx_dim, kPoison);
        }
        s.kv = e::RefKv((size_t) m_.n_layer());
        return s;
    }

    static int64_t pos(const Seq& s) { return s.kv.layers.empty() ? 0 : s.kv.layers[0].next_pos; }

    uint8_t* part(Seq& s, int il, e::KvPart p) {
        ImageLayer& g = s.img[(size_t) il];
        switch (p) {
            case e::KvPart::Window: return (uint8_t*) g.window.data();
            case e::KvPart::Rows: return (uint8_t*) g.rows.data();
            case e::KvPart::Keys: return (uint8_t*) g.keys.data();
            case e::KvPart::CarryKv: return (uint8_t*) g.carry_kv.data();
            case e::KvPart::CarryScore: return (uint8_t*) g.carry_score.data();
        }
        return nullptr;
    }

    // the reference state at position p from the image alone (the device's view of the sequence)
    void rebuild(Seq& s, int64_t p) {
        const e::AttnDims& ad = m_.attn_dims();
        const size_t D = (size_t) ad.head_dim, ID = (size_t) ad.idx_dim;
        const int64_t W = s.geom.window_cells, lo = std::max<int64_t>(0, p - (ad.n_swa - 1));
        s.kv = e::RefKv((size_t) m_.n_layer());
        for (int il = 0; il < m_.n_layer(); ++il) {
            const e::AttnLayer& l = m_.roles()[(size_t) il];
            const ImageLayer& g = s.img[(size_t) il];
            e::RefKv::Layer& k = s.kv.layers[(size_t) il];
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
    void write_image(Seq& s, int64_t p0, int64_t T) {
        const e::AttnDims& ad = m_.attn_dims();
        const size_t D = (size_t) ad.head_dim, ID = (size_t) ad.idx_dim;
        const int64_t W = s.geom.window_cells, p1 = p0 + T;
        for (int il = 0; il < m_.n_layer(); ++il) {
            const e::AttnLayer& l = m_.roles()[(size_t) il];
            ImageLayer& g = s.img[(size_t) il];
            const e::RefKv::Layer& k = s.kv.layers[(size_t) il];
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

    // what the next call reads of two states at the same position ("" when equal)
    std::string state_diff(const e::RefKv& a, const e::RefKv& b) const {
        const e::AttnDims& ad = m_.attn_dims();
        const size_t D = (size_t) ad.head_dim;
        for (int il = 0; il < m_.n_layer(); ++il) {
            const e::RefKv::Layer& x = a.layers[(size_t) il];
            const e::RefKv::Layer& y = b.layers[(size_t) il];
            const std::string at = "layer " + std::to_string(il) + ": ";
            if (x.next_pos != y.next_pos) return at + "the positions";
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

    // the prefill GEMM's numerics for the tokens xs of one expert (a model of experts_gemm.hpp's f16 tiles: f16 weights
    // and activations, the clamp of expert_act.hpp, h in f16), into ys: not the decode's numerics, so a pair that
    // changes sides between it and the CPU experts changes its bits
    void gemm(const e::ExpertLayerHost& L, int32_t ex, const std::vector<float>& xs, int64_t n, std::vector<float>& ys) const {
        const e::DenseDims& d = m_.dense_dims();
        const int64_t E = d.n_embd, F = d.n_ff;
        auto view = [&](int p) {
            const void* data = L.data[p] + (size_t) ex * L.bytes[p];
            return p == 2 ? e::QWeightView{L.type[2], F, E, data} : e::QWeightView{L.type[p], E, F, data};
        };
        std::vector<float> g((size_t) (n * F)), u(g.size()), h(g.size());
        e::qref_matmul_f16(view(1), xs.data(), E, n, 1, g.data(), F, nullptr);
        e::qref_matmul_f16(view(0), xs.data(), E, n, 1, u.data(), F, nullptr);
        for (size_t i = 0; i < h.size(); ++i) h[i] = e::act::swiglu_clamp(g[i], u[i], L.limit);
        ys.assign((size_t) (n * E), 0.f);
        e::qref_matmul_f16(view(2), h.data(), F, n, 1, ys.data(), E, nullptr);
    }

    // the routed experts of a call's rows xn [T][E]: the router; decode, the hits by their leaders (decode's hit
    // numerics, mmvq_ref.hpp) and the misses as the engine's CPU jobs over every row at once; a prefill chunk, the
    // split's model: the experts outside its basis (the hot set's slab in the deterministic mode, the slab as it is
    // otherwise) with at most kSplitPairs pairs on the CPU experts, every other pair on the GEMM; the slot-order combine
    // into out [T][E]. The slab is place_'s at this call.
    void experts(int il, const float* xn, int64_t T, float* out, bool prefill) {
        const e::DenseDims& d = m_.dense_dims();
        const e::LayerWeights& L = m_.layer(il);
        const e::ExpertLayerHost& X = m_.experts()[(size_t) il];
        const int K = d.n_expert_used;
        const int64_t E = d.n_embd, P = T * K;
        e::RefRouterTrace tr;
        e::ref_router(e::RefNumerics::Engine, d,
                      e::RefRouterWeights{e::RefMatrix{e::kRefTypeBF16, E, d.n_expert, L.router.gate_inp, e::RefActs::Auto},
                                          L.router.bias},
                      xn, T, tr);
        std::vector<int32_t> slot_of((size_t) d.n_expert, -1);
        for (int32_t x = 0, k = 0; x < d.n_expert; ++x)
            if (place_(il, x, calls_)) slot_of[(size_t) x] = k++;
        std::vector<float> y((size_t) (P * E));
        // the CPU's side: decode's misses, or a prefill chunk's split (a slot -1 for each expert the CPU computes)
        std::vector<int32_t> cpu_side(slot_of);
        if (prefill) {
            std::vector<int64_t> pairs((size_t) d.n_expert, 0);
            for (int32_t x : tr.ids) ++pairs[(size_t) x];
            for (int32_t x = 0; x < d.n_expert; ++x) {
                const bool outside = split_hot_ ? !hot_slab(il, x, calls_) : slot_of[(size_t) x] < 0;
                const bool on_cpu = outside && pairs[(size_t) x] > 0 && pairs[(size_t) x] <= kSplitPairs;
                cpu_side[(size_t) x] = on_cpu ? -1 : 0;
                split_in_slab += on_cpu && slot_of[(size_t) x] >= 0;
            }
            // the GEMM's pairs, by expert
            for (int32_t x = 0; x < d.n_expert; ++x) {
                if (cpu_side[(size_t) x] < 0 || !pairs[(size_t) x]) continue;
                std::vector<int64_t> ps;
                std::vector<float> xs, ys;
                for (int64_t p = 0; p < P; ++p)
                    if (tr.ids[(size_t) p] == x) {
                        ps.push_back(p);
                        xs.insert(xs.end(), xn + (p / K) * E, xn + (p / K + 1) * E);
                    }
                gemm(X, x, xs, (int64_t) ps.size(), ys);
                for (size_t i = 0; i < ps.size(); ++i) std::memcpy(y.data() + ps[i] * E, ys.data() + i * E, (size_t) E * 4);
                gemm_pairs += (int64_t) ps.size();
            }
        } else {
            // the hits: each leader's pairs, every hit pair covered once
            std::vector<int> covered((size_t) P, 0);
            for (int64_t p = 0; p < P; ++p) {
                if (e::expert_hit_leader(tr.ids.data(), P, p, slot_of.data()) != p) continue;
                for (int64_t q = p; q < P; ++q) {
                    if (tr.ids[(size_t) q] != tr.ids[(size_t) p]) continue;
                    e::mmvq_ref_expert(X, d, tr.ids[(size_t) q], xn + (q / K) * E, y.data() + q * E);
                    ++covered[(size_t) q];
                    grouped_hits += in_step_ && q != p;
                    ++hit_pairs;
                }
            }
            for (int64_t p = 0; p < P; ++p)
                if ((slot_of[(size_t) tr.ids[(size_t) p]] >= 0) != (covered[(size_t) p] == 1)) die("a hit pair not covered once");
        }
        // the CPU's pairs: decode_serve's jobs (the split's on a prefill chunk)
        std::vector<int32_t> job_of((size_t) d.n_expert, -1), job_expert, miss((size_t) P);
        std::vector<e::CpuExpertJob> jobs((size_t) d.n_expert);
        int64_t hits = 0;
        const size_t n = e::decode_miss_jobs(tr.ids.data(), P, K, cpu_side.data(), job_of.data(), job_expert, jobs.data(),
                                             miss.data(), hits);
        (prefill ? split_cpu_pairs : miss_pairs) += P - hits;
        int64_t rows = 0;
        for (size_t k = 0; k < n; ++k) {
            for (int q = 0; q < 3; ++q) jobs[k].rows.part[q] = X.data[q] + (size_t) job_expert[k] * X.bytes[q];
            rows += (int64_t) jobs[k].tokens.size();
            shared_jobs += in_step_ && jobs[k].tokens.size() > 1;
        }
        if (n) {
            std::vector<float> yc((size_t) (rows * E));
            cpu_.run(X, d, xn, T, jobs.data(), n, yc.data());
            for (int64_t p = 0; p < P; ++p)
                if (miss[(size_t) p] >= 0)
                    std::memcpy(y.data() + p * E, yc.data() + (size_t) miss[(size_t) p] * E, (size_t) E * 4);
        }
        e::ref_expert_combine(y.data(), tr.scaled.data(), T, K, (int) E, nullptr, out);
    }

    // one call: T tokens of one sequence (s of size 1, a prefill chunk or a decode token) or one token of each of
    // B sequences (a batch step: s.size() rows, T 1); logits of each sequence's last row
    void call(std::vector<Seq*>& s, const int32_t* tokens, int64_t T, std::vector<std::vector<float>>& logits, bool prefill) {
        const e::DenseDims& d = m_.dense_dims();
        const e::AttnDims& ad = m_.attn_dims();
        const int64_t E = d.n_embd, R = (int64_t) s.size() * T;
        std::vector<int64_t> p0;
        for (Seq* q : s) p0.push_back(pos(*q));
        std::vector<float> x((size_t) (R * E)), h(x.size()), a(x.size()), moe(x.size());
        const e::QWeightView& te = m_.token_embd();
        e::ref_embed(e::RefMatrix{te.type, te.k, te.rows, te.data, e::RefActs::Fork}, tokens, R, x.data());
        e::RefOptions o;
        o.numerics = e::RefNumerics::Engine;
        o.mode = e::ReaderMode::Reuse;
        for (int il = 0; il < m_.n_layer(); ++il) {
            e::ref_rms_norm(x.data(), m_.layer(il).attn_norm, R, (int) E, d.rms_eps, h.data());
            // the attention per sequence on its own state
            for (size_t i = 0; i < s.size(); ++i)
                e::ref_attn_forward(ad, m_.roles()[(size_t) il], weights_[(size_t) il], s[i]->kv, h.data() + (int64_t) i * T * E,
                                    T, p0[i], o, a.data() + (int64_t) i * T * E, nullptr);
            for (size_t k = 0; k < x.size(); ++k) x[k] += a[k];
            // the routed experts over every row at once
            e::ref_rms_norm(x.data(), m_.layer(il).ffn_norm, R, (int) E, d.rms_eps, h.data());
            experts(il, h.data(), R, moe.data(), prefill);
            for (size_t k = 0; k < x.size(); ++k) x[k] += moe[k];
        }
        ++calls_;
        logits.assign(s.size(), std::vector<float>((size_t) d.n_vocab));
        const e::QWeightView& out = m_.output();
        std::vector<float> last((size_t) E);
        for (size_t i = 0; i < s.size(); ++i) {
            write_image(*s[i], p0[i], T);
            e::ref_rms_norm(x.data() + ((int64_t) i * T + T - 1) * E, m_.output_norm(), 1, (int) E, d.rms_eps, last.data());
            e::ref_matmul(e::RefMatrix{out.type, out.k, out.rows, out.data, e::RefActs::Auto}, last.data(), 1, logits[i].data());
        }
    }

    const e::Model& m_;
    e::CpuExperts cpu_;
    int64_t chunk_, ctx_, slot_ctx_;
    std::vector<e::RefLayerWeights> weights_;
    Seq live_;
    std::vector<Seq> slots_;
    Placement place_;
    int64_t calls_ = 0;
    bool in_step_ = false;
    bool split_hot_ = true;   // the prompts read now are in the deterministic mode, or alone (Device::read_beside)
};

// One request: its prompt, max_new, sampler seed (0: greedy), its mode (the deterministic mode, or off)
struct Req {
    std::vector<int32_t> ids;
    int64_t max_new = 0;
    uint64_t seed = 0;
    bool det = true;
};

// What a request gave: its tokens and the logits of each of its decode calls (its first token's are the prompt's)
struct Got {
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> logits;
    std::string finish;
    bool reproducible = false;
};

std::unique_ptr<e::Sampler> sampler_of(const Req& r) {
    if (!r.seed) return nullptr;
    e::SamplerParams sp;
    sp.temperature = 0.8;
    sp.top_k = 8;
    sp.has_seed = true;
    sp.seed = r.seed;
    return std::make_unique<e::Sampler>(sp);
}

// The devices of the checks: one for each batch (the CPU reference's batch on 3 threads) and one for each alone run
// (1 thread); with --gpu the same engine for both, its records cleared
std::function<std::shared_ptr<Device>(bool alone)> make_device;

// the request alone on a fresh device (make_device's, or `owner`): a GEN of its whole max_new through the Session
// (today's path), its prompt in chunks of `chunk` (0: the device's)
Got alone(const Req& r, int64_t chunk = 0, std::shared_ptr<Device> owner = nullptr) {
    if (!owner) owner = make_device(true);
    Device& dev = *owner;
    dev.begin_request();
    e::Session s(dev, {});
    e::SessionRequest q;
    q.ids = r.ids;
    q.max_new = r.max_new;
    q.snapshot = false;
    q.chunk = chunk;
    std::unique_ptr<e::Sampler> sm = sampler_of(r);
    q.sampler = sm.get();
    if (const std::string why = s.check(q); !why.empty()) die("alone: " + why);
    const e::SessionResult res = s.run(q, e::SessionEvents{});
    Got g;
    g.tokens = res.tokens;
    g.finish = res.finish;
    // the live calls: the prompt's chunks, then one per decode token
    const int64_t c = chunk > 0 ? chunk : dev.chunk();
    const size_t chunks = (size_t) (((int64_t) r.ids.size() + c - 1) / c);
    g.logits.assign(dev.calls.begin() + (std::ptrdiff_t) chunks, dev.calls.end());
    return g;
}

// The serve loop's batch on one device: requests admitted (BGEN), steps between them, every request's result.
struct Batch {
    explicit Batch(std::shared_ptr<Device> d) : owner(std::move(d)), dev(*owner), session(dev, {}), slots(dev, {}) {}
    Batch() : Batch(make_device(false)) {}
    std::shared_ptr<Device> owner;
    Device& dev;
    e::Session session;
    e::BatchSlots slots;
    std::vector<Got> got;
    std::vector<int> slot_of_req;
    int unflushed = 0;   // the BDONE lines a step left without its flush (the serve loop's stdout, a pipe)
    // BGEN: the prompt and the first token as a GEN of one token, then the admission into slot b; with interleave,
    // the prompt in chunks of it and one step after each chunk but the last (the serve loop's interleaved read)
    void admit(int b, const Req& r, int64_t interleave = 0) {
        if (!slots.fits((int64_t) r.ids.size(), r.max_new).empty()) die("a request that does not fit a slot");
        e::SessionRequest q;
        q.ids = r.ids;
        q.max_new = 1;
        q.snapshot = false;
        q.chunk = interleave;
        std::unique_ptr<e::Sampler> sm = sampler_of(r);
        q.sampler = sm.get();
        const bool others = slots.active() > 0;
        e::SessionEvents ev;
        if (interleave > 0) ev.between = [&] { step(); };
        if (const std::string why = session.check(q); !why.empty()) die("admit: " + why);
        // the serve loop's: alone, the request's start; beside the slots, the read in the request's mode
        if (others) dev.read_beside(true, r.det);
        else dev.begin_request();
        const e::SessionResult res = session.run(q, ev);
        dev.read_beside(false, true);
        if (res.finish != "length" || res.generated != 1) die("an admission that ended at its first token");
        if (others) slots.stalled(1.0);
        Got g;
        g.tokens = res.tokens;
        got.push_back(g);
        slot_of_req.push_back(b);
        dev.slot_calls[b].clear();
        e::BatchSlots::Admission a;
        a.request = (int64_t) got.size() - 1;
        a.position = (int64_t) r.ids.size();
        a.first = res.tokens.front();
        a.max_new = r.max_new;
        a.sampler = std::move(sm);
        a.deterministic = r.det;
        slots.admit(b, std::move(a));
    }
    // a request read beside the slots that never joins one (max_new 1, as a BGEN
    // whose request ends at its first token): its prompt and first token through the Session, no admission
    void read_only(const Req& r) {
        e::SessionRequest q;
        q.ids = r.ids;
        q.max_new = 1;
        q.snapshot = false;
        if (const std::string why = session.check(q); !why.empty()) die("read_only: " + why);
        const bool others = slots.active() > 0;
        if (others) dev.read_beside(true, r.det);
        else dev.begin_request();
        session.run(q, e::SessionEvents{});
        dev.read_beside(false, true);
        if (others) slots.stalled(1.0);
    }
    int step() {
        e::BatchSlots::Events ev;
        ev.token = [&](int b, int32_t t) {
            for (size_t k = got.size(); k-- > 0;)
                if (slot_of_req[k] == b && got[k].finish.empty()) {
                    got[k].tokens.push_back(t);
                    break;
                }
        };
        ev.done = [&](const e::BatchDone& d) {
            Got& g = got[(size_t) d.request];
            g.finish = d.finish;
            g.reproducible = d.reproducible;
            g.logits = dev.slot_calls[d.slot];
            ++unflushed;
        };
        ev.flush = [&] { unflushed = 0; };
        return slots.step(ev);
    }
    void drain() {
        while (slots.active()) step();
    }
};

bool same_got(const Got& a, const Got& b, size_t n_tokens) {
    if (a.tokens.size() < n_tokens || b.tokens.size() < n_tokens) return false;
    if (!std::equal(a.tokens.begin(), a.tokens.begin() + (std::ptrdiff_t) n_tokens, b.tokens.begin())) return false;
    const size_t n_logits = n_tokens ? n_tokens - 1 : 0;
    if (a.logits.size() < n_logits || b.logits.size() < n_logits) return false;
    for (size_t i = 0; i < n_logits; ++i)
        if (a.logits[i].size() != b.logits[i].size() || !same_bits(a.logits[i].data(), b.logits[i].data(), a.logits[i].size() * 4))
            return false;
    return true;
}

std::string against_alone(const Req& r, const Got& g) {
    const Got want = alone(r);
    if (g.finish != want.finish) return "finish " + g.finish + " against " + want.finish;
    if (!same_got(g, want, want.tokens.size())) return "the tokens or a step's logits differ";
    return "";
}

std::vector<int32_t> random_ids(std::mt19937& g, int n, int32_t vocab, int32_t first) {
    std::uniform_int_distribution<int32_t> u(0, vocab - 1);
    std::vector<int32_t> v((size_t) n);
    for (int32_t& x : v) x = u(g);
    v[0] = first;
    return v;
}


// The engine on CUDA as the device (--gpu): its live sequence and its batch slots (forward.hpp), every call's logits
// recorded as the CPU reference records them.
class GpuDevice final : public Device {
public:
    GpuDevice(e::Forward& fw, int64_t n_vocab) : fw_(fw), n_vocab_(n_vocab) {}
    int64_t ctx() const override { return kCtx; }
    int64_t chunk() const override { return kChunk; }
    int64_t n_vocab() const override { return n_vocab_; }
    void reset() override { fw_.reset(); }
    int checkpoints() const override { return 0; }
    void save_checkpoint(int) override { die("no checkpoints here"); }
    void restore_checkpoint(int) override { die("no checkpoints here"); }
    void prefill(const int32_t* t, int64_t n, std::vector<float>& logits) override {
        fw_.prefill(t, n, logits);
        calls.push_back(logits);
    }
    void decode(int32_t t, std::vector<float>& logits) override {
        fw_.decode(t, logits);
        calls.push_back(logits);
    }
    int batch_slots() const override { return fw_.batch_slots(); }
    int64_t batch_ctx() const override { return fw_.batch_ctx(); }
    void batch_admit(int b) override { fw_.batch_admit(b); }
    void batch_decode(const int* rows, const int32_t* tokens, int B, std::vector<std::vector<float>>& logits) override {
        fw_.decode_batch(rows, tokens, B, logits);
        for (int i = 0; i < B; ++i) slot_calls[(size_t) rows[i]].push_back(logits[(size_t) i]);
        ++steps;
        rows_total += B;
    }
    // the serve loop's calls around a request (with swaps on, the placement moves under the batch)
    void begin_request() override { fw_.begin_request(e::HotPolicy::Adaptive, false, false, true); }
    void read_beside(bool on, bool deterministic) override {
        fw_.read_beside(on);
        if (on) fw_.prompt_deterministic(deterministic);
    }

private:
    e::Forward& fw_;
    int64_t n_vocab_;
};

// checks 1 to 4 (and, on the CPU reference, check 5 and the grouping counts) on make_device's devices
void run_checks(const e::Model& m, bool cpu);

// the engine on the device, for check (g4) of the kernel tests (batch_test_gpu.cu): checks 1 to 4 against its own alone
// runs, the placement a hot set's (no swaps); --gpu-swaps: the same with the adaptive hot set swapping at every
// token (2 spare slots a layer, margin -1: a swap even at equal counts), so the batch's placement follows its rows'
// routes and the alone runs' their own, and each request is still its alone run bit for bit
int run_gpu(const std::vector<std::string>& args, bool swaps) {
    if (args.empty() || args.size() % 2) die("--gpu takes pairs GGUF HOT_SET (a hot set file: hot_sets.py's split.json)");
    for (size_t a = 0; a < args.size(); a += 2) {
        const e::HostModel host = e::load_gguf_model(args[a]);
        e::Model m;
        std::string err;
        if (!m.init(host, false, false, err)) die(err);
        e::ExpertConfig ec;
        ec.hot_set = args[a + 1];
        ec.vram_gib = 4;
        ec.ram_gib = 4;
        ec.threads = 4;
        ec.hot.spare = swaps ? 2 : 0;   // 0: the slab never moves, the same placement
        if (swaps) {
            ec.hot.margin_mul = 0;
            ec.hot.margin_add = -1;
        }
        e::ForwardConfig fc;
        fc.ctx = kCtx;
        fc.chunk = kChunk;
        fc.checkpoints = 0;
        fc.batch = kSlots;
        fc.batch_ctx = kSlotCtx;
        e::Forward fw;
        if (!fw.init(m, ec, fc, err)) die(err);
        fw.print_memory(stdout);
        auto dev = std::make_shared<GpuDevice>(fw, m.dense_dims().n_vocab);
        make_device = [dev](bool) {
            dev->calls.clear();
            for (auto& c : dev->slot_calls) c.clear();
            dev->steps = dev->rows_total = 0;
            return std::shared_ptr<Device>(dev);
        };
        std::printf("%s on the GPU, hot set %s (%s):\n", args[a].c_str(), args[a + 1].c_str(),
                    swaps ? "swaps at every token" : "no swaps");
        run_checks(m, false);
        std::printf("batch graphs: %lld captures, %lld of them updates of an instantiated graph\n",
                    (long long) fw.batch_captures(), (long long) fw.batch_updates());
        expect(fw.batch_updates() > 0, "a batch size's graph captured again for other slots and updated in place");
    }
    std::printf("dsv41_engine_batch_test %s: %s\n", swaps ? "--gpu-swaps" : "--gpu", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

}  // namespace

int run_kernels();   // batch_test_gpu.cu

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (!args.empty() && (args[0] == "--gpu" || args[0] == "--gpu-swaps"))
        return run_gpu(std::vector<std::string>(args.begin() + 1, args.end()), args[0] == "--gpu-swaps");
    if (args.size() == 1 && args[0] == "--kernels") return run_kernels();
    if (args.size() != 1) {
        std::fprintf(stderr, "usage: dsv41_engine_batch_test GGUF (tiny-b; the CPU reference)\n"
                             "       dsv41_engine_batch_test --gpu GGUF HOT_SET [GGUF HOT_SET ...] (the engine, a dev pod)\n"
                             "       dsv41_engine_batch_test --gpu-swaps GGUF HOT_SET [...] (the same, the hot set swapping)\n"
                             "       dsv41_engine_batch_test --kernels (the batched kernels on synthetic tensors, a dev pod)\n");
        return 2;
    }
    const e::HostModel host = e::load_gguf_model(args[0]);
    e::Model m;
    std::string err;
    if (!m.init(host, true, false, err)) die(err);
    make_device = [&m](bool alone) { return std::shared_ptr<Device>(std::make_shared<RefDevice>(m, alone ? 1 : 3)); };
    run_checks(m, true);
    std::printf("dsv41_engine_batch_test: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

namespace {

// check 8: the Session on a backend whose n-th call (the prompt's, then each decode) returns rows[n]
class RowsBackend final : public e::SessionBackend {
public:
    explicit RowsBackend(const std::vector<std::vector<float>>& rows) : rows_(rows) {}
    int64_t ctx() const override { return (int64_t) rows_.size() + 1; }
    int64_t chunk() const override { return 1; }
    int64_t n_vocab() const override { return 129280; }
    void reset() override { next_ = 0; }
    int checkpoints() const override { return 0; }
    void save_checkpoint(int) override {}
    void restore_checkpoint(int) override {}
    void prefill(const int32_t*, int64_t, std::vector<float>& logits) override { logits = rows_.at(next_++); }
    void decode(int32_t, std::vector<float>& logits) override { logits = rows_.at(next_++); }

private:
    const std::vector<std::vector<float>>& rows_;
    size_t next_ = 0;
};

void run_checks(const e::Model& m, bool cpu) {
    const e::DenseDims& d = m.dense_dims();
    const int32_t vocab = d.n_vocab;
    std::printf("%d layers, n_swa %d, %d experts (%d used), chunk %lld, %d slots of %lld positions (rings of %d cells "
                "against the live sequence's %lld)\n", m.n_layer(), m.attn_dims().n_swa, d.n_expert,
                d.n_expert_used, (long long) kChunk, kSlots, (long long) kSlotCtx, m.attn_dims().n_swa,
                (long long) (m.attn_dims().n_swa - 1 + kChunk));
    std::mt19937 g(20261014u);

    // 1: four requests, admitted one after the other, decoding together
    {
        Batch bt;
        const int lens[kSlots] = {21, 37, 50, 64};
        const int64_t max_new[kSlots] = {6, 9, 12, 15};
        std::vector<Req> reqs;
        for (int k = 0; k < kSlots; ++k) {
            reqs.push_back({random_ids(g, lens[k], vocab, k), max_new[k], 0});
            bt.admit(k, reqs.back());
            bt.step();   // a step between two admissions, as the serve loop runs one when no line waits
        }
        std::vector<int> rows;
        while (bt.slots.active()) rows.push_back(bt.step());
        // the device's counts before the alone runs: on the GPU they reuse this device and clear its counts
        const int64_t steps = bt.dev.steps, rows_total = bt.dev.rows_total;
        const int64_t drain_rows = std::accumulate(rows.begin(), rows.end(), (int64_t) 0);
        bool shrank = false;
        for (size_t i = 1; i < rows.size(); ++i) shrank = shrank || rows[i] < rows[i - 1];
        for (int k = 0; k < kSlots; ++k) {
            const std::string diff = against_alone(reqs[(size_t) k], bt.got[(size_t) k]);
            expect(diff.empty() && bt.got[(size_t) k].reproducible,
                   "request " + std::to_string(k) + " (" + std::to_string(lens[k]) + " prompt tokens, max_new " +
                       std::to_string(max_new[k]) + ", slot " + std::to_string(k) + "): " +
                       (diff.empty() ? "every token and step's logits its alone run's bit for bit" : diff) +
                       (bt.got[(size_t) k].reproducible ? "; marked reproducible" : "; marked not reproducible"));
        }
        // kSlots steps between the admissions (1 to 4 rows, one per slot admitted), then the drain's
        const int64_t admit_rows = (int64_t) kSlots * (kSlots + 1) / 2;
        expect(shrank && rows.front() == kSlots && steps == kSlots + (int64_t) rows.size() &&
                   rows_total == admit_rows + drain_rows,
               "the batch: " + std::to_string(steps) + " steps, " + std::to_string(rows_total) + " rows (" +
                   std::to_string(kSlots) + " steps of " + std::to_string(admit_rows) + " rows between the admissions, " +
                   std::to_string(rows.size()) + " of " + std::to_string(drain_rows) +
                   " after), 4 rows at first, shrinking as slots end");
        if (cpu) {
            const RefDevice& rd = static_cast<const RefDevice&>(bt.dev);
            expect(rd.grouped_hits > 0 && rd.shared_jobs > 0,
                   "grouped by expert: " + std::to_string(rd.grouped_hits) + " hit pairs computed by another row's leader, " +
                       std::to_string(rd.shared_jobs) + " CPU jobs of two rows or more");
        }
    }

    // 2: a fifth request read between two steps while three slots decode
    {
        Batch bt;
        std::vector<Req> reqs;
        for (int k = 0; k < 3; ++k) {
            reqs.push_back({random_ids(g, 18 + 9 * k, vocab, 10 + k), 10 + 2 * k, 0});
            bt.admit(k, reqs.back());
        }
        for (int s = 0; s < 3; ++s) bt.step();
        reqs.push_back({random_ids(g, 41, vocab, 20), 8, 0});
        bt.admit(3, reqs.back());
        const int rows = bt.step();
        bt.drain();
        bool ok = rows == 4;
        std::string what;
        for (size_t k = 0; k < reqs.size(); ++k) {
            const std::string diff = against_alone(reqs[k], bt.got[k]);
            ok = ok && diff.empty();
            what += (k ? "; " : "") + std::string("request ") + std::to_string(k) + (diff.empty() ? " bit for bit" : ": " + diff);
        }
        expect(ok, "a prompt read between two steps (41 tokens beside 3 decoding slots), then 4 rows a step: " + what);
        expect(bt.slots.totals().stalls >= 1, "the stall of the read counted against the slots' steps (" +
                                                  std::to_string(bt.slots.totals().stalls) + " stalls)");
    }

    // 3: B = 1, greedy and sampled
    for (uint64_t seed : {0ull, 7ull}) {
        Batch bt;
        const Req r{random_ids(g, 33, vocab, 30), 11, seed};
        bt.admit(2, r);
        bt.drain();
        // the device's counts before the alone run clears them (check 1); the first token is the admission's
        const int64_t steps = bt.dev.steps, rows_total = bt.dev.rows_total;
        const std::string diff = against_alone(r, bt.got[0]);
        expect(diff.empty() && bt.got[0].reproducible && rows_total == steps &&
                   steps == (int64_t) bt.got[0].tokens.size() - 1,
               std::string("one slot alone (B = 1), ") + (seed ? "sampled, seed 7" : "greedy") + ": " +
                   std::to_string(steps) + " steps, " + std::to_string(rows_total) + " rows (one a step), " +
                   (diff.empty() ? "bit for bit its alone run" : diff) +
                   (bt.got[0].reproducible ? ", marked reproducible" : ", marked not reproducible"));
    }

    // 4: BSTOP, and a request past a slot's positions
    {
        Batch bt;
        std::vector<Req> reqs;
        for (int k = 0; k < 2; ++k) {
            reqs.push_back({random_ids(g, 25 + 11 * k, vocab, 40 + k), 14, 0});
            bt.admit(k, reqs.back());
        }
        for (int s = 0; s < 4; ++s) bt.step();
        bt.slots.stop(0);
        bt.step();
        const Got stopped = bt.got[0];
        bt.drain();
        const Got want0 = alone(reqs[0]);
        const bool ok0 = stopped.finish == "cancel" && same_got(stopped, want0, stopped.tokens.size()) &&
                         stopped.tokens.size() == 5;
        const std::string diff1 = against_alone(reqs[1], bt.got[1]);
        expect(ok0 && diff1.empty(), "BSTOP: slot 0 ended (" + stopped.finish + ") before the next step with " +
                                         std::to_string(stopped.tokens.size()) + " tokens, its alone run's; slot 1 " +
                                         (diff1.empty() ? "bit for bit its alone run" : diff1));
        // measured on a pod: every slot left stopped at once (all their clients gone) ends in a step with no row to
        // decode, which flushes the BDONE lines too: no later step would, and the server frees a slot at its BDONE
        for (int k = 0; k < 2; ++k) bt.admit(k, reqs[(size_t) k]);
        bt.step();
        bt.slots.stop(0);
        bt.slots.stop(1);
        const int rows = bt.step();
        const bool both = bt.got[2].finish == "cancel" && bt.got[3].finish == "cancel";
        expect(rows == 0 && both && bt.unflushed == 0 && bt.slots.active() == 0,
               "BSTOP of every slot at once: both end (cancel) in a step of " + std::to_string(rows) +
                   " rows, their BDONE lines flushed (" + std::to_string(bt.unflushed) + " left unflushed)");
        const std::string why = bt.slots.fits(150, 20);
        expect(!why.empty() && bt.slots.fits(150, 11).empty(), "a request past a slot's " + std::to_string(kSlotCtx) +
                                                                   " positions runs alone: " + why);
    }

    // 5: the geometries, and the negative controls (the CPU reference's images)
    if (cpu) {
        Batch bt;
        const Req r{random_ids(g, 45, vocab, 50), 9, 0};
        bt.admit(1, r);
        for (int s = 0; s < 4; ++s) bt.step();
        const std::string rt = static_cast<RefDevice&>(bt.dev).checkpoint_round_trip(1);
        expect(rt.empty(), "a checkpoint packed from a slot's ring of " + std::to_string(m.attn_dims().n_swa) +
                               " cells restores into the live ring: " + (rt.empty() ? "the same state" : rt));
        for (const Spoil sp : {Spoil::WindowRun, Spoil::Carry}) {
            auto spoilt = std::make_shared<RefDevice>(m, 3);
            spoilt->spoil = sp;
            Batch bad(spoilt);
            bad.admit(0, r);
            bad.drain();
            const std::string diff = against_alone(r, bad.got[0]);
            expect(!diff.empty(), std::string("an admission without ") +
                                      (sp == Spoil::WindowRun ? "one window run" : "the pooling carry") +
                                      (diff.empty() ? ": not seen" : ": seen (" + diff + ")"));
        }
    }

    // 6: a long prompt read in chunks of kInterleave with one step after each but the last, beside 3 decoding
    // slots; every device of the check has its geometry (a prefill chunk of kLongChunk, kLongCtx positions)
    if (cpu) {
        auto device = [&m](int threads) { return std::make_shared<RefDevice>(m, threads, kLongChunk, kLongCtx, kLongCtx); };
        Batch bt(device(3));
        std::vector<Req> reqs;
        for (int k = 0; k < 3; ++k) {
            reqs.push_back({random_ids(g, 20 + 7 * k, vocab, 60 + k), 12, 0});
            bt.admit(k, reqs.back());
        }
        for (int s = 0; s < 2; ++s) bt.step();
        const Req longr{random_ids(g, 2600, vocab, 70), 6, 0};
        const int64_t steps0 = bt.dev.steps, rows0 = bt.dev.rows_total;
        bt.admit(3, longr, kInterleave);
        const int64_t between = bt.dev.steps - steps0, between_rows = bt.dev.rows_total - rows0;
        bt.drain();
        const Got want = alone(longr, kInterleave, device(1));
        const bool same = bt.got[3].finish == want.finish && same_got(bt.got[3], want, want.tokens.size());
        const int64_t chunks = (2600 + kInterleave - 1) / kInterleave;
        expect(same && between == chunks - 1 && between_rows == 3 * (chunks - 1),
               "a 2600-token prompt read in " + std::to_string(chunks) + " chunks of " + std::to_string(kInterleave) +
                   " (prefill chunk " + std::to_string(kLongChunk) + ") with " + std::to_string(between) + " steps of " +
                   std::to_string(between_rows) + " rows between them, beside 3 decoding slots: " +
                   (same ? "its tokens and logits its alone run's with the same chunks bit for bit"
                         : "it differs from its alone run with the same chunks"));
        bool ok = true;
        std::string what;
        for (size_t k = 0; k < reqs.size(); ++k) {
            const Got w = alone(reqs[k], 0, device(1));
            const bool eq = bt.got[k].finish == w.finish && same_got(bt.got[k], w, w.tokens.size());
            ok = ok && eq;
            what += (k ? "; " : "") + std::string("slot ") + std::to_string(k) + (eq ? " bit for bit" : " differs");
        }
        expect(ok, "the 3 slots decoding through the interleaved read: each its alone run's (C03's invariance): " + what);
    }

    // 7: the lanes' argmax against the plain loop
    if (cpu) {
        auto plain = [](const std::vector<float>& v) {
            int32_t best = 0;
            for (size_t i = 1; i < v.size(); ++i)
                if (v[i] > v[(size_t) best]) best = (int32_t) i;
            return best;
        };
        constexpr size_t kRealVocab = 129280;   // DeepSeek-V4.1-Flash's, the rows a real step picks from
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        std::uniform_real_distribution<float> u(-30.f, 30.f);
        std::vector<std::vector<float>> rows;
        for (int r = 0; r < 2000; ++r) {
            const size_t n = r < 40 ? (size_t) r + 1 : r % 7 == 0 ? (size_t) (g() % 300 + 1) : kRealVocab;
            std::vector<float> v(n);
            for (float& x : v) x = u(g);
            if (r % 3 == 1) {   // a tie of the maximum at two to five ids
                const float top = 31.f + (float) (r % 5);
                for (int k = 0; k < 2 + r % 4; ++k) v[g() % n] = top;
            }
            if (r % 5 == 2)
                for (int k = 0; k < 8; ++k) v[g() % n] = nan;
            if (r % 11 == 3) v[g() % n] = -inf;
            if (r % 13 == 4) v[g() % n] = inf;
            rows.push_back(std::move(v));
        }
        rows.push_back(std::vector<float>(1000, -inf));
        std::vector<float> lead(1000, 1.f);
        lead[0] = nan;
        lead[500] = 2.f;
        rows.push_back(std::move(lead));
        int64_t differ = 0;
        for (const std::vector<float>& v : rows) differ += e::batch_argmax(v) != plain(v);
        expect(differ == 0, "batch_argmax picks the plain loop's id on " + std::to_string(rows.size()) +
                            " rows (ties, NaN, -inf, +inf, lengths 1 to 40 and " + std::to_string(kRealVocab) + "): " +
                            std::to_string(differ) + " differ");

        // 8: a request alone, its every token picked by the Session from the same rows
        RowsBackend rb(rows);
        e::Session lone(rb, {});
        e::SessionRequest q;
        q.ids = {1};
        q.max_new = (int64_t) rows.size();
        const e::SessionResult res = lone.run(q, e::SessionEvents{});
        int64_t lone_differ = res.tokens.size() == rows.size() ? 0 : (int64_t) rows.size();
        for (size_t k = 0; k < res.tokens.size() && k < rows.size(); ++k) lone_differ += res.tokens[k] != plain(rows[k]);
        expect(res.finish == "length" && lone_differ == 0,
               "a request alone (the Session, greedy) picks the plain loop's id on each of the " +
                   std::to_string(rows.size()) + " rows, " + std::to_string(res.tokens.size()) + " tokens: " +
                   std::to_string(lone_differ) + " differ");
    }

    // 9: a lone slot (admitted with nothing decoding, every step one row) with a
    // prompt read beside it whose request never joins a slot (max_new 1): that read moves the slot's hot set;
    // the slot's bits do not follow it, so the slot ends reproducible and its alone run's bit for bit,
    // the same as with nothing read beside it (check 3's case); on the CPU reference the read beside moves the slot to
    // another placement too (the odd experts from the read on)
    {
        const Req r{random_ids(g, 29, vocab, 80), 10, 0};
        Batch lone;
        lone.admit(0, r);
        lone.drain();
        bool moved = false;   // the CPU reference's slab: the hot set's, the odd experts once the read beside starts
        std::shared_ptr<Device> dev = cpu ? std::make_shared<RefDevice>(m, 3, kChunk, kCtx, kSlotCtx,
                                                                        [&moved](int il, int32_t x, int64_t call) {
                                                                            return moved ? odd_slab(il, x, call)
                                                                                         : hot_slab(il, x, call);
                                                                        })
                                          : make_device(false);
        Batch bt(dev);
        bt.admit(0, r);
        for (int s = 0; s < 3; ++s) bt.step();
        moved = cpu;
        bt.read_only({random_ids(g, 23, vocab, 81), 1, 0});
        bt.drain();
        const std::string diff = against_alone(r, bt.got[0]);
        expect(lone.got[0].reproducible && bt.got[0].reproducible && diff.empty(),
               std::string("a lone slot with a max_new 1 request read beside it after 3 steps") +
                   (cpu ? " (and the slab moved to the odd experts from then on)" : "") + ": " +
                   (bt.got[0].reproducible ? "marked reproducible" : "marked not reproducible") +
                   " (alone, nothing read beside it: " +
                   (lone.got[0].reproducible ? "marked reproducible" : "marked not reproducible") + "); its tokens " +
                   (diff.empty() ? "its alone run's bit for bit" : diff));
    }

    // 10. four identical requests sent together,
    // admitted one after another into a batch whose slab is not the alone run's (the odd experts; then a placement
    // drawn anew at every call: a swap at every call and layer), each one its alone run's bit for bit (every token and
    // step's logits; the alone run on the hot set's slab), every one marked reproducible; prompts and steps through
    // both tiers (decode hits and misses, the prefill split's CPU experts and the GEMM)
    if (cpu) {
        const Req r{random_ids(g, 30, vocab, 90), 12, 0};
        const Got want = alone(r);
        for (const auto& [name, pl] : {std::pair<const char*, Placement>{"the odd experts in the slab", odd_slab},
                                       std::pair<const char*, Placement>{"a slab drawn anew at every call", moving_slab}}) {
            auto dev = std::make_shared<RefDevice>(m, 3, kChunk, kCtx, kSlotCtx, pl);
            Batch bt(dev);
            for (int k = 0; k < kSlots; ++k) {
                bt.admit(k, r);
                bt.step();
            }
            bt.drain();
            int same = 0, rep = 0;
            for (int k = 0; k < kSlots; ++k) {
                same += bt.got[(size_t) k].finish == want.finish && same_got(bt.got[(size_t) k], want, want.tokens.size());
                rep += bt.got[(size_t) k].reproducible;
            }
            char what[400];
            std::snprintf(what, sizeof what, "4 identical requests (30 prompt tokens, 12 generated) in a batch on %s: %d of 4 "
                          "their alone run's bit for bit, %d marked reproducible (decode %lld hit and %lld miss pairs, "
                          "prefill %lld split CPU pairs and %lld GEMM pairs)", name, same, rep, (long long) dev->hit_pairs,
                          (long long) dev->miss_pairs, (long long) dev->split_cpu_pairs, (long long) dev->gemm_pairs);
            expect(same == kSlots && rep == kSlots && dev->hit_pairs > 0 && dev->miss_pairs > 0 && dev->split_cpu_pairs > 0,
                   what);
        }
    }

    // 11. four different requests under a slab drawn anew at every call, one joining mid-batch (its prompt read
    // beside three decoding slots), one of them sampled (seed 7), and a one-token request read beside them: each its
    // alone run's bit for bit, every one reproducible; then what the check guards against: the same with the fork's
    // CPU numerics for the misses (the earlier engine; its alone runs on the hot set's slab in the same numerics)
    // differs, and a prompt read beside the slots in the mode off (its prefill split from the slab as it is) differs
    // from its alone run and ends not reproducible
    if (cpu) {
        auto run = [&](bool fourth_det, std::vector<Req>& reqs, Batch& bt) {
            for (int k = 0; k < 3; ++k) {
                bt.admit(k, reqs[(size_t) k]);
                bt.step();
            }
            for (int s = 0; s < 3; ++s) bt.step();
            reqs[3].det = fourth_det;
            bt.admit(3, reqs[3]);
            bt.read_only({random_ids(g, 19, vocab, 95), 1, 0});
            bt.drain();
        };
        std::vector<Req> reqs;
        for (int k = 0; k < kSlots; ++k)
            reqs.push_back({random_ids(g, 21 + 11 * k, vocab, 100 + k), 9 + 2 * k, k == 1 ? 7ull : 0ull});
        {
            auto dev = std::make_shared<RefDevice>(m, 3, kChunk, kCtx, kSlotCtx, moving_slab);
            Batch bt(dev);
            run(true, reqs, bt);
            bool ok = true;
            std::string what;
            for (size_t k = 0; k < reqs.size(); ++k) {
                const std::string diff = against_alone(reqs[k], bt.got[k]);
                ok = ok && diff.empty() && bt.got[k].reproducible;
                what += (k ? "; " : "") + std::string("request ") + std::to_string(k) +
                        (diff.empty() ? " bit for bit" : ": " + diff) + (bt.got[k].reproducible ? ", reproducible" : ", not reproducible");
            }
            expect(ok && dev->split_in_slab > 0,
                   "4 different requests (one sampled, one joining mid-batch beside 3 decoding slots, a one-token request "
                   "read beside them) under a slab drawn anew at every call: " + what + " (" +
                       std::to_string(dev->split_in_slab) + " split CPU experts the slab held at their call)");
        }
        {
            auto fork_dev = std::make_shared<RefDevice>(m, 3, kChunk, kCtx, kSlotCtx, moving_slab, e::CpuNumerics::Fork);
            Batch bt(fork_dev);
            run(true, reqs, bt);
            int differ = 0;
            for (size_t k = 0; k < reqs.size(); ++k) {
                const Got want = alone(reqs[k], 0, std::make_shared<RefDevice>(m, 1, kChunk, kCtx, kSlotCtx, hot_slab,
                                                                               e::CpuNumerics::Fork));
                differ += !(bt.got[k].finish == want.finish && same_got(bt.got[k], want, want.tokens.size()));
            }
            expect(differ > 0, "the same with the fork's CPU numerics for the misses (the default): " +
                                   std::to_string(differ) + " of 4 requests differ from their alone runs (seen)");
        }
        {
            auto dev = std::make_shared<RefDevice>(m, 3, kChunk, kCtx, kSlotCtx, moving_slab);
            Batch bt(dev);
            run(false, reqs, bt);
            Req alone_req = reqs[3];
            alone_req.det = true;
            const std::string diff = against_alone(alone_req, bt.got[3]);
            expect(!diff.empty() && !bt.got[3].reproducible,
                   std::string("the request read beside the slots in the mode off (its prefill split from the slab as it "
                               "is): ") + (diff.empty() ? "its alone run's (not seen)" : "differs from its alone run (seen)") +
                       (bt.got[3].reproducible ? ", marked reproducible" : ", marked not reproducible"));
        }
    }

    // 12. the server's promotion (serve/server.py
    // generate_batched). A request on the solo path (a GEN of its whole max_new) stopped after k tokens, then sent as a
    // BGEN of its prompt and those tokens into slot 0, and 3 other requests admitted beside it: its every token, and the
    // logits of every call from its first decode on (the solo run's decodes, the admission's read of its last token,
    // the slot's steps), its alone run's bit for bit. An RTX PRO 6000 session's promoted chains left their alone runs at
    // tokens 67 and 91 while the Session read that last token by prefill(T=1).
    for (int64_t k : {1, 2, 5}) {
        const Req r{random_ids(g, 30, vocab, (int32_t) (110 + k)), 14, 0};
        const Got want = alone(r);
        Batch bt;
        bt.dev.begin_request();
        e::SessionRequest q;
        q.ids = r.ids;
        q.max_new = r.max_new;
        q.snapshot = false;
        int64_t emitted = 0;
        e::SessionEvents ev;
        ev.token = [&](int32_t) { ++emitted; };
        ev.stop = [&] { return emitted >= k; };
        if (const std::string why = bt.session.check(q); !why.empty()) die("promotion: " + why);
        const size_t first_call = (size_t) (((int64_t) r.ids.size() + kChunk - 1) / kChunk);
        const e::SessionResult solo = bt.session.run(q, ev);
        Req moved = r;
        moved.ids.insert(moved.ids.end(), solo.tokens.begin(), solo.tokens.end());
        moved.max_new = r.max_new - k;
        bt.admit(0, moved);
        // the live calls from the first decode on: the solo run's k - 1 decodes and the admission's one read call
        const bool calls_ok = bt.dev.calls.size() == first_call + (size_t) k;
        std::vector<std::vector<float>> calls(bt.dev.calls.begin() + (std::ptrdiff_t) first_call, bt.dev.calls.end());
        for (int b = 1; b < kSlots; ++b) {
            bt.admit(b, {random_ids(g, 18 + 5 * b, vocab, (int32_t) (120 + b)), 8, 0});
            bt.step();
        }
        bt.drain();
        Got chain;
        chain.tokens = solo.tokens;
        chain.tokens.insert(chain.tokens.end(), bt.got[0].tokens.begin(), bt.got[0].tokens.end());
        chain.logits = calls;
        chain.logits.insert(chain.logits.end(), bt.got[0].logits.begin(), bt.got[0].logits.end());
        const bool same = solo.finish == "cancel" && bt.got[0].finish == want.finish &&
                          chain.tokens.size() == want.tokens.size() && same_got(chain, want, want.tokens.size());
        expect(same && calls_ok && bt.got[0].reproducible,
               "a request stopped on the solo path after " + std::to_string(k) + " tokens and promoted into slot 0 with "
               "its prompt and tokens, 3 requests beside it: " +
                   (same ? "every token and every call's logits from its first decode on its alone run's bit for bit"
                         : "its tokens or a call's logits differ from its alone run's") +
                   (bt.got[0].reproducible ? ", marked reproducible" : ", marked not reproducible"));
    }

    // 13. the diagnostic: on the same state (a prompt and k tokens fed, replayed from position 0 in the
    // deterministic mode), the next token read by prefill(T=1) against decode: how many of the logits differ (reported,
    // not a check: what feeding it by decode avoids)
    for (int64_t k : {1, 2, 5}) {
        const Req r{random_ids(g, 30, vocab, (int32_t) (130 + k)), k + 1, 0};
        std::vector<int32_t> fed;
        auto replay = [&](Device& dev) {
            dev.reset();
            dev.begin_request();
            std::vector<float> logits;
            for (size_t p = 0; p < r.ids.size(); p += (size_t) kChunk)
                dev.prefill(r.ids.data() + p, std::min<int64_t>(kChunk, (int64_t) (r.ids.size() - p)), logits);
            fed.clear();
            for (int64_t s = 0; s < k; ++s) {
                fed.push_back(e::batch_argmax(logits));
                if (s + 1 < k) dev.decode(fed.back(), logits);
            }
        };
        std::vector<float> by_decode, by_prefill;
        {
            auto owner = make_device(true);
            replay(*owner);
            owner->decode(fed.back(), by_decode);
        }
        {
            auto owner = make_device(true);
            replay(*owner);
            owner->prefill(&fed.back(), 1, by_prefill);
        }
        int64_t differ = 0;
        float max_abs = 0;
        for (size_t i = 0; i < by_decode.size() && i < by_prefill.size(); ++i) {
            differ += std::memcmp(&by_decode[i], &by_prefill[i], 4) != 0;
            max_abs = std::max(max_abs, std::fabs(by_decode[i] - by_prefill[i]));
        }
        std::printf("info the token after %lld (position %zu): read by prefill(T=1) against decode, %lld of %zu logits "
                    "differ (max |diff| %.3g), argmax %d against %d\n", (long long) k, r.ids.size() + (size_t) k - 1,
                    (long long) differ, by_decode.size(), (double) max_abs, e::batch_argmax(by_prefill),
                    e::batch_argmax(by_decode));
    }
}

}  // namespace
