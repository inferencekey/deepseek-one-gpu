// src/dsv41/engine/forward.cpp - the layer loop of include/strata/dsv41/engine/forward.hpp.
//
// Every buffer is the size of the largest chunk and is allocated in init (Strata's prefill.cpp carve); a decode step
// is a chunk of one token whose attention takes attn_decode's split and whose experts take the hit/miss split. The
// stream copies are [T][HC][E] f32 (x), the sublayer coefficients [T][HC] and [T][HC][HC]; pre_prev (the previous
// FFN's collapse weights, the V4.1 lag) and pre_f swap after each layer.
//
// The parity mode (cfg.forced, selection.hpp) uploads the reference's router ids and index rows of each call and layer
// before the router and the attention take them (a chunk may be part of a reference ubatch); the dumps write what the
// call used, so a forced run's dump is the reference's selections with the engine's scores. With forced_moe_input
// each layer's h is then overwritten by the reference's MoE input right after the FFN's collapse and
// norm: the dumps' ffn_norm are the reference's, their ffn_moe_out and weights the engine's MoE path on it.
//
// Timing (cfg.timing) records an event pair around each stage and reads them all once the call's final wait is done,
// so the stages run as they would untimed (the loop waited after every stage). Decode with cfg.graphs and no
// timing runs the token graph (run_graph): token_body queues the token's whole sequence, captured once; per
// token the host fills the mapped inputs, launches it, serves the doorbells and waits for the logits (the inputs
// and the logits cross PCIe through mapped memory, moved by the graph's own launches, so no copy engine work of the
// token queues behind the hot set's swap copies). Without the
// graph (timing, --no-graphs) a decode token queues the same device work layer by layer from run(), the experts'
// call serving its doorbell before the next layer is queued.
#include "strata/dsv41/engine/forward.hpp"

#include "strata/dsv41/engine/indexer.hpp"

#include "ffn_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 forward: %s\n", msg);
    std::exit(1);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) failf("%s: %s", what, cudaGetErrorString(e));
}

constexpr double kMiB = 1024.0 * 1024.0;

void write_file(const std::string& path, const void* data, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) failf("cannot write %s: %s", path.c_str(), std::strerror(errno));
    const bool ok = std::fwrite(data, 1, bytes, f) == bytes;
    if (std::fclose(f) != 0 || !ok) failf("writing %s failed", path.c_str());
}

const char* role_name(AttnRole r) {
    switch (r) {
        case AttnRole::Window: return "window";
        case AttnRole::Full: return "Full";
        case AttnRole::Reindex: return "Reindex";
        case AttnRole::Reuse: return "Reuse";
    }
    return "?";
}

}  // namespace

struct Forward::Impl {
    const Model* m = nullptr;
    ForwardConfig cfg;
    cudaStream_t stream = nullptr;
    QmmContext qmm;   // the stream's one qmatmul context, before the engines that hold it
    AttnEngine attn;
    DenseEngine dense;
    KvCache kv;
    ExpertEngine experts;
    EngramGather gather;
    std::vector<void*> allocs;
    size_t buffer_bytes = 0;

    int32_t* tokens = nullptr;
    float *x = nullptr, *pre_buf[2] = {nullptr, nullptr}, *pre_a = nullptr, *post = nullptr, *comb = nullptr;
    float *pre_prev = nullptr, *pre_f = nullptr;   // pre_buf[0] and [1], swapped after each layer
    float *h = nullptr, *a = nullptr, *sh = nullptr, *ffn = nullptr, *w = nullptr, *logits = nullptr;
    uint8_t* hq = nullptr;          // decode: h's q8_1 from hc_decode_step, the sublayer's matmuls read it
    int32_t* ids = nullptr;
    std::vector<int32_t> rows;      // the engram rows of a chunk [T][cols]
    std::vector<float> host_x;      // a dumped l_last
    // the parity mode: the reference's router ids [T][K] and index rows [T][top_k] of the current call and layer
    int32_t *forced_ids = nullptr, *forced_rows = nullptr;
    std::vector<int32_t> forced_ids_host, forced_rows_host;
    std::vector<float> moe_in_host;   // the reference's MoE input of the current call and layer
    // the dumps: an index source's scores [T][ld] recomputed from its queries, the host copies of the selections
    float* dump_scores = nullptr;
    int64_t dump_scores_ld = 0;
    // the dumps' MoE values (dump_values): the routed experts' sum [T][E] of the current layer before the
    // shared expert is added, the real-model oracle's `out` (the experts run with shared null, the add after)
    float* moe_out = nullptr;
    std::vector<int32_t> host_i32;
    std::vector<float> host_f32, host_bias;

    // timing: events from a pool, each stage's pair with the sums it adds to, read after the call's wait
    std::vector<cudaEvent_t> tev;
    size_t tev_used = 0;
    struct Mark {
        double* total;
        double* layer;
        cudaEvent_t a, b;
    };
    std::vector<Mark> marks;
    // the token graph and the mapped pinned host memory its first launch reads (copy_from_mapped, no
    // copy engine): the token and its position, per engram layer its rows, per layer the forced or synthetic router
    // ids [layer][K] and an index source's forced rows [layer][top_k] (with their device copies g_ids, g_rows); the
    // logits its last launch writes into mapped memory (h_logits); each region's device view (*_m); the dumps'
    // per-layer copies. A decode token of run() takes the same inputs and logits.
    cudaGraphExec_t g_token = nullptr;
    std::vector<void*> host_allocs;
    int32_t *h_token = nullptr, *h_pos = nullptr, *h_ids = nullptr, *h_rows = nullptr;
    int32_t *h_token_m = nullptr, *h_pos_m = nullptr, *h_ids_m = nullptr, *h_rows_m = nullptr;
    float *h_logits = nullptr, *h_logits_m = nullptr;
    int32_t *g_ids = nullptr, *g_rows = nullptr;
    std::vector<uint8_t*> eng_stage, eng_stage_m, eng_dev;
    size_t eng_bytes = 0;
    // the token graph's device time (ForwardTimes::graph_ms)
    cudaEvent_t graph_a = nullptr, graph_b = nullptr;
    std::vector<int32_t> eng_rows;
    float *snap_x = nullptr, *snap_probs = nullptr, *snap_iq = nullptr, *snap_iw = nullptr;
    int32_t *snap_ids = nullptr, *snap_idx = nullptr;
    // the MoE values' per-layer copies (dump_values): ffn_norm [layer][E], the weights [layer][K], the routed sum
    float *snap_cur = nullptr, *snap_w = nullptr, *snap_out = nullptr;
    // the batch: each slot's KV cache; the step's tokens and positions in mapped memory (h_btok, h_bpos, their
    // device views) and on the device (tokens, d_bpos), its rows' logits [batch][n_vocab] on the device and in mapped
    // memory; per batch size B its graph and the slots it was captured with (index B)
    std::vector<std::unique_ptr<KvCache>> slot_kv;
    int32_t *h_btok = nullptr, *h_btok_m = nullptr, *h_bpos = nullptr, *h_bpos_m = nullptr, *d_bpos = nullptr;
    float *blogits = nullptr, *h_blogits = nullptr, *h_blogits_m = nullptr;
    cudaGraphExec_t g_batch[kMmvqMaxCols + 1] = {};
    std::vector<int> g_batch_slots[kMmvqMaxCols + 1];

    ~Impl() {
        if (g_token) cudaGraphExecDestroy(g_token);
        for (cudaGraphExec_t g : g_batch)
            if (g) cudaGraphExecDestroy(g);
        for (void* p : allocs) cudaFree(p);
        for (void* p : host_allocs) cudaFreeHost(p);
        for (cudaEvent_t e : tev) cudaEventDestroy(e);
        for (cudaEvent_t e : {graph_a, graph_b})
            if (e) cudaEventDestroy(e);
        if (stream) cudaStreamDestroy(stream);
    }

    template <typename T> bool alloc(T*& p, size_t n, const char* what, std::string& err) {
        void* q = nullptr;
        if (const cudaError_t e = cudaMalloc(&q, std::max<size_t>(n * sizeof(T), 16)); e != cudaSuccess) {
            err = std::string("forward buffer ") + what + ": " + cudaGetErrorString(e);
            return false;
        }
        allocs.push_back(q);
        buffer_bytes += n * sizeof(T);
        p = static_cast<T*>(q);
        return true;
    }

    // mapped pinned memory (cudaHostAllocMapped, zeroed): the host's pointer in p, the device's view in *dev
    template <typename T> bool mapped(T*& p, T** dev, size_t n, const char* what, std::string& err) {
        void* q = nullptr;
        if (const cudaError_t e = cudaHostAlloc(&q, std::max<size_t>(n * sizeof(T), 16), cudaHostAllocMapped); e != cudaSuccess) {
            err = std::string("forward mapped ") + what + ": " + cudaGetErrorString(e);
            return false;
        }
        host_allocs.push_back(q);
        std::memset(q, 0, std::max<size_t>(n * sizeof(T), 16));
        if (const cudaError_t e = cudaHostGetDevicePointer((void**) dev, q, 0); e != cudaSuccess) {
            err = std::string("forward: the device view of mapped ") + what + ": " + cudaGetErrorString(e);
            return false;
        }
        p = static_cast<T*>(q);
        return true;
    }

    // a decode token's inputs from mapped memory in one launch: the token, the position, with sel the router ids,
    // with rows the index rows, every engram layer's rows (each region filled by the host before the launch)
    void decode_inputs(bool sel, bool rows) {
        MappedCopy c[kMaxMappedCopies];
        int n = 0;
        auto add = [&](void* dst, const void* src, size_t bytes) {
            if (n == kMaxMappedCopies) failf("a decode token's inputs: more than %d mapped copies", kMaxMappedCopies);
            c[n++] = MappedCopy{dst, src, (int64_t) (bytes / 4)};
        };
        const int K = m->dense_dims().n_expert_used, top_k = m->attn_dims().top_k;
        add(tokens, h_token_m, 4);
        add(attn.scratch().pos, h_pos_m, 4);
        if (sel) add(g_ids, h_ids_m, (size_t) m->n_layer() * K * 4);
        if (rows) add(g_rows, h_rows_m, (size_t) m->n_layer() * top_k * 4);
        for (size_t e = 0; e < eng_stage.size(); ++e) add(eng_dev[e], eng_stage_m[e], eng_bytes);
        copy_from_mapped(c, n, stream);
    }

    cudaEvent_t event() {
        if (tev_used == tev.size()) {
            cudaEvent_t e = nullptr;
            ck(cudaEventCreate(&e), "cudaEventCreate");
            tev.push_back(e);
        }
        return tev[tev_used++];
    }
    // fn's device time added to *total (and *layer) once the call is done, when timing
    template <typename F> void timed(double& total, double* layer, F&& fn) {
        if (!cfg.timing) return fn();
        Mark mk{&total, layer, event(), nullptr};
        ck(cudaEventRecord(mk.a, stream), "cudaEventRecord");
        fn();
        mk.b = event();
        ck(cudaEventRecord(mk.b, stream), "cudaEventRecord");
        marks.push_back(mk);
    }
    void read_marks() {
        for (const Mark& mk : marks) {
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, mk.a, mk.b), "cudaEventElapsedTime");
            *mk.total += ms;
            if (mk.layer) *mk.layer += ms;
        }
        marks.clear();
        tev_used = 0;
    }
    void token_body(KvCache& kvc, int64_t p0);
    // a batch step's inputs from mapped memory (its B tokens and positions, every engram layer's B rows) and its
    // sequence for B > 1 rows on the slots' caches kvs at host positions p (batch_body, forward.hpp)
    void batch_inputs(int B);
    void batch_body(int B, KvCache* const* kvs, const int64_t* p);
};

// The token graph's sequence for the token at p0 of the sequence in kvc (the live one, or a batch slot's, B = 1)
// (captured once: p0 is the capture's position, only the host's bookkeeping of attn_decode uses it; the launches read
// the position from the attention's scratch). run()'s decode order, with the experts split in halves around the
// shared expert so the CPU's misses overlap it and the hits.
void Forward::Impl::token_body(KvCache& kvc, int64_t p0) {
    const DenseDims& dd = m->dense_dims();
    const int K = dd.n_expert_used, top_k = m->attn_dims().top_k;
    const int64_t E = dd.n_embd;
    const bool sel = cfg.forced || cfg.routing;
    decode_inputs(sel, cfg.forced != nullptr);
    pre_prev = pre_buf[0];
    pre_f = pre_buf[1];
    dense_embed(dense, m->token_embd(), tokens, 1, x, pre_prev);
    // each sublayer's hyper-connections in one hc_decode_step, the FFN output's expansion folded into
    // the next layer's (except before an engram, which adds to the expanded x, and with the dumps, which read x
    // after every layer)
    const bool defer = !snap_x;
    bool pending = false;
    for (int il = 0; il < m->n_layer(); ++il) {
        const LayerWeights& L = m->layer(il);
        if (L.engram >= 0) {
            if (pending) hc_expand(dense, ffn, x, post, comb, 1, x);
            pending = false;
            engram_forward(dense, L.eng, eng_dev[(size_t) L.engram], 1, x);
        }
        hc_decode_step(dense, L.hc_attn, pending ? ffn : nullptr, x, pre_prev, L.attn_norm, pre_a, post, comb, h, hq);
        const int32_t* frows = cfg.forced && L.role.index_source ? g_rows + (size_t) il * top_k : nullptr;
        attn_decode(attn, L.role, L.attn, kvc, h, p0, a, cfg.readers, frows, true, hq);
        hc_decode_step(dense, L.hc_ffn, a, x, pre_a, L.ffn_norm, pre_f, post, comb, h, hq);
        dense_router(dense, L.router, h, 1, ids, w, RouterMatmul::Auto, sel ? g_ids + (size_t) il * K : nullptr);
        experts.decode_publish(il, h, ids, 1);
        dense_shared_expert(dense, L.shared, h, 1, sh, hq);
        // with the MoE dumps the routed sum is copied before the shared expert's add (the same bits: the
        // combine's own add is one rounding too)
        experts.decode_finish(il, h, ids, w, snap_out ? nullptr : sh, 1, ffn, hq);
        if (snap_out) {
            ck(cudaMemcpyAsync(snap_out + (size_t) il * E, ffn, (size_t) E * 4, cudaMemcpyDeviceToDevice, stream),
               "a dump's copy");
            expert_add_shared(ffn, sh, E, stream);
        }
        pending = defer;
        if (!defer) hc_expand(dense, ffn, x, post, comb, 1, x);
        std::swap(pre_prev, pre_f);
        if (snap_x) {
            auto snap = [&](void* dst, const void* src, size_t bytes) {
                ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream), "a dump's copy");
            };
            const AttnDims& ad = m->attn_dims();
            snap(snap_x + (size_t) il * kHc * E, x, (size_t) kHc * E * 4);
            snap(snap_ids + (size_t) il * K, ids, (size_t) K * 4);
            snap(snap_probs + (size_t) il * dd.n_expert, dense.scratch().probs, (size_t) dd.n_expert * 4);
            if (snap_cur) {
                snap(snap_cur + (size_t) il * E, h, (size_t) E * 4);
                snap(snap_w + (size_t) il * K, w, (size_t) K * 4);
            }
            if (L.role.index_source) {
                snap(snap_idx + (size_t) il * top_k, kvc.layer(il).ids, (size_t) top_k * 4);
                snap(snap_iq + (size_t) il * ad.idx_heads * ad.idx_dim, attn.scratch().iq, (size_t) ad.idx_heads * ad.idx_dim * 4);
                snap(snap_iw + (size_t) il * ad.idx_heads, attn.scratch().iw, (size_t) ad.idx_heads * 4);
            }
        }
    }
    if (pending) hc_expand(dense, ffn, x, post, comb, 1, x);
    dense_head(dense, x, pre_prev, 1, m->output_norm(), m->output(), logits);
    copy_to_mapped(h_logits_m, logits, dd.n_vocab, stream);
}

void Forward::Impl::batch_inputs(int B) {
    MappedCopy c[kMaxMappedCopies];
    int n = 0;
    c[n++] = MappedCopy{tokens, h_btok_m, B};
    c[n++] = MappedCopy{d_bpos, h_bpos_m, B};
    for (size_t e = 0; e < eng_stage.size(); ++e) {
        if (n == kMaxMappedCopies) failf("a batch step's inputs: more than %d mapped copies", kMaxMappedCopies);
        c[n++] = MappedCopy{eng_dev[e], eng_stage_m[e], (int64_t) ((size_t) B * eng_bytes / 4)};
    }
    copy_from_mapped(c, n, stream);
}

void Forward::Impl::batch_body(int B, KvCache* const* kvs, const int64_t* p) {
    const DenseDims& dd = m->dense_dims();
    const int64_t E = dd.n_embd;
    batch_inputs(B);
    pre_prev = pre_buf[0];
    pre_f = pre_buf[1];
    dense_embed(dense, m->token_embd(), tokens, B, x, pre_prev);
    // token_body's order with B rows: each sublayer's hc step (the rows' q8_1 after it), the FFN output's expansion
    // folded into the next layer's step (not before an engram layer, which adds to the expanded stream)
    bool pending = false;
    QActs qa;
    for (int il = 0; il < m->n_layer(); ++il) {
        const LayerWeights& L = m->layer(il);
        if (L.engram >= 0) {
            if (pending) hc_expand(dense, ffn, x, post, comb, B, x);
            pending = false;
            engram_forward(dense, L.eng, eng_dev[(size_t) L.engram], B, x);
        }
        hc_prefill_step(dense, L.hc_attn, pending ? ffn : nullptr, x, pre_prev, L.attn_norm, B, pre_a, post, comb, h);
        qact_quantize(h, nullptr, E, B, E, QActLayout::Q8_1, hq, stream, qa);
        attn_decode_batch(attn, L.role, L.attn, kvs, B, h, hq, d_bpos, p, a, cfg.readers);
        hc_prefill_step(dense, L.hc_ffn, a, x, pre_a, L.ffn_norm, B, pre_f, post, comb, h);
        qact_quantize(h, nullptr, E, B, E, QActLayout::Q8_1, hq, stream, qa);
        dense_router(dense, L.router, h, B, ids, w);
        experts.decode_publish(il, h, ids, B);
        dense_shared_expert(dense, L.shared, h, B, sh, hq);
        experts.decode_finish(il, h, ids, w, sh, B, ffn, hq);
        pending = true;
        std::swap(pre_prev, pre_f);
    }
    if (pending) hc_expand(dense, ffn, x, post, comb, B, x);
    dense_head(dense, x, pre_prev, B, m->output_norm(), m->output(), blogits);
    copy_to_mapped(h_blogits_m, blogits, (int64_t) B * dd.n_vocab, stream);
}

namespace {

// the batch (its rows at most MMVQ's columns: every batched op runs them on its decode path), for Forward::init
// and the dry run alike. --batch-ctx is read only with a batch: --batch 0 takes every ctx it took before, the
// default 32768 included (the dev pod's forced runs at ctx 2048 were refused)
void check_batch_config(const ForwardConfig& cfg) {
    if (cfg.batch < 0 || cfg.batch > kMmvqMaxCols) failf("--batch %d (0..%lld)", cfg.batch, (long long) kMmvqMaxCols);
    if (!cfg.batch) return;
    if (cfg.batch_ctx < 1 || cfg.batch_ctx > cfg.ctx)
        failf("--batch %d with --batch-ctx %lld (1..ctx %lld)", cfg.batch, (long long) cfg.batch_ctx, (long long) cfg.ctx);
    if (cfg.forced || cfg.routing || !cfg.dump_dir.empty() || cfg.readers != ReaderMode::Reuse)
        failf("--batch with a forced selection, a synthetic routing, dumps or --readers dense (the batch reads with Reuse "
              "Mode and dumps nothing)");
}

}  // namespace

Forward::Forward() = default;
Forward::~Forward() = default;

bool Forward::init(const Model& m, const ExpertConfig& ecfg, const ForwardConfig& cfg_in, std::string& err) {
    if (impl_) failf("Forward::init called twice");
    if (m.dry()) failf("Forward::init on a dry model (forward_dry_run plans it)");
    impl_ = std::make_unique<Impl>();
    Impl& I = *impl_;
    I.m = &m;
    I.cfg = cfg_in;
    ForwardConfig& cfg = I.cfg;
    if (cfg.ctx < 1 || cfg.chunk < 1) failf("ctx %lld, chunk %lld", (long long) cfg.ctx, (long long) cfg.chunk);
    cfg.chunk = std::min(cfg.chunk, cfg.ctx);
    if (cfg.chunk > 65535) failf("chunk %lld (at most 65535)", (long long) cfg.chunk);
    if (!cfg.dump_dir.empty() && mkdir(cfg.dump_dir.c_str(), 0755) != 0 && errno != EEXIST)
        failf("cannot make %s: %s", cfg.dump_dir.c_str(), std::strerror(errno));
    ck(cudaStreamCreateWithFlags(&I.stream, cudaStreamNonBlocking), "cudaStreamCreate");
    const AttnDims& ad = m.attn_dims();
    const DenseDims& dd = m.dense_dims();
    // one qmatmul context for the stream: its dequant scratch the largest matrix any of the three engines multiplies
    // at once (attn_q_b's 80 MiB at the real dims; follow-up: one scratch per stream)
    if (cfg.forced_moe_input && !cfg.forced) failf("the reference's MoE input without a forced selection");
    if (!I.qmm.init(err, cfg.prefill, std::max(attn_max_matrix_elems(ad), (int64_t) dd.n_ff * dd.n_embd))) return false;
    if (!I.kv.init(ad, m.roles(), KvCacheConfig{cfg.ctx, cfg.chunk}, err)) return false;
    if (cfg.checkpoints < 0) failf("checkpoints %d (at least 0)", cfg.checkpoints);
    if (!I.kv.init_checkpoints(cfg.checkpoints, err)) return false;
    slot_history_.assign((size_t) cfg.checkpoints, {});
    if (!I.attn.init(ad, cfg.chunk, cfg.ctx, I.stream, err, cfg.prefill, &I.qmm)) return false;
    check_batch_config(cfg);
    const int nb = std::max(1, cfg.batch);
    if (!I.dense.init(dd, cfg.chunk, nb, cfg.engram_chunk, I.stream, err, cfg.prefill, &I.qmm)) return false;
    if (!I.experts.init(m, ecfg, cfg.chunk, &I.qmm, I.stream, false, err)) return false;
    if (cfg.batch) {
        if (!I.attn.init_batch(cfg.batch, err)) return false;
        for (int b = 0; b < cfg.batch; ++b) {
            I.slot_kv.push_back(std::make_unique<KvCache>());
            if (!I.slot_kv.back()->init(ad, m.roles(), KvCacheConfig{cfg.batch_ctx, 1}, err)) {
                err = "batch slot " + std::to_string(b) + " of " + std::to_string(cfg.batch) + ": " + err;
                return false;
            }
        }
        batch_pos_.assign((size_t) cfg.batch, 0);
        batch_history_.assign((size_t) cfg.batch, {});
    }
    I.experts.set_timing(cfg.timing);
    layer_times_.assign((size_t) m.n_layer(), ForwardLayerTimes{});
    const EngramHash& eh = m.host().engram;
    if (!eh.layer_ids.empty() && !I.gather.init(cfg.chunk * eh.cols(), m.table(0).row_bytes(), 8, err)) return false;
    const int64_t T = cfg.chunk, E = dd.n_embd, K = dd.n_expert_used, NL = m.n_layer();
    // a decode token's inputs and logits in mapped memory
    if (!I.mapped(I.h_token, &I.h_token_m, 1, "token", err) || !I.mapped(I.h_pos, &I.h_pos_m, 1, "position", err) ||
        !I.mapped(I.h_logits, &I.h_logits_m, (size_t) dd.n_vocab, "logits", err))
        return false;
    I.eng_bytes = eh.layer_ids.empty() ? 0 : (size_t) eh.cols() * m.table(0).row_bytes();
    if (I.eng_bytes % 4) failf("engram rows of %zu bytes per token: the mapped copy moves 4-byte words", I.eng_bytes);
    // a decode token's rows, one per batch row
    for (size_t e = 0; e < eh.layer_ids.size(); ++e) {
        uint8_t *st = nullptr, *sm = nullptr, *dv = nullptr;
        if (!I.mapped(st, &sm, I.eng_bytes * nb, "engram rows", err) || !I.alloc(dv, I.eng_bytes * nb, "engram rows", err))
            return false;
        I.eng_stage.push_back(st);
        I.eng_stage_m.push_back(sm);
        I.eng_dev.push_back(dv);
    }
    if ((cfg.forced || cfg.routing) && (!I.mapped(I.h_ids, &I.h_ids_m, (size_t) (NL * K), "router ids", err) ||
                                        !I.alloc(I.g_ids, (size_t) (NL * K), "router ids", err)))
        return false;
    if (cfg.forced && (!I.mapped(I.h_rows, &I.h_rows_m, (size_t) (NL * ad.top_k), "index rows", err) ||
                       !I.alloc(I.g_rows, (size_t) (NL * ad.top_k), "index rows", err)))
        return false;
    if (cfg.batch && (!I.mapped(I.h_btok, &I.h_btok_m, (size_t) cfg.batch, "batch tokens", err) ||
                      !I.mapped(I.h_bpos, &I.h_bpos_m, (size_t) cfg.batch, "batch positions", err) ||
                      !I.mapped(I.h_blogits, &I.h_blogits_m, (size_t) cfg.batch * dd.n_vocab, "batch logits", err) ||
                      !I.alloc(I.d_bpos, (size_t) cfg.batch, "batch positions", err) ||
                      !I.alloc(I.blogits, (size_t) cfg.batch * dd.n_vocab, "batch logits", err)))
        return false;
    ck(cudaEventCreate(&I.graph_a), "cudaEventCreate (token graph)");
    ck(cudaEventCreate(&I.graph_b), "cudaEventCreate (token graph)");
    if (!cfg.dump_dir.empty() &&
        (!I.alloc(I.snap_x, (size_t) (NL * kHc * E), "dumped streams", err) ||
         !I.alloc(I.snap_ids, (size_t) (NL * K), "dumped router ids", err) ||
         !I.alloc(I.snap_probs, (size_t) (NL * dd.n_expert), "dumped router probs", err) ||
         !I.alloc(I.snap_idx, (size_t) (NL * ad.top_k), "dumped index rows", err) ||
         !I.alloc(I.snap_iq, (size_t) (NL * ad.idx_heads * ad.idx_dim), "dumped index queries", err) ||
         !I.alloc(I.snap_iw, (size_t) (NL * ad.idx_heads), "dumped index weights", err)))
        return false;
    if (!cfg.dump_dir.empty() && cfg.dump_values &&
        (!I.alloc(I.snap_cur, (size_t) (NL * E), "dumped MoE inputs", err) ||
         !I.alloc(I.snap_w, (size_t) (NL * K), "dumped expert weights", err) ||
         !I.alloc(I.snap_out, (size_t) (NL * E), "dumped routed sums", err) ||
         !I.alloc(I.moe_out, (size_t) (T * E), "dumped routed sums", err)))
        return false;
    bool any_index = false;
    for (const AttnLayer& l : m.roles()) any_index = any_index || l.index_source;
    if (cfg.forced && cfg.routing) failf("a forced selection and a synthetic routing at once");
    if (cfg.forced && (!I.alloc(I.forced_ids, (size_t) (T * K), "forced router ids", err) ||
                       !I.alloc(I.forced_rows, (size_t) (T * ad.top_k), "forced index rows", err)))
        return false;
    if (cfg.routing && !I.alloc(I.forced_ids, (size_t) (T * K), "synthetic router ids", err)) return false;
    // the dumps' index scores: a ratio-1 source's keys reach ctx rows (the parity runs are the tiny models': 16 MiB)
    I.dump_scores_ld = (cfg.ctx + 31) / 32 * 32;
    if (!cfg.dump_dir.empty() && cfg.dump_values && any_index &&
        !I.alloc(I.dump_scores, (size_t) (T * I.dump_scores_ld), "dumped index scores", err))
        return false;
    return I.alloc(I.tokens, (size_t) T, "tokens", err) && I.alloc(I.x, (size_t) (T * kHc * E), "stream", err) &&
           I.alloc(I.pre_buf[0], (size_t) (T * kHc), "pre", err) && I.alloc(I.pre_a, (size_t) (T * kHc), "pre", err) &&
           I.alloc(I.pre_buf[1], (size_t) (T * kHc), "pre", err) && I.alloc(I.post, (size_t) (T * kHc), "post", err) &&
           I.alloc(I.comb, (size_t) (T * kHc * kHc), "comb", err) && I.alloc(I.h, (size_t) (T * E), "sublayer input", err) &&
           I.alloc(I.a, (size_t) (T * E), "attention output", err) && I.alloc(I.sh, (size_t) (T * E), "shared expert", err) &&
           I.alloc(I.ffn, (size_t) (T * E), "ffn output", err) && I.alloc(I.ids, (size_t) (T * K), "expert ids", err) &&
           I.alloc(I.w, (size_t) (T * K), "expert weights", err) && I.alloc(I.logits, (size_t) dd.n_vocab, "logits", err) &&
           I.alloc(I.hq, qact_bytes(QActLayout::Q8_1, nb, E), "decode q8_1 rows", err);
}

const ExpertEngine& Forward::experts() const { return impl_->experts; }

ForwardTimes Forward::times() const {
    ForwardTimes t = times_;
    t.host_syncs += impl_->experts.times().host_syncs + impl_->gather.syncs() - gather_syncs0_;
    return t;
}

void Forward::reset_times() {
    times_ = ForwardTimes{};
    layer_times_.assign(layer_times_.size(), ForwardLayerTimes{});
    gather_syncs0_ = impl_->gather.syncs();
    impl_->experts.reset_times();
}

void Forward::set_timing(bool on) {
    impl_->cfg.timing = on;
    impl_->experts.set_timing(on);
}

void Forward::set_pool_spin_us(int us) { impl_->experts.set_pool_spin_us(us); }

void Forward::reset() {
    impl_->kv.reset();
    pos_ = 0;
    history_.clear();
}

void Forward::save_checkpoint(int slot) {
    Impl& I = *impl_;
    I.kv.save_checkpoint(slot, I.stream);
    slot_history_[(size_t) slot] = history_;
}

void Forward::restore_checkpoint(int slot) {
    Impl& I = *impl_;
    I.kv.restore_checkpoint(slot, I.stream);
    pos_ = I.kv.checkpoint_position(slot);
    history_ = slot_history_[(size_t) slot];
    if ((int64_t) history_.size() != pos_)
        failf("restore_checkpoint: checkpoint %d at position %lld holds %zu tokens of history", slot, (long long) pos_,
              history_.size());
}

int Forward::checkpoints() const { return impl_->kv.checkpoints(); }
KvCache::CheckpointTimes Forward::checkpoint_times(int slot) const { return impl_->kv.checkpoint_times(slot); }
size_t Forward::checkpoint_slot_bytes() const { return impl_->kv.checkpoint_slot_bytes(); }
size_t Forward::checkpoint_device_bytes() const { return impl_->kv.checkpoint_device_bytes(); }

void Forward::set_forced(const SelectionDump* forced) {
    if (!impl_->cfg.forced || !forced) failf("set_forced: init had no forced selection, or the new one is null");
    impl_->cfg.forced = forced;
}

namespace {

// the device ms per replay of a graph launched reps times on st after one warm-up launch; after each replay `each`
template <typename F> double replay_ms(cudaGraphExec_t g, cudaStream_t st, int reps, F&& each) {
    cudaEvent_t a = nullptr, b = nullptr;
    ck(cudaEventCreate(&a), "cudaEventCreate");
    ck(cudaEventCreate(&b), "cudaEventCreate");
    ck(cudaGraphLaunch(g, st), "cudaGraphLaunch (warm-up)");
    ck(cudaStreamSynchronize(st), "the warm-up");
    double ms = 0;
    for (int r = 0; r < reps; ++r) {
        ck(cudaEventRecord(a, st), "cudaEventRecord");
        ck(cudaGraphLaunch(g, st), "cudaGraphLaunch");
        ck(cudaEventRecord(b, st), "cudaEventRecord");
        ck(cudaEventSynchronize(b), "cudaEventSynchronize");
        float t = 0;
        ck(cudaEventElapsedTime(&t, a, b), "cudaEventElapsedTime");
        ms += t;
        each();
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / reps;
}

template <typename F> cudaGraphExec_t capture(cudaStream_t st, F&& body, const char* what) {
    cudaGraph_t gr = nullptr;
    cudaGraphExec_t ex = nullptr;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), what);
    body();
    ck(cudaStreamEndCapture(st, &gr), what);
    ck(cudaGraphInstantiate(&ex, gr, 0), what);
    ck(cudaGraphDestroy(gr), what);
    return ex;
}

}  // namespace

ForwardAttnProfile Forward::profile_decode_attention(int reps) {
    Impl& I = *impl_;
    const Model& m = *I.m;
    if (I.cfg.forced) failf("the attention profile runs without a forced selection");
    if (reps < 1) failf("the attention profile: %d replays", reps);
    const int64_t p = pos_;
    if (p + 1 > I.cfg.ctx) failf("the attention profile at position %lld: past the context of %lld", (long long) p, (long long) I.cfg.ctx);
    struct Book { int64_t next_pos, ids_p0, ids_T; };
    std::vector<Book> saved;
    for (int il = 0; il < m.n_layer(); ++il) {
        const KvLayerBuffers& L = I.kv.layer(il);
        saved.push_back({L.next_pos, L.ids_p0, L.ids_T});
    }
    *I.h_pos = (int32_t) p;
    ck(cudaMemcpyAsync(I.attn.scratch().pos, I.h_pos, 4, cudaMemcpyHostToDevice, I.stream), "the profile's position");
    ForwardAttnProfile out;
    out.reps = reps;
    out.split = true;
    I.attn.set_kernel_marks(true);
    for (int il = 0; il < m.n_layer(); ++il) {
        const LayerWeights& L = m.layer(il);
        I.attn.clear_kernel_marks();
        cudaGraphExec_t g = capture(I.stream, [&] {
            attn_decode(I.attn, L.role, L.attn, I.kv, I.h, p, I.a, I.cfg.readers, nullptr, true, I.hq);
        }, "the attention profile's capture");
        std::vector<double> km((size_t) kAttnKernels, 0.0);
        // the events are the graph's nodes: each timed replay records them again before it is read
        out.layer_ms.push_back(replay_ms(g, I.stream, reps, [&] {
            if (out.split && !I.attn.read_kernel_marks(km.data())) out.split = false;
        }));
        for (double& v : km) v /= reps;
        out.kernel_ms.push_back(km);
        ck(cudaGraphExecDestroy(g), "cudaGraphExecDestroy");
    }
    I.attn.clear_kernel_marks();
    I.attn.set_kernel_marks(false);
    for (int il = 0; il < m.n_layer(); ++il) {
        KvLayerBuffers& L = I.kv.layer(il);
        L.next_pos = saved[(size_t) il].next_pos;
        L.ids_p0 = saved[(size_t) il].ids_p0;
        L.ids_T = saved[(size_t) il].ids_T;
    }
    return out;
}

double Forward::profile_head(int reps) {
    Impl& I = *impl_;
    if (reps < 1) failf("the head profile: %d replays", reps);
    cudaGraphExec_t g = capture(I.stream, [&] {
        dense_head(I.dense, I.x, I.pre_prev, 1, I.m->output_norm(), I.m->output(), I.logits);
    }, "the head profile's capture");
    const double ms = replay_ms(g, I.stream, reps, [] {});
    ck(cudaGraphExecDestroy(g), "cudaGraphExecDestroy");
    return ms;
}

ForwardHcProfile Forward::profile_decode_hc(int reps) {
    Impl& I = *impl_;
    const Model& m = *I.m;
    if (reps < 1) failf("the hyper-connection profile: %d replays", reps);
    ForwardHcProfile out;
    for (int il = 0; il < m.n_layer(); ++il) {
        const LayerWeights& L = m.layer(il);
        // the token graph folds the previous FFN's expansion into this attention step except at layer 0 and an
        // engram layer (token_body)
        const bool fold = il > 0 && L.engram < 0;
        auto attn_step = [&] {
            hc_decode_step(I.dense, L.hc_attn, fold ? I.ffn : nullptr, I.x, I.pre_buf[0], L.attn_norm, I.pre_a, I.post,
                           I.comb, I.h, I.hq);
        };
        auto ffn_step = [&] {
            hc_decode_step(I.dense, L.hc_ffn, I.a, I.x, I.pre_a, L.ffn_norm, I.pre_buf[1], I.post, I.comb, I.h, I.hq);
        };
        cudaGraphExec_t g = capture(I.stream, [&] {
            attn_step();
            ffn_step();
        }, "the hyper-connection profile's capture");
        out.layer_ms.push_back(replay_ms(g, I.stream, reps, [] {}));
        ck(cudaGraphExecDestroy(g), "cudaGraphExecDestroy");
        // each step alone with its phase marks, the split summed over the replays
        I.dense.set_step_marks(true);
        for (int k = 0; k < 2; ++k) {
            cudaGraphExec_t gm = capture(I.stream, [&] { k == 0 ? attn_step() : ffn_step(); }, "the marked step's capture");
            replay_ms(gm, I.stream, reps, [&] {
                const HcStepSplit s = hc_step_split(I.dense);
                HcStepSplit& t = out.split;
                t.rms_us += s.rms_us;
                t.q8_us += s.q8_us;
                t.mixes_us += s.mixes_us;
                t.ticket_us += s.ticket_us;
                t.norm_us += s.norm_us;
                t.end_us += s.end_us;
                ++out.marked;
            });
            ck(cudaGraphExecDestroy(gm), "cudaGraphExecDestroy");
        }
        I.dense.set_step_marks(false);
    }
    if (out.marked > 0) {
        HcStepSplit& t = out.split;
        const double n = (double) out.marked;
        for (double* v : {&t.rms_us, &t.q8_us, &t.mixes_us, &t.ticket_us, &t.norm_us, &t.end_us}) *v /= n;
    }
    return out;
}

void Forward::prefill(const int32_t* tokens, int64_t n, std::vector<float>& logits) {
    if (n < 1) failf("an empty prompt");
    for (int64_t i = 0; i < n; i += impl_->cfg.chunk) run(tokens + i, std::min(impl_->cfg.chunk, n - i), logits);
    prompt_.prefill(read_beside_);
}

void Forward::prompt_read() {
    const HotPromptEnd e = prompt_.end();
    if (e == HotPromptEnd::Merge) impl_->experts.merge_prompt();
    else if (e == HotPromptEnd::Done) impl_->experts.prompt_done();
}

void Forward::decode(int32_t token, std::vector<float>& logits) {
    const ForwardConfig& c = impl_->cfg;
    prompt_read();
    if (c.graphs && !c.timing && !c.forced_moe_input) run_graph(token, logits);
    else run(&token, 0, logits);
    const auto t0 = std::chrono::steady_clock::now();
    impl_->experts.decode_boundary();
    times_.boundary_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void Forward::prompt_deterministic(bool on) { impl_->experts.prompt_deterministic(on); }

void Forward::begin_request(HotPolicy p, bool reset, bool resume, bool deterministic) {
    impl_->experts.begin_request(p, reset, resume, deterministic);
    prompt_.begin_request();
}
void Forward::end_request() { impl_->experts.end_request(); }
void Forward::set_hot(const HotConfig& c) { impl_->experts.set_hot(c); }
void Forward::set_trace(bool on) { impl_->experts.set_trace(on); }

// T >= 1: a prefill chunk; T == 0: one decode token, every op launched from here (timing, --no-graphs)
void Forward::run(const int32_t* tokens, int64_t T_in, std::vector<float>& logits) {
    Impl& I = *impl_;
    const Model& m = *I.m;
    const DenseDims& dd = m.dense_dims();
    const bool decode = T_in == 0;
    const int64_t T = decode ? 1 : T_in, p0 = pos_, E = dd.n_embd;
    if (p0 + T > I.cfg.ctx)
        failf("positions %lld .. %lld are past the context of %lld (--ctx)", (long long) p0, (long long) (p0 + T - 1),
              (long long) I.cfg.ctx);
    for (int64_t i = 0; i < T; ++i)
        if (tokens[i] < 0 || tokens[i] >= m.token_embd().rows)
            failf("token %d at position %lld is outside the vocabulary of %lld", tokens[i], (long long) (p0 + i),
                  (long long) m.token_embd().rows);
    const auto t_start = std::chrono::steady_clock::now();
    const int u = (int) dump_steps_.size();
    const SelectionDump* forced = I.cfg.forced;
    const int64_t ref_call = forced ? forced->call_of(p0, T) : -1;
    const int K = dd.n_expert_used, top_k = m.attn_dims().top_k;
    I.pre_prev = I.pre_buf[0];
    I.pre_f = I.pre_buf[1];
    I.experts.begin_call(T, !decode);
    // a decode token's token and engram rows through mapped memory (as the token graph's), its swaps started
    if (decode) {
        *I.h_token = tokens[0];
        const MappedCopy c{I.tokens, I.h_token_m, 1};
        copy_from_mapped(&c, 1, I.stream);
        I.experts.decode_started();
    } else {
        ck(cudaMemcpyAsync(I.tokens, tokens, (size_t) T * 4, cudaMemcpyHostToDevice, I.stream), "the tokens");
    }
    I.timed(times_.embed_ms, nullptr, [&] { dense_embed(I.dense, m.token_embd(), I.tokens, T, I.x, I.pre_prev); });

    // the tokens before the chunk the engram's n-grams look back to
    const EngramHash& eh = m.host().engram;
    const int n_prev = (int) std::min<int64_t>(p0, eh.n_gram - 1);
    const int32_t* prev = history_.data() + (history_.size() - (size_t) n_prev);
    char name[96];
    // token_body's fused steps and deferred expansion (not with the dumps, which read each layer's expanded x): a
    // decode token's hc_decode_step, a prefill chunk's hc_prefill_step
    const bool defer = I.cfg.dump_dir.empty();
    void* const hq = decode ? I.hq : nullptr;
    bool pending = false;
    for (int il = 0; il < m.n_layer(); ++il) {
        const LayerWeights& L = m.layer(il);
        ForwardLayerTimes& lt = layer_times_[(size_t) il];
        if (L.engram >= 0) {
            if (pending) I.timed(times_.hc_ms, &lt.dense_ms, [&] { hc_expand(I.dense, I.ffn, I.x, I.post, I.comb, T, I.x); });
            pending = false;
            I.timed(times_.engram_ms, &lt.dense_ms, [&] {
                I.rows.resize((size_t) (T * eh.cols()));
                engram_rows(eh, L.engram, tokens, T, prev, n_prev, I.rows.data());
                const void* rows = nullptr;
                if (decode) {
                    const size_t e = (size_t) L.engram;
                    engram_copy_rows(m.table(L.engram), I.rows.data(), eh.cols(), I.eng_stage[e]);
                    const MappedCopy c{I.eng_dev[e], I.eng_stage_m[e], (int64_t) (I.eng_bytes / 4)};
                    copy_from_mapped(&c, 1, I.stream);
                    rows = I.eng_dev[e];
                } else {
                    rows = I.gather.gather(m.table(L.engram), I.rows.data(), T * eh.cols(), I.stream);
                }
                engram_forward(I.dense, L.eng, rows, T, I.x);
            });
        }
        I.timed(times_.hc_ms, &lt.dense_ms, [&] {
            if (decode) {
                hc_decode_step(I.dense, L.hc_attn, pending ? I.ffn : nullptr, I.x, I.pre_prev, L.attn_norm, I.pre_a, I.post,
                               I.comb, I.h, hq);
                return;
            }
            hc_prefill_step(I.dense, L.hc_attn, pending ? I.ffn : nullptr, I.x, I.pre_prev, L.attn_norm, T, I.pre_a, I.post,
                            I.comb, I.h);
        });
        // a pageable source: cudaMemcpyAsync returns once the bytes are staged, so the host vector may be refilled
        const int32_t* forced_rows = nullptr;
        if (forced && L.role.index_source) {
            forced->index_rows(ref_call, il, p0, T, L.role.ratio, top_k, (p0 + T) / L.role.ratio, I.forced_rows_host);
            ck(cudaMemcpyAsync(I.forced_rows, I.forced_rows_host.data(), I.forced_rows_host.size() * 4,
                               cudaMemcpyHostToDevice, I.stream), "the forced index rows");
            forced_rows = I.forced_rows;
        }
        I.timed(times_.attention_ms, &lt.attention_ms, [&] {
            if (decode) attn_decode(I.attn, L.role, L.attn, I.kv, I.h, p0, I.a, I.cfg.readers, forced_rows, false, hq);
            else attn_prefill(I.attn, L.role, L.attn, I.kv, I.h, T, p0, I.a, I.cfg.readers, forced_rows);
        });
        const int32_t* forced_ids = nullptr;
        if (forced || I.cfg.routing) {
            if (forced) forced->router(ref_call, il, p0, T, K, dd.n_expert, I.forced_ids_host);
            else I.cfg.routing(il, p0, T, I.forced_ids_host);
            if ((int64_t) I.forced_ids_host.size() != T * K)
                failf("layer %d: %zu router ids for %lld tokens of %d", il, I.forced_ids_host.size(), (long long) T, K);
            ck(cudaMemcpyAsync(I.forced_ids, I.forced_ids_host.data(), I.forced_ids_host.size() * 4,
                               cudaMemcpyHostToDevice, I.stream), "the forced router ids");
            forced_ids = I.forced_ids;
        }
        I.timed(times_.hc_ms, &lt.dense_ms, [&] {
            if (decode) {
                hc_decode_step(I.dense, L.hc_ffn, I.a, I.x, I.pre_a, L.ffn_norm, I.pre_f, I.post, I.comb, I.h, hq);
                return;
            }
            hc_prefill_step(I.dense, L.hc_ffn, I.a, I.x, I.pre_a, L.ffn_norm, T, I.pre_f, I.post, I.comb, I.h);
        });
        // the reference's MoE input in place of h; hc_decode_step's q8_1 of the engine's own h is then
        // stale, so the FFN's matmuls quantize the replaced h themselves
        void* ffn_q = hq;
        if (I.cfg.forced_moe_input) {
            forced->moe_input(ref_call, il, p0, T, E, I.moe_in_host);
            ck(cudaMemcpyAsync(I.h, I.moe_in_host.data(), I.moe_in_host.size() * 4, cudaMemcpyHostToDevice, I.stream),
               "the reference's MoE input");
            ffn_q = nullptr;
        }
        I.timed(times_.router_ms, &lt.dense_ms,
                [&] { dense_router(I.dense, L.router, I.h, T, I.ids, I.w, RouterMatmul::Auto, forced_ids); });
        // a prefill's routing on its way to the host before the shared expert, which the host's plan overlaps
        if (!decode) I.experts.prefill_fetch(il, I.h, I.ids, T);
        I.timed(times_.shared_ms, &lt.dense_ms, [&] { dense_shared_expert(I.dense, L.shared, I.h, T, I.sh, ffn_q); });
        I.timed(times_.experts_ms, &lt.experts_ms, [&] {
            I.experts.forward(il, I.h, I.ids, I.w, I.moe_out ? nullptr : I.sh, T, !decode, I.ffn, ffn_q);
            if (!I.moe_out) return;
            // the routed sum kept for the dump, then the shared expert's add (as token_body)
            ck(cudaMemcpyAsync(I.moe_out, I.ffn, (size_t) (T * E) * 4, cudaMemcpyDeviceToDevice, I.stream), "a dump's copy");
            expert_add_shared(I.ffn, I.sh, T * E, I.stream);
        });
        pending = defer;
        if (!defer) I.timed(times_.hc_ms, &lt.dense_ms, [&] { hc_expand(I.dense, I.ffn, I.x, I.post, I.comb, T, I.x); });
        std::swap(I.pre_prev, I.pre_f);
        if (!I.cfg.dump_dir.empty()) dump_layer(il, u, p0, T, false);
    }
    if (pending) I.timed(times_.hc_ms, nullptr, [&] { hc_expand(I.dense, I.ffn, I.x, I.post, I.comb, T, I.x); });
    I.timed(times_.head_ms, nullptr, [&] {
        dense_head(I.dense, I.x + (T - 1) * kHc * E, I.pre_prev + (T - 1) * kHc, 1, m.output_norm(), m.output(), I.logits);
    });
    logits.resize((size_t) dd.n_vocab);
    if (decode) copy_to_mapped(I.h_logits_m, I.logits, dd.n_vocab, I.stream);
    else ck(cudaMemcpyAsync(logits.data(), I.logits, logits.size() * 4, cudaMemcpyDeviceToHost, I.stream), "the logits");
    ck(cudaStreamSynchronize(I.stream), "the forward pass");
    if (decode) std::memcpy(logits.data(), I.h_logits, logits.size() * 4);
    ++times_.host_syncs;
    I.read_marks();
    I.experts.end_call();
    if (!I.cfg.dump_dir.empty()) {
        std::snprintf(name, sizeof name, "/result_output.0-L999-u%04d.bin", u);
        write_file(I.cfg.dump_dir + name, logits.data(), logits.size() * 4);
    }
    dump_steps_.push_back({p0, T});
    history_.insert(history_.end(), tokens, tokens + T);
    pos_ += T;
    times_.total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    ++times_.calls;
    times_.tokens += T;
}

// One decode token from the token graph (session.cpp session_run_token): the mapped inputs filled on the host,
// the graph captured at the first call (attn_decode's checks and bookkeeping run in the capture) or the attention's
// bookkeeping advanced without a launch, one launch, the next batch of swap copies queued, every layer's
// doorbell served in order while it runs, one wait, the logits read from mapped memory.
void Forward::run_graph(int32_t token, std::vector<float>& logits) {
    Impl& I = *impl_;
    const Model& m = *I.m;
    const DenseDims& dd = m.dense_dims();
    const int64_t p0 = pos_;
    if (p0 + 1 > I.cfg.ctx) failf("position %lld is past the context of %lld (--ctx)", (long long) p0, (long long) I.cfg.ctx);
    if (token < 0 || token >= m.token_embd().rows)
        failf("token %d at position %lld is outside the vocabulary of %lld", token, (long long) p0, (long long) m.token_embd().rows);
    const auto t_start = std::chrono::steady_clock::now();
    const int u = (int) dump_steps_.size();
    const SelectionDump* forced = I.cfg.forced;
    const int64_t ref_call = forced ? forced->call_of(p0, 1) : -1;
    const int K = dd.n_expert_used, top_k = m.attn_dims().top_k;
    *I.h_token = token;
    *I.h_pos = (int32_t) p0;
    const EngramHash& eh = m.host().engram;
    const int n_prev = (int) std::min<int64_t>(p0, eh.n_gram - 1);
    const int32_t* prev = history_.data() + (history_.size() - (size_t) n_prev);
    for (size_t e = 0; e < I.eng_stage.size(); ++e) {
        I.eng_rows.resize((size_t) eh.cols());
        engram_rows(eh, (int) e, &token, 1, prev, n_prev, I.eng_rows.data());
        engram_copy_rows(m.table((int) e), I.eng_rows.data(), eh.cols(), I.eng_stage[e]);
    }
    for (int il = 0; il < m.n_layer() && (forced || I.cfg.routing); ++il) {
        if (forced) forced->router(ref_call, il, p0, 1, K, dd.n_expert, I.forced_ids_host);
        else I.cfg.routing(il, p0, 1, I.forced_ids_host);
        if ((int64_t) I.forced_ids_host.size() != K) failf("layer %d: %zu router ids for a token of %d", il, I.forced_ids_host.size(), K);
        std::copy(I.forced_ids_host.begin(), I.forced_ids_host.end(), I.h_ids + (size_t) il * K);
        const AttnLayer& role = m.layer(il).role;
        if (forced && role.index_source) {
            forced->index_rows(ref_call, il, p0, 1, role.ratio, top_k, (p0 + 1) / role.ratio, I.forced_rows_host);
            std::copy(I.forced_rows_host.begin(), I.forced_rows_host.end(), I.h_rows + (size_t) il * top_k);
        }
    }
    I.experts.begin_call(1, false);
    if (!I.g_token) {
        cudaGraph_t gr = nullptr;
        ck(cudaStreamBeginCapture(I.stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture (token)");
        I.token_body(I.kv, p0);
        ck(cudaStreamEndCapture(I.stream, &gr), "cudaStreamEndCapture (token)");
        ck(cudaGraphInstantiate(&I.g_token, gr, 0), "cudaGraphInstantiate (token)");
        ck(cudaGraphDestroy(gr), "cudaGraphDestroy (token)");
    } else {
        for (int il = 0; il < m.n_layer(); ++il) {
            const AttnLayer& role = m.layer(il).role;
            attn_decode_advance(role, I.kv, p0, I.cfg.readers, forced && role.index_source);
        }
    }
    // the graph holds no event wait, so an open layered reset's rest is waited for before it (a
    // prefill before the token has waited per layer: nothing is left then)
    I.experts.wait_restore_all();
    ck(cudaEventRecord(I.graph_a, I.stream), "cudaEventRecord (token graph)");
    ck(cudaGraphLaunch(I.g_token, I.stream), "cudaGraphLaunch (token)");
    ck(cudaEventRecord(I.graph_b, I.stream), "cudaEventRecord (token graph)");
    ++times_.graph_launches;
    auto t_lap = t_start;
    auto lap = [&t_lap] {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - t_lap).count();
        t_lap = now;
        return ms;
    };
    times_.launch_ms += lap();
    // the next swaps' copies queued while the graph runs (their host time is ExpertTimes::swap_queue_ms), then
    // the doorbells
    I.experts.decode_started();
    lap();
    for (int il = 0; il < m.n_layer(); ++il) I.experts.decode_serve(il, 1);
    times_.serve_ms += lap();
    ck(cudaStreamSynchronize(I.stream), "the token graph");
    logits.resize((size_t) dd.n_vocab);
    std::memcpy(logits.data(), I.h_logits, logits.size() * 4);
    times_.tail_ms += lap();
    ++times_.host_syncs;
    float graph_ms = 0;
    ck(cudaEventElapsedTime(&graph_ms, I.graph_a, I.graph_b), "cudaEventElapsedTime (token graph)");
    times_.graph_ms += graph_ms;
    I.experts.end_call();
    if (!I.cfg.dump_dir.empty()) {
        for (int il = 0; il < m.n_layer(); ++il) dump_layer(il, u, p0, 1, true);
        char name[96];
        std::snprintf(name, sizeof name, "/result_output.0-L999-u%04d.bin", u);
        write_file(I.cfg.dump_dir + name, logits.data(), logits.size() * 4);
    }
    dump_steps_.push_back({p0, 1});
    history_.push_back(token);
    ++pos_;
    times_.total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    ++times_.calls;
    ++times_.tokens;
}

int Forward::batch_slots() const { return impl_->cfg.batch; }
int64_t Forward::batch_ctx() const { return impl_->cfg.batch_ctx; }

int64_t Forward::batch_position(int b) const {
    if (b < 0 || b >= impl_->cfg.batch) failf("batch slot %d of %d", b, impl_->cfg.batch);
    return batch_pos_[(size_t) b];
}

size_t Forward::batch_slot_bytes() const { return impl_->slot_kv.empty() ? 0 : impl_->slot_kv[0]->bytes().total(); }

void Forward::batch_admit(int b) {
    Impl& I = *impl_;
    if (b < 0 || b >= I.cfg.batch) failf("batch_admit: slot %d of %d", b, I.cfg.batch);
    if (pos_ > I.cfg.batch_ctx)
        failf("batch_admit: the live sequence is at %lld, a batch slot holds %lld (--batch-ctx)", (long long) pos_,
              (long long) I.cfg.batch_ctx);
    I.slot_kv[(size_t) b]->copy_state_from(I.kv, I.stream);
    batch_pos_[(size_t) b] = pos_;
    batch_history_[(size_t) b] = history_;
}

// one step of the batch (forward.hpp): the rows' tokens, positions and engram rows into mapped memory, the graph
// of B rows captured again when its slots differ from the last capture's (cudaGraphExecUpdate into the instantiated one
// when it takes the new launches, else a new instance) or the attention's bookkeeping advanced, one launch, the next
// batch of swap copies queued, every layer's doorbell served, one wait, the rows' logits read from mapped memory.
void Forward::decode_batch(const int* slots, const int32_t* tokens, int B, std::vector<std::vector<float>>& logits) {
    Impl& I = *impl_;
    const Model& m = *I.m;
    const DenseDims& dd = m.dense_dims();
    if (B < 1 || B > I.cfg.batch) failf("decode_batch: %d rows of %d slots", B, I.cfg.batch);
    if (!prompt_held_) prompt_read();   // a prompt read between steps ends at its last chunk
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<KvCache*> kvs((size_t) B);
    std::vector<int64_t> p((size_t) B);
    const EngramHash& eh = m.host().engram;
    for (int i = 0; i < B; ++i) {
        const int b = slots[i];
        if (b < 0 || b >= I.cfg.batch || std::count(slots, slots + i, b))
            failf("decode_batch: row %d takes slot %d (of %d, each once)", i, b, I.cfg.batch);
        const int32_t t = tokens[i];
        if (t < 0 || t >= m.token_embd().rows)
            failf("decode_batch: slot %d's token %d is outside the vocabulary of %lld", b, t, (long long) m.token_embd().rows);
        p[(size_t) i] = batch_pos_[(size_t) b];
        if (p[(size_t) i] + 1 > I.cfg.batch_ctx)
            failf("decode_batch: slot %d at position %lld, it holds %lld (--batch-ctx)", b, (long long) p[(size_t) i],
                  (long long) I.cfg.batch_ctx);
        kvs[(size_t) i] = I.slot_kv[(size_t) b].get();
        // the row's token and position (B = 1 takes the token graph's mapped inputs), its engram rows
        if (B == 1) {
            *I.h_token = t;
            *I.h_pos = (int32_t) p[0];
        }
        I.h_btok[i] = t;
        I.h_bpos[i] = (int32_t) p[(size_t) i];
        const std::vector<int32_t>& hist = batch_history_[(size_t) b];
        const int n_prev = (int) std::min<int64_t>(p[(size_t) i], eh.n_gram - 1);
        for (size_t e = 0; e < I.eng_stage.size(); ++e) {
            I.eng_rows.resize((size_t) eh.cols());
            engram_rows(eh, (int) e, &t, 1, hist.data() + (hist.size() - (size_t) n_prev), n_prev, I.eng_rows.data());
            engram_copy_rows(m.table((int) e), I.eng_rows.data(), eh.cols(), I.eng_stage[e] + (size_t) i * I.eng_bytes);
        }
    }
    I.experts.begin_call(B, false);
    const std::vector<int> want(slots, slots + B);
    cudaGraphExec_t& g = I.g_batch[B];
    if (!g || I.g_batch_slots[B] != want) {
        // the capture runs the attention's host checks and bookkeeping (attn_decode's, attn_decode_batch's)
        cudaGraph_t gr = nullptr;
        ck(cudaStreamBeginCapture(I.stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture (batch)");
        if (B == 1) I.token_body(*kvs[0], p[0]);
        else I.batch_body(B, kvs.data(), p.data());
        ck(cudaStreamEndCapture(I.stream, &gr), "cudaStreamEndCapture (batch)");
        cudaGraphExecUpdateResultInfo info{};
        if (g && cudaGraphExecUpdate(g, gr, &info) == cudaSuccess) {
            ++batch_updates_;
        } else {
            cudaGetLastError();   // a refused update leaves an error to clear
            if (g) ck(cudaGraphExecDestroy(g), "cudaGraphExecDestroy (batch)");
            g = nullptr;
            ck(cudaGraphInstantiate(&g, gr, 0), "cudaGraphInstantiate (batch)");
        }
        ck(cudaGraphDestroy(gr), "cudaGraphDestroy (batch)");
        I.g_batch_slots[B] = want;
        ++batch_captures_;
    } else {
        for (int il = 0; il < m.n_layer(); ++il) {
            const AttnLayer& role = m.layer(il).role;
            if (B == 1) attn_decode_advance(role, *kvs[0], p[0], I.cfg.readers, false);
            else attn_decode_batch_advance(role, kvs.data(), B, p.data(), I.cfg.readers);
        }
    }
    I.experts.wait_restore_all();
    ck(cudaEventRecord(I.graph_a, I.stream), "cudaEventRecord (batch graph)");
    ck(cudaGraphLaunch(g, I.stream), "cudaGraphLaunch (batch)");
    ck(cudaEventRecord(I.graph_b, I.stream), "cudaEventRecord (batch graph)");
    ++times_.graph_launches;
    auto t_lap = t_start;
    auto lap = [&t_lap] {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - t_lap).count();
        t_lap = now;
        return ms;
    };
    times_.launch_ms += lap();
    I.experts.decode_started();
    lap();
    for (int il = 0; il < m.n_layer(); ++il) I.experts.decode_serve(il, B);
    times_.serve_ms += lap();
    ck(cudaStreamSynchronize(I.stream), "the batch graph");
    logits.resize((size_t) B);
    const float* src = B == 1 ? I.h_logits : I.h_blogits;
    for (int i = 0; i < B; ++i) logits[(size_t) i].assign(src + (size_t) i * dd.n_vocab, src + (size_t) (i + 1) * dd.n_vocab);
    times_.tail_ms += lap();
    ++times_.host_syncs;
    float graph_ms = 0;
    ck(cudaEventElapsedTime(&graph_ms, I.graph_a, I.graph_b), "cudaEventElapsedTime (batch graph)");
    times_.graph_ms += graph_ms;
    I.experts.end_call();
    for (int i = 0; i < B; ++i) {
        ++batch_pos_[(size_t) slots[i]];
        batch_history_[(size_t) slots[i]].push_back(tokens[i]);
    }
    const auto t0 = std::chrono::steady_clock::now();
    // no deterministic wait for a step of several rows (their bits do not follow the placement anyway)
    I.experts.decode_boundary(B > 1);
    times_.boundary_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    times_.total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    ++times_.calls;
    times_.tokens += B;
}

// the dumps of layer il after call u: l_last, the router's selection and biased probs (the fork's ffn_moe_topk [T][K]
// and ffn_moe_probs_biased [T][NE]), an index source's selection and scores (idx_top_k [T][top_k] with -1 padding,
// idx_score [T][n_k] with -inf past each position's visible rows, as the fork's masked scores)
void Forward::dump_layer(int il, int u, int64_t p0, int64_t T, bool snapshot) {
    Impl& I = *impl_;
    const Model& m = *I.m;
    const DenseDims& dd = m.dense_dims();
    const AttnDims& ad = m.attn_dims();
    const LayerWeights& L = m.layer(il);
    const int64_t E = dd.n_embd, K = dd.n_expert_used, NE = dd.n_expert;
    // where the values are: the live buffers right after the layer, or the token graph's copies of the layer
    const AttnScratch& s = I.attn.scratch();
    const size_t l = (size_t) il, IQ = (size_t) ad.idx_heads * ad.idx_dim;
    const float* x = snapshot ? I.snap_x + l * kHc * E : I.x;
    const int32_t* ids = snapshot ? I.snap_ids + l * K : I.ids;
    const float* probs = snapshot ? I.snap_probs + l * NE : I.dense.scratch().probs;
    const int32_t* idx = snapshot ? I.snap_idx + l * ad.top_k : I.kv.layer(il).ids;
    const float* iq = snapshot ? I.snap_iq + l * IQ : s.iq;
    const float* iw = snapshot ? I.snap_iw + l * ad.idx_heads : s.iw;
    const float* cur = snapshot ? I.snap_cur + l * E : I.h;
    const float* w = snapshot ? I.snap_w + l * K : I.w;
    const float* moe = snapshot ? I.snap_out + l * E : I.moe_out;
    char name[96];
    auto put = [&](const char* op, const void* data, size_t bytes) {
        std::snprintf(name, sizeof name, "/%s.0-L%03d-u%04d.bin", op, il, u);
        write_file(I.cfg.dump_dir + name, data, bytes);
    };
    auto down = [&](void* dst, const void* src, size_t bytes, const char* what) {
        ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, I.stream), what);
        ck(cudaStreamSynchronize(I.stream), what);
    };
    const bool values = I.cfg.dump_values;
    if (values) {
        I.host_x.resize((size_t) (T * kHc * E));
        down(I.host_x.data(), x, I.host_x.size() * 4, "l_last");
        put("l_last", I.host_x.data(), I.host_x.size() * 4);
        // the real-model oracle's kinds (SPEC.md "Real-model oracle mapping"): cur, weights and out
        I.host_x.resize((size_t) (T * E));
        down(I.host_x.data(), cur, I.host_x.size() * 4, "ffn_norm");
        put("ffn_norm", I.host_x.data(), I.host_x.size() * 4);
        down(I.host_x.data(), moe, I.host_x.size() * 4, "ffn_moe_out");
        put("ffn_moe_out", I.host_x.data(), I.host_x.size() * 4);
        I.host_x.resize((size_t) (T * K));
        down(I.host_x.data(), w, I.host_x.size() * 4, "ffn_moe_weights_scaled");
        put("ffn_moe_weights_scaled", I.host_x.data(), I.host_x.size() * 4);
    }
    I.host_i32.resize((size_t) (T * K));
    down(I.host_i32.data(), ids, I.host_i32.size() * 4, "ffn_moe_topk");
    put("ffn_moe_topk", I.host_i32.data(), I.host_i32.size() * 4);
    if (!values && !L.role.index_source) return;
    if (!values) {
        I.host_i32.resize((size_t) (T * ad.top_k));
        down(I.host_i32.data(), idx, I.host_i32.size() * 4, "idx_top_k");
        put("idx_top_k", I.host_i32.data(), I.host_i32.size() * 4);
        return;
    }
    // probs + bias rounded once, as router_select_kernel's selection key
    I.host_f32.resize((size_t) (T * NE));
    I.host_bias.resize((size_t) NE);
    down(I.host_f32.data(), probs, I.host_f32.size() * 4, "ffn_moe_probs");
    down(I.host_bias.data(), L.router.bias, I.host_bias.size() * 4, "exp_probs_b");
    for (int64_t t = 0; t < T; ++t)
        for (int64_t e = 0; e < NE; ++e) I.host_f32[(size_t) (t * NE + e)] += I.host_bias[(size_t) e];
    put("ffn_moe_probs_biased", I.host_f32.data(), I.host_f32.size() * 4);
    if (!L.role.index_source) return;
    const int r = L.role.ratio;
    const int64_t n_k = (p0 + T) / r;
    I.host_i32.resize((size_t) (T * ad.top_k));
    down(I.host_i32.data(), idx, I.host_i32.size() * 4, "idx_top_k");
    put("idx_top_k", I.host_i32.data(), I.host_i32.size() * 4);
    I.host_f32.assign((size_t) (T * n_k), -INFINITY);
    if (n_k > 0) {
        if (n_k > I.dump_scores_ld) failf("layer %d: %lld index keys, the dump's score rows hold %lld", il, (long long) n_k, (long long) I.dump_scores_ld);
        // this layer's queries and weights (the attention's scratch still holds them, or the graph's copy), the
        // chunk's positions (the scratch's: every layer of a call has the same)
        const IndexerDims id{ad.idx_heads, ad.top_k, r};
        indexer_scores(iq, iw, s.pos, T, I.kv.layer(L.role.key_src).keys, n_k, id, I.dump_scores, I.dump_scores_ld,
                       IndexerScorer::Auto, I.stream);
        std::vector<float> rows((size_t) (T * I.dump_scores_ld));
        down(rows.data(), I.dump_scores, rows.size() * 4, "idx_score");
        for (int64_t t = 0; t < T; ++t) {
            const int64_t vis = indexer_visible(p0 + t, r, n_k);
            std::copy(rows.begin() + t * I.dump_scores_ld, rows.begin() + t * I.dump_scores_ld + vis, I.host_f32.begin() + t * n_k);
        }
    }
    put("idx_score", I.host_f32.data(), I.host_f32.size() * 4);
}

void print_batch_cost(std::FILE* out, const Model& m, const ForwardConfig& cfg) {
    if (cfg.batch < 1) return;
    const KvBytes b = kv_cache_bytes(m.attn_dims(), m.roles(), KvCacheConfig{cfg.batch_ctx, 1});
    std::fprintf(out, "batch slots: %d live sequences of %.1f MiB device KV each (a slot of %lld positions: window %.1f, rows "
                 "%.1f, keys %.1f MiB), %.1f MiB in all beside the live sequence's; batched decode of up to %d rows a step\n",
                 cfg.batch, (double) b.total() / kMiB, (long long) cfg.batch_ctx, (double) b.window / kMiB,
                 (double) b.rows / kMiB, (double) b.keys / kMiB, (double) cfg.batch * b.total() / kMiB, cfg.batch);
}

void print_host_homes_cost(std::FILE* out, const ExpertEngine& experts) {
    if (!experts.hot().host_homes) return;
    const double mib = (double) experts.host_home_bytes() / kMiB;
    // a limit names the homes held of the slab's experts
    char flag[96];
    if (experts.hot().host_homes_max < 0) std::snprintf(flag, sizeof flag, "--hot-homes all");
    else std::snprintf(flag, sizeof flag, "--hot-homes %lld: %lld of %lld slab experts", (long long) experts.hot().host_homes_max,
                       (long long) experts.host_homes_held(), (long long) experts.vram_experts());
    // the homes are the engine's, held once: a further conversation adds its checkpoint slot only
    std::fprintf(out, "fixed host RAM beside the checkpoints: the hot set's host homes, %.1f MiB pinned host RAM (%s), "
                 "held once for the engine whatever the number of conversations (not part of a conversation's cost)\n",
                 mib, flag);
}

void Forward::print_memory(std::FILE* out) const {
    const Impl& I = *impl_;
    const KvBytes& kb = I.kv.bytes();
    std::fprintf(out, "device memory: model %.1f MiB, KV cache %.1f MiB (window %.1f, rows %.1f, keys %.1f) at ctx %lld, "
                 "chunk %lld: qmatmul context %.1f MiB (the stream's one), attention workspace %.1f MiB, dense workspace "
                 "%.1f MiB, engram gather %.1f MiB, forward buffers %.1f MiB; pinned engram tables %.1f MiB\n",
                 (double) I.m->device_bytes() / kMiB, (double) kb.total() / kMiB, (double) kb.window / kMiB,
                 (double) kb.rows / kMiB, (double) kb.keys / kMiB, (long long) I.cfg.ctx, (long long) I.cfg.chunk,
                 (double) I.qmm.device_bytes() / kMiB, (double) I.attn.workspace_bytes() / kMiB,
                 (double) I.dense.workspace_bytes() / kMiB, (double) I.gather.device_bytes() / kMiB,
                 (double) I.buffer_bytes / kMiB, (double) I.m->pinned_bytes() / kMiB);
    const int n = I.kv.checkpoints();
    std::fprintf(out, "conversation checkpoints: %d slots of %.1f MiB pinned host RAM each (a checkpoint of %lld positions), "
                 "%.1f MiB in all; on the device the live KV and the in-place carry snapshot (%.1f KiB)\n", n,
                 (double) I.kv.checkpoint_slot_bytes() / kMiB, (long long) I.cfg.ctx,
                 (double) n * I.kv.checkpoint_slot_bytes() / kMiB, (double) I.kv.checkpoint_device_bytes() / 1024.0);
    print_host_homes_cost(out, I.experts);
    print_batch_cost(out, *I.m, I.cfg);
    I.experts.print_summary(out);
}

void forward_dry_run(const Model& m, const ExpertEngine& experts, const ForwardConfig& cfg, const int32_t* tokens,
                     int64_t n_prompt, int64_t n_predict, std::FILE* out) {
    const AttnDims& ad = m.attn_dims();
    const EngramHash& eh = m.host().engram;
    if (n_prompt < 1) failf("dry run: an empty prompt");
    check_batch_config(cfg);
    if (cfg.ctx < n_prompt + n_predict)
        failf("dry run: %lld prompt tokens and %lld steps need %lld positions, --ctx is %lld", (long long) n_prompt,
              (long long) n_predict, (long long) (n_prompt + n_predict), (long long) cfg.ctx);
    const int64_t chunk = std::min(cfg.chunk, cfg.ctx);
    if (chunk < 1 || chunk > 65535) failf("dry run: chunk %lld (1..65535)", (long long) chunk);
    for (int64_t i = 0; i < n_prompt; ++i)
        if (tokens[i] < 0 || tokens[i] >= m.token_embd().rows)
            failf("dry run: token %d at position %lld is outside the vocabulary of %lld", tokens[i], (long long) i,
                  (long long) m.token_embd().rows);
    const int64_t end = n_prompt + n_predict;
    std::fprintf(out, "dry run: %d layers, %lld prompt tokens in %lld chunk(s) of up to %lld, %lld decode steps, positions "
                 "0 .. %lld of --ctx %lld, readers %s\n",
                 m.n_layer(), (long long) n_prompt, (long long) ((n_prompt + chunk - 1) / chunk), (long long) chunk,
                 (long long) n_predict, (long long) (end - 1), (long long) cfg.ctx,
                 cfg.readers == ReaderMode::Dense ? "dense" : "reuse");
    // each layer's attention at the run's last position: the rows it attends besides the window
    for (int il = 0; il < m.n_layer(); ++il) {
        const AttnLayer& l = m.roles()[(size_t) il];
        const int64_t visible = l.ratio ? end / l.ratio : 0;
        const bool selects = l.index_source || (l.role() == AttnRole::Reuse && cfg.readers == ReaderMode::Reuse);
        std::fprintf(out, "dry run:   layer %2d %-7s ratio %d%s: window %d, %lld compressed rows visible at the end, attends %lld\n",
                     il, role_name(l.role()), l.ratio, m.layer(il).engram >= 0 ? " + engram" : "", ad.n_swa,
                     (long long) visible, (long long) (selects ? std::min<int64_t>(visible, ad.top_k) : visible));
    }
    // the engram rows of every chunk and step, hashed as the run will, inside the tables
    std::vector<int32_t> seq(tokens, tokens + n_prompt), rows;
    seq.resize((size_t) end, tokens[n_prompt - 1]);   // the decode steps' tokens are unknown: the last one repeated
    int64_t n_rows = 0;
    for (size_t e = 0; e < eh.layer_ids.size(); ++e) {
        for (int64_t p0 = 0; p0 < end;) {
            const int64_t T = p0 < n_prompt ? std::min(chunk, n_prompt - p0) : 1;
            const int n_prev = (int) std::min<int64_t>(p0, eh.n_gram - 1);
            rows.resize((size_t) (T * eh.cols()));
            engram_rows(eh, (int) e, seq.data() + p0, T, seq.data() + p0 - n_prev, n_prev, rows.data());
            for (int32_t r : rows)
                if (r < 0 || r >= m.table((int) e).n_rows())
                    failf("dry run: engram layer %d row %d outside its table of %lld", eh.layer_ids[e], r,
                          (long long) m.table((int) e).n_rows());
            n_rows += (int64_t) rows.size();
            p0 += T;
        }
    }
    std::fprintf(out, "dry run: engram: %lld rows hashed over %zu layers, all inside their tables\n", (long long) n_rows,
                 eh.layer_ids.size());
    // the parity mode: every call's selections as the run will read them, checked against its chunks and layers
    if (cfg.forced) {
        const SelectionDump& F = *cfg.forced;
        const DenseDims& dd = m.dense_dims();
        std::vector<int32_t> sel;
        std::vector<float> moe_in;
        int64_t u = 0, n_router = 0, n_index = 0;
        for (int64_t p0 = 0; p0 < end; ++u) {
            const int64_t T = p0 < n_prompt ? std::min(chunk, n_prompt - p0) : 1;
            const int64_t ru = F.call_of(p0, T);
            for (int il = 0; il < m.n_layer(); ++il) {
                F.router(ru, il, p0, T, dd.n_expert_used, dd.n_expert, sel);
                ++n_router;
                if (cfg.forced_moe_input) F.moe_input(ru, il, p0, T, dd.n_embd, moe_in);
                const AttnLayer& l = m.roles()[(size_t) il];
                if (l.index_source) {
                    F.index_rows(ru, il, p0, T, l.ratio, ad.top_k, (p0 + T) / l.ratio, sel);
                    ++n_index;
                }
            }
            p0 += T;
        }
        std::fprintf(out, "dry run: forced selection from %s: %lld calls in its %lld, %lld router and %lld index "
                     "selections read and checked%s\n", F.dir().c_str(), (long long) u, (long long) F.n_calls(),
                     (long long) n_router, (long long) n_index,
                     cfg.forced_moe_input ? ", and as many MoE inputs (ffn_norm, finite)" : "");
    }
    if (cfg.prefill != QPrefill::DequantF16) {
        std::fprintf(out, "dry run: prefill matmuls above %lld tokens on the fork's MMQ (int8 activations)\n",
                     (long long) kMmvqMaxCols);
    }
    experts.print_summary(out);
    experts.dry_plan(std::min(chunk, n_prompt), out);
    experts.dry_restore(out);
}

}  // namespace strata::dsv41::engine
