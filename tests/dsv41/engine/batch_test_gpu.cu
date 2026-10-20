// tests/dsv41/engine/batch_test_gpu.cu - the batched
// decode's kernels on CUDA against the same kernels one row at a time, on synthetic tensors (a dev pod; no model).
//
//   dsv41_engine_batch_test --kernels
//
// A batch row's value must be the row's value decoding alone under the same placement, bit for bit, so
// each check runs a batch and the same rows alone and compares every value:
//   (g1) qmatmul_q's MMVQ at 2 to 8 columns (the batch's projections, the shared expert, the router's neighbours, the
//        head) against one column at a time, at every dense shape of the real model (qmatmul_synth.hpp, attn_output_a
//        with its 8 groups);
//   (g2) qmatmul_experts at every T of 2 to 8 tokens (one grouped kernel per T; the hits grouped by expert, one staged
//        read of an expert's rows for every pair routed to it: qmatmul.cu mmvq1_experts_grouped_kernel) against each
//        token's call of one, every hit pair's row, at the real expert dims (gate and up q2_K [2304][5120], down
//        q3_K), 12 experts of which the even ones are in the slab, routings that share experts across tokens;
//   (g3) attn_decode_batch over 4 sequences, each its own KV cache (a batch slot's: a ring of n_swa cells) with a
//        synthetic history of another length, against attn_decode on a copy of each sequence alone: three decode
//        tokens through the six roles (attention_synth.cuh: window, Full r2, Reuse r2, Full r1, Reindex r1, Reuse r1),
//        every layer's output row and, after the three tokens, every byte of every cache, at tiny-a's, tiny-b's and the
//        real dims.
// The engine end to end (the batch's graphs, the per-B capture and its update, the hc steps, the router, the head) is
// `--gpu` (batch_test.cpp), on the tiny models. PASS or FAIL lines, exit 1 on any FAIL.
#include "attention_synth.cuh"
#include "qmatmul_synth.hpp"

#include "strata/dsv41/engine/attention.hpp"
#include "strata/dsv41/engine/kv_cache.hpp"
#include "strata/dsv41/engine/qmatmul.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

using synth::ck;
using synth::download;
using synth::upload;

int64_t bits_differ(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return (int64_t) std::max(a.size(), b.size());
    int64_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], 4) != 0;
    return n;
}

std::vector<float> normal(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

// (g1)
void columns(cudaStream_t st) {
    e::QmmContext qmm;
    std::string err;
    if (!qmm.init(err, e::QPrefill::DequantF16, 20480)) {
        expect(false, "(g1) qmatmul context: " + err);
        return;
    }
    std::mt19937_64 rng(20261014u);
    for (const qsynth::Shape& sh : qsynth::real_shapes()) {
        const std::vector<uint8_t> wh = qsynth::random_blocks(sh.type, sh.rows, sh.k, 1.f / std::sqrt((float) sh.k), rng);
        void* wd = upload(wh);
        const e::QWeightView w{sh.type, sh.k, sh.rows, wd};
        const int64_t k = sh.k * sh.groups, T = e::kMmvqMaxCols;
        const std::vector<float> x = normal((size_t) (T * k), (uint32_t) sh.rows);
        float* xd = upload(x);
        const size_t row_q = e::qact_bytes(e::QActLayout::Q8_1, 1, k);
        void* qd = synth::dev_alloc<uint8_t>(e::qact_bytes(e::QActLayout::Q8_1, T, k));
        float* yd = synth::dev_alloc<float>((size_t) (T * sh.rows));
        e::QActs q;
        e::qact_quantize(xd, nullptr, k, T, k, e::QActLayout::Q8_1, qd, st, q);
        int64_t differ = 0;
        for (int64_t n = 2; n <= T; n *= 2) {
            e::qmatmul_q(qmm, w, e::QActs{e::QActLayout::Q8_1, n, k, qd}, yd, sh.rows, sh.groups, st, sh.name);
            ck(cudaStreamSynchronize(st), "(g1) batch");
            const std::vector<float> batch = download(yd, (size_t) (n * sh.rows));
            for (int64_t t = 0; t < n; ++t) {
                e::qmatmul_q(qmm, w, e::QActs{e::QActLayout::Q8_1, 1, k, static_cast<uint8_t*>(qd) + t * row_q}, yd,
                             sh.rows, sh.groups, st, sh.name);
                ck(cudaStreamSynchronize(st), "(g1) one column");
                const std::vector<float> one = download(yd, (size_t) sh.rows);
                differ += bits_differ(one, std::vector<float>(batch.begin() + t * sh.rows, batch.begin() + (t + 1) * sh.rows));
            }
        }
        char what[256];
        std::snprintf(what, sizeof what, "(g1) %s (%s, %lld x %lld, %d groups): every column of 2, 4 and 8 the one-column "
                      "value bit for bit (%lld values differ)", sh.name, e::qtype_name(sh.type), (long long) sh.rows,
                      (long long) sh.k, sh.groups, (long long) differ);
        expect(differ == 0, what);
        for (void* p : {wd, (void*) xd, qd, (void*) yd}) cudaFree(p);
    }
}

// (g2)
void experts(cudaStream_t st) {
    const int64_t E = 5120, F = 2304;
    const int NE = 12, K = 6;
    const int types[3] = {e::kQTypeQ2K, e::kQTypeQ2K, e::kQTypeQ3K};   // up, gate, down
    size_t part[3], off[3], slot_bytes = 0;
    for (int p = 0; p < 3; ++p) {
        part[p] = qsynth::random_blocks_bytes(types[p], p == 2 ? E : F, p == 2 ? F : E);
        off[p] = slot_bytes;
        slot_bytes += (part[p] + 255) / 256 * 256;
    }
    // the even experts in the slab at slots 0 .. 5
    std::vector<uint8_t> slab((size_t) (NE / 2) * slot_bytes);
    std::mt19937_64 rng(7u);
    for (int s = 0; s < NE / 2; ++s)
        for (int p = 0; p < 3; ++p)
            qsynth::random_blocks_into(slab.data() + (size_t) s * slot_bytes + off[p], types[p], p == 2 ? E : F,
                                       p == 2 ? F : E, p == 2 ? 1.f / 48.f : 1.f / 72.f, rng);
    std::vector<int32_t> slot_of((size_t) NE, -1);
    for (int x = 0; x < NE; x += 2) slot_of[(size_t) x] = x / 2;
    void* sd = upload(slab);
    int32_t* so = upload(slot_of);
    e::QExpertSlab sl;
    sl.base = sd;
    sl.slot_of = so;
    sl.slot_bytes = slot_bytes;
    for (int p = 0; p < 3; ++p) {
        sl.part_offset[p] = off[p];
        sl.type[p] = types[p];
    }
    sl.n_embd = E;
    sl.n_ff = F;
    e::qmatmul_experts_check(sl, "(g2) slab");
    const int64_t Tm = e::kMmvqMaxCols, Pm = Tm * K;
    void* scratch = synth::dev_alloc<uint8_t>(e::qmatmul_experts_scratch_bytes(sl, Tm, Pm));
    float* hd = synth::dev_alloc<float>((size_t) (Pm * F));
    float* yd = synth::dev_alloc<float>((size_t) (Pm * E));
    std::mt19937 pick(20261014u);
    for (int64_t T : {2, 3, 4, 5, 6, 7, 8}) {
        // each token K distinct experts of 12: tokens share many
        std::vector<int32_t> ids;
        for (int64_t t = 0; t < T; ++t) {
            std::vector<int32_t> all(NE);
            for (int i = 0; i < NE; ++i) all[(size_t) i] = i;
            std::shuffle(all.begin(), all.end(), pick);
            ids.insert(ids.end(), all.begin(), all.begin() + K);
        }
        const std::vector<float> x = normal((size_t) (T * E), (uint32_t) T);
        float* xd = upload(x);
        int32_t* idd = upload(ids);
        e::qmatmul_experts(sl, xd, T, idd, K, 10.f, hd, scratch, yd, st);
        ck(cudaStreamSynchronize(st), "(g2) batch");
        const std::vector<float> batch = download(yd, (size_t) (T * K * E));
        int64_t differ = 0, hits = 0, shared = 0;
        for (int64_t t = 0; t < T; ++t) {
            e::qmatmul_experts(sl, xd + t * E, 1, idd + t * K, K, 10.f, hd, scratch, yd, st);
            ck(cudaStreamSynchronize(st), "(g2) one token");
            const std::vector<float> one = download(yd, (size_t) (K * E));
            for (int s = 0; s < K; ++s) {
                const int32_t ex = ids[(size_t) (t * K + s)];
                if (slot_of[(size_t) ex] < 0) continue;
                ++hits;
                shared += e::expert_hit_leader(ids.data(), T * K, t * K + s, slot_of.data()) != t * K + s;
                differ += bits_differ(std::vector<float>(one.begin() + s * E, one.begin() + (s + 1) * E),
                                      std::vector<float>(batch.begin() + (t * K + s) * E, batch.begin() + (t * K + s + 1) * E));
            }
        }
        char what[256];
        std::snprintf(what, sizeof what, "(g2) qmatmul_experts at %lld tokens: %lld hit pairs (%lld computed by another "
                      "token's leader), every row the one-token call's bit for bit (%lld values differ)", (long long) T,
                      (long long) hits, (long long) shared, (long long) differ);
        expect(differ == 0 && shared > 0, what);
        cudaFree(xd);
        cudaFree(idd);
    }
    for (void* p : {sd, (void*) so, scratch, (void*) hd, (void*) yd}) cudaFree(p);
}

// the bytes of every buffer of a cache
std::vector<uint8_t> cache_bytes(e::KvCache& kv) {
    std::vector<uint8_t> out;
    const e::AttnDims& d = kv.dims();
    auto add = [&](const void* p, size_t n) {
        if (!p || !n) return;
        const size_t at = out.size();
        out.resize(at + n);
        ck(cudaMemcpy(out.data() + at, p, n, cudaMemcpyDeviceToHost), "cache bytes");
    };
    for (const e::AttnLayer& l : kv.layers()) {
        const e::KvLayerBuffers& b = kv.layer(l.index);
        add(b.window, (size_t) b.window_cells * d.head_dim * 2);
        add(b.rows, (size_t) b.rows_cap * d.head_dim * 2);
        add(b.keys, (size_t) b.rows_cap * d.idx_dim * 2);
        if (l.kv_source) {
            add(b.carry_kv, (size_t) l.ratio * d.head_dim * 4);
            add(b.carry_score, (size_t) l.ratio * d.head_dim * 4);
        }
    }
    return out;
}

// (g3)
void attention(const char* name, const e::AttnDims& d, int64_t max_ctx, const int64_t (&hist)[4], cudaStream_t st) {
    constexpr int B = 4, kSteps = 3;
    synth::Model model;
    synth::make_model(model, d, 1234u, e::kQTypeQ2K);
    e::AttnEngine eng;
    std::string err;
    if (!eng.init(d, e::kMmvqMaxCols, max_ctx, st, err) || !eng.init_batch(B, err)) {
        expect(false, std::string("(g3) ") + name + ": " + err);
        return;
    }
    std::vector<std::unique_ptr<e::KvCache>> batch, alone;
    for (int b = 0; b < B; ++b)
        for (auto* set : {&batch, &alone}) {
            set->push_back(std::make_unique<e::KvCache>());
            if (!set->back()->init(d, model.layers, e::KvCacheConfig{max_ctx, 1}, err)) {
                expect(false, std::string("(g3) ") + name + ": " + err);
                return;
            }
            synth::write_history(*set->back(), hist[b], 77u + (uint64_t) b);
        }
    std::vector<e::KvCache*> kvs;
    for (auto& k : batch) kvs.push_back(k.get());
    const int64_t E = d.n_embd;
    const size_t row_q = e::qact_bytes(e::QActLayout::Q8_1, 1, E);
    float* xd = synth::dev_alloc<float>((size_t) (B * E));
    void* qd = synth::dev_alloc<uint8_t>(e::qact_bytes(e::QActLayout::Q8_1, B, E));
    float* outb = synth::dev_alloc<float>((size_t) (B * E));
    float* outa = synth::dev_alloc<float>((size_t) E);
    int32_t* posd = synth::dev_alloc<int32_t>(B);
    int64_t differ = 0, rows = 0;
    for (int s = 0; s < kSteps; ++s) {
        int64_t p[B];
        int32_t p32[B];
        for (int b = 0; b < B; ++b) p32[b] = (int32_t) (p[b] = hist[b] + s);
        // on the stream that reads them (it is non-blocking: a legacy-stream copy is not ordered before its kernels)
        ck(cudaMemcpyAsync(posd, p32, sizeof p32, cudaMemcpyHostToDevice, st), "(g3) positions");
        for (size_t il = 0; il < model.layers.size(); ++il) {
            const e::AttnLayer& l = model.layers[il];
            const std::vector<float> x = synth::make_input(B, (int) E, (uint32_t) (s * 16 + il));
            ck(cudaMemcpyAsync(xd, x.data(), x.size() * 4, cudaMemcpyHostToDevice, st), "(g3) x");
            e::QActs q;
            e::qact_quantize(xd, nullptr, E, B, E, e::QActLayout::Q8_1, qd, st, q);
            e::attn_decode_batch(eng, l, model.dev[il], kvs.data(), B, xd, qd, posd, p, outb);
            ck(cudaStreamSynchronize(st), "(g3) batch");
            const std::vector<float> got = download(outb, (size_t) (B * E));
            for (int b = 0; b < B; ++b) {
                e::attn_decode(eng, l, model.dev[il], *alone[(size_t) b], xd + b * E, p[b], outa, e::ReaderMode::Reuse,
                               nullptr, false, static_cast<uint8_t*>(qd) + b * row_q);
                ck(cudaStreamSynchronize(st), "(g3) alone");
                const std::vector<float> want = download(outa, (size_t) E);
                differ += bits_differ(want, std::vector<float>(got.begin() + b * E, got.begin() + (b + 1) * E));
                ++rows;
            }
        }
    }
    int64_t cache_differ = 0;
    for (int b = 0; b < B; ++b) {
        const std::vector<uint8_t> x = cache_bytes(*batch[(size_t) b]), y = cache_bytes(*alone[(size_t) b]);
        cache_differ += x.size() != y.size() || std::memcmp(x.data(), y.data(), x.size()) != 0;
    }
    char what[320];
    std::snprintf(what, sizeof what, "(g3) %s: attn_decode_batch of 4 sequences (histories %lld, %lld, %lld, %lld of %lld "
                  "positions), %d tokens through the six roles: %lld output rows, %lld values differ from attn_decode "
                  "alone; %lld caches differ after them", name, (long long) hist[0], (long long) hist[1], (long long) hist[2],
                  (long long) hist[3], (long long) max_ctx, kSteps, (long long) rows, (long long) differ,
                  (long long) cache_differ);
    expect(differ == 0 && cache_differ == 0, what);
    for (void* q : {(void*) xd, qd, (void*) outb, (void*) outa, (void*) posd}) cudaFree(q);
}

}  // namespace

int run_kernels() {
    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
    columns(st);
    experts(st);
    attention("tiny-a dims", synth::dims_tiny_a(), 160, {20, 37, 64, 150}, st);
    attention("tiny-b dims", synth::dims_tiny_b(), 160, {21, 40, 63, 149}, st);
    attention("real dims", e::attn_dims_real(), 32768, {700, 4097, 16384, 32760}, st);
    cudaStreamDestroy(st);
    std::printf("dsv41_engine_batch_test --kernels: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
