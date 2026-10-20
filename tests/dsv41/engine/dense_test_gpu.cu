// tests/dsv41/engine/dense_test_gpu.cu - (b) of dsv41_engine_dense_test: the CUDA dense ops against the CPU reference
// (dense_ref.hpp) in the engine's numerics, stage by stage, at tiny and real dims (see dense_test.cpp for (a)).
//
// A case is one token chunk through every dense op in a layer's order on a synthetic model (dense_synth.cuh): the
// embedding, the engram (rows hashed on the host, gathered from the pinned table), the attention sublayer's mixes,
// collapse and norm (with the one-hot pre_init), its expansion of a random attention output, the FFN sublayer's mixes,
// collapse and norm, the router, the shared expert, its expansion, and the head over the chunk's last rows. Every op
// runs on the GPU first; then, for sampled tokens (all of a call of at most 16, else 8 spread over it with the first
// and the last), the reference recomputes it from the GPU's own input to that op, against these tolerances:
//
//   matrix outputs (hc_fn, engram_wkv, the router logits, the shared expert, the output head)   |d| <= 2^-12 of the
//       sum of the absolute terms on MMVQ, MMQ and the router's GEMV (the reference quantizes the GPU's own input
//       bit for bit, its integer or bf16 products are exact: only f32 sums differ), 2^-10 on the dequant path and the
//       router's cuBLAS (f16 or bf16 tensor-core sums; qmatmul_test_gpu.cu, attention_test_gpu.cu)
//   everything else, from the GPU's inputs   |d| <= 2^-14 of the row's largest value (the same f32 operations in the
//       same order; the device's expf, logf and sqrtf against glibc's and the norms' double sums in another order);
//       the f16 copies the dequant path reads 2^-10 (one f16 rounding)
//   exact   the embedding rows, pre_init and the gathered engram rows equal the reference's bytes
//   router  the GPU's ids equal the reference's selection from the GPU's logits; a differing expert is accepted only
//       where its biased probability is within 2^-20 of the row's largest of the one it replaced (a near-tie the two
//       sides' probabilities, a few ulp apart, break differently); the weights where the ids agree
//
// A prefill call (T > 1) also runs hc_prefill_step the same way, every value bit for bit.
// A decode token (T = 1) also runs hc_decode_step against the unfused ops it replaces: every value it
// writes (the expanded stream, rms(x), the mixes, pre, post, comb, the collapse, the norm and its q8_1) equals theirs
// bit for bit.
//
// The engram projection runs in sub-chunks (engram_chunk), so its kv and gate are checked on the last sub-chunk's
// tokens, the ones its scratch holds. Cases: tiny-a and tiny-b dims (64 tokens with 16-token engram sub-chunks, 4 and
// 1); the real dims (1, 4, and 4096 tokens with 1024-token sub-chunks) with an engram table of --table-rows rows.
//
// --dry-run (the dev VM, verify.sh; no GPU, no CUDA call): every case above built on the host, and for every call the
// host-side work of a real run up to its first launch: the ops' weight checks and the reference's numerics for the
// call's path (each matrix run by the reference on one token), and the engram rows of the case's tokens hashed and
// checked against the table's size.
#include "dense_synth.cuh"
#include "dense_test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

namespace e = strata::dsv41::engine;
namespace kn = strata::kernels;
using synth::download;

constexpr double kTolGemm = 1.0 / 1024;
constexpr double kTolQGemm = 1.0 / 4096;
constexpr double kTolElem = 1.0 / 16384;
constexpr double kTolF16 = 1.0 / 1024;
constexpr double kTolTie = 1.0 / (1 << 20);

struct Stage {
    double tol = 0, worst = 0;
    int64_t values = 0, over = 0;
};
struct Report {
    std::map<std::string, Stage> st;
    int64_t sets = 0, sets_exact = 0, sets_near = 0, sets_bad = 0;
    void add(const std::string& name, double tol, double err, double unit) {
        Stage& s = st[name];
        s.tol = tol;
        ++s.values;
        const double ratio = unit > 0 ? err / unit : (err > 0 ? INFINITY : 0);
        s.worst = std::max(s.worst, ratio);
        if (!(err <= tol * unit) && !(err == 0)) ++s.over;
    }
    bool ok() const {
        bool good = sets_bad == 0;
        for (const auto& [n, s] : st) good = good && s.over == 0;
        return good;
    }
    void print(const char* name) const {
        std::printf("%s %s\n", ok() ? "ok  " : "FAIL", name);
        for (const auto& [n, s] : st)
            std::printf("    %-34s %9lld values, worst %.3g of its unit (tolerance %.3g), %lld over\n", n.c_str(),
                        (long long) s.values, s.worst, s.tol, (long long) s.over);
        if (sets)
            std::printf("    %-34s %9lld sets: %lld identical, %lld differ only in near-ties, %lld otherwise\n", "router top-k",
                        (long long) sets, (long long) sets_exact, (long long) sets_near, (long long) sets_bad);
    }
};

void cmp_rows(Report& rep, const std::string& name, double tol, const float* gpu, const float* ref, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        double mx = 0;
        for (int64_t c = 0; c < cols; ++c) mx = std::max(mx, (double) std::fabs(ref[r * cols + c]));
        for (int64_t c = 0; c < cols; ++c) rep.add(name, tol, std::fabs((double) gpu[r * cols + c] - ref[r * cols + c]), mx);
    }
}
void cmp_scaled(Report& rep, const std::string& name, double tol, const float* gpu, const float* ref, const double* scale, int64_t n) {
    for (int64_t i = 0; i < n; ++i) rep.add(name, tol, std::fabs((double) gpu[i] - ref[i]), scale[i]);
}
void cmp_exact(Report& rep, const std::string& name, const void* gpu, const void* ref, size_t bytes, size_t unit) {
    for (size_t i = 0; i < bytes; i += unit)
        rep.add(name, 0, std::memcmp((const char*) gpu + i, (const char*) ref + i, unit) == 0 ? 0 : 1, 1);
}

std::vector<int64_t> sample(int64_t n, int64_t k) {
    std::vector<int64_t> s;
    if (n <= k * 2) {
        for (int64_t i = 0; i < n; ++i) s.push_back(i);
    } else {
        for (int64_t i = 0; i < k; ++i) s.push_back(i * (n - 1) / (k - 1));
    }
    return s;
}
template <typename T> std::vector<T> rows_of(const T* dev, int64_t ld, const std::vector<int64_t>& rows, int64_t cols) {
    std::vector<T> h((size_t) (rows.size() * cols));
    for (size_t i = 0; i < rows.size(); ++i)
        synth::ck(cudaMemcpy(h.data() + i * cols, dev + rows[i] * ld, (size_t) cols * sizeof(T), cudaMemcpyDeviceToHost), "rows_of");
    return h;
}
template <typename T> std::vector<T> host_rows(const std::vector<T>& h, int64_t ld, const std::vector<int64_t>& rows, int64_t cols) {
    std::vector<T> out((size_t) (rows.size() * cols));
    for (size_t i = 0; i < rows.size(); ++i) std::memcpy(out.data() + i * cols, h.data() + rows[i] * ld, sizeof(T) * (size_t) cols);
    return out;
}
std::vector<float> f16_to_f32(const std::vector<uint16_t>& h) {
    std::vector<float> v(h.size());
    for (size_t i = 0; i < h.size(); ++i) v[i] = kn::f32_from_f16(h[i]);
    return v;
}

// the reference numerics of a matrix in a GPU call of T rows: the path qmatmul takes under the prefill
e::RefActs ref_acts(int type, int64_t T, e::QPrefill p) {
    switch (e::qmatmul_path(p, T)) {
        case e::QPath::Mmvq: return e::RefActs::Q8_1;
        case e::QPath::DequantF16: return e::RefActs::F16;
        case e::QPath::Mmq: return e::qact_layout(type, T) == e::QActLayout::MmqD2S6 ? e::RefActs::MmqD2S6 : e::RefActs::MmqD4;
    }
    return e::RefActs::Auto;
}
e::RefMatrix at(const e::RefMatrix& m, int64_t T, e::QPrefill p) {
    e::RefMatrix r = m;
    if (r.type != e::kRefTypeBF16) r.acts = ref_acts(m.type, T, p);
    return r;
}
double tol_of(const e::RefMatrix& m, int64_t T) {
    if (m.type == e::kRefTypeBF16) return T <= e::kRouterGemvMaxCols ? kTolQGemm : kTolGemm;
    return m.acts == e::RefActs::F16 ? kTolGemm : kTolQGemm;
}
const char* path_of(const e::RefMatrix& m, int64_t T, e::QPrefill p) {
    if (m.type == e::kRefTypeBF16) return T <= e::kRouterGemvMaxCols ? "bf16 GEMV" : "bf16 cuBLAS";
    return e::qpath_name(e::qmatmul_path(p, T));
}

// y = w x on the sampled rows in the call's numerics, checked against the GPU's
void check_matmul(Report& rep, const std::string& name, const e::RefMatrix& w, int64_t T, e::QPrefill p, const std::vector<float>& x,
                  int64_t ns, const std::vector<float>& gpu) {
    const e::RefMatrix r = at(w, T, p);
    std::vector<float> y((size_t) (ns * w.rows));
    std::vector<double> sc(y.size());
    e::ref_matmul(r, x.data(), ns, y.data());
    e::ref_matmul_abs_scale(r, x.data(), ns, sc.data());
    cmp_scaled(rep, name, tol_of(r, T), gpu.data(), y.data(), sc.data(), (int64_t) y.size());
}

struct Ctx {
    dsynth::Model& m;
    e::DenseEngine* eng;      // null in a dry run
    e::EngramGather* gather;
    e::QPrefill prefill;
    int64_t engram_chunk = 0;
    bool dry = false;
    std::map<std::string, int64_t>* paths = nullptr;
};

std::vector<int32_t> tokens_of(const e::DenseDims& d, int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> u(0, d.n_vocab - 1);
    std::vector<int32_t> t((size_t) n);
    for (auto& x : t) x = u(rng);
    return t;
}

// --dry-run: the host-side checks of every call and the reference's numerics for its path
void dry_case(Ctx& c, int64_t T, int64_t n_head, const std::vector<int32_t>& tok, const std::vector<int32_t>& prev) {
    const e::DenseDims& d = c.m.d;
    const int C = d.engram_cols();
    e::embed_check_weights(d, c.m.token_embd.view(), "dense_embed");
    e::engram_check_weights(d, c.m.eng, "engram_forward");
    e::hc_check_weights(d, c.m.hc[0], "hc_mixes");
    e::hc_check_weights(d, c.m.hc[1], "hc_mixes");
    e::shared_check_weights(d, c.m.sh, "dense_shared_expert");
    e::router_check_weights(d, c.m.router, "dense_router");
    e::head_check_weights(d, c.m.output.view(), "dense_head");
    std::vector<int32_t> rows((size_t) (T * C));
    e::engram_rows(c.m.hash, 0, tok.data(), T, prev.data(), (int) prev.size(), rows.data());
    const int64_t need = c.m.hash.rows_needed(0);
    for (int32_t r : rows)
        if (r < 0 || r >= need) die("dry run: engram row " + std::to_string(r) + " outside the buckets' " + std::to_string(need));
    const int64_t Te = std::min(T, c.engram_chunk);
    const std::pair<const e::RefMatrix*, int64_t> mats[] = {
        {&c.m.rhc[0].fn, T}, {&c.m.rhc[1].fn, T}, {&c.m.reng.wkv, Te}, {&c.m.rrouter.gate_inp, T}, {&c.m.rsh.gate, T},
        {&c.m.rsh.up, T}, {&c.m.rsh.down, T}, {nullptr, n_head}};
    for (const auto& [mp, rows_T] : mats) {
        const e::RefMatrix base = mp ? *mp : c.m.output.ref();
        const e::RefMatrix r = at(base, rows_T, c.prefill);
        const std::vector<float> x = synth::make_input(1, (int) r.k, 5);
        std::vector<float> y((size_t) r.rows);
        e::ref_matmul(r, x.data(), 1, y.data());
        ++(*c.paths)[path_of(r, rows_T, c.prefill)];
    }
}

bool run_case(Ctx& c, int64_t T, int64_t n_head, uint32_t seed, Report& rep) {
    const e::DenseDims& d = c.m.d;
    const int64_t E = d.n_embd, X = (int64_t) e::kHc * E, C = d.engram_cols(), EK = C * d.engram_key, NE = d.n_expert,
                  K = d.n_expert_used, F = d.n_ff, KV = E * (e::kHc + 1);
    const std::vector<int32_t> tok = tokens_of(d, T, seed);
    // odd seeds: a chunk at the sequence start; even: three tokens before it
    const std::vector<int32_t> prev = seed % 2 ? std::vector<int32_t>{} : tokens_of(d, d.engram_ngram - 1, seed + 1);
    if (c.dry) {
        dry_case(c, T, n_head, tok, prev);
        return true;
    }
    e::DenseEngine& g = *c.eng;
    const e::DenseScratch& s = g.scratch();
    const std::vector<int64_t> ts = sample(T, 8);
    const int64_t ns = (int64_t) ts.size();
    int32_t* tok_dev = synth::upload(tok);
    float* x = synth::dev_alloc<float>((size_t) (T * X));
    float* pre0 = synth::dev_alloc<float>((size_t) (T * e::kHc));
    float* pre_a = synth::dev_alloc<float>((size_t) (T * e::kHc));
    float* post_a = synth::dev_alloc<float>((size_t) (T * e::kHc));
    float* comb_a = synth::dev_alloc<float>((size_t) (T * 16));
    float* pre_f = synth::dev_alloc<float>((size_t) (T * e::kHc));
    float* post_f = synth::dev_alloc<float>((size_t) (T * e::kHc));
    float* comb_f = synth::dev_alloc<float>((size_t) (T * 16));
    float* h = synth::dev_alloc<float>((size_t) (T * E));
    float* sh_out = synth::dev_alloc<float>((size_t) (T * E));
    int32_t* ids = synth::dev_alloc<int32_t>((size_t) (T * K));
    float* wts = synth::dev_alloc<float>((size_t) (T * K));
    float* logits = synth::dev_alloc<float>((size_t) (n_head * d.n_vocab));
    auto sync = [](const char* w) { synth::ck(cudaDeviceSynchronize(), w); };

    // 1. embedding
    e::dense_embed(g, c.m.token_embd.view(), tok_dev, T, x, pre0);
    sync("dense_embed");
    {
        std::vector<int32_t> st;
        for (int64_t t : ts) st.push_back(tok[(size_t) t]);
        std::vector<float> emb((size_t) (ns * E));
        e::ref_embed(c.m.token_embd.ref(), st.data(), ns, emb.data());
        const auto gx = rows_of(x, X, ts, X);
        for (int64_t i = 0; i < ns; ++i)
            for (int cc = 0; cc < e::kHc; ++cc)
                cmp_exact(rep, "embedding rows (4 copies)", gx.data() + (i * e::kHc + cc) * E, emb.data() + i * E, (size_t) E * 4, 4);
        const auto gp = rows_of(pre0, e::kHc, ts, e::kHc);
        for (int64_t i = 0; i < ns; ++i) {
            const float one_hot[4] = {1.f, 0.f, 0.f, 0.f};
            cmp_exact(rep, "pre_init", gp.data() + i * e::kHc, one_hot, 16, 4);
        }
    }
    // 2. engram
    {
        std::vector<int32_t> rows((size_t) (T * C));
        e::engram_rows(c.m.hash, 0, tok.data(), T, prev.data(), (int) prev.size(), rows.data());
        const auto x0 = download(x, (size_t) (T * X));
        const void* rows_dev = c.gather->gather(c.m.table, rows.data(), T * C, g.stream());
        e::engram_forward(g, c.m.eng, rows_dev, T, x);
        sync("engram_forward");
        const auto gr = download(static_cast<const uint8_t*>(rows_dev), (size_t) (T * C * 84));
        for (int64_t t : ts)
            for (int64_t b = 0; b < C; ++b)
                cmp_exact(rep, "gathered engram rows", gr.data() + (t * C + b) * 84, c.m.table.rows() + (int64_t) rows[(size_t) (t * C + b)] * 84, 84, 84);
        const int64_t Te = s.engram_T, t_first = T - Te;
        std::vector<int64_t> tl, tl_local;
        for (int64_t t : sample(Te, 8)) { tl.push_back(t_first + t); tl_local.push_back(t); }
        const int64_t nl = (int64_t) tl.size();
        std::vector<int32_t> rl;
        for (int64_t t : tl) rl.insert(rl.end(), rows.begin() + t * C, rows.begin() + (t + 1) * C);
        std::vector<float> emb((size_t) (nl * EK));
        e::ref_engram_emb(c.m.table.rows(), d.engram_key, rl.data(), nl * C, emb.data());
        std::vector<float> gemb;
        if (e::qmatmul_path(c.prefill, Te) == e::QPath::Mmvq) {
            gemb = rows_of(s.emb, EK, tl_local, EK);
            cmp_exact(rep, "engram rows dequantized", gemb.data(), emb.data(), gemb.size() * 4, 4);
        } else {
            const auto gh = rows_of(s.embh, EK, tl_local, EK);
            std::vector<uint16_t> want(emb.size());
            for (size_t i = 0; i < emb.size(); ++i) want[i] = kn::f16_from_f32(emb[i]);
            cmp_exact(rep, "engram rows dequantized (f16)", gh.data(), want.data(), gh.size() * 2, 2);
            gemb = f16_to_f32(gh);
        }
        const auto gkv = rows_of(s.engram_kv, KV, tl_local, KV);
        check_matmul(rep, "engram_wkv", c.m.reng.wkv, Te, c.prefill, gemb, nl, gkv);
        std::vector<float> gate((size_t) (nl * e::kHc)), out((size_t) (nl * X));
        const auto xs = host_rows(x0, X, tl, X);
        e::ref_engram_gate(d, c.m.reng, gkv.data(), xs.data(), nl, gate.data(), out.data());
        cmp_rows(rep, "engram gate", kTolElem, rows_of(s.engram_gate, e::kHc, tl, e::kHc).data(), gate.data(), nl, e::kHc);
        cmp_rows(rep, "engram out (the stream)", kTolElem, rows_of(x, X, tl, X).data(), out.data(), nl, X);
    }
    // 3. the attention sublayer: mixes, collapse with pre_init, norm, expansion of a random attention output
    auto check_mixes = [&](const char* tag, int sub, const float* pre, const float* post, const float* comb, const std::vector<float>& xin) {
        std::vector<float> hin((size_t) (ns * X));
        e::ref_hc_norm(d, xin.data(), ns, hin.data());
        std::vector<float> gin;
        if (e::qmatmul_path(c.prefill, T) == e::QPath::Mmvq) {
            gin = rows_of(s.hc_in, X, ts, X);
            cmp_rows(rep, std::string(tag) + " rms(x)", kTolElem, gin.data(), hin.data(), ns, X);
        } else {
            gin = f16_to_f32(rows_of(s.hc_inh, X, ts, X));
            cmp_rows(rep, std::string(tag) + " rms(x) (f16)", kTolF16, gin.data(), hin.data(), ns, X);
        }
        const auto gm = rows_of(s.mixes, e::kHcMix, ts, e::kHcMix);
        check_matmul(rep, std::string(tag) + " hc_fn", c.m.rhc[sub].fn, T, c.prefill, gin, ns, gm);
        std::vector<float> rp((size_t) (ns * e::kHc)), rq(rp.size()), rc((size_t) (ns * 16));
        e::ref_hc_coeffs(d, c.m.rhc[sub], gm.data(), ns, rp.data(), rq.data(), rc.data());
        cmp_rows(rep, std::string(tag) + " pre", kTolElem, rows_of(pre, e::kHc, ts, e::kHc).data(), rp.data(), ns, e::kHc);
        cmp_rows(rep, std::string(tag) + " post", kTolElem, rows_of(post, e::kHc, ts, e::kHc).data(), rq.data(), ns, e::kHc);
        cmp_rows(rep, std::string(tag) + " comb (Sinkhorn)", kTolElem, rows_of(comb, 16, ts, 16).data(), rc.data(), ns, 16);
    };
    auto check_collapse = [&](const char* tag, const std::vector<float>& xin, const float* pre, const std::vector<float>& normw, const float* out) {
        const auto gp = rows_of(pre, e::kHc, ts, e::kHc);
        std::vector<float> col((size_t) (ns * E)), nrm(col.size());
        e::ref_hc_collapse(xin.data(), gp.data(), ns, (int) E, true, col.data());
        const auto gcol = rows_of(s.collapsed, E, ts, E);
        cmp_rows(rep, std::string(tag) + " collapse", kTolElem, gcol.data(), col.data(), ns, E);
        e::ref_rms_norm(gcol.data(), normw.data(), ns, (int) E, d.rms_eps, nrm.data());
        cmp_rows(rep, std::string(tag) + " norm", kTolElem, rows_of(out, E, ts, E).data(), nrm.data(), ns, E);
    };
    auto check_expand = [&](const char* tag, const std::vector<float>& y, const std::vector<float>& xin, const float* post, const float* comb) {
        std::vector<float> out((size_t) (ns * X));
        e::ref_hc_expand(y.data(), xin.data(), rows_of(post, e::kHc, ts, e::kHc).data(), rows_of(comb, 16, ts, 16).data(), ns, (int) E,
                         out.data());
        cmp_rows(rep, std::string(tag) + " expansion", kTolElem, rows_of(x, X, ts, X).data(), out.data(), ns, X);
    };
    {
        const auto xin = rows_of(x, X, ts, X);
        e::hc_mixes(g, c.m.hc[0], x, T, pre_a, post_a, comb_a);
        sync("hc_mixes");
        check_mixes("attention", 0, pre_a, post_a, comb_a, xin);
        e::hc_collapse_norm(g, x, pre0, c.m.attn_norm.dev, T, h);
        sync("hc_collapse_norm");
        check_collapse("attention", xin, pre0, c.m.attn_norm.host, h);
        const std::vector<float> y = synth::make_input(T, (int) E, seed + 3);
        float* y_dev = synth::upload(y);
        e::hc_expand(g, y_dev, x, post_a, comb_a, T, x);
        sync("hc_expand");
        check_expand("attention", host_rows(y, E, ts, E), xin, post_a, comb_a);
        cudaFree(y_dev);
    }
    // 4. the FFN sublayer: mixes, collapse with the attention pre, norm, router, shared expert, expansion
    {
        const auto xin = rows_of(x, X, ts, X);
        e::hc_mixes(g, c.m.hc[1], x, T, pre_f, post_f, comb_f);
        sync("hc_mixes");
        check_mixes("FFN", 1, pre_f, post_f, comb_f, xin);
        e::hc_collapse_norm(g, x, pre_a, c.m.ffn_norm.dev, T, h);
        sync("hc_collapse_norm");
        check_collapse("FFN", xin, pre_a, c.m.ffn_norm.host, h);
        const auto xn = rows_of(h, E, ts, E);
        e::dense_router(g, c.m.router, h, T, ids, wts);
        sync("dense_router");
        const auto gl = rows_of(s.logits, NE, ts, NE);
        check_matmul(rep, "router logits (bf16)", c.m.rrouter.gate_inp, T, c.prefill, xn, ns, gl);
        e::RefRouterTrace tr;
        tr.logits = gl;
        e::ref_router_select(e::RefNumerics::Engine, d, c.m.bias.host.data(), ns, tr);
        cmp_rows(rep, "router probs", kTolElem, rows_of(s.probs, NE, ts, NE).data(), tr.probs.data(), ns, NE);
        const auto gid = rows_of(ids, K, ts, K);
        const auto gw = rows_of(wts, K, ts, K);
        for (int64_t i = 0; i < ns; ++i) {
            ++rep.sets;
            const int32_t* a = gid.data() + i * K;
            const int32_t* b = tr.ids.data() + i * K;
            if (std::equal(a, a + K, b)) {
                ++rep.sets_exact;
                cmp_rows(rep, "router weights", kTolElem, gw.data() + i * K, tr.scaled.data() + i * K, 1, K);
                continue;
            }
            const float* bz = tr.biased.data() + i * NE;
            double mx = 0;
            for (int64_t j = 0; j < NE; ++j) mx = std::max(mx, (double) std::fabs(bz[j]));
            bool near = true;
            for (int64_t k = 0; k < K; ++k)
                near = near && a[k] >= 0 && a[k] < NE && std::fabs((double) bz[a[k]] - bz[b[k]]) <= kTolTie * mx;
            if (near) ++rep.sets_near;
            else ++rep.sets_bad;
        }
        e::dense_shared_expert(g, c.m.sh, h, T, sh_out);
        sync("dense_shared_expert");
        const auto gu = rows_of(s.sh_up, F, ts, F), gg = rows_of(s.sh_gate, F, ts, F), ga = rows_of(s.sh_act, F, ts, F);
        check_matmul(rep, "shared up", c.m.rsh.up, T, c.prefill, xn, ns, gu);
        check_matmul(rep, "shared gate", c.m.rsh.gate, T, c.prefill, xn, ns, gg);
        std::vector<float> act((size_t) (ns * F));
        e::ref_swiglu_clamp(gg.data(), gu.data(), ns * F, d.swiglu_limit, act.data());
        cmp_rows(rep, "shared SwiGLU (clamped)", kTolElem, ga.data(), act.data(), ns, F);
        const auto go = rows_of(sh_out, E, ts, E);
        check_matmul(rep, "shared down", c.m.rsh.down, T, c.prefill, ga, ns, go);
        e::hc_expand(g, sh_out, x, post_f, comb_f, T, x);
        sync("hc_expand");
        check_expand("FFN", go, xin, post_f, comb_f);
    }
    // 5. a decode token: hc_decode_step against the unfused ops it replaces (hc_expand, hc_mixes,
    // hc_collapse_norm, qact_quantize of the normed row), bit for bit, with the expansion (post and comb read and
    // written in the same buffers, as the forward does) and without, on a random stream and sublayer output
    if (T == 1) {
        const std::vector<float> x0 = synth::make_input(1, (int) X, seed + 7), y = synth::make_input(1, (int) E, seed + 8);
        const auto post0 = download(post_f, e::kHc), comb0 = download(comb_f, 16);
        const size_t qb = e::qact_bytes(e::QActLayout::Q8_1, 1, E);
        for (bool expand : {true, false}) {
            const char* tag = expand ? "hc_decode_step (expansion)" : "hc_decode_step";
            float* y_dev = synth::upload(y);
            float* xu = synth::upload(x0);
            float* xf = synth::upload(x0);
            float* pu = synth::upload(post0);
            float* cu = synth::upload(comb0);
            float* pf = synth::upload(post0);
            float* cf = synth::upload(comb0);
            float* pre_u = synth::dev_alloc<float>(e::kHc);
            float* pre_s = synth::dev_alloc<float>(e::kHc);
            float* hu = synth::dev_alloc<float>((size_t) E);
            float* hf = synth::dev_alloc<float>((size_t) E);
            uint8_t* qu = synth::dev_alloc<uint8_t>(qb);
            uint8_t* qf = synth::dev_alloc<uint8_t>(qb);
            if (expand) e::hc_expand(g, y_dev, xu, pu, cu, 1, xu);
            e::hc_mixes(g, c.m.hc[0], xu, 1, pre_u, pu, cu);
            e::hc_collapse_norm(g, xu, pre_a, c.m.attn_norm.dev, 1, hu);
            e::QActs qa;
            e::qact_quantize(hu, nullptr, E, 1, E, e::QActLayout::Q8_1, qu, g.stream(), qa);
            sync("the unfused ops");
            const auto in_u = download(s.hc_in, (size_t) X), mix_u = download(s.mixes, e::kHcMix), col_u = download(s.collapsed, (size_t) E);
            e::hc_decode_step(g, c.m.hc[0], expand ? y_dev : nullptr, xf, pre_a, c.m.attn_norm.dev, pre_s, pf, cf, hf, qf);
            sync("hc_decode_step");
            const std::string t = tag;
            cmp_exact(rep, t + ": x", download(xf, (size_t) X).data(), download(xu, (size_t) X).data(), (size_t) X * 4, 4);
            cmp_exact(rep, t + ": rms(x)", download(s.hc_in, (size_t) X).data(), in_u.data(), (size_t) X * 4, 4);
            cmp_exact(rep, t + ": mixes", download(s.mixes, e::kHcMix).data(), mix_u.data(), e::kHcMix * 4, 4);
            cmp_exact(rep, t + ": pre", download(pre_s, e::kHc).data(), download(pre_u, e::kHc).data(), e::kHc * 4, 4);
            cmp_exact(rep, t + ": post", download(pf, e::kHc).data(), download(pu, e::kHc).data(), e::kHc * 4, 4);
            cmp_exact(rep, t + ": comb", download(cf, 16).data(), download(cu, 16).data(), 16 * 4, 4);
            cmp_exact(rep, t + ": collapse", download(s.collapsed, (size_t) E).data(), col_u.data(), (size_t) E * 4, 4);
            cmp_exact(rep, t + ": norm", download(hf, (size_t) E).data(), download(hu, (size_t) E).data(), (size_t) E * 4, 4);
            cmp_exact(rep, t + ": norm q8_1", download(qf, qb).data(), download(qu, qb).data(), qb, 36);
            for (void* q : std::initializer_list<void*>{y_dev, xu, xf, pu, cu, pf, cf, pre_u, pre_s, hu, hf, qu, qf}) cudaFree(q);
        }
    }
    // 6. a prefill chunk: hc_prefill_step against the unfused ops it replaces (hc_expand, hc_mixes,
    // hc_collapse_norm), bit for bit, with the expansion and without, on a random stream and sublayer output
    if (T > 1) {
        const std::vector<float> x0 = synth::make_input(T, (int) X, seed + 9), y = synth::make_input(T, (int) E, seed + 10);
        const auto post0 = download(post_f, (size_t) (T * e::kHc)), comb0 = download(comb_f, (size_t) (T * 16));
        const bool f16 = T > e::kMmvqMaxCols;
        auto rms_in = [&] {   // the stream's normed rows the hc_fn matmul read, as bytes
            std::vector<uint8_t> b((size_t) (T * X) * (f16 ? 2 : 4));
            synth::ck(cudaMemcpy(b.data(), f16 ? (const void*) s.hc_inh : (const void*) s.hc_in, b.size(), cudaMemcpyDeviceToHost), "rms(x)");
            return b;
        };
        for (bool expand : {true, false}) {
            const std::string t = expand ? "hc_prefill_step (expansion)" : "hc_prefill_step";
            float* y_dev = synth::upload(y);
            float* xu = synth::upload(x0);
            float* xf = synth::upload(x0);
            float* pu = synth::upload(post0);
            float* cu = synth::upload(comb0);
            float* pf = synth::upload(post0);
            float* cf = synth::upload(comb0);
            float* pre_u = synth::dev_alloc<float>((size_t) (T * e::kHc));
            float* pre_s = synth::dev_alloc<float>((size_t) (T * e::kHc));
            float* hu = synth::dev_alloc<float>((size_t) (T * E));
            float* hf = synth::dev_alloc<float>((size_t) (T * E));
            if (expand) e::hc_expand(g, y_dev, xu, pu, cu, T, xu);
            e::hc_mixes(g, c.m.hc[0], xu, T, pre_u, pu, cu);
            e::hc_collapse_norm(g, xu, pre_a, c.m.attn_norm.dev, T, hu);
            sync("the unfused ops");
            const auto in_u = rms_in();
            const auto mix_u = download(s.mixes, (size_t) (T * e::kHcMix)), col_u = download(s.collapsed, (size_t) (T * E));
            e::hc_prefill_step(g, c.m.hc[0], expand ? y_dev : nullptr, xf, pre_a, c.m.attn_norm.dev, T, pre_s, pf, cf, hf);
            sync("hc_prefill_step");
            const size_t nx = (size_t) (T * X), ne = (size_t) (T * E);
            cmp_exact(rep, t + ": x", download(xf, nx).data(), download(xu, nx).data(), nx * 4, 4);
            const auto in_f = rms_in();
            cmp_exact(rep, t + ": rms(x)", in_f.data(), in_u.data(), in_u.size(), f16 ? 2 : 4);
            cmp_exact(rep, t + ": mixes", download(s.mixes, (size_t) (T * e::kHcMix)).data(), mix_u.data(), (size_t) (T * e::kHcMix) * 4, 4);
            cmp_exact(rep, t + ": pre", download(pre_s, (size_t) (T * e::kHc)).data(), download(pre_u, (size_t) (T * e::kHc)).data(), (size_t) (T * e::kHc) * 4, 4);
            cmp_exact(rep, t + ": post", download(pf, (size_t) (T * e::kHc)).data(), download(pu, (size_t) (T * e::kHc)).data(), (size_t) (T * e::kHc) * 4, 4);
            cmp_exact(rep, t + ": comb", download(cf, (size_t) (T * 16)).data(), download(cu, (size_t) (T * 16)).data(), (size_t) (T * 16) * 4, 4);
            cmp_exact(rep, t + ": collapse", download(s.collapsed, ne).data(), col_u.data(), ne * 4, 4);
            cmp_exact(rep, t + ": norm", download(hf, ne).data(), download(hu, ne).data(), ne * 4, 4);
            for (void* q : std::initializer_list<void*>{y_dev, xu, xf, pu, cu, pf, cf, pre_u, pre_s, hu, hf}) cudaFree(q);
        }
    }
    // 8. head over the last n_head rows
    {
        const int64_t r0 = T - n_head;
        e::dense_head(g, x + r0 * X, pre_f + r0 * e::kHc, n_head, c.m.output_norm.dev, c.m.output.view(), logits);
        sync("dense_head");
        const auto gx = download(x + r0 * X, (size_t) (n_head * X));
        const auto gp = download(pre_f + r0 * e::kHc, (size_t) (n_head * e::kHc));
        std::vector<float> col((size_t) (n_head * E)), nrm(col.size());
        e::ref_hc_collapse(gx.data(), gp.data(), n_head, (int) E, false, col.data());
        const auto gcol = download(s.head_in, (size_t) (n_head * E));
        cmp_rows(rep, "head collapse (unfused)", kTolElem, gcol.data(), col.data(), n_head, E);
        e::ref_rms_norm(gcol.data(), c.m.output_norm.host.data(), n_head, (int) E, d.rms_eps, nrm.data());
        const auto gn = download(s.head_norm, (size_t) (n_head * E));
        cmp_rows(rep, "head norm", kTolElem, gn.data(), nrm.data(), n_head, E);
        check_matmul(rep, "output (q6_K)", c.m.output.ref(), n_head, c.prefill, gn, n_head, download(logits, (size_t) (n_head * d.n_vocab)));
    }
    for (void* p : std::initializer_list<void*>{tok_dev, x, pre0, pre_a, post_a, comb_a, pre_f, post_f, comb_f, h, sh_out, ids, wts, logits})
        cudaFree(p);
    return rep.ok();
}

const char* prefill_name(e::QPrefill p) { return p == e::QPrefill::Mmq ? "MMQ prefill" : "dequant + cuBLAS prefill"; }

bool dims_cases(const char* name, const e::DenseDims& d, uint32_t seed, int64_t table_rows, int64_t engram_chunk,
                const std::vector<std::pair<int64_t, int64_t>>& calls, e::QPrefill prefill, bool dry,
                std::map<std::string, int64_t>& paths) {
    dsynth::Model m;
    dsynth::make_model(m, d, seed, table_rows, !dry);
    e::DenseEngine eng;
    e::EngramGather gather;
    int64_t max_T = 1, max_head = 1;
    for (const auto& [T, nh] : calls) { max_T = std::max(max_T, T); max_head = std::max(max_head, nh); }
    if (!dry) {
        std::string err;
        if (!eng.init(d, max_T, max_head, engram_chunk, nullptr, err, prefill)) die(err);
        if (!gather.init(max_T * d.engram_cols(), 84, 8, err)) die(err);
        if (std::string(name).rfind("real", 0) == 0)
            std::printf("real dims: dense workspace %.1f MiB at %lld tokens (engram sub-chunks of %lld), engram table %lld rows "
                        "(%.0f MiB pinned)\n", (double) eng.workspace_bytes() / (1 << 20), (long long) max_T,
                        (long long) eng.engram_chunk(), (long long) table_rows, (double) table_rows * 84 / (1 << 20));
    }
    bool ok = true;
    for (const auto& [T, nh] : calls) {
        Ctx c{m, dry ? nullptr : &eng, dry ? nullptr : &gather, prefill, std::min(engram_chunk, max_T), dry, &paths};
        Report rep;
        run_case(c, T, nh, seed + (uint32_t) T, rep);
        char title[224];
        std::snprintf(title, sizeof title, "%s, %lld tokens (%s), head over the last %lld, engram sub-chunks of %lld", name,
                      (long long) T, T > e::kMmvqMaxCols ? prefill_name(prefill) : "MMVQ, router GEMV", (long long) nh,
                      (long long) std::min(engram_chunk, max_T));
        if (dry) {
            std::printf("dry  %s: every call checked\n", title);
            continue;
        }
        rep.print(title);
        ok = ok && rep.ok();
    }
    return ok;
}

}  // namespace

int run_gpu(bool quick, bool mmq, bool dry, int64_t table_rows) {
    if (!dry) {
        int dev_id = 0;
        cudaDeviceProp p{};
        synth::ck(cudaGetDevice(&dev_id), "cudaGetDevice");
        synth::ck(cudaGetDeviceProperties(&p, dev_id), "cudaGetDeviceProperties");
        std::printf("device: %s, sm_%d%d, %.1f GiB\n", p.name, p.major, p.minor, (double) p.totalGlobalMem / (1 << 30));
    }
    const e::QPrefill prefill = mmq ? e::QPrefill::Mmq : e::QPrefill::DequantF16;
    std::map<std::string, int64_t> paths;
    bool ok = true;
    const std::vector<std::pair<int64_t, int64_t>> tiny = {{64, 4}, {4, 2}, {1, 1}};
    ok = dims_cases("tiny-a dims (E 256, 8 experts, 2 used)", dsynth::dims_tiny_a(), 21, 65536, 16, tiny, prefill, dry, paths) && ok;
    ok = dims_cases("tiny-b dims (E 512, 32 experts, 6 used)", dsynth::dims_tiny_b(), 22, 65536, 16, tiny, prefill, dry, paths) && ok;
    std::vector<std::pair<int64_t, int64_t>> real = {{1, 1}, {4, 1}, {4096, 2}};
    if (quick) real.resize(1);
    ok = dims_cases("real dims (E 5120, 384 experts, 6 used, vocab 129280)", e::dense_dims_real(), 23, table_rows, 1024, real,
                    prefill, dry, paths) && ok;
    if (dry) {
        std::printf("dry run: every call's weight checks, engram rows and reference numerics passed; matrices run by the reference per path:");
        for (const auto& [path, n] : paths) std::printf(" %s %lld,", path.c_str(), (long long) n);
        std::printf("\ndsv41_engine_dense_test (b) --dry-run: PASS\n");
        return 0;
    }
    std::printf("dsv41_engine_dense_test (b): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
