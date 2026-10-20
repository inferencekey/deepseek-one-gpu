// tests/dsv41/engine/attention_test_gpu.cu - (b) of dsv41_engine_attention_test: the CUDA attention layer against
// the CPU reference (RefNumerics::Engine), stage by stage, at tiny and real dims (see attention_test.cpp for (a)).
//
// The GPU path feeds its projections as qmatmul does: int8 per 32 values on MMVQ (calls of up to 8 tokens or
// blocks), f16 on the dequant path above that (QPrefill::DequantF16, the default) or int8 tiles on MMQ (--mmq, the
// option); or f16 on the cuBLAS test path (--f16). It does not emulate the fork's Q8_K (SPEC.md "Engine numerics and
// fork numerics"), so nothing here is compared with the fork: (a) checks the reference against the fork's dumps.
//
// Every call runs on the GPU first; then, for sampled tokens (all of a call of at most 16, else 8 spread over it
// with the first and the last) and sampled blocks, the reference recomputes each stage from the GPU's own input to
// that stage (so a difference is that stage's, not an earlier one's), against these tolerances:
//
//   GEMM outputs (comp kv / gate, index weights, wo_a, wo_b)   q2_K weights on MMVQ or MMQ: |d| <= 2^-12 of the sum
//       of the absolute terms: the reference quantizes the GPU's own input bit for bit and its integer products are
//       exact, so only the f32 products and sums differ, at most n 2^-24 for chains of n <= K / 8 <= 1024 additions
//       here (qmatmul_test_gpu.cu); f16 weights (--f16) and q2_K weights on the dequant path: 2^-10 of the same sum:
//       the f16 weights and activations are the reference's bit for bit, f16 products are exact, cuBLAS sums in f32
//       (a tensor core's accumulation can truncate: K 2^-23 is the bound at K = 8192), the reference in double; a
//       wrong row or column is O(1) either way
//   GEMM then a norm, rope or rotation (qr, q, kv, index keys, index queries)   |d| <= 2^-8 of the row's largest
//       value (the GEMM bound above through the norm is at most 2^-10 times sqrt(K) / 3.5 for K <= 8192)
//   no GEMM, from the GPU's inputs (pooled, normed and roped rows)   |d| <= 2^-14 of the row's largest value (f32
//       expf against double)
//   de-rope (stored as f16)   |d| <= 2^-10 of the row's largest value (one f16 rounding is 2^-11)
//   attention   |d| <= 2^-11 of sum_c p_c |v_c| (q and keys enter exactly as f16, the scores sum in f32 on the
//       tensor cores; a missing cell moves the output by its p, 1/24 at the tiny dims). A decode token's attn_kernel
//       keeps about 22 bits of p (the hi/lo split). A prefill chunk's prefill_attn_kernel rounds p once to f16, a
//       relative error of up to 2^-11 per cell, so its worst case alone reaches the tolerance: the bound holds by
//       the errors' independence across cells (round to nearest, signs mixed) and the running maximum's cell
//       entering with p exactly 1, not by a margin in the worst case. The margin is measured, not assumed: the two
//       kernels are separate stages here and the run ends with each one's worst ratio against the tolerance
//   top-k   the GPU's ids against indexer_ref_topk on the reference's scores of the GPU's queries, weights and
//       keys, a differing row accepted only when its score is within the 2e-5 of the scale from the threshold
//   stores   the window cells, compressed rows, index keys and the carried pooling state equal the f16 (or f32) of
//       the GPU's own rows bit for bit
//
// The reference recomputes each projection on the path the GPU's call took (qmatmul's for the call's T tokens, or B
// completed blocks for the index keys, under the engine's QPrefill), from the GPU's f32 input (the de-rope from its
// f16, the index keys from the f16 latent on the f16 paths).
//
// Cases: tiny-a and tiny-b dims (tiny_dsv41.py) with the six roles (attention_synth.cuh), a 64-token prompt in
// chunks of 39 and 25 (a block carried over the chunk boundary at ratio 2) then 8 decode steps, in Reuse Mode and
// with --readers dense; the real dims with a synthetic history: decode at 4K, 16K, 64K and 262,144 positions (Reuse
// Mode), decode at 16K with --readers dense, and a 4096-token prefill chunk at offsets 0 and 16K.
//
// --dry-run (the dev VM, verify.sh; no GPU, no CUDA call): every case above built on the host, and for every call the
// host-side work of a real run up to its first launch: the engine's weight checks (attn_check_weights) and the
// reference's per-call numerics (call_weights), each present matrix run by the reference on one token in the call's
// numerics. A view of the wrong type or a path the reference lacks fails here before a pod.
#include "attention_synth.cuh"
#include "attention_test_common.hpp"

#include "strata/dsv41/engine/indexer.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace e = strata::dsv41::engine;
namespace kn = strata::kernels;
using synth::download;

constexpr double kTolGemm = 1.0 / 1024;
constexpr double kTolQGemm = 1.0 / 4096;
constexpr double kTolProj = 1.0 / 256;
constexpr double kTolElem = 1.0 / 16384;
constexpr double kTolDerope = 1.0 / 1024;
constexpr double kTolAttn = 1.0 / 2048;
constexpr double kTolIdx = 2e-5;

void parallel_for(int64_t n, const std::function<void(int64_t)>& f) {
    const int nt = (int) std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < nt; ++i)
        th.emplace_back([&] { for (int64_t k; (k = next++) < n;) f(k); });
    for (auto& t : th) t.join();
}

struct Stage {
    double tol = 0, worst = 0;   // worst |d| / bound unit
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
            std::printf("    %-34s %9lld sets: %lld identical, %lld differ only in near-ties, %lld otherwise\n", "top-k",
                        (long long) sets, (long long) sets_exact, (long long) sets_near, (long long) sets_bad);
    }
};

// the attention's worst |d| / sum p |v| over every case, per kernel: prefill_attn_kernel (T > 1, p in f16) and
// attn_kernel (decode, p hi/lo)
struct AttnWorst {
    double prefill = 0, decode = 0;
};
AttnWorst g_attn_worst;

// row-relative: |gpu - ref| against the row's largest |ref|
void cmp_rows(Report& rep, const std::string& name, double tol, const float* gpu, const float* ref, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        double mx = 0;
        for (int64_t c = 0; c < cols; ++c) mx = std::max(mx, (double) std::fabs(ref[r * cols + c]));
        for (int64_t c = 0; c < cols; ++c)
            rep.add(name, tol, std::fabs((double) gpu[r * cols + c] - ref[r * cols + c]), mx);
    }
}
// against a per-value scale
void cmp_scaled(Report& rep, const std::string& name, double tol, const float* gpu, const float* ref, const double* scale, int64_t n) {
    for (int64_t i = 0; i < n; ++i) rep.add(name, tol, std::fabs((double) gpu[i] - ref[i]), scale[i]);
}
void cmp_exact16(Report& rep, const std::string& name, const uint16_t* gpu, const float* src, int64_t n) {
    for (int64_t i = 0; i < n; ++i) rep.add(name, 0, gpu[i] == kn::f16_from_f32(src[i]) ? 0 : 1, 1);
}
void cmp_exact32(Report& rep, const std::string& name, const float* gpu, const float* src, int64_t n) {
    for (int64_t i = 0; i < n; ++i) rep.add(name, 0, std::memcmp(gpu + i, src + i, 4) == 0 ? 0 : 1, 1);
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

struct Ctx {
    synth::Model& m;
    e::AttnEngine* eng;   // null in a dry run
    e::KvCache* kv;
    e::ReaderMode mode;
    e::QPrefill prefill;
    bool dry = false;
    std::map<std::string, int64_t>* dry_paths = nullptr;   // a dry run's tally: matrices per path
};


// the reference numerics of a quantized matrix in a GPU call of T columns: the path qmatmul takes under the prefill
e::RefActs ref_acts(int type, int64_t T, e::QPrefill p) {
    switch (e::qmatmul_path(p, T)) {
        case e::QPath::Mmvq: return e::RefActs::Q8_1;
        case e::QPath::DequantF16: return e::RefActs::F16;
        case e::QPath::Mmq: return e::qact_layout(type, T) == e::QActLayout::MmqD2S6 ? e::RefActs::MmqD2S6 : e::RefActs::MmqD4;
    }
    return e::RefActs::Auto;
}

// the layer's reference weights in the numerics of a GPU call of T tokens that completed B blocks; the matrices the
// role lacks stay empty (their type is the default 0, which no path reads)
e::RefLayerWeights call_weights(const e::RefLayerWeights& w, int64_t T, int64_t B, e::QPrefill p) {
    e::RefLayerWeights r = w;
    for (e::RefMatrix* mm : {&r.wq_a, &r.wq_b, &r.wkv, &r.wo_a, &r.wo_b, &r.comp_kv, &r.comp_gate, &r.idx_q_b, &r.idx_proj})
        if (mm->data && mm->type != e::kRefTypeF16) mm->acts = ref_acts(mm->type, T, p);
    if (r.idx_k.data && r.idx_k.type != e::kRefTypeF16 && B > 0) r.idx_k.acts = ref_acts(r.idx_k.type, B, p);
    return r;
}

// a GEMM output's tolerance: f16 weights and the dequant path sum f16 products in cuBLAS, MMVQ and MMQ int8 ones
double tol_gemm(const e::RefMatrix& m) { return m.type == e::kRefTypeF16 || m.acts == e::RefActs::F16 ? kTolGemm : kTolQGemm; }
// the f16 paths read the f16 latent, MMVQ and MMQ quantize the f32 one
bool f16_input(const e::RefMatrix& m) { return m.type == e::kRefTypeF16 || m.acts == e::RefActs::F16; }

// --dry-run: the call's host-side checks (attn_check_weights, as attn_prefill makes them) and the reference's
// numerics for it, every present matrix run on one token as the comparison would
void dry_call(Ctx& c, int il, int64_t T, int64_t p0) {
    const e::AttnLayer& l = c.m.layers[(size_t) il];
    const int64_t B = l.kv_source ? (p0 + T) / l.ratio - p0 / l.ratio : 0;
    e::attn_check_weights(c.m.d, l, c.m.dev[(size_t) il]);
    const e::RefLayerWeights rw = call_weights(c.m.ref[(size_t) il], T, B, c.prefill);
    const std::pair<const char*, const e::RefMatrix*> mats[] = {
        {"attn_q_a", &rw.wq_a}, {"attn_q_b", &rw.wq_b}, {"attn_kv", &rw.wkv}, {"attn_output_b", &rw.wo_b},
        {"attn_compressor_kv", &rw.comp_kv}, {"attn_compressor_gate", &rw.comp_gate}, {"indexer.attn_k", &rw.idx_k},
        {"indexer.attn_q_b", &rw.idx_q_b}, {"indexer.proj", &rw.idx_proj}};
    for (const auto& [name, mm] : mats) {
        if (!mm->data || (mm == &rw.idx_k && B == 0)) continue;
        const std::vector<float> x = synth::make_input(1, (int) mm->k, (uint32_t) il);
        std::vector<float> y((size_t) mm->rows);
        e::ref_matmul(*mm, x.data(), 1, y.data());
        const char* path = mm->type == e::kRefTypeF16 ? "f16 weights (cuBLAS)" : e::qpath_name(e::qmatmul_path(c.prefill, mm == &rw.idx_k ? B : T));
        ++(*c.dry_paths)[path];
    }
    // attn_output_a per group, as run_call's comparison
    const int64_t gd = (int64_t) c.m.d.n_head * c.m.d.head_dim / c.m.d.o_groups;
    std::vector<float> dr((size_t) gd * c.m.d.o_groups), oa((size_t) c.m.d.o_groups * c.m.d.o_lora);
    e::ref_stage_wo_a(c.m.d, rw, dr.data(), 1, oa.data());
    ++(*c.dry_paths)[rw.wo_a.type == e::kRefTypeF16 ? "f16 weights (cuBLAS)" : e::qpath_name(e::qmatmul_path(c.prefill, T))];
}

// one layer call on the GPU and its stage checks
void run_call(Ctx& c, int il, int64_t T, int64_t p0, uint32_t seed, Report& rep) {
    if (c.dry) {
        dry_call(c, il, T, p0);
        return;
    }
    const e::AttnDims& d = c.m.d;
    const e::AttnLayer& l = c.m.layers[(size_t) il];
    const bool quantized = c.m.wq_a.type != e::kQTypeF16;
    const int D = d.head_dim, H = d.n_head, r = l.ratio, ID = d.idx_dim, IH = d.idx_heads, K = d.top_k;
    const int64_t E = d.n_embd, HD = (int64_t) H * D, OO = (int64_t) d.o_groups * d.o_lora;
    const std::vector<float> x = synth::make_input(T, d.n_embd, seed);
    e::KvLayerBuffers& L = c.kv->layer(il);
    std::vector<float> carry_kv, carry_score;
    if (l.kv_source) {
        carry_kv = download(L.carry_kv, (size_t) r * D);
        carry_score = download(L.carry_score, (size_t) r * D);
    }
    float* x_dev = synth::upload(x);
    float* out_dev = synth::dev_alloc<float>((size_t) (T * E));
    if (T == 1) e::attn_decode(*c.eng, l, c.m.dev[(size_t) il], *c.kv, x_dev, p0, out_dev, c.mode);
    else e::attn_prefill(*c.eng, l, c.m.dev[(size_t) il], *c.kv, x_dev, T, p0, out_dev, c.mode);
    synth::ck(cudaDeviceSynchronize(), "attn_prefill");
    const e::AttnScratch& s = c.eng->scratch();
    const e::RefLayerWeights rw = call_weights(c.m.ref[(size_t) il], T, s.B, c.prefill);
    const std::vector<int64_t> ts = sample(T, 8);
    const int64_t ns = (int64_t) ts.size();
    std::vector<int32_t> pos((size_t) ns);
    std::vector<float> xs((size_t) (ns * E));
    for (int64_t i = 0; i < ns; ++i) {
        pos[(size_t) i] = (int32_t) (p0 + ts[(size_t) i]);
        std::memcpy(xs.data() + i * E, x.data() + ts[(size_t) i] * E, sizeof(float) * (size_t) E);
    }
    // query and window
    const auto g_qr = rows_of(s.qr, d.q_lora, ts, d.q_lora);
    const auto g_q = rows_of(s.q, HD, ts, HD);
    const auto g_kv = rows_of(s.kv, D, ts, D);
    {
        std::vector<float> qr((size_t) (ns * d.q_lora)), q((size_t) (ns * HD)), kvr((size_t) (ns * D));
        e::ref_stage_q(d, l, rw, xs.data(), ns, pos.data(), qr.data(), q.data());
        cmp_rows(rep, "qr", kTolProj, g_qr.data(), qr.data(), ns, d.q_lora);
        e::ref_stage_q_from_qr(d, l, rw, g_qr.data(), ns, pos.data(), q.data());
        cmp_rows(rep, "q (from the GPU's qr)", kTolProj, g_q.data(), q.data(), ns * H, D);
        e::ref_stage_kv(d, l, rw, xs.data(), ns, pos.data(), kvr.data());
        cmp_rows(rep, "kv", kTolProj, g_kv.data(), kvr.data(), ns, D);
        std::vector<int64_t> cells;
        for (int32_t p : pos) cells.push_back(p % L.window_cells);
        const auto win = rows_of(L.window, D, cells, D);
        cmp_exact16(rep, "window cells = f16(kv)", win.data(), g_kv.data(), ns * D);
    }
    // compressor
    if (l.kv_source) {
        const auto g_ckv = rows_of(s.ckv, D, ts, D);
        std::vector<float> ckv((size_t) (ns * D));
        std::vector<double> sc((size_t) (ns * D));
        e::ref_matmul(rw.comp_kv, xs.data(), ns, ckv.data());
        e::ref_matmul_abs_scale(rw.comp_kv, xs.data(), ns, sc.data());
        cmp_scaled(rep, "compressor kv", tol_gemm(rw.comp_kv), g_ckv.data(), ckv.data(), sc.data(), ns * D);
        if (l.gate) {
            const auto g_cs = rows_of(s.cscore, D, ts, D);
            e::ref_matmul(rw.comp_gate, xs.data(), ns, ckv.data());
            e::ref_matmul_abs_scale(rw.comp_gate, xs.data(), ns, sc.data());
            cmp_scaled(rep, "compressor gate", tol_gemm(rw.comp_gate), g_cs.data(), ckv.data(), sc.data(), ns * D);
        }
        const float* cs_dev = l.gate ? s.cscore : s.ckv;
        const int64_t n_carry = p0 % r, first = p0 / r, end = (p0 + T) / r, B = end - first;
        if (s.B != B || s.first_block != first) die("layer " + std::to_string(il) + ": the engine pooled " + std::to_string(s.B) + " blocks from " + std::to_string(s.first_block) + ", expected " + std::to_string(B) + " from " + std::to_string(first));
        auto src_row = [&](int64_t p, bool score, float* dst) {
            if (p < p0) std::memcpy(dst, (score ? carry_score : carry_kv).data() + (p - (p0 - n_carry)) * D, sizeof(float) * (size_t) D);
            else synth::ck(cudaMemcpy(dst, (score ? cs_dev : s.ckv) + (p - p0) * D, sizeof(float) * (size_t) D, cudaMemcpyDeviceToHost), "row");
        };
        for (int64_t b : sample(B, 8)) {
            std::vector<float> gk((size_t) (r * D)), gs((size_t) (r * D)), pooled((size_t) D), latent((size_t) D), rows((size_t) D);
            for (int i = 0; i < r; ++i) {
                src_row((first + b) * r + i, false, gk.data() + i * D);
                src_row((first + b) * r + i, true, gs.data() + i * D);
            }
            e::ref_stage_pool(d, l, rw, e::RefNumerics::Engine, gk.data(), gs.data(), 1, first + b, pooled.data(), latent.data(), rows.data());
            const auto g_p = download(s.pooled + b * D, (size_t) D), g_l = download(s.latent + b * D, (size_t) D),
                       g_r = download(s.rowsf + b * D, (size_t) D);
            cmp_rows(rep, "pooled", kTolElem, g_p.data(), pooled.data(), 1, D);
            cmp_rows(rep, "normed rows", kTolElem, g_l.data(), latent.data(), 1, D);
            cmp_rows(rep, "roped rows", kTolElem, g_r.data(), rows.data(), 1, D);
            const auto cached = download(L.rows + (first + b) * D, (size_t) D);
            cmp_exact16(rep, "compressed rows = f16(roped rows)", cached.data(), g_r.data(), D);
            if (l.key_owner) {
                // the key projection's input: the f32 latent (MMVQ, MMQ) or its f16 (the f16 paths)
                std::vector<float> knew((size_t) ID), kf((size_t) ID), lat((size_t) D);
                if (!f16_input(rw.idx_k)) {
                    lat = g_l;
                } else {
                    const auto g_lh = download(s.latenth + b * D, (size_t) D);
                    for (int i = 0; i < D; ++i) lat[(size_t) i] = kn::f32_from_f16(g_lh[(size_t) i]);
                }
                e::ref_stage_index_keys(d, l, rw, lat.data(), 1, first + b, knew.data(), kf.data());
                const auto g_k = download(s.idx_kf + b * ID, (size_t) ID);
                cmp_rows(rep, "index keys (from the GPU's latent)", kTolProj, g_k.data(), kf.data(), 1, ID);
                const auto ck16 = download(L.keys + (first + b) * ID, (size_t) ID);
                cmp_exact16(rep, "index keys = f16(GPU keys)", ck16.data(), g_k.data(), ID);
            }
        }
        // the carried state: positions end r .. p0 + T - 1
        const auto g_ck = download(L.carry_kv, (size_t) r * D), g_cs = download(L.carry_score, (size_t) r * D);
        for (int64_t p = end * r; p < p0 + T; ++p) {
            std::vector<float> a((size_t) D), b((size_t) D);
            src_row(p, false, a.data());
            src_row(p, true, b.data());
            cmp_exact32(rep, "pooling state", g_ck.data() + (p - end * r) * D, a.data(), D);
            cmp_exact32(rep, "pooling state", g_cs.data() + (p - end * r) * D, b.data(), D);
        }
    }
    // indexer
    std::vector<int32_t> g_ids;
    const int32_t* ids_dev = nullptr;
    if (l.index_source) ids_dev = L.ids;
    else if (r && c.mode == e::ReaderMode::Reuse) ids_dev = c.kv->layer(l.topk_src).ids;
    if (ids_dev) g_ids = rows_of(ids_dev, K, ts, K);
    if (l.index_source) {
        const auto g_iq = rows_of(s.iq, (int64_t) IH * ID, ts, (int64_t) IH * ID);
        const auto g_iw = rows_of(s.iw, IH, ts, IH);
        std::vector<float> iq((size_t) (ns * IH * ID)), iq_rot(iq.size()), iw((size_t) (ns * IH));
        e::ref_stage_index_query(d, rw, g_qr.data(), xs.data(), ns, pos.data(), iq.data(), iq_rot.data(), iw.data());
        cmp_rows(rep, "index queries (from the GPU's qr)", kTolProj, g_iq.data(), iq_rot.data(), ns * IH, ID);
        std::vector<double> sc((size_t) (ns * IH));
        e::ref_matmul_abs_scale(rw.idx_proj, xs.data(), ns, sc.data());
        const double ws = 1.0 / std::sqrt((double) (ID * IH));
        for (double& v : sc) v *= ws;
        cmp_scaled(rep, "index weights", tol_gemm(rw.idx_proj), g_iw.data(), iw.data(), sc.data(), ns * IH);
        // the selection: the reference's top-k of the GPU's own queries, weights and keys
        const int64_t n_k = (p0 + T) / r;
        const auto keys = download(c.kv->layer(l.key_src).keys, (size_t) (n_k * ID));
        for (int64_t i = 0; i < ns; ++i) {
            const int64_t vis = e::indexer_visible(pos[(size_t) i], r, n_k);
            std::vector<float> sco((size_t) vis);
            std::vector<double> scl((size_t) vis);
            const int64_t blk = 4096;
            parallel_for((vis + blk - 1) / blk, [&](int64_t b) {
                const int64_t j0 = b * blk, n = std::min<int64_t>(blk, vis - j0);
                e::indexer_ref_scores(g_iq.data() + i * IH * ID, g_iw.data() + i * IH, IH, keys.data() + j0 * ID, n, T, sco.data() + j0);
                e::indexer_ref_abs_scale(g_iq.data() + i * IH * ID, g_iw.data() + i * IH, IH, keys.data() + j0 * ID, n, scl.data() + j0);
            });
            std::vector<int32_t> rid((size_t) K);
            e::indexer_ref_topk(sco.data(), vis, K, rid.data());
            const int64_t n_sel = std::min<int64_t>(K, vis);
            std::vector<int32_t> gsel(g_ids.begin() + i * K, g_ids.begin() + i * K + n_sel), rsel(rid.begin(), rid.begin() + n_sel);
            bool tail = true;
            for (int64_t k = n_sel; k < K; ++k) tail = tail && g_ids[(size_t) (i * K + k)] == -1;
            ++rep.sets;
            if (gsel == rsel && tail) { ++rep.sets_exact; continue; }
            int32_t thr = rsel.empty() ? 0 : rsel[0];
            for (int32_t j : rsel) if (sco[(size_t) j] < sco[(size_t) thr]) thr = j;
            std::vector<int32_t> diff;
            std::set_symmetric_difference(gsel.begin(), gsel.end(), rsel.begin(), rsel.end(), std::back_inserter(diff));
            bool near = tail && std::is_sorted(gsel.begin(), gsel.end());
            for (int32_t j : diff)
                near = near && j >= 0 && j < vis &&
                       std::fabs((double) sco[(size_t) j] - sco[(size_t) thr]) <= kTolIdx * (scl[(size_t) j] + scl[(size_t) thr]);
            if (near) ++rep.sets_near;
            else ++rep.sets_bad;
        }
    }
    // attention over the cells the engine attends, from the GPU's q and cache
    const auto g_attn = rows_of(s.attn, HD, ts, HD);
    {
        const e::KvLayerBuffers* V = r ? &c.kv->layer(l.kv_src) : nullptr;
        const int64_t n_rows = V ? (p0 + T) / r : 0;
        for (int64_t i = 0; i < ns; ++i) {
            const int64_t p = pos[(size_t) i];
            std::vector<int64_t> wcells, crows;
            for (int64_t q = std::max<int64_t>(0, p - d.n_swa + 1); q <= p; ++q) wcells.push_back(q % L.window_cells);
            if (r) {
                const int64_t vis = std::min<int64_t>(n_rows, (p + 1) / r);
                if (ids_dev) {
                    for (int k = 0; k < K; ++k) {
                        const int32_t j = g_ids[(size_t) (i * K + k)];
                        if (j >= 0 && j < vis) crows.push_back(j);
                    }
                } else {
                    for (int64_t j = 0; j < vis; ++j) crows.push_back(j);
                }
            }
            const auto win = rows_of(L.window, D, wcells, D);
            std::vector<uint16_t> comp;
            if (!crows.empty()) comp = rows_of(V->rows, D, crows, D);
            std::vector<const uint16_t*> keys;
            for (size_t k = 0; k < wcells.size(); ++k) keys.push_back(win.data() + k * D);
            for (size_t k = 0; k < crows.size(); ++k) keys.push_back(comp.data() + k * D);
            std::vector<float> att((size_t) HD);
            std::vector<double> scl((size_t) HD);
            e::ref_stage_attend(d, e::RefNumerics::Engine, g_q.data() + i * HD, keys.data(), nullptr, (int64_t) keys.size(), rw.sinks, T,
                                att.data(), nullptr, nullptr);
            e::ref_stage_attend_scale(d, g_q.data() + i * HD, keys.data(), (int64_t) keys.size(), rw.sinks, scl.data());
            const std::string stage = std::string(T > 1 ? "attention (prefill" : "attention (decode") +
                                      (s.n_splits > 1 ? ", split, merged)" : ")");
            cmp_scaled(rep, stage, kTolAttn, g_attn.data() + i * HD, att.data(), scl.data(), HD);
            double& worst = T > 1 ? g_attn_worst.prefill : g_attn_worst.decode;
            worst = std::max(worst, rep.st[stage].worst);
        }
    }
    // output: de-rope, wo_a, wo_b, each from the GPU's input
    {
        const auto g_dr = rows_of(s.derope, HD, ts, HD);
        const auto g_oa = rows_of(s.oa, OO, ts, OO);
        const auto g_out = rows_of(out_dev, E, ts, E);
        std::vector<float> dr((size_t) (ns * HD)), oa((size_t) (ns * OO)), out((size_t) (ns * E)), drf((size_t) (ns * HD));
        e::ref_stage_output(d, l, rw, g_attn.data(), ns, pos.data(), dr.data(), oa.data(), out.data());
        for (size_t i = 0; i < drf.size(); ++i) drf[i] = kn::f32_from_f16(g_dr[i]);
        cmp_rows(rep, "de-rope (f16)", kTolDerope, drf.data(), dr.data(), ns * H, D);
        e::ref_stage_wo_a(d, rw, drf.data(), ns, oa.data());
        std::vector<double> sc((size_t) (ns * OO));
        {
            const int64_t gd = HD / d.o_groups;
            std::vector<float> xg((size_t) (ns * gd));
            std::vector<double> sg((size_t) (ns * d.o_lora));
            for (int g = 0; g < d.o_groups; ++g) {
                for (int64_t t = 0; t < ns; ++t) std::memcpy(xg.data() + t * gd, drf.data() + t * HD + g * gd, sizeof(float) * (size_t) gd);
                e::RefMatrix wg = rw.wo_a;
                wg.rows = d.o_lora;
                const size_t row_bytes = quantized ? (size_t) (gd / e::kQK) * e::qtype_block_bytes(wg.type) : (size_t) gd * 2;
                wg.data = static_cast<const uint8_t*>(rw.wo_a.data) + (size_t) g * d.o_lora * row_bytes;
                e::ref_matmul_abs_scale(wg, xg.data(), ns, sg.data());
                for (int64_t t = 0; t < ns; ++t)
                    for (int o = 0; o < d.o_lora; ++o) sc[(size_t) (t * OO + (int64_t) g * d.o_lora + o)] = sg[(size_t) (t * d.o_lora + o)];
            }
        }
        cmp_scaled(rep, "wo_a (from the GPU's de-rope)", tol_gemm(rw.wo_a), g_oa.data(), oa.data(), sc.data(), ns * OO);
        std::vector<double> so((size_t) (ns * E));
        e::ref_matmul(rw.wo_b, g_oa.data(), ns, out.data());
        e::ref_matmul_abs_scale(rw.wo_b, g_oa.data(), ns, so.data());
        cmp_scaled(rep, "wo_b (from the GPU's wo_a)", tol_gemm(rw.wo_b), g_out.data(), out.data(), so.data(), ns * E);
    }
    cudaFree(x_dev);
    cudaFree(out_dev);
}

// one chunk (or decode step) through every layer in order
void run_chunk(Ctx& c, int64_t T, int64_t p0, uint32_t seed, Report& rep) {
    for (size_t il = 0; il < c.m.layers.size(); ++il) run_call(c, (int) il, T, p0, seed + (uint32_t) il * 7919u, rep);
}

// the projection path of a case: f16 weights take the cuBLAS test path whatever the prefill choice
const char* path_name(int wtype, e::QPrefill p) {
    if (wtype == e::kQTypeF16) return "the cuBLAS test path";
    return p == e::QPrefill::Mmq ? "MMQ prefill" : "dequant + cuBLAS prefill";
}

bool tiny_case(const char* name, const e::AttnDims& d, uint32_t seed, int wtype, e::QPrefill prefill, bool dry,
               std::map<std::string, int64_t>& paths) {
    bool ok = true;
    synth::Model m;
    synth::make_model(m, d, seed, wtype, !dry);
    for (e::ReaderMode mode : {e::ReaderMode::Reuse, e::ReaderMode::Dense}) {
        e::AttnEngine eng;
        e::KvCache kv;
        if (!dry) {
            std::string err;
            if (!eng.init(d, 64, 2048, nullptr, err, prefill)) die(err);
            eng.set_attn_output(true);   // the attention stage reads the f32 output
            if (!kv.init(d, m.layers, e::KvCacheConfig{2048, 64}, err)) die(err);
        }
        Ctx c{m, dry ? nullptr : &eng, dry ? nullptr : &kv, mode, prefill, dry, &paths};
        Report rep;
        run_chunk(c, 39, 0, seed + 1, rep);
        run_chunk(c, 25, 39, seed + 2, rep);
        for (int s = 0; s < 8; ++s) run_chunk(c, 1, 64 + s, seed + 10 + (uint32_t) s, rep);
        char title[192];
        std::snprintf(title, sizeof title, "%s, %s weights (%s), six roles, a 64-token prompt in chunks of 39 + 25 then 8 decode steps, %s",
                      name, e::qtype_name(wtype), path_name(wtype, prefill), mode == e::ReaderMode::Reuse ? "Reuse Mode" : "--readers dense");
        if (dry) {
            std::printf("dry  %s: 60 calls checked\n", title);
            continue;
        }
        rep.print(title);
        ok = ok && rep.ok();
    }
    return ok;
}

bool real_cases(bool quick, int wtype, e::QPrefill prefill, bool dry, std::map<std::string, int64_t>& paths) {
    const e::AttnDims d = e::attn_dims_real();
    synth::Model m;
    synth::make_model(m, d, 1234, wtype, !dry);
    e::AttnEngine eng;
    e::KvCache kv;
    const int64_t max_ctx = 262144;
    if (!dry) {
        std::string err;
        if (!eng.init(d, 4096, max_ctx, nullptr, err, prefill)) die(err);
        eng.set_attn_output(true);
        if (!kv.init(d, m.layers, e::KvCacheConfig{max_ctx, 4096}, err)) die(err);
        std::printf("real dims: engine workspace %.1f MiB, KV cache %.1f MiB for 6 layers at %lld positions\n",
                    (double) eng.workspace_bytes() / (1 << 20), (double) kv.bytes().total() / (1 << 20), (long long) max_ctx);
    }
    bool ok = true;
    struct Case { const char* name; int64_t hist, T; e::ReaderMode mode; };
    std::vector<Case> cases = {
        {"decode at 4K", 4095, 1, e::ReaderMode::Reuse},
        {"decode at 16K", 16383, 1, e::ReaderMode::Reuse},
        {"decode at 64K", 65535, 1, e::ReaderMode::Reuse},
        {"decode at 262,144", 262143, 1, e::ReaderMode::Reuse},
        {"decode at 16K, --readers dense", 16383, 1, e::ReaderMode::Dense},
        {"prefill 4096 at offset 0", 0, 4096, e::ReaderMode::Reuse},
        {"prefill 4096 at offset 16K", 16384, 4096, e::ReaderMode::Reuse},
    };
    if (quick) cases.resize(1);
    uint32_t seed = 77;
    for (const Case& cs : cases) {
        if (!dry) {
            kv.reset();
            if (cs.hist > 0) synth::write_history(kv, cs.hist, seed);
        }
        Ctx c{m, dry ? nullptr : &eng, dry ? nullptr : &kv, cs.mode, prefill, dry, &paths};
        Report rep;
        run_chunk(c, cs.T, cs.hist, seed, rep);
        if (dry) {
            std::printf("dry  real dims, %s weights (%s), six roles, %s: 6 calls checked\n", e::qtype_name(wtype), path_name(wtype, prefill), cs.name);
            seed += 101;
            continue;
        }
        char title[192];
        std::snprintf(title, sizeof title, "real dims (64 heads x 512, top-k 512), %s weights (%s), six roles, %s (attention splits %d)",
                      e::qtype_name(wtype), path_name(wtype, prefill), cs.name, eng.scratch().n_splits);
        rep.print(title);
        ok = ok && rep.ok();
        seed += 101;
    }
    return ok;
}

}  // namespace

int run_gpu(bool quick, bool f16, bool mmq, bool dry) {
    if (!dry) {
        int dev_id = 0;
        cudaDeviceProp p{};
        synth::ck(cudaGetDevice(&dev_id), "cudaGetDevice");
        synth::ck(cudaGetDeviceProperties(&p, dev_id), "cudaGetDeviceProperties");
        std::printf("device: %s, sm_%d%d, %.1f GiB\n", p.name, p.major, p.minor, (double) p.totalGlobalMem / (1 << 30));
    }
    bool ok = true;
    const int wtype = f16 ? e::kQTypeF16 : e::kQTypeQ2K;
    const e::QPrefill prefill = mmq ? e::QPrefill::Mmq : e::QPrefill::DequantF16;
    std::map<std::string, int64_t> paths;
    ok = tiny_case("tiny-a dims (4 heads x 256, top-k 8, window 16)", synth::dims_tiny_a(), 11, wtype, prefill, dry, paths) && ok;
    ok = tiny_case("tiny-b dims (8 heads x 512, top-k 16, window 16)", synth::dims_tiny_b(), 12, wtype, prefill, dry, paths) && ok;
    ok = real_cases(quick, wtype, prefill, dry, paths) && ok;
    if (dry) {
        std::printf("dry run: every call's weight checks and reference numerics passed; matrices run by the reference per path:");
        for (const auto& [path, n] : paths) std::printf(" %s %lld,", path.c_str(), (long long) n);
        std::printf("\ndsv41_engine_attention_test (b) --dry-run: PASS\n");
        return 0;
    }
    std::printf("attention margin: prefill_attn_kernel (p in f16) worst %.3g of sum p |v|, %.0f%% of the tolerance %.3g; "
                "attn_kernel (decode, p hi/lo) worst %.3g, %.0f%%\n",
                g_attn_worst.prefill, 100 * g_attn_worst.prefill / kTolAttn, kTolAttn, g_attn_worst.decode,
                100 * g_attn_worst.decode / kTolAttn);
    std::printf("dsv41_engine_attention_test (b): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
