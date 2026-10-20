// tests/dsv41/engine/indexer_test.cpp - parity of the lightning indexer (include/strata/dsv41/engine/indexer.hpp).
//
//   dsv41_engine_indexer_test --cases FILE [FILE ...]   (a) GPU free: the CPU reference against the stock fork's
//                                                       op dumps (lists of tools/dsv41/engine/indexer_cases.py)
//   dsv41_engine_indexer_test --gpu [--quick]           (b) CUDA against the CPU reference on random inputs
//
// (a) For every case (index source layer, ubatch) and query t at position pos0 + t: the dump's idx_score must be
// -inf exactly on the rows the compressed causal mask hides (j >= (pos + 1) / ratio), the reference must give the
// dump's scores on the visible rows bit for bit (tolerance 0: SPEC.md "Numerics", the reference reproduces the fork's
// CPU order), and its top-k must be the dump's idx_top_k restricted to the visible rows as a set, where a differing
// row is accepted only when its score equals the selection threshold exactly (a tie the fork's std::partial_sort broke
// another way; ours goes to the lowest index).
// (b) Per case (dims, keys, queries): both scorers on sampled queries against the reference, |gpu - ref| <= kTol *
// scale (scale = the sum of the absolute terms, indexer_ref_abs_scale); the top-k kernel on the GPU's own scores
// equal to indexer_ref_topk's ids exactly; indexer_select's ids (every query, chunked) against the reference's, a
// differing row accepted only when its reference score is within the tolerance of the threshold; and the top-k on
// tied and equal scores equal to the reference's exactly.
#include "strata/dsv41/engine/indexer.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace e = strata::dsv41::engine;
namespace kn = strata::kernels;

namespace {

constexpr int D = e::kIndexerDim;
// 2 * (dims + heads) * 2^-24 at 32 heads is 1.9e-5 of the scale: the worst case of two summation orders
constexpr double kTol = 2e-5;

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "dsv41_engine_indexer_test: %s\n", m.c_str());
    std::exit(2);
}

template <typename T> std::vector<T> read_raw(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    f.seekg(0, std::ios::end);
    const size_t bytes = (size_t) f.tellg();
    if (bytes != count * sizeof(T))
        die(path + ": " + std::to_string(bytes) + " bytes, expected " + std::to_string(count * sizeof(T)));
    f.seekg(0);
    std::vector<T> v(count);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) bytes);
    return v;
}

uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }

// ---------------------------------------------------------------- (a) dumps
struct DumpTotals {
    int64_t cases = 0, queries = 0, decode_queries = 0, rows = 0, rows_exact = 0, masked = 0, sets = 0, sets_exact = 0,
            sets_tied = 0, diffs_shown = 0;
    double max_abs = 0, max_rel = 0;
};

bool run_dump_case(const std::string& line, DumpTotals& tot) {
    std::istringstream in(line);
    std::string tag, fq, fk, fw, fs, ft;
    int layer, ubatch, ratio, n_heads, dim, top_k;
    int64_t pos0, nt, n_comp;
    in >> tag >> layer >> ubatch >> pos0 >> nt >> ratio >> n_heads >> dim >> top_k >> n_comp >> fq >> fk >> fw >> fs >> ft;
    if (!in || tag != "case") die("bad case line: " + line);
    if (dim != D) die("layer " + std::to_string(layer) + ": indexer dim " + std::to_string(dim) + ", the engine has 128");
    const int k_out = (int) std::min<int64_t>(n_comp, top_k);
    const auto q = read_raw<float>(fq, (size_t) (nt * n_heads * D));
    const auto keys = read_raw<uint16_t>(fk, (size_t) (n_comp * D));
    const auto w = read_raw<float>(fw, (size_t) (nt * n_heads));
    const auto score = read_raw<float>(fs, (size_t) (nt * n_comp));
    const auto topk = read_raw<int32_t>(ft, (size_t) (nt * k_out));
    std::vector<float> ref((size_t) n_comp);
    std::vector<int32_t> ids((size_t) top_k);
    bool ok = true;
    char where[96];
    for (int64_t t = 0; t < nt; ++t) {
        const int64_t pos = pos0 + t, vis = e::indexer_visible(pos, ratio, n_comp);
        std::snprintf(where, sizeof where, "L%d u%d t%lld (pos %lld, %lld visible)", layer, ubatch, (long long) t,
                      (long long) pos, (long long) vis);
        const float* ds = score.data() + t * n_comp;
        for (int64_t j = 0; j < n_comp; ++j) {
            const bool hidden = std::isinf(ds[j]) && ds[j] < 0;
            if (hidden != (j >= vis)) {
                std::printf("FAIL %s: idx_score row %lld is %g, the mask formula says %s\n", where, (long long) j,
                            ds[j], j >= vis ? "hidden" : "visible");
                ok = false;
                break;
            }
        }
        tot.masked += n_comp - vis;
        e::indexer_ref_scores(q.data() + t * n_heads * D, w.data() + t * n_heads, n_heads, keys.data(), vis, nt, ref.data());
        for (int64_t j = 0; j < vis; ++j) {
            if (!std::isfinite(ref[j])) {
                std::printf("FAIL %s: reference score of row %lld is %g\n", where, (long long) j, ref[j]);
                ok = false;
            }
            const double a = std::fabs((double) ref[j] - (double) ds[j]);
            tot.max_abs = std::max(tot.max_abs, a);
            tot.max_rel = std::max(tot.max_rel, a / std::max(1e-30, std::fabs((double) ds[j])));
            if (bits(ref[j]) == bits(ds[j])) ++tot.rows_exact;
            else if (tot.diffs_shown++ < 8)
                std::printf("diff %s: row %lld reference %.9g (0x%08x) dump %.9g (0x%08x)\n", where, (long long) j, ref[j],
                            bits(ref[j]), ds[j], bits(ds[j]));
        }
        tot.rows += vis;
        // the selection, against the fork's set restricted to the visible rows, on the fork's own scores
        e::indexer_ref_topk(ds, vis, top_k, ids.data());
        const int64_t n_sel = std::min<int64_t>(top_k, vis);
        std::vector<int32_t> mine(ids.begin(), ids.begin() + n_sel), fork;
        for (int i = 0; i < k_out; ++i)
            if (topk[(size_t) (t * k_out + i)] < vis) fork.push_back(topk[(size_t) (t * k_out + i)]);
        std::sort(fork.begin(), fork.end());
        ++tot.sets;
        if ((int64_t) fork.size() != n_sel) {
            std::printf("FAIL %s: the fork selected %zu visible rows, expected %lld\n", where, fork.size(), (long long) n_sel);
            ok = false;
        } else if (mine == fork) {
            ++tot.sets_exact;
        } else {
            float thr = INFINITY;
            for (int32_t j : mine) thr = std::min(thr, ds[j]);
            std::vector<int32_t> diff;
            std::set_symmetric_difference(mine.begin(), mine.end(), fork.begin(), fork.end(), std::back_inserter(diff));
            bool tied = true;
            for (int32_t j : diff) tied = tied && ds[j] == thr;
            if (tied) {
                ++tot.sets_tied;
            } else {
                std::printf("FAIL %s: top-k differs beyond ties (%zu rows)\n", where, diff.size());
                ok = false;
            }
        }
        ++tot.queries;
        if (nt == 1) ++tot.decode_queries;
    }
    ++tot.cases;
    return ok;
}

// one group per "# DUMP: MODEL" header of a case list (indexer_cases.py writes one per dump directory)
bool report_group(const std::string& header, DumpTotals& tot, bool ok) {
    ok = ok && tot.cases > 0 && tot.rows_exact == tot.rows;
    std::printf("%s %s\n  %lld cases (layer x ubatch), %lld queries (%lld in decode ubatches); masked rows -inf as "
                "the formula says: %lld\n  scores: %lld/%lld visible rows bit-identical, max abs err %.3g, max rel err "
                "%.3g (tolerance 0)\n  top-k: %lld/%lld sets identical, %lld differ only in ties at the threshold, "
                "%lld otherwise\n",
                ok ? "PASS" : "FAIL", header.c_str(), (long long) tot.cases, (long long) tot.queries,
                (long long) tot.decode_queries, (long long) tot.masked, (long long) tot.rows_exact, (long long) tot.rows,
                tot.max_abs, tot.max_rel, (long long) tot.sets_exact, (long long) tot.sets, (long long) tot.sets_tied,
                (long long) (tot.sets - tot.sets_exact - tot.sets_tied));
    tot = DumpTotals{};
    return ok;
}

int run_dumps(const std::vector<std::string>& files) {
    bool all_ok = true;
    int64_t groups = 0;
    for (const auto& path : files) {
        std::ifstream f(path);
        if (!f) die("cannot open " + path);
        DumpTotals tot;
        std::string line, header;
        bool ok = true;
        while (std::getline(f, line)) {
            if (line.rfind("# ", 0) == 0) {
                if (!header.empty()) { all_ok = report_group(header, tot, ok) && all_ok; ++groups; }
                header = line.substr(2);
                ok = true;
                continue;
            }
            if (line.empty()) continue;
            if (header.empty()) die(path + ": a case line before any '# DUMP: MODEL' header");
            ok = run_dump_case(line, tot) && ok;
        }
        if (header.empty()) die(path + ": no case lines");
        all_ok = report_group(header, tot, ok) && all_ok;
        ++groups;
    }
    std::printf("dsv41_engine_indexer_test (a): %s (%lld dumps)\n", all_ok ? "PASS" : "FAIL", (long long) groups);
    return all_ok ? 0 : 1;
}

// ---------------------------------------------------------------- (b) GPU
void ck(cudaError_t r, const char* w) {
    if (r != cudaSuccess) die(std::string(w) + ": " + cudaGetErrorString(r));
}
template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    return p;
}
template <typename T> T* up(const std::vector<T>& h) {
    T* p = dev<T>(h.size());
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <typename T> std::vector<T> down(const T* p, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

void parallel_for(int64_t n, const std::function<void(int64_t)>& f) {
    const int nt = (int) std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < nt; ++i)
        th.emplace_back([&] { for (int64_t k; (k = next++) < n;) f(k); });
    for (auto& t : th) t.join();
}

struct GpuCase {
    const char* name;
    int n_heads, top_k, ratio;
    int64_t n_k, nq, pos0;
    size_t scratch_cap;
};

// keys with a shared direction plus noise (the scores spread as a trained indexer's do), queries along it, weights
// of both signs at the fork's 1 / sqrt(dim * heads) scale
struct Inputs {
    std::vector<float> q, w;
    std::vector<uint16_t> keys;
    std::vector<int32_t> pos;
};
Inputs make_inputs(const GpuCase& c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    Inputs in;
    std::vector<float> dir(D);
    for (auto& x : dir) x = nd(rng);
    in.keys.resize((size_t) (c.n_k * D));
    for (int64_t j = 0; j < c.n_k; ++j) {
        const float a = nd(rng);
        for (int d = 0; d < D; ++d) in.keys[(size_t) (j * D + d)] = kn::f16_from_f32(0.25f * (a * dir[d] + nd(rng)));
    }
    in.q.resize((size_t) (c.nq * c.n_heads * D));
    for (size_t i = 0; i < in.q.size(); ++i) in.q[i] = 0.2f * dir[i % D] + 0.5f * nd(rng);
    in.w.resize((size_t) (c.nq * c.n_heads));
    const float ws = 1.0f / std::sqrt((float) (D * c.n_heads));
    for (auto& x : in.w) x = ws * (0.5f + nd(rng));
    in.pos.resize((size_t) c.nq);
    for (int64_t i = 0; i < c.nq; ++i) in.pos[(size_t) i] = (int32_t) (c.pos0 + i);
    return in;
}

bool run_gpu_case(const GpuCase& c) {
    const e::IndexerDims d{c.n_heads, c.top_k, c.ratio};
    const Inputs in = make_inputs(c, (uint32_t) (c.n_k * 7 + c.nq * 13 + c.n_heads));
    const e::IndexerPlan plan = e::indexer_plan(c.nq, c.n_k, c.scratch_cap);
    float *d_q = up(in.q), *d_w = up(in.w);
    uint16_t* d_k = up(in.keys);
    int32_t* d_pos = up(in.pos);
    float* d_scratch = dev<float>(plan.scratch_bytes / sizeof(float));
    int32_t *d_ids_auto = dev<int32_t>((size_t) (c.nq * c.top_k)), *d_ids_simt = dev<int32_t>((size_t) (c.nq * c.top_k));
    const e::IndexerScorer auto_is = e::indexer_resolve(e::IndexerScorer::Auto, c.nq);
    e::indexer_select(d_q, d_w, d_pos, c.nq, d_k, c.n_k, d, plan, d_scratch, e::IndexerScorer::Auto, d_ids_auto, nullptr);
    e::indexer_select(d_q, d_w, d_pos, c.nq, d_k, c.n_k, d, plan, d_scratch, e::IndexerScorer::Simt, d_ids_simt, nullptr);
    ck(cudaDeviceSynchronize(), "select");
    const auto ids_auto = down(d_ids_auto, (size_t) (c.nq * c.top_k)), ids_simt = down(d_ids_simt, (size_t) (c.nq * c.top_k));

    // the sampled queries: all of a small call, else the first, the last and 6 between
    std::vector<int64_t> sample;
    if (c.nq <= 16) for (int64_t i = 0; i < c.nq; ++i) sample.push_back(i);
    else for (int i = 0; i < 8; ++i) sample.push_back((int64_t) i * (c.nq - 1) / 7);
    const int64_t ns = (int64_t) sample.size();
    std::vector<float> sq((size_t) (ns * c.n_heads * D)), sw((size_t) (ns * c.n_heads));
    std::vector<int32_t> spos((size_t) ns);
    for (int64_t s = 0; s < ns; ++s) {
        const int64_t i = sample[(size_t) s];
        std::copy_n(in.q.begin() + i * c.n_heads * D, c.n_heads * D, sq.begin() + s * c.n_heads * D);
        std::copy_n(in.w.begin() + i * c.n_heads, c.n_heads, sw.begin() + s * c.n_heads);
        spos[(size_t) s] = in.pos[(size_t) i];
    }
    float *d_sq = up(sq), *d_sw = up(sw);
    int32_t* d_spos = up(spos);
    const int64_t ld = plan.ld;
    float* d_ss = dev<float>((size_t) (ns * ld));
    int32_t* d_sids = dev<int32_t>((size_t) (ns * c.top_k));
    std::vector<int64_t> vis((size_t) ns);
    for (int64_t s = 0; s < ns; ++s) vis[(size_t) s] = e::indexer_visible(spos[(size_t) s], c.ratio, c.n_k);

    // the reference on the sampled queries
    std::vector<std::vector<float>> ref((size_t) ns);
    std::vector<std::vector<double>> scale((size_t) ns);
    std::vector<std::vector<int32_t>> ref_ids((size_t) ns, std::vector<int32_t>((size_t) c.top_k));
    parallel_for(ns, [&](int64_t s) {
        const int64_t v = vis[(size_t) s];
        ref[(size_t) s].resize((size_t) v);
        scale[(size_t) s].resize((size_t) v);
        e::indexer_ref_scores(sq.data() + s * c.n_heads * D, sw.data() + s * c.n_heads, c.n_heads, in.keys.data(), v,
                              c.nq, ref[(size_t) s].data());
        e::indexer_ref_abs_scale(sq.data() + s * c.n_heads * D, sw.data() + s * c.n_heads, c.n_heads, in.keys.data(), v,
                                 scale[(size_t) s].data());
        e::indexer_ref_topk(ref[(size_t) s].data(), v, c.top_k, ref_ids[(size_t) s].data());
    });

    bool ok = true;
    std::printf("%s: heads %d, top-k %d, ratio %d, %lld keys, %lld queries from position %lld; plan %lld queries x %lld "
                "per pass (%zu bytes of scratch); auto scorer %s\n",
                c.name, c.n_heads, c.top_k, c.ratio, (long long) c.n_k, (long long) c.nq, (long long) c.pos0,
                (long long) plan.q_chunk, (long long) ld, plan.scratch_bytes,
                auto_is == e::IndexerScorer::TensorCore ? "tensor cores" : "fp32");
    const e::IndexerScorer scorers[2] = {e::IndexerScorer::Simt, e::IndexerScorer::TensorCore};
    for (const auto which : scorers) {
        const char* sname = which == e::IndexerScorer::Simt ? "fp32" : "tensor cores";
        ck(cudaMemset(d_ss, 0xff, (size_t) (ns * ld) * 4), "memset");   // NaN: a row the scorer skips shows up
        e::indexer_scores(d_sq, d_sw, d_spos, ns, d_k, c.n_k, d, d_ss, ld, which, nullptr);
        e::indexer_topk(d_ss, ld, d_spos, ns, c.n_k, d, d_sids, nullptr);
        ck(cudaDeviceSynchronize(), "scores");
        const auto gs = down(d_ss, (size_t) (ns * ld));
        const auto gi = down(d_sids, (size_t) (ns * c.top_k));
        double max_err = 0, max_abs = 0;
        int64_t over = 0, rows = 0, topk_same = 0;
        std::vector<int32_t> own((size_t) c.top_k);
        for (int64_t s = 0; s < ns; ++s) {
            const int64_t v = vis[(size_t) s];
            const float* g = gs.data() + s * ld;
            for (int64_t j = 0; j < v; ++j) {
                const double err = std::fabs((double) g[j] - (double) ref[(size_t) s][(size_t) j]);
                const double sc = scale[(size_t) s][(size_t) j];
                max_abs = std::max(max_abs, err);
                if (sc > 0) max_err = std::max(max_err, err / sc);
                if (!(err <= kTol * sc)) ++over;   // NaN counts as over
            }
            rows += v;
            e::indexer_ref_topk(g, v, c.top_k, own.data());
            topk_same += std::equal(own.begin(), own.end(), gi.begin() + s * c.top_k);
        }
        const bool sok = over == 0 && topk_same == ns;
        ok = ok && sok;
        std::printf("  %s %s scores on %lld sampled queries (%lld rows): max |err| %.3g, max |err| / scale %.3g "
                    "(tolerance %.1g), %lld over; top-k kernel on its own scores identical to the reference rule %lld/%lld\n",
                    sok ? "ok  " : "FAIL", sname, (long long) ns, (long long) rows, max_abs, max_err, kTol, (long long) over,
                    (long long) topk_same, (long long) ns);
    }
    // indexer_select's ids (chunked, every query) on the sampled queries against the reference's selection
    for (int pass = 0; pass < 2; ++pass) {
        const auto& ids = pass == 0 ? ids_auto : ids_simt;
        int64_t exact = 0, explained = 0, unexplained = 0, agree = 0, total = 0;
        for (int64_t s = 0; s < ns; ++s) {
            const int64_t i = sample[(size_t) s], v = vis[(size_t) s], n_sel = std::min<int64_t>(c.top_k, v);
            std::vector<int32_t> g(ids.begin() + i * c.top_k, ids.begin() + i * c.top_k + n_sel);
            std::vector<int32_t> r(ref_ids[(size_t) s].begin(), ref_ids[(size_t) s].begin() + n_sel);
            bool tail_ok = true;
            for (int64_t k = n_sel; k < c.top_k; ++k) tail_ok = tail_ok && ids[(size_t) (i * c.top_k + k)] == -1;
            std::vector<int32_t> common;
            std::set_intersection(g.begin(), g.end(), r.begin(), r.end(), std::back_inserter(common));
            agree += (int64_t) common.size();
            total += n_sel;
            if (g == r && tail_ok) { ++exact; continue; }
            // a differing row is a near-tie: its reference score within the tolerance of the reference threshold
            const auto& rs = ref[(size_t) s];
            const auto& sc = scale[(size_t) s];
            int32_t thr_j = r.empty() ? 0 : r[0];
            for (int32_t j : r) if (rs[(size_t) j] < rs[(size_t) thr_j]) thr_j = j;
            std::vector<int32_t> diff;
            std::set_symmetric_difference(g.begin(), g.end(), r.begin(), r.end(), std::back_inserter(diff));
            bool near = tail_ok && std::is_sorted(g.begin(), g.end());
            for (int32_t j : diff)
                near = near && j >= 0 && j < v &&
                       std::fabs((double) rs[(size_t) j] - (double) rs[(size_t) thr_j]) <=
                           kTol * (sc[(size_t) j] + sc[(size_t) thr_j]);
            if (near) ++explained;
            else ++unexplained;
        }
        ok = ok && unexplained == 0;
        std::printf("  %s indexer_select (%s, %lld passes) ids vs the reference's on the sampled queries: %lld identical, "
                    "%lld differ only in near-ties, %lld otherwise; rows in common %lld/%lld (%.4f%%)\n",
                    unexplained == 0 ? "ok  " : "FAIL",
                    pass == 0 ? (auto_is == e::IndexerScorer::TensorCore ? "auto = tensor cores" : "auto = fp32") : "fp32",
                    (long long) ((c.nq + plan.q_chunk - 1) / plan.q_chunk), (long long) exact, (long long) explained,
                    (long long) unexplained, (long long) agree, (long long) total, total ? 100.0 * agree / total : 100.0);
    }
    cudaFree(d_q); cudaFree(d_w); cudaFree(d_k); cudaFree(d_pos); cudaFree(d_scratch); cudaFree(d_ids_auto);
    cudaFree(d_ids_simt); cudaFree(d_sq); cudaFree(d_sw); cudaFree(d_spos); cudaFree(d_ss); cudaFree(d_sids);
    return ok;
}

// the top-k kernel alone on continuous, 64-level and all-equal scores (with a few NaN and -0/+0 among them): the
// reference rule exactly, at sizes on both sides of the per-warp rows
bool run_topk_ties(int64_t n_k, int top_k, int64_t nq) {
    const e::IndexerDims d{1, top_k, 1};
    const int64_t ld = (n_k + 31) / 32 * 32;
    std::mt19937 rng((uint32_t) (n_k + top_k));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int32_t> pos((size_t) nq);
    for (int64_t i = 0; i < nq; ++i) pos[(size_t) i] = (int32_t) (n_k - 1 - i * (n_k / (2 * nq)));   // shrinking rows
    int32_t* d_pos = up(pos);
    float* d_s = dev<float>((size_t) (nq * ld));
    int32_t* d_ids = dev<int32_t>((size_t) (nq * top_k));
    const char* names[3] = {"continuous", "64 levels", "equal"};
    bool ok = true;
    std::printf("top-k on synthetic scores, %lld rows, top-k %d, %lld queries:", (long long) n_k, top_k, (long long) nq);
    for (int set = 0; set < 3; ++set) {
        std::vector<float> s((size_t) (nq * ld), 0.f);
        for (auto& x : s) {
            const float r = nd(rng);
            x = set == 0 ? r : set == 1 ? std::floor(r * 8.f) / 8.f : 0.25f;
        }
        s[0] = NAN;
        s[1] = -0.0f;
        s[2] = 0.0f;
        if (nq > 1) s[(size_t) ld + 3] = -INFINITY;
        ck(cudaMemcpy(d_s, s.data(), s.size() * 4, cudaMemcpyHostToDevice), "upload");
        e::indexer_topk(d_s, ld, d_pos, nq, n_k, d, d_ids, nullptr);
        ck(cudaDeviceSynchronize(), "topk");
        const auto g = down(d_ids, (size_t) (nq * top_k));
        std::vector<int32_t> r((size_t) top_k);
        int64_t same = 0;
        for (int64_t i = 0; i < nq; ++i) {
            e::indexer_ref_topk(s.data() + i * ld, e::indexer_visible(pos[(size_t) i], 1, n_k), top_k, r.data());
            same += std::equal(r.begin(), r.end(), g.begin() + i * top_k);
        }
        ok = ok && same == nq;
        std::printf(" %s %lld/%lld", names[set], (long long) same, (long long) nq);
    }
    std::printf(" identical: %s\n", ok ? "ok" : "FAIL");
    cudaFree(d_pos); cudaFree(d_s); cudaFree(d_ids);
    return ok;
}

int run_gpu(bool quick) {
    int dev_id = 0;
    cudaDeviceProp p{};
    ck(cudaGetDevice(&dev_id), "cudaGetDevice");
    ck(cudaGetDeviceProperties(&p, dev_id), "cudaGetDeviceProperties");
    std::printf("device: %s, sm_%d%d, %.1f GiB\n", p.name, p.major, p.minor, (double) p.totalGlobalMem / (1 << 30));
    const size_t cap = e::kIndexerScratchCap;
    const int H = e::kIndexerHeads, K = e::kIndexerTopK;
    // nq 4096 needs at least 2048 rows at ratio 2, so the 1K-key prefill case is 2048 queries from position 0; the
    // 16K case with a small cap checks uneven passes (1000 queries per pass)
    std::vector<GpuCase> cases = {
        {"tiny-a decode", 4, 8, 2, 256, 1, 511, cap},
        {"tiny-a prefill", 4, 8, 2, 256, 64, 0, cap},
        {"tiny-b decode", 8, 16, 1, 256, 1, 255, cap},
        {"tiny-b prefill", 8, 16, 1, 256, 64, 192, cap},
        {"real 1K decode", H, K, 2, 1024, 1, 2047, cap},
        {"real 1K prefill", H, K, 2, 1024, 2048, 0, cap},
        {"real 16K decode", H, K, 2, 16384, 1, 32767, cap},
        {"real 16K prefill", H, K, 2, 16384, 4096, 32768 - 4096, cap},
        {"real 16K prefill, 1000 per pass", H, K, 2, 16384, 4096, 32768 - 4096, (size_t) 1000 * 16384 * 4},
        {"real 4K fresh prompt (causal)", H, K, 1, 4096, 4096, 0, cap},
        {"real 131K decode", H, K, 2, 131072, 1, 262143, cap},
        {"real 131K prefill", H, K, 2, 131072, 4096, 262144 - 4096, cap},
    };
    if (quick) cases.resize(6);
    bool ok = true;
    for (const auto& c : cases) ok = run_gpu_case(c) && ok;
    ok = run_topk_ties(256, 16, 8) && ok;
    ok = run_topk_ties(40000, 512, 16) && ok;
    ok = run_topk_ties(131072, 512, 16) && ok;
    std::printf("dsv41_engine_indexer_test (b): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--cases") == 0) return run_dumps(std::vector<std::string>(argv + 2, argv + argc));
    if (argc >= 2 && std::strcmp(argv[1], "--gpu") == 0) return run_gpu(argc >= 3 && std::strcmp(argv[2], "--quick") == 0);
    std::fprintf(stderr, "usage: dsv41_engine_indexer_test --cases FILE [FILE ...] | --gpu [--quick]\n");
    return 2;
}
