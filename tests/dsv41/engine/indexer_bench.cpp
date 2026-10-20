// tests/dsv41/engine/indexer_bench.cpp - bench of the lightning indexer at real dims (32 heads, dim 128, top-k
// 512) on synthetic inputs (GPU, no model): per (queries, keys) the ms of indexer_select (the Auto scorer plus the
// top-k, every pass) and of its parts, the GB/s of the compulsory bytes, the scorers' TFLOP/s and the scratch.
//
//   dsv41_engine_indexer_bench [--reps N] [--ratio R] [--scratch-mib M] [--nq A,B,..] [--nk A,B,..]
//
// Defaults: 10 reps, ratio 2 (the real ratio-2 index sources; 131,072 rows is the 262,144-token context), the
// default scratch cap, nq 1 and 4096, n_k 4096, 16384, 65536 and 131072. The queries are the last nq positions of a
// context of n_k * ratio tokens, so every query sees about n_k rows (from position 0 when nq exceeds it).
// Compulsory bytes: the keys once, the queries, the weights, the positions and the ids. FLOPs: 2 * 128 * heads per
// visible (query, row). Memory: the device bytes of the indexer's buffers (inputs, scratch, ids), which is what the
// 48 GB budget is checked against.
#include "strata/dsv41/engine/indexer.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

void ck(cudaError_t r, const char* w) {
    if (r != cudaSuccess) {
        std::fprintf(stderr, "dsv41_engine_indexer_bench: %s: %s\n", w, cudaGetErrorString(r));
        std::exit(2);
    }
}

std::vector<int64_t> parse_list(const char* s) {
    std::vector<int64_t> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        const long long x = std::strtoll(p, &end, 10);
        if (end == p || x <= 0) {
            std::fprintf(stderr, "dsv41_engine_indexer_bench: bad list '%s'\n", s);
            std::exit(2);
        }
        v.push_back(x);
        p = *end == ',' ? end + 1 : end;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 10, ratio = 2;
    size_t cap = e::kIndexerScratchCap;
    std::vector<int64_t> nqs = {1, 4096}, nks = {4096, 16384, 65536, 131072};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "dsv41_engine_indexer_bench: %s needs a value\n", a.c_str()); return 2; }
        const char* v = argv[++i];
        if (a == "--reps") reps = std::atoi(v);
        else if (a == "--ratio") ratio = std::atoi(v);
        else if (a == "--scratch-mib") cap = (size_t) std::atoll(v) << 20;
        else if (a == "--nq") nqs = parse_list(v);
        else if (a == "--nk") nks = parse_list(v);
        else { std::fprintf(stderr, "dsv41_engine_indexer_bench: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (reps < 1 || ratio < 1 || cap == 0) { std::fprintf(stderr, "dsv41_engine_indexer_bench: bad --reps/--ratio/--scratch-mib\n"); return 2; }
    int dev_id = 0;
    cudaDeviceProp p{};
    ck(cudaGetDevice(&dev_id), "cudaGetDevice");
    ck(cudaGetDeviceProperties(&p, dev_id), "cudaGetDeviceProperties");
    std::printf("device: %s, sm_%d%d, %d SMs, %.1f GiB; heads %d, dim %d, top-k %d, ratio %d, scratch cap %zu MiB, %d reps\n",
                p.name, p.major, p.minor, p.multiProcessorCount, (double) p.totalGlobalMem / (1 << 30), e::kIndexerHeads,
                e::kIndexerDim, e::kIndexerTopK, ratio, cap >> 20, reps);
    std::printf("%6s %7s %7s %6s %10s %10s %10s %9s %9s %8s %8s %8s %11s %11s\n", "nq", "n_k", "q_chunk", "passes",
                "select ms", "tc ms", "fp32 ms", "topk ms", "auto", "GB/s", "TF tc", "TF fp32", "scratch MiB", "buffers MiB");
    const e::IndexerDims d{e::kIndexerHeads, e::kIndexerTopK, ratio};
    const int H = d.n_heads, D = e::kIndexerDim;
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    for (const int64_t n_k : nks) {
        std::mt19937 rng((uint32_t) n_k);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> dir(D);
        for (auto& x : dir) x = nd(rng);
        std::vector<uint16_t> keys((size_t) (n_k * D));
        for (int64_t j = 0; j < n_k; ++j) {
            const float a = nd(rng);
            for (int c = 0; c < D; ++c) keys[(size_t) (j * D + c)] = strata::kernels::f16_from_f32(0.25f * (a * dir[c] + nd(rng)));
        }
        uint16_t* d_k = nullptr;
        ck(cudaMalloc(&d_k, keys.size() * 2), "malloc");
        ck(cudaMemcpy(d_k, keys.data(), keys.size() * 2, cudaMemcpyHostToDevice), "upload");
        for (const int64_t nq : nqs) {
            const int64_t ctx = n_k * ratio, pos0 = ctx - nq > 0 ? ctx - nq : 0;
            std::vector<float> q((size_t) (nq * H * D)), w((size_t) (nq * H));
            std::vector<int32_t> pos((size_t) nq);
            for (size_t i = 0; i < q.size(); ++i) q[i] = 0.2f * dir[i % D] + 0.5f * nd(rng);
            for (auto& x : w) x = (0.5f + nd(rng)) / 64.0f;
            for (int64_t i = 0; i < nq; ++i) pos[(size_t) i] = (int32_t) (pos0 + i);
            const e::IndexerPlan plan = e::indexer_plan(nq, n_k, cap);
            float *d_q = nullptr, *d_w = nullptr, *d_s = nullptr;
            int32_t *d_pos = nullptr, *d_ids = nullptr;
            const size_t buf = keys.size() * 2 + q.size() * 4 + w.size() * 4 + pos.size() * 4 + plan.scratch_bytes +
                               (size_t) (nq * d.top_k) * 4;
            ck(cudaMalloc(&d_q, q.size() * 4), "malloc");
            ck(cudaMalloc(&d_w, w.size() * 4), "malloc");
            ck(cudaMalloc(&d_pos, pos.size() * 4), "malloc");
            ck(cudaMalloc(&d_s, plan.scratch_bytes), "malloc scratch");
            ck(cudaMalloc(&d_ids, (size_t) (nq * d.top_k) * 4), "malloc");
            ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "upload");
            ck(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "upload");
            ck(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice), "upload");
            const int64_t nfirst = nq < plan.q_chunk ? nq : plan.q_chunk;   // the scorers alone: one pass
            auto timed = [&](auto f) {
                f();
                ck(cudaDeviceSynchronize(), "warm-up");
                ck(cudaEventRecord(e0), "record");
                for (int r = 0; r < reps; ++r) f();
                ck(cudaEventRecord(e1), "record");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                return (double) ms / reps;
            };
            const double t_sel = timed([&] {
                e::indexer_select(d_q, d_w, d_pos, nq, d_k, n_k, d, plan, d_s, e::IndexerScorer::Auto, d_ids, nullptr);
            });
            const bool tc_ok = e::indexer_resolve(e::IndexerScorer::Auto, e::kIndexerTcMinQueries) == e::IndexerScorer::TensorCore;
            const double t_tc = tc_ok ? timed([&] {
                e::indexer_scores(d_q, d_w, d_pos, nfirst, d_k, n_k, d, d_s, plan.ld, e::IndexerScorer::TensorCore, nullptr);
            }) : 0.0;
            const double t_fp = timed([&] {
                e::indexer_scores(d_q, d_w, d_pos, nfirst, d_k, n_k, d, d_s, plan.ld, e::IndexerScorer::Simt, nullptr);
            });
            const double t_tk = timed([&] { e::indexer_topk(d_s, plan.ld, d_pos, nfirst, n_k, d, d_ids, nullptr); });
            double vis_first = 0;
            for (int64_t i = 0; i < nfirst; ++i) vis_first += (double) e::indexer_visible(pos0 + i, ratio, n_k);
            const double flop_first = 2.0 * D * H * vis_first;
            const double bytes = (double) keys.size() * 2 + (double) q.size() * 4 + (double) w.size() * 4 +
                                 (double) pos.size() * 4 + (double) (nq * d.top_k) * 4;
            const e::IndexerScorer auto_is = e::indexer_resolve(e::IndexerScorer::Auto, nfirst);
            std::printf("%6lld %7lld %7lld %6lld %10.3f %10.3f %10.3f %9.3f %9s %8.1f %8.2f %8.2f %11.1f %11.1f\n",
                        (long long) nq, (long long) n_k, (long long) plan.q_chunk,
                        (long long) ((nq + plan.q_chunk - 1) / plan.q_chunk), t_sel, t_tc, t_fp, t_tk,
                        auto_is == e::IndexerScorer::TensorCore ? "tc" : "fp32", bytes / (t_sel * 1e6),
                        tc_ok ? flop_first / (t_tc * 1e9) : 0.0, flop_first / (t_fp * 1e9),
                        (double) plan.scratch_bytes / (1 << 20), (double) buf / (1 << 20));
            if (nq == 4096 && n_k == 131072 && ratio == 2)
                std::printf("       the 262,144-token context: %.1f ms per 4096-token ubatch for one ratio-2 index source "
                            "(layers 2, 8, 14; the ratio-1 sources 20, 24, 28, 32, 36 see 262,144 rows: --nk 262144 "
                            "--ratio 1), %.1f MiB of buffers; the stock fork's whole dense part took ~12.7 s per 4096-token "
                            "ubatch (A100 SXM 80 GB pod, 16K prompt)\n", t_sel, (double) buf / (1 << 20));
            cudaFree(d_q); cudaFree(d_w); cudaFree(d_pos); cudaFree(d_s); cudaFree(d_ids);
        }
        cudaFree(d_k);
    }
    std::printf("columns: select = indexer_select (Auto scorer + top-k, every pass); tc / fp32 / topk = one pass of "
                "q_chunk queries; GB/s = compulsory bytes / select; TF = TFLOP/s of the scorer's pass\n");
    return 0;
}
