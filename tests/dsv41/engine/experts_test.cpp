// tests/dsv41/engine/experts_test.cpp - parity of the routed experts (include/strata/dsv41/engine/experts.hpp),
// the grouped prefill GEMM (experts_gemm.hpp).
//
//   dsv41_engine_experts_test --cases FILE [FILE ...]          (a) GPU free: the CPU experts against the stock fork's
//                                                              op dumps (the lists of tools/dsv41/engine/dense_cases.py)
//   dsv41_engine_experts_test --gpu [--dry-run] GGUF HOT_DIR [GGUF HOT_DIR ...]
//                                                              (b) the expert tiers on CUDA against the CPU references,
//                                                              in three placements; --dry-run their host side only
//   dsv41_engine_experts_test --gemm [--dry-run]               (c) the grouped prefill GEMM at the real dims against
//                                                              its reference, one-hot rows bit for bit, its timing
//                                                              (run_gemm below); --dry-run its host side only
//   dsv41_engine_experts_test --determinism GGUF               (d), GPU free: the deterministic mode on the CPU
//                                                              reference path (run_determinism below)
//
// (a) Per dump, ubatch and layer: the dump's own ffn_norm, ffn_moe_topk (slot order) and ffn_moe_weights_scaled
// through CpuExperts in the fork's numerics (CpuNumerics::Fork: strata-moe's fast kernel behind plain pointers; the
// engine's own are the GPU's, which dsv41_engine_expert_bits_test checks) and the combine in slot order give ffn_moe_out
// bit for bit, with 1 thread and with 3 (a value never depends on the thread that computes it). The same inputs
// and selections through the prefill GEMM's numerics on the host (f16 weights and activations, qref_matmul_f16) within
// kPlacementRelL2 of ffn_moe_out.
//
// (b) Per GGUF (tiny-a, tiny-b: the real tensor types) and placement, from the hot sets tools/dsv41/engine/hot_sets.py
// writes into HOT_DIR: all-VRAM (all-vram.json, every expert), all-CPU (no hot set: every expert cold, from the mapping)
// and split (split.json: per layer the even experts in VRAM, the odd ones in the warm tier, pinned by strata-moe's
// residency from the GGUF with O_DIRECT, except every fourth, which stays cold); these three with the prefill split
// off (every prefill pair on the GPU), then all-CPU and split again with the prefill split's CPU side taking every
// expert outside the slab (PrefillSplit::Cpu).
// Every layer gets random inputs, routings (distinct experts per token), weights and a shared-expert output, as decode
// calls of 1 and 4 tokens (the hit/miss split) and a prefill call of 64 tokens, checked as:
//   all-CPU decode       bit for bit the CPU reference (the engine's CPU experts, in the GPU's numerics,
//                        then the slot-order combine)
//   GPU pairs            within kTolEngine (rel L2 per call) of the engine-numerics reference: the expert's rows
//                        through qmatmul_ref.hpp, a decode hit on the path qmatmul takes for its row count (q8_1 and
//                        MMVQ up to 8, f16 and the dequant path above), a prefill pair on the grouped GEMM's f16 at any
//                        count, the clamp, down the same way; the GPU's f32 sums, its expf and a requantized h differ
//                        from it by about 1e-5 to 3e-4 (h in f16), a wrong expert, slot or row by O(1)
//   split decode         its hits and misses combined: within kTolEngine of the hits' engine reference and the misses'
//                        CPU reference added in slot order
//   prefill              bit for bit the same in the three placements (an expert's rows go through the grouped
//                        GEMM's tiles, and a row's values do not depend on the other rows of its tile or pass, whether
//                        the slab, the ring or the prefetch region holds the expert)
//   prefill split (cpu)  as decode: all-CPU bit for bit the CPU reference, split within kTolEngine of its slab pairs'
//                        engine reference and its other pairs' CPU reference
//   all-VRAM vs all-CPU  within kPlacementRelL2 (the bound for the fork's numerics: int8 per 32 values against the fork's
//                        int8 per 256); decode bit for bit the same in all five
//                        placements (a miss is a hit's value on the CPU), so a token's output does not follow the slab
#include "dump_lists.hpp"
#include "qmatmul_synth.hpp"

#include "strata/dsv41/engine/experts.hpp"
#include "strata/dsv41/engine/experts_gemm.hpp"
#include "strata/dsv41/engine/hot_policy.hpp"
#include "strata/dsv41/engine/model.hpp"
#include "strata/dsv41/engine/qmatmul_ref.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
    std::fprintf(stderr, "dsv41_engine_experts_test: %s\n", m.c_str());
    std::exit(2);
}

constexpr double kTolEngine = 1e-3;
constexpr double kPlacementRelL2 = 3e-2;

uint32_t fbits(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    return b;
}

double rel_l2(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

int64_t bits_differ(const float* a, const float* b, size_t n) {
    int64_t d = 0;
    for (size_t i = 0; i < n; ++i) d += fbits(a[i]) != fbits(b[i]);
    return d;
}

// the CPU experts of every pair of a call, grouped by expert in first-use order as the engine groups its misses
void cpu_pairs(e::CpuExperts& cpu, const e::ExpertLayerHost& L, const e::DenseDims& d, const float* x, int64_t T,
               const int32_t* ids, float* y_pairs) {
    const int K = d.n_expert_used;
    std::vector<e::CpuExpertJob> jobs;
    std::vector<int32_t> job_of((size_t) d.n_expert, -1);
    std::vector<std::vector<int32_t>> pairs;
    for (int64_t p = 0; p < T * K; ++p) {
        const int32_t ex = ids[p];
        if (job_of[(size_t) ex] < 0) {
            job_of[(size_t) ex] = (int32_t) jobs.size();
            e::CpuExpertJob j;
            for (int q = 0; q < 3; ++q) j.rows.part[q] = L.data[q] + (size_t) ex * L.bytes[q];
            jobs.push_back(j);
            pairs.emplace_back();
        }
        jobs[(size_t) job_of[(size_t) ex]].tokens.push_back((int32_t) (p / K));
        pairs[(size_t) job_of[(size_t) ex]].push_back((int32_t) p);
    }
    int64_t y0 = 0;
    for (e::CpuExpertJob& j : jobs) {
        j.y0 = y0;
        y0 += (int64_t) j.tokens.size();
    }
    std::vector<float> y((size_t) (y0 * d.n_embd));
    cpu.run(L, d, x, T, jobs.data(), jobs.size(), y.data());
    for (size_t k = 0; k < jobs.size(); ++k)
        for (size_t i = 0; i < pairs[k].size(); ++i)
            std::memcpy(y_pairs + (size_t) pairs[k][i] * d.n_embd, y.data() + (size_t) (jobs[k].y0 + (int64_t) i) * d.n_embd,
                        (size_t) d.n_embd * 4);
}

void engine_pairs(const e::ExpertLayerHost& L, const e::DenseDims& d, const float* x, int64_t T, const int32_t* ids,
                  bool prefill, float* y_eng);

// ---------------------------------------------------------------- (a)
int run_cases(const std::vector<std::string>& lists) {
    const std::vector<dumps::Dump> all = dumps::read_lists(lists);
    bool ok = true;
    for (const dumps::Dump& dump : all) {
        const e::HostModel host = e::load_gguf_model(dump.model);
        e::Model m;
        std::string err;
        if (!m.init(host, true, false, err)) die(err);
        const e::DenseDims& d = m.dense_dims();
        const int K = d.n_expert_used;
        int64_t calls = 0, values = 0, differ = 0;
        for (int threads : {1, 3}) {
            e::CpuExperts cpu(threads, e::CpuNumerics::Fork);
            for (int il = 0; il < m.n_layer(); ++il) {
                const e::ExpertLayerHost& L = m.experts()[(size_t) il];
                e::cpu_experts_check(L, d, e::CpuNumerics::Fork);
                for (const dumps::Ubatch& ub : dump.ubs) {
                    const std::vector<float> x = dumps::load_f32(dump.get("ffn_norm", 0, il, ub.u));
                    const std::vector<int32_t> ids = dumps::load_i32(dump.get("ffn_moe_topk", 0, il, ub.u));
                    const std::vector<float> w = dumps::load_f32(dump.get("ffn_moe_weights_scaled", 0, il, ub.u));
                    const std::vector<float> fork = dumps::load_f32(dump.get("ffn_moe_out", 0, il, ub.u));
                    const int64_t T = ub.T;
                    if ((int64_t) ids.size() != T * K || (int64_t) x.size() != T * d.n_embd) die(dump.header + ": shapes");
                    std::vector<float> y((size_t) (T * K * d.n_embd)), out((size_t) (T * d.n_embd));
                    cpu_pairs(cpu, L, d, x.data(), T, ids.data(), y.data());
                    e::ref_expert_combine(y.data(), w.data(), T, K, d.n_embd, nullptr, out.data());
                    ++calls;
                    values += (int64_t) out.size();
                    const int64_t n = bits_differ(out.data(), fork.data(), out.size());
                    differ += n;
                    if (n && differ == n)
                        std::printf("diff %s layer %d ubatch %d (%d threads): %lld values, rel L2 %.3g\n", dump.header.c_str(),
                                    il, ub.u, threads, (long long) n, rel_l2(out.data(), fork.data(), out.size()));
                }
            }
        }
        std::printf("%s %s: ffn_moe_out of %lld calls (%d layers, %zu ubatches, 1 and 3 threads), %lld values, %s\n",
                    differ ? "FAIL" : "ok  ", dump.header.c_str(), (long long) calls, m.n_layer(), dump.ubs.size(),
                    (long long) values, differ ? "DIFFER" : "bit-identical to the fork");
        ok = ok && differ == 0;
        // the prefill passes' numerics (experts_gemm.hpp: the dequant path's f16 at every row count) on the
        // dump's own inputs and selections, against the fork's ffn_moe_out
        double worst = 0;
        for (int il = 0; il < m.n_layer(); ++il)
            for (const dumps::Ubatch& ub : dump.ubs) {
                const std::vector<float> x = dumps::load_f32(dump.get("ffn_norm", 0, il, ub.u));
                const std::vector<int32_t> ids = dumps::load_i32(dump.get("ffn_moe_topk", 0, il, ub.u));
                const std::vector<float> w = dumps::load_f32(dump.get("ffn_moe_weights_scaled", 0, il, ub.u));
                const std::vector<float> fork = dumps::load_f32(dump.get("ffn_moe_out", 0, il, ub.u));
                std::vector<float> y((size_t) (ub.T * K * d.n_embd)), out((size_t) (ub.T * d.n_embd));
                engine_pairs(m.experts()[(size_t) il], d, x.data(), ub.T, ids.data(), true, y.data());
                e::ref_expert_combine(y.data(), w.data(), ub.T, K, d.n_embd, nullptr, out.data());
                worst = std::max(worst, rel_l2(out.data(), fork.data(), out.size()));
            }
        const bool near = worst <= kPlacementRelL2;
        std::printf("%s %s: the prefill GEMM's numerics (f16 weights and activations, f32 sums) against the fork's "
                    "ffn_moe_out: worst rel L2 %.3g over %d layers x %zu ubatches (bound %.3g)\n",
                    near ? "ok  " : "FAIL", dump.header.c_str(), worst, m.n_layer(), dump.ubs.size(), kPlacementRelL2);
        ok = ok && near;
    }
    std::printf("dsv41_engine_experts_test (a): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- (b)
void ck(cudaError_t r, const char* w) {
    if (r != cudaSuccess) die(std::string(w) + ": " + cudaGetErrorString(r));
}

// the engine-numerics reference of one expert over n rows: a prefill pass's grouped GEMM (f16, the dequant path's
// numerics at every n, experts_gemm.hpp), or decode's qmatmul path at n rows (q8_1 up to kMmvqMaxCols, else f16)
void engine_expert(const e::ExpertLayerHost& L, const e::DenseDims& d, int32_t ex, const std::vector<const float*>& rows,
                   bool prefill, float* y) {
    const int64_t n = (int64_t) rows.size(), E = d.n_embd, F = d.n_ff;
    std::vector<float> x((size_t) (n * E)), g((size_t) (n * F)), u((size_t) (n * F)), h((size_t) (n * F));
    for (int64_t i = 0; i < n; ++i) std::memcpy(x.data() + i * E, rows[(size_t) i], (size_t) E * 4);
    auto view = [&](int p) {
        const void* data = L.data[p] + (size_t) ex * L.bytes[p];
        return p == 2 ? e::QWeightView{L.type[2], F, E, data} : e::QWeightView{L.type[p], E, F, data};
    };
    auto mm = [&](const e::QWeightView& w, const float* in, int64_t k, float* out) {
        if (!prefill && n <= e::kMmvqMaxCols) {
            e::QActsHost a;
            e::qref_quantize(in, k, n, k, e::QActLayout::Q8_1, a);
            e::qref_matmul(w, a, 1, out, w.rows, nullptr);
        } else {
            e::qref_matmul_f16(w, in, k, n, 1, out, w.rows, nullptr);
        }
    };
    mm(view(1), x.data(), E, g.data());
    mm(view(0), x.data(), E, u.data());
    for (size_t i = 0; i < h.size(); ++i) {
        const float gg = std::min(g[i], L.limit), uu = std::min(std::max(u[i], -L.limit), L.limit);
        h[i] = gg / (1.0f + std::exp(-gg)) * uu;
    }
    mm(view(2), h.data(), F, y);
}

struct Call {
    int64_t T = 0;
    std::vector<float> x, w, shared;
    std::vector<int32_t> ids;
};

Call make_call(std::mt19937& rng, const e::DenseDims& d, int64_t T) {
    Call c;
    c.T = T;
    const int K = d.n_expert_used;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> uw(0.05f, 0.6f);
    c.x.resize((size_t) (T * d.n_embd));
    c.shared.resize(c.x.size());
    for (float& v : c.x) v = nd(rng);
    for (float& v : c.shared) v = 0.1f * nd(rng);
    std::vector<int32_t> perm((size_t) d.n_expert);
    for (int i = 0; i < d.n_expert; ++i) perm[(size_t) i] = i;
    for (int64_t t = 0; t < T; ++t) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int s = 0; s < K; ++s) {
            c.ids.push_back(perm[(size_t) s]);
            c.w.push_back(uw(rng));
        }
    }
    return c;
}

// the references of one call: per pair the CPU's y and the engine's y (each expert over all of its pairs' rows, on the
// prefill or the decode numerics)
void engine_pairs(const e::ExpertLayerHost& L, const e::DenseDims& d, const float* x, int64_t T, const int32_t* ids,
                  bool prefill, float* y_eng) {
    const int K = d.n_expert_used;
    const int64_t E = d.n_embd, P = T * K;
    for (int32_t ex = 0; ex < d.n_expert; ++ex) {
        std::vector<const float*> rows;
        std::vector<int64_t> pairs;
        for (int64_t p = 0; p < P; ++p)
            if (ids[p] == ex) {
                rows.push_back(x + (p / K) * E);
                pairs.push_back(p);
            }
        if (rows.empty()) continue;
        std::vector<float> y((size_t) ((int64_t) rows.size() * E));
        engine_expert(L, d, ex, rows, prefill, y.data());
        for (size_t i = 0; i < pairs.size(); ++i)
            std::memcpy(y_eng + pairs[i] * E, y.data() + (int64_t) i * E, (size_t) E * 4);
    }
}

void references(e::CpuExperts& cpu, const e::ExpertLayerHost& L, const e::DenseDims& d, const Call& c,
                std::vector<float>& y_cpu, std::vector<float>& y_eng) {
    const int K = d.n_expert_used;
    const int64_t E = d.n_embd, P = c.T * K;
    y_cpu.assign((size_t) (P * E), 0.f);
    y_eng.assign((size_t) (P * E), 0.f);
    cpu_pairs(cpu, L, d, c.x.data(), c.T, c.ids.data(), y_cpu.data());
    engine_pairs(L, d, c.x.data(), c.T, c.ids.data(), c.T > e::kExpertSplitMaxT, y_eng.data());
}

int run_gpu(const std::vector<std::string>& args, bool dry) {
    bool ok = true;
    if (args.empty() || args.size() % 2) die("--gpu takes pairs GGUF HOT_DIR (tools/dsv41/engine/hot_sets.py)");
    for (size_t a = 0; a < args.size(); a += 2) {
        const std::string& path = args[a];
        const e::HostModel host = e::load_gguf_model(path);
        e::Model m;
        std::string err;
        if (!m.init(host, true, false, err)) die(err);
        const e::DenseDims& d = m.dense_dims();
        const int K = d.n_expert_used;
        const int64_t E = d.n_embd;
        const std::string all_hot = args[a + 1] + "/all-vram.json", split_hot = args[a + 1] + "/split.json";
        struct Placement {
            const char* name;
            e::ExpertConfig cfg;
        };
        // the three placements with every prefill pair on the GPU, then two with the prefill split's CPU side
        constexpr size_t kGpuPrefill = 3;
        std::vector<Placement> places(5);
        places[0].name = "all-VRAM";
        places[0].cfg.hot_set = all_hot;
        places[0].cfg.vram_gib = 4;
        places[1].name = "all-CPU";
        places[2].name = "split";
        places[2].cfg.hot_set = split_hot;
        places[2].cfg.vram_gib = 4;
        places[2].cfg.ram_gib = 4;
        places[3] = places[1];
        places[3].name = "all-CPU, prefill split cpu";
        places[4] = places[2];
        places[4].name = "split, prefill split cpu";
        for (size_t i = 0; i < places.size(); ++i) {
            places[i].cfg.threads = 4;
            places[i].cfg.prefill_split = i < kGpuPrefill ? e::PrefillSplit::Off : e::PrefillSplit::Cpu;
        }
        const std::vector<int64_t> Ts = {1, 4, 64};
        std::mt19937 rng(20261007u);
        std::vector<std::vector<Call>> calls((size_t) m.n_layer());
        for (int l = 0; l < m.n_layer(); ++l)
            for (int64_t T : Ts) calls[(size_t) l].push_back(make_call(rng, d, T));

        // the engine's CPU experts (the GPU's numerics) and the fork's (the dry run's measure of the engine's numerics)
        e::CpuExperts cpu(4), fork_cpu(4, e::CpuNumerics::Fork);
        if (dry) {
            for (Placement& p : places) {
                e::ExpertEngine eng;
                if (!eng.init(m, p.cfg, 64, nullptr, nullptr, true, err)) die(err);
                std::printf("dry run %s, %s:\n", path.c_str(), p.name);
                eng.print_summary(stdout);
                eng.dry_plan(64, stdout);
            }
            // every call's references on the host, as the GPU run computes them: how far the engine's numerics are
            // from the fork's on these weights (the measure behind kPlacementRelL2)
            double worst = 0;
            for (int l = 0; l < m.n_layer(); ++l)
                for (const Call& c : calls[(size_t) l]) {
                    std::vector<float> yc, ye, oc(c.x.size()), oe(c.x.size());
                    references(fork_cpu, m.experts()[(size_t) l], d, c, yc, ye);
                    e::ref_expert_combine(yc.data(), c.w.data(), c.T, K, (int) E, c.shared.data(), oc.data());
                    e::ref_expert_combine(ye.data(), c.w.data(), c.T, K, (int) E, c.shared.data(), oe.data());
                    worst = std::max(worst, rel_l2(oe.data(), oc.data(), oc.size()));
                }
            std::printf("dry run %s: the references of every call: engine numerics vs the fork's CPU experts, worst rel L2 "
                        "%.3g (bound %.3g)\n", path.c_str(), worst, kPlacementRelL2);
            ok = ok && worst <= kPlacementRelL2;
            continue;
        }

        e::QmmContext qmm;
        if (!qmm.init(err, e::QPrefill::DequantF16, (int64_t) d.n_ff * E)) die(err);
        cudaStream_t stream = nullptr;
        ck(cudaStreamCreate(&stream), "cudaStreamCreate");
        // outputs per placement, layer, call
        std::vector<std::vector<std::vector<std::vector<float>>>> outs(places.size());
        double worst_place = 0, worst_eng = 0;
        for (size_t pi = 0; pi < places.size(); ++pi) {
            e::ExpertEngine eng;
            if (!eng.init(m, places[pi].cfg, 64, &qmm, stream, false, err)) die(err);
            eng.print_summary(stdout);
            outs[pi].resize((size_t) m.n_layer());
            for (int l = 0; l < m.n_layer(); ++l) {
                for (const Call& c : calls[(size_t) l]) {
                    const int64_t P = c.T * K;
                    float *dx, *dw, *dsh, *dout;
                    int32_t* dids;
                    ck(cudaMalloc(&dx, c.x.size() * 4), "cudaMalloc");
                    ck(cudaMalloc(&dw, c.w.size() * 4), "cudaMalloc");
                    ck(cudaMalloc(&dsh, c.shared.size() * 4), "cudaMalloc");
                    ck(cudaMalloc(&dout, c.x.size() * 4), "cudaMalloc");
                    ck(cudaMalloc(&dids, (size_t) P * 4), "cudaMalloc");
                    ck(cudaMemcpy(dx, c.x.data(), c.x.size() * 4, cudaMemcpyHostToDevice), "upload");
                    ck(cudaMemcpy(dw, c.w.data(), c.w.size() * 4, cudaMemcpyHostToDevice), "upload");
                    ck(cudaMemcpy(dsh, c.shared.data(), c.shared.size() * 4, cudaMemcpyHostToDevice), "upload");
                    ck(cudaMemcpy(dids, c.ids.data(), (size_t) P * 4, cudaMemcpyHostToDevice), "upload");
                    const bool prefill = c.T > e::kExpertSplitMaxT;
                    eng.forward(l, dx, dids, dw, dsh, c.T, prefill, dout);
                    std::vector<float> out(c.x.size());
                    ck(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost), "download");
                    for (void* q : {(void*) dx, (void*) dw, (void*) dsh, (void*) dout, (void*) dids}) cudaFree(q);

                    // the expected output of this placement and call
                    std::vector<float> yc, ye;
                    references(cpu, m.experts()[(size_t) l], d, c, yc, ye);
                    const e::ExpertLayerTiers& tiers = eng.tiers(l);
                    // a pair outside the slab runs on the CPU in decode and in a prefill split
                    const bool cpu_side = !prefill || eng.split_call(c.T);
                    std::vector<float> ymix((size_t) (P * E));
                    for (int64_t p = 0; p < P; ++p) {
                        const bool gpu = !cpu_side || tiers.slot[(size_t) c.ids[(size_t) p]] >= 0;
                        std::memcpy(ymix.data() + p * E, (gpu ? ye : yc).data() + p * E, (size_t) E * 4);
                    }
                    std::vector<float> want(out.size());
                    e::ref_expert_combine(ymix.data(), c.w.data(), c.T, K, (int) E, c.shared.data(), want.data());
                    char where[160];
                    std::snprintf(where, sizeof where, "%s %s layer %d, %s of %lld tokens", path.c_str(), places[pi].name,
                                  l, prefill ? "prefill" : "decode", (long long) c.T);
                    bool all_cpu = cpu_side;
                    for (int64_t p = 0; p < P; ++p) all_cpu = all_cpu && tiers.slot[(size_t) c.ids[(size_t) p]] < 0;
                    if (all_cpu) {
                        const int64_t n = bits_differ(out.data(), want.data(), out.size());
                        if (n) {
                            std::printf("FAIL %s: %lld values differ from the CPU reference\n", where, (long long) n);
                            ok = false;
                        }
                    } else {
                        const double r = rel_l2(out.data(), want.data(), out.size());
                        worst_eng = std::max(worst_eng, r);
                        if (!(r <= kTolEngine)) {
                            std::printf("FAIL %s: rel L2 %.3g from its reference (bound %.3g)\n", where, r, kTolEngine);
                            ok = false;
                        }
                    }
                    outs[pi][(size_t) l].push_back(std::move(out));
                }
            }
            eng.reset_times();
        }
        // across placements: prefill bit for bit; decode bit for bit too, in
        // every placement (the CPU experts are the hits' numerics), and all-VRAM against all-CPU within the old bound
        // as before
        int64_t prefill_differ = 0, decode_differ = 0;
        for (int l = 0; l < m.n_layer(); ++l)
            for (size_t ci = 0; ci < Ts.size(); ++ci) {
                const std::vector<float>& vram = outs[0][(size_t) l][ci];
                if (Ts[ci] > e::kExpertSplitMaxT) {
                    for (size_t pi = 1; pi < kGpuPrefill; ++pi)
                        prefill_differ += bits_differ(outs[pi][(size_t) l][ci].data(), vram.data(), vram.size());
                } else {
                    worst_place = std::max(worst_place, rel_l2(vram.data(), outs[1][(size_t) l][ci].data(), vram.size()));
                    for (size_t pi = 1; pi < places.size(); ++pi)
                        decode_differ += bits_differ(outs[pi][(size_t) l][ci].data(), vram.data(), vram.size());
                }
            }
        if (prefill_differ) {
            std::printf("FAIL %s: prefill outputs differ between placements in %lld values\n", path.c_str(), (long long) prefill_differ);
            ok = false;
        }
        if (decode_differ) {
            std::printf("FAIL %s: decode outputs differ between placements in %lld values (C05)\n", path.c_str(), (long long) decode_differ);
            ok = false;
        }
        if (!(worst_place <= kPlacementRelL2)) ok = false;
        std::printf("%s %s: three placements and two with the prefill split on the CPU, %d layers, calls of 1, 4 (decode) "
                    "and 64 (prefill) tokens: GPU calls within %.3g of their references (worst %.3g), all-CPU decode and "
                    "split prefill bit for bit the engine's CPU experts (the GPU's numerics), GPU prefill bit-identical "
                    "across placements (%lld differ), decode bit-identical across the five placements (%lld differ; "
                    "all-VRAM vs all-CPU rel L2 %.3g, bound %.3g)\n",
                    ok ? "ok  " : "FAIL", path.c_str(), m.n_layer(), kTolEngine, worst_eng, (long long) prefill_differ,
                    (long long) decode_differ, worst_place, kPlacementRelL2);
        cudaStreamDestroy(stream);
    }
    std::printf("dsv41_engine_experts_test (b)%s: %s\n", dry ? " dry run" : "", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- (c)
// one expert of the real dims on the host (0 up, 1 gate, 2 down) and on the device
struct GemmExpert {
    std::vector<uint8_t> part[3];
    void* dev[3] = {nullptr, nullptr, nullptr};
    e::QWeightView view(int p) const {
        return p == 2 ? e::QWeightView{e::kQTypeQ3K, 2304, 5120, part[2].data()} : e::QWeightView{e::kQTypeQ2K, 5120, 2304, part[p].data()};
    }
};

std::vector<GemmExpert> gemm_experts(int n, uint64_t seed, bool upload) {
    const e::DenseDims d = e::dense_dims_real();
    std::mt19937_64 rng(seed);
    std::vector<GemmExpert> ex((size_t) n);
    for (GemmExpert& x : ex) {
        x.part[0] = qsynth::random_blocks(e::kQTypeQ2K, d.n_ff, d.n_embd, 1.f / std::sqrt((float) d.n_embd), rng);
        x.part[1] = qsynth::random_blocks(e::kQTypeQ2K, d.n_ff, d.n_embd, 1.f / std::sqrt((float) d.n_embd), rng);
        x.part[2] = qsynth::random_blocks(e::kQTypeQ3K, d.n_embd, d.n_ff, 1.f / std::sqrt((float) d.n_ff), rng);
        if (!upload) continue;
        for (int p = 0; p < 3; ++p) {
            ck(cudaMalloc(&x.dev[p], x.part[p].size()), "cudaMalloc (expert)");
            ck(cudaMemcpy(x.dev[p], x.part[p].data(), x.part[p].size(), cudaMemcpyHostToDevice), "upload (expert)");
        }
    }
    return ex;
}

// the tiles of experts with rows[k] consecutive rows each, as experts.cpp cuts them
std::vector<e::ExpertTile> gemm_tiles(const std::vector<GemmExpert>& ex, const std::vector<int64_t>& rows) {
    std::vector<e::ExpertTile> t;
    int64_t row0 = 0;
    for (size_t k = 0; k < rows.size(); ++k) {
        for (int64_t r = 0; r < rows[k]; r += e::kExpertTileRows) {
            e::ExpertTile x{};
            for (int p = 0; p < 3; ++p) x.w[p] = ex[k].dev[p];
            x.row0 = (int32_t) (row0 + r);
            x.rows = (int32_t) std::min<int64_t>(e::kExpertTileRows, rows[k] - r);
            t.push_back(x);
        }
        row0 += rows[k];
    }
    return t;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    ck(cudaMalloc(&p, std::max<size_t>(v.size() * sizeof(T), 16)), "cudaMalloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}

// The grouped GEMM of the prefill passes (experts_gemm.hpp) at the real dims (n_embd 5120, n_ff 2304, q2_K gate and
// up, q3_K down) on random experts, without the engine around it:
//   tiles   six experts of 1, 7, 64, 128, 129 and 300 rows (one-row tiles, partial and full ones, a 2-row remainder),
//           their rows gathered from a shuffled x through expert_gather_rows_f16: h of gate, up and the clamp against
//           qref_matmul_f16 and the clamp on the host, y of down against qref_matmul_f16 of the GPU's h, each expert
//           within kTolEngine (rel L2; the GPU's f32 sums and its expf, a wrong row, tile or weight is O(1))
//   one-hot rows t with a single 1 at k_t (k_t = t for 512 rows, so every position of two blocks, then 64 random):
//           y[t][n] is one weight, which must be the dequant path's f16 value (qref_dequant_row rounded) exactly, for
//           q2_K (gate, as a plain grouped GEMM) and q3_K (down)
//   timing  (reported) 48 experts of 256 rows each (a 16384-token chunk's mean rows per expert) and of 32: the gather,
//           gate/up and down per call and in TFLOP/s (6 n_embd n_ff FLOP per row)
// --dry-run: the shapes through expert_gemm_check, the tiles and the one-hot references on the host, no CUDA call.
int run_gemm(bool dry) {
    const e::DenseDims d = e::dense_dims_real();
    const int64_t E = d.n_embd, F = d.n_ff;
    const float limit = 10.f;
    e::expert_gemm_check(e::kQTypeQ2K, e::kQTypeQ2K, e::kQTypeQ3K, E, F);
    bool ok = true;
    const std::vector<int64_t> rows = {1, 7, 64, 128, 129, 300};
    int64_t R = 0;
    for (int64_t r : rows) R += r;
    std::vector<GemmExpert> ex = gemm_experts((int) rows.size(), 51, !dry);
    const std::vector<e::ExpertTile> tiles = gemm_tiles(ex, rows);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) (R * E));
    for (float& v : x) v = nd(rng);
    std::vector<int32_t> perm((size_t) R);
    for (int64_t i = 0; i < R; ++i) perm[(size_t) i] = (int32_t) i;
    std::shuffle(perm.begin(), perm.end(), rng);
    // the one-hot rows' positions and the expected weights: gate (q2_K) rows [F][E], down (q3_K) rows [E][F]
    auto one_hot_k = [&](int64_t K) {
        std::vector<int64_t> k;
        for (int64_t t = 0; t < 512; ++t) k.push_back(t);
        for (int t = 0; t < 64; ++t) k.push_back((int64_t) (rng() % (uint32_t) K));
        return k;
    };
    const std::vector<int64_t> k_gate = one_hot_k(E), k_down = one_hot_k(F);
    auto expected = [&](const e::QWeightView& w, const std::vector<int64_t>& ks) {
        std::vector<float> row((size_t) w.k), want((size_t) ks.size() * (size_t) w.rows);
        for (int64_t n = 0; n < w.rows; ++n) {
            e::qref_dequant_row(w, n, row.data());
            for (size_t t = 0; t < ks.size(); ++t)
                want[t * (size_t) w.rows + (size_t) n] = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(row[(size_t) ks[t]]));
        }
        return want;
    };
    const std::vector<float> want_gate = expected(ex[0].view(1), k_gate), want_down = expected(ex[0].view(2), k_down);
    if (dry) {
        std::printf("dry run (c): %zu experts of", rows.size());
        for (int64_t r : rows) std::printf(" %lld", (long long) r);
        std::printf(" rows (%lld in all) in %zu tiles of up to %d; one-hot references of %zu q2_K and %zu q3_K rows made "
                    "on the host; timing: 48 experts x 256 and x 32 rows\n", (long long) R, tiles.size(),
                    e::kExpertTileRows, k_gate.size(), k_down.size());
        std::printf("dsv41_engine_experts_test (c) dry run: PASS\n");
        return 0;
    }
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "cudaStreamCreate");
    float* dx = upload(x);
    int32_t* dperm = upload(perm);
    e::ExpertTile* dtiles = upload(tiles);
    uint16_t *dxg = nullptr, *dh = nullptr;
    float* dy = nullptr;
    const int64_t Rmax = std::max<int64_t>(R, (int64_t) k_gate.size());
    ck(cudaMalloc(&dxg, (size_t) (Rmax * E) * 2), "cudaMalloc");
    ck(cudaMalloc(&dh, (size_t) (Rmax * F) * 2), "cudaMalloc");
    ck(cudaMalloc(&dy, (size_t) (Rmax * std::max(E, F)) * 4), "cudaMalloc");
    e::expert_gather_rows_f16(dx, dperm, 1, R, (int) E, dxg, s);
    e::expert_gate_up(dtiles, (int64_t) tiles.size(), e::kQTypeQ2K, dxg, E, F, limit, dh, s);
    e::expert_gemm(dtiles, (int64_t) tiles.size(), 2, e::kQTypeQ3K, dh, F, E, dy, E, s);
    std::vector<uint16_t> h16((size_t) (R * F));
    std::vector<float> y((size_t) (R * E));
    ck(cudaMemcpyAsync(h16.data(), dh, h16.size() * 2, cudaMemcpyDeviceToHost, s), "download");
    ck(cudaMemcpyAsync(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost, s), "download");
    ck(cudaStreamSynchronize(s), "the tiles case");
    double worst_h = 0, worst_y = 0;
    int64_t r0 = 0;
    for (size_t k = 0; k < rows.size(); ++k) {
        const int64_t n = rows[k];
        std::vector<float> xs((size_t) (n * E)), g((size_t) (n * F)), u((size_t) (n * F)), h((size_t) (n * F)),
            hg((size_t) (n * F)), yr((size_t) (n * E));
        for (int64_t i = 0; i < n; ++i)
            std::memcpy(xs.data() + i * E, x.data() + (size_t) perm[(size_t) (r0 + i)] * (size_t) E, (size_t) E * 4);
        e::qref_matmul_f16(ex[k].view(1), xs.data(), E, n, 1, g.data(), F, nullptr);
        e::qref_matmul_f16(ex[k].view(0), xs.data(), E, n, 1, u.data(), F, nullptr);
        for (size_t i = 0; i < h.size(); ++i) {
            const float gg = std::min(g[i], limit), uu = std::min(std::max(u[i], -limit), limit);
            h[i] = gg / (1.0f + std::exp(-gg)) * uu;
            hg[i] = strata::kernels::f32_from_f16(h16[(size_t) (r0 * F) + i]);
        }
        e::qref_matmul_f16(ex[k].view(2), hg.data(), F, n, 1, yr.data(), E, nullptr);
        const double rh = rel_l2(hg.data(), h.data(), h.size()), ry = rel_l2(y.data() + r0 * E, yr.data(), yr.size());
        worst_h = std::max(worst_h, rh);
        worst_y = std::max(worst_y, ry);
        if (!(rh <= kTolEngine) || !(ry <= kTolEngine)) {
            std::printf("FAIL (c) expert %zu of %lld rows: h rel L2 %.3g, y rel L2 %.3g (bound %.3g)\n", k, (long long) n, rh,
                        ry, kTolEngine);
            ok = false;
        }
        r0 += n;
    }
    std::printf("%s (c) tiles: %zu experts, %lld rows in %zu tiles: gate/up/clamp worst rel L2 %.3g, down %.3g (bound %.3g)\n",
                ok ? "ok  " : "FAIL", rows.size(), (long long) R, tiles.size(), worst_h, worst_y, kTolEngine);

    // one-hot rows: every output a single weight
    auto one_hot = [&](int part, int type, int64_t K, int64_t N, const std::vector<int64_t>& ks, const std::vector<float>& want,
                       const char* what) {
        const int64_t n = (int64_t) ks.size();
        std::vector<float> xo((size_t) (n * K), 0.f);
        for (int64_t t = 0; t < n; ++t) xo[(size_t) (t * K + ks[(size_t) t])] = 1.f;
        std::vector<int32_t> id((size_t) n);
        for (int64_t t = 0; t < n; ++t) id[(size_t) t] = (int32_t) t;
        std::vector<GemmExpert> one(1);
        for (int p = 0; p < 3; ++p) one[0].dev[p] = ex[0].dev[p];
        const std::vector<e::ExpertTile> tl = gemm_tiles(one, {n});
        float* dxo = upload(xo);
        int32_t* did = upload(id);
        e::ExpertTile* dtl = upload(tl);
        e::expert_gather_rows_f16(dxo, did, 1, n, (int) K, dxg, s);
        e::expert_gemm(dtl, (int64_t) tl.size(), part, type, dxg, K, N, dy, N, s);
        std::vector<float> got((size_t) (n * N));
        ck(cudaMemcpyAsync(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost, s), "download");
        ck(cudaStreamSynchronize(s), "the one-hot case");
        int64_t bad = 0;
        for (size_t i = 0; i < got.size(); ++i) bad += !(got[i] == want[i]);
        std::printf("%s (c) one-hot %s: %lld rows x %lld weights, %lld differ from the dequant path's f16\n",
                    bad ? "FAIL" : "ok  ", what, (long long) n, (long long) N, (long long) bad);
        for (void* q : {(void*) dxo, (void*) did, (void*) dtl}) cudaFree(q);
        return bad == 0;
    };
    ok = one_hot(1, e::kQTypeQ2K, E, F, k_gate, want_gate, "q2_K (gate)") && ok;
    ok = one_hot(2, e::kQTypeQ3K, F, E, k_down, want_down, "q3_K (down)") && ok;
    for (void* q : {(void*) dx, (void*) dperm, (void*) dtiles, (void*) dxg, (void*) dh, (void*) dy}) cudaFree(q);
    for (GemmExpert& g : ex)
        for (void* q : g.dev) cudaFree(q);

    // timing: 48 experts, the rows of each in its own tiles
    std::vector<GemmExpert> bench = gemm_experts(48, 52, true);
    for (int64_t per : {256, 32}) {
        const int64_t n = 48 * per;
        std::vector<float> xb((size_t) (n * E));
        for (float& v : xb) v = nd(rng);
        std::vector<int32_t> id((size_t) n);
        for (int64_t i = 0; i < n; ++i) id[(size_t) i] = (int32_t) i;
        const std::vector<e::ExpertTile> tl = gemm_tiles(bench, std::vector<int64_t>(48, per));
        float* dxb = upload(xb);
        int32_t* did = upload(id);
        e::ExpertTile* dtl = upload(tl);
        uint16_t *dxg2 = nullptr, *dh2 = nullptr;
        float* dy2 = nullptr;
        ck(cudaMalloc(&dxg2, (size_t) (n * E) * 2), "cudaMalloc");
        ck(cudaMalloc(&dh2, (size_t) (n * F) * 2), "cudaMalloc");
        ck(cudaMalloc(&dy2, (size_t) (n * E) * 4), "cudaMalloc");
        cudaEvent_t ev[4];
        for (cudaEvent_t& v : ev) ck(cudaEventCreate(&v), "cudaEventCreate");
        const int reps = 5;
        float ms[3] = {0, 0, 0};
        for (int rep = 0; rep <= reps; ++rep) {
            ck(cudaEventRecord(ev[0], s), "cudaEventRecord");
            e::expert_gather_rows_f16(dxb, did, 1, n, (int) E, dxg2, s);
            ck(cudaEventRecord(ev[1], s), "cudaEventRecord");
            e::expert_gate_up(dtl, (int64_t) tl.size(), e::kQTypeQ2K, dxg2, E, F, limit, dh2, s);
            ck(cudaEventRecord(ev[2], s), "cudaEventRecord");
            e::expert_gemm(dtl, (int64_t) tl.size(), 2, e::kQTypeQ3K, dh2, F, E, dy2, E, s);
            ck(cudaEventRecord(ev[3], s), "cudaEventRecord");
            ck(cudaEventSynchronize(ev[3]), "cudaEventSynchronize");
            if (rep == 0) continue;   // the warm-up
            for (int i = 0; i < 3; ++i) {
                float t = 0;
                ck(cudaEventElapsedTime(&t, ev[i], ev[i + 1]), "cudaEventElapsedTime");
                ms[i] += t / reps;
            }
        }
        const double gu_flop = (double) n * 4.0 * E * F, down_flop = (double) n * 2.0 * E * F;
        std::printf("(c) timing: 48 experts x %lld rows (%zu tiles): gather %.3f ms, gate/up/clamp %.3f ms (%.1f TFLOP/s), "
                    "down %.3f ms (%.1f TFLOP/s), all %.3f ms (%.1f TFLOP/s)\n",
                    (long long) per, tl.size(), ms[0], ms[1], gu_flop / (ms[1] * 1e9), ms[2], down_flop / (ms[2] * 1e9),
                    ms[0] + ms[1] + ms[2], (gu_flop + down_flop) / ((ms[0] + ms[1] + ms[2]) * 1e9));
        for (cudaEvent_t v : ev) cudaEventDestroy(v);
        for (void* q : {(void*) dxb, (void*) did, (void*) dtl, (void*) dxg2, (void*) dh2, (void*) dy2}) cudaFree(q);
    }
    for (GemmExpert& g : bench)
        for (void* q : g.dev) cudaFree(q);
    cudaStreamDestroy(s);
    std::printf("dsv41_engine_experts_test (c): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- (d)
// The deterministic mode (tools/dsv41/engine/DETERMINISM.md) on the CPU
// reference path, GPU free. Every layer's routed experts of a GGUF decode token by token while the layer's slab
// follows a HotLayerPlan (adaptive, a swap at every token: margin 0 x - 1), driven by the rules ExpertEngine follows
// (hot_policy.hpp hot_request_start at a request's start, hot_boundary_waits at each boundary; begin_request,
// decode_started and decode_boundary in experts.cpp): a hit runs on the engine's numerics reference (engine_expert:
// q8_1 rows and MMVQ's sums, decode's path), a miss on the CPU experts (CpuExperts in the fork's numerics, Q8_K),
// combined in slot order (ref_expert_combine, combine_kernel's order); an expert gives different
// last bits on the two, which is the placement dependence once measured (the GPU's numerics removed it from the
// engine; this check keeps it to test the mode's placement rules, which stay). A token's routing is a fixed random router's top K over
// its input, and the next token's input is the last layer's output after an RMS norm, so one changed bit changes the
// rest of the request. Injected per run: whether a batch's copies are done at each boundary (a random draw, the copy
// streams' speed), the order the pairs' outputs land in (the doorbell's arrival), the CPU pool's threads (1 or 3).
// Requests as det_sampling_check.py's item 7: A (cold), A again (other timing, order and threads), B, A after B. Pass:
// with the mode on, the three A's every output bit for bit the same; with it off (today's rules: no reset, a batch
// still copying deferred) some A differs, so the check sees the failure it guards against; and off with hot_reset=1
// (every request from the hot set's slab, as the benches ran) they still differ: the copies' timing alone.
// in the mode the reset runs at each request's end (end_request,
// hot_request_end_resets), so its start's reset moves nothing (checked) and the A's are still bit for bit the same.
// (host homes, the reset layer by layer): the mode on again with every slab expert's permanent home
// (HotLayerPlan's host homes: real host copies of the experts' bytes, which the misses read; a swap's victim dropped),
// and the reset at a request's end restored in place layer by layer (restore_in_place, HotRestore's order): each layer's
// copies land only when the next request's computation of that layer waits for them (wait_before), and how many layers
// had already landed when it starts is injected per request. The slab is modelled as the bytes each slot holds (a hit
// computes the expert whose bytes are in its slot). Pass: every output bit for bit the two-way run's (same placements,
// same tokens); every slot of a layer holding what the planner says when the layer computes (the placement each layer
// computes with is the hot set's slab); no home ever changed (a dropped victim's home stays valid, the
// misses read it). And without the per-layer wait (the negative control) a layer computes with a slot that still holds
// its victim, and the outputs differ: the check sees the failure it guards against. (partial host
// homes): the mode on with the first half of the layers homed in full, the next one for the end of its slab and the
// rest without homes (spare homes, the two-way path), as the engine's home order leaves them when the RAM is short; a
// victim copied down writes the bytes its slot holds into its new home, which a miss then reads; the layers homed in
// full restore layer by layer, the others in rounds after. Pass: every output bit for bit the two-way run's, some
// victims copied down, no layer computing with a slot not yet restored, every home its expert's bytes.
namespace det {

constexpr int kPromptTokens = 4, kDecodeTokens = 24;

struct Sim {
    const e::Model* m = nullptr;
    e::DenseDims d;
    e::HotConfig cfg;
    double decay = 0;
    std::vector<e::HotLayerPlan> plans;
    std::vector<std::vector<float>> router;          // per layer [n_expert][E]
    std::vector<std::pair<int, e::HotSwap>> batch;   // the open batch (decode_started's)
    bool open = false;
    int64_t swaps = 0, deferred = 0, waited = 0, hits = 0, misses = 0;
    // per layer per slot the expert whose bytes the slot holds; host homes (the bytes of each home,
    // the misses read them), the layered reset (its copies not landed yet per layer: slot and expert; HotRestore's
    // order), whether a layer's computation waits for its copies (false: the negative control), and the computations
    // that found a slot holding other bytes than the planner's
    std::vector<std::vector<int32_t>> content;
    bool host_homes = false, layered = false, layer_wait = true;
    std::vector<std::vector<std::vector<uint8_t>>> homes;   // [layer][home]: parts 0, 1, 2 one after another
    std::vector<std::vector<std::pair<int32_t, int32_t>>> pending;
    e::HotRestore restore;
    int64_t stale = 0, stale_layers = 0, layer_waits = 0;   // hits on another expert's bytes; layer calls with such a slot
    // the victims copied down into a home (the bytes their slot held, as the down copy writes them)
    int64_t downs = 0;
};

struct Run {
    std::vector<std::vector<float>> out;   // per decode token: the last layer's output row
    int64_t hits = 0, misses = 0, swaps = 0, deferred = 0, waited = 0, reset = 0, end_reset = 0;
};

// a layered reset's copies of layer l land (the stream's wait for its event)
void land(Sim& s, int l) {
    for (const auto& [slot, ex] : s.pending[(size_t) l]) s.content[(size_t) l][(size_t) slot] = ex;
    s.pending[(size_t) l].clear();
}

// the bytes expert ex has in layer l's mapping, parts 0, 1, 2 one after another (a home's layout)
std::vector<uint8_t> expert_bytes(const Sim& s, int l, int32_t ex) {
    const e::ExpertLayerHost& L = s.m->experts()[(size_t) l];
    std::vector<uint8_t> b;
    for (int q = 0; q < 3; ++q) b.insert(b.end(), L.data[q] + (size_t) ex * L.bytes[q], L.data[q] + (size_t) (ex + 1) * L.bytes[q]);
    return b;
}

void publish(Sim& s) {
    for (const auto& [l, x] : s.batch) {
        // with host homes modelled, a victim copied down writes the bytes its slot holds into its new home
        if (s.host_homes && x.to_home >= 0) {
            const int32_t old = s.plans[(size_t) l].slot_of(x.out);
            s.homes[(size_t) l][(size_t) x.to_home] = expert_bytes(s, l, s.content[(size_t) l][(size_t) old]);
            ++s.downs;
        }
        s.plans[(size_t) l].publish(x);
        s.content[(size_t) l][(size_t) x.slot] = x.in;   // its copy was done
    }
    s.swaps += (int64_t) s.batch.size();
    s.batch.clear();
    s.open = false;
}

// the slab back to the hot set's in rounds of the spare slots (begin_request's reset, and the end_request); the
// swaps it publishes are not the request's
int64_t reset_slab(Sim& s) {
    const int64_t s0 = s.swaps;
    int64_t n = 0;
    for (;;) {
        std::vector<e::HotSwap> sw;
        for (int l = 0; l < s.m->n_layer(); ++l) {
            sw.clear();
            s.plans[(size_t) l].plan_static(s.cfg.spare, sw);
            for (const e::HotSwap& x : sw) s.batch.emplace_back(l, x);
        }
        if (s.batch.empty()) break;
        n += (int64_t) s.batch.size();
        publish(s);
    }
    s.swaps = s0;
    return n;
}

// the reset in place layer by layer (ExpertEngine's end_request with host homes): the planners at
// the hot set's slab at once, each layer's copies pending until a computation of it waits for them; a
// layer whose victims may need a copy down (partial host homes) left to the rounds after it (end_request's)
int64_t restore_layers(Sim& s) {
    if (s.restore.open()) die("--determinism: a layered reset while the last one is open");
    s.restore.begin(s.m->n_layer());
    int64_t n = 0;
    std::vector<e::HotSwap> sw;
    for (int l = 0; l < s.m->n_layer(); ++l) {
        sw.clear();
        if (s.plans[(size_t) l].restores_in_place()) n += s.plans[(size_t) l].restore_in_place(sw);
        for (const e::HotSwap& x : sw) {
            if (x.to_home >= 0) die("--determinism: a layered reset copied a victim down");
            s.pending[(size_t) l].emplace_back(x.slot, x.in);
        }
        s.restore.record(l);
    }
    return n;
}

// one layer's routed experts on one token: hits on the engine's numerics, misses on the CPU in a shuffled job order,
// every pair's output written in a shuffled order, then the slot-order combine
void layer(Sim& s, e::CpuExperts& cpu, int l, const std::vector<float>& x, std::mt19937& order, std::vector<int32_t>& ids,
           std::vector<float>& out) {
    const e::DenseDims& d = s.d;
    const int K = d.n_expert_used;
    const int64_t E = d.n_embd;
    const e::ExpertLayerHost& L = s.m->experts()[(size_t) l];
    const e::HotLayerPlan& hp = s.plans[(size_t) l];
    // before the layer reads its slab, the wait for its restore (the engine's wait_restore)
    if (s.restore.open() && s.layer_wait && s.restore.wait_before(l)) {
        land(s, l);
        ++s.layer_waits;
    }
    for (int sl = 0; sl < hp.n_slots(); ++sl)
        if (hp.slot_expert(sl) >= 0 && s.content[(size_t) l][(size_t) sl] != hp.slot_expert(sl)) ++s.stale_layers;
    std::vector<std::pair<float, int32_t>> sc((size_t) d.n_expert);
    for (int32_t ex = 0; ex < d.n_expert; ++ex) {
        double a = 0;
        for (int64_t i = 0; i < E; ++i) a += (double) x[(size_t) i] * s.router[(size_t) l][(size_t) (ex * E + i)];
        sc[(size_t) ex] = {(float) a, ex};
    }
    std::partial_sort(sc.begin(), sc.begin() + K, sc.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    ids.assign((size_t) K, 0);
    std::vector<float> w((size_t) K);
    float wsum = 0;
    for (int k = 0; k < K; ++k) {
        ids[(size_t) k] = sc[(size_t) k].second;
        w[(size_t) k] = std::exp(sc[(size_t) k].first - sc[0].first);
        wsum += w[(size_t) k];
    }
    for (float& v : w) v /= wsum;
    std::vector<float> y((size_t) (K * E));
    std::vector<int> pairs((size_t) K);
    for (int k = 0; k < K; ++k) pairs[(size_t) k] = k;
    std::shuffle(pairs.begin(), pairs.end(), order);
    // the misses as CPU jobs, one per expert, in the shuffled order (the engine's is first use)
    std::vector<e::CpuExpertJob> jobs;
    std::vector<int> job_pair;
    for (int k : pairs) {
        if (hp.slot_of(ids[(size_t) k]) >= 0) continue;
        e::CpuExpertJob j;
        const int32_t h = s.host_homes ? hp.home_of(ids[(size_t) k]) : -1;
        for (int q = 0; q < 3; ++q)
            j.rows.part[q] = h >= 0 ? s.homes[(size_t) l][(size_t) h].data() + (q > 0 ? L.bytes[0] : 0) + (q > 1 ? L.bytes[1] : 0)
                                    : L.data[q] + (size_t) ids[(size_t) k] * L.bytes[q];
        j.tokens = {0};
        j.y0 = (int64_t) jobs.size();
        jobs.push_back(j);
        job_pair.push_back(k);
    }
    std::vector<float> ycpu(jobs.size() * (size_t) E);
    if (!jobs.empty()) cpu.run(L, d, x.data(), 1, jobs.data(), jobs.size(), ycpu.data());
    for (size_t j = 0; j < jobs.size(); ++j)
        std::memcpy(y.data() + (size_t) job_pair[j] * E, ycpu.data() + j * (size_t) E, (size_t) E * 4);
    for (int k : pairs) {
        const int32_t slot = hp.slot_of(ids[(size_t) k]);
        if (slot < 0) continue;
        // the expert whose bytes the slot holds (the planner's, unless a copy into it has not landed)
        s.stale += s.content[(size_t) l][(size_t) slot] != ids[(size_t) k];
        engine_expert(L, d, s.content[(size_t) l][(size_t) slot], {x.data()}, false, y.data() + (size_t) k * E);
    }
    s.hits += K - (int64_t) jobs.size();
    s.misses += (int64_t) jobs.size();
    out.assign((size_t) E, 0.f);
    e::ref_expert_combine(y.data(), w.data(), 1, K, (int) E, nullptr, out.data());
}

// the next layer's (and token's) input: x + out after an RMS norm
void next_input(std::vector<float>& x, const std::vector<float>& out) {
    double ss = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = x[i] + out[i];
        ss += (double) x[i] * x[i];
    }
    const float r = 1.0f / std::sqrt((float) (ss / (double) x.size()) + 1e-6f);
    for (float& v : x) v *= r;
}

// one request of the prompt seeded by `prompt`: its start, its prompt read token by token (counted, no swaps: a
// prefill drains and plans nothing), then kDecodeTokens decode tokens with a batch per token and a boundary after each
Run request(Sim& s, bool deterministic, bool reset_key, uint32_t prompt, uint32_t timing, int threads) {
    const e::DenseDims& d = s.d;
    Run r;
    const int64_t s0 = s.swaps, h0 = s.hits, m0 = s.misses, f0 = s.deferred, w0 = s.waited;
    e::CpuExperts cpu(threads, e::CpuNumerics::Fork);
    std::mt19937 when(timing), order(timing ^ 0x9e3779b9u);
    // begin_request: the open batch published, then the reset in rounds of the spare slots
    if (s.open) publish(s);
    const e::HotRequestStart st = e::hot_request_start(deterministic, reset_key, false);
    if (st.reset) r.reset = reset_slab(s);
    s.swaps = s0;
    for (e::HotLayerPlan& hp : s.plans) hp.begin_request(e::HotPolicy::Adaptive, st.resume);
    // a layered reset's progress when this request starts (the copy engine's speed): its first layers landed, the
    // negative control none
    if (s.restore.open()) {
        const int landed = s.layer_wait ? (int) (when() % (uint32_t) (s.m->n_layer() + 1)) : 0;
        for (int l = 0; l < landed; ++l) land(s, l);
    }
    std::mt19937 rng(prompt);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) d.n_embd), out;
    std::vector<int32_t> ids;
    for (int t = 0; t < kPromptTokens; ++t) {
        for (float& v : x) v = nd(rng);
        for (int l = 0; l < s.m->n_layer(); ++l) {
            layer(s, cpu, l, x, order, ids, out);
            s.plans[(size_t) l].count_prompt(ids.data(), (int64_t) ids.size());
            next_input(x, out);
        }
        // after the call's wait every layer's copies have landed and the reset is seen done (end_call's poll); the
        // negative control's copies land after the whole prompt, as a slow copy engine's might
        if (s.restore.open() && (s.layer_wait || t == kPromptTokens - 1)) {
            for (int l = 0; l < s.m->n_layer(); ++l) land(s, l);
            s.restore.done();
        }
    }
    for (e::HotLayerPlan& hp : s.plans) hp.prompt_done(s.cfg);
    for (int t = 0; t < kDecodeTokens; ++t) {
        // decode_started: the next batch when none is open
        if (!s.open) {
            std::vector<e::HotSwap> sw;
            for (int l = 0; l < s.m->n_layer(); ++l) {
                sw.clear();
                s.plans[(size_t) l].plan(s.cfg, sw);
                for (const e::HotSwap& x2 : sw) s.batch.emplace_back(l, x2);
            }
            s.open = !s.batch.empty();
        }
        for (int l = 0; l < s.m->n_layer(); ++l) {
            layer(s, cpu, l, x, order, ids, out);
            s.plans[(size_t) l].count_token(ids.data(), (int) ids.size(), s.decay);
            next_input(x, out);
        }
        r.out.push_back(x);
        // decode_boundary: the copies done or not (the injected timing); still copying, the mode waits or defers
        if (s.open) {
            const bool done = (when() & 1u) != 0;
            if (done) {
                publish(s);
            } else if (e::hot_boundary_waits(deterministic, false)) {
                ++s.waited;
                publish(s);
            } else {
                ++s.deferred;
            }
        }
        for (const e::HotLayerPlan& hp : s.plans)
            if (const std::string why = hp.check(); !why.empty()) die("--determinism: the planner: " + why);
    }
    r.hits = s.hits - h0;
    r.misses = s.misses - m0;
    r.swaps = s.swaps - s0;
    r.deferred = s.deferred - f0;
    r.waited = s.waited - w0;
    // end_request: in the mode the open batch published and the slab reset after the reply, in the idle time
    if (e::hot_request_end_resets(deterministic)) {
        if (s.open) publish(s);
        r.end_reset = s.layered ? restore_layers(s) + reset_slab(s) : reset_slab(s);
        s.swaps = s0 + r.swaps;
    }
    return r;
}

// per layer: every home still holds its expert's bytes as the mapping has them
bool homes_intact(const Sim& s) {
    for (int l = 0; l < s.m->n_layer(); ++l) {
        const e::ExpertLayerHost& L = s.m->experts()[(size_t) l];
        const e::HotLayerPlan& hp = s.plans[(size_t) l];
        for (int32_t ex = 0; ex < hp.n_expert(); ++ex) {
            const int32_t h = hp.home_of(ex);
            if (h < 0) continue;
            const uint8_t* b = s.homes[(size_t) l][(size_t) h].data();
            for (int q = 0; q < 3; ++q) {
                if (std::memcmp(b, L.data[q] + (size_t) ex * L.bytes[q], L.bytes[q]) != 0) return false;
                b += L.bytes[q];
            }
        }
    }
    return true;
}

// the first decode token whose output differs from a's in some bit, -1 when none
int first_differ(const Run& a, const Run& b) {
    for (size_t t = 0; t < a.out.size(); ++t)
        if (bits_differ(a.out[t].data(), b.out[t].data(), a.out[t].size())) return (int) t;
    return -1;
}

}  // namespace det

int run_determinism(const std::string& path) {
    const e::HostModel host = e::load_gguf_model(path);
    e::Model m;
    std::string err;
    if (!m.init(host, true, false, err)) die(err);
    const e::DenseDims& d = m.dense_dims();
    if (d.n_expert < 16) die("--determinism: " + path + " has " + std::to_string(d.n_expert) + " experts (16 or more)");
    for (int l = 0; l < m.n_layer(); ++l) e::cpu_experts_check(m.experts()[(size_t) l], d, e::CpuNumerics::Fork);
    bool ok = true;
    // the mode on; off; off with hot_reset=1 on every request; on with host homes and the reset layer
    // by layer, and the same without the per-layer wait (the negative control); on with partial host
    // homes (the first half of the layers homed in full, the next one for the end of its slab, the rest none)
    std::vector<det::Run> two_way;   // the mode on's runs (the two-way path)
    for (const int mode : {0, 1, 2, 3, 4, 5}) {
        const bool det_on = mode == 0 || mode >= 3, reset_key = mode == 2, homes = mode >= 3, partial = mode == 5;
        const char* name = mode == 0 ? "on" : mode == 1 ? "off" : mode == 2 ? "off with hot_reset=1"
                         : mode == 3 ? "on, host homes, the reset layer by layer"
                         : mode == 4 ? "on, host homes, the reset layer by layer without the per-layer wait"
                                     : "on, partial host homes, the reset layer by layer and in rounds";
        det::Sim s;
        s.host_homes = s.layered = homes;
        s.layer_wait = mode != 4;
        s.m = &m;
        s.d = d;
        s.cfg.policy = e::HotPolicy::Adaptive;
        s.cfg.max_swaps = 2;
        s.cfg.spare = 2;
        s.cfg.half_life = 8;
        s.cfg.margin_mul = 0;
        s.cfg.margin_add = -1;
        e::hot_config_check(s.cfg);
        s.decay = std::pow(2.0, -1.0 / s.cfg.half_life);
        // per layer: a quarter of the experts in the slab, the next half warm in homes (2 spare), the rest cold in an
        // unpinned mapping (never promoted)
        const int32_t NE = d.n_expert, nv = NE / 4, nw = NE / 2;
        e::HotLayerPlan::Layout lay;
        lay.n_expert = NE;
        for (int32_t ex = 0; ex < nv; ++ex) lay.vram.push_back(ex);
        for (int32_t ex = nv; ex < nv + nw; ++ex) lay.warm.push_back(ex);
        lay.spare = s.cfg.spare;
        lay.spare_homes = homes ? 0 : s.cfg.spare;
        if (homes) lay.homed = lay.vram;
        s.plans.resize((size_t) m.n_layer());
        // partial: layers [0, cut) homed in full, layer cut the last half of its slab (as the engine's home order cuts
        // a layer), the rest none, the last two with the spare homes
        const int cut = m.n_layer() / 2;
        int full_layers = 0, cut_layers = 0, none_layers = 0;
        for (int l = 0; l < m.n_layer(); ++l) {
            e::HotLayerPlan::Layout ll = lay;
            if (partial && l >= cut) {
                ll.spare_homes = s.cfg.spare;
                ll.homed.assign(lay.vram.end() - (l == cut ? nv / 2 : 0), lay.vram.end());
            }
            s.plans[(size_t) l].init(ll);
            const int h = s.plans[(size_t) l].homed();
            (h == nv ? full_layers : h > 0 ? cut_layers : none_layers) += homes;
        }
        // the slab's bytes as init fills it; host homes: each home's bytes copied from the mapping
        s.content.assign((size_t) m.n_layer(), {});
        s.pending.assign((size_t) m.n_layer(), {});
        s.homes.assign((size_t) m.n_layer(), {});
        for (int l = 0; l < m.n_layer(); ++l) {
            const e::HotLayerPlan& hp = s.plans[(size_t) l];
            for (int sl = 0; sl < hp.n_slots(); ++sl) s.content[(size_t) l].push_back(hp.slot_expert(sl));
            if (!homes) continue;
            // every home with its expert's bytes (a spare home empty until a victim is copied down into it)
            s.homes[(size_t) l].assign((size_t) hp.n_homes(), {});
            for (int32_t ex = 0; ex < NE; ++ex)
                if (const int32_t h = hp.home_of(ex); h >= 0) s.homes[(size_t) l][(size_t) h] = det::expert_bytes(s, l, ex);
        }
        std::mt19937 rr(20261013u);
        std::normal_distribution<float> nd(0.f, 1.f);
        s.router.assign((size_t) m.n_layer(), std::vector<float>((size_t) (NE * d.n_embd)));
        for (auto& r : s.router)
            for (float& v : r) v = nd(rr);
        constexpr uint32_t kA = 101, kB = 202;
        const det::Run a0 = det::request(s, det_on, reset_key, kA, 1, 1);
        const det::Run a1 = det::request(s, det_on, reset_key, kA, 2, 3);
        const det::Run b = det::request(s, det_on, reset_key, kB, 3, 1);
        const det::Run a2 = det::request(s, det_on, reset_key, kA, 4, 3);
        const int p1 = det::first_differ(a0, a1), p2 = det::first_differ(a0, a2);
        auto line = [](const char* what, const det::Run& r) {
            std::printf("  %s: %lld hits, %lld misses, %lld swaps (%lld deferred, %lld waited), reset %lld swaps before "
                        "it and %lld after it\n", what, (long long) r.hits, (long long) r.misses, (long long) r.swaps,
                        (long long) r.deferred, (long long) r.waited, (long long) r.reset, (long long) r.end_reset);
        };
        std::printf("%s, deterministic %s (%d prompt and %d decode tokens per request, %d layers):\n", path.c_str(),
                    name, det::kPromptTokens, det::kDecodeTokens, m.n_layer());
        line("A cold", a0);
        line("A again (other timing, arrival order, 3 threads)", a1);
        line("B", b);
        line("A after B", a2);
        const bool same = p1 < 0 && p2 < 0;
        // in the mode the reset is after each request, so no request's start moves an expert, and every request
        // that swapped has its end reset them back
        const bool moved = det_on && (a0.reset || a1.reset || b.reset || a2.reset || !a0.end_reset || !a1.end_reset ||
                                      !b.end_reset || !a2.end_reset);
        // the mode off must show today's failure, else this test could not see it
        bool want = det_on ? same && !moved : !same;
        if (mode == 0) two_way = {a0, a1, b, a2};
        if (homes) {
            // against the two-way runs request by request, every slot as the planner says when its
            // layer computes, every home its expert's bytes
            const std::vector<det::Run> runs = {a0, a1, b, a2};
            int differ = -1;
            for (size_t i = 0; i < runs.size() && differ < 0; ++i)
                if (det::first_differ(two_way[i], runs[i]) >= 0) differ = (int) i;
            const bool intact = det::homes_intact(s);
            const char* req[] = {"A cold", "A again", "B", "A after B"};
            if (mode == 5) {
                // the layers short of homes copy victims down (into homes the misses then read), and the
                // outputs are still the two-way run's
                const bool good = differ < 0 && s.stale == 0 && s.stale_layers == 0 && intact && s.layer_waits > 0 &&
                                  s.downs > 0 && full_layers > 0 && cut_layers > 0;
                std::printf("%s partial host homes (%d layers homed in full, %d in part, %d none): every request bit for "
                            "bit the two-way run's, %lld victims copied down, %lld layer waits, %lld layers computed with a "
                            "slot not yet restored, every home its expert's bytes: %s\n", good ? "ok  " : "FAIL", full_layers,
                            cut_layers, none_layers, (long long) s.downs, (long long) s.layer_waits,
                            (long long) s.stale_layers,
                            differ >= 0 ? (std::string(req[differ]) + " differs").c_str() : intact ? "yes" : "a home changed");
                want = want && good;
            } else if (mode == 3) {
                const bool good = differ < 0 && s.stale == 0 && s.stale_layers == 0 && intact && s.layer_waits > 0;
                std::printf("%s host homes, layer by layer: every request bit for bit the two-way run's, %lld layer waits, "
                            "%lld layers computed with a slot not yet restored, every home its expert's bytes: %s\n",
                            good ? "ok  " : "FAIL", (long long) s.layer_waits, (long long) s.stale_layers,
                            differ >= 0 ? (std::string(req[differ]) + " differs").c_str() : intact ? "yes" : "a home changed");
                want = want && good;
            } else {
                const bool seen = differ >= 0 && s.stale > 0;
                std::printf("%s host homes, layer by layer without the per-layer wait (negative control): %lld hits on a "
                            "slot still holding its victim, %s\n", seen ? "ok  " : "FAIL", (long long) s.stale,
                            differ >= 0 ? (std::string(req[differ]) + " differs from the two-way run, as expected").c_str()
                                        : "every request the same (the check cannot see a missing wait)");
                // A's against each other may or may not differ here: the verdict is the stale reads and the difference
                const bool w2 = seen && !moved;
                ok = ok && w2;
                continue;
            }
        }
        if (det_on)
            std::printf("%s deterministic %s: every reset after its request (C02): %s\n", moved ? "FAIL" : "ok  ", name,
                        moved ? "a request's start reset, or one ended without its reset" : "none before one, each after one");
        std::printf("%s deterministic %s: A again %s, A after B %s%s\n", want ? "ok  " : "FAIL", name,
                    p1 < 0 ? "bit-identical to A cold" : ("first differs at decode token " + std::to_string(p1)).c_str(),
                    p2 < 0 ? "bit-identical to A cold" : ("first differs at decode token " + std::to_string(p2)).c_str(),
                    det_on ? "" : (same ? " (expected a difference: the mode off defers a batch still copying)"
                                        : " (today's behaviour, as expected)"));
        ok = ok && want;
    }
    std::printf("dsv41_engine_experts_test (d): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() >= 2 && args[0] == "--cases") return run_cases(std::vector<std::string>(args.begin() + 1, args.end()));
    if (!args.empty() && args[0] == "--gpu") {
        bool dry = false;
        std::vector<std::string> models;
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--dry-run") dry = true;
            else models.push_back(args[i]);
        }
        return run_gpu(models, dry);
    }
    if (args.size() == 2 && args[0] == "--determinism") return run_determinism(args[1]);
    if (!args.empty() && args[0] == "--gemm") {
        if (args.size() > 2 || (args.size() == 2 && args[1] != "--dry-run")) die("--gemm [--dry-run]");
        return run_gemm(args.size() == 2);
    }
    std::fprintf(stderr, "usage: dsv41_engine_experts_test --cases FILE [FILE ...] | --gpu [--dry-run] GGUF HOT_DIR ... | "
                         "--gemm [--dry-run] | --determinism GGUF\n");
    return 2;
}
