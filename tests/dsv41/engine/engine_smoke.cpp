// tests/dsv41/engine/engine_smoke.cpp - dsv41_engine_smoke (a bench: reported): the whole
// engine at the real model's shapes on a synthetic model that fits a dev pod, since no 246 GiB model is loaded.
//
//   dsv41_engine_smoke [--layers N] [--experts N] [--vram-experts N | --vram-experts-gb G] [--ctx N] [--chunk N]
//                      [--prompt N] [--n-predict N] [--table-rows N] [--threads N] [--ring-set N] [--ring-sets N]
//                      [--prefetch-experts N] [--pool-spin-us N] [--memory-table] [--pod-link GEN:WIDTH]
//                      [--pod-tflops T] [--project NAME:GEN:WIDTH:VRAM_GIB:TFLOPS]... [--counts FILE]
//                      [--prefix-cache N [--floor-target NAME:GEN:WIDTH]] [--dry-run]
//
// The model: the real dims, roles and tensor types (SPEC.md "Tensors": q2_K matrices, q3_K downs, q6_K head, bf16
// router, f32 norms) of the first --layers layers (default 40; the engram on those of layers 1 and 14 it has), with
// --experts routed experts per layer (default 8, 6 used; 384 is the real count) and engram tables of the real row width
// with --table-rows rows each (default 4M, 336 MiB). Weights are random blocks as
// tiny_dsv41.py makes them (qmatmul_synth.hpp), about 1 / sqrt(fan_in) spread; the routed experts are made straight
// into pinned host memory, as the target keeps the whole model (so the copies read them in place, experts.hpp). The
// first experts of every layer sit in a VRAM slab through a hot-set file: --vram-experts per layer (default half), or
// as many as --vram-experts-gb GiB holds over all layers (16.12 GiB is 44% of 8 x 384 experts, the target's hot
// fraction); the others stay cold in pinned RAM.
//
// --counts FILE: the routing and the hot set of the real model instead of the synthetic router's (random
// weights route nearly uniformly). FILE is a strata-moe-counts/1 serving profile (a hybrid serving run, 1530
// decode calls per layer); smoke layer l stands for its layer l * 40 / --layers (8 layers: 0, 5, .., 35). Every call's
// router ids are drawn per token from that layer's decode counts plus one (n_used distinct experts; the router's
// weights for them, as the forced mode), and the slab holds the same number of experts, chosen by hot_set.py's rule
// (a global greedy by count over the layers), so the decode hit rate and the prefill pass sizes are the real model's.
//
// The run: with --memory-table first the target's non-expert device memory per chunk of 2048 to --chunk tokens: the
// real 40 layers at ctx 262,144 (a model of max(6, --prefetch-experts) experts per layer, so the prefetch region is
// the target's; the router rows of the other experts up to 384 added, 40 x 10,244 B per expert), the dense model, the
// KV cache, the attention, dense and forward workspaces, the expert call buffers, the ring (its share printed: 3 sets
// of 48 experts) and the prefetch region, measured (cudaMemGetInfo) and freed. Then the pinned H2D bandwidth (256 MiB copies) and the largest chunk from
// --chunk (default 4096) down to 2048 that fits beside the slab.
// A random prompt of --prompt tokens (default: that chunk, one call) at --ctx (default 16384, raised to fit) is
// prefilled with per-stage times; per layer its front (attention, dense ops) and its expert phase (copy GB and GB/s,
// compute, overlap, idle gaps; experts.hpp ExpertLayerStats) are printed, then the two gates' numbers (copy GB/s
// against the pinned H2D bandwidth, copy/compute overlap) and the routed experts' compute in TFLOP/s against
// --pod-tflops (gate: at least 35%), split by launch (gather, gate/up, down) with the passes' width (tiles per
// pass, blocks per launch, the GPU's SMs). Then --n-predict greedy steps three times: timed (per
// stage, every op launched; the split per layer per token printed), untimed (ms per token, each token one CUDA graph,
// host syncs, graph launches, doorbells and CPU pool handoffs per token), and untimed with the pool spinning
// --pool-spin-us (default 50) before it blocks; each with its hit rate, the CPU misses' ms and GB/s (per thread too)
// and the 40-layer projection: (ms per token - head) x 40 / --layers + head, the head the timed phase's
// (target: >= 30 tok/s on the A100 with --counts). Then, at the next position: each layer's decode
// attention alone in a graph of its own, 20 replays (Forward::profile_decode_attention: the device ms per layer and
// the split by fused launch from the graph's event nodes), its mean per layer over these layers, by role, and over the
// real 40 layers by their roles (a role not measured: Full's for Reindex, else the mean); the head alone in a graph
// (20 replays) against the device-to-device copy bandwidth and the published HBM bandwidth (memory clock and bus
// width); and one line of the cycle's targets: attention <= 0.15 ms per layer (the 40 layers' mean), hyper-connections
// and norms <= 0.06 ms per layer (the timed phase), the head >= 70% of the published HBM bandwidth (of the copy's when
// the device reports none), >= 30 tok/s (the pool-spin phase's projection). Then: each layer's hyper-connections
// and norms alone in a graph (Forward::profile_decode_hc, 20 replays: the token graph's two hc_decode_step), each step
// alone with its %globaltimer phase marks (the split of where its device time goes), and three requests on the
// adaptive hot set, each from the static slab (hot_reset), a one-token prompt, then --n-predict steps from the token
// graph with the pool spinning, its ms per token (wall clock, the boundaries' host work included) against the static
// pool-spin phase's, both hit rates printed, and where its token's time goes (the graph's device ms; the host's
// inputs and launch, the swap copies queued, the doorbells, the stream's end, the boundary; the static phase's
// beside it): first copy only (HotConfig::copy_only) with a swap
// whenever a layer can start one (margin 0:-1: count(in) > 0 x count(out) - 1, at most 2 per layer per token), every
// swap's copies made, the promoted expert up and the victim down, the victims of this pinned synthetic model into
// scratch homes as the real model's go into theirs, and none published, so the slab and the hit rate stay the static
// phase's (the run fails if a swap was published, no copy ran or a layer's slab moved); then copy only at the default
// margin (1.5 x + 1, for reference); then the adaptive request with a swap whenever a layer can start one, its swaps
// published (the slab and the hit rate move, its ratio printed for comparison only). Then the targets:
// hyper-connections and norms <= 0.10 ms per layer (graph), and the first copy-only run's token <= 1.1x the static
// one with its batches deferred whenever a batch outlasts a token (a boundary that finds the batch
// copying leaves it open, so a batch longer than a token shows deferred boundaries, not a longer token). Every phase
// before them keeps the hot set static (--hot-policy static).
// after the prefill's per-layer lines, one line of the three dev-pod targets on the real
// 40 layers: the attention per layer by role (<= 30 ms at 16K), the routed-expert compute (<= 2.4 s) and the plan gaps
// plus stalls (<= 0.6 s), both scaled by 40 / --layers; a target at another chunk or expert count is printed as
// measured, its verdict the 16K bar's.
//
// The projection to 40 layers (--project, with --memory-table and --pod-link: this pod's PCIe link, `nvidia-smi
// --query-gpu=pcie.link.gen.max,pcie.link.width.max`), per target (VRAM_GIB of device memory, TFLOPS its published
// dense FP16 tensor throughput) and per layer of the real 40 by its attention role:
//   link        raw PCIe: GEN 3/4/5 at 8/16/32 GT/s per lane, x WIDTH lanes, 128b/130b, / 8 bits (x16: 15.8/31.5/63.0 GB/s)
//   bandwidth   B = link_target x (this run's ring GB/s / link_pod): the copy efficiency measured here, carried over
//   hot         HOT = (VRAM_GIB - the measured non-expert GiB at the chunk - 2 GiB margin) / the slab of all 40 x 384
//               experts (183.3 GiB), within 0..1; printed for every chunk measured,
//               with the copies of a 16K prompt at it (16384 / chunk x 40 x C_bytes / B, the copy bound)
//   bytes       C_bytes = (1 - HOT) x 384 x 12,810,240 B: every non-resident expert, once per chunk
//   scale       S = 1 (this pod's compute) and S = TFLOPS / --pod-tflops (the published ratio; only with --pod-tflops)
//   front       F = the measured front of the layer's role (attention + dense ops; a role not measured: Full's for
//               Reindex, else the measured mean) / S
//   compute     X = the measured mean expert compute per layer / S
//   prefetch    P = min(F, prefetch_experts x 12,810,240 B / B): the copies hidden behind the front
//   layer       T = F + max(X, C_bytes / B - P)
//   TTFT 16K    (16384 / chunk) x (sum of T over the 40 layers + (the measured embed and head) / S), the copies
//               repeated per chunk, at the run's chunk; printed only when the model has the real 384 experts (the
//               compute and the routing depend on it)
// The exit code: every speed target above is a PASS or FAIL
// line in the output, never the exit code, which is 0 unless a call fails, a logit is not finite or a copy-only check
// fails (2, or the engine library's own 1); --prefix-cache's correctness gates below are exit codes. --dry-run (the
// dev VM, no CUDA call) checks the arguments and prints the plan: the model's size, the slab, the pinned RAM, the
// chunks tried, the targets.
//
// --prefix-cache N (the fast second turn; instead of the run above, at --chunk exactly): the prefix cache
// of session.hpp on the same model and routing. A conversation's turn 1 is a random prompt of --prompt tokens
// (default the chunk) and --n-predict greedy tokens; turn 2 is turn 1's prompt, its generated tokens and N new ones,
// so it resumes from the live state (all of turn 1 but its last token, which turn 2 reads with the new ones). Four
// engines, one after the other on the same model: (A) free: turn 1, then turn 2 timed per stage (its RESUME, its time
// to the first token), then the same tokens again, which no snapshot covers, read cold from position 0 (its TTFT, and
// the free logits of both); the 40-layer projection of turn 2's TTFT is (TTFT - embed - head) x 40 / --layers + embed
// + head (this pod's link and hot fraction, compute scale 1; goal: <= 1 s). Turn 2 is a short read, so its routed
// experts take the prefill split (experts.hpp): its pairs on the slab, through the ring and on the CPU,
// the ring's copy time (device) and the CPU's (host), against the split's estimates and the rates measured at init.
// Then the physical floor of turn 2's experts on --floor-target (default RTX PRO 6000:5:16): the bytes of the experts
// outside the slab it touches (x 40 / --layers) over the sum of two bandwidths, the target's PCIe (its link x this
// pod's copy efficiency, the pinned H2D over --pod-link's raw link; 1 without --pod-link) and the CPU pool's GB/s
// measured here (the target's host is not measured: this pod's figure stands in), printed with the rest of turn 2
// (TTFT less its routed experts, projected as above) against the 1 s goal, and the same floor on this pod. (B) and
// (D): turn 2's tokens, and turn
// 1's prompt with N other new tokens, each read cold with their selections dumped (ForwardConfig::dump_values off:
// the router's and the index sources' only). (C) forced from B's dump (selection.hpp: a resumed call takes part of a
// reference call): turn 1 feeding A's tokens, turn 2 (a live resume) against B's logits; then forced from D's: turn 1's
// prompt again (cold, one token fed), and D's tokens, which extend only its prompt, so the prompt-end snapshot is
// restored, against D's logits. Gates (exit 1 otherwise): both resumes of the expected kind and length, both rel L2
// <= 5e-2 (forced parity is the correctness gate). Each engine is destroyed, its pinned
// host memory freed, before the next is built, and the host's MemAvailable is printed before and after each one.
#include "qmatmul_synth.hpp"

#include "strata/dsv41/engine/experts.hpp"
#include "strata/dsv41/engine/experts_gemm.hpp"
#include "strata/dsv41/engine/forward.hpp"
#include "strata/dsv41/engine/model.hpp"
#include "strata/dsv41/engine/selection.hpp"
#include "strata/dsv41/engine/session.hpp"

#include <cuda_runtime.h>
#include <sys/stat.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "dsv41_engine_smoke: %s\n", m.c_str());
    std::exit(2);
}

void ck(cudaError_t r, const char* what) {
    if (r != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(r));
}

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kExpertBytes = 12810240.0;   // one real expert (SPEC.md): q2_K up and gate, q3_K down
constexpr int kRealExperts = 384;
constexpr int kRealLayers = 40;
constexpr int64_t kTargetCtx = 262144;   // the goal's context: the KV cache and the indexer's rows at it
constexpr double kMarginGiB = 2.0;       // VRAM left free on a target besides what the memory table measures
constexpr int kDecodePhases = 3;         // timed, token graph, token graph with the pool spinning
constexpr int kHotRequests = 3;          // copy only (stress), copy only (default margin), published (stress)

// pinned host memory, freed with its owner
struct PinnedFree {
    void operator()(uint8_t* p) const { cudaFreeHost(p); }
};

// the synthetic model's bytes, owned (HostModel::keep)
struct Store {
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<std::unique_ptr<uint8_t, PinnedFree>> pinned;
};

struct Builder {
    std::shared_ptr<Store> store = std::make_shared<Store>();
    std::vector<std::pair<std::string, e::HostTensor>> pending;

    void put(const std::string& name, int type, std::vector<int64_t> ne, const uint8_t* data, size_t bytes) {
        e::HostTensor t;
        t.type = type;
        t.ne = std::move(ne);
        t.data = data;
        t.bytes = bytes;
        pending.emplace_back(name, t);
    }
    void add(const std::string& name, int type, std::vector<int64_t> ne, std::vector<uint8_t> bytes) {
        store->blobs.push_back(std::move(bytes));
        put(name, type, std::move(ne), store->blobs.back().data(), store->blobs.back().size());
    }
    // random blocks of rows x k values made in pinned host memory
    void add_pinned(const std::string& name, int type, std::vector<int64_t> ne, int64_t rows, int64_t k, float std,
                    std::mt19937_64& rng) {
        const size_t bytes = qsynth::random_blocks_bytes(type, rows, k);
        void* p = nullptr;
        ck(cudaHostAlloc(&p, bytes, cudaHostAllocDefault), ("pinned " + name).c_str());
        store->pinned.emplace_back(static_cast<uint8_t*>(p));
        qsynth::random_blocks_into(static_cast<uint8_t*>(p), type, rows, k, std, rng);
        put(name, type, std::move(ne), static_cast<uint8_t*>(p), bytes);
    }
};

std::vector<uint8_t> quant(std::mt19937_64& rng, int type, int64_t k, int64_t rows, float std) {
    return qsynth::random_blocks(type, rows, k, std, rng);
}

std::vector<uint8_t> f32s(std::mt19937_64& rng, int64_t n, float mean, float std) {
    std::normal_distribution<float> nd(mean, std);
    std::vector<uint8_t> b((size_t) n * 4);
    for (int64_t i = 0; i < n; ++i) {
        const float v = nd(rng);
        std::memcpy(b.data() + i * 4, &v, 4);
    }
    return b;
}

std::vector<uint8_t> bf16s(std::mt19937_64& rng, int64_t n, float std) {
    std::normal_distribution<float> nd(0.f, std);
    std::vector<uint8_t> b((size_t) n * 2);
    for (int64_t i = 0; i < n; ++i) {
        const float v = nd(rng);
        uint32_t u;
        std::memcpy(&u, &v, 4);
        const uint16_t x = (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
        std::memcpy(b.data() + i * 2, &x, 2);
    }
    return b;
}

bool is_prime(uint64_t p) {
    if (p < 2) return false;
    for (uint64_t q = 2; q * q <= p; ++q)
        if (p % q == 0) return false;
    return true;
}

// one layer's tensors (its own generator, so layers are made on several threads)
void make_layer(Builder& b, const e::HostModel& h, int il, uint64_t seed, int64_t table_rows) {
    std::mt19937_64 rng(seed);
    const e::AttnDims& a = h.attn;
    const e::DenseDims& d = h.dense;
    const e::AttnLayer& l = h.layers[(size_t) il];
    const int64_t E = d.n_embd, HD = (int64_t) a.n_head * a.head_dim, OO = (int64_t) a.o_groups * a.o_lora;
    const std::string p = "blk." + std::to_string(il) + ".";
    auto mat = [&](const std::string& n, int type, int64_t k, int64_t rows) {
        b.add(p + n + ".weight", type, {k, rows}, quant(rng, type, k, rows, 1.f / std::sqrt((float) k)));
    };
    auto vec = [&](const std::string& n, int64_t count, float mean, float std) {
        b.add(p + n, e::kTypeF32, {count}, f32s(rng, count, mean, std));
    };
    mat("attn_q_a", e::kQTypeQ2K, E, a.q_lora);
    mat("attn_q_b", e::kQTypeQ2K, a.q_lora, HD);
    mat("attn_kv", e::kQTypeQ2K, E, a.head_dim);
    mat("attn_output_a", e::kQTypeQ2K, HD / a.o_groups, OO);
    mat("attn_output_b", e::kQTypeQ2K, OO, E);
    if (l.kv_source) {
        mat("attn_compressor_kv", e::kQTypeQ2K, E, a.head_dim);
        vec("attn_compressor_norm.weight", a.head_dim, 1.f, 0.1f);
    }
    if (l.gate) mat("attn_compressor_gate", e::kQTypeQ2K, E, a.head_dim);
    if (l.key_owner) {
        mat("indexer.attn_k", e::kQTypeQ2K, a.head_dim, a.idx_dim);
        vec("indexer.k_norm.weight", a.idx_dim, 1.f, 0.1f);
    }
    if (l.index_source) {
        mat("indexer.attn_q_b", e::kQTypeQ2K, a.q_lora, (int64_t) a.idx_heads * a.idx_dim);
        mat("indexer.proj", e::kQTypeQ2K, E, a.idx_heads);
    }
    vec("attn_q_a_norm.weight", a.q_lora, 1.f, 0.1f);
    vec("attn_kv_a_norm.weight", a.head_dim, 1.f, 0.1f);
    vec("attn_sinks.weight", a.n_head, 0.f, 1.f);
    for (const char* s : {"hc_attn", "hc_ffn"}) {
        mat(std::string(s) + "_fn", e::kQTypeQ2K, (int64_t) e::kHc * E, e::kHcMix);
        vec(std::string(s) + "_base.weight", e::kHcMix, 0.f, 0.5f);
        vec(std::string(s) + "_scale.weight", 3, 1.f, 0.25f);
    }
    vec("attn_norm.weight", E, 1.f, 0.1f);
    vec("ffn_norm.weight", E, 1.f, 0.1f);
    b.add(p + "ffn_gate_inp.weight", e::kTypeBF16, {E, d.n_expert}, bf16s(rng, E * d.n_expert, 1.f / std::sqrt((float) E)));
    vec("exp_probs_b.bias", d.n_expert, 0.f, 0.05f);
    mat("ffn_gate_shexp", e::kQTypeQ2K, E, d.n_ff);
    mat("ffn_up_shexp", e::kQTypeQ2K, E, d.n_ff);
    mat("ffn_down_shexp", e::kQTypeQ3K, d.n_ff, E);
    b.add_pinned(p + "ffn_up_exps.weight", e::kQTypeQ2K, {E, d.n_ff, d.n_expert}, d.n_ff * d.n_expert, E,
                 1.f / std::sqrt((float) E), rng);
    b.add_pinned(p + "ffn_gate_exps.weight", e::kQTypeQ2K, {E, d.n_ff, d.n_expert}, d.n_ff * d.n_expert, E,
                 1.f / std::sqrt((float) E), rng);
    b.add_pinned(p + "ffn_down_exps.weight", e::kQTypeQ3K, {d.n_ff, E, d.n_expert}, E * d.n_expert, d.n_ff,
                 1.f / std::sqrt((float) d.n_ff), rng);
    if (h.engram.index_of(il) >= 0) {
        const int64_t EK = (int64_t) d.engram_cols() * d.engram_key;
        mat("engram_wkv", e::kQTypeQ2K, EK, E * (e::kHc + 1));
        b.add(p + "engram_q.weight", e::kQTypeQ2K, {E, e::kHc}, quant(rng, e::kQTypeQ2K, E, e::kHc, 1.f));
        b.add(p + "engram_k.weight", e::kQTypeQ2K, {E, e::kHc}, quant(rng, e::kQTypeQ2K, E, e::kHc, 1.f));
        b.add(p + "engram_embd.weight", e::kQTypeQ2K, {d.engram_key, table_rows},
              quant(rng, e::kQTypeQ2K, d.engram_key, table_rows, 1.f));
    }
}

e::HostModel synthetic_model(int n_layer, int n_expert, int64_t table_rows) {
    e::HostModel h;
    h.source = "synthetic: the real dims, " + std::to_string(n_layer) + " layers, " + std::to_string(n_expert) +
               " experts per layer";
    h.attn = e::attn_dims_real();
    h.dense = e::dense_dims_real();
    h.dense.n_expert = n_expert;
    h.dense.n_expert_used = std::min(6, n_expert);
    h.layers = e::attn_layers_real();
    h.layers.resize((size_t) n_layer);
    h.clamp_exp.assign((size_t) n_layer, 10.f);
    // the engram hash of the real layers 1 and 14 that the model has: primes just below table_rows / buckets, offsets
    // from 0 per layer
    e::EngramHash& eh = h.engram;
    eh.n_gram = h.dense.engram_ngram;
    eh.heads = h.dense.engram_heads;
    eh.pad = 2;
    for (int32_t l : {1, 14})
        if (l < n_layer) eh.layer_ids.push_back(l);
    std::mt19937_64 rng(5);
    for (size_t li = 0; li < eh.layer_ids.size(); ++li) {
        for (int i = 0; i < eh.n_gram; ++i) eh.multipliers.push_back(rng() | 1u);
        uint64_t acc = 0;
        for (int c = 0; c < eh.cols(); ++c) {
            uint64_t p = (uint64_t) (table_rows / eh.cols()) - (uint64_t) c;
            while (!is_prime(p)) --p;
            eh.primes.push_back(p);
            eh.offsets.push_back(acc);
            acc += p;
        }
    }
    eh.token_map.resize((size_t) h.dense.n_vocab);
    for (int32_t& t : eh.token_map) t = (int32_t) (rng() % (uint64_t) (h.dense.n_vocab * 3 / 4));
    eh.check();

    std::vector<Builder> per_layer((size_t) n_layer);
    {
        std::vector<std::thread> th;
        const int nt = (int) std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
        for (int t = 0; t < nt; ++t)
            th.emplace_back([&, t] {
                for (int il = t; il < n_layer; il += nt) make_layer(per_layer[(size_t) il], h, il, 1000 + (uint64_t) il, table_rows);
            });
        for (std::thread& x : th) x.join();
    }
    Builder top;
    const int64_t E = h.dense.n_embd;
    top.add("token_embd.weight", e::kQTypeQ2K, {E, h.dense.n_vocab}, quant(rng, e::kQTypeQ2K, E, h.dense.n_vocab, 1.f));
    top.add("output.weight", e::kQTypeQ6K, {E, h.dense.n_vocab}, quant(rng, e::kQTypeQ6K, E, h.dense.n_vocab, 1.f / std::sqrt((float) E)));
    top.add("output_norm.weight", e::kTypeF32, {E}, f32s(rng, E, 1.f, 0.1f));
    // one owner for every layer's bytes
    struct All {
        std::vector<std::shared_ptr<Store>> parts;
    };
    auto all = std::make_shared<All>();
    all->parts.push_back(top.store);
    for (Builder& b : per_layer) {
        all->parts.push_back(b.store);
        for (auto& [n, t] : b.pending) h.tensors.emplace(n, t);
    }
    for (auto& [n, t] : top.pending) h.tensors.emplace(n, t);
    h.keep = all;
    return h;
}

void print_times(const char* what, const e::ForwardTimes& t, const e::ExpertTimes& x) {
    std::printf("%s: %lld calls, %lld tokens, %.1f ms (%.2f ms per call): embed %.2f, engram %.2f, attention %.2f, hc and "
                "norms %.2f, router %.2f, shared expert %.2f, routed experts %.2f (host: routing fetch %.2f, doorbell wait "
                "%.2f for %lld rings, plan %.2f, CPU %.2f, queue %.2f, staging %.2f and its waits %.2f; %lld slab pairs, "
                "%lld other, %lld experts through the ring, %lld prefetched, %lld of them used), head %.2f ms; %lld host "
                "syncs, %lld graph launches\n",
                what, (long long) t.calls, (long long) t.tokens, t.total_ms, t.calls ? t.total_ms / (double) t.calls : 0.0,
                t.embed_ms, t.engram_ms, t.attention_ms, t.hc_ms, t.router_ms, t.shared_ms, t.experts_ms, x.fetch_ms,
                x.ring_wait_ms, (long long) x.doorbells, x.plan_ms, x.cpu_ms, x.queue_ms, x.stage_ms, x.stage_wait_ms,
                (long long) x.hits, (long long) x.misses,
                (long long) x.ring_experts, (long long) x.prefetched, (long long) x.prefetch_used, t.head_ms,
                (long long) t.host_syncs, (long long) t.graph_launches);
}

// where a token graph phase's wall ms per token go (ForwardTimes, ExpertTimes), n tokens
void print_where(const char* what, double wall, const e::ForwardTimes& t, const e::ExpertTimes& x, double n) {
    if (t.graph_launches == 0) return;
    const double launch = t.launch_ms / n, queue = x.swap_queue_ms / n, serve = t.serve_ms / n, tail = t.tail_ms / n,
                 boundary = t.boundary_ms / n;
    std::printf("dsv41_engine_smoke: %s, where a token's %.2f ms (wall) go: the graph %.2f ms on the device; the host's "
                "inputs and launch %.3f, swap copies queued %.3f, doorbells %.2f (polls %.2f, CPU misses %.2f), the "
                "stream's end and the logits %.2f, the boundary %.3f, the rest %.3f (the caller's loop)\n",
                what, wall, t.graph_ms / n, launch, queue, serve, x.ring_wait_ms / n, x.cpu_ms / n, tail, boundary,
                wall - launch - queue - serve - tail - boundary);
}

// raw PCIe bandwidth in GB/s of GEN:WIDTH (128b/130b from gen 3 on)
double link_gbps(int gen, int width) {
    const double gt = gen == 3 ? 8.0 : gen == 4 ? 16.0 : gen == 5 ? 32.0 : gen == 6 ? 64.0 : 0.0;
    if (gt == 0.0 || width < 1) die("a PCIe link GEN:WIDTH with GEN 3 to 6, " + std::to_string(gen) + ":" + std::to_string(width));
    return gt * width * (128.0 / 130.0) / 8.0;
}

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, c);) out.push_back(x);
    return out;
}

struct Target {
    std::string name;
    double link = 0, vram_gib = 0, tflops = 0;
};

// one chunk size's non-expert device memory on the target (the memory table)
struct ChunkMemory {
    int64_t chunk = 0;
    double model_gib = 0, forward_gib = 0;
    double ring_gib = 0;   // of forward_gib: the prefill ring (3 sets of 48 experts)
    double total_gib() const { return model_gib + forward_gib; }
};

// pinned H2D: 256 MiB per copy, after one warm copy
double h2d_gbps() {
    const size_t bytes = size_t(256) << 20;
    void *h = nullptr, *d = nullptr;
    ck(cudaHostAlloc(&h, bytes, cudaHostAllocDefault), "cudaHostAlloc (H2D bench)");
    ck(cudaMalloc(&d, bytes), "cudaMalloc (H2D bench)");
    std::memset(h, 1, bytes);
    cudaStream_t s = nullptr;
    cudaEvent_t a = nullptr, b = nullptr;
    ck(cudaStreamCreate(&s), "cudaStreamCreate");
    ck(cudaEventCreate(&a), "cudaEventCreate");
    ck(cudaEventCreate(&b), "cudaEventCreate");
    ck(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s), "H2D");
    const int reps = 8;
    ck(cudaEventRecord(a, s), "cudaEventRecord");
    for (int i = 0; i < reps; ++i) ck(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s), "H2D");
    ck(cudaEventRecord(b, s), "cudaEventRecord");
    ck(cudaEventSynchronize(b), "cudaEventSynchronize");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaStreamDestroy(s);
    cudaFree(d);
    cudaFreeHost(h);
    return (double) bytes * reps / (ms * 1e-3) / 1e9;
}

// device-to-device copy bandwidth (bytes read plus written per second, 512 MiB copies), the HBM
// figure the head is held against beside the published one
double d2d_gbps() {
    const size_t bytes = size_t(512) << 20;
    void *a = nullptr, *b = nullptr;
    ck(cudaMalloc(&a, bytes), "cudaMalloc (D2D bench)");
    ck(cudaMalloc(&b, bytes), "cudaMalloc (D2D bench)");
    ck(cudaMemset(a, 1, bytes), "cudaMemset (D2D bench)");
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    ck(cudaEventCreate(&e0), "cudaEventCreate");
    ck(cudaEventCreate(&e1), "cudaEventCreate");
    ck(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice), "D2D");
    const int reps = 8;
    ck(cudaEventRecord(e0), "cudaEventRecord");
    for (int i = 0; i < reps; ++i) ck(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice), "D2D");
    ck(cudaEventRecord(e1), "cudaEventRecord");
    ck(cudaEventSynchronize(e1), "cudaEventSynchronize");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(a);
    cudaFree(b);
    return 2.0 * (double) bytes * reps / (ms * 1e-3) / 1e9;
}

// the published HBM bandwidth from the device's memory clock and bus width (2 transfers per clock), 0 if unknown
double hbm_peak_gbps() {
    int clock_khz = 0, bus_bits = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, 0) != cudaSuccess ||
        cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

const char* role_name(e::AttnRole r) {
    switch (r) {
        case e::AttnRole::Window: return "window";
        case e::AttnRole::Full: return "Full";
        case e::AttnRole::Reindex: return "Reindex";
        case e::AttnRole::Reuse: return "Reuse";
    }
    return "?";
}

// The memory table (the header): the real 40 layers at kTargetCtx per chunk size, smallest first, until one does not
// fit.
std::vector<ChunkMemory> measure_memory(const std::vector<int64_t>& chunks, const e::ExpertConfig& ec, int64_t table_rows) {
    const int n_expert = std::max(6, std::min(ec.prefetch_experts, kRealExperts));
    auto t0 = std::chrono::steady_clock::now();
    const e::HostModel h = synthetic_model(kRealLayers, n_expert, table_rows);
    e::Model m;
    std::string err;
    size_t free0 = 0, free1 = 0, total = 0;
    if (!m.init(h, false, false, err)) die("the memory table's model: " + err);
    // the router of the experts this model lacks: ffn_gate_inp's bf16 row and exp_probs_b's f32 per expert and layer
    const double router_gib = (double) kRealLayers * (kRealExperts - n_expert) * (h.dense.n_embd * 2.0 + 4.0) / kGiB;
    const double model_gib = (double) m.device_bytes() / kGiB + router_gib;
    std::printf("dsv41_engine_smoke: memory: the real %d layers at ctx %lld (%d experts per layer, made in %.0f ms): dense "
                "model %.2f GiB (%.2f measured, %.2f for the router rows of the other %d experts)\n",
                kRealLayers, (long long) kTargetCtx, n_expert,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), model_gib,
                (double) m.device_bytes() / kGiB, router_gib, kRealExperts - n_expert);
    std::vector<ChunkMemory> rows;
    for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
        e::ExpertConfig none = ec;
        none.hot_set.clear();
        none.vram_gib = 0;
        e::ForwardConfig fc;
        fc.ctx = kTargetCtx;
        fc.chunk = *it;
        ck(cudaMemGetInfo(&free0, &total), "cudaMemGetInfo");
        e::Forward fw;
        if (!fw.init(m, none, fc, err)) {
            cudaGetLastError();
            std::printf("dsv41_engine_smoke: memory, chunk %lld: does not fit at ctx %lld (%s)\n", (long long) *it,
                        (long long) kTargetCtx, err.c_str());
            break;
        }
        ck(cudaMemGetInfo(&free1, &total), "cudaMemGetInfo");
        ChunkMemory r;
        r.chunk = *it;
        r.model_gib = model_gib;
        r.forward_gib = (double) (free0 - free1) / kGiB;
        r.ring_gib = (double) fw.experts().ring_bytes() / kGiB;
        std::printf("dsv41_engine_smoke: memory, chunk %lld: %.2f GiB beside the routed experts (dense model %.2f, KV cache, "
                    "workspaces, expert buffers, ring and prefetch region %.2f; of it the ring %.2f GiB, %d sets of %d "
                    "experts)\n",
                    (long long) r.chunk, r.total_gib(), r.model_gib, r.forward_gib, r.ring_gib, ec.ring_sets, ec.ring_set);
        fw.print_memory(stdout);
        rows.push_back(r);
    }
    return rows;
}

// --counts: the hybrid's serving profile (moe_counts.h, strata-moe-counts/1), its decode counts per
// layer and expert, hot_set.py's default --counts-calls
struct CountsProfile {
    std::string path;
    std::vector<std::vector<int64_t>> decode;   // [profile layer][expert]
};

CountsProfile load_counts(const std::string& path, int n_expert) {
    std::ifstream f(path);
    if (!f) die("--counts " + path + ": cannot read it");
    CountsProfile c;
    c.path = path;
    try {
        const nlohmann::json j = nlohmann::json::parse(f);
        if (j.at("format") != "strata-moe-counts/1") die("--counts " + path + ": not a strata-moe-counts/1 file");
        for (const nlohmann::json& l : j.at("layers")) {
            if (l.at("layer").get<int64_t>() != (int64_t) c.decode.size())
                die("--counts " + path + ": layer " + std::to_string(l.at("layer").get<int64_t>()) + " out of order");
            if (l.at("n_expert").get<int>() != n_expert)
                die("--counts " + path + ": " + std::to_string(l.at("n_expert").get<int>()) + " experts per layer, the "
                    "smoke has " + std::to_string(n_expert) + " (--experts)");
            c.decode.push_back(l.at("decode").get<std::vector<int64_t>>());
            if ((int) c.decode.back().size() != n_expert) die("--counts " + path + ": a decode list of the wrong length");
            int64_t sum = 0;
            for (int64_t v : c.decode.back()) {
                if (v < 0) die("--counts " + path + ": a negative count");
                sum += v;
            }
            if (sum == 0) die("--counts " + path + ": layer " + std::to_string(c.decode.size() - 1) + " has no decode counts");
        }
    } catch (const nlohmann::json::exception& ex) {
        die("--counts " + path + ": " + ex.what());
    }
    if (c.decode.empty()) die("--counts " + path + ": no layers");
    return c;
}

// smoke layer l stands for profile layer l * n / n_layer: the smoke's layers spread over the real ones (8 of 40:
// 0, 5, .., 35)
int profile_layer(const CountsProfile& c, int l, int n_layer) { return (int) ((int64_t) l * (int64_t) c.decode.size() / n_layer); }

// hot_set.py's pack: a global greedy over the smoke's layers by count, ties to the better rank in its own layer, then
// to the lower layer; n_hot experts in all. Per layer its experts in descending count (the slab's slot order).
std::vector<std::vector<int32_t>> counts_hot_set(const CountsProfile& c, int n_layer, int n_expert, int64_t n_hot) {
    struct Cand {
        int64_t neg_count;
        int rank, layer;
        int32_t expert;
        bool operator<(const Cand& o) const {
            if (neg_count != o.neg_count) return neg_count < o.neg_count;
            if (rank != o.rank) return rank < o.rank;
            if (layer != o.layer) return layer < o.layer;
            return expert < o.expert;
        }
    };
    std::vector<Cand> cands;
    for (int l = 0; l < n_layer; ++l) {
        const std::vector<int64_t>& n = c.decode[(size_t) profile_layer(c, l, n_layer)];
        std::vector<int32_t> order((size_t) n_expert);
        for (int32_t e = 0; e < n_expert; ++e) order[(size_t) e] = e;
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return n[(size_t) a] > n[(size_t) b]; });
        for (int r = 0; r < n_expert; ++r) cands.push_back({-n[(size_t) order[(size_t) r]], r, l, order[(size_t) r]});
    }
    std::sort(cands.begin(), cands.end());
    std::vector<std::vector<int32_t>> hot((size_t) n_layer);
    for (int64_t i = 0; i < std::min<int64_t>(n_hot, (int64_t) cands.size()); ++i)
        hot[(size_t) cands[(size_t) i].layer].push_back(cands[(size_t) i].expert);
    return hot;
}

// The routing --counts draws: per smoke layer the profile layer's decode counts plus one (an expert the profile never
// saw keeps a chance below every seen one, as the real model routes it a few times in a 16K prompt), each token's
// n_used distinct experts drawn by weight (repeats redrawn) from a generator seeded by layer and position, so a run
// and its decode steps route the same way every time.
struct CountsRouting {
    std::vector<std::vector<double>> cdf;   // [smoke layer][expert], ends at 1
    int n_used = 0;
    void operator()(int il, int64_t p0, int64_t T, std::vector<int32_t>& ids) const {
        const std::vector<double>& c = cdf[(size_t) il];
        ids.resize((size_t) (T * n_used));
        for (int64_t t = 0; t < T; ++t) {
            std::mt19937_64 rng(((uint64_t) il << 40) ^ (uint64_t) (p0 + t) ^ 0x9e3779b97f4a7c15ull);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            int32_t* row = ids.data() + t * n_used;
            for (int s = 0; s < n_used;) {
                const int32_t e = (int32_t) std::min<size_t>(std::upper_bound(c.begin(), c.end(), u(rng)) - c.begin(), c.size() - 1);
                if (std::find(row, row + s, e) == row + s) row[s++] = e;
            }
        }
    }
};

CountsRouting counts_routing(const CountsProfile& c, int n_layer, int n_used) {
    CountsRouting r;
    r.n_used = n_used;
    for (int l = 0; l < n_layer; ++l) {
        const std::vector<int64_t>& n = c.decode[(size_t) profile_layer(c, l, n_layer)];
        double sum = 0;
        for (int64_t v : n) sum += (double) v + 1.0;
        std::vector<double> cdf(n.size());
        double acc = 0;
        for (size_t e = 0; e < n.size(); ++e) cdf[e] = (acc += ((double) n[e] + 1.0) / sum);
        cdf.back() = 1.0;
        r.cdf.push_back(std::move(cdf));
    }
    return r;
}

// the hot set's share of each smoke layer's raw decode counts (hot_set.py's in-sample hit rate), printed; the mean
double print_counts_hot_set(const CountsProfile& c, const std::vector<std::vector<int32_t>>& hot, int n_layer) {
    double sum = 0;
    std::printf("dsv41_engine_smoke: --counts %s (%zu layers): smoke layer -> profile layer, VRAM experts, their share of "
                "its decode counts:", c.path.c_str(), c.decode.size());
    for (int l = 0; l < n_layer; ++l) {
        const std::vector<int64_t>& n = c.decode[(size_t) profile_layer(c, l, n_layer)];
        int64_t all = 0, hit = 0;
        for (int64_t v : n) all += v;
        for (int32_t e : hot[(size_t) l]) hit += n[(size_t) e];
        sum += (double) hit / (double) all;
        std::printf(" %d->%d %zu %.1f%%%s", l, profile_layer(c, l, n_layer), hot[(size_t) l].size(), 100.0 * hit / all,
                    l + 1 < n_layer ? "," : "\n");
    }
    std::printf("dsv41_engine_smoke: --counts hot set: expected decode hit rate %.1f%% (in-sample, hot_set.py's rule)\n",
                100.0 * sum / n_layer);
    return sum / n_layer;
}

// The host's MemAvailable in GiB (/proc/meminfo; -1 when it cannot be read), flushed at once: the first pod session
// lost its host during this run (host OOM suspected), so the log must keep the last figure
void print_host_ram(const char* when) {
    std::ifstream f("/proc/meminfo");
    std::string key;
    long long kb = 0;
    double gib = -1;
    while (f >> key >> kb) {
        if (key == "MemAvailable:") {
            gib = (double) kb / (1024.0 * 1024.0);
            break;
        }
        f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    std::printf("dsv41_engine_smoke: prefix cache: host RAM %s: MemAvailable %.1f GiB\n", when, gib);
    std::fflush(stdout);
}

// --prefix-cache N: the four engines of the header, one at a time (each pins its slab and ring, and its own
// pinned staging and buffers on the host); each is destroyed, its pinned memory freed, before the next is built
double rel_l2(const std::vector<float>& a, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += ((double) a[i] - ref[i]) * ((double) a[i] - ref[i]);
        den += (double) ref[i] * ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

// the floor's target (--floor-target NAME:GEN:WIDTH) and this pod's copy rates
struct FloorTarget {
    std::string name = "RTX PRO 6000";
    double link = 0;     // raw PCIe GB/s (link_gbps)
};

int run_prefix_cache(const e::Model& m, const e::ExpertConfig& ec, const CountsProfile& counts, int64_t chunk,
                     int64_t n_prompt, int64_t n_predict, int64_t n_new, const FloorTarget& target, double h2d,
                     double pod_link) {
    const e::DenseDims& dd = m.dense_dims();
    const int64_t P = n_prompt, G = n_predict, N = n_new;
    e::ForwardConfig base;
    base.ctx = P + G + N;
    base.chunk = chunk;
    if (!counts.decode.empty()) base.routing = counts_routing(counts, m.n_layer(), dd.n_expert_used);
    std::mt19937 rng(7);
    auto random_ids = [&](int64_t n) {
        std::vector<int32_t> v((size_t) n);
        for (int32_t& t : v) t = (int32_t) (rng() % (uint32_t) dd.n_vocab);
        return v;
    };
    const std::vector<int32_t> prompt = random_ids(P), new1 = random_ids(N);
    std::vector<int32_t> new2 = random_ids(N);
    std::string err;
    auto engine = [&](const e::ForwardConfig& fc) {
        auto fw = std::make_unique<e::Forward>();
        if (!fw->init(m, ec, fc, err)) die("prefix cache: " + err);
        return fw;
    };
    auto finite = [](const std::vector<float>& v, const char* what) {
        for (float x : v)
            if (!std::isfinite(x)) die(std::string("prefix cache: non-finite logits in ") + what);
    };
    auto expect = [](const e::SessionResult& r, e::ResumeKind kind, int64_t reuse, const char* what) {
        if (r.plan.kind != kind || r.plan.reuse != reuse)
            die(std::string("prefix cache: ") + what + ": " + e::resume_kind_name(r.plan.kind) + " resume of " +
                std::to_string(r.plan.reuse) + ", expected " + e::resume_kind_name(kind) + " of " + std::to_string(reuse));
    };
    std::printf("dsv41_engine_smoke: prefix cache: turn 1 = %lld prompt tokens + %lld generated, turn 2 = turn 1 + %lld new "
                "tokens (%lld), chunk %lld, %s routing\n", (long long) P, (long long) G, (long long) N,
                (long long) (P + G + N), (long long) chunk, counts.decode.empty() ? "the synthetic router's" : "--counts");

    // (A) free: turn 1, turn 2 (a live resume), then the same tokens cold
    std::vector<int32_t> gen, turn2;
    std::vector<float> warm_free, cold_free;
    double ttft_warm = 0, ttft_cold = 0, proj = 0, floor_ms = 0, floor_pod_ms = 0, rest_proj = 0;
    print_host_ram("before engine A");
    {
        auto fw = engine(base);
        e::ForwardBackend be(*fw, base, dd.n_vocab);
        e::Session s(be, {});   // no end of reply: the synthetic model's ids mean nothing
        e::SessionRequest t1;
        t1.ids = prompt;
        t1.max_new = G;
        const e::SessionResult r1 = s.run(t1, {});
        gen = r1.tokens;
        std::printf("dsv41_engine_smoke: prefix cache (A): turn 1 %s: %lld tokens read in %.1f ms (TTFT %.1f ms), %lld "
                    "generated in %.1f ms; %zu tokens held, snapshot at %lld\n",
                    e::resume_kind_name(r1.plan.kind), (long long) r1.read, r1.prompt_ms, r1.first_token_ms,
                    (long long) r1.generated, r1.decode_ms, s.live().size(), (long long) s.snapshot_tokens());
        turn2 = prompt;
        turn2.insert(turn2.end(), gen.begin(), gen.end());
        turn2.insert(turn2.end(), new1.begin(), new1.end());
        fw->set_timing(true);
        fw->reset_times();
        e::SessionRequest t2;
        t2.ids = turn2;
        t2.max_new = 1;
        e::SessionEvents ev;
        ev.prompt_logits = [&](const std::vector<float>& l) { warm_free = l; };
        const e::SessionResult r2 = s.run(t2, ev);
        expect(r2, e::ResumeKind::Live, P + G - 1, "(A) turn 2");
        finite(warm_free, "(A) turn 2");
        ttft_warm = r2.first_token_ms;
        const e::ForwardTimes t = fw->times();
        const e::ExpertTimes& x = fw->experts().times();
        const double fixed = t.embed_ms + t.head_ms;
        proj = (ttft_warm - fixed) * kRealLayers / m.n_layer() + fixed;
        std::printf("dsv41_engine_smoke: prefix cache (A): turn 2: RESUME %lld (%s) of %zu tokens, %lld read, TTFT %.1f ms "
                    "(device: embed %.2f, attention %.2f, hc and norms %.2f, router %.2f, shared %.2f, routed experts %.2f "
                    "with %.1f MiB of experts through the ring, head %.2f)\n",
                    (long long) r2.plan.reuse, e::resume_kind_name(r2.plan.kind), turn2.size(), (long long) r2.read,
                    ttft_warm, t.embed_ms, t.attention_ms, t.hc_ms, t.router_ms, t.shared_ms, t.experts_ms,
                    x.copy_bytes / (1024.0 * 1024.0), t.head_ms);
        // the split of its routed experts: the slab, the ring (its copies' device time), the CPU (host time)
        const e::PrefillSplitRates sr = fw->experts().split_rates();
        double ring_ms = 0;
        for (const e::ExpertLayerStats& st : fw->experts().layer_stats()) ring_ms += st.copy_ms;
        std::printf("dsv41_engine_smoke: prefix cache (A): turn 2's routed experts, the prefill split in %lld of %lld layer calls: "
                    "%lld pairs on the slab, %lld through the ring (%lld experts, %.1f MiB, copies %.1f ms on the copy "
                    "stream: %.2f GB/s), %lld on the CPU (%lld experts, %.1f MiB, %.1f ms on the host: %.2f GB/s); the "
                    "split's estimates summed over the layers: ring %.1f ms, CPU %.1f ms with its rows' upload (rates "
                    "measured at init: PCIe %.2f GB/s, CPU %.2f GB/s + %.4f ms per further pair)\n",
                    (long long) x.split_calls, (long long) x.calls, (long long) x.hits,
                    (long long) (x.misses - x.split_cpu_pairs), (long long) x.ring_experts,
                    x.copy_bytes / (1024.0 * 1024.0), ring_ms, ring_ms > 0 ? x.copy_bytes / (ring_ms * 1e6) : 0.0,
                    (long long) x.split_cpu_pairs, (long long) x.split_cpu_experts, x.split_cpu_bytes / (1024.0 * 1024.0),
                    x.split_cpu_ms, x.split_cpu_ms > 0 ? x.split_cpu_bytes / (x.split_cpu_ms * 1e6) : 0.0,
                    x.split_ring_est_ms, x.split_cpu_est_ms, sr.pcie_gbs, sr.cpu_gbs, sr.cpu_pair_ms);
        // the physical floor: the touched experts outside the slab over PCIe + CPU bandwidth, at 40 layers
        const double touched = x.touched_bytes * kRealLayers / m.n_layer();
        const double eff = pod_link > 0 ? h2d / pod_link : 1.0, pcie = target.link * eff;
        floor_ms = touched / ((pcie + sr.cpu_gbs) * 1e6);
        floor_pod_ms = touched / ((h2d + sr.cpu_gbs) * 1e6);
        rest_proj = (ttft_warm - t.experts_ms - fixed) * kRealLayers / m.n_layer() + fixed;
        std::printf("dsv41_engine_smoke: prefix cache (A): turn 2's physical floor: %lld experts outside the slab touched "
                    "over %d layers, %.2f GiB (x %d / %d: %.2f GiB at 40 layers); on %s: PCIe %.1f GB/s (its link %.1f "
                    "GB/s x this pod's copy efficiency %.2f%s) + CPU %.2f GB/s (this pod's pool) = %.1f ms; on this pod "
                    "(pinned H2D %.2f GB/s + the same CPU): %.1f ms\n",
                    (long long) x.touched_experts, m.n_layer(), x.touched_bytes / kGiB, kRealLayers, m.n_layer(),
                    touched / kGiB, target.name.c_str(), pcie, target.link, eff,
                    pod_link > 0 ? "" : ", no --pod-link: taken as 1", sr.cpu_gbs, floor_ms, h2d, floor_pod_ms);
        fw->reset_times();
        ev.prompt_logits = [&](const std::vector<float>& l) { cold_free = l; };
        const e::SessionResult rc = s.run(t2, ev);   // the same tokens: neither the live state nor the snapshot extends
        expect(rc, e::ResumeKind::Cold, 0, "(A) the cold read");
        finite(cold_free, "(A) the cold read");
        ttft_cold = rc.first_token_ms;
        std::printf("dsv41_engine_smoke: prefix cache (A): the same %zu tokens cold: TTFT %.1f ms (%lld chunks); free "
                    "logits of turn 2 against them: rel L2 %.3g (reported: free selections may flip)\n",
                    turn2.size(), ttft_cold, (long long) ((turn2.size() + chunk - 1) / chunk), rel_l2(warm_free, cold_free));
    }
    print_host_ram("after engine A, freed");
    std::printf("dsv41_engine_smoke: prefix cache: turn 2 TTFT %.1f ms (RESUME %lld of %zu tokens) against a cold prefill "
                "of the same tokens %.1f ms (%.1fx); 40-layer projection of turn 2's TTFT %.1f ms (goal <= 1000 ms: %s)\n",
                ttft_warm, (long long) (P + G - 1), turn2.size(), ttft_cold, ttft_cold / ttft_warm, proj,
                proj <= 1000.0 ? "met" : "missed");
    std::printf("dsv41_engine_smoke: prefix cache: turn 2's floor on %s: its experts' %.1f ms + the rest of turn 2 projected "
                "%.1f ms = %.1f ms (goal <= 1000 ms: %s at the floor)\n", target.name.c_str(), floor_ms, rest_proj,
                floor_ms + rest_proj, floor_ms + rest_proj <= 1000.0 ? "within reach" : "out of reach");

    // (B), (D): the cold references with their selections
    if (new2[0] == gen[0]) new2[0] = (new2[0] + 1) % dd.n_vocab;   // D must leave turn 1 right after its prompt
    std::vector<int32_t> other = prompt;
    other.insert(other.end(), new2.begin(), new2.end());
    const std::string dumps = "dsv41_engine_smoke.prefix";
    if (mkdir(dumps.c_str(), 0755) != 0 && errno != EEXIST) die("cannot make " + dumps);
    auto reference = [&](const std::string& name, const std::vector<int32_t>& ids, std::vector<float>& logits) {
        print_host_ram(("before engine " + name).c_str());
        {
            e::ForwardConfig fc = base;
            fc.dump_dir = dumps + "/" + name;
            fc.dump_values = false;
            auto fw = engine(fc);
            e::ForwardBackend be(*fw, fc, dd.n_vocab);
            e::Session s(be, {});
            e::SessionRequest r;
            r.ids = ids;
            r.max_new = 1;
            e::SessionEvents ev;
            ev.prompt_logits = [&](const std::vector<float>& l) { logits = l; };
            const e::SessionResult res = s.run(r, ev);
            finite(logits, name.c_str());
            std::ofstream mf(fc.dump_dir + "/engine-manifest.json");
            mf << "{\"tool\": \"dsv41_engine_smoke --prefix-cache\", \"prompt_token_ids\": [";
            for (size_t i = 0; i < ids.size(); ++i) mf << (i ? ", " : "") << ids[i];
            mf << "], \"generated_token_ids\": [" << res.tokens[0] << "], \"ubatches\": [";
            const auto& steps = fw->dump_steps();
            for (size_t u = 0; u < steps.size(); ++u)
                mf << (u ? ", " : "") << "{\"ubatch\": " << u << ", \"pos0\": " << steps[u].pos0 << ", \"n_tokens\": "
                   << steps[u].n_tokens << "}";
            mf << "]}\n";
            if (!mf) die("writing " + fc.dump_dir + "/engine-manifest.json failed");
            std::printf("dsv41_engine_smoke: prefix cache (%s): %zu tokens cold in %zu calls, selections in %s\n",
                        name.c_str(), ids.size(), steps.size(), fc.dump_dir.c_str());
        }
        print_host_ram(("after engine " + name + ", freed").c_str());
    };
    std::vector<float> ref_turn2, ref_other;
    reference("B", turn2, ref_turn2);
    reference("D", other, ref_other);

    // (C) forced: turn 1 feeding A's tokens, turn 2 against B; turn 1's prompt again, then D's tokens against D
    const e::SelectionDump dump_b = e::SelectionDump::open(dumps + "/B"), dump_d = e::SelectionDump::open(dumps + "/D");
    bool ok = false;
    print_host_ram("before engine C");
    {
        e::ForwardConfig fc = base;
        fc.routing = nullptr;   // the references' router ids instead
        fc.forced = &dump_b;
        auto fw = engine(fc);
        e::ForwardBackend be(*fw, fc, dd.n_vocab);
        e::Session s(be, {});
        e::SessionRequest t1;
        t1.ids = prompt;
        t1.max_new = G;
        t1.feed.assign(gen.begin(), gen.end() - 1);
        const e::SessionResult r1 = s.run(t1, {});
        int64_t same = 0;
        for (int64_t i = 0; i < G; ++i) same += r1.tokens[(size_t) i] == gen[(size_t) i];
        std::vector<float> warm_forced, restored_forced;
        e::SessionRequest t2;
        t2.ids = turn2;
        t2.max_new = 1;
        e::SessionEvents ev;
        ev.prompt_logits = [&](const std::vector<float>& l) { warm_forced = l; };
        const e::SessionResult r2 = s.run(t2, ev);
        expect(r2, e::ResumeKind::Live, P + G - 1, "(C) turn 2");
        fw->set_forced(&dump_d);
        e::SessionRequest t1b;
        t1b.ids = prompt;
        t1b.max_new = 2;
        t1b.feed = {gen[0]};
        const e::SessionResult r1b = s.run(t1b, {});
        expect(r1b, e::ResumeKind::Cold, 0, "(C) turn 1 again");
        e::SessionRequest t3;
        t3.ids = other;
        t3.max_new = 1;
        ev.prompt_logits = [&](const std::vector<float>& l) { restored_forced = l; };
        const e::SessionResult r3 = s.run(t3, ev);
        expect(r3, e::ResumeKind::PromptEnd, P, "(C) turn 3");
        finite(warm_forced, "(C) turn 2");
        finite(restored_forced, "(C) turn 3");
        const double l2_live = rel_l2(warm_forced, ref_turn2), l2_restore = rel_l2(restored_forced, ref_other);
        ok = l2_live <= 5e-2 && l2_restore <= 5e-2;
        std::printf("dsv41_engine_smoke: prefix cache (C) forced: turn 1 picked %lld of A's %lld tokens; turn 2 (RESUME "
                    "%lld, live) logits rel L2 %.3g against B's cold read, turn 3 (RESUME %lld, prompt-end restore after "
                    "%lld fed token) %.3g against D's (<= 5e-2: %s); restore TTFT %.1f ms\n",
                    (long long) same, (long long) G, (long long) r2.plan.reuse, l2_live, (long long) r3.plan.reuse,
                    (long long) (r1b.generated - 1), l2_restore, ok ? "PASS" : "FAIL", r3.first_token_ms);
    }
    print_host_ram("after engine C, freed");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int n_layer = 40, n_expert = 8, vram_experts = -1, threads = 0, ring_set = 48, ring_sets = 3, prefetch = 32;
    int pool_spin = 50;
    double vram_gb = -1;
    int64_t ctx = 16384, chunk = 4096, n_prompt = 0, n_predict = 32, table_rows = int64_t(1) << 22, prefix_new = 0;
    bool memory_table = false, dry = false;
    std::string counts_path;
    double pod_link = 0, pod_tflops = 0;
    std::vector<Target> targets;
    FloorTarget floor_target;
    floor_target.link = link_gbps(5, 16);
    bool floor_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto str = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        auto val = [&]() -> int64_t { return std::atoll(str().c_str()); };
        if (a == "--layers") n_layer = (int) val();
        else if (a == "--experts") n_expert = (int) val();
        else if (a == "--vram-experts") vram_experts = (int) val();
        else if (a == "--vram-experts-gb") vram_gb = std::atof(str().c_str());
        else if (a == "--ctx") ctx = val();
        else if (a == "--chunk") chunk = val();
        else if (a == "--prompt") n_prompt = val();
        else if (a == "--n-predict") n_predict = val();
        else if (a == "--table-rows") table_rows = val();
        else if (a == "--threads") threads = (int) val();
        else if (a == "--ring-set") ring_set = (int) val();
        else if (a == "--ring-sets") ring_sets = (int) val();
        else if (a == "--prefetch-experts") prefetch = (int) val();
        else if (a == "--pool-spin-us") pool_spin = (int) val();
        else if (a == "--memory-table") memory_table = true;
        else if (a == "--pod-tflops") pod_tflops = std::atof(str().c_str());
        else if (a == "--dry-run") dry = true;
        else if (a == "--counts") counts_path = str();
        else if (a == "--prefix-cache") prefix_new = val();
        else if (a == "--floor-target") {
            const std::vector<std::string> f = split(str(), ':');
            if (f.size() != 3) die("--floor-target NAME:GEN:WIDTH");
            floor_target.name = f[0];
            floor_target.link = link_gbps(std::atoi(f[1].c_str()), std::atoi(f[2].c_str()));
            floor_set = true;
        }
        else if (a == "--pod-link") {
            const std::vector<std::string> f = split(str(), ':');
            if (f.size() != 2) die("--pod-link GEN:WIDTH");
            pod_link = link_gbps(std::atoi(f[0].c_str()), std::atoi(f[1].c_str()));
        } else if (a == "--project") {
            const std::vector<std::string> f = split(str(), ':');
            if (f.size() != 5) die("--project NAME:GEN:WIDTH:VRAM_GIB:TFLOPS");
            Target t;
            t.name = f[0];
            t.link = link_gbps(std::atoi(f[1].c_str()), std::atoi(f[2].c_str()));
            t.vram_gib = std::atof(f[3].c_str());
            t.tflops = std::atof(f[4].c_str());
            if (!(t.vram_gib > 0) || !(t.tflops > 0)) die("--project " + f[0] + ": VRAM_GIB > 0, TFLOPS > 0");
            targets.push_back(t);
        } else {
            die("unknown argument " + a);
        }
    }
    if (n_layer < 1 || n_layer > 40) die("--layers N (1..40)");
    if (n_expert < 6 || n_expert > 384) die("--experts N (6..384)");
    if (chunk < 1 || chunk > 65535) die("--chunk N (1..65535)");
    if (!targets.empty() && (pod_link <= 0 || !memory_table))
        die("--project needs --pod-link (this pod's PCIe GEN:WIDTH) and --memory-table (the target's memory)");
    if (pod_tflops < 0) die("--pod-tflops T >= 0 (0: no published number for this pod's GPU)");
    if (n_prompt > 65535 || n_predict < 0) die("--prompt N (at most 65535), --n-predict N >= 0");
    if (prefix_new < 0 || (prefix_new > 0 && (n_predict < 2 || memory_table || !targets.empty())))
        die("--prefix-cache N: N >= 1 new tokens, --n-predict >= 2, not with --memory-table or --project");
    if (floor_set && prefix_new == 0) die("--floor-target is --prefix-cache's");

    // the slab: a whole number of experts per layer, the slot as experts.cpp packs it (each part 256-aligned)
    const e::DenseDims real = e::dense_dims_real();
    const size_t parts[3] = {qsynth::random_blocks_bytes(e::kQTypeQ2K, real.n_ff, real.n_embd),
                             qsynth::random_blocks_bytes(e::kQTypeQ2K, real.n_ff, real.n_embd),
                             qsynth::random_blocks_bytes(e::kQTypeQ3K, real.n_embd, real.n_ff)};
    double slot_bytes = 0;
    for (size_t b : parts) slot_bytes += (double) ((b + 255) / 256 * 256);
    if (vram_gb >= 0) vram_experts = (int) std::floor(vram_gb * kGiB / ((double) n_layer * slot_bytes));
    if (vram_experts < 0) vram_experts = n_expert / 2;
    vram_experts = std::min(vram_experts, n_expert);
    const double hot_fraction = (double) vram_experts / n_expert;
    const double all_experts_gib = (double) kRealLayers * kRealExperts * slot_bytes / kGiB;   // the target's full slab
    const double slab_gib = (double) vram_experts * n_layer * slot_bytes / kGiB;
    std::printf("experts: %d per layer of %.0f B, %d in VRAM (%.1f%%, %.2f GiB over %d layers), the rest cold in pinned "
                "RAM (%.2f GiB; %.2f GiB pinned for all of them)\n",
                n_expert, slot_bytes, vram_experts, 100.0 * hot_fraction, slab_gib, n_layer,
                (double) (n_expert - vram_experts) * n_layer * slot_bytes / kGiB, (double) n_expert * n_layer * slot_bytes / kGiB);
    // --counts: the same slab, its experts by the profile's counts, and every call routed by them
    CountsProfile counts;
    std::vector<std::vector<int32_t>> hot_experts((size_t) n_layer);
    if (!counts_path.empty()) {
        counts = load_counts(counts_path, n_expert);
        hot_experts = counts_hot_set(counts, n_layer, n_expert, (int64_t) vram_experts * n_layer);
        print_counts_hot_set(counts, hot_experts, n_layer);
    } else {
        for (std::vector<int32_t>& h : hot_experts)
            for (int32_t x = 0; x < vram_experts; ++x) h.push_back(x);
    }
    std::vector<int64_t> chunks;
    for (int64_t c = chunk; c >= std::min<int64_t>(chunk, 2048); c /= 2) chunks.push_back(c);
    if (dry && prefix_new > 0) {
        const int64_t P = n_prompt > 0 ? n_prompt : chunk;
        std::printf("dry run: prefix cache at chunk %lld: turn 1 = %lld prompt tokens + %lld generated, turn 2 = turn 1 + "
                    "%lld new (%lld tokens, ctx %lld): RESUME %lld (live) expected, its TTFT against the same tokens cold "
                    "and its 40-layer projection; references B (turn 2's tokens) and D (the prompt + %lld others) read "
                    "cold with their selections dumped; forced from them: turn 2 (live) and turn 3 (RESUME %lld, prompt-end "
                    "restore), rel L2 <= 5e-2 each\n",
                    (long long) chunk, (long long) P, (long long) n_predict, (long long) prefix_new,
                    (long long) (P + n_predict + prefix_new), (long long) (P + n_predict + prefix_new),
                    (long long) (P + n_predict - 1), (long long) prefix_new, (long long) P);
        char eff[96] = "copy efficiency 1 (no --pod-link)";
        if (pod_link > 0) std::snprintf(eff, sizeof eff, "this pod's copy efficiency, pinned H2D / its %.1f GB/s link", pod_link);
        std::printf("dry run: turn 2 (%lld tokens read) takes the prefill split up to %lld tokens: its pairs on the slab, "
                    "the ring and the CPU, against the split's estimates; its physical floor on %s: the touched experts "
                    "outside the slab x %d / %d over PCIe %.1f GB/s x %s + this pod's measured CPU GB/s\n",
                    (long long) (prefix_new + 1), (long long) e::ExpertConfig{}.split_max_t, floor_target.name.c_str(),
                    kRealLayers, n_layer, floor_target.link, eff);
        std::printf("dsv41_engine_smoke: dry run done, no CUDA call\n");
        return 0;
    }
    if (dry) {
        if (memory_table)
            std::printf("dry run: the memory table first: the real %d layers (%d experts per layer, the router of %d) at "
                        "ctx %lld, chunks %lld up to %lld (doubling)\n",
                        kRealLayers, std::max(6, std::min(prefetch, kRealExperts)), kRealExperts, (long long) kTargetCtx,
                        (long long) chunks.back(), (long long) chunks.front());
        std::printf("dry run: chunks tried from %lld down to %lld (halving), prompt %s, ctx at least %lld, %lld decode steps "
                    "x 3 (timed; token graph; token graph with the pool spinning %d us), then each layer's decode attention, "
                    "the head and each layer's hyper-connections alone in graphs (20 replays, then each step with its "
                    "phase marks), then E08's %d requests with swaps in flight (adaptive: copy only with a swap whenever "
                    "a layer can start one, the gate; copy only at the default margin; published with a swap whenever "
                    "a layer can start one), ring %d sets of %d (%.2f GiB of VRAM), prefetch %d experts\n",
                    (long long) chunk, (long long) std::min<int64_t>(chunk, 2048),
                    n_prompt > 0 ? std::to_string(n_prompt).c_str() : "the chunk",
                    (long long) std::max(ctx, std::max(n_prompt, chunk) + kDecodePhases * n_predict + kHotRequests * (1 + n_predict)),
                    (long long) n_predict, pool_spin, kHotRequests, ring_sets, ring_set,
                    (double) ring_sets * ring_set * slot_bytes / kGiB,
                    prefetch);
        if (pod_link > 0) {
            char tf[96] = "no published compute (the projection at compute scale 1 only)";
            if (pod_tflops > 0) std::snprintf(tf, sizeof tf, "its published dense FP16 tensor %.2f TFLOPS", pod_tflops);
            std::printf("dry run: this pod's link %.1f GB/s, %s\n", pod_link, tf);
        }
        for (const Target& t : targets) {
            std::printf("dry run: projection %s: link %.1f GB/s, %.1f GiB of VRAM: hot fraction (%.1f - the measured "
                        "non-expert GiB - %.0f) / %.2f GiB of experts; compute scales 1",
                        t.name.c_str(), t.link, t.vram_gib, t.vram_gib, kMarginGiB, all_experts_gib);
            if (pod_tflops > 0) std::printf(" and %.2f (%.2f / %.2f TFLOPS)", t.tflops / pod_tflops, t.tflops, pod_tflops);
            std::printf("%s\n", n_expert == kRealExperts ? "" : " (skipped on the pod: not the real 384 experts)");
        }
        std::printf("dsv41_engine_smoke: dry run done, no CUDA call\n");
        return 0;
    }

    e::ExpertConfig ec;
    ec.hot.policy = e::HotPolicy::Static;   // the swaps' phase sets its own policy
    ec.threads = threads;
    ec.ring_set = ring_set;
    ec.ring_sets = ring_sets;
    ec.prefetch_experts = prefetch;
    // the target's memory first, its model freed before this run's pins its experts
    std::vector<ChunkMemory> memory;
    if (memory_table) memory = measure_memory(chunks, ec, table_rows);

    auto t0 = std::chrono::steady_clock::now();
    const e::HostModel h = synthetic_model(n_layer, n_expert, table_rows);
    std::printf("model: %s, %zu tensors, made in %.0f ms\n", h.source.c_str(), h.tensors.size(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    e::Model m;
    std::string err;
    if (!m.init(h, false, false, err)) die(err);
    const e::DenseDims& dd = m.dense_dims();

    const double h2d = h2d_gbps();
    char link_txt[64] = "";
    if (pod_link > 0) std::snprintf(link_txt, sizeof link_txt, " on a %.1f GB/s link", pod_link);
    std::printf("dsv41_engine_smoke: pinned H2D %.2f GB/s (256 MiB copies)%s\n", h2d, link_txt);

    // the hot set: the first vram_experts experts of every layer in VRAM, or with --counts the same number by the
    // profile's counts
    const std::string hot = "dsv41_engine_smoke.hot.json";
    {
        std::ofstream f(hot);
        f << "{\"format\": \"strata-moe-hot-set/1\", \"budget_gib\": " << std::ceil(slab_gib + 1) << ", \"layers\": [";
        for (int l = 0; l < m.n_layer(); ++l) {
            f << (l ? ", " : "") << "{\"layer\": " << l << ", \"n_expert\": " << n_expert << ", \"n_embd\": "
              << dd.n_embd << ", \"n_ff\": " << dd.n_ff
              << ", \"types\": {\"up\": \"q2_K\", \"gate\": \"q2_K\", \"down\": \"q3_K\"}, \"experts\": [";
            for (size_t x = 0; x < hot_experts[(size_t) l].size(); ++x) f << (x ? ", " : "") << hot_experts[(size_t) l][x];
            f << "]}";
        }
        f << "]}\n";
        if (!f) die("cannot write " + hot);
    }
    ec.hot_set = hot;
    ec.vram_gib = std::ceil(slab_gib + 1);
    if (prefix_new > 0) {
        const int rc = run_prefix_cache(m, ec, counts, chunk, n_prompt > 0 ? n_prompt : chunk, n_predict, prefix_new,
                                        floor_target, h2d, pod_link);
        std::remove(hot.c_str());
        return rc;
    }
    // the prompt, the three decode phases, then the requests of one prompt token and n_predict steps each (the
    // profiles leave the position where it is). An earlier 4 n_predict + 2 left out the second request: the 16K
    // smokes died there past --ctx (exit 1)
    const int64_t ctx_need = std::max(n_prompt, chunk) + kDecodePhases * n_predict + kHotRequests * (1 + n_predict);
    ctx = std::max(ctx, ctx_need);

    // the largest chunk that fits beside the slab
    std::unique_ptr<e::Forward> fw;
    for (int64_t c : chunks) {
        e::ForwardConfig fc;
        fc.ctx = ctx;
        fc.chunk = c;
        fc.timing = true;
        if (!counts.decode.empty()) fc.routing = counts_routing(counts, n_layer, dd.n_expert_used);
        fw = std::make_unique<e::Forward>();
        if (fw->init(m, ec, fc, err)) {
            chunk = c;
            break;
        }
        cudaGetLastError();
        std::printf("dsv41_engine_smoke: chunk %lld does not fit beside the %.2f GiB slab (%s)\n", (long long) c, slab_gib,
                    err.c_str());
        fw.reset();
    }
    std::remove(hot.c_str());
    if (!fw) die("no chunk from " + std::to_string(chunks.front()) + " down to " + std::to_string(chunks.back()) + " fits");
    if (n_prompt <= 0) n_prompt = chunk;
    fw->print_memory(stdout);

    std::mt19937 rng(7);
    std::vector<int32_t> prompt((size_t) n_prompt);
    for (int32_t& t : prompt) t = (int32_t) (rng() % (uint32_t) dd.n_vocab);
    std::vector<float> logits;
    auto finite = [&](const char* what) {
        for (float v : logits)
            if (!std::isfinite(v)) die(std::string("non-finite logits after ") + what);
    };
    fw->prefill(prompt.data(), n_prompt, logits);
    finite("the prefill");
    const e::ForwardTimes pt = fw->times();
    print_times("prefill", pt, fw->experts().times());

    // per layer: the front and the expert phase (experts.hpp ExpertLayerStats)
    const std::vector<e::ForwardLayerTimes>& lt = fw->layer_times();
    const std::vector<e::ExpertLayerStats>& xs = fw->experts().layer_stats();
    double bytes = 0, copy_ms = 0, overlap_ms = 0, compute_ms = 0, stall_ms = 0, gap_ms = 0;
    double gather_ms = 0, gate_up_ms = 0, down_ms = 0;
    int64_t passes = 0;
    std::map<std::string, std::pair<double, int>> front_of_role, prefill_attn_of_role;
    for (int l = 0; l < m.n_layer(); ++l) {
        const e::ExpertLayerStats& s = xs[(size_t) l];
        const double front = lt[(size_t) l].attention_ms + lt[(size_t) l].dense_ms;
        const char* role = role_name(m.roles()[(size_t) l].role());
        front_of_role[role].first += front;
        ++front_of_role[role].second;
        prefill_attn_of_role[role].first += lt[(size_t) l].attention_ms;
        ++prefill_attn_of_role[role].second;
        std::printf("layer %2d %-7s: front %.2f ms (attention %.2f, dense %.2f); experts %.2f ms: %d passes, copies %.3f GB "
                    "in %.2f ms (%.2f GB/s), compute %.2f ms (gather %.2f, gate/up %.2f, down %.2f), overlap %.1f%%, "
                    "plan gap %.2f ms, stalls %.2f ms\n",
                    l, role, front, lt[(size_t) l].attention_ms, lt[(size_t) l].dense_ms, lt[(size_t) l].experts_ms,
                    (int) s.passes, s.copy_bytes / 1e9, s.copy_ms, s.copy_ms > 0 ? s.copy_bytes / (s.copy_ms * 1e6) : 0.0,
                    s.compute_ms, s.gather_ms, s.gate_up_ms, s.down_ms, s.copy_ms > 0 ? 100.0 * s.overlap_ms / s.copy_ms : 0.0,
                    s.plan_gap_ms, s.stall_ms);
        bytes += s.copy_bytes;
        copy_ms += s.copy_ms;
        overlap_ms += s.overlap_ms;
        compute_ms += s.compute_ms;
        gather_ms += s.gather_ms;
        gate_up_ms += s.gate_up_ms;
        down_ms += s.down_ms;
        passes += s.passes;
        stall_ms += s.stall_ms;
        gap_ms += s.plan_gap_ms;
    }
    const double ring_gbps = copy_ms > 0 ? bytes / (copy_ms * 1e6) : 0.0;
    const double overlap_pct = copy_ms > 0 ? 100.0 * overlap_ms / copy_ms : 0.0;
    std::printf("dsv41_engine_smoke: prefill chunk %lld at ctx %lld, %d layers x %d experts (%.1f%% in VRAM): %.1f ms; expert "
                "copies %.2f GB at %.2f GB/s = %.1f%% of the pinned H2D %.2f GB/s (gate >= 80%%: %s), overlap %.1f%% (gate "
                ">= 70%%: %s); expert compute %.1f ms, idle gaps: plan %.1f ms, stalls %.1f ms\n",
                (long long) n_prompt, (long long) ctx, n_layer, n_expert, 100.0 * hot_fraction, pt.total_ms, bytes / 1e9,
                ring_gbps, 100.0 * ring_gbps / h2d, h2d, ring_gbps >= 0.8 * h2d ? "PASS" : "FAIL", overlap_pct,
                overlap_pct >= 70.0 ? "PASS" : "FAIL", compute_ms, gap_ms, stall_ms);
    // the routed experts' compute against the pod's dense FP16 tensor peak (6 n_embd n_ff FLOP per pair: gate,
    // up and down), by launch, and how wide the passes' launches are (a tile is one block per 64 gate/up features and
    // per 128 down rows)
    {
        const double pairs = (double) n_prompt * dd.n_expert_used, ef = (double) dd.n_embd * dd.n_ff;
        const double flop = pairs * m.n_layer() * 6.0 * ef;
        const double tflops = compute_ms > 0 ? flop / (compute_ms * 1e9) : 0.0;
        auto rate = [&](double f, double ms) { return ms > 0 ? pairs * m.n_layer() * f * ef / (ms * 1e9) : 0.0; };
        const double tiles = (double) fw->experts().times().tiles, per_pass = passes > 0 ? tiles / (double) passes : 0.0;
        int sms = 0;
        ck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "cudaDeviceGetAttribute");
        std::printf("dsv41_engine_smoke: routed-expert compute (%s routing) %.1f ms per layer for %.0f pairs (%.2f TFLOP): "
                    "gather %.1f, gate/up %.1f (%.1f TFLOP/s), down %.1f (%.1f TFLOP/s); %.0f grouped GEMM tiles per layer "
                    "in %.1f passes, %.1f tiles per pass = %.0f gate/up and %.0f down blocks per launch on %d SMs: "
                    "%.1f TFLOP/s",
                    counts.decode.empty() ? "the synthetic router's" : "--counts", compute_ms / m.n_layer(), pairs,
                    flop / m.n_layer() / 1e12, gather_ms / m.n_layer(), gate_up_ms / m.n_layer(), rate(4.0, gate_up_ms),
                    down_ms / m.n_layer(), rate(2.0, down_ms), tiles / m.n_layer(), (double) passes / m.n_layer(), per_pass,
                    per_pass * dd.n_ff / e::kExpertGateUpCols, per_pass * dd.n_embd / e::kExpertDownCols, sms, tflops);
        if (pod_tflops > 0)
            std::printf(" = %.1f%% of this pod's published dense FP16 tensor %.1f TFLOPS (gate >= 35%%: %s)\n",
                        100.0 * tflops / pod_tflops, pod_tflops, tflops >= 0.35 * pod_tflops ? "PASS" : "FAIL");
        else
            std::printf(" (no published peak for this pod's GPU: the gate is not computed)\n");
    }

    // the three dev-pod targets of this prefill, on the real 40 layers: the attention per
    // layer by role (a role not measured: Full's for Reindex, else the mean), the expert compute and the idle gaps
    // (plan gaps + stalls) scaled by 40 / --layers
    {
        double mean = 0;
        for (const auto& [r, v] : prefill_attn_of_role) mean += v.first;
        mean /= m.n_layer();
        double real = 0;
        for (const e::AttnLayer& l : e::attn_layers_real()) {
            auto it = prefill_attn_of_role.find(role_name(l.role()));
            if (it == prefill_attn_of_role.end() && l.role() == e::AttnRole::Reindex) it = prefill_attn_of_role.find("Full");
            real += it == prefill_attn_of_role.end() ? mean : it->second.first / it->second.second;
        }
        const double scale = (double) kRealLayers / m.n_layer(), x40 = compute_ms * scale, g40 = (gap_ms + stall_ms) * scale;
        const double attn = real / kRealLayers;
        std::printf("dsv41_engine_smoke: E09 targets (a %lld-token prefill, %d experts per layer, %s routing; the real 40 "
                    "layers): attention %.2f ms per layer (<= 30 at 16K: %s), routed-expert compute %.0f ms (<= 2400: %s), "
                    "plan gaps + stalls %.0f ms (gaps %.0f, stalls %.0f; <= 600: %s)\n",
                    (long long) n_prompt, n_expert, counts.decode.empty() ? "the synthetic router's" : "--counts", attn,
                    attn <= 30.0 ? "PASS" : "FAIL", x40, x40 <= 2400.0 ? "PASS" : "FAIL", g40, gap_ms * scale,
                    stall_ms * scale, g40 <= 600.0 ? "PASS" : "FAIL");
    }

    // the 40-layer projection (the header's formula)
    if (!targets.empty()) {
        if (n_expert != kRealExperts) {
            std::printf("dsv41_engine_smoke: projection skipped: %d experts per layer, the formula needs the real %d\n",
                        n_expert, kRealExperts);
        } else {
            double mean_front = 0;
            for (const auto& [r, v] : front_of_role) mean_front += v.first;
            mean_front /= m.n_layer();
            auto front_of = [&](e::AttnRole r) {
                auto it = front_of_role.find(role_name(r));
                if (it == front_of_role.end() && r == e::AttnRole::Reindex) it = front_of_role.find("Full");
                return it == front_of_role.end() ? mean_front : it->second.first / it->second.second;
            };
            const double x_layer = compute_ms / m.n_layer(), eff = ring_gbps / pod_link;
            const double fixed = pt.embed_ms + pt.head_ms, n_chunks = 16384.0 / (double) n_prompt;
            // the target's hot fraction at a chunk size: what its VRAM holds of the full slab beside the measured rest
            auto hot_of = [&](const Target& t, const ChunkMemory& r) {
                return std::min(1.0, std::max(0.0, (t.vram_gib - r.total_gib() - kMarginGiB) / all_experts_gib));
            };
            for (const Target& t : targets) {
                const double bw = t.link * eff, p_ms = prefetch * kExpertBytes / (bw * 1e6);
                for (const ChunkMemory& r : memory) {
                    const double hot = hot_of(t, r), c_ms = (1.0 - hot) * kRealExperts * kExpertBytes / (bw * 1e6);
                    std::printf("dsv41_engine_smoke: projection %s, chunk %lld: %.2f GiB beside the experts + %.0f GiB margin "
                                "of %.1f GiB: %.2f GiB of experts, %.1f%% hot; a 16K prompt's copies %.1f GB in %.2f s at "
                                "%.1f GB/s (the copy bound)\n",
                                t.name.c_str(), (long long) r.chunk, r.total_gib(), kMarginGiB, t.vram_gib,
                                hot * all_experts_gib, 100.0 * hot,
                                16384.0 / r.chunk * kRealLayers * (1.0 - hot) * kRealExperts * kExpertBytes / 1e9,
                                16384.0 / r.chunk * kRealLayers * c_ms / 1e3, bw);
                }
                const auto r = std::find_if(memory.begin(), memory.end(), [&](const ChunkMemory& x) { return x.chunk == chunk; });
                if (r == memory.end()) {
                    std::printf("dsv41_engine_smoke: projection %s skipped: the memory table has no chunk %lld at ctx %lld\n",
                                t.name.c_str(), (long long) chunk, (long long) kTargetCtx);
                    continue;
                }
                const double hot = hot_of(t, *r), c_ms = (1.0 - hot) * kRealExperts * kExpertBytes / (bw * 1e6);
                std::vector<double> scales{1.0};
                if (pod_tflops > 0) scales.push_back(t.tflops / pod_tflops);
                for (size_t si = 0; si < scales.size(); ++si) {
                    const double scale = scales[si];
                    double sum = 0, sum_front = 0;
                    for (const e::AttnLayer& l : e::attn_layers_real()) {
                        const double f = front_of(l.role()) / scale, x = x_layer / scale;
                        sum += f + std::max(x, c_ms - std::min(f, p_ms));
                        sum_front += f;
                    }
                    const double per_chunk = sum + fixed / scale;
                    std::printf("dsv41_engine_smoke: projection %s: link %.1f GB/s x copy efficiency %.2f = %.1f GB/s, %.1f%% "
                                "hot at chunk %lld: %.2f GB of copies per layer per chunk (%.1f ms), front %.1f ms and "
                                "expert compute %.1f ms per 40 layers (compute x 1/%.2f%s), %.1f ms per 40 layers per "
                                "%lld-token chunk; 16K TTFT %.2f s in %.0f chunk(s)\n",
                                t.name.c_str(), t.link, eff, bw, 100.0 * hot, (long long) chunk,
                                (1.0 - hot) * kRealExperts * kExpertBytes / 1e9, c_ms, sum_front, kRealLayers * x_layer / scale,
                                scale, si == 0 ? ": this pod's compute" : ": the published dense FP16 tensor ratio",
                                per_chunk, (long long) n_prompt, n_chunks * per_chunk / 1e3, n_chunks);
                }
            }
        }
    }

    // decode: timed (per stage), then untimed from the token graph without and with the pool spinning
    auto argmax = [&] {
        int32_t best = 0;
        for (size_t i = 1; i < logits.size(); ++i)
            if (logits[i] > logits[(size_t) best]) best = (int32_t) i;
        return best;
    };
    struct Phase {
        const char* name;
        bool timing;
        int spin;
    };
    const Phase phases[] = {{"decode (timed, every op launched)", true, 0}, {"decode (token graph)", false, 0},
                            {"decode (token graph, pool spin)", false, pool_spin}};
    static_assert(sizeof phases / sizeof phases[0] == kDecodePhases);
    double head_ms = 0;   // per token, from the timed phase (the projection's fixed part)
    double hc_layer_ms = 0, tok_s = 0;   // the timed phase's hc per layer, the last phase's projected tok/s
    double static_wall_ms = 0, static_hit = 0;   // the last phase's wall ms per token and hit rate
    e::ForwardTimes static_ft;                   // the last phase's times (where its token's time goes)
    e::ExpertTimes static_xt;
    for (const Phase& ph : phases) {
        fw->set_timing(ph.timing);
        fw->set_pool_spin_us(ph.spin);
        fw->reset_times();
        const auto w0 = std::chrono::steady_clock::now();
        for (int64_t s = 0; s < n_predict; ++s) {
            fw->decode(argmax(), logits);
            finite("a decode step");
        }
        static_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() /
                         (double) std::max<int64_t>(1, n_predict);
        const e::ForwardTimes t = fw->times();
        const e::CpuPoolTimes pool = fw->experts().cpu_pool_times();
        print_times(ph.name, t, fw->experts().times());
        const double n = (double) std::max<int64_t>(1, t.calls);
        const e::ExpertTimes& x = fw->experts().times();
        std::printf("dsv41_engine_smoke: %s: %.2f ms per token, %.2f host syncs per token (%.2f per layer), %.1f graph "
                    "launches per token, %.1f doorbells per token (host wait %.3f ms); CPU pool (spin %d us): %.1f handoffs "
                    "per token, wake %.3f ms and return %.3f ms per token\n",
                    ph.name, t.total_ms / n, (double) t.host_syncs / n, (double) t.host_syncs / n / m.n_layer(),
                    (double) t.graph_launches / n, (double) x.doorbells / n, x.ring_wait_ms / n, ph.spin,
                    (double) pool.phases / n, pool.wake_ms / n, pool.return_ms / n);
        if (ph.timing) {
            // the split per layer per token (device times of each stage, every op launched)
            head_ms = t.head_ms / n;
            const double ln = n * m.n_layer();
            std::printf("dsv41_engine_smoke: %s, per layer per token: attention %.3f ms, hc and norms %.3f, router %.3f, "
                        "shared expert %.3f, routed experts %.3f (the hits, the CPU's misses and the wait for them), engram "
                        "%.3f; per token: embed %.3f, head %.3f ms\n",
                        ph.name, t.attention_ms / ln, t.hc_ms / ln, t.router_ms / ln, t.shared_ms / ln, t.experts_ms / ln,
                        t.engram_ms / ln, t.embed_ms / n, head_ms);
            hc_layer_ms = t.hc_ms / ln;
        }
        // the split and the CPU kernel; the 40 layers at this pod's ms per layer, the head once
        const double proj = (t.total_ms / n - head_ms) * kRealLayers / m.n_layer() + head_ms;
        const int64_t pairs = x.hits + x.misses;
        const int cpu_threads = fw->experts().cpu_threads();
        std::printf("dsv41_engine_smoke: %s, %s routing: hit rate %.1f%% (%lld GPU hits, %lld CPU misses), %.2f CPU "
                    "experts per layer per token, CPU %.3f ms per layer per token reading %.2f GB/s (%.2f GB/s per thread "
                    "of %d); 40-layer projection %.1f ms per token (%.1f tok/s; goal >= 30: %s)\n",
                    ph.name, counts.decode.empty() ? "the synthetic router's" : "--counts",
                    pairs ? 100.0 * (double) x.hits / (double) pairs : 0.0, (long long) x.hits, (long long) x.misses,
                    (double) x.cpu_experts / n / m.n_layer(), x.cpu_ms / n / m.n_layer(),
                    x.cpu_ms > 0 ? x.cpu_bytes / (x.cpu_ms * 1e6) : 0.0,
                    x.cpu_ms > 0 ? x.cpu_bytes / (x.cpu_ms * 1e6) / cpu_threads : 0.0, cpu_threads, proj, 1e3 / proj,
                    1e3 / proj >= 30.0 ? "PASS" : "FAIL");
        tok_s = 1e3 / proj;
        static_hit = pairs ? (double) x.hits / (double) pairs : 0.0;
        static_ft = t;
        static_xt = x;
    }
    print_where(phases[kDecodePhases - 1].name, static_wall_ms, static_ft, static_xt,
                (double) std::max<int64_t>(1, n_predict));

    // each layer's decode attention alone in a graph (its device time and the split by fused launch),
    // the 40 real layers' mean by role (a role not measured: Full's for Reindex, else the measured mean), the head's
    // bandwidth, then the cycle's targets
    const int prof_reps = 20;
    const e::ForwardAttnProfile ap = fw->profile_decode_attention(prof_reps);
    std::map<std::string, std::pair<double, int>> attn_of_role;
    double attn_mean = 0;
    for (int l = 0; l < m.n_layer(); ++l) {
        const char* role = role_name(m.roles()[(size_t) l].role());
        attn_of_role[role].first += ap.layer_ms[(size_t) l];
        ++attn_of_role[role].second;
        attn_mean += ap.layer_ms[(size_t) l] / m.n_layer();
        std::printf("dsv41_engine_smoke: decode attention (graph, position %lld), layer %d %s: %.3f ms", (long long) fw->position(),
                    l, role, ap.layer_ms[(size_t) l]);
        if (ap.split) {
            const char* sep = ":";
            for (int k = 0; k < e::kAttnKernels; ++k) {
                const double v = ap.kernel_ms[(size_t) l][(size_t) k];
                if (v <= 0) continue;
                std::printf("%s %s %.4f", sep, e::attn_kernel_name((e::AttnKernel) k), v);
                sep = ",";
            }
        }
        std::printf("\n");
    }
    if (!ap.split) std::printf("dsv41_engine_smoke: decode attention: the graphs' event nodes could not be read (no per-kernel split)\n");
    double attn40 = 0;
    for (const e::AttnLayer& l : e::attn_layers_real()) {
        auto it = attn_of_role.find(role_name(l.role()));
        if (it == attn_of_role.end() && l.role() == e::AttnRole::Reindex) it = attn_of_role.find("Full");
        attn40 += (it == attn_of_role.end() ? attn_mean : it->second.first / it->second.second) / kRealLayers;
    }
    std::printf("dsv41_engine_smoke: decode attention (graph, %d replays per layer): mean %.3f ms per layer over these %d, by role",
                prof_reps, attn_mean, m.n_layer());
    for (const auto& [r, v] : attn_of_role) std::printf(" %s %.3f", r.c_str(), v.first / v.second);
    std::printf("; the real 40 layers' mean by role %.3f ms\n", attn40);
    const double head_graph_ms = fw->profile_head(prof_reps), head_gb = (double) e::qweight_bytes(m.output()) / 1e9;
    const double copy_gbps = d2d_gbps(), peak_gbps = hbm_peak_gbps(), head_gbps = head_gb / (head_graph_ms * 1e-3);
    const double hbm = peak_gbps > 0 ? peak_gbps : copy_gbps;
    std::printf("dsv41_engine_smoke: head (graph, %d replays): %.3f ms for %.3f GB of q6_K = %.0f GB/s: %.1f%% of the "
                "device-to-device copy %.0f GB/s, %s\n",
                prof_reps, head_graph_ms, head_gb, head_gbps, 100.0 * head_gbps / copy_gbps, copy_gbps,
                peak_gbps > 0 ? (std::to_string((int) std::lround(100.0 * head_gbps / peak_gbps)) + "% of the published " +
                                 std::to_string((int) std::lround(peak_gbps)) + " GB/s").c_str()
                              : "no published bandwidth");
    std::printf("dsv41_engine_smoke: E05d targets: attention %.3f ms per layer (the real 40 layers' mean, graph; <= 0.15: "
                "%s), hc and norms %.3f ms per layer (timed; <= 0.06: %s), head %.1f%% of the %s HBM bandwidth (>= 70%%: "
                "%s), decode %.1f tok/s (40-layer projection, %s; >= 30: %s)\n",
                attn40, attn40 <= 0.15 ? "PASS" : "FAIL", hc_layer_ms, hc_layer_ms <= 0.06 ? "PASS" : "FAIL",
                100.0 * head_gbps / hbm, peak_gbps > 0 ? "published" : "copy", head_gbps >= 0.7 * hbm ? "PASS" : "FAIL",
                tok_s, phases[2].name, tok_s >= 30.0 ? "PASS" : "FAIL");
    // the hyper-connections and norms per layer from graphs, the steps' phase split, then the swaps in flight
    const e::ForwardHcProfile hcp = fw->profile_decode_hc(prof_reps);
    double hc_graph_ms = 0;
    std::printf("dsv41_engine_smoke: hc and norms (graph, %d replays per layer: the attention's and the FFN's "
                "hc_decode_step), ms per layer:", prof_reps);
    for (size_t l = 0; l < hcp.layer_ms.size(); ++l) {
        std::printf("%s %.4f", l ? "," : "", hcp.layer_ms[l]);
        hc_graph_ms += hcp.layer_ms[l] / (double) hcp.layer_ms.size();
    }
    std::printf("; mean %.4f\n", hc_graph_ms);
    const e::HcStepSplit& hs = hcp.split;
    std::printf("dsv41_engine_smoke: hc_decode_step phases (%lld marked launches, each step alone in a graph; "
                "%%globaltimer, us from the first block's start, the mean): row blocks' stream and rms %.2f, q8_1 and "
                "hc_fn rows staged %.2f, mixes written %.2f, ticket %.2f; the norm block's collapse and norm %.2f; the "
                "last block's stream and coefficients (the kernel's end) %.2f; against %.2f us per step from the "
                "layers' graphs (the rest: the launches)\n",
                (long long) hcp.marked, hs.rms_us, hs.q8_us, hs.mixes_us, hs.ticket_us, hs.norm_us, hs.end_us,
                hc_graph_ms * 1e3 / 2.0);
    // the requests with swaps, each from the static slab:
    // copy only with a swap whenever a layer can start one (margin 0:-1; the swaps' copies, up and down, as the
    // adaptive request would start them, none published: the slab and the hit rate stay the static phase's), the
    // gate's run; copy only at the default margin, printed for reference; then the adaptive request itself with a
    // swap whenever a layer can start one, its ratio printed only (the swaps move the slab, so its hit rate moves too).
    // Each prints where its token's time goes (ForwardTimes) beside the static phase's.
    struct HotRun {
        double ratio = 0, batch_ms = 0;
        int64_t done = 0, deferred = 0, boundaries = 0;
    };
    HotRun gate;
    if (fw->experts().hot_active()) {
        std::vector<std::vector<int32_t>> slab0((size_t) m.n_layer());
        for (int l = 0; l < m.n_layer(); ++l) slab0[(size_t) l] = fw->experts().tiers(l).vram;
        const e::HotConfig hot0 = fw->experts().hot();
        auto run = [&](bool copy_only, bool stress) {
            e::HotConfig hcfg = hot0;
            hcfg.policy = e::HotPolicy::Adaptive;
            if (stress) {
                hcfg.margin_mul = 0;
                hcfg.margin_add = -1;
            }
            hcfg.copy_only = copy_only;
            fw->set_hot(hcfg);
            fw->set_timing(false);
            fw->set_pool_spin_us(pool_spin);
            fw->begin_request(e::HotPolicy::Adaptive, true, false, false);
            const int32_t first = argmax();
            fw->prefill(&first, 1, logits);
            fw->reset_times();
            const auto w0 = std::chrono::steady_clock::now();
            for (int64_t s = 0; s < n_predict; ++s) {
                fw->decode(argmax(), logits);
                finite(copy_only ? "a decode step with copy-only swaps" : "a decode step with swaps");
            }
            const double n = (double) std::max<int64_t>(1, n_predict);
            const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() / n;
            const e::ForwardTimes ft = fw->times();
            const e::ExpertTimes x = fw->experts().times();   // before begin_request publishes or drops the open batch
            fw->begin_request(e::HotPolicy::Adaptive, false, false, false);
            const int64_t pairs = x.hits + x.misses, swaps = copy_only ? x.swaps_dropped : x.swaps;
            HotRun r;
            r.ratio = static_wall_ms > 0 ? wall / static_wall_ms : 0.0;
            r.done = x.swap_batches_done;
            r.batch_ms = r.done ? x.swap_copy_ms / (double) r.done : 0.0;
            r.deferred = x.swap_deferred;
            r.boundaries = n_predict;
            char what[160];
            std::snprintf(what, sizeof what, "decode with swaps in flight (adaptive, margin %g:%g, %d spare slots and at "
                          "most %d swaps per layer per token%s)", hcfg.margin_mul, hcfg.margin_add, hcfg.spare,
                          hcfg.max_swaps, copy_only ? "; copy only" : "");
            std::printf("dsv41_engine_smoke: %s: %.2f ms per token (wall), hit rate %.1f%%, %.2f swaps per token %s "
                        "(%.1f MiB per token, %.2f GB/s on the copy streams), %lld batches started, %lld done (%.2f ms "
                        "each on the device), %lld of %lld boundaries found the batch copying; the static pool-spin "
                        "phase %.2f ms per token (wall), hit rate %.1f%%: %.3fx\n",
                        what, wall, pairs ? 100.0 * (double) x.hits / (double) pairs : 0.0, (double) swaps / n,
                        copy_only ? "copied and dropped" : "published", x.swap_bytes / n / (1024.0 * 1024.0),
                        x.swap_copy_ms > 0 ? x.swap_bytes / (x.swap_copy_ms * 1e6) : 0.0, (long long) x.swap_batches,
                        (long long) r.done, r.batch_ms, (long long) r.deferred, (long long) r.boundaries, static_wall_ms,
                        100.0 * static_hit, r.ratio);
            print_where(what, wall, ft, x, n);
            if (copy_only) {
                if (x.swaps != 0) die("the copy-only request published " + std::to_string(x.swaps) + " swaps");
                if (stress && x.swap_batches == 0) die("the copy-only request started no batch of copies");
                for (int l = 0; l < m.n_layer(); ++l)
                    if (fw->experts().tiers(l).vram != slab0[(size_t) l])
                        die("the copy-only request moved layer " + std::to_string(l) + "'s slab");
            }
            return r;
        };
        gate = run(true, true);
        run(true, false);
        run(false, true);
        fw->set_hot(hot0);
    } else {
        std::printf("dsv41_engine_smoke: decode with swaps in flight: skipped, no layer can swap (--hot-spare 0 or every "
                    "expert in the slab)\n");
    }
    // the swaps' gate: copy only within 1.1x the static token, and a batch longer than a token deferred at the
    // boundaries that find it copying (not waited for: a longer token)
    const bool measured = gate.ratio > 0;
    const bool defers = gate.batch_ms <= static_wall_ms || gate.deferred > 0;
    std::printf("dsv41_engine_smoke: E08 targets: hc and norms %.4f ms per layer (graph; <= 0.10: %s), a token with the "
                "swaps' copies in flight (copy only, margin 0:-1) %.3fx one without (<= 1.1: %s; batches of %.2f ms on "
                "the device against a %.2f ms token, %lld of %lld boundaries deferred: %s)\n",
                hc_graph_ms, hc_graph_ms <= 0.10 ? "PASS" : "FAIL", gate.ratio,
                !measured ? "not measured" : gate.ratio <= 1.1 ? "PASS" : "FAIL", gate.batch_ms, static_wall_ms,
                (long long) gate.deferred, (long long) gate.boundaries,
                !measured ? "not measured" : defers ? "PASS" : "FAIL (a batch longer than a token was waited for)");
    const int64_t steps = kDecodePhases + (fw->experts().hot_active() ? kHotRequests : 0);
    std::printf("dsv41_engine_smoke: %lld prompt tokens at ctx %lld, %lld decode steps x %lld, every logit finite: done "
                "(the targets above are reported, not exit codes)\n",
                (long long) n_prompt, (long long) ctx, (long long) n_predict, (long long) steps);
    return 0;
}
