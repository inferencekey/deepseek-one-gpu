// src/dsv41/engine/experts.cpp - the routed expert tiers and calls of include/strata/dsv41/engine/experts.hpp.
//
// Strata first: the hot set is strata-moe's file and loader (moe_hot_set.h), the
// warm tier strata-moe's Residency (moe_residency.h: its pinned buffer, its O_DIRECT fill of the warm lists, bound
// through ExpertTensors), the prefill passes strata-moe's plan and schedule (moe_prefill.h plan_prefill /
// run_prefill), the misses its fast CPU kernel (experts_cpu.cpp), the counts its ActivationCounts (moe_counts.h). What
// is new is the GPU side: the slab and the ring are plain device buffers whose experts qmatmul reads in the
// GGUF's own blocks, one expert at a time over its gathered rows, instead of ggml-cuda's mul_mat_id graph
// (no ggml in the engine path).
//
// the PrefillEngine is moe_gpu.cpp's GpuTier::Prefill on plain CUDA: its copy stream, its copied and consumed
// events per ring set and stage_free per staging set, its pinned staging laid out by stage_layout and filled in
// kStageChunk pieces (by this engine's CPU pool instead of ggml-cpu's threads). New here: the prefetch region (the
// next layer's first non-resident experts, read by its passes like slab experts), a call that returns without
// waiting for its work, and the per-layer timing of the copies and passes.
//
// a prefill pass is two grouped launches over all of its experts (experts_gemm.hpp, on tensor cores, the q2_K /
// q3_K blocks read where the slab, the ring or the prefetch region holds them) instead of three qmatmuls per expert,
// and the slab's pairs are regrouped into whole experts before the passes run (regroup_slab).
//
// decode without a host round trip on the device's side, Strata's doorbell protocol (core/layer.hpp Doorbell,
// layer.cpp doorbell_init's mapped pinned regions, session.cpp session_run_token's host loop: poll the ring, run the
// pool, write the flag; bounded by a timeout and by stream queries that notice a fault) and its token graph's VRAM
// tier (session.hpp TokenHits: a static residency row per layer on the device, the hits built and run there). The
// decode had measured 0.84 ms per layer for its routed experts on the A100 although its hits read ~68 MB (~36 us):
// one host sync per layer for the routing, then up to 6 experts x 4 launches queued from the host.
//
// the prefill split (experts.hpp), Strata's hit/miss split carried to short prefill chunks: the CPU side
// is decode's (CpuExperts on the warm tier or the mapping), the GPU side the prefill passes over the pairs left to it,
// the rates measured once in init (measure_split_rates).
//
// the per-request hot set (experts.hpp, hot_policy.hpp). Strata's adaptive swaps (src/program/generate.cpp
// ~6660-6900) move experts between its VRAM cache and an arena that holds every expert
// in RAM; here the slab's victims have no RAM copy of their own (the warm tier holds the others), so a swap is
// make-before-break: the promoted expert lands in a spare slot and the victim's bytes in a spare home before either
// is published, and the residency rows the token graph reads (session.hpp TokenHits' row, a device buffer the graph
// takes by pointer) change only between tokens. Host homes (--hot-homes all): the slab's experts get RAM homes of
// their own as Strata's arena gives every expert, so the victims need no copy down; the deterministic reset then
// restores in place, layer by layer (experts.hpp, hot_policy.hpp HotRestore).
#include "strata/dsv41/engine/experts.hpp"

#include "strata/dsv41/engine/experts_gemm.hpp"
#include "ffn_kernels.hpp"
#include "moe_counts.h"
#include "moe_hot_set.h"
#include "moe_prefill.h"
#include "moe_residency.h"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 experts: %s\n", msg);
    std::exit(1);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) failf("%s: %s", what, cudaGetErrorString(e));
}

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kMiB = 1024.0 * 1024.0;
constexpr size_t kAlign = 256;
size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
// moe_gpu.cpp's kStageChunk: an expert part (1.3 to 4.2 MiB at DeepSeek-V4.1's q2_K / q3_K) splits over the pool
constexpr size_t kStageChunk = 1 << 20;
// prefetch only when every expert is likely routed to: T K >= 4 n_expert (experts.hpp)
constexpr int64_t kPrefetchPairsPerExpert = 4;

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// the doorbell poll's pause (session.cpp's STRATA_SPIN_PAUSE)
void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// the CUDA runtime's pinned allocator for the Residency's warm buffer (moe.cpp passes the same pair)
void* pinned_alloc(size_t bytes, std::string* error) {
    void* p = nullptr;
    if (const cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocDefault); e != cudaSuccess) {
        *error = std::string("cudaHostAlloc: ") + cudaGetErrorString(e);
        return nullptr;
    }
    return p;
}
void pinned_free(void* p) { cudaFreeHost(p); }

// host memory the copy engine reads directly: cudaHostAlloc'd or cudaHostRegister'ed
bool is_pinned(const void* p) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        cudaGetLastError();   // an unregistered pointer is not an error worth keeping
        return false;
    }
    return a.type == cudaMemoryTypeHost;
}

// an expert's part p (0 up, 1 gate, 2 down) as a qmatmul view
QWeightView part_view(const DenseDims& d, const ExpertLayerHost& L, int p, const void* data) {
    return p == 2 ? QWeightView{L.type[2], d.n_ff, d.n_embd, data} : QWeightView{L.type[p], d.n_embd, d.n_ff, data};
}

// the length of [a0, a1) inside [b0, b1)
double overlap(double a0, double a1, double b0, double b1) { return std::max(0.0, std::min(a1, b1) - std::max(a0, b0)); }

// A call's prefill tiles at most: each (pass, slot) group adds at most one partial tile to its rows / kExpertTileRows,
// an expert is one group (the ring's, or the slab's after regroup_slab) plus one per full slab pass that cuts it.
int64_t tile_capacity(int64_t pairs, int64_t n_expert, int64_t max_pass) {
    return pairs / kExpertTileRows + n_expert + pairs / max_pass + 2;
}

// The tiles of a pass: per slot its rows in pieces of kExpertTileRows.
int64_t pass_tiles(const dsv41::PrefillPass& p, std::vector<int64_t>& count) {
    int32_t n_slots = 0;
    for (int32_t s : p.slots) n_slots = std::max(n_slots, s + 1);
    count.assign((size_t) n_slots, 0);
    for (int32_t s : p.slots) ++count[(size_t) s];
    int64_t t = 0;
    for (int64_t c : count) t += (c + kExpertTileRows - 1) / kExpertTileRows;
    return t;
}

// The plan's slab passes as whole experts. plan_prefill (strata-moe's, shared with moe_gpu.cpp's ggml graphs)
// cuts the slab's pairs in pair order every kPrefillMaxPassPairs, so each slab expert came back in every slab pass with
// a few rows (13 passes x 168 experts of ~23 rows per layer at chunk 16384). Here the slab passes' pairs are
// sorted by slot, in pair order within a slot, and cut into passes of at most max_rows: an expert is cut only when it
// alone has more. The rows stay 0 .. n_slab - 1 in the new order, so the ring passes after them keep theirs.
void regroup_slab(dsv41::PrefillPlan& plan, int64_t max_rows) {
    size_t n_slab = 0;
    while (n_slab < plan.passes.size() && plan.passes[n_slab].set < 0) ++n_slab;
    if (n_slab == 0) return;
    int32_t n_slots = 0;
    int64_t n = 0;
    for (size_t k = 0; k < n_slab; ++k) {
        for (int32_t s : plan.passes[k].slots) n_slots = std::max(n_slots, s + 1);
        n += (int64_t) plan.passes[k].pairs.size();
    }
    std::vector<int64_t> begin((size_t) n_slots + 1, 0);
    for (size_t k = 0; k < n_slab; ++k)
        for (int32_t s : plan.passes[k].slots) ++begin[(size_t) s + 1];
    for (int32_t s = 0; s < n_slots; ++s) begin[(size_t) s + 1] += begin[(size_t) s];
    std::vector<int32_t> pairs((size_t) n);
    std::vector<int64_t> at(begin.begin(), begin.end() - 1);
    for (size_t k = 0; k < n_slab; ++k) {
        const dsv41::PrefillPass& p = plan.passes[k];
        for (size_t i = 0; i < p.pairs.size(); ++i) pairs[(size_t) at[(size_t) p.slots[i]]++] = p.pairs[i];
    }
    std::vector<dsv41::PrefillPass> passes;
    auto open = [&](int64_t row0) {
        passes.emplace_back();
        passes.back().row0 = row0;
    };
    open(0);
    for (int32_t s = 0; s < n_slots; ++s) {
        int64_t i = begin[(size_t) s];
        const int64_t end = begin[(size_t) s + 1];
        if (i == end) continue;
        if (!passes.back().pairs.empty() && (int64_t) passes.back().pairs.size() + (end - i) > max_rows) open(i);
        while (i < end) {
            if ((int64_t) passes.back().pairs.size() == max_rows) open(i);
            dsv41::PrefillPass& p = passes.back();
            const int64_t take = std::min(end - i, max_rows - (int64_t) p.pairs.size());
            p.pairs.insert(p.pairs.end(), pairs.begin() + i, pairs.begin() + i + take);
            p.slots.insert(p.slots.end(), (size_t) take, s);
            i += take;
        }
    }
    for (size_t k = n_slab; k < plan.passes.size(); ++k) passes.push_back(std::move(plan.passes[k]));
    plan.passes.swap(passes);
}

}  // namespace

struct ExpertEngine::Impl final : dsv41::PrefillEngine {
    const Model* m = nullptr;
    std::vector<ExpertLayerTiers>* tiers = nullptr;   // the engine's (a published swap changes them)
    DenseDims d;
    ExpertConfig cfg;
    int64_t max_T = 0, max_pass = 0;
    bool dry = false;
    QmmContext* qmm = nullptr;
    cudaStream_t stream = nullptr;

    dsv41::HotSet hot;
    bool has_hot = false;
    std::unique_ptr<dsv41::Residency> residency;
    std::unique_ptr<dsv41::ActivationCounts> counts;
    std::unique_ptr<CpuExperts> cpu;

    // the slab: per layer per slot the three parts on the device; the ring: per set per slot; the prefetch region:
    // per slot, the ring's slot layout
    void* slab = nullptr;
    size_t slab_bytes = 0;
    std::vector<std::vector<const void*>> slab_parts;   // [layer][slot * 3 + part]
    void* ring = nullptr;
    size_t ring_bytes = 0, ring_slot_bytes = 0;
    size_t part_offset[3] = {0, 0, 0};                  // a ring slot's parts (the layers' largest)
    void* pf = nullptr;
    size_t pf_bytes = 0;
    int n_pf = 0;
    size_t warm_bytes = 0;
    int64_t n_vram = 0, n_warm = 0, n_cold = 0;

    // where the copies read: per layer, whether the mapping is pinned; whether the warm tier is
    bool warm_pinned = false;
    std::vector<uint8_t> mapping_pinned;
    // the copy stream and its events (moe_gpu.cpp), the staging (pinned, kDirectAlign-aligned)
    cudaStream_t copy_stream = nullptr;
    std::vector<cudaEvent_t> copied, consumed, stage_free;
    cudaEvent_t pf_done = nullptr, pf_free = nullptr;
    void* staging_alloc = nullptr;
    char* staging = nullptr;
    size_t stage_bytes = 0, stage_slot_bytes = 0, stage_offset[3] = {0, 0, 0};
    // the prefetch order of each layer (experts.hpp), and what the region holds
    std::vector<std::vector<int32_t>> pf_order;
    int pf_layer = -1;
    std::vector<int32_t> pf_experts;

    // per-call buffers: host pinned, device. A prefill pass: d_x16 and d_h16 f16 and its tiles (experts_gemm.hpp),
    // every pass of a call at its own place in h_tiles / d_tiles. Decode: d_hh (h f32) and d_qx (q8_1 rows) of the
    // hits, d_y pair-indexed, d_ycpu the CPU's rows, d_row_id the identity row map of a layer without a doorbell
    // h_perm, h_row and h_tiles are mapped pinned memory, moved to the device by a kernel (copy_from_mapped)
    // from their device views *_m, so a layer's uploads never queue behind the ring's and the prefetch's copies on
    // the copy engine
    int32_t* h_ids = nullptr;
    int32_t *h_perm = nullptr, *h_perm_m = nullptr;
    int32_t *h_row = nullptr, *h_row_m = nullptr;
    int32_t* d_perm = nullptr;
    int32_t* d_row = nullptr;
    int32_t* d_row_id = nullptr;
    float *d_y = nullptr, *d_ycpu = nullptr, *d_hh = nullptr;
    void* d_qx = nullptr;
    uint16_t *d_x16 = nullptr, *d_h16 = nullptr;
    // decode's VRAM tier on the device (session.hpp TokenHits): per layer the slab's layout and its residency row
    // (d_slot_of [layer][expert]: slot or -1); whether a layer rings the doorbell, whether it has hits at all
    int32_t* d_slot_of = nullptr;
    std::vector<QExpertSlab> slabs;
    std::vector<uint8_t> rings, has_slab;
    // the doorbell (layer.hpp Doorbell): mapped pinned memory, a host and a device view of each region. db_x [T][E]
    // and db_ids [P] the device publishes, db_seq its ring; db_y the CPU's rows, db_miss each pair's row (-1: a hit),
    // db_flag the ring value the host has served. rings_served: the rings decode_serve has handled
    struct Mapped {
        float *x = nullptr, *y = nullptr;
        int32_t *ids = nullptr, *miss = nullptr;
        uint32_t *seq = nullptr, *flag = nullptr;
    };
    Mapped db_h, db_d;
    uint32_t rings_served = 0;
    ExpertTile *h_tiles = nullptr, *h_tiles_m = nullptr;
    ExpertTile* d_tiles = nullptr;
    // the routing's fetch (prefill_fetch, or forward's own): its event, the layer and chunk it was queued for
    cudaEvent_t fetch_ev = nullptr;
    int fetched_il = -1;
    int64_t fetched_T = 0;
    // the call's slab passes wait for the layer's prefetch only from the first one that reads the region (a slot
    // at or past n_vram_call); before, every pass waited, behind the whole prefetch
    bool pf_wait = false;
    int32_t n_vram_call = 0;
    int64_t tile_cap = 0;
    std::vector<int64_t> pass_tile0, pass_tiles;   // per pass of the call: its first tile and its tile count
    size_t buffer_bytes = 0, pinned_bytes = 0;
    std::vector<void*> dev_allocs, host_allocs;

    // the current prefill call (PrefillEngine's callbacks)
    int il = -1;
    const float* xn = nullptr;
    ExpertTimes* times = nullptr;
    dsv41::PrefillPlan plan;
    std::vector<int32_t> hit_row, expert_of_pair, ext_slot;
    std::vector<uint8_t> staged, waited, pf_used;
    std::vector<CopyChunk> stage_chunks;
    int64_t n_slab_passes = 0, slab_passes_done = 0;
    // scratch of the split and the passes; the decode jobs, reused (their token lists keep their capacity)
    std::vector<int32_t> group_begin, group_count, cursor;
    std::vector<CpuExpertJob> jobs;
    size_t n_jobs = 0;
    std::vector<int32_t> job_of, job_expert, miss_cursor;

    // the prefill split (experts.hpp): chunks of up to split_T tokens (0: none take it), its rates, its pinned input
    // rows h_x [split_T][E] and CPU rows h_ysplit [split_T K][E], the event of their upload on the copy stream; per
    // call each expert's pairs, the experts outside the slab by pair count, which run on the CPU, and the GPU's pairs
    // (their expert ids, their indices in the call)
    int64_t split_T = 0;
    PrefillSplitRates rates;
    float *h_x = nullptr, *h_ysplit = nullptr;
    cudaEvent_t cpu_in = nullptr;
    std::vector<int32_t> pair_count, split_order, gpu_eop, gpu_pairs;
    std::vector<uint8_t> on_cpu;
    // per layer whether each expert is in the hot set's slab (the slab init
    // fills), and whether the prompt being read is in the deterministic mode: its split then picks its CPU experts
    // among the experts outside that slab, not outside the slab as it is now (split_plan)
    std::vector<std::vector<uint8_t>> in_hot_slab;
    bool split_hot_slab = false;

    // the hot set: per layer its planner, its spare slots and its homes' rows (the warm tier's slots in the hot
    // set's warm order, then the spare homes); the residency rows' mirror in mapped pinned memory (h_slot_of_m its
    // device view: the upload reads it with the device's threads, not a copy engine, so it never queues behind the
    // swaps' copies) and whether a row changed; the open batch of (layer, swap) and its events on the copy stream,
    // the swaps published at the last boundary (--hot-check reads them), the route trace
    std::vector<HotLayerPlan> plans;
    std::vector<int> spare_slots;
    std::vector<std::vector<ExpertRows>> homes;
    int32_t *h_slot_of = nullptr, *h_slot_of_m = nullptr;
    bool rows_dirty = false;
    std::vector<std::pair<int, HotSwap>> batch, published;
    // the promoted experts go up on the copy stream, the victims down on a stream of their own (the device's other
    // copy engine: PCIe carries both directions at once)
    cudaStream_t down_stream = nullptr;
    cudaEvent_t swap_a = nullptr, swap_b = nullptr, swap_c = nullptr;
    bool batch_open = false, hot_on = false;
    // HotConfig::copy_only: the open batch is dropped when done, not published; per layer whose mapping is pinned
    // (its victims have no home to go to), `spare` scratch homes for the victims' copies down, made at the first
    // copy-only config
    bool batch_copy_only = false;
    std::vector<std::vector<ExpertRows>> sinks;
    size_t sink_bytes = 0;
    HotPolicy policy = HotPolicy::Static;
    bool det = false;   // the running request's deterministic mode (hot_policy.hpp)
    double decay = 1.0;
    size_t spare_slab_bytes = 0, spare_home_bytes = 0;
    int64_t n_promotable = 0;
    // host homes: per layer the slab experts' homes in the hot set's VRAM order (empty: none, the
    // two-way path or a pinned mapping), and how many of the layer's homes are the warm tier's (those after them are
    // host homes, pinned whatever the warm tier's mode); their bytes
    std::vector<std::vector<ExpertRows>> vram_homes;
    std::vector<int32_t> warm_homes;
    size_t host_home_bytes = 0;
    // per layer the slab experts with a host home (in VRAM order; the home order picks them), how many
    std::vector<std::vector<int32_t>> homed;
    int64_t n_homed = 0;
    // the layered reset (hot_policy.hpp HotRestore): its stream, an event at its start and one after each layer's
    // copies, each layer's wait on the engine's stream between two events (its device ms), the order's host side,
    // and the idle reset it reports into
    bool layered = false;
    cudaStream_t restore_stream = nullptr;
    cudaEvent_t restore_a = nullptr;
    std::vector<cudaEvent_t> restore_ev, rwait_a, rwait_b;
    std::vector<uint8_t> rwait_timed;   // per layer: its rwait events were recorded around a wait
    HotRestore restore;
    ExpertEngine::IdleReset* idle = nullptr;
    std::FILE* trace = nullptr;
    int64_t trace_request = 0;
    bool trace_begun = false, trace_on = true;
    std::vector<uint8_t> check_buf;

    // timing (ExpertEngine::set_timing): events from a pool, per layer of the current forward call the intervals
    bool timing = false;
    std::vector<cudaEvent_t> tev;
    size_t tev_used = 0;
    cudaEvent_t base = nullptr;
    struct Span {
        cudaEvent_t a = nullptr, b = nullptr;
    };
    // a pass: its start, after the gather, after gate/up, its end
    struct PassMarks {
        cudaEvent_t a = nullptr, gathered = nullptr, gate_up = nullptr, b = nullptr;
    };
    struct LayerMarks {
        int layer = -1;
        // the routing's fetch queued; the end of the work the caller queued after it (the shared expert, which
        // the host's plan overlaps); the first pass's turn
        cudaEvent_t fetched = nullptr, ready = nullptr, start = nullptr;
        std::vector<Span> copies;
        std::vector<PassMarks> passes;
        Span prefetch;                                    // this layer's prefetch, queued by the call before
        double copy_bytes = 0;
    };
    std::vector<LayerMarks> marks;
    LayerMarks* cur = nullptr;
    Span pending_pf;      // the prefetch queued for pf_layer
    double pending_pf_bytes = 0;

    ~Impl() override {
        if (trace) std::fclose(trace);
        for (void* p : dev_allocs) cudaFree(p);
        for (void* p : host_allocs) cudaFreeHost(p);
        if (staging_alloc) cudaFreeHost(staging_alloc);
        if (slab) cudaFree(slab);
        if (ring) cudaFree(ring);
        if (pf) cudaFree(pf);
        for (auto* v : {&copied, &consumed, &stage_free, &tev, &restore_ev, &rwait_a, &rwait_b})
            for (cudaEvent_t e : *v) cudaEventDestroy(e);
        for (cudaEvent_t e : {pf_done, pf_free, cpu_in, swap_a, swap_b, swap_c, fetch_ev, restore_a})
            if (e) cudaEventDestroy(e);
        if (copy_stream) cudaStreamDestroy(copy_stream);
        if (down_stream) cudaStreamDestroy(down_stream);
        if (restore_stream) cudaStreamDestroy(restore_stream);
    }

    const ExpertLayerHost& layer_host(int l) const { return m->experts()[(size_t) l]; }

    // where expert e's parts are on the host: its home (the warm tier's slot, a spare one after a swap, or its host
    // home), or the model's mapping
    ExpertRows host_rows(int l, int32_t e, ExpertTier tier) const {
        if (tier == ExpertTier::Warm) {
            const int32_t h = plans[(size_t) l].home_of(e);
            if (h < 0) failf("layer %d: expert %d is warm in the tier map and has no home", l, e);
            return homes[(size_t) l][(size_t) h];
        }
        return mapping_rows(l, e);
    }
    ExpertRows mapping_rows(int l, int32_t e) const {
        ExpertRows r;
        const ExpertLayerHost& L = layer_host(l);
        for (int p = 0; p < 3; ++p) r.part[p] = L.data[p] + (size_t) e * L.bytes[p];
        return r;
    }
    // the copy engine reads expert e of layer l where it is (else it is staged)
    bool source_pinned(int l, int32_t e) const {
        if ((*tiers)[(size_t) l].tier[(size_t) e] != ExpertTier::Warm) return mapping_pinned[(size_t) l] != 0;
        // a host home is pinned whatever the warm tier's mode (before the planners exist every warm expert is the
        // warm tier's)
        const HotLayerPlan& hp = plans[(size_t) l];
        return warm_pinned || (hp.n_expert() > 0 && hp.home_of(e) >= warm_homes[(size_t) l]);
    }

    template <typename T> T* dev(size_t n, const char* what, std::string& err) {
        void* p = nullptr;
        if (const cudaError_t e = cudaMalloc(&p, std::max<size_t>(n * sizeof(T), 16)); e != cudaSuccess) {
            err = std::string("experts: ") + what + " (" + std::to_string(n * sizeof(T)) + " bytes): " + cudaGetErrorString(e);
            return nullptr;
        }
        dev_allocs.push_back(p);
        buffer_bytes += n * sizeof(T);
        return static_cast<T*>(p);
    }
    // mapped pinned memory (doorbell_init's cudaHostAllocMapped): the host's pointer, and the device's in *dev
    template <typename T> T* mapped(size_t n, T** dev, const char* what, std::string& err) {
        void* p = nullptr;
        if (const cudaError_t e = cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 64), cudaHostAllocMapped); e != cudaSuccess) {
            err = std::string("experts: mapped ") + what + ": " + cudaGetErrorString(e);
            return nullptr;
        }
        host_allocs.push_back(p);
        pinned_bytes += n * sizeof(T);
        std::memset(p, 0, std::max<size_t>(n * sizeof(T), 64));
        if (const cudaError_t e = cudaHostGetDevicePointer((void**) dev, p, 0); e != cudaSuccess) {
            err = std::string("experts: the device view of mapped ") + what + ": " + cudaGetErrorString(e);
            return nullptr;
        }
        return static_cast<T*>(p);
    }
    template <typename T> T* pinned(size_t n, const char* what, std::string& err) {
        void* p = nullptr;
        if (const cudaError_t e = cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 16), cudaHostAllocDefault); e != cudaSuccess) {
            err = std::string("experts: pinned ") + what + ": " + cudaGetErrorString(e);
            return nullptr;
        }
        host_allocs.push_back(p);
        pinned_bytes += n * sizeof(T);
        return static_cast<T*>(p);
    }

    cudaEvent_t timing_event() {
        if (tev_used == tev.size()) {
            cudaEvent_t e = nullptr;
            ck(cudaEventCreate(&e), "cudaEventCreate (expert timing)");
            tev.push_back(e);
        }
        return tev[tev_used++];
    }
    cudaEvent_t mark(cudaStream_t s) {
        const cudaEvent_t e = timing_event();
        ck(cudaEventRecord(e, s), "cudaEventRecord (expert timing)");
        return e;
    }

    char* ring_slot(int set, int k) const {
        return static_cast<char*>(ring) + ((size_t) set * (size_t) cfg.ring_set + (size_t) k) * ring_slot_bytes;
    }
    char* pf_slot(int k) const { return static_cast<char*>(pf) + (size_t) k * ring_slot_bytes; }
    char* stage_slot(int h, int k, int part) const {
        return staging + ((size_t) h * (size_t) cfg.ring_set + (size_t) k) * stage_slot_bytes + stage_offset[part];
    }

    // ---- PrefillEngine (moe_prefill.h) on the copy stream and the engine's stream, as moe_gpu.cpp's Prefill
    void stage_wait(int h) override {
        const auto t0 = std::chrono::steady_clock::now();
        ck(cudaEventSynchronize(stage_free[(size_t) h]), "cudaEventSynchronize (prefill staging)");
        times->stage_wait_ms += ms_since(t0);
        ++times->host_syncs;
    }

    // the pass's staged experts from where they are into its staging set, on the CPU pool. A set's first use in a call
    // needs no wait: the previous call's copies out of it were waited for by its passes, which the routing fetch of
    // this call waited for
    void stage(const dsv41::PrefillPass& p) override {
        const auto t0 = std::chrono::steady_clock::now();
        const ExpertLayerTiers& t = (*tiers)[(size_t) il];
        const ExpertLayerHost& L = layer_host(il);
        stage_chunks.clear();
        for (size_t k = 0; k < p.experts.size(); ++k) {
            if (!p.staged[k]) continue;
            const ExpertRows src = host_rows(il, p.experts[k], t.tier[(size_t) p.experts[k]]);
            for (int q = 0; q < 3; ++q) {
                char* dst = stage_slot(p.stage, (int) k, q);
                for (size_t off = 0; off < L.bytes[q]; off += kStageChunk)
                    stage_chunks.push_back({src.part[q] + off, dst + off, std::min(kStageChunk, L.bytes[q] - off)});
                times->staged_bytes += (double) L.bytes[q];
            }
        }
        cpu->copy(stage_chunks.data(), stage_chunks.size());
        times->stage_ms += ms_since(t0);
    }

    void copy_wait_compute(int set) override {
        ck(cudaStreamWaitEvent(copy_stream, consumed[(size_t) set], 0), "cudaStreamWaitEvent (prefill copy)");
        waited[(size_t) set] = 1;
    }

    void copy(const dsv41::PrefillPass& p) override {
        // a set's first copy of the call waits for the previous call's last pass on it
        if (!waited[(size_t) p.set]) copy_wait_compute(p.set);
        const ExpertLayerTiers& t = (*tiers)[(size_t) il];
        const ExpertLayerHost& L = layer_host(il);
        Span sp;
        if (timing) sp.a = mark(copy_stream);
        for (size_t k = 0; k < p.experts.size(); ++k) {
            const int32_t e = p.experts[k];
            const ExpertRows src = p.staged[k] ? ExpertRows{} : host_rows(il, e, t.tier[(size_t) e]);
            char* dst = ring_slot(p.set, (int) k);
            for (int q = 0; q < 3; ++q) {
                const void* from = p.staged[k] ? (const void*) stage_slot(p.stage, (int) k, q) : (const void*) src.part[q];
                ck(cudaMemcpyAsync(dst + part_offset[q], from, L.bytes[q], cudaMemcpyHostToDevice, copy_stream),
                   "an expert into the prefill ring");
            }
            const double bytes = (double) (L.bytes[0] + L.bytes[1] + L.bytes[2]);
            times->copy_bytes += bytes;
            if (timing) cur->copy_bytes += bytes;
        }
        if (timing) {
            sp.b = mark(copy_stream);
            cur->copies.push_back(sp);
        }
        times->ring_experts += (int64_t) p.experts.size();
        ck(cudaEventRecord(copied[(size_t) p.set], copy_stream), "cudaEventRecord (prefill copy)");
        if (p.stage >= 0) ck(cudaEventRecord(stage_free[(size_t) p.stage], copy_stream), "cudaEventRecord (prefill staging)");
    }

    void compute_wait_copy(int set) override {
        ck(cudaStreamWaitEvent(stream, copied[(size_t) set], 0), "cudaStreamWaitEvent (prefill pass)");
    }

    // Every pass's rows grouped by slot (a ring pass already is, by expert) at its rows row0 .., each pair's output row
    // (all n_pairs of the call: a split's CPU pairs have theirs already), and its tiles (each slot's rows in pieces of kExpertTileRows, its parts where the pass reads them), on the host,
    // then one upload of each before the first pass. A pass queues kernels only: in the first A100 run a layer's
    // compute grew 10 to 14 ms per pass (16 to 20 passes per layer: 180 to 238 ms, where experts_test --gemm's rates
    // give ~80 ms of kernels), about one ring set's copy (205 MB at 20 GB/s), the likely wait of each pass's small
    // uploads of its pairs and tiles behind the ring's copies on the copy engine.
    void prepare_passes(int n_vram_l, bool pf_here, int64_t n_pairs) {
        const size_t n_pass = plan.passes.size();
        pass_tile0.assign(n_pass, 0);
        pass_tiles.assign(n_pass, 0);
        int64_t nt = 0;
        for (size_t k = 0; k < n_pass; ++k) {
            const dsv41::PrefillPass& p = plan.passes[k];
            const int64_t n = (int64_t) p.pairs.size();
            const int n_slots = p.set < 0 ? n_vram_l + (pf_here ? (int) pf_experts.size() : 0) : cfg.ring_set;
            group_count.assign((size_t) n_slots, 0);
            for (int32_t s : p.slots) ++group_count[(size_t) s];
            group_begin.assign((size_t) n_slots + 1, 0);
            for (int s = 0; s < n_slots; ++s) group_begin[(size_t) s + 1] = group_begin[(size_t) s] + group_count[(size_t) s];
            cursor.assign(group_begin.begin(), group_begin.end() - 1);
            int32_t* perm = h_perm + p.row0;
            for (int64_t i = 0; i < n; ++i) {
                const int32_t pos = cursor[(size_t) p.slots[(size_t) i]]++;
                perm[pos] = p.pairs[(size_t) i];
                h_row[p.pairs[(size_t) i]] = (int32_t) (p.row0 + pos);
            }
            pass_tile0[k] = nt;
            for (int s = 0; s < n_slots; ++s) {
                const int64_t c = group_count[(size_t) s];
                if (!c) continue;
                ExpertTile t{};
                for (int q = 0; q < 3; ++q) {
                    if (p.set >= 0) t.w[q] = ring_slot(p.set, s) + part_offset[q];
                    else if (s < n_vram_l) t.w[q] = slab_parts[(size_t) il][(size_t) s * 3 + (size_t) q];
                    else t.w[q] = pf_slot(s - n_vram_l) + part_offset[q];
                }
                if (((uintptr_t) t.w[0] | (uintptr_t) t.w[1] | (uintptr_t) t.w[2]) & 15)
                    failf("layer %d: slot %d's parts at %p, %p, %p: the grouped GEMM reads them from 16-byte boundaries", il, s,
                          t.w[0], t.w[1], t.w[2]);
                for (int64_t r = 0; r < c; r += kExpertTileRows) {
                    if (nt == tile_cap) failf("layer %d: more than %lld prefill tiles in a call", il, (long long) tile_cap);
                    t.row0 = (int32_t) (group_begin[(size_t) s] + r);
                    t.rows = (int32_t) std::min<int64_t>(kExpertTileRows, c - r);
                    h_tiles[nt++] = t;
                }
            }
            pass_tiles[k] = nt - pass_tile0[k];
        }
        times->tiles += nt;
        // one launch reads all three from mapped memory (not a copy engine, see h_perm)
        MappedCopy c[3];
        int n = 0;
        if (plan.n_rows) c[n++] = MappedCopy{d_perm, h_perm_m, plan.n_rows};
        if (n_pairs) c[n++] = MappedCopy{d_row, h_row_m, n_pairs};
        if (nt) c[n++] = MappedCopy{d_tiles, h_tiles_m, nt * (int64_t) (sizeof(ExpertTile) / 4)};
        if (n) copy_from_mapped(c, n, stream);
    }

    void compute(const dsv41::PrefillPass& p) override {
        const size_t k = (size_t) (&p - plan.passes.data());
        // the first slab pass that reads the prefetch region waits for its copies (before its start mark: the wait is
        // a stall, not compute)
        if (pf_wait && p.set < 0 && !p.slots.empty() && *std::max_element(p.slots.begin(), p.slots.end()) >= n_vram_call) {
            ck(cudaStreamWaitEvent(stream, pf_done, 0), "cudaStreamWaitEvent (prefetch)");
            pf_wait = false;
        }
        PassMarks pm;
        if (timing) pm.a = mark(stream);
        const ExpertLayerHost& L = layer_host(il);
        const int64_t n = (int64_t) p.pairs.size(), E = d.n_embd, F = d.n_ff, nt = pass_tiles[k];
        const ExpertTile* tiles = d_tiles + pass_tile0[k];
        expert_gather_rows_f16(xn, d_perm + p.row0, d.n_expert_used, n, d.n_embd, d_x16, stream);
        if (timing) pm.gathered = mark(stream);
        expert_gate_up(tiles, nt, L.type[1], d_x16, E, F, L.limit, d_h16, stream);
        if (timing) pm.gate_up = mark(stream);
        expert_gemm(tiles, nt, 2, L.type[2], d_h16, F, E, d_y + p.row0 * E, E, stream);
        if (timing) {
            pm.b = mark(stream);
            cur->passes.push_back(pm);
        }
        if (p.set >= 0) {
            ck(cudaEventRecord(consumed[(size_t) p.set], stream), "cudaEventRecord (prefill pass)");
        } else if (++slab_passes_done == n_slab_passes && pf_free) {
            // the prefetch region is read by the slab's passes only: the next prefetch may overwrite it now (no
            // region and no event when every expert is in the slab)
            ck(cudaEventRecord(pf_free, stream), "cudaEventRecord (prefetch region)");
        }
    }

    // The first experts of layer l's prefetch order into the prefetch region, on the copy stream once the region's
    // last reader is done; only for a prefill chunk of T tokens that likely routes to every expert, and not for a
    // split chunk (its PCIe time goes to the experts its split sends through the ring).
    void prefetch(int l, int64_t T) {
        if (!n_pf || T * d.n_expert_used < kPrefetchPairsPerExpert * d.n_expert || pf_order[(size_t) l].empty() ||
            T <= split_T)
            return;
        const ExpertLayerTiers& t = (*tiers)[(size_t) l];
        const ExpertLayerHost& L = layer_host(l);
        // the order's first n_pf experts outside the slab (a swap may have promoted some of them since init)
        pf_experts.clear();
        for (int32_t e : pf_order[(size_t) l]) {
            if ((int) pf_experts.size() == n_pf) break;
            if (t.slot[(size_t) e] < 0) pf_experts.push_back(e);
        }
        if (pf_experts.empty()) return;
        ck(cudaStreamWaitEvent(copy_stream, pf_free, 0), "cudaStreamWaitEvent (prefetch)");
        pending_pf = Span{};
        if (timing) pending_pf.a = mark(copy_stream);
        for (size_t k = 0; k < pf_experts.size(); ++k) {
            const ExpertRows src = host_rows(l, pf_experts[k], t.tier[(size_t) pf_experts[k]]);
            for (int q = 0; q < 3; ++q)
                ck(cudaMemcpyAsync(pf_slot((int) k) + part_offset[q], src.part[q], L.bytes[q], cudaMemcpyHostToDevice,
                                   copy_stream), "an expert into the prefetch region");
        }
        if (timing) pending_pf.b = mark(copy_stream);
        ck(cudaEventRecord(pf_done, copy_stream), "cudaEventRecord (prefetch)");
        pending_pf_bytes = (double) pf_experts.size() * (double) (L.bytes[0] + L.bytes[1] + L.bytes[2]);
        times->prefetch_bytes += pending_pf_bytes;
        times->prefetched += (int64_t) pf_experts.size();
        pf_layer = l;
    }

    // The split of a prefill call of layer l (experts.hpp), from pair_count: the experts outside the slab with a
    // pair, fewest pairs first, go to the CPU while that lowers the call's end, max(ring, CPU) + the CPU rows' upload,
    // in ms at the measured rates: ring = their bytes / pcie_gbs, CPU = per expert its bytes / cpu_gbs plus cpu_pair_ms
    // per pair after its first, upload = their pairs x a row of n_embd f32 / pcie_gbs (it follows the ring's copies on
    // the copy stream). Sorted by pair count the CPU's cost grows and the ring's falls at every step, so the first step
    // that does not help ends it. PrefillSplit::Cpu takes them all. Fills on_cpu and the CPU jobs (jobs[0 .. n),
    // each expert's tokens in pair order, their rows y0 .. one after another); returns n.
    // in the deterministic mode the candidates are the experts outside the hot set's slab whatever the slab holds
    // now (a batch's swaps follow every row's routes), so which experts the CPU computes, and so the GPU's prefill
    // GEMM computes the others, is a function of the prompt alone: a prompt read beside decoding slots gets its alone
    // read's bits (a candidate in the slab now is read from its host home or the mapping; a hot set expert swapped
    // out goes through the ring as any GPU pair outside the slab)
    size_t split_plan(int l, int64_t T) {
        const ExpertLayerTiers& t = (*tiers)[(size_t) l];
        const ExpertLayerHost& L = layer_host(l);
        const int K = d.n_expert_used;
        const int64_t P = T * K, NE = d.n_expert;
        const std::vector<uint8_t>& hot = in_hot_slab[(size_t) l];
        split_order.clear();
        for (int32_t e = 0; e < NE; ++e) {
            const bool outside = split_hot_slab ? !hot[(size_t) e] : t.slot[(size_t) e] < 0;
            if (outside && pair_count[(size_t) e] > 0) split_order.push_back(e);
        }
        std::stable_sort(split_order.begin(), split_order.end(),
                         [&](int32_t a, int32_t b) { return pair_count[(size_t) a] < pair_count[(size_t) b]; });
        size_t n = split_order.size();
        if (cfg.prefill_split == PrefillSplit::Measured && !(rates.pcie_gbs > 0 && rates.cpu_gbs > 0)) {
            n = 0;   // no rates (no expert was measured): the ring takes every one
        } else if (cfg.prefill_split == PrefillSplit::Measured) {
            const double bytes = (double) (L.bytes[0] + L.bytes[1] + L.bytes[2]);
            const double per_ms_pcie = rates.pcie_gbs * 1e6, per_ms_cpu = rates.cpu_gbs * 1e6;   // bytes per ms
            const double ring_ms = bytes / per_ms_pcie, cpu_ms = bytes / per_ms_cpu, row_ms = 4.0 * d.n_embd / per_ms_pcie;
            double ring = ring_ms * (double) split_order.size(), cpu = 0, up = 0;
            n = 0;
            for (int32_t e : split_order) {
                const double c = (double) pair_count[(size_t) e];
                const double ring2 = ring - ring_ms, cpu2 = cpu + cpu_ms + rates.cpu_pair_ms * (c - 1), up2 = up + c * row_ms;
                if (std::max(ring2, cpu2) + up2 >= std::max(ring, cpu) + up) break;
                ring = ring2;
                cpu = cpu2;
                up = up2;
                ++n;
            }
            times->split_ring_est_ms += ring;
            times->split_cpu_est_ms += cpu + up;
        }
        on_cpu.assign((size_t) NE, 0);
        job_of.assign((size_t) NE, -1);
        for (size_t k = 0; k < n; ++k) {
            const int32_t e = split_order[k];
            on_cpu[(size_t) e] = 1;
            job_of[(size_t) e] = (int32_t) k;
            jobs[k].rows = host_rows(l, e, t.tier[(size_t) e]);
            jobs[k].tokens.clear();
        }
        for (int64_t p = 0; p < P; ++p)
            if (on_cpu[(size_t) h_ids[p]]) jobs[(size_t) job_of[(size_t) h_ids[p]]].tokens.push_back((int32_t) (p / K));
        int64_t y0 = 0;
        for (size_t k = 0; k < n; ++k) {
            jobs[k].y0 = y0;
            y0 += (int64_t) jobs[k].tokens.size();
        }
        return n;
    }

    // The split's rates (PrefillSplitRates), once: up to 16 of the first such layer's experts outside the slab (at
    // most a ring set) copied from where they are into the ring's first set on the copy stream (events; one warm-up
    // copy first), then computed on the CPU pool on one token each and on 16 tokens each (host clock, the faster of
    // two runs each; zero input rows, which the kernel computes like any other).
    void measure_split_rates() {
        int l0 = -1;
        std::vector<int32_t> cal;
        for (int l = 0; l < m->n_layer() && l0 < 0; ++l) {
            for (int32_t e = 0; e < d.n_expert && (int) cal.size() < std::min(16, cfg.ring_set); ++e)
                if ((*tiers)[(size_t) l].tier[(size_t) e] != ExpertTier::Vram) cal.push_back(e);
            if (!cal.empty()) l0 = l;
        }
        if (l0 < 0) return;
        const ExpertLayerTiers& t = (*tiers)[(size_t) l0];
        const ExpertLayerHost& L = layer_host(l0);
        const double bytes = (double) (L.bytes[0] + L.bytes[1] + L.bytes[2]);
        auto copy_all = [&](size_t n) {
            for (size_t k = 0; k < n; ++k) {
                const ExpertRows src = host_rows(l0, cal[k], t.tier[(size_t) cal[k]]);
                for (int q = 0; q < 3; ++q)
                    ck(cudaMemcpyAsync(ring_slot(0, (int) k) + part_offset[q], src.part[q], L.bytes[q], cudaMemcpyHostToDevice,
                                       copy_stream), "the split's rate copies");
            }
        };
        cudaEvent_t a = nullptr, b = nullptr;
        ck(cudaEventCreate(&a), "cudaEventCreate (split rates)");
        ck(cudaEventCreate(&b), "cudaEventCreate (split rates)");
        copy_all(1);
        ck(cudaEventRecord(a, copy_stream), "cudaEventRecord (split rates)");
        copy_all(cal.size());
        ck(cudaEventRecord(b, copy_stream), "cudaEventRecord (split rates)");
        ck(cudaEventSynchronize(b), "the split's rate copies");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime (split rates)");
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        rates.pcie_gbs = bytes * (double) cal.size() / ((double) ms * 1e6);

        constexpr int64_t kTokens = 16;
        const std::vector<float> x((size_t) (kTokens * d.n_embd), 0.f);
        std::vector<float> y((size_t) ((int64_t) cal.size() * kTokens * d.n_embd));
        std::vector<CpuExpertJob> js(cal.size());
        auto best_ms = [&](int64_t tokens) {
            for (size_t k = 0; k < cal.size(); ++k) {
                js[k].rows = host_rows(l0, cal[k], t.tier[(size_t) cal[k]]);
                js[k].tokens.resize((size_t) tokens);
                for (int64_t i = 0; i < tokens; ++i) js[k].tokens[(size_t) i] = (int32_t) i;
                js[k].y0 = (int64_t) k * tokens;
            }
            double best = 0;
            for (int rep = 0; rep < 2; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                cpu->run(L, d, x.data(), kTokens, js.data(), js.size(), y.data());
                const double t = ms_since(t0);
                best = rep == 0 ? t : std::min(best, t);
            }
            return best;
        };
        const double one = best_ms(1), many = best_ms(kTokens);
        rates.cpu_gbs = bytes * (double) cal.size() / (one * 1e6);
        rates.cpu_pair_ms = std::max(0.0, (many - one) / ((double) cal.size() * (double) (kTokens - 1)));
    }

    // ---- the hot set (experts.hpp, hot_policy.hpp)
    void* slab_part(int l, int32_t slot, int p) const {
        return const_cast<void*>(slab_parts[(size_t) l][(size_t) slot * 3 + (size_t) p]);
    }

    // layer l's swap s published: the planner, the tier map and the residency row's mirror (uploaded by upload_rows)
    void publish_swap(int l, const HotSwap& s) {
        HotLayerPlan& hp = plans[(size_t) l];
        const int32_t old = hp.slot_of(s.out);
        hp.publish(s);
        ExpertLayerTiers& t = (*tiers)[(size_t) l];
        t.tier[(size_t) s.in] = ExpertTier::Vram;
        t.slot[(size_t) s.in] = s.slot;
        t.vram[(size_t) s.slot] = s.in;
        // a victim with a home (its new spare one, or its host home) is warm, else back to its pinned mapping
        t.tier[(size_t) s.out] = hp.home_of(s.out) >= 0 ? ExpertTier::Warm : ExpertTier::Cold;
        t.slot[(size_t) s.out] = -1;
        t.vram[(size_t) old] = -1;
        int32_t* row = h_slot_of + (size_t) l * (size_t) d.n_expert;
        row[s.in] = s.slot;
        row[s.out] = -1;
        rows_dirty = true;
    }

    // layer l's swap s: the promoted expert from its home or pinned mapping into its free slot on the copy stream, the
    // victim from its slot into its free home (none when its mapping is pinned or it has a host home) on the down
    // stream; the bytes copied (copy only: a victim without a home goes down into the layer's k-th scratch home, k its
    // swap's index in the layer's batch, below `spare`, unless host homes drop the victims: then nothing goes down)
    double queue_swap(int l, const HotSwap& s, int k) {
        const ExpertLayerHost& L = layer_host(l);
        const ExpertRows src = s.from_home >= 0 ? homes[(size_t) l][(size_t) s.from_home] : mapping_rows(l, s.in);
        const int32_t old = plans[(size_t) l].slot_of(s.out);
        double bytes = 0;
        for (int p = 0; p < 3; ++p) {
            ck(cudaMemcpyAsync(slab_part(l, s.slot, p), src.part[p], L.bytes[p], cudaMemcpyHostToDevice, copy_stream),
               "a promoted expert into the slab");
            bytes += (double) L.bytes[p];
        }
        if (s.to_home >= 0 || (batch_copy_only && !cfg.hot.host_homes)) {
            // the home is this engine's: a warm-tier slot (strata-moe's residency only filled it), a spare one, or
            // (copy only) a scratch one
            if (s.to_home < 0 && (size_t) k >= sinks[(size_t) l].size())
                failf("copy-only swap %d of layer %d: the layer has %zu scratch homes", k, l, sinks[(size_t) l].size());
            const ExpertRows& to = s.to_home >= 0 ? homes[(size_t) l][(size_t) s.to_home] : sinks[(size_t) l][(size_t) k];
            double down = 0;
            for (int p = 0; p < 3; ++p) {
                ck(cudaMemcpyAsync(const_cast<uint8_t*>(to.part[p]), slab_part(l, old, p), L.bytes[p], cudaMemcpyDeviceToHost,
                                   down_stream), "a demoted expert into its home");
                down += (double) L.bytes[p];
            }
            bytes += down;
            // the copies down that are left (a copy-only scratch one is dropped with its swap)
            if (s.to_home >= 0) {
                ++batch_downs;
                batch_down_bytes += down;
            }
        }
        return bytes;
    }

    // A batch: every layer's next swaps (reset: toward the hot set's experts), their copies after an event on the copy
    // stream (the down stream waits for it, so both intervals start there) and before one on each stream; false when
    // no layer has one
    double batch_bytes = 0, batch_down_bytes = 0;
    int64_t batch_downs = 0;
    bool start_batch(bool reset) {
        if (batch_open) failf("a batch of swaps started while another is open");
        batch.clear();
        batch_copy_only = !reset && cfg.hot.copy_only;
        std::vector<HotSwap> sw;
        for (int l = 0; l < m->n_layer(); ++l) {
            if (!spare_slots[(size_t) l]) continue;
            sw.clear();
            if (reset) plans[(size_t) l].plan_static(spare_slots[(size_t) l], sw);
            else plans[(size_t) l].plan(cfg.hot, sw);
            for (const HotSwap& x : sw) batch.emplace_back(l, x);
        }
        if (batch.empty()) return false;
        const auto t0 = std::chrono::steady_clock::now();
        ck(cudaEventRecord(swap_a, copy_stream), "cudaEventRecord (swaps)");
        ck(cudaStreamWaitEvent(down_stream, swap_a, 0), "cudaStreamWaitEvent (swaps)");
        batch_bytes = batch_down_bytes = 0;
        batch_downs = 0;
        for (size_t i = 0, k = 0; i < batch.size(); ++i) {
            k = i > 0 && batch[i].first == batch[i - 1].first ? k + 1 : 0;
            batch_bytes += queue_swap(batch[i].first, batch[i].second, (int) k);
        }
        ck(cudaEventRecord(swap_b, copy_stream), "cudaEventRecord (swaps)");
        ck(cudaEventRecord(swap_c, down_stream), "cudaEventRecord (swaps)");
        if (!reset) times->swap_queue_ms += ms_since(t0);
        batch_open = true;
        return true;
    }

    // The open batch published once its copies are done (wait: the host waits for them); false while they copy
    bool finish_batch(bool wait, bool reset) {
        if (!batch_open) return true;
        for (cudaEvent_t e : {swap_b, swap_c}) {
            if (wait) {
                ck(cudaEventSynchronize(e), "the swaps' copies");
            } else {
                const cudaError_t q = cudaEventQuery(e);
                if (q == cudaErrorNotReady) return false;
                ck(q, "the swaps' copies");
            }
        }
        // the batch's device time: from its start to the later stream's end
        float up = 0, down = 0;
        ck(cudaEventElapsedTime(&up, swap_a, swap_b), "cudaEventElapsedTime (swaps)");
        ck(cudaEventElapsedTime(&down, swap_a, swap_c), "cudaEventElapsedTime (swaps)");
        const float ms = std::max(up, down);
        published.clear();
        for (const auto& [l, x] : batch) {
            if (batch_copy_only) {
                plans[(size_t) l].abandon(x);
                continue;
            }
            publish_swap(l, x);
            published.emplace_back(l, x);
        }
        (reset ? times->reset_swaps : batch_copy_only ? times->swaps_dropped : times->swaps) += (int64_t) batch.size();
        (reset ? times->reset_bytes : times->swap_bytes) += batch_bytes;
        if (!batch_copy_only) {
            (reset ? times->reset_downs : times->swap_downs) += batch_downs;
            (reset ? times->reset_down_bytes : times->swap_down_bytes) += batch_down_bytes;
        }
        if (!reset) {
            times->swap_copy_ms += ms;
            ++times->swap_batches_done;
        }
        batch.clear();
        batch_open = false;
        batch_copy_only = false;
        return true;
    }

    // HotConfig::copy_only's scratch homes, once: `spare` per layer that swaps and whose victims have no home (its
    // mapping pinned); false with err when the pinned allocation fails
    bool make_sinks(std::string& err) {
        if (!sinks.empty() || cfg.hot.host_homes) return true;
        sinks.assign((size_t) m->n_layer(), {});
        for (int l = 0; l < m->n_layer(); ++l) {
            if (!spare_slots[(size_t) l] || !mapping_pinned[(size_t) l]) continue;
            const ExpertLayerHost& L = layer_host(l);
            size_t bytes = 0, off[3] = {0, 0, 0};
            for (int p = 0; p < 3; ++p) {
                off[p] = bytes;
                bytes += align_up(L.bytes[p]);
            }
            const int n = spare_slots[(size_t) l];
            uint8_t* block = pinned<uint8_t>((size_t) n * bytes, "the copy-only scratch homes", err);
            if (!block) return false;
            sink_bytes += (size_t) n * bytes;
            for (int k = 0; k < n; ++k) {
                ExpertRows r;
                for (int p = 0; p < 3; ++p) r.part[p] = block + (size_t) k * bytes + off[p];
                sinks[(size_t) l].push_back(r);
            }
        }
        return true;
    }

    // the residency rows to the device on the engine's stream, before its next work, once one has changed: the whole
    // mirror (n_layer x n_expert int32, 61 KB on the real model) in one launch that reads mapped memory (the host
    // changes the mirror only at a boundary or a request's start, never while a token runs)
    void upload_rows() {
        if (!rows_dirty) return;
        const MappedCopy c{d_slot_of, h_slot_of_m, (int64_t) m->n_layer() * d.n_expert};
        copy_from_mapped(&c, 1, stream);
        rows_dirty = false;
    }

    // every swap published and its rows queued (a prefill call, a request's start)
    void drain() {
        if (!hot_on) return;
        finish_batch(true, false);
        upload_rows();
    }

    // --hot-check: every layer's device residency row, the host's slot map and tier map against the planner, its
    // invariants, and the bytes of every slot and home the last published swaps filled against the model's mapping
    void check_residency() {
        const int n_layer = m->n_layer();
        const size_t NE = (size_t) d.n_expert;
        poll_restore(true);   // a layered reset's copies done before the slots are read
        ck(cudaStreamSynchronize(stream), "the residency rows' upload (--hot-check)");
        std::vector<int32_t> dev((size_t) n_layer * NE);
        ck(cudaMemcpy(dev.data(), d_slot_of, dev.size() * 4, cudaMemcpyDeviceToHost), "the residency rows (--hot-check)");
        for (int l = 0; l < n_layer; ++l) {
            const HotLayerPlan& hp = plans[(size_t) l];
            const ExpertLayerTiers& t = (*tiers)[(size_t) l];
            if (const std::string why = hp.check(); !why.empty()) failf("--hot-check, layer %d: %s", l, why.c_str());
            for (size_t e = 0; e < NE; ++e) {
                const int32_t s = hp.slot_of((int32_t) e);
                const ExpertTier want = s >= 0 ? ExpertTier::Vram : hp.home_of((int32_t) e) >= 0 ? ExpertTier::Warm : ExpertTier::Cold;
                if (dev[(size_t) l * NE + e] != s || t.slot[e] != s || h_slot_of[(size_t) l * NE + e] != s || t.tier[e] != want)
                    failf("--hot-check, layer %d expert %zu: device row %d, host slot map %d, mirror %d, planner %d; tier %d, "
                          "planner's %d", l, e, dev[(size_t) l * NE + e], t.slot[e], h_slot_of[(size_t) l * NE + e], s,
                          (int) t.tier[e], (int) want);
            }
            for (int s = 0; s < hp.n_slots(); ++s)
                if (t.vram[(size_t) s] != hp.slot_expert(s))
                    failf("--hot-check, layer %d slot %d: the tier map holds %d, the planner %d", l, s, t.vram[(size_t) s],
                          hp.slot_expert(s));
        }
        for (const auto& [l, x] : published) {
            const ExpertLayerHost& L = layer_host(l);
            const ExpertRows want_in = mapping_rows(l, x.in), want_out = mapping_rows(l, x.out);
            for (int p = 0; p < 3; ++p) {
                check_buf.resize(L.bytes[p]);
                ck(cudaMemcpy(check_buf.data(), slab_part(l, x.slot, p), L.bytes[p], cudaMemcpyDeviceToHost), "a slot (--hot-check)");
                if (std::memcmp(check_buf.data(), want_in.part[p], L.bytes[p]) != 0)
                    failf("--hot-check, layer %d: slot %d's part %d is not expert %d's bytes", l, x.slot, p, x.in);
                // the victim's home: the one it was copied into, or (host homes) the one it kept, still its bytes
                const int32_t h = x.to_home >= 0 ? x.to_home : plans[(size_t) l].home_of(x.out);
                if (h >= 0 && std::memcmp(homes[(size_t) l][(size_t) h].part[p], want_out.part[p], L.bytes[p]) != 0)
                    failf("--hot-check, layer %d: home %d's part %d is not expert %d's bytes", l, h, p, x.out);
            }
        }
        ++times->hot_checks;
    }

    // ---- host homes and the layered reset (experts.hpp, hot_policy.hpp HotRestore)

    // layer l's planner layout from the tiers, the hot set's warm order and the homed experts (choose_homes)
    HotLayerPlan::Layout layout_of(int l, bool pinned) const {
        const ExpertLayerTiers& t = (*tiers)[(size_t) l];
        HotLayerPlan::Layout lay;
        lay.n_expert = d.n_expert;
        for (int32_t e : t.vram)
            if (e >= 0) lay.vram.push_back(e);
        lay.spare = spare_slots[(size_t) l];
        lay.mapping_pinned = pinned;
        if ((size_t) l < homed.size()) lay.homed = homed[(size_t) l];
        for (const dsv41::HotLayer& h : hot.layers)
            if (h.layer == l)
                for (int32_t e : h.warm)
                    if (t.tier[(size_t) e] == ExpertTier::Warm) lay.warm.push_back(e);
        return lay;
    }

    // one expert's home of layer l, each part 256-byte aligned as the slab packs it
    size_t home_bytes_of(int l) const {
        size_t b = 0;
        for (int p = 0; p < 3; ++p) b += align_up(layer_host(l).bytes[p]);
        return b;
    }

    // the slab experts that get a host home, in the home order (experts.hpp), and their
    // bytes: the layers that swap (skip_pinned: those whose mapping is not pinned, a pinned mapping being its experts'
    // host copy already; a dry run cannot tell and takes every one) in layer order, each with every slab expert while
    // host_homes_max allows (-1: no limit), the layer the limit cuts through from the end of its VRAM order. Exits when
    // the limit is more than the slab has.
    void choose_homes(bool skip_pinned) {
        const int n_layer = m->n_layer();
        homed.assign((size_t) n_layer, {});
        host_home_bytes = 0;
        n_homed = 0;
        if (!cfg.hot.host_homes) return;
        auto takes = [&](int l) { return spare_slots[(size_t) l] && !(skip_pinned && mapping_pinned[(size_t) l]); };
        int64_t eligible = 0;
        for (int l = 0; l < n_layer; ++l)
            for (int32_t e : (*tiers)[(size_t) l].vram) eligible += takes(l) && e >= 0;
        const int64_t want = cfg.hot.host_homes_max;
        if (want > eligible)
            failf("--hot-homes %lld: the slab has %lld experts that can take a host home (in the layers that swap%s)",
                  (long long) want, (long long) eligible, skip_pinned ? " and whose mapping is not pinned" : "");
        int64_t left = want < 0 ? eligible : want;
        for (int l = 0; l < n_layer && left > 0; ++l) {
            if (!takes(l)) continue;
            std::vector<int32_t> v;
            for (int32_t e : (*tiers)[(size_t) l].vram)
                if (e >= 0) v.push_back(e);
            const size_t take = (size_t) std::min<int64_t>(left, (int64_t) v.size());
            homed[(size_t) l].assign(v.end() - (std::ptrdiff_t) take, v.end());
            left -= (int64_t) take;
            n_homed += (int64_t) take;
            host_home_bytes += take * home_bytes_of(l);
        }
    }

    // a pinned home for every homed expert (choose_homes), filled from the mapping by the CPU pool before the slab is
    // (from them); false with err when the host cannot pin them
    bool fill_host_homes(std::string& err) {
        const int n_layer = m->n_layer();
        vram_homes.assign((size_t) n_layer, {});
        choose_homes(true);
        std::vector<CopyChunk> chunks;
        size_t pinned_before = 0;
        for (int l = 0; l < n_layer; ++l) {
            if (homed[(size_t) l].empty()) continue;
            const ExpertLayerHost& L = layer_host(l);
            size_t home_bytes = 0, off[3] = {0, 0, 0};
            for (int p = 0; p < 3; ++p) {
                off[p] = home_bytes;
                home_bytes += align_up(L.bytes[p]);
            }
            const std::vector<int32_t>& experts = homed[(size_t) l];
            void* block = nullptr;
            const size_t bytes = experts.size() * home_bytes;
            if (const cudaError_t e = cudaHostAlloc(&block, std::max<size_t>(bytes, 16), cudaHostAllocDefault); e != cudaSuccess) {
                char msg[320];
                std::snprintf(msg, sizeof msg, "experts: the host homes of layer %d (%zu experts, %.1f MiB; %.1f MiB "
                              "pinned before it, %.1f MiB in all): %s; --hot-homes spare keeps the two-way path", l,
                              experts.size(), (double) bytes / kMiB, (double) pinned_before / kMiB,
                              (double) host_home_bytes / kMiB, cudaGetErrorString(e));
                err = msg;
                return false;
            }
            host_allocs.push_back(block);
            pinned_before += bytes;
            uint8_t* base = static_cast<uint8_t*>(block);
            chunks.clear();
            for (size_t k = 0; k < experts.size(); ++k) {
                ExpertRows r;
                for (int p = 0; p < 3; ++p) {
                    r.part[p] = base + k * home_bytes + off[p];
                    chunks.push_back({L.data[p] + (size_t) experts[k] * L.bytes[p], const_cast<uint8_t*>(r.part[p]), L.bytes[p]});
                }
                vram_homes[(size_t) l].push_back(r);
            }
            cpu->copy(chunks.data(), chunks.size());
        }
        return true;
    }

    // layer l's in-place restored swap s (the planner has it already): the tier map and the residency row's mirror
    void apply_restored(int l, const HotSwap& s) {
        ExpertLayerTiers& t = (*tiers)[(size_t) l];
        t.tier[(size_t) s.in] = ExpertTier::Vram;
        t.slot[(size_t) s.in] = s.slot;
        t.vram[(size_t) s.slot] = s.in;
        t.tier[(size_t) s.out] = plans[(size_t) l].home_of(s.out) >= 0 ? ExpertTier::Warm : ExpertTier::Cold;
        t.slot[(size_t) s.out] = -1;
        int32_t* row = h_slot_of + (size_t) l * (size_t) d.n_expert;
        row[s.in] = s.slot;
        row[s.out] = -1;
        rows_dirty = true;
    }

    // The slab back to the hot set's in place, layer by layer on the restore stream: each layer's copies from the
    // promoted experts' homes into their slots, then its event (HotRestore's order); the planners, the tier map and
    // the rows (queued on the engine's stream) at the hot set's at once. The engine's stream is idle (a request's end)
    // and no batch is open. Returns the swaps and adds their bytes to *bytes. Amendment 3d: a layer whose victims may
    // need a copy down (partial host homes) is left to the rounds after this (its event recorded with no copies), and
    // *rounds counts the layers that swap and were left.
    int64_t restore_layers(double* bytes, int* rounds) {
        poll_restore(true);   // the last one's events are reused: seen done first
        const int n_layer = m->n_layer();
        restore.begin(n_layer);
        rwait_timed.assign((size_t) n_layer, 0);
        ck(cudaEventRecord(restore_a, restore_stream), "cudaEventRecord (restore)");
        int64_t n = 0;
        std::vector<HotSwap> sw;
        for (int l = 0; l < n_layer; ++l) {
            sw.clear();
            if (spare_slots[(size_t) l] && plans[(size_t) l].restores_in_place()) plans[(size_t) l].restore_in_place(sw);
            else *rounds += spare_slots[(size_t) l] != 0;
            const ExpertLayerHost& L = layer_host(l);
            for (const HotSwap& x : sw) {
                const ExpertRows src = x.from_home >= 0 ? homes[(size_t) l][(size_t) x.from_home] : mapping_rows(l, x.in);
                for (int p = 0; p < 3; ++p) {
                    ck(cudaMemcpyAsync(slab_part(l, x.slot, p), src.part[p], L.bytes[p], cudaMemcpyHostToDevice,
                                       restore_stream), "a restored expert into the slab");
                    *bytes += (double) L.bytes[p];
                }
                apply_restored(l, x);
            }
            n += (int64_t) sw.size();
            ck(cudaEventRecord(restore_ev[(size_t) l], restore_stream), "cudaEventRecord (a layer's restore)");
            restore.record(l);
        }
        upload_rows();
        return n;
    }

    // An open layered reset seen done (wait: the host waits for its last event; else false while it still copies): its
    // device ms and the waits the engine's stream made for it (every call that waited has been waited for by the host
    // when this runs) into the idle reset; the order closed
    bool poll_restore(bool wait) {
        if (!restore.open()) return true;
        const cudaEvent_t last = restore_ev.back();
        if (wait) {
            ck(cudaEventSynchronize(last), "the layered restore");
        } else {
            const cudaError_t q = cudaEventQuery(last);
            if (q == cudaErrorNotReady) return false;
            ck(q, "the layered restore");
        }
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, restore_a, last), "cudaEventElapsedTime (restore)");
        idle->device_ms = ms;
        for (int l = 0; l < restore.n_layer(); ++l) {
            if (!rwait_timed[(size_t) l]) continue;
            float w = 0;
            ck(cudaEventSynchronize(rwait_b[(size_t) l]), "a layer's wait for its restore");
            ck(cudaEventElapsedTime(&w, rwait_a[(size_t) l], rwait_b[(size_t) l]), "cudaEventElapsedTime (a layer's wait)");
            idle->wait_ms += w;
            ++idle->layers_waited;
            if (w > idle->wait_max_ms || idle->wait_max_layer < 0) {
                idle->wait_max_ms = w;
                idle->wait_max_layer = l;
            }
        }
        idle->seen_done = true;
        restore.done();
        return true;
    }

    // before layer il's work that reads the slab: the engine's stream waits for layer il's restore, timed between two
    // events (never inside a graph capture: the token graph waits once before its launch, wait_restore_all)
    void wait_restore(int il) {
        if (!restore.open() || !restore.wait_before(il)) return;
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        ck(cudaStreamIsCapturing(stream, &cs), "cudaStreamIsCapturing");
        if (cs != cudaStreamCaptureStatusNone) failf("layer %d would wait for its restore inside a graph capture", il);
        ck(cudaEventRecord(rwait_a[(size_t) il], stream), "cudaEventRecord (a layer's wait)");
        ck(cudaStreamWaitEvent(stream, restore_ev[(size_t) il], 0), "cudaStreamWaitEvent (a layer's restore)");
        ck(cudaEventRecord(rwait_b[(size_t) il], stream), "cudaEventRecord (a layer's wait)");
        rwait_timed[(size_t) il] = 1;
    }

    // the route trace (experts.hpp ExpertConfig::route_trace; the format in tools/dsv41/engine/hot_policy_sim.py)
    void trace_ids(char kind, int l, int64_t T, const int32_t* ids) {
        if (!trace_begun) {
            std::fprintf(trace, "R %lld %s reset=0 resume=0\n", (long long) trace_request, hot_policy_name(policy));
            trace_begun = true;
        }
        const int K = d.n_expert_used;
        for (int64_t t = 0; t < (kind == 'P' ? 1 : T); ++t) {
            if (kind == 'P') std::fprintf(trace, "P %d %lld", l, (long long) T);
            else std::fprintf(trace, "D %d", l);
            const int64_t n = kind == 'P' ? T * K : K;
            for (int64_t i = 0; i < n; ++i) std::fprintf(trace, " %d", ids[t * K + i]);
            std::fputc('\n', trace);
        }
        if (std::ferror(trace)) failf("writing the route trace %s failed", cfg.route_trace.c_str());
    }
};

ExpertEngine::ExpertEngine() = default;
ExpertEngine::~ExpertEngine() = default;

bool ExpertEngine::init(const Model& m, const ExpertConfig& cfg, int64_t max_T, QmmContext* qmm, void* stream, bool dry,
                        std::string& err) {
    if (impl_) failf("ExpertEngine::init called twice");
    impl_ = std::make_unique<Impl>();
    Impl& I = *impl_;
    I.m = &m;
    I.tiers = &tiers_;
    I.d = m.dense_dims();
    I.cfg = cfg;
    I.max_T = max_T;
    I.dry = dry;
    I.qmm = qmm;
    I.stream = (cudaStream_t) stream;
    I.times = &times_;
    I.idle = &idle_reset_;
    const DenseDims& d = I.d;
    const int n_layer = m.n_layer();
    const int64_t NE = d.n_expert, K = d.n_expert_used;
    if (max_T < 1 || max_T > 65535) failf("calls of up to %lld tokens (1..65535)", (long long) max_T);
    if (cfg.ring_set < 1 || cfg.ring_sets < 1 || cfg.stage_sets < 1 || cfg.prefetch_experts < 0)
        failf("a ring of %d sets of %d experts, %d staging sets, %d prefetched experts", cfg.ring_sets, cfg.ring_set,
              cfg.stage_sets, cfg.prefetch_experts);
    if (!dry && !qmm) failf("ExpertEngine::init without a qmatmul context");

    // the tiers: the hot set's VRAM experts and (with a warm tier) its warm lists, every other expert cold
    tiers_.assign((size_t) n_layer, ExpertLayerTiers{});
    for (ExpertLayerTiers& t : tiers_) {
        t.tier.assign((size_t) NE, ExpertTier::Cold);
        t.slot.assign((size_t) NE, -1);
    }
    I.has_hot = !cfg.hot_set.empty();
    if (I.has_hot) {
        I.hot = dsv41::load_hot_set(cfg.hot_set.c_str());
        for (const dsv41::HotLayer& h : I.hot.layers) {
            if (h.layer >= n_layer) failf("hot set %s: layer %d, the model has %d", cfg.hot_set.c_str(), h.layer, n_layer);
            const ExpertLayerHost& L = m.experts()[(size_t) h.layer];
            const int types[3] = {(int) h.up_type, (int) h.gate_type, (int) h.down_type};
            if (h.n_expert != NE || h.n_embd != d.n_embd || h.n_ff != d.n_ff || types[0] != L.type[0] ||
                types[1] != L.type[1] || types[2] != L.type[2])
                failf("hot set %s, layer %d: %lld experts of [%lld, %lld] types %d/%d/%d, the model has %lld of [%d, %d] "
                      "types %d/%d/%d", cfg.hot_set.c_str(), h.layer, (long long) h.n_expert, (long long) h.n_embd,
                      (long long) h.n_ff, types[0], types[1], types[2], (long long) NE, d.n_embd, d.n_ff, L.type[0],
                      L.type[1], L.type[2]);
            ExpertLayerTiers& t = tiers_[(size_t) h.layer];
            for (int32_t e : h.experts) {
                t.tier[(size_t) e] = ExpertTier::Vram;
                t.slot[(size_t) e] = (int32_t) t.vram.size();
                t.vram.push_back(e);
            }
            if (cfg.ram_gib > 0)
                for (int32_t e : h.warm) t.tier[(size_t) e] = ExpertTier::Warm;
        }
        if (cfg.ram_gib > 0 && I.hot.ram_budget_gib < 0)
            failf("--ram-gb %g needs a hot set with warm lists (hot_set.py --ram-budget-gib); %s has none", cfg.ram_gib,
                  cfg.hot_set.c_str());
    } else if (cfg.ram_gib > 0) {
        failf("--ram-gb %g without a hot set: the warm lists come from the hot set", cfg.ram_gib);
    }
    // spare slab slots in every layer with experts in and out of the hot set's slab (experts.hpp)
    hot_config_check(cfg.hot);
    I.spare_slots.assign((size_t) n_layer, 0);
    for (int l = 0; l < n_layer; ++l) {
        std::vector<int32_t>& v = tiers_[(size_t) l].vram;
        if (cfg.hot.spare > 0 && !v.empty() && (int64_t) v.size() < NE) {
            I.spare_slots[(size_t) l] = cfg.hot.spare;
            v.resize(v.size() + (size_t) cfg.hot.spare, -1);
        }
    }
    // each layer's prefetch order: the hot set's warm list as it orders it, then the cold experts by id
    I.pf_order.assign((size_t) n_layer, {});
    for (int l = 0; l < n_layer; ++l) {
        const ExpertLayerTiers& t = tiers_[(size_t) l];
        std::vector<int32_t>& o = I.pf_order[(size_t) l];
        for (const dsv41::HotLayer& h : I.hot.layers)
            if (h.layer == l)
                for (int32_t e : h.warm)
                    if (t.tier[(size_t) e] == ExpertTier::Warm) o.push_back(e);
        for (int32_t e = 0; e < NE; ++e)
            if (t.tier[(size_t) e] == ExpertTier::Cold) o.push_back(e);
    }

    // sizes: the slab (each part 256-byte aligned), the warm tier as the Residency packs it, the ring's slots
    for (int l = 0; l < n_layer; ++l) {
        const ExpertLayerHost& L = m.experts()[(size_t) l];
        size_t slot = 0, warm_slot = 0;
        for (int p = 0; p < 3; ++p) {
            slot += align_up(L.bytes[p]);
            warm_slot += dsv41::direct_slot_bytes(L.bytes[p]);
        }
        for (ExpertTier x : tiers_[(size_t) l].tier) {
            if (x == ExpertTier::Vram) { I.slab_bytes += slot; ++I.n_vram; }
            if (x == ExpertTier::Warm) { I.warm_bytes += warm_slot; ++I.n_warm; }
            if (x == ExpertTier::Cold) ++I.n_cold;
        }
        I.slab_bytes += slot * (size_t) I.spare_slots[(size_t) l];
        I.spare_slab_bytes += slot * (size_t) I.spare_slots[(size_t) l];
    }
    // the host homes the layers that swap take (in a dry run as if no mapping were pinned;
    // fill_host_homes chooses them again without the pinned layers)
    I.choose_homes(false);
    if (I.slab_bytes > 0 && (double) I.slab_bytes > cfg.vram_gib * kGiB)
        failf("hot set %s: its %lld VRAM experts and %d spare slots per layer (--hot-spare; %.3f GiB) take %.3f GiB, more "
              "than --vram-gb %g", cfg.hot_set.c_str(), (long long) I.n_vram, cfg.hot.spare,
              (double) I.spare_slab_bytes / kGiB, (double) I.slab_bytes / kGiB, cfg.vram_gib);
    if ((double) I.warm_bytes > cfg.ram_gib * kGiB && I.n_warm > 0)
        failf("hot set %s: its %lld warm experts take %.3f GiB, more than --ram-gb %g", cfg.hot_set.c_str(),
              (long long) I.n_warm, (double) I.warm_bytes / kGiB, cfg.ram_gib);
    size_t off = 0, part_max[3] = {0, 0, 0};
    for (int p = 0; p < 3; ++p) {
        I.part_offset[p] = off;
        for (const ExpertLayerHost& L : m.experts()) part_max[p] = std::max(part_max[p], L.bytes[p]);
        off += align_up(part_max[p]);
    }
    I.ring_slot_bytes = off;
    const bool any_ring = I.n_vram != (int64_t) n_layer * NE;
    I.ring_bytes = any_ring ? I.ring_slot_bytes * (size_t) cfg.ring_set * (size_t) cfg.ring_sets : 0;
    int64_t max_order = 0;
    for (const std::vector<int32_t>& o : I.pf_order) max_order = std::max<int64_t>(max_order, (int64_t) o.size());
    I.n_pf = (int) std::min<int64_t>(cfg.prefetch_experts, max_order);
    I.pf_bytes = I.ring_slot_bytes * (size_t) I.n_pf;
    const dsv41::StageLayout sl = dsv41::stage_layout(part_max);
    I.stage_slot_bytes = sl.slot_bytes;
    for (int p = 0; p < 3; ++p) I.stage_offset[p] = sl.part_offset[p];
    // allocated once a source turns out unpinned (below; the dry run reports what it would take)
    I.stage_bytes = any_ring ? (size_t) cfg.stage_sets * (size_t) cfg.ring_set * I.stage_slot_bytes : 0;
    I.max_pass = std::min<int64_t>(std::max<int64_t>(dsv41::kPrefillMaxPassPairs, max_T), max_T * K);
    if (cfg.split_max_t < 0) failf("a prefill split for chunks of up to %lld tokens", (long long) cfg.split_max_t);
    I.split_T = any_ring && cfg.prefill_split != PrefillSplit::Off ? std::min(cfg.split_max_t, max_T) : 0;

    // every expert's views: qmatmul's checks, the grouped GEMM's for the prefill passes, the CPU kernel's types; and
    // decode's slab per layer as the fill below packs it (a slot's parts 256-byte aligned in order, the layers' slots
    // one after another), for qmatmul_experts. A layer rings the doorbell when the CPU may have a miss of it, or when
    // counts are recorded (they take the ids on the host)
    I.slabs.assign((size_t) n_layer, QExpertSlab{});
    I.rings.assign((size_t) n_layer, 0);
    I.has_slab.assign((size_t) n_layer, 0);
    size_t slab_at = 0;
    for (int l = 0; l < n_layer; ++l) {
        const ExpertLayerHost& L = m.experts()[(size_t) l];
        for (int p = 0; p < 3; ++p) qmatmul_check(part_view(d, L, p, L.data[p]), 1, L.name[p].c_str());
        expert_gemm_check(L.type[0], L.type[1], L.type[2], d.n_embd, d.n_ff);
        bool any_cpu = false;
        for (ExpertTier x : tiers_[(size_t) l].tier) any_cpu = any_cpu || x != ExpertTier::Vram;
        if (any_cpu) cpu_experts_check(L, d, cfg.cpu_numerics);
        QExpertSlab& sl = I.slabs[(size_t) l];
        for (int p = 0; p < 3; ++p) {
            sl.part_offset[p] = sl.slot_bytes;
            sl.slot_bytes += align_up(L.bytes[p]);
            sl.type[p] = L.type[p];
        }
        sl.n_embd = d.n_embd;
        sl.n_ff = d.n_ff;
        sl.base = reinterpret_cast<const void*>(slab_at);   // an offset into the slab until it is allocated
        slab_at += tiers_[(size_t) l].vram.size() * sl.slot_bytes;
        I.has_slab[(size_t) l] = !tiers_[(size_t) l].vram.empty();
        I.rings[(size_t) l] = any_cpu || !cfg.counts.empty() || !cfg.route_trace.empty();
        if (I.has_slab[(size_t) l]) qmatmul_experts_check(sl, L.name[1].c_str());
    }
    if (slab_at != I.slab_bytes) failf("the decode slab's layout takes %zu bytes, the slab %zu", slab_at, I.slab_bytes);
    if (dry) return true;

    // the CPU pool (the misses; first the host homes' fill)
    I.cpu = std::make_unique<CpuExperts>(cfg.threads, cfg.cpu_numerics);
    I.cpu->set_spin_us(cfg.pool_spin_us);
    // where the copies read: a layer's mapping is pinned when the caller registered it (the smoke does); only the
    // experts outside the slab are ever copied
    I.mapping_pinned.assign((size_t) n_layer, 0);
    for (int l = 0; l < n_layer; ++l) {
        const ExpertLayerHost& L = m.experts()[(size_t) l];
        I.mapping_pinned[(size_t) l] = is_pinned(L.data[0]) && is_pinned(L.data[1]) && is_pinned(L.data[2]);
    }
    // the host homes, filled from the mapping, then the slab from them
    if (!I.fill_host_homes(err)) return false;

    // the slab, filled from the mapping (or the host homes)
    if (I.slab_bytes) {
        if (const cudaError_t e = cudaMalloc(&I.slab, I.slab_bytes); e != cudaSuccess) {
            err = "the expert slab (" + std::to_string(I.slab_bytes >> 20) + " MiB): " + cudaGetErrorString(e);
            return false;
        }
    }
    I.slab_parts.assign((size_t) n_layer, {});
    char* next = static_cast<char*>(I.slab);
    for (QExpertSlab& sl : I.slabs) sl.base = static_cast<const char*>(I.slab) + reinterpret_cast<uintptr_t>(sl.base);
    for (int l = 0; l < n_layer; ++l) {
        const ExpertLayerHost& L = m.experts()[(size_t) l];
        const std::vector<ExpertRows>& vh = I.vram_homes[(size_t) l];
        const std::vector<int32_t>& hd = I.homed[(size_t) l];
        for (int32_t e : tiers_[(size_t) l].vram) {   // a spare slot (-1) is left as it is
            const size_t k = (size_t) (std::find(hd.begin(), hd.end(), e) - hd.begin());
            for (int p = 0; p < 3; ++p) {
                const uint8_t* src = e < 0 ? nullptr : k < vh.size() ? vh[k].part[p] : L.data[p] + (size_t) e * L.bytes[p];
                if (src) ck(cudaMemcpy(next, src, L.bytes[p], cudaMemcpyHostToDevice), "the expert slab");
                I.slab_parts[(size_t) l].push_back(next);
                next += align_up(L.bytes[p]);
            }
        }
    }
    if (I.ring_bytes) {
        if (const cudaError_t e = cudaMalloc(&I.ring, I.ring_bytes); e != cudaSuccess) {
            err = "the prefill ring (" + std::to_string(I.ring_bytes >> 20) + " MiB): " + cudaGetErrorString(e);
            return false;
        }
    }

    // the warm tier: strata-moe's Residency (pinned by default with a GPU), each hot-set layer bound once
    if (I.n_warm > 0) {
        const dsv41::RamModeChoice mode = dsv41::choose_ram_mode(std::getenv("STRATA_MOE_RAM_MODE"), true);
        if (!mode.error.empty()) failf("%s", mode.error.c_str());
        static const dsv41::PinnedHost pinned{pinned_alloc, pinned_free};
        I.residency = std::make_unique<dsv41::Residency>(I.hot, cfg.ram_gib, true, 8, 16, ::pread, mode.mode, &pinned,
                                                         dsv41::ColdPath::kPageCache);
        I.warm_pinned = I.residency->ram_mode() == dsv41::RamMode::kPinned;
        for (const dsv41::HotLayer& h : I.hot.layers) {
            const ExpertLayerHost& L = m.experts()[(size_t) h.layer];
            dsv41::ExpertTensors t;
            t.layer = h.layer;
            t.n_expert = NE;
            for (int p = 0; p < 3; ++p) {
                t.name[p] = L.name[p].c_str();
                t.data[p] = L.data[p];
                t.type[p] = (ggml_type) L.type[p];
                t.expert_bytes[p] = L.bytes[p];
            }
            I.residency->bind_layer(t);
        }
    }
    // the experts whose copies are staged (an unpinned source; only the experts outside the slab are ever copied)
    bool any_staged = false;
    I.warm_homes.assign((size_t) n_layer, 0);
    I.plans.assign((size_t) n_layer, HotLayerPlan{});   // source_pinned asks a warm expert's planner, once it has one
    I.in_hot_slab.assign((size_t) n_layer, std::vector<uint8_t>((size_t) NE, 0));
    for (int l = 0; l < n_layer; ++l)
        for (int32_t e = 0; e < NE; ++e) I.in_hot_slab[(size_t) l][(size_t) e] = tiers_[(size_t) l].slot[(size_t) e] >= 0;
    for (int l = 0; l < n_layer; ++l) {
        for (int32_t e = 0; e < NE; ++e)
            any_staged = any_staged || (tiers_[(size_t) l].tier[(size_t) e] != ExpertTier::Vram && !I.source_pinned(l, e));
        // a prefetch copies pinned sources only, straight into the region
        std::vector<int32_t>& o = I.pf_order[(size_t) l];
        o.erase(std::remove_if(o.begin(), o.end(), [&](int32_t e) { return !I.source_pinned(l, e); }), o.end());
    }
    // each layer's homes (the warm tier's slots in the hot set's warm order, then the spare homes a layer whose
    // mapping is not pinned needs for its victims, or the host homes of the slab's experts in its
    // VRAM order) and its planner
    I.homes.assign((size_t) n_layer, {});
    for (int l = 0; l < n_layer; ++l) {
        const ExpertLayerHost& L = m.experts()[(size_t) l];
        const std::vector<ExpertRows>& vh = I.vram_homes[(size_t) l];
        HotLayerPlan::Layout lay = I.layout_of(l, I.mapping_pinned[(size_t) l] != 0);
        for (int32_t e : lay.warm) {
            const char* const* w = I.residency->warm_rows(l, e);
            if (!w) failf("layer %d: expert %d is warm in the tier map and not in the warm tier", l, e);
            ExpertRows r;
            for (int p = 0; p < 3; ++p) r.part[p] = reinterpret_cast<const uint8_t*>(w[p]);
            I.homes[(size_t) l].push_back(r);
        }
        I.warm_homes[(size_t) l] = (int32_t) lay.warm.size();
        if (vh.size() != lay.homed.size())
            failf("layer %d: %zu host homes for %zu homed slab experts", l, vh.size(), lay.homed.size());
        I.homes[(size_t) l].insert(I.homes[(size_t) l].end(), vh.begin(), vh.end());
        if (lay.spare > 0 && !lay.warm.empty() && !I.warm_pinned)
            failf("--hot-spare %d: layer %d's warm tier is not pinned (STRATA_MOE_RAM_MODE): a swap copies from pinned "
                  "memory only (pinned is the default with a GPU), or --hot-spare 0 keeps the slab as the hot set has it",
                  lay.spare, l);
        // the spare homes, unless every slab expert has a host home (a layer with only some keeps them)
        if (lay.spare > 0 && !lay.mapping_pinned && !lay.warm.empty() && lay.homed.size() != lay.vram.size()) {
            size_t home_bytes = 0, part_off[3] = {0, 0, 0};
            for (int p = 0; p < 3; ++p) {
                part_off[p] = home_bytes;
                home_bytes += align_up(L.bytes[p]);
            }
            uint8_t* block = I.pinned<uint8_t>((size_t) lay.spare * home_bytes, "the spare homes", err);
            if (!block) return false;
            I.spare_home_bytes += (size_t) lay.spare * home_bytes;
            for (int k = 0; k < lay.spare; ++k) {
                ExpertRows r;
                for (int p = 0; p < 3; ++p) r.part[p] = block + (size_t) k * home_bytes + part_off[p];
                I.homes[(size_t) l].push_back(r);
            }
            lay.spare_homes = lay.spare;
        }
        if (lay.spare > 0) {
            I.n_promotable += (int64_t) lay.warm.size();
            if (lay.mapping_pinned)
                for (ExpertTier x : tiers_[(size_t) l].tier) I.n_promotable += x == ExpertTier::Cold;
        }
        I.plans[(size_t) l].init(lay);
        I.plans[(size_t) l].begin_request(cfg.hot.policy, false);
        I.hot_on = I.hot_on || lay.spare > 0;
    }
    // the reset layer by layer when a layer that swaps drops all its victims with host homes (they, or its pinned
    // mapping, hold every expert); the layers whose victims may still need a copy down reset in rounds
    // beside it (end_request)
    I.layered = false;
    for (int l = 0; l < n_layer; ++l)
        I.layered = I.layered || (cfg.hot.host_homes && I.spare_slots[(size_t) l] && I.plans[(size_t) l].restores_in_place());
    I.policy = cfg.hot.policy;
    I.decay = std::pow(2.0, -1.0 / cfg.hot.half_life);
    if (I.hot_on && cfg.hot.copy_only && !I.make_sinks(err)) return false;
    if (!cfg.route_trace.empty()) {
        I.trace = std::fopen(cfg.route_trace.c_str(), "w");
        if (!I.trace) failf("--route-trace %s: cannot write it", cfg.route_trace.c_str());
        std::fprintf(I.trace, "# dsv41-route-trace/1 n_layer %d n_expert %lld n_used %lld\n", n_layer, (long long) NE,
                     (long long) K);
    }
    max_order = 0;
    for (const std::vector<int32_t>& o : I.pf_order) max_order = std::max<int64_t>(max_order, (int64_t) o.size());
    I.n_pf = (int) std::min<int64_t>(cfg.prefetch_experts, max_order);
    I.pf_bytes = I.ring_slot_bytes * (size_t) I.n_pf;
    if (I.pf_bytes) {
        if (const cudaError_t e = cudaMalloc(&I.pf, I.pf_bytes); e != cudaSuccess) {
            err = "the prefetch region (" + std::to_string(I.pf_bytes >> 20) + " MiB): " + cudaGetErrorString(e);
            return false;
        }
    }

    // the copy stream, its events, the staging (moe_gpu.cpp alloc_prefill)
    if (any_ring) {
        ck(cudaStreamCreateWithFlags(&I.copy_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags (prefill copy)");
        auto sync_event = [] {
            cudaEvent_t e = nullptr;
            ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreateWithFlags");
            return e;
        };
        for (int s = 0; s < cfg.ring_sets; ++s) {
            I.copied.push_back(sync_event());
            I.consumed.push_back(sync_event());
        }
        for (int h = 0; h < cfg.stage_sets; ++h) I.stage_free.push_back(sync_event());
        I.pf_done = sync_event();
        I.pf_free = sync_event();
        if (I.hot_on) {   // the swaps' batches, timed (ExpertTimes::swap_copy_ms)
            ck(cudaStreamCreateWithFlags(&I.down_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags (swaps)");
            ck(cudaEventCreate(&I.swap_a), "cudaEventCreate (swaps)");
            ck(cudaEventCreate(&I.swap_b), "cudaEventCreate (swaps)");
            ck(cudaEventCreate(&I.swap_c), "cudaEventCreate (swaps)");
        }
        if (I.layered) {   // the layered reset: its stream, its timed events
            ck(cudaStreamCreateWithFlags(&I.restore_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags (restore)");
            ck(cudaEventCreate(&I.restore_a), "cudaEventCreate (restore)");
            for (auto* v : {&I.restore_ev, &I.rwait_a, &I.rwait_b}) {
                v->assign((size_t) n_layer, nullptr);
                for (cudaEvent_t& e : *v) ck(cudaEventCreate(&e), "cudaEventCreate (restore)");
            }
        }
        if (!any_staged) I.stage_bytes = 0;
        if (any_staged) {
            // one block more: the layout's parts are kDirectAlign-aligned from an aligned base (moe_gpu.cpp)
            if (const cudaError_t e = cudaHostAlloc(&I.staging_alloc, I.stage_bytes + dsv41::kDirectAlign, cudaHostAllocDefault);
                e != cudaSuccess) {
                err = "pinned prefill staging (" + std::to_string(I.stage_bytes >> 20) + " MiB): " + cudaGetErrorString(e);
                return false;
            }
            const uintptr_t a = reinterpret_cast<uintptr_t>(I.staging_alloc);
            I.staging = reinterpret_cast<char*>((a + dsv41::kDirectAlign - 1) / dsv41::kDirectAlign * dsv41::kDirectAlign);
        }
    }

    if (!cfg.counts.empty()) I.counts = std::make_unique<dsv41::ActivationCounts>(cfg.counts, 4096);

    // the call buffers: a prefill pass's f16 rows and tiles; decode's (at most kExpertSplitMaxT tokens: qmatmul's
    // MMVQ columns) h and q8_1 rows of the hits, their pair-indexed outputs, the CPU's rows, and the doorbell
    const int64_t P = max_T * K, PS = kExpertSplitMaxT * K, E = d.n_embd, F = d.n_ff;
    I.tile_cap = tile_capacity(P, NE, I.max_pass);
    if (!(I.h_ids = I.pinned<int32_t>((size_t) P, "ids", err)) ||
        !(I.h_perm = I.mapped<int32_t>((size_t) P, &I.h_perm_m, "pairs", err)) ||
        !(I.h_row = I.mapped<int32_t>((size_t) P, &I.h_row_m, "rows", err)) ||
        !(I.h_tiles = I.mapped<ExpertTile>((size_t) I.tile_cap, &I.h_tiles_m, "prefill tiles", err)))
        return false;
    ck(cudaEventCreateWithFlags(&I.fetch_ev, cudaEventDisableTiming), "cudaEventCreateWithFlags (the routing's fetch)");
    if (!(I.d_perm = I.dev<int32_t>((size_t) P, "pairs", err)) || !(I.d_row = I.dev<int32_t>((size_t) P, "rows", err)) ||
        !(I.d_x16 = I.dev<uint16_t>((size_t) (I.max_pass * E), "prefill input rows", err)) ||
        !(I.d_h16 = I.dev<uint16_t>((size_t) (I.max_pass * F), "prefill activation", err)) ||
        !(I.d_tiles = I.dev<ExpertTile>((size_t) I.tile_cap, "prefill tiles", err)) ||
        !(I.d_y = I.dev<float>((size_t) (P * E), "expert outputs", err)) ||
        !(I.d_ycpu = I.dev<float>((size_t) (PS * E), "CPU outputs", err)) ||
        !(I.d_hh = I.dev<float>((size_t) (PS * F), "decode activation", err)) ||
        !(I.d_qx = I.dev<uint8_t>(qmatmul_experts_scratch_bytes(I.slabs[0], kExpertSplitMaxT, PS), "decode q8_1 rows", err)) ||
        !(I.d_row_id = I.dev<int32_t>((size_t) PS, "decode identity rows", err)) ||
        !(I.d_slot_of = I.dev<int32_t>((size_t) (n_layer * NE), "decode residency rows", err)) ||
        !(I.h_slot_of = I.mapped<int32_t>((size_t) (n_layer * NE), &I.h_slot_of_m, "the residency rows' mirror", err)))
        return false;
    for (const ExpertLayerHost& L : m.experts())
        for (int p = 0; p < 3; ++p)
            if (L.type[p] != m.experts()[0].type[p]) failf("%s: the layers' expert types differ", L.name[p].c_str());
    // the residency rows and the identity row map, once; h zeroed, so the q8_1 rows of a miss's h (quantized with
    // the hits', never read) are finite from the first call
    {
        for (int l = 0; l < n_layer; ++l) {
            std::copy(tiers_[(size_t) l].slot.begin(), tiers_[(size_t) l].slot.end(), I.h_slot_of + (size_t) l * (size_t) NE);
            I.slabs[(size_t) l].slot_of = I.d_slot_of + (size_t) l * (size_t) NE;
        }
        ck(cudaMemcpy(I.d_slot_of, I.h_slot_of, (size_t) (n_layer * NE) * 4, cudaMemcpyHostToDevice), "the decode residency rows");
        std::vector<int32_t> id((size_t) PS);
        for (int64_t p = 0; p < PS; ++p) id[(size_t) p] = (int32_t) p;
        ck(cudaMemcpy(I.d_row_id, id.data(), id.size() * 4, cudaMemcpyHostToDevice), "the decode identity rows");
        ck(cudaMemset(I.d_hh, 0, (size_t) (PS * F) * 4), "the decode activation");
    }
    Impl::Mapped& h = I.db_h;
    Impl::Mapped& dv = I.db_d;
    if (!(h.x = I.mapped<float>((size_t) (kExpertSplitMaxT * E), &dv.x, "doorbell input rows", err)) ||
        !(h.ids = I.mapped<int32_t>((size_t) PS, &dv.ids, "doorbell ids", err)) ||
        !(h.y = I.mapped<float>((size_t) (PS * E), &dv.y, "CPU output rows", err)) ||
        !(h.miss = I.mapped<int32_t>((size_t) PS, &dv.miss, "CPU row of each pair", err)) ||
        !(h.seq = I.mapped<uint32_t>(1, &dv.seq, "doorbell ring", err)) ||
        !(h.flag = I.mapped<uint32_t>(1, &dv.flag, "doorbell flag", err)))
        return false;
    // the decode jobs: one per expert at most, each token list at the split's tokens
    I.jobs.resize((size_t) NE);
    for (CpuExpertJob& j : I.jobs) j.tokens.reserve((size_t) kExpertSplitMaxT);
    I.job_of.reserve((size_t) NE);
    I.job_expert.reserve((size_t) NE);
    if (I.split_T > 0) {
        if (!(I.h_x = I.pinned<float>((size_t) (I.split_T * E), "the split's input rows", err)) ||
            !(I.h_ysplit = I.pinned<float>((size_t) (I.split_T * K * E), "the split's CPU rows", err)))
            return false;
        ck(cudaEventCreateWithFlags(&I.cpu_in, cudaEventDisableTiming), "cudaEventCreateWithFlags (split)");
        if (cfg.prefill_split == PrefillSplit::Measured) I.measure_split_rates();
    }
    I.group_count.reserve((size_t) NE);
    I.group_begin.reserve((size_t) NE + 1);
    I.cursor.reserve((size_t) NE + 1);
    I.miss_cursor.reserve((size_t) NE);
    layer_stats_.assign((size_t) n_layer, ExpertLayerStats{});
    layer_pairs_.assign((size_t) n_layer, 0);
    layer_hits_.assign((size_t) n_layer, 0);
    return true;
}

void ExpertEngine::begin_call(int64_t T, bool prefill) {
    Impl& I = *impl_;
    if (I.dry) failf("begin_call on a dry ExpertEngine");
    I.timing = timing_;
    I.tev_used = 0;
    I.marks.clear();
    if (timing_) I.base = I.mark(I.stream);
    I.pf_layer = -1;
    if (prefill) I.drain();   // a prefill reads a slab that does not move
    if (prefill && I.copy_stream) I.prefetch(0, T);
}

void ExpertEngine::prefill_fetch(int il, const float* xn, const int32_t* ids, int64_t T) {
    Impl& I = *impl_;
    if (I.dry) failf("prefill_fetch on a dry ExpertEngine");
    const int64_t P = T * I.d.n_expert_used;
    if (T < 1 || T > I.max_T) failf("layer %d: a prefill call of %lld tokens (1..%lld)", il, (long long) T, (long long) I.max_T);
    ck(cudaMemcpyAsync(I.h_ids, ids, (size_t) P * 4, cudaMemcpyDeviceToHost, I.stream), "the routing to the host");
    if (split_call(T))
        ck(cudaMemcpyAsync(I.h_x, xn, (size_t) (T * I.d.n_embd) * 4, cudaMemcpyDeviceToHost, I.stream), "the input rows to the host");
    ck(cudaEventRecord(I.fetch_ev, I.stream), "cudaEventRecord (the routing's fetch)");
    if (timing_) {
        I.marks.emplace_back();
        I.marks.back().layer = il;
        I.marks.back().fetched = I.mark(I.stream);
    }
    I.fetched_il = il;
    I.fetched_T = T;
}

void ExpertEngine::forward(int il, const float* xn, const int32_t* ids, const float* weights, const float* shared,
                           int64_t T, bool prefill, float* out, const void* xq) {
    Impl& I = *impl_;
    if (I.dry) failf("forward on a dry ExpertEngine");
    I.wait_restore(il);   // this layer's slots complete before its passes or hits read them
    if (!prefill) {
        decode_publish(il, xn, ids, T);
        decode_finish(il, xn, ids, weights, shared, T, out, xq);
        decode_serve(il, T);
        return;
    }
    const DenseDims& d = I.d;
    const int K = d.n_expert_used;
    const int64_t P = T * K, E = d.n_embd, NE = d.n_expert;
    if (T < 1 || T > I.max_T) failf("layer %d: a prefill call of %lld tokens (1..%lld)", il, (long long) T, (long long) I.max_T);
    const ExpertLayerTiers& tiers = tiers_[(size_t) il];
    const ExpertLayerHost& L = I.m->experts()[(size_t) il];
    const bool split = split_call(T);
    I.timing = timing_;
    // the one host wait of the call: the routing, into a pinned buffer (a split call: and its input rows, the CPU's),
    // queued by prefill_fetch right after the router, or here
    if (I.fetched_il != il || I.fetched_T != T) prefill_fetch(il, xn, ids, T);
    I.fetched_il = -1;
    Impl::LayerMarks* lm = nullptr;
    if (timing_ && !I.marks.empty() && I.marks.back().layer == il && !I.marks.back().ready) {
        lm = &I.marks.back();
        lm->ready = I.mark(I.stream);
    }
    auto t0 = std::chrono::steady_clock::now();
    ck(cudaEventSynchronize(I.fetch_ev), "the routing to the host");
    ++times_.host_syncs;
    times_.fetch_ms += ms_since(t0);
    for (int64_t p = 0; p < P; ++p)
        if (I.h_ids[p] < 0 || I.h_ids[p] >= NE) failf("layer %d: pair %lld routed to expert %d of %lld", il, (long long) p, I.h_ids[p], (long long) NE);
    if (I.counts) I.counts->record(il, NE, I.h_ids, K, T);
    if (I.spare_slots[(size_t) il]) I.plans[(size_t) il].count_prompt(I.h_ids, P);
    if (I.trace && I.trace_on) I.trace_ids('P', il, T, I.h_ids);
    ++times_.calls;

    t0 = std::chrono::steady_clock::now();
    if (!I.copy_stream && I.n_vram != (int64_t) I.m->n_layer() * NE)
        failf("layer %d: a prefill call with experts outside the slab and no ring", il);
    I.il = il;
    I.xn = xn;
    I.cur = lm;
    I.expert_of_pair.assign(I.h_ids, I.h_ids + P);
    // the slab's slots, then this layer's prefetched experts as the slots after them (read in place)
    const bool pf_here = I.pf_layer == il;
    const int32_t nv = (int32_t) tiers.vram.size();
    I.ext_slot.assign(tiers.slot.begin(), tiers.slot.end());
    if (pf_here)
        for (size_t k = 0; k < I.pf_experts.size(); ++k) I.ext_slot[(size_t) I.pf_experts[k]] = nv + (int32_t) k;
    I.staged.assign((size_t) NE, 0);
    for (int32_t e = 0; e < NE; ++e) I.staged[(size_t) e] = I.ext_slot[(size_t) e] < 0 && !I.source_pinned(il, e);
    // the experts outside the slab this call touches; a split's CPU experts (none otherwise), the GPU's pairs
    I.pair_count.assign((size_t) NE, 0);
    for (int64_t p = 0; p < P; ++p) ++I.pair_count[(size_t) I.h_ids[p]];
    int64_t touched = 0;
    for (int32_t e = 0; e < NE; ++e) touched += tiers.slot[(size_t) e] < 0 && I.pair_count[(size_t) e] > 0;
    const double expert_bytes = (double) (L.bytes[0] + L.bytes[1] + L.bytes[2]);
    times_.touched_experts += touched;
    times_.touched_bytes += (double) touched * expert_bytes;
    const size_t n_cpu = split ? I.split_plan(il, T) : 0;
    if (!n_cpu) I.on_cpu.assign((size_t) NE, 0);
    I.gpu_eop.clear();
    I.gpu_pairs.clear();
    for (int64_t p = 0; p < P; ++p) {
        if (I.on_cpu[(size_t) I.h_ids[p]]) continue;
        I.gpu_eop.push_back(I.h_ids[p]);
        I.gpu_pairs.push_back((int32_t) p);
    }
    dsv41::plan_prefill(I.gpu_eop, NE, &I.ext_slot, I.staged, I.cfg.ring_set, I.cfg.ring_sets, I.cfg.stage_sets,
                        I.plan, I.hit_row, I.max_pass);
    for (dsv41::PrefillPass& pass : I.plan.passes)   // the plan's pair indices back to the call's
        for (int32_t& q : pass.pairs) q = I.gpu_pairs[(size_t) q];
    regroup_slab(I.plan, I.max_pass);
    // the CPU's rows follow the GPU's in d_y: pair p of CPU job j is its row n_rows + y0 + (its place in the job)
    const int64_t n_gpu = I.plan.n_rows;
    int64_t n_cpu_rows = 0;
    if (n_cpu) {
        I.miss_cursor.assign(n_cpu, 0);
        for (int64_t p = 0; p < P; ++p) {
            const int32_t e = I.h_ids[p];
            if (!I.on_cpu[(size_t) e]) continue;
            const int32_t j = I.job_of[(size_t) e];
            I.h_row[p] = (int32_t) (n_gpu + I.jobs[(size_t) j].y0 + I.miss_cursor[(size_t) j]++);
            ++n_cpu_rows;
        }
    }
    if (split) ++times_.split_calls;
    if (I.plan.staged_experts && !I.staging) failf("layer %d: %lld experts to stage and no staging", il, (long long) I.plan.staged_experts);
    times_.plan_ms += ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    I.n_slab_passes = 0;
    I.slab_passes_done = 0;
    for (const dsv41::PrefillPass& p : I.plan.passes) {
        (p.set < 0 ? times_.hits : times_.misses) += (int64_t) p.pairs.size();
        I.n_slab_passes += p.set < 0;
    }
    times_.misses += n_cpu_rows;
    if (pf_here) {
        I.pf_used.assign(I.pf_experts.size(), 0);
        for (int32_t e : I.expert_of_pair)
            if (I.ext_slot[(size_t) e] >= nv) I.pf_used[(size_t) (I.ext_slot[(size_t) e] - nv)] = 1;
        for (uint8_t u : I.pf_used) times_.prefetch_used += u;
        if (lm) {
            lm->prefetch = I.pending_pf;
            lm->copy_bytes += I.pending_pf_bytes;
        }
    }
    // the passes' uploads first (from mapped memory), then the passes; the first slab pass that reads the
    // prefetch region waits for it (compute)
    I.pf_wait = pf_here;
    I.n_vram_call = nv;
    I.prepare_passes(nv, pf_here, P);
    if (lm) lm->start = I.mark(I.stream);
    if (!I.n_slab_passes && I.copy_stream) ck(cudaEventRecord(I.pf_free, I.stream), "cudaEventRecord (prefetch region)");
    I.waited.assign((size_t) I.cfg.ring_sets, 0);
    dsv41::run_prefill(I.plan, I);
    // the split's CPU experts while the device runs the passes, then their rows up behind the ring's copies
    if (n_cpu) {
        const auto tc = std::chrono::steady_clock::now();
        I.cpu->run(L, d, I.h_x, T, I.jobs.data(), n_cpu, I.h_ysplit);
        times_.split_cpu_ms += ms_since(tc);
        times_.split_cpu_experts += (int64_t) n_cpu;
        times_.split_cpu_pairs += n_cpu_rows;
        times_.split_cpu_bytes += (double) n_cpu * expert_bytes;
        ck(cudaMemcpyAsync(I.d_y + n_gpu * E, I.h_ysplit, (size_t) (n_cpu_rows * E) * 4, cudaMemcpyHostToDevice,
                           I.copy_stream), "the split's CPU rows");
        ck(cudaEventRecord(I.cpu_in, I.copy_stream), "cudaEventRecord (the split's CPU rows)");
        ck(cudaStreamWaitEvent(I.stream, I.cpu_in, 0), "cudaStreamWaitEvent (the split's CPU rows)");
    }
    expert_combine(I.d_y, I.d_ycpu, I.d_row, weights, shared, T, K, (int) E, out, I.stream);
    I.pf_layer = -1;
    if (il + 1 < I.m->n_layer() && I.copy_stream) I.prefetch(il + 1, T);
    I.cur = nullptr;
    times_.queue_ms += ms_since(t0);
}

namespace {

void check_decode(int il, int64_t T, int n_layer) {
    if (il < 0 || il >= n_layer || T < 1 || T > kExpertSplitMaxT)
        failf("a decode call of layer %d of %d, %lld tokens (1..%lld)", il, n_layer, (long long) T, (long long) kExpertSplitMaxT);
}

}  // namespace

void ExpertEngine::decode_publish(int il, const float* xn, const int32_t* ids, int64_t T) {
    Impl& I = *impl_;
    if (I.dry) failf("decode_publish on a dry ExpertEngine");
    check_decode(il, T, I.m->n_layer());
    if (!I.rings[(size_t) il]) return;
    expert_doorbell_publish(xn, ids, T * I.d.n_embd, T * I.d.n_expert_used, I.db_d.x, I.db_d.ids, I.db_d.seq, I.stream);
}

void ExpertEngine::decode_finish(int il, const float* xn, const int32_t* ids, const float* weights, const float* shared,
                                 int64_t T, float* out, const void* xq) {
    Impl& I = *impl_;
    if (I.dry) failf("decode_finish on a dry ExpertEngine");
    check_decode(il, T, I.m->n_layer());
    const DenseDims& d = I.d;
    const int K = d.n_expert_used;
    const int64_t P = T * K;
    if (I.has_slab[(size_t) il])
        qmatmul_experts(I.slabs[(size_t) il], xn, T, ids, K, I.m->experts()[(size_t) il].limit, I.d_hh, I.d_qx, I.d_y,
                        I.stream, xq);
    const int32_t* rows = I.d_row_id;
    if (I.rings[(size_t) il]) {
        // the CPU's rows exist once the host has served the ring: wait for its flag right before they are read
        expert_doorbell_wait(I.db_d.flag, I.db_d.seq, I.stream);
        expert_misses_in(I.db_d.y, I.db_d.miss, P, d.n_embd, I.d_ycpu, I.d_row, I.stream);
        rows = I.d_row;
    }
    expert_combine(I.d_y, I.d_ycpu, rows, weights, shared, T, K, d.n_embd, out, I.stream);
}

void ExpertEngine::decode_serve(int il, int64_t T) {
    Impl& I = *impl_;
    if (I.dry) failf("decode_serve on a dry ExpertEngine");
    check_decode(il, T, I.m->n_layer());
    const DenseDims& d = I.d;
    const int K = d.n_expert_used;
    const int64_t P = T * K, NE = d.n_expert;
    ++times_.calls;
    layer_pairs_[(size_t) il] += P;
    if (!I.rings[(size_t) il]) {   // every expert in the slab: every pair a hit, nothing for the host
        times_.hits += P;
        layer_hits_[(size_t) il] += P;
        return;
    }
    // the ring (session_run_token's loop): a memory poll, a stream query every 2 ms to notice a fault or a stream
    // that finished without ringing, and a 20 s bound
    const uint32_t want = ++I.rings_served;
    volatile const uint32_t* seq = I.db_h.seq;
    auto t0 = std::chrono::steady_clock::now();
    auto last_query = t0;
    for (uint32_t spins = 0; *seq != want; ++spins) {
        cpu_relax();
        if ((spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_query > std::chrono::milliseconds(2)) {
            last_query = now;
            const cudaError_t q = cudaStreamQuery(I.stream);
            if (q != cudaErrorNotReady && *seq != want)
                failf("layer %d: the doorbell never rang (ring %u, want %u): %s", il, *seq, want,
                      q == cudaSuccess ? "the stream finished first" : cudaGetErrorString(q));
        }
        if (now - t0 > std::chrono::seconds(20)) failf("layer %d: no doorbell ring %u in 20 s (at %u)", il, want, *seq);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    times_.ring_wait_ms += ms_since(t0);
    ++times_.doorbells;

    // the split: the misses as CPU jobs by expert in first-use order (the engine's, reused), each pair's CPU row
    t0 = std::chrono::steady_clock::now();
    const ExpertLayerTiers& tiers = tiers_[(size_t) il];
    const ExpertLayerHost& L = I.m->experts()[(size_t) il];
    const int32_t* ids = I.db_h.ids;
    for (int64_t p = 0; p < P; ++p)
        if (ids[p] < 0 || ids[p] >= NE) failf("layer %d: pair %lld routed to expert %d of %lld", il, (long long) p, ids[p], (long long) NE);
    if (I.counts) I.counts->record(il, NE, ids, K, T);
    if (I.spare_slots[(size_t) il]) I.plans[(size_t) il].count_token(ids, (int) P, I.decay);
    if (I.trace && I.trace_on) I.trace_ids('D', il, T, ids);
    int64_t n_hit = 0;
    I.job_of.assign((size_t) NE, -1);
    I.n_jobs = decode_miss_jobs(ids, P, K, tiers.slot.data(), I.job_of.data(), I.job_expert, I.jobs.data(), I.db_h.miss, n_hit);
    int64_t n_miss = 0;
    for (size_t k = 0; k < I.n_jobs; ++k) {
        const int32_t e = I.job_expert[k];
        I.jobs[k].rows = I.host_rows(il, e, tiers.tier[(size_t) e]);
        n_miss += (int64_t) I.jobs[k].tokens.size();
    }
    // the call's distinct experts, its CPU jobs and the slab experts its hits read (a pair's expert counted once)
    int64_t distinct_hits = 0;
    for (int64_t p = 0; p < P; ++p) {
        if (tiers.slot[(size_t) ids[p]] < 0) continue;
        bool first = true;
        for (int64_t q = 0; q < p && first; ++q) first = ids[q] != ids[p];
        distinct_hits += first;
    }
    times_.distinct += (int64_t) I.n_jobs + distinct_hits;
    times_.plan_ms += ms_since(t0);
    times_.hits += n_hit;
    times_.misses += n_miss;
    layer_hits_[(size_t) il] += n_hit;

    // the misses on the CPU from the published rows into mapped memory, then the flag (the device's wait reads it)
    if (n_miss) {
        const auto tc = std::chrono::steady_clock::now();
        I.cpu->run(L, d, I.db_h.x, T, I.jobs.data(), I.n_jobs, I.db_h.y);
        times_.cpu_ms += ms_since(tc);
        times_.cpu_experts += (int64_t) I.n_jobs;
        times_.cpu_bytes += (double) I.n_jobs * (double) (L.bytes[0] + L.bytes[1] + L.bytes[2]);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *(volatile uint32_t*) I.db_h.flag = want;
}

void ExpertEngine::end_call() {
    Impl& I = *impl_;
    I.poll_restore(false);   // after the caller's wait: a layered reset this call waited for is done
    if (!I.timing || I.marks.empty()) return;
    // every interval from the call's base event, in ms
    auto at = [&](cudaEvent_t e) {
        float ms = 0;
        ck(cudaEventSynchronize(e), "cudaEventSynchronize (expert timing)");
        ck(cudaEventElapsedTime(&ms, I.base, e), "cudaEventElapsedTime (expert timing)");
        return (double) ms;
    };
    for (const Impl::LayerMarks& lm : I.marks) {
        ExpertLayerStats& st = layer_stats_[(size_t) lm.layer];
        // the front ends at ready (after the shared expert the caller queued behind the fetch)
        const double fetched = at(lm.ready ? lm.ready : lm.fetched), start = at(lm.start);
        std::vector<std::pair<double, double>> passes;
        for (const Impl::PassMarks& s : lm.passes) {
            const double a = at(s.a), g = at(s.gathered), u = at(s.gate_up), b = at(s.b);
            passes.emplace_back(a, b);
            st.gather_ms += g - a;
            st.gate_up_ms += u - g;
            st.down_ms += b - u;
        }
        double busy = 0;
        for (const auto& [a, b] : passes) busy += b - a;
        auto add_copy = [&](const Impl::Span& s, bool before_fetch_counts) {
            const double a = at(s.a), b = at(s.b);
            st.copy_ms += b - a;
            for (const auto& [pa, pb] : passes) st.overlap_ms += overlap(a, b, pa, pb);
            if (before_fetch_counts) st.overlap_ms += overlap(a, b, a, fetched);
        };
        for (const Impl::Span& s : lm.copies) add_copy(s, false);
        if (lm.prefetch.a) add_copy(lm.prefetch, true);
        st.copy_bytes += lm.copy_bytes;
        st.compute_ms += busy;
        st.plan_gap_ms += std::max(0.0, (passes.empty() ? start : passes.front().first) - fetched);
        if (!passes.empty()) st.stall_ms += std::max(0.0, passes.back().second - passes.front().first - busy);
        st.passes += (int64_t) passes.size();
        ++st.calls;
    }
    I.marks.clear();
}

void ExpertEngine::begin_request(HotPolicy p, bool reset_key, bool resume_key, bool deterministic) {
    Impl& I = *impl_;
    if (I.dry) failf("begin_request on a dry ExpertEngine");
    I.policy = p;
    I.det = deterministic;
    I.split_hot_slab = deterministic;
    const HotRequestStart st = hot_request_start(deterministic, reset_key, resume_key);
    const bool reset = st.reset;
    if (I.hot_on) {
        I.finish_batch(true, false);
        if (reset) {
            const auto t0 = std::chrono::steady_clock::now();
            while (I.start_batch(true)) {
                I.finish_batch(true, true);
                if (I.cfg.hot.check) {
                    I.upload_rows();
                    I.check_residency();
                }
            }
            times_.reset_ms += ms_since(t0);
        }
        // a reset starts the request from the hot set's slab and from zero counts, as a fresh one
        for (HotLayerPlan& hp : I.plans) hp.begin_request(p, st.resume);
        I.upload_rows();
        I.published.clear();
        if (I.cfg.hot.check) I.check_residency();
    }
    if (I.trace) {
        std::fprintf(I.trace, "R %lld %s reset=%d resume=%d\n", (long long) ++I.trace_request, hot_policy_name(p),
                     reset ? 1 : 0, st.resume ? 1 : 0);
        I.trace_begun = true;
    }
}

void ExpertEngine::end_request() {
    Impl& I = *impl_;
    if (I.dry) failf("end_request on a dry ExpertEngine");
    I.poll_restore(true);   // the last layered reset's numbers kept in the report its request printed
    idle_reset_ = IdleReset{};
    if (!I.hot_on || !hot_request_end_resets(I.det)) return;
    // begin_request's reset: the rounds of the spare slots back to the hot set's, waited for here, into idle_reset_'s
    // round_* fields; the counters the reset adds to are restored, so the next request's times start from its own
    auto rounds = [&]() {
        const auto t1 = std::chrono::steady_clock::now();
        const ExpertTimes t0 = times_;
        while (I.start_batch(true)) {
            I.finish_batch(true, true);
            if (I.cfg.hot.check) {
                I.upload_rows();
                I.check_residency();
            }
        }
        I.upload_rows();
        idle_reset_.round_swaps = times_.reset_swaps - t0.reset_swaps;
        idle_reset_.round_bytes = times_.reset_bytes - t0.reset_bytes;
        idle_reset_.round_downs = times_.reset_downs - t0.reset_downs;
        idle_reset_.round_down_bytes = times_.reset_down_bytes - t0.reset_down_bytes;
        idle_reset_.round_ms = ms_since(t1);
        times_.reset_swaps = t0.reset_swaps;
        times_.reset_bytes = t0.reset_bytes;
        times_.reset_downs = t0.reset_downs;
        times_.reset_down_bytes = t0.reset_down_bytes;
    };
    const auto t0 = std::chrono::steady_clock::now();
    I.finish_batch(true, false);
    if (I.layered) {
        // in place, layer by layer, not waited for: the next request waits per layer; partial host homes:
        // then the layers whose victims may need a copy down (partial host homes) in rounds, waited for here
        idle_reset_.layered = true;
        const int64_t in_place = I.restore_layers(&idle_reset_.bytes, &idle_reset_.round_layers);
        idle_reset_.queue_ms = ms_since(t0);
        I.published.clear();
        if (idle_reset_.round_layers) rounds();
        idle_reset_.swaps = in_place + idle_reset_.round_swaps;
        idle_reset_.bytes += idle_reset_.round_bytes;
        if (I.cfg.hot.check) I.check_residency();
        idle_reset_.ms = ms_since(t0);
        if (!in_place) I.poll_restore(true);   // nothing copied layer by layer: done at once
        return;
    }
    rounds();
    I.published.clear();
    if (I.cfg.hot.check) I.check_residency();
    idle_reset_.swaps = idle_reset_.round_swaps;
    idle_reset_.bytes = idle_reset_.round_bytes;
    idle_reset_.ms = ms_since(t0);
}

void ExpertEngine::prompt_done() {
    Impl& I = *impl_;
    if (!I.hot_on) return;
    if (I.batch_open) failf("prompt_done with swaps in flight (a prefill drains them)");
    for (HotLayerPlan& hp : I.plans) hp.prompt_done(I.cfg.hot);
}

void ExpertEngine::merge_prompt() {
    Impl& I = *impl_;
    if (!I.hot_on) return;
    const double mass = (double) I.d.n_expert_used / (1.0 - I.decay);
    for (HotLayerPlan& hp : I.plans) hp.merge_prompt(mass);
}

void ExpertEngine::decode_started() {
    Impl& I = *impl_;
    if (!I.hot_on || I.batch_open) return;
    if (I.start_batch(false)) ++times_.swap_batches;
}

void ExpertEngine::decode_boundary(bool batched) {
    Impl& I = *impl_;
    if (!I.hot_on) return;
    const bool check = I.cfg.hot.check;
    I.published.clear();
    I.poll_restore(false);
    if (I.batch_open && !I.finish_batch(false, false)) {
        // still copying: the deterministic mode (and --hot-check) waits for it here, so the swaps planned at this
        // token are published after it whatever the copy streams' speed; the mode off leaves it open for a later one,
        // and so does a batched step (its rows are not reproducible, the wait would only cost)
        if (hot_boundary_waits(I.det && !batched, check)) {
            const auto t0 = std::chrono::steady_clock::now();
            I.finish_batch(true, false);
            ++times_.swap_waits;
            times_.swap_wait_ms += ms_since(t0);
        } else {
            ++times_.swap_deferred;
        }
    }
    I.upload_rows();
    if (check) I.check_residency();
}

void ExpertEngine::set_hot(const HotConfig& c) {
    Impl& I = *impl_;
    if (c.spare != I.cfg.hot.spare) failf("set_hot: %d spare slots, init made %d", c.spare, I.cfg.hot.spare);
    if (c.host_homes != I.cfg.hot.host_homes)
        failf("set_hot: host homes %s, init made them %s", c.host_homes ? "on" : "off", I.cfg.hot.host_homes ? "on" : "off");
    hot_config_check(c);
    if (c.copy_only && I.hot_on && !I.dry) {
        std::string err;
        if (!I.make_sinks(err)) failf("set_hot (copy only): %s", err.c_str());
    }
    I.cfg.hot = c;
    I.decay = std::pow(2.0, -1.0 / c.half_life);
}

void ExpertEngine::set_trace(bool on) { impl_->trace_on = on; }

const HotConfig& ExpertEngine::hot() const { return impl_->cfg.hot; }
bool ExpertEngine::hot_active() const { return impl_ && impl_->hot_on; }
size_t ExpertEngine::host_home_bytes() const { return impl_ ? impl_->host_home_bytes : 0; }
int64_t ExpertEngine::host_homes_held() const { return impl_ ? impl_->n_homed : 0; }
int64_t ExpertEngine::vram_experts() const { return impl_ ? impl_->n_vram : 0; }

void ExpertEngine::wait_restore_all() {
    Impl& I = *impl_;
    if (!I.restore.open() || I.poll_restore(false)) return;
    // the layers not waited for yet, at once: their first one's events time the wait for the last layer's event
    int first = -1;
    for (int l = 0; l < I.restore.n_layer(); ++l)
        if (I.restore.wait_before(l) && first < 0) first = l;
    if (first < 0) return;
    ck(cudaEventRecord(I.rwait_a[(size_t) first], I.stream), "cudaEventRecord (the token's wait)");
    ck(cudaStreamWaitEvent(I.stream, I.restore_ev.back(), 0), "cudaStreamWaitEvent (the restore's last layer)");
    ck(cudaEventRecord(I.rwait_b[(size_t) first], I.stream), "cudaEventRecord (the token's wait)");
    I.rwait_timed[(size_t) first] = 1;
}
HotPolicy ExpertEngine::request_policy() const { return impl_->policy; }
bool ExpertEngine::request_deterministic() const { return impl_->det; }

void ExpertEngine::prompt_deterministic(bool on) { impl_->split_hot_slab = on; }

void ExpertEngine::set_pool_spin_us(int us) { impl_->cpu->set_spin_us(us); }
CpuPoolTimes ExpertEngine::cpu_pool_times() const { return impl_->cpu ? impl_->cpu->pool_times() : CpuPoolTimes{}; }
int ExpertEngine::cpu_threads() const { return impl_ && impl_->cpu ? impl_->cpu->threads() : 0; }
size_t ExpertEngine::ring_bytes() const { return impl_ ? impl_->ring_bytes : 0; }
PrefillSplitRates ExpertEngine::split_rates() const { return impl_ ? impl_->rates : PrefillSplitRates{}; }
bool ExpertEngine::split_call(int64_t T) const { return impl_ && T <= impl_->split_T; }

void ExpertEngine::reset_times() {
    times_ = ExpertTimes{};
    layer_stats_.assign(layer_stats_.size(), ExpertLayerStats{});
    layer_pairs_.assign(layer_pairs_.size(), 0);
    layer_hits_.assign(layer_hits_.size(), 0);
    if (impl_ && impl_->cpu) impl_->cpu->reset_pool_times();
}

void ExpertEngine::dry_plan(int64_t T, std::FILE* out) const {
    const Impl& I = *impl_;
    const DenseDims& d = I.d;
    const int K = d.n_expert_used;
    const int64_t NE = d.n_expert;
    dsv41::PrefillPlan plan;
    std::vector<int32_t> eop((size_t) (T * K)), hit_row;
    // the dry run cannot tell a pinned mapping: every expert outside the slab staged, the worst case for the passes
    std::vector<uint8_t> staged((size_t) NE, 0);
    // pair (t, s) to expert (t K + s) mod NE: every expert reached, a token's slots distinct (K <= NE)
    for (int64_t p = 0; p < T * K; ++p) eop[(size_t) p] = (int32_t) (p % NE);
    int64_t passes = 0, slab_pairs = 0, ring = 0, staged_n = 0, rows = 0, hits = 0, misses = 0, tiles = 0;
    std::vector<int64_t> count;
    for (size_t l = 0; l < tiers_.size(); ++l) {
        const ExpertLayerTiers& t = tiers_[l];
        for (int64_t e = 0; e < NE; ++e) staged[(size_t) e] = t.slot[(size_t) e] < 0;
        dsv41::plan_prefill(eop, NE, &t.slot, staged, I.cfg.ring_set, I.cfg.ring_sets, I.cfg.stage_sets, plan, hit_row,
                            I.max_pass);
        regroup_slab(plan, I.max_pass);
        int64_t layer_tiles = 0;
        for (const dsv41::PrefillPass& p : plan.passes) layer_tiles += pass_tiles(p, count);
        if (layer_tiles > tile_capacity(T * K, NE, I.max_pass))
            failf("dry run: layer %zu's prefill takes %lld tiles, the buffers hold %lld", l, (long long) layer_tiles,
                  (long long) tile_capacity(T * K, NE, I.max_pass));
        tiles += layer_tiles;
        if (plan.n_rows != T * K) failf("dry run: layer %zu's prefill plan has %lld rows for %lld pairs", l, (long long) plan.n_rows, (long long) (T * K));
        for (const dsv41::PrefillPass& p : plan.passes) {
            if ((int64_t) p.pairs.size() > I.max_pass) failf("dry run: a pass of %zu pairs, the buffers hold %lld", p.pairs.size(), (long long) I.max_pass);
            if (p.set < 0) slab_pairs += (int64_t) p.pairs.size();
        }
        passes += (int64_t) plan.passes.size();
        ring += plan.ring_experts;
        staged_n += plan.staged_experts;
        rows += plan.n_rows;
        for (int s = 0; s < K; ++s) (t.slot[(size_t) eop[(size_t) s]] >= 0 ? hits : misses) += 1;
    }
    const bool pf = I.n_pf > 0 && T * K >= kPrefetchPairsPerExpert * NE;
    int64_t n_rings = 0, n_slab = 0;
    for (size_t l = 0; l < tiers_.size(); ++l) {
        n_rings += I.rings[l];
        n_slab += I.has_slab[l];
    }
    std::fprintf(out, "dry run: experts, a %lld-token prefill chunk on %zu layers: %lld passes (%lld grouped GEMM tiles of up to "
                 "%d rows), %lld of %lld pairs on the slab, %lld experts through the ring (%d sets of %d; %lld of them staged "
                 "through %d pinned sets when the mapping is not pinned), %s; a decode step: %lld hits on the GPU (4 grouped "
                 "launches in each of %lld layers with a slab), %lld misses on the CPU (a doorbell in %lld of %zu layers)\n",
                 (long long) T, tiers_.size(), (long long) passes, (long long) tiles, kExpertTileRows, (long long) slab_pairs,
                 (long long) rows, (long long) ring,
                 I.cfg.ring_sets, I.cfg.ring_set, (long long) staged_n, I.cfg.stage_sets,
                 pf && T > I.split_T ? "the next layer's prefetch on (pinned sources only)"
                    : !I.n_pf ? "nothing to prefetch" : pf ? "no prefetch at this chunk (a split chunk)"
                    : "no prefetch at this chunk (T x K < 4 n_expert)",
                 (long long) hits, (long long) n_slab, (long long) misses, (long long) n_rings, tiers_.size());
    if (T <= I.split_T)
        std::fprintf(out, "dry run: experts, the %lld-token chunk takes the prefill split: the experts above through the ring "
                     "are its upper bound, the CPU takes %s at run time\n", (long long) T,
                     I.cfg.prefill_split == PrefillSplit::Cpu ? "every one of them" : "those with few pairs by the measured rates");
}

void ExpertEngine::dry_restore(std::FILE* out) const {
    const Impl& I = *impl_;
    const int n_layer = (int) tiers_.size();
    const bool homes = I.cfg.hot.host_homes;
    // the copies' bytes of one expert of layer l
    auto bytes_of = [&](int l) {
        const ExpertLayerHost& L = I.m->experts()[(size_t) l];
        return (double) (L.bytes[0] + L.bytes[1] + L.bytes[2]);
    };
    HotRestore order;
    order.begin(n_layer);
    // the layers with every slab expert homed restore in place, the others in rounds (partial or none)
    int64_t layers = 0, displaced = 0, rounds = 0, in_place = 0, swaps_in = 0, swaps_rounds = 0, downs = 0;
    double up_in = 0, up_rounds = 0, down = 0;
    HotConfig c = I.cfg.hot;
    for (int l = 0; l < n_layer; ++l) {
        if (!I.spare_slots[(size_t) l]) {
            order.record(l);
            continue;
        }
        HotLayerPlan p;
        HotLayerPlan::Layout lay = I.layout_of(l, false);
        const bool full = homes && lay.homed.size() == lay.vram.size();
        if (!full && !lay.warm.empty()) lay.spare_homes = lay.spare;   // init's spare homes, as it makes them
        p.init(lay);
        // displaced as far as the promotable experts allow: the prompt policy's target is every promotable expert
        // outside the slab the prompt counted (once each), the hot set's experts after them
        p.begin_request(HotPolicy::Prompt, false);
        std::vector<int32_t> ids;
        for (int32_t e = 0; e < p.n_expert(); ++e)
            if (p.slot_of(e) < 0 && p.source(e) == HotSource::Home) ids.push_back(e);
        p.count_prompt(ids.data(), (int64_t) ids.size());
        p.prompt_done(c);
        std::vector<int32_t> home0((size_t) p.n_expert());
        for (int32_t e = 0; e < p.n_expert(); ++e) home0[(size_t) e] = p.home_of(e);
        std::vector<HotSwap> sw;
        for (;;) {
            sw.clear();
            p.plan(c, sw);
            if (sw.empty()) break;
            for (const HotSwap& x : sw) {
                // host homes in full drop every victim; otherwise a homed victim is dropped, any other copied down
                if (full ? x.to_home >= 0 : (x.to_home >= 0) == p.has_fixed_home(x.out))
                    failf("dry run: layer %d: expert %d leaves the slab %s", l, x.out,
                          x.to_home >= 0 ? "copied down with a host home of its own" : "dropped with no host copy");
                p.publish(x);
            }
        }
        for (int32_t e : lay.vram) displaced += p.slot_of(e) < 0;
        if (full) {
            // the engine's order: this layer's copies, then its event; the next request's layer l waits for it alone
            sw.clear();
            p.restore_in_place(sw);
            for (const HotSwap& x : sw) {
                if (x.to_home >= 0 || x.from_home != home0[(size_t) x.in])
                    failf("dry run: layer %d: restoring expert %d did not copy it up from its own home", l, x.in);
                up_in += bytes_of(l);
            }
            order.record(l);
            if (!order.wait_before(l) || order.wait_before(l))
                failf("dry run: layer %d: its computation does not wait for its own restore exactly once", l);
            swaps_in += (int64_t) sw.size();
            ++in_place;
        } else {
            int r = 0;
            while (!p.is_static()) {
                sw.clear();
                p.plan_static(c.spare, sw);
                if (sw.empty()) failf("dry run: layer %d: the reset in rounds stalled", l);
                for (const HotSwap& x : sw) {
                    up_rounds += bytes_of(l);
                    down += x.to_home >= 0 ? bytes_of(l) : 0;
                    downs += x.to_home >= 0;
                    p.publish(x);
                }
                swaps_rounds += (int64_t) sw.size();
                ++r;
            }
            rounds = std::max<int64_t>(rounds, r);
            order.record(l);
        }
        if (!p.is_static()) failf("dry run: layer %d: not at the hot set's slab after the reset", l);
        if (const std::string why = p.check(); !why.empty()) failf("dry run: layer %d after the reset: %s", l, why.c_str());
        // a dropped victim's host home is still its own: every expert's home as init placed it (partial: the homed ones')
        for (int32_t e = 0; e < p.n_expert() && homes; ++e)
            if ((full || p.has_fixed_home(e)) && p.home_of(e) != home0[(size_t) e])
                failf("dry run: layer %d: expert %d's home moved", l, e);
        ++layers;
    }
    order.done();
    // the copies' time at the RTX PRO 6000 pod's rates: 26.12 GB/s up alone (the
    // engine's init there), ~20.4 GB/s each way at once
    const double s_in = up_in / 26.12e9, s_rounds = up_rounds / 20.4e9;
    char flag[48];
    if (I.cfg.hot.host_homes_max < 0) std::snprintf(flag, sizeof flag, "--hot-homes all");
    else std::snprintf(flag, sizeof flag, "--hot-homes %lld", (long long) I.cfg.hot.host_homes_max);
    if (!homes)
        std::fprintf(out, "dry run: hot set reset (--hot-homes spare): %lld layers displaced by %lld experts, reset in at most "
                     "%lld rounds of %d per layer: %lld swaps, %.1f MiB up and %.1f down, every layer back at the hot set's "
                     "slab; the next request waits for all of it; about %.2f s at 20.4 GB/s each way\n", (long long) layers,
                     (long long) displaced, (long long) rounds, c.spare, (long long) swaps_rounds, up_rounds / kMiB,
                     down / kMiB, s_rounds);
    else if (in_place == layers)
        std::fprintf(out, "dry run: hot set reset (%s): %lld layers displaced by %lld experts, restored in place "
                     "layer by layer: %lld swaps, %.1f MiB up and 0.0 down, every layer back at the hot set's slab with every "
                     "home its own; %d events in layer order, the next request's layer l waiting for layer l's alone; about "
                     "%.2f s at 26.12 GB/s up\n", flag, (long long) layers, (long long) displaced, (long long) swaps_in,
                     up_in / kMiB, n_layer, s_in);
    else
        std::fprintf(out, "dry run: hot set reset (%s, partial host homes: %lld of %lld slab experts homed, %.1f%%): %lld "
                     "layers displaced by %lld experts; %lld layers with every slab expert homed restored in place layer by "
                     "layer: %lld swaps, %.1f MiB up and 0.0 down (about %.2f s at 26.12 GB/s up), the next request's layer "
                     "l waiting for layer l's alone; the other %lld in at most %lld rounds of %d per layer, waited for before "
                     "the next request: %lld swaps, %.1f MiB up and %.1f down, %lld victims copied down (about %.2f s at "
                     "20.4 GB/s each way); every layer back at the hot set's slab, every homed expert's home its own\n", flag,
                     (long long) I.n_homed, (long long) I.n_vram, I.n_vram ? 100.0 * (double) I.n_homed / (double) I.n_vram : 0.0,
                     (long long) layers, (long long) displaced, (long long) in_place, (long long) swaps_in, up_in / kMiB,
                     s_in, (long long) (layers - in_place), (long long) rounds, c.spare, (long long) swaps_rounds,
                     up_rounds / kMiB, down / kMiB, (long long) downs, s_rounds);
}

void ExpertEngine::print_summary(std::FILE* out) const {
    const Impl& I = *impl_;
    std::fprintf(out, "experts: VRAM %lld experts (%.3f GiB slab, --vram-gb %g), warm %lld (%.3f GiB, --ram-gb %g%s), cold %lld "
                 "(the model's mapping); hot set %s\n",
                 (long long) I.n_vram, (double) I.slab_bytes / kGiB, I.cfg.vram_gib, (long long) I.n_warm,
                 (double) I.warm_bytes / kGiB, I.cfg.ram_gib, I.residency ? ", pinned by strata-moe's residency" : "",
                 (long long) I.n_cold, I.has_hot ? I.cfg.hot_set.c_str() : "none");
    int64_t pinned_layers = 0;
    for (uint8_t p : I.mapping_pinned) pinned_layers += p;
    char mapping[96];
    if (I.dry) std::snprintf(mapping, sizeof mapping, "if a source is not pinned; not checked in a dry run");
    else std::snprintf(mapping, sizeof mapping, "mapping pinned in %lld of %zu layers", (long long) pinned_layers, I.mapping_pinned.size());
    std::fprintf(out, "experts: prefill ring %.1f MiB (%d sets of %d) and prefetch region %.1f MiB (%d experts) on a copy "
                 "stream, pinned staging %.1f MiB (%d sets; %s), call buffers %.1f MiB on the device and %.1f MiB pinned, "
                 "%d CPU threads (spin %d us)\n",
                 (double) I.ring_bytes / kMiB, I.cfg.ring_sets, I.cfg.ring_set, (double) I.pf_bytes / kMiB, I.n_pf,
                 (double) I.stage_bytes / kMiB, I.cfg.stage_sets, mapping, (double) I.buffer_bytes / kMiB,
                 (double) I.pinned_bytes / kMiB, I.cpu ? I.cpu->threads() : 0, I.cfg.pool_spin_us);
    int64_t spare_layers = 0;
    for (int x : I.spare_slots) spare_layers += x > 0;
    if (!spare_layers) {
        std::fprintf(out, "experts: hot set static (%s)\n", I.cfg.hot.spare == 0 ? "--hot-spare 0"
                     : "no layer has experts both in and out of the slab");
    } else {
        char promo[96];
        if (I.dry) std::snprintf(promo, sizeof promo, "the warm tier's experts promotable (pinned mappings not known in a dry run)");
        else std::snprintf(promo, sizeof promo, "%lld experts promotable (pinned)", (long long) I.n_promotable);
        char copy_only[96] = "";
        if (I.cfg.hot.copy_only)
            std::snprintf(copy_only, sizeof copy_only, "; copy only, nothing published (%.1f MiB of scratch homes)",
                          (double) I.sink_bytes / kMiB);
        std::fprintf(out, "experts: hot set policy %s by default (a request may name another), %d spare slots per layer in "
                     "%lld layers (%.1f MiB of the slab), %.1f MiB of spare pinned homes, at most %d swaps per layer per "
                     "token, half-life %g decode tokens, margin %g x + %g; %s%s%s%s\n",
                     hot_policy_name(I.cfg.hot.policy), I.cfg.hot.spare, (long long) spare_layers,
                     (double) I.spare_slab_bytes / kMiB, (double) I.spare_home_bytes / kMiB, I.cfg.hot.max_swaps,
                     I.cfg.hot.half_life, I.cfg.hot.margin_mul, I.cfg.hot.margin_add, promo,
                     I.cfg.hot.check ? "; --hot-check" : "", I.cfg.route_trace.empty() ? "" : "; route trace on",
                     copy_only);
        // which path the swaps and the deterministic reset take, and why
        int64_t pinned_swap_layers = 0;
        for (size_t l = 0; l < I.spare_slots.size() && !I.dry; ++l) pinned_swap_layers += I.spare_slots[l] && I.mapping_pinned[l];
        // the layers with every slab expert homed, the one the limit cut, and the homed experts' share
        int64_t full_layers = 0, swap_layers = 0, cut_layer = -1, cut_n = 0, cut_of = 0, short_layers = 0;
        for (size_t l = 0; l < I.spare_slots.size(); ++l) {
            if (!I.spare_slots[l]) continue;
            ++swap_layers;
            int64_t nv = 0;
            for (int32_t e : tiers_[l].vram) nv += e >= 0;
            const int64_t h = l < I.homed.size() ? (int64_t) I.homed[l].size() : 0;
            full_layers += h > 0 && h == nv;
            // a layer short of homes for its slab (a pinned mapping is its experts' host copy)
            short_layers += h < nv && (I.dry || !I.mapping_pinned[l]);
            if (h > 0 && h < nv) cut_layer = (int64_t) l, cut_n = h, cut_of = nv;
        }
        const bool partial = I.cfg.hot.host_homes && short_layers > 0;
        if (partial)
            std::fprintf(out, "experts: hot set homes: partial host homes (--hot-homes %lld): %lld of the slab's %lld VRAM "
                         "experts keep a permanent pinned home (%.1f%%), %.1f MiB%s: every slab expert in %lld of the %lld "
                         "layers that swap (the first in layer order)%s; there a swap copies its expert up only and the "
                         "deterministic reset restores in place layer by layer on its own stream, the next request waiting "
                         "per layer; in the others a homed victim is dropped, any other copied down into a spare home, and "
                         "the reset runs in rounds of the spare slots, waited for before the next request\n",
                         (long long) I.cfg.hot.host_homes_max, (long long) I.n_homed, (long long) I.n_vram,
                         I.n_vram ? 100.0 * (double) I.n_homed / (double) I.n_vram : 0.0, (double) I.host_home_bytes / kMiB,
                         I.dry ? " (dry run: as if no mapping were pinned)" : "", (long long) full_layers,
                         (long long) swap_layers,
                         cut_layer < 0 ? "" : (", layer " + std::to_string(cut_layer) + "'s last " + std::to_string(cut_n) +
                                               " of its " + std::to_string(cut_of) + " by the hot set's order").c_str());
        else if (I.cfg.hot.host_homes)
            std::fprintf(out, "experts: hot set homes: host homes (--hot-homes all): every slab expert keeps a permanent "
                         "pinned home, %.1f MiB%s%s; a swap copies its expert up only and drops its victim (no copy down, "
                         "no spare homes); the deterministic reset restores in place layer by layer on its own stream, "
                         "the next request waiting per layer\n", (double) I.host_home_bytes / kMiB,
                         I.dry ? " (dry run: as if no mapping were pinned)" : "",
                         pinned_swap_layers ? "; the layers with a pinned mapping use it as their homes" : "");
        else
            // with the deterministic mode on, this path's idle reset is the slow one
            std::fprintf(out, "experts: hot set homes: two-way (--hot-homes spare): a swap copies its victim down into a "
                         "spare home%s; the deterministic reset in rounds of the spare slots, the next request waiting for "
                         "all of it%s\n", pinned_swap_layers ? " (a pinned mapping's victims go back to it)" : "",
                         I.cfg.hot.deterministic
                             ? "; the deterministic mode is on (--deterministic on), so this idle reset is the slow one: "
                               "the next request waited 1.67-1.91 s for it on the RTX PRO 6000 ("
                               "2026-10-14, 2885-3311 swaps after each 256-token reply); tools/dsv41/pod/e07_gate.py's "
                               "gate.json hot_homes is the --hot-homes value this host's RAM holds"
                             : "");
    }
    if (!I.split_T) {
        std::fprintf(out, "experts: prefill split off (%s)\n", I.cfg.prefill_split == PrefillSplit::Off ? "--prefill-split off"
                     : I.cfg.split_max_t == 0 ? "--split-max-t 0" : "every expert in the slab");
    } else if (I.cfg.prefill_split == PrefillSplit::Cpu) {
        std::fprintf(out, "experts: prefill split for chunks of up to %lld tokens: every expert outside the slab on the CPU "
                     "(--prefill-split cpu)\n", (long long) I.split_T);
    } else if (I.dry) {
        std::fprintf(out, "experts: prefill split for chunks of up to %lld tokens: the ring and the CPU by the PCIe and CPU "
                     "rates measured at init (no CUDA call in a dry run)\n", (long long) I.split_T);
    } else {
        std::fprintf(out, "experts: prefill split for chunks of up to %lld tokens, by the rates measured at init: PCIe %.2f "
                     "GB/s into the ring, CPU %.2f GB/s on one token per expert + %.4f ms per further pair\n",
                     (long long) I.split_T, I.rates.pcie_gbs, I.rates.cpu_gbs, I.rates.cpu_pair_ms);
    }
}

}  // namespace strata::dsv41::engine
