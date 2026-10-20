// include/strata/dsv41/engine/experts.hpp - the routed experts of the engine (SPEC.md section 7): per layer the
// experts of a hot set in a VRAM slab, a warm list in pinned host RAM, every other expert cold where the model maps it,
// and for a token chunk the sum over its slots, in the fork's slot order, of each slot's routing weight times its
// expert's SwiGLU-clamped FFN, plus the shared expert's output.
//
// Plain pointers in front of src/dsv41 (which is tied to ggml):
// the hot-set file of moe_hot_set.h (the VRAM experts and the warm lists of each layer), the warm
// tier of moe_residency.h (its pinned buffer filled with O_DIRECT reads of the GGUF, bound through ExpertTensors), the
// CPU kernel of cpu_kernel.h (strata-moe's AVX2 rows against Q8_K activations, ggml-cpu's vec_dot to the bit:
// the oracle's numerics, CpuNumerics::Fork, beside the engine's own GPU numerics on the CPU), the
// prefill plan and schedule of moe_prefill.h, and the activation counts of moe_counts.h. No ggml graph, no ggml-cuda:
// decode's GPU experts run on qmatmul.hpp's MMVQ, every hit of a call in one launch per matrix; a prefill
// pass runs all of its experts in two grouped tensor-core launches (src/dsv41/engine/experts_gemm.hpp).
//
//   decode (prefill false, T <= kExpertSplitMaxT): the hybrid's measured split (28 tok/s decode)
//       with no host round trip on the GPU's side (Strata's doorbell protocol: core/layer.hpp Doorbell,
//       session.hpp's token graph). The router's ids stay on the device: one kernel copies them and the input rows into
//       mapped pinned memory and rings a doorbell (a counter in mapped memory) for the host. Hits, the pairs whose
//       expert is in the slab, run on the GPU in four launches whatever the routing (qmatmul.hpp qmatmul_experts:
//       Strata's native_expert_grouped pattern, the layer's residency row on the device, a block row per pair, gate,
//       up and the clamp fused, then down; qmatmul's MMVQ values bit for bit). Meanwhile the host, polling the ring,
//       hands the misses to the CPU pool (already spinning with pool_spin_us) from the warm tier or the mapping.
//       With CpuNumerics::Gpu the CPU computes them in the hits' own arithmetic, the input rows quantized
//       to q8_1 as quantize_q8_1_kernel, gate and up in mmvq_kernel's order per row, the clamp of expert_act.hpp, h
//       quantized to q8_1, down; every value the GPU's hit of the same expert bit for bit, so a token's output does
//       not depend on which of its experts are in the slab (CpuNumerics::Fork: the fork's CPU experts, Q8_K and ggml-cpu's
//       vec_dot, whose last bits differed from the hits': DETERMINISM.md row b). Their rows and each pair's row go to
//       mapped memory, then the host writes the ring's value to a flag; a one-thread kernel waits for it on the
//       device right before the combine, which reads the rows. No cudaStreamSynchronize, and the queued launches are
//       the same at every token, so a token's layers capture into one CUDA graph (forward.hpp). A layer whose every
//       expert is in the slab skips the doorbell (unless counts are recorded: they need the ids on the host).
//   prefill (prefill true): every pair on the GPU (moe_prefill.h's plan and schedule, as strata-moe's
//       moe_gpu.cpp runs them): the slab's experts where they are, every other expert copied once into a ring of
//       device slots on a copy stream of its own, the passes on the engine's stream, the two meeting through events
//       (a set's copy waits for the last pass that read it, a pass for its set's copy). A source in pinned host memory
//       (the warm tier, or a mapping the caller pinned) is copied straight from there; any other is staged first into
//       pinned staging by the CPU pool, a staging set refilled once the copies out of it are done (a host wait per
//       pass). The slab's passes come first, so the copy stream fills the ring while they run; their pairs are
//       regrouped into whole experts (at most max(4096, max_T) rows per pass, an expert cut only above that), so each
//       slab expert's weights are read once per call; a ring pass takes up to ring_set experts and as many rows
//       (plan_prefill's cap at the engine's, not the hook's 4096). Every pass's pairs, tiles and output rows are planned on the host
//       and uploaded once per call before the first pass, so the passes queue kernels only (no small upload of theirs
//       waits behind the ring's copies on the copy engine). A pass: its input rows gathered by expert and rounded to f16,
//       one launch for gate, up and the clamp of every expert's rows (h in f16), one for down, in tiles of 128 rows
//       of one expert, the dequant path's numerics (qmatmul.hpp, QPrefill::DequantF16) for every row count.
//       Prefetch: once a layer's passes are queued, the next layer's first non-resident experts (the hot set's warm
//       order, then the cold ones by id; pinned sources only) are copied into a prefetch region while the next
//       layer's attention and dense ops run, and that layer's passes read them in place like slab experts; only for
//       chunks where every expert is likely routed to (T K >= 4 n_expert: under uniform routing an expert misses all
//       pairs with probability e^-4, 1.8%), since an unused prefetched expert is a wasted copy, and never for a split
//       chunk (below), whose PCIe time goes to the experts its split picks.
//   prefill split (T <= split_max_t, PrefillSplit::Measured): a short read (a resumed turn's new tokens)
//       touches most of a layer's experts with a few pairs each, so copying every one of them made the turn 2 (513
//       tokens) copy-bound: 11.7 GiB through the ring over 8 layers, 720 of its 777 ms. Strata's
//       hit/miss split instead (the hybrid's measured method): the slab's pairs on the GPU as above; of the other
//       experts the ones with many pairs go through the ring and the ones with few pairs run on the CPU pool from
//       where they are (the pinned warm tier or the mapping, the decode misses' kernel and numerics, the call's input
//       rows fetched with its routing), all at once. The split is chosen per layer from rates measured at init
//       (PrefillSplitRates: the copy stream's GB/s from host to the ring, the pool's GB/s on one token per expert and
//       its cost per extra pair) so that the ring's copies and the CPU's experts end together: the experts by pair
//       count, fewest first, go to the CPU while that lowers max(ring, CPU) + the CPU rows' upload. The CPU's rows go
//       to the device on the copy stream after the ring's copies, into the call's output rows past the GPU's, and the
//       combine waits for them.
//   combine: out[t] = (((y_0 w_0 + y_1 w_1) + ...) + y_{K-1} w_{K-1}) + shared[t], each product and sum rounded as the
//       fork's ggml_mul, its view adds (llama-graph.cpp build_moe_ffn) and ffn_out's add, whichever tier made y.
//   hot set (hot_policy.hpp): each request's policy moves the slab's contents
//       while it decodes. The slab has ExpertConfig::hot.spare free slots per layer beyond the hot set's (a layer
//       with experts in and out of it), and, when a layer's mapping is not pinned, as many spare pinned homes beside
//       its warm tier's slots. After every decode token (decode_boundary, between two tokens, after the forward's
//       wait for the stream) the swaps whose copies are done are published (the host's tier map and the residency
//       rows, the device's rows uploaded on the engine's stream from mapped memory, read by the next token's graph);
//       once a token's device work is queued (decode_started: right after the token graph's launch) each layer's
//       next swaps are planned and their copies queued (the promoted expert's bytes from its pinned home or mapping
//       into a free slot on the copy stream, the victim's from its slot into a free home on a stream of its own: the
//       device's other copy engine), one batch at a time: a batch still copying at a boundary (an event query, never
//       a wait) is published at a later one, and no other starts meanwhile; the deterministic mode (the default,
//       hot_policy.hpp) waits for it there instead, and starts every request from the hot set's slab. Nothing of a decode token waits for the
//       copy streams: its inputs, logits and residency rows cross through mapped memory (as
//       cudaMemcpyAsync calls they queued behind the batch's copies on the copy engines, a token taking its static
//       time plus the batch's bytes at 29 GB/s). A prefill call, a request's start and its reset to the static slab
//       publish everything first (drain), so a prefill sees a slab that does not move.
//   host homes (HotConfig::host_homes, --hot-homes all):
//       every expert of the hot set's slab gets a permanent home in pinned host RAM at init (in each layer whose
//       mapping is not pinned; the slab is filled from them), so a swap copies its promoted expert up and drops its
//       victim (no down stream copy, no spare homes), and the deterministic reset (end_request, and a request's start
//       after one in the mode off) restores the slab in place layer by layer on a restore stream of its own, an event
//       after each layer (hot_policy.hpp HotRestore): the planners are the hot set's slab at once, the next request's
//       prefill passes and eager decode hits of layer l wait for layer l's event only, the token graph for the last
//       one before its launch, and nothing waits once a call has seen the last event done. Without it (the default,
//       or the gate found no RAM for the homes) the two-way path and the reset in rounds, each waited for by the host.
//   partial host homes (HotConfig::host_homes_max, --hot-homes N): the first N slab experts of the
//       home order get a home, the rest keep the two-way path (hot_policy.hpp). The home order does not depend on any
//       request: the swapping layers whose mapping is not pinned in layer order, every slab expert of a layer before
//       the next layer's, and the one layer the limit cuts through takes its slab experts from the end of its VRAM order
//       (the hot set's least counted first: the planner's victims among equal counts, plan_target's and plan_lfu's tie
//       rule, so the experts most often evicted at a request's start). Whole layers rather than the most evicted
//       experts across all of them: a layer with any slab expert left without a home recycles its promoted warm
//       experts' homes for that expert's copies down, so its reset needs copies down too and cannot restore in place;
//       spread over every layer the homes would put every layer on the reset in rounds (and the only per-expert data,
//       the route trace of 2026-10-09, is the routing under the A100's placement, not this one's evictions). So a
//       layer with every slab expert homed restores in place layer by layer as above, and every other layer that swaps
//       (the cut one and those after it) is reset in rounds of its spare slots, its victims copied down as before,
//       which end_request runs right after queuing the layered part and waits for, so the next request finds it done.
#pragma once

#include "strata/dsv41/engine/dense.hpp"
#include "strata/dsv41/engine/hot_policy.hpp"
#include "strata/dsv41/engine/model.hpp"
#include "strata/dsv41/engine/qmatmul.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

/// The most tokens of a call on the hit/miss split (decode); prefill calls take any count up to max_T.
constexpr int64_t kExpertSplitMaxT = kMmvqMaxCols;

enum class ExpertTier : uint8_t { Vram, Warm, Cold };

/// Where a split prefill chunk (T <= ExpertConfig::split_max_t) computes the pairs outside the slab.
///   Off       every pair on the GPU, through the ring (the experts test's bit-identical placements)
///   Measured  the ring and the CPU pool by the measured rates (the default)
///   Cpu       every one on the CPU pool (tests: bit for bit the CPU experts')
enum class PrefillSplit : uint8_t { Off, Measured, Cpu };

/// The rates the split is chosen from, measured by ExpertEngine::init (0 before, in a dry run, or with no expert
/// outside the slab): the copy stream's GB/s from an expert's host source into the ring, the CPU pool's GB/s over
/// experts on one token each, and its milliseconds per further pair of an expert.
struct PrefillSplitRates {
    double pcie_gbs = 0, cpu_gbs = 0, cpu_pair_ms = 0;
};

/// The arithmetic of the CPU experts.
///   Gpu   decode's hit arithmetic on the GPU (experts_cpu_mmvq.hpp: q8_1 activations, mmvq.cuh's traits in
///         mmvq_kernel's order, expert_act.hpp's clamp), so an expert's output has the same bits on either tier and
///         a token's do not depend on the placement: the engine's (q2_K and q3_K)
///   Fork  the fork's CPU experts bit for bit (strata-moe's fast kernel: Q8_K activations, ggml-cpu's AVX2 vec_dot,
///         ggml_swiglu_clamp with glibc's expf): the oracle's numerics, for the parity against its op dumps
///         (dsv41_engine_experts_test --cases, dsv41-layer-ref) (q2_K, q3_K, q4_K, q6_K)
enum class CpuNumerics : uint8_t { Gpu, Fork };

struct ExpertConfig {
    std::string hot_set;     ///< a moe_hot_set.h file ("": no expert in VRAM, none warm: every expert cold)
    double vram_gib = 0;     ///< the budget of the VRAM slab, which the hot set's experts must fit (0: no slab)
    double ram_gib = 0;      ///< the budget of the warm tier (0: none; the hot set's warm lists need it); its memory
                             ///< is strata-moe's STRATA_MOE_RAM_MODE (pinned by default with a GPU, moe_residency.h)
    int threads = 0;         ///< CPU threads of the misses (0: the hardware's, at most 16)
    /// prefill: experts per ring set. 48 (strata-moe's STRATA_MOE_PREFILL_GROUP default is 16): the A100 run had
    /// 16 to 20 passes per layer of a 16384-token chunk with sets of 16; a set of 48 takes a layer's ~184 ring experts
    /// (216 outside a 44% slab, 32 of them prefetched) in 4 passes of ~12,300 rows (uniform routing) or ~2,200 (the
    /// real routing's ~45 rows per cold expert), for 1.15 GiB more VRAM in 3 sets (1.72 GiB instead of 0.57), which
    /// the memory table takes from the hot set
    int ring_set = 48;
    int ring_sets = 3;       ///< prefill: ring sets (STRATA_MOE_PREFILL_SETS' default: one computed, one copied, one queued)
    int stage_sets = 2;      ///< prefill: pinned staging sets of ring_set experts for unpinned sources
                             ///< (STRATA_MOE_PREFILL_STAGE's default: one filled while the copy stream reads the other;
                             ///< pinned host RAM, about 1.15 GiB at 48 per set)
    /// prefill: experts of the next layer copied during its attention (0: none). 32, two sets of 16 (410 MB at
    /// 12,810,240 B): the L4 smoke spent 56 ms per layer on a 4096-token chunk's attention, in which PCIe 4 x16's
    /// 31.5 GB/s moves about 137 experts, so 32 are covered at every chunk size; the smoke measures more.
    int prefetch_experts = 32;
    /// the CPU pool's workers and caller spin this long before blocking (CpuExperts). 50 (the
    /// A100 smoke with --counts: wake 1.386 -> 0.814 ms and return 0.310 -> 0.063 ms per token against 0, 10.28 ->
    /// 9.72 ms per token)
    int pool_spin_us = 50;
    std::string counts;      ///< a moe_counts.h file the routed experts are counted into ("": none)
    /// prefill: chunks of at most this many tokens take the split (experts.hpp; 0: none). 2048: the bound
    /// set for a resumed turn's short read, which leaves the 16K chunks on the
    /// ring (CPU prefill discarded there: thousands of pairs per expert); its pinned buffers are its input rows and its
    /// CPU rows, 2048 x (1 + 6) rows of 5120 f32, 294 MB at the real dims
    int64_t split_max_t = 2048;
    PrefillSplit prefill_split = PrefillSplit::Measured;
    HotConfig hot;           ///< the per-request hot set (hot_policy.hpp); hot.spare 0: the slab never moves
    /// --route-trace: every call's routed ids, per layer, written to this file ("": none; format in
    /// tools/dsv41/engine/hot_policy_sim.py): a request's R line, a prefill call's P lines, a decode call's D lines
    std::string route_trace;
    /// dsv41-engine --cpu-numerics: the CPU experts' arithmetic. Gpu
    /// (this struct's default, the tests') keeps a token's bits independent of the placement; Fork is the fork's
    /// arithmetic, with which a batched request's bits follow its neighbours. dsv41-engine sets Fork by default for
    /// single-user speed: the 2026-10-19 A/B (RTX PRO 6000, AMD EPYC 9575F, the single user) 63.34 tok/s in the
    /// fork's numerics against 47.04 in the GPU's
    CpuNumerics cpu_numerics = CpuNumerics::Gpu;
};

/// Wall times of the calls so far (ms, host clocks), by part, and their counts.
struct ExpertTimes {
    double fetch_ms = 0;     ///< prefill: the routing down to the host, including the stream's wait
    double ring_wait_ms = 0; ///< decode: the host's polls of the doorbell, from decode_serve's call to the ring
    double plan_ms = 0;      ///< the split (decode, from the ring's ids) or the prefill plan on the host
    double cpu_ms = 0;       ///< the misses on the CPU (decode)
    double queue_ms = 0;     ///< queuing the GPU work after the plan (prefill: with the staging and its waits, and a
                             ///< split's CPU experts)
    double stage_ms = 0;     ///< prefill: the CPU pool's copies into pinned staging
    double stage_wait_ms = 0;///< prefill: the host's waits for a staging set's copies
    double copy_bytes = 0;   ///< prefill: expert bytes copied into the ring
    double staged_bytes = 0; ///< prefill: of them through pinned staging
    double prefetch_bytes = 0;  ///< prefill: expert bytes copied into the prefetch region
    int64_t calls = 0;
    int64_t host_syncs = 0;  ///< host waits on the device: prefill's routing fetch per call and staging waits
    int64_t doorbells = 0;   ///< decode: rings served (a layer every expert of which is in the slab rings none)
    int64_t hits = 0;        ///< pairs computed from the slab (prefill: and from the prefetch region)
    int64_t misses = 0;      ///< pairs computed elsewhere: on the CPU (decode, a prefill split) or from the ring (prefill)
    int64_t ring_experts = 0;
    int64_t prefetched = 0;  ///< prefill: experts prefetched; of them used by their layer's call
    int64_t prefetch_used = 0;
    int64_t tiles = 0;       ///< prefill: grouped GEMM tiles (up to 128 rows of one expert)
    int64_t cpu_experts = 0; ///< decode: CPU jobs (one per missed expert and call, its rows read once for its tokens)
    int64_t distinct = 0;    ///< decode: the distinct experts of each call's pairs, summed (a batch step's union)
    double cpu_bytes = 0;    ///< decode: the expert bytes those jobs read
    /// prefill: the bytes of the experts outside the slab with a pair in a call, per layer and call, whichever tier
    /// computed them (a prefetched one too): what a split must move or read at the least
    double touched_bytes = 0;
    int64_t touched_experts = 0;
    /// prefill split (experts.hpp): its calls; the experts, pairs and bytes it gave the CPU, the CPU's wall time (the
    /// host runs it while the device works), and the split's own estimates of its ring and CPU times, summed
    int64_t split_calls = 0, split_cpu_experts = 0, split_cpu_pairs = 0;
    double split_cpu_bytes = 0, split_cpu_ms = 0, split_ring_est_ms = 0, split_cpu_est_ms = 0;
    /// the hot set: swaps published, their copies' bytes (the promoted experts into the slab, the victims into
    /// homes) and device ms (events: a batch's start to the later of its two streams' ends), the batches started and
    /// finished, the boundaries at which the open batch was still copying (it stays open), the host's ms queuing the
    /// batches' copies, and a request's reset to the static slab (swaps, bytes, host ms) (HotConfig::copy_only: the
    /// swaps whose copies ran and were dropped unpublished, their bytes and ms counted too)
    int64_t swaps = 0, swap_batches = 0, swap_batches_done = 0, swap_deferred = 0, reset_swaps = 0, swaps_dropped = 0;
    double swap_bytes = 0, swap_copy_ms = 0, swap_queue_ms = 0, reset_bytes = 0, reset_ms = 0;
    /// of those swaps (published, or the reset's), the ones whose victim was copied down into a home
    /// (the two-way path's; none with host homes in full), and their bytes down
    int64_t swap_downs = 0, reset_downs = 0;
    double swap_down_bytes = 0, reset_down_bytes = 0;
    int64_t hot_checks = 0;  ///< --hot-check: boundaries checked
    /// the deterministic mode: the boundaries that found the open batch still copying and waited for it (where
    /// the mode off defers it), and the host's ms in those waits (the mode's decode cost on the swaps' side)
    int64_t swap_waits = 0;
    double swap_wait_ms = 0;
};

/// One layer's prefill expert phase, from CUDA events (ExpertEngine::set_timing), summed over its calls. Times are
/// device intervals: the copies on the copy stream (the ring's and the layer's prefetch), the passes on the engine's
/// stream, from the routing's fetch (the end of the layer's front: attention, dense ops, router) to the last pass.
///   overlap_ms  the copies' time while a pass ran, plus the prefetch's time before the routing's fetch (the
///               engine's stream then runs the layer's front); copy_ms - overlap_ms is copy time nothing hid
///   plan_gap_ms the engine's stream idle from the routing's fetch (from the end of the work queued after it,
///               the shared expert) to the first pass (the host's plan beyond that work, the uploads)
///   stall_ms    the engine's stream idle between the first pass's start and the last's end (waits for copies)
///   gather_ms, gate_up_ms, down_ms  the passes' compute_ms by launch: the rows' gather into f16, gate/up with the
///               clamp, down (experts_gemm.hpp)
struct ExpertLayerStats {
    int64_t calls = 0, passes = 0;
    double copy_bytes = 0, copy_ms = 0, compute_ms = 0, overlap_ms = 0, plan_gap_ms = 0, stall_ms = 0;
    double gather_ms = 0, gate_up_ms = 0, down_ms = 0;
};

/// The CPU pool's handoffs (CpuExperts): per phase the wake (the caller posting to the last worker starting) and the
/// return (the last worker done to the caller running again), summed.
struct CpuPoolTimes {
    int64_t phases = 0;
    double wake_ms = 0, return_ms = 0;
};

/// One layer's tiers: per expert its tier and slab slot (-1 outside the slab). Warm: in a pinned home (the warm tier's
/// slot or, after a swap, a spare home); Cold: the model's mapping.
struct ExpertLayerTiers {
    std::vector<ExpertTier> tier;
    std::vector<int32_t> slot;
    std::vector<int32_t> vram;   ///< per slab slot its expert, -1 a free slot (at init the hot set's order, then the spares)
};

class CpuExperts;   // experts_cpu.cpp

class ExpertEngine {
public:
    ExpertEngine();
    ~ExpertEngine();
    ExpertEngine(const ExpertEngine&) = delete;
    ExpertEngine& operator=(const ExpertEngine&) = delete;

    /// The tiers of every layer of the model from the config's hot set (checked against the model's expert tensors),
    /// the slab filled, the warm tier bound and filled, the ring, the prefetch region, the copy stream, the pinned
    /// staging (when a source is not pinned) and the buffers of calls of up to max_T tokens allocated, the passes on
    /// `stream` with qmm (the stream's one qmatmul context: one dequant scratch per stream). With dry no CUDA
    /// call is made: the tiers, every expert's views and the CPU kernel's types are checked and the sizes reported.
    /// False with `err` set when the device or the host has no room.
    bool init(const Model& m, const ExpertConfig& cfg, int64_t max_T, QmmContext* qmm, void* stream, bool dry,
              std::string& err);

    /// At the start of each forward call of T tokens: the timing base and, for a prefill chunk, layer 0's prefetch.
    void begin_call(int64_t T, bool prefill);
    /// a prefill call's routing ids (device [T][n_expert_used]; a split
    /// call's xn too) queued to the host now, with an event, right after the router, so the caller can queue other
    /// work (the shared expert) before forward: forward's one host wait then waits for the fetch only, and the host
    /// plans while the device runs that work. forward fetches itself when this was not called for its layer and T.
    void prefill_fetch(int il, const float* xn, const int32_t* ids, int64_t T);
    /// out [T][E] (device) = the routed experts of layer il over xn [T][E] (ffn_norm) for the router's ids and
    /// weights [T][n_expert_used] (device), plus shared [T][E] (device, the shared expert). prefill: every pair on the
    /// GPU, the host waiting once for the routing (the call has queued the next layer's prefetch too); otherwise the
    /// hit/miss split, T <= kExpertSplitMaxT: decode_publish, decode_finish and decode_serve in a row. out is written
    /// in the stream's order when this returns.
    void forward(int il, const float* xn, const int32_t* ids, const float* weights, const float* shared, int64_t T,
                 bool prefill, float* out, const void* xq = nullptr);
    /// Decode in its device and host halves (forward.hpp captures a token's device halves into one graph):
    /// decode_publish queues layer il's doorbell (xn [T][E] and ids [T][n_expert_used], device, into mapped memory,
    /// then the ring); decode_finish queues the hits on the slab, the wait for the CPU's rows and the combine into
    /// out, plus shared (as forward); between the two the caller may queue other work (the shared expert) that the
    /// CPU's misses overlap. Neither makes a host step or bakes a value of the call: the same launches serve any
    /// routing. decode_serve(il, T) is the host's half of a published layer, called once per layer in the order the
    /// device reaches them, while the device runs: it polls the ring (no driver wait), splits the ids, runs the misses
    /// on the CPU into mapped memory and writes the flag the device waits for. It fails naming the layer when the
    /// ring does not come (the stream finished or faulted first, or 20 s passed). xq (decode): xn's
    /// q8_1 rows when the caller has them (hc_decode_step's), which the hits read instead of quantizing xn.
    void decode_publish(int il, const float* xn, const int32_t* ids, int64_t T);
    void decode_finish(int il, const float* xn, const int32_t* ids, const float* weights, const float* shared,
                       int64_t T, float* out, const void* xq = nullptr);
    void decode_serve(int il, int64_t T);
    /// After the caller's wait for the stream at the end of a call: the timing events of its layers read into
    /// layer_stats (timing only).
    void end_call();

    /// a request starts (before its prefill): everything published, with reset the slab back to the hot set's
    /// experts (in rounds of the spare slots, the host waiting for each), then every layer's planner on policy p
    /// (its counts cleared, unless resume without reset: the request continues the engine's sequence, a Live resume
    /// or the prompt-end snapshot, and its planners keep the last request's counts, hot_policy.hpp); the route
    /// trace's R line. deterministic (hot_policy.hpp hot_request_start) always resets and never resumes, and
    /// the request's boundaries wait for their batches (hot_boundary_waits).
    void begin_request(HotPolicy p, bool reset, bool resume, bool deterministic);
    /// a request ended (its last token out): in its deterministic mode (hot_request_end_resets) the open batch
    /// published and the slab reset to the hot set's as begin_request's reset does it, so the next request's start
    /// finds it done; the rounds' swaps, MiB and host ms go to idle_reset() (not to times(), which the next request
    /// resets). With host homes the reset is queued layer by layer (above) and returns without waiting for it. Nothing
    /// otherwise.
    void end_request();
    struct IdleReset {
        int64_t swaps = 0;
        double bytes = 0, ms = 0;   ///< ms: the host's (the rounds' waits; layered: queuing the layers' copies)
        /// restored layer by layer (host homes): the copies' device ms from the first layer's start
        /// to the last one's event, and the next request's waits for the layers' events on the engine's stream, summed,
        /// the longest and its layer, and how many layers waited; known once a call has seen the last event done
        /// (seen_done)
        bool layered = false, seen_done = false;
        double device_ms = 0, wait_ms = 0, wait_max_ms = 0;
        int wait_max_layer = -1, layers_waited = 0;
        /// layered: the host's ms queuing the layers' copies (ms then adds the rounds below). Partial
        /// host homes: the layers without a home for every slab expert reset in rounds right after (in swaps and bytes
        /// above too): their swaps, bytes up and down, the victims copied down, the host's ms, and how many layers
        double queue_ms = 0, round_bytes = 0, round_down_bytes = 0, round_ms = 0;
        int64_t round_swaps = 0, round_downs = 0;
        int round_layers = 0;
    };
    /// The reset end_request did last (zeros: none); a layered one's device times fill in once it is seen done.
    const IdleReset& idle_reset() const { return idle_reset_; }
    /// before the token graph's launch (it holds no event wait), the engine's stream waits for the
    /// rest of an open layered restore at once (its last event); nothing when none is open or every layer was waited
    /// for (a prefill before the graph waits per layer).
    void wait_restore_all();
    /// the request's prompt is read (before its first decode token): prompt and adaptive take its target.
    void prompt_done();
    /// a prompt read beside decoding batch slots is read: its counts join the
    /// batch's at one row's decode mass (HotLayerPlan::merge_prompt), no target of its own.
    void merge_prompt();
    /// a decode token's device work is queued (the token graph launched; run()'s token before its layers): when
    /// no batch is open the next one is planned and its copies queued on the copy streams, running beside the token.
    void decode_started();
    /// between two decode tokens, after the caller's wait for the stream: the open batch published when its
    /// copies are done (an event query; still copying, it stays open: deferred) and the residency rows uploaded on the
    /// engine's stream, before the next token; with hot.check the host waits for the open batch first and checks
    /// every layer's residency (hot_policy.hpp HotConfig::check); in the request's deterministic mode it waits for
    /// the open batch too (ExpertTimes::swap_waits). Batched, a step of two
    /// rows or more: the mode's wait is left out there, the open batch deferred as in the mode off (hot.check still
    /// waits): no row's bits follow the placement, so the deferral changes none of them.
    void decode_boundary(bool batched = false);
    /// The planner's knobs from now on (not its spare slots, fixed at init); exits when c.spare differs.
    void set_hot(const HotConfig& c);
    /// --route-trace: false leaves the next calls out of the trace (their R line is still written).
    void set_trace(bool on);
    const HotConfig& hot() const;
    /// Whether some layer can swap (spare slots and an expert outside the slab).
    bool hot_active() const;
    HotPolicy request_policy() const;
    /// whether the running request is in the deterministic mode
    bool request_deterministic() const;
    /// the prompts read from now on are in the deterministic mode (or not):
    /// a split chunk's CPU experts are then picked among the experts outside the hot set's slab, not the slab as it
    /// is now, so a prompt read beside decoding batch slots gets its alone read's bits. begin_request sets it to its
    /// request's mode; the serve loop sets it for a prompt read beside the slots, which starts no request of its own.
    void prompt_deterministic(bool on);
    /// per layer, the decode calls' pairs and hits since the last reset_times.
    const std::vector<int64_t>& decode_layer_pairs() const { return layer_pairs_; }
    const std::vector<int64_t>& decode_layer_hits() const { return layer_hits_; }
    void set_timing(bool on) { timing_ = on; }
    void set_pool_spin_us(int us);

    /// The dry run of a prefill chunk of T tokens and of a decode step on every layer: the prefill plan of a
    /// routing that reaches every expert of the layer (pair p to expert p mod n_expert spread over the chunk), its
    /// passes, ring experts and rows; the decode split of the first token's slots. Host only.
    void dry_plan(int64_t T, std::FILE* out) const;
    /// The dry run of the hot set's reset (host only, the planners as init lays them out): every layer that swaps
    /// displaced as far as its promotable experts allow, then reset as the engine resets it (with host homes in place,
    /// layer by layer: the order of HotRestore's events, a layer's computation waiting for its own; otherwise in rounds
    /// of the spare slots), checked back at the hot set's slab with every home as init placed it (a dropped victim's
    /// host home still its own), the copies' MiB up and down reported. Exits naming the layer when a check fails.
    void dry_restore(std::FILE* out) const;
    /// the host homes' pinned bytes (0 without them; in a dry run what they would take).
    size_t host_home_bytes() const;
    /// the slab experts with a host home, and the slab's experts (the hot set's VRAM ones)
    int64_t host_homes_held() const;
    int64_t vram_experts() const;

    const ExpertLayerTiers& tiers(int il) const { return tiers_[(size_t) il]; }
    const ExpertTimes& times() const { return times_; }
    const std::vector<ExpertLayerStats>& layer_stats() const { return layer_stats_; }
    CpuPoolTimes cpu_pool_times() const;
    /// The CPU pool's threads (0 before init or in a dry run).
    int cpu_threads() const;
    /// The prefill ring's device bytes (ring_sets x ring_set slots of the layers' largest expert; 0 when every
    /// expert is in the slab), also in a dry run.
    size_t ring_bytes() const;
    /// The split's rates as init measured them.
    PrefillSplitRates split_rates() const;
    /// Whether a prefill chunk of T tokens takes the split (the config's mode and bound, an expert outside the slab).
    bool split_call(int64_t T) const;
    void reset_times();
    /// The tier sizes and budgets, one line per tier.
    void print_summary(std::FILE* out) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    std::vector<ExpertLayerTiers> tiers_;
    ExpertTimes times_;
    IdleReset idle_reset_;
    std::vector<ExpertLayerStats> layer_stats_;
    std::vector<int64_t> layer_pairs_, layer_hits_;
    bool timing_ = false;
};

// ---- the CPU experts, GPU free (experts_cpu.cpp); the engine's misses and the tests' reference ----

/// The base rows of one expert's parts on the host (0 up, 1 gate, 2 down).
struct ExpertRows {
    const uint8_t* part[3] = {nullptr, nullptr, nullptr};
};

/// Exits naming the part unless the CPU experts compute it in `numerics` and this CPU runs their kernels (AVX2, FMA,
/// F16C).
void cpu_experts_check(const ExpertLayerHost& L, const DenseDims& d, CpuNumerics numerics = CpuNumerics::Gpu);

/// One expert's share of a CPU call: its rows and the tokens routed to it, whose outputs are y rows y0 ..
/// y0 + tokens.size() - 1.
struct CpuExpertJob {
    ExpertRows rows;
    std::vector<int32_t> tokens;
    int64_t y0 = 0;
};

/// One host copy of a parallel copy (moe_prefill.h's StageChunk behind a plain pointer).
struct CopyChunk {
    const void* src = nullptr;
    void* dst = nullptr;
    size_t bytes = 0;
};

/// The misses' FFN on the CPU, on a pool of threads: for every job, y [y0 + i][E] = down(clamp(gate x, up x)) of
/// token tokens[i] of x [T][E] (f32, host), one pass over each expert's rows for all of its tokens. Every value is the
/// numerics' (CpuNumerics: the GPU's hit bit for bit, or the fork's CPU experts') whatever the threads and the
/// grouping. A call allocates nothing once its buffers
/// have grown to the call's size. The pool hands each of a call's four phases to its workers; with spin_us > 0 the
/// workers and the caller poll that long before they block on the pool's condition variables (the handoff that
/// pool_times measures).
class CpuExperts {
public:
    explicit CpuExperts(int threads, CpuNumerics numerics = CpuNumerics::Gpu);
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;
    int threads() const { return threads_; }
    CpuNumerics numerics() const { return numerics_; }
    void run(const ExpertLayerHost& L, const DenseDims& d, const float* x, int64_t T, const CpuExpertJob* jobs,
             size_t n_jobs, float* y);
    /// Every chunk's memcpy, shared out over the pool (prefill's staging).
    void copy(const CopyChunk* chunks, size_t n);
    void set_spin_us(int us);
    CpuPoolTimes pool_times() const;
    void reset_pool_times();

    struct Pool;
    struct GpuActs;   // CpuNumerics::Gpu: the call's activations as its row kernel reads them (experts_cpu.cpp)

private:
    int threads_ = 1;
    CpuNumerics numerics_ = CpuNumerics::Gpu;
    std::unique_ptr<Pool> pool_;
    std::unique_ptr<GpuActs> gpu_;
    std::vector<uint8_t> qx_, qh_;     // quantized rows (q8_1 or Q8_K) of x per token, of h per output row
    std::vector<float> g_, u_;         // gate and up per output row
    std::vector<const void*> act_x_, act_h_;   // per output row: its token's quantized input, its h
    std::vector<float*> out_g_, out_u_, out_y_;
};

/// Decode's split of a call's pairs ids[0 .. P) on the host (ExpertEngine::decode_serve): every pair whose expert has
/// no slab slot (slot[e] < 0) joins its expert's CPU job, the jobs in first-use order and each job's tokens in pair
/// order, so an expert that several tokens of a call route to is read once for all of them (a batch's tokens in
/// one call; CpuExperts::run's values do not depend on the grouping). job_of
/// [n_expert] all -1 on entry (each job's index after), job_expert each job's expert, jobs room for every expert (their
/// rows are the caller's to fill), miss[p] pair p's row of the jobs' output (-1: a hit). Returns the jobs; hits the
/// pairs on the slab.
size_t decode_miss_jobs(const int32_t* ids, int64_t P, int K, const int32_t* slot, int32_t* job_of,
                        std::vector<int32_t>& job_expert, CpuExpertJob* jobs, int32_t* miss, int64_t& hits);

/// The combine on the host, the reference of the device's: out [T][E] = sum over slots in order of y[p] w[p], plus
/// shared (null: none). y [T * n_used][E] pair-indexed.
void ref_expert_combine(const float* y, const float* w, int64_t T, int n_used, int E, const float* shared, float* out);

}  // namespace strata::dsv41::engine
