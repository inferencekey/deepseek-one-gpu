// src/dsv41/moe_profile.cpp - per-layer counts and, with STRATA_MOE_PROFILE=1, times of strata_moe_forward.
//
// Every call adds its hits and misses (pairs computed by the hit path and by the CPU) to its layer, which
// strata_moe_layer_counts returns (the replay harness prints them), and the range of SwiGLU clamp limits its
// calls computed with, which strata_moe_layer_clamp returns (dsv41-moe-clamp-check compares it with the GGUF).
// With STRATA_MOE_PROFILE=1 a call also adds, in ms: submit (the submitting thread queueing the hit path), gpu
// (CUDA events around the hit path's uploads, graph and download), cpu (from the call's start to the end of the
// CPU phases), sync (the part of cpu a thread spends, on average, waiting at the barriers between the CPU phases
// for the other threads: a slow or descheduled thread shows up here), wait (the submitting thread blocked on the
// hit path after its CPU work) and op (the whole call; a layer's first call includes the upload of its hot
// experts, before its CPU phases start). Decode calls (1 token) and batch calls (more tokens) are kept apart, and
// the per-layer summary is printed to stderr at exit.
//
// Also with STRATA_MOE_PROFILE=1: the process's major page faults (getrusage) from the first call on, each call
// taking the faults since the previous one finished (the dense part of the graph between them included), per
// token of its kind; and with the residency (STRATA_MOE_RAM_GB) a tier table per layer: the pairs computed
// from the warm and from the cold tier, the MiB per call each gave the CPU (the cold ones as read, whole 4096-byte
// blocks), cold (ms from queueing the first cold read to the end of the last, per call), the cold reads' GB/s
// (bytes over that time, so with several rounds it includes the CPU time the later rounds waited for), and cwait
// (the part of cpu a thread spends, on average, waiting for cold bytes). With STRATA_MOE_COLD=pagecache the
// cold experts are computed from the fork's mapping: cold MiB is their bytes, read through the page cache inside
// cpu (its misses are the major faults above), and cold, GB/s and cwait are 0.
//
// the prefill calls (moe_prefill.h) get a table of their own per layer: the bytes copied to the ring (and the
// part staged first: by the CPU threads, or read with O_DIRECT under STRATA_MOE_PREFILL_COLD=direct), the copy
// stream's busy ms and GB/s, the compute stream's busy ms, the GPU ms of the call, the overlap of the two streams
// ((copy + compute - gpu) / the shorter one), the scheduling thread's time filling the staging (copying, or
// waiting for the direct reads) and its staging wait, and op, the whole call, over the prefill calls that were not
// their layer's first call (that one also uploads the slab and fills the warm tier, seconds that are no ubatch's
// MoE time); and the sum of the layer means, one ubatch's worth. Only the prefill calls are in it: the batch-call
// table above also holds the smaller batches, the server's 2-token warmup among them. Their pairs are hits in the
// tables above and stay out of the tier table, which is the CPU's misses.
#include "moe_op.h"

#include "ggml.h"

#include <sys/resource.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace strata::dsv41 {
namespace {

struct Totals {
    int64_t calls = 0, tokens = 0, hits = 0, misses = 0;
    double submit_ms = 0.0, gpu_ms = 0.0, cpu_ms = 0.0, sync_ms = 0.0, wait_ms = 0.0, op_ms = 0.0;
    int64_t major_faults = 0;
    int64_t tiered_calls = 0, mapped_calls = 0, warm_pairs = 0, cold_pairs = 0;
    double warm_bytes = 0.0, cold_bytes = 0.0, cold_ms = 0.0, cold_wait_ms = 0.0;
    // the prefill calls (moe_op.h, Call::prefill_*)
    int64_t prefill_calls = 0, prefill_passes = 0, prefill_pairs = 0;
    double prefill_bytes = 0.0, prefill_staged_bytes = 0.0, prefill_copy_ms = 0.0, prefill_compute_ms = 0.0;
    double prefill_gpu_ms = 0.0, prefill_stage_ms = 0.0, prefill_stage_wait_ms = 0.0;
    int64_t prefill_op_calls = 0;  // the prefill calls that were not their layer's first call
    double prefill_op_ms = 0.0;    // their op ms

    void add(const Totals& o) {
        prefill_calls += o.prefill_calls;
        prefill_passes += o.prefill_passes;
        prefill_pairs += o.prefill_pairs;
        prefill_bytes += o.prefill_bytes;
        prefill_staged_bytes += o.prefill_staged_bytes;
        prefill_copy_ms += o.prefill_copy_ms;
        prefill_compute_ms += o.prefill_compute_ms;
        prefill_gpu_ms += o.prefill_gpu_ms;
        prefill_stage_ms += o.prefill_stage_ms;
        prefill_stage_wait_ms += o.prefill_stage_wait_ms;
        prefill_op_calls += o.prefill_op_calls;
        prefill_op_ms += o.prefill_op_ms;
        calls += o.calls;
        tokens += o.tokens;
        hits += o.hits;
        misses += o.misses;
        major_faults += o.major_faults;
        tiered_calls += o.tiered_calls;
        mapped_calls += o.mapped_calls;
        warm_pairs += o.warm_pairs;
        cold_pairs += o.cold_pairs;
        warm_bytes += o.warm_bytes;
        cold_bytes += o.cold_bytes;
        cold_ms += o.cold_ms;
        cold_wait_ms += o.cold_wait_ms;
        submit_ms += o.submit_ms;
        gpu_ms += o.gpu_ms;
        cpu_ms += o.cpu_ms;
        sync_ms += o.sync_ms;
        wait_ms += o.wait_ms;
        op_ms += o.op_ms;
    }
};

struct LayerTotals {
    Totals decode, batch;  // calls of 1 token, of more tokens
    float min_limit = 0.0f, max_limit = 0.0f;  // over every call of the layer
};

std::mutex g_mutex;
std::map<int, LayerTotals> g_layers;
int64_t g_major_faults = 0;  // the process's count when the last call finished (or profiling started)

int64_t major_faults_now() {
    rusage u{};
    getrusage(RUSAGE_SELF, &u);
    return (int64_t) u.ru_majflt;
}

// The residency's columns, for the layers whose calls of this kind had it; nothing without it.
void print_tiers(bool decode) {
    Totals all;
    for (const auto& [layer, lt] : g_layers) all.add(decode ? lt.decode : lt.batch);
    if (all.tiered_calls == 0) return;
    std::fprintf(stderr, "strata-moe tiers, %s (per call: mean MiB and ms)%s\n", decode ? "decode calls" : "batch calls",
                 all.mapped_calls == all.tiered_calls ? "; STRATA_MOE_COLD=pagecache: cold experts from the fork's "
                                                         "mapping, cold MiB their bytes, no read timed"
                 : all.mapped_calls == 0 ? "; STRATA_MOE_COLD=direct: cold experts read with O_DIRECT"
                                         : "; calls of both cold paths, pagecache and direct");
    std::fprintf(stderr, "%5s %9s %9s %9s %9s %8s %8s %8s\n", "layer", "warm", "cold", "warm MiB", "cold MiB", "cold",
                 "GB/s", "cwait");
    constexpr double kMiB = 1024.0 * 1024.0;
    double pass_cold_ms = 0.0, pass_wait_ms = 0.0, pass_warm = 0.0, pass_cold = 0.0;
    for (const auto& [layer, lt] : g_layers) {
        const Totals& t = decode ? lt.decode : lt.batch;
        if (t.tiered_calls == 0) continue;
        const double n = (double) t.tiered_calls;
        std::fprintf(stderr, "%5d %9lld %9lld %9.1f %9.1f %8.3f %8.2f %8.3f\n", layer, (long long) t.warm_pairs,
                     (long long) t.cold_pairs, t.warm_bytes / n / kMiB, t.cold_bytes / n / kMiB, t.cold_ms / n,
                     t.cold_ms > 0.0 ? t.cold_bytes / t.cold_ms / 1e6 : 0.0, t.cold_wait_ms / n);
        pass_cold_ms += t.cold_ms / n;
        pass_wait_ms += t.cold_wait_ms / n;
        pass_warm += t.warm_bytes / n;
        pass_cold += t.cold_bytes / n;
    }
    // the CPU's calls only: a prefill call's pairs are in the prefill table
    const int64_t vram = all.hits - all.prefill_pairs, pairs = vram + all.misses;
    std::fprintf(stderr, "tiers over all layers: VRAM %lld, warm %lld, cold %lld of %lld pairs (%.1f / %.1f / %.1f%%); "
                 "per forward pass (sum of the layer means): warm %.1f MiB, cold %.1f MiB read in %.3f ms (%.2f GB/s), "
                 "cwait %.3f ms\n",
                 (long long) vram, (long long) all.warm_pairs, (long long) all.cold_pairs, (long long) pairs,
                 pairs > 0 ? 100.0 * (double) vram / (double) pairs : 0.0,
                 pairs > 0 ? 100.0 * (double) all.warm_pairs / (double) pairs : 0.0,
                 pairs > 0 ? 100.0 * (double) all.cold_pairs / (double) pairs : 0.0, pass_warm / kMiB,
                 pass_cold / kMiB, pass_cold_ms, pass_cold_ms > 0.0 ? pass_cold / pass_cold_ms / 1e6 : 0.0,
                 pass_wait_ms);
}

// The overlap of the two GPU streams: the share of the shorter one's busy time hidden under the other's (0: one
// after the other, 1: the shorter wholly under the longer).
double overlap(double copy_ms, double compute_ms, double gpu_ms) {
    const double shorter = std::min(copy_ms, compute_ms);
    if (shorter <= 0.0) return 0.0;
    return std::clamp((copy_ms + compute_ms - gpu_ms) / shorter, 0.0, 1.0);
}

// the prefill calls' table, for the layers that had any.
void print_prefill() {
    Totals all;
    for (const auto& [layer, lt] : g_layers) all.add(lt.batch);
    if (all.prefill_calls == 0) return;
    constexpr double kMiB = 1024.0 * 1024.0;
    std::fprintf(stderr, "strata-moe prefill, batch calls on the GPU (per call: mean; MiB copied to the ring and of "
                 "them staged, the copy and compute streams' busy ms, the GPU ms from the call's start to its last "
                 "copy or pass, overlap of the two streams, the scheduling thread's ms filling the staging and "
                 "waiting for a staging set, the whole call's ms over the op calls, those not the layer's first)\n");
    std::fprintf(stderr, "%5s %6s %7s %7s %9s %9s %8s %7s %8s %8s %7s %8s %8s %6s %9s\n", "layer", "calls", "tokens",
                 "passes", "MiB", "staged", "copy", "GB/s", "compute", "gpu", "overlap", "stage", "swait", "opcall",
                 "op");
    Totals pass;  // the sum over layers of the per-call means: one ubatch of the forward pass
    int64_t op_calls = 0;  // the most op calls any layer had
    for (const auto& [layer, lt] : g_layers) {
        const Totals& t = lt.batch;
        if (t.prefill_calls == 0) continue;
        const double n = (double) t.prefill_calls;
        const double op_ms = t.prefill_op_calls > 0 ? t.prefill_op_ms / (double) t.prefill_op_calls : 0.0;
        std::fprintf(stderr,
                     "%5d %6lld %7lld %7.1f %9.1f %9.1f %8.2f %7.2f %8.2f %8.2f %7.2f %8.2f %8.2f %6lld %9.2f\n",
                     layer, (long long) t.prefill_calls, (long long) t.tokens, (double) t.prefill_passes / n,
                     t.prefill_bytes / n / kMiB, t.prefill_staged_bytes / n / kMiB, t.prefill_copy_ms / n,
                     t.prefill_copy_ms > 0.0 ? t.prefill_bytes / t.prefill_copy_ms / 1e6 : 0.0,
                     t.prefill_compute_ms / n, t.prefill_gpu_ms / n,
                     overlap(t.prefill_copy_ms, t.prefill_compute_ms, t.prefill_gpu_ms), t.prefill_stage_ms / n,
                     t.prefill_stage_wait_ms / n, (long long) t.prefill_op_calls, op_ms);
        pass.prefill_op_ms += op_ms;
        op_calls = std::max(op_calls, t.prefill_op_calls);
        pass.prefill_bytes += t.prefill_bytes / n;
        pass.prefill_staged_bytes += t.prefill_staged_bytes / n;
        pass.prefill_copy_ms += t.prefill_copy_ms / n;
        pass.prefill_compute_ms += t.prefill_compute_ms / n;
        pass.prefill_gpu_ms += t.prefill_gpu_ms / n;
        pass.prefill_stage_ms += t.prefill_stage_ms / n;
    }
    std::fprintf(stderr, "prefill per ubatch (sum of the layer means): %.1f GiB copied (%.1f GiB staged) in %.1f ms of "
                 "copy stream (%.2f GB/s), %.1f ms of compute, %.1f ms of GPU time (overlap %.2f), %.1f ms filling the "
                 "staging on the host, %.1f ms of MoE op over %lld op calls per layer (a layer's first call left "
                 "out)\n",
                 pass.prefill_bytes / (1024.0 * kMiB), pass.prefill_staged_bytes / (1024.0 * kMiB),
                 pass.prefill_copy_ms,
                 pass.prefill_copy_ms > 0.0 ? pass.prefill_bytes / pass.prefill_copy_ms / 1e6 : 0.0,
                 pass.prefill_compute_ms, pass.prefill_gpu_ms,
                 overlap(pass.prefill_copy_ms, pass.prefill_compute_ms, pass.prefill_gpu_ms), pass.prefill_stage_ms,
                 pass.prefill_op_ms, (long long) op_calls);
}

void print_table(const char* title, bool decode) {
    std::fprintf(stderr, "strata-moe profile, %s (per call: mean ms)\n", title);
    std::fprintf(stderr, "%5s %7s %8s %9s %9s %6s %8s %8s %8s %8s %8s %8s\n", "layer", "calls", "tokens", "hits",
                 "misses", "hit%", "submit", "gpu", "cpu", "sync", "wait", "op");
    Totals all, pass;  // pass: the sum over layers of the per-call means, the MoE time of one forward pass
    int64_t pass_tokens = 0;  // the tokens of the forward passes: the most any layer saw
    for (const auto& [layer, lt] : g_layers) {
        const Totals& t = decode ? lt.decode : lt.batch;
        if (t.calls == 0) continue;
        pass_tokens = std::max(pass_tokens, t.tokens);
        const double n = (double) t.calls;
        const int64_t pairs = t.hits + t.misses;
        std::fprintf(stderr, "%5d %7lld %8lld %9lld %9lld %6.1f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n", layer,
                     (long long) t.calls, (long long) t.tokens, (long long) t.hits, (long long) t.misses,
                     pairs > 0 ? 100.0 * (double) t.hits / (double) pairs : 0.0, t.submit_ms / n, t.gpu_ms / n,
                     t.cpu_ms / n, t.sync_ms / n, t.wait_ms / n, t.op_ms / n);
        all.add(t);
        pass.submit_ms += t.submit_ms / n;
        pass.gpu_ms += t.gpu_ms / n;
        pass.cpu_ms += t.cpu_ms / n;
        pass.sync_ms += t.sync_ms / n;
        pass.wait_ms += t.wait_ms / n;
        pass.op_ms += t.op_ms / n;
    }
    const int64_t pairs = all.hits + all.misses;
    std::fprintf(stderr, "all layers: %lld calls, hit rate %.1f%%; per forward pass (sum of the layer means): "
                 "submit %.3f gpu %.3f cpu %.3f sync %.3f wait %.3f op %.3f ms\n",
                 (long long) all.calls, pairs > 0 ? 100.0 * (double) all.hits / (double) pairs : 0.0, pass.submit_ms,
                 pass.gpu_ms, pass.cpu_ms, pass.sync_ms, pass.wait_ms, pass.op_ms);
    std::fprintf(stderr, "major page faults: %lld over %lld tokens, %.1f per token (the whole process, from the first "
                 "call on)\n", (long long) all.major_faults, (long long) pass_tokens,
                 pass_tokens > 0 ? (double) all.major_faults / (double) pass_tokens : 0.0);
    print_tiers(decode);
}

void print_summary() {
    std::lock_guard<std::mutex> lock(g_mutex);
    print_table("decode calls (1 token)", true);
    print_table("batch calls (more than 1 token)", false);
    print_prefill();
}

}  // namespace

bool profiling() {
    static const bool on = [] {
        const char* env = std::getenv("STRATA_MOE_PROFILE");
        if (env == nullptr || std::strcmp(env, "") == 0 || std::strcmp(env, "0") == 0) return false;
        if (std::strcmp(env, "1") != 0) GGML_ABORT("strata-moe: STRATA_MOE_PROFILE=%s: expected 1 (on), 0 or unset", env);
        g_major_faults = major_faults_now();
        std::atexit(print_summary);
        std::fprintf(stderr, "strata-moe: STRATA_MOE_PROFILE=1, per-layer times printed at exit\n");
        return true;
    }();
    return on;
}

void profile_record(const Call& c, double op_ms) {
    Totals t;
    t.calls = 1;
    t.tokens = c.n_tokens;
    t.hits = c.n_hits;
    t.misses = (int64_t) c.miss_pairs.size();
    if (profiling()) {
        t.submit_ms = c.submit_ms;
        t.gpu_ms = c.gpu_ms;
        t.cpu_ms = c.cpu_ms;
        t.sync_ms = (double) c.sync_ns.v.load(std::memory_order_relaxed) / 1e6 / (double) c.nth;
        t.wait_ms = c.wait_ms;
        t.op_ms = op_ms;
        if (c.prefill) {
            t.prefill_calls = 1;
            t.prefill_passes = c.prefill_passes;
            t.prefill_pairs = c.n_hits;
            t.prefill_bytes = (double) c.prefill_bytes;
            t.prefill_staged_bytes = (double) c.prefill_staged_bytes;
            t.prefill_copy_ms = c.prefill_copy_ms;
            t.prefill_compute_ms = c.prefill_compute_ms;
            t.prefill_gpu_ms = c.prefill_gpu_ms;
            t.prefill_stage_ms = c.prefill_stage_ms;
            t.prefill_stage_wait_ms = c.prefill_stage_wait_ms;
        } else if (c.residency != nullptr) {
            // the tier table is the CPU's misses; a prefill call has none
            t.tiered_calls = 1;
            t.mapped_calls = c.cold_mapped ? 1 : 0;
            t.warm_pairs = c.warm_pairs;
            t.cold_pairs = c.cold_pairs;
            t.warm_bytes = (double) c.warm_bytes;
            t.cold_bytes = (double) c.cold_bytes;
            t.cold_ms = c.cold_ms;
            t.cold_wait_ms = (double) c.cold_wait_ns.v.load(std::memory_order_relaxed) / 1e6 / (double) c.nth;
        }
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (profiling()) {
        const int64_t faults = major_faults_now();
        t.major_faults = faults - g_major_faults;
        g_major_faults = faults;
    }
    const auto [it, first] = g_layers.try_emplace(c.layer);
    if (c.prefill && profiling() && !first) {
        t.prefill_op_calls = 1;
        t.prefill_op_ms = op_ms;
    }
    LayerTotals& lt = it->second;
    lt.min_limit = first ? c.limit : std::min(lt.min_limit, c.limit);
    lt.max_limit = first ? c.limit : std::max(lt.max_limit, c.limit);
    (c.n_tokens == 1 ? lt.decode : lt.batch).add(t);
}

bool profile_layer_counts(int layer, int64_t* hits, int64_t* misses) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_layers.find(layer);
    if (it == g_layers.end()) return false;
    *hits = it->second.decode.hits + it->second.batch.hits;
    *misses = it->second.decode.misses + it->second.batch.misses;
    return true;
}

bool profile_layer_clamp(int layer, float* min_limit, float* max_limit, int64_t* calls) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_layers.find(layer);
    if (it == g_layers.end()) return false;
    *min_limit = it->second.min_limit;
    *max_limit = it->second.max_limit;
    *calls = it->second.decode.calls + it->second.batch.calls;
    return true;
}

}  // namespace strata::dsv41
