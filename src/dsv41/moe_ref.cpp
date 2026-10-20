// src/dsv41/moe_ref.cpp - one call of strata_moe_forward (include/strata/dsv41/moe.h): the hit/miss split, the
// CPU work on the misses with the reference or the fast kernel, and the combine.
//
// The CPU reference computes what the fork's stock expert path computes on the CPU backend, with the same ggml-cpu
// kernels, in five phases split over ggml-cpu's threads with a barrier between them:
//   0. every token's input is quantized to the up/gate type's vec_dot_type with ggml-cpu's from_float, as
//      mul_mat_id quantizes src1 before its dot products;
//   1. per expert with a miss and block of n_ff rows: for every missed (token, slot) routed to it, gate . x and
//      up . x with ggml-cpu's vec_dot of the weight type;
//   2. per missed (token, slot): h = SwiGLU(gate, up), ggml_vec_swiglu_f32 as ggml_swiglu_split runs it, or the
//      scalar loop of ggml_swiglu_clamp when the layer has a limit; h is then quantized to down's vec_dot_type;
//   3. per expert with a miss and block of n_embd rows: y = down . h for its missed (token, slot) pairs;
//   4. out = y[slot 0] * w[0] + y[slot 1] * w[1] + ..., left to right, the order of the stock graph's adds, with
//      y * w of a hit taken from the hit path (moe_op.h) instead.
// With no hits every output element goes through the same kernels on the same inputs as in the stock graph, so
// the result is bit-identical to it (tests/dsv41/moe_ref_test.cpp). The file is built with -ffp-contract=off so
// the clamp and phase 4 round every multiply and add on its own, like the separate stock ops.
//
// The fast kernel (STRATA_MOE_CPU=fast, cpu_kernel.h) replaces the vec_dot calls of phases 1 and 3: a work item is
// a tile of kFastRowsPerItem rows of one expert, computed for every missed pair of that expert at once (decode: 1
// token, prefill: all the ubatch's tokens routed there), one pass over each row for all of them. Its rows are
// ggml-cpu's AVX2 vec_dot to the bit, and phases 0, 2 and 4 are shared, so it gives the reference's output bit for bit
// (tests/dsv41/moe_cpu_fast_test.cpp). The work goes over the op's own ggml-cpu threads, which llama.cpp already
// binds where the process runs (the pod scripts start it under numactl --cpunodebind on the GPU's node), through
// atomic item counters on their own cache lines; the per-pair outputs start on cache lines, and an item's rows are
// whole lines of them, so no two threads write one line. Calls are pooled, so no call allocates once the process
// has seen its largest ubatch.
//
// Phases 1 to 3 run in rounds of experts (moe_op.h): one round of every missed expert without the residency; with
// it (moe_residency.h) the warm experts and the first cold ones are round 0 and the other cold ones follow
// in rounds that fit the staging, a work item on a cold expert first waiting for its bytes. Every (token, slot)
// still goes through the same kernels on the same inputs, so the rounds change no bit.
//
// A prefill call (moe_prefill.h) has no misses: the hit path computes every pair, the other threads copy its
// staging chunks while it schedules, and phase 4 adds the slots from its output rows in the same order.
#include "moe_op.h"

#include "cpu_kernel.h"
#include "ggml_cpu_vec.h"
#include "moe_residency.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
#include <utility>

namespace strata::dsv41 {
namespace {

// rows per work item in phases 1 and 3: enough to amortize the item fetch, few enough to balance 8+ threads on
// one token's 6 experts (2304 / 16 = 144 items per expert)
constexpr int64_t kRowsPerItem = 16;
// rows per item of phases 1 and 3 with the fast kernel: 64 rows of gate and up are 215 KB of q2_K, 64 of down
// 63 KB of q3_K, and one token's 6 experts still give 216 and 480 items to share out
constexpr int64_t kFastRowsPerItem = 64;
// rows per item of phase 4
constexpr int64_t kOutRowsPerItem = 512;

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

RowFormat row_format(const ggml_tensor* t, int64_t n_per_row) {
    const ggml_type_traits_cpu* cpu = ggml_get_type_traits_cpu(t->type);
    if (cpu->vec_dot == nullptr) {
        GGML_ABORT("strata-moe: tensor %s has type %s, which has no ggml-cpu dot product", t->name,
                   ggml_type_name(t->type));
    }
    // mul_mat runs a multi-row kernel (nrows 2, ARM int8 matmul) for these: one row at a time would not match it
    if (cpu->nrows != 1) {
        GGML_ABORT("strata-moe: tensor %s has type %s, whose ggml-cpu dot product takes %lld rows at once", t->name,
                   ggml_type_name(t->type), (long long) cpu->nrows);
    }
    RowFormat f;
    f.vec_dot = cpu->vec_dot;
    f.act_type = cpu->vec_dot_type;
    f.act_quantize = ggml_get_type_traits_cpu(f.act_type)->from_float;
    if (f.act_quantize == nullptr) {
        GGML_ABORT("strata-moe: tensor %s (%s): ggml-cpu cannot convert f32 to its activation type %s", t->name,
                   ggml_type_name(t->type), ggml_type_name(f.act_type));
    }
    if (n_per_row % ggml_blck_size(t->type) != 0 || n_per_row % ggml_blck_size(f.act_type) != 0) {
        GGML_ABORT("strata-moe: tensor %s (%s): row length %lld is not whole %s / %s blocks", t->name,
                   ggml_type_name(t->type), (long long) n_per_row, ggml_type_name(t->type),
                   ggml_type_name(f.act_type));
    }
    f.act_bytes = ggml_row_size(f.act_type, n_per_row);
    return f;
}

float row_dot(const RowFormat& f, int64_t n, const void* row, const void* act) {
    float s = 0.0f;
    f.vec_dot((int) n, &s, 0, row, 0, act, 0, 1);
    return s;
}

// The source checks the fork's ggml_custom_4d call relies on; a mismatch aborts naming the tensor and value.
void check_type(const ggml_tensor* t, ggml_type want, const char* role) {
    if (t->type != want) {
        GGML_ABORT("strata-moe: %s %s has type %s, expected %s", role, t->name, ggml_type_name(t->type),
                   ggml_type_name(want));
    }
}

void check_shape(const ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, const char* role) {
    if (t->ne[0] != ne0 || t->ne[1] != ne1 || t->ne[2] != ne2 || t->ne[3] != 1) {
        GGML_ABORT("strata-moe: %s %s has shape [%lld, %lld, %lld, %lld], expected [%lld, %lld, %lld, 1]", role, t->name,
                   (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                   (long long) ne0, (long long) ne1, (long long) ne2);
    }
    if (t->nb[0] != ggml_type_size(t->type)) {
        GGML_ABORT("strata-moe: %s %s has nb[0] = %zu, expected packed rows (%zu)", role, t->name, t->nb[0],
                   ggml_type_size(t->type));
    }
}

void check_host(const ggml_tensor* t, const char* role) {
    if (t->data == nullptr) GGML_ABORT("strata-moe: %s %s has no data", role, t->name);
    if (t->buffer != nullptr && !ggml_backend_buffer_is_host(t->buffer)) {
        GGML_ABORT("strata-moe: %s %s is in buffer %s, not host memory in ggml's layout (CPU_REPACK needs --no-repack)",
                   role, t->name, ggml_backend_buffer_name(t->buffer));
    }
}

// the layer of an expert tensor named blk.<layer>.ffn_<kind>_exps.weight (the hot set is keyed by it)
int layer_of(const ggml_tensor* t, const char* kind) {
    int layer = -1, end = 0;
    char want[64];
    std::snprintf(want, sizeof(want), "blk.%%d.ffn_%s_exps.weight%%n", kind);
    if (std::sscanf(t->name, want, &layer, &end) != 1 || t->name[end] != '\0' || layer < 0) {
        GGML_ABORT("strata-moe: %s_exps tensor is named '%s', expected blk.<layer>.ffn_%s_exps.weight", kind, t->name,
                   kind);
    }
    return layer;
}

const char* row_ptr(const ggml_tensor* t, int64_t e, int64_t r) {
    return static_cast<const char*>(t->data) + e * t->nb[2] + r * t->nb[1];
}

// Fills the hit layout of moe_op.h from the pairs of the experts resident in `slots`.
void split_hits(Call& c, const std::vector<int32_t>& slots) {
    const int64_t n_pairs = c.n_tokens * c.n_used;
    c.hit_row.assign((size_t) n_pairs, -1);
    c.hit_tokens.clear();
    c.hit_width = 0;
    c.n_hits = 0;
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        int64_t n = 0;
        for (int64_t s = 0; s < c.n_used; ++s) {
            const int32_t e = *reinterpret_cast<const int32_t*>(static_cast<const char*>(c.ids->data) +
                                                                t * c.ids->nb[1] + s * c.ids->nb[0]);
            n += slots[(size_t) e] >= 0;
        }
        if (n > 0) c.hit_tokens.push_back((int32_t) t);
        c.hit_width = std::max(c.hit_width, n);
        c.n_hits += n;
    }
    const size_t n_rows = c.hit_tokens.size() * (size_t) c.hit_width;
    c.hit_slots.resize(n_rows);
    c.hit_experts.resize(n_rows);
    c.hit_weights.assign(n_rows, 0.0f);
    std::vector<int32_t>& expert_of_slot = c.expert_of_slot;  // the inverse of `slots`, for the padding
    expert_of_slot.clear();
    for (size_t e = 0; e < slots.size(); ++e) {
        if (slots[e] < 0) continue;
        if ((size_t) slots[e] >= expert_of_slot.size()) expert_of_slot.resize((size_t) slots[e] + 1, -1);
        expert_of_slot[(size_t) slots[e]] = (int32_t) e;
    }
    std::vector<uint8_t>& taken = c.slot_taken;
    taken.resize(expert_of_slot.size());
    for (size_t j = 0; j < c.hit_tokens.size(); ++j) {
        const int64_t t = c.hit_tokens[j];
        const size_t row0 = j * (size_t) c.hit_width;
        int64_t i = 0;
        for (int64_t s = 0; s < c.n_used; ++s) {
            const int32_t e = *reinterpret_cast<const int32_t*>(static_cast<const char*>(c.ids->data) +
                                                                t * c.ids->nb[1] + s * c.ids->nb[0]);
            if (slots[(size_t) e] < 0) continue;
            const size_t row = row0 + (size_t) i++;
            c.hit_slots[row] = slots[(size_t) e];
            c.hit_experts[row] = e;
            c.hit_weights[row] = *reinterpret_cast<const float*>(static_cast<const char*>(c.weights->data) +
                                                                 t * c.weights->nb[2] + s * c.weights->nb[1]);
            c.hit_row[(size_t) (t * c.n_used + s)] = (int32_t) row;
        }
        // padding takes the lowest slots the token does not use, with weight 0; its output rows are never read.
        // The slots of one token must be distinct: ggml-cuda's mul_mat_id (mm_ids_helper in mmid.cu) counts every
        // slot of a token in the experts' row offsets but gives a token one row per expert, so a repeated expert
        // leaves a row of the compact layout unwritten and the kernel writes it over other tokens' outputs
        std::fill(taken.begin(), taken.end(), 0);
        for (int64_t k = 0; k < i; ++k) taken[(size_t) c.hit_slots[row0 + (size_t) k]] = 1;
        size_t free_slot = 0;
        for (; i < c.hit_width; ++i) {
            while (taken[free_slot]) ++free_slot;
            const size_t row = row0 + (size_t) i;
            c.hit_slots[row] = (int32_t) free_slot;
            c.hit_experts[row] = expert_of_slot[free_slot];
            taken[free_slot] = 1;
        }
    }
}

// Fills a reset call from the op's sources; every check aborts naming the tensor and the value.
void init_call(Call& c, const ggml_tensor* dst, float limit, HitPath* hits, Residency* residency) {
    c.t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 6; ++i) {
        if (dst->src[i] == nullptr) GGML_ABORT("strata-moe: %s has no src[%d]", dst->name, i);
    }
    c.cur = dst->src[0];
    c.ids = dst->src[1];
    c.weights = dst->src[2];
    c.up = dst->src[3];
    c.gate = dst->src[4];
    c.down = dst->src[5];
    c.limit = limit;

    c.n_embd = c.cur->ne[0];
    c.n_tokens = c.cur->ne[1];
    c.n_used = c.ids->ne[0];
    c.n_ff = c.up->ne[1];
    c.n_expert = c.up->ne[2];

    check_type(dst, GGML_TYPE_F32, "output");
    check_type(c.cur, GGML_TYPE_F32, "input");
    check_type(c.ids, GGML_TYPE_I32, "selected_experts");
    check_type(c.weights, GGML_TYPE_F32, "weights");
    check_shape(dst, c.n_embd, c.n_tokens, 1, "output");
    check_shape(c.cur, c.n_embd, c.n_tokens, 1, "input");
    check_shape(c.ids, c.n_used, c.n_tokens, 1, "selected_experts");
    check_shape(c.weights, 1, c.n_used, c.n_tokens, "weights");
    check_shape(c.up, c.n_embd, c.n_ff, c.n_expert, "up_exps");
    check_shape(c.gate, c.n_embd, c.n_ff, c.n_expert, "gate_exps");
    check_shape(c.down, c.n_ff, c.n_embd, c.n_expert, "down_exps");
    if (!ggml_is_contiguous(dst)) GGML_ABORT("strata-moe: output %s is not contiguous", dst->name);
    check_host(c.cur, "input");
    check_host(c.ids, "selected_experts");
    check_host(c.weights, "weights");
    check_host(c.up, "up_exps");
    check_host(c.gate, "gate_exps");
    check_host(c.down, "down_exps");
    if (!(limit == 0.0f || (limit > 0.0f && std::isfinite(limit)))) {
        GGML_ABORT("strata-moe: %s: SwiGLU clamp limit %g, expected 0 (none) or a finite limit > 0", dst->name, limit);
    }

    c.layer = layer_of(c.up, "up");
    if (layer_of(c.gate, "gate") != c.layer || layer_of(c.down, "down") != c.layer) {
        GGML_ABORT("strata-moe: %s, %s and %s are not one layer's experts", c.up->name, c.gate->name, c.down->name);
    }

    c.up_fmt = row_format(c.up, c.n_embd);
    c.gate_fmt = row_format(c.gate, c.n_embd);
    c.down_fmt = row_format(c.down, c.n_ff);
    if (c.kernel == CpuKernel::kFast) {
        if (!fast_kernel_cpu_ok()) {
            GGML_ABORT("strata-moe: STRATA_MOE_CPU=fast needs AVX2, FMA and F16C, which this CPU lacks; "
                       "STRATA_MOE_CPU=ref runs on ggml-cpu's own kernels");
        }
        for (const ggml_tensor* t : {c.up, c.gate, c.down}) {
            if (!fast_kernel_supports(t->type)) {
                GGML_ABORT("strata-moe: STRATA_MOE_CPU=fast has no kernel for %s (%s; it has q2_K, q3_K, q4_K, "
                           "q6_K); STRATA_MOE_CPU=ref runs any type", t->name, ggml_type_name(t->type));
            }
        }
    }

    const int64_t n_pairs = c.n_tokens * c.n_used;
    for (int64_t t = 0; t < c.n_tokens; ++t) {
        for (int64_t s = 0; s < c.n_used; ++s) {
            const int32_t e = *reinterpret_cast<const int32_t*>(static_cast<const char*>(c.ids->data) +
                                                                t * c.ids->nb[1] + s * c.ids->nb[0]);
            if (e < 0 || e >= c.n_expert) {
                GGML_ABORT("strata-moe: selected_experts %s: token %lld slot %lld is expert %d, outside [0, %lld)",
                           c.ids->name, (long long) t, (long long) s, e, (long long) c.n_expert);
            }
        }
    }

    const std::vector<int32_t>* slots = hits != nullptr ? hits->slots(c) : nullptr;
    if (hits != nullptr && hits->prefill(c, residency)) {
        // the hit path took every pair; no miss is left for the CPU
        c.prefill = true;
        c.hits = hits;
    } else if (slots != nullptr) {
        if ((int64_t) slots->size() != c.n_expert) {
            GGML_ABORT("strata-moe: layer %d: the hot set has %zu experts, %s has %lld", c.layer, slots->size(),
                       c.up->name, (long long) c.n_expert);
        }
        split_hits(c, *slots);
        if (c.n_hits > 0) c.hits = hits;
    }

    // the misses grouped by expert: the experts in host memory now (the fork's tensors or the warm tier) in
    // first-use order, then the ones the residency reads from their files (STRATA_MOE_COLD=direct) in first-use
    // order, so the CPU starts on the ready ones while the others are read; each expert's pairs in pair order
    const std::vector<uint8_t>* cold = residency != nullptr ? &residency->prepare(c) : nullptr;
    auto expert_of = [&c](int64_t p) {
        return *reinterpret_cast<const int32_t*>(static_cast<const char*>(c.ids->data) + (p / c.n_used) * c.ids->nb[1] +
                                                 (p % c.n_used) * c.ids->nb[0]);
    };
    auto is_miss = [&c](int64_t p) { return c.hits == nullptr || c.hit_row[(size_t) p] < 0; };
    c.index_of.assign((size_t) c.n_expert, -1);
    for (const uint8_t want_cold : {0, 1}) {
        for (int64_t p = 0; p < n_pairs; ++p) {
            if (!is_miss(p)) continue;
            const int32_t e = expert_of(p);
            if ((cold != nullptr && (*cold)[(size_t) e]) != (bool) want_cold) continue;
            if (c.index_of[(size_t) e] < 0) {
                c.index_of[(size_t) e] = (int32_t) c.experts.size();
                c.experts.push_back(e);
                c.pair_begin.push_back(0);
            }
            ++c.pair_begin[(size_t) c.index_of[(size_t) e]];
        }
        if (!want_cold) c.n_ready = (int64_t) c.experts.size();
    }
    for (int64_t p = 0; p < n_pairs; ++p) {
        if (is_miss(p)) c.miss_pairs.push_back((int32_t) p);
    }
    // counts to offsets; filling moves each expert's offset to its end, the shift puts the offsets back
    int32_t offset = 0;
    for (int32_t& b : c.pair_begin) {
        const int32_t n = b;
        b = offset;
        offset += n;
    }
    c.pair_begin.push_back(offset);
    c.expert_pairs.resize((size_t) offset);
    for (const int32_t p : c.miss_pairs) {
        int32_t& cursor = c.pair_begin[(size_t) c.index_of[(size_t) expert_of(p)]];
        c.expert_pairs[(size_t) cursor++] = p;
    }
    for (size_t k = c.experts.size(); k > 0; --k) c.pair_begin[k] = c.pair_begin[k - 1];
    c.pair_begin[0] = 0;

    if (residency != nullptr) {
        c.residency = residency;
        residency->bind(c);
    } else {
        const size_t n = c.experts.size();
        c.base_up.resize(n);
        c.base_gate.resize(n);
        c.base_down.resize(n);
        for (size_t k = 0; k < n; ++k) {
            c.base_up[k] = row_ptr(c.up, c.experts[k], 0);
            c.base_gate[k] = row_ptr(c.gate, c.experts[k], 0);
            c.base_down[k] = row_ptr(c.down, c.experts[k], 0);
        }
        c.round_end.assign(1, (int64_t) n);
    }
    c.prepare_rounds();

    c.qx_up.resize((size_t) c.n_tokens * c.up_fmt.act_bytes);
    // gate shares up's quantized input when both take the same activation type
    if (c.gate_fmt.act_type != c.up_fmt.act_type) c.qx_gate.resize((size_t) c.n_tokens * c.gate_fmt.act_bytes);
    c.g.resize((size_t) (n_pairs * c.n_ff));
    c.u.resize(c.g.size());
    c.qh.resize((size_t) n_pairs * c.down_fmt.act_bytes);
    c.y.resize((size_t) (n_pairs * c.n_embd));

    if (c.kernel == CpuKernel::kFast) {
        // every supported type takes Q8_K, so gate and up share the input rows
        const size_t n = c.expert_pairs.size();
        c.act_x.resize(n);
        c.act_h.resize(n);
        c.out_g.resize(n);
        c.out_u.resize(n);
        c.out_y.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const int64_t p = c.expert_pairs[i];
            c.act_x[i] = c.qx_up.data() + (p / c.n_used) * c.up_fmt.act_bytes;
            c.act_h[i] = c.qh.data() + p * c.down_fmt.act_bytes;
            c.out_g[i] = c.g.data() + p * c.n_ff;
            c.out_u[i] = c.u.data() + p * c.n_ff;
            c.out_y[i] = c.y.data() + p * c.n_embd;
        }
    }
}

// The calls in flight, by output tensor, and the finished ones kept for reuse (as many as were ever in flight at
// once, normally one); in steady state neither vector allocates.
std::mutex g_calls_mutex;
std::vector<std::pair<const ggml_tensor*, std::unique_ptr<Call>>> g_active;
std::vector<std::unique_ptr<Call>> g_pool;

Call* join(const ggml_tensor* dst, int nth, float limit, HitPath* hits, Residency* residency, CpuKernel kernel) {
    std::lock_guard<std::mutex> lock(g_calls_mutex);
    for (const auto& [t, c] : g_active) {
        if (t == dst) return c.get();
    }
    std::unique_ptr<Call> c;
    if (g_pool.empty()) {
        c = std::make_unique<Call>();
    } else {
        c = std::move(g_pool.back());
        g_pool.pop_back();
    }
    c->reset(nth, kernel);
    init_call(*c, dst, limit, hits, residency);
    g_active.emplace_back(dst, std::move(c));
    return g_active.back().second.get();
}

// the last thread out records the call and returns it to the pool; every thread has joined by then (they all
// passed the barriers)
void leave(const ggml_tensor* dst, Call& c) {
    if (c.left.fetch_add(1, std::memory_order_acq_rel) + 1 == c.nth) {
        profile_record(c, ms_since(c.t0));
        counts_record(c);
        std::lock_guard<std::mutex> lock(g_calls_mutex);
        for (size_t i = 0; i < g_active.size(); ++i) {
            if (g_active[i].first != dst) continue;
            g_pool.push_back(std::move(g_active[i].second));
            g_active.erase(g_active.begin() + (std::ptrdiff_t) i);
            break;
        }
    }
}

// ggml_compute_forward_swiglu_clamp_f32's scalar loop, element for element
float swiglu_clamp(float g, float u, float limit) {
    g = std::min(g, limit);
    u = std::clamp(u, -limit, limit);
    return g / (1.0f + std::exp(-g)) * u;
}

// Work items are handed out through atomic counters, so a thread's ith does not matter: every output element is
// computed by one thread in a fixed order, whichever thread that is.
void run(Call& c, ggml_tensor* dst) {
    const int64_t n_embd = c.n_embd, n_ff = c.n_ff, n_used = c.n_used;
    const int64_t n_experts = (int64_t) c.experts.size();
    const int64_t n_miss = (int64_t) c.miss_pairs.size();
    const size_t up_bytes = c.up_fmt.act_bytes, down_bytes = c.down_fmt.act_bytes;
    const bool own_gate_act = !c.qx_gate.empty();
    const uint8_t* qx_gate = own_gate_act ? c.qx_gate.data() : c.qx_up.data();
    const size_t gate_bytes = own_gate_act ? c.gate_fmt.act_bytes : up_bytes;
    const bool fast = c.kernel == CpuKernel::kFast;
    // the barriers between the CPU phases; with STRATA_MOE_PROFILE=1 each thread adds the time it waited there
    const bool timed = profiling();
    auto phase_barrier = [&c, timed] {
        if (!timed) return c.barrier.wait();
        const auto t = std::chrono::steady_clock::now();
        c.barrier.wait();
        c.sync_ns.v.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count(),
            std::memory_order_relaxed);
    };

    // the first thread here queues the cold reads and the hits, then joins the CPU phases
    Residency* const residency = c.residency;
    const bool issuer = residency != nullptr && c.n_ready < n_experts && !c.cold_issued.exchange(true);
    if (issuer) residency->issue(c);
    const bool submitter = c.hits != nullptr && !c.submitted.exchange(true);
    if (submitter) {
        const auto t = std::chrono::steady_clock::now();
        c.hits->submit(c);
        c.submit_ms = ms_since(t);
    } else if (c.prefill) {
        // the submitter's staging copies, until it closes them
        c.stage.help();
    }

    // 0: the inputs in the activation types of the up / gate dot products
    for (int64_t t; n_miss > 0 && (t = c.next_quantize.v.fetch_add(1)) < c.n_tokens;) {
        const float* x = reinterpret_cast<const float*>(static_cast<const char*>(c.cur->data) + t * c.cur->nb[1]);
        c.up_fmt.act_quantize(x, c.qx_up.data() + t * up_bytes, n_embd);
        if (own_gate_act) c.gate_fmt.act_quantize(x, c.qx_gate.data() + t * gate_bytes, n_embd);
    }
    phase_barrier();

    // h is this thread's: one allocation per thread, ever
    thread_local std::vector<float> h_buffer;
    if (n_miss > 0 && h_buffer.size() < (size_t) n_ff) h_buffer.resize((size_t) n_ff);
    float* const h = h_buffer.data();
    const int64_t ff_rows = fast ? kFastRowsPerItem : kRowsPerItem;
    const int64_t ff_blocks = (n_ff + ff_rows - 1) / ff_rows;
    const int64_t embd_rows = fast ? kFastRowsPerItem : kRowsPerItem;
    const int64_t embd_blocks = (n_embd + embd_rows - 1) / embd_rows;
    const int64_t n_rounds = (int64_t) c.round_end.size();
    for (int64_t round = 0; round < n_rounds; ++round) {
        const int64_t k0 = round == 0 ? 0 : c.round_end[(size_t) round - 1], k1 = c.round_end[(size_t) round];
        PaddedCounter* const next = &c.round_next[(size_t) round * 3];

        // 1: gate . x and up . x, one block of rows of one expert per item
        for (int64_t item; (item = next[0].v.fetch_add(1)) < (k1 - k0) * ff_blocks;) {
            const int64_t k = k0 + item / ff_blocks, r0 = (item % ff_blocks) * ff_rows;
            const int64_t r1 = std::min(r0 + ff_rows, n_ff);
            const int32_t b = c.pair_begin[(size_t) k], n = c.pair_begin[(size_t) k + 1] - b;
            if (k >= c.n_ready) {
                residency->wait(c, k, 0);
                residency->wait(c, k, 1);
            }
            const char* const up_w = c.base_up[(size_t) k];
            const char* const gate_w = c.base_gate[(size_t) k];
            if (fast) {
                fast_kernel_rows(c.gate->type, gate_w, c.gate->nb[1], r0, r1, n_embd, &c.act_x[(size_t) b],
                                 &c.out_g[(size_t) b], n);
                fast_kernel_rows(c.up->type, up_w, c.up->nb[1], r0, r1, n_embd, &c.act_x[(size_t) b],
                                 &c.out_u[(size_t) b], n);
                continue;
            }
            for (int64_t r = r0; r < r1; ++r) {
                const char* gate_row = gate_w + r * c.gate->nb[1];
                const char* up_row = up_w + r * c.up->nb[1];
                for (int32_t i = b; i < b + n; ++i) {
                    const int64_t p = c.expert_pairs[(size_t) i];
                    const int64_t t = p / n_used;
                    c.g[(size_t) (p * n_ff + r)] = row_dot(c.gate_fmt, n_embd, gate_row, qx_gate + t * gate_bytes);
                    c.u[(size_t) (p * n_ff + r)] = row_dot(c.up_fmt, n_embd, up_row, c.qx_up.data() + t * up_bytes);
                }
            }
        }
        phase_barrier();

        // 2: h = SwiGLU(gate, up), in down's activation type, for the round's pairs
        const int64_t i0 = c.pair_begin[(size_t) k0], n_round_pairs = c.pair_begin[(size_t) k1] - i0;
        for (int64_t i; (i = next[1].v.fetch_add(1)) < n_round_pairs;) {
            const int64_t p = c.expert_pairs[(size_t) (i0 + i)];
            const float* g = c.g.data() + p * n_ff;
            const float* u = c.u.data() + p * n_ff;
            if (c.limit > 0.0f) {
                for (int64_t j = 0; j < n_ff; ++j) h[j] = swiglu_clamp(g[j], u[j], c.limit);
            } else {
                ggml_vec_swiglu_f32((int) n_ff, h, g, u);
            }
            c.down_fmt.act_quantize(h, c.qh.data() + p * down_bytes, n_ff);
        }
        phase_barrier();

        // 3: y = down . h
        for (int64_t item; (item = next[2].v.fetch_add(1)) < (k1 - k0) * embd_blocks;) {
            const int64_t k = k0 + item / embd_blocks, r0 = (item % embd_blocks) * embd_rows;
            const int64_t r1 = std::min(r0 + embd_rows, n_embd);
            const int32_t b = c.pair_begin[(size_t) k], n = c.pair_begin[(size_t) k + 1] - b;
            if (k >= c.n_ready) residency->wait(c, k, 2);
            const char* const down_w = c.base_down[(size_t) k];
            if (fast) {
                fast_kernel_rows(c.down->type, down_w, c.down->nb[1], r0, r1, n_ff, &c.act_h[(size_t) b],
                                 &c.out_y[(size_t) b], n);
                continue;
            }
            for (int64_t r = r0; r < r1; ++r) {
                const char* down_row = down_w + r * c.down->nb[1];
                for (int32_t i = b; i < b + n; ++i) {
                    const int64_t p = c.expert_pairs[(size_t) i];
                    c.y[(size_t) (p * n_embd + r)] = row_dot(c.down_fmt, n_ff, down_row, c.qh.data() + p * down_bytes);
                }
            }
        }
        phase_barrier();
        // every thread is past the round: its staging slots take the reads of round + 2
        if (residency != nullptr && !c.round_claimed[(size_t) round].exchange(true)) residency->round_done(c, round);
    }
    if (issuer) residency->finish(c);
    if (!c.cpu_timed.exchange(true)) c.cpu_ms = ms_since(c.t0);

    // the one sync of the hit path per call; every thread waits at the barrier until its output is there
    if (c.hits != nullptr) {
        if (submitter) {
            const auto t = std::chrono::steady_clock::now();
            c.hit_yw = c.hits->wait(c);
            c.wait_ms = ms_since(t);
        }
        c.barrier.wait();
    }

    // 4: out = sum over the slots, left to right, of y * w (a hit's y * w from the hit path)
    const int64_t out_blocks = (n_embd + kOutRowsPerItem - 1) / kOutRowsPerItem;
    for (int64_t item; (item = c.next_combine.v.fetch_add(1)) < c.n_tokens * out_blocks;) {
        const int64_t t = item / out_blocks, r0 = (item % out_blocks) * kOutRowsPerItem;
        const int64_t r1 = std::min(r0 + kOutRowsPerItem, n_embd);
        float* out = reinterpret_cast<float*>(static_cast<char*>(dst->data) + t * dst->nb[1]);
        for (int64_t s = 0; s < n_used; ++s) {
            const int64_t p = t * n_used + s;
            const int32_t hit = c.hits != nullptr ? c.hit_row[(size_t) p] : -1;
            if (hit >= 0) {
                const float* yw = c.hit_yw + (size_t) hit * n_embd;
                for (int64_t r = r0; r < r1; ++r) out[r] = s == 0 ? yw[r] : out[r] + yw[r];
                continue;
            }
            const float w = *reinterpret_cast<const float*>(static_cast<const char*>(c.weights->data) +
                                                            t * c.weights->nb[2] + s * c.weights->nb[1]);
            const float* y = c.y.data() + p * n_embd;
            for (int64_t r = r0; r < r1; ++r) {
                const float v = y[r] * w;
                out[r] = s == 0 ? v : out[r] + v;
            }
        }
    }
}

}  // namespace

void Call::reset(int n_threads, CpuKernel cpu_kernel) {
    nth = n_threads;
    kernel = cpu_kernel;
    barrier.reset(n_threads);
    left.store(0, std::memory_order_relaxed);
    next_quantize.v.store(0, std::memory_order_relaxed);
    next_combine.v.store(0, std::memory_order_relaxed);
    submitted.store(false, std::memory_order_relaxed);
    cold_issued.store(false, std::memory_order_relaxed);
    cold_left.store(0, std::memory_order_relaxed);
    cold_end_ns.store(0, std::memory_order_relaxed);
    cold_wait_ns.v.store(0, std::memory_order_relaxed);
    residency = nullptr;
    n_ready = 0;
    round_end.clear();
    warm_pairs = cold_pairs = warm_reads = cold_reads = 0;
    warm_bytes = cold_bytes = 0;
    cold_mapped = false;
    cold_ms = 0.0;
    cpu_timed.store(false, std::memory_order_relaxed);
    sync_ns.v.store(0, std::memory_order_relaxed);
    hits = nullptr;
    n_hits = 0;
    hit_width = 0;
    hit_yw = nullptr;
    experts.clear();
    pair_begin.clear();
    expert_pairs.clear();
    miss_pairs.clear();
    qx_gate.clear();
    hit_tokens.clear();
    hit_row.clear();
    submit_ms = gpu_ms = cpu_ms = wait_ms = 0.0;
    prefill = false;
    stage.reset();
    prefill_passes = 0;
    prefill_bytes = prefill_staged_bytes = 0;
    prefill_copy_ms = prefill_compute_ms = prefill_gpu_ms = prefill_stage_ms = prefill_stage_wait_ms = 0.0;
}

void Call::prepare_rounds() {
    const size_t n = round_end.size();
    if (n > round_capacity) {
        round_next = std::make_unique<PaddedCounter[]>(n * 3);
        round_claimed = std::make_unique<std::atomic<bool>[]>(n);
        round_capacity = n;
    }
    for (size_t i = 0; i < n * 3; ++i) round_next[i].v.store(0, std::memory_order_relaxed);
    for (size_t i = 0; i < n; ++i) round_claimed[i].store(false, std::memory_order_relaxed);
}

void forward(ggml_tensor* dst, int nth, float limit, HitPath* hits, Residency* residency, CpuKernel kernel) {
    Call* const call = join(dst, nth, limit, hits, residency, kernel);
    run(*call, dst);
    leave(dst, *call);
}

}  // namespace strata::dsv41
