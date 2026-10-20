// tests/dsv41/moe_cpu_fast_test.cpp - strata-moe's fast CPU kernel (STRATA_MOE_CPU=fast, cpu_kernel.h)
// against the reference (STRATA_MOE_CPU=ref), GPU free.
//
// 1. Rows: fast_kernel_rows of q2_K, q3_K, q4_K and q6_K rows against ggml-cpu's own vec_dot of the same row and
//    Q8_K activation, at row ranges that leave a tail of the kernel's 4-row tiles and token counts that leave a
//    tail of its 8-token tiles (1, 3, 8, 9, 17).
// 2. The op: strata::dsv41::forward with each kernel on the same inputs, every pair on the CPU, at
//    DeepSeek-V4.1-Flash dims (n_embd 5120, n_ff 2304, 384 experts, 6 used, gate/up q2_K, down q3_K, SwiGLU clamp
//    10, and the types swapped without the clamp) and OLMoE-1B-7B dims (n_embd 2048, n_ff 1024, 64 experts, 8 used,
//    q4_K gate/up with q4_K or q6_K down, no clamp, as olmoe-1b-7b-0924-q4_k_m.gguf has them; and q6_K gate/up with
//    the clamp), 1, 5 and 32 tokens. The router picks distinct experts per token from a pool of filled ones.
//
// The task's bound is relative L2 <= 1e-5 (fp32 accumulation order). The kernel runs ggml-cpu's AVX2 integer and
// float chain per (row, token) unchanged, so it is exact: the test requires 0 and prints the actual numbers, so any
// change that moves a bit shows up here first.
#include "cpu_kernel.h"
#include "moe_op.h"
#include "strata/dsv41/moe.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using strata::dsv41::CpuKernel;

constexpr int kThreads = 8;
constexpr double kMaxRelL2 = 0.0;

struct Rng {
    uint64_t s;
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * (float) (next() >> 40) * (1.0f / 16777216.0f); }
};

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------- 1. rows against ggml-cpu's vec_dot

bool rows_case(ggml_type type, int64_t n_per_row, int64_t n_rows, int64_t r0, int64_t r1, int64_t n_acts,
               uint64_t seed) {
    Rng rng{seed};
    std::vector<float> src((size_t) (n_per_row * n_rows));
    for (float& v : src) v = rng.uniform(-0.25f, 0.25f);
    const size_t row_bytes = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> w(row_bytes * (size_t) n_rows);
    ggml_quantize_chunk(type, src.data(), w.data(), 0, n_rows, n_per_row, nullptr);

    const ggml_type_traits_cpu* cpu = ggml_get_type_traits_cpu(type);
    const size_t act_bytes = ggml_row_size(GGML_TYPE_Q8_K, n_per_row);
    std::vector<uint8_t> acts(act_bytes * (size_t) n_acts);
    std::vector<float> x((size_t) n_per_row);
    for (int64_t t = 0; t < n_acts; ++t) {
        for (float& v : x) v = rng.uniform(-1.0f, 1.0f);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), acts.data() + t * act_bytes, n_per_row);
    }
    std::vector<const void*> act_ptrs((size_t) n_acts);
    std::vector<float> fast((size_t) (n_acts * n_rows), NAN);
    std::vector<float*> out_ptrs((size_t) n_acts);
    for (int64_t t = 0; t < n_acts; ++t) {
        act_ptrs[(size_t) t] = acts.data() + t * act_bytes;
        out_ptrs[(size_t) t] = fast.data() + t * n_rows;
    }
    strata::dsv41::fast_kernel_rows(type, reinterpret_cast<const char*>(w.data()), row_bytes, r0, r1, n_per_row,
                                    act_ptrs.data(), out_ptrs.data(), n_acts);

    int64_t differ = 0, touched = 0;
    for (int64_t t = 0; t < n_acts; ++t) {
        for (int64_t r = 0; r < n_rows; ++r) {
            const float got = fast[(size_t) (t * n_rows + r)];
            if (r < r0 || r >= r1) {
                touched += !std::isnan(got);  // rows outside [r0, r1) must stay unwritten
                continue;
            }
            float want = 0.0f;
            cpu->vec_dot((int) n_per_row, &want, 0, w.data() + (size_t) r * row_bytes, 0, act_ptrs[(size_t) t], 0, 1);
            differ += std::memcmp(&got, &want, sizeof(float)) != 0;
        }
    }
    const bool pass = differ == 0 && touched == 0;
    std::printf("%-4s rows %-4s n %4lld rows [%2lld, %3lld) of %3lld  tokens %2lld  results differing from ggml-cpu "
                "vec_dot %lld of %lld, rows outside the range written %lld\n",
                pass ? "ok" : "FAIL", ggml_type_name(type), (long long) n_per_row, (long long) r0, (long long) r1,
                (long long) n_rows, (long long) n_acts, (long long) differ, (long long) ((r1 - r0) * n_acts),
                (long long) touched);
    return pass;
}

// ---------------------------------------------------------------- 2. the op, fast against ref

struct Dims {
    const char* name;
    int64_t n_embd, n_ff, n_expert, n_used;
    std::vector<int32_t> pool;  // the filled experts
};

// One [n_per_row, n_rows, n_expert] expert tensor in a calloc'd buffer: the experts outside the pool stay unmapped
// zero pages.
ggml_tensor* make_experts(ggml_context* ctx, const Dims& d, ggml_type type, int64_t n_per_row, int64_t n_rows,
                          const char* name, float amplitude, uint64_t seed) {
    ggml_tensor* t = ggml_new_tensor_3d(ctx, type, n_per_row, n_rows, d.n_expert);
    ggml_set_name(t, name);
    t->data = std::calloc(1, ggml_nbytes(t));
    if (t->data == nullptr) {
        std::fprintf(stderr, "FAIL: calloc %zu bytes for %s\n", ggml_nbytes(t), name);
        std::exit(1);
    }
    std::vector<float> src((size_t) (n_per_row * n_rows));
    for (const int32_t e : d.pool) {
        Rng rng{seed * 1000003u + (uint64_t) e};
        for (float& v : src) v = rng.uniform(-amplitude, amplitude);
        ggml_quantize_chunk(type, src.data(), static_cast<char*>(t->data) + (size_t) e * t->nb[2], 0, n_rows,
                            n_per_row, nullptr);
    }
    return t;
}

CpuKernel g_kernel = CpuKernel::kRef;

void op(ggml_tensor* dst, int, int nth, void*) {
    strata::dsv41::forward(dst, nth, strata_moe_clamp(dst), nullptr, nullptr, g_kernel);
}

std::vector<float> compute(ggml_tensor* const* args, int64_t n_embd, int64_t n_tok, float limit, CpuKernel kernel,
                           double* ms) {
    ggml_init_params params = {/*.mem_size =*/ (size_t) 16 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* srcs[6];
    std::copy(args, args + 6, srcs);
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_tok, 1, 1, srcs, 6, op, GGML_N_TASKS_MAX, nullptr);
    strata_moe_set_clamp(out, limit);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    g_kernel = kernel;
    const auto t0 = std::chrono::steady_clock::now();
    if (ggml_graph_compute_with_ctx(ctx, gf, kThreads) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: graph compute\n");
        std::exit(1);
    }
    *ms = ms_since(t0);
    std::vector<float> v(ggml_get_data_f32(out), ggml_get_data_f32(out) + n_embd * n_tok);
    ggml_free(ctx);
    return v;
}

struct OpCase {
    ggml_type gate_up, down;
    float limit;
};

bool op_case(const Dims& d, const OpCase& oc, ggml_tensor* up, ggml_tensor* gate, ggml_tensor* down, int64_t n_tok,
             uint64_t seed) {
    ggml_init_params params = {/*.mem_size =*/ (size_t) 8 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    ggml_context* ctx = ggml_init(params);
    Rng rng{seed};
    ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.n_embd, n_tok);
    for (int64_t i = 0; i < d.n_embd * n_tok; ++i) ggml_get_data_f32(cur)[i] = rng.uniform(-1.0f, 1.0f);
    ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, d.n_used, n_tok);
    ggml_tensor* weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, d.n_used, n_tok);
    for (int64_t t = 0; t < n_tok; ++t) {
        std::vector<int32_t> pick = d.pool;
        for (size_t i = pick.size() - 1; i > 0; --i) std::swap(pick[i], pick[rng.next() % (uint64_t) (i + 1)]);
        for (int64_t s = 0; s < d.n_used; ++s) {
            static_cast<int32_t*>(ids->data)[t * d.n_used + s] = pick[(size_t) s];
            ggml_get_data_f32(weights)[t * d.n_used + s] = rng.uniform(0.05f, 0.5f);
        }
    }
    ggml_tensor* args[] = {cur, ids, weights, up, gate, down};
    double ref_ms = 0.0, fast_ms = 0.0;
    const std::vector<float> ref = compute(args, d.n_embd, n_tok, oc.limit, CpuKernel::kRef, &ref_ms);
    const std::vector<float> fast = compute(args, d.n_embd, n_tok, oc.limit, CpuKernel::kFast, &fast_ms);

    double diff2 = 0.0, ref2 = 0.0, max_abs = 0.0;
    int64_t differ = 0;
    bool finite = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        finite = finite && std::isfinite(ref[i]) && std::isfinite(fast[i]);
        const double e = (double) fast[i] - ref[i];
        diff2 += e * e;
        ref2 += (double) ref[i] * ref[i];
        max_abs = std::max(max_abs, std::fabs(e));
        differ += std::memcmp(&ref[i], &fast[i], sizeof(float)) != 0;
    }
    const double rel = std::sqrt(diff2 / ref2);
    const bool pass = finite && ref2 > 0.0 && rel <= kMaxRelL2;
    std::printf("%-4s op %-6s gate/up %-4s down %-4s tokens %2lld %-9s rel L2 fast vs ref %.3e  max |diff| %.3e  "
                "elements differing %lld of %zu  ||ref|| %.3e  ref %6.1f ms  fast %6.1f ms\n",
                pass ? "ok" : "FAIL", d.name, ggml_type_name(oc.gate_up), ggml_type_name(oc.down), (long long) n_tok,
                oc.limit > 0.0f ? "clamp 10" : "no clamp", rel, max_abs, (long long) differ, ref.size(),
                std::sqrt(ref2), ref_ms, fast_ms);
    ggml_free(ctx);
    return pass;
}

}  // namespace

int main() {
    ggml_cpu_init();
    if (!strata::dsv41::fast_kernel_cpu_ok()) {
        std::printf("dsv41_moe_cpu_fast_test: FAIL: this CPU lacks AVX2, FMA or F16C, the fast kernel cannot run\n");
        return 1;
    }
    int failed = 0, cases = 0;
    uint64_t seed = 1;

    std::printf("dsv41_moe_cpu_fast_test: rows against ggml-cpu's vec_dot (bit for bit)\n");
    for (const ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        for (const int64_t n_acts : {1, 3, 8, 9, 17}) {
            failed += !rows_case(type, 2304, 67, 1, 66, n_acts, seed++);
            ++cases;
        }
    }

    const float limit = 10.0f;
    const Dims dsv41 = {"dsv41", 5120, 2304, 384, 6, {0, 383, 7, 64, 100, 129, 191, 200, 255, 301, 342, 377}};
    const Dims olmoe = {"olmoe", 2048, 1024, 64, 8, {0, 63, 3, 9, 17, 22, 30, 31, 40, 47, 52, 58, 60, 61}};
    struct Group {
        const Dims* dims;
        std::vector<OpCase> cases;
    };
    const Group groups[] = {
        {&dsv41, {{GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, limit}, {GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, 0.0f}}},
        {&olmoe,
         {{GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 0.0f}, {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 0.0f},
          {GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, limit}}},
    };
    std::printf("dsv41_moe_cpu_fast_test: the op, fast against ref, every pair on the CPU, %d threads\n", kThreads);
    for (const Group& g : groups) {
        const Dims& d = *g.dims;
        for (const OpCase& oc : g.cases) {
            ggml_init_params params = {/*.mem_size =*/ 4 * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr,
                                       /*.no_alloc =*/ true};
            ggml_context* wctx = ggml_init(params);
            // amplitudes as in moe_ref_test.cpp: gate/up pre-activations with a std of a few units, so the clamp
            // at 10 is active for some of them
            ggml_tensor* up =
                make_experts(wctx, d, oc.gate_up, d.n_embd, d.n_ff, "blk.0.ffn_up_exps.weight", 0.25f, seed++);
            ggml_tensor* gate =
                make_experts(wctx, d, oc.gate_up, d.n_embd, d.n_ff, "blk.0.ffn_gate_exps.weight", 0.25f, seed++);
            ggml_tensor* down =
                make_experts(wctx, d, oc.down, d.n_ff, d.n_embd, "blk.0.ffn_down_exps.weight", 0.05f, seed++);
            for (const int64_t n_tok : {1, 5, 32}) {
                failed += !op_case(d, oc, up, gate, down, n_tok, seed++);
                ++cases;
            }
            std::free(up->data);
            std::free(gate->data);
            std::free(down->data);
            ggml_free(wctx);
        }
    }
    std::printf("dsv41_moe_cpu_fast_test: %s (%d of %d cases failed; required: bit for bit, rel L2 %.0e, against "
                "the task's bound 1e-5)\n",
                failed ? "FAIL" : "PASS", failed, cases, kMaxRelL2);
    return failed ? 1 : 0;
}
