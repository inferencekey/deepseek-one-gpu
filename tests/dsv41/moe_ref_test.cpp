// tests/dsv41/moe_ref_test.cpp - strata_moe_forward (the CPU reference) against the stock expert path of the
// fork's build_moe_ffn, both on ggml's CPU backend, at DeepSeek-V4.1-Flash dims: n_embd 5120, n_ff 2304, 384
// experts, 6 used, 1 and 32 tokens. No model, no GPU.
//
// The expert tensors are full size (384 experts) but only a pool of 12 experts is filled (random values quantized
// with ggml_quantize_chunk); the router picks 6 distinct experts per token from the pool. The stock side is the graph
// build_moe_ffn makes for a separate gate/up SiLU MoE: mul_mat_id up and gate, ggml_swiglu_clamp (or
// ggml_swiglu_split without a limit), mul_mat_id down, times the weights, the slot views added left to right.
// Prints the relative L2 ||ref - stock|| / ||stock|| per case and fails above kMaxRelL2.
#include "strata/dsv41/moe.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

constexpr int64_t kEmbd = 5120;
constexpr int64_t kFf = 2304;
constexpr int64_t kExperts = 384;
constexpr int64_t kUsed = 6;
constexpr int kPool = 12;
constexpr float kLimit = 10.0f;
constexpr int kThreads = 8;
// The reference runs the stock path's own ggml-cpu kernels in the stock order, so every case is bit-identical
// (measured 0.000e+00 in all 9). Any difference is a change in the reference or in the fork's CPU kernels: fail on it.
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

// One [n_per_row, n_rows, kExperts] expert tensor in a calloc'd buffer: the untouched experts stay unmapped zero
// pages, so a full-size Q4_K tensor (2.6 GB) costs only its filled experts.
struct ExpertTensor {
    ggml_tensor* t = nullptr;
    void* data = nullptr;
};

ExpertTensor make_experts(ggml_context* ctx, ggml_type type, int64_t n_per_row, int64_t n_rows, const char* name,
                          const std::vector<int32_t>& pool, float amplitude, uint64_t seed) {
    ExpertTensor x;
    x.t = ggml_new_tensor_3d(ctx, type, n_per_row, n_rows, kExperts);
    ggml_set_name(x.t, name);
    x.data = std::calloc(1, ggml_nbytes(x.t));
    if (x.data == nullptr) {
        std::fprintf(stderr, "FAIL: calloc %zu bytes for %s\n", ggml_nbytes(x.t), name);
        std::exit(1);
    }
    x.t->data = x.data;
    const size_t expert_bytes = x.t->nb[2];
    std::vector<std::thread> workers;
    for (int w = 0; w < kThreads; ++w) {
        workers.emplace_back([&, w] {
            std::vector<float> src((size_t) (n_per_row * n_rows));
            for (size_t k = (size_t) w; k < pool.size(); k += kThreads) {
                Rng rng{seed * 1000003u + (uint64_t) pool[k]};
                for (float& v : src) v = rng.uniform(-amplitude, amplitude);
                ggml_quantize_chunk(type, src.data(), static_cast<char*>(x.data) + (size_t) pool[k] * expert_bytes, 0,
                                    n_rows, n_per_row, nullptr);
            }
        });
    }
    for (std::thread& t : workers) t.join();
    return x;
}

struct TypeSet {
    ggml_type type;
    ExpertTensor up, gate, down;
};

struct Case {
    const TypeSet* gate_up;
    const TypeSet* down;
    int n_tokens;
    bool clamp;
};

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void compute(ggml_context* ctx, ggml_tensor* out) {
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, kThreads) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: graph compute for %s\n", out->name);
        std::exit(1);
    }
}

// true when the case passes
bool run_case(const Case& c, const std::vector<int32_t>& pool, uint64_t seed) {
    ggml_init_params params = {/*.mem_size =*/ (size_t) 256 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    ggml_context* ctx = ggml_init(params);
    const int64_t n_tok = c.n_tokens;

    Rng rng{seed};
    ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbd, n_tok);
    ggml_set_name(cur, "ffn_norm");
    for (int64_t i = 0; i < kEmbd * n_tok; ++i) ggml_get_data_f32(cur)[i] = rng.uniform(-1.0f, 1.0f);

    ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kUsed, n_tok);
    ggml_set_name(ids, "ffn_moe_topk");
    ggml_tensor* weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kUsed, n_tok);
    ggml_set_name(weights, "ffn_moe_weights_scaled");
    for (int64_t t = 0; t < n_tok; ++t) {
        std::vector<int32_t> pick = pool;
        for (int i = kPool - 1; i > 0; --i) std::swap(pick[(size_t) i], pick[rng.next() % (uint64_t) (i + 1)]);
        for (int64_t s = 0; s < kUsed; ++s) {
            static_cast<int32_t*>(ids->data)[t * kUsed + s] = pick[(size_t) s];
            ggml_get_data_f32(weights)[t * kUsed + s] = rng.uniform(0.05f, 0.5f);
        }
    }

    ggml_tensor* up_exps = c.gate_up->up.t;
    ggml_tensor* gate_exps = c.gate_up->gate.t;
    ggml_tensor* down_exps = c.down->down.t;

    // stock: build_moe_ffn's expert path (src/llama-graph.cpp, separate gate/up, LLM_FFN_SILU)
    ggml_tensor* cur3 = ggml_reshape_3d(ctx, cur, kEmbd, 1, n_tok);
    ggml_tensor* up = ggml_mul_mat_id(ctx, up_exps, cur3, ids);
    ggml_tensor* gate = ggml_mul_mat_id(ctx, gate_exps, cur3, ids);
    ggml_tensor* act = c.clamp ? ggml_swiglu_clamp(ctx, gate, up, kLimit) : ggml_swiglu_split(ctx, gate, up);
    ggml_tensor* experts = ggml_mul_mat_id(ctx, down_exps, act, ids);
    experts = ggml_mul(ctx, experts, weights);
    ggml_tensor* stock = ggml_view_2d(ctx, experts, kEmbd, n_tok, experts->nb[2], 0);
    for (int64_t i = 1; i < kUsed; ++i) {
        stock = ggml_add(ctx, stock, ggml_view_2d(ctx, experts, kEmbd, n_tok, experts->nb[2], i * experts->nb[1]));
    }
    ggml_set_name(stock, "ffn_moe_out stock");

    ggml_tensor* args[] = {cur, ids, weights, up_exps, gate_exps, down_exps};
    ggml_tensor* ref = ggml_custom_4d(ctx, GGML_TYPE_F32, kEmbd, n_tok, 1, 1, args, 6, strata_moe_forward,
                                      GGML_N_TASKS_MAX, nullptr);
    strata_moe_set_clamp(ref, c.clamp ? kLimit : 0.0f);
    ggml_set_name(ref, "ffn_moe_out strata");

    auto t0 = std::chrono::steady_clock::now();
    compute(ctx, stock);
    const double stock_ms = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    compute(ctx, ref);
    const double ref_ms = ms_since(t0);

    // how much of the clamp the case exercises: gate values above the limit, up values outside +-limit
    int64_t gate_over = 0, up_over = 0;
    const int64_t n_h = kFf * kUsed * n_tok;
    for (int64_t i = 0; i < n_h; ++i) {
        gate_over += ggml_get_data_f32(gate)[i] > kLimit;
        up_over += std::fabs(ggml_get_data_f32(up)[i]) > kLimit;
    }

    double diff2 = 0.0, ref2 = 0.0, max_abs = 0.0, worst_token = 0.0;
    bool finite = true;
    for (int64_t t = 0; t < n_tok; ++t) {
        double tok_diff2 = 0.0, tok_ref2 = 0.0;
        for (int64_t i = t * kEmbd; i < (t + 1) * kEmbd; ++i) {
            const double a = ggml_get_data_f32(stock)[i], b = ggml_get_data_f32(ref)[i];
            finite = finite && std::isfinite(a) && std::isfinite(b);
            tok_diff2 += (b - a) * (b - a);
            tok_ref2 += a * a;
            max_abs = std::max(max_abs, std::fabs(b - a));
        }
        diff2 += tok_diff2;
        ref2 += tok_ref2;
        worst_token = std::max(worst_token, std::sqrt(tok_diff2 / tok_ref2));
    }
    const double rel = std::sqrt(diff2 / ref2);
    const bool pass = finite && ref2 > 0.0 && rel <= kMaxRelL2;
    std::printf("%-4s gate/up %-4s down %-4s tokens %2d %-8s rel L2 %.3e  worst token %.3e  max |diff| %.3e  "
                "||stock|| %.3e  gate>limit %5.2f%% |up|>limit %5.2f%%  stock %6.1f ms  ref %6.1f ms\n",
                pass ? "ok" : "FAIL", ggml_type_name(c.gate_up->type), ggml_type_name(c.down->type), c.n_tokens,
                c.clamp ? "clamp 10" : "no clamp", rel, worst_token, max_abs, std::sqrt(ref2), 100.0 * gate_over / n_h,
                100.0 * up_over / n_h, stock_ms, ref_ms);
    ggml_free(ctx);
    return pass;
}

}  // namespace

int main() {
    ggml_cpu_init();
    // the pool spans the expert range, both ends included
    std::vector<int32_t> pool = {0, 383, 7, 64, 100, 129, 191, 200, 255, 301, 342, 377};

    ggml_init_params params = {/*.mem_size =*/ 16 * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr,
                               /*.no_alloc =*/ true};
    ggml_context* wctx = ggml_init(params);
    // amplitudes: gate/up pre-activations come out with a std of about 6 against inputs in [-1, 1], so the clamp
    // at 10 is active for some of them
    std::vector<TypeSet> sets;
    sets.reserve(3);
    const auto t0 = std::chrono::steady_clock::now();
    for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K}) {
        TypeSet s;
        s.type = type;
        s.up = make_experts(wctx, type, kEmbd, kFf, "blk.0.ffn_up_exps.weight", pool, 0.25f, 1 + type);
        s.gate = make_experts(wctx, type, kEmbd, kFf, "blk.0.ffn_gate_exps.weight", pool, 0.25f, 2 + type);
        s.down = make_experts(wctx, type, kFf, kEmbd, "blk.0.ffn_down_exps.weight", pool, 0.05f, 3 + type);
        sets.push_back(s);
    }
    std::printf("dsv41_moe_ref_test: n_embd %lld n_ff %lld experts %lld (%d filled) used %lld, %d threads, "
                "weights quantized in %.1f s\n",
                (long long) kEmbd, (long long) kFf, (long long) kExperts, kPool, (long long) kUsed, kThreads,
                ms_since(t0) / 1000.0);

    const TypeSet &q2 = sets[0], &q3 = sets[1], &q4 = sets[2];
    const Case cases[] = {
        {&q2, &q2, 1, true},  {&q2, &q2, 32, true}, {&q3, &q3, 1, true},  {&q3, &q3, 32, true},
        {&q4, &q4, 1, true},  {&q4, &q4, 32, true}, {&q2, &q3, 1, true},  {&q2, &q3, 32, true},
        {&q4, &q4, 32, false},
    };
    int failed = 0;
    uint64_t seed = 42;
    for (const Case& c : cases) failed += !run_case(c, pool, seed++);

    for (TypeSet& s : sets) {
        std::free(s.up.data);
        std::free(s.gate.data);
        std::free(s.down.data);
    }
    ggml_free(wctx);
    std::printf("dsv41_moe_ref_test: %s (%d of %zu cases above rel L2 %.0e)\n", failed ? "FAIL" : "PASS", failed,
                sizeof(cases) / sizeof(cases[0]), kMaxRelL2);
    return failed ? 1 : 0;
}
