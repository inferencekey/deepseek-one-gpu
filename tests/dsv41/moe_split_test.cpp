// tests/dsv41/moe_split_test.cpp - the hit/miss split and the combine of strata_moe_forward (moe_op.h), GPU
// free. The GPU tier is replaced by a CPU stand-in that computes each hit slot's y * w with the CPU kernel of the
// call itself (a one-slot call of the same op on the same token and expert), so with any resident set the output must
// equal the all-CPU output bit for bit: a wrong row, slot or order in the split or the combine shows up as a
// difference. The stand-in also counts hit tokens whose slots repeat, which ggml-cuda's mul_mat_id cannot take
// (moe_ref.cpp, split_hits): there must be none. A second run with the stand-in's rows negated checks that the
// hit rows are the ones the combine reads.
//
// DeepSeek-V4.1-Flash dims and types: n_embd 5120, n_ff 2304, 384 experts (a pool of 12 filled), 6 used, gate/up
// Q2_K, down Q3_K, SwiGLU clamp 10; 1 and 32 tokens; resident sets of 0, 3, 6, 9 and 12 of the pool experts; the
// misses on the reference and on the fast CPU kernel (STRATA_MOE_CPU's two values).
#include "cpu_kernel.h"
#include "moe_op.h"
#include "strata/dsv41/moe.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using strata::dsv41::Call;
using strata::dsv41::CpuKernel;
using strata::dsv41::HitPath;

constexpr int64_t kEmbd = 5120;
constexpr int64_t kFf = 2304;
constexpr int64_t kExperts = 384;
constexpr int64_t kUsed = 6;
constexpr float kLimit = 10.0f;
constexpr int kThreads = 8;
const std::vector<int32_t> kPool = {0, 383, 7, 64, 100, 129, 191, 200, 255, 301, 342, 377};

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

ggml_tensor* make_experts(ggml_context* ctx, ggml_type type, int64_t n_per_row, int64_t n_rows, const char* name,
                          float amplitude, uint64_t seed) {
    ggml_tensor* t = ggml_new_tensor_3d(ctx, type, n_per_row, n_rows, kExperts);
    ggml_set_name(t, name);
    // calloc: the experts outside the pool stay unmapped zero pages
    t->data = std::calloc(1, ggml_nbytes(t));
    if (t->data == nullptr) {
        std::fprintf(stderr, "FAIL: calloc %zu bytes for %s\n", ggml_nbytes(t), name);
        std::exit(1);
    }
    std::vector<float> src((size_t) (n_per_row * n_rows));
    for (const int32_t e : kPool) {
        Rng rng{seed * 1000003u + (uint64_t) e};
        for (float& v : src) v = rng.uniform(-amplitude, amplitude);
        ggml_quantize_chunk(type, src.data(), static_cast<char*>(t->data) + (size_t) e * t->nb[2], 0, n_rows,
                            n_per_row, nullptr);
    }
    return t;
}

// The GPU tier's place in the split: the resident experts of layer 0 and, per hit row, y * w from a one-slot call
// of the CPU reference on the row's token, expert and weight.
class CpuStandIn final : public HitPath {
public:
    std::vector<int32_t> slot_of = std::vector<int32_t>((size_t) kExperts, -1);
    bool negate = false;
    CpuKernel kernel = CpuKernel::kRef;  // the outer call's
    int64_t submits = 0, rows = 0, repeated = 0;  // repeated: hit tokens with a slot twice (ggml-cuda needs distinct)

    const std::vector<int32_t>* slots(const Call& c) override { return c.layer == 0 ? &slot_of : nullptr; }

    void submit(Call& c) override {
        ++submits;
        const int64_t n_rows = (int64_t) c.hit_tokens.size() * c.hit_width;
        for (size_t j = 0; j < c.hit_tokens.size(); ++j) {
            std::vector<int32_t> token_slots(c.hit_slots.begin() + (int64_t) j * c.hit_width,
                                             c.hit_slots.begin() + (int64_t) (j + 1) * c.hit_width);
            std::sort(token_slots.begin(), token_slots.end());
            repeated += std::adjacent_find(token_slots.begin(), token_slots.end()) != token_slots.end();
        }
        out_.assign((size_t) (n_rows * c.n_embd), 0.0f);
        ggml_init_params params = {/*.mem_size =*/ 8 * ggml_tensor_overhead() + 4 * (size_t) c.n_embd * sizeof(float),
                                   /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
        for (int64_t row = 0; row < n_rows; ++row) {
            const int64_t t = c.hit_tokens[(size_t) (row / c.hit_width)];
            ggml_context* ctx = ggml_init(params);
            ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.n_embd, 1);
            std::copy_n(reinterpret_cast<const float*>(static_cast<const char*>(c.cur->data) + t * c.cur->nb[1]),
                        c.n_embd, ggml_get_data_f32(cur));
            ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
            static_cast<int32_t*>(ids->data)[0] = c.hit_experts[(size_t) row];
            ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1);
            ggml_get_data_f32(w)[0] = c.hit_weights[(size_t) row];
            ggml_tensor* args[] = {cur, ids, w, const_cast<ggml_tensor*>(c.up), const_cast<ggml_tensor*>(c.gate),
                                   const_cast<ggml_tensor*>(c.down)};
            // no graph: forward runs the op directly, single-threaded, from inside the outer call
            ggml_tensor* one = ggml_custom_4d(ctx, GGML_TYPE_F32, c.n_embd, 1, 1, 1, args, 6, nullptr, 1, nullptr);
            strata::dsv41::forward(one, 1, c.limit, nullptr, nullptr, kernel);
            float* dst = out_.data() + row * c.n_embd;
            for (int64_t r = 0; r < c.n_embd; ++r) dst[r] = negate ? -ggml_get_data_f32(one)[r] : ggml_get_data_f32(one)[r];
            ggml_free(ctx);
        }
        rows += n_rows;
    }

    const float* wait(Call&) override { return out_.data(); }

private:
    std::vector<float> out_;
};

HitPath* g_path = nullptr;
strata::dsv41::CpuKernel g_kernel = strata::dsv41::CpuKernel::kRef;

void split_op(ggml_tensor* dst, int, int nth, void*) {
    strata::dsv41::forward(dst, nth, strata_moe_clamp(dst), g_path, nullptr, g_kernel);
}

struct Inputs {
    ggml_context* ctx = nullptr;
    ggml_tensor *cur = nullptr, *ids = nullptr, *weights = nullptr;
    int64_t n_tok = 0;
};

Inputs make_inputs(int64_t n_tok, uint64_t seed) {
    Inputs in;
    in.n_tok = n_tok;
    ggml_init_params params = {/*.mem_size =*/ (size_t) 64 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    in.ctx = ggml_init(params);
    Rng rng{seed};
    in.cur = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, kEmbd, n_tok);
    for (int64_t i = 0; i < kEmbd * n_tok; ++i) ggml_get_data_f32(in.cur)[i] = rng.uniform(-1.0f, 1.0f);
    in.ids = ggml_new_tensor_2d(in.ctx, GGML_TYPE_I32, kUsed, n_tok);
    in.weights = ggml_new_tensor_3d(in.ctx, GGML_TYPE_F32, 1, kUsed, n_tok);
    for (int64_t t = 0; t < n_tok; ++t) {
        std::vector<int32_t> pick = kPool;
        for (size_t i = pick.size() - 1; i > 0; --i) std::swap(pick[i], pick[rng.next() % (uint64_t) (i + 1)]);
        for (int64_t s = 0; s < kUsed; ++s) {
            static_cast<int32_t*>(in.ids->data)[t * kUsed + s] = pick[(size_t) s];
            ggml_get_data_f32(in.weights)[t * kUsed + s] = rng.uniform(0.05f, 0.5f);
        }
    }
    return in;
}

std::vector<float> compute(const Inputs& in, ggml_tensor* up, ggml_tensor* gate, ggml_tensor* down, HitPath* path) {
    ggml_init_params params = {/*.mem_size =*/ (size_t) 16 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* args[] = {in.cur, in.ids, in.weights, up, gate, down};
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, kEmbd, in.n_tok, 1, 1, args, 6, split_op, GGML_N_TASKS_MAX,
                                      nullptr);
    strata_moe_set_clamp(out, kLimit);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    g_path = path;
    if (ggml_graph_compute_with_ctx(ctx, gf, kThreads) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: graph compute\n");
        std::exit(1);
    }
    std::vector<float> v(ggml_get_data_f32(out), ggml_get_data_f32(out) + kEmbd * in.n_tok);
    ggml_free(ctx);
    return v;
}

double rel_l2(const std::vector<float>& ref, const std::vector<float>& x) {
    double d2 = 0.0, r2 = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        d2 += ((double) x[i] - ref[i]) * ((double) x[i] - ref[i]);
        r2 += (double) ref[i] * ref[i];
    }
    return std::sqrt(d2 / r2);
}

}  // namespace

int main() {
    ggml_cpu_init();
    ggml_init_params params = {/*.mem_size =*/ 4 * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr,
                               /*.no_alloc =*/ true};
    ggml_context* wctx = ggml_init(params);
    ggml_tensor* up = make_experts(wctx, GGML_TYPE_Q2_K, kEmbd, kFf, "blk.0.ffn_up_exps.weight", 0.25f, 11);
    ggml_tensor* gate = make_experts(wctx, GGML_TYPE_Q2_K, kEmbd, kFf, "blk.0.ffn_gate_exps.weight", 0.25f, 12);
    ggml_tensor* down = make_experts(wctx, GGML_TYPE_Q3_K, kFf, kEmbd, "blk.0.ffn_down_exps.weight", 0.05f, 13);
    std::printf("dsv41_moe_split_test: n_embd %lld n_ff %lld experts %lld (%zu filled) used %lld, q2_K gate/up, "
                "q3_K down, clamp %g, %d threads\n",
                (long long) kEmbd, (long long) kFf, (long long) kExperts, kPool.size(), (long long) kUsed, kLimit,
                kThreads);

    int failed = 0, cases = 0;
    for (const CpuKernel kernel : {CpuKernel::kRef, CpuKernel::kFast}) {
        if (kernel == CpuKernel::kFast && !strata::dsv41::fast_kernel_cpu_ok()) {
            std::printf("FAIL: this CPU lacks AVX2, FMA or F16C, the fast kernel cannot run\n");
            ++failed;
            continue;
        }
        g_kernel = kernel;
        const char* name = kernel == CpuKernel::kFast ? "fast" : "ref";
        uint64_t seed = 7;
        for (const int64_t n_tok : {1, 32}) {
            const Inputs in = make_inputs(n_tok, seed++);
            const std::vector<float> all_cpu = compute(in, up, gate, down, nullptr);
            for (const size_t n_resident : {0, 3, 6, 9, 12}) {
                // a random resident set of the pool, in a random slot order
                Rng rng{seed++};
                std::vector<int32_t> pick = kPool;
                for (size_t i = pick.size() - 1; i > 0; --i) std::swap(pick[i], pick[rng.next() % (uint64_t) (i + 1)]);
                CpuStandIn path;
                path.kernel = kernel;
                for (size_t k = 0; k < n_resident; ++k) path.slot_of[(size_t) pick[k]] = (int32_t) (n_resident - 1 - k);
                int64_t want_hits = 0;
                for (int64_t i = 0; i < kUsed * n_tok; ++i) {
                    want_hits += path.slot_of[(size_t) static_cast<const int32_t*>(in.ids->data)[i]] >= 0;
                }

                const std::vector<float> split = compute(in, up, gate, down, &path);
                const double rel = rel_l2(all_cpu, split);
                const int64_t rows = path.rows;
                path.negate = true;
                const double rel_negated = rel_l2(all_cpu, compute(in, up, gate, down, &path));

                // with hits, one submit per call; the stand-in's rows cover every hit (plus padding) with no slot
                // twice in a token; negated rows must change the output exactly when there are hits
                const bool calls_ok = want_hits == 0 ? path.submits == 0 : path.submits == 2 && rows >= want_hits;
                const bool used_ok = want_hits == 0 ? rel_negated == 0.0 : rel_negated > 0.0;
                const bool pass = rel == 0.0 && calls_ok && used_ok && path.repeated == 0;
                failed += !pass;
                ++cases;
                std::printf("%-4s %-4s tokens %2lld resident %2zu/%zu hits %3lld of %3lld  rel L2 vs all-CPU %.3e  "
                            "stand-in rows %3lld  tokens with a repeated slot %lld  negated rows: rel L2 %.3e\n",
                            pass ? "ok" : "FAIL", name, (long long) n_tok, n_resident, kPool.size(),
                            (long long) want_hits, (long long) (kUsed * n_tok), rel, (long long) rows,
                            (long long) path.repeated, rel_negated);
            }
            ggml_free(in.ctx);
        }
    }

    std::free(up->data);
    std::free(gate->data);
    std::free(down->data);
    ggml_free(wctx);
    std::printf("dsv41_moe_split_test: %s (%d of %d cases failed; required: rel L2 0 against all-CPU)\n",
                failed ? "FAIL" : "PASS", failed, cases);
    return failed ? 1 : 0;
}
