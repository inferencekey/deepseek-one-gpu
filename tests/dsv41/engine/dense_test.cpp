// tests/dsv41/engine/dense_test.cpp - parity of the dense ops (include/strata/dsv41/engine/dense.hpp, the
// reference dense_ref.hpp, the engram hash engram.hpp).
//
//   dsv41_engine_dense_test --cases FILE [FILE ...]   (a) GPU free: the CPU reference against the stock fork's op
//                                                     dumps (lists of tools/dsv41/engine/dense_cases.py)
//   dsv41_engine_dense_test --gpu [--quick] [--mmq] [--dry-run] [--table-rows N]
//                                                     (b) CUDA against the CPU reference, dense_test_gpu.cu;
//                                                     --dry-run its host-side checks only (GPU free, verify.sh);
//                                                     N the real-dims engram table's rows (default 4M, 336 MiB)
//
// (a) Per dump, every ubatch, every layer: each op of SPEC.md sections 1, 2, 3, 6, 7 (shared expert) and 8 is
// computed by the reference in fork numerics (RefActs::Fork) from the dump's own inputs to that op, so a difference
// is that op's, and compared with the dump bit for bit:
//   1   embd (the token rows), hc_init (the copies), hc_pre_init (the one-hot);
//   2   the engram's rows (engram_rows from the ubatch's tokens and the ones before it in the sequence) gathered from
//       the GGUF's table and dequantized (engram_embd), engram_kv, engram_gate, engram_out;
//   3   hc_mixes, hc_pre, hc_post, hc_comb of both sublayers, the collapses hc_attn_pre (with the previous layer's FFN
//       pre, hc_pre_init at layer 0) and hc_ffn_pre, attn_norm and ffn_norm, the expansions hc_attn_post (from the
//       dump's attn_out) and l_last (from ffn_out = ffn_moe_out + ffn_shexp);
//   6   ffn_moe_logits, probs, probs_biased, the argsort (every position: the fork's std::sort order, ties included),
//       ffn_moe_topk, weights, weights_sum, sum_clamped, weights_norm, weights_scaled;
//   7   ffn_up, ffn_gate, ffn_swiglu_limited, ffn_shexp, ffn_out;
//   8   hc_out (the last token's unfused collapse), result_norm, result_output (q6_K).
// Then the engine's numerics on the same inputs (qmatmul's default paths for the quantized matrices, the engine's
// bf16 router and tie order): the worst rel L2 per matrix output against the dump, bounded by kEngineRelL2 (int8 per
// 32 values or f16 activations where the fork rounds to int8 per 256: about 1% per matmul on these random weights,
// a wrong row or layout O(1)), and the router's top-k sets that differ from the fork's (reported, not bounded: a
// near-tie of the biased probabilities can swap an expert).
#include "dense_test_common.hpp"
#include "dump_lists.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace e = strata::dsv41::engine;
using dumps::Dump;
using dumps::DumpFile;
using dumps::Ubatch;
using dumps::load_f32;
using dumps::load_i32;

namespace {

// ---------------------------------------------------------------- the model from its GGUF
struct Layer {
    e::RefHcWeights hc_attn, hc_ffn;
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    e::RefRouterWeights router;
    e::RefSharedWeights sh;
    int engram = -1;                   // engram index, -1 without
    e::RefEngramWeights eng;
    std::vector<float> eq, ek;         // engram_q, engram_k dequantized
    const uint8_t* table = nullptr;    // engram_embd rows
    int64_t table_rows = 0;
};
struct Model {
    std::unique_ptr<strata::GgufFile> g;
    e::DenseDims d;
    e::EngramHash eh;
    std::vector<Layer> layers;
    e::RefMatrix token_embd, output;
    const float* output_norm = nullptr;
};

double meta_num(const strata::GgufFile& g, const std::string& key) {
    const strata::MetaValue* v = g.get("deepseek41." + key);
    if (!v || !v->is_num()) die(g.path() + ": no numeric deepseek41." + key);
    return v->num();
}
const strata::MetaValue& meta_arr(const strata::GgufFile& g, const std::string& key) {
    const strata::MetaValue* v = g.get("deepseek41." + key);
    if (!v || v->type != strata::MetaType::ARRAY) die(g.path() + ": no array deepseek41." + key);
    return *v;
}

e::RefMatrix matrix(const strata::GgufFile& g, const std::string& name, int type) {
    const strata::TensorInfo* t = g.find(name);
    if (!t) die(g.path() + ": no tensor " + name);
    if ((int) t->type != type || t->shape.size() != 2)
        die(g.path() + ": " + name + " is " + t->type_name() + ", expected 2D of type " + std::to_string(type));
    return e::RefMatrix{type, (int64_t) t->shape[0], (int64_t) t->shape[1], g.tensor_data(*t), e::RefActs::Fork};
}
const float* vec_f32(const strata::GgufFile& g, const std::string& name, int64_t n) {
    const strata::TensorInfo* t = g.find(name);
    if (!t) die(g.path() + ": no tensor " + name);
    if (t->type != 0 || (int64_t) t->elements() != n)
        die(g.path() + ": " + name + " is " + t->type_name() + " of " + std::to_string(t->elements()) + " values, expected f32 of " + std::to_string(n));
    return reinterpret_cast<const float*>(g.tensor_data(*t));
}

Model load_model(const std::string& path) {
    Model m;
    m.g = std::make_unique<strata::GgufFile>(path);
    const strata::GgufFile& g = *m.g;
    const strata::MetaValue* arch = g.get("general.architecture");
    if (!arch || arch->s != "deepseek41") die(path + ": not a deepseek41 GGUF");
    e::DenseDims& d = m.d;
    d.n_embd = (int) meta_num(g, "embedding_length");
    d.n_expert = (int) meta_num(g, "expert_count");
    d.n_expert_used = (int) meta_num(g, "expert_used_count");
    d.n_ff = (int) meta_num(g, "expert_feed_forward_length");
    d.rms_eps = (float) meta_num(g, "attention.layer_norm_rms_epsilon");
    d.hc_eps = (float) meta_num(g, "hyper_connection.epsilon");
    d.sinkhorn_iters = (int) meta_num(g, "hyper_connection.sinkhorn_iterations");
    d.weights_norm = meta_num(g, "expert_weights_norm") != 0;
    d.weights_scale = (float) meta_num(g, "expert_weights_scale");
    d.engram_heads = (int) meta_num(g, "engram.head_count");
    d.engram_key = (int) meta_num(g, "engram.key_length");
    d.engram_ngram = (int) meta_num(g, "engram.max_ngram_size");
    if ((int) meta_num(g, "hyper_connection.count") != e::kHc) die(path + ": hyper_connection.count is not 4");
    if ((int) meta_num(g, "expert_gating_func") != 4) die(path + ": expert_gating_func is not 4 (sqrtsoftplus)");
    const int n_layer = (int) meta_num(g, "block_count");
    const strata::MetaValue& clamp = meta_arr(g, "swiglu_clamp_shexp");
    if (clamp.elem != strata::MetaType::F32 || (int64_t) clamp.count != n_layer || n_layer > 64) die(path + ": swiglu_clamp_shexp is not one f32 per layer");
    for (int i = 0; i < n_layer; ++i)
        if ((float) clamp.items[(size_t) i].f != 10.f) die(path + ": swiglu_clamp_shexp is not 10 on every layer (DenseDims has one limit)");
    d.swiglu_limit = 10.f;
    e::EngramHash& h = m.eh;
    h.n_gram = d.engram_ngram;
    h.heads = d.engram_heads;
    h.pad = (int64_t) meta_num(g, "engram.pad_id");
    // the full arrays (MetaValue::items keeps the first 64; the real token map has 129,280 entries)
    for (uint64_t v : strata::meta_int_array(meta_arr(g, "engram.layer_ids"))) h.layer_ids.push_back((int32_t) v);
    for (uint64_t v : strata::meta_int_array(meta_arr(g, "engram.token_map"))) h.token_map.push_back((int32_t) v);
    h.multipliers = strata::meta_int_array(meta_arr(g, "engram.multipliers"));
    h.primes = strata::meta_int_array(meta_arr(g, "engram.primes"));
    h.offsets = strata::meta_int_array(meta_arr(g, "engram.offsets"));
    h.check();
    m.token_embd = matrix(g, "token_embd.weight", e::kRefTypeQ2K);
    m.output = matrix(g, "output.weight", e::kRefTypeQ6K);
    m.output_norm = vec_f32(g, "output_norm.weight", d.n_embd);
    d.n_vocab = (int) m.output.rows;
    const int64_t E = d.n_embd;
    for (int i = 0; i < n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        Layer L;
        L.hc_attn = {matrix(g, b + "hc_attn_fn.weight", e::kRefTypeQ2K), vec_f32(g, b + "hc_attn_base.weight", e::kHcMix),
                     vec_f32(g, b + "hc_attn_scale.weight", 3)};
        L.hc_ffn = {matrix(g, b + "hc_ffn_fn.weight", e::kRefTypeQ2K), vec_f32(g, b + "hc_ffn_base.weight", e::kHcMix),
                    vec_f32(g, b + "hc_ffn_scale.weight", 3)};
        L.attn_norm = vec_f32(g, b + "attn_norm.weight", E);
        L.ffn_norm = vec_f32(g, b + "ffn_norm.weight", E);
        L.router = {matrix(g, b + "ffn_gate_inp.weight", e::kRefTypeBF16), vec_f32(g, b + "exp_probs_b.bias", d.n_expert)};
        L.sh = {matrix(g, b + "ffn_gate_shexp.weight", e::kRefTypeQ2K), matrix(g, b + "ffn_up_shexp.weight", e::kRefTypeQ2K),
                matrix(g, b + "ffn_down_shexp.weight", e::kRefTypeQ3K)};
        L.engram = h.index_of(i);
        if (L.engram >= 0) {
            L.eng.wkv = matrix(g, b + "engram_wkv.weight", e::kRefTypeQ2K);
            L.eq = e::ref_dequant(matrix(g, b + "engram_q.weight", e::kRefTypeQ2K));
            L.ek = e::ref_dequant(matrix(g, b + "engram_k.weight", e::kRefTypeQ2K));
            L.eng.q = L.eq.data();
            L.eng.k = L.ek.data();
            const e::RefMatrix t = matrix(g, b + "engram_embd.weight", e::kRefTypeQ2K);
            if (t.k != d.engram_key) die(path + ": engram_embd rows of " + std::to_string(t.k));
            L.table = static_cast<const uint8_t*>(t.data);
            L.table_rows = t.rows;
            if (L.table_rows < h.rows_needed(L.engram)) die(path + ": engram_embd has too few rows for its buckets");
        }
        m.layers.push_back(std::move(L));
    }
    return m;
}

// the same weights in the engine's numerics: every quantized matrix on qmatmul's default path for the call
Model& engine_numerics(Model& m) {
    auto eng = [](e::RefMatrix& w) { w.acts = e::RefActs::Auto; };
    for (Layer& L : m.layers) {
        eng(L.hc_attn.fn); eng(L.hc_ffn.fn); eng(L.router.gate_inp); eng(L.sh.gate); eng(L.sh.up); eng(L.sh.down);
        if (L.engram >= 0) eng(L.eng.wkv);
    }
    eng(m.output);
    return m;
}

// ---------------------------------------------------------------- comparisons
struct OpStats {
    int64_t files = 0, values = 0, identical = 0;
    double max_abs = 0;
};
struct Run {
    std::map<std::string, OpStats> ops;
    int64_t diffs_shown = 0;
    bool ok = true;
};

uint32_t fbits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }

void cmp_bits(Run& run, const std::string& op, const std::string& where, const float* a, const float* b, int64_t n) {
    OpStats& s = run.ops[op];
    ++s.files;
    for (int64_t i = 0; i < n; ++i) {
        ++s.values;
        if (fbits(a[i]) == fbits(b[i]) || (std::isnan(a[i]) && std::isnan(b[i]))) { ++s.identical; continue; }
        s.max_abs = std::max(s.max_abs, (double) std::fabs(a[i] - b[i]));
        if (run.diffs_shown++ < 12)
            std::printf("diff %s %s: [%lld] reference %.9g (0x%08x) fork %.9g (0x%08x)\n", op.c_str(), where.c_str(), (long long) i,
                        a[i], fbits(a[i]), b[i], fbits(b[i]));
    }
}
void cmp_i32(Run& run, const std::string& op, const std::string& where, const int32_t* a, const int32_t* b, int64_t n) {
    OpStats& s = run.ops[op];
    ++s.files;
    for (int64_t i = 0; i < n; ++i) {
        ++s.values;
        if (a[i] == b[i]) { ++s.identical; continue; }
        if (run.diffs_shown++ < 12)
            std::printf("diff %s %s: [%lld] reference %d fork %d\n", op.c_str(), where.c_str(), (long long) i, a[i], b[i]);
    }
}

bool report(const Run& run, const char* name) {
    bool ok = run.ok;
    std::printf("  %s:\n", name);
    for (const auto& [op, s] : run.ops) {
        const bool good = s.identical == s.values;
        ok = ok && good;
        std::printf("    %-30s %5lld calls %9lld values, %s", op.c_str(), (long long) s.files, (long long) s.values,
                    good ? "bit-identical" : "DIFFER");
        if (!good) std::printf(" (%lld differ, max |err| %.3g)", (long long) (s.values - s.identical), s.max_abs);
        std::printf("\n");
    }
    return ok;
}

double rel_l2(const float* a, const float* b, int64_t n) {
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

struct Drift {
    std::map<std::string, std::pair<double, std::string>> worst;   // op -> (rel L2, where)
    int64_t sets = 0, sets_differ = 0;
    void add(const std::string& op, const std::string& where, const float* a, const float* b, int64_t n) {
        const double r = rel_l2(a, b, n);
        auto& w = worst[op];
        if (r >= w.first) w = {r, where};
    }
};

std::vector<float> seg(const std::vector<float>& v, int64_t off, int64_t n) {
    return std::vector<float>(v.begin() + off, v.begin() + off + n);
}

// the tokens before ubatch u's first, oldest first (up to n_gram - 1)
std::vector<int32_t> prev_tokens(const Dump& dump, const Ubatch& ub, int n) {
    std::vector<int32_t> seq;
    for (const Ubatch& x : dump.ubs) {
        if (x.pos0 >= ub.pos0) break;
        if (x.pos0 != (int64_t) seq.size()) die(dump.header + ": ubatches are not one sequence in order");
        seq.insert(seq.end(), x.tokens.begin(), x.tokens.end());
    }
    const int64_t k = std::min<int64_t>(n, (int64_t) seq.size());
    return std::vector<int32_t>(seq.end() - k, seq.end());
}

// one dump, one numerics: every op from the dump's own inputs; fork = bit for bit into run, engine = rel L2 into drift
void run_dump(const Dump& dump, Model& m, bool fork, Run* run, Drift* drift) {
    const e::DenseDims& d = m.d;
    const int64_t E = d.n_embd, X = (int64_t) e::kHc * E, NE = d.n_expert, K = d.n_expert_used, F = d.n_ff;
    const e::RefNumerics num = fork ? e::RefNumerics::Fork : e::RefNumerics::Engine;
    auto F32 = [&](const char* op, int occ, int il, int u) { return load_f32(dump.get(op, occ, il, u)); };
    // a matrix output: bit for bit (fork) or rel L2 (engine); every other op: bit for bit in fork numerics only (the
    // engine computes them in the same f32 arithmetic, so its drift is the matmuls')
    auto out_q = [&](const std::string& op, const std::string& where, const std::vector<float>& mine, const std::vector<float>& theirs) {
        if (mine.size() != theirs.size()) die(where + ": " + op + " has " + std::to_string(theirs.size()) + " values, the reference " + std::to_string(mine.size()));
        if (fork) cmp_bits(*run, op, where, mine.data(), theirs.data(), (int64_t) mine.size());
        else drift->add(op, where, mine.data(), theirs.data(), (int64_t) mine.size());
    };
    auto out_e = [&](const std::string& op, const std::string& where, const std::vector<float>& mine, const std::vector<float>& theirs) {
        if (!fork) return;
        out_q(op, where, mine, theirs);
    };
    for (const Ubatch& ub : dump.ubs) {
        const int64_t T = ub.T;
        const int u = ub.u;
        const std::string wu = "u" + std::to_string(u);
        // 1. embedding
        {
            std::vector<float> embd((size_t) (T * E));
            e::ref_embed(m.token_embd, ub.tokens.data(), T, embd.data());
            out_e("embd", wu, embd, F32("embd", 0, -1, u));
            const std::vector<float> de = F32("embd", 0, -1, u);
            std::vector<float> init((size_t) (T * X)), pre((size_t) (T * e::kHc), 0.f);
            for (int64_t t = 0; t < T; ++t) {
                for (int c = 0; c < e::kHc; ++c) std::memcpy(init.data() + (t * e::kHc + c) * E, de.data() + t * E, sizeof(float) * (size_t) E);
                pre[(size_t) (t * e::kHc)] = 1.f;
            }
            out_e("hc_init", wu, init, F32("hc_init", 0, -1, u));
            out_e("hc_pre_init", wu, pre, F32("hc_pre_init", 0, -1, u));
        }
        const std::vector<int32_t> prev = prev_tokens(dump, ub, d.engram_ngram - 1);
        for (int il = 0; il < (int) m.layers.size(); ++il) {
            Layer& L = m.layers[(size_t) il];
            const std::string w = "L" + std::to_string(il) + " " + wu;
            std::vector<float> x = il == 0 ? F32("hc_init", 0, -1, u) : F32("l_last", 0, il - 1, u);
            // 2. engram
            if (L.engram >= 0) {
                const int C = d.engram_cols();
                std::vector<int32_t> rows((size_t) (T * C));
                e::engram_rows(m.eh, L.engram, ub.tokens.data(), T, prev.data(), (int) prev.size(), rows.data());
                for (int32_t r : rows)
                    if (r < 0 || r >= L.table_rows) die(w + ": engram row " + std::to_string(r) + " outside the table");
                std::vector<float> emb((size_t) (T * C * d.engram_key)), kv((size_t) (T * E * (e::kHc + 1)));
                e::ref_engram_emb(L.table, d.engram_key, rows.data(), T * C, emb.data());
                out_e("engram_embd", w, emb, F32("engram_embd", 0, il, u));
                e::ref_matmul(L.eng.wkv, F32("engram_embd", 0, il, u).data(), T, kv.data());
                out_q("engram_kv", w, kv, F32("engram_kv", 0, il, u));
                std::vector<float> gate((size_t) (T * e::kHc)), out((size_t) (T * X));
                e::ref_engram_gate(d, L.eng, F32("engram_kv", 0, il, u).data(), x.data(), T, gate.data(), out.data());
                out_e("engram_gate", w, gate, F32("engram_gate", 0, il, u));
                out_e("engram_out", w, out, F32("engram_out", 0, il, u));
                x = F32("engram_out", 0, il, u);
            }
            // 3. the attention sublayer's mixes, collapse, norm and expansion
            std::vector<float> mixes((size_t) (T * e::kHcMix)), pre((size_t) (T * e::kHc)), post(pre.size()), comb((size_t) (T * 16));
            e::ref_hc_mixes(d, L.hc_attn, x.data(), T, mixes.data());
            out_q("hc_mixes", w, mixes, F32("hc_mixes", 0, il, u));
            e::ref_hc_coeffs(d, L.hc_attn, F32("hc_mixes", 0, il, u).data(), T, pre.data(), post.data(), comb.data());
            out_e("hc_pre", w, pre, F32("hc_pre", 0, il, u));
            out_e("hc_post", w, post, F32("hc_post", 0, il, u));
            out_e("hc_comb", w, comb, F32("hc_comb", 0, il, u));
            const std::vector<float> prev_pre = il == 0 ? F32("hc_pre_init", 0, -1, u) : F32("hc_pre", 1, il - 1, u);
            std::vector<float> col((size_t) (T * E)), nrm((size_t) (T * E)), xo((size_t) (T * X));
            e::ref_hc_collapse(x.data(), prev_pre.data(), T, (int) E, true, col.data());
            out_e("hc_attn_pre", w, col, F32("hc_attn_pre", 0, il, u));
            e::ref_rms_norm(F32("hc_attn_pre", 0, il, u).data(), L.attn_norm, T, (int) E, d.rms_eps, nrm.data());
            out_e("attn_norm", w, nrm, F32("attn_norm", 0, il, u));
            e::ref_hc_expand(F32("attn_out", 0, il, u).data(), x.data(), F32("hc_post", 0, il, u).data(),
                             F32("hc_comb", 0, il, u).data(), T, (int) E, xo.data());
            out_e("hc_attn_post", w, xo, F32("hc_attn_post", 0, il, u));
            // the FFN sublayer
            const std::vector<float> xa = F32("hc_attn_post", 0, il, u);
            e::ref_hc_mixes(d, L.hc_ffn, xa.data(), T, mixes.data());
            out_q("hc_mixes", w, mixes, F32("hc_mixes", 1, il, u));
            e::ref_hc_coeffs(d, L.hc_ffn, F32("hc_mixes", 1, il, u).data(), T, pre.data(), post.data(), comb.data());
            out_e("hc_pre", w, pre, F32("hc_pre", 1, il, u));
            out_e("hc_post", w, post, F32("hc_post", 1, il, u));
            out_e("hc_comb", w, comb, F32("hc_comb", 1, il, u));
            e::ref_hc_collapse(xa.data(), F32("hc_pre", 0, il, u).data(), T, (int) E, true, col.data());
            out_e("hc_ffn_pre", w, col, F32("hc_ffn_pre", 0, il, u));
            e::ref_rms_norm(F32("hc_ffn_pre", 0, il, u).data(), L.ffn_norm, T, (int) E, d.rms_eps, nrm.data());
            out_e("ffn_norm", w, nrm, F32("ffn_norm", 0, il, u));
            const std::vector<float> xn = F32("ffn_norm", 0, il, u);
            // 6. router
            e::RefRouterTrace tr;
            e::ref_router(num, d, L.router, xn.data(), T, tr);
            out_q("ffn_moe_logits", w, tr.logits, F32("ffn_moe_logits", 0, il, u));
            tr.logits = F32("ffn_moe_logits", 0, il, u);
            e::ref_router_select(num, d, L.router.bias, T, tr);
            const std::vector<int32_t> fork_ids = load_i32(dump.get("ffn_moe_topk", 0, il, u));
            if (fork) {
                out_e("ffn_moe_probs", w, tr.probs, F32("ffn_moe_probs", 0, il, u));
                out_e("ffn_moe_probs_biased", w, tr.biased, F32("ffn_moe_probs_biased", 0, il, u));
                cmp_i32(*run, "ffn_moe_argsort", w, tr.order.data(), load_i32(dump.get("ffn_moe_argsort", 0, il, u)).data(), T * NE);
                cmp_i32(*run, "ffn_moe_topk", w, tr.ids.data(), fork_ids.data(), T * K);
                out_e("ffn_moe_weights", w, tr.weights, F32("ffn_moe_weights", 0, il, u));
                out_e("ffn_moe_weights_sum", w, tr.sum, F32("ffn_moe_weights_sum", 0, il, u));
                out_e("ffn_moe_weights_sum_clamped", w, tr.sum_clamped, F32("ffn_moe_weights_sum_clamped", 0, il, u));
                out_e("ffn_moe_weights_norm", w, tr.norm, F32("ffn_moe_weights_norm", 0, il, u));
                out_e("ffn_moe_weights_scaled", w, tr.scaled, F32("ffn_moe_weights_scaled", 0, il, u));
            } else {
                // the engine's own logits, then its selection against the fork's set
                e::ref_router(num, d, L.router, xn.data(), T, tr);
                for (int64_t t = 0; t < T; ++t) {
                    std::vector<int32_t> a(tr.ids.begin() + t * K, tr.ids.begin() + (t + 1) * K),
                        b(fork_ids.begin() + t * K, fork_ids.begin() + (t + 1) * K);
                    std::sort(a.begin(), a.end());
                    std::sort(b.begin(), b.end());
                    ++drift->sets;
                    drift->sets_differ += a != b;
                }
            }
            // 7. shared expert
            e::RefSharedTrace sh;
            e::ref_shared_expert(d, L.sh, xn.data(), T, sh);
            out_q("ffn_up", w, sh.up, F32("ffn_up", 0, il, u));
            out_q("ffn_gate", w, sh.gate, F32("ffn_gate", 0, il, u));
            std::vector<float> act((size_t) (T * F));
            e::ref_swiglu_clamp(F32("ffn_gate", 0, il, u).data(), F32("ffn_up", 0, il, u).data(), T * F, d.swiglu_limit, act.data());
            out_e("ffn_swiglu_limited", w, act, F32("ffn_swiglu_limited", 0, il, u));
            std::vector<float> shexp((size_t) (T * E));
            e::ref_matmul(L.sh.down, F32("ffn_swiglu_limited", 0, il, u).data(), T, shexp.data());
            out_q("ffn_shexp", w, shexp, F32("ffn_shexp", 0, il, u));
            const std::vector<float> moe = F32("ffn_moe_out", 0, il, u), dsh = F32("ffn_shexp", 0, il, u);
            std::vector<float> fo((size_t) (T * E));
            for (size_t i = 0; i < fo.size(); ++i) fo[i] = moe[i] + dsh[i];
            out_e("ffn_out", w, fo, F32("ffn_out", 0, il, u));
            e::ref_hc_expand(F32("ffn_out", 0, il, u).data(), xa.data(), F32("hc_post", 1, il, u).data(),
                             F32("hc_comb", 1, il, u).data(), T, (int) E, xo.data());
            out_e("l_last", w, xo, F32("l_last", 0, il, u));
        }
        // 8. head: the last token's stream and the last layer's FFN pre
        {
            const int last = (int) m.layers.size() - 1;
            const std::vector<float> xl = seg(F32("l_last", 0, last, u), (T - 1) * X, X),
                                     pl = seg(F32("hc_pre", 1, last, u), (T - 1) * e::kHc, e::kHc);
            std::vector<float> hc_out((size_t) E), rn((size_t) E), logits((size_t) d.n_vocab);
            e::ref_head(d, xl.data(), pl.data(), 1, m.output_norm, m.output, hc_out.data(), rn.data(), logits.data());
            out_e("hc_out", wu, hc_out, F32("hc_out", 0, -1, u));
            e::ref_rms_norm(F32("hc_out", 0, -1, u).data(), m.output_norm, 1, (int) E, d.rms_eps, rn.data());
            out_e("result_norm", wu, rn, F32("result_norm", 0, -1, u));
            e::ref_matmul(m.output, F32("result_norm", 0, -1, u).data(), 1, logits.data());
            out_q("result_output", wu, logits, F32("result_output", 0, -1, u));
        }
    }
}

int run_dumps(const std::vector<std::string>& lists) {
    bool all_ok = true;
    const std::vector<Dump> dumps = dumps::read_lists(lists);
    for (const Dump& dump : dumps) {
        for (const Ubatch& ub : dump.ubs)
            if ((int64_t) ub.tokens.size() != ub.T) die(dump.header + ": ubatch " + std::to_string(ub.u) + " has no tokens line (dense_cases.py)");
        Model m = load_model(dump.model);
        int64_t queries = 0, decode = 0;
        for (const Ubatch& ub : dump.ubs) { queries += ub.T; decode += ub.T == 1; }
        std::printf("%s\n  %zu layers (engram before", dump.header.c_str(), m.layers.size());
        for (int32_t l : m.eh.layer_ids) std::printf(" %d", l);
        std::printf("), %zu ubatches, %lld tokens (%lld in decode ubatches), E %d, %d experts (%d used), FF %d, vocab %d, "
                    "engram %d buckets x %d\n", dump.ubs.size(), (long long) queries, (long long) decode, m.d.n_embd,
                    m.d.n_expert, m.d.n_expert_used, m.d.n_ff, m.d.n_vocab, m.d.engram_cols(), m.d.engram_key);
        Run run;
        run_dump(dump, m, true, &run, nullptr);
        bool ok = report(run, "fork numerics, every op from the dump's inputs");
        Drift dr;
        run_dump(dump, engine_numerics(m), false, nullptr, &dr);
        std::printf("  engine numerics (qmatmul's MMVQ and dequant paths, the bf16 router), worst rel L2 per ubatch against "
                    "the fork (bound %.0e):\n", kEngineRelL2);
        for (const auto& [op, wv] : dr.worst) {
            const bool good = wv.first <= kEngineRelL2;
            ok = ok && good;
            std::printf("    %-30s %.3g at %s: %s\n", op.c_str(), wv.first, wv.second.c_str(), good ? "ok" : "FAIL");
        }
        std::printf("    router top-k sets (its own logits and tie order): %lld of %lld differ from the fork's (reported, not bounded)\n",
                    (long long) dr.sets_differ, (long long) dr.sets);
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", dump.header.c_str());
        all_ok = all_ok && ok;
    }
    std::printf("dsv41_engine_dense_test (a): %s (%zu dumps)\n", all_ok ? "PASS" : "FAIL", dumps.size());
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--cases") == 0) return run_dumps(std::vector<std::string>(argv + 2, argv + argc));
    if (argc >= 2 && std::strcmp(argv[1], "--gpu") == 0) {
        bool quick = false, mmq = false, dry = false;
        int64_t table_rows = kDefaultTableRows;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--quick") == 0) quick = true;
            else if (std::strcmp(argv[i], "--mmq") == 0) mmq = true;
            else if (std::strcmp(argv[i], "--dry-run") == 0) dry = true;
            else if (std::strcmp(argv[i], "--table-rows") == 0 && i + 1 < argc) table_rows = std::atoll(argv[++i]);
            else { std::fprintf(stderr, "dsv41_engine_dense_test: unknown option %s\n", argv[i]); return 2; }
        }
        if (table_rows < 65536) { std::fprintf(stderr, "dsv41_engine_dense_test: --table-rows %lld (at least 65536)\n", (long long) table_rows); return 2; }
        return run_gpu(quick, mmq, dry, table_rows);
    }
    std::fprintf(stderr, "usage: dsv41_engine_dense_test --cases FILE [FILE ...] | --gpu [--quick] [--mmq] [--dry-run] [--table-rows N]\n");
    return 2;
}
