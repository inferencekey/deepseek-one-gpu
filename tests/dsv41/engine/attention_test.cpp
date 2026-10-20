// tests/dsv41/engine/attention_test.cpp - parity of the attention layer (include/strata/dsv41/engine/attention.hpp).
//
//   dsv41_engine_attention_test --cases FILE [FILE ...]   (a) GPU free: the CPU reference against the stock fork's
//                                                         op dumps (lists of tools/dsv41/engine/attention_cases.py)
//   dsv41_engine_attention_test --gpu [--quick] [--f16] [--mmq] [--dry-run]
//                                                         (b) CUDA against the CPU reference, attention_test_gpu.cu,
//                                                         on q2_K weights (qmatmul; --mmq its MMQ prefill option) or
//                                                         f16 weights (cuBLAS, --f16); --dry-run its host-side
//                                                         checks only (GPU free, verify.sh)
//
// (a) Per dump the reference runs every layer over every ubatch in order, from the dump's attention input
// (attn_norm) and the dump's own GGUF weights (q2_K), with its own KV state (window, carried pooling state, rows and
// index keys at their own ratio), in RefNumerics::Fork and the fork's cell layout (W window cells, cell = position;
// C compressed cells, both from the dump), and compares every op the layer's role computes with the dump, tolerance
// 0 (bit for bit): qr, q, kv, the compressor's projections, gathered blocks, pooled, normed and roped rows, the index
// keys, the indexer query, rotation and weights, the attention mask, kq (on the cells the mask shows), kq_soft_max,
// kqv, the attention output, de-rope, wo_a, attn_out, and the cache contents the fork attends (raw_k, comp_k, idx_k).
// The reference's own top-k is checked against the fork's idx_top_k as a set (a difference only at a tie with the
// threshold score is accepted); the attention then uses the fork's set, so that a tie the two break
// differently cannot hide an attention difference.
// Two passes per dump: ReaderMode::Dense (the fork, every query), and ReaderMode::Reuse (the engine's default), where
// a reader's queries are compared only where both modes attend the same rows ((p + 1) / ratio <= top_k: every query
// of the short dumps, the first ones of the 64-token prompts) and every other layer is compared in full. A compressor
// call that completes no block (the fork computes a masked dummy block there) has nothing of the engine to compare
// in comp_kv and idx_k_new: those files are counted as skipped. Then the engine's numerics against the same dumps
// (run_engine_numerics: a bound with the fork's selection, a report with its own), twice: the production path (the
// GGUF's q2_K weights through qmatmul's default paths: int8 activations on MMVQ for ubatches of up to 8 tokens,
// the f16 dequant path above) and the f16 test path (f16 weights and activations), both with double sums.
#include "attention_test_common.hpp"
#include "dump_lists.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/dsv41/engine/indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

// ---------------------------------------------------------------- the dump lists (dump_lists.hpp)
using dumps::Dump;
using dumps::DumpFile;
using dumps::Ubatch;
using dumps::load_f16;
using dumps::load_f32;
using dumps::load_i32;
using dumps::read_lists;

// ---------------------------------------------------------------- the model from its GGUF
struct Model {
    std::unique_ptr<strata::GgufFile> g;
    e::AttnDims d;
    std::vector<e::AttnLayer> layers;
    std::vector<e::RefLayerWeights> fork_w, engine_q, engine_f16;
    std::vector<std::vector<uint16_t>> f16_store;   // the f16 test path's weights
};

double meta_num(const strata::GgufFile& g, const std::string& key) {
    const strata::MetaValue* v = g.get("deepseek41." + key);
    if (!v || !v->is_num()) die(g.path() + ": no numeric deepseek41." + key);
    return v->num();
}

e::RefMatrix matrix(const strata::GgufFile& g, const std::string& name, bool required) {
    const strata::TensorInfo* t = g.find(name);
    if (!t) {
        if (required) die(g.path() + ": no tensor " + name);
        return {};
    }
    if (t->type != 10 || t->shape.size() != 2) die(g.path() + ": " + name + " is " + t->type_name() + ", the fork numerics take 2D q2_K");
    return e::RefMatrix{e::kRefTypeQ2K, (int64_t) t->shape[0], (int64_t) t->shape[1], g.tensor_data(*t)};
}

const float* vec_f32(const strata::GgufFile& g, const std::string& name, int64_t n, bool required) {
    const strata::TensorInfo* t = g.find(name);
    if (!t) {
        if (required) die(g.path() + ": no tensor " + name);
        return nullptr;
    }
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
    e::AttnDims& d = m.d;
    d.n_embd = (int) meta_num(g, "embedding_length");
    d.n_head = (int) meta_num(g, "attention.head_count");
    d.head_dim = (int) meta_num(g, "attention.key_length");
    d.n_rot = (int) meta_num(g, "rope.dimension_count");
    d.q_lora = (int) meta_num(g, "attention.q_lora_rank");
    d.o_groups = (int) meta_num(g, "attention.output_group_count");
    d.o_lora = (int) meta_num(g, "attention.output_lora_rank");
    d.n_swa = (int) meta_num(g, "attention.sliding_window");
    d.idx_heads = (int) meta_num(g, "attention.indexer.head_count");
    d.idx_dim = (int) meta_num(g, "attention.indexer.key_length");
    d.top_k = (int) meta_num(g, "attention.indexer.top_k");
    d.rms_eps = (float) meta_num(g, "attention.layer_norm_rms_epsilon");
    d.rope_base = (float) meta_num(g, "rope.freq_base");
    d.comp_rope_base = (float) meta_num(g, "attention.compress_rope_freq_base");
    d.yarn_factor = (float) meta_num(g, "rope.scaling.factor");
    d.yarn_orig_ctx = (int) meta_num(g, "rope.scaling.original_context_length");
    d.yarn_beta_fast = (float) meta_num(g, "rope.scaling.yarn_beta_fast");
    d.yarn_beta_slow = (float) meta_num(g, "rope.scaling.yarn_beta_slow");
    const int n_layer = (int) meta_num(g, "block_count");
    const strata::MetaValue* cr = g.get("deepseek41.attention.compress_ratios");
    if (!cr || cr->type != strata::MetaType::ARRAY || (int) cr->items.size() < n_layer) die(path + ": compress_ratios");
    std::vector<int> ratios((size_t) n_layer), flags((size_t) n_layer, 0);
    auto has = [&](int i, const char* t) { return g.find("blk." + std::to_string(i) + "." + t + ".weight") != nullptr; };
    for (int i = 0; i < n_layer; ++i) {
        ratios[(size_t) i] = (int) (int64_t) cr->items[(size_t) i].u;
        flags[(size_t) i] = (has(i, "attn_compressor_kv") ? e::kKvSource : 0) | (has(i, "attn_compressor_gate") ? e::kGate : 0) |
                            (has(i, "indexer.attn_k") ? e::kKeyOwner : 0) | (has(i, "indexer.attn_q_b") ? e::kIndexSource : 0);
    }
    m.layers = e::attn_layers(ratios, flags);
    for (int i = 0; i < n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        const e::AttnLayer& l = m.layers[(size_t) i];
        e::RefLayerWeights w;
        w.wq_a = matrix(g, b + "attn_q_a.weight", true);
        w.wq_b = matrix(g, b + "attn_q_b.weight", true);
        w.wkv = matrix(g, b + "attn_kv.weight", true);
        w.wo_a = matrix(g, b + "attn_output_a.weight", true);
        w.wo_b = matrix(g, b + "attn_output_b.weight", true);
        w.comp_kv = matrix(g, b + "attn_compressor_kv.weight", l.kv_source);
        w.comp_gate = matrix(g, b + "attn_compressor_gate.weight", l.gate);
        w.idx_k = matrix(g, b + "indexer.attn_k.weight", l.key_owner);
        w.idx_q_b = matrix(g, b + "indexer.attn_q_b.weight", l.index_source);
        w.idx_proj = matrix(g, b + "indexer.proj.weight", l.index_source);
        w.q_a_norm = vec_f32(g, b + "attn_q_a_norm.weight", d.q_lora, true);
        w.kv_norm = vec_f32(g, b + "attn_kv_a_norm.weight", d.head_dim, true);
        w.comp_norm = vec_f32(g, b + "attn_compressor_norm.weight", d.head_dim, l.kv_source);
        w.idx_k_norm = vec_f32(g, b + "indexer.k_norm.weight", d.idx_dim, l.key_owner);
        w.sinks = vec_f32(g, b + "attn_sinks.weight", d.n_head, true);
        m.fork_w.push_back(w);
        // the engine numerics: the same q2_K blocks against qmatmul's activations (the layout of each call's rows),
        // and the f16 test path's weights, dequantized and rounded to f16 as the GPU holds them there
        e::RefLayerWeights wq = w, wf = w;
        for (e::RefMatrix* mm : {&wq.wq_a, &wq.wq_b, &wq.wkv, &wq.wo_a, &wq.wo_b, &wq.comp_kv, &wq.comp_gate, &wq.idx_k,
                                 &wq.idx_q_b, &wq.idx_proj})
            mm->acts = e::RefActs::Auto;
        for (e::RefMatrix* mm : {&wf.wq_a, &wf.wq_b, &wf.wkv, &wf.wo_a, &wf.wo_b, &wf.comp_kv, &wf.comp_gate, &wf.idx_k,
                                 &wf.idx_q_b, &wf.idx_proj}) {
            if (!mm->data) continue;
            m.f16_store.push_back(e::ref_to_f16(e::ref_dequant(*mm)));
            mm->type = e::kRefTypeF16;
            mm->data = m.f16_store.back().data();
        }
        m.engine_q.push_back(wq);
        m.engine_f16.push_back(wf);
    }
    return m;
}

// ---------------------------------------------------------------- comparisons
struct OpStats {
    int64_t files = 0, values = 0, identical = 0, skipped = 0;
    double max_abs = 0;
    double worst_rel_l2 = 0;
};

struct Run {
    std::string name;
    std::map<std::string, OpStats> ops;
    int64_t diffs_shown = 0;
    int64_t topk_sets = 0, topk_exact = 0, topk_tied = 0, topk_other = 0;
    int64_t reuse_rows = 0, reuse_rows_skipped = 0;
    bool ok = true;
};

uint32_t fbits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }

// bit for bit: a[i] vs b[i] for the elements `use(i)` selects
template <typename F>
void cmp_bits(Run& run, const std::string& op, const char* where, const float* a, const float* b, int64_t n, F use) {
    OpStats& s = run.ops[op];
    ++s.files;
    for (int64_t i = 0; i < n; ++i) {
        if (!use(i)) continue;
        ++s.values;
        const bool same = fbits(a[i]) == fbits(b[i]) || (std::isnan(a[i]) && std::isnan(b[i]));
        if (same) { ++s.identical; continue; }
        s.max_abs = std::max(s.max_abs, (double) std::fabs(a[i] - b[i]));
        if (run.diffs_shown++ < 12)
            std::printf("diff %s %s: [%lld] reference %.9g (0x%08x) fork %.9g (0x%08x)\n", op.c_str(), where, (long long) i,
                        a[i], fbits(a[i]), b[i], fbits(b[i]));
    }
}
void cmp_bits(Run& run, const std::string& op, const char* where, const float* a, const float* b, int64_t n) {
    cmp_bits(run, op, where, a, b, n, [](int64_t) { return true; });
}
void cmp_f16(Run& run, const std::string& op, const char* where, const uint16_t* a, const uint16_t* b, int64_t n) {
    OpStats& s = run.ops[op];
    ++s.files;
    for (int64_t i = 0; i < n; ++i) {
        ++s.values;
        if (a[i] == b[i]) { ++s.identical; continue; }
        if (run.diffs_shown++ < 12)
            std::printf("diff %s %s: [%lld] reference 0x%04x fork 0x%04x\n", op.c_str(), where, (long long) i, a[i], b[i]);
    }
}

double rel_l2(const float* a, const float* b, int64_t n) {
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

// the reference's own top-k against the fork's set, restricted to the visible rows, ties at the threshold accepted
void cmp_topk(Run& run, const char* where, const e::RefTrace& tr, const std::vector<int32_t>& fork, int64_t k_out,
              int64_t T, int64_t p0, int ratio, int top_k) {
    for (int64_t t = 0; t < T; ++t) {
        const int64_t vis = e::indexer_visible(p0 + t, ratio, tr.n_k);
        std::vector<int32_t> mine, theirs;
        for (int i = 0; i < top_k; ++i)
            if (tr.topk[(size_t) (t * top_k + i)] >= 0) mine.push_back(tr.topk[(size_t) (t * top_k + i)]);
        for (int64_t i = 0; i < k_out; ++i)
            if (fork[(size_t) (t * k_out + i)] < vis) theirs.push_back(fork[(size_t) (t * k_out + i)]);
        std::sort(theirs.begin(), theirs.end());
        ++run.topk_sets;
        if (mine == theirs) { ++run.topk_exact; continue; }
        const float* s = tr.scores.data() + t * tr.n_k;
        float thr = INFINITY;
        for (int32_t j : mine) thr = std::min(thr, s[j]);
        std::vector<int32_t> diff;
        std::set_symmetric_difference(mine.begin(), mine.end(), theirs.begin(), theirs.end(), std::back_inserter(diff));
        bool tied = mine.size() == theirs.size();
        for (int32_t j : diff) tied = tied && s[j] == thr;
        if (tied) {
            ++run.topk_tied;
        } else {
            ++run.topk_other;
            run.ok = false;
            std::printf("FAIL top-k %s t%lld: the reference's set differs from the fork's beyond ties (%zu rows)\n", where,
                        (long long) t, diff.size());
        }
    }
}

std::vector<float> transpose3(const std::vector<float>& a, int64_t n0, int64_t n1, int64_t n2) {   // [n0][n1][n2] -> [n1][n0][n2]
    std::vector<float> out(a.size());
    for (int64_t i = 0; i < n0; ++i)
        for (int64_t j = 0; j < n1; ++j)
            std::memcpy(out.data() + (j * n0 + i) * n2, a.data() + (i * n1 + j) * n2, sizeof(float) * (size_t) n2);
    return out;
}

// one pass over a dump: every ubatch, every layer, in Fork numerics and `mode`
void run_pass(const Dump& dump, const Model& m, e::ReaderMode mode, Run& run) {
    const e::AttnDims& d = m.d;
    const int n_layer = (int) m.layers.size(), H = d.n_head, D = d.head_dim, ID = d.idx_dim;
    e::RefKv kv((size_t) n_layer);
    for (const Ubatch& ub : dump.ubs) {
        const int64_t T = ub.T, p0 = ub.pos0;
        for (int il = 0; il < n_layer; ++il) {
            const e::AttnLayer& l = m.layers[(size_t) il];
            const int r = l.ratio;
            char where[64];
            std::snprintf(where, sizeof where, "L%d u%d", il, ub.u);
            const DumpFile& fx = dump.get("attn_norm", 0, il, ub.u);
            const std::vector<float> x = load_f32(fx);
            e::RefForkLayout layout;
            layout.W = r ? dump.get("raw_k", 0, il, ub.u).ne[2] : dump.get("kq", 0, il, ub.u).ne[0];
            layout.C = r ? dump.get("comp_k", 0, il, ub.u).ne[2] : 0;
            const int64_t N = layout.W + layout.C;
            e::RefOptions o;
            o.numerics = e::RefNumerics::Fork;
            o.mode = mode;
            o.layout = &layout;
            std::vector<int32_t> fork_topk;
            int64_t k_out = 0;
            if (l.index_source) {
                const DumpFile& ft = dump.get("idx_top_k", 0, il, ub.u);
                fork_topk = load_i32(ft);
                k_out = ft.ne[0];
                o.topk_override = fork_topk.data();
                o.topk_override_k = k_out;
            }
            e::RefTrace tr;
            std::vector<float> out((size_t) (T * d.n_embd));
            e::ref_attn_forward(d, l, m.fork_w[(size_t) il], kv, x.data(), T, p0, o, out.data(), &tr);

            // which tokens this pass may compare after the attention: a Reuse reader only where both modes coincide
            std::vector<char> tok((size_t) T, 1);
            const bool reader = r && !l.index_source;
            if (reader && mode == e::ReaderMode::Reuse)
                for (int64_t t = 0; t < T; ++t) {
                    tok[(size_t) t] = (p0 + t + 1) / r <= d.top_k;
                    run.reuse_rows += tok[(size_t) t];
                    run.reuse_rows_skipped += !tok[(size_t) t];
                }
            auto tok_major = [&](int64_t per_tok) { return [&tok, per_tok](int64_t i) { return tok[(size_t) (i / per_tok)] != 0; }; };

            cmp_bits(run, "qr", where, tr.qr.data(), load_f32(dump.get("qr", 0, il, ub.u)).data(), (int64_t) tr.qr.size());
            cmp_bits(run, "q", where, tr.q.data(), load_f32(dump.get("q", 0, il, ub.u)).data(), (int64_t) tr.q.size());
            cmp_bits(run, "kv", where, tr.kv.data(), load_f32(dump.get("kv", 0, il, ub.u)).data(), (int64_t) tr.kv.size());
            if (l.kv_source) {
                cmp_bits(run, "comp_state_kv", where, tr.ckv.data(), load_f32(dump.get(r == 1 ? "comp_state_score" : "comp_state_kv", 0, il, ub.u)).data(),
                         (int64_t) tr.ckv.size());
                cmp_bits(run, "comp_state_score", where, tr.cscore.data(), load_f32(dump.get("comp_state_score", 0, il, ub.u)).data(),
                         (int64_t) tr.cscore.size());
                const char* names[5] = {"comp_kv.0 (gathered kv)", "comp_kv.1 (gathered score)", "comp_kv.2 (pooled)",
                                        "comp_kv.3 (normed)", "comp_kv.4 (roped)"};
                if (tr.B == 0) {
                    for (const char* nm : names) ++run.ops[nm].skipped;
                    if (l.key_owner) ++run.ops["idx_k_new"].skipped;
                } else {
                    const std::vector<float>* mine[5] = {&tr.gathered_kv, &tr.gathered_score, &tr.pooled, &tr.latent, &tr.rows};
                    for (int occ = 0; occ < 5; ++occ) {
                        const DumpFile& f = dump.get("comp_kv", occ, il, ub.u);
                        if ((size_t) f.count() != mine[occ]->size())
                            die(std::string(where) + ": comp_kv." + std::to_string(occ) + " has " + std::to_string(f.count()) +
                                " values, the reference " + std::to_string(mine[occ]->size()));
                        cmp_bits(run, names[occ], where, mine[occ]->data(), load_f32(f).data(), f.count());
                    }
                    if (l.key_owner)
                        cmp_bits(run, "idx_k_new", where, tr.idx_k_new.data(), load_f32(dump.get("idx_k_new", 0, il, ub.u)).data(),
                                 (int64_t) tr.idx_k_new.size());
                }
            }
            if (l.index_source) {
                cmp_bits(run, "idx_q", where, tr.iq.data(), load_f32(dump.get("idx_q", 0, il, ub.u)).data(), (int64_t) tr.iq.size());
                cmp_bits(run, "idx_q_rot", where, tr.iq_rot.data(), load_f32(dump.get("idx_q_rot", 0, il, ub.u)).data(), (int64_t) tr.iq_rot.size());
                cmp_bits(run, "idx_weights", where, tr.iw.data(), load_f32(dump.get("idx_weights", 0, il, ub.u)).data(), (int64_t) tr.iw.size());
                cmp_topk(run, where, tr, fork_topk, k_out, T, p0, r, d.top_k);
                // the index keys the fork scores: the key source's rows
                const DumpFile& fk = dump.get("idx_k", 0, il, ub.u);
                const std::vector<uint16_t> dk = load_f16(fk);
                const std::vector<uint16_t>& keys = kv.layers[(size_t) l.key_src].keys;
                cmp_f16(run, "idx_k (cache)", where, keys.data(), dk.data(), std::min<int64_t>((int64_t) keys.size(), fk.count()));
                // the selection mask: the compressed part of the reference's mask
                const std::vector<float> cm = load_f32(dump.get("comp_top_k_mask", 0, il, ub.u));
                std::vector<float> mine((size_t) (T * layout.C));
                for (int64_t t = 0; t < T; ++t)
                    std::memcpy(mine.data() + t * layout.C, tr.mask.data() + t * N + layout.W, sizeof(float) * (size_t) layout.C);
                cmp_bits(run, "comp_top_k_mask", where, mine.data(), cm.data(), (int64_t) mine.size());
            }
            // the cache contents the fork attends: window cells (positions written so far) and compressed rows
            if (r) {
                const std::vector<uint16_t> rk = load_f16(dump.get("raw_k", 0, il, ub.u));
                const std::vector<uint16_t>& win = kv.layers[(size_t) il].window;
                cmp_f16(run, "raw_k (cache)", where, win.data(), rk.data(), (int64_t) win.size());
                const std::vector<uint16_t> ck = load_f16(dump.get("comp_k", 0, il, ub.u));
                const std::vector<uint16_t>& rows = kv.layers[(size_t) l.kv_src].rows;
                cmp_f16(run, "comp_k (cache)", where, rows.data(), ck.data(), (int64_t) rows.size());
                const std::vector<uint16_t> ka = load_f16(dump.get("k_all", 0, il, ub.u));
                cmp_f16(run, "k_all (window part)", where, win.data(), ka.data(), (int64_t) win.size());
                cmp_f16(run, "k_all (compressed part)", where, rows.data(), ka.data() + layout.W * D, (int64_t) rows.size());
                cmp_bits(run, "kq_mask", where, tr.mask.data(), load_f32(dump.get("kq_mask", 0, il, ub.u)).data(), T * N, tok_major(N));
            }
            // attention: kq where the fork's mask shows the cell, the rest bit for bit
            {
                const std::vector<float> dmask = r ? load_f32(dump.get("kq_mask", 0, il, ub.u)) : tr.mask;
                const std::vector<float> dkq = load_f32(dump.get("kq", 0, il, ub.u));
                cmp_bits(run, "kq (shown cells)", where, tr.kq.data(), dkq.data(), H * T * N, [&](int64_t i) {
                    const int64_t c = i % N, t = (i / N) % T;
                    return tok[(size_t) t] && dmask[(size_t) (t * N + c)] == 0.f;
                });
                cmp_bits(run, "kq_soft_max", where, tr.kq_soft_max.data(), load_f32(dump.get("kq_soft_max", 0, il, ub.u)).data(),
                         H * T * N, [&](int64_t i) { return tok[(size_t) ((i / N) % T)] != 0; });
                const std::vector<float> kqv = transpose3(tr.attn, T, H, D);   // [T][H][D] -> [H][T][D]
                cmp_bits(run, "kqv", where, kqv.data(), load_f32(dump.get("kqv", 0, il, ub.u)).data(), H * T * D,
                         [&](int64_t i) { return tok[(size_t) ((i / D) % T)] != 0; });
                cmp_bits(run, r ? "attn_out_raw" : "attn_raw", where, tr.attn.data(),
                         load_f32(dump.get(r ? "attn_out_raw" : "attn_raw", 0, il, ub.u)).data(), T * H * D, tok_major(H * D));
            }
            cmp_bits(run, "attn_derope", where, tr.derope.data(), load_f32(dump.get("attn_derope", 0, il, ub.u)).data(), T * H * D,
                     tok_major(H * D));
            const int64_t OL = d.o_lora, OG = d.o_groups;
            const std::vector<float> oa = transpose3(tr.oa, T, OG, OL);   // [T][OG][OL] -> [OG][T][OL]
            cmp_bits(run, "attn_wo_a", where, oa.data(), load_f32(dump.get("attn_wo_a", 0, il, ub.u)).data(), OG * T * OL,
                     [&](int64_t i) { return tok[(size_t) ((i / OL) % T)] != 0; });
            cmp_bits(run, "attn_out", where, tr.out.data(), load_f32(dump.get("attn_out", 0, il, ub.u)).data(), T * d.n_embd,
                     tok_major(d.n_embd));
            (void) ID;
        }
    }
}

// The engine's numerics against the same dump, --readers dense. With the fork's selection (pass 1) attn_out per
// layer and ubatch must be within rel L2 kEngineRelL2: the fork rounds the input of each of the layer's four chained
// projections to Q8_K (int8 per 256 values, about 0.6% rel each) where the f16 path keeps f16 (measured worst 1.9-2.5%
// per dump on tiny-a and tiny-b, 2026-10-07) and the quantized path rounds it to int8 per 32 values on MMVQ and
// takes the f16 weights of its dequant path above 8 tokens (the same bound; tools/dsv41/engine/README.md has
// the measured values), while a semantic error (a wrong rope, row or cell) is O(1). With its own
// selection (pass 2, reported, not bounded) a query whose index scores sit near the top-k threshold can select
// another row than the fork does, which moves that query's output by far more: how often, and by how much, is what
// have to know about the f16 path.
constexpr double kEngineRelL2 = 5e-2;
struct EngineDrift {
    double worst = 0;
    std::string where;
    int64_t sets = 0, sets_differ = 0;
};
void run_engine_numerics(const Dump& dump, const Model& m, const std::vector<e::RefLayerWeights>& weights,
                         bool fork_selection, EngineDrift& r) {
    const int n_layer = (int) m.layers.size();
    e::RefKv kv((size_t) n_layer);
    for (const Ubatch& ub : dump.ubs)
        for (int il = 0; il < n_layer; ++il) {
            const e::AttnLayer& l = m.layers[(size_t) il];
            const std::vector<float> x = load_f32(dump.get("attn_norm", 0, il, ub.u));
            e::RefOptions o;
            o.numerics = e::RefNumerics::Engine;
            o.mode = e::ReaderMode::Dense;
            std::vector<int32_t> fork_topk;
            int64_t k_out = 0;
            if (l.index_source) {
                const DumpFile& ft = dump.get("idx_top_k", 0, il, ub.u);
                fork_topk = load_i32(ft);
                k_out = ft.ne[0];
                if (fork_selection) {
                    o.topk_override = fork_topk.data();
                    o.topk_override_k = k_out;
                }
            }
            e::RefTrace tr;
            std::vector<float> out((size_t) (ub.T * m.d.n_embd));
            e::ref_attn_forward(m.d, l, weights[(size_t) il], kv, x.data(), ub.T, ub.pos0, o, out.data(), &tr);
            const double e2 = rel_l2(out.data(), load_f32(dump.get("attn_out", 0, il, ub.u)).data(), (int64_t) out.size());
            if (e2 > r.worst) {
                r.worst = e2;
                r.where = "L" + std::to_string(il) + " u" + std::to_string(ub.u);
            }
            if (l.index_source)
                for (int64_t t = 0; t < ub.T; ++t) {
                    const int64_t vis = e::indexer_visible(ub.pos0 + t, l.ratio, tr.n_k);
                    std::vector<int32_t> mine, theirs;
                    for (int i = 0; i < m.d.top_k; ++i)
                        if (tr.topk[(size_t) (t * m.d.top_k + i)] >= 0) mine.push_back(tr.topk[(size_t) (t * m.d.top_k + i)]);
                    for (int64_t i = 0; i < k_out; ++i)
                        if (fork_topk[(size_t) (t * k_out + i)] < vis) theirs.push_back(fork_topk[(size_t) (t * k_out + i)]);
                    std::sort(theirs.begin(), theirs.end());
                    ++r.sets;
                    r.sets_differ += mine != theirs;
                }
        }
}

bool report(const Run& run) {
    bool ok = run.ok;
    std::printf("  %s:\n", run.name.c_str());
    for (const auto& [op, s] : run.ops) {
        const bool good = s.identical == s.values;
        ok = ok && good;
        std::printf("    %-28s %5lld calls %9lld values, %s", op.c_str(), (long long) s.files, (long long) s.values,
                    good ? "bit-identical" : "DIFFER");
        if (!good) std::printf(" (%lld differ, max |err| %.3g)", (long long) (s.values - s.identical), s.max_abs);
        if (s.skipped) std::printf(", %lld dummy blocks skipped", (long long) s.skipped);
        std::printf("\n");
    }
    std::printf("    top-k sets: %lld, %lld identical to the fork's, %lld differ only in ties at the threshold, %lld otherwise\n",
                (long long) run.topk_sets, (long long) run.topk_exact, (long long) run.topk_tied, (long long) run.topk_other);
    if (run.reuse_rows || run.reuse_rows_skipped)
        std::printf("    reader queries where both modes coincide (compared): %lld; not compared: %lld\n",
                    (long long) run.reuse_rows, (long long) run.reuse_rows_skipped);
    return ok;
}

int run_dumps(const std::vector<std::string>& lists) {
    bool all_ok = true;
    const std::vector<Dump> dumps = read_lists(lists);
    for (const Dump& dump : dumps) {
        const Model m = load_model(dump.model);
        int64_t queries = 0, decode = 0;
        for (const Ubatch& ub : dump.ubs) { queries += ub.T; decode += ub.T == 1; }
        std::printf("%s\n  %zu layers (roles:", dump.header.c_str(), m.layers.size());
        const char* rn[4] = {"window", "Full", "Reindex", "Reuse"};
        for (const auto& l : m.layers) std::printf(" %d %s", l.index, rn[(int) l.role()]);
        std::printf("), %zu ubatches, %lld tokens (%lld in decode ubatches), dims E %d H %d D %d, top-k %d\n", dump.ubs.size(),
                    (long long) queries, (long long) decode, m.d.n_embd, m.d.n_head, m.d.head_dim, m.d.top_k);
        bool ok = true;
        for (e::ReaderMode mode : {e::ReaderMode::Dense, e::ReaderMode::Reuse}) {
            Run run;
            run.name = mode == e::ReaderMode::Dense ? "fork numerics, --readers dense (every query)"
                                                    : "fork numerics, Reuse Mode (readers where the modes coincide)";
            run_pass(dump, m, mode, run);
            ok = report(run) && ok;
        }
        struct Path { const char* name; const std::vector<e::RefLayerWeights>* w; };
        for (const Path& pth : {Path{"q2_K weights, qmatmul's MMVQ and dequant paths", &m.engine_q},
                                Path{"f16 weights and activations, the test path", &m.engine_f16}}) {
            EngineDrift fsel, own;
            run_engine_numerics(dump, m, *pth.w, true, fsel);
            run_engine_numerics(dump, m, *pth.w, false, own);
            const bool eok = fsel.worst <= kEngineRelL2;
            std::printf("  engine numerics (%s, --readers dense), attn_out per layer and ubatch:\n"
                        "    with the fork's selection: worst rel L2 %.3g at %s (bound %.0e): %s\n"
                        "    with its own selection: worst rel L2 %.3g at %s; %lld of %lld index-source queries select other "
                        "rows than the fork (reported, not bounded)\n",
                        pth.name, fsel.worst, fsel.where.c_str(), kEngineRelL2, eok ? "ok" : "FAIL", own.worst,
                        own.where.c_str(), (long long) own.sets_differ, (long long) own.sets);
            ok = ok && eok;
        }
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", dump.header.c_str());
        all_ok = all_ok && ok;
    }
    std::printf("dsv41_engine_attention_test (a): %s (%zu dumps)\n", all_ok ? "PASS" : "FAIL", dumps.size());
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--cases") == 0) return run_dumps(std::vector<std::string>(argv + 2, argv + argc));
    if (argc >= 2 && std::strcmp(argv[1], "--gpu") == 0) {
        bool quick = false, f16 = false, mmq = false, dry = false;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--quick") == 0) quick = true;
            else if (std::strcmp(argv[i], "--f16") == 0) f16 = true;
            else if (std::strcmp(argv[i], "--mmq") == 0) mmq = true;
            else if (std::strcmp(argv[i], "--dry-run") == 0) dry = true;
            else { std::fprintf(stderr, "dsv41_engine_attention_test: unknown option %s\n", argv[i]); return 2; }
        }
        return run_gpu(quick, f16, mmq, dry);
    }
    std::fprintf(stderr, "usage: dsv41_engine_attention_test --cases FILE [FILE ...] | --gpu [--quick] [--f16] [--mmq] [--dry-run]\n");
    return 2;
}
