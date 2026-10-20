// tools/dsv41/engine/dsv41-layer-ref.cpp - the first layers of a deepseek41
// GGUF on the CPU reference, every op dumped, in one of three numerics, so the real model's parity gap against the
// real-model oracle (the stock fork's dsv41-moe-dump on an A100) can be traced to the op it enters in. CPU only (no CUDA call): it runs on the pod beside the engine runs.
//
//   dsv41-layer-ref --model F.gguf --tokens-file F --numerics N --out DIR [--layers 1|2] [--oracle DIR]
//                   [--window-cells N] [--threads N] [--dry-run]
//
// One call of the prompt's T tokens at positions 0 .. T - 1 (the oracle's first ubatch: its 31-token prompt), through
// layers 0 .. --layers - 1 (default 2: the window-only layers 0 and 1 with the engram before layer 1), each
// op from the previous op of the same numerics (a chain, as the engine and the fork compute it):
//   fork-cpu   the stock fork's CPU backend (RefNumerics::Fork, RefActs::Fork: Q8_K activations into every K-quant
//              dot, the bf16 router's tinyBLAS order, ggml's soft_max over the fork's cells; SPEC.md "Numerics on the
//              fork's CPU path"), bit for bit where the layout is the fork's (tools/dsv41/verify.sh checks it against
//              tiny-b's op dump). The oracle did NOT run its dense
//              part on the CPU (--n-gpu-layers 99, only the routed experts of layers 0-25 and the engram table
//              on the CPU), so this is not expected to land on the oracle's cur.
//   fork-cuda  the oracle's own GPU arithmetic for the quantized matmuls: the activations quantized as ggml-cuda does
//              on the A100 (MMVQ's q8_1 for up to kMmvqMaxCols tokens, MMQ's int8 tiles above: ggml_cuda_should_use_mmq
//              is true for q2_K, q3_K and q6_K on every device with Turing tensor cores, the fork's mmq.cu 266-320),
//              qmatmul_ref's exact sums of them (MMQ and MMVQ are the fork's own GPU arithmetic); the
//              attention, norms and router in the engine's numerics (f32 ops, double sums: the oracle's flash attention
//              is an f32-accumulating f16 kernel); the routed experts on the CPU as the oracle ran those of layers
//              0-25 (CpuExperts in CpuNumerics::Fork: the fork's CPU experts bit for bit).
//   engine     the engine's GPU numerics (RefNumerics::Engine, RefActs::Auto: qmatmul's default paths, the dequant
//              path's f16 activations above kMmvqMaxCols tokens; the bf16 router's exact products), the routed
//              experts as a prefill pass computes them (f16 weights and activations, experts_gemm.hpp). The real
//              engine's prefill splits the experts outside its slab between the ring and the CPU (experts.hpp); this
//              reference takes the GPU's numerics for every pair.
// With --oracle DIR (the real-model oracle's dsv41-moe-dump directory, or a fork op dump: tiny-b in verify.sh) the routed experts of
// the chain take the oracle's expert ids (as a forced run), and each layer also gets the isolated MoE: the oracle's
// cur through this numerics' router (the weights at the oracle's ids, and its own top-k) and routed experts with the
// oracle's ids and weights (occurrence 1 of ffn_moe_weights_scaled, ffn_moe_topk and ffn_moe_out).
//
// Dumps (DIR/<op>.<occ>-L<layer>-u0000.bin, the fork's op-dump names, raw little endian, the fork's row-major layout
// with the token slowest; model level ops at layer 999) and DIR/layer-ref-manifest.json (the numerics, the prompt,
// every file in the order computed with its op, occurrence, layer, type and row width). tools/dsv41/engine/
// layer_parity.py compares them with each other, with the oracle and with an engine run's dumps. --window-cells: the
// fork's KV cells its CPU attention runs over (fork-cpu only; default 256: the used cells padded to 256, as the tiny
// dumps' kq show for 64 tokens). --dry-run: the model loaded and checked, the oracle's files of the call read and
// sized, the plan printed, nothing computed.
#include "strata/dsv41/engine/attention_ref.hpp"
#include "strata/dsv41/engine/dense_ref.hpp"
#include "strata/dsv41/engine/experts.hpp"
#include "strata/dsv41/engine/model.hpp"
#include "strata/dsv41/engine/qmatmul_ref.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "dsv41-layer-ref: %s\n", m.c_str());
    std::exit(2);
}

enum class Numerics { ForkCpu, ForkCuda, Engine };

const char* numerics_name(Numerics n) {
    switch (n) {
        case Numerics::ForkCpu: return "fork-cpu";
        case Numerics::ForkCuda: return "fork-cuda";
        case Numerics::Engine: return "engine";
    }
    return "?";
}

bool exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// mkdir -p
void make_dirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i < path.size() && path[i] != '/') continue;
        const std::string d = path.substr(0, i);
        if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST) die("cannot make " + d + ": " + std::strerror(errno));
    }
}

template <typename V> std::vector<V> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() % sizeof(V)) die(path + ": " + std::to_string(b.size()) + " bytes, not whole values");
    std::vector<V> v(b.size() / sizeof(V));
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

// the oracle's MoE kinds of call 0: the dsv41-moe-dump names (cur, out, topk, weights) or a fork op dump's
struct Oracle {
    std::string dir;
    bool moe_dump = false;
    std::string path(const char* kind, int il) const {
        static const char* const op_names[][2] = {{"cur", "ffn_norm"}, {"out", "ffn_moe_out"}, {"topk", "ffn_moe_topk"},
                                                  {"weights", "ffn_moe_weights_scaled"}};
        char name[96];
        for (const auto& k : op_names)
            if (std::strcmp(k[0], kind) == 0) {
                if (moe_dump) std::snprintf(name, sizeof name, "/%s-L%03d-u0000.bin", kind, il);
                else std::snprintf(name, sizeof name, "/%s.0-L%03d-u0000.bin", k[1], il);
                return dir + name;
            }
        die(std::string("no oracle kind ") + kind);
    }
    template <typename V> std::vector<V> get(const char* kind, int il, size_t n) const {
        std::vector<V> v = read_file<V>(path(kind, il));
        if (v.size() != n)
            die(path(kind, il) + ": " + std::to_string(v.size()) + " values, the call needs " + std::to_string(n) +
                " (the oracle's first ubatch must be the prompt)");
        return v;
    }
};

struct Dumper {
    std::string dir;
    std::string files;   // the manifest's entries
    size_t n = 0, bytes = 0;
    void put(const char* op, int occ, int il, const void* data, size_t count, bool i32, int64_t row) {
        char name[128];
        std::snprintf(name, sizeof name, "%s.%d-L%03d-u0000.bin", op, occ, il < 0 ? 999 : il);
        const std::string path = dir + "/" + name;
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) die("cannot write " + path + ": " + std::strerror(errno));
        const bool ok = std::fwrite(data, 4, count, f) == count;
        if (std::fclose(f) != 0 || !ok) die("writing " + path + " failed");
        std::ostringstream o;
        o << (n ? ",\n  " : "\n  ") << "{\"file\": \"" << name << "\", \"op\": \"" << op << "\", \"occurrence\": " << occ
          << ", \"layer\": " << il << ", \"dtype\": \"" << (i32 ? "i32" : "f32") << "\", \"row\": " << row
          << ", \"count\": " << count << "}";
        files += o.str();
        ++n;
        bytes += count * 4;
    }
    void f32(const char* op, int occ, int il, const std::vector<float>& v, int64_t row) {
        put(op, occ, il, v.data(), v.size(), false, row);
    }
    void i32(const char* op, int occ, int il, const std::vector<int32_t>& v, int64_t row) {
        put(op, occ, il, v.data(), v.size(), true, row);
    }
};

// [n0][n1][n2] -> [n1][n0][n2] (the fork's kqv and attn_wo_a put the token in the middle)
std::vector<float> transpose3(const std::vector<float>& a, int64_t n0, int64_t n1, int64_t n2) {
    std::vector<float> out(a.size());
    for (int64_t i = 0; i < n0; ++i)
        for (int64_t j = 0; j < n1; ++j)
            std::memcpy(out.data() + (j * n0 + i) * n2, a.data() + (i * n1 + j) * n2, sizeof(float) * (size_t) n2);
    return out;
}

class LayerRef {
public:
    LayerRef(const e::Model& m, Numerics n, int64_t T, int window_cells, int threads)
        : m_(m), n_(n), T_(T), window_cells_(window_cells), cpu_(threads, e::CpuNumerics::Fork) {}

    e::RefActs acts(int type) const {
        switch (n_) {
            case Numerics::ForkCpu: return e::RefActs::Fork;
            case Numerics::Engine: return e::RefActs::Auto;
            case Numerics::ForkCuda:
                if (type == e::kRefTypeBF16) return e::RefActs::Auto;
                switch (e::qact_layout(type, T_)) {
                    case e::QActLayout::Q8_1: return e::RefActs::Q8_1;
                    case e::QActLayout::MmqD4: return e::RefActs::MmqD4;
                    case e::QActLayout::MmqD2S6: return e::RefActs::MmqD2S6;
                }
        }
        die("no activations for the numerics");
    }
    e::RefMatrix mat(const e::QWeightView& w) const { return e::RefMatrix{w.type, w.k, w.rows, w.data, acts(w.type)}; }
    e::RefNumerics ref_numerics() const { return n_ == Numerics::ForkCpu ? e::RefNumerics::Fork : e::RefNumerics::Engine; }

    // the routed experts of xn [T][E] for ids and weights [T][K]: y per pair in this numerics, then the fork's combine
    std::vector<float> routed(int il, const std::vector<float>& xn, const std::vector<int32_t>& ids,
                              const std::vector<float>& w) {
        const e::DenseDims& d = m_.dense_dims();
        const e::ExpertLayerHost& L = m_.experts()[(size_t) il];
        const int64_t E = d.n_embd, F = d.n_ff, K = d.n_expert_used, P = T_ * K;
        std::vector<float> y((size_t) (P * E));
        // the pairs of each expert, in first-use order (the engine's decode split builds its CPU jobs the same way)
        std::vector<int32_t> experts;
        std::vector<std::vector<int64_t>> pairs((size_t) d.n_expert);
        for (int64_t p = 0; p < P; ++p) {
            const int32_t x = ids[(size_t) p];
            if (x < 0 || x >= d.n_expert) die("layer " + std::to_string(il) + ": expert id " + std::to_string(x));
            if (pairs[(size_t) x].empty()) experts.push_back(x);
            pairs[(size_t) x].push_back(p);
        }
        if (n_ != Numerics::Engine) {
            std::vector<e::CpuExpertJob> jobs(experts.size());
            int64_t y0 = 0;
            for (size_t j = 0; j < experts.size(); ++j) {
                for (int p = 0; p < 3; ++p) jobs[j].rows.part[p] = L.data[p] + (size_t) experts[j] * L.bytes[p];
                for (int64_t p : pairs[(size_t) experts[j]]) jobs[j].tokens.push_back((int32_t) (p / K));
                jobs[j].y0 = y0;
                y0 += (int64_t) jobs[j].tokens.size();
            }
            std::vector<float> rows((size_t) (y0 * E));
            cpu_.run(L, d, xn.data(), T_, jobs.data(), jobs.size(), rows.data());
            for (size_t j = 0; j < experts.size(); ++j) {
                const std::vector<int64_t>& ps = pairs[(size_t) experts[j]];
                for (size_t i = 0; i < ps.size(); ++i)
                    std::memcpy(y.data() + ps[i] * E, rows.data() + (jobs[j].y0 + (int64_t) i) * E, sizeof(float) * (size_t) E);
            }
        } else {
            for (int32_t x : experts) {
                const std::vector<int64_t>& ps = pairs[(size_t) x];
                const int64_t n = (int64_t) ps.size();
                std::vector<float> in((size_t) (n * E)), g((size_t) (n * F)), u((size_t) (n * F)), h((size_t) (n * F)),
                    out((size_t) (n * E));
                for (int64_t i = 0; i < n; ++i)
                    std::memcpy(in.data() + i * E, xn.data() + (ps[(size_t) i] / K) * E, sizeof(float) * (size_t) E);
                auto view = [&](int p) {
                    const void* data = L.data[p] + (size_t) x * L.bytes[p];
                    return p == 2 ? e::QWeightView{L.type[2], F, E, data} : e::QWeightView{L.type[p], E, F, data};
                };
                e::qref_matmul_f16(view(1), in.data(), E, n, 1, g.data(), F, nullptr);
                e::qref_matmul_f16(view(0), in.data(), E, n, 1, u.data(), F, nullptr);
                e::ref_swiglu_clamp(g.data(), u.data(), n * F, L.limit, h.data());
                e::qref_matmul_f16(view(2), h.data(), F, n, 1, out.data(), E, nullptr);
                for (int64_t i = 0; i < n; ++i)
                    std::memcpy(y.data() + ps[(size_t) i] * E, out.data() + i * E, sizeof(float) * (size_t) E);
            }
        }
        std::vector<float> o((size_t) (T_ * E));
        e::ref_expert_combine(y.data(), w.data(), T_, (int) K, (int) E, nullptr, o.data());
        return o;
    }

    // the router's weights at given ids [T][K] from its unbiased probs, as ref_router_select makes them from its own
    // ids: the probs at the ids, their sum in double (ggml_vec_sum_f32), its clamp, the division and the scale, each
    // rounded
    std::vector<float> weights_at(const e::RefRouterTrace& tr, const std::vector<int32_t>& ids) const {
        const e::DenseDims& d = m_.dense_dims();
        const int64_t K = d.n_expert_used, NE = d.n_expert;
        const bool scaled = d.weights_scale != 0.0f && d.weights_scale != 1.0f;
        std::vector<float> w((size_t) (T_ * K));
        for (int64_t t = 0; t < T_; ++t) {
            float* wt = w.data() + t * K;
            double sum = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                wt[k] = tr.probs[(size_t) (t * NE + ids[(size_t) (t * K + k)])];
                sum += (double) wt[k];
            }
            const float clamped = std::max((float) sum, 6.103515625e-5f);
            for (int64_t k = 0; k < K; ++k) {
                if (d.weights_norm) wt[k] = wt[k] / clamped;
                if (scaled) wt[k] = wt[k] * d.weights_scale;
            }
        }
        return w;
    }

    void run(const std::vector<int32_t>& tokens, int n_layers, const Oracle* oracle, Dumper& out) {
        const e::DenseDims& d = m_.dense_dims();
        const e::AttnDims& ad = m_.attn_dims();
        const int64_t T = T_, E = d.n_embd, X = (int64_t) e::kHc * E, K = d.n_expert_used;
        const e::HostModel& host = m_.host();
        // 1. embedding
        std::vector<float> embd((size_t) (T * E));
        const e::QWeightView& te = m_.token_embd();
        e::ref_embed(e::RefMatrix{te.type, te.k, te.rows, te.data, e::RefActs::Fork}, tokens.data(), T, embd.data());
        out.f32("embd", 0, -1, embd, E);
        std::vector<float> x((size_t) (T * X)), prev_pre((size_t) (T * e::kHc), 0.f);
        for (int64_t t = 0; t < T; ++t) {
            for (int c = 0; c < e::kHc; ++c) std::memcpy(x.data() + (t * e::kHc + c) * E, embd.data() + t * E, sizeof(float) * (size_t) E);
            prev_pre[(size_t) (t * e::kHc)] = 1.f;
        }
        e::RefKv kv((size_t) m_.n_layer());
        for (int il = 0; il < n_layers; ++il) {
            const e::LayerWeights& L = m_.layer(il);
            // 2. engram
            if (L.engram >= 0) {
                const int C = d.engram_cols();
                std::vector<int32_t> rows((size_t) (T * C));
                e::engram_rows(host.engram, L.engram, tokens.data(), T, nullptr, 0, rows.data());
                const e::EngramTable& tab = m_.table(L.engram);
                for (int32_t r : rows)
                    if (r < 0 || r >= tab.n_rows()) die("layer " + std::to_string(il) + ": engram row " + std::to_string(r) + " outside the table");
                std::vector<float> emb((size_t) (T * C * d.engram_key)), ekv((size_t) (T * E * (e::kHc + 1))),
                    gate((size_t) (T * e::kHc)), eo((size_t) (T * X));
                e::ref_engram_emb(tab.rows(), d.engram_key, rows.data(), T * C, emb.data());
                const e::RefEngramWeights ew{mat(L.eng.wkv), L.eng.q, L.eng.k};
                e::ref_engram(d, ew, emb.data(), x.data(), T, ekv.data(), gate.data(), eo.data());
                out.f32("engram_embd", 0, il, emb, (int64_t) C * d.engram_key);
                out.f32("engram_kv", 0, il, ekv, E * (e::kHc + 1));
                out.f32("engram_gate", 0, il, gate, e::kHc);
                out.f32("engram_out", 0, il, eo, X);
                x = eo;
            }
            // 3. the attention sublayer
            const e::RefHcWeights hca{mat(L.hc_attn.fn), L.hc_attn.base, L.hc_attn.scale};
            const e::RefHcWeights hcf{mat(L.hc_ffn.fn), L.hc_ffn.base, L.hc_ffn.scale};
            std::vector<float> mixes((size_t) (T * e::kHcMix)), pre_a((size_t) (T * e::kHc)), post(pre_a.size()),
                comb((size_t) (T * 16));
            e::ref_hc_mixes(d, hca, x.data(), T, mixes.data());
            e::ref_hc_coeffs(d, hca, mixes.data(), T, pre_a.data(), post.data(), comb.data());
            out.f32("hc_mixes", 0, il, mixes, e::kHcMix);
            out.f32("hc_pre", 0, il, pre_a, e::kHc);
            out.f32("hc_post", 0, il, post, e::kHc);
            out.f32("hc_comb", 0, il, comb, 16);
            std::vector<float> col((size_t) (T * E)), nrm((size_t) (T * E));
            e::ref_hc_collapse(x.data(), prev_pre.data(), T, (int) E, true, col.data());
            e::ref_rms_norm(col.data(), L.attn_norm, T, (int) E, d.rms_eps, nrm.data());
            out.f32("hc_attn_pre", 0, il, col, E);
            out.f32("attn_norm", 0, il, nrm, E);
            const e::AttnLayer& role = L.role;
            if (role.ratio != 0) die("layer " + std::to_string(il) + " is not window-only: --layers takes the first window layers");
            const e::AttnLayerWeights& aw = L.attn;
            e::RefLayerWeights rw;
            rw.wq_a = mat(aw.wq_a);
            rw.wq_b = mat(aw.wq_b);
            rw.wkv = mat(aw.wkv);
            rw.wo_a = mat(aw.wo_a);
            rw.wo_b = mat(aw.wo_b);
            rw.q_a_norm = aw.q_a_norm;
            rw.kv_norm = aw.kv_norm;
            rw.sinks = aw.sinks;
            e::RefForkLayout layout{window_cells_, 0};
            e::RefOptions o;
            o.numerics = ref_numerics();
            o.mode = e::ReaderMode::Dense;
            if (n_ == Numerics::ForkCpu) o.layout = &layout;
            e::RefTrace tr;
            std::vector<float> aout((size_t) (T * E));
            e::ref_attn_forward(ad, role, rw, kv, nrm.data(), T, 0, o, aout.data(), &tr);
            const int64_t HD = (int64_t) ad.n_head * ad.head_dim;
            out.f32("qr", 0, il, tr.qr, ad.q_lora);
            out.f32("q", 0, il, tr.q, HD);
            out.f32("kv", 0, il, tr.kv, ad.head_dim);
            out.f32("kqv", 0, il, transpose3(tr.attn, T, ad.n_head, ad.head_dim), ad.head_dim);
            out.f32("attn_raw", 0, il, tr.attn, HD);
            out.f32("attn_derope", 0, il, tr.derope, HD);
            out.f32("attn_wo_a", 0, il, transpose3(tr.oa, T, ad.o_groups, ad.o_lora), ad.o_lora);
            out.f32("attn_out", 0, il, aout, E);
            std::vector<float> xa((size_t) (T * X));
            e::ref_hc_expand(aout.data(), x.data(), post.data(), comb.data(), T, (int) E, xa.data());
            out.f32("hc_attn_post", 0, il, xa, X);
            // the FFN sublayer: mixes, collapse with the attention's pre (the V4.1 lag), norm = the MoE input
            std::vector<float> pre_f((size_t) (T * e::kHc));
            e::ref_hc_mixes(d, hcf, xa.data(), T, mixes.data());
            e::ref_hc_coeffs(d, hcf, mixes.data(), T, pre_f.data(), post.data(), comb.data());
            out.f32("hc_mixes", 1, il, mixes, e::kHcMix);
            out.f32("hc_pre", 1, il, pre_f, e::kHc);
            out.f32("hc_post", 1, il, post, e::kHc);
            out.f32("hc_comb", 1, il, comb, 16);
            e::ref_hc_collapse(xa.data(), pre_a.data(), T, (int) E, true, col.data());
            std::vector<float> xn((size_t) (T * E));
            e::ref_rms_norm(col.data(), L.ffn_norm, T, (int) E, d.rms_eps, xn.data());
            out.f32("hc_ffn_pre", 0, il, col, E);
            out.f32("ffn_norm", 0, il, xn, E);
            // 6. router (its own top-k), 7. shared expert, routed experts (the oracle's ids when given)
            const e::RefRouterWeights rtw{e::RefMatrix{e::kRefTypeBF16, E, d.n_expert, L.router.gate_inp, acts(e::kRefTypeBF16)},
                                          L.router.bias};
            e::RefRouterTrace rt;
            e::ref_router(ref_numerics(), d, rtw, xn.data(), T, rt);
            out.f32("ffn_moe_logits", 0, il, rt.logits, d.n_expert);
            out.f32("ffn_moe_probs_biased", 0, il, rt.biased, d.n_expert);
            out.i32("ffn_moe_topk", 0, il, rt.ids, K);
            const std::vector<int32_t> ids = oracle ? oracle->get<int32_t>("topk", il, (size_t) (T * K)) : rt.ids;
            const std::vector<float> w = weights_at(rt, ids);
            out.f32("ffn_moe_weights_scaled", 0, il, w, K);
            const e::RefSharedWeights shw{mat(L.shared.gate), mat(L.shared.up), mat(L.shared.down)};
            e::RefSharedTrace sh;
            e::ref_shared_expert(d, shw, xn.data(), T, sh);
            out.f32("ffn_shexp", 0, il, sh.out, E);
            const std::vector<float> moe = routed(il, xn, ids, w);
            out.f32("ffn_moe_out", 0, il, moe, E);
            std::vector<float> fo((size_t) (T * E));
            for (size_t i = 0; i < fo.size(); ++i) fo[i] = moe[i] + sh.out[i];
            out.f32("ffn_out", 0, il, fo, E);
            std::vector<float> xl((size_t) (T * X));
            e::ref_hc_expand(fo.data(), xa.data(), post.data(), comb.data(), T, (int) E, xl.data());
            out.f32("l_last", 0, il, xl, X);
            // the isolated MoE: the oracle's cur, ids and weights through this numerics' router and experts
            if (oracle) {
                const std::vector<float> cur = oracle->get<float>("cur", il, (size_t) (T * E));
                const std::vector<float> ow = oracle->get<float>("weights", il, (size_t) (T * K));
                e::RefRouterTrace ort;
                e::ref_router(ref_numerics(), d, rtw, cur.data(), T, ort);
                out.f32("ffn_moe_weights_scaled", 1, il, weights_at(ort, ids), K);
                out.i32("ffn_moe_topk", 1, il, ort.ids, K);
                out.f32("ffn_moe_out", 1, il, routed(il, cur, ids, ow), E);
            }
            x = xl;
            prev_pre = pre_f;
        }
    }

private:
    const e::Model& m_;
    Numerics n_;
    int64_t T_;
    int window_cells_;
    e::CpuExperts cpu_;
};

std::vector<int32_t> read_tokens(const std::string& path) {
    std::ifstream f(path);
    if (!f) die("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream in(s);
    std::vector<int32_t> ids;
    std::string tok;
    while (in >> tok) {
        char* end = nullptr;
        const long v = std::strtol(tok.c_str(), &end, 10);
        if (*end != '\0' || v < 0 || v > INT32_MAX) die(path + ": '" + tok + "' is not a token id");
        ids.push_back((int32_t) v);
    }
    if (ids.empty()) die(path + ": no token ids");
    return ids;
}

// a JSON string literal (dsv41-engine.cpp's)
std::string json_string(const std::string& v) {
    std::string s = "\"";
    for (const char c : v) {
        if (c == '"' || c == '\\') {
            s += '\\';
            s += c;
        } else if ((unsigned char) c < 0x20) {
            char u[7];
            std::snprintf(u, sizeof u, "\\u%04x", (unsigned) (unsigned char) c);
            s += u;
        } else {
            s += c;
        }
    }
    return s + '"';
}

std::string join(const std::vector<int32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, tokens_file, out_dir, oracle_dir, numerics_text;
    int n_layers = 2, window_cells = 256, threads = 0;
    bool dry = false;
    auto value = [&](int& i) -> std::string {
        if (i + 1 >= argc) die(std::string(argv[i]) + " needs a value");
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model") model_path = value(i);
        else if (a == "--tokens-file") tokens_file = value(i);
        else if (a == "--numerics") numerics_text = value(i);
        else if (a == "--out") out_dir = value(i);
        else if (a == "--layers") n_layers = std::atoi(value(i).c_str());
        else if (a == "--oracle") oracle_dir = value(i);
        else if (a == "--window-cells") window_cells = std::atoi(value(i).c_str());
        else if (a == "--threads") threads = std::atoi(value(i).c_str());
        else if (a == "--dry-run") dry = true;
        else if (a == "-h" || a == "--help") {
            std::printf("usage: dsv41-layer-ref --model F.gguf --tokens-file F --numerics fork-cpu|fork-cuda|engine --out DIR\n"
                        "                       [--layers 1|2] [--oracle DIR] [--window-cells N] [--threads N] [--dry-run]\n");
            return 0;
        } else die("unknown argument " + a + " (--help)");
    }
    if (model_path.empty() || tokens_file.empty() || out_dir.empty() || numerics_text.empty())
        die("--model, --tokens-file, --numerics and --out are required (--help)");
    Numerics n = Numerics::ForkCpu;
    if (numerics_text == "fork-cpu") n = Numerics::ForkCpu;
    else if (numerics_text == "fork-cuda") n = Numerics::ForkCuda;
    else if (numerics_text == "engine") n = Numerics::Engine;
    else die("--numerics " + numerics_text + ": fork-cpu, fork-cuda or engine");
    if (n_layers < 1 || n_layers > 2) die("--layers " + std::to_string(n_layers) + ": 1 or 2 (the window-only layers)");
    const std::vector<int32_t> tokens = read_tokens(tokens_file);
    const int64_t T = (int64_t) tokens.size();
    if (window_cells < T) die("--window-cells " + std::to_string(window_cells) + ": fewer than the prompt's tokens");

    const auto t0 = std::chrono::steady_clock::now();
    const e::HostModel host = e::load_gguf_model(model_path);
    // the host views of every weight (the dry model: no CUDA call, the engram tables a view of the mapping)
    e::Model model;
    std::string err;
    if (!model.init(host, true, false, err)) die(err);
    const e::DenseDims& d = model.dense_dims();
    for (int32_t t : tokens)
        if (t >= d.n_vocab) die("token " + std::to_string(t) + " is outside the vocabulary of " + std::to_string(d.n_vocab));
    for (int il = 0; il < n_layers; ++il) {
        if (model.layer(il).role.ratio != 0) die("layer " + std::to_string(il) + " is not window-only");
        e::cpu_experts_check(model.experts()[(size_t) il], d, e::CpuNumerics::Fork);
    }
    Oracle oracle{oracle_dir};
    if (!oracle_dir.empty()) {
        oracle.moe_dump = exists(oracle_dir + "/cur-L000-u0000.bin");
        if (!oracle.moe_dump && !exists(oracle_dir + "/ffn_norm.0-L000-u0000.bin"))
            die(oracle_dir + ": neither a dsv41-moe-dump (cur-L000-u0000.bin) nor a fork op dump (ffn_norm.0-L000-u0000.bin)");
        for (int il = 0; il < n_layers; ++il) {
            oracle.get<float>("cur", il, (size_t) (T * d.n_embd));
            oracle.get<float>("weights", il, (size_t) (T * d.n_expert_used));
            oracle.get<int32_t>("topk", il, (size_t) (T * d.n_expert_used));
        }
    }
    std::printf("dsv41-layer-ref: %s: %d layers, n_embd %d, %d experts (%d used); %lld tokens at positions 0 .. %lld, "
                "layers 0 .. %d, numerics %s%s%s, loaded in %.1f s\n", model_path.c_str(), model.n_layer(), d.n_embd,
                d.n_expert, d.n_expert_used, (long long) T, (long long) (T - 1), n_layers - 1, numerics_name(n),
                oracle_dir.empty() ? "" : "; the oracle's expert ids and the isolated MoE from ",
                oracle_dir.empty() ? "" : oracle_dir.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (dry) {
        std::printf("dsv41-layer-ref: dry run: every view checked%s; nothing computed\n",
                    oracle_dir.empty() ? "" : ", the oracle's cur, weights and topk of the call read and sized");
        return 0;
    }
    make_dirs(out_dir);
    Dumper dump;
    dump.dir = out_dir;
    LayerRef ref(model, n, T, window_cells, threads);
    const auto t1 = std::chrono::steady_clock::now();
    ref.run(tokens, n_layers, oracle_dir.empty() ? nullptr : &oracle, dump);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::ofstream mf(out_dir + "/layer-ref-manifest.json");
    mf << "{\"tool\": \"dsv41-layer-ref\", \"model_path\": " << json_string(model_path) << ", \"numerics\": \""
       << numerics_name(n) << "\", \"oracle\": " << json_string(oracle_dir) << ", \"n_tokens\": " << T << ", \"layers\": " << n_layers
       << ", \"window_cells\": " << window_cells << ", \"seconds\": " << secs << ",\n \"prompt_token_ids\": ["
       << join(tokens) << "],\n \"files\": [" << dump.files << "]}\n";
    if (!mf) die("writing " + out_dir + "/layer-ref-manifest.json failed");
    std::printf("dsv41-layer-ref: %s: %zu dumps (%.1f MiB) in %.1f s\n", out_dir.c_str(), dump.n,
                (double) dump.bytes / (1024.0 * 1024.0), secs);
    return 0;
}
