// src/dsv41/engine/model.cpp - the GGUF loader and the device model of include/strata/dsv41/engine/model.hpp.
//
// The loader is the tests' (dense_test.cpp, attention_test.cpp load_model) made one: the deepseek41.* keys into
// AttnDims and DenseDims, the roles from the compress ratios and the tensors present, the engram arrays through
// meta_int_array (the real token map has 129,280 entries, MetaValue::items keeps 64). The device model copies every
// dense weight into one arena (one cudaMalloc, each tensor 256-byte aligned: the router's GEMV reads 16 bytes at a time,
// qmatmul's q2_K 4) and the engram tables into pinned host RAM.
#include "strata/dsv41/engine/model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/dsv41/engine/attention_ref.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 model: %s\n", msg);
    std::exit(1);
}

const MetaValue* meta(const GgufFile& g, const std::string& key) { return g.get("deepseek41." + key); }

double meta_num(const GgufFile& g, const std::string& key) {
    const MetaValue* v = meta(g, key);
    if (!v || !v->is_num()) failf("%s: no numeric deepseek41.%s", g.path().c_str(), key.c_str());
    return v->num();
}

const MetaValue& meta_arr(const GgufFile& g, const std::string& key) {
    const MetaValue* v = meta(g, key);
    if (!v || v->type != MetaType::ARRAY) failf("%s: no array deepseek41.%s", g.path().c_str(), key.c_str());
    return *v;
}

// every element of an f32 array, from the mapped file (items keeps the first 64 only)
std::vector<float> meta_f32_array(const GgufFile& g, const std::string& key) {
    const MetaValue& v = meta_arr(g, key);
    if (v.elem != MetaType::F32 || !v.raw) failf("%s: deepseek41.%s is not an f32 array", g.path().c_str(), key.c_str());
    std::vector<float> out((size_t) v.count);
    std::memcpy(out.data(), v.raw, out.size() * sizeof(float));
    return out;
}

std::vector<int64_t> meta_ints(const GgufFile& g, const std::string& key) {
    std::vector<int64_t> out;
    try {
        for (uint64_t x : meta_int_array(meta_arr(g, key))) out.push_back((int64_t) x);
    } catch (const std::exception& e) {
        failf("%s: deepseek41.%s: %s", g.path().c_str(), key.c_str(), e.what());
    }
    return out;
}

constexpr size_t kArenaAlign = 256;
size_t align_up(size_t n) { return (n + kArenaAlign - 1) / kArenaAlign * kArenaAlign; }

}  // namespace

const HostTensor* HostModel::find(const std::string& name) const {
    const auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
}

const HostTensor& HostModel::get(const std::string& name) const {
    const HostTensor* t = find(name);
    if (!t) failf("%s: no tensor %s", source.c_str(), name.c_str());
    return *t;
}

HostModel load_gguf_model(const std::string& path) {
    std::shared_ptr<GgufModel> gm;
    try {
        gm = std::make_shared<GgufModel>(GgufModel::open(path));
    } catch (const std::exception& e) {
        failf("%s: %s", path.c_str(), e.what());
    }
    const GgufFile& g = gm->meta();
    HostModel h;
    h.source = path;
    h.keep = gm;
    // the architecture guard: the engine is DeepSeek-V4.1-Flash's only (Strata's check_architecture, its keys)
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) failf("%s: missing general.architecture", path.c_str());
    if (arch->s != "deepseek41")
        failf("%s: architecture is '%s', this engine requires 'deepseek41'", path.c_str(), arch->s.c_str());
    if ((int) meta_num(g, "hyper_connection.count") != kHc) failf("%s: hyper_connection.count is not %d", path.c_str(), kHc);
    // the end of a reply (--serve stops on it); the tiny models' no_vocab files have none
    if (const MetaValue* eos = g.get("tokenizer.ggml.eos_token_id"); eos && eos->is_num()) h.eos_id = (int32_t) eos->num();
    if ((int) meta_num(g, "expert_gating_func") != 4)
        failf("%s: expert_gating_func is not 4 (sqrtsoftplus, SPEC.md Divergences 1)", path.c_str());
    if (meta(g, "expert_shared_count") && (int) meta_num(g, "expert_shared_count") != 1)
        failf("%s: expert_shared_count is not 1", path.c_str());
    if (meta(g, "hash_layer_count") && (int) meta_num(g, "hash_layer_count") != 0)
        failf("%s: hash_layer_count is not 0 (the engine routes every layer through the gate)", path.c_str());

    AttnDims& a = h.attn;
    a.n_embd = (int) meta_num(g, "embedding_length");
    a.n_head = (int) meta_num(g, "attention.head_count");
    a.head_dim = (int) meta_num(g, "attention.key_length");
    a.n_rot = (int) meta_num(g, "rope.dimension_count");
    a.q_lora = (int) meta_num(g, "attention.q_lora_rank");
    a.o_groups = (int) meta_num(g, "attention.output_group_count");
    a.o_lora = (int) meta_num(g, "attention.output_lora_rank");
    a.n_swa = (int) meta_num(g, "attention.sliding_window");
    a.idx_heads = (int) meta_num(g, "attention.indexer.head_count");
    a.idx_dim = (int) meta_num(g, "attention.indexer.key_length");
    a.top_k = (int) meta_num(g, "attention.indexer.top_k");
    a.rms_eps = (float) meta_num(g, "attention.layer_norm_rms_epsilon");
    a.rope_base = (float) meta_num(g, "rope.freq_base");
    a.comp_rope_base = (float) meta_num(g, "attention.compress_rope_freq_base");
    a.yarn_factor = (float) meta_num(g, "rope.scaling.factor");
    a.yarn_orig_ctx = (int) meta_num(g, "rope.scaling.original_context_length");
    a.yarn_beta_fast = (float) meta_num(g, "rope.scaling.yarn_beta_fast");
    a.yarn_beta_slow = (float) meta_num(g, "rope.scaling.yarn_beta_slow");
    if ((int) meta_num(g, "attention.head_count_kv") != 1) failf("%s: attention.head_count_kv is not 1 (MQA)", path.c_str());

    DenseDims& d = h.dense;
    d.n_embd = a.n_embd;
    d.n_expert = (int) meta_num(g, "expert_count");
    d.n_expert_used = (int) meta_num(g, "expert_used_count");
    d.n_ff = (int) meta_num(g, "expert_feed_forward_length");
    d.rms_eps = a.rms_eps;
    d.hc_eps = (float) meta_num(g, "hyper_connection.epsilon");
    d.sinkhorn_iters = (int) meta_num(g, "hyper_connection.sinkhorn_iterations");
    d.weights_norm = meta_num(g, "expert_weights_norm") != 0;
    d.weights_scale = (float) meta_num(g, "expert_weights_scale");
    d.engram_heads = (int) meta_num(g, "engram.head_count");
    d.engram_key = (int) meta_num(g, "engram.key_length");
    d.engram_ngram = (int) meta_num(g, "engram.max_ngram_size");

    const int n_layer = (int) meta_num(g, "block_count");
    if (n_layer < 1) failf("%s: block_count %d", path.c_str(), n_layer);
    const std::vector<float> clamp_sh = meta_f32_array(g, "swiglu_clamp_shexp");
    h.clamp_exp = meta_f32_array(g, "swiglu_clamp_exp");
    if ((int) clamp_sh.size() < n_layer || (int) h.clamp_exp.size() < n_layer)
        failf("%s: swiglu_clamp_shexp / _exp have %zu / %zu values for %d layers", path.c_str(), clamp_sh.size(),
              h.clamp_exp.size(), n_layer);
    h.clamp_exp.resize((size_t) n_layer);
    for (int i = 0; i < n_layer; ++i) {
        if (clamp_sh[(size_t) i] != clamp_sh[0])
            failf("%s: swiglu_clamp_shexp differs between layers (DenseDims has one limit)", path.c_str());
        // the CPU experts are the fork's clamped SwiGLU only (ggml_swiglu_clamp); 0 would be ggml_swiglu_split
        if (!(h.clamp_exp[(size_t) i] > 0.f)) failf("%s: swiglu_clamp_exp of layer %d is %g", path.c_str(), i, h.clamp_exp[(size_t) i]);
    }
    d.swiglu_limit = clamp_sh[0];

    EngramHash& eh = h.engram;
    eh.n_gram = d.engram_ngram;
    eh.heads = d.engram_heads;
    eh.pad = (int64_t) meta_num(g, "engram.pad_id");
    for (int64_t v : meta_ints(g, "engram.layer_ids")) eh.layer_ids.push_back((int32_t) v);
    for (int64_t v : meta_ints(g, "engram.token_map")) eh.token_map.push_back((int32_t) v);
    for (int64_t v : meta_ints(g, "engram.multipliers")) eh.multipliers.push_back((uint64_t) v);
    for (int64_t v : meta_ints(g, "engram.primes")) eh.primes.push_back((uint64_t) v);
    for (int64_t v : meta_ints(g, "engram.offsets")) eh.offsets.push_back((uint64_t) v);
    eh.check();

    // every tensor of every shard, by name
    for (size_t s = 0; s < gm->size(); ++s) {
        const GgufFile& f = gm->shard(s);
        for (const TensorInfo& t : f.tensors()) {
            if (!gm->in_bounds(t, s))
                failf("%s: tensor %s (%s) is not inside its file", f.path().c_str(), t.name.c_str(), t.type_name());
            HostTensor x;
            x.type = (int) t.type;
            for (uint64_t n : t.shape) x.ne.push_back((int64_t) n);
            x.data = f.tensor_data(t);
            x.bytes = (size_t) tensor_payload_bytes(t);
            h.tensors.emplace(t.name, std::move(x));
        }
    }
    const HostTensor& out = h.get("output.weight");
    if (out.ne.size() != 2) failf("%s: output.weight is not 2D", path.c_str());
    d.n_vocab = (int) out.ne[1];

    // the roles from the compress ratios and the tensors each layer carries (deepseek41.cpp 209-256)
    const std::vector<int64_t> ratios = meta_ints(g, "attention.compress_ratios");
    if ((int) ratios.size() < n_layer) failf("%s: %zu compress ratios for %d layers", path.c_str(), ratios.size(), n_layer);
    std::vector<int> r((size_t) n_layer), flags((size_t) n_layer, 0);
    auto has = [&](int i, const char* t) { return h.find("blk." + std::to_string(i) + "." + t + ".weight") != nullptr; };
    for (int i = 0; i < n_layer; ++i) {
        r[(size_t) i] = (int) ratios[(size_t) i];
        flags[(size_t) i] = (has(i, "attn_compressor_kv") ? kKvSource : 0) | (has(i, "attn_compressor_gate") ? kGate : 0) |
                            (has(i, "indexer.attn_k") ? kKeyOwner : 0) | (has(i, "indexer.attn_q_b") ? kIndexSource : 0);
    }
    h.layers = attn_layers(r, flags);
    return h;
}

Model::~Model() {
    tables_.clear();
    if (arena_) cudaFree(arena_);
}

bool Model::init(const HostModel& h, bool dry, bool attn_f16, std::string& err) {
    if (host_) failf("Model::init called twice");
    host_ = &h;
    dry_ = dry;
    attn_f16_ = attn_f16;
    const AttnDims& ad = h.attn;
    const DenseDims& dd = h.dense;
    const int n_layer = (int) h.layers.size();

    // what goes into the arena: (host bytes, size, where its device pointer goes); laid out first, copied after
    struct Piece {
        const void* src;
        size_t bytes;
        const void** dst;
    };
    std::vector<Piece> pieces;
    pieces.reserve((size_t) n_layer * 40);
    auto put = [&](const void* src, size_t bytes, const void** dst) { pieces.push_back({src, bytes, dst}); };
    auto matrix = [&](const std::string& name, QWeightView& v) {
        const HostTensor& t = h.get(name);
        if (t.ne.size() != 2) failf("%s: %s is not 2D", h.source.c_str(), name.c_str());
        v = QWeightView{t.type, t.ne[0], t.ne[1], nullptr};
        put(t.data, t.bytes, &v.data);
    };
    auto vec_f32 = [&](const std::string& name, int64_t n, const float*& v) {
        const HostTensor& t = h.get(name);
        int64_t count = 1;
        for (int64_t x : t.ne) count *= x;
        if (t.type != kTypeF32 || count != n)
            failf("%s: %s is type %d of %lld values, expected f32 of %lld", h.source.c_str(), name.c_str(), t.type,
                  (long long) count, (long long) n);
        put(t.data, t.bytes, reinterpret_cast<const void**>(&v));
    };
    // a host-made copy (f16 attention weights, f32 engram q and k) that the arena then holds; a moved vector keeps its
    // buffer, so host_copies_ may grow while pieces point into its elements
    auto own = [&](std::vector<uint8_t> bytes, const void** dst) {
        host_copies_.push_back(std::move(bytes));
        put(host_copies_.back().data(), host_copies_.back().size(), dst);
    };
    auto dequant_f32 = [&](const std::string& name) {
        const HostTensor& t = h.get(name);
        if (t.type != kQTypeQ2K || t.ne.size() != 2) failf("%s: %s is not a 2D q2_K matrix", h.source.c_str(), name.c_str());
        const std::vector<float> v = ref_dequant(RefMatrix{kRefTypeQ2K, t.ne[0], t.ne[1], t.data});
        std::vector<uint8_t> b(v.size() * sizeof(float));
        std::memcpy(b.data(), v.data(), b.size());
        return b;
    };

    matrix("token_embd.weight", token_embd_);
    matrix("output.weight", output_);
    vec_f32("output_norm.weight", dd.n_embd, output_norm_);
    layers_.assign((size_t) n_layer, LayerWeights{});
    experts_.assign((size_t) n_layer, ExpertLayerHost{});
    for (int i = 0; i < n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        LayerWeights& L = layers_[(size_t) i];
        L.role = h.layers[(size_t) i];
        AttnLayerWeights& w = L.attn;
        // the attention's matrices: as stored, or dequantized to f16 for the cuBLAS test path (--f16)
        auto attn_matrix = [&](const char* name, bool present, QWeightView& v) {
            if (!present) return;
            if (!attn_f16) return matrix(b + name + ".weight", v);
            const HostTensor& t = h.get(b + name + ".weight");
            if (t.type != kQTypeQ2K || t.ne.size() != 2)
                failf("%s: %s%s is not a 2D q2_K matrix: --f16 dequantizes q2_K", h.source.c_str(), b.c_str(), name);
            const std::vector<uint16_t> f16 = ref_to_f16(ref_dequant(RefMatrix{kRefTypeQ2K, t.ne[0], t.ne[1], t.data}));
            std::vector<uint8_t> bytes(f16.size() * 2);
            std::memcpy(bytes.data(), f16.data(), bytes.size());
            v = QWeightView{kQTypeF16, t.ne[0], t.ne[1], nullptr};
            own(std::move(bytes), &v.data);
        };
        const AttnLayer& l = L.role;
        attn_matrix("attn_q_a", true, w.wq_a);
        attn_matrix("attn_q_b", true, w.wq_b);
        attn_matrix("attn_kv", true, w.wkv);
        attn_matrix("attn_output_a", true, w.wo_a);
        attn_matrix("attn_output_b", true, w.wo_b);
        attn_matrix("attn_compressor_kv", l.kv_source, w.comp_kv);
        attn_matrix("attn_compressor_gate", l.gate, w.comp_gate);
        attn_matrix("indexer.attn_k", l.key_owner, w.idx_k);
        attn_matrix("indexer.attn_q_b", l.index_source, w.idx_q_b);
        attn_matrix("indexer.proj", l.index_source, w.idx_proj);
        vec_f32(b + "attn_q_a_norm.weight", ad.q_lora, w.q_a_norm);
        vec_f32(b + "attn_kv_a_norm.weight", ad.head_dim, w.kv_norm);
        if (l.kv_source) vec_f32(b + "attn_compressor_norm.weight", ad.head_dim, w.comp_norm);
        if (l.key_owner) vec_f32(b + "indexer.k_norm.weight", ad.idx_dim, w.idx_k_norm);
        vec_f32(b + "attn_sinks.weight", ad.n_head, w.sinks);

        matrix(b + "hc_attn_fn.weight", L.hc_attn.fn);
        vec_f32(b + "hc_attn_base.weight", kHcMix, L.hc_attn.base);
        vec_f32(b + "hc_attn_scale.weight", 3, L.hc_attn.scale);
        matrix(b + "hc_ffn_fn.weight", L.hc_ffn.fn);
        vec_f32(b + "hc_ffn_base.weight", kHcMix, L.hc_ffn.base);
        vec_f32(b + "hc_ffn_scale.weight", 3, L.hc_ffn.scale);
        vec_f32(b + "attn_norm.weight", dd.n_embd, L.attn_norm);
        vec_f32(b + "ffn_norm.weight", dd.n_embd, L.ffn_norm);
        {
            const HostTensor& t = h.get(b + "ffn_gate_inp.weight");
            if (t.type != kTypeBF16 || t.ne.size() != 2 || t.ne[0] != dd.n_embd || t.ne[1] != dd.n_expert)
                failf("%s: %sffn_gate_inp.weight is not bf16 [%d, %d]", h.source.c_str(), b.c_str(), dd.n_embd, dd.n_expert);
            put(t.data, t.bytes, reinterpret_cast<const void**>(&L.router.gate_inp));
        }
        vec_f32(b + "exp_probs_b.bias", dd.n_expert, L.router.bias);
        matrix(b + "ffn_gate_shexp.weight", L.shared.gate);
        matrix(b + "ffn_up_shexp.weight", L.shared.up);
        matrix(b + "ffn_down_shexp.weight", L.shared.down);

        L.engram = h.engram.index_of(i);
        if (L.engram >= 0) {
            matrix(b + "engram_wkv.weight", L.eng.wkv);
            own(dequant_f32(b + "engram_q.weight"), reinterpret_cast<const void**>(&L.eng.q));
            own(dequant_f32(b + "engram_k.weight"), reinterpret_cast<const void**>(&L.eng.k));
        }

        // the routed experts stay on the host: experts.hpp places them
        ExpertLayerHost& x = experts_[(size_t) i];
        x.layer = i;
        x.limit = h.clamp_exp[(size_t) i];
        const char* parts[3] = {"ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"};
        for (int p = 0; p < 3; ++p) {
            x.name[p] = b + parts[p] + ".weight";
            const HostTensor& t = h.get(x.name[p]);
            if (t.ne.size() != 3 || t.ne[2] != dd.n_expert)
                failf("%s: %s is not [., ., %d]", h.source.c_str(), x.name[p].c_str(), dd.n_expert);
            x.type[p] = t.type;
            x.data[p] = t.data;
            x.bytes[p] = t.bytes / (size_t) dd.n_expert;
        }
    }

    // the engram tables: pinned host RAM, a view of the mapping in a dry run
    for (size_t e = 0; e < h.engram.layer_ids.size(); ++e) {
        const int il = h.engram.layer_ids[e];
        const HostTensor& t = h.get("blk." + std::to_string(il) + ".engram_embd.weight");
        if (t.type != kQTypeQ2K || t.ne.size() != 2 || t.ne[0] != dd.engram_key)
            failf("%s: engram_embd of layer %d is not q2_K rows of %d", h.source.c_str(), il, dd.engram_key);
        if (t.ne[1] < h.engram.rows_needed((int) e))
            failf("%s: engram_embd of layer %d has %lld rows, its buckets need %lld", h.source.c_str(), il,
                  (long long) t.ne[1], (long long) h.engram.rows_needed((int) e));
        const int row_bytes = (int) (t.ne[0] / kQK * (int64_t) qtype_block_bytes(kQTypeQ2K));
        auto tab = std::make_unique<EngramTable>();
        if (dry) {
            tab->view(t.data, t.ne[1], row_bytes);
        } else {
            if (!tab->alloc(t.ne[1], row_bytes, err)) return false;
            std::memcpy(tab->data(), t.data, t.bytes);
            pinned_bytes_ += t.bytes;
        }
        tables_.push_back(std::move(tab));
    }

    // the arena: one allocation, then every piece at its aligned offset
    size_t total = 0;
    for (const Piece& p : pieces) total = align_up(total) + p.bytes;
    if (dry) {
        for (const Piece& p : pieces) *p.dst = p.src;
    } else {
        if (const cudaError_t e = cudaMalloc(&arena_, std::max<size_t>(total, 16)); e != cudaSuccess) {
            err = "the model's dense weights (" + std::to_string(total >> 20) + " MiB) on the device: " + cudaGetErrorString(e);
            return false;
        }
        arena_bytes_ = total;
        size_t off = 0;
        for (const Piece& p : pieces) {
            off = align_up(off);
            char* dev = static_cast<char*>(arena_) + off;
            if (const cudaError_t e = cudaMemcpy(dev, p.src, p.bytes, cudaMemcpyHostToDevice); e != cudaSuccess) {
                err = std::string("uploading the model's weights: ") + cudaGetErrorString(e);
                return false;
            }
            *p.dst = dev;
            off += p.bytes;
        }
    }
    check();
    return true;
}

void Model::check() const {
    const AttnDims& ad = host_->attn;
    const DenseDims& dd = host_->dense;
    embed_check_weights(dd, token_embd_, "token_embd");
    head_check_weights(dd, output_, "output");
    if (!output_norm_) failf("output_norm is missing");
    char what[96];
    for (int i = 0; i < n_layer(); ++i) {
        const LayerWeights& L = layers_[(size_t) i];
        attn_check_weights(ad, L.role, L.attn);
        std::snprintf(what, sizeof what, "layer %d hc_attn", i);
        hc_check_weights(dd, L.hc_attn, what);
        std::snprintf(what, sizeof what, "layer %d hc_ffn", i);
        hc_check_weights(dd, L.hc_ffn, what);
        std::snprintf(what, sizeof what, "layer %d router", i);
        router_check_weights(dd, L.router, what);
        std::snprintf(what, sizeof what, "layer %d shared expert", i);
        shared_check_weights(dd, L.shared, what);
        if (!L.attn_norm || !L.ffn_norm) failf("layer %d: attn_norm or ffn_norm is missing", i);
        if (L.engram >= 0) {
            std::snprintf(what, sizeof what, "layer %d engram", i);
            engram_check_weights(dd, L.eng, what);
        }
        // a routed expert's parts: [n_ff][E] up and gate, [E][n_ff] down, whole blocks of a qmatmul type
        const ExpertLayerHost& x = experts_[(size_t) i];
        for (int p = 0; p < 3; ++p) {
            const int64_t k = p == 2 ? dd.n_ff : dd.n_embd, rows = p == 2 ? dd.n_embd : dd.n_ff;
            const QWeightView v{x.type[p], k, rows, x.data[p]};
            if (qtype_block_bytes(x.type[p]) == 0 || x.bytes[p] != qweight_bytes(v))
                failf("%s: %s is type %d with %zu bytes per expert, expected a K-quant of [%lld][%lld]",
                      host_->source.c_str(), x.name[p].c_str(), x.type[p], x.bytes[p], (long long) rows, (long long) k);
            qmatmul_check(v, 1, x.name[p].c_str());
        }
    }
}

}  // namespace strata::dsv41::engine
