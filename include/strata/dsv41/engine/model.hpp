// include/strata/dsv41/engine/model.hpp - the model the engine runs: every tensor of SPEC.md "Tensors" read
// from a deepseek41 GGUF (or made by a test as a synthetic stand-in) on the host, then the dense part on the device as
// the weight views attention.hpp and dense.hpp take, the engram tables in pinned host RAM
// and the routed experts left on the host for experts.hpp's tiers.
//
// Starting point: Strata's GGUF reader (include/strata/artifact/gguf_reader.hpp,
// mmap, no ggml) with the meta_int_array for the engram arrays, and the NativeDense / NativeEmbed / NativeHead
// pattern of one device copy per weight; the architecture guard is Strata's check_architecture with DeepSeek-V4.1's
// keys (deepseek41, sqrt(softplus) gating, 4 hyper-connections, one shared expert, no hash layers). The layer roles
// come from attn_layers() over the compress ratios and the tensors each layer carries, as the fork's
// load_arch_tensors works them out (deepseek41.cpp 209-256).
#pragma once

#include "strata/dsv41/engine/attention.hpp"
#include "strata/dsv41/engine/dense.hpp"
#include "strata/dsv41/engine/engram.hpp"
#include "strata/dsv41/engine/qmatmul.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

/// GGUF type ids beyond qmatmul.hpp's: f32 vectors and the router's bf16.
constexpr int kTypeF32 = 0;
constexpr int kTypeBF16 = 30;

/// One tensor on the host: its GGUF type id, ne (fastest first) and its bytes, mapped from the file or owned by the
/// HostModel.
struct HostTensor {
    int type = 0;
    std::vector<int64_t> ne;
    const uint8_t* data = nullptr;
    size_t bytes = 0;
};

/// A model on the host: the geometry, the layer roles, the routed experts' SwiGLU limits, the engram hash and every
/// tensor by its GGUF name.
struct HostModel {
    std::string source;              ///< the GGUF path, or what made a synthetic model
    AttnDims attn;
    DenseDims dense;
    std::vector<AttnLayer> layers;   ///< attn_layers() of the compress ratios and the tensors present
    std::vector<float> clamp_exp;    ///< swiglu_clamp_exp per layer: the routed experts' limit
    EngramHash engram;
    int32_t eos_id = -1;             ///< tokenizer.ggml.eos_token_id (-1: the GGUF has none, as the tiny models)
    std::map<std::string, HostTensor> tensors;
    std::shared_ptr<const void> keep;   ///< what keeps the bytes alive: the GGUF mapping or a synthetic model's buffers

    const HostTensor* find(const std::string& name) const;
    /// Exits naming the source and the tensor when it is missing.
    const HostTensor& get(const std::string& name) const;
};

/// Opens a GGUF (every shard of a split model) and reads it as a HostModel: the architecture guard (deepseek41,
/// expert_gating_func 4, hyper_connection.count 4, one shared expert, no hash layers, one SwiGLU limit for the shared
/// experts), the dims, the layer roles, the engram arrays read in full, every tensor's type, shape and bytes. Exits
/// naming the file and the key or tensor on anything it cannot use.
HostModel load_gguf_model(const std::string& path);

/// The routed experts of one layer on the host as the GGUF stores them (ffn_up_exps and ffn_gate_exps [n_expert]
/// [n_ff][n_embd], ffn_down_exps [n_expert][n_embd][n_ff]): part p of expert e at data[p] + e * bytes[p], parts in
/// strata-moe's order (0 up, 1 gate, 2 down; moe_residency.h). The limit is the layer's swiglu_clamp_exp.
struct ExpertLayerHost {
    int layer = -1;
    int type[3] = {0, 0, 0};
    const uint8_t* data[3] = {nullptr, nullptr, nullptr};
    size_t bytes[3] = {0, 0, 0};
    std::string name[3];
    float limit = 0.f;
};

/// One layer's weights as the engine's ops take them.
struct LayerWeights {
    AttnLayer role;
    AttnLayerWeights attn;
    HcWeights hc_attn, hc_ffn;
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    RouterWeights router;
    SharedExpertWeights shared;
    int engram = -1;   ///< the engram index (EngramHash::index_of), -1 without one
    EngramWeights eng;
};

/// The model the forward pass reads. On the device: every dense weight in one arena, the attention matrices as stored
/// (q2_K) or, with attn_f16, dequantized to f16 (the cuBLAS test path), engram_q and engram_k dequantized to f32
/// (dense.hpp EngramWeights). In pinned host RAM: the engram tables. On the host as the HostModel has them: the routed
/// experts (experts.hpp places them). With dry (the dev VM, no CUDA call) every device view points at the host bytes
/// instead, as the tests' dry runs do: the host-side weight checks never read them.
class Model {
public:
    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model();

    /// False with `err` set when the device has no room for the arena or a table cannot be pinned. Runs check().
    bool init(const HostModel& h, bool dry, bool attn_f16, std::string& err);

    const HostModel& host() const { return *host_; }
    const AttnDims& attn_dims() const { return host_->attn; }
    const DenseDims& dense_dims() const { return host_->dense; }
    int n_layer() const { return (int) layers_.size(); }
    const LayerWeights& layer(int il) const { return layers_[(size_t) il]; }
    const std::vector<AttnLayer>& roles() const { return host_->layers; }
    const std::vector<ExpertLayerHost>& experts() const { return experts_; }
    const QWeightView& token_embd() const { return token_embd_; }
    const QWeightView& output() const { return output_; }
    const float* output_norm() const { return output_norm_; }
    /// The engram table of engram index e.
    const EngramTable& table(int e) const { return *tables_[(size_t) e]; }
    size_t device_bytes() const { return arena_bytes_; }
    size_t pinned_bytes() const { return pinned_bytes_; }
    bool dry() const { return dry_; }
    bool attn_f16() const { return attn_f16_; }

    /// Every view through the checks the ops make before their first launch (attn_check_weights, hc_, engram_,
    /// router_, shared_, embed_ and head_check_weights) and every routed expert part's size against the dims. Exits
    /// naming the layer and the tensor.
    void check() const;

private:
    const HostModel* host_ = nullptr;
    bool dry_ = false, attn_f16_ = false;
    std::vector<LayerWeights> layers_;
    std::vector<ExpertLayerHost> experts_;
    std::vector<std::unique_ptr<EngramTable>> tables_;
    QWeightView token_embd_, output_;
    const float* output_norm_ = nullptr;
    void* arena_ = nullptr;
    size_t arena_bytes_ = 0, pinned_bytes_ = 0;
    std::vector<std::vector<uint8_t>> host_copies_;   // f16 attention weights, f32 engram q and k (host side)
};

}  // namespace strata::dsv41::engine
