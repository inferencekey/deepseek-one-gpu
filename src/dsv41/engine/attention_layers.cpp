// src/dsv41/engine/attention_layers.cpp - the attention geometry, the layer roles and the rope arguments, shared by
// the CUDA path (attention.cu) and the CPU reference (attention_ref.cpp). See include/strata/dsv41/engine/attention.hpp.
#include "strata/dsv41/engine/attention.hpp"

#include "fork_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine {

AttnDims attn_dims_real() { return AttnDims{}; }

AttnRole AttnLayer::role() const {
    if (ratio == 0) return AttnRole::Window;
    if (index_source) return kv_source ? AttnRole::Full : AttnRole::Reindex;
    return AttnRole::Reuse;
}

std::vector<AttnLayer> attn_layers(const std::vector<int>& ratios, const std::vector<int>& flags) {
    if (ratios.size() != flags.size()) fork_cpu::fail("attn_layers", "ratios and flags differ in length");
    std::vector<AttnLayer> out(ratios.size());
    int last[3] = {-1, -1, -1};
    for (size_t i = 0; i < ratios.size(); ++i) {
        AttnLayer& l = out[i];
        l.index = (int) i;
        l.ratio = ratios[i];
        l.kv_source = flags[i] & kKvSource;
        l.gate = flags[i] & kGate;
        l.key_owner = flags[i] & kKeyOwner;
        l.index_source = flags[i] & kIndexSource;
        if (l.kv_source) last[0] = (int) i;
        if (l.key_owner) last[1] = (int) i;
        if (l.index_source) last[2] = (int) i;
        if (l.ratio == 0) continue;
        char msg[160];
        if (last[0] < 0 || last[1] < 0 || last[2] < 0) {
            std::snprintf(msg, sizeof msg, "layer %zu reads a compressed stream before any layer publishes one", i);
            fork_cpu::fail("attn_layers", msg);
        }
        if (ratios[(size_t) last[0]] != l.ratio) {
            std::snprintf(msg, sizeof msg, "layer %zu compresses at ratio %d but reads layer %d at ratio %d", i, l.ratio,
                          last[0], ratios[(size_t) last[0]]);
            fork_cpu::fail("attn_layers", msg);
        }
        if (l.kv_source && !l.gate && l.ratio != 1) {
            std::snprintf(msg, sizeof msg, "layer %zu pools %d tokens per row without a gate", i, l.ratio);
            fork_cpu::fail("attn_layers", msg);
        }
        l.kv_src = last[0];
        l.key_src = last[1];
        l.topk_src = last[2];
    }
    return out;
}

std::vector<AttnLayer> attn_layers_real() {
    // SPEC.md "Layer roles": ratio 0 for 0-1, 2 for 2-19, 1 for 20-39; sources 2, 8, 14 (gate), 20 (no gate); key
    // owners 2, 8, 14, 20; index sources 2, 8, 14, 20, 24, 28, 32, 36
    std::vector<int> ratios(40), flags(40, 0);
    for (int i = 0; i < 40; ++i) ratios[(size_t) i] = i < 2 ? 0 : i < 20 ? 2 : 1;
    for (int i : {2, 8, 14}) flags[(size_t) i] = kKvSource | kGate | kKeyOwner | kIndexSource;
    flags[20] = kKvSource | kKeyOwner | kIndexSource;
    for (int i : {24, 28, 32, 36}) flags[(size_t) i] = kIndexSource;
    return attn_layers(ratios, flags);
}

RopeArgs attn_rope_args(const AttnDims& d, bool compressed) {
    fork_cpu::RopeCfg c;
    c.n_rot = d.n_rot;
    if (compressed) {
        c.base = d.comp_rope_base;
        c.freq_scale = 1.0f / d.yarn_factor;
        c.ext_factor = 1.0f;
        c.attn_factor = fork_cpu::dsv4_rope_attn_factor(c.freq_scale, c.ext_factor);
        c.beta_fast = d.yarn_beta_fast;
        c.beta_slow = d.yarn_beta_slow;
        c.n_ctx_orig = d.yarn_orig_ctx;
    } else {
        c.base = d.rope_base;
    }
    c = fork_cpu::rope_cfg_finish(c);
    RopeArgs a;
    a.theta_scale = c.theta_scale;
    a.freq_scale = c.freq_scale;
    a.ext_factor = c.ext_factor;
    a.attn_factor = c.attn_factor;
    a.mscale = c.ext_factor != 0.0f ? c.attn_factor * std::fma(0.1f, logf(1.0f / c.freq_scale), 1.0f) : c.attn_factor;
    a.corr0 = c.corr0;
    a.corr1 = c.corr1;
    a.n_rot = c.n_rot;
    return a;
}

}  // namespace strata::dsv41::engine
