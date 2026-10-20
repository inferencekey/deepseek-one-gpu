// src/dsv41/moe_hot_set.cpp - the hot-set file's loader (moe_hot_set.h), with no GPU and no ggml backend: the
// GPU tier of strata-moe (moe_gpu.cpp) and the dedicated engine (src/dsv41/engine/experts.cpp) read the same files.
#include "moe_hot_set.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <exception>
#include <fstream>
#include <set>
#include <string>

namespace strata::dsv41 {
namespace {

constexpr const char* kHotSetFormat = "strata-moe-hot-set/1";

ggml_type type_by_name(const std::string& name, const char* path, int layer, const char* role) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (ggml_get_type_traits((ggml_type) t)->blck_size > 0 && name == ggml_type_name((ggml_type) t)) {
            return (ggml_type) t;
        }
    }
    GGML_ABORT("strata-moe: hot set %s, layer %d: %s type '%s' is not a ggml type", path, layer, role, name.c_str());
}

}  // namespace

HotSet load_hot_set(const char* path) {
    using nlohmann::json;
    HotSet hot;
    hot.path = path;
    std::ifstream f(path);
    if (!f) GGML_ABORT("strata-moe: STRATA_MOE_HOT=%s: cannot open the file", path);
    try {
        const json j = json::parse(f);
        const std::string format = j.at("format").get<std::string>();
        if (format != kHotSetFormat) {
            GGML_ABORT("strata-moe: hot set %s: format '%s', expected '%s'", path, format.c_str(), kHotSetFormat);
        }
        hot.budget_gib = j.at("budget_gib").get<double>();
        // the warm lists (hot_set.py --ram-budget-gib) are optional: a file without them has no warm tier
        if (j.contains("ram_budget_gib")) hot.ram_budget_gib = j.at("ram_budget_gib").get<double>();
        std::set<int> seen;
        for (const json& jl : j.at("layers")) {
            HotLayer l;
            l.layer = jl.at("layer").get<int>();
            l.n_expert = jl.at("n_expert").get<int64_t>();
            l.n_embd = jl.at("n_embd").get<int64_t>();
            l.n_ff = jl.at("n_ff").get<int64_t>();
            if (l.layer < 0 || !seen.insert(l.layer).second) {
                GGML_ABORT("strata-moe: hot set %s: layer %d is negative or listed twice", path, l.layer);
            }
            if (l.n_expert <= 0 || l.n_embd <= 0 || l.n_ff <= 0) {
                GGML_ABORT("strata-moe: hot set %s, layer %d: n_expert %lld, n_embd %lld, n_ff %lld", path, l.layer,
                           (long long) l.n_expert, (long long) l.n_embd, (long long) l.n_ff);
            }
            const json& types = jl.at("types");
            l.up_type = type_by_name(types.at("up").get<std::string>(), path, l.layer, "up");
            l.gate_type = type_by_name(types.at("gate").get<std::string>(), path, l.layer, "gate");
            l.down_type = type_by_name(types.at("down").get<std::string>(), path, l.layer, "down");
            std::vector<uint8_t> used((size_t) l.n_expert, 0);
            for (const json& je : jl.at("experts")) {
                const int64_t e = je.get<int64_t>();
                if (e < 0 || e >= l.n_expert || used[(size_t) e]) {
                    GGML_ABORT("strata-moe: hot set %s, layer %d: expert %lld is outside [0, %lld) or listed twice",
                               path, l.layer, (long long) e, (long long) l.n_expert);
                }
                used[(size_t) e] = 1;
                l.experts.push_back((int32_t) e);
            }
            if (hot.ram_budget_gib >= 0.0) {
                for (const json& je : jl.at("warm")) {
                    const int64_t e = je.get<int64_t>();
                    if (e < 0 || e >= l.n_expert || used[(size_t) e]) {
                        GGML_ABORT("strata-moe: hot set %s, layer %d: warm expert %lld is outside [0, %lld), or "
                                   "listed twice or also in VRAM", path, l.layer, (long long) e, (long long) l.n_expert);
                    }
                    used[(size_t) e] = 1;
                    l.warm.push_back((int32_t) e);
                }
            }
            hot.layers.push_back(std::move(l));
        }
    } catch (const std::exception& e) {
        GGML_ABORT("strata-moe: hot set %s: %s", path, e.what());
    }
    return hot;
}

}  // namespace strata::dsv41
