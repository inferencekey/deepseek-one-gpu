// op-dump: runs token ids with greedy decoding through the stock llama.cpp fork and records, for every evaluated
// ubatch, the inputs and outputs of the DeepSeek-V4.1 ops listed in tools/dsv41/engine/SPEC.md: every graph tensor
// whose callback name is one of the spec's op names (or every named tensor with --ops all). The engine's kernels are
// checked against these files op by op (tools/dsv41/engine/op_dump.py reads and compares them).
//
// It only reads the graph through the eval callback, so the fork stays unmodified. A name can occur more than once
// per layer and ubatch (build_norm names every normalization "norm", build_hc_mixes runs twice per layer), so each
// file carries the occurrence index in graph order, which is the same in every run of the same graph.

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <array>
#include <clocale>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::ordered_json;

[[noreturn]] static void fail(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "op-dump: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::exit(1);
}

// The callback names of the ops tools/dsv41/engine/SPEC.md specifies, in forward order. The per layer ones carry
// "-<layer>"; hc_init, hc_pre_init, hc_out, result_norm and result_output are model level (layer -1 in the dump).
static const char * const SPEC_OPS[] = {
    // embedding and hyper-connection stream
    "embd", "hc_init", "hc_pre_init",
    // engram (before the block)
    "engram_embd", "engram_kv", "engram_gate", "engram_out",
    // hyper-connection mixes, collapse and expansion (attention and FFN sublayers)
    "hc_mixes", "hc_pre", "hc_post", "hc_comb", "hc_attn_pre", "hc_attn_post", "hc_ffn_pre", "l_last",
    // norms; build_norm also names every normalized tensor "norm" before its weight, which this list leaves out:
    // each weighted result has a name of its own (the window KV norm is inside "kv", the compressor's in comp_kv)
    "attn_norm", "ffn_norm", "qr",
    // attention: query, window KV, compressor, index keys, indexer, sparse attention, output projection
    "q", "kv", "comp_state_kv", "comp_state_score", "comp_kv", "idx_k_new", "idx_q", "idx_q_rot", "idx_weights",
    "idx_k", "idx_score", "idx_top_k", "raw_k", "comp_k", "k_all", "comp_top_k_mask", "kq_mask", "kq",
    "kq_soft_max", "kqv", "attn_raw", "attn_out_raw", "attn_derope", "attn_wo_a", "attn_out",
    // router and experts
    "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_probs_biased", "ffn_moe_argsort", "ffn_moe_topk", "ffn_moe_weights",
    "ffn_moe_weights_sum", "ffn_moe_weights_sum_clamped", "ffn_moe_weights_norm", "ffn_moe_weights_scaled", "ffn_moe_up", "ffn_moe_gate",
    "ffn_moe_swiglu_limited", "ffn_moe_down", "ffn_moe_weighted", "ffn_moe_out", "ffn_up", "ffn_gate",
    "ffn_swiglu_limited", "ffn_shexp", "ffn_out",
    // head
    "hc_out", "result_norm", "result_output",
};

struct host_tensor {
    ggml_type              type;
    std::array<int64_t, 4> ne;
    std::vector<uint8_t>   data; // packed, ne[0] fastest
};

struct dump_state {
    bool              all_ops = false;
    std::set<std::string> ops;
    fs::path          dir;   // empty: no files, only the run
    int               decode_call = 0;
    int               ubatch      = -1;
    bool              decode_open = false; // a llama_decode call started and has computed nothing yet
    bool              seen_hc_init = false; // the current ubatch's hc_init was computed
    int64_t           n_pos       = 0;
    std::map<std::tuple<std::string, int>, int> occurrence; // (base, layer) -> files so far in this ubatch
    std::map<std::string, int> files_per_op;
    json files    = json::array();
    json ubatches = json::array();
};

static const char * type_dtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_F16: return "f16";
        case GGML_TYPE_I32: return "i32";
        default:            return nullptr;
    }
}

// Copies t from whichever backend holds it into a packed host buffer, following its strides (views such as
// ffn_moe_topk keep their source's row stride).
static host_tensor copy_to_host(const ggml_tensor * t) {
    if (t->buffer == nullptr && (t->view_src == nullptr || t->view_src->buffer == nullptr)) {
        fail("tensor %s (op %s) has no backend buffer", t->name, ggml_op_desc(t));
    }
    host_tensor h;
    h.type = t->type;
    for (int i = 0; i < 4; ++i) {
        h.ne[i] = t->ne[i];
    }
    // the bytes between the first and the last element, which is what a strided view spans
    size_t span = ggml_type_size(t->type);
    for (int i = 0; i < 4; ++i) {
        span += (size_t) (t->ne[i] - 1) * t->nb[i];
    }
    std::vector<uint8_t> raw(span);
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

    const size_t esize = ggml_type_size(t->type);
    h.data.resize(ggml_nelements(t) * esize);
    uint8_t * dst = h.data.data();
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t off = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    std::memcpy(dst, raw.data() + off, esize);
                    dst += esize;
                }
            }
        }
    }
    return h;
}

// "<base>-<layer>" as named by llama_context::graph_get_cb, or a model level "<base>" (layer -1)
static void split_name(const char * name, std::string & base, int & layer) {
    const char * dash = std::strrchr(name, '-');
    if (dash != nullptr && dash[1] != '\0') {
        char * end = nullptr;
        const long il = std::strtol(dash + 1, &end, 10);
        if (*end == '\0' && il >= 0) {
            base.assign(name, dash - name);
            layer = (int) il;
            return;
        }
    }
    base  = name;
    layer = -1;
}

static json src_names(const ggml_tensor * t) {
    json out = json::array();
    for (int i = 0; i < GGML_MAX_SRC && t->src[i] != nullptr; ++i) {
        out.push_back(t->src[i]->name);
    }
    return out;
}

static bool dump_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto & st = *static_cast<dump_state *>(user_data);
    if (t->name[0] == '\0') {
        return false;
    }
    std::string base;
    int         layer = -1;
    split_name(t->name, base, layer);
    const bool is_hc_init = base == "hc_init" && layer == -1;
    const bool wanted       = is_hc_init || (st.all_ops ? true : st.ops.count(base) > 0);
    if (!wanted || type_dtype(t->type) == nullptr) {
        // a wanted op of another type is a spec error, not something to skip
        if (wanted && !st.all_ops) {
            fail("%s (op %s) has type %s, the dump supports f32, f16 and i32", t->name, ggml_op_desc(t),
                 ggml_type_name(t->type));
        }
        return false;
    }
    if (ask) {
        return true;
    }

    // a ubatch opens with the first tensor a llama_decode call computes, or with the hc_init (the first op of
    // every DeepSeek-V4.1 graph) of a further ubatch of the same call
    if (st.decode_open || (is_hc_init && st.seen_hc_init)) {
        st.ubatch++;
        st.decode_open  = false;
        st.seen_hc_init = false;
        st.occurrence.clear();
        st.ubatches.push_back({ { "ubatch", st.ubatch }, { "decode_call", st.decode_call }, { "pos0", st.n_pos } });
    }
    if (is_hc_init) {
        // hc_init is [n_embd, hc, n_tokens]
        st.seen_hc_init = true;
        st.ubatches.back()["n_tokens"] = t->ne[2];
        st.n_pos += t->ne[2];
    }
    if (st.ubatch < 0) {
        fail("%s was computed outside a llama_decode call", t->name);
    }
    const int occ = st.occurrence[{ base, layer }]++;
    st.files_per_op[base]++;
    if (st.dir.empty()) {
        return true;
    }

    const host_tensor h = copy_to_host(t);
    char name[160];
    std::snprintf(name, sizeof(name), "%s.%d-L%03d-u%04d.bin", base.c_str(), occ, layer < 0 ? 999 : layer, st.ubatch);
    const fs::path path = st.dir / name;
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(h.data.data()), (std::streamsize) h.data.size());
    if (!f) {
        fail("cannot write %s (%zu bytes)", path.c_str(), h.data.size());
    }
    st.files.push_back({
        { "file", name },
        { "op", base },
        { "occurrence", occ },
        { "layer", layer },
        { "ubatch", st.ubatch },
        { "dtype", type_dtype(h.type) },
        { "ne", h.ne },
        { "source_tensor", t->name },
        { "ggml_op", ggml_op_desc(t) },
        { "src", src_names(t) },
        { "bytes", h.data.size() },
    });
    return true;
}

static void print_usage(int, char ** argv) {
    std::printf("\n");
    std::printf("op-dump: records the DeepSeek-V4.1 spec ops of a greedy run of the stock fork.\n\n");
    std::printf("usage: %s -m MODEL --prompt-tokens N --prompt-seed S -n N_PREDICT [--dump-dir DIR] [--ops spec|all]\n",
                argv[0]);
    std::printf("       [llama.cpp options]\n\n");
    std::printf("  --prompt-tokens N  prompt of N token ids drawn uniformly from the vocabulary with --prompt-seed\n");
    std::printf("  --prompt-ids LIST  or a comma separated list of token ids\n");
    std::printf("  --dump-dir DIR     directory to create for the dump (must not exist or be empty); without it the\n");
    std::printf("                     run only generates and prints the token ids and the logits hashes\n");
    std::printf("  --ops spec|all     the SPEC.md op names (default) or every named graph tensor\n");
    std::printf("  -n N_PREDICT       tokens to generate (> 0); all of them are decoded, the last one too\n\n");
    std::printf("Decoding is greedy (argmax of the last logits), sampling options are ignored. One raw little\n");
    std::printf("endian file per (op, occurrence, layer, ubatch), named <op>.<occ>-L<layer>-u<ubatch>.bin (layer 999:\n");
    std::printf("model level), plus dump-manifest.json. Read and compare with tools/dsv41/engine/op_dump.py.\n");
}

static uint64_t fnv1a64(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * 1099511628211ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string        dump_dir;
    std::string        ops_mode = "spec";
    std::string        prompt_ids;
    int                prompt_tokens = 0;
    long               prompt_seed   = -1;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        auto value = [&](const char * opt) {
            if (i + 1 >= argc) {
                fail("%s needs a value", opt);
            }
            return std::string(argv[++i]);
        };
        if (std::strcmp(argv[i], "--dump-dir") == 0) {
            dump_dir = value("--dump-dir");
        } else if (std::strcmp(argv[i], "--ops") == 0) {
            ops_mode = value("--ops");
        } else if (std::strcmp(argv[i], "--prompt-tokens") == 0) {
            prompt_tokens = std::atoi(value("--prompt-tokens").c_str());
        } else if (std::strcmp(argv[i], "--prompt-seed") == 0) {
            prompt_seed = std::atol(value("--prompt-seed").c_str());
        } else if (std::strcmp(argv[i], "--prompt-ids") == 0) {
            prompt_ids = value("--prompt-ids");
        } else {
            rest.push_back(argv[i]);
        }
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }
    if (params.n_predict <= 0) {
        fail("-n N_PREDICT must be > 0, got %d", params.n_predict);
    }
    if (ops_mode != "spec" && ops_mode != "all") {
        fail("--ops %s: expected spec or all", ops_mode.c_str());
    }
    if (prompt_ids.empty() == (prompt_tokens <= 0)) {
        fail("pass exactly one of --prompt-ids LIST and --prompt-tokens N (> 0, with --prompt-seed)");
    }
    if (prompt_tokens > 0 && prompt_seed < 0) {
        fail("--prompt-tokens needs --prompt-seed S (>= 0)");
    }

    dump_state st;
    st.all_ops = ops_mode == "all";
    for (const char * op : SPEC_OPS) {
        st.ops.insert(op);
    }
    if (!dump_dir.empty()) {
        st.dir = dump_dir;
        if (fs::exists(st.dir) && !fs::is_empty(st.dir)) {
            fail("--dump-dir %s exists and is not empty", dump_dir.c_str());
        }
        std::error_code ec;
        fs::create_directories(st.dir, ec);
        if (ec) {
            fail("cannot create --dump-dir %s: %s", dump_dir.c_str(), ec.message().c_str());
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = dump_cb_eval;
    params.cb_eval_user_data = &st;
    // a warmup decode would add ubatches that are not part of the run
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    llama_model *   model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fail("failed to load the model %s or create its context", params.model.path.c_str());
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt;
    if (!prompt_ids.empty()) {
        std::stringstream ss(prompt_ids);
        std::string       item;
        while (std::getline(ss, item, ',')) {
            prompt.push_back(std::atoi(item.c_str()));
        }
    } else {
        // std::mt19937 and uniform_int_distribution as the fork's tests/test-llama-archs.cpp get_tokens()
        std::mt19937                    gen((uint32_t) prompt_seed);
        std::uniform_int_distribution<> dis(0, n_vocab - 1);
        for (int i = 0; i < prompt_tokens; ++i) {
            prompt.push_back(dis(gen));
        }
    }
    for (const llama_token tok : prompt) {
        if (tok < 0 || tok >= n_vocab) {
            fail("prompt token id %d is outside the vocabulary [0, %d)", tok, n_vocab);
        }
    }
    const int n_prompt = (int) prompt.size();
    const int n_ctx    = (int) llama_n_ctx(ctx);
    if (n_prompt + params.n_predict > n_ctx) {
        fail("n_prompt %d + n_predict %d exceeds the context size %d (-c)", n_prompt, params.n_predict, n_ctx);
    }
    if (n_prompt > (int) llama_n_batch(ctx)) {
        fail("n_prompt %d exceeds the batch size %u (-b): the prompt must fit one llama_decode call", n_prompt,
             llama_n_batch(ctx));
    }

    // decode call 0 is the prompt, call k the k-th generated token; the last generated token is decoded too
    std::vector<llama_token> generated;
    std::vector<llama_token> input = prompt;
    json                     logits_hashes = json::array();
    uint64_t                 all_logits    = 1469598103934665603ull;
    for (int k = 0; k <= params.n_predict; ++k) {
        if (k > 0) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            int best = 0;
            for (int i = 1; i < n_vocab; ++i) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            char hex[17];
            const uint64_t h = fnv1a64(logits, (size_t) n_vocab * sizeof(float));
            all_logits = fnv1a64(logits, (size_t) n_vocab * sizeof(float), all_logits);
            std::snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) h);
            logits_hashes.push_back(hex);
            generated.push_back(best);
            input = { best };
        }
        st.decode_call = k;
        st.decode_open = true;
        if (llama_decode(ctx, llama_batch_get_one(input.data(), (int32_t) input.size())) != 0) {
            fail("llama_decode failed on decode call %d (%zu tokens)", k, input.size());
        }
    }

    if (!st.all_ops) {
        // every spec op must have been seen: a missing one means the graph changed under the spec
        std::vector<std::string> missing;
        for (const char * op : SPEC_OPS) {
            if (st.files_per_op.count(op) == 0) {
                missing.push_back(op);
            }
        }
        if (!missing.empty()) {
            std::string list;
            for (const auto & m : missing) {
                list += " " + m;
            }
            std::fprintf(stderr, "op-dump: spec ops never computed by this model:%s\n", list.c_str());
        }
    }

    char all_hex[17];
    std::snprintf(all_hex, sizeof(all_hex), "%016llx", (unsigned long long) all_logits);
    std::printf("generated_token_ids:");
    for (const llama_token tok : generated) {
        std::printf(" %d", tok);
    }
    std::printf("\nlogits: fnv1a64 %s over %zu steps x %d floats\n", all_hex, generated.size(), n_vocab);

    if (!st.dir.empty()) {
        json cmdline = json::array();
        for (int i = 0; i < argc; ++i) {
            cmdline.push_back(argv[i]);
        }
        json per_op = json::object();
        for (const auto & [op, n] : st.files_per_op) {
            per_op[op] = n;
        }
        const json manifest = {
            { "tool", "op-dump" },
            { "fork_commit", llama_commit() },
            { "fork_build_number", llama_build_number() },
            { "argv", cmdline },
            { "model_path", params.model.path },
            { "ops", ops_mode },
            { "prompt_token_ids", prompt },
            { "n_prompt", n_prompt },
            { "n_predict", params.n_predict },
            { "sampling", "greedy argmax of the last logits" },
            { "n_ctx", n_ctx },
            { "n_batch", (int) llama_n_batch(ctx) },
            { "n_ubatch", (int) llama_n_ubatch(ctx) },
            { "n_threads", params.cpuparams.n_threads },
            { "n_embd", llama_model_n_embd(model) },
            { "n_layer", llama_model_n_layer(model) },
            { "n_vocab", n_vocab },
            { "generated_token_ids", generated },
            { "logits_fnv1a64", logits_hashes },
            { "logits_fnv1a64_all", all_hex },
            { "files_per_op", per_op },
            { "ubatches", st.ubatches },
            { "files", st.files },
        };
        const fs::path mpath = st.dir / "dump-manifest.json";
        std::ofstream mf(mpath);
        mf << manifest.dump(1) << "\n";
        if (!mf) {
            fail("cannot write %s", mpath.c_str());
        }
        std::printf("dump: %s (%zu ubatches, %zu files, %zu op names)\n", dump_dir.c_str(), st.ubatches.size(),
                    st.files.size(), st.files_per_op.size());
    }

    llama_backend_free();
    return 0;
}
