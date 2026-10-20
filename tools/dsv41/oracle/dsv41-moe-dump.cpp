// dsv41-moe-dump: runs a prompt with greedy decoding through the stock llama.cpp fork and records, for every MoE
// layer and every evaluated ubatch, the tensors the Strata build_moe_ffn hook receives and must reproduce:
// the MoE input, the selected experts, the routing weights fed to the experts and the combined routed-expert
// output. See tools/dsv41/oracle/README.md for the dump layout and tools/dsv41/oracle/read_dump.py to read it.
//
// It only reads the graph through the eval callback, so the fork stays unmodified.

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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::ordered_json;

[[noreturn]] static void fail(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "dsv41-moe-dump: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::exit(1);
}

enum dump_kind { KIND_CUR, KIND_TOPK, KIND_WEIGHTS, KIND_OUT, KIND_COUNT };

static const char * const KIND_NAMES[KIND_COUNT] = { "cur", "topk", "weights", "out" };

// The routing weights build_moe_ffn feeds to the experts are the last of these the graph produces for the model,
// in this graph order (src/llama-graph.cpp build_moe_ffn).
static const char * const WEIGHT_STAGES[] = {
    "ffn_moe_weights", "ffn_moe_weights_softmax", "ffn_moe_weights_norm", "ffn_moe_weights_scaled",
};
static constexpr int N_WEIGHT_STAGES = sizeof(WEIGHT_STAGES) / sizeof(WEIGHT_STAGES[0]);

struct host_tensor {
    std::string              source;   // name of the graph tensor the data came from
    ggml_type                type;
    std::array<int64_t, 4>   ne;
    std::array<int64_t, 4>   source_ne;
    std::vector<uint8_t>     data;     // packed, ne[0] fastest
};

struct pending_weights {
    int         stage;
    host_tensor tensor;
};

struct dump_state {
    fs::path dir;
    int      decode_call = 0; // 0: the prompt, k > 0: the k-th generated token as input

    int     first_moe_layer = -1;
    int64_t n_pos           = 0;   // token positions of the ubatches seen so far

    std::map<int, std::array<int, KIND_COUNT>>     counts;        // per MoE layer, files written per kind
    std::map<int, std::string>                     weights_stage; // per MoE layer, the weights name used
    std::map<std::pair<int, int>, pending_weights> pending;       // (layer, ubatch) -> latest weights stage seen

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

// Copies t from whichever backend holds it into a packed host buffer, following its strides: ffn_moe_topk is a
// view of the argsort result with a row stride of n_expert, not n_expert_used.
static host_tensor copy_to_host(const ggml_tensor * t) {
    if (type_dtype(t->type) == nullptr) {
        fail("tensor %s has type %s, the dump supports f32, f16 and i32", t->name, ggml_type_name(t->type));
    }
    if (t->buffer == nullptr) {
        fail("tensor %s has no backend buffer", t->name);
    }

    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

    host_tensor h;
    h.source = t->name;
    h.type   = t->type;
    for (int i = 0; i < 4; ++i) {
        h.ne[i]        = t->ne[i];
        h.source_ne[i] = t->ne[i];
    }

    const size_t esize = ggml_type_size(t->type);
    if (ggml_is_contiguous(t)) {
        h.data = std::move(raw);
        return h;
    }
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

static void write_file(dump_state & st, dump_kind kind, int layer, int ubatch, const host_tensor & h) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s-L%03d-u%04d.bin", KIND_NAMES[kind], layer, ubatch);
    const fs::path path = st.dir / name;
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(h.data.data()), (std::streamsize) h.data.size());
    if (!f) {
        fail("cannot write %s (%zu bytes)", path.c_str(), h.data.size());
    }

    json entry = {
        { "file", name },
        { "kind", KIND_NAMES[kind] },
        { "layer", layer },
        { "ubatch", ubatch },
        { "dtype", type_dtype(h.type) },
        { "ne", h.ne },
        { "source_tensor", h.source },
    };
    if (h.source_ne != h.ne) {
        entry["source_ne"] = h.source_ne;
    }
    st.files.push_back(std::move(entry));
    st.counts[layer][kind]++;
}

// The last weights stage of each (layer, ubatch) is known only once the layer is done, so weights wait here
// until the end of each llama_decode call.
static void flush_weights(dump_state & st) {
    for (auto & [key, p] : st.pending) {
        const int    layer = key.first;
        const char * stage = WEIGHT_STAGES[p.stage];
        auto it = st.weights_stage.find(layer);
        if (it == st.weights_stage.end()) {
            st.weights_stage[layer] = stage;
        } else if (it->second != stage) {
            fail("layer %d: routing weights come from %s in ubatch %d but from %s before", layer, stage, key.second,
                it->second.c_str());
        }
        write_file(st, KIND_WEIGHTS, layer, key.second, p.tensor);
    }
    st.pending.clear();
}

// "<base>-<layer>" as named by llama_context::graph_get_cb; false for other names
static bool split_name(const char * name, std::string & base, int & layer) {
    const char * dash = std::strrchr(name, '-');
    if (dash == nullptr || dash[1] == '\0') {
        return false;
    }
    char * end = nullptr;
    const long il = std::strtol(dash + 1, &end, 10);
    if (*end != '\0' || il < 0) {
        return false;
    }
    base.assign(name, dash - name);
    layer = (int) il;
    return true;
}

static int weight_stage(const std::string & base) {
    for (int i = 0; i < N_WEIGHT_STAGES; ++i) {
        if (base == WEIGHT_STAGES[i]) {
            return i;
        }
    }
    return -1;
}

static bool wanted(const std::string & base) {
    return base == "ffn_moe_logits" || base == "ffn_moe_topk" || base == "ffn_moe_out" || weight_stage(base) >= 0;
}

static bool dump_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto & st = *static_cast<dump_state *>(user_data);
    std::string base;
    int         layer = -1;
    if (!split_name(t->name, base, layer) || !wanted(base)) {
        return false;
    }
    if (ask) {
        return true;
    }

    auto & counts = st.counts[layer];
    if (base == "ffn_moe_logits") {
        // the MoE input is the activation operand of the router mul_mat
        if (t->op != GGML_OP_MUL_MAT || t->src[1] == nullptr) {
            fail("%s is op %s, expected MUL_MAT with the MoE input as src[1]", t->name, ggml_op_name(t->op));
        }
        const ggml_tensor * cur = t->src[1];
        if (cur->ne[2] != 1 || cur->ne[3] != 1) {
            fail("MoE input %s (src[1] of %s) has ne [%lld, %lld, %lld, %lld], expected [n_embd, n_tokens, 1, 1]",
                cur->name, t->name, (long long) cur->ne[0], (long long) cur->ne[1], (long long) cur->ne[2],
                (long long) cur->ne[3]);
        }
        if (st.first_moe_layer < 0) {
            st.first_moe_layer = layer;
        }
        const int ubatch = counts[KIND_CUR];
        if (layer == st.first_moe_layer) {
            st.ubatches.push_back({
                { "ubatch", ubatch },
                { "decode_call", st.decode_call },
                { "pos0", st.n_pos },
                { "n_tokens", cur->ne[1] },
            });
            st.n_pos += cur->ne[1];
        }
        write_file(st, KIND_CUR, layer, ubatch, copy_to_host(cur));
        return true;
    }

    // every other kind belongs to the ubatch whose MoE input was recorded last for this layer
    const int ubatch = counts[KIND_CUR] - 1;
    if (ubatch < 0) {
        fail("%s was computed before ffn_moe_logits-%d, cannot assign it a ubatch", t->name, layer);
    }
    if (base == "ffn_moe_topk") {
        write_file(st, KIND_TOPK, layer, ubatch, copy_to_host(t));
    } else if (base == "ffn_moe_out") {
        write_file(st, KIND_OUT, layer, ubatch, copy_to_host(t));
    } else {
        const int stage = weight_stage(base);
        auto it = st.pending.find({ layer, ubatch });
        if (it == st.pending.end() || it->second.stage < stage) {
            // ffn_moe_weights_norm is 2D [n_expert_used, n_tokens]; the experts get it reshaped to
            // [1, n_expert_used, n_tokens] like the other stages, with the same bytes
            const bool is_2d = base == "ffn_moe_weights_norm";
            const bool shape_ok = is_2d ? (t->ne[2] == 1 && t->ne[3] == 1) : (t->ne[0] == 1 && t->ne[3] == 1);
            if (!shape_ok) {
                fail("weights %s has ne [%lld, %lld, %lld, %lld], expected %s", t->name, (long long) t->ne[0],
                    (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    is_2d ? "[n_expert_used, n_tokens, 1, 1]" : "[1, n_expert_used, n_tokens, 1]");
            }
            host_tensor h = copy_to_host(t);
            if (is_2d) {
                h.ne = { 1, t->ne[0], t->ne[1], 1 };
            }
            st.pending[{ layer, ubatch }] = { stage, std::move(h) };
        }
    }
    return true;
}

// every MoE layer must have all four kinds for every ubatch
static void check_complete(const dump_state & st) {
    if (st.counts.empty()) {
        fail("no ffn_moe_logits-<layer> tensor was computed: the model has no build_moe_ffn layer");
    }
    const int n_ubatch = (int) st.ubatches.size();
    for (const auto & [layer, counts] : st.counts) {
        for (int k = 0; k < KIND_COUNT; ++k) {
            if (counts[k] != n_ubatch) {
                fail("layer %d has %d %s files, expected one per ubatch (%d)", layer, counts[k], KIND_NAMES[k],
                    n_ubatch);
            }
        }
    }
}

static void print_usage(int, char ** argv) {
    std::printf("\n");
    std::printf("dsv41-moe-dump: records the per-layer MoE tensors of a greedy run of the stock fork.\n\n");
    std::printf("usage: %s --dump-dir DIR -m MODEL (-p PROMPT | -f FILE) -n N_PREDICT [llama.cpp options]\n\n",
                argv[0]);
    std::printf("  --dump-dir DIR   directory to create for the dump; it must not exist or be empty\n");
    std::printf("  -n N_PREDICT     tokens to generate, required (> 0); all of them are decoded and dumped\n\n");
    std::printf("Decoding is greedy (argmax of the last logits), end-of-generation tokens do not stop it, the\n");
    std::printf("sampling options are ignored. The other options are the common llama.cpp ones listed above.\n");
    std::printf("Writes one raw little-endian file per (kind, layer, ubatch): cur (MoE input, src[1] of\n");
    std::printf("ffn_moe_logits), topk (ffn_moe_topk), weights (last ffn_moe_weights* stage), out (ffn_moe_out),\n");
    std::printf("plus dump-manifest.json. Read it with read_dump.py.\n");
}

static llama_token argmax_token(llama_context * ctx, const llama_vocab * vocab) {
    const float * logits  = llama_get_logits_ith(ctx, -1);
    const int     n_vocab = llama_vocab_n_tokens(vocab);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // --dump-dir is ours; the rest goes to the common llama.cpp parser
    std::string        dump_dir;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-dir") == 0) {
            if (i + 1 >= argc) {
                fail("--dump-dir needs a directory");
            }
            dump_dir = argv[++i];
        } else {
            rest.push_back(argv[i]);
        }
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }
    if (dump_dir.empty()) {
        fail("--dump-dir DIR is required (see --help)");
    }
    if (params.n_predict <= 0) {
        fail("-n N_PREDICT must be > 0, got %d", params.n_predict);
    }
    if (params.prompt.empty()) {
        fail("no prompt: pass -p PROMPT or -f FILE");
    }

    dump_state st;
    st.dir = dump_dir;
    std::error_code ec;
    if (fs::exists(st.dir) && !fs::is_empty(st.dir)) {
        fail("--dump-dir %s exists and is not empty", dump_dir.c_str());
    }
    fs::create_directories(st.dir, ec);
    if (ec) {
        fail("cannot create --dump-dir %s: %s", dump_dir.c_str(), ec.message().c_str());
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
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> prompt = common_tokenize(ctx, params.prompt, add_bos, true);
    if (prompt.empty()) {
        fail("the prompt tokenizes to 0 tokens");
    }
    const int n_prompt = (int) prompt.size();
    const int n_ctx    = (int) llama_n_ctx(ctx);
    const int n_batch  = (int) llama_n_batch(ctx);
    if (n_prompt + params.n_predict > n_ctx) {
        fail("n_prompt %d + n_predict %d exceeds the context size %d (-c)", n_prompt, params.n_predict, n_ctx);
    }
    if (n_prompt > n_batch) {
        fail("n_prompt %d exceeds the batch size %d (-b): the prompt must fit one llama_decode call", n_prompt,
            n_batch);
    }
    std::fprintf(stderr, "dsv41-moe-dump: %d prompt tokens, generating %d, dumping into %s\n", n_prompt,
                 params.n_predict, dump_dir.c_str());

    // decode call 0 is the prompt, call k the k-th generated token; the last generated token is decoded too, so
    // the dump covers n_prompt + n_predict positions
    std::vector<llama_token> generated;
    std::vector<llama_token> input = prompt;
    for (int k = 0; k <= params.n_predict; ++k) {
        if (k > 0) {
            generated.push_back(argmax_token(ctx, vocab));
            input = { generated.back() };
        }
        st.decode_call = k;
        if (llama_decode(ctx, llama_batch_get_one(input.data(), (int32_t) input.size())) != 0) {
            fail("llama_decode failed on decode call %d (%zu tokens)", k, input.size());
        }
        flush_weights(st);
    }
    check_complete(st);

    std::string text;
    for (const llama_token tok : generated) {
        text += common_token_to_piece(ctx, tok);
    }

    json layers = json::array();
    for (const auto & [layer, stage] : st.weights_stage) {
        layers.push_back({ { "layer", layer }, { "weights_tensor", stage } });
    }
    json cmdline = json::array();
    for (int i = 0; i < argc; ++i) {
        cmdline.push_back(argv[i]);
    }
    const json manifest = {
        { "tool", "dsv41-moe-dump" },
        { "fork_commit", llama_commit() },
        { "fork_build_number", llama_build_number() },
        { "argv", cmdline },
        { "model_path", params.model.path },
        { "prompt", params.prompt },
        { "add_bos", add_bos },
        { "prompt_token_ids", prompt },
        { "n_prompt", n_prompt },
        { "n_predict", params.n_predict },
        { "seed", params.sampling.seed },
        { "sampling", "greedy argmax of the last logits, end-of-generation does not stop" },
        { "n_ctx", n_ctx },
        { "n_batch", n_batch },
        { "n_ubatch", (int) llama_n_ubatch(ctx) },
        { "n_embd", llama_model_n_embd(model) },
        { "n_layer", llama_model_n_layer(model) },
        { "generated_token_ids", generated },
        { "generated_text", text },
        { "moe_layers", layers },
        { "ubatches", st.ubatches },
        { "files", st.files },
    };
    const fs::path mpath = st.dir / "dump-manifest.json";
    std::ofstream mf(mpath);
    mf << manifest.dump(1) << "\n";
    if (!mf) {
        fail("cannot write %s", mpath.c_str());
    }

    std::printf("generated_token_ids:");
    for (const llama_token tok : generated) {
        std::printf(" %d", tok);
    }
    std::printf("\ndump: %s (%zu MoE layers, %zu ubatches, %zu files)\n", dump_dir.c_str(), st.counts.size(),
                st.ubatches.size(), st.files.size());

    llama_backend_free();
    return 0;
}
