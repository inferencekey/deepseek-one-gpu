// dsv41-moe-replay: feeds every (layer, ubatch) of a dsv41-moe-dump directory to libstrata-moe's strata_moe_forward,
// with the dump's own MoE input, selected experts and routing weights, and compares the result with the dump's
// ffn_moe_out. Each layer is measured on the stock inputs, so the error of one layer does not cascade into the
// next. The expert tensors come straight from the model's GGUF
// (mmap, all shards), so the dense part of the model is never loaded.
//
// The op runs as the fork runs it: one ggml_custom_4d on ggml-cpu's threads, loaded with dlopen from
// libstrata-moe.so next to this executable. The library's own environment applies: without STRATA_MOE_HOT it is
// the CPU reference, with STRATA_MOE_HOT and STRATA_MOE_VRAM_GB on a CUDA device the GPU tier computes the hits.
// Prints per layer the relative L2 of out over all ubatches and the worst ubatch, and the hit/miss split the
// library reports (strata_moe_layer_counts).
//
// It only reads ggml, gguf and the dump, so it links libggml-base and libggml-cpu of the fork build.

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "strata/dsv41/moe.h"

#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

[[noreturn]] static void fail(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "dsv41-moe-replay: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::exit(1);
}

typedef int (*layer_counts_t)(int layer, int64_t * hits, int64_t * misses);

// one GGUF shard: its metadata and the whole file mapped read-only
struct shard {
    std::string    path;
    gguf_context * meta = nullptr;
    const char *   base = nullptr;
};

static shard open_shard(const std::string & path) {
    shard s;
    s.path = path;
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    s.meta = gguf_init_from_file(path.c_str(), params);
    if (s.meta == nullptr) {
        fail("cannot read the GGUF header of %s", path.c_str());
    }
    const int fd = open(path.c_str(), O_RDONLY);
    struct stat st = {};
    if (fd < 0 || fstat(fd, &st) != 0) {
        fail("cannot open %s: %s", path.c_str(), std::strerror(errno));
    }
    void * p = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        fail("cannot mmap %s (%lld bytes): %s", path.c_str(), (long long) st.st_size, std::strerror(errno));
    }
    close(fd);
    s.base = static_cast<const char *>(p);
    return s;
}

// The model's GGUF shards (<prefix>-00001-of-0000N.gguf names the rest), the routed expert tensors and the
// per-layer SwiGLU clamp limit, read as the fork reads them.
struct model {
    std::vector<shard> shards;
    std::string        arch;
    std::vector<float> clamp;   // per layer, or one value for all; empty: no <arch>.swiglu_clamp_exp key
    ggml_context *     ctx = nullptr;

    explicit model(const std::string & path) {
        shards.push_back(open_shard(path));
        const gguf_context * m = shards[0].meta;
        const int64_t kid_arch = gguf_find_key(m, "general.architecture");
        if (kid_arch < 0) {
            fail("%s has no general.architecture", path.c_str());
        }
        arch = gguf_get_val_str(m, kid_arch);

        const int64_t kid_split = gguf_find_key(m, "split.count");
        const int n_split = kid_split < 0 ? 1 : (int) gguf_get_val_u16(m, kid_split);
        if (n_split > 1) {
            char suffix[64];
            std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", 1, n_split);
            if (path.size() <= std::strlen(suffix) || path.compare(path.size() - std::strlen(suffix), std::string::npos, suffix) != 0) {
                fail("%s has split.count %d but is not named <prefix>%s", path.c_str(), n_split, suffix);
            }
            const std::string prefix = path.substr(0, path.size() - std::strlen(suffix));
            for (int i = 2; i <= n_split; ++i) {
                std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, n_split);
                shards.push_back(open_shard(prefix + suffix));
            }
        }

        // the fork's hparams.swiglu_clamp_exp: an f32 array with one value per layer, or one f32 for all
        const std::string key = arch + ".swiglu_clamp_exp";
        const int64_t kid = gguf_find_key(m, key.c_str());
        if (kid >= 0) {
            if (gguf_get_kv_type(m, kid) == GGUF_TYPE_ARRAY) {
                if (gguf_get_arr_type(m, kid) != GGUF_TYPE_FLOAT32) {
                    fail("%s is an array of %s, expected f32", key.c_str(), gguf_type_name(gguf_get_arr_type(m, kid)));
                }
                const float * v = static_cast<const float *>(gguf_get_arr_data(m, kid));
                clamp.assign(v, v + gguf_get_arr_n(m, kid));
            } else if (gguf_get_kv_type(m, kid) == GGUF_TYPE_FLOAT32) {
                clamp.assign(1, gguf_get_val_f32(m, kid));
            } else {
                fail("%s has type %s, expected f32 or an f32 array", key.c_str(), gguf_type_name(gguf_get_kv_type(m, kid)));
            }
        }

        ggml_init_params params = { /*.mem_size =*/ 4096 * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ctx = ggml_init(params);
    }

    // the layer's limit as the fork's build_moe_ffn passes it: > 1e-6 is the DeepSeek SwiGLU clamp, else 0 (plain
    // SwiGLU)
    float limit(int layer) const {
        if (clamp.empty()) {
            return 0.0f;
        }
        const size_t i = clamp.size() == 1 ? 0 : (size_t) layer;
        if (i >= clamp.size()) {
            fail("%s.swiglu_clamp_exp has %zu values, no value for layer %d", arch.c_str(), clamp.size(), layer);
        }
        if (clamp[i] <= 1e-6f) {
            return 0.0f;
        }
        if (arch != "deepseek4" && arch != "deepseek41") {
            fail("layer %d has swiglu_clamp_exp %g but architecture %s: the replay knows the clamp of deepseek4 and deepseek41 only",
                layer, clamp[i], arch.c_str());
        }
        return clamp[i];
    }

    // blk.<layer>.ffn_<kind>_exps.weight, its data in the shard's mapping (host memory, ggml's layout)
    ggml_tensor * expert(int layer, const char * kind) {
        char name[GGML_MAX_NAME];
        std::snprintf(name, sizeof(name), "blk.%d.ffn_%s_exps.weight", layer, kind);
        for (const shard & s : shards) {
            const int64_t id = gguf_find_tensor(s.meta, name);
            if (id < 0) {
                continue;
            }
            const int64_t * ne = gguf_get_tensor_ne(s.meta, id);
            ggml_tensor * t = ggml_new_tensor(ctx, gguf_get_tensor_type(s.meta, id), 3, ne);
            if (ne[3] != 1 || ggml_nbytes(t) != gguf_get_tensor_size(s.meta, id)) {
                fail("%s in %s: shape [%lld, %lld, %lld, %lld], expected 3D expert weights", name, s.path.c_str(),
                    (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
            }
            ggml_set_name(t, name);
            t->data = const_cast<char *>(s.base + gguf_get_data_offset(s.meta) + gguf_get_tensor_offset(s.meta, id));
            return t;
        }
        fail("tensor %s is in none of the %zu shards", name, shards.size());
    }
};

// one dump file into memory, checked against its manifest entry
template <typename T>
static std::vector<T> load(const fs::path & dir, const json & entry, const char * dtype) {
    if (entry.at("dtype").get<std::string>() != dtype) {
        fail("%s has dtype %s, expected %s", entry.at("file").get<std::string>().c_str(),
            entry.at("dtype").get<std::string>().c_str(), dtype);
    }
    int64_t n = 1;
    for (const json & d : entry.at("ne")) {
        n *= d.get<int64_t>();
    }
    const fs::path path = dir / entry.at("file").get<std::string>();
    std::vector<T> v((size_t) n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(v.data()), (std::streamsize) (n * sizeof(T)));
    if (!f || f.peek() != EOF) {
        fail("%s: expected exactly %lld bytes", path.c_str(), (long long) (n * sizeof(T)));
    }
    return v;
}

static void print_usage(const char * argv0) {
    std::printf("usage: %s -m MODEL --dump-dir DIR [-t N_THREADS] [--max-rel-l2 X]\n\n", argv0);
    std::printf("  -m MODEL          the GGUF the dump was made from (the first shard of a split model)\n");
    std::printf("  --dump-dir DIR    a dsv41-moe-dump directory\n");
    std::printf("  -t N_THREADS      ggml-cpu threads for the op (default 8)\n");
    std::printf("  --max-rel-l2 X    exit 1 when a layer's relative L2 of out is above X (default 2e-2)\n\n");
    std::printf("Feeds each (layer, ubatch) of the dump to strata_moe_forward from libstrata-moe.so (next to this\n");
    std::printf("executable) and prints per layer the relative L2 ||strata - dump|| / ||dump|| of ffn_moe_out over\n");
    std::printf("all ubatches, the worst ubatch, and the hits (GPU tier) and misses (CPU) the library counted.\n");
    std::printf("STRATA_MOE_HOT, STRATA_MOE_VRAM_GB and STRATA_MOE_PROFILE apply as in the fork (include/strata/dsv41/moe.h).\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path, dump_dir;
    int    n_threads = 8;
    double max_rel   = 2e-2;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> const char * {
            if (i + 1 >= argc) {
                fail("%s needs a value", a.c_str());
            }
            return argv[++i];
        };
        if (a == "-m") {
            model_path = value();
        } else if (a == "--dump-dir") {
            dump_dir = value();
        } else if (a == "-t") {
            n_threads = std::atoi(value());
        } else if (a == "--max-rel-l2") {
            max_rel = std::atof(value());
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fail("unknown argument %s (see --help)", a.c_str());
        }
    }
    if (model_path.empty() || dump_dir.empty()) {
        fail("-m MODEL and --dump-dir DIR are required (see --help)");
    }
    if (n_threads < 1) {
        fail("-t must be >= 1");
    }

    // the op as the fork loads it: dlopen of libstrata-moe.so, found through this executable's RUNPATH $ORIGIN
    void * lib = dlopen("libstrata-moe.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        fail("%s", dlerror());
    }
    auto * forward = (ggml_custom_op_t) dlsym(lib, "strata_moe_forward");
    auto * counts  = (layer_counts_t) dlsym(lib, "strata_moe_layer_counts");
    if (forward == nullptr || counts == nullptr) {
        fail("libstrata-moe.so: %s", dlerror());
    }

    const fs::path dir = dump_dir;
    json manifest;
    {
        std::ifstream f(dir / "dump-manifest.json");
        if (!f) {
            fail("%s has no dump-manifest.json", dump_dir.c_str());
        }
        manifest = json::parse(f);
    }
    std::map<std::tuple<std::string, int, int>, json> files;
    for (const json & e : manifest.at("files")) {
        files[{ e.at("kind").get<std::string>(), e.at("layer").get<int>(), e.at("ubatch").get<int>() }] = e;
    }
    auto entry = [&](const char * kind, int layer, int ubatch) -> const json & {
        const auto it = files.find({ kind, layer, ubatch });
        if (it == files.end()) {
            fail("%s has no %s file for layer %d ubatch %d", dump_dir.c_str(), kind, layer, ubatch);
        }
        return it->second;
    };
    const int64_t n_embd = manifest.at("n_embd").get<int64_t>();

    ggml_cpu_init();
    model m(model_path);
    const char * hot = std::getenv("STRATA_MOE_HOT");
    std::printf("model:    %s (%s, %zu shards)\n", model_path.c_str(), m.arch.c_str(), m.shards.size());
    std::printf("dump:     %s (%zu ubatches, %zu MoE layers)\n", dump_dir.c_str(), manifest.at("ubatches").size(),
        manifest.at("moe_layers").size());
    std::printf("threads:  %d, STRATA_MOE_HOT=%s\n\n", n_threads, hot ? hot : "(unset)");
    std::printf("%5s  %8s  %6s  %10s  %10s  %9s  %9s  %6s  %9s\n", "layer", "ubatches", "tokens", "relL2 out", "max/ubatch",
        "hits", "misses", "hit%", "ms");

    double worst = 0.0;
    int64_t all_hits = 0, all_misses = 0;
    for (const json & ml : manifest.at("moe_layers")) {
        const int layer = ml.at("layer").get<int>();
        ggml_tensor * up   = m.expert(layer, "up");
        ggml_tensor * gate = m.expert(layer, "gate");
        ggml_tensor * down = m.expert(layer, "down");
        const float limit = m.limit(layer);

        double diff2 = 0.0, ref2 = 0.0, max_ub = 0.0;
        int64_t tokens = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (const json & ub : manifest.at("ubatches")) {
            const int u = ub.at("ubatch").get<int>();
            const json & e_cur = entry("cur", layer, u);
            const json & e_topk = entry("topk", layer, u);
            std::vector<float>   cur  = load<float>(dir, e_cur, "f32");
            std::vector<int32_t> topk = load<int32_t>(dir, e_topk, "i32");
            std::vector<float>   w    = load<float>(dir, entry("weights", layer, u), "f32");
            std::vector<float>   ref  = load<float>(dir, entry("out", layer, u), "f32");
            const int64_t n_tok  = e_cur.at("ne")[1].get<int64_t>();
            const int64_t n_used = e_topk.at("ne")[0].get<int64_t>();
            if (e_cur.at("ne")[0].get<int64_t>() != n_embd || (int64_t) ref.size() != n_embd * n_tok ||
                (int64_t) topk.size() != n_used * n_tok || (int64_t) w.size() != n_used * n_tok) {
                fail("layer %d ubatch %d: cur, topk, weights and out shapes do not agree", layer, u);
            }
            std::vector<float> out((size_t) (n_embd * n_tok));

            ggml_init_params params = { /*.mem_size =*/ 8 * ggml_tensor_overhead() + ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * t_cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tok);
            ggml_tensor * t_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
            ggml_tensor * t_w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n_tok);
            t_cur->data = cur.data();
            t_ids->data = topk.data();
            t_w->data   = w.data();
            ggml_tensor * args[] = { t_cur, t_ids, t_w, up, gate, down };
            ggml_tensor * t_out = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_tok, 1, 1, args, 6, forward, GGML_N_TASKS_MAX, nullptr);
            strata_moe_set_clamp(t_out, limit);
            t_out->data = out.data();
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, t_out);
            ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
            std::vector<uint8_t> work(plan.work_size);
            plan.work_data = work.data();
            if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
                fail("layer %d ubatch %d: graph compute failed", layer, u);
            }
            ggml_free(ctx);

            double d2 = 0.0, r2 = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double d = (double) out[i] - (double) ref[i];
                d2 += d * d;
                r2 += (double) ref[i] * (double) ref[i];
            }
            diff2 += d2;
            ref2  += r2;
            max_ub = std::max(max_ub, r2 > 0.0 ? std::sqrt(d2 / r2) : (d2 > 0.0 ? INFINITY : 0.0));
            tokens += n_tok;
        }
        const double ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double rel = ref2 > 0.0 ? std::sqrt(diff2 / ref2) : (diff2 > 0.0 ? INFINITY : 0.0);
        worst = std::max(worst, rel);
        int64_t hits = 0, misses = 0;
        if (!counts(layer, &hits, &misses)) {
            fail("libstrata-moe.so has no counts for layer %d after its calls", layer);
        }
        all_hits   += hits;
        all_misses += misses;
        std::printf("%5d  %8zu  %6lld  %10.3e  %10.3e  %9lld  %9lld  %6.1f  %9.1f\n", layer, manifest.at("ubatches").size(),
            (long long) tokens, rel, max_ub, (long long) hits, (long long) misses,
            hits + misses > 0 ? 100.0 * (double) hits / (double) (hits + misses) : 0.0, ms);
        std::fflush(stdout);
    }

    const bool pass = worst <= max_rel;
    std::printf("\nmax relative L2 of out over layers: %.3e (bound %.1e): %s; hits %lld, misses %lld, hit rate %.1f%%\n",
        worst, max_rel, pass ? "PASS" : "FAIL", (long long) all_hits, (long long) all_misses,
        all_hits + all_misses > 0 ? 100.0 * (double) all_hits / (double) (all_hits + all_misses) : 0.0);
    return pass ? 0 : 1;
}
