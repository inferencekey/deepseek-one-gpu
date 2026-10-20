// dsv41-moe-clamp-check: runs a model through the fork's real build_moe_ffn hook path on the CPU and checks that
// the SwiGLU clamp limit every routed-expert op computed with is the GGUF's own value for its layer, in every
// call, including the calls of graphs llama.cpp reuses after the graph-build objects are gone. Made for the tiny
// random DeepSeek-V4 of tools/dsv41/oracle/hook-clamp-test.sh, which runs it with the hook and without.
//
// It decodes a fixed prompt, then N single tokens greedily (same shape, so llama.cpp reuses the graph), each
// llama_decode from a different stack depth: memory the graph build left on the stack is overwritten between
// calls, as in llama-server, where warmup and serving decode from different depths. Always prints a hash of every
// logit of every call (the same with and without the hook means bit-identical output). With LLAMA_STRATA_MOE=1 it
// also prints, per layer, the GGUF limit and the smallest and largest limit libstrata-moe computed with
// (strata_moe_layer_clamp), and exits 1 unless they are equal in every layer and every layer ran every call.
//
// CPU only (no layer offloaded, no repacking: the hook reads the experts in ggml's layout from host memory).

#include "gguf.h"
#include "llama.h"

#include <dlfcn.h>

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

[[noreturn]] static void fail(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "dsv41-moe-clamp-check: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::exit(1);
}

typedef int (*layer_clamp_t)(int layer, float * min_limit, float * max_limit, int64_t * calls);

// <arch>.swiglu_clamp_exp per layer, as the fork reads it (an f32 array of block_count values, or one f32)
static std::vector<float> gguf_limits(const char * path) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, params);
    if (g == nullptr) {
        fail("%s: cannot read the GGUF", path);
    }
    const int64_t kid_arch = gguf_find_key(g, "general.architecture");
    if (kid_arch < 0) {
        fail("%s has no general.architecture", path);
    }
    const std::string arch = gguf_get_val_str(g, kid_arch);
    const int64_t kid_n = gguf_find_key(g, (arch + ".block_count").c_str());
    const int64_t kid = gguf_find_key(g, (arch + ".swiglu_clamp_exp").c_str());
    if (kid_n < 0 || kid < 0) {
        fail("%s has no %s.block_count or %s.swiglu_clamp_exp", path, arch.c_str(), arch.c_str());
    }
    const uint32_t n_layer = gguf_get_val_u32(g, kid_n);
    std::vector<float> limits;
    if (gguf_get_kv_type(g, kid) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, kid) == GGUF_TYPE_FLOAT32 &&
        gguf_get_arr_n(g, kid) == n_layer) {
        const float * v = static_cast<const float *>(gguf_get_arr_data(g, kid));
        limits.assign(v, v + n_layer);
    } else if (gguf_get_kv_type(g, kid) == GGUF_TYPE_FLOAT32) {
        limits.assign(n_layer, gguf_get_val_f32(g, kid));
    } else {
        fail("%s.swiglu_clamp_exp is not an f32 or an array of %u f32", arch.c_str(), n_layer);
    }
    gguf_free(g);
    return limits;
}

// llama_decode after `depth` nested frames, each with a stack buffer written with `fill`
static int decode_at_depth(llama_context * ctx, llama_batch batch, int depth, unsigned char fill) {
    if (depth == 0) {
        return llama_decode(ctx, batch);
    }
    volatile unsigned char pad[2048];
    for (size_t i = 0; i < sizeof(pad); ++i) {
        pad[i] = fill;
    }
    const int rc = decode_at_depth(ctx, batch, depth - 1, fill);
    return rc + (pad[0] != fill);
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int n_gen = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "-n" && i + 1 < argc) {
            n_gen = std::atoi(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            std::printf("usage: %s -m MODEL [-n N_GEN]\n\n", argv[0]);
            std::printf("Decodes a fixed 8-token prompt and N_GEN (default 8) greedy tokens on the CPU, each call from a\n");
            std::printf("different stack depth, and prints a hash of all logits. With LLAMA_STRATA_MOE=1, exits 1 unless\n");
            std::printf("every layer's op computed with the GGUF's swiglu_clamp_exp of that layer in every call.\n");
            return 0;
        } else {
            fail("unknown argument '%s' (see --help)", a.c_str());
        }
    }
    if (model_path == nullptr || n_gen < 2) {
        fail("-m MODEL is required and -n must be at least 2 (see --help)");
    }
    const char * env = std::getenv("LLAMA_STRATA_MOE");
    const bool hook = env != nullptr && std::strcmp(env, "1") == 0;
    const std::vector<float> want = gguf_limits(model_path);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers    = 0;
    mparams.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fail("cannot load %s", model_path);
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 128;
    cparams.n_batch         = 64;
    cparams.n_ubatch        = 64;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fail("cannot create a context for %s", model_path);
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // FNV-1a over the bits of every logit of every call
    uint64_t hash = 1469598103934665603ull;
    std::vector<llama_token> tokens = { 1, 5, 9, 17, 33, 65, 3, 7 };
    std::vector<llama_token> generated;
    for (int k = 0; k <= n_gen; ++k) {
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
        if (decode_at_depth(ctx, batch, 1 + 3 * (k % 4), (unsigned char) (0x5a + k)) != 0) {
            fail("llama_decode failed on call %d", k);
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 0; i < n_vocab; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &logits[i], sizeof(bits));
            for (int b = 0; b < 4; ++b) {
                hash = (hash ^ ((bits >> (8 * b)) & 0xff)) * 1099511628211ull;
            }
            best = logits[i] > logits[best] ? i : best;
        }
        generated.push_back(best);
        tokens = { best };
    }
    const int32_t n_reused = llama_perf_context(ctx).n_reused;
    const int n_calls = n_gen + 1;

    std::printf("model:   %s (%zu layers), %d decode calls (an 8-token prompt, %d single tokens), graphs reused %d\n",
        model_path, want.size(), n_calls, n_gen, n_reused);
    std::printf("tokens: ");
    for (const llama_token t : generated) {
        std::printf(" %d", t);
    }
    std::printf("\nlogits:  fnv1a64 %016" PRIx64 " (LLAMA_STRATA_MOE=%s)\n", hash, hook ? "1" : "unset");
    if (n_reused < n_gen - 1) {
        fail("only %d of the %d single-token calls reused the graph: the check needs reused graphs", n_reused, n_gen);
    }
    if (!hook) {
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    // the library the fork loaded (dlopen of the same name returns that handle)
    void * lib = dlopen("libstrata-moe.so", RTLD_NOW | RTLD_NOLOAD);
    if (lib == nullptr) {
        fail("LLAMA_STRATA_MOE=1 but libstrata-moe.so is not loaded: %s", dlerror());
    }
    auto * layer_clamp = (layer_clamp_t) dlsym(lib, "strata_moe_layer_clamp");
    if (layer_clamp == nullptr) {
        fail("%s", dlerror());
    }
    std::printf("\nlayer  GGUF swiglu_clamp_exp  op min        op max        calls\n");
    bool ok = true;
    for (size_t il = 0; il < want.size(); ++il) {
        float lo = 0.0f, hi = 0.0f;
        int64_t calls = 0;
        if (!layer_clamp((int) il, &lo, &hi, &calls)) {
            fail("libstrata-moe.so has no call of layer %zu", il);
        }
        const float expect = want[il] > 1e-6f ? want[il] : 0.0f;
        const bool pass = lo == expect && hi == expect && calls == n_calls;
        ok = ok && pass;
        std::printf("%5zu  %-20.9g  %-12.9g  %-12.9g  %5" PRId64 "  %s\n", il, want[il], lo, hi, calls, pass ? "ok" : "FAIL");
    }
    std::printf("\nclamp limit seen by the op = GGUF value in every layer and call: %s\n", ok ? "PASS" : "FAIL");
    llama_free(ctx);
    llama_model_free(model);
    return ok ? 0 : 1;
}
