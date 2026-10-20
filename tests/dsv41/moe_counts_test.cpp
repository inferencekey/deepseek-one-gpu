// tests/dsv41/moe_counts_test.cpp - strata-moe's activation counts (src/dsv41/moe_counts.h), GPU free. Its
// files go in the working directory (ctest: the build tree).
//
// 1. Format: decode and batch calls of two layers give the exact strata-moe-counts/1 text (the same text
//    tools/dsv41/oracle/hot_set_test.py feeds hot_set.py --counts), write() puts it in the file whole and leaves no
//    <path>.tmp behind.
// 2. Writes: with every = 2 the file is written at the first call and then after every second one, each time with
//    all the counts so far.
// 3. Aborts (in a child process, the abort message printed): an expert id outside the layer, a layer whose expert
//    count changes, a path that cannot be written (at the first call, not at exit).
// 4. Through the op: a child process with STRATA_MOE_COUNTS set runs strata::dsv41::forward (CPU reference, one
//    layer of 16 q4_K/q6_K experts) on 1 and 5 tokens and exits; the file its exit handler wrote equals the counts
//    of the routed experts.
#include "moe_counts.h"
#include "moe_op.h"
#include "strata/dsv41/moe.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using strata::dsv41::ActivationCounts;

int g_failed = 0;

void check(bool ok, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
void check(bool ok, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::printf("%-4s ", ok ? "ok" : "FAIL");
    std::vprintf(fmt, args);
    std::printf("\n");
    va_end(args);
    g_failed += !ok;
}

std::string cwd_path(const char* name) {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        std::printf("FAIL: getcwd: %s\n", std::strerror(errno));
        std::exit(1);
    }
    return std::string(cwd) + "/" + name;
}

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

bool exists(const std::string& path) { return access(path.c_str(), F_OK) == 0; }

// Runs fn in a child process; returns its wait status and puts its stderr in *err.
int run_child(const std::function<void()>& fn, std::string* err) {
    std::fflush(stdout);
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) std::exit(1);
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(pipe_fds[1], STDERR_FILENO);
        close(pipe_fds[0]);
        fn();
        _exit(0);
    }
    close(pipe_fds[1]);
    char buf[4096];
    for (ssize_t n; (n = read(pipe_fds[0], buf, sizeof(buf))) > 0;) err->append(buf, (size_t) n);
    close(pipe_fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return status;
}

// The case passes when the child aborts (SIGABRT, as GGML_ABORT does) with every one of `needles` in its stderr's
// strata-moe line, which is printed under the case.
void check_aborts(const std::function<void()>& fn, const char* what, const std::vector<std::string>& needles) {
    std::string err;
    const int status = run_child(fn, &err);
    std::string line = err.substr(0, err.find('\n'));
    for (size_t b = 0; b < err.size();) {
        const size_t e = err.find('\n', b);
        const std::string l = err.substr(b, e == std::string::npos ? std::string::npos : e - b);
        if (l.find(": strata-moe: ") != std::string::npos) line = l;
        if (e == std::string::npos) break;
        b = e + 1;
    }
    bool has = true;
    for (const std::string& n : needles) has = has && line.find(n) != std::string::npos;
    check(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && has, "%s", what);
    std::printf("       %s\n", line.c_str());
}

// selected_experts [n_used, n_tokens] with the given ids, token-major
ggml_tensor* ids_tensor(ggml_context* ctx, int64_t n_used, const std::vector<int32_t>& ids) {
    ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, (int64_t) ids.size() / n_used);
    std::memcpy(t->data, ids.data(), ids.size() * sizeof(int32_t));
    return t;
}

// ---------------------------------------------------------------- 1, 2, 3: the counter itself

// layer 3 with 4 experts, layer 1 with 6, 2 slots: two decode calls of layer 3, one 3-token batch call of each
const char* const kExpected =
    "{\"format\": \"strata-moe-counts/1\", \"layers\": [\n"
    " {\"layer\": 1, \"n_expert\": 6, \"decode_calls\": 0, \"batch_calls\": 1, \"batch_tokens\": 3, "
    "\"decode\": [0, 0, 0, 0, 0, 0], \"batch\": [1, 0, 2, 0, 0, 3]},\n"
    " {\"layer\": 3, \"n_expert\": 4, \"decode_calls\": 2, \"batch_calls\": 1, \"batch_tokens\": 3, "
    "\"decode\": [1, 2, 0, 1], \"batch\": [0, 3, 2, 1]}]}\n";

void test_counter() {
    ggml_init_params params = {/*.mem_size =*/ (size_t) 1 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    ggml_context* ctx = ggml_init(params);
    const std::string path = cwd_path("dsv41_moe_counts_test.format.json");
    std::remove(path.c_str());
    {
        ActivationCounts counts(path, 1000);
        counts.record(3, 4, ids_tensor(ctx, 2, {1, 3}));
        counts.record(3, 4, ids_tensor(ctx, 2, {0, 1}));
        counts.record(3, 4, ids_tensor(ctx, 2, {1, 2, 2, 3, 1, 1}));
        counts.record(1, 6, ids_tensor(ctx, 2, {5, 2, 0, 5, 2, 5}));
        const std::string text = counts.json();
        check(text == kExpected, "format: two layers, decode and batch calls give the strata-moe-counts/1 text:\n%s",
              text.c_str());
        counts.write();
        check(slurp(path) == kExpected && !exists(path + ".tmp"), "write(): the file holds that text, no %s left",
              (path + ".tmp").c_str());
    }

    // every = 2: written at calls 1, 3 and 5, each time with every count so far
    const std::string every_path = cwd_path("dsv41_moe_counts_test.every.json");
    std::remove(every_path.c_str());
    ActivationCounts counts(every_path, 2);
    ActivationCounts shadow(cwd_path("dsv41_moe_counts_test.shadow.json"), 1000000);
    std::string last_written;
    bool ok = true;
    for (int call = 1; call <= 5; ++call) {
        ggml_tensor* ids = ids_tensor(ctx, 2, {call % 4, (call + 1) % 4});
        counts.record(0, 4, ids);
        shadow.record(0, 4, ids);
        const bool want_write = call % 2 == 1;
        const std::string now = slurp(every_path);
        ok = ok && (want_write ? now == shadow.json() : now == last_written);
        last_written = now;
    }
    check(ok, "every 2 calls: the file written at calls 1, 3 and 5 with all the counts so far, unchanged between");
    std::remove(every_path.c_str());
    std::remove(path.c_str());

    check_aborts([&] { ActivationCounts c(cwd_path("dsv41_moe_counts_test.bad.json"), 10);
                       c.record(0, 4, ids_tensor(ctx, 2, {1, 4})); },
                 "an expert id outside the layer aborts, naming it:", {"layer 0, token 0 slot 1 is expert 4"});
    check_aborts([&] { ActivationCounts c(cwd_path("dsv41_moe_counts_test.bad.json"), 10);
                       c.record(0, 4, ids_tensor(ctx, 2, {1, 2}));
                       c.record(0, 8, ids_tensor(ctx, 2, {1, 2})); },
                 "a layer whose expert count changes aborts:", {"layer 0 has 8 experts in this call and 4 before"});
    check_aborts([&] { ActivationCounts c("/nonexistent-dir/counts.json", 10);
                       c.record(0, 4, ids_tensor(ctx, 2, {1, 2})); },
                 "a path that cannot be written aborts at the first call:",
                 {"STRATA_MOE_COUNTS: cannot write /nonexistent-dir/counts.json.tmp"});
    std::remove(cwd_path("dsv41_moe_counts_test.bad.json").c_str());
    std::remove(cwd_path("dsv41_moe_counts_test.shadow.json").c_str());
    ggml_free(ctx);
}

// ---------------------------------------------------------------- 4: through the op

constexpr int64_t kEmbd = 256, kFf = 256, kExperts = 16, kUsed = 4;

void ref_op(ggml_tensor* dst, int, int nth, void*) {
    strata::dsv41::forward(dst, nth, strata_moe_clamp(dst), nullptr, nullptr, strata::dsv41::CpuKernel::kRef);
}

void test_through_op() {
    const std::string path = cwd_path("dsv41_moe_counts_test.op.json");
    std::remove(path.c_str());
    uint64_t rng = 7;
    auto next = [&rng] {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        return rng >> 33;
    };
    // the routing of both calls: 1 token, then 5, each slot a distinct expert of the token
    std::vector<std::vector<int32_t>> calls;
    for (const int64_t n_tok : {1, 5}) {
        std::vector<int32_t> ids;
        for (int64_t t = 0; t < n_tok; ++t) {
            std::vector<int32_t> pick;
            while ((int64_t) pick.size() < kUsed) {
                const int32_t e = (int32_t) (next() % kExperts);
                bool seen = false;
                for (const int32_t x : pick) seen = seen || x == e;
                if (!seen) pick.push_back(e);
            }
            ids.insert(ids.end(), pick.begin(), pick.end());
        }
        calls.push_back(ids);
    }

    std::string err;
    const int status = run_child([&] {
        setenv("STRATA_MOE_COUNTS", path.c_str(), 1);
        setenv("STRATA_MOE_COUNTS_EVERY", "1000", 1);
        ggml_init_params params = {/*.mem_size =*/ (size_t) 64 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
        ggml_context* ctx = ggml_init(params);
        const struct {
            ggml_type type;
            int64_t n_per_row, n_rows;
            const char* name;
        } parts[3] = {{GGML_TYPE_Q4_K, kEmbd, kFf, "blk.2.ffn_up_exps.weight"},
                      {GGML_TYPE_Q4_K, kEmbd, kFf, "blk.2.ffn_gate_exps.weight"},
                      {GGML_TYPE_Q6_K, kFf, kEmbd, "blk.2.ffn_down_exps.weight"}};
        ggml_tensor* w[3];
        for (int p = 0; p < 3; ++p) {
            std::vector<float> src((size_t) (parts[p].n_per_row * parts[p].n_rows * kExperts));
            for (float& v : src) v = (float) (next() % 1000) / 4000.0f - 0.125f;
            w[p] = ggml_new_tensor_3d(ctx, parts[p].type, parts[p].n_per_row, parts[p].n_rows, kExperts);
            ggml_set_name(w[p], parts[p].name);
            ggml_quantize_chunk(parts[p].type, src.data(), w[p]->data, 0, parts[p].n_rows * kExperts,
                                parts[p].n_per_row, nullptr);
        }
        for (const std::vector<int32_t>& ids : calls) {
            const int64_t n_tok = (int64_t) ids.size() / kUsed;
            ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbd, n_tok);
            for (int64_t i = 0; i < kEmbd * n_tok; ++i) {
                ggml_get_data_f32(cur)[i] = (float) (next() % 100) / 50.0f - 1.0f;
            }
            ggml_tensor* sel = ids_tensor(ctx, kUsed, ids);
            ggml_tensor* weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kUsed, n_tok);
            for (int64_t i = 0; i < kUsed * n_tok; ++i) ggml_get_data_f32(weights)[i] = 0.25f;
            ggml_tensor* args[] = {cur, sel, weights, w[0], w[1], w[2]};
            ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, kEmbd, n_tok, 1, 1, args, 6, ref_op, GGML_N_TASKS_MAX,
                                              nullptr);
            strata_moe_set_clamp(out, 0.0f);
            ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) _exit(5);
        }
        std::exit(0);  // the exit handlers write the counts
    }, &err);
    ggml_init_params params = {/*.mem_size =*/ (size_t) 1 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    ggml_context* ctx = ggml_init(params);
    ActivationCounts want(cwd_path("dsv41_moe_counts_test.want.json"), 1000000);
    for (const std::vector<int32_t>& ids : calls) want.record(2, kExperts, ids_tensor(ctx, kUsed, ids));
    const std::string got = slurp(path);
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0 && got == want.json() &&
              err.find("strata-moe: STRATA_MOE_COUNTS=" + path + ": per-(layer, expert) activation counts written "
                       "every 1000 calls and at exit") != std::string::npos,
          "through the op: STRATA_MOE_COUNTS set, 1 + 5 tokens on layer 2, the file written at exit has the routed "
          "experts' counts (child exit status %d):\n%s", status, got.c_str());
    std::remove(path.c_str());
    std::remove(cwd_path("dsv41_moe_counts_test.want.json").c_str());
    ggml_free(ctx);
}

}  // namespace

int main() {
    ggml_cpu_init();
    test_counter();
    test_through_op();
    std::printf("dsv41_moe_counts_test: %s (%d failed)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
