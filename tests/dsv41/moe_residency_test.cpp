// tests/dsv41/moe_residency_test.cpp - the cold reader and the tier map of strata-moe's residency
// (moe_residency.h), GPU free. Its files go in the working directory (ctest: the build tree), since O_DIRECT needs
// a real filesystem, not tmpfs.
//
// 1. Cold reader: on a 3 MiB + 1234 byte file of random bytes, mapped_range finds every range's file and offset
//    from /proc/self/maps, and read_direct gives the mapped bytes bit for bit: ranges at unaligned offsets, across
//    block boundaries and ending at the end of the file (a short read there); with a pread that returns at most 8 KiB
//    per call and EINTR first, every call it makes is 4096-aligned in offset, length and buffer address, and the
//    bytes still match. A short read before the wanted bytes, an anonymous range and a budget or RLIMIT_MEMLOCK too
//    small for the warm tier abort (in a child process; the abort message is printed).
// 2. Tier map and the op: one layer of 16 experts (n_embd 512, n_ff 256, gate/up q4_K, down q6_K, as the OLMoE smoke
//    model has them) written at unaligned offsets of a file and mapped as the fork maps a GGUF; 4 warm experts, the
//    rest cold. strata::dsv41::forward with the residency gives the output of forward on the mapped tensors bit for
//    bit, with both CPU kernels, 1, 5 and 32 tokens and rounds of 2 and 16 cold experts (so 32 tokens take several
//    rounds through the two-round staging); each call reads every cold expert it routes to once (3 parts) and no
//    warm one. A hot set's VRAM experts are cold without a GPU tier and outside the map with one.
//    STRATA_MOE_COLD=pagecache: the cold experts are computed from the mapped tensors, so the output is the
//    same bits with no cold read at all (only the warm fill and the head check, once); prepare marks no expert for
//    reading while the tier map still has them cold; choose_cold_path's default (pagecache) and errors. All other
//    cases run with direct.
// 3. The warm tier's RAM modes (STRATA_MOE_RAM_MODE): choose_ram_mode's default (pinned with a CUDA device, locked
//    without) and its errors (an unknown value, pinned without a device); warm_alloc_error names the size and the
//    limit of each mode; the op gives the mapped tensors' output bit for bit with the warm tier locked, plain and
//    pinned (a stand-in for cudaHostAlloc that returns memory off the 4096 alignment, which the residency must
//    round up; the CUDA allocator itself needs a GPU and is checked on the pod); under an RLIMIT_MEMLOCK of 4 KiB,
//    plain and pinned allocate and locked aborts; a failing pinned allocator aborts naming the size.
#include "cpu_kernel.h"
#include "moe_op.h"
#include "moe_residency.h"
#include "strata/dsv41/moe.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace {

using strata::dsv41::ColdPath;
using strata::dsv41::CpuKernel;
using strata::dsv41::DirectSpan;
using strata::dsv41::HotLayer;
using strata::dsv41::HotSet;
using strata::dsv41::kDirectAlign;
using strata::dsv41::MemoryLimits;
using strata::dsv41::PinnedHost;
using strata::dsv41::RamMode;
using strata::dsv41::Residency;

constexpr int64_t kEmbd = 512;
constexpr int64_t kFf = 256;
constexpr int64_t kExperts = 16;
constexpr int64_t kUsed = 4;
constexpr int kThreads = 4;
const std::vector<int32_t> kWarm = {3, 9, 0, 14};

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

struct Rng {
    uint64_t s;
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * (float) (next() >> 40) * (1.0f / 16777216.0f); }
};

// a file of `bytes` mapped read-only and shared, as llama.cpp maps a GGUF
struct MappedFile {
    std::string path;
    int fd = -1;
    char* map = nullptr;
    size_t size = 0;
};

MappedFile write_and_map(const char* name, const std::vector<char>& bytes) {
    MappedFile f;
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        std::printf("FAIL: getcwd: %s\n", std::strerror(errno));
        std::exit(1);
    }
    f.path = std::string(cwd) + "/" + name;
    FILE* out = std::fopen(f.path.c_str(), "wb");
    if (out == nullptr || std::fwrite(bytes.data(), 1, bytes.size(), out) != bytes.size() || std::fclose(out) != 0) {
        std::printf("FAIL: cannot write %s\n", f.path.c_str());
        std::exit(1);
    }
    f.size = bytes.size();
    f.fd = open(f.path.c_str(), O_RDONLY);
    f.map = static_cast<char*>(mmap(nullptr, f.size, PROT_READ, MAP_SHARED, f.fd, 0));
    if (f.map == MAP_FAILED) {
        std::printf("FAIL: mmap %s: %s\n", f.path.c_str(), std::strerror(errno));
        std::exit(1);
    }
    return f;
}

void unmap(MappedFile& f) {
    munmap(f.map, f.size);
    close(f.fd);
    unlink(f.path.c_str());
}

int open_direct(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        std::printf("FAIL: open %s with O_DIRECT: %s\n", path.c_str(), std::strerror(errno));
        std::exit(1);
    }
    return fd;
}

char* aligned_buffer(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, kDirectAlign, bytes) != 0) std::exit(1);
    return static_cast<char*>(p);
}

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

// The abort message in a child's stderr: the line GGML_ABORT prints ("file:line: strata-moe: ..."), else the first.
std::string abort_line(const std::string& err) {
    for (size_t b = 0; b < err.size();) {
        const size_t e = err.find('\n', b);
        const std::string line = err.substr(b, e == std::string::npos ? std::string::npos : e - b);
        if (line.find(": strata-moe: ") != std::string::npos) return line;
        if (e == std::string::npos) break;
        b = e + 1;
    }
    return err.substr(0, err.find('\n'));
}

// Runs fn in a child process: the case passes when the child aborts (SIGABRT), as GGML_ABORT does, and its abort
// message holds every one of `needles`. The abort message is printed under the case.
void check_aborts(const std::function<void()>& fn, const char* what, const std::vector<std::string>& needles = {}) {
    std::string err;
    const int status = run_child(fn, &err);
    const std::string line = abort_line(err);
    bool has = true;
    for (const std::string& n : needles) has = has && line.find(n) != std::string::npos;
    check(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && has, "%s", what);
    std::printf("       %s\n", line.c_str());
}

// ---------------------------------------------------------------- 1. the cold reader

std::atomic<int64_t> g_calls{0}, g_misaligned{0}, g_bytes{0};
bool g_eintr_pending = false;

ssize_t pread_small_chunks(int fd, void* buf, size_t n, off_t offset) {
    ++g_calls;
    if (reinterpret_cast<uintptr_t>(buf) % kDirectAlign || (size_t) offset % kDirectAlign || n % kDirectAlign) {
        ++g_misaligned;
    }
    if (g_eintr_pending) {
        g_eintr_pending = false;
        errno = EINTR;
        return -1;
    }
    return pread(fd, buf, std::min<size_t>(n, 8192), offset);
}

ssize_t pread_eof_after_one_block(int fd, void* buf, size_t n, off_t offset) {
    return offset == 0 ? pread(fd, buf, std::min(n, kDirectAlign), offset) : 0;
}

ssize_t pread_counting(int fd, void* buf, size_t n, off_t offset) {
    const ssize_t r = pread(fd, buf, n, offset);
    if (r > 0) {
        ++g_calls;
        g_bytes += r;
    }
    return r;
}

void test_reader() {
    Rng rng{1};
    std::vector<char> bytes((3u << 20) + 1234);
    for (char& b : bytes) b = (char) rng.next();
    MappedFile f = write_and_map("dsv41_moe_residency_test.reader.bin", bytes);
    const int fd = open_direct(f.path);
    const uint64_t size = f.size;
    const struct {
        uint64_t offset;
        size_t len;
    } ranges[] = {{0, 100}, {1, 4095}, {4095, 2}, {4096, 4096}, {12345, 1 << 20}, {size - 1000, 1000},
                  {size - 5000, 5000}, {size - 1, 1}, {777, (size_t) (size - 777)}};
    for (const auto& r : ranges) {
        const strata::dsv41::MappedRange m = strata::dsv41::mapped_range(f.map + r.offset, r.len, "range");
        const DirectSpan s = strata::dsv41::direct_span(r.offset, r.len);
        const size_t slot = strata::dsv41::direct_slot_bytes(r.len);
        char* buf = aligned_buffer(slot);
        strata::dsv41::read_direct(fd, f.path.c_str(), s, r.len, buf, pread);
        const bool same = std::memcmp(buf + s.lead, f.map + r.offset, r.len) == 0;
        const bool eof = s.offset + s.len > size;
        check(m.path == f.path && m.offset == r.offset && s.len <= slot && same,
              "range at %8llu, %8zu bytes: maps offset %8llu, O_DIRECT span %8zu bytes from %8llu%s, bytes %s",
              (unsigned long long) r.offset, r.len, (unsigned long long) m.offset, s.len,
              (unsigned long long) s.offset, eof ? " (short read at the end of the file)" : "",
              same ? "identical to the mapping" : "DIFFER");

        g_calls = 0;
        g_misaligned = 0;
        g_eintr_pending = true;
        std::memset(buf, 0, slot);
        strata::dsv41::read_direct(fd, f.path.c_str(), s, r.len, buf, pread_small_chunks);
        const bool same_chunked = std::memcmp(buf + s.lead, f.map + r.offset, r.len) == 0;
        check(same_chunked && g_misaligned == 0,
              "  the same through 8 KiB short reads after an EINTR: %lld calls, %lld misaligned, bytes %s",
              (long long) g_calls.load(), (long long) g_misaligned.load(), same_chunked ? "identical" : "DIFFER");
        std::free(buf);
    }

    const DirectSpan s = strata::dsv41::direct_span(100, 3 * kDirectAlign);
    char* buf = aligned_buffer(strata::dsv41::direct_slot_bytes(3 * kDirectAlign));
    check_aborts([&] { strata::dsv41::read_direct(fd, f.path.c_str(), s, 3 * kDirectAlign, buf, pread_eof_after_one_block); },
                 "a read that ends before the wanted bytes aborts:");
    check_aborts([&] { strata::dsv41::read_direct(fd, f.path.c_str(), s, 3 * kDirectAlign, buf + 1, pread); },
                 "a buffer off the 4096 alignment aborts:");
    std::free(buf);
    std::vector<char> anon(4096);
    check_aborts([&] { strata::dsv41::mapped_range(anon.data(), anon.size(), "heap bytes"); },
                 "a range outside any file mapping aborts:");
    close(fd);
    unmap(f);
}

// ---------------------------------------------------------------- 2. the tier map and the op

struct Model {
    MappedFile file;
    ggml_context* ctx = nullptr;
    ggml_tensor *up = nullptr, *gate = nullptr, *down = nullptr;
};

// the three expert tensors at unaligned (32-byte, as GGUF aligns) offsets of one file, down ending at its end
Model make_model() {
    Model m;
    const struct {
        ggml_type type;
        int64_t n_per_row, n_rows;
        const char* name;
        float amplitude;
    } parts[3] = {{GGML_TYPE_Q4_K, kEmbd, kFf, "blk.0.ffn_up_exps.weight", 0.25f},
                  {GGML_TYPE_Q4_K, kEmbd, kFf, "blk.0.ffn_gate_exps.weight", 0.25f},
                  {GGML_TYPE_Q6_K, kFf, kEmbd, "blk.0.ffn_down_exps.weight", 0.05f}};
    Rng rng{2};
    std::vector<char> bytes(3 * kDirectAlign + 160);
    for (char& b : bytes) b = (char) rng.next();
    size_t offsets[3];
    for (int p = 0; p < 3; ++p) {
        offsets[p] = bytes.size();
        const size_t row = ggml_row_size(parts[p].type, parts[p].n_per_row);
        std::vector<float> src((size_t) (parts[p].n_per_row * parts[p].n_rows * kExperts));
        for (float& v : src) v = rng.uniform(-parts[p].amplitude, parts[p].amplitude);
        bytes.resize(offsets[p] + row * (size_t) (parts[p].n_rows * kExperts));
        ggml_quantize_chunk(parts[p].type, src.data(), bytes.data() + offsets[p], 0, parts[p].n_rows * kExperts,
                            parts[p].n_per_row, nullptr);
        if (p < 2) bytes.resize(bytes.size() + 96, 0x5a);
    }
    m.file = write_and_map("dsv41_moe_residency_test.model.bin", bytes);
    ggml_init_params params = {/*.mem_size =*/ 4 * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr,
                               /*.no_alloc =*/ true};
    m.ctx = ggml_init(params);
    ggml_tensor** out[3] = {&m.up, &m.gate, &m.down};
    for (int p = 0; p < 3; ++p) {
        ggml_tensor* t = ggml_new_tensor_3d(m.ctx, parts[p].type, parts[p].n_per_row, parts[p].n_rows, kExperts);
        ggml_set_name(t, parts[p].name);
        t->data = m.file.map + offsets[p];
        *out[p] = t;
    }
    return m;
}

HotSet hot_set(const std::vector<int32_t>& vram) {
    HotSet set;
    set.path = "(test hot set)";
    set.budget_gib = 0.0;
    set.ram_budget_gib = 1.0;
    HotLayer l;
    l.layer = 0;
    l.n_expert = kExperts;
    l.n_embd = kEmbd;
    l.n_ff = kFf;
    l.up_type = GGML_TYPE_Q4_K;
    l.gate_type = GGML_TYPE_Q4_K;
    l.down_type = GGML_TYPE_Q6_K;
    l.experts = vram;
    l.warm = kWarm;
    set.layers.push_back(l);
    return set;
}

Residency* g_residency = nullptr;
CpuKernel g_kernel = CpuKernel::kRef;

void residency_op(ggml_tensor* dst, int, int nth, void*) {
    strata::dsv41::forward(dst, nth, strata_moe_clamp(dst), nullptr, g_residency, g_kernel);
}

struct Inputs {
    ggml_context* ctx = nullptr;
    ggml_tensor *cur = nullptr, *ids = nullptr, *weights = nullptr;
    int64_t n_tok = 0;
    std::set<int32_t> cold_used;  // the distinct cold experts the tokens route to
};

Inputs make_inputs(int64_t n_tok, uint64_t seed) {
    Inputs in;
    in.n_tok = n_tok;
    ggml_init_params params = {/*.mem_size =*/ (size_t) 8 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    in.ctx = ggml_init(params);
    Rng rng{seed};
    in.cur = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, kEmbd, n_tok);
    for (int64_t i = 0; i < kEmbd * n_tok; ++i) ggml_get_data_f32(in.cur)[i] = rng.uniform(-1.0f, 1.0f);
    in.ids = ggml_new_tensor_2d(in.ctx, GGML_TYPE_I32, kUsed, n_tok);
    in.weights = ggml_new_tensor_3d(in.ctx, GGML_TYPE_F32, 1, kUsed, n_tok);
    const std::set<int32_t> warm(kWarm.begin(), kWarm.end());
    for (int64_t t = 0; t < n_tok; ++t) {
        std::vector<int32_t> pick((size_t) kExperts);
        for (int32_t e = 0; e < kExperts; ++e) pick[(size_t) e] = e;
        for (size_t i = pick.size() - 1; i > 0; --i) std::swap(pick[i], pick[rng.next() % (uint64_t) (i + 1)]);
        for (int64_t s = 0; s < kUsed; ++s) {
            const int32_t e = pick[(size_t) s];
            static_cast<int32_t*>(in.ids->data)[t * kUsed + s] = e;
            ggml_get_data_f32(in.weights)[t * kUsed + s] = rng.uniform(0.05f, 0.5f);
            if (!warm.count(e)) in.cold_used.insert(e);
        }
    }
    return in;
}

std::vector<float> compute(const Inputs& in, const Model& m, Residency* residency) {
    ggml_init_params params = {/*.mem_size =*/ (size_t) 4 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* args[] = {in.cur, in.ids, in.weights, m.up, m.gate, m.down};
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, kEmbd, in.n_tok, 1, 1, args, 6, residency_op,
                                      GGML_N_TASKS_MAX, nullptr);
    strata_moe_set_clamp(out, 0.0f);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    g_residency = residency;
    if (ggml_graph_compute_with_ctx(ctx, gf, kThreads) != GGML_STATUS_SUCCESS) {
        std::printf("FAIL: graph compute\n");
        std::exit(1);
    }
    std::vector<float> v(ggml_get_data_f32(out), ggml_get_data_f32(out) + kEmbd * in.n_tok);
    ggml_free(ctx);
    return v;
}

void test_tiers() {
    Model m = make_model();
    std::printf("tier map and op: n_embd %lld n_ff %lld, %lld experts, %lld used, q4_K gate/up, q6_K down, warm "
                "experts {3, 9, 0, 14}, %d threads\n", (long long) kEmbd, (long long) kFf, (long long) kExperts,
                (long long) kUsed, kThreads);

    // the VRAM experts of a hot set: cold without a GPU tier, outside the map (not cold) with one
    for (const bool in_vram : {false, true}) {
        Residency r(hot_set({5, 6}), 1.0, in_vram, 2, 16, pread, RamMode::kLocked, nullptr, ColdPath::kDirect);
        strata::dsv41::Call c;
        c.layer = 0;
        c.up = m.up;
        c.gate = m.gate;
        c.down = m.down;
        const std::vector<uint8_t>& cold = r.prepare(c);
        bool ok = cold.size() == (size_t) kExperts;
        for (int32_t e = 0; ok && e < kExperts; ++e) {
            const bool warm = e == 3 || e == 9 || e == 0 || e == 14;
            const bool vram = e == 5 || e == 6;
            ok = cold[(size_t) e] == (uint8_t) (!warm && !(vram && in_vram));
        }
        check(ok && r.warm_experts() == 4, "tier map, %s GPU tier: warm {0, 3, 9, 14}, VRAM {5, 6} %s, the rest cold",
              in_vram ? "with a" : "without a", in_vram ? "outside the map" : "cold");
    }

    for (const CpuKernel kernel : {CpuKernel::kRef, CpuKernel::kFast}) {
        if (kernel == CpuKernel::kFast && !strata::dsv41::fast_kernel_cpu_ok()) {
            check(false, "this CPU lacks AVX2, FMA or F16C, the fast kernel cannot run");
            continue;
        }
        g_kernel = kernel;
        const char* name = kernel == CpuKernel::kFast ? "fast" : "ref";
        for (const int64_t round : {2, 16}) {
            Residency r(hot_set({}), 1.0, false, 3, round, pread_counting, RamMode::kLocked, nullptr,
                        ColdPath::kDirect);
            uint64_t seed = 11;
            for (const int64_t n_tok : {1, 5, 32}) {
                const Inputs in = make_inputs(n_tok, seed++);
                const std::vector<float> mapped = compute(in, m, nullptr);
                g_calls = 0;
                g_bytes = 0;
                // the layer's first call also reads the warm slots and the head check (4 + 3 reads, once)
                const bool first = n_tok == 1;
                const std::vector<float> tiered = compute(in, m, &r);
                const int64_t want_reads = 3 * (int64_t) in.cold_used.size() + (first ? 3 * 4 + 3 : 0);
                const bool same = std::memcmp(mapped.data(), tiered.data(), mapped.size() * sizeof(float)) == 0;
                const int64_t n_rounds = ((int64_t) in.cold_used.size() + round - 1) / round;
                check(same && g_calls == want_reads,
                      "%-4s rounds of %2lld, %2lld tokens: %2zu cold experts in %lld round%s, %3lld reads (expected "
                      "%3lld%s), %7.1f KiB; output %s",
                      name, (long long) round, (long long) n_tok, in.cold_used.size(), (long long) n_rounds,
                      n_rounds == 1 ? "" : "s", (long long) g_calls.load(), (long long) want_reads,
                      first ? ", with the warm fill" : "", (double) g_bytes.load() / 1024.0,
                      same ? "identical to the mapped tensors' bit for bit" : "DIFFERS from the mapped tensors'");
                ggml_free(in.ctx);
            }
        }
    }

    check_aborts([&] {
        Residency r(hot_set({}), 0.0001, false, 1, 16, pread, RamMode::kLocked, nullptr, ColdPath::kDirect);
    },
                 "a warm tier over STRATA_MOE_RAM_GB aborts:");

    // STRATA_MOE_COLD=pagecache: the cold experts from the mapping, no read per call
    {
        Residency r(hot_set({5, 6}), 1.0, true, 2, 16, pread, RamMode::kLocked, nullptr, ColdPath::kPageCache);
        strata::dsv41::Call c;
        c.layer = 0;
        c.up = m.up;
        c.gate = m.gate;
        c.down = m.down;
        const std::vector<uint8_t>& read = r.prepare(c);
        const std::vector<uint8_t>& cold = r.cold_map(0);
        bool ok = read.size() == (size_t) kExperts && cold.size() == (size_t) kExperts;
        for (int32_t e = 0; ok && e < kExperts; ++e) {
            const bool warm = e == 3 || e == 9 || e == 0 || e == 14;
            ok = read[(size_t) e] == 0 && cold[(size_t) e] == (uint8_t) (!warm && e != 5 && e != 6);
        }
        check(ok, "pagecache tier map: no expert read per call, warm {0, 3, 9, 14}, VRAM {5, 6} outside the map, the "
              "rest cold");
    }
    for (const CpuKernel kernel : {CpuKernel::kRef, CpuKernel::kFast}) {
        if (kernel == CpuKernel::kFast && !strata::dsv41::fast_kernel_cpu_ok()) continue;
        g_kernel = kernel;
        Residency r(hot_set({}), 1.0, false, 3, 16, pread_counting, RamMode::kLocked, nullptr, ColdPath::kPageCache);
        uint64_t seed = 31;
        for (const int64_t n_tok : {1, 5, 32}) {
            const Inputs in = make_inputs(n_tok, seed++);
            const std::vector<float> mapped = compute(in, m, nullptr);
            g_calls = 0;
            const bool first = n_tok == 1;
            const std::vector<float> tiered = compute(in, m, &r);
            const int64_t want_reads = first ? 3 * 4 + 3 : 0;
            const bool same = std::memcmp(mapped.data(), tiered.data(), mapped.size() * sizeof(float)) == 0;
            check(same && g_calls == want_reads,
                  "%-4s pagecache, %2lld tokens: %2zu cold experts from the mapping, %2lld reads (expected %2lld%s); "
                  "output %s", kernel == CpuKernel::kFast ? "fast" : "ref", (long long) n_tok, in.cold_used.size(),
                  (long long) g_calls.load(), (long long) want_reads, first ? ", the warm fill" : ", none",
                  same ? "identical to the mapped tensors' bit for bit" : "DIFFERS from the mapped tensors'");
            ggml_free(in.ctx);
        }
    }
    const struct {
        const char* env;
        ColdPath path;
        bool is_default;
        const char* error;
    } paths[] = {{nullptr, ColdPath::kPageCache, true, nullptr},  {"", ColdPath::kPageCache, true, nullptr},
                 {"pagecache", ColdPath::kPageCache, false, nullptr}, {"direct", ColdPath::kDirect, false, nullptr},
                 {"odirect", ColdPath::kPageCache, false, "STRATA_MOE_COLD=odirect: expected pagecache (the default) or "
                                                          "direct"}};
    for (const auto& p : paths) {
        const strata::dsv41::ColdPathChoice got = strata::dsv41::choose_cold_path(p.env);
        const bool ok = p.error != nullptr ? got.error == p.error
                                           : got.error.empty() && got.path == p.path && got.is_default == p.is_default;
        check(ok, "choose_cold_path(%s%s%s): %s", p.env ? "\"" : "", p.env ? p.env : "null", p.env ? "\"" : "",
              got.error.empty() ? (std::string(strata::dsv41::cold_path_name(got.path)) +
                                   (got.is_default ? ", the default" : "")).c_str()
                                : got.error.c_str());
    }
    ggml_free(m.ctx);
    unmap(m.file);
}

// ---------------------------------------------------------------- 3. the warm tier's RAM modes

// a stand-in for cudaHostAlloc: memory 64 bytes past a page start (cudaHostAlloc promises less than 4096), and the
// pointers it handed out, to check that the residency frees the one it got
int64_t g_pinned_allocs = 0, g_pinned_frees = 0;
std::set<void*> g_pinned_live;

void* fake_pinned_alloc(size_t bytes, std::string*) {
    void* base = mmap(nullptr, bytes + 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return nullptr;
    char* p = static_cast<char*>(base) + 64;
    std::memcpy(base, &bytes, sizeof(bytes));  // the length, for the free
    ++g_pinned_allocs;
    g_pinned_live.insert(p);
    return p;
}

void fake_pinned_free(void* p) {
    ++g_pinned_frees;
    if (g_pinned_live.erase(p) == 0) {
        std::printf("FAIL: the residency freed %p, which the pinned allocator did not hand out\n", p);
        std::exit(1);
    }
    char* base = static_cast<char*>(p) - 64;
    size_t bytes = 0;
    std::memcpy(&bytes, base, sizeof(bytes));
    munmap(base, bytes + 4096);
}

void* failing_pinned_alloc(size_t, std::string* error) {
    *error = "cudaErrorMemoryAllocation (out of memory, test stand-in)";
    return nullptr;
}

const PinnedHost kFakePinned{fake_pinned_alloc, fake_pinned_free};

void test_ram_modes() {
    using strata::dsv41::choose_ram_mode;
    std::printf("RAM modes (STRATA_MOE_RAM_MODE):\n");
    const struct {
        const char* env;
        bool device;
        RamMode mode;
        bool is_default;
        const char* error;  // a part of the error text, or null for none
    } choices[] = {
        {nullptr, true, RamMode::kPinned, true, nullptr},  {"", true, RamMode::kPinned, true, nullptr},
        {nullptr, false, RamMode::kLocked, true, nullptr}, {"", false, RamMode::kLocked, true, nullptr},
        {"pinned", true, RamMode::kPinned, false, nullptr}, {"locked", true, RamMode::kLocked, false, nullptr},
        {"locked", false, RamMode::kLocked, false, nullptr}, {"plain", true, RamMode::kPlain, false, nullptr},
        {"plain", false, RamMode::kPlain, false, nullptr},
        {"pinned", false, RamMode::kPinned, false, "STRATA_MOE_RAM_MODE=pinned needs a CUDA device"},
        {"mlock", true, RamMode::kLocked, false,
         "STRATA_MOE_RAM_MODE=mlock: expected pinned (the default with a CUDA device), locked (the default without "
         "one) or plain"},
    };
    for (const auto& c : choices) {
        const strata::dsv41::RamModeChoice got = choose_ram_mode(c.env, c.device);
        const bool ok = c.error != nullptr ? got.error.find(c.error) != std::string::npos
                                           : got.error.empty() && got.mode == c.mode && got.is_default == c.is_default;
        check(ok, "choose_ram_mode(%s%s%s, %s CUDA device): %s", c.env ? "\"" : "", c.env ? c.env : "null",
              c.env ? "\"" : "", c.device ? "a" : "no",
              got.error.empty() ? (std::string(strata::dsv41::ram_mode_name(got.mode)) +
                                   (got.is_default ? ", the default" : "")).c_str()
                                : got.error.c_str());
    }

    // the messages: each names the size, the cause and its mode's limit (a measured pod's limits)
    MemoryLimits pod;
    pod.memlock = "0.008 GiB (8388608 bytes)";
    pod.cgroup = "116.415 GiB (124999999488 bytes) (memory.limit_in_bytes)";
    pod.cgroup_bytes = 124999999488ull;
    const size_t warm = 42949505024ull;
    const struct {
        RamMode mode;
        std::vector<std::string> needles;
    } messages[] = {
        {RamMode::kPinned, {"STRATA_MOE_RAM_MODE=pinned: cudaHostAlloc of the 40.000 GiB warm tier (42949505024 bytes) "
                            "failed: CAUSE", "memory cgroup, whose limit is 116.415 GiB (124999999488 bytes)",
                            "outside RLIMIT_MEMLOCK"}},
        {RamMode::kLocked, {"STRATA_MOE_RAM_MODE=locked: mlock of the 40.000 GiB warm tier (42949505024 bytes) failed: "
                            "CAUSE", "RLIMIT_MEMLOCK is 0.008 GiB (8388608 bytes)", "ulimit -l unlimited",
                            "STRATA_MOE_RAM_MODE=pinned"}},
        {RamMode::kPlain, {"STRATA_MOE_RAM_MODE=plain: pre-faulting the 40.000 GiB warm tier (42949505024 bytes) "
                           "failed: CAUSE", "the memory cgroup's limit is 116.415 GiB (124999999488 bytes)"}},
    };
    for (const auto& m : messages) {
        const std::string text = strata::dsv41::warm_alloc_error(m.mode, warm, "CAUSE", pod);
        bool ok = text.rfind("strata-moe: ", 0) == 0;
        for (const std::string& n : m.needles) ok = ok && text.find(n) != std::string::npos;
        check(ok, "warm_alloc_error, %s: %s", strata::dsv41::ram_mode_name(m.mode), text.c_str());
    }
    const MemoryLimits here = strata::dsv41::memory_limits();
    check(!here.memlock.empty() && !here.cgroup.empty(), "memory_limits here: RLIMIT_MEMLOCK %s, memory cgroup %s",
          here.memlock.c_str(), here.cgroup.c_str());

    // the op with the warm tier in each mode, against the mapped tensors
    Model m = make_model();
    g_kernel = CpuKernel::kRef;
    for (const RamMode mode : {RamMode::kLocked, RamMode::kPlain, RamMode::kPinned}) {
        g_pinned_allocs = g_pinned_frees = 0;
        bool same = true;
        {
            Residency r(hot_set({}), 1.0, false, 3, 16, pread, mode, &kFakePinned, ColdPath::kDirect);
            uint64_t seed = 21;
            for (const int64_t n_tok : {1, 5, 32}) {
                const Inputs in = make_inputs(n_tok, seed++);
                const std::vector<float> mapped = compute(in, m, nullptr);
                const std::vector<float> tiered = compute(in, m, &r);
                same = same && std::memcmp(mapped.data(), tiered.data(), mapped.size() * sizeof(float)) == 0;
                ggml_free(in.ctx);
            }
        }
        const bool pinned = mode == RamMode::kPinned;
        const bool alloc_ok = pinned ? g_pinned_allocs == 1 && g_pinned_frees == 1 : g_pinned_allocs == 0;
        check(same && alloc_ok, "ref, warm tier %s%s, 1/5/32 tokens: output %s", strata::dsv41::ram_mode_name(mode),
              pinned ? " (allocator 64 bytes off the 4096 alignment: allocated and freed once)" : "",
              same ? "identical to the mapped tensors' bit for bit" : "DIFFERS from the mapped tensors'");
    }

    // RLIMIT_MEMLOCK at 4 KiB, as the RunPod containers have 8 MiB: plain and pinned need none of it, locked aborts
    const auto under_memlock = [&](RamMode mode, const PinnedHost* pinned) {
        rlimit lim{4096, 4096};
        if (setrlimit(RLIMIT_MEMLOCK, &lim) != 0) _exit(3);
        Residency r(hot_set({}), 1.0, false, 1, 16, pread, mode, pinned, ColdPath::kDirect);
        strata::dsv41::Call c;
        c.layer = 0;
        c.up = m.up;
        c.gate = m.gate;
        c.down = m.down;
        r.prepare(c);  // fills the warm tier with O_DIRECT reads
    };
    for (const RamMode mode : {RamMode::kPlain, RamMode::kPinned}) {
        std::string err;
        const int status = run_child([&] { under_memlock(mode, &kFakePinned); }, &err);
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "warm tier %s under RLIMIT_MEMLOCK 4 KiB: allocated and "
              "filled (child exit status %d)", strata::dsv41::ram_mode_name(mode), status);
    }
    if (geteuid() == 0) {
        std::printf("skip a locked warm tier over RLIMIT_MEMLOCK: running as root, whose CAP_IPC_LOCK ignores the "
                    "limit\n");
    } else {
        check_aborts([&] { under_memlock(RamMode::kLocked, nullptr); },
                     "a locked warm tier over RLIMIT_MEMLOCK 4 KiB aborts (mlock fails), naming the size and the limit:",
                     {"STRATA_MOE_RAM_MODE=locked: mlock of the", "RLIMIT_MEMLOCK is 0.000 GiB (4096 bytes)"});
    }
    const PinnedHost failing{failing_pinned_alloc, fake_pinned_free};
    check_aborts([&] {
        Residency r(hot_set({}), 1.0, false, 1, 16, pread, RamMode::kPinned, &failing, ColdPath::kDirect);
    },
                 "a failing pinned allocator aborts, naming the size, the cause and the cgroup limit:",
                 {"STRATA_MOE_RAM_MODE=pinned: cudaHostAlloc of the", "cudaErrorMemoryAllocation (out of memory, test stand-in)",
                  "memory cgroup, whose limit is"});
    check_aborts([&] {
        Residency r(hot_set({}), 1.0, false, 1, 16, pread, RamMode::kPinned, nullptr, ColdPath::kDirect);
    },
                 "pinned without the CUDA runtime's allocator aborts:");
    ggml_free(m.ctx);
    unmap(m.file);
}

}  // namespace

int main() {
    ggml_cpu_init();
    test_reader();
    test_tiers();
    test_ram_modes();
    std::printf("dsv41_moe_residency_test: %s (%d failed)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
