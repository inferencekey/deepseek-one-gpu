// src/dsv41/moe_residency.cpp - the warm and cold tiers of strata-moe (moe_residency.h).
//
// Why each choice:
// - An A100 SXM pod run had 40k-60k major page faults per second: the 246 GiB model is mmapped into a 116.4 GiB
//   cgroup, so the missed experts came from NVMe through a page cache that kept evicting them. The warm experts sit
//   in a buffer the kernel does not reclaim. With STRATA_MOE_COLD=direct the page cache is out of the expert
//   path: the cold ones are read with O_DIRECT, and the fork's mapping of the routed experts is dropped once a
//   layer's tiers are filled (MADV_DONTNEED on the mapping, then POSIX_FADV_DONTNEED on the file range). That is
//   safe: the mapping is read-only and file-backed, so a page dropped by mistake would be read back from the file,
//   never lost, and with the hook on nothing but strata-moe reads the routed experts. Only the pages wholly inside
//   the expert tensors are dropped, never a neighbour tensor's.
// - pagecache, the default: an A100 SXM pod showed direct throwing away a page cache that worked as a ~100 GB
//   warm tier (hook-70 served the same misses in ~59 ms per decode pass against direct's 168 ms of waiting), so the
//   cold experts are computed from the fork's mapping and its pages are left to the kernel: the warm tier takes the
//   experts a profile says are hot, the page cache whatever else the RAM left over holds.
// - The warm slots are filled with the same O_DIRECT reads as the cold tier, not by copying from the fork's
//   mapping: copying would pull the whole warm set through the page cache (40 GiB on the pod) and evict the
//   dense weights and the engram rows it is meant to keep. They are the same bytes (dsv41_moe_residency_test
//   compares them with the mapping), and the first read of each layer is compared with its mapping too.
// - Reads go to a small pool of reader threads with plain pread, not io_uring: Docker's default seccomp profile
//   blocks the io_uring syscalls, and blocking preads from a few threads keep an NVMe queue as busy (each read is
//   one expert part, 1-5 MB, which the block layer splits into parallel requests itself).
// - The warm buffer is CUDA pinned memory by default with a GPU (STRATA_MOE_RAM_MODE, moe_residency.h): a later pod
//   (RunPod, 2026-10-06) had RLIMIT_MEMLOCK 8 MiB hard and no CAP_IPC_LOCK, so the mlock of 40 GiB failed, while
//   cudaHostAlloc pins through the driver, outside RLIMIT_MEMLOCK (vllm-exl3 pinned 144 GiB on RunPod pods).
//   O_DIRECT reads into pinned pages are ordinary reads into resident memory; the buffer is rounded up to 4096
//   bytes, as they need, since cudaHostAlloc promises less alignment than that.
// - Every read is whole 4096-byte blocks into a slot of round_up(len, 4096) + 4096 bytes, so a part at any file
//   offset fits, and the expert's rows start at the same address modulo 4096 as in the mapping.
// - Cold experts come in rounds of round_experts: the staging holds two rounds, the reads of round r + 1 run while
//   the CPU computes round r, and round r + 2 is queued into round r's slots once every thread is past it. A decode
//   step routes fewer experts than a round (DeepSeek-V4.1 6, OLMoE 8), so it is one round and the CPU starts on
//   the warm experts while the cold ones are read; a prompt ubatch takes several rounds with bounded staging.
#include "moe_residency.h"

#include "ggml.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

// Linux 5.14 (MADV_POPULATE_WRITE); older C library headers lack the name
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

namespace strata::dsv41 {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kMiB = 1024.0 * 1024.0;

uint64_t round_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// anonymous memory for the warm buffer and the staging: page-aligned, as O_DIRECT needs
char* map_anonymous(size_t bytes, const char* what) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        GGML_ABORT("strata-moe: cannot map %.3f GiB for the %s: %s", (double) bytes / kGiB, what, std::strerror(errno));
    }
    return static_cast<char*>(p);
}

std::string bytes_text(uint64_t v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GiB (%llu bytes)", (double) v / kGiB, (unsigned long long) v);
    return buf;
}

// The first line of `path`, or empty when it cannot be read.
std::string first_line(const char* path) {
    std::ifstream f(path);
    std::string line;
    if (f) std::getline(f, line);
    return line;
}

}  // namespace

const char* ram_mode_name(RamMode m) {
    switch (m) {
        case RamMode::kPinned: return "pinned";
        case RamMode::kLocked: return "locked";
        case RamMode::kPlain: return "plain";
    }
    return "?";
}

const char* cold_path_name(ColdPath p) {
    switch (p) {
        case ColdPath::kPageCache: return "pagecache";
        case ColdPath::kDirect: return "direct";
    }
    return "?";
}

ColdPathChoice choose_cold_path(const char* env) {
    ColdPathChoice c;
    if (env == nullptr || *env == '\0') {
        c.is_default = true;
    } else if (std::strcmp(env, "pagecache") == 0) {
        c.path = ColdPath::kPageCache;
    } else if (std::strcmp(env, "direct") == 0) {
        c.path = ColdPath::kDirect;
    } else {
        c.error = std::string("STRATA_MOE_COLD=") + env + ": expected pagecache (the default) or direct";
    }
    return c;
}

RamModeChoice choose_ram_mode(const char* env, bool cuda_device) {
    RamModeChoice c;
    if (env == nullptr || *env == '\0') {
        c.mode = cuda_device ? RamMode::kPinned : RamMode::kLocked;
        c.is_default = true;
    } else if (std::strcmp(env, "pinned") == 0) {
        c.mode = RamMode::kPinned;
        if (!cuda_device) {
            c.error = "STRATA_MOE_RAM_MODE=pinned needs a CUDA device (the warm tier is cudaHostAlloc memory) and "
                      "there is none; set locked (mlock) or plain";
        }
    } else if (std::strcmp(env, "locked") == 0) {
        c.mode = RamMode::kLocked;
    } else if (std::strcmp(env, "plain") == 0) {
        c.mode = RamMode::kPlain;
    } else {
        c.error = std::string("STRATA_MOE_RAM_MODE=") + env +
                  ": expected pinned (the default with a CUDA device), locked (the default without one) or plain";
    }
    return c;
}

MemoryLimits memory_limits() {
    MemoryLimits m;
    rlimit lim{};
    if (getrlimit(RLIMIT_MEMLOCK, &lim) != 0) {
        m.memlock = std::string("unknown (getrlimit: ") + std::strerror(errno) + ")";
    } else {
        m.memlock = lim.rlim_cur == RLIM_INFINITY ? "unlimited" : bytes_text(lim.rlim_cur);
    }
    // cgroup v2 first, then v1; a v1 limit near 2^63 is v1's "no limit"
    const struct {
        const char* path;
        const char* name;
    } files[2] = {{"/sys/fs/cgroup/memory.max", "memory.max"},
                  {"/sys/fs/cgroup/memory/memory.limit_in_bytes", "memory.limit_in_bytes"}};
    m.cgroup = "unknown (neither /sys/fs/cgroup/memory.max nor memory/memory.limit_in_bytes is readable)";
    for (const auto& f : files) {
        const std::string v = first_line(f.path);
        if (v.empty()) continue;
        char* end = nullptr;
        const unsigned long long n = std::strtoull(v.c_str(), &end, 10);
        if (v == "max" || (*end == '\0' && n >= (1ull << 62))) {
            m.cgroup = std::string("unlimited (") + f.name + " " + v + ")";
        } else if (*end == '\0') {
            m.cgroup_bytes = n;
            m.cgroup = bytes_text(n) + " (" + f.name + ")";
        } else {
            m.cgroup = std::string("unreadable (") + f.name + " '" + v + "')";
        }
        break;
    }
    return m;
}

std::string warm_alloc_error(RamMode m, size_t bytes, const std::string& cause, const MemoryLimits& limits) {
    char size[96];
    std::snprintf(size, sizeof(size), "%.3f GiB warm tier (%zu bytes)", (double) bytes / kGiB, bytes);
    std::ostringstream o;
    o << "strata-moe: STRATA_MOE_RAM_MODE=" << ram_mode_name(m) << ": ";
    switch (m) {
        case RamMode::kPinned:
            o << "cudaHostAlloc of the " << size << " failed: " << cause << "; pinned memory is outside RLIMIT_MEMLOCK ("
              << limits.memlock << ") but counts in the memory cgroup, whose limit is " << limits.cgroup
              << "; lower STRATA_MOE_RAM_GB or set STRATA_MOE_RAM_MODE=locked or plain";
            break;
        case RamMode::kLocked:
            o << "mlock of the " << size << " failed: " << cause << "; RLIMIT_MEMLOCK is " << limits.memlock
              << " (raise it: ulimit -l unlimited, docker run --ulimit memlock=-1; or STRATA_MOE_RAM_MODE=pinned "
                 "with a CUDA device), and the memory cgroup, whose limit is "
              << limits.cgroup << ", must hold it";
            break;
        case RamMode::kPlain:
            o << "pre-faulting the " << size << " failed: " << cause << "; the memory cgroup's limit is "
              << limits.cgroup << " (RLIMIT_MEMLOCK " << limits.memlock << " does not apply to plain memory)";
            break;
    }
    return o.str();
}

DirectSpan direct_span(uint64_t offset, size_t len) {
    DirectSpan s;
    s.offset = offset / kDirectAlign * kDirectAlign;
    s.len = (size_t) (round_up(offset + len, kDirectAlign) - s.offset);
    s.lead = (size_t) (offset - s.offset);
    return s;
}

size_t direct_slot_bytes(size_t len) { return (size_t) round_up(len, kDirectAlign) + kDirectAlign; }

void read_direct(int fd, const char* path, const DirectSpan& s, size_t len, char* dst, PreadFn pread_fn) {
    if (reinterpret_cast<uintptr_t>(dst) % kDirectAlign != 0 || s.offset % kDirectAlign != 0 ||
        s.len % kDirectAlign != 0) {
        GGML_ABORT("strata-moe: O_DIRECT read of %s: offset %" PRIu64 ", %zu bytes into %p, not all %zu-aligned", path,
                   s.offset, s.len, (void*) dst, kDirectAlign);
    }
    const size_t need = s.lead + len;
    size_t got = 0;
    while (got < s.len) {
        const ssize_t n = pread_fn(fd, dst + got, s.len - got, (off_t) (s.offset + got));
        if (n < 0) {
            if (errno == EINTR) continue;
            GGML_ABORT("strata-moe: O_DIRECT pread of %s at offset %" PRIu64 " (%zu bytes) failed: %s", path,
                       s.offset + got, s.len - got, std::strerror(errno));
        }
        if (n == 0) break;  // the end of the file
        got += (size_t) n;
        // a read that ends inside a block ends at the end of the file: the next offset could not be O_DIRECT
        if (got % kDirectAlign != 0) break;
    }
    if (got < need) {
        GGML_ABORT("strata-moe: short O_DIRECT read of %s: %zu of the %zu bytes wanted at offset %" PRIu64, path, got,
                   need, s.offset);
    }
}

MappedRange mapped_range(const void* p, size_t len, const char* what) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(p), last = first + len - 1;
    std::ifstream maps("/proc/self/maps");
    if (!maps) GGML_ABORT("strata-moe: cannot read /proc/self/maps to find the file of %s", what);
    MappedRange range;
    bool found_first = false, found_last = false;
    uint64_t last_offset = 0;
    std::string last_path;
    for (std::string line; std::getline(maps, line);) {
        unsigned long lo = 0, hi = 0, off = 0;
        char perms[8];
        int path_at = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx %7s %lx %*s %*s %n", &lo, &hi, perms, &off, &path_at) < 4) continue;
        const std::string path = path_at > 0 ? line.substr((size_t) path_at) : std::string();
        if (first >= lo && first < hi) {
            found_first = true;
            range.path = path;
            range.offset = off + (first - lo);
        }
        if (last >= lo && last < hi) {
            found_last = true;
            last_path = path;
            last_offset = off + (last - lo);
        }
    }
    if (!found_first || range.path.empty() || range.path[0] != '/') {
        GGML_ABORT("strata-moe: %s at %p is not in a file mapping (mapping '%s'): the cold tier reads the routed "
                   "experts from their GGUF files, so they must be mmapped (-lm mmap, not --no-mmap)",
                   what, p, range.path.c_str());
    }
    if (!found_last || last_path != range.path || last_offset != range.offset + len - 1) {
        GGML_ABORT("strata-moe: %s (%zu bytes at %p) is not one contiguous range of %s", what, len, p,
                   range.path.c_str());
    }
    return range;
}

Residency::Residency(const HotSet& hot, double ram_budget_gib, bool hot_in_vram, int n_readers,
                     int64_t round_experts, PreadFn pread_fn, RamMode ram_mode, const PinnedHost* pinned,
                     ColdPath cold_path)
    : hot_(hot), round_experts_(round_experts), pread_fn_(pread_fn), ram_mode_(ram_mode), cold_path_(cold_path) {
    if (ram_mode_ == RamMode::kPinned) {
        if (pinned == nullptr || pinned->alloc == nullptr || pinned->free == nullptr) {
            GGML_ABORT("strata-moe: STRATA_MOE_RAM_MODE=pinned without the CUDA runtime's pinned allocator");
        }
        pinned_ = *pinned;
    }
    if (hot_.ram_budget_gib < 0.0) {
        GGML_ABORT("strata-moe: STRATA_MOE_RAM_GB needs the warm lists of a hot set from hot_set.py "
                   "--ram-budget-gib; %s has none", hot_.path.c_str());
    }
    size_t max_slot = 0;
    int64_t n_vram_cold = 0;
    for (const HotLayer& h : hot_.layers) {
        Layer& l = layers_[h.layer];
        l.hot = &h;
        const struct {
            ggml_type type;
            int64_t n_per_row, n_rows;
            const char* role;
        } parts[3] = {{h.up_type, h.n_embd, h.n_ff, "up"}, {h.gate_type, h.n_embd, h.n_ff, "gate"},
                      {h.down_type, h.n_ff, h.n_embd, "down"}};
        for (int p = 0; p < 3; ++p) {
            if (parts[p].n_per_row % ggml_blck_size(parts[p].type) != 0) {
                GGML_ABORT("strata-moe: hot set %s, layer %d: %s rows of %lld are not whole %s blocks",
                           hot_.path.c_str(), h.layer, parts[p].role, (long long) parts[p].n_per_row,
                           ggml_type_name(parts[p].type));
            }
            l.part_bytes[p] = ggml_row_size(parts[p].type, parts[p].n_per_row) * (size_t) parts[p].n_rows;
            l.part_offset[p] = l.slot_bytes;
            l.slot_bytes += direct_slot_bytes(l.part_bytes[p]);
        }
        max_slot = std::max(max_slot, l.slot_bytes);
        l.cold.assign((size_t) h.n_expert, 1);
        l.warm_slot.assign((size_t) h.n_expert, -1);
        if (hot_in_vram) {
            for (const int32_t e : h.experts) l.cold[(size_t) e] = 0;
        } else {
            n_vram_cold += (int64_t) h.experts.size();
        }
        for (size_t k = 0; k < h.warm.size(); ++k) {
            l.warm_slot[(size_t) h.warm[k]] = (int32_t) k;
            l.cold[(size_t) h.warm[k]] = 0;
        }
        warm_size_ += h.warm.size() * l.slot_bytes;
        n_warm_ += (int64_t) h.warm.size();
        l.read = cold_path_ == ColdPath::kDirect ? l.cold : std::vector<uint8_t>(l.cold.size(), 0);
    }
    if ((double) warm_size_ > ram_budget_gib * kGiB) {
        GGML_ABORT("strata-moe: hot set %s: the warm tier needs %.3f GiB (%zu bytes, %lld experts), more than "
                   "STRATA_MOE_RAM_GB=%g", hot_.path.c_str(), (double) warm_size_ / kGiB, warm_size_,
                   (long long) n_warm_, ram_budget_gib);
    }
    if (warm_size_ > 0) alloc_warm();
    char* next = warm_buf_;
    for (auto& [index, l] : layers_) {
        l.warm = next;
        next += l.hot->warm.size() * l.slot_bytes;
        l.warm_rows.assign(l.hot->warm.size() * 3, nullptr);
    }
    // the readers fill the warm tier in both paths; only direct reads cold experts into staging
    if (cold_path_ == ColdPath::kDirect) {
        staging_stride_ = max_slot;
        staging_size_ = 2 * (size_t) round_experts_ * staging_stride_;
        staging_ = map_anonymous(staging_size_, "cold tier staging");
    }
    for (int i = 0; i < n_readers; ++i) readers_.emplace_back([this] { reader(); });

    char cold_text[160];
    if (cold_path_ == ColdPath::kDirect) {
        std::snprintf(cold_text, sizeof(cold_text), "read with O_DIRECT by %d threads in rounds of %lld experts "
                      "(staging %.1f MiB)", n_readers, (long long) round_experts_, (double) staging_size_ / kMiB);
    } else {
        std::snprintf(cold_text, sizeof(cold_text), "computed from the fork's mapping through the page cache, "
                      "which strata-moe leaves alone (warm fill by %d threads)", n_readers);
    }
    std::fprintf(stderr,
                 "strata-moe: residency from hot set %s: warm tier %lld experts in %.3f GiB of %s RAM "
                 "(STRATA_MOE_RAM_GB=%g, packed for %g GiB), every other expert not in VRAM cold "
                 "(STRATA_MOE_COLD=%s), %s\n",
                 hot_.path.c_str(), (long long) n_warm_, (double) warm_size_ / kGiB, ram_mode_name(ram_mode_),
                 ram_budget_gib, hot_.ram_budget_gib, cold_path_name(cold_path_), cold_text);
    if (n_vram_cold > 0) {
        std::fprintf(stderr, "strata-moe: no GPU tier: the %lld VRAM experts of %s are cold\n", (long long) n_vram_cold,
                     hot_.path.c_str());
    }
}

void Residency::alloc_warm() {
    const MemoryLimits limits = memory_limits();
    const auto fail = [&](const std::string& cause) {
        GGML_ABORT("%s", warm_alloc_error(ram_mode_, warm_size_, cause, limits).c_str());
    };
    // pinned and plain memory are charged to the memory cgroup up front, locked memory too: one that cannot hold
    // the warm tier at all is refused before a cgroup OOM kill could end the process without a message
    if (limits.cgroup_bytes > 0 && warm_size_ >= limits.cgroup_bytes) {
        fail("it does not fit under the memory cgroup's limit");
    }
    const int64_t t0 = now_ns();
    switch (ram_mode_) {
        case RamMode::kPinned: {
            // cudaHostAlloc guarantees less than kDirectAlign: one block more, and the buffer starts at the next one
            warm_alloc_size_ = warm_size_ + kDirectAlign;
            std::string err;
            warm_alloc_ = pinned_.alloc(warm_alloc_size_, &err);
            if (warm_alloc_ == nullptr) fail(err);
            warm_buf_ = reinterpret_cast<char*>(round_up(reinterpret_cast<uintptr_t>(warm_alloc_), kDirectAlign));
            break;
        }
        case RamMode::kLocked:
            warm_alloc_size_ = warm_size_;
            warm_alloc_ = warm_buf_ = map_anonymous(warm_size_, "warm tier");
            if (mlock(warm_buf_, warm_size_) != 0) fail(std::strerror(errno));
            break;
        case RamMode::kPlain:
            warm_alloc_size_ = warm_size_;
            warm_alloc_ = warm_buf_ = map_anonymous(warm_size_, "warm tier");
            // huge pages only make the faults and the TLB cheaper: a refusal (THP disabled) is logged, not fatal
            if (madvise(warm_buf_, warm_size_, MADV_HUGEPAGE) != 0) {
                std::fprintf(stderr, "strata-moe: MADV_HUGEPAGE on the plain warm tier refused (%s): 4 KiB pages\n",
                             std::strerror(errno));
            }
            std::fprintf(stderr, "strata-moe: pre-faulting the %.3f GiB plain warm tier (memory cgroup limit %s; an "
                         "OOM kill here means the cgroup cannot hold it)\n", (double) warm_size_ / kGiB,
                         limits.cgroup.c_str());
            if (madvise(warm_buf_, warm_size_, MADV_POPULATE_WRITE) != 0) {
                const int err = errno;
                fail(std::string("madvise(MADV_POPULATE_WRITE): ") + std::strerror(err) +
                     (err == EINVAL ? " (it needs Linux 5.14 or later)" : ""));
            }
            break;
    }
    std::fprintf(stderr, "strata-moe: STRATA_MOE_RAM_MODE=%s: the %.3f GiB warm tier allocated in %.0f ms "
                 "(RLIMIT_MEMLOCK %s, memory cgroup limit %s)\n", ram_mode_name(ram_mode_),
                 (double) warm_size_ / kGiB, (double) (now_ns() - t0) / 1e6, limits.memlock.c_str(),
                 limits.cgroup.c_str());
}

Residency::~Residency() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    queue_cv_.notify_all();
    for (std::thread& t : readers_) t.join();
    if (warm_alloc_ != nullptr) {
        if (ram_mode_ == RamMode::kPinned) {
            pinned_.free(warm_alloc_);
        } else {
            munmap(warm_alloc_, warm_alloc_size_);
        }
    }
    if (staging_ != nullptr) munmap(staging_, staging_size_);
    for (const auto& [path, fd] : fds_) close(fd);
}

const std::vector<uint8_t>& Residency::cold_map(int layer) const {
    const auto it = layers_.find(layer);
    if (it == layers_.end()) GGML_ABORT("strata-moe: hot set %s has no layer %d", hot_.path.c_str(), layer);
    return it->second.cold;
}

const char* const* Residency::warm_rows(int layer, int32_t expert) const {
    const auto it = layers_.find(layer);
    if (it == layers_.end()) GGML_ABORT("strata-moe: hot set %s has no layer %d", hot_.path.c_str(), layer);
    const Layer& l = it->second;
    if (expert < 0 || (size_t) expert >= l.warm_slot.size()) {
        GGML_ABORT("strata-moe: layer %d: expert %d is outside [0, %zu)", layer, expert, l.warm_slot.size());
    }
    const int32_t slot = l.warm_slot[(size_t) expert];
    return slot < 0 ? nullptr : &l.warm_rows[(size_t) slot * 3];
}

uint64_t Residency::read_expert(int layer, int32_t expert, char* dst, const size_t part_offset[3],
                                std::atomic<int64_t>& left, const char* rows[3]) {
    const auto it = layers_.find(layer);
    if (it == layers_.end()) GGML_ABORT("strata-moe: hot set %s has no layer %d", hot_.path.c_str(), layer);
    const Layer& l = it->second;
    if (!l.bound) {
        GGML_ABORT("strata-moe: layer %d: a direct prefill read before the layer's first call bound its GGUF files",
                   layer);
    }
    if (expert < 0 || (size_t) expert >= l.cold.size()) {
        GGML_ABORT("strata-moe: layer %d: expert %d is outside [0, %zu)", layer, expert, l.cold.size());
    }
    std::vector<Job> jobs(3);
    uint64_t bytes = 0;
    for (int p = 0; p < 3; ++p) {
        Job& j = jobs[(size_t) p];
        j.fd = l.fd[p];
        j.path = l.path[p];
        j.len = l.part_bytes[p];
        j.span = direct_span(l.file_offset[p] + (uint64_t) expert * l.part_bytes[p], j.len);
        j.dst = dst + part_offset[p];
        j.left = &left;
        rows[p] = j.dst + j.span.lead;
        bytes += j.span.len;
    }
    left.fetch_add(3, std::memory_order_relaxed);
    push(jobs);
    return bytes;
}

Residency::Layer& Residency::layer(int index, const char* tensor) {
    const auto it = layers_.find(index);
    if (it == layers_.end()) {
        GGML_ABORT("strata-moe: hot set %s has no layer %d (%s): STRATA_MOE_RAM_GB needs the tiers of every MoE "
                   "layer", hot_.path.c_str(), index, tensor);
    }
    return it->second;
}

int Residency::open_direct(const std::string& path) {
    const auto it = fds_.find(path);
    if (it != fds_.end()) return it->second;
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        GGML_ABORT("strata-moe: cannot open %s with O_DIRECT for the cold tier: %s", path.c_str(), std::strerror(errno));
    }
    fds_.emplace(path, fd);
    return fd;
}

void Residency::bind_layer(const ExpertTensors& t) {
    Layer& l = layer(t.layer, t.name[0]);
    if (l.bound) GGML_ABORT("strata-moe: layer %d (%s): its expert tensors are bound twice", t.layer, t.name[0]);
    bind_files(l, t);
    l.bound = true;
}

void Residency::bind_files(Layer& l, const ExpertTensors& t) {
    const auto t0 = std::chrono::steady_clock::now();
    const HotLayer& h = *l.hot;
    const ggml_type types[3] = {h.up_type, h.gate_type, h.down_type};
    for (int p = 0; p < 3; ++p) {
        if (t.type[p] != types[p] || t.n_expert != h.n_expert || t.expert_bytes[p] != l.part_bytes[p]) {
            GGML_ABORT("strata-moe: hot set %s, layer %d: %s is %s with %lld experts of %zu bytes, the hot set has %s, "
                       "%lld experts of %zu bytes", hot_.path.c_str(), h.layer, t.name[p], ggml_type_name(t.type[p]),
                       (long long) t.n_expert, t.expert_bytes[p], ggml_type_name(types[p]), (long long) h.n_expert,
                       l.part_bytes[p]);
        }
        const size_t bytes = l.part_bytes[p] * (size_t) h.n_expert;
        const MappedRange m = mapped_range(t.data[p], bytes, t.name[p]);
        l.fd[p] = open_direct(m.path);
        l.path[p] = fds_.find(m.path)->first.c_str();
        l.file_offset[p] = m.offset;

        // the offset found in /proc/self/maps gives the mapped bytes: the head of expert 0 against the mapping
        alignas(kDirectAlign) char head[2 * kDirectAlign];
        const size_t n = std::min(l.part_bytes[p], kDirectAlign);
        const DirectSpan s = direct_span(m.offset, n);
        read_direct(l.fd[p], l.path[p], s, n, head, pread_fn_);
        if (std::memcmp(head + s.lead, t.data[p], n) != 0) {
            GGML_ABORT("strata-moe: %s: the %zu bytes at offset %" PRIu64 " of %s differ from the mapped tensor",
                       t.name[p], n, m.offset, l.path[p]);
        }
    }

    // the warm slots, all parts of all warm experts at once over the reader threads
    std::atomic<int64_t> left{0};
    std::vector<Job> jobs;
    for (size_t k = 0; k < h.warm.size(); ++k) {
        for (int p = 0; p < 3; ++p) {
            Job j;
            j.fd = l.fd[p];
            j.path = l.path[p];
            j.len = l.part_bytes[p];
            j.span = direct_span(l.file_offset[p] + (uint64_t) h.warm[k] * l.part_bytes[p], j.len);
            j.dst = l.warm + k * l.slot_bytes + l.part_offset[p];
            j.left = &left;
            l.warm_rows[k * 3 + (size_t) p] = j.dst + j.span.lead;
            jobs.push_back(j);
        }
    }
    left.store((int64_t) jobs.size());
    push(jobs);
    while (left.load(std::memory_order_acquire) != 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
    const double fill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // direct: the fork's pages of the layer's routed experts, unmapped, then out of the page cache
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    size_t dropped = 0;
    for (int p = 0; cold_path_ == ColdPath::kDirect && p < 3; ++p) {
        const size_t bytes = l.part_bytes[p] * (size_t) h.n_expert;
        const uintptr_t lo = round_up(reinterpret_cast<uintptr_t>(t.data[p]), page);
        const uintptr_t hi = (reinterpret_cast<uintptr_t>(t.data[p]) + bytes) / page * page;
        if (hi > lo && madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED) != 0) {
            GGML_ABORT("strata-moe: madvise(MADV_DONTNEED) of %s (%zu bytes) failed: %s", t.name[p], (size_t) (hi - lo),
                       std::strerror(errno));
        }
        const int err = posix_fadvise(l.fd[p], (off_t) l.file_offset[p], (off_t) bytes, POSIX_FADV_DONTNEED);
        if (err != 0) {
            GGML_ABORT("strata-moe: posix_fadvise(POSIX_FADV_DONTNEED) of %s in %s failed: %s", t.name[p], l.path[p],
                       std::strerror(err));
        }
        dropped += bytes;
    }

    const size_t warm_bytes = h.warm.size() * (l.part_bytes[0] + l.part_bytes[1] + l.part_bytes[2]);
    int64_t n_cold = 0;
    for (const uint8_t x : l.cold) n_cold += x;
    char pages[96];
    if (cold_path_ == ColdPath::kDirect) {
        std::snprintf(pages, sizeof(pages), "the fork's pages of its %.1f MiB of routed experts dropped",
                      (double) dropped / kMiB);
    } else {
        std::snprintf(pages, sizeof(pages), "the fork's pages left to the page cache");
    }
    std::fprintf(stderr,
                 "strata-moe: layer %d: %zu warm experts (%.1f MiB) read into the warm tier in %.0f ms (%.2f GB/s), "
                 "%lld cold; %s\n",
                 h.layer, h.warm.size(), (double) warm_bytes / kMiB, fill_ms,
                 fill_ms > 0.0 ? (double) warm_bytes / fill_ms / 1e6 : 0.0, (long long) n_cold, pages);
}

void Residency::push(const std::vector<Job>& jobs) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (const Job& j : jobs) queue_.push_back(j);
    }
    queue_cv_.notify_all();
}

void Residency::reader() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;
            job = queue_.front();
            queue_.pop_front();
        }
        read_direct(job.fd, job.path, job.span, job.len, job.dst, pread_fn_);
        if (job.ready != nullptr) job.ready->store(1, std::memory_order_release);
        if (job.end_ns != nullptr) {
            const int64_t t = now_ns();
            int64_t seen = job.end_ns->load(std::memory_order_relaxed);
            while (seen < t && !job.end_ns->compare_exchange_weak(seen, t, std::memory_order_relaxed)) {}
        }
        job.left->fetch_sub(1, std::memory_order_acq_rel);
    }
}

}  // namespace strata::dsv41
