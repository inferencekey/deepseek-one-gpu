// src/dsv41/moe_residency.h - where strata-moe keeps every routed expert the GPU tier does not, with
// STRATA_MOE_RAM_GB set (src/dsv41/moe.cpp). Three tiers per (layer, expert), from the hot-set file:
//   VRAM  the hot set (moe_gpu.h), computed by the GPU tier;
//   warm  the layer's warm list, in one RAM buffer (CUDA pinned, mlock'd or plain, STRATA_MOE_RAM_MODE) filled at
//         the layer's first call;
//   cold  every other expert, by STRATA_MOE_COLD (ColdPath): computed straight from the fork's mmapped tensors
//         through the page cache (pagecache, the default), or read during each call that routes to it from
//         the GGUF shard with O_DIRECT pread into reusable staging, never through the page cache (direct).
// Warm and cold experts are computed by the CPU kernel of the call, from the warm buffer, the mapping or staging.
// With direct, after a layer's first call the fork's mmap pages of its routed experts are dropped and never touched
// again, so the page cache keeps only what the rest of the model reads; pagecache leaves them alone
// (moe_residency.cpp has the why of each choice).
#pragma once

#include "moe_hot_set.h"
#include "moe_op.h"

#include <sys/types.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::dsv41 {

// The block size every O_DIRECT offset, length and buffer address is a multiple of: the logical block size of any
// NVMe or SATA disk is 512 or 4096 bytes, and 4096 also matches the page size of the buffers.
constexpr size_t kDirectAlign = 4096;

// The bytes [offset, offset + len) of a file as one O_DIRECT read: whole blocks around them, starting `lead`
// bytes before them.
struct DirectSpan {
    uint64_t offset = 0;
    size_t len = 0;
    size_t lead = 0;
};
DirectSpan direct_span(uint64_t offset, size_t len);

// The buffer a DirectSpan of `len` bytes needs at any offset: round_up(len, 4096) + 4096.
size_t direct_slot_bytes(size_t len);

using PreadFn = ssize_t (*)(int fd, void* buf, size_t n, off_t offset);

// Reads span `s` of `fd` (opened with O_DIRECT) into dst (4096-aligned, s.len bytes) with pread_fn: retries on
// EINTR and continues after a short read; the read may stop early only at the end of the file, once the
// s.lead + len wanted bytes are in. Aborts naming the file, the offset and errno on anything else.
void read_direct(int fd, const char* path, const DirectSpan& s, size_t len, char* dst, PreadFn pread_fn);

// The file and the offset in it that the `len` bytes at p are mapped from (/proc/self/maps). Aborts naming `what`
// when they are not one contiguous range of one file mapping.
struct MappedRange {
    std::string path;
    uint64_t offset = 0;
};
MappedRange mapped_range(const void* p, size_t len, const char* what);

// How the warm buffer holds its RAM (STRATA_MOE_RAM_MODE):
//   pinned  CUDA pinned host memory (cudaHostAlloc): the driver pins the pages, so RLIMIT_MEMLOCK does not apply (the
//           RunPod containers have 8 MiB of it and no CAP_IPC_LOCK); the default with a CUDA device
//   locked  anonymous memory locked with mlock, within RLIMIT_MEMLOCK; the default without a CUDA device
//   plain   anonymous memory with MADV_HUGEPAGE, pre-faulted (MADV_POPULATE_WRITE), not locked: it stays resident
//           while nothing swaps it out or the memory cgroup does not OOM (the pods have no swap); only when set
enum class RamMode { kPinned, kLocked, kPlain };
const char* ram_mode_name(RamMode m);

// The mode STRATA_MOE_RAM_MODE=`env` selects (null or empty: the default for cuda_device), or, in `error`, why it
// cannot: an unknown value, or pinned without a CUDA device.
struct RamModeChoice {
    RamMode mode = RamMode::kLocked;
    bool is_default = false;
    std::string error;
};
RamModeChoice choose_ram_mode(const char* env, bool cuda_device);

// How the cold experts reach the CPU (STRATA_MOE_COLD):
//   pagecache  from the fork's mmapped tensors, through the page cache, whose pages strata-moe neither drops nor
//              advises; the default: on an A100 SXM pod (116.4 GiB cgroup, 2026-10-06) the page cache served
//              hook-70's misses in about 59 ms per decode pass, while direct read 768.6 MiB per token at 4.75 GB/s
//              and waited 168 ms of a 204 ms pass
//   direct     O_DIRECT reads into staging, the fork's pages of the routed experts dropped
enum class ColdPath { kPageCache, kDirect };
const char* cold_path_name(ColdPath p);

// The path STRATA_MOE_COLD=`env` selects (null or empty: pagecache, the default), or, in `error`, why it cannot.
struct ColdPathChoice {
    ColdPath path = ColdPath::kPageCache;
    bool is_default = false;
    std::string error;
};
ColdPathChoice choose_cold_path(const char* env);

// The CUDA runtime's pinned host allocator, which this GPU-free object library does not link: moe.cpp passes
// cudaHostAlloc and cudaFreeHost. alloc returns null and sets *error on failure.
struct PinnedHost {
    void* (*alloc)(size_t bytes, std::string* error) = nullptr;
    void (*free)(void* p) = nullptr;
};

// The memory limits a warm-tier allocation can hit, as text: RLIMIT_MEMLOCK and the memory cgroup's limit (v2
// memory.max or v1 memory.limit_in_bytes, "unknown" when neither is readable).
struct MemoryLimits {
    std::string memlock;
    std::string cgroup;
    uint64_t cgroup_bytes = 0;  // the cgroup's limit; 0 when it has none or it is unknown
};
MemoryLimits memory_limits();

// The abort message of a failed warm-tier allocation of `bytes` in mode m: the size, `cause` and the limit that
// applies to that mode.
std::string warm_alloc_error(RamMode m, size_t bytes, const std::string& cause, const MemoryLimits& limits);

// One MoE layer's routed expert tensors as plain pointers: per part (0 up, 1 gate, 2 down) the mapped bytes of
// expert 0 in its GGUF file, its type and the bytes of one expert (expert e at data + e * expert_bytes). strata-moe's
// calls make it from their ggml tensors (prepare); the dedicated engine (src/dsv41/engine/experts.cpp) from its own
// mapping of the GGUF, so both fill the warm tier through the same code.
struct ExpertTensors {
    int layer = -1;
    int64_t n_expert = 0;
    const char* name[3] = {nullptr, nullptr, nullptr};
    const void* data[3] = {nullptr, nullptr, nullptr};
    ggml_type type[3] = {GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT};
    size_t expert_bytes[3] = {0, 0, 0};
};

// The tier map, the warm buffer and the cold reader. Thread safety: prepare and bind run under the caller's call
// lock (moe_ref.cpp, join); issue and finish bracket the use of the staging by one call at a time. The methods that
// take a Call are in moe_residency_call.cpp, the rest (no ggml tensor, no Call) in moe_residency.cpp.
class Residency {
public:
    // The tiers of every layer of `hot`: its VRAM experts stay out of this map when hot_in_vram (the GPU tier
    // has them) and are cold otherwise. Allocates the warm buffer in ram_mode (pinned through `pinned`, which that
    // mode needs; aborts when it needs more than ram_budget_gib, or when the allocation fails, with
    // warm_alloc_error), n_readers reader threads, which read with pread_fn, and with ColdPath::kDirect the staging
    // of two rounds of round_experts cold experts.
    Residency(const HotSet& hot, double ram_budget_gib, bool hot_in_vram, int n_readers, int64_t round_experts,
              PreadFn pread_fn, RamMode ram_mode, const PinnedHost* pinned, ColdPath cold_path);
    ~Residency();
    Residency(const Residency&) = delete;
    Residency& operator=(const Residency&) = delete;

    // Per call, before the misses are grouped: per expert of the call's layer, 1 when the call must read it from
    // its file (a cold expert with ColdPath::kDirect; none with kPageCache). At the layer's first call it binds the
    // layer (bind_layer) from the call's expert tensors.
    const std::vector<uint8_t>& prepare(const Call& c);
    // Once per layer, before its warm rows are read: checks the expert tensors against the hot set, finds their GGUF
    // files, fills the layer's warm slots with O_DIRECT reads and, with kDirect, drops the mapping's pages of its
    // experts. Aborts naming the layer when it is bound twice or the hot set has none.
    void bind_layer(const ExpertTensors& t);
    // Per call, once the misses are grouped with the ready (warm, and with kPageCache mapped cold) experts first:
    // the base row of each expert, the rounds and the staging slot of each cold expert read from its file.
    void bind(Call& c);
    // The call's first thread, before the CPU phases: takes the staging and queues the reads of rounds 0 and 1.
    void issue(Call& c);
    // Blocks until part (0 up, 1 gate, 2 down) of the call's k-th expert is readable.
    void wait(Call& c, int64_t k, int part);
    // One thread, after round r: its staging slots are free, the reads of round r + 2 go into them.
    void round_done(Call& c, int64_t r);
    // The thread that issued, after the CPU phases: waits for the reader threads to let go of the call and
    // releases the staging.
    void finish(Call& c);

    // the warm tier's experts over all layers
    int64_t warm_experts() const { return n_warm_; }
    // per expert of `layer`: 1 when it is cold (in neither VRAM nor the warm tier), whatever the cold path; aborts
    // naming the layer when the hot set has none
    const std::vector<uint8_t>& cold_map(int layer) const;
    ColdPath cold_path() const { return cold_path_; }
    RamMode ram_mode() const { return ram_mode_; }
    // the first rows of `expert`'s up, gate and down parts in the warm tier of `layer` ([3], null until the
    // layer's first call has filled its tier), or null when the expert is not warm; aborts naming the layer when the
    // hot set has none
    const char* const* warm_rows(int layer, int32_t expert) const;
    // STRATA_MOE_PREFILL_COLD=direct: queues the O_DIRECT reads of the up, gate and down parts of `expert` of
    // `layer` from its GGUF files into dst + part_offset[part] (dst kDirectAlign-aligned, each part with
    // direct_slot_bytes of room: moe_prefill.h, StageLayout) on the reader threads, whatever the cold path. Adds 3
    // to `left` first, and each read takes 1 off once its bytes are in; rows[part] gets where the part's first row
    // will be. Returns the bytes read. Aborts naming the layer before the layer's first call has bound its files
    // (prepare), or on an expert outside the layer.
    uint64_t read_expert(int layer, int32_t expert, char* dst, const size_t part_offset[3], std::atomic<int64_t>& left,
                         const char* rows[3]);

private:
    struct Layer {
        const HotLayer* hot = nullptr;
        std::vector<uint8_t> cold;       // per expert
        std::vector<uint8_t> read;       // per expert: read from its file per call (cold, with kDirect)
        std::vector<int32_t> warm_slot;  // per expert: its warm slot, or -1
        size_t part_bytes[3] = {};       // one expert's up, gate, down bytes
        size_t part_offset[3] = {};      // where each part starts in a slot
        size_t slot_bytes = 0;
        char* warm = nullptr;            // the layer's warm slots
        std::vector<const char*> warm_rows;  // [warm slot, part]: the expert's first row
        bool bound = false;
        int fd[3] = {-1, -1, -1};
        const char* path[3] = {};
        uint64_t file_offset[3] = {};    // where the tensor's expert 0 starts in its file
    };
    struct Job {
        int fd = -1;
        const char* path = nullptr;
        DirectSpan span;
        size_t len = 0;
        char* dst = nullptr;
        std::atomic<uint8_t>* ready = nullptr;  // set once the bytes are in
        std::atomic<int64_t>* left = nullptr;   // decremented last: after it the job touches nothing of its owner
        std::atomic<int64_t>* end_ns = nullptr; // the latest completion, steady clock ns
    };

    Layer& layer(int index, const char* tensor);
    void bind_files(Layer& l, const ExpertTensors& t);
    int open_direct(const std::string& path);
    void queue_round(Call& c, int64_t r);
    void push(const std::vector<Job>& jobs);
    void reader();

    HotSet hot_;
    std::map<int, Layer> layers_;
    int64_t round_experts_ = 0;
    PreadFn pread_fn_ = nullptr;

    void alloc_warm();

    RamMode ram_mode_ = RamMode::kLocked;
    ColdPath cold_path_ = ColdPath::kPageCache;
    PinnedHost pinned_;
    void* warm_alloc_ = nullptr;  // what was allocated: warm_buf_ is it rounded up to kDirectAlign
    size_t warm_alloc_size_ = 0;
    char* warm_buf_ = nullptr;
    size_t warm_size_ = 0;
    int64_t n_warm_ = 0;
    char* staging_ = nullptr;
    size_t staging_size_ = 0, staging_stride_ = 0;
    std::map<std::string, int> fds_;

    // one call at a time uses the staging, from issue to finish
    std::mutex staging_mutex_;
    std::unique_lock<std::mutex> busy_{staging_mutex_, std::defer_lock};

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<Job> queue_;
    bool stop_ = false;
    std::vector<std::thread> readers_;
};

}  // namespace strata::dsv41
