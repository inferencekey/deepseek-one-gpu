// src/dsv41/moe_gpu.cpp - the GPU tier of strata-moe (moe_gpu.h).
//
// strata-moe creates its own ggml CUDA backend on device 0 from the fork's libggml-cuda (the same library the
// fork's process already loaded, so one ggml) and one VRAM buffer holding a slab per layer: the up, gate and down
// rows of the layer's hot experts, [n_embd, n_ff, n_hot] and [n_ff, n_embd, n_hot] in the fork's own types,
// copied from the fork's host expert tensors at the layer's first call. Per call the hit path uploads the hit
// tokens' inputs, slots and weights through pinned staging memory, runs a small graph on that backend (the
// stock expert path on the slab: mul_mat_id up and gate, ggml_swiglu_clamp or ggml_swiglu_split, mul_mat_id
// down, times the weights, all ggml-cuda kernels) and downloads y * w per hit slot; the combine on the CPU adds
// them in slot order (moe_op.h). One ggml_backend_synchronize per call.
//
// The hit layout keeps the stock decode shape: one token with h hits is a [n_embd, 1, 1] input with ids [h, 1],
// as the stock graph's [n_embd, 1, 1] with ids [6, 1], so a decode step takes the same mmvq kernels.
//
// the prefill path (moe_prefill.h): a batch call of at least STRATA_MOE_PREFILL_MIN tokens runs every pair
// here. The call's whole input goes up once; each pass is a graph that gathers its pairs' input rows (get_rows),
// runs the same expert path on the slab or on one ring set ([n_embd, 1, n_pairs] with ids [1, n_pairs]: every pair
// a column, no padding) and downloads y * w per pair. The ring is one VRAM buffer per part (up, gate, down), laid
// out per call for the call's layer types, so a set is a contiguous [n_per_row, n_rows, set_size] tensor; the
// copies run on a CUDA stream of strata-moe's own (cudaMemcpyAsync from the pinned warm tier or the pinned
// staging), the passes on the backend's stream, and the two meet through CUDA events (ggml_backend_event_wait and
// record on the backend's side). The ring is a COMPUTE buffer, zeroed once: ggml-cuda then never memsets the
// padding after a quantized tensor placed in it (which here is the next set's first expert), and MMQ's reads past
// a set's last row always land on finite weights or zeros.
#include "moe_gpu.h"

#include "moe_residency.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
// struct ggml_backend_event: the hit path's timing events are CUDA events with timing on, recorded on the
// backend's stream through ggml_backend_event_record (ggml-cuda's own events are created with timing off)
#include "ggml-backend-impl.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>

namespace strata::dsv41 {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) GGML_ABORT("strata-moe: %s: %s", what, cudaGetErrorString(err));
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

float elapsed_ms(const ggml_backend_event& from, const ggml_backend_event& to) {
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(from.context), static_cast<cudaEvent_t>(to.context)),
               "cudaEventElapsedTime");
    return ms;
}

// the staging copies of one job are handed out in chunks of this size: an expert part (1.3 to 4.2 MiB at
// DeepSeek-V4.1's q2_K / q3_K) splits over several threads, each faulting its share of the fork's mapping in
constexpr size_t kStageChunk = 1 << 20;

class GpuTier final : public HitPath {
public:
    GpuTier(const HotSet& hot, double budget_gib, const PrefillConfig& prefill);

    const std::vector<int32_t>* slots(const Call& c) override;
    bool prefill(Call& c, const Residency* residency) override;

    void submit(Call& c) override;
    const float* wait(Call& c) override;

private:
    struct Slab {
        const HotLayer* hot = nullptr;
        ggml_tensor *up = nullptr, *gate = nullptr, *down = nullptr;
        std::vector<int32_t> slot_of;  // per expert: its slot, or -1
        bool filled = false;
    };
    class Prefill;  // the PrefillEngine of one call

    void fill(Slab& s, const Call& c);
    char* staging(size_t bytes);
    void alloc_prefill(size_t free_after_slab);
    void submit_prefill(Call& c);
    ggml_backend_event* timing_event(size_t i);
    // slot k of staging set h, and its part `part` (moe_prefill.h, StageLayout)
    char* stage_base(int h, size_t k) const {
        return stage_ + ((size_t) h * (size_t) prefill_.set_size + k) * stage_slot_;
    }
    char* stage_slot(int h, size_t k, int part) const { return stage_base(h, k) + stage_part_offset_[part]; }

    HotSet hot_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_type_t buft_ = nullptr;
    ggml_context* slab_ctx_ = nullptr;
    ggml_backend_buffer_t slab_buf_ = nullptr;
    std::vector<Slab> slabs_;  // by layer
    ggml_gallocr_t galloc_ = nullptr;
    ggml_backend_buffer_t staging_ = nullptr;
    ggml_context* graph_ctx_ = nullptr;  // the last call's graph, kept until its work is done
    float* out_host_ = nullptr;

    std::set<int> cpu_only_layers_;  // the layers without hot experts that have been logged

    // held from submit to wait: one call at a time uses the backend, the graph and the staging memory
    std::mutex mutex_;
    std::unique_lock<std::mutex> busy_{mutex_, std::defer_lock};
    ggml_backend_event start_{}, end_{};

    // the prefill path (prefill_.min_tokens 0: none of it is allocated)
    PrefillConfig prefill_;
    size_t part_cap_[3] = {};          // the ring's bytes per slot per part (up, gate, down), the hot set's largest
    ggml_backend_buffer_t ring_buf_ = nullptr;
    char* ring_[3] = {};               // each part's region
    ggml_backend_buffer_t stage_buf_ = nullptr;  // pinned
    char* stage_ = nullptr;
    size_t stage_slot_ = 0, stage_part_offset_[3] = {};
    ggml_backend_buffer_t x_buf_ = nullptr;  // the call's input [n_embd, n_tokens] f32
    cudaStream_t copy_stream_ = nullptr;
    std::vector<ggml_backend_event> copied_, consumed_;  // per ring set: its copy done, its last pass done
    std::vector<cudaEvent_t> stage_free_;                // per staging set: the copy out of it done
    std::vector<ggml_backend_event> timing_;  // STRATA_MOE_PROFILE=1: [pass][copy start, copy end, pass start, end]
    ggml_backend_event prefill_start_{};
    bool prefill_logged_ = false;
};

GpuTier::GpuTier(const HotSet& hot, double budget_gib, const PrefillConfig& prefill) : hot_(hot), prefill_(prefill) {
    backend_ = ggml_backend_cuda_init(0);
    if (backend_ == nullptr) GGML_ABORT("strata-moe: ggml_backend_cuda_init(0) failed");
    buft_ = ggml_backend_get_default_buffer_type(backend_);
    char desc[256];
    ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_cuda_get_device_memory(0, &free_bytes, &total_bytes);

    int max_layer = -1;
    for (const HotLayer& l : hot_.layers) max_layer = std::max(max_layer, l.layer);
    slabs_.resize((size_t) (max_layer + 1));
    const ggml_init_params params = {/*.mem_size =*/ (3 * hot_.layers.size() + 1) * ggml_tensor_overhead(),
                                     /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true};
    slab_ctx_ = ggml_init(params);
    int64_t n_hot = 0, n_hot_layers = 0;
    for (const HotLayer& l : hot_.layers) {
        if (l.experts.empty()) continue;
        ++n_hot_layers;
        Slab& s = slabs_[(size_t) l.layer];
        s.hot = &l;
        const int64_t n = (int64_t) l.experts.size();
        const struct {
            ggml_type type;
            int64_t n_per_row;
            const char* role;
        } rows[] = {{l.up_type, l.n_embd, "up"}, {l.gate_type, l.n_embd, "gate"}, {l.down_type, l.n_ff, "down"}};
        for (const auto& r : rows) {
            if (r.n_per_row % ggml_blck_size(r.type) != 0) {
                GGML_ABORT("strata-moe: hot set %s, layer %d: %s rows of %lld are not whole %s blocks",
                           hot_.path.c_str(), l.layer, r.role, (long long) r.n_per_row, ggml_type_name(r.type));
            }
        }
        s.up = ggml_new_tensor_3d(slab_ctx_, l.up_type, l.n_embd, l.n_ff, n);
        s.gate = ggml_new_tensor_3d(slab_ctx_, l.gate_type, l.n_embd, l.n_ff, n);
        s.down = ggml_new_tensor_3d(slab_ctx_, l.down_type, l.n_ff, l.n_embd, n);
        ggml_format_name(s.up, "strata-moe.blk.%d.ffn_up_hot", l.layer);
        ggml_format_name(s.gate, "strata-moe.blk.%d.ffn_gate_hot", l.layer);
        ggml_format_name(s.down, "strata-moe.blk.%d.ffn_down_hot", l.layer);
        s.slot_of.assign((size_t) l.n_expert, -1);
        for (size_t k = 0; k < l.experts.size(); ++k) s.slot_of[(size_t) l.experts[k]] = (int32_t) k;
        n_hot += n;
    }

    if (n_hot == 0) GGML_ABORT("strata-moe: hot set %s lists no expert in any layer", hot_.path.c_str());
    const size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(slab_ctx_, buft_);
    if ((double) need > budget_gib * kGiB) {
        GGML_ABORT("strata-moe: hot set %s needs %.3f GiB of VRAM (%zu bytes), more than STRATA_MOE_VRAM_GB=%g",
                   hot_.path.c_str(), (double) need / kGiB, need, budget_gib);
    }
    if (need > free_bytes) {
        GGML_ABORT("strata-moe: hot set %s needs %.3f GiB of VRAM, CUDA0 (%s) has %.3f GiB free", hot_.path.c_str(),
                   (double) need / kGiB, desc, (double) free_bytes / kGiB);
    }
    slab_buf_ = ggml_backend_alloc_ctx_tensors_from_buft(slab_ctx_, buft_);
    if (slab_buf_ == nullptr) {
        GGML_ABORT("strata-moe: cannot allocate the %.3f GiB VRAM slab of hot set %s", (double) need / kGiB,
                   hot_.path.c_str());
    }
    galloc_ = ggml_gallocr_new(buft_);

    cuda_check(cudaSetDevice(0), "cudaSetDevice(0)");
    cudaEvent_t start = nullptr, end = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate");
    cuda_check(cudaEventCreate(&end), "cudaEventCreate");
    start_ = {ggml_backend_get_device(backend_), start};
    end_ = {ggml_backend_get_device(backend_), end};

    std::fprintf(stderr,
                 "strata-moe: GPU tier on CUDA0 (%s): hot set %s, %lld experts in %lld of %zu layers, slab %.3f "
                 "GiB (STRATA_MOE_VRAM_GB=%g, packed for %g GiB; %.3f of %.3f GiB free before it)\n",
                 desc, hot_.path.c_str(), (long long) n_hot, (long long) n_hot_layers, hot_.layers.size(),
                 (double) need / kGiB, budget_gib, hot_.budget_gib, (double) free_bytes / kGiB,
                 (double) total_bytes / kGiB);
    if (prefill_.min_tokens > 0) alloc_prefill(free_bytes - need);
}

// The ring, its events, the copy stream and the pinned staging, sized by the hot set's largest expert parts.
void GpuTier::alloc_prefill(size_t free_after_slab) {
    for (const HotLayer& l : hot_.layers) {
        const size_t parts[3] = {ggml_row_size(l.up_type, l.n_embd) * (size_t) l.n_ff,
                                 ggml_row_size(l.gate_type, l.n_embd) * (size_t) l.n_ff,
                                 ggml_row_size(l.down_type, l.n_ff) * (size_t) l.n_embd};
        for (int p = 0; p < 3; ++p) part_cap_[p] = std::max(part_cap_[p], parts[p]);
    }
    // per part region: every slot, then the padding ggml-cuda allots after a quantized tensor (MATRIX_ROW_PADDING,
    // 512 elements of a row; one row of the largest part bounds it), 256-byte aligned
    const size_t n_slots = (size_t) prefill_.n_sets * (size_t) prefill_.set_size;
    size_t region[3], ring_bytes = 0;
    for (int p = 0; p < 3; ++p) {
        size_t row_bytes = 0;
        for (const HotLayer& l : hot_.layers) {
            const ggml_type type = p == 0 ? l.up_type : p == 1 ? l.gate_type : l.down_type;
            row_bytes = std::max(row_bytes, ggml_row_size(type, 512));
        }
        region[p] = GGML_PAD(n_slots * part_cap_[p] + row_bytes, 256);
        ring_bytes += region[p];
    }
    if (ring_bytes > free_after_slab) {
        GGML_ABORT("strata-moe: the prefill ring (STRATA_MOE_PREFILL_SETS=%d x STRATA_MOE_PREFILL_GROUP=%d experts) "
                   "needs %.3f GiB of VRAM, CUDA0 has %.3f GiB free after the slab; lower them, STRATA_MOE_VRAM_GB, "
                   "or set STRATA_MOE_PREFILL_MIN=0", prefill_.n_sets, prefill_.set_size, (double) ring_bytes / kGiB,
                   (double) free_after_slab / kGiB);
    }
    ring_buf_ = ggml_backend_buft_alloc_buffer(buft_, ring_bytes);
    if (ring_buf_ == nullptr) {
        GGML_ABORT("strata-moe: cannot allocate the %.3f GiB prefill ring", (double) ring_bytes / kGiB);
    }
    ggml_backend_buffer_set_usage(ring_buf_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_clear(ring_buf_, 0);
    char* base = static_cast<char*>(ggml_backend_buffer_get_base(ring_buf_));
    for (int p = 0; p < 3; ++p) {
        ring_[p] = base;
        base += region[p];
    }

    const StageLayout layout = stage_layout(part_cap_);
    for (int p = 0; p < 3; ++p) stage_part_offset_[p] = layout.part_offset[p];
    stage_slot_ = layout.slot_bytes;
    const size_t stage_bytes = (size_t) prefill_.n_stage_sets * (size_t) prefill_.set_size * stage_slot_;
    // one block more: the layout's O_DIRECT reads need a kDirectAlign-aligned base, more than the buffer promises
    stage_buf_ = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), stage_bytes + kDirectAlign);
    if (stage_buf_ == nullptr) {
        GGML_ABORT("strata-moe: cannot allocate %.3f GiB of pinned prefill staging (STRATA_MOE_PREFILL_STAGE=%d sets "
                   "of %d experts)", (double) stage_bytes / kGiB, prefill_.n_stage_sets, prefill_.set_size);
    }
    stage_ = reinterpret_cast<char*>(
        GGML_PAD(reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(stage_buf_)), kDirectAlign));

    cuda_check(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
    const auto sync_event = [dev] {
        cudaEvent_t e = nullptr;
        cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreateWithFlags");
        return ggml_backend_event{dev, e};
    };
    for (int s = 0; s < prefill_.n_sets; ++s) {
        copied_.push_back(sync_event());
        consumed_.push_back(sync_event());
    }
    for (int h = 0; h < prefill_.n_stage_sets; ++h) {
        stage_free_.push_back(static_cast<cudaEvent_t>(sync_event().context));
    }
    cudaEvent_t start = nullptr;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate");
    prefill_start_ = {dev, start};

    std::fprintf(stderr,
                 "strata-moe: prefill path: batch calls of >= %lld tokens (STRATA_MOE_PREFILL_MIN) run every pair on "
                 "CUDA0, the experts outside the slab streamed through a ring of %d sets x %d experts "
                 "(STRATA_MOE_PREFILL_SETS, STRATA_MOE_PREFILL_GROUP: %.1f MiB of VRAM beside the slab) and the staged "
                 "ones through %d sets of pinned staging (STRATA_MOE_PREFILL_STAGE: %.1f MiB)\n",
                 (long long) prefill_.min_tokens, prefill_.n_sets, prefill_.set_size,
                 (double) ring_bytes / (1024.0 * 1024.0), prefill_.n_stage_sets,
                 (double) stage_bytes / (1024.0 * 1024.0));
    std::fprintf(stderr, "strata-moe: STRATA_MOE_PREFILL_COLD=%s%s: the staged experts %s\n",
                 prefill_cold_name(prefill_.cold), prefill_.cold_is_default ? " (the default)" : "",
                 prefill_.cold == PrefillCold::kDirect
                     ? "read from their GGUF files with O_DIRECT by the residency's reader threads"
                     : "copied from the fork's mapping, through the page cache, by the op's CPU threads");
}

// The profile's timing event i (created with timing on, as start_ and end_), made on first use.
ggml_backend_event* GpuTier::timing_event(size_t i) {
    while (timing_.size() <= i) {
        cudaEvent_t e = nullptr;
        cuda_check(cudaEventCreate(&e), "cudaEventCreate");
        timing_.push_back({ggml_backend_get_device(backend_), e});
    }
    return &timing_[i];
}

// The layer's hot experts from the fork's host tensors into its slab, once the call shows they match the hot set.
void GpuTier::fill(Slab& s, const Call& c) {
    const auto t0 = std::chrono::steady_clock::now();
    const struct {
        const ggml_tensor* host;
        ggml_tensor* slab;
        const char* role;
    } parts[] = {{c.up, s.up, "up"}, {c.gate, s.gate, "gate"}, {c.down, s.down, "down"}};
    for (const auto& [host, slab, role] : parts) {
        if (host->type != slab->type || host->ne[0] != slab->ne[0] || host->ne[1] != slab->ne[1] ||
            host->ne[2] != s.hot->n_expert || host->nb[2] != slab->nb[2]) {
            GGML_ABORT("strata-moe: hot set %s, layer %d: %s is %s [%lld, %lld, %lld], the hot set has %s %s "
                       "[%lld, %lld, %lld]",
                       hot_.path.c_str(), c.layer, host->name, ggml_type_name(host->type), (long long) host->ne[0],
                       (long long) host->ne[1], (long long) host->ne[2], role, ggml_type_name(slab->type),
                       (long long) slab->ne[0], (long long) slab->ne[1], (long long) s.hot->n_expert);
        }
        for (size_t k = 0; k < s.hot->experts.size(); ++k) {
            const char* src = static_cast<const char*>(host->data) + (size_t) s.hot->experts[k] * host->nb[2];
            ggml_backend_tensor_set(slab, src, k * slab->nb[2], slab->nb[2]);
        }
    }
    const double bytes = (double) (ggml_nbytes(s.up) + ggml_nbytes(s.gate) + ggml_nbytes(s.down));
    std::fprintf(stderr, "strata-moe: layer %d: %zu hot experts (%.1f MiB) copied to VRAM in %.0f ms\n", c.layer,
                 s.hot->experts.size(), bytes / (1024.0 * 1024.0),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

// Pinned host memory of at least `bytes`, so the uploads and the download run asynchronously.
char* GpuTier::staging(size_t bytes) {
    if (staging_ == nullptr || ggml_backend_buffer_get_size(staging_) < bytes) {
        if (staging_ != nullptr) ggml_backend_buffer_free(staging_);
        staging_ = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), bytes);
        if (staging_ == nullptr) GGML_ABORT("strata-moe: cannot allocate %zu bytes of pinned staging memory", bytes);
    }
    return static_cast<char*>(ggml_backend_buffer_get_base(staging_));
}

// One line per layer at its first call, hits or not: its slab copied to VRAM, or that it has no hot experts.
// make_call runs this under its mutex, after the previous op's wait, so the backend is idle here.
const std::vector<int32_t>* GpuTier::slots(const Call& c) {
    const bool resident = c.layer < (int) slabs_.size() && slabs_[(size_t) c.layer].up != nullptr;
    if (!resident) {
        if (cpu_only_layers_.insert(c.layer).second) {
            std::fprintf(stderr, "strata-moe: layer %d: no hot experts in %s, every pair on the CPU\n", c.layer,
                         hot_.path.c_str());
        }
        return nullptr;
    }
    Slab& s = slabs_[(size_t) c.layer];
    if (!s.filled) {
        fill(s, c);
        s.filled = true;
    }
    return &s.slot_of;
}

// a batch call of at least min_tokens tokens is a prefill call: its plan, every pair a row of it.
bool GpuTier::prefill(Call& c, const Residency* residency) {
    if (prefill_.min_tokens == 0 || c.n_tokens < prefill_.min_tokens) return false;
    const ggml_tensor* parts[3] = {c.up, c.gate, c.down};
    for (int p = 0; p < 3; ++p) {
        if ((size_t) parts[p]->nb[2] > part_cap_[p]) {
            GGML_ABORT("strata-moe: %s: %zu bytes per expert, the prefill ring's slots hold %zu (the hot set %s has "
                       "smaller %s experts in every layer)", parts[p]->name, parts[p]->nb[2], part_cap_[p],
                       hot_.path.c_str(), p == 0 ? "up" : p == 1 ? "gate" : "down");
        }
    }
    const int64_t n_pairs = c.n_tokens * c.n_used;
    c.expert_of_pair.resize((size_t) n_pairs);
    for (int64_t p = 0; p < n_pairs; ++p) {
        c.expert_of_pair[(size_t) p] = *reinterpret_cast<const int32_t*>(
            static_cast<const char*>(c.ids->data) + (p / c.n_used) * c.ids->nb[1] + (p % c.n_used) * c.ids->nb[0]);
    }
    // straight from the warm tier only when it is CUDA pinned; every other expert outside the slab is staged
    const bool pinned_warm = residency != nullptr && residency->ram_mode() == RamMode::kPinned;
    c.staged.assign((size_t) c.n_expert, 1);
    for (int64_t e = 0; pinned_warm && e < c.n_expert; ++e) {
        c.staged[(size_t) e] = residency->warm_rows(c.layer, (int32_t) e) == nullptr;
    }
    const bool resident = c.layer < (int) slabs_.size() && slabs_[(size_t) c.layer].up != nullptr;
    plan_prefill(c.expert_of_pair, c.n_expert, resident ? &slabs_[(size_t) c.layer].slot_of : nullptr, c.staged,
                 prefill_.set_size, prefill_.n_sets, prefill_.n_stage_sets, c.plan, c.hit_row);
    c.n_hits = n_pairs;
    if (!prefill_logged_) {
        prefill_logged_ = true;
        int64_t slab_pairs = 0;
        for (const PrefillPass& p : c.plan.passes) slab_pairs += p.set < 0 ? (int64_t) p.pairs.size() : 0;
        std::fprintf(stderr,
                     "strata-moe: first prefill call: layer %d, %lld tokens, %lld pairs: %lld in the slab, %lld "
                     "experts streamed (%lld staged: %s) in %zu passes\n",
                     c.layer, (long long) c.n_tokens, (long long) n_pairs, (long long) slab_pairs,
                     (long long) c.plan.ring_experts, (long long) c.plan.staged_experts,
                     residency == nullptr ? "no residency, every expert outside the slab"
                     : pinned_warm        ? "the cold ones"
                                          : "the warm tier is not pinned, every expert outside the slab",
                     c.plan.passes.size());
    }
    return true;
}

// One prefill call's engine (moe_prefill.h) on the tier's backend stream and copy stream.
class GpuTier::Prefill final : public PrefillEngine {
public:
    Prefill(GpuTier& t, Call& c, const size_t* part_bytes, const std::vector<ggml_cgraph*>& graphs,
            const std::vector<ggml_tensor*>& outs, const std::vector<ggml_tensor*>& ins, const char* in_host,
            float* out_host, bool timed)
        : t_(t), c_(c), graphs_(graphs), outs_(outs), ins_(ins), in_host_(in_host), out_host_(out_host), timed_(timed) {
        for (int p = 0; p < 3; ++p) part_bytes_[p] = part_bytes[p];
    }

    void stage_wait(int stage) override {
        const auto t0 = std::chrono::steady_clock::now();
        cuda_check(cudaEventSynchronize(t_.stage_free_[(size_t) stage]), "cudaEventSynchronize (prefill staging)");
        c_.prefill_stage_wait_ms += ms_since(t0);
    }

    // pagecache: the pass's job on the op's threads; direct: the residency's O_DIRECT reads, waited for here
    void stage(const PrefillPass& p) override {
        const auto t0 = std::chrono::steady_clock::now();
        rows_.assign(p.experts.size() * 3, nullptr);
        if (t_.prefill_.cold == PrefillCold::kPageCache) {
            c_.stage.run(next_job_++);
            for (size_t k = 0; k < p.experts.size(); ++k) {
                for (int part = 0; p.staged[k] && part < 3; ++part) {
                    rows_[k * 3 + (size_t) part] = t_.stage_slot(p.stage, k, part);
                }
            }
        } else {
            std::atomic<int64_t> left{0};
            for (size_t k = 0; k < p.experts.size(); ++k) {
                if (!p.staged[k]) continue;
                direct_bytes_ += c_.residency->read_expert(c_.layer, p.experts[k], t_.stage_base(p.stage, k),
                                                           t_.stage_part_offset_, left, &rows_[k * 3]);
            }
            while (left.load(std::memory_order_acquire) != 0) std::this_thread::yield();
        }
        c_.prefill_stage_ms += ms_since(t0);
    }

    // the bytes the direct reads moved
    uint64_t direct_bytes() const { return direct_bytes_; }

    void copy_wait_compute(int set) override {
        const cudaEvent_t consumed = static_cast<cudaEvent_t>(t_.consumed_[(size_t) set].context);
        cuda_check(cudaStreamWaitEvent(t_.copy_stream_, consumed, 0), "cudaStreamWaitEvent (prefill copy)");
    }

    void copy(const PrefillPass& p) override {
        if (timed_) record_copy(pass_index(p) * 4 + 0);
        const char* const* warm = nullptr;
        for (size_t k = 0; k < p.experts.size(); ++k) {
            const int32_t e = p.experts[k];
            if (!p.staged[k]) warm = c_.residency->warm_rows(c_.layer, e);
            for (int part = 0; part < 3; ++part) {
                const char* src = p.staged[k] ? rows_[k * 3 + (size_t) part] : warm[part];
                char* dst = t_.ring_[part] + ((size_t) p.set * (size_t) t_.prefill_.set_size + k) * part_bytes_[part];
                cuda_check(cudaMemcpyAsync(dst, src, part_bytes_[part], cudaMemcpyHostToDevice, t_.copy_stream_),
                           "cudaMemcpyAsync (prefill ring)");
                c_.prefill_bytes += part_bytes_[part];
            }
        }
        if (timed_) record_copy(pass_index(p) * 4 + 1);
        cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(t_.copied_[(size_t) p.set].context), t_.copy_stream_),
                   "cudaEventRecord (prefill copy)");
        if (p.stage >= 0) {
            cuda_check(cudaEventRecord(t_.stage_free_[(size_t) p.stage], t_.copy_stream_),
                       "cudaEventRecord (prefill staging)");
        }
    }

    void compute_wait_copy(int set) override { ggml_backend_event_wait(t_.backend_, &t_.copied_[(size_t) set]); }

    void compute(const PrefillPass& p) override {
        const size_t i = pass_index(p);
        if (timed_) ggml_backend_event_record(t_.timing_event(i * 4 + 2), t_.backend_);
        ggml_cgraph* gf = graphs_[i];
        if (!ggml_gallocr_alloc_graph(t_.galloc_, gf)) {
            GGML_ABORT("strata-moe: layer %d: cannot allocate prefill pass %zu (%zu pairs)", c_.layer, i,
                       p.pairs.size());
        }
        // the pass's three inputs (pair rows, slots, weights), next to each other in the pinned input block
        const size_t n = p.pairs.size();
        for (int j = 0; j < 3; ++j) {
            ggml_backend_tensor_set_async(t_.backend_, ins_[i * 3 + (size_t) j],
                                          in_host_ + ((size_t) j * (size_t) c_.plan.n_rows + (size_t) p.row0) * 4, 0,
                                          n * 4);
        }
        const ggml_status st = ggml_backend_graph_compute_async(t_.backend_, gf);
        if (st != GGML_STATUS_SUCCESS) {
            GGML_ABORT("strata-moe: layer %d: prefill pass %zu failed with status %d", c_.layer, i, st);
        }
        ggml_backend_tensor_get_async(t_.backend_, outs_[i], out_host_ + (size_t) p.row0 * (size_t) c_.n_embd, 0,
                                      ggml_nbytes(outs_[i]));
        if (timed_) ggml_backend_event_record(t_.timing_event(i * 4 + 3), t_.backend_);
        if (p.set >= 0) ggml_backend_event_record(&t_.consumed_[(size_t) p.set], t_.backend_);
    }

private:
    size_t pass_index(const PrefillPass& p) const { return (size_t) (&p - c_.plan.passes.data()); }
    void record_copy(size_t i) {
        cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(t_.timing_event(i)->context), t_.copy_stream_),
                   "cudaEventRecord (prefill timing)");
    }

    GpuTier& t_;
    Call& c_;
    size_t part_bytes_[3] = {};
    const std::vector<ggml_cgraph*>& graphs_;
    const std::vector<ggml_tensor*>& outs_;
    const std::vector<ggml_tensor*>& ins_;  // [pass][rows, ids, weights]
    const char* in_host_;
    float* out_host_;
    bool timed_ = false;
    size_t next_job_ = 0;
    std::vector<const char*> rows_;  // [k, part]: where the last staged pass's staged experts are in staging
    uint64_t direct_bytes_ = 0;
};

void GpuTier::submit(Call& c) {
    if (c.prefill) return submit_prefill(c);
    busy_.lock();
    Slab& s = slabs_[(size_t) c.layer];

    const int64_t n_tok = (int64_t) c.hit_tokens.size(), width = c.hit_width, n_embd = c.n_embd;
    if (graph_ctx_ != nullptr) ggml_free(graph_ctx_);
    const ggml_init_params params = {/*.mem_size =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
                                     /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true};
    graph_ctx_ = ggml_init(params);
    ggml_tensor* x = ggml_new_tensor_3d(graph_ctx_, GGML_TYPE_F32, n_embd, 1, n_tok);
    ggml_tensor* ids = ggml_new_tensor_2d(graph_ctx_, GGML_TYPE_I32, width, n_tok);
    ggml_tensor* w = ggml_new_tensor_3d(graph_ctx_, GGML_TYPE_F32, 1, width, n_tok);
    ggml_set_input(x);
    ggml_set_input(ids);
    ggml_set_input(w);
    ggml_tensor* up = ggml_mul_mat_id(graph_ctx_, s.up, x, ids);
    ggml_tensor* gate = ggml_mul_mat_id(graph_ctx_, s.gate, x, ids);
    ggml_tensor* act = c.limit > 0.0f ? ggml_swiglu_clamp(graph_ctx_, gate, up, c.limit)
                                      : ggml_swiglu_split(graph_ctx_, gate, up);
    ggml_tensor* y = ggml_mul_mat_id(graph_ctx_, s.down, act, ids);
    ggml_tensor* out = ggml_mul(graph_ctx_, y, w);
    ggml_set_output(out);
    ggml_cgraph* gf = ggml_new_graph_custom(graph_ctx_, 16, false);
    ggml_build_forward_expand(gf, out);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor* node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend_, node)) {
            GGML_ABORT("strata-moe: layer %d: ggml-cuda cannot run %s on %s (%s)", c.layer, ggml_op_desc(node),
                       node->src[0]->name, ggml_type_name(node->src[0]->type));
        }
    }
    if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
        GGML_ABORT("strata-moe: layer %d: cannot allocate the hit graph (%lld tokens x %lld slots)", c.layer,
                   (long long) n_tok, (long long) width);
    }

    const size_t x_bytes = ggml_nbytes(x), ids_bytes = ggml_nbytes(ids), w_bytes = ggml_nbytes(w);
    const size_t out_bytes = ggml_nbytes(out);
    const auto pad = [](size_t n) { return GGML_PAD(n, 256); };
    char* host = staging(pad(x_bytes) + pad(ids_bytes) + pad(w_bytes) + out_bytes);
    float* x_host = reinterpret_cast<float*>(host);
    char* ids_host = host + pad(x_bytes);
    char* w_host = ids_host + pad(ids_bytes);
    out_host_ = reinterpret_cast<float*>(w_host + pad(w_bytes));
    for (int64_t j = 0; j < n_tok; ++j) {
        std::memcpy(x_host + j * n_embd, static_cast<const char*>(c.cur->data) + c.hit_tokens[(size_t) j] * c.cur->nb[1],
                    (size_t) n_embd * sizeof(float));
    }
    std::memcpy(ids_host, c.hit_slots.data(), ids_bytes);
    std::memcpy(w_host, c.hit_weights.data(), w_bytes);

    const bool timed = profiling();
    if (timed) ggml_backend_event_record(&start_, backend_);
    ggml_backend_tensor_set_async(backend_, x, x_host, 0, x_bytes);
    ggml_backend_tensor_set_async(backend_, ids, ids_host, 0, ids_bytes);
    ggml_backend_tensor_set_async(backend_, w, w_host, 0, w_bytes);
    const ggml_status st = ggml_backend_graph_compute_async(backend_, gf);
    if (st != GGML_STATUS_SUCCESS) GGML_ABORT("strata-moe: layer %d: the hit graph failed with status %d", c.layer, st);
    ggml_backend_tensor_get_async(backend_, out, out_host_, 0, out_bytes);
    if (timed) ggml_backend_event_record(&end_, backend_);
}

void GpuTier::submit_prefill(Call& c) {
    busy_.lock();
    cuda_check(cudaSetDevice(0), "cudaSetDevice(0)");
    const PrefillPlan& plan = c.plan;
    const int64_t n_embd = c.n_embd, n_ff = c.n_ff, n_tok = c.n_tokens;
    const ggml_tensor* host_t[3] = {c.up, c.gate, c.down};
    const size_t part_bytes[3] = {c.up->nb[2], c.gate->nb[2], c.down->nb[2]};

    // pagecache: the staging jobs, one per staged pass, its staged experts' parts from the fork's mapping, in chunks
    c.stage_table.resize(0);
    for (const PrefillPass& p : plan.passes) {
        if (p.stage < 0 || prefill_.cold != PrefillCold::kPageCache) continue;
        c.stage_table.emplace_back();
        std::vector<StageChunk>& job = c.stage_table.back();
        for (size_t k = 0; k < p.experts.size(); ++k) {
            if (!p.staged[k]) continue;
            for (int part = 0; part < 3; ++part) {
                const char* src =
                    static_cast<const char*>(host_t[part]->data) + (size_t) p.experts[k] * part_bytes[part];
                char* dst = stage_slot(p.stage, k, part);
                for (size_t off = 0; off < part_bytes[part]; off += kStageChunk) {
                    job.push_back({src + off, dst + off, std::min(kStageChunk, part_bytes[part] - off)});
                }
            }
        }
    }
    c.stage.begin(&c.stage_table);

    // pinned host memory: the input, every pass's rows, slots and weights, the output rows
    const auto pad = [](size_t n) { return GGML_PAD(n, 256); };
    const size_t x_bytes = (size_t) (n_embd * n_tok) * sizeof(float);
    const size_t in_bytes = 3 * (size_t) plan.n_rows * 4;
    char* host = staging(pad(x_bytes) + pad(in_bytes) + (size_t) plan.n_rows * (size_t) n_embd * sizeof(float));
    for (int64_t t = 0; t < n_tok; ++t) {
        std::memcpy(host + (size_t) (t * n_embd) * sizeof(float),
                    static_cast<const char*>(c.cur->data) + t * c.cur->nb[1], (size_t) n_embd * sizeof(float));
    }
    char* in_host = host + pad(x_bytes);
    int32_t* rows_host = reinterpret_cast<int32_t*>(in_host);
    int32_t* slots_host = rows_host + plan.n_rows;
    float* w_host = reinterpret_cast<float*>(slots_host + plan.n_rows);
    for (const PrefillPass& p : plan.passes) {
        for (size_t i = 0; i < p.pairs.size(); ++i) {
            const int64_t pair = p.pairs[i], row = p.row0 + (int64_t) i;
            rows_host[row] = (int32_t) (pair / c.n_used);
            slots_host[row] = p.slots[i];
            w_host[row] = *reinterpret_cast<const float*>(static_cast<const char*>(c.weights->data) +
                                                          (pair / c.n_used) * c.weights->nb[2] +
                                                          (pair % c.n_used) * c.weights->nb[1]);
        }
    }
    out_host_ = reinterpret_cast<float*>(in_host + pad(in_bytes));

    // the call's input on the GPU: grown, never shrunk (the previous call is done)
    if (x_buf_ == nullptr || ggml_backend_buffer_get_size(x_buf_) < x_bytes) {
        if (x_buf_ != nullptr) ggml_backend_buffer_free(x_buf_);
        x_buf_ = ggml_backend_buft_alloc_buffer(buft_, x_bytes);
        if (x_buf_ == nullptr) GGML_ABORT("strata-moe: cannot allocate %zu bytes of VRAM for a prefill input", x_bytes);
        ggml_backend_buffer_set_usage(x_buf_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    }

    // one graph per pass, all built (and the allocator grown to the largest) before anything is queued, so no pass
    // reallocates the compute buffer under one still running
    const size_t n_pass = plan.passes.size();
    if (graph_ctx_ != nullptr) ggml_free(graph_ctx_);
    const ggml_init_params params = {
        /*.mem_size =*/ (n_pass * 16 + 1) * ggml_tensor_overhead() + n_pass * ggml_graph_overhead_custom(16, false),
        /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true};
    graph_ctx_ = ggml_init(params);
    ggml_tensor* x_all = ggml_new_tensor_2d(graph_ctx_, GGML_TYPE_F32, n_embd, n_tok);
    if (ggml_backend_tensor_alloc(x_buf_, x_all, ggml_backend_buffer_get_base(x_buf_)) != GGML_STATUS_SUCCESS) {
        GGML_ABORT("strata-moe: layer %d: cannot place the prefill input", c.layer);
    }
    const Slab* slab = c.layer < (int) slabs_.size() ? &slabs_[(size_t) c.layer] : nullptr;
    std::vector<ggml_cgraph*> graphs(n_pass);
    std::vector<ggml_tensor*> outs(n_pass), ins(3 * n_pass);
    for (size_t i = 0; i < n_pass; ++i) {
        const PrefillPass& p = plan.passes[i];
        const int64_t n = (int64_t) p.pairs.size();
        ggml_tensor* w_t[3];
        if (p.set < 0) {
            w_t[0] = slab->up;
            w_t[1] = slab->gate;
            w_t[2] = slab->down;
        } else {
            const struct {
                ggml_type type;
                int64_t n_per_row, n_rows;
            } shape[3] = {{c.up->type, n_embd, n_ff}, {c.gate->type, n_embd, n_ff}, {c.down->type, n_ff, n_embd}};
            for (int part = 0; part < 3; ++part) {
                w_t[part] = ggml_new_tensor_3d(graph_ctx_, shape[part].type, shape[part].n_per_row, shape[part].n_rows,
                                               prefill_.set_size);
                char* addr = ring_[part] + (size_t) p.set * (size_t) prefill_.set_size * part_bytes[part];
                if (ggml_backend_tensor_alloc(ring_buf_, w_t[part], addr) != GGML_STATUS_SUCCESS) {
                    GGML_ABORT("strata-moe: layer %d: cannot place ring set %d of %s", c.layer, p.set,
                               host_t[part]->name);
                }
            }
        }
        ggml_tensor* rows = ggml_new_tensor_1d(graph_ctx_, GGML_TYPE_I32, n);
        ggml_tensor* ids = ggml_new_tensor_2d(graph_ctx_, GGML_TYPE_I32, 1, n);
        ggml_tensor* w = ggml_new_tensor_3d(graph_ctx_, GGML_TYPE_F32, 1, 1, n);
        ggml_set_input(rows);
        ggml_set_input(ids);
        ggml_set_input(w);
        ins[i * 3 + 0] = rows;
        ins[i * 3 + 1] = ids;
        ins[i * 3 + 2] = w;
        ggml_tensor* x = ggml_reshape_3d(graph_ctx_, ggml_get_rows(graph_ctx_, x_all, rows), n_embd, 1, n);
        ggml_tensor* up = ggml_mul_mat_id(graph_ctx_, w_t[0], x, ids);
        ggml_tensor* gate = ggml_mul_mat_id(graph_ctx_, w_t[1], x, ids);
        ggml_tensor* act = c.limit > 0.0f ? ggml_swiglu_clamp(graph_ctx_, gate, up, c.limit)
                                          : ggml_swiglu_split(graph_ctx_, gate, up);
        ggml_tensor* out = ggml_mul(graph_ctx_, ggml_mul_mat_id(graph_ctx_, w_t[2], act, ids), w);
        ggml_set_output(out);
        outs[i] = out;
        graphs[i] = ggml_new_graph_custom(graph_ctx_, 16, false);
        ggml_build_forward_expand(graphs[i], out);
        for (int j = 0; j < ggml_graph_n_nodes(graphs[i]); ++j) {
            const ggml_tensor* node = ggml_graph_node(graphs[i], j);
            if (!ggml_backend_supports_op(backend_, node)) {
                GGML_ABORT("strata-moe: layer %d: ggml-cuda cannot run %s on %s (%s) in a prefill pass", c.layer,
                           ggml_op_desc(node), node->src[0]->name, ggml_type_name(node->src[0]->type));
            }
        }
        if (!ggml_gallocr_reserve(galloc_, graphs[i])) {
            GGML_ABORT("strata-moe: layer %d: cannot reserve VRAM for prefill pass %zu (%lld pairs)", c.layer, i,
                       (long long) n);
        }
    }

    const bool timed = profiling();
    if (timed) ggml_backend_event_record(&prefill_start_, backend_);
    ggml_backend_tensor_set_async(backend_, x_all, host, 0, x_bytes);
    Prefill engine(*this, c, part_bytes, graphs, outs, ins, in_host, out_host_, timed);
    run_prefill(plan, engine);
    c.stage.close();
    c.prefill_passes = (int64_t) n_pass;
    c.prefill_staged_bytes = c.stage.bytes() + engine.direct_bytes();
}

const float* GpuTier::wait(Call& c) {
    if (c.prefill) {
        ggml_backend_synchronize(backend_);
        cuda_check(cudaStreamSynchronize(copy_stream_), "cudaStreamSynchronize (prefill copy)");
        if (profiling()) {
            // busy time per stream, and the GPU time from the call's start to the last pass or copy
            for (size_t i = 0; i < c.plan.passes.size(); ++i) {
                const ggml_backend_event* e = &timing_[i * 4];
                const bool ring = c.plan.passes[i].set >= 0;
                if (ring) c.prefill_copy_ms += elapsed_ms(e[0], e[1]);
                c.prefill_compute_ms += elapsed_ms(e[2], e[3]);
                c.prefill_gpu_ms = std::max(c.prefill_gpu_ms, (double) elapsed_ms(prefill_start_, e[3]));
                if (ring) c.prefill_gpu_ms = std::max(c.prefill_gpu_ms, (double) elapsed_ms(prefill_start_, e[1]));
            }
            c.gpu_ms = c.prefill_gpu_ms;
        }
        busy_.unlock();
        return out_host_;
    }
    ggml_backend_synchronize(backend_);
    if (profiling()) {
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(start_.context),
                                        static_cast<cudaEvent_t>(end_.context)),
                   "cudaEventElapsedTime");
        c.gpu_ms = ms;
    }
    busy_.unlock();
    return out_host_;
}

}  // namespace

std::unique_ptr<HitPath> make_gpu_tier(const HotSet& hot, double vram_budget_gib, const PrefillConfig& prefill) {
    return std::make_unique<GpuTier>(hot, vram_budget_gib, prefill);
}

}  // namespace strata::dsv41
