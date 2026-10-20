// src/dsv41/engine/kv_cache.cpp - the per-sequence KV cache, see include/strata/dsv41/engine/kv_cache.hpp.
#include "strata/dsv41/engine/kv_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "dsv41 kv cache: %s\n", what);
    std::exit(1);
}

}  // namespace

KvBytes kv_cache_bytes(const AttnDims& d, const std::vector<AttnLayer>& layers, const KvCacheConfig& cfg) {
    KvBytes b;
    const size_t D = (size_t) d.head_dim;
    for (const AttnLayer& l : layers) {
        b.window += (size_t) (d.n_swa - 1 + cfg.max_chunk) * D * 2;
        const int64_t cap = l.kv_source || l.key_owner ? (cfg.max_ctx + l.ratio - 1) / l.ratio : 0;
        if (l.kv_source) {
            b.rows += (size_t) cap * D * 2;
            b.carry += (size_t) 4 * l.ratio * D * 4;   // kv and score, and their in-place snapshot
            b.snap_carry += (size_t) 2 * l.ratio * D * 4;
        }
        if (l.key_owner) b.keys += (size_t) cap * d.idx_dim * 2;
        if (l.index_source) b.ids += (size_t) cfg.max_chunk * d.top_k * 4;
    }
    return b;
}

KvCache::~KvCache() {
    for (void* p : allocs_) cudaFree(p);
    for (Slot& s : slots_) {
        if (s.host) cudaFreeHost(s.host);
        for (void* e : s.ev)
            if (e) cudaEventDestroy((cudaEvent_t) e);
    }
}

bool KvCache::init(const AttnDims& d, const std::vector<AttnLayer>& layers, const KvCacheConfig& cfg, std::string& err) {
    if (!bufs_.empty()) fail("init called twice");
    if (cfg.max_ctx < 1 || cfg.max_chunk < 1 || cfg.max_chunk > cfg.max_ctx) {
        char msg[128];
        std::snprintf(msg, sizeof msg, "max_ctx %lld, max_chunk %lld (1 <= max_chunk <= max_ctx)", (long long) cfg.max_ctx,
                      (long long) cfg.max_chunk);
        fail(msg);
    }
    d_ = d;
    cfg_ = cfg;
    layers_ = layers;
    bufs_.resize(layers.size());
    snap_carry_.assign(layers.size(), nullptr);
    auto alloc = [&](size_t bytes, const char* what, int il, size_t& part) -> void* {
        void* p = nullptr;
        if (const cudaError_t e = cudaMalloc(&p, bytes); e != cudaSuccess) {
            char msg[192];
            std::snprintf(msg, sizeof msg, "%s of layer %d: %zu bytes: %s", what, il, bytes, cudaGetErrorString(e));
            err = msg;
            return nullptr;
        }
        allocs_.push_back(p);
        part += bytes;
        if (const cudaError_t e = cudaMemset(p, 0, bytes); e != cudaSuccess) {
            err = std::string("cudaMemset: ") + cudaGetErrorString(e);
            return nullptr;
        }
        return p;
    };
    const size_t D = (size_t) d.head_dim;
    for (size_t i = 0; i < layers.size(); ++i) {
        const AttnLayer& l = layers[i];
        KvLayerBuffers& b = bufs_[i];
        const int il = (int) i;
        b.window_cells = d.n_swa - 1 + cfg.max_chunk;
        if (!(b.window = (uint16_t*) alloc((size_t) b.window_cells * D * 2, "window", il, bytes_.window))) return false;
        if (l.kv_source || l.key_owner) b.rows_cap = (cfg.max_ctx + l.ratio - 1) / l.ratio;
        if (l.kv_source) {
            if (!(b.rows = (uint16_t*) alloc((size_t) b.rows_cap * D * 2, "compressed rows", il, bytes_.rows))) return false;
            if (!(b.carry_kv = (float*) alloc((size_t) l.ratio * D * 4, "pooling state kv", il, bytes_.carry))) return false;
            if (!(b.carry_score = (float*) alloc((size_t) l.ratio * D * 4, "pooling state score", il, bytes_.carry))) return false;
            if (!(snap_carry_[i] = (float*) alloc((size_t) 2 * l.ratio * D * 4, "snapshot of the pooling state", il,
                                                   bytes_.carry)))
                return false;
            bytes_.snap_carry += (size_t) 2 * l.ratio * D * 4;
        }
        if (l.key_owner &&
            !(b.keys = (uint16_t*) alloc((size_t) b.rows_cap * d.idx_dim * 2, "index keys", il, bytes_.keys)))
            return false;
        if (l.index_source &&
            !(b.ids = (int32_t*) alloc((size_t) cfg.max_chunk * d.top_k * 4, "top-k ids", il, bytes_.ids)))
            return false;
    }
    geom_ = KvCheckpointGeometry{d, layers, d.n_swa - 1 + cfg.max_chunk};
    const KvBytes want = kv_cache_bytes(d, layers, cfg);
    if (want.total() != bytes_.total() || want.snap_carry != bytes_.snap_carry) fail("kv_cache_bytes differs from init's allocations");
    return true;
}

bool KvCache::init_checkpoints(int n, std::string& err) {
    if (bufs_.empty() || !slots_.empty() || n < 0) fail("init_checkpoints: before init, twice, or a negative count");
    slot_bytes_ = kv_checkpoint_bytes(geom_, cfg_.max_ctx);
    slots_.resize((size_t) n);
    for (int k = 0; k < n; ++k) {
        Slot& s = slots_[(size_t) k];
        if (const cudaError_t e = cudaHostAlloc((void**) &s.host, slot_bytes_, cudaHostAllocDefault); e != cudaSuccess) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "conversation checkpoint %d of %d: %zu bytes (%.1f MiB) of pinned host RAM for "
                          "a checkpoint of %lld positions, %.1f MiB for all %d: %s", k, n, slot_bytes_,
                          slot_bytes_ / 1048576.0, (long long) cfg_.max_ctx, (double) n * slot_bytes_ / 1048576.0, n,
                          cudaGetErrorString(e));
            err = msg;
            s.host = nullptr;
            return false;
        }
        for (void*& ev : s.ev)
            if (cudaEventCreate((cudaEvent_t*) &ev) != cudaSuccess) {
                err = "cudaEventCreate for a conversation checkpoint";
                return false;
            }
    }
    return true;
}

KvLayerBuffers& KvCache::layer(int il) {
    if (il < 0 || (size_t) il >= bufs_.size()) {
        char msg[96];
        std::snprintf(msg, sizeof msg, "layer %d outside the cache's %zu layers", il, bufs_.size());
        fail(msg);
    }
    return bufs_[(size_t) il];
}

const KvLayerBuffers& KvCache::layer(int il) const { return const_cast<KvCache*>(this)->layer(il); }

void KvCache::reset() {
    for (auto& b : bufs_) {
        b.next_pos = 0;
        b.ids_p0 = -1;
        b.ids_T = 0;
    }
    snap_pos_ = -1;
    inplace_slot_ = -1;
}

void KvCache::check_slot(int slot, const char* what) const {
    if (slot < 0 || (size_t) slot >= slots_.size()) {
        char msg[128];
        std::snprintf(msg, sizeof msg, "%s: checkpoint %d outside the %zu slots", what, slot, slots_.size());
        fail(msg);
    }
}

namespace {

// the device address of a span's first row
void* span_device(const KvLayerBuffers& b, const KvSpan& s) {
    uint8_t* base = nullptr;
    switch (s.part) {
        case KvPart::Window: base = (uint8_t*) b.window; break;
        case KvPart::Rows: base = (uint8_t*) b.rows; break;
        case KvPart::Keys: base = (uint8_t*) b.keys; break;
        case KvPart::CarryKv: base = (uint8_t*) b.carry_kv; break;
        case KvPart::CarryScore: base = (uint8_t*) b.carry_score; break;
    }
    return base + (size_t) s.first * s.row_bytes;
}

}  // namespace

void KvCache::save_checkpoint(int slot, void* stream) {
    check_slot(slot, "save_checkpoint");
    const int64_t p = bufs_[0].next_pos;
    for (size_t i = 0; i < bufs_.size(); ++i) {
        const KvLayerBuffers& b = bufs_[i];
        if (b.next_pos != p) {
            char msg[128];
            std::snprintf(msg, sizeof msg, "save_checkpoint: layer %zu at position %lld, layer 0 at %lld", i,
                          (long long) b.next_pos, (long long) p);
            fail(msg);
        }
        if (!snap_carry_[i]) continue;
        const size_t bytes = (size_t) layers_[i].ratio * d_.head_dim * 4;
        if (cudaMemcpyAsync(snap_carry_[i], b.carry_kv, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t) stream) != cudaSuccess ||
            cudaMemcpyAsync(snap_carry_[i] + bytes / 4, b.carry_score, bytes, cudaMemcpyDeviceToDevice,
                            (cudaStream_t) stream) != cudaSuccess)
            fail("save_checkpoint: the carry's in-place snapshot failed");
    }
    snap_pos_ = p;
    inplace_slot_ = slot;
    Slot& s = slots_[(size_t) slot];
    const cudaStream_t st = (cudaStream_t) stream;
    if (cudaEventRecord((cudaEvent_t) s.ev[0], st) != cudaSuccess) fail("save_checkpoint: cudaEventRecord");
    size_t n = 0;
    for (const KvSpan& sp : kv_checkpoint_spans(geom_, p)) {
        if (sp.offset + sp.bytes() > slot_bytes_) fail("save_checkpoint: a span past the slot");
        if (cudaMemcpyAsync(s.host + sp.offset, span_device(bufs_[(size_t) sp.layer], sp), sp.bytes(),
                            cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "save_checkpoint: layer %d's %s (%zu bytes) to checkpoint %d failed", sp.layer,
                          kv_part_name(sp.part), sp.bytes(), slot);
            fail(msg);
        }
        n += sp.bytes();
    }
    if (cudaEventRecord((cudaEvent_t) s.ev[1], st) != cudaSuccess) fail("save_checkpoint: cudaEventRecord");
    s.pos = p;
    s.bytes = n;
    s.saved = true;
}

bool KvCache::can_restore_in_place(int slot) const {
    return slot == inplace_slot_ && snap_pos_ >= 0 && !bufs_.empty() && bufs_[0].next_pos >= snap_pos_ &&
           bufs_[0].next_pos - snap_pos_ <= cfg_.max_chunk;
}

void KvCache::restore_checkpoint(int slot, void* stream) {
    check_slot(slot, "restore_checkpoint");
    Slot& s = slots_[(size_t) slot];
    if (s.pos < 0) {
        char msg[96];
        std::snprintf(msg, sizeof msg, "restore_checkpoint: checkpoint %d was never saved", slot);
        fail(msg);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const bool in_place = can_restore_in_place(slot);
    if (cudaEventRecord((cudaEvent_t) s.ev[2], st) != cudaSuccess) fail("restore_checkpoint: cudaEventRecord");
    if (!in_place) {
        for (const KvSpan& sp : kv_checkpoint_spans(geom_, s.pos))
            if (cudaMemcpyAsync(span_device(bufs_[(size_t) sp.layer], sp), s.host + sp.offset, sp.bytes(),
                                cudaMemcpyHostToDevice, st) != cudaSuccess) {
                char msg[160];
                std::snprintf(msg, sizeof msg, "restore_checkpoint: layer %d's %s (%zu bytes) from checkpoint %d failed",
                              sp.layer, kv_part_name(sp.part), sp.bytes(), slot);
                fail(msg);
            }
    }
    for (size_t i = 0; i < bufs_.size(); ++i) {
        KvLayerBuffers& b = bufs_[i];
        b.next_pos = s.pos;
        b.ids_p0 = -1;
        b.ids_T = 0;
        if (!snap_carry_[i]) continue;
        // in place: the live carry from its snapshot; from the host: the snapshot from the live carry just restored,
        // so the next restore of this slot can be in place
        const size_t bytes = (size_t) layers_[i].ratio * d_.head_dim * 4;
        float* to_kv = in_place ? b.carry_kv : snap_carry_[i];
        float* to_score = in_place ? b.carry_score : snap_carry_[i] + bytes / 4;
        const float* from_kv = in_place ? snap_carry_[i] : b.carry_kv;
        const float* from_score = in_place ? snap_carry_[i] + bytes / 4 : b.carry_score;
        if (cudaMemcpyAsync(to_kv, from_kv, bytes, cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
            cudaMemcpyAsync(to_score, from_score, bytes, cudaMemcpyDeviceToDevice, st) != cudaSuccess)
            fail("restore_checkpoint: the carry copies failed");
    }
    if (cudaEventRecord((cudaEvent_t) s.ev[3], st) != cudaSuccess) fail("restore_checkpoint: cudaEventRecord");
    snap_pos_ = s.pos;
    inplace_slot_ = slot;
    s.in_place = in_place;
    s.restored = true;
    s.restore_bytes = in_place ? 0 : s.bytes;
}

int64_t KvCache::checkpoint_position(int slot) const {
    check_slot(slot, "checkpoint_position");
    return slots_[(size_t) slot].pos;
}

KvCache::CheckpointTimes KvCache::checkpoint_times(int slot) const {
    check_slot(slot, "checkpoint_times");
    const Slot& s = slots_[(size_t) slot];
    CheckpointTimes t;
    t.save_bytes = s.bytes;
    t.restore_bytes = s.restore_bytes;
    t.in_place = s.in_place;
    auto ms = [](void* a, void* b) {
        float v = 0;
        if (cudaEventSynchronize((cudaEvent_t) b) != cudaSuccess || cudaEventElapsedTime(&v, (cudaEvent_t) a, (cudaEvent_t) b) != cudaSuccess)
            fail("checkpoint_times: the copies' events");
        return (double) v;
    };
    if (s.saved) t.save_ms = ms(s.ev[0], s.ev[1]);
    if (s.restored) t.restore_ms = ms(s.ev[2], s.ev[3]);
    return t;
}

void KvCache::copy_state_from(const KvCache& src, void* stream) {
    if (src.bufs_.empty() || src.bufs_.size() != bufs_.size()) fail("copy_state_from: a cache of another model");
    const int64_t p = src.bufs_[0].next_pos;
    for (size_t i = 0; i < src.bufs_.size(); ++i)
        if (src.bufs_[i].next_pos != p) {
            char msg[128];
            std::snprintf(msg, sizeof msg, "copy_state_from: layer %zu at position %lld, layer 0 at %lld", i,
                          (long long) src.bufs_[i].next_pos, (long long) p);
            fail(msg);
        }
    if (p > cfg_.max_ctx) {
        char msg[128];
        std::snprintf(msg, sizeof msg, "copy_state_from: position %lld, this cache holds %lld", (long long) p,
                      (long long) cfg_.max_ctx);
        fail(msg);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    for (const KvCopy& c : kv_state_copies(src.geom_, geom_, p)) {
        const KvSpan from{c.layer, c.part, c.src, c.count, c.row_bytes, 0}, to{c.layer, c.part, c.dst, c.count, c.row_bytes, 0};
        if (cudaMemcpyAsync(span_device(bufs_[(size_t) c.layer], to), span_device(src.bufs_[(size_t) c.layer], from),
                            from.bytes(), cudaMemcpyDeviceToDevice, st) != cudaSuccess) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "copy_state_from: layer %d's %s (%zu bytes) failed", c.layer, kv_part_name(c.part),
                          from.bytes());
            fail(msg);
        }
    }
    for (KvLayerBuffers& b : bufs_) {
        b.next_pos = p;
        b.ids_p0 = -1;
        b.ids_T = 0;
    }
    snap_pos_ = -1;
    inplace_slot_ = -1;
}

void KvCache::set_position(int il, int64_t next_pos) {
    if (next_pos < 0 || next_pos > cfg_.max_ctx) {
        char msg[128];
        std::snprintf(msg, sizeof msg, "layer %d: position %lld outside 0..%lld", il, (long long) next_pos, (long long) cfg_.max_ctx);
        fail(msg);
    }
    layer(il).next_pos = next_pos;
}

}  // namespace strata::dsv41::engine
