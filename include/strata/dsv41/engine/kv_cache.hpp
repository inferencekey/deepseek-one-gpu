// include/strata/dsv41/engine/kv_cache.hpp - the per-sequence KV cache of the attention, on the device.
//
// What a layer keeps (SPEC.md "KV cache"; the fork's raw / csa / lid caches and its pooling state):
//
//   every layer      the window: a ring of window_cells f16 rows of head_dim, position p at cell p % window_cells.
//                    window_cells = n_swa - 1 + max_chunk: a chunk of max_chunk tokens writes its rows before its
//                    attention reads them, and its first query still needs the n_swa - 1 rows before it.
//   KV source        the compressed rows at the layer's own ratio: row j (positions j r .. j r + r - 1) at row j,
//                    max_ctx / r rows of head_dim f16 (the fork sizes every source for ratio 1 and fills half of a
//                    ratio-2 one); and the pooling state: the positions of the block not yet complete, up to r - 1
//                    rows of kv and score f32.
//   key owner        the index keys, row j at row j, max_ctx / r rows of idx_dim f16 (rotated, 16-byte rows).
//   index source     the last chunk's top-k ids [max_chunk][top_k] i32, which its Reuse readers attend with.
//
// Readers own no rows. At the real dims and 262,144 tokens this is 800 MiB of rows and keys (SPEC.md: the per-ratio
// layout) plus 40 windows of 4,223 cells (165 MiB at max_chunk 4096), against the fork's 1.31 GiB: kv_budget.py.
//
// Strata's qsa.hpp keeps a paged pool behind a page table for KV streaming; this cache is one sequence's contiguous
// rows (the target holds the whole 262K cache in VRAM), so a row's address is its index.
// The host tracks each layer's next position; attn_prefill checks a chunk against it and fails naming both.
//
// the prefix cache: one snapshot of the sequence's running state, Strata's
// ConversationCheckpoint (core/conversation_cache.hpp) on this cache: the KV stays in place (a row's address is its
// index, as Strata's identity-layout pages) and only what positions past the snapshot overwrite is copied, every
// layer's next position and a KV source's pooling carry (the open block's kv and score), device to device. Restoring
// it continues the sequence at the snapshot's position: window cells, compressed rows and index keys past it are
// written again as new tokens arrive, and attention reads no position past the current one. An index source's top-k
// ids are not kept: every call's index source writes them before its readers read them (attention.cu checks it). The
// one limit is the window ring: the positions written after the snapshot reuse the cells of the n_swa - 1 positions
// before it once more than max_chunk of them were written (window_cells = n_swa - 1 + max_chunk), so a snapshot is
// restorable in place while the sequence is at most max_chunk positions past it.
//
// several conversations: N checkpoint slots in pinned host RAM
// (VRAM is the expert slab's), each kv_checkpoint_bytes(max_ctx) (kv_checkpoint.hpp: the window cells of the last
// n_swa - 1 positions, the completed compressed rows and index keys, the pooling carry). save_checkpoint copies the
// spans at the current position device to host on the stream (queued: the decode after it waits for them in stream
// order, nothing on the host does); restore_checkpoint copies them back host to device in one hop, unless the slot is
// the one the live sequence was last saved to or restored from and the window ring still holds its cells: then the
// in-place restore (the carry only, device to device; the most recent conversation kept on the device, at the cost of
// the carry snapshot's bytes). Every copy of a slot is timed by events on the stream, read when asked
// (checkpoint_times). At the real dims (attn_layers_real, chunk 16384) a checkpoint of the 16,413-token first turn is
// 55.1 MiB and one of 262,144 positions, a slot, 805.0 MiB (kv_checkpoint_bytes; tools/dsv41/pod/e07_gate.py counts N
// slots).
//
// batched decode: every batch slot is a KvCache of its own, of the slot's
// positions and a ring of n_swa cells (max_chunk 1: a slot only decodes), filled from the live sequence's cache by
// copy_state_from when a request is admitted. At the real dims a slot of 32,768 positions holds 5 MiB of window,
// 80 MiB of compressed rows and 20 MiB of index keys (kv_budget.py slot_bytes; tools/dsv41/pod/e07_gate.py counts
// them against the expert slab).
#pragma once

#include "strata/dsv41/engine/attention.hpp"
#include "strata/dsv41/engine/kv_checkpoint.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

struct KvCacheConfig {
    int64_t max_ctx = 262144;   ///< positions one sequence may reach
    int64_t max_chunk = 4096;   ///< the largest prefill chunk (the window ring's slack)
};

struct KvLayerBuffers {
    uint16_t* window = nullptr;   ///< [window_cells][head_dim]
    int64_t window_cells = 0;
    uint16_t* rows = nullptr;     ///< KV sources: [rows_cap][head_dim]
    uint16_t* keys = nullptr;     ///< key owners: [rows_cap][idx_dim]
    int64_t rows_cap = 0;
    float* carry_kv = nullptr;    ///< KV sources: [ratio][head_dim], the incomplete block's positions
    float* carry_score = nullptr;
    int32_t* ids = nullptr;       ///< index sources: [max_chunk][top_k]
    int64_t next_pos = 0;         ///< host: the position the layer's next chunk starts at
    int64_t ids_p0 = -1, ids_T = 0;   ///< host: the chunk `ids` belongs to
};

/// Bytes the cache allocated, by part.
struct KvBytes {
    size_t window = 0, rows = 0, keys = 0, carry = 0, ids = 0;
    size_t snap_carry = 0;   ///< of carry, the in-place snapshot's
    size_t total() const { return window + rows + keys + carry + ids; }
};

/// The bytes KvCache::init allocates for a config, without allocating (the dry run's and the gate's figure).
KvBytes kv_cache_bytes(const AttnDims& d, const std::vector<AttnLayer>& layers, const KvCacheConfig& cfg);

class KvCache {
public:
    KvCache() = default;
    ~KvCache();
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;

    /// Allocates every layer's buffers (zeroed). False with `err` naming the buffer when the device has no room.
    bool init(const AttnDims& d, const std::vector<AttnLayer>& layers, const KvCacheConfig& cfg, std::string& err);

    const AttnDims& dims() const { return d_; }
    const KvCacheConfig& config() const { return cfg_; }
    const std::vector<AttnLayer>& layers() const { return layers_; }
    KvLayerBuffers& layer(int il);
    const KvLayerBuffers& layer(int il) const;
    const KvBytes& bytes() const { return bytes_; }

    /// Every layer back to position 0 (the contents stay, nothing reads them before they are written again); the
    /// checkpoint slots stay too, the in-place snapshot does not.
    void reset();
    /// Tests and benches that write a synthetic history: the layer continues at `next_pos`.
    void set_position(int il, int64_t next_pos);
    /// the state of `src` at its position (every layer at the same one) copied
    /// into this cache device to device on `stream` (kv_state_copies: the two rings may differ), which continues the
    /// sequence there; src is unchanged. Exits naming the part when this cache cannot hold that position.
    void copy_state_from(const KvCache& src, void* stream);

    /// n conversation checkpoint slots in pinned host RAM, kv_checkpoint_bytes(max_ctx) each (0: none). False
    /// with `err` naming the slot and its bytes when the host cannot pin them.
    bool init_checkpoints(int n, std::string& err);
    int checkpoints() const { return (int) slots_.size(); }
    /// The pinned bytes of one slot (a checkpoint of max_ctx positions).
    size_t checkpoint_slot_bytes() const { return slot_bytes_; }
    /// The device bytes the in-place restore keeps beside the live KV (the pooling carry's snapshot).
    size_t checkpoint_device_bytes() const { return bytes_.snap_carry; }
    const KvCheckpointGeometry& checkpoint_geometry() const { return geom_; }
    /// Every layer's state at the current position into slot `slot`, on `stream` (a cudaStream_t: device to host,
    /// queued), and the carry's in-place snapshot. Every layer must be at the same position.
    void save_checkpoint(int slot, void* stream);
    /// Every layer back to slot `slot`'s position, on `stream`: in place when it can (see above), else host to
    /// device. Exits when the slot was never saved.
    void restore_checkpoint(int slot, void* stream);
    /// The position slot `slot` holds (-1: never saved).
    int64_t checkpoint_position(int slot) const;
    /// The last save and the last restore of a slot: the bytes each copied (an in-place restore none from the host),
    /// their copies' ms on the stream (-1: none yet) and whether the restore was in place. Waits for the copies.
    struct CheckpointTimes {
        size_t save_bytes = 0, restore_bytes = 0;
        double save_ms = -1, restore_ms = -1;
        bool in_place = false;
    };
    CheckpointTimes checkpoint_times(int slot) const;

private:
    AttnDims d_{};
    KvCacheConfig cfg_{};
    std::vector<AttnLayer> layers_;
    std::vector<KvLayerBuffers> bufs_;
    KvBytes bytes_{};
    std::vector<void*> allocs_;
    std::vector<float*> snap_carry_;   // per layer: a KV source's carry kv then score (2 x ratio x head_dim f32)
    int64_t snap_pos_ = -1;            // the in-place snapshot's position (-1: none since init or the last reset)
    int inplace_slot_ = -1;            // the slot the in-place snapshot is a copy of
    struct Slot {
        uint8_t* host = nullptr;       // pinned, slot_bytes_
        int64_t pos = -1;
        size_t bytes = 0, restore_bytes = 0;
        bool in_place = false, saved = false, restored = false;
        void* ev[4] = {nullptr, nullptr, nullptr, nullptr};   // cudaEvent_t: save start, end, restore start, end
    };
    KvCheckpointGeometry geom_;
    std::vector<Slot> slots_;
    size_t slot_bytes_ = 0;
    void check_slot(int slot, const char* what) const;
    bool can_restore_in_place(int slot) const;
};

}  // namespace strata::dsv41::engine
