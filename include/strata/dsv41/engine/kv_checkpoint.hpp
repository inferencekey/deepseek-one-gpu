// include/strata/dsv41/engine/kv_checkpoint.hpp - what a
// conversation checkpoint holds of the KV cache (kv_cache.hpp) at a position P, and where each part goes in its host
// buffer. Plain C++ (no CUDA): KvCache copies these spans to and from pinned host RAM, the dev VM's tests
// (tests/dsv41/engine/checkpoint_test.cpp) check the same spans on a host image of the device buffers.
//
// At position P (the next position the sequence writes) a layer's later reads need, of what earlier chunks wrote:
//   every layer    the window cells of positions max(0, P - (n_swa - 1)) .. P - 1 (the first query's n_swa - 1 rows
//                  before it), cell p % window_cells: one or two runs of the ring (two when they wrap past its end);
//   KV source      the compressed rows 0 .. P / ratio - 1 (the blocks completed before P) and the pooling carry (the
//                  open block's kv and score rows, ratio x head_dim f32 each, as the device keeps them);
//   key owner      the index keys 0 .. P / ratio - 1.
// Nothing else: an index source's top-k ids are written by every call before its readers read them (kv_cache.hpp),
// a reader owns no rows, every row past P is written again before it is read. Restoring the spans and every layer's
// next position at P continues the sequence as if it had been read straight to P: the same values at the same cells
// for every read of the next call (checkpoint_test.cpp checks it bit for bit on the CPU reference).
//
// The spans of a checkpoint are packed one after the other, layer by layer, in its host buffer; a slot is sized for
// the largest P (max_ctx), kv_checkpoint_bytes(max_ctx), so any conversation fits any slot.
#pragma once

#include "strata/dsv41/engine/attention.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::dsv41::engine {

enum class KvPart : int { Window, Rows, Keys, CarryKv, CarryScore };
const char* kv_part_name(KvPart p);

/// One contiguous run of a layer's buffer: `count` rows of `row_bytes` from row `first` of the part, at byte `offset`
/// of the checkpoint's host buffer.
struct KvSpan {
    int layer = 0;
    KvPart part = KvPart::Window;
    int64_t first = 0, count = 0;
    size_t row_bytes = 0, offset = 0;
    size_t bytes() const { return (size_t) count * row_bytes; }
};

/// The geometry the spans follow: the dims, the layer roles and the ring's size (KvCacheConfig's max_chunk:
/// window_cells = n_swa - 1 + max_chunk).
struct KvCheckpointGeometry {
    AttnDims dims{};
    std::vector<AttnLayer> layers;
    int64_t window_cells = 0;
};

/// The spans of a checkpoint at position p (0 <= p), packed from offset 0 in layer order. Exits naming the value
/// when p is negative or the geometry has no ring.
std::vector<KvSpan> kv_checkpoint_spans(const KvCheckpointGeometry& g, int64_t p);
/// The host bytes of a checkpoint at position p: the sum of its spans.
size_t kv_checkpoint_bytes(const KvCheckpointGeometry& g, int64_t p);

/// one device to device copy of a state at position p between two caches of
/// the same dims and layers whose rings differ (the live sequence's n_swa - 1 + chunk cells, a batch slot's n_swa):
/// `count` rows of `row_bytes` of a layer's part, from row `src` of one cache to row `dst` of the other.
struct KvCopy {
    int layer = 0;
    KvPart part = KvPart::Window;
    int64_t src = 0, dst = 0, count = 0;
    size_t row_bytes = 0;
};
/// The copies that move the state at position p from a cache of geometry `from` to one of geometry `to`: what
/// kv_checkpoint_spans names, the window's positions cut where either ring wraps (position q at cell q % cells of
/// each), the compressed rows, index keys and carry row for row. A checkpoint's host buffer packs the same bytes in the
/// same order in both geometries (the window's positions ascending), so one saved from either restores into either.
/// Exits naming the value when the dims or the layers differ, or a ring is shorter than n_swa.
std::vector<KvCopy> kv_state_copies(const KvCheckpointGeometry& from, const KvCheckpointGeometry& to, int64_t p);

}  // namespace strata::dsv41::engine
