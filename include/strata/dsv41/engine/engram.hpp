// include/strata/dsv41/engine/engram.hpp - the host half of the engram (SPEC.md section 2): the n-gram row hash
// and the tables, which stay in pinned host RAM and are gathered per token (each real
// table is 384M rows of 256 q2_K values, 84 bytes a row, about 30 GiB; the target has >= 256 GB of RAM).
//
// Hash (deepseek41.cpp 262-356, llm_graph_input_engram::set_input): per token the n-gram c_0 = map(t_i), c_s =
// map(t_{i-s}) for s = 1 .. n_gram - 1 through engram.token_map; a predecessor before the sequence start, and every
// older one, reads as pad_id (the look-back stops at the first missing token), and so does a token outside the map.
// rolling_1 = c_0 m_0 ^ c_1 m_1, rolling_s = rolling_{s-1} ^ c_s m_s (u64, wrapping; the layer's n_gram multipliers),
// and for head h the bucket b = (s - 1) heads + h reads row rolling_s % prime[b] + offset[b] (as i32, the fork's
// type). Strata's PLE hash (kernels/ngram.hpp ngram_mixed) mixes its n-gram differently (no per-bucket prime and
// offset, no look-back stop), so this is the fork's formula written out, not an adaptation of it.
//
// Gather: Strata's PleTable::gather_batch (src/kernels/ngram.cpp; rows of a chunk in one call, read from a resident
// table) with the decode moved to the GPU: the raw q2_K rows of a chunk are copied by host threads into a pinned
// staging buffer, then one cudaMemcpyAsync puts them on the device, where engram_forward (dense.hpp) dequantizes them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::dsv41::engine {

/// The engram metadata of a model (GGUF deepseek41.engram.*): per engram layer e (the index in layer_ids)
/// multipliers [e][n_gram], primes and offsets [e][cols].
struct EngramHash {
    int n_gram = 4;
    int heads = 8;
    int64_t pad = 2;                       ///< engram.pad_id
    std::vector<int32_t> layer_ids;        ///< engram.layer_ids
    std::vector<int32_t> token_map;        ///< engram.token_map
    std::vector<uint64_t> multipliers, primes, offsets;
    int cols() const { return (n_gram - 1) * heads; }
    /// The engram index of layer il (its position in layer_ids), -1 for a layer without one.
    int index_of(int il) const;
    /// The row count engram layer e's buckets need: max over its buckets of offset + prime (the fork's load check).
    int64_t rows_needed(int e) const;
    /// Exits naming the problem when the arrays do not match n_gram, heads and the layer count, or a prime is 0.
    void check() const;
};

/// rows [T][cols] of engram layer e for T consecutive tokens of one sequence, given the n_prev (<= n_gram - 1)
/// tokens before them, oldest first (fewer than n_gram - 1 only at the sequence start: those positions are missing).
void engram_rows(const EngramHash& h, int e, const int32_t* tokens, int64_t T, const int32_t* prev, int n_prev,
                 int32_t* rows);

/// One engram table in pinned host RAM: n_rows rows of row_bytes (84 for 256 q2_K values), owned (alloc) or a view
/// of memory the caller keeps pinned (view).
class EngramTable {
public:
    EngramTable() = default;
    ~EngramTable();
    EngramTable(const EngramTable&) = delete;
    EngramTable& operator=(const EngramTable&) = delete;
    /// cudaHostAlloc of n_rows x row_bytes. False with `err` set when it fails.
    bool alloc(int64_t n_rows, int row_bytes, std::string& err);
    void view(const uint8_t* rows, int64_t n_rows, int row_bytes);
    uint8_t* data() { return owned_; }
    const uint8_t* rows() const { return rows_; }
    int64_t n_rows() const { return n_rows_; }
    int row_bytes() const { return row_bytes_; }

private:
    uint8_t* owned_ = nullptr;
    const uint8_t* rows_ = nullptr;
    int64_t n_rows_ = 0;
    int row_bytes_ = 0;
};

/// The n rows of `rows` (each < t.n_rows(), exits otherwise) copied in order into dst (n x row_bytes, host): a decode
/// token's rows into the pinned buffer its captured graph copies to the device (forward.hpp).
void engram_copy_rows(const EngramTable& t, const int32_t* rows, int64_t n, uint8_t* dst);

/// The gather of a chunk's rows: a pinned staging buffer and a device buffer of max_rows rows, host threads for the
/// copy. One per stream; a call waits for its previous copy out of the staging buffer before reusing it.
class EngramGather {
public:
    EngramGather() = default;
    ~EngramGather();
    EngramGather(const EngramGather&) = delete;
    EngramGather& operator=(const EngramGather&) = delete;
    /// threads: host copy threads for calls of more than 1 MiB (smaller ones copy on the caller's thread).
    bool init(int64_t max_rows, int row_bytes, int threads, std::string& err);
    /// The n rows of `rows` (each < t.n_rows(), exits otherwise) on the device, in order, copied on `stream`; the
    /// pointer stays valid until the next call. last_host_ms: the host copy's wall time of this call. The staging is
    /// reused once the previous copy out of it is done: a host wait only when it is not (syncs counts them).
    const void* gather(const EngramTable& t, const int32_t* rows, int64_t n, void* stream);
    double last_host_ms() const { return last_host_ms_; }
    int64_t syncs() const { return syncs_; }
    size_t device_bytes() const { return (size_t) max_rows_ * (size_t) row_bytes_; }

private:
    uint8_t* staging_ = nullptr;
    void* dev_ = nullptr;
    void* done_ = nullptr;   // cudaEvent_t: the last copy out of the staging buffer
    int64_t max_rows_ = 0;
    int row_bytes_ = 0;
    int threads_ = 1;
    double last_host_ms_ = 0;
    int64_t syncs_ = 0;
};

}  // namespace strata::dsv41::engine
