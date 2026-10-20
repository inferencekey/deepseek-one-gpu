// src/dsv41/engine/kv_checkpoint.cpp - the spans of a conversation checkpoint, see
// include/strata/dsv41/engine/kv_checkpoint.hpp.
#include "strata/dsv41/engine/kv_checkpoint.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine {

const char* kv_part_name(KvPart p) {
    switch (p) {
        case KvPart::Window: return "window";
        case KvPart::Rows: return "compressed rows";
        case KvPart::Keys: return "index keys";
        case KvPart::CarryKv: return "pooling carry kv";
        case KvPart::CarryScore: return "pooling carry score";
    }
    return "?";
}

std::vector<KvSpan> kv_checkpoint_spans(const KvCheckpointGeometry& g, int64_t p) {
    if (p < 0 || g.window_cells < g.dims.n_swa) {
        std::fprintf(stderr, "dsv41 kv checkpoint: position %lld, %lld window cells (at least n_swa %d)\n", (long long) p,
                     (long long) g.window_cells, g.dims.n_swa);
        std::exit(1);
    }
    std::vector<KvSpan> out;
    size_t offset = 0;
    auto add = [&](int il, KvPart part, int64_t first, int64_t count, size_t row_bytes) {
        if (count <= 0) return;
        out.push_back(KvSpan{il, part, first, count, row_bytes, offset});
        offset += (size_t) count * row_bytes;
    };
    const size_t D = (size_t) g.dims.head_dim;
    const int64_t W = g.window_cells, lo = std::max<int64_t>(0, p - (g.dims.n_swa - 1));
    for (size_t i = 0; i < g.layers.size(); ++i) {
        const AttnLayer& l = g.layers[i];
        const int il = (int) i;
        // the window's positions lo .. p - 1 at cells lo % W ..: the run to the ring's end, then the rest from cell 0
        const int64_t c0 = lo % W, n = p - lo, first_run = std::min(n, W - c0);
        add(il, KvPart::Window, c0, first_run, D * 2);
        add(il, KvPart::Window, 0, n - first_run, D * 2);
        const int64_t blocks = l.ratio ? p / l.ratio : 0;
        if (l.kv_source) {
            add(il, KvPart::Rows, 0, blocks, D * 2);
            add(il, KvPart::CarryKv, 0, l.ratio, D * 4);
            add(il, KvPart::CarryScore, 0, l.ratio, D * 4);
        }
        if (l.key_owner) add(il, KvPart::Keys, 0, blocks, (size_t) g.dims.idx_dim * 2);
    }
    return out;
}

size_t kv_checkpoint_bytes(const KvCheckpointGeometry& g, int64_t p) {
    size_t n = 0;
    for (const KvSpan& s : kv_checkpoint_spans(g, p)) n += s.bytes();
    return n;
}

std::vector<KvCopy> kv_state_copies(const KvCheckpointGeometry& from, const KvCheckpointGeometry& to, int64_t p) {
    const AttnDims& a = from.dims;
    const AttnDims& b = to.dims;
    bool same = a.head_dim == b.head_dim && a.idx_dim == b.idx_dim && a.n_swa == b.n_swa && from.layers.size() == to.layers.size();
    for (size_t i = 0; same && i < from.layers.size(); ++i) {
        const AttnLayer &x = from.layers[i], &y = to.layers[i];
        same = x.ratio == y.ratio && x.kv_source == y.kv_source && x.key_owner == y.key_owner;
    }
    if (!same || p < 0 || from.window_cells < a.n_swa || to.window_cells < b.n_swa) {
        std::fprintf(stderr, "dsv41 kv state copy: position %lld between caches of other dims or layers, or a ring of "
                     "%lld or %lld cells (at least n_swa %d)\n", (long long) p, (long long) from.window_cells,
                     (long long) to.window_cells, a.n_swa);
        std::exit(1);
    }
    std::vector<KvCopy> out;
    const size_t D = (size_t) a.head_dim;
    const int64_t Wa = from.window_cells, Wb = to.window_cells, lo = std::max<int64_t>(0, p - (a.n_swa - 1));
    for (size_t i = 0; i < from.layers.size(); ++i) {
        const AttnLayer& l = from.layers[i];
        const int il = (int) i;
        // the window's positions lo .. p - 1, each run ending where either ring wraps
        for (int64_t q = lo; q < p;) {
            const int64_t ca = q % Wa, cb = q % Wb, n = std::min({p - q, Wa - ca, Wb - cb});
            out.push_back(KvCopy{il, KvPart::Window, ca, cb, n, D * 2});
            q += n;
        }
        const int64_t blocks = l.ratio ? p / l.ratio : 0;
        if (l.kv_source) {
            if (blocks) out.push_back(KvCopy{il, KvPart::Rows, 0, 0, blocks, D * 2});
            out.push_back(KvCopy{il, KvPart::CarryKv, 0, 0, l.ratio, D * 4});
            out.push_back(KvCopy{il, KvPart::CarryScore, 0, 0, l.ratio, D * 4});
        }
        if (l.key_owner && blocks) out.push_back(KvCopy{il, KvPart::Keys, 0, 0, blocks, (size_t) a.idx_dim * 2});
    }
    return out;
}

}  // namespace strata::dsv41::engine
