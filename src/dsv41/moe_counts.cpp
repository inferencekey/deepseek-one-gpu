// src/dsv41/moe_counts.cpp - the activation counts of strata_moe_forward (moe_counts.h) and the process's counter,
// configured from the environment at the first call:
//   STRATA_MOE_COUNTS=<path>        the counts file; unset or empty: no counts
//   STRATA_MOE_COUNTS_EVERY=<n>     calls between two writes (default 4096: about 100 DeepSeek-V4.1 decode tokens of
//                                   40 layers, so a server killed without its exit handlers loses at most that)
#include "moe_counts.h"

#include "moe_op.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <utility>

namespace strata::dsv41 {

ActivationCounts::ActivationCounts(std::string path, int64_t every) : path_(std::move(path)), every_(every) {
    if (path_.empty()) GGML_ABORT("strata-moe: activation counts need a file path");
    if (every_ < 1) GGML_ABORT("strata-moe: activation counts every %lld calls: expected >= 1", (long long) every_);
}

template <class IdAt>
void ActivationCounts::record_ids(int layer, int64_t n_expert, int64_t n_used, int64_t n_tokens, IdAt id_at) {
    std::lock_guard<std::mutex> lock(mutex_);
    Layer& l = layers_[layer];
    if (l.decode.empty()) {
        l.decode.assign((size_t) n_expert, 0);
        l.batch.assign((size_t) n_expert, 0);
    } else if ((int64_t) l.decode.size() != n_expert) {
        GGML_ABORT("strata-moe: activation counts: layer %d has %lld experts in this call and %zu before", layer,
                   (long long) n_expert, l.decode.size());
    }
    std::vector<int64_t>& to = n_tokens == 1 ? l.decode : l.batch;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t s = 0; s < n_used; ++s) {
            const int32_t e = id_at(t, s);
            if (e < 0 || e >= n_expert) {
                GGML_ABORT("strata-moe: activation counts: layer %d, token %lld slot %lld is expert %d, outside [0, "
                           "%lld)", layer, (long long) t, (long long) s, e, (long long) n_expert);
            }
            ++to[(size_t) e];
        }
    }
    if (n_tokens == 1) {
        ++l.decode_calls;
    } else {
        ++l.batch_calls;
        l.batch_tokens += n_tokens;
    }
    // the first call writes at once, so a path that cannot be written aborts now
    if (calls_++ % every_ == 0) write_locked();
}

void ActivationCounts::record(int layer, int64_t n_expert, const ggml_tensor* ids) {
    record_ids(layer, n_expert, ids->ne[0], ids->ne[1], [ids](int64_t t, int64_t s) {
        return *reinterpret_cast<const int32_t*>(static_cast<const char*>(ids->data) + t * ids->nb[1] + s * ids->nb[0]);
    });
}

void ActivationCounts::record(int layer, int64_t n_expert, const int32_t* ids, int64_t n_used, int64_t n_tokens) {
    record_ids(layer, n_expert, n_used, n_tokens, [ids, n_used](int64_t t, int64_t s) { return ids[t * n_used + s]; });
}

std::string ActivationCounts::json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return json_locked();
}

std::string ActivationCounts::json_locked() const {
    std::ostringstream o;
    o << "{\"format\": \"strata-moe-counts/1\", \"layers\": [";
    const char* sep = "\n";
    for (const auto& [layer, l] : layers_) {
        o << sep << " {\"layer\": " << layer << ", \"n_expert\": " << l.decode.size() << ", \"decode_calls\": "
          << l.decode_calls << ", \"batch_calls\": " << l.batch_calls << ", \"batch_tokens\": " << l.batch_tokens;
        for (const auto& [name, v] : {std::make_pair("decode", &l.decode), std::make_pair("batch", &l.batch)}) {
            o << ", \"" << name << "\": [";
            for (size_t e = 0; e < v->size(); ++e) o << (e ? ", " : "") << (*v)[e];
            o << "]";
        }
        o << "}";
        sep = ",\n";
    }
    o << "]}\n";
    return o.str();
}

void ActivationCounts::write() const {
    std::lock_guard<std::mutex> lock(mutex_);
    write_locked();
}

void ActivationCounts::write_locked() const {
    const std::string tmp = path_ + ".tmp";
    const std::string text = json_locked();
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (f == nullptr) {
        GGML_ABORT("strata-moe: STRATA_MOE_COUNTS: cannot write %s: %s", tmp.c_str(), std::strerror(errno));
    }
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    const int err = errno;
    if (std::fclose(f) != 0 || !ok) {
        GGML_ABORT("strata-moe: STRATA_MOE_COUNTS: writing %s failed: %s", tmp.c_str(),
                   std::strerror(ok ? errno : err));
    }
    if (std::rename(tmp.c_str(), path_.c_str()) != 0) {
        GGML_ABORT("strata-moe: STRATA_MOE_COUNTS: cannot rename %s to %s: %s", tmp.c_str(), path_.c_str(),
                   std::strerror(errno));
    }
}

namespace {

// The process's counter (null: STRATA_MOE_COUNTS unset). Never freed: the exit handler writes it last.
ActivationCounts* g_counts = nullptr;

void write_at_exit() { g_counts->write(); }

ActivationCounts* process_counts() {
    static ActivationCounts* const counts = [] () -> ActivationCounts* {
        const char* path = std::getenv("STRATA_MOE_COUNTS");
        if (path == nullptr || *path == '\0') return nullptr;
        int64_t every = 4096;
        const char* env = std::getenv("STRATA_MOE_COUNTS_EVERY");
        if (env != nullptr && *env != '\0') {
            char* end = nullptr;
            errno = 0;
            const long long n = std::strtoll(env, &end, 10);
            if (errno != 0 || *end != '\0' || n < 1) {
                GGML_ABORT("strata-moe: STRATA_MOE_COUNTS_EVERY=%s: expected a whole number >= 1", env);
            }
            every = n;
        }
        g_counts = new ActivationCounts(path, every);
        std::atexit(write_at_exit);
        std::fprintf(stderr, "strata-moe: STRATA_MOE_COUNTS=%s: per-(layer, expert) activation counts written every "
                     "%lld calls and at exit\n", path, (long long) every);
        return g_counts;
    }();
    return counts;
}

}  // namespace

void counts_record(const Call& c) {
    ActivationCounts* const counts = process_counts();
    if (counts != nullptr) counts->record(c.layer, c.n_expert, c.ids);
}

}  // namespace strata::dsv41
