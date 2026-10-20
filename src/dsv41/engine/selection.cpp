// src/dsv41/engine/selection.cpp - the reference selections of include/strata/dsv41/engine/selection.hpp.
#include "strata/dsv41/engine/selection.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sys/stat.h>

namespace strata::dsv41::engine {
namespace {

[[noreturn]] void failf(const char* fmt, ...) {
    char msg[768];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "dsv41 forced selection: %s\n", msg);
    std::exit(1);
}

bool exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

}  // namespace

SelectionDump SelectionDump::open(const std::string& dir) {
    using nlohmann::json;
    SelectionDump s;
    s.dir_ = dir;
    for (const char* name : {"dump-manifest.json", "engine-manifest.json"}) {
        if (exists(dir + "/" + name)) {
            s.manifest_ = dir + "/" + name;
            break;
        }
    }
    if (s.manifest_.empty()) failf("%s: no dump-manifest.json or engine-manifest.json", dir.c_str());
    std::ifstream f(s.manifest_);
    if (!f) failf("cannot open %s: %s", s.manifest_.c_str(), std::strerror(errno));
    try {
        const json j = json::parse(f);
        s.prompt_ = j.at("prompt_token_ids").get<std::vector<int32_t>>();
        // an engine run forced by another reference feeds that reference's tokens, not its own picks
        const char* fed_key = j.contains("fed_token_ids") ? "fed_token_ids" : "generated_token_ids";
        s.fed_ = j.at(fed_key).get<std::vector<int32_t>>();
        for (const json& ub : j.at("ubatches")) {
            const int64_t u = ub.at("ubatch").get<int64_t>();
            if (u != (int64_t) s.calls_.size()) failf("%s: ubatch %lld listed in place of %zu", s.manifest_.c_str(), (long long) u, s.calls_.size());
            s.calls_.push_back({ub.at("pos0").get<int64_t>(), ub.at("n_tokens").get<int64_t>()});
        }
    } catch (const std::exception& e) {
        failf("%s: %s", s.manifest_.c_str(), e.what());
    }
    if (s.prompt_.empty() || s.calls_.empty()) failf("%s: no prompt or no calls", s.manifest_.c_str());
    return s;
}

int64_t SelectionDump::call_of(int64_t p0, int64_t T) const {
    for (size_t u = 0; u < calls_.size(); ++u) {
        const Call& c = calls_[u];
        if (c.pos0 <= p0 && p0 + T <= c.pos0 + c.n_tokens) return (int64_t) u;
    }
    failf("positions %lld .. %lld are in no call of %s (fewer decode steps, or a chunk across two of its ubatches)",
          (long long) p0, (long long) (p0 + T - 1), manifest_.c_str());
}

namespace {

// a dump file's 4-byte values (i32 or f32), little endian as the dumps write them
template <typename V> std::vector<V> read_values(const std::string& dir, const char* op, int64_t u, int il) {
    static_assert(sizeof(V) == 4, "4-byte values");
    char name[96];
    std::snprintf(name, sizeof name, "/%s.0-L%03d-u%04lld.bin", op, il, (long long) u);
    const std::string path = dir + name;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) failf("cannot open %s: %s", path.c_str(), std::strerror(errno));
    std::vector<V> v;
    V buf[1024];
    size_t n;
    while ((n = std::fread(buf, 4, 1024, f)) > 0) v.insert(v.end(), buf, buf + n);
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) failf("reading %s failed", path.c_str());
    return v;
}

}  // namespace

std::vector<int32_t> SelectionDump::read_i32(const char* op, int64_t u, int il) const {
    return read_values<int32_t>(dir_, op, u, il);
}

std::vector<float> SelectionDump::read_f32(const char* op, int64_t u, int il) const {
    return read_values<float>(dir_, op, u, il);
}

void SelectionDump::moe_input(int64_t u, int il, int64_t p0, int64_t T, int64_t E, std::vector<float>& rows) const {
    const Call& c = calls_[(size_t) u];
    const std::vector<float> all = read_f32("ffn_norm", u, il);
    if ((int64_t) all.size() != c.n_tokens * E)
        failf("ffn_norm of layer %d, call %lld: %zu values, expected %lld tokens x %lld", il, (long long) u, all.size(),
              (long long) c.n_tokens, (long long) E);
    rows.assign(all.begin() + (p0 - c.pos0) * E, all.begin() + (p0 - c.pos0 + T) * E);
    for (size_t i = 0; i < rows.size(); ++i)
        if (!std::isfinite(rows[i]))
            failf("ffn_norm of layer %d, call %lld: value %zu of position %lld is not finite", il, (long long) u,
                  i % (size_t) E, (long long) (p0 + (int64_t) (i / (size_t) E)));
}

void SelectionDump::router(int64_t u, int il, int64_t p0, int64_t T, int K, int n_expert, std::vector<int32_t>& ids) const {
    const Call& c = calls_[(size_t) u];
    const std::vector<int32_t> all = read_i32("ffn_moe_topk", u, il);
    if ((int64_t) all.size() != c.n_tokens * K)
        failf("ffn_moe_topk of layer %d, call %lld: %zu ids, expected %lld tokens x %d", il, (long long) u, all.size(),
              (long long) c.n_tokens, K);
    ids.assign(all.begin() + (p0 - c.pos0) * K, all.begin() + (p0 - c.pos0 + T) * K);
    for (int64_t t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i) {
            const int32_t e = ids[(size_t) (t * K + i)];
            if (e < 0 || e >= n_expert) failf("ffn_moe_topk of layer %d, call %lld, token %lld: expert %d of %d", il, (long long) u, (long long) t, e, n_expert);
            for (int j = 0; j < i; ++j)
                if (ids[(size_t) (t * K + j)] == e) failf("ffn_moe_topk of layer %d, call %lld, token %lld: expert %d twice", il, (long long) u, (long long) t, e);
        }
}

void SelectionDump::index_rows(int64_t u, int il, int64_t p0, int64_t T, int ratio, int top_k, int64_t n_k,
                               std::vector<int32_t>& rows) const {
    const Call& c = calls_[(size_t) u];
    const std::vector<int32_t> ref = read_i32("idx_top_k", u, il);
    if (ref.empty() || (int64_t) ref.size() % c.n_tokens)
        failf("idx_top_k of layer %d, call %lld: %zu rows for %lld tokens", il, (long long) u, ref.size(), (long long) c.n_tokens);
    const int64_t k = (int64_t) ref.size() / c.n_tokens, t0 = p0 - c.pos0;
    rows.assign((size_t) (T * top_k), -1);
    std::vector<int32_t> sel;
    for (int64_t t = 0; t < T; ++t) {
        // indexer.hpp's indexer_visible: the rows of the blocks complete at the position
        const int64_t vis = std::min<int64_t>(n_k, (p0 + t + 1) / ratio);
        sel.clear();
        for (int64_t i = 0; i < k; ++i) {
            const int32_t j = ref[(size_t) ((t0 + t) * k + i)];
            if (j >= 0 && j < vis) sel.push_back(j);
        }
        std::sort(sel.begin(), sel.end());
        sel.erase(std::unique(sel.begin(), sel.end()), sel.end());
        if ((int64_t) sel.size() > top_k)
            failf("idx_top_k of layer %d, call %lld, token %lld: %zu visible rows, top-k %d", il, (long long) u, (long long) t,
                  sel.size(), top_k);
        // fewer than min(top_k, vis) rows would attend less than the reference did through its masked cells: refuse
        if ((int64_t) sel.size() < std::min<int64_t>(top_k, vis))
            failf("idx_top_k of layer %d, call %lld, token %lld: %zu of the %lld visible rows, top-k %d", il, (long long) u,
                  (long long) t, sel.size(), (long long) vis, top_k);
        std::copy(sel.begin(), sel.end(), rows.begin() + t * top_k);
    }
}

}  // namespace strata::dsv41::engine
