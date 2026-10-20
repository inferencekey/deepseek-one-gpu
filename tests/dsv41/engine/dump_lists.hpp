// tests/dsv41/engine/dump_lists.hpp - the op-dump case lists the engine's (a) tests read (tools/dsv41/engine/
// attention_cases.py, dense_cases.py): per dump a '# DIR: MODEL' header, then `model PATH`, `ubatch U POS0 N`,
// optionally `tokens U ID ...`, and `file OP OCC LAYER U DTYPE NE0 NE1 NE2 NE3 PATH` lines (raw little-endian
// files, ggml ne order). Shared by dsv41_engine_attention_test and dsv41_engine_dense_test.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace dumps {

[[noreturn]] inline void fail(const std::string& m) {
    std::fprintf(stderr, "dump lists: %s\n", m.c_str());
    std::exit(2);
}

struct DumpFile {
    std::string dtype, path;
    int64_t ne[4] = {1, 1, 1, 1};
    int64_t count() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};
struct Ubatch {
    int u = 0;
    int64_t pos0 = 0, T = 0;
    std::vector<int32_t> tokens;   ///< from a `tokens` line (dense_cases.py), empty otherwise
};
struct Dump {
    std::string header, model;
    std::vector<Ubatch> ubs;
    std::map<std::tuple<std::string, int, int, int>, DumpFile> files;   // (op, occ, layer, ubatch)
    const DumpFile* find(const std::string& op, int occ, int layer, int u) const {
        const auto it = files.find({op, occ, layer, u});
        return it == files.end() ? nullptr : &it->second;
    }
    const DumpFile& get(const std::string& op, int occ, int layer, int u) const {
        const DumpFile* f = find(op, occ, layer, u);
        if (!f) fail(header + ": no " + op + "." + std::to_string(occ) + " of layer " + std::to_string(layer) + " ubatch " + std::to_string(u));
        return *f;
    }
};

inline std::vector<Dump> read_lists(const std::vector<std::string>& paths) {
    std::vector<Dump> out;
    for (const auto& path : paths) {
        std::ifstream f(path);
        if (!f) fail("cannot open " + path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line.rfind("# ", 0) == 0) { out.emplace_back(); out.back().header = line.substr(2); continue; }
            if (out.empty()) fail(path + ": a line before any '# DIR: MODEL' header");
            Dump& d = out.back();
            std::istringstream in(line);
            std::string tag;
            in >> tag;
            if (tag == "model") {
                in >> d.model;
            } else if (tag == "ubatch") {
                Ubatch u;
                in >> u.u >> u.pos0 >> u.T;
                d.ubs.push_back(u);
            } else if (tag == "tokens") {
                int u = -1;
                in >> u;
                Ubatch* ub = nullptr;
                for (Ubatch& x : d.ubs)
                    if (x.u == u) ub = &x;
                if (!ub) fail(path + ": tokens of an unlisted ubatch: " + line);
                for (int32_t t; in >> t;) ub->tokens.push_back(t);
                if ((int64_t) ub->tokens.size() != ub->T) fail(path + ": ubatch " + std::to_string(u) + " lists " + std::to_string(ub->tokens.size()) + " tokens of " + std::to_string(ub->T));
                continue;
            } else if (tag == "file") {
                std::string op;
                int occ, layer, u;
                DumpFile df;
                in >> op >> occ >> layer >> u >> df.dtype >> df.ne[0] >> df.ne[1] >> df.ne[2] >> df.ne[3] >> df.path;
                d.files[{op, occ, layer, u}] = df;
            } else {
                fail(path + ": bad line: " + line);
            }
            if (!in) fail(path + ": bad line: " + line);
        }
    }
    if (out.empty()) fail("no dump in the case lists");
    return out;
}

template <typename T> std::vector<T> read_raw(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open " + path);
    f.seekg(0, std::ios::end);
    const size_t bytes = (size_t) f.tellg();
    if (bytes != count * sizeof(T))
        fail(path + ": " + std::to_string(bytes) + " bytes, expected " + std::to_string(count * sizeof(T)));
    f.seekg(0);
    std::vector<T> v(count);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) bytes);
    return v;
}
inline std::vector<float> load_f32(const DumpFile& f) { return read_raw<float>(f.path, (size_t) f.count()); }
inline std::vector<uint16_t> load_f16(const DumpFile& f) { return read_raw<uint16_t>(f.path, (size_t) f.count()); }
inline std::vector<int32_t> load_i32(const DumpFile& f) { return read_raw<int32_t>(f.path, (size_t) f.count()); }

}  // namespace dumps
