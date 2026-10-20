// tests/dsv41/engine/attention_bench.cu - bench of the attention layer at real dims (64 heads x 512, top-k 512,
// window 128) on synthetic weights and a synthetic history (attention_synth.cuh; GPU, no model): per layer role, the
// ms of a 4096-token prefill chunk ending at each prefill context and of one decode token at each decode context,
// split into the call's stages (AttnTimes: projections, compressor, indexer, attention, output), and the device
// bytes: the whole model's KV cache (all 40 layers, the real roles, at the largest context), the engine's workspace,
// the benched layers' weights, one Full layer's attention weights against their f16 size, and the device memory in
// use at the end (what the 48 GB budget is checked against).
//
//   dsv41_engine_attention_bench [--reps N] [--prefill-ctx A,B,..] [--decode-ctx A,B,..] [--chunk T] [--f16 | --mmq]
//                                [--peak-tflops F]
//
// Defaults: 5 reps (the median is reported, and the minimum), prefill contexts 16384 and 262144 (the chunk is the
// last T = 4096 positions of the context), decode contexts 4096, 65536 and 262144 (the token is the context's last
// position), Reuse Mode. The roles are the real layers 0 (window), 2 (Full, ratio 2), 3 (Reuse, ratio 2), 20 (Full,
// ratio 1), 21 (Reuse, ratio 1) and 24 (Reindex), run in that order in every chunk (a reader needs its source's
// selection of the same chunk). The projections run on q2_K weights through qmatmul (MMVQ for decode, the
// dequant + cuBLAS f16 path for a prefill chunk, or with --mmq its MMQ option), or with --f16 on cuBLAS f16 GEMMs of
// f16 weights (the f16 test path, for comparison: the acceptance wants q2_K prefill within 1.1x of it per role).
// The workspace line includes qmatmul's dequant scratch and cuBLAS workspace.
//
// each prefill table is followed by its split by op (AttnTimes::op_ms, the
// AttnOp list: every projection, rope, the compressor's pieces, the selection, the attention, the de-rope, wo_a,
// wo_b), and the real 40 layers' mean per layer, each layer taking its role's row (window: layers 0-1; Full r2: 2,
// 8, 14; Reuse r2: the other 15 of 3-19; Full r1: 20; Reindex: 24, 28, 32, 36; Reuse r1: the other 15 of 21-39).
// With a 16384-token chunk that mean is the target's (<= 30 ms per layer at 16K, a PASS or FAIL line; the dev
// pod runs `--chunk 16384 --prefill-ctx 16384 --decode-ctx 16384`).
//
// each prefill table is also followed by the attention's roofline per role: its
// queries x heads x keys attended x D (the keys a query attends: its window positions and its selected rows, the
// visible ones in min(top_k, (p + 1) / ratio)), the FLOPs (4 per product: q.k and p.v, a multiply and an add each),
// the attend op's ms and its TFLOP/s against the device's published dense FP16 tensor peak (--peak-tflops, the pod
// script's table: A100 312; 0 leaves the percentage out); and the selection's scorer floor per index source (queries
// x visible keys x idx_heads x idx_dim x 2 FLOPs at that peak), the select op's ms beside it. At 16K the attend
// (<= 12 ms) and select (<= 4 ms) targets of every role get PASS or FAIL lines.
#include "attention_synth.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

std::vector<int64_t> parse_list(const char* s) {
    std::vector<int64_t> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        const long long x = std::strtoll(p, &end, 10);
        if (end == p || x <= 0) {
            std::fprintf(stderr, "dsv41_engine_attention_bench: bad list '%s'\n", s);
            std::exit(2);
        }
        v.push_back(x);
        p = *end == ',' ? end + 1 : end;
    }
    return v;
}

struct Role {
    const char* name;
    int layer;       // the real layer
    int synth;       // its weights in synth::Model (synth_layers' order)
};

e::AttnTimes median(std::vector<e::AttnTimes> v) {
    std::sort(v.begin(), v.end(), [](const e::AttnTimes& a, const e::AttnTimes& b) { return a.total_ms < b.total_ms; });
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 5;
    int64_t chunk = 4096;
    double peak = 0;
    bool f16 = false, mmq = false;
    std::vector<int64_t> prefill_ctx = {16384, 262144}, decode_ctx = {4096, 65536, 262144};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--f16") { f16 = true; continue; }
        if (a == "--mmq") { mmq = true; continue; }
        if (i + 1 >= argc) { std::fprintf(stderr, "dsv41_engine_attention_bench: %s needs a value\n", a.c_str()); return 2; }
        const char* v = argv[++i];
        if (a == "--reps") reps = std::max(1, std::atoi(v));
        else if (a == "--prefill-ctx") prefill_ctx = parse_list(v);
        else if (a == "--decode-ctx") decode_ctx = parse_list(v);
        else if (a == "--chunk") chunk = std::atoll(v);
        else if (a == "--peak-tflops") peak = std::atof(v);
        else { std::fprintf(stderr, "dsv41_engine_attention_bench: unknown option %s\n", a.c_str()); return 2; }
    }
    int64_t max_ctx = 0;
    for (int64_t c : prefill_ctx) {
        if (c < chunk) { std::fprintf(stderr, "dsv41_engine_attention_bench: prefill context %lld below the chunk %lld\n", (long long) c, (long long) chunk); return 2; }
        max_ctx = std::max(max_ctx, c);
    }
    for (int64_t c : decode_ctx) max_ctx = std::max(max_ctx, c);

    int dev = 0;
    cudaDeviceProp p{};
    synth::ck(cudaGetDevice(&dev), "cudaGetDevice");
    synth::ck(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
    size_t free0 = 0, total = 0;
    synth::ck(cudaMemGetInfo(&free0, &total), "cudaMemGetInfo");
    std::printf("device: %s, sm_%d%d, %d SMs, %.1f GiB (%.1f GiB free at start)\n", p.name, p.major, p.minor,
                p.multiProcessorCount, (double) total / (1 << 30), (double) free0 / (1 << 30));

    const e::AttnDims d = e::attn_dims_real();
    synth::Model m;
    synth::make_model(m, d, 4321, f16 ? e::kQTypeF16 : e::kQTypeQ2K);
    size_t weight_bytes = 0, full_bytes = 0, full_f16 = 0;
    for (const synth::Matrix* mm : {&m.wq_a, &m.wq_b, &m.wkv, &m.wo_a, &m.wo_b}) {
        weight_bytes += mm->host.size();
        full_bytes += mm->host.size();
        full_f16 += mm->f16_bytes();
    }
    for (size_t i = 0; i < m.layers.size(); ++i)
        for (const synth::Matrix* mm : {&m.comp_kv[i], &m.comp_gate[i], &m.idx_k[i], &m.idx_q_b[i], &m.idx_proj[i]}) {
            weight_bytes += mm->host.size();
            if (i == 1) {   // synth layer 1: Full ratio 2 with a gate, every attention matrix a layer can have
                full_bytes += mm->host.size();
                full_f16 += mm->f16_bytes();
            }
        }

    const std::vector<e::AttnLayer> layers = e::attn_layers_real();
    e::KvCache kv;
    e::AttnEngine eng;
    std::string err;
    if (!kv.init(d, layers, e::KvCacheConfig{max_ctx, chunk}, err)) { std::fprintf(stderr, "KV cache: %s\n", err.c_str()); return 1; }
    if (!eng.init(d, chunk, max_ctx, nullptr, err, mmq ? e::QPrefill::Mmq : e::QPrefill::DequantF16)) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
    eng.set_timing(true);
    const e::KvBytes& kb = kv.bytes();
    std::printf("KV cache, 40 layers at %lld positions: %.1f MiB (window %.1f, rows %.1f, index keys %.1f, pooling state %.3f, "
                "top-k ids %.1f)\n", (long long) max_ctx, (double) kb.total() / (1 << 20), (double) kb.window / (1 << 20),
                (double) kb.rows / (1 << 20), (double) kb.keys / (1 << 20), (double) kb.carry / (1 << 20), (double) kb.ids / (1 << 20));
    std::printf("engine workspace (chunk %lld, %s prefill): %.1f MiB; benched layers' weights (%s, the q/kv/output projections "
                "shared): %.1f MiB\n", (long long) chunk, mmq ? "MMQ" : "dequant + cuBLAS", (double) eng.workspace_bytes() / (1 << 20),
                e::qtype_name(m.wq_a.type), (double) weight_bytes / (1 << 20));
    std::printf("one Full layer's attention weights (L2: q, kv, output, compressor, gate, indexer): %.1f MiB %s, %.1f MiB as f16 "
                "(%.3f of it)\n", (double) full_bytes / (1 << 20), e::qtype_name(m.wq_a.type), (double) full_f16 / (1 << 20),
                (double) full_bytes / (double) full_f16);

    const Role roles[] = {{"window (L0)", 0, 0}, {"Full r2 (L2)", 2, 1}, {"Reuse r2 (L3)", 3, 2},
                          {"Full r1 (L20)", 20, 3}, {"Reuse r1 (L21)", 21, 5}, {"Reindex (L24)", 24, 4}};
    for (const Role& r : roles)
        if (layers[(size_t) r.layer].role() != m.layers[(size_t) r.synth].role()) { std::fprintf(stderr, "role table mismatch at L%d\n", r.layer); return 1; }

    float* x = synth::upload(synth::make_input(chunk, d.n_embd, 5));
    float* out = synth::dev_alloc<float>((size_t) (chunk * d.n_embd));
    auto run = [&](int64_t hist, int64_t T, const char* what) {
        synth::write_history(kv, hist, (uint64_t) hist + 17);
        std::printf("\n%s: %lld tokens at positions %lld..%lld, %d reps (median, min of the total)\n", what, (long long) T,
                    (long long) hist, (long long) (hist + T - 1), reps);
        std::printf("  %-16s %9s %9s %9s %9s %9s %9s %9s %7s\n", "role", "total", "min", "proj", "compress", "indexer", "attention",
                    "output", "splits");
        std::vector<std::vector<e::AttnTimes>> t(6);
        std::vector<int> splits(6);
        for (int rep = 0; rep < reps; ++rep) {
            for (const Role& r : roles) kv.set_position(r.layer, hist);
            for (size_t i = 0; i < 6; ++i) {
                const Role& r = roles[i];
                const e::AttnLayer& l = layers[(size_t) r.layer];
                if (T == 1) e::attn_decode(eng, l, m.dev[(size_t) r.synth], kv, x, hist, out);
                else e::attn_prefill(eng, l, m.dev[(size_t) r.synth], kv, x, T, hist, out);
                t[i].push_back(eng.times());
                splits[i] = eng.scratch().n_splits;
            }
        }
        double sum = 0;
        for (size_t i = 0; i < 6; ++i) {
            const e::AttnTimes md = median(t[i]);
            float mn = md.total_ms;
            for (const auto& a : t[i]) mn = std::min(mn, a.total_ms);
            sum += md.total_ms;
            std::printf("  %-16s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %7d\n", roles[i].name, md.total_ms, mn, md.proj_ms,
                        md.compress_ms, md.indexer_ms, md.attention_ms, md.output_ms, splits[i]);
        }
        std::printf("  (ms; the six roles together %.3f ms)\n", sum);
        if (T == 1) return;
        // the split by op, and the real 40 layers by role
        std::printf("  by op (median call of each role, ms):\n  %-16s", "role");
        for (int o = 0; o < e::kAttnOps; ++o) std::printf(" %9s", e::attn_op_name((e::AttnOp) o));
        std::printf("\n");
        std::vector<e::AttnTimes> md(6);
        for (size_t i = 0; i < 6; ++i) {
            md[i] = median(t[i]);
            std::printf("  %-16s", roles[i].name);
            for (int o = 0; o < e::kAttnOps; ++o) std::printf(" %9.3f", md[i].op_ms[o]);
            std::printf("\n");
        }
        // each real layer takes its role's row: the window, the Full and Reuse rows by ratio, Reindex
        double real = 0, real_ops[e::kAttnOps] = {};
        for (const e::AttnLayer& l : layers) {
            size_t i = 0;
            switch (l.role()) {
                case e::AttnRole::Window: i = 0; break;
                case e::AttnRole::Full: i = l.ratio == 2 ? 1 : 3; break;
                case e::AttnRole::Reuse: i = l.ratio == 2 ? 2 : 4; break;
                case e::AttnRole::Reindex: i = 5; break;
            }
            real += md[i].total_ms;
            for (int o = 0; o < e::kAttnOps; ++o) real_ops[o] += md[i].op_ms[o];
        }
        const double n = (double) layers.size();
        std::printf("  the real %zu layers by role: %.3f ms per layer (%.1f ms in all); per layer by op:", layers.size(), real / n, real);
        for (int o = 0; o < e::kAttnOps; ++o) std::printf(" %s %.3f", e::attn_op_name((e::AttnOp) o), real_ops[o] / n);
        std::printf("\n");
        // the attention's roofline and the selection's scorer floor per role
        const int ia = (int) e::AttnOp::Attend, is = (int) e::AttnOp::Select;
        std::printf("  attend roofline (queries x heads x keys attended x D, 4 FLOPs per product: q.k and p.v):\n");
        std::printf("  %-16s %8s %6s %10s %5s %10s %10s %9s %9s\n", "role", "queries", "heads", "keys/query", "D", "GFLOP",
                    "attend ms", "TFLOP/s", peak > 0 ? "of peak" : "");
        double flop_all = 0, attend_all = 0, attend_worst = 0, select_worst = 0;
        std::vector<double> flops(6);
        for (size_t i = 0; i < 6; ++i) {
            const e::AttnLayer& l = layers[(size_t) roles[i].layer];
            double keys = 0;
            for (int64_t q = hist; q < hist + T; ++q) {
                keys += (double) std::min<int64_t>(q + 1, d.n_swa);
                if (l.ratio) keys += (double) std::min<int64_t>(d.top_k, (q + 1) / l.ratio);
            }
            flops[i] = 4.0 * (double) d.n_head * (double) d.head_dim * keys;
            const double ms = md[i].op_ms[ia], tf = ms > 0 ? flops[i] / (ms * 1e9) : 0;
            attend_worst = std::max(attend_worst, ms);
            std::printf("  %-16s %8lld %6d %10.1f %5d %10.1f %10.3f %9.1f", roles[i].name, (long long) T, d.n_head,
                        keys / (double) T, d.head_dim, flops[i] / 1e9, ms, tf);
            if (peak > 0) std::printf(" %8.1f%%", 100.0 * tf / peak);
            std::printf("\n");
        }
        for (const e::AttnLayer& l : layers) {
            size_t i = 0;
            switch (l.role()) {
                case e::AttnRole::Window: i = 0; break;
                case e::AttnRole::Full: i = l.ratio == 2 ? 1 : 3; break;
                case e::AttnRole::Reuse: i = l.ratio == 2 ? 2 : 4; break;
                case e::AttnRole::Reindex: i = 5; break;
            }
            flop_all += flops[i];
            attend_all += md[i].op_ms[ia];
        }
        std::printf("  the real %zu layers: %.3f ms of attend per layer, %.1f TFLOP/s", layers.size(), attend_all / n,
                    flop_all / (attend_all * 1e9));
        if (peak > 0) std::printf(" (%.1f%% of the published %.1f)", 100.0 * flop_all / (attend_all * 1e9) / peak, peak);
        std::printf("\n  select (an index source's scores and top-k), its scorer's FLOPs (queries x visible keys x %d heads x %d x 2)%s:\n",
                    d.idx_heads, d.idx_dim, peak > 0 ? " and their time at the peak" : "");
        for (size_t i = 0; i < 6; ++i) {
            const e::AttnLayer& l = layers[(size_t) roles[i].layer];
            if (!l.index_source) continue;
            double keys = 0;
            for (int64_t q = hist; q < hist + T; ++q) keys += (double) ((q + 1) / l.ratio);
            const double fl = 2.0 * keys * d.idx_heads * d.idx_dim, ms = md[i].op_ms[is];
            select_worst = std::max(select_worst, ms);
            std::printf("    %-16s %.1f GFLOP, select %.3f ms (%.1f TFLOP/s of scores)", roles[i].name, fl / 1e9, ms,
                        ms > 0 ? fl / (ms * 1e9) : 0.0);
            if (peak > 0) std::printf(", the scores alone take %.3f ms at %.1f TFLOP/s", fl / (peak * 1e9), peak);
            std::printf("\n");
        }
        if (T == 16384) {
            std::printf("dsv41_engine_attention_bench: E09 target: prefill attention per layer at 16K (a 16384-token chunk at "
                        "positions %lld..%lld, the real 40 layers by role) %.2f ms (<= 30: %s)\n", (long long) hist,
                        (long long) (hist + T - 1), real / n, real / n <= 30.0 ? "PASS" : "FAIL");
            std::printf("dsv41_engine_attention_bench: E09 target: attend per layer at 16K, the slowest role %.2f ms (<= 12: %s; "
                        "the real 40 layers' mean %.2f)\n", attend_worst, attend_worst <= 12.0 ? "PASS" : "FAIL", attend_all / n);
            std::printf("dsv41_engine_attention_bench: E09 target: select per index source layer at 16K, the slowest role %.2f ms "
                        "(<= 4: %s)\n", select_worst, select_worst <= 4.0 ? "PASS" : "FAIL");
        }
    };
    for (int64_t c : prefill_ctx) {
        char what[96];
        std::snprintf(what, sizeof what, "prefill, a %lld-token chunk at context %lld", (long long) chunk, (long long) c);
        run(c - chunk, chunk, what);
    }
    for (int64_t c : decode_ctx) {
        char what[96];
        std::snprintf(what, sizeof what, "decode at context %lld", (long long) c);
        run(c - 1, 1, what);
    }
    size_t free1 = 0;
    synth::ck(cudaMemGetInfo(&free1, &total), "cudaMemGetInfo");
    std::printf("\ndevice memory in use by the bench at the end: %.2f GiB (KV cache, workspace, weights, inputs)\n",
                (double) (free0 - free1) / (1 << 30));
    cudaFree(x);
    cudaFree(out);
    return 0;
}
