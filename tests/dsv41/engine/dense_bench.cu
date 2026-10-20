// tests/dsv41/engine/dense_bench.cu - bench of the dense ops at real dims (E 5120, 384 experts, 6 used,
// FF 2304, vocab 129280) on a synthetic model (dense_synth.cuh; GPU, no model): the ms of each op for a 4096-token
// chunk and for one decode token (DenseTimes, CUDA events around each call), the engram gather from pinned host RAM
// (host copy GB/s and the copy to the device), the head for one output row, the router with its GEMV and with
// cuBLAS bf16 at 1 to 8 tokens, and the decode sum of every non-attention, non-routed-expert op over the 40 layers
// plus the embedding and the head (per layer two hc_mixes, two hc_collapse_norm, two hc_expand, the router and the
// shared expert; two engram layers).
//
//   dsv41_engine_dense_bench [--reps N] [--chunk T] [--table-rows N] [--mmq]
//
// Defaults: 7 reps (the median is reported), chunk 4096, an engram table of 4M rows (336 MiB pinned; the real tables
// are 384M rows, and a gather's cost depends on the rows it touches, not on the table's size, beyond the cache),
// QPrefill::DequantF16 (--mmq: the MMQ option). The workspace line includes qmatmul's dequant scratch.
#include "dense_synth.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// the median over reps of the field of DenseTimes `f` after calling `op`
double time_op(int reps, e::DenseEngine& eng, const std::function<void()>& op, float e::DenseTimes::*f) {
    std::vector<double> v;
    op();   // warm
    for (int i = 0; i < reps; ++i) {
        op();
        v.push_back(eng.times().*f);
    }
    return median(v);
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 7;
    int64_t chunk = 4096, table_rows = 1 << 22;
    bool mmq = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) chunk = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--table-rows") == 0 && i + 1 < argc) table_rows = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--mmq") == 0) mmq = true;
        else {
            std::fprintf(stderr, "usage: dsv41_engine_dense_bench [--reps N] [--chunk T] [--table-rows N] [--mmq]\n");
            return 2;
        }
    }
    if (reps < 1 || chunk < 9 || chunk > 65535 || table_rows < 65536) {
        std::fprintf(stderr, "dsv41_engine_dense_bench: reps >= 1, chunk 9..65535, table rows >= 65536\n");
        return 2;
    }
    int dev_id = 0;
    cudaDeviceProp p{};
    synth::ck(cudaGetDevice(&dev_id), "cudaGetDevice");
    synth::ck(cudaGetDeviceProperties(&p, dev_id), "cudaGetDeviceProperties");
    std::printf("device: %s, sm_%d%d, %.1f GiB; %d reps, chunk %lld, %s prefill\n", p.name, p.major, p.minor,
                (double) p.totalGlobalMem / (1 << 30), reps, (long long) chunk, mmq ? "MMQ" : "dequant + cuBLAS");
    const e::DenseDims d = e::dense_dims_real();
    dsynth::Model m;
    const auto t_gen = std::chrono::steady_clock::now();
    dsynth::make_model(m, d, 31, table_rows, true);
    std::printf("synthetic model and engram table (%lld rows, %.0f MiB pinned) in %.1f s\n", (long long) table_rows,
                (double) table_rows * 84 / (1 << 20), std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen).count());
    e::DenseEngine eng;
    e::EngramGather gather;
    std::string err;
    const int threads = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    if (!eng.init(d, chunk, 1, 1024, nullptr, err, mmq ? e::QPrefill::Mmq : e::QPrefill::DequantF16) ||
        !gather.init(chunk * d.engram_cols(), 84, threads, err)) {
        std::fprintf(stderr, "dsv41_engine_dense_bench: %s\n", err.c_str());
        return 1;
    }
    eng.set_timing(true);
    std::printf("dense workspace %.1f MiB at %lld tokens (engram sub-chunks of %lld), gather buffers %.1f MiB, %d host copy threads\n",
                (double) eng.workspace_bytes() / (1 << 20), (long long) chunk, (long long) eng.engram_chunk(),
                (double) gather.device_bytes() / (1 << 20), threads);
    const int64_t E = d.n_embd, X = (int64_t) e::kHc * E, C = d.engram_cols(), K = d.n_expert_used;
    std::vector<int32_t> tok((size_t) chunk);
    {
        std::mt19937 rng(7);
        for (auto& t : tok) t = (int32_t) (rng() % (uint32_t) d.n_vocab);
    }
    int32_t* tok_dev = synth::upload(tok);
    float* x = synth::dev_alloc<float>((size_t) (chunk * X));
    float* pre = synth::dev_alloc<float>((size_t) (chunk * e::kHc));
    float* post = synth::dev_alloc<float>((size_t) (chunk * e::kHc));
    float* comb = synth::dev_alloc<float>((size_t) (chunk * 16));
    float* h = synth::dev_alloc<float>((size_t) (chunk * E));
    float* y = synth::dev_alloc<float>((size_t) (chunk * E));
    int32_t* ids = synth::dev_alloc<int32_t>((size_t) (chunk * K));
    float* wts = synth::dev_alloc<float>((size_t) (chunk * K));
    float* logits = synth::dev_alloc<float>((size_t) d.n_vocab);
    synth::ck(cudaMemset(y, 0, (size_t) (chunk * E) * 4), "cudaMemset");
    std::vector<int32_t> rows((size_t) (chunk * C));
    e::engram_rows(m.hash, 0, tok.data(), chunk, nullptr, 0, rows.data());

    struct Row { const char* op; double prefill = 0, decode = 0; };
    std::vector<Row> table;
    double gather_host[2] = {0, 0}, gather_h2d[2] = {0, 0};
    for (int pass = 0; pass < 2; ++pass) {
        const int64_t T = pass == 0 ? chunk : 1;
        auto put = [&](const char* op, double ms) {
            auto it = std::find_if(table.begin(), table.end(), [&](const Row& r) { return std::strcmp(r.op, op) == 0; });
            if (it == table.end()) { table.push_back(Row{op}); it = table.end() - 1; }
            (pass == 0 ? it->prefill : it->decode) = ms;
        };
        put("embedding", time_op(reps, eng, [&] { e::dense_embed(eng, m.token_embd.view(), tok_dev, T, x, pre); }, &e::DenseTimes::embed_ms));
        // the gather: host copy and the copy to the device, then the device half alone
        {
            std::vector<double> hv, dv;
            cudaEvent_t a, b;
            synth::ck(cudaEventCreate(&a), "cudaEventCreate");
            synth::ck(cudaEventCreate(&b), "cudaEventCreate");
            for (int i = 0; i <= reps; ++i) {
                synth::ck(cudaEventRecord(a, nullptr), "cudaEventRecord");
                gather.gather(m.table, rows.data(), T * C, nullptr);
                synth::ck(cudaEventRecord(b, nullptr), "cudaEventRecord");
                synth::ck(cudaEventSynchronize(b), "cudaEventSynchronize");
                float ms = 0;
                synth::ck(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
                if (i == 0) continue;
                hv.push_back(gather.last_host_ms());
                dv.push_back((double) ms - gather.last_host_ms());
            }
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            gather_host[pass] = median(hv);
            gather_h2d[pass] = std::max(0.0, median(dv));
            put("engram gather (host copy)", gather_host[pass]);
            put("engram gather (to the device)", gather_h2d[pass]);
        }
        const void* rows_dev = gather.gather(m.table, rows.data(), T * C, nullptr);
        put("engram (device: dequant, wkv, gate)", time_op(reps, eng, [&] { e::engram_forward(eng, m.eng, rows_dev, T, x); }, &e::DenseTimes::engram_ms));
        put("hc_mixes (norm, hc_fn, Sinkhorn)", time_op(reps, eng, [&] { e::hc_mixes(eng, m.hc[0], x, T, pre, post, comb); }, &e::DenseTimes::mixes_ms));
        put("hc_collapse_norm", time_op(reps, eng, [&] { e::hc_collapse_norm(eng, x, pre, m.attn_norm.dev, T, h); }, &e::DenseTimes::collapse_ms));
        put("hc_expand", time_op(reps, eng, [&] { e::hc_expand(eng, y, x, post, comb, T, x); }, &e::DenseTimes::expand_ms));
        put("router", time_op(reps, eng, [&] { e::dense_router(eng, m.router, h, T, ids, wts); }, &e::DenseTimes::router_ms));
        put("shared expert", time_op(reps, eng, [&] { e::dense_shared_expert(eng, m.sh, h, T, y); }, &e::DenseTimes::shared_ms));
        if (pass == 1)
            put("head (1 row, q6_K output)", time_op(reps, eng, [&] { e::dense_head(eng, x, pre, 1, m.output_norm.dev, m.output.view(), logits); }, &e::DenseTimes::head_ms));
    }
    std::printf("\n%-40s %14s %12s\n", "op (real dims)", "chunk ms", "decode ms");
    for (const Row& r : table) std::printf("%-40s %14.4f %12.4f\n", r.op, r.prefill, r.decode);
    const double rows_bytes[2] = {(double) (chunk * C * 84), (double) (C * 84)};
    for (int pass = 0; pass < 2; ++pass)
        std::printf("engram gather, %s: %lld rows (%.2f MiB): host copy %.3f ms (%.2f GB/s from pinned RAM), to the device %.3f ms (%.2f GB/s)\n",
                    pass == 0 ? "chunk" : "decode", (long long) (pass == 0 ? chunk * C : C), rows_bytes[pass] / (1 << 20),
                    gather_host[pass], rows_bytes[pass] / (gather_host[pass] * 1e6), gather_h2d[pass],
                    gather_h2d[pass] > 0 ? rows_bytes[pass] / (gather_h2d[pass] * 1e6) : 0.0);
    auto dec = [&](const char* op) {
        for (const Row& r : table)
            if (std::strcmp(r.op, op) == 0) return r.decode;
        return 0.0;
    };
    const double layer = 2 * (dec("hc_mixes (norm, hc_fn, Sinkhorn)") + dec("hc_collapse_norm") + dec("hc_expand")) + dec("router") + dec("shared expert");
    const double engram = dec("engram gather (host copy)") + dec("engram gather (to the device)") + dec("engram (device: dequant, wkv, gate)");
    const double total = 40 * layer + 2 * engram + dec("embedding") + dec("head (1 row, q6_K output)");
    std::printf("decode, one token, every non-attention non-routed-expert op: %.3f ms per layer x 40 + engram %.3f ms x 2 + "
                "embedding + head = %.2f ms (reported, not gated; each op timed alone with its launch and event sync)\n",
                layer, engram, total);

    // the router's two matmuls on the same tokens: its GEMV against cuBLAS bf16 (each with the bf16 rounding and the
    // selection around it)
    std::printf("\nrouter (ffn_gate_inp 384 x 5120 bf16, 3.75 MiB) with its GEMV and with cuBLAS bf16:\n");
    for (int64_t T : {1, 2, 4, 8}) {
        const double g = time_op(reps, eng, [&] { e::dense_router(eng, m.router, h, T, ids, wts, e::RouterMatmul::Gemv); }, &e::DenseTimes::router_ms);
        const double b = time_op(reps, eng, [&] { e::dense_router(eng, m.router, h, T, ids, wts, e::RouterMatmul::Cublas); }, &e::DenseTimes::router_ms);
        std::printf("  %lld tokens: GEMV %.4f ms, cuBLAS %.4f ms\n", (long long) T, g, b);
    }
    size_t free_b = 0, total_b = 0;
    synth::ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    std::printf("device memory in use at the end: %.1f MiB\n", (double) (total_b - free_b) / (1 << 20));
    return 0;
}
