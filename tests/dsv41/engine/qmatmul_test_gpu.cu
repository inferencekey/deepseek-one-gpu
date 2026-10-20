// tests/dsv41/engine/qmatmul_test_gpu.cu - (b) of dsv41_engine_qmatmul_test: qmatmul on the GPU against its CPU
// reference (qmatmul_ref.hpp), every real dense shape (qmatmul_synth.hpp real_shapes: q2_K, q3_K, q6_K) at T in
// {1, 2, 3, 4, 8, 9, 64, 4096} columns (--quick: 1 and 64), on random blocks (tiny_dsv41.py's generators) and N(0, 1)
// activations with one 8-sigma value per row, on every path: MMVQ at T <= 8, and above both the default dequant +
// cuBLAS f16 path and the MMQ option.
//
// Per shape:
//   dequant     the dequant path's f16 weights (qweight_dequant_f16) of up to 384 rows (spread, the first and the
//               last of each group among them) equal qref_dequant_row's values rounded to f16, bit for bit
// Per (shape, T), MMVQ and MMQ:
//   quantizer   the GPU's activations (qact_quantize, the layout the path reads at T) equal qref_quantize of the same
//               input bit for bit: every q, scale and stored sum, all T rows
//   matmul      y of qmatmul_q against qref_matmul of the GPU's own activations, on every column of up to 8 rows (8
//               spread over T, the first and the last among them) and the dequant check's output rows: |d| <= 2^-12 of
//               the sum of the absolute values of the terms. The integer products are exact on both sides; the GPU's
//               f32 sums of n terms are off by at most n 2^-24 of that sum, n <= K / 8 = 2560 additions at K = 20480
//               (two per 16 values in MMQ, fewer in MMVQ), while a wrong scale, block or column is O(1).
//   f16 input   at T = 9 the MMQ quantizer from f16 bits (the de-rope's path) equals the f32 quantizer of their values.
// Per (shape, T > 8), the dequant path:
//   matmul      qmatmul_f16 on the f16 of the input against qref_matmul_f16 on the same rows and columns: |d| <= K 2^-23
//               (at least 2^-13) of the sum of the absolute products. The f16 weights and activations are the
//               reference's bit for bit and their products exact; cuBLAS sums K products in f32 (a tensor core's
//               accumulation can truncate: at most 2^-23 of the running sum per addition). Twice: with a scratch that
//               holds the matrix whole (36 Mi values; attn_output_a takes the strided batched GEMM) and with one of
//               1 Mi values (row slices for every matrix of more, per group for attn_output_a).
//   f32 input   at T = 9 qmatmul on the f32 input (its own f16 conversion) equals qmatmul_f16 on the host's f16 of it,
//               bit for bit.
#include "qmatmul_synth.hpp"

#include "strata/dsv41/engine/qmatmul_ref.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace e = strata::dsv41::engine;

constexpr double kTolMatmul = 1.0 / 4096;

void ck(cudaError_t r, const char* w) {
    if (r != cudaSuccess) {
        std::fprintf(stderr, "dsv41_engine_qmatmul_test: %s: %s\n", w, cudaGetErrorString(r));
        std::exit(2);
    }
}

std::vector<int64_t> spread(int64_t n, int64_t k) {
    std::vector<int64_t> s;
    if (n <= k) {
        for (int64_t i = 0; i < n; ++i) s.push_back(i);
    } else {
        for (int64_t i = 0; i < k; ++i) s.push_back(i * (n - 1) / (k - 1));
    }
    return s;
}

bool same_acts(const e::QActsHost& a, const e::QActsHost& b, int64_t& differ) {
    differ = 0;
    for (size_t i = 0; i < a.q.size(); ++i) differ += a.q[i] != b.q[i];
    for (size_t i = 0; i < a.d.size(); ++i) differ += std::memcmp(&a.d[i], &b.d[i], 4) != 0;
    for (size_t i = 0; i < a.s.size(); ++i) differ += std::memcmp(&a.s[i], &b.s[i], 4) != 0;
    return differ == 0 && a.q.size() == b.q.size() && a.d.size() == b.d.size() && a.s.size() == b.s.size();
}

struct Ctxs {
    e::QmmContext& f16;         // DequantF16, whole matrices up to 36 Mi values
    e::QmmContext& f16_small;   // DequantF16, 1 Mi values: row slices
    e::QmmContext& mmq;         // Mmq
};
constexpr int64_t kScratchWhole = 36ll << 20, kScratchSmall = 1ll << 20;

// the sampled rows ts of the GPU's activations
e::QActsHost sub_acts(const e::QActsHost& g, const std::vector<int64_t>& ts) {
    e::QActsHost sub;
    sub.layout = g.layout;
    sub.T = (int64_t) ts.size();
    sub.k = g.k;
    const int64_t K = g.k, G = g.group();
    for (int64_t t : ts) {
        sub.q.insert(sub.q.end(), g.q.begin() + t * K, g.q.begin() + (t + 1) * K);
        sub.d.insert(sub.d.end(), g.d.begin() + t * K / G, g.d.begin() + (t + 1) * K / G);
        if (!g.s.empty()) {
            const int64_t S = (int64_t) g.s.size() / g.T;
            sub.s.insert(sub.s.end(), g.s.begin() + t * S, g.s.begin() + (t + 1) * S);
        }
    }
    return sub;
}

// |gpu - ref| over the sampled rows ts and columns cols of y [T][rows] against scale units: worst ratio, count over tol
void compare(const float* y_dev, int64_t rows, const std::vector<int64_t>& ts, const std::vector<int64_t>& cols,
             const std::vector<float>& yr, const std::vector<double>& scale, double tol, double& worst, int64_t& over) {
    worst = 0;
    over = 0;
    std::vector<float> row((size_t) rows);
    for (size_t i = 0; i < ts.size(); ++i) {
        ck(cudaMemcpy(row.data(), y_dev + ts[i] * rows, sizeof(float) * row.size(), cudaMemcpyDeviceToHost), "download y");
        for (size_t c = 0; c < cols.size(); ++c) {
            const double d = std::fabs((double) row[(size_t) cols[c]] - yr[i * cols.size() + c]);
            const double u = scale[i * cols.size() + c];
            const double ratio = u > 0 ? d / u : (d > 0 ? INFINITY : 0);
            worst = std::max(worst, ratio);
            over += !(ratio <= tol);
        }
    }
}

struct Device {
    void* w = nullptr;
    float* x = nullptr;
    uint16_t* xh = nullptr;
    void* acts = nullptr;
    float* y = nullptr;
    float* y2 = nullptr;
    uint16_t* wrow = nullptr;
    ~Device() {
        cudaFree(w);
        cudaFree(x);
        cudaFree(xh);
        cudaFree(acts);
        cudaFree(y);
        cudaFree(y2);
        cudaFree(wrow);
    }
};

bool run_shape(Ctxs& qc, const qsynth::Shape& sh, const std::vector<int64_t>& Ts, uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int64_t K = sh.k * sh.groups, rows_g = sh.rows / sh.groups, Tmax = *std::max_element(Ts.begin(), Ts.end());
    const std::vector<uint8_t> wb = qsynth::random_blocks(sh.type, sh.rows, sh.k, 1.f / std::sqrt((float) sh.k), rng);
    const e::QWeightView wh{sh.type, sh.k, sh.rows, wb.data()};
    Device dv;
    ck(cudaMalloc(&dv.w, wb.size()), "cudaMalloc(w)");
    ck(cudaMemcpy(dv.w, wb.data(), wb.size(), cudaMemcpyHostToDevice), "upload w");
    const e::QWeightView wd{sh.type, sh.k, sh.rows, dv.w};
    ck(cudaMalloc(&dv.x, sizeof(float) * (size_t) (Tmax * K)), "cudaMalloc(x)");
    ck(cudaMalloc(&dv.xh, 2 * (size_t) (Tmax * K)), "cudaMalloc(xh)");
    ck(cudaMalloc(&dv.acts, std::max(e::qact_bytes_max(e::QPrefill::Mmq, Tmax, K), 2 * (size_t) (Tmax * K))), "cudaMalloc(acts)");
    ck(cudaMalloc(&dv.y, sizeof(float) * (size_t) (Tmax * sh.rows)), "cudaMalloc(y)");
    ck(cudaMalloc(&dv.y2, sizeof(float) * (size_t) (9 * sh.rows)), "cudaMalloc(y2)");   // the T = 9 f32-input check
    ck(cudaMalloc(&dv.wrow, 2 * (size_t) sh.k), "cudaMalloc(wrow)");
    // output rows to check: spread over each group, its first and last among them
    std::vector<int64_t> cols;
    for (int g = 0; g < sh.groups; ++g)
        for (int64_t r : spread(rows_g, std::max<int64_t>(2, 384 / sh.groups))) cols.push_back(g * rows_g + r);
    bool ok = true;
    {
        int64_t differ = 0;
        std::vector<uint16_t> got((size_t) sh.k);
        std::vector<float> want((size_t) sh.k);
        for (int64_t r : cols) {
            e::qweight_dequant_f16(wd, r, 1, dv.wrow, nullptr);
            ck(cudaMemcpy(got.data(), dv.wrow, 2 * got.size(), cudaMemcpyDeviceToHost), "download dequant");
            e::qref_dequant_row(wh, r, want.data());
            for (int64_t i = 0; i < sh.k; ++i) differ += got[(size_t) i] != strata::kernels::f16_from_f32(want[(size_t) i]);
        }
        ok = differ == 0;
        std::printf("%s %-36s %-4s dequant to f16, %zu rows x %lld: %s\n", ok ? "ok  " : "FAIL", sh.name, e::qtype_name(sh.type),
                    cols.size(), (long long) sh.k, differ ? ("DIFFERS in " + std::to_string(differ) + " values").c_str() : "bit-exact");
    }
    const double tol_f16 = (double) std::max<int64_t>(1024, sh.k) / 8388608.0;
    for (int64_t T : Ts) {
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) (T * K));
        for (auto& v : x) v = nd(rng);
        for (int64_t t = 0; t < T; ++t) x[(size_t) (t * K + (int64_t) (rng() % (uint64_t) K))] = 8.f;
        ck(cudaMemcpy(dv.x, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice), "upload x");
        const std::vector<int64_t> ts = spread(T, 8);
        // MMVQ at T <= 8, the MMQ option above
        {
            e::QmmContext& qmm = T <= e::kMmvqMaxCols ? qc.f16 : qc.mmq;
            const e::QActLayout l = e::qact_layout(sh.type, T);
            e::QActs a;
            e::qact_quantize(dv.x, nullptr, K, T, K, l, dv.acts, nullptr, a);
            e::qmatmul_q(qmm, wd, a, dv.y, sh.rows, sh.groups, nullptr, sh.name);
            ck(cudaDeviceSynchronize(), "qmatmul");
            // the quantizer, bit for bit
            std::vector<uint8_t> bytes(e::qact_bytes(l, T, K));
            ck(cudaMemcpy(bytes.data(), dv.acts, bytes.size(), cudaMemcpyDeviceToHost), "download acts");
            e::QActsHost gq, rq;
            e::qref_unpack(bytes.data(), l, T, K, gq);
            e::qref_quantize(x.data(), K, T, K, l, rq);
            int64_t differ = 0;
            const bool q_ok = same_acts(gq, rq, differ);
            // f16 input: the same values through the f16 path
            bool h_ok = true;
            int64_t h_differ = 0;
            if (T == 9) {
                std::vector<uint16_t> xh(x.size());
                std::vector<float> xr(x.size());
                for (size_t i = 0; i < x.size(); ++i) {
                    xh[i] = strata::kernels::f16_from_f32(x[i]);
                    xr[i] = strata::kernels::f32_from_f16(xh[i]);
                }
                ck(cudaMemcpy(dv.xh, xh.data(), 2 * xh.size(), cudaMemcpyHostToDevice), "upload xh");
                e::QActs ah;
                e::qact_quantize(nullptr, dv.xh, K, T, K, l, dv.acts, nullptr, ah);
                ck(cudaMemcpy(bytes.data(), dv.acts, bytes.size(), cudaMemcpyDeviceToHost), "download acts (f16)");
                e::QActsHost gh, rh;
                e::qref_unpack(bytes.data(), l, T, K, gh);
                e::qref_quantize(xr.data(), K, T, K, l, rh);
                h_ok = same_acts(gh, rh, h_differ);
            }
            // the matmul, from the GPU's own activations
            const e::QActsHost sub = sub_acts(gq, ts);
            std::vector<float> yr(ts.size() * cols.size());
            std::vector<double> scale(yr.size());
            e::qref_matmul_cols(wh, sub, sh.groups, cols, yr.data(), scale.data());
            double worst = 0;
            int64_t over = 0;
            compare(dv.y, sh.rows, ts, cols, yr, scale, kTolMatmul, worst, over);
            const bool good = q_ok && h_ok && over == 0;
            ok = ok && good;
            std::printf("%s %-36s %-4s T %4lld %-14s %-8s quantizer %s%s, %zu x %zu outputs worst %.3g of the term sum (tolerance %.3g)%s\n",
                        good ? "ok  " : "FAIL", sh.name, e::qtype_name(sh.type), (long long) T, e::qpath_name(qmm.path(T)),
                        e::qact_layout_name(l), q_ok ? "bit-exact" : ("DIFFERS in " + std::to_string(differ) + " fields").c_str(),
                        T == 9 ? (h_ok ? ", f16 input bit-exact" : (", f16 input DIFFERS in " + std::to_string(h_differ)).c_str()) : "",
                        ts.size(), cols.size(), worst, kTolMatmul, over ? (", " + std::to_string(over) + " over").c_str() : "");
        }
        if (T <= e::kMmvqMaxCols) continue;
        // the dequant path on the f16 of x, whole and in slices
        std::vector<uint16_t> xh(x.size());
        std::vector<float> xs(ts.size() * (size_t) K);
        for (size_t i = 0; i < x.size(); ++i) xh[i] = strata::kernels::f16_from_f32(x[i]);
        for (size_t i = 0; i < ts.size(); ++i) std::copy(x.begin() + ts[i] * K, x.begin() + (ts[i] + 1) * K, xs.begin() + (int64_t) i * K);
        ck(cudaMemcpy(dv.xh, xh.data(), 2 * xh.size(), cudaMemcpyHostToDevice), "upload xh");
        std::vector<float> yr(ts.size() * cols.size());
        std::vector<double> scale(yr.size());
        e::qref_matmul_f16_cols(wh, xs.data(), K, (int64_t) ts.size(), sh.groups, cols, yr.data(), scale.data());
        for (e::QmmContext* qmm : {&qc.f16, &qc.f16_small}) {
            e::qmatmul_f16(*qmm, wd, dv.xh, K, T, dv.y, sh.rows, sh.groups, nullptr, sh.name);
            ck(cudaDeviceSynchronize(), "qmatmul_f16");
            double worst = 0;
            int64_t over = 0;
            compare(dv.y, sh.rows, ts, cols, yr, scale, tol_f16, worst, over);
            // f32 input: qmatmul's own conversion gives the same result bit for bit
            bool x_ok = true;
            if (T == 9 && qmm == &qc.f16) {
                e::qmatmul(*qmm, wd, dv.x, K, T, dv.y2, sh.rows, dv.acts, sh.groups, nullptr, sh.name);
                ck(cudaDeviceSynchronize(), "qmatmul");
                std::vector<float> a((size_t) (T * sh.rows)), b(a.size());
                ck(cudaMemcpy(a.data(), dv.y, sizeof(float) * a.size(), cudaMemcpyDeviceToHost), "download y");
                ck(cudaMemcpy(b.data(), dv.y2, sizeof(float) * b.size(), cudaMemcpyDeviceToHost), "download y2");
                x_ok = std::memcmp(a.data(), b.data(), sizeof(float) * a.size()) == 0;
            }
            const bool whole = sh.rows * sh.k <= qmm->scratch_elems();
            const bool good = over == 0 && x_ok;
            ok = ok && good;
            std::printf("%s %-36s %-4s T %4lld %-14s %-8s %s%s, %zu x %zu outputs worst %.3g of the term sum (tolerance %.3g)%s\n",
                        good ? "ok  " : "FAIL", sh.name, e::qtype_name(sh.type), (long long) T, e::qpath_name(qmm->path(T)), "f16",
                        whole ? "whole matrix" : ("slices of " + std::to_string(qmm->scratch_elems() / sh.k) + " rows").c_str(),
                        T == 9 && qmm == &qc.f16 ? (x_ok ? ", f32 input bit-exact" : ", f32 input DIFFERS") : "", ts.size(), cols.size(),
                        worst, tol_f16, over ? (", " + std::to_string(over) + " over").c_str() : "");
        }
    }
    return ok;
}

}  // namespace

int run_gpu(bool quick) {
    int dev = 0;
    cudaDeviceProp p{};
    ck(cudaGetDevice(&dev), "cudaGetDevice");
    ck(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
    std::printf("device: %s, sm_%d%d, %.1f GiB\n", p.name, p.major, p.minor, (double) p.totalGlobalMem / (1 << 30));
    e::QmmContext f16, f16_small, mmq;
    std::string err;
    if (!f16.init(err, e::QPrefill::DequantF16, kScratchWhole) || !f16_small.init(err, e::QPrefill::DequantF16, kScratchSmall) ||
        !mmq.init(err, e::QPrefill::Mmq, 0)) {
        std::fprintf(stderr, "dsv41_engine_qmatmul_test: %s\n", err.c_str());
        return 1;
    }
    Ctxs qc{f16, f16_small, mmq};
    const std::vector<int64_t> Ts = quick ? std::vector<int64_t>{1, 64} : std::vector<int64_t>{1, 2, 3, 4, 8, 9, 64, 4096};
    bool ok = true;
    uint64_t seed = 99;
    for (const qsynth::Shape& sh : qsynth::real_shapes()) ok = run_shape(qc, sh, Ts, seed++) && ok;
    std::printf("dsv41_engine_qmatmul_test (b): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
