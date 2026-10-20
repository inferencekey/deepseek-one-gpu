// include/strata/dsv41/engine/qmatmul_ref.hpp - the CPU reference of qmatmul.hpp (src/dsv41/engine/
// qmatmul_ref.cpp): the engine's activation quantizers bit for bit, and its matmul as the exact value of what the
// kernels compute, summed in double.
//
// What the kernels compute, per output (qmatmul.hpp "Numerics"): with x^ = d * q the dequantized activation of its
// group (d as the layout stores it: f16 in q8_1 and D2S6, f32 in D4) and w^ the weight a block decodes to,
//   q8_1 (MMVQ, every type) and D4 (MMQ q3_K, q6_K): sum over k of w^_k x^_k, w^ the exact value of ggml's dequant
//       (q2_K d * sc * q - dmin * m, q3_K d * (sc - 32) * q, q6_K d * sc * (q - 32), d and dmin f16);
//   D2S6 (MMQ q2_K): per 16-value group with f16(d * sc) and f16(dmin * m) as MMQ's tiles round them,
//       f16(d sc) d64 sum(q2 q8) - f16(dmin m) M, M the group's f16 sum of the raw activations for groups 0..5 of
//       every 128 values and d64 sum(q8) for groups 6 and 7.
// Every integer sum is exact on the GPU, so the GPU differs from this only in its f32 products and sums:
// qmatmul_ref_matmul's abs_scale (the sum of the absolute values of the terms) bounds that.
#pragma once

#include "strata/dsv41/engine/qmatmul.hpp"

#include <cstdint>
#include <vector>

namespace strata::dsv41::engine {

/// Quantized activations on the host, layout-independent: q [T][k], d [T][k / group] the scales as the kernels read
/// them (group 32, or 64 for D2S6), s the stored sums as floats: q8_1 [T][k / 32] (not read by the K-quant dots),
/// D2S6 [T][k / 16] (groups 6 and 7 of each 128 hold none: 0), D4 none.
struct QActsHost {
    QActLayout layout = QActLayout::Q8_1;
    int64_t T = 0, k = 0;
    std::vector<int8_t> q;
    std::vector<float> d, s;
    int64_t group() const { return layout == QActLayout::MmqD2S6 ? 64 : 32; }
};

/// The engine's quantizer (qact_quantize) of x [T][ld] f32, bit for bit. f16 input is its f32 values.
void qref_quantize(const float* x, int64_t ld, int64_t T, int64_t k, QActLayout l, QActsHost& out);
/// The device bytes of qact_quantize (qact_bytes(l, T, k) of them) in host form.
void qref_unpack(const void* bytes, QActLayout l, int64_t T, int64_t k, QActsHost& out);
/// The dequantized activation of value i of row t.
float qref_act(const QActsHost& a, int64_t t, int64_t i);

/// Row r of a quantized matrix (host view) as ggml's dequantize_row_q2_K / _q3_K / _q6_K gives it in f32.
void qref_dequant_row(const QWeightView& w, int64_t r, float* out);

/// y [T][ldy] = qmatmul_q's result for host weights and activations (groups as there), in double, rounded to f32 at
/// the end; abs_scale [T][ldy] (may be null) the sum of the absolute values of the terms. Rows are computed on
/// several threads when the work is large.
void qref_matmul(const QWeightView& w, const QActsHost& a, int groups, float* y, int64_t ldy, double* abs_scale);
/// The same for the listed output columns only (cols of the T rows; a sampled check of a large matrix): y and
/// abs_scale [T][cols.size()].
void qref_matmul_cols(const QWeightView& w, const QActsHost& a, int groups, const std::vector<int64_t>& cols, float* y,
                      double* abs_scale);

/// qmatmul_f16's value (QPath::DequantF16): y [T][ldy] = f16(x) . f16(w^)^T for x [T][ldx] f32 (each value rounded to
/// f16, so f16 input is its own value), w^ qref_dequant_row's value rounded to f16 (the device's dequant, bit for
/// bit), the exact f16 products summed in double; abs_scale [T][ldy] (may be null) the sum of their absolute values.
/// groups as qref_matmul.
void qref_matmul_f16(const QWeightView& w, const float* x, int64_t ldx, int64_t T, int groups, float* y, int64_t ldy,
                     double* abs_scale);
/// The same for the listed output columns: y and abs_scale [T][cols.size()].
void qref_matmul_f16_cols(const QWeightView& w, const float* x, int64_t ldx, int64_t T, int groups,
                          const std::vector<int64_t>& cols, float* y, double* abs_scale);

}  // namespace strata::dsv41::engine
