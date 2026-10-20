// src/dsv41/engine/qmatmul_mmq.cu - qmatmul's prefill path on the fork's ggml-cuda MMQ (MIT), see qmatmul_mmq.hpp.
// The pattern of Strata's src/prefill/moe_mmq.cu (Context::run, fits): mul_mat_q_case from the fork's mmq.cuh with
// the q2_K, q3_K and q6_K instances compiled into this library (src/dsv41/engine/CMakeLists.txt), and
// src/prefill/ggml_cuda_host.cu for the host symbols of ggml-cuda.cu they reference. Dense matrices take no expert
// ids; attn_output_a's groups are MMQ channels.
#include "qmatmul_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::dsv41::engine::mmq {

void* context_new() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
    return new ggml_backend_cuda_context(dev);
}

void context_delete(void* ctx) { delete static_cast<ggml_backend_cuda_context*>(ctx); }

int ds_layout(int type) { return (int) mmq_get_q8_1_ds_layout((ggml_type) type); }

int tile_bytes() { return (int) sizeof(block_q8_1_mmq); }

bool fits(int type, int64_t w_rows) {
    // mul_mat_q_case's choice (moe_mmq.cu fits): the fallback configs when the rows are not a multiple of 128, then
    // a tile size whose config exists for the card and fits its shared memory
    const bool fallback = w_rows % 128 != 0;
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) type, J, fallback, cc);
            any = c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= info.devices[id].smpbo;
        }
        if (!any) return false;
    }
    return info.device_count > 0;
}

void run(void* ctx, int type, const void* w, int64_t k, int64_t rows_g, int groups, const void* acts, int64_t T,
         float* y, int64_t ldy, cudaStream_t stream) {
    const int64_t bpr = k / QK_K;
    // a channel's activations start (k / 128) column tiles of T blocks after the previous one's (int units)
    const int64_t y_channel = k / QK8_1_MMQ * T * (int64_t) (sizeof(block_q8_1_mmq) / sizeof(int));
    const mmq_args a = {(const char*) w, (ggml_type) type, (const int*) acts, nullptr, nullptr, y, nullptr,
                        k, rows_g, T, bpr, T, ldy,
                        groups, groups, rows_g * bpr, y_channel, rows_g,
                        1, 1, 0, 0, 0,
                        T, T};
    auto& c = *static_cast<ggml_backend_cuda_context*>(ctx);
    switch ((ggml_type) type) {
        case GGML_TYPE_Q2_K: mul_mat_q_case<GGML_TYPE_Q2_K>(c, a, stream); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(c, a, stream); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(c, a, stream); break;
        default:
            std::fprintf(stderr, "dsv41 qmatmul: MMQ has no instance for type %d here\n", type);
            std::exit(1);
    }
}

}  // namespace strata::dsv41::engine::mmq
