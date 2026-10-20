// include/strata/dsv41/moe.h - the routed experts of a llama.cpp MoE layer, behind the fork's build_moe_ffn hook.
//
// The fork (third_party/llama-dsv41/0001-strata-moe-hook.patch) builds, with LLAMA_STRATA_MOE=1, one ggml custom op
// in place of the stock expert path (mul_mat_id up/gate, SwiGLU, mul_mat_id down, times the routing weights, sum
// over the slots) and loads strata_moe_forward from libstrata-moe.so to compute it. The library links the fork's own
// libggml-base, libggml-cpu and libggml-cuda, so one ggml lives in the process.
//
// The CPU reference runs the stock CPU path's own ggml-cpu kernels (from_float to the weight type's
// vec_dot_type, vec_dot per row, ggml_vec_swiglu_f32) in the stock order, so its output is bit-identical to the
// stock path on ggml's CPU backend (tests/dsv41/moe_ref_test.cpp has the measured numbers).
//
// The GPU tier: with STRATA_MOE_HOT (a hot-set file from tools/dsv41/oracle/hot_set.py) and
// STRATA_MOE_VRAM_GB (the VRAM budget in GiB) set and a CUDA device present, the library keeps the hot experts of
// every layer in its own VRAM buffer, computes the (token, slot) pairs routed to them with the fork's ggml-cuda
// kernels while the CPU reference computes the rest, and adds both in the stock slot order
// (src/dsv41/moe_op.h). STRATA_MOE_PROFILE=1 prints per-layer times at exit. Without a hot set, or without a
// CUDA device, every pair runs on the CPU, and it says so once on stderr.
//
// The fast kernel gives the CPU misses strata-moe's own AVX2 kernel (src/dsv41/cpu_kernel.h), several expert rows and all
// of an expert's tokens per pass over its weights. STRATA_MOE_CPU selects the kernel: fast (the default) or ref
// (the reference). On the fork's AVX2 ggml-cpu both give the same bits (tests/dsv41/moe_cpu_fast_test.cpp);
// fast computes q2_K, q3_K, q4_K and q6_K experts and aborts naming the tensor on any other type.
//
// The residency (src/dsv41/moe_residency.h): with STRATA_MOE_RAM_GB set and a hot set with warm lists
// (hot_set.py --ram-budget-gib), the warm experts of every layer live in one RAM buffer (STRATA_MOE_RAM_MODE: CUDA
// pinned by default with a CUDA device, mlock'd without one, or plain pre-faulted memory) and every expert in
// neither VRAM nor the warm tier is cold, read with O_DIRECT from its GGUF file into reusable staging during the
// calls that route to it; the fork's mmap pages of the routed experts are dropped after each layer's first call, so
// the page cache no longer carries them. The CPU computes warm and cold experts with the same kernel, so the output
// is the same bits (tests/dsv41/moe_residency_test.cpp, tools/dsv41/oracle/residency-pressure-test.sh).
//
// STRATA_MOE_COLD selects the cold tier's path, pagecache (the default: cold experts computed from the fork's
// mapping, its pages left to the kernel, so the page cache serves what the warm tier does not hold) or direct (the
// O_DIRECT reads above). STRATA_MOE_COUNTS=<path> writes per-(layer, expert) activation counts while serving
// (src/dsv41/moe_counts.h), which hot_set.py --counts turns into the hot set and the warm lists.
//
// The prefill path: with a GPU tier, a batch call of at least STRATA_MOE_PREFILL_MIN tokens (default 64, 0 turns it off) is a
// prefill call (src/dsv41/moe_prefill.h): every (token, slot) pair runs on the GPU, the slab's experts from VRAM and
// every other expert of the call copied once into a ring of device slots (STRATA_MOE_PREFILL_SETS sets of
// STRATA_MOE_PREFILL_GROUP experts) on a copy stream overlapped with the passes that use them, from the pinned warm
// tier or, for the rest, through pinned staging (STRATA_MOE_PREFILL_STAGE sets) that the op's CPU threads fill from
// the fork's mapping (STRATA_MOE_PREFILL_COLD=pagecache, the default) or the residency's reader threads fill with
// O_DIRECT reads of the GGUF files (direct, which needs STRATA_MOE_RAM_GB). The CPU only adds the slots in the
// stock order. Decode calls keep the hit/miss split. Without a GPU tier batch calls
// run on the CPU path, and it says so once.
#pragma once

#include "ggml.h"

#include <stdint.h>
#include <string.h>

// libstrata-moe.so is built with hidden visibility: these are its exported symbols
#if defined(_WIN32)
#define STRATA_MOE_API
#else
#define STRATA_MOE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// A ggml_custom_op_t: the fork passes it to ggml_custom_4d and ggml-cpu calls it on every thread of the graph
/// (ith in [0, nth)), all at once; the threads split the work and meet at internal barriers.
///
/// dst: f32 [n_embd, n_tokens], the MoE output (named ffn_moe_out by the fork). Sources, in this order:
///   src[0] cur               f32 [n_embd, n_tokens]          the MoE input
///   src[1] selected_experts  i32 [n_used, n_tokens]          slot order as the router's top-k
///   src[2] weights           f32 [1, n_used, n_tokens]       the final routing weights
///   src[3] up_exps               [n_embd, n_ff, n_expert]    any ggml type with a one-row ggml-cpu vec_dot
///   src[4] gate_exps             [n_embd, n_ff, n_expert]    the same, of any such type
///   src[5] down_exps             [n_ff, n_embd, n_expert]
/// dst->op_params[STRATA_MOE_CLAMP_PARAM]: the SwiGLU clamp limit as an f32 (ggml_swiglu_clamp:
/// silu(min(gate, limit)) * clamp(up, -limit, limit)), or 0 for plain SwiGLU (ggml_swiglu_split); set it with
/// strata_moe_set_clamp after ggml_custom_4d. It is a value in the op itself because the op runs every time
/// llama.cpp reuses the graph, long after the graph-build objects are gone. userdata must be null: a pointer there
/// is the hook's contract (b1e6002), whose pointer into graph-build memory dangled once the graph was reused.
///
/// Aborts through GGML_ABORT, naming the tensor and the value, on a type, shape, layout or expert id it cannot
/// compute exactly; expert tensors must be in host memory in ggml's own layout (not a CPU_REPACK buffer), also
/// for the hot experts, which are copied from there. The expert tensors' names give the layer
/// (blk.<layer>.ffn_{up,gate,down}_exps.weight, as the fork names them). A hot set that does not match the
/// tensors, a slab over the budget or the free VRAM, or a CUDA error aborts too. With STRATA_MOE_RAM_GB the expert
/// tensors must also be mmapped from their GGUF files (the cold tier reads those files), and a warm tier over the
/// budget, a warm-tier allocation that fails (cudaHostAlloc, mlock or the pre-fault, by STRATA_MOE_RAM_MODE) or a
/// failed O_DIRECT read aborts naming the size, the limit or the file.
STRATA_MOE_API void strata_moe_forward(struct ggml_tensor* dst, int ith, int nth, void* userdata);

/// The (token, slot) pairs of `layer` (its blk.<layer> index) computed so far on the GPU tier (hits) and on the
/// CPU (misses). Returns 0, leaving both untouched, when no call of the layer has finished yet.
STRATA_MOE_API int strata_moe_layer_counts(int layer, int64_t* hits, int64_t* misses);

/// The smallest and largest SwiGLU clamp limit (0: none) the calls of `layer` finished so far have computed with,
/// and how many calls that is. Returns 0, leaving the outputs untouched, when no call of the layer has finished.
STRATA_MOE_API int strata_moe_layer_clamp(int layer, float* min_limit, float* max_limit, int64_t* calls);

/// The op_params slot (int32 index) of the clamp limit: the last one, after ggml's custom-op params (fun, n_tasks,
/// userdata), which ggml_custom_4d writes from the start. The fork's hook writes the same slot.
#define STRATA_MOE_CLAMP_PARAM (GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1)

static inline void strata_moe_set_clamp(struct ggml_tensor* op, float limit) {
    memcpy(&op->op_params[STRATA_MOE_CLAMP_PARAM], &limit, sizeof(limit));
}

static inline float strata_moe_clamp(const struct ggml_tensor* op) {
    float limit;
    memcpy(&limit, &op->op_params[STRATA_MOE_CLAMP_PARAM], sizeof(limit));
    return limit;
}

#ifdef __cplusplus
}
#endif
