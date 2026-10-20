# strata-dsv41-engine: the dedicated DeepSeek-V4.1-Flash engine, plain C++ and CUDA; the fork
# is its correctness oracle, and its ggml-cuda MMQ is the optional prefill matmul on quantized weights
# (strata-dsv41-mmq below). Built with -DSTRATA_DSV41_ENGINE=ON on a CUDA build
# (tools/dsv41/build-inside.sh sets it). The tests and benches land in the build root, so the flavor's strata tarball
# carries them (tools/dsv41/pod/devgpu.sh runs them on a dev pod); they link the static CUDA runtime, so they need
# only the driver's libcuda there, plus cuBLAS (the f16 projections, kept for the tests) from the tarball's bin/ or
# the pod.

# The fork's MMQ (mmq.cuh with the q2_K, q3_K and q6_K instances) and the host symbols of ggml-cuda.cu it references
# (src/prefill/ggml_cuda_host.cu), compiled as Strata's strata_mmq compiles them (CMakeLists.txt, prompt-speed plan
# step 2b) but against the static CUDA runtime the engine binaries link. ggml-base is static (BUILD_SHARED_LIBS OFF).
if(STRATA_GGML_DIR)
  set(_dsv41_gg ${STRATA_GGML_DIR}/ggml)
else()
  set(_dsv41_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
endif()
set(_dsv41_mmq ${_dsv41_gg}/src/ggml-cuda)
foreach(_f mmq.cuh common.cuh template-instances/mmq-instance-q2_k.cu template-instances/mmq-instance-q3_k.cu
           template-instances/mmq-instance-q6_k.cu)
  if(NOT EXISTS ${_dsv41_mmq}/${_f})
    message(FATAL_ERROR "strata-dsv41-engine: the ggml checkout has no ${_dsv41_mmq}/${_f}")
  endif()
endforeach()
add_library(strata-dsv41-mmq STATIC qmatmul_mmq.cu "${PROJECT_SOURCE_DIR}/src/prefill/ggml_cuda_host.cu"
            ${_dsv41_mmq}/template-instances/mmq-instance-q2_k.cu ${_dsv41_mmq}/template-instances/mmq-instance-q3_k.cu
            ${_dsv41_mmq}/template-instances/mmq-instance-q6_k.cu)
target_include_directories(strata-dsv41-mmq PRIVATE ${_dsv41_gg}/include ${_dsv41_gg}/src ${_dsv41_mmq}
                           "${PROJECT_SOURCE_DIR}/include")
target_compile_options(strata-dsv41-mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
target_link_libraries(strata-dsv41-mmq PUBLIC ggml-base CUDA::cudart_static)
set_target_properties(strata-dsv41-mmq PROPERTIES POSITION_INDEPENDENT_CODE ON)

# the model loader, the routed expert tiers and the layer loop. The tiers stand on src/dsv41 behind plain
# pointers (include/strata/dsv41/engine/experts.hpp): strata-moe's hot-set loader, its residency (the warm tier), its
# CPU kernel, its prefill plan and its activation counts are compiled here against Strata's own static ggml-base (their
# ggml types and GGML_ABORT; no ggml-cpu, no ggml-cuda), with the fork's ggml-common.h (the K-quant blocks the CPU
# kernel reads) and vendored nlohmann json (the hot-set file, and the manifests of selection.cpp).
set(_dsv41_moe "${PROJECT_SOURCE_DIR}/src/dsv41")
add_library(strata-dsv41-engine STATIC indexer.cu indexer_ref.cpp attention.cu attention_layers.cpp attention_ref.cpp
            kv_cache.cpp kv_checkpoint.cpp qmatmul.cu qmatmul_ref.cpp dense.cu hc.cu engram.cu engram_hash.cpp router.cu dense_ffn.cu
            dense_ref.cpp model.cpp experts.cpp experts_cpu.cpp experts_cpu_mmvq.cpp mmvq_ref.cpp experts_kernels.cu
            experts_gemm.cu forward.cpp
            selection.cpp session.cpp batch.cpp hot_policy.cpp sampler.cpp
            ${_dsv41_moe}/cpu_kernel.cpp ${_dsv41_moe}/cpu_kernel_avx2.cpp ${_dsv41_moe}/moe_residency.cpp
            ${_dsv41_moe}/moe_counts.cpp ${_dsv41_moe}/moe_prefill.cpp ${_dsv41_moe}/moe_hot_set.cpp)
target_compile_features(strata-dsv41-engine PUBLIC cxx_std_17)
target_include_directories(strata-dsv41-engine PUBLIC "${PROJECT_SOURCE_DIR}/include"
                           PRIVATE "${_dsv41_moe}" "${_dsv41_gg}/src" "${_dsv41_gg}/../vendor")
target_link_libraries(strata-dsv41-engine PUBLIC CUDA::cudart_static CUDA::cublas Threads::Threads
                      PRIVATE strata-dsv41-mmq ggml-base strata_warnings)
set_target_properties(strata-dsv41-engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
# the CPU references reproduce the fork's _mm256_fmadd_ps lanes and its compiler's contractions with std::fma and
# must fuse nothing else (fork_cpu.hpp); attention_layers.cpp computes the fork's rope constants the same way;
# qmatmul_ref.cpp repeats the engine quantizers' f32 operations one by one; dense_ref.cpp the fork's dense ops;
# experts_cpu.cpp the fork's Q8_K quantizer and SwiGLU clamp
set_source_files_properties(indexer_ref.cpp attention_ref.cpp attention_layers.cpp qmatmul_ref.cpp dense_ref.cpp PROPERTIES COMPILE_OPTIONS "-mfma;-ffp-contract=off")
set_source_files_properties(experts_cpu.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# the GPU-order reference repeats the decode kernels' float operations one by one (std::fma for __fmaf_rn), and the
# CPU experts in the GPU's numerics are its AVX2 form (expert_act.hpp: one exp for both sides)
set_source_files_properties(mmvq_ref.cpp PROPERTIES COMPILE_OPTIONS "-mfma;-ffp-contract=off")
set_source_files_properties(experts_cpu_mmvq.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")
# the hot set's counts are tools/dsv41/engine/hot_policy_sim.py's double operations one by one
set_source_files_properties(hot_policy.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# strata-moe's fast CPU kernel with its own flags (src/dsv41/CMakeLists.txt): ggml-cpu's AVX2 baseline, nothing fused
set_source_files_properties(${_dsv41_moe}/cpu_kernel_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")

# the engine end to end (tools/dsv41/engine/dsv41-engine.cpp): a GGUF, token ids in, greedy token ids out;
# --serve, resident behind Strata's stdin protocol with the prefix cache (session.cpp); --serve's sampled
# requests (sampler.cpp)
add_executable(dsv41-engine "${PROJECT_SOURCE_DIR}/tools/dsv41/engine/dsv41-engine.cpp")
target_link_libraries(dsv41-engine PRIVATE strata-dsv41-engine strata_warnings)
set_target_properties(dsv41-engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# the first layers of the real model on the CPU reference in three numerics,
# every op dumped (tools/dsv41/engine/dsv41-layer-ref.cpp); CPU only, it runs on the pod beside the engine
add_executable(dsv41-layer-ref "${PROJECT_SOURCE_DIR}/tools/dsv41/engine/dsv41-layer-ref.cpp")
target_link_libraries(dsv41-layer-ref PRIVATE strata-dsv41-engine strata_warnings)
set_target_properties(dsv41-layer-ref PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(STRATA_BUILD_TESTS)
  # (a) the CPU reference against the op dumps (GPU free: --cases, the lists of
  # tools/dsv41/engine/indexer_cases.py; tools/dsv41/verify.sh runs it), (b) CUDA against the CPU reference on random
  # inputs at tiny and real dims (--gpu, on a dev pod)
  add_executable(dsv41_engine_indexer_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/indexer_test.cpp")
  target_link_libraries(dsv41_engine_indexer_test PRIVATE strata-dsv41-engine Threads::Threads strata_warnings)
  set_target_properties(dsv41_engine_indexer_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # bench (not a test): ms, GB/s, TFLOP/s and scratch bytes at real dims (GPU)
  add_executable(dsv41_engine_indexer_bench "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/indexer_bench.cpp")
  target_link_libraries(dsv41_engine_indexer_bench PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_indexer_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # (a) the attention reference against the op dumps (GPU free: --cases, the lists of
  # tools/dsv41/engine/attention_cases.py; verify.sh runs it), (b) the CUDA layer against the reference stage by
  # stage at tiny and real dims (--gpu, on a dev pod)
  add_executable(dsv41_engine_attention_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/attention_test.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/attention_test_gpu.cu")
  target_link_libraries(dsv41_engine_attention_test PRIVATE strata-dsv41-engine Threads::Threads strata_warnings)
  set_target_properties(dsv41_engine_attention_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # bench (not a test): prefill and decode ms per layer role and stage, KV and workspace bytes at real dims (GPU)
  add_executable(dsv41_engine_attention_bench "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/attention_bench.cu")
  target_link_libraries(dsv41_engine_attention_bench PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_attention_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # (a) the qmatmul reference against hand-quantized blocks and against the fork's CPU dots (ggml-cpu, q8_K) on
  # the tiny GGUFs' matrices (GPU free: --cpu; verify.sh runs it), (b) the GPU quantizers and matmuls against the
  # reference at every real dense shape (--gpu, on a dev pod)
  add_executable(dsv41_engine_qmatmul_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/qmatmul_test.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/qmatmul_test_gpu.cu")
  target_link_libraries(dsv41_engine_qmatmul_test PRIVATE strata-dsv41-engine ggml-cpu ggml-base Threads::Threads strata_warnings)
  set_target_properties(dsv41_engine_qmatmul_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # bench (not a test): GB/s of the decode GEMV and TFLOP/s of the prefill GEMM at the real shapes, against
  # cuBLAS f16 on the same shapes (GPU)
  add_executable(dsv41_engine_qmatmul_bench "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/qmatmul_bench.cu")
  target_link_libraries(dsv41_engine_qmatmul_bench PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_qmatmul_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # (a) the dense ops' reference (HC, engram, router, norms, shared expert, embedding, head) against the op
  # dumps (GPU free: --cases, the lists of tools/dsv41/engine/dense_cases.py; verify.sh runs it and the host side of
  # (b), --gpu --dry-run), (b) the CUDA ops against the reference at tiny and real dims (--gpu, on a dev pod)
  add_executable(dsv41_engine_dense_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/dense_test.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/dense_test_gpu.cu")
  target_link_libraries(dsv41_engine_dense_test PRIVATE strata-dsv41-engine Threads::Threads strata_warnings)
  set_target_properties(dsv41_engine_dense_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # bench (not a test): ms per dense op for a 4096-token chunk and a decode token, the engram gather, the head
  # and the router's GEMV against cuBLAS bf16 at real dims (GPU)
  add_executable(dsv41_engine_dense_bench "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/dense_bench.cu")
  target_link_libraries(dsv41_engine_dense_bench PRIVATE strata-dsv41-engine Threads::Threads strata_warnings)
  set_target_properties(dsv41_engine_dense_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # (a) the CPU experts against the op dumps' ffn_moe_out bit for bit (GPU free: --cases, the lists of
  # tools/dsv41/engine/dense_cases.py; verify.sh runs it), (b) the expert tiers on CUDA in three placements against the
  # CPU and engine-numerics references (--gpu GGUF..., on a dev pod; --gpu --dry-run its host side, verify.sh);
  # (d) the deterministic mode on the CPU reference path (--determinism GGUF, GPU free; verify.sh, q03-test.sh)
  add_executable(dsv41_engine_experts_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/experts_test.cpp")
  target_link_libraries(dsv41_engine_experts_test PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_experts_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # the CPU experts in the GPU's numerics against the GPU-order reference
  # (mmvq_ref.hpp) bit for bit on random q2_K / q3_K experts, and the speed of both CPU numerics (GPU free; verify.sh and
  # quality/c05-test.sh run it); --gpu the CUDA kernels against the CPU experts bit for bit (a dev pod, devgpu.sh c05)
  add_executable(dsv41_engine_expert_bits_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/expert_bits_test.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/expert_bits_test_gpu.cu")
  set_source_files_properties("${PROJECT_SOURCE_DIR}/tests/dsv41/engine/expert_bits_test.cpp" PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  target_link_libraries(dsv41_engine_expert_bits_test PRIVATE strata-dsv41-engine Threads::Threads strata_warnings)
  set_target_properties(dsv41_engine_expert_bits_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # the grouped GEMM's table decoder on the host against the dequant reference, bit for bit (GPU free;
  # verify.sh runs it)
  add_executable(dsv41_engine_qdequant_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/qdequant_test.cu")
  target_link_libraries(dsv41_engine_qdequant_test PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_qdequant_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # the per-request hot set's planner (GPU free: verify.sh runs its checks, and --replay on a routing trace
  # against tools/dsv41/engine/hot_policy_sim.py)
  add_executable(dsv41_engine_hot_policy_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/hot_policy_test.cpp")
  set_source_files_properties("${PROJECT_SOURCE_DIR}/tests/dsv41/engine/hot_policy_test.cpp" PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  target_include_directories(dsv41_engine_hot_policy_test PRIVATE "${_dsv41_gg}/../vendor")
  target_link_libraries(dsv41_engine_hot_policy_test PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_hot_policy_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # the host sampler of --serve (top_k, top_p, min_p, temperature, the seeded draw) and the Session's pick (GPU
  # free; verify.sh runs it, tools/dsv41/engine/sampler_test.sh builds it with the host compiler too)
  add_executable(dsv41_engine_sampler_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/sampler_test.cpp")
  target_link_libraries(dsv41_engine_sampler_test PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_sampler_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # the Session feeds a promoted request's one token left by decode, as alone (a recording backend; GPU
  # free, no model; verify.sh and quality/c05-test.sh run it, tools/dsv41/engine/session_test.sh builds it with the host
  # compiler too, with and without the fix)
  add_executable(dsv41_engine_session_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/session_test.cpp")
  target_link_libraries(dsv41_engine_session_test PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_session_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # the Session's conversation checkpoints on the CPU reference path, save, restore and evict bit for bit against
  # a straight read (GPU free; verify.sh and quality/c02-test.sh run it on tiny-b)
  add_executable(dsv41_engine_checkpoint_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/checkpoint_test.cpp")
  target_link_libraries(dsv41_engine_checkpoint_test PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_checkpoint_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # batched decode on the CPU reference path, each request of a batch bit for bit its alone run (GPU free;
  # verify.sh and quality/c03-test.sh run it on tiny-b); --gpu the engine and --kernels the batched kernels the same way
  # (a dev pod, tools/dsv41/pod/devgpu.sh's steps)
  add_executable(dsv41_engine_batch_test "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/batch_test.cpp"
                 "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/batch_test_gpu.cu")
  target_link_libraries(dsv41_engine_batch_test PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_batch_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # bench (not a test): the whole engine at the real shapes on a synthetic model with few experts per layer, one
  # 4096-token prefill chunk at context 16K and 32 decode steps, per-stage times (GPU)
  add_executable(dsv41_engine_smoke "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/engine_smoke.cpp")
  # --counts reads a strata-moe-counts/1 profile with the vendored nlohmann json
  target_include_directories(dsv41_engine_smoke PRIVATE "${_dsv41_gg}/../vendor")
  target_link_libraries(dsv41_engine_smoke PRIVATE strata-dsv41-engine strata_warnings)
  set_target_properties(dsv41_engine_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # the dev pod's first step (tools/dsv41/pod/devgpu.sh): a kernel and a copy, so a host whose CUDA is broken is left
  # before the payload goes up
  add_executable(dsv41_cuda_probe "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/cuda_probe.cu")
  target_link_libraries(dsv41_cuda_probe PRIVATE CUDA::cudart_static)
  set_target_properties(dsv41_cuda_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  # the qualification's CPU experts rate, decode-shaped, through the engine's own
  # CpuExperts as this build compiles it (so it ships in the tarball, not with the standalone machine probe); GPU free.
  # The qualification uploads it alone, before the payload: --as-needed drops the cuBLAS the engine library names and
  # the probe never calls
  add_executable(dsv41_cpu_experts_probe "${PROJECT_SOURCE_DIR}/tests/dsv41/engine/cpu_experts_probe.cpp")
  target_link_libraries(dsv41_cpu_experts_probe PRIVATE strata-dsv41-engine strata_warnings)
  target_link_options(dsv41_cpu_experts_probe PRIVATE "LINKER:--as-needed")
  set_target_properties(dsv41_cpu_experts_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
