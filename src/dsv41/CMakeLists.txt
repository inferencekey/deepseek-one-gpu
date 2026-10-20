# strata-moe: the routed experts behind the llama.cpp fork's build_moe_ffn hook (include/strata/dsv41/moe.h), and its
# tests. Built only with -DSTRATA_DSV41_FORK_BUILD=<the fork's build tree>, after the fork: the library links that
# tree's libggml-base, libggml-cpu and libggml-cuda (not Strata's static ggml) so the fork's process has one ggml,
# and the fork loads it from its bin/ at run time (tools/dsv41/build-inside.sh copies it there).
if(NOT STRATA_GGML_DIR)
  message(FATAL_ERROR "STRATA_DSV41_FORK_BUILD needs STRATA_GGML_DIR, the fork checkout the build tree was made from")
endif()

set(_dsv41_fork_bin "${STRATA_DSV41_FORK_BUILD}/bin")
foreach(_lib ggml-base ggml-cpu ggml-cuda)
  set(_so "${_dsv41_fork_bin}/lib${_lib}.so")
  if(NOT EXISTS "${_so}")
    message(FATAL_ERROR "STRATA_DSV41_FORK_BUILD=${STRATA_DSV41_FORK_BUILD}: ${_so} not found, build the fork first")
  endif()
  add_library(dsv41_fork_${_lib} SHARED IMPORTED)
  set_target_properties(dsv41_fork_${_lib} PROPERTIES
    IMPORTED_LOCATION "${_so}"
    INTERFACE_INCLUDE_DIRECTORIES "${STRATA_GGML_DIR}/ggml/include"
    INTERFACE_COMPILE_DEFINITIONS GGML_SHARED)
endforeach()

# moe_ref.cpp calls ggml_vec_swiglu_f32, which libggml-cpu exports without a public header (ggml_cpu_vec.h):
# check here that this fork build still exports it with that name, instead of failing at link or dlopen time
include(CheckCXXSourceCompiles)
unset(STRATA_DSV41_FORK_HAS_VEC_SWIGLU CACHE)  # re-check on every configure: the fork tree can be rebuilt
set(CMAKE_REQUIRED_LIBRARIES "${_dsv41_fork_bin}/libggml-cpu.so")
set(CMAKE_REQUIRED_LINK_OPTIONS "LINKER:-rpath-link,${_dsv41_fork_bin}")
check_cxx_source_compiles([[
extern "C" void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g);
int main() {
    float y = 0.0f, x = 1.0f, g = 1.0f;
    ggml_vec_swiglu_f32(1, &y, &x, &g);
    return 0;
}]] STRATA_DSV41_FORK_HAS_VEC_SWIGLU)
unset(CMAKE_REQUIRED_LIBRARIES)
unset(CMAKE_REQUIRED_LINK_OPTIONS)
if(NOT STRATA_DSV41_FORK_HAS_VEC_SWIGLU)
  message(FATAL_ERROR "STRATA_DSV41_FORK_BUILD=${STRATA_DSV41_FORK_BUILD}: ${_dsv41_fork_bin}/libggml-cpu.so does not "
                      "export ggml_vec_swiglu_f32 (declared in src/dsv41/ggml_cpu_vec.h from the fork's 5210c7c5)")
endif()

# the CPU reference and the fast kernel, the split and the combine, the profile, the warm and cold tiers, the
# activation counts, the prefill plan and schedule, the hot-set file: the library, the GPU-free tests and
# the CPU bench link these. cpu_kernel*.cpp, moe_residency.cpp, moe_counts.cpp, moe_prefill.cpp and moe_hot_set.cpp
# take plain pointers too: the dedicated engine (src/dsv41/engine) builds them against Strata's own ggml-base.
add_library(strata-moe-cpu OBJECT moe_ref.cpp moe_profile.cpp cpu_kernel.cpp cpu_kernel_avx2.cpp moe_residency.cpp
            moe_residency_call.cpp moe_counts.cpp moe_prefill.cpp moe_hot_set.cpp)
target_compile_features(strata-moe-cpu PUBLIC cxx_std_17)
target_include_directories(strata-moe-cpu PUBLIC "${PROJECT_SOURCE_DIR}/include")
# cpu_kernel_avx2.cpp reads the K-quant blocks through the fork's ggml-common.h; moe_hot_set.cpp parses with the
# fork's vendored nlohmann json
target_include_directories(strata-moe-cpu PRIVATE "${STRATA_GGML_DIR}/ggml/src" "${STRATA_GGML_DIR}/vendor")
target_link_libraries(strata-moe-cpu PUBLIC dsv41_fork_ggml-base dsv41_fork_ggml-cpu PRIVATE strata_warnings)
# the clamp and the weighted sum round every multiply and add on their own, as the separate stock ggml ops do
target_compile_options(strata-moe-cpu PRIVATE -ffp-contract=off)
# the fast kernel only: the fork's ggml-cpu baseline (AVX2, FMA, F16C); fast_kernel_cpu_ok, in cpu_kernel.cpp, checks
# the CPU before any of it runs
set_source_files_properties(cpu_kernel_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_target_properties(strata-moe-cpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

# the GPU tier (moe_gpu.cpp) calls ggml-cuda's public API, the CUDA runtime for its timing events and the fork's
# ggml-backend-impl.h for struct ggml_backend_event
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
add_library(strata-moe SHARED moe.cpp moe_gpu.cpp)
target_compile_features(strata-moe PRIVATE cxx_std_17)
target_include_directories(strata-moe PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${STRATA_GGML_DIR}/ggml/src")
target_link_libraries(strata-moe PUBLIC dsv41_fork_ggml-base
  PRIVATE strata-moe-cpu dsv41_fork_ggml-cpu dsv41_fork_ggml-cuda CUDA::cudart Threads::Threads strata_warnings)
set_target_properties(strata-moe PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
# libggml-cpu needs libggml-base from the same directory at link time
target_link_options(strata-moe PRIVATE "LINKER:-rpath-link,${_dsv41_fork_bin}")

if(STRATA_BUILD_TESTS)
  # the binaries carry RUNPATH $ORIGIN, but the fork's libggml* are not next to them: the tests get the fork's bin/
  # in front of the caller's LD_LIBRARY_PATH (which may hold a libcuda stub), and this build's dsv41/ in front of
  # that, so the libstrata-moe.so under test is this build's, not the copy in the fork's bin/
  set(_dsv41_test_env "LD_LIBRARY_PATH=path_list_prepend:${_dsv41_fork_bin}"
                      "LD_LIBRARY_PATH=path_list_prepend:${CMAKE_BINARY_DIR}/dsv41")

  # reference vs the stock mul_mat_id expert graph on ggml's CPU backend at DeepSeek-V4.1 dims (no model, no GPU),
  # through the shared library: with STRATA_MOE_HOT unset it is the CPU-only mode, and STRATA_MOE_CPU=ref picks the
  # reference (the fast kernel's own test is dsv41_moe_cpu_fast_test)
  add_executable(dsv41_moe_ref_test "${PROJECT_SOURCE_DIR}/tests/dsv41/moe_ref_test.cpp")
  target_compile_features(dsv41_moe_ref_test PRIVATE cxx_std_17)
  target_link_libraries(dsv41_moe_ref_test PRIVATE strata-moe dsv41_fork_ggml-cpu Threads::Threads strata_warnings)
  # libggml-cuda's libcuda.so.1 is the driver's, absent from the builder: its symbols stay unresolved at link time
  target_link_options(dsv41_moe_ref_test PRIVATE "LINKER:-rpath-link,${_dsv41_fork_bin}" "LINKER:--allow-shlib-undefined")
  set_target_properties(dsv41_moe_ref_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  add_test(NAME dsv41_moe_ref_test COMMAND dsv41_moe_ref_test)
  set_tests_properties(dsv41_moe_ref_test PROPERTIES ENVIRONMENT_MODIFICATION "${_dsv41_test_env}"
                       ENVIRONMENT "STRATA_MOE_CPU=ref")

  # the hit/miss split and the combine order: random resident sets with a CPU stand-in for the GPU tier give the
  # all-CPU output bit for bit, with either CPU kernel (no GPU, no CUDA library)
  add_executable(dsv41_moe_split_test "${PROJECT_SOURCE_DIR}/tests/dsv41/moe_split_test.cpp")
  target_include_directories(dsv41_moe_split_test PRIVATE "${PROJECT_SOURCE_DIR}/src/dsv41")
  target_link_libraries(dsv41_moe_split_test PRIVATE strata-moe-cpu Threads::Threads strata_warnings)
  target_link_options(dsv41_moe_split_test PRIVATE "LINKER:-rpath-link,${_dsv41_fork_bin}")
  set_target_properties(dsv41_moe_split_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  add_test(NAME dsv41_moe_split_test COMMAND dsv41_moe_split_test)
  set_tests_properties(dsv41_moe_split_test PROPERTIES ENVIRONMENT_MODIFICATION "${_dsv41_test_env}")

  # the fast CPU kernel against the reference, bit for bit, at DeepSeek-V4.1 and OLMoE dims, and its rows
  # against ggml-cpu's vec_dot at odd row ranges and token counts (no GPU)
  add_executable(dsv41_moe_cpu_fast_test "${PROJECT_SOURCE_DIR}/tests/dsv41/moe_cpu_fast_test.cpp")
  target_include_directories(dsv41_moe_cpu_fast_test PRIVATE "${PROJECT_SOURCE_DIR}/src/dsv41")
  target_link_libraries(dsv41_moe_cpu_fast_test PRIVATE strata-moe-cpu Threads::Threads strata_warnings)
  target_link_options(dsv41_moe_cpu_fast_test PRIVATE "LINKER:-rpath-link,${_dsv41_fork_bin}")
  set_target_properties(dsv41_moe_cpu_fast_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  add_test(NAME dsv41_moe_cpu_fast_test COMMAND dsv41_moe_cpu_fast_test)
  set_tests_properties(dsv41_moe_cpu_fast_test PROPERTIES ENVIRONMENT_MODIFICATION "${_dsv41_test_env}")

  # the cold reader (O_DIRECT alignment, short reads, the mapped bytes bit for bit) and the tier map (the op
  # with warm and cold experts against the mapped tensors, bit for bit), GPU free; its files go in the build tree
  add_executable(dsv41_moe_residency_test "${PROJECT_SOURCE_DIR}/tests/dsv41/moe_residency_test.cpp")
  target_include_directories(dsv41_moe_residency_test PRIVATE "${PROJECT_SOURCE_DIR}/src/dsv41")
  target_link_libraries(dsv41_moe_residency_test PRIVATE strata-moe-cpu Threads::Threads strata_warnings)
  target_link_options(dsv41_moe_residency_test PRIVATE "LINKER:-rpath-link,${_dsv41_fork_bin}")
  set_target_properties(dsv41_moe_residency_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  add_test(NAME dsv41_moe_residency_test COMMAND dsv41_moe_residency_test WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  set_tests_properties(dsv41_moe_residency_test PROPERTIES ENVIRONMENT_MODIFICATION "${_dsv41_test_env}")

  # the activation counts (the file's format, the periodic and exit writes, through the op) (no GPU); its files
  # go in the build tree
  add_executable(dsv41_moe_counts_test "${PROJECT_SOURCE_DIR}/tests/dsv41/moe_counts_test.cpp")
  target_include_directories(dsv41_moe_counts_test PRIVATE "${PROJECT_SOURCE_DIR}/src/dsv41")
  target_link_libraries(dsv41_moe_counts_test PRIVATE strata-moe-cpu Threads::Threads strata_warnings)
  target_link_options(dsv41_moe_counts_test PRIVATE "LINKER:-rpath-link,${_dsv41_fork_bin}")
  set_target_properties(dsv41_moe_counts_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  add_test(NAME dsv41_moe_counts_test COMMAND dsv41_moe_counts_test WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  set_tests_properties(dsv41_moe_counts_test PROPERTIES ENVIRONMENT_MODIFICATION "${_dsv41_test_env}")

  # the prefill plan (every pair one row, every expert outside the slab copied once), the ring schedule on a
  # fake two-stream engine (no slot overwritten before its pass ran, no staging refilled before its copy), the staging
  # jobs over threads, and a prefill call through the op with a CPU engine, bit for bit against all-CPU with the
  # staging filled both ways (STRATA_MOE_PREFILL_COLD; direct reads its file with O_DIRECT, so a real filesystem: no GPU)
  add_executable(dsv41_moe_prefill_test "${PROJECT_SOURCE_DIR}/tests/dsv41/moe_prefill_test.cpp")
  target_include_directories(dsv41_moe_prefill_test PRIVATE "${PROJECT_SOURCE_DIR}/src/dsv41")
  target_link_libraries(dsv41_moe_prefill_test PRIVATE strata-moe-cpu Threads::Threads strata_warnings)
  target_link_options(dsv41_moe_prefill_test PRIVATE "LINKER:-rpath-link,${_dsv41_fork_bin}")
  set_target_properties(dsv41_moe_prefill_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  add_test(NAME dsv41_moe_prefill_test COMMAND dsv41_moe_prefill_test WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  set_tests_properties(dsv41_moe_prefill_test PROPERTIES ENVIRONMENT_MODIFICATION "${_dsv41_test_env}")

  # the fork patch's 64-bit indexer arithmetic (ggml-cuda unary_cuda and reduce_rows_f32): the launch numbers of
  # the old int code and the new int64_t code at the indexer score shapes, and that the fork tree has the new code
  # (no GPU: the kernels themselves only run on the pod)
  add_executable(dsv41_indexer_index_test "${PROJECT_SOURCE_DIR}/tests/dsv41/indexer_index_test.cpp")
  target_compile_features(dsv41_indexer_index_test PRIVATE cxx_std_17)
  target_link_libraries(dsv41_indexer_index_test PRIVATE strata_warnings)
  set_target_properties(dsv41_indexer_index_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
  add_test(NAME dsv41_indexer_index_test COMMAND dsv41_indexer_index_test "${STRATA_GGML_DIR}")

  # microbenchmark (not a test): ms per call, GB/s of expert bytes and the ratio to the machine's read
  # bandwidth, fast and ref, at DeepSeek-V4.1 dims on synthetic weights larger than the LLC
  add_executable(dsv41_moe_cpu_bench "${PROJECT_SOURCE_DIR}/tests/dsv41/moe_cpu_bench.cpp")
  target_include_directories(dsv41_moe_cpu_bench PRIVATE "${PROJECT_SOURCE_DIR}/src/dsv41"
                             "${STRATA_GGML_DIR}/ggml/src")
  target_link_libraries(dsv41_moe_cpu_bench PRIVATE strata-moe-cpu Threads::Threads strata_warnings)
  target_link_options(dsv41_moe_cpu_bench PRIVATE "LINKER:-rpath-link,${_dsv41_fork_bin}")
  set_target_properties(dsv41_moe_cpu_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dsv41")
endif()
