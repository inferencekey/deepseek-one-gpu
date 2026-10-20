# op-dump, built against a configured and built llama.cpp fork tree without touching the fork:
#
#   cmake -S tools/dsv41/engine/op-dump -B BUILD -DFORK_DIR=<fork checkout> -DFORK_BUILD=<fork build tree>
#
# tools/dsv41/engine/run-oracles.sh does this in the cu128 builder image against $BUILD_ROOT/cu128/fork, whose
# libraries are byte-identical to the cu128 tarball's, and runs the result against the tarball's bin/.
cmake_minimum_required(VERSION 3.24)
project(dsv41_op_dump CXX)

set(FORK_DIR "" CACHE PATH "llama.cpp fork checkout (headers)")
set(FORK_BUILD "" CACHE PATH "the fork's build tree (bin/ with libllama, libllama-common, libggml*)")

foreach(f "${FORK_DIR}/include/llama.h" "${FORK_DIR}/common/arg.h" "${FORK_DIR}/vendor/nlohmann/json.hpp")
    if(NOT EXISTS "${f}")
        message(FATAL_ERROR "FORK_DIR=${FORK_DIR}: ${f} not found")
    endif()
endforeach()

set(fork_libs)
foreach(lib llama-common llama ggml ggml-base)
    set(so "${FORK_BUILD}/bin/lib${lib}.so")
    if(NOT EXISTS "${so}")
        message(FATAL_ERROR "FORK_BUILD=${FORK_BUILD}: ${so} not found, build the fork first")
    endif()
    list(APPEND fork_libs "${so}")
endforeach()

add_executable(op-dump op-dump.cpp)
target_compile_features(op-dump PRIVATE cxx_std_17)
target_compile_definitions(op-dump PRIVATE LLAMA_SHARED GGML_SHARED)
target_include_directories(op-dump PRIVATE
    "${FORK_DIR}/include" "${FORK_DIR}/common" "${FORK_DIR}/ggml/include" "${FORK_DIR}/vendor")
# as tools/dsv41/oracle/CMakeLists.txt: the fork's libraries resolve each other through -rpath-link at link time,
# libggml-cuda's libcuda.so.1 is the driver's and absent from the builder (--allow-shlib-undefined); at run time
# the libraries come from the directory in LD_LIBRARY_PATH (the tarball's bin/)
target_link_libraries(op-dump PRIVATE ${fork_libs})
target_link_options(op-dump PRIVATE "LINKER:-rpath-link,${FORK_BUILD}/bin" "LINKER:--allow-shlib-undefined")
