# dsv41-moe-dump, dsv41-moe-replay and dsv41-moe-clamp-check, built against a configured and built llama.cpp fork
# tree without touching the fork:
#
#   cmake -S tools/dsv41/oracle -B BUILD -DFORK_DIR=<fork checkout> -DFORK_BUILD=<fork build tree>
#
# tools/dsv41/build-inside.sh does this per flavor and writes the executables into the fork's bin/, next to the
# shared libraries they load (dsv41-moe-replay dlopens libstrata-moe.so from there, as the fork does).
cmake_minimum_required(VERSION 3.24)
project(dsv41_oracle CXX)

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

add_executable(dsv41-moe-dump dsv41-moe-dump.cpp)
target_compile_features(dsv41-moe-dump PRIVATE cxx_std_17)
target_compile_definitions(dsv41-moe-dump PRIVATE LLAMA_SHARED GGML_SHARED)
target_include_directories(dsv41-moe-dump PRIVATE
    "${FORK_DIR}/include" "${FORK_DIR}/common" "${FORK_DIR}/ggml/include" "${FORK_DIR}/vendor")
# the fork's libraries resolve each other (libggml -> libggml-cuda -> the CUDA runtime) at link time through
# -rpath-link; at run time through RUNPATH $ORIGIN, since the executable sits next to them in bin/.
# libggml-cuda's libcuda.so.1 is the driver's, absent from the builder (only the toolkit's libcuda.so stub is
# there), so its symbols stay unresolved at link time: --allow-shlib-undefined.
target_link_libraries(dsv41-moe-dump PRIVATE ${fork_libs})
target_link_options(dsv41-moe-dump PRIVATE "LINKER:-rpath-link,${FORK_BUILD}/bin" "LINKER:--allow-shlib-undefined")
set_target_properties(dsv41-moe-dump PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
    RUNTIME_OUTPUT_DIRECTORY "${FORK_BUILD}/bin")

# the replay reads the GGUF and runs strata_moe_forward on ggml-cpu: libggml-base and libggml-cpu, the op's
# signature from Strata's include/strata/dsv41/moe.h, libstrata-moe.so itself through dlopen at run time
set(strata_include "${CMAKE_CURRENT_SOURCE_DIR}/../../../include")
foreach(lib ggml-base ggml-cpu)
    if(NOT EXISTS "${FORK_BUILD}/bin/lib${lib}.so")
        message(FATAL_ERROR "FORK_BUILD=${FORK_BUILD}: ${FORK_BUILD}/bin/lib${lib}.so not found, build the fork first")
    endif()
endforeach()
add_executable(dsv41-moe-replay dsv41-moe-replay.cpp)
target_compile_features(dsv41-moe-replay PRIVATE cxx_std_17)
target_compile_definitions(dsv41-moe-replay PRIVATE GGML_SHARED)
target_include_directories(dsv41-moe-replay PRIVATE "${FORK_DIR}/ggml/include" "${FORK_DIR}/vendor" "${strata_include}")
target_link_libraries(dsv41-moe-replay PRIVATE "${FORK_BUILD}/bin/libggml-base.so" "${FORK_BUILD}/bin/libggml-cpu.so" ${CMAKE_DL_LIBS})
target_link_options(dsv41-moe-replay PRIVATE "LINKER:-rpath-link,${FORK_BUILD}/bin")
set_target_properties(dsv41-moe-replay PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
    RUNTIME_OUTPUT_DIRECTORY "${FORK_BUILD}/bin")

# the clamp check runs the fork's hook path through libllama on the CPU (tools/dsv41/oracle/hook-clamp-test.sh) and
# asks libstrata-moe.so, which the fork loaded, for the limits its op computed with
add_executable(dsv41-moe-clamp-check dsv41-moe-clamp-check.cpp)
target_compile_features(dsv41-moe-clamp-check PRIVATE cxx_std_17)
target_compile_definitions(dsv41-moe-clamp-check PRIVATE LLAMA_SHARED GGML_SHARED)
target_include_directories(dsv41-moe-clamp-check PRIVATE "${FORK_DIR}/include" "${FORK_DIR}/ggml/include")
target_link_libraries(dsv41-moe-clamp-check PRIVATE "${FORK_BUILD}/bin/libllama.so" "${FORK_BUILD}/bin/libggml-base.so" ${CMAKE_DL_LIBS})
target_link_options(dsv41-moe-clamp-check PRIVATE "LINKER:-rpath-link,${FORK_BUILD}/bin" "LINKER:--allow-shlib-undefined")
set_target_properties(dsv41-moe-clamp-check PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
    RUNTIME_OUTPUT_DIRECTORY "${FORK_BUILD}/bin")
