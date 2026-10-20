// tests/dsv41/engine/attention_test_common.hpp - helpers shared by the two halves of dsv41_engine_attention_test:
// attention_test.cpp (main, (a) the CPU reference against the fork's dumps) and attention_test_gpu.cpp ((b) CUDA
// against the CPU reference).
#pragma once

#include "strata/dsv41/engine/attention.hpp"
#include "strata/dsv41/engine/attention_ref.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

[[noreturn]] inline void die(const std::string& m) {
    std::fprintf(stderr, "dsv41_engine_attention_test: %s\n", m.c_str());
    std::exit(2);
}

/// (b): CUDA against the CPU reference at tiny and real dims (attention_test_gpu.cu); f16: the cuBLAS test path's
/// weights, mmq: QPrefill::Mmq, dry: the cases' host-side checks only, no CUDA call (the dev VM)
int run_gpu(bool quick, bool f16, bool mmq, bool dry);
