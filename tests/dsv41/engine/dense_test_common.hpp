// tests/dsv41/engine/dense_test_common.hpp - shared by the two halves of dsv41_engine_dense_test: dense_test.cpp
// (main, (a) the CPU reference against the fork's dumps) and dense_test_gpu.cu ((b) CUDA against the CPU reference).
#pragma once

#include "strata/dsv41/engine/dense.hpp"
#include "strata/dsv41/engine/dense_ref.hpp"
#include "strata/dsv41/engine/engram.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

[[noreturn]] inline void die(const std::string& m) {
    std::fprintf(stderr, "dsv41_engine_dense_test: %s\n", m.c_str());
    std::exit(2);
}

/// (a): the engine numerics' matrix outputs against the fork's, worst rel L2 per ubatch
constexpr double kEngineRelL2 = 5e-2;

/// (b): the real-dims engram table's default row count, 4M rows of 84 bytes (336 MiB of pinned host RAM; the real
/// tables are 384M rows)
constexpr int64_t kDefaultTableRows = 1 << 22;

/// (b): CUDA against the CPU reference at tiny and real dims (dense_test_gpu.cu); mmq: QPrefill::Mmq, dry: the cases'
/// host-side checks only, no CUDA call (the dev VM); table_rows: the real-dims engram table's rows (84 bytes each)
int run_gpu(bool quick, bool mmq, bool dry, int64_t table_rows);
