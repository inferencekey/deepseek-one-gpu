// src/dsv41/cpu_kernel.cpp - the run-time check of the fast CPU kernel (cpu_kernel.h), built without the kernel's
// AVX2 flags so it runs on any x86-64; strata-moe (moe.cpp) and the dedicated engine (src/dsv41/engine/experts_cpu.cpp)
// ask it before any of cpu_kernel_avx2.cpp runs.
#include "cpu_kernel.h"

namespace strata::dsv41 {

bool fast_kernel_cpu_ok() {
    static const bool ok = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    }();
    return ok;
}

}  // namespace strata::dsv41
