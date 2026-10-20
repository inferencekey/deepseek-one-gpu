// tests/dsv41/indexer_index_test.cpp - the int32 overflow of the DeepSeek-V4.1 indexer scores in ggml-cuda,
// and the fork patch's fix (third_party/llama-dsv41/0001-strata-moe-hook.patch), GPU free.
//
// build_indexer_top_k (src/models/deepseek41.cpp in the fork) makes one f32 score tensor of n_comp x ubatch x 32
// elements (32 = deepseek41.attention.indexer.head_count; n_comp grows to the context) and runs cont, relu, mul and
// sum_rows on it. The kernels cannot run here, so this test checks the launch arithmetic both ways, at the shapes
// that matter, with the expressions of ggml/src/ggml-cuda at 5210c7c5 and after the patch:
//   unary_cuda, before:  const int k = ggml_nelements(src0); num_blocks = (k + 255) / 256; dim3(num_blocks)
//              after:    int64_t k, int64_t num_blocks asserted <= INT_MAX, i = (int64_t) blockDim.x * blockIdx.x + tid
//   reduce_rows_f32, before: x[row * ncols + i] in int;  after: x[(int64_t) row * ncols + i]
// A launch is valid when 1 <= blocks <= 2^31 - 1 (CUDA's grid x limit) and the blocks cover every element; an index is
// right when the largest one, computed the kernel's way, equals the true largest offset. The old code must fail
// exactly where the pods did (2^31 elements and up: 131072 x 512 crashed on a B300 with "invalid argument" in
// ggml_cuda_op_relu), the new code nowhere up to 32768 x 8192. With the fork tree's path as argument it also checks
// that the tree has the new code and not the old.
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

constexpr int64_t kIndexerHeads = 32;
constexpr int64_t kBlock = 256;  // CUDA_NEG_BLOCK_SIZE
constexpr uint64_t kMaxGridX = 2147483647ull;

// The old unary_cuda's grid: k narrowed to int (two's complement, as gcc and nvcc do), int block count, then dim3's
// unsigned int. Returns the block count CUDA would be asked for.
uint64_t old_unary_blocks(int64_t n) {
    const int k = (int) (uint32_t) (uint64_t) n;
    const int num_blocks = (k + (int) kBlock - 1) / (int) kBlock;
    return (uint64_t) (unsigned int) num_blocks;
}

// The old kernel's largest index i = blockDim.x * blockIdx.x + threadIdx.x in int, for the old grid.
int64_t old_unary_last_index(int64_t n) {
    const uint64_t blocks = old_unary_blocks(n);
    const uint32_t i = (uint32_t) (kBlock * (blocks - 1) + (kBlock - 1));  // unsigned product, then int
    const int64_t last = (int64_t) (int) i;
    // the kernel returns for i >= k: the largest index it writes is min(last, k - 1) in its own int arithmetic
    const int k = (int) (uint32_t) (uint64_t) n;
    return last < k ? last : (int64_t) k - 1;
}

uint64_t new_unary_blocks(int64_t n) { return (uint64_t) ((n + kBlock - 1) / kBlock); }

int64_t new_unary_last_index(int64_t n) {
    const int64_t blocks = (int64_t) new_unary_blocks(n);
    const int64_t last = blocks * kBlock - 1;
    return last < n ? last : n - 1;
}

// reduce_rows_f32 over the sum_rows of the scores: ncols = 32 heads, nrows = n / 32, the last row's last element.
int64_t old_reduce_last(int64_t n) {
    const int ncols = (int) kIndexerHeads, row = (int) (n / kIndexerHeads - 1), i = ncols - 1;
    return (int64_t) (int) ((uint32_t) row * (uint32_t) ncols + (uint32_t) i);  // int product, wrapped
}

int64_t new_reduce_last(int64_t n) {
    const int ncols = (int) kIndexerHeads, row = (int) (n / kIndexerHeads - 1), i = ncols - 1;
    return (int64_t) row * ncols + i;
}

bool has(const std::string& text, const char* needle) { return text.find(needle) != std::string::npos; }

std::string read_file(const std::string& path, bool* ok) {
    std::ifstream f(path);
    std::stringstream s;
    s << f.rdbuf();
    *ok = (bool) f;
    return s.str();
}

}  // namespace

int main(int argc, char** argv) {
    struct Shape {
        int64_t ctx, ubatch;
        bool old_ok;  // the old code works here
    } shapes[] = {
        {4096, 4096, true},      {16384, 2048, true},    {16384, 4095, true},   {16384, 4096, false},
        {32768, 2048, false},    {32768, 4096, false},   {131072, 256, true},   {131072, 512, false},
        {32768, 8192, false},
    };
    int failed = 0, cases = 0;
    std::printf("dsv41_indexer_index_test: score tensor n_comp x ubatch x %lld f32, n_comp = ctx; unary_cuda blocks "
                "of %lld; CUDA grid x <= %llu\n", (long long) kIndexerHeads, (long long) kBlock,
                (unsigned long long) kMaxGridX);
    for (const Shape& s : shapes) {
        const int64_t n = s.ctx * s.ubatch * kIndexerHeads;
        const uint64_t ob = old_unary_blocks(n), nb = new_unary_blocks(n);
        const bool old_grid_ok = ob >= 1 && ob <= kMaxGridX && ob * (uint64_t) kBlock >= (uint64_t) n;
        const bool old_index_ok = old_unary_last_index(n) == n - 1;
        const bool old_reduce_ok = old_reduce_last(n) == n - 1;
        const bool new_grid_ok = nb >= 1 && nb <= (uint64_t) INT_MAX && nb * (uint64_t) kBlock >= (uint64_t) n;
        const bool new_index_ok = new_unary_last_index(n) == n - 1;
        const bool new_reduce_ok = new_reduce_last(n) == n - 1;
        const bool old_all = old_grid_ok && old_index_ok && old_reduce_ok;
        // the old code: fails exactly where expected; the new code: always right
        const bool pass = old_all == s.old_ok && new_grid_ok && new_index_ok && new_reduce_ok;
        failed += !pass;
        ++cases;
        std::printf("%-4s ctx %6lld x ubatch %4lld: %11lld elements (2^%.2f)  old: blocks %10llu (%s), last index %s, "
                    "sum_rows last offset %s  new: blocks %8llu, last index and sum_rows offset %s\n",
                    pass ? "ok" : "FAIL", (long long) s.ctx, (long long) s.ubatch, (long long) n,
                    std::log2((double) n), (unsigned long long) ob,
                    old_grid_ok ? "valid grid" : "invalid argument", old_index_ok ? "right" : "WRONG",
                    old_reduce_ok ? "right" : "WRONG", (unsigned long long) nb,
                    new_grid_ok && new_index_ok && new_reduce_ok ? "right" : "WRONG");
    }

    if (argc > 1) {
        const std::string fork = argv[1];
        bool ok_unary = false, ok_reduce = false;
        const std::string unary = read_file(fork + "/ggml/src/ggml-cuda/unary.cu", &ok_unary);
        const std::string reduce = read_file(fork + "/ggml/src/ggml-cuda/reduce_rows.cuh", &ok_reduce);
        const struct {
            const char* what;
            bool pass;
        } checks[] = {
            {"unary.cu is readable", ok_unary},
            {"reduce_rows.cuh is readable", ok_reduce},
            {"unary_op_kernel takes const int64_t k",
             has(unary, "static __global__ void unary_op_kernel(const T * x, T * dst, const int64_t k)")},
            {"unary_op_kernel's index is int64_t",
             has(unary, "const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;")},
            {"unary_cuda takes const int64_t k",
             has(unary, "static void unary_cuda(const T * x, T * dst, const int64_t k, cudaStream_t stream)")},
            {"unary_cuda asserts its grid", has(unary, "GGML_ASSERT(num_blocks <= INT_MAX);")},
            {"the old int k is gone", !has(unary, "static void unary_cuda(const T * x, T * dst, const int k,")},
            {"reduce_rows_f32 indexes in int64_t", has(reduce, "x[(int64_t) row * ncols + i]")},
            {"the old int index is gone", !has(reduce, "x[row * ncols + i]")},
        };
        for (const auto& c : checks) {
            failed += !c.pass;
            ++cases;
            std::printf("%-4s fork %s: %s\n", c.pass ? "ok" : "FAIL", fork.c_str(), c.what);
        }
    }
    std::printf("dsv41_indexer_index_test: %s (%d of %d cases failed)\n", failed ? "FAIL" : "PASS", failed, cases);
    return failed ? 1 : 0;
}
