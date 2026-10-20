// tests/dsv41/engine/expert_bits_test_gpu.cu - the
// CUDA expert kernels against the CPU experts in the GPU's numerics, bit for bit, on synthetic tensors (a dev pod; no
// model).
//
//   dsv41_engine_expert_bits_test --gpu
//
//   (g1) one matvec: qact_quantize's q8_1 bytes against mmvq_cpu::quantize's, and qmatmul_q's MMVQ (one column:
//        mmvq1_kernel on staged rows; 2 to 8 columns: mmvq_kernel) against mmvq_cpu::rows, every value of random q2_K
//        [2304][5120], q3_K [5120][2304] (the real gate/up and down) and q2_K [64][256] matrices at 1, 2, 4 and 8
//        activations;
//   (g2) the FFN: qmatmul_experts (decode's hits: gate, up and the clamp in one launch, h to q8_1, down; one token: the
//        per-pair kernels, 2 to 8: the grouped kernel) against CpuExperts in the GPU numerics, every pair's row, 8
//        real-size experts all in the slab, 6 pairs a token at 1, 2, 4 and 8 tokens;
//   (g3) the weighted sum: the device's slot-order combine (expert_combine) of every pair from the slab against the
//        same combine with the odd experts' rows from the CPU (the doorbell's row map: a miss's row in ycpu), and
//        against ref_expert_combine of the CPU's rows on the host.
//   (g0) not a check, an info line: the upload order an earlier (g1) had, a pageable
//        cudaMemcpy and then the quantizer on this non-blocking stream at once, against the CPU's q8_1 on 32 trials.
// The dev VM checks the CPU experts against the line-for-line reference (expert_bits_test.cpp); this closes the chain.
// Every upload waits for its copy to land (upload below): the kernels run on a cudaStreamNonBlocking stream, which
// is not ordered after the legacy stream's cudaMemcpy, and a pageable cudaMemcpy to the device may return before its
// DMA has landed (CUDA runtime API, "API synchronization behavior"). PASS or FAIL lines, exit 1 on any FAIL.
#include "qmatmul_synth.hpp"

#include "../../../src/dsv41/engine/experts_cpu_mmvq.hpp"
#include "../../../src/dsv41/engine/ffn_kernels.hpp"
#include "strata/dsv41/engine/experts.hpp"
#include "strata/dsv41/engine/mmvq_ref.hpp"
#include "strata/dsv41/engine/qmatmul.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

void ck(cudaError_t r, const char* w) {
    if (r != cudaSuccess) {
        std::fprintf(stderr, "dsv41_engine_expert_bits_test --gpu: %s: %s\n", w, cudaGetErrorString(r));
        std::exit(2);
    }
}
template <typename T> T* dev_alloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, (n ? n : 1) * sizeof(T)), "cudaMalloc");
    return p;
}
// the copy landed before any kernel on the test's stream reads it (the header's note; (g0) shows the old order)
template <typename T> T* upload(const std::vector<T>& h) {
    T* p = dev_alloc<T>(h.size());
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    ck(cudaDeviceSynchronize(), "upload");
    return p;
}
template <typename T> std::vector<T> download(const T* p, size_t n) {
    std::vector<T> h(n);
    if (n) ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
// equal bits, or both NaN (the device's NaN is canonical, the host's keeps a payload)
int64_t differ(const float* a, const float* b, size_t n) {
    int64_t d = 0;
    for (size_t i = 0; i < n; ++i) d += std::memcmp(&a[i], &b[i], 4) != 0 && !(std::isnan(a[i]) && std::isnan(b[i]));
    return d;
}
std::vector<float> normal(size_t n, float scale, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(g);
    return v;
}

constexpr int64_t kE = 5120, kF = 2304;
constexpr float kLimit = 10.f;

// (g1)
void matvec(cudaStream_t st) {
    e::QmmContext qmm;
    std::string err;
    if (!qmm.init(err, e::QPrefill::DequantF16, 20480)) {
        expect(false, "(g1) qmatmul context: " + err);
        return;
    }
    struct Shape {
        const char* name;
        int type;
        int64_t rows, k;
    };
    // gate/up and down are also the shared expert's shapes: every K a q8_1 activation of the experts has (5120, 2304)
    const Shape shapes[] = {{"gate/up q2_K [2304][5120]", e::kQTypeQ2K, kF, kE},
                            {"down q3_K [5120][2304]", e::kQTypeQ3K, kE, kF},
                            {"q2_K [64][256]", e::kQTypeQ2K, 64, 256}};
    std::mt19937_64 rng(41u);
    for (const Shape& sh : shapes) {
        const std::vector<uint8_t> wh = qsynth::random_blocks(sh.type, sh.rows, sh.k, 1.f / std::sqrt((float) sh.k), rng);
        const size_t row_bytes = wh.size() / (size_t) sh.rows;
        void* wd = upload(wh);
        const e::QWeightView w{sh.type, sh.k, sh.rows, wd};
        const size_t row_q = (size_t) (sh.k / 32) * e::kQ81BlockBytes;
        int64_t q_bad = 0, y_bad = 0, values = 0;
        std::string per_t;   // each T's differing blocks and values, so a failure names where it is
        for (int64_t T : {1, 2, 4, 8}) {
            const int64_t q_before = q_bad, y_before = y_bad;
            const std::vector<float> x = normal((size_t) (T * sh.k), 1.f, (uint32_t) (T * 7 + sh.rows));
            float* xd = upload(x);
            void* qd = dev_alloc<uint8_t>(e::qact_bytes(e::QActLayout::Q8_1, T, sh.k));
            float* yd = dev_alloc<float>((size_t) (T * sh.rows));
            e::QActs q;
            e::qact_quantize(xd, nullptr, sh.k, T, sh.k, e::QActLayout::Q8_1, qd, st, q);
            e::qmatmul_q(qmm, w, q, yd, sh.rows, 1, st, sh.name);
            ck(cudaStreamSynchronize(st), "(g1) qmatmul_q");
            const std::vector<uint8_t> gq = download(static_cast<const uint8_t*>(qd), (size_t) T * row_q);
            const std::vector<float> gy = download(yd, (size_t) (T * sh.rows));
            std::vector<uint8_t> cq((size_t) T * row_q);
            std::vector<std::vector<float>> cy((size_t) T, std::vector<float>((size_t) sh.rows));
            std::vector<float> cf((size_t) (T * e::mmvq_cpu::act_floats(sh.k)));
            std::vector<e::mmvq_cpu::Act> acts;
            std::vector<float*> outs;
            for (int64_t t = 0; t < T; ++t) {
                e::mmvq_cpu::quantize(x.data() + t * sh.k, sh.k, cq.data() + (size_t) t * row_q);
                acts.push_back(e::mmvq_cpu::prepare_act(cq.data() + (size_t) t * row_q, sh.k,
                                                         cf.data() + t * e::mmvq_cpu::act_floats(sh.k)));
                outs.push_back(cy[(size_t) t].data());
            }
            for (size_t b = 0; b < cq.size(); b += e::kQ81BlockBytes)
                q_bad += std::memcmp(cq.data() + b, gq.data() + b, e::kQ81BlockBytes) != 0;
            e::mmvq_cpu::rows(sh.type, wh.data(), row_bytes, 0, sh.rows, sh.k, acts.data(), outs.data(), T);
            for (int64_t t = 0; t < T; ++t) y_bad += differ(cy[(size_t) t].data(), gy.data() + t * sh.rows, (size_t) sh.rows);
            values += T * sh.rows;
            per_t += (per_t.empty() ? "" : ", ") + std::to_string(T) + ": " + std::to_string(q_bad - q_before) + "/" +
                     std::to_string(y_bad - y_before);
            for (void* p : {(void*) xd, qd, (void*) yd}) cudaFree(p);
        }
        char what[320];
        std::snprintf(what, sizeof what, "(g1) %s at 1, 2, 4 and 8 activations: %lld q8_1 blocks differ from the CPU's, "
                      "%lld of %lld MMVQ values differ from mmvq_cpu::rows (blocks/values by T: %s)", sh.name,
                      (long long) q_bad, (long long) y_bad, (long long) values, per_t.c_str());
        expect(q_bad == 0 && y_bad == 0, what);
        cudaFree(wd);
    }
}

// (g0) the upload order an earlier (g1) had: x by a pageable cudaMemcpy (the legacy stream), then qact_quantize on the
// non-blocking stream at once, with nothing between them that waits for the copy. One device buffer takes a new x per
// trial, so a launch that ran before the DMA landed reads the previous trial's values there and its q8_1 differs
// from the CPU's. An info line, not a check: whether a trial races depends on the host, the driver and the copy's
// size (the L4 of 2026-10-17 showed it at the 5120-wide gate/up input, in an earlier (g1)).
void upload_order_probe(cudaStream_t st) {
    constexpr int64_t T = 8, k = kE;
    constexpr int kTrials = 32;
    const size_t row_q = (size_t) (k / 32) * e::kQ81BlockBytes;
    float* xd = dev_alloc<float>((size_t) (T * k));
    void* qd = dev_alloc<uint8_t>(e::qact_bytes(e::QActLayout::Q8_1, T, k));
    std::vector<uint8_t> cq((size_t) T * row_q);
    int raced = 0;
    int64_t bad = 0;
    for (int i = 0; i < kTrials; ++i) {
        const std::vector<float> x = normal((size_t) (T * k), 1.f, (uint32_t) (1000 + i));
        ck(cudaMemcpy(xd, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "(g0) x");
        e::QActs q;
        e::qact_quantize(xd, nullptr, k, T, k, e::QActLayout::Q8_1, qd, st, q);
        ck(cudaStreamSynchronize(st), "(g0) qact_quantize");
        const std::vector<uint8_t> gq = download(static_cast<const uint8_t*>(qd), (size_t) T * row_q);
        for (int64_t t = 0; t < T; ++t) e::mmvq_cpu::quantize(x.data() + t * k, k, cq.data() + (size_t) t * row_q);
        int64_t n = 0;
        for (size_t b = 0; b < cq.size(); b += e::kQ81BlockBytes)
            n += std::memcmp(cq.data() + b, gq.data() + b, e::kQ81BlockBytes) != 0;
        raced += n > 0;
        bad += n;
    }
    std::printf("info (g0) round 1's upload order (a pageable cudaMemcpy of x [8][5120], then qact_quantize on the "
                "non-blocking stream at once), %d trials: %d read input that had not landed (%lld q8_1 blocks differ from "
                "the CPU's)\n", kTrials, raced, (long long) bad);
    for (void* p : {(void*) xd, qd}) cudaFree(p);
}

// (g2) and (g3)
void ffn(cudaStream_t st) {
    constexpr int NE = 8, K = 6;
    const int types[3] = {e::kQTypeQ2K, e::kQTypeQ2K, e::kQTypeQ3K};   // up, gate, down
    // the host layer (CpuExperts reads every expert at data[p] + e bytes[p]) and the slab (slot e = expert e)
    std::vector<uint8_t> part[3];
    e::ExpertLayerHost L;
    std::mt19937_64 rng(43u);
    size_t off[3], slot_bytes = 0;
    for (int p = 0; p < 3; ++p) {
        const int64_t rows = p == 2 ? kE : kF, k = p == 2 ? kF : kE;
        part[p] = qsynth::random_blocks(types[p], rows * NE, k, p == 2 ? 1.f / 48.f : 1.f / 72.f, rng);
        L.type[p] = types[p];
        L.data[p] = part[p].data();
        L.bytes[p] = part[p].size() / NE;
        L.name[p] = p == 0 ? "up" : p == 1 ? "gate" : "down";
        off[p] = slot_bytes;
        slot_bytes += (L.bytes[p] + 255) / 256 * 256;
    }
    L.layer = 0;
    L.limit = kLimit;
    e::DenseDims d;
    d.n_embd = (int) kE;
    d.n_ff = (int) kF;
    d.n_expert = NE;
    d.n_expert_used = K;
    std::vector<uint8_t> slab((size_t) NE * slot_bytes, 0);
    for (int x = 0; x < NE; ++x)
        for (int p = 0; p < 3; ++p) std::memcpy(slab.data() + (size_t) x * slot_bytes + off[p], L.data[p] + (size_t) x * L.bytes[p], L.bytes[p]);
    std::vector<int32_t> slot_of((size_t) NE);
    for (int x = 0; x < NE; ++x) slot_of[(size_t) x] = x;
    void* sd = upload(slab);
    int32_t* so = upload(slot_of);
    e::QExpertSlab sl;
    sl.base = sd;
    sl.slot_of = so;
    sl.slot_bytes = slot_bytes;
    for (int p = 0; p < 3; ++p) {
        sl.part_offset[p] = off[p];
        sl.type[p] = types[p];
    }
    sl.n_embd = kE;
    sl.n_ff = kF;
    e::qmatmul_experts_check(sl, "(g2) slab");
    e::cpu_experts_check(L, d);
    const int64_t Tm = e::kMmvqMaxCols, Pm = Tm * K;
    void* scratch = dev_alloc<uint8_t>(e::qmatmul_experts_scratch_bytes(sl, Tm, Pm));
    float* hd = dev_alloc<float>((size_t) (Pm * kF));
    float* yd = dev_alloc<float>((size_t) (Pm * kE));
    float* ycd = dev_alloc<float>((size_t) (Pm * kE));
    float* outd = dev_alloc<float>((size_t) (Tm * kE));
    e::CpuExperts cpu(4);
    std::mt19937 pick(47u);
    for (int64_t T : {1, 2, 4, 8}) {
        const int64_t P = T * K;
        std::vector<int32_t> ids;
        for (int64_t t = 0; t < T; ++t) {
            std::vector<int32_t> all(NE);
            for (int i = 0; i < NE; ++i) all[(size_t) i] = i;
            std::shuffle(all.begin(), all.end(), pick);
            ids.insert(ids.end(), all.begin(), all.begin() + K);
        }
        const std::vector<float> x = normal((size_t) (T * kE), 1.f, (uint32_t) (100 + T));
        float* xd = upload(x);
        int32_t* idd = upload(ids);
        e::qmatmul_experts(sl, xd, T, idd, K, kLimit, hd, scratch, yd, st);
        ck(cudaStreamSynchronize(st), "(g2) qmatmul_experts");
        const std::vector<float> gy = download(yd, (size_t) (P * kE));
        // the CPU experts on every pair, grouped as decode's misses
        std::vector<int32_t> none((size_t) NE, -1), job_of((size_t) NE, -1), job_expert, miss((size_t) P);
        std::vector<e::CpuExpertJob> jobs((size_t) NE);
        int64_t hits = 0;
        const size_t nj = e::decode_miss_jobs(ids.data(), P, K, none.data(), job_of.data(), job_expert, jobs.data(), miss.data(), hits);
        int64_t rows = 0;
        for (size_t j = 0; j < nj; ++j) {
            for (int p = 0; p < 3; ++p) jobs[j].rows.part[p] = L.data[p] + (size_t) job_expert[j] * L.bytes[p];
            rows += (int64_t) jobs[j].tokens.size();
        }
        std::vector<float> yc((size_t) (rows * kE)), cy((size_t) (P * kE));
        cpu.run(L, d, x.data(), T, jobs.data(), nj, yc.data());
        for (int64_t p = 0; p < P; ++p) std::memcpy(cy.data() + p * kE, yc.data() + (size_t) miss[(size_t) p] * kE, (size_t) kE * 4);
        const int64_t bad = differ(cy.data(), gy.data(), cy.size());
        char what[320];
        std::snprintf(what, sizeof what, "(g2) qmatmul_experts (gate/up q2_K [2304][5120], down q3_K [5120][2304]) at %lld "
                      "token%s (%lld pairs, %zu experts): %lld of %lld values differ from CpuExperts in the GPU numerics",
                      (long long) T, T > 1 ? "s" : "", (long long) P, nj, (long long) bad, (long long) cy.size());
        expect(bad == 0, what);

        // (g3): the odd experts' rows from the CPU through the doorbell's row map, the even ones from the slab
        std::uniform_real_distribution<float> wd(0.05f, 0.6f);
        std::vector<float> w((size_t) P);
        for (float& v : w) v = wd(pick);
        float* wdev = upload(w);
        std::vector<int32_t> all_gpu((size_t) P), mixed((size_t) P);
        std::vector<float> ycpu;
        for (int64_t p = 0; p < P; ++p) {
            all_gpu[(size_t) p] = (int32_t) p;
            if (ids[(size_t) p] % 2 == 0) {
                mixed[(size_t) p] = (int32_t) p;
            } else {
                mixed[(size_t) p] = -(int32_t) (ycpu.size() / kE) - 1;
                ycpu.insert(ycpu.end(), cy.begin() + p * kE, cy.begin() + (p + 1) * kE);
            }
        }
        ck(cudaMemcpy(ycd, ycpu.data(), std::max<size_t>(ycpu.size(), 1) * 4, cudaMemcpyHostToDevice), "(g3) ycpu");
        int32_t* ra = upload(all_gpu);
        int32_t* rm = upload(mixed);
        e::expert_combine(yd, ycd, ra, wdev, nullptr, T, K, (int) kE, outd, st);
        ck(cudaStreamSynchronize(st), "(g3) combine, all hits");
        const std::vector<float> oa = download(outd, (size_t) (T * kE));
        e::expert_combine(yd, ycd, rm, wdev, nullptr, T, K, (int) kE, outd, st);
        ck(cudaStreamSynchronize(st), "(g3) combine, odd experts from the CPU");
        const std::vector<float> om = download(outd, (size_t) (T * kE));
        std::vector<float> oh((size_t) (T * kE));
        e::ref_expert_combine(cy.data(), w.data(), T, K, (int) kE, nullptr, oh.data());
        const int64_t mix_bad = differ(om.data(), oa.data(), oa.size()), host_bad = differ(oh.data(), oa.data(), oa.size());
        std::snprintf(what, sizeof what, "(g3) the weighted sum of those [5120] rows at %lld token%s: the odd experts from "
                      "the CPU %lld values differ from all from the slab, the host's combine of the CPU's rows %lld", (long long) T,
                      T > 1 ? "s" : "", (long long) mix_bad, (long long) host_bad);
        expect(mix_bad == 0 && host_bad == 0 && !ycpu.empty(), what);
        for (void* p : {(void*) xd, (void*) idd, (void*) wdev, (void*) ra, (void*) rm}) cudaFree(p);
    }
    for (void* p : {sd, (void*) so, scratch, (void*) hd, (void*) yd, (void*) ycd, (void*) outd}) cudaFree(p);
}

}  // namespace

int run_gpu() {
    int dev = -1;
    cudaDeviceProp prop{};
    ck(cudaGetDevice(&dev), "cudaGetDevice");
    ck(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    std::printf("dsv41_engine_expert_bits_test --gpu on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
    matvec(st);
    ffn(st);
    upload_order_probe(st);
    cudaStreamDestroy(st);
    std::printf("dsv41_engine_expert_bits_test --gpu: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
