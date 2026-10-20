// tests/dsv41/engine/cpu_experts_probe.cpp - dsv41_cpu_experts_probe: the CPU experts part of a pod session's machine
// qualification. The single user's gap between sessions (57.06 and 56.89 tok/s against 61.55) is the host's
// CPU expert rate alone, 44-45 GB/s on the EPYC 9575F and 9355P hosts against 100.4 on an EPYC 9655, and neither the machine
// probe's host read (585-619 GB/s on the slow hosts) nor the engine's
// init rate predicts it. So this measures what decode's misses do, with the engine's own code: one token per expert
// through CpuExperts::run in the fork's numerics (the engine's default), one expert
// per call (decode: 13.9 misses a token over 40 layers' calls), the pool spinning the engine's default 50 us, on
// synthetic q2_K gate/up and q3_K down experts at the model's dims (DenseDims' defaults, which dense_dims_real returns:
// 12,810,240 B each, the layout of vcruz305 Q2_K c4a08554). kExperts distinct experts, about twice the largest EPYC L3
// (1152 MiB, 9684X), so every run reads its weights from RAM (pageable memory here, pinned in the engine: the same DRAM
// reads). tools/dsv41/pod/pod-session.sh runs it after dsv41_machine_probe passes, before the payload, with the
// engine's --threads 16 and, when the GPU has a NUMA node, --numa-node N: the engine's numactl --cpunodebind
// (e07-run.sh) as this process's CPU affinity, the weights first touched there.
//
//   dsv41_cpu_experts_probe --threads N [--min-gbs G] [--numa-node N]
//
// Prints
//   dsv41_cpu_experts_probe: measure cpu_experts_gbs VALUE bound >=G ok|FAIL (how it was measured)
// (bound "none" and status "info" without --min-gbs), the CPUs its threads may run on and last ran on, the machine
// line, then "dsv41_cpu_experts_probe: PASS, ..." or "dsv41_cpu_experts_probe: FAIL, ...". Exit 1 on a FAIL (the
// rate under its bound, or a NUMA node that cannot be bound), 2 on bad arguments. GPU free: it runs on the dev VM.
#include "qmatmul_synth.hpp"

#include "strata/dsv41/engine/dense.hpp"
#include "strata/dsv41/engine/experts.hpp"

#include <dirent.h>
#include <sched.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

constexpr int kExperts = 192;   // 2.46 GB a run, twice the 1152 MiB L3 of the largest EPYC
constexpr int kRuns = 7;        // timed, after one untimed run (page faults, the pool's first wake)
constexpr int kSpinUs = 50;     // ExpertConfig::pool_spin_us, the engine's default

int usage(const char* why) {
    std::fprintf(stderr, "dsv41_cpu_experts_probe: %s\nusage: dsv41_cpu_experts_probe --threads N [--min-gbs G] "
                         "[--numa-node N]\n", why);
    return 2;
}

std::string read_text(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return "";
    std::string out;
    char b[4096];
    for (size_t n; (n = std::fread(b, 1, sizeof b, f)) > 0;) out.append(b, n);
    std::fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

// the value of the first "key:" line of a /proc file, trimmed ("" when absent)
std::string proc_field(const char* path, const char* key) {
    const std::string t = read_text(path);
    for (size_t s = 0; s < t.size();) {
        size_t n = t.find('\n', s);
        if (n == std::string::npos) n = t.size();
        const std::string line = t.substr(s, n - s);
        s = n + 1;
        const size_t c = line.find(':');
        if (c == std::string::npos) continue;
        std::string k = line.substr(0, c);
        while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();
        if (k != key) continue;
        std::string v = line.substr(c + 1);
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
        return v;
    }
    return "";
}

// a kernel cpulist ("0-15,32-47") into a set; false when it does not parse
bool parse_cpulist(const std::string& s, std::set<int>& out) {
    const char* p = s.c_str();
    while (*p) {
        char* end = nullptr;
        const long a = std::strtol(p, &end, 10);
        if (end == p || a < 0) return false;
        long b = a;
        p = end;
        if (*p == '-') {
            b = std::strtol(p + 1, &end, 10);
            if (end == p + 1 || b < a) return false;
            p = end;
        }
        for (long c = a; c <= b; ++c) out.insert((int) c);
        if (*p == ',') ++p;
        else if (*p) return false;
    }
    return !out.empty();
}

std::string cpulist(const std::set<int>& s) {
    std::string out;
    for (auto it = s.begin(); it != s.end();) {
        const int a = *it;
        int b = a;
        for (++it; it != s.end() && *it == b + 1; ++it) b = *it;
        out += (out.empty() ? "" : ",") + std::to_string(a) + (b > a ? "-" + std::to_string(b) : "");
    }
    return out.empty() ? "none" : out;
}

// the CPU each of this process's threads last ran on (/proc/self/task/*/stat field 39), and how many threads
std::set<int> threads_last_cpus(int& n_threads) {
    std::set<int> cpus;
    n_threads = 0;
    DIR* d = opendir("/proc/self/task");
    if (!d) return cpus;
    for (dirent* t; (t = readdir(d));) {
        if (t->d_name[0] == '.') continue;
        const std::string st = read_text(std::string("/proc/self/task/") + t->d_name + "/stat");
        const size_t paren = st.rfind(')');
        if (paren == std::string::npos) continue;
        // after "pid (comm)" the fields start at 3 (state): field 39 is the 37th
        const char* p = st.c_str() + paren + 1;
        int field = 2;
        long cpu = -1;
        while (*p && field < 39) {
            while (*p == ' ') ++p;
            ++field;
            if (field == 39) cpu = std::strtol(p, nullptr, 10);
            while (*p && *p != ' ') ++p;
        }
        if (cpu >= 0) cpus.insert((int) cpu);
        ++n_threads;
    }
    closedir(d);
    return cpus;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    int threads = 0, numa_node = -1;
    double min_gbs = 0;
    for (int a = 1; a < argc; ++a) {
        const std::string k = argv[a];
        if (a + 1 >= argc) return usage((k + " needs a value").c_str());
        char* end = nullptr;
        const char* v = argv[++a];
        if (k == "--threads") {
            const long n = std::strtol(v, &end, 10);
            if (*end != '\0' || n < 1 || n > 1024) return usage("--threads needs a whole number of 1 to 1024");
            threads = (int) n;
        } else if (k == "--min-gbs") {
            min_gbs = std::strtod(v, &end);
            if (*end != '\0' || !(min_gbs > 0)) return usage("--min-gbs needs a number above 0");
        } else if (k == "--numa-node") {
            const long n = std::strtol(v, &end, 10);
            if (*end != '\0' || n < 0) return usage("--numa-node needs a node number >= 0");
            numa_node = (int) n;
        } else {
            return usage(("unknown argument " + k).c_str());
        }
    }
    if (!threads) return usage("no --threads (the engine's --threads, 16 in e07-run.sh)");

    // the engine's binding first, so the pool's threads inherit it and the weights are first touched on the node
    std::string bind = "unbound (no --numa-node)";
    if (numa_node >= 0) {
        const std::string path = "/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist";
        std::set<int> node, allowed;
        cpu_set_t cur;
        CPU_ZERO(&cur);
        if (!parse_cpulist(read_text(path), node)) {
            std::printf("dsv41_cpu_experts_probe: FAIL NUMA node %d: no CPU list in %s\n", numa_node, path.c_str());
            return 1;
        }
        if (sched_getaffinity(0, sizeof cur, &cur) != 0) {
            std::printf("dsv41_cpu_experts_probe: FAIL sched_getaffinity: %s\n", std::strerror(errno));
            return 1;
        }
        cpu_set_t want;
        CPU_ZERO(&want);
        for (int c : node)
            if (c < CPU_SETSIZE && CPU_ISSET(c, &cur)) {
                CPU_SET(c, &want);
                allowed.insert(c);
            }
        if (allowed.empty() || sched_setaffinity(0, sizeof want, &want) != 0) {
            std::printf("dsv41_cpu_experts_probe: FAIL NUMA node %d (CPUs %s): none of them in this process's CPUs or "
                        "sched_setaffinity refused\n", numa_node, cpulist(node).c_str());
            return 1;
        }
        bind = "bound to NUMA node " + std::to_string(numa_node) + "'s CPUs " + cpulist(allowed) +
               " (e07-run.sh's numactl --cpunodebind)";
    }

    // dense_dims_real() is DenseDims{}; its definition (dense.cu) would link the CUDA kernels and cuBLAS, which the
    // qualification uploads without
    const e::DenseDims real{};
    const int64_t E = real.n_embd, F = real.n_ff;
    e::DenseDims d = real;
    d.n_expert = kExperts;
    const int types[3] = {e::kQTypeQ2K, e::kQTypeQ2K, e::kQTypeQ3K};   // up, gate, down: the model's
    std::vector<uint8_t> part[3];
    e::ExpertLayerHost L;
    L.layer = 0;
    L.limit = real.swiglu_limit;
    const auto t_fill = std::chrono::steady_clock::now();
    std::mt19937_64 rng(20261019u);
    for (int p = 0; p < 3; ++p) {
        const int64_t rows = p == 2 ? E : F, k = p == 2 ? F : E;
        part[p].resize(qsynth::random_blocks_bytes(types[p], rows * kExperts, k));
        qsynth::random_blocks_into(part[p].data(), types[p], rows * kExperts, k, p == 2 ? 1.f / 48.f : 1.f / 72.f, rng);
        L.type[p] = types[p];
        L.data[p] = part[p].data();
        L.bytes[p] = part[p].size() / kExperts;
        L.name[p] = p == 0 ? "up" : p == 1 ? "gate" : "down";
    }
    const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fill).count();
    e::cpu_experts_check(L, d, e::CpuNumerics::Fork);   // exits naming what this CPU lacks (AVX2, FMA, F16C)

    e::CpuExperts cpu(threads, e::CpuNumerics::Fork);
    cpu.set_spin_us(kSpinUs);
    std::vector<float> x((size_t) E), y((size_t) E);
    std::mt19937 g(29u);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (float& v : x) v = nd(g);
    std::vector<e::CpuExpertJob> job(1);
    job[0].tokens = {0};
    const double expert_bytes = (double) (L.bytes[0] + L.bytes[1] + L.bytes[2]);
    const double run_bytes = expert_bytes * kExperts;
    std::vector<double> gbs;
    for (int r = 0; r <= kRuns; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int ex = 0; ex < kExperts; ++ex) {
            for (int p = 0; p < 3; ++p) job[0].rows.part[p] = L.data[p] + (size_t) ex * L.bytes[p];
            cpu.run(L, d, x.data(), 1, job.data(), 1, y.data());
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (r) gbs.push_back(run_bytes / (s * 1e9));
    }
    const double rate = median(gbs);
    char how[640];
    std::snprintf(how, sizeof how, "median of %d runs after one, min %.1f max %.1f; a run: %d distinct synthetic "
                  "experts of %.2f MB (q2_K gate/up, q3_K down, n_embd %lld, n_ff %lld), %.2f GB, one token per expert, one "
                  "expert per CpuExperts::run, fork numerics, %d pool threads, spin %d us; %s", kRuns,
                  *std::min_element(gbs.begin(), gbs.end()), *std::max_element(gbs.begin(), gbs.end()), kExperts,
                  expert_bytes / 1e6, (long long) E, (long long) F, run_bytes / 1e9, threads, kSpinUs, bind.c_str());
    const bool ok = !(min_gbs > 0) || rate >= min_gbs;
    if (min_gbs > 0)
        std::printf("dsv41_cpu_experts_probe: measure cpu_experts_gbs %.1f bound >=%g %s (%s)\n", rate, min_gbs,
                    ok ? "ok" : "FAIL", how);
    else
        std::printf("dsv41_cpu_experts_probe: measure cpu_experts_gbs %.1f bound none info (%s)\n", rate, how);

    // where the pool ran: the CPUs it may use (the affinity after the binding, the cgroup's cpuset) and last used
    int n_threads = 0;
    const std::set<int> last = threads_last_cpus(n_threads);
    const std::string cpuset = read_text("/sys/fs/cgroup/cpuset.cpus.effective");
    std::printf("dsv41_cpu_experts_probe: threads: %d (the caller and %d pool workers) last ran on CPUs %s; allowed "
                "CPUs %s; cgroup cpuset.cpus.effective %s; fill %.1f s\n", n_threads, threads - 1,
                cpulist(last).c_str(), proc_field("/proc/self/status", "Cpus_allowed_list").c_str(),
                cpuset.empty() ? "none (no /sys/fs/cgroup/cpuset.cpus.effective)" : cpuset.c_str(), fill_s);
    const std::string model = proc_field("/proc/cpuinfo", "model name");
    std::printf("dsv41_cpu_experts_probe: machine CPU %s, %u CPUs online, MemTotal %s\n",
                model.empty() ? "unknown" : model.c_str(), std::thread::hardware_concurrency(),
                proc_field("/proc/meminfo", "MemTotal").c_str());
    if (ok) {
        std::printf("dsv41_cpu_experts_probe: PASS, %s\n", min_gbs > 0 ? "the CPU experts rate within its bound"
                                                                        : "no bound given (info only)");
        return 0;
    }
    std::printf("dsv41_cpu_experts_probe: FAIL, cpu_experts_gbs %.1f (bound >=%g)\n", rate, min_gbs);
    return 1;
}
