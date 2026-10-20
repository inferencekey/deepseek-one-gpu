// src/dsv41/moe_residency_call.cpp - the residency's work on one op call of strata-moe (moe_residency.h): the first
// call of a layer binds it (prepare), every call gets the base rows of its missed experts (bind) and, with
// STRATA_MOE_COLD=direct, the rounds of O_DIRECT reads its CPU phases wait for (issue, wait, round_done, finish). The
// tiers themselves (the warm buffer, its fill, the reader threads) are moe_residency.cpp, which takes plain pointers
// and no Call, so the dedicated engine builds it without strata-moe's op.
#include "moe_residency.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace strata::dsv41 {

const std::vector<uint8_t>& Residency::prepare(const Call& c) {
    Layer& l = layer(c.layer, c.up->name);
    if (!l.bound) {
        ExpertTensors t;
        t.layer = c.layer;
        t.n_expert = c.up->ne[2];
        const ggml_tensor* tensors[3] = {c.up, c.gate, c.down};
        for (int p = 0; p < 3; ++p) {
            const ggml_tensor* x = tensors[p];
            // the plain-pointer form describes contiguous experts: expert e at data + e * nb[2], nothing after them
            if (x->ne[2] != t.n_expert || ggml_nbytes(x) != x->nb[2] * (size_t) x->ne[2]) {
                GGML_ABORT("strata-moe: layer %d: %s is %s [%lld, %lld, %lld] of %zu bytes, not %lld contiguous experts "
                           "of %zu bytes", c.layer, x->name, ggml_type_name(x->type), (long long) x->ne[0],
                           (long long) x->ne[1], (long long) x->ne[2], ggml_nbytes(x), (long long) t.n_expert, x->nb[2]);
            }
            t.name[p] = x->name;
            t.data[p] = x->data;
            t.type[p] = x->type;
            t.expert_bytes[p] = x->nb[2];
        }
        bind_layer(t);
    }
    return l.read;
}

void Residency::bind(Call& c) {
    Layer& l = layer(c.layer, c.up->name);
    const int64_t n = (int64_t) c.experts.size(), n_cold = n - c.n_ready;
    c.base_up.resize((size_t) n);
    c.base_gate.resize((size_t) n);
    c.base_down.resize((size_t) n);
    const char** bases[3] = {c.base_up.data(), c.base_gate.data(), c.base_down.data()};
    const size_t expert_bytes = l.part_bytes[0] + l.part_bytes[1] + l.part_bytes[2];
    const ggml_tensor* tensors[3] = {c.up, c.gate, c.down};
    c.cold_mapped = cold_path_ == ColdPath::kPageCache;
    for (int64_t k = 0; k < c.n_ready; ++k) {
        const int32_t e = c.experts[(size_t) k];
        const int32_t slot = l.warm_slot[(size_t) e];
        const int64_t pairs = c.pair_begin[(size_t) k + 1] - c.pair_begin[(size_t) k];
        if (slot >= 0) {
            for (int p = 0; p < 3; ++p) bases[p][k] = l.warm_rows[(size_t) slot * 3 + (size_t) p];
            c.warm_pairs += pairs;
            ++c.warm_reads;
            c.warm_bytes += expert_bytes;
        } else if (l.cold[(size_t) e] && cold_path_ == ColdPath::kPageCache) {
            // pagecache: a cold expert is ready where the fork mapped it
            for (int p = 0; p < 3; ++p) {
                bases[p][k] = static_cast<const char*>(tensors[p]->data) + (size_t) e * tensors[p]->nb[2];
            }
            c.cold_pairs += pairs;
            ++c.cold_reads;
            c.cold_bytes += expert_bytes;
        } else {
            GGML_ABORT("strata-moe: layer %d: expert %d is a miss in neither the warm nor the cold tier (hot set %s "
                       "has it in VRAM, and the GPU tier did not compute it)", c.layer, e, hot_.path.c_str());
        }
    }

    c.round_end.clear();
    if (n_cold == 0) {
        c.round_end.push_back(n);
    } else {
        for (int64_t j = 0; j < n_cold; j += round_experts_) {
            c.round_end.push_back(c.n_ready + std::min(j + round_experts_, n_cold));
        }
    }
    if ((size_t) (3 * n_cold) > c.cold_capacity) {
        c.cold_ready = std::make_unique<std::atomic<uint8_t>[]>((size_t) (3 * n_cold));
        c.cold_capacity = (size_t) (3 * n_cold);
    }
    for (int64_t j = 0; j < 3 * n_cold; ++j) c.cold_ready[(size_t) j].store(0, std::memory_order_relaxed);
    for (int64_t j = 0; j < n_cold; ++j) {
        const int64_t k = c.n_ready + j;
        const int32_t e = c.experts[(size_t) k];
        const int64_t slot = (j / round_experts_ % 2) * round_experts_ + j % round_experts_;
        for (int p = 0; p < 3; ++p) {
            const DirectSpan s = direct_span(l.file_offset[p] + (uint64_t) e * l.part_bytes[p], l.part_bytes[p]);
            bases[p][k] = staging_ + (size_t) slot * staging_stride_ + l.part_offset[p] + s.lead;
            c.cold_bytes += s.len;
        }
        c.cold_pairs += c.pair_begin[(size_t) k + 1] - c.pair_begin[(size_t) k];
    }
    c.cold_reads += n_cold;
}

void Residency::queue_round(Call& c, int64_t r) {
    const Layer& l = layers_.at(c.layer);
    const int64_t n_cold = (int64_t) c.experts.size() - c.n_ready;
    std::vector<Job> jobs;
    for (int64_t j = r * round_experts_; j < std::min((r + 1) * round_experts_, n_cold); ++j) {
        const int32_t e = c.experts[(size_t) (c.n_ready + j)];
        const int64_t slot = (r % 2) * round_experts_ + j % round_experts_;
        for (int p = 0; p < 3; ++p) {
            Job job;
            job.fd = l.fd[p];
            job.path = l.path[p];
            job.len = l.part_bytes[p];
            job.span = direct_span(l.file_offset[p] + (uint64_t) e * l.part_bytes[p], job.len);
            job.dst = staging_ + (size_t) slot * staging_stride_ + l.part_offset[p];
            job.ready = &c.cold_ready[(size_t) (j * 3 + p)];
            job.left = &c.cold_left;
            job.end_ns = &c.cold_end_ns;
            jobs.push_back(job);
        }
    }
    push(jobs);
}

void Residency::issue(Call& c) {
    busy_.lock();
    c.cold_t0 = std::chrono::steady_clock::now();
    c.cold_left.store(3 * ((int64_t) c.experts.size() - c.n_ready), std::memory_order_relaxed);
    queue_round(c, 0);
    if (c.round_end.size() > 1) queue_round(c, 1);
}

void Residency::wait(Call& c, int64_t k, int part) {
    std::atomic<uint8_t>& ready = c.cold_ready[(size_t) ((k - c.n_ready) * 3 + part)];
    if (ready.load(std::memory_order_acquire)) return;
    const auto t0 = std::chrono::steady_clock::now();
    while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
    if (profiling()) {
        c.cold_wait_ns.v.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
            std::memory_order_relaxed);
    }
}

void Residency::round_done(Call& c, int64_t r) {
    if (r + 2 < (int64_t) c.round_end.size()) queue_round(c, r + 2);
}

void Residency::finish(Call& c) {
    while (c.cold_left.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    const int64_t t0 = std::chrono::duration_cast<std::chrono::nanoseconds>(c.cold_t0.time_since_epoch()).count();
    c.cold_ms = (double) (c.cold_end_ns.load(std::memory_order_relaxed) - t0) / 1e6;
    busy_.unlock();
}

}  // namespace strata::dsv41
