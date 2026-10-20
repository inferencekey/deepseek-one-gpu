// tools/dsv41/engine/dsv41-engine.cpp - the dedicated DeepSeek-V4.1-Flash engine end to end: loads a deepseek41
// GGUF (src/dsv41/engine/model.cpp), places the routed experts (experts.cpp), runs a prompt of token ids through every
// layer in chunks and then greedy decode (forward.cpp), and prints the generated token ids. Token ids in, token ids
// out: the tokenizer and the chat template live in serve/server.py.
//
//   dsv41-engine --model F.gguf (--tokens 1,2,3 | --tokens-file F) [--n-predict N] [--ctx N] [--chunk N]
//                [--hot-set F --vram-gb G [--ram-gb G]] [--readers reuse|dense] [--f16] [--dump-layers DIR]
//                [--threads N] [--counts F] [--times] [--force-selection DIR] [--ring-set N] [--ring-sets N]
//                [--stage-sets N] [--prefetch-experts N] [--pool-spin-us N] [--prefill-split M] [--split-max-t N]
//                [--no-graphs] [--forced-moe-input] [--prefill-mmq] [--hot-policy P] [--hot-swaps N] [--hot-spare N]
//                [--hot-half-life H] [--hot-margin MUL:ADD] [--hot-check] [--route-trace F] [--deterministic on|off]
//                [--hot-homes all|spare|N] [--dry-run]
//
// Greedy decode feeds every generated token back, as the fork's op-dump does: a prompt and N steps are 1 + N calls,
// the logits of call i give generated token i (ties to the lower id), the last call's logits are dumped only. With
// --dump-layers DIR every call writes l_last.0-L<layer>-u<call>.bin ([T][HC][E] f32, the fork's l_last) and
// result_output.0-L999-u<call>.bin ([n_vocab] f32 of the call's last token), with each layer's selections and their
// scores in the fork's names (forward.hpp), and the run writes engine-manifest.json (prompt, generated and fed ids,
// calls): tools/dsv41/engine/engine_parity.py compares them with a fork dump or another run. --force-selection DIR
// is the parity mode (selection.hpp): every call takes the router ids and index rows of the reference in DIR (a fork
// op dump or another run's --dump-layers), and the decode steps feed the reference's tokens, so every call has the
// reference's inputs; the generated ids are still the engine's own picks, and a --chunk below the reference's ubatch
// takes its selections a chunk at a time (the multi-chunk paths under the fork's parity). --dry-run (the
// dev VM, no CUDA call) loads the model with host stand-ins, checks every view and every expert's tier, and plans every
// call (forward.hpp forward_dry_run). --forced-moe-input, with --force-selection,
// feeds every layer's MoE the reference's input (its ffn_norm files, forward.hpp ForwardConfig::forced_moe_input), so
// the dumps' ffn_moe_out and weights measure the router and the experts alone (oracle_parity.py's isolated mode);
// --prefill-mmq runs the prefill matmuls on the fork's MMQ (int8 activations, the oracle's own GPU arithmetic)
// instead of the dequant path's f16 activations.
//
// --serve: resident, the protocol of Strata's `strata --serve` (src/program/generate.cpp, its --serve block) so
// serve/server.py's StrataEngine drives it unchanged. On stdout `INFO key=value ...` and `READY <ctx> stop` once
// loaded, then per stdin line `GEN <max_new> [key=value ...] <id,id,...>`: `RESUME <n>` when n prompt tokens are
// reused (session.hpp's prefix cache), `PP <read> <prompt tokens> <ms> <tok/s>` after each prompt chunk (read counts
// the reused ones), `T <id>` per generated token, then `DONE <generated> <prompt tokens> <prompt ms> <decode ms>
// <stop|length|cancel> 0 0 <reused>` (no draft tokens: the two zeros are Strata's drafts accepted and offered), or
// `ERR <reason>` for a request that cannot run (nothing of it ran). `STOP` ends the running request after its current
// chunk or token, `QUIT` (or the end of stdin) ends the process. A request without temperature, or at temperature=0,
// decodes greedily (the argmax, ties to the lower id), its top_k, top_p, min_p and seed accepted and unused.
// Sampling: temperature=F above 0 samples on the host (sampler.hpp: llama.cpp's
// chain top_k -> top_p -> min_p -> temperature -> draw over the logits the device returns), with top_k=N (0: off, any
// N up to the vocabulary), top_p=F in [0, 1], min_p=F in [0, 1] and seed=N (0 .. 2^64 - 1; a std::mt19937_64 per
// request, seeded with it, or without one from std::random_device), a bad value of any of them refused with ERR
// naming it. Active penalties (penalty_repeat other than 1, penalty_freq or penalty_present other than 0) are refused
// with ERR naming the OpenAI field too; cvec and penalty_last_n are accepted and unused. ckpt=0 keeps no prompt-end
// snapshot (Strata's one-shot calls), times=1 times the request's decode per stage (every op launched, no token graph:
// the split of a decode step), any other key is refused. Before each DONE the engine writes the request's
// line to stderr (it ends in "greedy" or in "sampled: temperature T, top_k K, top_p P, min_p M, seed S (given |
// from std::random_device)", the seed the reply can be drawn again with), the decode's routed experts (hits, CPU
// misses, the CPU's ms and GB/s, the doorbell waits) and, with times=1, the per-stage times (print_times).
// --eos-ids (default: the GGUF's tokenizer.ggml.eos_token_id) end a reply; `--eos-ids none` ends none, so a request generates exactly its
// max_new tokens (the decode benches of tools/dsv41/pod/e07_bench.py). --serve --dry-run (the dev VM, no CUDA call) speaks the same
// protocol with a host stand-in for the device: the same Session, positions, checkpoints and restore rule, and as the
// "greedy" token of every call the last token read plus one (mod n_vocab), so a test can predict every line (its
// logits are one-hot, 1 there and 0 elsewhere: a sampled request draws over them with the same sampler).
//
// Conversation checkpoints: --checkpoints N (default 4) conversations kept beside the live state,
// each the state at a prompt's end in a pinned host slot sized for --ctx positions (session.hpp: the longest prefix
// resumes, a superseded turn's slot is reused, else a free one, else the least recently used is evicted). At load the
// engine writes the cost per conversation ("conversation checkpoints: N slots of X MiB pinned host RAM each ...", the
// dry run too, from the same spans) and the INFO line has checkpoints=N; each request's "checkpoints:" line names the
// slot restored (its tokens, the MiB copied host to device or "in place on the device", the copies' ms), the slot
// saved (tokens, MiB device to host, ms), the superseded ones, an eviction, and how many slots are held.
//
// The per-request hot set (experts.hpp, hot_policy.hpp): --hot-policy static|prompt|adaptive (default adaptive)
// is a request's policy, --hot-swaps (2) the swaps started per layer per token, --hot-spare (2) the spare slab slots
// per layer, --hot-half-life (128) and --hot-margin (1.5:1) adaptive's decay and hysteresis, --hot-check waits for
// each token's swaps and checks the residency after every token (the tests), --route-trace F writes every call's
// routed ids (tools/dsv41/engine/hot_policy_sim.py replays them). The serve protocol's GEN keys hot=static|prompt|
// adaptive (the request's policy), hot_reset=1 (the slab back to the hot set's before the request, so one engine
// load measures every policy from the same start), trace=0 (the request left out of the route trace) and
// prefill_times=1 (the prompt timed per stage: its split line). Each request's decode line gives the
// policy, the hit rate overall and per layer, the swaps (count, MiB, copy ms, batches, deferred) and the reset. A
// request that reuses the prefix (RESUME, without hot_reset=1) keeps the planners' counts of the last one: a
// conversation's next turn plans from them plus its suffix's (hot_policy.hpp).
//
// The deterministic mode (tools/dsv41/engine/DETERMINISM.md), --deterministic on
// (the default) and the GEN key det=1|0 per request: a request starts from the hot set's slab with zero counts (as
// hot_reset=1, and a RESUME keeps no counts) and each decode boundary waits for the open batch of swaps instead of
// deferring it, so the experts a token computes on the GPU and on the CPU follow from the request's own prompt and
// tokens, never from the copy streams' timing or the requests before it. Each request's line names the mode
// ("deterministic" or "deterministic off", before "greedy" or the sampler), and its hot set line ends in the mode with
// the boundaries that waited and their ms. The mode's reset runs after a
// request's DONE (Forward::end_request, hot_policy.hpp hot_request_end_resets), in the idle time before the next one,
// whose start's reset then finds nothing to move ("reset before it 0 swaps"); the hot set line ends in that idle
// reset (its swaps, MiB, ms) and how long the request waited for it (from its GEN line's arrival to the reset's end).
//
// Host homes: --hot-homes all gives every expert of the hot set's slab a
// permanent pinned host home (experts.hpp), so a swap copies up only and drops its victim, and the deterministic
// reset restores in place layer by layer while the next request runs, its layer l waiting for layer l's copies only;
// --hot-homes spare (the default) is the two-way path and the reset in rounds before. tools/dsv41/pod/e07_gate.py
// chooses: all when the host's RAM holds the homes with its margin. The load lines name the path ("experts: hot set
// homes: ...") and the homes' RAM ("fixed host RAM beside the checkpoints: ..."); the hot set line's idle reset is
// then the layers' copies on the device (ms from the first to the last layer's event) and its wait the host's plus
// the request's layers' waits on the device, which the next line splits ("idle reset layer by layer (host homes):
// ..."); --serve --dry-run adds the dry reset of the placement (ExpertEngine::dry_restore). --hot-homes N
// homes the first N slab experts of the home order (whole layers in layer order, experts.hpp) when the RAM holds no
// more (the gate's choice); the load line says how many of the slab's experts are homed ("experts: hot set homes:
// partial host homes ..."), the layers short of homes reset in rounds after the layered part ("idle reset in rounds"),
// and every request's "copies down left" line counts its swaps' copies down and the reset's before it.
//
// Batched decode: --batch N (1 to 8) batch slots of --batch-ctx positions (default 32768) beside
// the live sequence, in Strata's batch protocol (src/program/generate.cpp, which serve/server.py's
// StrataEngine.generate_batched drives; batch.hpp): `BGEN <slot> <max_new> [key=value ...] <ids>` reads the request
// as a GEN of one token (its RESUME, PP, T and DONE lines), then copies the live sequence into the slot and prints
// `BADM <slot> 1`; from then on, between two stdin lines, every active slot decodes one token per step (forward.hpp's
// batched decode), each printed as `BT <slot> <id>`, until `BDONE <slot> <generated> <stop|length|cancel> <ms>
// rep=<1|0>`. A request that cannot go on in a slot (max_new 1, an end of reply or a stop at its first token, a prompt
// and max_new past --batch-ctx) runs to its end alone on the live sequence instead and ends with `BADM <slot> 0
// rep=<1|0>` (with alone= and why= when its positions were the reason, below). `BSTOP <slot>` ends that slot before
// the next step; `BYIELD` is not taken (a prompt read never gives way; it is interleaved with the steps instead).
// INFO adds batch_slots=N batch_ctx=C slot_cache=0. A prompt read while
// slots decode holds their steps (logged per request) and starts no hot set reset of its own; rep=1 marks a request
// in the deterministic mode, rep=0 one in the mode off (with --cpu-numerics gpu a token's bits do not
// depend on the expert placement, so a request is reproducible whatever its neighbours). A promoted request (a GEN
// stopped, then a BGEN of its prompt and tokens) has one token left, which the Session feeds by decode as alone
// (session.hpp: the request's line "its 1 token read was fed by decode"); reading it by prefill(T=1) made such
// chains differ from alone. With --cpu-numerics fork (the default) no request beside decoding slots says rep=1, BDONE's
// or BADM's (its bits follow their placement; one with no slot beside it keeps the mode's). A request alone still goes
// through GEN as before (the server sends GEN when nothing else runs).
//
// Long prompts beside decoding slots: --interleave-chunk N (default --chunk, the engine's
// prefill chunk; at most it; 0: read whole between two steps): a request whose prompt is read while batch slots
// decode reads it in chunks of N tokens (fixed, so its chunk boundaries depend on it alone) with one batch step after
// every chunk but the last, so no decoding slot waits for more than one chunk (and the step) at a time; a BSTOP that
// comes meanwhile ends its slot before the next of those steps. The default is the prefill chunk because a chunk on
// the real model copies nearly every warm expert it touches whatever its size (A100, chunk 16384: ~103 GiB a
// chunk), so a smaller one costs about as much and only multiplies the chunks. A prompt read with no slot decoding
// is read as before (--chunk). The request's line "read beside N decoding batch slots" gives its chunks, each
// chunk's read time (mean and longest), the steps between them and the longest hold. INFO adds interleave_chunk=N
// with --batch.
//
// A BGEN refused for its positions (its prompt and max_new past --batch-ctx) runs
// alone as before, and says so: its line "ran alone on the live sequence, not in batch slot S: <fits()'s reason>"
// with the decoding slots it held and for how long, and its `BADM <slot> 0 rep=<1|0> alone=<slots held> why=<the
// reason>` (the reason to the end of the line); a chunk read beside decoding slots that takes more than 4.5 s warns on
// stderr. Before it runs, its line "runs alone on the live sequence, not in batch slot S: <the
// reason>; it holds K decoding batch slots until its end" and the control line `ALONE <slot> <K> why=<the reason>`
// (before its RESUME and PP lines). A prompt read beside the slots changes no slot's bits, and in the deterministic
// mode it picks its prefill split from the hot set's slab as alone (Forward::prompt_deterministic).
// Once the slots are all idle, after the "batch:" line, one "batch
// steps, <part>:" line each for all the steps, those within 15 s of a prompt read beside the slots and the others:
// a step's wall time inside the step (the graph's launch, the doorbells served with their waits, the CPU jobs and
// their plan, the tail, the boundary and its swap waits, the device's graph) and between two steps (the picks, the BT
// lines and their flush, the admissions, the loop), and per step the routed pairs, their distinct experts, hits,
// misses, CPU jobs and swaps. A prompt read beside the slots merges its experts' counts into theirs
// (Forward::read_beside, hot_policy.hpp HotLayerPlan::merge_prompt) instead of setting a target of its own.
#include "strata/dsv41/engine/batch.hpp"
#include "strata/dsv41/engine/experts.hpp"
#include "strata/dsv41/engine/forward.hpp"
#include "strata/dsv41/engine/model.hpp"
#include "strata/dsv41/engine/sampler.hpp"
#include "strata/dsv41/engine/selection.hpp"
#include "strata/dsv41/engine/session.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace e = strata::dsv41::engine;

namespace {

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "dsv41-engine: %s\n", m.c_str());
    std::exit(2);
}

void usage() {
    std::printf(
        "usage: dsv41-engine --model F.gguf (--tokens 1,2,3 | --tokens-file F) [--n-predict N] [--ctx N] [--chunk N]\n"
        "                    [--hot-set F --vram-gb G [--ram-gb G]] [--readers reuse|dense] [--f16] [--dump-layers DIR]\n"
        "                    [--threads N] [--counts F] [--times] [--force-selection DIR] [--ring-set N] [--ring-sets N]\n"
        "                    [--stage-sets N] [--prefetch-experts N] [--pool-spin-us N] [--prefill-split M]\n"
        "                    [--split-max-t N] [--no-graphs] [--forced-moe-input] [--prefill-mmq] [--hot-policy P]\n"
        "                    [--hot-swaps N] [--hot-spare N] [--hot-half-life H] [--hot-margin MUL:ADD] [--hot-check]\n"
        "                    [--route-trace F] [--deterministic on|off] [--hot-homes all|spare|N] [--cpu-numerics gpu|fork]\n"
        "                    [--dry-run]\n"
        "       dsv41-engine --serve --model F.gguf [--ctx N] [--chunk N] [--eos-ids L] [--log-ids] [--checkpoints N]\n"
        "                    [--batch N [--batch-ctx N] [--interleave-chunk N]] [placement and decode options as above]\n"
        "                    [--dry-run]\n"
        "  --model F.gguf      a deepseek41 GGUF (any other architecture is refused)\n"
        "  --serve             resident: Strata's stdin protocol (GEN / T / PP / RESUME / DONE / STOP), the prefix cache\n"
        "  --eos-ids L         --serve: the ids that end a reply, comma separated (default: the GGUF's eos_token_id;\n"
        "                      none: no id ends a reply, every request generates its max_new tokens)\n"
        "  --log-ids           --serve: each request's prompt and generated ids on stderr (a batch slot's every\n"
        "                      token too, at its end: request N: slot_token_ids)\n"
        "  --checkpoints N     --serve: conversations kept, each a checkpoint of a prompt's end in pinned host RAM\n"
        "                      sized for --ctx positions (default 4; the longest prefix resumes, the least recent goes)\n"
        "  --batch N           --serve: N batch slots (1 to 8; default 0, one request at a time): requests sent with BGEN\n"
        "                      decode together, one token per slot per step (Strata's batch protocol)\n"
        "  --batch-ctx N       --serve --batch: the positions a batch slot holds (default 32768, at most --ctx); a longer\n"
        "                      request runs alone with the full context\n"
        "  --interleave-chunk N  --serve --batch: a prompt read while slots decode is read in chunks of N tokens with\n"
        "                      one batch step between two (default and at most --chunk; 0: read whole between steps)\n"
        "  --tokens L          the prompt's token ids, comma separated; --tokens-file F: whitespace or comma separated\n"
        "  --n-predict N       greedy decode steps (default 16)\n"
        "  --ctx N             positions of the sequence, the KV cache's size (default 4096)\n"
        "  --chunk N           the largest prefill chunk (default 4096, at most 65535)\n"
        "  --hot-set F         a strata-moe hot-set file: the experts kept in VRAM and the warm lists (default: none)\n"
        "  --vram-gb G         the VRAM budget of the hot set's slab\n"
        "  --ram-gb G          the budget of the warm tier (pinned host RAM; needs the hot set's warm lists)\n"
        "  --readers M         reuse (default, the tech report's Reuse Mode) or dense (every visible row, as the fork)\n"
        "  --f16               the attention projections on f16 weights (the cuBLAS test path)\n"
        "  --dump-layers DIR   every call's l_last per layer and logits, and engine-manifest.json\n"
        "  --threads N         CPU threads of the expert misses (default: the hardware's, at most 16)\n"
        "  --counts F          per-(layer, expert) activation counts (strata-moe-counts/1) for hot_set.py\n"
        "  --times             per-stage times of the prefill and of the decode steps\n"
        "  --force-selection DIR  the router ids, index rows and decode tokens of a reference dump (parity mode)\n"
        "  --ring-set N, --ring-sets N, --stage-sets N  prefill: experts per ring set (48), ring sets (3), pinned\n"
        "                      staging sets (2)\n"
        "  --prefetch-experts N  prefill: experts of the next layer copied during its attention (32; 0: none)\n"
        "  --pool-spin-us N    the CPU pool polls this long before it blocks (default 50)\n"
        "  --prefill-split M   prefill chunks of up to --split-max-t tokens: measured (default: the experts outside the\n"
        "                      slab with few pairs on the CPU, the others through the ring, by rates measured at start),\n"
        "                      cpu (all of them on the CPU) or off (all through the ring)\n"
        "  --split-max-t N     the largest chunk of the prefill split (default 2048; 0: none)\n"
        "  --no-graphs         decode launches every op (default: each token one CUDA graph)\n"
        "  --forced-moe-input  with --force-selection: every layer's MoE input is the reference's (its ffn_norm files)\n"
        "  --prefill-mmq       the prefill matmuls on the fork's MMQ (int8 activations) instead of f16 activations\n"
        "  --hot-policy P      the VRAM hot set of a request: static (the hot set's), prompt (its prompt's most used\n"
        "                      experts) or adaptive (prompt, then a decayed LFU over the decode; the default)\n"
        "  --hot-swaps N       swaps started per layer per token (default 2)\n"
        "  --hot-spare N       spare slab slots per layer for the swaps (default 2; 0: the slab never moves)\n"
        "  --hot-half-life H   adaptive: decode tokens over which a count halves (default 128)\n"
        "  --hot-margin M:A    adaptive: a swap needs count(in) > M count(out) + A (default 1.5:1; A < 0 swaps at ties)\n"
        "  --hot-check         wait for each token's swaps and check the residency rows and filled slots after it\n"
        "  --route-trace F     every call's routed experts per layer into F (dsv41-route-trace/1)\n"
        "  --deterministic M   on (default): every request from the hot set's slab with zero counts, each token's swaps\n"
        "                      published right after it, so identical requests give identical tokens; off: a\n"
        "                      request keeps the last one's slab, a batch still copying is published later\n"
        "  --hot-homes M       all: every slab expert keeps a pinned host home, a swap copies up only and the deterministic\n"
        "                      reset runs layer by layer beside the next request; spare (default): the two-way path;\n"
        "                      N: the first N slab experts of the home order (whole layers in layer order) homed, the\n"
        "                      other layers on the two-way path\n"
        "  --cpu-numerics M    the CPU experts' arithmetic: fork (default: the fork's CPU experts, the\n"
        "                      fast path; a batched request's bits follow its neighbours, so no slot ends rep=1) or gpu\n"
        "                      (the GPU hits' bit for bit, so a token's bits do not depend on the placement; slower)\n"
        "  --dry-run           no CUDA: the model's views, the expert tiers and every call's plan\n");
}

std::vector<int32_t> parse_ids(const std::string& text, const std::string& what) {
    std::vector<int32_t> ids;
    std::string s = text;
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream in(s);
    std::string tok;
    while (in >> tok) {
        char* end = nullptr;
        const long v = std::strtol(tok.c_str(), &end, 10);
        if (*end != '\0' || v < 0 || v > INT32_MAX) die(what + ": '" + tok + "' is not a token id");
        ids.push_back((int32_t) v);
    }
    if (ids.empty()) die(what + ": no token ids");
    return ids;
}

int32_t argmax(const std::vector<float>& v) {
    int32_t best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[(size_t) best]) best = (int32_t) i;
    return best;
}

std::string join(const std::vector<int32_t>& v, const char* sep) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + std::to_string(v[i]);
    return s;
}

// a JSON string literal: a path may hold quotes, backslashes or control characters
std::string json_string(const std::string& v) {
    std::string s = "\"";
    for (const char c : v) {
        if (c == '"' || c == '\\') {
            s += '\\';
            s += c;
        } else if ((unsigned char) c < 0x20) {
            char u[7];
            std::snprintf(u, sizeof u, "\\u%04x", (unsigned) (unsigned char) c);
            s += u;
        } else {
            s += c;
        }
    }
    return s + '"';
}

void print_times(std::FILE* out, const char* what, const e::ForwardTimes& t, const e::ExpertTimes& x) {
    std::fprintf(out, "times %s: %lld calls, %lld tokens, %.1f ms total: embed %.2f, engram %.2f, attention %.2f, hc and norms "
                "%.2f, router %.2f, shared expert %.2f, routed experts %.2f (host: routing fetch %.2f, doorbell wait %.2f "
                "for %lld rings, plan %.2f, CPU %.2f, queue %.2f; %lld slab pairs, %lld other pairs, %lld experts %.1f MiB "
                "through the ring, %lld prefetched of which %lld used), head %.2f ms; %lld host syncs, %lld graph "
                "launches\n",
                what, (long long) t.calls, (long long) t.tokens, t.total_ms, t.embed_ms, t.engram_ms, t.attention_ms,
                t.hc_ms, t.router_ms, t.shared_ms, t.experts_ms, x.fetch_ms, x.ring_wait_ms, (long long) x.doorbells,
                x.plan_ms, x.cpu_ms, x.queue_ms,
                (long long) x.hits, (long long) x.misses, (long long) x.ring_experts, x.copy_bytes / (1024.0 * 1024.0),
                (long long) x.prefetched, (long long) x.prefetch_used, t.head_ms, (long long) t.host_syncs,
                (long long) t.graph_launches);
    if (x.split_calls)
        std::fprintf(out, "times %s: prefill split in %lld calls: %lld experts (%lld pairs, %.1f MiB) on the CPU in %.2f ms (its "
                    "estimate %.2f ms with the rows' upload, the ring's %.2f ms); %lld experts outside the slab touched "
                    "(%.1f MiB)\n",
                    what, (long long) x.split_calls, (long long) x.split_cpu_experts, (long long) x.split_cpu_pairs,
                    x.split_cpu_bytes / (1024.0 * 1024.0), x.split_cpu_ms, x.split_cpu_est_ms, x.split_ring_est_ms,
                    (long long) x.touched_experts, x.touched_bytes / (1024.0 * 1024.0));
}

// --serve --dry-run's device: positions, and per checkpoint slot the position it holds, the logits one-hot at the
// last token read plus one; per batch slot its position, each row's logits one-hot at its token plus one
class DryBackend final : public e::SessionBackend, public e::BatchBackend {
public:
    DryBackend(int64_t ctx, int64_t chunk, int64_t n_vocab, int checkpoints, int batch, int64_t batch_ctx)
        : ctx_(ctx), chunk_(chunk), n_vocab_(n_vocab), slot_pos_((size_t) checkpoints, -1), batch_ctx_(batch_ctx),
          batch_pos_((size_t) batch, -1) {}
    int64_t ctx() const override { return ctx_; }
    int64_t chunk() const override { return chunk_; }
    int64_t n_vocab() const override { return n_vocab_; }
    void reset() override { pos_ = 0; }
    int checkpoints() const override { return (int) slot_pos_.size(); }
    void save_checkpoint(int slot) override { slot_pos_.at((size_t) slot) = pos_; }
    void restore_checkpoint(int slot) override {
        if (slot_pos_.at((size_t) slot) < 0) die("dry run: checkpoint " + std::to_string(slot) + " was never saved");
        pos_ = slot_pos_[(size_t) slot];
    }
    void prefill(const int32_t* t, int64_t n, std::vector<float>& logits) override {
        if (n < 1 || n > chunk_ || pos_ + n > ctx_) die("dry run: a prefill call of " + std::to_string(n) + " tokens at " + std::to_string(pos_));
        pos_ += n;
        one_hot(t[n - 1], logits);
    }
    void decode(int32_t t, std::vector<float>& logits) override {
        if (pos_ + 1 > ctx_) die("dry run: a decode step past the context");
        ++pos_;
        one_hot(t, logits);
    }
    int batch_slots() const override { return (int) batch_pos_.size(); }
    int64_t batch_ctx() const override { return batch_ctx_; }
    void batch_admit(int b) override {
        if (pos_ > batch_ctx_) die("dry run: an admission at " + std::to_string(pos_) + " past --batch-ctx");
        batch_pos_.at((size_t) b) = pos_;
    }
    void batch_decode(const int* slots, const int32_t* tokens, int B, std::vector<std::vector<float>>& logits) override {
        logits.resize((size_t) B);
        for (int i = 0; i < B; ++i) {
            int64_t& p = batch_pos_.at((size_t) slots[i]);
            if (p < 0 || p + 1 > batch_ctx_) die("dry run: a step of slot " + std::to_string(slots[i]) + " at " + std::to_string(p));
            ++p;
            one_hot(tokens[i], logits[(size_t) i]);
        }
    }

private:
    void one_hot(int32_t last, std::vector<float>& logits) const {
        logits.assign((size_t) n_vocab_, 0.f);
        logits[(size_t) ((last + 1) % n_vocab_)] = 1.f;
    }
    int64_t ctx_, chunk_, n_vocab_, pos_ = 0;
    std::vector<int64_t> slot_pos_;
    int64_t batch_ctx_;
    std::vector<int64_t> batch_pos_;
};

// A request's checkpoints on its own stderr line, before its DONE: the
// one restored (its tokens, the bytes copied host to device or in place, the copies' ms on the stream), the one its
// prompt's end was saved to (tokens, bytes, ms), the superseded ones dropped and the one evicted. fw: null in the dry
// run (no copies, no times).
std::string checkpoint_text(const e::Session& session, const e::SessionResult& res, const e::Forward* fw) {
    std::string out;
    char s[256];
    const auto& slots = session.slots();
    if (res.plan.kind == e::ResumeKind::PromptEnd) {
        const int k = res.plan.slot;
        if (fw) {
            const e::KvCache::CheckpointTimes t = fw->checkpoint_times(k);
            if (t.in_place)
                std::snprintf(s, sizeof s, "restored %d (%lld tokens, in place on the device, %.2f ms)", k,
                              (long long) res.plan.reuse, t.restore_ms);
            else
                std::snprintf(s, sizeof s, "restored %d (%lld tokens, %.1f MiB host to device, %.2f ms)", k,
                              (long long) res.plan.reuse, t.restore_bytes / 1048576.0, t.restore_ms);
        } else {
            std::snprintf(s, sizeof s, "restored %d (%lld tokens, dry run)", k, (long long) res.plan.reuse);
        }
        out += s;
    } else {
        out += res.plan.kind == e::ResumeKind::Live ? "none restored (live)" : "none restored (cold)";
    }
    if (res.saved >= 0) {
        const int64_t tokens = slots[(size_t) res.saved].position();
        if (fw) {
            const e::KvCache::CheckpointTimes t = fw->checkpoint_times(res.saved);
            std::snprintf(s, sizeof s, ", saved to %d (%lld tokens, %.1f MiB device to host, %.2f ms)", res.saved,
                          (long long) tokens, t.save_bytes / 1048576.0, t.save_ms);
        } else {
            std::snprintf(s, sizeof s, ", saved to %d (%lld tokens, dry run)", res.saved, (long long) tokens);
        }
        out += s;
    } else {
        out += ", none saved";
    }
    std::snprintf(s, sizeof s, ", %d superseded", res.superseded);
    out += s;
    if (res.evicted >= 0) {
        std::snprintf(s, sizeof s, ", evicted %d (%lld tokens, the least recently used)", res.evicted,
                      (long long) res.evicted_tokens);
        out += s;
    }
    int held = 0;
    for (const e::ConversationSlot& c : slots) held += c.held();
    std::snprintf(s, sizeof s, "; %d of %zu slots held, %lld evictions", held, slots.size(), (long long) session.evictions());
    return out + s;
}

// A GEN line's keys beside the request's (times=1, hot=, hot_reset=1, trace=0, prefill_times=1; the sampling
// keys; det=)
struct GenKeys {
    bool times = false, prefill_times = false, hot_reset = false, trace = true, has_hot = false;
    e::HotPolicy hot = e::HotPolicy::Adaptive;
    bool has_det = false, det = true;
    bool sampled = false;   // temperature above 0: sp is the request's sampler
    e::SamplerParams sp;
};

// The sampling keys' values, whole or refused (a GEN line's ERR names the key)
bool parse_real(const std::string& v, double lo, double hi, double& out) {
    char* end = nullptr;
    out = std::strtod(v.c_str(), &end);
    return !v.empty() && *end == '\0' && std::isfinite(out) && out >= lo && out <= hi;
}

bool parse_count(const std::string& v, uint64_t& out) {
    char* end = nullptr;
    errno = 0;
    out = std::strtoull(v.c_str(), &end, 10);
    return !v.empty() && v[0] >= '0' && v[0] <= '9' && *end == '\0' && errno == 0;
}

// the OpenAI field a penalty key carries (serve/server.py's StrataEngine.sampling_keys), for its ERR
const char* api_penalty_name(const std::string& key) {
    return key == "penalty_repeat" ? "repetition_penalty"
           : key == "penalty_freq" ? "frequency_penalty"
                                   : "presence_penalty";
}

// A GEN line's request: "" when it parsed, else the ERR reason
std::string parse_gen(const std::string& line, e::SessionRequest& r, GenKeys& k) {
    const char* p = line.c_str() + 4;
    char* end = nullptr;
    const long long max_new = std::strtoll(p, &end, 10);
    if (end == p) return "expected: GEN <max_new> [key=value ...] <id,id,...>";
    r.max_new = max_new;
    std::istringstream in(end);
    std::string tok, ids, temperature, top_k, top_p, min_p, seed;
    while (in >> tok) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            if (!ids.empty()) return "more than one id list";
            ids = tok;
            continue;
        }
        const std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
        const double v = std::strtod(val.c_str(), nullptr);
        if (key == "ckpt") r.snapshot = v != 0;
        else if (key == "times") k.times = v != 0;
        else if (key == "prefill_times") k.prefill_times = v != 0;
        else if (key == "hot_reset") k.hot_reset = v != 0;
        else if (key == "trace") k.trace = v != 0;
        else if (key == "det") {
            if (val != "0" && val != "1") return "det=" + val + ": 1 (the deterministic mode) or 0 (off)";
            k.has_det = true;
            k.det = val == "1";
        }
        else if (key == "hot") {
            if (!e::parse_hot_policy(val, k.hot)) return "hot=" + val + ": static, prompt or adaptive";
            k.has_hot = true;
        }
        else if (key == "temperature") temperature = val;
        else if (key == "top_k") top_k = val;
        else if (key == "top_p") top_p = val;
        else if (key == "min_p") min_p = val;
        else if (key == "seed") seed = val;
        else if (key == "penalty_repeat" || key == "penalty_freq" || key == "penalty_present") {
            if (v != (key == "penalty_repeat" ? 1 : 0))
                return key + "=" + val + " (the API's " + api_penalty_name(key) + "): this engine applies no penalties";
        }
        else if (key != "cvec" && key != "penalty_last_n")
            return "unknown key " + key;
    }
    // Temperature above 0 samples; at 0 (or absent) the request is greedy and the other keys stay unused, as
    // before the sampler (the greedy path takes no new check)
    double t = 0;
    if (!temperature.empty() && !parse_real(temperature, 0, HUGE_VAL, t))
        return "temperature=" + temperature + ": a number >= 0";
    if (t > 0) {
        k.sampled = true;
        k.sp.temperature = t;
        uint64_t n = 0;
        if (!top_k.empty()) {
            if (!parse_count(top_k, n) || n > (uint64_t) INT64_MAX)
                return "top_k=" + top_k + ": a whole number >= 0 (0: off)";
            k.sp.top_k = (int64_t) n;
        }
        if (!top_p.empty() && !parse_real(top_p, 0, 1, k.sp.top_p)) return "top_p=" + top_p + ": a number in [0, 1]";
        if (!min_p.empty() && !parse_real(min_p, 0, 1, k.sp.min_p)) return "min_p=" + min_p + ": a number in [0, 1]";
        if (!seed.empty()) {
            if (!parse_count(seed, k.sp.seed)) return "seed=" + seed + ": a whole number in [0, 2^64 - 1]";
            k.sp.has_seed = true;
        }
    }
    if (ids.empty()) return "no token ids";
    std::replace(ids.begin(), ids.end(), ',', ' ');
    std::istringstream id_in(ids);
    while (id_in >> tok) {
        char* e2 = nullptr;
        const long v = std::strtol(tok.c_str(), &e2, 10);
        if (*e2 != '\0' || v < 0 || v > INT32_MAX) return "'" + tok + "' is not a token id";
        r.ids.push_back((int32_t) v);
    }
    return "";
}

// A request's hot set: its policy, the swaps since the counters' reset (count, MiB, copy ms on the copy stream,
// batches, boundaries that found the open batch still copying, checks), and the hit rate per layer
std::string hot_text(const e::Forward& fw) {
    const e::ExpertEngine& x = fw.experts();
    const e::ExpertTimes& t = x.times();
    char s[320];
    std::snprintf(s, sizeof s, "hot set %s%s: %lld swaps (%.1f MiB copied in %.2f ms on the copy stream, %lld batches, "
                  "%lld deferred), %lld checks", e::hot_policy_name(x.request_policy()),
                  x.hot_active() ? "" : " (static: no spare slots)", (long long) t.swaps, t.swap_bytes / (1024.0 * 1024.0),
                  t.swap_copy_ms, (long long) t.swap_batches, (long long) t.swap_deferred, (long long) t.hot_checks);
    std::string out = s;
    out += "; hit rate per layer";
    const std::vector<int64_t>& p = x.decode_layer_pairs();
    const std::vector<int64_t>& h = x.decode_layer_hits();
    for (size_t l = 0; l < p.size(); ++l) {
        std::snprintf(s, sizeof s, "%s%.3f", l ? "," : " ", p[l] ? (double) h[l] / (double) p[l] : 0.0);
        out += s;
    }
    return out;
}

// The request's mode on its hot set line: deterministic, with the boundaries that waited for their batch of swaps
// (where the mode off defers it) and the host's ms in those waits, or off
std::string det_text(const e::Forward& fw) {
    const e::ExpertTimes& t = fw.experts().times();
    if (!fw.experts().request_deterministic()) return "deterministic off (det=0)";
    char s[160];
    std::snprintf(s, sizeof s, "deterministic: %lld boundaries waited for their swaps' copies, %.2f ms", (long long) t.swap_waits,
                  t.swap_wait_ms);
    return s;
}

// The decode of a request on the device (Forward's times since the prompt's end): the routed experts' hit/miss split
// and the CPU's share (the real model's decode against the 30 tok/s target),
// the hot set (the policy, the swaps, the reset before the request, the hit rate per layer), with times=1 the
// per-stage split too
void print_decode(const e::Forward& fw, long long n_req, int64_t steps, double decode_ms, bool timed,
                  const e::ExpertTimes& reset, const e::ExpertEngine::IdleReset& idle, double idle_wait_ms) {
    const e::ForwardTimes t = fw.times();
    const e::ExpertTimes& x = fw.experts().times();
    const double pairs = (double) (x.hits + x.misses), per = steps > 0 ? 1.0 / (double) steps : 0.0;
    std::fprintf(stderr, "dsv41-engine serve: request %lld: decode experts: %lld steps, %lld hits and %lld misses (hit "
                 "rate %.4f), %lld CPU jobs reading %.1f MiB in %.2f ms (%.3f ms per step, %.2f GB/s), doorbell waits "
                 "%.2f ms over %lld doorbells, plan %.2f ms; %.3f ms per step%s\n",
                 n_req, (long long) steps, (long long) x.hits, (long long) x.misses, pairs > 0 ? (double) x.hits / pairs : 0.0,
                 (long long) x.cpu_experts, x.cpu_bytes / (1024.0 * 1024.0), x.cpu_ms, x.cpu_ms * per,
                 x.cpu_ms > 0 ? x.cpu_bytes / (x.cpu_ms * 1e6) : 0.0, x.ring_wait_ms, (long long) x.doorbells, x.plan_ms,
                 decode_ms * per, timed ? " (timed: every op launched, no token graph)" : "");
    std::fprintf(stderr, "dsv41-engine serve: request %lld: decode %s; reset before it %lld swaps (%.1f MiB) in %.1f ms; "
                 "%s; idle reset after the last request %lld swaps (%.1f MiB) in %.1f ms, waited %.1f ms\n", n_req,
                 hot_text(fw).c_str(), (long long) reset.reset_swaps, reset.reset_bytes / (1024.0 * 1024.0), reset.reset_ms,
                 det_text(fw).c_str(), (long long) idle.swaps, idle.bytes / (1024.0 * 1024.0),
                 idle.layered ? std::max(idle.device_ms, idle.ms) : idle.ms, idle_wait_ms + idle.wait_ms);
    if (idle.layered)
        std::fprintf(stderr, "dsv41-engine serve: request %lld: idle reset layer by layer (host homes): %lld swaps copied up "
                     "only (%.1f MiB, none down), %.1f ms on the device%s, queued in %.1f ms; this request waited %.1f ms on "
                     "the host and %.1f ms on the device over %d layers (at most %.1f ms, layer %d)\n", n_req,
                     (long long) (idle.swaps - idle.round_swaps), (idle.bytes - idle.round_bytes) / (1024.0 * 1024.0),
                     idle.device_ms, idle.seen_done ? "" : " (not seen done yet)", idle.queue_ms, idle_wait_ms, idle.wait_ms,
                     idle.layers_waited, idle.wait_max_ms, idle.wait_max_layer);
    // Partial host homes: the layers without a home for every slab expert, reset in rounds after the layered part
    if (idle.layered && idle.round_layers)
        std::fprintf(stderr, "dsv41-engine serve: request %lld: idle reset in rounds (partial host homes, %d layers): %lld "
                     "swaps (%.1f MiB up, %.1f MiB down: %lld victims copied down) in %.1f ms on the host, before this "
                     "request was read\n", n_req, idle.round_layers, (long long) idle.round_swaps,
                     (idle.round_bytes - idle.round_down_bytes) / (1024.0 * 1024.0), idle.round_down_bytes / (1024.0 * 1024.0),
                     (long long) idle.round_downs, idle.round_ms);
    // and the copies down left with any host homes: this request's swaps' and the reset's before it
    const e::ExpertEngine& ex = fw.experts();
    if (ex.hot().host_homes && ex.hot_active())
        std::fprintf(stderr, "dsv41-engine serve: request %lld: copies down left (host homes held %lld of %lld slab experts): "
                     "%lld of this request's %lld swaps copied their victim down (%.1f MiB), %lld of the %lld swaps of the "
                     "idle reset before it (%.1f MiB)\n", n_req, (long long) ex.host_homes_held(), (long long) ex.vram_experts(),
                     (long long) x.swap_downs, (long long) x.swaps, x.swap_down_bytes / (1024.0 * 1024.0),
                     (long long) idle.round_downs, (long long) idle.swaps, idle.round_down_bytes / (1024.0 * 1024.0));
    if (!timed) return;
    char what[64];
    std::snprintf(what, sizeof what, "decode (request %lld)", n_req);
    print_times(stderr, what, t, x);
}

// A request's prompt timed per stage (prefill_times=1; Forward's timed stages, the experts' device
// intervals per layer summed): the front (attention; engram, hc and norms, router, shared expert), the routed experts
// (their compute by launch, the copies and how much of them the passes hid, the idle gaps), and the host's parts
void print_prefill(const e::Forward& fw, long long n_req) {
    const e::ForwardTimes t = fw.times();
    const e::ExpertTimes& x = fw.experts().times();
    e::ExpertLayerStats s;
    for (const e::ExpertLayerStats& l : fw.experts().layer_stats()) {
        s.passes += l.passes;
        s.copy_bytes += l.copy_bytes;
        s.copy_ms += l.copy_ms;
        s.compute_ms += l.compute_ms;
        s.overlap_ms += l.overlap_ms;
        s.plan_gap_ms += l.plan_gap_ms;
        s.stall_ms += l.stall_ms;
        s.gather_ms += l.gather_ms;
        s.gate_up_ms += l.gate_up_ms;
        s.down_ms += l.down_ms;
    }
    std::fprintf(stderr, "dsv41-engine serve: request %lld: prefill split: %lld tokens in %lld calls, %.1f ms (timed): front "
                 "attention %.1f, dense %.1f (engram %.1f, hc and norms %.1f, router %.1f, shared expert %.1f), routed experts "
                 "%.1f; their device time over the layers: compute %.1f (gather %.1f, gate/up %.1f, down %.1f) in %lld passes, "
                 "copies %.2f GB in %.1f ms of which %.1f hidden behind the passes or the front, plan gaps %.1f, stalls %.1f; "
                 "host: routing fetch %.1f, plan %.1f, queue %.1f (staging %.1f, its waits %.1f), split CPU %.1f; embed %.1f, "
                 "head %.1f ms\n",
                 n_req, (long long) t.tokens, (long long) t.calls, t.total_ms, t.attention_ms,
                 t.engram_ms + t.hc_ms + t.router_ms + t.shared_ms, t.engram_ms, t.hc_ms, t.router_ms, t.shared_ms,
                 t.experts_ms, s.compute_ms, s.gather_ms, s.gate_up_ms, s.down_ms, (long long) s.passes, s.copy_bytes / 1e9,
                 s.copy_ms, s.overlap_ms, s.plan_gap_ms, s.stall_ms, x.fetch_ms, x.plan_ms, x.queue_ms, x.stage_ms,
                 x.stage_wait_ms, x.split_cpu_ms, t.embed_ms, t.head_ms);
    char what[64];
    std::snprintf(what, sizeof what, "prefill (request %lld)", n_req);
    print_times(stderr, what, t, x);
}

// The end of a request's stderr line: "greedy", or the sampler's parameters and the seed it drew with
std::string sampling_text(const e::Sampler* s) {
    if (!s) return "greedy";
    const e::SamplerParams& p = s->params();
    char t[256];
    std::snprintf(t, sizeof t, "sampled: temperature %g, top_k %lld, top_p %g, min_p %g, seed %llu (%s)", p.temperature,
                  (long long) p.top_k, p.top_p, p.min_p, (unsigned long long) s->seed(),
                  p.has_seed ? "given" : "from std::random_device");
    return t;
}

// A batch slot's end on its own stderr line, before its BDONE: its tokens, its steps and how many of them it shared
// with other slots, the stall other requests' prompts put on its steps, and whether the deterministic mode's guarantee
// holds for it (whenever the request is in the mode with --cpu-numerics gpu; never with --cpu-numerics fork)
void print_batch_done(const e::BatchDone& d) {
    const char* rep = d.reproducible ? "reproducible (the deterministic mode: its bits do not depend on the expert "
                                       "placement)"
                      : !d.deterministic ? "not reproducible (deterministic off)"
                                         : "not reproducible (--cpu-numerics fork: a routed expert's bits follow its "
                                           "tier, so its neighbours' placement)";
    std::fprintf(stderr, "dsv41-engine serve: request %lld: batch slot %d: %lld generated (%s) in %.1f ms, %lld steps, %lld "
                 "of them beside other slots (%.2f rows a step), steps stalled %.1f ms by other requests' prompts; %s\n",
                 (long long) d.request, d.slot, (long long) d.generated, d.finish.c_str(), d.ms, (long long) d.steps,
                 (long long) d.shared_steps, d.steps ? (double) d.rows / (double) d.steps : 0.0, d.stall_ms, rep);
}

// Where the batch's steps spend their wall time, summed over the steps of one
// part of the batch (all of them, those within kAfterReadS of a prompt read beside the slots, the others): inside a
// step the backend's batch_decode (Forward: the graph's launch with the rows' inputs, the doorbells served, of them
// the waits for the device's ring, the CPU jobs and their plan, the tail to the logits, the boundary with its swap
// waits; the device's own graph time), between two steps the picks, the BT lines and their flush, the admissions'
// copies and the loop (its stdin poll, a request's own lines); per step the routed pairs, their distinct experts, the
// hits, the misses and their CPU jobs, the swaps published (decode_serve's counts over the rows' union)
constexpr double kAfterReadS = 15.0;   // steady-c4's window (chat_load.py steady): it opens at a read's end
// A chunk read beside decoding slots past this warns (the target: no decode gap past 5 s)
constexpr double kChunkWarnMs = 4500.0;
struct StepParts {
    int64_t steps = 0, rows = 0, captures = 0;
    double inside_ms = 0, pick_ms = 0, out_ms = 0, admit_ms = 0, loop_ms = 0;
    e::ForwardTimes f;
    e::ExpertTimes x;
    void add(const e::BatchSlots::Totals& t0, const e::BatchSlots::Totals& t1, const e::ForwardTimes& f0,
             const e::ForwardTimes& f1, const e::ExpertTimes& x0, const e::ExpertTimes& x1, int64_t captured) {
        steps += t1.steps - t0.steps;
        rows += t1.rows - t0.rows;
        inside_ms += t1.ms - t0.ms;
        pick_ms += t1.pick_ms - t0.pick_ms;
        out_ms += t1.out_ms - t0.out_ms;
        admit_ms += t1.admit_ms - t0.admit_ms;
        loop_ms += t1.loop_ms - t0.loop_ms;
        captures += captured;
        f.launch_ms += f1.launch_ms - f0.launch_ms;
        f.serve_ms += f1.serve_ms - f0.serve_ms;
        f.tail_ms += f1.tail_ms - f0.tail_ms;
        f.boundary_ms += f1.boundary_ms - f0.boundary_ms;
        f.graph_ms += f1.graph_ms - f0.graph_ms;
        x.ring_wait_ms += x1.ring_wait_ms - x0.ring_wait_ms;
        x.cpu_ms += x1.cpu_ms - x0.cpu_ms;
        x.plan_ms += x1.plan_ms - x0.plan_ms;
        x.hits += x1.hits - x0.hits;
        x.misses += x1.misses - x0.misses;
        x.distinct += x1.distinct - x0.distinct;
        x.cpu_experts += x1.cpu_experts - x0.cpu_experts;
        x.cpu_bytes += x1.cpu_bytes - x0.cpu_bytes;
        x.swaps += x1.swaps - x0.swaps;
        x.swap_bytes += x1.swap_bytes - x0.swap_bytes;
        x.swap_deferred += x1.swap_deferred - x0.swap_deferred;
        x.swap_waits += x1.swap_waits - x0.swap_waits;
        x.swap_wait_ms += x1.swap_wait_ms - x0.swap_wait_ms;
    }
};

void print_step_parts(const char* part, const StepParts& p) {
    if (!p.steps) return;
    const double n = (double) p.steps, mib = 1024.0 * 1024.0, pairs = (double) (p.x.hits + p.x.misses);
    const double between = p.pick_ms + p.out_ms + p.admit_ms + p.loop_ms;
    std::fprintf(stderr, "dsv41-engine serve: batch steps, %s: %lld steps of %.2f rows, "
                 "%.2f ms a step from one step's start to the next's (the prompts read between them apart): inside the "
                 "step %.2f (graph launch %.2f, doorbells served %.2f of which doorbell waits %.2f, CPU jobs %.2f and their "
                 "plan %.2f, tail %.2f, boundary %.2f of which swap waits %.2f at %lld steps; the device's graph %.2f; %lld "
                 "graphs captured again), between steps %.3f (the picks %.3f, the BT lines and their flush %.3f, admissions "
                 "%.3f, the loop and its stdin poll %.3f); per step %.1f routed pairs to %.1f distinct experts, %.1f hits "
                 "(hit rate %.4f) and %.1f misses in %.1f CPU jobs reading %.1f MiB (%.2f GB/s), %.2f swaps (%.1f MiB) "
                 "published, %.2f deferred\n",
                 part, (long long) p.steps, (double) p.rows / n, (p.inside_ms + between) / n, p.inside_ms / n,
                 p.f.launch_ms / n, p.f.serve_ms / n, p.x.ring_wait_ms / n, p.x.cpu_ms / n, p.x.plan_ms / n, p.f.tail_ms / n,
                 p.f.boundary_ms / n, p.x.swap_wait_ms / n, (long long) p.x.swap_waits, p.f.graph_ms / n,
                 (long long) p.captures, between / n, p.pick_ms / n, p.out_ms / n, p.admit_ms / n, p.loop_ms / n, pairs / n,
                 (double) p.x.distinct / n, (double) p.x.hits / n, pairs > 0 ? (double) p.x.hits / pairs : 0.0,
                 (double) p.x.misses / n, (double) p.x.cpu_experts / n, p.x.cpu_bytes / mib / n,
                 p.x.cpu_ms > 0 ? p.x.cpu_bytes / (p.x.cpu_ms * 1e6) : 0.0, (double) p.x.swaps / n,
                 p.x.swap_bytes / mib / n, (double) p.x.swap_deferred / n);
}

// --serve: Strata's stdin protocol over the Session (see the top of this file). A reader thread takes the lines, so
// a STOP reaches a running request; the replies go to stdout, flushed per line. fw: the device (null in the dry run).
// --batch N (batch: the slots' device, null without): BGEN reads a request as a GEN of one token, then continues
// it in its batch slot (BADM <slot> 1, then its BT lines in the steps and its BDONE), or, when it cannot (one token,
// an end of reply at once, a stop, a prompt and max_new past --batch-ctx), runs it to its end alone on the live
// sequence (BADM <slot> 0; one past --batch-ctx says ALONE <slot> <slots held> why=<reason> before it runs). Between
// two lines, while any slot decodes, the loop runs a step (batch.hpp).
// interleave: the chunk of a prompt read while slots decode (0: read whole between two steps).
// numerics: the CPU experts' (--cpu-numerics): a slot is reproducible only with the GPU's.
int serve(e::SessionBackend& backend, e::Forward* fw, const std::vector<int32_t>& eos, bool log_ids,
          const std::string& info, const e::HotConfig& hot, e::BatchBackend* batch, int64_t interleave,
          e::CpuNumerics numerics) {
    const bool tier_free = numerics == e::CpuNumerics::Gpu;
    e::Session session(backend, eos);
    std::unique_ptr<e::BatchSlots> slots;
    if (batch) slots = std::make_unique<e::BatchSlots>(*batch, eos);
    std::mutex mu;
    std::condition_variable cv;
    // each line with its arrival (how long a request waited for the last one's idle reset)
    using Clock = std::chrono::steady_clock;
    std::deque<std::pair<std::string, Clock::time_point>> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::thread([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.emplace_back(l, Clock::now());
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    }).detach();
    std::printf("INFO %s\n", info.c_str());
    std::printf("READY %lld stop\n", (long long) backend.ctx());
    std::fflush(stdout);
    // When the last request's idle reset (ExpertEngine::end_request) ended on the host; the reset itself is read
    // at the next request's report (a layered one's device numbers are known by then)
    Clock::time_point idle_end{};
    auto busy = [&] { return slots && slots->active() > 0; };
    // A prompt is being read in chunks between steps (no idle reset under it, though the slots may all end)
    bool reading = false;
    // The request ended and nothing decodes: the deterministic mode's reset after it
    auto idle = [&] {
        if (!fw || busy() || reading) return;
        fw->end_request();
        idle_end = Clock::now();
    };
    // With --log-ids, each slot's every token from its admission's first, written at its end (the request's
    // generated_token_ids line has its admission's token only)
    std::vector<std::vector<int32_t>> slot_ids(slots ? (size_t) slots->slots() : 0);
    // One step of the batch, its lines; once the slots are all idle, the batch's totals and the idle reset
    const e::BatchSlots::Events step_ev{
        [&](int b, int32_t t) {
            std::printf("BT %d %d\n", b, t);
            if (log_ids) slot_ids[(size_t) b].push_back(t);
        },
        [&](const e::BatchDone& done) {
            e::BatchDone d = done;
            d.reproducible = d.reproducible && tier_free;
            print_batch_done(d);
            if (log_ids)
                std::fprintf(stderr, "dsv41-engine serve: request %lld: slot_token_ids: %s\n", (long long) d.request,
                             join(slot_ids[(size_t) d.slot], " ").c_str());
            std::fflush(stderr);
            std::printf("BDONE %d %lld %s %.1f rep=%d\n", d.slot, (long long) d.generated, d.finish.c_str(), d.ms,
                        d.reproducible ? 1 : 0);
        },
        [] { std::fflush(stdout); }};
    // The steps' parts (StepParts), by whether a prompt read beside the slots ended within kAfterReadS
    StepParts parts_all, parts_after, parts_other;
    char after_name[80];
    std::snprintf(after_name, sizeof after_name, "within %.0f s of a prompt read beside the slots", kAfterReadS);
    Clock::time_point read_end{};
    bool has_read = false;
    auto step = [&] {
        const e::BatchSlots::Totals t0 = slots->totals();
        const e::ForwardTimes f0 = fw ? fw->times() : e::ForwardTimes{};
        const e::ExpertTimes x0 = fw ? fw->experts().times() : e::ExpertTimes{};
        const int64_t c0 = fw ? fw->batch_captures() : 0;
        const auto at = Clock::now();
        slots->step(step_ev);
        const e::ForwardTimes f1 = fw ? fw->times() : e::ForwardTimes{};
        const e::ExpertTimes x1 = fw ? fw->experts().times() : e::ExpertTimes{};
        const int64_t captured = fw ? fw->batch_captures() - c0 : 0;
        const bool after = has_read && std::chrono::duration<double>(at - read_end).count() < kAfterReadS;
        for (StepParts* p : {&parts_all, after ? &parts_after : &parts_other})
            p->add(t0, slots->totals(), f0, f1, x0, x1, captured);
        if (busy()) return;
        const e::BatchSlots::Totals& t = slots->totals();
        std::fprintf(stderr, "dsv41-engine serve: batch: %lld steps of %.2f rows on average (%lld rows) in %.1f ms, %.2f ms a "
                     "step, %.1f tok/s over the steps; %lld admissions, %lld prompts read between steps holding the batch "
                     "%.1f ms\n", (long long) t.steps, t.steps ? (double) t.rows / (double) t.steps : 0.0, (long long) t.rows,
                     t.ms, t.steps ? t.ms / (double) t.steps : 0.0, t.ms > 0 ? (double) t.rows / (t.ms / 1e3) : 0.0,
                     (long long) t.admissions, (long long) t.stalls, t.stall_ms);
        print_step_parts("all", parts_all);
        print_step_parts(after_name, parts_after);
        print_step_parts("the other steps", parts_other);
        std::fflush(stderr);
        parts_all = parts_after = parts_other = StepParts{};
        has_read = false;
        slots->reset_totals();
        idle();
    };
    // The BSTOP lines that came while a prompt is read between steps end their slots before the next step (a
    // client that left frees its slot during a long read); every other line keeps its place
    auto take_bstops = [&] {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->first.rfind("BSTOP ", 0) == 0) {
                slots->stop(std::atoi(it->first.c_str() + 6));
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    };
    for (int64_t n_req = 0;;) {
        std::string line;
        Clock::time_point arrived;
        {
            std::unique_lock<std::mutex> lk(mu);
            if (busy()) {
                if (lines.empty()) {   // nothing to read: the batch's next step
                    lk.unlock();
                    step();
                    continue;
                }
            } else {
                cv.wait(lk, [&] { return !lines.empty() || eof; });
                if (lines.empty()) return 0;
            }
            line = std::move(lines.front().first);
            arrived = lines.front().second;
            lines.pop_front();
        }
        // the part of the idle reset this line waited for: from its arrival to the reset's end, if it came first
        const double idle_wait_ms = std::max(0.0, std::chrono::duration<double, std::milli>(idle_end - arrived).count());
        if (line.empty()) continue;
        if (line == "QUIT") return 0;
        // BSTOP <slot> ends that slot before the next step; BYIELD (a shorter request waits) is not taken: a prompt
        // read here never gives way (it is interleaved with the steps in chunks instead)
        if (line.rfind("BSTOP ", 0) == 0) {
            if (slots) slots->stop(std::atoi(line.c_str() + 6));
            continue;
        }
        if (line.rfind("BYIELD", 0) == 0) continue;
        int admit_slot = -1;
        int64_t admit_max_new = 0;
        if (line.rfind("BGEN ", 0) == 0) {
            char* e1 = nullptr;
            const long b = std::strtol(line.c_str() + 5, &e1, 10);
            if (!slots || e1 == line.c_str() + 5 || b < 0 || b >= slots->slots() || slots->busy((int) b)) {
                std::printf("ERR BGEN: no such free slot (--batch %d)\n", slots ? slots->slots() : 0);
                std::fflush(stdout);
                continue;
            }
            admit_slot = (int) b;
            line = "GEN" + std::string(e1);
        }
        e::SessionRequest req;
        GenKeys keys;
        std::string why = line.rfind("GEN ", 0) == 0 ? parse_gen(line, req, keys)
                                                     : "expected: GEN <max_new> [key=value ...] <id,id,...>";
        const bool timed = keys.times;
        if (why.empty()) why = session.check(req);
        // A request beside decoding slots joins their hot set as it is: keys that need the engine alone are refused
        const bool others = busy();
        if (why.empty() && others && (keys.times || keys.prefill_times || keys.hot_reset || keys.has_hot))
            why = "times=1, prefill_times=1, hot_reset=1 and hot= need the engine alone; batch slots are decoding";
        if (!why.empty()) {
            std::printf("ERR %s\n", why.c_str());
            std::fflush(stdout);
            continue;
        }
        // A BGEN that can go on in its slot reads its prompt and its first token only (Strata's "GEN 1").
        // One refused for its positions (a prompt or a max_new past a slot) runs alone, holding the slots
        // that decode: said on its own stderr line and on its BADM (alone=<slots held> why=<fits()'s reason>)
        const std::string unfit = admit_slot >= 0 ? slots->fits((int64_t) req.ids.size(), req.max_new) : "";
        const bool fits = admit_slot >= 0 && unfit.empty();
        const bool alone_past = admit_slot >= 0 && !fits && req.max_new >= 2;
        if (fits) {
            admit_max_new = req.max_new;
            req.max_new = 1;
        }
        stop.store(false);   // a STOP sent after the last request's DONE is not this one's
        // A sampled request's own generator, seeded once for the request (a batch slot goes on drawing from it)
        std::unique_ptr<e::Sampler> sampler;
        if (keys.sampled) {
            sampler = std::make_unique<e::Sampler>(keys.sp);
            req.sampler = sampler.get();
        }
        e::SessionEvents ev;
        ev.resume = [](const e::ResumePlan& p) {
            if (p.reuse > 0) std::printf("RESUME %lld\n", (long long) p.reuse);
            std::fflush(stdout);
        };
        const e::ResumePlan plan0 = session.plan(req.ids);   // run() takes the same plan: nothing changed since
        const int64_t read0 = plan0.reuse;
        ev.chunk = [&](int64_t read, int64_t total, double ms) {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) read, (long long) total, ms,
                        ms > 0 ? (double) (read - read0) / (ms / 1e3) : 0.0);
            std::fflush(stdout);
        };
        ev.token = [](int32_t t) {
            std::printf("T %d\n", t);
            std::fflush(stdout);
        };
        ev.stop = [&] { return stop.load(); };
        // The request's hot set (its policy, the reset to the hot set's slab), its trace, its prompt timed per
        // stage with prefill_times=1. A request beside decoding slots starts nothing of its own: the hot set goes on
        // with the batch's; with --cpu-numerics gpu that changes none of its bits
        e::ExpertTimes reset;
        const bool det = keys.has_det ? keys.det : hot.deterministic;
        const auto t_req = Clock::now();
        // Beside decoding slots the prompt is read in chunks of
        // `interleave` with one step after each but the last; each hold (a chunk read, from the last step's end to the
        // next step) counts against the slots' steps. Each chunk's read time (the session measures it) runs
        // from the last chunk's end or the last step's, the first from the request's start
        const int beside = others ? slots->active() : 0;
        const bool interleaved = others && interleave > 0;
        int64_t chunks = 0, steps_between = 0;
        double hold_max = 0, chunk_ms_sum = 0, chunk_ms_max = 0;
        auto t_mark = t_req, t_chunk = t_req;
        if (interleaved) {
            req.chunk = std::min<int64_t>(interleave, backend.chunk());
            reading = true;
            if (fw) fw->hold_prompt(true);
            ev.between = [&] {
                take_bstops();
                const double held = std::chrono::duration<double, std::milli>(Clock::now() - t_mark).count();
                slots->stalled(held);
                hold_max = std::max(hold_max, held);
                ++steps_between;
                if (busy()) step();
                t_mark = t_chunk = Clock::now();
            };
        }
        if (others) {
            const auto chunk_ev = ev.chunk;
            const long long req_no = (long long) n_req + 1;
            const int64_t chunk_tokens = interleaved ? req.chunk : backend.chunk();
            ev.chunk = [&chunks, &chunk_ms_sum, &chunk_ms_max, &t_chunk, &read_end, &has_read, chunk_ev, req_no,
                        chunk_tokens, beside](int64_t read, int64_t total, double ms) {
                const auto now = Clock::now();
                const double took = std::chrono::duration<double, std::milli>(now - t_chunk).count();
                t_chunk = now;
                chunk_ms_sum += took;
                chunk_ms_max = std::max(chunk_ms_max, took);
                ++chunks;
                read_end = now;
                has_read = true;
                // A chunk read beside the slots near the 5 s decode-gap target; the time is the host's (two RTX PRO
                // 6000 Server machines each read a 16K chunk beside four slots in 2.59 s on average, 3.29 at most; an
                // RTX PRO 6000 Workstation card about 4.6 s, another session 7.68 s)
                if (took > kChunkWarnMs)
                    std::fprintf(stderr, "dsv41-engine serve: request %lld: warning: a chunk read beside %d decoding batch "
                                 "slots took %.1f ms (up to %lld tokens), past %.1f s: their users wait that long for a token "
                                 "(the target gap is 5 s; host dependent: a smaller --interleave-chunk does not make a "
                                 "chunk much cheaper on the real model)\n",
                                 req_no, beside, took, (long long) chunk_tokens, kChunkWarnMs / 1e3);
                chunk_ev(read, total, ms);
            };
        }
        // Beside the slots the prompt's counts join theirs, no target of its own (with --cpu-numerics gpu the slots'
        // bits do not follow the placement); the prompt's prefill split in its own mode
        // (alone, begin_request below sets it)
        if (fw) fw->read_beside(others);
        if (fw && others) fw->prompt_deterministic(det);
        // A request that runs alone past a slot says so before it runs, on stderr and on its
        // control line (ALONE <slot> <decoding slots held> why=<fits()'s reason>), so the server knows at once
        if (alone_past) {
            std::fprintf(stderr, "dsv41-engine serve: request %lld: runs alone on the live sequence, not in batch slot %d: "
                         "%s; it holds %d decoding batch slots until its end\n", (long long) n_req + 1,
                         admit_slot, unfit.c_str(), beside);
            std::fflush(stderr);
            std::printf("ALONE %d %d why=%s\n", admit_slot, beside, unfit.c_str());
            std::fflush(stdout);
        }
        if (fw && !others) {
            fw->set_timing(keys.prefill_times);
            fw->reset_times();
            // a request that reuses the prefix the last request left (a live resume, or the checkpoint it saved: the
            // other checkpoints are other conversations) is the conversation's next turn: its planners keep their
            // counts, unless the request is in the deterministic mode (the hot set's slab and zero counts,
            // hot_policy.hpp)
            fw->begin_request(keys.has_hot ? keys.hot : hot.policy, keys.hot_reset,
                              read0 > 0 && session.continues_last(plan0), det);
            reset = fw->experts().times();
            fw->set_trace(keys.trace);
        } else if (fw) {
            fw->reset_times();
            reset = fw->experts().times();
        }
        // the decode's counters from the prompt's end on (and its per-stage times with times=1); false when the
        // request ended before its prompt was read in full
        bool decoding = false;
        ev.prompt_logits = [&](const std::vector<float>&) {
            if (!fw) return;
            if (keys.prefill_times) print_prefill(*fw, (long long) n_req + 1);
            decoding = true;
            fw->set_timing(timed);
            fw->reset_times();
        };
        const e::SessionResult res = session.run(req, ev);
        reading = false;
        if (fw && interleaved) fw->hold_prompt(false);
        if (fw) fw->read_beside(false);
        if (others) {
            read_end = Clock::now();
            has_read = true;
        }
        ++n_req;
        // the request's stderr lines before its DONE, so a client that reads the log at DONE finds them
        std::fprintf(stderr, "dsv41-engine serve: request %lld: %zu prompt tokens, %s resume of %lld, %lld read in %.1f "
                     "ms, %lld generated (%s) in %.1f ms, first token at %.1f ms; %zu tokens held, snapshot at %lld; %s; %s\n",
                     (long long) n_req, req.ids.size(), e::resume_kind_name(res.plan.kind), (long long) res.plan.reuse,
                     (long long) res.read, res.prompt_ms, (long long) res.generated, res.finish.c_str(), res.decode_ms,
                     res.first_token_ms, session.live().size(), (long long) session.snapshot_tokens(),
                     det ? "deterministic" : "deterministic off", sampling_text(sampler.get()).c_str());
        // A promoted request's one token left, fed by decode as alone (session.hpp)
        if (res.fed_unfed)
            std::fprintf(stderr, "dsv41-engine serve: request %lld: its 1 token read was fed by decode: the token the last "
                         "request emitted and did not feed, as the request alone feeds it\n",
                         (long long) n_req);
        std::fprintf(stderr, "dsv41-engine serve: request %lld: checkpoints: %s\n", (long long) n_req,
                     checkpoint_text(session, res, fw).c_str());
        if (others) {
            // The prompt was read between two steps, the slots waiting for it, in
            // chunks with a step after each but the last, the last hold its last chunk and its first token (and its
            // whole decode when it runs alone past a slot's positions)
            const double held = std::chrono::duration<double, std::milli>(Clock::now() - t_mark).count();
            slots->stalled(held);
            hold_max = std::max(hold_max, held);
            const double all = std::chrono::duration<double, std::milli>(Clock::now() - t_req).count();
            std::fprintf(stderr, "dsv41-engine serve: request %lld: read beside %d decoding batch slots, %lld tokens in %lld "
                         "chunks of %s with %lld batch steps between them, a chunk "
                         "read in %.1f ms on average and %.1f at most, %.1f ms from its start to its %s, the slots held at "
                         "most %.1f ms at once; %s\n", (long long) n_req, beside, (long long) res.read,
                         (long long) chunks,
                         interleaved ? std::to_string(req.chunk).c_str() : "the engine's --chunk (not interleaved)",
                         (long long) steps_between, chunks ? chunk_ms_sum / chunks : 0.0, chunk_ms_max, all,
                         res.generated > 1 ? "last token" : "first token", hold_max,
                         !det ? "not reproducible (deterministic off)"
                         : tier_free ? "reproducible: its expert placement followed theirs, which no longer moves its "
                                       "bits"
                                     : "not reproducible (--cpu-numerics fork: its bits follow the placement)");
        }
        if (alone_past) {
            const double held = beside ? std::chrono::duration<double, std::milli>(Clock::now() - t_req).count() : 0.0;
            std::fprintf(stderr, "dsv41-engine serve: request %lld: ran alone on the live sequence, not in batch slot %d: %s "
                         "(C04 audit 2); %lld generated (%s), holding %d decoding batch slots for %.1f ms%s\n",
                         (long long) n_req, admit_slot, unfit.c_str(), (long long) res.generated, res.finish.c_str(),
                         beside, held,
                         det && !tier_free && beside ? "; not reproducible (--cpu-numerics fork: its bits follow the "
                                                       "placement)" : "");
        }
        if (decoding) {
            // beside decoding slots no idle reset ran before it
            print_decode(*fw, (long long) n_req, std::max<int64_t>(res.generated - 1, 0), res.decode_ms, timed, reset,
                         others ? e::ExpertEngine::IdleReset{} : fw->experts().idle_reset(), others ? 0.0 : idle_wait_ms);
            fw->set_timing(false);
        } else if (fw) {
            fw->set_timing(false);
        }
        if (log_ids)
            std::fprintf(stderr, "dsv41-engine serve: request %lld: prompt_token_ids: %s\ndsv41-engine serve: request %lld: "
                         "generated_token_ids: %s\n", (long long) n_req, join(req.ids, " ").c_str(), (long long) n_req,
                         join(res.tokens, " ").c_str());
        std::fflush(stderr);
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %lld\n", (long long) res.generated, req.ids.size(), res.prompt_ms,
                    res.decode_ms, res.finish.c_str(), (long long) res.plan.reuse);
        // BADM <slot> 1 (it goes on in its slot) or 0 (done), rep= as BDONE's for a request that ended here
        if (admit_slot >= 0) {
            const bool cont = fits && res.generated == 1 && res.finish == "length" &&
                              (int64_t) session.live().size() == (int64_t) req.ids.size();
            if (cont) {
                e::BatchSlots::Admission a;
                a.request = n_req;
                a.position = (int64_t) req.ids.size();
                a.first = res.tokens.front();
                a.max_new = admit_max_new;
                a.sampler = std::move(sampler);
                a.deterministic = det;
                a.t0 = t_req;
                slots->admit(admit_slot, std::move(a));
                if (log_ids) slot_ids[(size_t) admit_slot] = {res.tokens.front()};
                std::printf("BADM %d 1\n", admit_slot);
            } else {
                // With --cpu-numerics fork a request read or run beside decoding slots took its
                // bits from their placement, so rep=0 as its "read beside" line says; alone, det's as before
                const bool rep = det && (tier_free || beside == 0);
                if (alone_past)
                    std::printf("BADM %d 0 rep=%d alone=%d why=%s\n", admit_slot, rep ? 1 : 0, beside, unfit.c_str());
                else
                    std::printf("BADM %d 0 rep=%d\n", admit_slot, rep ? 1 : 0);
            }
        }
        std::fflush(stdout);
        // The deterministic mode's reset, after the reply (the next request finds the slab at the hot set's), only
        // once no slot decodes
        idle();
    }
}

// --serve: the model and the device (or the dry run's stand-ins), then the protocol
int serve_main(const std::string& model_path, const std::string& eos_text, bool log_ids, bool dry, bool f16,
               e::ForwardConfig fc, const e::ExpertConfig& ec, int64_t interleave) {
    const auto t0 = std::chrono::steady_clock::now();
    fc.chunk = std::min(fc.chunk, fc.ctx);
    // The default chunk of an interleaved read is the engine's prefill chunk; one given is at most it
    if (interleave < 0) interleave = fc.chunk;
    if (interleave > fc.chunk)
        die("--interleave-chunk " + std::to_string(interleave) + " is past the prefill chunk of " + std::to_string(fc.chunk) +
            " (--chunk, at most --ctx)");
    const e::HostModel host = e::load_gguf_model(model_path);
    e::Model model;
    std::string err;
    if (!model.init(host, dry, f16, err)) die(err);
    const e::DenseDims& dd = model.dense_dims();
    std::vector<int32_t> eos;
    // "none": no end of reply, the benches' fixed-length decodes
    if (eos_text.empty()) {
        if (host.eos_id >= 0) eos.push_back(host.eos_id);
    } else if (eos_text != "none") {
        eos = parse_ids(eos_text, "--eos-ids");
    }
    for (int32_t t : eos)
        if (t >= dd.n_vocab) die("--eos-ids: " + std::to_string(t) + " is outside the vocabulary of " + std::to_string(dd.n_vocab));
    std::fprintf(stderr, "dsv41-engine serve: %s: %d layers, %d experts (%d used), vocab %d, ctx %lld, chunk %lld, readers "
                 "%s, end of reply %s%s\n", model_path.c_str(), model.n_layer(), dd.n_expert, dd.n_expert_used, dd.n_vocab,
                 (long long) fc.ctx, (long long) fc.chunk, fc.readers == e::ReaderMode::Dense ? "dense" : "reuse",
                 eos.empty() ? "none" : join(eos, ",").c_str(), dry ? "; dry run: no CUDA call, stand-in tokens" : "");
    // batch_slots=N (StrataEngine.batch: the server's parallel requests), slot_cache=0 (a slot's sessions do not go
    // back to the solo path)
    const std::string info = "context=" + std::to_string(fc.ctx) + " chunk=" + std::to_string(fc.chunk) + " vocab=" +
                             std::to_string(dd.n_vocab) + " eos=" + (eos.empty() ? "none" : join(eos, ",")) + " dry=" +
                             (dry ? "1" : "0") + " checkpoints=" + std::to_string(fc.checkpoints) +
                             (fc.batch ? " batch_slots=" + std::to_string(fc.batch) + " batch_ctx=" +
                                             std::to_string(fc.batch_ctx) + " slot_cache=0 interleave_chunk=" +
                                             std::to_string(interleave)
                                       : std::string()) +
                             " engine=dsv41-engine";
    // The fork's numerics (the default) said once at the load
    if (ec.cpu_numerics == e::CpuNumerics::Fork)
        std::fprintf(stderr, "dsv41-engine serve: the CPU experts in the fork's numerics (--cpu-numerics fork, the default)"
                     ": a routed expert's bits follow its tier, so no batch slot ends reproducible\n");
    if (dry) {
        e::ExpertEngine experts;
        if (!experts.init(model, ec, fc.chunk, nullptr, nullptr, true, err)) die(err);
        // The checkpoints' cost as the device's KvCache would pin it (kv_checkpoint.hpp), so the gate's count
        // can be checked against the engine's on the dev VM
        const e::KvCheckpointGeometry g{model.attn_dims(), model.roles(), model.attn_dims().n_swa - 1 + fc.chunk};
        const size_t slot = e::kv_checkpoint_bytes(g, fc.ctx);
        std::fprintf(stderr, "conversation checkpoints: %d slots of %.1f MiB pinned host RAM each (a checkpoint of %lld "
                     "positions), %.1f MiB in all (%zu bytes each); dry run: nothing pinned\n", fc.checkpoints,
                     slot / 1048576.0, (long long) fc.ctx, (double) fc.checkpoints * slot / 1048576.0, slot);
        e::print_host_homes_cost(stderr, experts);
        e::print_batch_cost(stderr, model, fc);
        experts.print_summary(stderr);
        experts.dry_restore(stderr);
        DryBackend b(fc.ctx, fc.chunk, dd.n_vocab, fc.checkpoints, fc.batch, fc.batch_ctx);
        return serve(b, nullptr, eos, log_ids, info, ec.hot, fc.batch ? &b : nullptr, interleave, ec.cpu_numerics);
    }
    e::Forward fw;
    if (!fw.init(model, ec, fc, err)) die(err);
    fw.print_memory(stderr);
    std::fprintf(stderr, "dsv41-engine serve: loaded in %.0f ms\n",
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    e::ForwardBackend b(fw, fc, dd.n_vocab);
    return serve(b, &fw, eos, log_ids, info, ec.hot, fc.batch ? &b : nullptr, interleave, ec.cpu_numerics);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, tokens_text, tokens_file, dump_dir, force_dir;
    int64_t n_predict = 16;
    bool dry = false, f16 = false, serve_mode = false, log_ids = false;
    std::string eos_text;
    int checkpoints = -1;   // --serve's conversation checkpoints (-1: the default, 4)
    int64_t interleave = -1;   // --serve --batch's --interleave-chunk (-1: the default, --chunk)
    e::ForwardConfig fc;
    e::ExpertConfig ec;
    // The fork's CPU numerics by default, the fast path (single user 63.34 tok/s against 47.04 in the GPU's,
    // measured on an RTX PRO 6000); --cpu-numerics gpu keeps the placement-independent bits on request
    ec.cpu_numerics = e::CpuNumerics::Fork;
    auto value = [&](int& i) -> std::string {
        if (i + 1 >= argc) die(std::string(argv[i]) + " needs a value");
        return argv[++i];
    };
    auto number = [&](int& i) -> double {
        const std::string v = value(i);
        char* end = nullptr;
        const double x = std::strtod(v.c_str(), &end);
        if (*end != '\0' || !(x >= 0)) die(std::string(argv[i - 1]) + " " + v + ": expected a number >= 0");
        return x;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--model") model_path = value(i);
        else if (a == "--tokens") tokens_text = value(i);
        else if (a == "--tokens-file") tokens_file = value(i);
        else if (a == "--n-predict") n_predict = (int64_t) number(i);
        else if (a == "--ctx") fc.ctx = (int64_t) number(i);
        else if (a == "--chunk") fc.chunk = (int64_t) number(i);
        else if (a == "--hot-set") ec.hot_set = value(i);
        else if (a == "--vram-gb") ec.vram_gib = number(i);
        else if (a == "--ram-gb") ec.ram_gib = number(i);
        else if (a == "--readers") {
            const std::string m = value(i);
            if (m == "reuse") fc.readers = e::ReaderMode::Reuse;
            else if (m == "dense") fc.readers = e::ReaderMode::Dense;
            else die("--readers " + m + ": expected reuse or dense");
        }
        else if (a == "--f16") f16 = true;
        else if (a == "--dump-layers") dump_dir = value(i);
        else if (a == "--threads") ec.threads = (int) number(i);
        else if (a == "--counts") ec.counts = value(i);
        else if (a == "--times") fc.timing = true;
        else if (a == "--force-selection") force_dir = value(i);
        else if (a == "--ring-set") ec.ring_set = (int) number(i);
        else if (a == "--ring-sets") ec.ring_sets = (int) number(i);
        else if (a == "--stage-sets") ec.stage_sets = (int) number(i);
        else if (a == "--prefetch-experts") ec.prefetch_experts = (int) number(i);
        else if (a == "--pool-spin-us") ec.pool_spin_us = (int) number(i);
        else if (a == "--prefill-split") {
            const std::string m = value(i);
            if (m == "measured") ec.prefill_split = e::PrefillSplit::Measured;
            else if (m == "cpu") ec.prefill_split = e::PrefillSplit::Cpu;
            else if (m == "off") ec.prefill_split = e::PrefillSplit::Off;
            else die("--prefill-split " + m + ": expected measured, cpu or off");
        }
        else if (a == "--split-max-t") ec.split_max_t = (int64_t) number(i);
        else if (a == "--no-graphs") fc.graphs = false;
        else if (a == "--forced-moe-input") fc.forced_moe_input = true;
        else if (a == "--prefill-mmq") fc.prefill = e::QPrefill::Mmq;
        else if (a == "--hot-policy") {
            const std::string m = value(i);
            if (!e::parse_hot_policy(m, ec.hot.policy)) die("--hot-policy " + m + ": expected static, prompt or adaptive");
        }
        else if (a == "--hot-swaps") ec.hot.max_swaps = (int) number(i);
        else if (a == "--hot-spare") ec.hot.spare = (int) number(i);
        else if (a == "--hot-half-life") ec.hot.half_life = number(i);
        else if (a == "--hot-margin") {
            // MUL:ADD, ADD may be negative (the tests' swaps at equal counts)
            const std::string m = value(i);
            const size_t k = m.find(':');
            char *e1 = nullptr, *e2 = nullptr;
            const std::string a1 = k == std::string::npos ? "" : m.substr(0, k), a2 = k == std::string::npos ? "" : m.substr(k + 1);
            ec.hot.margin_mul = std::strtod(a1.c_str(), &e1);
            ec.hot.margin_add = std::strtod(a2.c_str(), &e2);
            if (a1.empty() || a2.empty() || *e1 != '\0' || *e2 != '\0') die("--hot-margin " + m + ": expected MUL:ADD");
        }
        else if (a == "--hot-check") ec.hot.check = true;
        else if (a == "--hot-homes") {
            // N: at most N slab experts homed in the home order (experts.hpp)
            const std::string v = value(i);
            char* end = nullptr;
            const long long n = v == "all" || v == "spare" ? 0 : std::strtoll(v.c_str(), &end, 10);
            if (v != "all" && v != "spare" && (v.empty() || *end != '\0' || n < 1))
                die("--hot-homes " + v + ": all, spare or a number of slab experts >= 1");
            ec.hot.host_homes = v != "spare";
            ec.hot.host_homes_max = v == "all" || v == "spare" ? -1 : n;
        }
        else if (a == "--deterministic") {
            const std::string m = value(i);
            if (m != "on" && m != "off") die("--deterministic " + m + ": expected on or off");
            ec.hot.deterministic = m == "on";
        }
        else if (a == "--cpu-numerics") {
            const std::string m = value(i);
            if (m != "gpu" && m != "fork") die("--cpu-numerics " + m + ": expected gpu or fork");
            ec.cpu_numerics = m == "gpu" ? e::CpuNumerics::Gpu : e::CpuNumerics::Fork;
        }
        else if (a == "--route-trace") ec.route_trace = value(i);
        else if (a == "--dry-run") dry = true;
        else if (a == "--serve") serve_mode = true;
        else if (a == "--eos-ids") eos_text = value(i);
        else if (a == "--log-ids") log_ids = true;
        else if (a == "--batch") {
            const double n = number(i);
            if (n > (double) e::kMmvqMaxCols || n != std::floor(n))
                die("--batch " + std::to_string(n) + ": 0 (off) to " + std::to_string(e::kMmvqMaxCols) + " slots");
            fc.batch = (int) n;
        }
        else if (a == "--batch-ctx") {
            const double n = number(i);
            if (n < 1 || n != std::floor(n)) die("--batch-ctx " + std::to_string(n) + ": a whole number of positions >= 1");
            fc.batch_ctx = (int64_t) n;
        }
        else if (a == "--interleave-chunk") {
            const double n = number(i);
            if (n != std::floor(n) || n > 65535) die("--interleave-chunk " + std::to_string(n) + ": a whole number of tokens, 0 to 65535");
            interleave = (int64_t) n;
        }
        else if (a == "--checkpoints") {
            const double n = number(i);
            if (n < 1 || n != std::floor(n) || n > 1024) die("--checkpoints " + std::to_string(n) + ": a whole number of slots, 1 to 1024");
            checkpoints = (int) n;
        }
        else die("unknown argument " + a + " (--help)");
    }
    if (model_path.empty()) die("--model is required (--help)");
    if (serve_mode) {
        if (!tokens_text.empty() || !tokens_file.empty() || !dump_dir.empty() || !force_dir.empty() || fc.forced_moe_input)
            die("--serve takes its prompts on stdin: not with --tokens, --tokens-file, --dump-layers, --force-selection or "
                "--forced-moe-input");
        fc.checkpoints = checkpoints < 0 ? 4 : checkpoints;
        if (fc.batch && fc.batch_ctx > fc.ctx) die("--batch-ctx " + std::to_string(fc.batch_ctx) + " is past --ctx " + std::to_string(fc.ctx));
        if (fc.batch && fc.readers != e::ReaderMode::Reuse) die("--batch reads with Reuse Mode: not with --readers dense");
        if (interleave >= 0 && !fc.batch) die("--interleave-chunk is --batch's (a prompt read while slots decode)");
        return serve_main(model_path, eos_text, log_ids, dry, f16, fc, ec, interleave);
    }
    if (!eos_text.empty() || log_ids || checkpoints >= 0 || fc.batch)
        die("--eos-ids, --log-ids, --checkpoints and --batch are --serve's");
    fc.checkpoints = 0;   // one prompt and its decode: nothing to resume
    if (fc.forced_moe_input && force_dir.empty()) die("--forced-moe-input takes the reference of --force-selection");
    if (tokens_text.empty() == tokens_file.empty()) die("one of --tokens and --tokens-file is required");
    std::vector<int32_t> prompt;
    if (!tokens_file.empty()) {
        std::ifstream f(tokens_file);
        if (!f) die("cannot open " + tokens_file);
        std::stringstream ss;
        ss << f.rdbuf();
        prompt = parse_ids(ss.str(), tokens_file);
    } else {
        prompt = parse_ids(tokens_text, "--tokens");
    }
    fc.dump_dir = dump_dir;
    fc.chunk = std::min(fc.chunk, fc.ctx);
    std::unique_ptr<e::SelectionDump> forced;
    if (!force_dir.empty()) {
        forced = std::make_unique<e::SelectionDump>(e::SelectionDump::open(force_dir));
        if (forced->prompt() != prompt) die("--force-selection " + force_dir + ": its prompt is not the one given");
        if ((int64_t) forced->fed().size() < n_predict)
            die("--force-selection " + force_dir + ": " + std::to_string(forced->fed().size()) + " decode tokens, --n-predict " +
                std::to_string(n_predict));
        fc.forced = forced.get();
    }

    const auto t0 = std::chrono::steady_clock::now();
    const e::HostModel host = e::load_gguf_model(model_path);
    e::Model model;
    std::string err;
    if (!model.init(host, dry, f16, err)) die(err);
    const e::DenseDims& dd = model.dense_dims();
    std::printf("dsv41-engine: %s: %d layers, n_embd %d, %d experts (%d used), vocab %d; weights checked%s\n",
                model_path.c_str(), model.n_layer(), dd.n_embd, dd.n_expert, dd.n_expert_used, dd.n_vocab,
                f16 ? ", attention on f16 weights" : "");
    std::printf("prompt: %zu tokens; %lld decode steps; ctx %lld, chunk %lld, readers %s%s%s%s%s\n", prompt.size(),
                (long long) n_predict, (long long) fc.ctx, (long long) fc.chunk,
                fc.readers == e::ReaderMode::Dense ? "dense" : "reuse",
                forced ? "; selections and decode tokens forced from " : "", forced ? force_dir.c_str() : "",
                fc.forced_moe_input ? ", and every layer's MoE input (its ffn_norm)" : "",
                fc.prefill == e::QPrefill::Mmq ? "; prefill matmuls on MMQ (int8 activations)" : "");

    // The fork's numerics (the default) said once at the load
    if (ec.cpu_numerics == e::CpuNumerics::Fork)
        std::fprintf(stderr, "dsv41-engine serve: the CPU experts in the fork's numerics (--cpu-numerics fork, the default)"
                     ": a routed expert's bits follow its tier, so no batch slot ends reproducible\n");
    if (dry) {
        e::ExpertEngine experts;
        if (!experts.init(model, ec, std::min(fc.chunk, fc.ctx), nullptr, nullptr, true, err)) die(err);
        e::forward_dry_run(model, experts, fc, prompt.data(), (int64_t) prompt.size(), n_predict, stdout);
        std::printf("dsv41-engine: dry run done, no CUDA call\n");
        return 0;
    }

    e::Forward fw;
    if (!fw.init(model, ec, fc, err)) die(err);
    fw.print_memory(stdout);
    std::printf("load: %.0f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    std::vector<float> logits;
    std::vector<int32_t> generated, fed;
    fw.begin_request(ec.hot.policy, false, false, ec.hot.deterministic);
    fw.prefill(prompt.data(), (int64_t) prompt.size(), logits);
    if (fc.timing) print_times(stdout, "prefill", fw.times(), fw.experts().times());
    fw.reset_times();
    for (int64_t s = 0; s < n_predict; ++s) {
        generated.push_back(argmax(logits));
        fed.push_back(forced ? forced->fed()[(size_t) s] : generated.back());
        fw.decode(fed.back(), logits);
    }
    if (fc.timing && n_predict > 0) print_times(stdout, "decode", fw.times(), fw.experts().times());
    if (n_predict > 0) {
        const e::ExpertTimes& x = fw.experts().times();
        const double pairs = (double) (x.hits + x.misses);
        std::printf("decode: %lld steps, hit rate %.4f, %s\n", (long long) n_predict, pairs > 0 ? (double) x.hits / pairs : 0.0,
                    hot_text(fw).c_str());
    }
    std::printf("generated_token_ids: %s\n", join(generated, " ").c_str());
    if (forced) std::printf("fed_token_ids: %s\n", join(fed, " ").c_str());

    if (!dump_dir.empty()) {
        std::ofstream mf(dump_dir + "/engine-manifest.json");
        mf << "{\"tool\": \"dsv41-engine\", \"model_path\": " << json_string(model_path) << ", \"readers\": \""
           << (fc.readers == e::ReaderMode::Dense ? "dense" : "reuse") << "\", \"f16\": " << (f16 ? "true" : "false")
           << ", \"hot_set\": " << json_string(ec.hot_set) << ", \"n_embd\": " << dd.n_embd << ", \"n_layer\": " << model.n_layer()
           << ", \"n_vocab\": " << dd.n_vocab << ",\n \"prompt_token_ids\": [" << join(prompt, ", ")
           << "],\n \"generated_token_ids\": [" << join(generated, ", ") << "],\n \"fed_token_ids\": [" << join(fed, ", ")
           << "],\n \"forced_selection\": " << json_string(force_dir) << ", \"forced_moe_input\": "
           << (fc.forced_moe_input ? "true" : "false") << ", \"prefill\": \""
           << (fc.prefill == e::QPrefill::Mmq ? "mmq" : "dequant-f16") << "\",\n \"ubatches\": [";
        const auto& steps = fw.dump_steps();
        for (size_t u = 0; u < steps.size(); ++u)
            mf << (u ? ", " : "") << "{\"ubatch\": " << u << ", \"pos0\": " << steps[u].pos0 << ", \"n_tokens\": "
               << steps[u].n_tokens << "}";
        mf << "]}\n";
        if (!mf) die("writing " + dump_dir + "/engine-manifest.json failed");
        std::printf("dump: %s (%zu calls)\n", dump_dir.c_str(), steps.size());
    }
    return 0;
}
