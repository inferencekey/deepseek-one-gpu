# Determinism of the decode

Three properties are checked: identical temperature-0 requests on one server load give identical tokens, cold and
after another request (the greedy check); the same seed gives the same sampled reply (the seed check); and a batch of
identical temperature-0 requests gives the tokens of one request alone (the batch check). A first session on
2026-10-13 (A100-SXM4-80GB, the real Q2_K) failed the first two: prompt A greedy three times diverged at token 14
("again", 305 against 362) and 16 ("after B", 64018 against 1747), seed 1234 twice at token 15. Its engine log: the
adaptive hot set made 3790 swaps in request 1 (196 batches, 59 of them still copying at a boundary and published
later), and the decode's pairs split differently between requests (request 1: 50970 hits and 10230 misses; request 2:
51741 and 9459). This page lists every source of run-to-run variation in the decode path (rows a to f), what the
deterministic mode changes, and what it costs. Placement-independent expert bits (below) removed row b, the GPU and CPU
experts' different bits; an RTX PRO 6000 session then found one more path, the promoted request's last token read by
prefill (the section after it). The engine now starts with the CPU experts in the fork's numerics, so row b is back for
batched requests by default (the last section before the checks).

## The sources, read in the code (dev VM, 2026-10-13)

| | source | varies between identical requests? | evidence |
|---|---|---|---|
| a | expert placement: the adaptive hot set's swaps, the copy streams' deferral, the prefill's prefetch | **yes** (the cause) | `experts.cpp` `finish_batch` (860) queries the batch's events (`cudaEventQuery`, 866) and returns false while they copy; `decode_boundary` (1790) then left the batch open (`++swap_deferred`, 1804): when a swap is published depended on the copy streams' speed. `begin_request` (1744) kept the slab the last request left (no reset unless `hot_reset=1`) and, on a RESUME, its counts. The prefill's prefetch (`prefetch`, 625) picks the first experts of a fixed order that are outside the slab: a function of the placement, not of time. |
| b | GPU and CPU numerics of the same expert | **yes, by design** at first; **no** with `--cpu-numerics gpu` (the default for a while); **yes** again for batched requests under the current default `--cpu-numerics fork` (below) | a hit: `qmatmul.cu` `qmatmul_experts` quantizes the rows to q8_1 (766) and sums with MMVQ (771); a miss: `experts_cpu.cpp` quantized to Q8_K (224) and summed with the fork's AVX2 `vec_dot` (`cpu_kernel.h fast_kernel_rows`). experts_test (b) stated the gap: within 1e-3 rel L2 per call, not bit for bit. The deterministic mode left it (below); placement-independent expert bits made the CPU experts compute the hit's arithmetic (`experts_cpu_mmvq.cpp`, `CpuNumerics::Gpu`), bit for bit; that is now the `--cpu-numerics gpu` option, not the default. |
| c | the order the hit and miss outputs are summed in the combine | no | `experts_kernels.cu` `combine_kernel` (24-40): per token, the pairs in slot order s = 0 .. K-1 (the router's top-k order, the fork's `build_moe_ffn` views), `acc = s == 0 ? v : __fadd_rn(acc, v)` (36), each pair's row found through the row map whatever tier made it; the CPU rows reach the device before the combine through the doorbell flag (`doorbell_wait_kernel`, 59; `misses_in_kernel`, 67), so arrival order never enters a sum. Kept in slot order rather than expert index: it is the fork's order, and experts_test (a) holds the CPU path bit for bit to the fork's `ffn_moe_out` with it. The placement-independent bits kept it too (below): the slot order is the token's router order, a function of the token alone, as an expert-index order would be. |
| d | GPU atomics or nondeterministic reductions (attention, indexer, hc, router) | no | `hc.cu` 297 and 385: `atomicAdd` on a ticket only elects the block that finishes the step; the values it reduces were written by fixed blocks in a fixed order. `indexer.cu` 353 and 358: integer `atomicAdd` into the top-k radix histogram (counts commute), the selected rows then written by index (`emit_row`, ties to the lower row). No float atomics in `attention.cu`, `router.cu`, `dense.cu`, `qmatmul.cu`. cuBLAS (`router.cu` 166, `qmatmul.cu` 682 and 700) runs with a fixed workspace per stream (`qmatmul.cu` 535, `dense.cu` 190) and the default atomics-off mode, which cuBLAS documents as bit-reproducible on one GPU model and toolkit. |
| e | CPU thread partitioning | no | `experts_cpu.cpp` (12, 117): work items from an atomic counter, each output element computed whole by one item, so the thread that runs it never changes it; experts_test (a) holds it bit for bit with 1 and 3 threads, and (d) below with 1 and 3. |
| f | the prefix snapshot path | no for the greedy check; yes in general across histories | `session.cpp` `plan` (67-70): a request reuses the live state or the prompt-end snapshot only when its ids strictly extend them, so an identical prompt always starts cold (the 2026-10-13 log: "cold resume of 0" for every A). A request that does extend a cached prefix gets that prefix's KV as the earlier request computed it (chunked prefill or decode steps), which can differ in its last bits from a cold read of the same tokens: the reply is a function of the history, the same for the same history. `KvCache::reset` (`kv_cache.cpp` 89) keeps the buffers, but a cold sequence reads only positions it wrote (`attention.cu` 249-256: the pooling carry is read only for positions before p0 of this sequence). |

Outside the decode path, one more: the prefill split's rates are measured once at the engine's start
(`measure_split_rates`, `experts.cpp` 716, called at 1389), so which of a prefill chunk's experts the CPU computes can
differ between two engine loads; within a load the rates are fixed and `split_plan` (662) is a function of the
placement and the routing. The greedy check is one load. The sampler (`sampler.cpp`) is a `std::mt19937_64` seeded per request
(27-28): the seed check failed only through the logits.

## The deterministic mode

On by default for every request; `dsv41-engine --deterministic off` sets the default off, the GEN
key `det=0|1` sets it per request; over the server, a request's `"strata_deterministic": false` and the
deepseek-v4.1 config's `"deterministic": false`.

- **Placement fixed per request** (`hot_policy.hpp` `hot_request_start`): a request starts from the hot set's slab
  with zero counts (the reset of `hot_reset=1`; a RESUME keeps no counts), whatever the requests before it did. After
  a reset the slab holds the hot set's experts and the homes the warm tier's (a swap moves an expert between the slab
  and a home, so the set in each is restored); slot and home numbers may differ, which no kernel's numerics read
  (MMVQ finds a pair's slot through the residency row).
- **Swaps at deterministic points** (`hot_boundary_waits`): a decode boundary that finds the open batch still copying
  waits for it, so the swaps planned at token t (from the request's own counts) are published right after token t.
  The adaptive policy stays: it measured 62.6 tok/s against the static slab's 42.0 on short prompts.
- **Combine and reductions**: already in a fixed order (rows c, d, e above); nothing changed.

Why not placement-independent numerics at first (they came later for the batch check, below): (b) would need the
GPU's expert path in the CPU's Q8_K arithmetic (or the reverse), the decode's hottest kernel, and a parity rework
against the fork; the reset costs one request start. Measured reset cost before the mode (`hot_reset=1` in the
benches, the real Q2_K): 3119 and 3298 swaps in 1923 and 2008 ms on the A100-SXM4-80GB (2026-10-09), 1120 and 3522
swaps in 632 and 2001 ms on the RTX PRO 6000 Blackwell WS (2026-10-10), before a request's first token. The decode
cost of the waits is measured on a GPU (`det_sampling_check.py`'s cost row, the mode on against off on prompt S, 3
runs each); the dev VM has no GPU to time it. The same row reports each request's first token with the mode on and
off: the engine line's `first token at` is timed from `session.run`'s start, after `begin_request`'s reset, so the row
gives it, the hot set line's `reset before it ... in X ms` and their sum.

With conversation checkpoints the reset moves to the end of a request. After a request's DONE the serve loop calls
`Forward::end_request` (`ExpertEngine::end_request`, `hot_policy.hpp` `hot_request_end_resets`: in the mode): the open
batch published, then `begin_request`'s rounds back to the hot set's slab, while the engine is idle. The next request
in the mode then starts from that slab with zero counts, as before, and its start's reset finds nothing to move
(`reset before it 0 swaps`); the reset at the start stays for a request after one in the mode off, which does not
reset at its end. A request that arrives while the idle reset still runs waits only for the rest of it: its hot set
line ends in `idle reset after the last request N swaps (X MiB) in Y ms, waited W ms` (W from its GEN line's arrival
to the reset's end), and `det_sampling_check.py` counts that wait in the reset before its first token. One
consequence: a request in the mode off right after one in the mode now starts from the hot set's slab (the mode's end
left it there) instead of the slab the decode left; the mode off is for measurement only.

Host homes (`--hot-homes all`, README "Host homes and the reset layer by layer"): with host homes the end's reset
restores in place layer by layer on its own stream and is not waited for; the planners are the hot set's slab at once,
and each layer's computation in the next request waits for that layer's copies (`HotRestore`), so the placement every
layer computes with is still exactly the hot set's slab and the rule above holds unchanged. The swaps themselves are
the two-way path's swap for swap (only the victims' copies down go), so the tokens are the same. Partial host homes
(`--hot-homes N`): the layers homed in full restore that way, the others in rounds that `end_request` waits for;
neither changes the placement each layer computes with, and the swaps are still the two-way path's swap for swap, so
the rule holds unchanged.

Batched decode (`dsv41-engine --batch N`), as it stood before placement-independent bits (next section): a request
that decodes alone keeps everything above (the server sends it as a GEN, the solo path). A request that decodes in a
batch slot beside another, or whose prompt was read while another slot decoded, is not reproducible in the mode: the
batch has one hot set whose swaps follow the union of every slot's routes, and a request start does not reset it
under the others, so the placement a slot computes with depends on its neighbours (and an expert's GPU and CPU values
differ in their last bits, row b above). Its BDONE says rep=0 and its log line why; the server marks the reply
`"strata_reproducible": false`. The same holds for a slot that decodes alone while another request's prompt is read
beside it, whether or not that request then joins a slot (the read's counts join the slot's hot set,
`BatchSlots::read_beside`; one token, an end of reply at once, a client that left during the read, a request run
alone past a slot): its BDONE says rep=0 (`dsv41_engine_batch_test` check 9). A client that needs the guarantee sends
`"strata_reproducible": "required"` (docs/DETAILS.md, Strata's own keys): the request waits until no slot decodes,
decodes alone in the mode and its reply says `"strata_reproducible": true`. Every batched kernel already computes a
row as it is computed alone (batch invariance under the same placement, `dsv41_engine_batch_test`), so batched
requests could be made reproducible by a CPU expert path bit-identical to the GPU's or a placement rule independent
of neighbours.

Long prompts beside decoding slots (`--interleave-chunk N`): a prompt read beside decoding slots is read in chunks of
N with a batch step between two. Its chunk boundaries depend on the request alone, so under the same placement its
tokens are its alone run's with the same chunks, and the slots decoding through it keep theirs
(`dsv41_engine_batch_test` check 6); it is still read beside other slots, so its BDONE says rep=0 as above. N
defaults to `--chunk`, so by default its chunks are the ones it is read in alone. A prompt read with no slot decoding
is read in `--chunk` as before, and a request alone keeps everything above.

Batched speed: a prompt read beside decoding slots no longer replaces the batch's hot set counts with its own and
takes no prompt target; its counts join the batch's (hot_policy.hpp `merge_prompt`). That changes only the placement
of batched steps, which were not reproducible already (rep=0 above). And a step of two rows or more no longer waits at
its boundary for the open batch of swaps (`decode_boundary(batched)`): its rows are not reproducible whatever the
boundary does, so the batch is deferred as in the mode off; a step of one row still waits. A request alone in the
mode never reads beside slots nor steps beside another: its start (reset, zero counts, its own prompt target) and its
boundaries are as above, bit for bit (q03-test.sh, c02-test.sh).

What the mode does not change: a conversation's next turn (a RESUME) starts its hot set from the hot set's slab too,
so it loses the adaptive policy's carried counts (`hot_policy_sim.py two-turn`, 4 layers of 384 experts: turn 2's hit
rate 0.830 resumed against 0.797 fresh).

## Placement-independent expert bits (`--cpu-numerics gpu`, 2026-10-16)

This section and the next describe the period when the CPU experts computed in the GPU's numerics by default. That
path is now `--cpu-numerics gpu`; the current default is in the last section before the checks.

The batch check asks that a batch of 4 identical temperature-0 requests give one alone's tokens. Row b above was the
one source left that turned a placement into a token: the batch's single hot set follows the union of its rows'
routes, so an expert a request alone gets from the slab it may get from the CPU in a batch, and the two gave
different last bits. The engine removed the difference instead of fixing the placement:

| difference (at first) | GPU hit (`qmatmul_experts`) | CPU miss | with `--cpu-numerics gpu` |
|---|---|---|---|
| activation quantization | q8_1: blocks of 32, `d = amax / 127` kept finite, stored as f16 with the f16 sum (`quantize_q8_1_kernel`, `mmvq.cuh q8_1_warp`) | Q8_K: blocks of 256, `d = 1 / (-127 / max)` in f32, `bsums` (`quantize_row_q8_K_ref`) | the CPU quantizes to q8_1 with the kernel's operations (`experts_cpu_mmvq.cpp quantize`: the IEEE division, `roundf`'s halves away from zero, the butterfly sum, the f16 conversion of `f16_bits.hpp`) |
| float order of a row | per q8_1 block `fma(d8, dp * sc, sum)` (nvcc's contraction choice, unwritten), 128 threads' running sums over the superblocks `kbx = tid / 16 + 8 i`, warps 1 to 3 added to warp 0 in turn, the xor butterfly | the fork's AVX2 lanes: integer sums over a superblock, two `fmadd` a superblock into 8 lanes, `hsum_float_8` | the traits' float operations written as `_rn` intrinsics (`mmvq.cuh`, so nothing is left to the compiler), and the CPU runs those threads' sums in that order: thread `tid = 16 j + 8 n + k` is lane `k` of the accumulator of thread group `j` and half `n`, the integer dots one `maddubs` and one `madd` by the lane's scale (exact), the float steps the kernel's (`experts_cpu_mmvq.cpp`) |
| exp in the SwiGLU clamp | CUDA's `expf` (2 ulp) | glibc's `expf` | one implementation for both (`expert_act.hpp expf_det`: Cephes' expf, one rounding per operation; within 1.003 ulp of the exact exp on every 61st float of [-87.33, 88.72]), used by every CUDA FFN clamp (`ffn_device.cuh`) and the CPU's |
| h and down | h to q8_1, MMVQ | h to Q8_K, `vec_dot` | as the rows above |
| the sum of the experts' outputs | the device's combine in slot order, whichever tier made a row (row c) | the same | unchanged: slot order is a fixed order per token, as expert index would be, and it keeps the fork's |
| a prompt's prefill split (`split_plan`, chunks of up to 2048 tokens) | the experts outside the slab with few pairs on the CPU, the others on the prefill GEMM (f16 tensor cores) | | in the deterministic mode the candidates are the experts outside the hot set's slab, not outside the slab as a batch has moved it (`ExpertEngine::prompt_deterministic`): the split, and so each pair's arithmetic, is the prompt's own; the GPU's passes are bit-identical whether the slab, the ring or the prefetch region holds an expert (experts_test (b)) |

The integer dot products are exact on both sides, so with these the CPU's value of an expert is the GPU's bit for bit
(`dsv41_engine_expert_bits_test` on the dev VM against the line-for-line reference `mmvq_ref.cpp`; `--gpu` on a dev pod
against the CUDA kernels). A request in the deterministic mode is then reproducible whatever decodes or is read beside
it (with `--cpu-numerics gpu`): BDONE and BADM say rep=1 for it, rep=0 only in the mode off; the server's
`"strata_reproducible"` follows (docs/DETAILS.md), and a read beside a lone slot has nothing left to mark. The fork's
CPU numerics stay as `CpuNumerics::Fork` for the oracle parity (experts_test (a), dsv41-layer-ref). What it costs is
the CPU's: the q8_1 blocks of 32 put two float steps per 32 values on the CPU where the fork's Q8_K has two per 256,
and the CPU experts in the GPU numerics take 1.6 to 1.7x the fork's numerics' time on one thread on the dev VM (AMD
Ryzen 7 5700X, one real-size expert, 1 and 4 tokens) and about the same at 8 threads for one token, where the memory
bounds both (`dsv41_engine_expert_bits_test`'s table). The deterministic mode's reset and waits stay: they keep a
request's placement, and so its speed, its own.

Checks (dev VM): `dsv41_engine_expert_bits_test`; `dsv41_engine_batch_test` checks 9 to 11 (4 identical requests on
another slab and on one drawn anew at every call, 4 different ones with one joining mid-batch and a prompt read
beside, each its alone run bit for bit and rep=1; the negative controls: the fork's CPU numerics for the misses, and a
read beside the slots in the mode off, differ); `dsv41_engine_experts_test --gpu` (a dev pod) requires decode bit for
bit the same in all five placements; `dsv41_engine_batch_test --gpu-swaps` (a dev pod) runs its checks 1 to 4 on the
engine with the hot set swapping at every token.

## The batch check on the RTX PRO 6000: the promoted chains, and the fix (2026-10-17, the GPU's numerics)

The batch-determinism session (`tools/dsv41/pod/e07.sh` E07_PLAN=cdet; logs `det-batch.json` and `quality-engine.log`
with `--log-ids`; NVIDIA RTX PRO 6000 Blackwell Server Edition, AMD EPYC 9335, the real Q2_K, every request greedy
with max_tokens 256, the deterministic mode on) failed the batch check: in each phase one of the 4 replies left the
same request alone, at token 67 (identical prompts, streamed and not: 13953 alone, 1425 batched) and 91 (different
prompts: 42830 against 13750), and every reply said `"strata_reproducible": true`. The engine's slot lines put each
phase's token chains back together:

| phase | the chain that differs (engine requests) | the other 3 chains |
|---|---|---|
| identical | 20 on the solo path, then 21 in slot 0: apart at 67 | 22, 23, 24: 256 tokens each, the alone run's |
| stream | 25, then 26: apart at 67 | 27, 28, 29: the alone run's |
| different | 30, then 31 (the docs prompt): apart at 91 | 32, 33, 34 (126, 256, 147 tokens): their alone runs' |

The 9 chains that went into a slot at once (about 1,950 tokens in steps of 3 to 4 rows, their prompts read beside
decoding slots, the batch's placement by the union of its routes) matched their alone runs bit for bit: the batched
kernels, the placement and the placement-independent expert bits hold on the real engine. The 3 that differ are the
promoted ones (serve/server.py generate_batched): a request that started on the solo path, was STOPped after 2 tokens
when the others arrived, and was sent again as a BGEN of its prompt and both tokens. The engine held the prompt and
the first token (the second was emitted, never fed), so Session::run read that one token left with `prefill(T=1)`:
the prefill expert path (`begin_call(T, prefill)`), the H2D token copy and `hc_prefill_step`, where the request alone
feeds the same token through `Forward::decode` (the token graph, `hc_decode_step`). That one position's state had
other last bits, carried forward until a near tie flipped ("Lookups repeat" against "Lookup repeats", the stream
joining up again two tokens later).

The fix (session.hpp): the Session keeps the token the last request emitted and did not feed; a live resume whose one
token left is that token feeds it with decode, as alone, and the engine's log says so ("its 1 token read was fed by
decode"). A longer tail (a conversation's next turn) stays a prompt read, as alone; nothing else changed numerics. The
engine has no copy of a slot back to the live sequence (INFO `slot_cache=0`), so a request goes back to the live
sequence only as a new read under the same rule. With it, a request decoded in the deterministic mode beside other
slots says rep=1 and `"strata_reproducible": true` again (the flag was held false until the fix was measured);
`det_batch_check.py` checks every reply's flag against its own chain, so a true for a chain not shown equal fails the
next session.

Tests that fail without the fix and pass with it: `dsv41_engine_session_test` (`tools/dsv41/engine/session_test.sh`
builds it on session.cpp and on a copy without the fix: the 3 promotions saw `prefill 1`), `dsv41_engine_batch_test`
check 12 (a request stopped after 1, 2 and 5 tokens and promoted into slot 0 with 3 beside it, every call's logits its
alone run's) with check 13's prefill(T=1) against decode lines, and `det_batch_check.py` through c05-test.sh on
fakes.py with `FAKE_ENGINE_PROMOTION_PREFILL` (the engine before the fix: only the promoted chains apart). The dev
pod's c05b plan runs checks 12 and 13 on the CUDA engine. The rerun's dry run planned 59.2 min, about 2.57 USD at 2.60
USD/h, with the model copied from a network volume (72.6 min, about 3.15 USD, downloading it from Hugging Face).

That rerun (2026-10-19, RTX PRO 6000 Blackwell Server Edition, AMD EPYC 9575F, the real Q2_K, greedy, max_tokens
256, the CPU experts in the GPU's numerics) passed the batch check: 12 chains decoded beside other slots, 3 of them
promoted, each its alone run's tokens and each reply saying `"strata_reproducible": true`. Results: docs/RESULTS.md.

## The current default: the fork's CPU numerics (2026-10-19)

The same rerun's A/B measured what the GPU's numerics cost: the single user (the deterministic mode off, load-short's
turns of 256 tokens at a 1K context, RTX PRO 6000 Blackwell Server Edition) decoded 47.04 tok/s with the CPU experts in
the GPU's numerics against 63.34 in the fork's (-25.7%). The requirement became "no quality loss from batching"
rather than bit-identical batched replies, so dsv41-engine starts with `--cpu-numerics fork`, the fast path;
`--cpu-numerics gpu` stays as an explicit option (in the config's `args`).

With the fork's numerics a routed expert's last bits depend again on whether the GPU's slab or the CPU computed it
(row b), and a batch places the experts by the union of its rows' routes. So under the default a request in the
deterministic mode that decodes or is read beside other slots is NOT the same tokens as the same request alone, and
the engine says so (dsv41-engine.cpp `serve`, `tier_free` = the GPU's numerics):

- a slot's end (BDONE): rep = deterministic && tier_free, so every reply that decoded in a batch slot says rep=0 and
  `"strata_reproducible": false` under the default;
- a request that ended at its admission (BADM `<slot>` 0, one token, or one past `--batch-ctx` that ran alone): rep =
  deterministic && (tier_free || no slot decoding beside it);
- its "read beside" and "ran alone" log lines say "not reproducible (--cpu-numerics fork: its bits follow the
  placement)".

A request alone in the deterministic mode keeps its guarantee in either numerics: nothing moves its placement but
itself. Identical tokens with batch slots on therefore come two ways: `"strata_reproducible": "required"` (the request
waits until no slot decodes and decodes alone; its reply says `true`), or an engine started with `--cpu-numerics gpu`
(every request in the mode reproducible beside others, at the single user's 47.04 against 63.34 tok/s above). The
final session with the default numerics (2026-10-20) logged 0 of its batched requests reproducible, as the rule says.

## Checks (dev VM)

- `dsv41_engine_experts_test --determinism tiny-b.gguf` (experts_test (d)): tiny-b's 6 layers decode with each
  layer's slab driven by `HotLayerPlan` and the two rules above, hits on the engine-numerics reference, misses on the
  CPU experts, the slot-order combine; injected per request: whether a batch's copies are done at each boundary, the
  order the pairs' outputs land in, 1 or 3 CPU threads. Requests A, A, B, A (4 prompt and 24 decode tokens each):
  mode on, the three A's bit for bit the same; off, the second A differs from decode token 0 (the slab the first
  left); off with `hot_reset=1`, from token 1 (the copies' timing alone). With the reset at the end, every request's
  reset runs at its end (31-33 swaps each), none at its start, and the A's are still bit for bit the same (280 hits
  and 728 misses each, as before the move). Host homes: the mode with host homes and the reset layer by layer (each
  request's restore progress injected, the slab modelled as the bytes each slot holds) bit for bit the two-way runs,
  no layer computing with a slot not yet restored, every home intact; without the per-layer wait (the negative
  control) hits read a victim's bytes and the outputs differ. Partial host homes: the mode with partial host homes (3
  layers homed in full, 1 in part, 2 none on tiny-b; the victims copied down write their slot's bytes into the homes
  the misses read) bit for bit the two-way runs.
- `dsv41_engine_hot_policy_test`: the two rules, the reset at the end, host homes and restore order, and partial homes.
- `tools/dsv41/quality/q03-test.sh`: the protocol's `det=` key and `--deterministic off` in the dry run, the server's
  forwarding, and the first determinism session rehearsed on `fakes.py`'s placement-dependent engine
  (`FAKE_ENGINE_PLACEMENT=5`): the greedy and seed checks pass in the mode, the mode-off row fails the greedy check at 5
  with both requests' placement records.
