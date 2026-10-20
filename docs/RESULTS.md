# Measured results

Every number on this page was measured on rented cloud machines with the real model, DeepSeek-V4.1-Flash in
Q2_K (`vcruz305/DeepSeek-V4.1-Flash-GGUF`, revision `c4a085541cb53f67ee5e57b63d255e80cef286e7`, 7 files, 264.5 GB,
size and sha256 checked against Hugging Face before each run). Each row says what it was measured on. Nothing here
is an estimate unless it says so.

- [Machines](#machines)
- [Speed with several users](#speed-with-several-users)
- [Before batch slots: one request at a time](#before-batch-slots-one-request-at-a-time)
- [A 64K prompt beside users who are decoding](#a-64k-prompt-beside-users-who-are-decoding)
- [Cached conversations](#cached-conversations)
- [Robustness of the server](#robustness-of-the-server)
- [Quality](#quality)
- [Does batching change quality?](#does-batching-change-quality)
- [Memory actually used](#memory-actually-used)
- [How the numbers were taken](#how-the-numbers-were-taken)

## Machines

All runs used one NVIDIA RTX PRO 6000 Blackwell (96 GB, 97,887 MiB) on RunPod, except the stock-fork quality run,
which used an A100.

| Name below | GPU | CPU | Host link | Memory limit of the pod |
|---|---|---|---|---|
| Server / 9575F | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9575F | PCIe Gen 5 x16 | 283 GB |
| Server / 9655 | RTX PRO 6000 Blackwell Server Edition | AMD EPYC 9655 | PCIe Gen 5 x16 | |
| Workstation / 9355P | RTX PRO 6000 Blackwell Workstation Edition | AMD EPYC 9355P | PCIe Gen 4 x16 (28.6 GB/s host to GPU) | 282 GB |
| A100 / 7742 | A100-SXM4-80GB | AMD EPYC 7742 | | 286 GB |

The engine runs best when it is bound to the CPU socket (NUMA node) the GPU hangs off. The final runs (2026-10-20)
did that with `numactl --cpunodebind`; on the same build that took one user from about 53 to 63 tok/s.

## Speed with several users

Steady state: every prompt already read, each user decoding in its own batch slot, temperature 0, mean of
3 windows of 15 s with exactly N users in every step. Prompts were about 16K tokens per user.

| Users at once | All users together (tok/s) | Each user (tok/s) | Speed-up | Machine | Date | Build |
|---|---|---|---|---|---|---|
| 1 | 63.11 | 63.11 | 1.00x | Server / 9575F, NUMA-bound | 2026-10-20 | 281712c |
| 4 | 127.58 | 31.87 (slowest), spread 0.0 to 0.2% | 2.02x | Server / 9575F, NUMA-bound | 2026-10-20 | 281712c |
| 8 | 152.53 | 19.07 (median), 19.00 (slowest) | 2.42x the 63.11 above (2.54x that session's own baseline) | Server / 9655 | 2026-10-16 | 763675b |

The same 4 replies measured end to end, with their 16K prompt reads included (each prompt read pauses the users who
are already decoding): **95.03 tok/s** for the 4 together, 23.84 tok/s for the slowest whole reply
(Server / 9655, 2026-10-16, build 763675b).

Batch slots cost the single user nothing measurable: 61.55 tok/s with batch slots on against 60.11 off in the
2026-10-16 session (+2.4%).

## Before batch slots: one request at a time

For reference, the server before batch slots answered one request at a time (FIFO). Users decoded at full speed but
waited in line, so the time to the first token grew with every user in front (Workstation / 9355P, 2026-10-13,
build 293e660):

| Users | First token, median (s) | Waiting in the queue, median (s) | Each user while decoding (tok/s) |
|---|---|---|---|
| 1 | 3.311 | 0.0 | 54.05 |
| 2 | 17.319 | 10.964 | 50.29 |
| 4 | 40.415 | 33.962 | 49.71 |
| 8 | 87.029 | 80.3 | 48.56 |

## A 64K prompt beside users who are decoding

A long prompt is read in chunks of 16,384 tokens with a decode step for the other users between chunks, so nobody
freezes while it is read (Workstation / 9355P, 2026-10-16, server and tools 5c582c3, engine b08375d):

| What | Result |
|---|---|
| Longest pause between two tokens of any of the 4 decoding users | 4.048 s (3.737 s during the 65,565-token read) |
| Users who kept decoding through the read | 4 of 4 |
| First token of the 64K prompt, beside the 4 users | 14.928 s |
| First token of the same prompt alone | 16.674 s |

The pause follows the host's PCIe speed. This host moved 28.6 GB/s from RAM to GPU; on the Server Edition hosts
(about 56 GB/s) a 16K chunk beside decoding users took 2.67 s on average and 3.31 s at most (2026-10-16).

## Cached conversations

Four users with 16K-token conversations taking turns, back to back. Every second and third turn resumed its cached
prefix instead of reading the history again (Server / 9575F, 2026-10-14, build 940bbe6):

| What | Result |
|---|---|
| Turns 2+ that resumed a cached prefix | 8 of 8 |
| First token on turns 2+ | p50 1.257 s, p95 1.375 s, max 1.418 s |
| First token on turn 1 (16,413-token prompt read cold) | 3.777 s |
| The same, with 5 s of "reading time" between turns | max 0.636 s |

## Robustness of the server

8 of 8 checks passed over HTTP on the real model, with 8 slots and a queue of 8 (Workstation / 9355P, 2026-10-16):

| Check | What it proves |
|---|---|
| stream | a streamed reply arrives in chunks with usage |
| non-stream | a plain JSON reply |
| batched | two requests decode together |
| disconnect-stream, disconnect-non-stream | a client that hangs up frees its slot; the next waiting request starts (first token 0.302 s and 0.267 s after) and the others keep going |
| overflow | with 8 running and 8 waiting, the next request gets `503` with `Retry-After: 7` and code `queue_full`; the queue then drains |
| no-max-tokens | a request without `max_tokens` gets its slot's room (8,192 tokens by default) without stalling the others |
| slot-room | a 40,001-token prompt fits a slot beside 4 users; a prompt past the slot's 69,632 positions is cut or refused with a clear error |

The per-request deadline (1,800 s) and the engine watchdog (300 s) were tested only against a fake engine.

## Quality

lm-eval 0.4.13 on fixed-seed subsets (seed 20261006): GSM8K 150 items, IFEval 150 items, HumanEval 164 items (all of
it), greedy, thinking off. Metrics: GSM8K exact match (flexible extract), IFEval prompt-level strict accuracy,
HumanEval pass@1. Differences are paired over the same items, with a 95% bootstrap interval (5,000 resamples, seed 0).

| Run | GSM8K | IFEval | HumanEval | Where and when |
|---|---|---|---|---|
| DeepSeek API (`deepseek-flash`) | 0.960 | 0.853 | 0.982 | DeepSeek's servers, 2026-10-06 |
| DeepSeek API, second run | 0.960 | 0.833 | 0.970 | DeepSeek's servers, 2026-10-11 |
| This engine, Q2_K, one request at a time | 0.927 | 0.800 | 0.890 | Workstation / 9355P, 2026-10-13 |
| Stock llama.cpp fork, same Q2_K file | not run | not run | 0.878 | A100 / 7742, 2026-10-13 |

How to read it:

- **On HumanEval the engine matches the stock fork running the same weights**: engine minus fork +1.2 points,
  95% CI [-1.8, +4.3]. 16 of the engine's 18 failed items also fail on the fork. So the gap to the API on HumanEval
  (engine minus API, 2026-10-11 run: -7.9 points, CI [-12.2, -3.7]) is the cost of the 2-bit Q2_K quantization, not
  of this engine.
- On GSM8K and IFEval the engine is 3.3 points under the 2026-10-11 API run on each (5 of 150 items); both intervals
  include zero (GSM8K [-6.7, 0.0], IFEval [-9.3, +2.0]). The fork was not run on them, so the cause is not measured.
- The failures read like a weaker model, not like a broken one: every reply complete and well formed, no garbage,
  no runaway repetition. HumanEval failures are logic errors (an off-by-one, the wrong recurrence).
- Long context, same session: the engine found a "needle" exactly at 16K, 64K and 256K tokens (262,077 tokens).

## Does batching change quality?

The same three tasks, sent 4 at a time so that they share batch slots, against the one-at-a-time engine run above
(0.927 / 0.800 / 0.890). Three sessions on equivalent builds:

| Session | Machine | GSM8K | IFEval | HumanEval | HumanEval minus one-at-a-time (95% CI) |
|---|---|---|---|---|---|
| 2026-10-19 | Server / 9575F | 0.940 (+1.3) | 0.787 (-1.3) | 0.884 | -0.6 [-3.0, +1.2] |
| 2026-10-19 | Workstation / 9355P | 0.920 (-0.7) | 0.800 (0.0) | 0.878 | -1.2 [-3.7, +1.2] |
| 2026-10-20 | Server / 9575F, NUMA-bound | 0.927 (0.0) | 0.800 (0.0) | 0.866 | -2.4 [-4.9, -0.6] |

Plainly:

- GSM8K and IFEval: no loss. No interval excludes zero.
- HumanEval: a small loss, between 0.6 and 2.4 points per session. Pooled over the three sessions it is -1.4 points,
  95% CI [-3.5, +0.2]. In the last session alone the interval lies just below zero.
- The loss is not random noise spread over items: the same few tasks (HumanEval 108, and 115, 140, 162 in two of
  three sessions) fail repeatedly. The cause is not isolated yet. Candidates: the batched kernels and shared expert
  placement, the deterministic mode, or another engine change since the one-at-a-time run.
- Why batched replies can differ at all: by default the CPU computes missed experts with the stock fork's arithmetic
  (`--cpu-numerics fork`, the fast path), whose last bits differ from the GPU's. Which experts the GPU holds depends
  on everyone in the batch, so a reply decoded beside others is not token-identical to the same request alone
  (the reply says `"strata_reproducible": false`). See [DETERMINISM.md](../tools/dsv41/engine/DETERMINISM.md) for
  the two ways to get identical tokens and what they cost.

## Memory actually used

From the final runs (Server / 9575F, 2026-10-20), 8 batch slots of 69,632 positions, 8 conversation checkpoints,
context 262,144:

| Part | Size |
|---|---|
| GPU: attention, dense layers, embeddings (non-expert) | 23.1 GiB |
| GPU: hot expert set | 5,825 of 15,360 routed experts, 69.5 GiB |
| GPU: batch slots' KV cache | 1.70 GiB (costs about 142 experts of the hot set) |
| GPU memory in use at peak | 96,929 MiB of 97,887 |
| RAM: warm experts (pinned) | 9,535 experts, 113.9 GiB |
| RAM: engram tables (pinned) | 60.1 GiB |
| RAM: conversation checkpoints | 6.3 GiB |
| RAM: working memory allowance | 12 GiB |
| RAM: minimum for this placement | 193.2 GiB, plus an 8 GiB margin |
| RAM: "host homes" for the hot set (faster swaps and resets, optional) | up to 69.5 GiB more; 5,298 of 5,825 fitted |
| RAM in use at peak (memory cgroup) | 263.6 GiB of a 263.6 GiB limit (283 GB) |

## How the numbers were taken

- Speed: `tools/dsv41/quality/chat_load.py` against the HTTP server, streaming `/v1/chat/completions`. Per user:
  tokens in the window over the window. Together: the sum.
- Long prompt and robustness: `chat_load.py long` and `tools/dsv41/quality/robust_check.py`.
- Quality: `tools/dsv41/quality/run_lm_eval.sh` and `api_proxy.py` (lm-eval talks to the server through the proxy),
  the subsets in `tools/dsv41/quality/subsets_lean2h_seed20261006.json`, differences by
  `tools/dsv41/quality/analyze.py`.
- Placement: `tools/dsv41/pod/e07_gate.py` built the hot set of every run from the expert usage profile in
  [data/dsv41-expert-counts.json](../data/dsv41-expert-counts.json) (format `strata-moe-counts/1`, 40 layers x 384
  experts, recorded with the stock fork on real prompts).
- Each session first qualified the machine (GPU memory bandwidth, PCIe speed in both directions, host memory
  bandwidth, CPU expert throughput of at least 71 GB/s, clocks and power limit) and refused hosts below the bounds.
- The session scripts are in `tools/dsv41/pod/` (`e07.sh` and `e07-run.sh`).
