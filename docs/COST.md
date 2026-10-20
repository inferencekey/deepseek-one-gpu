# What it costs: self-hosted against the DeepSeek API

Short answer: **per token, the DeepSeek API is cheaper, by about 4 to 18 times.** Run this engine when
you need what the API cannot give you (your data stays on your machine, no shared limits, a cost you fix yourself),
not to save money on tokens.

## The prices used

**DeepSeek API**, model `deepseek-flash` (DeepSeek-V4.1-Flash), from
<https://api-docs.deepseek.com/quick_start/pricing>, fetched 2026-10-20. USD per 1M tokens:

| | Off-peak | Peak |
|---|---|---|
| Input, cache hit | 0.003 | 0.006 |
| Input, cache miss | 0.15 | 0.30 |
| Output | 0.60 | 1.20 |

Off-peak is half the peak price. Peak hours are 01:00 to 04:00 and 06:00 to 10:00 UTC, Monday to Friday; every
other hour is off-peak. DeepSeek can change these prices.

**Self-hosted**: one rented NVIDIA RTX PRO 6000 Blackwell Server Edition on RunPod (SECURE cloud) at
**2.49 USD per hour**, the price of the pod the final speed numbers were measured on. Cloud GPU prices change over
time and between providers; put your own price into the formulas below.

## Cost per million generated tokens

Self-hosted cost = hourly price / (tokens per second x 3,600) x 1,000,000. The speeds are the measured ones from
[RESULTS.md](RESULTS.md).

| Setup | Speed (tok/s, all users) | USD per hour per user | USD per month per user (24/7, 730 h) | USD per 1M generated tokens |
|---|---|---|---|---|
| Self-hosted, 1 user | 63.11 | 2.49 | 1,818 | **10.96** |
| Self-hosted, 4 users at once | 127.58 | 0.62 | 454 | **5.42** |
| Self-hosted, 4 users, with their 16K prompt reads | 95.03 | 0.62 | 454 | **7.28** |
| Self-hosted, 8 users at once | 152.53 | 0.31 | 227 | **4.53** |
| DeepSeek API, output, off-peak | | | | **0.60** |
| DeepSeek API, output, peak | | | | **1.20** |

How many times the API's output price each self-hosted row is:

| Self-hosted setup | vs API off-peak (0.60) | vs API peak (1.20) |
|---|---|---|
| 1 user | 18.3x | 9.1x |
| 4 users | 9.0x | 4.5x |
| 4 users, prompts included | 12.1x | 6.1x |
| 8 users | 7.6x | 3.8x |

What this table leaves out, on both sides:

- The API also bills input tokens; the self-hosted rows other than "prompts included" leave out the time spent
  reading prompts. Neither side is flattered much: see the per-chat example below, which counts both.
- The self-hosted numbers assume the pod is generating 100% of the time. If your 4 users are idle half the
  time, the real cost per token doubles.
- Not counted for the pod: disk, and the start-up (the 264.5 GB model download and load, about 10 minutes).

## One typical chat turn

2,000 input tokens and 500 output tokens.

**DeepSeek API**

| | Off-peak | Peak |
|---|---|---|
| Input, cache miss: 2,000 x 0.15 (or 0.30) / 1M | 0.00030 | 0.00060 |
| Output: 500 x 0.60 (or 1.20) / 1M | 0.00030 | 0.00060 |
| **Total** | **0.00060 USD** | **0.00120 USD** |

With the input already in DeepSeek's cache (a later turn of the same conversation), off-peak drops to about
0.00031 USD.

**Self-hosted at 2.49 USD/h** (estimate built from measured speeds):

- Reading the 2,000-token prompt: about 1 s of pod time. Measured reference points: a 1,057-token prompt reached its
  first token in 0.91 s, and a 16,413-token prompt in 3.78 s, on the same kind of pod. While a prompt is read, the
  other users' decoding pauses, so the read costs the whole pod: about 1 s x 2.49 / 3,600 = 0.0007 USD.
- Writing 500 tokens:
  - alone, at 63.11 tok/s: 7.9 s of the whole pod, 0.0055 USD;
  - as one of 4 users at once, at 31.87 tok/s, paying a quarter of the pod: 500 x 5.42 / 1M = 0.0027 USD.
- **Total: about 0.0062 USD alone, about 0.0034 USD as one of 4 busy users.** That is about 3 to 10 times the API's
  price for the same turn.

## When self-hosting makes sense anyway

- **Your data never leaves your machine.** Prompts, documents and replies stay on hardware you control. That can
  be a hard requirement: personal or health data, contracts, source code under NDA, regulated industries, or an
  air-gapped network with no internet access at all.
- **No shared limits and no provider outages.** The API has account-level concurrency limits, peak-hour pricing and
  shared capacity. Your own server has none of those: it has 8 batch slots and a queue you size, and it answers
  `503` with `Retry-After` only when you have filled it.
- **A fixed cost you control.** A rented or owned machine costs the same whatever you send through it, which some
  budgets prefer to a per-token bill. You also choose when to upgrade: a model version does not change under you.
- **Owned hardware, amortized.** The rented price above includes the provider's margin. If you own the machine and
  keep it busy, the comparison changes; see the break-even below.
- **You want to change the engine.** It is MIT licensed: change the sampler, the scheduling, the placement, or add
  your own endpoints.

## Break-even for owned hardware

Monthly cost of an owned machine:

```
owned_per_month = hardware_price / months_of_use + power_and_hosting_per_month
```

What the API would charge for the same work in a month, at full load:

```
output_tokens_per_month = tok_per_s x 3,600 x 730
api_per_month = output_tokens_per_month x (output_price + input_tokens_per_output_token x input_price) / 1,000,000
```

Owning is cheaper when `owned_per_month < api_per_month`, so the most you can pay for the machine is:

```
max_hardware_price = months_of_use x (api_per_month - power_and_hosting_per_month)
```

**Example. The hardware and power figures are assumptions for illustration, not quotes:**

- Load: 4 users busy 24/7 with chat-like turns of 2,000 input and 500 output tokens (4 input tokens per output
  token, every input a cache miss). Speed: 95.03 tok/s, the measured 4-user rate with prompt reads included (its
  prompts were 16K tokens, heavier than these, so this is on the safe side).
  That is 249.7 million output tokens a month.
- API price for that month: 249.7 x (0.60 + 4 x 0.15) = **299.7 USD off-peak**, 249.7 x (1.20 + 4 x 0.30) =
  **599.4 USD peak**.
- Assumed power and hosting: 0.8 kW x 730 h x 0.15 USD/kWh = **87.6 USD a month**.
- Assumed use: 36 months.
- Break-even machine price (GPU, a server with about 300 GB of RAM, disk):
  36 x (299.7 - 87.6) = **about 7,600 USD** against off-peak prices,
  36 x (599.4 - 87.6) = **about 18,400 USD** against peak prices.

Read it this way: at off-peak API prices, owning only pays off if the whole machine costs less than about
7,600 USD over three years and is busy all the time. At peak prices the bar is about 18,400 USD. Below full load,
both numbers fall in proportion. Plug in your own hardware quote, power price, load and input/output mix.
