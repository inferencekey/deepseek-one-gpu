# MoE parity oracle, replay and hot sets

`dsv41-moe-dump` runs a prompt with greedy decoding through the stock llama.cpp fork and records, for every MoE
layer and every evaluated ubatch, the four tensors the Strata `build_moe_ffn` hook receives and must
reproduce. `read_dump.py` checks a dump and compares two. The DeepSeek-V4.1-Flash dump comes from an A100 pod (below).

The tool reads the graph through the eval callback (`cb_eval`), so the fork is not modified. It is built by
`tools/dsv41/build-inside.sh` against each flavor's fork build tree (CMake project in this directory) and ships
in both fork tarballs as `bin/dsv41-moe-dump`; `verify.sh` runs its `--help` per flavor.

## What is recorded

One raw little-endian file per (kind, layer, ubatch), named `<kind>-L<layer>-u<ubatch>.bin`, values packed
with `ne[0]` fastest (ggml order), plus `dump-manifest.json`.

| Kind | Graph tensor (src/llama-graph.cpp `build_moe_ffn`) | dtype | Shape `ne` |
|---|---|---|---|
| `cur` | MoE input: `src[1]` of the `ffn_moe_logits` mul_mat (the router matmul), e.g. `ffn_norm-<il>` | f32 | `[n_embd, n_tok]` |
| `topk` | `ffn_moe_topk-<il>`, the selected experts (a view of the argsort, copied row by row) | i32 | `[n_expert_used, n_tok]` |
| `weights` | the last of `ffn_moe_weights`, `_softmax`, `_norm`, `_scaled` the graph produces: what multiplies the expert outputs | f32 | `[1, n_expert_used, n_tok]` |
| `out` | `ffn_moe_out-<il>`, the sum of the weighted expert outputs, before the shared expert is added | f32 | `[n_embd, n_tok]` |

`n_tok` is the ubatch's token count, except in a layer the model computes for the output rows only:
OLMoE's last layer applies `inp_out_ids` before its FFN, so its prompt ubatch has 1 row. DeepSeek-V4.1 prunes
after the last layer, so all 40 layers have every row. `ffn_moe_weights_norm` is 2D in the graph; it is
recorded as `[1, n_expert_used, n_tok]` (the same bytes the experts get after the reshape), with the graph
shape in the file's `source_ne`.

`dump-manifest.json` holds the fork commit and build number (from the linked `libllama-common`), the full
command line, model path, prompt, `add_bos`, prompt token ids, `n_prompt`, `n_predict`, seed, the context's
`n_ctx` / `n_batch` / `n_ubatch`, `n_embd`, `n_layer`, the generated token ids and text, the weights tensor
used per layer (`moe_layers`), the ubatches (`decode_call` 0 is the prompt, `k` the k-th generated token as
input, with `pos0` and `n_tokens`), and per file: name, kind, layer, ubatch, dtype, `ne[0..3]` and the source
tensor name.

Decoding: the prompt in one `llama_decode` call (it must fit `-b`), then every generated token, the last one
included, so the dump covers `n_prompt + n_predict` positions. Greedy is the argmax of the last logits; the
sampling options are ignored and end-of-generation tokens do not stop the run (fixed-size dumps). No warmup
decode. The tool fails, naming the tensor and value, on an unsupported dtype, an unexpected shape or op, a
missing kind in a layer, or a non-empty `--dump-dir`.

The eval callback makes the scheduler compute the graph in pieces that end at each recorded tensor, so the
CUDA backend cannot fuse across them (e.g. the top-k MoE fusion). The oracle is the stock graph without
those fusions; the hook's dump uses the same tool, so both sides run the same way.

## CPU smoke on a dev machine (measured)

Model: `allenai/OLMoE-1B-7B-0924-GGUF` revision `70df85ed7132bf21b5acbcb33817f58e7d3cb949`,
`olmoe-1b-7b-0924-q4_k_m.gguf`, 4,213,511,776 bytes, sha256
`f3ae25f6c7e83efb0e3632ef068b321f4d4a644bcfb6c0e01dd8c00e94e53f14` (16 layers, all MoE, 64 experts, 8 used,
n_embd 2048), in `$BUILD_ROOT/testmodels/`. Run from the unpacked cu128 fork tarball inside the cu128
builder image with the toolkit's libcuda stub (no GPU, CPU backend only), 2026-10-06:

```
dsv41-moe-dump --dump-dir DIR -m $BUILD_ROOT/testmodels/olmoe-1b-7b-0924-q4_k_m.gguf \
  -f tools/dsv41/oracle/prompt.txt -n 16 -ngl 0 -c 256 -t 8 -s 42
```

`prompt.txt` is 32 tokens with OLMoE's tokenizer (no BOS). 17 ubatches (the prompt, then 16 single tokens),
1088 files, 12,122,176 bytes of tensors and a 217 KB manifest; 4.6 s wall per run on the dev machine (12 cores, CPU only),
model load included.
`read_dump.py` reports the 16 layers with the four kinds, 0 size mismatches and 0 shape problems; two runs
with the same arguments compare at relative L2 0 in every layer, 100% topk match and identical token ids.

## The DeepSeek-V4.1-Flash dump on an A100 pod

The oracle comes from the cu128 fork tarball of `tools/dsv41/build.sh` (`llama-dsv41-<commit8>-cu128-sm80-89.tar.gz`),
not an older prebuilt runtime, with its own AVX2 `libggml-cpu` (do not swap in a natively built one). The model is
on the pod's local NVMe (`pod/model-copy.sh` copies it there). The flags are those of the stock llama-server
reference run (14 expert layers in VRAM, 16 threads on the GPU's NUMA node), minus the server-only ones (`--host`, `--port`, `--jinja`,
`--no-reasoning-preserve`: the tool feeds the raw prompt without a chat template):

```
tar -xzf llama-dsv41-<commit8>-cu128-sm80-89.tar.gz -C /root
R=/root/llama-dsv41-<commit8>-cu128-sm80-89
M=/root/models/q2k/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf
NODE=$(nvidia-smi topo -m | awk '/^GPU0/{print $(NF-1)}')
env -u LD_LIBRARY_PATH numactl --cpunodebind="${NODE:-0}" "$R/bin/dsv41-moe-dump" \
  --dump-dir /root/dumps/oracle-dsv41-q2k -m "$M" -f prompt.txt -n 64 -s 42 \
  -lm mmap --n-gpu-layers 99 \
  --override-tensor 'blk\.(2[6-9]|3[0-9])\.ffn_(down|gate|up)_exps\.weight=CUDA0' \
  --override-tensor 'ffn_(down|gate|up)_exps\.weight=CPU' \
  --override-tensor 'engram_embd\.weight=CPU' \
  --flash-attn on --ctx-size 4096 --batch-size 2048 --ubatch-size 1024 \
  --threads 16 --threads-batch 32 --numa numactl
```

`env -u LD_LIBRARY_PATH`: the binaries find the bundled CUDA 12.8 runtime through RUNPATH `$ORIGIN`, and an
`LD_LIBRARY_PATH` from the pod image (CUDA 12.4 libraries) would win over it. `prompt.txt` is this directory's
file. Then copy `/root/dumps/oracle-dsv41-q2k` back and check it with `read_dump.py`.

Disk space, from n_embd 5120, 6 experts used and 40 MoE layers, f32 for `cur`, `weights` and `out`,
i32 for `topk`, every layer with every row:

- per token and layer: `cur` 5120 x 4 = 20,480 B, `out` 20,480 B, `topk` 6 x 4 = 24 B, `weights` 24 B:
  41,008 B
- per token, 40 layers: 1,640,320 B (1.56 MiB)
- `n_prompt + 64` tokens: with `n_prompt` = 32, 96 x 1,640,320 = 157,470,720 B (150 MiB); each extra prompt
  token adds 1.56 MiB
- files: (1 + 64) ubatches x 40 layers x 4 kinds = 10,400, plus a manifest of about 2 MB (the OLMoE manifest
  is 217 KB for 1088 files)

The same per-token formula gives exactly the measured OLMoE dump: (15 layers x 48 rows + 1 layer x 17 rows) x
(2 x 2048 x 4 + 2 x 8 x 4) B = 737 x 16,448 = 12,122,176 B.

## How the hook is compared against it

Build `dsv41-moe-dump` against the fork with the hook patch and runs the command above with the same
prompt file and flags into a second directory, then run:

```
read_dump.py --compare ORACLE_DIR HOOK_DIR
```

The hook must keep the graph names on its tensors: `ffn_moe_logits` (router matmul, `src[1]` = MoE input),
`ffn_moe_topk`, the weights stage and `ffn_moe_out` on the hook's output, or the dump fails naming what is
missing. `--compare` checks both dumps, finds the first generated token where they differ and compares only
the ubatches whose inputs still match (the prompt and the decode calls before that token). Per layer it prints
the relative L2 error `||B - A|| / ||A||` of `out` over all compared tokens and the worst single ubatch, the
same for `cur` and `weights` (the hook's inputs, which come from the unchanged dense graph), and the topk
exact-match count (same expert in the same slot) and same-set count per token. The acceptance bar is relative
L2 of `out` <= 2e-2 in all 40 layers and the same 64 generated token ids. With `--tokens-must-match` the compare
exits 1 unless the generated token ids are identical (`pod/t04a.sh` gates its greedy check on it).

`read_dump.py --expert-placement DIR` reads where a dump's run put the routed experts from its command line
(the manifest's `argv`): for each `blk.N.ffn_{up,gate,down}_exps.weight` the first `--override-tensor` pattern
that regex-searches the name, as llama.cpp assigns buffers. It prints `gpu_layers A-B` (the layers whose experts
went to a CUDA buffer, which must be one range) and one `override PATTERN=BUFFER` line per override it used, and
exits 1 when an expert tensor matches no override or the run used `-cmoe` / `-ncmoe`. On the A100 oracle dump (run-1):
`gpu_layers 26-39`, overrides `blk\.(2[6-9]|3[0-9])\.ffn_(down|gate|up)_exps\.weight=CUDA0` and
`ffn_(down|gate|up)_exps\.weight=CPU`.

### The hook on a dev machine (OLMoE, CPU, measured)

The cu128 fork tarball b1e60020 with `libstrata-moe.so` from Strata 58a01600 unpacked, in the cu128 builder
image with the libcuda stub, the CPU smoke command above, 2026-10-06:

- Hook off (`LLAMA_STRATA_MOE` unset), against `run-a` (the stock CPU smoke dump,
  5210c7c5): relative L2 of `out` 0.000e+00 in all 16 layers, topk 100% exact, the same 16 token ids. The
  patched fork builds the stock graph byte for byte.
- Hook on (`LLAMA_STRATA_MOE=1`) needs `--no-repack` (`-nr`): on this AVX2 VM the Q4_K experts are loaded into
  `CPU_REPACK` buffers, whose layout is not ggml's, and the hook aborts naming the tensor
  (`blk.0.ffn_up_exps.weight is in buffer CPU_REPACK`). `-nr` also changes the stock dense matmuls, so the
  like-for-like baseline is stock with `-nr`: stock `-nr` against run-a (both stock, only the CPU kernel layout
  differs) gives relative L2 of `out` 3.5e-4 in layer 0 and up to 1.17e-1 by layer 7 with the same 16 token ids
  (on this model and prompt any non-bit-exact change grows through the layers: every matmul re-quantizes its
  input, so a perturbation turns into Q8 rounding flips).
- Hook on with `-nr` (dump `t03r2-onnr`) against stock with `-nr` (`t03r2-offnr`): relative L2 of `out`
  0.000e+00 in all 16 layers over 48 positions, `cur` and weights 0.000e+00, topk 100% exact, the same 16
  token ids. The reference runs the stock ggml-cpu kernels, so it is bit-identical to the stock graph; a log
  line at `-v` (`LLAMA_STRATA_MOE=1, routed experts run in libstrata-moe.so`) shows the hook built the op.
  The dump takes 1.7 s wall (8 threads, model load included), stock `-nr` 1.6 s.
- An earlier reference that dequantized the expert rows and accumulated in its own f32 order was 9.2e-5 off in
  layer 0 and reached 1.08e-1 through the layers, with different tokens from generated token 5 on; it was
  replaced, since a reference that drifts cannot tell a kernel bug from the cascade.

`read_dump.py` needs python3 and numpy, for example `$BUILD_ROOT/venv/bin/python`
(`python3 -m venv $BUILD_ROOT/venv && $BUILD_ROOT/venv/bin/pip install numpy`, numpy 2.5.3 when measured);
the builder images have no numpy.

## The GPU tier, the replay harness and hot sets

### The GPU tier of strata-moe

With `STRATA_MOE_HOT=<hot-set json>` and `STRATA_MOE_VRAM_GB=<GiB>` set and a CUDA device present,
`libstrata-moe.so` keeps a fixed hot set of experts per layer in VRAM (`src/dsv41/moe_gpu.cpp`): its own ggml CUDA
backend on device 0 (the fork's `libggml-cuda`, so still one ggml), one VRAM buffer with a slab per layer (the
hot experts' up, gate and down rows in the fork's own types, copied from the fork's host expert tensors at the
layer's first call). Per op the (token, slot) pairs routed to a resident expert are hits: their inputs, slab slots
and weights go through pinned memory to a small graph on that backend (mul_mat_id up and gate, the stock SwiGLU
clamp or plain SwiGLU, mul_mat_id down, times the weights, all ggml-cuda kernels) while the op's ggml-cpu threads
compute the misses with the CPU reference; after one `ggml_backend_synchronize` the combine adds every token's
slots left to right as the stock graph does, taking y * w per slot from the GPU or the CPU
(`src/dsv41/moe_op.h`). The routed experts must still all be in host memory (`-cmoe` or `-ot exps=CPU`, plus `-nr`):
the hot ones are copied from there. Every layer logs one line at its first call, hits or not: its hot experts
copied to VRAM (size and time), or that it has no hot experts and runs on the CPU.

- Without `STRATA_MOE_HOT`, or with it and no CUDA device, every pair runs on the CPU; the library says which on
  stderr once (`strata-moe: STRATA_MOE_HOT unset: every routed expert on the CPU`), and which CPU kernel the misses
  run on (`STRATA_MOE_CPU=fast` by default or `ref`, see `../README.md`; both give the same bits).
- `STRATA_MOE_VRAM_GB` is required with `STRATA_MOE_HOT` (GiB, 2^30 bytes). The slab is sized with ggml-cuda's
  allocator first and aborts above the budget or the free VRAM. A hot set whose types or shapes differ from the
  fork's tensors, an op ggml-cuda cannot run, a malformed file or a CUDA error abort naming the value.
- `STRATA_MOE_PROFILE=1`: per layer, separately for decode (1 token) and batch calls, the calls, tokens, hits,
  misses and the mean submit, gpu (CUDA events around uploads, graph and download), cpu (call start to the end of
  the CPU phases), wait (the submitting thread blocked on the GPU after its CPU work) and op times in ms, printed
  at exit with the per-forward-pass sums.
- A token's hits keep the stock decode shape on the GPU (one input row, ids `[hits, 1]`, so a decode step takes the
  same mmvq kernels); in a batch, tokens with fewer hits are padded, with weight 0 and rows that are not read, on
  the lowest resident slots the token does not use. A token's slots must be distinct: ggml-cuda's `mul_mat_id`
  (`mm_ids_helper`, `ggml/src/ggml-cuda/mmid.cu`) counts every slot of a token in the experts' row offsets but
  gives the token one row per expert, so a repeated expert leaves a row of the compact layout unwritten and the
  batched kernel writes it over other tokens' outputs. Padding by repeating the first hit did exactly that on the
  first A100 run (the hook dump's prompt, from layer 1 on).
- The SwiGLU clamp limit is a value in the op, `dst->op_params[STRATA_MOE_CLAMP_PARAM]` (the last slot, after
  ggml's custom-op params; 0 for plain SwiGLU), written by the fork's hook and by `strata_moe_set_clamp`. A set
  userdata aborts: the first hook (`b1e6002`) passed a pointer to `hparams.swiglu_clamp_exp[il]` of the
  `llm_graph_params` copy on `process_ubatch`'s stack, valid only while the call that built the graph ran, and
  every reused graph read whatever was at that address later. That is why the first A100 run's hook servers aborted
  with `SwiGLU clamp limit -nan` after their warmup. `strata_moe_layer_clamp` returns, per layer, the smallest and
  largest limit its calls computed with.
- First GPU run: an A100-SXM4-80GB pod, 2026-10-06 (`pod/t04a.sh`).

`oracle/hook-clamp-test.sh` (in `verify.sh`, CPU, no model) runs the fork's real hook path with a clamp: the fork's
own `test-llama-archs -a deepseek4 -s 1 -o DIR` writes its tiny random DeepSeek-V4 MoE (4 layers, 2 experts, 1
used, f32; test-llama-archs has no DeepSeek-V4.1 fixture, and deepseek4 goes through the same `build_moe_ffn` hook
with the same clamp), `tiny_moe_model.py` gives each layer its own limit (0.05, 0.06, 0.07, 0.08, small enough to
clamp) and drops the routed expert scale tensors the hook does not compute, and `dsv41-moe-clamp-check` decodes an
8-token prompt and 8 greedy tokens from changing stack depths (7 of the 8 single-token calls reuse the graph),
without and with `LLAMA_STRATA_MOE=1`. On the dev machine (CPU only), cu128 build, 2026-10-06: the limit the op computed with
equals the GGUF value in all 4 layers and all 9 calls, and the logits hash is the same with and without the hook
(`fnv1a64 b1672b134833fe3a`, bit-identical output). The b1e6002 fork with this libstrata-moe aborts on the set
userdata; the b1e6002 fork with its own library, reproduced with a scratch program printing the userdata at each
call, read the right limits while the stack happened to hold them, then 9.97e+17 (the stack fill pattern) in
call 3, then aborted with `SwiGLU clamp limit 0` in call 4.

`dsv41_moe_split_test` (GPU-free, in the ctest set) replaces the GPU with a CPU stand-in that computes each hit
slot with the reference itself, so any resident set must give the all-CPU output bit for bit. DeepSeek dims
(n_embd 5120, n_ff 2304, 384 experts, 12 filled, 6 used, Q2_K gate/up, Q3_K down, clamp 10), resident sets of
0, 3, 6, 9 and 12 of the 12, 1 and 32 tokens: relative L2 0.000e+00 in all 10 cases; negating the stand-in's rows
moves the output (relative L2 0.95 to 2.0) in every case with hits, so the combine reads them. The stand-in also
counts hit tokens with a repeated slot and the test requires none: 0 in all 10 cases; the first padding (repeat
the first hit) fails the 32-token cases with 3, 6 and 9 resident experts.

### dsv41-moe-replay

```
dsv41-moe-replay -m MODEL --dump-dir DIR [-t N_THREADS] [--max-rel-l2 X]
```

Reads the routed expert tensors straight from `MODEL` (mmap of every shard; the dense part is never loaded) and,
per MoE layer and ubatch of the dump, feeds the dump's `cur`, `topk` and `weights` to `strata_moe_forward` as the
fork runs it (one `ggml_custom_4d` on ggml-cpu's threads, `libstrata-moe.so` loaded with `dlopen` from next to the
executable), then compares with the dump's `out`. Every layer gets the stock inputs, so the per-layer error does
not cascade. Per layer: relative L2 `||strata - dump|| / ||dump||` over all ubatches and the worst ubatch, and the
hits and misses the library counted (`strata_moe_layer_counts`). Exit 1 above `--max-rel-l2` (default 2e-2). The
clamp limit is the GGUF's `<arch>.swiglu_clamp_exp`, as the fork reads it, set in the op as the fork sets it. The
library's environment applies.

The bound only means something against a dump computed on the same backend: run-1 computed the experts of layers
0-25 on ggml-cpu and of 26-39 on CUDA0, and on the first A100 run the GPU tier (70 GiB hot set, 100% hits) gave
2.5e-8 to 5.6e-8 in layers 26-39 and 1.7e-2 to 2.2e-2 in layers 0-25, the CPU-only replay the mirror image
(max 2.307e-2). The parity check is therefore the replay with run-1's own placement (`hot_set.py --layers-full`
with `read_dump.py --expert-placement`), bound 1e-6; the two single-tier replays report the cross-backend gap.

On the dev machine, CPU only (the cu128 fork tarball b1e60020 with Strata 34923ce9 unpacked in the cu128 builder image
with the libcuda stub, 8 threads, 2026-10-06), against the stock `-nr` dump:

```
dsv41-moe-replay -m $BUILD_ROOT/testmodels/olmoe-1b-7b-0924-q4_k_m.gguf \
  --dump-dir $BUILD_ROOT/oracle-smoke/t03r2-offnr -t 8 --max-rel-l2 0
```

relative L2 0.000e+00 in all 16 layers (worst ubatch 0.000e+00), 5896 misses and 0 hits, 0.94 s wall with the
model mmap. With `STRATA_MOE_HOT` and `STRATA_MOE_VRAM_GB` set and no GPU it logs the CPU-only mode and gives the
same 0.000e+00. With the fast CPU kernel (Strata 2e0480c): the same with `STRATA_MOE_CPU=fast` and with `ref`, 0.000e+00 in all 16
layers each (per-pass times in `../README.md`).

### hot_set.py

```
hot_set.py (--dump DIR | --counts FILE) [--dump DIR | --counts FILE ...] [--counts-calls decode|all]
           (--gguf MODEL | --layout FILE) --budget-gib GIB [--layers-full A-B] [--ram-budget-gib GIB] -o OUT.json
hot_set.py --propose-ram-budget --cgroup-limit-bytes N --working-set-gib GIB --margin-gib GIB [--numa-free-kib KIB]
```

Counts the activations per (layer, expert) in the `topk` files and fills the budget with a global greedy over all
layers (highest count first; ties to the better rank in the layer, then the lower layer, so experts never seen
fill the layers round-robin). Expert sizes per layer come from the GGUF's tensor headers (`--gguf`, all shards) or
from a layout file (`--layout`; `layout-dsv41-q2k.json` is DeepSeek-V4.1-Flash Q2_K: gate/up Q2_K, down Q3_K,
5120 x 2304, 384 experts, 12,810,240 B per expert, which times 40 x 384 is exactly the 183.25 GiB of routed experts
of the Q2_K GGUF). The packing reserves ggml-cuda's per-tensor alignment and row padding
in full, so a set within the budget here is within it in strata-moe. The GGUF headers are read by
`gguf_header.py` (python3 only); `split.count` may be any GGUF integer type (the DeepSeek Q2_K shards write it as
UINT16, which the first version rejected, so it read shard 1 only). `--layers-full A-B` keeps every expert of
layers A to B in expert id order (so slab slot = expert id) and none of the others, the placement of a stock run
with those layers' experts on the GPU, and fails when it exceeds the budget: on run-1, `--layers-full 26-39` is
5376 experts, 64.138 GiB, 35.0% in-sample hit rate (layers 26-39 at 100%).

`--ram-budget-gib GIB` also packs the warm tier of `STRATA_MOE_RAM_GB`: the same global greedy over the
experts the VRAM set left (next-most-used first, never-seen ones round-robin), each at the size of its slot in
strata-moe's warm buffer, every part (up, gate, down) rounded up to 4096 bytes plus one 4096-byte block
(12,824,576 B per DeepSeek-V4.1 Q2_K expert, against 12,810,240 of data), so a warm list within the budget here
is within it in strata-moe. Every expert in neither list is cold.

`--counts FILE` reads a counts file strata-moe wrote while serving (`STRATA_MOE_COUNTS`,
`src/dsv41/moe_counts.h`, format `strata-moe-counts/1`: per layer `n_expert`, `decode_calls`, `batch_calls`,
`batch_tokens` and the per-expert `decode` and `batch` counts). Every `--counts` and `--dump` source is summed per
(layer, expert); `--counts-calls decode` (the default) takes the files' 1-token calls only, the decode the speed
runs measure, and `all` adds their batch (prompt) calls. A file of another format, a layer whose expert count
differs between files or from the model, and a layer with no activations at all fail naming it. On the OLMoE smoke
(dev machine, CPU only, 2026-10-07) the routing of other text predicts the measured prompt's only when it is varied: a 1 GiB warm
list (281 of 1024 experts, 27%) from one 64-token prose prompt caught 23.6% of the measured prompt's decode pairs,
from the six `tools/dsv41/pod/prompts` at 128 tokens 51.8%, and in-sample (the measured prompt's own dump) 63.0%.

`--propose-ram-budget` sizes a warm tier, with no dumps or model: the cgroup's limit minus
`--working-set-gib` (the dense part's and the engram rows' resident memory, measured on the pod) minus
`--margin-gib`, and with `--numa-free-kib` (the GPU node's MemFree) at most that minus the same margin, rounded down
to whole GiB; it prints the arithmetic and `proposed_ram_budget_gib N`, and fails below 1 GiB. With an A100 SXM pod's
116.415 GiB cgroup, 4.5 GiB and 8 GiB it proposes 103 GiB, capped to 78 GiB by that pod's 86.11 GiB of GPU node
MemFree at its gate. `t04a.sh --speed` calls it for `RAM_RUNS=auto`.

`hot_set_test.py` (in `verify.sh`) writes tiny split GGUF header pairs with `split.count` as each integer type
(UINT8 to INT64) and checks that both shards are read, that a string `split.count` fails naming it, the
`--layers-full` output and budget check (the first hot_set.py, which read shard 1 only, fails 10 of its first 11 checks), and the warm
lists: after one VRAM expert per layer, at 90,112 bytes a warm expert, in count order, and none without
`--ram-budget-gib`; counts files read and merged, decode against all calls, a dump plus a counts file, the
four failures above, and `--propose-ram-budget` with that pod's numbers (103, 78), a limit with no room and missing
inputs.

Output (`strata-moe-hot-set/1`): `budget_gib`, `layers_full` (the range, or null), `bytes`, `layout`, `dumps`,
`counts` (the counts files), `counts_calls` (null without them), `activations`, `hits`,
`expected_hit_rate`, and per layer `layer`, `n_expert`, `n_embd`, `n_ff`, `types` (`up`, `gate`, `down` as
`ggml_type_name`), `expert_bytes`, `activations`, `hits` and `experts` (the resident ids, in slab slot order:
descending count). With `--ram-budget-gib` also `ram_budget_gib`, `warm_bytes`, `warm_hits` and
`expected_warm_rate`, and per layer `warm` (the warm ids, in slot order) and `warm_hits`; strata-moe reads the
warm lists only when `ram_budget_gib` is there.

On the A100 oracle dump run-1 (31 prompt tokens + 64 generated, 570 activations per
layer, 22,800 in all) with `--layout layout-dsv41-q2k.json`, 2026-10-06:

| Budget | Experts (per layer) | Packed | Expected hit rate (in-sample) |
|---:|---|---:|---:|
| 40 GiB | 3352 (59 to 95; most layers 85) | 39.991 GiB | 96.1% (21,919 of 22,800) |
| 60 GiB | 5029 (124 to 146) | 59.998 GiB | 100.0% |
| 70 GiB | 5867 (146 or 147) | 69.996 GiB | 100.0% |

The dump activates only 59 to 146 distinct experts per layer, 4233 in all (50.50 GiB), so from that budget on
every expert it saw is resident and the in-sample rate is 100%: it is an upper bound measured on the prompt the
counts come from. The
decode runs of `pod/t04a.sh` use another prompt and report the real hit rate (`STRATA_MOE_PROFILE=1`).

