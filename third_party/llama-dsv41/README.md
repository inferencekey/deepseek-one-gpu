# llama-dsv41 fork patch

`0001-strata-moe-hook.patch` is the one local commit of the llama.cpp fork checkout (`$FORK_DIR`, branch
`feat/strata-moe-hook`) on the base commit `5210c7c5ed61dddaee6ed476623abf4b63093d16` (branch `runtime/deepseek41`
of the vcruz305 llama.cpp fork with DeepSeek-V4.1 support). It changes `src/llama-graph.cpp` (the hook, below) and
two ggml-cuda kernels (the int32 fix at the end):

- `LLAMA_STRATA_MOE=1`: `build_moe_ffn` builds one `ggml_custom_4d` after the final routing weights with sources
  `cur`, the selected experts, the weights (both viewed to `n_expert_used(il)` slots, as the stock sum adds),
  `ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps`; the layer's SwiGLU clamp limit (0 when it has none) is an f32
  in the op's last `op_params` slot and the userdata is null. A value in the op lives as long as the graph: the
  first version of this commit (`b1e6002`) passed `&hparams.swiglu_clamp_exp[il]`, a pointer into the
  `llm_graph_params` copy on `process_ubatch`'s stack, which dangled once llama.cpp reused the graph from another
  call (the first A100 run aborted with `SwiGLU clamp limit -nan`). The op is `strata_moe_forward` (`include/strata/dsv41/moe.h`), loaded with `dlopen` from
  `libstrata-moe.so` next to `libllama` at first use. The output is named `ffn_moe_out`.
- Unset, empty or `0`: the stock graph, unchanged. Any other value aborts.
- With the hook on, a MoE layer with merged gate_up experts, expert biases or scales, an activation other than
  SwiGLU, weights applied before the FFN, another architecture's SwiGLU clamp or a LoRA adapter on the experts
  aborts naming the layer; an expert tensor outside host memory aborts naming it and its buffer. On the CPU the
  experts land in `CPU_REPACK` buffers by default, so the hook needs `--no-repack`.

`libstrata-moe.so` links the fork's `libggml-base` and `libggml-cpu`, so it is built after the fork and cannot be a
link-time dependency of `libllama`; `dlopen` also keeps the stock build free of it when the hook is off.

The same commit widens two ggml-cuda kernels to 64-bit element indices, unconditionally, since below 2^31
elements nothing changes:

- `ggml/src/ggml-cuda/unary.cu`: `unary_op_kernel` and `unary_cuda` took the element count as an `int`. The
  DeepSeek-V4.1 indexer's scores (`build_indexer_top_k`) are one f32 tensor of n_comp x ubatch x 32 elements, and its
  relu at 2^31 of them (131072 x 512 on a B300; also 16384 x 4096 and 32768 x 2048) wrapped `k` to `INT_MIN`: the
  block count `(INT_MIN + 255) / 256 = -8388607` became 4286578689 blocks through `dim3`, over the 2^31 - 1 grid
  limit, so the launch failed with `CUDA error: invalid argument` in `ggml_cuda_op_relu`. Now `k`, the block count
  and the index are `int64_t`, and the block count is asserted to fit the grid.
- `ggml/src/ggml-cuda/reduce_rows.cuh`: `reduce_rows_f32` (sum_rows, mean) read `x[row * ncols + i]` in `int`, which
  wraps above 2^31 elements (32768 x 4096); now `(int64_t) row * ncols + i`.

The rest of the chain was 64-bit already: `cont` (cpy.cu, `int64_t` indices, its grid asserted), `mul` (bin_bcast,
`size_t` offsets, its grid asserted). `tests/dsv41/indexer_index_test.cpp` checks the launch arithmetic of both
versions at those shapes and that the fork tree has the new code (GPU free; the kernels themselves run on a GPU machine).

Apply on a fresh checkout:

```
git -C <fork> checkout -b feat/strata-moe-hook 5210c7c5ed61dddaee6ed476623abf4b63093d16
git -C <fork> am <Strata>/third_party/llama-dsv41/0001-strata-moe-hook.patch
```

Regenerate after changing the fork commit: `git -C "$FORK_DIR" format-patch -1 HEAD --stdout >
third_party/llama-dsv41/0001-strata-moe-hook.patch`. `tools/dsv41/build-inside.sh` refuses a fork checkout whose
HEAD~1 is not the base or whose HEAD is not this patch (`git patch-id --stable`), and records the base and the
patch's sha256 in the fork tarball's MANIFEST.
