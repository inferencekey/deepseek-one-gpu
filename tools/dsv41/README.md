# DeepSeek-V4.1 build toolchain, MoE hook and expert residency

Builds the llama.cpp fork (`$FORK_DIR`, branch `feat/strata-moe-hook`: base 5210c7c5 plus
`third_party/llama-dsv41/0001-strata-moe-hook.patch`, see its README) and Strata against the fork's ggml, which
carries the DeepSeek-V4.1 ops (`ggml_dsv4_hc_*`, `ggml_lightning_indexer`), in two CUDA flavors, inside the
NVIDIA CUDA devel Ubuntu 22.04 images. No GPU is needed. Both repos are mounted read-only. `build-inside.sh`
refuses a fork checkout whose HEAD~1 is not 5210c7c5 or whose HEAD is not that patch (`git patch-id --stable`).

## Flavors

A CUDA 13 binary does not start on a driver older than 580, and the A100 80GB PCIe pods on RunPod run 570
(570.172.08, seen 2026-10-05), so sm_80/89 are built with CUDA 12.8. sm_120 stays on CUDA 13.0: Strata's
CMake warns that RTX 50 engines built with CUDA 12.8 crashed on long prompts (#220, #224).

| Flavor | CUDA | Minimum driver (Linux) | Archs | GPUs | CUDA runtime libs |
|---|---|---|---|---|---|
| `cu128` | 12.8.1 (nvcc 12.8.93) | >= 570.26 | sm_80, sm_89 | A100, RTX 4090 | bundled in both tarballs: `libcudart.so.12` (12.8.90), `libcublas.so.12`, `libcublasLt.so.12` (12.8.4.1), copied from the builder image |
| `cu130` | 13.0.1 (nvcc 13.0.88) | >= 580.65 | sm_120 (`sm_120a` in the fork) | RTX PRO 6000, RTX 5090 | bundled in both tarballs: `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, copied from the builder image (since 2026-10-08; before, the pod was to provide them) |

Both flavors need `libcuda.so.1` from the pod's driver. Both runtimes are bundled because the pod image used
so far, `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel`, has CUDA 12.4 libraries, older than the 12.8 and 13.0
the binaries were linked against. The pod scripts pick the flavor from the GPU they create (`gpu_flavor` in
`flavors.sh`: the RunPod ids with " Blackwell " or "RTX 50xx" are cu130) and ask RunPod for hosts whose driver
runs it (`POD_CUDA_VERSIONS`: 12.8, 12.9 or 13.0 for cu128, 13.0 for cu130). Drivers seen on RunPod pods: A100 80GB PCIe 570.172.08 (cu128), B300
580.173.02, RTX PRO 6000 595.91.07 (cu130).

The flavor table lives in `flavors.sh`, sourced by `build.sh`, `verify.sh` and `pick-flavor.sh`. The builder
bases in `Dockerfile` and the bare `ubuntu:22.04` in `verify.sh` are pinned by digest.

## Build

```
tools/dsv41/build.sh                     # both flavors in sequence, incremental
tools/dsv41/build.sh --clean             # wipes each built flavor's build dir and tarballs first
CUDA_FLAVOR=12.8 tools/dsv41/build.sh    # one flavor (12.8 or 13.0); verify.sh takes it too
tools/dsv41/verify.sh                    # the build and oracle checks for both flavors, no GPU
```

Overrides: `CUDA_FLAVOR` (default: both), `FORK_DIR` (default `$HOME/llama-dsv41`), `BUILD_ROOT` (default
`$HOME/dsv41-build`, must be outside both repos), `JOBS` (default 8: nvcc takes 2-3 GB per job, the 31 GB dev
machine the defaults were set on fits 8). No ccache.

For each flavor `build.sh` builds the builder image from `Dockerfile` with `--build-arg CUDA_FLAVOR=cu128` or
`cu130` (tagged `strata-dsv41-build:<flavor>-<Dockerfile hash>`, reused while the Dockerfile is unchanged; it
adds git, python3 and pip CMake 3.31.6 / Ninja, since apt's CMake 3.22 is older than Strata's 3.24), then runs
`build-inside.sh` in it as the invoking user (no root-owned files): fork configure and build, Strata configure
and build, packaging. It prints the wall time per flavor and the total.

## Picking the flavor on a pod

```
tools/dsv41/pick-flavor.sh DIST_DIR
```

Run on the pod (it needs `flavors.sh` next to it), with `DIST_DIR` the directory holding the tarballs. It reads
`nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader`, finds the flavor that builds the
GPU's arch, checks the driver against that flavor's minimum and prints `key=value` lines: `flavor`, `cuda`,
`arch`, `driver`, `min_driver`, `fork_tarball`, `strata_tarball`. It exits 1 with the reason when the arch is in
no flavor, the driver is too old for the flavor, the pod mixes GPUs, or `DIST_DIR` does not hold exactly one
tarball per project for the flavor.

## Where things land

| Path | What |
|---|---|
| `$BUILD_ROOT/<flavor>/fork` | fork build tree (`bin/llama-server`, `bin/llama-cli`, `bin/lib*.so`, and `bin/dsv41-moe-dump` from the oracle build) |
| `$BUILD_ROOT/<flavor>/oracle` | CMake build dir of `tools/dsv41/oracle` (`dsv41-moe-dump`, see `oracle/README.md`) |
| `$BUILD_ROOT/<flavor>/strata` | Strata build tree (`strata`, every test and parity binary, `ctest` here) |
| `$BUILD_ROOT/dist/llama-dsv41-<commit8>-cu128-sm80-89.tar.gz` | `bin/` + `MANIFEST` |
| `$BUILD_ROOT/dist/strata-<commit8>-cu128-sm80-89.tar.gz` | `bin/` + `MANIFEST` |
| `$BUILD_ROOT/dist/llama-dsv41-<commit8>-cu130-sm120.tar.gz` | `bin/` + `MANIFEST` |
| `$BUILD_ROOT/dist/strata-<commit8>-cu130-sm120.tar.gz` | `bin/` + `MANIFEST` |

Each tarball unpacks into a directory of the same name. `MANIFEST` lists the repo URL, full commit, branch,
whether tracked files had changes, the flavor, CUDA archs, whether the CUDA runtime is bundled, the exact CMake
flags, nvcc and gcc versions, the build time (UTC) and the sha256 of every file in the tarball. Builds are not
bit-reproducible: do not compare sha256 across rebuilds.

- Fork tarball: `llama-server`, `llama-cli`, `dsv41-moe-dump` (the MoE parity oracle) and `dsv41-moe-replay` (the
  per-layer replay of an oracle dump through `libstrata-moe.so`), both built from this repo's
  `tools/dsv41/oracle` against the fork's libraries (MANIFEST names their Strata commit), and the shared libraries
  they load from the build
  (`libggml{,-base,-cpu,-cuda}.so.0`, `libllama.so.0`, `libllama-common.so.0`, `libmtmd.so.0`,
  `libllama-server-impl.so`, `libllama-cli-impl.so`), plus `libgomp.so.1` (ggml-cpu uses OpenMP and a bare
  ubuntu:22.04 has no libgomp; it is the build's gcc 11 copy), plus `libstrata-moe.so` (Strata's
  `src/dsv41`, the op `LLAMA_STRATA_MOE=1` loads with `dlopen` from next to `libllama`; it links the tarball's
  `libggml-base` and `libggml-cpu`: the CPU reference calls ggml-cpu's own quantize, dot product and SwiGLU kernels,
  which is what makes it bit-identical to the stock path; it is still the one ggml of the process; with the GPU tier it
  also links the tarball's `libggml-cuda` and the CUDA runtime for its GPU tier, see `oracle/README.md`). Its
  MANIFEST adds `fork_base`, `fork_patch` (path, Strata commit, sha256) and the Strata commit of `dsv41-moe-dump`,
  `dsv41-moe-replay` and `libstrata-moe.so`.
- RUNPATH: every binary the build makes, in both tarballs, has RUNPATH `$ORIGIN` (the fork's build tree had
  its absolute `bin/` path), so `bin/` runs from wherever it is unpacked without `LD_LIBRARY_PATH`. An
  `LD_LIBRARY_PATH` set by a pod image still wins over RUNPATH: unset it when it points at another CUDA.
- Strata tarball: all executables of the build (`strata` and the test, parity and bench binaries, so the GPU
  tests can run on a pod). Strata links its ggml statically, so no `.so` from the build is needed.
- CUDA runtime: bundled in the tarballs of both flavors (see the flavor table). `libcuda.so.1`
  is never bundled: the driver supplies it (only the fork links it directly).

## Flags

Fork: `-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF
-DGGML_CCACHE=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_CUDA_ARCHITECTURES=<flavor archs>
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH=$ORIGIN`,
targets `llama-server llama-cli`.

Oracle (`tools/dsv41/oracle`, built after the fork): `-G Ninja -DCMAKE_BUILD_TYPE=Release
-DFORK_DIR=$FORK_DIR -DFORK_BUILD=$BUILD_ROOT/<flavor>/fork`; it sets RUNPATH `$ORIGIN` itself.

Strata: `-G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<flavor archs>
-DSTRATA_GGML_DIR=$FORK_DIR -DSTRATA_PORTABLE=ON -DGGML_CCACHE=OFF -DSTRATA_BUILD_TESTS=ON
-DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_DSV41_FORK_BUILD=$BUILD_ROOT/<flavor>/fork
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH=$ORIGIN`, all targets. `STRATA_DSV41_FORK_BUILD` adds
`libstrata-moe.so`, `dsv41_moe_ref_test`, `dsv41_moe_split_test`, `dsv41_moe_cpu_fast_test` and
`dsv41_moe_cpu_bench` in the build's `dsv41/` (not in the Strata tarball); `build-inside.sh` copies the library into the fork's `bin/`.

## What was turned off, and why

- **Nothing in Strata.** Every Strata target, the ggml-cuda MMQ prompt path included (`strata_mmq`: MMQ
  instances q2_0, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs, q8_0 and `quantize.cu` from the
  fork's tree), compiles against the fork's ggml with Strata's default options, in both flavors. The opt-in
  `STRATA_MMQ_KQUANTS`, `STRATA_Q6K_EXPERTS` and `STRATA_ORCA_Q4KS_MMQ` stay at their default (off) and were
  not tried.
- **Host CPU tuning, both sides.** The fork has `GGML_NATIVE=OFF`. Strata forces `GGML_NATIVE=ON` for its static
  ggml-cpu unless `STRATA_PORTABLE=ON` (CMakeLists.txt, `STRATA_NATIVE_EXPERTS` block), so the build sets it;
  on Linux it does nothing else. Both ggml-cpu builds get ggml's AVX2 baseline (SSE4.2, AVX, AVX2, FMA, F16C,
  BMI2), so a pod CPU without the build machine's extensions does not hit "Illegal instruction".
- **NCCL in the fork** (`GGML_CUDA_NCCL=OFF`). The CUDA devel images ship `libnccl.so.2` (190 MB in 13.0) and
  ggml-cuda links it by default, for the multi-GPU all-reduce. The pods do not have it, and the earlier pod
  builds of the fork were made without it. One GPU per process loses nothing.
- **The prebuilt web UI in the fork** (`LLAMA_USE_PREBUILT_UI=OFF`). Otherwise the build downloads the browser
  UI from a Hugging Face bucket; the shallow clone's build number (b50) has no release there, so it fell back
  to `latest`, which moves. llama-server is built without the embedded UI; its HTTP API is unchanged.

## Notes

- sm_120 is `sm_120a` in the fork: ggml-cuda replaces a plain 120 with `120a-real` (the arch-specific
  Blackwell code, FP4 MMA). Strata's own code has `sm_120`.
- Without a GPU the fork's binaries need a `libcuda.so.1` to load: in the builder `verify.sh` points
  `LD_LIBRARY_PATH` at the toolkit stub (`/usr/local/cuda/lib64/stubs/libcuda.so`), so `--version` prints and
  logs "CUDA driver is a stub library". In the bare ubuntu:22.04 it installs a copy of that stub
  (`$BUILD_ROOT/<flavor>/cudastub/`) as the system `libcuda.so.1`, where a pod's driver puts the real one,
  and runs the binaries of both flavors with no `LD_LIBRARY_PATH`.
- The GPU-free test set `ctest -R '^(conversation_|gguf_reader_test|gguf_split_test|tuning_header_arch_family_test|dsv41_moe_ref_test|dsv41_moe_split_test|dsv41_moe_cpu_fast_test)'`
  is 13 tests: 12 pass and `conversation_snapshot_test` reports Skipped (exit 77, it needs a GPU).
  `dsv41_moe_ref_test` (12 s on the dev machine, 8 threads) is the strata-moe CPU reference against the stock
  mul_mat_id expert graph at DeepSeek-V4.1 dims; it requires bit-identical output (relative L2 0.000e+00 in all 9
  cases, Q2_K, Q3_K, Q4_K, 1 and 32 tokens, with and without the clamp). `dsv41_moe_split_test` (19 s) runs
  the hit/miss split and the combine with a CPU stand-in for the GPU tier and requires the all-CPU output bit for
  bit for every resident set (relative L2 0.000e+00 in all 10 cases per CPU kernel, 20 in all) and
  no hit token with a repeated slot. `dsv41_moe_ref_test` runs with `STRATA_MOE_CPU=ref` (ctest sets it), and
  `dsv41_moe_cpu_fast_test` (42 s) is the fast kernel against the reference (see the CPU kernel section below). verify.sh
  prints the three tables (see `oracle/README.md`) and a 16-expert smoke run of `dsv41_moe_cpu_bench`. The tests
  load the build's own `libstrata-moe.so` (ctest puts the build's `dsv41/` first in `LD_LIBRARY_PATH`, then the
  fork's `bin/`).
- The `strata` tarball's ggml is a second, static copy of the fork's ggml (the fork tarball has the shared one).
  Loading one libggml in one process is the engine's job (`src/dsv41/engine`), not the build's.
- `iq_parity` and its fixtures are not registered: they need python3 with numpy, which the builder image does
  not have (Strata's configure says so).

## Measured

On the dev machine (12 cores, 31 GB RAM, no GPU), `build.sh --clean` from an empty build root (no builder image
cached for either flavor), both flavors in sequence, `-j8`, 2026-10-06. Strata commit ecfcd9bb (the commit the
four tarballs and their MANIFESTs carry; this section was added in the docs-only commit after it), fork
5210c7c5.

| Phase | cu128 (sm_80, sm_89) | cu130 (sm_120) |
|---|---|---|
| fork configure + build (143 ELF images per arch in `libggml-cuda.so`) | 12m00s | 7m56s |
| Strata configure + build (319 steps) | 3m21s | 2m20s |
| builder image, wipe, container start, packaging (gzip) | ~2m08s | ~19s |
| **flavor wall time, as printed by build.sh** | **17m29s (1049 s)** | **10m35s (635 s)** |

**Total `build.sh --clean`: 28m04s (1684 s).** `verify.sh` for both flavors right after: **42 s**, exit 0.
The cu128 packaging is the slow part of its last row: gzip of the bundled `libcublasLt.so.12` (752 MB).

Sizes: tarballs cu128 fork 647 MB, cu128 Strata 809 MB (each bundles the CUDA 12.8 runtime, 869 MB
uncompressed), cu130 fork 47 MB, cu130 Strata 115 MB; build trees cu128 267 MB (fork) and 863 MB (Strata), cu130
212 MB and 494 MB.

Both Strata MANIFESTs say `tracked_changes: yes`: during the build two untracked notes files had uncommitted
edits. Nothing that is compiled differed from ecfcd9bb.

The MoE hook (2026-10-06, dev machine, `-j8`): `build.sh` incremental over the first build's dirs after the hook commit
(libllama, the oracle, Strata reconfigure, strata-moe and its test, both flavors packaged): **2m20s**; `verify.sh`
right after: **1m24s**, exit 0 (the two `dsv41_moe_ref_test` runs take 12 s each). Fork b1e60020 (5210c7c5 plus
the patch), Strata 4b12092b. libstrata-moe.so is 38 KB.

The hook's second version (2026-10-06, dev machine, `-j8`), the reference on ggml-cpu's kernels: `build.sh` incremental after the
strata-moe commit (Strata reconfigure, strata-moe and its test, both flavors packaged): **2m06s**; `verify.sh`
right after: **1m24s**, exit 0. Fork b1e60020 unchanged, Strata 58a01600. libstrata-moe.so is still 38 KB,
NEEDED `libggml-base.so.0` and `libggml-cpu.so.0`, RUNPATH `$ORIGIN`.

The GPU tier (2026-10-06, dev machine, `-j8`): `build.sh` incremental after the GPU tier, replay, hot-set and pod-script commits
(the oracle project with `dsv41-moe-replay`, Strata reconfigure, strata-moe and both dsv41 tests, both flavors
packaged): **2m24s**; `verify.sh` right after: **2m01s**, exit 0. Fork b1e60020 unchanged, Strata 34923ce9.
libstrata-moe.so is 203 KB in both flavors, NEEDED `libggml-cpu.so.0`, `libggml-cuda.so.0`, `libggml-base.so.0`
and `libcudart.so.12` (cu128, bundled) or `libcudart.so.13` (cu130, from the pod), RUNPATH `$ORIGIN`;
`dsv41-moe-replay` is 184 KB. Both flavors compile the GPU tier; it has not run on a GPU yet.

The GPU tier's fixes (2026-10-06, dev machine, `-j8`), after the clamp, slab-fill, tool and pod-script commits: `build.sh`
incremental (libllama with the fork fix, `test-llama-archs` for the first time in cu130, the oracle project with
`dsv41-moe-clamp-check`, strata-moe and both dsv41 tests, both flavors packaged): **2m28s**; `verify.sh` right
after: **2m05s**, exit 0, with `hook-clamp-test.sh`, `hot_set_test.py`, `t04a-lib-test.sh` and the 56-line dry
run passing in both flavors. After the padding commit: `build.sh` **2m17s**, `verify.sh` **2m05s**, exit 0. Fork
4aab6a5a (5210c7c5 plus the patch), Strata d779eeeb. From that cu128 fork tarball on OLMoE, CPU: the replay
against `t03r2-offnr` gives 0.000e+00 in all 16 layers, and the hook-on dump against it 0.000e+00 in all 16 layers
with the same 16 tokens.

## The GPU tier's pod run

`pod/t04a.sh` is the whole A100 session of the GPU tier: parity of the replay against run-1 with run-1's own expert
placement (bound 1e-6), greedy parity of the hook against stock with the same placement (identical 64 tokens),
the stock baseline, decode with the hook at 40, 60 and 70 GiB hot sets, and the single-tier replays as
information. Its header lists what to upload and the steps; `bash tools/dsv41/pod/t04a.sh --dry-run` prints every
command without running anything. It stops itself within `CAP_MIN` (110 min): each command runs under `timeout`
with the time left, a step whose estimate no longer fits is skipped and recorded in `steps.tsv`.
`pick-flavor.sh --fork-only` picks the flavor without a strata tarball (the fork tarball carries
`libstrata-moe.so`).

Every step writes a start line and an end line with its exit status to `steps.log` (and `t04a.log`); the pass/fail
lines go to `verdicts.txt`; the EXIT trap writes the script's exit status and last step as the final line of
`steps.log`, and the ERR trap logs a failing command and its line before `set -e` ends the run. The diagnostics
(`pod/t04a-lib.sh`: snapshots, `env.txt`) write a missing file or tool as absent instead of failing.
`pod/t04a-lib-test.sh` (in `verify.sh`) runs the snapshot against a fake cgroup v1 tree without `memory.peak`, a
v2 tree and an empty one, with `nvidia-smi` failing and missing, and checks the step and exit lines of a run that
dies on an unhandled error.

The A100 pod sessions below are numbered in the order they ran (pod 1 to pod 7).

### Pod 1 (2026-10-06, A100-SXM4-80GB)

Failed, and fixed on the dev machine afterwards:
- every hook `llama-server` aborted after its warmup with `SwiGLU clamp limit -nan`: the fork hook passed the
  limit as a pointer into the graph-build objects, which dangled in reused graphs (`oracle/README.md`, the GPU
  tier; the limit is now a value in the op, tested by `oracle/hook-clamp-test.sh`);
- the replay bound of 2e-2 against run-1 cannot pass for a single tier (cross-backend gap, `oracle/README.md`,
  dsv41-moe-replay): parity is now against run-1's own placement;
- the hook's greedy dump diverged from stock in the prompt already: layer 0 matched at the CPU/CUDA gap
  (1.85e-2), layer 1 was 1.365. In that layer one token had 5 hits against 6 for the others, and its padding
  repeated an expert, which ggml-cuda's batched `mul_mat_id` cannot take: tokens 0, 2 and 6 came out wrong (token
  2 with identical routing, out norm 27.1 against 4.2), and the error cascaded. Layer 0 had no padded token. The
  padding now uses distinct slots (`oracle/README.md`, the GPU tier; checked by `dsv41_moe_split_test`);
- the script died silently on the cgroup v1 pod (`snapshot` failing on the absent `memory.peak` under `set -e`),
  and `hot_set.py --gguf` read one shard only (`split.count` is UINT16 there).

The 40 GiB server logged no copy line for layers 5, 6 and 31: the slab was filled at a layer's first call with a
hit, and the warmup decode had no hit in those layers (85 of 384 experts hot); the next call aborted on the clamp
before they had one. The 60 and 70 GiB servers (124 to 147 hot experts per layer) logged 38 and 40 layers. Not a
copy bug: the slab is now filled at the layer's first call, hits or not, and every layer logs one line.

Measured there (the same pod): stock baseline 18.92 tok/s at 120 tokens and 19.15 at 480 (14 expert layers in
VRAM, peak 71,623 MiB); stock with every routed expert in host memory and `-nr`: 12.16 and 11.42 tok/s. The GPU
replay at 70 GiB (100% hits) took, per forward pass of decode calls (sum of the per-layer means): gpu 11.5 ms,
cpu 6.5 ms, wait 7.2 ms, op 15.5 ms.

Note for the CPU kernel work (next section): in the hook's greedy run (60 GiB hot set, 52.2% decode hit rate) the CPU
miss path took 426.4 ms per forward pass against 6.7 ms on the GPU. The misses run the bit-exact hook reference
(one-row vec_dot per expert row, a work item per 16 rows), which is far from a fast CPU expert pool; the speed of
the miss path is the CPU kernel's job, measured against the same-pod baseline.

### Pod 2 and the pod 3 speed run (2026-10-06)

Pod 2, A100-SXM4-80GB, EPYC 7713, cgroup v1, CPU quota 27.2
cores, memory limit 124,999,999,488 B (116.4 GiB, page cache included), GPU NUMA node 1, fork 4aab6a5a with
Strata 9fb253d, about 1.94 USD. The pod 1 fixes held (no silent death, every shard read, no clamp abort):
`replay-oracle` PASS, max rel L2 5.935e-08 (bound 1e-6, 35.0% hits); stock with the oracle flags and `-nr`
matched run-1 bit for bit over 64 tokens; the hook's greedy run matched stock in 5 of 64 tokens (first difference
at generated token 5). There, layer 0's MoE out matched to 4.977e-08 but layer 1's MoE input `cur` already
differed by 1.412e-03, so the first difference enters outside the routed-expert op (open at the time; the engine's parity work covers it).
The host was bad: stock decoded at 2.04 / 1.91 tok/s (120 / 480 tokens) against 18.92 / 19.15 on pod 1, the
server read 390 MB/s from disk with 71.3M major page faults (331M by the end) while the cgroup page cache sat at
the limit, the GPU's NUMA node had 2.5 to 3 GB free (the other node 151 GB), load average about 25. Its hook
profile is the source of the 260-520 ms per forward pass: hook-40 / 60 / 70 GiB at 38.7 / 53.6 / 61.7% decode
hits, cpu 520.064 / 462.008 / 261.182 ms, gpu 5.504 / 6.800 / 7.277 ms. Not valid as a speed measurement.

Pod 3, A100-SXM4-80GB, EPYC 7713, cgroup v2, quota 27.2 cores,
`memory.max` 124,999,999,488 B, GPU NUMA node 0 (102 to 190 GB free), host load average 32 to 89 (neighbours),
the throwaway `t04a-speed.sh` (now `t04a.sh --speed`), about 0.99 USD. Health gate PASS (stock 120-token decode
20.36 tok/s). Tok/s are llama-server `predicted_per_second`, measured run after a warm run:

| Run | 120 tok/s | 480 tok/s | decode hits | per forward pass: submit / gpu / cpu / wait / op (ms) | peak VRAM |
|---|---:|---:|---:|---|---:|
| stock, 14 expert layers (26-39) in VRAM | 20.36 | 23.47 | | | 72,043 MiB |
| hook-40 (the hook's reference on the misses) | 16.49 | 15.95 | 38.7% | 2.687 / 5.164 / 83.609 / 0.459 / 85.633 | 46,067 MiB |
| hook-60 | 15.29 | 18.50 | 53.6% | 2.716 / 6.176 / 78.456 / 0.617 / 80.528 | 66,555 MiB |
| hook-70 | 17.61 | 23.58 | 61.7% | 2.507 / 6.531 / 50.659 / 0.729 / 52.663 | 77,209 MiB |
| stock, every routed expert in host memory, `-nr` | 5.58 | 16.43 | | | 5,943 MiB |

The per-pass means cover all four requests of a server, warm ones included, and the page cache still thrashed
(cgroup `pgmajfault` 16.5M to 89.6M during the benches, 40k to 60k per second; `file` 123.3 to 124.3 GB against
the limit, `anon` 0.22 to 0.54 GB): the 264 GB model is mmap'd into a 116.4 GiB cgroup, so missed experts are
partly read back from the NVMe. On this healthy host the miss path took 50.7 to 83.6 ms per pass, which is 22 to
23 GB/s of missed experts (40 layers x 6 slots x the miss rate x 12,810,240 B: 1.18 GB at 70 GiB, 1.88 GB at 40
GiB), not the 10x-off 260-520 ms of pod 2.

## strata-moe's own CPU kernel (`STRATA_MOE_CPU`)

`src/dsv41/cpu_kernel_avx2.cpp` computes the routed experts' rows for the misses, selected at run time by
`STRATA_MOE_CPU=fast` (the default, also when unset) or `ref` (the hook's reference, kept for the tests); any other
value aborts. Design:

- **Bit for bit with the reference.** For q2_K, q3_K, q4_K and q6_K (DeepSeek-V4.1 gate/up and down; the OLMoE
  smoke model's q4_K and q6_K) the kernels are the `__AVX2__` branches of the fork's
  `ggml-cpu/arch/x86/quants.c` with the loop over tokens moved inside the loop over blocks (Strata's precedent,
  `src/kernels/cpu/kq_avx2.cpp`): the weight-side unpacking (bits, 4/6-bit scales and mins, fp16 scales) is done
  once per block, every token routed to the expert then runs ggml's integer chain and float operations in
  ggml's order. The activations are ggml's Q8_K (`from_float`), quantized once per token and once per (token,
  slot) for h; SwiGLU with or without the clamp and the combine are the reference's. So the output equals the
  reference's, which equals stock ggml-cpu's.
- **One row per pass, all of an expert's tokens.** A work item is 64 rows of one expert for all its missed
  pairs (decode: 1 token; a prompt ubatch: every token routed there), up to 8 tokens per pass over a row. Tiles of
  4 rows through the same block index measured slower (below), so a pass streams one row front to back.
- **Threads: the op's own ggml-cpu threads**, not a second pool: llama.cpp already starts them on the GPU's NUMA
  node (`numactl --cpunodebind` in the pod scripts) and a second pool would compete with them for the pod's
  27.2-core quota. Items are taken from atomic counters on their own cache lines; the barrier's count and
  generation words sit on separate lines (the waiters spin on the generation only); the per-pair outputs start on
  cache lines and an item writes whole lines, so no two threads write one line.
- **No per-call allocation.** The op's calls are pooled and reset (every buffer keeps its capacity), the misses
  are grouped per expert in one flat array (no vector of vectors), the SwiGLU scratch is per thread.
- **Fails loudly.** `fast` aborts naming the tensor on another type (`strata-moe: STRATA_MOE_CPU=fast has no kernel
  for blk.0.ffn_up_exps.weight (f32; it has q2_K, q3_K, q4_K, q6_K); STRATA_MOE_CPU=ref runs any type`, the f32
  tiny model of `hook-clamp-test.sh`, which therefore sets `ref`), and on a CPU without AVX2, FMA and F16C
  (checked at run time; the kernel file alone is built with `-mavx2 -mfma -mf16c`).
- **Profile.** `STRATA_MOE_PROFILE=1` adds a `sync` column: the mean time a thread waits at the barriers between
  the CPU phases, inside `cpu`. A slow or descheduled thread shows up there.

### Measured on the dev machine (2026-10-06)

AMD Ryzen 7 5700X (Zen 3, AVX2 + FMA, no AVX-512, 32 MB L3), 12 vCPUs, 31 GB RAM shared with other work (load
average 2 to 6 during the runs from other processes), the cu128 builder image, Strata 2e0480c, fork 4aab6a5a.
`build.sh` incremental over the GPU tier's dirs: 2m20s; `verify.sh` right after: 3m40s, exit 0 (13 tests per flavor,
12 passed, 1 skipped; both dry runs; the bench smoke run).

`dsv41_moe_cpu_fast_test` (both flavors, 42 s): the 20 row cases (each type, rows [1, 66) of 67, 1, 3, 8, 9 and
17 tokens) give 0 results differing from ggml-cpu's `vec_dot` and no row outside the range written; the 15 op
cases (DeepSeek-V4.1 dims: q2_K/q3_K clamp 10 and q3_K/q2_K; OLMoE dims: q4_K/q4_K, q4_K/q6_K, q6_K/q4_K clamp 10;
1, 5 and 32 tokens each) give relative L2 0.000e+00 and 0 differing elements, fast against ref. The acceptance bound
is 1e-5; the test requires 0.

`dsv41_moe_cpu_bench` (192 experts, 2.46 GB, 6 used, clamp 10, 10 timed calls after 2 warm-up calls, every pair a
miss), first of two runs; read bandwidth 23.42 GB/s at 1 thread, 30.10 at 2, 28.24 at 8, best 30.10 GB/s:

| tokens | threads | ref ms | fast ms | fast GB/s | fast % of best read | fast / ref | same bits |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 6.872 | 6.879 | 11.17 | 37.1% | 1.00x | yes |
| 1 | 2 | 4.066 | 4.069 | 18.89 | 62.8% | 1.00x | yes |
| 1 | 4 | 3.021 | 2.879 | 26.70 | 88.7% | 1.05x | yes |
| 1 | 6 | 3.013 | 3.355 | 22.91 | 76.1% | 0.90x | yes |
| 1 | **8** | 3.410 | **3.400** | **22.61** | **75.1%** | **1.00x** | yes |
| 1 | 10 | 3.315 | 4.398 | 17.48 | 58.1% | 0.75x | yes |
| 1 | 12 | 6.289 | 6.053 | 12.70 | 42.2% | 1.04x | yes |
| 32 | 1 | 187.644 | 204.487 | 7.66 | 25.4% | 0.92x | yes |
| 32 | 2 | 129.990 | 111.496 | 14.04 | 46.6% | 1.17x | yes |
| 32 | 4 | 74.221 | 75.311 | 20.79 | 69.1% | 0.99x | yes |
| 32 | 8 | 64.738 | 64.505 | 24.27 | 80.6% | 1.00x | yes |
| 32 | 12 | 90.999 | 84.740 | 18.47 | 61.4% | 1.07x | yes |

The second run (best read 28.23 GB/s): 1 token 8 threads ref 3.041 ms, fast 3.002 ms, 25.60 GB/s = 90.7% of the
read bandwidth, 1.01x; 1 token 4 threads fast 2.911 ms (93.5%), 1.12x; 32 tokens 8 threads 61.199 against 65.504
ms, 1.07x. A 32-token call reads 122 distinct experts of 192 (1.6 tokens per expert), so it is a bandwidth case
too. Prompt-sized ubatches (one run, best read 26.97 GB/s, 3 timed calls): 256 tokens 389.2 / 346.4 ms (8
threads, 1.12x) and 384.0 / 262.9 ms (12 threads, 1.46x); 1024 tokens 1059.9 / 849.3 ms (8 threads, 1.25x) and
1088.8 / 889.4 ms (12 threads, 1.22x), 192 experts read per call, compute-bound.

Reading: on this machine the reference is not far from the memory bound: at 1 token and 4 to 8 threads it already reads
75 to 97% of the measured read bandwidth, and one thread alone does 11 GB/s. Both kernels run the same per-token
integer chain (that is what makes them bit-identical), so per core they are equal, and fast gains only where the
weight-side unpacking is shared by several tokens: 1.1 to 1.5x on prompt-sized ubatches. At 10 and 12 threads,
with the machine's other load, both lose up to 2x and vary from run to run (threads waiting at barriers for a
descheduled one).

Rejected, measured the same way: tiles of 4 rows through the same block index, 1 token, 1 thread: 20.2 ms against
ref's 8.2 (16 experts), 14.7 against 7.1 with a software prefetch 6 blocks ahead in each row stream (48 experts),
where one row per pass gives 8.1 against 7.9; 32 tokens, 8 threads, 48 experts: 74.4 ms with 4-row tiles against
45.7 with one row (ref 58).

OLMoE (`dsv41-moe-replay` against `oracle-smoke/t03r2-offnr`, 8 threads, CPU, the cu128 fork tarball's tools with
`libstrata-moe.so` of Strata 2e0480c): `STRATA_MOE_CPU=fast` gives relative L2 0.000e+00 in all 16 layers (worst
ubatch 0.000e+00), PASS at bound 0, as `ref` does. Per forward pass (sum of the layer means), three alternating
runs each at load average 2.4 to 3.1: decode cpu fast 33.2 / 24.0 / 20.9 ms (sync 4.2 / 6.7 / 3.8), ref
23.5 / 23.9 / 30.6 ms (sync 5.0 / 5.0 / 8.3); the 32-token prompt ubatch (one call per layer, which also faults
the mmap'd experts in; the first run's 977.6 ms had a cold page cache) fast 192.4 / 210.0 ms, ref 312.8 / 315.8 /
361.8 ms. With 3 unrelated builds on the machine (load average 6 to 10) the same runs swing 2x for
either kernel (decode 27 to 92 ms, sync up to 52 ms per pass), so per-pass numbers here need a quiet machine.

OLMoE greedy, hook on with `-nr` (`dsv41-moe-dump ... -n 16 -ngl 0 -c 256 -t 8 -s 42 -nr`, `LLAMA_STRATA_MOE=1`)
against stock with `-nr` (`t03r2-offnr`), `read_dump.py --compare`: with `STRATA_MOE_CPU=fast` the 16 generated
tokens are identical (16 of 16), relative L2 of `out` 0.000e+00 in all 16 layers over 48 positions, `cur` and
weights 0.000e+00, topk 100% exact; with `STRATA_MOE_CPU=ref` the same 0.000e+00 everywhere and 16 of 16 tokens.

### The pod run

`t04a.sh --speed` runs the speed table of both kernels against the same pod's stock baseline, with the health
gate of the pod 3 run. Upload the payload of the header of `pod/t04a.sh` (the fork tarball with this build's
`libstrata-moe.so`, `run-1/`, a copy of `tools/dsv41`), then on the pod:

```
bash tools/dsv41/pod/t04a.sh --dry-run --speed            # anywhere first: 60 lines, the steps below
CAP_MIN=100 bash /root/t04a/dsv41/pod/t04a.sh --speed      # on the pod; HEALTH_MIN=15, CPU_KERNELS="fast ref"
```

Steps: setup, the 40, 60 and 70 GiB hot sets, the health line and the gated stock baseline (exit 3 under
`HEALTH_MIN` tok/s at 120 tokens), then `hook-40-fast`, `hook-40-ref`, `hook-60-fast`, `hook-60-ref`,
`hook-70-fast`, `hook-70-ref` (each 120 and 480 tokens, warm then measured, profile with `sync` at exit), then
`stock-hostexps`. `bench.tsv` has the tok/s, each `hook-*.log` ends with its profile. A step that would not fit
under `CAP_MIN` is skipped and recorded; `CPU_KERNELS=fast` halves the hook runs.

## Three-tier expert residency (`STRATA_MOE_RAM_GB`)

Why: on the healthy pod 3 A100 the CPU miss path took 50.7 to 83.6 ms per forward pass with 40k to 60k major page
faults per second. The 246 GiB model is mmapped into a 116.4 GiB cgroup, so missed experts came from NVMe through
a page cache that kept evicting them; the CPU kernel already reads at 75 to 91% of the dev machine's memory bandwidth.
So strata-moe now owns where every routed expert lives (`src/dsv41/moe_residency.h`, `.cpp` for the why of each
choice):

- **VRAM**: the hot set, the GPU tier above, unchanged.
- **Warm**: the hot set's warm list (`hot_set.py --ram-budget-gib`, `oracle/README.md`), one buffer of
  `STRATA_MOE_RAM_GB` GiB held as `STRATA_MOE_RAM_MODE` says (below: CUDA pinned by default with a GPU, `mlock`ed
  without one), filled at each layer's first call with the same O_DIRECT reads as the cold tier (copying from the
  fork's mapping would pull the whole warm set through the page cache).
- **Cold**: every other expert, read during each call that routes to it with O_DIRECT `pread` from its GGUF
  shard, whose path and offset come from the fork's mapping in `/proc/self/maps` (the first block of every expert
  tensor is compared with the mapping). 8 reader threads (`STRATA_MOE_COLD_THREADS`), not io_uring, which Docker's
  default seccomp profile blocks. Reads are whole 4096-byte blocks into slots of round_up(len, 4096) + 4096, so
  rows keep their mapped alignment. Cold experts come in rounds of 16 (`STRATA_MOE_COLD_ROUND`) over a two-round
  staging: the CPU starts on the warm experts while round 0 is read, round r + 1 is read while round r computes.
- After a layer's first call the fork's pages of its routed experts are dropped (`MADV_DONTNEED`, then
  `POSIX_FADV_DONTNEED`, only pages wholly inside the expert tensors). Safe: the mapping is read-only and
  file-backed, and with the hook nothing else reads the routed experts.
- Same bits: warm and cold experts go through the call's CPU kernel on the same inputs; only where the rows are
  read from changes.
- Fails loudly: a warm tier over the budget, a warm-tier allocation refused (the message names the mode, the size
  and the limit of that mode, below), experts not mmapped from a file (`--no-mmap`), a layer the hot set lacks, a
  short or failed read (file, offset, errno).
- Profile (`STRATA_MOE_PROFILE=1`): major page faults per token (getrusage, whole process, from the first call on)
  for both modes, and with the residency a tier table per layer: warm and cold pairs, MiB per call, cold read ms
  (first read queued to last done), GB/s, and `cwait` (mean time a thread waited for cold bytes).
- Unset `STRATA_MOE_RAM_GB`: as before the residency. A hot set without VRAM experts makes no GPU tier and needs no
  `STRATA_MOE_VRAM_GB`.

`tests/dsv41/moe_residency_test.cpp` (ctest, in `verify.sh`): the cold reader on a file of 3 MiB + 1234 bytes
(unaligned ranges, ranges ending at EOF, 8 KiB short reads after an EINTR: every call 4096-aligned, the bytes equal
to the mapping); aborts on a short read before the wanted bytes, an unaligned buffer, a non-file range, a warm tier
over the budget; the tier map; and the op with 4 warm and 12 cold experts of q4_K/q6_K
weights at unaligned file offsets against the mapped tensors, bit for bit, ref and fast, 1/5/32 tokens, rounds of
2 (up to 6 rounds through the 2-round staging) and 16, each cold expert read once per call and no warm one. The RAM
modes: below.

### Measured on the dev machine (2026-10-06)

Ryzen 7 5700X, 12 vCPUs, 31 GB RAM shared, disk a QEMU virtual disk on the host's SSD (its reads may include the
hypervisor's cache, so the O_DIRECT baseline on the same file is the comparison), load average 2.5 to 4.9. Strata
8ff6783, fork 4aab6a5a, cu128. `build.sh` incremental: 2m40s, exit 0; `verify.sh` both flavors: 3m41s, exit 0
(14 tests per flavor, `dsv41_moe_residency_test` PASS, `hot_set_test.py` PASS, `t04a-lib-test.sh` PASS, the
`--speed` dry run 67 lines with `hot-set-70-ram40` and `hook-70-ram40`).

`oracle/residency-pressure-test.sh $BUILD_ROOT/t04c/run2`: OLMoE (4.2 GB) in the cu128 builder image under
`docker run --memory 3g --memory-swap 3g`, 16 greedy tokens, 8 threads, `-nr`, model dropped from the page cache
before each run; the hot set has no VRAM experts and a 1 GiB warm tier (281 experts, 0.998 GiB, 67.3% in-sample
warm rate from `t03r2-offnr`):

| Run (same 3 GB limit) | Tokens vs stock -nr unlimited | rel L2 of out, all 16 layers | Major faults per decode token | per prompt token | Decode op per pass | Wall |
|---|---|---:|---:|---:|---:|---:|
| mmap-only (the hook, `STRATA_MOE_RAM_GB` unset) | 16 of 16 identical | 0.000e+00 | 790.9 | 394.6 | 110.6 ms | 8.4 s |
| tiered (warm 1 GiB + cold O_DIRECT) | 16 of 16 identical | 0.000e+00 | 1.6 | 16.3 | 86.3 ms | 7.3 s |

Major faults per decode token are 494x lower, per prompt token 24x (the prompt's faults are the dense weights,
evicted by the load's `MAP_POPULATE` under the limit). `memory.peak` reaches the 3 GiB limit in both runs (it
counts the page cache). Tiers in decode: 63.0% of pairs warm, 37.0% cold, 173.0 MiB read per forward pass in
73.1 ms (2.48 GB/s), threads waiting 54.4 ms of it; in the prompt 1767.4 MiB per pass at 3.30 GB/s. The warm fill:
about 26 ms per layer of 70 MiB (2.8 to 3.0 GB/s). O_DIRECT baseline of the same file right after: `dd`, 1 MiB
blocks, 0.89 GB/s with one stream and 3.29 GB/s with 8; `fio --rw=randread --direct=1 --ioengine=psync` (in a
ubuntu:22.04 container, 10 s each): 1 MiB blocks 953 MB/s with 1 job, 2,467 MB/s with 8; 4 MiB blocks 2,184 MB/s
with 1 job, 1,897 MB/s with 8. So the cold tier reads at the disk's parallel O_DIRECT rate, and on this machine a decode
pass waits mostly on the disk. The first run (`t04c/run1`, load average 6.7) gave the same tokens and 0.000e+00,
772.8 against 0.2 faults per decode token and 2.22 GB/s cold reads; its times are not comparable (load).

### The pod run

`t04a.sh --speed` adds per `RAM_RUNS` pair (default `70:40`) the hot set `hot-70-ram40.json` (`hot_set.py
--budget-gib 70 --ram-budget-gib 40`; on run-1 with the layout: 5867 VRAM experts, 3349 warm, 40.000 GiB, every
other of the 15,360 experts cold) and `hook-70-ram40` (fast kernel, `STRATA_MOE_RAM_GB=40`, after `ulimit -l
unlimited`, dropped after pod 4, below), 120 and 480 tokens, profile at exit. 70 + 40 leaves the dense part, the engram page cache and the
staging (410 MiB at 16-expert rounds) about 36 GiB of the 116.4 GiB cgroup. Every `bench.tsv` line now ends with
the server's major faults during the request and per second, so the stock baseline, `hook-70-fast` and
`hook-70-ram40` can be compared on the same pod.

### The pod 4 check and the warm tier's RAM modes (`STRATA_MOE_RAM_MODE`)

Pod 4 (2026-10-06, A100-SXM4-80GB, RunPod EUR-IS-1, driver 580.159.04, cgroup v1 116.4 GiB, 1.19 USD): `hook-70-ram40` never ran. The container had
`RLIMIT_MEMLOCK` 8192 KiB soft and hard and no `CAP_IPC_LOCK`, `ulimit -l unlimited` was refused, and the library
aborted on the warm tier as designed: `mlock of the 40.000 GiB warm tier (42949505024 bytes) failed: Cannot allocate
memory; RLIMIT_MEMLOCK is 0.008 GiB (8388608 bytes)`, server exit 134. The other runs on that pod, fast kernel,
tok/s at 120 / 480 tokens: stock 17.64 / 21.45, hook-40 13.88 / 13.03, hook-60 13.77 / 10.38, hook-70
10.09 / 13.90, stock with every expert on the host 10.90 / 11.14; major faults 15k to 40k per second, decode op 76
to 94 ms per pass, of which the CPU part 74 to 92 ms. All below pod 3 (stock 20.36 / 23.47, hook-70 17.61 / 23.58): the
GPU was on NUMA node 1, which had 0.67 to 41.5 GB free during the run (node 0: 164 to 198 GB), so these numbers
describe that host, not the code. The VRAM hit rates equal pod 2's and pod 3's (38.7 / 53.6 / 61.7%).

So the warm tier no longer depends on `RLIMIT_MEMLOCK` on the pods (`src/dsv41/moe_residency.h`, `RamMode`):

| `STRATA_MOE_RAM_MODE` | Memory | Default | Limit it hits, named in the abort |
|---|---|---|---|
| `pinned` | `cudaHostAlloc`: the driver pins the pages, outside `RLIMIT_MEMLOCK` (an earlier vLLM-EXL3 host-offload experiment pinned 144 GiB this way on RunPod) | with a CUDA device | the memory cgroup |
| `locked` | anonymous memory, `mlock` | without a CUDA device | `RLIMIT_MEMLOCK` (and the cgroup) |
| `plain` | anonymous memory, `MADV_HUGEPAGE`, pre-faulted with `MADV_POPULATE_WRITE` (Linux 5.14+), not locked: resident while nothing swaps it (the pods have no swap) and the cgroup does not OOM | only when set | the memory cgroup |

The library logs the mode and what chose it (`strata-moe: STRATA_MOE_RAM_MODE=pinned (the default with a CUDA
device)`), then the allocation with `RLIMIT_MEMLOCK` and the cgroup limit. Every mode aborts up front when the warm
tier does not fit under the cgroup's limit, and on a failed allocation with the mode, the size, the cause and its
limit; an unknown mode, `pinned` without a CUDA device and `STRATA_MOE_RAM_MODE` without `STRATA_MOE_RAM_GB` abort
too. The O_DIRECT fill and the cold tier are unchanged; `cudaHostAlloc` promises less than 4096-byte alignment, so
the pinned buffer is one block larger and starts at the next 4096 boundary. The numerics do not depend on the mode:
only where the warm bytes live changes.

Dev machine checks (2026-10-06, the machine above, load average 1.6 to 3.2, Strata 1a1aaf8 (the tarballs' MANIFEST),
fork 4aab6a5a, both flavors built in 2m13s and `verify.sh` exit 0 in 3m39s):

- `dsv41_moe_residency_test` adds: `choose_ram_mode` for every value with and without a device (unset: pinned with,
  locked without; unknown value and pinned without a device: errors); `warm_alloc_error` per mode with pod 4's limits
  (each names the size in GiB and bytes and its limit); the op bit for bit against the mapped tensors with the warm
  tier locked, plain and pinned (pinned through a stand-in for `cudaHostAlloc` that returns memory 64 bytes off the
  4096 alignment, allocated and freed once); under `RLIMIT_MEMLOCK` 4 KiB plain and pinned allocate and fill, locked
  aborts; a failing pinned allocator aborts naming the size, the cause and the cgroup limit.
- `oracle/residency-pressure-test.sh $BUILD_ROOT/t04c-fix/pressure2`, OLMoE under `--memory 3g`, 1 GiB warm
  tier (281 experts, 0.998 GiB), cu128:

| Run | `--ulimit memlock` | Tokens vs stock -nr unlimited | rel L2, all layers | Major faults per decode token | per prompt token | Decode op per pass |
|---|---|---|---:|---:|---:|---:|
| mmap-only | -1 | 16 of 16 identical | 0.000e+00 | 846.8 | 321.5 | 102.1 ms |
| tiered-locked (`locked`) | -1 | 16 of 16 identical | 0.000e+00 | 0.0 | 17.1 | 83.0 ms |
| tiered-plain (`plain`) | 8388608 | 16 of 16 identical | 0.000e+00 | 2.1 | 21.8 | 95.8 ms |
| locked-memlock-8m (unset: `locked`, the default without a GPU) | 8388608 | aborts, exit 134, as required | | | | |

  The abort: `STRATA_MOE_RAM_MODE=locked: mlock of the 0.998 GiB warm tier (1071427584 bytes) failed: Cannot
  allocate memory; RLIMIT_MEMLOCK is 0.008 GiB (8388608 bytes) (...), and the memory cgroup, whose limit is 3.000
  GiB (3221225472 bytes) (memory.max), must hold it`. Warm tier allocation: locked 510 ms, plain 458 ms (with the
  pre-fault). Decode cold reads 2.58 GB/s (locked) and 2.25 GB/s (plain) against `dd` O_DIRECT 3.20 GB/s with 8
  streams right after (1.06 GB/s with one). Run 1 (`t04c-fix/pressure1`, the same source before the commits, load
  average 1.6 to 2.2) gave the same verdicts: 768.8 / 2.2 / 0.0 major faults per decode token, decode op per pass
  102.7 / 84.0 / 95.9 ms for mmap-only / locked / plain. In both runs plain is 12 to 15% slower per decode pass
  than locked; its cold wait is 7 to 16 ms longer (52 against 59 ms here, 50 against 66 in run 1), and this machine's
  shared disk does not tell whether the rest is the mode. The pods use pinned, not plain.
- `pinned`, the pod default, cannot run here (no GPU). Checked here: both flavors build it, `verify.sh` checks that
  libstrata-moe.so imports `cudaHostAlloc` and `cudaFreeHost`, and the unit test checks that it is the default with
  a device and that the residency handles its alignment. The pod run checks the real allocation: its mode line is in
  `verdicts.txt`.

`t04a.sh --speed` after pod 4: no `ulimit -l` (the residency runs start like the others; `RAM_MODE` sets
`STRATA_MOE_RAM_MODE`, unset leaves the library default, pinned); `env.txt` and the health line record
`RLIMIT_MEMLOCK` (soft, hard) and `CapEff`; each residency run's mode lines go to `verdicts.txt`; `SPEED_BUDGETS`
(default `40 60 70`) picks the hook runs' hot sets and `EXTRA=0` skips `stock-hostexps`; and a NUMA gate right after
the environment snapshot records every node's MemFree and exits 3 when the GPU's node has less than the largest warm
tier of `RAM_RUNS` plus 8 GiB (48 GiB for `70:40`), before the model copy, so a host like pod 4's costs a few minutes.
A 75-minute session of stock, hook-70 and hook-70-ram40 only:

    SPEED_BUDGETS=70 RAM_RUNS=70:40 CPU_KERNELS=fast EXTRA=0 CAP_MIN=60 bash /root/t04a/dsv41/pod/t04a.sh --speed

Its dry run (in `verify.sh`): env, numa-gate, pick-flavor, unpack, manifest-sha256, numactl, model-copy,
snapshot-setup, hot-set-70, hot-set-70-ram40, baseline, hook-70-fast, hook-70-ram40, snapshot-end.

## A profiled warm tier, the page cache for the rest (`STRATA_MOE_COLD`, `STRATA_MOE_COUNTS`)

Why (pod 5, A100 SXM, 2026-10-06): tok/s at 120 / 480 tokens were stock
19.52 / 19.61, hook-70 20.71 / 18.05 and hook-70 with a 40 GiB pinned warm tier and O_DIRECT cold 4.22 / 4.40. The
residency held (0 major faults) but was configured wrong: the warm list had no signal (`hot_set.py` predicted 0.0%
warm, run-1's 95 tokens are covered by the VRAM set), and dropping the fork's pages to read every other expert with
O_DIRECT (768.6 MiB per token at 4.75 GB/s, 168 ms of waiting in a 204 ms pass) threw away a page cache that served
hook-70's misses in about 59 ms.

What changed:

- `STRATA_MOE_COLD=pagecache|direct` (`src/dsv41/moe_residency.h`, `ColdPath`; needs `STRATA_MOE_RAM_GB`, logged
  once with what chose it): `pagecache`, the new default, computes the cold experts straight from the fork's
  mmapped tensors and neither drops nor advises its pages, so whatever RAM the warm tier leaves is page cache for
  them; only the warm fill reads with O_DIRECT, and no staging is allocated. `direct` is the previous O_DIRECT cold path, unchanged. The
  profile's tier table names the path; with pagecache its cold MiB are the experts' bytes and no read is timed
  (the page cache's misses are in the major faults).
- `STRATA_MOE_COUNTS=<path>` (`src/dsv41/moe_counts.h`): per-(layer, expert) activation counts of every finished
  call, decode and batch apart, whichever tier computed the pair; json (`strata-moe-counts/1`) written whole and
  renamed at the first call (a bad path aborts then), every `STRATA_MOE_COUNTS_EVERY` calls (default 4096) and at
  exit. `hot_set.py --counts` (several files merge by summing, `--counts-calls decode|all`) builds the hot set and
  the warm lists from them (`oracle/README.md`).
- `hot_set.py --propose-ram-budget`: the warm tier from the cgroup's limit, a measured working set and a margin,
  capped by the GPU node's MemFree (`oracle/README.md`).
- `pod/prompts/` (code, prose, chat; its README) and `t04a.sh --speed --profile`: after the baseline, the hook with
  counts on those prompts, then every hot set of the speed runs from the counts; `RAM_RUNS=auto` sizes the warm
  tier at the NUMA gate (`WORKING_SET_GIB` 4.5, `MARGIN_GIB` 8, from the pod 5 cgroup: 47.75 GB during the
  hook-70-ram40 benches, of which 42.95 GB pinned); `COLD` sets `STRATA_MOE_COLD`; every run prints its plan with
  estimates from the pod 5 step times, and a dry run over `CAP_MIN` exits 2. For the previous residency selection
  the plan says 29.3 min, and pod 5 took 28.6.

### Measured on the dev machine (2026-10-07)

Ryzen 7 5700X, 12 vCPUs, 31 GB RAM shared, load average 0.2 to 7.2 during the runs. Strata on top of 97e6f5a (the
profiled warm tier commits), fork 4aab6a5a unchanged. `build.sh` both flavors incremental 2m18s, exit 0; `verify.sh` both flavors
3m41s, exit 0: 15 tests per flavor with `dsv41_moe_counts_test` and the pagecache cases of
`dsv41_moe_residency_test` (the op bit for bit against the mapped tensors with ref and fast, 1, 5 and 32 tokens,
no read but the warm fill), `hot_set_test.py` and `t04a-lib-test.sh` PASS, and the dry run of the profiled session:

    SPEED_BUDGETS=70 RAM_RUNS=auto CPU_KERNELS=fast EXTRA=0 CAP_MIN=60 bash /root/t04a/dsv41/pod/t04a.sh --speed --profile

53 lines; steps env, numa-gate, ram-auto, setup, hot-set-profile, baseline, profile (6 prompts), hot-set-70 and
hot-set-70-ram`<auto>` from the counts, hook-70-fast, hook-70-ram`<auto>`, snapshot-end; plan 37.8 min of script
plus 6.3 outside it (create, upload, pull and delete on pod 5), a 44.1-minute session, about 1.17 USD at pod 5's
1.59 USD/h.

`oracle/residency-pressure-test.sh $BUILD_ROOT/t04d/pressure3`: OLMoE (4.2 GB) under `docker --memory 3g`,
cu128, 8 threads, `-nr`, the model dropped from the page cache before each run. The warm list (1 GiB, 281
experts) comes from the counts of the six profiling prompts at 128 tokens, one counts file each, merged by
`hot_set.py`; the measured prompt is `oracle/prompt.txt`, which none of them is. Stock -nr without a limit is the
reference.

| Run (3 GB limit) | Tokens vs stock | rel L2, all layers | Decode op per pass | Major faults per decode token | per prompt token | Warm / cold pairs |
|---|---|---:|---:|---:|---:|---|
| mmap-only | 16 of 16 identical | 0.000e+00 | 108.8 ms | 745.4 | 395.8 | |
| tiered-direct (locked, O_DIRECT) | 16 of 16 identical | 0.000e+00 | 112.0 ms | 13.8 | 16.3 | 51.8 / 48.2% |
| tiered-pagecache (locked) | 16 of 16 identical | 0.000e+00 | 100.7 ms | 750.2 | 429.3 | 51.8 / 48.2% |
| tiered-plain (pagecache) | 16 of 16 identical | 0.000e+00 | 98.9 ms | 792.5 | 353.9 | 51.8 / 48.2% |

tiered-pagecache is 7.5% faster per decode pass than mmap-only, but its major faults per decode token are not
lower: 750.2 against 745.4. The same on every run of the day: `t04d/pressure2` 106.7 against 113.3 ms (-5.8%),
778.6 against 721.9 faults; a 2 GiB warm tier (`t04d/pressure-ram2`, 75.2% warm) 89.1 against 113.4 ms (-21.5%),
783.6 against 780.1. Only direct, which keeps the experts out of the page cache, takes the decode faults to about
0; its decode pass was 112.0, 105.4 and 65.4 ms in those three runs (the 2 GiB one reads a quarter of the pairs).
Dropping just the warm experts' pages in pagecache mode (an uncommitted experiment, `t04d/exp-dropwarm1`) gave
769.4 faults against 761.8 without it, so the duplicates are not the cause. What the numbers fit: under the limit
LRU already keeps the experts the measured prompt uses most, and a warm list from other prompts (51.8% against
63.0% in-sample) does no better while it takes 1 GiB away from the page cache that the cold experts and the dense
weights share. The acceptance bar was therefore redefined: same
tokens and rel L2 0 against stock -nr, tiered-pagecache at most 5% slower per decode pass than mmap-only, and the
major faults reported without a bar, since in pagecache mode the page cache still serves the cold experts by design.
The pod run decides the speed against hook-70. O_DIRECT baseline of the same file right after: `dd`, 1 MiB blocks, 1.23 GB/s with one stream and
3.27 GB/s with 8.

## Prefill streaming (`STRATA_MOE_PREFILL_*`) and the indexer's int32 overflow

Why: decode is met (the profiled warm tier, pod 7: hook-70-ram103 28.52 / 28.00 tok/s against stock 22.06 / 23.38 at 120 / 480
tokens), but a long prompt runs its batch calls through the same hit/miss split, and a prompt ubatch touches nearly
every expert. A 4096-token ubatch routes 24576 (token, slot) pairs over 384 experts, 64 per expert on average, so
practically all of them are used: with a 70 GiB slab (5867 experts) the other 9493 experts, 113.3 GiB, would be
read by the CPU per ubatch (~22 GB/s, ~5.5 s of reads alone) and then computed there. The bar is a 16K-token
prompt's first token in <= 60 s (>= ~270 tok/s) on the same A100 pod; the stock fork's reference is ~40-45 tok/s
for short prompts, and long prompts were never measured on the A100. Separately, the fork crashed with `CUDA error:
invalid argument` in `ggml_cuda_op_relu` at 131072 x 512 on a B300: the indexer's int32 overflow.

### The prefill path (`src/dsv41/moe_prefill.h`, `moe_gpu.cpp`)

With a GPU tier, a batch call of at least `STRATA_MOE_PREFILL_MIN` tokens is a prefill call: every pair runs on the
GPU and the CPU only adds the slots in the stock order (phase 4 of `moe_ref.cpp`, unchanged). The plan
(`plan_prefill`) puts the pairs of the slab's experts into slab passes and copies every other expert of the call once
into a ring of device slots, `STRATA_MOE_PREFILL_SETS` sets of `STRATA_MOE_PREFILL_GROUP` experts, a set per ring
pass in turn. A pass is one ggml-cuda graph: `get_rows` gathers its pairs' input rows from the call's input (uploaded
once), then the stock expert path (`mul_mat_id` up and gate, the SwiGLU clamp, `mul_mat_id` down, times the
weights) with every pair a column of its own (`[n_embd, 1, n_pairs]`, ids `[1, n_pairs]`), so no padding rows, and
`y * w` per pair comes back to pinned memory. At most 4096 pairs per pass (~350 MiB of pass buffers at
DeepSeek-V4.1's dims, against ~2 GiB for all 24576 pairs of a 4096-token ubatch).

The copies run on a CUDA stream of strata-moe's own, the passes on the ggml backend's stream (`run_prefill`):

    ring pass p on set s, staged through h:  [host waits: the copy out of h done] the CPU threads fill h
                                             copy stream: [wait: the last pass on s done] copies, done marks of s and h
                                             compute stream: wait for s's copy, the pass, done mark of s

so a set is refilled only after the last pass that read it, a pass runs only after its own copy, and staging is
refilled only after the copy that read it; no host sync per expert. An expert's bytes come straight from the warm
tier when it is CUDA pinned (the residency's default with a GPU); anything else (a cold expert in the fork's mapping, a
locked or plain warm tier, no residency) is staged into `STRATA_MOE_PREFILL_STAGE` sets of pinned staging, by
`STRATA_MOE_PREFILL_COLD`: with `pagecache` (the default) the op's other ggml-cpu threads copy it from the mapping in
1 MiB chunks (`StageJobs`, lock free; several threads fault the mapping in at once), with `direct` the residency's
reader threads read it from its GGUF file with O_DIRECT (`Residency::read_expert`, the cold tier's reader, never through
the page cache) while the scheduling thread waits. Both fill one layout (`StageLayout`): each part at a 4096-aligned
offset with room for an O_DIRECT read at any file offset, so the ring copy reads a part from where it landed. The
staged passes are spread evenly among the others so the staging overlaps the PCIe stream. The ring is one VRAM buffer
per part (up, gate, down) laid out per call for the call's layer types, so a set is a contiguous `[n_per_row, n_rows,
set_size]` tensor; it is a COMPUTE buffer zeroed once, so ggml-cuda never memsets the row padding after a set (which
is the next set's first expert), and MMQ's reads past a set's last row always land on finite weights or zeros.

| Selector (logged once) | Default | What |
|---|---|---|
| `STRATA_MOE_PREFILL_MIN` | 64 | batch calls of at least this many tokens take the prefill path (64 tokens route 384 pairs, about one per expert); 0: off, batch calls split as before |
| `STRATA_MOE_PREFILL_GROUP` | 16 | experts per ring set: 16 x 12,810,240 B = 195.5 MiB, ~1024 pairs per pass at a 4096-token ubatch |
| `STRATA_MOE_PREFILL_SETS` | 3 | ring sets: one computed, one copied, one queued; 586.5 MiB of VRAM beside the slab (checked against what the slab leaves free) |
| `STRATA_MOE_PREFILL_STAGE` | 2 | pinned staging sets: one filled while the copy stream reads the other (391.4 MiB: each part 4096-aligned with a block of room) |
| `STRATA_MOE_PREFILL_COLD` | pagecache | how the staged experts reach the staging: `pagecache` copies them from the fork's mapping on the op's threads (the page cache served the decode misses well on pod 5), `direct` reads them with O_DIRECT on the residency's reader threads (pod 5: 4.75 GB/s cold); `direct` needs `STRATA_MOE_RAM_GB`, an unknown value aborts naming it, the choice is logged with the prefill path |

Without a GPU tier (no hot set, a hot set without VRAM experts, or no CUDA device) batch calls stay on the CPU path,
and the tier's log line says so. Decode calls keep the hit/miss split. `STRATA_MOE_PREFILL_MIN=0` restores the previous
behaviour exactly. `STRATA_MOE_COLD=direct` still drops the fork's pages after a layer's first call; with
`STRATA_MOE_PREFILL_COLD=pagecache` the prefill path then re-faults its staged experts from the mapping (correct,
just not O_DIRECT), with `direct` it reads them from the file like the decode misses.

`STRATA_MOE_PROFILE=1` adds a prefill table per layer: calls, passes, MiB copied to the ring and of them staged, the
copy stream's and the compute stream's busy ms (CUDA events around each pass's copies and graph), copy GB/s, the
GPU ms of the call, the overlap of the two streams ((copy + compute - gpu) / the shorter one: 1 when the shorter is
hidden under the longer), the scheduling thread's ms filling the staging (copying, or waiting for the direct reads)
and waiting for a staging set, the op calls (the prefill calls that were not their layer's first call, which also
uploads the slab and fills the warm tier) and their mean ms of the whole call, and the sum of the layer means (one
ubatch), the MoE op's ms per ubatch among them.

The arithmetic of the default ubatch (4096): an expert is 12,810,240
B (up and gate q2_K 5120 x 2304, 3,870,720 B each; down q3_K 2304 x 5120, 5,068,800 B). Every expert outside the
slab is copied once per ubatch whatever its pair count, so the streamed bytes per prompt scale with the number of
ubatches. Outside a 70 GiB slab that is 121.6 GB per ubatch, 5.5 s at 22 GB/s, 22.1 s of PCIe for a 16K prompt in
four ubatches (44.2 s at 2048, 88 s at 1024). The `--prefill` session's 49 GiB slab is below. The staged share is
what neither VRAM nor a pinned warm tier holds: with the profiled 103 GiB warm tier (8623 experts) 870 experts (11.1 GB
per ubatch) at hot-70, read through the page cache by the CPU threads (`STRATA_MOE_PREFILL_COLD=pagecache`) or with
O_DIRECT (`direct`). The outputs come back per pair: 24576 x 5120 x 4 B = 503 MB per layer,
20.1 GB per ubatch over the other PCIe direction.

### The indexer's int32 overflow (the fork patch)

`build_indexer_top_k` (`src/models/deepseek41.cpp` in the fork) makes one f32 score tensor of n_comp x ubatch x 32
elements (`deepseek41.attention.indexer.head_count` 32; n_comp grows to the context) and runs cont, relu, mul and
sum_rows on it. `unary_cuda` took the element count as an `int`: at 2^31 elements it wrapped to `INT_MIN`, the
block count `(INT_MIN + 255) / 256 = -8388607` became 4286578689 blocks through `dim3`, over CUDA's 2^31 - 1 grid
limit, hence `invalid argument` in `ggml_cuda_op_relu` (131072 x 512 x 32 = 2^31 exactly; also 16384 x 4096 and
32768 x 2048). `reduce_rows_f32` (sum_rows) indexed `row * ncols + i` in `int`, which wraps above 2^31 (32768 x 4096).
Both are 64-bit now, unconditionally (the patch only widens arithmetic), and `unary_cuda` asserts its block count
fits the grid; `cont` and `mul` were 64-bit already (`third_party/llama-dsv41/README.md`). The fix is in the one fork
commit `b4d9cb1` (base 5210c7c5, the patch re-exported; the tarballs' MANIFEST records its sha256), so stock gets it
too. `dsv41_indexer_index_test` checks the launch arithmetic of both versions at the shapes above and that the fork
tree has the new code; the kernels themselves only run on the pod.

What the fix does not change: the scores' size. Two of them are live at once (the mul_mat output and its cont), 2 x
32 x ctx x ubatch x 4 B: 32 GiB at 32768 x 4096, 16 GiB at 16384 x 4096 or 32768 x 2048. They live in the fork's
compute buffer on CUDA0 beside the slab, which is why the `--prefill` session runs two server configurations and a
smaller hot set than the decode runs (below).

### The pod session (`pod/t04a.sh --prefill`, `pod/prefill_prompts.py`)

The prompts (`prefill_prompts.py build`): the fork's own sources at 5210c7c5 (`git show`, every `*.c`, `*.cpp`,
`*.h`, `*.cu`, `*.cuh` under `src/` and `ggml/src/` in path order), cut on whole lines to 4096, 8192, 16384 and 32704
tokens (32768 less the 4K prompt's 64 generated tokens) between a fixed head and question, each starting where the
previous ended, so no two share a prefix; raw completion text, since the GGUF adds no BOS. Token counts come from the
fork's `llama-tokenize` (built for verify only) with a vocab-only GGUF made from the first 64 MiB of shard 1 (the
whole metadata: 129280 tokens, 127741 merges, pre-tokenizer `joyai-llm`):

```
curl -L -r 0-67108863 -o head.bin https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/resolve/c4a085541cb53f67ee5e57b63d255e80cef286e7/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf
python3 tools/dsv41/pod/prefill_prompts.py vocab head.bin $BUILD_ROOT/vocab/deepseek41-vocab.gguf
```

`verify.sh` builds the prompts with it and checks them (sha256, counts within 2%). On the pod each prompt's
`prompt_n` must equal the manifest's count.

The session: env, the NUMA gate and the auto warm tier of `--speed`, setup, the
prompts' check (each prompt and its answer must fit its configuration's ctx), one hot set from the profiled pod run's counts
(`$PAYLOAD/counts.json`, the `counts.json` that run wrote) at `PREFILL_VRAM` GiB with
the auto warm tier, then the arms. The indexer's scores rule out one 32K server at ubatch 4096 next to a useful slab,
so there are two server configurations, every arm with every routed expert in host memory, `-nr` and `-np 1`, hook
and stock of a configuration differing only in `LLAMA_STRATA_MOE=1`:

| Configuration | ctx | batch = ubatch | Prompts | Indexer scores |
|---|---|---|---|---|
| A (`PREFILL_CTX_A`, `PREFILL_UB_A`) | 16896 | 4096 | 4K (64 greedy tokens), 8K, 16K | 16.50 GiB |
| B (`PREFILL_CTX_B`, `PREFILL_UB_B`) | 33280 | 2048 | 32K | 16.25 GiB |

The arms, in order: `hook-prefill-a`, `hook-prefill-b`, `stock-prefill-a`, `stock-prefill-b`, then
`hook-prefill-a-direct` when it runs. Each prompt is one request, one run; the 16K TTFT of `hook-prefill-a` against
the 60 s bar (the client's time to the first token), the 4K greedy tokens of the hook against stock in A for the
token agreement, and after every request a crash check (the server alive, no `CUDA error` or `GGML_ASSERT` in its
log). After each arm `verdicts.txt` gets, per prompt and once over all the arm's prompts, the server's ms per ubatch
(prompt ms / ubatches); for a hook arm (`STRATA_MOE_PROFILE=1`, summary at exit) the arm line also splits it into the
MoE op's ms per ubatch and the rest, the dense part, which is the risk the bar carries, and then the prefill
profile per ubatch. The op time is the profile's prefill line: the sum of the layer means of the prefill calls' op
ms, leaving out each layer's first call (it uploads the slab and fills the warm tier, seconds that are no ubatch's).
It is not the batch-call table's op, which also averages in the smaller batches, `llama-server`'s 2-token warmup
(`{bos, eos}`, every expert routed) among them, which is then every layer's first call. It is the arm's mean over
all its prompts, so the split is reported per arm and not per prompt.
`hook-prefill-a` runs with `STRATA_MOE_PREFILL_COLD` at its default (pagecache); when its profile is staging-bound
(the host's staging ms per ubatch over the copy stream's busy ms: the PCIe copies waited for the staging) and the
time left holds its estimate, `hook-prefill-a-direct` reruns the 16K prompt with `STRATA_MOE_PREFILL_COLD=direct`;
otherwise `verdicts.txt` says why not. Payload: the cu128 fork tarball of this build, `tools/dsv41`, `counts.json`,
`prefill-prompts/`.

```
CAP_MIN=63 bash /root/t04a/dsv41/pod/t04a.sh --prefill
```

The hot set: what the fork leaves for the slab in the worse configuration, less a reserve. Per configuration the plan
prints the pod 7 measurement (A100 SXM 80 GB: the fork took 79.250 - 74.302 = 4.948 GiB before the slab at ctx
4096 x ubatch 1024, its own 1 GiB of indexer scores among them) plus what is known to grow: the scores (2 x 32 x ctx x
ubatch x 4 B, less the 1 GiB already in), the ring (3 x 16 x 12,810,240 B), a pass's buffers (~350 MiB) and the
input (ubatch x 5120 x 4 B):

    A: 4.95 + 16.50 - 1.00 + ring 0.57 + pass 0.34 + input 0.08 = 21.44 GiB
    B: 4.95 + 16.25 - 1.00 + ring 0.57 + pass 0.34 + input 0.04 = 21.15 GiB
    hot set = floor(79.25 - 21.44 - PREFILL_RESERVE_GIB 8) = 49 GiB, margin 8.81 GiB

The reserve holds what nothing has measured, the KV cache at ctx 33280 and the fork's other compute buffers at
ubatch 4096: 8 GiB is twice pod 7's 3.95 GiB outside the scores (the dense weights, the KV cache at 4096 and the
compute buffers at 1024 together). Each server's peak VRAM goes to the log, so the next session can set the reserve
from a measurement. `PREFILL_VRAM` overrides the hot set; a margin under 2 GiB exits 2. At 49 GiB about 4107
experts are resident (49 GiB / 12,810,240 B, before ggml's per-tensor padding), so about 11253 are streamed per
ubatch: 144.2 GB, 6.6 s at 22 GB/s, 26.2 s of PCIe for the 16K prompt in four ubatches; with the profiled 103 GiB warm
tier (8623 experts) about 2630 of them (33.7 GB per ubatch) are staged. In B a 2048-token ubatch still routes 32
pairs per expert on average, so the 32K prompt streams the same bytes in 16 ubatches: ~2307 GB, ~105 s of PCIe.

The plan, from the pod 7 step times (setup 4.2 min; the hook with the 103 GiB auto warm tier listening after 636 s,
the pinned allocation's 104.5 s in it, and its warm fill at 5.4 GB/s; stock listening after 345 s) and assumed
prefill rates (the hook at 100 tok/s, a third of its bar; stock at the 45 tok/s stock reference), must fit a
70-minute session with the 6.3 minutes outside the script, and CAP_MIN. With all four arms the script takes 73.3 min,
so the plan drops `stock-prefill-b` (18.3 min, the 32K stock arm) first and says so; `hook-prefill-a-direct` (14.1
min) is conditional and outside the total.

### Measured on the dev machine (2026-10-07)

Ryzen 7 5700X, 12 vCPUs, 31 GB RAM shared, no GPU, load average 1.0 to 2.1. Strata on top of 3da2b80, fork
`b4d9cb1` (5210c7c5 plus the patch). `build.sh` both flavors incremental 2m41s, exit 0, no compiler warning;
`verify.sh` both flavors 4m41s, exit 0: 17 tests per flavor, among them

- `dsv41_moe_prefill_test` PASS, 130 cases: 108 plans (64, 512 and 4096 tokens; slabs of 0, 1, 147 and 384 experts;
  0, 10 and 100% staged; rings 16 x 3 / 2, 1 x 1 / 1, 5 x 2 / 3): every pair one row, every expert outside the slab
  copied once (a 4096-token ubatch with no slab: 384 experts in 24 passes at 16 x 3), sets and staging in turn, the
  staged passes spread. 15 schedules, 200 random interleavings each of a fake two-stream engine with CUDA's event
  semantics, each run with its own GPU speed and stream bias: no slot overwritten before its pass ran, no staging
  refilled under a queued copy, passes in order, each expert copied once, at most one host wait per staged pass; the
  same checker catches a schedule without the ring's wait, and one without the staging wait, in every case where a
  set or a staging set is reused. The staging jobs over 8 threads (257 jobs, 51.0 MiB). A prefill call through the
  op with a CPU engine (host ring and staging, each pass computed from the ring's bytes): 0 differing output floats
  against all-CPU in all 5 prefill cases (64 to 96 tokens, staged 0 to 100%, up to 39.1 MiB staged by the op's
  threads), and a 32-token call stays off the path.
- `dsv41_indexer_index_test` PASS, 18 cases: the old code gives a valid grid below 2^31 elements and `invalid
  argument` (4286578689 blocks) at exactly 2^31 (16384 x 4096, 32768 x 2048, 131072 x 512), a wrong sum_rows offset
  above it (32768 x 4096); the new code is right at every shape up to 32768 x 8192 (2^33); the fork tree has the new
  code and not the old.

OLMoE parity with the new code built in (`dsv41-moe-dump ... -n 16 -ngl 0 -c 256 -t 8 -s 42 -nr`, CPU, cu128 build
tree, `read_dump.py --tokens-must-match --compare`): hook on (`STRATA_MOE_CPU=fast`) against stock `-nr` of the same
build, 16 of 16 tokens and rel L2 of `out` 0.000e+00 in all 16 layers; the same with a 1 GiB hot set
(`STRATA_MOE_VRAM_GB=1`) and no CUDA device, which logs `no CUDA device: no GPU tier, its VRAM experts run on the
CPU; batch calls (prefill) run on the CPU path too`; and stock of this build against the hook's stock dump `t03r2-offnr`, 0.000e+00
too (the patch changes nothing on the CPU).

Second version (after the review), same machine, load average 1.9 to 2.4. Strata
92296cc and 6b2cb06 (`STRATA_MOE_PREFILL_COLD`, the profile's wording) and cec7e8f (the two configurations), fork
`b4d9cb1` unchanged. `build.sh` both flavors incremental 2m30s, exit 0, no compiler warning; `verify.sh` both flavors
4m47s, exit 0, 17 tests per flavor:

- `dsv41_moe_prefill_test` PASS, 143 cases: the 130 above, `choose_prefill_cold` (unset and empty are pagecache, the
  default; `odirect` is refused naming it), and every op case run with both staging paths, the experts now mapped
  from a file in the build tree: 0 differing output floats against all-CPU with pagecache and with direct in all 6
  cases, and 0 between the two; direct read each staged expert once with O_DIRECT (18 experts, 39.3 MiB of whole
  blocks for the 39.1 MiB the op threads copy with pagecache) and the op threads copied nothing.
- `pod/t04a-lib-test.sh` PASS: `ubatch_verdicts` on a log in the library's profile format, whose batch-call table
  holds a 2-token warmup call (a mean op of 24000 ms per pass), gives a 16K prompt of 48000 ms 4 ubatches of 12000 ms
  and a 4K one of 11000 ms 1 ubatch, and the arm 5 ubatches of 11800 ms split into 9500 ms of MoE op (the prefill
  line's, not the batch table's) and 2300 ms dense; staging 8000 ms against a 3400 ms copy stream is staging-bound,
  3000 ms is not, a stock arm without a log is unknown.

OLMoE parity, the second version's cu128 tarball (`dsv41-moe-dump ... -n 16 -ngl 0 -c 256 -t 8 -s 42 -nr`, the CUDA stub,
`read_dump.py --tokens-must-match --compare`): hook (`STRATA_MOE_CPU=fast`) against stock `-nr`, and with the 1 GiB
hot set and no CUDA device, 16 of 16 tokens and 0.000e+00 in all 16 layers; stock against `t03r2-offnr` 0.000e+00.
`STRATA_MOE_PREFILL_COLD=direct` without `STRATA_MOE_RAM_GB` aborts with `STRATA_MOE_PREFILL_COLD=direct needs
STRATA_MOE_RAM_GB, the residency whose reader threads read the staged experts`, `odirect` with `expected pagecache
(the default) or direct`.

The `--prefill` dry run (`PREFILL_PROMPTS=<built> CAP_MIN=63`), exit 0 in both flavors: prompts of 4094, 8178, 16378
and 32699 tokens (-0.05, -0.17, -0.04, -0.02% of their targets); the VRAM lines above (A 21.44 GiB, B 21.15, hot set
49 GiB computed, margin 8.81); plan env 0.1, numa-gate 0.1, setup 4.5, prompts and hot set 0.3, `hook-prefill-a`
16.2, `hook-prefill-b` 16.8, `stock-prefill-a` 16.8, snapshot and pack 0.2: 55.0 min of script plus 6.3 outside, a
61.3-minute session, with `stock-prefill-b` (18.3 min) dropped because all four arms take 73.3 min of script, over
63.0; `hook-prefill-a-direct` 14.1 min, conditional.

Third version (after a second review), same machine, load average 1.4 to 2.1. Strata 2845081 (the prefill table's op calls and
op ms, and the op sum in the prefill per ubatch line), b843433 (a comment reflow) and bcc15c5 (`ubatch_verdicts`
splits dense and expert time per arm from that line), fork `b4d9cb1` unchanged. `build.sh` both flavors incremental,
exit 0, no compiler warning; `verify.sh` both flavors 4m46s, exit 0, 17 of 17 tests per flavor, the prefill test 143
cases PASS, `pod/t04a-lib-test.sh` PASS (the warmup case above), the `--prefill` dry run with `CAP_MIN=63` exit 0 and
the same plan (55.0 min of script, a 61.3-minute session, `stock-prefill-b` dropped). OLMoE parity with the third version's
cu128 tarball, the same command and the CUDA stub: hook `-nr` against stock `-nr`, the 1 GiB hot set
(`STRATA_MOE_VRAM_GB=1`) with no CUDA device against stock, and stock against `t03r2-offnr`, 16 of 16 tokens and max
rel L2 of `out` 0.000e+00 each; `odirect` and `direct` without `STRATA_MOE_RAM_GB` abort with the messages above.
With no GPU no call takes the prefill path, so the new line's format was checked by rendering the source's format
string with sample values and parsing it with `UBATCH_PY` (experts and dense as in the lib test).

Not measured, and what could miss the bar on the pod: the dense part at ubatch 4096 (on a B300 with every expert in
VRAM the stock fork prefilled at 131 tok/s after 4K tokens and 167 after 10K with ubatch 512 in an earlier long-context
measurement, so the per-token cost outside the experts is not small; the session
now reports dense against expert time per ubatch); the staged experts' rate through the page cache (33.7 GB per
ubatch at hot-49, against pod 5's 4.75 GB/s with O_DIRECT, which the direct rerun measures when staging bounds
the passes); the VRAM of the KV cache and the compute buffers at the larger ctx and ubatch, which the 8 GiB reserve
stands for; the hook server's start with the warm tier (10.6 min on pod 7) and the assumed prefill rates of the plan;
and the copy GB/s from pinned memory on that pod's PCIe gen4 (22 GB/s is the planning figure). The pod session
reports each.
