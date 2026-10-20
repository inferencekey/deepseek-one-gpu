#!/usr/bin/env bash
# Runs inside one flavor's builder container (tools/dsv41/build.sh starts it and passes the flavor from
# flavors.sh): checks that the fork checkout is the base commit plus the Strata hook patch, configures and builds
# the fork, then Strata against the fork's ggml (with strata-moe on the fork's libggml-base), in $FLAVOR_ROOT, then
# writes the flavor's two tarballs with their MANIFEST into $DIST.
set -euo pipefail
: "${FORK_DIR:?}" "${STRATA_DIR:?}" "${FLAVOR_ROOT:?}" "${DIST:?}" "${JOBS:?}" "${FLAVOR_TAG:?}" "${CUDA_ARCHS:?}" \
  "${ARCH_TAG:?}" "${CUDA_MAJOR:?}"

# the fork is built from its branch HEAD, which must be FORK_BASE plus exactly this patch (same patch-id), so the
# MANIFEST's base and patch sha256 describe what was compiled
FORK_BASE=5210c7c5ed61dddaee6ed476623abf4b63093d16
FORK_PATCH="$STRATA_DIR/third_party/llama-dsv41/0001-strata-moe-hook.patch"
fork_parent="$(git -C "$FORK_DIR" rev-parse HEAD~1)"
[[ "$fork_parent" == "$FORK_BASE" ]] ||
  { echo "build-inside.sh: fork HEAD~1 is $fork_parent, expected the base $FORK_BASE" >&2; exit 1; }
head_id="$(git -C "$FORK_DIR" show HEAD | git patch-id --stable | cut -d' ' -f1)"
patch_id="$(git patch-id --stable < "$FORK_PATCH" | cut -d' ' -f1)"
[[ -n "$patch_id" && "$head_id" == "$patch_id" ]] ||
  { echo "build-inside.sh: fork HEAD (patch-id $head_id) is not $FORK_PATCH (patch-id $patch_id)" >&2; exit 1; }

# the fork: llama-server and llama-cli with their shared libraries. GGML_NATIVE=OFF gives ggml-cpu ggml's AVX2
# baseline instead of this VM's CPU (a GGML_NATIVE runtime died with "Illegal instruction" on another host).
# GGML_CUDA_NCCL=OFF: the CUDA devel images ship libnccl (190 MB), which ggml-cuda would link for multi-GPU
# all-reduce; the pods do not have it and the earlier pod builds of the fork were made without it.
# LLAMA_USE_PREBUILT_UI=OFF: otherwise the build downloads the browser UI from a Hugging Face bucket (the shallow
# clone's build number has no release there, so it takes "latest", which moves); llama-server's HTTP API is the same.
# RPATH_FLAGS (both projects): every binary gets RUNPATH $ORIGIN, the tarball's bin/ where its libraries are, instead
# of the build tree's absolute paths, so bin/ runs without LD_LIBRARY_PATH.
RPATH_FLAGS=(-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN')
FORK_FLAGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_CUDA=ON
  -DGGML_NATIVE=OFF
  -DGGML_CUDA_NCCL=OFF
  -DGGML_CCACHE=OFF
  -DLLAMA_CURL=OFF
  -DLLAMA_USE_PREBUILT_UI=OFF
  "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS"
  "${RPATH_FLAGS[@]}"
)
FORK_TARGETS=(llama-server llama-cli)
# built for verify.sh only, not packaged: the fork's tiny random models for the hook's clamp test
# (tools/dsv41/oracle/hook-clamp-test.sh), and llama-tokenize, which counts the prefill prompts' tokens with a
# vocab-only DeepSeek-V4.1 GGUF (tools/dsv41/pod/prefill_prompts.py)
FORK_TEST_TARGETS=(test-llama-archs llama-tokenize)

# dsv41-moe-dump and dsv41-moe-replay (tools/dsv41/oracle), built against the fork's build tree and shipped in the
# fork tarball; dsv41-moe-clamp-check, built there too, is for verify.sh
ORACLE_FLAGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  "-DFORK_DIR=$FORK_DIR"
  "-DFORK_BUILD=$FLAVOR_ROOT/fork"
)

# Strata: ggml from the fork's tree. Strata forces GGML_NATIVE=ON for its static ggml-cpu unless STRATA_PORTABLE
# is on (CMakeLists.txt, STRATA_NATIVE_EXPERTS block), so STRATA_PORTABLE=ON is how its ggml-cpu gets
# GGML_NATIVE=OFF with the AVX2 baseline; on Linux it changes nothing else (the static MSVC runtime is MSVC only).
# STRATA_BUILD_CONVERSATION_TESTS=ON registers the conversation_* tests, part of the GPU-free set verify.sh runs.
# STRATA_DSV41_FORK_BUILD builds libstrata-moe.so (src/dsv41) against the fork tree's libggml-base, libggml-cpu and
# libggml-cuda, and its tests dsv41_moe_ref_test and dsv41_moe_split_test.
# STRATA_DSV41_ENGINE builds the dedicated engine (src/dsv41/engine, no ggml) with its test and bench in the build
# root, so the strata tarball carries them for tools/dsv41/pod/devgpu.sh.
STRATA_FLAGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DSTRATA_ENABLE_CUDA=ON
  "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS"
  "-DSTRATA_GGML_DIR=$FORK_DIR"
  -DSTRATA_PORTABLE=ON
  -DGGML_CCACHE=OFF
  -DSTRATA_BUILD_TESTS=ON
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON
  "-DSTRATA_DSV41_FORK_BUILD=$FLAVOR_ROOT/fork"
  -DSTRATA_DSV41_ENGINE=ON
  "${RPATH_FLAGS[@]}"
)

FORK_BUILD="$FLAVOR_ROOT/fork"
STRATA_BUILD="$FLAVOR_ROOT/strata"
ORACLE_BUILD="$FLAVOR_ROOT/oracle"
mkdir -p "$HOME" "$DIST"

step() { printf '\n=== %s [%s]\n' "$*" "$(date -u +%H:%M:%S)"; }

step "fork: configure"
cmake -S "$FORK_DIR" -B "$FORK_BUILD" "${FORK_FLAGS[@]}"
step "fork: build ${FORK_TARGETS[*]} ${FORK_TEST_TARGETS[*]} (-j$JOBS)"
cmake --build "$FORK_BUILD" -j "$JOBS" --target "${FORK_TARGETS[@]}" "${FORK_TEST_TARGETS[@]}"

# the oracle CMake project writes dsv41-moe-dump, dsv41-moe-replay and dsv41-moe-clamp-check into the fork's bin/,
# next to the libraries they load (the replay dlopens libstrata-moe.so, copied there below)
step "oracle: configure and build dsv41-moe-dump, dsv41-moe-replay, dsv41-moe-clamp-check"
cmake -S "$STRATA_DIR/tools/dsv41/oracle" -B "$ORACLE_BUILD" "${ORACLE_FLAGS[@]}"
cmake --build "$ORACLE_BUILD" -j "$JOBS"

step "strata: configure"
cmake -S "$STRATA_DIR" -B "$STRATA_BUILD" "${STRATA_FLAGS[@]}"
step "strata: build all (-j$JOBS)"
cmake --build "$STRATA_BUILD" -j "$JOBS"
# the fork dlopens libstrata-moe.so from libllama's directory (LLAMA_STRATA_MOE=1)
cp -a "$STRATA_BUILD/dsv41/libstrata-moe.so" "$FORK_BUILD/bin/"

# ELF executables directly in a directory (the build products, not CMake's probe binaries in CMakeFiles/)
list_executables() {
  find "$1" -maxdepth 1 -type f -perm -u+x -print0 | sort -z | while IFS= read -r -d '' f; do
    [[ "$(head -c 4 "$f")" == $'\x7fELF' ]] && printf '%s\n' "$f"
  done
}

# Copies the executables and every shared library they load from the build tree into <stage>/bin. libgomp (the
# fork's ggml-cpu uses OpenMP) is not in a bare ubuntu:22.04 and comes from the same gcc 11 the build used, so it
# is bundled too, and so is the flavor's CUDA runtime (libcudart, libcublas, libcublasLt), since the pods' image
# has an older CUDA 12 (flavors.sh). The driver's libcuda.so.1 never is.
stage_bin() {
  local stage="$1" libdir="$2"; shift 2
  mkdir -p "$stage/bin"
  local exe lib
  for exe in "$@"; do
    cp -a "$exe" "$stage/bin/"
    while read -r lib; do
      case "$lib" in
        "$FLAVOR_ROOT"/*|*/libgomp.so.*|*/libcudart.so."$CUDA_MAJOR"|*/libcublas.so."$CUDA_MAJOR"|*/libcublasLt.so."$CUDA_MAJOR")
          cp -aL "$lib" "$stage/bin/" ;;
      esac
    done < <(LD_LIBRARY_PATH="$libdir" ldd "$exe" | awk '$2 == "=>" && $3 ~ /^\// { print $3 }')
  done
}

# MANIFEST: provenance of the tarball and the sha256 of every file in it (MANIFEST itself excluded)
write_manifest() {
  local stage="$1" project="$2" repo="$3" flags="$4" extra="$5"
  local dirty="no"
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]] || dirty="yes"
  {
    echo "project: $project"
    echo "repo: $(git -C "$repo" remote get-url origin 2>/dev/null || echo "$repo")"
    echo "commit: $(git -C "$repo" rev-parse HEAD)"
    echo "branch: $(git -C "$repo" rev-parse --abbrev-ref HEAD)"
    echo "tracked_changes: $dirty"
    echo "cuda_flavor: $FLAVOR_TAG"
    echo "cuda_archs: $CUDA_ARCHS"
    echo "bundled_cuda_runtime: yes (.so.$CUDA_MAJOR)"
    echo "cmake_flags: $flags"
    [[ -z "$extra" ]] || echo "$extra"
    echo "nvcc: $(nvcc --version | tail -n 2 | head -n 1)"
    echo "cxx: $(c++ --version | head -n 1)"
    echo "built_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "sha256:"
    (cd "$stage" && find . -type f ! -name MANIFEST -printf '%P\n' | sort | xargs -d '\n' sha256sum)
  } > "$stage/MANIFEST"
}

package() {
  local project="$1" repo="$2" flags="$3" extra="$4" libdir="$5"; shift 5
  local name="$project-$(git -C "$repo" rev-parse --short=8 HEAD)-$FLAVOR_TAG-$ARCH_TAG"
  local stage="$DIST/$name"
  rm -rf "$stage" "$DIST/$project"-*-"$FLAVOR_TAG"-*.tar.gz
  stage_bin "$stage" "$libdir" "$@"
  write_manifest "$stage" "$project" "$repo" "$flags" "$extra"
  tar -C "$DIST" -czf "$DIST/$name.tar.gz" "$name"
  rm -rf "$stage"
  echo "packaged $DIST/$name.tar.gz"
}

step "package"
fork_exes=()
for t in "${FORK_TARGETS[@]}" dsv41-moe-dump dsv41-moe-replay libstrata-moe.so; do fork_exes+=("$FORK_BUILD/bin/$t"); done
strata_commit="$(git -C "$STRATA_DIR" rev-parse HEAD)"
fork_extra="fork_base: $FORK_BASE
fork_patch: third_party/llama-dsv41/0001-strata-moe-hook.patch (Strata $strata_commit), sha256 $(sha256sum "$FORK_PATCH" | cut -d' ' -f1)
dsv41-moe-dump, dsv41-moe-replay: Strata $strata_commit tools/dsv41/oracle, cmake_flags: ${ORACLE_FLAGS[*]}
libstrata-moe.so: Strata $strata_commit src/dsv41, built with the Strata cmake_flags of its tarball"
package llama-dsv41 "$FORK_DIR" "${FORK_FLAGS[*]}" "$fork_extra" "$FORK_BUILD/bin" "${fork_exes[@]}"
mapfile -t strata_exes < <(list_executables "$STRATA_BUILD")
package strata "$STRATA_DIR" "${STRATA_FLAGS[*]}" "" "$STRATA_BUILD" "${strata_exes[@]}"
