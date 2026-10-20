#!/usr/bin/env bash
# Builds the llama.cpp fork (llama-dsv41) and Strata against the fork's ggml, then packages both, once per CUDA
# flavor (tools/dsv41/flavors.sh), each in its builder image (tools/dsv41/Dockerfile). Needs no GPU. See
# tools/dsv41/README.md.
#
#   tools/dsv41/build.sh                     # both flavors in sequence, incremental: reuses the build dirs
#   tools/dsv41/build.sh --clean             # wipes each built flavor's build dir and tarballs first
#   CUDA_FLAVOR=12.8 tools/dsv41/build.sh    # one flavor: 12.8 (sm_80, sm_89) or 13.0 (sm_120)
#
# Environment overrides: CUDA_FLAVOR, FORK_DIR, BUILD_ROOT, BUILD_TAG (flavors.sh), JOBS (default 8: nvcc takes 2-3 GB per job on a
# 31 GB VM).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/flavors.sh"
STRATA_DIR="$(cd "$here/../.." && pwd)"
FORK_DIR="${FORK_DIR:-$HOME/llama-dsv41}"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
JOBS="${JOBS:-8}"

clean=0
for arg in "$@"; do
  case "$arg" in
    --clean) clean=1 ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "build.sh: unknown argument '$arg' (only --clean)" >&2; exit 2 ;;
  esac
done

flavors="$(selected_flavors)" || exit 2
[[ -f "$FORK_DIR/ggml/include/ggml.h" ]] || { echo "build.sh: no llama.cpp checkout at FORK_DIR=$FORK_DIR" >&2; exit 1; }
# build dirs are wiped with --clean: refuse anything inside either repo or an empty path
case "$BUILD_ROOT/" in
  /|"$STRATA_DIR"/*|"$FORK_DIR"/*) echo "build.sh: BUILD_ROOT=$BUILD_ROOT must be outside both repos" >&2; exit 1 ;;
esac
DIST="$BUILD_ROOT/dist${BUILD_TAG:-}"   # BUILD_TAG: flavors.sh
# a git worktree's .git file points into its main repository's .git, which the packaging's git reads too (the
# MANIFEST's commit): mounted read only beside the worktree when it lies outside it
git_mounts=()
git_common="$(git -C "$STRATA_DIR" rev-parse --path-format=absolute --git-common-dir)"
case "$git_common/" in "$STRATA_DIR"/*) ;; *) git_mounts=(-v "$git_common:$git_common:ro") ;; esac

fmt_time() { printf '%dm%02ds (%d s)' $(($1 / 60)) $(($1 % 60)) "$1"; }

start=$(date +%s)
for flavor in $flavors; do
  flavor_config "$flavor"
  flavor_start=$(date +%s)
  flavor_root="$BUILD_ROOT/$FLAVOR_TAG${BUILD_TAG:-}"
  echo "build.sh: flavor $FLAVOR_TAG (CUDA $flavor, archs $CUDA_ARCHS) in $flavor_root"
  if (( clean )); then
    echo "build.sh: wiping $flavor_root and $DIST/*-$FLAVOR_TAG-*.tar.gz"
    rm -rf "$flavor_root" "$DIST"/*-"$FLAVOR_TAG"-*.tar.gz
  fi
  mkdir -p "$flavor_root" "$DIST"

  image="$(builder_image)"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "build.sh: building $image"
    docker build -q --build-arg "CUDA_FLAVOR=$FLAVOR_TAG" -t "$image" "$here"
  fi

  # the repos are mounted read-only (the build must not modify either) at their host paths, so paths in logs and CMake
  # caches are the same inside and outside the container; files are written as the invoking user, not root
  docker run --rm --init \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/home -e FORK_DIR="$FORK_DIR" -e STRATA_DIR="$STRATA_DIR" -e JOBS="$JOBS" \
    -e FLAVOR_ROOT="$flavor_root" -e DIST="$DIST" -e FLAVOR_TAG="$FLAVOR_TAG" -e CUDA_ARCHS="$CUDA_ARCHS" \
    -e ARCH_TAG="$ARCH_TAG" -e CUDA_MAJOR="$CUDA_MAJOR" \
    -v "$FORK_DIR:$FORK_DIR:ro" -v "$STRATA_DIR:$STRATA_DIR:ro" "${git_mounts[@]}" -v "$BUILD_ROOT:$BUILD_ROOT" \
    -w "$flavor_root" \
    "$image" bash "$STRATA_DIR/tools/dsv41/build-inside.sh"

  echo "build.sh: flavor $FLAVOR_TAG done, wall time $(fmt_time $(( $(date +%s) - flavor_start )))"
done

echo "build.sh: done, total wall time $(fmt_time $(( $(date +%s) - start )))"
ls -l "$DIST"
