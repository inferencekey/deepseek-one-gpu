#!/usr/bin/env bash
# Builds dsv41_machine_probe (tests/dsv41/engine/machine_probe.cu, the machine qualification of every pod session)
# on its own, once per CUDA flavor (tools/dsv41/flavors.sh: CUDA 12.8 for
# sm_80/89, 13.0 for sm_120), with nvcc alone in that flavor's builder image (tools/dsv41/Dockerfile, as build.sh:
# nvidia/cuda:12.8.1- and 13.0.1-devel-ubuntu22.04 pinned by digest). The engine's CMake, its build dirs and the strata
# tarballs are not touched, so the payload's dsv41-engine stays the measured binary. Needs no GPU.
#
#   tools/dsv41/pod/build-machine-probe.sh                   # both flavors
#   CUDA_FLAVOR=13.0 tools/dsv41/pod/build-machine-probe.sh  # one
#
# Writes $BUILD_ROOT/<tag>/machine-probe/dsv41_machine_probe (cudart linked statically, SASS and PTX of each of the
# flavor's archs) and its MANIFEST: the source's sha256 and the Strata commit it was built at, the image, nvcc's
# version, the archs, the time and the binary's sha256. e07.sh ships it only when the MANIFEST's source is HEAD's and
# the binary is the MANIFEST's. Environment overrides: CUDA_FLAVOR, BUILD_ROOT.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/dsv41/flavors.sh
source "$here/../flavors.sh"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
SRC=tests/dsv41/engine/machine_probe.cu
case "${1:-}" in
  "") ;;
  -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
  *) echo "build-machine-probe.sh: no arguments (CUDA_FLAVOR picks a flavor)" >&2; exit 2 ;;
esac
flavors="$(selected_flavors)" || exit 2
src_sha="$(sha256sum "$STRATA_DIR/$SRC" | cut -d' ' -f1)"
commit="$(git -C "$STRATA_DIR" rev-parse --short=8 HEAD)"
# a source that is not HEAD's is built and said so: e07.sh refuses to ship it
[[ "$(git -C "$STRATA_DIR" show "HEAD:$SRC" 2>/dev/null | sha256sum | cut -d' ' -f1)" == "$src_sha" ]] || commit+="+modified"

for flavor in $flavors; do
  flavor_config "$flavor"
  image="$(builder_image)"
  if ! docker image inspect "$image" > /dev/null 2>&1; then
    echo "build-machine-probe.sh: building $image"
    docker build -q --build-arg "CUDA_FLAVOR=$FLAVOR_TAG" -t "$image" "$here/.."
  fi
  dir="$BUILD_ROOT/$FLAVOR_TAG/machine-probe"
  mkdir -p "$dir"
  gencode=()
  for a in ${CUDA_ARCHS//;/ }; do gencode+=(-gencode "arch=compute_$a,code=[sm_$a,compute_$a]"); done
  # the repo read-only at its host path, the output written as the invoking user
  docker run --rm --init --user "$(id -u):$(id -g)" -e HOME=/tmp/home --entrypoint nvcc \
    -v "$STRATA_DIR:$STRATA_DIR:ro" -v "$dir:$dir" -w "$dir" "$image" \
    -O2 -std=c++17 "${gencode[@]}" -cudart static -Xcompiler -pthread \
      -o "$dir/dsv41_machine_probe" "$STRATA_DIR/$SRC"
  nvcc_version="$(docker run --rm --entrypoint nvcc "$image" --version | tail -n 1)"
  {
    echo "source: $SRC sha256 $src_sha"
    echo "commit: $commit"
    echo "image: $image"
    echo "nvcc: $nvcc_version"
    echo "archs: $CUDA_ARCHS"
    echo "built_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "$(sha256sum "$dir/dsv41_machine_probe" | cut -d' ' -f1)  dsv41_machine_probe"
  } > "$dir/MANIFEST"
  echo "build-machine-probe.sh: $dir/dsv41_machine_probe ($FLAVOR_TAG, sm ${CUDA_ARCHS//;/ sm }, $nvcc_version)"
  sed 's/^/  /' "$dir/MANIFEST"
done
