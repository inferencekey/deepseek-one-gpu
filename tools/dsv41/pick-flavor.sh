#!/usr/bin/env bash
# Run on a pod: picks the build flavor (tools/dsv41/flavors.sh) for this pod's GPU arch and NVIDIA driver, read
# from nvidia-smi, and prints it with the two tarballs of that flavor found in DIST_DIR (the directory holding
# the tarballs build.sh wrote, $BUILD_ROOT/dist on the build machine). Needs flavors.sh next to it.
#
#   tools/dsv41/pick-flavor.sh [--fork-only] DIST_DIR
#
# Prints key=value lines (flavor, cuda, arch, driver, min_driver, fork_tarball, strata_tarball). Exits 1 with the
# reason when no flavor fits: the arch is not built in any flavor, or the driver is older than the flavor needs.
# --fork-only: the run needs only the fork tarball (it carries libstrata-moe.so), so no strata tarball is looked
# for and no strata_tarball line is printed.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/flavors.sh"

fail() { echo "pick-flavor.sh: $*" >&2; exit 1; }

fork_only=0
if [[ "${1:-}" == "--fork-only" ]]; then fork_only=1; shift; fi
(( $# == 1 )) || { sed -n '6p' "${BASH_SOURCE[0]}" | sed 's/^# *//' >&2; exit 2; }
dist="$1"
[[ -d "$dist" ]] || fail "DIST_DIR '$dist' is not a directory"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found on PATH"

query="$(nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader)" \
  || fail "nvidia-smi --query-gpu failed"
# one line per GPU: a pod with mixed GPUs would need two flavors at once
gpus="$(sort -u <<<"$query" | sed '/^$/d')"
[[ -n "$gpus" ]] || fail "nvidia-smi listed no GPU"
(( $(wc -l <<<"$gpus") == 1 )) || fail "GPUs differ on this pod, one flavor cannot serve all of them: $(tr '\n' ';' <<<"$gpus")"

IFS=', ' read -r driver cap <<<"$gpus"
[[ "$driver" =~ ^[0-9]+(\.[0-9]+)+$ ]] || fail "unexpected driver_version '$driver' from nvidia-smi"
[[ "$cap" =~ ^[0-9]+\.[0-9]$ ]] || fail "unexpected compute_cap '$cap' from nvidia-smi"
arch="${cap/./}"

built=""
picked=""
for f in "${DSV41_FLAVORS[@]}"; do
  flavor_config "$f"
  built+="${built:+, }$FLAVOR_TAG: $CUDA_ARCHS"
  if [[ ";$CUDA_ARCHS;" == *";$arch;"* ]]; then picked="$f"; break; fi
done
[[ -n "$picked" ]] || fail "sm_$arch (compute_cap $cap) is not built in any flavor ($built)"

# driver >= MIN_DRIVER, compared as version numbers
if [[ "$(printf '%s\n' "$MIN_DRIVER" "$driver" | sort -V | head -n 1)" != "$MIN_DRIVER" ]]; then
  fail "driver $driver is too old: sm_$arch is built only in $FLAVOR_TAG (CUDA $picked), which needs driver >= $MIN_DRIVER"
fi

tarball() {
  local project="$1" hits
  hits="$(cd "$dist" && ls -1 -- "$project"-*-"$FLAVOR_TAG-$ARCH_TAG".tar.gz 2>/dev/null || true)"
  [[ -n "$hits" && $(wc -l <<<"$hits") -eq 1 ]] \
    || fail "expected one $project-<commit8>-$FLAVOR_TAG-$ARCH_TAG.tar.gz in $dist, found: ${hits:-none}"
  echo "$hits"
}
fork_tgz="$(tarball llama-dsv41)"
(( fork_only )) || strata_tgz="$(tarball strata)"

echo "flavor=$FLAVOR_TAG"
echo "cuda=$picked"
echo "arch=sm_$arch"
echo "driver=$driver"
echo "min_driver=$MIN_DRIVER"
echo "fork_tarball=$fork_tgz"
(( fork_only )) || echo "strata_tarball=$strata_tgz"
