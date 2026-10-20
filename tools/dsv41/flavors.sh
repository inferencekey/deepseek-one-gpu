# Build flavors of the DeepSeek-V4.1 toolchain, sourced by build.sh, verify.sh and pick-flavor.sh.
# A flavor is one CUDA toolkit and the GPU archs built with it. sm_120 stays on CUDA 13.0 (Strata's CMake warns
# that CUDA 12.8 engines crashed on RTX 50, #220/#224); sm_80/89 go on CUDA 12.8 so the pods with a 570 driver
# (A100 80GB PCIe on RunPod: 570.172.08) can run them. Minimum Linux drivers are NVIDIA's for each toolkit. Both
# flavors bundle their CUDA runtime (libcudart, libcublas, libcublasLt) in the tarballs: the pod image's CUDA 12.4
# is older than either.

DSV41_FLAVORS=(12.8 13.0)

# BUILD_TAG (default empty): a suffix of the flavors' build dirs and of the tarballs' dist dir under BUILD_ROOT, so a
# second worktree builds and tests beside the first without touching its tarballs (for example BUILD_TAG=-c03:
# cu128-c03, cu130-c03 and dist-c03). build.sh, verify.sh, the quality tests, e07.sh and devgpu.sh read it.

# flavor_config <12.8|13.0> sets:
#   FLAVOR_TAG           cu128 / cu130, in image tags, build dirs and tarball names
#   CUDA_ARCHS           CMAKE_CUDA_ARCHITECTURES of the flavor
#   ARCH_TAG             the archs in tarball names (sm80-89)
#   MIN_DRIVER           the oldest NVIDIA Linux driver that runs the flavor's toolkit
#   CUDA_MAJOR           major of the CUDA runtime sonames (libcudart.so.<major>)
#   POD_CUDA_VERSIONS    the CUDA versions a RunPod host's driver must run to take the flavor (REST
#                        allowedCudaVersions, the catalog's cudaVersions), comma separated; a newer driver runs
#                        an older toolkit's binaries, so the newer minors are listed too (on 2026-10-14 every
#                        RTX PRO 6000 in stock was on a CUDA 13.2 host)
flavor_config() {
  case "$1" in
    12.8) FLAVOR_TAG=cu128 CUDA_ARCHS="80;89"  MIN_DRIVER=570.26 CUDA_MAJOR=12 POD_CUDA_VERSIONS=12.8,12.9,13.0,13.1,13.2 ;;
    13.0) FLAVOR_TAG=cu130 CUDA_ARCHS="120"    MIN_DRIVER=580.65 CUDA_MAJOR=13 POD_CUDA_VERSIONS=13.0,13.1,13.2 ;;
    *) echo "unknown CUDA_FLAVOR '$1' (expected one of: ${DSV41_FLAVORS[*]})" >&2; return 1 ;;
  esac
  ARCH_TAG="sm${CUDA_ARCHS//;/-}"
}

# gpu_flavor <RunPod GPU id>: the flavor built for that GPU. RunPod names its sm_120 GPUs "... Blackwell ..." (the
# RTX PRO 6000 Server and Workstation Editions, RTX PRO 4500 and 5000) or "GeForce RTX 50xx"; every other GPU the
# pod scripts take (A100, L40S, RTX 4090, ...) is sm_80 to sm_89. B200 and B300 (sm_100, sm_103) are in no flavor:
# the cu128 CUDA probe fails on them and the pod goes.
gpu_flavor() {
  case "$1" in
    *" Blackwell "*|*"RTX 50"[0-9][0-9]*) echo 13.0 ;;
    *) echo 12.8 ;;
  esac
}

# builder_image: the builder image tag of the configured flavor. It follows the Dockerfile's content, so an
# edited Dockerfile gets a fresh image and an unchanged one is reused from the local cache.
builder_image() {
  local dockerfile
  dockerfile="$(dirname "${BASH_SOURCE[0]}")/Dockerfile"
  echo "strata-dsv41-build:$FLAVOR_TAG-$(sha256sum "$dockerfile" | cut -c1-12)"
}

# selected_flavors:the flavors named by $CUDA_FLAVOR, or all of them when it is unset or empty
selected_flavors() {
  if [[ -n "${CUDA_FLAVOR:-}" ]]; then
    flavor_config "$CUDA_FLAVOR" >/dev/null || return 1
    echo "$CUDA_FLAVOR"
  else
    printf '%s\n' "${DSV41_FLAVORS[@]}"
  fi
}
