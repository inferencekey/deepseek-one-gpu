#!/usr/bin/env bash
# The host sampler's unit test on the dev VM with the host
# compiler (no CUDA, no Docker: src/dsv41/engine/sampler.cpp, session.cpp and batch.cpp, whose batch_argmax the
# Session's greedy pick takes, are plain C++; C++20 as the CMake build,
# for core/conversation_cache.hpp's match the Session reuses), then the same test on a
# copy of sampler.cpp whose generator ignores the request's seed (`rng_(seed_)` -> `rng_(0)`, every request drawing
# the same stream): the test must pass on the source and fail on the copy, on the seeds' check.
#
#   bash tools/dsv41/engine/sampler_test.sh        # CXX (default c++) picks the compiler
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
CXX="${CXX:-c++}"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

build() {   # build SAMPLER_CPP OUT
  "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -I"$STRATA_DIR/include" "$1" "$STRATA_DIR/src/dsv41/engine/session.cpp" \
    "$STRATA_DIR/src/dsv41/engine/batch.cpp" "$STRATA_DIR/tests/dsv41/engine/sampler_test.cpp" -o "$2"
}

echo "--- sampler_test.cpp on src/dsv41/engine/sampler.cpp ($("$CXX" --version | head -n 1))"
build "$STRATA_DIR/src/dsv41/engine/sampler.cpp" "$tmp/sampler_test"
rc=0; "$tmp/sampler_test" > "$tmp/ok.txt" || rc=$?
cat "$tmp/ok.txt"
[[ $rc == 0 ]] && grep -qx "dsv41_engine_sampler_test: PASS" "$tmp/ok.txt" ||
  { echo "sampler_test.sh: FAIL: the test fails on the source (exit $rc)"; exit 1; }

echo "--- the same test with the seeding broken on purpose (rng_(seed_) -> rng_(0) in a copy of sampler.cpp)"
sed 's/rng_(seed_)/rng_(0)/' "$STRATA_DIR/src/dsv41/engine/sampler.cpp" > "$tmp/sampler_broken.cpp"
(( $(grep -c 'rng_(0)' "$tmp/sampler_broken.cpp") == 1 )) ||
  { echo "sampler_test.sh: FAIL: sampler.cpp no longer seeds its generator with rng_(seed_): update the mutation"; exit 1; }
build "$tmp/sampler_broken.cpp" "$tmp/sampler_test_broken"
rc=0; "$tmp/sampler_test_broken" > "$tmp/broken.txt" || rc=$?
grep "^FAIL" "$tmp/broken.txt" | sed 's/^/  broken: /' || true
[[ $rc == 1 ]] && grep -q "^FAIL seeds 1234 and 1235: 0 of 100 draws differ" "$tmp/broken.txt" ||
  { echo "sampler_test.sh: FAIL: the broken seeding was not caught (exit $rc)"; tail -n 5 "$tmp/broken.txt"; exit 1; }
echo "sampler_test.sh: PASS (the test passes on the source and fails with the seed ignored)"
