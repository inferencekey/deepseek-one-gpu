#!/usr/bin/env bash
# The batched decode's promoted-request regression: the planner's unit test on the dev VM with the host compiler (no
# CUDA, no Docker: src/dsv41/engine/hot_policy.cpp is plain C++, C++20 as the CMake build, the fork's vendored
# nlohmann/json for the replay), then the same test on a copy of hot_policy.cpp with the fix taken out
# (HotPrompt::begin_request's body emptied: only a prefill opens a request's prompt, the engine before the fix):
# the test must pass on the source and fail on the copy, on the promoted batch's check alone.
#
#   bash tools/dsv41/engine/hot_policy_test.sh     # CXX (default c++) picks the compiler, FORK_DIR the fork's checkout
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
FORK_DIR="${FORK_DIR:-$HOME/llama-dsv41}"
CXX="${CXX:-c++}"
[[ -f "$FORK_DIR/vendor/nlohmann/json.hpp" ]] ||
  { echo "hot_policy_test.sh: no nlohmann/json.hpp under FORK_DIR=$FORK_DIR/vendor" >&2; exit 1; }
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

build() {   # build HOT_POLICY_CPP OUT (hot_policy.cpp's own -ffp-contract=off, as src/dsv41/engine/CMakeLists.txt)
  "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$STRATA_DIR/include" -I"$FORK_DIR/vendor" "$1" \
    "$STRATA_DIR/tests/dsv41/engine/hot_policy_test.cpp" -o "$2"
}

echo "--- hot_policy_test.cpp on src/dsv41/engine/hot_policy.cpp ($("$CXX" --version | head -n 1))"
build "$STRATA_DIR/src/dsv41/engine/hot_policy.cpp" "$tmp/hot_policy_test"
rc=0; "$tmp/hot_policy_test" > "$tmp/ok.txt" || rc=$?
grep -E "^(FAIL|ok   promoted batch)" "$tmp/ok.txt" || true
[[ $rc == 0 ]] && grep -qx "dsv41_engine_hot_policy_test: PASS" "$tmp/ok.txt" ||
  { echo "hot_policy_test.sh: FAIL: the test fails on the source (exit $rc)"; exit 1; }

echo "--- the same test without the fix (HotPrompt::begin_request emptied in a copy of hot_policy.cpp: as before the fix)"
sed '/^void HotPrompt::begin_request() {$/{n;N;d}' "$STRATA_DIR/src/dsv41/engine/hot_policy.cpp" > "$tmp/hot_policy_unfixed.cpp"
(( $(grep -A1 '^void HotPrompt::begin_request() {$' "$tmp/hot_policy_unfixed.cpp" | grep -cx '}') == 1 )) &&
  (( $(diff "$STRATA_DIR/src/dsv41/engine/hot_policy.cpp" "$tmp/hot_policy_unfixed.cpp" | grep -c '^<') == 2 )) ||
  { echo "hot_policy_test.sh: FAIL: HotPrompt::begin_request is no longer two lines in one place: update the mutation"; exit 1; }
build "$tmp/hot_policy_unfixed.cpp" "$tmp/hot_policy_test_unfixed"
rc=0; "$tmp/hot_policy_test_unfixed" > "$tmp/unfixed.txt" || rc=$?
grep "^FAIL" "$tmp/unfixed.txt" | sed 's/^/  without the fix: /' || true
n_fail="$(grep -c "^FAIL" "$tmp/unfixed.txt" || true)"
n="$(grep -c "^FAIL promoted batch: .*(0 swaps in the [0-9]* steps before, 0 in its [0-9]*)" "$tmp/unfixed.txt" || true)"
[[ $rc == 1 && $n_fail == 1 && $n == 1 ]] ||
  { echo "hot_policy_test.sh: FAIL: the promoted batch's frozen hot set was not caught alone (exit $rc, $n_fail FAIL lines)"; exit 1; }
echo "hot_policy_test.sh: PASS (the test passes on the source and fails with the promoted request's prompt left open)"
