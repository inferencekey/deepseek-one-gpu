#!/usr/bin/env bash
# The Session's unit test on the dev VM with the host compiler (no
# CUDA, no Docker: src/dsv41/engine/session.cpp and batch.cpp are plain C++, C++20 as the CMake build), then the same
# test on a copy of session.cpp with the fix taken out (`feed_unfed = ...` -> `feed_unfed = false && ...`: a promoted
# request's one token left read by prefill(T=1), the code before the fix): the test must pass on the source and fail
# on the copy, on the promotion's checks.
#
#   bash tools/dsv41/engine/session_test.sh        # CXX (default c++) picks the compiler
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
CXX="${CXX:-c++}"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

build() {   # build SESSION_CPP OUT
  "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -I"$STRATA_DIR/include" "$1" "$STRATA_DIR/src/dsv41/engine/batch.cpp" \
    "$STRATA_DIR/src/dsv41/engine/sampler.cpp" "$STRATA_DIR/tests/dsv41/engine/session_test.cpp" -o "$2"
}

echo "--- session_test.cpp on src/dsv41/engine/session.cpp ($("$CXX" --version | head -n 1))"
build "$STRATA_DIR/src/dsv41/engine/session.cpp" "$tmp/session_test"
rc=0; "$tmp/session_test" > "$tmp/ok.txt" || rc=$?
cat "$tmp/ok.txt"
[[ $rc == 0 ]] && grep -qx "dsv41_engine_session_test: PASS" "$tmp/ok.txt" ||
  { echo "session_test.sh: FAIL: the test fails on the source (exit $rc)"; exit 1; }

echo "--- the same test without the fix (feed_unfed = false && ... in a copy of session.cpp: prefill(T=1) as before)"
sed 's/const bool feed_unfed = /const bool feed_unfed = false \&\& /' "$STRATA_DIR/src/dsv41/engine/session.cpp" \
  > "$tmp/session_unfixed.cpp"
(( $(grep -c 'feed_unfed = false &&' "$tmp/session_unfixed.cpp") == 1 )) ||
  { echo "session_test.sh: FAIL: session.cpp no longer sets feed_unfed in one place: update the mutation"; exit 1; }
build "$tmp/session_unfixed.cpp" "$tmp/session_test_unfixed"
rc=0; "$tmp/session_test_unfixed" > "$tmp/unfixed.txt" || rc=$?
grep "^FAIL" "$tmp/unfixed.txt" | sed 's/^/  without the fix: /' || true
n="$(grep -c "^FAIL promoted after [125] (the BGEN of its prompt and tokens): .* no prefill call (saw: prefill 1)$" "$tmp/unfixed.txt" || true)"
[[ $rc == 1 && $n == 3 ]] ||
  { echo "session_test.sh: FAIL: the promotion read by prefill(T=1) was not caught (exit $rc, $n of 3 promotions)"; exit 1; }
echo "session_test.sh: PASS (the test passes on the source and fails with the promotion's token read by prefill)"
