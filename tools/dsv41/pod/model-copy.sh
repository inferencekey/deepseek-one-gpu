#!/usr/bin/env bash
# model-copy.sh SRC DST: every SRC/*.gguf copied to DST with 4 parallel dd streams per shard (28 for the 7 Q2_K
# shards: 4 min 17 s from a RunPod network volume on an A100 pod), then every
# shard's size against SRC's and its content at fixed offsets (cmp of the first and last MiB and of the MiB on each
# side of every stream's boundary); exits non-zero on any failed dd stream or mismatch. SRC is only read (dd if=,
# stat, cmp): the model's network volume is never written. Run by tools/dsv41/pod/t04a.sh and e07-run.sh.
set -e
mkdir -p "$2"; PARTS=4; BS=16777216; MIB=1048576
fail() { echo "model-copy.sh: $1" >&2; exit 1; }
pids=()
for f in "$1"/*.gguf; do
  out="$2/$(basename "$f")"; sz=$(stat -c %s "$f"); truncate -s "$sz" "$out"
  chunk=$(( (sz + PARTS * BS - 1) / (PARTS * BS) ))
  for p in $(seq 0 $((PARTS - 1))); do
    dd if="$f" of="$out" bs=$BS skip=$((p * chunk)) seek=$((p * chunk)) count=$chunk conv=notrunc status=none &
    pids+=("$!")
  done
done
# a bare `wait` returns 0 whatever its children did: each stream's own status
for pid in "${pids[@]}"; do wait "$pid" || fail "a dd stream (pid $pid) failed"; done
for f in "$1"/*.gguf; do
  out="$2/$(basename "$f")"; a=$(stat -c %s "$f"); b=$(stat -c %s "$out")
  echo "$(basename "$f") volume $a local $b"; [ "$a" = "$b" ] || fail "$(basename "$f"): size $b, the volume's $a"
  # a stream that wrote nothing leaves the truncated file's zeros at its ends: check both sides of every boundary
  chunk=$(( (a + PARTS * BS - 1) / (PARTS * BS) )); offs="0 $(( a > MIB ? a - MIB : 0 ))"
  for p in $(seq 1 $((PARTS - 1))); do
    s=$(( p * chunk * BS )); if (( s < a )); then offs="$offs $(( s >= MIB ? s - MIB : 0 )) $s"; fi
  done
  for o in $offs; do
    cmp -s -i "$o" -n "$MIB" "$f" "$out" || fail "$(basename "$f"): the MiB at byte $o differs from the volume's"
  done
  echo "$(basename "$f") content equal at bytes $offs"
done
