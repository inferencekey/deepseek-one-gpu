#!/usr/bin/env bash
# hf-download.sh MANIFEST DST JSON: the model's files of MANIFEST (pod/hf_model.py manifest: the HF repo at a pinned
# commit, each file's size and LFS sha256) downloaded into DST by aria2c, every file at once with 16 parallel HTTP
# range connections each (aria2c's per-server maximum), then each file's size and sha256 against the manifest
# (hf_model.py check); the download's bytes, seconds and GB/s and the check's in JSON. Exits non-zero on a failed
# download or a mismatch. Run by e07-run.sh on a pod without the model's network volume. HF_TOKEN (the environment, else ~/.hf-token) is optional and only goes into aria2c's input file
# (mode 600, deleted on exit): never on a command line, never logged.
# The throughput guard, with HF_GUARD_END_BY set (unix seconds: the session's deadline less
# the rest of its plan, HF_GUARD_REST_S seconds, which e07-run.sh passes): HF_GUARD_AFTER_S (default 180) seconds into
# the download, the bytes on disk give its rate; when the projected end passes HF_GUARD_END_BY, aria2c is stopped,
# the partial files deleted, "hf-download.sh: download too slow: X GB/s ..." printed and recorded in JSON (too_slow),
# and the exit status is 3, so the session pulls and deletes the pod instead of spending its time on the download.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="$1" dst="$2" json="$3"
hf_model="$here/hf_model.py"
token="${HF_TOKEN:-$(cat ~/.hf-token 2>/dev/null || true)}"
mkdir -p "$dst"
input="$(mktemp)"
trap 'rm -f "$input"' EXIT
chmod 600 "$input"
n=0 bytes=0
while read -r url name size; do
  { echo "$url"; echo "  out=$name"; [[ -z "$token" ]] || echo "  header=Authorization: Bearer $token"; } >> "$input"
  n=$(( n + 1 )) bytes=$(( bytes + size ))
done < <(python3 "$hf_model" urls "$manifest")
echo "hf-download.sh: $n files, $bytes bytes, $([[ -n "$token" ]] && echo "with HF_TOKEN" || echo "anonymous")"
t0=$(date +%s.%N)
aria2c --input-file="$input" --dir="$dst" --max-concurrent-downloads="$n" --max-connection-per-server=16 --split=16 \
  --min-split-size=64M --file-allocation=trunc --max-tries=10 --retry-wait=5 --timeout=60 --auto-file-renaming=false \
  --allow-overwrite=true --console-log-level=warn --summary-interval=60 --show-console-readout=false &
aria=$!
if [[ -n "${HF_GUARD_END_BY:-}" ]]; then
  after="${HF_GUARD_AFTER_S:-180}"
  while kill -0 "$aria" 2>/dev/null && (( $(date +%s) - ${t0%.*} < after )); do sleep 1; done
  if kill -0 "$aria" 2>/dev/null; then
    # the bytes written so far: the blocks the files hold (--file-allocation=trunc sets each size up front, sparse)
    got="$(du -s --block-size=1 "$dst" | cut -f1)"
    verdict="$(python3 - "$got" "$bytes" "$t0" "$HF_GUARD_END_BY" "${HF_GUARD_REST_S:-0}" <<'EOF'
import sys, time
got, total, t0, end_by, rest = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
now = time.time()
rate = got / (now - t0)
eta = now + (total - got) / rate if rate > 0 else float("inf")
left = "never" if eta == float("inf") else f"{eta - end_by:.0f} s past" if eta > end_by else f"{end_by - eta:.0f} s before"
print(f"{'slow' if eta > end_by else 'ok'} {rate / 1e9:.3f} GB/s ({got / 1e9:.1f} of {total / 1e9:.1f} GB in {now - t0:.0f} "
      f"s): the projected end is {left} the time the rest of the plan ({rest} s) leaves for it")
EOF
)"
    echo "hf-download.sh: the guard after $after s: ${verdict%% *}, ${verdict#* }"
    if [[ "$verdict" == slow* ]]; then
      kill "$aria" 2>/dev/null || true
      wait "$aria" 2>/dev/null || true
      rm -f -- "${dst:?}"/*.gguf "${dst:?}"/*.aria2
      echo "hf-download.sh: download too slow: ${verdict#slow }"
      python3 -c "import json, sys; json.dump({'too_slow': True, 'gb_per_s': float(sys.argv[2]), 'guard': sys.argv[3]},
open(sys.argv[1], 'w'), indent=1)" "$json" "$(cut -d' ' -f2 <<<"$verdict")" "${verdict#slow }"
      exit 3
    fi
  fi
fi
wait "$aria"
t1=$(date +%s.%N)
rm -f "$dst"/*.aria2
python3 "$hf_model" check "$manifest" "$dst" --json "$json.check" && rc=0 || rc=$?
python3 - "$json" "$manifest" "$bytes" "$t0" "$t1" <<'EOF'
import json, sys
out, manifest, nbytes, t0, t1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
m, c = json.load(open(manifest)), json.load(open(out + ".check"))
secs = t1 - t0
r = {"repo": m["repo"], "revision": m["revision"], "files": len(m["files"]), "bytes": nbytes,
     "download_s": round(secs, 1), "gb_per_s": round(nbytes / 1e9 / secs, 3), "check_s": c["check_s"], "ok": c["ok"],
     "sha256": c["files"]}
json.dump(r, open(out, "w"), indent=1)
print(f"hf-download.sh: {nbytes / 1e9:.1f} GB in {secs:.0f} s = {r['gb_per_s']} GB/s; size and sha256 check "
      f"{c['check_s']} s: {'ok' if c['ok'] else 'FAIL'}")
EOF
rm -f "$json.check"
exit "$rc"
