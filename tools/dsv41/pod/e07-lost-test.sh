#!/usr/bin/env bash
# The lost-session check, run locally (on 2026-10-18 RunPod removed a cdet pod at 17:36 UTC mid
# lm-eval, after 7 steps exit 0, and the session's one pull at the end brought nothing back): no pod, no RunPod, no
# money. The real tools/dsv41/pod/e07.sh session (E07_PLAN=cdet on the volume's path, NETWORK_VOLUME=testvolume01, QUALIFY=0: the CUDA probe alone)
# against a fake RunPod (RUNPOD_API), ssh and scp stubs that run the pod's commands in a directory (q02-test.sh
# section 6's way). The fake pod's run finishes one step per poll, each step's outputs the real ones of the 2026-10-17
# cdet session (CDET_RUN's out/): setup (out/env.log, out/setup.log), then load-short (out/load-short.log,
# out/cbatch/load-short.json, and out/big.log, 100 MiB, above pull_steps' cap), then steady-c4 (out/steady-c4.log,
# out/cbatch/steady-c4.json), and right after the poll that saw steady-c4 the pod is gone: every ssh and scp exit 255,
# GET and DELETE answer 404 (the 2026-10-18 pod). POLL_S=1, POLL_FAILS=3. Checks:
#   1. e07.sh exits 1 (the run reported no exit status), its log has a mid-run pull after setup and after load-short,
#      big.log named as left for the final pull, the failed pull after steady-c4 a WARNING, the final pull's "nothing
#      pulled", the delete's 404
#   2. OUT/out has both steps' outputs, byte for byte the pod's, out/steps.log their two lines, no big.log and
#      nothing of steady-c4
#   3. e07.log and OUT/session-lost say "lost after step load-short", the steps finished and pulled (setup,
#      load-short), the one a poll saw and no pull brought (steady-c4), and the plan's steps with no line pulled
#      (volume-before first, volume-after last)
#   4. e07_report.py wrote the cdet sheet with the "Session lost" section, the two steps and load-short's run
# Exits 1 if any check fails.
#
#   BUILD_TAG=-c04 COUNTS=<counts.json> bash tools/dsv41/pod/e07-lost-test.sh
#
# The payload is e07.sh's own (BUILD_ROOT's tarballs, conversations, HF manifest: as its --payload-only). CDET_RUN
# defaults to $BUILD_ROOT/engine/e07a-pod/20261017T041837Z: the results directory of a real cdet session (the
# 2026-10-17 one), whose out/ logs the fake pod replays; the test exits 2 without it. E07LOST_KEEP=1 keeps the
# test's files.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/dsv41-build}"
CDET_RUN="${CDET_RUN:-$BUILD_ROOT/engine/e07a-pod/20261017T041837Z}"
src="$CDET_RUN/out"
for f in env.log setup.log plan.log load-short.log cbatch/load-short.json steady-c4.log cbatch/steady-c4.json; do
  [[ -f "$src/$f" ]] || { echo "e07-lost-test.sh: no $src/$f" >&2; exit 2; }
done
tmp="$(mktemp -d)"
cleanup() { [[ -n "${E07LOST_KEEP:-}" ]] && echo "e07-lost-test.sh: kept $tmp" || rm -rf -- "${tmp:?}"; }
trap cleanup EXIT
failed=0
check() {
  if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; failed=$((failed + 1)); fi
}
q="$tmp/fake" st="$tmp/fake/state"
mkdir -p "$q/bin" "$st"
cat > "$q/runpod_api.py" <<'EOF'
#!/usr/bin/env python3
"""e07-lost-test.sh's RunPod (pod-session.sh's RUNPOD_API): price lists 2.09 USD/h; create makes lost-pod-1 on SSH
port 22001; get answers RUNNING with 127.0.0.1 and that port until E07LOST_STATE/lost exists, then 404 as RunPod did
on 2026-10-18 (exit 1); delete answers 204, after the loss 404 (exit 0, as runpod_api.py: a missing pod is deleted)."""
import json
import os
import sys

st = os.environ["E07LOST_STATE"]
lost = os.path.exists(os.path.join(st, "lost"))
cmd, args = sys.argv[1], sys.argv[2:]
opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
if cmd == "price":
    print(json.dumps({"gpu": opt["--gpu"], "cloud": opt["--cloud"], "price": 2.09, "availability": "High"}))
elif cmd == "create":
    print(json.dumps({"id": "lost-pod-1", "costPerHr": 2.59, "gpu": opt["--gpu"], "cloud": opt["--cloud"],
                      "dc": opt.get("--dc") or "any", "machineId": "lost-machine-1", "memoryInGb": 377}))
elif cmd == "get":
    if lost:
        print(f'runpod_api.py: get {args[0]}: HTTP 404 {{"error":"pod not found","status":404}}', file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"id": args[0], "desiredStatus": "RUNNING", "publicIp": "127.0.0.1", "port22": 22001,
                      "memoryInGb": 377}))
elif cmd == "delete":
    with open(os.path.join(st, "deleted"), "a") as f:
        f.write(args[0] + "\n")
    print(f"deleted {args[0]} (HTTP {404 if lost else 204})")
else:
    sys.exit(f"runpod_api.py (e07-lost-test.sh fake): {cmd}?")
EOF
cat > "$q/bin/ssh" <<'EOF'
#!/usr/bin/env bash
# e07-lost-test.sh's ssh: the pod is E07LOST_STATE/root (its /root/). Once E07LOST_STATE/lost exists every call exits
# 255. `true` answers; the CUDA probe says ok; the detached start writes out/plan.log and runs nothing; each poll
# first finishes the run's next step (setup, load-short, steady-c4, each with its outputs from E07LOST_SRC), the
# third one then losing the pod; every other command (the mid-run pull's tar) runs locally with /root/ in the pod's
# dir
while [[ "${1:-}" == -* ]]; do
  case "$1" in -p) shift 2 ;; -i|-o) shift 2 ;; *) shift ;; esac
done
shift   # root@<ip>
cmd="$*" r="$E07LOST_STATE/root" o="$E07LOST_STATE/root/e07/out" s="$E07LOST_SRC"
echo "ssh: $cmd" >> "$E07LOST_STATE/ssh.log"
[[ ! -e "$E07LOST_STATE/lost" ]] || { echo "ssh: connect to host 127.0.0.1 port 22001: Connection refused (e07-lost-test.sh)" >&2; exit 255; }
mkdir -p "$r"
case "$cmd" in
  true) exit 0 ;;
  *"/root/dsv41_cuda_probe"*) echo "dsv41_cuda_probe: ok, the e07-lost-test.sh stub (sm_120), 1 device(s)"; exit 0 ;;
  *"nohup setsid"*)
    mkdir -p "$o/cbatch"
    cp "$s/plan.log" "$o/"
    exit 0 ;;
  *"|| echo running; tail -n 1 "*)
    n=$(( $(cat "$E07LOST_STATE/polls" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$E07LOST_STATE/polls"
    case "$n" in
      1) cp "$s/env.log" "$s/setup.log" "$o/"
         echo "step setup: exit 0, 7 s" >> "$o/steps.log" ;;
      2) cp "$s/load-short.log" "$o/"; cp "$s/cbatch/load-short.json" "$o/cbatch/"
         truncate -s 100M "$o/big.log"
         echo "step load-short: exit 0, 22 s" >> "$o/steps.log" ;;
      3) cp "$s/steady-c4.log" "$o/"; cp "$s/cbatch/steady-c4.json" "$o/cbatch/"
         echo "step steady-c4: exit 0, 275 s" >> "$o/steps.log"
         # this poll answers, then the pod is gone (RunPod's removal)
         bash -c "${cmd//\/root\//$r/}"
         touch "$E07LOST_STATE/lost"
         exit 0 ;;
    esac ;;
esac
exec bash -c "${cmd//\/root\//$r/}"
EOF
cat > "$q/bin/scp" <<'EOF'
#!/usr/bin/env bash
# e07-lost-test.sh's scp: into or out of the pod's dir (e07-lost-test.sh's ssh), exit 255 once the pod is lost; a
# payload tarball is recorded, not copied
while [[ "${1:-}" == -* ]]; do
  case "$1" in -P|-i|-o) shift 2 ;; *) shift ;; esac
done
[[ ! -e "$E07LOST_STATE/lost" ]] || { echo "scp: connect to host 127.0.0.1 port 22001: Connection refused (e07-lost-test.sh)" >&2; exit 255; }
r="$E07LOST_STATE/root"
mkdir -p "$r"
case "$2" in
  root@*:*) echo "${2#*:} $(basename "$1")" >> "$E07LOST_STATE/uploads"
    [[ "$1" == *.tar.gz ]] || cp "$1" "$r/${2#*:/root/}" ;;
  *) cp "$r/${1#*:/root/}" "$2" ;;
esac
EOF
chmod +x "$q/bin"/* "$q/runpod_api.py"
: > "$q/key"; : > "$q/key.pub"
run="$tmp/run/20261018T170305Z" sheets="$tmp/sheets"
rc=0
( export PATH="$q/bin:$PATH" E07LOST_STATE="$st" E07LOST_SRC="$src"
  E07_PLAN=cdet NETWORK_VOLUME=testvolume01 QUALIFY=0 POLL_S=1 POLL_FAILS=3 RUNPOD_API="$q/runpod_api.py" RUNPOD_API_KEY=e07-lost-test-fake \
    SSH_KEY="$q/key" OUT="$run" SHEET_ROOT="$sheets" timeout 900 bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" \
    > "$tmp/e07.out" 2>&1 < /dev/null ) || rc=$?
l="$run/e07.log"
grep -E "pulled out/|mid-run|on the pod: |poll failed|FAIL: the pod|nothing pulled|deleted|lost|finished and|seen by|plan's steps" "$l" |
  sed 's/^/  /' | cut -c1-220 || true

echo "--- 1. the session: the mid-run pulls, the loss, the final pull's nothing, the delete"
check "e07.sh exits 1: the run reported no exit status" \
  '(( rc == 1 )) && grep -q "FAIL: e07-run.sh reported no exit status" "$l" && grep -q "FAIL: the pod did not answer 3 polls in a row" "$l"'
check "a mid-run pull after setup and after load-short, each with its time and size" \
  'grep -q "\] pulled out/ mid-run after .step setup: exit 0, 7 s.: [0-9]* s, [0-9]* KiB$" "$l" &&
   grep -q "\] pulled out/ mid-run after .step load-short: exit 0, 22 s.: [0-9]* s, [0-9]* KiB$" "$l"'
check "big.log (100 MiB) named as left for the final pull, once" \
  '(( $(grep -c "\] pull mid-run: above 64 MiB, left for the final pull: out/big.log$" "$l") == 1 ))'
check "the pull after steady-c4 failed as a WARNING, the polls went on until POLL_FAILS" \
  'grep -q "\] on the pod: step steady-c4: exit 0, 275 s$" "$l" &&
   grep -q "WARNING: pulling out/ mid-run failed (exit 255, [0-9]* s): ssh: connect to host 127.0.0.1 port 22001: Connection refused (e07-lost-test.sh); the run goes on$" "$l" &&
   (( $(grep -c "WARNING: poll failed (exit 255)" "$l") == 3 ))'
check "the final pull brought nothing; the delete answered 404" \
  'grep -q "WARNING: archiving /root/e07/out on the pod failed (exit 255): nothing pulled$" "$l" && [[ ! -e "$run/out-logs.tar.gz" ]] &&
   grep -q "^deleted lost-pod-1 (HTTP 404)$" "$tmp/e07.out" && [[ "$(cat "$st/deleted")" == lost-pod-1 ]]'

echo "--- 2. OUT/out: both steps' outputs, nothing past them"
# same_out FILE...: each of OUT/out's files is the fake pod's source byte for byte
same_out() { local f; for f in "$@"; do cmp -s "$src/$f" "$run/out/$f" || return 1; done; }
check "setup's and load-short's outputs are the pod's byte for byte; out/steps.log the two lines" \
  'same_out env.log setup.log plan.log load-short.log cbatch/load-short.json &&
   [[ "$(cat "$run/out/steps.log")" == "$(printf "step setup: exit 0, 7 s\nstep load-short: exit 0, 22 s")" ]]'
check "no big.log, no steady-c4.json, no mid-run archive left behind" \
  '[[ ! -e "$run/out/big.log" && ! -e "$run/out/steady-c4.log" && ! -e "$run/out/cbatch/steady-c4.json" ]] && ! ls "$run"/out-steps.tar.gz* > /dev/null 2>&1'

echo "--- 3. e07.log and OUT/session-lost: the step it was lost after, the steps pulled and lost"
sl="$run/session-lost"
check "lost after step load-short, setup and load-short pulled, steady-c4 seen and not pulled" \
  'grep -q "^lost after step load-short: the final pull brought nothing back, out/ is the last mid-run pull.s (out/steps.log.s last line: step load-short: exit 0, 22 s)$" "$sl" &&
   grep -qx "finished and pulled (out/steps.log): setup load-short" "$sl" &&
   grep -qx "seen by a poll after the last pull, not pulled: step steady-c4: exit 0, 275 s" "$sl" &&
   grep -q "\] lost after step load-short: " "$l" && grep -q "\] seen by a poll after the last pull, not pulled: step steady-c4" "$l"'
check "the plan's steps with no line pulled: volume-before first, steady-c4 among them, volume-after last" \
  'left="$(sed -n "s/^the plan.s steps with no line pulled (lost or never run): //p" "$sl")" &&
   [[ "$left" == "volume-before "* && "$left" == *" volume-after" && " $left " == *" steady-c4 "* &&
      " $left " != *" setup "* && " $left " != *" load-short "* ]] && grep -q "\] the plan.s steps with no line pulled" "$l"'

echo "--- 4. the cdet sheet from what was pulled"
sheet="$sheets/2026-10-18-cdet/README.md"
check "the sheet has the Session lost section with session-lost's lines and the two steps" \
  '[[ -f "$sheet" ]] && grep -q "^## Session lost$" "$sheet" && grep -q "^- lost after step load-short: " "$sheet" &&
   grep -q "^- seen by a poll after the last pull, not pulled: step steady-c4" "$sheet" &&
   grep -qx -- "- step setup: exit 0, 7 s" "$sheet" && grep -qx -- "- step load-short: exit 0, 22 s" "$sheet" &&
   ! grep -q "WARNING: e07_report.py failed" "$l" && cmp -s "$src/cbatch/load-short.json" "$sheets/2026-10-18-cdet/logs/cbatch/load-short.json" &&
   [[ -f "$sheets/2026-10-18-cdet/logs/session-lost" ]]'
(( failed == 0 )) || { echo "  the tail of e07.out:"; tail -n 12 "$tmp/e07.out" | sed 's/^/    /'; }

echo "e07-lost-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
