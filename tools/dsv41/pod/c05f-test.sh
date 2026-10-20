#!/usr/bin/env bash
# The CPU experts floor check, run locally: no pod, no RunPod, no money. The real tools/dsv41/pod/e07.sh
# session (E07_PLAN=cfinal on the volume's path) against a fake RunPod (RUNPOD_API), ssh and scp stubs that run the
# pod's commands in a directory per pod (q02-test.sh section 6's way), a dsv41_machine_probe stub that always passes
# and, in place of the strata tarball's dsv41_cpu_experts_probe that pod-session.sh uploads, a stub that prints the
# real probe's lines with its machine's rate: a machine named slow-cpu-* reads at 44.7 GB/s (the cfinal rerun's EPYC
# 9355P, 2026-10-19), any other at 100.4 (a good session's EPYC 9655). nvidia-smi topo -m's stub puts GPU0 on NUMA
# node 1. Checks:
#   1. floor: machines slow-cpu-1 then good-2 at the default floor: pod 1 passes the machine probe, then its CPU
#      experts probe runs with --threads 16 --min-gbs 71 --numa-node 1 and fails; e07.log has "qualification: CPU
#      experts 44.7 GB/s < 71 ..."; pod 1 is deleted, its cost in cost.txt; pod 2 passes and
#      only pod 2 gets the payload; the session exits 0
#   2. the floor sheet's machine table has the CPU experts column: pod 1 "**44.7 FAIL** (>=71)" and rejected naming
#      cpu_experts_gbs, pod 2 100.4 and passed
#   3. override: CPU_EXPERTS_FLOOR_GBS=40 on slow-cpu-1: its 44.7 passes, the payload goes to pod 1
#   4. e07.sh refuses CPU_EXPERTS_FLOOR_GBS=0 (exit 2) and its dry run prints the CPU experts step
#   5. BAD_MACHINES="other,known-bad-1" on machines known-bad-1 then good-2: pod 1 is deleted at once, before
#      any probe, e07.log saying it was excluded by BAD_MACHINES, and pod 2 gets the payload; the dry run prints the
#      list and e07.sh refuses an id that is not one (exit 2)
#   6. a run whose steps.log says "stop: download too slow" makes e07.log say "slow download machine: <id>"
# Exits 1 if any check fails.
#
#   BUILD_TAG=-c04 COUNTS=<counts.json> bash tools/dsv41/pod/c05f-test.sh
#
# The payload is e07.sh's own (BUILD_ROOT's tarballs, conversations, HF manifest: as its --payload-only), so the
# tarballs must carry dsv41_cpu_experts_probe. C05F_KEEP=1 keeps the test's files.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR="$(cd "$here/../../.." && pwd)"
tmp="$(mktemp -d)"
cleanup() { [[ -n "${C05F_KEEP:-}" ]] && echo "c05f-test.sh: kept $tmp" || rm -rf -- "${tmp:?}"; }
trap cleanup EXIT
failed=0
check() {
  if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; failed=$((failed + 1)); fi
}
q="$tmp/fake"
mkdir -p "$q/bin" "$q/probe/cu130/machine-probe" "$q/probe/cu128/machine-probe"
cat > "$q/runpod_api.py" <<'EOF'
#!/usr/bin/env python3
"""c05f-test.sh's RunPod (pod-session.sh's RUNPOD_API): price lists 2.09 USD/h; create takes the next machine of
C05F_STATE/machines (none left: exit 3, no stock), its pod c05f-pod-N on SSH port 22000 + N; get answers with
127.0.0.1 and that port; delete records the pod."""
import json
import os
import sys

st = os.environ["C05F_STATE"]
cmd, args = sys.argv[1], sys.argv[2:]
opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
if cmd == "price":
    print(json.dumps({"gpu": opt["--gpu"], "cloud": opt["--cloud"], "price": 2.09, "availability": "High"}))
elif cmd == "create":
    machines = open(os.path.join(st, "machines")).read().split()
    created = os.path.join(st, "created")
    n = len(open(created).read().splitlines()) if os.path.exists(created) else 0
    if n >= len(machines):
        print("no stock (c05f-test.sh fake)", file=sys.stderr)
        sys.exit(3)
    with open(created, "a") as f:
        f.write(f"c05f-pod-{n + 1} {machines[n]}\n")
    with open(os.path.join(st, f"port-{22001 + n}"), "w") as f:
        f.write(machines[n])
    print(json.dumps({"id": f"c05f-pod-{n + 1}", "costPerHr": 2.59, "gpu": opt["--gpu"], "cloud": opt["--cloud"],
                      "dc": opt.get("--dc") or "any", "machineId": machines[n], "memoryInGb": 377}))
elif cmd == "get":
    n = int(args[0].rsplit("-", 1)[1])
    print(json.dumps({"id": args[0], "desiredStatus": "RUNNING", "publicIp": "127.0.0.1", "port22": 22000 + n,
                      "memoryInGb": 377}))
elif cmd == "delete":
    with open(os.path.join(st, "deleted"), "a") as f:
        f.write(args[0] + "\n")
    print(f"deleted {args[0]} (HTTP 204, c05f-test.sh fake)")
else:
    sys.exit(f"runpod_api.py (c05f-test.sh fake): {cmd}?")
EOF
cat > "$q/bin/ssh" <<'EOF'
#!/usr/bin/env bash
# c05f-test.sh's ssh: the pod is C05F_STATE/root-<port> (its /root/), its machine C05F_STATE/port-<port>. `true`
# answers; the CUDA probe says ok; the run's detached start writes the finished run's exit status and steps.log
# (C05F_STOP set: that stop line, as e07-run.sh's stop writes it) instead of running it; every other command (the probes, the polls, the pull) runs locally with /root/ in the pod's
# dir and FAKE_MACHINE set
while [[ "${1:-}" == -* ]]; do
  case "$1" in -p) port="$2"; shift 2 ;; -i|-o) shift 2 ;; *) shift ;; esac
done
shift   # root@<ip>
cmd="$*" r="$C05F_STATE/root-$port"
mkdir -p "$r"
echo "ssh $port: $cmd" >> "$C05F_STATE/ssh.log"
case "$cmd" in
  true) exit 0 ;;
  *"/root/dsv41_cuda_probe"*) echo "dsv41_cuda_probe: ok, the c05f-test.sh stub (sm_120), 1 device(s)"; exit 0 ;;
  *"nohup setsid"*)
    mkdir -p "$r/e07/out"
    if [[ -n "${C05F_STOP:-}" ]]; then
      echo 1 > "$r/e07/out/e07-run.exit"
      printf '%s\ndone\n' "$C05F_STOP" > "$r/e07/out/steps.log"
    else
      echo 0 > "$r/e07/out/e07-run.exit"
      echo done > "$r/e07/out/steps.log"
    fi
    echo "plan: cfinal, readers: dense" > "$r/e07/out/plan.log"
    echo "$port" >> "$C05F_STATE/started"
    exit 0 ;;
esac
FAKE_MACHINE="$(cat "$C05F_STATE/port-$port")" exec bash -c "${cmd//\/root\//$r/}"
EOF
cat > "$q/bin/scp" <<'EOF'
#!/usr/bin/env bash
# c05f-test.sh's scp: into or out of the pod's dir (c05f-test.sh's ssh); a payload tarball is recorded, not copied;
# the strata tarball's dsv41_cpu_experts_probe goes up as C05F_CPU_STUB (the real one would measure this VM)
while [[ "${1:-}" == -* ]]; do
  case "$1" in -P) port="$2"; shift 2 ;; -i|-o) shift 2 ;; *) shift ;; esac
done
r="$C05F_STATE/root-$port"
mkdir -p "$r"
case "$2" in
  root@*:*) echo "$port ${2#*:} $(basename "$1")" >> "$C05F_STATE/uploads"
    if [[ "$(basename "$1")" == dsv41_cpu_experts_probe ]]; then cp "$C05F_CPU_STUB" "$r/${2#*:/root/}"
    elif [[ "$1" != *.tar.gz ]]; then cp "$1" "$r/${2#*:/root/}"; fi ;;
  *) cp "$r/${1#*:/root/}" "$2" ;;
esac
EOF
cat > "$q/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# c05f-test.sh's nvidia-smi: topo -m with GPU0 on NUMA node 1 (the columns of a one-GPU host's matrix)
[[ "$*" == "topo -m" ]] || { echo "nvidia-smi (c05f-test.sh stub): $*?" >&2; exit 1; }
printf '\tGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \t16-31\t1\t\tN/A\n'
EOF
cat > "$q/cpu-stub" <<'EOF'
#!/usr/bin/env python3
"""c05f-test.sh's dsv41_cpu_experts_probe: tests/dsv41/engine/cpu_experts_probe.cpp's lines with FAKE_MACHINE's rate
(slow-cpu-*: 44.7 GB/s, the cfinal rerun's EPYC 9355P; else 100.4, a good session's EPYC 9655) against --min-gbs; its
arguments recorded in C05F_STATE/cpu-args."""
import os
import sys

args = sys.argv[1:]
with open(os.path.join(os.environ["C05F_STATE"], "cpu-args"), "a") as f:
    f.write(os.environ["FAKE_MACHINE"] + ": " + " ".join(args) + "\n")
opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
rate = 44.7 if os.environ["FAKE_MACHINE"].startswith("slow-cpu") else 100.4
floor = float(opt["--min-gbs"])
ok = rate >= floor
print(f"dsv41_cpu_experts_probe: measure cpu_experts_gbs {rate:.1f} bound >={floor:g} {'ok' if ok else 'FAIL'} "
      f"(the c05f-test.sh stub, {opt['--threads']} pool threads, NUMA node {opt.get('--numa-node', 'none')})")
print(f"dsv41_cpu_experts_probe: machine CPU the c05f-test.sh stub of {os.environ['FAKE_MACHINE']}")
print("dsv41_cpu_experts_probe: PASS, the CPU experts rate within its bound" if ok else
      f"dsv41_cpu_experts_probe: FAIL, cpu_experts_gbs {rate:.1f} (bound >={floor:g})")
sys.exit(0 if ok else 1)
EOF
cat > "$q/probe/cu130/machine-probe/dsv41_machine_probe" <<'EOF'
#!/usr/bin/env bash
# c05f-test.sh's dsv41_machine_probe: a good RTX PRO 6000 machine (q02-test.sh's numbers), every measure within its bound
for m in "gpu_read_gbs 1520.3" "h2d_gbs 26.5" "d2h_gbs 26.4" "doorbell_median_us 8.2" "host_read_gbs 375.2"; do
  echo "dsv41_machine_probe: measure $m bound none info (the c05f-test.sh stub)"
done
echo "dsv41_machine_probe: machine NVIDIA RTX PRO 6000 Blackwell Server Edition, the c05f-test.sh stub of $FAKE_MACHINE"
echo "dsv41_machine_probe: PASS, every measure within its bound"
EOF
chmod +x "$q/bin"/* "$q/runpod_api.py" "$q/cpu-stub" "$q/probe/cu130/machine-probe/dsv41_machine_probe"
cp "$q/probe/cu130/machine-probe/dsv41_machine_probe" "$q/probe/cu128/machine-probe/"
src_sha="$(git -C "$STRATA_DIR" show HEAD:tests/dsv41/engine/machine_probe.cu | sha256sum | cut -d' ' -f1)"
for t in "cu130|120" "cu128|80;89"; do
  printf 'source: tests/dsv41/engine/machine_probe.cu sha256 %s\ncommit: c05f-test.sh stub\nnvcc: none (a stub)\narchs: %s\n%s  dsv41_machine_probe\n' \
    "$src_sha" "${t#*|}" "$(sha256sum "$q/probe/${t%|*}/machine-probe/dsv41_machine_probe" | cut -d' ' -f1)" > "$q/probe/${t%|*}/machine-probe/MANIFEST"
done
: > "$q/key"; : > "$q/key.pub"

# session NAME MACHINES [VAR=VALUE...]: e07.sh's cfinal session on the volume against the fakes, its exit status in
# $tmp/NAME/rc
session() {
  local name="$1" machines="$2"; shift 2
  mkdir -p "$tmp/$name"
  echo "$machines" > "$tmp/$name/machines"
  touch "$tmp/$name/uploads"
  ( export PATH="$q/bin:$PATH" C05F_STATE="$tmp/$name" C05F_CPU_STUB="$q/cpu-stub"
    rc=0; env "$@" E07_PLAN=cfinal NETWORK_VOLUME=testvolume01 HF_CLOUDS= CAP_MIN=5 RUNPOD_API="$q/runpod_api.py" \
      RUNPOD_API_KEY=c05f-test-fake SSH_KEY="$q/key" MACHINE_PROBE_ROOT="$q/probe" OUT="$tmp/$name/20261019T200000Z" \
      SHEET_ROOT="$tmp/$name/sheets" timeout 900 bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" > "$tmp/$name/e07.out" 2>&1 < /dev/null || rc=$?
    echo "$rc" > "$tmp/$name/rc" )
}
session floor "slow-cpu-1 good-2" &
p1=$!
session override "slow-cpu-1" CPU_EXPERTS_FLOOR_GBS=40 &
p2=$!
session excluded "known-bad-1 good-2" BAD_MACHINES="other,known-bad-1" &
p3=$!
session slowdl "good-1" "C05F_STOP=stop: download too slow: 0.13 GB/s (c05f-test.sh stub)" &
p4=$!
wait "$p1" "$p2" "$p3" "$p4"

echo "--- 1. floor: the slow CPU host rejected after the machine probe passed, the next one passes"
r="$tmp/floor" l="$tmp/floor/20261019T200000Z/e07.log" pl="$tmp/floor/20261019T200000Z/probe.log"
grep -E "CPU experts|machine qualification failed|unfit|cost|payload" "$l" | sed 's/^/  /' | cut -c1-220 || true
check "pod 1 passed the machine probe, then the CPU experts probe ran with the engine's 16 threads, the floor 71 and the GPU's NUMA node 1" \
  'grep -q "\] dsv41_machine_probe: PASS" "$l" &&
   grep -qx "slow-cpu-1: --threads 16 --min-gbs 71 --numa-node 1" "$r/cpu-args" &&
   grep -qx "qualification: c05f-pod-1 (NVIDIA RTX PRO 6000 Blackwell Server Edition): CPU experts floor 71 GB/s (CPU_EXPERTS_FLOOR_GBS), 16 pool threads" "$pl"'
check "the rejection line: qualification: CPU experts 44.7 GB/s < 71, in e07.log and probe.log" \
  'grep -q "\] qualification: CPU experts 44.7 GB/s < 71 on c05f-pod-1 (machine slow-cpu-1, exit 1; CPU_EXPERTS_FLOOR_GBS): the machine qualification failed$" "$l" &&
   grep -q "^qualification: CPU experts 44.7 GB/s < 71 on c05f-pod-1 " "$pl" &&
   grep -q "^dsv41_cpu_experts_probe: measure cpu_experts_gbs 44.7 bound >=71 FAIL " "$pl"'
check "pod 1 deleted with its cost counted, pod 2 passes and alone gets the payload, exit 0" \
  '[[ "$(cat "$r/rc")" == 0 ]] && grep -q "\] c05f-pod-1 (machine slow-cpu-1) is unfit: deleting it, remembering the machine" "$l" &&
   grep -q "^pod c05f-pod-1: [0-9]* s at 2.59 USD/h = " "$tmp/floor/20261019T200000Z/cost.txt" &&
   [[ "$(head -n 1 "$r/deleted")" == c05f-pod-1 ]] &&
   [[ "$(grep tar.gz "$r/uploads")" == "22002 /root/e07.tar.gz payload-cu130.tar.gz" ]] && [[ "$(cat "$r/started")" == 22002 ]] &&
   grep -qx "good-2: --threads 16 --min-gbs 71 --numa-node 1" "$r/cpu-args" &&
   grep -q "\] dsv41_cpu_experts_probe: PASS, the CPU experts rate within its bound" "$l"'

echo "--- 2. floor: the sheet's machine table has the CPU experts column"
sheet="$r/sheets/2026-10-19-cfinal/README.md"
grep "^| c05f-pod-\|^| pod " "$sheet" 2>/dev/null | sed 's/^/  /' || true
check "pod 1's CPU experts 44.7 FAIL (>=71) and rejected naming cpu_experts_gbs; pod 2's 100.4 and passed" \
  'grep -q "^| pod | machine | offer | .* | host read GB/s | CPU experts GB/s | .* | verdict |$" "$sheet" &&
   grep -q "^| c05f-pod-1 | slow-cpu-1 | .* | 375.2 | \*\*44.7 FAIL\*\* (>=71) | .* | rejected: cpu_experts_gbs 44.7 (bound >=71) |$" "$sheet" &&
   grep -q "^| c05f-pod-2 | good-2 | .* | 375.2 | 100.4 | .* | passed |$" "$sheet" &&
   grep -q "^Bounds: .*CPU experts GB/s >=71" "$sheet"'

echo "--- 3. override: CPU_EXPERTS_FLOOR_GBS=40 lets the 44.7 GB/s host pass"
r="$tmp/override" l="$tmp/override/20261019T200000Z/e07.log"
check "slow-cpu-1 passes at the floor 40, the payload goes to pod 1, exit 0" \
  '[[ "$(cat "$r/rc")" == 0 ]] && grep -qx "slow-cpu-1: --threads 16 --min-gbs 40 --numa-node 1" "$r/cpu-args" &&
   grep -q "\] dsv41_cpu_experts_probe: measure cpu_experts_gbs 44.7 bound >=40 ok " "$l" &&
   [[ "$(grep tar.gz "$r/uploads")" == "22001 /root/e07.tar.gz payload-cu130.tar.gz" ]]'

echo "--- 4. e07.sh's checks and its dry run"
rc=0; CPU_EXPERTS_FLOOR_GBS=0 E07_PLAN=cfinal OUT="$tmp/bad" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/bad.out" 2>&1 || rc=$?
check "CPU_EXPERTS_FLOOR_GBS=0 refused, exit 2" \
  '(( rc == 2 )) && grep -q "CPU_EXPERTS_FLOOR_GBS=.0. is not a number above 0" "$tmp/bad.out"'
rc=0; E07_PLAN=cfinal OUT="$tmp/dry" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry.out" 2>&1 || rc=$?
grep "dsv41_cpu_experts_probe\|CPU experts" "$tmp/dry.out" | sed 's/^/  /' || true
check "the dry run exits 0 and plans the CPU experts probe with --threads 16 --min-gbs 71" \
  '(( rc == 0 )) && grep -q "^  qualification:   \"timeout 120 /root/dsv41_cpu_experts_probe --threads 16 --min-gbs 71 \[--numa-node <the GPU.s>\]\":$" "$tmp/dry.out" &&
   grep -q "^  qualification:   CPU experts >= 71 GB/s (CPU_EXPERTS_FLOOR_GBS;" "$tmp/dry.out"'

echo "--- 5. BAD_MACHINES excludes a known-bad machine"
r="$tmp/excluded" l="$tmp/excluded/20261019T200000Z/e07.log"
grep -E "BAD_MACHINES|created" "$l" | sed 's/^/  /' | cut -c1-220 || true
check "pod 1 on known-bad-1 deleted at once, no probe on it; pod 2 on good-2 alone gets the payload, exit 0" \
  '[[ "$(cat "$r/rc")" == 0 ]] && grep -q "\] c05f-pod-1 landed on machine known-bad-1, excluded by BAD_MACHINES: deleting it$" "$l" &&
   [[ "$(head -n 1 "$r/deleted")" == c05f-pod-1 ]] && ! grep -q "^ssh 22001:" "$r/ssh.log" &&
   ! grep -q "^known-bad-1:" "$r/cpu-args" && grep -q "\] created c05f-pod-2 on machine good-2 " "$l" &&
   [[ "$(grep tar.gz "$r/uploads")" == "22002 /root/e07.tar.gz payload-cu130.tar.gz" ]] && [[ "$(cat "$r/started")" == 22002 ]]'
rc=0; BAD_MACHINES="other,known-bad-1" E07_PLAN=cfinal OUT="$tmp/dry-bad" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry-bad.out" 2>&1 || rc=$?
check "the dry run prints the list, and none without BAD_MACHINES" \
  '(( rc == 0 )) && grep -qx "machines excluded by BAD_MACHINES (a pod landing on one is deleted at once): other known-bad-1" "$tmp/dry-bad.out" &&
   grep -qx "machines excluded by BAD_MACHINES (a pod landing on one is deleted at once): none" "$tmp/dry.out"'
rc=0; BAD_MACHINES="good-1;rm" E07_PLAN=cfinal OUT="$tmp/dry-refused" bash "$STRATA_DIR/tools/dsv41/pod/e07.sh" --dry-run > "$tmp/dry-refused.out" 2>&1 || rc=$?
check "an id that is not one refused, exit 2" \
  '(( rc == 2 )) && grep -q "BAD_MACHINES has .good-1;rm., not a machine id" "$tmp/dry-refused.out"'

echo "--- 6. a slow HF download names its machine"
l="$tmp/slowdl/20261019T200000Z/e07.log"
grep "download too slow\|slow download machine" "$l" | sed 's/^/  /' || true
check "e07.log has 'slow download machine: good-1' and the session fails" \
  '[[ "$(cat "$tmp/slowdl/rc")" != 0 ]] && grep -q "\] slow download machine: good-1$" "$l" &&
   ! grep -q "slow download machine" "$tmp/floor/20261019T200000Z/e07.log"'
if (( failed > 0 )); then
  for n in floor override excluded slowdl; do echo "  $n: the tail of e07.out:"; tail -n 8 "$tmp/$n/e07.out" | sed 's/^/    /'; done
fi

echo "c05f-test.sh: $([[ $failed == 0 ]] && echo PASS || echo "FAIL ($failed)")"
(( failed == 0 ))
