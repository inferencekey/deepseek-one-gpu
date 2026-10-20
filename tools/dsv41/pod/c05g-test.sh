#!/usr/bin/env bash
# Tests the binding of the engine to the GPU's NUMA node without a pod (tools/dsv41/pod/t04a-lib.sh numa_binding,
# which e07-run.sh calls for every engine process): on a fake two-node topology (NUMA_ROOT; node 0's CPUs the ones this
# shell may run on, node 1's CPUs none of them) with a fake numactl on PATH and the real taskset, under e07-run.sh's
# `set -uo pipefail` and t04a.sh's `set -Eeuo pipefail`:
#   1. numactl takes the memory policy: numactl --cpunodebind=0 --preferred=0, one "worked" line
#   2. numactl refuses --preferred as Docker's default seccomp profile does (exit 1, set_mempolicy: Operation not
#      permitted): numactl --cpunodebind=0, the refusal's stderr on one line, this process's Seccomp and CAP_SYS_NICE
#   3. numactl refuses every form (exit 1, No NUMA available): taskset -c <node 0's cpulist>, a process under it runs
#      on exactly those CPUs
#   4. node 1 (CPUs outside this shell's): taskset refused too, unbound, every form refused
#   5. node 0 without a cpulist: taskset not tried, unbound
#   6. no NUMA node of the GPU (N/A) and a one-node topology: unbound, saying why, numactl never called
# The e07-run.sh side (placement.txt and env.log of a session falling back) is cfinal-test.sh's rehearsal. Exits 1 if
# any check fails.
#
#   bash tools/dsv41/pod/c05g-test.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
failed=0
check() {
  if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; failed=$((failed + 1)); fi
}

mine="$(grep Cpus_allowed_list /proc/self/status | cut -f2)"
mkdir -p "$tmp/two/node0" "$tmp/two/node1" "$tmp/nolist/node0" "$tmp/nolist/node1" "$tmp/one/node0"
echo "$mine" > "$tmp/two/node0/cpulist"
echo "$mine" > "$tmp/one/node0/cpulist"
# a CPU number past every CPU this host can have
echo 4095 > "$tmp/two/node1/cpulist"

# fake_numactl MODE: numactl on $tmp/bin-MODE that records its call; accept: runs the command, preferred: refuses
# --preferred as under Docker's default seccomp profile, none: refuses everything, unused: accept's, for the cases that
# must not call it
fake_numactl() {
  mkdir -p "$tmp/bin-$1"
  cat > "$tmp/bin-$1/numactl" <<EOF
#!/usr/bin/env bash
echo "numactl \$*" >> "$tmp/calls-$1"
case "$1:\$*" in
  preferred:*--preferred*) printf 'set_mempolicy: Operation not permitted\nsetting preferred node: Operation not permitted\n' >&2; exit 1 ;;
  none:*) echo "numactl: No NUMA available on this system" >&2; exit 1 ;;
esac
while [[ "\${1:-}" == --* ]]; do shift; done
exec "\$@"
EOF
  chmod +x "$tmp/bin-$1/numactl"
}
for m in accept preferred none unused; do fake_numactl "$m"; done

# bind_case NAME MODE ROOT NODE OPTS: numa_binding NODE in a fresh shell with OPTS (set -uo pipefail or
# set -Eeuo pipefail), the fake numactl MODE and NUMA_ROOT ROOT; out-NAME holds NUMA_BIND, log-NAME the lines,
# cpus-NAME the CPUs a process under the prefix may run on, rc-NAME the shell's exit
bind_case() {
  local rc=0
  PATH="$tmp/bin-$2:$PATH" NUMA_ROOT="$3" bash -c "$5
    source '$here/t04a-lib.sh'
    numa_binding '$4'
    echo \"\${NUMA_BIND[*]}\" > '$tmp/out-$1'
    printf '%s\n' \"\${NUMA_BIND_LOG[@]}\" > '$tmp/log-$1'
    \"\${NUMA_BIND[@]}\" grep Cpus_allowed_list /proc/self/status | cut -f2 > '$tmp/cpus-$1'" || rc=$?
  echo "$rc" > "$tmp/rc-$1"
}
bind_case memory accept "$tmp/two" 0 "set -uo pipefail"
bind_case refused preferred "$tmp/two" 0 "set -uo pipefail"
bind_case refused-e preferred "$tmp/two" 0 "set -Eeuo pipefail"
bind_case taskset none "$tmp/two" 0 "set -uo pipefail"
bind_case taskset-e none "$tmp/two" 0 "set -Eeuo pipefail"
bind_case outside none "$tmp/two" 1 "set -Eeuo pipefail"
bind_case nolist none "$tmp/nolist" 0 "set -Eeuo pipefail"
bind_case na unused "$tmp/two" N/A "set -Eeuo pipefail"
bind_case one unused "$tmp/one" 0 "set -Eeuo pipefail"
for c in memory refused refused-e taskset taskset-e outside nolist na one; do
  echo "  $c: bound with '$(cat "$tmp/out-$c")', exit $(cat "$tmp/rc-$c")"
  sed 's/^/    /' "$tmp/log-$c"
done

check "1. numactl takes the memory policy: numactl --cpunodebind=0 --preferred=0, one line, it worked" \
  '[[ "$(cat "$tmp/out-memory")" == "numactl --cpunodebind=0 --preferred=0" ]] &&
   [[ "$(cat "$tmp/log-memory")" == "binding: numactl --cpunodebind=0 --preferred=0 worked (node 0'"'"'s CPUs, its memory preferred)" ]] &&
   [[ "$(cat "$tmp/rc-memory")" == 0 ]]'
check "2. --preferred refused: numactl --cpunodebind=0, the refusal's stderr on one line, the process's Seccomp and CAP_SYS_NICE, then it worked" \
  'for c in refused refused-e; do
     [[ "$(cat "$tmp/out-$c")" == "numactl --cpunodebind=0" && "$(cat "$tmp/rc-$c")" == 0 ]] || exit 1
     [[ "$(sed -n 1p "$tmp/log-$c")" == "binding: numactl --cpunodebind=0 --preferred=0 refused, exit 1: set_mempolicy: Operation not permitted; setting preferred node: Operation not permitted" ]] || exit 1
     sed -n 2p "$tmp/log-$c" | grep -qE "^binding: this process: Seccomp [0-9]+ \(2: a filter\), CapEff [0-9a-f]+, CAP_SYS_NICE (present|absent)$" || exit 1
     grep -qx "binding: numactl --cpunodebind=0 worked (node 0.s CPUs; the default local policy first-touches the memory there)" "$tmp/log-$c" || exit 1
   done'
check "3. numactl refuses every form: taskset -c <node 0's cpulist>, a process under it on those CPUs, each refusal on its line" \
  'for c in taskset taskset-e; do
     [[ "$(cat "$tmp/out-$c")" == "taskset -c $mine" && "$(cat "$tmp/cpus-$c")" == "$mine" && "$(cat "$tmp/rc-$c")" == 0 ]] || exit 1
     grep -qx "binding: numactl --cpunodebind=0 --preferred=0 refused, exit 1: numactl: No NUMA available on this system" "$tmp/log-$c" &&
     grep -qx "binding: numactl --cpunodebind=0 refused, exit 1: numactl: No NUMA available on this system" "$tmp/log-$c" &&
     [[ "$(tail -n 1 "$tmp/log-$c")" == "binding: taskset -c $mine worked (node 0"*"s CPUs from $tmp/two/node0/cpulist; the default local policy first-touches the memory there)" ]] || exit 1
   done'
check "4. node 1 outside this shell's CPUs: taskset refused too (the real taskset's error), unbound, every form refused" \
  '[[ -z "$(cat "$tmp/out-outside")" && "$(cat "$tmp/cpus-outside")" == "$mine" && "$(cat "$tmp/rc-outside")" == 0 ]] &&
   grep -q "^binding: taskset -c 4095 refused, exit 1: taskset: " "$tmp/log-outside" &&
   [[ "$(tail -n 1 "$tmp/log-outside")" == "binding: unbound, every form refused" ]]'
check "5. no cpulist for the node: taskset not tried, unbound" \
  '[[ -z "$(cat "$tmp/out-nolist")" ]] &&
   grep -qx "binding: taskset not tried, no CPU list in $tmp/nolist/node0/cpulist" "$tmp/log-nolist" &&
   [[ "$(tail -n 1 "$tmp/log-nolist")" == "binding: unbound, every form refused" ]]'
check "6. no NUMA node of the GPU, and one node: unbound saying why, numactl never called" \
  '[[ -z "$(cat "$tmp/out-na")" && -z "$(cat "$tmp/out-one")" ]] &&
   [[ "$(cat "$tmp/log-na")" == "binding: unbound, no NUMA node of the GPU (nvidia-smi topo -m'"'"'s GPU0 row: '"'"'N/A'"'"')" ]] &&
   [[ "$(cat "$tmp/log-one")" == "binding: unbound, 1 NUMA node(s) under $tmp/one: node 0 holds every CPU and all the memory" ]] &&
   [[ ! -e "$tmp/calls-unused" ]]'

if (( failed )); then echo "c05g-test.sh: $failed check(s) FAILED"; exit 1; fi
echo "c05g-test.sh: every check PASS"
