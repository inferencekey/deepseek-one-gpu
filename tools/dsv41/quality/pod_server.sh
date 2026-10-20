#!/usr/bin/env bash
# The server of a quality pod session (tools/dsv41/pod/e07-run.sh, E07_PLAN=qfork and qengine), started in its own
# session (setsid), so pod-steps.sh's reap of the step that starts it leaves it running for the lm-eval and needle
# steps after it; `stop` ends it (SIGTERM, which both servers take as Ctrl+C: llama-server exits normally,
# serve/server.py sends its engine QUIT; then SIGKILL to the session after 60 s), then waits up to 45 s more until the
# server's pid is gone and nothing listens on its port nor accepts a connection there (an earlier
# session's next server died in 6 s on "port 8090 is already in use" right after a SIGKILL at 60 s); a port that
# never frees is exit 1 naming the holder's pid and command line. 60 + 45 s stays inside the server-stop step's
# timeout (2 x its 60 s estimate), so the failure is this script's own, not the step's timeout. POD_SERVER_TERM_S and
# POD_SERVER_PORT_FREE_S (default 60 and 45) shorten them for c04-test.sh. Run from the payload directory; the command
# goes to out/quality-server.cmd, its output to out/quality-server.log, its pid to out/quality-server.pid, its port
# to out/quality-server.port. Each start waits until the server answers (or it exits: exit 1 with its log's tail).
#
#   bash pod_server.sh fork MODEL PORT NUMA...     the fork's llama-server (payload fork/), t04a.sh's stock flags of
#                                                  the measured baseline (14 expert layers in VRAM, every other
#                                                  routed expert on the CPU) with --ctx-size 32768, -np Q_FORK_NP
#                                                  slots (default 1, lm-eval's NUM_CONCURRENT matches it;
#                                                  the 2026-10-11 qfork session's -np 4
#                                                  decoded at 1.6-2.3 tok/s per slot), --kv-unified (harmless at one
#                                                  slot: the slot's context is the whole 32768 either way) and
#                                                  --prefill-assistant (the HumanEval continuation); --threads
#                                                  min(16, cpus) and --threads-batch min(32, cpus), cpus nproc under the
#                                                  NUMA binding capped by the cgroup's CPU quota, floor(quota / period)
#                                                  and at least 1 (t04a's 16 and 32 on a host with that many CPUs; 16
#                                                  and 18 on the 2026-10-11 host's 18 vCPU quota); ready
#                                                  at /health; nvidia-smi samples the GPU's memory used every 1 s, in
#                                                  its own session too, into out/quality-gpu.csv until `stop`
#                                                  (the fork server's peak on the qfork sheet, the
#                                                  headroom of --ctx-size 32768 on the 80 GB GPU)
#   bash pod_server.sh engine MODEL PORT PLACE NUMA...  serve/server.py (src/) with the deepseek-v4.1 profile on
#                                                  ./dsv41-engine, run from the payload directory (the config's
#                                                  cwd), with the gate's placement PLACE, --chunk 16384,
#                                                  --ctx 262144 (the 256K needle) and the GGUF's end of sentence;
#                                                  repeat_stop_tokens 0 (no reply cut that the fork does not make);
#                                                  ready at its "ready: http" line
#   bash pod_server.sh stop
#
# NUMA is the binding prefix of e07-run.sh (numactl or taskset, t04a-lib.sh numa_binding; empty when unbound).
set -uo pipefail
mode="${1:?fork, engine or stop}"
shift
log=out/quality-server.log pidf=out/quality-server.pid portf=out/quality-server.port

port_holder() {   # port_holder PORT: exit 0 when nothing listens on PORT nor accepts there; else exit 1 saying who holds it
  python3 - "$1" <<'PY'
import os
import socket
import sys

port = int(sys.argv[1])
inodes = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        rows = open(table).read().splitlines()[1:]
    except OSError:
        continue
    for row in rows:
        f = row.split()
        if f[3] == "0A" and int(f[1].rsplit(":", 1)[1], 16) == port:   # 0A: TCP_LISTEN
            inodes.add(f[9])
holders = []
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        continue
    for fd in fds:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in inodes:
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError:
                cmd = "command line unreadable"
            holders.append(f"pid {pid} ({cmd})")
            break
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
    accepts = True
except OSError:
    accepts = False
finally:
    s.close()
if not inodes and not accepts:
    sys.exit(0)
if holders:
    print("held by " + "; ".join(holders))
elif inodes:
    print("a listening socket that no readable /proc/<pid>/fd names")
else:
    print("no listening socket in /proc/net/tcp, yet 127.0.0.1 accepts a connection on it")
sys.exit(1)
PY
}

wait_ready() {   # wait_ready CHECK: until CHECK passes, or the server exits
  local pid="$1"; shift
  while ! "$@" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "pod_server.sh: the server exited: $(tail -n 5 "$log")"; return 1; }
    sleep 5
  done
}

case "$mode" in
  fork)
    model="$1" port="$2"; shift 2
    echo "$port" > "$portf"
    # t04a.sh's STOCK14 and COMMON (the stock baseline: 18.92 / 19.15 tok/s decode on the A100 SXM4, pod 1 of tools/dsv41/README.md), with
    # the context, the slots, the threads and the server's own flags of this plan
    np="${Q_FORK_NP:-1}"
    [[ "$np" =~ ^[1-9][0-9]*$ ]] || { echo "pod_server.sh: Q_FORK_NP='$np' is not a whole number of slots >= 1" >&2; exit 2; }
    # the CPUs the server may run on: nproc under the same NUMA binding,
    # capped by the cgroup's CPU quota, which nproc does not see (the 2026-10-11 host's nproc was 128 for an 18 vCPU
    # quota). Q_FORK_CGROUP stands in for /sys/fs/cgroup in the rehearsal (q02-test.sh)
    ncpu="$("$@" nproc)" || { echo "pod_server.sh: nproc under '$*' failed" >&2; exit 1; }
    cg="${Q_FORK_CGROUP:-/sys/fs/cgroup}" quota="" period="" limit="none (no cpu.max nor cpu/cpu.cfs_quota_us under $cg)"
    if [[ -r "$cg/cpu.max" ]]; then
      read -r quota period < "$cg/cpu.max"
      limit="v2 $cg/cpu.max $quota $period"
      [[ "$quota" == max ]] && quota=""   # no quota
    elif [[ -r "$cg/cpu/cpu.cfs_quota_us" ]]; then
      quota="$(cat "$cg/cpu/cpu.cfs_quota_us")" period="$(cat "$cg/cpu/cpu.cfs_period_us" 2>/dev/null)"
      limit="v1 $cg/cpu/cpu.cfs_quota_us $quota cfs_period_us $period"
      [[ "$quota" == -1 ]] && quota=""   # no quota
    fi
    cpus="$ncpu" source=nproc
    if [[ -n "$quota" ]]; then
      [[ "$quota" =~ ^[0-9]+$ && "$period" =~ ^[1-9][0-9]*$ ]] ||
        { echo "pod_server.sh: cgroup CPU limit '$limit' is not a quota and a period in microseconds" >&2; exit 1; }
      q=$(( quota / period ))
      (( q >= 1 )) || q=1   # a quota under one CPU still runs one thread
      (( q < ncpu )) && cpus="$q" source="the cgroup quota"
    fi
    threads=$(( cpus < 16 ? cpus : 16 )) threads_batch=$(( cpus < 32 ? cpus : 32 ))
    echo "fork threads: nproc $ncpu${*:+ under $*}, cgroup CPU limit: $limit; cpus $cpus (from $source);" \
      "--threads $threads --threads-batch $threads_batch -np $np"
    l14='blk\.(2[6-9]|3[0-9])\.ffn_(down|gate|up)_exps\.weight=CUDA0'
    cpu_exps='ffn_(down|gate|up)_exps\.weight=CPU'
    numa=()
    (( $# )) && numa=(--numa numactl)   # llama-server's NUMA mode: the CPU affinity the binding set (numactl or taskset), as t04a.sh ran it
    cmd=(env -u LD_LIBRARY_PATH "$@" fork/llama-server -m "$model" --override-tensor "$l14" --override-tensor "$cpu_exps"
         -lm mmap --n-gpu-layers 99 --override-tensor 'engram_embd\.weight=CPU' --flash-attn on --ctx-size 32768
         --batch-size 2048 --ubatch-size 1024 --threads "$threads" --threads-batch "$threads_batch" "${numa[@]}"
         -np "$np" --kv-unified --prefill-assistant --host 127.0.0.1 --port "$port" --jinja --no-reasoning-preserve)
    printf '%q ' "${cmd[@]}" > out/quality-server.cmd
    setsid "${cmd[@]}" > "$log" 2>&1 < /dev/null &
    echo $! > "$pidf"
    server="$!"
    setsid nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -lms 1000 \
      > out/quality-gpu.csv 2>&1 < /dev/null &
    echo $! > out/quality-gpu.pid
    wait_ready "$server" curl -sf "http://127.0.0.1:$port/health" -o /dev/null && echo "llama-server ready on $port"
    ;;
  engine)
    model="$1" port="$2" place="$3"; shift 3
    echo "$port" > "$portf"
    eos="$(python3 -c "import sys; sys.path.insert(0, 'src/tools'); from gguf_reader import GGUFFile
from pathlib import Path; print(GGUFFile(Path(sys.argv[1])).metadata['tokenizer.ggml.eos_token_id'])" "$model")"
    # cwd: the engine runs from the payload directory, as the other engine plans run it, so the placement's relative
    # --hot-set out/gate/hot-set.json (and a relative model) resolve there, not under src/ where serve/server.py runs
    # (a qengine session's engine once aborted on the hot set it could not open)
    # shellcheck disable=SC2086  # the placement is whole words
    python3 -c "import json, os, sys
print(json.dumps({'exe': os.path.abspath('dsv41-engine'), 'cwd': os.getcwd(), 'args': ['--model', sys.argv[1],
  *sys.argv[2].split(), '--chunk', '16384', '--ctx', '262144', '--eos-ids', sys.argv[3]], 'model_profile': 'deepseek-v4.1',
  'tokenizer_gguf': sys.argv[1], 'log': os.path.abspath('out/quality-engine.log'), 'repeat_stop_tokens': 0}, indent=1))" \
      "$model" "$place" "$eos" > out/quality-server.json
    cmd=("$@" python3 -m serve.server --engine strata --config "$PWD/out/quality-server.json" --port "$port")
    printf '%q ' "${cmd[@]}" > out/quality-server.cmd
    (cd src && exec setsid "${cmd[@]}") > "$log" 2>&1 < /dev/null &
    echo $! > "$pidf"
    wait_ready "$!" grep -q "ready: http" "$log" && echo "serve/server.py ready on $port"
    ;;
  stop)
    gpu="$(cat out/quality-gpu.pid 2>/dev/null)" && kill -TERM "$gpu" 2>/dev/null && echo "GPU memory sampler $gpu stopped"
    pid="$(cat "$pidf" 2>/dev/null)" || { echo "pod_server.sh: no server was started"; exit 0; }
    port="$(cat "$portf" 2>/dev/null)" ||
      { echo "pod_server.sh: FAIL: server $pid has no $portf, its port cannot be checked free" >&2; exit 1; }
    term_s="${POD_SERVER_TERM_S:-60}" free_s="${POD_SERVER_PORT_FREE_S:-45}"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in $(seq "$term_s"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { kill -KILL -- "-$pid" 2>/dev/null; echo "pod_server.sh: SIGKILL after $term_s s"; }
    # the next server binds the same port: the pid gone and the port free first, within free_s
    t0=$SECONDS
    while :; do
      if kill -0 "$pid" 2>/dev/null; then held="server $pid is still alive"
      elif held="$(port_holder "$port")"; then break
      else held="port $port is in use, $held"; fi
      (( SECONDS - t0 < free_s )) || {
        echo "pod_server.sh: FAIL: $held $free_s s after the stop's last signal, the next server cannot start on port $port" >&2
        exit 1; }
      sleep 1
    done
    if (( SECONDS == t0 )); then echo "port $port free"; else echo "port $port free after $(( SECONDS - t0 )) s more"; fi
    echo "server $pid stopped"
    ;;
  *) echo "pod_server.sh: '$mode': fork, engine or stop" >&2; exit 2 ;;
esac
