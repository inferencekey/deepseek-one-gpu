"""DeepSeek-V4.1-Flash on one RTX PRO 6000 (deepseek-one-gpu) as an InferenceKey custom backend.

Why a custom backend: the model (763B parameters, Mixture-of-Experts, GGUF
Q2_K, 264.5 GB) is larger than any single GPU. This repository's engine keeps
the hot experts in the 96 GB of VRAM and computes the rest on CPU threads from
host RAM. No built-in InferenceKey backend runs that engine, so this backend
brings it:

``setup()`` (once per process)
    1. checks the machine (an NVIDIA GPU with about 96 GB, enough RAM),
    2. downloads the prebuilt engine (the ``v0.1.0`` release asset, checked
       against its sha256) and the GGUF at the pinned revision, in parallel,
    3. plans the placement with ``tools/dsv41/pod/e07_gate.py`` (which experts
       go to VRAM, and whether the rest fits in RAM),
    4. writes the server configuration (README, step 5), starts
       ``serve.server`` on a free loopback port bound to the GPU's NUMA node,
       and waits for ``/health``.

``process(job)`` (per request)
    forwards the OpenAI chat request (messages and sampling fields, also from
    ``extra_body``) to the server's ``/v1/chat/completions`` and returns
    ``{"text", "reasoning", "finish_reason", "usage", "timings"}``.

The Python side of the engine (``serve/``, the placement tool and its two data
files) travels inside the backend package: ``publish.py`` copies them next to
this file under ``strata/``. Run from a checkout of the repository, the backend
uses the checkout instead.

Settings are class attributes. Override them with ``DSV41_*`` environment
variables or the runtime's ``--config-json`` (keys in lower case).
``server_engine = "mock"`` starts the server's scripted mock engine instead
(no GPU, no download, canned text): it exists to test the wiring of this
backend on any machine.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from inferencekey.backend import BackendContext, CustomBackend, InvalidInput, Job, Result

HERE = Path(__file__).resolve().parent

# Fields of an OpenAI chat body that are passed to the server untouched.
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "stop", "seed", "presence_penalty",
                  "frequency_penalty", "repetition_penalty", "response_format")
# Fields the server takes beside the OpenAI ones (docs/DETAILS.md); the platform
# carries them in `extra_body`, the async jobs API at the top level.
_EXTRA_KEYS = ("reasoning_effort", "chat_template_kwargs", "strata_reproducible") + _SAMPLING_KEYS


def _log(msg: str) -> None:
    print(f"[deepseek-one-gpu] {msg}", file=sys.stderr, flush=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (atomic rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "deepseek-one-gpu-backend"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=8 << 20)
    tmp.rename(dest)


def _run(cmd: List[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60).stdout


def _gpu_vram_mib() -> int:
    """memory.total of GPU 0, MiB (nvidia-smi)."""
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not found: this backend needs an NVIDIA GPU and its driver")
    out = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    return int(out.splitlines()[0].strip())


def _mem_total_kib() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    raise RuntimeError("no MemTotal in /proc/meminfo")


def _cgroup_limit() -> str:
    """The memory cgroup's limit in bytes, or "unlimited" (the same rule as tools/dsv41/pod/t04a-lib.sh)."""
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(f).read_text().split()[0]
        except (OSError, IndexError):
            continue
        if v == "max" or (v.isdigit() and int(v) >= 1 << 62):
            return "unlimited"
        if v.isdigit():
            return v
    return "unlimited"


def _gpu_numa_node() -> Optional[int]:
    """The NUMA node of GPU 0: the "NUMA Affinity" column of `nvidia-smi topo -m` (the pod scripts' rule)."""
    try:
        out = _run(["nvidia-smi", "topo", "-m"])
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if line.startswith("GPU0"):
            fields = line.split()
            if len(fields) >= 2 and fields[-2].isdigit():
                return int(fields[-2])
    return None


def _numa_prefix(node: Optional[int]) -> List[str]:
    """The command prefix that runs the server on the GPU's NUMA node, or [] (unbound).

    The forms are tried on `true`, in order, as tools/dsv41/pod/t04a-lib.sh does:
    numactl with the node's memory preferred, numactl on its CPUs only (Docker's
    default seccomp profile refuses set_mempolicy), then taskset on the node's
    CPU list (no libnuma).
    """
    nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
    if node is None or len(nodes) < 2:
        _log(f"NUMA: unbound ({'no NUMA node for GPU 0' if node is None else 'one NUMA node'})")
        return []
    forms = [["numactl", f"--cpunodebind={node}", f"--preferred={node}"], ["numactl", f"--cpunodebind={node}"]]
    try:
        forms.append(["taskset", "-c", Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()])
    except OSError:
        pass
    for form in forms:
        try:
            if subprocess.run(form + ["true"], capture_output=True, timeout=10).returncode == 0:
                _log(f"NUMA: {' '.join(form)}")
                return form
        except (OSError, subprocess.SubprocessError):
            pass
        _log(f"NUMA: {' '.join(form)} refused")
    _log("NUMA: unbound, every form refused (expect lower speed)")
    return []


class DeepSeekOneGpuBackend(CustomBackend):
    name = "deepseek-v41-flash-one-gpu"
    version = "0.1.0"
    task_type = "text2text"
    description = "DeepSeek-V4.1-Flash (763B MoE, Q2_K) on one RTX PRO 6000 96 GB, hot experts in VRAM"

    # The engine, prebuilt for CUDA 13.0 and sm_120 (Blackwell); it bundles its CUDA runtime.
    engine_url = ("https://github.com/inferencekey/deepseek-one-gpu/releases/download/v0.1.0/"
                  "strata-281712ce-cu130-sm120.tar.gz")
    engine_sha256 = "dbfed1bb6b0a56922d1b19d6fd5de28eda93bf2229aa6795bf820be53590d55b"
    # The weights: seven files, 264.5 GB, the revision every published number was measured with.
    hf_repo = "vcruz305/DeepSeek-V4.1-Flash-GGUF"
    hf_revision = "c4a085541cb53f67ee5e57b63d255e80cef286e7"
    gguf_pattern = "DeepSeek-V4.1-Flash-Q2_K-*"
    model_dir = ""  # "" = <cache>/models/q2k; point it at an existing download to skip it
    # The published configuration (README, steps 4 and 5).
    threads = 16
    prefetch_experts = 128
    checkpoints = 8
    batch = 8
    batch_ctx = 69632
    chunk = 16384
    ctx = 262144
    # Request defaults. The worker gives each /process call 300 s, so a reply has
    # to fit in it: 4,096 tokens take about 215 s at the 19 tok/s of a user among 8.
    default_max_tokens = 4096
    request_timeout_s = 290
    ready_timeout_s = 1800  # loading 264.5 GB takes a few minutes; more on slow disks
    server_engine = "strata"  # "mock": the server's scripted engine, for wiring tests only

    # ------------------------------------------------------------------ setup

    def _setting(self, ctx: BackendContext, key: str) -> Any:
        default = getattr(self, key)
        raw = os.environ.get(f"DSV41_{key.upper()}")
        if raw is None:
            raw = (ctx.config or {}).get(key)
        if raw is None:
            return default
        return type(default)(raw) if not isinstance(default, str) else str(raw)

    @staticmethod
    def _repo_root() -> Path:
        """Where serve/ and tools/ are: the copy publish.py packages, else this checkout."""
        packaged = HERE / "strata"
        return packaged if (packaged / "serve" / "server.py").is_file() else HERE.parents[1]

    def setup(self, ctx: BackendContext) -> None:
        for key in ("engine_url", "engine_sha256", "hf_repo", "hf_revision", "gguf_pattern", "model_dir",
                    "threads", "prefetch_experts", "checkpoints", "batch", "batch_ctx", "chunk", "ctx",
                    "default_max_tokens", "request_timeout_s", "ready_timeout_s", "server_engine"):
            setattr(self, key, self._setting(ctx, key))
        if self.server_engine not in ("strata", "mock"):
            raise RuntimeError(f"server_engine {self.server_engine!r}: 'strata' or 'mock'")
        self.root = self._repo_root()
        cache = Path(os.environ.get("DSV41_CACHE_DIR", Path.home() / ".cache" / "deepseek-one-gpu"))
        self.run_dir = cache / "run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.timings: Dict[str, float] = {}
        t_setup = time.time()

        if self.server_engine == "strata":
            cmd = self._prepare_strata(cache)
        else:
            _log("server_engine=mock: the server's scripted mock engine, no GPU, no download")
            cmd = [sys.executable, "-m", "serve.server", "--engine", "mock"]

        self.port = _free_port()
        cmd += ["--port", str(self.port)]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(self.root)] + [p for p in env.get("PYTHONPATH", "").split(
            os.pathsep) if p])
        _log("starting: " + " ".join(cmd))
        # Same stdout/stderr as this process, so the server's log lands in the worker log.
        self.proc = subprocess.Popen(cmd, cwd=str(self.root), env=env, stdin=subprocess.DEVNULL)
        # The runtime's SIGTERM handler exits through atexit; the worker also
        # stops the whole process group, so the server never outlives us.
        atexit.register(self._stop)

        t_load = time.time()
        deadline = t_load + float(self.ready_timeout_s)
        while True:
            code = self.proc.poll()
            if code is not None:
                raise RuntimeError(f"serve.server exited with code {code} before it was ready"
                                   f" (engine log: {self.run_dir / 'engine.log'})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if r.status == 200 and json.loads(r.read()).get("loaded", True) is not False:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError):
                pass
            if time.time() > deadline:
                self._stop()
                raise RuntimeError(f"serve.server not ready after {self.ready_timeout_s} s")
            time.sleep(2.0)
        self.timings["server_load_s"] = round(time.time() - t_load, 1)
        self.timings["setup_s"] = round(time.time() - t_setup, 1)
        _log(f"ready on 127.0.0.1:{self.port}: {json.dumps(self.timings)}")

    def _prepare_strata(self, cache: Path) -> List[str]:
        """Downloads, placement and the config; returns the server command (NUMA prefix included)."""
        vram_mib = _gpu_vram_mib()
        if vram_mib < 90 * 1024:
            raise RuntimeError(f"GPU 0 has {vram_mib} MiB; this engine needs an RTX PRO 6000 with 96 GB")
        engine_dir = cache / "engine" / Path(self.engine_url).name.removesuffix(".tar.gz")
        model_dir = Path(self.model_dir) if self.model_dir else cache / "models" / "q2k"

        def fetch_engine() -> Path:
            exe = engine_dir / "bin" / "dsv41-engine"
            if exe.is_file():
                return exe
            tarball = cache / "engine" / Path(self.engine_url).name
            _log(f"downloading {self.engine_url}")
            _download(self.engine_url, tarball)
            got = _sha256(tarball)
            if self.engine_sha256 and got != self.engine_sha256:
                tarball.unlink(missing_ok=True)
                raise RuntimeError(f"{tarball.name}: sha256 {got}, expected {self.engine_sha256}")
            tmp = Path(tempfile.mkdtemp(dir=cache / "engine", prefix=".extract-"))
            with tarfile.open(tarball) as tf:
                try:
                    tf.extractall(tmp, filter="data")
                except TypeError:  # Python without extraction filters
                    tf.extractall(tmp)
            # The archive holds one top-level folder, strata-<commit>-cu130-sm120/.
            (tmp / engine_dir.name).rename(engine_dir)
            shutil.rmtree(tmp, ignore_errors=True)
            tarball.unlink(missing_ok=True)
            return exe

        def fetch_model() -> Path:
            from huggingface_hub import snapshot_download

            _log(f"downloading {self.hf_repo}@{self.hf_revision} {self.gguf_pattern} to {model_dir}")
            snapshot_download(repo_id=self.hf_repo, revision=self.hf_revision,
                              allow_patterns=[self.gguf_pattern], local_dir=str(model_dir))
            shards = sorted(model_dir.glob("DeepSeek-V4.1-Flash-Q2_K-00001-of-*.gguf"))
            if not shards:
                raise RuntimeError(f"no DeepSeek-V4.1-Flash-Q2_K-00001-of-*.gguf in {model_dir}")
            return shards[0]

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_engine, f_model = pool.submit(fetch_engine), pool.submit(fetch_model)
            exe, model = f_engine.result(), f_model.result()
        self.timings["downloads_s"] = round(time.time() - t0, 1)

        # Placement: the hot set for VRAM and the RAM check (README, step 4).
        placement = self.run_dir / "placement"
        gate = subprocess.run(
            [sys.executable, str(self.root / "tools/dsv41/pod/e07_gate.py"),
             "--vram-mib", str(vram_mib), "--cgroup-bytes", _cgroup_limit(),
             "--mem-total-kib", str(_mem_total_kib()),
             "--counts", str(self.root / "data/dsv41-expert-counts.json"),
             "--layout", str(self.root / "tools/dsv41/oracle/layout-dsv41-q2k.json"),
             "--model-headers", str(model),
             "--prefetch-experts", str(self.prefetch_experts), "--checkpoints", str(self.checkpoints),
             "--batch", str(self.batch), "--batch-ctx", str(self.batch_ctx), "--out", str(placement)],
            capture_output=True, text=True)
        if gate.returncode != 0:
            raise RuntimeError(f"placement (e07_gate.py) failed: {(gate.stdout + gate.stderr).strip()[-1500:]}")
        g = json.loads((placement / "gate.json").read_text())

        # The server's configuration (README, step 5).
        args = ["--model", str(model), "--hot-set", str(placement / "hot-set.json"),
                "--vram-gb", str(g["vram_gb"]), "--ram-gb", str(g["ram_gb"]), "--threads", str(self.threads),
                "--hot-homes", str(g["hot_homes"]), "--prefetch-experts", str(self.prefetch_experts),
                "--checkpoints", str(self.checkpoints), "--batch", str(self.batch),
                "--batch-ctx", str(self.batch_ctx), "--chunk", str(self.chunk), "--ctx", str(self.ctx),
                "--eos-ids", "1"]
        config = self.run_dir / "deepseek.json"
        config.write_text(json.dumps({"exe": str(exe), "args": args, "model_profile": "deepseek-v4.1",
                                      "tokenizer_gguf": str(model), "log": str(self.run_dir / "engine.log"),
                                      "repeat_stop_tokens": 0}, indent=1))
        return _numa_prefix(_gpu_numa_node()) + [
            sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(config)]

    def _stop(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ---------------------------------------------------------------- process

    def _body(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        messages = inp.get("messages")
        if not isinstance(messages, list) or not messages:
            prompt = inp.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise InvalidInput("expected 'messages' (OpenAI chat) or 'prompt'")
            messages = [{"role": "user", "content": prompt}]
        if inp.get("tools"):
            raise InvalidInput("tool calls are not supported by this engine yet")
        body: Dict[str, Any] = {
            "model": "deepseek-v4.1-flash",
            "messages": messages,
            "max_tokens": inp.get("max_tokens") or int(self.default_max_tokens),
            "stream": False,
        }
        for key in _SAMPLING_KEYS:
            if inp.get(key) is not None:
                body[key] = inp[key]
        extra = inp.get("extra_body") if isinstance(inp.get("extra_body"), dict) else {}
        for key in _EXTRA_KEYS:
            if key in extra:
                body[key] = extra[key]
            elif key in inp and key not in body:
                body[key] = inp[key]
        return body

    def process(self, job: Job) -> Result:
        body = self._body(job.input)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=float(self.request_timeout_s)) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if 400 <= e.code < 500 and e.code != 429:
                raise InvalidInput(f"serve.server {e.code}: {detail}") from e
            raise RuntimeError(f"serve.server {e.code}: {detail}") from e
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = resp.get("usage") or {}
        t = resp.get("timings") or {}
        total_s = time.time() - t0
        generated = usage.get("completion_tokens")
        out: Dict[str, Any] = {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "timings": {
                "prompt_tokens": t.get("prompt_n", usage.get("prompt_tokens")),
                "prompt_tokens_per_s": round(t["prompt_per_second"], 1) if t.get("prompt_per_second") else None,
                "generated_tokens": t.get("predicted_n", generated),
                "generation_tokens_per_s": round(t["predicted_per_second"], 1) if t.get("predicted_per_second")
                else None,
                "total_s": round(total_s, 2),
            },
        }
        return Result(output=out)
