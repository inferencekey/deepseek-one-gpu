"""Package backend.py with the engine's Python side and publish it to your tenant as a custom backend.

Run once (and again whenever you change backend.py: bump VERSION first).

    python publish.py

The package holds backend.py and, under strata/, the files of this repository
the backend runs: the server (serve/, without its tests), the placement tool
(tools/dsv41/pod/e07_gate.py and the three modules it imports), the tokenizer
reader and the two data files of the published placement. About 1 MB; no
weights and no engine binary (setup() downloads both on the worker).

Reads INFERENCEKEY_BASE_URL, INFERENCEKEY_TENANT_ID and INFERENCEKEY_USER_TOKEN
(a dashboard session token: the upload is a tenant-level action, so it is
authenticated as a user, not with the project's ik_sdk_ token). You can also
skip the upload and drag the .tar.gz into Dashboard -> Custom backends.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from inferencekey import publish_custom_backend
from inferencekey.backend import package_backend

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = "0.1.0"
SLUG = "deepseek-v41-flash-one-gpu"
DESCRIPTION = "DeepSeek-V4.1-Flash (763B MoE, Q2_K) on one RTX PRO 6000 96 GB, hot experts in VRAM (CUDA 13)"

# What the backend runs from this repository, relative to its root.
RUNTIME_FILES = [
    "LICENSE",
    "NOTICE",
    "data/dsv41-expert-counts.json",
    "tools/gguf_reader.py",
    "tools/strata_tokenizer.py",
    "tools/dsv41/pod/e07_gate.py",
    "tools/dsv41/oracle/gguf_header.py",
    "tools/dsv41/oracle/hot_set.py",
    "tools/dsv41/oracle/layout-dsv41-q2k.json",
    "tools/dsv41/engine/kv_budget.py",
]
RUNTIME_DIRS = ["serve"]


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def stage() -> Path:
    """dist/stage/: backend.py at the root, the repository's runtime files under strata/."""
    out = HERE / "dist" / "stage"
    shutil.rmtree(out, ignore_errors=True)
    (out / "strata").mkdir(parents=True)
    shutil.copy2(HERE / "backend.py", out / "backend.py")
    for rel in RUNTIME_FILES:
        dest = out / "strata" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest)
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", "test_*.py", "mcp_fake_server.py", "chat_golden.json")
    for rel in RUNTIME_DIRS:
        shutil.copytree(REPO / rel, out / "strata" / rel, ignore=skip)
    return out


def main() -> int:
    _load_dotenv()
    pkg = package_backend(
        src=str(stage()),
        entrypoint="backend:DeepSeekOneGpuBackend",
        requirements=str(HERE / "requirements.txt"),
        name=SLUG,
        slug=SLUG,
        version=VERSION,
        task_type="text2text",
        requires_python=">=3.11,<3.14",
        description=DESCRIPTION,
        out_dir=str(HERE / "dist"),
    )
    print(f"packaged {pkg.path} ({pkg.size_bytes} bytes, sha256 {pkg.sha256[:12]})")

    tenant_id = os.environ.get("INFERENCEKEY_TENANT_ID")
    token = os.environ.get("INFERENCEKEY_USER_TOKEN")
    if not tenant_id or not token:
        print(
            "INFERENCEKEY_TENANT_ID / INFERENCEKEY_USER_TOKEN not set: upload the "
            "package from Dashboard -> Custom backends instead.",
            file=sys.stderr,
        )
        return 0

    result = publish_custom_backend(tenant_id=tenant_id, package_path=str(pkg.path), token=token)
    print(f"published custom backend slug={result.get('slug')} id={result.get('id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
