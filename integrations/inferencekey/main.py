"""A 763B model on one GPU you control: DeepSeek-V4.1-Flash through InferenceKey.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()).
The backend is a **custom Python backend** (`deepseek-v41-flash-one-gpu`,
published with publish.py) that downloads this repository's engine and the
264.5 GB GGUF, plans which experts go to VRAM, and runs `serve.server` on your
worker's RTX PRO 6000.

The chat goes through the InferenceKey gateway with the official `openai`
client: the gateway exposes an OpenAI-compatible base URL per workload,
`<base>/endpoint/<project>/<slug>/v1`, and callers send `model="default"`.
A custom backend answers with its own JSON object as the message content
(`{"text", "reasoning", "finish_reason", "usage", "timings"}`), so this script
decodes it.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py [--prompt "..."] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import openai
from openai import OpenAI

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

HERE = Path(__file__).resolve().parent
SLUG = os.environ.get("DSV41_WORKLOAD_SLUG", "deepseek-v41-flash")
BACKEND_SLUG = "deepseek-v41-flash-one-gpu"  # from publish.py
OUT_DIR = HERE / "outputs"
DEFAULT_PROMPT = "Explain mixture-of-experts models in two sentences."
# First start: the worker builds the backend venv, then setup() downloads the
# engine (682 MB) and 264.5 GB of GGUF and loads them. Later starts reuse both.
READY_TIMEOUT = int(os.environ.get("IK_READY_TIMEOUT", "7200"))


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def decode(content: str) -> dict:
    """The backend's output object from the message content (plain text is kept as is)."""
    try:
        out = json.loads(content)
    except (TypeError, ValueError):
        return {"text": content}
    return out if isinstance(out, dict) else {"text": content}


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="DeepSeek-V4.1-Flash on one RTX PRO 6000",
        slug=SLUG,
        model="vcruz305/DeepSeek-V4.1-Flash-GGUF",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2text",
        worker_id=os.environ["IK_WORKER_ID"],
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        # No execution_policy_config: the platform stores no replica count for a
        # fixed policy, so passing one would make every ensure() re-PATCH it.
        execution_policy=ExecutionPolicy.FIXED,
    )
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    endpoint_base = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{SLUG}/v1"

    ref = None
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        print(f"waiting for the worker (first start downloads 264.5 GB; timeout {READY_TIMEOUT} s)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=READY_TIMEOUT)
        print(f"ready after {round(time.time() - t0, 1)} s")

        client = OpenAI(base_url=endpoint_base, api_key=os.environ["INFERENCEKEY_API_KEY"])
        t1 = time.time()
        completion = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
        )
        out = decode(completion.choices[0].message.content)
        out["round_trip_s"] = round(time.time() - t1, 2)
        t = out.get("timings") or {}
        print(f"\n{out.get('text')}\n")
        print(f"{t.get('generated_tokens')} tokens at {t.get('generation_tokens_per_s')} tok/s, "
              f"round trip {out['round_trip_s']} s")
        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "answer.json").write_text(json.dumps({"prompt": args.prompt, **out}, indent=1,
                                                        ensure_ascii=False))
        print(f"wrote {OUT_DIR / 'answer.json'}")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except openai.APIError as e:
        print(f"chat request failed: {e.message}", file=sys.stderr)
        return 1
    except TimeoutError as e:
        print(f"error: {e}; raise IK_READY_TIMEOUT or check the worker in the Manager UI", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (to raise its concurrency or try more prompts).
        keep_alive = os.environ.get("IK_KEEP_ALIVE") == "1"
        if keep_alive and ref is None:
            print("ensure() did not complete, nothing was created to keep (IK_KEEP_ALIVE=1)")
        elif keep_alive:
            print(f"kept workload {SLUG} (IK_KEEP_ALIVE=1); endpoint: {endpoint_base} (model \"default\")")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
