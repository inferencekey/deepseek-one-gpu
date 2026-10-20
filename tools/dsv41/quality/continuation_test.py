#!/usr/bin/env python3
"""tools/dsv41/quality/continuation_test.py - a HumanEval request that ends in a partial assistant message is
continued through each backend of api_proxy.py, without a GPU (fakes.py, no model, no money).

  llamacpp  the proxy, then fakes.py llama-server (the fork's rendering with its default --prefill-assistant): the
            forwarded request keeps the partial message last and as lm-eval sent it, with thinking off
            (chat_template_kwargs "enable_thinking": false), the prompt rendered from it continues the message, and
            the reply lm-eval reads is only the new text (the server starts it with the message, the proxy cuts that)
  strata    the proxy, then the real serve/server.py (deepseek-v4.1 profile) on fakes.py's engine: the prompt ids the
            engine reads continue the message (continuation.check_prompt on their text), the reply's content is only
            the new text, and a request that does not end with the assistant's message reads the ids it read before
            continuation support (the template with its generation prompt, tokenized), byte for byte
  deepseek  the proxy, then fakes.py deepseek-api: /beta/chat/completions with "prefix": true on the partial message
            (the API's chat prefix completion), /chat/completions without it otherwise

Then the negative control, so the test can fail: fakes.py llama-server with --no-prefill-assistant opens a new turn,
and the same checks must report it (a second <｜Assistant｜>, the partial message closed by the end of sentence); its
reply does not start with the message, so the llamacpp proxy answers 502.

    python3 tools/dsv41/quality/continuation_test.py --vocab $VOCAB --venv $QUALITY_VENV --out DIR

--vocab is the real tokenizer GGUF ($BUILD_ROOT/vocab/deepseek41-vocab.gguf), --venv a venv with aiohttp (the
proxy's, run_lm_eval.sh's QUALITY_VENV). Exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import continuation as C  # noqa: E402
import fakes  # noqa: E402

failed = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failed.append(name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, proc: subprocess.Popen, what: str, s: float = 60) -> None:
    end = time.time() + s
    while time.time() < end:
        if proc.poll() is not None:
            raise SystemExit(f"continuation_test.py: {what} exited ({proc.returncode})")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.2)
    raise SystemExit(f"continuation_test.py: {what} did not listen on {port} within {s} s")


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def records(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    py = str(Path(a.venv) / "bin" / "python")
    procs = []

    def start(cmd, log, port, what, cwd=None):
        p = subprocess.Popen(cmd, stdout=open(out / log, "w"), stderr=subprocess.STDOUT, cwd=cwd)
        procs.append(p)
        wait_port(port, p, what)
        return p

    def proxy(backend, upstream=None, extra=()):
        port = free_port()
        cmd = [py, str(HERE / "api_proxy.py"), "--backend", backend, "--port", str(port),
               "--usage", str(out / f"usage-{backend}.jsonl"), *extra]
        if upstream:
            cmd += ["--upstream", upstream]
        start(cmd, f"proxy-{backend}.log", port, f"the {backend} proxy")
        return f"http://127.0.0.1:{port}/v1/chat/completions"

    try:
        from serve.model_profile import DeepSeekV41Profile
        from serve.frontend import ChatTemplate
        prof = DeepSeekV41Profile(a.vocab)
        tpl = ChatTemplate(source=prof.template_source)
        items = C.items()
        for f in out.glob("*.jsonl"):
            f.unlink()

        # llamacpp: the fork's rendering with its default --prefill-assistant
        lp = free_port()
        start([sys.executable, str(HERE / "fakes.py"), "llama-server", "--tokenizer-gguf", a.vocab, "--record",
               str(out / "llama.jsonl"), "--port", str(lp), "-np", "4", "--prefill-assistant"], "llama.log", lp,
              "fakes.py llama-server")
        url = proxy("llamacpp", f"http://127.0.0.1:{lp}/v1/chat/completions")
        replies = [post(url, {"messages": it["messages"], "max_tokens": 64, "temperature": 0})["choices"][0]["message"]
                   ["content"] for it in items]
        for it, r, reply in zip(items, records(out / "llama.jsonl"), replies):
            partial = it["messages"][-1]["content"]
            sent = r["body"]["messages"][-1]
            check(f"llamacpp {it['task_id']}: the partial assistant message is forwarded last, unchanged",
                  sent == it["messages"][-1], repr(sent)[:120])
            check(f"llamacpp {it['task_id']}: thinking off (chat_template_kwargs)",
                  (r["body"].get("chat_template_kwargs") or {}).get("enable_thinking") is False)
            bad = C.check_prompt(r["prompt"], partial)
            check(f"llamacpp {it['task_id']}: the prompt continues the message", not bad, "; ".join(bad))
            check(f"llamacpp {it['task_id']}: the reply is only the new text (the server's echo of the message cut)",
                  reply == fakes.REPLY, repr(reply)[:80])

        # strata: the real server on the fake engine
        sp = free_port()
        cfg = {"exe": str(HERE / "fakes.py"),
               "args": ["--tokenizer-gguf", a.vocab, "--record", str(out / "engine.jsonl"), "--ctx", "32768"],
               "model_profile": "deepseek-v4.1", "tokenizer_gguf": a.vocab, "log": str(out / "engine.log"),
               "repeat_stop_tokens": 0}
        (out / "strata-config.json").write_text(json.dumps(cfg, indent=1))
        start([sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(out / "strata-config.json"),
               "--port", str(sp)], "strata-server.log", sp, "serve/server.py", cwd=ROOT)
        url = proxy("strata", f"http://127.0.0.1:{sp}/v1/chat/completions")
        replies = [post(url, {"messages": it["messages"], "max_tokens": 64, "temperature": 0})["choices"][0]["message"]
                   ["content"] for it in items]
        plain = [{"role": "user", "content": "What is 6 times 7?"}]
        post(url, {"messages": plain, "max_tokens": 16, "temperature": 0})
        recs = records(out / "engine.jsonl")
        for it, r, reply in zip(items, recs, replies):
            partial = it["messages"][-1]["content"]
            bad = C.check_prompt(prof.tokenizer.decode(r["ids"]), partial)
            check(f"strata {it['task_id']}: the engine's prompt ids continue the message", not bad, "; ".join(bad))
            check(f"strata {it['task_id']}: the reply is only the new text", reply == fakes.REPLY, repr(reply)[:80])
        before = prof.tokenizer.encode(tpl.render(plain, add_generation_prompt=True, **prof.template_kwargs(
            {"enable_thinking": False})), parse_special=True)
        check("strata: a request ending with the user's message reads the ids it read before continuation support",
              len(recs) == 4 and recs[3]["ids"] == before, f"{len(recs)} requests recorded")

        # deepseek: chat prefix completion on the beta base URL
        dp = free_port()
        key = out / "fake-key"
        key.write_text("sk-fakefake0123456789\n")   # a fake key for fakes.py deepseek-api, not a credential
        os.chmod(key, 0o600)
        start([sys.executable, str(HERE / "fakes.py"), "deepseek-api", "--key-file", str(key), "--record",
               str(out / "api.jsonl"), "--port", str(dp)], "api.log", dp, "fakes.py deepseek-api")
        url = proxy("deepseek", None, ["--key-file", str(key), "--api-base", f"http://127.0.0.1:{dp}"])
        for it in items:
            post(url, {"messages": it["messages"], "max_tokens": 64, "temperature": 0})
        post(url, {"messages": plain, "max_tokens": 16, "temperature": 0})
        recs = records(out / "api.jsonl")
        for it, r in zip(items, recs):
            last = r["body"]["messages"][-1]
            check(f"deepseek {it['task_id']}: /beta with prefix true on the partial message",
                  r["beta"] and last.get("prefix") is True and last["content"] == it["messages"][-1]["content"])
            check(f"deepseek {it['task_id']}: thinking disabled", r["body"].get("thinking") == {"type": "disabled"})
        check("deepseek: a request ending with the user's message goes to /chat/completions without prefix",
              len(recs) == 4 and not recs[3]["beta"] and "prefix" not in recs[3]["body"]["messages"][-1])

        # the negative control: a server that opens a new turn must fail the same checks
        np_ = free_port()
        start([sys.executable, str(HERE / "fakes.py"), "llama-server", "--tokenizer-gguf", a.vocab, "--record",
               str(out / "llama-no-prefill.jsonl"), "--port", str(np_), "--no-prefill-assistant"],
              "llama-no-prefill.log", np_, "fakes.py llama-server --no-prefill-assistant")
        url = proxy("llamacpp", f"http://127.0.0.1:{np_}/v1/chat/completions")
        try:
            post(url, {"messages": items[0]["messages"], "max_tokens": 64, "temperature": 0})
            status = 200
        except urllib.error.HTTPError as e:
            status = e.code
        check("negative control: the proxy answers 502 to a reply that does not start with the message", status == 502,
              f"status {status}")
        r = records(out / "llama-no-prefill.jsonl")[0]
        bad = C.check_prompt(r["prompt"], items[0]["messages"][-1]["content"])
        check("negative control: a new assistant turn fails the prompt checks", len(bad) >= 2, repr(bad))
        print("  the new turn's failures: " + "; ".join(bad))
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    print(f"continuation_test.py: {'PASS' if not failed else 'FAIL'} ({len(failed)} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
