#!/usr/bin/env python3
"""tools/dsv41/quality/fork_prompt_test.py - do the fork's real llama-server and Strata's server get byte-identical
prompt text from the same lm-eval request? Without a GPU, no money.

The fork: a CPU-only llama-server built from $FORK_DIR (no CUDA, its build directory outside the repos) on
tiny-a.gguf with the real tokenizer (tools/dsv41/engine/tiny_vocab.py --model: its first 512 tokens, so the fork
renders the real begin-of-sentence text), the pod's chat flags (quality/pod_server.sh: --jinja --prefill-assistant
--no-reasoning-preserve -np 4 --kv-unified) and --chat-template-file set to the tokenizer.chat_template of --vocab.
Strata: the real serve/server.py (deepseek-v4.1 profile) on fakes.py's engine, as continuation_test.py runs it.

For the 3 items of humaneval_continuation_items.json (the last message is the assistant's) and one plain chat
request, thinking off:
  - the body api_proxy.py --backend llamacpp forwards, sent to the fork's /apply-template, gives the same text as the
    prompt ids Strata's engine reads (through api_proxy.py --backend strata), decoded: byte for byte;
  - the fork's chat route reads that prompt too: the prompt_tokens of the same body through the llamacpp proxy equal
    the fork's /tokenize of the /apply-template text;
  - the reply lm-eval reads through the llamacpp proxy is the fork's reply less the continued message's text, which
    the fork's reply starts with (the proxy's cut; a plain request's reply is the fork's as it is).
Then the negative control, so the test can fail: the fork's plain chat prompt with thinking on (it ends with <think>)
differs from Strata's with it off.

    python3 tools/dsv41/quality/fork_prompt_test.py --llama-server BIN --model tiny-a.gguf --vocab $VOCAB \\
        --venv $QUALITY_VENV --out DIR

Prints each prompt's tail (from the last <｜Assistant｜>) and exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import continuation as C  # noqa: E402
from continuation_test import check, failed, free_port, post, records, wait_port  # noqa: E402

PLAIN = [{"role": "user", "content": "What is 6 times 7?"}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--llama-server", required=True, help="the CPU-only fork build's llama-server")
    ap.add_argument("--model", required=True, help="tiny-a.gguf (tools/dsv41/engine/tiny_dsv41.py --config a)")
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

    def proxy(backend, upstream):
        port = free_port()
        start([py, str(HERE / "api_proxy.py"), "--backend", backend, "--port", str(port), "--upstream", upstream,
               "--usage", str(out / f"usage-{backend}.jsonl")], f"proxy-{backend}.log", port, f"the {backend} proxy")
        return f"http://127.0.0.1:{port}/v1/chat/completions"

    def forwarded(body: dict) -> dict:
        """The body api_proxy.py --backend llamacpp sends upstream (its own rewrite, in the venv that has aiohttp)."""
        r = subprocess.run([py, "-c", "import json, sys; import api_proxy as P; b = json.load(sys.stdin); "
                            "P.rewrite(b, 'llamacpp', None, 'upstream'); print(json.dumps(b))"],
                           input=json.dumps(body), capture_output=True, text=True, check=True, cwd=HERE)
        return json.loads(r.stdout)

    try:
        from serve.model_profile import DeepSeekV41Profile
        from gguf_reader import GGUFFile
        prof = DeepSeekV41Profile(a.vocab)
        for f in out.glob("*.jsonl"):
            f.unlink()
        model = out / "tiny-a-vocab.gguf"
        subprocess.run([sys.executable, str(ROOT / "tools/dsv41/engine/tiny_vocab.py"), a.vocab, str(model), "--model",
                        a.model], check=True, stdout=subprocess.DEVNULL)
        template = out / "chat-template.jinja"
        template.write_text(GGUFFile(Path(a.vocab)).metadata["tokenizer.chat_template"])

        fp = free_port()
        start([a.llama_server, "-m", str(model), "--jinja", "--chat-template-file", str(template),
               "--prefill-assistant", "--no-reasoning-preserve", "-np", "4", "--kv-unified", "--ctx-size", "8192",
               "--threads", "4", "--host", "127.0.0.1", "--port", str(fp)], "llama-server.log", fp,
              "the CPU llama-server")
        fork = f"http://127.0.0.1:{fp}"
        fork_url = proxy("llamacpp", f"{fork}/v1/chat/completions")

        sp = free_port()
        cfg = {"exe": str(HERE / "fakes.py"),
               "args": ["--tokenizer-gguf", a.vocab, "--record", str(out / "engine.jsonl"), "--ctx", "32768"],
               "model_profile": "deepseek-v4.1", "tokenizer_gguf": a.vocab, "log": str(out / "engine.log"),
               "repeat_stop_tokens": 0}
        (out / "strata-config.json").write_text(json.dumps(cfg, indent=1))
        start([sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(out / "strata-config.json"),
               "--port", str(sp)], "strata-server.log", sp, "serve/server.py", cwd=ROOT)
        strata_url = proxy("strata", f"http://127.0.0.1:{sp}/v1/chat/completions")

        reqs = [(it["task_id"], it["messages"]) for it in C.items()] + [("plain chat", PLAIN)]
        prompts = []
        for name, msgs in reqs:
            body = {"messages": msgs, "max_tokens": 2, "temperature": 0}
            post(strata_url, body)
            strata = prof.tokenizer.decode(records(out / "engine.jsonl")[-1]["ids"])
            sent = forwarded(body)
            fork_prompt = post(f"{fork}/apply-template", sent)["prompt"]
            tail = fork_prompt[fork_prompt.rfind(C.ASSISTANT):]
            print(f"  {name}: the fork's prompt from the last {C.ASSISTANT}: {tail[:60]!r} ... {tail[-40:]!r} "
                  f"({len(fork_prompt)} characters)")
            same = fork_prompt == strata
            at = next((i for i, (x, y) in enumerate(zip(fork_prompt, strata)) if x != y),
                      min(len(fork_prompt), len(strata)))
            check(f"{name}: the fork's /apply-template and Strata's engine prompt are byte-identical", same,
                  f"at character {at}: fork {fork_prompt[at - 20:at + 40]!r}, strata {strata[at - 20:at + 40]!r}")
            n = len(post(f"{fork}/tokenize", {"content": fork_prompt, "add_special": False,
                                              "parse_special": True})["tokens"])
            # the tiny random model's own tokens may not parse as the template's reply format (the fork's 500), so
            # its 2 reply tokens are forced to "a"
            forced = {"logit_bias": [["a", 100]]}
            proxied = post(fork_url, {**body, **forced})
            raw = post(f"{fork}/v1/chat/completions", {**sent, **forced})["choices"][0]["message"]["content"]
            reply = proxied["choices"][0]["message"]["content"]
            check(f"{name}: the fork's chat route reads that prompt ({n} tokens)",
                  proxied["usage"]["prompt_tokens"] == n, f"prompt_tokens {proxied['usage']['prompt_tokens']}")
            echo = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else ""
            check(f"{name}: the reply through the proxy is the fork's ({len(raw)} characters) less the continued "
                  f"message's {len(echo)}", raw == echo + "aa" and reply == "aa",
                  f"fork {raw[-40:]!r}, proxy {reply!r}")
            prompts.append({"request": name, "fork": fork_prompt, "strata": strata, "identical": same,
                            "fork_reply": raw, "proxy_reply": reply})
        (out / "prompts.json").write_text(json.dumps(prompts, indent=1, ensure_ascii=False) + "\n")

        # (a continued message renders the same either way: the template closes it as a past turn, with </think>)
        on = forwarded({"messages": PLAIN, "max_tokens": 2, "temperature": 0})
        on["chat_template_kwargs"]["enable_thinking"] = True
        thinking = post(f"{fork}/apply-template", on)["prompt"]
        check("negative control: the fork's plain chat prompt with thinking on differs from Strata's with it off",
              thinking != prompts[-1]["strata"], "the same text")
        print(f"  the negative control's tail: {thinking[thinking.rfind(C.ASSISTANT):][:40]!r}")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    print(f"fork_prompt_test.py: {'PASS' if not failed else 'FAIL'} ({len(failed)} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
