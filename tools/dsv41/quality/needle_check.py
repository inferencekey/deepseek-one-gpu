#!/usr/bin/env python3
"""tools/dsv41/quality/needle_check.py - the long-context needle check of the quality runs.

Classic passkey retrieval: one random 7-digit number (random.Random(20261011), one per length) hidden at 50% depth in
a fixed filler sentence repeated, then the question. Each prompt is sized with the GGUF tokenizer the engine uses
(tools/strata_tokenizer.py, bit for bit the fork's): the user message rendered with the
model's chat template (deepseek-v4.1 profile, thinking off, the generation prompt) is the largest whole number of
filler sentences that keeps the prompt at most LENGTH - 64 tokens (room for the 32-token answer and the server's
slack), so a 16K, 64K or 256K context holds it.

  needle_check.py build --vocab GGUF --out needle.json [--lengths 16k,64k,256k]
      writes the prompts: per length its name, the context it is sized for, its prompt tokens, the depth, the
      expected answer and the chat messages
  needle_check.py run URL --prompts needle.json --out results.json [--lengths 16k]
      posts each prompt to URL (a /v1/chat/completions: the api_proxy.py of the system, so thinking is off and the
      usage is logged) greedy with max_tokens 32, and scores exact match: the reply, stripped of white space and a
      final period, equals the number (whether it contains the number is reported too). Exits 1 unless every
      prompt is answered with 200 and matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SEED = 20261011
DEPTH = 0.5
ROOM = 64
MAX_TOKENS = 32
LENGTHS = {"16k": 16384, "64k": 65536, "256k": 262144}
INTRO = ("There is an important piece of information hidden inside a lot of irrelevant text. Find it and memorize "
         "it. I will quiz you about it.\n\n")
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
NEEDLE = "The pass key is {key}. Remember it. {key} is the pass key. "
QUESTION = "\n\nWhat is the pass key? Answer with the 7-digit number only."


def text_of(n: int, key: str) -> str:
    half = int(n * DEPTH)
    return INTRO + FILLER * half + NEEDLE.format(key=key) + FILLER * (n - half) + QUESTION


def build(vocab: str, names: list[str]) -> dict:
    from serve.frontend import ChatTemplate
    from serve.model_profile import DeepSeekV41Profile
    prof = DeepSeekV41Profile(vocab)
    tpl = ChatTemplate(source=prof.template_source)
    kw = prof.template_kwargs({"enable_thinking": False})

    def count(n, key):
        msgs = [{"role": "user", "content": text_of(n, key)}]
        return len(prof.tokenizer.encode(tpl.render(msgs, add_generation_prompt=True, **kw), parse_special=True))

    rng = random.Random(SEED)
    prompts = []
    for name in names:
        ctx = LENGTHS[name]
        key = str(rng.randrange(1_000_000, 10_000_000))
        target = ctx - ROOM
        per = (count(2000, key) - count(1000, key)) / 1000      # tokens per filler sentence
        n = int((target - count(0, key)) / per)
        while count(n, key) > target:
            n -= 1
        while count(n + 1, key) <= target:
            n += 1
        prompts.append({"name": name, "context": ctx, "prompt_tokens": count(n, key), "filler_sentences": n,
                        "depth": DEPTH, "answer": key,
                        "messages": [{"role": "user", "content": text_of(n, key)}]})
    return {"seed": SEED, "vocab": str(vocab), "vocab_sha256": hashlib.sha256(Path(vocab).read_bytes()).hexdigest(),
            "max_tokens": MAX_TOKENS, "filler": FILLER, "prompts": prompts}


def score(reply: str, answer: str) -> dict:
    norm = reply.strip().rstrip(".").strip()
    return {"exact": norm == answer, "contains": answer in reply}


def run(url: str, prompts: dict, names: list[str] | None) -> list[dict]:
    out = []
    for p in prompts["prompts"]:
        if names and p["name"] not in names:
            continue
        body = {"messages": p["messages"], "max_tokens": prompts["max_tokens"], "temperature": 0, "seed": 1234}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                status, data = r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            status, data = e.code, {"error": e.read().decode("utf-8", "replace")[:500]}
        seconds = round(time.time() - t0, 2)
        reply = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "" if status == 200 else ""
        res = {"name": p["name"], "context": p["context"], "prompt_tokens": p["prompt_tokens"], "answer": p["answer"],
               "status": status, "seconds": seconds, "reply": reply, "usage": data.get("usage"),
               "error": data.get("error"), **score(reply, p["answer"])}
        out.append(res)
        print(f"{'ok  ' if res['exact'] else 'FAIL'} needle {p['name']} ({p['prompt_tokens']} prompt tokens): "
              f"{status}, {seconds} s, reply {reply[:40]!r}, expected {p['answer']}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--vocab", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--lengths", default="16k,64k,256k")
    r = sub.add_parser("run")
    r.add_argument("url")
    r.add_argument("--prompts", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--lengths", default="", help="comma-separated names (default: every prompt)")
    a = ap.parse_args()
    if a.cmd == "build":
        names = a.lengths.split(",")
        bad = [n for n in names if n not in LENGTHS]
        if bad:
            raise SystemExit(f"needle_check.py: unknown lengths {bad}: {', '.join(LENGTHS)}")
        d = build(a.vocab, names)
        a.out.write_text(json.dumps(d, ensure_ascii=False) + "\n")
        for p in d["prompts"]:
            print(f"needle {p['name']}: {p['prompt_tokens']} prompt tokens (at most {p['context'] - ROOM}), "
                  f"{p['filler_sentences']} filler sentences, the number at depth {p['depth']}")
        print(f"needle_check.py build: {a.out} sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
        return 0
    prompts = json.loads(a.prompts.read_text())
    names = [n for n in a.lengths.split(",") if n]
    missing = [n for n in names if n not in {p["name"] for p in prompts["prompts"]}]
    if missing:
        raise SystemExit(f"needle_check.py: {a.prompts} has no prompt {missing}")
    res = run(a.url, prompts, names)
    a.out.write_text(json.dumps({"prompts": str(a.prompts), "results": res}, indent=1, ensure_ascii=False) + "\n")
    ok = res and all(x["status"] == 200 and x["exact"] for x in res)
    print(f"needle_check.py run: {'PASS' if ok else 'FAIL'}, {sum(x['exact'] for x in res)} of {len(res)} exact")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
