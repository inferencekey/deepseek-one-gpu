#!/usr/bin/env python3
"""tools/dsv41/quality/continuation.py - is lm-eval's HumanEval gen_prefix continued, not answered with a new turn?

humaneval_instruct ends each request with a partial assistant message ("Here is the completed function:\\n```python\\n"
plus the function's signature and docstring) that the model must continue; a server that opens a new assistant turn
instead restarts the code block and every item fails (the defect that invalidated the vLLM-EXL3 experiment's first
API and EXL3 runs). humaneval_continuation_items.json holds 3 such requests (the subsets' first 3 HumanEval items, as
the 2026-10-06 API run sent them, with that run's reply).

check_prompt() reads a DeepSeek-V4.1 prompt as rendered for one of them: it must end with the partial message's text,
with exactly one <｜Assistant｜> after the last <｜User｜> and no <｜end▁of▁sentence｜> after it (a new turn closes
the partial message with the end of sentence and opens a second <｜Assistant｜>). check_reply() reads the reply: it
continues the code, so it does not start the message again ("Here is the completed function" or the code fence).

    continuation.py apply-template URL [--out F]   the fork's llama-server: its /apply-template prompt of each item
                                                   (the same rendering as its chat route, --prefill-assistant)
    continuation.py chat URL [--out F]             any /v1/chat/completions (the proxy, a server): each item's
                                                   reply, greedy, the item's own stop strings and max tokens

Each prints one `ok`/`FAIL` line per item and check and exits 1 on any FAIL; --out saves the prompts or replies (the
chat mode's replies are the 3 items a person checks by hand before the full run).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ITEMS = Path(__file__).resolve().parent / "humaneval_continuation_items.json"
USER, ASSISTANT, EOS = "<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"


def items() -> list[dict]:
    return json.loads(ITEMS.read_text())["items"]


def check_prompt(prompt: str, partial: str) -> list[str]:
    """The failures of a rendered prompt that should continue `partial` (empty: it does)."""
    bad = []
    if not prompt.endswith(partial):
        bad.append("the prompt does not end with the partial assistant message")
    tail = prompt[prompt.rfind(USER):] if USER in prompt else prompt
    if tail.count(ASSISTANT) != 1:
        bad.append(f"{tail.count(ASSISTANT)} {ASSISTANT} after the last {USER}, not 1 (a new turn opens a second)")
    if EOS in tail:
        bad.append(f"{EOS} after the last {USER}: the partial message was closed")
    return bad


def check_reply(reply: str, partial: str) -> list[str]:
    """The failures of a reply to a request that ends with `partial` (empty: it continues the code)."""
    first = partial.splitlines()[0]
    if reply.lstrip().startswith((first, "```")):
        return [f"the reply starts the message again: {reply[:60]!r}"]
    return []


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("mode", choices=["apply-template", "chat"])
    ap.add_argument("url", help="the server's base URL (apply-template) or its /v1/chat/completions (chat)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    failed, saved = 0, []
    for it in items():
        partial = it["messages"][-1]["content"]
        if a.mode == "apply-template":
            got = post(a.url.rstrip("/") + "/apply-template", {"messages": it["messages"]})["prompt"]
            bad = check_prompt(got, partial)
        else:
            got = post(a.url, {"messages": it["messages"], "max_tokens": it["max_gen_toks"], "temperature": 0,
                               "stop": it["until"][:4], "seed": 1234})["choices"][0]["message"]["content"] or ""
            bad = check_reply(got, partial)
        saved.append({"task_id": it["task_id"], a.mode: got, "failures": bad})
        for b in bad:
            print(f"FAIL {it['task_id']}: {b}")
        if not bad and a.mode == "apply-template":
            print(f"ok   {it['task_id']}: the prompt continues the message")
        elif not bad:
            print(f"ok   {it['task_id']}: the reply continues the code: {got[:60]!r}")
        failed += bool(bad)
    if a.out:
        a.out.write_text(json.dumps(saved, indent=1, ensure_ascii=False) + "\n")
    print(f"continuation.py {a.mode}: {'PASS' if not failed else 'FAIL'}, {len(saved) - failed} of {len(saved)} items")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
