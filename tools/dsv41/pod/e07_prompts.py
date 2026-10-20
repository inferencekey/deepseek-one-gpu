#!/usr/bin/env python3
"""The speed prompts of the pod sessions (tools/dsv41/pod/e07.sh): token-id prompts of exact lengths from a fixed text corpus,
tokenized by the server's own tokenizer, so every run reads the same ids and records them.

  e07_prompts.py build --fork DIR --vocab GGUF --out DIR [--commit C]
      the corpus of pod/prefill_prompts.py (the fork's sources at commit C, default its base 5210c7c5, one file after
      another), cut into consecutive segments, one per prompt, so no prompt shares a prefix with another (a prefix
      cache cannot reuse one for the next): each segment takes whole corpus lines until tools/strata_tokenizer.py
      (the tokenizer serve/server.py builds from the GGUF's metadata) encodes it to at least
      the prompt's length, and the prompt is the first that many ids. Writes <name>.ids (the ids, space separated)
      per prompt and manifest.json (lengths, sha256 of each file, the corpus lines each took, the commit, the
      tokenizer GGUF's sha256).
  e07_prompts.py check DIR
      every prompt file's sha256 and length against the manifest; exits 1 otherwise.
  e07_prompts.py turns DIR --vocab GGUF --out FILE
      the e07b plan's two-turn chat: a JSON list of two user messages, the text of p16k (its ids
      decoded by the same tokenizer) and a short follow-up (TURN2), for engine/two_turn_check.py --turns.

The prompts (PROMPTS): short (128 tokens, the 480-token decode at short context), p16k (16,384: the 16K TTFT, turn 1
of the 16K conversation), turn2 (512 new tokens of its second turn), p64k (65,536: TTFT and decode at 64K), p256k
(261,664 = 262,144 - 480: the 256K TTFT, then 480 decode steps that fill the 262,144-position context).
"""

import argparse
import hashlib
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "..", ".."))
import prefill_prompts  # noqa: E402
import strata_tokenizer  # noqa: E402

FORMAT = "strata-e07-prompts/1"
PROMPTS = [("short", 128), ("p16k", 16384), ("turn2", 512), ("p64k", 65536), ("p256k", 262144 - 480)]
TURN2 = "In three sentences, what does the code above do?"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(a):
    tok = strata_tokenizer.Tokenizer.from_gguf(a.vocab)
    lines = prefill_prompts.corpus_lines(a.fork, a.commit)
    os.makedirs(a.out, exist_ok=True)
    man = {"format": FORMAT, "commit": a.commit, "tokenizer_gguf": os.path.abspath(a.vocab),
           "tokenizer_sha256": sha256_file(a.vocab), "prompts": []}
    line_no = 0
    for name, n in PROMPTS:
        first, text, ids, est = line_no, [], [], 0
        # whole lines until the segment reaches n ids: batches of 64 lines counted on their own until the estimate
        # reaches n, then the joined text encoded (a merge or a pre-tokenizer split may cross a batch's end), more
        # batches while it falls short
        while len(ids) < n:
            batch = []
            for _ in range(64):
                line = next(lines, None)
                if line is None:
                    sys.exit(f"e07_prompts.py: the corpus ends after {line_no} lines, before {name} has {n} ids")
                batch.append(line)
                line_no += 1
            text += batch
            est += len(tok.encode("".join(batch)))
            if est >= n:
                ids = tok.encode("".join(text))
        ids = ids[:n]
        path = os.path.join(a.out, f"{name}.ids")
        with open(path, "w") as f:
            f.write(" ".join(map(str, ids)) + "\n")
        man["prompts"].append({"name": name, "tokens": n, "file": f"{name}.ids", "sha256": sha256_file(path),
                               "corpus_lines": [first, line_no]})
        print(f"{path}: {n} ids from corpus lines {first} to {line_no - 1}")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"e07_prompts.py: {len(PROMPTS)} prompts in {a.out} ({sum(n for _, n in PROMPTS)} ids, corpus "
          f"{a.commit[:8]}, tokenizer {man['tokenizer_sha256'][:12]})")


def check(d):
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    bad = 0
    if man.get("format") != FORMAT or [(p["name"], p["tokens"]) for p in man["prompts"]] != PROMPTS:
        print(f"FAIL {d}: the manifest is not {FORMAT} with the prompts {PROMPTS}")
        return 1
    for p in man["prompts"]:
        path = os.path.join(d, p["file"])
        n = len(open(path).read().split())
        good = sha256_file(path) == p["sha256"] and n == p["tokens"]
        bad += not good
        print(f"{'ok  ' if good else 'FAIL'} {p['name']}: {n} ids (want {p['tokens']}), sha256 "
              f"{'matches' if good else 'or length differs from'} the manifest")
    return 1 if bad else 0


def turns(a):
    if check(a.dir):
        return 1
    tok = strata_tokenizer.Tokenizer.from_gguf(a.vocab)
    with open(os.path.join(a.dir, "p16k.ids")) as f:
        text = tok.decode([int(x) for x in f.read().split()])
    with open(a.out, "w") as f:
        json.dump([text, TURN2], f, ensure_ascii=False)
    print(f"e07_prompts.py: {a.out}: turn 1 the text of p16k ({len(text)} characters, {len(tok.encode(text))} ids "
          f"re-encoded), turn 2 {TURN2!r}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--fork", required=True)
    b.add_argument("--vocab", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--commit", default=prefill_prompts.BASE_COMMIT)
    c = sub.add_parser("check")
    c.add_argument("dir")
    t = sub.add_parser("turns")
    t.add_argument("dir")
    t.add_argument("--vocab", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "turns":
        return turns(a)
    if a.cmd == "build":
        build(a)
        return 0
    return check(a.dir)


if __name__ == "__main__":
    sys.exit(main())
