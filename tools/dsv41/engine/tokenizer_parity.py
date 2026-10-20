#!/usr/bin/env python3
"""tools/dsv41/engine/tokenizer_parity.py - the server's DeepSeek-V4.1 tokenizer against the fork's.

Text becomes token ids in serve/server.py, with tools/strata_tokenizer.py built from the
GGUF's tokenizer metadata, and those ids must be the fork's llama-tokenize's bit for bit. For each vocab-only GGUF
given (the real tokenizer $VOCAB and tiny_vocab.py's 512-token cut of it), every line of the corpus and the whole
corpus file as one text are tokenized by both, with the special tokens parsed (llama-tokenize's default, how the
server encodes a rendered prompt), and the lines that spell a special token's characters again without
(--no-parse-special: Strata's parse_special=False); then decode(encode(text)) must give the text back. Exits 1 on the
first vocab with a difference, printing the first ones.

    python3 tools/dsv41/engine/tokenizer_parity.py --llama-tokenize "env LD_LIBRARY_PATH=$fork $fork/llama-tokenize" \\
        --corpus tools/dsv41/engine/tokenizer_corpus.txt $VOCAB tiny-vocab.gguf
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import strata_tokenizer as ST  # noqa: E402

SPECIAL_HINTS = ("<", "｜", "▁")   # characters of DeepSeek-V4.1's special tokens: those lines are run plain too


def llama_ids(cmd: list[str], vocab: str, text: str, parse_special: bool) -> list[int]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".txt", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        args = cmd + ["-m", vocab, "-f", path, "--ids", "--log-disable", "--no-escape"]
        if not parse_special:
            args.append("--no-parse-special")
        r = subprocess.run(args, capture_output=True, text=True)
    finally:
        Path(path).unlink()
    lines = [x for x in r.stdout.splitlines() if x.startswith("[")]
    if r.returncode != 0 or len(lines) != 1:
        raise SystemExit(f"tokenizer_parity.py: llama-tokenize failed (exit {r.returncode}): {r.stderr[-500:]}")
    return json.loads(lines[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--llama-tokenize", required=True, help="the fork's llama-tokenize command line")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("vocab", nargs="+")
    a = ap.parse_args()
    cmd = shlex.split(a.llama_tokenize)
    text = Path(a.corpus).read_text(encoding="utf-8")
    lines = [x for x in text.split("\n") if x]
    if len(lines) < 200:
        raise SystemExit(f"tokenizer_parity.py: {a.corpus}: {len(lines)} lines, the corpus needs at least 200")
    samples = [(x, True) for x in lines] + [(text, True)] + [(x, False) for x in lines if any(h in x for h in SPECIAL_HINTS)]
    failed = 0
    for vocab in a.vocab:
        t0 = time.time()
        tok = ST.Tokenizer.from_gguf(vocab)
        bad, n_ids, n_special = [], 0, 0
        for s, parse_special in samples:
            ref = llama_ids(cmd, vocab, s, parse_special)
            ours = tok.encode(s, parse_special=parse_special)
            n_ids += len(ref)
            n_special += sum(1 for t in ref if tok.token_types[t] in (3, 4))
            if ours != ref:
                bad.append((s, parse_special, ref, ours))
            elif parse_special and tok.decode(ours) != s:
                bad.append((s, parse_special, "round trip", tok.decode(ours)))
        n_plain = sum(1 for _, p in samples if not p)
        print(f"tokenizer_parity.py: {vocab}: {len(tok.tokens)} tokens, pre {tok.pre}: {len(samples) - n_plain} texts "
              f"({len(lines)} lines and the whole corpus) with special tokens parsed and {n_plain} without, {n_ids} ids "
              f"({n_special} special): {len(bad)} differ from llama-tokenize or do not round trip ({time.time() - t0:.1f} s)")
        for s, p, ref, ours in bad[:5]:
            print(f"  FAIL {'special' if p else 'plain'} {s[:60]!r}:\n    llama-tokenize {ref}\n    strata         {ours}")
        failed += bool(bad)
    if failed:
        print(f"tokenizer_parity.py: FAIL: {failed} of {len(a.vocab)} vocabularies differ")
        return 1
    print(f"tokenizer_parity.py: PASS: {len(a.vocab)} vocabularies, every text equal to llama-tokenize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
