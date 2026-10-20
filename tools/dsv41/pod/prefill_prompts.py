#!/usr/bin/env python3
"""The prefill prompts (tools/dsv41/pod/t04a.sh --prefill): deterministic code-corpus prompts of 4K, 8K, 16K and
32K DeepSeek-V4.1 tokens, their token counts measured with the model's own tokenizer.

  prefill_prompts.py vocab HEADER OUT.gguf
      a vocab-only GGUF from the head of DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf (its first 64 MiB hold the
      whole metadata: 129280 tokens, 127741 merges, pre-tokenizer joyai-llm), without the split keys and the tensors,
      as the fork's models/ggml-vocab-*.gguf files are; llama-tokenize loads it with vocab_only
  prefill_prompts.py build --fork DIR --tokenizer CMD --out DIR [--commit C] [--targets N,N,...]
      the prompts from the fork's sources at commit C (default the base 5210c7c5: git show, so the working tree does
      not matter): every *.c, *.cpp, *.h, *.cu, *.cuh under src/ and ggml/src/ in path order, each after a
      "// ===== path =====" line, one corpus; prompt i takes the lines right after prompt i-1's (no two share a
      prefix, so a server cannot reuse one's cache for another), between a fixed head and question, as many whole
      lines as keep it at or under its target. CMD is a llama-tokenize command line without the file
      (... -m vocab.gguf --show-count --log-disable -f); it is run once per candidate.
      Writes prefill-<name>.txt per prompt and manifest.json (counts, sha256, the corpus lines each prompt took).
  prefill_prompts.py check DIR [--tolerance 0.02]
      the manifest's prompts: each file's sha256 and its measured count within the tolerance of its target; exits 1
      otherwise. Prints one line per prompt.

The targets: 4096, 8192, 16384 and 32704 (32768 less the 64 generated tokens of the greedy check, so a server with
--ctx-size 32768 holds every prompt with room to answer). The prompts are raw completion text: the GGUF has
tokenizer.ggml.add_bos_token false, so llama-server's /completion evaluates exactly the counted tokens, which
t04a.sh checks against the server's prompt_n.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
import gguf_header  # noqa: E402

FORMAT = "strata-prefill-prompts/1"
BASE_COMMIT = "5210c7c5ed61dddaee6ed476623abf4b63093d16"
DEFAULT_TARGETS = [4096, 8192, 16384, 32704]
EXTENSIONS = (".c", ".cpp", ".h", ".cu", ".cuh")
HEAD = "The text below is the source code of llama.cpp and ggml, one file after another.\n\n"
QUESTION = "\n\nQuestion: in two sentences, what do the source files above implement?\nAnswer:"


class PromptError(Exception):
    pass


def name_of(target):
    return f"{round(target / 1024)}k"


def make_vocab(header_path, out_path):
    header = gguf_header.read_header(header_path)
    if not any(k == "tokenizer.ggml.tokens" for k in header.metadata):
        raise PromptError(f"{header_path}: no tokenizer.ggml.tokens in the metadata")
    header.metadata = {k: v for k, v in header.metadata.items() if not k.startswith("split.")}
    header.tensors = []
    gguf_header.write_gguf(out_path, header, lambda name: b"")
    print(f"vocab-only GGUF {out_path}: {len(header.metadata)} keys, "
          f"{len(header.metadata['tokenizer.ggml.tokens'][1])} tokens, no tensors")


def corpus_lines(fork, commit):
    names = subprocess.run(["git", "-C", fork, "ls-tree", "-r", "--name-only", commit, "--", "src", "ggml/src"],
                           check=True, capture_output=True, text=True).stdout.split("\n")
    files = sorted(n for n in names if n.endswith(EXTENSIONS))
    for path in files:
        text = subprocess.run(["git", "-C", fork, "show", f"{commit}:{path}"], check=True, capture_output=True
                              ).stdout.decode("utf-8", errors="replace")
        yield f"// ===== {path} =====\n"
        for line in text.splitlines(keepends=True):
            yield line
        if not text.endswith("\n"):
            yield "\n"


def count_tokens(tokenizer, text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        r = subprocess.run(shlex.split(tokenizer) + [path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    m = re.search(r"Total number of tokens: (\d+)", r.stdout)
    if r.returncode != 0 or m is None:
        raise PromptError(f"tokenizer failed (exit {r.returncode}): {r.stderr.strip()[-400:]}")
    return int(m.group(1))


def build(fork, commit, tokenizer, out, targets):
    os.makedirs(out, exist_ok=True)
    # enough corpus for every prompt at 2 characters per token, the low end for code
    need_chars = 2 * 6 * sum(targets)
    lines, chars = [], 0
    for line in corpus_lines(fork, commit):
        lines.append(line)
        chars += len(line)
        if chars >= need_chars:
            break
    start = 0
    prompts = []
    for target in targets:
        def text_of(n):
            return HEAD + "".join(lines[start:start + n]) + QUESTION
        # the largest whole-line count at or under the target: a doubling search, then bisection
        lo, hi = 0, 64
        while True:
            if start + hi > len(lines):
                raise PromptError(f"the corpus ran out at {len(lines)} lines for the {target}-token prompt")
            if count_tokens(tokenizer, text_of(hi)) > target:
                break
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if count_tokens(tokenizer, text_of(mid)) <= target:
                lo = mid
            else:
                hi = mid
        text = text_of(lo)
        tokens = count_tokens(tokenizer, text)
        name = name_of(target)
        file = f"prefill-{name}.txt"
        with open(os.path.join(out, file), "w") as f:
            f.write(text)
        prompts.append({"name": name, "target": target, "tokens": tokens, "file": file,
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "corpus_lines": [start, start + lo],
                        "first_line": lines[start].rstrip("\n")[:120]})
        print(f"{name}: {tokens} tokens (target {target}, {100.0 * (tokens - target) / target:+.2f}%), corpus lines "
              f"{start} to {start + lo}, {len(text)} characters, from '{lines[start].rstrip()[:60]}'")
        start += lo
    manifest = {"format": FORMAT, "source": {"repo": "llama-dsv41", "commit": commit, "paths": ["src", "ggml/src"],
                                             "extensions": list(EXTENSIONS)},
                "tokenizer": tokenizer, "add_bos": False, "prompts": prompts}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(prompts)} prompts and manifest.json to {out}")


def check(directory, tolerance):
    path = os.path.join(directory, "manifest.json")
    try:
        with open(path) as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        raise PromptError(f"{path}: {e}")
    if manifest.get("format") != FORMAT:
        raise PromptError(f"{path}: format {manifest.get('format')!r}, expected {FORMAT!r}")
    ok = True
    for p in manifest["prompts"]:
        with open(os.path.join(directory, p["file"]), "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        off = (p["tokens"] - p["target"]) / p["target"]
        good = digest == p["sha256"] and abs(off) <= tolerance
        ok &= good
        print(f"{'ok' if good else 'FAIL'} {p['name']}: {p['tokens']} tokens, target {p['target']} ({100 * off:+.2f}%, "
              f"tolerance {100 * tolerance:.0f}%), sha256 {'matches' if digest == p['sha256'] else 'DIFFERS'}, "
              f"{p['file']}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vocab")
    v.add_argument("header")
    v.add_argument("out")
    b = sub.add_parser("build")
    b.add_argument("--fork", required=True)
    b.add_argument("--commit", default=BASE_COMMIT)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)))
    c = sub.add_parser("check")
    c.add_argument("dir")
    c.add_argument("--tolerance", type=float, default=0.02)
    a = ap.parse_args()
    try:
        if a.cmd == "vocab":
            make_vocab(a.header, a.out)
        elif a.cmd == "build":
            build(a.fork, a.commit, a.tokenizer, a.out, [int(t) for t in a.targets.split(",")])
        elif not check(a.dir, a.tolerance):
            return 1
    except (PromptError, gguf_header.GgufError, subprocess.CalledProcessError) as e:
        print(f"prefill_prompts.py: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
