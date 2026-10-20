#!/usr/bin/env python3
"""tools/dsv41/engine/llama_unicode.py - llama.cpp's codepoint table for tools/strata_tokenizer.py.

The fork's tokenizer runs DeepSeek-V4.1's `joyai-llm` pre-tokenizer through std::regex on a "collapsed" text in which
every non-ASCII codepoint stands for its category (llama.cpp src/unicode.cpp, unicode_regex_split), and the category
is llama.cpp's own table (src/unicode-data.cpp, generated from one Unicode version), not Python's unicodedata (which
follows the Python release). strata_tokenizer.py reads that table from tools/llama_unicode_flags.txt, which this
script writes from the fork's sources and verify.sh checks against them:

    python3 tools/dsv41/engine/llama_unicode.py write $FORK_DIR tools/llama_unicode_flags.txt
    python3 tools/dsv41/engine/llama_unicode.py check $FORK_DIR tools/llama_unicode_flags.txt

The file: comment lines, then `W <hex>` per whitespace codepoint (unicode_set_whitespace) and `<start hex> <flags
hex>` per range of unicode_ranges_flags (each range lasts to the next start; the flags are one category bit of
unicode.h's unicode_cpt_flags, 0 past the last codepoint).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HEADER = ("# llama.cpp's unicode_ranges_flags and unicode_set_whitespace (src/unicode-data.cpp of the fork), for\n"
          "# tools/strata_tokenizer.py's joyai-llm pre-tokenizer; written by tools/dsv41/engine/llama_unicode.py\n")


def table(fork: str) -> str:
    src = (Path(fork) / "src" / "unicode-data.cpp").read_text(encoding="utf-8")

    def block(name: str) -> str:
        a = src.index(name + " = {")
        return src[a:src.index("};", a)]

    ranges = re.findall(r"\{0x([0-9A-F]+), 0x([0-9A-F]+)\}", block("unicode_ranges_flags"))
    spaces = re.findall(r"0x([0-9A-F]+),", block("unicode_set_whitespace"))
    if not ranges or not spaces or int(ranges[0][0], 16) != 0:
        raise SystemExit(f"llama_unicode.py: {fork}/src/unicode-data.cpp: no tables in the expected form")
    lines = [HEADER]
    lines += [f"W {int(c, 16):06X}\n" for c in spaces]
    lines += [f"{int(a, 16):06X} {int(f, 16):04X}\n" for a, f in ranges]
    return "".join(lines)


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in ("write", "check"):
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    mode, fork, out = sys.argv[1:]
    text = table(fork)
    if mode == "write":
        Path(out).write_text(text, encoding="utf-8")
        print(f"llama_unicode.py: {out}: {text.count(chr(10)) - 2} lines from {fork}/src/unicode-data.cpp")
        return 0
    if Path(out).read_text(encoding="utf-8") != text:
        print(f"llama_unicode.py: FAIL: {out} differs from {fork}/src/unicode-data.cpp (write it again)")
        return 1
    print(f"llama_unicode.py: {out} matches {fork}/src/unicode-data.cpp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
