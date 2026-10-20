"""tools/strata_tokenizer.py - the tokenizer, as the reference implementation of record.

WHY THIS EXISTS AND WHERE IT SITS.  The GGUF carries the whole tokenizer in metadata: `tokenizer.ggml.model`
= `gpt2`, 248,320 `tokens`, 247,587 `merges`, `token_type`, `pre` = `qwen35`, and a Jinja `chat_template`.
None of that needs the 35 GB of weights, so it is extracted into the pack's `tokenizer/` directory
(docs/pack-format.md §6) and this module is what reads it back.  A C++ port follows and is tested against
this one.

THE DECISIVE PROPERTY IS THE ROUND TRIP.  Byte-level BPE maps each BYTE to a printable unicode character so
that any UTF-8 input is representable with no UNK token.  Every failure mode of that mapping - the wrong
offset for a byte, a merge applied in the wrong order, a pre-tokenizer split that drops a character - still
produces plausible token ids.  `decode(encode(s)) == s` is the check that sees them, and it is exact because
the byte layer is lossless.  It is asserted over a corpus chosen to hit the boundaries: multi-byte UTF-8,
emoji (4-byte), combining marks, whitespace runs, and C0 control bytes.

DEEPSEEK-V4.1 (E06, `pre` = `joyai-llm`).  llama.cpp runs that pre-tokenizer as three regexes, each splitting the
pieces of the one before, through std::regex on a "collapsed" text: every non-ASCII codepoint replaced by one byte
for its category from llama.cpp's own table (src/unicode.cpp unicode_regex_split, src/unicode-data.cpp), and every
`\\p{..}` of the regex by that byte plus the category's ASCII members.  `LlamaRegexSplit` does exactly that with
Python's `re` (ASCII `\\s`, as std::regex's), the table read from tools/llama_unicode_flags.txt
(tools/dsv41/engine/llama_unicode.py), so a codepoint's category is llama.cpp's and not this Python's unicodedata.
Special tokens are split out as llama.cpp's tokenizer_st_partition does: the longest token text first, each over the
text fragments the earlier ones left.  A BPE piece the vocabulary lacks becomes its bytes' tokens (llama.cpp's byte
fallback).  The `qwen35` path above is unchanged, and only it needs the `regex` module.
"""
from __future__ import annotations

import bisect
import json
import pathlib
import re
import sys

try:
    import regex   # the qwen35 pre-tokenizer's classes; joyai-llm needs only `re`
except ImportError:
    regex = None

# ------------------------------------------------------------------ byte <-> unicode (GPT-2 byte-level BPE)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte encoder: 256 bytes -> 256 printable characters, reversibly.

    Bytes 33..126, 161..172 and 174..255 map to themselves; the remaining 68 (space, newline, the C0
    controls and the high range that would be invisible) are shifted into 256+n so that no byte is
    unprintable.  The shift is the whole trick and getting its ORDER wrong is invisible in the output ids.
    """
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# The `qwen35` pre-tokenizer, transcribed from the ORACLE rather than from the family resemblance:
# `.ref/llama.cpp/src/llama-vocab.cpp` L396, `case LLAMA_VOCAB_PRE_TYPE_QWEN35`.  The commented-out line
# above it is the `tokenizer.json` original, and llama.cpp's active version differs from it - it spells the
# contraction classes out instead of using `(?i:...)`, which is behaviourally the same.
#
# THE `\p{M}` IS THE WHOLE POINT OF THIS BEING TRANSCRIBED.  The QWEN3 pattern two cases earlier (L389) is
# the same shape with `\p{L}` where this has `[\p{L}\p{M}]` and without `\p{M}` in the punctuation negation.
# Writing the QWEN3 pattern for `qwen35` looks right, round-trips perfectly, and is wrong: a combining mark
# is `\p{M}`, so without it the mark gets swallowed into the following punctuation run and a token like `_j`
# never forms.  It cost 3 strings out of 1875 and only `tokenize_oracle_check` could see it.
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


# The `joyai-llm` pre-tokenizer: the fork's (llama-dsv41) src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM (shared
# with DEEPSEEK3_LLM), the three C++ literals as written there (Python's escapes are the same ones).
JOYAI_LLM_EXPRS = [
    "\\p{N}{1,3}",
    "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|"
    "\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
]

# src/unicode.h unicode_cpt_flags categories, and unicode_regex_split's k_ucat_enum / k_ucat_cpt / k_ucat_map
_NUMBER, _LETTER, _MARK, _PUNCT, _SYMBOL = 0x02, 0x04, 0x10, 0x20, 0x40
_UCAT = {"\\p{N}": _NUMBER, "\\p{L}": _LETTER, "\\p{P}": _PUNCT, "\\p{M}": _MARK, "\\p{S}": _SYMBOL,
         "\\p{Lu}": _LETTER, "\\p{Ll}": _LETTER, "\\p{Lt}": _LETTER, "\\p{Lm}": _LETTER, "\\p{Lo}": _LETTER}
_UCAT_CPT = {_NUMBER: "\xd1", _LETTER: "\xd2", _PUNCT: "\xd3", _MARK: "\xd4", _SYMBOL: "\xd5"}
_UCAT_MAP = {_NUMBER: "0-9", _LETTER: "A-Za-z", _PUNCT: "!-#%-*,-/:-;?-@\\[-\\]_\\{\\}", _MARK: "",
             _SYMBOL: "\\$\\+<->^`\\|\\~"}


class LlamaRegexSplit:
    """llama.cpp's unicode_regex_split for regexes it gives std::regex (none of these has a custom splitter)."""

    TABLE = pathlib.Path(__file__).resolve().parent / "llama_unicode_flags.txt"

    def __init__(self, exprs: list[str]):
        starts, flags, self.space = [], [], set()
        for line in self.TABLE.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            f = line.split()
            if f[0] == "W":
                self.space.add(int(f[1], 16))
            else:
                starts.append(int(f[0], 16))
                flags.append(int(f[1], 16))
        self.starts, self.flags = starts, flags
        self.collapsed_chars: dict[str, str] = {}
        self.parts = []
        for e in exprs:
            collapse = any(k in e for k in _UCAT)
            if collapse and any(ord(c) >= 128 for c in e):
                raise ValueError("a regex with both unicode categories and non-ASCII characters (llama.cpp refuses it)")
            self.parts.append((collapse, re.compile(self._collapse(e) if collapse else e, re.ASCII)))

    @staticmethod
    def _collapse(e: str) -> str:
        """unicode_regex_split's regex_expr_collapsed: each known \\p{..} as its category byte and ASCII members."""
        out, inside, i = [], False, 0
        while i < len(e):
            c = e[i]
            if c == "[" and (i == 0 or e[i - 1] != "\\"):
                out.append(c)
                inside = True
                i += 1
                continue
            if inside and c == "]" and e[i - 1] != "\\":
                out.append(c)
                inside = False
                i += 1
                continue
            if c == "\\" and i + 3 < len(e) and e[i + 1] == "p" and e[i + 2] == "{":
                j = e.find("}", i + 3)
                if j != -1 and j <= i + 10 and e[i:j + 1] in _UCAT:
                    cat = _UCAT[e[i:j + 1]]
                    body = _UCAT_CPT[cat] + _UCAT_MAP[cat]
                    out.append(body if inside else "[" + body + "]")
                    i = j + 1
                    continue
            out.append(c)
            i += 1
        return "".join(out)

    def _collapsed(self, ch: str) -> str:
        """text_collapsed's byte for one codepoint: ASCII as itself, whitespace 0x0B, else its category's byte
        (0xD0 for any category the regexes do not name)."""
        r = self.collapsed_chars.get(ch)
        if r is None:
            cp = ord(ch)
            if cp < 128:
                r = ch
            elif cp in self.space:
                r = "\x0b"
            else:
                r = _UCAT_CPT.get(self.flags[bisect.bisect_right(self.starts, cp) - 1], "\xd0")
            self.collapsed_chars[ch] = r
        return r

    def split(self, text: str) -> list[str]:
        collapsed = "".join(self._collapsed(c) for c in text)
        # the wregex path: non-ASCII whitespace as 0x0B (std::wregex's \\s does not match it)
        wide = "".join("\x0b" if ord(c) > 0x7F and ord(c) in self.space else c for c in text)
        offsets = [len(text)]
        for collapse, pat in self.parts:
            subject, out, start = (collapsed if collapse else wide), [], 0
            for off in offsets:
                end = start + off
                pos = start
                for m in pat.finditer(subject, start, end):   # the segment ends the subject, as the C++ iterator's range
                    if m.start() > pos:
                        out.append(m.start() - pos)
                    out.append(m.end() - m.start())
                    pos = m.end()
                if pos < end:
                    out.append(end - pos)
                start = end
            offsets = out
        pieces, start = [], 0
        for off in offsets:
            pieces.append(text[start:start + off])
            start += off
        return pieces


class Tokenizer:
    def __init__(self, tokens: list[str], merges: list[str], token_types: list[int] | None = None,
                 pre: str = "qwen35", special_ids: dict[str, int] | None = None):
        self.tokens = tokens
        self.pre = pre
        self.token_types = token_types
        self.special_ids = special_ids or {}
        self.ids = {t: i for i, t in enumerate(tokens)}
        if len(self.ids) != len(tokens):
            raise ValueError("vocabulary has duplicate tokens: %d entries, %d unique"
                             % (len(tokens), len(self.ids)))
        # Merge rules as (left, right) -> rank.  A merge list that is not a total order over its pairs, or one
        # naming a token that is not in the vocabulary, would make BPE silently produce different ids than the
        # model was trained with, so both are checked here rather than discovered as a bad answer later.
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError("merge %d is not a pair: %r" % (i, m))
            if parts[0] not in self.ids or parts[1] not in self.ids:
                raise ValueError("merge %d names a token outside the vocabulary: %r" % (i, m))
            self.ranks[(parts[0], parts[1])] = i
        if pre == "joyai-llm":
            self._split = LlamaRegexSplit(JOYAI_LLM_EXPRS)
        else:
            if regex is None:
                raise ImportError("the qwen35 pre-tokenizer needs the regex module (requirements.txt)")
            self._re = regex.compile(QWEN35_PATTERN)

        # The literals matched directly instead of being run through BPE.  GGUF token types: 3 = CONTROL,
        # 4 = USER_DEFINED.  The two classes behave DIFFERENTLY and llama.cpp's own tokenizer settled which:
        #
        #   * type 4 (USER_DEFINED: `<think>`, `<tool_call>`, `<tool_response>`, ...) is matched ALWAYS,
        #     with or without parse_special.
        #   * type 3 (CONTROL: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, ...) is matched ONLY when
        #     parse_special is set.
        #
        # Measured, not assumed: with parse_special=False the oracle still emitted `<tool_response>` as one
        # token and mine emitted four, and the three disagreements were exactly the type-4 cases while every
        # type-3 case agreed.  Treating both classes alike costs 3 strings in 1229 and silently changes a chat
        # prompt, because `<|im_end|>` decomposed into ordinary pieces is not the token the model expects.
        self.special_tokens: dict[str, int] = {}
        if token_types:
            for i, ty in enumerate(token_types):
                if ty in (3, 4):
                    self.special_tokens[tokens[i]] = i
        always = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 4]
        self.control_tokens = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 3]
        # Longest literal first, or `<|im_end|>` could match a shorter prefix of itself.  `regex.escape` so a
        # token containing regex metacharacters (several do: `<|`, `[`, `(`) is matched literally.
        if pre == "joyai-llm":
            # llama.cpp's cache_special_tokens order (longest text first; equal lengths by id, which the fork's
            # unstable sort leaves open, and no two of DeepSeek-V4.1's overlap)
            self._st_order = sorted(self.special_tokens.items(), key=lambda kv: (-len(kv[0]), kv[1]))
        else:
            self._always_re = self._alt(always)
            self._special_re = self._alt(list(self.special_tokens))

    @staticmethod
    def _alt(literals: list[str]):
        if not literals:
            return None
        return regex.compile("|".join(regex.escape(s) for s in sorted(literals, key=len, reverse=True)))

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_gguf(cls, path) -> "Tokenizer":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(path)).metadata
        need = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
        missing = [k for k in need if k not in md]
        if missing:
            raise ValueError("GGUF is missing %s; it does not carry a tokenizer" % ", ".join(missing))
        model = md.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError("expected a byte-level BPE tokenizer (model 'gpt2'), got %r" % model)
        special = {k: int(md[k]) for k in md
                   if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        return cls(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                   list(md.get("tokenizer.ggml.token_type") or []) or None,
                   md.get("tokenizer.ggml.pre", "qwen35"), special)

    # -------------------------------------------------------------- the algorithm
    def _bpe(self, word: str) -> list[str]:
        """Merge `word` (already byte-mapped) by LOWEST RANK first, repeatedly - not left to right.

        Applying merges in list order rather than rank order is the classic BPE bug: it produces a different
        segmentation and a plausible token count.

        A word longer than HEAP_MIN symbols (a long CJK run, a minified blob: one pre-token piece of thousands of
        bytes) goes to _bpe_heap, which applies the same merges in the same order without rescanning every pair
        after each merge (#268: that rescan is O(n^2) and took seconds on an 8K-character CJK run).
        """
        if len(word) > self.HEAP_MIN:
            return self._bpe_heap(word)
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts

    HEAP_MIN = 64       # words up to this many symbols keep the scan above, unchanged

    def _bpe_heap(self, word: str) -> list[str]:
        """_bpe's result in O(n log n): the symbols as a linked list, every adjacent pair with a rank in a heap
        keyed (rank, position).  Popping the lowest rank and then the lowest position is exactly the order the
        scan picks (its strict `<` keeps the leftmost of equal ranks), and a symbol's position is where it starts
        in the word, which only its left neighbour's merge can change - so a popped pair whose two symbols are
        no longer the ones it was pushed with is stale and skipped."""
        import heapq
        parts = list(word)
        n = len(parts)
        nxt = list(range(1, n)) + [-1]
        prv = list(range(-1, n - 1))
        ranks = self.ranks
        heap = []
        for i in range(n - 1):
            r = ranks.get((parts[i], parts[i + 1]))
            if r is not None:
                heap.append((r, i, parts[i], parts[i + 1]))
        heapq.heapify(heap)
        while heap:
            r, i, left, right = heapq.heappop(heap)
            j = nxt[i]
            if parts[i] != left or j < 0 or parts[j] != right:
                continue                                # stale: one of its symbols has merged since
            parts[i] = left + right
            parts[j] = None                             # j is gone: i takes its place in the list
            k = nxt[j]
            nxt[i] = k
            if k >= 0:
                prv[k] = i
                r2 = ranks.get((parts[i], parts[k]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, i, parts[i], parts[k]))
            p = prv[i]
            if p >= 0:
                r2 = ranks.get((parts[p], parts[i]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, p, parts[p], parts[i]))
        return [s for s in parts if s is not None]

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in self._re.findall(text):
            mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                i = self.ids.get(tok)
                if i is None:
                    raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                out.append(i)
        return out

    def _encode_matching(self, text: str, pat, plain=()) -> list[int]:
        """Encode `text`, emitting any literal `pat` matches as single tokens and BPE-ing the rest.

        The split happens on the RAW text, before the byte mapping, because a special token's string is a
        literal to match rather than bytes to decompose.  Everything between the matches is tokenized
        normally - which is why a near-miss like `<|im_star` still costs ordinary tokens.  A match that starts
        inside one of the `plain` (start, end) spans is left to the text around it (#537).
        """
        if pat is None:
            return self._encode_plain(text)
        out: list[int] = []
        pos = 0
        for m in pat.finditer(text):
            if plain and any(a <= m.start() < b for a, b in plain):
                continue
            if m.start() > pos:
                out.extend(self._encode_plain(text[pos:m.start()]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._encode_plain(text[pos:]))
        return out

    def _encode_plain_llama(self, text: str) -> list[int]:
        """joyai-llm: llama.cpp's llm_tokenizer_bpe_session on one text fragment (its byte fallback included)."""
        out: list[int] = []
        for piece in self._split.split(text):
            mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                i = self.ids.get(tok)
                if i is not None:
                    out.append(i)
                    continue
                for ch in tok:
                    j = self.ids.get(ch)
                    if j is None:
                        raise KeyError("byte token %r is not in the vocabulary" % ch)
                    out.append(j)
        return out

    def _encode_llama(self, text: str, parse_special: bool, plain=()) -> list[int]:
        """joyai-llm: llama.cpp's tokenizer_st_partition, then each text fragment through BPE.  Type 3 (CONTROL)
        literals are matched only with parse_special, type 4 always; a match that starts inside a `plain` span is
        text (#537)."""
        frags: list[tuple[int, int, int | None]] = [(0, len(text), None)]
        for lit, tid in self._st_order:
            if not parse_special and self.token_types[tid] == 3:
                continue
            out = []
            for a, b, t in frags:
                if t is not None:
                    out.append((a, b, t))
                    continue
                pos = a
                while True:
                    m = text.find(lit, pos, b)
                    while m >= 0 and plain and any(x <= m < y for x, y in plain):
                        m = text.find(lit, m + 1, b)
                    if m < 0:
                        break
                    if m > pos:
                        out.append((pos, m, None))
                    out.append((m, m + len(lit), tid))
                    pos = m + len(lit)
                if pos < b:
                    out.append((pos, b, None))
            frags = out
        ids: list[int] = []
        for a, b, t in frags:
            if t is not None:
                ids.append(t)
            else:
                ids.extend(self._encode_plain_llama(text[a:b]))
        return ids

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        """Tokenize `text`.

        `parse_special` controls only the type-3 CONTROL literals such as `<|im_end|>`; the type-4
        USER_DEFINED ones such as `<think>` are matched either way.  See the note in `__init__`.
        `plain`: (start, end) spans of `text` that are ordinary text even where they spell a literal - a
        `</think>` quoted in a message (#537) - and are tokenized with the text around them.
        """
        if self.pre == "joyai-llm":
            return self._encode_llama(text, parse_special, plain)
        return self._encode_matching(text, self._special_re if parse_special else self._always_re, plain)

    def token_bytes(self, i: int) -> bytes:
        """The raw bytes of one token (a multi-byte character can be split across tokens)."""
        cache = self.__dict__.setdefault("_bytes_cache", {})
        b = cache.get(i)
        if b is None:
            if i < 0 or i >= len(self.tokens):
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, len(self.tokens)))
            if self.token_types and self.token_types[i] in (3, 4) and \
                    any(ch not in UNICODE_TO_BYTE for ch in self.tokens[i]):
                # a special token's text is stored as is, not byte-mapped (DeepSeek-V4.1's <｜User｜>): its UTF-8
                b = cache[i] = self.tokens[i].encode("utf-8")
                return b
            raw = bytearray()
            for ch in self.tokens[i]:
                v = UNICODE_TO_BYTE.get(ch)
                if v is None:
                    raise KeyError("token %d contains a character outside the byte alphabet: %r" % (i, ch))
                raw.append(v)
            b = cache[i] = bytes(raw)
        return b

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)


# ------------------------------------------------------------------ the pack's tokenizer/ directory
def extract(gguf_path, out_dir) -> dict:
    """Write the tokenizer into `<out_dir>/tokenizer/` so the engine never opens the weight shards for it."""
    out = pathlib.Path(out_dir) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer.from_gguf(gguf_path)
    cfg = {
        "model": "gpt2",
        "pre": tk.pre,
        "vocab_size": len(tk.tokens),
        "n_merges": len(tk.ranks),
        "special_ids": tk.special_ids,
        "add_bos_token": False,
        # The pattern is SHIPPED, not recomputed by the reader: it is transcribed from llama.cpp for the
        # declared `pre` type, and a C++ port that re-derived it would be free to get `\p{M}` wrong again.
        "pre_pattern": QWEN35_PATTERN,
        "pre_pattern_source": ".ref/llama.cpp src/llama-vocab.cpp L396 (LLAMA_VOCAB_PRE_TYPE_QWEN35)",
    }
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join("%s %s" % k for k, _ in
                                              sorted(tk.ranks.items(), key=lambda kv: kv[1])), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(tk.token_types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    # the model's own chat template: fine-tunes change it (Swift 1.5 differs from Qwen3.8-Flash-Next's)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    tpl = GGUFFile(pathlib.Path(gguf_path)).metadata.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(tpl, encoding="utf-8", newline="\n")
    return cfg


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="round-trip a corpus and report")
    args = ap.parse_args()
    cfg = extract(args.gguf, args.out)
    print("tokenizer/: vocab %d, merges %d, pre %s, specials %s"
          % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"], cfg["special_ids"]))
    if args.check:
        tk = Tokenizer.from_gguf(args.gguf)
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
                  "def f(x):\n\treturn x  # comment\n", "\u4f60\u597d\uff0c\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "e\u0301\u0301 combining", "\x00\x01\x7f control",
                  "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
                  "x" * 5000]
        bad = 0
        for s in corpus:
            ids = tk.encode(s)
            back = tk.decode(ids)
            if back != s:
                print("  *** ROUND TRIP FAILED *** %r -> %d ids -> %r" % (s[:40], len(ids), back[:40]))
                bad += 1
        print("round trip: %d strings, %d failed" % (len(corpus), bad))
        return 0 if bad == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
