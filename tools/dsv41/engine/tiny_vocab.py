#!/usr/bin/env python3
"""tools/dsv41/engine/tiny_vocab.py - a vocab-only GGUF of the first N tokens of DeepSeek-V4.1's tokenizer.

The tiny random models (tiny_dsv41.py) have a 512-token `no_vocab` vocabulary, so the server needs a tokenizer whose
ids stay below 512 to talk to them: the real tokenizer ($VOCAB, the vocab-only GGUF of tools/dsv41/README.md) cut
to its first N tokens (the three specials, the 256 byte characters, then the first merges' results), the merges whose
two parts and result are all kept (in their order), the token types cut alike, every other key (the chat template,
the pre-tokenizer, the end-of-sentence id 1) as it is. It is a real BPE tokenizer of N tokens, which the fork's
llama-tokenize reads as well (tokenizer_parity.py checks the two agree on it too).

--model M writes M (a tiny_dsv41.py model) with this tokenizer in place of its no_vocab one instead: M's other keys
and tensors as they are, so the fork's llama-server loads it with the real begin-of-sentence text and chat template
(quality/fork_prompt_test.py). N must be M's vocab_size.

    python3 tools/dsv41/engine/tiny_vocab.py $VOCAB tiny-vocab.gguf [--n 512] [--model tiny-a.gguf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))
import gguf_header  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("vocab")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--model", help="a tiny_dsv41.py model to write with this tokenizer")
    a = ap.parse_args()
    h = gguf_header.read_header(a.vocab)
    md = h.metadata
    t, tokens, et = md["tokenizer.ggml.tokens"]
    if a.n < 259 or a.n > len(tokens):
        raise SystemExit(f"tiny_vocab.py: --n {a.n}: at least the specials and the 256 byte characters (259), at most {len(tokens)}")
    keep = tokens[:a.n]
    kept = set(keep)
    md["tokenizer.ggml.tokens"] = (t, keep, et)
    tt, types, tet = md["tokenizer.ggml.token_type"]
    md["tokenizer.ggml.token_type"] = (tt, types[:a.n], tet)
    mt, merges, met = md["tokenizer.ggml.merges"]
    cut = []
    for m in merges:
        left, right = m.split(" ")
        if left in kept and right in kept and left + right in kept:
            cut.append(m)
    md["tokenizer.ggml.merges"] = (mt, cut, met)
    for key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id"):
        if key in md and gguf_header.integer_value(a.vocab, h, key) >= a.n:
            raise SystemExit(f"tiny_vocab.py: {key} is outside the first {a.n} tokens")
    if a.model:
        m = gguf_header.read_header(a.model)
        arch = m.metadata["general.architecture"][1]
        n_vocab = gguf_header.integer_value(a.model, m, f"{arch}.vocab_size")
        if n_vocab != a.n:
            raise SystemExit(f"tiny_vocab.py: {a.model} has vocab_size {n_vocab}, not --n {a.n}")
        m.metadata = {**{k: v for k, v in m.metadata.items() if not k.startswith("tokenizer.")},
                      **{k: v for k, v in md.items() if k.startswith("tokenizer.")}}
        raw = Path(a.model).read_bytes()
        at = {t["name"]: (m.data_start + t["offset"], gguf_header.tensor_nbytes(t["type"], t["ne"])) for t in m.tensors}
        gguf_header.write_gguf(a.out, m, lambda name: raw[at[name][0]:at[name][0] + at[name][1]])
    else:
        h.tensors = []
        gguf_header.write_gguf(a.out, h, lambda name: b"")
    print(f"tiny_vocab.py: {a.out}: {'the tensors of ' + a.model + ', ' if a.model else ''}{a.n} tokens, "
          f"{len(cut)} of {len(merges)} merges, pre "
          f"{md['tokenizer.ggml.pre'][1]}, chat template {'kept' if 'tokenizer.chat_template' in md else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
