"""serve/model_profile.py - what the server does differently per model, behind one object.

The server was written for Qwen3.8-Flash-Next (the default profile: everything as before).  The dedicated
DeepSeek-V4.1-Flash engine (`dsv41-engine --serve`, the same stdin protocol) needs its own:

  * tokenizer and chat template from the GGUF's metadata (tools/strata_tokenizer.py's `joyai-llm` path, proven
    against the fork's llama-tokenize by tools/dsv41/engine/tokenizer_parity.py), the template rendered with the
    GGUF's begin-of-sentence text as `bos_token` and its `from_json` filter;
  * the end of a reply: the GGUF's eos_token_id (<｜end▁of▁sentence｜>, the template's end of an assistant turn);
  * thinking off unless asked (the template's default; a reasoning effort or enable_thinking turns it on), and the
    prompt then ends with </think>, so the reply is content from its first token;
  * no tool calls yet: the model writes them in its DSML format, which OutputParser does not read, so a request with
    tools is refused (400) instead of answered with the calls as text;
  * a chat request whose last message is the assistant's continues it (lm-eval's HumanEval gen_prefix): `turn_end`
    is the text the template closes an assistant turn with (the GGUF's end-of-sentence token), which
    Service.render_prompt cuts off the prompt rendered without a generation prompt.  The Qwen profile has none
    (None), so its requests keep today's behaviour: a new assistant turn.
  * sampling: dsv41-engine samples on the host over the whole vocabulary, so a request's top_k goes to it as given
    (Strata's engine keeps at most 64 candidates: the Qwen profile caps it there), and seed 0 is a seed (Strata's
    engine reads 0 as none: the Qwen profile sends seeds >= 1).
  * determinism: dsv41-engine's deterministic mode is on by default; a request's "strata_deterministic" (true or
    false) goes to it as the GEN key det=, and the config's "deterministic": false starts it with --deterministic
    off.  The Qwen engine has no such mode: its profile sends no key and refuses the config's.
  * several cached conversations: the config's "checkpoints": N starts dsv41-engine with --checkpoints N (its
    default 4); the Qwen profile refuses the key (its engine's conversation cache has its own flags).
  * batched decode: the config's "parallel": N (or the server's --parallel N) starts dsv41-engine with --batch N as
    it starts Strata's engine, and "batch_ctx": N with --batch-ctx N (the positions of a batch slot);
    StrataEngine.generate_batched drives both.  A reply that went through the batch slots carries
    "strata_reproducible": true only when it is in the deterministic mode and its bits did not depend on its
    neighbours (the engine's rep=1: started with --cpu-numerics gpu, or nothing decoded beside it); under the default
    --cpu-numerics fork a reply decoded beside others says false.  A request with "strata_reproducible": "required"
    waits for the slots and decodes alone, its reply "strata_reproducible": true (the Qwen profile, without the
    engine's deterministic mode, refuses it with a 400).  The Qwen profile refuses "batch_ctx".
  * long prompts and robustness: the config's "interleave_chunk": N starts dsv41-engine with --interleave-chunk N (a
    prompt read while slots decode, in chunks of N with a batch step between two; the engine's default, its --chunk);
    the Qwen profile refuses it.  With "parallel" and no "batch_ctx", this profile's slots hold 69632 positions (a 64K
    prompt and its reply fit a slot, at most the config's --ctx): 8 such slots cost ~1.7 GiB of device KV, about 142
    experts fewer in the slab of an RTX PRO 6000 (e07_gate.py), and a request longer than a slot runs alone and holds
    the batch while it decodes (only when nothing else decodes, below).  The server's queue holds at most
    "queue_max" waiting requests (this profile's default 8; past it a 503 with Retry-After), a request ends at
    "request_deadline_s" (this profile's default 1800 s, with its text and finish "length", below), and with
    "parallel" the engine watchdog (no engine line for "engine_silence_s" while a request is in it: the engine is
    ended and every pending request fails) runs.  The Qwen profile's defaults keep today's behaviour: no queue bound,
    no deadline, no watchdog in batch mode (the config may still set "queue_max" and "request_deadline_s").
  * batched decode speed: the config's "hot_spare": N starts dsv41-engine with --hot-spare N, the free slab slots per
    layer for the hot set's swaps (absent: the engine's 2, unchanged); the hot set must leave their room
    (e07_gate.py --hot-spare N).  The Qwen profile refuses it.
  * reply length: a request without max_tokens answers at most this profile's default_max_tokens, 8192 (finish
    "length"; the Qwen profile: the context's room, as before); with batch slots an explicit max_tokens past a slot's
    room is cut to it when the prompt fits a slot, and a prompt past a slot is refused (400) while other requests
    decode; a request reaching request_deadline_s ends with its text and finish "length" (an error only when it has
    generated nothing).

A profile's server options that are Qwen's alone (vision, effort_position "end", reasoning loop recovery, MCP tools)
are refused at start (`refused_options`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


class ModelProfile:
    """Qwen3.8-Flash-Next, the server's own model: its tokenizer pack, its template file, <|im_end|> ends a reply."""

    name = "qwen3.8"
    model_name = "qwen3.8-flash-next"
    thinking_default = True      # the template thinks unless enable_thinking is false
    tools = True
    turn_end = None              # no continuation of a final assistant message: it is answered with a new turn
    engine_top_k_max = 64        # the GEN line's top_k cap (Strata's sampled path keeps at most 64 candidates)
    engine_seed_min = 1          # the smallest seed sent (Strata's engine reads seed 0 as none)
    engine_det_key = False       # the engine has no deterministic mode to switch per request
    engine_checkpoints = False   # the config's "checkpoints" is dsv41-engine's (Qwen: --conversation-cache-mib)
    engine_batch_ctx = False     # the config's "batch_ctx" is dsv41-engine's --batch-ctx
    engine_interleave = False    # the config's "interleave_chunk" is dsv41-engine's --interleave-chunk
    engine_hot_spare = False     # the config's "hot_spare" is dsv41-engine's --hot-spare
    default_batch_ctx = None     # a batch slot's positions with no "batch_ctx" (None: the engine's own)
    queue_max = None             # waiting requests at most before a 503 (None: no bound, today's)
    request_deadline_s = None    # a request's deadline (None: none, today's)
    batch_watchdog = False       # the engine watchdog in batch mode (engine_silence_s)
    default_max_tokens = None    # a reply without max_tokens (None: the context's room)

    def stop_ids(self, tok) -> set[int]:
        return set(tok.encode("<|im_end|>", parse_special=True) + tok.encode("<|endoftext|>", parse_special=True))

    def template_kwargs(self, kwargs: dict) -> dict:
        return kwargs

    def refused_options(self, cfg: dict) -> list[str]:
        return []


class DeepSeekV41Profile(ModelProfile):
    """DeepSeek-V4.1-Flash on dsv41-engine: tokenizer, template and end of reply from a GGUF (the model's own, or a
    vocab-only one such as tools/dsv41/engine/tiny_vocab.py's for the tiny models)."""

    name = "deepseek-v4.1"
    model_name = "deepseek-v4.1-flash"
    thinking_default = False
    tools = False
    engine_top_k_max = None      # dsv41-engine's host sampler takes any top_k (0: off)
    engine_seed_min = 0          # and seed 0 as a seed
    engine_det_key = True        # a request's strata_deterministic as det=0|1
    engine_checkpoints = True    # the config's "checkpoints": N as --checkpoints N
    engine_batch_ctx = True      # the config's "batch_ctx": N as --batch-ctx N
    engine_interleave = True     # the config's "interleave_chunk": N as --interleave-chunk N
    engine_hot_spare = True      # the config's "hot_spare": N as --hot-spare N (absent: the engine's 2)
    # a 64K prompt (65536) and a 4K reply fit a slot; 217.6 MiB of device KV a slot (kv_budget.py)
    default_batch_ctx = 69632
    queue_max = 8                # 8 waiting by default
    # 30 min, past the longest reply this engine gives in it (a 32K-token reply at the RTX PRO 6000's
    # ~18 ms a batch step is about 10 min, at 30 tok/s alone about 18 min)
    request_deadline_s = 1800.0
    batch_watchdog = True        # the engine watchdog with "parallel"
    # a request without max_tokens answers at most 8192
    # tokens, finish "length" there: the slot's room (about 53K for a 16K prompt in 69632) took longer than the 1800 s
    # deadline at the 18-30 tok/s a user of a batch gets, and a runaway reply then lost its text
    default_max_tokens = 8192

    def __init__(self, gguf: str):
        import strata_tokenizer as ST
        from gguf_reader import GGUFFile
        md = GGUFFile(Path(gguf)).metadata
        if md.get("tokenizer.ggml.pre") != "joyai-llm":
            raise ValueError(f"{gguf}: pre-tokenizer {md.get('tokenizer.ggml.pre')!r}, DeepSeek-V4.1's is 'joyai-llm'")
        if not md.get("tokenizer.chat_template"):
            raise ValueError(f"{gguf}: no tokenizer.chat_template")
        self.tokenizer = ST.Tokenizer.from_gguf(gguf)
        self.template_source = md["tokenizer.chat_template"]
        tokens = md["tokenizer.ggml.tokens"]
        self.bos_token = tokens[int(md["tokenizer.ggml.bos_token_id"])]
        self.eos_id = int(md["tokenizer.ggml.eos_token_id"])
        self.turn_end = tokens[self.eos_id]     # <｜end▁of▁sentence｜>: the template's end of an assistant turn

    def stop_ids(self, tok) -> set[int]:
        return {self.eos_id}

    def template_kwargs(self, kwargs: dict) -> dict:
        out = dict(kwargs, bos_token=self.bos_token)
        # Strata's effort levels (frontend.EFFORT) ask for thinking; the template thinks only with enable_thinking
        if out.get("reasoning_effort") and "enable_thinking" not in out:
            out["enable_thinking"] = True
        return out

    def refused_options(self, cfg: dict) -> list[str]:
        refused = []
        if cfg.get("vision"):
            refused.append("vision")
        if cfg.get("effort_position", "start") != "start":
            refused.append("effort_position")
        if cfg.get("reasoning_loop_recovery"):
            refused.append("reasoning_loop_recovery")
        if cfg.get("mcp_servers"):
            refused.append("mcp_servers")
        return refused


PROFILES = ("qwen3.8", "deepseek-v4.1")
