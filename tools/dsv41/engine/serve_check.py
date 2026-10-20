#!/usr/bin/env python3
"""tools/dsv41/engine/serve_check.py - dsv41-engine --serve behind serve/server.py, end to end.

Three parts, failing loudly (exit 1): profile and protocol on their first difference, http after reporting each of its
checks (non-streaming, streaming, the resumed turn, tools), so one pod run shows all of them:

  profile   (with --vocab: the real tokenizer GGUF) the deepseek-v4.1 profile of serve/model_profile.py renders
            DeepSeek-V4.1's own chat template: <｜begin▁of▁sentence｜>, the system prompt, <｜User｜>, <｜Assistant｜>
            and </think> (thinking off by default) or <think> (a reasoning effort), <｜end▁of▁sentence｜> as the only
            stop id, and a request with tools refused.
  protocol  serve/server.py's StrataEngine (the class the server drives the engine with) against
            `dsv41-engine --serve`: READY, the prefix cache's three paths (a live resume, the prompt-end restore, a cold
            read) through RESUME and DONE, the window ring's limit on the in-place restore (at most --chunk positions
            past the checkpoint; one further, the pinned host copy restores it), PP progress, the end of a reply on
            --eos-ids, STOP during a long reply, and ERR for a request that cannot run (a penalty, an id outside the
            vocabulary, the context). Several conversations (multi_checks): 4 alternating over the default 4
            checkpoints, each turn 2+ resuming its own; a 5th evicting the least recently used; one slot resuming
            none of them. Sampled requests, the GEN
            line as the deepseek-v4.1 profile sends it: temperature 0.7, top_k 40, top_p 0.95 and seed 1234 twice
            give the same tokens, seed 1235 others (dry: and not the greedy ones), each request's stderr line names
            its sampler and seed; two requests without a seed report different seeds from std::random_device and one
            of those seeds draws its request's tokens again; seed 0 is a seed; temperature 0 is greedy.
  http      the server itself (python -m serve.server --engine strata, the profile, the tokenizer GGUF) on a free port:
            /v1/chat/completions non-streaming and streaming (OpenAI JSON and SSE), then a second turn that resumes
            (usage.prompt_tokens_details.cached_tokens > 0), tools refused with 400, then sampling: a sampled
            request with seed 0 twice gives the same content (dry: different with seed 1), the engine's log has the
            seed 0 and the top_k the request gave, and frequency_penalty 0.5 is refused with 400 naming it.

Every path it passes on (the engine, the model, the tokenizer GGUF, --out, engine arguments that name files) is made
absolute first: the server runs from the repository's root, the engine from its own directory (a
pod's first HTTP check once died on a config path relative to another cwd). Every process it starts ends with it, on
success, on a failed check and on SIGTERM (the pod's step timeout): the engines and the server it started directly,
and whatever the server started (its engine), which would otherwise outlive it and slow the pod's next step.

--reference says what the generated ids must equal. `dry`: the engine runs with --dry-run (the dev VM; every reply
is the last token read plus one, mod the vocabulary, so every T line is known in advance, and every comparison is
exact). `cli`: a real run (the dev pod, tiny-b): the ids of `dsv41-engine --tokens-file <the prompt's ids> --n-predict
N` with the same engine arguments, --ctx and --chunk, the engine's greedy ids for that prompt. A request the server
reads cold computes what that run computes (the same chunks, then the same decode steps), so its ids (the engine's
--log-ids line) and its content (the server's Detokenizer and OutputParser on those ids) must be equal; a resumed
request reads its positions in other calls, a different sum order on a tiny random model, so its ids are reported,
not gated (forced parity, the prefix-cache smoke's, is the gate), while its RESUME and
DONE fields are checked in both modes. Every engine it starts (the protocol's, the server's through its config, each
reference run) gets --prefill-split off last: the measured split decides each process's
CPU/GPU share from its own timings, so two processes could sum a chunk differently and flip a near-tie of the tiny
random model (once seen: the server's ...250 383 56 411 against the reference's ...250 4 295 249).

    python3 tools/dsv41/engine/serve_check.py --engine bin/dsv41-engine --model tiny-b.gguf --tokenizer-gguf \\
        tiny-vocab.gguf --reference dry --out /tmp/serve-check [--vocab $VOCAB] [--engine-args "..."]

The real model (tools/dsv41/pod/e07-run.sh): `--parts profile,http` leaves out the protocol part (its checks
are the tiny model's: a 512-token vocabulary, --ctx 256), `--ctx` is the served engine's and each reference run's
(default 2048), `--load-s` how long the server may take to get ready (default 300; the real model loads 246 GiB), and
`--chats FILE` (a JSON list of user messages) posts each one after the http checks, non-streaming with
`--chat-max-tokens` (default 256), its reply saved in OUT/chats.json and printed: the quality spot check, reported,
not gated (a request that is not answered with 200 fails). The first request's reference runs before the server
starts, and `--one-engine` runs none while it is up (the real model's GPU and pinned RAM hold one engine): the
resumed turn's ids are then reported and its content checked against them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate, OutputParser, openai_to_messages  # noqa: E402
from serve.model_profile import DeepSeekV41Profile  # noqa: E402
from serve.server import Detokenizer, Service, StrataEngine  # noqa: E402

N_VOCAB_TINY = 512
REFERENCE_S = 1800   # one reference run: the real model's load and its few greedy steps


class CheckFailed(Exception):
    pass


# ------------------------------------------------------------------------------------------------ processes
def descendants(pid: int) -> list:
    """The live processes below pid (Linux /proc; zombies left out), children before grandchildren."""
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()   # after "(comm)": state, ppid, ...
        except (OSError, IndexError):
            continue
        if fields[0] != "Z":
            kids.setdefault(int(fields[1]), []).append(int(d))
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(0), []):
            out.append(c)
            todo.append(c)
    return out


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def stop(pids, grace_s=20.0) -> int:
    """SIGTERM to each pid, SIGKILL to those still running after grace_s; zombies of ours reaped. -> how many ran."""
    pids = [p for p in dict.fromkeys(pids) if alive(p)]
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    until = time.monotonic() + grace_s
    while any(alive(p) for p in pids) and time.monotonic() < until:
        time.sleep(0.2)
    for p in pids:
        if alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    return len(pids)


def check(cond, what):
    if not cond:
        raise CheckFailed(what)
    print(f"ok   {what}")


def same(got, want, what, strict):
    """Token ids that must be equal (strict), or are reported (a free run of another chunking, see the header)."""
    if strict or got == want:
        check(got == want, f"{what}: {got}" + ("" if got == want else f", expected {want}"))
    else:
        print(f"note {what}: {got}, the one-chunk reference {want} (another chunking of a tiny random model: reported)")


# ------------------------------------------------------------------------------------------------ profile
def profile_checks(vocab: str):
    p = DeepSeekV41Profile(vocab)
    t = ChatTemplate(source=p.template_source)
    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}]
    off = t.render(msgs, add_generation_prompt=True, **p.template_kwargs({}))
    check(off == "<｜begin▁of▁sentence｜>You are terse.<｜User｜>Hi<｜Assistant｜></think>",
          f"profile: the template without thinking: {off!r}")
    on = t.render(msgs, add_generation_prompt=True, **p.template_kwargs({"reasoning_effort": "xhigh"}))
    check(on.endswith("<｜Assistant｜><think>"), "profile: a reasoning effort turns thinking on (<think>)")
    two = t.render(msgs + [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "More"}],
                   add_generation_prompt=True, **p.template_kwargs({}))
    check("<｜Assistant｜></think>Hello.<｜end▁of▁sentence｜><｜User｜>More" in two,
          "profile: an earlier assistant turn ends with <｜end▁of▁sentence｜>")
    ids = p.tokenizer.encode(off, parse_special=True)
    check(ids[0] == 0 and 128803 in ids and ids[-2:] == [128804, 128822],
          f"profile: the rendered prompt's specials are single tokens (BOS 0, <｜User｜> 128803, "
          f"<｜Assistant｜> 128804, </think> 128822): {ids}")
    check(p.stop_ids(p.tokenizer) == {1}, "profile: <｜end▁of▁sentence｜> (id 1) is the stop id")
    svc = Service(SimpleNamespace(max_context=4096), p.tokenizer, t, profile=p)
    try:
        svc.prepare(msgs, [{"name": "f", "parameters": {}}], {})
        check(False, "profile: a request with tools is refused")
    except ValueError as e:
        check("not supported" in str(e), f"profile: a request with tools is refused ({e})")
    _, thinking, _ = svc.prepare(msgs, None, {})
    check(thinking is False, "profile: thinking off unless asked (the parser reads content from the first token)")


# ------------------------------------------------------------------------------------------------ the engine
class Reference:
    """What the engine's greedy ids for a prompt must be."""

    def __init__(self, mode, exe, engine_args, out: Path):
        self.mode, self.exe, self.args, self.out, self.n = mode, exe, engine_args, out, 0

    def ids(self, prompt, n, eos=(), ctx_args=("--ctx", "2048")):
        """ctx_args: the served engine's --ctx (and --chunk), so a cold read's chunks are the reference's."""
        if self.mode == "dry":
            out, last = [], prompt[-1]
            while len(out) < n:
                last = (last + 1) % N_VOCAB_TINY
                out.append(last)
                if last in eos:
                    break
            return out
        self.n += 1
        f = self.out / f"reference-{self.n}.txt"
        f.write_text(" ".join(map(str, prompt)))
        try:   # a hung reference run is killed, not waited for until the pod's step timeout
            r = subprocess.run([self.exe, *self.args, *ctx_args, "--tokens-file", str(f), "--n-predict", str(n)],
                               capture_output=True, text=True, timeout=REFERENCE_S)
        except subprocess.TimeoutExpired:
            raise CheckFailed(f"the reference run of {f} took more than {REFERENCE_S} s") from None
        line = [x for x in r.stdout.splitlines() if x.startswith("generated_token_ids:")]
        if r.returncode != 0 or len(line) != 1:
            raise CheckFailed(f"the reference run failed (exit {r.returncode}): {r.stderr[-800:]}")
        out = [int(x) for x in line[0].split(":", 1)[1].split()]
        for i, t in enumerate(out):   # the CLI feeds every token; a reply ends at its first end-of-reply id
            if t in eos:
                return out[:i + 1]
        return out


def run(engine: StrataEngine, ids, max_new, sampling=None):
    """-> (tokens, the DONE fields, the RESUME count)."""
    toks = [t for t in engine.generate(ids, max_new, sampling or {}, threading.Event()) if t is not None]
    return toks, dict(engine.last), engine.reused


# The end of an engine request's stderr line (dsv41-engine.cpp sampling_text)
SAMPLING_RE = re.compile(r"^dsv41-engine serve: request (\d+): \d+ prompt tokens, .*; (greedy|sampled: .*)$", re.M)
SEED_RE = re.compile(r"seed (\d+) \((given|from std::random_device)\)$")


def last_sampling(log: Path) -> str:
    """The sampling text of the last request in an engine's log ("" when there is none)."""
    found = SAMPLING_RE.findall(log.read_text(errors="replace"))
    return found[-1][1] if found else ""


# A request's checkpoints line (dsv41-engine.cpp checkpoint_text)
CKPT_RE = re.compile(r"^dsv41-engine serve: request (\d+): checkpoints: (.*)$", re.M)
# and the cost per conversation at load
COST_RE = re.compile(r"^conversation checkpoints: (\d+) slots of ([0-9.]+) MiB pinned host RAM each", re.M)


def last_checkpoints(log: Path) -> str:
    """The checkpoints line of the last request in an engine's log ("" when there is none)."""
    found = CKPT_RE.findall(log.read_text(errors="replace"))
    return found[-1][1] if found else ""


def multi_checks(a, ref: Reference, out: Path):
    """4 conversations alternate over the default 4
    checkpoints, each turn 2+ resumes its own last prompt (RESUME of it, the checkpoints line naming its slot) although
    another conversation ran between; a 5th conversation evicts the least recently used one, whose next turn is read
    cold; with --checkpoints 1 (a single prompt-end snapshot) the same alternation resumes nothing. The cost line at
    load names the 4 slots."""
    dry = ref.mode == "dry"
    eos, small = 500, ("--ctx", "256", "--chunk", "8")
    for slots in (4, 1):
        log = out / f"protocol-multi-{slots}-engine.log"
        extra = [] if slots == 4 else ["--checkpoints", "1"]
        e = StrataEngine(a.engine, ["--model", a.model, *a.engine_args, *small, "--eos-ids", str(eos), *extra], log=str(log))
        e.top_k_max, e.seed_min = None, 0
        try:
            cost = COST_RE.search(log.read_text(errors="replace"))
            check(e.info.get("checkpoints") == slots and cost and int(cost.group(1)) == slots,
                  f"protocol ({slots} checkpoints): INFO checkpoints={e.info.get('checkpoints')}, the cost line "
                  f"'{cost.group(0) if cost else 'missing'}'")
            users = 5 if slots == 4 else 2
            last = [None] * users   # each conversation's last prompt and reply
            resumed = []

            def turn(u):
                if last[u] is None:
                    ids = [100 + 20 * u + i for i in range(10 + u)]   # distinct from the first token on
                else:
                    ids = last[u][0] + last[u][1] + [(31 + 7 * u + len(last[u][0])) % 400]
                toks, d, r = run(e, ids, 3)
                if toks and toks[-1] == eos:
                    raise CheckFailed(f"protocol: conversation {u}'s reply ended at --eos-ids: pick other ids")
                prev = last[u]
                last[u] = (ids, toks)
                return ids, prev, r, last_checkpoints(log)

            for t in range(3):
                for u in range(min(users, 4)):
                    ids, prev, r, line = turn(u)
                    if prev is None or slots == 1:
                        check(r == 0, f"protocol ({slots} checkpoints): conversation {u} turn {t + 1} read cold: RESUME {r}")
                    else:
                        resumed.append(r == len(prev[0]) and line.startswith(f"restored {u} ("))
                        check(resumed[-1], f"protocol ({slots} checkpoints): conversation {u} turn {t + 1} resumes its "
                              f"own checkpoint: RESUME {r} (its last prompt {len(prev[0])}), '{line}'")
            if slots == 1:
                continue
            # a 5th conversation: the least recently used checkpoint (conversation 0's) goes
            _, _, r, line = turn(4)
            check(r == 0 and "evicted 0 (" in line and line.endswith("4 of 4 slots held, 1 evictions"),
                  f"protocol: a 5th conversation is read cold and evicts conversation 0's checkpoint: RESUME {r}, '{line}'")
            _, prev, r, line = turn(0)
            check(r == 0 and "evicted 1 (" in line,
                  f"protocol: conversation 0's next turn, its checkpoint evicted, is read cold: RESUME {r}, '{line}'")
            _, prev, r, line = turn(2)
            check(r == len(prev[0]), f"protocol: conversation 2, still held, resumes: RESUME {r}, '{line}'")
        finally:
            e.close()


def protocol_checks(a, ref: Reference, out: Path):
    dry = ref.mode == "dry"
    eos, small = 500, ("--ctx", "256", "--chunk", "8")
    args = ["--model", a.model, *a.engine_args, *small, "--eos-ids", str(eos)]
    e = StrataEngine(a.engine, args, log=str(out / "protocol-engine.log"))
    e.top_k_max, e.seed_min = None, 0   # the deepseek-v4.1 profile's GEN keys (serve/model_profile.py; main() sets them)
    try:
        check(e.max_context == 256 and e.can_stop, f"protocol: READY 256 stop (context {e.max_context})")
        check(e.info.get("chunk") == 8 and e.info.get("eos") == eos, f"protocol: INFO {e.info}")
        p1 = list(range(10, 30))
        t1, d1, r1 = run(e, p1, 5)
        same(t1, ref.ids(p1, 5, (eos,), small), "protocol: a cold request's tokens (3 chunks of 8)", True)
        check(d1["finish"] == ("stop" if t1[-1] == eos else "length") and d1["generated"] == len(t1) and
              d1["reused"] == 0 and r1 == 0, f"protocol: DONE of a cold request {d1}")
        check(e.progress == (20, 20), f"protocol: PP after the prompt's last chunk {e.progress}")
        p2 = p1 + t1 + [100, 101]
        t2, d2, r2 = run(e, p2, 4)
        n2 = len(p1) + len(t1) - 1
        check(r2 == n2 and d2["reused"] == n2, f"protocol: live resume: RESUME {n2} (the prompt and the fed tokens), DONE {d2}")
        same(t2, ref.ids(p2, 4, (eos,), small), "protocol: the resumed request's tokens", dry)
        p3 = p2 + [(t2[0] + 7) % N_VOCAB_TINY]
        t3, d3, r3 = run(e, p3, 3)
        check(r3 == len(p2) and d3["reused"] == len(p2), f"protocol: prompt-end restore: RESUME {len(p2)}, DONE {d3}")
        same(t3, ref.ids(p3, 3, (eos,), small), "protocol: the restored request's tokens", dry)
        p4 = [10, 11, 99]
        t4, d4, r4 = run(e, p4, 2)
        check(r4 == 0 and d4["reused"] == 0, f"protocol: a diverging prompt is read cold {d4}")
        same(t4, ref.ids(p4, 2, (eos,), small), "protocol: the cold request's tokens (one chunk)", True)
        # the window ring: 8 fed tokens past the checkpoint restore in place (--chunk 8), 9 from its pinned host copy
        # (with a single snapshot it could not be restored past the ring and the request was read cold)
        for fed, how in ((8, "in place on the device"), (9, "host to device")):
            q = list(range(40, 48))
            tq, _, _ = run(e, q, fed + 1)
            _, _, r = run(e, q + [(tq[0] + 1) % N_VOCAB_TINY], 1)
            if len(tq) == fed + 1:
                line = last_checkpoints(out / "protocol-engine.log")
                check(r == 8 and line.startswith("restored ") and (dry or how in line),
                      f"protocol: {fed} tokens fed after the prompt: RESUME {r} (expected 8, the checkpoint restored"
                      f"{'' if dry else ' ' + how}): '{line}'")
            else:
                print(f"note protocol: the window check's reply ended at --eos-ids after {len(tq)} tokens: skipped")
        if dry:
            t6, d6, _ = run(e, [495], 10)
            check(t6 == [496, 497, 498, 499, 500] and d6["finish"] == "stop", f"protocol: --eos-ids 500 ends the reply {d6}")
        sampled_checks(e, ref, out / "protocol-engine.log", small, eos)
        for what, ids, n, s in (("frequency_penalty 0.5", [1, 2], 2, {"temperature": 0.7, "frequency_penalty": 0.5}),
                                ("an id outside the vocabulary", [1, 600], 2, None),
                                ("250 prompt tokens + 10 at ctx 256", list(range(250)), 10, None)):
            try:
                run(e, ids, n, s)
                check(False, f"protocol: ERR for {what}")
            except ValueError as err:
                check(True, f"protocol: ERR for {what}: {err}")
        t7, d7, _ = run(e, [1, 2, 3], 2)
        check(d7["generated"] == len(t7) and d7["reused"] == 0, "protocol: a request after the ERRs runs")
        same(t7, ref.ids([1, 2, 3], 2, (eos,), small), "protocol: its tokens (one chunk)", True)
    finally:
        e.close()
    multi_checks(a, ref, out)
    if not dry:   # the STOP check needs a reply that cannot end before it
        return
    # STOP during a long reply: the consumer leaves after 3 tokens, StrataEngine sends STOP and reads to DONE
    e = StrataEngine(a.engine, ["--model", a.model, *a.engine_args, "--ctx", "2000000", "--chunk", "8"],
                     log=str(out / "protocol-stop-engine.log"))
    try:
        g = e.generate([5, 6], 1999000, {}, threading.Event())
        got = []
        while len(got) < 3:
            t = next(g)
            if t is not None:   # None: a PP line's heartbeat
                got.append(t)
        g.close()
        d = dict(e.last)
        check(got == [7, 8, 9] and d["finish"] == "cancel" and 3 <= d["generated"] < 1999000,
              f"protocol: STOP ends a reply early: {got} read, {d['generated']} of 1999000 generated ({d['finish']})")
        k = d["generated"]
        live = [5, 6] + [(6 + i) % N_VOCAB_TINY for i in range(1, k + 1)]
        _, d8, r8 = run(e, live + [1], 1)
        check(r8 == 2 + k - 1, f"protocol: after a STOP the live state is the prompt and the fed tokens: RESUME {r8}")
    finally:
        e.close()


def sampled_checks(e: StrataEngine, ref: Reference, log: Path, small, eos):
    """Sampled requests through the protocol (the header's list). On a real model (cli) a sampled request's
    tokens are as deterministic as its logits: reported, not gated."""
    dry = ref.mode == "dry"
    p, n = [7, 8, 9], 12
    s = {"temperature": 0.7, "top_k": 40, "top_p": 0.95, "seed": 1234}
    ta, da, _ = run(e, p, n, s)
    check(da["generated"] == len(ta) and last_sampling(log) ==
          "sampled: temperature 0.7, top_k 40, top_p 0.95, min_p 0, seed 1234 (given)",
          f"protocol: a sampled request (temperature 0.7 top_k 40 top_p 0.95 seed 1234): {ta}, its log line "
          f"'{last_sampling(log)}'")
    tb, _, _ = run(e, p, n, s)
    same(tb, ta, "protocol: the same sampled request again: the same tokens", dry)
    tc, _, _ = run(e, p, n, dict(s, seed=1235))
    check(tc != ta, f"protocol: seed 1235: other tokens {tc}")
    if dry:
        greedy = ref.ids(p, n, (eos,), small)
        check(ta != greedy, f"protocol: the sampled tokens are not the greedy ones {greedy}")
    run(e, p, n, dict(s, seed=0))
    check(last_sampling(log).endswith("seed 0 (given)"), f"protocol: seed 0 is a seed: '{last_sampling(log)}'")
    free = []
    for _ in range(2):
        tf, _, _ = run(e, p, n, {"temperature": 0.7, "top_k": 40, "top_p": 0.95})
        m = SEED_RE.search(last_sampling(log))
        check(m is not None and m.group(2) == "from std::random_device",
              f"protocol: a sampled request without a seed reports the one it drew with: '{last_sampling(log)}'")
        free.append((tf, int(m.group(1))))
    check(free[0][1] != free[1][1], f"protocol: two requests without a seed: seeds {free[0][1]} and {free[1][1]}")
    tr, _, _ = run(e, p, n, dict(s, seed=free[0][1]))
    same(tr, free[0][0], f"protocol: seed {free[0][1]} (the first one's, reported) draws its tokens again", dry)
    tg, _, _ = run(e, p, n, {"temperature": 0})
    check(last_sampling(log) == "greedy", f"protocol: temperature 0 is greedy: '{last_sampling(log)}'")
    same(tg, ref.ids(p, n, (eos,), small), "protocol: the greedy request's tokens", True)


# ------------------------------------------------------------------------------------------------ the server
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(base, body, stream=False):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read().decode("utf-8")
            return r.status, r.headers.get("Content-Type", ""), raw
    except urllib.error.HTTPError as e:
        with e:
            return e.code, e.headers.get("Content-Type", ""), e.read().decode("utf-8")


def text_of(tok, eos_id, ids):
    """The content the server makes of these ids: its Detokenizer and OutputParser (thinking off), the end of reply
    left out."""
    d, p, out = Detokenizer(tok), OutputParser(thinking=False), []
    for t in ids:
        if t == eos_id:
            break
        out += p.feed(d.push(t))
    return "".join(ev.text or "" for ev in out + p.finish() if ev.kind == "content")


def http_checks(a, ref: Reference, out: Path):
    prof = DeepSeekV41Profile(a.tokenizer_gguf)
    tok, tpl = prof.tokenizer, ChatTemplate(source=prof.template_source)
    # every path absolute (main): the server reads the config from ROOT, the engine starts in its own directory
    cfg = {"exe": a.engine, "args": ["--model", a.model, *a.engine_args, "--ctx", str(a.ctx), "--eos-ids",
                                     str(prof.eos_id), "--log-ids"],
           "model_profile": "deepseek-v4.1", "tokenizer_gguf": a.tokenizer_gguf, "log": str(out / "http-engine.log"),
           "repeat_stop_tokens": 0}
    (out / "http-config.json").write_text(json.dumps(cfg, indent=1))
    svc = Service(SimpleNamespace(max_context=a.ctx), tok, tpl, profile=prof)
    n, dry = a.max_tokens, ref.mode == "dry"

    def expected(body):
        msgs, tools, kw = openai_to_messages(body)
        ids, _, _ = svc.prepare(msgs, tools, kw, n)
        return ids, ref.ids(ids, n, {prof.eos_id}, ("--ctx", str(a.ctx)))

    # the first request's reference before the server starts (the real model's GPU and pinned RAM hold one
    # engine at a time)
    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello there, engine."}]
    body = {"model": "deepseek-v4.1-flash", "messages": msgs, "max_tokens": n}
    prompt, want = expected(body)
    port = free_port()
    log = open(out / "http-server.log", "w")
    srv = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config",
                            str(out / "http-config.json"), "--port", str(port)], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(2 * a.load_s):
            if srv.poll() is not None:
                raise CheckFailed(f"the server exited ({srv.returncode}): see {out / 'http-server.log'}")
            if "ready: http" in (out / "http-server.log").read_text(errors="replace"):
                break
            time.sleep(0.5)
        else:
            raise CheckFailed(f"the server did not get ready within {a.load_s} s")

        failed = []

        def part(name, fn):
            """One group of checks: a failure (or a malformed answer) is reported and the next group runs."""
            try:
                fn()
            except (CheckFailed, KeyError, IndexError, TypeError, ValueError) as e:
                print(f"FAIL http {name}: {e if isinstance(e, CheckFailed) else repr(e)}")
                failed.append(name)

        first = {}   # the non-streaming reply, which the resumed turn's history carries

        def non_streaming():
            st, ct, raw = post(base, body)
            check(st == 200 and ct.startswith("application/json"), f"http: non-streaming answers 200 JSON ({st}, {ct})")
            j = json.loads(raw)
            msg = first["msg"] = j["choices"][0]["message"]
            check(j["object"] == "chat.completion" and j["id"].startswith("chatcmpl-") and msg["role"] == "assistant",
                  "http: an OpenAI chat.completion object")
            check(j["usage"]["prompt_tokens"] == len(prompt) and j["usage"]["completion_tokens"] == len(want),
                  f"http: usage {j['usage']} for {len(prompt)} prompt tokens and {len(want)} generated")
            check((msg["content"] or "") == text_of(tok, prof.eos_id, want),
                  f"http: the content is the text of the engine's greedy ids {want}: {msg['content']!r}")
            check(j["choices"][0]["finish_reason"] == ("stop" if want[-1] == prof.eos_id else "length"),
                  "http: finish_reason")

        def streaming():
            st, ct, raw = post(base, dict(body, stream=True))
            check(st == 200 and ct.startswith("text/event-stream"), f"http: streaming answers 200 SSE ({st}, {ct})")
            events = [x[len("data: "):] for x in raw.split("\n") if x.startswith("data: ")]
            check(events[-1] == "[DONE]", "http: the stream ends with data: [DONE]")
            chunks = [json.loads(x) for x in events[:-1]]
            check(all(c["object"] == "chat.completion.chunk" and c["id"] == chunks[0]["id"] for c in chunks),
                  f"http: {len(chunks)} chat.completion.chunk objects of one id")
            check(chunks[0]["choices"][0]["delta"].get("role") == "assistant", "http: the first chunk names the role")
            text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
            check(text == text_of(tok, prof.eos_id, want),
                  f"http: the streamed content equals the non-streamed one: {text!r}")
            check(chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length") and "usage" in chunks[-1],
                  "http: the last chunk has finish_reason and usage")

        second = {}

        def resumed_turn():
            check("msg" in first, "http: the second turn has the first turn's reply to carry")
            body2 = dict(body, messages=msgs + [{"role": "assistant", "content": first["msg"]["content"] or ""},
                                                {"role": "user", "content": "And once more."}])
            if a.one_engine:   # no reference run beside the server's engine: the ids are reported (second_ids)
                m2, t2, kw2 = openai_to_messages(body2)
                prompt2 = svc.prepare(m2, t2, kw2, n)[0]
                second["want"] = None
            else:
                prompt2, second["want"] = expected(body2)
            st, _, raw = post(base, body2)
            check(st == 200, f"http: the second turn answers ({st})")
            j2 = second["reply"] = json.loads(raw)
            cached = second["cached"] = j2["usage"]["prompt_tokens_details"]["cached_tokens"]
            if dry or cached > 0:
                check(0 < cached < len(prompt2),
                      f"http: the second turn resumes: {cached} of {len(prompt2)} prompt tokens cached")
            else:
                print("note http: the second turn was read cold: the earlier reply's text tokenizes differently at the "
                      "template's edge (the tiny vocabulary has no special tokens)")

        def tools():
            f = {"type": "function", "function": {"name": "f", "parameters": {}}}
            st, _, raw = post(base, dict(body, tools=[f]))
            check(st == 400 and "not supported" in raw, f"http: tools are refused with 400 ({st})")

        for name, fn in (("non-streaming", non_streaming), ("streaming", streaming), ("resumed turn", resumed_turn),
                         ("tools", tools)):
            part(name, fn)
        # the engine's --log-ids lines, one per request that reached it, in order (the refused one does not)
        logged = [[int(t) for t in x.split(": ")[-1].split()] for x in
                  (out / "http-engine.log").read_text(errors="replace").splitlines() if "generated_token_ids:" in x]
        print(f"note http: the engine's log has the generated ids of {len(logged)} requests (3 expected)")

        def ids_of(i, what):
            check(len(logged) > i, f"http: the engine's log has the {what}'s ids")
            return logged[i]

        part("non-streaming", lambda: same(ids_of(0, "non-streaming request"), want,
                                           "http: the non-streaming request's ids (the engine's log)", True))
        part("streaming", lambda: same(ids_of(1, "streaming request"), want, "http: the streaming request's ids", True))

        def second_ids():
            check("reply" in second, "http: the second turn answered, so its ids can be compared")
            got, want2 = ids_of(2, "second turn"), second["want"]
            if want2 is None:
                print(f"note http: the second turn's ids {got} (--one-engine: no reference run beside the server)")
                want2 = got
            same(got, want2, "http: the second turn's ids", dry or second["cached"] == 0)
            if got == want2:
                check((second["reply"]["choices"][0]["message"]["content"] or "") == text_of(tok, prof.eos_id, want2),
                      "http: the second turn's content is the text of its ids")

        part("resumed turn", second_ids)

        def sampling():
            """The deepseek-v4.1 profile's sampled requests (seed 0, top_k 40 as given) and a penalty's 400."""
            bs = dict(body, temperature=0.7, top_p=0.95, top_k=40, seed=0)
            replies = []
            for b in (bs, bs, dict(bs, seed=1)):
                st, _, raw = post(base, b)
                check(st == 200, f"http: a sampled request (seed {b['seed']}) answers ({st})")
                replies.append(json.loads(raw)["choices"][0]["message"]["content"] or "")
            same(replies[1], replies[0], "http: seed 0 twice: the same content", dry)
            if dry:
                check(replies[2] != replies[0], f"http: seed 1: other content {replies[2]!r}")
            elog = (out / "http-engine.log").read_text(errors="replace")
            check(elog.count("sampled: temperature 0.7, top_k 40, top_p 0.95, min_p 0, seed 0 (given)") == 2,
                  "http: the engine's log has both seed 0 requests with top_k 40 (the profile sends both as given)")
            st, _, raw = post(base, dict(bs, frequency_penalty=0.5))
            check(st == 400 and "frequency_penalty" in raw, f"http: frequency_penalty 0.5 is refused with 400 naming "
                  f"it ({st}: {raw[:200]})")

        part("sampling", sampling)

        def chats():
            replies = []
            for i, text in enumerate(json.loads(Path(a.chats).read_text())):
                body_c = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": text}],
                          "max_tokens": a.chat_max_tokens}
                t0 = time.time()
                st, _, raw = post(base, body_c)
                check(st == 200, f"http: chat {i + 1} answers ({st})")
                j = json.loads(raw)
                replies.append({"prompt": text, "reply": j["choices"][0]["message"]["content"],
                                "finish_reason": j["choices"][0]["finish_reason"], "usage": j["usage"],
                                "seconds": round(time.time() - t0, 2)})
                print(f"note http: chat {i + 1}: {text!r} -> {replies[-1]['reply']!r} ({j['usage']}, "
                      f"{replies[-1]['seconds']} s)")
            (out / "chats.json").write_text(json.dumps(replies, indent=1, ensure_ascii=False))

        if a.chats:
            part("chats", chats)
        if failed:
            raise CheckFailed(f"http: {len(dict.fromkeys(failed))} of {6 if a.chats else 5} check groups failed: "
                              f"{', '.join(dict.fromkeys(failed))}")
    finally:
        # the server's engine is its child: taken now, while the server still runs, it is stopped too should the
        # server leave it behind
        theirs = descendants(srv.pid)
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
        left = stop(theirs)
        if left:
            print(f"note http: {left} processes of the server outlived it, stopped")
        log.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--engine", required=True, help="dsv41-engine")
    ap.add_argument("--model", required=True,
                    help="a deepseek41 GGUF: of a 512-token vocabulary (tiny-a, tiny-b) for the protocol part")
    ap.add_argument("--tokenizer-gguf", required=True, help="the model's tokenizer (tiny_vocab.py's 512-token cut)")
    ap.add_argument("--reference", choices=["dry", "cli"], required=True)
    ap.add_argument("--engine-args", default="", help="more engine arguments (placement, readers)")
    ap.add_argument("--vocab", help="the real tokenizer GGUF: the profile checks")
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="profile,protocol,http", help="comma separated: profile, protocol, http")
    ap.add_argument("--ctx", type=int, default=2048, help="the served engine's --ctx and each reference run's")
    ap.add_argument("--load-s", type=int, default=300, help="seconds the server may take to get ready")
    ap.add_argument("--chats", help="a JSON list of user messages, each posted after the http checks (reported)")
    ap.add_argument("--chat-max-tokens", type=int, default=256)
    ap.add_argument("--one-engine", action="store_true",
                    help="no reference run while the server's engine is loaded: the resumed turn's ids are reported")
    a = ap.parse_args()
    parts = a.parts.split(",")
    if not parts or any(x not in ("profile", "protocol", "http") for x in parts):
        ap.error(f"--parts {a.parts}: profile, protocol and http only")
    a.engine, a.model, a.tokenizer_gguf = (os.path.abspath(x) for x in (a.engine, a.model, a.tokenizer_gguf))
    # --prefill-split off last, so it wins over one in --engine-args: the exact-id checks need one split everywhere
    a.engine_args = [os.path.abspath(x) if os.path.exists(x) else x for x in shlex.split(a.engine_args)] + \
        ["--prefill-split", "off"] + (["--dry-run"] if a.reference == "dry" else [])
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = Reference(a.reference, a.engine, ["--model", a.model, *a.engine_args], out)
    # the pod's step timeout sends SIGTERM: leave through the finally blocks, which stop what was started
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if a.vocab and "profile" in parts:
            profile_checks(a.vocab)
        if "protocol" in parts:
            protocol_checks(a, ref, out)
        if "http" in parts:
            http_checks(a, ref, out)
    except CheckFailed as e:
        print(f"FAIL {e}")
        print(f"serve_check.py: FAIL ({a.reference}; logs in {out})")
        return 1
    finally:
        left = stop(descendants(os.getpid()))
        if left:
            print(f"note serve_check.py: {left} processes it started were still running, stopped")
    done = [x if x != "profile" or a.vocab else "profile skipped (no --vocab)" for x in parts]
    print(f"serve_check.py: PASS ({a.reference}: {', '.join(done)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
