#!/usr/bin/env python3
"""tools/dsv41/engine/two_turn_check.py - a two-turn chat through serve/server.py on dsv41-engine (the
real-model session's check that a conversation's second turn resumes the first's state; with --turns, turn 1 the
16K prompt's text, turn 2's first token at the client, the HTTP check on
the same server load).

    python3 tools/dsv41/engine/two_turn_check.py --engine bin/dsv41-engine --model MODEL.gguf --tokenizer-gguf GGUF \\
        --out DIR [--engine-args "..."] [--ctx N] [--load-s S] [--max-tokens N] [--turns FILE] [--http] [--dry-run]

The server (python -m serve.server --engine strata, the deepseek-v4.1 profile) runs the engine with --engine-args,
--ctx, the profile's end of reply and --route-trace DIR/route-trace.txt. Turn 1 posts the first user message of
--turns (a JSON list of two user messages, pod/e07_prompts.py turns writes one: the 16K prompt's text and a short
follow-up; default: the Fibonacci prompt of pod/e07_chats.json and a follow-up), non-streaming, max_tokens
--max-tokens; turn 2 posts the conversation so far (turn 1's reply as the assistant's message) and the second message,
streaming, and times its first content chunk at the client (first_token_s). Checks (exit 1 when one fails): both answer
200; turn 2's usage reports cached prompt tokens (the server's RESUME); the engine's log has exactly one request per
answered post (the server sends no other: no warm-up, no retry) and the route trace one R line for each, tied by the
engine's request number: turn 1's says resume=0 and turn 2's resume=1 (ExpertEngine::begin_request: the request
continues the engine's sequence, so the hot set's planners keep turn 1's counts); the engine's log has turn 2's decode
line. --http (the engine then runs with --log-ids) adds, on the same server load after turn 2, serve_check.py's API
checks without a reference run (the real model's GPU holds one engine): a new conversation posted non-streaming (an
OpenAI chat.completion, its finish_reason and usage) and streaming (SSE chat.completion.chunk objects of one id, the
role first, finish_reason and usage last, data: [DONE]), each reply's content the text of the engine's own ids for it
(serve_check.text_of on its --log-ids line), the two replies' ids compared (reported: the second restores the first's
prompt end and may read other experts from another hot set), and tools refused with 400; its verdict is kept apart
(http.ok) from the turns' (turns_ok). Reported in DIR/two-turn.json and printed: per turn the prompt and cached tokens,
the completion tokens, the client's seconds (turn 2's first token too), the engine's prompt read (its "N read in M ms"
line) and its decode (steps, hit rate, ms per step and tok/s). --dry-run: the engine runs with --dry-run (the dev VM:
the dry engine writes no R lines and no decode line, so those two checks are skipped and said so).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_check as sc  # noqa: E402  (its server helpers: free_port, post, descendants, stop, text_of)

DEFAULT_TURNS = ["Write a Python function that returns the n-th Fibonacci number iteratively, with a one-line docstring.",
                 "Now make it recursive with memoization, in at most five lines."]
READ_RE = re.compile(r"dsv41-engine serve: request (\d+): (\d+) prompt tokens, (\S+) resume of (\d+), (\d+) read in "
                     r"([0-9.]+) ms, (\d+) generated")
DECODE_RE = re.compile(r"request (\d+): decode experts: (\d+) steps, (\d+) hits and (\d+) misses \(hit rate ([0-9.]+)\).*?"
                       r"; ([0-9.]+) ms per step")
R_RE = re.compile(r"^R (\d+) (\S+) reset=([01]) resume=([01])$", re.M)
# --http's new conversation (serve_check.py's first request) and its reply length
HTTP_MESSAGES = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello there, engine."}]
HTTP_MAX_TOKENS = 12


def post_stream(base, body):
    """A streaming post: (status, Content-Type, the chunks, whether data: [DONE] ended it, the seconds to the first
    chunk with content (None without one), the seconds to the end)."""
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, chunks, done = time.time(), None, [], False
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                c = json.loads(line[len("data: "):])
                chunks.append(c)
                if first is None and c["choices"][0]["delta"].get("content"):
                    first = round(time.time() - t0, 3)
            return r.status, r.headers.get("Content-Type", ""), chunks, done, first, round(time.time() - t0, 2)
    except urllib.error.HTTPError as e:
        with e:
            return e.code, e.headers.get("Content-Type", ""), [], False, None, round(time.time() - t0, 2)


def http_part(base, res):
    """--http on the running server; each group's failure is recorded and the next group runs. -> the posts the
    engine answered (non-streaming, streaming), for the engine log's ids afterwards."""
    h = res["http"]
    body = {"model": "deepseek-v4.1-flash", "messages": HTTP_MESSAGES, "max_tokens": HTTP_MAX_TOKENS}
    answered = []

    def part(name, fn):
        try:
            fn()
        except (sc.CheckFailed, KeyError, IndexError, TypeError, ValueError) as e:
            print(f"FAIL http {name}: {e if isinstance(e, sc.CheckFailed) else repr(e)}")
            h["failed"].append(name)

    def non_streaming():
        st, ct, raw = sc.post(base, body)
        sc.check(st == 200 and ct.startswith("application/json"), f"http: non-streaming answers 200 JSON ({st}, {ct})")
        answered.append("non-streaming")
        j = json.loads(raw)
        msg = j["choices"][0]["message"]
        sc.check(j["object"] == "chat.completion" and j["id"].startswith("chatcmpl-") and msg["role"] == "assistant",
                 "http: an OpenAI chat.completion object")
        finish = j["choices"][0]["finish_reason"]
        sc.check(finish in ("stop", "length"), f"http: finish_reason {finish}")
        sc.check(j["usage"]["prompt_tokens"] > 0 and 0 < j["usage"]["completion_tokens"] <= HTTP_MAX_TOKENS,
                 f"http: usage {j['usage']}")
        h["non_streaming"] = {"content": msg["content"] or "", "usage": j["usage"]}

    def streaming():
        st, ct, chunks, done, first, secs = post_stream(base, body)
        sc.check(st == 200 and ct.startswith("text/event-stream"), f"http: streaming answers 200 SSE ({st}, {ct})")
        answered.append("streaming")
        sc.check(done, "http: the stream ends with data: [DONE]")
        sc.check(chunks and all(c["object"] == "chat.completion.chunk" and c["id"] == chunks[0]["id"] for c in chunks),
                 f"http: {len(chunks)} chat.completion.chunk objects of one id")
        sc.check(chunks[0]["choices"][0]["delta"].get("role") == "assistant", "http: the first chunk names the role")
        sc.check(chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length") and "usage" in chunks[-1],
                 "http: the last chunk has finish_reason and usage")
        h["streaming"] = {"content": "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks),
                          "usage": chunks[-1]["usage"], "first_token_s": first, "client_s": secs}

    def tools():
        f = {"type": "function", "function": {"name": "f", "parameters": {}}}
        st, _, raw = sc.post(base, dict(body, tools=[f]))
        sc.check(st == 400 and "not supported" in raw, f"http: tools are refused with 400 ({st})")

    for name, fn in (("non-streaming", non_streaming), ("streaming", streaming), ("tools", tools)):
        part(name, fn)
    return answered


def http_ids(res, logged, answered, tok, eos_id):
    """--http after the server stopped: each answered reply against the engine's ids for it (the --log-ids lines,
    one per request in order: turn 1, turn 2, then the answered posts)."""
    h = res["http"]
    ids = {}
    for i, name in enumerate(answered):
        key = name.replace("-", "_")
        try:
            sc.check(len(logged) > 2 + i, f"http: the engine's log has the {name} request's ids")
            got = ids[name] = logged[2 + i]
            reply = h.get(key)
            sc.check(reply is not None, f"http: the {name} reply was read")
            sc.check(reply["content"] == sc.text_of(tok, eos_id, got),
                     f"http: the {name} content is the text of the engine's ids {got}: {reply['content']!r}")
            sc.check(reply["usage"]["completion_tokens"] == len(got),
                     f"http: the {name} usage counts the engine's {len(got)} ids")
        except sc.CheckFailed as e:
            print(f"FAIL {e}")
            h["failed"].append(f"{name} ids")
    if len(ids) == 2:
        same = h["same_ids"] = ids["non-streaming"] == ids["streaming"]
        print(f"note http: the streamed reply's ids {'equal' if same else 'differ from'} the non-streamed one's "
              f"({ids['streaming']} against {ids['non-streaming']}; reported: the second restores the first's prompt end)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-args", default="")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--load-s", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--turns", help="a JSON list of the two user messages")
    ap.add_argument("--http", action="store_true", help="the API checks on the same server load after the two turns")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    turns = json.loads(Path(a.turns).read_text()) if a.turns else DEFAULT_TURNS
    if len(turns) != 2 or not all(isinstance(t, str) and t for t in turns):
        ap.error("--turns: a JSON list of two user messages")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from serve.model_profile import DeepSeekV41Profile  # noqa: E402  (serve_check put the repository on the path)
    prof = DeepSeekV41Profile(os.path.abspath(a.tokenizer_gguf))
    trace = out / "route-trace.txt"
    eng_args = [os.path.abspath(x) if os.path.exists(x) else x for x in shlex.split(a.engine_args)]
    cfg = {"exe": os.path.abspath(a.engine),
           "args": ["--model", os.path.abspath(a.model), *eng_args, "--ctx", str(a.ctx), "--eos-ids", str(prof.eos_id),
                    "--route-trace", str(trace)] + (["--log-ids"] if a.http else []) +
                   (["--dry-run"] if a.dry_run else []),
           "model_profile": "deepseek-v4.1", "tokenizer_gguf": os.path.abspath(a.tokenizer_gguf),
           "log": str(out / "engine.log"), "repeat_stop_tokens": 0}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    port = sc.free_port()
    log = open(out / "server.log", "w")
    srv = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config",
                            str(out / "config.json"), "--port", str(port)], cwd=sc.ROOT, stdout=log,
                           stderr=subprocess.STDOUT)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    base = f"http://127.0.0.1:{port}"
    res = {"turns": [], "ok": False}
    if a.http:
        res["http"] = {"ok": False, "failed": []}
    failed, answered = [], []
    t_load = time.time()
    try:
        for _ in range(2 * a.load_s):
            if srv.poll() is not None:
                raise sc.CheckFailed(f"the server exited ({srv.returncode}): see {out / 'server.log'}")
            if "ready: http" in (out / "server.log").read_text(errors="replace"):
                break
            time.sleep(0.5)
        else:
            raise sc.CheckFailed(f"the server did not get ready within {a.load_s} s")
        res["load_s"] = round(time.time() - t_load, 1)
        msgs = []
        for i, text in enumerate(turns):
            msgs.append({"role": "user", "content": text})
            body = {"model": "deepseek-v4.1-flash", "messages": msgs, "max_tokens": a.max_tokens}
            if i == 0:
                t0 = time.time()
                st, _, raw = sc.post(base, body)
                secs = round(time.time() - t0, 2)
                if st != 200:
                    raise sc.CheckFailed(f"turn 1 answers {st}: {raw[:300]}")
                j = json.loads(raw)
                reply, u, first = j["choices"][0]["message"]["content"] or "", j["usage"], None
            else:
                # turn 2 streamed: its first token at the client is the conversation's second-turn latency
                st, _, chunks, done, first, secs = post_stream(base, body)
                if st != 200 or not done or not chunks:
                    raise sc.CheckFailed(f"turn 2 answers {st} ({len(chunks)} chunks, [DONE] {done})")
                reply = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
                u = chunks[-1]["usage"]
            msgs.append({"role": "assistant", "content": reply})
            res["turns"].append({"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                                 "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                                 "client_s": secs, "first_token_s": first, "reply": reply})
        if not res["turns"][1]["cached_tokens"]:
            failed.append("turn 2 reports no cached prompt tokens (no RESUME)")
        if a.http:
            answered = http_part(base, res)
    except sc.CheckFailed as e:
        failed.append(str(e))
    finally:
        theirs = sc.descendants(srv.pid)
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
        sc.stop(theirs)
        log.close()
    # the engine's lines and the trace: the server sends no other request than the posts it answered (no warm-up,
    # no retry), so the engine has exactly that many, turn 1's and turn 2's first, numbered in order; any other count
    # fails rather than pick one
    elog = (out / "engine.log").read_text(errors="replace") if (out / "engine.log").exists() else ""
    reads = {int(m.group(1)): m for m in READ_RE.finditer(elog)}
    decodes = {int(m.group(1)): m for m in DECODE_RE.finditer(elog)}
    reqs = sorted(reads)
    want_reqs = len(res["turns"]) + len(answered)
    if len(res["turns"]) == 2 and len(reqs) == want_reqs:
        for t, n in zip(res["turns"], reqs):
            r = reads[n]
            t.update({"engine_request": n, "resume_kind": r.group(3), "reused": int(r.group(4)), "read": int(r.group(5)),
                      "prompt_ms": float(r.group(6))})
            d = decodes.get(n)
            if d:
                ms = float(d.group(6))
                t["decode"] = {"steps": int(d.group(2)), "hit_rate": float(d.group(5)), "ms_per_step": ms,
                               "tok_s": round(1e3 / ms, 2) if ms > 0 else None}
        if "decode" not in res["turns"][1]:
            if a.dry_run:
                print("note two_turn_check.py: the dry engine writes no decode line: the decode report is skipped")
            else:
                failed.append("the engine's log has no decode line of turn 2")
    elif not failed:
        failed.append(f"the engine's log has {len(reqs)} requests' lines {reqs}, not {want_reqs} (the answered posts)")
    # the route trace's R lines carry the engine's request number (ExpertEngine::begin_request counts the requests as
    # the serve loop's "request N" lines do, from 1): one per request, turn 1's resume=0 and turn 2's resume=1
    rl = [m.groups() for m in R_RE.finditer(trace.read_text(errors="replace"))] if trace.exists() else []
    res["trace_requests"] = [{"request": int(r[0]), "policy": r[1], "reset": int(r[2]), "resume": int(r[3])} for r in rl]
    by_req = {}
    for r in res["trace_requests"]:
        by_req.setdefault(r["request"], []).append(r)
    if a.dry_run and not rl:
        print("note two_turn_check.py: the dry engine writes no route trace: the resume=1 check is skipped")
    elif len(reqs) != want_reqs:
        failed.append(f"the route trace's R lines {res['trace_requests']} cannot be tied to the two turns: the engine's "
                      f"log has {len(reqs)} requests")
    elif sorted(by_req) != reqs or any(len(v) != 1 for v in by_req.values()):
        failed.append(f"the route trace's R lines {res['trace_requests']} are not one per request, for engine requests "
                      f"{reqs}")
    else:
        for i, (n, want) in enumerate(zip(reqs, (0, 1))):
            r = by_req[n][0]
            if i < len(res["turns"]):
                res["turns"][i]["trace"] = r
            if r["resume"] != want:
                failed.append(f"turn {i + 1} (engine request {n}): the route trace's R line says resume={r['resume']}, "
                              f"not {want}")
    if a.http:
        logged = [[int(t) for t in x.split(": ")[-1].split()] for x in elog.splitlines() if "generated_token_ids:" in x]
        http_ids(res, logged, answered, prof.tokenizer, prof.eos_id)
        if not answered and not res["http"]["failed"]:
            res["http"]["failed"].append(f"not run: {'; '.join(failed) or 'the turns did not finish'}")
        res["http"]["ok"] = not res["http"]["failed"]
    res["failed"] = failed
    res["turns_ok"] = not failed
    res["ok"] = not failed and (not a.http or res["http"]["ok"])
    (out / "two-turn.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for i, t in enumerate(res["turns"]):
        d = t.get("decode", {})
        print(f"two_turn_check.py: turn {i + 1}: {t['prompt_tokens']} prompt tokens ({t['cached_tokens']} cached; engine: "
              f"{t.get('resume_kind', '?')} resume of {t.get('reused', '?')}, {t.get('read', '?')} read in "
              f"{t.get('prompt_ms', '?')} ms), {t['completion_tokens']} generated, decode {d.get('tok_s')} tok/s "
              f"(hit rate {d.get('hit_rate')}), {t['client_s']} s at the client"
              + (f", first token at {t['first_token_s']} s" if i == 1 else ""))
    if rl:
        print(f"two_turn_check.py: route trace R lines {res['trace_requests']}")
    for f in failed:
        print(f"FAIL {f}")
    if a.http:
        print(f"two_turn_check.py: http {'PASS' if res['http']['ok'] else 'FAIL: ' + ', '.join(res['http']['failed'])}")
    print(f"two_turn_check.py: {'PASS' if res['ok'] else 'FAIL'} ({out})")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
