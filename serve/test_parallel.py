"""#465: "parallel": N - several requests decode together in the engine's batch slots (--batch N).

A fake engine (a Python script speaking the engine's GEN / BGEN / BT / BDONE / BADM / BSTOP lines) runs behind the
real StrataEngine and Service, so the server's side is tested without a GPU: requests up to N run at once, more
wait for a slot, /metrics shows the slots, each request's history row is its own, and a conversation's next turn
goes back to the slot that holds it."""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, engine_args, parallel_args, serve

# The fake engine: one token every STEP seconds per active slot (a "window" serves every active slot at once);
# GEN (solo) streams T lines; BGEN reads the prompt (one T line, DONE), answers BADM and continues in the slot.
# Every reply is the bytes of "ok, done." then <|im_end|> (ByteTokenizer: 257). For serve/test_robust.py: a prompt
# with VERYLONG gets a reply of 400 "la " (about 24 s at STEP), --log-stops puts the BSTOPs in --log too, --die-at K
# exits 3 one second after the K-th GEN or BGEN line and --hang-at K stops printing there for good (alive, silent).
# --batch-ctx C says batch_ctx=C on INFO and a BGEN past C positions runs alone (its ALONE line
# first, BADM <slot> 0 rep=0 alone=K why=... at its end, the slots not stepping meanwhile); DONE's reuse is the live
# sequence's prefix the prompt extends (dsv41-engine's live resume: the prompt and every fed token of the last request).
# --rep: BDONE and an ALONE BADM say rep=, as dsv41-engine's do: rep=1 for a
# request in the deterministic mode (no det=0 key on its BGEN), rep=0 for one with det=0.
FAKE_BATCH = r'''import queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
fit = int(args[args.index("--fit") + 1]) if "--fit" in args else slots
STEP = 0.02
CH = 32
lines, stop = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
BCTX = int(args[args.index("--batch-ctx") + 1]) if "--batch-ctx" in args else 0
print("INFO engine=0.1.39" + (f" batch_slots={fit}" if fit >= 2 else "") + (f" batch_ctx={BCTX}" if BCTX else "") +
      (" slot_cache=1" if "--slotcache" in args else ""), flush=True)
print("READY 4096 stop", flush=True)
LONG, VERYLONG = list(b"LONGREPLY"), list(b"VERYLONG")
DIE = int(args[args.index("--die-at") + 1]) if "--die-at" in args else 0
HANG = int(args[args.index("--hang-at") + 1]) if "--hang-at" in args else 0
def has(ids, w):
    return any(ids[i:i + len(w)] == w for i in range(len(ids)))
def reply(ids):            # the rest of the reply after what the prompt already ends with (a request continued)
    ids = [int(x) for x in ids]
    n_la = 400 if has(ids, VERYLONG) else 15 if has(ids, LONG) else 0
    R = list(b"ok, " + b"la " * n_la + b"done." if n_la else b"ok, done.") + [257]
    k = max(k for k in range(len(R)) if k == 0 or ids[-k:] == R[:k])
    return R[k:]
active = {}          # slot -> [tokens left, max_new, produced]
rep_of = {}          # slot -> its BDONE's rep= (--rep)
stopped = set()      # BSTOPped slots: they end "cancel"
def window():        # one batch window: every active slot one token
    for b in sorted(active):
        left, max_new, produced = active[b]
        t = left.pop(0) if left else 257
        produced += 1
        print(f"BT {b} {t}", flush=True)
        if t == 257 or produced >= max_new:
            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'
            stopped.discard(b)
            print(f"BDONE {b} {produced} {fin} 1.0" + (f" rep={rep_of.get(b, 1)}" if "--rep" in args else ""), flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
n_req = 0
live = []            # the live sequence: the last request's prompt and every token of it fed
while True:
    try:
        line = lines.get(timeout=STEP if active else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BYIELD "):
        continue                              # too late: the read it was for has ended
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        if log and "--log-stops" in args:
            log.write(f"BSTOP {b} {int(b in active)}\n"); log.flush()
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        continue
    if line.startswith(("GEN ", "BGEN ")):
        n_req += 1
        if n_req == DIE:                      # silent for 1 s first, so a request can wait behind this one
            time.sleep(1.0)
            sys.exit(3)
        while n_req == HANG:
            time.sleep(3600)
        f = line.split()
        slot = int(f[1]) if f[0] == "BGEN" else None
        max_new = int(f[2] if f[0] == "BGEN" else f[1])
        ids = f[-1].split(",")
        det = 0 if "det=0" in f[3:-1] else 1
        if log:
            log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\n"); log.flush()
        pen = next((k for k in f[2:-1] if k.startswith("penalty_freq=")), None)
        if pen:                               # dsv41-engine's: an ERR in place of the DONE (and of a BADM)
            print(f"ERR {pen} (the API's frequency_penalty): this engine applies no penalties", flush=True)
            continue
        toks = reply(ids)
        stop.clear()
        reuse = len(live) if len(ids) > len(live) and [int(x) for x in ids[:len(live)]] == live else 0
        unfit = slot is not None and BCTX and max_new > 1 and len(ids) + max_new - 1 > BCTX
        if unfit:
            why = (f"{len(ids)} prompt tokens and {max_new} to generate need {len(ids) + max_new - 1} positions, a "
                   f"batch slot holds {BCTX}")
            print(f"ALONE {slot} {len(active)} why={why}", flush=True)
        # the prompt read: CH tokens a chunk, a PP line each; a BYIELD <s> gives way at a chunk boundary
        given = None
        for q in range(CH, len(ids) - 1, CH):
            time.sleep(STEP / 2)
            print(f"PP {q} {len(ids)} 1 1", flush=True)
            held = []
            while True:
                try:
                    l2 = lines.get_nowait()
                except queue.Empty:
                    break
                if l2 and l2.startswith("BYIELD "):
                    ys = int(l2.split()[1])
                    # a request running alone (ALONE said) never gives way, as dsv41-engine's never does
                    if len(ids) - 1 - q > CH and (slot is None or ys == slot) and given is None and not unfit:
                        given = (ys, q)
                else:
                    held.append(l2)
            for l2 in held:
                lines.put(l2)
            if given:
                break
            if active and (q // CH) % 4 == 0:     # the slots decode between the prompt's chunks (as the engine)
                window()
        if given:
            if log:
                log.write(f"YIELD {given[0]} {given[1]}\n"); log.flush()
            print(f"YIELDED {given[0]} {given[1]}", flush=True)
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 0", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        n = 1 if slot is not None and not unfit else max_new
        out = 0
        for t in toks[:n]:
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")
        print(f"DONE {out} {len(ids)} 5.0 {out * STEP * 1000:.1f} {fin} 0 0 {reuse}", flush=True)
        live = [int(x) for x in ids] + toks[:max(0, out - 1)]
        if slot is not None and unfit:
            print(f"BADM {slot} 0 rep={det} alone={len(active)} why={why}", flush=True)
            continue
        if slot is not None:
            cont = fin == "length" and max_new > 1
            if cont:
                active[slot] = [toks[1:], max_new, 1]
                rep_of[slot] = det
            print(f"BADM {slot} {1 if cont else 0}", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
'''


class ParallelArgs(unittest.TestCase):
    def test_parallel_config_key(self):
        self.assertEqual(parallel_args({"parallel": 4}, []), ["--batch", "4"])
        self.assertEqual(parallel_args({"parallel": 1}, []), [])
        self.assertEqual(parallel_args({}, []), [])
        self.assertEqual(parallel_args({"parallel": 2}, ["--batch", "3"]), [])       # the args' own value wins
        self.assertEqual(parallel_args({"parallel": "4"}, []), [])                   # said and ignored
        self.assertEqual(parallel_args({"parallel": True}, []), [])
        self.assertEqual(parallel_args({"parallel": 12}, []), ["--batch", "12"])     # the engine warns and caps
    def test_parallel_above_the_window_says_so_unless_batch_mtp(self):
        import contextlib, io, os
        def said(args, env):
            buf = io.StringIO()
            old = os.environ.get("STRATA_BATCH_MTP")
            if env is None:
                os.environ.pop("STRATA_BATCH_MTP", None)
            else:
                os.environ["STRATA_BATCH_MTP"] = env
            try:
                with contextlib.redirect_stdout(buf):
                    r = parallel_args({"parallel": 12}, args)
            finally:
                if old is None:
                    os.environ.pop("STRATA_BATCH_MTP", None)
                else:
                    os.environ["STRATA_BATCH_MTP"] = old
            self.assertEqual(r, ["--batch", "12"])
            return "at most" in buf.getvalue()
        self.assertTrue(said([], None))                  # 0.1.39: the window holds 8 rows
        self.assertTrue(said([], "0"))
        self.assertFalse(said(["--batch-mtp"], None))    # --batch-mtp waves more through the window
        self.assertFalse(said([], "1"))

        args = engine_args({"args": ["--pack", "p"], "parallel": 2})
        self.assertEqual(args[-2:], ["--batch", "2"])


class PickSlot(unittest.TestCase):
    def engine(self, n):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch = n
        e.slot_order = list(range(n))
        e.slot_busy = [False] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_used = [0.0] * n
        e.slot_live = [None] * n
        return e

    def test_the_slot_that_holds_the_conversation(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5, 6]), 2)       # the longest held start
        self.assertEqual(e.pick_slot([1, 2, 3, 9]), 0)
        self.assertEqual(e.pick_slot([7, 8]), 1)                    # no match: the empty slot first
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([7, 8]), 0)                    # then the one used longest ago
        e.slot_busy = [True] * 3
        self.assertIsNone(e.pick_slot([7, 8]))
        e.slot_busy = [False] * 3
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)                 # a held prompt is never the WHOLE prompt

    def test_slots_view(self):
        e = self.engine(2)
        e.slot_held[1] = [1, 2]
        e.slot_live[0] = {"slot": 0, "state": "decoding", "prompt_tokens": 9, "generated": 4,
                          "started": time.time() - 2, "first_token": time.time() - 1}
        v = e.slots_view()
        self.assertEqual(v[0]["state"], "decoding")
        self.assertEqual(v[0]["generated"], 4)
        self.assertEqual(v[1], {"slot": 1, "state": "idle", "held_tokens": 2})


class FakeEngineService(unittest.TestCase):
    """The real StrataEngine and Service over HTTP, the fake engine behind them (no tests of its own: ParallelService's
    and serve/test_robust.py's)."""

    def start(self, slots, fit=None, slot_cache=False, rep=False, extra_args=()):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        extra = ["--batch", str(slots)] + (["--fit", str(fit)] if fit is not None else []) + ["--log", str(self.log)]
        extra += ["--slotcache"] if slot_cache else []
        extra += ["--rep"] if rep else []
        extra += list(extra_args)
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", extra)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, text, max_tokens=64, **extra):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "reasoning_effort": "none",
                **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())


class ParallelService(FakeEngineService):
    """The real StrataEngine and Service over HTTP, the fake engine behind them."""

    def test_an_engine_error_on_the_solo_path_answers_at_once(self):
        """An ERR the engine gives in place of a request's DONE (dsv41-engine's penalty refusal) is a 400
        at once, with no wait for a DONE that never comes (a det-sampling run once waited 300 s here), and
        the next request runs; the same for a BGEN refused beside a decoding slot (no BADM follows)."""
        self.start(2)
        def status_of(**extra):
            t0 = time.monotonic()
            try:
                self.chat("alone", **extra)
                return 200, time.monotonic() - t0
            except urllib.error.HTTPError as e:
                return e.code, time.monotonic() - t0
        code, took = status_of(frequency_penalty=0.5)
        self.assertEqual(code, 400)
        self.assertLess(took, 5.0)
        self.assertEqual(self.chat("alone")["choices"][0]["message"]["content"], "ok, done.")
        out = {}
        th = threading.Thread(target=lambda: out.update(long=self.chat("hi LONGREPLY")))
        th.start()
        time.sleep(0.15)                      # the long one decodes (solo, then in a slot once this one comes)
        code, took = status_of(frequency_penalty=0.5)
        th.join(30)
        self.assertEqual(code, 400)
        self.assertLess(took, 5.0)
        self.assertEqual(out["long"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        self.assertEqual(self.chat("alone")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_reports_fewer_slots(self):
        self.start(4, fit=2)
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_stop_strings_in_a_batch_slot(self):
        """#454: a stop string cuts the answer in --batch mode too, and the slot is freed for the next request."""
        self.start(2)
        body = {"messages": [{"role": "user", "content": "hi LONGREPLY"}], "max_tokens": 64, "reasoning_effort": "none",
                "stop": ["la la"]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            c = json.loads(r.read().decode())["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok, ", "stop"))
        self.assertEqual(self.chat("again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_turns_batching_off(self):
        self.start(4, fit=0)
        self.assertEqual(self.engine.batch, 0)
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")

    def test_concurrent_requests_share_the_slots(self):
        self.start(2)
        # the first one decodes long enough for the others to arrive while it runs (also on a busy PC)
        texts = ["first question LONGREPLY", "second question, longer", "a third one waits for a slot"]
        reply = {t: "ok, " + "la " * 15 + "done." if "LONGREPLY" in t else "ok, done." for t in texts}
        out, errors, seen = {}, [], []

        def go(t):
            try:
                out[t] = self.chat(t)
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=go, args=(t,)) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        deadline = time.time() + 30
        while any(th.is_alive() for th in threads) and time.time() < deadline:
            m = self.get("/metrics")["live"]
            seen.append(m)
            time.sleep(0.01)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        for t in texts:
            self.assertEqual(out[t]["choices"][0]["message"]["content"], reply[t])
        # the slots were reported, and a request was admitted while another decoded in a slot (the fake logs how many
        # slots were active at each admission)
        self.assertTrue(any(m.get("parallel") == 2 and len(m.get("slots", [])) == 2 for m in seen))
        admitted = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertIn(1, admitted)
        # every request recorded its own row; nothing is left running
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)
            self.assertEqual(self.svc.live_reqs, {})
            self.assertFalse(self.svc.status["busy"])
        self.assertEqual(sorted(r["output_tokens"] for r in self.svc.history),
                         sorted(len(reply[t]) + 1 for t in texts))
        # never more than two in the slots at once (the fake logs how many were active at each admission)
        active = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertTrue(all(a <= 1 for a in active), active)

    def race(self, first, second, delay=0.15, max_tokens=64):
        """`first` sent, `second` after `delay` seconds; -> {text: (reply, seconds it took)}"""
        res = {}

        def go(t):
            t0 = time.time()
            r = self.chat(t, max_tokens=max_tokens)
            res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
        a = threading.Thread(target=go, args=(first,))
        a.start()
        time.sleep(delay)
        go(second)
        a.join(30)
        return res

    def test_a_long_prompt_gives_way_to_a_short_one(self):
        """#656: a long solo prompt read gives way at a chunk boundary; the short request is answered first, then the
        long one goes on (in the slot its part waited in) and is answered too."""
        self.start(2)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        # the long read takes ~1 s (3,000 tokens, 32 a chunk, 10 ms a chunk): the short one did not wait for it
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)
        self.assertFalse(any(self.engine.slot_busy))

    def test_an_admission_gives_way_too(self):
        """The same while another request decodes in a slot: the long one is being admitted, a short one waits."""
        self.start(3)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        self.assertFalse(any(self.engine.slot_busy))

    def test_no_way_given_to_a_long_one(self):
        """Two prompts of about the same length: no BYIELD (it only goes to a much shorter one)."""
        self.start(2)
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_left_alone_it_goes_back_to_the_solo_path(self):
        """A request decoding in a slot whose neighbour finished goes back to the solo path (MTP drafts), continued
        from its slot - only with an engine that keeps the slots' conversations (INFO slot_cache=1)."""
        self.start(2, slot_cache=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.05)
        short = self.chat("short", max_tokens=200)
        long.join(30)
        self.assertEqual(short["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(res["long"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        # solo, then both in slots, then the long one alone again: a GEN after the BGENs
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN", kinds)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)

    def test_a_batched_reply_says_whether_it_is_reproducible(self):
        """A reply that decoded in a batch slot (beside other requests: alone
        it takes the solo path) carries "strata_reproducible": true in the deterministic mode (the engine's default, or
        "strata_deterministic": true) and false in the mode off ("strata_deterministic": false, or an engine started
        with --deterministic off), also when it ends at its stop token before its BDONE (rep=) is read; a request
        answered alone (GEN) carries no such field."""
        self.start(2, rep=True)
        self.engine.det_key = True   # dsv41-engine's (model_profile.py engine_det_key)
        alone = self.chat("alone")
        self.assertNotIn("strata_reproducible", alone)

        def pair(**extra):
            out = {}
            th = threading.Thread(target=lambda: out.setdefault("long", self.chat("LONGREPLY please", max_tokens=200,
                                                                                    **extra)))
            th.start()
            time.sleep(0.05)
            out["short"] = self.chat("short", **extra)
            th.join(30)
            self.assertEqual(out["long"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
            return out
        on = pair()
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertIn("BGEN", kinds)
        self.assertIs(on["short"].get("strata_reproducible"), True)
        self.assertIs(on["long"].get("strata_reproducible"), True)
        off = pair(strata_deterministic=False)
        self.assertIs(off["short"].get("strata_reproducible"), False)
        self.assertIs(off["long"].get("strata_reproducible"), False)
        self.engine.det_default = False   # main()'s value for an engine started with --deterministic off
        self.assertIs(pair(strata_deterministic=False)["short"].get("strata_reproducible"), False)
        self.assertIs(pair(strata_deterministic=True)["long"].get("strata_reproducible"), True)

    def three_decoding(self):
        """Three long requests decoding in batch slots (the first one solo, then STOPped into a slot by the second);
        -> their threads and replies, once the three slots decode."""
        out, threads = {}, []
        for k in range(3):
            t = f"long {k} LONGREPLY"
            threads.append(threading.Thread(target=lambda t=t: out.setdefault(t, self.chat(t, max_tokens=200))))
            threads[-1].start()
            time.sleep(0.05)
        deadline = time.time() + 10
        def states():
            return [(r or {}).get("state") for r in self.engine.slot_live[:3]]
        while time.time() < deadline and states() != ["decoding"] * 3:
            time.sleep(0.005)
        self.assertEqual(sum(self.engine.slot_busy), 3, self.engine.slots_view())
        return threads, out

    def test_a_required_reproducible_reply_waits_and_decodes_alone(self):
        """With 3 slots decoding, a request with "strata_reproducible": "required" does not
        join the batch: it waits for the slots to end, then runs alone (GEN with no slot active, in the deterministic
        mode: det=1) and its reply says "strata_reproducible": true; the batched ones say true too (in the mode)."""
        self.assertIn(" det=1", StrataEngine.sampling_keys({"strata_reproducible": "required"}, det_key=True))
        self.assertIn(" det=1", StrataEngine.sampling_keys({"strata_reproducible": "required",
                                                            "strata_deterministic": True}, det_key=True))
        self.start(4, rep=True)
        self.engine.det_key = True   # dsv41-engine's (model_profile.py engine_det_key)
        threads, out = self.three_decoding()
        before = len(self.log.read_text().splitlines())
        t0 = time.time()
        req = self.chat("required one", strata_reproducible="required")
        took = time.time() - t0
        for th in threads:
            th.join(30)
        after = self.log.read_text().splitlines()[before:]
        self.assertEqual(len(after), 1, after)                          # no BGEN: it never joined the batch
        kind, slot, _, active = after[0].split()
        self.assertEqual((kind, slot, active), ("GEN", "None", "0"))     # alone: no slot active when it was read
        self.assertEqual(req["choices"][0]["message"]["content"], "ok, done.")
        self.assertIs(req.get("strata_reproducible"), True)
        self.assertGreater(took, 0.3)                                    # it waited for the slots' ~55 tokens
        for t, r in out.items():
            self.assertEqual(r["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.", t)
            self.assertIs(r.get("strata_reproducible"), True, t)
        self.assertFalse(any(self.engine.slot_busy))

    def test_a_normal_request_still_batches(self):
        """The same 3 slots decoding: a request without the option is admitted into the fourth slot at once."""
        self.start(4, rep=True)
        self.engine.det_key = True
        threads, out = self.three_decoding()
        before = len(self.log.read_text().splitlines())
        req = self.chat("normal one")
        for th in threads:
            th.join(30)
        after = self.log.read_text().splitlines()[before:]
        self.assertEqual(len(after), 1, after)
        kind, _, _, active = after[0].split()
        self.assertEqual((kind, active), ("BGEN", "3"))                 # into a slot beside the 3 decoding
        self.assertEqual(req["choices"][0]["message"]["content"], "ok, done.")
        self.assertIs(req.get("strata_reproducible"), True)             # in the deterministic mode

    def test_required_reproducible_refused_values(self):
        """A 400 before anything is sent: another value, with "strata_deterministic": false, or an engine without a
        deterministic mode; one request at a time (batching off) the required reply is alone too and says true."""
        import urllib.error

        def status(**extra):
            try:
                self.chat("refused", **extra)
                return 200
            except urllib.error.HTTPError as e:
                with e:
                    return e.code
        self.start(2)
        self.assertEqual(status(strata_reproducible="required"), 400)   # det_key False: the Qwen engine's
        self.engine.det_key = True
        self.assertEqual(status(strata_reproducible=True), 400)
        self.assertEqual(status(strata_reproducible="yes"), 400)
        self.assertEqual(status(strata_reproducible="required", strata_deterministic=False), 400)
        self.assertNotIn("GEN", self.log.read_text() if self.log.exists() else "")
        # /v1/messages and /v1/responses check it as the chat route does, before anything is sent
        for path, body in (("/v1/messages", {"messages": [{"role": "user", "content": "refused"}], "max_tokens": 8}),
                           ("/v1/responses", {"input": "refused", "max_output_tokens": 8})):
            for extra in ({"strata_reproducible": "yes"}, {"strata_reproducible": "required", "strata_deterministic": False}):
                req = urllib.request.Request(self.base + path, data=json.dumps({**body, **extra}).encode(),
                                             headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError, msg=f"{path} {extra}") as ctx:
                    urllib.request.urlopen(req, timeout=30)
                with ctx.exception as e:
                    self.assertEqual(e.code, 400, f"{path} {extra}")
                    self.assertIn("strata_reproducible", e.read().decode())
        self.assertNotIn("GEN", self.log.read_text() if self.log.exists() else "")
        self.tearDown()
        self.start(4, fit=0)
        self.engine.det_key = True
        self.assertEqual(self.engine.batch, 0)
        self.assertIs(self.chat("alone", strata_reproducible="required").get("strata_reproducible"), True)
        self.assertNotIn("strata_reproducible", self.chat("alone again"))

    def test_next_turn_goes_back_to_its_slot(self):
        self.start(2)
        a = "conversation A, first turn"
        r1 = {}
        th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
        th.start()
        time.sleep(0.05)
        first = self.chat(a)
        th.join(30)
        deadline = time.time() + 10                     # a slot left at its stop token is freed at its BDONE
        while not [h for h in self.engine.slot_held if h] and time.time() < deadline:
            time.sleep(0.02)
        held = [h for h in self.engine.slot_held if h]
        self.assertTrue(held, "a finished slot request keeps what its slot holds")
        # the held tokens are the prompt and every token fed: all of the reply but its last token
        reply = list(b"ok, done.") + [257]
        mine = [h for h in held if h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(b"conversation A") >= 0]
        self.assertEqual(len(mine), 1, held)
        slot = self.engine.slot_held.index(mine[0])
        self.assertEqual(self.engine.pick_slot(mine[0] + [10, 11]), slot)
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")


if __name__ == "__main__":
    unittest.main()
