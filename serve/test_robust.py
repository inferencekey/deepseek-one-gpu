"""The server stays robust with batch slots.

serve/test_parallel.py's fake engine (its batch protocol, a VERYLONG prompt's 24 s reply, --die-at and --hang-at) runs
behind the real StrataEngine and Service over HTTP: a client that leaves mid-stream, or a non-streamed one that hangs up,
frees its slot for the next waiting request; past queue_max waiting requests the next one gets a 503 with Retry-After
and a JSON error naming the queue; a dead or a hung engine turns every running and waiting request into a clear error
within the watchdog's time, nothing hangs; non-streamed replies decode in the slots beside the others; a request past
its deadline ends with its text and finish "length". And the config keys: interleave_chunk, queue_max,
request_deadline_s.

A request moved from the solo path into a slot counts as cached only the prefix cached before its own read; with
batch slots a max_tokens past a slot's room is cut to it,
a prompt past a slot is refused (400) beside decoding requests and runs alone when nothing else decodes, its ALONE line
seen at its start (the server's log, /metrics) and its reply's strata_ran_alone on the chat, Messages and Responses
routes; a request without max_tokens takes the profile's default_max_tokens; the 503s' Retry-After (at
least 5 s, the alone request's hold on top; an engine that is down: 120 s and a code). After a burst of
cancelled runners and waiting requests the next request is answered within a few seconds. The
reply's strata_max_tokens says a max_tokens cut to a slot's room, and the limit given to a request that asked for none
("asked": null), on the three routes."""
import contextlib
import http.client
import io
import json
import socket
import threading
import time
import unittest

from serve.test_parallel import FakeEngineService


class Client:
    """One request on a socket of its own, so the test can hang up on it: its status, its SSE lines as they come (each
    with its arrival time), or its whole body."""

    def __init__(self, base: str, text: str, stream: bool, max_tokens: int | None = 2000,
                 path: str = "/v1/chat/completions"):
        host, port = base.rsplit("/", 1)[1].split(":")
        self.conn = http.client.HTTPConnection(host, int(port), timeout=120)
        body = {"messages": [{"role": "user", "content": text}], "reasoning_effort": "none", "stream": stream}
        if path == "/v1/responses":
            body = {"input": text, "stream": stream}
        if max_tokens is not None:
            body["max_output_tokens" if path == "/v1/responses" else "max_tokens"] = max_tokens
        self.path = path
        self.stream, self.status, self.headers, self.body = stream, None, {}, None
        self.events, self.error, self.done = [], None, threading.Event()
        self.thread = threading.Thread(target=self._go, args=(json.dumps(body),), daemon=True)
        self.thread.start()

    def _go(self, body):
        try:
            self.conn.request("POST", self.path, body=body, headers={"Content-Type": "application/json"})
            self.sock = self.conn.sock      # the response keeps it after the connection lets go of it (Connection: close)
            r = self.conn.getresponse()
            self.status, self.headers = r.status, dict(r.getheaders())
            if not self.stream or r.status != 200:
                self.body = json.loads(r.read().decode())
                return
            while True:
                line = r.fp.readline()
                if not line:
                    return
                line = line.decode().strip()
                if line.startswith("data: "):
                    self.events.append((time.monotonic(), line[6:]))
        except (OSError, ValueError, http.client.HTTPException) as e:
            self.error = e
        finally:
            self.done.set()

    def content_chunks(self) -> int:
        return sum(1 for _, x in self.events if x != "[DONE]" and json.loads(x).get("choices", [{}])[0]
                   .get("delta", {}).get("content"))

    def text(self) -> str:
        if not self.stream:
            return self.body["choices"][0]["message"]["content"]
        return "".join(json.loads(x)["choices"][0]["delta"].get("content") or "" for _, x in self.events
                       if x != "[DONE]" and "choices" in json.loads(x))

    def stream_error(self):
        errs = [json.loads(x)["error"] for _, x in self.events if x != "[DONE]" and "error" in json.loads(x)]
        return errs[0] if errs else None

    def hang_up(self):
        sock = getattr(self, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        self.conn.close()


def wait_until(cond, timeout=20.0, step=0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


class Robust(FakeEngineService):
    def start(self, slots, **kw):
        """FakeEngineService's, the fake logging its BSTOPs too (check_freed reads them)."""
        kw["extra_args"] = ["--log-stops", *kw.get("extra_args", ())]
        super().start(slots, **kw)

    def tearDown(self):
        for c in getattr(self, "clients", []):
            c.hang_up()
        super().tearDown()

    def client(self, text, stream=True, max_tokens=2000, path="/v1/chat/completions"):
        c = Client(self.base, text, stream, max_tokens, path)
        self.clients = getattr(self, "clients", []) + [c]
        return c

    def decoding(self):
        return sum(1 for x in self.engine.slots_view() if x["state"] == "decoding")

    def log_lines(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def fill_slots(self, n, stream=True):
        """n VERYLONG requests decoding in the n slots (the first one alone first, then moved into a slot)."""
        runners = [self.client(f"VERYLONG runner {k}", stream=stream) for k in range(n)]
        self.assertTrue(wait_until(lambda: self.decoding() == n), self.engine.slots_view())
        return runners

    def check_freed(self, runners, gone):
        """`gone` hung up while every slot decoded and a short request waited: the short one is answered, while the
        other runners still decode, and the server freed the slot with a BSTOP."""
        waiting = self.client("short one", max_tokens=64)
        self.assertTrue(wait_until(lambda: self.svc.queue_view()["waiting"] == 1), self.svc.queue_view())
        time.sleep(0.3)
        self.assertEqual(waiting.content_chunks(), 0, "it waited for a slot")
        t0 = time.monotonic()
        gone.hang_up()
        self.assertTrue(waiting.done.wait(10), "the waiting request was never answered")
        took = time.monotonic() - t0
        self.assertEqual(waiting.status, 200)
        self.assertEqual(waiting.text(), "ok, done.")
        self.assertLess(took, 5.0)
        others = [r for r in runners if r is not gone]
        self.assertTrue(all(not r.done.is_set() for r in others), "the other runners still decode")
        stops = [x for x in self.log_lines() if x.startswith("BSTOP ")]
        self.assertTrue(any(x.endswith(" 1") for x in stops), stops)   # a BSTOP for a slot that was decoding

    def test_a_streaming_client_that_leaves_frees_its_slot(self):
        self.start(2)
        runners = self.fill_slots(2)
        self.assertTrue(wait_until(lambda: runners[0].content_chunks() > 5))
        self.check_freed(runners, runners[0])

    def test_a_non_streaming_client_that_leaves_frees_its_slot(self):
        """The socket check between tokens (the handler's watch, every 0.5 s) sees a non-streamed client hang up."""
        self.start(2)
        runners = [self.client("VERYLONG quiet one", stream=False), self.client("VERYLONG runner", stream=True)]
        self.assertTrue(wait_until(lambda: self.decoding() == 2), self.engine.slots_view())
        self.check_freed(runners, runners[0])

    def test_the_ninth_waiting_request_gets_503(self):
        self.start(2)
        self.svc.queue_max = 8                          # the deepseek-v4.1 profile's default
        runners = self.fill_slots(2)
        waiting = [self.client(f"waiting {k}", max_tokens=64) for k in range(8)]
        self.assertTrue(wait_until(lambda: self.svc.queue_view()["waiting"] == 8), self.svc.queue_view())
        ninth = self.client("the ninth", max_tokens=64)
        self.assertTrue(ninth.done.wait(5), "a full queue answers at once")
        self.assertEqual(ninth.status, 503)
        retry = ninth.headers.get("Retry-After")
        self.assertTrue(retry and retry.isdigit() and int(retry) >= 5, ninth.headers)   # RETRY_AFTER_MIN_S
        err = ninth.body["error"]
        self.assertEqual(err["code"], "queue_full")
        self.assertIn("queue", err["message"])
        self.assertEqual(err["queue"], {"waiting": 8, "running": 2, "queue_max": 8, "retry_after_s": int(retry)})
        self.assertTrue(all(not w.done.is_set() for w in waiting), "the 8 waiting ones still wait")
        for c in runners + waiting:                    # every client leaves: the server drains, nothing is left
            c.hang_up()
        self.assertTrue(wait_until(lambda: self.svc.queue_view()["admitted"] == 0 and not any(self.engine.slot_busy)),
                        (self.svc.queue_view(), self.engine.slot_busy))
        self.assertEqual(self.chat("after")["choices"][0]["message"]["content"], "ok, done.")

    def test_after_a_burst_of_cancelled_waiters_the_next_request_is_answered(self):
        """Robust_check.py's overflow on the RTX PRO 6000: 8 slots decoding, 8 requests waiting, streamed
        and not mixed, then every client hangs up at once: the next request is answered within a few seconds (there
        the engine left the cancelled slots' BDONE lines unflushed and it waited for a slot until the watchdog)."""
        self.start(8)
        self.svc.queue_max = 8
        runners = [self.client(f"VERYLONG runner {k}", stream=k % 2 == 0) for k in range(8)]
        self.assertTrue(wait_until(lambda: self.decoding() == 8), self.engine.slots_view())
        waiting = [self.client(f"waiting {k}", stream=k % 2 == 1, max_tokens=64) for k in range(8)]
        self.assertTrue(wait_until(lambda: self.svc.queue_view()["waiting"] == 8), self.svc.queue_view())
        for c in runners + waiting:
            c.hang_up()
        self.assertTrue(wait_until(lambda: self.svc.queue_view()["admitted"] == 0 and not any(self.engine.slot_busy)),
                        (self.svc.queue_view(), self.engine.slot_busy))
        t0 = time.monotonic()
        last = self.client("after", stream=False, max_tokens=64)
        self.assertTrue(last.done.wait(5), "the request after the burst was never answered")
        self.assertEqual(last.status, 200)
        self.assertEqual(last.text(), "ok, done.")
        self.assertLess(time.monotonic() - t0, 5.0)

    def broken(self, flag):
        """Two runners decoding in two of three slots, a third request (non-streamed) whose BGEN breaks the engine (its
        4th GEN or BGEN line: the first runner's solo GEN, its BGEN when the second came, the second's BGEN), and a
        fourth waiting behind it for the control lines: -> the clients and the seconds until every one had its
        answer."""
        self.start(3, extra_args=[flag, "4"])
        self.engine.silence_s = 2.0                     # the watchdog's time (engine_silence_s)
        self.engine.batch_watchdog = True
        runners = self.fill_slots(2)
        t0 = time.monotonic()
        third = self.client("the third", stream=False, max_tokens=64)
        self.assertTrue(wait_until(lambda: self.engine.in_engine == 3 or not self.engine.alive()), self.engine.in_engine)
        fourth = self.client("the fourth", stream=False, max_tokens=64)
        clients = runners + [third, fourth]
        self.assertTrue(all(c.done.wait(15) for c in clients), "a request hung")
        return clients, time.monotonic() - t0

    def check_errors(self, clients, words):
        runners, rest = clients[:2], clients[2:]
        for r in runners:                               # mid-stream: the stream's error event, then its end
            self.assertEqual(r.status, 200)
            self.assertIsNotNone(r.stream_error(), r.events[-3:])
            self.assertIn(words, r.stream_error()["message"])
            self.assertEqual(r.events[-1][1], "[DONE]")
        for c in rest:                                  # not started: a 503 with the reason, a code and Retry-After
            self.assertEqual(c.status, 503, c.body)
            self.assertIn(words, c.body["error"]["message"])
            self.assertEqual(c.body["error"]["code"], "engine_unavailable")
            self.assertEqual(c.headers.get("Retry-After"), "120")

    def test_a_dead_engine_fails_every_pending_request(self):
        clients, took = self.broken("--die-at")
        self.check_errors(clients, "the engine stopped")
        self.assertLess(took, 5.0)

    def test_a_hung_engine_fails_every_pending_request_within_the_watchdog(self):
        clients, took = self.broken("--hang-at")
        self.check_errors(clients, "watchdog")
        self.assertLess(took, 2.0 + 4.0)                # the watchdog's 2 s, its 0.25 s beat, the requests' ends
        self.assertFalse(self.engine.alive())

    def test_non_streaming_replies_are_batched(self):
        """Non-streamed requests take the batch slots as streamed ones do: admitted while another one
        decodes, every reply whole."""
        self.start(3)
        c = [self.client(f"LONGREPLY {k}", stream=False) for k in range(3)]
        self.assertTrue(all(x.done.wait(30) for x in c))
        self.assertEqual([x.status for x in c], [200] * 3)
        self.assertEqual([x.text() for x in c], ["ok, " + "la " * 15 + "done."] * 3)
        admitted = [int(x.split()[3]) for x in self.log_lines() if x.startswith("BGEN")]
        self.assertEqual(sorted(admitted)[-2:], [1, 2], admitted)   # beside one, then two decoding slots

    def test_a_request_past_its_deadline_ends_with_its_text(self):
        """The deadline (1 s here) reached by replies of 400 "la " at the fake's 20 ms a token (24 s): each ends
        with the text it had and finish "length", streamed and not, no error and nothing dropped."""
        self.start(2)
        self.svc.request_deadline_s = 1.0
        streamed, quiet = self.client("VERYLONG s"), self.client("VERYLONG q", stream=False)
        self.assertTrue(streamed.done.wait(10) and quiet.done.wait(10))
        self.assertIsNone(streamed.stream_error(), streamed.events[-3:])
        finishes = [json.loads(x)["choices"][0]["finish_reason"] for _, x in streamed.events
                    if x != "[DONE]" and json.loads(x).get("choices")]
        self.assertEqual(finishes[-1], "length")
        self.assertEqual(streamed.events[-1][1], "[DONE]")
        self.assertTrue(streamed.text().startswith("ok, la la"), streamed.text()[:40])
        self.assertEqual(quiet.status, 200, quiet.body)
        self.assertEqual(quiet.body["choices"][0]["finish_reason"], "length")
        self.assertTrue(quiet.text().startswith("ok, la la"), quiet.text()[:40])
        self.assertGreater(quiet.body["usage"]["completion_tokens"], 5)
        self.assertTrue(wait_until(lambda: not any(self.engine.slot_busy)))

    def test_a_moved_request_counts_only_its_cached_prefix(self):
        """A request read cold on the solo path, then moved into a slot when another came (its BGEN resumes its own
        prompt and tokens, the fake's DONE says that reuse), reports 0 cached prompt tokens, never past its prompt."""
        self.start(2)
        first = self.client("LONGREPLY first")
        self.assertTrue(wait_until(lambda: first.content_chunks() > 3), first.events[-2:])
        second = self.client("the second", max_tokens=64)
        self.assertTrue(first.done.wait(20) and second.done.wait(20))
        usage = json.loads(next(x for _, x in reversed(first.events) if x != "[DONE]"))["usage"]
        self.assertTrue(any(x.startswith("BGEN ") for x in self.log_lines()), self.log_lines())   # it was moved
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 0, usage)
        self.assertLessEqual(usage["prompt_tokens_details"]["cached_tokens"], usage["prompt_tokens"])

    def test_max_tokens_past_a_slot_is_cut_to_its_room(self):
        """Slots of 300 positions; a max_tokens of 2000 with a prompt that fits a slot answers up to the slot's
        room (300 - prompt + 1 tokens), finish "length", in a slot."""
        self.start(2, extra_args=["--batch-ctx", "300"])
        c = self.client("VERYLONG cut", stream=False, max_tokens=2000)
        self.assertTrue(c.done.wait(30))
        self.assertEqual(c.status, 200, c.body)
        u = c.body["usage"]
        self.assertEqual(c.body["choices"][0]["finish_reason"], "length")
        self.assertEqual(u["completion_tokens"], 300 - u["prompt_tokens"] + 1, u)

    def test_the_cut_is_said_in_the_reply_on_every_route(self):
        """A reply that ends "stop" before the slot's room says nothing of the cut by itself, so the
        reply carries strata_max_tokens {asked, max_tokens} (the room) on the three routes, streamed or not; a
        max_tokens within the room is not cut and the field is absent."""
        self.start(2, extra_args=["--batch-ctx", "300"])
        for path, stream in (("/v1/chat/completions", False), ("/v1/chat/completions", True), ("/v1/messages", False),
                             ("/v1/responses", False)):
            for asked in (2000, 8):
                c = self.client("short one", stream=stream, max_tokens=asked, path=path)
                self.assertTrue(c.done.wait(30))
                self.assertEqual(c.status, 200, c.body)
                body = c.body
                if stream:
                    body = [json.loads(x) for _, x in c.events if x != "[DONE]" and "usage" in json.loads(x)][-1]
                u = body["usage"]
                prompt = u.get("prompt_tokens", u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0))
                if asked == 2000:
                    self.assertEqual(body.get("strata_max_tokens"), {"asked": 2000, "max_tokens": 300 - prompt + 1},
                                     (path, stream, body))
                else:
                    self.assertNotIn("strata_max_tokens", body, (path, stream))

    def test_the_limit_without_max_tokens_is_said_on_every_route(self):
        """A request that asks for no max_tokens is given one by the server (a
        slot's room, or the profile's default under it), and its reply says it as strata_max_tokens {"asked":
        null, "max_tokens": M} on the three routes, streamed or not, whether the reply ends "stop" before M or
        "length" at it; a max_tokens asked and kept still leaves the field out."""
        self.start(2, extra_args=["--batch-ctx", "300"])

        def last(c):
            self.assertTrue(c.done.wait(30))
            self.assertEqual(c.status, 200, c.body)
            if not c.stream:
                return c.body
            return [json.loads(x) for _, x in c.events if x != "[DONE]" and "usage" in json.loads(x)][-1]

        for path, stream in (("/v1/chat/completions", False), ("/v1/chat/completions", True), ("/v1/messages", False),
                             ("/v1/responses", False)):
            # no default: the slot's room, and the fake's short reply stops before it
            self.svc.profile.default_max_tokens = None
            body = last(self.client("short one", stream=stream, max_tokens=None, path=path))
            u = body["usage"]
            prompt = u.get("prompt_tokens", u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0))
            self.assertEqual(body.get("strata_max_tokens"), {"asked": None, "max_tokens": 300 - prompt + 1},
                             (path, stream, body))
            # a default of 40, under the slot's room: the fake's long reply ends "length" there
            self.svc.profile.default_max_tokens = 40
            body = last(self.client("VERYLONG limit", stream=stream, max_tokens=None, path=path))
            self.assertEqual(body.get("strata_max_tokens"), {"asked": None, "max_tokens": 40}, (path, stream, body))
            u = body["usage"]
            self.assertEqual(u.get("completion_tokens", u.get("output_tokens")), 40, (path, stream, body))
            # a max_tokens asked and kept
            body = last(self.client("short one", stream=stream, max_tokens=8, path=path))
            self.assertNotIn("strata_max_tokens", body, (path, stream))

    def test_a_prompt_past_a_slot_is_refused_beside_decoding_requests(self):
        """Slots of 300 positions, a runner decoding: a prompt past a slot gets a 400 naming the slot's size at
        once; once nothing decodes the same prompt is answered (alone)."""
        self.start(2, extra_args=["--batch-ctx", "300"])
        runner = self.client("VERYLONG runner", max_tokens=200)
        self.assertTrue(wait_until(lambda: runner.content_chunks() > 3))
        long_text = "a long prompt " + "word " * 80
        c = self.client(long_text, stream=False, max_tokens=8)
        self.assertTrue(c.done.wait(5), "refused at once")
        self.assertEqual(c.status, 400, c.body)
        self.assertIn("past a batch slot (300 positions", c.body["error"]["message"])
        runner.hang_up()
        self.assertTrue(wait_until(lambda: not any(self.engine.slot_busy) and not self.engine.in_engine))
        again = self.client(long_text, stream=False, max_tokens=8)
        self.assertTrue(again.done.wait(20))
        self.assertEqual(again.status, 200, again.body)

    def ran_alone_on(self, path):
        """On one route: a prompt past a slot (300 positions) decoding alone on the solo path; a short request comes,
        so it is moved into a slot, which it does not fit: it runs alone (the engine's ALONE line), /metrics shows it
        while it runs, the server's log says it at once, and its reply carries strata_ran_alone -> that field."""
        long_text = "VERYLONG and long " + "word " * 80
        c = self.client(long_text, stream=False, max_tokens=150, path=path)
        # its first tokens out (a request still reading would give way to the short one instead, #656)
        self.assertTrue(wait_until(lambda: any(st.get("generated", 0) > 3 for st, _ in list(self.svc.live_reqs.values()))))
        short = self.client("short one", max_tokens=16)
        seen = wait_until(lambda: (self.svc.metrics()["live"].get("ran_alone") or {}).get("held_slots") == 0, 10, 0.01)
        self.assertTrue(c.done.wait(30) and short.done.wait(30))
        self.assertTrue(seen, "no ran_alone in /metrics while it ran")
        self.assertEqual((c.status, short.status), (200, 200), c.body)
        self.assertIsNone(self.svc.metrics()["live"].get("ran_alone"))
        self.assertGreaterEqual(self.svc.metrics()["live"]["ran_alone_total"], 1)
        return c.body.get("strata_ran_alone")

    def test_ran_alone_is_said_at_its_start_and_on_every_route(self):
        self.start(2, extra_args=["--batch-ctx", "300"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):          # the server's log lines (print)
            for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                ran = self.ran_alone_on(path)
                self.assertEqual(ran and ran["held_slots"], 0, (path, ran))
                self.assertIn("a batch slot holds 300", ran["reason"])
        said = [x for x in out.getvalue().splitlines() if "this request runs alone, not in a slot" in x]
        self.assertEqual(len(said), 3, out.getvalue()[-600:])

    def test_no_max_tokens_takes_the_profile_default(self):
        """Without max_tokens (absent or 0) a reply ends at the profile's default_max_tokens, finish "length"."""
        self.start(2)
        self.svc.profile.default_max_tokens = 40
        for mt in (None, 0):
            c = self.client("VERYLONG default", stream=False, max_tokens=mt)
            self.assertTrue(c.done.wait(30))
            self.assertEqual(c.status, 200, c.body)
            self.assertEqual((c.body["usage"]["completion_tokens"], c.body["choices"][0]["finish_reason"]),
                             (40, "length"))


class Config(unittest.TestCase):
    def test_c04_config_keys(self):
        from serve.model_profile import DeepSeekV41Profile, ModelProfile
        from serve.server import interleave_args, queue_max_of, request_deadline_of
        dsv41 = DeepSeekV41Profile.__new__(DeepSeekV41Profile)
        self.assertEqual(interleave_args({"args": []}, dsv41), [])
        self.assertEqual(interleave_args({"args": [], "interleave_chunk": 0}, dsv41), ["--interleave-chunk", "0"])
        self.assertEqual(interleave_args({"args": [], "interleave_chunk": 2048}, dsv41), ["--interleave-chunk", "2048"])
        for cfg, prof in (({"args": [], "interleave_chunk": -1}, dsv41), ({"args": [], "interleave_chunk": "8"}, dsv41),
                          ({"args": [], "interleave_chunk": True}, dsv41), ({"args": [], "interleave_chunk": 1024}, None),
                          ({"args": [], "interleave_chunk": 1024}, ModelProfile()),
                          ({"args": ["--interleave-chunk", "2"], "interleave_chunk": 1024}, dsv41)):
            with self.assertRaises(ValueError, msg=str(cfg)):
                interleave_args(cfg, prof)
        # "hot_spare" as --hot-spare, deepseek-v4.1's alone; absent: the engine's default
        from serve.server import hot_spare_args
        self.assertEqual(hot_spare_args({"args": []}, dsv41), [])
        self.assertEqual(hot_spare_args({"args": [], "hot_spare": 4}, dsv41), ["--hot-spare", "4"])
        self.assertEqual(hot_spare_args({"args": [], "hot_spare": 0}, dsv41), ["--hot-spare", "0"])
        for cfg, prof in (({"args": [], "hot_spare": -1}, dsv41), ({"args": [], "hot_spare": "4"}, dsv41),
                          ({"args": [], "hot_spare": True}, dsv41), ({"args": [], "hot_spare": 4}, None),
                          ({"args": [], "hot_spare": 4}, ModelProfile()),
                          ({"args": ["--hot-spare", "2"], "hot_spare": 4}, dsv41)):
            with self.assertRaises(ValueError, msg=str(cfg)):
                hot_spare_args(cfg, prof)
        # the profiles' defaults: deepseek-v4.1 8 waiting and 1800 s, Qwen3.8 none (as before)
        self.assertEqual((queue_max_of({}, dsv41), request_deadline_of({}, dsv41)), (8, 1800.0))
        self.assertEqual((queue_max_of({}, ModelProfile()), request_deadline_of({}, None)), (None, None))
        self.assertEqual((queue_max_of({"queue_max": 0}, dsv41), queue_max_of({"queue_max": None}, dsv41)), (0, None))
        self.assertEqual((request_deadline_of({"request_deadline_s": 0}, dsv41),
                          request_deadline_of({"request_deadline_s": 90}, ModelProfile())), (None, 90.0))
        for bad in ({"queue_max": -1}, {"queue_max": "8"}, {"queue_max": True}):
            with self.assertRaises(ValueError):
                queue_max_of(bad, dsv41)
        for bad in ({"request_deadline_s": -1}, {"request_deadline_s": "60"}, {"request_deadline_s": float("inf")}):
            with self.assertRaises(ValueError):
                request_deadline_of(bad, dsv41)
        # with --batch and no "batch_ctx", deepseek-v4.1's slots hold 69632 positions (at most --ctx);
        # Qwen3.8, a config's own value or flag, no --batch or no --ctx: nothing added (the engine's default)
        from serve.server import batch_ctx_default_args
        batch = ["--ctx", "262144", "--batch", "8"]
        self.assertEqual(batch_ctx_default_args({}, dsv41, batch), ["--batch-ctx", "69632"])
        self.assertEqual(batch_ctx_default_args({}, dsv41, ["--ctx", "32768", "--batch", "8"]), ["--batch-ctx", "32768"])
        for cfg, prof, args in (({}, ModelProfile(), batch), ({}, None, batch), ({"batch_ctx": 4096}, dsv41, batch),
                                ({}, dsv41, batch + ["--batch-ctx", "4096"]), ({}, dsv41, ["--ctx", "262144"]),
                                ({}, dsv41, ["--batch", "8"])):
            self.assertEqual(batch_ctx_default_args(cfg, prof, args), [], msg=str((cfg, args)))
        # a whole number must follow --ctx, else a ValueError naming --ctx (not an IndexError)
        for args in (["--batch", "8", "--ctx"], ["--batch", "8", "--ctx", "big"], ["--ctx", "0", "--batch", "8"]):
            with self.assertRaisesRegex(ValueError, "--ctx", msg=str(args)):
                batch_ctx_default_args({}, dsv41, args)

    def test_slot_room_without_max_tokens(self):
        """With batch slots a request without max_tokens whose prompt fits a slot answers to the slot's
        end (batch_ctx - prompt + 1), never past the context; a prompt past a slot keeps the context's room; no batch
        slots (or the Qwen engine, which reports no batch_ctx): the context's room, as before."""
        from types import SimpleNamespace
        from serve.server import Service
        def room(info, prompt, ctx_room):
            return Service.slot_room(SimpleNamespace(engine=SimpleNamespace(info=info)), prompt, ctx_room)
        slots = {"batch_slots": "8", "batch_ctx": "69632"}
        self.assertEqual(room(slots, 16384, 245000), 69632 - 16384 + 1)
        self.assertEqual(room(slots, 16384, 1000), 1000)
        self.assertEqual(room(slots, 69631, 190000), 2)
        self.assertEqual(room(slots, 69632, 190000), 190000)       # past a slot: it runs alone, as before
        self.assertEqual(room({"batch_slots": "8"}, 16384, 245000), 245000)
        self.assertEqual(room({"checkpoints": "4"}, 16384, 245000), 245000)

    def test_retry_after_floor_and_alone_hold(self):
        """Retry-After is at least RETRY_AFTER_MIN_S; while a request runs alone past a slot
        its hold left is added (its tokens left at its own rate, or ALONE_RATE_FLOOR before it has 2)."""
        import collections
        from types import SimpleNamespace
        from serve.server import ALONE_RATE_FLOOR, RETRY_AFTER_MIN_S, Service
        svc = Service.__new__(Service)
        svc.status_lock, svc.finished_s, svc.admitted = threading.Lock(), collections.deque([0.5]), {}
        svc.parallel, svc.request_deadline_s = 1, 1800.0
        svc.engine = SimpleNamespace(batch=8, alone_now=None)
        self.assertEqual(svc.retry_after_s(0), RETRY_AFTER_MIN_S)
        now = time.monotonic()
        svc.engine.alone_now = {"since": now - 10.0, "generated": 100, "max_new": 300, "slot": 0, "held_slots": 4,
                                "reason": "r"}
        self.assertAlmostEqual(svc.alone_hold_s(), 20.0, delta=0.5)           # 200 left at 10 tok/s
        self.assertIn(svc.retry_after_s(0), (RETRY_AFTER_MIN_S + 20, RETRY_AFTER_MIN_S + 21))   # ceil of ~20.0
        svc.engine.alone_now = {"since": now, "generated": 0, "max_new": 400, "slot": 0, "held_slots": 4, "reason": "r"}
        self.assertAlmostEqual(svc.alone_hold_s(), 400 / ALONE_RATE_FLOOR, delta=0.5)
        svc.request_deadline_s = 5.0
        self.assertLessEqual(svc.alone_hold_s(), 5.0)
        self.assertEqual(svc.alone_view()["held_slots"], 4)

    def test_badm_alone_is_reported(self):
        """`BADM <slot> 0 rep=0 alone=N why=...` (a request refused a slot for its positions) puts
        {"reason", "held_slots"} on the engine's last reply, which the chat reply carries as strata_ran_alone."""
        import queue
        import threading
        from serve.server import StrataEngine
        e = StrataEngine.__new__(StrataEngine)
        e.lines, e.last, e._ctl_mode, e._ctl_result, e._yielded = queue.Queue(), None, "batch", None, None
        why = "16414 prompt tokens and 70000 to generate need 86413 positions, a batch slot holds 69632"
        for x in ("T 5", "DONE 1 16414 3000.0 20.0 length 0 0 0", f"BADM 3 0 rep=0 alone=4 why={why}"):
            e.lines.put(x + "\n")
        got = []
        list(e._control(threading.Event(), got.append))
        self.assertEqual(got, [5])
        self.assertEqual(e._ctl_result, ("badm", False))
        self.assertEqual(e.last["ran_alone"], {"reason": why, "held_slots": 4})
        self.assertIs(e.last["reproducible"], False)


if __name__ == "__main__":
    unittest.main()
