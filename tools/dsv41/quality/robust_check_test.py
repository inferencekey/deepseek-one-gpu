#!/usr/bin/env python3
"""tools/dsv41/quality/robust_check_test.py - slot-room's clamp acceptance (robust_check.clamp_verdict) and
no-max-tokens' NEAR acceptance (robust_check.near_verdict), both robust_check.room_verdict, against the real
serve/server.py on serve/test_parallel.py's fake engine (slots of 300 positions, a 4096 context), without a GPU.

The clong session of 2026-10-16 (session 5): the server cut max_tokens 32000 to the slot's room of 137, but the model ended its
answer after 13 tokens (finish "stop"), and the check, which wanted finish "length" at the room, failed. The check now
takes the cut from the reply's strata_max_tokens, the server's own word, and accepts a reply that stops within it:

  early stop, cut proven     a short reply (the fake's "ok, done.") to a max_tokens of 2000 past its slot's room:
                             finish "stop", fewer tokens than the room, strata_max_tokens {asked 2000, max_tokens the
                             room}: PASS
  no cut                     the same request on a server whose cut is turned off (Service.slot_room returns what it
                             is given): no strata_max_tokens, so nothing proves a cut: FAIL; and the early stop's
                             reply saying the server kept max_tokens 2000 (no cut): FAIL
  session 5's figures        prompt 69496 of a slot of 69632 (room 137), finish "stop" after 13 tokens, with the
                             strata_max_tokens the server now sends {asked 32000, max_tokens 137}: PASS; the same
                             without it (what session 5's server sent): FAIL; finish "length" at 137: PASS; "length"
                             short of the room or "stop" past it: FAIL

The same for NEAR: NEAR's turn asks for no max_tokens and wanted finish "length" at the slot's
room, the same weakness; the server now says the limit it gave such a request, strata_max_tokens {"asked": null,
"max_tokens": M}:

  NEAR early stop, said      the fake's short reply with no max_tokens: finish "stop" before the room,
                             strata_max_tokens {asked null, max_tokens the room}: PASS; the field saying a max_tokens
                             asked: FAIL
  NEAR, nothing said         the same on a server that says no limit (strata_max_tokens turned off, what sessions 4 and
                             5's server sent): FAIL
  sessions 4 and 5's NEAR    prompt 69473 of a slot of 69632 (room 160), finish "length" after 160 tokens (both
                             sessions' quality-server.log), with the field the server now sends {asked null, max_tokens
                             160}: PASS; without it: FAIL; "stop" after 13: PASS; a default under the room (100): FAIL

    PYTHONPATH=<Strata> python3 tools/dsv41/quality/robust_check_test.py -v
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parents[2])]

import robust_check  # noqa: E402
import serve.server  # noqa: E402
from serve.test_parallel import FakeEngineService  # noqa: E402

SLOT = 300
ASKED = 2000             # past a 300-position slot's room, within the fake's 4096 context


class ClampOnTheFakeServer(FakeEngineService):
    def ask(self) -> tuple[robust_check.Client, dict]:
        c = robust_check.Client(self.base, "short one", False, ASKED, 30)
        self.assertTrue(c.done.wait(30), "no reply")
        return c, robust_check.clamp_verdict(c.status, c.body, SLOT, ASKED)

    def test_early_stop_with_the_cut_proven_passes(self):
        self.start(2, extra_args=["--batch-ctx", str(SLOT)])
        c, v = self.ask()
        self.assertEqual(c.status, 200, c.body)
        room = SLOT - c.body["usage"]["prompt_tokens"] + 1
        self.assertEqual(c.body["strata_max_tokens"], {"asked": ASKED, "max_tokens": room})
        self.assertEqual((v["clamp_finish"], v["clamp_room"], v["clamp_max_tokens"]), ("stop", room, room), v)
        self.assertLess(v["clamp_completion_tokens"], room, v)
        self.assertTrue(v["clamp_ok"], v)
        # the reply saying the server kept the max_tokens asked: nothing was cut
        kept = copy.deepcopy(c.body)
        kept["strata_max_tokens"]["max_tokens"] = ASKED
        self.assertFalse(robust_check.clamp_verdict(200, kept, SLOT, ASKED)["clamp_ok"])

    def test_no_cut_fails(self):
        self.start(2, extra_args=["--batch-ctx", str(SLOT)])
        self.svc.slot_room = lambda prompt, max_new: max_new       # the server's cut turned off
        c, v = self.ask()
        self.assertEqual(c.status, 200, c.body)
        self.assertNotIn("strata_max_tokens", c.body)
        self.assertEqual(v["clamp_finish"], "stop", v)
        self.assertFalse(v["clamp_ok"], v)


class NearOnTheFakeServer(FakeEngineService):
    def ask(self) -> tuple[robust_check.Client, dict]:
        c = robust_check.Client(self.base, "short one", False, None, 30)
        self.assertTrue(c.done.wait(30), "no reply")
        return c, robust_check.near_verdict(c.status, c.body, SLOT, self.svc.profile.default_max_tokens)

    def test_near_early_stop_with_the_limit_said_passes(self):
        self.start(2, extra_args=["--batch-ctx", str(SLOT)])
        c, v = self.ask()
        self.assertEqual(c.status, 200, c.body)
        room = SLOT - c.body["usage"]["prompt_tokens"] + 1
        self.assertEqual(c.body["strata_max_tokens"], {"asked": None, "max_tokens": room})
        self.assertEqual((v["near_finish"], v["near_room"], v["near_max_tokens"]), ("stop", room, room), v)
        self.assertLess(v["near_completion_tokens"], room, v)
        self.assertTrue(v["near_ok"], v)
        # the field saying a max_tokens was asked: not NEAR's request
        asked = copy.deepcopy(c.body)
        asked["strata_max_tokens"]["asked"] = room
        self.assertFalse(robust_check.near_verdict(200, asked, SLOT, None)["near_ok"])

    def test_near_without_the_limit_said_fails(self):
        self.start(2, extra_args=["--batch-ctx", str(SLOT)])
        with mock.patch.object(serve.server, "strata_max_tokens", lambda asked, used: None):
            c, v = self.ask()
        self.assertEqual(c.status, 200, c.body)
        self.assertNotIn("strata_max_tokens", c.body)
        self.assertEqual(v["near_finish"], "stop", v)
        self.assertFalse(v["near_ok"], v)


def session5(finish: str, n: int, cut: bool = True) -> dict:
    """slot-room's clamp turn as session 5's robust.json has it (prompt 69496, cached 69473), with or without the
    strata_max_tokens the server now sends."""
    body = {"choices": [{"finish_reason": finish}],
            "usage": {"prompt_tokens": 69496, "completion_tokens": n, "prompt_tokens_details": {"cached_tokens": 69473}}}
    if cut:
        body["strata_max_tokens"] = {"asked": 32000, "max_tokens": 137}
    return body


class ClampOnSession5(unittest.TestCase):
    def verdict(self, body):
        return robust_check.clamp_verdict(200, body, 69632, robust_check.LONG_MAX_TOKENS)

    def test_session5(self):
        v = self.verdict(session5("stop", 13))
        self.assertEqual((v["clamp_room"], v["clamp_ok"]), (137, True), v)
        self.assertFalse(self.verdict(session5("stop", 13, cut=False))["clamp_ok"])
        self.assertTrue(self.verdict(session5("length", 137))["clamp_ok"])
        self.assertFalse(self.verdict(session5("length", 120))["clamp_ok"])
        self.assertFalse(self.verdict(session5("stop", 138))["clamp_ok"])


def near_session(finish: str, n: int, said: bool = True) -> dict:
    """no-max-tokens' NEAR turn as sessions 4 and 5 ran it (prompt 69473, no max_tokens; quality-server.log: "done: 160
    tokens ... (length"), with or without the strata_max_tokens the server now sends."""
    body = {"choices": [{"finish_reason": finish}], "usage": {"prompt_tokens": 69473, "completion_tokens": n}}
    if said:
        body["strata_max_tokens"] = {"asked": None, "max_tokens": 160}
    return body


class NearOnSessions4And5(unittest.TestCase):
    def test_sessions_4_and_5(self):
        def ok(body, default=8192):
            return robust_check.near_verdict(200, body, 69632, default)["near_ok"]
        self.assertEqual(robust_check.near_verdict(200, near_session("length", 160), 69632, 8192)["near_room"], 160)
        self.assertTrue(ok(near_session("length", 160)))
        self.assertFalse(ok(near_session("length", 160, said=False)))
        self.assertTrue(ok(near_session("stop", 13)))
        self.assertFalse(ok(near_session("length", 159)))
        self.assertFalse(ok(near_session("length", 160), default=100))


if __name__ == "__main__":
    unittest.main()
