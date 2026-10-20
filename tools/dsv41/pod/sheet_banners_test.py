#!/usr/bin/env python3
"""tools/dsv41/pod/sheet_banners_test.py - sheet_banners.py on a temporary sheet root,
run locally: a README per sheet its table names (a title, a body with each current sheet's verdict figures), the
hand-written banner of 2026-10-15-cbatch-clong in place.

  banners and index    every sheet in BANNERS has its banner right under its title between the markers, each link
                       to an existing sheet; the hand-written banner is gone; the body is kept byte for byte;
                       VERDICTS.md lists items 9-14 with their current and superseded sheets; every batch
                       determinism sheet has a banner, the final one (current for items 11 and 14) item 14's acceptance
  twice                a second run changes no file (same bytes, nothing written), --check exits 0 after the first
                       run and 1 before it
  a banner changed     a banner's text edited in the table: the marked block is replaced, not stacked
  table errors         a sheet the table names missing, or a current sheet without its verdict's figures: exit 2,
                       nothing written
"""
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sheet_banners as sb  # noqa: E402

BODY = "Session of `tools/dsv41/pod/e07.sh`.\n\n## Item 11\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"


def sheets():
    names = set(sb.BANNERS) | set(sb.LEGACY)
    for v in sb.VERDICTS:
        names |= {*([v["sheet"]] if v["sheet"] else []), *v["superseded"]}
    return sorted(names)


def make_root(root):
    for s in sheets():
        os.makedirs(os.path.join(root, s))
        text = f"# {s}\n\n"
        if s in sb.LEGACY:
            text += sb.LEGACY[s] + " Item 11 ... superseded by the old sheet.\n\n"
        text += BODY + "".join(e + "\n" for v in sb.VERDICTS if v["sheet"] == s for e in v["evidence"])
        with open(os.path.join(root, s, "README.md"), "w") as f:
            f.write(text)


def snapshot(root):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                out[p] = (fh.read(), os.stat(p).st_mtime_ns)
    return out


class Banners(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        make_root(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, *p):
        with open(os.path.join(self.root, *p)) as f:
            return f.read()

    def test_banners_and_index(self):
        self.assertEqual(sb.main([self.root]), 0)
        for s in sb.BANNERS:
            lines = self.read(s, "README.md").split("\n")
            self.assertEqual(lines[0], f"# {s}")
            self.assertEqual(lines[1:6], ["", sb.BEGIN, sb.BANNERS[s], sb.END, ""], s)
            self.assertTrue(lines[3].startswith("> **"), s)
            for target in re.findall(r"\]\(\.\./([^)]+)\)", lines[3]):
                self.assertTrue(os.path.isfile(os.path.join(self.root, target)), f"{s} links {target}")
            self.assertIn(BODY, "\n".join(lines[6:]), s)
        self.assertNotIn(sb.LEGACY["2026-10-15-cbatch-clong"], self.read("2026-10-15-cbatch-clong", "README.md"))
        index = self.read("VERDICTS.md")
        for v in sb.VERDICTS:
            row = next(ln for ln in index.split("\n") if ln.startswith(f"| {v['item']} | "))
            self.assertIn(f"[{v['sheet']}]({v['sheet']}/README.md)", row)
            for s in v["superseded"]:
                self.assertIn(f"[{s}]({s}/README.md)", row)
        self.assertEqual(sorted(v["item"] for v in sb.VERDICTS), [9, 10, 11, 12, 13, 14])
        for s in sb.C05_SPEED + sb.C05_NO_RUN + [sb.C05_FINAL]:
            self.assertIn(s, sb.BANNERS)
        self.assertEqual({v["sheet"] for v in sb.VERDICTS if v["item"] in (11, 14)}, {sb.C05_FINAL})
        self.assertIn("accepted on review", sb.BANNERS[sb.C05_FINAL])
        for target in re.findall(r"\]\(([^)]+)\)", index):
            self.assertTrue(os.path.isfile(os.path.join(self.root, target)), target)

    def test_twice(self):
        self.assertEqual(sb.main(["--check", self.root]), 1)
        self.assertEqual(sb.main([self.root]), 0)
        before = snapshot(self.root)
        self.assertEqual(sb.main(["--check", self.root]), 0)
        self.assertEqual(sb.main([self.root]), 0)
        self.assertEqual(snapshot(self.root), before)

    def test_a_banner_changed(self):
        sb.main([self.root])
        s = "2026-10-16-clong"
        old = sb.BANNERS[s]
        sb.BANNERS[s] = "> **Superseded (test).** See " + sb.link("2026-10-16-clong-1547") + "."
        try:
            self.assertEqual(sb.main([self.root]), 0)
            text = self.read(s, "README.md")
            self.assertEqual(text.count(sb.BEGIN), 1)
            self.assertIn(sb.BANNERS[s], text)
            self.assertNotIn(old, text)
        finally:
            sb.BANNERS[s] = old

    def test_table_errors(self):
        os.rename(os.path.join(self.root, "2026-10-16-clong-1547"), os.path.join(self.root, "gone"))
        self.assertEqual(sb.main([self.root]), 2)
        os.rename(os.path.join(self.root, "gone"), os.path.join(self.root, "2026-10-16-clong-1547"))
        with open(os.path.join(self.root, sb.C05_FINAL, "README.md"), "w") as f:
            f.write(f"# {sb.C05_FINAL}\n\n" + BODY)   # its items 11 and 14 figures gone
        self.assertEqual(sb.main([self.root]), 2)
        self.assertFalse(os.path.exists(os.path.join(self.root, "VERDICTS.md")))
        self.assertNotIn(sb.BEGIN, self.read("2026-10-16-clong", "README.md"))


if __name__ == "__main__":
    unittest.main()
