#!/usr/bin/env python3
"""tools/dsv41/pod/sheet_banners.py - the superseded banners of the Part C session sheets (written by e07_report.py
under SHEET_ROOT) and the index of the current verdict of each item (9-14), both written from the one table below. An
item whose session has not run yet is a row with no sheet: VERDICTS.md says it is pending. A current sheet may carry a
banner too, a note on how its verdict line reads (item 14, accepted on review). The table records this project's own
sessions of 2026-10-13 to 2026-10-20 (their public summary: docs/RESULTS.md); with other sheets, edit the table.

  sheet_banners.py [--check] [SHEET_ROOT]

SHEET_ROOT is e07.sh's (default $BUILD_ROOT/sheets, BUILD_ROOT default ~/dsv41-build). Each sheet in BANNERS gets its banner right
under its title, between two marker comments; a run replaces what is between them, so running it twice changes
nothing (a file is only written when its text changes). A banner written by hand before this script (LEGACY) is taken
out the first time. SHEET_ROOT/VERDICTS.md is written from VERDICTS: per item the sheet that holds its current
verdict, the verdict's numbers, the build, the date and the sheets it supersedes. Before writing, every sheet the table
names must exist (its README.md) and each current sheet must contain its verdict's figures (EVIDENCE, the sheet's own
text), else exit 2 naming them. e07_report.py rewrites a sheet whole, so after regenerating a superseded sheet run this
again. --check writes nothing and exits 1 when a file would change.

When a new session supersedes a verdict: add its row to VERDICTS (the old current sheet moves to "superseded") and a
banner for the old sheet to BANNERS, then run this script.
"""
import os
import sys

BEGIN = "<!-- sheet_banners.py: begin (tools/dsv41/pod/sheet_banners.py writes this banner from its table) -->"
END = "<!-- sheet_banners.py: end -->"
INDEX = "VERDICTS.md"
SOURCE = "review of 2026-10-16"
SOURCE_C05 = "review of 2026-10-20"


def link(sheet):
    """A sheet's README from another sheet's directory."""
    return f"[{sheet}](../{sheet}/README.md)"


INDEX_LINK = f"[{INDEX}](../{INDEX})"

# Part C (items 9-14): the current verdict of each item, from the sheets (the final cfinal session holds items 11
# and 14). The batch determinism sessions' sheets: those with speed figures, those with no run or a lost pod, the
# final one
C05_SPEED = ["2026-10-17-cdet-0418", "2026-10-19-cdet", "2026-10-19-cfinal", "2026-10-19-cfinal-1535",
             "2026-10-19-cfinal-2033"]
C05_NO_RUN = ["2026-10-17-cdet", "2026-10-18-cdet", "2026-10-19-cfinal-1955", "2026-10-20-cfinal",
              "2026-10-20-cfinal-0221", "2026-10-20-cfinal-0228"]
C05_FINAL = "2026-10-20-cfinal-0317"
CUR11 = (f"{link(C05_FINAL)} (the final session, build 281712c: the single user 63.11 tok/s, steady-c4 127.58 "
         "tok/s = 2.02x it, the slowest user 31.87 tok/s, PASS; 8 users 152.53 tok/s as reported in session 4)")

VERDICTS = [
    {"item": 9, "goal": "baseline: N = 1, 2, 4, 8 users against today's FIFO server (reported)",
     "sheet": "2026-10-13-cbase-2320", "date": "2026-10-13", "build": "strata-293e6603",
     "verdict": "PASS: TTFT p50 3.311 / 17.319 / 40.415 / 87.029 s at N = 1 / 2 / 4 / 8, queue p50 0.0 / "
                "10.964 / 33.962 / 80.3 s, per-user decode 54.05 / 50.29 / 49.71 / 48.56 tok/s, aggregate 31.88 / "
                "22.81 / 22.38 / 21.96 tok/s (RTX PRO 6000 Workstation)",
     "evidence": ["| 1 | 2 | 3.311 | 4.287 |", "| 8 | 16 | 87.029 | 88.189 | 80.3 |"],
     "superseded": ["2026-10-13-cbase", "2026-10-13-cbase-2303"]},
    {"item": 10, "goal": "4 cached 16K conversations, turn 2+ first token <= 2 s",
     "sheet": "2026-10-14-ccache-2121", "date": "2026-10-14", "build": "strata-940bbe6e",
     "verdict": "PASS: 8 of 8 turns 2+ resumed a cached prefix, first token at most 1.418 s when not "
                "queued (p50 1.257, p95 1.375 s), the reset wait at most 0.692 s",
     "evidence": ["Item 10 (4 alternating): PASS", "at most 1.418 s when not queued"],
     "superseded": ["2026-10-14-ccache", "2026-10-14-ccache-1743"]},
    {"item": 11, "goal": "batched decode: 4 users >= 2x the single user, each >= 20 tok/s; 8 users reported",
     "sheet": C05_FINAL, "date": "2026-10-20", "build": "281712c (tarball strata-281712ce)",
     "verdict": "PASS (the final session, the engine's defaults: --cpu-numerics fork, the "
                "deterministic mode on): the single user 63.11 tok/s with batching on against session 4's 61.55 "
                "(+2.5%); steady-c4 127.58 tok/s = 2.02x the single user, the slowest user 31.87 tok/s (RTX PRO 6000 "
                "Blackwell Server Edition, AMD EPYC 9575F, the engine bound with numactl --cpunodebind=1); 8 users "
                "152.53 tok/s as reported in session 4 (2026-10-16-cbatch-clong, build 763675b, not run since)",
     "evidence": ["the single user with batching on: PASS: 63.11 tok/s against session 4's 61.55 (+2.5%",
                  "steady-c4: PASS: 127.58 tok/s in aggregate", "= 2.02x the single user on the batched server",
                  "the slowest user 31.87 tok/s (goal: each >= 20)"],
     "superseded": ["2026-10-14-cbatch", "2026-10-15-cbatch-clong", "2026-10-15-cbatch-clong-1543",
                    "2026-10-15-cbatch-clong-2201", "2026-10-16-cbatch-clong"] + C05_SPEED},
    {"item": 12, "goal": "no user frozen by another's 64K prompt (largest gap <= 5 s)",
     "sheet": "2026-10-16-clong-1547", "date": "2026-10-16",
     "build": "serve and tools 5c582c3, engine tarball strata-b08375d3",
     "verdict": "PASS (session 6): the largest gap 4.048 s (3.737 s during the 65565-token read; bar 5 s), every user "
                "decoded through the read; the long TTFT 14.928 s beside them against 16.674 s alone (0.895x). The "
                "margin follows the host's H2D: 28.6 GB/s here (RTX PRO 6000 Workstation, PCIe Gen 4 x16)",
     "evidence": ["item 12: PASS: the largest gap between two tokens of any of the 4 users 4.048 s (3.737 s",
                  "14.928 s beside them against 16.674 s alone (0.895 x", "H2D 28.6 GB/s"],
     "superseded": ["2026-10-15-cbatch-clong", "2026-10-15-cbatch-clong-1543", "2026-10-15-cbatch-clong-2201",
                    "2026-10-16-cbatch-clong", "2026-10-16-clong"]},
    {"item": 13, "goal": "served and robust: stream, batched, disconnects, overflow, limits",
     "sheet": "2026-10-16-clong-1547", "date": "2026-10-16",
     "build": "serve and tools 5c582c3, engine tarball strata-b08375d3",
     "verdict": "PASS (session 6): 8 of 8 checks (P = 8 slots, queue_max 8): stream, non-stream, batched, "
                "disconnect stream and non-stream, overflow, no-max-tokens, slot-room",
     "evidence": ["Item 13: PASS (8 of 8 checks; P = 8 slots, queue_max 8)."],
     "superseded": ["2026-10-15-cbatch-clong", "2026-10-15-cbatch-clong-1543", "2026-10-15-cbatch-clong-2201",
                    "2026-10-16-cbatch-clong", "2026-10-16-clong"]},
    # item 14 read as no quality loss from batching (2026-10-19); the final session's strict interval rule failed on
    # HumanEval and the result was accepted on review
    {"item": 14, "goal": "no quality loss from batching: item 3 batched on all three tasks, item 4 against the "
                         "engine quality run",
     "sheet": C05_FINAL, "date": "2026-10-20", "build": "281712c (tarball strata-281712ce)",
     "verdict": "PASS, accepted on review (2026-10-20): GSM8K and IFEval no "
                "loss; HumanEval -2.4 points, CI [-4.9, -0.6], within the 3-point bar; the strict interval rule fails "
                "on HumanEval",
     "evidence": ["| gsm8k_cot_llama | 150 | 0.927 | 0.927 | +0.000 | [-0.027, +0.027] |",
                  "| ifeval | 150 | 0.800 | 0.800 | +0.000 | [-0.040, +0.040] |",
                  "| humaneval_instruct | 164 | 0.890 | 0.866 | -0.024 | [-0.049, -0.006] |",
                  "item 4 against Q02's engine run FAIL: humaneval_instruct: diff -0.024, 95% CI [-0.049, -0.006], "
                  "the interval lies below zero"],
     "superseded": C05_SPEED + C05_NO_RUN},
]

# the banner under each superseded sheet's title (the sheets' figures as they stand; numbers are never changed here)
BANNERS = {
    "2026-10-13-cbase": (
        f"> **No run, not a verdict ({SOURCE}; index: {INDEX_LINK}).** No out/ came back from this session. Item 9's "
        f"baseline is in {link('2026-10-13-cbase-2320')}."),
    "2026-10-13-cbase-2303": (
        f"> **No run, not a verdict ({SOURCE}; index: {INDEX_LINK}).** No out/ came back from this session. Item 9's "
        f"baseline is in {link('2026-10-13-cbase-2320')}."),
    "2026-10-14-cbatch": (
        f"> **Superseded for item 11 ({SOURCE}; index: {INDEX_LINK}).** The first cbatch session (build 90f5fed2). Its 4 "
        "users' line (FAIL) is the end-to-end figure, not item 11's verdict ("
        "the note under the table). Item 11's current verdict is in "
        f"{CUR11}. This sheet stays as the record of this session."),
    "2026-10-15-cbatch-clong": (
        f"> **Superseded ({SOURCE}; index: {INDEX_LINK}).** Item 11 on this sheet "
        "(steady-c4 FAIL, 1.33x, slots of 32768 positions) is superseded by "
        f"{CUR11}. Items 12 and 13 of this sheet (PASS on this "
        f"session's code) are superseded by {link('2026-10-16-clong-1547')} (session 6: the largest gap 4.048 s, 8 "
        "of 8 checks, PASS). This sheet stays as the record of this session."),
    "2026-10-15-cbatch-clong-1543": (
        f"> **Superseded ({SOURCE}; index: {INDEX_LINK}).** The batch speed and long-prompt verification of "
        "2026-10-15 15:43 (tarball strata-82d8d921): item 11 PASS here (2.21x), items 12 and 13 PASS (the largest "
        "gap 3.325 s, 7 of 7 checks), on the code before the later review fixes. Item 11's current verdict "
        f"is in {CUR11}; "
        f"items 12 and 13's in {link('2026-10-16-clong-1547')} (session 6: the largest gap 4.048 s, 8 of 8 checks, "
        "the eighth added since). This sheet stays as the record of this session."),
    "2026-10-15-cbatch-clong-2201": (
        f"> **Not a verdict ({SOURCE}; index: {INDEX_LINK}).** Session 3 of 2026-10-15 22:01 (build 763675b) ran on "
        "a slow host (Intel Xeon 6760P, host read 109.5 GB/s in the qualification, the engine's CPU expert jobs at "
        "9.3-10.8 GB/s), so its item 11 (FAIL, the slowest user 15.00 tok/s) measured the host, not the code; then "
        "the long prompt's server did not get ready (port 8090 still held by the server before: a harness race, "
        "fixed since), so items 12 and 13 did not run. It led to the qualification's host "
        f"read floor and the port wait. Item 11's verdict is in {CUR11}; items 12 and 13's in "
        f"{link('2026-10-16-clong-1547')} (session 6)."),
    "2026-10-16-cbatch-clong": (
        f"> **Superseded for items 11, 12 and 13 ({SOURCE}, {SOURCE_C05}; index: {INDEX_LINK}).** Item 11 on this "
        "sheet (session 4, build 763675b: steady-c4 123.82 tok/s = 2.06x the single user with batching off, 60.11 "
        "tok/s; the slowest user 30.93 tok/s; PASS) was the verdict until the batch determinism work; its current verdict is in "
        f"{link(C05_FINAL)} (the final session, build 281712c, the engine's defaults: the single "
        "user 63.11 tok/s, +2.5% against this sheet's 61.55; steady-c4 127.58 tok/s = 2.02x, the slowest user 31.87 "
        "tok/s, PASS). This sheet's 8 users (152.53 tok/s, reported, not a bar) stay the last 8-user figure: no "
        "session since ran 8 users. Items 12 (PASS, the largest gap 3.342 s) and 13 (no result: the robust step "
        f"killed at 480 s by an overflow hang, fixed since) are superseded by {link('2026-10-16-clong-1547')} "
        "(session 6: the largest gap 4.048 s, 8 of 8 checks, PASS)."),
    "2026-10-16-clong": (
        f"> **Superseded ({SOURCE}; index: {INDEX_LINK}).** Session 5 (engine tarball strata-b08375d3, the server "
        "before the near-turn hardening): item 12 PASS (the largest gap 3.342 s), item 13 FAIL 7 of 8 (slot-room's clamp check wanted "
        "finish \"length\" while the model stopped on its own after 13 of its 137 tokens of room; the check takes the "
        "server's strata_max_tokens since). Items 12 and 13's current verdicts are in "
        f"{link('2026-10-16-clong-1547')} (session 6, the same engine tarball, serve and tools 5c582c3: the largest "
        "gap 4.048 s, 8 of 8 checks, PASS)."),
    # every session before the final one, as its README.md and logs/e07.log say
    "2026-10-17-cdet": (
        f"> **No run, not a verdict ({SOURCE_C05}; index: {INDEX_LINK}).** The first cdet session (build e4066852): "
        "the HF download guard stopped it at 0.133 GB/s before any engine step (0.256 USD). "
        f"Items 11 and 14's current verdicts are in {link(C05_FINAL)}."),
    "2026-10-17-cdet-0418": (
        f"> **Superseded ({SOURCE_C05}; index: {INDEX_LINK}).** The second cdet session (build e4066852, AMD "
        "EPYC 9335, the CPU experts in the GPU's numerics): item 14 as it then stood (identical tokens) FAIL, the 3 "
        "promoted chains apart at tokens 67 and 91 (fixed since); the single user 50.15 tok/s FAIL, "
        "steady-c4 109.06 tok/s = 2.15x; HumanEval not run. Item 14 was redefined on 2026-10-19 as no quality loss from "
        "batching. Items 11 and 14's current verdicts are in "
        f"{link(C05_FINAL)}. This sheet stays as the record of this session."),
    "2026-10-18-cdet": (
        f"> **Lost pod, not a verdict ({SOURCE_C05}; index: {INDEX_LINK}).** The pod (its "
        "qualification passed) ran its steps into lm-eval, then stopped answering and RunPod removed "
        "it (HTTP 404): no out/ came back, so the sheet below says no run (1.578 USD). Items 11 and 14's current "
        f"verdicts are in {link(C05_FINAL)}."),
    "2026-10-19-cdet": (
        f"> **Superseded ({SOURCE_C05}; index: {INDEX_LINK}).** The cdet rerun (build 4e3f287d, AMD "
        "EPYC 9575F, the CPU experts in the GPU's numerics): item 14 as it then stood (identical tokens) PASS, 12 "
        "chains equal to alone; the single user 48.14 tok/s and steady-c4 27.60 tok/s FAIL; HumanEval timed out. "
        "Its A/B (the single user, the mode off: 47.04 tok/s in the GPU's numerics against 63.34 in the fork's) led "
        "to item 14's redefinition and the fork's numerics by default. Items 11 and "
        f"14's current verdicts are in {link(C05_FINAL)}. This sheet stays as the record of this session and of the "
        "A/B."),
    "2026-10-19-cfinal": (
        f"> **Superseded ({SOURCE_C05}; index: {INDEX_LINK}).** The first cfinal session (build b8409281): item 14 "
        "PASS on this run (HumanEval -0.006 against the engine quality run), item 11 FAIL (the single user 56.89 tok/s, "
        "steady-c4 45.33 tok/s: the batched decode's regression, fixed since). Items 11 and 14's current verdicts "
        f"are in {link(C05_FINAL)} (the three tasks batched again: HumanEval -0.024 there)."),
    "2026-10-19-cfinal-1535": (
        f"> **Superseded ({SOURCE_C05}; index: {INDEX_LINK}).** The cfinal rerun with that fix (build f9f373bd, "
        "RTX PRO 6000 Workstation, AMD EPYC 9355P): item 14 PASS on this run (HumanEval -0.012), steady-c4 128.09 "
        "tok/s PASS, the single user 57.06 tok/s FAIL (the host's CPU experts, hence the CPU experts floor). Items 11 and 14's current "
        f"verdicts are in {link(C05_FINAL)}."),
    "2026-10-19-cfinal-1955": (
        f"> **No run, not a verdict ({SOURCE_C05}; index: {INDEX_LINK}).** No machine passed the qualification (one "
        "rejected for its 547 MHz SM clock; the session spent at most 0.035 USD); no step ran. Items 11 and 14's "
        f"current verdicts are in {link(C05_FINAL)}."),
    "2026-10-19-cfinal-2033": (
        f"> **Lost pod, speed only, not a verdict ({SOURCE_C05}; index: {INDEX_LINK}).** The cfinal session on a "
        "CPU-qualified host (build 281712c, AMD EPYC 9655): steady-c4 121.42 tok/s PASS, the single user 53.66 tok/s "
        "FAIL (the engine was not NUMA-bound, fixed since); the pod stopped answering during IFEval, so item 14 "
        f"has no result (1.178 USD). Items 11 and 14's current verdicts are in {link(C05_FINAL)}."),
    "2026-10-20-cfinal": (
        f"> **No run, not a verdict ({SOURCE_C05}; index: {INDEX_LINK}).** The HF download guard stopped it at 0.127 "
        "GB/s before any engine step (0.246 USD). Items 11 and 14's current verdicts are in "
        f"{link(C05_FINAL)}."),
    "2026-10-20-cfinal-0221": (
        f"> **No run, not a verdict ({SOURCE_C05}; index: {INDEX_LINK}).** No machine passed the qualification (one "
        "rejected for its 198.0 GB/s host read; the session spent at most 0.039 USD); no step ran. Items 11 and 14's "
        f"current verdicts are in {link(C05_FINAL)}."),
    "2026-10-20-cfinal-0228": (
        f"> **No run, not a verdict ({SOURCE_C05}; index: {INDEX_LINK}).** The HF download guard stopped it at 0.134 "
        "GB/s before any engine step (0.250 USD; the same slow machine as 2026-10-20-cfinal, so "
        f"BAD_MACHINES since). Items 11 and 14's current verdicts are in {link(C05_FINAL)}."),
    C05_FINAL: (
        f"> **Current for items 11 and 14; item 14 accepted on review ({SOURCE_C05}; index: {INDEX_LINK}).** This "
        "sheet's item 14 line says FAIL by the strict interval rule: HumanEval -0.024 against the engine quality run, 95% "
        "CI [-0.049, -0.006], below zero (GSM8K and IFEval +0.000). The loss is 2.4 points, within the 3-point bar; "
        "over the three batched sessions HumanEval moved -0.6, -1.2 and -2.4 points (pooled -1.4, paired bootstrap "
        "95% CI [-3.5, +0.2]). It was accepted on review on 2026-10-20, so item 14 is "
        "PASS by that acceptance; the rule stays as written and the figures below are unchanged."),
}

# banners written by hand before this script, taken out once (their first line's start)
LEGACY = {"2026-10-15-cbatch-clong": "> **Superseded (2026-10-15).**"}


class TableError(Exception):
    pass


def readme(root, sheet):
    return os.path.join(root, sheet, "README.md")


def check_table(root):
    """Every sheet the table names exists, and each current sheet holds its verdict's figures."""
    named = set(BANNERS) | set(LEGACY)
    for v in VERDICTS:
        named |= {*([v["sheet"]] if v["sheet"] else []), *v["superseded"]}
    for b in BANNERS.values():
        named |= {part.split("](")[0] for part in b.split("[")[1:] if "](../" in part and part.split("](")[0] != INDEX}
    missing = sorted(s for s in named if not os.path.isfile(readme(root, s)))
    if missing:
        raise TableError("no README.md for " + ", ".join(missing) + f" under {root}")
    for v in VERDICTS:
        if not v["sheet"]:
            continue   # pending: no session yet
        with open(readme(root, v["sheet"])) as f:
            text = f.read()
        lost = [e for e in v["evidence"] if e not in text]
        if lost:
            raise TableError(f"item {v['item']}: {v['sheet']}/README.md does not say " + "; ".join(repr(e) for e in lost))
        if v["sheet"] in v["superseded"]:
            raise TableError(f"item {v['item']}: {v['sheet']} is both current and superseded")


def with_banner(text, sheet):
    """The README's text with BANNERS[sheet] under its title: the marked block replaced, else inserted after the title
    (and LEGACY's hand-written banner taken out)."""
    lines = text.split("\n")
    if not lines or not lines[0].startswith("# "):
        raise TableError(f"{sheet}/README.md does not start with a '# ' title")
    block = [BEGIN, BANNERS[sheet], END]
    if BEGIN in lines:
        i = lines.index(BEGIN)
        if END not in lines[i:]:
            raise TableError(f"{sheet}/README.md has {BEGIN!r} without {END!r}")
        j = lines.index(END, i)
        return "\n".join(lines[:i] + block + lines[j + 1:])
    rest = lines[1:]
    while rest and rest[0] == "":
        rest = rest[1:]
    old = LEGACY.get(sheet)
    if old and rest and rest[0].startswith(old):
        while rest and rest[0].startswith(">"):
            rest = rest[1:]
        while rest and rest[0] == "":
            rest = rest[1:]
    return "\n".join([lines[0], ""] + block + [""] + rest)


def index_text():
    rows = []
    for v in VERDICTS:
        current = f"[{v['sheet']}]({v['sheet']}/README.md)" if v["sheet"] else "none yet"
        rows.append(f"| {v['item']} | {v['goal']} | {current} | {v['verdict']} | "
                    f"{v['build']} | {v['date']} | "
                    + ", ".join(f"[{s}]({s}/README.md)" for s in v["superseded"]) + " |")
    return "\n".join([
        "# Part C: the current verdict of each item",
        "",
        "Written by Strata's tools/dsv41/pod/sheet_banners.py from its table; edit the "
        "table there and run it again, do not edit this file. Each superseded sheet says so in a banner under its "
        "title, with a link to the sheet that holds the current verdict. Every verdict on the RTX PRO 6000; the public "
        "summary is docs/RESULTS.md.",
        "",
        "| item | goal | current sheet | verdict | build | date | superseded (or not a verdict) |",
        "|---|---|---|---|---|---|---|",
    ] + rows) + "\n"


def plan(root):
    """(path, new text) of every file that would change."""
    check_table(root)
    out = []
    for sheet in sorted(BANNERS):
        p = readme(root, sheet)
        with open(p) as f:
            old = f.read()
        new = with_banner(old, sheet)
        if new != old:
            out.append((p, new))
    p = os.path.join(root, INDEX)
    old = None
    if os.path.exists(p):
        with open(p) as f:
            old = f.read()
    if old != index_text():
        out.append((p, index_text()))
    return out


def main(argv):
    check = "--check" in argv
    args = [a for a in argv if a != "--check"]
    if len(args) > 1 or any(a.startswith("-") for a in args):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    root = args[0] if args else os.path.join(os.environ.get("BUILD_ROOT", os.path.expanduser("~/dsv41-build")), "sheets")
    root = os.environ.get("SHEET_ROOT", root) if not args else root
    try:
        changes = plan(root)
    except TableError as e:
        print(f"sheet_banners.py: {e}", file=sys.stderr)
        return 2
    for p, new in changes:
        print(f"sheet_banners.py: {'would change' if check else 'wrote'} {p}")
        if not check:
            with open(p, "w") as f:
                f.write(new)
    if not changes:
        print(f"sheet_banners.py: {len(BANNERS)} banners and {INDEX} up to date under {root}")
    return 1 if check and changes else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
