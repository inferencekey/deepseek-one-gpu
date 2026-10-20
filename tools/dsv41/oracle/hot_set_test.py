#!/usr/bin/env python3
"""Tests hot_set.py on tiny generated split GGUF header pairs and a tiny dump, with python3 only.

  hot_set_test.py

For split.count written as each GGUF integer type (UINT8 to INT64; the DeepSeek Q2_K shards have UINT16, which the first
version rejected), a 2-shard model whose layer 0 experts are in shard 1 and layer 1 experts in shard 2 must give a hot set
with both layers; a string split.count must fail naming the key. --layers-full 1-1 must keep every expert of layer
1 in id order and none of layer 0, and fail when the budget cannot hold them. --ram-budget-gib must pack the warm
lists after the VRAM set at strata-moe's warm slot cost (each part rounded up to 4096 bytes plus 4096: 90,112 bytes
per expert here), in descending count, and a hot set without it must have no warm lists.
--counts reads strata-moe counts files (strata-moe-counts/1, the text dsv41_moe_counts_test checks the library
writes): two files merge by summing, --counts-calls decode (the default) ignores the batch counts and all adds them,
a dump and a counts file sum too; a wrong format, a layer whose expert count differs between files or from the
model, and a layer without activations fail naming it. --propose-ram-budget: an A100 SXM pod's cgroup limit with a 4.5
GiB working set and an 8 GiB margin gives 103 GiB, its GPU node's 86.11 GiB MemFree caps it at 78, and a limit with
no room fails. Exits 1 on any failure.
"""

import json
import os
import struct
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import gguf_header  # noqa: E402

N_EMBD, N_FF, N_EXPERT, N_USED = 256, 256, 4, 2
Q2_K, Q3_K = 10, 11
INT_TYPES = {0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32", 5: "INT32", 10: "UINT64", 11: "INT64"}


def write_shard(path, split_type, split_value, layer):
    tensors = []
    for kind, ne, t in (("up", [N_EMBD, N_FF, N_EXPERT], Q2_K), ("gate", [N_EMBD, N_FF, N_EXPERT], Q2_K),
                        ("down", [N_FF, N_EMBD, N_EXPERT], Q3_K)):
        tensors.append({"name": f"blk.{layer}.ffn_{kind}_exps.weight", "ne": ne, "type": t, "offset": 0})
    metadata = {"general.architecture": (gguf_header.STRING, "deepseek41", None),
                "split.count": (split_type, split_value, None)}
    header = gguf_header.Header(3, metadata, tensors, gguf_header.DEFAULT_ALIGNMENT, 0)
    sizes = {t["name"]: gguf_header.tensor_nbytes(t["type"], t["ne"]) for t in tensors}
    gguf_header.write_gguf(path, header, lambda name: bytes(sizes[name]))


def split_type_in_file(path):
    """The value type id written after the split.count key, read from the raw bytes."""
    with open(path, "rb") as f:
        raw = f.read(4096)
    key = b"split.count"
    at = raw.index(struct.pack("<Q", len(key)) + key) + 8 + len(key)
    return struct.unpack_from("<I", raw, at)[0]


def write_dump(d):
    os.makedirs(d)
    files = []
    for layer in (0, 1):
        name = f"topk-L{layer:03d}-u0000.bin"
        # 3 tokens: expert 3 twice, expert 1 three times, expert 0 once
        ids = [3, 1, 1, 3, 1, 0]
        with open(os.path.join(d, name), "wb") as f:
            f.write(struct.pack(f"<{len(ids)}i", *ids))
        files.append({"kind": "topk", "layer": layer, "ubatch": 0, "file": name, "dtype": "i32", "ne": [N_USED, 3, 1, 1]})
    with open(os.path.join(d, "dump-manifest.json"), "w") as f:
        json.dump({"files": files}, f)


def hot_set(*args):
    p = subprocess.run([sys.executable, os.path.join(here, "hot_set.py"), *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    failed = 0

    def check(ok, what, out=""):
        nonlocal failed
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failed += 1
            print(out)

    with tempfile.TemporaryDirectory() as tmp:
        dump = os.path.join(tmp, "dump")
        write_dump(dump)
        for t, name in INT_TYPES.items():
            prefix = os.path.join(tmp, f"m{t}")
            first = f"{prefix}-00001-of-00002.gguf"
            write_shard(first, t, 2, 0)
            write_shard(f"{prefix}-00002-of-00002.gguf", t, 2, 1)
            out_json = os.path.join(tmp, f"hot{t}.json")
            rc, out = hot_set("--dump", dump, "--gguf", first, "--budget-gib", "1", "-o", out_json)
            layers = []
            if rc == 0:
                with open(out_json) as f:
                    layers = [(l["layer"], len(l["experts"]), l["types"]["down"]) for l in json.load(f)["layers"]]
            check(rc == 0 and split_type_in_file(first) == t and layers == [(0, 4, "q3_K"), (1, 4, "q3_K")],
                  f"split.count as {name} (type {split_type_in_file(first)}): both shards read, layers {layers}", out)

        prefix = os.path.join(tmp, "mstr")
        write_shard(f"{prefix}-00001-of-00002.gguf", gguf_header.STRING, "2", 0)
        rc, out = hot_set("--dump", dump, "--gguf", f"{prefix}-00001-of-00002.gguf", "--budget-gib", "1", "-o",
                          os.path.join(tmp, "x.json"))
        check(rc == 1 and "split.count has GGUF value type 8" in out, "split.count as a string: fails naming it", out)

        first = os.path.join(tmp, "m2-00001-of-00002.gguf")
        out_json = os.path.join(tmp, "full.json")
        rc, out = hot_set("--dump", dump, "--gguf", first, "--budget-gib", "1", "--layers-full", "1-1", "-o", out_json)
        experts = {}
        if rc == 0:
            with open(out_json) as f:
                experts = {l["layer"]: l["experts"] for l in json.load(f)["layers"]}
        check(rc == 0 and experts == {0: [], 1: [0, 1, 2, 3]}, f"--layers-full 1-1: experts per layer {experts}", out)

        rc, out = hot_set("--dump", dump, "--gguf", first, "--budget-gib", "0.0001", "--layers-full", "0-1", "-o",
                          os.path.join(tmp, "x.json"))
        check(rc == 1 and "more than --budget-gib" in out, "--layers-full over the budget: fails", out)

        # the warm tier: VRAM for one expert per layer (71,168 bytes plus the slab allowance of 940), warm for one
        # per layer at 90,112 bytes each: VRAM takes expert 1 (3 activations), warm expert 3 (2) in both layers
        def packed(*extra):
            out_json = os.path.join(tmp, "warm.json")
            rc, out = hot_set("--dump", dump, "--gguf", first, *extra, "-o", out_json)
            if rc != 0:
                return rc, out, None
            with open(out_json) as f:
                return rc, out, json.load(f)

        rc, out, doc = packed("--budget-gib", repr(2 * 72108 / 2**30), "--ram-budget-gib", repr(2 * 90112 / 2**30))
        got = doc and ({l["layer"]: l["experts"] for l in doc["layers"]}, {l["layer"]: l["warm"] for l in doc["layers"]},
                       doc["warm_bytes"])
        check(got == ({0: [1], 1: [1]}, {0: [3], 1: [3]}, 180224),
              f"--ram-budget-gib after one VRAM expert per layer: VRAM, warm lists and warm bytes {got}", out)
        rc, out, doc = packed("--budget-gib", "0", "--ram-budget-gib", "1")
        got = doc and ({l["layer"]: l["experts"] for l in doc["layers"]}, {l["layer"]: l["warm"] for l in doc["layers"]})
        check(got == ({0: [], 1: []}, {0: [1, 3, 0, 2], 1: [1, 3, 0, 2]}),
              f"--budget-gib 0 --ram-budget-gib 1: every expert warm in count order {got}", out)
        rc, out, doc = packed("--budget-gib", "1")
        check(doc is not None and "ram_budget_gib" not in doc and all("warm" not in l for l in doc["layers"]),
              "without --ram-budget-gib: no warm lists", out)

        # counts files (moe_counts.h): layer -> (decode, batch) counts of the 4 experts
        def counts_file(name, layers, n_expert=N_EXPERT, fmt="strata-moe-counts/1"):
            path = os.path.join(tmp, name)
            rows = [f' {{"layer": {layer}, "n_expert": {n_expert}, "decode_calls": {sum(d)}, "batch_calls": 1, '
                    f'"batch_tokens": {sum(b) // N_USED}, "decode": {json.dumps(d)}, "batch": {json.dumps(b)}}}'
                    for layer, (d, b) in sorted(layers.items())]
            with open(path, "w") as f:
                f.write(f'{{"format": "{fmt}", "layers": [\n' + ",\n".join(rows) + "]}\n")
            return path

        one_vram = repr(72108 / 2**30)  # one expert of one layer and its slab allowance
        a = counts_file("a.json", {0: ([0, 1, 5, 0], [9, 0, 0, 0]), 1: ([2, 0, 0, 1], [0, 0, 0, 9])})
        b = counts_file("b.json", {0: ([0, 4, 0, 0], [0, 0, 0, 0]), 1: ([0, 0, 0, 3], [0, 0, 0, 0])})

        def from_counts(*extra):
            """hot_set.py without the dump: (rc, output, the hot-set doc or None)"""
            out_json = os.path.join(tmp, "counts-hot.json")
            rc, out = hot_set("--gguf", first, *extra, "-o", out_json)
            if rc != 0:
                return rc, out, None
            with open(out_json) as f:
                return rc, out, json.load(f)

        def experts_of(*extra):
            rc, out, doc = from_counts(*extra)
            if doc is None:
                return rc, out, None
            return rc, out, {l["layer"]: (l["experts"], l.get("warm")) for l in doc["layers"]}

        rc, out, got = experts_of("--counts", a, "--budget-gib", one_vram)
        check(got == {0: ([2], None), 1: ([], None)},
              f"--counts a: decode counts only, layer 0 expert 2 (5) takes the one VRAM slot {got}", out)
        rc, out, got = experts_of("--counts", a, "--counts", b, "--budget-gib", one_vram, "--ram-budget-gib",
                                  repr(2 * 90112 / 2**30))
        check(got == {0: ([1], [2]), 1: ([], [3])},
              f"--counts a --counts b merged (layer 0 experts 1 and 2 both 5, 1 first by rank; then layer 1 expert "
              f"3 at 1 + 3 = 4): VRAM and the 2 warm slots {got}", out)
        rc, out, got = experts_of("--counts", a, "--counts-calls", "all", "--budget-gib", one_vram)
        check(got == {0: ([], None), 1: ([3], None)},
              f"--counts-calls all: decode + batch, layer 1 expert 3 (1 + 9) over layer 0 expert 0 (0 + 9) {got}", out)
        rc2, out2, doc2 = from_counts("--counts", a, "--counts-calls", "all", "--budget-gib", "1")
        hits = doc2 and {l["layer"]: l["activations"] for l in doc2["layers"]}
        check(hits == {0: 15, 1: 12}, f"--counts-calls all: activations per layer decode + batch {hits}", out2)
        rc, out, doc = from_counts("--dump", dump, "--counts", a, "--budget-gib", "1")
        act = doc and {l["layer"]: l["activations"] for l in doc["layers"]}
        check(act == {0: 6 + 6, 1: 6 + 3} and doc["counts_calls"] == "decode",
              f"--dump and --counts sum: activations {act}", out)
        bad = counts_file("bad-format.json", {0: ([1, 0, 0, 0], [0, 0, 0, 0])}, fmt="strata-moe-counts/0")
        rc, out = hot_set("--counts", bad, "--gguf", first, "--budget-gib", "1", "-o", os.path.join(tmp, "x.json"))
        check(rc == 1 and "format 'strata-moe-counts/0', expected 'strata-moe-counts/1'" in out,
              "a counts file of another format fails naming it", out)
        wide = counts_file("wide.json", {0: ([1, 0, 0, 0, 0, 0], [0] * 6)}, n_expert=6)
        rc, out = hot_set("--counts", a, "--counts", wide, "--gguf", first, "--budget-gib", "1", "-o",
                          os.path.join(tmp, "x.json"))
        check(rc == 1 and "layer 0 has 6 experts, an earlier counts file 4" in out,
              "counts files that disagree on a layer's experts fail", out)
        rc, out = hot_set("--counts", wide, "--gguf", first, "--budget-gib", "1", "-o", os.path.join(tmp, "x.json"))
        check(rc == 1 and "layer 0: 6 experts in the counts files, 4 in" in out,
              "a counts file that disagrees with the model fails", out)
        idle = counts_file("idle.json", {0: ([0, 0, 0, 0], [3, 0, 0, 0])})
        rc, out = hot_set("--counts", idle, "--gguf", first, "--budget-gib", "1", "-o", os.path.join(tmp, "x.json"))
        check(rc == 1 and "layer 0 has no activations in the dumps and counts files (--counts-calls decode)" in out,
              "a layer without decode activations fails naming it", out)

        # --propose-ram-budget with an A100 SXM pod's numbers
        def propose(*extra):
            return hot_set("--propose-ram-budget", *extra)

        c5 = ("--cgroup-limit-bytes", "124999999488", "--working-set-gib", "4.5", "--margin-gib", "8")
        rc, out = propose(*c5)
        check(rc == 0 and out.rstrip().endswith("proposed_ram_budget_gib 103"),
              f"--propose-ram-budget, c5's 116.415 GiB cgroup, 4.5 GiB working set, 8 GiB margin: 103 GiB\n{out}", out)
        rc, out = propose(*c5, "--numa-free-kib", str(int(86.11 * 2**20)))
        check(rc == 0 and out.rstrip().endswith("proposed_ram_budget_gib 78"),
              f"--propose-ram-budget with c5's GPU node MemFree 86.11 GiB: 78 GiB\n{out}", out)
        rc, out = propose("--cgroup-limit-bytes", str(12 * 2**30), "--working-set-gib", "4.5", "--margin-gib", "8")
        check(rc == 1 and "below 1 GiB, no room for a warm tier" in out, "a 12 GiB limit has no room: fails", out)
        rc, out = propose("--cgroup-limit-bytes", "124999999488")
        check(rc == 2 and "needs --working-set-gib, --margin-gib" in out, "missing inputs fail naming them", out)

    print(f"hot_set_test: {'FAIL' if failed else 'PASS'} ({failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
