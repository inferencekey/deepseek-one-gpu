#!/usr/bin/env python3
"""The real model from Hugging Face for a pod without the model's network volume (tools/dsv41/pod/e07.sh), python3
standard library only.

  hf_model.py manifest --repo R --revision REV --pattern GLOB --count N --out FILE
      the files of R at REV (the HF API's tree; REV resolved to its full commit) whose names match GLOB, exactly N of
      them, each with its size and its LFS sha256, and its resolve URL pinned to that commit, as JSON
  hf_model.py urls MANIFEST
      one "URL NAME SIZE" line per file (pod/hf-download.sh's input)
  hf_model.py headers MANIFEST DIR
      the head of every shard, HEAD_DIR/head-N.bin as engine/gguf_layout.py sparse reads it (the first 64 MiB of
      shard 1, 8 MiB of the others, SPEC.md "Real model header"), by HTTP range requests: the gate's model headers
      before the download
  hf_model.py check MANIFEST DIR [--json FILE]
      every file in DIR against the manifest: its size, then its sha256 (the files hashed in parallel); exits 1 on a
      missing file or a mismatch

HF_TOKEN (the environment, else ~/.hf-token, where e07.sh puts it on the pod) is optional: the repo is public. It is
sent only to huggingface.co (an unredirected header: never to the storage a resolve URL redirects to) and never
printed. Every request sends a User-Agent.
"""

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HF = "https://huggingface.co"
UA = "strata-dsv41-e07"
MIB = 1 << 20
# gguf_layout.py sparse's heads: shard 1 carries the tokenizer's metadata, the others only their tensor lists
HEAD_BYTES = (64 * MIB, 8 * MIB)
HASH_BLOCK = 16 * MIB


def fail(msg):
    print(f"hf_model.py: {msg}", file=sys.stderr)
    sys.exit(1)


def token():
    t = os.environ.get("HF_TOKEN", "")
    if not t:
        try:
            with open(os.path.expanduser("~/.hf-token")) as f:
                t = f.read().strip()
        except OSError:
            pass
    return t


def open_url(url, byte_range=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if byte_range:
        req.add_header("Range", "bytes=%d-%d" % byte_range)
    t = token()
    if t:
        req.add_unredirected_header("Authorization", "Bearer " + t)
    try:
        return urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        fail(f"{url}: HTTP {e.code}")
    except urllib.error.URLError as e:
        fail(f"{url}: {e.reason}")


def api(path):
    with open_url(HF + "/api/models/" + path) as r:
        return json.load(r)


def manifest(a):
    sha = api(f"{a.repo}/revision/{a.revision}").get("sha") or fail(f"{a.repo}@{a.revision}: no commit")
    if not sha.startswith(a.revision):
        fail(f"{a.repo}@{a.revision} resolves to {sha}")
    files = [f for f in api(f"{a.repo}/tree/{sha}?recursive=true")
             if f.get("type") == "file" and fnmatch.fnmatch(f["path"], a.pattern)]
    if len(files) != a.count:
        fail(f"{a.repo}@{sha}: {len(files)} files match {a.pattern}, expected {a.count}")
    out = {"repo": a.repo, "revision": sha, "files": []}
    for f in sorted(files, key=lambda f: f["path"]):
        lfs = f.get("lfs") or fail(f"{f['path']}: not an LFS file, no sha256")
        if lfs["size"] != f["size"]:
            fail(f"{f['path']}: size {f['size']}, its LFS pointer says {lfs['size']}")
        out["files"].append({"name": f["path"], "size": f["size"], "sha256": lfs["oid"],
                             "url": f"{HF}/{a.repo}/resolve/{sha}/{urllib.parse.quote(f['path'])}"})
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    total = sum(f["size"] for f in out["files"])
    print(f"hf_model.py: {a.repo}@{sha}: {len(files)} files, {total} bytes ({total / (1 << 30):.1f} GiB) in {a.out}")
    return 0


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        fail(f"{path}: {e}")


def urls(a):
    for f in load(a.manifest)["files"]:
        print(f["url"], f["name"], f["size"])
    return 0


def headers(a):
    os.makedirs(a.dir, exist_ok=True)
    for i, f in enumerate(load(a.manifest)["files"]):
        n = min(HEAD_BYTES[0 if i == 0 else 1], f["size"])
        with open_url(f["url"], (0, n - 1)) as r:
            data = r.read()
        if len(data) != n:
            fail(f"{f['name']}: the range 0-{n - 1} gave {len(data)} bytes")
        with open(os.path.join(a.dir, f"head-{i + 1}.bin"), "wb") as fh:
            fh.write(data)
        print(f"{a.dir}/head-{i + 1}.bin: the first {n} bytes of {f['name']}")
    return 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(HASH_BLOCK):
            h.update(block)
    return h.hexdigest()


def check(a):
    files, t0 = load(a.manifest)["files"], time.time()
    bad, rows = 0, []
    for f in files:
        p = os.path.join(a.dir, f["name"])
        size = os.path.getsize(p) if os.path.exists(p) else None
        if size != f["size"]:
            print(f"FAIL {p}: {size} bytes, the HF API says {f['size']}")
            bad += 1
    if not bad:
        # hashlib releases the GIL on large blocks: one thread per file reads them all at once
        with concurrent.futures.ThreadPoolExecutor(len(files)) as ex:
            got = list(ex.map(lambda f: sha256(os.path.join(a.dir, f["name"])), files))
        for f, g in zip(files, got):
            ok = g == f["sha256"]
            bad += not ok
            rows.append({"name": f["name"], "size": f["size"], "sha256": g, "ok": ok})
            print(f"{'ok  ' if ok else 'FAIL'} {f['name']}: {f['size']} bytes, sha256 {g}"
                  f"{'' if ok else ', the HF API says ' + f['sha256']}")
    secs = time.time() - t0
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"files": rows, "check_s": round(secs, 1), "ok": not bad}, fh, indent=1)
    print(f"hf_model.py: {len(files) - bad} of {len(files)} files of the HF API's size and sha256 ({secs:.0f} s)")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("manifest")
    s.add_argument("--repo", required=True)
    s.add_argument("--revision", required=True)
    s.add_argument("--pattern", required=True)
    s.add_argument("--count", type=int, required=True)
    s.add_argument("--out", required=True)
    sub.add_parser("urls").add_argument("manifest")
    s = sub.add_parser("headers")
    s.add_argument("manifest")
    s.add_argument("dir")
    s = sub.add_parser("check")
    s.add_argument("manifest")
    s.add_argument("dir")
    s.add_argument("--json", default="")
    a = p.parse_args()
    return {"manifest": manifest, "urls": urls, "headers": headers, "check": check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
