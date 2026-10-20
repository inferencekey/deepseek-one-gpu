#!/usr/bin/env python3
"""GGUF layouts for the engine's host-side dry runs, python3 only: the real model's file layout without its
weights, and a split copy of a tiny model.

  gguf_layout.py sparse HEAD_DIR OUT_DIR
      the 7 shards of DeepSeek-V4.1-Flash Q2_K (DeepSeek-V4.1-Flash-Q2_K-0000N-of-00007.gguf) from their headers
      (HEAD_DIR/head-N.bin: the first 64 MiB of shard 1 and 8 MiB of shards 2-7, SPEC.md "Real model header"): each
      file is its shard's header bytes, then a hole up to the shard's real size (a sparse file: the 246 GiB take the
      headers' 112 MiB on disk and read back as zeros), so dsv41-engine --dry-run loads the real metadata, tensor
      shapes, types and offsets across all 7 files. Each size is the end of the shard's last tensor, padded to the
      alignment as every tensor is, which must equal the shard's measured size (SHARD_BYTES), and each
      header's split.no must be its shard's.
  gguf_layout.py sizes DIR
      the 7 shards in DIR against SHARD_BYTES, byte for byte (the pod's local copy of the model); exits 1 on a
      missing shard or another size.
  gguf_layout.py split IN.gguf N OUT_DIR
      IN's tensors in N shards OUT_DIR/<stem>-0000i-of-0000N.gguf, as the fork's llama-gguf-split writes them: the
      first carries every metadata key, every shard split.no, split.count and split.tensors.count (the keys
      strata::GgufModel checks), the tensors in file order, about the same bytes per shard.

The dry runs of tools/dsv41/verify.sh then show that the engine reads a split model as it reads the whole one (the
same plan from the split tiny model) and that it takes the real layout (every view, tier and call of the real model
planned, no CUDA call).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
from gguf_header import GgufError, integer_value, read_header, tensor_nbytes, write_gguf  # noqa: E402

N_SHARDS = 7
SHARD_NAME = "DeepSeek-V4.1-Flash-Q2_K-{:05d}-of-00007.gguf"
# the shards' sizes in bytes, measured on a copy of vcruz305/DeepSeek-V4.1-Flash-GGUF Q2_K (revision c4a08554)
SHARD_BYTES = [43072989056, 44805480352, 14935107904, 44217904672, 44804598176, 44803713696, 27875485600]
UINT16, UINT32 = 2, 4


def fail(msg):
    print(f"gguf_layout.py: {msg}", file=sys.stderr)
    sys.exit(1)


def shard_end(h):
    """Where the shard ends: its last tensor's data padded to general.alignment, as the fork writes every tensor."""
    end = h.data_start + max(t["offset"] + tensor_nbytes(t["type"], t["ne"]) for t in h.tensors)
    return (end + h.alignment - 1) // h.alignment * h.alignment


def sparse(head_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_tensors = 0
    for i in range(N_SHARDS):
        head = os.path.join(head_dir, f"head-{i + 1}.bin")
        try:
            h = read_header(head)
        except (GgufError, OSError) as e:
            fail(str(e))
        no = integer_value(head, h, "split.no")
        if no != i:
            fail(f"{head}: split.no is {no}, expected {i}")
        if os.path.getsize(head) < h.data_start:
            fail(f"{head}: {os.path.getsize(head)} bytes, its header takes {h.data_start}")
        end = shard_end(h)
        if end != SHARD_BYTES[i]:
            fail(f"{head}: its tensors end at byte {end} (padded), the shard measures {SHARD_BYTES[i]}")
        out = os.path.join(out_dir, SHARD_NAME.format(i + 1))
        with open(head, "rb") as f:
            header = f.read(h.data_start)
        with open(out, "wb") as f:
            f.write(header)
            f.truncate(end)
        n_tensors += len(h.tensors)
        print(f"{out}: header {h.data_start} bytes, {len(h.tensors)} tensors, {end} bytes "
              f"({end / (1 << 30):.2f} GiB, sparse)")
    print(f"gguf_layout.py: {N_SHARDS} sparse shards, {n_tensors} tensors, {sum(SHARD_BYTES) / (1 << 30):.2f} GiB "
          f"in all, in {out_dir}")


def sizes(d):
    bad = 0
    for i, want in enumerate(SHARD_BYTES):
        p = os.path.join(d, SHARD_NAME.format(i + 1))
        got = os.path.getsize(p) if os.path.exists(p) else None
        bad += got != want
        print(f"{'ok  ' if got == want else 'FAIL'} {p}: {got} bytes, measured {want}")
    print(f"gguf_layout.py: {N_SHARDS - bad} of {N_SHARDS} shards of the measured size")
    return 1 if bad else 0


def split(path, n, out_dir):
    try:
        h = read_header(path)
    except (GgufError, OSError) as e:
        fail(str(e))
    if not 2 <= n <= len(h.tensors):
        fail(f"{path}: {len(h.tensors)} tensors cannot make {n} shards")
    with open(path, "rb") as f:
        def data(t):
            f.seek(h.data_start + t["offset"])
            return f.read(tensor_nbytes(t["type"], t["ne"]))
        blobs = {t["name"]: data(t) for t in h.tensors}
    # contiguous runs of tensors, each shard closing once it holds its share of the bytes
    total = sum(len(b) for b in blobs.values())
    groups, run, acc = [], [], 0
    for k, t in enumerate(h.tensors):
        run.append(t)
        acc += len(blobs[t["name"]])
        left_tensors, left_shards = len(h.tensors) - k - 1, n - len(groups) - 1
        if left_shards and (acc >= total * (len(groups) + 1) / n or left_tensors == left_shards):
            groups.append(run)
            run = []
    groups.append(run)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    meta, n_tensors = h.metadata, len(h.tensors)
    for i, g in enumerate(groups):
        keys = {k: v for k, v in meta.items() if not k.startswith("split.")} if i == 0 else {}
        keys["split.no"] = (UINT16, i, None)
        keys["split.count"] = (UINT16, n, None)
        keys["split.tensors.count"] = (UINT32, n_tensors, None)
        if "general.alignment" in meta:
            keys["general.alignment"] = meta["general.alignment"]
        h.metadata, h.tensors = keys, g
        out = os.path.join(out_dir, f"{stem}-{i + 1:05d}-of-{n:05d}.gguf")
        write_gguf(out, h, lambda name: blobs[name])
        print(f"{out}: {len(g)} tensors, {sum(len(blobs[t['name']]) for t in g)} bytes of data, {len(keys)} keys")


def main(argv):
    if len(argv) == 3 and argv[0] == "sparse":
        sparse(argv[1], argv[2])
        return 0
    if len(argv) == 2 and argv[0] == "sizes":
        return sizes(argv[1])
    if len(argv) == 4 and argv[0] == "split" and argv[2].isdigit():
        split(argv[1], int(argv[2]), argv[3])
        return 0
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
