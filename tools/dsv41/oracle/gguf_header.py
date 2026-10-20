"""GGUF headers for the dsv41 tools, with python3 only (no numpy, no gguf-py): read the metadata and tensor infos of
one file, write a file from them.

  read_header(path)              -> Header (version, metadata, tensors, alignment, data_start)
  write_gguf(path, header, data) -> writes header.metadata and header.tensors, each tensor's bytes from data(name)
  integer_value(header, key)     -> the value of an integer key of any GGUF integer type, None when absent

The layout is GGUF v3 as ggml/src/gguf.cpp at the fork's 5210c7c5 reads and writes it: little endian, tensor data
offsets relative to the data section, every tensor padded to general.alignment (default 32).
"""

import struct

MAGIC = b"GGUF"
# GGUF value type id -> struct format of one element; 8 (string) and 9 (array) are read on their own
VALUE_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9
INTEGER_TYPES = (0, 1, 2, 3, 4, 5, 10, 11)  # UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64
FLOAT32 = 6
DEFAULT_ALIGNMENT = 32

# ggml type id -> (ggml_type_name, elements per block, bytes per block), ggml.h / ggml.c at the fork's 5210c7c5
GGML_TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20), 6: ("q5_0", 32, 22),
    7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74),
    18: ("iq3_xxs", 256, 98), 19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136), 29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2),
    39: ("mxfp4", 32, 17),
}


class GgufError(Exception):
    pass


class Header:
    """metadata: key -> (value type, value, element type of an array or None), in file order;
    tensors: list of {"name", "ne", "type", "offset"} in file order."""

    def __init__(self, version, metadata, tensors, alignment, data_start):
        self.version = version
        self.metadata = metadata
        self.tensors = tensors
        self.alignment = alignment
        self.data_start = data_start


def tensor_nbytes(ggml_type, ne):
    if ggml_type not in GGML_TYPES:
        raise GgufError(f"ggml type id {ggml_type} is not in this tool's table")
    name, blck, size = GGML_TYPES[ggml_type]
    if ne[0] % blck:
        raise GgufError(f"row of {ne[0]} elements is not whole {name} blocks of {blck}")
    n = ne[0] // blck * size
    for d in ne[1:]:
        n *= d
    return n


def _pad(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def read_header(path):
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize("<" + fmt)
            b = f.read(n)
            if len(b) != n:
                raise GgufError(f"{path}: truncated header")
            return struct.unpack("<" + fmt, b)

        def rd_str():
            (n,) = rd("Q")
            return f.read(n).decode("utf-8", "replace")

        def rd_value(t):
            if t == STRING:
                return rd_str()
            if t not in VALUE_FORMATS:
                raise GgufError(f"{path}: unknown GGUF value type {t}")
            return rd(VALUE_FORMATS[t])[0]

        if f.read(4) != MAGIC:
            raise GgufError(f"{path}: not a GGUF file")
        (version,) = rd("I")
        if version != 3:
            raise GgufError(f"{path}: GGUF version {version}, this tool reads version 3")
        n_tensors, n_kv = rd("QQ")
        metadata = {}
        for _ in range(n_kv):
            key = rd_str()
            (t,) = rd("I")
            if t == ARRAY:
                (et,) = rd("I")
                (n,) = rd("Q")
                if et == STRING:
                    value = [rd_str() for _ in range(n)]
                elif et in VALUE_FORMATS:
                    value = list(rd(VALUE_FORMATS[et] * n)) if n else []
                else:
                    raise GgufError(f"{path}: {key}: unknown array element type {et}")
                metadata[key] = (t, value, et)
            else:
                metadata[key] = (t, rd_value(t), None)
        tensors = []
        for _ in range(n_tensors):
            name = rd_str()
            (n_dims,) = rd("I")
            ne = list(rd("Q" * n_dims))
            (t,) = rd("I")
            (offset,) = rd("Q")
            tensors.append({"name": name, "ne": ne, "type": t, "offset": offset})
        alignment = DEFAULT_ALIGNMENT
        if "general.alignment" in metadata:
            alignment = integer_value_of(path, "general.alignment", metadata["general.alignment"])
        data_start = _pad(f.tell(), alignment)
    return Header(version, metadata, tensors, alignment, data_start)


def integer_value_of(path, key, entry):
    t, value, _ = entry
    if t not in INTEGER_TYPES:
        raise GgufError(f"{path}: {key} has GGUF value type {t}, expected an integer type")
    return value


def integer_value(path, header, key):
    entry = header.metadata.get(key)
    return None if entry is None else integer_value_of(path, key, entry)


def write_gguf(path, header, data):
    """Writes header.metadata and header.tensors (offsets recomputed) and, per tensor, data(name): its bytes."""
    out = bytearray()

    def put(fmt, *v):
        out.extend(struct.pack("<" + fmt, *v))

    def put_str(s):
        b = s.encode("utf-8")
        put("Q", len(b))
        out.extend(b)

    def put_value(t, v):
        if t == STRING:
            put_str(v)
        else:
            put(VALUE_FORMATS[t], v)

    out.extend(MAGIC)
    put("I", 3)
    put("QQ", len(header.tensors), len(header.metadata))
    for key, (t, value, et) in header.metadata.items():
        put_str(key)
        put("I", t)
        if t == ARRAY:
            put("I", et)
            put("Q", len(value))
            for v in value:
                put_value(et, v)
        else:
            put_value(t, value)
    offset = 0
    blobs = []
    for ti in header.tensors:
        blob = data(ti["name"])
        if len(blob) != tensor_nbytes(ti["type"], ti["ne"]):
            raise GgufError(f"{ti['name']}: {len(blob)} bytes, its type and shape say {tensor_nbytes(ti['type'], ti['ne'])}")
        put_str(ti["name"])
        put("I", len(ti["ne"]))
        put("Q" * len(ti["ne"]), *ti["ne"])
        put("I", ti["type"])
        put("Q", offset)
        blobs.append(blob)
        offset = _pad(offset + len(blob), header.alignment)
    out.extend(b"\0" * (_pad(len(out), header.alignment) - len(out)))
    with open(path, "wb") as f:
        f.write(out)
        for blob in blobs:
            f.write(blob)
            f.write(b"\0" * (_pad(len(blob), header.alignment) - len(blob)))
