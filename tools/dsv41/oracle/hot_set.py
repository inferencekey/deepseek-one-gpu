#!/usr/bin/env python3
"""Builds the hot set strata-moe keeps in VRAM (STRATA_MOE_HOT, include/strata/dsv41/moe.h) from dump directories.

  hot_set.py (--dump DIR | --counts FILE) [--dump DIR | --counts FILE ...] [--counts-calls decode|all]
             (--gguf MODEL | --layout FILE) --budget-gib GIB [--layers-full A-B] [--ram-budget-gib GIB] -o OUT.json
  hot_set.py --propose-ram-budget --cgroup-limit-bytes N --working-set-gib GIB --margin-gib GIB
             [--numa-free-kib KIB]

Counts the activations of every (layer, expert) in the topk files of the dumps (dsv41-moe-dump, see README.md)
and in strata-moe's counts files (STRATA_MOE_COUNTS, src/dsv41/moe_counts.h: a serving profile; their decode
calls, or with --counts-calls all their batch calls too), summed over every source given, then takes experts in
descending count until the budget is full: a global greedy over all layers, so each layer
gets the top N_layer of its own counts and the hit rate on the dumps is the highest the budget allows. Ties go to
the expert with the better rank in its own layer, then to the lower layer, so experts never seen in the dumps
fill the layers round-robin. With --layers-full A-B it takes instead every expert of layers A to B, in expert id
order, and none of the other layers: the placement of a stock run that keeps those layers' experts on the GPU
(read_dump.py --expert-placement prints it for a dump), for parity against it; the budget must hold them. Each
expert's byte size per layer comes from the GGUF's tensor headers (--gguf, all shards) or from a layout file
(--layout, for a model that is not on this machine). Prints the experts and the hit rate per layer and writes the
json strata-moe reads.

With --ram-budget-gib it also packs the warm tier (STRATA_MOE_RAM_GB): the same global greedy over the experts
the VRAM set left, next-most-used first, into a warm list per layer. A warm expert costs what strata-moe's warm
buffer gives it: per part (up, gate, down), its bytes rounded up to 4096 plus one 4096-byte block, the slot an
O_DIRECT read at any offset fills (src/dsv41/moe_residency.cpp). Every other expert is cold, read from the GGUF.

--propose-ram-budget sizes the warm tier from the memory it can take without starving the rest of the
process: the cgroup's limit minus the working set of the dense part and the engram rows (measured on a pod, given
in GiB) minus a margin, and, with --numa-free-kib (the MemFree of the GPU's NUMA node, where the pinned tier should
land), at most that minus the same margin. It prints the arithmetic and a last line "proposed_ram_budget_gib N",
N in whole GiB rounded down, and fails when N would be below 1. With STRATA_MOE_COLD=pagecache the page cache gets
what is left; nothing else is sized here.

The hit rate is measured on the same dumps the counts come from (in-sample), so it is an upper bound for other
prompts. Needs python3 only.
"""

import argparse
import array
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_header import GGML_TYPES, GgufError, integer_value, read_header  # noqa: E402

FORMAT = "strata-moe-hot-set/1"
COUNTS_FORMAT = "strata-moe-counts/1"
GIB = 1 << 30
KINDS = ("up", "gate", "down")

TYPE_BY_NAME = {name: (blck, size) for name, blck, size in GGML_TYPES.values()}

# strata-moe allocates every slab tensor with ggml-cuda's buffer type, which aligns each tensor to 128 bytes and
# pads a quantized tensor whose rows are not a multiple of 512 elements (MATRIX_ROW_PADDING) by up to 512 elements:
# the packing reserves both in full per tensor, so a hot set within the budget here is within it in strata-moe too
CUDA_ALIGN = 128
CUDA_ROW_PADDING = 512
# the warm tier's slot of one expert part: whole 4096-byte blocks plus one, as moe_residency.cpp sizes it
DIRECT_ALIGN = 4096


class HotSetError(Exception):
    pass


def row_bytes(type_name, n):
    blck, size = TYPE_BY_NAME[type_name]
    if n % blck:
        raise HotSetError(f"row of {n} elements is not whole {type_name} blocks of {blck}")
    return n // blck * size


class Layout:
    """Per layer: n_expert, n_embd, n_ff and the up/gate/down types of the routed expert tensors."""

    def __init__(self, layers, source):
        self.layers = layers  # layer -> dict
        self.source = source

    def expert_bytes(self, layer):
        l = self.layers[layer]
        return (row_bytes(l["types"]["up"], l["n_embd"]) * l["n_ff"] + row_bytes(l["types"]["gate"], l["n_embd"]) * l["n_ff"]
                + row_bytes(l["types"]["down"], l["n_ff"]) * l["n_embd"])

    def warm_slot_bytes(self, layer):
        """The RAM strata-moe's warm tier takes for one expert of the layer (DIRECT_ALIGN)."""
        l = self.layers[layer]
        parts = (row_bytes(l["types"]["up"], l["n_embd"]) * l["n_ff"], row_bytes(l["types"]["gate"], l["n_embd"]) * l["n_ff"],
                 row_bytes(l["types"]["down"], l["n_ff"]) * l["n_embd"])
        return sum(-(-p // DIRECT_ALIGN) * DIRECT_ALIGN + DIRECT_ALIGN for p in parts)

    def slab_allowance(self, layer):
        """The bytes ggml-cuda may add to the layer's three slab tensors (see CUDA_ROW_PADDING)."""
        l = self.layers[layer]
        rows = {"up": l["n_embd"], "gate": l["n_embd"], "down": l["n_ff"]}
        return sum(CUDA_ALIGN + (row_bytes(l["types"][k], CUDA_ROW_PADDING) if rows[k] % CUDA_ROW_PADDING else 0)
                   for k in KINDS)


def read_gguf_tensors(path):
    """name -> (ggml type id, ne) of every tensor in one GGUF file, and its split.count (1 when absent)."""
    try:
        header = read_header(path)
        split_count = integer_value(path, header, "split.count")
    except GgufError as e:
        raise HotSetError(str(e))
    return {t["name"]: (t["type"], t["ne"]) for t in header.tensors}, 1 if split_count is None else split_count


def layout_from_gguf(path):
    tensors, split_count = read_gguf_tensors(path)
    if split_count > 1:
        m = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", path)
        if not m or int(m.group(2)) != split_count:
            raise HotSetError(f"{path}: split.count {split_count} but not named <prefix>-00001-of-{split_count:05d}.gguf")
        for i in range(2, split_count + 1):
            tensors.update(read_gguf_tensors(f"{m.group(1)}-{i:05d}-of-{split_count:05d}.gguf")[0])
    layers = {}
    for name, (t, ne) in tensors.items():
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight", name)
        if not m:
            continue
        if t not in GGML_TYPES:
            raise HotSetError(f"{name}: ggml type id {t} is not in this tool's table")
        if len(ne) != 3:
            raise HotSetError(f"{name}: shape {ne}, expected [n_per_row, n_rows, n_expert]")
        l = layers.setdefault(int(m.group(1)), {"types": {}, "ne": {}})
        l["types"][m.group(2)] = GGML_TYPES[t][0]
        l["ne"][m.group(2)] = ne
    for layer, l in layers.items():
        if set(l["types"]) != set(KINDS):
            raise HotSetError(f"layer {layer}: expert tensors {sorted(l['types'])}, expected up, gate and down")
        up, gate, down = (l["ne"][k] for k in KINDS)
        if up != gate or down != [up[1], up[0], up[2]]:
            raise HotSetError(f"layer {layer}: up {up}, gate {gate}, down {down} do not agree")
        l.update(n_embd=up[0], n_ff=up[1], n_expert=up[2])
        del l["ne"]
    return Layout(layers, os.path.abspath(path))


def layout_from_file(path, dump_layers):
    """A layout file: {"n_expert", "n_embd", "n_ff", "types": {"up", "gate", "down"}} for every MoE layer."""
    with open(path) as f:
        d = json.load(f)
    try:
        l = {"n_expert": int(d["n_expert"]), "n_embd": int(d["n_embd"]), "n_ff": int(d["n_ff"]),
             "types": {k: d["types"][k] for k in KINDS}}
    except KeyError as e:
        raise HotSetError(f"{path}: missing {e}")
    for k in KINDS:
        if l["types"][k] not in TYPE_BY_NAME:
            raise HotSetError(f"{path}: {k} type {l['types'][k]!r} is not in this tool's table")
    return Layout({layer: l for layer in dump_layers}, os.path.abspath(path))


def count_activations(dump_dirs):
    """(layer -> list of counts per expert id seen so far), total activations per layer."""
    counts = {}
    for d in dump_dirs:
        with open(os.path.join(d, "dump-manifest.json")) as f:
            manifest = json.load(f)
        for e in manifest["files"]:
            if e["kind"] != "topk":
                continue
            if e["dtype"] != "i32":
                raise HotSetError(f"{d}/{e['file']}: dtype {e['dtype']}, expected i32")
            ids = array.array("i")
            with open(os.path.join(d, e["file"]), "rb") as f:
                ids.frombytes(f.read())
            n = e["ne"][0] * e["ne"][1]
            if len(ids) != n:
                raise HotSetError(f"{d}/{e['file']}: {len(ids)} ids, its shape {e['ne']} says {n}")
            c = counts.setdefault(e["layer"], {})
            for x in ids:
                c[x] = c.get(x, 0) + 1
    return counts


def add_counts_files(counts, paths, calls):
    """Adds strata-moe counts files (moe_counts.h) to `counts` (layer -> {expert id: count}): their decode counts,
    and with calls == "all" their batch counts too. Returns layer -> n_expert as the files give it."""
    n_expert = {}
    for path in paths:
        with open(path) as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError as e:
                raise HotSetError(f"{path}: not json ({e})")
        if d.get("format") != COUNTS_FORMAT:
            raise HotSetError(f"{path}: format {d.get('format')!r}, expected {COUNTS_FORMAT!r}")
        for l in d["layers"]:
            layer, n = int(l["layer"]), int(l["n_expert"])
            if n_expert.setdefault(layer, n) != n:
                raise HotSetError(f"{path}: layer {layer} has {n} experts, an earlier counts file {n_expert[layer]}")
            kinds = ("decode", "batch") if calls == "all" else ("decode",)
            c = counts.setdefault(layer, {})
            for kind in kinds:
                if len(l[kind]) != n:
                    raise HotSetError(f"{path}: layer {layer}: {len(l[kind])} {kind} counts for {n} experts")
                for e, x in enumerate(l[kind]):
                    if x < 0:
                        raise HotSetError(f"{path}: layer {layer} expert {e}: negative {kind} count {x}")
                    if x:
                        c[e] = c.get(e, 0) + x
    return n_expert


def propose_ram_budget(cgroup_limit_bytes, working_set_gib, margin_gib, numa_free_kib=None):
    """(proposed warm tier in whole GiB, lines of arithmetic); raises HotSetError below 1 GiB."""
    if cgroup_limit_bytes <= 0:
        raise HotSetError(f"--cgroup-limit-bytes {cgroup_limit_bytes}: expected the cgroup's limit, > 0")
    if working_set_gib < 0 or margin_gib < 0:
        raise HotSetError(f"--working-set-gib {working_set_gib:g}, --margin-gib {margin_gib:g}: expected >= 0")
    limit = cgroup_limit_bytes / GIB
    by_cgroup = limit - working_set_gib - margin_gib
    lines = [f"cgroup limit {limit:.3f} GiB - working set {working_set_gib:g} GiB - margin {margin_gib:g} GiB = "
             f"{by_cgroup:.3f} GiB"]
    budget = by_cgroup
    if numa_free_kib is not None:
        if numa_free_kib < 0:
            raise HotSetError(f"--numa-free-kib {numa_free_kib}: expected >= 0")
        by_node = numa_free_kib / (1 << 20) - margin_gib
        lines.append(f"GPU NUMA node MemFree {numa_free_kib / (1 << 20):.3f} GiB - margin {margin_gib:g} GiB = "
                     f"{by_node:.3f} GiB")
        budget = min(budget, by_node)
    proposed = int(budget // 1) if budget >= 1 else 0
    lines.append(f"warm tier: the smaller, rounded down to whole GiB: {proposed} GiB")
    if proposed < 1:
        raise HotSetError("; ".join(lines) + ": below 1 GiB, no room for a warm tier")
    return proposed, lines


def parse_layers(spec):
    m = re.fullmatch(r"(\d+)-(\d+)", spec)
    if not m or int(m.group(1)) > int(m.group(2)):
        raise HotSetError(f"--layers-full {spec!r}: expected A-B with A <= B")
    return range(int(m.group(1)), int(m.group(2)) + 1)


def pack_full(counts, layout, layers, budget_bytes):
    """Every expert of `layers`, in expert id order, none of the other layers."""
    chosen = {layer: [] for layer in counts}
    used = 0
    for layer in layers:
        if layer not in counts or layer not in layout.layers:
            raise HotSetError(f"--layers-full: layer {layer} is not a MoE layer of the dumps and {layout.source}")
        chosen[layer] = list(range(layout.layers[layer]["n_expert"]))
        used += len(chosen[layer]) * layout.expert_bytes(layer) + layout.slab_allowance(layer)
    if used > budget_bytes:
        raise HotSetError(f"--layers-full: layers {layers.start}-{layers.stop - 1} take {used / GIB:.3f} GiB, more "
                          f"than --budget-gib {budget_bytes / GIB:g}")
    return chosen, used


def pack(counts, layout, budget_bytes):
    """Global greedy: layer -> chosen experts in descending count, and the bytes they take."""
    candidates = []
    for layer in sorted(counts):
        if layer not in layout.layers:
            raise HotSetError(f"layer {layer} is in the dumps but has no routed expert tensors in {layout.source}")
        n_expert = layout.layers[layer]["n_expert"]
        bad = [e for e in counts[layer] if not 0 <= e < n_expert]
        if bad:
            raise HotSetError(f"layer {layer}: expert ids {bad[:5]} outside [0, {n_expert})")
        order = sorted(range(n_expert), key=lambda e: (-counts[layer].get(e, 0), e))
        candidates += [(-counts[layer].get(e, 0), rank, layer, e) for rank, e in enumerate(order)]
    candidates.sort()
    chosen = {layer: [] for layer in counts}
    used = 0
    for _, _, layer, e in candidates:
        cost = layout.expert_bytes(layer) + (0 if chosen[layer] else layout.slab_allowance(layer))
        if used + cost > budget_bytes:
            continue
        chosen[layer].append(e)
        used += cost
    return chosen, used


def pack_warm(counts, layout, vram, budget_bytes):
    """The warm tier: the same global greedy as pack over the experts not in `vram`, at the warm slot cost."""
    candidates = []
    for layer in sorted(counts):
        n_expert = layout.layers[layer]["n_expert"]
        taken = set(vram[layer])
        order = sorted(range(n_expert), key=lambda e: (-counts[layer].get(e, 0), e))
        candidates += [(-counts[layer].get(e, 0), rank, layer, e) for rank, e in enumerate(order) if e not in taken]
    candidates.sort()
    warm = {layer: [] for layer in counts}
    used = 0
    for _, _, layer, e in candidates:
        cost = layout.warm_slot_bytes(layer)
        if used + cost > budget_bytes:
            continue
        warm[layer].append(e)
        used += cost
    return warm, used


def main_propose(args):
    missing = [f"--{n.replace('_', '-')}" for n in ("cgroup_limit_bytes", "working_set_gib", "margin_gib")
               if getattr(args, n) is None]
    if missing:
        print(f"hot_set.py: --propose-ram-budget needs {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        proposed, lines = propose_ram_budget(args.cgroup_limit_bytes, args.working_set_gib, args.margin_gib,
                                             args.numa_free_kib)
    except HotSetError as e:
        print(f"hot_set.py: {e}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    print(f"proposed_ram_budget_gib {proposed}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="append", default=[], metavar="DIR", help="dsv41-moe-dump directory")
    ap.add_argument("--counts", action="append", default=[], metavar="FILE",
                    help="strata-moe counts file (STRATA_MOE_COUNTS)")
    ap.add_argument("--counts-calls", choices=("decode", "all"), default="decode",
                    help="which calls of the counts files count: decode (1-token calls, the default) or all")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--gguf", metavar="MODEL", help="the model's GGUF (first shard of a split model)")
    src.add_argument("--layout", metavar="FILE", help="layout json (n_expert, n_embd, n_ff, types) for every layer")
    ap.add_argument("--budget-gib", type=float, help="VRAM for the hot experts, GiB (2^30 bytes)")
    ap.add_argument("--layers-full", metavar="A-B", help="every expert of layers A to B and nothing else")
    ap.add_argument("--ram-budget-gib", type=float, metavar="GIB",
                    help="also pack the warm tier (STRATA_MOE_RAM_GB): RAM for the next-most-used experts, GiB")
    ap.add_argument("-o", "--output", help="hot-set json to write")
    ap.add_argument("--propose-ram-budget", action="store_true",
                    help="only print a warm tier size from the memory limits below (no dumps, no model)")
    ap.add_argument("--cgroup-limit-bytes", type=int, metavar="N", help="the memory cgroup's limit, bytes")
    ap.add_argument("--working-set-gib", type=float, metavar="GIB",
                    help="the dense part's and engram rows' resident memory, measured, GiB")
    ap.add_argument("--margin-gib", type=float, metavar="GIB", help="memory left unclaimed, GiB")
    ap.add_argument("--numa-free-kib", type=int, metavar="KIB", help="MemFree of the GPU's NUMA node, KiB")
    args = ap.parse_args()
    if args.propose_ram_budget:
        return main_propose(args)
    missing = [what for what, ok in (("--dump or --counts", args.dump or args.counts),
                                     ("--gguf or --layout", args.gguf or args.layout),
                                     ("--budget-gib", args.budget_gib is not None), ("-o", args.output)) if not ok]
    if missing:
        print(f"hot_set.py: needs {', '.join(missing)} (or --propose-ram-budget)", file=sys.stderr)
        return 2
    try:
        counts = count_activations(args.dump)
        if args.dump and not counts:
            raise HotSetError("the dumps have no topk files")
        counts_experts = add_counts_files(counts, args.counts, args.counts_calls)
        if not counts:
            raise HotSetError("the counts files have no layers")
        for layer in sorted(counts):
            if not sum(counts[layer].values()):
                raise HotSetError(f"layer {layer} has no activations in the dumps and counts files (--counts-calls "
                                  f"{args.counts_calls})")
        layout = layout_from_gguf(args.gguf) if args.gguf else layout_from_file(args.layout, counts.keys())
        for layer, n in counts_experts.items():
            if layer in layout.layers and layout.layers[layer]["n_expert"] != n:
                raise HotSetError(f"layer {layer}: {n} experts in the counts files, {layout.layers[layer]['n_expert']} "
                                  f"in {layout.source}")
        if args.layers_full:
            chosen, used = pack_full(counts, layout, parse_layers(args.layers_full), args.budget_gib * GIB)
        else:
            chosen, used = pack(counts, layout, args.budget_gib * GIB)
        if args.ram_budget_gib is not None:
            if args.ram_budget_gib < 0:
                raise HotSetError(f"--ram-budget-gib {args.ram_budget_gib:g}: expected >= 0")
            warm, warm_used = pack_warm(counts, layout, chosen, args.ram_budget_gib * GIB)
        else:
            warm, warm_used = None, 0
    except (HotSetError, OSError, ValueError) as e:
        print(f"hot_set.py: {e}", file=sys.stderr)
        return 1

    if args.dump:
        print(f"dumps:   {', '.join(args.dump)}")
    if args.counts:
        print(f"counts:  {', '.join(args.counts)} ({args.counts_calls} calls)")
    print(f"layout:  {layout.source}")
    print(f"budget:  {args.budget_gib:g} GiB, packed {used / GIB:.3f} GiB (with ggml-cuda's per-tensor padding)")
    if args.layers_full:
        print(f"layers:  every expert of layers {args.layers_full}, none of the others (--layers-full)")
    if warm is not None:
        print(f"warm:    {args.ram_budget_gib:g} GiB of warm-tier RAM, packed {warm_used / GIB:.3f} GiB (with the "
              f"4096-byte O_DIRECT slots)")
    print()
    warm_head = f"  {'warm':>5}  {'w hits':>6}  {'warm%':>6}" if warm is not None else ""
    print(f"{'layer':>5}  {'experts':>7}  {'of':>4}  {'MiB':>9}  {'activations':>11}  {'hits':>6}  {'hit%':>6}  "
          f"{'seen':>5}{warm_head}")
    out_layers = []
    tot_act = tot_hits = tot_warm_hits = 0
    for layer in sorted(chosen):
        l = layout.layers[layer]
        act = sum(counts[layer].values())
        hits = sum(counts[layer].get(e, 0) for e in chosen[layer])
        tot_act += act
        tot_hits += hits
        mib = len(chosen[layer]) * layout.expert_bytes(layer) / (1 << 20)
        entry = {"layer": layer, "n_expert": l["n_expert"], "n_embd": l["n_embd"], "n_ff": l["n_ff"],
                 "types": l["types"], "expert_bytes": layout.expert_bytes(layer), "activations": act,
                 "hits": hits, "experts": chosen[layer]}
        warm_cols = ""
        if warm is not None:
            warm_hits = sum(counts[layer].get(e, 0) for e in warm[layer])
            tot_warm_hits += warm_hits
            entry.update(warm=warm[layer], warm_hits=warm_hits)
            warm_cols = f"  {len(warm[layer]):>5}  {warm_hits:>6}  {100.0 * warm_hits / act:>6.1f}"
        print(f"{layer:>5}  {len(chosen[layer]):>7}  {l['n_expert']:>4}  {mib:>9.1f}  {act:>11}  {hits:>6}  "
              f"{100.0 * hits / act:>6.1f}  {len(counts[layer]):>5}{warm_cols}")
        out_layers.append(entry)
    n_hot = sum(len(v) for v in chosen.values())
    rate = tot_hits / tot_act
    print(f"(seen: distinct experts activated in the dumps and counts files)")
    print()
    print(f"{n_hot} experts in {sum(1 for v in chosen.values() if v)} of {len(chosen)} layers, {used / GIB:.3f} GiB; "
          f"expected hit rate on these sources {100.0 * rate:.1f}% ({tot_hits} of {tot_act} activations, in-sample)")

    doc = {"format": FORMAT, "budget_gib": args.budget_gib, "layers_full": args.layers_full, "bytes": used,
           "layout": layout.source,
           "dumps": [os.path.abspath(d) for d in args.dump], "counts": [os.path.abspath(c) for c in args.counts],
           "counts_calls": args.counts_calls if args.counts else None, "activations": tot_act, "hits": tot_hits,
           "expected_hit_rate": rate, "layers": out_layers}
    if warm is not None:
        n_warm = sum(len(v) for v in warm.values())
        warm_rate = tot_warm_hits / tot_act
        print(f"warm tier: {n_warm} experts, {warm_used / GIB:.3f} GiB; expected warm rate {100.0 * warm_rate:.1f}%, "
              f"cold {100.0 * (tot_act - tot_hits - tot_warm_hits) / tot_act:.1f}% (in-sample)")
        doc.update(ram_budget_gib=args.ram_budget_gib, warm_bytes=warm_used, warm_hits=tot_warm_hits,
                   expected_warm_rate=warm_rate)
    with open(args.output, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
