#!/usr/bin/env python3
"""The real-model session's placement and memory gate (tools/dsv41/pod/e07-run.sh, and e07.sh --dry-run locally): the VRAM hot
set and the pinned warm tier of every engine run of the session, and whether the pod's memory cgroup holds them.

  e07_gate.py --vram-mib N --cgroup-bytes N|unlimited --mem-total-kib N --counts FILE --layout FILE
              --model-headers SHARD1 --out DIR [--nonexpert-gib G] [--margin-gib G] [--working-gib G] [--hot-spare N]
              [--prefetch-experts N] [--checkpoints N] [--ctx N] [--batch N [--batch-ctx N]]

  VRAM   the slab gets the GPU's memory less what the engine needs beside the routed experts at the session's
         largest configuration (--nonexpert-gib, default 20.22: chunk 16384 at ctx 262144, measured by
         dsv41_engine_smoke --memory-table on an A100 pod, 2026-10-08) and a margin (--margin-gib, default 2),
         rounded down to 0.25 GiB (the measurement rule): the engine's --vram-gb.
         tools/dsv41/oracle/hot_set.py fills it from the real counts profile (--counts, a real-model run's decode
         counts), most used experts first, less the adaptive hot set's spare slots (--hot-spare, default 2, the engine's: per layer that
         many experts' slots, each part 256-byte aligned as experts.cpp packs them), which the swaps of the
         per-request hot set copy into. --prefetch-experts N (default 32, the engine's, inside --nonexpert-gib's
         measurement) adds N - 32 expert slots of the prefetch region beside the experts (dsv41-engine
         --prefetch-experts N in every run of the session). --batch N live
         batch slots of --batch-ctx positions (default 32768, dsv41-engine's) take their device KV from the slab too
         (kv_budget.py slot_bytes, the engine's kv_cache_bytes of a slot: a ring of n_swa cells, the rows and keys of
         its positions; 105.1 MiB a slot of 32768 on the real model) plus the batch's rows' logits (N x n_vocab f32);
         the gate prints the live slots, their KV and what the slab loses (GiB and experts of a slot's size), and
         refuses a placement whose slab they leave without a hot set.
  RAM    every other expert goes to the warm tier (the hot set's warm lists; --ram-gb holds them all, so none is
         cold and nothing is read from the NVMe in steady state), pinned, plus the two engram tables (pinned, their
         size from the model's tensor headers, --model-headers: shard 1 of the split model, all 7 read), plus the
         spare homes (as many pinned expert slots per layer as spare slots, the victims' copies), plus
         the engine's conversation checkpoints, --checkpoints
         (default 4, dsv41-engine --serve's) slots pinned for a checkpoint of --ctx positions each (default 262144,
         every plan's; tools/dsv41/engine/kv_budget.py checkpoint_bytes, the engine's kv_checkpoint_bytes on the
         model's layer roles: 805.0 MiB a slot on the real model), plus
         --working-gib (default 12: the dense part's pages, staging, the CPU pool's buffers, the server and the
         tools) must fit under the memory cgroup's limit (or MemTotal when unlimited) with RAM_MARGIN_GIB (8) to
         spare: not a flag, so no GPU_TYPES override passes a pod with less (the L40S fitted 250 GB by 0.05 GiB). A
         placement that does not fit is refused naming every part's size (the checkpoints' count and slot size too).
  HOMES  the host homes, a permanent pinned home for every
         expert of the slab (its bytes as the slab packs them, the hot set's), so the swaps and the deterministic reset
         copy up only (dsv41-engine --hot-homes all). They replace the spare homes and are turned on only when the
         RAM above holds them too with RAM_MARGIN_GIB to spare; otherwise (partial host homes) as many
         of the slab's experts as that RAM holds with the same margin get one, in the engine's home order (experts.hpp:
         the layers that swap in layer order, every slab expert of a layer before the next layer's, the layer the RAM
         cuts through from the end of its VRAM order), the layers not homed in full keeping their spare homes and the
         two-way path (dsv41-engine --hot-homes N, the gate printing the homes held of the slab's experts); with room
         for none the placement keeps the two-way path (--hot-homes spare) and the gate says why, naming the homes'
         size and how much is missing. Not a verdict on the placement: a host without room for the homes still passes.
         --host-homes off keeps them off. Each home is the slab's slot of its layer (every part 256-byte aligned).
Writes DIR/hot-set.json and DIR/gate.json (every number, the engine's --vram-gb, --ram-gb and --hot-homes, the
verdict) and prints the arithmetic; exits 1 when the memory does not fit (the session stops before the model copy), 2
on a bad input.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "oracle"))
from gguf_header import GgufError, read_header, tensor_nbytes  # noqa: E402

GIB = 1 << 30
SLAB_ALIGN = 256   # experts.cpp: each part of a slab slot (and of a spare home) 256-byte aligned
# the host RAM left over beside everything the session pins and works in, at least
RAM_MARGIN_GIB = 8.0
# the prefetch region inside --nonexpert-gib's measurement: the engine's default (experts.hpp prefetch_experts)
PREFETCH_MEASURED = 32
# the batch rows' logits (a decode step's [rows][n_vocab] f32 on the device), the real model's vocabulary
N_VOCAB = 129280


def shards(shard1):
    """Every shard of the split model shard1 belongs to."""
    m = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", shard1)
    if not m:
        sys.exit(f"e07_gate.py: {shard1} is not shard 1 of a split model")
    return [f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)]


def checkpoint_slot_bytes(shard1, ctx):
    """one conversation checkpoint slot's pinned bytes, a checkpoint of ctx positions (kv_budget.py)."""
    sys.path.insert(0, os.path.join(here, "..", "engine"))
    import kv_budget  # noqa: E402
    try:
        return kv_budget.checkpoint_bytes(kv_budget.read_model(shards(shard1)), ctx)
    except (GgufError, OSError, KeyError) as e:
        sys.exit(f"e07_gate.py: {e}")


def batch_slot_bytes(shard1, positions):
    """one batch slot's device KV, a slot of `positions` (kv_budget.py slot_bytes, the engine's kv_cache_bytes)."""
    sys.path.insert(0, os.path.join(here, "..", "engine"))
    import kv_budget  # noqa: E402
    try:
        return kv_budget.slot_bytes(kv_budget.read_model(shards(shard1)), positions)
    except (GgufError, OSError, KeyError) as e:
        sys.exit(f"e07_gate.py: {e}")


def layer_slots(layout, layers):
    """Each layer's slab slot (an expert's three parts, each 256-byte aligned as experts.cpp packs them): a spare
    home's and a host home's bytes too."""
    from hot_set import layout_from_file, row_bytes  # noqa: E402
    lay = layout_from_file(layout, layers).layers
    slots = {}
    for l in layers:
        x, t = lay[l], lay[l]["types"]
        slots[l] = sum(-(-p // SLAB_ALIGN) * SLAB_ALIGN for p in (
            row_bytes(t["up"], x["n_embd"]) * x["n_ff"], row_bytes(t["gate"], x["n_embd"]) * x["n_ff"],
            row_bytes(t["down"], x["n_ff"]) * x["n_embd"]))
    return slots


def partial_homes(swapping, slots, spare, room):
    """the most slab experts whose homes fit in `room` bytes in the engine's home order (the layers
    that swap in order, each in full, the cut one from the end of its VRAM order), with the spare homes every layer not
    homed in full keeps (`spare` slots each). Returns the homes held, their bytes, those spare homes' bytes and the cut
    (layer, its experts homed, its slab experts) or None."""
    best = (0, 0, sum(spare * slots[l] for l, _ in swapping), None)
    full_bytes = 0
    for k, (l, n) in enumerate(swapping):
        rest_spare = sum(spare * slots[x] for x, _ in swapping[k + 1:])
        # layers before k in full, layer k cut (it keeps its spare homes), the rest none
        m = min(n - 1, int((room - full_bytes - spare * slots[l] - rest_spare) // slots[l]))
        if m < 0:
            break
        held = sum(x for _, x in swapping[:k]) + m
        if held > best[0]:
            best = (held, full_bytes + m * slots[l], spare * slots[l] + rest_spare, (l, m, n) if m else None)
        # layer k in full, if that fits with the rest's spare homes
        if full_bytes + n * slots[l] + rest_spare > room:
            break
        full_bytes += n * slots[l]
    return best


def engram_bytes(shard1):
    """The engram_embd tables' bytes over every shard of the split model shard1 belongs to."""
    total, found = 0, 0
    for path in shards(shard1):
        try:
            h = read_header(path)
        except (GgufError, OSError) as e:
            sys.exit(f"e07_gate.py: {e}")
        for t in h.tensors:
            if t["name"].endswith(".engram_embd.weight"):
                total += tensor_nbytes(t["type"], t["ne"])
                found += 1
    if found != 2:
        sys.exit(f"e07_gate.py: {found} engram_embd tables in {shard1}'s shards, the model has 2")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--vram-mib", type=int, required=True, help="the GPU's memory.total (nvidia-smi), MiB")
    ap.add_argument("--cgroup-bytes", required=True, help="the memory cgroup's limit, or unlimited")
    ap.add_argument("--mem-total-kib", type=int, required=True, help="/proc/meminfo MemTotal")
    ap.add_argument("--counts", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--model-headers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nonexpert-gib", type=float, default=20.22)
    ap.add_argument("--margin-gib", type=float, default=2.0)
    ap.add_argument("--working-gib", type=float, default=12.0)
    ap.add_argument("--hot-spare", type=int, default=2, help="spare slab slots per layer (dsv41-engine --hot-spare)")
    ap.add_argument("--prefetch-experts", type=int, default=PREFETCH_MEASURED,
                    help="the prefetch region's experts (dsv41-engine --prefetch-experts)")
    ap.add_argument("--checkpoints", type=int, default=4,
                    help="the engine's conversation checkpoints (dsv41-engine --serve --checkpoints, default 4)")
    ap.add_argument("--ctx", type=int, default=262144, help="the engine's --ctx, a checkpoint slot's positions")
    ap.add_argument("--batch", type=int, default=0, help="the engine's batch slots (dsv41-engine --batch, 0: none)")
    ap.add_argument("--batch-ctx", type=int, default=32768, help="the positions of a batch slot (--batch-ctx)")
    ap.add_argument("--host-homes", choices=("auto", "off"), default="auto",
                    help="auto turns the slab's host homes on when the RAM holds them (default)")
    a = ap.parse_args()
    if a.hot_spare < 0 or a.prefetch_experts < 0 or a.checkpoints < 0 or a.ctx < 1 or not 0 <= a.batch <= 8 or \
            not 1 <= a.batch_ctx <= a.ctx:
        print(f"e07_gate.py: --hot-spare {a.hot_spare}, --prefetch-experts {a.prefetch_experts}, --checkpoints "
              f"{a.checkpoints}, --ctx {a.ctx}, --batch {a.batch} (0..8), --batch-ctx {a.batch_ctx} (1..ctx)")
        return 2
    os.makedirs(a.out, exist_ok=True)
    vram_gib = a.vram_mib / 1024
    # the spare slots' and homes' bytes: per MoE layer of the counts profile (the layout's expert in every one)
    # one expert slot per spare; the prefetch region's slots past the measured 32
    from hot_set import HotSetError, layout_from_file, row_bytes  # noqa: E402
    try:
        with open(a.counts) as f:
            n_moe = len(json.load(f)["layers"])
        l = layout_from_file(a.layout, [0]).layers[0]
        t, slot = l["types"], 0
        for part in (row_bytes(t["up"], l["n_embd"]) * l["n_ff"], row_bytes(t["gate"], l["n_embd"]) * l["n_ff"],
                     row_bytes(t["down"], l["n_ff"]) * l["n_embd"]):
            slot += -(-part // SLAB_ALIGN) * SLAB_ALIGN
    except (HotSetError, OSError, KeyError, ValueError) as e:
        print(f"e07_gate.py: {a.layout}, {a.counts}: {e}")
        return 2
    nonexpert = round(a.nonexpert_gib + (a.prefetch_experts - PREFETCH_MEASURED) * slot / GIB, 3)
    # the live batch slots' KV and the batch rows' logits, beside the measured non-expert part
    batch_slot = batch_slot_bytes(a.model_headers, a.batch_ctx) if a.batch else 0
    batch_gib = (a.batch * batch_slot + a.batch * N_VOCAB * 4) / GIB if a.batch else 0.0
    nonexpert = round(nonexpert + batch_gib, 3)
    budget = math.floor((vram_gib - nonexpert - a.margin_gib) * 4) / 4
    batch_text = (f"; the batch's {a.batch} live slots of {a.batch_ctx} positions {a.batch} x {batch_slot / 2**20:.1f} MiB "
                  f"KV + their logits = {batch_gib:.2f} GiB of it" if a.batch else "")
    if budget <= 0:
        print(f"e07_gate.py: refused: {vram_gib:.2f} GiB of VRAM leave no slab beside {nonexpert} + {a.margin_gib} GiB"
              f"{batch_text}")
        return 1
    spare_gib = a.hot_spare * n_moe * slot / GIB
    hot_budget = math.floor((budget - spare_gib) * 4) / 4
    if hot_budget <= 0:
        print(f"e07_gate.py: refused: a slab of {budget} GiB leaves no hot set beside {spare_gib:.2f} GiB of spare slots"
              f"{batch_text}")
        return 1
    hot = os.path.join(a.out, "hot-set.json")
    # the warm tier's budget: every expert's (183.25 GiB in all, SPEC.md) and one more, so hot_set.py packs every
    # expert the slab left into the warm lists
    ram_gb = 190
    r = subprocess.run([sys.executable, os.path.join(here, "..", "oracle", "hot_set.py"), "--counts", a.counts,
                        "--layout", a.layout, "--budget-gib", str(hot_budget), "--ram-budget-gib", str(ram_gb), "-o", hot],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        return 2
    with open(hot) as f:
        h = json.load(f)
    n_vram = sum(len(l["experts"]) for l in h["layers"])
    n_warm = sum(len(l["warm"]) for l in h["layers"])
    n_all = sum(l["n_expert"] for l in h["layers"])
    warm_gib = h["warm_bytes"] / GIB
    eng_gib = engram_bytes(a.model_headers) / GIB
    slot_bytes = checkpoint_slot_bytes(a.model_headers, a.ctx)
    ckpt_gib = a.checkpoints * slot_bytes / GIB
    need = warm_gib + eng_gib + spare_gib + ckpt_gib + a.working_gib
    if a.cgroup_bytes == "unlimited":
        limit_gib, limit_src = a.mem_total_kib / (1 << 20), "MemTotal (the cgroup has no limit)"
    else:
        limit_gib, limit_src = int(a.cgroup_bytes) / GIB, "the memory cgroup's limit"
    fits = n_vram + n_warm == n_all and need + RAM_MARGIN_GIB <= limit_gib
    # the host homes in place of the spare homes, on when they fit too; else as many as
    # fit in the engine's home order, the layers not homed in full keeping their spare homes
    try:
        slots = layer_slots(a.layout, [x["layer"] for x in h["layers"]])
    except (HotSetError, OSError, KeyError, ValueError) as e:
        print(f"e07_gate.py: {a.layout}: {e}")
        return 2
    swapping = [(x["layer"], len(x["experts"])) for x in h["layers"]
                if a.hot_spare > 0 and 0 < len(x["experts"]) < x["n_expert"]]
    n_homeable = sum(n for _, n in swapping)
    homes_bytes = sum(n * slots[l] for l, n in swapping)
    homes_gib = homes_bytes / GIB
    need_homes = need - spare_gib + homes_gib
    homes_fit = fits and n_homeable > 0 and need_homes + RAM_MARGIN_GIB <= limit_gib
    n_homes, part_bytes, part_spare, cut = (n_homeable, homes_bytes, 0, None) if homes_fit else (0, 0, 0, None)
    if fits and not homes_fit and a.host_homes == "auto" and n_homeable > 0:
        n_homes, part_bytes, part_spare, cut = partial_homes(swapping, slots, a.hot_spare,
                                                             (limit_gib - RAM_MARGIN_GIB - (need - spare_gib)) * GIB)
    need_part = need - spare_gib + (part_bytes + part_spare) / GIB
    homes_on = n_homes > 0 and a.host_homes == "auto"
    partial = homes_on and n_homes < n_homeable
    short = (f"{need_homes:.2f} GiB + {RAM_MARGIN_GIB} margin against {limit_gib:.2f} GiB, "
             f"{need_homes + RAM_MARGIN_GIB - limit_gib:.2f} GiB short")
    if homes_on and not partial:
        homes_why = "the RAM holds them with the margin"
    elif partial:
        homes_why = (f"the RAM holds {n_homes} of the slab's {n_homeable} experts' homes with the margin "
                     f"({100 * n_homes / n_homeable:.1f}%; all of them: {short})")
    elif a.host_homes == "off":
        homes_why = "--host-homes off"
    elif not fits:
        homes_why = "the placement does not fit"
    else:
        homes_why = f"the host RAM cannot hold them: {short}"
    hot_homes = "spare" if not homes_on else str(n_homes) if partial else "all"
    g = {"vram_total_gib": round(vram_gib, 2), "nonexpert_gib": nonexpert, "prefetch_experts": a.prefetch_experts,
         "margin_gib": a.margin_gib,
         "vram_gb": budget, "hot_spare": a.hot_spare, "spare_gib": round(spare_gib, 3), "hot_set_budget_gib": hot_budget,
         "ram_gb": ram_gb, "hot_set": hot, "vram_experts": n_vram, "warm_experts": n_warm,
         "cold_experts": n_all - n_vram - n_warm, "slab_gib": round(h["bytes"] / GIB, 3),
         "warm_gib": round(warm_gib, 3), "engram_gib": round(eng_gib, 3), "working_gib": a.working_gib,
         "checkpoints": a.checkpoints, "ctx": a.ctx, "checkpoint_slot_bytes": slot_bytes,
         "checkpoints_gib": round(ckpt_gib, 3),
         "pinned_gib": round(warm_gib + eng_gib + spare_gib + ckpt_gib, 3), "need_gib": round(need, 2),
         "host_homes_gib": round(homes_gib, 3), "host_homes_bytes": homes_bytes, "need_with_homes_gib": round(need_homes, 2), "host_homes": homes_on,
         "hot_homes": hot_homes, "host_homes_reason": homes_why,
         # the homes held (of the slab's experts that can take one), their bytes, the spare homes the
         # layers not homed in full keep, the RAM with them, the layer the RAM cut and how many of its experts it took
         "host_homes_held": n_homes if homes_on else 0, "host_homes_of": n_homeable,
         "host_homes_fraction": round(n_homes / n_homeable, 4) if homes_on and n_homeable else 0.0,
         "host_homes_held_bytes": part_bytes if homes_on else 0, "partial_spare_bytes": part_spare if partial else 0,
         "need_with_partial_homes_gib": round(need_part, 2) if partial else None,
         "host_homes_cut": cut if partial else None,
         "ram_margin_gib": RAM_MARGIN_GIB,
         "limit_gib": round(limit_gib, 2), "limit_source": limit_src, "expected_hit_rate": h["expected_hit_rate"],
         # the live batch slots, a slot's device KV, theirs with the rows' logits, and the slab they cost in experts
         "batch": a.batch, "batch_ctx": a.batch_ctx, "batch_slot_bytes": batch_slot, "batch_gib": round(batch_gib, 3),
         "batch_slab_experts_lost": int(batch_gib * GIB // slot) if a.batch else 0,
         "fits": fits}
    with open(os.path.join(a.out, "gate.json"), "w") as f:
        json.dump(g, f, indent=1)
    print(f"e07_gate.py: VRAM {vram_gib:.2f} GiB - {nonexpert} beside the experts (chunk 16384, ctx 262144, a prefetch "
          f"region of {a.prefetch_experts} experts) - "
          f"{a.margin_gib} margin = slab {budget} GiB (--vram-gb): the hot set {hot_budget} GiB beside "
          f"{a.hot_spare} spare slots per layer ({spare_gib:.2f} GiB, --hot-spare): {n_vram} experts, "
          f"{g['slab_gib']} GiB, expected decode hit rate {100 * h['expected_hit_rate']:.1f}% (in-sample)")
    print(f"e07_gate.py: warm tier {n_warm} experts, {warm_gib:.2f} GiB pinned (--ram-gb {ram_gb}), cold "
          f"{g['cold_experts']}; engram tables {eng_gib:.2f} GiB pinned; spare homes {spare_gib:.2f} GiB pinned; "
          f"conversation checkpoints {a.checkpoints} x {slot_bytes / GIB:.3f} GiB = {ckpt_gib:.2f} GiB pinned "
          f"(--checkpoints {a.checkpoints}, a checkpoint of {a.ctx} positions each); + {a.working_gib} GiB working = "
          f"{need:.2f} GiB + {RAM_MARGIN_GIB} margin against {limit_gib:.2f} GiB, {limit_src}: "
          f"{'PASS' if fits else 'FAIL'}")
    if a.batch:
        print(f"e07_gate.py: batch: {a.batch} live slots of {a.batch_ctx} positions, "
              f"{batch_slot / 2**20:.1f} MiB device KV each, {a.batch * batch_slot / 2**20:.1f} MiB in all + the rows' "
              f"logits {a.batch * N_VOCAB * 4 / 2**20:.1f} MiB = {batch_gib:.2f} GiB, taken from the slab (--batch "
              f"{a.batch} --batch-ctx {a.batch_ctx}): it loses about {g['batch_slab_experts_lost']} experts of "
              f"{slot / 2**20:.1f} MiB ({100 * g['batch_slab_experts_lost'] / max(n_vram, 1):.1f}% of the {n_vram} it holds)")
    if homes_on and not partial:
        print(f"e07_gate.py: host homes: the slab's {n_vram} experts' homes, "
              f"{homes_gib:.2f} GiB pinned in place of the spare homes' {spare_gib:.2f}: {need_homes:.2f} GiB + "
              f"{RAM_MARGIN_GIB} margin against {limit_gib:.2f} GiB: ON (--hot-homes all), the swaps and the "
              "deterministic reset copy up only")
    elif partial:
        full = sum(1 for l, n in swapping if cut is None or l < cut[0])
        cut_text = f", layer {cut[0]}'s last {cut[1]} of its {cut[2]}" if cut else ""
        print(f"e07_gate.py: host homes: PARTIAL (--hot-homes {n_homes}): {n_homes} of the "
              f"slab's {n_homeable} experts homed, {100 * n_homes / n_homeable:.1f}% (the first {full} of the "
              f"{len(swapping)} layers that swap in full{cut_text}), {part_bytes / GIB:.2f} GiB pinned, the other layers' "
              f"spare homes {part_spare / GIB:.2f}: {need_part:.2f} GiB + {RAM_MARGIN_GIB} margin against "
              f"{limit_gib:.2f} GiB; all {n_homeable} would need {homes_gib:.2f} GiB ({short}); the layers homed in "
              "full restore layer by layer, the others keep the two-way path")
    elif fits:
        print(f"e07_gate.py: host homes: refused, the two-way path (--hot-homes spare): "
              + (homes_why if a.host_homes == "off" else
                 f"the host RAM cannot hold the slab's {n_vram} experts' homes of {homes_gib:.2f} GiB in place of the "
                 f"spare homes' {spare_gib:.2f} beside the warm tier's {warm_gib:.2f}, the engram tables' "
                 f"{eng_gib:.2f}, the checkpoints' {ckpt_gib:.2f} and {a.working_gib} GiB working, with "
                 f"{RAM_MARGIN_GIB} GiB to spare, in {limit_gib:.2f} GiB: "
                 f"{need_homes + RAM_MARGIN_GIB - limit_gib:.2f} GiB short"))
    # a placement that fits only without the checkpoints is refused for them, by name
    if not fits and n_vram + n_warm == n_all and need - ckpt_gib + RAM_MARGIN_GIB <= limit_gib:
        print(f"e07_gate.py: refused: the host RAM cannot hold {a.checkpoints} conversation checkpoints of "
              f"{slot_bytes / GIB:.3f} GiB ({ckpt_gib:.2f} GiB) beside the warm tier's {warm_gib:.2f}, the engram "
              f"tables' {eng_gib:.2f}, the spare homes' {spare_gib:.2f} and {a.working_gib} GiB working, with "
              f"{RAM_MARGIN_GIB} GiB to spare, in {limit_gib:.2f} GiB")
    return 0 if fits else 1


if __name__ == "__main__":
    sys.exit(main())
