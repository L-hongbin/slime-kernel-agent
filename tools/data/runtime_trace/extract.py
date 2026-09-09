"""Convert bounded NVBit events into an auditable observed program graph.

No source parsing, task adapters or numerical equivalence assumptions. Register
semantics are a conservative SASS subset. Byte-level memory dependencies require
happens-before; atomic log order is never interpreted as cross-thread order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path

EVENT = struct.Struct("<QQIIIIII")
SCHEMA = "runtime_program_graph/v1"
REG = re.compile(r"\b(?:UR\d+|R\d+|UP\d+|P\d+)\b")
SIMPLE_DEST = {
    "MOV",
    "UMOV",
    "LDC",
    "ULDC",
    "LDG",
    "LDS",
    "LDL",
    "S2R",
    "S2UR",
    "FADD",
    "FMUL",
    "FFMA",
    "DADD",
    "DMUL",
    "DFMA",
    "IMAD",
    "IADD",
    "SHF",
    "SHL",
    "SHR",
    "LOP",
    "LOP3",
    "ULOP3",
    "F2F",
    "F2I",
    "I2F",
    "I2I",
    "FSEL",
    "SEL",
    "MUFU",
    "PRMT",
    "BREV",
    "FLO",
    "POPC",
    "LEA",
    "ULEA",
    "VIADD",
    "IADD3",
    "UIADD3",
    "UIMAD",
    "IABS",
    "I2FP",
    "HFMA2",
    "HADD2",
    "HMUL2",
}
NO_DEST = {"STG", "STS", "STL", "EXIT", "BRA", "NOP", "BAR", "MEMBAR", "BSSY", "BSYNC", "BREAK", "WARPSYNC", "DEPBAR"}


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registers(operand, wide=False):
    values = []
    for m in REG.finditer(operand["text"]):
        name = m.group()
        values.append(name)
        if name.startswith(("R", "UR")):
            count = max(1, operand["bytes"] // 4) if operand["type"] in ("REG", "UREG") else 1
            suffix = operand["text"][m.end() :]
            if suffix.startswith(".128"):
                count = max(count, 4)
            elif suffix.startswith(".64"):
                count = max(count, 2)
            if wide:
                count = max(count, 2)
            prefix = re.match(r"[A-Z]+", name).group()
            number = int(name[len(prefix) :])
            values.extend(f"{prefix}{number+i}" for i in range(1, count))
    return values


def register_contract(inst):
    op = inst["opcode"].split(".")[0]
    operands = inst["operands"]
    if op in SIMPLE_DEST:
        n = 1
        # Integer carry outputs precede the arithmetic inputs; carry inputs
        # of .X forms follow them and are read normally, including !UP0.
        if op in ("IMAD", "UIMAD", "IADD3", "UIADD3"):
            while n < len(operands) and operands[n]["type"] in ("PRED", "UPRED"):
                n += 1
        elif ".X" in inst["opcode"] or len(operands) > 1 and operands[1]["type"] in ("PRED", "UPRED"):
            return [], [], False
    elif op in ("ISETP", "FSETP", "DSETP", "UISETP", "PLOP3", "UPLOP3"):
        n = 2
    elif op in NO_DEST:
        n = 0
    else:
        return [], [], False
    writes = []
    reads = []
    for i, operand in enumerate(operands):
        wide = op in ("IMAD", "UIMAD") and ".WIDE" in inst["opcode"] and i in (0, n + 2)
        rr = registers(operand, wide)
        if i < n:
            writes.extend(rr)
        else:
            reads.extend((r, i) for r in rr)
    if inst["predicate"] >= 0:
        reads.append((("UP" if inst["predicate_uniform"] else "P") + str(inst["predicate"]), "predicate"))
    return writes, reads, True


def normalized_operands(inst):
    names = {}

    def replace(m):
        name = m.group()
        prefix = re.match(r"[A-Z]+", name).group()
        if name not in names:
            names[name] = f"{prefix}@{len(names)}"
        return names[name]

    result = []
    for op in inst["operands"]:
        text = op["text"]
        if op["type"] == "CBANK":
            text = "constant_argument"
        elif inst["opcode"].split(".")[0] == "BRA":
            text = "observed_branch_target"
        else:
            text = REG.sub(replace, text)
        result.append({"type": op["type"], "expression": text, "bytes": op["bytes"]})
    return result


def allocation_at(address, allocations):
    found = [a for a in allocations if a["base"] <= address < a["base"] + a["bytes"]]
    if len(found) != 1:
        return None
    a = found[0]
    return (a["role"], address - a["base"])


def build_graph(prefix, context, allocations=(), identity=None):
    started = time.monotonic()
    prefix = Path(prefix)
    meta = Path(str(prefix) + ".jsonl")
    before = {str(meta): sha(meta)}
    records = [json.loads(x) for x in meta.read_text().splitlines()]
    configs = [r for r in records if r["type"] == "config"]
    if len(configs) != 1:
        raise ValueError("exactly one trace configuration required")
    config = configs[0]
    instructions = {r["id"]: r for r in records if r["type"] == "instruction"}
    contracts = {iid: register_contract(inst) for iid, inst in instructions.items()}
    operand_labels = {iid: normalized_operands(inst) for iid, inst in instructions.items()}
    launches = {r["id"]: r for r in records if r["type"] == "launch"}
    completed = {r["launch"]: r for r in records if r["type"] == "complete"}
    memory_boundaries = {r["before_launch"] for r in records if r["type"] == "memory_api"}
    memory_boundaries.update(r["launch"] for r in records if r["type"] == "scope" and r["enabled"])
    gaps = {
        "serialized_diagnostic_execution",
        "unobserved_input_paths",
        "compressed_instruction_graph_not_dynamic_isomorphism",
    }
    if not records or records[-1]["type"] != "end":
        gaps.add("trace_process_incomplete")
    if not allocations:
        gaps.add("boundary_allocations_unavailable")
    if any(r["type"] == "unsupported_api" for r in records):
        gaps.add("unsupported_cuda_api")
    nodes = {}
    edge_counts = Counter()
    edge_examples = {}
    patterns = defaultdict(lambda: defaultdict(list))
    constants = defaultdict(set)
    event_count = 0
    mem = []
    prev = {}
    last_reg = defaultdict(dict)
    node_gaps = defaultdict(set)
    launch_complete = {}
    unsafe_memory_launches = {r.get("launch", 0) for r in records if r["type"] == "unsupported_api"}
    thread_epoch = Counter()
    barrier_counts = defaultdict(Counter)
    barrier_nodes = defaultdict(dict)
    barrier_ids = defaultdict(lambda: defaultdict(list))

    def node(key, kind, attrs, evidence=None):
        if key not in nodes:
            nodes[key] = {"id": key, "kind": kind, "attrs": attrs, "unknowns": [], "evidence": evidence or {}}
        return nodes[key]

    def edge(a, b, kind, attrs=None, evidence=None):
        key = (a, b, kind, canonical(attrs or {}))
        edge_counts[key] += 1
        if evidence and key not in edge_examples:
            edge_examples[key] = evidence

    for allocation in allocations:
        node(
            "buffer:" + allocation["role"],
            "buffer",
            {"role": allocation["role"], "bytes": allocation["bytes"]},
            {"base": allocation["base"]},
        )
    for lid, launch in launches.items():
        lk = f"launch:{lid}"
        node(
            lk,
            "launch",
            {"grid": launch["grid"], "block": launch["block"], "dynamic_shared": launch["dynamic_shared"]},
            {"name": launch["name"], "stream": launch["stream"]},
        )
        if not launch["selected"]:
            node_gaps[lk].add("launch_not_captured")
            gaps.add("launches_not_captured")
            launch_complete[lid] = False
            continue
        c = completed.get(lid)
        if c is None:
            node_gaps[lk].add("missing_launch_completion")
            gaps.add("missing_launch_completion")
            launch_complete[lid] = False
            continue
        path = Path(f"{prefix}.launch{lid}.bin")
        before[str(path)] = sha(path)
        raw = path.read_bytes()
        if len(raw) != c["events"] * EVENT.size:
            raise ValueError("binary length does not match committed event count")
        all_ctas = config["cta_limit"] < 0 or config["cta_limit"] >= math.prod(launch["grid"])
        launch_complete[lid] = all_ctas and not c["dropped"]
        if not all_ctas:
            gaps.add("cta_sampling")
            node_gaps[lk].add("cta_sampling")
        if c["dropped"]:
            gaps.add("events_dropped")
            node_gaps[lk].add("events_dropped")
        for seq, values in enumerate(EVENT.iter_unpack(raw)):
            event_count += 1
            addr, const, iid, cta, tid, pred, mask, _ = values
            inst = instructions[iid]
            key = f"l{lid}:i{iid}"
            baseop = inst["opcode"].split(".")[0]
            n = node(
                key,
                "instruction",
                {
                    "opcode": inst["opcode"],
                    "space": inst["space"],
                    "load": inst["load"],
                    "store": inst["store"],
                    "size": inst["size"],
                    "operands": operand_labels[iid],
                    "predicate_negated": inst["sass"].lstrip().startswith("@!"),
                    "predicate_uniform": inst["predicate_uniform"],
                },
                {
                    "function": inst["function"],
                    "offset": inst["offset"],
                    "sass": inst["sass"],
                    "instruction_id": iid,
                    "launch": lid,
                },
            )
            t = (lid, cta, tid)
            epoch = thread_epoch[t]
            e = {
                "node": key,
                "launch": lid,
                "cta": cta,
                "thread": tid,
                "seq": seq,
                "epoch": epoch,
                "addr": addr,
                "size": inst["size"],
                "space": inst["space"],
                "read": inst["load"],
                "write": inst["store"],
            }
            pattern = patterns[key][cta, tid]
            if pattern and pattern[-1][:2] == [pred, mask]:
                pattern[-1][2] += 1
            else:
                pattern.append([pred, mask, 1])
            if t in prev:
                edge(prev[t], key, "thread_sequence", {"predicate_executed": bool(pred)})
            prev[t] = key
            # No membership/contains edge: all-instruction launch adjacency would
            # turn a local neighborhood into a whole-kernel comparison.
            n["attrs"]["launch_configuration"] = {k: launch[k] for k in ("grid", "block", "dynamic_shared")}
            writes, reads, known = contracts[iid]
            if not known:
                node_gaps[key].add("unsupported_register_semantics")
                gaps.add("unsupported_register_semantics")
                # Unknown destinations can invalidate any last-writer fact.
                if pred:
                    last_reg[t].clear()
            else:
                for r, port in reads:
                    if not pred and port != "predicate":
                        continue
                    source = last_reg[t].get(r)
                    if source:
                        edge(source, key, "predicate" if port == "predicate" else "register", {"port": port})
                    else:
                        node_gaps[key].add("register_live_in:" + r)
                if pred:
                    for r in writes:
                        last_reg[t][r] = key
            if pred and any(o["type"] == "CBANK" for o in inst["operands"]):
                if inst.get("captured_constant_bytes", 0):
                    binding = allocation_at(const, allocations)
                    if binding:
                        constants[key].add(canonical({"buffer": binding[0], "offset": binding[1]}))
                    else:
                        node_gaps[key].add("unclassified_constant_role")
                        gaps.add("unclassified_constant_role")
                        constants[key].add(canonical({"unclassified_bits": 8 * inst["captured_constant_bytes"]}))
                        n["evidence"].setdefault("unclassified_constant_examples", [])
                        if len(n["evidence"]["unclassified_constant_examples"]) < 2:
                            n["evidence"]["unclassified_constant_examples"].append(const)
                else:
                    node_gaps[key].add("constant_value_unobserved")
                    gaps.add("constant_value_unobserved")
            if pred and baseop == "BAR":
                # Only full CTA BAR.SYNC with immediate barrier ID is modeled.
                if (
                    inst["opcode"] in ("BAR.SYNC", "BAR.SYNC.DEFER_BLOCKING")
                    and len(inst["operands"]) == 1
                    and inst["operands"][0]["type"] == "IMM_UINT64"
                ):
                    barrier_counts[lid, cta][tid] += 1
                    barrier_ids[lid, cta][tid].append(inst["operands"][0]["text"])
                    barrier_nodes[lid, cta][tid, epoch] = key
                    thread_epoch[t] += 1
                else:
                    node_gaps[key].add("unsupported_barrier")
                    gaps.add("unsupported_barrier")
            if pred and (inst["load"] or inst["store"]) and inst["space"] not in ("CONSTANT", "NONE"):
                if (
                    inst["mrefs"] != 1
                    or inst["space"] not in ("GLOBAL", "SHARED", "LOCAL")
                    or inst["size"] not in (1, 2, 4, 8, 16)
                ):
                    node_gaps[key].add("unsupported_memory_semantics")
                    gaps.add("unsupported_memory_semantics")
                    unsafe_memory_launches.add(lid)
                    continue
                if inst["space"] == "GLOBAL":
                    binding = allocation_at(addr, allocations)
                    if (
                        binding
                        and allocation_at(addr + inst["size"] - 1, allocations)
                        and allocation_at(addr + inst["size"] - 1, allocations)[0] == binding[0]
                    ):
                        e["region"] = binding[0]
                        e["offset"] = binding[1]
                        e["address_key"] = ("GLOBAL", binding[0])
                    else:
                        node_gaps[key].add("unresolved_allocation")
                        gaps.add("unresolved_allocation")
                        continue
                else:
                    e["region"] = inst["space"]
                    e["offset"] = addr
                    e["address_key"] = (inst["space"], lid, cta, tid if inst["space"] == "LOCAL" else 0)
                if inst["load"] and inst["store"]:
                    node_gaps[key].add("atomic_read_modify_write")
                    gaps.add("atomic_read_modify_write")
                    unsafe_memory_launches.add(lid)
                    continue
                mem.append(e)
    valid_barriers = set()
    for (lid, cta), counts in barrier_counts.items():
        expected = math.prod(launches[lid]["block"])
        if (
            launch_complete[lid]
            and len(counts) == expected
            and len(set(counts.values())) == 1
            and len({tuple(v) for v in barrier_ids[lid, cta].values()}) == 1
        ):
            valid_barriers.add((lid, cta))
        else:
            gaps.add("unverified_barrier_participation")

    def happens_before(a, b):
        if a["launch"] != b["launch"]:
            lo, hi = a["launch"], b["launch"]
            return (
                lo < hi
                and launches[lo]["stream"] == launches[hi]["stream"]
                and all(launch_complete.get(i, False) and i not in unsafe_memory_launches for i in range(lo, hi + 1))
                and not any(lo < i <= hi for i in memory_boundaries)
            )
        if a["cta"] == b["cta"] and a["thread"] == b["thread"]:
            return a["seq"] < b["seq"]
        return a["cta"] == b["cta"] and (a["launch"], a["cta"]) in valid_barriers and a["epoch"] < b["epoch"]

    byte_writes = defaultdict(list)
    read_events = []
    memory_patterns = defaultdict(list)
    for e in mem:
        memory_patterns[e["node"]].append((e["cta"], e["thread"], e["region"], e["offset"], e["size"]))
        if e["write"]:
            for byte in range(e["offset"], e["offset"] + e["size"]):
                byte_writes[e["address_key"], byte].append(e)
        if e["read"]:
            read_events.append(e)
    if sum(len(v) for v in byte_writes.values()) > 4000000:
        raise ValueError("memory dependency budget exceeded; reduce diagnostic capture")
    for read in read_events:
        if (
            read["launch"] in unsafe_memory_launches
            or read["space"] == "GLOBAL"
            and (
                not launch_complete[read["launch"]]
                or len({launch_record["stream"] for launch_record in launches.values()}) > 1
            )
        ):
            node_gaps[read["node"]].add("memory_scope_incomplete")
            gaps.add("memory_scope_incomplete")
            continue
        sources = Counter()
        unknown = False
        for byte in range(read["offset"], read["offset"] + read["size"]):
            writes = byte_writes.get((read["address_key"], byte), [])
            if len(writes) > 512:
                unknown = True
                gaps.add("memory_history_budget")
                continue
            prior = [w for w in writes if happens_before(w, read)]
            unordered = [w for w in writes if not happens_before(w, read) and not happens_before(read, w)]
            # A later launch is not unordered just because its stream differs:
            # without original synchronization, its access may race this read.
            if unordered:
                unknown = True
                continue
            latest = [w for w in prior if not any(w is not x and happens_before(w, x) for x in prior)]
            if len(latest) == 1:
                sources[latest[0]["node"]] += 1
            elif len(latest) > 1:
                unknown = True
            elif (
                read["space"] == "GLOBAL"
                and read["region"].startswith("input:")
                and not any(0 < i <= read["launch"] for i in memory_boundaries)
                and not any(i <= read["launch"] for i in unsafe_memory_launches)
                and not any(i <= read["launch"] and not launch_complete.get(i, False) for i in launches)
            ):
                sources["buffer:" + read["region"]] += 1
            else:
                unknown = True
        for producer, nbytes in sources.items():
            edge(
                producer,
                read["node"],
                "memory",
                {"bytes_per_observed_read": nbytes},
                {"read_launch": read["launch"], "read_event": read["seq"], "offset": read["offset"]},
            )
        if unknown:
            node_gaps[read["node"]].add("memory_producer_unknown_or_racing")
            gaps.add("memory_producer_unknown_or_racing")
    # Export observed final writes to declared output regions. Racing stores do
    # not acquire a unique producer merely because their log records came last.
    for allocation in allocations:
        output_roles = [r for r in [allocation["role"], *allocation.get("aliases", [])] if r.startswith("output:")]
        if not output_roles:
            continue
        output_key = "result:" + sorted(set(output_roles))[0]
        node(output_key, "output", {"roles": sorted(set(output_roles)), "bytes": allocation["bytes"]})
        if (
            not all(launch_complete.values())
            or unsafe_memory_launches
            or len({launch_record["stream"] for launch_record in launches.values()}) > 1
        ):
            node_gaps[output_key].add("output_scope_incomplete")
            gaps.add("output_scope_incomplete")
            continue
        sources = Counter()
        unknown = False
        for byte in range(allocation["bytes"]):
            writes = byte_writes.get((("GLOBAL", allocation["role"]), byte), [])
            if len(writes) > 512:
                unknown = True
                gaps.add("memory_history_budget")
                continue
            latest = [w for w in writes if not any(w is not x and happens_before(w, x) for x in writes)]
            if len(latest) == 1:
                sources[latest[0]["node"]] += 1
            else:
                unknown = True
        for producer, nbytes in sources.items():
            edge(producer, output_key, "output_memory", {"bytes": nbytes})
        if unknown:
            node_gaps[output_key].add("output_producer_unknown_or_racing")
            gaps.add("output_producer_unknown_or_racing")
    for key, n in nodes.items():
        if key in patterns:
            ranges = []
            for (cta, tid), pattern in sorted(patterns[key].items()):
                if ranges and ranges[-1][0] == cta and ranges[-1][2] + 1 == tid and ranges[-1][3] == pattern:
                    ranges[-1][2] = tid
                else:
                    ranges.append([cta, tid, tid, pattern])
            n["attrs"]["execution_pattern"] = {
                "encoding": "cta_thread_ranges_of_predicate_mask_runs",
                "ranges": ranges,
            }
        if key in constants:
            n["attrs"]["constant_values"] = [json.loads(x) for x in sorted(constants[key])]
        if key in memory_patterns:
            # Sort threads, retaining within-thread address order including loop
            # revisits. Never sort away a transpose or a changed loop traversal.
            bythread = defaultdict(list)
            for cta, tid, region, offset, size in memory_patterns[key]:
                bythread[cta, tid].append([region, offset, size])
            n["attrs"]["memory_pattern"] = {
                "sha256": digest(sorted((list(k), v) for k, v in bythread.items())),
                "access_count": len(memory_patterns[key]),
            }
            n["evidence"]["memory_examples"] = memory_patterns[key][:8]
        if n["kind"] == "instruction":
            lid = n["evidence"]["launch"]
            if not launch_complete.get(lid, False):
                node_gaps[key].add("instruction_capture_incomplete")
        n["unknowns"] = sorted(node_gaps[key])
    edges = [
        {
            "source": a,
            "target": b,
            "kind": kind,
            "attrs": {**json.loads(attrs), "observed_count": count},
            "unknowns": [],
            "evidence": edge_examples.get((a, b, kind, attrs), {}),
        }
        for (a, b, kind, attrs), count in sorted(edge_counts.items())
    ]
    for path, h in before.items():
        if sha(path) != h:
            raise RuntimeError("trace changed while analyzing: " + path)
    return {
        "schema_version": SCHEMA,
        "id": identity or prefix.name,
        "context": context,
        "nodes": list(nodes.values()),
        "edges": edges,
        "coverage": {
            "complete": False,
            "scope": "predicated SASS instruction occurrences compressed by instruction; bounded observed inputs",
            "unknowns": sorted(gaps),
            "capture_complete": all(launch_complete.values()) and bool(launch_complete),
            "register_contract": "conservative SASS opcode subset",
            "memory_contract": "byte overlap with same-thread/same-stream or verified full CTA barrier ordering",
        },
        "unknowns": [],
        "provenance": {
            "files": before,
            "extractor_sha256": sha(__file__),
            "config": config,
            "event_count": event_count,
            "memory_events": len(mem),
            "elapsed_seconds": time.monotonic() - started,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--allocations", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    allocations = json.loads(a.allocations.read_text())["allocations"] if a.allocations else []
    graph = build_graph(a.trace, json.loads(a.context.read_text()), allocations)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(graph, f, indent=2)
    print(
        json.dumps(
            {
                "graph": str(a.output),
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "coverage": graph["coverage"],
                "seconds": graph["provenance"]["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
