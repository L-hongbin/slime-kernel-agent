"""Coarse kernel/library graph from memory runs, contracts and storage leases."""

import argparse
import bisect
import hashlib
import json
import math
import struct
import time
from collections import defaultdict
from pathlib import Path

RUN = struct.Struct("<QQIIII")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def union(intervals):
    result = []
    for lo, hi in sorted(set(intervals)):
        if lo >= hi:
            continue
        if result and lo <= result[-1][1]:
            result[-1][1] = max(hi, result[-1][1])
        else:
            result.append([lo, hi])
    return result


def overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None


def normalize_leases(allocations):
    result = []
    for i, source in enumerate(allocations):
        a = dict(source)
        a.setdefault("id", f"storage:{i}")
        a.setdefault("roles", [a["role"]] if "role" in a else [])
        a.setdefault("views", [])
        a.setdefault("first_seq", 0)
        a.setdefault("last_seq", None)
        result.append(a)
    # Distinct live storage wrappers can alias the same physical bytes (e.g.
    # DLPack views). Overlapping lifetimes distinguish aliasing from address reuse.
    parents = list(range(len(result)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, a in enumerate(result):
        for j, b in enumerate(result[:i]):
            memory_overlap = overlap((a["base"], a["base"] + a["bytes"]), (b["base"], b["base"] + b["bytes"]))
            lifetime_overlap = max(a["first_seq"], b["first_seq"]) < min(
                a["last_seq"] if a["last_seq"] is not None else math.inf,
                b["last_seq"] if b["last_seq"] is not None else math.inf,
            )
            if memory_overlap and lifetime_overlap:
                parents[root(i)] = root(j)
    groups = defaultdict(list)
    for i, a in enumerate(result):
        groups[root(i)].append(a)
    merged = []
    for values in groups.values():
        base = min(a["base"] for a in values)
        first = values[0]
        record = {
            **first,
            "base": base,
            "bytes": max(a["base"] + a["bytes"] for a in values) - base,
            "roles": sorted({r for a in values for r in a["roles"]}),
            "first_seq": min(a["first_seq"] for a in values),
            "last_seq": None if any(a["last_seq"] is None for a in values) else max(a["last_seq"] for a in values),
            "aliased_storage_ids": [a["id"] for a in values],
            "role_views": {},
        }
        for a in values:
            for role, view in a.get("role_views", {}).items():
                record["role_views"][role] = {**view, "offset_bytes": view["offset_bytes"] + a["base"] - base}
        merged.append(record)
    return merged


class Bindings:
    def __init__(self, allocations, seq):
        self.active = sorted(
            [a for a in allocations if a["first_seq"] <= seq and (a["last_seq"] is None or a["last_seq"] > seq)],
            key=lambda a: a["base"],
        )
        self.bases = [a["base"] for a in self.active]

    def find(self, address, size=1):
        at = bisect.bisect_right(self.bases, address) - 1
        if at < 0:
            return None
        a = self.active[at]
        if a["base"] <= address and address + size <= a["base"] + a["bytes"]:
            return a, address - a["base"]
        return None


def writer_index(runs):
    values = sorted(runs)
    starts, maximum = [], []
    end = -1
    for run in values:
        starts.append(run[0])
        end = max(end, run[1])
        maximum.append(end)
    return values, starts, maximum


def related(index, lo, hi):
    values, starts, maximum = index
    return values[bisect.bisect_right(maximum, lo) : bisect.bisect_left(starts, hi)]


def same_owner(read, write, lo, hi):
    # Tuple: interval start/end, first thread owner, element width, log index.
    if read[3] == write[3] and read[2] * read[3] - read[0] == write[2] * write[3] - write[0]:
        return True

    def owner(run, byte):
        return run[2] + (byte - run[0]) // run[3]

    return owner(read, lo) == owner(read, hi - 1) == owner(write, lo) == owner(write, hi - 1)


def entry_read_regions(reads, writes):
    """Reads with no in-kernel prior/other-thread writer are entry-version reads.

    Atomic log order is used ONLY when affine ownership proves the same thread.
    The remaining overlap is explicit unknown, not automatically an input edge.
    """
    index = writer_index(writes)
    entry, unknown = [], []
    for read in reads:
        overlaps = [w for w in related(index, read[0], read[1]) if overlap(read, w)]
        if len(overlaps) > 512:
            unknown.append(read[:2])
            continue
        cuts = sorted({read[0], read[1], *[v for w in overlaps for v in overlap(read, w)]})
        for lo, hi in zip(cuts, cuts[1:], strict=False):
            relevant = [w for w in overlaps if w[0] < hi and w[1] > lo]
            safe = all(same_owner(read, w, lo, hi) and read[4] < w[4] for w in relevant)
            (entry if safe else unknown).append((lo, hi))
    return union(entry), union(unknown)


def conflicting_writes(writes):
    index = writer_index(writes)
    conflicts = []
    for write in writes:
        candidates = related(index, write[0], write[1])
        if len(candidates) > 512:
            conflicts.append(write[:2])
            continue
        for other in candidates:
            common = overlap(write, other)
            if common and not same_owner(write, other, *common):
                conflicts.append(common)
    return union(conflicts)


def matrix_regions(pointer, rows, columns, leading, size):
    if rows < 0 or columns < 0 or leading < max(1, rows):
        raise ValueError("invalid GEMM matrix extent")
    if not rows or not columns:
        return []
    if leading == rows:
        return [(pointer, pointer + rows * columns * size)]
    if columns > 100000:
        raise ValueError("strided matrix region budget")
    return [(pointer + col * leading * size, pointer + (col * leading + rows) * size) for col in range(columns)]


def view_regions(view, budget=100000):
    if any(size == 0 for size in view["shape"]):
        return []
    width = view["element_size"]
    ranges = [[view["offset_bytes"], view["offset_bytes"] + width]]
    for stride, size in sorted(zip(view["stride"], view["shape"], strict=True)):
        step = stride * width
        if size <= 1 or step == 0:
            continue
        if len(ranges) == 1 and step == ranges[0][1] - ranges[0][0]:
            ranges[0][1] += (size - 1) * step
        elif len(ranges) * size <= budget:
            ranges = union((lo + i * step, hi + i * step) for lo, hi in ranges for i in range(size))
        else:
            raise ValueError("output view region budget")
    return ranges


def gemm_accesses(attrs):
    sizes = {0: 4, 1: 8, 2: 2, 14: 2}  # CUDA real F32/F64/F16/BF16
    if not attrs.get("state_query_ok") or attrs["alpha"] is None or attrs["beta"] is None:
        raise ValueError("GEMM scalar or handle state unavailable")
    if attrs["ta"] not in (0, 1, 2) or attrs["tb"] not in (0, 1, 2):
        raise ValueError("unsupported transpose enum")
    if any(attrs[t] not in sizes for t in ("at", "bt", "ct")):
        raise ValueError("unsupported GEMM element type")
    m, n, k = (attrs[x] for x in ("m", "n", "k"))
    if min(m, n, k) < 0:
        raise ValueError("negative GEMM dimension")
    result = []
    if not m or not n:
        return result
    if attrs["alpha"] != 0 and k:
        ar, ac = (m, k) if attrs["ta"] == 0 else (k, m)
        br, bc = (k, n) if attrs["tb"] == 0 else (n, k)
        for key, rows, cols, ld, kind in [("a", ar, ac, "lda", "at"), ("b", br, bc, "ldb", "bt")]:
            result.append((key, "read", matrix_regions(attrs[key], rows, cols, attrs[ld], sizes[attrs[kind]])))
    c = matrix_regions(attrs["c"], m, n, attrs["ldc"], sizes[attrs["ct"]])
    if attrs["beta"] != 0:
        result.append(("c", "read", c))
    result.append(("c", "write", c))
    return result


def launch_configuration(record):
    count = record.get("launch_num_attrs")
    return {"launch_num_attrs": count}, ([] if count == 0 else ["opaque_launch_configuration"])


def build_graph(prefix, allocations, context=None):
    started = time.monotonic()
    prefix = Path(prefix)
    lines = Path(str(prefix) + ".jsonl").read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or not records:
                raise
            records.append(
                {
                    "type": "memory_api_unknown",
                    "name": "truncated_metadata_record",
                    "seq": max(r.get("seq", 0) for r in records) + 1,
                }
            )
    config = next(r for r in records if r["type"] == "config")
    if config["schema"] != "coarse-memory-runs/v1":
        raise ValueError("old instruction traces must not enter coarse workflow")
    allocations = normalize_leases(allocations)
    by_buffer = {a["id"]: a for a in allocations}
    launches = {r["id"]: r for r in records if r["type"] == "launch"}
    completions = {r["id"]: r for r in records if r["type"] == "complete"}
    endings = {r["parent"]: r for r in records if r["type"] == "library_end"}
    scopes = [r["seq"] for r in records if r["type"] == "scope" and r["enabled"]]
    first = scopes[0] if scopes else min((r["seq"] for r in launches.values()), default=0)
    nodes, memory_unknown = [], []
    current_parent = None
    for r in records:
        if r["type"] == "library_begin":
            current_parent = r["seq"]
        if r["type"] == "library_end":
            current_parent = None
        if r["type"] in ("memory_api_unknown", "unsupported_launch") and r["seq"] >= first and current_parent is None:
            memory_unknown.append(r)
        if r["type"] == "launch" and r["parent"] >= 0:
            continue
        if r["type"] not in ("launch", "library_begin", "copy", "fill") or r["seq"] < first:
            continue
        if r["type"] in ("copy", "fill") and current_parent is not None:
            continue
        kind = {"launch": "kernel", "library_begin": "library", "copy": "copy", "fill": "fill"}[r["type"]]
        node = {
            "id": f"{kind}:{r['seq']}",
            "kind": kind,
            "seq": r["seq"],
            "stream": r["stream"],
            "reads": [],
            "writes": [],
            "unknowns": [],
            "evidence": {},
            "configuration": {},
            "footprint_complete": True,
        }
        binding = Bindings(allocations, r["seq"])
        if kind == "kernel":
            node["evidence"] = {"name": r["name"], "launch_id": r["id"]}
            node["implementation"] = r.get("implementation_version", r["implementation"])
            if r.get("implementation_inspected") is False:
                node["implementation"] = "uninspected"
                node["unknowns"].append("opaque_implementation_configuration")
            node["configuration"] = {k: r[k] for k in ("grid", "block", "shared")}
            attrs, unknowns = launch_configuration(r)
            node["configuration"].update(attrs)
            node["unknowns"].extend(unknowns)
            args = []
            for arg in r["arguments"]:
                match = binding.find(arg["bits"]) if arg["bytes"] == 8 else None
                if match:
                    a, offset = match
                    args.append(
                        {"index": arg["index"], "kind": "address_binding", "buffer": a["id"], "offset": offset}
                    )
                elif arg["bytes"] <= 8:
                    args.append(
                        {"index": arg["index"], "kind": "argument_bits", "bytes": arg["bytes"], "bits": arg["bits"]}
                    )
                else:
                    args.append({"index": arg["index"], "kind": "opaque_argument", "bytes": arg["bytes"]})
                    node["unknowns"].append("opaque_argument_configuration")
            node["configuration"]["arguments"] = args
            if r.get("arguments_present") is False:
                node["unknowns"].append("opaque_argument_configuration")
            done = completions.get(r["id"])
            if done is None:
                node["unknowns"].append("launch_completion_missing")
                node["footprint_complete"] = False
                nodes.append(node)
                continue
            node["end_seq"] = done["seq"]
            if not r["memory_traced"]:
                node["footprint_complete"] = False
                node["unknowns"].append(r.get("memory_skip_reason", "memory_capture_disabled"))
                nodes.append(node)
                continue
            full_ctas = config["cta_limit"] < 0 or config["cta_limit"] >= math.prod(r["grid"])
            if done["dropped"] or not full_ctas or r["unsupported_memory_instructions"]:
                node["footprint_complete"] = False
                node["unknowns"].extend(
                    key
                    for key, present in [
                        ("memory_runs_dropped", done["dropped"]),
                        ("cta_sampling", not full_ctas),
                        ("unsupported_memory_instruction", r["unsupported_memory_instructions"]),
                    ]
                    if present
                )
            data = Path(f"{prefix}.launch{r['id']}.bin").read_bytes()
            if len(data) != done["runs"] * RUN.size:
                raise ValueError("memory run length does not match completed capture")
            per_buffer = defaultdict(lambda: {"reads": [], "writes": [], "atomic": False})
            unmapped = 0
            for seq, (address, owner, length, width, mode, _) in enumerate(RUN.iter_unpack(data)):
                a = binding.find(address, length)
                if a is None:
                    unmapped += length
                    continue
                storage, offset = a
                access = (offset, offset + length, owner, width, seq)
                if mode & 1:
                    per_buffer[storage["id"]]["reads"].append(access)
                if mode & 2:
                    per_buffer[storage["id"]]["writes"].append(access)
                if mode == 3:
                    per_buffer[storage["id"]]["atomic"] = True
            node["evidence"].update(
                raw_runs=done["runs"],
                dropped_runs=done["dropped"],
                unmapped_access_bytes=unmapped,
                dropped_runs_exact=done.get("drop_count_exact", True),
            )
            if unmapped:
                node["unknowns"].append("unmapped_storage_access")
                node["footprint_complete"] = False
            for storage, accesses in per_buffer.items():
                reads, writes = accesses["reads"], accesses["writes"]
                entry, ambiguous = entry_read_regions(reads, writes)
                conflict = conflicting_writes(writes)
                if reads:
                    node["reads"].append(
                        {
                            "buffer": storage,
                            "regions": union(x[:2] for x in reads),
                            "entry_regions": entry,
                            "internal_or_unordered_regions": ambiguous,
                            "port": None,
                            "evidence": "observed_global_memory_runs",
                        }
                    )
                if writes:
                    node["writes"].append(
                        {
                            "buffer": storage,
                            "regions": union(x[:2] for x in writes),
                            "conflicting_regions": conflict,
                            "atomic_unknown": accesses["atomic"],
                            "port": None,
                            "evidence": "observed_global_memory_runs",
                        }
                    )
                if ambiguous:
                    node["unknowns"].append("internal_or_unordered_read_version")
                if conflict or accesses["atomic"]:
                    node["unknowns"].append("write_version_ambiguous")
        else:
            node["evidence"] = {"api": r.get("api", "cuMemcpyDtoDAsync_v2")}
            if kind == "library":
                node["configuration"] = {k: v for k, v in r["attrs"].items() if k not in ("a", "b", "c")}
                children = [x for x in launches.values() if x["parent"] == r["seq"]]
                if not children:
                    node["unknowns"].append("opaque_library_implementation_configuration")
                node["configuration"]["child_launch_num_attrs"] = [x.get("launch_num_attrs") for x in children]
                for child in children:
                    node["unknowns"].extend(launch_configuration(child)[1])
                    if child.get("implementation_inspected") is False:
                        node["unknowns"].append("opaque_library_implementation_configuration")
                if r["attrs"].get("workspace_mode", "unknown") == "unknown":
                    node["unknowns"].append("opaque_workspace_configuration")
                node["evidence"]["kernel_launches"] = [x["id"] for x in children]
                node["implementation"] = digest(
                    [
                        r["api"],
                        [
                            (x["name"], x["grid"], x["block"], x["shared"], x.get("implementation_version"))
                            for x in children
                        ],
                    ]
                )
                node["operation"] = "gemm"
                end = endings.get(r["seq"])
                node["end_seq"] = end["seq"] if end else r["seq"]
                try:
                    if end is None or end["status"] != 0:
                        raise ValueError("library call unsuccessful or incomplete")
                    requested = gemm_accesses(r["attrs"])
                except ValueError as exc:
                    requested = []
                    node["unknowns"].append(str(exc))
                    node["footprint_complete"] = False
                node["evidence"]["private_library_memory"] = "opaque; public GEMM operand contract only"
            elif kind == "copy":
                node["implementation"] = "cuda_device_copy"
                node["end_seq"] = r["seq"]
                node["configuration"] = {"bytes": r["bytes"]}
                requested = [
                    ("source", "read", [(r["source"], r["source"] + r["bytes"])]),
                    ("target", "write", [(r["target"], r["target"] + r["bytes"])]),
                ]
            else:
                node["implementation"] = "cuda_memset"
                node["end_seq"] = r["seq"]
                node["configuration"] = {k: r[k] for k in ("value", "bytes", "element_bytes")}
                node["evidence"]["api"] = "cuMemsetAsync"
                requested = [("target", "write", [(r["target"], r["target"] + r["bytes"])])]
            for port, mode, intervals in requested:
                groups = defaultdict(list)
                for lo, hi in intervals:
                    found = binding.find(lo, hi - lo)
                    if not found:
                        node["unknowns"].append("unmapped_contract_operand")
                        node["footprint_complete"] = False
                        continue
                    storage, offset = found
                    groups[storage["id"]].append((offset, offset + hi - lo))
                for storage, regions in groups.items():
                    access = {
                        "buffer": storage,
                        "regions": union(regions),
                        "port": port,
                        "evidence": "successful_library_api_contract" if kind == "library" else "cuda_memory_api",
                    }
                    if mode == "read":
                        access.update(entry_regions=access["regions"], internal_or_unordered_regions=[])
                        node["reads"].append(access)
                    else:
                        access.update(conflicting_regions=[], atomic_unknown=False)
                        node["writes"].append(access)
        node["unknowns"] = sorted(set(node["unknowns"]))
        nodes.append(node)
    nodes.sort(key=lambda n: n["seq"])
    stream_slots = {}
    for node in nodes:
        stream_slots.setdefault(node["stream"], len(stream_slots))
        node["configuration"]["stream_slot"] = stream_slots[node["stream"]]
    output_node = {
        "id": "forward_output",
        "kind": "output",
        "seq": max(r.get("seq", 0) for r in records) + 1,
        "stream": -1,
        "reads": [],
        "writes": [],
        "footprint_complete": True,
    }
    output_unknowns = []
    for allocation in allocations:
        for role in allocation["roles"]:
            if not role.startswith("output:"):
                continue
            try:
                view = allocation.get("role_views", {}).get(role)
                regions = view_regions(view) if view else [[0, allocation["bytes"]]]
                output_node["reads"].append(
                    {
                        "buffer": allocation["id"],
                        "port": role,
                        "regions": regions,
                        "entry_regions": regions,
                        "evidence": "returned_tensor_view",
                    }
                )
            except ValueError as exc:
                output_unknowns.append(str(exc))
    edges, versions = version_dependencies(
        nodes + [output_node], allocations, memory_unknown, [r for r in records if r["type"] == "order"]
    )
    outputs = [edge for edge in edges if edge["target"] == "forward_output"]
    edges = [edge for edge in edges if edge["target"] != "forward_output"]
    for node in nodes:
        for access in node["reads"] + node["writes"]:
            a = by_buffer[access["buffer"]]
            access["buffer_roles"] = a["roles"]
            access["buffer_bytes"] = a["bytes"]
    return {
        "schema": "coarse-component-graph/v1",
        "context": context or {},
        "nodes": nodes,
        "edges": edges,
        "buffers": allocations,
        "versions": versions,
        "outputs": outputs,
        "output_unknowns": output_unknowns,
        "coverage": {
            "kernel_launches": len(launches),
            "total_launches_reported": next((r["launches"] for r in reversed(records) if r["type"] == "end"), None),
            "completed_kernel_launches": len(completions),
            "memory_traced_kernels": sum(x["memory_traced"] for x in launches.values()),
            "contract_covered_kernels": sum(x["parent"] >= 0 for x in launches.values()),
            "complete_node_footprints": sum(n["footprint_complete"] for n in nodes),
            "node_count": len(nodes),
            "memory_unknown_events": memory_unknown,
            "trace_process_complete": records[-1]["type"] == "end",
            "config": config,
        },
        "provenance": {
            "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(Path(str(prefix) + ".jsonl").read_bytes()).hexdigest(),
            "parse_seconds": time.monotonic() - started,
        },
    }


def version_dependencies(nodes, allocations, memory_unknown=(), orders=()):
    """Region versions require original stream order, never capture serialization."""
    # Only user/program synchronization is recorded. The collector's diagnostic
    # cuCtxSynchronize calls are excluded at the native reentrancy guard.
    ancestors, tails, events, pending = {}, {}, {}, defaultdict(set)
    global_barrier = set()
    unknown_memory_nodes = [
        {
            "id": f"unknown_memory:{e['seq']}",
            "seq": e["seq"],
            "stream": e.get("stream", -1000 - e["seq"]),
            "footprint_complete": False,
        }
        for e in memory_unknown
    ]
    timeline = sorted(
        [("node", n) for n in nodes + unknown_memory_nodes] + [("order", e) for e in orders], key=lambda x: x[1]["seq"]
    )
    for kind, item in timeline:
        if kind == "node":
            parents = set(global_barrier) | pending[item["stream"]]
            if item["stream"] in tails:
                parents.add(tails[item["stream"]])
            ancestors[item["id"]] = parents | {a for parent in parents for a in ancestors.get(parent, ())}
            tails[item["stream"]] = item["id"]
        elif item["action"] == "record":
            events[item["event"]] = tails.get(item["stream"])
        elif item["action"] == "wait" and events.get(item["event"]) is not None:
            pending[item["stream"]].add(events[item["event"]])
        elif item["action"] == "destroy":
            events.pop(item["event"], None)
        elif item["action"] == "device_sync":
            global_barrier.update(tails.values())
    writers = defaultdict(list)
    versions = []
    for n in nodes:
        for access in n["writes"]:
            version = {
                "id": f"{access['buffer']}@{n['id']}",
                "buffer": access["buffer"],
                "producer": n["id"],
                "regions": access["regions"],
                "seq": n["seq"],
                "stream": n["stream"],
                "complete": n["footprint_complete"],
                "conflicts": access["conflicting_regions"],
                "atomic_unknown": access["atomic_unknown"],
            }
            versions.append(version)
            writers[access["buffer"]].append(version)
    initial = {a["id"]: a for a in allocations}
    unknown_calls = [n for n in nodes if not n["footprint_complete"]] + unknown_memory_nodes
    edges = []
    for node in nodes:
        for read in node["reads"]:
            candidates = [w for w in writers[read["buffer"]] if w["producer"] != node["id"]]
            grouped = defaultdict(list)
            for lo, hi in read["entry_regions"]:
                relevant = [w for w in candidates if any(overlap((lo, hi), region) for region in w["regions"])]
                cuts = sorted(
                    {
                        lo,
                        hi,
                        *[
                            v
                            for w in relevant
                            for r in w["regions"]
                            if overlap((lo, hi), r)
                            for v in overlap((lo, hi), r)
                        ],
                    }
                )
                for start, end in zip(cuts, cuts[1:], strict=False):
                    ws = [w for w in relevant if any(overlap((start, end), r) for r in w["regions"])]
                    before = [w for w in ws if w["producer"] in ancestors[node["id"]]]
                    unordered = [
                        w
                        for w in ws
                        if w["producer"] not in ancestors[node["id"]] and node["id"] not in ancestors[w["producer"]]
                    ]
                    latest_versions = [
                        w
                        for w in before
                        if not any(w["producer"] in ancestors[other["producer"]] for other in before if other is not w)
                    ]
                    latest = latest_versions[0] if len(latest_versions) == 1 else None
                    possible_hidden_writer = any(
                        unknown["id"] != node["id"]
                        and node["id"] not in ancestors[unknown["id"]]
                        and not (latest and unknown["id"] in ancestors[latest["producer"]])
                        for unknown in unknown_calls
                    )
                    unknown_between = possible_hidden_writer
                    if unordered or len(latest_versions) > 1:
                        source, version, certainty = None, None, "unordered_stream_writers"
                    elif latest:
                        conflict = any(overlap((start, end), r) for r in latest["conflicts"])
                        source, version = latest["producer"], latest["id"]
                        certainty = (
                            "proven_region_dependency"
                            if latest["complete"]
                            and node["footprint_complete"]
                            and not conflict
                            and not latest["atomic_unknown"]
                            and not unknown_between
                            else "partial_or_ambiguous_dependency"
                        )
                    elif (
                        initial[read["buffer"]]["roles"]
                        and any(
                            r.startswith(("input:", "parameter:", "state:")) for r in initial[read["buffer"]]["roles"]
                        )
                        and not unknown_between
                    ):
                        source, version, certainty = (
                            "initial:" + read["buffer"],
                            read["buffer"] + "@entry",
                            "initial_value_read" if node["footprint_complete"] else "possible_initial_value_read",
                        )
                    else:
                        source, version, certainty = None, None, "entry_value_unobserved"
                    grouped[source, version, certainty].append((start, end))
            for (source, version, certainty), regions in grouped.items():
                edges.append(
                    {
                        "source": source,
                        "target": node["id"],
                        "buffer": read["buffer"],
                        "version": version,
                        "regions": union(regions),
                        "port": read["port"],
                        "certainty": certainty,
                        "evidence": read["evidence"],
                    }
                )
    return edges, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--allocations", type=Path, required=True)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph = build_graph(
        args.trace,
        json.loads(args.allocations.read_text())["allocations"],
        json.loads(args.context.read_text()) if args.context else {},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(graph, handle, indent=2)
    print(
        json.dumps(
            {
                "coverage": graph["coverage"],
                "edges": len(graph["edges"]),
                "seconds": graph["provenance"]["parse_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
