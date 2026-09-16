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


def subtract_regions(left, right):
    """Exact half-open interval subtraction, preserving every observed hole."""
    right = union(map(tuple, right))
    result = []
    index = 0
    for lo, hi in union(map(tuple, left)):
        while index < len(right) and right[index][1] <= lo:
            index += 1
        cursor = lo
        for j in range(index, len(right)):
            start, end = right[j]
            if end <= cursor:
                continue
            if start >= hi:
                break
            if cursor < start:
                result.append([cursor, min(start, hi)])
            cursor = max(cursor, end)
            if cursor >= hi:
                break
        if cursor < hi:
            result.append([cursor, hi])
    return result


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

    def pieces(self, address, size):
        """A merged region may span adjacent allocations; split by live leases."""
        end = address + size
        while address < end:
            match = self.find(address)
            if match:
                allocation, offset = match
                length = min(end - address, allocation["bytes"] - offset)
                yield allocation, offset, length
            else:
                index = bisect.bisect_right(self.bases, address)
                length = min(end, self.bases[index] if index < len(self.bases) else end) - address
                yield None, 0, length
            address += length


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
    if not writes:
        # With no in-kernel writers every read observes the entry version.
        # This is exactly the loop below with an empty writer index, without
        # repeated bisection/cut construction for every memory run.
        return union(read[:2] for read in reads), []
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
    if len(writes) < 2:
        return []
    ordered = sorted(writes)
    previous_hi = ordered[0][1]
    disjoint = True
    for write in ordered[1:]:
        if write[0] < previous_hi:
            disjoint = False
            break
        previous_hi = write[1]
    if disjoint:
        # The slow path can only append an overlap of two write intervals.
        # Half-open, pairwise-disjoint intervals have none.
        return []
    index = writer_index(ordered)
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


def matrix_bounding_span(rows, columns, leading):
    """Element span of a column-major matrix, including leading-dimension gaps."""
    if rows < 0 or columns < 0 or leading < max(1, rows):
        raise ValueError("invalid GEMM matrix extent")
    if not rows or not columns:
        return 0
    if columns > 100000:
        raise ValueError("strided matrix region budget")
    return (columns - 1) * leading + rows


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


def _single_gemm_accesses(attrs):
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
        c_span = matrix_bounding_span(m, n, attrs["ldc"])
        c_bounds = (attrs["c"], attrs["c"] + c_span * sizes[attrs["ct"]])
        for key, rows, cols, ld, kind in [("a", ar, ac, "lda", "at"), ("b", br, bc, "ldb", "bt")]:
            operand_span = matrix_bounding_span(rows, cols, attrs[ld])
            operand_bounds = (attrs[key], attrs[key] + operand_span * sizes[attrs[kind]])
            if c_bounds[0] < operand_bounds[1] and operand_bounds[0] < c_bounds[1]:
                raise ValueError("GEMM C aliases " + key.upper())
            result.append((key, "read", matrix_regions(attrs[key], rows, cols, attrs[ld], sizes[attrs[kind]])))
    c = matrix_regions(attrs["c"], m, n, attrs["ldc"], sizes[attrs["ct"]])
    if attrs["beta"] != 0:
        result.append(("c", "read", c))
    result.append(("c", "write", c))
    return result


def gemm_accesses(attrs):
    """Public cuBLAS GEMM contract, including strided batches in element units."""
    batch_count = attrs.get("batch_count")
    if batch_count is None:
        return _single_gemm_accesses(attrs)
    if type(batch_count) is not int or batch_count < 0:
        raise ValueError("invalid strided-batched GEMM batch count")
    if batch_count == 0:
        return []
    strides = {key: attrs.get(key) for key in ("stride_a", "stride_b", "stride_c")}
    if any(type(value) is not int or value < 0 for value in strides.values()):
        raise ValueError("invalid strided-batched GEMM stride")
    if batch_count > 100000:
        raise ValueError("strided-batched GEMM batch count budget")
    sizes = {0: 4, 1: 8, 2: 2, 14: 2}
    if any(attrs.get(kind) not in sizes for kind in ("at", "bt", "ct")):
        raise ValueError("unsupported GEMM element type")
    if attrs.get("ta") not in (0, 1, 2) or attrs.get("tb") not in (0, 1, 2):
        raise ValueError("unsupported transpose enum")
    m, n, k = (attrs[key] for key in ("m", "n", "k"))
    if min(m, n, k) < 0:
        raise ValueError("negative GEMM dimension")
    ar, ac = (m, k) if attrs["ta"] == 0 else (k, m)
    br, bc = (k, n) if attrs["tb"] == 0 else (n, k)
    spans = {
        "a": matrix_bounding_span(ar, ac, attrs["lda"]),
        "b": matrix_bounding_span(br, bc, attrs["ldb"]),
        "c": matrix_bounding_span(m, n, attrs["ldc"]),
    }
    if batch_count > 1 and strides["stride_c"] < spans["c"]:
        raise ValueError("overlapping strided-batched GEMM C batches")

    def bounds(port):
        size = sizes[attrs[port + "t"]]
        return (
            attrs[port],
            attrs[port] + ((batch_count - 1) * strides["stride_" + port] + spans[port]) * size,
        )

    c_bounds = bounds("c")
    for port in ("a", "b"):
        operand_bounds = bounds(port)
        if c_bounds[0] < operand_bounds[1] and operand_bounds[0] < c_bounds[1]:
            raise ValueError("strided-batched GEMM C aliases " + port.upper())
    grouped = defaultdict(list)
    interval_count = 0
    for batch in range(batch_count):
        batched = dict(attrs)
        for pointer, stride, kind in (("a", "stride_a", "at"), ("b", "stride_b", "bt"), ("c", "stride_c", "ct")):
            batched[pointer] = attrs[pointer] + batch * strides[stride] * sizes[attrs[kind]]
        for port, mode, intervals in _single_gemm_accesses(batched):
            interval_count += len(intervals)
            if interval_count > 100000:
                raise ValueError("strided-batched GEMM interval budget")
            grouped[port, mode].extend(intervals)
    # One logical cuBLAS operand is one graph interface.  Merging intervals
    # preserves byte-exact batch gaps while preventing repeated same-buffer
    # writes from becoming duplicate versions at one library node.
    return [(port, mode, union(intervals)) for (port, mode), intervals in grouped.items()]


def launch_configuration(record):
    count = record.get("launch_num_attrs")
    if type(count) is not int or count < 0:
        return {"launch_num_attrs": count}, ["opaque_launch_configuration"]
    summary = record.get("launch_attributes")
    # Historical ordinary launches predate the detailed field.  They have no
    # extensible attributes, so preserve their established complete meaning.
    if summary is None:
        return {"launch_num_attrs": count}, ([] if count == 0 else ["opaque_launch_configuration"])
    if not isinstance(summary, dict) or summary.get("complete") is not True:
        return {"launch_num_attrs": count}, ["opaque_launch_configuration"]
    values = summary.get("values")
    if not isinstance(values, list) or len(values) != count:
        return {"launch_num_attrs": count}, ["opaque_launch_configuration"]
    scalar_ids = {
        "cooperative",
        "synchronization_policy",
        "cluster_scheduling_policy_preference",
        "programmatic_stream_serialization",
        "priority",
        "mem_sync_domain",
        "preferred_shared_memory_carveout",
    }
    triple_ids = {"cluster_dimension", "preferred_cluster_dimension"}
    pair_ids = {"mem_sync_domain_map"}
    try:
        for value in values:
            if not isinstance(value, dict) or set(value) != {"id", "value"}:
                raise ValueError("invalid launch attribute entry")
            attribute_id, attribute_value = value["id"], value["value"]
            if attribute_id == "ignore":
                if attribute_value is not True:
                    raise ValueError("invalid ignore attribute")
            elif attribute_id in scalar_ids:
                if type(attribute_value) is not int:
                    raise ValueError("invalid scalar launch attribute")
            elif attribute_id in triple_ids:
                if (
                    not isinstance(attribute_value, list)
                    or len(attribute_value) != 3
                    or any(type(element) is not int for element in attribute_value)
                ):
                    raise ValueError("invalid cluster launch attribute")
            elif attribute_id in pair_ids:
                if (
                    not isinstance(attribute_value, list)
                    or len(attribute_value) != 2
                    or any(type(element) is not int for element in attribute_value)
                ):
                    raise ValueError("invalid memory-domain launch attribute")
            else:
                raise ValueError("unsupported launch attribute")
    except (TypeError, ValueError):
        return {"launch_num_attrs": count}, ["opaque_launch_configuration"]
    # The native producer sorts by semantic ID.  Require strict order and
    # uniqueness so a malformed summary cannot smuggle an ambiguous config
    # into a matching signature.
    ids = [value["id"] for value in values]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        return {"launch_num_attrs": count}, ["opaque_launch_configuration"]
    return {"launch_num_attrs": count, "launch_attributes": values}, []


def _cubin_witness(path):
    """Validate bounded CUDA ELF extents and identify actual kernel entries.

    NVBit 1.8 may return false even after writing a cubin successfully. A
    nonempty file alone is insufficient evidence: require the complete section
    table, data extents, and an executable section for the requested entry.
    """
    size = path.stat().st_size
    if not 64 <= size <= 512 * 1024**2:
        raise ValueError("cubin size outside diagnostic bound")
    with path.open("rb") as handle:
        header = struct.unpack("<16sHHIQQQIHHHHHH", handle.read(64))
        ident, kind, machine = header[:3]
        phoff, shoff, ehsize, phsize, phnum, shsize, shnum, shstr = (
            header[5],
            header[6],
            header[8],
            header[9],
            header[10],
            header[11],
            header[12],
            header[13],
        )
        if (
            ident[:7] != b"\x7fELF\x02\x01\x01"
            or kind not in {1, 2}
            or machine != 190
            or ehsize != 64
            or shsize != 64
            or not 0 < shnum < 65535
            or not 0 < shstr < shnum
            or shoff < 64
            or shoff + shsize * shnum > size
            or (phnum and (phsize != 56 or phoff < 64 or phoff + phsize * phnum > size))
        ):
            raise ValueError("invalid or truncated CUDA ELF header")
        handle.seek(shoff)
        sections = list(struct.iter_unpack("<IIQQQQIIQQ", handle.read(shsize * shnum)))
        for section in sections:
            if section[1] != 8 and section[4] + section[5] > size:  # SHT_NOBITS has no file data.
                raise ValueError("truncated CUDA ELF section")
        strings = sections[shstr]
        if strings[1] != 3:
            raise ValueError("invalid CUDA ELF section-name table")
        handle.seek(strings[4])
        names = handle.read(strings[5])
        entries = set()
        for section in sections:
            start = section[0]
            end = names.find(b"\0", start)
            if start >= len(names) or end < 0:
                raise ValueError("invalid CUDA ELF section name")
            name = names[start:end]
            if name.startswith(b".text.") and section[1] == 1 and section[2] & 4 and section[5] > 0:
                entries.add(name[6:].decode("utf-8"))
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest(), entries


def library_binary_configuration(prefix, record, cache=None):
    """Resolve a native-owned module dump without trusting a path from JSON."""
    summary = record.get("library_binary")
    if not isinstance(summary, dict) or summary.get("complete") is not True:
        return None
    module_id, module_bytes, entry = (
        summary.get("module_id"),
        summary.get("module_bytes"),
        summary.get("entry_selector"),
    )
    if (
        type(module_id) is not int
        or module_id < 0
        or type(module_bytes) is not int
        or module_bytes <= 0
        or not isinstance(entry, str)
        or not entry
        or len(entry) > 4096
    ):
        return None
    path = Path(f"{prefix}.module{module_id}.cubin")
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size != module_bytes:
            return None
        key = (path, module_bytes, path.stat().st_mtime_ns)
        if cache is None:
            cache = {}
        if key not in cache:
            cache[key] = _cubin_witness(path)
        fingerprint, entries = cache[key]
        if entry not in entries:
            return None
        return {
            "cubin_sha256": fingerprint,
            # The selector is meaningful only inside this exact cubin hash;
            # it is never an implementation identity by itself.
            "entry_selector": entry,
        }
    except (OSError, ValueError, UnicodeError, struct.error):
        return None


def build_graph(prefix, allocations, context=None):
    started = time.monotonic()
    prefix = Path(prefix)
    binary_cache = {}
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
    record_format = next((r for r in records if r["type"] == "record_format"), {})
    region_summary = record_format.get("format") == "exact_byte_regions/v1"
    config = {**config, **{k: v for k, v in record_format.items() if k != "type"}}
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
            launch_format = r.get("record_format", record_format.get("format", "ordered_memory_runs/v1"))
            if launch_format not in {"exact_byte_regions/v1", "ordered_memory_runs/v1"}:
                raise ValueError("unknown per-launch record format")
            region_summary = launch_format == "exact_byte_regions/v1"
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
            per_buffer = defaultdict(lambda: {"reads": [], "writes": [], "atomic": False, "conflicts": []})
            unmapped = 0
            for seq, (address, owner, length, width, mode, _) in enumerate(RUN.iter_unpack(data)):
                if region_summary:
                    if mode not in (1, 2, 4, 8):
                        raise ValueError("invalid exact-region record mode")
                    for storage, offset, size in binding.pieces(address, length):
                        if storage is None:
                            unmapped += size
                            continue
                        access = (offset, offset + size, 0, 1, 0)
                        target = per_buffer[storage["id"]]
                        if mode == 1:
                            target["reads"].append(access)
                        elif mode == 2:
                            target["writes"].append(access)
                        elif mode == 4:
                            target["conflicts"].append(access[:2])
                        else:
                            target["atomic"] = True
                    continue
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
                record_unit="merged_byte_interval" if region_summary else "ordered_memory_run",
            )
            if unmapped:
                node["unknowns"].append("unmapped_storage_access")
                node["footprint_complete"] = False
            for storage, accesses in per_buffer.items():
                reads, writes = accesses["reads"], accesses["writes"]
                if region_summary:
                    all_reads, all_writes = union(r[:2] for r in reads), union(w[:2] for w in writes)
                    entry = subtract_regions(all_reads, all_writes)
                    ambiguous = subtract_regions(all_reads, entry)
                    conflict = union(accesses["conflicts"])
                else:
                    entry, ambiguous = entry_read_regions(reads, writes)
                    conflict = conflicting_writes(writes)
                if reads:
                    # The no-writer fast path already returned this exact
                    # canonical union for entry regions; do not recompute it.
                    read_regions = entry if not writes else union(x[:2] for x in reads)
                    node["reads"].append(
                        {
                            "buffer": storage,
                            "regions": read_regions,
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
                    if region_summary:
                        node["unknowns"].append("inplace_order_requires_detailed_capture")
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
                child_configurations = []
                for child in children:
                    child_configuration, child_unknowns = launch_configuration(child)
                    binary_configuration = library_binary_configuration(prefix, child, binary_cache)
                    if binary_configuration is None:
                        child_unknowns.append("opaque_library_implementation_configuration")
                    else:
                        child_configuration = {**child_configuration, "library_binary": binary_configuration}
                    child_configurations.append(child_configuration)
                    node["unknowns"].extend(child_unknowns)
                    if child.get("implementation_inspected") is not True:
                        node["unknowns"].append("opaque_library_implementation_configuration")
                node["configuration"]["child_launch_configurations"] = child_configurations
                if r["attrs"].get("workspace_mode", "unknown") == "unknown":
                    node["unknowns"].append("opaque_workspace_configuration")
                node["evidence"]["kernel_launches"] = [x["id"] for x in children]
                node["implementation"] = digest(
                    [
                        r["api"],
                        [
                            (
                                x["grid"],
                                x["block"],
                                x["shared"],
                                child_configuration,
                            )
                            for x, child_configuration in zip(children, child_configurations, strict=True)
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
    # Every version owns canonical disjoint intervals. Binary search preserves
    # the exact overlap result without rescanning a huge strided footprint for
    # each read interval and each partition cell.
    indexed = {}

    def intersecting(version, lo, hi, field="regions"):
        key = (id(version), field)
        if key not in indexed:
            regions = union(map(tuple, version[field]))
            indexed[key] = (regions, [r[0] for r in regions], [r[1] for r in regions])
        regions, starts, ends = indexed[key]
        return regions[bisect.bisect_right(ends, lo) : bisect.bisect_left(starts, hi)]

    for node in nodes:
        for read in node["reads"]:
            candidates = [w for w in writers[read["buffer"]] if w["producer"] != node["id"]]
            grouped = defaultdict(list)
            for lo, hi in read["entry_regions"]:
                relevant = [w for w in candidates if intersecting(w, lo, hi)]
                cuts = sorted(
                    {
                        lo,
                        hi,
                        *[
                            v
                            for w in relevant
                            for r in intersecting(w, lo, hi)
                            if overlap((lo, hi), r)
                            for v in overlap((lo, hi), r)
                        ],
                    }
                )
                for start, end in zip(cuts, cuts[1:], strict=False):
                    ws = [w for w in relevant if intersecting(w, start, end)]
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
                        conflict = bool(intersecting(latest, start, end, "conflicts"))
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
