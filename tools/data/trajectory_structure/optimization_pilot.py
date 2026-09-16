"""Task-independent source-pattern pilot joined to existing correct-only timings.

No candidate execution, training reward, whole-program semantic equivalence, or
causal attribution. Rules inspect AST nodes, not problem names or model prose.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

from tools.data.trajectory_candidates.screen import observation_state, sha, version_hash

from .optimization_strategies import STRATEGIES, extend_kernel_features
from .structure import PARSERS, callable_name, native_tokens, text, walk

RULES = {name: rule["template"] for name, rule in STRATEGIES.items() if rule["scope"] in {"kernel", "native"}}
LOOPS = {"for_statement", "while_statement", "do_statement"}
VECTOR = re.compile(
    r"\b(?:(?:float|double|int|uint|short|ushort|char|uchar|longlong|ulonglong)[24]|(?:__)?half2|(?:__)?nv_bfloat162)\b"
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def children_of_type(node, kind):
    return [n for n in walk(node) if n.type == kind]


def strip_comments(source):
    root = PARSERS["CUDA_KERNELS"].parse(source.encode()).root_node
    data = bytearray(source.encode())
    for node in walk(root):
        if node.type == "comment":
            for i in range(node.start_byte, node.end_byte):
                if data[i] not in (10, 13):
                    data[i] = 32
    return data.decode()


def inspect_kernel(component):
    """Return auditable source features and a strict alpha-normalized structure.

    Known API names, arithmetic, literals, field selectors and global identifiers
    stay intact. Repeated local declarations disable alpha renaming to avoid
    conflating shadowed bindings. This is not a semantic canonical form.
    """
    source = component["source"]
    clean = strip_comments(source)
    masked = re.sub(r"(?m)^\s*#pragma[^\n]*", lambda m: " " * len(m[0]), clean)
    root = PARSERS["CUDA_KERNELS"].parse(masked.encode()).root_node
    fn = next((n for n in walk(root) if n.type == "function_definition"), None)
    if fn is None or fn.has_error:
        return None
    body = fn.child_by_field_name("body")
    nodes = list(walk(body))
    evidence = collections.defaultdict(list)

    def add(kind, node, explanation):
        evidence[kind].append(
            {"line": component["line"] + node.start_point.row, "code": text(node)[:900], "reason": explanation}
        )

    calls = children_of_type(body, "call_expression")
    for n in calls:
        target = text(n.child_by_field_name("function"))
        if target.startswith("__shfl"):
            add("warp_shuffle", n, "warp shuffle intrinsic invocation")

    # Detect declarations, then actual indexed accesses; declarations alone do not count.
    shared = set()
    vector_pointers = set()
    bindings = []
    params = next(n for n in walk(fn) if n.type == "parameter_list")
    for n in [*walk(params), *nodes]:
        if n.type not in {"declaration", "parameter_declaration"}:
            continue
        for decl in n.children_by_field_name("declarator"):
            name = callable_name(decl)
            if name is None:
                continue
            name = text(name)
            bindings.append(name)
            if "__shared__" in native_tokens(n):
                shared.add(name)
            if VECTOR.search(text(n.child_by_field_name("type"))) and "*" in text(decl):
                vector_pointers.add(name)

    shared_modes = collections.defaultdict(set)
    shared_nodes = {}
    for n in nodes:
        if n.type not in {"subscript_expression", "pointer_expression"}:
            continue
        arg = n.child_by_field_name("argument")
        base = text(arg)
        parent = n.parent
        assignment = parent if parent is not None and parent.type == "assignment_expression" else None
        lhs = assignment.child_by_field_name("left") if assignment else None
        is_write = lhs is not None and lhs.start_byte <= n.start_byte and n.end_byte <= lhs.end_byte
        if base in shared:
            shared_modes[base].add("write" if is_write else "read")
            shared_nodes[base] = n
        # A vector cast must be inside a dereference/index, not merely declared.
        if base in vector_pointers or (arg is not None and VECTOR.search(text(arg)) and "*" in text(arg)):
            add("vector_memory", n, "explicit vector-pointer access in source")
    barrier = any(text(n.child_by_field_name("function")) == "__syncthreads" for n in calls)
    if barrier:
        for name, modes in shared_modes.items():
            if modes == {"read", "write"}:
                add("shared_staging", shared_nodes[name], f"shared array {name} read and written with block barrier")

    for m in re.finditer(r"(?m)^\s*#pragma\s+unroll(?:\s+(\d+))?\s*\n\s*for\s*\(", clean):
        if m[1] not in {"0", "1"}:
            evidence["unroll_request"].append(
                {
                    "line": component["line"] + clean[: m.start()].count("\n"),
                    "code": m[0].strip(),
                    "reason": "explicit loop-unroll request",
                }
            )

    # Narrow, task-independent def/use pattern: a loop-carried scalar, followed
    # by a transformed store. Excludes raw copying of the accumulator and += on
    # output memory. Does not claim all fusion styles are covered.
    for loop in (n for n in nodes if n.type in LOOPS):
        accumulated = set()
        loop_body = loop.child_by_field_name("body")
        if loop_body is None:
            continue
        condition = loop.child_by_field_name("condition")
        counters = set(native_tokens(condition)) if condition is not None else set()
        for a in children_of_type(loop_body, "assignment_expression"):
            left, right = a.child_by_field_name("left"), a.child_by_field_name("right")
            if left is None or right is None or left.type != "identifier":
                continue
            op = text(a.child_by_field_name("operator"))
            name = text(left)
            if name not in counters and (
                op in {"+=", "*=", "-=", "/="} or (op == "=" and name in native_tokens(right))
            ):
                accumulated.add(name)
        for a in (n for n in nodes if n.type == "assignment_expression" and n.start_byte > loop.end_byte):
            left, right = a.child_by_field_name("left"), a.child_by_field_name("right")
            if left is None or right is None or left.type not in {"subscript_expression", "pointer_expression"}:
                continue
            if text(left.child_by_field_name("argument")) in shared:
                continue
            used = accumulated.intersection(native_tokens(right))
            transformed = any(
                n.type in {"binary_expression", "call_expression", "conditional_expression"} for n in walk(right)
            )
            if used and transformed:
                add("reduction_epilogue", a, f"post-loop transformed store of accumulator(s): {sorted(used)}")

    extend_kernel_features(component, fn, clean, evidence)
    mapping = {name: f"local_{i}" for i, name in enumerate(bindings)} if len(set(bindings)) == len(bindings) else {}
    decl = fn.child_by_field_name("declarator")
    name_node = callable_name(decl)
    tokens = []
    for n in walk(root, all_children=True):
        if n.children or not text(n).strip():
            continue
        value = text(n)
        if name_node is not None and n.start_byte == name_node.start_byte and n.end_byte == name_node.end_byte:
            value = "kernel_entry"
        elif n.type == "identifier":
            value = mapping.get(value, value)
        tokens.append((n.type, value))
    # Masking pragmas for parsing must not erase unroll settings from identity.
    structure = [tokens, re.findall(r"(?m)^\s*#pragma[^\n]*", clean), component.get("templates", [])]
    return {"features": dict(evidence), "structure_hash": digest(structure), "alpha_normalized": bool(mapping)}


def profile_matches(name, profiles):
    pattern = re.compile(r"(?<![\w:])" + re.escape(name) + r"\s*(?:<|\()")
    return [p for p in profiles if pattern.search(p.get("name", ""))]


def finite_positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def load_timing(turn, raw):
    state, speed = observation_state(turn)
    if state != "correct_timed":
        return None, state
    env = raw.get("env_result") or (raw.get("metadata") or {}).get("env_result") or {}
    s = env.get("env_state", env)
    ref, kernel = s.get("reference_runtime"), s.get("kernel_runtime")
    if not all(finite_positive(x) for x in (ref, kernel, speed)) or not math.isclose(
        ref / kernel, speed, rel_tol=1e-6
    ):
        return None, "invalid_timing_ratio"
    if s.get("correctness") is not True or s.get("compiled") is not True or s.get("error_message"):
        return None, "raw_result_inconsistent"
    return {
        "speedup": speed,
        "reference_ms": ref,
        "kernel_ms": kernel,
        "profiles": (s.get("metadata") or {}).get("profiling", {}).get("kernels", []),
        "num_perf_trials": (s.get("metadata") or {}).get("num_perf_trials"),
    }, "correct_timed"


def extract_instances(turn, timing):
    components = turn["components"]
    globals_hash = digest(
        sorted(
            c["syntax_hash"]
            for c in components
            if c["section"] == "CUDA_KERNELS" and c["kind"] in {"macro", "file_context"}
        )
    )
    output = []
    for c in components:
        if c["kind"] != "kernel" or not c["valid_syntax"] or c["section"] != "CUDA_KERNELS":
            continue
        if sum(k["name"] == c["name"] and k["kind"] == "kernel" for k in components) != 1:
            continue
        matches = profile_matches(c["name"], timing["profiles"])
        if not matches:
            continue
        parsed = inspect_kernel(c)
        if parsed is None:
            continue
        # Same local structure is only a retention candidate: no launch-argument,
        # helper body, role or whole-program wiring equivalence is asserted.
        for feature, evidence in parsed["features"].items():
            output.append(
                {
                    "feature": feature,
                    "kernel": c["qualified_name"],
                    "line": c["line"],
                    "source": c["source"],
                    "evidence": evidence,
                    "instance_hash": digest([feature, parsed["structure_hash"], globals_hash]),
                    "structure_hash": parsed["structure_hash"],
                    "alpha_normalized": parsed["alpha_normalized"],
                    "profile_names": [p["name"] for p in matches],
                    "profile_cuda_us": sum(p.get("cuda_time_us", 0) for p in matches),
                    "scope": "source_structure_retention_candidate",
                }
            )
    return output


def analyze_trajectory(structure, raws):
    turns, ignored = [], collections.Counter()
    for turn in structure["turns"]:
        raw = raws[turn["turn_idx"]]
        if hashlib.sha256(raw["response"].encode()).hexdigest() != turn["observation"]["response_sha256"]:
            raise ValueError("raw/structure response mismatch")
        timing, status = load_timing(turn, raw)
        if timing is None:
            ignored[status] += 1
            continue
        instances = extract_instances(turn, timing)
        turns.append(
            {
                "turn": turn["turn_idx"] + 1,
                "timing": timing,
                "source_available": bool(turn["complete_sections"] and turn["valid_modelnew"]),
                "program_version": version_hash(turn),
                "instances": instances,
            }
        )
    comparisons = []
    for a, b in zip(turns, turns[1:], strict=False):
        labels_a, labels_b = {i["feature"] for i in a["instances"]}, {i["feature"] for i in b["instances"]}
        same_program = a["program_version"] is not None and a["program_version"] == b["program_version"]
        ref_drift = b["timing"]["reference_ms"] / a["timing"]["reference_ms"] - 1
        comparisons.append(
            {
                "from": a["turn"],
                "to": b["turn"],
                "speedup_before": a["timing"]["speedup"],
                "speedup_after": b["timing"]["speedup"],
                "speedup_ratio": b["timing"]["speedup"] / a["timing"]["speedup"],
                "kernel_time_ratio": a["timing"]["kernel_ms"] / b["timing"]["kernel_ms"],
                "reference_drift": ref_drift,
                "same_program": same_program,
                "new_labels": sorted(labels_b - labels_a),
                "removed_labels": sorted(labels_a - labels_b),
                "stable_reference": abs(ref_drift) <= 0.05,
                "comparable_source": a["source_available"] and b["source_available"],
            }
        )
    best = max(turns, key=lambda t: (t["timing"]["speedup"], -t["turn"])) if turns else None
    origins = []
    if best:
        for unit in best["instances"]:
            if sum(i["instance_hash"] == unit["instance_hash"] for i in best["instances"]) != 1:
                continue
            seen = []
            for t in turns:
                if t["turn"] > best["turn"]:
                    break
                matches = [i for i in t["instances"] if i["instance_hash"] == unit["instance_hash"]]
                if len(matches) == 1:
                    seen.append(t["turn"])
            origins.append(
                {
                    "feature": unit["feature"],
                    "kernel": unit["kernel"],
                    "instance_hash": unit["instance_hash"],
                    "first_correct_observed_turn": min(seen),
                    "best_turn": best["turn"],
                    "observed_turns": seen,
                }
            )
    return {
        "identity": structure["identity"],
        "turns": turns,
        "ignored": dict(ignored),
        "comparisons": comparisons,
        "best_turn": best["turn"] if best else None,
        "origins": origins,
    }


def summarize(results):
    summary = {}
    for model in sorted({r["identity"][0] for r in results}):
        rows = [r for r in results if r["identity"][0] == model]
        timings = [t for r in rows for t in r["turns"]]
        pairs = [p for r in rows for p in r["comparisons"] if p["stable_reference"] and p["comparable_source"]]
        categories = {}
        for label in RULES:
            introductions = [p for p in pairs if label in p["new_labels"] and not p["same_program"]]
            ratios = [p["kernel_time_ratio"] for p in introductions]
            categories[label] = {
                "profile_linked_turns": sum(any(i["feature"] == label for i in t["instances"]) for t in timings),
                "label_introductions": len(ratios),
                "faster_over_5pct": sum(x > 1.05 for x in ratios),
                "slower_over_5pct": sum(x < 1 / 1.05 for x in ratios),
                "median_kernel_time_ratio": statistics.median(ratios) if ratios else None,
            }
        summary[model] = {
            "trajectories": len(rows),
            "correct_timed_trajectories": sum(bool(r["turns"]) for r in rows),
            "at_least_two_correct_timed": sum(len(r["turns"]) >= 2 for r in rows),
            "correct_timed_turns": len(timings),
            "profile_linked_feature_turns": sum(bool(t["instances"]) for t in timings),
            "stable_reference_correct_pairs": len(pairs),
            "earlier_structure_in_best_trajectories": sum(
                any(o["first_correct_observed_turn"] < o["best_turn"] for o in r["origins"]) for r in rows
            ),
            "earlier_non_epilogue_structure_in_best": sum(
                any(
                    o["feature"] != "reduction_epilogue" and o["first_correct_observed_turn"] < o["best_turn"]
                    for o in r["origins"]
                )
                for r in rows
            ),
            "same_program_pairs": sum(p["same_program"] for p in pairs),
            "same_program_over_5pct_variation": sum(
                p["same_program"] and (p["kernel_time_ratio"] > 1.05 or p["kernel_time_ratio"] < 1 / 1.05)
                for p in pairs
            ),
            "categories": categories,
            "ignored_turns": dict(sum((collections.Counter(r["ignored"]) for r in rows), collections.Counter())),
        }
    return summary


def run(root, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    manifest = json.loads((root / "manifest.json").read_text())
    inputs = {i["name"]: i for i in manifest["inputs"]}
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("optimization_strategies.py"),
        Path(__file__).with_name("structure.py"),
        Path("tools/data/trajectory_candidates/screen.py"),
    ]
    hashes = {str(p.resolve()): sha(p) for p in code_paths}
    index_path = root / "trajectory_index.json"
    hashes[str(index_path.resolve())] = sha(index_path)
    hashes[str((root / "manifest.json").resolve())] = sha(root / "manifest.json")
    results = []
    for entry in json.loads(index_path.read_text()):
        source = root / entry["path"]
        hashes[str(source.resolve())] = sha(source)
        structure = json.loads(source.read_text())
        model, *_, group = structure["identity"]
        raw_path = Path(inputs[model]["path"]) / f"g{int(group):03}.json"
        raw_hash = sha(raw_path)
        if raw_hash != inputs[model]["files"][raw_path.name]:
            raise ValueError(f"input changed: {raw_path}")
        hashes[str(raw_path.resolve())] = raw_hash
        raws = {
            r.get("turn_idx", (r.get("metadata") or {}).get("turn_idx")): r for r in json.loads(raw_path.read_text())
        }
        result = analyze_trajectory(structure, raws)
        result["source_path"] = str(source.resolve())
        result["artifact"] = source.name
        (output / "trajectories" / source.name).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        results.append(result)
        if len(results) % 100 == 0:
            print(f"processed {len(results)} trajectories", flush=True)
    summary = summarize(results)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    selection = []
    for feature in RULES:
        for direction in ("faster", "slower"):
            options = []
            for r in results:
                for p in r["comparisons"]:
                    ratio = p["kernel_time_ratio"]
                    if (
                        feature in p["new_labels"]
                        and p["stable_reference"]
                        and p["comparable_source"]
                        and not p["same_program"]
                        and (ratio > 1.05 if direction == "faster" else ratio < 1 / 1.05)
                    ):
                        options.append(
                            {
                                "feature": feature,
                                "direction": direction,
                                "identity": r["identity"],
                                "artifact": r["artifact"],
                                "comparison": p,
                            }
                        )
            options.sort(key=lambda r: digest(["optimization-pilot-v1", r["identity"], r["comparison"]["to"]]))
            selection.extend(options[:2])
    (output / "review_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2))
    changed = [path for path, original in hashes.items() if sha(path) != original]
    protocol = {
        "rules": RULES,
        "score": "raw speedup, correct/compiled/nondecoy/error-free only",
        "selection": "all 800 existing diagnostic-pool trajectories; no new GPU timing",
        "comparison": "successive eligible correct turns, possibly nonadjacent; ref drift <=5%",
        "timing_noise": "5% screening threshold, not a confidence interval",
        "retention": "unique alpha-normalized kernel-source structure plus CUDA global context; candidate only",
        "limitations": [
            "not unseen-task validation",
            "no per-feature causal effect",
            "no automatic general fusion/coalescing proof",
            "no launch/helper/role equivalence",
            "profiling is a separate invocation from timing trials",
        ],
        "hashes": hashes,
        "code_and_inputs_unchanged": not changed,
        "changed_paths": changed,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    if changed:
        raise RuntimeError(f"inputs changed: {changed}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.structure_root, args.output)


if __name__ == "__main__":
    main()
