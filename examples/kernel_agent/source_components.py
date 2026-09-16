"""Single-turn, source-only component descriptions for component credit.

The output deliberately records syntactic reachability rather than execution or
semantic equivalence.  It supplies conservative unit signatures to the reward
layer; matching units across turns and allocating any reward are separate work.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import time
from typing import Any

from tools.data.trajectory_structure.optimization_pilot import inspect_kernel, profile_matches
from tools.data.trajectory_structure.optimization_strategies import inspect_program
from tools.data.trajectory_structure.structure import analyze_response

SCHEMA = "kernel-source-components/v1"
_NATIVE_SECTIONS = frozenset({"CUDA_KERNELS", "APPLY_BINDINGS"})
# This is deliberately narrower than the source-feature vocabulary.  In
# particular, cublasLtMatmulDescSetAttribute is configuration *for* a compute
# call, not a separately creditable computation.
_LIBRARY_COMPUTE = re.compile(
    r"^(?:"
    r"cublas(?:[SDCHZ]?gemm(?:Ex|Batched(?:Ex)?|StridedBatched(?:Ex)?)?|"
    r"Gemm(?:Ex|Batched(?:Ex)?|StridedBatched(?:Ex)?)?|LtMatmul|"
    r"[SDCHZ]?gemv(?:Batched|StridedBatched)?)|"
    r"cudnn(?:Convolution(?:Forward|Backward(?:Data|Filter))|"
    r"Activation(?:Forward|Backward)|Pooling(?:Forward|Backward)|Softmax(?:Forward|Backward)|"
    r"BatchNormalization(?:Forward(?:Training|Inference)|Backward)|Normalization(?:Forward|Backward))|"
    r"cutlass::gemm(?:::\w+)*"
    r")$",
    re.IGNORECASE,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _native_context(snapshot: dict[str, Any], sections: set[str]) -> list[dict[str, Any]]:
    """Keep file-level tokens and macros: function bodies are represented separately."""
    contexts = []
    for component in snapshot["components"]:
        if component["section"] not in sections:
            continue
        # analyze_native masks both functions and macros out of file_context.
        # Retain macro tokens explicitly because they are compilation inputs.
        if component["kind"] in {"file_context", "macro"}:
            contexts.append(
                {
                    "section": component["section"],
                    "kind": component["kind"],
                    "qualified_name": component["qualified_name"],
                    "tokens": component.get("_tokens", []),
                }
            )
    return sorted(contexts, key=lambda item: (item["section"], item["kind"], item["qualified_name"]))


def _components_by_id(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {component["id"]: component for component in snapshot["components"]}


def _source_paths(snapshot: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    """Follow only unique static candidates from ``ModelNew.forward``.

    Resolution is name-only in the underlying parser.  Consequently an
    ambiguous or external edge is not treated as source-use confirmation.
    """
    components = _components_by_id(snapshot)
    roots = [
        component["id"]
        for component in snapshot["components"]
        if component["section"] == "MODEL_NEW"
        and component["kind"] == "python_function"
        and component["name"] == "forward"
        and component.get("scope", [])
        and component["scope"][-1] == "ModelNew"
    ]
    unknowns: list[str] = []
    if not roots:
        unknowns.append("modelnew_forward_not_parsed")
    elif len(roots) > 1:
        unknowns.append("modelnew_forward_ambiguous")
        roots = []

    outgoing: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for edge in snapshot["calls"]:
        outgoing[edge["from"]].append(edge)

    paths: dict[str, list[str]] = {}
    pending = [(root, [root]) for root in roots]
    while pending:
        current, path = pending.pop()
        if current in paths:
            continue
        paths[current] = path
        for edge in outgoing.get(current, []):
            targets = edge.get("candidate_targets", [])
            if edge.get("resolution") != "unique_name_candidate" or len(targets) != 1:
                continue
            target = targets[0]
            if target in components and target not in paths:
                pending.append((target, [*path, target]))
    return paths, unknowns


def _native_helper_closure(
    component: dict[str, Any], edges_by_source: dict[str, list[dict[str, Any]]], components: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Collect directly reachable native helper definitions for signature scope."""
    helpers: list[dict[str, Any]] = []
    seen = {component["id"]}
    pending = [component["id"]]
    while pending:
        current = pending.pop()
        for edge in edges_by_source.get(current, []):
            if edge.get("resolution") != "unique_name_candidate" or len(edge.get("candidate_targets", [])) != 1:
                continue
            target = components.get(edge["candidate_targets"][0])
            if (
                target is None
                or target["id"] in seen
                or target["section"] not in _NATIVE_SECTIONS
                or target["kind"] != "function"
            ):
                continue
            seen.add(target["id"])
            helpers.append(target)
            pending.append(target["id"])
    return sorted(helpers, key=lambda item: (item["section"], item["qualified_name"], item["line"]))


def _launch_interfaces(
    component: dict[str, Any], incoming: dict[str, list[dict[str, Any]]], components: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Describe host launch sites without treating them as reward units."""
    interfaces = []
    for edge in incoming.get(component["id"], []):
        if edge.get("kind") != "kernel_launch":
            continue
        owner = components.get(edge["from"])
        if owner is None:
            continue
        interfaces.append(
            {
                "owner": owner["qualified_name"],
                "owner_section": owner["section"],
                "owner_valid_syntax": owner["valid_syntax"],
                "owner_signature": owner.get("_tokens", []),
                "call": edge.get("expression"),
                "arguments": edge.get("arguments", []),
                "launch_config": edge.get("launch_config"),
            }
        )
    return sorted(interfaces, key=lambda item: _digest(item))


def _profile_entries(profiles: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [entry for entry in profiles or [] if isinstance(entry, dict) and isinstance(entry.get("name"), str)]


def _library_profile_matches(target: str, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Runtime profilers may include the API spelling or a decorated implementation
    # name.  Match a token boundary rather than a broad substring.
    pattern = re.compile(r"(?<![\w:])" + re.escape(target) + r"(?:\b|<|\()")
    return [entry for entry in profiles if pattern.search(entry["name"])]


def _unit_id(kind: str, component: dict[str, Any], ordinal: int = 0) -> str:
    return f"{kind}:{_digest([component['section'], component['qualified_name'], component['line'], ordinal])[:24]}"


def _unit_unknowns(component: dict[str, Any], use_observed: bool) -> list[str]:
    unknowns = []
    if not component.get("valid_syntax", False):
        unknowns.append("component_syntax_invalid")
    if not use_observed:
        unknowns.append("use_not_observed")
    return unknowns


def _signature_unknowns(
    component: dict[str, Any],
    helpers: list[dict[str, Any]],
    launch_interfaces: list[dict[str, Any]],
    snapshot: dict[str, Any],
    edges_by_source: dict[str, list[dict[str, Any]]],
    components: dict[str, dict[str, Any]],
) -> list[str]:
    """Report only dependencies needed to name this source implementation."""
    unknowns = []
    if not component.get("valid_syntax", False):
        unknowns.append("component_syntax_invalid")
    if not snapshot["sections"].get(component["section"], {}).get("syntax_valid", False):
        unknowns.append("component_compile_context_syntax_invalid")
    if any(not helper.get("valid_syntax", False) for helper in helpers):
        unknowns.append("helper_syntax_invalid")
    if any(not interface.get("owner_valid_syntax", False) for interface in launch_interfaces):
        unknowns.append("launch_interface_syntax_invalid")
    # A local helper candidate participates in implementation identity. The
    # parser deliberately does not type-resolve overloads, so do not pretend an
    # ambiguous candidate is an external intrinsic and omit its definition.
    pending = [component["id"], *(helper["id"] for helper in helpers)]
    for source in pending:
        for edge in edges_by_source.get(source, []):
            local_helpers = [
                components[target]
                for target in edge.get("candidate_targets", [])
                if target in components
                and components[target]["section"] in _NATIVE_SECTIONS
                and components[target]["kind"] == "function"
            ]
            if local_helpers and (
                edge.get("resolution") != "unique_name_candidate" or len(edge.get("candidate_targets", [])) != 1
            ):
                unknowns.append("local_helper_resolution_ambiguous_or_uncertain")
    return unknowns


def _signature(
    *,
    kind: str,
    component: dict[str, Any],
    contexts: list[dict[str, Any]],
    helpers: list[dict[str, Any]],
    launch_interfaces: list[dict[str, Any]],
    call: dict[str, Any] | None = None,
) -> str | None:
    if not component.get("valid_syntax", False):
        return None
    # Preserve names, literals, operator order and full token streams.  This is
    # intentionally not alpha- or algebraically-normalized.
    payload = {
        "schema": SCHEMA,
        "kind": kind,
        "implementation": component.get("_tokens", []),
        "helpers": [helper.get("_tokens", []) for helper in helpers],
        "native_compile_context": contexts,
        "launch_interfaces": launch_interfaces,
        "library_call": (
            None
            if call is None
            else {
                "target": call.get("target"),
                "expression": call.get("expression"),
                "arguments": call.get("arguments", []),
            }
        ),
    }
    return _digest(payload)


def _features_for_snapshot(snapshot: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    """Return one feature set per concrete component, never per strategy label."""
    features: dict[str, set[str]] = collections.defaultdict(set)
    evidence: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for component in snapshot["components"]:
        if component["section"] == "CUDA_KERNELS" and component["kind"] == "kernel" and component.get("valid_syntax"):
            parsed = inspect_kernel(component)
            if parsed is None:
                continue
            for label, rows in parsed["features"].items():
                features[component["id"]].add(label)
                evidence[component["id"]].extend(
                    {"feature": label, "reason": row.get("reason"), "line": row.get("line")} for row in rows[:4]
                )
    for row in inspect_program(snapshot):
        for component in snapshot["components"]:
            if row["section"] == component["section"] and row["component"] == component["qualified_name"]:
                features[component["id"]].add(row["feature"])
                evidence[component["id"]].append(
                    {"feature": row["feature"], "reason": row.get("reason"), "line": row.get("line")}
                )
    return {key: sorted(value) for key, value in features.items()}, evidence


def _snapshot(kernel_code: str) -> tuple[dict[str, Any], bool]:
    """Parse a full agent response, or retain standalone CUDA as residual source."""
    snapshot = analyze_response(kernel_code)
    if snapshot["selection_mode"] != "empty" or not kernel_code.strip():
        return snapshot, False
    # The public API also accepts a bare CUDA source string. It has no invented
    # Python/FFI path, so each unit stays unobserved unless a profile names it.
    wrapped = "\n\n".join(
        [
            f"### CUDA_KERNELS\n```cpp\n{kernel_code}\n```",
            "### APPLY_BINDINGS\n```cpp\nvoid source_components_placeholder() {}\n```",
            "### MODEL_NEW\n```python\nclass ModelNew:\n    pass\n```",
        ]
    )
    return analyze_response(wrapped), True


def analyze_source_components(kernel_code: str, profiles: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Describe source units from one candidate response without executing it.

    ``profiles`` is optional existing profiler metadata.  A profile-name match can
    mark a unit used, but does not change its source signature or infer dataflow.
    """
    start = time.monotonic()
    if not isinstance(kernel_code, str):
        return {
            "schema": SCHEMA,
            "units": [],
            "unknowns": ["kernel_code_not_string"],
            "wall_seconds": time.monotonic() - start,
        }

    snapshot, raw_cuda_source = _snapshot(kernel_code)
    components = _components_by_id(snapshot)
    paths, unknowns = _source_paths(snapshot)
    if raw_cuda_source:
        unknowns.append("raw_cuda_source_without_ffi_path")
    if snapshot["selection_mode"] != "last_complete_group":
        unknowns.append(f"source_selection:{snapshot['selection_mode']}")
    for section, detail in snapshot["sections"].items():
        if not detail.get("syntax_valid", False):
            unknowns.append(f"section_syntax_invalid:{section}")
    edges_by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for edge in snapshot["calls"]:
        edges_by_source[edge["from"]].append(edge)
        for target in edge.get("candidate_targets", []):
            incoming[target].append(edge)
    profile_rows = _profile_entries(profiles)
    features, evidence = _features_for_snapshot(snapshot)
    units: list[dict[str, Any]] = []

    for component in snapshot["components"]:
        if component["section"] != "CUDA_KERNELS" or component["kind"] != "kernel":
            continue
        profile_hits = profile_matches(component["name"], profile_rows)
        source_path = paths.get(component["id"])
        use_observed = bool(source_path or profile_hits)
        helpers = _native_helper_closure(component, edges_by_source, components)
        launches = _launch_interfaces(component, incoming, components)
        identity_unknowns = _signature_unknowns(component, helpers, launches, snapshot, edges_by_source, components)
        contexts = _native_context(
            snapshot,
            {
                component["section"],
                *(helper["section"] for helper in helpers),
                *(launch["owner_section"] for launch in launches),
            },
        )
        unit = {
            "id": _unit_id("kernel", component),
            "kind": "kernel",
            "features": features.get(component["id"], []),
            "signature": (
                _signature(
                    kind="kernel",
                    component=component,
                    contexts=contexts,
                    helpers=helpers,
                    launch_interfaces=launches,
                )
                if not identity_unknowns
                else None
            ),
            "unknowns": sorted(set(_unit_unknowns(component, use_observed) + identity_unknowns)),
            "use_observed": use_observed,
            "evidence": {
                "section": component["section"],
                "qualified_name": component["qualified_name"],
                "line": component["line"],
                "end_line": component["end_line"],
                "valid_syntax": component["valid_syntax"],
                "source_path": source_path,
                "profile_names": [row["name"] for row in profile_hits],
                "features": evidence.get(component["id"], []),
                "helpers": [helper["qualified_name"] for helper in helpers],
                "launch_interfaces": [entry["owner"] for entry in launches],
            },
        }
        units.append(unit)

    for component in snapshot["components"]:
        if component["section"] not in _NATIVE_SECTIONS or component["kind"] != "function":
            continue
        helpers = _native_helper_closure(component, edges_by_source, components)
        for ordinal, call in enumerate(component.get("calls", [])):
            if not _LIBRARY_COMPUTE.search(call["target"]):
                continue
            profile_hits = _library_profile_matches(call["target"], profile_rows)
            source_path = paths.get(component["id"])
            use_observed = bool(source_path or profile_hits)
            identity_unknowns = _signature_unknowns(component, helpers, [], snapshot, edges_by_source, components)
            contexts = _native_context(snapshot, {component["section"], *(helper["section"] for helper in helpers)})
            units.append(
                {
                    "id": _unit_id("library", component, ordinal),
                    "kind": "library",
                    "features": sorted(set(features.get(component["id"], [])) | {"library_offload"}),
                    "signature": (
                        _signature(
                            kind="library",
                            component=component,
                            contexts=contexts,
                            helpers=helpers,
                            launch_interfaces=[],
                            call=call,
                        )
                        if not identity_unknowns
                        else None
                    ),
                    "unknowns": sorted(set(_unit_unknowns(component, use_observed) + identity_unknowns)),
                    "use_observed": use_observed,
                    "evidence": {
                        "section": component["section"],
                        "qualified_name": component["qualified_name"],
                        "line": call["line"],
                        "valid_syntax": component["valid_syntax"],
                        "target": call["target"],
                        "arguments": call.get("arguments", []),
                        "source_path": source_path,
                        "profile_names": [row["name"] for row in profile_hits],
                        "features": evidence.get(component["id"], []),
                        "helpers": [helper["qualified_name"] for helper in helpers],
                    },
                }
            )

    units.sort(key=lambda unit: (unit["kind"], unit["id"]))
    return {
        "schema": SCHEMA,
        "units": units,
        "unknowns": sorted(set(unknowns)),
        "wall_seconds": time.monotonic() - start,
    }
