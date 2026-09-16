"""Source-level state facts and a join to observed FFI/CUDA/state inventories.

State declarations are syntactic candidates. Runtime observations confirm concrete
calls and storage identities, not C++ read/write sets or output dependency.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

from tools.data.trajectory_candidates.screen import decision_facts

from .structure import PARSERS, callable_name, text, walk


def owner_at(turn, section, line):
    candidates = [
        c
        for c in turn["components"]
        if c["section"] == section and c["kind"] != "file_context" and c["line"] <= line <= c["end_line"]
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["end_line"] - c["line"])["id"]


def state_facts(turn):
    facts = []
    for section in ("CUDA_KERNELS", "APPLY_BINDINGS"):
        source = turn["sections"].get(section, {}).get("source", "")
        tree = PARSERS[section].parse(source.encode())
        for node in walk(tree.root_node):
            if node.type not in {"declaration", "field_declaration"}:
                continue
            ancestors = []
            parent = node.parent
            while parent is not None:
                ancestors.append(parent.type)
                parent = parent.parent
            local = "function_definition" in ancestors
            static = any(
                child.type == "storage_class_specifier" and text(child) == "static" for child in node.children
            )
            if local and not static:
                continue
            for decl in node.children_by_field_name("declarator"):
                if any(x.type == "function_declarator" for x in walk(decl)):
                    continue
                name = text(callable_name(decl))
                if not name:
                    continue
                line = node.start_point.row + 1
                facts.append(
                    {
                        "kind": "native_state_declaration",
                        "section": section,
                        "line": line,
                        "owner_component": owner_at(turn, section, line),
                        "name": name,
                        "type": text(node.child_by_field_name("type")),
                        "storage": (
                            "function_static"
                            if local
                            else "object_field" if node.type == "field_declaration" else "file_scope"
                        ),
                        "declaration": text(node),
                        "execution_status": "static_candidate",
                    }
                )
    python = turn["sections"].get("MODEL_NEW", {}).get("source", "")
    try:
        tree = ast.parse(python)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    facts.append(
                        {
                            "kind": "python_instance_assignment",
                            "section": "MODEL_NEW",
                            "line": node.lineno,
                            "owner_component": owner_at(turn, "MODEL_NEW", node.lineno),
                            "name": target.attr,
                            "value": ast.unparse(node.value) if node.value is not None else None,
                            "execution_status": "static_candidate",
                        }
                    )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"register_buffer", "register_parameter"}
        ):
            facts.append(
                {
                    "kind": "python_registered_state",
                    "section": "MODEL_NEW",
                    "line": node.lineno,
                    "owner_component": owner_at(turn, "MODEL_NEW", node.lineno),
                    "call": ast.unparse(node),
                    "execution_status": "static_candidate",
                }
            )
    for call in turn["calls"]:
        name = call["target"]
        if re.match(r"^(cuda(?:Malloc|Free)|cublas\w*(?:Create|Destroy)|cudnn(?:Create|Destroy))", name):
            facts.append(
                {
                    "kind": "native_resource_operation",
                    "owner_component": call["from"],
                    "section": next(c["section"] for c in turn["components"] if c["id"] == call["from"]),
                    "line": call["line"],
                    "target": name,
                    "arguments": call.get("arguments"),
                    "execution_status": "static_candidate",
                }
            )
    return {
        "state_facts": facts,
        "configuration_facts": decision_facts(turn),
        "semantics": "source_declarations_and_literal_calls_not_runtime_state_or_semantic_equivalence",
    }


def runtime_inventory(turn, observation):
    by_export = {}
    by_call = {c["call_id"]: c for c in observation["calls"]}
    activities = collections.defaultdict(list)
    for a in observation.get("cuda_activities", []):
        if a["call_id"] is not None:
            activities[a["call_id"]].append(a)
    for call in observation["calls"]:
        export = call["export"]
        item = by_export.setdefault(
            export,
            {
                "calls": 0,
                "phases": collections.Counter(),
                "python_components": set(),
                "activities": collections.Counter(),
                "launch_apis": set(),
            },
        )
        item["calls"] += 1
        item["phases"][call["phase"]] += 1
        caller = call["caller"]
        if caller["file"].endswith("/model_new.py"):
            owner = owner_at(turn, "MODEL_NEW", caller["line"])
            if owner is not None:
                item["python_components"].add(owner)
        for a in activities[call["call_id"]]:
            item["activities"][(a["kind"], a["name"])] += 1
            item["launch_apis"].update(a["launch_apis"])
    for item in by_export.values():
        item["python_components"] = sorted(item["python_components"])
        item["launch_apis"] = sorted(item["launch_apis"])
        item["phases"] = dict(item["phases"])
        item["activities"] = [
            {"kind": kind, "name": name, "count": count} for (kind, name), count in item["activities"].most_common()
        ]
    slots = collections.defaultdict(list)
    for snapshot in observation["snapshots"]:
        for leaf in snapshot["model_state"]["leaves"]:
            if leaf["kind"] == "tensor":
                slots[json.dumps(leaf["path"], ensure_ascii=False)].append(
                    {
                        "phase": snapshot["phase"],
                        "label": snapshot["label"],
                        **{
                            k: leaf.get(k)
                            for k in ["storage_id", "tensor_id", "shape", "stride", "storage_nbytes", "torch_version"]
                        },
                    }
                )
    state_slots = []
    for path, observations in slots.items():
        storage_ids = {o["storage_id"] for o in observations if o["storage_id"] is not None}
        argument_calls = []
        for call in by_call.values():
            if any(
                x.get("storage_id") in storage_ids
                for x in call.get("args_before", {}).get("leaves", [])
                if x["kind"] == "tensor"
            ):
                argument_calls.append(call["call_id"])
        state_slots.append(
            {
                "path": json.loads(path),
                "observations": observations,
                "storage_ids": sorted(storage_ids),
                "passed_as_ffi_argument_calls": argument_calls,
                "read_write_or_necessity": "unknown",
            }
        )
    return {
        "exports": by_export,
        "python_tensor_state": state_slots,
        "capture": observation.get("capture"),
        "snapshot_limits_reached": any(s["model_state"]["at_item_limit"] for s in observation["snapshots"]),
        "native_global_state": "unobserved_not_reconstructed_from_tensor_pointers",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for path in sorted(args.runtime_root.glob("*/observation.json")):
        observation = json.loads(path.read_text())
        identity = observation.get("source_identity", {}).get("identity")
        if not isinstance(identity, dict):
            continue
        case = f"{identity['model']}_g{identity['group']:03}_t{identity['turn']}"
        payload = json.loads((args.payload_root / (case + ".json")).read_text())
        expected = hashlib.sha256(payload["custom_code"].encode()).hexdigest()
        if expected != observation["source_identity"]["custom_code_sha256"]:
            raise ValueError(f"runtime/source identity mismatch: {path}")
        structure_path = args.structure_root / "trajectories" / (identity["trajectory_id"] + ".json")
        if hashlib.sha256(structure_path.read_bytes()).hexdigest() != payload["provenance"]["structure_sha256"]:
            raise ValueError(f"saved structure changed since replay preparation: {structure_path}")
        structure = json.loads(structure_path.read_text())
        turn = next(t for t in structure["turns"] if t["turn_idx"] == identity["turn"] - 1)
        joined = {
            "identity": identity,
            "runtime_source_sha256": expected,
            **state_facts(turn),
            "runtime": runtime_inventory(turn, observation),
        }
        (args.output / (path.parent.name + ".json")).write_text(json.dumps(joined, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "case": path.parent.name,
                    "state_facts": len(joined["state_facts"]),
                    "configuration_facts": len(joined["configuration_facts"]),
                    "runtime_exports": len(joined["runtime"]["exports"]),
                    "python_tensor_slots": len(joined["runtime"]["python_tensor_state"]),
                }
            )
        )


if __name__ == "__main__":
    main()
