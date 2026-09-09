"""Cross-turn coarse call correspondence from regional interfaces and dataflow.

Kernel names and implementation IDs never nominate matches. Version identifiers
are compared only after the observed interface/dependency correspondence exists.
"""

from collections import defaultdict
from .extract import canonical, digest


def anchors(graph):
    buffers = {a["id"]: a for a in graph["buffers"]}
    result = {}
    incoming = defaultdict(list)
    for edge in graph["edges"]:
        incoming[edge["target"]].append(edge)
    for node in graph["nodes"]:
        roots = set()
        for edge in incoming[node["id"]]:
            source = edge["source"]
            if source and source.startswith("initial:"):
                roots.update(
                    r for r in buffers[edge["buffer"]]["roles"] if r.startswith(("input:", "parameter:", "state:"))
                )
            elif source in result:
                roots.update(result[source])
        result[node["id"]] = sorted(roots)
    return result


def describe(graph):
    roots = anchors(graph)
    buffers = {a["id"]: a for a in graph["buffers"]}
    incoming = defaultdict(list)
    nodes = {n["id"]: n for n in graph["nodes"]}
    for edge in graph["edges"]:
        incoming[edge["target"]].append(edge)
    descriptions = {}
    for node in graph["nodes"]:

        def buffer_role(storage, node_id=node["id"]):
            stable = sorted(r for r in buffers[storage]["roles"] if r.startswith(("input:", "parameter:", "state:")))
            if stable:
                return {"boundary_roles": stable}
            producers = []
            for edge in incoming[node_id]:
                if edge["buffer"] == storage and edge["source"] in nodes:
                    parent = nodes[edge["source"]]
                    producers.append({"kind": parent.get("operation", parent["kind"]), "roots": roots[parent["id"]]})
            return {"temporary_bytes": buffers[storage]["bytes"], "producers": sorted(producers, key=canonical)}

        def port(access):
            return {
                "role": buffer_role(access["buffer"]),
                "regions": access["regions"],
                "bytes": buffers[access["buffer"]]["bytes"],
                "port": access["port"],
            }

        interface = {
            "kind": node.get("operation", node["kind"]),
            "reads": sorted([port(a) for a in node["reads"]], key=canonical),
            # Output/temporary role may change when a consumer is added.
            "writes": sorted(
                [
                    {"regions": a["regions"], "bytes": buffers[a["buffer"]]["bytes"], "port": a["port"]}
                    for a in node["writes"]
                ],
                key=canonical,
            ),
            "roots": roots[node["id"]],
        }
        configuration = dict(node["configuration"])
        if "arguments" in configuration:
            configuration["arguments"] = [
                {
                    **{k: v for k, v in a.items() if k != "buffer"},
                    **({"binding": buffer_role(a["buffer"])} if "buffer" in a else {}),
                }
                for a in configuration["arguments"]
            ]
        if node["kind"] == "library":
            configuration["api"] = node["evidence"]["api"]
        descriptions[node["id"]] = {"interface": interface, "configuration": configuration, "key": digest(interface)}
    return descriptions


def compare(left, right):
    keys = ("task", "input_signature", "environment_signature")
    absent = [k for k in keys if k not in left.get("context", {}) or k not in right.get("context", {})]
    changed = [k for k in keys if k not in absent and left["context"][k] != right["context"][k]]
    result = {
        "schema": "coarse-call-correspondence/v1",
        "matches": [],
        "ambiguous": [],
        "context_missing": absent,
        "context_changed": changed,
        "training_state_equivalence": False,
        "source_similarity_used": False,
        "creative_origin_proven": False,
    }
    if absent or changed:
        return result
    a, b = describe(left), describe(right)
    an, bn = ({n["id"]: n for n in g["nodes"]} for g in (left, right))
    groups = defaultdict(lambda: {"left": [], "right": []})
    for side, descriptions in (("left", a), ("right", b)):
        for key, d in descriptions.items():
            groups[d["key"]][side].append(key)
    matched_left, matched_right = set(), set()
    for group in groups.values():
        if not group["left"] or not group["right"]:
            continue
        if len(group["left"]) != 1 or len(group["right"]) != 1:
            result["ambiguous"].append(group)
            continue
        x, y = group["left"][0], group["right"][0]
        assert a[x]["interface"] == b[y]["interface"]
        cfg_keys = set(a[x]["configuration"]) | set(b[y]["configuration"])
        differences = {
            k: {"before": a[x]["configuration"].get(k), "after": b[y]["configuration"].get(k)}
            for k in sorted(cfg_keys)
            if a[x]["configuration"].get(k) != b[y]["configuration"].get(k)
        }
        impl_same = an[x]["implementation"] == bn[y]["implementation"]
        incomplete = not an[x]["footprint_complete"] or not bn[y]["footprint_complete"]
        opaque_config = any(
            any(u.startswith("opaque_") and u.endswith("_configuration") for u in n["unknowns"])
            for n in (an[x], bn[y])
        )
        ambiguous_effects = any("write_version_ambiguous" in n["unknowns"] for n in (an[x], bn[y]))
        if incomplete:
            relation = "partial_observed_correspondence"
        elif ambiguous_effects:
            relation = "same_observed_call_effects_ambiguous"
        elif not an[x]["reads"] and not an[x]["writes"]:
            relation = "no_observed_data_effect_call_correspondence"
        elif opaque_config:
            relation = "same_regional_structure_configuration_unknown"
        elif differences:
            relation = "configuration_changed"
        elif not impl_same:
            relation = "implementation_changed"
        else:
            relation = "retained_observed_component"
        result["matches"].append(
            {
                "before": x,
                "after": y,
                "relation": relation,
                "configuration_changes": differences,
                "implementation_same": impl_same,
                "interface_evidence": a[x]["interface"],
                "unknowns_before": an[x]["unknowns"],
                "unknowns_after": bn[y]["unknowns"],
                "meaning": "observed call implementation/interface correspondence, not historical invention",
            }
        )
        matched_left.add(x)
        matched_right.add(y)
    result["unmatched_before"] = sorted(set(a) - matched_left)
    result["unmatched_after"] = sorted(set(b) - matched_right)
    result["unmatched_meaning"] = (
        "no reliable one-to-one interface match; fusion/split/replacement/unknown remain distinct possibilities"
    )
    return result
