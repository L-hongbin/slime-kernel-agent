"""Deterministic stratified and Kimi sampling for the CSP-DAG release review."""

from __future__ import annotations

import collections
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.data.synthesize.review_only.csp_dag_review_common import (
    ADDITIVE_LINEAGE_FIELDS,
    LANES,
    REQUIRED_FAMILIES,
    SEED,
    _ast_hash,
    _canonical,
    _class_count,
    _code,
    _code_bucket,
    _exact_file_binding,
    _histogram,
    _normalized_for_schema,
    _operator_bucket,
    _ref_hash,
    _row_hash,
    _sha256_file,
    _uuid,
)


def _feature(lane: str, row: Mapping[str, Any], manifest: Mapping[str, Any], root: str) -> dict[str, Any]:
    code = _code(row)
    families = set(manifest.get("actual_low_level_families", []))
    return {
        "lane": lane,
        "uuid": _uuid(row),
        "root_base_uuid": root,
        "reference_sha256": _ref_hash(code),
        "normalized_ast_sha256": _ast_hash(code),
        "families": sorted(families),
        "code_bucket": _code_bucket(code),
        "operator_bucket": _operator_bucket(manifest),
        "single_model_policy": (
            _class_count(manifest) == 1
            and manifest.get("top_level_class_policy") == "single_model_only"
            and manifest.get("semantic_multiclass_status") == "deferred_until_real_subgraph_helper_lowering"
        ),
        "safe_scatter": ".scatter(" in code,
        "sdpa": "scaled_dot_product_attention" in families,
        "loss": "loss_distance" in families,
        "short": _code_bucket(code) == "<20",
        "long": _code_bucket(code) == ">=75",
        "code": code,
    }


def _selected_histogram(lane: str, manifests: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    if lane == "shape":
        values = (manifest.get("shape_intervention", {}).get("assigned_target_value") for manifest in manifests)
    elif lane == "dtype":
        values = (manifest.get("assigned_target") for manifest in manifests)
    else:
        raise ValueError(f"lane has no intervention histogram:{lane}")
    return _histogram(values)


def _additive_row_errors(
    entry: Mapping[str, Any],
    additive_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    *,
    index: int,
    lane: str,
    lane_index: int,
    root: str,
    selected: Path,
    selected_manifest: Path,
    additive_schema: Any,
) -> list[str]:
    errors: list[str] = []
    code = _code(source_row)
    if (
        set(entry) != ADDITIVE_LINEAGE_FIELDS
        or entry.get("additive_row_index") != index
        or entry.get("lane") != lane
        or entry.get("lane_row_index") != lane_index
        or entry.get("uuid") != _uuid(source_row)
        or entry.get("root_base_uuid") != root
        or entry.get("parent_uuid") != source_manifest.get("parent_uuid")
        or entry.get("reference_sha256") != _ref_hash(code)
        or entry.get("normalized_ast_sha256") != _ast_hash(code)
        or entry.get("source_manifest_row_sha256") != _row_hash(source_manifest)
    ):
        errors.append(f"additive lineage identity mismatch:{index}")
    for name, path in (("source_parquet", selected), ("source_manifest", selected_manifest)):
        error = _exact_file_binding(entry.get(name), path, f"additive:{index}:{name}")
        if error:
            errors.append(error)
    if _canonical(additive_row) != _canonical(_normalized_for_schema(source_row, additive_schema)):
        errors.append(f"additive parquet row differs from selected source row:{index}")
    return errors


def _requirements() -> set[str]:
    return {
        *(f"family:{item}" for item in sorted(REQUIRED_FAMILIES)),
        *(f"code_bucket:{item}" for item in ("<20", "20-34", "35-49", "50-74", ">=75")),
        *(f"operator_bucket:{item}" for item in ("1", "2-4", "5-9", "10-15", ">=16")),
        *(f"lane:{item}" for item in LANES),
        "safe_scatter",
        "sdpa",
        "loss",
        "short",
        "long",
    }


def _covers(item: Mapping[str, Any]) -> set[str]:
    covered = {
        f"lane:{item['lane']}",
        f"code_bucket:{item['code_bucket']}",
        f"operator_bucket:{item['operator_bucket']}",
    }
    covered.update(f"family:{family}" for family in item["families"])
    covered.update(name for name in ("safe_scatter", "sdpa", "loss", "short", "long") if item[name])
    return covered


def _rank(tag: str, item: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{SEED}|{tag}|{item['lane']}|{item['uuid']}".encode()).hexdigest()


def _stratified_sample(items: Sequence[dict[str, Any]], count: int) -> tuple[list[dict[str, Any]], list[str]]:
    required = _requirements()
    available = set().union(*(_covers(item) for item in items)) if items else set()
    missing = sorted(required - available)
    selected: list[dict[str, Any]] = []
    uncovered = set(required)
    while uncovered:
        choices = [item for item in items if item not in selected]
        if not choices:
            break
        best = max(
            choices,
            key=lambda item: (len(_covers(item) & uncovered), _rank("cover:" + ",".join(sorted(uncovered)), item)),
        )
        if not (_covers(best) & uncovered):
            break
        selected.append(best)
        uncovered -= _covers(best)
    if uncovered:
        missing.extend(sorted(uncovered))
    if len(selected) > count:
        missing.append(f"sample budget exhausted:{len(selected)}>{count}")
        return selected, missing
    selected_ids = {(item["lane"], item["uuid"]) for item in selected}
    remaining = sorted(
        (item for item in items if (item["lane"], item["uuid"]) not in selected_ids),
        key=lambda item: _rank("fill", item),
    )
    selected.extend(remaining[: count - len(selected)])
    return selected, missing


def _write_samples(output: Path, items: Sequence[dict[str, Any]], *, name: str) -> dict[str, Any]:
    path = output / f"{name}.jsonl"
    with path.open("w") as handle:
        for item in items:
            review = {key: value for key, value in item.items() if key != "code"}
            review["code_sha256"] = _ref_hash(item["code"])
            review["code"] = item["code"]
            handle.write(_canonical(review) + "\n")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path), "rows": len(items)}


def _write_kimi_packet(output: Path, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 28:
        raise ValueError(f"Kimi packet must contain exactly 28 assignments, got {len(items)}")
    packet_dir = output / "kimi_k3_packet"
    if packet_dir.exists():
        raise FileExistsError(f"Kimi packet already exists:{packet_dir}")
    packet_dir.mkdir()
    (packet_dir / "raw_outputs").mkdir()
    assignments: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        assignment_id = f"kimi_k3_{index:02d}"
        prompt = "\n".join(
            (
                "审计以下独立生成的 PyTorch 低层算子样本。只判断语义、实现风险和是否像重复变体；不要执行代码。",
                f"assignment_id: {assignment_id}",
                f"lane: {item['lane']}; uuid: {item['uuid']}; root_base_uuid: {item['root_base_uuid']}",
                f"strata: families={','.join(item['families'])}; code_bucket={item['code_bucket']}; operator_bucket={item['operator_bucket']}; source_form=single_model_only; semantic_multi_class=deferred",
                f"relationship_metadata: {_canonical(item.get('kimi_relationship_metadata', {}))}",
                *(
                    [
                        "related_source_evidence:",
                        *[
                            "--- related role={role}; lane={lane}; uuid={uuid}; reference_sha256={reference_sha256}; code_sha256={code_sha256} ---\n```python\n{code}\n```".format(
                                **related
                            )
                            for related in item.get("kimi_related_samples", [])
                        ],
                    ]
                    if item.get("kimi_related_samples")
                    else []
                ),
                "输出契约：回复的唯一内容必须是一个 UTF-8 JSON object；不要使用 Markdown fence，不要在 JSON 前后添加解释、标题、致谢或任何其他文本。",
                "该 object 的字段集合必须且只能是 assignment_id、uuid、verdict、severity、reasons、duplicate_concern、semantic_concern。assignment_id 和 uuid 必须逐字回显本 assignment。",
                "verdict 只能是 PASS、CONDITIONAL 或 FAIL；severity 只能是 P0、P1、P2 或 none；reasons 必须是非空字符串，或每一项均为非空字符串的非空数组；duplicate_concern 和 semantic_concern 必须是 JSON boolean。",
                "一致性约束：PASS 必须配 severity=none 且两个 concern 都为 false；CONDITIONAL 必须配 severity=P2；FAIL 必须配 severity=P0 或 P1。",
                "```python",
                item["code"].rstrip(),
                "```",
                "",
            )
        )
        prompt_path = packet_dir / f"{assignment_id}.md"
        prompt_path.write_text(prompt)
        assignments.append(
            {
                "assignment_id": assignment_id,
                "lane": item["lane"],
                "uuid": item["uuid"],
                "root_base_uuid": item["root_base_uuid"],
                "reference_sha256": item["reference_sha256"],
                "relationship_metadata": item.get("kimi_relationship_metadata", {}),
                "related_sources": [
                    {key: related[key] for key in ("role", "lane", "uuid", "reference_sha256", "code_sha256")}
                    for related in item.get("kimi_related_samples", [])
                ],
                "prompt_path": str(prompt_path.resolve()),
                "prompt_sha256": _sha256_file(prompt_path),
                "raw_output_path": str((packet_dir / "raw_outputs" / f"{assignment_id}.raw.txt").resolve()),
                "raw_output_sha256": None,
                "execution_gate": "final_artifacts_ready_notification_required",
            }
        )
    assignment_path = packet_dir / "assignments.jsonl"
    assignment_path.write_text("".join(_canonical(item) + "\n" for item in assignments))
    protocol = packet_dir / "README.md"
    protocol.write_text(
        "# Kimi K3 base-shape-dtype final review packet\n\n"
        "This packet is prepared but not submitted.  Submit only after the final-artifact-ready notification.  Preserve every response byte-for-byte at the matching `raw_output_path`; then write `kimi_output_hashes.jsonl` with assignment_id, raw_output_path, and SHA-256.  Do not overwrite prompts or assignments after submission.\n"
    )
    return {
        "assignments": {
            "path": str(assignment_path.resolve()),
            "sha256": _sha256_file(assignment_path),
            "rows": len(assignments),
        },
        "packet_readme": {"path": str(protocol.resolve()), "sha256": _sha256_file(protocol)},
        "packet_dir": str(packet_dir.resolve()),
        "submission_performed": False,
    }


def _kimi_required_examples(
    items: Sequence[dict[str, Any]], near_payload: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Require one complete root+sibling group and one cross-root near pair."""
    by_root: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for item in items:
        by_root[item["root_base_uuid"]][item["lane"]] = item
    complete = [root for root, lanes in by_root.items() if set(lanes) == set(LANES)]
    gaps: list[str] = []
    if not complete:
        return [], ["Kimi lacks a complete base/shape/dtype sibling group"]
    sibling_root = min(complete, key=lambda root: hashlib.sha256(f"{SEED}|sibling|{root}".encode()).hexdigest())
    required = [dict(by_root[sibling_root][lane]) for lane in LANES]
    pairs = near_payload.get("approximate_graph_sensitivity", {}).get("top_pairs", [])
    if not isinstance(pairs, list):
        return required, ["Kimi lacks readable base cross-root near-neighbor pairs"]
    base_by_uuid = {item["uuid"]: item for item in items if item["lane"] == "base"}
    pair = next(
        (
            value
            for value in pairs
            if isinstance(value, Mapping)
            and value.get("left_uuid") in base_by_uuid
            and value.get("right_uuid") in base_by_uuid
            and value["left_uuid"] != value["right_uuid"]
        ),
        None,
    )
    if pair is None:
        return required, ["Kimi lacks a cross-root near-neighbor pair from retained near-dedup"]
    near_left, near_right = base_by_uuid[pair["left_uuid"]], base_by_uuid[pair["right_uuid"]]
    by_identity = {(item["lane"], item["uuid"]): item for item in required}
    for source in (near_left, near_right):
        by_identity.setdefault((source["lane"], source["uuid"]), dict(source))
    required = list(by_identity.values())
    pair_metadata = {key: value for key, value in pair.items() if key not in {"left_uuid", "right_uuid"}}
    for item in required:
        sibling = item["root_base_uuid"] == sibling_root
        near = item["uuid"] in {pair["left_uuid"], pair["right_uuid"]}
        item["kimi_relation"] = (
            "complete_sibling_group"
            if sibling and not near
            else (
                "cross_root_near_neighbor"
                if near and not sibling
                else "complete_sibling_group_and_cross_root_near_neighbor"
            )
        )
        item["kimi_relationship_metadata"] = {
            "complete_sibling_group_root": sibling_root if sibling else None,
            "cross_root_near_neighbor": (
                {
                    "left_uuid": pair["left_uuid"],
                    "right_uuid": pair["right_uuid"],
                    "peer_uuid": pair["right_uuid"] if item["uuid"] == pair["left_uuid"] else pair["left_uuid"],
                    **pair_metadata,
                }
                if near
                else None
            ),
        }
        related: list[dict[str, Any]] = []
        if sibling:
            related.extend(
                {
                    "role": "same_root_sibling",
                    "lane": peer["lane"],
                    "uuid": peer["uuid"],
                    "reference_sha256": peer["reference_sha256"],
                    "code_sha256": _ref_hash(peer["code"]),
                    "code": peer["code"],
                }
                for peer in (by_root[sibling_root][lane] for lane in LANES)
                if peer["uuid"] != item["uuid"]
            )
        if near:
            peer = near_right if item["uuid"] == pair["left_uuid"] else near_left
            related.append(
                {
                    "role": "cross_root_near_neighbor",
                    "lane": peer["lane"],
                    "uuid": peer["uuid"],
                    "reference_sha256": peer["reference_sha256"],
                    "code_sha256": _ref_hash(peer["code"]),
                    "code": peer["code"],
                }
            )
        item["kimi_related_samples"] = related
    return required, gaps


def _kimi_exact_sample(
    items: Sequence[dict[str, Any]], near_payload: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    forced, gaps = _kimi_required_examples(items, near_payload)
    if gaps:
        return forced, gaps
    forced_ids = {(item["lane"], item["uuid"]) for item in forced}
    selected = list(forced)
    remaining = [item for item in items if (item["lane"], item["uuid"]) not in forced_ids]
    uncovered = _requirements() - set().union(*(_covers(item) for item in selected))
    while uncovered and len(selected) < 28:
        best = max(
            remaining,
            key=lambda item: (len(_covers(item) & uncovered), _rank("kimi:" + ",".join(sorted(uncovered)), item)),
        )
        if not (_covers(best) & uncovered):
            break
        selected.append(best)
        remaining.remove(best)
        uncovered -= _covers(best)
    if uncovered:
        return selected, [
            *(f"Kimi uncovered stratum:{value}" for value in sorted(uncovered)),
            f"Kimi cover budget:{len(selected)}/28",
        ]
    selected.extend(sorted(remaining, key=lambda item: _rank("kimi-fill", item))[: 28 - len(selected)])
    return selected, [] if len(selected) == 28 else [f"Kimi exact 28-row packet unavailable:{len(selected)}"]
