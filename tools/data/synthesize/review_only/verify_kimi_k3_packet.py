"""Fail-closed verifier for the base/shape/dtype Kimi K3 review packet.

The final reviewer deliberately only prepares the packet.  This module runs
after a human has submitted that packet to Kimi and saved the 28 verbatim
responses.  It never invokes a model or imports the synthesis pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.data.synthesize.review_only import csp_dag_review_common, csp_dag_review_runtime, csp_dag_review_sampling

CONTRACT = "csp_dag_shape_dtype_kimi_verify_only_v1"
REVIEW_CONTRACT = "csp_dag_shape_dtype_final_independent_review_v1"
EXPECTED_ASSIGNMENT_IDS = tuple(f"kimi_k3_{index:02d}" for index in range(28))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIRECT_REQUIRED_FIELDS = frozenset(
    {
        "assignment_id",
        "uuid",
        "verdict",
        "severity",
        "reasons",
        "duplicate_concern",
        "semantic_concern",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {
        "assignment_id",
        "lane",
        "uuid",
        "root_base_uuid",
        "reference_sha256",
        "relationship_metadata",
        "related_sources",
        "prompt_path",
        "prompt_sha256",
        "raw_output_path",
        "raw_output_sha256",
        "execution_gate",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _read_jsonl(path: Path, label: str, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"missing {label}:{path}")
        return []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        errors.append(f"non-UTF-8 {label}:{path}")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            errors.append(f"blank {label} row:{line_number}")
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"malformed {label} JSON:{line_number}")
            continue
        if not isinstance(value, dict):
            errors.append(f"non-object {label} row:{line_number}")
            continue
        rows.append(value)
    return rows


def _expected_prompt_path(packet_dir: Path, assignment_id: str) -> Path:
    return packet_dir / f"{assignment_id}.md"


def _expected_raw_path(packet_dir: Path, assignment_id: str) -> Path:
    return packet_dir / "raw_outputs" / f"{assignment_id}.raw.txt"


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _meaningful_reason(value: Any) -> bool:
    if _nonempty_text(value):
        return True
    return isinstance(value, list) and bool(value) and all(_nonempty_text(item) for item in value)


def _parse_response(
    raw_path: Path, expected_assignment_id: str, expected_uuid: str
) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    """Parse the exact response envelope required by the current packet."""
    try:
        text = raw_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, None, ["raw output is not UTF-8"]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None, None, ["raw output is not one unfenced JSON object"]
    if not isinstance(value, dict):
        return None, None, ["raw output JSON is not an object"]

    errors: list[str] = []
    if set(value) != _DIRECT_REQUIRED_FIELDS:
        errors.append("response field set mismatch")
    if value.get("assignment_id") != expected_assignment_id:
        errors.append("response assignment_id mismatch")
    if value.get("uuid") != expected_uuid:
        errors.append("response uuid mismatch")
    if value.get("verdict") not in {"PASS", "CONDITIONAL", "FAIL"}:
        errors.append("response invalid verdict")
    if value.get("severity") not in {"P0", "P1", "P2", "none"}:
        errors.append("response invalid severity")
    if not _meaningful_reason(value.get("reasons")):
        errors.append("response lacks nonempty reasons")
    for key in ("duplicate_concern", "semantic_concern"):
        if type(value.get(key)) is not bool:
            errors.append(f"response {key} is not boolean")
    verdict = value.get("verdict")
    severity = value.get("severity")
    concerns = value.get("duplicate_concern") is True or value.get("semantic_concern") is True
    if (
        verdict == "PASS"
        and (severity != "none" or concerns)
        or verdict == "CONDITIONAL"
        and severity != "P2"
        or verdict == "FAIL"
        and severity not in {"P0", "P1"}
    ):
        errors.append("response verdict/severity/concern mismatch")
    return "direct_envelope_v1", value, errors


def _assignment_errors(
    packet_dir: Path, assignments: Sequence[dict[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    errors: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    prompt_bindings: list[dict[str, Any]] = []
    uuids: set[str] = set()
    references: set[str] = set()
    if len(assignments) != len(EXPECTED_ASSIGNMENT_IDS):
        errors.append(f"assignment count is not exactly 28:{len(assignments)}")
    if [row.get("assignment_id") for row in assignments] != list(EXPECTED_ASSIGNMENT_IDS):
        errors.append("assignment rows are not in fixed kimi_k3_00..27 order")
    for row_index, assignment in enumerate(assignments):
        assignment_id = assignment.get("assignment_id")
        if not isinstance(assignment_id, str) or assignment_id not in EXPECTED_ASSIGNMENT_IDS:
            errors.append(f"invalid assignment_id:{row_index}")
            continue
        if assignment_id in by_id:
            errors.append(f"duplicate assignment_id:{assignment_id}")
            continue
        by_id[assignment_id] = assignment
        if set(assignment) != _ASSIGNMENT_FIELDS:
            errors.append(f"assignment field set mismatch:{assignment_id}")
        uuid = assignment.get("uuid")
        if not _nonempty_text(uuid):
            errors.append(f"invalid assignment uuid:{assignment_id}")
        elif uuid in uuids:
            errors.append(f"duplicate assignment uuid:{uuid}")
        else:
            uuids.add(uuid)
        if (
            assignment.get("lane") not in {"base", "shape", "dtype"}
            or not _nonempty_text(assignment.get("root_base_uuid"))
            or not _is_sha256(assignment.get("reference_sha256"))
            or not isinstance(assignment.get("relationship_metadata"), dict)
            or not isinstance(assignment.get("related_sources"), list)
            or assignment.get("raw_output_sha256") is not None
            or assignment.get("execution_gate") != "final_artifacts_ready_notification_required"
        ):
            errors.append(f"assignment metadata mismatch:{assignment_id}")
        reference = assignment.get("reference_sha256")
        if isinstance(reference, str) and reference in references:
            errors.append(f"duplicate assignment reference:{reference}")
        elif isinstance(reference, str):
            references.add(reference)
        expected_prompt = _expected_prompt_path(packet_dir, assignment_id).resolve()
        expected_raw = _expected_raw_path(packet_dir, assignment_id).resolve()
        prompt_value = assignment.get("prompt_path")
        raw_value = assignment.get("raw_output_path")
        prompt_path = Path(str(prompt_value)).resolve()
        raw_path = Path(str(raw_value)).resolve()
        if prompt_path != expected_prompt:
            errors.append(f"assignment prompt path mismatch:{assignment_id}")
        if raw_path != expected_raw:
            errors.append(f"assignment raw path mismatch:{assignment_id}")
        if not prompt_path.is_file():
            errors.append(f"missing prompt:{assignment_id}")
            continue
        actual_prompt_sha256 = _sha256_file(prompt_path)
        if not _is_sha256(assignment.get("prompt_sha256")) or assignment.get("prompt_sha256") != actual_prompt_sha256:
            errors.append(f"assignment prompt SHA-256 mismatch:{assignment_id}")
        prompt_bindings.append(
            {
                "assignment_id": assignment_id,
                "uuid": uuid,
                "path": str(prompt_path),
                "sha256": actual_prompt_sha256,
            }
        )
    if set(by_id) != set(EXPECTED_ASSIGNMENT_IDS):
        errors.append("assignment identifiers are not the exact fixed kimi_k3_00..27 set")
    return errors, by_id, prompt_bindings


def _review_summary_errors(
    packet_dir: Path,
    assignments_path: Path,
    review_summary_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    expected_summary = packet_dir.parent / "summary.json"
    if review_summary_path.resolve() != expected_summary.resolve():
        errors.append("review summary is not the packet's owning summary")
    if not review_summary_path.is_file():
        return [*errors, f"missing review summary:{review_summary_path}"], {}
    try:
        summary = json.loads(review_summary_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [*errors, "review summary is not valid UTF-8 JSON"], {}
    if not isinstance(summary, dict):
        return [*errors, "review summary is not an object"], {}
    if (
        summary.get("contract") != REVIEW_CONTRACT
        or summary.get("passed") is not True
        or summary.get("p1_errors") != []
        or summary.get("review_only") is not True
        or summary.get("training_approved") is not False
    ):
        errors.append("review summary contract/verdict/governance mismatch")
    audit_path = Path(__file__).resolve().with_name("audit_csp_dag_final.py")
    expected_review_source = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in {
            "audit": audit_path,
            "common": Path(csp_dag_review_common.__file__).resolve(),
            "runtime": Path(csp_dag_review_runtime.__file__).resolve(),
            "sampling": Path(csp_dag_review_sampling.__file__).resolve(),
        }.items()
    }
    if summary.get("review_source") != expected_review_source:
        errors.append("review summary source binding mismatch")
    packet = summary.get("kimi_k3_packet")
    packet_readme = packet_dir / "README.md"
    expected_assignment_binding = {
        "path": str(assignments_path.resolve()),
        "sha256": _sha256_file(assignments_path) if assignments_path.is_file() else None,
        "rows": 28,
    }
    expected_packet_readme_binding = {
        "path": str(packet_readme.resolve()),
        "sha256": _sha256_file(packet_readme) if packet_readme.is_file() else None,
    }
    if (
        not isinstance(packet, dict)
        or set(packet) != {"assignments", "packet_readme", "packet_dir", "submission_performed"}
        or packet.get("packet_dir") != str(packet_dir.resolve())
        or packet.get("assignments") != expected_assignment_binding
        or packet.get("packet_readme") != expected_packet_readme_binding
        or packet.get("submission_performed") is not False
    ):
        errors.append("review summary Kimi packet binding mismatch")
    return errors, summary


def _hash_manifest_errors(
    packet_dir: Path, hash_rows: Sequence[dict[str, Any]], assignments: Mapping[str, dict[str, Any]]
) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    by_id: dict[str, str] = {}
    if len(hash_rows) != len(EXPECTED_ASSIGNMENT_IDS):
        errors.append(f"output-hash count is not exactly 28:{len(hash_rows)}")
    if [row.get("assignment_id") for row in hash_rows] != list(EXPECTED_ASSIGNMENT_IDS):
        errors.append("output-hash rows are not in fixed kimi_k3_00..27 order")
    for row_index, row in enumerate(hash_rows):
        assignment_id = row.get("assignment_id")
        if not isinstance(assignment_id, str) or assignment_id not in assignments:
            errors.append(f"hash row invalid assignment_id:{row_index}")
            continue
        if assignment_id in by_id:
            errors.append(f"duplicate output hash assignment_id:{assignment_id}")
            continue
        if set(row) != {"assignment_id", "raw_output_path", "raw_output_sha256"}:
            errors.append(f"hash row unexpected or missing fields:{assignment_id}")
            continue
        if not _is_sha256(row.get("raw_output_sha256")):
            errors.append(f"hash row invalid SHA-256:{assignment_id}")
            continue
        expected_raw = _expected_raw_path(packet_dir, assignment_id).resolve()
        raw_path = Path(str(row.get("raw_output_path"))).resolve()
        if raw_path != expected_raw:
            errors.append(f"hash row raw path mismatch:{assignment_id}")
            continue
        by_id[assignment_id] = str(row["raw_output_sha256"])
    if set(by_id) != set(EXPECTED_ASSIGNMENT_IDS):
        errors.append("output-hash identifiers are not the exact fixed kimi_k3_00..27 set")
    return errors, by_id


def verify(packet_dir: Path, review_summary_path: Path, output_dir: Path) -> dict[str, Any]:
    """Verify a completed fixed Kimi packet and write a source/input-bound summary."""
    packet_dir = packet_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Kimi verify output already exists:{output_dir}")
    output_dir.mkdir(parents=True)
    errors: list[str] = []
    assignments_path = packet_dir / "assignments.jsonl"
    hashes_path = packet_dir / "kimi_output_hashes.jsonl"
    review_errors, review_summary = _review_summary_errors(packet_dir, assignments_path, review_summary_path.resolve())
    errors.extend(review_errors)
    assignments = _read_jsonl(assignments_path, "assignments", errors)
    hash_rows = _read_jsonl(hashes_path, "kimi output hashes", errors)
    assignment_errors, assignments_by_id, prompt_bindings = _assignment_errors(packet_dir, assignments)
    errors.extend(assignment_errors)
    hash_errors, hashes_by_id = _hash_manifest_errors(packet_dir, hash_rows, assignments_by_id)
    errors.extend(hash_errors)

    raw_directory = packet_dir / "raw_outputs"
    expected_raw_paths = {
        _expected_raw_path(packet_dir, assignment_id).resolve() for assignment_id in EXPECTED_ASSIGNMENT_IDS
    }
    if not raw_directory.is_dir():
        errors.append(f"missing raw output directory:{raw_directory}")
    else:
        actual_raw_paths = {path.resolve() for path in raw_directory.iterdir()}
        if actual_raw_paths != expected_raw_paths:
            errors.append("raw output directory does not contain exactly the 28 assigned files")

    reviews: list[dict[str, Any]] = []
    blocking_reviews: list[dict[str, Any]] = []
    for assignment_id in EXPECTED_ASSIGNMENT_IDS:
        assignment = assignments_by_id.get(assignment_id)
        if assignment is None:
            continue
        raw_path = _expected_raw_path(packet_dir, assignment_id).resolve()
        if not raw_path.is_file():
            errors.append(f"missing raw output:{assignment_id}")
            continue
        raw_sha256 = _sha256_file(raw_path)
        manifest_sha256 = hashes_by_id.get(assignment_id)
        if manifest_sha256 != raw_sha256:
            errors.append(f"raw output SHA-256 mismatch:{assignment_id}")
            continue
        schema, response, response_errors = _parse_response(raw_path, assignment_id, str(assignment.get("uuid")))
        errors.extend(f"{assignment_id}: {error}" for error in response_errors)
        if schema == "direct_envelope_v1" and isinstance(response, dict):
            if response.get("verdict") == "FAIL" or response.get("severity") in {"P0", "P1"}:
                blocking_reviews.append(
                    {
                        "assignment_id": assignment_id,
                        "uuid": assignment.get("uuid"),
                        "verdict": response.get("verdict"),
                        "severity": response.get("severity"),
                    }
                )
        reviews.append(
            {
                "assignment_id": assignment_id,
                "uuid": assignment.get("uuid"),
                "raw_output_path": str(raw_path),
                "raw_output_sha256": raw_sha256,
                "response_schema": schema,
                "reported_verdict": response.get("verdict") if isinstance(response, dict) else None,
                "reported_severity": response.get("severity") if isinstance(response, dict) else None,
            }
        )

    input_material = {
        "review_summary": {
            "path": str(review_summary_path.resolve()),
            "sha256": _sha256_file(review_summary_path) if review_summary_path.is_file() else None,
            "contract": review_summary.get("contract") if isinstance(review_summary, dict) else None,
        },
        "assignments_sha256": _sha256_file(assignments_path) if assignments_path.is_file() else None,
        "kimi_output_hashes_sha256": _sha256_file(hashes_path) if hashes_path.is_file() else None,
        "prompts": sorted(prompt_bindings, key=lambda item: item["assignment_id"]),
        "raw_outputs": sorted(reviews, key=lambda item: item["assignment_id"]),
    }
    summary = {
        "contract_version": CONTRACT,
        "verifier_source_path": str(Path(__file__).resolve()),
        "verifier_source_sha256": _source_sha256(),
        "packet_dir": str(packet_dir),
        "verified_input_payload_sha256": hashlib.sha256(_canonical(input_material).encode("utf-8")).hexdigest(),
        "input_bindings": input_material,
        "response_schema_counts": {
            schema: sum(1 for review in reviews if review["response_schema"] == schema)
            for schema in sorted(
                {review["response_schema"] for review in reviews if review["response_schema"] is not None}
            )
        },
        "review_disposition_counts": {
            key: sum(1 for review in reviews if review.get("reported_verdict") == key)
            for key in ("PASS", "CONDITIONAL", "FAIL")
        },
        "blocking_reviews": blocking_reviews,
        "reviews": reviews,
        "p1_errors": errors,
        "passed": not errors and not blocking_reviews and len(reviews) == len(EXPECTED_ASSIGNMENT_IDS),
    }
    summary_path = output_dir / "kimi_verify_summary.json"
    summary_path.write_text(_canonical(summary) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = _sha256_file(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", required=True, type=Path)
    parser.add_argument("--review-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.packet_dir, args.review_summary, args.output_dir)
    print(_canonical(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
