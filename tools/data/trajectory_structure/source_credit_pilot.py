"""Offline audit of the production single-stage source-component credit path.

This reconstructs enough ``Sample`` metadata from exported trajectories to call
``attribute_best_source_components`` unchanged.  The synthetic one-token mask
is solely to exercise its trainability guard: this program must never provide
these reconstructed samples to training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from examples.kernel_agent.source_component_reward import attribute_best_source_components, source_request_identity
from examples.kernel_agent.source_components import analyze_source_components
from examples.kernel_agent.utils import extract_cuda_agent_kernel_code
from slime.utils.types import Sample


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_reward(record: dict[str, Any]) -> float:
    value = record.get("task_reward")
    if value is None:
        value = record.get("reward")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0
    return float(value)


def label_ground_truth(record: dict[str, Any]) -> str:
    label = record.get("label")
    if isinstance(label, dict) and isinstance(label.get("ground_truth"), str):
        return label["ground_truth"]
    return ""


def env_metadata(record: dict[str, Any]) -> dict[str, Any]:
    env = record.get("env_result")
    state = env.get("env_state") if isinstance(env, dict) else None
    return state.get("metadata", {}) if isinstance(state, dict) and isinstance(state.get("metadata"), dict) else {}


def restored_remove(record: dict[str, Any]) -> tuple[bool, bool]:
    """Undo only the exporter’s terminal soft-finalization removal."""
    removed = bool(record.get("remove_sample", False))
    reason = record.get("remove_reason")
    soft_restored = removed and isinstance(reason, str) and reason.startswith("finalize_")
    return (False if soft_restored else removed), soft_restored


def sample_status(value: Any) -> Sample.Status:
    by_name = {item.value: item for item in Sample.Status}
    return by_name.get(str(getattr(value, "value", value)).lower(), Sample.Status.FAILED)


def reconstruct_sample(record: dict[str, Any]) -> tuple[Sample, dict[str, Any]]:
    response = record.get("response") if isinstance(record.get("response"), str) else ""
    code = extract_cuda_agent_kernel_code(response)
    metadata = env_metadata(record)
    ground_truth = label_ground_truth(record)
    entry_point = str(metadata.get("kernel_entry_point") or "ModelNew")
    precision = str(metadata.get("precision") or "unknown")
    removed, soft_restored = restored_remove(record)
    profiles = metadata.get("profiling", {}).get("kernels", []) if isinstance(metadata.get("profiling"), dict) else []
    sample = Sample(
        index=record.get("sample_index"),
        group_index=record.get("group_index"),
        group_id=record.get("group_id"),
        response=response,
        response_length=max(len(response), 1),
        # Exported rollout rows omit token masks. This non-empty placeholder is
        # an offline eligibility fixture, never trainable rollout data.
        loss_mask=[1],
        remove_sample=removed,
        status=sample_status(record.get("status")),
        metadata={
            "env_extra_info": record.get("env_extra_info") or {},
            "env_result": record.get("env_result") or {},
            "is_pad_turn": bool(record.get("is_pad_turn", False)),
            "source_component_identity": source_request_identity(
                ground_truth, code, entry_point=entry_point, precision=precision, submitted=False
            ),
            "source_component_profiles": profiles if isinstance(profiles, list) else [],
            "offline_source_credit_reconstruction": {
                "synthetic_nonempty_loss_mask": True,
                "soft_finalize_removal_restored": soft_restored,
                "source_identity_reconstructed": True,
            },
        },
    )
    audit = {
        "turn_idx": record.get("turn_idx"),
        "q": finite_reward(record),
        "q_source": "task_reward" if record.get("task_reward") is not None else "reward_fallback_task_reward_null",
        "multi_turn_reward": record.get("multi_turn_reward"),
        "export_remove_sample": bool(record.get("remove_sample", False)),
        "export_remove_reason": record.get("remove_reason"),
        "reconstructed_remove_sample": removed,
        "soft_finalize_removal_restored": soft_restored,
        "status": sample.status.value,
        "correctness": (record.get("env_extra_info") or {}).get("correctness"),
        "is_pad_turn": bool(record.get("is_pad_turn", False)),
        "source_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "entry_point": entry_point,
        "precision": precision,
    }
    return sample, audit


def unit_view(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": result.get("schema"),
        "unknowns": result.get("unknowns", []),
        "wall_seconds": result.get("wall_seconds"),
        "units": [
            {
                key: unit.get(key)
                for key in ("id", "kind", "features", "signature", "unknowns", "use_observed", "evidence")
            }
            for unit in result.get("units", [])
        ],
    }


def audit_rows(trajectory: str, rows: list[dict[str, Any]], input_sha256: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError(f"trajectory is not a list: {trajectory}")
    rows = sorted(rows, key=lambda row: row.get("turn_idx", -1))
    if [row.get("turn_idx") for row in rows] != list(range(len(rows))):
        raise ValueError("source credit audit needs contiguous, unique turns starting at zero")
    samples, base_scores, turns = [], [], []
    for row in rows:
        sample, turn = reconstruct_sample(row)
        samples.append(sample)
        base_scores.append(turn["q"])
        direct = analyze_source_components(
            extract_cuda_agent_kernel_code(sample.response), profiles=sample.metadata["source_component_profiles"]
        )
        turn["source"] = unit_view(direct)
        turns.append(turn)
    allocation = attribute_best_source_components(samples, base_scores)
    credits = allocation["credits"]
    if len(credits) != len(rows):
        raise ValueError(f"credit length mismatch: {trajectory}")
    for turn, credit in zip(turns, credits, strict=True):
        turn["credit"] = credit
    hard_removed_credit = sum(
        credit
        for row, turn, credit in zip(rows, turns, credits, strict=True)
        if turn["reconstructed_remove_sample"] and not turn["soft_finalize_removal_restored"]
    )
    budget_error = math.fsum(credits) - allocation["quality_budget"]
    if not math.isclose(budget_error, 0.0, abs_tol=1e-9):
        raise ValueError(f"budget conservation failure {trajectory}: {budget_error}")
    if not math.isclose(hard_removed_credit, 0.0, abs_tol=1e-12):
        raise ValueError(f"hard-removed sample received credit {trajectory}: {hard_removed_credit}")
    return {
        "trajectory": trajectory,
        "input_sha256": input_sha256,
        "turns": turns,
        "allocation": allocation,
        "checks": {"budget_error": budget_error, "hard_removed_credit": hard_removed_credit},
    }


def audit_one(path: Path) -> dict[str, Any]:
    return audit_rows(path.stem, json.loads(path.read_text()), canonical_sha256(path))


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def pt_record(value: Any, position: int) -> dict[str, Any]:
    """Select the saved rollout fields needed by this offline-only audit."""
    metadata = _field(value, "metadata", {}) or {}
    return {
        "position": position,
        "sample_index": _field(value, "index"),
        "group_index": _field(value, "group_index"),
        "group_id": _field(value, "group_id"),
        "turn_idx": metadata.get("turn_idx", _field(value, "turn_idx", 0)),
        "is_pad_turn": bool(metadata.get("is_pad_turn", False)),
        "remove_sample": bool(_field(value, "remove_sample", False)),
        "remove_reason": metadata.get("remove_reason"),
        "status": _field(value, "status", "failed"),
        "reward": _field(value, "reward"),
        "task_reward": metadata.get("task_reward"),
        "multi_turn_reward": metadata.get("multi_turn_reward"),
        "env_extra_info": metadata.get("env_extra_info") or {},
        "env_result": metadata.get("env_result") or {},
        "label": _field(value, "label"),
        "response": _field(value, "response", ""),
    }


def pt_trajectories(path: Path) -> list[tuple[str, list[dict[str, Any]], str]]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    rows = loaded.get("samples") if isinstance(loaded, dict) else loaded
    if not isinstance(rows, list):
        raise ValueError(f"unsupported torch rollout payload: {path}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for position, value in enumerate(rows):
        row = pt_record(value, position)
        key = str(row["group_id"] if row["group_id"] is not None else row["sample_index"])
        grouped.setdefault(key, []).append(row)
    digest = canonical_sha256(path)
    return [(f"group_{key}", values, digest) for key, values in sorted(grouped.items(), key=lambda item: int(item[0]))]


def summarize(results: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    allocations = [result["allocation"] for result in results]
    all_turns = [turn for result in results for turn in result["turns"]]
    cross = [
        unit
        for allocation in allocations
        for unit in allocation["units"]
        if unit.get("origin_turn") is not None and unit["origin_turn"] < allocation["best_turn"]
    ]
    failed_origins = 0
    for result in results:
        for unit in result["allocation"]["units"]:
            origin = unit.get("origin_turn")
            if origin is not None and not result["turns"][origin]["source"].get("units", []):
                # The allocation may refer to an eligible proposal with no
                # current source units only if a future producer changes shape;
                # retain an explicit diagnostic rather than infer success.
                continue
            if origin is not None:
                state = result["turns"][origin]["status"]
                if state != "completed" or result["turns"][origin]["correctness"] is not True:
                    failed_origins += 1
    return {
        "trajectories": len(results),
        "turns": len(all_turns),
        "allocation_status": dict(Counter(allocation["status"] for allocation in allocations)),
        "quality_budget": math.fsum(allocation["quality_budget"] for allocation in allocations),
        "credit": math.fsum(turn["credit"] for turn in all_turns),
        "budget_error": math.fsum(result["checks"]["budget_error"] for result in results),
        "hard_removed_credit": math.fsum(result["checks"]["hard_removed_credit"] for result in results),
        "cross_turn_units": len(cross),
        "cross_turn_trajectory_fraction": (
            sum(allocation["resolved_cross_turn_units"] > 0 for allocation in allocations) / len(allocations)
            if allocations
            else 0.0
        ),
        "failed_origin_units_proxy": failed_origins,
        "residual_fraction_mean": (
            math.fsum(allocation["residual_fraction"] for allocation in allocations) / len(allocations)
            if allocations
            else 0.0
        ),
        "source_analysis_wall_seconds": math.fsum(
            allocation.get("source_analysis_wall_seconds", 0.0) for allocation in allocations
        ),
        "direct_description_wall_seconds": math.fsum(turn["source"]["wall_seconds"] or 0.0 for turn in all_turns),
        "pilot_wall_seconds": wall_seconds,
        "soft_finalize_restored_turns": sum(turn["soft_finalize_removal_restored"] for turn in all_turns),
        "offline_only": {
            "synthetic_loss_mask": True,
            "must_not_be_used_for_training": True,
            "source_identity_reconstructed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Exported trajectory directory or saved rollout .pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.input.is_dir():
        work = [(path.stem, None, path) for path in sorted(args.input.glob("trajectory_*.json"))]
    elif args.input.suffix == ".pt":
        work = [(name, rows, digest) for name, rows, digest in pt_trajectories(args.input)]
    else:
        raise ValueError(f"Unsupported input: {args.input}")
    if args.limit is not None:
        work = work[: args.limit]
    if not work:
        raise ValueError(f"No trajectories under {args.input}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / "allocations").mkdir()
    start = time.monotonic()
    results = []
    for index, (name, rows, source) in enumerate(work, 1):
        result = audit_one(source) if rows is None else audit_rows(name, rows, source)
        results.append(result)
        (args.output / "allocations" / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        if index % 25 == 0 or index == len(work):
            print(json.dumps({"completed": index, "total": len(work)}, ensure_ascii=False), flush=True)
    summary = summarize(results, time.monotonic() - start)
    summary.update(
        input=str(args.input.resolve()),
        input_files=len(work),
        script_sha256=canonical_sha256(Path(__file__)),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
