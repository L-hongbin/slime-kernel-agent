"""Read-only contract audit for a postprocessed source-credit rollout dump.

The audit never reconstructs identities, token masks, rewards, or Samples.  It
only reads a saved ``rollout_*.pt`` and reports whether the actual serialized
fields are sufficient for source credit, predictive DPPO, and trajectory packing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from examples.kernel_agent.component_reward import (
    ADDITIVE_MODE,
    ComponentRewardContractError,
    validate_component_reward_record,
)
from examples.kernel_agent.source_component_reward import IDENTITY_SCHEMA
from examples.kernel_agent.utils import extract_cuda_agent_kernel_code


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def metadata(sample: Any) -> dict[str, Any]:
    value = field(sample, "metadata", {})
    return value if isinstance(value, dict) else {}


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sequence_length(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def compact(value: Any, limit: int = 400) -> Any:
    if isinstance(value, str):
        return value[:limit]
    return value


def env_precheck(meta: dict[str, Any]) -> Any:
    extra = meta.get("env_extra_info")
    if isinstance(extra, dict) and "precheck" in extra:
        return extra["precheck"]
    env = meta.get("env_result")
    state = env.get("env_state") if isinstance(env, dict) else None
    return state.get("precheck") if isinstance(state, dict) else None


def audit_predictive(sample: Any, errors: list[str], *, strict: bool) -> dict[str, Any]:
    response_length = field(sample, "response_length")
    tokens = field(sample, "tokens")
    loss_mask = field(sample, "loss_mask")
    log_probs = field(sample, "rollout_log_probs")
    identifiers = field(sample, "rollout_topk_token_ids")
    support_log_probs = field(sample, "rollout_topk_log_probs")
    valid_mask = field(sample, "rollout_topk_valid_mask")
    if not strict:
        mask_length = sequence_length(loss_mask)
        if mask_length is None:
            errors.append("excluded_turn_missing_loss_mask")
        elif any(loss_mask):
            errors.append("excluded_turn_has_trainable_loss_mask")
        return {
            "excluded": True,
            "response_length": response_length,
            "loss_mask_length": mask_length,
        }
    if not isinstance(response_length, int) or response_length <= 0:
        errors.append("missing_or_nonpositive_response_length")
        return {}
    for name, value in (("tokens", tokens), ("loss_mask", loss_mask), ("rollout_log_probs", log_probs)):
        length = sequence_length(value)
        if length is None:
            errors.append(f"missing_{name}")
        elif name == "tokens" and length < response_length:
            errors.append("tokens_shorter_than_response")
        elif name != "tokens" and length != response_length:
            errors.append(f"{name}_length_mismatch")
    trainable = (
        np.asarray(loss_mask, dtype=np.bool_) if loss_mask is not None else np.zeros(response_length, dtype=bool)
    )
    values = (identifiers, support_log_probs, valid_mask)
    if all(value is None for value in values) and not trainable.any():
        return {"response_length": response_length, "all_masked_missing_predictive_support": True}
    if any(value is None for value in values):
        errors.append("missing_or_partial_predictive_topk_support")
        return {"response_length": response_length}
    if log_probs is not None and len(trainable) == len(log_probs) and not np.all(np.isfinite(np.asarray(log_probs))):
        errors.append("rollout_log_probs_nonfinite")
    if loss_mask is not None and not all(value in {0, 1} for value in loss_mask):
        errors.append("loss_mask_not_binary")
    ids, logs, valid = (np.asarray(identifiers), np.asarray(support_log_probs), np.asarray(valid_mask))
    if ids.ndim != 2 or logs.ndim != 2 or valid.ndim != 2:
        errors.append("predictive_topk_not_rank2")
        return {"response_length": response_length, "shapes": [list(x.shape) for x in (ids, logs, valid)]}
    if ids.shape != logs.shape or ids.shape != valid.shape or ids.shape[0] != response_length:
        errors.append("predictive_topk_shape_mismatch")
        return {"response_length": response_length, "shapes": [list(x.shape) for x in (ids, logs, valid)]}
    if ids.shape[1] == 0:
        errors.append("predictive_topk_empty_support_width")
        return {"response_length": response_length, "topk_shape": list(ids.shape)}
    if ids.dtype != np.int32:
        errors.append("predictive_topk_token_ids_not_int32")
    if logs.dtype != np.float32:
        errors.append("predictive_topk_log_probs_not_float32")
    if valid.dtype != np.bool_:
        errors.append("predictive_topk_valid_mask_not_bool")
        return {"response_length": response_length, "topk_shape": list(ids.shape)}
    if len(trainable) == valid.shape[0] and not np.all(np.any(valid, axis=1)[trainable]):
        errors.append("predictive_topk_empty_trainable_support_row")
    if len(trainable) == valid.shape[0]:
        valid_counts = valid.sum(axis=1)
        allowed_counts = np.isin(valid_counts, [ids.shape[1] - 1, ids.shape[1]])
        if not np.all(allowed_counts[trainable]):
            errors.append("predictive_topk_trainable_valid_count_not_topk_or_topk_plus_sample")
    if not np.all(np.isfinite(logs[valid])):
        errors.append("predictive_topk_nonfinite_valid_logprob")
    if np.any(valid & (ids < 0)):
        errors.append("predictive_topk_negative_valid_token_id")
    sorted_ids = np.sort(np.where(valid, ids, -1), axis=1)
    if np.any((sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)):
        errors.append("predictive_topk_duplicate_valid_token_id")
    if isinstance(tokens, (list, tuple)) and len(tokens) >= response_length and ids.shape[0] == response_length:
        response_tokens = np.asarray(tokens[-response_length:])
        sampled_matches = valid & (ids.astype(np.int64, copy=False) == response_tokens[:, None])
        sampled_counts = sampled_matches.sum(axis=1)
        if len(trainable) == response_length and not np.all(sampled_counts[trainable] == 1):
            errors.append("sampled_trainable_response_token_not_exactly_once_in_predictive_support")
        if log_probs is not None and len(trainable) == response_length and len(log_probs) == response_length:
            sampled_slots = np.argmax(sampled_matches, axis=1)
            supported = np.take_along_axis(logs, sampled_slots[:, None], axis=1)[:, 0]
            if not np.all(np.isclose(supported[trainable], np.asarray(log_probs)[trainable], rtol=1e-5, atol=1e-6)):
                errors.append("sampled_trainable_logprob_mismatch_predictive_support")
    return {
        "response_length": response_length,
        "token_length": sequence_length(tokens),
        "loss_mask_length": sequence_length(loss_mask),
        "rollout_log_probs_length": sequence_length(log_probs),
        "topk_shape": list(ids.shape),
        "valid_rows": int(np.sum(np.any(valid, axis=1))),
    }


def audit_sample(sample: Any, ordinal: int) -> dict[str, Any]:
    errors: list[str] = []
    meta = metadata(sample)
    removed = bool(field(sample, "remove_sample", False))
    reason = meta.get("remove_reason")
    pad = bool(meta.get("is_pad_turn", False))
    saved_status = field(sample, "status", "")
    status = str(getattr(saved_status, "value", saved_status)).lower()
    hard_excluded = (
        pad or status == "aborted" or (removed and not (isinstance(reason, str) and reason.startswith("finalize_")))
    )
    response = field(sample, "response")
    if not isinstance(response, str) or not response:
        if not hard_excluded:
            errors.append("missing_response")
        response = "" if not isinstance(response, str) else response
    identity = meta.get("source_component_identity")
    label = field(sample, "label")
    ground_truth = label.get("ground_truth") if isinstance(label, dict) else None
    code = extract_cuda_agent_kernel_code(response)
    raw_sha, code_sha = sha256_text(response), sha256_text(code)
    if not isinstance(identity, dict) and not hard_excluded:
        errors.append("missing_source_component_identity")
    else:
        if isinstance(identity, dict) and identity.get("schema") != IDENTITY_SCHEMA:
            errors.append("source_identity_schema_mismatch")
        if (
            isinstance(identity, dict)
            and isinstance(response, str)
            and response
            and identity.get("candidate_source_sha256") != code_sha
        ):
            errors.append("candidate_source_sha256_mismatch")
        if isinstance(identity, dict) and not hard_excluded and not isinstance(ground_truth, str):
            errors.append("missing_ground_truth_for_task_identity")
        elif (
            isinstance(identity, dict)
            and isinstance(ground_truth, str)
            and identity.get("task_sha256") != sha256_text(ground_truth)
        ):
            errors.append("task_sha256_mismatch")
        if isinstance(identity, dict) and not hard_excluded:
            for name in ("entry_point", "precision"):
                if not isinstance(identity.get(name), str) or not identity[name]:
                    errors.append(f"missing_source_identity_{name}")
            submitted = identity.get("submitted")
            if not isinstance(submitted, bool):
                errors.append("source_identity_submitted_not_bool")
            else:
                precheck = env_precheck(meta)
                if precheck == "failed" and submitted:
                    errors.append("precheck_failure_marked_submitted")
                if precheck != "failed" and not submitted:
                    errors.append("nonprecheck_turn_marked_unsubmitted")
    task_reward = meta.get("task_reward")
    component = meta.get("component_reward")
    if not hard_excluded and not finite(task_reward):
        errors.append("missing_or_nonfinite_task_reward")
    if not isinstance(component, dict) and not hard_excluded:
        errors.append("missing_component_reward_record")
    else:
        turn_idx = meta.get("turn_idx")
        if isinstance(component, dict):
            for name in ("quality_budget", "turn_credit", "turn_target"):
                if not finite(component.get(name)):
                    errors.append(f"missing_or_nonfinite_component_{name}")
        targets, credits = (
            (component.get("targets"), component.get("credits")) if isinstance(component, dict) else (None, None)
        )
        if not hard_excluded and (
            not isinstance(turn_idx, int) or not isinstance(targets, list) or not isinstance(credits, list)
        ):
            errors.append("component_targets_or_turn_index_missing")
        elif not hard_excluded and not (0 <= turn_idx < len(targets) == len(credits)):
            errors.append("component_target_vector_length_mismatch")
        elif not hard_excluded:
            if component.get("turn_credit") != credits[turn_idx]:
                errors.append("component_turn_credit_not_vector_value")
            if component.get("turn_target") != targets[turn_idx]:
                errors.append("component_turn_target_not_vector_value")
            if component.get("mode") == ADDITIVE_MODE:
                settings = SimpleNamespace(
                    component_reward_mode=ADDITIVE_MODE,
                    component_reward_scale=component.get("scale"),
                    component_reward_min_speedup=component.get("anchor_min_speedup"),
                )
                try:
                    validate_component_reward_record(settings, component)
                except ComponentRewardContractError as error:
                    errors.append(f"invalid_additive_component_record:{error}")
                else:
                    if field(sample, "reward") != component["baseline_rewards"][turn_idx]:
                        errors.append("sample_reward_not_saved_baseline_reward")
                    if meta.get("multi_turn_reward") != component["baseline_returns"][turn_idx]:
                        errors.append("multi_turn_reward_not_saved_baseline_return")
            else:
                if field(sample, "reward") != component.get("turn_target"):
                    errors.append("sample_reward_not_component_turn_target")
                if meta.get("multi_turn_reward") != component.get("turn_target"):
                    errors.append("multi_turn_reward_not_component_turn_target")
            if finite(task_reward) and isinstance(component.get("base_scores"), list):
                if len(component["base_scores"]) != len(targets) or component["base_scores"][turn_idx] != task_reward:
                    errors.append("task_reward_not_original_component_base_score")
    credit = component.get("turn_credit") if isinstance(component, dict) else None
    protected = bool(meta.get("component_reward_soft_finalize_protected", False))
    if protected and removed:
        errors.append("soft_finalize_protected_sample_removed")
    if (removed or pad or status == "aborted") and finite(credit) and credit != 0.0:
        errors.append("removed_pad_or_aborted_sample_received_credit")
    predictive = audit_predictive(sample, errors, strict=not hard_excluded)
    return {
        "ordinal": ordinal,
        "group_id": field(sample, "group_id"),
        "sample_index": field(sample, "index"),
        "turn_idx": meta.get("turn_idx"),
        "errors": sorted(set(errors)),
        "response_sha256": raw_sha,
        "candidate_source_sha256": code_sha,
        "identity": identity,
        "task_reward": task_reward,
        "sample_reward": field(sample, "reward"),
        "component_credit": credit,
        "component_target": component.get("turn_target") if isinstance(component, dict) else None,
        "remove_sample": removed,
        "remove_reason": reason,
        "soft_finalize_protected": protected,
        "is_pad_turn": pad,
        "status": status,
        "hard_excluded": hard_excluded,
        "predictive": predictive,
    }


def group_checks(samples: list[Any], audits: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if any(type(audit["turn_idx"]) is not int for audit in audits):
        return ["turn_indices_not_ordered_contiguous"]
    ordered = sorted(zip(samples, audits, strict=True), key=lambda item: item[1]["turn_idx"])
    turns = [audit["turn_idx"] for _, audit in ordered]
    if turns != list(range(len(turns))):
        errors.append("turn_indices_not_ordered_contiguous")
    active = [(sample, audit) for sample, audit in ordered if not audit["hard_excluded"]]
    identities = [audit["identity"] for _, audit in active if isinstance(audit["identity"], dict)]
    contexts = {(item.get("task_sha256"), item.get("entry_point"), item.get("precision")) for item in identities}
    if active and len(contexts) != 1:
        errors.append("source_identity_context_inconsistent")
    records = [metadata(sample).get("component_reward") for sample, _ in active]
    if not active:
        return errors
    if not all(isinstance(record, dict) for record in records):
        return [*errors, "missing_component_reward_record_in_group"]
    first = records[0]
    for record in records[1:]:
        if any(
            record.get(key) != first.get(key)
            for key in ("credits", "targets", "quality_budget", "best_turn", "status")
        ):
            errors.append("component_allocation_not_identical_within_group")
            break
    credits = first.get("credits")
    budget = first.get("quality_budget")
    if not isinstance(credits, list) or not all(finite(c) for c in credits) or not finite(budget):
        errors.append("component_budget_or_credits_missing")
    elif not math.isclose(math.fsum(credits), float(budget), abs_tol=1e-9):
        errors.append("component_credit_budget_not_conserved")
    return errors


def raw_case(sample: Any, audit: dict[str, Any]) -> dict[str, Any]:
    """Reviewable raw payload: do not truncate prompt/response source evidence."""
    return {
        "audit": audit,
        "prompt": field(sample, "prompt"),
        "response": field(sample, "response"),
        "metadata_summary": {
            key: metadata(sample).get(key)
            for key in (
                "turn_idx",
                "task_reward",
                "multi_turn_reward",
                "component_reward",
                "source_component_identity",
                "component_reward_soft_finalize_protected",
                "remove_reason",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-cases", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    loaded = torch.load(args.input, map_location="cpu", weights_only=False)
    samples = loaded.get("samples") if isinstance(loaded, dict) else None
    if not isinstance(samples, list):
        raise ValueError("rollout dump must be a dict with a samples list")
    args.output.mkdir(parents=True)
    (args.output / "cases").mkdir()
    audits = [audit_sample(sample, ordinal) for ordinal, sample in enumerate(samples)]
    groups: dict[str, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
    for sample, audit in zip(samples, audits, strict=True):
        groups[str(audit["group_id"])].append((sample, audit))
    group_errors = {group: group_checks(*zip(*items, strict=True)) for group, items in groups.items()}
    errors = [error for audit in audits for error in audit["errors"]] + [
        error for values in group_errors.values() for error in values
    ]
    selected = []
    for group, items in groups.items():
        if group_errors[group] or any(audit["errors"] for _, audit in items):
            selected.append((group, items[0]))
        elif any(audit["component_credit"] not in {None, 0.0} for _, audit in items):
            selected.append((group, items[0]))
        if len(selected) >= args.review_cases:
            break
    for index, (group, (sample, audit)) in enumerate(selected):
        (args.output / "cases" / f"case_{index:02d}_group_{group}.json").write_text(
            json.dumps(raw_case(sample, audit), ensure_ascii=False, indent=2, default=str)
        )
    summary = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "samples": len(samples),
        "trajectories": len(groups),
        "sample_error_counts": dict(Counter(error for audit in audits for error in audit["errors"])),
        "group_error_counts": dict(Counter(error for values in group_errors.values() for error in values)),
        "cross_turn_trajectories": sum(
            bool(items[0][1]["identity"])
            and isinstance(metadata(items[0][0]).get("component_reward"), dict)
            and metadata(items[0][0])["component_reward"].get("resolved_cross_turn_units", 0) > 0
            for items in groups.values()
        ),
        "fallback_trajectories": sum(
            isinstance(metadata(items[0][0]).get("component_reward"), dict)
            and metadata(items[0][0])["component_reward"].get("status") == "unavailable_best_turn_fallback"
            for items in groups.values()
        ),
        "passed": not errors,
        "review_cases": len(selected),
        "contract": "read_only_no_identity_or_token_reconstruction",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.output / "samples.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2, default=str))
    if errors:
        raise SystemExit("audit failed; see summary.json and samples.json")


if __name__ == "__main__":
    main()
