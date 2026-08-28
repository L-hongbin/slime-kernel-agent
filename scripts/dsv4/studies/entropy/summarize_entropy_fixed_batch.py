#!/usr/bin/env python3
"""Summarize and sanity-check fixed-batch entropy A/B text logs.

The Megatron log line for ``step s`` is emitted after ``optimizer.step()``, but
its loss/entropy dictionary was computed by the forward pass immediately before
that update.  Therefore, when every arm replays the same literal debug dump:

* step 0 entropy is the pre-update baseline;
* ``H(step 1) - H(step 0)`` is the realized effect of update 0;
* ``H(step 2) - H(step 1)`` is the realized effect of update 1.

This script deliberately does not call a tracking service.  It extracts the
Python dictionaries already present in slime's text logs, reports the relevant
train/rollout metrics, and fails closed on observable start-contract mismatches.

Example::

    python scripts/dsv4/studies/entropy/summarize_entropy_fixed_batch.py \
      --arm token_predictive=/tmp/entropy_ab_token_predictive.out \
      --arm token_ppo=/tmp/entropy_ab_token_ppo.out

New fixed-batch runs log ``train/entropy_common_probe`` on the immutable
pre-filter response mask.  Whenever a strict ``mis_tis`` arm is compared, this
common-denominator metric is mandatory for every arm and is used for H0/H1/H2.
Comparisons without strict MIS may still summarize legacy logs by falling back
to ``train/entropy_loss``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_RECORD_RE = re.compile(r"\b(?:[A-Za-z0-9_]+-)?step\s+(?P<step>\d+):\s*(?P<payload>\{.*\})")
ROLLOUT_RECORD_RE = re.compile(r"\b(?:rollout|perf)\s+(?P<step>\d+):\s*(?P<payload>\{.*\})")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
CLIP_CONTRACT_RE = re.compile(
    r"^(?P<base>.*),eps_clip=(?P<low>[^,]+)," r"eps_clip_high=(?P<high>[^,]+),eps_clip_c=(?P<c>[^,]+)$"
)
LORA_ZERO_OK_MARKER = "[V4_ENTROPY_AB_ZERO_LORA_OUT_OK]"
LORA_ZERO_FAIL_MARKER = "[V4_ENTROPY_AB_ZERO_LORA_OUT_FAIL]"

HISTORICAL_MIS_CONFIG = {
    "aggregation": "turns_geometric",
    "token_veto_threshold": 1e-4,
    "lower": 0.99,
    "upper": 1.01,
    "use_advantage": False,
}
OLDACTOR_TIS_NOOP_MIS_CONFIG = {"aggregation": "turns_geometric"}

DEFAULT_BATCH_CHECK_KEYS = (
    "train/global_batch_size",
    "rollout/response_lengths",
    "rollout/total_lengths",
    "rollout/raw_reward",
    "rollout/rewards",
    "rollout/truncated",
)

CORE_TRAIN_KEYS = {
    "train/loss",
    "train/pg_loss",
    "train/entropy_loss",
    "train/entropy_common_probe",
    "train/pg_clipfrac",
    "train/pg_upper_clipfrac",
    "train/pg_lower_clipfrac",
    "train/pg_dual_clipfrac",
    "train/ppo_kl",
    "train/train_rollout_logprob_abs_diff",
    "train/global_batch_size",
}

MEASUREMENT_NOTES = (
    "step s metrics come from the forward before update s, even though the text line is emitted after optimizer.step",
    "step0 is pre-update; step1-step0 and step2-step1 are realized same-batch post-update entropy deltas",
    "dppo/* first-order values are logit-space predictors, while delta-H is the observed optimizer result",
)

COMMON_PROBE_METRIC = "train/entropy_common_probe"
LEGACY_ENTROPY_METRIC = "train/entropy_loss"


class SummaryError(ValueError):
    """Raised when a log cannot be parsed or violates an unambiguous contract."""


class _NonFiniteNameTransformer(ast.NodeTransformer):
    """Allow logger representations of nan/inf without permitting eval()."""

    _VALUES = {
        "nan": float("nan"),
        "NaN": float("nan"),
        "inf": float("inf"),
        "Infinity": float("inf"),
    }

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in self._VALUES:
            raise SummaryError(f"unsupported name {node.id!r} in metric dictionary")
        return ast.copy_location(ast.Constant(self._VALUES[node.id]), node)


def _parse_metric_dict(raw: str, *, source: str, line_number: int) -> dict[str, Any]:
    try:
        tree = ast.parse(raw, mode="eval")
        tree = _NonFiniteNameTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        value = ast.literal_eval(tree)
    except (SyntaxError, ValueError, TypeError, SummaryError) as exc:
        raise SummaryError(f"{source}:{line_number}: malformed metric dictionary: {exc}") from exc
    if not isinstance(value, dict):
        raise SummaryError(f"{source}:{line_number}: metric payload is not a dictionary")
    return value


def _numeric_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[key] = float(value)
    return metrics


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return left == right


def _merge_record(
    records: dict[int, dict[str, float]],
    step: int,
    incoming: dict[str, float],
    *,
    source: str,
    line_number: int,
) -> None:
    existing = records.setdefault(step, {})
    for key, value in incoming.items():
        if key in existing and not _same_value(existing[key], value):
            raise SummaryError(
                f"{source}:{line_number}: conflicting duplicate value for step={step} "
                f"metric={key}: {existing[key]} vs {value}"
            )
        existing[key] = value


def _set_metadata(metadata: dict[str, Any], key: str, value: Any, *, source: str, line_number: int) -> None:
    if key in metadata and not _same_value(metadata[key], value):
        raise SummaryError(f"{source}:{line_number}: conflicting metadata {key}: {metadata[key]!r} vs {value!r}")
    metadata[key] = value


def _parse_bool_or_text(raw: str) -> bool | str:
    if raw in {"True", "true", "1"}:
        return True
    if raw in {"False", "false", "0"}:
        return False
    return raw


def _extract_metadata(line: str, metadata: dict[str, Any], *, source: str, line_number: int) -> None:
    stripped = line.strip()
    if LORA_ZERO_OK_MARKER in stripped:
        _set_metadata(metadata, "lora_zero_ok", True, source=source, line_number=line_number)
    if LORA_ZERO_FAIL_MARKER in stripped:
        _set_metadata(metadata, "lora_zero_fail", True, source=source, line_number=line_number)
    exact_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("arm", re.compile(r"^entropy_ab_arm=(\S+)$")),
        ("repo", re.compile(r"^repo=(\S+)$")),
        ("hf", re.compile(r"^hf=(\S+)$")),
        ("load", re.compile(r"^load=(\S+)$")),
        ("reduction", re.compile(r"^reduction=(\S+)$")),
        ("debug_data_sha256", re.compile(r"^debug_data_sha256=(\S+)$")),
        ("source_debug_data", re.compile(r"^source_debug_data=(\S+)$")),
        ("source_debug_data_sha256", re.compile(r"^source_debug_data_sha256=(\S+)$")),
        ("critical_code_sha256", re.compile(r"^critical_code_sha256=(\S+)$")),
        ("base_manifest_sha256", re.compile(r"^base_manifest_sha256=(\S+)$")),
        ("megatron_revision", re.compile(r"^megatron_revision=(\S+)$")),
        ("tilekernels_revision", re.compile(r"^tilekernels_revision=(\S+)$")),
        ("resolved_contract", re.compile(r"^resolved_contract=(\S+)$")),
        ("code_revision", re.compile(r"^code_revision=(\S+)$")),
        ("git_revision", re.compile(r"^git_revision=(\S+)$")),
        ("sequence_mis_config", re.compile(r"^sequence_mis_config=(.+)$")),
    )
    for key, pattern in exact_patterns:
        match = pattern.search(stripped)
        if match:
            _set_metadata(metadata, key, match.group(1), source=source, line_number=line_number)

    debug_match = re.match(r"^debug_data=(\S+)\s+subsample=(\S+)$", stripped)
    if debug_match:
        _set_metadata(metadata, "debug_data", debug_match.group(1), source=source, line_number=line_number)
        _set_metadata(
            metadata,
            "debug_subsample",
            float(debug_match.group(2)),
            source=source,
            line_number=line_number,
        )

    batch_match = re.match(
        r"^global_batch_size=(\d+)\s+rollout_batch_size=(\d+).*\bnum_rollout=(\d+)$",
        stripped,
    )
    if batch_match:
        _set_metadata(metadata, "global_batch_size", int(batch_match.group(1)), source=source, line_number=line_number)
        _set_metadata(
            metadata, "rollout_batch_size", int(batch_match.group(2)), source=source, line_number=line_number
        )
        _set_metadata(metadata, "num_rollout", int(batch_match.group(3)), source=source, line_number=line_number)

    # Megatron's argument table remains available even when the arm-specific
    # preamble was not redirected into the final LOG file.
    argument_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("debug_data", re.compile(r"\bload_debug_rollout_data\s+\.+\s+(\S+)\s*$")),
        ("debug_subsample", re.compile(r"\bload_debug_rollout_data_subsample\s+\.+\s+(\S+)\s*$")),
        ("calculate_per_token_loss", re.compile(r"\bcalculate_per_token_loss\s+\.+\s+(\S+)\s*$")),
        ("custom_pg_reducer", re.compile(r"\bcustom_pg_loss_reducer_function_path\s+\.+\s+(\S+)\s*$")),
        ("policy_loss_mode", re.compile(r"\bpolicy_loss_mode\s+\.+\s+(\S+)\s*$")),
        ("eps_clip", re.compile(r"\beps_clip\s+\.+\s+(\S+)\s*$")),
        ("eps_clip_high", re.compile(r"\beps_clip_high\s+\.+\s+(\S+)\s*$")),
        ("eps_clip_c", re.compile(r"\beps_clip_c\s+\.+\s+(\S+)\s*$")),
        ("no_load_optim", re.compile(r"\bno_load_optim\s+\.+\s+(\S+)\s*$")),
        ("no_load_rng", re.compile(r"\bno_load_rng\s+\.+\s+(\S+)\s*$")),
        ("seed", re.compile(r"\bseed\s+\.+\s+(\S+)\s*$")),
    )
    for key, pattern in argument_patterns:
        match = pattern.search(stripped)
        if not match:
            continue
        raw_value = match.group(1)
        if key == "debug_subsample" and raw_value not in {"None", "null"}:
            value: Any = float(raw_value)
        elif key in {"eps_clip", "eps_clip_high", "eps_clip_c"}:
            value = None if raw_value in {"None", "null"} else float(raw_value)
        elif key in {"calculate_per_token_loss", "no_load_optim", "no_load_rng"}:
            value = _parse_bool_or_text(raw_value)
        elif key == "seed":
            value = int(raw_value)
        else:
            value = raw_value
        _set_metadata(metadata, key, value, source=source, line_number=line_number)

    subsample_match = re.search(
        r"Subsample loaded debug rollout data using ratio=([0-9.eE+-]+).*num rows (\d+) -> (\d+)",
        stripped,
    )
    if subsample_match:
        _set_metadata(
            metadata, "debug_subsample", float(subsample_match.group(1)), source=source, line_number=line_number
        )
        _set_metadata(
            metadata, "debug_rows_before", int(subsample_match.group(2)), source=source, line_number=line_number
        )
        _set_metadata(
            metadata, "debug_rows_after", int(subsample_match.group(3)), source=source, line_number=line_number
        )


def _infer_reduction(metadata: dict[str, Any]) -> str | None:
    if "reduction" in metadata:
        return str(metadata["reduction"])
    per_token = metadata.get("calculate_per_token_loss")
    reducer = str(metadata.get("custom_pg_reducer", ""))
    if per_token is True:
        return "global_token"
    if per_token is False and "get_completion_mean_pg_loss_reducer" in reducer:
        return "completion_equal"
    if per_token is False:
        return "group_mean"
    return None


@dataclass
class ArmLog:
    name: str
    path: Path
    metadata: dict[str, Any]
    train_steps: dict[int, dict[str, float]]
    rollout_steps: dict[int, dict[str, float]]


def parse_arm_log(name: str, path: Path) -> ArmLog:
    metadata: dict[str, Any] = {}
    train_steps: dict[int, dict[str, float]] = {}
    rollout_steps: dict[int, dict[str, float]] = {}
    source = str(path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise SummaryError(f"cannot read {source}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, 1):
        line = ANSI_RE.sub("", raw_line)
        _extract_metadata(line, metadata, source=source, line_number=line_number)

        train_match = TRAIN_RECORD_RE.search(line)
        if train_match:
            payload = _parse_metric_dict(train_match.group("payload"), source=source, line_number=line_number)
            numeric = _numeric_metrics(payload)
            if "train/step" in numeric and any(key.startswith("train/") for key in numeric):
                step = int(train_match.group("step"))
                payload_step = int(numeric["train/step"])
                if payload_step != step:
                    raise SummaryError(
                        f"{source}:{line_number}: text step {step} disagrees with train/step={payload_step}"
                    )
                _merge_record(train_steps, step, numeric, source=source, line_number=line_number)

        rollout_match = ROLLOUT_RECORD_RE.search(line)
        if rollout_match:
            payload = _parse_metric_dict(rollout_match.group("payload"), source=source, line_number=line_number)
            numeric = {key: value for key, value in _numeric_metrics(payload).items() if key.startswith("rollout/")}
            if numeric:
                _merge_record(
                    rollout_steps,
                    int(rollout_match.group("step")),
                    numeric,
                    source=source,
                    line_number=line_number,
                )

    inferred_reduction = _infer_reduction(metadata)
    if inferred_reduction is not None:
        metadata["reduction"] = inferred_reduction
    if not train_steps:
        raise SummaryError(f"{source}: no actor train step dictionaries found")
    return ArmLog(name=name, path=path, metadata=metadata, train_steps=train_steps, rollout_steps=rollout_steps)


def _is_train_metric_selected(key: str) -> bool:
    if key in CORE_TRAIN_KEYS:
        return True
    if key.startswith("dppo/"):
        return "first_order" in key or key.startswith("dppo/signed_update_")
    if not key.startswith("train/"):
        return False
    tail = key.removeprefix("train/").lower()
    return "tis" in tail or tail == "ois" or "is_ratio" in tail or tail.startswith("mis_")


def _selected_train_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] for key in sorted(metrics) if _is_train_metric_selected(key)}


def _close(values: Iterable[float], *, rel_tol: float, abs_tol: float) -> bool:
    values = list(values)
    if not values or any(not math.isfinite(value) for value in values):
        return False
    reference = values[0]
    return all(math.isclose(reference, value, rel_tol=rel_tol, abs_tol=abs_tol) for value in values[1:])


def _spread(values: Iterable[float]) -> float:
    values = list(values)
    return max(values) - min(values)


def _check_entry(level: str, name: str, detail: str) -> dict[str, str]:
    return {"level": level, "name": name, "detail": detail}


def _is_strict_mis_arm(arm: ArmLog) -> bool:
    """Return whether an arm applies the canonical strict sequence-MIS filter."""

    return str(arm.metadata.get("arm", arm.name)) == "mis_tis"


def _has_common_probe_for_all_observed_steps(arm: ArmLog) -> bool:
    return bool(arm.train_steps) and all(COMMON_PROBE_METRIC in metrics for metrics in arm.train_steps.values())


def _declares_new_common_probe_contract(arm: ArmLog) -> bool:
    raw_contract = arm.metadata.get("resolved_contract")
    if raw_contract is None or "common_probe=original_mask_global_token" not in str(raw_contract):
        return False
    _, clip_contract = _split_clip_contract(str(raw_contract))
    return clip_contract is not None


def _split_clip_contract(raw_contract: str) -> tuple[str, tuple[float, float, float | None] | None]:
    match = CLIP_CONTRACT_RE.fullmatch(raw_contract)
    if match is None:
        if ",eps_clip=" in raw_contract:
            raise SummaryError(f"malformed clip fields in resolved_contract={raw_contract!r}")
        return raw_contract, None
    try:
        eps_clip = float(match.group("low"))
        eps_clip_high = float(match.group("high"))
        raw_c = match.group("c")
        eps_clip_c = None if raw_c == "none" else float(raw_c)
    except ValueError as exc:
        raise SummaryError(f"non-numeric clip field in resolved_contract={raw_contract!r}") from exc
    values = (eps_clip, eps_clip_high, eps_clip_c)
    if any(value is not None and not math.isfinite(value) for value in values):
        raise SummaryError(f"non-finite clip field in resolved_contract={raw_contract!r}")
    return match.group("base"), values


def _expected_clip_contract(arm: ArmLog) -> tuple[float, float, float | None] | None:
    label = str(arm.metadata.get("arm", arm.name))
    if label in {"mis_tis", "oldactor_tis"}:
        return (0.20, 0.28, None)
    if label == "token_predictive_nomask":
        return (1e30, 1e30, 5.0)
    if label in {
        "token_predictive",
        "token_ppo",
        "ppo_rollout_denom",
        "ppo_recompute_denom",
        "completion_predictive",
        "completion_ppo",
    }:
        return (0.20, 0.20, 5.0)
    return None


def build_summary(
    arms: list[ArmLog],
    *,
    entropy_atol: float = 1e-4,
    entropy_rtol: float = 1e-4,
    batch_atol: float = 1e-8,
    batch_rtol: float = 1e-8,
    start_mismatch: str = "fail",
    allow_incomplete: bool = False,
    batch_check_keys: tuple[str, ...] = DEFAULT_BATCH_CHECK_KEYS,
) -> dict[str, Any]:
    if len(arms) < 2:
        raise SummaryError("fixed-batch A/B summary requires at least two --arm logs")
    if start_mismatch not in {"fail", "warn"}:
        raise SummaryError(f"invalid start_mismatch={start_mismatch!r}")
    if len({arm.name for arm in arms}) != len(arms):
        raise SummaryError("arm names must be unique")

    checks: list[dict[str, str]] = []

    def mismatch(name: str, detail: str) -> None:
        checks.append(_check_entry("FAIL" if start_mismatch == "fail" else "WARN", name, detail))

    required_steps = (0, 1, 2)
    strict_mis_present = any(_is_strict_mis_arm(arm) for arm in arms)
    new_common_probe_contract_present = any(_declares_new_common_probe_contract(arm) for arm in arms)
    common_probe_required = strict_mis_present or new_common_probe_contract_present
    all_arms_have_common_probe = all(_has_common_probe_for_all_observed_steps(arm) for arm in arms)
    entropy_metric = (
        COMMON_PROBE_METRIC if common_probe_required or all_arms_have_common_probe else LEGACY_ENTROPY_METRIC
    )
    for arm in arms:
        if arm.metadata.get("lora_zero_fail") is True:
            checks.append(_check_entry("FAIL", f"{arm.name}:lora_zero", f"observed {LORA_ZERO_FAIL_MARKER}"))
        elif arm.metadata.get("lora_zero_ok") is not True:
            checks.append(_check_entry("FAIL", f"{arm.name}:lora_zero", f"missing {LORA_ZERO_OK_MARKER}"))
        else:
            checks.append(_check_entry("PASS", f"{arm.name}:lora_zero", f"observed {LORA_ZERO_OK_MARKER}"))
        missing_steps = [step for step in required_steps if step not in arm.train_steps]
        if missing_steps:
            level = "WARN" if allow_incomplete else "FAIL"
            checks.append(_check_entry(level, f"{arm.name}:required_steps", f"missing train steps {missing_steps}"))
        for step, metrics in sorted(arm.train_steps.items()):
            entropy = metrics.get(entropy_metric)
            if entropy is None:
                detail = f"{entropy_metric} missing"
                if strict_mis_present:
                    detail += "; strict mis_tis comparisons require the original-mask common probe"
                elif new_common_probe_contract_present:
                    detail += "; a new-generation common-probe contract requires it on every arm"
                checks.append(
                    _check_entry(
                        "FAIL",
                        f"{arm.name}:step{step}:entropy",
                        detail,
                    )
                )
            elif not math.isfinite(entropy):
                checks.append(
                    _check_entry(
                        "FAIL",
                        f"{arm.name}:step{step}:entropy",
                        f"non-finite {entropy_metric} value {entropy}",
                    )
                )
            for key, value in _selected_train_metrics(metrics).items():
                if not math.isfinite(value):
                    checks.append(_check_entry("FAIL", f"{arm.name}:step{step}:{key}", f"non-finite value {value}"))
            for key, value in arm.rollout_steps.get(step, {}).items():
                if not math.isfinite(value):
                    checks.append(_check_entry("FAIL", f"{arm.name}:rollout{step}:{key}", f"non-finite value {value}"))

        label = arm.metadata.get("arm", arm.name)
        raw_contract = arm.metadata.get("resolved_contract")
        if raw_contract is not None:
            _, clip_contract = _split_clip_contract(str(raw_contract))
            if clip_contract is None:
                if "common_probe=original_mask_global_token" in str(raw_contract):
                    checks.append(
                        _check_entry(
                            "WARN",
                            f"{arm.name}:clip_contract",
                            "legacy common-probe contract predates explicit eps_clip/eps_clip_high/eps_clip_c",
                        )
                    )
            else:
                expected_clip = _expected_clip_contract(arm)
                if expected_clip is not None and clip_contract != expected_clip:
                    checks.append(
                        _check_entry(
                            "FAIL",
                            f"{arm.name}:clip_contract",
                            f"expected {expected_clip}, got {clip_contract}",
                        )
                    )
                else:
                    checks.append(
                        _check_entry(
                            "PASS",
                            f"{arm.name}:clip_contract",
                            f"resolved exact config {clip_contract}",
                        )
                    )

                clip_arg_keys = ("eps_clip", "eps_clip_high", "eps_clip_c")
                missing_clip_args = [key for key in clip_arg_keys if key not in arm.metadata]
                if missing_clip_args:
                    checks.append(
                        _check_entry(
                            "FAIL",
                            f"{arm.name}:clip_args",
                            f"Megatron argument table is missing {missing_clip_args}",
                        )
                    )
                else:
                    clip_args = tuple(arm.metadata[key] for key in clip_arg_keys)
                    if clip_args != clip_contract:
                        checks.append(
                            _check_entry(
                                "FAIL",
                                f"{arm.name}:clip_args",
                                f"argument table {clip_args} disagrees with contract {clip_contract}",
                            )
                        )
                    else:
                        checks.append(
                            _check_entry(
                                "PASS",
                                f"{arm.name}:clip_args",
                                f"argument table matches {clip_contract}",
                            )
                        )
        if "predictive" in str(label):
            first_order = [
                key
                for key in arm.train_steps.get(0, {})
                if key.startswith("dppo/") and ("first_order" in key or key.startswith("dppo/signed_update_"))
            ]
            if not first_order:
                checks.append(
                    _check_entry(
                        "WARN", f"{arm.name}:dppo_first_order", "predictive arm has no dppo first-order metrics"
                    )
                )
        if str(label) in {"mis_tis", "oldactor_tis"}:
            train0 = arm.train_steps.get(0, {})
            if not any("tis" in key.lower() or key.endswith("/ois") for key in train0):
                checks.append(_check_entry("WARN", f"{arm.name}:tis", f"{label} arm has no train TIS/OIS metrics"))
        if str(label) == "mis_tis":
            has_rollout_mis = any("mis" in key.lower() for metrics in arm.rollout_steps.values() for key in metrics)
            if not has_rollout_mis:
                checks.append(_check_entry("FAIL", f"{arm.name}:mis", "mis_tis arm has no rollout MIS metrics"))
        expected_config = HISTORICAL_MIS_CONFIG if str(label) == "mis_tis" else OLDACTOR_TIS_NOOP_MIS_CONFIG
        raw_config = arm.metadata.get("sequence_mis_config")
        if raw_config is None:
            checks.append(
                _check_entry(
                    "FAIL" if str(label) == "mis_tis" else "WARN",
                    f"{arm.name}:sequence_mis_config",
                    "not observable in log",
                )
            )
        else:
            try:
                parsed_config = json.loads(str(raw_config))
            except json.JSONDecodeError as exc:
                checks.append(
                    _check_entry(
                        "FAIL",
                        f"{arm.name}:sequence_mis_config",
                        f"invalid JSON: {exc}",
                    )
                )
            else:
                if parsed_config != expected_config:
                    checks.append(
                        _check_entry(
                            "FAIL",
                            f"{arm.name}:sequence_mis_config",
                            f"expected {expected_config}, got {parsed_config}",
                        )
                    )
                else:
                    checks.append(
                        _check_entry(
                            "PASS",
                            f"{arm.name}:sequence_mis_config",
                            f"resolved exact config {parsed_config}",
                        )
                    )
        if str(label) != "mis_tis":
            reject_rates = [
                (step, metrics["rollout/mis_reject_rate"])
                for step, metrics in sorted(arm.rollout_steps.items())
                if "rollout/mis_reject_rate" in metrics
            ]
            if not reject_rates:
                checks.append(
                    _check_entry(
                        "PASS",
                        f"{arm.name}:mis_noop",
                        "canonical no-bound/no-veto config; reject metric absent when train log-prob postprocess is skipped",
                    )
                )
            elif any(rate != 0.0 for _, rate in reject_rates):
                checks.append(
                    _check_entry(
                        "FAIL",
                        f"{arm.name}:mis_noop",
                        f"canonical no-filter arm must reject no sequences, got {reject_rates}",
                    )
                )
            else:
                checks.append(_check_entry("PASS", f"{arm.name}:mis_noop", "mis_reject_rate=0 at every observed step"))

        for digest_key in (
            "source_debug_data_sha256",
            "debug_data_sha256",
            "critical_code_sha256",
            "base_manifest_sha256",
        ):
            digest = arm.metadata.get(digest_key)
            if digest is not None and not SHA256_RE.fullmatch(str(digest)):
                checks.append(
                    _check_entry(
                        "FAIL",
                        f"{arm.name}:{digest_key}",
                        f"expected 64 lowercase hex characters, got {digest!r}",
                    )
                )
        git_revision = arm.metadata.get("git_revision")
        if git_revision is not None and not GIT_REVISION_RE.fullmatch(str(git_revision)):
            checks.append(
                _check_entry(
                    "FAIL",
                    f"{arm.name}:git_revision",
                    f"expected a 40-64 character lowercase hex object id, got {git_revision!r}",
                )
            )
        for revision_key in ("megatron_revision", "tilekernels_revision"):
            revision = arm.metadata.get(revision_key)
            if revision is not None and not GIT_REVISION_RE.fullmatch(str(revision)):
                checks.append(
                    _check_entry(
                        "FAIL",
                        f"{arm.name}:{revision_key}",
                        f"expected a 40-64 character lowercase hex object id, got {revision!r}",
                    )
                )

    # Observable start contract. Missing metadata cannot prove divergence, but
    # it must be called out so aggregate similarity is not mistaken for identity.
    metadata_keys = (
        "hf",
        "load",
        "repo",
        "git_revision",
        "critical_code_sha256",
        "base_manifest_sha256",
        "megatron_revision",
        "tilekernels_revision",
        "source_debug_data_sha256",
        "debug_subsample",
        "global_batch_size",
        "no_load_optim",
        "no_load_rng",
        "seed",
    )
    for key in metadata_keys:
        present = [(arm.name, arm.metadata[key]) for arm in arms if key in arm.metadata]
        if len(present) != len(arms):
            missing = [arm.name for arm in arms if key not in arm.metadata]
            checks.append(_check_entry("WARN", f"metadata:{key}", f"not observable in logs for {missing}"))
            continue
        unique = {repr(value) for _, value in present}
        if len(unique) != 1:
            checks.append(_check_entry("FAIL", f"metadata:{key}", f"arm values differ: {present}"))
        else:
            checks.append(_check_entry("PASS", f"metadata:{key}", f"shared value {present[0][1]!r}"))

    # Clip values are an intentional loss-axis difference: historical
    # MIS/old-actor arms reproduce donor ivbx2rz8 (0.20/0.28/no-C), whereas
    # direct/predictive controls retain their declared clipping.  Compare the
    # common portion of the resolved contract and validate clips per arm above.
    contracts = [
        (arm.name, str(arm.metadata["resolved_contract"])) for arm in arms if "resolved_contract" in arm.metadata
    ]
    if len(contracts) != len(arms):
        missing = [arm.name for arm in arms if "resolved_contract" not in arm.metadata]
        checks.append(_check_entry("WARN", "metadata:resolved_contract", f"not observable in logs for {missing}"))
    else:
        normalized_contracts = [(name, _split_clip_contract(raw_contract)[0]) for name, raw_contract in contracts]
        if len({contract for _, contract in normalized_contracts}) != 1:
            checks.append(
                _check_entry(
                    "FAIL",
                    "metadata:resolved_contract",
                    f"non-clip contract fields differ: {normalized_contracts}",
                )
            )
        else:
            checks.append(
                _check_entry(
                    "PASS",
                    "metadata:resolved_contract",
                    f"shared non-clip contract {normalized_contracts[0][1]!r}",
                )
            )

    # Effective replay paths may intentionally differ: old-actor/TIS arms use a
    # derived v0-stamped copy, while direct rollout-logprob arms can consume the
    # immutable source.  The source SHA is the cross-arm batch identity.  Still
    # require identical bytes whenever two arms claim the same effective path,
    # and retain the old fail-closed path check when source SHA provenance is
    # unavailable (legacy logs).
    effective_paths = [(arm.name, arm.metadata.get("debug_data")) for arm in arms]
    effective_hashes = [(arm.name, arm.metadata.get("debug_data_sha256")) for arm in arms]
    have_shared_source_sha = (
        all("source_debug_data_sha256" in arm.metadata for arm in arms)
        and len({arm.metadata["source_debug_data_sha256"] for arm in arms}) == 1
    )
    if all(path is not None for _, path in effective_paths):
        unique_paths = {str(path) for _, path in effective_paths}
        if len(unique_paths) == 1:
            if (
                all(digest is not None for _, digest in effective_hashes)
                and len({str(digest) for _, digest in effective_hashes}) == 1
            ):
                checks.append(
                    _check_entry(
                        "PASS",
                        "metadata:debug_data",
                        f"shared effective path/hash {effective_paths[0][1]!r}",
                    )
                )
            else:
                checks.append(
                    _check_entry(
                        "FAIL",
                        "metadata:debug_data_sha256",
                        f"same effective path has differing/missing hashes: {effective_hashes}",
                    )
                )
        elif (
            all(digest is not None for _, digest in effective_hashes)
            and len({str(digest) for _, digest in effective_hashes}) == 1
        ):
            checks.append(
                _check_entry(
                    "PASS",
                    "metadata:debug_data",
                    f"effective paths differ but bytes share SHA256={effective_hashes[0][1]}: {effective_paths}",
                )
            )
        elif have_shared_source_sha:
            checks.append(
                _check_entry(
                    "WARN",
                    "metadata:debug_data",
                    f"effective paths differ (possibly original vs v0-derived), but source SHA is shared: {effective_paths}",
                )
            )
        else:
            checks.append(
                _check_entry(
                    "FAIL",
                    "metadata:debug_data",
                    f"effective paths differ without a verified shared source SHA: {effective_paths}",
                )
            )
    else:
        missing = [name for name, path in effective_paths if path is None]
        checks.append(_check_entry("WARN", "metadata:debug_data", f"not observable in logs for {missing}"))

    debug_paths = [str(arm.metadata["debug_data"]) for arm in arms if "debug_data" in arm.metadata]
    if any("{rollout_id}" in path for path in debug_paths):
        checks.append(
            _check_entry(
                "FAIL",
                "fixed_dump_literal",
                "debug_data contains {rollout_id}; step-to-step delta is not guaranteed to replay one batch",
            )
        )

    # The common probe always has one original-mask/global-token denominator.
    # Legacy entropy_loss is comparable only when the arm reducer is the same.
    step0_entropies = {
        arm.name: arm.train_steps[0][entropy_metric]
        for arm in arms
        if 0 in arm.train_steps and entropy_metric in arm.train_steps[0]
    }
    reductions = {arm.name: arm.metadata.get("reduction") for arm in arms}
    if len(step0_entropies) == len(arms):
        values = list(step0_entropies.values())
        if _close(values, rel_tol=entropy_rtol, abs_tol=entropy_atol):
            checks.append(
                _check_entry(
                    "PASS",
                    "step0_entropy",
                    f"all arms close; spread={_spread(values):.9g}, values={step0_entropies}",
                )
            )
        elif (
            entropy_metric == LEGACY_ENTROPY_METRIC
            and len({value for value in reductions.values() if value is not None}) > 1
        ):
            checks.append(
                _check_entry(
                    "WARN",
                    "step0_entropy_cross_reducer",
                    f"spread={_spread(values):.9g}; reducers={reductions}; values={step0_entropies}",
                )
            )
        else:
            mismatch(
                "step0_entropy",
                f"same-reducer arms exceed tolerance; spread={_spread(values):.9g}, values={step0_entropies}",
            )

        if entropy_metric == LEGACY_ENTROPY_METRIC:
            by_reduction: dict[str, list[tuple[str, float]]] = {}
            for arm in arms:
                reduction = str(arm.metadata.get("reduction", "unknown"))
                by_reduction.setdefault(reduction, []).append((arm.name, step0_entropies[arm.name]))
            for reduction, items in sorted(by_reduction.items()):
                if len(items) < 2:
                    continue
                values = [value for _, value in items]
                if not _close(values, rel_tol=entropy_rtol, abs_tol=entropy_atol):
                    mismatch(f"step0_entropy:{reduction}", f"values differ: {items}")

    # Batch checks use exact same-denominator aggregates whenever available.
    for key in batch_check_keys:
        values: list[tuple[str, float]] = []
        for arm in arms:
            if key.startswith("train/"):
                value = arm.train_steps.get(0, {}).get(key)
            else:
                value = arm.rollout_steps.get(0, {}).get(key)
            if value is not None:
                values.append((arm.name, value))
        if not values:
            checks.append(_check_entry("WARN", f"batch:{key}", "metric absent from every arm"))
            continue
        if len(values) != len(arms):
            missing = [arm.name for arm in arms if arm.name not in {name for name, _ in values}]
            checks.append(_check_entry("WARN", f"batch:{key}", f"metric absent for {missing}"))
            continue
        numeric_values = [value for _, value in values]
        if _close(numeric_values, rel_tol=batch_rtol, abs_tol=batch_atol):
            checks.append(_check_entry("PASS", f"batch:{key}", f"shared/close values {values}"))
        else:
            mismatch(f"batch:{key}", f"values differ: {values}")

    arm_output: dict[str, Any] = {}
    for arm in arms:
        steps: dict[str, Any] = {}
        previous_entropy: float | None = None
        previous_step: int | None = None
        for step, raw_metrics in sorted(arm.train_steps.items()):
            entropy = raw_metrics.get(entropy_metric)
            delta_h = (
                entropy - previous_entropy
                if previous_step is not None
                and step == previous_step + 1
                and previous_entropy is not None
                and entropy is not None
                else None
            )
            steps[str(step)] = {
                "phase": "pre_update_baseline" if step == 0 else f"post_update_{step - 1}",
                "entropy": entropy,
                "delta_h_from_previous_step": delta_h,
                "train_metrics": _selected_train_metrics(raw_metrics),
                "rollout_metrics": dict(sorted(arm.rollout_steps.get(step, {}).items())),
            }
            previous_entropy = entropy
            previous_step = step
        arm_output[arm.name] = {
            "path": str(arm.path),
            "metadata": dict(sorted(arm.metadata.items())),
            "steps": steps,
        }

    levels = {check["level"] for check in checks}
    status = "FAIL" if "FAIL" in levels else "WARN" if "WARN" in levels else "PASS"
    measurement_notes = list(MEASUREMENT_NOTES)
    if entropy_metric == COMMON_PROBE_METRIC:
        measurement_notes.append(
            "H/delta-H use train/entropy_common_probe: the original pre-MIS "
            "response mask with a global-token denominator"
        )
    else:
        measurement_notes.append(
            "legacy fallback: H/delta-H use train/entropy_loss, which follows each arm's configured loss reducer"
        )
    return {
        "schema_version": 2,
        "status": status,
        "entropy_metric": entropy_metric,
        "measurement_notes": measurement_notes,
        "start_checks": checks,
        "arms": arm_output,
    }


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.9g}" if value < 0 else f"{value:.9g}"


def _format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={_format_float(value)}" for key, value in metrics.items()) or "(none)"


def render_text(summary: dict[str, Any]) -> str:
    lines = [f"fixed-batch entropy A/B status: {summary['status']}", ""]
    lines.append(f"Entropy metric: {summary['entropy_metric']}")
    lines.append("")
    lines.append("Step semantics:")
    lines.extend(f"  - {note}" for note in summary["measurement_notes"])
    lines.append("")
    lines.append("Start/batch checks:")
    for check in summary["start_checks"]:
        lines.append(f"  [{check['level']}] {check['name']}: {check['detail']}")

    lines.append("")
    lines.append("Observed steps:")
    for arm_name, arm in summary["arms"].items():
        reduction = arm["metadata"].get("reduction", "unknown")
        lines.append(f"  arm={arm_name} reduction={reduction} path={arm['path']}")
        for step, values in arm["steps"].items():
            lines.append(
                f"    step={step} phase={values['phase']} H={_format_float(values['entropy'])} "
                f"delta_H={_format_float(values['delta_h_from_previous_step'])}"
            )
            core = {
                key: value
                for key, value in values["train_metrics"].items()
                if key in CORE_TRAIN_KEYS
                and key
                not in {
                    "train/entropy_loss",
                    "train/entropy_common_probe",
                    "train/global_batch_size",
                }
            }
            dppo = {key: value for key, value in values["train_metrics"].items() if key.startswith("dppo/")}
            correction = {
                key: value
                for key, value in values["train_metrics"].items()
                if key not in CORE_TRAIN_KEYS and not key.startswith("dppo/")
            }
            rollout = {key: value for key, value in values["rollout_metrics"].items() if "mis" not in key.lower()}
            mis = {key: value for key, value in values["rollout_metrics"].items() if "mis" in key.lower()}
            lines.append(f"      pg/clip: {_format_metrics(core)}")
            if dppo:
                lines.append(f"      dppo-first-order: {_format_metrics(dppo)}")
            if rollout:
                lines.append(f"      rollout: {_format_metrics(rollout)}")
            if correction or mis:
                lines.append(f"      MIS/TIS: {_format_metrics({**mis, **correction})}")
    return "\n".join(lines) + "\n"


def _parse_arm_argument(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--arm must be NAME=PATH")
    name, path = raw.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--arm must contain a non-empty NAME and PATH")
    return name, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", type=_parse_arm_argument, required=True, metavar="NAME=PATH")
    parser.add_argument("--json", action="store_true", help="emit the complete machine-readable summary")
    parser.add_argument("--entropy-atol", type=float, default=1e-4)
    parser.add_argument("--entropy-rtol", type=float, default=1e-4)
    parser.add_argument("--batch-atol", type=float, default=1e-8)
    parser.add_argument("--batch-rtol", type=float, default=1e-8)
    parser.add_argument(
        "--start-mismatch",
        choices=("fail", "warn"),
        default="fail",
        help="whether numeric step0/batch mismatches are fatal (metadata identity mismatches always fail)",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true", help="warn instead of fail before steps 0,1,2 all exist"
    )
    parser.add_argument("--warnings-as-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        arms = [parse_arm_log(name, path) for name, path in args.arm]
        summary = build_summary(
            arms,
            entropy_atol=args.entropy_atol,
            entropy_rtol=args.entropy_rtol,
            batch_atol=args.batch_atol,
            batch_rtol=args.batch_rtol,
            start_mismatch=args.start_mismatch,
            allow_incomplete=args.allow_incomplete,
        )
    except SummaryError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(render_text(summary), end="")
    if summary["status"] == "FAIL" or (args.warnings_as_errors and summary["status"] == "WARN"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
