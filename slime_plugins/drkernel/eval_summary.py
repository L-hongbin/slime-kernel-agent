"""Summarize DrKernel/KernelGym evaluation artifacts."""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_FAST_THRESHOLDS = (1.0, 1.2)
KERNELGYM_RESULT_KEYS = {
    "compiled",
    "compilation",
    "correctness",
    "decoy_kernel",
    "is_decoy_kernel",
    "speedup",
    "performance",
}
CONTAINER_KEYS = ("samples", "records", "results", "data")


def load_records(path: str | Path) -> list[Any]:
    """Load common eval artifact formats into top-level candidate records."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return list(iter_candidate_records(records))

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return list(iter_candidate_records(payload))

    if suffix in {".pt", ".pth"}:
        import torch

        payload = torch.load(path, map_location="cpu")
        return list(iter_candidate_records(payload))

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            payload = pickle.load(f)
        return list(iter_candidate_records(payload))

    raise ValueError(f"unsupported eval artifact format: {path}")


def iter_candidate_records(payload: Any) -> Iterable[Any]:
    """Yield likely sample/result records from nested eval artifacts."""

    if isinstance(payload, list | tuple):
        for item in payload:
            yield from iter_candidate_records(item)
        return

    if _extract_kernelgym_response(payload) is not None:
        yield payload
        return

    for key in CONTAINER_KEYS:
        value = _get_value(payload, key)
        if isinstance(value, list | tuple):
            for item in value:
                yield from iter_candidate_records(item)
            return

    if _looks_like_sample_record(payload):
        yield payload


def summarize_records(
    records: Iterable[Any], fast_thresholds: Iterable[float] = DEFAULT_FAST_THRESHOLDS
) -> dict[str, Any]:
    """Compute compile/correctness/fast rates from KernelGym responses.

    `fast@x_rate` uses all candidate records as the denominator, matching the
    practical eval question "how many prompts produced a correct fast kernel".
    `fast@x_correct_rate` uses only correct non-decoy records, matching the
    older DrKernel metric that reports speed among correct submissions.
    """

    records = list(records)
    total = len(records)
    compiled_count = 0
    raw_correct_count = 0
    correct_count = 0
    decoy_count = 0
    missing_response_count = 0
    speedups: list[float] = []
    correct_speedups: list[float] = []

    for record in records:
        response = _extract_kernelgym_response(record)
        if response is None:
            missing_response_count += 1
            continue

        compiled = _as_bool(_get_first(response, ("compiled", "compilation")))
        raw_correct = _as_bool(_get_value(response, "correctness"))
        decoy = _as_bool(_get_first(response, ("decoy_kernel", "is_decoy_kernel")))
        speedup = _as_float(_get_first(response, ("speedup", "performance")))

        if compiled:
            compiled_count += 1
        if raw_correct:
            raw_correct_count += 1
        if decoy:
            decoy_count += 1
        if speedup is not None:
            speedups.append(speedup)

        correct = raw_correct and not decoy
        if correct:
            correct_count += 1
            if speedup is not None:
                correct_speedups.append(speedup)

    evaluated_count = total - missing_response_count
    summary: dict[str, Any] = {
        "total": total,
        "evaluated": evaluated_count,
        "missing_response": missing_response_count,
        "compiled_count": compiled_count,
        "compile_rate": _rate(compiled_count, total),
        "raw_correct_count": raw_correct_count,
        "raw_correctness_rate": _rate(raw_correct_count, total),
        "correct_count": correct_count,
        "correctness_rate": _rate(correct_count, total),
        "decoy_count": decoy_count,
        "decoy_rate": _rate(decoy_count, total),
        "speedup_count": len(speedups),
        "correct_speedup_count": len(correct_speedups),
    }

    for threshold in fast_thresholds:
        label = _threshold_label(threshold)
        fast_count = sum(1 for speedup in correct_speedups if speedup >= threshold)
        summary[f"fast@{label}_count"] = fast_count
        summary[f"fast@{label}_rate"] = _rate(fast_count, total)
        summary[f"fast@{label}_correct_rate"] = _rate(fast_count, correct_count)

    return summary


def _extract_kernelgym_response(record: Any) -> Mapping[str, Any] | None:
    metadata = _get_value(record, "metadata")
    if isinstance(metadata, Mapping):
        response = _extract_from_kernelgym_container(metadata.get("kernelgym"))
        if response is not None:
            return response

    response = _extract_from_kernelgym_container(_get_value(record, "kernelgym"))
    if response is not None:
        return response

    response = _get_value(record, "response")
    if _is_kernelgym_result(response):
        return response

    if _is_kernelgym_result(record):
        return record

    return None


def _extract_from_kernelgym_container(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    response = value.get("response")
    if _is_kernelgym_result(response):
        return response
    if _is_kernelgym_result(value):
        return value
    return None


def _is_kernelgym_result(value: Any) -> bool:
    return isinstance(value, Mapping) and any(key in value for key in KERNELGYM_RESULT_KEYS)


def _looks_like_sample_record(value: Any) -> bool:
    if value is None:
        return False
    return _get_value(value, "metadata") is not None or _get_value(value, "response") is not None


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _get_first(value: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        item = _get_value(value, key)
        if item is not None:
            return item
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _threshold_label(value: float) -> str:
    return str(float(value))
