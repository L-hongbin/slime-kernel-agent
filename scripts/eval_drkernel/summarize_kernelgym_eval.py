#!/usr/bin/env python3
"""Summarize compile/correctness/fast rates from KernelGym eval artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
import random
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
_ARTIFACT_SUFFIXES = (".pt", ".pth", ".json", ".jsonl", ".pkl", ".pickle")


def _collect_artifact_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in ("eval_*.pt", "eval_*.pth", "eval_*.json", "eval_*.jsonl", "*.pt", "*.pth"):
        for item in sorted(root.glob(pattern)):
            resolved = item.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
    return candidates


def _pick_preferred_artifact(candidates: list[Path]) -> Path:
    if not candidates:
        raise FileNotFoundError("no eval artifact candidates")
    eval_candidates = [item for item in candidates if item.name.startswith("eval_")]
    pool = eval_candidates or candidates
    return sorted(pool, key=lambda item: item.name)[-1]


def resolve_eval_artifact(path: str | Path) -> Path:
    """Resolve a concrete artifact file from a path or run directory."""

    path = Path(path)
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"eval artifact not found: {path}")

    if path.name == "rollout_data":
        return _pick_preferred_artifact(_collect_artifact_candidates(path))

    rollout_data = path / "dumps" / "rollout_data"
    if rollout_data.is_dir():
        candidates = _collect_artifact_candidates(rollout_data)
        if candidates:
            return _pick_preferred_artifact(candidates)

    candidates = _collect_artifact_candidates(path)
    if not candidates:
        seen: set[Path] = set()
        for suffix in _ARTIFACT_SUFFIXES:
            for item in sorted(path.rglob(f"*{suffix}")):
                resolved = item.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)

    if not candidates:
        raise FileNotFoundError(f"no eval artifact ({', '.join(_ARTIFACT_SUFFIXES)}) under {path}")

    return _pick_preferred_artifact(candidates)


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
    """Compute compile/correctness/fast rates from KernelGym responses."""

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


def get_record_index(record: Any, fallback: int) -> int:
    for key in ("index", "sample_index"):
        value = _get_value(record, key)
        if value is not None:
            return int(value)
    return fallback


def _record_review_bucket(record: Any) -> str:
    metadata = _get_value(record, "metadata")
    if isinstance(metadata, Mapping):
        if metadata.get("extract_error"):
            return "extract_error"
        kernelgym = metadata.get("kernelgym")
        if isinstance(kernelgym, Mapping) and kernelgym.get("extract_error"):
            return "extract_error"

    response = _extract_kernelgym_response(record)
    if response is None:
        return "missing_response"

    compiled = _as_bool(_get_first(response, ("compiled", "compilation")))
    raw_correct = _as_bool(_get_value(response, "correctness"))
    decoy = _as_bool(_get_first(response, ("decoy_kernel", "is_decoy_kernel")))
    if compiled and raw_correct and not decoy:
        return "success"
    if compiled:
        return "compiled_not_correct"
    return "compile_failed"


def select_review_indices(
    records: list[Any],
    *,
    max_samples: int,
    indices: list[int] | None = None,
    seed: int = 0,
) -> list[int]:
    if indices is not None:
        return [idx for idx in indices if 0 <= idx < len(records)]

    if max_samples <= 0 or not records:
        return []

    if len(records) <= max_samples:
        return list(range(len(records)))

    buckets: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        buckets.setdefault(_record_review_bucket(record), []).append(idx)

    bucket_order = (
        "success",
        "compiled_not_correct",
        "compile_failed",
        "extract_error",
        "missing_response",
    )
    ordered_buckets = [name for name in bucket_order if name in buckets]
    ordered_buckets.extend(name for name in buckets if name not in ordered_buckets)

    rng = random.Random(seed)
    chosen: list[int] = []
    per_bucket = max(1, max_samples // len(ordered_buckets))
    for bucket in ordered_buckets:
        pool = buckets[bucket][:]
        rng.shuffle(pool)
        for idx in pool[:per_bucket]:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= max_samples:
                return chosen[:max_samples]

    remaining = [idx for idx in range(len(records)) if idx not in chosen]
    rng.shuffle(remaining)
    for idx in remaining:
        chosen.append(idx)
        if len(chosen) >= max_samples:
            break
    return chosen[:max_samples]


def _truncate_text(text: str, *, head: int, tail: int) -> str:
    if not text:
        return "(empty)\n"
    if len(text) <= head + tail + 80:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n... [{omitted} chars omitted] ...\n\n{text[-tail:]}"


def format_review_sample_text(
    record: Any,
    *,
    prompt_head: int = 3000,
    prompt_tail: int = 1500,
    response_head: int = 5000,
    response_tail: int = 3000,
    kernel_code_head: int = 4000,
    kernel_code_tail: int = 2000,
) -> str:
    metadata = _get_value(record, "metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    kg = metadata.get("kernelgym") if isinstance(metadata, Mapping) else None
    if not isinstance(kg, Mapping):
        kg = _get_value(record, "kernelgym") or {}
    if not isinstance(kg, Mapping):
        kg = {}

    response = _extract_kernelgym_response(record) or {}
    request = kg.get("request") if isinstance(kg.get("request"), Mapping) else {}

    lines = [
        f"index={get_record_index(record, -1)} "
        f"problem_id={metadata.get('problem_id')} name={metadata.get('name')}",
        f"status={_get_value(record, 'status')} "
        f"response_length={_get_value(record, 'response_length')} "
        f"reward={_get_value(record, 'reward')}",
        f"benchmark={metadata.get('benchmark')} level={metadata.get('level')} split={metadata.get('split')}",
        "",
        "=== KERNELGYM ===",
    ]

    extract_error = kg.get("extract_error") or metadata.get("extract_error")
    if extract_error:
        lines.append(f"extract_error: {extract_error}")
    lines.append(f"kernelgym.reward: {kg.get('reward')}")

    if response:
        lines.extend(
            [
                f"status={response.get('status')} compiled={response.get('compiled')} "
                f"correctness={response.get('correctness')} decoy={response.get('decoy_kernel')}",
                f"error_code={response.get('error_code')}",
                f"speedup={response.get('speedup')}",
                f"error_message={response.get('error_message')}",
            ]
        )
        resp_meta = response.get("metadata")
        if isinstance(resp_meta, Mapping):
            for key in ("coverage_backend", "resolved_backend", "backend"):
                if resp_meta.get(key) is not None:
                    lines.append(f"metadata.{key}={resp_meta.get(key)}")

    if metadata.get("kernel_submission"):
        lines.append(f"kernel_submission: {metadata.get('kernel_submission')}")
    if request.get("backend") is not None:
        lines.append(f"request.backend: {request.get('backend')}")

    prompt = _get_value(record, "prompt") or ""
    model_response = _get_value(record, "response") or ""
    lines.extend(
        [
            "",
            "=== PROMPT (truncated) ===",
            _truncate_text(str(prompt), head=prompt_head, tail=prompt_tail),
            "",
            "=== MODEL RESPONSE (truncated) ===",
            _truncate_text(str(model_response), head=response_head, tail=response_tail),
        ]
    )

    kernel_code = request.get("kernel_code")
    if kernel_code:
        lines.extend(
            [
                "",
                "=== EXTRACTED kernel_code sent to KernelGym (truncated) ===",
                _truncate_text(str(kernel_code), head=kernel_code_head, tail=kernel_code_tail),
            ]
        )

    return "\n".join(lines) + "\n"


def export_review_samples(
    records: list[Any],
    review_dir: str | Path,
    *,
    max_samples: int = 8,
    indices: list[int] | None = None,
    seed: int = 0,
    summary: Mapping[str, Any] | None = None,
    source_path: str | Path | None = None,
) -> list[Path]:
    """Write stratified review txt files for manual inspection."""

    out_dir = Path(review_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = select_review_indices(records, max_samples=max_samples, indices=indices, seed=seed)
    written: list[Path] = []
    for idx in chosen:
        record = records[idx]
        sample_idx = get_record_index(record, idx)
        path = out_dir / f"sample_{sample_idx:03d}.txt"
        path.write_text(format_review_sample_text(record), encoding="utf-8")
        written.append(path)

    summary_lines = []
    if source_path is not None:
        summary_lines.append(f"source: {source_path}")
    if summary is not None:
        summary_lines.append("metrics:")
        summary_lines.append(json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True))
    summary_lines.append("")
    summary_lines.append(f"exported {len(written)} / {len(records)} samples:")
    for path in written:
        summary_lines.append(f"  {path}")
    (out_dir / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help=(
            "Eval artifact file (.pt/.json/.jsonl/...) or run directory "
            "(e.g. checkpoints/Qwen3.5-9B/20260522_041800); directories are "
            "resolved via dumps/rollout_data/eval_*.pt"
        ),
    )
    parser.add_argument("--output", help="Optional path to write the JSON summary")
    parser.add_argument(
        "--fast-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_FAST_THRESHOLDS),
        help="Speedup thresholds for fast@ metrics",
    )
    parser.add_argument(
        "--review-samples",
        type=int,
        default=8,
        help="Number of stratified review txt files to write beside the artifact",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip writing sample_XXX.txt review files",
    )
    parser.add_argument(
        "--review-indices",
        help="Comma-separated record indices to export instead of stratified sampling (e.g. 0,5,10)",
    )
    parser.add_argument(
        "--review-seed",
        type=int,
        default=0,
        help="Random seed for stratified review sampling",
    )
    return parser.parse_args()


def _parse_review_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    artifact_path = resolve_eval_artifact(args.input)
    print(f"Using artifact: {artifact_path}", flush=True)

    records = load_records(artifact_path)
    summary = summarize_records(records, fast_thresholds=args.fast_thresholds)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    if not args.no_review:
        review_dir = artifact_path.parent
        written = export_review_samples(
            records,
            review_dir,
            max_samples=args.review_samples,
            indices=_parse_review_indices(args.review_indices),
            seed=args.review_seed,
            summary=summary,
            source_path=artifact_path,
        )
        print(f"Wrote {len(written)} review files to {review_dir}", flush=True)


if __name__ == "__main__":
    main()
