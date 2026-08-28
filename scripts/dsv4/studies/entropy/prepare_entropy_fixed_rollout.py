#!/usr/bin/env python3
"""Prepare a provenance-stamped frozen rollout for entropy A/B replay.

The source dump is never modified.  A derived dump is written to a distinct,
previously-nonexistent path after CPU-only structural checks.  The checks mirror
the fields consumed by TRLOO, predictive-mask diagnostics, and rollout routing
replay in the fixed-batch launcher.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def select_debug_subsample(samples: list[dict[str, Any]], ratio: float | None) -> list[dict[str, Any]]:
    """Mirror ``RolloutManager._get_rollout_data``'s first/last subsample."""
    if ratio is None:
        return list(samples)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"load subsample ratio must be in (0, 1], got {ratio}")
    rough_num_rows = int(len(samples) * ratio)
    if rough_num_rows <= 0:
        raise ValueError(f"load subsample ratio {ratio} selects no rows from a {len(samples)}-sample dump")
    # Keep this spelling aligned with rollout.py, including odd-size floor
    # behavior for the negative tail index.
    return samples[: rough_num_rows // 2] + samples[(-rough_num_rows) // 2 :]


def _require_sequence(value: Any, *, field: str, sample_index: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"sample[{sample_index}] {field} must be a sequence")
    return value


def _audit_predictive_support(
    sample: dict[str, Any],
    *,
    sample_position: int,
    response_tokens: Sequence[int],
    rollout_log_probs: np.ndarray,
    predictive_top_k: int,
) -> None:
    width = predictive_top_k + 1
    ids = np.asarray(sample.get("rollout_topk_token_ids"))
    log_probs = np.asarray(sample.get("rollout_topk_log_probs"))
    valid = np.asarray(sample.get("rollout_topk_valid_mask"))
    expected_shape = (len(response_tokens), width)
    for field, array in (
        ("rollout_topk_token_ids", ids),
        ("rollout_topk_log_probs", log_probs),
        ("rollout_topk_valid_mask", valid),
    ):
        if array.shape != expected_shape:
            raise ValueError(f"sample[{sample_position}] {field} shape {array.shape} != {expected_shape}")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"sample[{sample_position}] predictive token ids must be integer, got {ids.dtype}")
    if not np.issubdtype(log_probs.dtype, np.floating):
        raise ValueError(f"sample[{sample_position}] predictive log probs must be floating, got {log_probs.dtype}")
    if not np.issubdtype(valid.dtype, np.bool_):
        raise ValueError(f"sample[{sample_position}] predictive valid mask must be bool, got {valid.dtype}")

    valid_counts = valid.sum(axis=1)
    if np.any((valid_counts < predictive_top_k) | (valid_counts > width)):
        raise ValueError(
            f"sample[{sample_position}] predictive support must contain top-{predictive_top_k} "
            "plus the sampled token when needed"
        )
    if not np.isfinite(log_probs[valid]).all():
        raise ValueError(f"sample[{sample_position}] predictive valid log probs contain NaN/Inf")

    for token_offset, (sampled_id, sampled_log_prob) in enumerate(
        zip(response_tokens, rollout_log_probs, strict=True)
    ):
        matches = valid[token_offset] & (ids[token_offset] == int(sampled_id))
        if int(matches.sum()) != 1:
            raise ValueError(
                f"sample[{sample_position}] predictive row {token_offset} must contain sampled token "
                f"{sampled_id} exactly once"
            )
        support_log_prob = float(log_probs[token_offset][matches][0])
        if not math.isclose(support_log_prob, float(sampled_log_prob), rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"sample[{sample_position}] predictive sampled-token logprob mismatch at response token "
                f"{token_offset}: support={support_log_prob}, rollout={float(sampled_log_prob)}"
            )


def audit_samples(
    samples: list[dict[str, Any]],
    *,
    expected_group_size: int = 16,
    expected_turn: int = 0,
    expected_layers: int = 43,
    expected_routing_topk: int = 6,
    expected_predictive_top_k: int = 20,
    expected_num_experts: int = 256,
    expected_gen_weight_version: int | None = None,
    expected_samples: int | None = None,
) -> dict[str, int]:
    if not samples:
        raise ValueError("rollout dump contains no samples")
    if expected_samples is not None and len(samples) != expected_samples:
        raise ValueError(f"selected sample count {len(samples)} != expected {expected_samples}")
    if expected_group_size <= 0:
        raise ValueError(f"expected_group_size must be positive, got {expected_group_size}")

    group_counts: Counter[Any] = Counter()
    trajectory_turns: set[tuple[int, int]] = set()
    seen_indices: set[int] = set()
    response_tokens_total = 0
    routed_rows_total = 0

    for position, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample[{position}] must be a dict, got {type(sample).__name__}")
        group_index = sample.get("group_index")
        if group_index is None:
            raise ValueError(f"sample[{position}] is missing group_index")
        group_counts[group_index] += 1

        sample_index = sample.get("index")
        if sample_index is None:
            raise ValueError(f"sample[{position}] is missing trajectory index")
        sample_index = int(sample_index)
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"sample[{position}] metadata must be a dict")
        if "turn_idx" not in metadata or int(metadata["turn_idx"]) != expected_turn:
            raise ValueError(f"sample[{position}] turn_idx={metadata.get('turn_idx')!r}, expected {expected_turn}")
        trajectory_key = (sample_index, expected_turn)
        if trajectory_key in trajectory_turns:
            raise ValueError(f"duplicate trajectory/turn pair {trajectory_key}")
        trajectory_turns.add(trajectory_key)
        seen_indices.add(sample_index)
        if "multi_turn_reward" not in metadata:
            raise ValueError(f"sample[{position}] missing metadata.multi_turn_reward required by TRLOO")
        if expected_gen_weight_version is not None:
            actual_version = metadata.get("gen_weight_version")
            if actual_version is None or int(actual_version) != expected_gen_weight_version:
                raise ValueError(
                    f"sample[{position}] gen_weight_version={actual_version!r}, "
                    f"expected {expected_gen_weight_version}"
                )

        tokens = _require_sequence(sample.get("tokens"), field="tokens", sample_index=position)
        response_length = sample.get("response_length")
        if not isinstance(response_length, int) or response_length <= 0:
            raise ValueError(f"sample[{position}] response_length must be positive, got {response_length!r}")
        if response_length >= len(tokens):
            raise ValueError(
                f"sample[{position}] response_length={response_length} must be smaller than len(tokens)={len(tokens)}"
            )
        response_tokens = tokens[-response_length:]
        response_tokens_total += response_length

        loss_mask = _require_sequence(sample.get("loss_mask"), field="loss_mask", sample_index=position)
        if len(loss_mask) != response_length:
            raise ValueError(
                f"sample[{position}] loss_mask length {len(loss_mask)} != response_length {response_length}"
            )
        rollout_log_probs_seq = _require_sequence(
            sample.get("rollout_log_probs"), field="rollout_log_probs", sample_index=position
        )
        if len(rollout_log_probs_seq) != response_length:
            raise ValueError(
                f"sample[{position}] rollout_log_probs length {len(rollout_log_probs_seq)} "
                f"!= response_length {response_length}"
            )
        rollout_log_probs = np.asarray(rollout_log_probs_seq, dtype=np.float64)
        if not np.isfinite(rollout_log_probs).all():
            raise ValueError(f"sample[{position}] rollout_log_probs contain NaN/Inf")

        _audit_predictive_support(
            sample,
            sample_position=position,
            response_tokens=response_tokens,
            rollout_log_probs=rollout_log_probs,
            predictive_top_k=expected_predictive_top_k,
        )

        routed = np.asarray(sample.get("rollout_routed_experts"))
        expected_routing_shape = (len(tokens) - 1, expected_layers, expected_routing_topk)
        if routed.shape != expected_routing_shape:
            raise ValueError(
                f"sample[{position}] rollout_routed_experts shape {routed.shape} " f"!= {expected_routing_shape}"
            )
        if not np.issubdtype(routed.dtype, np.integer):
            raise ValueError(f"sample[{position}] routed experts must be integer, got {routed.dtype}")
        if routed.size and (int(routed.min()) < 0 or int(routed.max()) >= expected_num_experts):
            raise ValueError(f"sample[{position}] routed expert ids must be in [0, {expected_num_experts})")
        routed_rows_total += routed.shape[0]

    incomplete = {key: count for key, count in group_counts.items() if count != expected_group_size}
    if incomplete:
        preview = list(incomplete.items())[:8]
        raise ValueError(f"prompt groups are incomplete: expected {expected_group_size} samples each, got {preview}")

    return {
        "samples": len(samples),
        "groups": len(group_counts),
        "trajectories": len(seen_indices),
        "response_tokens": response_tokens_total,
        "routing_rows": routed_rows_total,
    }


def _load_dump(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"rollout dump must contain a dict, got {type(data).__name__}")
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("rollout dump must contain a samples list")
    return data


def _save_new_path_atomically(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing derived dump: {output}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(data, tmp_path)
        # Hard-link publication is atomic and, unlike os.replace(), cannot
        # overwrite an output that appeared after the exists() check.
        os.link(tmp_path, output)
    finally:
        tmp_path.unlink(missing_ok=True)


def audit_fixed_rollout_dump(
    source: str | Path,
    *,
    load_subsample_ratio: float | None = None,
    expected_samples: int | None = None,
    expected_group_size: int = 16,
    expected_turn: int = 0,
    expected_layers: int = 43,
    expected_routing_topk: int = 6,
    expected_predictive_top_k: int = 20,
    expected_num_experts: int = 256,
    expected_gen_weight_version: int | None = None,
    expected_rollout_id: int | None = None,
) -> dict[str, dict[str, int]]:
    """Read-only structural audit of both the full and selected replay batch."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source rollout dump does not exist: {source_path}")
    data = _load_dump(source_path)
    if expected_rollout_id is not None:
        actual_rollout_id = data.get("rollout_id")
        if actual_rollout_id is None or int(actual_rollout_id) != expected_rollout_id:
            raise ValueError(f"rollout dump rollout_id={actual_rollout_id!r}, expected {expected_rollout_id}")
    samples = data["samples"]
    audit_kwargs = dict(
        expected_group_size=expected_group_size,
        expected_turn=expected_turn,
        expected_layers=expected_layers,
        expected_routing_topk=expected_routing_topk,
        expected_predictive_top_k=expected_predictive_top_k,
        expected_num_experts=expected_num_experts,
        expected_gen_weight_version=expected_gen_weight_version,
    )
    full_stats = audit_samples(samples, **audit_kwargs)
    selected_samples = select_debug_subsample(samples, load_subsample_ratio)
    selected_stats = audit_samples(selected_samples, expected_samples=expected_samples, **audit_kwargs)
    return {"full": full_stats, "selected": selected_stats}


def prepare_fixed_rollout_dump(
    source: str | Path,
    output: str | Path,
    *,
    gen_weight_version: int = 0,
    load_subsample_ratio: float | None = None,
    expected_samples: int | None = None,
    expected_group_size: int = 16,
    expected_turn: int = 0,
    expected_layers: int = 43,
    expected_routing_topk: int = 6,
    expected_predictive_top_k: int = 20,
    expected_num_experts: int = 256,
) -> dict[str, dict[str, int]]:
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("source and output must be different; the original dump is immutable")
    if not source_path.is_file():
        raise FileNotFoundError(f"source rollout dump does not exist: {source_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing derived dump: {output_path}")

    source_data = _load_dump(source_path)
    prepared_samples = []
    for position, sample in enumerate(source_data["samples"]):
        if not isinstance(sample, dict):
            raise ValueError(f"sample[{position}] must be a dict, got {type(sample).__name__}")
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"sample[{position}] metadata must be a dict")
        prepared = dict(sample)
        prepared_metadata = dict(metadata)
        prepared_metadata["gen_weight_version"] = int(gen_weight_version)
        prepared["metadata"] = prepared_metadata
        prepared_samples.append(prepared)

    audit_kwargs = dict(
        expected_group_size=expected_group_size,
        expected_turn=expected_turn,
        expected_layers=expected_layers,
        expected_routing_topk=expected_routing_topk,
        expected_predictive_top_k=expected_predictive_top_k,
        expected_num_experts=expected_num_experts,
        expected_gen_weight_version=gen_weight_version,
    )
    full_stats = audit_samples(prepared_samples, **audit_kwargs)
    selected_samples = select_debug_subsample(prepared_samples, load_subsample_ratio)
    selected_stats = audit_samples(selected_samples, expected_samples=expected_samples, **audit_kwargs)

    source_stat = source_path.stat()
    prepared_data = dict(source_data)
    prepared_data["samples"] = prepared_samples
    prepared_data["fixed_behavior_replay"] = {
        "source_path": str(source_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "gen_weight_version": int(gen_weight_version),
        "load_subsample_ratio": load_subsample_ratio,
        "full_stats": full_stats,
        "selected_stats": selected_stats,
    }
    _save_new_path_atomically(prepared_data, output_path)
    return {"full": full_stats, "selected": selected_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and audit a non-destructive v0-stamped frozen rollout dump.")
    parser.add_argument("source", help="Immutable source rollout .pt")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", help="New derived .pt; must not already exist")
    mode.add_argument("--audit-only", action="store_true", help="Read and validate without writing any file")
    parser.add_argument("--gen-weight-version", type=int, default=0)
    parser.add_argument(
        "--expected-gen-weight-version",
        type=int,
        default=None,
        help="Audit-only: require every selected/full sample to carry this generation version",
    )
    parser.add_argument("--expected-rollout-id", type=int, default=None)
    parser.add_argument("--load-subsample-ratio", type=float, default=None)
    parser.add_argument("--expected-samples", type=int, default=None, help="Expected rows after subsampling")
    parser.add_argument("--expected-group-size", type=int, default=16)
    parser.add_argument("--expected-turn", type=int, default=0)
    parser.add_argument("--expected-layers", type=int, default=43)
    parser.add_argument("--expected-routing-topk", type=int, default=6)
    parser.add_argument("--expected-predictive-top-k", type=int, default=20)
    parser.add_argument("--expected-num-experts", type=int, default=256)
    args = parser.parse_args()

    common_kwargs = dict(
        load_subsample_ratio=args.load_subsample_ratio,
        expected_samples=args.expected_samples,
        expected_group_size=args.expected_group_size,
        expected_turn=args.expected_turn,
        expected_layers=args.expected_layers,
        expected_routing_topk=args.expected_routing_topk,
        expected_predictive_top_k=args.expected_predictive_top_k,
        expected_num_experts=args.expected_num_experts,
    )
    if args.audit_only:
        stats = audit_fixed_rollout_dump(
            args.source,
            expected_gen_weight_version=args.expected_gen_weight_version,
            expected_rollout_id=args.expected_rollout_id,
            **common_kwargs,
        )
        print(
            "FIXED_REPLAY_AUDIT_PASS "
            f"source={Path(args.source).resolve()} "
            f"expected_gen_weight_version={args.expected_gen_weight_version} "
            f"full={stats['full']} selected={stats['selected']}"
        )
    else:
        assert args.output is not None
        stats = prepare_fixed_rollout_dump(
            args.source,
            args.output,
            gen_weight_version=args.gen_weight_version,
            **common_kwargs,
        )
        print(
            "FIXED_REPLAY_DUMP_PASS "
            f"source={Path(args.source).resolve()} output={Path(args.output).resolve()} "
            f"gen_weight_version={args.gen_weight_version} full={stats['full']} selected={stats['selected']}"
        )


if __name__ == "__main__":
    main()
