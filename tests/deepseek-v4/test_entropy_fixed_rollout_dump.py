"""CPU-only contracts for non-destructive fixed-behavior rollout preparation."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.dsv4.studies.entropy.prepare_entropy_fixed_rollout import (
    audit_fixed_rollout_dump,
    audit_samples,
    prepare_fixed_rollout_dump,
)

NUM_GPUS = 0


def _sample(*, group_index: int, index: int, response_length: int = 2) -> dict:
    prompt_tokens = [10, 11]
    response_tokens = [1000 + index * 2 + i for i in range(response_length)]
    tokens = prompt_tokens + response_tokens
    rollout_log_probs = np.asarray([-0.25 - 0.01 * i for i in range(response_length)], dtype=np.float32)

    support_ids = np.empty((response_length, 21), dtype=np.int32)
    support_log_probs = np.full((response_length, 21), -7.0, dtype=np.float32)
    support_valid = np.zeros((response_length, 21), dtype=np.bool_)
    for row, sampled_id in enumerate(response_tokens):
        support_ids[row, :20] = np.asarray([sampled_id] + list(range(200, 219)), dtype=np.int32)
        support_ids[row, 20] = 0
        support_log_probs[row, 0] = rollout_log_probs[row]
        support_valid[row, :20] = True

    routed = np.zeros((len(tokens) - 1, 43, 6), dtype=np.int32)
    routed[:, 3:, :] = np.arange(6, dtype=np.int32)
    return {
        "group_index": group_index,
        "index": index,
        "group_id": index,
        "tokens": tokens,
        "response": "x",
        "response_length": response_length,
        "loss_mask": [1] * response_length,
        "rollout_log_probs": rollout_log_probs.tolist(),
        "rollout_topk_token_ids": support_ids,
        "rollout_topk_log_probs": support_log_probs,
        "rollout_topk_valid_mask": support_valid,
        "rollout_routed_experts": routed,
        "metadata": {
            "turn_idx": 0,
            "multi_turn_reward": float(index % 2),
        },
    }


def _four_complete_groups() -> list[dict]:
    return [_sample(group_index=group, index=group * 16 + offset) for group in range(4) for offset in range(16)]


def test_prepare_stamps_derived_dump_without_mutating_source(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "derived" / "rollout_0.genv0.pt"
    samples = _four_complete_groups()
    samples[0]["metadata"]["gen_weight_version"] = 99
    torch.save({"rollout_id": 0, "samples": samples}, source)
    source_bytes = source.read_bytes()

    stats = prepare_fixed_rollout_dump(
        source,
        output,
        load_subsample_ratio=0.5,
        expected_samples=32,
    )

    assert stats["full"]["samples"] == 64
    assert stats["full"]["groups"] == 4
    assert stats["selected"]["samples"] == 32
    assert stats["selected"]["groups"] == 2

    original = torch.load(source, map_location="cpu", weights_only=False)
    assert source.read_bytes() == source_bytes
    assert original["samples"][0]["metadata"]["gen_weight_version"] == 99
    assert "gen_weight_version" not in original["samples"][1]["metadata"]

    derived = torch.load(output, map_location="cpu", weights_only=False)
    assert {sample["metadata"].get("gen_weight_version") for sample in derived["samples"]} == {0}
    assert derived["fixed_behavior_replay"]["source_path"] == str(source.resolve())
    assert derived["fixed_behavior_replay"]["selected_stats"]["groups"] == 2


def test_prepare_refuses_in_place_or_overwrite(tmp_path):
    source = tmp_path / "source.pt"
    torch.save({"samples": _four_complete_groups()}, source)

    with pytest.raises(ValueError, match="original dump is immutable"):
        prepare_fixed_rollout_dump(source, source)

    output = tmp_path / "already-there.pt"
    output.write_bytes(b"do not overwrite")
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_fixed_rollout_dump(source, output)
    assert output.read_bytes() == before


def test_audit_only_validates_selected_groups_without_writing(tmp_path):
    source = tmp_path / "source.pt"
    samples = _four_complete_groups()
    for sample in samples:
        sample["metadata"]["gen_weight_version"] = 0
    torch.save({"rollout_id": 0, "samples": samples}, source)
    before = source.read_bytes()

    stats = audit_fixed_rollout_dump(
        source,
        load_subsample_ratio=0.5,
        expected_samples=32,
        expected_gen_weight_version=0,
        expected_rollout_id=0,
    )

    assert stats["full"]["groups"] == 4
    assert stats["selected"]["groups"] == 2
    assert source.read_bytes() == before
    assert list(tmp_path.iterdir()) == [source]


def test_audit_rejects_incomplete_prompt_group():
    samples = _four_complete_groups()[:-1]
    for sample in samples:
        sample["metadata"]["gen_weight_version"] = 0
    with pytest.raises(ValueError, match="prompt groups are incomplete"):
        audit_samples(samples, expected_gen_weight_version=0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda sample: sample["metadata"].__setitem__("turn_idx", 1), "turn_idx"),
        (lambda sample: sample.__setitem__("rollout_log_probs", [-0.1]), "rollout_log_probs length"),
        (
            lambda sample: sample.__setitem__("rollout_routed_experts", np.zeros((1, 43, 6), dtype=np.int32)),
            "rollout_routed_experts shape",
        ),
        (
            lambda sample: sample.__setitem__("rollout_topk_token_ids", np.zeros((2, 20), dtype=np.int32)),
            "rollout_topk_token_ids shape",
        ),
    ],
)
def test_audit_rejects_turn_logprob_routing_and_topk_corruption(mutation, message):
    samples = _four_complete_groups()
    for sample in samples:
        sample["metadata"]["gen_weight_version"] = 0
    bad = deepcopy(samples)
    mutation(bad[0])
    with pytest.raises(ValueError, match=message):
        audit_samples(bad, expected_gen_weight_version=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
