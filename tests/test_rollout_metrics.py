import base64
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

repo_root = Path(__file__).resolve().parents[1]
repo_root_path = str(repo_root)
if repo_root_path not in sys.path:
    sys.path.insert(0, repo_root_path)

from slime.observability.rollout_metrics import (
    _compute_exp_rollout_metrics,
    _compute_top_p_kept_vocab_metrics,
    _compute_verify_rl_metrics,
    _iter_response_diversity_groups,
    compute_metrics_from_samples,
)
from slime.utils.misc import decode_int32_meta_array
from slime.utils.types import Sample

NUM_GPUS = 0


def _make_args():
    return Namespace(sglang_speculative_algorithm=False, num_layers=2, moe_router_topk=2)


@pytest.mark.unit
def test_verify_metrics_separate_roles_and_exclude_padding():
    samples = [
        Sample(group_index=1, reward=1.0, response_length=100, metadata={"turn_idx": 0}),
        Sample(
            group_index=2,
            reward=3.0,
            response_length=20,
            loss_mask=torch.tensor([1, 0] * 10),
            metadata={
                "role": "verify",
                "turn_idx": 0,
                "verify_extracted_response": "fix",
                "verify_kernel_reward": 3.0,
            },
        ),
        Sample(
            group_index=2,
            reward=3.0,
            response_length=60,
            metadata={"role": "kernel", "verify_trajectory": True, "env_extra_info": {"correctness": True}},
        ),
        Sample(group_index=3, reward=999.0, response_length=999, metadata={"is_pad_turn": True}),
        Sample(group_index=4, reward=999.0, response_length=999, metadata={"role": "pad"}),
        Sample(group_index=2, reward=999.0, metadata={"verify_scoring_branch": "anchor"}),
    ]
    metrics = _compute_verify_rl_metrics(Namespace(), samples)

    assert metrics["verify/group_fraction"] == 0.5
    assert metrics["verify/group_count"] == 1
    assert metrics["verify/total_group_count"] == 2
    for prefix, reward, length in [("kernel/ordinary", 1.0, 100), ("verify", 3.0, 20), ("verify/kernel", 3.0, 60)]:
        assert metrics[f"{prefix}/sample_count"] == 1
        assert metrics[f"{prefix}/reward/mean"] == reward
        assert metrics[f"{prefix}/response_len/mean"] == length
    assert metrics["verify/trainable_response_len/mean"] == 10
    assert metrics["verify/format_valid_fraction"] == 1
    assert metrics["verify/kernel/correctness_rate"] == 1
    assert metrics["verify/scored_fraction"] == 1
    assert "verify/improvement/mean" not in metrics


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode,baseline_key",
    [("history", "history_baseline"), ("anchor", "verify_anchor_reward")],
)
def test_verify_metrics_compute_pair_improvement_without_mutation(mode, baseline_key):
    samples = [
        Sample(
            group_index=5,
            reward=reward if mode == "history" else reward - 2.0,
            response_length=10,
            metadata={
                "role": "verify",
                "turn_idx": 2 * turn,
                "verify_source_reward": -100.0,
                "verify_kernel_reward": reward,
                "verify_anchor_key": "shared" if mode == "anchor" else None,
                baseline_key: 2.0,
            },
        )
        for turn, reward in enumerate([1.0, 2.0, 6.0])
    ]
    original = deepcopy([s.to_dict() for s in samples])
    metrics = _compute_verify_rl_metrics(Namespace(verify_advantage_baseline=mode), samples)

    assert metrics["verify/reward/mean"] == (3.0 if mode == "history" else 1.0)
    assert metrics["verify/kernel_reward/mean"] == 3.0
    assert metrics["verify/baseline_reward/mean"] == 2.0
    assert metrics["verify/improvement/mean"] == 1.0
    assert metrics["verify/improvement/count"] == 3
    assert metrics["verify/improvement/min"] == -1.0
    for outcome in ("win", "tie", "loss"):
        assert metrics[f"verify/improvement/{outcome}_rate"] == pytest.approx(1 / 3)
    assert metrics["verify/anchor/reward/count"] == (1 if mode == "anchor" else 0)
    if mode == "anchor":
        assert metrics["verify/anchor/reward/mean"] == 2.0
    assert [s.to_dict() for s in samples] == original


@pytest.mark.unit
def test_verify_history_metrics_preserve_zero_baseline():
    sample = Sample(
        reward=0.7,
        metadata={
            "role": "verify",
            "verify_kernel_reward": 0.7,
            "history_baseline": 0.0,
            "verify_source_reward": 99.0,
        },
    )
    original = deepcopy(sample.metadata)
    metrics = _compute_verify_rl_metrics(Namespace(verify_advantage_baseline="history"), [sample])
    assert metrics["verify/baseline_reward/mean"] == 0.0
    assert metrics["verify/improvement/mean"] == pytest.approx(0.7)
    assert metrics["verify/missing_baseline_count"] == 0
    assert sample.metadata == original


@pytest.mark.unit
def test_verify_metrics_masked_outputs_count_tokens_but_not_rewards():
    samples = [
        Sample(
            reward=0.0,
            response_length=length,
            remove_sample=True,
            metadata={"role": "verify", "remove_reason": reason, "verify_kernel_reward": 99.0},
        )
        for length, reason in enumerate(
            ["invalid_verify_format", "verify_diagnosis_incomplete", "verify_scoring_version_mismatch"], start=1
        )
    ]
    samples[-1].metadata["verify_extracted_response"] = "valid format but wrong version"
    samples.append(Sample(reward=99.0, response_length=4, status=Sample.Status.ABORTED, metadata={"role": "verify"}))
    metrics = _compute_verify_rl_metrics(Namespace(), samples)

    assert metrics["verify/sample_count"] == 4
    assert metrics["verify/valid_sample_count"] == 0
    assert metrics["verify/reward/count"] == 0
    assert metrics["verify/kernel_reward/count"] == 0
    assert metrics["verify/response_len/mean"] == 2.5
    assert metrics["verify/format_valid_fraction"] == 0.25
    assert metrics["verify/removed_fraction"] == 0.75
    assert metrics["verify/aborted_fraction"] == 0.25
    for reason in ("invalid_verify_format", "verify_diagnosis_incomplete", "verify_scoring_version_mismatch"):
        assert metrics[f"verify/{reason}/fraction"] == 0.25
    assert "verify/reward/mean" not in metrics
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.unit
def test_verify_metrics_missing_nonfinite_values_and_correctness_coverage():
    samples = [
        Sample(reward=float("nan"), metadata={"role": "verify", "verify_kernel_reward": float("inf")}),
        Sample(reward=2.0, metadata={"role": "verify", "verify_kernel_reward": 2.0}),
        Sample(metadata={"verify_trajectory": True, "env_extra_info": {"correctness": None}}),
        Sample(metadata={"verify_trajectory": True, "env_result": {"env_extra_info": {"correctness": False}}}),
        Sample(metadata={"verify_trajectory": True, "env_extra_info": {"correctness": True, "decoy_kernel": True}}),
    ]
    metrics = _compute_verify_rl_metrics(Namespace(verify_advantage_baseline="history"), samples)

    assert metrics["verify/reward/count"] == 1
    assert metrics["verify/kernel_reward/count"] == 1
    assert metrics["verify/missing_baseline_count"] == 1
    assert metrics["verify/improvement/count"] == 0
    assert "verify/improvement/mean" not in metrics
    assert metrics["verify/kernel/correctness_count"] == 2
    assert metrics["verify/kernel/correctness_rate"] == 0
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.unit
def test_verify_metrics_disabled_and_empty_buffer_fallback():
    samples = [Sample(group_index=1, reward=1.0)]
    assert _compute_verify_rl_metrics(Namespace(), samples) == {}
    for batch in (samples, []):
        metrics = _compute_verify_rl_metrics(Namespace(verify_rollout_ratio=0.5), batch)
        assert metrics["verify/group_fraction"] == 0
        assert metrics["verify/sample_count"] == 0
        assert metrics["verify/response_len/count"] == 0
        assert "verify/response_len/mean" not in metrics


@pytest.mark.unit
def test_verify_metrics_integrate_with_regular_rollout_metrics_without_exp_flag():
    args = Namespace(advantage_estimator="ppo", log_reward_category=None, log_exp_metrics=False)
    samples = [Sample(reward=2.0, response_length=5, metadata={"role": "verify"})]

    metrics = compute_metrics_from_samples(args, samples)

    assert metrics["verify/reward/mean"] == 2.0
    assert metrics["verify/response_len/mean"] == 5


@pytest.mark.unit
def test_top_p_kept_vocab_metric_uses_loss_mask():
    samples = [
        Sample(
            response_length=4,
            loss_mask=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
            rollout_top_p_token_offsets=torch.tensor([0, 3, 8, 10, 20], dtype=torch.int32),
        ),
        Sample(
            response_length=2,
            loss_mask=None,
            rollout_top_p_token_offsets=torch.tensor([0, 4, 9], dtype=torch.int32),
        ),
    ]

    metrics = _compute_top_p_kept_vocab_metrics(samples)

    assert metrics["top_p_kept_vocab_per_token"] == pytest.approx(3.5)


@pytest.mark.unit
def test_top_p_kept_vocab_metric_skips_removed_samples():
    samples = [
        Sample(
            response_length=3,
            loss_mask=[1, 1, 1],
            remove_sample=True,
            rollout_top_p_token_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        )
    ]

    assert _compute_top_p_kept_vocab_metrics(samples) == {}


@pytest.mark.unit
def test_response_diversity_groups_multi_turn_samples_by_prompt_and_turn():
    samples = [
        Sample(group_index=7, rollout_id=10, metadata={"turn_idx": 0}),
        Sample(group_index=7, rollout_id=11, metadata={"turn_idx": 0}),
        Sample(group_index=7, rollout_id=10, metadata={"turn_idx": 1}),
        Sample(group_index=7, rollout_id=11, metadata={"turn_idx": 1}),
    ]

    groups = list(_iter_response_diversity_groups(Namespace(n_samples_per_prompt=2), samples))

    assert [[(sample.rollout_id, sample.metadata["turn_idx"]) for sample in group] for group in groups] == [
        [(10, 0), (11, 0)],
        [(10, 1), (11, 1)],
    ]


@pytest.mark.unit
def test_exp_rollout_metrics_cover_reward_groups_turns_and_async_state():
    samples = [
        Sample(
            group_index=7,
            reward=0.5,
            response_length=10,
            metadata={
                "turn_idx": 0,
                "task_reward": 0.5,
                "reward_component": {"correctness": 0.5},
                "gen_weight_version": 3,
                "gen_submit_time": 1.0,
                "engine_weight_version_span": False,
                "engine_weight_version_mismatch": False,
                "env_extra_info": {"compilation": True, "correctness": True, "speedup": 1.2},
            },
        ),
        Sample(
            group_index=7,
            reward=-0.25,
            response_length=20,
            remove_sample=True,
            metadata={
                "turn_idx": 0,
                "task_reward": -0.25,
                "reward_component": {"failed": -0.25},
                "gen_weight_version": 2,
                "gen_submit_time": 2.0,
                "engine_weight_version_span": True,
                "engine_weight_version_mismatch": True,
                "env_extra_info": {"compilation": True, "correctness": False},
            },
        ),
    ]

    metrics = _compute_exp_rollout_metrics(Namespace(log_exp_metrics=True), samples)

    assert metrics["exp/rollout/reward/final/mean"] == pytest.approx(0.125)
    assert metrics["exp/rollout/group/reward_range/mean"] == pytest.approx(0.75)
    assert metrics["exp/rollout/group/all_equal_fraction"] == 0.0
    assert metrics["exp/rollout/turn/0/response_length/mean"] == pytest.approx(15.0)
    assert metrics["exp/rollout/sample/removed_fraction"] == pytest.approx(0.5)
    assert metrics["exp/rollout/async/engine_version_span_fraction"] == pytest.approx(0.5)
    assert metrics["exp/rollout/async/engine_version_mismatch_fraction"] == pytest.approx(0.5)
    assert metrics["exp/rollout/reward/component/correctness/mean"] == pytest.approx(0.5)
    assert metrics["exp/rollout/reward/component/failed/mean"] == pytest.approx(-0.25)


def _b64_int32(values: list[int]) -> str:
    return base64.b64encode(np.array(values, dtype=np.int32).tobytes()).decode("ascii")


@pytest.mark.unit
def test_decode_int32_meta_array_decodes_base64_to_tensor():
    decoded = decode_int32_meta_array({"routed_experts": _b64_int32([1, 2, 3])}, "routed_experts")

    assert torch.is_tensor(decoded)
    assert decoded.dtype == torch.int32
    torch.testing.assert_close(decoded, torch.tensor([1, 2, 3], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_merges_top_p_tensors():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.tokens == [0, 1, 10, 20]
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_can_skip_terminal_status_for_streaming_chunks():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
        update_terminal_info=False,
    )

    assert sample.status is Sample.Status.PENDING
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_decodes_routed_experts():
    sample = Sample(tokens=[101, 102, 103])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts.shape == (2, 2, 2)
    torch.testing.assert_close(
        sample.rollout_routed_experts,
        torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.int32),
    )


@pytest.mark.unit
def test_append_response_tokens_ignores_split_pd_routed_experts():
    sample = Sample(tokens=[101, 102, 103, 104])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "pd_prefill_routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "pd_decode_routed_experts": _b64_int32([8, 9, 10, 11]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts is None


@pytest.mark.unit
def test_append_response_tokens_rejects_mismatched_routed_experts_shape():
    sample = Sample(tokens=[101, 102, 103])

    with pytest.raises(ValueError, match="routed_experts element count"):
        sample.append_response_tokens(
            _make_args(),
            tokens=[],
            trainable=True,
            meta_info={
                "routed_experts": _b64_int32([0, 1, 2, 3]),
                "finish_reason": {"type": "stop"},
            },
        )


@pytest.mark.unit
def test_append_response_tokens_pads_top_p_for_non_trainable_tokens():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        rollout_top_p_token_ids=torch.tensor([10, 11], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 2], dtype=torch.int32),
    )

    sample.append_response_tokens(tokens=[200, 201, 202], trainable=False)

    assert sample.tokens == [0, 1, 200, 201, 202]
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 0, 0, 0]
    assert sample.rollout_log_probs == [-0.1, 0.0, 0.0, 0.0]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([10, 11], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_requires_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="trainable response tokens require rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], trainable=True)


@pytest.mark.unit
def test_append_response_tokens_rejects_non_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="non-trainable response tokens should not pass rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], log_probs=[-0.1], trainable=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
