from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

repo_root = Path(__file__).resolve().parents[1]
repo_root_path = str(repo_root)
if repo_root_path in sys.path:
    sys.path.remove(repo_root_path)
sys.path.insert(0, repo_root_path)

from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.kernel_reward import _compute_dynamic_auxiliary_gate, reward_post_process_by_group
from slime.ray.rollout import RolloutManager
from slime.utils.types import Sample


def _make_manager(*, advantage_estimator: str, use_multi_turn: bool, grpo_std_normalization: bool = False):
    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.custom_reward_post_process_func = None
    manager.args = SimpleNamespace(
        advantage_estimator=advantage_estimator,
        grpo_std_normalization=grpo_std_normalization,
        max_turns=2,
        n_samples_per_prompt=3,
        reward_key=None,
        rewards_normalization=True,
        rollout_batch_size=2,
        use_multi_turn=use_multi_turn,
        multi_turn_gamma=1.0,
    )
    return manager


def _enable_ctm(args) -> None:
    args.use_conditional_truncation_mask = True
    args.conditional_truncation_mask_prob = 1.0
    args.conditional_truncation_repeat_window = 2
    args.rollout_max_response_len = 4


def _make_sample(index: int, group_index: int, reward: float, turn_idx: int | None = None) -> Sample:
    metadata = {}
    if turn_idx is not None:
        metadata["turn_idx"] = turn_idx
    return Sample(index=index, group_index=group_index, reward=reward, metadata=metadata)


def _set_reward_component(
    sample: Sample,
    *,
    correctness_score: float = 0.0,
    performance_score: float = 0.0,
    coverage_score: float = 0.0,
    failed: float | None = None,
    overlong_penalty: float = 0.0,
) -> None:
    sample.metadata["kernel_score"] = {
        "correctness": correctness_score,
        "performance": performance_score,
        "coverage": coverage_score,
    }
    sample.metadata["reward_component"] = {
        "correctness": 0.5 * correctness_score,
        "performance": 0.5 * performance_score,
        "coverage": 0.5 * coverage_score,
        "failed": failed,
        "overlong_penalty": overlong_penalty,
    }
    sample.metadata["env_extra_info"] = {
        "correctness": correctness_score > 0.0,
        "decoy_kernel": False,
    }


@pytest.mark.parametrize(
    ("num_correct", "group_size", "expected"),
    [
        (0, 16, 0.0),
        (1, 16, 0.0),
        (4, 16, (3.0 / 15.0) ** 0.5),
        (16, 16, 1.0),
        (1, 1, 0.0),
    ],
)
def test_dynamic_auxiliary_gate(num_correct: int, group_size: int, expected: float):
    assert _compute_dynamic_auxiliary_gate(num_correct, group_size) == pytest.approx(expected)


def test_dynamic_reward_weights_keep_half_maxima_and_apply_before_rloo(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", True)
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    samples = [
        _make_sample(0, 0, 0.0),
        _make_sample(1, 0, 0.8),
        _make_sample(2, 0, 1.2),
        _make_sample(3, 0, 0.0),
    ]
    _set_reward_component(samples[0])
    _set_reward_component(samples[1], correctness_score=1.0, performance_score=0.2, coverage_score=0.4)
    _set_reward_component(samples[2], correctness_score=1.0, performance_score=0.6, coverage_score=0.8)
    _set_reward_component(samples[3])

    raw_rewards, advantages = reward_post_process_by_group(manager.args, samples)

    gate = (1.0 / 3.0) ** 0.5
    expected_raw = [0.0, 0.5 + gate * 0.3, 0.5 + gate * 0.7, 0.0]
    mean = sum(expected_raw) / len(expected_raw)
    expected_advantages = [(reward - mean) * 4.0 / 3.0 for reward in expected_raw]
    assert raw_rewards == pytest.approx(expected_raw)
    assert advantages == pytest.approx(expected_advantages)
    assert samples[1].metadata["kernel_score"] == {
        "correctness": 1.0,
        "performance": 0.2,
        "coverage": 0.4,
    }
    assert samples[1].metadata["reward_component"]["performance"] == pytest.approx(gate * 0.1)
    assert samples[1].metadata["reward_component"]["coverage"] == pytest.approx(gate * 0.2)


def test_dynamic_reward_all_correct_matches_fixed_half_weights(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", True)
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    samples = [_make_sample(index, 0, reward) for index, reward in enumerate([0.6, 0.8, 1.0])]
    for sample, performance_reward in zip(samples, [0.1, 0.2, 0.3], strict=True):
        _set_reward_component(
            sample,
            correctness_score=1.0,
            performance_score=2.0 * performance_reward,
            coverage_score=2.0 * (sample.reward - 0.5 - performance_reward),
        )

    raw_rewards, _advantages = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == pytest.approx([0.6, 0.8, 1.0])


def test_dynamic_reward_rebuilds_overlong_penalty_from_components(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", True)
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    correct = _make_sample(0, 0, 99.0)
    incorrect = _make_sample(1, 0, 0.0)
    _set_reward_component(correct, correctness_score=1.0, performance_score=1.0, overlong_penalty=-0.2)
    _set_reward_component(incorrect)
    correct.metadata["task_reward"] = 1.0
    correct.metadata["overlong_penalty"] = 0.2

    raw_rewards, _advantages = reward_post_process_by_group(manager.args, [correct, incorrect])

    # C=1 makes the auxiliary gate zero; the penalty is rebuilt from components.
    assert raw_rewards == pytest.approx([0.3, 0.0])


def test_dynamic_reward_uses_failed_component_as_branch_sentinel(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", True)
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    kernel_failure = _make_sample(0, 0, 99.0)
    output_mismatch = _make_sample(1, 0, 99.0)
    _set_reward_component(
        kernel_failure,
        correctness_score=1.0,
        performance_score=1.0,
        failed=0.0,
        overlong_penalty=-0.1,
    )
    _set_reward_component(
        output_mismatch,
        correctness_score=1.0,
        performance_score=1.0,
        failed=0.25,
        overlong_penalty=-0.1,
    )

    raw_rewards, _advantages = reward_post_process_by_group(manager.args, [kernel_failure, output_mismatch])

    assert raw_rewards == pytest.approx([-0.1, 0.15])


def test_dynamic_reward_rebuilds_trloo_returns_after_per_turn_weighting(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", True)
    manager = _make_manager(advantage_estimator="trloo", use_multi_turn=True)
    manager.args.multi_turn_gamma = 0.5
    samples = [
        _make_sample(0, 0, 0.6, turn_idx=0),
        _make_sample(1, 0, 0.7, turn_idx=0),
        _make_sample(0, 0, 0.9, turn_idx=1),
        _make_sample(1, 0, 0.0, turn_idx=1),
    ]
    _set_reward_component(samples[0], correctness_score=1.0, performance_score=0.2)
    _set_reward_component(samples[1], correctness_score=1.0, performance_score=0.4)
    _set_reward_component(samples[2], correctness_score=1.0, performance_score=0.8)
    _set_reward_component(samples[3])
    for sample in samples:
        sample.metadata["multi_turn_reward"] = -999.0

    raw_rewards, advantages = reward_post_process_by_group(manager.args, samples)

    # turn 0: C=N => gate 1; turn 1: C=1 => gate 0.
    assert raw_rewards == pytest.approx([0.85, 0.7, 0.5, 0.0])
    assert advantages == pytest.approx([0.15, -0.15, 0.5, -0.5])


@pytest.mark.parametrize(
    ("advantage_estimator", "grpo_std_normalization"),
    [
        ("grpo", False),
        ("grpo", True),
        ("gspo", False),
        ("rloo", False),
        ("reinforce_plus_plus_baseline", False),
    ],
)
def test_post_process_rewards_by_group_matches_original_last_turn(
    advantage_estimator: str, grpo_std_normalization: bool
):
    manager = _make_manager(
        advantage_estimator=advantage_estimator,
        grpo_std_normalization=grpo_std_normalization,
        use_multi_turn=False,
    )
    samples = [
        _make_sample(0, 0, 1.0),
        _make_sample(1, 0, 2.0),
        _make_sample(2, 0, 4.0),
        _make_sample(3, 1, 3.0),
        _make_sample(4, 1, 6.0),
        _make_sample(5, 1, 9.0),
    ]

    raw_rewards, rewards = manager._post_process_rewards(samples)
    raw_rewards_by_group, rewards_by_group = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards_by_group == raw_rewards
    assert rewards_by_group == pytest.approx(rewards)


@pytest.mark.parametrize(
    ("advantage_estimator", "rloo_scale"),
    [("grpo", 1.0), ("rloo", 1.5), ("reinforce_plus_plus_baseline", 1.0)],
)
def test_reward_post_process_by_group_normalizes_each_turn(advantage_estimator: str, rloo_scale: float):
    manager = _make_manager(advantage_estimator=advantage_estimator, use_multi_turn=True)
    samples = [
        _make_sample(0, 0, 1.0, turn_idx=0),
        _make_sample(1, 0, 2.0, turn_idx=0),
        _make_sample(2, 0, 4.0, turn_idx=0),
        _make_sample(0, 0, 3.0, turn_idx=1),
        _make_sample(1, 0, 6.0, turn_idx=1),
        _make_sample(2, 0, 9.0, turn_idx=1),
        _make_sample(3, 1, 2.0, turn_idx=0),
        _make_sample(4, 1, 5.0, turn_idx=0),
        _make_sample(5, 1, 8.0, turn_idx=0),
        _make_sample(3, 1, 4.0, turn_idx=1),
        _make_sample(4, 1, 7.0, turn_idx=1),
        _make_sample(5, 1, 10.0, turn_idx=1),
    ]

    raw_rewards, rewards = reward_post_process_by_group(manager.args, samples)
    expected_rewards = []
    for start in range(0, len(raw_rewards), manager.args.n_samples_per_prompt):
        group_rewards = raw_rewards[start : start + manager.args.n_samples_per_prompt]
        group_mean = sum(group_rewards) / len(group_rewards)
        expected_rewards.extend((reward - group_mean) * rloo_scale for reward in group_rewards)

    assert rewards == pytest.approx(expected_rewards)


def test_reward_post_process_by_group_handles_single_valid_sample_after_pad_masking():
    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True, grpo_std_normalization=True)
    valid_sample = _make_sample(0, 0, 3.0, turn_idx=1)
    pad_sample = _make_sample(1, 0, 0.0, turn_idx=1)
    pad_sample.remove_sample = True
    pad_sample.loss_mask = [0]
    pad_sample.metadata["is_pad_turn"] = True

    raw_rewards, rewards = reward_post_process_by_group(manager.args, [valid_sample, pad_sample])

    assert raw_rewards == [3.0, 0.0]
    assert rewards == pytest.approx([0.0, 0.0])


def test_trloo_uses_kernel_failed_scores_only_for_all_failed_group(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    manager = _make_manager(advantage_estimator="trloo", use_multi_turn=False)
    samples = [
        _make_sample(0, 0, 0.0),
        _make_sample(1, 0, 0.0),
        _make_sample(2, 0, 0.0),
    ]
    kernel_failed_scores = [-1.0, -0.75, -0.25]
    for sample, kernel_failed_score in zip(samples, kernel_failed_scores, strict=True):
        sample.metadata.update(
            {
                "multi_turn_reward": 0.0,
                "kernel_failed_score": kernel_failed_score,
                "task_reward": 0.0,
                "reward_component": {"failed": 0.0},
            }
        )

    assert [sample.metadata["reward_component"]["failed"] for sample in samples] == [0.0, 0.0, 0.0]

    raw_rewards, rewards = reward_post_process_by_group(manager.args, samples)

    expected_raw_rewards = [-0.5, -0.375, -0.125]
    assert raw_rewards == expected_raw_rewards
    assert rewards == pytest.approx([-0.25, -0.0625, 0.3125])
    assert [sample.metadata["task_reward"] for sample in samples] == expected_raw_rewards
    assert [sample.metadata["reward_component"]["failed"] for sample in samples] == expected_raw_rewards


def test_failed_group_reward_uses_configured_nonzero_failed_score(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "failed_score", -2.0)
    manager = _make_manager(advantage_estimator="trloo", use_multi_turn=False)
    samples = [_make_sample(index, 0, -1.0) for index in range(3)]
    kernel_failed_scores = [-1.0, -0.75, -0.25]
    for sample, kernel_failed_score in zip(samples, kernel_failed_scores, strict=True):
        sample.metadata.update({"multi_turn_reward": -1.0, "kernel_failed_score": kernel_failed_score})

    raw_rewards, rewards = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == pytest.approx([-0.5, -0.375, -0.125])
    assert rewards == pytest.approx([-0.25, -0.0625, 0.3125])


def test_failed_group_reward_can_be_disabled(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", False)
    manager = _make_manager(advantage_estimator="trloo", use_multi_turn=False)
    samples = [_make_sample(index, 0, 0.0) for index in range(3)]
    for sample, kernel_failed_score in zip(samples, [-1.0, -0.75, -0.25], strict=True):
        sample.metadata.update({"multi_turn_reward": 0.0, "kernel_failed_score": kernel_failed_score})

    raw_rewards, rewards = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == [0.0, 0.0, 0.0]
    assert rewards == pytest.approx([0.0, 0.0, 0.0])


def test_kernel_failed_scores_do_not_change_group_with_nonzero_reward(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    manager = _make_manager(advantage_estimator="trloo", use_multi_turn=False)
    samples = [
        _make_sample(0, 0, 0.0),
        _make_sample(1, 0, 0.0),
        _make_sample(2, 0, 0.5),
    ]
    for sample, kernel_failed_score in zip(samples, [-1.0, -0.75, -0.25], strict=True):
        sample.metadata.update({"multi_turn_reward": sample.reward, "kernel_failed_score": kernel_failed_score})

    raw_rewards, _ = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == [0.0, 0.0, 0.5]


def test_all_failed_reward_from_old_dump_without_kernel_failed_scores_is_unchanged(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    manager = _make_manager(advantage_estimator="trloo", use_multi_turn=False)
    samples = [_make_sample(index, 0, 0.0) for index in range(3)]
    for sample in samples:
        sample.metadata["multi_turn_reward"] = 0.0

    raw_rewards, rewards = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == [0.0, 0.0, 0.0]
    assert rewards == pytest.approx([0.0, 0.0, 0.0])


def _make_ctm_candidate(
    index: int,
    reward: float,
    *,
    correctness: bool,
    status: Sample.Status,
    decoy_kernel: bool = False,
    tokens: list[int] | None = None,
) -> Sample:
    sample = _make_sample(index, 0, reward, turn_idx=1)
    sample.tokens = [0, 1, 2, 3] if tokens is None else tokens
    sample.response_length = 4
    sample.loss_mask = [1] * 4
    sample.status = status
    sample.metadata["env_extra_info"] = {
        "correctness": correctness,
        "decoy_kernel": decoy_kernel,
    }
    return sample


def test_ctm_masks_advantage_after_full_group_normalization():
    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True)
    _enable_ctm(manager.args)
    samples = [
        _make_ctm_candidate(0, 1.0, correctness=False, status=Sample.Status.COMPLETED),
        _make_ctm_candidate(1, 2.0, correctness=False, status=Sample.Status.TRUNCATED),
        _make_ctm_candidate(2, 4.0, correctness=False, status=Sample.Status.COMPLETED),
    ]

    raw_rewards, rewards = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == [1.0, 2.0, 4.0]
    assert rewards == pytest.approx([-4.0 / 3.0, 0.0, 5.0 / 3.0])
    assert samples[1].reward == 2.0
    assert samples[1].remove_sample is False
    assert samples[1].metadata["conditional_truncation_masked"] is True


@pytest.mark.parametrize(
    ("correctness", "status", "decoy_kernel", "expected_masked"),
    [
        (True, Sample.Status.COMPLETED, False, True),
        (False, Sample.Status.TRUNCATED, False, True),
        (False, Sample.Status.COMPLETED, False, False),
        (True, Sample.Status.COMPLETED, True, False),
    ],
)
def test_ctm_requires_non_incorrect_response(correctness, status, decoy_kernel, expected_masked):
    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True)
    _enable_ctm(manager.args)
    candidate = _make_ctm_candidate(0, 1.0, correctness=correctness, status=status, decoy_kernel=decoy_kernel)

    _raw_rewards, rewards = reward_post_process_by_group(manager.args, [candidate])

    assert candidate.remove_sample is False
    assert candidate.metadata.get("conditional_truncation_masked", False) is expected_masked
    assert candidate.metadata.get("conditional_truncation_masking_eligible", False) is expected_masked
    assert rewards == pytest.approx([0.0])


@pytest.mark.parametrize(
    ("response_length", "tokens"),
    [
        (3, [0, 1, 2]),
        (4, [0, 1, 0, 1]),
    ],
)
def test_ctm_rejects_non_max_length_or_repeated_response(response_length, tokens):
    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True)
    _enable_ctm(manager.args)
    candidate = _make_ctm_candidate(0, 1.0, correctness=True, status=Sample.Status.COMPLETED, tokens=tokens)
    candidate.response_length = response_length
    candidate.loss_mask = [1] * response_length

    _raw_rewards, rewards = reward_post_process_by_group(manager.args, [candidate])

    assert candidate.remove_sample is False
    assert candidate.metadata.get("conditional_truncation_masking_eligible", False) is False
    assert rewards == pytest.approx([0.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
