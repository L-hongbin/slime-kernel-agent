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
from examples.kernel_agent.kernel_reward import (
    _apply_dynamic_group_reward_weights,
    _compute_dynamic_auxiliary_gate,
    annotate_group_difficulty,
    calculate_kernel_reward,
    post_process_rollout_rewards,
    resolve_rollout_reward_processors,
    reward_post_process_by_group,
)
from slime.observability.rollout_metrics import compute_reward_post_process_metrics
from slime.ray.rollout import RolloutManager
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("mode", ["baseline", "anchor"])
@pytest.mark.parametrize("estimator", ["rloo", "trloo", "grpo"])
def test_verify_utility_reaches_training_advantage(monkeypatch, dynamic, mode, estimator):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", dynamic)
    args = _make_manager(advantage_estimator=estimator, use_multi_turn=True).args
    samples = [
        Sample(
            index=index,
            group_index=0,
            reward=reward,
            metadata={"role": "verify", "turn_idx": 0, "multi_turn_reward": reward, "verify_reward_mode": mode},
        )
        for index, reward in enumerate([0.3, 0.7])
    ]
    raw, advantages = reward_post_process_by_group(args, samples)
    assert raw == pytest.approx([0.3, 0.7])
    expected = [0.3, 0.7] if mode == "anchor" else ([-0.2, 0.2] if estimator == "grpo" else [-0.4, 0.4])
    assert advantages == pytest.approx(expected)


def test_singleton_verify_anchor_keeps_absolute_improvement():
    args = _make_manager(advantage_estimator="rloo", use_multi_turn=False).args
    sample = Sample(reward=-0.2, metadata={"role": "verify", "verify_reward_mode": "anchor"})
    assert reward_post_process_by_group(args, [sample])[1] == pytest.approx([-0.2])


def test_pending_shared_anchor_cannot_enter_training():
    args = _make_manager(advantage_estimator="rloo", use_multi_turn=True).args
    sample = Sample(reward=0.8, metadata={"role": "verify", "verify_reward_mode": "pending_anchor"})
    with pytest.raises(ValueError, match="settled by the group rollout"):
        reward_post_process_by_group(args, [sample])


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("estimator", ["rloo", "trloo", "grpo"])
@pytest.mark.parametrize("group_size", [1, 2])
def test_verify_history_advantage_preserves_signed_improvement_and_raw_reward(
    monkeypatch, dynamic, estimator, group_size
):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", dynamic)
    args = _make_manager(advantage_estimator=estimator, use_multi_turn=True, grpo_std_normalization=True).args
    args.verify_advantage_baseline = "history"
    samples = []
    for index in range(group_size):
        for turn_idx, reward in enumerate([0.75, 0.75, 0.125, 0.125]):
            samples.append(
                Sample(
                    index=index,
                    group_index=0,
                    reward=reward,
                    metadata={
                        "role": "verify" if turn_idx % 2 == 0 else "kernel",
                        "turn_idx": turn_idx,
                        "verify_trajectory": True,
                        "verify_reward_mode": "baseline",
                        "verify_source_reward": 0.25,
                        "multi_turn_reward": reward,
                    },
                )
            )
    raw, advantages = reward_post_process_by_group(args, samples)
    assert raw == pytest.approx([0.75, 0.75, 0.125, 0.125] * group_size)
    assert advantages == pytest.approx([0.5, 0.0, -0.125, 0.0] * group_size)
    assert [sample.reward for sample in samples] == raw


@pytest.mark.parametrize("source_reward", [None, float("nan"), float("inf")])
def test_verify_history_advantage_requires_finite_baseline(source_reward):
    args = _make_manager(advantage_estimator="rloo", use_multi_turn=True).args
    args.verify_advantage_baseline = "history"
    sample = Sample(reward=0.7, metadata={"role": "verify", "verify_source_reward": source_reward})
    with pytest.raises(ValueError, match="finite metadata"):
        reward_post_process_by_group(args, [sample])


def test_verify_history_advantage_does_not_subtract_from_anchor_difference():
    args = _make_manager(advantage_estimator="rloo", use_multi_turn=True).args
    args.verify_advantage_baseline = "history"
    sample = Sample(reward=0.4, metadata={"role": "verify", "verify_reward_mode": "anchor"})
    with pytest.raises(ValueError, match="cannot be applied to anchor-scored"):
        reward_post_process_by_group(args, [sample])


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("mode", ["baseline", "anchor"])
def test_verify_pairs_keep_own_kernel_reward_without_future_pair_returns(monkeypatch, dynamic, mode):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", dynamic)
    args = _make_manager(advantage_estimator="trloo", use_multi_turn=True).args
    samples = [
        Sample(
            index=index,
            group_index=0,
            reward=reward,
            metadata={"role": "verify", "turn_idx": turn, "multi_turn_reward": reward, "verify_reward_mode": mode},
        )
        for index, turn, reward in [(0, 0, 0.3), (0, 1, 0.9), (1, 0, 0.7), (1, 1, 0.1)]
    ]
    raw, advantages = reward_post_process_by_group(args, samples)
    assert raw == pytest.approx([0.3, 0.9, 0.7, 0.1])
    expected = raw if mode == "anchor" else [-0.4, 0.8, 0.4, -0.8]
    assert advantages == pytest.approx(expected)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("mode", ["baseline", "anchor"])
@pytest.mark.parametrize("estimator", ["rloo", "trloo", "grpo"])
def test_joint_verify_kernel_training_preserves_per_turn_rewards(monkeypatch, dynamic, mode, estimator):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", dynamic)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    args = _make_manager(advantage_estimator=estimator, use_multi_turn=True).args
    samples = []
    expected_raw = []
    expected_advantages = []
    for index, kernel_rewards in enumerate([(0.5, 1.1), (0.9, 0.3)]):
        for pair_idx, kernel_reward in enumerate(kernel_rewards):
            group_advantage = [-0.2, 0.4][pair_idx] * (1 if index == 0 else -1)
            if estimator != "grpo":
                group_advantage *= 2
            for offset, role in enumerate(["verify", "kernel"]):
                is_anchor_verify = mode == "anchor" and role == "verify"
                reward = kernel_reward - 0.2 if is_anchor_verify else kernel_reward
                sample = Sample(
                    index=index,
                    group_index=0,
                    reward=reward,
                    metadata={
                        "role": role,
                        "turn_idx": 2 * pair_idx + offset,
                        "verify_trajectory": True,
                        "multi_turn_reward": reward,
                        "verify_reward_mode": mode,
                    },
                )
                if role == "kernel":
                    # Dynamic/failed-group rewriting would desynchronize the pair's assigned utility.
                    _set_reward_component(sample, correctness_score=0.0, performance_score=9.0, failed=-1.0)
                samples.append(sample)
                expected_raw.append(reward)
                expected_advantages.append(reward if is_anchor_verify else group_advantage)

    raw, advantages = reward_post_process_by_group(args, samples)
    assert raw == pytest.approx(expected_raw)
    assert advantages == pytest.approx(expected_advantages)
    assert [sample.reward for sample in samples] == pytest.approx(expected_raw)


def _make_manager(*, advantage_estimator: str, use_multi_turn: bool, grpo_std_normalization: bool = False):
    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.custom_reward_post_process_func = None
    manager.rollout_id = 0
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
        wandb_always_use_train_step=False,
        use_wandb=False,
        use_tensorboard=False,
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


@pytest.mark.parametrize(
    "num_correct,group_size,expected",
    [
        (0, 8, 0.8),
        (1, 8, 0.9),
        (2, 8, 1.0),
        (4, 8, 1.0),
        (6, 8, 1.0),
        (7, 8, 1.1),
        (8, 8, 1.2),
        (1, 2, 1.0),
        (0, 0, 1.0),
        (0, 1, 1.0),
        (1, 1, 1.0),
    ],
)
def test_piecewise_dynamic_gate_default_regions(num_correct, group_size, expected):
    args = SimpleNamespace(dynamic_reward_gate="piecewise")
    assert _compute_dynamic_auxiliary_gate(num_correct, group_size, args=args) == pytest.approx(expected)


def test_piecewise_dynamic_gate_custom_bounds_are_continuous_and_monotonic():
    args = SimpleNamespace(
        dynamic_reward_gate="piecewise",
        dynamic_reward_gate_range=[0.4, 2.0],
        difficulty_thresholds=[0.2, 0.8],
    )
    gates = [_compute_dynamic_auxiliary_gate(correct, 100, args=args) for correct in range(101)]
    assert gates == sorted(gates)
    assert gates[0] == 0.4
    assert gates[10] == pytest.approx(0.7)
    assert gates[20:81] == [1.0] * 61
    assert gates[90] == pytest.approx(1.5)
    assert gates[100] == 2.0
    assert max(right - left for left, right in zip(gates, gates[1:], strict=False)) <= 0.05 + 1e-12


def test_piecewise_gate_boosts_auxiliary_components_and_records_positive_delta(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_performance_weight", 0.5)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "coverage_reward_weight", 0.5)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "coverage_reward_enable", True)
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.rollout_reward_post_processors = ["dynamic-weight", "overlong-penalty"]
    args.dynamic_reward_gate = "piecewise"
    args.dynamic_reward_gate_range = [0.8, 2.0]
    args.rollout_max_response_len = 100
    args.overlong_buffer_len = 100
    args.overlong_penalty_factor = 0.1
    samples = [_make_sample(i, 0, 1.5 + 0.5 * i) for i in range(2)]
    for i, sample in enumerate(samples):
        sample.response_length = 100
        sample.tokens = [0] * 100
        _set_reward_component(sample, correctness_score=1.0, performance_score=1.0 + i, coverage_score=1.0)
    for _ in range(2):
        assert post_process_rollout_rewards(args, samples) == pytest.approx([2.4, 3.4])
        assert [s.metadata["reward_component"]["coverage"] for s in samples] == [1.0, 1.0]
        assert [s.metadata["reward_component"]["correctness"] for s in samples] == [0.5, 0.5]
        metrics = compute_reward_post_process_metrics(samples)
        assert metrics["rollout/dynamic_reward/gate_mean"] == 2.0
        assert metrics["rollout/dynamic_reward/gate_boosted_fraction"] == 1.0
        assert metrics["rollout/dynamic_reward/gate_scaled_fraction"] == 0.0
        assert metrics["rollout/dynamic_reward/performance_reward_delta_mean"] == 0.75


def test_annotate_group_difficulty_records_shared_group_statistics():
    correct = _make_sample(0, 0, 1.0, turn_idx=1)
    incorrect = _make_sample(1, 0, 0.0, turn_idx=1)
    removed = _make_sample(2, 0, 0.0, turn_idx=1)
    padded = _make_sample(3, 0, 0.0, turn_idx=1)
    aborted = _make_sample(4, 0, 0.0, turn_idx=1)
    _set_reward_component(correct, correctness_score=1.0)
    _set_reward_component(incorrect)
    _set_reward_component(removed)
    _set_reward_component(padded)
    _set_reward_component(aborted)
    removed.remove_sample = True
    padded.metadata["is_pad_turn"] = True
    aborted.status = Sample.Status.ABORTED

    samples = [correct, incorrect, removed, padded, aborted]
    assert annotate_group_difficulty(samples) == (1, 2)

    for sample in samples:
        assert sample.metadata["group_num_correct"] == 1
        assert sample.metadata["group_num_valid"] == 2
        assert sample.metadata["group_correct_rate"] == pytest.approx(0.5)
        assert sample.metadata["group_difficulty"] == pytest.approx(0.5)


def test_annotate_group_difficulty_does_not_mark_empty_group_as_hard():
    removed = _make_sample(0, 0, 0.0)
    removed.remove_sample = True

    assert annotate_group_difficulty([removed]) == (0, 0)
    assert removed.metadata["group_correct_rate"] == 0.0
    assert removed.metadata["group_difficulty"] == 0.0


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
    assert [sample.reward for sample in samples] == pytest.approx(expected_raw)
    assert samples[1].metadata["kernel_score"] == {
        "correctness": 1.0,
        "performance": 0.2,
        "coverage": 0.4,
    }
    assert samples[1].metadata["reward_component"]["performance"] == pytest.approx(gate * 0.1)
    assert samples[1].metadata["reward_component"]["coverage"] == pytest.approx(gate * 0.2)
    for sample in samples:
        assert sample.metadata["group_num_correct"] == 2
        assert sample.metadata["group_num_valid"] == 4
        assert sample.metadata["group_correct_rate"] == pytest.approx(0.5)
        assert sample.metadata["group_difficulty"] == pytest.approx(0.5)


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
    assert correct.reward == pytest.approx(0.3)
    assert correct.reward == pytest.approx(
        sum(v for v in correct.metadata["reward_component"].values() if v is not None)
    )
    # Reprocessing must rebuild from base scores, not compound weighting or penalties.
    assert reward_post_process_by_group(manager.args, [correct, incorrect])[0] == pytest.approx(raw_rewards)
    assert correct.reward == pytest.approx(0.3)


def test_dynamic_reward_writeback_preserves_other_reward_keys(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", True)
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    manager.args.reward_key = "kernel"
    sample = _make_sample(0, 0, 99.0)
    original_reward = {"kernel": 99.0, "other": 7.0}
    sample.reward = original_reward
    _set_reward_component(sample, correctness_score=1.0, performance_score=1.0, overlong_penalty=-0.2)

    raw, _ = reward_post_process_by_group(manager.args, [sample])

    assert raw == pytest.approx([0.3])
    assert sample.reward == pytest.approx({"kernel": 0.3, "other": 7.0})
    assert original_reward == {"kernel": 99.0, "other": 7.0}


def test_dynamic_reward_writeback_preserves_captured_verify_history(monkeypatch):
    from examples.kernel_agent import kernel_agent_data_source as source_module

    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", True)
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    manager.args.verify_advantage_baseline = "history"
    source = source_module.KernelAgentDataSource.__new__(source_module.KernelAgentDataSource)
    source.args = manager.args
    sample = _make_sample(0, 0, 0.25)
    sample.metadata["trajectory_states"] = ["failed"]
    _set_reward_component(sample, failed=0.25, overlong_penalty=-0.1)
    monkeypatch.setattr(
        source_module, "extract_cuda_agent_kernel_code", lambda response: f"### {source_module.CUDA_SECTIONS[0]}\n```"
    )
    candidate = source._prepare_verify_candidate(sample, rollout_id=3)
    assert candidate is not None

    reward_post_process_by_group(manager.args, [sample])

    assert sample.reward == pytest.approx(0.15)
    assert candidate.reward == 0.25
    assert candidate.metadata["verify_source_reward"] == 0.25


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
    assert [kernel_failure.reward, output_mismatch.reward] == pytest.approx(raw_rewards)


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
    assert [sample.reward for sample in samples] == pytest.approx([0.6, 0.7, 0.5, 0.0])
    repeated_raw, repeated_advantages = reward_post_process_by_group(manager.args, samples)
    assert repeated_raw == pytest.approx(raw_rewards)
    assert repeated_advantages == pytest.approx(advantages)


@pytest.mark.parametrize(
    "processors",
    [
        ["none"],
        ["dynamic-weight"],
        ["overlong-penalty"],
        ["dynamic-weight", "overlong-penalty"],
        ["overlong-penalty", "dynamic-weight"],
    ],
)
def test_rollout_reward_processors_override_legacy_and_keep_components_consistent(monkeypatch, processors):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", True)
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.rollout_reward_post_processors = processors
    args.overlong_penalty = True
    args.overlong_buffer_len = 100
    args.overlong_penalty_factor = 0.2
    args.rollout_max_response_len = 100
    env = {"status": "completed", "compiled": True, "correctness": True, "speedup": 1.0}
    samples = []
    for i in range(3):
        sample = _make_sample(i, 0, 0.0)
        sample.response_length = 100
        sample.tokens = [0] * 100
        details = calculate_kernel_reward({**env, "correctness": i < 2}, CUDA_AGENT_CONFIGS["reward"])
        sample.reward = details.pop("reward")
        sample.metadata.update(details)
        samples.append(sample)
    base_scores = [dict(s.metadata["kernel_score"]) for s in samples]
    penalty = 0.2 if "overlong-penalty" in processors else 0.0
    # Scoring cannot use incomplete group statistics. Length shaping settles
    # first, through the same interface that owns final rollout processing.
    assert post_process_rollout_rewards(args, samples, stage="sample") == pytest.approx(
        [1.0 - penalty, 1.0 - penalty, -penalty]
    )
    task_reward = 0.5 + 0.5 * ((0.5**0.5) if "dynamic-weight" in processors else 1.0)
    expected = [task_reward - penalty, task_reward - penalty, -penalty]

    for _ in range(2):
        assert post_process_rollout_rewards(args, samples) == pytest.approx(expected)
        assert [s.reward for s in samples] == pytest.approx(expected)
        for sample, scores in zip(samples, base_scores, strict=True):
            metadata = sample.metadata
            assert metadata["kernel_score"] == scores
            assert metadata["reward_component"]["overlong_penalty"] == pytest.approx(-penalty)
            assert metadata["overlong_penalty"] == pytest.approx(penalty)
            assert sample.reward == pytest.approx(
                sum(v for v in metadata["reward_component"].values() if v is not None)
            )
            assert sample.reward == pytest.approx(metadata["task_reward"] - penalty)
    # The existing training hook delegates shaping, then centers rewards separately.
    raw, advantages = reward_post_process_by_group(args, samples)
    assert raw == pytest.approx(expected)
    assert advantages == pytest.approx([reward - sum(expected) / 3 for reward in expected])
    assert [s.reward for s in samples] == pytest.approx(expected)


def test_rollout_length_processor_does_not_require_group_or_advantage_args():
    args = SimpleNamespace(
        reward_key=None,
        rollout_reward_post_processors=["overlong-penalty"],
        rollout_max_response_len=100,
        overlong_buffer_len=20,
        overlong_penalty_factor=0.2,
    )
    sample = Sample(
        reward=1.0,
        response_length=90,
        tokens=[0] * 90,
        metadata={"task_reward": 1.0, "reward_component": {"correctness": 1.0}},
    )
    assert post_process_rollout_rewards(args, [sample]) == pytest.approx([0.9])
    assert sample.reward == pytest.approx(0.9)


@pytest.mark.parametrize(
    "metadata,removed",
    [
        ({"role": "verify", "verify_source_reward": 0.7}, False),
        ({"role": "kernel", "verify_trajectory": True}, False),
        ({"role": "pad", "is_pad_turn": True}, True),
    ],
)
def test_rollout_reward_processors_leave_settled_verify_and_pad_untouched(metadata, removed):
    import copy

    args = _make_manager(advantage_estimator="grpo", use_multi_turn=True).args
    args.rollout_reward_post_processors = ["dynamic-weight", "overlong-penalty"]
    sample = Sample(reward=0.7, metadata=metadata, remove_sample=removed)
    before = copy.deepcopy(sample.to_dict())
    assert post_process_rollout_rewards(args, [sample]) == [0.7]
    assert sample.to_dict() == before


def test_sample_reward_stage_shapes_verify_kernel_before_utility_settlement():
    args = SimpleNamespace(
        rollout_reward_post_processors=["dynamic-weight", "overlong-penalty"],
        use_multi_turn=True,
        rollout_max_response_len=100,
        overlong_buffer_len=20,
        overlong_penalty_factor=0.2,
    )
    kernel = Sample(
        reward=1.0,
        response_length=100,
        tokens=[0] * 100,
        metadata={
            "role": "kernel",
            "verify_trajectory": True,
            "verify_source_reward": 0.3,
            "task_reward": 1.0,
            "reward_component": {"correctness": 1.0},
        },
    )
    # No turn_idx or complete group: only the length processor may run here.
    assert post_process_rollout_rewards(args, [kernel], stage="sample") == pytest.approx([0.8])
    assert kernel.metadata["verify_source_reward"] == 0.3
    verify = Sample(reward=kernel.reward, metadata={"role": "verify", "verify_trajectory": True})
    assert post_process_rollout_rewards(args, [verify], stage="sample") == pytest.approx([0.8])
    # Once utilities are settled, neither turn gets reshaped in the rollout stage.
    assert post_process_rollout_rewards(args, [verify, kernel]) == pytest.approx([0.8, 0.8])
    assert kernel.metadata["reward_component"]["overlong_penalty"] == pytest.approx(-0.2)


def test_rollout_reward_processors_reject_unknown_stage():
    with pytest.raises(ValueError, match="stage"):
        post_process_rollout_rewards(SimpleNamespace(), [], stage="unknown")


def test_dynamic_reward_metrics_use_group_gates_and_ungated_performance(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_performance_weight", 0.5)
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=True).args
    args.rollout_reward_post_processors = ["dynamic-weight"]
    samples = []
    groups = []
    for group_idx, turn, correct_flags in [(0, 0, [True, True, False]), (0, 1, [True, True]), (1, 0, [True, False])]:
        group = []
        for correct in correct_flags:
            sample = _make_sample(len(samples), group_idx, float(correct), turn_idx=turn)
            _set_reward_component(
                sample,
                correctness_score=float(correct),
                performance_score=float(correct),
                failed=None if correct else 0.0,
            )
            samples.append(sample)
            group.append(sample)
        groups.append(group)

    # The dynamic-filter preview must not produce final metrics, and it must
    # not erase the baseline used to measure the final performance delta.
    for group in groups:
        _apply_dynamic_group_reward_weights(
            group, [s.reward for s in group], {**CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight": True}
        )
    assert compute_reward_post_process_metrics(samples) == {}
    partial_gate = 0.5**0.5
    for _ in range(2):
        post_process_rollout_rewards(args, samples)
        metrics = compute_reward_post_process_metrics(samples)
        prefix = "rollout/dynamic_reward/"
        assert metrics[f"{prefix}group_count"] == 3
        assert metrics[f"{prefix}sample_count"] == 7
        assert metrics[f"{prefix}gate_mean"] == pytest.approx((partial_gate + 1.0) / 3)
        assert metrics[f"{prefix}gate_min"] == 0.0
        assert metrics[f"{prefix}gate_max"] == 1.0
        assert metrics[f"{prefix}gate_p25"] == pytest.approx(partial_gate / 2)
        assert metrics[f"{prefix}gate_p50"] == pytest.approx(partial_gate)
        assert metrics[f"{prefix}gate_p75"] == pytest.approx((partial_gate + 1.0) / 2)
        for name in ("gate_zero_fraction", "gate_scaled_fraction", "gate_one_fraction"):
            assert metrics[f"{prefix}{name}"] == pytest.approx(1 / 3)
        assert metrics[f"{prefix}performance_reward_delta_mean"] == pytest.approx((partial_gate - 1.5) / 7)
        assert metrics[f"{prefix}performance_reward_delta_min"] == -0.5
        assert metrics[f"{prefix}performance_reward_delta_max"] == 0.0


@pytest.mark.parametrize(
    "metadata,removed,status",
    [
        ({"role": "verify"}, False, Sample.Status.COMPLETED),
        ({"verify_trajectory": True}, False, Sample.Status.COMPLETED),
        ({"verify_scoring_branch": "anchor"}, False, Sample.Status.COMPLETED),
        ({"role": "pad"}, False, Sample.Status.COMPLETED),
        ({"is_pad_turn": True}, False, Sample.Status.COMPLETED),
        ({}, True, Sample.Status.COMPLETED),
        ({}, False, Sample.Status.ABORTED),
    ],
)
def test_dynamic_reward_metrics_exclude_non_training_samples(metadata, removed, status):
    sample = Sample(
        reward=0.0,
        metadata={**metadata, "dynamic_reward": {"gate": 0.5, "performance_reward_delta": -0.25}},
        remove_sample=removed,
        status=status,
    )
    assert compute_reward_post_process_metrics([sample]) == {}


@pytest.mark.parametrize("processors,legacy", [(["none"], True), (None, False)])
def test_disabled_dynamic_reward_clears_stale_metric_records(monkeypatch, processors, legacy):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "enable_dynamic_reward_weight", legacy)
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.rollout_reward_post_processors = processors
    sample = Sample(reward=1.0, metadata={"dynamic_reward": {"gate": 0.0, "performance_reward_delta": -1.0}})
    assert post_process_rollout_rewards(args, [sample]) == [1.0]
    assert "dynamic_reward" not in sample.metadata
    assert compute_reward_post_process_metrics([sample]) == {}


@pytest.mark.parametrize("train_step_axis", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_dynamic_reward_metrics_logged_after_final_processing(monkeypatch, caplog, dynamic, train_step_axis):
    import logging

    import slime.ray.rollout as rollout_module

    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=False)
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = reward_post_process_by_group
    manager.rollout_id = 4
    manager.args.wandb_always_use_train_step = train_step_axis
    manager.args.global_batch_size = 3
    manager.args.rollout_reward_post_processors = ["dynamic-weight"] if dynamic else ["none"]
    samples = [_make_sample(i, 0, float(i < 2)) for i in range(3)]
    for i, sample in enumerate(samples):
        sample.tokens = [1, 2]
        sample.response_length = 1
        _set_reward_component(
            sample, correctness_score=float(i < 2), performance_score=float(i < 2), failed=None if i < 2 else 0.0
        )
    logged = []

    def capture_log(args, metrics, step_key):
        assert args is manager.args
        assert step_key == "rollout/step"
        # The sample has already been shaped; this is not the earlier rollout log.
        assert samples[0].reward == pytest.approx(0.5 + 0.5 * (0.5**0.5))
        logged.append(dict(metrics))

    monkeypatch.setattr(rollout_module.logging_utils, "log", capture_log)
    with caplog.at_level(logging.INFO):
        manager._convert_samples_to_train_data(samples)
    if not dynamic:
        assert logged == []
        assert "reward post-process" not in caplog.text
        return
    assert len(logged) == 1
    assert logged[0]["rollout/step"] == (8 if train_step_axis else 4)
    if train_step_axis:
        assert logged[0]["train/step"] == 8
    assert logged[0]["rollout/dynamic_reward/gate_mean"] == pytest.approx(0.5**0.5)
    assert logged[0]["rollout/dynamic_reward/performance_reward_delta_mean"] == pytest.approx(((0.5**0.5) - 1) / 3)
    assert "reward post-process 4" in caplog.text


def test_dynamic_reward_metrics_reach_tensorboard_as_scalars(monkeypatch):
    from slime.observability import logging_utils
    from slime.observability.tensorboard_utils import _TensorboardAdapter

    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.rollout_reward_post_processors = ["dynamic-weight"]
    args.use_tensorboard = True
    samples = [_make_sample(i, 0, 1.0) for i in range(2)]
    for sample in samples:
        _set_reward_component(sample, correctness_score=1.0, performance_score=1.0)
    post_process_rollout_rewards(args, samples)
    metrics = compute_reward_post_process_metrics(samples)
    scalars, flushes = [], []
    # Exercise the real adapter without opening a TensorBoard file or tracker.
    adapter = object.__new__(_TensorboardAdapter)
    adapter._writer = SimpleNamespace(
        add_scalar=lambda key, value, step: scalars.append((key, value, step)),
        flush=lambda: flushes.append(True),
    )
    monkeypatch.setattr(logging_utils, "_TensorboardAdapter", lambda args: adapter)
    logging_utils.log(args, {**metrics, "rollout/step": 4}, step_key="rollout/step")
    assert len(scalars) == len(metrics)
    assert all(isinstance(value, (int, float)) and step == 4 for _, value, step in scalars)
    assert ("rollout/dynamic_reward/gate_mean", 1.0, 4) in scalars
    assert ("rollout/dynamic_reward/performance_reward_delta_mean", 0.0, 4) in scalars
    assert flushes == [True]


@pytest.mark.parametrize("dynamic,overlong", [(False, False), (True, False), (False, True), (True, True)])
def test_rollout_reward_processor_legacy_selection(dynamic, overlong):
    args = SimpleNamespace(overlong_penalty=overlong)
    expected = ({"dynamic-weight"} if dynamic else set()) | ({"overlong-penalty"} if overlong else set())
    assert resolve_rollout_reward_processors(args, {"enable_dynamic_reward_weight": dynamic}) == expected


@pytest.mark.parametrize(
    "processors", [["none", "dynamic-weight"], ["dynamic-weight", "dynamic-weight"], ["unknown"], [], "none"]
)
def test_rollout_reward_processor_rejects_invalid_selection(processors):
    with pytest.raises(ValueError):
        resolve_rollout_reward_processors(SimpleNamespace(rollout_reward_post_processors=processors), {})


@pytest.mark.parametrize(
    "processors", [["none"], ["dynamic-weight"], ["overlong-penalty"], ["dynamic-weight", "overlong-penalty"]]
)
def test_rollout_reward_processor_args_parse_and_validate(processors):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--rollout-reward-post-processors",
            *processors,
            "--custom-reward-post-process-path",
            "examples.kernel_agent.kernel_reward.reward_post_process_by_group",
        ]
    )
    assert args.rollout_reward_post_processors == processors
    assert args.dynamic_reward_gate == "sqrt"
    assert args.difficulty_thresholds == [0.25, 0.75]
    assert args.dynamic_reward_gate_range == [0.8, 1.2]
    _validate_rollout_reward_post_process_args(args)


def test_piecewise_dynamic_gate_args_parse_and_validate():
    import argparse

    from slime.utils.arguments import (
        _validate_difficulty_thresholds_args,
        _validate_rollout_reward_post_process_args,
        get_slime_extra_args_provider,
    )

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--dynamic-reward-gate",
            "piecewise",
            "--difficulty-thresholds",
            "0.2",
            "0.8",
            "--dynamic-reward-gate-range",
            "0.4",
            "2.0",
        ]
    )
    _validate_difficulty_thresholds_args(args)
    _validate_rollout_reward_post_process_args(args)
    assert args.difficulty_thresholds == [0.2, 0.8]
    assert args.dynamic_reward_gate_range == [0.4, 2.0]
    assert args.rollout_reward_post_processors is None  # Gate selection does not enable shaping.
    assert _compute_dynamic_auxiliary_gate(1, 10, args=args) == pytest.approx(0.7)
    assert _compute_dynamic_auxiliary_gate(9, 10, args=args) == pytest.approx(1.5)


@pytest.mark.parametrize(
    "gate_range",
    [
        [-0.1, 1.2],
        [1.1, 1.2],
        [0.8, 0.9],
        [float("nan"), 1.2],
        [0.8, float("inf")],
        [1.2, 0.8],
        [],
        [0.8],
        [0.8, 1.0, 1.2],
    ],
)
def test_piecewise_dynamic_gate_args_reject_invalid_values(gate_range):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    args = SimpleNamespace(
        dynamic_reward_gate="piecewise", rollout_reward_post_processors=None, dynamic_reward_gate_range=gate_range
    )
    with pytest.raises(ValueError, match="Piecewise"):
        _validate_rollout_reward_post_process_args(args)


@pytest.mark.parametrize(
    "thresholds",
    [
        [],
        [0.0, 0.75],
        [-0.1, 0.75],
        [0.75, 0.25],
        [0.25, 0.25],
        [0.25, 1.0],
        [0.25, 1.1],
        [float("nan"), 0.75],
        [0.25, float("inf")],
    ],
)
@pytest.mark.parametrize("mode", ["sqrt", "piecewise"])
def test_difficulty_thresholds_validation_is_independent_of_reward_mode(thresholds, mode):
    from slime.utils.arguments import _validate_difficulty_thresholds_args

    with pytest.raises(ValueError, match="--difficulty-thresholds"):
        _validate_difficulty_thresholds_args(
            SimpleNamespace(difficulty_thresholds=thresholds, dynamic_reward_gate=mode)
        )


@pytest.mark.parametrize("thresholds", [[0.5], [0.1, 0.4, 0.8]])
def test_shared_difficulty_thresholds_allow_more_buckets_but_piecewise_requires_two(thresholds):
    import argparse

    from slime.utils.arguments import (
        _validate_difficulty_thresholds_args,
        _validate_rollout_reward_post_process_args,
        get_slime_extra_args_provider,
    )

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1", "--difficulty-thresholds", *map(str, thresholds)])
    assert args.difficulty_thresholds == thresholds
    _validate_difficulty_thresholds_args(args)
    _validate_rollout_reward_post_process_args(args)
    args.dynamic_reward_gate = "piecewise"
    with pytest.raises(ValueError, match="exactly two --difficulty-thresholds"):
        _validate_rollout_reward_post_process_args(args)


@pytest.mark.parametrize(
    "processors,buffer_len,factor",
    [
        (["none", "dynamic-weight"], 100, 1.0),
        (["dynamic-weight", "dynamic-weight"], 100, 1.0),
        (["overlong-penalty"], 0, 1.0),
        (["overlong-penalty"], 100, -1.0),
        (["overlong-penalty"], 100, float("nan")),
        (["overlong-penalty"], 100, float("inf")),
    ],
)
def test_rollout_reward_processor_args_reject_invalid_values(processors, buffer_len, factor):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    with pytest.raises(ValueError):
        _validate_rollout_reward_post_process_args(
            SimpleNamespace(
                rollout_reward_post_processors=processors,
                overlong_buffer_len=buffer_len,
                overlong_penalty_factor=factor,
            )
        )


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
    for sample in samples:
        assert sample.metadata["group_num_correct"] == 0
        assert sample.metadata["group_num_valid"] == 3
        assert sample.metadata["group_correct_rate"] == 0.0
        assert sample.metadata["group_difficulty"] == 1.0


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


def test_ctm_annotates_without_changing_group_advantages():
    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True)
    _enable_ctm(manager.args)
    samples = [
        _make_ctm_candidate(0, 1.0, correctness=False, status=Sample.Status.COMPLETED),
        _make_ctm_candidate(1, 2.0, correctness=False, status=Sample.Status.TRUNCATED),
        _make_ctm_candidate(2, 4.0, correctness=False, status=Sample.Status.COMPLETED),
    ]

    raw_rewards, rewards = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == [1.0, 2.0, 4.0]
    assert rewards == pytest.approx([-4.0 / 3.0, -1.0 / 3.0, 5.0 / 3.0])
    assert samples[1].reward == 2.0
    assert samples[1].remove_sample is False
    assert samples[1].metadata["conditional_truncation_masked"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_ctm_flags_reach_train_data_without_masking_rewards(enabled):
    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True)
    _enable_ctm(manager.args)
    manager.args.use_conditional_truncation_mask = enabled
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = reward_post_process_by_group
    samples = [
        _make_ctm_candidate(i, float(i + 1), correctness=False, status=Sample.Status.TRUNCATED) for i in range(3)
    ]
    # Inherited annotations must not mask removed/padded or newly ineligible samples.
    for sample in samples:
        sample.metadata["conditional_truncation_masked"] = True
    samples[1].metadata["env_result"] = {"error": "RUNTIME_ERROR"}
    samples[2].remove_sample = True

    data = manager._convert_samples_to_train_data(samples)

    assert data["rewards"][:2] == pytest.approx([-0.5, 0.5])
    assert data["loss_masks"] == [[1] * 4, [1] * 4, [0] * 4]
    if enabled:
        assert data["conditional_truncation_masked"] == [True, False, False]
        assert "conditional_truncation_masked" not in samples[1].metadata
    else:
        assert "conditional_truncation_masked" not in data


def test_ctm_flags_follow_dp_partition_order(monkeypatch):
    import slime.ray.rollout as rollout_module

    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True)
    manager.args.global_batch_size = 4
    manager.train_parallel_config = {"dp_size": 2}
    monkeypatch.setattr(
        rollout_module, "build_dp_schedule", lambda *a, **kw: ([[2, 0], [3, 1]], [[[0, 1]], [[0, 1]]], [1], [4])
    )
    monkeypatch.setattr(rollout_module.ray, "put", lambda data: data)
    data = {
        "tokens": [[0, 1]] * 4,
        "response_lengths": [1] * 4,
        "loss_masks": [[1]] * 4,
        "rewards": [1.0, 2.0, 3.0, 4.0],
        "rollout_ids": [0, 1, 2, 3],
        "conditional_truncation_masked": [True, False, False, True],
    }
    shards = [box.inner for box in manager._split_train_data_by_dp(data)]
    assert shards[0]["rollout_ids"] == [2, 0]
    assert shards[0]["conditional_truncation_masked"] == [False, True]
    assert shards[1]["rollout_ids"] == [3, 1]
    assert shards[1]["conditional_truncation_masked"] == [True, False]


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
    "env_state",
    [
        {"error": "RUNTIME_ERROR"},
        {"error_code": "RUNTIME_ERROR"},
        {"error": "CORRECTNESS_ERROR"},
        {"error": "KERNEL_EVAL_FAILED", "error_code": "CORRECTNESS_ERROR"},
        {"metadata": {"runtime_error": "illegal memory access"}},
        {"metadata": {"correctness_runtime_error": "launch failed"}},
        {"metadata": {"correctness_output_mismatch": True}},
        {"compiled": True, "correctness": False, "status": "completed"},
        {"compiled": True, "status": "timeout", "error": "KERNEL_EVAL_TIMEOUT"},
        {"compiled": True, "correctness": True, "status": "failed", "error": "KERNEL_EVAL_FAILED"},
        {"compiled": True, "correctness": True, "error_message": "reference timing failed"},
        {"compiled": True, "correctness": True, "metadata": {"error": "profiling failed"}},
        {"metadata": {"performance_error": "timing failed"}},
        {"metadata": {"profiling_error": "profiler failed"}},
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_ctm_keeps_runtime_and_later_errors_even_when_truncated(env_state, wrapped):
    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True)
    _enable_ctm(manager.args)  # rho=1: eligibility alone decides masking.
    candidate = _make_ctm_candidate(
        0, 1.0, correctness=env_state.get("correctness", False), status=Sample.Status.TRUNCATED
    )
    candidate.metadata["env_result"] = {"env_state": env_state} if wrapped else env_state
    other = _make_ctm_candidate(1, 3.0, correctness=False, status=Sample.Status.COMPLETED)

    raw_rewards, advantages = reward_post_process_by_group(manager.args, [candidate, other])

    assert raw_rewards == [1.0, 3.0]
    assert advantages == pytest.approx([-1.0, 1.0])
    assert not candidate.metadata.get("conditional_truncation_masking_eligible", False)
    assert not candidate.metadata.get("conditional_truncation_masked", False)


@pytest.mark.parametrize(
    "feedback",
    [
        {"compilation": True, "correctness": False},
        {"correctness_output_mismatch": True},
        {"correctness_runtime_error": "launch failed"},
    ],
)
def test_ctm_keeps_execution_errors_in_compact_feedback(feedback):
    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True)
    _enable_ctm(manager.args)
    candidate = _make_ctm_candidate(0, 1.0, correctness=False, status=Sample.Status.TRUNCATED)
    candidate.metadata["env_extra_info"].update(feedback)
    other = _make_ctm_candidate(1, 3.0, correctness=False, status=Sample.Status.COMPLETED)
    assert reward_post_process_by_group(manager.args, [candidate, other])[1] == pytest.approx([-1.0, 1.0])
    assert not candidate.metadata.get("conditional_truncation_masked", False)


@pytest.mark.parametrize(
    "env_state,correctness",
    [
        ({"compiled": False, "error": "COMPILATION_ERROR", "status": "failed"}, False),
        ({"compiled": None, "error": "PRECHECK_ERROR", "status": "failed"}, False),
        ({"compiled": False, "error": "SYNTAX_ERROR", "status": "failed"}, False),
        ({"compiled": None, "error": "KERNEL_EVAL_TIMEOUT", "status": "timeout"}, False),
        ({"compiled": True, "correctness": True, "status": "completed"}, True),
    ],
)
def test_ctm_still_masks_truncated_pre_execution_failures_and_success(env_state, correctness):
    manager = _make_manager(advantage_estimator="grpo", use_multi_turn=True)
    _enable_ctm(manager.args)
    candidate = _make_ctm_candidate(0, 1.0, correctness=correctness, status=Sample.Status.TRUNCATED)
    candidate.metadata["env_result"] = {"env_state": env_state}
    other = _make_ctm_candidate(1, 3.0, correctness=False, status=Sample.Status.COMPLETED)
    assert reward_post_process_by_group(manager.args, [candidate, other])[1] == pytest.approx([-1.0, 1.0])
    assert candidate.metadata["conditional_truncation_masked"] is True


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
