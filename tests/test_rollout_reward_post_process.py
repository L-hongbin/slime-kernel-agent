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
    _compute_dynamic_coverage_scale,
    _compute_dynamic_reward_offset,
    annotate_group_difficulty,
    calculate_kernel_reward,
    post_process_rollout_rewards,
    reward_post_process_by_group,
)
from examples.kernel_agent.utils import _set_multi_turn_rewards

from slime.observability.rollout_metrics import (
    _compute_group_correct_rate_metrics,
    compute_reward_post_process_metrics,
)
from slime.ray.rollout import RolloutManager
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.mark.parametrize("mode", ["correct-rate", "correct-rate-sqrt", "piecewise", "piecewise-sqrt"])
@pytest.mark.parametrize("amplitude", [0.0, 0.1, 0.2, 0.5])
def test_dynamic_offset_group16_bounds_and_monotonicity(mode, amplitude):
    args = SimpleNamespace(dynamic_reward_amplitude=amplitude)
    offsets = [_compute_dynamic_reward_offset(k, 16, mode, args=args) for k in range(17)]
    assert offsets[0] == pytest.approx(-amplitude)
    assert offsets[-1] == pytest.approx(amplitude)
    assert offsets == sorted(offsets)
    for offset in offsets:
        assert -amplitude <= offset <= amplitude
        assert (0.5 - offset) + (0.5 + offset) == pytest.approx(1.0)
    if mode.startswith("piecewise"):
        assert offsets[6:11] == [0.0] * 5
        assert offsets == pytest.approx([-offset for offset in reversed(offsets)])
        assert offsets[4] == pytest.approx(-amplitude * (0.5 if mode == "piecewise-sqrt" else 0.25))
    elif mode == "correct-rate":
        assert offsets[8] == 0.0
    else:
        assert offsets[4] == 0.0


@pytest.mark.parametrize("amplitude", [-0.1, 0.51, float("nan"), float("inf")])
def test_dynamic_offset_rejects_invalid_amplitude(amplitude):
    with pytest.raises(ValueError, match="--dynamic-reward-amplitude"):
        _compute_dynamic_reward_offset(1, 16, "piecewise", args=SimpleNamespace(dynamic_reward_amplitude=amplitude))


def test_global_piecewise_uses_linear_curve_and_inclusive_custom_thresholds():
    args = SimpleNamespace(dynamic_reward_amplitude=0.1, difficulty_thresholds=[0.25, 0.75])
    expected = [-0.1, -0.0875, -0.075, -0.0625, *([0.0] * 9), 0.0625, 0.075, 0.0875, 0.1]
    actual = [_compute_dynamic_reward_offset(k, 16, "global-piecewise", args=args) for k in range(17)]
    assert actual == pytest.approx(expected)
    for correct, size in ((0, 0), (0, 1), (1, 1)):
        assert _compute_dynamic_reward_offset(correct, size, "global-piecewise", args=args) == 0.0


@pytest.mark.parametrize("mode,power", [("global-piecewise", 1.0), ("global-piecewise-sqrt", 0.5)])
def test_global_piecewise_uses_rate_curve_without_threshold_rescaling(mode, power):
    args = SimpleNamespace(dynamic_reward_amplitude=0.1)
    for correct in range(17):
        p = correct / 16
        expected = 0.0 if 6 <= correct <= 10 else 0.1 * (2 * p**power - 1)
        assert _compute_dynamic_reward_offset(correct, 16, mode, args=args) == pytest.approx(expected)
    # Global sqrt crosses zero at p=0.25, even below the hard threshold.
    if mode == "global-piecewise-sqrt":
        assert _compute_dynamic_reward_offset(4, 16, mode, args=args) == 0.0
        assert _compute_dynamic_reward_offset(5, 16, mode, args=args) > 0.0
    args.difficulty_thresholds = [0.25, 0.75]
    for correct in (4, 12):
        assert _compute_dynamic_reward_offset(correct, 16, mode, args=args) == 0.0


@pytest.mark.parametrize("component", ["correctness", "speedup"])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("curve,power", [("global-piecewise", 1.0), ("global-piecewise-sqrt", 0.5)])
def test_global_piecewise_cli_adjusts_only_selected_component(monkeypatch, component, inverse, curve, power):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    for key in ("init_correct_weight", "init_performance_weight", "coverage_reward_weight"):
        monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], key, 0.5)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", False)
    mode = f"inverse-{curve}" if inverse else curve
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1", f"--dynamic-reward-{component}", mode])
    _validate_rollout_reward_post_process_args(args)
    assert args.dynamic_reward_range is None
    samples = [_make_sample(i, 0, float(i < 5)) for i in range(16)]
    for i, sample in enumerate(samples):
        _set_reward_component(sample, correctness_score=float(i < 5), performance_score=float(i < 5))
    offset = 0.1 * (2 * (5 / 16) ** power - 1)
    expected_weight = 0.5 - offset if inverse else 0.5 + offset
    expected = [0.5 + expected_weight] * 5 + [0.0] * 11
    for _ in range(2):
        assert post_process_rollout_rewards(args, samples) == pytest.approx(expected)
        record = samples[0].metadata["dynamic_reward"]
        assert record[f"{component}_weight"] == pytest.approx(expected_weight)
        other = "speedup" if component == "correctness" else "correctness"
        assert f"{other}_weight" not in record
        metrics = compute_reward_post_process_metrics(samples)
        assert metrics[f"rollout/dynamic_reward/{component}_offset_mean"] == pytest.approx(expected_weight - 0.5)


@pytest.mark.parametrize(
    "mode", ["global-piecewise", "global-piecewise-sqrt", "inverse-global-piecewise", "inverse-global-piecewise-sqrt"]
)
def test_global_piecewise_is_not_a_coverage_mode(mode):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args(["--rollout-batch-size", "1", "--dynamic-reward-coverage", mode])
    with pytest.raises(ValueError, match="--dynamic-reward-coverage"):
        _validate_rollout_reward_post_process_args(SimpleNamespace(dynamic_reward_coverage=mode))
    with pytest.raises(ValueError, match="Unknown dynamic reward mode"):
        _compute_dynamic_coverage_scale(1, 16, mode)


@pytest.mark.parametrize("component", ["correctness", "speedup"])
@pytest.mark.parametrize(
    "mode", ["global-piecewise", "inverse-global-piecewise", "global-piecewise-sqrt", "inverse-global-piecewise-sqrt"]
)
@pytest.mark.parametrize("thresholds", [[0.5], [0.1, 0.4, 0.8], [0.7, 0.3], [0.0, 0.7], [0.3, float("nan")]])
def test_global_piecewise_requires_two_valid_thresholds(component, mode, thresholds):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    args = SimpleNamespace(difficulty_thresholds=thresholds, **{f"dynamic_reward_{component}": mode})
    with pytest.raises(ValueError, match="--difficulty-thresholds"):
        _validate_rollout_reward_post_process_args(args)


@pytest.mark.parametrize(
    "mode",
    ["correct-rate", "correct-rate-sqrt", "piecewise", "piecewise-sqrt", "global-piecewise", "global-piecewise-sqrt"],
)
@pytest.mark.parametrize("correct", range(17))
@pytest.mark.parametrize("base_correctness,base_speedup", [(0.5, 0.5), (0.4, 0.6)])
def test_matched_modes_transfer_weight_without_changing_total(
    monkeypatch, mode, correct, base_correctness, base_speedup
):
    config = CUDA_AGENT_CONFIGS["reward"]
    monkeypatch.setitem(config, "init_correct_weight", base_correctness)
    monkeypatch.setitem(config, "init_performance_weight", base_speedup)
    monkeypatch.setitem(config, "performance_reward_requires_correctness", False)
    monkeypatch.setitem(config, "apply_failed_group_reward", False)
    args = SimpleNamespace(
        dynamic_reward_correctness=f"inverse-{mode}",
        dynamic_reward_speedup=mode,
        dynamic_reward_amplitude=0.2,
    )
    samples = [_make_sample(i, 0, 0.0) for i in range(16)]
    for i, sample in enumerate(samples):
        _set_reward_component(sample, correctness_score=float(i < correct), performance_score=0.8)
    first = post_process_rollout_rewards(args, samples)
    assert post_process_rollout_rewards(args, samples) == pytest.approx(first)
    for i, sample in enumerate(samples):
        record = sample.metadata["dynamic_reward"]
        wc, ws = record["correctness_weight"], record["speedup_weight"]
        assert wc + ws == pytest.approx(base_correctness + base_speedup)
        assert record["correctness_offset"] == -record["speedup_offset"]
        assert base_correctness - 0.2 - 1e-12 <= wc <= base_correctness + 0.2 + 1e-12
        assert base_speedup - 0.2 - 1e-12 <= ws <= base_speedup + 0.2 + 1e-12
        assert sample.reward == pytest.approx(wc * (i < correct) + ws * 0.8)
    metrics = compute_reward_post_process_metrics(samples)
    assert metrics["rollout/dynamic_reward/correctness_weight_mean"] == pytest.approx(wc)
    assert metrics["rollout/dynamic_reward/speedup_weight_mean"] == pytest.approx(ws)


@pytest.mark.parametrize("component", ["correctness", "speedup"])
@pytest.mark.parametrize("base", [0.0, 0.05, -0.1, float("nan"), float("inf")])
def test_additive_weight_rejects_unsafe_base_weight(component, base):
    config = {**CUDA_AGENT_CONFIGS["reward"]}
    config["init_correct_weight" if component == "correctness" else "init_performance_weight"] = base
    args = SimpleNamespace(**{f"dynamic_reward_{component}": "piecewise", "dynamic_reward_amplitude": 0.1})
    with pytest.raises(ValueError, match="base weight"):
        _apply_dynamic_group_reward_weights([], [], config, args=args)


@pytest.mark.parametrize("component", ["correctness", "speedup"])
def test_zero_amplitude_allows_zero_base_weight(component):
    config = {**CUDA_AGENT_CONFIGS["reward"], "init_correct_weight": 0.0, "init_performance_weight": 0.0}
    args = SimpleNamespace(**{f"dynamic_reward_{component}": "inverse-correct-rate", "dynamic_reward_amplitude": 0.0})
    sample = _make_sample(0, 0, 0.0)
    _set_reward_component(sample)
    assert _apply_dynamic_group_reward_weights([sample], [0.0], config, args=args, record_metrics=True) == [0.0]
    assert sample.metadata["dynamic_reward"][f"{component}_weight"] == 0.0


@pytest.mark.parametrize("value", ["-0.1", "0.51", "nan", "inf"])
def test_amplitude_cli_rejects_invalid_values(value):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1", "--dynamic-reward-amplitude", value])
    with pytest.raises(ValueError, match="--dynamic-reward-amplitude"):
        _validate_rollout_reward_post_process_args(args)


@pytest.mark.parametrize("component", ["correctness", "speedup"])
@pytest.mark.parametrize("mode", ["piecewise", "piecewise-sqrt", "inverse-piecewise", "inverse-piecewise-sqrt"])
def test_additive_piecewise_needs_no_range_and_preserves_singleton(component, mode):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1", f"--dynamic-reward-{component}", mode])
    _validate_rollout_reward_post_process_args(args)
    assert args.dynamic_reward_amplitude == 0.1
    assert args.dynamic_reward_range is None
    for correct in (0, 1):
        assert _compute_dynamic_reward_offset(correct, 1, mode.removeprefix("inverse-"), args=args) == 0.0


_DYNAMIC_MODES_AND_QUARTER_SCALES = [
    (None, 1.0),
    ("correct-rate", 0.25),
    ("correct-rate-sqrt", 0.5),
    ("piecewise", 0.95),
    ("piecewise-sqrt", 0.9),
    ("inverse-correct-rate", 4.0),
    ("inverse-correct-rate-sqrt", 2.0),
    ("inverse-piecewise", 1 / 0.95),
    ("inverse-piecewise-sqrt", 1 / 0.9),
]


_ADDITIVE_QUARTER_SCALES = {
    None: 1.0,
    "correct-rate": 0.9,
    "correct-rate-sqrt": 1.0,
    "piecewise": 0.95,
    "piecewise-sqrt": 0.9,
    "inverse-correct-rate": 1.1,
    "inverse-correct-rate-sqrt": 1.0,
    "inverse-piecewise": 1.05,
    "inverse-piecewise-sqrt": 1.1,
}


@pytest.mark.parametrize("component", ["correctness", "speedup", "coverage"])
@pytest.mark.parametrize("mode,scale", _DYNAMIC_MODES_AND_QUARTER_SCALES)
@pytest.mark.parametrize("reward_key", [None, "score"])
def test_dynamic_components_are_independent_and_idempotent(monkeypatch, component, mode, scale, reward_key):
    if component != "coverage":
        scale = _ADDITIVE_QUARTER_SCALES[mode]
    config = CUDA_AGENT_CONFIGS["reward"]
    for key, value in {
        "init_correct_weight": 0.5,
        "init_performance_weight": 0.5,
        "coverage_reward_weight": 0.5,
        "coverage_reward_enable": True,
        "apply_failed_group_reward": False,
    }.items():
        monkeypatch.setitem(config, key, value)
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.dynamic_reward_range = [0.8, 1.2]
    args.reward_key = reward_key
    setattr(args, f"dynamic_reward_{component}", mode)
    samples = []
    for i in range(4):
        base = 1.2 if i == 0 else -0.7
        sample = _make_sample(i, 0, base - 0.1)
        _set_reward_component(
            sample,
            correctness_score=float(i == 0),
            performance_score=0.8 if i == 0 else 0.0,
            coverage_score=0.6 if i == 0 else 0.0,
            failed=None if i == 0 else -0.7,
            length_score=-0.1,
        )
        sample.metadata.update(raw_task_reward=base, task_reward=base, return_reward=7.0)
        sample.metadata["reward_component"]["return_reward"] = 7.0
        if reward_key:
            sample.reward = {"score": sample.reward, "other": 123}
        samples.append(sample)
    bases = {"correctness": 0.5, "speedup": 0.4, "coverage": 0.3}
    expected = [1.1 + bases[component] * (scale - 1), -0.8, -0.8, -0.8]
    for _ in range(2):
        assert post_process_rollout_rewards(args, samples) == pytest.approx(expected)
        for name, key in (("correctness", "correctness"), ("speedup", "performance"), ("coverage", "coverage")):
            assert samples[0].metadata["reward_component"][key] == pytest.approx(
                bases[name] * (scale if name == component else 1)
            )
        for i, sample in enumerate(samples):
            assert sample.get_reward_value(args) == pytest.approx(expected[i])
            assert sample.metadata["raw_task_reward"] == (1.2 if i == 0 else -0.7)
            assert sample.metadata["reward_component"]["return_reward"] == sample.metadata["return_reward"] == 7.0
            assert sample.metadata["reward_component"]["length"] == -0.1
            if reward_key:
                assert sample.reward["other"] == 123
        metrics = compute_reward_post_process_metrics(samples)
        if mode is None:
            assert metrics == {}
        else:
            prefix = f"rollout/dynamic_reward/{component}"
            assert metrics[f"{prefix}_scale_mean"] == pytest.approx(scale)
            assert metrics[f"{prefix}_reward_delta_mean"] == pytest.approx(bases[component] * (scale - 1) / 4)
            expected_fields = {f"{component}_scale", f"{component}_reward_delta"}
            if component != "coverage":
                expected_fields.update({f"{component}_weight", f"{component}_offset"})
                assert metrics[f"{prefix}_weight_mean"] == pytest.approx(0.5 * scale)
                assert metrics[f"{prefix}_offset_mean"] == pytest.approx(0.5 * (scale - 1))
            assert set(samples[0].metadata["dynamic_reward"]) == expected_fields


def test_three_dynamic_modes_compose_without_sharing_scales():
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.dynamic_reward_correctness = "inverse-correct-rate"
    args.dynamic_reward_speedup = "correct-rate-sqrt"
    args.dynamic_reward_coverage = "piecewise-sqrt"
    args.dynamic_reward_range = [0.8, 1.2]
    samples = [_make_sample(i, 0, 1.5 if i == 0 else 0.0) for i in range(4)]
    for i, sample in enumerate(samples):
        _set_reward_component(
            sample, correctness_score=float(i == 0), performance_score=float(i == 0), coverage_score=float(i == 0)
        )
    assert post_process_rollout_rewards(args, samples) == pytest.approx([0.55 + 0.5 + 0.45, 0, 0, 0])
    record = samples[0].metadata["dynamic_reward"]
    assert record["correctness_scale"] == pytest.approx(1.1)
    assert record["speedup_scale"] == 1.0
    assert record["coverage_scale"] == pytest.approx(0.9)


@pytest.mark.parametrize("component", ["correctness", "speedup", "coverage"])
@pytest.mark.parametrize("mode,scale", _DYNAMIC_MODES_AND_QUARTER_SCALES)
def test_dynamic_component_cli_choices(component, mode, scale):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    for selection in (
        [[], [f"--dynamic-reward-{component}", "None"]] if mode is None else [[f"--dynamic-reward-{component}", mode]]
    ):
        args = parser.parse_args(["--rollout-batch-size", "1", "--dynamic-reward-range", "0.8", "1.2", *selection])
        _validate_rollout_reward_post_process_args(args)
        assert not hasattr(args, "dynamic_reward_gate")
        for name in ("correctness", "speedup", "coverage"):
            assert getattr(args, f"dynamic_reward_{name}") == (mode if name == component else None)
        if component == "coverage":
            assert _compute_dynamic_coverage_scale(1, 4, mode, args=args) == pytest.approx(scale)
        else:
            base_mode = mode.removeprefix("inverse-") if mode else None
            offset = _compute_dynamic_reward_offset(1, 4, base_mode, args=args)
            if mode and mode.startswith("inverse-"):
                offset = -offset
            assert 0.5 + offset == pytest.approx(0.5 * _ADDITIVE_QUARTER_SCALES[mode])


@pytest.mark.parametrize("mode,power", [("correct-rate", 1.0), ("correct-rate-sqrt", 0.5)])
@pytest.mark.parametrize("inverse", [False, True])
def test_rate_scales_include_singletons_and_zero_without_range(mode, power, inverse):
    selected = f"inverse-{mode}" if inverse else mode
    for correct, size in ((0, 0), (0, 1), (1, 1), (0, 16), (1, 16), (4, 16), (16, 16)):
        expected = (correct / size) ** power if size else 0.0
        if inverse and expected > 0:
            expected = 1 / expected
        assert _compute_dynamic_coverage_scale(correct, size, selected) == pytest.approx(expected)


@pytest.mark.parametrize("component", ["coverage"])
@pytest.mark.parametrize("mode", ["inverse-piecewise", "inverse-piecewise-sqrt"])
@pytest.mark.parametrize("lower", [0.0, 1e-320])
def test_inverse_piecewise_rejects_nonfinite_reciprocal(component, mode, lower):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    args = SimpleNamespace(dynamic_reward_range=[lower, 1.2], **{f"dynamic_reward_{component}": mode})
    with pytest.raises(ValueError, match="finite inverse"):
        _validate_rollout_reward_post_process_args(args)
    with pytest.raises(ValueError, match="finite inverse"):
        _compute_dynamic_coverage_scale(0, 4, mode, args=args)


@pytest.mark.parametrize("mode", ["inverse-piecewise", "inverse-piecewise-sqrt"])
def test_inverse_piecewise_preserves_tiny_positive_lower_bound(mode):
    args = SimpleNamespace(dynamic_reward_range=[1e-200, 1.2])
    assert _compute_dynamic_coverage_scale(0, 4, mode, args=args) == pytest.approx(1e200)


def _make_manager(*, advantage_estimator: str, use_multi_turn: bool, grpo_std_normalization: bool = False):
    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.custom_reward_post_process_func = None
    manager.rollout_id = 0
    manager.args = SimpleNamespace(
        dynamic_reward_range=[0.0, 1.0],
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


@pytest.mark.parametrize("custom_kernel_hook", [False, True])
@pytest.mark.parametrize("estimator", ["argmaxrl", "tailrl"])
def test_argmaxrl_aggregate_reward_grouping_turns_padding_and_no_mutation(custom_kernel_hook, estimator):
    manager = _make_manager(advantage_estimator=estimator, use_multi_turn=True, grpo_std_normalization=True)
    if custom_kernel_hook:
        manager.custom_reward_post_process_func = reward_post_process_by_group
    # Intentionally interleaved prompt/turn groups; ignore configured fixed N=3.
    samples = [
        Sample(index=i, group_index=g, reward=r, loss_mask=[1], metadata={"turn_idx": t, "return_reward": 999})
        for i, g, t, r in [(0, 0, 0, 1), (2, 1, 0, 2), (0, 0, 1, 0), (1, 0, 0, 3), (3, 1, 0, 2), (1, 0, 1, 2)]
    ]
    samples += [
        Sample(index=9, group_index=0, reward=100, remove_sample=True, metadata={"turn_idx": 0}),
        Sample(index=10, group_index=0, reward=100, loss_mask=[0], metadata={"turn_idx": 0}),
    ]
    original = [s.reward for s in samples]
    raw, advantages = manager._post_process_rewards(samples)
    assert raw == original
    expected = [-2, 0, -2, 2, 0, 2, 0, 0] if estimator == "tailrl" else [1, 2, 0, 5, 2, 4, 0, 0]
    assert advantages == pytest.approx(expected)
    assert [s.reward for s in samples] == original
    assert all("raw_task_reward" not in s.metadata for s in samples)


@pytest.mark.parametrize("estimator", ["argmaxrl", "tailrl"])
def test_argmaxrl_custom_reward_output_is_used_once(estimator):
    manager = _make_manager(advantage_estimator=estimator, use_multi_turn=False)
    calls = []

    def custom(_args, samples):
        calls.append(len(samples))
        return [1.0, 2.0], [2.0, 4.0]

    manager.custom_reward_post_process_func = custom
    samples = [Sample(index=i, group_index=0, reward=r) for i, r in enumerate([1, 2])]
    raw, advantages = manager._post_process_rewards(samples)
    assert calls == [2]
    assert raw == [1, 2]
    assert advantages == pytest.approx([-2, 2] if estimator == "tailrl" else [2, 6])


def test_argmaxrl_kernel_offset_ctm_and_raw_reward_preserved():
    manager = _make_manager(advantage_estimator="argmaxrl", use_multi_turn=False)
    manager.args.argmaxrl_reward_offset = 1.0
    manager.args.use_conditional_truncation_mask = True
    manager.custom_reward_post_process_func = reward_post_process_by_group
    samples = [Sample(index=i, group_index=0, reward=r, metadata={}) for i, r in enumerate([-1.0, 1.0])]
    for sample in samples:
        sample.metadata["conditional_truncation_masked"] = True
    raw, advantages = manager._post_process_rewards(samples)
    assert raw == [-1, 1]
    assert advantages == [0, 4]
    assert [s.reward for s in samples] == [-1, 1]
    # Non-truncated samples are ineligible; calling the CTM annotator clears
    # stale flags rather than preserving a previous selection.
    assert all("conditional_truncation_masked" not in s.metadata for s in samples)


@pytest.mark.parametrize("estimator", ["argmaxrl", "tailrl"])
def test_argmaxrl_train_conversion_and_dp_shards_preserve_complete_group_weights(monkeypatch, estimator):
    import slime.ray.rollout as rollout_module

    manager = _make_manager(advantage_estimator=estimator, use_multi_turn=False)
    manager.custom_reward_post_process_func = reward_post_process_by_group
    manager.custom_convert_samples_to_train_data_func = None
    manager.args.global_batch_size = 4
    manager.args.micro_batch_size = 1
    manager.args.use_dynamic_batch_size = False
    manager.args.balance_data = False
    manager.args.balance_by_flops = False
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    samples = [
        Sample(index=i, rollout_id=i, group_index=i // 2, reward=r, tokens=[1, 2], response_length=1)
        for i, r in enumerate([1.0, 3.0, 2.0, 4.0])
    ]
    data = manager._convert_samples_to_train_data(samples)
    assert data["raw_reward"] == [1, 3, 2, 4]
    assert data["rewards"] == ([-2, 2, -2, 2] if estimator == "tailrl" else [1, 5, 2, 6])
    monkeypatch.setattr(rollout_module.ray, "put", lambda data: data)
    shards = [box.inner for box in manager._split_train_data_by_dp(data)]
    # Strided DP scheduling puts one candidate from each prompt on each rank.
    assert shards[0]["rewards"] == ([-2, -2] if estimator == "tailrl" else [1, 2])
    assert shards[1]["rewards"] == ([2, 2] if estimator == "tailrl" else [5, 6])


def _laser_args(**overrides):
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.overlong_penalty = "laser-d"
    args.rollout_max_response_len = 16
    args.laser_d_min_length = 4
    args.laser_d_length_interval = 4
    args.laser_d_update_interval = 2
    args.laser_d_monitor_groups = 2
    args.laser_d_length_score = 0.5
    args.difficulty_thresholds = [1 / 3, 2 / 3]
    args.rollout_global_dataset = True
    args.custom_reward_post_process_path = "examples.kernel_agent.kernel_reward.reward_post_process_by_group"
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _laser_group(correct=1, lengths=(8,) * 8, group_index=0):
    samples = []
    for index, length in enumerate(lengths):
        sample = _make_sample(index, group_index, float(index < correct))
        sample.status = Sample.Status.COMPLETED
        sample.response_length = length
        sample.tokens = [0] * length
        sample.metadata["task_reward"] = float(index < correct)
        _set_reward_component(
            sample, correctness_score=float(index < correct), performance_score=float(index < correct)
        )
        samples.append(sample)
    return samples


@pytest.mark.parametrize(
    "size,correct,bucket", [(8, 0, 0), (8, 2, 0), (8, 3, 1), (8, 5, 1), (8, 6, 2), (3, 1, 1), (3, 2, 2)]
)
def test_laser_d_difficulty_uses_consistent_inclusive_boundaries(size, correct, bucket):
    from examples.kernel_agent.length_reward import laser_d_bucket

    assert laser_d_bucket(correct, size, [1 / 3, 2 / 3]) == bucket


def test_laser_d_search_uses_paper_ecr_and_includes_non_grid_cap():
    from examples.kernel_agent.length_reward import select_laser_d_budgets

    records = [
        {"bucket": 0, "lengths": [2] * 7 + [9]},
        {"bucket": 1, "lengths": [2] * 3 + [9] * 5},
        {"bucket": 2, "lengths": [2] * 2 + [9] * 6},
    ]
    assert select_laser_d_budgets(records, [1 / 3, 2 / 3], 4, 16, 4) == [12, 4, 4]
    assert select_laser_d_budgets(records[:1], [1 / 3, 2 / 3], 4, 10, 4) == [10, 10, 10]
    assert select_laser_d_budgets([], [1 / 3, 2 / 3], 4, 16, 4) == [16, 16, 16]


@pytest.mark.parametrize("gate", [None, "correct-rate-sqrt", "piecewise-sqrt"])
def test_laser_d_step_bonus_is_idempotent_and_composes_with_gate(gate):
    from examples.kernel_agent.length_reward import LaserDBudgetController

    args = _laser_args(dynamic_reward_speedup=gate)
    controller = LaserDBudgetController(args, SimpleNamespace(metadata={}), 0)
    samples = _laser_group(correct=2, lengths=(4, 5, 3, 3, 3, 3, 3, 3))
    for sample in samples:
        sample.metadata["raw_task_reward"] = sample.reward
    controller.observe(samples)
    post_process_rollout_rewards(args, samples, stage="sample")
    assert [sample.reward for sample in samples] == [1.5, 1.0, 0, 0, 0, 0, 0, 0]
    auxiliary_gate = 0.9 if gate == "piecewise-sqrt" else 1.0
    for _ in range(2):
        rewards = post_process_rollout_rewards(args, samples)
        assert rewards[:2] == pytest.approx([1 + 0.5 * auxiliary_gate, 0.5 + 0.5 * auxiliary_gate])
        assert rewards[2:] == [0] * 6
        assert [sample.metadata["raw_task_reward"] for sample in samples] == [1.0, 1.0, 0, 0, 0, 0, 0, 0]
        assert samples[0].metadata["reward_component"]["length"] == 0.5
        for sample in samples:
            assert sample.metadata["length_score"] == sample.metadata["reward_component"]["length"]
            assert "overlong_penalty" not in sample.metadata
            assert "overlong_penalty" not in sample.metadata["reward_component"]
            assert "length_bonus" not in sample.metadata["reward_component"]
            assert "length_score" not in sample.metadata["reward_component"]
            assert sample.reward == pytest.approx(
                sum(value for value in sample.metadata["reward_component"].values() if value is not None)
            )
    metrics = compute_reward_post_process_metrics(samples)
    assert metrics["rollout/laser_d/length_score_mean"] == 0.5 / 8
    assert metrics["rollout/laser_d/bonus_fraction"] == 1 / 8


def test_laser_d_budget_snapshot_and_update_affect_only_future_rollouts():
    from examples.kernel_agent.length_reward import LaserDBudgetController

    args, source = _laser_args(), SimpleNamespace(metadata={})
    first = LaserDBudgetController(args, source, 0)
    old_samples = _laser_group()
    first.observe(old_samples)
    assert source.metadata == {}  # No half-completed rollout state is checkpointed.
    assert first.finish()["laser_d/hard/budget_next"] == 8
    assert old_samples[0].metadata["laser_d"]["budget"] == 4
    second = LaserDBudgetController(args, source, 1)
    new_samples = _laser_group(lengths=(4,) * 8)
    second.observe(new_samples)
    post_process_rollout_rewards(args, new_samples)
    assert new_samples[0].reward == 1.5
    assert second.finish()["laser_d/budget_updated"] == 0
    third = LaserDBudgetController(args, source, 2)
    third.observe(_laser_group(lengths=(4,) * 8))
    assert third.finish()["laser_d/hard/budget_next"] == 4
    assert post_process_rollout_rewards(args, old_samples)[0] == 1.0
    assert new_samples[0].metadata["laser_d"]["budget"] == 8


def test_laser_d_reservoir_and_budget_restore_through_dataset_checkpoint(tmp_path):
    import copy

    from examples.kernel_agent.length_reward import LaserDBudgetController

    from slime.rollout.data_source import RolloutDataSourceWithBuffer

    args = _laser_args(save=str(tmp_path), load=str(tmp_path), rollout_shuffle=False)
    source = object.__new__(RolloutDataSourceWithBuffer)
    source.args, source.metadata, source.dataset = args, {}, None
    source.sample_offset = source.epoch_id = source.sample_group_index = source.sample_index = 0
    controller = LaserDBudgetController(args, source, 0)
    controller.observe(_laser_group())
    controller.finish()
    controller = LaserDBudgetController(args, source, 1)
    for index in range(10):
        controller.observe(_laser_group(lengths=(index + 1,) * 8))
    controller.finish()
    assert len(source.metadata["laser_d"]["records"]) == 2
    assert source.metadata["laser_d"]["seen_groups"] == 10
    source.save(1)
    restored = object.__new__(RolloutDataSourceWithBuffer)
    restored.args, restored.metadata, restored.dataset = args, {}, None
    restored.load(1)
    assert restored.metadata == source.metadata
    before = copy.deepcopy(source.metadata)
    for data in (source, restored):
        continuation = LaserDBudgetController(args, data, 2)
        continuation.observe(_laser_group(lengths=(9,) * 8))
        continuation.finish()
    assert restored.metadata == source.metadata
    assert source.metadata != before
    with pytest.raises(ValueError, match="checkpoint configuration"):
        LaserDBudgetController(_laser_args(laser_d_min_length=5), source, 3)


def test_laser_d_requires_collector_snapshot_only_at_training_stage():
    args, samples = _laser_args(), _laser_group()
    assert post_process_rollout_rewards(args, samples, stage="sample")[0] == 1.0
    with pytest.raises(ValueError, match="budget snapshot"):
        post_process_rollout_rewards(args, samples)


def test_laser_d_trloo_keeps_bonus_local_to_current_turn():
    from examples.kernel_agent.length_reward import LaserDBudgetController

    args = _laser_args(advantage_estimator="trloo", use_multi_turn=True, multi_turn_gamma=0.5)
    turns = [_laser_group(correct=2, lengths=lengths) for lengths in ((4, 8), (8, 4))]
    controller = LaserDBudgetController(args, SimpleNamespace(metadata={}), 0)
    for turn_idx, group in enumerate(turns):
        for sample in group:
            sample.metadata.update(turn_idx=turn_idx, multi_turn_reward=100.0)
    for trajectory in zip(*turns, strict=True):
        _set_multi_turn_rewards(args, list(trajectory), "env_done")
    for group in turns:
        controller.observe(group)
        post_process_rollout_rewards(args, group)
    samples = turns[0] + turns[1]
    raw, advantages = reward_post_process_by_group(args, samples)
    assert raw == pytest.approx([2.0, 1.5, 1.0, 1.5])
    assert all(abs(value) > 0 for value in advantages)
    assert [sample.reward for sample in samples] == [1.5, 1.0, 1.0, 1.5]
    assert [sample.metadata["return_reward"] for sample in samples] == [0.5, 0.5, 0.0, 0.0]


def test_laser_d_dictionary_reward_preserves_unrelated_keys():
    from examples.kernel_agent.length_reward import LaserDBudgetController

    args = _laser_args(reward_key="score")
    samples = _laser_group(correct=2, lengths=(4, 8))
    for sample in samples:
        sample.reward = {"score": sample.reward, "other": 9.0}
    LaserDBudgetController(args, SimpleNamespace(metadata={}), 0).observe(samples)
    assert post_process_rollout_rewards(args, samples) == [1.5, 1.0]
    assert samples[0].reward == {"score": 1.5, "other": 9.0}
    assert samples[1].reward == {"score": 1.0, "other": 9.0}


def test_laser_d_variable_group_sizes_use_group_weighted_ecr():
    from examples.kernel_agent.length_reward import select_laser_d_budgets

    records = [
        {"bucket": 1, "lengths": [4, 4, 12, 12]},  # C_min=2; ECR(4)=1.
        {"bucket": 1, "lengths": [4, 4, 4, 12, 12, 12, 12, 12]},  # C_min=3; ECR(4)=1.125.
    ]
    assert select_laser_d_budgets(records, [1 / 3, 2 / 3], 4, 16, 4) == [16, 4, 16]


@pytest.mark.parametrize(
    "override",
    [
        {"laser_d_min_length": 0},
        {"laser_d_min_length": 17},
        {"laser_d_length_interval": 0},
        {"laser_d_update_interval": 0},
        {"laser_d_monitor_groups": 0},
        {"laser_d_length_score": -1},
        {"laser_d_length_score": float("nan")},
        {"laser_d_length_score": float("inf")},
        {"difficulty_thresholds": [0.5]},
        {"rollout_global_dataset": False},
        {"custom_reward_post_process_path": None},
        {"rollout_function_path": "custom.rollout"},
    ],
)
def test_laser_d_rejects_invalid_configuration(override):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    with pytest.raises(ValueError):
        _validate_rollout_reward_post_process_args(_laser_args(**override))


@pytest.mark.parametrize("length_score", [None, 0.25])
def test_laser_d_cli_routes_default_rollout_and_keeps_full_async(length_score):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--overlong-penalty",
            "laser-d",
            "--rollout-max-response-len",
            "4096",
            "--custom-reward-post-process-path",
            "examples.kernel_agent.kernel_reward.reward_post_process_by_group",
            *(["--laser-d-length-score", str(length_score)] if length_score is not None else []),
        ]
    )
    assert args.rollout_global_dataset is True
    _validate_rollout_reward_post_process_args(args)
    assert args.laser_d_length_score == (0.5 if length_score is None else length_score)
    assert args.laser_d_update_interval == 20
    assert args.rollout_function_path == "examples.kernel_agent.kernel_reward.generate_rollout"
    args.rollout_function_path = "examples.kernel_agent.fully_async_rollout.generate_rollout_fully_async"
    _validate_rollout_reward_post_process_args(args)
    assert args.rollout_function_path.endswith("generate_rollout_fully_async")


@pytest.mark.parametrize("gate", [None, "correct-rate-sqrt"])
@pytest.mark.parametrize("method", [None, "dapo"])
def test_kernel_reward_hook_routes_prefilter_settlement_even_without_length_shaping(gate, method):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    args = _laser_args(
        overlong_penalty=method,
        dynamic_reward_speedup=gate,
        dynamic_reward_coverage=gate,
        overlong_buffer_len=16,
        overlong_penalty_factor=1.0,
        rollout_function_path="slime.rollout.sglang_rollout.generate_rollout",
    )
    _validate_rollout_reward_post_process_args(args)
    assert args.rollout_function_path == "examples.kernel_agent.kernel_reward.generate_rollout"
    args.rollout_function_path = "examples.kernel_agent.fully_async_rollout.generate_rollout_fully_async"
    _validate_rollout_reward_post_process_args(args)
    assert args.rollout_function_path.endswith("generate_rollout_fully_async")


@pytest.mark.parametrize("method", [None, "dapo", "laser-d"])
def test_kernel_sglang_collector_settles_before_dynamic_filter(monkeypatch, method):
    import asyncio

    from examples.kernel_agent.kernel_reward import generate_rollout

    from slime.rollout import sglang_rollout
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    args = _laser_args(
        overlong_penalty=method,
        overlong_buffer_len=16,
        overlong_penalty_factor=1.0,
        rollout_batch_size=1,
        n_samples_per_prompt=8,
        over_sampling_batch_size=1,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        use_rollout_routing_replay=False,
        laser_d_monitor_groups=10,
    )
    supplied = [
        _laser_group(lengths=(12,) * 8, group_index=0),
        _laser_group(correct=8, lengths=(4,) * 8, group_index=1),
    ]
    source = SimpleNamespace(metadata={}, get_samples=lambda count: [supplied.pop(0)])

    class State:
        remaining_batch_size = 0

        def __init__(self):
            self.pendings = set()

        def submit_generate_tasks(self, groups):
            async def completed(group):
                return group

            self.remaining_batch_size += len(groups)
            self.pendings.update(asyncio.create_task(completed(group)) for group in groups)

        def reset(self):
            self.remaining_batch_size = 0

    state = State()
    visited = []

    def filter_group(fn, args, group):
        visited.append(group[0].group_index)
        first = group[0].group_index == 0
        if method == "laser-d":
            assert group[0].metadata["laser_d"]["budget"] == 4
            expected = 1.0 if first else 1.5
        elif method == "dapo":
            expected = 0.25 if first else 0.75
        else:
            expected = 1.0
        assert group[0].reward == expected
        return DynamicFilterOutput(keep=group[0].group_index == 1, reason="test")

    async def abort(*args):
        return []

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "call_dynamic_filter", filter_group)
    monkeypatch.setattr(sglang_rollout, "abort", abort)
    output = generate_rollout(args, 0, source)
    assert visited == [0, 1]
    assert output.samples[0][0].group_index == 1
    if method == "laser-d":
        assert output.metrics["laser_d/hard/budget_next"] == 12
        assert source.metadata["laser_d"]["budgets"] == [12, 16, 4]
    else:
        assert "laser_d" not in source.metadata


@pytest.mark.parametrize("method", [None, "dapo", "laser-d"])
def test_failed_group_replacement_excludes_length_penalty_and_writes_components(monkeypatch, method):
    from examples.kernel_agent.length_reward import LaserDBudgetController

    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    args = _laser_args(overlong_penalty=method, overlong_buffer_len=16, overlong_penalty_factor=1.0)
    samples = _laser_group(correct=0, lengths=(4, 8))
    for sample, stage_score in zip(samples, [-1.0, -0.75], strict=True):
        sample.metadata["kernel_failed_score"] = stage_score
    if method == "laser-d":
        LaserDBudgetController(args, SimpleNamespace(metadata={}), 0).observe(samples)
    expected = [-0.75, -0.875] if method == "dapo" else [-0.5, -0.375]
    for _ in range(2):
        assert post_process_rollout_rewards(args, samples) == expected
        assert [sample.reward for sample in samples] == expected
        for sample in samples:
            assert sum(v for v in sample.metadata["reward_component"].values() if v is not None) == sample.reward
    assert [sample.metadata["task_reward"] for sample in samples] == [-0.5, -0.375]


def test_failed_group_replacement_preserves_length_penalty_exactly_once(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "failed_score", -1.0)
    args = _laser_args(
        overlong_penalty="dapo",
        dynamic_reward_speedup="correct-rate-sqrt",
        overlong_buffer_len=16,
        overlong_penalty_factor=1.0,
    )
    samples = _laser_group(correct=0, lengths=(8, 8))
    for sample, stage_score in zip(samples, [-0.25, -0.75], strict=True):
        sample.reward = sample.metadata["task_reward"] = -0.5
        sample.metadata["reward_component"]["failed"] = -0.5
        sample.metadata["kernel_failed_score"] = stage_score
    for _ in range(2):
        assert post_process_rollout_rewards(args, samples) == [-0.625, -0.875]
        assert all(sample.metadata["reward_component"]["length"] == -0.5 for sample in samples)


def test_correct_samples_at_default_failure_reward_are_not_replaced(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    args = _laser_args(overlong_penalty="dapo", overlong_buffer_len=16, overlong_penalty_factor=1.0)
    samples = _laser_group(correct=2, lengths=(16, 16))
    for sample in samples:
        sample.metadata["kernel_failed_score"] = -1.0
    assert post_process_rollout_rewards(args, samples) == [0.0, 0.0]
    assert all(sample.metadata["task_reward"] == 1.0 for sample in samples)


@pytest.mark.parametrize(
    "base,spread,threshold,replace",
    [
        (0.0, 0.0, 0.0, True),
        (0.0, 0.0, 0.001, True),
        (0.0, 1e-8, 1.0, False),
        (0.25, 0.0, 1.0, False),
        (0.25, 0.0005, 0.001, False),
        (0.25, 0.004, 0.01, False),
    ],
)
def test_failed_group_requires_exact_default_score_independent_of_variance_threshold(
    monkeypatch, base, spread, threshold, replace
):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "failed_score", 0.0)
    args = _laser_args(overlong_penalty=None, reward_std_threshold=threshold)
    samples = _laser_group(correct=0, lengths=(4, 4))
    original = [base, base + spread]
    for sample, reward, score in zip(samples, original, [-1.0, -0.75], strict=True):
        sample.reward = sample.metadata["task_reward"] = reward
        sample.metadata["reward_component"]["failed"] = reward
        sample.metadata["kernel_failed_score"] = score
    expected = [-0.5, -0.375] if replace else original
    assert post_process_rollout_rewards(args, samples) == pytest.approx(expected)


def test_failed_group_does_not_use_length_penalty_to_hide_task_variance(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    args = _laser_args(overlong_penalty="dapo", overlong_buffer_len=16, overlong_penalty_factor=1.0)
    samples = _laser_group(correct=0, lengths=(4, 8))
    for sample, reward in zip(samples, [0.25, 0.5], strict=True):
        sample.reward = sample.metadata["task_reward"] = reward
        sample.metadata["reward_component"]["failed"] = reward
        sample.metadata["kernel_failed_score"] = -1.0
    assert post_process_rollout_rewards(args, samples) == [0.0, 0.0]
    assert [sample.metadata["task_reward"] for sample in samples] == [0.25, 0.5]


def test_uniform_failure_stages_still_fail_formal_filter(monkeypatch):
    from examples.kernel_agent.kernel_filter import filter_cuda_kernel_group

    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    args = _laser_args(
        overlong_penalty="dapo",
        overlong_buffer_len=16,
        overlong_penalty_factor=1.0,
        n_samples_per_prompt=2,
        min_group_size=2,
        target_group_size=2,
    )
    samples = _laser_group(correct=0, lengths=(4, 8))
    for sample in samples:
        sample.metadata["kernel_failed_score"] = -1.0
    assert post_process_rollout_rewards(args, samples) == [-0.75, -1.0]
    assert not filter_cuda_kernel_group(args, samples).keep


def test_training_does_not_reweight_a_filtered_subset():
    args = _laser_args(overlong_penalty=None, dynamic_reward_speedup="correct-rate-sqrt")
    samples = _laser_group(correct=3, lengths=(4, 4, 4, 4))
    post_process_rollout_rewards(args, samples)
    selected = [samples[0], samples[3]]
    expected = [sample.reward for sample in selected]
    raw, _ = reward_post_process_by_group(args, selected)
    assert raw == expected
    assert selected[0].metadata["dynamic_reward"]["speedup_scale"] == pytest.approx(1 + 0.2 * (2 * (3 / 4) ** 0.5 - 1))
    assert selected[0].metadata["reward_component"]["performance"] == pytest.approx(
        0.5 + 0.1 * (2 * (3 / 4) ** 0.5 - 1)
    )


def test_trloo_reads_settled_turn_rewards_even_without_shaping():
    args = _laser_args(overlong_penalty=None, advantage_estimator="trloo", use_multi_turn=True, multi_turn_gamma=0.5)
    samples = []
    for turn in range(2):
        group = _laser_group(correct=2, lengths=(4, 4))
        for sample in group:
            sample.metadata.update(turn_idx=turn, multi_turn_reward=999.0)
        samples.extend(group)
    for index in range(2):
        _set_multi_turn_rewards(args, [samples[index], samples[index + 2]], "env_done")
    raw, _ = reward_post_process_by_group(args, samples)
    assert raw == [1.5, 1.5, 1.0, 1.0]
    assert all(sample.reward == 1.0 for sample in samples)


def test_failed_group_is_settled_before_filter_and_future_return(monkeypatch):
    from examples.kernel_agent.kernel_filter import filter_cuda_kernel_group

    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    args = _laser_args(
        overlong_penalty=None,
        advantage_estimator="trloo",
        use_multi_turn=True,
        multi_turn_gamma=0.5,
        n_samples_per_prompt=2,
        min_group_size=2,
        target_group_size=2,
    )
    turns = [_laser_group(correct=correct, lengths=(4, 4)) for correct in (0, 2)]
    for turn, group in enumerate(turns):
        for sample, stage_score in zip(group, [-1.0, -0.75], strict=True):
            sample.metadata.update(turn_idx=turn, kernel_failed_score=stage_score)
    for trajectory in zip(*turns, strict=True):
        _set_multi_turn_rewards(args, list(trajectory), "env_done")
    for group in turns:
        post_process_rollout_rewards(args, group)
    assert filter_cuda_kernel_group(args, turns[0]).keep
    assert [sample.reward for sample in turns[0]] == [-0.5, -0.375]
    # A successful future turn must not suppress failure-stage replacement of turn 0.
    raw, _ = reward_post_process_by_group(args, turns[0] + turns[1])
    assert raw == [0.0, 0.125, 1.0, 1.0]
    assert [sample.reward for sample in turns[0]] == [-0.5, -0.375]


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_trloo_future_credit_excludes_all_group_shaping_and_survives_filtering(monkeypatch, gamma):
    from examples.kernel_agent.utils import postprocess_turn_samples

    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    args = _laser_args(
        advantage_estimator="trloo",
        use_multi_turn=True,
        multi_turn_gamma=gamma,
        overlong_penalty="dapo",
        overlong_buffer_len=16,
        overlong_penalty_factor=1.0,
        dynamic_reward_speedup="correct-rate-sqrt",
        dynamic_reward_coverage="correct-rate-sqrt",
        finalize_mode=None,
    )
    turns = [_laser_group(correct=correct, lengths=(length, length)) for correct, length in [(2, 4), (0, 8), (1, 12)]]
    for turn, group in enumerate(turns):
        for sample, stage_score in zip(group, [-1.0, -0.75], strict=True):
            sample.metadata.update(turn_idx=turn, kernel_failed_score=stage_score, raw_task_reward=sample.reward)
        # Real generation settles DAPO before completing the trajectory.
        post_process_rollout_rewards(args, group, stage="sample")
    for trajectory in zip(*turns, strict=True):
        postprocess_turn_samples(args, list(trajectory), "env_done")
    for group in turns:
        post_process_rollout_rewards(args, group)
    assert [sample.reward for sample in turns[1]] == [-1.0, -0.875]
    assert [sample.metadata["task_reward"] for sample in turns[1]] == [-0.5, -0.375]
    assert turns[2][0].metadata["task_reward"] == pytest.approx(1.0 + 0.1 * (2 * 0.5**0.5 - 1))
    all_samples = [sample for group in turns for sample in group]
    assert [sample.metadata["raw_task_reward"] for sample in all_samples] == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    for sample in all_samples:
        assert sample.metadata["reward_component"]["return_reward"] == sample.metadata["return_reward"]
    expected = [0.85 + gamma**2, 0.85, -1.0 + gamma, -0.875, 0.25 + 0.1 * (2 * 0.5**0.5 - 1), -0.75]
    assert reward_post_process_by_group(args, all_samples)[0] == pytest.approx(expected)
    assert [
        sum(value for value in sample.metadata["reward_component"].values() if value is not None)
        for sample in all_samples
    ] == pytest.approx(expected)
    # Future groups may be filtered out, but frozen raw task credit is retained.
    assert reward_post_process_by_group(args, turns[0])[0] == pytest.approx(expected[:2])
    assert [sample.reward for sample in turns[0]] == pytest.approx([0.85, 0.85])


def test_trloo_missing_raw_future_credit_cannot_silently_use_shaped_returns():
    args = _laser_args(advantage_estimator="trloo", use_multi_turn=True)
    sample = _laser_group(correct=1, lengths=(4,))[0]
    sample.metadata.update(turn_idx=0, multi_turn_reward=999.0)
    with pytest.raises(ValueError, match="return_reward"):
        reward_post_process_by_group(args, [sample])


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_trloo_future_credit_prefers_frozen_raw_task_reward(gamma):
    args = _laser_args(advantage_estimator="trloo", use_multi_turn=True, multi_turn_gamma=gamma)
    samples = [_make_sample(0, 0, reward, turn_idx=turn) for turn, reward in enumerate([0.2, -0.8, 0.3])]
    for sample, raw_task_reward in zip(samples, [1.0, 0.0, 2.0], strict=True):
        sample.metadata.update(raw_task_reward=raw_task_reward, task_reward=99.0)
    _set_multi_turn_rewards(args, samples, "env_done")
    assert [s.metadata["return_reward"] for s in samples] == pytest.approx([2 * gamma**2, 2 * gamma, 0.0])
    assert reward_post_process_by_group(args, samples)[0] == pytest.approx([0.2 + 2 * gamma**2, -0.8 + 2 * gamma, 0.3])
    assert [s.metadata["raw_task_reward"] for s in samples] == [1.0, 0.0, 2.0]


def test_raw_future_credit_excludes_removed_turns_and_supports_dictionary_reward():
    args = _laser_args(advantage_estimator="trloo", use_multi_turn=True, multi_turn_gamma=0.5, reward_key="score")
    samples = []
    for turn, (reward, task_reward) in enumerate([(0.8, 1.0), (99.0, 99.0), (0.6, 1.0)]):
        sample = _make_sample(0, 0, reward, turn_idx=turn)
        sample.reward = {"score": reward, "other": 9.0}
        sample.metadata["task_reward"] = task_reward
        samples.append(sample)
    samples[1].remove_sample = True
    _set_multi_turn_rewards(args, samples, "env_done")
    assert samples[0].metadata["return_reward"] == 0.25
    assert samples[0].metadata["reward_component"]["return_reward"] == 0.25
    assert samples[2].metadata["reward_component"]["return_reward"] == 0.0
    assert reward_post_process_by_group(args, [samples[0], samples[2]])[0] == [1.05, 0.6]
    assert samples[0].reward == {"score": 0.8, "other": 9.0}


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
    length_score: float = 0.0,
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
        "length": length_score,
    }
    sample.metadata["env_extra_info"] = {
        "correctness": correctness_score > 0.0,
        "decoy_kernel": False,
    }


@pytest.mark.parametrize(
    "num_correct,group_size,expected",
    [
        (0, 8, 0.8),
        (1, 8, 0.875),
        (2, 8, 0.95),
        (3, 8, 1.0),
        (4, 8, 1.0),
        (5, 8, 1.0),
        (6, 8, 1.05),
        (7, 8, 1.125),
        (8, 8, 1.2),
        (1, 3, 1.0),
        (2, 3, 1.0),
        (1, 2, 1.0),
        (0, 0, 1.0),
        (0, 1, 1.0),
        (1, 1, 1.0),
    ],
)
def test_piecewise_dynamic_gate_default_regions(num_correct, group_size, expected):
    args = SimpleNamespace(
        dynamic_reward_speedup="piecewise",
        dynamic_reward_coverage="piecewise",
        dynamic_reward_range=[0.8, 1.2],
    )
    assert _compute_dynamic_coverage_scale(
        num_correct, group_size, args=args, mode=getattr(args, "dynamic_reward_speedup", None)
    ) == pytest.approx(expected)


def test_piecewise_dynamic_gate_equal_thirds_cli_matches_explicit_range_for_group_16():
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
            "--dynamic-reward-speedup",
            "piecewise",
            "--dynamic-reward-range",
            "0.8",
            "1.2",
        ]
    )
    assert args.difficulty_thresholds == [1 / 3, 2 / 3]
    _validate_difficulty_thresholds_args(args)
    _validate_rollout_reward_post_process_args(args)
    expected = [0.8, 0.8375, 0.875, 0.9125, 0.95, 0.9875, *([1.0] * 5), 1.0125, 1.05, 1.0875, 1.125, 1.1625, 1.2]
    for config in (
        args,
        SimpleNamespace(
            dynamic_reward_speedup="piecewise",
            dynamic_reward_coverage="piecewise",
            dynamic_reward_range=[0.8, 1.2],
        ),
    ):
        gates = [
            _compute_dynamic_coverage_scale(
                correct, 16, args=config, mode=getattr(config, "dynamic_reward_speedup", None)
            )
            for correct in range(17)
        ]
        assert gates == pytest.approx(expected)


def test_piecewise_dynamic_gate_custom_bounds_are_continuous_and_monotonic():
    args = SimpleNamespace(
        dynamic_reward_speedup="piecewise",
        dynamic_reward_coverage="piecewise",
        dynamic_reward_range=[0.4, 2.0],
        difficulty_thresholds=[0.2, 0.8],
    )
    gates = [
        _compute_dynamic_coverage_scale(correct, 100, args=args, mode=getattr(args, "dynamic_reward_speedup", None))
        for correct in range(101)
    ]
    assert gates == sorted(gates)
    assert gates[0] == 0.4
    assert gates[10] == pytest.approx(0.7)
    assert gates[20:81] == [1.0] * 61
    assert gates[90] == pytest.approx(1.5)
    assert gates[100] == 2.0
    assert max(right - left for left, right in zip(gates, gates[1:], strict=False)) <= 0.05 + 1e-12


def test_piecewise_square_root_gate_for_group_16():
    args = SimpleNamespace(
        dynamic_reward_speedup="piecewise-sqrt",
        dynamic_reward_coverage="piecewise-sqrt",
        dynamic_reward_range=[0.8, 1.2],
    )
    expected = [0.8, 0.8197, 0.8419, 0.8677, 0.9, 0.95, *([1.0] * 5), 1.05, 1.1, 1.1323, 1.1581, 1.1803, 1.2]
    gates = [
        _compute_dynamic_coverage_scale(correct, 16, args=args, mode=getattr(args, "dynamic_reward_speedup", None))
        for correct in range(17)
    ]
    assert gates == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("language", ["en", "zh"])
def test_documented_group_16_gate_table_matches_implementation(language):
    document = (repo_root / "docs" / language / "get_started" / "usage.md").read_text(encoding="utf-8")
    global_heading = (
        "##### 全局线性曲线加中间不调整区间"
        if language == "zh"
        else "##### Global linear curve with a neutral interval"
    )
    global_section = document.split(global_heading, 1)[1]
    global_table = next(block for block in global_section.split("\n\n") if block.startswith("|"))
    assert len(global_table.splitlines()[2:]) == 17
    for correct, row in enumerate(global_table.splitlines()[2:]):
        count, *values = [cell.strip() for cell in row.strip("|").split("|")]
        assert int(count) == correct
        for mode, value in zip(("piecewise", "global-piecewise"), values, strict=True):
            offset = _compute_dynamic_reward_offset(correct, 16, mode)
            assert value == f"{0.5 - offset:.5f} / {0.5 + offset:.5f}"
    heading = "##### Group=16 的最终权重" if language == "zh" else "##### Final weights for Group=16"
    weight_section = document.split(heading, 1)[1]
    weight_table = next(block for block in weight_section.split("\n\n") if block.startswith("|"))
    assert len(weight_table.splitlines()[2:]) == 17
    for correct, row in enumerate(weight_table.splitlines()[2:]):
        count, rate, *values = [cell.strip() for cell in row.strip("|").split("|")]
        assert int(count) == correct
        assert float(rate) == correct / 16
        for mode, value in zip(("piecewise", "piecewise-sqrt"), values, strict=True):
            offset = _compute_dynamic_reward_offset(
                correct, 16, mode, args=SimpleNamespace(dynamic_reward_amplitude=0.2)
            )
            assert value == f"{0.5 - offset:.4f} / {0.5 + offset:.4f}"
    section = (
        document.split("##### G=16", 1)[1] if language == "zh" else document.split("##### Gate values for G=16", 1)[1]
    )
    table = next(block for block in section.split("\n\n") if block.startswith("|"))
    rows = table.splitlines()[2:]
    assert len(rows) == 17
    for correct, row in enumerate(rows):
        count, rate, *values = [cell.strip() for cell in row.strip("|").split("|")]
        assert int(count) == correct
        assert float(rate.rstrip("%")) == correct / 16 * 100
        assert len(values) == 3
        for mode, value in zip(("correct-rate-sqrt", "piecewise", "piecewise-sqrt"), values, strict=True):
            gate = _compute_dynamic_coverage_scale(
                correct,
                16,
                args=SimpleNamespace(
                    dynamic_reward_speedup=mode,
                    dynamic_reward_range=([0.0, 1.0] if mode == "correct-rate-sqrt" else [0.8, 1.2]),
                ),
                mode=getattr(
                    SimpleNamespace(
                        dynamic_reward_speedup=mode,
                        dynamic_reward_range=([0.0, 1.0] if mode == "correct-rate-sqrt" else [0.8, 1.2]),
                    ),
                    "dynamic_reward_speedup",
                    None,
                ),
            )
            assert value == f"{gate:.4f}"


@pytest.mark.parametrize("mode,power", [("piecewise", 1.0), ("piecewise-sqrt", 0.5)])
def test_piecewise_gate_modes_preserve_bounds_neutral_region_and_continuity(mode, power):
    args = SimpleNamespace(
        dynamic_reward_speedup=mode,
        dynamic_reward_range=([0.0, 1.0] if mode == "correct-rate-sqrt" else [0.8, 1.2]),
    )
    gates = [
        _compute_dynamic_coverage_scale(correct, 300, args=args, mode=getattr(args, "dynamic_reward_speedup", None))
        for correct in range(301)
    ]
    assert gates == sorted(gates)
    assert gates[0] == pytest.approx(0.8)
    assert gates[-1] == pytest.approx(1.2)
    assert gates[100:201] == [1.0] * 101
    assert gates[50] == pytest.approx(1 - 0.2 * 0.5**power)
    assert gates[250] == pytest.approx(1 + 0.2 * 0.5**power)
    for correct in (999999, 1000000, 2000000, 2000001):
        assert _compute_dynamic_coverage_scale(
            correct, 3000000, args=args, mode=getattr(args, "dynamic_reward_speedup", None)
        ) == pytest.approx(1.0, abs=0.001)
    for correct, size in ((0, 0), (0, 1), (1, 1)):
        assert (
            _compute_dynamic_coverage_scale(
                correct, size, args=args, mode=getattr(args, "dynamic_reward_speedup", None)
            )
            == 1.0
        )


@pytest.mark.parametrize("num_correct,expected_gate", [(4, 0.9), (8, 1.0), (12, 1.1)])
def test_piecewise_square_root_gate_preview_and_final_reward_match(monkeypatch, num_correct, expected_gate):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_performance_weight", 0.5)
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.dynamic_reward_range = [0.8, 1.2]
    args.dynamic_reward_speedup = "piecewise-sqrt"
    args.dynamic_reward_coverage = "piecewise-sqrt"
    samples = [_make_sample(i, 0, float(i < num_correct)) for i in range(16)]
    for i, sample in enumerate(samples):
        _set_reward_component(
            sample, correctness_score=float(i < num_correct), performance_score=float(i < num_correct)
        )
    preview = _apply_dynamic_group_reward_weights(
        samples,
        [s.reward for s in samples],
        CUDA_AGENT_CONFIGS["reward"],
        args=args,
    )
    assert all("dynamic_reward" not in s.metadata for s in samples)
    expected = [0.5 + 0.5 * expected_gate] * num_correct + [0.0] * (16 - num_correct)
    assert preview == pytest.approx(expected)
    for _ in range(2):
        assert post_process_rollout_rewards(args, samples) == pytest.approx(expected)
        assert [s.reward for s in samples] == pytest.approx(expected)
        metrics = compute_reward_post_process_metrics(samples)
        assert metrics["rollout/dynamic_reward/speedup_scale_mean"] == pytest.approx(expected_gate)
        assert metrics["rollout/dynamic_reward/speedup_reward_delta_mean"] == pytest.approx(
            num_correct / 16 * 0.5 * (expected_gate - 1)
        )


@pytest.mark.parametrize("std_normalization", [False, True])
def test_inverse_correct_rate_cli_without_auxiliary_gate(caplog, std_normalization):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--advantage-estimator",
            "grpo",
            "--dynamic-reward-correctness",
            "inverse-correct-rate",
        ]
        + ([] if std_normalization else ["--disable-grpo-std-normalization"])
    )
    _validate_rollout_reward_post_process_args(args)
    assert args.dynamic_reward_speedup is None
    assert args.grpo_std_normalization is std_normalization
    assert "can cancel inverse-correct-rate" not in caplog.text
    assert "Rollout reward shaping requires" in caplog.text


@pytest.mark.parametrize("num_correct", [0, 1, 2, 16])
@pytest.mark.parametrize("std_normalization", [False, True])
def test_inverse_correct_rate_binary_centering(monkeypatch, num_correct, std_normalization):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_correct_weight", 1.0)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", False)
    args = _make_manager(
        advantage_estimator="grpo", use_multi_turn=False, grpo_std_normalization=std_normalization
    ).args
    args.dynamic_reward_correctness = "inverse-correct-rate"
    samples = [_make_sample(i, 0, float(i < num_correct)) for i in range(16)]
    for i, sample in enumerate(samples):
        _set_reward_component(sample, correctness_score=float(i < num_correct))
    post_process_rollout_rewards(args, samples)
    _, centered = reward_post_process_by_group(args, samples)
    p = num_correct / 16
    if not std_normalization:
        expected = [(float(i < num_correct) - p) * (1.0 - 0.1 * (2 * p - 1)) for i in range(16)]
        assert centered == pytest.approx(expected)
    else:
        # Group std cancels the common scale (up to the normalization epsilon).
        for sample in samples:
            sample.reward = float(sample.index < num_correct)
        _, unscaled = reward_post_process_by_group(args, samples)
        assert centered == pytest.approx(unscaled, rel=1e-5, abs=1e-6)


def test_inverse_correct_rate_groups_by_turn_and_excludes_ineligible_samples(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_correct_weight", 0.5)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", False)
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=True).args
    args.dynamic_reward_correctness = "inverse-correct-rate"
    samples = []
    for turn, outcomes in enumerate(([1, 0, 0, 0], [1, 1, 0, 0])):
        for i, correct in enumerate(outcomes):
            sample = _make_sample(i, 0, 0.5 * correct, turn_idx=turn)
            _set_reward_component(sample, correctness_score=correct)
            samples.append(sample)
    excluded = [_make_sample(10 + i, 0, 99.0, turn_idx=0) for i in range(3)]
    for sample in excluded:
        _set_reward_component(sample, correctness_score=1.0)
    excluded[0].remove_sample = True
    excluded[1].metadata["is_pad_turn"] = True
    excluded[2].status = Sample.Status.ABORTED
    post_process_rollout_rewards(args, samples + excluded)
    assert [s.reward for s in samples] == pytest.approx([0.55, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
    assert [s.reward for s in excluded] == [99.0] * 3
    for sample in samples:
        assert sample.metadata["group_num_valid"] == 4
    metrics = compute_reward_post_process_metrics(samples + excluded)
    assert metrics["rollout/dynamic_reward/correctness_scale_mean"] == pytest.approx(1.05)


def test_dynamic_correctness_disabled_preserves_component_and_metrics():
    sample = _make_sample(0, 0, 3.0)
    _set_reward_component(sample, correctness_score=1.0)
    sample.metadata["reward_component"]["correctness"] = 3.0
    args = SimpleNamespace(
        dynamic_reward_speedup="piecewise",
        dynamic_reward_correctness=None,
        dynamic_reward_range=[0.8, 1.2],
    )
    assert _apply_dynamic_group_reward_weights(
        [sample], [3.0], CUDA_AGENT_CONFIGS["reward"], args=args, record_metrics=True
    ) == [3.0]
    assert sample.metadata["dynamic_reward"] == {
        "speedup_scale": 1.0,
        "speedup_reward_delta": 0.0,
        "speedup_weight": 0.5,
        "speedup_offset": 0.0,
    }
    assert not any("correctness" in name for name in compute_reward_post_process_metrics([sample]))


def test_piecewise_gate_boosts_auxiliary_components_and_records_positive_delta(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_performance_weight", 0.5)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "coverage_reward_weight", 0.5)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "coverage_reward_enable", True)
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.overlong_penalty = "dapo"
    args.dynamic_reward_range = [0.8, 1.2]
    args.dynamic_reward_speedup = "piecewise"
    args.dynamic_reward_coverage = "piecewise"
    args.dynamic_reward_range = [0.8, 2.0]
    args.rollout_max_response_len = 100
    args.overlong_buffer_len = 100
    args.overlong_penalty_factor = 0.1
    samples = [_make_sample(i, 0, 1.5 + 0.5 * i) for i in range(2)]
    for i, sample in enumerate(samples):
        sample.response_length = 100
        sample.tokens = [0] * 100
        _set_reward_component(sample, correctness_score=1.0, performance_score=1.0 + i, coverage_score=1.0)
    for _ in range(2):
        assert post_process_rollout_rewards(args, samples) == pytest.approx([2.0, 2.6])
        assert [s.metadata["reward_component"]["coverage"] for s in samples] == [1.0, 1.0]
        assert [s.metadata["reward_component"]["correctness"] for s in samples] == [0.5, 0.5]
        metrics = compute_reward_post_process_metrics(samples)
        assert metrics["rollout/dynamic_reward/speedup_scale_mean"] == 1.2
        assert metrics["rollout/dynamic_reward/speedup_scale_boosted_fraction"] == 1.0
        assert metrics["rollout/dynamic_reward/speedup_scale_scaled_fraction"] == 0.0
        assert metrics["rollout/dynamic_reward/speedup_reward_delta_mean"] == pytest.approx(0.15)


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


def test_dynamic_reward_weights_keep_half_maxima_and_apply_before_rloo():
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    manager.args.dynamic_reward_speedup = "correct-rate-sqrt"
    manager.args.dynamic_reward_coverage = "correct-rate-sqrt"
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

    post_process_rollout_rewards(manager.args, samples)
    raw_rewards, advantages = reward_post_process_by_group(manager.args, samples)

    gate = 0.5**0.5
    speedup_scale = 1 + 0.2 * (2 * gate - 1)
    expected_raw = [0.0, 0.5 + speedup_scale * 0.1 + gate * 0.2, 0.5 + speedup_scale * 0.3 + gate * 0.4, 0.0]
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
    assert samples[1].metadata["reward_component"]["performance"] == pytest.approx(speedup_scale * 0.1)
    assert samples[1].metadata["reward_component"]["coverage"] == pytest.approx(gate * 0.2)
    for sample in samples:
        assert sample.metadata["group_num_correct"] == 2
        assert sample.metadata["group_num_valid"] == 4
        assert sample.metadata["group_correct_rate"] == pytest.approx(0.5)
        assert sample.metadata["group_difficulty"] == pytest.approx(0.5)


def test_dynamic_reward_all_correct_adds_amplitude_to_speedup_weight():
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    manager.args.dynamic_reward_speedup = "correct-rate-sqrt"
    manager.args.dynamic_reward_coverage = "correct-rate-sqrt"
    samples = [_make_sample(index, 0, reward) for index, reward in enumerate([0.6, 0.8, 1.0])]
    for sample, performance_reward in zip(samples, [0.1, 0.2, 0.3], strict=True):
        _set_reward_component(
            sample,
            correctness_score=1.0,
            performance_score=2.0 * performance_reward,
            coverage_score=2.0 * (sample.reward - 0.5 - performance_reward),
        )

    post_process_rollout_rewards(manager.args, samples)
    raw_rewards, _advantages = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == pytest.approx([0.62, 0.84, 1.06])


def test_dynamic_reward_rebuilds_overlong_penalty_from_components():
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    manager.args.dynamic_reward_speedup = "correct-rate-sqrt"
    manager.args.dynamic_reward_coverage = "correct-rate-sqrt"
    correct = _make_sample(0, 0, 99.0)
    incorrect = _make_sample(1, 0, 0.0)
    _set_reward_component(correct, correctness_score=1.0, performance_score=1.0, length_score=-0.2)
    _set_reward_component(incorrect)
    correct.metadata["task_reward"] = 1.0
    correct.metadata["length_score"] = -0.2

    post_process_rollout_rewards(manager.args, [correct, incorrect])
    raw_rewards, _advantages = reward_post_process_by_group(manager.args, [correct, incorrect])

    # One of two samples is correct; the penalty is retained exactly once.
    assert raw_rewards == pytest.approx([0.8 + 0.1 * (2 * 0.5**0.5 - 1), 0.0])
    assert correct.reward == pytest.approx(0.8 + 0.1 * (2 * 0.5**0.5 - 1))
    assert correct.reward == pytest.approx(
        sum(v for v in correct.metadata["reward_component"].values() if v is not None)
    )
    # Reprocessing must rebuild from base scores, not compound weighting or penalties.
    post_process_rollout_rewards(manager.args, [correct, incorrect])
    assert reward_post_process_by_group(manager.args, [correct, incorrect])[0] == pytest.approx(raw_rewards)
    assert correct.reward == pytest.approx(0.8 + 0.1 * (2 * 0.5**0.5 - 1))


def test_dynamic_reward_writeback_preserves_other_reward_keys():
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    manager.args.dynamic_reward_speedup = "correct-rate-sqrt"
    manager.args.dynamic_reward_coverage = "correct-rate-sqrt"
    manager.args.reward_key = "kernel"
    sample = _make_sample(0, 0, 99.0)
    original_reward = {"kernel": 99.0, "other": 7.0}
    sample.reward = original_reward
    _set_reward_component(sample, correctness_score=1.0, performance_score=1.0, length_score=-0.2)

    post_process_rollout_rewards(manager.args, [sample])
    raw, _ = reward_post_process_by_group(manager.args, [sample])

    assert raw == pytest.approx([0.9])
    assert sample.reward == pytest.approx({"kernel": 0.9, "other": 7.0})
    assert original_reward == {"kernel": 99.0, "other": 7.0}


def test_dynamic_reward_uses_failed_component_as_branch_sentinel():
    manager = _make_manager(advantage_estimator="rloo", use_multi_turn=False)
    manager.args.dynamic_reward_speedup = "correct-rate-sqrt"
    manager.args.dynamic_reward_coverage = "correct-rate-sqrt"
    kernel_failure = _make_sample(0, 0, 99.0)
    output_mismatch = _make_sample(1, 0, 99.0)
    _set_reward_component(
        kernel_failure,
        correctness_score=1.0,
        performance_score=1.0,
        failed=0.0,
        length_score=-0.1,
    )
    _set_reward_component(
        output_mismatch,
        correctness_score=1.0,
        performance_score=1.0,
        failed=0.25,
        length_score=-0.1,
    )

    post_process_rollout_rewards(manager.args, [kernel_failure, output_mismatch])
    raw_rewards, _advantages = reward_post_process_by_group(manager.args, [kernel_failure, output_mismatch])

    assert raw_rewards == pytest.approx([-0.1, 0.15])
    assert [kernel_failure.reward, output_mismatch.reward] == pytest.approx(raw_rewards)


def test_dynamic_reward_trloo_discounts_raw_return_rewards():
    manager = _make_manager(advantage_estimator="trloo", use_multi_turn=True)
    manager.args.dynamic_reward_speedup = "correct-rate-sqrt"
    manager.args.dynamic_reward_coverage = "correct-rate-sqrt"
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
        sample.metadata["task_reward"] = sample.reward
        sample.metadata["multi_turn_reward"] = -999.0
    for index in range(2):
        _set_multi_turn_rewards(manager.args, [samples[index], samples[index + 2]], "env_done")

    post_process_rollout_rewards(manager.args, samples)
    raw_rewards, advantages = reward_post_process_by_group(manager.args, samples)

    # Turn 0 adds a=0.1; turn 1 adds a*(2*sqrt(1/2)-1) to the speedup weight.
    last_reward = 0.5 + 0.8 * (0.5 + 0.1 * (2 * 0.5**0.5 - 1))
    assert raw_rewards == pytest.approx([1.07, 0.74, last_reward, 0.0])
    assert advantages == pytest.approx([0.33, -0.33, last_reward, -last_reward])
    assert [sample.reward for sample in samples] == pytest.approx([0.62, 0.74, last_reward, 0.0])
    repeated_raw, repeated_advantages = reward_post_process_by_group(manager.args, samples)
    assert repeated_raw == pytest.approx(raw_rewards)
    assert repeated_advantages == pytest.approx(advantages)


@pytest.mark.parametrize("gate", [None, "correct-rate-sqrt", "piecewise", "piecewise-sqrt"])
@pytest.mark.parametrize("overlong", [False, True])
def test_reward_switches_are_independent_and_keep_components_consistent(gate, overlong):
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.dynamic_reward_speedup = gate
    args.dynamic_reward_coverage = gate
    args.overlong_penalty = "dapo" if overlong else None
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
    penalty = 0.2 if overlong else 0.0
    # Scoring cannot use incomplete group statistics. Length shaping settles
    # first, through the same interface that owns final rollout processing.
    assert post_process_rollout_rewards(args, samples, stage="sample") == pytest.approx(
        [1.0 - penalty, 1.0 - penalty, -penalty]
    )
    task_reward = 1.0 + (0.1 * (2 * (2 / 3) ** 0.5 - 1) if gate == "correct-rate-sqrt" else 0.0)
    expected = [task_reward - penalty, task_reward - penalty, -penalty]

    for _ in range(2):
        assert post_process_rollout_rewards(args, samples) == pytest.approx(expected)
        assert [s.reward for s in samples] == pytest.approx(expected)
        assert [s.metadata["raw_task_reward"] for s in samples] == [1.0, 1.0, 0.0]
        for sample, scores in zip(samples, base_scores, strict=True):
            metadata = sample.metadata
            assert metadata["kernel_score"] == scores
            assert metadata["reward_component"]["length"] == pytest.approx(-penalty)
            assert metadata["length_score"] == pytest.approx(-penalty)
            assert "overlong_penalty" not in metadata
            assert "overlong_penalty" not in metadata["reward_component"]
            assert "length_bonus" not in metadata["reward_component"]
            assert "length_score" not in metadata["reward_component"]
            assert sample.reward == pytest.approx(
                sum(v for v in metadata["reward_component"].values() if v is not None)
            )
            assert sample.reward == pytest.approx(metadata["task_reward"] - penalty)
    # The training hook reads settled rewards and centers them without reshaping.
    raw, advantages = reward_post_process_by_group(args, samples)
    assert raw == pytest.approx(expected)
    assert advantages == pytest.approx([reward - sum(expected) / 3 for reward in expected])
    assert [s.reward for s in samples] == pytest.approx(expected)


def test_rollout_length_processor_does_not_require_group_or_advantage_args():
    args = SimpleNamespace(
        reward_key=None,
        overlong_penalty="dapo",
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


def test_rollout_reward_processors_reject_unknown_stage():
    with pytest.raises(ValueError, match="stage"):
        post_process_rollout_rewards(SimpleNamespace(), [], stage="unknown")


def test_dynamic_reward_metrics_use_group_gates_and_ungated_performance(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_performance_weight", 0.5)
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=True).args
    args.dynamic_reward_speedup = "correct-rate-sqrt"
    args.dynamic_reward_coverage = "correct-rate-sqrt"
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
        _apply_dynamic_group_reward_weights(group, [s.reward for s in group], CUDA_AGENT_CONFIGS["reward"], args=args)
    assert compute_reward_post_process_metrics(samples) == {}
    lower_scale = 1 + 0.2 * (2 * 0.5**0.5 - 1)
    upper_scale = 1 + 0.2 * (2 * (2 / 3) ** 0.5 - 1)
    for _ in range(2):
        post_process_rollout_rewards(args, samples)
        metrics = compute_reward_post_process_metrics(samples)
        prefix = "rollout/dynamic_reward/"
        assert metrics[f"{prefix}group_count"] == 3
        assert metrics[f"{prefix}sample_count"] == 7
        assert metrics[f"{prefix}speedup_scale_mean"] == pytest.approx((lower_scale + upper_scale + 1.2) / 3)
        assert metrics[f"{prefix}speedup_scale_min"] == pytest.approx(lower_scale)
        assert metrics[f"{prefix}speedup_scale_max"] == 1.2
        assert metrics[f"{prefix}speedup_scale_p25"] == pytest.approx((lower_scale + upper_scale) / 2)
        assert metrics[f"{prefix}speedup_scale_p50"] == pytest.approx(upper_scale)
        assert metrics[f"{prefix}speedup_scale_p75"] == pytest.approx((upper_scale + 1.2) / 2)
        assert metrics[f"{prefix}speedup_scale_zero_fraction"] == 0.0
        assert metrics[f"{prefix}speedup_scale_scaled_fraction"] == 0.0
        assert metrics[f"{prefix}speedup_scale_one_fraction"] == 0.0
        assert metrics[f"{prefix}speedup_reward_delta_mean"] == pytest.approx(
            (upper_scale + 0.5 * lower_scale - 1.3) / 7
        )
        assert metrics[f"{prefix}speedup_reward_delta_min"] == 0.0
        assert metrics[f"{prefix}speedup_reward_delta_max"] == pytest.approx(0.1)


@pytest.mark.parametrize("dynamic", [False, True])
def test_correct_rate_buckets_count_retained_prompt_turn_groups(dynamic):
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=True).args
    args.dynamic_reward_speedup = "correct-rate-sqrt" if dynamic else None
    args.dynamic_reward_coverage = "correct-rate-sqrt" if dynamic else None
    samples = []
    for group_idx, turn, num_correct, size in [
        (0, 0, 0, 2),
        (0, 1, 1, 3),
        (1, 0, 1, 2),
        (1, 1, 2, 3),
        (2, 0, 4, 4),
        (3, 0, 1, 1),
    ]:
        group = []
        for index in range(size):
            correct = index < num_correct
            sample = _make_sample(len(samples), group_idx, float(correct), turn_idx=turn)
            _set_reward_component(
                sample,
                correctness_score=float(correct),
                performance_score=float(correct),
                failed=None if correct else 0.0,
            )
            samples.append(sample)
            group.append(sample)
        # Both kernel collectors annotate complete groups independently of shaping.
        annotate_group_difficulty(group)
    post_process_rollout_rewards(args, samples)
    # Drop a whole group and the incorrect sample of the 2/3 group. Keep the
    # original complete-group rate, rather than recomputing it as 1 after filtering.
    retained = [s for s in samples if s.group_index != 3 and s.index != 9]
    for index, exclusion in enumerate(("removed", "aborted", "pad", "non_kernel")):
        sample = _make_sample(100 + index, 100 + index, 0.0, turn_idx=0)
        sample.metadata.update(
            group_correct_rate=1.0, dynamic_reward={"speedup_scale": 1.0, "speedup_reward_delta": 0.0}
        )
        if exclusion == "removed":
            sample.remove_sample = True
        elif exclusion == "aborted":
            sample.status = Sample.Status.ABORTED
        elif exclusion == "pad":
            sample.metadata["is_pad_turn"] = True
        else:
            sample.metadata["role"] = "other"
        retained.append(sample)

    metrics = _compute_group_correct_rate_metrics(retained)
    assert metrics["group_correct_rate_count"] == 5
    for bucket, count in (("low", 1), ("medium", 3), ("high", 1)):
        assert metrics[f"group_correct_rate_{bucket}_count"] == count
        assert metrics[f"group_correct_rate_{bucket}_fraction"] == pytest.approx(count / 5)
    assert not any("group_correct_rate" in key for key in compute_reward_post_process_metrics(retained))


@pytest.mark.parametrize("correct_rate", [None, True, "0.5", float("nan"), float("inf"), -0.1, 1.1])
def test_correct_rate_buckets_skip_missing_or_invalid_rates(correct_rate):
    sample = _make_sample(0, 0, 0.0)
    sample.metadata["group_correct_rate"] = correct_rate
    assert _compute_group_correct_rate_metrics([sample]) == {}
    assert _compute_group_correct_rate_metrics([_make_sample(1, 1, 0.0)]) == {}
    assert _compute_group_correct_rate_metrics([]) == {}


@pytest.mark.parametrize("explicit_none", [False, True])
def test_disabled_dynamic_reward_clears_stale_metric_records(explicit_none):
    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    if explicit_none:
        args.dynamic_reward_speedup = None
        args.dynamic_reward_coverage = None
    sample = Sample(reward=1.0, metadata={"dynamic_reward": {"speedup_scale": 0.0, "speedup_reward_delta": -1.0}})
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
    manager.args.dynamic_reward_speedup = "correct-rate-sqrt" if dynamic else None
    manager.args.dynamic_reward_coverage = "correct-rate-sqrt" if dynamic else None
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
        assert samples[0].reward == pytest.approx(1.0 + 0.1 * (2 * (2 / 3) ** 0.5 - 1))
        logged.append(dict(metrics))

    monkeypatch.setattr(rollout_module.logging_utils, "log", capture_log)
    post_process_rollout_rewards(manager.args, samples)
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
    assert logged[0]["rollout/dynamic_reward/speedup_scale_mean"] == pytest.approx(1 + 0.2 * (2 * (2 / 3) ** 0.5 - 1))
    assert logged[0]["rollout/dynamic_reward/speedup_reward_delta_mean"] == pytest.approx(
        0.2 * (2 * (2 / 3) ** 0.5 - 1) / 3
    )
    assert "reward post-process 4" in caplog.text


def test_dynamic_reward_metrics_reach_tensorboard_as_scalars(monkeypatch):
    from slime.observability import logging_utils
    from slime.observability.tensorboard_utils import _TensorboardAdapter

    args = _make_manager(advantage_estimator="grpo", use_multi_turn=False).args
    args.dynamic_reward_speedup = "correct-rate-sqrt"
    args.dynamic_reward_coverage = "correct-rate-sqrt"
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
    assert ("rollout/dynamic_reward/speedup_scale_mean", 1.2, 4) in scalars
    assert dict((key, value) for key, value, _ in scalars)[
        "rollout/dynamic_reward/speedup_reward_delta_mean"
    ] == pytest.approx(0.1)
    assert flushes == [True]


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(),
        SimpleNamespace(dynamic_reward_speedup=None, dynamic_reward_range=[0.8, 1.2]),
    ],
)
def test_disabled_dynamic_gate_is_neutral(args):
    assert _compute_dynamic_coverage_scale(1, 16, args=args, mode=getattr(args, "dynamic_reward_speedup", None)) == 1.0
    assert (
        _compute_dynamic_coverage_scale(16, 16, args=args, mode=getattr(args, "dynamic_reward_speedup", None)) == 1.0
    )


def test_thresholds_and_range_do_not_enable_dynamic_weight():
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--difficulty-thresholds",
            "0.2",
            "0.8",
            "--dynamic-reward-range",
            "0.4",
            "2",
        ]
    )
    _validate_rollout_reward_post_process_args(args)
    assert args.dynamic_reward_speedup is None
    sample = Sample(reward=0.7, metadata={})
    assert post_process_rollout_rewards(args, [sample]) == [0.7]
    assert sample.metadata == {}


def test_dppo_script_disables_dynamic_weight_by_default():
    script = repo_root / "examples/kernel_agent/run_qwen3.6_27B_full_async_dppo.sh"
    assert "--dynamic-reward-speedup None" in script.read_text(encoding="utf-8")


@pytest.mark.parametrize("gate", ["None", "correct-rate-sqrt", "piecewise", "piecewise-sqrt"])
@pytest.mark.parametrize("penalty", ["None", "dapo"])
def test_explicit_reward_method_choices(gate, penalty):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--dynamic-reward-speedup",
            gate,
            "--overlong-penalty",
            penalty,
            "--dynamic-reward-range",
            "0.8",
            "1.2",
        ]
    )
    _validate_rollout_reward_post_process_args(args)
    assert args.dynamic_reward_speedup == (None if gate == "None" else gate)
    assert args.overlong_penalty == (None if penalty == "None" else penalty)
    if gate == "None" and penalty == "None":
        sample = Sample(reward=0.7, metadata={})
        assert post_process_rollout_rewards(args, [sample]) == [0.7]
        assert sample.metadata == {}


@pytest.mark.parametrize("values", [[], ["True"], ["False"], ["unknown"]])
def test_overlong_penalty_rejects_bare_flag_and_unknown_methods(values):
    import argparse

    from slime.utils.arguments import get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--rollout-batch-size", "1", "--overlong-penalty", *values])
    assert exc.value.code == 2


@pytest.mark.parametrize("method", [True, False, "None", "unknown"])
def test_reward_runtime_rejects_unparsed_penalty_methods(method):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    args = SimpleNamespace(overlong_penalty=method)
    with pytest.raises(ValueError, match="--overlong-penalty"):
        _validate_rollout_reward_post_process_args(args)
    with pytest.raises(ValueError, match="--overlong-penalty"):
        post_process_rollout_rewards(args, [])


@pytest.mark.parametrize("gate", ["none", "unknown", ""])
def test_dynamic_gate_rejects_invalid_selection(gate):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    with pytest.raises(ValueError, match="--dynamic-reward-speedup"):
        _validate_rollout_reward_post_process_args(
            SimpleNamespace(
                dynamic_reward_speedup=gate,
                dynamic_reward_range=([0.0, 1.0] if gate == "correct-rate-sqrt" else [0.8, 1.2]),
            )
        )


@pytest.mark.parametrize("gate", [None, "correct-rate-sqrt", "piecewise", "piecewise-sqrt"])
@pytest.mark.parametrize("overlong", [False, True])
def test_rollout_reward_switch_args_parse_and_validate(gate, overlong):
    import argparse

    from slime.utils.arguments import _validate_rollout_reward_post_process_args, get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            *(["--dynamic-reward-speedup", gate, "--dynamic-reward-range", "0.8", "1.2"] if gate is not None else []),
            *(["--overlong-penalty", "dapo"] if overlong else []),
            "--custom-reward-post-process-path",
            "examples.kernel_agent.kernel_reward.reward_post_process_by_group",
        ]
    )
    assert not hasattr(args, "rollout_reward_post_processors")
    assert args.dynamic_reward_speedup == gate
    assert args.overlong_penalty == ("dapo" if overlong else None)
    assert args.difficulty_thresholds == [1 / 3, 2 / 3]
    assert args.dynamic_reward_range == (None if gate is None else [0.8, 1.2])
    assert not hasattr(args, "dynamic_reward_gate_power")
    _validate_rollout_reward_post_process_args(args)
    expected_range = None if gate is None else [0.8, 1.2]
    assert args.dynamic_reward_range == expected_range


@pytest.mark.parametrize("mode,power", [("piecewise", 1.0), ("piecewise-sqrt", 0.5)])
def test_piecewise_dynamic_gate_args_parse_and_validate(mode, power):
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
            "--dynamic-reward-speedup",
            mode,
            "--difficulty-thresholds",
            "0.2",
            "0.8",
            "--dynamic-reward-range",
            "0.4",
            "2.0",
        ]
    )
    _validate_difficulty_thresholds_args(args)
    _validate_rollout_reward_post_process_args(args)
    assert args.difficulty_thresholds == [0.2, 0.8]
    assert args.dynamic_reward_range == [0.4, 2.0]
    assert args.dynamic_reward_speedup == mode
    assert args.overlong_penalty is None  # Enabling a gate does not enable length shaping.
    assert _compute_dynamic_coverage_scale(
        1, 10, args=args, mode=getattr(args, "dynamic_reward_speedup", None)
    ) == pytest.approx(1 - 0.6 * 0.5**power)
    assert _compute_dynamic_coverage_scale(
        9, 10, args=args, mode=getattr(args, "dynamic_reward_speedup", None)
    ) == pytest.approx(1 + 0.5**power)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--dynamic-reward-gate", "piecewise"),
        ("--dynamic-reward-gate-range", "0.8"),
        ("--dynamic-reward-gate-power", "0.5"),
        ("--dynamic-reward-speedup", "sqrt"),
        ("--dynamic-reward-correctness", "inverse-gate"),
        ("--rollout-reward-post-processors", "dynamic-weight"),
    ],
)
def test_reward_args_reject_removed_options(option, value):
    import argparse

    from slime.utils.arguments import get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--rollout-batch-size", "1", option, value])
    assert exc.value.code == 2


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
@pytest.mark.parametrize("mode", ["piecewise", "piecewise-sqrt"])
def test_piecewise_dynamic_gate_args_reject_invalid_values(gate_range, mode):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    args = SimpleNamespace(dynamic_reward_coverage=mode, dynamic_reward_range=gate_range)
    with pytest.raises(ValueError, match="--dynamic-reward-range"):
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
@pytest.mark.parametrize("mode", ["correct-rate-sqrt", "piecewise", "piecewise-sqrt"])
def test_difficulty_thresholds_validation_is_independent_of_reward_mode(thresholds, mode):
    from slime.utils.arguments import _validate_difficulty_thresholds_args

    with pytest.raises(ValueError, match="--difficulty-thresholds"):
        _validate_difficulty_thresholds_args(
            SimpleNamespace(
                difficulty_thresholds=thresholds,
                dynamic_reward_speedup=mode,
                dynamic_reward_coverage=mode,
                dynamic_reward_range=([0.0, 1.0] if mode == "correct-rate-sqrt" else [0.8, 1.2]),
            )
        )


@pytest.mark.parametrize("thresholds", [[0.5], [0.1, 0.4, 0.8]])
@pytest.mark.parametrize("mode", ["piecewise", "piecewise-sqrt"])
def test_shared_difficulty_thresholds_allow_more_buckets_but_piecewise_requires_two(thresholds, mode):
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
    args.dynamic_reward_range = [0.8, 1.2]
    args.dynamic_reward_speedup = mode
    args.dynamic_reward_coverage = mode
    with pytest.raises(ValueError, match="exactly two --difficulty-thresholds"):
        _validate_rollout_reward_post_process_args(args)


@pytest.mark.parametrize(
    "buffer_len,factor",
    [
        (0, 1.0),
        (100, -1.0),
        (100, float("nan")),
        (100, float("inf")),
    ],
)
def test_overlong_penalty_args_reject_invalid_values(buffer_len, factor):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    with pytest.raises(ValueError):
        _validate_rollout_reward_post_process_args(
            SimpleNamespace(
                overlong_penalty="dapo",
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

    post_process_rollout_rewards(manager.args, samples)
    raw_rewards, rewards = reward_post_process_by_group(manager.args, samples)

    expected_raw_rewards = [-0.5, -0.375, -0.125]
    assert [sample.reward for sample in samples] == expected_raw_rewards
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

    post_process_rollout_rewards(manager.args, samples)
    raw_rewards, rewards = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == pytest.approx([-0.5, -0.375, -0.125])
    assert rewards == pytest.approx([-0.25, -0.0625, 0.3125])


def test_failed_group_reward_can_be_disabled(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", False)
    manager = _make_manager(advantage_estimator="trloo", use_multi_turn=False)
    samples = [_make_sample(index, 0, 0.0) for index in range(3)]
    for sample, kernel_failed_score in zip(samples, [-1.0, -0.75, -0.25], strict=True):
        sample.metadata.update({"multi_turn_reward": 0.0, "kernel_failed_score": kernel_failed_score})

    post_process_rollout_rewards(manager.args, samples)
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

    post_process_rollout_rewards(manager.args, samples)
    raw_rewards, _ = reward_post_process_by_group(manager.args, samples)

    assert raw_rewards == [0.0, 0.0, 0.5]


def test_all_failed_reward_from_old_dump_without_kernel_failed_scores_is_unchanged(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    manager = _make_manager(advantage_estimator="trloo", use_multi_turn=False)
    samples = [_make_sample(index, 0, 0.0) for index in range(3)]
    for sample in samples:
        sample.metadata["multi_turn_reward"] = 0.0

    post_process_rollout_rewards(manager.args, samples)
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


@pytest.mark.parametrize("mode", ["piecewise", "piecewise-sqrt", "inverse-piecewise", "inverse-piecewise-sqrt"])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_dynamic_gate_requires_explicit_range(mode, explicit_none):
    from slime.utils.arguments import _validate_rollout_reward_post_process_args

    args = SimpleNamespace(dynamic_reward_coverage=mode)
    if explicit_none:
        args.dynamic_reward_range = None
    with pytest.raises(ValueError, match="--dynamic-reward-range is required"):
        _validate_rollout_reward_post_process_args(args)
    with pytest.raises(ValueError, match="--dynamic-reward-range is required"):
        _compute_dynamic_coverage_scale(0, 1, args=args, mode=getattr(args, "dynamic_reward_coverage", None))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
