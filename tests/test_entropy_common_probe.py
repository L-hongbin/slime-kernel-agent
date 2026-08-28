"""CPU coverage for the exact fixed-batch entropy common-probe chain."""

from __future__ import annotations

import inspect
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.backends.megatron_utils.path_bootstrap import ensure_megatron_lm_on_sys_path

ensure_megatron_lm_on_sys_path()

from slime.backends.megatron_utils import cp_utils
from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils import model as model_module
from slime.observability import train_metric_utils as metric_module
from slime.observability.train_metric_utils import (
    ENTROPY_COMMON_PROBE_DENOMINATOR_KEY,
    ENTROPY_COMMON_PROBE_MASK_KEY,
    ENTROPY_COMMON_PROBE_METRIC_KEY,
    ENTROPY_COMMON_PROBE_NUMERATOR_KEY,
    add_entropy_common_probe_metric,
    format_train_metric_key,
)

NUM_GPUS = 0


def _set_cp(monkeypatch, *, size: int, rank: int) -> None:
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: size)
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda: rank)


def test_common_probe_sufficient_stats_use_original_global_token_mask(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    entropy = torch.tensor([1.0, 2.0, 3.0, 4.0])
    original_mask = torch.tensor([1, 1, 0, 1])
    post_mis_mask = torch.zeros_like(original_mask)

    numerator, denominator = loss_module._entropy_common_probe_sufficient_stats(
        entropy,
        total_lengths=[6],
        response_lengths=[4],
        loss_masks=[original_mask],
        qkv_format="thd",
        max_seq_lens=None,
    )

    torch.testing.assert_close(numerator, torch.tensor(7.0))
    torch.testing.assert_close(denominator, torch.tensor(3.0))
    assert post_mis_mask.sum().item() == 0  # population intentionally differs


def test_common_probe_cp2_bshd_actual_padding_tiles_exact_original_population(monkeypatch):
    """Exercise the real wave-2 shape: real 4520, response 3234, padded 5120."""
    previous_mode = cp_utils.get_cp_partition_mode()
    cp_utils.set_cp_partition_mode(cp_utils.CP_PARTITION_CONTIGUOUS)
    try:
        total_length, response_length, max_seq_len = 4520, 3234, 5120
        full_entropy = torch.linspace(0.01, 1.25, response_length)
        original_mask = (torch.arange(response_length) % 5 != 0).to(torch.int32)
        numerators = []
        denominators = []

        for rank in range(2):
            _set_cp(monkeypatch, size=2, rank=rank)
            local_entropy = cp_utils.slice_log_prob_with_cp(
                full_entropy,
                total_length,
                response_length,
                "bshd",
                max_seq_len,
            )
            numerator, denominator = loss_module._entropy_common_probe_sufficient_stats(
                local_entropy,
                total_lengths=[total_length],
                response_lengths=[response_length],
                loss_masks=[original_mask],
                qkv_format="bshd",
                max_seq_lens=[max_seq_len],
            )
            numerators.append(numerator)
            denominators.append(denominator)

        torch.testing.assert_close(sum(numerators), (full_entropy * original_mask).sum())
        torch.testing.assert_close(sum(denominators), original_mask.sum().to(full_entropy.dtype))
    finally:
        cp_utils.set_cp_partition_mode(previous_mode)


def test_policy_loss_common_probe_gate_fails_if_snapshot_was_not_plumbed(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    current_log_probs = torch.tensor([-1.0, -2.0])
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (
            torch.empty(0),
            {"log_probs": [current_log_probs], "entropy": [torch.tensor([0.2, 0.4])]},
        ),
    )
    monkeypatch.setattr(
        loss_module,
        "compute_policy_loss",
        lambda *args, **kwargs: {
            "pg_losses": torch.zeros(2),
            "pg_clipfrac": torch.zeros(2),
            "pg_upper_clipfrac": torch.zeros(2),
            "pg_lower_clipfrac": torch.zeros(2),
        },
    )
    args = Namespace(
        use_rollout_logprobs=True,
        use_opsm=False,
        advantage_estimator="grpo",
        policy_loss_mode="ppo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=5.0,
        get_mismatch_metrics=False,
        use_tis=False,
        entropy_coef=0.0,
        use_kl_loss=False,
        entropy_common_probe=True,
        qkv_format="thd",
    )
    batch = {
        "advantages": [torch.ones(2)],
        "rollout_log_probs": [current_log_probs.clone()],
        "response_lengths": [2],
        "total_lengths": [3],
        "unconcat_tokens": [torch.arange(3)],
        "loss_masks": [torch.ones(2)],
    }

    with pytest.raises(RuntimeError, match="requires training batch field"):
        loss_module.policy_loss_function(args, batch, torch.zeros((1, 3, 4)), torch.mean)


def test_common_probe_finalizer_divides_after_reduction_and_removes_raw_keys():
    metrics = {
        "loss": 1.25,
        ENTROPY_COMMON_PROBE_NUMERATOR_KEY: 17.5,
        ENTROPY_COMMON_PROBE_DENOMINATOR_KEY: 5.0,
    }

    assert add_entropy_common_probe_metric(metrics) is metrics
    assert metrics[ENTROPY_COMMON_PROBE_METRIC_KEY] == pytest.approx(3.5)
    assert ENTROPY_COMMON_PROBE_NUMERATOR_KEY not in metrics
    assert ENTROPY_COMMON_PROBE_DENOMINATOR_KEY not in metrics
    assert metrics["loss"] == 1.25
    assert format_train_metric_key(ENTROPY_COMMON_PROBE_METRIC_KEY) == "entropy/common_probe"


def test_common_probe_finalizer_requires_stats_when_explicitly_requested():
    metrics = {"loss": 1.25}
    assert add_entropy_common_probe_metric(metrics) is metrics

    with pytest.raises(RuntimeError, match="contain no common-probe"):
        add_entropy_common_probe_metric(metrics, required=True)


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_common_probe_ratio_cancels_train_reducer_outer_normalization(monkeypatch, calculate_per_token_loss):
    monkeypatch.setattr(metric_module.dist, "all_reduce", lambda values, group=None: None)
    keys = [ENTROPY_COMMON_PROBE_NUMERATOR_KEY, ENTROPY_COMMON_PROBE_DENOMINATOR_KEY]
    losses_reduced = [
        {"keys": keys, "values": torch.tensor([100.0, 6.0, 2.0])},
        {"keys": keys, "values": torch.tensor([50.0, 8.0, 2.0])},
    ]
    reduced = metric_module.reduce_train_step_metrics(
        losses_reduced,
        calculate_per_token_loss=calculate_per_token_loss,
        step_global_batch_size=10,
        cp_size=2,
        dp_with_cp_group=object(),
    )

    add_entropy_common_probe_metric(reduced)
    assert reduced == {ENTROPY_COMMON_PROBE_METRIC_KEY: pytest.approx(3.5)}


@pytest.mark.parametrize(
    "metrics,match",
    [
        ({ENTROPY_COMMON_PROBE_NUMERATOR_KEY: 1.0}, "missing"),
        (
            {
                ENTROPY_COMMON_PROBE_NUMERATOR_KEY: 1.0,
                ENTROPY_COMMON_PROBE_DENOMINATOR_KEY: 0.0,
            },
            "positive",
        ),
    ],
)
def test_common_probe_finalizer_fails_closed_on_partial_or_zero_stats(metrics, match):
    with pytest.raises(RuntimeError, match=match):
        add_entropy_common_probe_metric(metrics)


def test_rollout_logger_skips_private_common_probe_masks(monkeypatch):
    monkeypatch.setattr(cp_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(cp_utils.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        cp_utils.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 1,
    )
    captured = {}

    def capture(_metric_name, _args, _rollout_id, log_dict):
        captured.update(log_dict)
        return {}

    monkeypatch.setattr(metric_module, "gather_log_data", capture)
    args = Namespace(
        qkv_format="thd",
        ci_test=False,
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
    )
    rollout_data = {
        "tokens": [torch.arange(3)],
        "loss_masks": [torch.ones(2)],
        ENTROPY_COMMON_PROBE_MASK_KEY: [torch.ones(2)],
        "response_lengths": [2],
        "total_lengths": [3],
        "rollout_mask_sums": [torch.tensor(2.0)],
        "global_batch_sizes": [1],
        "num_microbatches": [1],
        "micro_batch_indices": [[0]],
    }

    metric_module.log_rollout_data(0, args, rollout_data)
    assert ENTROPY_COMMON_PROBE_MASK_KEY not in captured


def test_rollout_logger_reports_sampled_token_entropy(monkeypatch):
    monkeypatch.setattr(cp_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(cp_utils.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        cp_utils.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 1,
    )
    captured = {}

    def capture(_metric_name, _args, _rollout_id, log_dict):
        captured.update(log_dict)
        return {}

    monkeypatch.setattr(metric_module, "gather_log_data", capture)
    args = Namespace(
        qkv_format="thd",
        ci_test=False,
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
    )
    rollout_data = {
        "tokens": [torch.arange(4)],
        "loss_masks": [torch.tensor([1.0, 0.0, 1.0])],
        "rollout_log_probs": [torch.tensor([-0.25, -9.0, -0.75])],
        "response_lengths": [3],
        "total_lengths": [4],
        "rollout_mask_sums": [torch.tensor(2.0)],
        "global_batch_sizes": [1],
        "num_microbatches": [1],
        "micro_batch_indices": [[0]],
    }

    metric_module.log_rollout_data(0, args, rollout_data)

    assert captured["entropy_mc"] == pytest.approx((1.0, 2.0))


def test_model_requests_probe_field_and_finalizes_before_metric_formatting():
    train_step_source = inspect.getsource(model_module.train_one_step)
    assert "ENTROPY_COMMON_PROBE_MASK_KEY" in train_step_source

    train_source = inspect.getsource(model_module.train)
    finalize_at = train_source.index("add_entropy_common_probe_metric(")
    format_at = train_source.index("format_train_metric_key(key, role_tag)")
    assert finalize_at < format_at


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
