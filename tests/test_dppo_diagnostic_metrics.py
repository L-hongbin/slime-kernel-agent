"""CPU integration tests for predictive-DPPO diagnostic metric reduction."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.backends.megatron_utils import loss as loss_module  # noqa: E402

NUM_GPUS = 0


def test_predictive_logprob_path_extracts_entropy_directional_moment_with_response_alignment(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)
    observed_kwargs = {}

    def fake_calculate(logits, tokens, _tp_group, **kwargs):
        observed_kwargs.update(kwargs)
        rows = torch.arange(logits.size(0), dtype=logits.dtype)
        return rows.unsqueeze(-1), rows + 10.0, rows + 20.0

    monkeypatch.setattr(loss_module, "calculate_log_probs_and_entropy", fake_calculate)
    args = Namespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        log_probs_chunk_size=2,
        allgather_cp=False,
        policy_loss_mode="dppo_topk_kl_predictive",
    )
    _, result = loss_module.get_log_probs_and_entropy(
        torch.zeros((1, 5, 7), dtype=torch.float32),
        args=args,
        unconcat_tokens=[torch.arange(5)],
        total_lengths=[5],
        response_lengths=[2],
        with_entropy=True,
    )

    assert observed_kwargs["with_dppo_directional_moment"] is True
    torch.testing.assert_close(result["log_probs"][0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(result["entropy"][0], torch.tensor([12.0, 13.0]))
    torch.testing.assert_close(
        result["dppo_entropy_directional_moment"][0],
        torch.tensor([22.0, 23.0]),
    )


def test_predictive_diagnostics_use_pre_tis_reducer_and_emit_namespaced_joint_moments(monkeypatch):
    sampled_rollout = torch.tensor([0.4, 0.4, 0.4, 0.4], dtype=torch.float64)
    sampled_train = torch.tensor([0.5, 0.4, 0.45, 0.4], dtype=torch.float64)
    advantages = torch.tensor([1.0, 1.0, -2.0, -2.0], dtype=torch.float64)
    entropy = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
    entropy_unit_direction = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    # M = dH/deta|_{e_k-p} + p_k(log(p_k) + H).
    directional_moment = entropy_unit_direction + sampled_train * (sampled_train.log() + entropy)
    rollout_support = torch.tensor(
        [
            [0.4, 0.3, 0.1],
            [0.4, 0.3, 0.1],
            [0.4, 0.3, 0.3],
            [0.4, 0.3, 0.3],
        ],
        dtype=torch.float64,
    )
    train_support = torch.tensor(
        [
            [0.5, 0.2, 0.1],
            [0.4, 0.3, 0.1],
            [0.45, 0.5, 0.05],
            [0.4, 0.3, 0.3],
        ],
        dtype=torch.float64,
    )
    support_valid = torch.ones_like(rollout_support, dtype=torch.bool)

    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (
            torch.empty(0),
            {
                "log_probs": [sampled_train.log()],
                "entropy": [entropy],
                "dppo_entropy_directional_moment": [directional_moment],
            },
        ),
    )
    monkeypatch.setattr(
        loss_module,
        "_validate_dppo_predictive_support_batch",
        lambda *args, **kwargs: (
            torch.zeros_like(rollout_support, dtype=torch.long),
            rollout_support.log(),
            support_valid,
        ),
    )
    monkeypatch.setattr(
        loss_module,
        "get_dppo_predictive_support_log_probs",
        lambda *args, **kwargs: [train_support.log()],
    )
    monkeypatch.setattr(
        loss_module,
        "vanilla_tis_function",
        lambda **kwargs: (kwargs["pg_loss"], [torch.zeros(4, dtype=torch.float64)], {}),
    )

    # Any diagnostic accidentally reduced with the rebuilt post-TIS reducer
    # becomes 99. The expected joint moments below must use the original mean.
    monkeypatch.setattr(
        loss_module,
        "get_sum_of_sample_mean",
        lambda *args, **kwargs: (lambda value: value.sum() * 0 + 99.0),
    )

    args = Namespace(
        use_rollout_logprobs=True,
        use_opsm=False,
        advantage_estimator="grpo",
        policy_loss_mode="dppo_topk_kl_predictive",
        eps_clip=0.01,
        eps_clip_high=0.01,
        eps_clip_c=5.0,
        dppo_predictive_tail_estimator="aggregated",
        vocab_size=10,
        get_mismatch_metrics=False,
        use_tis=True,
        custom_tis_function_path=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    batch = {
        "advantages": [advantages],
        "log_probs": [sampled_train.log()],
        "rollout_log_probs": [sampled_rollout.log()],
        "response_lengths": [4],
        "total_lengths": [5],
        "unconcat_tokens": [torch.arange(5)],
        "loss_masks": [torch.ones(4, dtype=torch.float64)],
        "group_mask_sums": [torch.tensor(4.0)],
        "rollout_topk_token_ids": [torch.zeros_like(rollout_support, dtype=torch.long)],
        "rollout_topk_valid_mask": [support_valid],
    }

    _, metrics = loss_module.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 5, 10), dtype=torch.float64),
        sum_of_sample_mean=torch.mean,
    )

    torch.testing.assert_close(metrics["dppo/adv_positive_token_frac"], torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(metrics["dppo/adv_negative_token_frac"], torch.tensor(0.5, dtype=torch.float64))
    # The ordinary metric formatter maps these to train/dppo_importance_* in
    # both the text log and centralized W&B payload.
    torch.testing.assert_close(metrics["dppo_importance_ratio"], torch.tensor(1.09375, dtype=torch.float64))
    torch.testing.assert_close(metrics["dppo_importance_weight"], torch.tensor(1.09375, dtype=torch.float64))
    torch.testing.assert_close(metrics["dppo/upper_clip_joint_frac"], torch.tensor(0.25, dtype=torch.float64))
    torch.testing.assert_close(metrics["dppo/lower_clip_joint_frac"], torch.tensor(0.25, dtype=torch.float64))
    torch.testing.assert_close(
        metrics["dppo/train_sampled_prob_positive_clipped_joint_mean"],
        torch.tensor(0.5 / 4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["_entropy/upper_clipped_joint_mean"],
        torch.tensor(0.2 / 4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["_entropy/lower_clipped_joint_mean"],
        torch.tensor(0.6 / 4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["_entropy/adv_positive_ratio_ge_1_joint_mean"],
        torch.tensor((0.2 + 0.4) / 4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["_entropy/adv_negative_ratio_ge_1_joint_mean"],
        torch.tensor((0.6 + 0.8) / 4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["dppo/signed_update_unmasked"],
        torch.tensor(-0.5, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["dppo/signed_update_kept"],
        torch.tensor(-0.25, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["dppo/signed_update_mask_delta"],
        torch.tensor(0.25, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["dppo/sampled_logit_first_order_unmasked"],
        torch.tensor(-0.303125, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["dppo/sampled_logit_first_order_kept"],
        torch.tensor(-0.15, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["entropy/first_order_unmasked"],
        torch.tensor(-0.2875, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["entropy/first_order_kept"],
        torch.tensor(-0.15, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["entropy/first_order_upper_clipped"],
        torch.tensor(0.03125, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["entropy/first_order_lower_clipped"],
        torch.tensor(-0.16875, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics["entropy/first_order_unmasked"],
        metrics["entropy/first_order_kept"]
        + metrics["entropy/first_order_upper_clipped"]
        + metrics["entropy/first_order_lower_clipped"],
    )
    torch.testing.assert_close(
        metrics["entropy/first_order_mask_delta"],
        -metrics["entropy/first_order_upper_clipped"] - metrics["entropy/first_order_lower_clipped"],
    )
    torch.testing.assert_close(
        metrics["train_rollout_logprob_abs_diff"],
        (sampled_train.log() - sampled_rollout.log()).abs().mean(),
    )
    assert metrics["pg_clipfrac"].item() == 99.0
    assert all(torch.isfinite(value) for key, value in metrics.items() if key.startswith("dppo/"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
