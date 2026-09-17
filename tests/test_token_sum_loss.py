"""MiniRL PG token-sum / rollout-count reduction, without changing the objective."""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.utils.arguments import get_slime_extra_args_provider, slime_validate_args

NUM_GPUS = 0


def test_minirl_cli_is_opt_in():
    parser = get_slime_extra_args_provider()(ArgumentParser())
    assert not parser.parse_args(["--rollout-batch-size", "1"]).calculate_token_sum_loss
    args = parser.parse_args(["--rollout-batch-size", "1", "--calculate-token-sum-loss"])
    assert args.calculate_token_sum_loss
    assert not args.calculate_per_token_loss


def test_prompt_mean_cli_is_opt_in():
    parser = get_slime_extra_args_provider()(ArgumentParser())
    assert not parser.parse_args(["--rollout-batch-size", "1"]).calculate_per_prompt_loss
    args = parser.parse_args(["--rollout-batch-size", "1", "--calculate-per-prompt-loss"])
    assert args.calculate_per_prompt_loss
    assert not args.calculate_per_token_loss
    assert not args.calculate_token_sum_loss


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"calculate_per_token_loss": True}, "token-mean"),
        ({"calculate_token_sum_loss": True}, "token-sum"),
        ({"custom_pg_loss_reducer_function_path": "custom.reducer"}, "custom PG"),
        ({"loss_type": "sft_loss"}, "policy_loss"),
        ({"train_backend": "fsdp"}, "megatron"),
    ],
)
def test_prompt_mean_rejects_conflicting_contracts(overrides, match):
    with pytest.raises(ValueError, match=match):
        slime_validate_args(Namespace(calculate_per_prompt_loss=True, **overrides))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"calculate_per_token_loss": True}, "calculate-per-token-loss"),
        ({"custom_pg_loss_reducer_function_path": "custom.reducer"}, "custom PG loss reducer"),
        ({"loss_type": "sft_loss"}, "policy_loss"),
        ({"loss_type": "custom_loss"}, "policy_loss"),
    ],
)
def test_minirl_rejects_conflicting_contracts(overrides, match):
    with pytest.raises(ValueError, match=match):
        slime_validate_args(Namespace(calculate_token_sum_loss=True, **overrides))


def _loss_fixture(monkeypatch, *, mode="minirl", same_trajectory=False, reject=False):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)
    lengths = [2, 4, 1]
    # Mixed signs, a masked token, and one fully masked padding sample.
    log_probs = torch.zeros(7, requires_grad=True)
    masks = [torch.ones(2), torch.tensor([1.0, 1.0, 0.0, 1.0]), torch.zeros(1)]
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 10.0])
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *a, **kw: (
            None,
            {"log_probs": list(log_probs.split(lengths)), "entropy": [torch.ones(n) for n in lengths]},
        ),
    )
    if reject:
        # Return an unmodified PG tensor to prove the reducer itself honors
        # rejection masks, not just zeroes supplied by the TIS implementation.
        monkeypatch.setattr(
            loss_module,
            "vanilla_tis_function",
            lambda **kw: (kw["pg_loss"], [masks[0], torch.zeros(4), masks[2]], {}),
        )
    args = Namespace(
        calculate_token_sum_loss=mode == "minirl",
        calculate_per_prompt_loss=mode == "prompt",
        calculate_per_token_loss=mode == "per_token",
        loss_type="policy_loss",
        recompute_loss_function=False,
        allgather_cp=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        advantage_estimator="grpo",
        policy_loss_mode="ppo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        get_mismatch_metrics=False,
        use_tis=reject,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        qkv_format="thd",
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    denoms = [5.0, 5.0, 5.0] if same_trajectory else [2.0, 3.0, 0.0]
    batch = {
        "advantages": list(advantages.split(lengths)),
        "log_probs": [torch.zeros(n) for n in lengths],
        "rollout_log_probs": [torch.zeros(n) for n in lengths],
        "total_lengths": lengths,
        "response_lengths": lengths,
        "unconcat_tokens": [torch.zeros(n, dtype=torch.long) for n in lengths],
        "loss_masks": masks,
        "rollout_mask_sums": [torch.tensor(d) for d in denoms],
    }
    return args, batch, log_probs


@pytest.mark.parametrize("mode,expected", [("default", 0.0), ("minirl", 1.0), ("per_token", 1.0)])
def test_policy_reduction_keeps_existing_modes_and_metrics(monkeypatch, mode, expected):
    args, batch, log_probs = _loss_fixture(monkeypatch, mode=mode)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["rollout_mask_sums"],
        args.calculate_per_token_loss,
    )
    loss, metrics = loss_module.policy_loss_function(args, batch, torch.zeros(1, 7, 2), reducer)
    assert loss.item() == pytest.approx(expected)
    assert metrics["pg_loss"].item() == pytest.approx(expected)
    assert metrics["entropy_loss"].item() == pytest.approx(5.0 if mode == "per_token" else 2.0)
    loss.backward()
    expected_grad = torch.tensor([-1.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    if mode == "default":
        expected_grad[:2] /= 2
        expected_grad[2:6] /= 3
    torch.testing.assert_close(log_probs.grad, expected_grad)


@pytest.mark.parametrize("same_trajectory", [False, True])
@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_minirl_outer_normalizer_is_rollouts_not_tokens_or_turns(
    monkeypatch, same_trajectory, reject, num_microbatches
):
    args, batch, log_probs = _loss_fixture(monkeypatch, same_trajectory=same_trajectory, reject=reject)
    rollout_count = 1 if same_trajectory else 2
    loss, normalizer, _ = loss_module.loss_function(args, batch, num_microbatches, rollout_count, torch.zeros(1, 7, 2))
    assert normalizer == 1
    # Megatron's non-token contract divides each microbatch contribution by
    # num_microbatches, cancelling slime's prescale. Padding does not add loss.
    loss = loss / num_microbatches
    assert loss.item() == pytest.approx((-2.0 if reject else 1.0) / rollout_count)
    loss.backward()
    expected_grad = torch.tensor([-1.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    if reject:
        expected_grad[2:] = 0
    torch.testing.assert_close(log_probs.grad, expected_grad / rollout_count)


def test_minirl_turns_split_across_microbatches_match_unsplit(monkeypatch):
    args, batch, log_probs = _loss_fixture(monkeypatch, same_trajectory=True)
    total_loss = 0
    lengths = batch["response_lengths"]
    for i in range(3):
        mb = {key: [value[i]] for key, value in batch.items()}
        monkeypatch.setattr(
            loss_module,
            "get_log_probs_and_entropy",
            lambda *a, i=i, **kw: (
                None,
                {"log_probs": [log_probs.split(lengths)[i]], "entropy": [torch.ones(lengths[i])]},
            ),
        )
        loss, normalizer, _ = loss_module.loss_function(args, mb, 3, 1, torch.zeros(1, lengths[i], 2))
        assert normalizer == 1
        total_loss = total_loss + loss / 3
    assert total_loss.item() == pytest.approx(1.0)
    total_loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.tensor([-1.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0]))


@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_prompt_mean_outer_scaling_gradients_and_auxiliary_losses(monkeypatch, reject, split):
    args, batch, log_probs = _loss_fixture(monkeypatch, mode="prompt", reject=reject)
    # One prompt, two trajectories plus a fully masked padding sample.
    batch["prompt_mask_sums"] = torch.tensor([5.0, 5.0, 5.0])
    batch["prompt_loss_scales"] = torch.tensor([2.0, 2.0, 2.0])
    args.entropy_coef = 0.25
    lengths = batch["response_lengths"]
    nmb = 3 if split else 1
    total_loss = 0
    for indices in ([[0], [1], [2]] if split else [[0, 1, 2]]):
        mb = {key: [value[i] for i in indices] for key, value in batch.items()}
        monkeypatch.setattr(
            loss_module,
            "get_log_probs_and_entropy",
            lambda *a, indices=indices, **kw: (
                None,
                {
                    "log_probs": [log_probs.split(lengths)[i] for i in indices],
                    "entropy": [torch.ones(lengths[i]) for i in indices],
                },
            ),
        )
        if reject:
            monkeypatch.setattr(
                loss_module,
                "vanilla_tis_function",
                lambda indices=indices, **kw: (
                    kw["pg_loss"],
                    [torch.zeros_like(batch["loss_masks"][i]) if i == 1 else batch["loss_masks"][i] for i in indices],
                    {},
                ),
            )
        loss, normalizer, _ = loss_module.loss_function(args, mb, nmb, 2, torch.zeros(1, 7, 2))
        assert normalizer == 1
        total_loss = total_loss + loss / nmb
    # PG uses prompt token mean; entropy still uses trajectory mean. Rejection
    # changes the numerator, never the original five-token prompt denominator.
    assert total_loss.item() == pytest.approx((-2.0 if reject else 1.0) / 5 - (0.125 if reject else 0.25))
    total_loss.backward()
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0]) / 5
    if reject:
        expected[2:] = 0
    torch.testing.assert_close(log_probs.grad, expected)


def test_prompt_mean_requires_metadata(monkeypatch):
    args, batch, _ = _loss_fixture(monkeypatch, mode="prompt")
    with pytest.raises(ValueError, match="prompt_mask_sums"):
        loss_module.loss_function(args, batch, 1, 2, torch.zeros(1, 7, 2))


def test_minirl_shell_mode_does_not_enable_per_token():
    script = Path(__file__).resolve().parents[1] / "examples/kernel_agent/run_qwen3.6_27B_full_async_dppo.sh"
    source = script.read_text()
    assert 'CALC_LOSS_MODE="${CALC_LOSS_MODE:-"PerToken"}"' in source
    assert "TokenSum) EXP_ARGS+=(--calculate-token-sum-loss) ;;" in source
    assert "PerPrompt) EXP_ARGS+=(--calculate-per-prompt-loss) ;;" in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
