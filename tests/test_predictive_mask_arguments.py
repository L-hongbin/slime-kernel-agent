"""Launch-time distribution-consistency guards for predictive Top-K DPPO."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.utils.arguments import _validate_dppo_predictive_args

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "policy_loss_mode": "dppo_topk_kl_predictive",
        "dppo_predictive_top_k": 20,
        "vocab_size": 129280,
        "use_rollout_logprobs": True,
        "use_tis": False,
        "eps_clip": 0.15,
        "eps_clip_high": 0.15,
        "rollout_temperature": 1.0,
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
        "allgather_cp": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_paper_configuration_is_accepted():
    _validate_dppo_predictive_args(_args())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dppo_predictive_top_k": 0}, "positive integer"),
        ({"dppo_predictive_top_k": 129280}, "smaller than vocab_size"),
        ({"use_rollout_logprobs": False}, "use-rollout-logprobs"),
        ({"use_tis": True}, "incompatible"),
        ({"eps_clip_high": 0.2}, "same positive value"),
        ({"rollout_temperature": 0.8}, "temperature 1"),
        ({"rollout_top_p": 0.95}, "top-p 1"),
        ({"rollout_top_k": 50}, "top-k -1"),
        ({"allgather_cp": True}, "allgather-cp"),
    ],
)
def test_mismatched_distributions_fail_loud(override, message):
    with pytest.raises(ValueError, match=message):
        _validate_dppo_predictive_args(_args(**override))


def test_non_predictive_modes_remain_unconstrained():
    _validate_dppo_predictive_args(
        _args(
            policy_loss_mode="ppo",
            dppo_predictive_top_k=0,
            use_rollout_logprobs=False,
            rollout_temperature=0.7,
            rollout_top_p=0.9,
            rollout_top_k=40,
            allgather_cp=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
