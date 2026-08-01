"""Launch-time contract tests for Direct Double-Sided Importance Sampling."""

import math
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.utils.arguments import _validate_dis_args

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "policy_loss_mode": "dis",
        "use_rollout_logprobs": True,
        "use_tis": False,
        "eps_clip": 0.8,
        "eps_clip_high": 3.0,
        "eps_clip_c": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_requested_dis_contract_is_accepted():
    _validate_dis_args(_args())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"use_rollout_logprobs": False}, "use-rollout-logprobs"),
        ({"use_tis": True}, "incompatible"),
        ({"eps_clip_c": 5.0}, "does not use"),
        ({"eps_clip": 0.0}, "eps-clip"),
        ({"eps_clip": 1.0}, "eps-clip"),
        ({"eps_clip_high": 0.0}, "eps-clip-high"),
        ({"eps_clip_high": math.inf}, "eps-clip-high"),
    ],
)
def test_invalid_dis_contract_fails_loudly(override, message):
    with pytest.raises(ValueError, match=message):
        _validate_dis_args(_args(**override))


def test_other_policy_modes_are_not_constrained():
    _validate_dis_args(
        _args(
            policy_loss_mode="ppo",
            use_rollout_logprobs=False,
            use_tis=True,
            eps_clip=2.0,
            eps_clip_high=0.0,
            eps_clip_c=5.0,
        )
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
