"""Guard: rollout_temperature == 0 (greedy rollout) must not divide logits by zero.

Root cause of the R6 full-loop first-backward NaN: the smoke passed
``--rollout-temperature 0`` (greedy sampling) and the train-side log-prob path
divided logits by the rollout temperature to match rollout-time log-probs,
producing Inf logits -> NaN log_softmax -> NaN loss -> NaN grads
("found NaN in local grad norm for bucket #0" on the loss-computing PP stage).

Policy: training requires rollout_temperature > 0 (use 1 for on-policy RL).
- ``slime_validate_args`` rejects temperature <= 0 at launch (unless
  ``--debug-rollout-only``, which has no train side).
- The two loss.py scaling sites assert temperature > 0 as defense in depth.
"""

from argparse import Namespace
from unittest.mock import patch

import pytest
import torch

from slime.backends.megatron_utils import loss as loss_mod
from slime.utils.arguments import slime_validate_args


def _run_get_responses(temperature: float) -> torch.Tensor:
    torch.manual_seed(0)
    total, response, vocab = 6, 3, 11
    logits = torch.randn(1, total, vocab, dtype=torch.float32)
    tokens = torch.randint(0, vocab, (total,), dtype=torch.int64)
    args = Namespace(rollout_temperature=temperature, qkv_format="thd")

    with patch.object(loss_mod.mpu, "get_context_parallel_world_size", return_value=1):
        chunks = list(
            loss_mod.get_responses(
                logits,
                args=args,
                unconcat_tokens=[tokens],
                total_lengths=[total],
                response_lengths=[response],
            )
        )
    assert len(chunks) == 1
    logits_chunk, _ = chunks[0]
    return logits_chunk


def test_temperature_zero_asserts_in_log_prob_path():
    with pytest.raises(AssertionError, match="rollout_temperature must be > 0"):
        _run_get_responses(0.0)


def test_temperature_one_yields_finite_unscaled_logits():
    logits_chunk = _run_get_responses(1.0)
    assert torch.isfinite(logits_chunk).all()
    log_probs = torch.log_softmax(logits_chunk, dim=-1)
    assert torch.isfinite(log_probs).all()


@pytest.mark.parametrize("temperature", [0.5, 2.0])
def test_positive_temperature_still_scales(temperature):
    scaled = _run_get_responses(temperature)
    unscaled = _run_get_responses(1.0)
    assert torch.allclose(scaled, unscaled / temperature)


def _minimal_validate_args(temperature: float, debug_rollout_only: bool) -> Namespace:
    # Only the fields slime_validate_args touches before/at the temperature check.
    return Namespace(
        rollout_temperature=temperature,
        debug_rollout_only=debug_rollout_only,
        sequence_mis_config=None,
        sequence_mis_lower=None,
        sequence_mis_upper=None,
        sequence_mis_token_veto_threshold=None,
        sequence_mis_use_advantage=False,
        sequence_mis_aggregation=None,
    )


def test_validate_args_rejects_temperature_zero_for_training():
    args = _minimal_validate_args(0.0, debug_rollout_only=False)
    with pytest.raises(ValueError, match="--rollout-temperature must be > 0 for training"):
        slime_validate_args(args)


def test_validate_args_allows_temperature_zero_for_rollout_only_debug():
    args = _minimal_validate_args(0.0, debug_rollout_only=True)
    # Must get past the temperature check; later validations may need more
    # fields, so only assert the temperature check itself does not raise.
    try:
        slime_validate_args(args)
    except ValueError as e:
        assert "--rollout-temperature" not in str(e)
    except AttributeError:
        pass  # later, unrelated validations touching absent fields
