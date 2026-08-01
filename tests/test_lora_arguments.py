"""CPU-only validation tests for the model-independent LoRA CLI contract."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.utils.arguments import _validate_lora_args

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "lora_dim": 0,
        "lora_alpha": None,
        "lora_dropout": 0.0,
        "lora_rslora": False,
        "lora_plus_lambda": None,
        "dsv4_lora_shared_expert": False,
        "lora_adapter_resume_load": "",
        "lora_checkpoint_max_node_bytes": 2 * 1024**3,
        "use_lora_weight_sync": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_lora_validation_accepts_disabled_and_enabled_contracts():
    _validate_lora_args(_args())
    _validate_lora_args(_args(lora_plus_lambda=1.0))
    _validate_lora_args(
        _args(
            lora_dim=32,
            lora_alpha=32,
            lora_dropout=0.1,
            lora_rslora=True,
            lora_plus_lambda=4.0,
            dsv4_lora_shared_expert=True,
            lora_adapter_resume_load="/tmp/adapter",
            use_lora_weight_sync=True,
        )
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"lora_dim": -1}, "lora-dim"),
        ({"lora_dim": 4, "lora_alpha": 0}, "lora-alpha"),
        ({"lora_dropout": -0.1}, "lora-dropout"),
        ({"lora_dim": 4, "lora_dropout": 1.0}, "lora-dropout"),
        ({"lora_plus_lambda": 0.0}, "lora-plus-lambda"),
        ({"lora_checkpoint_max_node_bytes": 0}, "lora-checkpoint-max-node-bytes"),
    ],
)
def test_lora_validation_rejects_invalid_values(override, message):
    with pytest.raises(ValueError, match=message):
        _validate_lora_args(_args(**override))


@pytest.mark.parametrize(
    "override",
    [
        {"lora_alpha": 32},
        {"lora_dropout": 0.1},
        {"lora_rslora": True},
        {"lora_plus_lambda": 4.0},
        {"dsv4_lora_shared_expert": True},
        {"lora_adapter_resume_load": "/tmp/adapter"},
        {"use_lora_weight_sync": True},
    ],
)
def test_lora_validation_requires_positive_rank_for_active_options(override):
    with pytest.raises(ValueError, match="require --lora-dim"):
        _validate_lora_args(_args(**override))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
