import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.backends.megatron_utils import arguments as megatron_arguments
from slime.utils import wandb_utils

NUM_GPUS = 0


@pytest.mark.unit
def test_argument_table_redacts_nested_secrets_without_mutating_runtime_args(capsys):
    raw_secrets = {
        "WANDB_API_KEY": "raw-wandb-api-key",
        "HF_TOKEN": "raw-hf-token",
        "DB_PASSWORD": "raw-db-password",
        "CLIENT_SECRET": "raw-client-secret",
        "wandb_key": "raw-direct-wandb-key",
    }
    args = Namespace(
        train_env_vars={
            "WANDB_API_KEY": raw_secrets["WANDB_API_KEY"],
            "HF_TOKEN": raw_secrets["HF_TOKEN"],
            "PYTHONPATH": "/reviewable/python/path",
            "TOKENIZERS_PARALLELISM": "true",
            "nested": {
                "DB_PASSWORD": raw_secrets["DB_PASSWORD"],
                "CLIENT_SECRET": raw_secrets["CLIENT_SECRET"],
                "VISIBLE_SETTING": "still-visible",
            },
        },
        wandb_key=raw_secrets["wandb_key"],
        experiment_name="reviewable-experiment",
    )
    original = deepcopy(vars(args))

    installed_printer = megatron_arguments._megatron_arguments._print_args
    assert getattr(installed_printer, "_slime_secret_redacting", False)
    installed_printer("arguments", args)
    output = capsys.readouterr().out

    for raw_secret in raw_secrets.values():
        assert raw_secret not in output
    assert "WANDB_API_KEY" in output
    assert "<redacted>" in output
    assert "/reviewable/python/path" in output
    assert "TOKENIZERS_PARALLELISM" in output
    assert "still-visible" in output
    assert "reviewable-experiment" in output
    assert vars(args) == original


@pytest.mark.unit
def test_wandb_config_uses_same_non_mutating_redaction():
    args = Namespace(
        train_env_vars={"SERVICE_TOKEN": "raw-service-token", "NCCL_DEBUG": "INFO"},
        ordinary_value=17,
    )

    config = wandb_utils._args_to_config_dict(args)

    assert config == {
        "train_env_vars": {"SERVICE_TOKEN": "<redacted>", "NCCL_DEBUG": "INFO"},
        "ordinary_value": 17,
    }
    assert args.train_env_vars["SERVICE_TOKEN"] == "raw-service-token"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
