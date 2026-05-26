from types import SimpleNamespace

import pytest

from slime.utils.eval_config import build_eval_dataset_configs, ensure_dataset_list


def _args(**overrides):
    defaults = {
        "n_samples_per_eval_prompt": 1,
        "n_samples_per_prompt": 1,
        "eval_temperature": None,
        "rollout_temperature": None,
        "eval_top_p": None,
        "rollout_top_p": None,
        "eval_top_k": None,
        "rollout_top_k": None,
        "eval_max_response_len": None,
        "rollout_max_response_len": None,
        "eval_max_prompt_len": None,
        "rollout_max_prompt_len": None,
        "eval_max_context_len": None,
        "rollout_max_context_len": None,
        "eval_input_key": None,
        "input_key": None,
        "eval_label_key": None,
        "label_key": None,
        "eval_tool_key": None,
        "tool_key": None,
        "metadata_key": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.unit
def test_eval_config_resolves_prompt_response_and_context_lengths_from_defaults():
    raw_datasets = ensure_dataset_list(
        [
            {
                "name": "kernelbench_level1",
                "path": "data/kernelbench-level1-validation/train.parquet",
                "input_key": "ground_truth",
                "label_key": "ground_truth",
                "metadata_key": "extra_info",
            }
        ]
    )
    defaults = {
        "max_prompt_len": 32768,
        "max_response_len": 32768,
        "max_context_len": 32768,
    }

    dataset = build_eval_dataset_configs(_args(), raw_datasets, defaults)[0]

    assert dataset.max_prompt_len == 32768
    assert dataset.max_response_len == 32768
    assert dataset.max_context_len == 32768


def test_eval_dataset_config_resolves_prompt_and_context_lengths_from_args():
    args = _args(
        eval_max_prompt_len=4096,
        rollout_max_prompt_len=8191,
        eval_max_context_len=8192,
        rollout_max_context_len=16384,
        eval_max_response_len=512,
        rollout_max_response_len=1024,
    )

    dataset = build_eval_dataset_configs(args, [{"name": "kernelbench", "path": "eval.jsonl"}], {})[0]

    assert dataset.max_prompt_len == 4096
    assert dataset.max_context_len == 8192
    assert dataset.max_response_len == 512


def test_eval_dataset_config_length_overrides_precede_cli_args_and_cache_key():
    args = _args(
        eval_max_prompt_len=4096,
        rollout_max_prompt_len=8191,
        eval_max_context_len=8192,
        rollout_max_context_len=16384,
        eval_max_response_len=512,
        rollout_max_response_len=1024,
    )

    dataset = build_eval_dataset_configs(
        args,
        [{"name": "kernelbench", "path": "eval.jsonl", "max_prompt_len": 2048}],
        {"max_prompt_len": 3072, "max_context_len": 6144, "max_response_len": 768},
    )[0]

    assert dataset.max_prompt_len == 2048
    assert dataset.max_context_len == 6144
    assert dataset.max_response_len == 768
    assert 2048 in dataset.cache_key
