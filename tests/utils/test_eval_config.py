from types import SimpleNamespace

from slime.utils.eval_config import build_eval_dataset_configs


def test_eval_dataset_config_resolves_prompt_and_context_lengths_from_args():
    args = SimpleNamespace(
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
    args = SimpleNamespace(
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
