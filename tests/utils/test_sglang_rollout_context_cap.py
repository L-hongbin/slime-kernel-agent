from types import SimpleNamespace

from slime.rollout.sglang_rollout import _cap_sampling_params_by_context
from slime.utils.http_utils import get_sglang_client_concurrency


def test_cap_sampling_params_leaves_one_token_context_headroom():
    sample = SimpleNamespace(index=7)
    prompt_ids = list(range(984))
    sampling_params = {
        "max_new_tokens": 32768,
        "_slime_max_context_len": 32768,
        "temperature": 1.0,
    }

    capped = _cap_sampling_params_by_context(sample, prompt_ids, sampling_params)

    assert capped["max_new_tokens"] == 31783
    assert capped["temperature"] == 1.0
    assert "_slime_max_context_len" not in capped


def test_cap_sampling_params_truncates_when_prompt_fills_context_window():
    sample = SimpleNamespace(index=8)
    prompt_ids = list(range(32767))
    sampling_params = {
        "max_new_tokens": 128,
        "_slime_max_context_len": 32768,
    }

    capped = _cap_sampling_params_by_context(sample, prompt_ids, sampling_params)

    assert capped["max_new_tokens"] == 0


def test_sglang_client_concurrency_caps_per_engine_by_max_running_requests():
    args = SimpleNamespace(
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=2,
        sglang_server_concurrency=128,
        sglang_max_running_requests=64,
    )

    assert get_sglang_client_concurrency(args) == 64 * 4

    args.sglang_server_concurrency = 32
    assert get_sglang_client_concurrency(args) == int(32 * 1.5) * 4

    args.sglang_max_running_requests = None
    assert get_sglang_client_concurrency(args) == int(32 * 1.5) * 4
