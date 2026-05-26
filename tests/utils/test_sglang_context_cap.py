from slime.rollout.sglang_rollout import _cap_sampling_params_by_context


def test_cap_sampling_params_by_context_caps_completion_to_remaining_tokens():
    sampling_params = {"max_new_tokens": 65536, "_slime_max_context_len": 65504}

    _cap_sampling_params_by_context(1327, sampling_params)

    assert sampling_params["max_new_tokens"] == 64177
    assert "_slime_max_context_len" not in sampling_params


def test_cap_sampling_params_by_context_allows_zero_remaining_tokens():
    sampling_params = {"max_new_tokens": 1024, "_slime_max_context_len": 10}

    _cap_sampling_params_by_context(12, sampling_params)

    assert sampling_params["max_new_tokens"] == 0


def test_cap_sampling_params_by_context_noops_without_private_context_key():
    sampling_params = {"max_new_tokens": 128}

    _cap_sampling_params_by_context(64, sampling_params)

    assert sampling_params == {"max_new_tokens": 128}
