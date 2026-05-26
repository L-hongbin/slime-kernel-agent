from types import SimpleNamespace

from slime.rollout.sglang_rollout import _cap_sampling_params_by_context


def test_cap_sampling_params_by_context_caps_completion_to_remaining_tokens():
    sampling_params = {"max_new_tokens": 65536, "_slime_max_context_len": 65504}
    sample = SimpleNamespace(index=0)

    out = _cap_sampling_params_by_context(sample, [0] * 1327, sampling_params)

    # max_context_len - prompt_len - 1 = 65504 - 1327 - 1 = 64176 (one token headroom)
    assert out["max_new_tokens"] == 64176
    # Helper pops _slime_max_context_len from the original dict before deciding to cap
    assert "_slime_max_context_len" not in sampling_params
    assert "_slime_max_context_len" not in out
    # When capping, the helper returns a copy so the caller's original max_new_tokens
    # is preserved (slime relies on this for the multi-turn re-injection pattern).
    assert sampling_params["max_new_tokens"] == 65536


def test_cap_sampling_params_by_context_allows_zero_remaining_tokens():
    sampling_params = {"max_new_tokens": 1024, "_slime_max_context_len": 10}
    sample = SimpleNamespace(index=0)

    out = _cap_sampling_params_by_context(sample, [0] * 12, sampling_params)

    # max(0, 10 - 12 - 1) = 0
    assert out["max_new_tokens"] == 0


def test_cap_sampling_params_by_context_noops_without_private_context_key():
    sampling_params = {"max_new_tokens": 128}
    sample = SimpleNamespace(index=0)

    out = _cap_sampling_params_by_context(sample, [0] * 64, sampling_params)

    assert out == {"max_new_tokens": 128}


def test_cap_sampling_params_by_context_noop_when_remaining_above_request():
    # Plenty of room: prompt_len 100, context 4096 -> remaining 3995 >= max_new_tokens 1024
    sampling_params = {"max_new_tokens": 1024, "_slime_max_context_len": 4096}
    sample = SimpleNamespace(index=0)

    out = _cap_sampling_params_by_context(sample, [0] * 100, sampling_params)

    # No cap means no copy: max_new_tokens stays at 1024.
    assert out["max_new_tokens"] == 1024
    # _slime_max_context_len is popped regardless.
    assert "_slime_max_context_len" not in sampling_params
    assert "_slime_max_context_len" not in out
