"""Unit tests for spec-decoding / prefix-cache stat plumbing in ``_sample_for_turn``.

The kernel-agent custom generate builds each turn's Sample by hand and never
called ``update_from_meta_info``, so ``rollout/spec_accept_rate`` and
``rollout/prefix_cache_hit_rate`` were silently 0. ``_sample_for_turn`` now applies
the spec/prefix sub-updates from the engine meta_info; these tests pin that.
"""

from types import SimpleNamespace

from examples.kernel_agent.generate_with_cuda_agent import _sample_for_turn
from slime.utils.types import Sample


# Minimal env_result accepted by _extract_env_extra_info (irrelevant to the stat plumbing).
_ENV_RESULT = {
    "env_state": {
        "metadata": {},
        "correctness": True,
        "compiled": True,
        "speedup": 1.0,
        "decoy_kernel": False,
    }
}


def _call(meta_info, spec_algo):
    args = SimpleNamespace(sglang_speculative_algorithm=spec_algo)
    return _sample_for_turn(
        Sample(index=0, prompt="p", tokens=[1, 2]),
        prompt_ids=[1, 2],
        response="abc",
        response_ids=[3, 4, 5],
        log_probs=[-0.1, -0.2, -0.3],
        reward=0.0,
        status=Sample.Status.COMPLETED,
        turn_idx=0,
        env_result=_ENV_RESULT,
        args=args,
        meta_info=meta_info,
    )


_META = {
    # newer SGLang (>=0.5.12) spec field names
    "spec_num_correct_drafts": 6,
    "spec_num_proposed_drafts": 10,
    "spec_verify_ct": 3,
    "completion_tokens": 9,
    "cached_tokens": 4,
    "prompt_tokens": 12,
}


def test_spec_and_prefix_stats_populated_when_spec_enabled():
    s = _call(_META, spec_algo="EAGLE")
    # spec accept rate = correct / proposed
    assert s.spec_info.spec_accept_token_num == 6
    assert s.spec_info.spec_draft_token_num == 10
    assert abs(s.spec_info.spec_accept_rate - 0.6) < 1e-9
    # prefix cache hit rate = cached / prompt
    assert s.prefix_cache_info.cached_tokens == 4
    assert s.prefix_cache_info.total_prompt_tokens == 12
    assert abs(s.prefix_cache_info.prefix_cache_hit_rate - 4 / 12) < 1e-9


def test_spec_gated_off_but_prefix_cache_still_collected():
    s = _call(_META, spec_algo=None)
    # spec gated off -> untouched / zero
    assert s.spec_info.spec_draft_token_num == 0
    assert s.spec_info.spec_accept_rate == 0.0
    # prefix cache is collected regardless of spec algorithm
    assert s.prefix_cache_info.cached_tokens == 4
    assert s.prefix_cache_info.total_prompt_tokens == 12


def test_no_meta_info_is_noop():
    s = _call(None, spec_algo="EAGLE")
    assert s.spec_info.spec_draft_token_num == 0
    assert s.prefix_cache_info.total_prompt_tokens == 0


if __name__ == "__main__":
    test_spec_and_prefix_stats_populated_when_spec_enabled()
    test_spec_gated_off_but_prefix_cache_still_collected()
    test_no_meta_info_is_noop()
    print("OK: _sample_for_turn meta_info plumbing")
