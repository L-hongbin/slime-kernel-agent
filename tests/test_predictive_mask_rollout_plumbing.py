from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_AGENT_ROOT = REPO_ROOT / "examples" / "kernel_agent"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(KERNEL_AGENT_ROOT))

import generate_with_cuda_agent as cuda_agent

from slime.backends.megatron_utils import cp_utils
from slime.backends.megatron_utils.actor import _slice_predictive_support_with_cp
from slime.ray.rollout import RolloutManager
from slime.rollout import sglang_rollout
from slime.rollout.sglang_rollout import _append_predictive_support, _extract_predictive_support
from slime.utils.arguments import get_slime_extra_args_provider, slime_validate_args
from slime.utils.types import Sample

NUM_GPUS = 0


def _meta_info() -> dict:
    return {
        "output_token_logprobs": [[-0.25, 7, "seven"], [-1.5, 99, "sample"]],
        "output_top_logprobs": [
            [[-0.2, 7, "seven"], [-0.7, 8, "eight"]],
            [[-0.1, 3, "three"], [-0.2, 4, "four"]],
        ],
        "finish_reason": {"type": "stop"},
        "cached_tokens": 0,
        "prompt_tokens": 2,
    }


@pytest.mark.unit
def test_extract_predictive_support_is_compact_unique_and_uses_sampled_logprob():
    sampled_ids, sampled_lps, token_ids, log_probs, valid = _extract_predictive_support(_meta_info(), top_k=2)

    assert sampled_ids == [7, 99]
    assert sampled_lps == [-0.25, -1.5]
    assert token_ids.dtype == np.int32
    assert log_probs.dtype == np.float32
    assert valid.dtype == np.bool_
    assert token_ids.shape == log_probs.shape == valid.shape == (2, 3)

    # Sample 7 was already top-k: it appears once and the sampled-logprob copy
    # wins over the independently returned/rounded top-k value.
    np.testing.assert_array_equal(token_ids[0], [7, 8, 0])
    np.testing.assert_array_equal(valid[0], [True, True, False])
    assert log_probs[0, 0] == pytest.approx(-0.25)
    assert np.count_nonzero(token_ids[0, valid[0]] == 7) == 1

    # Sample 99 was outside top-k and occupies the extra K+1 slot.
    np.testing.assert_array_equal(token_ids[1], [3, 4, 99])
    np.testing.assert_array_equal(valid[1], [True, True, True])
    assert log_probs[1, 2] == pytest.approx(-1.5)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m.pop("output_top_logprobs"), "token-count mismatch"),
        (lambda m: m["output_top_logprobs"][0].pop(), "exactly 2 entries"),
        (lambda m: m["output_top_logprobs"][0].__setitem__(1, [-0.3, 7]), "duplicate token id"),
        (lambda m: m["output_token_logprobs"][0].__setitem__(0, float("nan")), "non-finite"),
    ],
)
def test_extract_predictive_support_rejects_missing_malformed_or_nan(mutate, match):
    meta = _meta_info()
    mutate(meta)
    with pytest.raises(ValueError, match=match):
        _extract_predictive_support(meta, top_k=2)


@pytest.mark.unit
def test_append_predictive_support_allows_only_masked_missing_prefix():
    sample = Sample(response_length=2, loss_mask=[0, 0])
    support = _extract_predictive_support(
        {
            "output_token_logprobs": [[-0.4, 5]],
            "output_top_logprobs": [[[-0.2, 1], [-0.3, 2]]],
        },
        top_k=2,
    )[2:]
    _append_predictive_support(sample, support, previous_response_length=2, top_k=2)
    assert sample.rollout_topk_token_ids.shape == (3, 3)
    assert not sample.rollout_topk_valid_mask[:2].any()
    assert sample.rollout_topk_valid_mask[2].all()

    trainable = Sample(response_length=1, loss_mask=[1])
    with pytest.raises(ValueError, match="existing trainable response tokens"):
        _append_predictive_support(trainable, support, previous_response_length=1, top_k=2)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, prompt, add_special_tokens=False):
        return [10, 11]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10, 11] if text == "prompt" else [7, 99]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "prompt"

    def decode(self, token_ids, skip_special_tokens=False):
        return "decoded"


class _GenerateState:
    tokenizer = _Tokenizer()
    processor = None
    active_lora_name = None
    apply_chat_template_kwargs = {}
    multi_turn_templates = {}


class _ContinuationTokenizer:
    eos_token_id = 248046
    pad_token_id = 248044
    eos_token = "<|im_end|>"
    pad_token = "<|endoftext|>"

    def __init__(self):
        self.encoded_texts = []

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return self.eos_token_id

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        self.encoded_texts.append(text)
        if text == "prompt":
            return {"input_ids": [1, 2]}
        if text == "</think>":
            return {"input_ids": [91]}
        if text == "\n</think>\n\n":
            return {"input_ids": [90, 91, 92]}
        return {"input_ids": [100, 101]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "prompt"

    def decode(self, token_ids, skip_special_tokens=False):
        return "decoded"


class _TokenHistoryGenerateState:
    tokenizer = _ContinuationTokenizer()
    processor = None
    active_lora_name = None
    apply_chat_template_kwargs = {"enable_thinking": True}
    multi_turn_template = None

    @staticmethod
    def _is_qwen3_5_model():
        return True


@pytest.mark.unit
def test_preserve_history_thinking_cli_enables_only_exact_history_mode():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())

    args = parser.parse_args(["--rollout-batch-size", "1", "--preserve-history-thinking"])

    assert args.preserve_history_thinking is True
    with pytest.raises(ValueError, match="requires --use-multi-turn"):
        slime_validate_args(args)

    removed_option = "--preserve-history-" + "token-prefix"
    with pytest.raises(SystemExit):
        parser.parse_args(["--rollout-batch-size", "1", removed_option])


@pytest.mark.unit
def test_preserve_history_thinking_does_not_inject_chat_template_kwargs(monkeypatch):
    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)
    args = SimpleNamespace(
        hf_checkpoint="unused",
        apply_chat_template_kwargs={"enable_thinking": False},
        preserve_history_thinking=True,
        multi_turn_prompt_config_path=None,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_server_concurrency=1,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=16,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        sglang_dp_size=1,
    )
    state = object.__new__(sglang_rollout.GenerateState)

    sglang_rollout.GenerateState.__init__(state, args)

    assert state.apply_chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.unit
def test_qwen_token_history_preserves_complete_generated_prefix():
    tokenizer = _ContinuationTokenizer()
    prompt_ids = [1, 2]
    response_ids = [3, 91, 4, tokenizer.eos_token_id]

    next_ids, exact_prefix_tokens, inserted_close = cuda_agent._build_qwen_next_turn_prompt_ids(
        tokenizer,
        prompt_ids,
        response_ids,
        "feedback",
        enable_thinking=True,
    )

    assert next_ids[: len(prompt_ids + response_ids)] == prompt_ids + response_ids
    assert exact_prefix_tokens == len(prompt_ids + response_ids)
    assert inserted_close is False


@pytest.mark.unit
@pytest.mark.parametrize("terminal_eos", [False, True])
def test_qwen_token_history_closes_truncated_thinking_without_reencoding_response(terminal_eos):
    tokenizer = _ContinuationTokenizer()
    prompt_ids = [1, 2]
    response_ids = [3, 4] + ([tokenizer.eos_token_id] if terminal_eos else [])

    next_ids, exact_prefix_tokens, inserted_close = cuda_agent._build_qwen_next_turn_prompt_ids(
        tokenizer,
        prompt_ids,
        response_ids,
        "feedback",
        enable_thinking=True,
    )

    assert next_ids[: len(prompt_ids) + 2] == prompt_ids + [3, 4]
    assert next_ids[len(prompt_ids) + 2 : len(prompt_ids) + 5] == [90, 91, 92]
    assert next_ids[len(prompt_ids) + 5] == tokenizer.eos_token_id
    assert exact_prefix_tokens == len(prompt_ids) + len(response_ids) - int(terminal_eos)
    assert inserted_close is True


@pytest.mark.unit
def test_cuda_agent_consistent_hashing_uses_sample_session_id():
    sample = Sample(session_id="trajectory-7")
    args = SimpleNamespace(router_policy="consistent_hashing")

    assert cuda_agent._sglang_routing_headers(args, sample) == {"X-SMG-Routing-Key": "trajectory-7"}
    assert cuda_agent._sglang_routing_headers(SimpleNamespace(router_policy="round_robin"), sample) is None


@pytest.mark.unit
def test_generate_group_assigns_unique_session_ids_before_custom_generation(monkeypatch):
    seen_session_ids = []

    async def fake_generate_and_rm(args, sample, sampling_params, evaluation=False):
        seen_session_ids.append(sample.session_id)
        return sample

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: SimpleNamespace(aborted=False))
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate_and_rm)
    group = [Sample(prompt="a"), Sample(prompt="b")]
    args = SimpleNamespace(
        sglang_enable_deterministic_inference=False,
        use_multi_turn=False,
        group_rm=False,
    )

    result = asyncio.run(sglang_rollout.generate_and_rm_group(args, group, {"max_new_tokens": 1}))

    assert result == group
    assert all(seen_session_ids)
    assert len(set(seen_session_ids)) == len(group)


@pytest.mark.unit
def test_cuda_agent_two_turn_generation_uses_exact_history_and_same_session_header(monkeypatch):
    payloads = []
    captured_headers = []

    async def recording_post(url, payload, max_retries=None, headers=None):
        payloads.append(payload)
        captured_headers.append(headers)
        return {"text": "answer", "meta_info": _meta_info()}

    async def fake_env(args, sample, response, turn_idx):
        return {"env_state": {"done": turn_idx == 1}}

    async def fake_reward(*args, **kwargs):
        return 1.0

    monkeypatch.setattr(cuda_agent, "GenerateState", lambda args: _TokenHistoryGenerateState())
    monkeypatch.setattr(cuda_agent, "post", recording_post)
    monkeypatch.setattr(cuda_agent, "cuda_kernel_env", fake_env)
    monkeypatch.setattr(cuda_agent, "reward_func", fake_reward)
    monkeypatch.setattr(cuda_agent, "_extract_env_extra_info", lambda result: {})
    monkeypatch.setattr(cuda_agent, "postprocess_turn_samples", lambda args, samples, finish_reason: samples)
    args = SimpleNamespace(
        max_turns=2,
        use_multi_turn=True,
        padding_turns=False,
        preserve_history_thinking=True,
        rollout_max_context_len=128,
        sglang_speculative_algorithm=None,
        dppo_predictive_top_k=0,
        use_rollout_routing_replay=False,
        use_lora_weight_sync=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="consistent_hashing",
    )
    sample = Sample(prompt="hello", session_id="trajectory-12")

    result = asyncio.run(cuda_agent._generate_impl(args, sample, {"max_new_tokens": 2}))

    assert len(payloads) == 2
    assert payloads[0]["input_ids"] == [1, 2]
    assert payloads[1]["input_ids"][:4] == [1, 2, 7, 99]
    assert payloads[1]["input_ids"][4:8] == [90, 91, 92, _ContinuationTokenizer.eos_token_id]
    assert captured_headers == [
        {"X-SMG-Routing-Key": "trajectory-12"},
        {"X-SMG-Routing-Key": "trajectory-12"},
    ]
    assert len(result) == 2
    assert result[1].metadata["history_exact_prefix_tokens"] == 4
    assert result[1].metadata["history_inserted_close_think"] is True


def _default_generate_args() -> SimpleNamespace:
    return SimpleNamespace(
        ci_test=False,
        dppo_predictive_top_k=2,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        use_rollout_routing_replay=False,
        use_lora_weight_sync=False,
        sglang_speculative_algorithm=None,
    )


@pytest.mark.unit
def test_default_sglang_generate_requests_and_captures_top_logprobs(monkeypatch):
    captured = {}

    async def fake_post(url, payload, headers=None):
        captured.update(payload)
        return {"text": "answer", "meta_info": _meta_info()}

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: _GenerateState())
    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    sample = Sample(prompt="hello")
    result = asyncio.run(sglang_rollout.generate(_default_generate_args(), sample, {"max_new_tokens": 2}))

    assert captured["return_logprob"] is True
    assert captured["top_logprobs_num"] == 2
    assert result.tokens == [10, 11, 7, 99]
    assert result.rollout_log_probs == [-0.25, -1.5]
    assert result.rollout_topk_token_ids.shape == (2, 3)
    assert result.rollout_topk_log_probs.dtype == np.float32
    assert result.rollout_topk_valid_mask.dtype == np.bool_


@pytest.mark.unit
def test_cuda_agent_generate_requests_and_captures_top_logprobs(monkeypatch):
    captured = {}

    async def fake_post(url, payload, max_retries=None, headers=None):
        captured.update(payload)
        captured["headers"] = headers
        return {"text": "answer", "meta_info": _meta_info()}

    async def fake_env(*args, **kwargs):
        return {"env_state": {"done": True}}

    async def fake_reward(*args, **kwargs):
        return 1.0

    monkeypatch.setattr(cuda_agent, "GenerateState", lambda args: _GenerateState())
    monkeypatch.setattr(cuda_agent, "post", fake_post)
    monkeypatch.setattr(cuda_agent, "cuda_kernel_env", fake_env)
    monkeypatch.setattr(cuda_agent, "reward_func", fake_reward)
    monkeypatch.setattr(cuda_agent, "_extract_env_extra_info", lambda result: {})
    monkeypatch.setattr(cuda_agent, "postprocess_turn_samples", lambda args, samples, finish_reason: samples)

    args = SimpleNamespace(
        max_turns=1,
        use_multi_turn=True,
        padding_turns=False,
        rollout_max_context_len=None,
        sglang_speculative_algorithm=None,
        dppo_predictive_top_k=2,
        use_rollout_routing_replay=False,
        use_lora_weight_sync=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="consistent_hashing",
    )
    result = asyncio.run(
        cuda_agent._generate_impl(
            args,
            Sample(prompt="hello", session_id="trajectory-11"),
            {"max_new_tokens": 2},
        )
    )

    assert captured["top_logprobs_num"] == 2
    assert captured["headers"] == {"X-SMG-Routing-Key": "trajectory-11"}
    assert len(result) == 1
    turn = result[0]
    assert turn.rollout_topk_token_ids.shape == (2, 3)
    assert turn.rollout_topk_token_ids.dtype == np.int32
    assert turn.rollout_topk_valid_mask[1].all()


def _make_manager(top_k: int = 2):
    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = lambda _args, samples: (
        [sample.reward for sample in samples],
        [sample.reward for sample in samples],
    )
    manager.args = SimpleNamespace(
        enable_turns_dp_partitions=False,
        reward_key=None,
        use_multi_turn=False,
        use_rollout_routing_replay=False,
        dppo_predictive_top_k=top_k,
    )
    return manager


def _supported_sample() -> Sample:
    sampled_ids, sampled_lps, token_ids, log_probs, valid = _extract_predictive_support(_meta_info(), top_k=2)
    return Sample(
        index=0,
        reward=1.0,
        tokens=[10, 11, *sampled_ids],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=sampled_lps,
        rollout_topk_token_ids=token_ids,
        rollout_topk_log_probs=log_probs,
        rollout_topk_valid_mask=valid,
    )


@pytest.mark.unit
def test_rollout_conversion_keeps_numpy_support_and_rejects_missing_train_data():
    manager = _make_manager()
    sample = _supported_sample()
    train_data = manager._convert_samples_to_train_data([sample])

    assert train_data["rollout_topk_token_ids"][0] is sample.rollout_topk_token_ids
    assert train_data["rollout_topk_log_probs"][0].dtype == np.float32
    assert train_data["rollout_topk_valid_mask"][0].dtype == np.bool_

    missing = _supported_sample()
    missing.rollout_topk_log_probs = None
    with pytest.raises(ValueError, match="incomplete predictive support"):
        manager._convert_samples_to_train_data([missing])

    all_missing = _supported_sample()
    all_missing.rollout_topk_token_ids = None
    all_missing.rollout_topk_log_probs = None
    all_missing.rollout_topk_valid_mask = None
    with pytest.raises(ValueError, match="trainable response tokens"):
        manager._convert_samples_to_train_data([all_missing])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda sample: sample.rollout_topk_log_probs.__setitem__((0, 0), np.nan),
            "non-finite logprobs",
        ),
        (
            lambda sample: sample.rollout_topk_token_ids.__setitem__((0, 1), 7),
            "duplicate token ids",
        ),
        (
            lambda sample: sample.rollout_topk_valid_mask.__setitem__((0, 1), False),
            "valid support entries",
        ),
        (
            lambda sample: sample.rollout_topk_token_ids.__setitem__((0, 0), 6),
            "sampled token id 7 exactly once",
        ),
        (
            lambda sample: sample.rollout_log_probs.__setitem__(0, -9.0),
            "sampled-token logprob mismatch",
        ),
    ],
)
def test_rollout_conversion_vectorized_validation_rejects_corrupt_rows(mutate, match):
    sample = _supported_sample()
    mutate(sample)
    with pytest.raises(ValueError, match=match):
        _make_manager()._convert_samples_to_train_data([sample])


@pytest.mark.unit
def test_rollout_conversion_normalizes_masked_pad_to_all_invalid_numpy_support():
    pad = Sample(index=0, reward=0.0, tokens=[0], response_length=1, loss_mask=[0], remove_sample=True)
    train_data = _make_manager()._convert_samples_to_train_data([pad])
    assert train_data["rollout_topk_token_ids"][0].shape == (1, 3)
    assert train_data["rollout_topk_token_ids"][0].dtype == np.int32
    assert not train_data["rollout_topk_valid_mask"][0].any()
    assert train_data["rollout_log_probs"] == [[0.0]]


@pytest.mark.unit
def test_zero_length_masked_first_sample_does_not_drop_later_rollout_logprobs():
    empty = Sample(index=0, reward=0.0, tokens=[10], response_length=0, loss_mask=[], remove_sample=True)
    trainable = _supported_sample()
    trainable.index = 1
    train_data = _make_manager()._convert_samples_to_train_data([empty, trainable])
    assert train_data["rollout_log_probs"] == [[], [-0.25, -1.5]]
    assert train_data["rollout_topk_token_ids"][0].shape == (0, 3)


@pytest.mark.unit
def test_actor_cp_slice_preserves_support_rows_and_converts_gpu_dtypes_on_cpu(monkeypatch):
    previous_mode = cp_utils.get_cp_partition_mode()
    cp_utils.set_cp_partition_mode(cp_utils.CP_PARTITION_CONTIGUOUS)
    try:
        response_length = 200
        support = np.arange(response_length * 3, dtype=np.int32).reshape(response_length, 3)
        pieces = []
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 2)
        for rank in range(2):
            monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda rank=rank: rank)
            pieces.append(
                _slice_predictive_support_with_cp(
                    support,
                    total_length=300,
                    response_length=response_length,
                    qkv_format="bshd",
                    max_seq_len=512,
                    dtype=torch.long,
                    device="cpu",
                )
            )
        reconstructed = torch.cat(pieces, dim=0)
        assert reconstructed.dtype == torch.long
        assert torch.equal(reconstructed, torch.from_numpy(support).long())
    finally:
        cp_utils.set_cp_partition_mode(previous_mode)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
