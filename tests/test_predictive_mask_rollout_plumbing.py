from __future__ import annotations

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

    async def fake_post(url, payload, max_retries=None):
        captured.update(payload)
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
    )
    result = asyncio.run(cuda_agent._generate_kernel_impl(args, Sample(prompt="hello"), {"max_new_tokens": 2}))

    assert captured["top_logprobs_num"] == 2
    assert len(result) == 1
    turn = result[0]
    assert type(turn) is Sample
    assert turn.rollout_topk_token_ids.shape == (2, 3)
    assert turn.rollout_topk_token_ids.dtype == np.int32
    assert turn.rollout_topk_valid_mask[1].all()
    assert turn.metadata["trajectory_states"] == ["failed"]


@pytest.mark.unit
def test_cuda_agent_generate_records_cumulative_trajectory_failures(monkeypatch):
    env_results = iter(
        [
            {
                "env_state": {"done": False, "error": "COMPILATION_ERROR"},
                "env_extra_info": {"correctness": False, "decoy_kernel": False},
            },
            {
                "env_state": {"done": False, "error": "DECOY_KERNEL_DETECTED"},
                "env_extra_info": {"correctness": True, "decoy_kernel": True},
            },
            {
                "env_state": {"done": True},
                "env_extra_info": {"correctness": True, "decoy_kernel": False},
            },
        ]
    )

    async def fake_post(url, payload, max_retries=None):
        return {"text": "answer", "meta_info": _meta_info()}

    async def fake_env(*args, **kwargs):
        return next(env_results)

    async def fake_reward(*args, **kwargs):
        return 1.0

    monkeypatch.setattr(cuda_agent, "GenerateState", lambda args: _GenerateState())
    monkeypatch.setattr(cuda_agent, "post", fake_post)
    monkeypatch.setattr(cuda_agent, "cuda_kernel_env", fake_env)
    monkeypatch.setattr(cuda_agent, "reward_func", fake_reward)
    monkeypatch.setattr(cuda_agent, "postprocess_turn_samples", lambda args, samples, finish_reason: samples)

    args = SimpleNamespace(
        max_turns=3,
        use_multi_turn=True,
        padding_turns=False,
        rollout_max_context_len=None,
        sglang_speculative_algorithm=None,
        dppo_predictive_top_k=2,
        use_rollout_routing_replay=False,
        use_lora_weight_sync=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )
    result = asyncio.run(cuda_agent._generate_kernel_impl(args, Sample(prompt="hello"), {"max_new_tokens": 2}))

    assert [sample.metadata["trajectory_states"] for sample in result] == [
        ["failed"],
        ["failed", "failed"],
        ["failed", "failed", "successed"],
    ]


@pytest.mark.unit
@pytest.mark.parametrize("baseline", ["group", "history", "anchor", "greedy-anchor"])
@pytest.mark.parametrize("pairs", [1, 2, 3])
@pytest.mark.parametrize("version_mismatch", [False, True])
@pytest.mark.parametrize("env_done", [False, True])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_cuda_agent_verify_trains_diagnosis_and_kernel_with_paired_rewards(
    monkeypatch, baseline, pairs, version_mismatch, env_done, gamma
):
    use_anchor = baseline in {"anchor", "greedy-anchor"}
    captured = []
    messages_seen = []
    env_samples = []
    state = _GenerateState()
    raw_verification = (
        "<think>private reasoning <VERIFY>draft</VERIFY></think>\n"
        "Unrelated preamble\n<VERIFY>diagnosis</VERIFY>\nUnrelated trailing prose"
    )

    def chat_template(messages, **kwargs):
        messages_seen.append(messages.copy())
        return "prompt"

    monkeypatch.setattr(state.tokenizer, "apply_chat_template", chat_template)

    async def fake_post(url, payload, max_retries=None):
        captured.append(payload)
        meta = _meta_info()
        is_diagnosis = (len(captured) - 1) % 2 == 0
        meta["weight_version"] = "8" if version_mismatch and not is_diagnosis else "7"
        return {
            "text": raw_verification if is_diagnosis else "kernel revision",
            "meta_info": meta,
        }

    async def fake_kernel_env(args, sample, response, turn_idx):
        assert sample.metadata["role"] == "kernel"
        assert response == "kernel revision"
        assert sample.label == {"ground_truth": "reference"}
        env_samples.append(sample)
        return {
            "env_state": {"done": env_done, "correctness": True},
            "env_extra_info": {"correctness": True, "decoy_kernel": False},
        }

    async def fake_reward(args, sample):
        assert sample.metadata["role"] == "kernel"
        assert sample.metadata["turn_idx"] == 0  # Exactly one evaluated kernel per branch.
        return 0.8 if sample.metadata["verify_scoring_branch"] == "verified" else 0.3

    monkeypatch.setattr(cuda_agent, "GenerateState", lambda args: state)
    monkeypatch.setattr(cuda_agent, "post", fake_post)
    monkeypatch.setattr(cuda_agent, "cuda_kernel_env", fake_kernel_env)
    monkeypatch.setattr(cuda_agent, "reward_func", fake_reward)
    monkeypatch.setitem(cuda_agent.CUDA_AGENT_CONFIGS, "log_rollout_info", False)

    args = SimpleNamespace(
        max_turns=3,
        use_multi_turn=True,
        padding_turns=True,
        rollout_max_context_len=None,
        sglang_speculative_algorithm=None,
        dppo_predictive_top_k=2,
        use_rollout_routing_replay=False,
        use_lora_weight_sync=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        advantage_estimator="trloo",
        multi_turn_gamma=gamma,
        verify_advantage_baseline=baseline,
        kernel_verify_max_turns=2 * pairs,
        verify_prompt_config_path=str(KERNEL_AGENT_ROOT / "prompt_config/verify_prompt/tvm_ffi_correctness_v1.jinja"),
    )
    sample = Sample(
        prompt=[
            {"role": "user", "content": "original operator task"},
            {"role": "assistant", "content": "source kernel"},
            {"role": "user", "content": "source feedback and verify request"},
        ],
        index=11,
        group_index=5,
        rollout_id=17,
        label={"ground_truth": "reference"},
        metadata={
            "role": "verify",
            "task_id": "old-kernel-task",
            "verify_source_env_result": {"env_state": {"error": "output mismatch"}},
            "history_baseline": 0.3,
        },
    )
    if use_anchor:
        sample.metadata["verify_anchor_key"] = "shared-group-attempt"
    result = asyncio.run(cuda_agent._generate_with_verify_impl(args, sample, {"max_new_tokens": 2}))

    assert all(payload["return_logprob"] for payload in captured)
    expected_pairs = 1 if version_mismatch or env_done else pairs
    calls_per_pair = 2  # Shared anchor generation belongs to the group worker, never a candidate.
    diagnosis_offset = 0
    kernel_offset = diagnosis_offset + 1
    assert len(captured) == expected_pairs * calls_per_pair
    assert len(env_samples) == expected_pairs * (calls_per_pair - 1)
    assert len(result) == 2 * expected_pairs
    assert [turn.metadata["turn_idx"] for turn in result] == list(range(2 * expected_pairs))
    assert [turn.metadata["role"] for turn in result] == ["verify", "kernel"] * expected_pairs
    assert all(turn.metadata["verify_trajectory"] for turn in result)
    assert all((turn.index, turn.group_index, turn.rollout_id) == (11, 5, 17) for turn in result)
    turn_groups = sglang_rollout._split_turns_as_sample_groups([result, result])
    assert len(turn_groups) == 2 * expected_pairs
    for turn_idx, group in enumerate(turn_groups):
        assert len(group) == 2
        assert all(turn.metadata["turn_idx"] == turn_idx for turn in group)
    for pair_idx, turn in enumerate(result[::2]):
        assert turn.metadata["history_baseline"] == 0.3  # Fixed across all later pairs.
        kernel = result[2 * pair_idx + 1]
        assert kernel.response == "kernel revision"
        assert kernel.prompt == messages_seen[pair_idx * calls_per_pair + kernel_offset]
        assert kernel.reward == pytest.approx(0.0 if version_mismatch else 0.8)
        assert kernel.metadata["multi_turn_reward"] == kernel.reward
        assert kernel.metadata["verify_scoring_branch"] == "verified"
        assert kernel.metadata["env_extra_info"]["correctness"] is True
        assert kernel.remove_sample is version_mismatch
        assert kernel.loss_mask == ([0, 0] if version_mismatch else [1, 1])
        assert kernel.rollout_log_probs == [-0.25, -1.5]
        assert kernel.rollout_topk_token_ids.shape == (2, 3)
        assert kernel.weight_versions == (["8"] if version_mismatch else ["7"])
        assert turn.metadata["verify_turn_idx"] == 2 * pair_idx
        assert turn.metadata["kernel_turn_idx"] == 2 * pair_idx + 1
        assert turn.reward == pytest.approx(0.0 if version_mismatch else 0.8)
        assert turn.metadata["verify_reward_mode"] == ("pending_anchor" if use_anchor else "baseline")
        if use_anchor:
            assert turn.metadata["verify_anchor_key"] == "shared-group-attempt"
            assert kernel.metadata["verify_anchor_key"] == "shared-group-attempt"
        prefix = messages_seen[pair_idx * calls_per_pair + diagnosis_offset]
        verified_prompt = messages_seen[pair_idx * calls_per_pair + kernel_offset]
        assert verified_prompt[:-2] == prefix == turn.prompt
        assert verified_prompt[-2] == {"role": "assistant", "content": raw_verification}
        assert verified_prompt[-3]["role"] == verified_prompt[-1]["role"] == "user"
        assert "<VERIFY>diagnosis</VERIFY>" in verified_prompt[-1]["content"]
        assert "private reasoning" not in verified_prompt[-1]["content"]
        assert "Unrelated" not in verified_prompt[-1]["content"]
        assert "<VERIFY>draft</VERIFY>" not in verified_prompt[-1]["content"]
        assert turn.response == raw_verification
        assert turn.tokens == [10, 11, 7, 99]
        assert turn.metadata["verify_extracted_response"] == "<VERIFY>diagnosis</VERIFY>"
        if pair_idx:
            assert prefix[:-2] == messages_seen[(pair_idx - 1) * calls_per_pair + kernel_offset]
            assert "<VERIFY>diagnosis</VERIFY>" in prefix[-3]["content"]
            assert prefix[-2]["content"] == "kernel revision"
            assert "VERIFY" in prefix[-1]["content"]
            assert turn.metadata["verify_source_env_result"] == result[2 * pair_idx - 1].metadata["env_result"]
    verify_sample = result[0]
    assert verify_sample.response == raw_verification
    expected_reward = 0.8 if not version_mismatch else 0.0
    assert verify_sample.reward == pytest.approx(expected_reward)
    assert verify_sample.metadata["multi_turn_reward"] == pytest.approx(expected_reward)
    assert verify_sample.metadata["verify_kernel_reward"] == pytest.approx(0.8)
    assert verify_sample.remove_sample is version_mismatch
    assert verify_sample.loss_mask == ([0, 0] if version_mismatch else [1, 1])
    assert verify_sample.weight_versions == ["7"]
    assert verify_sample.rollout_log_probs == [-0.25, -1.5]
    assert verify_sample.rollout_topk_token_ids.shape == (2, 3)
    assert len(verify_sample.metadata["verify_kernel_rollout"]) == 1
    assert "verify_anchor_rollout" not in verify_sample.metadata
    assert type(verify_sample) is Sample
    assert verify_sample.metadata["role"] == "verify"
    assert verify_sample.metadata["env_result"] == {"env_extra_info": {}}
    assert verify_sample.metadata["env_extra_info"] == {}
    assert verify_sample.metadata["env_time"] == 0.0
    assert "task_id" not in verify_sample.metadata


@pytest.mark.unit
@pytest.mark.parametrize(
    "response,expected",
    [
        ("prefix <VERIFY>first\nsecond</VERIFY> suffix", "<VERIFY>first\nsecond</VERIFY>"),
        ("thinking <VERIFY>draft</VERIFY></think><VERIFY>final</VERIFY>", "<VERIFY>final</VERIFY>"),
        ("<think><VERIFY>draft</VERIFY></think><VERIFY>final</VERIFY>", "<VERIFY>final</VERIFY>"),
        ("no verification", None),
        ("<VERIFY>unfinished", None),
        ("<VERIFY> \n </VERIFY>", None),
        ("</VERIFY>reversed<VERIFY>", None),
        ("<VERIFY>one</VERIFY><VERIFY>two</VERIFY>", None),
        ("<VERIFY>outer<VERIFY>inner</VERIFY></VERIFY>", None),
        ("<think><VERIFY>unfinished reasoning</VERIFY>", None),
        ("<think><VERIFY>reasoning only</VERIFY></think>", None),
    ],
)
def test_extract_verify_block_without_reasoning_or_surrounding_text(response, expected):
    assert cuda_agent.extract_verify_response(response) == expected


@pytest.mark.unit
@pytest.mark.parametrize("render_mode", ["jinja", "format"])
def test_verify_feedback_uses_existing_templates_without_mutating_environment(render_mode):
    feedback_template = "{{ feedback }}" if render_mode == "jinja" else "{feedback}"
    dict_template = (
        '{{ feedback_dict["verification"] }}' if render_mode == "jinja" else "{feedback_dict[verification]}"
    )
    source_env = {"env_state": {"correctness": False, "error": "mismatch"}}
    verification = "<VERIFY>repair indexing</VERIFY>"
    template = cuda_agent.PromptTemplate(feedback_template, render_mode)
    ordinary_feedback = cuda_agent._apply_feedback_template(source_env, template)
    verified_feedback = cuda_agent._apply_feedback_template(source_env, template, verification=verification)
    assert verified_feedback == ordinary_feedback + "\n\nVerification analysis for the next revision:\n" + verification
    assert (
        cuda_agent._apply_feedback_template(
            source_env, cuda_agent.PromptTemplate(dict_template, render_mode), verification=verification
        )
        == verification
    )
    assert source_env == {"env_state": {"correctness": False, "error": "mismatch"}}


@pytest.mark.unit
@pytest.mark.parametrize("response", ["no tags", "<VERIFY>unfinished", "<VERIFY> </VERIFY>"])
def test_invalid_verify_format_preserves_generated_data_and_skips_kernel(monkeypatch, response):
    verify = Sample(
        response=response,
        tokens=[10, 11, 7, 99],
        response_length=2,
        rollout_log_probs=[-0.25, -1.5],
        loss_mask=[1, 1],
        status=Sample.Status.COMPLETED,
        metadata={"role": "verify"},
    )
    calls = []

    async def generate_diagnosis(args, sample, sampling_params):
        calls.append(sample)
        assert (sample.metadata or {}).get("role") == "verify", "Invalid verify must not launch a kernel"
        return [verify]

    monkeypatch.setattr(cuda_agent, "_generate_kernel_impl", generate_diagnosis)
    source = Sample(prompt="verify request", metadata={"role": "verify", "verify_source_env_result": {}})
    result = asyncio.run(cuda_agent._generate_with_verify_impl(SimpleNamespace(use_multi_turn=True), source, {}))
    assert result == [verify]
    assert len(calls) == 1
    assert verify.response == response
    assert verify.tokens == [10, 11, 7, 99]
    assert verify.rollout_log_probs == [-0.25, -1.5]
    assert verify.response_length == 2
    assert verify.remove_sample
    assert verify.loss_mask == [0, 0]
    assert verify.reward == 0.0
    assert verify.metadata["remove_reason"] == "invalid_verify_format"


@pytest.mark.unit
def test_cuda_agent_verify_sample_skips_kernel_coverage_rejection():
    verify_sample = Sample(
        reward=0.0,
        response="<VERIFY>diagnosis</VERIFY>",
        response_length=1,
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        metadata={"role": "verify", "turn_idx": 0, "env_extra_info": {}},
    )
    args = SimpleNamespace(
        advantage_estimator="trloo",
        multi_turn_gamma=1.0,
        use_coverage_rs=True,
        coverage_rs_key="time_coverage",
        coverage_rs_threshold=0.3,
        coverage_rs_factor=0.1,
        finalize_mode="none",
    )

    result = cuda_agent.postprocess_turn_samples(args, [verify_sample], finish_reason="verify_complete")

    assert result == [verify_sample]
    assert verify_sample.remove_sample is False
    assert verify_sample.metadata["multi_turn_reward"] == 0.0
    assert verify_sample.metadata["trajectory_finish_reason"] == "verify_complete"


@pytest.mark.unit
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("baseline", [None, "anchor", "greedy-anchor"])
def test_shared_anchor_generates_one_direct_turn_and_cancels_evaluation(monkeypatch, cancelled, baseline):
    monkeypatch.setattr(cuda_agent, "GenerateState", lambda args: _GenerateState())

    async def run():
        started = asyncio.Event()
        cancelled_tasks = []
        source_env = {"env_state": {"error": "source failure"}}
        history = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "source kernel"}]
        sample = Sample(prompt=history.copy(), metadata={"role": "kernel", "verify_source_env_result": source_env})
        args = SimpleNamespace(max_turns=6, use_multi_turn=True)
        if baseline is not None:
            args.verify_advantage_baseline = baseline
        sampling_params = {"temperature": 0.8, "top_p": 0.9, "top_k": 50, "min_p": 0.1, "max_new_tokens": 64}
        original_params = sampling_params.copy()

        async def fake_trajectory(scoring_args, anchor, params):
            expected = (
                {**original_params, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0}
                if baseline == "greedy-anchor"
                else original_params
            )
            assert params == expected
            assert sampling_params == original_params
            assert scoring_args.max_turns == 1
            assert scoring_args.padding_turns is False
            assert anchor.prompt[:-1] == history
            assert anchor.prompt[-1]["content"] == cuda_agent._apply_feedback_template(
                source_env, cuda_agent._get_tool_response_template(_GenerateState())
            )
            anchor.metadata["task_id"] = "shared-anchor-task"
            started.set()
            if cancelled:
                await asyncio.Event().wait()
            return [Sample(reward=0.3, status=Sample.Status.COMPLETED)]

        async def cancel_eval(args, task_id, config):
            cancelled_tasks.append(task_id)

        monkeypatch.setattr(cuda_agent, "_generate_kernel_impl", fake_trajectory)
        monkeypatch.setattr(cuda_agent, "cancel_kernel_eval", cancel_eval)
        task = asyncio.create_task(cuda_agent.generate_anchor(args, sample, sampling_params))
        await started.wait()
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled_tasks == ["shared-anchor-task"]
        else:
            result = await task
            assert len(result) == 1
            assert result[0].reward == 0.3
            assert not cancelled_tasks
        assert sampling_params == original_params
        assert args.max_turns == 6

    asyncio.run(run())


@pytest.mark.unit
def test_verify_scoring_transport_failure_aborts_one_training_sample(monkeypatch):
    requests = 0

    async def fake_post(url, payload, max_retries=None):
        nonlocal requests
        requests += 1
        return {"text": "<VERIFY>diagnosis</VERIFY>" if requests == 1 else "kernel", "meta_info": _meta_info()}

    async def broken_env(args, sample, response, turn_idx):
        raise RuntimeError("evaluation transport failed")

    monkeypatch.setattr(cuda_agent, "GenerateState", lambda args: _GenerateState())
    monkeypatch.setattr(cuda_agent, "post", fake_post)
    monkeypatch.setattr(cuda_agent, "cuda_kernel_env", broken_env)
    monkeypatch.setitem(cuda_agent.CUDA_AGENT_CONFIGS, "log_rollout_info", False)
    args = SimpleNamespace(
        **vars(_default_generate_args()),
        max_turns=4,
        use_multi_turn=True,
        padding_turns=True,
        rollout_max_context_len=None,
        advantage_estimator="trloo",
        multi_turn_gamma=1.0,
    )
    sample = Sample(
        index=12,
        prompt="task and verification request",
        metadata={"role": "verify", "verify_source_env_result": {"env_state": {"error": "compile"}}},
    )
    result = asyncio.run(cuda_agent.generate(args, sample, {"max_new_tokens": 2}))
    assert len(result) == 1
    assert result[0].status == Sample.Status.ABORTED
    assert result[0].remove_sample
    assert result[0].loss_mask == [0]
    assert result[0].metadata["turn_idx"] == 0


@pytest.mark.unit
@pytest.mark.parametrize("role", [None, "kernel", "verify", "anchor"])
def test_generate_dispatches_to_role_specific_entry(monkeypatch, role):
    calls = []
    args = SimpleNamespace(max_turns=3, padding_turns=True)
    metadata = {} if role is None else {"role": "kernel" if role == "anchor" else role}
    if role == "anchor":
        metadata["verify_scoring_branch"] = "anchor"
    sample = Sample(metadata=metadata)
    params = {"max_new_tokens": 2}
    expected = [sample]

    async def kernel_entry(received_args, received_sample, received_params):
        assert received_args is args and received_sample is sample and received_params is params
        calls.append("kernel")
        return expected

    async def verify_entry(received_args, received_sample, received_params):
        assert received_args is args and received_sample is sample and received_params is params
        calls.append("verify")
        return expected

    monkeypatch.setattr(cuda_agent, "_generate_kernel_impl", kernel_entry)
    monkeypatch.setattr(cuda_agent, "_generate_with_verify_impl", verify_entry)
    assert asyncio.run(cuda_agent.generate(args, sample, params)) is expected
    assert calls == ["verify" if role == "verify" else "kernel"]
    assert args.max_turns == 3 and args.padding_turns


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry,role,match",
    [
        ("_generate_kernel_impl", "unknown", "Kernel sample role must be"),
        ("_generate_with_verify_impl", "kernel", "requires a verify sample"),
        ("_generate_with_verify_impl", None, "requires a verify sample"),
    ],
)
def test_role_specific_entries_reject_wrong_sample_before_generation(monkeypatch, entry, role, match):
    async def unexpected_generation(*args, **kwargs):
        raise AssertionError("Wrong role must not reach model generation or KernelEnv")

    monkeypatch.setattr(cuda_agent, "post", unexpected_generation)
    monkeypatch.setattr(cuda_agent, "cuda_kernel_env", unexpected_generation)
    metadata = {} if role is None else {"role": role}
    with pytest.raises(ValueError, match=match):
        asyncio.run(getattr(cuda_agent, entry)(SimpleNamespace(), Sample(metadata=metadata), {}))


@pytest.mark.unit
def test_cuda_agent_rejects_pad_sample_generation():
    args = SimpleNamespace()

    with pytest.raises(ValueError, match="synthetic and cannot be generated"):
        asyncio.run(cuda_agent._generate_kernel_impl(args, Sample(metadata={"role": "pad"}), {}))


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
