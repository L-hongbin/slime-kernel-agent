"""Causal forward/backward equivalence for packed TRLOO trajectories.

CPU tests exercise the real converter, CP layout, advantage and policy loss.
Only the fused vocabulary logprob/entropy kernel is replaced by torch softmax.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.kernel_agent.kernel_reward import reward_post_process_by_group
from examples.kernel_agent.utils import _apply_overlong_penalty, _context_len_for_turn

from slime.backends.megatron_utils import cp_utils
from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.data import DataIterator, get_batch
from slime.ray.rollout import RolloutManager
from slime.utils.arguments import _validate_turn_context_limits
from slime.utils.trajectory_packing import validate_trajectory_packing_args
from slime.utils.types import Sample

NUM_GPUS = 0


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 12)
        self.qkv = torch.nn.Linear(12, 36)
        self.head = torch.nn.Linear(12, 32)

    def forward(self, tokens):
        x = self.embedding(tokens)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = q @ k.T / q.shape[-1] ** 0.5
        scores = scores.masked_fill(torch.ones_like(scores, dtype=torch.bool).triu(1), -torch.inf)
        return self.head(torch.tanh(x + scores.softmax(-1) @ v))


def _args(**overrides):
    args = dict(
        pack_multi_turn_trajectories=False,
        use_multi_turn=True,
        advantage_estimator="trloo",
        custom_reward_post_process_path="examples.kernel_agent.kernel_reward.reward_post_process_by_group",
        reward_key=None,
        grpo_std_normalization=False,
        use_rollout_routing_replay=False,
        dppo_predictive_top_k=2,
        policy_loss_mode="dppo_topk_kl_predictive",
        use_rollout_logprobs=True,
        qkv_format="thd",
        allgather_cp=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=16,
        vocab_size=32,
        kl_coef=0.0,
        custom_advantage_function_path=None,
        use_opd=False,
        normalize_advantages=False,
        use_opsm=False,
        eps_clip=0.015,
        eps_clip_high=0.015,
        eps_clip_c=5.0,
        dppo_predictive_tail_estimator="aggregated",
        get_mismatch_metrics=False,
        use_tis=False,
        entropy_coef=0.01,
        use_kl_loss=False,
        loss_type="policy_loss",
        recompute_loss_function=False,
        calculate_per_token_loss=True,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _manager(args):
    cls = RolloutManager.__ray_metadata__.modified_class
    manager = cls.__new__(cls)
    manager.args = args
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = reward_post_process_by_group
    return manager


def _samples(model, *, repair=True, filtered=False, response_lengths=None):
    samples = []
    for index in range(2):
        prompt = [1, 2 + index]
        responses = ([4 + index, 5, 9], [6, 7 + index, 9], [8, 9])
        if response_lengths is not None:
            responses = [[(j + index) % 28 + 1 for j in range(length - 1)] + [9] for length in response_lengths]
        for turn, response in enumerate(responses):
            tokens = prompt + response
            with torch.no_grad():
                logs = (
                    model(torch.tensor(tokens, device=next(model.parameters()).device))
                    .log_softmax(-1)[len(prompt) - 1 : -1]
                    .cpu()
                )
            ids = torch.zeros((len(response), 3), dtype=torch.int32)
            support = torch.zeros((len(response), 3))
            valid = torch.zeros((len(response), 3), dtype=torch.bool)
            for j, target in enumerate(response):
                selected = list(dict.fromkeys([*logs[j].topk(2).indices.tolist(), target]))
                ids[j, : len(selected)] = torch.tensor(selected)
                support[j, : len(selected)] = logs[j, selected]
                valid[j, : len(selected)] = True
            removed = filtered and index == 1 and turn == 1
            samples.append(
                Sample(
                    index=index,
                    group_index=0,
                    group_id=index,
                    tokens=tokens,
                    response_length=len(response),
                    reward=float((index + turn) % 2),
                    loss_mask=[1] * len(response),
                    remove_sample=removed,
                    rollout_log_probs=logs.gather(1, torch.tensor(response)[:, None]).squeeze(1).tolist(),
                    rollout_topk_token_ids=ids.numpy(),
                    rollout_topk_log_probs=support.numpy(),
                    rollout_topk_valid_mask=valid.numpy(),
                    metadata={"turn_idx": turn, "multi_turn_reward": float((index + turn) % 2)},
                )
            )
            prompt = tokens[:-1] + [10, 11, 9, 12, 13] if repair else tokens + [12, 13]
        if filtered:
            samples.append(
                Sample(
                    index=index,
                    group_index=0,
                    group_id=index,
                    tokens=[0, 0],
                    response_length=1,
                    reward=0.0,
                    loss_mask=[0],
                    remove_sample=True,
                    rollout_log_probs=[0.0],
                    metadata={"turn_idx": 3, "is_pad_turn": True, "multi_turn_reward": 0.0},
                )
            )
    return samples


def _torch_logprobs(logits, tokens, _tp_group, *, with_entropy, chunk_size, with_dppo_directional_moment=False):
    logp = logits.log_softmax(-1)
    prob = logp.exp()
    entropy = -(prob * logp).sum(-1) if with_entropy else None
    sampled = logp.gather(1, tokens[:, None])
    if with_dppo_directional_moment:
        moment = (prob.square() * (logp + entropy[:, None])).sum(-1).detach()
        return sampled, entropy, moment
    return sampled, entropy


@pytest.fixture
def cpu_mpu(monkeypatch):
    torch.set_num_threads(1)
    # get_batch transfers cu_seqlens metadata explicitly; keep this CPU test's
    # metadata on CPU as well. No production placement or math is replaced.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    mpu = cp_utils.mpu
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda **kw: mpu.get_context_parallel_world_size())
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda **kw: True)
    monkeypatch.setattr(loss_module, "calculate_log_probs_and_entropy", _torch_logprobs)
    return monkeypatch


def _loss_and_grad(model, data, args, monkeypatch, cp_size):
    model.zero_grad()
    device = next(model.parameters()).device
    total_loss, total_normalizer = 0.0, 0.0
    metrics = {}
    scored = []
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: cp_size)
    for rank in range(cp_size):
        monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda rank=rank: rank)
        shard = copy.deepcopy(data)
        shard["total_lengths"] = [len(tokens) for tokens in data["tokens"]]
        for key in ("tokens", "target_tokens", "loss_masks"):
            if key in shard:
                shard[key] = [torch.tensor(value, device=device) for value in shard[key]]
        shard["group_mask_sums"] = torch.tensor(shard["group_mask_sums"], device=device)
        for key in (
            "rollout_log_probs",
            "token_rewards",
            "rollout_topk_token_ids",
            "rollout_topk_log_probs",
            "rollout_topk_valid_mask",
        ):
            if key not in shard:
                continue
            dtype = (
                torch.long
                if key.endswith("token_ids")
                else torch.bool if key.endswith("valid_mask") else torch.float32
            )
            shard[key] = [
                cp_utils.slice_log_prob_with_cp(
                    torch.tensor(value, dtype=dtype, device=device), total, response, args.qkv_format
                )
                for value, total, response in zip(
                    shard[key], shard["total_lengths"], shard["response_lengths"], strict=True
                )
            ]
        loss_module.compute_advantages_and_returns(args, shard)
        # One microbatch per physical sequence, as for the production long turns.
        iterator = DataIterator(shard, [[i] for i in range(len(data["tokens"]))])
        for tokens in shard["tokens"]:
            batch = get_batch(iterator, list(shard), pad_multiplier=1)
            logits = cp_utils.slice_with_cp(model(tokens), 0, args.qkv_format).unsqueeze(0)
            loss, normalizer, log = loss_module.loss_function(args, batch, len(data["tokens"]), 2, logits)
            # Undo Megatron's CP prescale; per-group mode also includes MBS/GBS scaling.
            factor = cp_size if args.calculate_per_token_loss else cp_size * len(data["tokens"])
            (loss / factor).backward()
            total_loss += float(loss.detach()) / factor
            total_normalizer += float(normalizer) / cp_size
            for key, val in zip(log["keys"], log["values"][1:], strict=True):
                metrics[key] = metrics.get(key, 0.0) + float(val)
            _, lp = loss_module.get_log_probs_and_entropy(
                logits.detach(),
                args=args,
                unconcat_tokens=batch.get("target_tokens") or batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
            )
            local_mask = cp_utils.slice_log_prob_with_cp(
                batch["loss_masks"][0], len(tokens), batch["response_lengths"][0], args.qkv_format
            ).bool()
            scored.extend(lp["log_probs"][0][local_mask].tolist())
    if args.calculate_per_token_loss:
        total_loss /= total_normalizer
        for p in model.parameters():
            p.grad.div_(total_normalizer)
    return total_loss, torch.cat([p.grad.flatten() for p in model.parameters()]), metrics, sorted(scored)


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("mode", ["ppo", "dppo_topk_kl_predictive"])
def test_causal_loss_and_gradient_equivalence(cpu_mpu, repair, filtered, per_token, cp_size, mode):
    torch.manual_seed(421)
    model = TinyCausalModel()
    samples = _samples(model, repair=repair, filtered=filtered)
    args = _args(calculate_per_token_loss=per_token, policy_loss_mode=mode)
    split = _manager(args)._convert_samples_to_train_data(copy.deepcopy(samples))
    args.pack_multi_turn_trajectories = True
    packed = _manager(args)._convert_samples_to_train_data(copy.deepcopy(samples))
    assert len(packed["tokens"]) == 2
    assert sum(map(len, packed["tokens"])) < sum(map(len, split["tokens"]))
    assert sum(map(sum, packed["loss_masks"])) == sum(map(sum, split["loss_masks"]))
    with torch.no_grad():
        model.head.weight.add_(torch.randn_like(model.head.weight) * 0.2)
    old = _loss_and_grad(model, split, args, cpu_mpu, cp_size)
    new = _loss_and_grad(model, packed, args, cpu_mpu, cp_size)
    assert old[0] == pytest.approx(new[0], abs=2e-7, rel=3e-6)
    assert old[1].norm() > 1e-5
    torch.testing.assert_close(old[1], new[1], atol=3e-7, rtol=3e-5)
    assert old[2] == pytest.approx(new[2], abs=3e-5, rel=2e-5)
    assert old[3] == pytest.approx(new[3], abs=2e-6)


def test_changed_scored_prefix_fails_instead_of_silently_training_different_context():
    samples = _samples(TinyCausalModel())
    samples[2].tokens[1] = 25
    with pytest.raises(ValueError, match="changed scored prefix"):
        _manager(_args(pack_multi_turn_trajectories=True))._convert_samples_to_train_data(samples)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"use_multi_turn": False}, "use-multi-turn"),
        ({"advantage_estimator": "grpo"}, "trloo"),
        ({"custom_reward_post_process_path": None}, "turn-aware"),
        ({"rollout_data_postprocess_path": "sequence_mis"}, "rollout-data-postprocess"),
        ({"policy_loss_mode": "cppo"}, "policy loss"),
        ({"hidden_dropout": 0.1}, "hidden-dropout"),
        ({"use_rollout_routing_replay": True}, "routing-replay"),
    ],
)
def test_incompatible_packing_options_fail_early(override, message):
    with pytest.raises(ValueError, match=message):
        validate_trajectory_packing_args(_args(pack_multi_turn_trajectories=True, **override))


def test_current_trloo_configuration_is_supported():
    validate_trajectory_packing_args(_args(pack_multi_turn_trajectories=True))


def test_native_mtp_packing_requires_token_normalization():
    args = _args(
        pack_multi_turn_trajectories=True,
        enable_mtp_training=True,
        spec=["slime_plugins.models.qwen3_5", "get_qwen3_5_spec"],
    )
    validate_trajectory_packing_args(args)
    args.calculate_per_token_loss = False
    with pytest.raises(ValueError, match="calculate-per-token-loss"):
        validate_trajectory_packing_args(args)
    args.calculate_per_token_loss = True
    args.spec = None
    with pytest.raises(ValueError, match="native Qwen"):
        validate_trajectory_packing_args(args)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
def test_mtp_targets_preserve_history_and_cp_padding(cpu_mpu, cp_size, qkv_format):
    cpu_mpu.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    cpu_mpu.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: cp_size)
    data = _manager(_args(pack_multi_turn_trajectories=True))._convert_samples_to_train_data(
        _samples(TinyCausalModel(), repair=True, filtered=True)
    )
    for key in ("tokens", "target_tokens", "loss_masks"):
        data[key] = [torch.tensor(value) for value in data[key]]
    data["total_lengths"] = [len(value) for value in data["tokens"]]
    width = (max(data["total_lengths"]) + 15) // 16 * 16
    data["max_seq_lens"] = [width] * len(data["tokens"])
    for rank in range(cp_size):
        cpu_mpu.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda rank=rank: rank)
        batch = get_batch(DataIterator(data, [[0, 1]]), list(data), pad_multiplier=16, qkv_format=qkv_format)
        # Independent layout oracle: each rank owns the first/mirrored chunk.
        expected_inputs, expected_targets = [], []
        for inputs, targets in zip(data["tokens"], data["target_tokens"], strict=True):
            padded_width = (
                width if qkv_format == "bshd" else (len(inputs) + 2 * cp_size - 1) // (2 * cp_size) * (2 * cp_size)
            )
            for values, output in ((inputs, expected_inputs), (targets, expected_targets)):
                padded = torch.nn.functional.pad(values, (0, padded_width - len(values)))
                chunks = padded.chunk(2 * cp_size)
                output.append(torch.cat([chunks[rank], chunks[2 * cp_size - 1 - rank]]))
        for actual, parts in ((batch["tokens"], expected_inputs), (batch["mtp_labels"], expected_targets)):
            if qkv_format == "bshd":
                expected = torch.stack(parts)
            else:
                expected = torch.cat(parts)
                expected = torch.nn.functional.pad(expected, (0, (-len(expected)) % 16)).unsqueeze(0)
            torch.testing.assert_close(actual, expected)
        assert batch["target_tokens"][0][4] == 9
        assert batch["unconcat_tokens"][0][4] == 10


def test_advantage_whitening_keeps_the_original_scored_token_population(cpu_mpu):
    cpu_mpu.setattr(cp_utils.mpu, "get_data_parallel_group", lambda **kw: None)
    # A one-rank collective is the identity; use the production whitening math.
    cpu_mpu.setattr(torch.distributed, "all_reduce", lambda tensor, **kw: None)
    torch.manual_seed(421)
    model = TinyCausalModel()
    samples = _samples(model, repair=True, filtered=True)
    args = _args(normalize_advantages=True)
    split = _manager(args)._convert_samples_to_train_data(copy.deepcopy(samples))
    args.pack_multi_turn_trajectories = True
    packed = _manager(args)._convert_samples_to_train_data(copy.deepcopy(samples))
    old = _loss_and_grad(model, split, args, cpu_mpu, 1)
    new = _loss_and_grad(model, packed, args, cpu_mpu, 1)
    assert old[0] == pytest.approx(new[0], abs=2e-7)
    torch.testing.assert_close(old[1], new[1], atol=3e-7, rtol=3e-5)


def test_three_turn_context_caps_also_control_second_turn_length_penalty():
    args = _args(
        max_turns=3,
        turn_max_context_lens=[24576, 32768, 40960],
        first_turn_max_context_len=None,
        rollout_max_context_len=40960,
        rollout_max_response_len=40960,
        overlong_penalty=True,
        overlong_use_effective_response_cap=True,
        overlong_buffer_len=4096,
        overlong_penalty_factor=0.2,
        overlong_penalty_turn_idx=1,
    )
    _validate_turn_context_limits(args)
    assert [_context_len_for_turn(args, i) for i in range(3)] == [24576, 32768, 40960]
    sample = Sample(tokens=[1] * 32768, response_length=8192, reward=1.0, metadata={"turn_idx": 1})
    _apply_overlong_penalty(args, [sample])
    assert sample.reward == pytest.approx(0.8)
    assert sample.metadata["overlong_effective_response_cap"] == 8192


@pytest.mark.parametrize("caps", [[24576, 32768], [24576, 0, 40960], [32768, 24576, 40960], [24576, 32768, 49152]])
def test_invalid_per_turn_context_caps_fail(caps):
    with pytest.raises(ValueError):
        _validate_turn_context_limits(
            _args(
                max_turns=3,
                turn_max_context_lens=caps,
                first_turn_max_context_len=None,
                rollout_max_context_len=40960,
            )
        )


def test_uneven_and_fully_masked_trajectories_keep_denominators_and_end_targets():
    samples = _samples(TinyCausalModel(), repair=True, filtered=True)
    # Keep trajectory 0 complete, trajectory 1 as one real and one pad turn.
    samples = samples[:4] + [samples[4], samples[-1]]
    samples[4].remove_sample = True
    packed = _manager(_args(pack_multi_turn_trajectories=True))._convert_samples_to_train_data(samples)
    assert packed["loss_normalization_counts"] == [9, 2]
    assert packed["target_tokens"][0][4] == 9
    assert packed["tokens"][0][4] == 10
    assert sum(packed["loss_masks"][1]) == 0
    assert len(packed["packed_turn_metrics"][0]["rewards"]) == 4
    assert len(packed["packed_turn_metrics"][1]["rewards"]) == 2


def test_dp_partition_and_actor_transfer_preserve_packed_fields(cpu_mpu):
    from slime.backends.megatron_utils import actor as actor_module
    from slime.backends.megatron_utils import model as model_module
    from slime.ray import rollout as rollout_module
    from slime.utils import data as data_module

    args = _args(
        pack_multi_turn_trajectories=True,
        enable_turns_dp_partitions=True,
        global_batch_size=2,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=16,
        balance_data=False,
        micro_batch_size=1,
    )
    manager = _manager(args)
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 2,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    packed = manager._convert_samples_to_train_data(_samples(TinyCausalModel()))
    cpu_mpu.setattr(rollout_module.ray, "put", lambda x: x)
    cpu_mpu.setattr(data_module.ray, "get", copy.deepcopy)
    refs = manager._split_train_data_by_dp(packed)
    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = args
    cpu_mpu.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    cpu_mpu.setattr(cp_utils.mpu, "get_data_parallel_rank", lambda **kw: 0)
    cpu_mpu.setattr(cp_utils.mpu, "get_data_parallel_world_size", lambda **kw: 2)
    cpu_mpu.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 2)
    shard = actor._get_rollout_data(refs)
    assert len(shard["tokens"]) == len(shard["target_tokens"]) == 1
    assert shard["token_rewards"][0].shape == shard["rollout_log_probs"][0].shape
    assert shard["rollout_topk_token_ids"][0].shape[0] == shard["token_rewards"][0].numel()
    assert shard["loss_normalization_counts"] == [8]
    assert shard["global_batch_sizes"] == [2]
    assert shard["target_tokens"][0][4] == 9
    assert shard["tokens"][0][4] == 10

    # Continue through the actual trainer's forward_step. Checking only the
    # converter or duplicating mtp_kwargs in a model test misses a handoff
    # regression that silently falls back to the causal input history.
    args.enable_mtp_training = True
    args.custom_megatron_before_train_step_hook_path = None
    args.data_pad_size_multiplier = 1
    args.seq_length = 32
    args.decoder_seq_length = None
    cpu_mpu.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    cpu_mpu.setattr(model_module, "get_args", lambda: args)
    cpu_mpu.setattr(model_module, "_v4_pp_adjust_tensor_shapes_fn", lambda *a: None)

    class ForwardCaptured(Exception):
        pass

    class CaptureModel:
        def zero_grad_buffer(self):
            pass

        def __call__(self, **kwargs):
            assert kwargs["input_ids"][0, 4] == 10
            assert kwargs["mtp_kwargs"]["mtp_labels"][0, 4] == 9
            assert kwargs["loss_mask"].shape == kwargs["input_ids"].shape
            raise ForwardCaptured

    def run_forward(**kwargs):
        kwargs["forward_step_func"](kwargs["data_iterator"][0], kwargs["model"][0])

    cpu_mpu.setattr(model_module, "get_forward_backward_func", lambda: run_forward)
    with pytest.raises(ForwardCaptured):
        model_module.train_one_step(
            args,
            0,
            0,
            [DataIterator(shard, [[0]])],
            [CaptureModel()],
            SimpleNamespace(zero_grad=lambda: None),
            None,
            1,
            2,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
