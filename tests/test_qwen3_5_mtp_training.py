"""Two-GPU packed CP regression for the native one-step MTP training path."""

import os
import subprocess
import sys
from pathlib import Path

NUM_GPUS = 2


def check_trajectory_packing(model):
    """Compare real turn conversion/batching and native head gradients under CP2."""
    import copy

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
    from test_trloo_trajectory_packing import TinyCausalModel, _args, _manager, _samples

    from slime.backends.megatron_utils.cp_utils import slice_with_cp
    from slime.backends.megatron_utils.data import DataIterator, get_batch

    original_ce = model.compute_language_model_loss
    observed = []
    embedded_ids = []

    def capture_embedding(_, positional, keyword):
        embedded_ids.append(keyword["input_ids"].detach().clone())

    embedding_hook = model.embedding.register_forward_pre_hook(capture_embedding, with_kwargs=True)

    def record_ce(labels, logits):
        loss = original_ce(labels, logits)
        observed.append((labels.detach(), loss.detach()))
        return loss

    model.compute_language_model_loss = record_ce
    for filtered in (False, True):
        samples = _samples(TinyCausalModel(), repair=True, filtered=filtered, response_lengths=[3, 5, 2])
        # Include a fully masked trajectory, on top of filtered/dummy turns.
        if filtered:
            for sample in samples:
                if sample.index == 1 or sample.metadata["turn_idx"] == 1:
                    sample.remove_sample = True
        args = _args()
        split = _manager(args)._convert_samples_to_train_data(copy.deepcopy(samples))
        args.pack_multi_turn_trajectories = True
        merged = _manager(args)._convert_samples_to_train_data(copy.deepcopy(samples))
        results = []
        for data in (split, merged):
            model.zero_grad(set_to_none=True)
            denominator = sum(data.get("loss_normalization_counts", [max(sum(m), 1) for m in data["loss_masks"]]))
            data["total_lengths"] = [len(t) for t in data["tokens"]]
            for key in ("tokens", "target_tokens", "loss_masks"):
                if key in data:
                    data[key] = [torch.tensor(t, device="cuda") for t in data[key]]
            # Unequal microbatch counts and more than one trajectory per packed
            # microbatch exercise step normalization and physical sequence seams.
            groups = [[i] for i in range(len(data["tokens"]))] if data is split else [list(range(len(data["tokens"])))]
            iterator = DataIterator(data, groups)
            loss_sum = torch.zeros((), device="cuda")
            for _ in groups:
                batch = get_batch(iterator, list(data), pad_multiplier=16)
                observed.clear()
                embedded_ids.clear()
                logits = model(
                    input_ids=batch["tokens"],
                    position_ids=None,
                    attention_mask=None,
                    packed_seq_params=batch["packed_seq_params"],
                    loss_mask=batch["full_loss_masks"],
                    mtp_kwargs={"mtp_labels": batch.get("mtp_labels", batch["tokens"])},
                )
                assert len(observed) == 1
                targets, losses = observed[0]
                oracle_labels, oracle_masks, oracle_teachers = [], [], []
                for seq, inputs, total, response, mask in zip(
                    batch.get("target_tokens") or batch["unconcat_tokens"],
                    batch["unconcat_tokens"],
                    batch["total_lengths"],
                    batch["response_lengths"],
                    batch["loss_masks"],
                    strict=True,
                ):
                    expected = torch.zeros_like(seq)
                    expected[:-2] = seq[2:]
                    selected = torch.zeros_like(seq, dtype=torch.float32)
                    for offset, enabled in enumerate(mask):
                        target_position = total - response + offset
                        if target_position >= 2:
                            selected[target_position - 2] = enabled
                    oracle_labels.append(slice_with_cp(expected, 0, "thd"))
                    oracle_masks.append(slice_with_cp(selected, 0, "thd"))
                    teacher = torch.zeros_like(inputs)
                    teacher[:-1] = inputs[1:]
                    oracle_teachers.append(slice_with_cp(teacher, 0, "thd"))
                expected = torch.cat(oracle_labels)
                expected = torch.nn.functional.pad(expected, (0, targets.numel() - expected.numel())).unsqueeze(0)
                selected = torch.cat(oracle_masks)
                selected = torch.nn.functional.pad(selected, (0, targets.numel() - selected.numel())).unsqueeze(0)
                torch.testing.assert_close(targets, expected)
                teacher = torch.cat(oracle_teachers)
                teacher = torch.nn.functional.pad(teacher, (0, targets.numel() - teacher.numel())).unsqueeze(0)
                # Directly inspect the native head embedding inputs, including
                # repaired history positions that later scored tokens attend to.
                torch.testing.assert_close(embedded_ids[-1], teacher)
                loss_sum += (losses * selected).sum()
                logits.float().sum().mul(0).backward()
                MTPLossLoggingHelper.clean_loss_in_tracker()
            dist.all_reduce(loss_sum, group=parallel_state.get_context_parallel_group())
            grads = {}
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if name.startswith("mtp."):
                    dist.all_reduce(param.grad, group=parallel_state.get_context_parallel_group())
                    grads[name] = param.grad.float() / (2 * denominator)
                else:
                    assert param.grad.abs().max() == 0, name
            results.append((loss_sum / denominator, grads, denominator))
        assert results[0][2] == results[1][2]
        torch.testing.assert_close(results[0][0], results[1][0], atol=2e-3, rtol=2e-3)
        errors = {}
        for name, grad in results[0][1].items():
            error = (grad - results[1][1][name]).norm() / grad.norm().clamp_min(1e-8)
            errors[name] = error.item()
            assert error < 0.03, (name, error.item())
        assert len(errors) == 12
        print(
            f"trajectory_packing filtered={filtered} loss={results[1][0].item()} denominator={results[1][2]} max_grad_relative_error={max(errors.values())}",
            flush=True,
        )
    model.compute_language_model_loss = original_ce
    embedding_hook.remove()


def execute():
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
        get_gpt_mtp_block_spec,
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper
    from megatron.core.transformer.transformer_config import TransformerConfig

    from slime_plugins.models.qwen3_5_mtp import QwenMTPGPTModel

    per_token = os.environ.get("MTP_TEST_PER_TOKEN", "0") == "1"
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    parallel_state.initialize_model_parallel(context_parallel_size=2)
    torch.manual_seed(42)
    tensor_parallel.model_parallel_cuda_manual_seed(42)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=64,
        ffn_hidden_size=256,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.2,
        context_parallel_size=2,
        params_dtype=torch.bfloat16,
        bf16=True,
        pipeline_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        qk_layernorm=True,
        attention_output_gate=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
        calculate_per_token_loss=per_token,
        attention_backend=AttnBackend.flash,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=1,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
    model = (
        QwenMTPGPTModel(
            config=config,
            transformer_layer_spec=spec,
            mtp_block_spec=get_gpt_mtp_block_spec(config, spec, use_transformer_engine=True),
            vocab_size=256,
            max_sequence_length=32,
            pre_process=True,
            post_process=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            rotary_percent=1.0,
        )
        .cuda()
        .train()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    full_tokens = torch.arange(1, 33, device="cuda").view(1, -1)
    indices = torch.cat(
        [
            (
                torch.arange(start + rank * 4, start + (rank + 1) * 4, device="cuda")
                if side == 0
                else torch.arange(start + (3 - rank) * 4, start + (4 - rank) * 4, device="cuda")
            )
            for start in (0, 16)
            for side in (0, 1)
        ]
    )
    tokens = full_tokens[:, indices]
    pos = torch.arange(32, device="cuda") % 16
    full_mask = ((pos >= 7) & (pos < 15)).float().view(1, -1)
    mask = full_mask[:, indices]
    cu = torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=16,
        max_seqlen_kv=16,
    )
    observed_labels = []
    observed_losses = []
    observed_features = []
    capture_features = [True]
    original_ce = model.compute_language_model_loss

    def record_ce(labels, logits):
        observed_labels.append(labels.detach().clone())
        loss = original_ce(labels, logits)
        observed_losses.append(loss.detach().clone())
        return loss

    def record_features(_, positional, keyword):
        if capture_features[0] and keyword.get("weight") is not None:
            observed_features.append(keyword["input_"])

    feature_hook = model.output_layer.register_forward_pre_hook(record_features, with_kwargs=True)

    model.compute_language_model_loss = record_ce
    MTPLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    before = model.mtp.layers[0].eh_proj.weight.detach().clone()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        observed_labels.clear()
        observed_losses.clear()
        observed_features.clear()
        capture_features[0] = True
        logits = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            packed_seq_params=packed,
            loss_mask=mask,
            mtp_kwargs={"mtp_labels": tokens},
        )
        assert len(observed_labels) == 1
        expected_masks = []
        for depth, actual in enumerate(observed_labels, start=2):
            expected = torch.zeros_like(full_tokens)
            for start in (0, 16):
                expected[:, start : start + 16 - depth] = full_tokens[:, start + depth : start + 16]
            torch.testing.assert_close(actual, expected[:, indices])
            # Independent oracle: select by the predicted token's position in
            # its original sequence, rather than rolling the supplied mask.
            expected_masks.append(((pos + depth >= 8) & (pos + depth < 16)).float().view(1, -1))
        expected_means = []
        for loss, oracle_mask in zip(observed_losses, expected_masks, strict=True):
            total = (loss * oracle_mask[:, indices]).sum()
            dist.all_reduce(total, group=parallel_state.get_context_parallel_group())
            expected_means.append(total / oracle_mask.sum())
        torch.testing.assert_close(MTPLossLoggingHelper.tracker["values"], torch.stack(expected_means))
        capture_features[0] = False
        logits.float().sum().mul(0).backward()
        nonzero = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            assert torch.isfinite(param.grad).all(), name
            if ".mtp." in "." + name:
                if param.grad.abs().max() > 0:
                    nonzero.append(name)
                dist.all_reduce(param.grad, group=parallel_state.get_context_parallel_group())
                # MCore per-token DDP sums gradients; slime reports the full
                # response-token denominator on every CP rank. Finalize sums
                # those duplicate counts and divides all parameter gradients.
                param.grad.div_(2 * full_mask.sum() if per_token else 2)
            else:
                assert param.grad.abs().max() == 0, name
        assert nonzero, "The native head received no gradients"
        values = MTPLossLoggingHelper.tracker["values"].detach().clone()
        assert values.numel() == 1 and torch.isfinite(values).all()
        MTPLossLoggingHelper.clean_loss_in_tracker()
        if step == 0:
            # Compare the injected auxiliary gradient to a direct CE objective
            # without MTPLossAutoScaler or rank-local normalization. The
            # reference sums CP gradients; production follows its selected
            # DDP/finalization mode (SUM with token normalization or AVG).
            actual_grads = {
                n: v.grad.clone() for n, v in model.named_parameters() if n.startswith("mtp.") and v.grad is not None
            }
            optimizer.zero_grad(set_to_none=True)
            observed_labels.clear()
            observed_losses.clear()
            observed_features.clear()
            capture_features[0] = True
            reference_logits = model(
                input_ids=tokens,
                position_ids=None,
                attention_mask=None,
                packed_seq_params=packed,
                loss_mask=mask,
                mtp_kwargs={"mtp_labels": tokens},
            )
            capture_features[0] = False
            reference_loss = 0
            for features, targets, oracle_mask in zip(observed_features, observed_labels, expected_masks, strict=True):
                projected, _ = model.output_layer(input_=features, weight=model.output_layer.weight.detach())
                reference_loss = (
                    reference_loss
                    + (original_ce(targets, projected) * oracle_mask[:, indices]).sum() / oracle_mask.sum() * 0.2
                )
            reference_loss.backward()
            for name, param in model.named_parameters():
                if name not in actual_grads:
                    continue
                dist.all_reduce(param.grad, group=parallel_state.get_context_parallel_group())
                error = (param.grad.float() - actual_grads[name].float()).norm()
                relative_error = error / param.grad.float().norm().clamp_min(1e-8)
                assert relative_error < 0.03, (name, relative_error.item())
                param.grad.copy_(actual_grads[name])
            del reference_logits, reference_loss
            MTPLossLoggingHelper.clean_loss_in_tracker()
        optimizer.step()
        print(
            f"rank={rank} per_token={per_token} step={step} mtp_loss={values.item()} nonzero_mtp_params={len(nonzero)}",
            flush=True,
        )
    assert not torch.equal(before, model.mtp.layers[0].eh_proj.weight)
    assert len(model.mtp.layers) == 1
    assert not any("mtp.layers.1." in key for key in model.state_dict())
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        input_ids=tokens,
        position_ids=None,
        attention_mask=None,
        packed_seq_params=packed,
        loss_mask=torch.zeros_like(mask),
        mtp_kwargs={"mtp_labels": tokens},
    )
    logits.float().sum().mul(0).backward()
    assert torch.equal(MTPLossLoggingHelper.tracker["values"], torch.zeros(1, device="cuda"))
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all() and param.grad.abs().max() == 0
    feature_hook.remove()
    model.compute_language_model_loss = original_ce
    MTPLossLoggingHelper.clean_loss_in_tracker()
    if per_token:
        check_trajectory_packing(model)
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        execute()
    else:
        env = os.environ.copy()
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        env["PYTHONPATH"] = (
            str(Path(__file__).resolve().parents[1]) + ":/root/Megatron-LM:" + env.get("PYTHONPATH", "")
        )
        for mode in ("0", "1"):
            env["MTP_TEST_PER_TOKEN"] = mode
            subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", __file__],
                env=env,
                check=True,
            )
