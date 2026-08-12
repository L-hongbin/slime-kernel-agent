import os
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_aspo_policy_loss,
    compute_cispo_policy_loss,
    compute_cppo_policy_loss,
    compute_dis_policy_loss,
    compute_dppo_binary_policy_loss,
    compute_dppo_predictive_topk_policy_loss,
    compute_drpo_policy_loss,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss_output,
    compute_ripo_policy_loss,
    compute_sequence_log_ratio,
    compute_up_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.train_metric_utils import (
    ENTROPY_COMMON_PROBE_DENOMINATOR_KEY,
    ENTROPY_COMMON_PROBE_MASK_KEY,
    ENTROPY_COMMON_PROBE_NUMERATOR_KEY,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    apply_temperature: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and optionally applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for optional scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.
        apply_temperature: Whether to divide outputs by `rollout_temperature`.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    # rollout_temperature == 0 (greedy) would divide logits by zero here:
    # Inf -> log_softmax NaN -> NaN loss/grads (R6 full-loop NaN root cause).
    # slime_validate_args rejects it at launch; assert as defense in depth.
    if apply_temperature:
        assert args.rollout_temperature > 0, (
            f"rollout_temperature must be > 0 in the train-side log-prob path, got "
            f"{args.rollout_temperature}: dividing logits by it would produce a NaN loss."
        )
        if args.rollout_temperature != 1.0:
            logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits_local_len: int,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits_local_len: Local sequence length on this rank.
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Skip keys where all values are None (e.g. entropy when not computed)
        if all(v is None for v in values):
            continue

        # Determine reference dtype/device from first non-None value
        ref_value = next(v for v in values if v is not None)
        ref_dtype = ref_value.dtype
        ref_device = ref_value.device

        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if value is None or e <= s:
                # This rank has no response logprobs for this sample
                full_resp = torch.zeros(
                    response_length,
                    dtype=ref_dtype,
                    device=ref_device,
                    requires_grad=True,
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (resp_start, response_length - resp_end))

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            new_values.append(
                slice_log_prob_with_cp(full_resp, total_length, response_length, args.qkv_format, max_seq_len)
            )

        res[key] = new_values


def _build_shifted_tokens(
    T: int,
    device: torch.device,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> torch.Tensor:
    """Build shifted target tokens for the full packed/padded logits."""
    cp_size = mpu.get_context_parallel_world_size()

    # --- zigzag CP: completely different layout ---
    if cp_size > 1 and not allgather_cp:
        full_tokens = torch.zeros(T, dtype=torch.long, device=device)
        end = 0
        for i, (tokens, total_length, response_length) in enumerate(
            zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            for half, base in ((0, end), (1, end + chunk_size_cp)):
                lo = logits_offset[half][0] - chunks_offset[half][0]
                hi = logits_offset[half][1] - chunks_offset[half][0]
                full_tokens[base + lo : base + hi] = tokens[tokens_offset[half][0] : tokens_offset[half][1]]
            end += 2 * chunk_size_cp
        return full_tokens

    # --- cp1 and allgather-CP both build global shifted tokens the same way ---
    T_global = sum(total_lengths) if allgather_cp else T
    full_tokens = torch.zeros(T_global, dtype=torch.long, device=device)

    if qkv_format == "thd" or allgather_cp:
        offset = 0
        for tokens, total_length in zip(unconcat_tokens, total_lengths, strict=False):
            full_tokens[offset : offset + total_length - 1] = tokens[1:total_length]
            offset += total_length
    else:  # bshd, cp1
        for i, (tokens, total_length) in enumerate(zip(unconcat_tokens, total_lengths, strict=False)):
            seq_start = max_seq_lens[i] * i
            full_tokens[seq_start : seq_start + total_length - 1] = tokens[1:total_length]

    # allgather-CP: slice to local chunk
    if allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        chunk_start = cp_rank * T
        chunk_end = chunk_start + T
        if chunk_end <= T_global:
            return full_tokens[chunk_start:chunk_end].contiguous()
        local = torch.zeros(T, dtype=torch.long, device=device)
        valid = T_global - chunk_start
        if valid > 0:
            local[:valid] = full_tokens[chunk_start:]
        return local

    return full_tokens


def _extract_per_sample(
    log_prob_full: torch.Tensor,
    entropy_full: torch.Tensor | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Slice per-sample response log-probs/entropy from full-length 1-D tensors."""
    cp_size = mpu.get_context_parallel_world_size()
    log_probs_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []

    if cp_size > 1 and not allgather_cp:
        # zigzag CP
        pos = 0
        for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size_cp, chunks_offset, logits_offset, _tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            lo0 = logits_offset[0][0] - chunks_offset[0][0]
            hi0 = logits_offset[0][1] - chunks_offset[0][0]
            lo1 = logits_offset[1][0] - chunks_offset[1][0]
            hi1 = logits_offset[1][1] - chunks_offset[1][0]

            lp = torch.cat(
                [
                    log_prob_full[pos + lo0 : pos + hi0],
                    log_prob_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                ],
                dim=0,
            )
            log_probs_list.append(lp)
            if entropy_full is not None:
                ent = torch.cat(
                    [
                        entropy_full[pos + lo0 : pos + hi0],
                        entropy_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                    ],
                    dim=0,
                )
                entropy_list.append(ent)
            pos += 2 * chunk_size_cp

    elif allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        local_len = log_prob_full.size(0)
        chunk_start = cp_rank * local_len
        chunk_end = chunk_start + local_len

        seq_start = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                log_probs_list.append(torch.zeros((0,), dtype=log_prob_full.dtype, device=log_prob_full.device))
                if entropy_full is not None:
                    entropy_list.append(torch.zeros((0,), dtype=entropy_full.dtype, device=entropy_full.device))
            else:
                log_probs_list.append(log_prob_full[s - chunk_start : e - chunk_start])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[s - chunk_start : e - chunk_start])
            seq_start += total_length

    else:
        # cp1
        if qkv_format == "thd":
            offset = 0
            for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
                end = offset + total_length
                start = end - response_length
                log_probs_list.append(log_prob_full[start - 1 : end - 1])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[start - 1 : end - 1])
                offset += total_length
        else:  # bshd
            for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
                end = max_seq_lens[i] * i + total_length
                start = end - response_length
                log_probs_list.append(log_prob_full[start - 1 : end - 1])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[start - 1 : end - 1])

    return log_probs_list, entropy_list


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    Computes on the **full** logits ``[T, V]`` tensor at once (instead of
    per-sample slicing) so backward traverses ``[T, V]`` only once, then
    extracts per-sample response portions.

    When ``entropy_coef == 0``, entropy is computed under ``torch.no_grad()``
    to avoid retaining the computation graph and to skip cloning.
    """
    assert non_loss_data
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    # Apply rollout temperature scaling to logits to match rollout-time log-probs.
    # rollout_temperature == 0 (greedy) would divide by zero -> NaN loss;
    # slime_validate_args rejects it at launch; assert as defense in depth.
    rollout_temperature = getattr(args, "rollout_temperature", 1.0)
    assert rollout_temperature > 0, (
        f"rollout_temperature must be > 0 in the train-side log-prob path, got "
        f"{rollout_temperature}: dividing logits by it would produce a NaN loss."
    )
    if rollout_temperature != 1.0:
        logits = logits / rollout_temperature
    logits = logits.contiguous()
    T = logits.size(0)
    device = logits.device
    tp_group = mpu.get_tensor_model_parallel_group()
    chunk_size = args.log_probs_chunk_size

    # --- build full shifted-token target tensor ---
    full_tokens = _build_shifted_tokens(
        T, device, unconcat_tokens, total_lengths, response_lengths, qkv_format, max_seq_lens, args.allgather_cp
    )

    # --- compute on full [T,V] logits at once via calculate_log_probs_and_entropy ---
    with_dppo_directional_moment = (
        with_entropy and getattr(args, "policy_loss_mode", "ppo") == "dppo_topk_kl_predictive"
    )
    log_prob_outputs = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
        with_dppo_directional_moment=with_dppo_directional_moment,
    )
    if with_dppo_directional_moment:
        log_prob_full, entropy_full, dppo_directional_moment_full = log_prob_outputs
    else:
        log_prob_full, entropy_full = log_prob_outputs
        dppo_directional_moment_full = None
    log_prob_full = log_prob_full.squeeze(-1)  # [T, 1] -> [T]

    # --- extract per-sample response portions ---
    log_probs_list, entropy_list = _extract_per_sample(
        log_prob_full,
        entropy_full,
        total_lengths,
        response_lengths,
        qkv_format,
        max_seq_lens,
        args.allgather_cp,
    )

    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list
    if with_dppo_directional_moment:
        _, dppo_directional_moment_list = _extract_per_sample(
            log_prob_full,
            dppo_directional_moment_full,
            total_lengths,
            response_lengths,
            qkv_format,
            max_seq_lens,
            args.allgather_cp,
        )
        res["dppo_entropy_directional_moment"] = dppo_directional_moment_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=T,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=device), res


def _embed_cp_local_response_values(
    values: list[torch.Tensor],
    *,
    logits_rows: int,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> torch.Tensor:
    """Embed CP-local response rows into the flattened local-logit layout.

    ``slice_log_prob_with_cp`` stores response-only tensors as the first CP
    response chunk followed by the second.  This is the inverse placement: it
    puts those compact rows at the exact logit positions that predict them.
    Trailing dimensions are preserved, so it works for predictive Top-K
    support ids and masks without ever copying a ``[response, vocab]`` tensor.
    """
    if allgather_cp:
        raise ValueError(
            "Predictive Top-K support placement does not support allgather_cp; "
            "slime_validate_args should reject this configuration."
        )
    if not values:
        raise ValueError("Predictive Top-K support cannot be empty for a training microbatch.")
    if not (len(values) == len(total_lengths) == len(response_lengths)):
        raise ValueError(
            "Predictive Top-K support/sample count mismatch: "
            f"values={len(values)}, total_lengths={len(total_lengths)}, response_lengths={len(response_lengths)}"
        )

    reference = values[0]
    if reference.ndim != 2:
        raise ValueError(f"Predictive Top-K response values must be rank 2, got {reference.shape}.")
    width = reference.size(1)
    embedded = torch.zeros((logits_rows, width), dtype=reference.dtype, device=reference.device)
    cp_size = mpu.get_context_parallel_world_size()

    if cp_size > 1:
        pos = 0
        for i, (value, total_length, response_length) in enumerate(
            zip(values, total_lengths, response_lengths, strict=True)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size, chunks_offset, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            lo0 = logits_offset[0][0] - chunks_offset[0][0]
            hi0 = logits_offset[0][1] - chunks_offset[0][0]
            lo1 = logits_offset[1][0] - chunks_offset[1][0]
            hi1 = logits_offset[1][1] - chunks_offset[1][0]
            len0, len1 = hi0 - lo0, hi1 - lo1
            if value.ndim != 2 or value.size(1) != width or value.size(0) != len0 + len1:
                raise ValueError(
                    "Predictive Top-K CP response shape mismatch for sample "
                    f"{i}: got {tuple(value.shape)}, expected ({len0 + len1}, {width})."
                )
            embedded[pos + lo0 : pos + hi0] = value[:len0]
            embedded[pos + chunk_size + lo1 : pos + chunk_size + hi1] = value[len0:]
            pos += 2 * chunk_size
        if pos > logits_rows:
            raise ValueError(f"Predictive Top-K CP placement used {pos} rows, but logits have {logits_rows}.")
        return embedded

    if qkv_format == "thd":
        offset = 0
        for i, (value, total_length, response_length) in enumerate(
            zip(values, total_lengths, response_lengths, strict=True)
        ):
            if value.ndim != 2 or value.shape != (response_length, width):
                raise ValueError(
                    f"Predictive Top-K response shape mismatch for sample {i}: "
                    f"got {tuple(value.shape)}, expected ({response_length}, {width})."
                )
            end = offset + total_length
            start = end - response_length
            embedded[start - 1 : end - 1] = value
            offset = end
    elif qkv_format == "bshd":
        if max_seq_lens is None:
            raise ValueError("max_seq_lens is required for BSHD predictive Top-K support placement.")
        for i, (value, total_length, response_length) in enumerate(
            zip(values, total_lengths, response_lengths, strict=True)
        ):
            if value.ndim != 2 or value.shape != (response_length, width):
                raise ValueError(
                    f"Predictive Top-K response shape mismatch for sample {i}: "
                    f"got {tuple(value.shape)}, expected ({response_length}, {width})."
                )
            end = max_seq_lens[i] * i + total_length
            start = end - response_length
            embedded[start - 1 : end - 1] = value
    else:
        raise ValueError(f"Unsupported qkv_format for predictive Top-K support: {qkv_format}")
    return embedded


def _cp_local_response_row_ranges(
    *,
    logits_rows: int,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> list[tuple[int, int]]:
    """Return local-logit row ranges that predict response tokens."""
    if allgather_cp:
        raise ValueError("Predictive Top-K response ranges do not support allgather_cp.")
    cp_size = mpu.get_context_parallel_world_size()
    ranges: list[tuple[int, int]] = []
    if cp_size > 1:
        pos = 0
        for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size, chunks_offset, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            lo0 = logits_offset[0][0] - chunks_offset[0][0]
            hi0 = logits_offset[0][1] - chunks_offset[0][0]
            lo1 = logits_offset[1][0] - chunks_offset[1][0]
            hi1 = logits_offset[1][1] - chunks_offset[1][0]
            if hi0 > lo0:
                ranges.append((pos + lo0, pos + hi0))
            if hi1 > lo1:
                ranges.append((pos + chunk_size + lo1, pos + chunk_size + hi1))
            pos += 2 * chunk_size
    elif qkv_format == "thd":
        offset = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=True):
            end = offset + total_length
            start = end - response_length
            if response_length:
                ranges.append((start - 1, end - 1))
            offset = end
    elif qkv_format == "bshd":
        if max_seq_lens is None:
            raise ValueError("max_seq_lens is required for BSHD predictive Top-K response ranges.")
        for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
            end = max_seq_lens[i] * i + total_length
            start = end - response_length
            if response_length:
                ranges.append((start - 1, end - 1))
    else:
        raise ValueError(f"Unsupported qkv_format for predictive Top-K response ranges: {qkv_format}")

    if any(start < 0 or end > logits_rows or end < start for start, end in ranges):
        raise ValueError(f"Predictive Top-K response row range is outside logits_rows={logits_rows}: {ranges}")
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _require_tensor_condition(condition: torch.Tensor, message: str) -> None:
    """Fail loudly without synchronizing the normal CUDA training path."""
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


@torch.no_grad()
def _vocab_parallel_selected_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    tp_group,
    tp_rank: int,
    tp_world_size: int,
    chunk_size: int,
    row_ranges: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Return normalized log-probs for selected global token ids.

    The implementation deliberately avoids ``log_softmax(logits)`` and avoids
    gathering response logits.  It scans the existing local logits in bounded
    row chunks, keeps only ``[T, K+1]``, and performs TP collectives only when
    TP>1. This matters for formal DS-V4 training, which has roughly 3 GiB HBM headroom.
    """
    if logits.ndim != 2 or token_ids.ndim != 2 or valid_mask.shape != token_ids.shape:
        raise ValueError(
            "Predictive selected-logprob shape mismatch: "
            f"logits={tuple(logits.shape)}, ids={tuple(token_ids.shape)}, valid={tuple(valid_mask.shape)}"
        )
    if logits.size(0) != token_ids.size(0):
        raise ValueError(f"Predictive selected-logprob row mismatch: logits={logits.size(0)}, ids={token_ids.size(0)}")
    if token_ids.dtype != torch.long or valid_mask.dtype != torch.bool:
        raise ValueError(
            f"Predictive support ids/mask must be torch.long/torch.bool, got {token_ids.dtype}/{valid_mask.dtype}."
        )
    if tp_world_size < 1 or not 0 <= tp_rank < tp_world_size:
        raise ValueError(f"Invalid TP coordinates rank={tp_rank}, world_size={tp_world_size}.")

    local_vocab_size = logits.size(1)
    global_vocab_size = local_vocab_size * tp_world_size
    _require_tensor_condition(
        ((~valid_mask) | ((token_ids >= 0) & (token_ids < global_vocab_size))).all(),
        f"Predictive support contains a token id outside the tensor-parallel vocabulary [0, {global_vocab_size}).",
    )

    result = torch.zeros(token_ids.shape, dtype=logits.dtype, device=logits.device)
    row_chunk = chunk_size if chunk_size and chunk_size > 0 else 512
    vocab_start = tp_rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size

    scan_ranges = row_ranges if row_ranges is not None else [(0, logits.size(0))]
    for range_start, range_end in scan_ranges:
        for start in range(range_start, range_end, row_chunk):
            end = min(start + row_chunk, range_end)
            chunk_valid = valid_mask[start:end]
            chunk_logits = logits[start:end]
            chunk_ids = token_ids[start:end]
            local_owner = chunk_valid & (chunk_ids >= vocab_start) & (chunk_ids < vocab_end)
            local_ids = (chunk_ids - vocab_start).clamp(min=0, max=local_vocab_size - 1)
            selected_logits = chunk_logits.gather(1, local_ids)
            selected_logits.masked_fill_(~local_owner, 0.0)

            if tp_world_size == 1:
                log_normalizer = torch.logsumexp(chunk_logits, dim=-1)
            else:
                row_max = chunk_logits.max(dim=-1).values
                dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tp_group)
                shifted_exp = chunk_logits - row_max.unsqueeze(-1)
                shifted_exp.exp_()
                denominator = shifted_exp.sum(dim=-1)
                del shifted_exp
                dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=tp_group)
                log_normalizer = row_max + denominator.log()
                dist.all_reduce(selected_logits, op=dist.ReduceOp.SUM, group=tp_group)

            selected_log_probs = selected_logits - log_normalizer.unsqueeze(-1)
            selected_log_probs.masked_fill_(~chunk_valid, 0.0)
            _require_tensor_condition(
                torch.isfinite(torch.where(chunk_valid, selected_log_probs, 0.0)).all(),
                "Non-finite current-policy predictive support log-probability.",
            )
            result[start:end] = selected_log_probs
    return result


def get_dppo_predictive_support_log_probs(
    logits: torch.Tensor,
    *,
    args: Namespace,
    support_token_ids: list[torch.Tensor],
    support_valid_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None,
) -> list[torch.Tensor]:
    """Compute current-policy probabilities on rollout's compact Top-K support."""
    if logits.dtype != torch.float32 or logits.ndim != 3:
        raise ValueError(f"Expected float32 rank-3 logits, got {logits.dtype} {tuple(logits.shape)}.")
    if args.qkv_format == "thd":
        if logits.size(0) != 1:
            raise ValueError(f"THD predictive logits require leading size 1, got {tuple(logits.shape)}.")
        flat_logits = logits.squeeze(0)
    else:
        flat_logits = logits.view(-1, logits.size(-1))

    temperature = getattr(args, "rollout_temperature", 1.0)
    if temperature <= 0:
        raise ValueError(f"rollout_temperature must be positive, got {temperature}.")
    if temperature != 1.0:
        flat_logits = flat_logits / temperature

    tp_world_size = mpu.get_tensor_model_parallel_world_size()
    effective_vocab_size = flat_logits.size(-1) * tp_world_size
    configured_vocab_size = getattr(args, "vocab_size", effective_vocab_size)
    if effective_vocab_size != configured_vocab_size:
        raise ValueError(
            "Predictive Top-K requires the train softmax support to match SGLang's vocabulary exactly; "
            f"train TP support={effective_vocab_size}, configured vocab_size={configured_vocab_size}. "
            "A padded output vocabulary would put probability mass on tokens absent from rollout."
        )

    full_ids = _embed_cp_local_response_values(
        support_token_ids,
        logits_rows=flat_logits.size(0),
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        allgather_cp=args.allgather_cp,
    )
    full_valid = _embed_cp_local_response_values(
        support_valid_masks,
        logits_rows=flat_logits.size(0),
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        allgather_cp=args.allgather_cp,
    )
    response_row_ranges = _cp_local_response_row_ranges(
        logits_rows=flat_logits.size(0),
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        allgather_cp=args.allgather_cp,
    )
    current_full = _vocab_parallel_selected_log_probs(
        flat_logits,
        full_ids,
        full_valid,
        tp_group=mpu.get_tensor_model_parallel_group(),
        tp_rank=mpu.get_tensor_model_parallel_rank(),
        tp_world_size=tp_world_size,
        chunk_size=getattr(args, "log_probs_chunk_size", 512),
        row_ranges=response_row_ranges,
    )
    current, _ = _extract_per_sample(
        current_full,
        None,
        total_lengths,
        response_lengths,
        args.qkv_format,
        max_seq_lens,
        args.allgather_cp,
    )
    return current


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration passed to `get_responses`; temperature scaling is
            disabled for value outputs.
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        apply_temperature=False,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=logits.size(1),
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=logits.device), res


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student log-probability tensors.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging
    rollout_data["opd_reverse_kl"] = reverse_kls


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "rloo", "trloo", "ppo",
    "reinforce_plus_plus", and "reinforce_plus_plus_baseline". When
    `args.normalize_advantages` is True, advantages are whitened across the
    data-parallel-with-context-parallel group using masked statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    If ``args.custom_advantage_function_path`` is set, it is called after KL computation
    and must populate ``rollout_data["advantages"]`` and
    ``rollout_data["returns"]``.

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    rollout_log_probs: list[torch.Tensor] | None = rollout_data.get("rollout_log_probs")
    log_probs: list[torch.Tensor] | None = (
        rollout_log_probs if args.use_rollout_logprobs else rollout_data.get("log_probs")
    )
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs or rollout_log_probs or values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]
    rollout_data["kl"] = kl

    if args.custom_advantage_function_path is not None:
        custom_adv_fn = load_function(args.custom_advantage_function_path)
        custom_adv_fn(args, rollout_data)
        advantages, returns = rollout_data["advantages"], rollout_data["returns"]

    elif args.advantage_estimator in ["grpo", "gspo", "rloo", "trloo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    # elif args.advantage_estimator == "rloo":
    #     rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
    #     advantages, returns = get_rloo_returns(
    #         rewards=rewards,
    #         kl=kl,
    #         n_samples_per_prompt=args.n_samples_per_prompt,
    #     )

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        assert (
            all_advs.size() == all_masks.size()
        ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
        # `all_advs` / `all_masks` only cover the tokens this CP rank owns, so the
        # statistics must be reduced over the DP group *with* context parallel.
        # The CP-excluding group makes every CP rank whiten its own zigzag slice
        # with its own mean/var, i.e. the two halves of one sequence get
        # different affine transforms.
        #
        # This has to stay unconditional: a CP rank can legitimately own zero
        # response tokens (prompt-heavy sequences put both of its chunks inside
        # the prompt), and skipping the collective on just that rank would
        # desync the all_reduce. `distributed_masked_whiten` handles an empty
        # local tensor — it contributes 0 to the reduced sums.
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        whitened_advs_flat = distributed_masked_whiten(
            all_advs,
            all_masks,
            process_group=dp_cp_group,
            shift_mean=True,
        )
        chunk_lengths = [chunk.size(0) for chunk in advantages]
        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def _validate_dppo_predictive_support_batch(
    args: Namespace,
    batch: RolloutBatch,
    sampled_old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and concatenate rollout Top-K support for one local CP batch."""
    field_names = (
        "rollout_topk_token_ids",
        "rollout_topk_log_probs",
        "rollout_topk_valid_mask",
    )
    missing = [name for name in field_names if name not in batch or batch[name] is None]
    if missing:
        raise ValueError(
            "dppo_topk_kl_predictive requires rollout Top-K support fields; missing " + ", ".join(missing)
        )

    ids_list = batch["rollout_topk_token_ids"]
    behavior_logs_list = batch["rollout_topk_log_probs"]
    valid_list = batch["rollout_topk_valid_mask"]
    sample_count = len(batch["response_lengths"])
    if not (len(ids_list) == len(behavior_logs_list) == len(valid_list) == sample_count):
        raise ValueError(
            "Predictive Top-K sample-count mismatch: "
            f"ids={len(ids_list)}, log_probs={len(behavior_logs_list)}, valid={len(valid_list)}, "
            f"responses={sample_count}."
        )

    expected_width = args.dppo_predictive_top_k + 1
    for i, (ids, behavior_logs, valid) in enumerate(zip(ids_list, behavior_logs_list, valid_list, strict=True)):
        if (
            not isinstance(ids, torch.Tensor)
            or not isinstance(behavior_logs, torch.Tensor)
            or not isinstance(valid, torch.Tensor)
        ):
            raise TypeError(f"Predictive Top-K sample {i} fields must be torch tensors after actor transfer.")
        if ids.ndim != 2 or ids.size(1) != expected_width:
            raise ValueError(
                f"Predictive Top-K ids sample {i} has shape {tuple(ids.shape)}, "
                f"expected [local_response, {expected_width}]."
            )
        if behavior_logs.shape != ids.shape or valid.shape != ids.shape:
            raise ValueError(
                f"Predictive Top-K field shape mismatch in sample {i}: "
                f"ids={tuple(ids.shape)}, log_probs={tuple(behavior_logs.shape)}, valid={tuple(valid.shape)}."
            )
        if ids.dtype != torch.long or behavior_logs.dtype != torch.float32 or valid.dtype != torch.bool:
            raise TypeError(
                f"Predictive Top-K dtypes in sample {i} must be long/float32/bool, got "
                f"{ids.dtype}/{behavior_logs.dtype}/{valid.dtype}."
            )

    support_ids = torch.cat(ids_list, dim=0)
    behavior_support_logs = torch.cat(behavior_logs_list, dim=0)
    support_valid = torch.cat(valid_list, dim=0)
    if support_ids.size(0) != sampled_old_log_probs.numel():
        raise ValueError(
            "Predictive Top-K/local sampled-logprob row mismatch: "
            f"support={support_ids.size(0)}, sampled={sampled_old_log_probs.numel()}."
        )

    local_sampled_ids: list[torch.Tensor] = []
    local_loss_masks: list[torch.Tensor] = []
    max_seq_lens = batch.get("max_seq_lens")
    for i, (tokens, loss_mask, total_length, response_length) in enumerate(
        zip(
            batch["unconcat_tokens"],
            batch["loss_masks"],
            batch["total_lengths"],
            batch["response_lengths"],
            strict=True,
        )
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        response_tokens = tokens[-response_length:] if response_length else tokens.new_empty((0,))
        local_sampled_ids.append(
            slice_log_prob_with_cp(
                response_tokens,
                total_length,
                response_length,
                args.qkv_format,
                max_seq_len,
            )
        )
        local_loss_masks.append(
            slice_log_prob_with_cp(
                loss_mask,
                total_length,
                response_length,
                args.qkv_format,
                max_seq_len,
            )
        )

    sampled_ids = torch.cat(local_sampled_ids, dim=0).to(device=support_ids.device, dtype=torch.long)
    active = torch.cat(local_loss_masks, dim=0).to(device=support_ids.device).bool()
    if sampled_ids.shape != sampled_old_log_probs.shape or active.shape != sampled_old_log_probs.shape:
        raise ValueError(
            "Predictive Top-K CP alignment mismatch: "
            f"sampled_ids={tuple(sampled_ids.shape)}, active={tuple(active.shape)}, "
            f"sampled_log_probs={tuple(sampled_old_log_probs.shape)}."
        )

    valid_count = support_valid.sum(dim=-1)
    bad_active_count = (
        active & (valid_count != args.dppo_predictive_top_k) & (valid_count != args.dppo_predictive_top_k + 1)
    )
    _require_tensor_condition(
        (~bad_active_count).all(),
        f"Predictive Top-K active rows must have K or K+1 valid entries for K={args.dppo_predictive_top_k}.",
    )
    _require_tensor_condition(
        torch.isfinite(torch.where(support_valid, behavior_support_logs, 0.0)).all(),
        "Predictive Top-K behavior support contains a non-finite valid log-probability.",
    )

    vocab_size = args.vocab_size
    _require_tensor_condition(
        ((~support_valid) | ((support_ids >= 0) & (support_ids < vocab_size))).all(),
        f"Predictive Top-K support token id is outside configured vocabulary [0, {vocab_size}).",
    )
    sorted_ids = torch.where(support_valid, support_ids, torch.full_like(support_ids, vocab_size)).sort(dim=-1).values
    duplicate = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] != vocab_size)
    _require_tensor_condition(~duplicate.any(), "Predictive Top-K support contains duplicate token ids in a row.")

    sampled_matches = support_valid & (support_ids == sampled_ids.unsqueeze(-1))
    sampled_match_count = sampled_matches.sum(dim=-1)
    _require_tensor_condition(
        ((~active) | (sampled_match_count == 1)).all(),
        "Predictive Top-K active rows must contain the sampled token exactly once.",
    )

    sampled_behavior_log_prob = torch.where(
        sampled_matches,
        behavior_support_logs,
        torch.zeros_like(behavior_support_logs),
    ).sum(dim=-1)
    old_logprob_matches = torch.isclose(
        sampled_behavior_log_prob,
        sampled_old_log_probs.detach().float(),
        rtol=1e-5,
        atol=2e-5,
    )
    _require_tensor_condition(
        ((~active) | old_logprob_matches).all(),
        "Predictive Top-K sampled behavior log-prob disagrees with rollout_log_probs.",
    )
    return support_ids, behavior_support_logs, support_valid


def _entropy_common_probe_sufficient_stats(
    entropy: torch.Tensor,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str,
    max_seq_lens: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CP-local entropy sum/count for the frozen original population."""
    field_lengths = {
        "total_lengths": len(total_lengths),
        "response_lengths": len(response_lengths),
        "loss_masks": len(loss_masks),
    }
    if len(set(field_lengths.values())) != 1 or not loss_masks:
        raise RuntimeError(f"entropy common-probe batch length mismatch: {field_lengths}")
    if qkv_format == "bshd":
        if max_seq_lens is None:
            raise RuntimeError("entropy common-probe BSHD reduction requires max_seq_lens")
        if len(max_seq_lens) != len(loss_masks):
            raise RuntimeError(
                "entropy common-probe max_seq_lens length mismatch: "
                f"max_seq_lens={len(max_seq_lens)}, loss_masks={len(loss_masks)}"
            )

    for index, (total_length, response_length, loss_mask) in enumerate(
        zip(total_lengths, response_lengths, loss_masks, strict=True)
    ):
        if not isinstance(loss_mask, torch.Tensor) or loss_mask.ndim != 1:
            shape = tuple(loss_mask.shape) if isinstance(loss_mask, torch.Tensor) else None
            raise RuntimeError(
                f"entropy common-probe loss_masks[{index}] must be a 1-D tensor, "
                f"got type={type(loss_mask).__name__} shape={shape}"
            )
        if loss_mask.numel() != int(response_length):
            raise RuntimeError(
                f"entropy common-probe mask/response length mismatch at sample {index}: "
                f"mask={loss_mask.numel()}, response_length={response_length}"
            )
        if int(total_length) <= 0 or int(response_length) < 0 or int(response_length) > int(total_length):
            raise RuntimeError(
                f"entropy common-probe invalid lengths at sample {index}: "
                f"response_length={response_length}, total_length={total_length}"
            )

    # Force global-token semantics regardless of the training loss reduction.
    # Each CP rank contributes only the response rows it owns; the two values
    # are all-reduced together by the normal train metric path, and their final
    # ratio therefore cancels its outer normalization exactly.
    global_token_reducer = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        loss_masks,
        sample_denoms=None,
        calculate_per_token_loss=True,
        qkv_format=qkv_format,
        max_seq_lens=max_seq_lens,
    )
    detached_entropy = entropy.detach()
    numerator = global_token_reducer(detached_entropy)
    denominator = global_token_reducer(torch.ones_like(detached_entropy))
    if numerator.numel() != 1 or denominator.numel() != 1:
        raise RuntimeError(
            "entropy common-probe sufficient statistics must be scalars, got "
            f"numerator={tuple(numerator.shape)}, denominator={tuple(denominator.shape)}"
        )
    return numerator.detach(), denominator.detach()


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]

    policy_loss_mode = getattr(args, "policy_loss_mode", "ppo")
    dis_ratio_level = getattr(args, "dis_ratio_level", "token")

    # Pre-gather log probs if needed by OPSM, GSPO, response-level CPPO,
    # or sequence-level DIS.
    need_full_log_probs = (
        args.use_opsm
        or args.advantage_estimator == "gspo"
        or policy_loss_mode == "cppo"
        or (policy_loss_mode == "dis" and dis_ratio_level == "sequence")
    )

    full_log_probs = None
    full_old_log_probs = None
    full_advantages = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        if policy_loss_mode == "cppo":
            full_advantages = [
                all_gather_with_cp(advantage, total_length, response_length)
                for advantage, total_length, response_length in zip(
                    batch["advantages"], total_lengths, response_lengths, strict=False
                )
            ]

    dis_log_ratio = None
    if policy_loss_mode == "dis" and dis_ratio_level == "sequence":
        dis_log_ratio = compute_sequence_log_ratio(
            full_log_probs=full_log_probs,
            full_rollout_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    # Policy diagnostics describe the policy/mask before any optional TIS/RS
    # rejection changes the loss mask. Preserve this reducer now; using the
    # rebuilt post-TIS reducer below would silently exclude rejected tokens.
    sum_of_sample_mean_for_dppo_metrics = sum_of_sample_mean
    _dppo = None
    if policy_loss_mode == "dis":
        _dppo = compute_dis_policy_loss(
            log_probs=log_probs,
            rollout_log_probs=old_log_probs,
            advantages=advantages,
            eps_clip=args.eps_clip,
            eps_clip_high=args.eps_clip_high,
            log_ratio=dis_log_ratio,
        )
        policy_loss_output = _dppo
    elif policy_loss_mode.startswith("dppo"):
        assert args.advantage_estimator != "gspo", (
            "DPPO ignores GSPO's sequence-level ppo_kl and would silently run "
            "per-token — gspo+dppo is not supported."
        )
        if policy_loss_mode == "dppo_topk_kl_predictive":
            support_ids, behavior_support_log_probs, support_valid_mask = _validate_dppo_predictive_support_batch(
                args, batch, old_log_probs
            )
            current_support_log_probs = torch.cat(
                get_dppo_predictive_support_log_probs(
                    logits,
                    args=args,
                    support_token_ids=batch["rollout_topk_token_ids"],
                    support_valid_masks=batch["rollout_topk_valid_mask"],
                    total_lengths=total_lengths,
                    response_lengths=response_lengths,
                    max_seq_lens=max_seq_lens,
                ),
                dim=0,
            )
            if current_support_log_probs.shape != behavior_support_log_probs.shape:
                raise ValueError(
                    "Predictive Top-K current/behavior support shape mismatch after CP placement: "
                    f"current={tuple(current_support_log_probs.shape)}, "
                    f"behavior={tuple(behavior_support_log_probs.shape)}."
                )
            # ``support_ids`` is validated above and consumed by the selected
            # logit gather. Keep this explicit assertion so a future plumbing
            # refactor cannot accidentally validate one tensor and gather another.
            if support_ids.shape != current_support_log_probs.shape:
                raise ValueError(
                    f"Predictive Top-K id/log-prob shape mismatch: {tuple(support_ids.shape)} "
                    f"vs {tuple(current_support_log_probs.shape)}."
                )
            _dppo = compute_dppo_predictive_topk_policy_loss(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                behavior_support_log_probs=behavior_support_log_probs,
                current_support_log_probs=current_support_log_probs,
                support_valid_mask=support_valid_mask,
                advantages=advantages,
                delta=args.eps_clip,
                tail_estimator=args.dppo_predictive_tail_estimator,
                vocab_size=args.vocab_size,
                eps_clip_c=args.eps_clip_c,
            )
        else:
            # Legacy binary divergence masks remain available for non-predictive
            # DPPO modes and preserve their previous behavior exactly.
            _dppo = compute_dppo_binary_policy_loss(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages,
                eps_clip=args.eps_clip,
                eps_clip_high=args.eps_clip_high,
                loss_mode=policy_loss_mode,
                eps_clip_c=args.eps_clip_c,
            )
        policy_loss_output = _dppo
    elif policy_loss_mode == "drpo":
        policy_loss_output = compute_drpo_policy_loss(
            log_probs, old_log_probs, advantages, args.eps_clip, args.eps_clip_high
        )
    elif policy_loss_mode == "cppo":
        per_sample_outputs = [
            compute_cppo_policy_loss(
                full_log_prob,
                full_old_log_prob,
                full_advantage,
                args.eps_clip,
                args.cppo_prefix_delta,
                args.cppo_weight_floor,
                args.eps_clip_c,
            )
            for full_log_prob, full_old_log_prob, full_advantage in zip(
                full_log_probs, full_old_log_probs, full_advantages, strict=False
            )
        ]
        policy_loss_output = {
            key: torch.cat(
                [
                    slice_log_prob_with_cp(
                        item[key],
                        total_length,
                        response_length,
                        args.qkv_format,
                        max_seq_len,
                    )
                    for item, total_length, response_length, max_seq_len in zip(
                        per_sample_outputs,
                        total_lengths,
                        response_lengths,
                        max_seq_lens if max_seq_lens is not None else [None] * len(total_lengths),
                        strict=False,
                    )
                ],
                dim=0,
            )
            for key in per_sample_outputs[0]
        }
    elif policy_loss_mode == "up":
        policy_loss_output = compute_up_policy_loss(
            log_probs, old_log_probs, advantages, args.eps_clip, args.eps_clip_high, args.eps_clip_c
        )
    elif policy_loss_mode == "aspo":
        policy_loss_output = compute_aspo_policy_loss(
            log_probs, old_log_probs, advantages, args.eps_clip, args.eps_clip_high, args.eps_clip_c
        )
    elif policy_loss_mode == "ripo":
        ripo_delta_high = args.ripo_delta_high if args.ripo_delta_high is not None else args.ripo_delta
        policy_loss_output = compute_ripo_policy_loss(
            log_probs,
            old_log_probs,
            advantages,
            args.ripo_delta,
            ripo_delta_high,
            args.ripo_ratio_min,
            args.ripo_ratio_max,
        )
    elif policy_loss_mode == "cispo":
        policy_loss_output = compute_cispo_policy_loss(
            log_probs, old_log_probs, advantages, args.eps_clip, args.eps_clip_high
        )
    else:
        policy_loss_output = compute_policy_loss_output(
            ppo_kl, advantages, args.eps_clip, args.eps_clip_high, args.eps_clip_c
        )

    pg_loss = policy_loss_output["pg_losses"]
    policy_loss_metrics = {key: value for key, value in policy_loss_output.items() if key != "pg_losses"}
    pg_clipfrac = policy_loss_metrics["pg_clipfrac"]
    dppo_extra_metrics = policy_loss_metrics if _dppo is not None else {}

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with
        # modified_response_masks for numerator correction (rejected tokens
        # zeroed in pg_loss). Denominators stay the precomputed per-group
        # totals from ``group_mask_sums`` (based on original loss_masks) —
        # same normalizer as the outer reducer, so pg_loss and the rest of the
        # reported metrics live in the same per-rollout-mean space.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["group_mask_sums"],
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    entropy_common_probe_stats = {}
    if getattr(args, "entropy_common_probe", False):
        common_probe_loss_masks = batch.get(ENTROPY_COMMON_PROBE_MASK_KEY)
        if common_probe_loss_masks is None:
            raise RuntimeError(
                f"--entropy-common-probe requires training batch field {ENTROPY_COMMON_PROBE_MASK_KEY!r}"
            )
        common_probe_numerator, common_probe_denominator = _entropy_common_probe_sufficient_stats(
            entropy,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            loss_masks=common_probe_loss_masks,
            qkv_format=args.qkv_format,
            max_seq_lens=max_seq_lens,
        )
        entropy_common_probe_stats = {
            ENTROPY_COMMON_PROBE_NUMERATOR_KEY: common_probe_numerator,
            ENTROPY_COMMON_PROBE_DENOMINATOR_KEY: common_probe_denominator,
        }

    detached_entropy = entropy.detach()
    detached_advantages = advantages.detach()
    ratio_ge_1 = log_probs.detach() >= old_log_probs.detach()
    positive_advantage = detached_advantages > 0
    negative_advantage = detached_advantages < 0
    clip_diagnostics = policy_loss_metrics
    entropy_groups = {
        "adv_positive_ratio_ge_1": positive_advantage & ratio_ge_1,
        "adv_positive_ratio_lt_1": positive_advantage & ~ratio_ge_1,
        "adv_negative_ratio_ge_1": negative_advantage & ratio_ge_1,
        "adv_negative_ratio_lt_1": negative_advantage & ~ratio_ge_1,
        "upper_clipped": clip_diagnostics["pg_upper_clipfrac"].detach().bool(),
        "lower_clipped": clip_diagnostics["pg_lower_clipfrac"].detach().bool(),
    }
    entropy_group_metrics = {}
    for group, indicator in entropy_groups.items():
        indicator = indicator.to(detached_entropy.dtype)
        entropy_group_metrics[f"_entropy/{group}_fraction"] = indicator
        entropy_group_metrics[f"_entropy/{group}_joint_mean"] = detached_entropy * indicator

    if _dppo is not None and args.policy_loss_mode == "dppo_topk_kl_predictive":

        # Exact first-order entropy change in logit space for the same
        # optimizer-ascent coefficient c=A*pi/mu used by the policy loss:
        #
        #   dz = c (e_k - p)
        #   dH = c * [sum_i p_i^2(log p_i + H)
        #             - p_k(log p_k + H)].
        #
        # The full-vocabulary sum is produced by the existing entropy softmax
        # pass above, so this remains exact without retaining or rescanning a
        # [response_tokens, vocab] tensor.  Every tensor below is observational
        # and detached; the policy loss and gradient are unchanged.
        directional_moment = torch.cat(log_probs_and_entropy["dppo_entropy_directional_moment"], dim=0).detach()
        metric_dtype = dppo_extra_metrics["dppo/signed_update_unmasked"].dtype
        entropy_for_metric = detached_entropy.to(metric_dtype)
        sampled_log_prob_for_metric = log_probs.detach().to(metric_dtype)
        sampled_prob_for_metric = sampled_log_prob_for_metric.exp()
        entropy_unit_direction = directional_moment.to(metric_dtype) - sampled_prob_for_metric * (
            sampled_log_prob_for_metric + entropy_for_metric
        )
        signed_update_unmasked = dppo_extra_metrics["dppo/signed_update_unmasked"]
        signed_update_kept = dppo_extra_metrics["dppo/signed_update_kept"]
        entropy_first_order_unmasked = entropy_unit_direction * signed_update_unmasked
        entropy_first_order_kept = entropy_unit_direction * signed_update_kept
        dppo_extra_metrics["entropy/first_order_unmasked"] = entropy_first_order_unmasked
        dppo_extra_metrics["entropy/first_order_kept"] = entropy_first_order_kept
        dppo_extra_metrics["entropy/first_order_mask_delta"] = entropy_first_order_kept - entropy_first_order_unmasked
        dppo_extra_metrics["entropy/first_order_upper_clipped"] = (
            entropy_first_order_unmasked * dppo_extra_metrics["dppo/upper_clip_joint_frac"]
        )
        dppo_extra_metrics["entropy/first_order_lower_clipped"] = (
            entropy_first_order_unmasked * dppo_extra_metrics["dppo/lower_clip_joint_frac"]
        )

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        # This metric is specifically a train-forward vs rollout-engine check.
        # ``old_log_probs`` may alias ``rollout_log_probs`` when
        # ``use_rollout_logprobs`` is enabled, which would make it identically zero.
        train_rollout_logprob_abs_diff = sum_of_sample_mean_for_dppo_metrics(
            (log_probs.detach() - rollout_log_probs).abs()
        )

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }
    reported_loss.update(entropy_common_probe_stats)
    for _k, _v in policy_loss_metrics.items():
        if _k == "pg_clipfrac":
            continue
        reported_loss[_k] = sum_of_sample_mean_for_dppo_metrics(_v).clone().detach()
    for _k, _v in entropy_group_metrics.items():
        reported_loss[_k] = sum_of_sample_mean_for_dppo_metrics(_v).clone().detach()

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac
        reported_loss["opsm_reject_rate"] = opsm_clipfrac / max(len(batch["loss_masks"]), 1)

    # Add OPD metrics if available
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    step_global_batch_size: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        step_global_batch_size: Sample count for the current training step
            (total across DP). Replaces the legacy ``args.global_batch_size``
            fallback so the train side stops depending on "every DP rank holds
            the same N samples".
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["group_mask_sums"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Diagnostic (env-gated): print per-microbatch loss / logits magnitude / mask
    # sums on the loss-computing stage, to compare the failing full loop against
    # the passing debug-train-only replay (which reports loss=0.069).
    if os.environ.get("SLIME_DEBUG_LOSS", "0") == "1":
        try:
            _l = loss.detach().float()
            _lg = logits.detach().float()
            print(
                f"[SLIME_DEBUG_LOSS] rank={dist.get_rank() if dist.is_initialized() else -1} "
                f"loss={_l.item():.6e} finite={bool(torch.isfinite(_l).all().item())} "
                f"logits_absmax={_lg.abs().amax().item():.3e} num_tokens={int(num_tokens)} "
                f"mask_sums={[int(m.sum().item()) for m in batch['loss_masks']][:4]}",
                flush=True,
            )
        except Exception as _e:  # never break training from a diagnostic
            print(f"[SLIME_DEBUG_LOSS] error: {_e}", flush=True)

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if not args.calculate_per_token_loss:
        loss = (
            loss
            * num_microbatches
            / step_global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            # values[0] is the consumer's reporting denominator after
            # all-reduce. For per-token-loss it must equal step total tokens
            # (only known by summing per-mb num_tokens across mbs / DP). For
            # per-rollout-mean it is a constant — ``step_global_batch_size`` —
            # so we leave a 0 placeholder here and let ``train_one_step``
            # substitute the constant directly, instead of routing it through
            # per-mb fractions.
            "values": torch.tensor(
                [
                    num_tokens if args.calculate_per_token_loss else 0,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
