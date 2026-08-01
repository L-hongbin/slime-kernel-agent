import math
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu

# --- CP sequence-partition mode -------------------------------------------------
#
# slime's default CP layout is Megatron's ZIGZAG (a.k.a. balanced) partition:
# the padded sequence is cut into ``2*cp_size`` chunks and rank ``r`` owns the
# mirror pair ``{r, 2*cp_size-1-r}`` so causal work is balanced across ranks.
#
# DeepSeek-V4-Flash needs a CONTIGUOUS partition instead: rank ``r`` owns the
# single contiguous block ``[r*l_local, (r+1)*l_local)`` (l_local = S/cp_size),
# because the V4 attention kernel's raw sliding-window / compressor look-back
# assume a monotone local position axis. Implemented as a re-labelling of which
# two of the ``2*cp_size`` chunks each rank owns: contiguous rank ``r`` owns the
# ADJACENT pair ``{2r, 2r+1}`` (their union is exactly ``[r*l_local, (r+1)*l_local)``).
#
# Crucially ``chunk_size`` is IDENTICAL in both modes, so the per-rank local
# tensor still holds ``2*chunk_size`` rows per sample and every downstream
# consumer that reads the returned (chunks/logits/tokens) offsets keeps working
# unchanged -- only *which global positions* those two chunks map to changes.
#
# The mode is a single process-wide setting (set once from args at init) rather
# than a threaded argument, because ``get_logits_and_tokens_offset_with_cp`` is
# also called from arg-less helpers (all_gather_with_cp, slice_log_prob_with_cp,
# get_sum_of_sample_mean). ``cp_size == 1`` is a strict no-op in either mode.
CP_PARTITION_ZIGZAG = "zigzag"
CP_PARTITION_CONTIGUOUS = "contiguous"
_CP_PARTITION_MODE = CP_PARTITION_ZIGZAG


def set_cp_partition_mode(mode: str) -> None:
    """Set the process-wide CP sequence-partition mode (called once from init)."""
    global _CP_PARTITION_MODE
    assert mode in (CP_PARTITION_ZIGZAG, CP_PARTITION_CONTIGUOUS), f"unknown cp_partition_mode: {mode!r}"
    _CP_PARTITION_MODE = mode


def get_cp_partition_mode() -> str:
    return _CP_PARTITION_MODE


def _resolve_cp_partition_mode(partition_mode: str | None) -> str:
    return _CP_PARTITION_MODE if partition_mode is None else partition_mode


def compute_cp_padded_max_seq_len(
    raw_max_seq_len: int,
    pad_size: int,
    cp_size: int,
    partition_mode: str | None = None,
) -> int:
    """Round ``raw_max_seq_len`` up to the bshd pad granularity (B1).

    Under contiguous CP the per-rank block ``l_local = max_seq_len / cp_size`` must
    be 128-aligned (HCA window / compressor look-back), so the pad granularity is
    raised to ``lcm(pad_size, cp_size*128)`` -- the rounded length becomes a
    multiple of ``cp_size*128`` and ``l_local % 128 == 0``. Strict no-op (unchanged
    ``pad_size``) at ``cp_size == 1`` or in the zigzag layout.
    """
    mode = _resolve_cp_partition_mode(partition_mode)
    if cp_size > 1 and mode == CP_PARTITION_CONTIGUOUS:
        pad_size = math.lcm(pad_size, cp_size * 128)
    return (raw_max_seq_len + pad_size - 1) // pad_size * pad_size


def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    partition_mode: str | None = None,
):
    """
    All offsets start from the begining of the prompt.
    """
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size > 1
    mode = _resolve_cp_partition_mode(partition_mode)

    prompt_length = total_length - response_length
    if qkv_format == "thd":
        chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    else:
        assert max_seq_len is not None, "max_seq_len must be provided for qkv_format=bshd"
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # the offset of 2 chunks
    if mode == CP_PARTITION_CONTIGUOUS:
        # Contiguous: rank r owns the adjacent chunk pair {2r, 2r+1}. Their union
        # [2r*chunk_size, (2r+2)*chunk_size) == [r*l_local, (r+1)*l_local).
        chunk_0 = (2 * cp_rank * chunk_size, (2 * cp_rank + 1) * chunk_size)
        chunk_1 = ((2 * cp_rank + 1) * chunk_size, (2 * cp_rank + 2) * chunk_size)
    else:
        chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
        chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)

    # the offset of 2 logits, note that the logits need a "-1".
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))

    # when the sequence is empty, make an empty slice to continue the gradient flow.
    if logits_0[0] < logits_0[1]:
        token_0 = (logits_0[0] + 1, logits_0[1] + 1)
    else:
        logits_0 = (0, 0)
        token_0 = (0, 0)

    if logits_1[0] < logits_1[1]:
        token_1 = (logits_1[0] + 1, logits_1[1] + 1)
    else:
        logits_1 = (0, 0)
        token_1 = (0, 0)

    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)


def get_sum_of_sample_mean(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    sample_denoms: list[torch.Tensor] | torch.Tensor | None = None,
    calculate_per_token_loss: bool = False,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Calculate correct sample mean for CP.

    The default (``sample_denoms=None``) is the legacy per-sample mean: each
    sample's denominator is its own ``loss_mask.sum()``. Callers that want a
    per-rollout token-weighted mean pass pre-computed per-sample denominators
    (already as GPU tensors — see actor side) where every sample in the same
    rollout group carries the same value (the sum of that rollout's mask
    totals across every sibling sample in the step). Pre-computing at the
    step level rather than per-mb is required — otherwise a rollout whose
    samples land in different micro-batches would get a partial denominator
    on each side.
    """
    if sample_denoms is None:
        sample_denoms = [m.sum() for m in loss_masks]

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum() / torch.clamp_min(denom, 1)
                    for x_i, loss_mask_i, denom in zip(
                        x.split(response_lengths, dim=0), loss_masks, sample_denoms, strict=False
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum()
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
                ]
            )

    else:
        cp_chunk_lengths: list[int] = []
        chunked_loss_masks: list[torch.Tensor] = []

        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_masks.append(torch.cat([loss_mask_0, loss_mask_1], dim=0))
            cp_chunk_lengths.append(chunked_loss_masks[i].size(0))

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum() / torch.clamp_min(denom, 1)
                    for x_i, chunked_loss_mask, denom in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, sample_denoms, strict=False
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum()
                    for x_i, chunked_loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, strict=False
                    )
                ]
            )

    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token


def reduce_train_step_metrics(
    losses_reduced: list[dict],
    *,
    calculate_per_token_loss: bool,
    step_global_batch_size: int,
    cp_size: int,
    dp_with_cp_group,
) -> dict[str, float]:
    """Aggregate per-mb log dicts into the dict ``train_one_step`` reports.

    Pipeline (1:1 with what the train loop used to do inline):
      1. Sum each metric's per-mb ``values`` tensor locally on this rank.
      2. All-reduce across the DP*CP group (``dp_with_cp_group``).
      3. Apply the per-mode divisor / cp_factor:
         - per-token-loss: divisor = ``values[0]`` = all-reduced ``num_tokens``,
           CP-inflated by ``cp_size`` because every CP rank computes the same
           num_tokens off the FULL (not chunked) masks; the
           ``cp_factor = cp_size`` multiplier cancels that inflation, leaving
           the genuine per-token average.
         - per-rollout-mean: divisor = constant ``step_global_batch_size`` from
           the rollout side, never all-reduced, so no CP inflation to cancel
           and ``cp_factor = 1``.

    Tests pass a mock ``dp_with_cp_group`` and monkeypatch ``dist.all_reduce``
    to a no-op, then pre-aggregate virtual ranks themselves — this exercises
    the same call shape as production while staying single-process.
    """
    keys = losses_reduced[0]["keys"]
    values = None
    for x in losses_reduced:
        values = x["values"] if values is None else values + x["values"]
    assert len(keys) + 1 == values.numel()
    dist.all_reduce(values, group=dp_with_cp_group)
    values = values.tolist()

    if calculate_per_token_loss:
        num_samples_or_tokens = values[0]
        cp_factor = cp_size
    else:
        num_samples_or_tokens = step_global_batch_size
        cp_factor = 1
    return {key: value * cp_factor / num_samples_or_tokens for key, value in zip(keys, values[1:], strict=False)}


def rollout_log_metric_contribution(
    per_rank_reducer_sum: float,
    *,
    cp_size: int,
    num_rollouts_in_rollout: int,
    dp_size: int,
) -> tuple[float, float]:
    """``(sum, count)`` tuple to hand the gather step for a per-rollout-mean
    metric on the rollout side (``log_rollout_data``).

    Sum across DP*CP ranks of ``count`` lands on ``num_rollouts_in_rollout``
    (``dp_size`` here is the no-CP DP width; the gather covers ``dp_size *
    cp_size`` ranks, and each rank emits the same ``count``, so the totals
    cancel out the ``cp_size`` in the sum). Result: ``Σsum / Σcount =
    sum_DP_full / num_rollouts`` — the same number ``train_one_step`` reports
    for the same samples (when ``num_steps_per_rollout == 1``).

    Pair with :func:`gather_and_reduce_log_dict` to do the full end-to-end
    in tests (single helper call per rank, returns the reduced number on
    the source rank).
    """
    sum_value = cp_size * per_rank_reducer_sum
    count = num_rollouts_in_rollout / dp_size
    return sum_value, count


def gather_and_reduce_log_dict(
    log_dict: dict,
    *,
    dp_size: int,
    dp_src_rank: int,
    dp_group,
) -> dict | None:
    """``dist.gather_object`` per-rank log_dicts + per-key reduction.

    Per key in the gathered dicts:
      - ``(sum, count)`` tuple → ``Σsum / Σcount`` (per-rollout-mean shape;
        pair with :func:`rollout_log_metric_contribution`).
      - plain value → ``Σ / dp_size`` (legacy mean-across-ranks; the only
        correct answer when ranks hold the same data).

    Returns the reduced dict on ``dp_src_rank``, ``None`` elsewhere. The
    caller adds whatever metric-name prefix / wandb plumbing it wants —
    this helper stays free of side effects so CPU multi-process unit tests
    can drive it directly with real ``torch.distributed``.
    """
    if dist.get_rank() == dp_src_rank:
        gathered = [None] * dp_size
        dist.gather_object(log_dict, gathered, dst=dp_src_rank, group=dp_group)
        reduced: dict = {}
        for key in log_dict:
            values = [d[key] for d in gathered]
            first = values[0]
            if isinstance(first, tuple) and len(first) == 2:
                total_sum = sum(v[0] for v in values)
                total_count = sum(v[1] for v in values)
                reduced[key] = total_sum / total_count if total_count else 0.0
            else:
                reduced[key] = sum(values) / dp_size
        return reduced
    dist.gather_object(log_dict, None, dst=dp_src_rank, group=dp_group)
    return None


def all_gather_with_cp(
    tensor: torch.Tensor,
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """
    Gather tensors across all ranks in the context parallel group.
    The first dimension of the output tensor will be the `response_length`.

    ``tensor`` must be interpreted with the same physical sequence layout that
    produced it.  In particular, BSHD actual-length padding partitions CP by
    the padded microbatch width (``max_seq_len``), not by ``total_length``.
    Omitting that width would recompute different chunk boundaries and either
    misplace rows or fail the local-shape assertion below.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return tensor

    if qkv_format == "bshd" and max_seq_len is None:
        raise ValueError("all_gather_with_cp requires max_seq_len for qkv_format='bshd'.")
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length,
        response_length,
        qkv_format,
        max_seq_len,
    )

    prompt_length = total_length - response_length

    chunk_0_length = logits_offset[0][1] - logits_offset[0][0]
    chunk_1_length = logits_offset[1][1] - logits_offset[1][0]
    expected_local_length = chunk_0_length + chunk_1_length
    if tensor.shape[0] != expected_local_length:
        raise AssertionError(
            "all_gather_with_cp local response length does not match its CP layout: "
            f"actual={tensor.shape[0]}, expected={expected_local_length} "
            f"(chunks={chunk_0_length}+{chunk_1_length}), total_length={total_length}, "
            f"response_length={response_length}, qkv_format={qkv_format!r}, "
            f"max_seq_len={max_seq_len}, logits_offset={logits_offset}."
        )
    chunk_0 = tensor[:chunk_0_length]
    chunk_1 = tensor[chunk_0_length:]

    def zero(len: int) -> torch.Tensor:
        return torch.zeros(
            [len] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=True,
        )

    # logprob should be within the range of [prompt_length - 1, total_length - 1]
    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        # all empty
        full_tensor = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        # only first chunk
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        full_tensor = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        # only second chunk
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    assert full_tensor.shape[0] == response_length, f"Expected {response_length}, got {full_tensor.shape}"
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
    return full_tensor


def slice_with_cp(
    tokens: torch.Tensor,
    pad_value: tuple[int, float, Callable],
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    partition_mode: str | None = None,
) -> torch.Tensor:
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    mode = _resolve_cp_partition_mode(partition_mode)

    if qkv_format == "bshd":
        assert max_seq_len is not None

    def pad_tokens(tokens, pad):
        if isinstance(pad_value, Callable):
            pad_func = pad_value
            tokens = pad_func(tokens, pad)
        else:
            # pad on the first dimension
            pad_tuple = (0, 0) * (tokens.dim() - 1) + (0, pad)
            tokens = F.pad(tokens, pad_tuple, value=pad_value)
        return tokens

    if cp_size == 1:
        if qkv_format == "bshd":
            pad = max_seq_len - tokens.size(0)
            tokens = pad_tokens(tokens, pad)
        return tokens

    token_len = len(tokens)
    if qkv_format == "thd":
        chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    else:
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # pad
    pad = 2 * cp_size * chunk_size - token_len
    tokens = pad_tokens(tokens, pad)

    if mode == CP_PARTITION_CONTIGUOUS:
        # Contiguous partition: rank r owns the adjacent chunk pair {2r, 2r+1},
        # i.e. a single contiguous block [r*l_local, (r+1)*l_local) with
        # l_local = 2*chunk_size. The V4 attention kernel needs l_local aligned
        # to the compressor / sliding-window granularity (128); B1 raises the
        # max_seq_len pad multiplier to lcm(*, cp_size*128) upstream to guarantee
        # it -- assert here so a mis-configured pad fails loud instead of
        # silently scrambling the local position axis.
        l_local = 2 * chunk_size
        if qkv_format == "bshd":
            assert l_local % 128 == 0, (
                f"contiguous CP requires l_local (=max_seq_len/cp_size={l_local}) to be a multiple of 128; "
                f"got max_seq_len={max_seq_len}, cp_size={cp_size}. Raise the max_seq_len pad multiplier "
                f"to a multiple of cp_size*128 (B1)."
            )
        start_1, end_1 = 2 * cp_rank * chunk_size, (2 * cp_rank + 1) * chunk_size
        start_2, end_2 = (2 * cp_rank + 1) * chunk_size, (2 * cp_rank + 2) * chunk_size
    else:
        # get 2 chunk for zigzag cp
        start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
        start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])


def slice_log_prob_with_cp(
    log_prob: list[float] | torch.Tensor,
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_token_len: int | None = None,
) -> list[float] | torch.Tensor:
    assert len(log_prob) == response_length, (
        f"log_prob length mismatch: len(log_prob)={len(log_prob)}, "
        f"response_length={response_length}, total_length={total_length}"
    )

    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return log_prob

    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_token_len
    )

    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) : logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) : logits_offset[1][1] - (prompt_length - 1)]

    if isinstance(log_prob, list):
        return chunk_1 + chunk_2
    else:
        return torch.cat([chunk_1, chunk_2], dim=0)
