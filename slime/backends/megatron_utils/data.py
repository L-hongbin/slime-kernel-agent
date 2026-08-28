from collections import Counter
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from slime.utils import accelerator
from slime.utils.types import RolloutBatch

from .cp_utils import compute_cp_padded_max_seq_len, slice_with_cp


def compute_bshd_max_seq_lens(
    total_lengths: Sequence[int],
    micro_batch_indices: Sequence[Sequence[int]],
    *,
    pad_size: int,
    cp_size: int,
    cp_partition_mode: str,
    pipeline_model_parallel_size: int = 1,
) -> list[int]:
    """Compute the padded BSHD width used by each scheduled sample.

    ``get_batch`` stacks all samples in one microbatch, so every sample in that
    microbatch must use the same width: the aligned maximum of its real
    ``total_length`` values.  With PP1, different microbatches may use different
    widths and avoid padding every sample to the longest sequence in the whole
    rollout.

    V4 PP communication shapes are currently fixed once per Megatron pipeline
    schedule.  Until that schedule is split by sequence shape, PP>1 must keep a
    rollout-wide width; doing otherwise would make the sender and receiver
    allocate different P2P buffers.  The explicit fallback here preserves the
    previously validated PP behavior instead of silently enabling unsafe
    per-microbatch shapes.

    The schedule is also validated as an exact partition of local samples.  A
    duplicate or missing index would otherwise leave a sample with a stale or
    unrelated padding width and can corrupt BSHD loss/routing offsets.
    """
    if pad_size <= 0:
        raise ValueError(f"pad_size must be positive, got {pad_size}")
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if pipeline_model_parallel_size <= 0:
        raise ValueError("pipeline_model_parallel_size must be positive, " f"got {pipeline_model_parallel_size}")

    num_samples = len(total_lengths)
    if num_samples == 0:
        if micro_batch_indices:
            raise ValueError("empty total_lengths requires an empty microbatch schedule")
        return []

    max_seq_lens: list[int | None] = [None] * num_samples
    microbatch_widths: list[int] = []
    for microbatch_id, indices in enumerate(micro_batch_indices):
        if not indices:
            raise ValueError(f"microbatch {microbatch_id} is empty")

        seen_in_microbatch: set[int] = set()
        raw_max_seq_len = 0
        for sample_index in indices:
            if not isinstance(sample_index, int):
                raise TypeError(f"microbatch {microbatch_id} has non-integer sample index " f"{sample_index!r}")
            if sample_index < 0 or sample_index >= num_samples:
                raise IndexError(
                    f"microbatch {microbatch_id} sample index {sample_index} is outside " f"[0, {num_samples})"
                )
            if sample_index in seen_in_microbatch or max_seq_lens[sample_index] is not None:
                raise ValueError(f"sample index {sample_index} appears more than once in the schedule")
            seen_in_microbatch.add(sample_index)

            total_length = int(total_lengths[sample_index])
            if total_length <= 0:
                raise ValueError(f"total_lengths[{sample_index}] must be positive, got {total_length}")
            raw_max_seq_len = max(raw_max_seq_len, total_length)

        padded_width = compute_cp_padded_max_seq_len(
            raw_max_seq_len,
            pad_size,
            cp_size,
            cp_partition_mode,
        )
        microbatch_widths.append(padded_width)
        for sample_index in indices:
            max_seq_lens[sample_index] = padded_width

    missing = [i for i, width in enumerate(max_seq_lens) if width is None]
    if missing:
        raise ValueError(f"microbatch schedule is missing sample indices {missing}")

    if pipeline_model_parallel_size > 1:
        rollout_width = max(microbatch_widths)
        return [rollout_width] * num_samples

    return [int(width) for width in max_seq_lens]


def summarize_bshd_padding(
    total_lengths: Sequence[int],
    max_seq_lens: Sequence[int],
) -> dict[str, int | float | str]:
    """Return compact, log-friendly actual-length padding statistics."""
    if len(total_lengths) != len(max_seq_lens):
        raise ValueError(
            "total_lengths and max_seq_lens must have the same size, got "
            f"{len(total_lengths)} and {len(max_seq_lens)}"
        )
    if not total_lengths:
        return {
            "samples": 0,
            "unique_widths": 0,
            "width_hist": "",
            "raw_slots": 0,
            "padded_slots": 0,
            "rollout_wide_slots": 0,
            "padding_overhead_pct": 0.0,
            "saved_vs_rollout_wide_pct": 0.0,
        }

    raw_lengths = [int(length) for length in total_lengths]
    padded_widths = [int(width) for width in max_seq_lens]
    for sample_index, (raw_length, padded_width) in enumerate(zip(raw_lengths, padded_widths, strict=True)):
        if raw_length <= 0:
            raise ValueError(f"total_lengths[{sample_index}] must be positive, got {raw_length}")
        if padded_width < raw_length:
            raise ValueError(
                f"max_seq_lens[{sample_index}]={padded_width} is smaller than "
                f"total_lengths[{sample_index}]={raw_length}"
            )

    width_counts = Counter(padded_widths)
    raw_slots = sum(raw_lengths)
    padded_slots = sum(padded_widths)
    rollout_wide_slots = max(padded_widths) * len(padded_widths)
    return {
        "samples": len(padded_widths),
        "unique_widths": len(width_counts),
        "width_hist": ",".join(f"{width}:{width_counts[width]}" for width in sorted(width_counts)),
        "raw_slots": raw_slots,
        "padded_slots": padded_slots,
        "rollout_wide_slots": rollout_wide_slots,
        "padding_overhead_pct": 100.0 * (padded_slots - raw_slots) / raw_slots,
        "saved_vs_rollout_wide_pct": 100.0 * (rollout_wide_slots - padded_slots) / rollout_wide_slots,
    }


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    qkv_format: str = "thd",
    allgather_cp: bool = False,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    """
    Generate a CP-ready micro-batch with packed sequence parameters.

    Steps:
    - Fetch raw fields via iterator.
    - Save original token tensors under "unconcat_tokens".
    - Slice tokens into two chunks for Context Parallelism (CP), concatenate, and pad to a configurable multiple.
    - Build cu_seqlens and `PackedSeqParams` with T-H-D layout (T: sequence length, H: attention heads, D: head dimension).

    Args:
        data_iterator: Iterator providing micro-batch data.
        keys: List of keys to fetch from the iterator.
        pad_multiplier: Multiplier for padding size calculation (default: 128).

    Returns a dict including:
    - "tokens": torch.LongTensor of shape [1, T_padded] on the current CUDA device
    - "unconcat_tokens": list[torch.LongTensor] for the micro-batch before CP slicing/concat
    - "packed_seq_params": PackedSeqParams with T-H-D settings (cu_seqlens on CUDA, dtype=int)
    Plus any other requested keys forwarded from the iterator.
    """

    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    tokens = batch["tokens"]
    # use 0 as the pad token id should be fine?
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if qkv_format == "bshd":
        max_seq_lens = batch["max_seq_lens"]
        assert max_seq_lens and all(width == max_seq_lens[0] for width in max_seq_lens), (
            "bshd samples in one microbatch must share max_seq_len, got " f"{max_seq_lens}"
        )
        max_seqlen = max_seq_lens[0]
        assert max([t.size(0) for t in tokens]) <= max_seqlen
        tokens = [slice_with_cp(t, pad_token_id, qkv_format, max_seqlen) for t in tokens]
        tokens = torch.stack(tokens)
        packed_seq_params = None
    elif qkv_format == "thd":
        if allgather_cp:
            # DSA mode: concatenate all sequences first, then slice once with CP.
            # We also pad the global stream to make per-rank chunks equal.
            cu_seqlens_list: list[int] = [0]
            for token_ids in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + token_ids.size(0))

            tokens = torch.cat(tokens, dim=0)
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

            cu_seqlens = torch.tensor(
                cu_seqlens_list,
                dtype=torch.int,
                device=accelerator.current_device(),
            )
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        else:
            tokens = [slice_with_cp(token_ids, pad_token_id, qkv_format) for token_ids in tokens]

            cu_seqlens = [0]
            for token_ids in tokens:
                cu_seqlens.append(cu_seqlens[-1] + token_ids.size(0))

            tokens = torch.cat(tokens)
            pad = (pad_size - tokens.size(0) % pad_size) % pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens.append(cu_seqlens[-1] + pad)

            # THD requires cu_seqlens in the original (pre-CP) lengths.
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int, device=accelerator.current_device()) * cp_size

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )
        tokens = tokens.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params

    # loss masks
    loss_masks = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
        strict=True,
    ):
        prompt_length = total_length - response_length
        # Align mask to token stream positions (prompt_length-1 left pad, 1 right pad)
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        if qkv_format == "thd" and allgather_cp:
            loss_masks.append(loss_mask)
            continue
        loss_mask = slice_with_cp(
            loss_mask,
            0,
            qkv_format,
            max_seqlen if qkv_format == "bshd" else None,
        )
        loss_masks.append(loss_mask)

    if qkv_format == "bshd":
        loss_masks = torch.stack(loss_masks)
    elif allgather_cp:
        # DSA: concatenate first (same as tokens), pad globally (same pad as above), then slice once.
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    else:
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}  # key -> concatenated tensor
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
        batch["multimodal_train_inputs"] = multimodal_data

    return batch


class DataIterator:
    """Iterator over a rollout dict following an explicit micro-batch index schedule."""

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_indices: list[list[int]],
    ) -> None:
        """Initialize an iterator over ``rollout_data``.

        Args:
            rollout_data: Dict of per-sample fields for this DP rank.
            micro_batch_indices: List of mbs, each mbs being the local sample indices to select.
        """
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        indices = self.micro_batch_indices[self.offset]
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                batch[key] = [vals[i] for i in indices]
        self.offset += 1
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def get_data_iterator(rollout_data: RolloutBatch) -> list[DataIterator]:
    """Build one ``DataIterator`` per VPP stage from the pre-computed schedule in ``rollout_data``."""
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    micro_batch_indices = rollout_data["micro_batch_indices"]
    return [DataIterator(rollout_data, micro_batch_indices) for _ in range(vpp_size)]


def tensors_to_cpu(tensor_list):
    """Move a list of GPU tensors to CPU for Ray object store transfer.

    Args:
        tensor_list: List of GPU tensors, or None.

    Returns:
        List of CPU tensors (detached), or None if input is None.
    """
    if tensor_list is None:
        return None
    return [t.detach().cpu() for t in tensor_list]


def tensors_to_gpu(tensor_list, device=None):
    """Move a list of CPU tensors back to GPU.

    Args:
        tensor_list: List of CPU tensors, or None.
        device: Target CUDA device. If None, uses current device.

    Returns:
        List of GPU tensors, or None if input is None.
    """
    if tensor_list is None:
        return None
    if device is None:
        device = accelerator.current_device()
    return [t.to(device=device, dtype=torch.float32) for t in tensor_list]
