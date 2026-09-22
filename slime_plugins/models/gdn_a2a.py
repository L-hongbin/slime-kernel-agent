"""Fused packing and reusable communication workspaces for GDN all-to-all."""

from __future__ import annotations

import threading
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _pack_rank_major_kernel(
    source,
    packed,
    widths,
    prefixes,
    permutation,
    rows,
    batch,
    total_width: tl.constexpr,
    source_stride_0,
    source_stride_1,
    source_stride_2,
    HAS_PERMUTATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = (tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    rank = tl.program_id(1)
    width = tl.load(widths + rank).to(tl.int64)
    prefix = tl.load(prefixes + rank).to(tl.int64)
    mask = offsets < rows * width
    row = offsets // width
    local_column = offsets - row * width
    packed_column = prefix + local_column
    if HAS_PERMUTATION:
        source_column = tl.load(permutation + packed_column, mask=mask, other=0).to(tl.int64)
    else:
        source_column = packed_column
    sequence = row // batch
    batch_index = row - sequence * batch
    source_offsets = sequence * source_stride_0 + batch_index * source_stride_1 + source_column * source_stride_2
    values = tl.load(source + source_offsets, mask=mask)
    tl.store(packed + rows * prefix + offsets, values, mask=mask)


@triton.jit
def _unpack_rank_major_kernel(
    packed,
    output,
    widths,
    prefixes,
    permutation,
    rows,
    total_width: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = (tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    rank = tl.program_id(1)
    width = tl.load(widths + rank).to(tl.int64)
    prefix = tl.load(prefixes + rank).to(tl.int64)
    mask = offsets < rows * width
    row = offsets // width
    local_column = offsets - row * width
    packed_column = prefix + local_column
    if HAS_PERMUTATION:
        output_column = tl.load(permutation + packed_column, mask=mask, other=0).to(tl.int64)
    else:
        output_column = packed_column
    values = tl.load(packed + rows * prefix + offsets, mask=mask)
    tl.store(output + row * total_width + output_column, values, mask=mask)


@triton.jit
def _unpack_sequence_kernel(
    packed,
    output,
    permutation,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    sequence = tl.program_id(0).to(tl.int64)
    offsets = (tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offsets < BATCH * WIDTH
    output_sequence = tl.load(permutation + sequence).to(tl.int64)
    output_offsets = output_sequence * BATCH * WIDTH + offsets
    values = tl.load(packed + sequence * BATCH * WIDTH + offsets, mask=mask)
    tl.store(output + output_offsets, values, mask=mask)


@triton.jit
def _unpack_equal_kernel(
    packed,
    output,
    rows,
    LOCAL_WIDTH: tl.constexpr,
    TOTAL_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = (tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = column < TOTAL_WIDTH
    source_rank = column // LOCAL_WIDTH
    local_column = column - source_rank * LOCAL_WIDTH
    packed_offsets = source_rank * rows * LOCAL_WIDTH + row * LOCAL_WIDTH + local_column
    values = tl.load(packed + packed_offsets, mask=mask)
    tl.store(output + row * TOTAL_WIDTH + column, values, mask=mask)


_workspace_lock = threading.Lock()
_communication_workspaces: dict[tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}


def _workspace_key(reference: torch.Tensor, slot: int) -> tuple[torch.device, torch.dtype, int, int]:
    if reference.is_cuda:
        stream = torch.cuda.current_stream(reference.device).cuda_stream
    else:
        stream = threading.get_ident()
    return reference.device, reference.dtype, stream, slot


def _get_communication_workspace(reference: torch.Tensor, numel: int, slot: int = 0) -> torch.Tensor:
    """Return a grow-only scratch buffer scoped to device, dtype, and execution stream."""
    key = _workspace_key(reference, slot)
    with _workspace_lock:
        workspace = _communication_workspaces.get(key)
        if workspace is None or workspace.numel() < numel:
            workspace = torch.empty(numel, dtype=reference.dtype, device=reference.device)
            _communication_workspaces[key] = workspace
    return workspace[:numel]


def clear_communication_workspaces() -> None:
    """Drop cached GDN all-to-all scratch buffers, primarily for tests."""
    with _workspace_lock:
        _communication_workspaces.clear()


@lru_cache(maxsize=32)
def _layout_tensors(rank_widths: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes = [0]
    for width in rank_widths[:-1]:
        prefixes.append(prefixes[-1] + width)
    return (
        torch.tensor(rank_widths, dtype=torch.int64, device=device),
        torch.tensor(prefixes, dtype=torch.int64, device=device),
    )


def _validate_permutation(permutation: torch.Tensor | None, total_width: int, device: torch.device) -> None:
    if permutation is None:
        return
    if permutation.device != device or permutation.dim() != 1 or permutation.numel() != total_width:
        raise ValueError(
            f"GDN head permutation must be a 1-D tensor of size {total_width} on {device}, "
            f"got shape={tuple(permutation.shape)} on {permutation.device}."
        )


def _pack_rank_major(
    source: torch.Tensor,
    rank_widths: tuple[int, ...],
    permutation: torch.Tensor | None,
) -> torch.Tensor:
    """Pack a [sequence, batch, width] tensor directly into rank-major scratch storage."""
    total_width = sum(rank_widths)
    if source.dim() != 3 or source.size(-1) != total_width:
        raise ValueError(f"GDN pack expects a 3-D tensor of width {total_width}, got shape {tuple(source.shape)}.")
    _validate_permutation(permutation, total_width, source.device)
    rows = source.size(0) * source.size(1)
    packed = _get_communication_workspace(source, rows * total_width)

    if source.is_cuda:
        widths, prefixes = _layout_tensors(rank_widths, source.device)
        max_width = max(rank_widths)
        grid = (triton.cdiv(rows * max_width, 256), len(rank_widths))
        _pack_rank_major_kernel[grid](
            source,
            packed,
            widths,
            prefixes,
            permutation if permutation is not None else source,
            rows,
            source.size(1),
            total_width,
            source.stride(0),
            source.stride(1),
            source.stride(2),
            HAS_PERMUTATION=permutation is not None,
            BLOCK_SIZE=256,
        )
    else:
        reordered = source.index_select(-1, permutation) if permutation is not None else source
        offset = 0
        for chunk in torch.split(reordered, rank_widths, dim=-1):
            flattened = chunk.contiguous().view(-1)
            packed[offset : offset + flattened.numel()].copy_(flattened)
            offset += flattened.numel()
    return packed


def _unpack_rank_major(
    packed: torch.Tensor,
    output: torch.Tensor,
    rank_widths: tuple[int, ...],
    permutation: torch.Tensor | None,
) -> None:
    """Unpack rank-major communication storage directly into a row-major output."""
    total_width = sum(rank_widths)
    if output.dim() != 3 or output.size(-1) != total_width or not output.is_contiguous():
        raise ValueError(
            f"GDN unpack expects a contiguous 3-D output of width {total_width}, got {tuple(output.shape)}."
        )
    _validate_permutation(permutation, total_width, output.device)
    rows = output.size(0) * output.size(1)
    if packed.numel() != rows * total_width:
        raise ValueError(f"Packed input has {packed.numel()} elements, expected {rows * total_width}.")

    if output.is_cuda:
        widths, prefixes = _layout_tensors(rank_widths, output.device)
        max_width = max(rank_widths)
        grid = (triton.cdiv(rows * max_width, 256), len(rank_widths))
        _unpack_rank_major_kernel[grid](
            packed,
            output,
            widths,
            prefixes,
            permutation if permutation is not None else output,
            rows,
            total_width,
            HAS_PERMUTATION=permutation is not None,
            BLOCK_SIZE=256,
        )
    else:
        output_2d = output.view(rows, total_width)
        offset = 0
        prefix = 0
        for width in rank_widths:
            chunk = packed[offset : offset + rows * width].view(rows, width)
            if permutation is None:
                output_2d[:, prefix : prefix + width].copy_(chunk)
            else:
                output_2d.index_copy_(1, permutation[prefix : prefix + width], chunk)
            offset += rows * width
            prefix += width


def _validate_sequence_permutation(permutation: torch.Tensor, sequence_length: int, device: torch.device) -> None:
    if permutation.device != device or permutation.dim() != 1 or permutation.numel() != sequence_length:
        raise ValueError(
            f"GDN sequence permutation must be a 1-D tensor of size {sequence_length} on {device}, "
            f"got shape={tuple(permutation.shape)} on {permutation.device}."
        )


def _pack_sequence(
    source: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Gather sequence rows directly into reusable all-to-all send storage."""
    if source.dim() != 3:
        raise ValueError(f"GDN sequence pack expects a 3-D tensor, got shape {tuple(source.shape)}.")
    _validate_sequence_permutation(permutation, source.size(0), source.device)
    packed = _get_communication_workspace(source, source.numel(), slot=0)
    torch.index_select(source, 0, permutation, out=packed.view_as(source))
    return packed


def _unpack_sequence(
    packed: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
) -> None:
    """Scatter rank-major sequence rows back into their original order."""
    if output.dim() != 3 or not output.is_contiguous() or packed.numel() != output.numel():
        raise ValueError(
            f"GDN sequence unpack expects a contiguous 3-D output with {packed.numel()} elements, "
            f"got shape {tuple(output.shape)}."
        )
    _validate_sequence_permutation(permutation, output.size(0), output.device)
    if output.is_cuda:
        row_width = output.size(1) * output.size(2)
        block_size = min(1024, triton.next_power_of_2(row_width))
        grid = (output.size(0), triton.cdiv(row_width, block_size))
        _unpack_sequence_kernel[grid](
            packed,
            output,
            permutation,
            BATCH=output.size(1),
            WIDTH=output.size(2),
            BLOCK_SIZE=block_size,
        )
    else:
        output.index_copy_(0, permutation, packed.view_as(output))


def _unpack_equal(packed: torch.Tensor, output: torch.Tensor, cp_size: int) -> None:
    """Transpose equal source-rank chunks into the restored hidden dimension."""
    if output.dim() != 3 or not output.is_contiguous() or output.size(-1) % cp_size != 0:
        raise ValueError(
            f"Equal GDN unpack expects a contiguous 3-D output whose width is divisible by cp_size={cp_size}, "
            f"got shape {tuple(output.shape)}."
        )
    if packed.numel() != output.numel():
        raise ValueError(f"Packed input has {packed.numel()} elements, expected {output.numel()}.")
    local_width = output.size(-1) // cp_size
    rows = output.size(0) * output.size(1)
    if output.is_cuda:
        grid = (rows, triton.cdiv(output.size(-1), 1024))
        _unpack_equal_kernel[grid](
            packed,
            output,
            rows,
            LOCAL_WIDTH=local_width,
            TOTAL_WIDTH=output.size(-1),
            BLOCK_SIZE=1024,
        )
    else:
        chunks = torch.split(packed, rows * local_width)
        torch.cat([chunk.view(output.size(0), output.size(1), local_width) for chunk in chunks], dim=-1, out=output)


def _all_to_all_single(
    output: torch.Tensor,
    input_: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group: torch.distributed.ProcessGroup,
) -> None:
    torch.distributed.all_to_all_single(
        output,
        input_,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )


class _FusedCPToHP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, rank_widths, cp_group, permutation, sequence_permutation):
        cp_size = cp_group.size()
        local_seq_len, batch, total_width = tensor.shape
        local_width = rank_widths[cp_group.rank()]
        rows = local_seq_len * batch
        send = _pack_rank_major(tensor, rank_widths, permutation)
        output = tensor.new_empty(local_seq_len * cp_size, batch, local_width)
        if sequence_permutation is not None:
            _validate_sequence_permutation(sequence_permutation, output.size(0), tensor.device)
            received = _get_communication_workspace(tensor, output.numel(), slot=1)
        else:
            received = output.view(-1)
        _all_to_all_single(
            received,
            send,
            [rows * local_width] * cp_size,
            [rows * width for width in rank_widths],
            cp_group,
        )
        if sequence_permutation is not None:
            torch.index_select(received.view_as(output), 0, sequence_permutation, out=output)

        ctx.rank_widths = rank_widths
        ctx.cp_group = cp_group
        ctx.input_shape = tensor.shape
        ctx.permutation = permutation
        ctx.sequence_permutation = sequence_permutation
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rank_widths = ctx.rank_widths
        cp_group = ctx.cp_group
        local_seq_len, batch, total_width = ctx.input_shape
        local_width = rank_widths[cp_group.rank()]
        rows = local_seq_len * batch
        if ctx.sequence_permutation is not None:
            send = _get_communication_workspace(grad_output, grad_output.numel(), slot=1)
            # This is a bijection: scatter directly instead of index_select's
            # generic zero-fill/index-add backward. Do not retain scratch in ctx.
            _unpack_sequence(grad_output.contiguous(), send.view_as(grad_output), ctx.sequence_permutation)
        else:
            send = grad_output.contiguous().view(-1)
        received = _get_communication_workspace(grad_output, rows * total_width)
        _all_to_all_single(
            received,
            send,
            [rows * width for width in rank_widths],
            [rows * local_width] * cp_group.size(),
            cp_group,
        )
        grad_input = grad_output.new_empty(ctx.input_shape)
        _unpack_rank_major(received, grad_input, rank_widths, ctx.permutation)
        return grad_input, None, None, None, None


class _RaggedHPToCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, rank_widths, cp_group, sequence_permutation):
        cp_size = cp_group.size()
        total_seq_len, batch, local_width = tensor.shape
        local_seq_len = total_seq_len // cp_size
        total_width = sum(rank_widths)
        rows = local_seq_len * batch
        send = (
            _pack_sequence(tensor, sequence_permutation)
            if sequence_permutation is not None
            else tensor.contiguous().view(-1)
        )
        received = _get_communication_workspace(tensor, rows * total_width, slot=1)
        _all_to_all_single(
            received,
            send,
            [rows * width for width in rank_widths],
            [rows * local_width] * cp_size,
            cp_group,
        )
        output = tensor.new_empty(local_seq_len, batch, total_width)
        _unpack_rank_major(received, output, rank_widths, None)

        ctx.rank_widths = rank_widths
        ctx.cp_group = cp_group
        ctx.input_shape = tensor.shape
        ctx.sequence_permutation = sequence_permutation
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rank_widths = ctx.rank_widths
        cp_group = ctx.cp_group
        total_seq_len, batch, local_width = ctx.input_shape
        local_seq_len = total_seq_len // cp_group.size()
        rows = local_seq_len * batch
        send = _pack_rank_major(grad_output, rank_widths, None)
        grad_input = grad_output.new_empty(ctx.input_shape)
        received = (
            _get_communication_workspace(grad_output, grad_input.numel(), slot=1)
            if ctx.sequence_permutation is not None
            else grad_input.view(-1)
        )
        _all_to_all_single(
            received,
            send,
            [rows * local_width] * cp_group.size(),
            [rows * width for width in rank_widths],
            cp_group,
        )
        if ctx.sequence_permutation is not None:
            _unpack_sequence(received, grad_input, ctx.sequence_permutation)
        return grad_input, None, None, None


class _FusedHPToCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, cp_group, sequence_permutation):
        cp_size = cp_group.size()
        total_seq_len, batch, local_width = tensor.shape
        if total_seq_len % cp_size != 0:
            raise ValueError(
                f"Fused HP-to-CP requires sequence length divisible by cp_size={cp_size}, "
                f"got shape {tuple(tensor.shape)}."
            )
        local_seq_len = total_seq_len // cp_size
        chunk_size = local_seq_len * batch * local_width
        send = _pack_sequence(tensor, sequence_permutation)
        received = _get_communication_workspace(tensor, tensor.numel(), slot=1)
        _all_to_all_single(
            received,
            send,
            [chunk_size] * cp_size,
            [chunk_size] * cp_size,
            cp_group,
        )
        output = tensor.new_empty(local_seq_len, batch, local_width * cp_size)
        _unpack_equal(received, output, cp_size)

        ctx.cp_group = cp_group
        ctx.input_shape = tensor.shape
        ctx.sequence_permutation = sequence_permutation
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        cp_group = ctx.cp_group
        cp_size = cp_group.size()
        total_seq_len, batch, local_width = ctx.input_shape
        local_seq_len = total_seq_len // cp_size
        chunk_size = local_seq_len * batch * local_width
        send = _pack_rank_major(grad_output, (local_width,) * cp_size, None)
        received = _get_communication_workspace(grad_output, total_seq_len * batch * local_width, slot=1)
        _all_to_all_single(
            received,
            send,
            [chunk_size] * cp_size,
            [chunk_size] * cp_size,
            cp_group,
        )
        grad_input = grad_output.new_empty(ctx.input_shape)
        _unpack_sequence(received, grad_input, ctx.sequence_permutation)
        return grad_input, None, None


def fused_cp_to_hp(
    tensor: torch.Tensor,
    rank_widths: tuple[int, ...],
    cp_group: torch.distributed.ProcessGroup,
    permutation: torch.Tensor | None = None,
    sequence_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    return _FusedCPToHP.apply(tensor, rank_widths, cp_group, permutation, sequence_permutation)


def ragged_hp_to_cp(
    tensor: torch.Tensor,
    rank_widths: tuple[int, ...],
    cp_group: torch.distributed.ProcessGroup,
    sequence_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    return _RaggedHPToCP.apply(tensor, rank_widths, cp_group, sequence_permutation)


def fused_hp_to_cp(
    tensor: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    sequence_permutation: torch.Tensor,
) -> torch.Tensor:
    return _FusedHPToCP.apply(tensor, cp_group, sequence_permutation)
