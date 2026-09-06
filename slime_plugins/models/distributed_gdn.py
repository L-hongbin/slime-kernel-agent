"""Packed-THD distributed GDN for the Megatron version pinned by slime.

The pinned Megatron already shards GDN projections across tensor parallel ranks
and converts context-parallel sequence shards into head shards before running
the recurrent kernel.  Its missing piece is packed-sequence support.  This
module backports the packed THD permutation from newer Megatron releases while
keeping the recurrent kernel selectable between FLA and FlashQLA.
"""

from functools import lru_cache, partial

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, causal_conv1d, tensor_a2a_cp2hp, tensor_a2a_hp2cp
from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

from .qwen_gdn_backend import get_chunk_gated_delta_rule


def _build_gdn_head_shards(
    num_key_heads: int, num_value_heads: int, cp_size: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Balance whole GQA head groups across CP ranks without splitting a group."""
    if num_key_heads <= 0 or num_value_heads <= 0:
        raise ValueError(f"GDN head counts must be positive, got key={num_key_heads}, value={num_value_heads}.")
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}.")
    if num_value_heads % num_key_heads != 0:
        raise ValueError(f"GDN value heads={num_value_heads} must be divisible by key heads={num_key_heads}.")
    if cp_size > num_key_heads:
        raise ValueError(
            f"Distributed GDN requires at least one TP-local key head per CP rank, got "
            f"key_heads={num_key_heads} and cp_size={cp_size}."
        )

    heads_per_rank, extra_ranks = divmod(num_key_heads, cp_size)
    key_head_counts = tuple(heads_per_rank + (rank < extra_ranks) for rank in range(cp_size))
    value_heads_per_key = num_value_heads // num_key_heads
    value_head_counts = tuple(count * value_heads_per_key for count in key_head_counts)
    return key_head_counts, value_head_counts


def _resolve_rank_split_sections(
    split_sections: tuple[int, ...],
    cp_size: int,
    rank_split_sections: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[int, ...], ...]:
    """Validate or construct each CP rank's width within every fused section."""
    if rank_split_sections is None:
        if any(section % cp_size != 0 for section in split_sections):
            raise ValueError(
                f"Every GDN projection section must be divisible by cp_size={cp_size}, " f"got {split_sections}."
            )
        return tuple(tuple(section // cp_size for section in split_sections) for _ in range(cp_size))

    if len(rank_split_sections) != cp_size:
        raise ValueError(f"Expected split sections for {cp_size} CP ranks, got {len(rank_split_sections)}.")
    if any(len(rank_sections) != len(split_sections) for rank_sections in rank_split_sections):
        raise ValueError(
            f"Every CP rank must provide {len(split_sections)} fused-section widths, " f"got {rank_split_sections}."
        )
    if any(width < 0 for rank_sections in rank_split_sections for width in rank_sections):
        raise ValueError(f"GDN rank split widths must be non-negative, got {rank_split_sections}.")
    for section_index, section in enumerate(split_sections):
        sharded_size = sum(rank_sections[section_index] for rank_sections in rank_split_sections)
        if sharded_size != section:
            raise ValueError(
                f"GDN section {section_index} has size {section}, but its CP shards sum to {sharded_size}."
            )
    return rank_split_sections


def _resolve_cu_seqlens(
    cu_seqlens_padded: torch.Tensor | None,
    cu_seqlens_actual: torch.Tensor | None,
    total_seq_len: int,
    name: str,
    cp_size: int,
) -> torch.Tensor:
    """Resolve and validate the packed boundaries consumed by GDN."""
    cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual
    if cu_seqlens is None:
        raise ValueError(f"GDN requires {name} for packed THD input.")

    total_cu = int(cu_seqlens[-1].item())
    if total_cu != total_seq_len:
        raise ValueError(f"GDN: {name}[-1]={total_cu} does not match total_sequence_length={total_seq_len}.")

    if cp_size > 1:
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        cp_partition_size = 2 * cp_size
        if bool((seq_lengths % cp_partition_size != 0).any()):
            raise ValueError(
                "All packed sequence lengths must be divisible by "
                f"2*cp_size={cp_partition_size} for zigzag CP, got {seq_lengths.tolist()}."
            )
    return cu_seqlens


def _get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: tuple[int, ...] | list[int] | None = None,
    rank_split_sections: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    """Return this CP rank's possibly uneven slice from each parameter section."""
    cp_size = cp_group.size()
    if cp_size == 1:
        return param

    cp_rank = cp_group.rank()
    dim_size = param.size(dim)
    split_sections = (dim_size,) if split_sections is None else tuple(split_sections)
    if sum(split_sections) != dim_size:
        raise ValueError(f"Parameter split sections {split_sections} do not sum to dimension size {dim_size}.")
    rank_split_sections = _resolve_rank_split_sections(split_sections, cp_size, rank_split_sections)

    local_sections = []
    for section_index, section in enumerate(torch.split(param, split_sections, dim)):
        rank_sizes = tuple(rank_sections[section_index] for rank_sections in rank_split_sections)
        slices = [slice(None)] * param.dim()
        start = sum(rank_sizes[:cp_rank])
        slices[dim] = slice(start, start + rank_sizes[cp_rank])
        local_sections.append(section[tuple(slices)])
    return local_sections[0] if len(local_sections) == 1 else torch.cat(local_sections, dim=dim)


def _build_thd_cp_a2a_perm(
    cu_seqlens: torch.Tensor, cp_size: int, total_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map slime's per-sequence zigzag CP layout to natural packed order."""
    cu = cu_seqlens.to(dtype=torch.long)
    local_seq_len = total_seq_len // cp_size
    positions = torch.arange(total_seq_len, device=cu.device)
    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    seq_lens = torch.diff(cu)
    half_chunks = seq_lens // (2 * cp_size)
    if bool((half_chunks == 0).any()):
        raise ValueError(
            f"Every packed sequence must contain at least 2*cp_size={2 * cp_size} tokens, "
            f"got lengths {seq_lens.tolist()}."
        )
    local_starts = cu[:-1] // cp_size
    global_starts = cu[:-1]

    half_chunk = half_chunks[seq_idx]
    position_in_seq = positions - global_starts[seq_idx]
    natural_chunk = position_in_seq // half_chunk
    offset = position_in_seq - natural_chunk * half_chunk

    load_balanced_chunk = torch.where(
        natural_chunk < cp_size,
        2 * natural_chunk,
        4 * cp_size - 2 * natural_chunk - 1,
    )
    rank = load_balanced_chunk // 2
    half_within_rank = load_balanced_chunk - 2 * rank
    index = rank * local_seq_len + local_starts[seq_idx] + half_within_rank * half_chunk + offset

    inverse = torch.empty_like(index)
    inverse[index] = positions
    return index, inverse


@lru_cache(maxsize=8)
def _build_head_perm_for_split_sections(
    split_sections: tuple[int, ...],
    cp_size: int,
    device: torch.device,
    rank_split_sections: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    """Permute fused sections into rank-major order for equal or ragged A2A."""
    rank_split_sections = _resolve_rank_split_sections(split_sections, cp_size, rank_split_sections)

    section_starts = [0]
    for section in split_sections:
        section_starts.append(section_starts[-1] + section)

    indices = []
    for cp_rank in range(cp_size):
        for section_index, section_start in enumerate(section_starts[:-1]):
            local_section = rank_split_sections[cp_rank][section_index]
            start = section_start + sum(rank_split_sections[rank][section_index] for rank in range(cp_rank))
            indices.append(torch.arange(start, start + local_section, device=device))
    return torch.cat(indices)


def _a2a_cp2hp_ragged(
    tensor: torch.Tensor,
    rank_widths: tuple[int, ...],
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Convert sequence shards to uneven head shards with all-to-all-v."""
    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor
    if tensor.dim() != 3:
        raise ValueError(f"Ragged CP-to-HP all-to-all requires a 3-D tensor, got {tensor.dim()} dimensions.")
    if len(rank_widths) != cp_size or sum(rank_widths) != tensor.size(-1):
        raise ValueError(
            f"Ragged CP-to-HP widths {rank_widths} must contain {cp_size} entries and sum to "
            f"hidden size {tensor.size(-1)}."
        )

    local_seq_len, batch, _ = tensor.shape
    packed = torch.cat([chunk.contiguous().view(-1) for chunk in torch.split(tensor, rank_widths, dim=-1)])
    input_split_sizes = [local_seq_len * batch * width for width in rank_widths]
    local_width = rank_widths[cp_group.rank()]
    output_split_sizes = [local_seq_len * batch * local_width] * cp_size
    exchanged = all_to_all(cp_group, packed, output_split_sizes, input_split_sizes)
    return exchanged.view(local_seq_len * cp_size, batch, local_width)


def _a2a_hp2cp_ragged(
    tensor: torch.Tensor,
    rank_widths: tuple[int, ...],
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Restore sequence shards from uneven head shards with all-to-all-v."""
    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor
    if tensor.dim() != 3 or tensor.size(0) % cp_size != 0:
        raise ValueError(
            f"Ragged HP-to-CP all-to-all requires a 3-D tensor with sequence length divisible by "
            f"cp_size={cp_size}, got shape {tuple(tensor.shape)}."
        )
    if len(rank_widths) != cp_size:
        raise ValueError(f"Expected {cp_size} ragged head widths, got {rank_widths}.")

    total_seq_len, batch, local_width = tensor.shape
    cp_rank = cp_group.rank()
    if local_width != rank_widths[cp_rank]:
        raise ValueError(f"CP rank {cp_rank} owns head width {rank_widths[cp_rank]}, got tensor width {local_width}.")

    local_seq_len = total_seq_len // cp_size
    packed = torch.cat([chunk.contiguous().view(-1) for chunk in torch.chunk(tensor, cp_size, dim=0)])
    input_split_sizes = [local_seq_len * batch * local_width] * cp_size
    output_split_sizes = [local_seq_len * batch * width for width in rank_widths]
    exchanged = all_to_all(cp_group, packed, output_split_sizes, input_split_sizes)
    received = torch.split(exchanged, output_split_sizes)
    return torch.cat(
        [chunk.view(local_seq_len, batch, width) for chunk, width in zip(received, rank_widths, strict=True)],
        dim=-1,
    )


def _reorder_nonpacked_zigzag(tensor: torch.Tensor, cp_size: int, *, undo: bool) -> torch.Tensor:
    """Convert between rank-major zigzag and natural sequence order."""
    num_chunks = 2 * cp_size
    if tensor.size(0) % num_chunks != 0:
        raise ValueError(f"Zigzag GDN sequence length {tensor.size(0)} must be divisible by 2*cp_size={num_chunks}.")
    chunks = torch.chunk(tensor, num_chunks, dim=0)
    if undo:
        order = [2 * rank for rank in range(cp_size)] + [num_chunks - 2 * rank - 1 for rank in range(cp_size)]
    else:
        order = [0] * num_chunks
        order[::2] = range(cp_size)
        order[1::2] = reversed(range(cp_size, num_chunks))
    return torch.cat([chunks[index] for index in order], dim=0)


def a2a_cp_to_hp_packed(
    projected: torch.Tensor,
    split_sections: tuple[int, ...],
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    cu_seqlens: torch.Tensor | None,
    total_seq_len: int,
    packed_seq_params: PackedSeqParams | None,
    rank_split_sections: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Convert CP sequence shards to head shards, including packed THD ordering."""
    rank_widths = None
    if cp_size > 1:
        head_perm = _build_head_perm_for_split_sections(split_sections, cp_size, projected.device, rank_split_sections)
        projected = projected.index_select(-1, head_perm)
        if rank_split_sections is not None:
            rank_widths = tuple(sum(rank_sections) for rank_sections in rank_split_sections)

    inverse = None
    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        if rank_widths is None:
            projected = tensor_a2a_cp2hp(
                projected,
                seq_dim=0,
                head_dim=-1,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
        else:
            projected = _a2a_cp2hp_ragged(projected, rank_widths, cp_group)
        if cp_size > 1:
            if cu_seqlens is None:
                raise ValueError("Packed THD GDN requires cu_seqlens.")
            index, inverse = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, total_seq_len)
            projected = projected.index_select(0, index)
    else:
        if rank_widths is None:
            projected = tensor_a2a_cp2hp(
                projected,
                seq_dim=0,
                head_dim=-1,
                cp_group=cp_group,
            )
        else:
            projected = _a2a_cp2hp_ragged(projected, rank_widths, cp_group)
            projected = _reorder_nonpacked_zigzag(projected, cp_size, undo=True)
    return projected, inverse


def a2a_hp_to_cp_packed(
    output: torch.Tensor,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    packed_seq_params: PackedSeqParams | None,
    inverse: torch.Tensor | None,
    rank_widths: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Restore the context-parallel layout after head-sharded GDN compute."""
    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        if cp_size > 1:
            if inverse is None:
                raise ValueError("Packed THD GDN requires the inverse CP permutation.")
            output = output.index_select(0, inverse)
        if rank_widths is not None:
            return _a2a_hp2cp_ragged(output, rank_widths, cp_group)
        return tensor_a2a_hp2cp(
            output,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
            redo_attention_load_balancing=False,
        )
    if rank_widths is not None:
        output = _reorder_nonpacked_zigzag(output, cp_size, undo=False)
        return _a2a_hp2cp_ragged(output, rank_widths, cp_group)
    return tensor_a2a_hp2cp(output, seq_dim=0, head_dim=-1, cp_group=cp_group)


class DistributedQwenGatedDeltaNet(GatedDeltaNet):
    """Native TP/CP-sharded GDN with packed THD and FlashQLA support."""

    def __init__(self, *module_args, args=None, **module_kwargs):
        super().__init__(*module_args, **module_kwargs)
        backend = getattr(args, "qwen_gdn_backend", "fla")
        if not self.config.deterministic_mode:
            self.gated_delta_rule = get_chunk_gated_delta_rule(backend)
        self.gdn_backend = backend
        self.recompute_norm_out = bool(
            getattr(self, "recompute_norm_out", False) or getattr(args, "qwen_gdn_recompute_norm_out", False)
        )
        self.norm_out_checkpoint = getattr(self, "norm_out_checkpoint", None)

        self.in_proj_split_sections = (
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
            self.v_dim_local_tp,
            self.num_value_heads // self.tp_size,
            self.num_value_heads // self.tp_size,
        )
        num_key_heads_local_tp = self.num_key_heads // self.tp_size
        num_value_heads_local_tp = self.num_value_heads // self.tp_size
        self.key_head_counts, self.value_head_counts = _build_gdn_head_shards(
            num_key_heads_local_tp,
            num_value_heads_local_tp,
            self.cp_size,
        )
        cp_rank = self.pg_collection.cp.rank()
        self.num_key_heads_local_cp = self.key_head_counts[cp_rank]
        self.num_value_heads_local_cp = self.value_head_counts[cp_rank]
        self.ragged_cp = num_key_heads_local_tp % self.cp_size != 0
        if self.ragged_cp:
            self.in_proj_rank_split_sections = tuple(
                (
                    key_heads * self.key_head_dim,
                    key_heads * self.key_head_dim,
                    value_heads * self.value_head_dim,
                    value_heads * self.value_head_dim,
                    value_heads,
                    value_heads,
                )
                for key_heads, value_heads in zip(self.key_head_counts, self.value_head_counts, strict=True)
            )
            self.qkv_rank_split_sections = tuple(
                (
                    key_heads * self.key_head_dim,
                    key_heads * self.key_head_dim,
                    value_heads * self.value_head_dim,
                )
                for key_heads, value_heads in zip(self.key_head_counts, self.value_head_counts, strict=True)
            )
            self.value_parameter_rank_split_sections = tuple((heads,) for heads in self.value_head_counts)
            self.output_rank_widths = tuple(heads * self.value_head_dim for heads in self.value_head_counts)
        else:
            self.in_proj_rank_split_sections = None
            self.qkv_rank_split_sections = None
            self.value_parameter_rank_split_sections = None
            self.output_rank_widths = None
        self.feat_dim_split = (
            self.num_key_heads_local_cp * self.key_head_dim * 2 + self.num_value_heads_local_cp * self.value_head_dim,
            self.num_value_heads_local_cp * self.value_head_dim,
            self.num_value_heads_local_cp,
            self.num_value_heads_local_cp,
        )

        # The pinned MCore marks these parameters as TP-sharded but omits the
        # metadata needed by slime's online all-gather weight synchronizer.
        for param in (self.conv1d.weight, self.dt_bias, self.A_log):
            param.tensor_model_parallel = True
            param.partition_dim = 0
            param.partition_stride = 1
        if self.conv1d.bias is not None:
            self.conv1d.bias.tensor_model_parallel = True
            self.conv1d.bias.partition_dim = 0
            self.conv1d.bias.partition_stride = 1

    def _gated_norm_and_a2a(
        self,
        core_output: torch.Tensor,
        gate: torch.Tensor,
        inverse: torch.Tensor | None,
        batch: int,
        total_seq_len: int,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        nvtx_range_push(suffix="gated_norm")
        norm_output = self._apply_gated_norm(core_output, gate)
        nvtx_range_pop(suffix="gated_norm")
        norm_output = norm_output.reshape(batch, total_seq_len, -1).transpose(0, 1).contiguous()
        return a2a_hp_to_cp_packed(
            norm_output,
            self.cp_size,
            self.pg_collection.cp,
            packed_seq_params,
            inverse,
            self.output_rank_widths,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del attention_mask, sequence_len_offset, kwargs
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context is not None:
            raise NotImplementedError("Distributed Qwen GDN is a training-only path.")

        local_seq_len, batch, _ = hidden_states.shape
        total_seq_len = local_seq_len * self.sp_size * self.cp_size

        if packed_seq_params is not None:
            if self.config.deterministic_mode:
                raise NotImplementedError("Packed THD GDN does not support deterministic mode.")
            if packed_seq_params.qkv_format != "thd":
                raise ValueError(
                    f"Distributed packed GDN supports qkv_format='thd', got {packed_seq_params.qkv_format!r}."
                )
            if batch != 1:
                raise ValueError(f"Packed THD GDN requires batch=1, got {batch}.")
            cu_seqlens_q = _resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_q_padded,
                packed_seq_params.cu_seqlens_q,
                total_seq_len,
                "cu_seqlens_q",
                self.cp_size,
            )
            cu_seqlens_kv = _resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_kv_padded,
                packed_seq_params.cu_seqlens_kv,
                total_seq_len,
                "cu_seqlens_kv",
                self.cp_size,
            )
            if not torch.equal(cu_seqlens_q, cu_seqlens_kv):
                raise ValueError("GDN currently requires identical Q and KV packed boundaries.")
        else:
            cu_seqlens_q = None

        nvtx_range_push(suffix="in_proj")
        projected, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        projected, inverse = a2a_cp_to_hp_packed(
            projected,
            self.in_proj_split_sections,
            self.cp_size,
            self.pg_collection.cp,
            cu_seqlens_q,
            total_seq_len,
            packed_seq_params,
            self.in_proj_rank_split_sections,
        )
        projected = projected.transpose(0, 1)
        qkv, gate, beta, alpha = torch.split(projected, self.feat_dim_split, dim=-1)
        gate = gate.reshape(batch, total_seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, total_seq_len, -1)
        alpha = alpha.reshape(batch, total_seq_len, -1)

        nvtx_range_push(suffix="conv1d")
        qkv_sections = [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp]
        conv_weight = _get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_sections,
            rank_split_sections=self.qkv_rank_split_sections,
        )
        conv_bias = (
            _get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_sections,
                rank_split_sections=self.qkv_rank_split_sections,
            )
            if self.conv1d.bias is not None
            else None
        )
        if self.config.deterministic_mode:
            qkv_t = qkv.transpose(1, 2).contiguous()
            conv_out = F.conv1d(
                qkv_t,
                conv_weight,
                bias=conv_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=conv_weight.size(0),
            )
            qkv = self.act_fn(conv_out[..., :total_seq_len]).transpose(1, 2)
        else:
            if self.activation not in ("silu", "swish"):
                raise ValueError(f"GDN causal convolution requires silu/swish, got {self.activation!r}.")
            qkv, _ = causal_conv1d(
                x=qkv,
                weight=conv_weight.squeeze(1),
                bias=conv_bias,
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens_q,
            )
        nvtx_range_pop(suffix="conv1d")

        query_key, value = torch.split(
            qkv,
            [
                2 * self.num_key_heads_local_cp * self.key_head_dim,
                self.num_value_heads_local_cp * self.value_head_dim,
            ],
            dim=-1,
        )
        query_key = query_key.reshape(batch, total_seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, total_seq_len, -1, self.value_head_dim)
        query, key = torch.chunk(query_key, 2, dim=2)
        repeat_factor = self.num_value_heads // self.num_key_heads
        if repeat_factor > 1:
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        value_parameter_sections = (self.num_value_heads // self.tp_size,)
        A_log = _get_parameter_local_cp(
            self.A_log,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=value_parameter_sections,
            rank_split_sections=self.value_parameter_rank_split_sections,
        )
        dt_bias = _get_parameter_local_cp(
            self.dt_bias,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=value_parameter_sections,
            rank_split_sections=self.value_parameter_rank_split_sections,
        )
        g = -A_log.float().exp() * F.softplus(alpha.float() + dt_bias)
        beta = beta.sigmoid()

        nvtx_range_push(suffix="gated_delta_rule")
        core_output, _ = self.gated_delta_rule(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            initial_state=None,
            output_final_state=False,
            # Preserve slime's existing Qwen numerical path.  Both FLA and
            # FlashQLA accept this interface.
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            cu_seqlens=cu_seqlens_q,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        gate = gate.contiguous()
        if self.recompute_norm_out:
            self.norm_out_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            norm_func = partial(
                self._gated_norm_and_a2a,
                inverse=inverse,
                batch=batch,
                total_seq_len=total_seq_len,
                packed_seq_params=packed_seq_params,
            )
            norm_output = self.norm_out_checkpoint.checkpoint(norm_func, core_output, gate)
        else:
            norm_output = self._gated_norm_and_a2a(
                core_output,
                gate,
                inverse,
                batch,
                total_seq_len,
                packed_seq_params,
            )

        nvtx_range_push(suffix="out_proj")
        output, output_bias = self.out_proj(norm_output)
        nvtx_range_pop(suffix="out_proj")
        if self.recompute_norm_out:
            self.norm_out_checkpoint.discard_output_and_register_recompute(output)
        return output, output_bias
