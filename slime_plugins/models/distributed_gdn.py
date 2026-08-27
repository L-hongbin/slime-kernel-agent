"""Packed-THD distributed GDN for the Megatron version pinned by slime.

The pinned Megatron already shards GDN projections across tensor parallel ranks
and converts context-parallel sequence shards into head shards before running
the recurrent kernel.  Its missing piece is packed-sequence support.  This
module backports the packed THD permutation from newer Megatron releases while
keeping the recurrent kernel selectable between FLA and FlashQLA.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, causal_conv1d, tensor_a2a_cp2hp, tensor_a2a_hp2cp
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

from .qwen_gdn_backend import get_chunk_gated_delta_rule


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
) -> torch.Tensor:
    """Return this CP rank's parameter slice without deprecated list indexing."""
    cp_size = cp_group.size()
    if cp_size == 1:
        return param

    if split_sections is not None:
        return torch.cat(
            [_get_parameter_local_cp(section, dim, cp_group) for section in torch.split(param, split_sections, dim)],
            dim=dim,
        )

    cp_rank = cp_group.rank()
    dim_size = param.size(dim)
    if dim_size % cp_size != 0:
        raise ValueError(f"Parameter dimension {dim_size} must be divisible by cp_size={cp_size}.")
    slices = [slice(None)] * param.dim()
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    return param[tuple(slices)]


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
    split_sections: tuple[int, ...], cp_size: int, device: torch.device
) -> torch.Tensor:
    """Permute fused projection sections so one A2A shards every section identically."""
    if any(section % cp_size != 0 for section in split_sections):
        raise ValueError(
            f"Every GDN projection section must be divisible by cp_size={cp_size}, " f"got {split_sections}."
        )

    section_starts = [0]
    for section in split_sections:
        section_starts.append(section_starts[-1] + section)

    indices = []
    for cp_rank in range(cp_size):
        for section_start, section in zip(section_starts[:-1], split_sections, strict=True):
            local_section = section // cp_size
            start = section_start + cp_rank * local_section
            indices.append(torch.arange(start, start + local_section, device=device))
    return torch.cat(indices)


def a2a_cp_to_hp_packed(
    projected: torch.Tensor,
    split_sections: tuple[int, ...],
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    cu_seqlens: torch.Tensor | None,
    total_seq_len: int,
    packed_seq_params: PackedSeqParams | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Convert CP sequence shards to head shards, including packed THD ordering."""
    if cp_size > 1:
        head_perm = _build_head_perm_for_split_sections(split_sections, cp_size, projected.device)
        projected = projected.index_select(-1, head_perm)

    inverse = None
    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        projected = tensor_a2a_cp2hp(
            projected,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
            undo_attention_load_balancing=False,
        )
        if cp_size > 1:
            if cu_seqlens is None:
                raise ValueError("Packed THD GDN requires cu_seqlens.")
            index, inverse = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, total_seq_len)
            projected = projected.index_select(0, index)
    else:
        projected = tensor_a2a_cp2hp(
            projected,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
        )
    return projected, inverse


def a2a_hp_to_cp_packed(
    output: torch.Tensor,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    packed_seq_params: PackedSeqParams | None,
    inverse: torch.Tensor | None,
) -> torch.Tensor:
    """Restore the context-parallel layout after head-sharded GDN compute."""
    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        if cp_size > 1:
            if inverse is None:
                raise ValueError("Packed THD GDN requires the inverse CP permutation.")
            output = output.index_select(0, inverse)
        return tensor_a2a_hp2cp(
            output,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
            redo_attention_load_balancing=False,
        )
    return tensor_a2a_hp2cp(output, seq_dim=0, head_dim=-1, cp_group=cp_group)


class DistributedQwenGatedDeltaNet(GatedDeltaNet):
    """Native TP/CP-sharded GDN with packed THD and FlashQLA support."""

    def __init__(self, *module_args, args=None, **module_kwargs):
        super().__init__(*module_args, **module_kwargs)
        backend = getattr(args, "qwen_gdn_backend", "fla")
        if not self.config.deterministic_mode:
            self.gated_delta_rule = get_chunk_gated_delta_rule(backend)
        self.gdn_backend = backend

        self.in_proj_split_sections = (
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
            self.v_dim_local_tp,
            self.num_value_heads // self.tp_size,
            self.num_value_heads // self.tp_size,
        )
        self.feat_dim_split = (
            (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,
            self.v_dim_local_tp // self.cp_size,
            self.num_value_heads // self.tp_size // self.cp_size,
            self.num_value_heads // self.tp_size // self.cp_size,
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
        )
        conv_bias = (
            _get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_sections,
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
                groups=self.conv_dim_local_tp // self.cp_size,
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
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, total_seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, total_seq_len, -1, self.value_head_dim)
        query, key = torch.chunk(query_key, 2, dim=2)
        repeat_factor = self.num_value_heads // self.num_key_heads
        if repeat_factor > 1:
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        A_log = _get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias = _get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.pg_collection.cp)
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

        nvtx_range_push(suffix="gated_norm")
        norm_output = self._apply_gated_norm(core_output, gate.contiguous())
        nvtx_range_pop(suffix="gated_norm")
        norm_output = norm_output.reshape(batch, total_seq_len, -1).transpose(0, 1).contiguous()
        norm_output = a2a_hp_to_cp_packed(
            norm_output,
            self.cp_size,
            self.pg_collection.cp,
            packed_seq_params,
            inverse,
        )

        nvtx_range_push(suffix="out_proj")
        output, output_bias = self.out_proj(norm_output)
        nvtx_range_pop(suffix="out_proj")
        return output, output_bias
