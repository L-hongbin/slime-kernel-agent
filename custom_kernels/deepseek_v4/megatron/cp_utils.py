"""Context-parallel (CP2) orchestration for V4-Flash training.

The CP2 contract lives in ``handoffs/deepseek-v4/dsv4_megatron_sharding_contract.md``:
this module provides the torch-level
comm/orchestration around the already-CP-capable A1 attention kernel (which threads
``q_pos0`` / ``raw_halo`` -- see ``attention/kernel.py``).  bshd-contiguous CP2: each
rank owns a CONTIGUOUS slice ``[r*l_local, (r+1)*l_local)`` of the sequence
(``cp_partition_mode`` = contiguous, NOT the slime zigzag), and the receptive field
is made fully-local by a single LEFT halo exchange + a compressed all-gather, so no
cross-rank softmax merge is needed.

Three primitives, all no-ops at ``cp_size == 1`` (the default path constructs none of
them, so non-CP behavior is bit-identical):

  * ``CpHaloExchange`` (design M5): ONE shared autograd op.  Forward: rank ``r>0``
    receives the previous rank's last ``halo`` hidden rows (P2P) and returns
    ``[halo || local]``; rank 0 returns ``local`` unchanged.  Backward: the grad of the
    received halo rows is sent BACK to the owner and ADDED into its local grad
    (``_LeftBoundaryExchange``).  A single instance per layer feeds BOTH the raw-KV
    projection and the compressor look-back -- two exchanges would drop one consumer's
    boundary gradient.

  * ``CpCompAllgather`` (design backward comm (ii)): all-gather each rank's OWNED
    compressed windows into the global comp axis in rank order; backward
    reduce-scatters the global ``dk_comp`` (summed over every rank's local attention
    contribution) to the owning ranks.  all-gather fwd <-> reduce-scatter bwd is the
    exact adjoint pair.

  * ``compressor_drop_windows`` (design B2): a rank compressing ``[halo || local]``
    drops the first ``halo // compress_rate`` windows as non-owned duplicates
    (CSA m=4 -> 32, HCA m=128 -> 1); the owner of a window is the rank of its LAST
    token, so with a 128-aligned ``l_local`` each rank emits exactly ``l_local / m``
    windows.

The primitives take an explicit :class:`CpInfo` (group / rank / size / global ranks) so
they are testable with a plain gloo process group; :func:`get_cp_info` is the thin
Megatron ``parallel_state`` adapter the model code uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

# Left-halo width (hidden rows exchanged between neighbours).  127 covers the raw
# sliding window; 128 keeps block-alignment for the compressor (HCA m=128 -> exactly
# one halo window; CSA m=4 -> 32) and for the kernel's raw axis (raw_halo % block_N).
CP_HALO = 128


@dataclass(frozen=True)
class CpInfo:
    """Context-parallel group handle.  ``size == 1`` marks the disabled (no-op) path."""

    group: object = None  # torch.distributed ProcessGroup or None
    rank: int = 0
    size: int = 1
    global_ranks: tuple[int, ...] | None = None  # cp-group ranks in the GLOBAL world

    @property
    def enabled(self) -> bool:
        return self.size > 1

    def left_global_rank(self) -> int | None:
        return self.global_ranks[self.rank - 1] if self.rank > 0 else None

    def right_global_rank(self) -> int | None:
        return self.global_ranks[self.rank + 1] if self.rank < self.size - 1 else None

    def local_halo(self, halo: int = CP_HALO) -> int:
        """Halo rows actually PREPENDED to this rank's tensors (0 on rank 0)."""
        return halo if (self.enabled and self.rank > 0) else 0

    def global_start(self, l_local: int) -> int:
        """Global position of this rank's first LOCAL token (contiguous partition)."""
        return self.rank * l_local if self.enabled else 0


def get_cp_info() -> CpInfo:
    """Build a :class:`CpInfo` from Megatron ``parallel_state``.

    Returns a disabled (``size == 1``) info whenever CP is off, torch.distributed is
    not initialized, or Megatron is unavailable -- so every call site degrades to the
    bit-identical non-CP path without a distributed dependency.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return CpInfo()
    try:
        from megatron.core import parallel_state
    except Exception:  # noqa: BLE001 -- standalone / non-Megatron harnesses
        return CpInfo()
    try:
        size = parallel_state.get_context_parallel_world_size()
    except Exception:  # noqa: BLE001 -- parallel_state not initialized
        return CpInfo()
    if size <= 1:
        return CpInfo()
    return CpInfo(
        group=parallel_state.get_context_parallel_group(),
        rank=parallel_state.get_context_parallel_rank(),
        size=size,
        global_ranks=tuple(parallel_state.get_context_parallel_global_ranks()),
    )


def assert_local_len_aligned(l_local: int, cp_info: CpInfo) -> None:
    """B1 guard (consumer side): under CP, ``l_local`` must be a multiple of 128.

    The slime data path raises the pad granularity to ``cp_size * 128`` so this holds;
    the assert catches a mis-configured launch before it silently corrupts HCA's
    zero-halo window alignment (l_local not 128-aligned -> non-integer window counts).
    """
    if cp_info.enabled:
        assert l_local % 128 == 0, (
            f"CP local sequence length must be a multiple of 128 (pad granularity "
            f"cp_size*128); got l_local={l_local} with cp_size={cp_info.size}"
        )


def compressor_drop_windows(local_halo: int, compress_rate: int) -> int:
    """Leading compressed windows to drop after compressing ``[halo || local]`` (B2).

    These windows lie fully inside the halo and are owned by the left neighbour.  0 on
    rank 0 (no halo).  CSA m=4 -> 128//4 = 32; HCA m=128 -> 128//128 = 1.
    """
    if local_halo == 0:
        return 0
    assert local_halo % compress_rate == 0, f"halo {local_halo} must be a multiple of compress_rate {compress_rate}"
    return local_halo // compress_rate


def _batch_p2p(ops: list) -> None:
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()


class CpHaloExchange(torch.autograd.Function):
    """Left-halo hidden exchange (design M5).  ONE op feeding kv_proj + compressor.

    Forward returns ``[halo || local]`` for rank ``r>0`` (halo = previous rank's last
    ``halo`` rows) and ``local`` unchanged for rank 0.  Backward sends the received
    halo rows' gradient back to the owner and adds it into the owner's local grad at
    ``[-halo:]``.
    """

    @staticmethod
    def forward(ctx, hidden, halo, group, rank, size, global_ranks):
        # hidden: [B, l_local, H] (this rank's contiguous shard, post input_layernorm).
        ctx.halo = halo
        ctx.group = group
        ctx.rank = rank
        ctx.size = size
        ctx.global_ranks = global_ranks
        if size <= 1 or halo == 0:
            return hidden
        B, _, H = hidden.shape
        left = global_ranks[rank - 1] if rank > 0 else None
        right = global_ranks[rank + 1] if rank < size - 1 else None
        recv_buf = hidden.new_empty(B, halo, H) if rank > 0 else None
        send_buf = hidden[:, -halo:, :].contiguous() if rank < size - 1 else None
        ops = []
        if recv_buf is not None:
            ops.append(dist.P2POp(dist.irecv, recv_buf, left, group))
        if send_buf is not None:
            ops.append(dist.P2POp(dist.isend, send_buf, right, group))
        _batch_p2p(ops)
        if rank > 0:
            return torch.cat([recv_buf, hidden], dim=1)  # [B, halo+l_local, H]
        return hidden

    @staticmethod
    def backward(ctx, grad_out):
        halo, group, rank, size, global_ranks = (
            ctx.halo,
            ctx.group,
            ctx.rank,
            ctx.size,
            ctx.global_ranks,
        )
        if size <= 1 or halo == 0:
            return grad_out, None, None, None, None, None
        B, _, H = grad_out.shape
        left = global_ranks[rank - 1] if rank > 0 else None
        right = global_ranks[rank + 1] if rank < size - 1 else None
        # grad of the received halo rows (r>0) is sent back to the owner (left neighbour).
        halo_grad = grad_out[:, :halo, :].contiguous() if rank > 0 else None
        grad_local = grad_out[:, halo:, :] if rank > 0 else grad_out
        recv_buf = grad_out.new_empty(B, halo, H) if rank < size - 1 else None
        ops = []
        if halo_grad is not None:
            ops.append(dist.P2POp(dist.isend, halo_grad, left, group))
        if recv_buf is not None:
            ops.append(dist.P2POp(dist.irecv, recv_buf, right, group))
        _batch_p2p(ops)
        if recv_buf is not None:
            # Clone before the in-place add so the incoming grad tensor is never mutated
            # (a size-1 batch slice can alias grad_out).
            grad_local = grad_local.clone()
            grad_local[:, -halo:, :] += recv_buf
        else:
            grad_local = grad_local.contiguous()
        return grad_local, None, None, None, None, None


def cp_halo_exchange(hidden: torch.Tensor, cp_info: CpInfo, halo: int = CP_HALO) -> torch.Tensor:
    """Prepend the left neighbour's ``halo`` hidden rows (rank>0); no-op at cp_size==1.

    Output is ``[B, l_local, H]`` on rank 0 / non-CP and ``[B, halo+l_local, H]`` on
    rank ``r>0``.  Feed the result to BOTH kv_proj and the compressor.
    """
    if not cp_info.enabled:
        return hidden
    return CpHaloExchange.apply(hidden, halo, cp_info.group, cp_info.rank, cp_info.size, cp_info.global_ranks)


class CpCompAllgather(torch.autograd.Function):
    """All-gather owned compressed windows -> global comp axis (rank order).

    Forward concatenates every rank's ``[B, 1, T_local, D]`` in rank order into
    ``[B, 1, cp_size*T_local, D]``.  Backward reduce-scatters the global ``dk_comp``:
    every rank's local attention produced a full-axis ``dk_comp`` contribution, so the
    true grad for a window is the SUM over ranks, delivered to its owner.
    """

    @staticmethod
    def forward(ctx, k_comp_local, group, rank, size):
        ctx.group = group
        ctx.rank = rank
        ctx.size = size
        if size <= 1:
            return k_comp_local
        k_comp_local = k_comp_local.contiguous()
        gathered = [torch.empty_like(k_comp_local) for _ in range(size)]
        dist.all_gather(gathered, k_comp_local, group=group)
        return torch.cat(gathered, dim=2)  # [B, 1, T_global, D]

    @staticmethod
    def backward(ctx, grad_global):
        group, rank, size = ctx.group, ctx.rank, ctx.size
        if size <= 1:
            return grad_global, None, None, None
        chunks = [c.contiguous() for c in grad_global.contiguous().chunk(size, dim=2)]
        out = torch.empty_like(chunks[rank])
        # out[rank] = sum over all ranks p of chunks_on_p[rank] -> owner gets summed grad.
        dist.reduce_scatter(out, chunks, group=group)
        return out, None, None, None


def cp_allgather_compressed(k_comp_local: torch.Tensor, cp_info: CpInfo) -> torch.Tensor:
    """Gather owned compressed KV into the global comp axis (rank order); no-op at cp1."""
    if not cp_info.enabled:
        return k_comp_local
    return CpCompAllgather.apply(k_comp_local, cp_info.group, cp_info.rank, cp_info.size)


__all__ = [
    "CP_HALO",
    "CpInfo",
    "get_cp_info",
    "assert_local_len_aligned",
    "compressor_drop_windows",
    "CpHaloExchange",
    "cp_halo_exchange",
    "CpCompAllgather",
    "cp_allgather_compressed",
]
