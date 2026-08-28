"""DeepSeek-V4-Flash Multi-Token-Prediction (MTP) head for the mcore model.

The V4-Flash checkpoint carries a single MTP module under ``mtp.0.*`` (1575
tensors).  It predicts the *next-next* token: at depth 1 it combines the depth-0
hidden state at position ``i`` with the embedding of token ``i+1`` and runs ONE
V4 decoder layer + the shared output head to produce logits for token ``i+2``.

Structure (verified against the checkpoint key dump + sglang's NextN reference at
``/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_nextn.py``):

  * ``enorm`` / ``hnorm``     : RMSNorms on the embedding / previous hc stream.
  * ``e_proj`` / ``h_proj``   : SPLIT projections.  V4 does NOT use V3's single
    ``eh_proj`` on a concat; instead it fuses
        ``fused = e_proj(enorm(embed))[:, :, None, :] + h_proj(hnorm(prev_stream))``
    keeping the ``[B, S, hc_mult, H]`` parallel-stream (mHC) structure that the V4
    decoder layer consumes (sglang ``DeepseekV4ModelNextN.forward``).
  * ``transformer_layer``     : one V4 decoder layer whose attention has NO
    compressor/indexer (sglang builds the NextN attention with
    ``compress_ratio=0`` -> main RoPE + sliding window, no compressor) and whose
    FFN is a learned top-k MoE (the ``mtp.0.ffn.gate`` has a correction ``bias``
    and no ``tid2eid``).  Its MoE routes LIVE — it is deliberately excluded from
    the rollout routing replay (``enable_routing_replay=False``).
  * ``hc_head``               : the MTP's OWN head-side hc collapse
    (``DeepseekV4HyperHead``); note ``hc_head_*`` is a NEW tensor group vs V3.
  * ``norm``                  : the MTP final RMSNorm (sglang ``shared_head.norm``).

Semantics mirror Megatron's ``MultiTokenPredictionLayer`` (roll ids by 1, embed
via the SHARED embedding, norm+project+fuse, one transformer layer, final norm,
then the SHARED output head computes the MTP logits).  The MTP module owns
everything EXCEPT the embedding and the output head, which are the host
``V4LanguageModel``'s (shared, on the last PP stage).

This module is gated: ``V4LanguageModel`` only builds it when MTP is enabled
(``--mtp-num-layers 1`` in training; ``--include-mtp`` in the slicer), so default
behavior is unchanged.
"""

from __future__ import annotations


import torch
import torch.distributed
import torch.utils.checkpoint  # noqa: F401  (used by the validated full-recompute path)
from megatron.core.transformer.multi_token_prediction import roll_tensor
from torch import nn

from slime.backends.megatron_utils.cp_utils import CP_PARTITION_CONTIGUOUS, get_cp_partition_mode

from .mcore_model import V4DecoderLayer
from .rope import V4RMSNorm


def roll_tensor_contiguous_cp(tensor, shifts: int = -1, dims: int = -1, cp_group=None):
    """Left-shift-by-one roll over the GLOBAL sequence for the CONTIGUOUS CP layout (M4).

    Under contiguous CP each rank owns a single contiguous block
    ``[r*l_local, (r+1)*l_local)``. The MTP label shift ``result[g] = input[g+1]``
    (last global position -> 0) is therefore a plain local ``torch.roll`` plus ONE
    boundary exchange: my last local position needs my RIGHT neighbour's first
    element; my first element feeds my LEFT neighbour's last position; the last
    rank shifts in 0. This is strictly simpler than megatron's zigzag
    ``roll_tensor`` (which ``chunk(2)``s the local tensor and does a calibrated
    2-boundary exchange) -- applying the zigzag version to a contiguous block would
    inject a spurious mid-block shift boundary. Returns ``(rolled, rolled.sum())``
    to match ``roll_tensor``'s contract.
    """
    assert shifts == -1, "contiguous-CP roll only implements the MTP left shift (shifts=-1)"

    if cp_group is None or cp_group.size() == 1:
        rolled = torch.roll(tensor, shifts=shifts, dims=dims)
        rolled.select(dims, shifts).fill_(0)
        return rolled, rolled.sum()

    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    local_rank = torch.distributed.get_rank(group=cp_group)
    world = len(global_ranks)
    prev_rank = global_ranks[local_rank - 1] if local_rank > 0 else None
    next_rank = global_ranks[local_rank + 1] if local_rank < world - 1 else None

    # The ORIGINAL first element this rank hands to its LEFT neighbour's tail.
    send_elem = tensor.select(dims, 0).contiguous()
    recv_buf = torch.empty_like(send_elem)

    rolled = torch.roll(tensor, shifts=shifts, dims=dims)

    ops = []
    if prev_rank is not None:
        ops.append(torch.distributed.isend(send_elem, dst=prev_rank))
    if next_rank is not None:
        ops.append(torch.distributed.irecv(recv_buf, src=next_rank))
    for op in ops:
        op.wait()

    index = [slice(None)] * rolled.dim()
    index[dims] = shifts  # the last position along the sequence axis
    if next_rank is not None:
        rolled[tuple(index)] = recv_buf
    else:
        rolled[tuple(index)] = 0

    return rolled, rolled.sum()


def _roll_tensor_cp_aware(tensor, shifts: int = -1, dims: int = -1, cp_group=None, packed_seq_params=None):
    """Dispatch the MTP roll to the layout-correct implementation.

    - CP disabled (``cp_group`` None or size 1): stock ``roll_tensor`` == plain torch.roll.
    - Contiguous CP (V4-Flash, bshd/no packed seq): :func:`roll_tensor_contiguous_cp`.
    - Otherwise (zigzag CP or packed sequences): stock ``roll_tensor``.
    """
    cp_enabled = cp_group is not None and cp_group.size() > 1
    if cp_enabled and get_cp_partition_mode() == CP_PARTITION_CONTIGUOUS:
        assert packed_seq_params is None, (
            "contiguous CP MTP roll does not support packed_seq_params "
            "(V4-Flash trains bshd, one right-padded doc per row)"
        )
        return roll_tensor_contiguous_cp(tensor, shifts=shifts, dims=dims, cp_group=cp_group)
    return roll_tensor(tensor, shifts=shifts, dims=dims, cp_group=cp_group, packed_seq_params=packed_seq_params)


class V4MultiTokenPredictionLayer(nn.Module):
    """A single V4 MTP head (depth-1).

    Composed from the existing V4 pieces: a no-compressor ``V4DecoderLayer`` plus
    the split enorm/e_proj + hnorm/h_proj fusion and the head-side ``hc_head``
    collapse.  The shared embedding and output head are supplied by the host model
    at call time (they are NOT owned here — they are tied to the main model).
    """

    def __init__(
        self,
        hf_config,
        *,
        expert_model_parallel_size: int = 1,
        expert_model_parallel_rank: int = 0,
        mcore_config=None,
        pg_collection=None,
    ):
        super().__init__()
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperHead

        self.hf_config = hf_config
        self.mcore_config = mcore_config
        self.hc_mult = hf_config.hc_mult
        self.hidden_size = hf_config.hidden_size

        self.enorm = V4RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.hnorm = V4RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        # Split projections are frozen and materialized in the trainer dtype.
        self.e_proj = nn.Linear(hf_config.hidden_size, hf_config.hidden_size, bias=False)
        self.h_proj = nn.Linear(hf_config.hidden_size, hf_config.hidden_size, bias=False)

        # One V4 decoder layer: no-compressor attention (sliding_attention path) +
        # learned top-k MoE that routes LIVE (excluded from rollout routing replay).
        self.transformer_layer = V4DecoderLayer(
            hf_config,
            0,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_model_parallel_rank=expert_model_parallel_rank,
            mcore_config=mcore_config,
            pg_collection=pg_collection,
            attn_layer_type="sliding_attention",
            mlp_type="moe",
            enable_routing_replay=False,
        )

        self.hc_head = DeepseekV4HyperHead(hf_config)
        self.norm = V4RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)

        # cp_group=None (CP=1) rolls as a plain torch.roll; under contiguous CP the
        # dispatcher (_roll_tensor_cp_aware) does a single right-neighbour boundary exchange.
        self.cp_group = getattr(pg_collection, "cp", None)

    def forward(self, input_ids, position_ids, hidden_states, position_embeddings, embedding):
        """Run the MTP head.

        Args:
            input_ids (Tensor): ``[B, S]`` token ids (UN-rolled).
            position_ids (Tensor | None): ``[B, S]`` positions (UN-rolled).
            hidden_states (Tensor): the main model's PRE-collapse hc stream
                ``[B, S, hc_mult, H]`` (the ``pre_hc_head`` stack — grad flows back
                into the main model, matching Megatron's MTP).
            position_embeddings (dict): ``{"main": (cos, sin), "compress": ...}``
                computed from the ORIGINAL positions (rope is NOT rolled, matching
                Megatron's MTP, which recomputes logits at the token's own position).
            embedding (Callable): ``embedding(input_ids, position_ids) -> [B, S, H]``
                the host model's shared embedding.

        Returns:
            (Tensor, Tensor, Tensor): the ``[B, S, H]`` MTP hidden states (feed to
            the shared output head), plus the rolled ``input_ids`` and
            ``position_ids`` (for MTP-label alignment at loss integration time).
        """
        B, S, hc_mult, H = hidden_states.shape
        # Roll the token ids (and positions) left by one: depth-1 MTP consumes the
        # embedding of the (i+1)-th token at position i.
        rolled_input_ids, _ = _roll_tensor_cp_aware(input_ids, shifts=-1, dims=-1, cp_group=self.cp_group)
        rolled_position_ids = position_ids
        if position_ids is not None:
            rolled_position_ids, _ = _roll_tensor_cp_aware(position_ids, shifts=-1, dims=-1, cp_group=self.cp_group)

        # Embedding of the (i+1)-th token via the SHARED embedding; detached like
        # Megatron's MTP (no grad into the shared embedding from the MTP branch — it
        # is frozen anyway).
        embed = embedding(rolled_input_ids, rolled_position_ids).detach()  # [B, S, H]

        # V4 SPLIT fusion (sglang NextN): broadcast e_proj(enorm(embed)) over the hc
        # streams and add the per-stream h_proj(hnorm(prev_stream)).
        e_out = self.e_proj(self.enorm(embed))  # [B, S, H]
        h_out = self.h_proj(self.hnorm(hidden_states.reshape(B * S * hc_mult, H)))
        h_out = h_out.reshape(B, S, hc_mult, H)
        fused = e_out.unsqueeze(2) + h_out  # [B, S, hc_mult, H]

        # One V4 decoder layer over the fused hc stream.  The MoE hash router would
        # use input_ids; this is a top-k MoE, so the rolled ids are threaded but the
        # learned router ignores them.
        #
        # Activation checkpointing: recompute the MTP decoder layer in the backward,
        # mirroring the main model's per-layer checkpointing, so the MTP
        # head does not add a full layer of resident activations at ctx-16k.  Simpler
        # than the main loop's routing-replay handling: the MTP MoE routes LIVE
        # (excluded from rollout replay), so no ROUTING_REPLAY_STAGE coordination is
        # needed.  ``fused`` already requires grad (it derives from the grad-carrying hc
        # stream), so the reentrant-checkpoint "needs a grad input" precondition holds.
        use_act_ckpt = (
            getattr(self.mcore_config, "recompute_granularity", None) == "full"
            and self.training
            and torch.is_grad_enabled()
        )
        if use_act_ckpt:
            fused = torch.utils.checkpoint.checkpoint(
                self.transformer_layer,
                fused,
                position_embeddings,
                rolled_input_ids,
                use_reentrant=True,
            )
        else:
            fused = self.transformer_layer(fused, position_embeddings, rolled_input_ids)

        # MTP's own hc collapse + final norm -> [B, S, H].
        out = self.norm(self.hc_head(fused))
        return out, rolled_input_ids, rolled_position_ids


# ======================================================================================
# MTP training-loss helpers (stock-Megatron-faithful, but dist-free so they unit-test).
#
# These mirror ``GPTModel.forward`` postprocess (Megatron
# multi_token_prediction.py-driven, gpt_model.py:624-706): roll the labels/loss_mask
# LEFT once before the loop, then once more per MTP depth, compute per-depth CE with the
# DETACHED shared output head, log to ``MTPLossLoggingHelper``, and seed the MTP
# backward with ``MTPLossAutoScaler`` on the main hidden states.  Factored out of the
# model so the label-alignment + scaling can be tested without Megatron process groups.
# ======================================================================================


def roll_mtp_labels_and_masks(mtp_labels, loss_mask, num_mtp_layers, *, cp_group=None, packed_seq_params=None):
    """Produce per-depth rolled (labels, loss_mask, num_tokens), matching Megatron.

    Megatron rolls once before the MTP loop (aligning to the main +1 target) then once
    per depth, so MTP depth ``d`` (0-indexed) predicts token ``i + d + 2`` and its
    labels are ``input_ids`` rolled left by ``d + 2`` (gpt_model.py:624-668).  For V4
    (a single MTP head, ``num_mtp_layers == 1``) depth 0 -> labels rolled left by 2.
    """
    mtp_labels = mtp_labels.clone()
    mtp_labels, _ = _roll_tensor_cp_aware(
        mtp_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
    )
    if loss_mask is None:
        loss_mask = torch.ones_like(mtp_labels)
    else:
        loss_mask, _ = _roll_tensor_cp_aware(
            loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
    out = []
    for _ in range(num_mtp_layers):
        mtp_labels, _ = _roll_tensor_cp_aware(
            mtp_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        loss_mask, num_tokens = _roll_tensor_cp_aware(
            loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        out.append((mtp_labels, loss_mask, num_tokens))
    return out


def apply_mtp_loss(
    *,
    mtp_hidden,
    main_hidden,
    mtp_labels,
    loss_mask,
    output_logits_fn,
    ce_fn,
    config,
    cp_group=None,
    packed_seq_params=None,
    training=True,
    avg_group=None,
):
    """Compute + log the MTP loss and seed its backward on ``main_hidden``.

    Mirrors Megatron ``GPTModel.forward`` MTP postprocess (gpt_model.py:653-706):
      * per-depth CE via ``ce_fn(labels, logits_bsv)`` on the DETACHED shared head
        logits ``output_logits_fn(mtp_hidden)`` (the head is NOT trained by MTP);
      * masked, mean-reduced loss logged to ``MTPLossLoggingHelper``;
      * ``MTPLossAutoScaler.apply(main_hidden, scale * loss / num_tokens)`` — a no-op in
        forward, in backward it seeds ``loss.backward()`` (scaled by the scheduler's
        ``set_loss_scale``), so the MTP loss trains the MTP head + backprops into the
        main model via the shared ``hc_stream`` that ``mtp_hidden`` was derived from.

    ``output_logits_fn(mtp_hidden) -> [B,S,V]`` and ``ce_fn(labels, logits_bsv) -> [B,S]``
    are injected so this is testable without Megatron's output layer / cross entropy.
    Returns the (autoscaler-wrapped) ``main_hidden`` to feed the main output head.
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper

    mtp_num_layers = int(config.mtp_num_layers or 1)
    rolled = roll_mtp_labels_and_masks(
        mtp_labels, loss_mask, mtp_num_layers, cp_group=cp_group, packed_seq_params=packed_seq_params
    )
    for depth, (labels_d, mask_d, num_tokens) in enumerate(rolled):
        mtp_logits = output_logits_fn(mtp_hidden)  # [B, S, V]
        mtp_loss = ce_fn(labels_d, mtp_logits)  # [B, S]
        mtp_loss = mask_d * mtp_loss
        if training:
            MTPLossLoggingHelper.save_loss_to_tracker(
                torch.sum(mtp_loss) / num_tokens, depth, mtp_num_layers, avg_group=avg_group
            )
        scale = config.mtp_loss_scaling_factor / mtp_num_layers
        if getattr(config, "calculate_per_token_loss", False):
            main_hidden = MTPLossAutoScaler.apply(main_hidden, scale * mtp_loss)
        else:
            main_hidden = MTPLossAutoScaler.apply(main_hidden, scale * mtp_loss / num_tokens)
    return main_hidden


__all__ = ["V4MultiTokenPredictionLayer", "roll_mtp_labels_and_masks", "apply_mtp_loss"]
