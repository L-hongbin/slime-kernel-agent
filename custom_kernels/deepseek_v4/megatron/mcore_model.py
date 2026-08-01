"""V4-Flash as a Megatron-core ``LanguageModule`` (M-impl).

This is the mcore model the orchestrator gate asks for: a custom
``LanguageModule`` subclass that exposes the GPTModel-compatible surface slime's
train loop + ``loss.py`` drive, while reusing the M1-parity-validated V4 compute
(the three tilelang kernels A1/B2/B1 + the torch seams) for everything outside
the MoE, and a custom V4 MoE sub-layer that threads ``input_ids`` to the V4
routers and applies the exact V4 clamp+SwiGLU expert activation.

Design contract: ``handoffs/deepseek-v4/dsv4_megatron_sharding_contract.md``.
At TP=PP=EP=1 the model is the M0 torch path (forward-parity-validated at M1)
wrapped in the mcore module surface; every Megatron parallel primitive
short-circuits at world_size==1, so this introduces NO collectives and NO
numerical change vs the M0 ``V4Model`` (which already PASSES parity vs HF eager).
At EP>1, the MoE path reuses Megatron token dispatchers (prefer
``flex/deepep``) and keeps V4's custom router + clamp/SwiGLU expert math; this
has an isolated EP2 smoke, not a full actor-training proof.

What is mcore-parallel vs kernel vs reused (see ``M_IMPL_NOTES.md`` for the table):
  * ``embedding``      -> ``LanguageModelEmbedding`` (VocabParallel, TP=1 -> replicated)
  * ``output_layer``   -> ``ColumnParallelLinear`` (TP=1 -> plain fp32 matmul; untied)
  * attention/compressor/mHC -> the M1-validated replicated torch + 3 kernels
    (the EP plan FORBIDS TP-ing attention; replicated == ColumnParallel at TP=1)
  * MoE -> custom ``V4MoELayer``: ``V4TopKRouter``/``V4HashRouter`` (sqrtsoftplus +
    e_score_correction_bias + renorm + routed_scaling) + ``V4GroupedExperts``
    (exact clamp(swiglu_limit)->SwiGLU) + replicated shared MLP. At EP=1 this is
    numerically identical to HF ``DeepseekV4SparseMoeBlock``. At EP>1, Megatron
    dispatch/combine wraps the same local expert math.

PP>1 still needs a Megatron scheduler shape adapter because V4's intermediate
activation is ``[B,S,hc_mult,H]`` while stock non-interleaved PP assumes
``[S,B,H]``. The model stages themselves can recompute RoPE locally from
``input_ids``/``position_ids`` and the received stream; the full PP scheduler
path remains a separate smoke. LoRA is applied in the provider, not here.
"""

import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (used by the validated full-recompute path)


def _clamp(x, min=None, max=None):
    """Clamp in the forward and use the identity straight-through gradient."""
    # clamp(x).detach() preserves the exact clamped value without an extra bf16
    # add/sub; x - x.detach() is exactly zero in value and has gradient one.
    return x.clamp(min=min, max=max).detach() + (x - x.detach())


from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from transformers.activations import ACT2FN

from .attention import V4Attention
from .cp_utils import CP_HALO, assert_local_len_aligned, get_cp_info
from .decoder import V4HyperConnection
from .rope import DeepseekV4RotaryEmbedding, V4RMSNorm


def _routing_replay_enabled():
    return os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1"


def _forward_v4_decoder_layers(
    layers,
    hidden_states,
    position_embeddings,
    input_ids,
    *,
    use_act_ckpt,
    recompute_num_layers=1,
    recompute_method="uniform",
    use_reentrant=True,
):
    """Run V4 decoder layers, optionally checkpointing layer segments.

    Megatron semantics, reimplemented for V4's hand-written decoder loop
    (``TransformerBlock``'s checkpoint path can't run here):

    - ``recompute_method="uniform"``: ``recompute_num_layers`` is the number of
      consecutive layers in one checkpointed unit; ALL layers are recomputed.
    - ``recompute_method="block"``: the FIRST ``recompute_num_layers`` layers
      are checkpointed individually; the remaining layers store their
      activations (memory-for-speed dial — backward skips their recompute).
    """
    if not use_act_ckpt:
        for layer in layers:
            hidden_states = layer(hidden_states, position_embeddings, input_ids)
        return hidden_states

    if isinstance(recompute_num_layers, bool) or not isinstance(recompute_num_layers, int):
        raise TypeError("V4 recompute_num_layers must be an integer, " f"got {recompute_num_layers!r}")
    if recompute_num_layers <= 0:
        raise ValueError("V4 recompute_num_layers must be greater than zero, " f"got {recompute_num_layers}")
    if recompute_method not in ("uniform", "block"):
        raise ValueError("V4 recompute_method must be 'uniform' or 'block', " f"got {recompute_method!r}")
    if recompute_method == "block" and recompute_num_layers > len(layers):
        raise ValueError(
            "V4 block recompute_num_layers cannot exceed the local decoder "
            f"layer count: K={recompute_num_layers}, local_layers={len(layers)}"
        )

    def _checkpoint_segment(hidden, segment):
        # Capture an immutable snapshot in the closure.  A default argument is
        # intentional: checkpoint executes this function again during backward,
        # after the loop variable has advanced to the final segment.
        def run_segment(hidden, positions, tokens, segment=segment):
            for layer in segment:
                hidden = layer(hidden, positions, tokens)
            return hidden

        return torch.utils.checkpoint.checkpoint(
            run_segment,
            hidden,
            position_embeddings,
            input_ids,
            use_reentrant=use_reentrant,
        )

    if recompute_method == "block":
        for idx, layer in enumerate(layers):
            if idx < recompute_num_layers:
                hidden_states = _checkpoint_segment(hidden_states, (layer,))
            else:
                hidden_states = layer(hidden_states, position_embeddings, input_ids)
        return hidden_states

    for start in range(0, len(layers), recompute_num_layers):
        segment = tuple(layers[start : start + recompute_num_layers])
        hidden_states = _checkpoint_segment(hidden_states, segment)
    return hidden_states


def _v4_stage_builds_embedding(pre_process, post_process, enable_mtp):
    """Whether this PP stage builds the shared embedding.

    Always on the first stage (``pre_process``).  Additionally on the LAST stage
    (``post_process``) when the MTP head is enabled, because the MTP head embeds its
    rolled ``input_ids`` there.  At PP=1 the first==last stage already qualifies via
    ``pre_process``.  Mirrors Megatron's ``pre_process or mtp_process`` embedding build.
    """
    return bool(pre_process or (post_process and enable_mtp))


# ======================================================================================
# MoE: custom V4 routers (input_ids-threaded) + grouped experts (exact clamp+SwiGLU).
#
# At EP=1 these are byte-faithful to HF DeepseekV4{TopK,Hash}Router / DeepseekV4Experts /
# DeepseekV4MLP (the shared expert).  We keep the HF parameter NAMES and shapes so the
# HF->mcore state-dict copy is the identity for the MoE sub-tree (see M_IMPL_NOTES key map).
# ======================================================================================
class V4TopKRouter(nn.Module):
    """Learned top-k router: sqrtsoftplus(scores) + e_score_correction_bias top-k,
    renorm, * routed_scaling_factor.  Faithful to ``DeepseekV4TopKRouter`` (HF:1019).

    NOTE: stock mcore ``TopKRouter`` only supports softmax/sigmoid and rejects an
    expert bias for non-sigmoid scoring, so V4's ``sqrtsoftplus`` + correction bias
    needs this custom router (sharding contract §risks 1).
    """

    def __init__(self, config, *, enable_routing_replay=True):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts  # alias -> n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)
        # ``enable_routing_replay=False`` keeps this router OUT of the rollout routing
        # replay machinery: it never constructs a ``RoutingReplay`` (which would append a
        # spurious entry to the global ``all_routing_replays`` list and shift the
        # main-layer offset mapping) and never gets a ``routing_replay`` attr, so
        # ``_select_indices`` falls through to a LIVE top-k. The MTP layer's MoE uses this
        # (it is not one of the main decoder layers whose routing the rollout records).
        if enable_routing_replay and _routing_replay_enabled():
            from slime.utils.routing_replay import register_routing_replay

            register_routing_replay(self)

    def _select_indices(self, scores):
        biased_scores = scores + self.e_score_correction_bias
        if not (_routing_replay_enabled() and hasattr(self, "routing_replay")):
            return torch.topk(biased_scores, self.top_k, dim=-1, sorted=False).indices

        stage = os.environ["ROUTING_REPLAY_STAGE"]
        if stage == "fallthrough":
            indices = torch.topk(biased_scores, self.top_k, dim=-1, sorted=False).indices
        elif stage == "record":
            indices = torch.topk(biased_scores, self.top_k, dim=-1, sorted=False).indices
            self.routing_replay.record(indices)
        elif stage == "replay_forward":
            indices = self.routing_replay.pop_forward()
        elif stage == "replay_backward":
            indices = self.routing_replay.pop_backward()
        else:
            raise ValueError(f"Unknown ROUTING_REPLAY_STAGE={stage!r}")

        assert indices.shape == (scores.shape[0], self.top_k), (
            f"replayed V4 top-k indices shape {indices.shape} does not match "
            f"scores rows {scores.shape[0]} and top_k {self.top_k}"
        )
        return indices.long()

    def forward(self, hidden_states, input_ids=None):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        # sglang scores in fp32: `scoring_func_impl(gating_output.float())` (topk.py:639).
        # Match it so the score_fn, the top-k SELECTION (borderline experts), and the
        # renorm all agree with the rollout. (e_score_correction_bias is already fp32.)
        scores = self.score_fn(logits.float())
        indices = self._select_indices(scores)
        weights = scores.gather(1, indices)
        # Match sglang exactly: renorm with a bare sum (no +1e-20 guard). Scores are
        # positive (sigmoid/sqrtsoftplus) so the denominator is >0; sglang omits the
        # guard (moe/topk.py:643 `topk_weights / topk_weights.sum(...)`).
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights * self.routed_scaling_factor, indices


from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HashRouter as _HFDeepseekV4HashRouter


class V4HashRouter(_HFDeepseekV4HashRouter):
    """The OFFICIAL HF ``DeepseekV4HashRouter`` (structure/init/``tid2eid`` buffer
    inherited upstream — tracks community updates), with ONE deliberate forward
    deviation: the weight renorm uses sglang's bare sum (NO ``+1e-20`` guard) so
    train-side gate weights bit-match the rollout engine (sglang moe/topk.py:643).
    Scores are strictly positive (sigmoid/sqrtsoftplus) so the denominator is >0.
    Needs ``input_ids`` threaded in (sharding contract §risks 2) — the single
    biggest deviation from stock MoELayer.
    """

    def forward(self, hidden_states, input_ids):
        assert input_ids is not None, "V4HashRouter requires input_ids"
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        # sglang-exact renorm (see class docstring) — the only line that differs
        # from the inherited HF forward.
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights * self.routed_scaling_factor, indices


# --- Packed-MXFP4 frozen experts (V4_FP4_FROZEN_EXPERTS=1) --------------------
# The OFFICIAL DeepSeek-V4-Flash checkpoint stores routed experts as packed
# MXFP4: int8 nibble pairs (E2M1, low nibble = even K element, high = odd) plus
# per-32-along-K E8M0 scales (uint8 exponent bytes, value = 2^(byte-127)).
# These bytes stay RESIDENT verbatim (uint8 buffers); compute is W4A16 — the
# weight is unpacked transiently to bf16 and multiplied against bf16
# activations, matching sglang's SM90 flashinfer_mxfp4/Marlin W4A16 serving of
# the same checkpoint. The decode is EXACT in bf16 (E2M1 magnitudes
# {0,.5,1,1.5,2,3,4,6} need <=3 significand bits; the scale is a power of two),
# validated bit-identical to deep_gemm.cast_back_from_fp4(gran_k=32) and to the
# secondary FP8 checkpoint's dequant of the same experts (lossless FP4->FP8
# expansion) on the official fixture — see
# handoffs/deepseek-v4/fp4_w4a16_design.md and
# tests/deepseek-v4/test_dsv4_fp4_frozen_experts.py.

_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.bfloat16,
)

_MXFP4_GROUP = 32  # E8M0 scale granularity along K (official checkpoint layout)


def _v4_fp4_frozen_experts_enabled():
    """Read the packed-MXFP4 mode at model-construction time."""
    return os.environ.get("V4_FP4_FROZEN_EXPERTS", "0") == "1"


def _unpack_mxfp4(w_pack, sf, out_dtype=torch.bfloat16):
    """packed E2M1 uint8 [O, K/2] + E8M0 uint8 [O, K/32] -> [O, K] bf16.

    MEMORY-CRITICAL: callers must treat the result as a transient and never
    retain it across autograd. bf16 multiply is exact here
    (power-of-two scale only shifts the exponent), so unlike the fp8 path there
    is no bf16-vs-fp32 dequant tradeoff at all."""
    assert (
        w_pack.dtype == torch.uint8 and sf.dtype == torch.uint8
    ), f"packed MXFP4 expects uint8 buffers, got {w_pack.dtype}/{sf.dtype}"
    O, Kh = w_pack.shape
    K = Kh * 2
    assert sf.shape == (O, K // _MXFP4_GROUP), f"scale shape {tuple(sf.shape)} != ({O}, {K // _MXFP4_GROUP})"
    lut = _E2M1_LUT.to(device=w_pack.device)
    lo = lut[(w_pack & 0x0F).long()]  # even K indices
    hi = lut[(w_pack >> 4).long()]  # odd K indices
    w = torch.stack((lo, hi), dim=-1).reshape(O, K)
    # E8M0 decode 2^(e-127) via torch's NATIVE float8_e8m0fnu cast — correct for
    # all 256 byte values incl. the boundaries (0x00 -> 2^-127, 0xFF -> NaN),
    # unlike the (byte<<23) bit-trick which yields 0.0/+Inf there (codex impl
    # review finding 2; bytes 1-254 are bit-identical either way, and the
    # boundary bytes are additionally hard-rejected by verify_fp4_loaded).
    scale = sf.view(torch.float8_e8m0fnu).to(torch.float32).to(torch.bfloat16)
    return (w.view(O, K // _MXFP4_GROUP, _MXFP4_GROUP) * scale[:, :, None]).reshape(O, K).to(out_dtype)


class _FrozenFp4ExpertLinear(torch.autograd.Function):
    """y = x @ W^T for a frozen packed-MXFP4 expert weight, WITHOUT retaining
    the bf16 weight (ctx saves only the resident packed buffers; the bf16 weight
    is a transient re-derived in forward and again in backward. No grad_weight
    is produced because routed experts are frozen."""

    @staticmethod
    def forward(ctx, x, w_pack, w_sf):
        ctx.save_for_backward(w_pack, w_sf)  # resident uint8 buffers — no new memory
        w = _unpack_mxfp4(w_pack, w_sf)  # transient, NOT saved
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, grad_out):
        w_pack, w_sf = ctx.saved_tensors
        w = _unpack_mxfp4(w_pack, w_sf)  # transient re-unpack
        grad_x = grad_out @ w  # grad_x = grad_y @ W  (y = x @ W^T)
        return grad_x, None, None


class V4GroupedExperts(nn.Module):
    """Grouped routed experts stored as 3D weights ``[E, 2I, H]`` / ``[E, H, I]``
    (HF ``DeepseekV4Experts`` layout) with the exact V4 gate: clamp(gate, max=limit),
    clamp(up, [-limit, limit]), then ``act_fn(gate) * up`` (HF:1009 ``_apply_gate``).

    mcore ``GroupedMLP``/``TEGroupedMLP`` apply a fused SwiGLU WITHOUT the clamp, so
    even at TP=1 the activation must be customized for parity (sharding contract
    §risks 2 / codex order step 3).  The grouped-GEMM here is the HF per-expert loop
    (numerically exact at EP=1); the ``GroupedMLP`` flat-weight + dispatcher path is
    swapped in at the EP>1 milestone where the all-to-all is non-trivial.
    """

    def __init__(
        self,
        config,
        *,
        expert_model_parallel_size=1,
        expert_model_parallel_rank=0,
        expert_data_parallel_group=None,
    ):
        super().__init__()
        self._checkpoint_config = config
        self.num_global_experts = config.num_local_experts
        self.expert_model_parallel_size = int(expert_model_parallel_size)
        self.expert_model_parallel_rank = int(expert_model_parallel_rank)
        # Expert shards are replicated across the expert-DP group.  Distributed
        # checkpointing must see exactly one main replica for every EP shard; in
        # particular, CP>1 can make expert-DP larger than one even when ordinary
        # model weights merely look CP-replicated.  This mirrors Megatron's native
        # GroupedMLP ``self.dp_group = pg_collection.expt_dp`` contract.
        self.expert_data_parallel_group = expert_data_parallel_group
        if self.expert_model_parallel_size < 1:
            raise ValueError("expert_model_parallel_size must be >= 1")
        if not 0 <= self.expert_model_parallel_rank < self.expert_model_parallel_size:
            raise ValueError(
                f"expert_model_parallel_rank={self.expert_model_parallel_rank} must be in "
                f"[0, {self.expert_model_parallel_size})"
            )
        if self.num_global_experts % self.expert_model_parallel_size:
            raise ValueError(
                f"num_global_experts={self.num_global_experts} must divide "
                f"expert_model_parallel_size={self.expert_model_parallel_size}"
            )
        self.num_experts = self.num_global_experts // self.expert_model_parallel_size
        self.local_expert_start = self.expert_model_parallel_rank * self.num_experts
        self.local_expert_end = self.local_expert_start + self.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size  # alias -> moe_intermediate_size
        # Packed-MXFP4 mode (V4_FP4_FROZEN_EXPERTS=1): the OFFICIAL checkpoint's
        # packed expert bytes stay resident verbatim — no bf16 Parameters are ever
        # created, torch_dist carries the packed uint8 tensors directly (see
        # sharded_state_dict), and compute is W4A16 via transient unpack. Read at
        # CONSTRUCTION time (both the conversion driver and the trainer export the
        # env before building the model).
        self._experts_fp4 = _v4_fp4_frozen_experts_enabled()
        if self._experts_fp4:
            E, H, I = self.num_experts, self.hidden_dim, self.intermediate_dim
            assert (
                H % (2 * _MXFP4_GROUP) == 0 and I % (2 * _MXFP4_GROUP) == 0
            ), f"packed MXFP4 needs dims divisible by {2 * _MXFP4_GROUP}, got H={H} I={I}"
            # gate_up: logical [E, 2I, H] -> packed [E, 2I, H/2] + scales [E, 2I, H/32]
            # down:    logical [E, H, I]  -> packed [E, H, I/2]  + scales [E, H, I/32]
            # Buffers (not Parameters): invisible to DDP/optimizer, untouched by
            # .bfloat16() casts. persistent=True is REQUIRED (unlike the post-load
            # fp8 buffers): these exist AT LOAD TIME, and Megatron's checkpoint load
            # finishes with module.load_state_dict(...) whose key set only contains
            # persistent buffers — persistent=False would make every packed key an
            # unexpected_key and strict loads would raise.
            self.register_buffer("gate_up_proj_fp4", torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8))
            self.register_buffer("gate_up_proj_sf", torch.zeros(E, 2 * I, H // _MXFP4_GROUP, dtype=torch.uint8))
            self.register_buffer("down_proj_fp4", torch.zeros(E, H, I // 2, dtype=torch.uint8))
            self.register_buffer("down_proj_sf", torch.zeros(E, H, I // _MXFP4_GROUP, dtype=torch.uint8))
        else:
            self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
            self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
            expert_parallel = self.expert_model_parallel_size > 1
            self.gate_up_proj.allreduce = not expert_parallel
            self.down_proj.allreduce = not expert_parallel
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        gate = _clamp(gate, max=self.limit)
        up = _clamp(up, min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def _expert_matmul(self, x, name, idx):
        """Apply one frozen expert, unpacking packed MXFP4 weights transiently."""
        if getattr(self, "_experts_fp4", False):
            return _FrozenFp4ExpertLinear.apply(
                x,
                getattr(self, f"{name}_fp4")[idx],
                getattr(self, f"{name}_sf")[idx],
            )
        return F.linear(x, getattr(self, name)[idx])

    def _expert_weight(self, name, idx):
        """Return expert ``idx``'s weight, unpacking packed MXFP4 on demand."""
        if getattr(self, "_experts_fp4", False):
            return _unpack_mxfp4(getattr(self, f"{name}_fp4")[idx], getattr(self, f"{name}_sf")[idx])
        return getattr(self, name)[idx]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.expert_model_parallel_size > 1:
            raise NotImplementedError(
                "V4GroupedExperts EP>1 currently supports checkpoint/materialization only; "
                "call forward_dispatched() with a Megatron MoE token dispatcher instead."
            )
        # hidden_states [N, H]; top_k_index [N, top_k]; top_k_weights [N, top_k].
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(self._expert_matmul(hidden_states[token_idx], "gate_up_proj", expert_idx))
            current = self._expert_matmul(current, "down_proj", expert_idx) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def forward_dispatched(self, hidden_states, tokens_per_expert, permuted_probs=None):
        """Run local experts on dispatcher-permuted tokens.

        Megatron's MoE dispatchers return tokens grouped by local expert, with
        ``tokens_per_expert`` giving each contiguous segment length.  This keeps
        V4's exact clamp+SwiGLU expert math while reusing Megatron/DeepEP for the
        cross-rank token dispatch/combine.
        """
        if tokens_per_expert.numel() != self.num_experts:
            raise ValueError(
                f"tokens_per_expert has {tokens_per_expert.numel()} entries, expected "
                f"{self.num_experts} local experts"
            )
        # Reject negatives, require counts to sum to the token rows, and handle an
        # empty dispatcher batch without entering any expert loop.
        if (tokens_per_expert < 0).any():
            raise ValueError(f"tokens_per_expert has a negative entry: {tokens_per_expert.tolist()}")
        total = int(tokens_per_expert.sum().item())
        if total != hidden_states.shape[0]:
            raise ValueError(f"tokens_per_expert sums to {total}, but hidden_states has {hidden_states.shape[0]} rows")
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        output = torch.empty_like(hidden_states)
        offset = 0
        for local_expert_idx, count_tensor in enumerate(tokens_per_expert):
            count = int(count_tensor.item())
            if count < 0:
                raise ValueError(f"tokens_per_expert[{local_expert_idx}] is negative: {count}")
            if count == 0:
                continue
            next_offset = offset + count
            current = self._apply_gate(
                self._expert_matmul(hidden_states[offset:next_offset], "gate_up_proj", local_expert_idx)
            )
            current = self._expert_matmul(current, "down_proj", local_expert_idx)
            if permuted_probs is not None:
                current = current * permuted_probs[offset:next_offset, None]
            output[offset:next_offset] = current.to(output.dtype)
            offset = next_offset
        if offset != hidden_states.shape[0]:
            raise ValueError(
                f"tokens_per_expert sums to {offset}, but hidden_states has " f"{hidden_states.shape[0]} rows"
            )
        return output

    def verify_fp4_loaded(self):
        """Post-load sanity for the packed mode. Rejects ANY invalid scale byte,
        not just floods (codex review 2026-07-16, finding 9): 0x00 decodes to
        2^-127 (an unpopulated zero-init buffer OR a nonsensical weight scale —
        the official checkpoint contains zero 0x00 scale bytes), and 0xFF is NaN
        in E8M0 — a single one poisons 32 consecutive weights with NaN/Inf."""
        assert getattr(self, "_experts_fp4", False), "verify_fp4_loaded requires packed-MXFP4 mode"
        for name in ("gate_up_proj", "down_proj"):
            sf = getattr(self, f"{name}_sf")
            n_zero = int((sf == 0x00).sum())
            n_nan = int((sf == 0xFF).sum())
            if n_zero or n_nan:
                raise RuntimeError(
                    f"packed-MXFP4 {name}_sf has invalid E8M0 scale bytes: "
                    f"{n_zero} x 0x00 (unpopulated buffer / wrong checkpoint family) and "
                    f"{n_nan} x 0xFF (NaN scale) out of {sf.numel()} — refusing to train"
                )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Shard grouped expert tensors on the global expert axis for EP conversion."""
        del metadata
        expert_dp_rank = self.expert_data_parallel_group.rank() if self.expert_data_parallel_group is not None else 0
        replica_id = (0, 0, expert_dp_rank)
        # Packed-MXFP4 mode: torch_dist carries the packed uint8 tensors verbatim.
        # These entries serve the LOAD path (and conversion-time SAVE); adapter-only
        # training saves filter to requires_grad adapter keys, so the packed buffers
        # never leak into adapter checkpoints.
        if getattr(self, "_experts_fp4", False):
            prepend_axis_num = len(sharded_offsets)
            ep_offset = (prepend_axis_num, self.expert_model_parallel_rank, self.expert_model_parallel_size)
            return {
                f"{prefix}{name}": ShardedTensor.from_rank_offsets(
                    f"{prefix}{name}",
                    getattr(self, name),
                    *sharded_offsets,
                    ep_offset,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                )
                for name in ("gate_up_proj_fp4", "gate_up_proj_sf", "down_proj_fp4", "down_proj_sf")
            }
        prepend_axis_num = len(sharded_offsets)
        ep_offset = (prepend_axis_num, self.expert_model_parallel_rank, self.expert_model_parallel_size)
        return {
            f"{prefix}gate_up_proj": ShardedTensor.from_rank_offsets(
                f"{prefix}gate_up_proj",
                self.gate_up_proj,
                *sharded_offsets,
                ep_offset,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
            f"{prefix}down_proj": ShardedTensor.from_rank_offsets(
                f"{prefix}down_proj",
                self.down_proj,
                *sharded_offsets,
                ep_offset,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
        }


class V4SharedExpertMLP(nn.Module):
    """Replicated shared expert (HF ``DeepseekV4MLP``): SiLU-gated MLP at moe width."""

    def __init__(self, config):
        super().__init__()
        self._checkpoint_config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        # V4 clamps the SwiGLU gate/up before the activation on the SHARED expert
        # too (sglang serves shared experts with swiglu_limit; deepseek_v2.py:675 +
        # silu_and_mul_masked_post_quant.cuh). Omitting it here diverged the rollout
        # on any token whose shared gate/up exceeds +-limit. Same clamp as the routed
        # V4GroupedExperts._apply_gate, applied in bf16.
        self.limit = config.swiglu_limit

    def _act(self, gate, up):
        gate = _clamp(gate, max=self.limit)
        up = _clamp(up, min=-self.limit, max=self.limit)
        assert self._checkpoint_config.hidden_act == "silu", (
            "DS-V4 shared-expert alignment requires hidden_act='silu'; " f"got {self._checkpoint_config.hidden_act!r}"
        )
        # Match sglang's silu_and_mul_clamp boundary: compute SiLU and the
        # product in fp32, then round to bf16 once.
        return (self.act_fn(gate.float()) * up.float()).to(gate.dtype)

    def forward(self, x):
        return self.down_proj(self._act(self.gate_proj(x), self.up_proj(x)))


class V4MoELayer(nn.Module):
    """V4 MoE sub-layer == HF ``DeepseekV4SparseMoeBlock`` (HF:1071), restructured into
    mcore-style router/experts/shared modules with ``input_ids`` threaded to the router.

    Module attribute names mirror HF (``gate`` / ``experts`` / ``shared_experts``) so
    the HF->mcore MoE state-dict copy is the identity.
    """

    def __init__(
        self,
        config,
        layer_idx,
        *,
        expert_model_parallel_size=1,
        expert_model_parallel_rank=0,
        mcore_config: TransformerConfig | None = None,
        pg_collection=None,
        mlp_type_override: str | None = None,
        enable_routing_replay: bool = True,
    ):
        super().__init__()
        # ``mlp_type_override`` decouples the router kind from ``config.mlp_layer_types``:
        # the V4 MTP head's FFN is a learned top-k MoE (its ``mtp.0.ffn.gate`` has a
        # correction ``bias`` and NO ``tid2eid``), so it is built with "moe" regardless
        # of what the layer-0 schedule says.  ``enable_routing_replay=False`` keeps the
        # MTP router live (see V4TopKRouter).
        mlp_type = mlp_type_override if mlp_type_override is not None else config.mlp_layer_types[layer_idx]
        self.is_hash = mlp_type == "hash_moe"
        self.num_global_experts = config.num_local_experts
        self.mcore_config = mcore_config
        self.gate = (
            V4HashRouter(config) if self.is_hash else V4TopKRouter(config, enable_routing_replay=enable_routing_replay)
        )
        self.experts = V4GroupedExperts(
            config,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_model_parallel_rank=expert_model_parallel_rank,
            expert_data_parallel_group=(
                getattr(pg_collection, "expt_dp", None) if pg_collection is not None else None
            ),
        )
        self.shared_experts = V4SharedExpertMLP(config)
        self.token_dispatcher = self._build_token_dispatcher(mcore_config, pg_collection)

    def _build_token_dispatcher(self, mcore_config, pg_collection):
        if self.experts.expert_model_parallel_size == 1:
            return None
        if mcore_config is None or pg_collection is None:
            raise ValueError("EP>1 V4MoELayer requires Megatron config and process groups")
        local_expert_indices = list(range(self.experts.local_expert_start, self.experts.local_expert_end))
        dispatcher_type = mcore_config.moe_token_dispatcher_type
        if dispatcher_type == "flex":
            return MoEFlexTokenDispatcher(
                self.experts.num_experts,
                local_expert_indices,
                config=mcore_config,
                pg_collection=pg_collection,
            )
        if dispatcher_type == "alltoall":
            return MoEAlltoAllTokenDispatcher(
                self.experts.num_experts,
                local_expert_indices,
                config=mcore_config,
                pg_collection=pg_collection,
            )
        if dispatcher_type == "allgather":
            return MoEAllGatherTokenDispatcher(
                self.experts.num_experts,
                local_expert_indices,
                config=mcore_config,
                pg_collection=pg_collection,
            )
        raise ValueError(f"Unsupported V4 MoE token dispatcher: {dispatcher_type}")

    def _indices_to_routing_tensors(self, indices, weights):
        num_tokens = indices.shape[0]
        routing_map = torch.zeros(num_tokens, self.num_global_experts, dtype=torch.bool, device=indices.device)
        routing_map.scatter_(1, indices, True)
        probs = torch.zeros(num_tokens, self.num_global_experts, dtype=torch.float32, device=weights.device)
        probs = probs.scatter_add(1, indices, weights.float())
        return routing_map, probs

    def _dispatch_experts(self, flat_hidden_states, indices, weights):
        routing_map, probs = self._indices_to_routing_tensors(indices, weights)
        dispatched_input, probs = self.token_dispatcher.dispatch_preprocess(flat_hidden_states, routing_map, probs)
        dispatched_input, probs = self.token_dispatcher.token_dispatch(dispatched_input, probs)
        expert_input, tokens_per_expert, permuted_probs = self.token_dispatcher.dispatch_postprocess(
            dispatched_input, probs
        )
        expert_output = self.experts.forward_dispatched(expert_input, tokens_per_expert, permuted_probs)
        output = self.token_dispatcher.combine_preprocess(expert_output)
        output = self.token_dispatcher.token_combine(output)
        return self.token_dispatcher.combine_postprocess(output)

    def forward(self, hidden_states, input_ids=None):
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states, input_ids)
        if self.token_dispatcher is None:
            routed = self.experts(flat, indices, weights)
        else:
            routed = self._dispatch_experts(flat, indices, weights)
        routed = routed.view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


# --- Fixed fp32 mHC post/comb combine (T5 probe S6/§5) -------------------------
#
# handoffs/deepseek-v4/t5_mhc_probe.md §S6: the dominant mHC train<->serve seam is THIS
# glue — casting post/comb to bf16 and materializing two bf16 terms flips 43-48% of
# output bytes (~3.3e-3 rel) vs sglang's stock mhc_post_tilelang, which keeps the
# coefficients fp32, forms c*d + sum_k a[k,j]*b[k,h] in fp32, and does ONE bf16 store.
# The combine reproduces that schedule with plain torch
# ops (fp32 coefficients, fp32 multiply/matmul/add, single final cast): bit-identical
# (n=1) / <=4e-5 bytediff (cuBLAS batched-matmul order residue) to stock serving AND
# more accurate vs an fp64 oracle (rel-err ~2.9e-3 -> ~1.7e-3).  Backward is plain
# autograd through the fp32 ops (LoRA adapter grads flow through attn/mlp output;
# hidden-stream grads through the matmul). The trainer-side fp32 schedule is
# the sole supported alignment direction.


def _v4_hc_post_combine(post, comb, branch_output, hidden_states, dtype):
    """One mHC post/comb combine site of ``V4DecoderLayer.forward`` (attn or ffn).

    Coefficients and the combine stay fp32 until one final ``dtype`` cast,
    matching sglang's stock ``mhc_post``.
    """
    return (
        post.float().unsqueeze(-1) * branch_output.float().unsqueeze(-2)
        + torch.matmul(comb.float().transpose(-1, -2), hidden_states.float())
    ).to(dtype)


# ======================================================================================
# Decoder layer: the M1-validated V4DecoderLayer math, with the custom V4MoELayer.
# ======================================================================================
class V4DecoderLayer(nn.Module):
    """Custom V4 decoder block (HF:1091). Threads the ``[B,S,hc_mult,H]`` stream, two
    mHC mix sites, ``input_ids`` to the MoE.  Identical math to the M0 ``V4DecoderLayer``
    (M1-validated) but with the mcore-structured ``V4MoELayer`` as ``mlp``.
    """

    def __init__(
        self,
        config,
        layer_idx,
        *,
        expert_model_parallel_size=1,
        expert_model_parallel_rank=0,
        mcore_config: TransformerConfig | None = None,
        pg_collection=None,
        attn_layer_type: str | None = None,
        mlp_type: str | None = None,
        enable_routing_replay: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = V4Attention(config, layer_idx, layer_type_override=attn_layer_type)
        self.mlp = V4MoELayer(
            config,
            layer_idx,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_model_parallel_rank=expert_model_parallel_rank,
            mcore_config=mcore_config,
            pg_collection=pg_collection,
            mlp_type_override=mlp_type,
            enable_routing_replay=enable_routing_replay,
        )
        self.input_layernorm = V4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = V4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = V4HyperConnection(config)
        self.ffn_hc = V4HyperConnection(config)

    def forward(self, hidden_states, position_embeddings, input_ids):
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(self.input_layernorm(collapsed), position_embeddings)
        hidden_states = _v4_hc_post_combine(post, comb, attn_output, hidden_states, dtype)

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        hidden_states = _v4_hc_post_combine(post, comb, mlp_output, hidden_states, dtype)
        return hidden_states


# ======================================================================================
# The LanguageModule subclass: the GPTModel-compatible surface slime drives.
# ======================================================================================
class V4LanguageModel(LanguageModule):
    """DeepSeek-V4-Flash as a Megatron ``LanguageModule`` (TP=PP=EP=1).

    Args:
        config: a Megatron ``TransformerConfig`` (drives the LanguageModule base,
            embedding, output_layer, process groups).
        hf_config: the HF ``DeepseekV4Config`` that drives the V4 compute modules
            (kept separate so the M1-validated modules are byte-identical to M0).
        pre_process / post_process: PP stage flags (both True at PP=1).
    """

    def __init__(
        self,
        config: TransformerConfig,
        hf_config,
        pre_process=True,
        post_process=True,
        layer_ids: list[int] | tuple[int, ...] | None = None,
        expert_model_parallel_size=1,
        expert_model_parallel_rank=0,
        enable_mtp: bool | None = None,
    ):
        super().__init__(config=config)
        self.hf_config = hf_config
        self.pre_process = pre_process
        self.post_process = post_process
        # MTP head gate (off by default; None -> False). Built ONLY on the last PP
        # stage (post_process) — it consumes the pre-collapse hc stream + the shared
        # output head, both of which live there.
        self.enable_mtp = bool(enable_mtp)
        self.layer_ids = tuple(range(hf_config.num_hidden_layers) if layer_ids is None else layer_ids)
        self.expert_model_parallel_size = int(expert_model_parallel_size)
        self.expert_model_parallel_rank = int(expert_model_parallel_rank)
        self.share_embeddings_and_output_weights = False  # tie_word_embeddings=False (config:178)
        self.model_type = ModelType.encoder_or_decoder
        self.max_sequence_length = (
            config.max_position_embeddings
            if hasattr(config, "max_position_embeddings")
            else hf_config.max_position_embeddings
        )
        self.vocab_size = hf_config.vocab_size
        self.position_embedding_type = "rope"
        self.input_tensor = None

        # V4's own interleaved partial RoPE is needed on every PP stage. Passing
        # cos/sin through pipeline P2P would require an extra non-standard payload;
        # recomputing it locally from position_ids is cheaper and deterministic.
        self.rotary_emb = DeepseekV4RotaryEmbedding(hf_config)

        # The MTP head (last PP stage) needs the SHARED embedding to embed its rolled
        # input_ids.  At PP=1 the first==last stage already builds it; at PP>1 the last
        # stage must build its own copy (Megatron builds the embedding on every
        # ``mtp_process`` stage, multi_token_prediction path).  V4's embedding is FROZEN
        # (LoRA touches attention only), so this copy stays byte-identical to the
        # first-stage embedding via the checkpoint load — no grad-tie/all-reduce needed
        # (unlike Megatron's trainable tied embedding).
        self._build_embedding = _v4_stage_builds_embedding(self.pre_process, self.post_process, self.enable_mtp)
        if self._build_embedding:
            # VocabParallelEmbedding; at TP=1 this is a plain replicated nn.Embedding.
            self.embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type="none",  # V4 owns its rotary; no learned pos-emb
                tp_group=self.pg_collection.tp,
            )

        self.layers = nn.ModuleList(
            [
                V4DecoderLayer(
                    hf_config,
                    i,
                    expert_model_parallel_size=self.expert_model_parallel_size,
                    expert_model_parallel_rank=self.expert_model_parallel_rank,
                    mcore_config=config,
                    pg_collection=self.pg_collection,
                )
                for i in self.layer_ids
            ]
        )

        if self.post_process:
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperHead

            self.hc_head = DeepseekV4HyperHead(hf_config)
            self.norm = V4RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
            # Untied lm_head as a ColumnParallelLinear (TP=1 -> plain matmul, fp32 out
            # via the LinearForLastLayer-style float cast in forward).
            self.output_layer = ColumnParallelLinear(
                hf_config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=False,  # parallel_output; TP=1 -> no gather needed
                skip_bias_add=False,
                tp_group=self.pg_collection.tp,
            )

            # V4 MTP head (gated). Lives on the last PP stage: it consumes the
            # pre-collapse hc stream produced here and the shared output_layer.
            self.mtp = None
            if self.enable_mtp:
                from .mtp import V4MultiTokenPredictionLayer

                self.mtp = V4MultiTokenPredictionLayer(
                    hf_config,
                    expert_model_parallel_size=self.expert_model_parallel_size,
                    expert_model_parallel_rank=self.expert_model_parallel_rank,
                    mcore_config=config,
                    pg_collection=self.pg_collection,
                )
        else:
            self.mtp = None

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def setup_embeddings_and_output_layer(self):
        """V4 override: tag the standard param attributes, skip the tied-embedding tie.

        Setting ``config.mtp_num_layers=1`` (so Megatron's schedule sets the MTP loss
        scale) makes the base ``LanguageModule.setup_embeddings_and_output_layer`` run
        the *tied*-embedding machinery (zero + all-reduce a duplicated embedding across
        the embedding group, using ``self.vp_stage`` / a configured embd group).  That
        is wrong for V4: its embedding and output head are UNTIED
        (``share_embeddings_and_output_weights=False``) and FROZEN (LoRA touches
        attention only).  The MTP head's last-stage embedding copy is byte-identical to
        the first-stage embedding because BOTH are loaded from the same checkpoint tensor
        (``embed.weight``) and never change, so no grad-tie/all-reduce is needed
        (``finalize_model_grads`` word-embedding all-reduce is itself a no-op on the
        frozen grads).  We therefore replicate only the base method's safe attribute
        tagging and skip the tie.  (PP>1 correctness rests on the frozen + same-checkpoint
        invariant; a multi-rank smoke is the remaining validation.)
        """
        if getattr(self, "_build_embedding", False) and hasattr(self, "embedding"):
            self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        if self.post_process and self.output_layer.weight is not None:
            self.output_layer.weight.is_embedding_or_output_parameter = True

    def set_input_tensor(self, input_tensor):
        """PP plumbing. At PP=1 this is never used (pre_process and post_process True)."""
        if isinstance(input_tensor, (list, tuple)):
            assert len(input_tensor) == 1
            input_tensor = input_tensor[0]
        self.input_tensor = input_tensor

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Return local state-dict keys with global layer checkpoint keys.

        ``nn.ModuleList`` names local layers as ``layers.0``, ``layers.1``, ... even
        when this process only builds a PP slice such as global layer 42. Megatron's
        distributed checkpointing expects the Python dict keys to stay local so the
        loaded tensors can be applied to this module, while ``ShardedTensor.key``
        records the global checkpoint namespace. This mirrors
        ``TransformerBlock.sharded_state_dict`` for non-homogeneous layers.
        """
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        for local_idx, global_idx in enumerate(self.layer_ids):
            local_expert_prefix = f"{prefix}layers.{local_idx}.mlp.experts."
            expert_state = self.layers[local_idx].mlp.experts.sharded_state_dict(
                local_expert_prefix, sharded_offsets, metadata
            )
            if local_idx == global_idx:
                state.update(expert_state)
                continue
            local_prefix = f"{prefix}layers.{local_idx}."
            global_prefix = f"{prefix}layers.{global_idx}."
            layer_state = {k: v for k, v in state.items() if k.startswith(local_prefix)}
            replace_prefix_for_sharding(layer_state, local_prefix, global_prefix)
            replace_prefix_for_sharding(
                expert_state,
                local_expert_prefix,
                f"{prefix}layers.{global_idx}.mlp.experts.",
            )
            state.update(expert_state)

        # MTP head experts (last PP stage): the same EP-sharding treatment as the main
        # layers.  ``V4GroupedExperts`` is a plain ``nn.Module``, so the default
        # ``super().sharded_state_dict`` recursion does NOT invoke its custom
        # (EP-aware) ``sharded_state_dict``; call it explicitly and overwrite the plain
        # entries.  No layer-id remap (the MTP head has a fixed ``mtp.*`` namespace).
        # Post-fp8-quantization this returns {} (adapter-only saves), matching the main
        # layers.  Only needed for EP>1 torch_dist conversion/load.
        if self.mtp is not None:
            mtp_expert_prefix = f"{prefix}mtp.transformer_layer.mlp.experts."
            mtp_expert_state = self.mtp.transformer_layer.mlp.experts.sharded_state_dict(
                mtp_expert_prefix, sharded_offsets, metadata
            )
            state.update(mtp_expert_state)

        # PP>1 MTP: the last-stage embedding is a DUPLICATE of the first-stage embedding
        # (both frozen, both loaded from the same ``embed.weight``).  Megatron's default
        # tags every ``embedding.word_embeddings.weight`` shard as the MAIN replica
        # ``(0, 0, dp)``; two main copies of the same key make dist-checkpoint validation
        # reject overlapping main shards.  Retag the duplicate (non-first-stage) copy as a
        # non-main replica, exactly like Megatron's MTP embedding tie
        # (``tie_word_embeddings_state_dict``).  At PP=1 ``pre_process`` is True so the
        # embedding stays the legitimate main copy and this is skipped.
        if self._build_embedding and not self.pre_process:
            emb_key = f"{prefix}embedding.word_embeddings.weight"
            if emb_key in state and torch.distributed.is_initialized():
                from megatron.core import parallel_state
                from megatron.core.transformer.multi_token_prediction import tie_word_embeddings_state_dict

                tie_word_embeddings_state_dict(
                    state,
                    self.embedding.word_embeddings.weight,
                    emb_key,
                    self.pg_collection.tp,
                    parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
        return state

    def restore_fp32_modules(self):
        """Force the fp32-kept modules back to fp32 after any whole-model low-precision
        cast (``.bfloat16()`` / ``.half()`` / slime's ``Float16Module``).

        Two of these mirror HF ``_keep_in_fp32_modules_strict``:
          * the mHC sites ``attn_hc`` / ``ffn_hc`` — the B1 (mHC) kernel REQUIRES fp32
            fn/base/scale (Sinkhorn projection runs in float);
          * the top-k router ``e_score_correction_bias`` — it is ADDED to the scores
            before the top-k argmax, so a bf16 bias rounds and flips borderline experts.
        ``hc_head`` is NOT in HF's strict list (``DeepseekV4HyperHead`` upcasts to float
        INTERNALLY); keeping its params fp32 here is just a local storage choice that
        matches the mHC sites (the hc-stream collapse is the same B1-style math) — not a
        strict-parity requirement.  The weighted RMSNorms are deliberately left bf16
        (see M_IMPL_NOTES "RMSNorm fp32 tradeoff" — quantified, accepted).  Idempotent /
        safe to re-call after a checkpoint load.
        """
        for layer in self.layers:
            layer.attn_hc.float()
            layer.ffn_hc.float()
            # the learned top-k router's correction bias (hash router has none).
            gate = layer.mlp.gate
            if hasattr(gate, "e_score_correction_bias"):
                gate.e_score_correction_bias.data = gate.e_score_correction_bias.data.float()
        if self.post_process:
            self.hc_head.float()  # local choice (see docstring), not HF-strict
        # The MTP head has the same fp32-kept sites: its transformer layer's two mHC
        # sites, its top-k router correction bias, and its own hc_head collapse.
        if self.mtp is not None:
            self.mtp.transformer_layer.attn_hc.float()
            self.mtp.transformer_layer.ffn_hc.float()
            mtp_gate = self.mtp.transformer_layer.mlp.gate
            if hasattr(mtp_gate, "e_score_correction_bias"):
                mtp_gate.e_score_correction_bias.data = mtp_gate.e_score_correction_bias.data.float()
            self.mtp.hc_head.float()
        return self

    def bfloat16(self):
        """Cast to bf16, then restore the fp32-kept modules (mHC + correction bias)."""
        super().bfloat16()
        return self.restore_fp32_modules()

    def half(self):
        """Cast to fp16, then restore the fp32-kept modules (mHC + correction bias)."""
        super().half()
        return self.restore_fp32_modules()

    def _embed(self, input_ids, position_ids):
        # LanguageModelEmbedding returns [S, B, H] (sequence-major); V4 modules work in
        # [B, S, H], so transpose back.  (At TP=1 no SP scatter happens.)
        emb = self.embedding(input_ids=input_ids, position_ids=position_ids)  # [S,B,H]
        return emb.transpose(0, 1).contiguous()  # [B,S,H]

    def _position_embeddings(self, x, position_ids, *, haloed_position_ids=None):
        emb = {
            "main": self.rotary_emb(x, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(x, position_ids=position_ids, layer_type="compress"),
        }
        # CP2: cos/sin for the halo'd raw-KV axis [global_start-halo, global_start+l_local).
        # Built once per stage (halo width + global_start are per-rank constants); the
        # attention module picks the "*_haloed" entry for its k_raw RoPE on rank>0.  x is
        # used only for dtype/device, so the longer haloed length comes from position_ids.
        if haloed_position_ids is not None:
            emb["main_haloed"] = self.rotary_emb(x, position_ids=haloed_position_ids, layer_type="main")
            emb["compress_haloed"] = self.rotary_emb(x, position_ids=haloed_position_ids, layer_type="compress")
        return emb

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        *,
        labels=None,
        packed_seq_params=None,
        loss_mask=None,
        mtp_kwargs=None,
        **kwargs,
    ):
        """GPTModel-compatible forward.

        slime always passes position_ids=None, attention_mask=None, labels=None.  V4
        derives causality structurally inside A1 and does not consume attention_mask.
        Returns fp32 logits [B,S,V] when labels is None (slime computes the loss
        externally); returns the [B,S,hc,H] stream when ``not post_process`` (PP
        intermediate stage — unused at PP=1).

        PACKED THD IS NOT SUPPORTED YET (hard-fail).  Real slime SFT packs multiple
        documents into a single [1,T] row with ``packed_seq_params.cu_seqlens``.  This
        forward synthesizes a GLOBAL ``position_ids = 0..T-1`` and A1/compressor apply
        ONE sliding-window/causal mask across the whole row, so attention + compression
        windows would LEAK across packed-document boundaries (wrong training signal).
        A correct THD path needs: per-document position_ids reset from cu_seqlens, A1 +
        compressor masks that respect cu_seqlens (no cross-doc attention), and a
        per-document compression-window reset.  Until that exists we hard-fail rather
        than silently train wrong; the SFT sanity must use non-packed (bshd) [B,S] data
        with one document per row (see contract §9 / M_IMPL_NOTES "deferred").
        """
        # TODO(THD): implement per-document position_ids + A1/compressor masking +
        # compression-window reset from packed_seq_params.cu_seqlens, then lift this.
        # raise (NOT assert): assert is stripped under `python -O`, which would let a
        # packed-data run silently train wrong; this must be an unconditional hard-fail.
        if packed_seq_params is not None:
            raise ValueError(
                "V4LanguageModel does not support packed THD (packed_seq_params) yet: A1 + "
                "the compressors would leak attention/compression windows across packed "
                "document boundaries and the global position_ids would be wrong. Use "
                "non-packed (bshd) [B,S] data (one document per row) for the SFT sanity; "
                "the per-document THD path is a documented deferred item."
            )
        B, S = input_ids.shape
        cp_info = get_cp_info()
        haloed_position_ids = None
        if cp_info.enabled and position_ids is not None:
            # Under CP the model owns synthesizing GLOBAL positions for its contiguous
            # shard (slime always passes position_ids=None); an externally supplied
            # position_ids would be inconsistent with the halo'd raw-KV axis this stage
            # builds, so hard-fail rather than silently mix local/global frames.
            raise ValueError(
                "V4LanguageModel context parallelism requires position_ids=None so the "
                "model can synthesize global positions for its contiguous shard; got an "
                "explicit position_ids."
            )
        if position_ids is None:
            if cp_info.enabled:
                # CP2: this rank owns the contiguous slice [rank*l_local, (rank+1)*l_local),
                # so synthesize GLOBAL positions -- RoPE and the compressed causal threshold
                # ((w+1)*m <= qpos+1) must see absolute positions.  The halo'd raw-KV axis
                # also needs cos/sin down to global_start-halo (rank>0) for the exchanged
                # boundary rows; build those "*_haloed" once per stage.
                assert_local_len_aligned(S, cp_info)
                global_start = cp_info.global_start(S)
                position_ids = (global_start + torch.arange(S, device=input_ids.device)).unsqueeze(0).expand(B, -1)
                if cp_info.rank > 0:
                    haloed_position_ids = (
                        (global_start - CP_HALO + torch.arange(CP_HALO + S, device=input_ids.device))
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
            else:
                position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        if self.pre_process:
            inputs_embeds = self._embed(input_ids, position_ids)
            position_embeddings = self._position_embeddings(
                inputs_embeds, position_ids, haloed_position_ids=haloed_position_ids
            )
            # [B, S, hc_mult, hidden] parallel-stream stack (HF:1292).
            hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.hf_config.hc_mult, -1).contiguous()
        else:
            # PP intermediate stage: the incoming [B,S,hc,H] stream stack.
            hidden_states = self.input_tensor
            if hidden_states is None:
                raise ValueError("V4LanguageModel PP stage received no input_tensor")
            rotary_input = hidden_states[..., 0, :] if hidden_states.ndim == 4 else hidden_states
            position_embeddings = self._position_embeddings(
                rotary_input, position_ids, haloed_position_ids=haloed_position_ids
            )

        # Activation checkpointing: recompute decoder-layer segments in the
        # backward instead of storing their internal activations.
        # ``config.recompute_num_layers`` controls the number of consecutive layers
        # in each segment (Megatron uniform semantics). v4_model_provider derives
        # this directly from Megatron's --recompute-granularity=full. Without it,
        # 1F1B keeps pp_size
        # microbatches of full-layer activations in flight and the train backward
        # OOMs at real scale (stage-0 OOM, ctx 8192 x 32 microbatches, 2026-07-05).
        # Compatible with rollout routing replay BY DESIGN: the recompute pass
        # runs under ROUTING_REPLAY_STAGE=replay_backward, which pops from the
        # separate backward_index (slime/utils/routing_replay.py). Only active
        # when grads are enabled, so the log-prob forward_only path is unchanged.
        use_act_ckpt = (
            getattr(self.config, "recompute_granularity", None) == "full" and self.training and torch.is_grad_enabled()
        )
        # use_reentrant: at ctx-16k the stage backward once accumulated
        # ~4.4GB/layer of held recompute transients -> OOM under
        # use_reentrant=False (pytorch#147449 class); switching to the reentrant
        # variant (nested backward per segment, immediate free) fixed the real
        # system. NOTE 2026-07-09: an isolated single-process repro
        # (scripts/dsv4/diagnostics/parity/test_act_ckpt_reentrant_memory.py) shows BOTH modes free
        # per-segment on torch 2.11 with any kernel impl (ours/torch/official
        # TileKernels) — the production accumulation needs full-stack
        # ingredients (1F1B in-flight microbatches / Megatron DDP main-grad
        # buffers), so reentrant is the sole validated production behavior.
        if use_act_ckpt and self.pre_process and not hidden_states.requires_grad:
            # Reentrant checkpoint requires >=1 input to require grad, else it does
            # NOT attach to autograd and the backward recompute is silently skipped
            # -> zero LoRA gradients on the FIRST PP stage (its input comes from the
            # FROZEN embeddings, requires_grad=False). Force it so the stage-0 LoRA
            # layers train + routing-replay backward pops fire (codex review
            # 2026-07-05; mirrors Megatron Bridge's PEFT recompute-input patch). The
            # gradient w.r.t. this input is computed then discarded (embeddings frozen).
            hidden_states.requires_grad_(True)
        recompute_method = getattr(self.config, "recompute_method", None)
        if use_act_ckpt and recompute_method not in (None, "uniform", "block"):
            raise ValueError(
                "V4 full activation recompute supports recompute_method 'uniform' or "
                f"'block'; got {recompute_method!r}"
            )
        recompute_num_layers = getattr(self.config, "recompute_num_layers", None)
        if recompute_num_layers is None:
            recompute_num_layers = 1
        hidden_states = _forward_v4_decoder_layers(
            self.layers,
            hidden_states,
            position_embeddings,
            input_ids,
            use_act_ckpt=use_act_ckpt,
            recompute_num_layers=recompute_num_layers,
            recompute_method=recompute_method or "uniform",
            use_reentrant=True,
        )

        if not self.post_process:
            return hidden_states  # the [B,S,hc,H] stream stack (PP, deferred)

        # The pre-collapse hc stream [B,S,hc,H] is what the MTP head consumes (sglang
        # NextN's ``spec_info.hidden_states`` == the main model's ``pre_hc_head``).
        hc_stream = hidden_states

        # Collapse the hc streams (hc_head) then final norm (HF:1309).
        main_hidden = self.norm(self.hc_head(hc_stream))  # [B,S,H]

        # MTP training loss (stock slime ``enable_mtp_training`` path: it passes
        # ``mtp_kwargs={"mtp_labels": tokens}``).  Compute + log the MTP loss and seed
        # its backward via MTPLossAutoScaler on ``main_hidden`` — so the main output
        # logits returned below carry the MTP backward, and slime's external main-loss
        # backward triggers it (the scheduler already set the MTP loss scale).  No-op
        # unless the MTP head is built (gated) AND mtp_labels are supplied.
        mtp_labels = (mtp_kwargs or {}).get("mtp_labels")
        if self.mtp is not None and mtp_labels is not None:
            main_hidden = self._apply_mtp_training_loss(
                main_hidden=main_hidden,
                hc_stream=hc_stream,
                input_ids=input_ids,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                mtp_labels=mtp_labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
            )

        logits = self._output_logits(main_hidden)  # [B,S,V] fp32

        if labels is None:
            return logits  # matches GPTModel's labels-None return contract ([B,S,V])

        # labels path (not used by slime, which computes loss externally): [b,s] CE.
        # compute_language_model_loss wants [S,B,V] logits.
        loss = self.compute_language_model_loss(labels, logits.transpose(0, 1).contiguous())
        return loss

    def _output_logits(self, hidden_bsh, weight=None):
        """Shared output head: [B,S,H] -> fp32 logits [B,S,V].

        output_layer (ColumnParallelLinear) wants [S,B,H] and returns [S,B,V]; cast to
        fp32 and return [B,S,V] to match GPTModel's labels-None contract.  ``weight``
        overrides the head weight (the MTP loss uses the DETACHED shared weight).
        """
        hs_sbh = hidden_bsh.transpose(0, 1).contiguous()  # [S,B,H]
        logits, _ = self.output_layer(hs_sbh, weight=weight)  # [S,B,V]
        return logits.float().transpose(0, 1).contiguous()  # [B,S,V]

    def _apply_mtp_training_loss(
        self,
        *,
        main_hidden,
        hc_stream,
        input_ids,
        position_ids,
        position_embeddings,
        mtp_labels,
        loss_mask,
        packed_seq_params,
    ):
        """Run the MTP head + fold its loss into the tracker / backward (stock path).

        Mirrors Megatron ``GPTModel.forward`` MTP postprocess: run the MTP layer over
        the pre-collapse hc stream, compute the per-depth CE with the DETACHED shared
        output head, log to ``MTPLossLoggingHelper``, and seed the backward via
        ``MTPLossAutoScaler`` on ``main_hidden``.  Returns the wrapped ``main_hidden``.

        MEMORY: this materializes the full MTP logits ``[B,S,V]`` (like V4's main logits,
        which are also un-fused).  Megatron's ``fuse_linear_cross_entropy`` avoids that
        extra ``[B,S,V]`` for the MTP loss; V4 does not use fused CE anywhere, so at
        ctx-16k + large vocab the MTP loss adds ~one main-logits' worth of activation.
        A fused linear-CE path for the MTP head is the follow-up memory optimization.
        """
        from megatron.core import parallel_state

        from .mtp import apply_mtp_loss

        mtp_hidden, _, _ = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hc_stream,
            position_embeddings=position_embeddings,
            embedding=self._embed,
        )
        # DETACH the shared output weight: the MTP loss must not train the output head
        # (Megatron detaches ``mtp_output_weight``); it trains the MTP head + main model.
        detached_weight = self.output_layer.weight.detach()
        avg_group = (
            parallel_state.get_data_parallel_group(with_context_parallel=True)
            if torch.distributed.is_initialized()
            else None
        )
        return apply_mtp_loss(
            mtp_hidden=mtp_hidden,
            main_hidden=main_hidden,
            mtp_labels=mtp_labels,
            loss_mask=loss_mask,
            output_logits_fn=lambda h: self._output_logits(h, weight=detached_weight),
            ce_fn=lambda labels, logits_bsv: self.compute_language_model_loss(
                labels, logits_bsv.transpose(0, 1).contiguous()
            ),
            config=self.config,
            cp_group=getattr(self.pg_collection, "cp", None),
            packed_seq_params=packed_seq_params,
            training=self.training,
            avg_group=avg_group,
        )


__all__ = [
    "V4LanguageModel",
    "V4DecoderLayer",
    "V4MoELayer",
    "V4TopKRouter",
    "V4HashRouter",
    "V4GroupedExperts",
    "V4SharedExpertMLP",
]
