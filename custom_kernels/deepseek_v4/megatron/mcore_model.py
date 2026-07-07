"""V4-Flash as a Megatron-core ``LanguageModule`` (M-impl).

This is the mcore model the orchestrator gate asks for: a custom
``LanguageModule`` subclass that exposes the GPTModel-compatible surface slime's
train loop + ``loss.py`` drive, while reusing the M1-parity-validated V4 compute
(the three tilelang kernels A1/B2/B1 + the torch seams) for everything outside
the MoE, and a custom V4 MoE sub-layer that threads ``input_ids`` to the V4
routers and applies the exact V4 clamp+SwiGLU expert activation.

Design contract: ``handoffs/deepseek-v4/v4_megatron_sharding_contract.md``.
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

# mHC mixing oracle: detach the Sinkhorn-derived `comb`/`post` in the residual
# combine so the backward does not propagate through the 20-iteration Sinkhorn VJP
# (DeepSeek-V4 RL stability recipe: freeze Sinkhorn in mHC as a mixing oracle).
# On by default; set V4_MHC_MIXING_ORACLE=0 to restore full backprop.
_MHC_MIXING_ORACLE = os.environ.get("V4_MHC_MIXING_ORACLE", "1") == "1"

# SwiGLU-limit clamp backward. torch.clamp has ZERO gradient outside [min,max],
# so on outlier activations (|gate/up| > swiglu_limit) the gradient flowing back
# THROUGH the frozen expert to the LoRA is killed. The DeepSeek-V4 paper does not
# specify the swiglu-limit clamp's backward (it uses QK-Clip nowhere, and FP4-QAT
# elsewhere), so we default to the straight-through estimator: forward clamps,
# backward is identity. Set V4_CLAMP_STE=0 for the true (grad-zeroing) clamp.
_CLAMP_STE = os.environ.get("V4_CLAMP_STE", "1") == "1"


def _clamp(x, min=None, max=None):
    if _CLAMP_STE:
        # STE, bit-exact forward: clamp(x).detach() carries the EXACT clamped value
        # (no bf16 add/sub on it), and (x - x.detach()) is exactly 0 in value but
        # has gradient 1 -> forward == torch.clamp, backward == identity.
        return x.clamp(min=min, max=max).detach() + (x - x.detach())
    return x.clamp(min=min, max=max)


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
from .decoder import V4HyperConnection
from .rope import DeepseekV4RotaryEmbedding, V4RMSNorm


def _routing_replay_enabled():
    return os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1"


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

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts  # alias -> n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)
        if _routing_replay_enabled():
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


class V4HashRouter(nn.Module):
    """Hash router: frozen ``tid2eid[input_ids]`` selection, learned-gate weights.
    Faithful to ``DeepseekV4HashRouter`` (HF:1040).  Needs ``input_ids`` threaded in
    (sharding contract §risks 2) — the single biggest deviation from stock MoELayer.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer(
            "tid2eid",
            torch.zeros(config.vocab_size, self.top_k, dtype=torch.long),
            persistent=True,
        )

    def forward(self, hidden_states, input_ids):
        assert input_ids is not None, "V4HashRouter requires input_ids"
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        # Match sglang exactly: renorm with a bare sum (no +1e-20 guard). Scores are
        # positive (sigmoid/sqrtsoftplus) so the denominator is >0; sglang omits the
        # guard (moe/topk.py:643 `topk_weights / topk_weights.sum(...)`).
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights * self.routed_scaling_factor, indices


# To MATCH sglang's fp8 rollout (minimize train_rollout_logprob_diff) the ue8m0
# scale format must be split into WEIGHT vs ACTIVATION — they differ:
#  * WEIGHTS: the checkpoint STORES ue8m0 (power-of-2) block scales, so requantizing
#    the bf16 weights needs use_ue8m0=True to bit-recover the stored fp8 (verified:
#    ue8m0=True -> 0.00% byte diff vs the checkpoint; ue8m0=False -> 88% diff).
#  * ACTIVATIONS: sglang's runtime activation quant is HARDWARE-gated, not checkpoint-
#    gated: DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL (sglang configurer.py). On H20
#    (Hopper, sm90 < sm100) it is FALSE => LINEAR activation scales (verified: sglang
#    activation quant matches deep_gemm ue8m0=False at 0.0005 mismatch, ue8m0=True at
#    0.91). Conflating the two (single flag) makes EITHER weights OR activations wrong
#    -> fp8 loses to bf16; splitting them is the fix. (codex kernel review 2026-07-06)
_V4_FP8_WEIGHT_UE8M0 = os.environ.get("V4_FP8_WEIGHT_UE8M0", "1") == "1"


def _default_act_ue8m0():
    env = os.environ.get("V4_FP8_ACT_UE8M0")
    if env is not None:
        return env == "1"
    try:  # Blackwell (sm100+) uses ue8m0 activation scales; Hopper/H20 uses linear
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
    except Exception:
        return False


_V4_FP8_ACT_UE8M0 = _default_act_ue8m0()


def _dequant_block_fp8(w_fp8, scale, blk):
    """[O,I] float8 + [Ob,Ib] scale -> [O,I] bf16 (block-broadcast, no full-scale
    materialization). A transient; callers must NOT retain it."""
    O, I = w_fp8.shape
    # multiply in fp32 (scale stays fp32), cast once to bf16 (codex review item 3)
    w = w_fp8.to(torch.float32).view(O // blk, blk, I // blk, blk)
    return (w * scale[:, None, :, None]).reshape(O, I).to(torch.bfloat16)


class _FrozenFp8ExpertLinear(torch.autograd.Function):
    """y = x @ W^T for a frozen fp8-stored expert weight, WITHOUT retaining the
    bf16 weight (the ctx saves only the resident fp8 buffers; the bf16 weight is a
    transient re-derived in forward and again in backward). This is the fix for the
    ctx-16k activation-recompute OOM. No grad_weight (frozen)."""

    @staticmethod
    def forward(ctx, x, w_fp8, scale, blk):
        ctx.save_for_backward(w_fp8, scale)  # fp8 buffers — already resident, no new memory
        ctx.blk = blk
        w = _dequant_block_fp8(w_fp8, scale, blk)  # transient, NOT saved
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, grad_out):
        w_fp8, scale = ctx.saved_tensors
        w = _dequant_block_fp8(w_fp8, scale, ctx.blk)  # transient re-dequant
        grad_x = grad_out @ w  # grad_x = grad_y @ W  (y = x @ W^T)
        return grad_x, None, None, None


def _pad_to_alignment(hidden, counts, align):
    """Scatter contiguous per-expert segments into `align`-multiple padded slots
    (deep_gemm's contiguous grouped GEMM needs each group padded to `align`).
    Returns (hidden_padded [Tp,H], m_indices [Tp], gather [T]) where
    gather[real_row] = padded_row, to un-pad the [Tp,O] output back to [T,O]."""
    pc = [((c + align - 1) // align) * align for c in counts]
    Tp = sum(pc)
    T = hidden.shape[0]
    hp = hidden.new_zeros(Tp, hidden.shape[1])
    m_idx = torch.empty(Tp, device=hidden.device, dtype=torch.int32)
    gather = torch.empty(T, device=hidden.device, dtype=torch.long)
    s = d = 0
    for e, (c, p) in enumerate(zip(counts, pc)):
        if c:
            hp[d : d + c] = hidden[s : s + c]
            gather[s : s + c] = torch.arange(d, d + c, device=hidden.device)
        m_idx[d : d + p] = e
        s += c
        d += p
    return hp, m_idx, gather


class _Fp8GroupedExpertMatmul(torch.autograd.Function):
    """Grouped y = x @ W^T over dispatcher-permuted experts, computed in FP8 with
    deep_gemm — the SAME kernel sglang uses to serve these experts at rollout, so
    the train forward matches the rollout forward (minimizes
    train_rollout_logprob_abs_diff). Forward: pad each expert to 128 rows,
    per-token fp8-quantize the activations (+ TMA-align the scale), grouped fp8
    GEMM, un-pad. Backward grad_x: per-expert bf16 (dequant the frozen fp8 weight
    transiently) — training-only, no sglang counterpart, and bf16 avoids
    fp8-gradient noise. No grad_weight (frozen). No bf16 weight is ever retained."""

    @staticmethod
    def forward(ctx, hidden, w_fp8, w_scale, tokens_per_expert, blk):
        from deep_gemm import (
            get_m_alignment_for_contiguous_layout,
            get_mn_major_tma_aligned_tensor,
            m_grouped_fp8_gemm_nt_contiguous,
            per_token_cast_to_fp8,
        )

        counts = tokens_per_expert.tolist()
        align = get_m_alignment_for_contiguous_layout()
        hp, m_idx, gather = _pad_to_alignment(hidden, counts, align)
        a_fp8, a_scale = per_token_cast_to_fp8(hp, use_ue8m0=_V4_FP8_ACT_UE8M0)
        a_scale = get_mn_major_tma_aligned_tensor(a_scale)
        out_pad = torch.empty(hp.shape[0], w_fp8.shape[1], dtype=torch.bfloat16, device=hidden.device)
        m_grouped_fp8_gemm_nt_contiguous((a_fp8, a_scale), (w_fp8, w_scale), out_pad, m_idx)
        out = out_pad.index_select(0, gather)  # un-pad -> [T, O]
        ctx.save_for_backward(w_fp8, w_scale, tokens_per_expert)
        ctx.blk = blk
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w_fp8, w_scale, tokens_per_expert = ctx.saved_tensors
        blk = ctx.blk
        grad_x = torch.empty(grad_out.shape[0], w_fp8.shape[2], dtype=grad_out.dtype, device=grad_out.device)
        off = 0
        for e, c in enumerate(tokens_per_expert.tolist()):
            if c:
                w = _dequant_block_fp8(w_fp8[e], w_scale[e], blk)  # [O,I] bf16 transient
                grad_x[off : off + c] = grad_out[off : off + c] @ w  # grad_y @ W -> [c, I]
            off += c
        return grad_x, None, None, None, None


class _FrozenFp8DenseLinear(torch.autograd.Function):
    """y = x @ W^T for a FROZEN fp8-stored dense weight, computed in fp8 via
    deep_gemm (per-token activation quant + fp8 matmul) — matches sglang's fp8
    serving of the shared expert / attention base, so the train forward matches
    the rollout forward. Backward grad_x is bf16 (dequant the frozen fp8 weight
    transiently; no grad_weight — frozen). No M-padding needed (dense fp8_gemm_nt
    handles arbitrary token counts)."""

    @staticmethod
    def forward(ctx, x, w_fp8, w_scale, blk):
        from deep_gemm import fp8_gemm_nt, get_mn_major_tma_aligned_tensor, per_token_cast_to_fp8

        x2d = x.reshape(-1, x.shape[-1])
        a_fp8, a_scale = per_token_cast_to_fp8(x2d, use_ue8m0=_V4_FP8_ACT_UE8M0)
        a_scale = get_mn_major_tma_aligned_tensor(a_scale)
        out = torch.empty(x2d.shape[0], w_fp8.shape[0], dtype=torch.bfloat16, device=x.device)
        fp8_gemm_nt((a_fp8, a_scale), (w_fp8, w_scale), out)
        ctx.save_for_backward(w_fp8, w_scale)
        ctx.blk = blk
        ctx.in_shape = x.shape
        return out.reshape(*x.shape[:-1], w_fp8.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        w_fp8, w_scale = ctx.saved_tensors
        w = _dequant_block_fp8(w_fp8, w_scale, ctx.blk)  # [O, I] bf16
        g2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x = (g2d @ w).reshape(*ctx.in_shape)
        return grad_x, None, None, None


def _quantize_linear_fp8(linear, blk):
    """Requantize a FROZEN nn.Linear weight to blockwise float8_e4m3fn (deep_gemm
    per_block layout), replacing the bf16 Parameter with fp8 + scale buffers.
    Idempotent; returns (w_fp8, w_scale) or None if already quantized / trainable."""
    from deep_gemm import per_block_cast_to_fp8

    if getattr(linear, "_fp8", False):
        return None
    w = linear.weight
    assert not w.requires_grad, "fp8 dense quantization is only for FROZEN linears"
    O, I = w.shape
    assert O % blk == 0 and I % blk == 0, f"dims ({O},{I}) must be multiples of {blk}"
    w_fp8, w_scale = per_block_cast_to_fp8(w.data.float(), use_ue8m0=_V4_FP8_WEIGHT_UE8M0)
    del linear.weight
    linear.register_buffer("weight_fp8", w_fp8, persistent=False)
    linear.register_buffer("weight_scale", w_scale, persistent=False)
    linear._fp8 = True


def _fp8_lora_adapter_forward(self, x):
    """Replacement forward for a megatron-bridge LoRA adapter (nn.Linear subclass)
    whose FROZEN base is fp8: base matmul in fp8 (matches sglang's fp8 serving of
    wq_a/wq_b/wkv/wo_b), LoRA delta in bf16 (bridge keeps LoRA in original precision).
    Mirrors TELinearAdapter.forward exactly except the base uses the fp8 kernel."""
    res = _FrozenFp8DenseLinear.apply(x, self.weight_fp8, self.weight_scale, self._FP8_BLOCK)
    if self.bias is not None:
        res = res + self.bias
    if not getattr(self, "_adapter_enabled", True):
        return res
    if getattr(self, "dropout_position", "post") == "pre":
        x = self.dropout(x)
    lora_res = self.linear_out(self.linear_in(x)) * self.scale
    if getattr(self, "dropout_position", "post") == "post":
        lora_res = self.dropout(lora_res)
    return res + lora_res


def quantize_lora_adapter_fp8(adapter, blk=128):
    """fp8 the FROZEN base weight of a LoRA-wrapped attention projection and patch
    its forward so the base runs in fp8 (weight ue8m0 + activation linear on H20) with
    the bf16 LoRA delta on top. Only for adapters whose base is fp8 in the checkpoint
    (self_attn wq_a/wq_b/wkv/wo_b). Idempotent."""
    import types

    if getattr(adapter, "_fp8", False):
        return
    _quantize_linear_fp8(adapter, blk)  # base weight -> fp8 buffers (asserts frozen)
    adapter._FP8_BLOCK = blk
    adapter.forward = types.MethodType(_fp8_lora_adapter_forward, adapter)


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

    def __init__(self, config, *, expert_model_parallel_size=1, expert_model_parallel_rank=0):
        super().__init__()
        self.num_global_experts = config.num_local_experts
        self.expert_model_parallel_size = int(expert_model_parallel_size)
        self.expert_model_parallel_rank = int(expert_model_parallel_rank)
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

    # --- FP8 frozen-expert storage (V4_FP8_FROZEN_EXPERTS=1) -------------------
    # The experts are FROZEN (LoRA is attention/compressor only) and the source
    # checkpoint is FP8, yet params_dtype=bf16 doubles their resident footprint —
    # the bulk of the ~63GB/GPU rest state. Requantize them to float8_e4m3fn +
    # per-expert scale AFTER load, freeing the bf16 copies (~half the expert
    # memory) so ctx-16k fits. Dequant per-expert in the forward loop (transient,
    # one expert at a time). This also matches sglang's fp8 rollout serving,
    # reducing the train/rollout expert-precision mismatch rather than adding one.
    _FP8_E4M3_MAX = 448.0
    _FP8_BLOCK = 128  # matches the source checkpoint's weight_block_size [128,128]

    def quantize_frozen_experts_fp8(self):
        """bf16 -> float8_e4m3fn with BLOCKWISE [128,128] scales, in place, idempotent.

        Blockwise (not per-tensor) matches the source/rollout FP8 format
        (config.json weight_block_size [128,128]), so training experts stay close
        to what sglang serves — reducing, not adding, train/rollout mismatch
        (codex review 2026-07-05, finding 3). Must be called AFTER the checkpoint
        load (weights populated). Quantizes one expert at a time so the transient
        fp32 copy is one-expert-sized, not whole-tensor (finding 4). Guarded to
        frozen params only (finding 2). No-op if already quantized."""
        if getattr(self, "_experts_fp8", False):
            return
        blk = self._FP8_BLOCK
        for name in ("gate_up_proj", "down_proj"):
            param = getattr(self, name)
            assert not param.requires_grad, (
                f"{name} is trainable; fp8 quantization is only for FROZEN experts "
                "(would strand optimizer/DDP references)"
            )
            w = param.data  # [E, O, I] bf16
            E, O, I = w.shape
            assert O % blk == 0 and I % blk == 0, f"{name} dims ({O},{I}) must be multiples of {blk} for blockwise fp8"
            Ob, Ib = O // blk, I // blk
            w_fp8 = torch.empty(E, O, I, dtype=torch.float8_e4m3fn, device=w.device)
            scale = torch.empty(E, Ob, Ib, dtype=torch.float32, device=w.device)
            for e in range(E):  # bound the fp32 transient to one expert
                we = w[e].float().reshape(Ob, blk, Ib, blk)  # ~one-expert fp32
                amax = we.abs().amax(dim=(1, 3)).clamp_(min=1e-8)  # [Ob, Ib]
                se = amax / self._FP8_E4M3_MAX
                w_fp8[e] = (
                    (we / se[:, None, :, None])
                    .clamp_(-self._FP8_E4M3_MAX, self._FP8_E4M3_MAX)
                    .to(torch.float8_e4m3fn)
                    .reshape(O, I)
                )
                scale[e] = se
                del we, amax, se
            delattr(self, name)  # drop the bf16 Parameter
            self.register_buffer(f"{name}_fp8", w_fp8, persistent=False)
            self.register_buffer(f"{name}_scale", scale, persistent=False)
            del w
        self._experts_fp8 = True

    def _expert_matmul(self, x, name, idx):
        """y = x @ W_idx^T for a FROZEN expert, without retaining the bf16 weight.

        The 16k OOM root cause: a plain `F.linear(x, dequant(W_fp8))` saves the
        dequantized bf16 weight for autograd, and under whole-layer activation
        recompute EVERY active expert's bf16 weight is held through the layer
        backward (~32 experts, the ~2GB overflow). This custom Function saves only
        the resident fp8 buffers (no extra memory) and re-dequantizes a single
        expert transiently inside forward AND backward — so no bf16 weight is ever
        retained. Weights stay fp8-precision (matches sglang's fp8 rollout); the
        matmul is bf16 (no fp8-gradient noise). For non-quantized params, plain
        F.linear."""
        if not getattr(self, "_experts_fp8", False):
            return F.linear(x, getattr(self, name)[idx])
        return _FrozenFp8ExpertLinear.apply(
            x,
            getattr(self, f"{name}_fp8")[idx],
            getattr(self, f"{name}_scale")[idx],
            self._FP8_BLOCK,
        )

    def _expert_weight(self, name, idx):
        """Return expert `idx`'s [O, I] weight in bf16, dequantizing on the fly
        when the frozen experts are stored in blockwise fp8.

        MEMORY-CRITICAL: this runs inside the activation-checkpoint recompute, and
        with whole-layer recompute EVERY active expert's dequantized weight is
        retained until the layer's backward completes (~32 experts/layer). Dequant
        entirely in bf16 (not fp32) to keep that retained footprint ~half — the
        fp32 path OOM'd the ctx-16k backward by ~1.7GB (2026-07-05). The bf16
        multiply differs from an fp32 multiply by ~1e-3 (codex finding 5), which is
        negligible for frozen fp8-precision inference weights. Block-broadcast the
        scale via a reshaped view (no full [O,I] scale materialization)."""
        if not getattr(self, "_experts_fp8", False):
            return getattr(self, name)[idx]
        blk = self._FP8_BLOCK
        w = getattr(self, f"{name}_fp8")[idx].to(torch.bfloat16)  # [O, I]
        O, I = w.shape
        s = getattr(self, f"{name}_scale")[idx].to(torch.bfloat16)  # [Ob, Ib]
        wb = w.view(O // blk, blk, I // blk, blk) * s[:, None, :, None]
        return wb.reshape(O, I)

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
        # Count validation up front so BOTH the fp8 fast path and the bf16 loop are
        # guarded (codex review 2026-07-05, items 1/5): reject negatives, require the
        # counts to sum to the token rows, and short-circuit an empty batch (the fp8
        # path would otherwise call deep_gemm on 0 rows).
        if (tokens_per_expert < 0).any():
            raise ValueError(f"tokens_per_expert has a negative entry: {tokens_per_expert.tolist()}")
        total = int(tokens_per_expert.sum().item())
        if total != hidden_states.shape[0]:
            raise ValueError(f"tokens_per_expert sums to {total}, but hidden_states has {hidden_states.shape[0]} rows")
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        # FP8 grouped-GEMM path (V4_FP8_EXPERT_GEMM, default on when experts are
        # fp8-stored + on GPU): compute the two expert matmuls in fp8 via deep_gemm
        # — the SAME kernel sglang serves these experts with — so the train forward
        # matches the rollout forward (minimizes train_rollout_logprob_abs_diff),
        # and no bf16 weight is materialized (fits ctx-16k). Falls back to the
        # per-expert bf16 loop below on CPU / non-fp8 (tests, EP=1 smokes).
        if (
            getattr(self, "_experts_fp8", False)
            and hidden_states.is_cuda
            and os.environ.get("V4_FP8_EXPERT_GEMM", "1") == "1"
        ):
            gate_up = _Fp8GroupedExpertMatmul.apply(
                hidden_states, self.gate_up_proj_fp8, self.gate_up_proj_scale, tokens_per_expert, self._FP8_BLOCK
            )
            current = self._apply_gate(gate_up)
            out = _Fp8GroupedExpertMatmul.apply(
                current, self.down_proj_fp8, self.down_proj_scale, tokens_per_expert, self._FP8_BLOCK
            )
            if permuted_probs is not None:
                out = out * permuted_probs[:, None]
            return out.to(hidden_states.dtype)
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

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Shard grouped expert tensors on the global expert axis for EP conversion."""
        del metadata
        # Once the frozen experts are quantized to fp8 (post-load), the bf16
        # Parameters no longer exist. sharded_state_dict is only reached again for
        # SAVES, and V4 saves are adapter-only (trainable params) — the frozen
        # experts are never saved — so returning {} here is correct. The LOAD path
        # always runs before quantization (bf16 params present), so it is unaffected.
        if getattr(self, "_experts_fp8", False):
            return {}
        prepend_axis_num = len(sharded_offsets)
        ep_offset = (prepend_axis_num, self.expert_model_parallel_rank, self.expert_model_parallel_size)
        return {
            f"{prefix}gate_up_proj": ShardedTensor.from_rank_offsets(
                f"{prefix}gate_up_proj",
                self.gate_up_proj,
                *sharded_offsets,
                ep_offset,
                prepend_axis_num=prepend_axis_num,
            ),
            f"{prefix}down_proj": ShardedTensor.from_rank_offsets(
                f"{prefix}down_proj",
                self.down_proj,
                *sharded_offsets,
                ep_offset,
                prepend_axis_num=prepend_axis_num,
            ),
        }


class V4SharedExpertMLP(nn.Module):
    """Replicated shared expert (HF ``DeepseekV4MLP``): SiLU-gated MLP at moe width."""

    def __init__(self, config):
        super().__init__()
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
        self._FP8_BLOCK = 128

    def _act(self, gate, up):
        gate = _clamp(gate, max=self.limit)
        up = _clamp(up, min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def quantize_fp8(self):
        """fp8-quantize the 3 frozen dense linears (V4_FP8_SHARED_EXPERT=1). Runs on
        every token, fp8 on disk + served fp8 by sglang -> fp8 compute here matches
        the rollout. Idempotent. Bias (if any) stays bf16 and is added in forward."""
        if getattr(self, "_fp8", False):
            return
        for lin in (self.gate_proj, self.up_proj, self.down_proj):
            _quantize_linear_fp8(lin, self._FP8_BLOCK)
        self._fp8 = True

    def _lin(self, linear, x):
        if getattr(linear, "_fp8", False):
            y = _FrozenFp8DenseLinear.apply(x, linear.weight_fp8, linear.weight_scale, self._FP8_BLOCK)
            return y if linear.bias is None else y + linear.bias
        return linear(x)

    def forward(self, x):
        if getattr(self, "_fp8", False):
            return self._lin(self.down_proj, self._act(self._lin(self.gate_proj, x), self._lin(self.up_proj, x)))
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
    ):
        super().__init__()
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.num_global_experts = config.num_local_experts
        self.mcore_config = mcore_config
        self.gate = V4HashRouter(config) if self.is_hash else V4TopKRouter(config)
        self.experts = V4GroupedExperts(
            config,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_model_parallel_rank=expert_model_parallel_rank,
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
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = V4Attention(config, layer_idx)
        self.mlp = V4MoELayer(
            config,
            layer_idx,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_model_parallel_rank=expert_model_parallel_rank,
            mcore_config=mcore_config,
            pg_collection=pg_collection,
        )
        self.input_layernorm = V4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = V4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = V4HyperConnection(config)
        self.ffn_hc = V4HyperConnection(config)

    def forward(self, hidden_states, position_embeddings, input_ids):
        dtype = hidden_states.dtype
        # mHC "mixing oracle" (DeepSeek-V4 RL recipe): the Sinkhorn-derived mixing
        # matrix `comb` and output gate `post` are used forward-only and detached in
        # the backward, so gradients do NOT propagate through the numerically-fragile
        # 20-iteration Sinkhorn VJP (the source of the first-backward grad NaN). The
        # residual-stream gradient still flows via `comb.T @ hidden_states` (constant
        # comb) and cross-layer gradient still flows via the differentiable `collapsed`
        # sublayer input. Ref: LMSYS "DeepSeek-V4 Day-0 RL" — freeze Sinkhorn in mHC.
        post, comb, collapsed = self.attn_hc(hidden_states)
        if _MHC_MIXING_ORACLE:
            post, comb = post.detach(), comb.detach()
        attn_output = self.self_attn(self.input_layernorm(collapsed), position_embeddings)
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        if _MHC_MIXING_ORACLE:
            post, comb = post.detach(), comb.detach()
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        hidden_states = post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )
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
    ):
        super().__init__(config=config)
        self.hf_config = hf_config
        self.pre_process = pre_process
        self.post_process = post_process
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

        if self.pre_process:
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

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

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

    def _position_embeddings(self, x, position_ids):
        return {
            "main": self.rotary_emb(x, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(x, position_ids=position_ids, layer_type="compress"),
        }

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        *,
        labels=None,
        packed_seq_params=None,
        loss_mask=None,
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
        if position_ids is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        if self.pre_process:
            inputs_embeds = self._embed(input_ids, position_ids)
            position_embeddings = self._position_embeddings(inputs_embeds, position_ids)
            # [B, S, hc_mult, hidden] parallel-stream stack (HF:1292).
            hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.hf_config.hc_mult, -1).contiguous()
        else:
            # PP intermediate stage: the incoming [B,S,hc,H] stream stack.
            hidden_states = self.input_tensor
            if hidden_states is None:
                raise ValueError("V4LanguageModel PP stage received no input_tensor")
            rotary_input = hidden_states[..., 0, :] if hidden_states.ndim == 4 else hidden_states
            position_embeddings = self._position_embeddings(rotary_input, position_ids)

        # Activation checkpointing (V4_ACT_CKPT=1): recompute each decoder layer
        # in the backward instead of storing activations. Megatron's --recompute-*
        # is a no-op for this custom loop; without this, 1F1B keeps pp_size
        # microbatches of full-layer activations in flight and the train backward
        # OOMs at real scale (stage-0 OOM, ctx 8192 x 32 microbatches, 2026-07-05).
        # Compatible with rollout routing replay BY DESIGN: the recompute pass
        # runs under ROUTING_REPLAY_STAGE=replay_backward, which pops from the
        # separate backward_index (slime/utils/routing_replay.py). Only active
        # when grads are enabled, so the log-prob forward_only path is unchanged.
        use_act_ckpt = os.environ.get("V4_ACT_CKPT", "0") == "1" and self.training and torch.is_grad_enabled()
        # use_reentrant: False (default) is the modern API but has a known peak-
        # memory bug where per-segment recompute activations are NOT freed across
        # the stage backward (pytorch#147449; "does not work well with DDP") — at
        # ctx-16k this accumulates ~4.4GB/layer -> OOM (measured per-layer). The
        # reentrant variant runs a nested backward per segment and frees each
        # layer's recompute immediately. V4_ACT_CKPT_REENTRANT=1 selects it.
        _reentrant = os.environ.get("V4_ACT_CKPT_REENTRANT", "1") == "1"
        if use_act_ckpt and _reentrant and self.pre_process and not hidden_states.requires_grad:
            # Reentrant checkpoint requires >=1 input to require grad, else it does
            # NOT attach to autograd and the backward recompute is silently skipped
            # -> zero LoRA gradients on the FIRST PP stage (its input comes from the
            # FROZEN embeddings, requires_grad=False). Force it so the stage-0 LoRA
            # layers train + routing-replay backward pops fire (codex review
            # 2026-07-05; mirrors Megatron Bridge's PEFT recompute-input patch). The
            # gradient w.r.t. this input is computed then discarded (embeddings frozen).
            hidden_states.requires_grad_(True)
        for layer in self.layers:
            if use_act_ckpt:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, position_embeddings, input_ids, use_reentrant=_reentrant
                )
            else:
                hidden_states = layer(hidden_states, position_embeddings, input_ids)

        if not self.post_process:
            return hidden_states  # the [B,S,hc,H] stream stack (PP, deferred)

        # Collapse the hc streams (hc_head) then final norm (HF:1309).
        hidden_states = self.norm(self.hc_head(hidden_states))  # [B,S,H]

        # output_layer wants [S,B,H]; returns [S,B,V].
        hs_sbh = hidden_states.transpose(0, 1).contiguous()
        logits, _ = self.output_layer(hs_sbh)  # [S,B,V]
        logits = logits.float()

        if labels is None:
            # [S,B,V] -> [B,S,V] (matches GPTModel's labels-None return contract).
            return logits.transpose(0, 1).contiguous()

        # labels path (not used by slime, which computes loss externally): [b,s] CE.
        loss = self.compute_language_model_loss(labels, logits)
        return loss


__all__ = [
    "V4LanguageModel",
    "V4DecoderLayer",
    "V4MoELayer",
    "V4TopKRouter",
    "V4HashRouter",
    "V4GroupedExperts",
    "V4SharedExpertMLP",
]
