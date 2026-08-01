"""V4Indexer — the official HF ``DeepseekV4Indexer`` with DeepSeek's official
high-performance scoring kernel (DeepGEMM ``fp8_mqa_logits``) on the hot path.

Adoption per the DeepSeek-V3.2 release notes: "indexer logit kernels ... are
available in DeepGEMM". Structure/params/buffers are inherited from the HF
class (checkpoint-compatible, tracks upstream); only the cacheless training
forward is overridden to route the score computation through
``deep_gemm.fp8_mqa_logits`` when available, with the HF-exact torch math as
the fallback/reference path on non-CUDA systems or when DeepGEMM is absent.

Why fp8 scoring is safe (and desirable) in training:
- The indexer produces DISCRETE top-k indices — no gradient flows through the
  selection, and the indexer's own params are frozen in the V4 LoRA recipe, so
  a forward-only fp8 kernel loses nothing.
- The rollout engine scores its indexer in fp8 through this same kernel family
  (sglang dsa_indexer -> deep_gemm.fp8_mqa_logits), so fp8 here REDUCES
  train<->rollout selection mismatch relative to bf16/fp32 torch scoring.

Math note (exactness of the mapping): the kernel computes
``logits[t,s] = sum_h w_eff[t,h] * relu(q_fp8[t,h] . kv_fp8[s]) * kv_sf[s]``
with per-token kv scales passed alongside the fp8 kv. Because every scale is
positive and ``relu(c*x) = c*relu(x)`` for ``c > 0``, folding the query's
dequant scale and HF's ``softmax_scale``/``weights_scaling`` into
``w_eff[t,h] = w[t,h] * weights_scaling * softmax_scale * q_sf[t,h]`` is
algebraically exact — the only approximation is the fp8 rounding itself.
HF's causal future-mask (``entry >= (pos+1)//rate``) maps exactly to the
kernel's per-query ``ke`` bound with ``clean_logits=True``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer as _HFDeepseekV4Indexer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

try:  # deep_gemm ships in the serving image; the fallback keeps CPU/dev usable
    import deep_gemm as _deep_gemm

    _HAS_DEEP_GEMM = hasattr(_deep_gemm, "fp8_mqa_logits")
except Exception:  # noqa: BLE001
    _deep_gemm = None
    _HAS_DEEP_GEMM = False

_FP8_AMAX = 448.0  # float8_e4m3fn dynamic range used by the DeepSeek quant recipe


def _use_deepgemm_scoring(device: torch.device) -> bool:
    return _HAS_DEEP_GEMM and device.type == "cuda"


def _quant_e4m3_lastdim(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (last-dim) e4m3 quantization: returns (fp8 tensor, fp32 scales)
    with ``x ≈ fp8 * sf`` and ``sf = amax(|x|)/448`` (DeepSeek recipe)."""
    sf = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).float() / _FP8_AMAX
    q = (x.float() / sf).clamp(-_FP8_AMAX, _FP8_AMAX).to(torch.float8_e4m3fn)
    return q, sf.squeeze(-1)


class V4Indexer(_HFDeepseekV4Indexer):
    """Official HF indexer + official DeepGEMM scoring kernel (train side).

    Training is CACHELESS (``past_key_values`` must be None — serving runs in
    sglang, which has its own dsa indexer); the override implements only the
    cacheless branch of the HF forward, reusing every inherited submodule.
    """

    def forward(
        self, hidden_states, q_residual, position_ids, past_key_values, layer_idx, *, window_position_offset=0
    ):
        # ``window_position_offset`` (CP2 hook, default 0 == non-CP): global position of
        # compressed window 0. Threaded for checkpoint compatibility; the maintained
        # trainer path is dense over compressed attention. A real CP sparse
        # indexer also needs GLOBAL ``position_ids`` (for q-RoPE + causal_threshold) AND
        # the GLOBAL compressed axis (not just this rank's owned windows) -- deferred.
        assert past_key_values is None, (
            "V4Indexer (train side) is cacheless; serving-side indexing lives in " "sglang's dsa indexer"
        )
        batch, seq_len, _ = hidden_states.shape

        # ---- compressed indexer keys (HF cacheless branch, inherited modules) --
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate = kv[:, :usable], gate[:, :usable]
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device) * self.compress_rate + window_position_offset
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed_kv = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed_kv = chunk_kv.new_zeros((batch, 0, self.head_dim))

        # ---- queries + per-head weights (HF-exact) ----------------------------
        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)  # [B, S, H, D]
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B, S, H]

        compressed_len = compressed_kv.shape[1]
        top_k = min(self.index_topk, compressed_len)
        if compressed_len == 0:
            return torch.full((batch, seq_len, 0), -1, dtype=torch.long, device=hidden_states.device)

        causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]

        if _use_deepgemm_scoring(hidden_states.device):
            index_scores = self._deepgemm_scores(q, compressed_kv, weights, causal_threshold)
        else:
            index_scores = self._torch_scores(q, compressed_kv, weights, causal_threshold)

        top_k_indices = index_scores.topk(top_k, dim=-1).indices  # [B, S, k]
        invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
        return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

    # ------------------------------------------------------------------ scoring

    def _torch_scores(self, q, compressed_kv, weights, causal_threshold):
        """HF-exact reference scoring — op-for-op the tail of the HF forward
        (same matmul layout and scale order, so outputs are bit-identical)."""
        scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        entry_indices = torch.arange(index_scores.shape[-1], device=index_scores.device)
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
        return index_scores.masked_fill(future_mask, float("-inf"))

    @torch.no_grad()
    def _deepgemm_scores(self, q, compressed_kv, weights, causal_threshold):
        """Official DeepGEMM fp8 indexer-logit kernel; one call per batch row
        (the kernel is batchless MQA: q [S,H,D] x kv [T,D]). no_grad: the
        selection is discrete and the indexer is frozen in the V4 recipe."""
        B, S, H, D = q.shape
        T = compressed_kv.shape[1]
        out = q.new_full((B, S, T), float("-inf"), dtype=torch.float32)
        ks = torch.zeros(S, dtype=torch.int32, device=q.device)
        for b in range(B):
            q_fp8, q_sf = _quant_e4m3_lastdim(q[b])  # [S,H,D] fp8, [S,H] fp32
            kv_fp8, kv_sf = _quant_e4m3_lastdim(compressed_kv[b])  # [T,D] fp8, [T] fp32
            # fold q dequant scale + HF softmax/weights scaling into the weights
            w_eff = (weights[b] * self.softmax_scale * q_sf).float().contiguous()  # [S,H]
            ke = causal_threshold[b].clamp(min=0, max=T).to(torch.int32).contiguous()
            logits = _deep_gemm.fp8_mqa_logits(
                q_fp8.contiguous(),
                (kv_fp8.contiguous(), kv_sf.contiguous()),
                w_eff,
                ks,
                ke,
                clean_logits=True,
            )  # [S, T], -inf outside [ks, ke)
            out[b] = logits.float()
        return out


__all__ = ["V4Indexer"]
