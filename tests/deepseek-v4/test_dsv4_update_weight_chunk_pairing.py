"""Unit tests for the DeepSeek-V4 weight-sync chunking invariants.

SGLang's V4 loader fuses raw-name weight pairs at load time, so an online
weight update must never split a pair across broadcast chunks:
- compressor ``wkv``/``wgate`` (attn and indexer variants),
- attention ``wq_a``/``wkv`` (fused into ``wqkv_a``; ``weight`` and
  ``weight_scale_inv`` are independent pairs).

These invariants were previously proven only inside the multi-node full-loop
smoke; here the pure detection helpers and the chunk-flush predicate semantics
are pinned as unit tests. Also covers the V4 LoRA merge math used for
LoRA-only sync.
"""

import pytest
import torch

from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import (
    _lora_base_weight_name,
    _merge_lora_weight,
    _v4_incomplete_sglang_loader_pairs,
    _v4_sglang_compressor_pair_key,
    _v4_sglang_wqkv_a_pair_key,
)

T = torch.zeros(1)  # placeholder tensor; the helpers only read names


def test_compressor_pair_key_matches_attn_and_indexer_variants():
    assert _v4_sglang_compressor_pair_key("layers.3.attn.compressor.wkv.weight") == (
        "layers.3.attn.compressor",
        "wkv",
    )
    assert _v4_sglang_compressor_pair_key("layers.3.attn.indexer.compressor.wgate.weight") == (
        "layers.3.attn.indexer.compressor",
        "wgate",
    )
    assert _v4_sglang_compressor_pair_key("layers.3.attn.wkv.weight") is None
    assert _v4_sglang_compressor_pair_key("layers.3.mlp.compressor.wkv.weight") is None


def test_wqkv_a_pair_key_fuses_wq_a_and_wkv_per_suffix():
    key_q = _v4_sglang_wqkv_a_pair_key("layers.7.attn.wq_a.weight")
    key_kv = _v4_sglang_wqkv_a_pair_key("layers.7.attn.wkv.weight")
    assert key_q == ("layers.7.attn.wqkv_a.weight", "q")
    assert key_kv == ("layers.7.attn.wqkv_a.weight", "kv")
    # weight_scale_inv forms its own independent pair
    key_q_scale = _v4_sglang_wqkv_a_pair_key("layers.7.attn.wq_a.weight_scale_inv")
    assert key_q_scale == ("layers.7.attn.wqkv_a.weight_scale_inv", "q")
    assert _v4_sglang_wqkv_a_pair_key("layers.7.attn.compressor.wkv.weight") is None


def test_complete_pairs_report_no_incompleteness():
    buffer = [
        ("layers.0.attn.compressor.wkv.weight", T),
        ("layers.0.attn.compressor.wgate.weight", T),
        ("layers.0.attn.wq_a.weight", T),
        ("layers.0.attn.wkv.weight", T),
        ("layers.0.mlp.shared_experts.down_proj.weight", T),
    ]
    assert _v4_incomplete_sglang_loader_pairs(buffer) == {}


def test_split_pair_is_detected_as_incomplete():
    # a chunk ending right after wkv (wgate would land in the next chunk)
    buffer = [
        ("layers.0.attn.compressor.wkv.weight", T),
    ]
    incomplete = _v4_incomplete_sglang_loader_pairs(buffer)
    assert incomplete == {"layers.0.attn.compressor": {"wkv"}}

    # fused wqkv_a split across suffix sides
    buffer = [
        ("layers.1.attn.wq_a.weight", T),
        ("layers.1.attn.wq_a.weight_scale_inv", T),
        ("layers.1.attn.wkv.weight", T),
    ]
    incomplete = _v4_incomplete_sglang_loader_pairs(buffer)
    assert incomplete == {"layers.1.attn.wqkv_a.weight_scale_inv": {"q"}}


def test_chunk_flush_predicate_defers_flush_until_pair_completes():
    """Semantics of _iter_non_expert_chunks' flush condition: an over-budget
    buffer must NOT flush while a V4 loader pair is incomplete, and must be
    allowed to flush once the pair completes."""

    def may_flush(buffer, over_budget, pair_protection=True):
        return bool(buffer) and over_budget and not (pair_protection and _v4_incomplete_sglang_loader_pairs(buffer))

    incomplete_buffer = [("layers.0.attn.compressor.wkv.weight", T)]
    complete_buffer = incomplete_buffer + [("layers.0.attn.compressor.wgate.weight", T)]

    assert not may_flush(incomplete_buffer, over_budget=True), "must hold the chunk open for wgate"
    assert may_flush(complete_buffer, over_budget=True)
    assert not may_flush(complete_buffer, over_budget=False)
    # without V4 pair protection (non-V4 models) budget alone decides
    assert may_flush(incomplete_buffer, over_budget=True, pair_protection=False)


def test_lora_base_weight_name_mapping():
    assert _lora_base_weight_name("layers.0.attn.wq_b.linear_in.weight") == "layers.0.attn.wq_b.weight"
    assert _lora_base_weight_name("layers.0.attn.wq_b.linear_out.weight") == "layers.0.attn.wq_b.weight"
    assert _lora_base_weight_name("layers.0.attn.wq_b.weight") is None


def test_merge_lora_weight_math_and_dtype():
    torch.manual_seed(0)
    base = torch.randn(6, 4, dtype=torch.bfloat16)
    lora_in = torch.randn(2, 4, dtype=torch.bfloat16)  # [rank, in]
    lora_out = torch.randn(6, 2, dtype=torch.bfloat16)  # [out, rank]
    scale = 0.5
    merged = _merge_lora_weight(base, lora_in, lora_out, scale)
    expected = (base.float() + lora_out.float() @ lora_in.float() * scale).to(torch.bfloat16)
    assert merged.dtype == base.dtype
    assert torch.equal(merged, expected)


def test_merge_lora_weight_rejects_shape_mismatch():
    base = torch.zeros(6, 4)
    with pytest.raises(ValueError, match="shape mismatch"):
        _merge_lora_weight(base, torch.zeros(2, 5), torch.zeros(6, 2), 1.0)
    with pytest.raises(ValueError, match="2D linear weights"):
        _merge_lora_weight(base.unsqueeze(0), torch.zeros(2, 4), torch.zeros(6, 2), 1.0)
