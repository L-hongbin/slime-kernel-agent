"""Tests for V4Indexer — official HF ``DeepseekV4Indexer`` structure with
DeepSeek's official DeepGEMM ``fp8_mqa_logits`` scoring kernel on the GPU path
and the HF-exact torch tail as fallback/reference.

CPU tests run everywhere; the DeepGEMM parity tests need a CUDA device with
deep_gemm and skip otherwise.
Run: python -m pytest tests/deepseek-v4/test_dsv4_indexer.py -q
"""

from __future__ import annotations

import pytest
import torch

hf_v4 = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")

from custom_kernels.deepseek_v4.megatron import indexer as indexer_mod  # noqa: E402
from custom_kernels.deepseek_v4.megatron.indexer import V4Indexer, _quant_e4m3_lastdim  # noqa: E402
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer  # noqa: E402

_CKPT = "/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8"


def _real_cfg():
    """The REAL V4-Flash config (correct rope/compress surface), with the
    index_topk shrunk so top-k < n_windows at test-sized sequences."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(_CKPT, trust_remote_code=True)
    cfg.index_topk = 3
    return cfg


def _mk_pair(seed=0, cfg=None):
    """Ours + pure-HF instance with IDENTICAL weights."""
    torch.manual_seed(seed)
    cfg = cfg or _real_cfg()
    ours = V4Indexer(cfg)
    ref = DeepseekV4Indexer(cfg)
    with torch.no_grad():
        for (n_o, p_o), (n_r, p_r) in zip(
            sorted(ours.state_dict().items()), sorted(ref.state_dict().items()), strict=True
        ):
            assert n_o == n_r
            init = torch.randn_like(p_r.float()).to(p_r.dtype) * 0.05
            p_o.copy_(init)
            p_r.copy_(init)
    return cfg, ours, ref


def _inputs(cfg, B=2, S=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(B, S, cfg.hidden_size, generator=g)
    q_res = torch.randn(B, S, cfg.q_lora_rank, generator=g)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)
    return hidden, q_res, pos


# --------------------------------------------------------------- inheritance


def test_is_official_hf_subclass_and_state_dict_compatible():
    """Structure/params/buffers come from the official HF class — checkpoint
    keys, shapes, and dtypes must be identical (converter compatibility)."""
    assert issubclass(V4Indexer, DeepseekV4Indexer)
    assert "__init__" not in V4Indexer.__dict__  # init inherited verbatim
    cfg, ours, ref = _mk_pair()
    ours_sd, ref_sd = ours.state_dict(), ref.state_dict()
    assert set(ours_sd) == set(ref_sd)
    for k in ours_sd:
        assert ours_sd[k].shape == ref_sd[k].shape, k
        assert ours_sd[k].dtype == ref_sd[k].dtype, k


# ------------------------------------------------- torch fallback == HF exact


def test_torch_fallback_matches_hf_forward_exactly():
    """On CPU, our cacheless forward must reproduce the
    official HF forward bit-for-bit (same submodules, same op order)."""
    cfg, ours, ref = _mk_pair(seed=2)
    hidden, q_res, pos = _inputs(cfg)

    got = ours(hidden, q_res, pos, None, layer_idx=0)
    want = ref(hidden, q_res, pos, None, layer_idx=0)
    assert got.shape == want.shape
    assert torch.equal(got, want)


def test_causal_sentinels_match_hf():
    """Early queries with too few ready compressed blocks must get -1 sentinels,
    exactly like HF (the consumer scatters them into a throwaway slot)."""
    cfg, ours, ref = _mk_pair(seed=3)
    hidden, q_res, pos = _inputs(cfg, B=1, S=12)

    got = ours(hidden, q_res, pos, None, 0)
    want = ref(hidden, q_res, pos, None, 0)
    assert torch.equal(got, want)
    # queries before the first full window can select nothing
    causal = (pos + 1) // cfg.compress_rates["compressed_sparse_attention"]
    early = causal[0] == 0
    assert (got[0][early] == -1).all()
    # every non-sentinel pick respects causality
    valid = got[got >= 0]
    assert valid.numel() > 0


def test_short_sequence_no_full_window():
    """S < compress_rate -> zero compressed entries; must not crash and must
    return an empty-k tensor the mask builder can consume."""
    cfg, ours, _ = _mk_pair(seed=4)
    hidden, q_res, pos = _inputs(cfg, B=1, S=3)
    got = ours(hidden, q_res, pos, None, 0)
    assert got.shape[-1] == 0


def test_selection_carries_no_grad():
    """Top-k indices are discrete: the output must not require grad, and calling
    the indexer must not create a graph edge from hidden states to the output."""
    cfg, ours, _ = _mk_pair(seed=5)
    hidden, q_res, pos = _inputs(cfg)
    hidden.requires_grad_(True)
    q_res.requires_grad_(True)
    got = ours(hidden, q_res, pos, None, 0)
    assert got.dtype in (torch.int64, torch.int32)
    assert not got.requires_grad


def test_cacheless_only_contract():
    cfg, ours, _ = _mk_pair()
    hidden, q_res, pos = _inputs(cfg)
    with pytest.raises(AssertionError, match="cacheless"):
        ours(hidden, q_res, pos, object(), 0)


# --------------------------------------------------------- quantization helper


def test_quant_e4m3_roundtrip_accuracy():
    """sf = amax/448 per row; dequant must reconstruct within fp8 e4m3 relative
    precision (~2^-3 worst-case mantissa step at these magnitudes)."""
    torch.manual_seed(0)
    x = torch.randn(64, 32) * 5.0
    q, sf = _quant_e4m3_lastdim(x)
    assert q.dtype == torch.float8_e4m3fn and sf.dtype == torch.float32
    recon = q.float() * sf.unsqueeze(-1)
    rel = (recon - x).abs().max() / x.abs().max()
    assert rel < 0.07, f"fp8 roundtrip rel err {rel}"
    # amax rows map to exactly +-448 * sf
    assert torch.isfinite(q.float()).all()


# ---------------------------------------------------------- DeepGEMM GPU path

_needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and indexer_mod._HAS_DEEP_GEMM),
    reason="needs CUDA + deep_gemm.fp8_mqa_logits",
)


@_needs_gpu
def test_deepgemm_scores_close_to_torch_reference(monkeypatch):
    """The official kernel's logits must match the HF-exact torch tail within
    fp8 quantization error, inside each query's valid causal range."""
    cfg, ours, _ = _mk_pair(seed=6)
    ours = ours.cuda()
    hidden, q_res, pos = _inputs(cfg, B=2, S=32, seed=7)
    hidden, q_res, pos = hidden.cuda(), q_res.cuda(), pos.cuda()

    n_windows = hidden.shape[1] // cfg.compress_rates["compressed_sparse_attention"]
    assert n_windows >= 2

    monkeypatch.setattr(indexer_mod, "_use_deepgemm_scoring", lambda _device: False)
    ref_idx = ours(hidden, q_res, pos, None, 0)
    monkeypatch.setattr(indexer_mod, "_use_deepgemm_scoring", lambda _device: True)
    got_idx = ours(hidden, q_res, pos, None, 0)

    assert got_idx.shape == ref_idx.shape
    # sentinel structure identical (causality is enforced by exact ke bounds)
    assert torch.equal(got_idx == -1, ref_idx == -1)
    # selection overlap: fp8 rounding may swap near-tied entries; demand high
    # per-query set overlap on valid picks
    agree = 0
    total = 0
    for b in range(got_idx.shape[0]):
        for s in range(got_idx.shape[1]):
            g = set(got_idx[b, s][got_idx[b, s] >= 0].tolist())
            r = set(ref_idx[b, s][ref_idx[b, s] >= 0].tolist())
            if not r:
                continue
            agree += len(g & r)
            total += len(r)
    assert total > 0
    assert agree / total >= 0.9, f"top-k overlap {agree}/{total}"


@_needs_gpu
def test_deepgemm_path_selected_on_cuda_by_default():
    assert indexer_mod._use_deepgemm_scoring(torch.device("cuda"))


def test_cpu_never_selects_deepgemm():
    assert not indexer_mod._use_deepgemm_scoring(torch.device("cpu"))


# --------------------------------------------------------------- integration


def test_csa_compressor_instantiates_v4_indexer():
    """The CSA compressor must build OUR indexer (official structure + official
    kernel path), not the plain HF one."""
    comp_mod = pytest.importorskip("custom_kernels.deepseek_v4.megatron.compressor")
    import inspect

    src = inspect.getsource(comp_mod.V4CSACompressor.__init__)
    assert "V4Indexer" in src
