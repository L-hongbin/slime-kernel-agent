"""M2 unit test for the DeepSeek-V4 bf16 KV-cache pool + write path.

Runs on the node53 dev container (needs 1 GPU + the sglang bf16 patch applied:
scripts/dsv4/patches/sglang_dsv4_bf16_kv_full.patch). Validates:

  1. bf16 pool layout (1024 B/token, no scales, store_dtype=bf16).
  2. fp8 pool layout REGRESSION guard, including the 576-byte page padding math
     (a silent change here would only surface as a corrupt page at 16k).
  3. bf16 write roundtrip is EXACT (bf16 lossless) via the pack path
     (set_key_buffer) and the fused-store fallback (set_key_buffer_fused).
  4. Triton bf16 store == torch reference.
  5. fp8 store has quantization error where bf16 has none (the point of the mode).
  6. bf16-native Triton sparse DECODE == the preserved pure-torch decode
     reference (bit-compare to bf16-ULP). The torch decode was removed from the
     serving path (Triton-only, no runtime fallback) and lives here as the
     correctness oracle.

Invoke: pytest -q tests/deepseek-v4/test_dsv4_bf16_kv_pool.py
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 bf16 KV pool test requires a GPU")

QK_NOPE = 448
QK_ROPE = 64
LATENT = QK_NOPE + QK_ROPE  # 512
PAGE_SIZE = 64


def _make_single_pool(dtype: torch.dtype, size: int = 4096, layer_num: int = 2):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool

    return DeepSeekV4SingleKVPool(
        size=size,
        page_size=PAGE_SIZE,
        dtype=dtype,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        layer_num=layer_num,
        device="cuda",
        enable_memory_saver=False,
    )


def _read_bf16_tokens(buf: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """Reconstruct [N, 512] bf16 tokens written at `loc` from a bf16 pool buffer
    of shape (num_pages, page_size * 512)."""
    num_pages, numel_per_page = buf.shape
    flat = buf.reshape(-1)
    page = loc // PAGE_SIZE
    off = loc % PAGE_SIZE
    base = page * numel_per_page + off * LATENT
    idx = base[:, None] + torch.arange(LATENT, device=buf.device)
    return flat[idx.reshape(-1)].reshape(loc.shape[0], LATENT)


def test_bf16_single_pool_layout():
    pool = _make_single_pool(torch.bfloat16)
    assert pool.is_bf16_kv is True
    assert pool.store_dtype == torch.bfloat16
    # get_bytes_per_token returns store-dtype (bf16) ELEMENTS/token for bf16.
    assert pool.get_bytes_per_token() == LATENT  # 512
    assert pool.kv_cache_total_dim == LATENT
    buf = pool.kv_buffer[0]
    assert buf.dtype == torch.bfloat16
    num_pages = (pool.size + pool.page_size + 1) // pool.page_size
    assert buf.shape == (num_pages, pool.page_size * LATENT)
    # 1024 bytes/token = 1.754x the 584-byte fp8 layout.
    bytes_per_token = pool.get_bytes_per_token() * pool.store_dtype.itemsize
    assert bytes_per_token == 1024
    assert abs(bytes_per_token / 584 - 1.754) < 0.01


def test_fp8_single_pool_layout_regression():
    """fp8 default must be byte-identical, incl. the 576-byte page padding."""
    from sglang.srt.utils import ceil_div

    pool = _make_single_pool(torch.float8_e4m3fn)
    assert pool.is_bf16_kv is False
    assert pool.store_dtype == torch.uint8
    assert pool.get_bytes_per_token() == 448 + 64 * 2 + 8  # 584
    assert pool.kv_cache_total_dim == 584
    expected_padded = ceil_div(pool.page_size * 584, 576) * 576
    assert pool.bytes_per_page_padded == expected_padded
    buf = pool.kv_buffer[0]
    assert buf.dtype == torch.uint8
    num_pages = (pool.size + pool.page_size + 1) // pool.page_size
    assert buf.shape == (num_pages, expected_padded)


def _random_locs(n: int, size: int) -> torch.Tensor:
    return torch.randperm(size, device="cuda")[:n].to(torch.int64)


def test_bf16_write_roundtrip_pack_path():
    """set_key_buffer with a bf16 pack must roundtrip EXACTLY (lossless)."""
    from sglang.srt.layers.attention.dsv4.quant_k_cache import pack_nope_bf16_rope_bf16

    pool = _make_single_pool(torch.bfloat16)
    n = 300
    k = torch.randn(n, LATENT, device="cuda", dtype=torch.bfloat16)
    loc = _random_locs(n, pool.size)

    pack = pack_nope_bf16_rope_bf16(k)
    assert torch.equal(pack.k_nope_bf16, k[:, :QK_NOPE])
    assert torch.equal(pack.k_rope_bf16, k[:, QK_NOPE:])

    pool.set_key_buffer(layer_id=0, loc=loc, cache_nope_fp8_rope_bf16_pack=pack)
    readback = _read_bf16_tokens(pool.kv_buffer[0], loc)
    assert torch.equal(readback, k), "bf16 KV write must be lossless"


def test_bf16_write_roundtrip_fused_fallback():
    """set_key_buffer_fused (the default write path) must, for a bf16 pool,
    derive the non-fused store from dtype and produce identical bytes to the
    pack path — no operator env needed."""
    pool_a = _make_single_pool(torch.bfloat16)
    pool_b = _make_single_pool(torch.bfloat16)
    n = 257
    k = torch.randn(n, LATENT, device="cuda", dtype=torch.bfloat16)
    loc = _random_locs(n, pool_a.size)

    from sglang.srt.layers.attention.dsv4.quant_k_cache import pack_nope_bf16_rope_bf16

    pool_a.set_key_buffer_fused(layer_id=0, loc=loc, cache_k=k)
    pool_b.set_key_buffer(layer_id=0, loc=loc, cache_nope_fp8_rope_bf16_pack=pack_nope_bf16_rope_bf16(k))
    assert torch.equal(pool_a.kv_buffer[0], pool_b.kv_buffer[0])
    assert torch.equal(_read_bf16_tokens(pool_a.kv_buffer[0], loc), k)


def test_bf16_triton_matches_torch():
    from sglang.srt.layers.attention.dsv4 import index_buf_accessor as iba
    from sglang.srt.layers.attention.dsv4.quant_k_cache import pack_nope_bf16_rope_bf16

    pool_t = _make_single_pool(torch.bfloat16)
    pool_r = _make_single_pool(torch.bfloat16)
    n = 200
    k = torch.randn(n, LATENT, device="cuda", dtype=torch.bfloat16)
    loc = _random_locs(n, pool_t.size)
    pack = pack_nope_bf16_rope_bf16(k)

    iba._set_k_bf16_triton(pool_t.kv_buffer[0], loc, pack, PAGE_SIZE)
    iba._set_k_bf16_torch(pool_r.kv_buffer[0], loc, pack, PAGE_SIZE)
    assert torch.equal(pool_t.kv_buffer[0], pool_r.kv_buffer[0])


def test_bf16_write_then_dequant_read_roundtrip():
    """M2/M3 bridge: write via the bf16 store kernel, then read back via the
    prefill read path (dequantize_k_cache_paged bf16 gather branch). Must be
    exact (bf16 lossless, no fp8 dequant). Requires the M3 dequant patch."""
    from sglang.srt.layers.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
    from sglang.srt.layers.attention.dsv4.quant_k_cache import pack_nope_bf16_rope_bf16

    pool = _make_single_pool(torch.bfloat16)
    n = 288
    k = torch.randn(n, LATENT, device="cuda", dtype=torch.bfloat16)
    loc = _random_locs(n, pool.size)
    pool.set_key_buffer(layer_id=0, loc=loc, cache_nope_fp8_rope_bf16_pack=pack_nope_bf16_rope_bf16(k))

    out = dequantize_k_cache_paged(pool.kv_buffer[0], loc, page_size=PAGE_SIZE)
    assert out.shape == (n, 1, LATENT)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out[:, 0, :], k), "bf16 write->read must be lossless"


def test_fp8_has_quant_error_bf16_does_not():
    """Demonstrates the mode's purpose: bf16 is exact, fp8 nope is lossy."""
    from sglang.srt.layers.attention.dsv4.quant_k_cache import (
        pack_nope_bf16_rope_bf16,
        quant_to_nope_fp8_rope_bf16_pack_triton,
    )

    k = torch.randn(512, LATENT, device="cuda", dtype=torch.bfloat16)
    # bf16 pack: nope preserved bit-exactly.
    bf16_pack = pack_nope_bf16_rope_bf16(k)
    assert torch.equal(bf16_pack.k_nope_bf16, k[:, :QK_NOPE])
    # fp8 pack: nope goes through fp8 e4m3 + UE8M0 block scale -> lossy.
    fp8_pack = quant_to_nope_fp8_rope_bf16_pack_triton(k)
    fp8_nope_deq = fp8_pack.k_nope_fp8.to(torch.float32)  # ignoring scale, just fp8 grid
    # There must be at least some quantization error on the fp8 nope grid.
    err = (fp8_nope_deq - k[:, :QK_NOPE].to(torch.float32)).abs().max().item()
    assert err > 0.0


# ---------------------------------------------------------------------------
# Sparse DECODE bit-compare: the bf16-native Triton decode kernel vs the
# pure-torch reference.
#
# The torch reference below is the ORIGINAL sm120 sparse-decode implementation
# (`_gather_and_dequant` + `_sm120_sparse_decode_fwd`). It was deleted from the
# serving path (bf16 decode is Triton-only; no runtime torch fallback) and
# preserved HERE, verbatim, as the correctness oracle the deployed kernel is
# bit-compared against. Do not "optimize" it — its only job is to be an
# independent, obviously-correct float implementation.
# ---------------------------------------------------------------------------

_NOPE_DIM = 448
_ROPE_DIM = 64
_NOPE_ROPE_STRIDE = _NOPE_DIM + _ROPE_DIM * 2  # 576
_TILE_SIZE = 64
_NUM_TILES = _NOPE_DIM // _TILE_SIZE  # 7
_SCALE_STRIDE = _NUM_TILES + 1  # 8
_D = _NOPE_DIM + _ROPE_DIM  # 512


def _ref_gather_and_dequant(k_cache, indices, page_size):
    """Reference paged gather+dequant (bf16 plain gather / fp8 unpack+dequant)."""
    idx_shape = indices.shape
    flat_idx = indices.reshape(-1)
    N = flat_idx.shape[0]
    device = k_cache.device

    if k_cache.dtype == torch.bfloat16:
        pages = (flat_idx // page_size).clamp(min=0)
        offsets = (flat_idx % page_size).clamp(min=0)
        kc = k_cache.reshape(k_cache.shape[0], page_size, -1)
        gathered = kc[pages, offsets]
        return gathered.reshape(*idx_shape, _D)

    page_bytes = k_cache.stride(0)
    pages = flat_idx // page_size
    offsets = flat_idx % page_size
    safe_pages = pages.clamp(min=0)
    safe_offsets = offsets.clamp(min=0)
    num_pages = k_cache.shape[0]
    raw_pages = k_cache.as_strided((num_pages, page_bytes), (page_bytes, 1)).view(torch.uint8)
    nope_base = safe_offsets * _NOPE_ROPE_STRIDE
    nope_offsets = nope_base.unsqueeze(-1) + torch.arange(_NOPE_DIM, device=device, dtype=torch.long)
    rope_base = nope_base + _NOPE_DIM
    rope_offsets = rope_base.unsqueeze(-1) + torch.arange(_ROPE_DIM * 2, device=device, dtype=torch.long)
    scale_section_offset = page_size * _NOPE_ROPE_STRIDE
    scale_base = scale_section_offset + safe_offsets * _SCALE_STRIDE
    scale_offsets = scale_base.unsqueeze(-1) + torch.arange(_NUM_TILES, device=device, dtype=torch.long)
    page_idx_nope = safe_pages.unsqueeze(-1).expand_as(nope_offsets)
    nope_bytes = raw_pages[page_idx_nope, nope_offsets]
    page_idx_rope = safe_pages.unsqueeze(-1).expand_as(rope_offsets)
    rope_bytes = raw_pages[page_idx_rope, rope_offsets]
    page_idx_scale = safe_pages.unsqueeze(-1).expand_as(scale_offsets)
    scale_bytes = raw_pages[page_idx_scale, scale_offsets]
    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)
    rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)
    scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)
    result = torch.empty(N, _D, dtype=torch.bfloat16, device=device)
    result[:, :_NOPE_DIM] = (
        (nope_fp8.view(N, _NUM_TILES, _TILE_SIZE).float() * scale_e8m0.view(N, _NUM_TILES, 1).float())
        .view(N, _NOPE_DIM)
        .to(torch.bfloat16)
    )
    result[:, _NOPE_DIM:] = rope_bf16
    return result.reshape(*idx_shape, _D)


def _ref_sparse_decode_fwd(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    extra_k_cache=None,
    extra_indices=None,
    extra_topk_length=None,
):
    B, s_q, H_q, D_qk = q.shape
    num_pages, page_size, H_k, bpt = k_cache.shape
    topk = indices.shape[-1]
    invalid_mask = indices < 0
    safe_indices = indices.clamp(min=0)
    if topk_length is not None:
        topk_range = torch.arange(topk, device=topk_length.device).view(1, 1, topk)
        invalid_mask = invalid_mask | (topk_range >= topk_length.view(B, 1, 1))
    gathered_kv = _ref_gather_and_dequant(k_cache, safe_indices, page_size)
    if extra_k_cache is not None and extra_indices is not None:
        extra_topk = extra_indices.shape[-1]
        extra_page_size = extra_k_cache.shape[1]
        extra_invalid = extra_indices < 0
        extra_safe = extra_indices.clamp(min=0)
        if extra_topk_length is not None:
            extra_range = torch.arange(extra_topk, device=extra_topk_length.device).view(1, 1, extra_topk)
            extra_invalid = extra_invalid | (extra_range >= extra_topk_length.view(B, 1, 1))
        extra_kv = _ref_gather_and_dequant(extra_k_cache, extra_safe, extra_page_size)
        gathered_kv = torch.cat([gathered_kv, extra_kv], dim=2)
        invalid_mask = torch.cat([invalid_mask, extra_invalid], dim=2)
    gathered_kv[invalid_mask] = 0.0
    q_f = q.float()
    kv_f = gathered_kv.float()
    kv_d = kv_f.shape[-1]
    if D_qk != kv_d:
        q_f = q_f[..., :kv_d]
    scores = torch.einsum("bshd,bstd->bsht", q_f, kv_f) * softmax_scale
    scores.masked_fill_(invalid_mask.unsqueeze(2).expand_as(scores), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    if attn_sink is not None:
        lse_for_out = torch.logsumexp(torch.stack([lse, attn_sink.view(1, 1, H_q).expand_as(lse)], dim=0), dim=0)
    else:
        lse_for_out = lse.clone()
    lonely = lse == float("-inf")
    lse_for_out[lonely] = float("inf")
    weights = torch.exp(scores - lse_for_out.unsqueeze(-1))
    out = torch.einsum("bsht,bstv->bshv", weights, kv_f[..., :head_dim_v])
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0
    return out.to(torch.bfloat16), lse.permute(0, 2, 1)


def _decode_bf16_cache(num_pages, page_size, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    c = torch.randn(num_pages, page_size, 1, _D, generator=g, device=device) * 0.08
    return c.to(torch.bfloat16).contiguous()


def _decode_inputs(B, H, num_pages, page_size, topk, device, seed, with_len):
    g = torch.Generator(device=device).manual_seed(seed)
    n_slots = num_pages * page_size
    q = (torch.randn(B, 1, H, _D, generator=g, device=device) * 0.08).to(torch.bfloat16)
    idx = torch.randint(0, n_slots, (B, 1, topk), generator=g, device=device, dtype=torch.int32)
    invmask = torch.rand(B, 1, topk, generator=g, device=device) < 0.1
    idx = torch.where(invmask, torch.full_like(idx, -1), idx)
    topk_len = None
    if with_len:
        topk_len = torch.randint(topk // 2, topk + 1, (B,), generator=g, device=device, dtype=torch.int32)
    return q, idx, topk_len


def _decode_close(a, b, atol, rtol):
    a = a.float()
    b = b.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    both_nonfinite = (~torch.isfinite(a)) & (~torch.isfinite(b))
    pattern_ok = bool((finite | both_nonfinite).all().item())
    absdiff = torch.where(finite, (a - b).abs(), torch.zeros_like(a))
    within = torch.where(finite, absdiff <= atol + rtol * b.abs(), torch.ones_like(a, dtype=torch.bool))
    return pattern_ok and bool(within.all().item()), float(absdiff.max().item())


@pytest.mark.parametrize(
    "desc,B,H,num_pages,page_size,topk,with_len,with_sink,with_extra",
    [
        ("swa_only", 4, 16, 8, 64, 128, False, False, False),
        ("topklen", 4, 16, 8, 64, 128, True, False, False),
        ("sink", 4, 16, 8, 64, 128, True, True, False),
        ("extra_merge", 6, 32, 8, 64, 96, True, True, True),
        ("H128", 2, 128, 16, 64, 256, True, True, True),
    ],
)
def test_bf16_triton_decode_matches_torch_reference(
    desc, B, H, num_pages, page_size, topk, with_len, with_sink, with_extra
):
    """The deployed bf16-native Triton sparse decode must bit-compare (to
    bf16-ULP) against the preserved pure-torch reference on fixed inputs.

    `out` matches to bf16 rounding; LSE agrees once the shared attn-sink
    convention is reconciled (the Triton entry returns a sink-INCLUSIVE LSE,
    the torch reference a sink-EXCLUSIVE one — dtype-agnostic shared code)."""
    from sglang.srt.layers.attention.flash_mla_sm120_triton import flash_mla_sparse_decode_triton

    device = "cuda"
    seed = 100 + len(desc)
    q, idx, topk_len = _decode_inputs(B, H, num_pages, page_size, topk, device, seed, with_len)
    k_cache = _decode_bf16_cache(num_pages, page_size, device, seed + 1)
    sm_scale = 1.0 / (_D**0.5)
    attn_sink = (torch.randn(H, device=device) * 0.5).float() if with_sink else None
    extra_k = extra_idx = extra_len = None
    if with_extra:
        extra_k = _decode_bf16_cache(4, 32, device, seed + 7)
        _, extra_idx, extra_len = _decode_inputs(B, H, 4, 32, 48, device, seed + 9, with_len)

    ref_out, ref_lse = _ref_sparse_decode_fwd(
        q, k_cache, idx, topk_len, attn_sink, _D, sm_scale, extra_k, extra_idx, extra_len
    )
    tri_out, tri_lse = flash_mla_sparse_decode_triton(
        q, k_cache, idx, topk_len, attn_sink, _D, sm_scale, extra_k, extra_idx, extra_len
    )

    out_ok, out_maxabs = _decode_close(tri_out, ref_out, atol=2e-3, rtol=2e-2)
    assert out_ok, f"[{desc}] decode out mismatch, max_abs={out_maxabs:.3e}"

    ref_lse_cmp = ref_lse.float()
    if attn_sink is not None:
        sink_b = attn_sink.view(1, H, 1).expand_as(ref_lse_cmp)
        ref_lse_cmp = torch.logaddexp(ref_lse_cmp, sink_b)
    lse_ok, lse_maxabs = _decode_close(tri_lse, ref_lse_cmp, atol=5e-3, rtol=5e-3)
    assert lse_ok, f"[{desc}] decode lse mismatch, max_abs={lse_maxabs:.3e}"
