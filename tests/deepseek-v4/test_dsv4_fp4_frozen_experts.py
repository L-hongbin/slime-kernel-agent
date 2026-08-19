"""Packed-MXFP4 frozen experts (V4_FP4_FROZEN_EXPERTS=1) — the OFFICIAL
DeepSeek-V4-Flash checkpoint's routed experts stay resident as packed E2M1
nibbles + per-32 E8M0 scales; compute is W4A16 (transient bf16 unpack), backward
returns dX only. Pins: unpack golden bit-parity (vs deep_gemm and vs the FP8
checkpoint's lossless expansion of the same expert), packed-mode construction,
forward/grad parity, dispatch, sharded_state_dict load entries, and guards.
Design: handoffs/deepseek-v4/fp4_w4a16_design.md."""

import json
import os
import pathlib

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = REPO / "local_artifacts" / "fp4_fixtures"


def _mk_cfg():
    class _Cfg:
        num_local_experts = 4
        hidden_size = 256
        intermediate_size = 128
        hidden_act = "silu"
        swiglu_limit = 7.0
        quantization_config = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        }

    return _Cfg


def _random_packed(output_dim, input_dim, seed=0):
    """Random packed E2M1 bytes + sane E8M0 scale bytes (2^-10..2^-3 range)."""
    g = torch.Generator().manual_seed(seed)
    w_pack = torch.randint(0, 256, (output_dim, input_dim // 2), generator=g, dtype=torch.int64).to(torch.uint8)
    sf = torch.randint(117, 125, (output_dim, input_dim // 32), generator=g, dtype=torch.int64).to(torch.uint8)
    return w_pack, sf


def _reference_unpack(w_pack, sf):
    """Independent pure-python reference (LUT written out elementwise)."""
    lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    output_dim, packed_input_dim = w_pack.shape
    input_dim = packed_input_dim * 2
    out = torch.empty(output_dim, input_dim, dtype=torch.float32)
    wp = w_pack.long()
    lut_t = torch.tensor(lut)
    out[:, 0::2] = lut_t[(wp & 0x0F)]
    out[:, 1::2] = lut_t[(wp >> 4)]
    scale = torch.pow(2.0, sf.float() - 127.0)
    return (out.view(output_dim, input_dim // 32, 32) * scale[:, :, None]).reshape(output_dim, input_dim)


def _fp4_env(monkeypatch):
    monkeypatch.setenv("V4_FP4_FROZEN_EXPERTS", "1")


def _mk_packed_module(monkeypatch, seed=0):
    _fp4_env(monkeypatch)
    from custom_kernels.deepseek_v4.megatron.mcore_model import V4GroupedExperts

    cfg = _mk_cfg()
    e = V4GroupedExperts(cfg())
    for name in ("gate_up_proj", "down_proj"):
        pk = getattr(e, f"{name}_fp4")
        sf = getattr(e, f"{name}_sf")
        expert_count, output_dim, packed_input_dim = pk.shape
        for i in range(expert_count):
            w, s = _random_packed(
                output_dim,
                packed_input_dim * 2,
                seed=seed + i + (0 if name == "gate_up_proj" else 100),
            )
            pk[i].copy_(w)
            sf[i].copy_(s)
    return e, cfg


# ---------------------------------------------------------------- unpack core


def test_unpack_matches_independent_reference():
    from custom_kernels.deepseek_v4.megatron.mcore_model import _unpack_mxfp4

    w_pack, sf = _random_packed(64, 128, seed=3)
    mine = _unpack_mxfp4(w_pack, sf).to(torch.float32)
    ref = _reference_unpack(w_pack, sf)
    # decode is exact in bf16 for these scale ranges -> bitwise equal after cast
    assert torch.equal(mine, ref.to(torch.bfloat16).to(torch.float32))


def test_unpack_scale_decode_all_256_bytes():
    """The E8M0 decode must match torch's native float8_e8m0fnu semantics for
    EVERY byte value, including the boundaries (0x00 -> 2^-127, 0xFF -> NaN) —
    the (byte<<23) bit-trick diverges there (codex impl review finding 2).
    Production never sees 0x00/0xFF (verify_fp4_loaded rejects them), but the
    primitive itself must not be silently wrong if reused in audit tooling."""
    from custom_kernels.deepseek_v4.megatron.mcore_model import _unpack_mxfp4

    # one row, K=512 -> 16 scale groups per row; use 16 rows x 16 groups = 256 bytes
    output_dim, input_dim = 16, 512
    w_pack = torch.full((output_dim, input_dim // 2), 0x22, dtype=torch.uint8)  # every element = +1.0
    sf = torch.arange(256, dtype=torch.uint8).reshape(output_dim, input_dim // 32)
    out = _unpack_mxfp4(w_pack, sf).to(torch.float32)
    ref_scale = sf.view(torch.float8_e8m0fnu).to(torch.float32).to(torch.bfloat16).to(torch.float32)
    ref = ref_scale.repeat_interleave(32, dim=1)  # value 1.0 * scale
    assert torch.equal(out.isnan(), ref.isnan())
    assert torch.equal(out[~ref.isnan()], ref[~ref.isnan()])
    # boundary spot checks: byte 0 decodes to 2^-127 (bf16 subnormal), byte 255 to NaN
    assert out[0, 0] != 0.0 or ref[0, 0] == 0.0  # 2^-127 may underflow bf16 subnormal range identically
    assert bool(out[15, -1].isnan())


def test_unpack_rejects_wrong_dtypes_and_shapes():
    from custom_kernels.deepseek_v4.megatron.mcore_model import _unpack_mxfp4

    w_pack, sf = _random_packed(64, 128)
    with pytest.raises(AssertionError):
        _unpack_mxfp4(w_pack.to(torch.int8), sf)
    with pytest.raises(AssertionError):
        _unpack_mxfp4(w_pack, sf[:, :-1])


@pytest.mark.skipif(not (FIXTURES / "official_mxfp4_layers3_expert0.npz").exists(), reason="golden fixture absent")
def test_unpack_golden_official_fixture_bitparity():
    """Official packed bytes must decode bit-identically to (a) deep_gemm's
    cast_back_from_fp4 reference and (b) the secondary FP8 checkpoint's dequant
    of the SAME expert (the FP8 checkpoint is a lossless expansion of the
    official FP4 — validated 2026-07-16, design doc)."""
    import numpy as np

    from custom_kernels.deepseek_v4.megatron.mcore_model import _unpack_mxfp4
    from custom_kernels.deepseek_v4.megatron.native_checkpoint import dequant_fp8_block

    off = np.load(FIXTURES / "official_mxfp4_layers3_expert0.npz")
    off_meta = json.load(open(FIXTURES / "official_mxfp4_layers3_expert0.meta.json"))["tensors"]
    fp8 = np.load(FIXTURES / "fp8_layers3_expert0.npz")
    fp8_meta = json.load(open(FIXTURES / "fp8_layers3_expert0.meta.json"))["tensors"]

    def _t(z, meta, name):
        raw = torch.from_numpy(z[name].copy())
        shape = meta[name]["shape"]
        dt = meta[name]["dtype"]
        view = {
            "I8": torch.uint8,
            "F8_E8M0": torch.uint8,
            "F8_E4M3": torch.float8_e4m3fn,
            "F32": torch.float32,
        }[dt]
        return raw.view(view).reshape(shape)

    for wn in ("w1", "w2", "w3"):
        pk = _t(off, off_meta, f"{wn}_weight")
        sf = _t(off, off_meta, f"{wn}_scale")
        mine = _unpack_mxfp4(pk, sf).to(torch.float32)
        try:
            from deep_gemm import cast_back_from_fp4

            sf_f32 = (sf.to(torch.int32) << 23).view(torch.float32)
            ref = cast_back_from_fp4(pk.view(torch.int8), sf_f32, gran_k=32).to(torch.float32)
            assert torch.equal(mine, ref), f"{wn}: unpack != deep_gemm.cast_back_from_fp4"
        except ImportError:
            pass
        w8 = _t(fp8, fp8_meta, f"{wn}_weight")
        s8 = _t(fp8, fp8_meta, f"{wn}_scale")
        fp8_deq = dequant_fp8_block(w8, s8, output_dtype=torch.float32)
        assert torch.equal(mine, fp8_deq), f"{wn}: official FP4 decode != FP8 ckpt lossless expansion"


# ------------------------------------------------------------- packed module


def test_packed_mode_construction(monkeypatch):
    e, cfg = _mk_packed_module(monkeypatch)
    expert_count = cfg.num_local_experts
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    assert not hasattr(e, "gate_up_proj") and not hasattr(e, "down_proj")
    assert e.gate_up_proj_fp4.shape == (expert_count, 2 * intermediate_size, hidden_size // 2)
    assert e.gate_up_proj_fp4.dtype == torch.uint8
    assert e.gate_up_proj_sf.shape == (expert_count, 2 * intermediate_size, hidden_size // 32)
    assert e.down_proj_fp4.shape == (expert_count, hidden_size, intermediate_size // 2)
    assert e.down_proj_sf.shape == (expert_count, hidden_size, intermediate_size // 32)
    # buffers only: nothing in named_parameters -> invisible to optimizer/DDP
    assert not any("proj" in n for n, _ in e.named_parameters())
    buf_names = {n for n, _ in e.named_buffers()}
    assert {"gate_up_proj_fp4", "gate_up_proj_sf", "down_proj_fp4", "down_proj_sf"} <= buf_names


def test_bf16_mode_unchanged_without_env():
    from custom_kernels.deepseek_v4.megatron.mcore_model import V4GroupedExperts

    assert os.environ.get("V4_FP4_FROZEN_EXPERTS", "0") != "1"
    e = V4GroupedExperts(_mk_cfg()())
    assert hasattr(e, "gate_up_proj") and isinstance(e.gate_up_proj, torch.nn.Parameter)
    assert not getattr(e, "_experts_fp4", False)


def test_frozen_fp4_matmul_forward_parity_and_grad(monkeypatch):
    """_FrozenFp4ExpertLinear must match dequant+linear in forward AND grad_x,
    while saving only the packed uint8 buffers (16k-OOM discipline)."""
    from custom_kernels.deepseek_v4.megatron.mcore_model import _FrozenFp4ExpertLinear, _unpack_mxfp4

    e, cfg = _mk_packed_module(monkeypatch)
    x = torch.randn(6, cfg.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    wf = e.gate_up_proj_fp4[0]
    sf = e.gate_up_proj_sf[0]
    y = _FrozenFp4ExpertLinear.apply(x, wf, sf)
    y.sum().backward()
    gx = x.grad.clone()
    x2 = x.detach().clone().requires_grad_(True)
    w_ref = _unpack_mxfp4(wf, sf)
    y2 = torch.nn.functional.linear(x2, w_ref)
    y2.sum().backward()
    assert torch.equal(y, y2), "forward must be bit-equal to dequant+linear (same math)"
    assert torch.equal(gx, x2.grad), "grad_x must be bit-equal"


def test_forward_dispatched_loop_matches_bf16_reference(monkeypatch):
    """CPU path (per-expert loop through _expert_matmul's fp4 branch) must be
    bit-equal to a bf16-mode module holding the unpacked weights."""
    from custom_kernels.deepseek_v4.megatron.mcore_model import V4GroupedExperts, _unpack_mxfp4

    e, cfg = _mk_packed_module(monkeypatch)
    monkeypatch.delenv("V4_FP4_FROZEN_EXPERTS")
    ref = V4GroupedExperts(cfg())
    with torch.no_grad():
        for i in range(cfg.num_local_experts):
            ref.gate_up_proj.data[i] = _unpack_mxfp4(e.gate_up_proj_fp4[i], e.gate_up_proj_sf[i])
            ref.down_proj.data[i] = _unpack_mxfp4(e.down_proj_fp4[i], e.down_proj_sf[i])
    ref.gate_up_proj.data = ref.gate_up_proj.data.bfloat16()
    ref.down_proj.data = ref.down_proj.data.bfloat16()
    hs = torch.randn(6, cfg.hidden_size, dtype=torch.bfloat16)
    tpe = torch.tensor([2, 1, 2, 1])
    out_fp4 = e.forward_dispatched(hs, tpe)
    out_ref = ref.forward_dispatched(hs, tpe)
    assert torch.equal(out_fp4, out_ref)
    assert torch.isfinite(out_fp4).all()


def test_sharded_state_dict_packed_entries(monkeypatch):
    e, cfg = _mk_packed_module(monkeypatch)
    sd = e.sharded_state_dict(prefix="layers.0.mlp.experts.")
    keys = set(sd.keys())
    assert keys == {
        f"layers.0.mlp.experts.{n}" for n in ("gate_up_proj_fp4", "gate_up_proj_sf", "down_proj_fp4", "down_proj_sf")
    }
    st = sd["layers.0.mlp.experts.gate_up_proj_fp4"]
    # EP axis 0 sharding metadata, uint8 payload
    assert st.data.dtype == torch.uint8


def test_sharded_state_dict_packed_entries_mark_expert_dp_replicas(monkeypatch):
    """CP replicas must not all present themselves as the main EP shard.

    PP1 x CP2 x EP4 has two copies of each expert shard.  This reproduces the
    distributed-checkpoint integrity check that rejected every shard with an
    access count of two when V4GroupedExperts left ``replica_id`` at its default.
    """
    _fp4_env(monkeypatch)
    from custom_kernels.deepseek_v4.megatron.mcore_model import V4GroupedExperts
    from megatron.core.dist_checkpointing.validation import _validate_sharding_for_key

    class _RankGroup:
        def __init__(self, rank):
            self._rank = rank

        def rank(self):
            return self._rank

    rank_shards = []
    for ep_rank in range(4):
        for expert_dp_rank in range(2):
            experts = V4GroupedExperts(
                _mk_cfg()(),
                expert_model_parallel_size=4,
                expert_model_parallel_rank=ep_rank,
                expert_data_parallel_group=_RankGroup(expert_dp_rank),
            )
            state = experts.sharded_state_dict(prefix="layers.0.mlp.experts.")
            for sharded in state.values():
                assert sharded.replica_id == (0, 0, expert_dp_rank)
            rank_shards.append((len(rank_shards), state["layers.0.mlp.experts.gate_up_proj_fp4"]))

    _validate_sharding_for_key(rank_shards)


def test_verify_fp4_loaded_guard(monkeypatch):
    _fp4_env(monkeypatch)
    from custom_kernels.deepseek_v4.megatron.mcore_model import V4GroupedExperts

    e = V4GroupedExperts(_mk_cfg()())
    with pytest.raises(RuntimeError, match="invalid E8M0 scale bytes"):
        e.verify_fp4_loaded()  # zero-init buffers = unpopulated (0x00)
    e2, _ = _mk_packed_module(monkeypatch)
    e2.verify_fp4_loaded()  # populated with sane bytes -> passes
    # a SINGLE 0xFF (NaN scale) must also be rejected — it poisons 32 weights
    e2.gate_up_proj_sf[1, 3, 0] = 0xFF
    with pytest.raises(RuntimeError, match="0xFF"):
        e2.verify_fp4_loaded()


# ---------------------------------------------------------------- converter


class _FakeCheckpoint:
    """Duck-typed NativeV4Checkpoint over an in-memory tensor dict."""

    def __init__(self, tensors):
        self._tensors = tensors

    def get_tensor(self, key):
        return self._tensors[key]

    def has_tensor(self, key):
        return key in self._tensors


def test_read_expert_tensors_packed_passthrough():
    from custom_kernels.deepseek_v4.megatron.native_checkpoint import read_expert_tensors

    hidden_size, intermediate_size = 256, 128
    tensors = {}
    parts = {}
    weight_shapes = {
        "w1": (intermediate_size, hidden_size),
        "w3": (intermediate_size, hidden_size),
        "w2": (hidden_size, intermediate_size),
    }
    for wn, (output_dim, input_dim) in weight_shapes.items():
        pk, sf = _random_packed(output_dim, input_dim, seed=hash(wn) % 1000)
        tensors[f"layers.3.ffn.experts.7.{wn}.weight"] = pk.view(torch.int8)  # safetensors I8
        tensors[f"layers.3.ffn.experts.7.{wn}.scale"] = sf.view(torch.float8_e8m0fnu)
        parts[wn] = (pk, sf)
    out = read_expert_tensors(_FakeCheckpoint(tensors), 3, 7)
    gu = out["layers.3.mlp.experts.gate_up_proj_fp4[7]"]
    gu_sf = out["layers.3.mlp.experts.gate_up_proj_sf[7]"]
    dn = out["layers.3.mlp.experts.down_proj_fp4[7]"]
    dn_sf = out["layers.3.mlp.experts.down_proj_sf[7]"]
    assert gu.dtype == torch.uint8 and gu_sf.dtype == torch.uint8
    assert torch.equal(gu, torch.cat([parts["w1"][0], parts["w3"][0]], dim=0))
    assert torch.equal(gu_sf, torch.cat([parts["w1"][1], parts["w3"][1]], dim=0))
    assert torch.equal(dn, parts["w2"][0])
    assert torch.equal(dn_sf, parts["w2"][1])


def test_read_expert_tensors_packed_missing_scale_raises():
    from custom_kernels.deepseek_v4.megatron.native_checkpoint import read_expert_tensors

    pk, sf = _random_packed(128, 256)
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": pk.view(torch.int8),
        # w1.scale missing
        "layers.0.ffn.experts.0.w3.weight": pk.view(torch.int8),
        "layers.0.ffn.experts.0.w3.scale": sf.view(torch.float8_e8m0fnu),
        "layers.0.ffn.experts.0.w2.weight": pk.view(torch.int8),
        "layers.0.ffn.experts.0.w2.scale": sf.view(torch.float8_e8m0fnu),
    }
    with pytest.raises(KeyError, match="missing scale"):
        read_expert_tensors(_FakeCheckpoint(tensors), 0, 0)


def test_dequant_fp8_block_accepts_e8m0_scales():
    """Official FP8 tensors (attention/shared experts/wo_a) carry F8_E8M0 scale
    BYTES instead of F32; dequant_fp8_block must decode them via the exact
    2^(e-127) float cast."""
    from custom_kernels.deepseek_v4.megatron.native_checkpoint import dequant_fp8_block

    w = torch.randn(256, 256).to(torch.float8_e4m3fn)
    sf_bytes = torch.randint(117, 125, (2, 2), dtype=torch.uint8)
    out_e8m0 = dequant_fp8_block(w, sf_bytes.view(torch.float8_e8m0fnu))
    out_f32 = dequant_fp8_block(w, torch.pow(2.0, sf_bytes.float() - 127.0))
    assert torch.equal(out_e8m0, out_f32)
