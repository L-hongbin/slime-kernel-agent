"""CPU unit tests for the V4-Flash CP2 orchestration layer (cp_utils + attention /
compressor CP paths).

GPU is unavailable, so every check runs on the CPU: the CUDA attention/compressor
kernels are mocked (or the torch reference pool is injected) and the distributed
comm is exercised with a real gloo process group via ``torch.multiprocessing.spawn``.
These tests cover the torch-level ORCHESTRATION -- shapes, indices, autograd flow,
and comm adjointness -- not the kernels themselves.

Coverage (``dsv4_megatron_sharding_contract.md`` CP2 invariants):
  (a) CpHaloExchange fwd/bwd incl. gradient accumulation back to the halo owner.
  (b) per-layer-type boundary drop counts (CSA m=4 -> 32, HCA m=128 -> 1).
  (c) compressed all-gather fwd <-> reduce-scatter bwd adjointness.
  (d) the l_local % 128 guard fires (B1).
  (e) cp_size==1 constructs NO CP op (bit-identical non-CP path).
  (+) compressor CP parity: haloed-compress + drop + global-position RoPE reconstructs
      the whole-sequence compressor output (CSA overlap boundary + HCA), on CPU with
      the real module + the exact torch reference pool.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

NUM_GPUS = 0

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from custom_kernels.deepseek_v4.megatron.cp_utils import (
    CP_HALO,
    CpCompAllgather,
    CpHaloExchange,
    CpInfo,
    assert_local_len_aligned,
    compressor_drop_windows,
    cp_allgather_compressed,
    cp_halo_exchange,
    get_cp_info,
)

_GLOO = dist.is_available() and dist.is_gloo_available()
pytestmark = pytest.mark.skipif(not _GLOO, reason="gloo backend required for CP2 comm tests")


# --------------------------------------------------------------------------------------
# gloo multiprocess harness
# --------------------------------------------------------------------------------------
def _run(worker, world, *args, port):
    """Spawn ``world`` gloo processes; re-raises any worker assertion as a test failure."""
    mp.spawn(worker, args=(world, port, *args), nprocs=world, join=True)


def _init(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)


def _world_cpinfo(rank, world):
    # cp group == the whole world here; global ranks are 0..world-1.
    return CpInfo(group=dist.group.WORLD, rank=rank, size=world, global_ranks=tuple(range(world)))


def _coef(global_positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Deterministic per-(position, channel) loss coefficient -- a pure function of the
    GLOBAL token position so a single-process reference can reconstruct it exactly."""
    p = global_positions.to(torch.float64).view(-1, 1)
    d = torch.arange(dim, dtype=torch.float64).view(1, -1)
    return torch.cos(0.1 * p + 0.03 * d)


# --------------------------------------------------------------------------------------
# (a) CpHaloExchange forward + backward (grad accumulation to the halo owner)
# --------------------------------------------------------------------------------------
def _halo_worker(rank, world, port, B, l_local, D, halo):
    _init(rank, world, port)
    torch.manual_seed(1234)
    # Identical full tensor on every rank -> each rank's shard is a slice of it, and the
    # single-process reference below uses the same values.
    full = torch.randn(B, world * l_local, D, dtype=torch.float64)
    cp = _world_cpinfo(rank, world)

    shard = full[:, rank * l_local : (rank + 1) * l_local].clone().requires_grad_(True)
    out = cp_halo_exchange(shard, cp, halo=halo)

    # ---- forward: [halo || local] for rank>0, local for rank0 ----
    if rank == 0:
        exp = full[:, 0:l_local]
        pos = torch.arange(0, l_local)
    else:
        exp = torch.cat([full[:, rank * l_local - halo : rank * l_local], shard.detach()], dim=1)
        pos = torch.arange(rank * l_local - halo, (rank + 1) * l_local)
    assert out.shape == exp.shape, (rank, out.shape, exp.shape)
    assert torch.allclose(out, exp), f"rank{rank} forward mismatch"

    # ---- backward: loss_r = (out_r * coef_r).sum, coef over the OUTPUT's global positions ----
    coef = _coef(pos, D).view(1, -1, D)
    (out * coef).sum().backward()

    # ---- single-process autograd reference (same full tensor + coefficients) ----
    ref = full.clone().requires_grad_(True)
    total = ref.new_zeros(())
    for r in range(world):
        if r == 0:
            o = ref[:, 0:l_local]
            gp = torch.arange(0, l_local)
        else:
            o = torch.cat([ref[:, r * l_local - halo : r * l_local], ref[:, r * l_local : (r + 1) * l_local]], dim=1)
            gp = torch.arange(r * l_local - halo, (r + 1) * l_local)
        total = total + (o * _coef(gp, D).view(1, -1, D)).sum()
    total.backward()
    ref_grad = ref.grad[:, rank * l_local : (rank + 1) * l_local]

    assert shard.grad is not None
    assert torch.allclose(shard.grad, ref_grad), f"rank{rank} grad mismatch (halo accumulation)"
    # The owner's last `halo` rows must have received the right-neighbour's boundary grad.
    if rank < world - 1:
        boundary = shard.grad[:, -halo:]
        interior_dup = ref_grad[:, -halo:]
        assert torch.allclose(boundary, interior_dup)
        # sanity: the boundary grad is genuinely the accumulated (doubled-coefficient) one,
        # not just the local contribution -> differs from the coefficient alone.
        gp_b = torch.arange((rank + 1) * l_local - halo, (rank + 1) * l_local)
        local_only = (_coef(gp_b, D).view(1, -1, D))[0]
        assert not torch.allclose(boundary[0], local_only), f"rank{rank} boundary grad not accumulated"
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3])
def test_halo_exchange_fwd_bwd(world):
    _run(_halo_worker, world, 2, 16, 4, 8, port=29610 + world)


# --------------------------------------------------------------------------------------
# (c) compressed all-gather (fwd) <-> reduce-scatter (bwd) adjointness + values
# --------------------------------------------------------------------------------------
def _allgather_worker(rank, world, port, B, T, D):
    _init(rank, world, port)
    cp = _world_cpinfo(rank, world)

    # Distinct local tensor per rank (seeded by rank) -> reconstructable everywhere.
    def local_of(r):
        g = torch.Generator().manual_seed(100 + r)
        return torch.randn(B, 1, T, D, generator=g, dtype=torch.float64)

    x = local_of(rank).clone().requires_grad_(True)
    X = cp_allgather_compressed(x, cp)

    # ---- forward value: X == concat of every rank's local, in rank order ----
    expected = torch.cat([local_of(r) for r in range(world)], dim=2)
    assert X.shape == (B, 1, world * T, D)
    assert torch.allclose(X, expected), f"rank{rank} allgather order/value mismatch"

    # ---- adjointness: sum_r <A x, y_r> == sum_r <x_r, A* y_r> ----
    gy = torch.Generator().manual_seed(500 + rank)
    y = torch.randn_like(X)  # per-rank distinct cotangent
    y = y + torch.randn(B, 1, world * T, D, generator=gy, dtype=torch.float64)
    (x_grad,) = torch.autograd.grad(X, x, grad_outputs=y)  # A* y  (reduce-scatter)
    lhs = (X * y).sum()
    rhs = (x * x_grad).sum()
    lhs_all = lhs.clone()
    rhs_all = rhs.clone()
    dist.all_reduce(lhs_all)
    dist.all_reduce(rhs_all)
    assert torch.allclose(lhs_all, rhs_all), f"rank{rank} allgather/reduce-scatter not adjoint"

    # ---- explicit reduce-scatter value: y_r constant (rank+1) -> owner gets sum_p (p+1) ----
    y_const = torch.full((B, 1, world * T, D), float(rank + 1), dtype=torch.float64)
    (g_const,) = torch.autograd.grad(X, x, grad_outputs=y_const)
    expect = float(sum(p + 1 for p in range(world)))
    assert torch.allclose(g_const, torch.full_like(g_const, expect)), f"rank{rank} reduce-scatter sum wrong"
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3])
def test_compressed_allgather_adjoint(world):
    _run(_allgather_worker, world, 2, 3, 4, port=29620 + world)


# --------------------------------------------------------------------------------------
# (b) boundary drop counts per layer type + owner rule
# --------------------------------------------------------------------------------------
def test_drop_window_counts_per_layer_type():
    assert compressor_drop_windows(CP_HALO, 4) == 32  # CSA m=4
    assert compressor_drop_windows(CP_HALO, 128) == 1  # HCA m=128
    assert compressor_drop_windows(0, 4) == 0  # rank0 (no halo)
    assert compressor_drop_windows(0, 128) == 0
    with pytest.raises(AssertionError):
        compressor_drop_windows(127, 4)  # halo must be a multiple of the compress rate


def test_owner_rule_window_counts():
    # With a 128-aligned l_local, each rank emits exactly l_local/m windows: haloed
    # window count (halo+l_local)/m minus the dropped halo//m equals l_local/m.
    for m in (4, 128):
        for l_local in (128, 256, 8192):
            haloed = (CP_HALO + l_local) // m
            kept = haloed - compressor_drop_windows(CP_HALO, m)
            assert kept == l_local // m


# --------------------------------------------------------------------------------------
# (d) B1 alignment guard
# --------------------------------------------------------------------------------------
def test_local_len_alignment_guard():
    enabled = CpInfo(group=None, rank=1, size=2, global_ranks=(0, 1))
    assert_local_len_aligned(128, enabled)  # ok
    assert_local_len_aligned(8192, enabled)  # ok
    with pytest.raises(AssertionError):
        assert_local_len_aligned(200, enabled)  # not a multiple of 128
    # disabled CP: never fires (non-128 lengths are fine at cp_size==1).
    assert_local_len_aligned(200, CpInfo())


# --------------------------------------------------------------------------------------
# (e) cp_size==1 constructs no CP op (bit-identical non-CP path)
# --------------------------------------------------------------------------------------
def test_cp_disabled_is_identity_noop():
    disabled = CpInfo()
    assert not disabled.enabled
    x = torch.randn(2, 5, 4, requires_grad=True)
    # No autograd Function is inserted: the SAME tensor object is returned.
    assert cp_halo_exchange(x, disabled) is x
    k = torch.randn(2, 1, 3, 4, requires_grad=True)
    assert cp_allgather_compressed(k, disabled) is k


def test_get_cp_info_without_distributed_is_disabled():
    # No process group initialized in the parent test process -> disabled info.
    info = get_cp_info()
    assert isinstance(info, CpInfo)
    assert not info.enabled and info.size == 1


def test_halo_function_size1_passthrough_grad():
    # Calling the Function directly at size==1 wraps the output in a (new) autograd node,
    # but the value is unchanged and the grad flows through as the identity -- no comm.
    x = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    out = CpHaloExchange.apply(x, 8, None, 0, 1, None)  # size==1 -> passthrough
    assert torch.equal(out, x)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_allgather_function_size1_passthrough_grad():
    k = torch.randn(2, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    out = CpCompAllgather.apply(k, None, 0, 1)  # size==1 -> passthrough
    assert torch.equal(out, k)
    (out * 3.0).sum().backward()
    assert torch.allclose(k.grad, torch.full_like(k, 3.0))


# --------------------------------------------------------------------------------------
# (+) compressor CP parity: haloed-compress + drop + global RoPE == whole sequence
# --------------------------------------------------------------------------------------
def _build_tiny_compressors():
    from custom_kernels.deepseek_v4.compression.reference import csa_compress_ref, hca_compress_ref
    from custom_kernels.deepseek_v4.megatron import compressor as C
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    cfg = DeepseekV4Config(
        hidden_size=256,
        head_dim=64,
        num_attention_heads=4,
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=4,
        num_hidden_layers=4,
        first_k_dense_replace=0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        intermediate_size=256,
    )
    # Inject the exact torch reference pool for the (otherwise CUDA) B2 kernels so the
    # module runs on CPU; RoPE + the drop logic are the code under test.
    C._csa_compress = csa_compress_ref
    C._hca_compress = hca_compress_ref
    return cfg, C


def _assert_compressor_cp_parity(compressor, m, l_local, world):
    B = 1
    hidden = compressor.kv_proj.weight.shape[1]
    torch.manual_seed(7)
    full = torch.randn(B, world * l_local, hidden, dtype=torch.float64)

    whole = compressor(full, position_offset=0, drop_windows=0)  # [B,1,S/m,D]
    assert whole.shape[2] == (world * l_local) // m

    parts = []
    for r in range(world):
        if r == 0:
            h, off, drop = full[:, 0:l_local], 0, 0
        else:
            h = torch.cat(
                [full[:, r * l_local - CP_HALO : r * l_local], full[:, r * l_local : (r + 1) * l_local]], dim=1
            )
            off, drop = r * l_local - CP_HALO, compressor_drop_windows(CP_HALO, m)
        p = compressor(h, position_offset=off, drop_windows=drop)
        assert p.shape[2] == l_local // m, (r, p.shape, l_local // m)  # owner rule
        parts.append(p)
    cp = torch.cat(parts, dim=2)
    assert cp.shape == whole.shape
    assert torch.allclose(cp, whole, atol=1e-5), (cp - whole).abs().max().item()


def test_csa_compressor_cp_parity():
    cfg, C = _build_tiny_compressors()
    comp = C.V4CSACompressor(cfg).double()
    _assert_compressor_cp_parity(comp, m=4, l_local=128, world=2)


def test_hca_compressor_cp_parity():
    cfg, C = _build_tiny_compressors()
    comp = C.V4HCACompressor(cfg).double()
    _assert_compressor_cp_parity(comp, m=128, l_local=128, world=2)


# --------------------------------------------------------------------------------------
# (+) V4Attention CP path: real gloo comm, mocked (but differentiable) A1 kernel.
#     Verifies (i) the kernel-arg wiring (q_pos0/raw_halo/haloed k_raw/global k_comp)
#     and (ii) the halo-dk_raw backward path -- the halo rows' gradient flows through the
#     SAME cp_halo_exchange back into the LEFT neighbour's local hidden grad.
# --------------------------------------------------------------------------------------
def _tiny_attn_config():
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        hidden_size=128,
        head_dim=64,
        num_attention_heads=4,
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=4,
        num_hidden_layers=4,
        first_k_dense_replace=0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        intermediate_size=256,
    )


def _pos_embeddings_for_rank(cfg, rank, l_local, dtype):
    """Build the {main, compress, *_haloed} cos/sin the CP attention reads, matching what
    mcore_model._position_embeddings produces: GLOBAL local positions + a halo'd axis."""
    from custom_kernels.deepseek_v4.megatron.rope import DeepseekV4RotaryEmbedding

    rope = DeepseekV4RotaryEmbedding(cfg).to(dtype)
    x = torch.zeros(1, 1, cfg.hidden_size, dtype=dtype)
    gs = rank * l_local
    local = (gs + torch.arange(l_local)).unsqueeze(0)
    emb = {
        "main": rope(x, position_ids=local, layer_type="main"),
        "compress": rope(x, position_ids=local, layer_type="compress"),
    }
    if rank > 0:
        haloed = (gs - CP_HALO + torch.arange(CP_HALO + l_local)).unsqueeze(0)
        emb["main_haloed"] = rope(x, position_ids=haloed, layer_type="main")
        emb["compress_haloed"] = rope(x, position_ids=haloed, layer_type="compress")
    return emb


def _attn_cp_worker(rank, world, port, layer_type):
    _init(rank, world, port)
    from custom_kernels.deepseek_v4.compression.reference import csa_compress_ref, hca_compress_ref
    from custom_kernels.deepseek_v4.megatron import attention as A
    from custom_kernels.deepseek_v4.megatron import compressor as Cmod

    Cmod._csa_compress = csa_compress_ref
    Cmod._hca_compress = hca_compress_ref

    cfg = _tiny_attn_config()
    dtype = torch.float64
    l_local = 128
    cp = _world_cpinfo(rank, world)
    A.get_cp_info = lambda: cp  # route the module's CP path through this gloo world

    attn = A.V4Attention(cfg, layer_idx=0, layer_type_override=layer_type).to(dtype)
    pos_emb = _pos_embeddings_for_rank(cfg, rank, l_local, dtype)
    m = attn.compress_rate
    this_halo = 0 if rank == 0 else CP_HALO
    cap = {}

    def make_mock(use_halo):
        def mock(q, k_raw, k_comp, sinks, window, mm, comp_topk_mask=None, q_pos0=0, raw_halo=0):
            cap["q_pos0"] = q_pos0
            cap["raw_halo"] = raw_halo
            cap["q_shape"] = tuple(q.shape)
            cap["k_raw_shape"] = tuple(k_raw.shape)
            cap["k_comp_shape"] = None if k_comp is None else tuple(k_comp.shape)
            base = k_raw if use_halo else k_raw[:, :, raw_halo:, :]  # drop halo rows if !use_halo
            out = q + base.sum(dim=2, keepdim=True)  # depends on (all / non-halo) k_raw rows
            if k_comp is not None:
                out = out + k_comp.sum(dim=2, keepdim=True)
            return out  # [B,H,S,D]

        return mock

    def run(use_halo):
        g = torch.Generator().manual_seed(2000 + rank)
        h = torch.randn(1, l_local, cfg.hidden_size, generator=g, dtype=dtype, requires_grad=True)
        A._v4flash_attention = make_mock(use_halo)
        out = attn(h, pos_emb)
        out.sum().backward()
        return h.grad.clone()

    grad_full = run(use_halo=True)

    # ---- (i) kernel-arg wiring ----
    assert cap["q_pos0"] == rank * l_local, (rank, cap["q_pos0"])
    assert cap["raw_halo"] == this_halo, (rank, cap["raw_halo"])
    assert cap["q_shape"][2] == l_local
    assert cap["k_raw_shape"][2] == this_halo + l_local, (rank, cap["k_raw_shape"])
    if layer_type == "sliding_attention":
        assert cap["k_comp_shape"] is None
    else:
        assert cap["k_comp_shape"][2] == (world * l_local) // m, (rank, cap["k_comp_shape"], m)

    grad_nohalo = run(use_halo=False)
    assert torch.isfinite(grad_full).all() and torch.isfinite(grad_nohalo).all()

    # ---- (ii) halo-dk_raw backward wiring ----
    # Dropping the neighbour's halo k_raw rows changes ONLY the gradient the left owner
    # receives at its last `halo` rows.  rank0 (owns the boundary rank1 pulls) must see a
    # difference exactly there and nowhere else; rank1's own shard grad is unchanged.
    if rank == 0 and world > 1:
        assert torch.allclose(grad_full[:, :-CP_HALO], grad_nohalo[:, :-CP_HALO]), "interior grad changed"
        assert not torch.allclose(
            grad_full[:, -CP_HALO:], grad_nohalo[:, -CP_HALO:]
        ), "boundary grad did NOT receive the neighbour's halo-dk_raw -> exchange not wired"
    if rank == world - 1:
        assert torch.allclose(grad_full, grad_nohalo), "last rank's shard grad should be halo-independent"
    dist.destroy_process_group()


@pytest.mark.parametrize(
    ("layer_type", "port"),
    [
        ("compressed_sparse_attention", 29640),
        ("heavily_compressed_attention", 29642),
        ("sliding_attention", 29644),
    ],
)
def test_attention_cp_wiring_and_halo_grad(layer_type, port):
    _run(_attn_cp_worker, 2, layer_type, port=port)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
