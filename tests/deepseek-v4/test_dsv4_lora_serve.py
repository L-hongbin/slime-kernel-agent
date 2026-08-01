"""Offline tests for the DeepSeek-V4 LoRA-adapter serving path.

Covers (no GPU, no sglang server needed):
  1. The megatron-free gate/rollout helpers (slime.utils.lora_utils) + their
     re-export from lora_adapter_sync.
  2. Adapter param detection + megatron→PEFT name mapping for all 6 train targets.
  3. build_lora_adapter_state_dict: exact requires_grad set, PEFT names, values
     preserved, and config (r / lora_alpha / target_modules).
  4. Consistency between the produced adapter target_modules and the sglang
     server --lora-target-modules (after sglang normalization).
  5. sglang-side registration (import-light asserts) + the compressor wkv+gate
     fusion (normalize_wkv_gate) that the served fused wkv_gate module requires.
  6. Launcher (full_loop_smoke.sh) LoRA env/arg wiring + safety gates.

Run: python -m pytest tests/deepseek-v4/test_dsv4_lora_serve.py -q
"""

from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from slime.backends.megatron_utils.update_weight.lora_adapter_sync import SLIME_LORA_ADAPTER_NAME as SYNC_NAME
from slime.backends.megatron_utils.update_weight.lora_adapter_sync import (
    adapter_side_and_base,
    build_lora_adapter_state_dict,
    is_adapter_param_name,
)
from slime.backends.megatron_utils.update_weight.lora_adapter_sync import lora_adapter_name as sync_lora_adapter_name
from slime.backends.megatron_utils.update_weight.lora_adapter_sync import (
    megatron_adapter_name_to_peft,
    peft_target_module_leaf,
)
from slime.backends.megatron_utils.update_weight.lora_adapter_sync import plan_lora_swap as sync_plan_lora_swap
from slime.backends.megatron_utils.update_weight.lora_adapter_sync import rollout_lora_path as sync_rollout_lora_path
from slime.backends.megatron_utils.update_weight.lora_adapter_sync import (
    use_lora_weight_sync as sync_use_lora_weight_sync,
)
from slime.utils.lora_utils import (
    SLIME_LORA_ADAPTER_NAME,
    all_alternating_lora_names,
    lora_adapter_name,
    plan_lora_swap,
    raise_on_failed_lora_load,
    rollout_lora_path,
    use_lora_weight_sync,
)

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 1. gate / rollout helpers
# ---------------------------------------------------------------------------


def test_adapter_name_constant():
    assert SLIME_LORA_ADAPTER_NAME == "slime_lora"
    # lora_adapter_sync re-exports the same objects (single source of truth).
    assert SYNC_NAME is SLIME_LORA_ADAPTER_NAME
    assert sync_rollout_lora_path is rollout_lora_path
    assert sync_use_lora_weight_sync is use_lora_weight_sync
    assert sync_lora_adapter_name is lora_adapter_name
    assert sync_plan_lora_swap is plan_lora_swap


def test_lora_adapter_name_alternates():
    """Consecutive syncs get DIFFERENT names (double-buffer precondition); the same
    name only recurs every 2 syncs (bounded name/id registry on the sglang side)."""
    names = [lora_adapter_name(v) for v in range(6)]
    # all derive from the base prefix
    assert all(n.startswith(SLIME_LORA_ADAPTER_NAME + "_") for n in names)
    # neighbors always differ
    for a, b in zip(names, names[1:], strict=False):
        assert a != b, names
    # exactly two distinct names, alternating with period 2
    assert set(names) == {"slime_lora_0", "slime_lora_1"}
    assert names[0] == names[2] == names[4]
    assert names[1] == names[3] == names[5]


def test_plan_lora_swap_load_before_unload():
    # first sync: only a load (no previous adapter to unload)
    assert plan_lora_swap("slime_lora_1", None) == [("load", "slime_lora_1")]
    # subsequent sync: LOAD the new name FIRST, THEN unload the old one
    plan = plan_lora_swap("slime_lora_0", "slime_lora_1")
    assert plan == [("load", "slime_lora_0"), ("unload", "slime_lora_1")]
    ops = [op for op, _ in plan]
    assert ops.index("load") < ops.index("unload")
    # defensive: never unload the name we just loaded
    assert plan_lora_swap("slime_lora_0", "slime_lora_0") == [("load", "slime_lora_0")]


@pytest.mark.parametrize(
    "relpath",
    [
        "slime/rollout/sglang_rollout.py",
        "slime/rollout/sglang_streaming_rollout.py",
        "examples/kernel_agent/generate_with_cuda_agent.py",
    ],
)
def test_all_generate_paths_attach_active_lora(relpath):
    """Every rollout /generate payload builder must route lora_path to the ACTIVE
    (alternating) adapter name, else the adapter-sync path silently serves
    the base model on that path (the V4 RL run uses the custom cuda-agent generate,
    NOT the default one)."""
    text = (REPO / relpath).read_text()
    assert "_rollout_lora_path(args, state.active_lora_name)" in text, relpath
    assert 'payload["lora_path"] = lora_path' in text, relpath


def test_all_alternating_lora_names():
    """Both alternating names, used to clear stale adapters before the first sync."""
    assert all_alternating_lora_names() == {"slime_lora_0", "slime_lora_1"}
    assert all_alternating_lora_names() == {lora_adapter_name(v) for v in range(2)}


class _FakeRemote:
    """Stands in for ``engine.<method>`` — ``.remote(**kw)`` records the call and
    mutates the fake engine's resident-adapter set (like sglang's mem-pool)."""

    def __init__(self, engine, kind):
        self.engine = engine
        self.kind = kind

    def remote(self, *, lora_name, **_ignored):
        self.engine.calls.append((self.kind, lora_name))
        if self.kind == "load":
            self.engine.resident.add(lora_name)
            return {"success": self.engine.load_ok}
        # unload (raises for an absent name, like sglang's 4xx — the first-sync
        # best-effort clear must swallow that).
        if lora_name not in self.engine.resident:
            raise RuntimeError(f"adapter {lora_name} not found")
        self.engine.resident.discard(lora_name)
        return {"success": True}


class _FakeEngine:
    def __init__(self, resident=()):
        self.calls: list[tuple[str, str]] = []
        self.resident: set[str] = set(resident)
        self.load_ok = True
        self.load_lora_adapter_from_tensors = _FakeRemote(self, "load")
        self.unload_lora_adapter = _FakeRemote(self, "unload")


def _make_updater(monkeypatch):
    uwd = pytest.importorskip("slime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    # ``ray.get`` -> identity so the swap runs synchronously against fake engines.
    monkeypatch.setattr(uwd.ray, "get", lambda x: x)
    obj = uwd.UpdateWeightFromDistributed.__new__(uwd.UpdateWeightFromDistributed)
    obj.weight_version = 0
    obj._lora_prev_adapter_name = None
    return obj


def test_lora_swap_first_sync_clears_stale_then_loads(monkeypatch):
    """First sync after a (re)start: best-effort clear BOTH alternating names
    (a prior run may have left them resident) BEFORE loading, so the load never
    collides and the mem-pool has a free slot. No unload-old (no previous)."""
    obj = _make_updater(monkeypatch)
    eng = _FakeEngine(resident={"slime_lora_0", "slime_lora_1"})  # stale from prior run

    obj.weight_version = 1  # update_weights() increments before the swap
    obj._apply_lora_adapter_swap([eng], {"k": 1}, {"r": 4})

    kinds = [k for k, _ in eng.calls]
    names = eng.calls
    # both stale names unloaded, then the new name loaded
    assert ("unload", "slime_lora_0") in names and ("unload", "slime_lora_1") in names
    load_idx = kinds.index("load")
    assert all(i < load_idx for i, k in enumerate(kinds) if k == "unload"), eng.calls
    assert names[load_idx] == ("load", "slime_lora_1")  # version 1 -> parity 1
    # exactly the new adapter resident; prev advanced
    assert eng.resident == {"slime_lora_1"}
    assert obj._lora_prev_adapter_name == "slime_lora_1"


def test_lora_swap_steady_state_load_before_unload(monkeypatch):
    """Second sync onward: load NEW before unloading OLD; no stale-clear."""
    obj = _make_updater(monkeypatch)
    eng = _FakeEngine(resident={"slime_lora_1"})
    obj._lora_prev_adapter_name = "slime_lora_1"

    obj.weight_version = 2  # parity 0
    obj._apply_lora_adapter_swap([eng], {"k": 1}, {"r": 4})

    # The swap itself only LOADS; the old adapter's unload is DEFERRED until
    # after continue_generation (sglang's unload waits for request refs to
    # drain, impossible while paused — the r7f/r7g sync hang, 2026-07-10).
    assert eng.calls == [("load", "slime_lora_0")]
    assert eng.resident == {"slime_lora_0", "slime_lora_1"}
    assert obj._lora_prev_adapter_name == "slime_lora_0"
    assert obj._lora_pending_unload == "slime_lora_1"

    # After resume, the deferred unload completes the swap.
    obj.rollout_engines = [eng]
    obj._finish_lora_adapter_swap()
    assert eng.calls[-1] == ("unload", "slime_lora_1")
    assert eng.resident == {"slime_lora_0"}
    assert obj._lora_pending_unload is None


def test_lora_swap_failed_load_does_not_unload_old(monkeypatch):
    """A failed load must raise and leave the OLD adapter resident + prev-name
    unchanged (never advance past a name that never went live)."""
    obj = _make_updater(monkeypatch)
    eng = _FakeEngine(resident={"slime_lora_1"})
    eng.load_ok = False
    obj._lora_prev_adapter_name = "slime_lora_1"

    obj.weight_version = 2
    with pytest.raises(RuntimeError, match="failed on"):
        obj._apply_lora_adapter_swap([eng], {"k": 1}, {"r": 4})

    # old adapter still resident (never unloaded), prev-name not advanced
    assert "slime_lora_1" in eng.resident
    assert ("unload", "slime_lora_1") not in eng.calls
    assert obj._lora_prev_adapter_name == "slime_lora_1"


def test_lora_swap_multi_cycle_never_collides(monkeypatch):
    """Drive several full swaps on a fresh engine; the name loaded each sync is
    never one still resident, and peak residency stays <= 2."""
    obj = _make_updater(monkeypatch)
    eng = _FakeEngine()
    obj.rollout_engines = [eng]
    for version in range(1, 7):
        obj.weight_version = version
        before = set(eng.resident)
        new_name = lora_adapter_name(version)
        assert new_name not in before, (version, before)
        obj._apply_lora_adapter_swap([eng], {"k": 1}, {"r": 4})
        obj._finish_lora_adapter_swap()  # post-resume deferred unload
        assert eng.resident == {new_name}
        assert len(eng.resident) <= 2


def test_lora_swap_pending_unload_retried_next_swap(monkeypatch):
    """If the deferred unload never ran (rank0 crash / timeout), the NEXT swap
    flushes the pending unload BEFORE loading, so residency stays <= 2 and the
    load never collides with a resident name."""
    obj = _make_updater(monkeypatch)
    eng = _FakeEngine()
    obj.rollout_engines = [eng]
    for version in range(1, 7):
        obj.weight_version = version
        new_name = lora_adapter_name(version)
        obj._apply_lora_adapter_swap([eng], {"k": 1}, {"r": 4})
        # deliberately NEVER call _finish_lora_adapter_swap: the swap itself must
        # flush the pending unload BEFORE loading, so the load never collides
        # with a resident copy of the same name and residency stays <= 2.
        assert new_name in eng.resident
        assert len(eng.resident) <= 2, (version, eng.resident)
        same_name_events = [k for k, n in eng.calls if n == new_name]
        if same_name_events.count("load") > 1:
            # a re-load of an alternating name must have been preceded by its unload
            last_unload = max(i for i, c in enumerate(eng.calls) if c == ("unload", new_name))
            last_load = max(i for i, c in enumerate(eng.calls) if c == ("load", new_name))
            assert last_unload < last_load, eng.calls


def test_raise_on_failed_lora_load():
    """A failed load must raise (so the caller never unloads the still-serving old
    adapter); success / None results (non-leader ranks) pass."""
    # all success -> no raise
    raise_on_failed_lora_load([{"success": True}, {"success": True}], "slime_lora_1")
    # None (non-leader no-op) treated as success
    raise_on_failed_lora_load([None, {"success": True}], "slime_lora_1")
    # dict without "success" key defaults to success (back-compat)
    raise_on_failed_lora_load([{"loaded": 5}], "slime_lora_1")
    # any success=False -> raise
    with pytest.raises(RuntimeError, match="failed on 1/2"):
        raise_on_failed_lora_load([{"success": True}, {"success": False}], "slime_lora_0")


def test_plan_lora_swap_full_alternation_sequence():
    """Drive the exact updater loop across several syncs: at every sync the new
    name is loaded before the old is unloaded, and the name we load is never one
    still resident from the previous sync."""
    prev = None
    resident: set[str] = set()
    for version in range(1, 6):
        new_name = lora_adapter_name(version)
        # the name we are about to load must NOT currently be resident
        assert new_name not in resident, (version, resident)
        for op, name in plan_lora_swap(new_name, prev):
            if op == "load":
                resident.add(name)
                # peak residency during the swap is at most 2 (needs 2 slots)
                assert len(resident) <= 2, (version, resident)
            else:
                assert name in resident
                resident.discard(name)
        # after the swap exactly the new adapter is resident
        assert resident == {new_name}, (version, resident)
        prev = new_name


def test_use_lora_weight_sync_flag_only():
    # default OFF
    assert use_lora_weight_sync(Namespace()) is False
    assert use_lora_weight_sync(Namespace(use_lora_weight_sync=False)) is False
    # flag ON
    assert use_lora_weight_sync(Namespace(use_lora_weight_sync=True)) is True


def test_rollout_lora_path_tracks_active_name():
    """Flag-off byte-identical: no lora_path attached to the /generate payload.
    Flag-on: the payload routes to the ACTIVE (alternating) adapter name the engine
    reports, and stays unset (base-only) until the first adapter is loaded."""
    # flag off -> None regardless of any active name
    assert rollout_lora_path(Namespace()) is None
    assert rollout_lora_path(Namespace(), active_name="slime_lora_1") is None
    # flag on -> the active name the engine currently serves
    on = Namespace(use_lora_weight_sync=True)
    assert rollout_lora_path(on, active_name="slime_lora_1") == "slime_lora_1"
    assert rollout_lora_path(on, active_name="slime_lora_0") == "slime_lora_0"
    # flag on but no adapter loaded yet -> base-only (zero-init adapter, delta 0)
    assert rollout_lora_path(on) is None
    assert rollout_lora_path(on, active_name=None) is None


def test_rollout_lora_path_supports_static_preloaded_eval_adapter():
    """Static eval serving must route every request to the named adapter without
    pretending that the training-time hot-sync path is enabled."""
    static = Namespace(use_lora_weight_sync=False, rollout_lora_name="eval_step_20")
    assert rollout_lora_path(static) == "eval_step_20"
    # An explicitly configured static checkpoint wins over stale dynamic state.
    assert rollout_lora_path(static, active_name="slime_lora_1") == "eval_step_20"


# ---------------------------------------------------------------------------
# 2. adapter detection + name mapping
# ---------------------------------------------------------------------------

# The 6 trainable V4 LoRA targets (megatron leaf → served PEFT leaf).
# Attention compressor kv_proj and gate_proj are physically distinct on the
# train side; sglang later fuses them into one wkv_gate module (see §5).
_MEGATRON_BASE = "module.module.decoder.layers.3.self_attn"
_TARGET_CASES = [
    (f"{_MEGATRON_BASE}.q_a_proj", "layers.3.attn.wq_a", "wq_a"),
    (f"{_MEGATRON_BASE}.q_b_proj", "layers.3.attn.wq_b", "wq_b"),
    (f"{_MEGATRON_BASE}.kv_proj", "layers.3.attn.wkv", "wkv"),
    (f"{_MEGATRON_BASE}.o_b_proj", "layers.3.attn.wo_b", "wo_b"),
    (f"{_MEGATRON_BASE}.compressor.kv_proj", "layers.3.attn.compressor.wkv", "wkv"),
    (f"{_MEGATRON_BASE}.compressor.gate_proj", "layers.3.attn.compressor.wgate", "wgate"),
]


def test_adapter_side_and_base():
    assert adapter_side_and_base("x.q_a_proj.linear_in.weight") == ("A", "x.q_a_proj")
    assert adapter_side_and_base("x.q_a_proj.linear_out.weight") == ("B", "x.q_a_proj")
    assert adapter_side_and_base("x.q_a_proj.weight") == (None, None)
    assert is_adapter_param_name("x.q_a_proj.linear_in.weight") is True
    assert is_adapter_param_name("x.q_a_proj.weight") is False
    # A frozen base param must not be treated as an adapter.
    assert is_adapter_param_name(f"{_MEGATRON_BASE}.o_a_proj.weight") is False


@pytest.mark.parametrize("base,serving_base,leaf", _TARGET_CASES)
def test_megatron_adapter_name_to_peft(base, serving_base, leaf):
    for side, suffix in (("A", ".linear_in.weight"), ("B", ".linear_out.weight")):
        name = base + suffix
        peft = megatron_adapter_name_to_peft(None, name, torch.empty(0))
        assert peft == f"base_model.model.{serving_base}.lora_{side}.weight"
        assert peft_target_module_leaf(peft) == leaf


def test_megatron_adapter_name_to_peft_non_adapter_is_none():
    assert megatron_adapter_name_to_peft(None, f"{_MEGATRON_BASE}.q_a_proj.weight") is None


# ---------------------------------------------------------------------------
# 3. build_lora_adapter_state_dict
# ---------------------------------------------------------------------------


def _fake_adapter_tensors(rank: int = 16, hidden: int = 32):
    """One (name, tensor) pair per adapter side for each of the 6 targets."""
    out = []
    for base, _serving, _leaf in _TARGET_CASES:
        a = torch.randn(rank, hidden)  # lora_A: [r, in]
        b = torch.randn(hidden, rank)  # lora_B: [out, r]
        out.append((base + ".linear_in.weight", a))
        out.append((base + ".linear_out.weight", b))
    return out


def test_build_lora_adapter_state_dict_names_values_config():
    named = _fake_adapter_tensors(rank=16, hidden=32)
    state_dict, config = build_lora_adapter_state_dict(None, named, scale=2.0)

    # exact PEFT key set (2 sides × 6 targets); compressor kv/gate keep distinct
    # keys (wkv vs wgate) — sglang fuses them into wkv_gate at load time.
    expected_keys = set()
    for _base, serving_base, _leaf in _TARGET_CASES:
        expected_keys.add(f"base_model.model.{serving_base}.lora_A.weight")
        expected_keys.add(f"base_model.model.{serving_base}.lora_B.weight")
    assert set(state_dict.keys()) == expected_keys

    # values preserved exactly (no copy/scale mutation of the tensors themselves).
    by_src = {n: t for n, t in named}
    for base, serving_base, _leaf in _TARGET_CASES:
        assert torch.equal(
            state_dict[f"base_model.model.{serving_base}.lora_A.weight"],
            by_src[base + ".linear_in.weight"],
        )

    # config: r read from lora_A leading dim; lora_alpha = round(scale*r).
    assert config["peft_type"] == "lora"
    assert config["r"] == 16
    assert config["lora_alpha"] == 32  # scale 2.0 * r 16
    assert config["lora_dropout"] == 0.0
    assert config["bias"] == "none"
    # target_modules = the served leaf names (compressor wkv+gate dedupe wkv).
    assert config["target_modules"] == sorted({"wq_a", "wq_b", "wkv", "wo_b", "wgate"})


def test_build_lora_adapter_state_dict_rejects_mixed_rank():
    named = [
        (f"{_MEGATRON_BASE}.q_a_proj.linear_in.weight", torch.randn(16, 32)),
        (f"{_MEGATRON_BASE}.q_a_proj.linear_out.weight", torch.randn(32, 16)),
        (f"{_MEGATRON_BASE}.q_b_proj.linear_in.weight", torch.randn(8, 32)),  # rank 8
        (f"{_MEGATRON_BASE}.q_b_proj.linear_out.weight", torch.randn(32, 8)),
    ]
    with pytest.raises(ValueError, match="inconsistent ranks"):
        build_lora_adapter_state_dict(None, named, scale=2.0)


def test_build_lora_adapter_state_dict_alpha_scaling():
    named = _fake_adapter_tensors(rank=8, hidden=16)
    _sd, config = build_lora_adapter_state_dict(None, named, scale=4.0)
    assert config["r"] == 8
    assert config["lora_alpha"] == 32  # 4.0 * 8


# ---------------------------------------------------------------------------
# 4/5. sglang-side registration + compressor fusion (import-light)
# ---------------------------------------------------------------------------


def _sglang_lora_utils():
    return pytest.importorskip("sglang.srt.lora.utils")


def test_sglang_supported_and_replicated_registration():
    utils = _sglang_lora_utils()
    from sglang.srt.utils.common import SUPPORTED_LORA_TARGET_MODULES

    v4_leaves = {"wq_a", "wkv", "wq_b", "wo_b", "wkv_gate"}
    assert v4_leaves <= set(SUPPORTED_LORA_TARGET_MODULES)
    # All 6 attention/compressor targets are replicated under attn-tp==1 so their
    # mem_pool buffers stay full (no /tp sharding).
    assert v4_leaves <= set(utils.REPLICATED_LINEAR_LORA_NAMES)
    assert v4_leaves <= set(utils._KNOWN_LORA_TARGET_MODULES)


def test_sglang_normalize_wgate_to_wkv_gate():
    utils = _sglang_lora_utils()
    # The adapter's target_modules (from slime) normalize to the server set:
    # wgate -> wkv_gate; the standalone attention wkv stays wkv.
    adapter_targets = {"wq_a", "wq_b", "wkv", "wo_b", "wgate"}
    normalized = utils.get_normalized_target_modules(adapter_targets)
    assert normalized == {"wq_a", "wq_b", "wkv", "wo_b", "wkv_gate"}
    # server --lora-target-modules token set covers the normalized adapter set.
    server_targets = utils.get_normalized_target_modules({"wq_a", "wkv", "wq_b", "wo_b", "wkv_gate"})
    assert normalized <= server_targets


def test_sglang_lora_wrapper_attr_proxy():
    """A LoRA wrapper must transparently proxy base-layer attributes (e.g.
    reduce_results, output_size) it does not itself define — the V4 empty-batch
    path reads wo_b.reduce_results off the (now wrapped) module — WITHOUT the
    proxy breaking nn.Module.register_parameter's hasattr() collision guard (a
    real `weight` Parameter previously raised KeyError("attribute 'weight'
    already exists") during __init__)."""
    layers = pytest.importorskip("sglang.srt.lora.layers")
    import torch.nn as nn

    class FakeBase(nn.Module):
        def __init__(self):
            super().__init__()
            # a real Parameter `weight` is what tripped register_parameter.
            self.weight = nn.Parameter(torch.randn(8, 4))
            self.bias = None
            self.reduce_results = False
            self.output_size = 8
            self.skip_bias_add = False

    w = layers.BaseLayerWithLoRA(FakeBase(), lora_backend=None)  # must not raise
    assert isinstance(w.weight, nn.Parameter)  # wrapper's own registered param
    assert w.reduce_results is False  # delegated to base_layer
    assert w.output_size == 8
    assert w.set_lora is False  # wrapper's own attribute wins
    with pytest.raises(AttributeError):
        _ = w.definitely_not_an_attribute

    # The tp==1 replicated wrap path (ReplicatedLinearWithLoRA) also constructs.
    r = layers.ReplicatedLinearWithLoRA(FakeBase(), lora_backend=None)
    assert r.output_size == 8
    assert r.reduce_results is False


def test_engine_load_lora_disk_transport_and_version_bump(tmp_path, monkeypatch):
    """The engine loads the adapter via the DISK-PATH transport: it writes a PEFT
    dir (adapter_model.safetensors + adapter_config.json) and POSTs /load_lora_adapter
    with lora_path=<dir> (NOT the from-tensors IPC path, which has the multi-worker
    shm unlink race). It bumps the version + records the active name + retains the
    dir only on success; unload frees the dir and clears the active name."""
    engmod = pytest.importorskip("slime.backends.sglang_utils.sglang_engine")
    SGLangEngine = engmod.SGLangEngine

    obj = SGLangEngine.__new__(SGLangEngine)
    obj.node_rank = 0
    obj._active_lora_name = None
    obj._lora_dirs = {}
    # single-node engine (all workers co-located): 8 gpus / 8 per node = 1 node.
    obj.num_gpus_per_engine = 8
    obj.args = Namespace(num_gpus_per_node=8, rollout_num_gpus_per_engine=8)

    calls = []

    def fake_make_request(endpoint, payload=None, timeout=None):
        calls.append((endpoint, payload))
        if endpoint == "load_lora_adapter":
            return {"success": fake_make_request.load_ok}
        return {"ok": True}

    obj._make_request = fake_make_request

    # Route the real writer at the pytest tmp dir so we exercise the actual
    # safetensors + adapter_config.json materialization without touching /dev/shm.
    import json as _json
    import os as _os

    from safetensors.torch import save_file as _save_file

    def write_dir(lora_name, tensors, config_dict):
        d = tmp_path / f"{lora_name}_{len(calls)}"
        d.mkdir()
        _save_file(
            {k: v.detach().cpu().contiguous() for k, v in tensors.items()},
            str(d / "adapter_model.safetensors"),
        )
        (d / "adapter_config.json").write_text(_json.dumps(config_dict))
        return str(d)

    monkeypatch.setattr(obj, "_write_lora_adapter_dir", write_dir)

    tensors = {"base_model.model.layers.0.attn.wq_a.lora_A.weight": torch.randn(4, 8)}
    cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["wq_a"]}

    # success -> POSTs /load_lora_adapter with a path, version bumped, name recorded,
    # dir retained (implicit-reload safe).
    fake_make_request.load_ok = True
    calls.clear()
    obj.load_lora_adapter_from_tensors("slime_lora_1", tensors, cfg, weight_version="7")
    load_calls = [(e, p) for e, p in calls if e == "load_lora_adapter"]
    assert len(load_calls) == 1 and "lora_path" in load_calls[0][1]
    assert _os.path.isdir(load_calls[0][1]["lora_path"])
    assert _os.path.exists(_os.path.join(load_calls[0][1]["lora_path"], "adapter_model.safetensors"))
    assert any(e == "update_weight_version" for e, _ in calls)
    assert obj.get_active_lora_name() == "slime_lora_1"
    assert "slime_lora_1" in obj._lora_dirs

    # failure -> version NOT bumped, active name unchanged, dir NOT retained/leaked.
    fake_make_request.load_ok = False
    calls.clear()
    obj.load_lora_adapter_from_tensors("slime_lora_0", tensors, cfg, weight_version="8")
    assert not any(e == "update_weight_version" for e, _ in calls)
    assert obj.get_active_lora_name() == "slime_lora_1"
    assert "slime_lora_0" not in obj._lora_dirs

    # unloading the OLD (non-active) name leaves the active name intact.
    obj.unload_lora_adapter("slime_lora_0")
    assert obj.get_active_lora_name() == "slime_lora_1"
    # unloading the ACTIVE name clears it AND frees its dir.
    obj.unload_lora_adapter("slime_lora_1")
    assert obj.get_active_lora_name() is None
    assert obj._lora_dirs == {}


def test_engine_lora_disk_transport_rejects_multinode():
    """Multi-node engines cannot use a node-0-local tmpfs path (workers on other
    nodes can't read it) — the load must fail loudly rather than half-load."""
    engmod = pytest.importorskip("slime.backends.sglang_utils.sglang_engine")
    SGLangEngine = engmod.SGLangEngine
    obj = SGLangEngine.__new__(SGLangEngine)
    obj.node_rank = 0
    obj._active_lora_name = None
    obj._lora_dirs = {}
    obj.num_gpus_per_engine = 16  # 16 gpus / 8 per node = 2 nodes
    obj.args = Namespace(num_gpus_per_node=8, rollout_num_gpus_per_engine=16)
    obj._make_request = lambda *a, **k: {"success": True}
    with pytest.raises(RuntimeError, match="single-node"):
        obj.load_lora_adapter_from_tensors("slime_lora_1", {"x": torch.randn(4, 8)}, {"r": 4}, weight_version="1")


def test_sglang_normalize_wkv_gate_fusion():
    """The served compressor is a single fused wkv_gate; the separate kv/gate
    adapter weights must fuse (kv-then-gate along dim 0) for both A and B."""
    lora_mod = pytest.importorskip("sglang.srt.lora.lora")
    LoRAAdapter = lora_mod.LoRAAdapter
    obj = LoRAAdapter.__new__(LoRAAdapter)  # no heavy __init__

    pre = "base_model.model.layers.3.attn"
    a_kv = torch.randn(16, 32)
    a_gate = torch.randn(16, 32)
    b_kv = torch.randn(20, 16)
    b_gate = torch.randn(20, 16)
    # attention wkv (standalone) must NOT be fused into the compressor module.
    a_attn = torch.randn(16, 32)
    weights = {
        f"{pre}.compressor.wkv.lora_A.weight": a_kv,
        f"{pre}.compressor.wgate.lora_A.weight": a_gate,
        f"{pre}.compressor.wkv.lora_B.weight": b_kv,
        f"{pre}.compressor.wgate.lora_B.weight": b_gate,
        f"{pre}.wkv.lora_A.weight": a_attn,
    }
    obj.normalize_wkv_gate(list(weights.keys()), weights)

    fused_a = f"{pre}.compressor.wkv_gate.lora_A.weight"
    fused_b = f"{pre}.compressor.wkv_gate.lora_B.weight"
    assert fused_a in weights and fused_b in weights
    # kv rows first, then gate rows.
    assert torch.equal(weights[fused_a], torch.cat([a_kv, a_gate], dim=0))
    assert torch.equal(weights[fused_b], torch.cat([b_kv, b_gate], dim=0))
    # originals consumed; standalone attention wkv untouched.
    assert f"{pre}.compressor.wkv.lora_A.weight" not in weights
    assert f"{pre}.compressor.wgate.lora_A.weight" not in weights
    assert f"{pre}.wkv.lora_A.weight" in weights
    assert torch.equal(weights[f"{pre}.wkv.lora_A.weight"], a_attn)


# ---------------------------------------------------------------------------
# 6. launcher wiring + safety gates
# ---------------------------------------------------------------------------


def _full_loop_text():
    return (REPO / "scripts" / "dsv4" / "full_loop_smoke.sh").read_text()


def test_launcher_lora_args():
    text = _full_loop_text()
    # wrapper input args (default OFF)
    assert "--sglang-enable-lora)" in text
    assert "--use-lora-weight-sync)" in text
    # unfuse wq_a/wkv when LoRA on
    assert "export SGLANG_OPT_FUSE_WQA_WKV=0" in text
    # sglang server args
    assert "--sglang-enable-lora" in text
    assert '--sglang-max-lora-rank "${SGLANG_MAX_LORA_RANK}"' in text
    assert "--sglang-lora-target-modules ${SGLANG_LORA_TARGET_MODULES}" in text
    # train-side sync flag
    assert "--use-lora-weight-sync" in text
    # forwarded into the runtime env of the engine actors: the launcher exports
    # SGLANG_OPT_FUSE_WQA_WKV (asserted above) and the cluster lib's SGLANG_*
    # prefix sweep transports every exported SGLANG_* var into the direct
    # driver's --train-env-vars payload.
    assert "TRAIN_ENV_VARS_JSON=$(dsv4_build_train_env_vars_json)" in text
    lib = (REPO / "scripts" / "dsv4" / "_dsv4_cluster_lib.sh").read_text()
    assert "for prefix in V4_ SGLANG_ SLIME_DEBUG_ SLIME_PATCH_" in lib


def test_launcher_lora_safety_gates():
    text = _full_loop_text()
    # Adapter weight sync without an enabled LoRA engine is fatal.
    assert re.search(r"USE_LORA_WEIGHT_SYNC.*==.*1.*&&.*SGLANG_ENABLE_LORA.*!=.*1", text)
    # The single DSpark runtime warns if LoRA uses an unvalidated spec algorithm.
    assert '"${SGLANG_SPECULATIVE_ALGORITHM}" != "DSPARK"' in text
    assert "validated only with DSPARK speculative decoding" in text


def test_launcher_max_loras_per_batch_default_and_guard():
    text = _full_loop_text()
    # The alternating swap needs two resident adapters -> default 2 when weight
    # sync is on, else 1 (byte-identical to before when off).
    assert "SGLANG_MAX_LORAS_PER_BATCH=${SGLANG_MAX_LORAS_PER_BATCH:-2}" in text
    assert "SGLANG_MAX_LORAS_PER_BATCH=${SGLANG_MAX_LORAS_PER_BATCH:-1}" in text
    # Fatal if the sync is on but only one slot is available.
    assert re.search(r"USE_LORA_WEIGHT_SYNC.*==.*1.*&&.*SGLANG_MAX_LORAS_PER_BATCH.*-lt.*2", text)
