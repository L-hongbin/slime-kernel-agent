"""CPU tests for canonical-source ownership in V4 LoRA weight sync."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# CI invokes changed tests as ``python tests/<file>.py``.  Put this checkout
# ahead of any installed slime package before the module-under-test import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slime.backends.megatron_utils.update_weight import update_weight_from_distributed as uwd

NUM_GPUS = 0

_ADAPTER_NAMES = (
    "module.module.layers.0.self_attention.linear_q_proj.linear_in.weight",
    "module.module.layers.0.self_attention.linear_q_proj.linear_out.weight",
)


class _Param:
    def __init__(self, *, requires_grad: bool):
        self.requires_grad = requires_grad


class _GatheredTensor:
    """Records the D2H path without requiring a CUDA tensor."""

    def __init__(self, name: str, trace: list[tuple], value: int):
        self.name = name
        self.trace = trace
        self.value = value

    def detach(self):
        self.trace.append(("detach", self.name))
        return self

    def to(self, device: str, *, copy: bool):
        self.trace.append(("to", self.name, device, copy))
        return torch.tensor([self.value], dtype=torch.bfloat16)


def _new_updater(*, is_pp_src_rank: bool):
    updater = uwd.UpdateWeightFromDistributed.__new__(uwd.UpdateWeightFromDistributed)
    updater.args = SimpleNamespace()
    updater.model = []
    updater._is_pp_src_rank = is_pp_src_rank
    updater.rollout_engines = [object()]
    updater.update_weight_metrics = {}
    updater.weight_version = 1
    return updater


def _install_connect_rank_fakes(monkeypatch, *, dp_rank: int, tp_rank: int, pp_rank: int):
    monkeypatch.setattr(
        uwd.mpu,
        "get_data_parallel_rank",
        lambda *, with_context_parallel: dp_rank,
    )
    monkeypatch.setattr(uwd.mpu, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(uwd.mpu, "get_pipeline_model_parallel_rank", lambda: pp_rank)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dp_rank,tp_rank,expected_source",
    [
        (0, 0, True),
        (1, 0, False),
        (0, 1, False),
    ],
)
def test_lora_connect_binds_engines_and_preserves_source_ownership_without_native_nccl(
    monkeypatch,
    dp_rank,
    tp_rank,
    expected_source,
):
    updater = _new_updater(is_pp_src_rank=False)
    updater.args = SimpleNamespace(use_lora_weight_sync=True)
    updater._model_update_groups = None
    engines = [object(), object()]
    lock = object()
    _install_connect_rank_fakes(monkeypatch, dp_rank=dp_rank, tp_rank=tp_rank, pp_rank=3)
    monkeypatch.setattr(
        uwd,
        "connect_rollout_engines_from_distributed",
        lambda *args, **kwargs: pytest.fail("LoRA adapter-only connect initialized native NCCL"),
    )

    updater.connect_rollout_engines(engines, lock, engine_gpu_counts=[4, 4], engine_gpu_offsets=[0, 4])

    assert updater.rollout_engines is engines
    assert updater.rollout_engine_lock is lock
    assert updater._engine_gpu_counts == [4, 4]
    assert updater._is_pp_src_rank is expected_source
    assert updater._model_update_groups is None
    if expected_source:
        assert updater._group_name == "slime-pp_3"


@pytest.mark.unit
def test_lora_reconnect_destroys_preexisting_native_group_without_recreating_it(monkeypatch):
    updater = _new_updater(is_pp_src_rank=False)
    updater.args = SimpleNamespace(use_lora_weight_sync=True)
    old_group = object()
    updater._model_update_groups = old_group
    engines = [object()]
    calls = []
    _install_connect_rank_fakes(monkeypatch, dp_rank=0, tp_rank=0, pp_rank=1)

    def fake_disconnect(args, group_name, group, bound_engines):
        calls.append((args, group_name, group, bound_engines))

    monkeypatch.setattr(uwd, "disconnect_rollout_engines_from_distributed", fake_disconnect)
    monkeypatch.setattr(
        uwd,
        "connect_rollout_engines_from_distributed",
        lambda *args, **kwargs: pytest.fail("LoRA reconnect recreated native NCCL"),
    )

    updater.connect_rollout_engines(engines, object(), engine_gpu_counts=[8])

    assert calls == [(updater.args, "slime-pp_1", old_group, engines)]
    assert updater._model_update_groups is None


@pytest.mark.unit
def test_full_weight_connect_keeps_native_nccl_reconnect_lifecycle(monkeypatch):
    updater = _new_updater(is_pp_src_rank=False)
    updater.args = SimpleNamespace(use_lora_weight_sync=False)
    old_group = object()
    new_group = object()
    updater._model_update_groups = old_group
    engines = [object(), object()]
    lock = object()
    calls = []
    _install_connect_rank_fakes(monkeypatch, dp_rank=0, tp_rank=0, pp_rank=2)

    def fake_disconnect(args, group_name, group, bound_engines):
        calls.append(("disconnect", args, group_name, group, bound_engines))

    def fake_connect(args, group_name, bound_engines, *, engine_gpu_counts):
        calls.append(("connect", args, group_name, bound_engines, engine_gpu_counts))
        return new_group

    monkeypatch.setattr(uwd, "disconnect_rollout_engines_from_distributed", fake_disconnect)
    monkeypatch.setattr(uwd, "connect_rollout_engines_from_distributed", fake_connect)

    updater.connect_rollout_engines(engines, lock, engine_gpu_counts=[2, 6], engine_gpu_offsets=[0, 2])

    assert calls == [
        ("disconnect", updater.args, "slime-pp_2", old_group, engines),
        ("connect", updater.args, "slime-pp_2", engines, [2, 6]),
    ]
    assert updater.rollout_engines is engines
    assert updater.rollout_engine_lock is lock
    assert updater._is_pp_src_rank is True
    assert updater._model_update_groups is new_group


def _install_adapter_param_fakes(monkeypatch, trace: list[tuple]):
    entries = [
        (_ADAPTER_NAMES[0], _Param(requires_grad=True)),
        ("module.module.layers.0.self_attention.linear_q_proj.weight", _Param(requires_grad=True)),
        (_ADAPTER_NAMES[1], _Param(requires_grad=True)),
        (
            "module.module.layers.0.self_attention.linear_k_proj.linear_in.weight",
            _Param(requires_grad=False),
        ),
    ]
    monkeypatch.setattr(uwd, "named_params_and_buffers", lambda _args, _model: entries)

    def fake_all_gather(name, _param):
        trace.append(("all_gather_param", name))
        return _GatheredTensor(name, trace, len(trace))

    monkeypatch.setattr(uwd, "all_gather_param", fake_all_gather)


@pytest.mark.unit
def test_only_canonical_pp_sources_materialize_cpu_payloads_without_changing_collective_order(monkeypatch):
    """Simulate PP2 x TP2 x four DP/CP/EP replicas (16 ranks).

    Megatron folds the DP/CP/EP replica coordinate into the native
    DP-with-context rank used by ``_is_pp_src_rank``.  Each PP stage therefore
    has one source at TP=0/replica=0, while every rank must execute the same
    adapter ``all_gather_param`` sequence.
    """

    source_count_by_pp = {0: 0, 1: 0}
    collective_orders = []
    for pp_rank in range(2):
        for tp_rank in range(2):
            for replica_rank in range(4):
                is_source = tp_rank == 0 and replica_rank == 0
                source_count_by_pp[pp_rank] += int(is_source)
                trace: list[tuple] = []
                _install_adapter_param_fakes(monkeypatch, trace)
                payload = _new_updater(is_pp_src_rank=is_source)._collect_local_adapter_named_tensors()

                collective_order = [event[1] for event in trace if event[0] == "all_gather_param"]
                collective_orders.append(collective_order)
                assert collective_order == list(_ADAPTER_NAMES)
                d2h_names = [event[1] for event in trace if event[0] == "to"]
                if is_source:
                    assert [name for name, _tensor in payload] == list(_ADAPTER_NAMES)
                    assert d2h_names == list(_ADAPTER_NAMES)
                else:
                    assert payload == []
                    assert d2h_names == []

    assert source_count_by_pp == {0: 1, 1: 1}
    assert all(order == collective_orders[0] for order in collective_orders)


@pytest.mark.unit
@pytest.mark.parametrize("is_source", [False, True])
def test_routed_expert_lora_fails_before_any_shard_collective(monkeypatch, is_source):
    entries = [
        (_ADAPTER_NAMES[0], _Param(requires_grad=True)),
        (
            "module.module.layers.0.mlp.experts.linear_fc1.linear_in.weight",
            _Param(requires_grad=True),
        ),
    ]
    monkeypatch.setattr(uwd, "named_params_and_buffers", lambda _args, _model: entries)
    monkeypatch.setattr(
        uwd,
        "all_gather_param",
        lambda *args, **kwargs: pytest.fail("routed-expert preflight started a shard collective"),
    )

    updater = _new_updater(is_pp_src_rank=is_source)
    with pytest.raises(RuntimeError, match=r"routed-expert LoRA.*EP shards require"):
        updater._collect_local_adapter_named_tensors()


@pytest.mark.unit
def test_replicated_shared_expert_lora_is_not_mistaken_for_routed_expert(monkeypatch):
    shared_name = "module.module.layers.0.mlp.shared_experts.down_proj.linear_in.weight"
    trace: list[tuple] = []
    monkeypatch.setattr(
        uwd,
        "named_params_and_buffers",
        lambda _args, _model: [(shared_name, _Param(requires_grad=True))],
    )

    def fake_all_gather(name, _param):
        trace.append(("all_gather_param", name))
        return _GatheredTensor(name, trace, 1)

    monkeypatch.setattr(uwd, "all_gather_param", fake_all_gather)

    payload = _new_updater(is_pp_src_rank=True)._collect_local_adapter_named_tensors()

    assert [name for name, _tensor in payload] == [shared_name]
    assert trace[0] == ("all_gather_param", shared_name)


def _install_update_fakes(monkeypatch, updater, *, rank: int, pp_size: int, local):
    updater._get_v4_lora_base_scales = lambda: {"base.weight": 2.0}
    updater._collect_local_adapter_named_tensors = lambda: local
    monkeypatch.setattr(uwd, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(uwd.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(uwd.mpu, "get_pipeline_model_parallel_world_size", lambda: pp_size)


def _capture_build_and_swap(monkeypatch, updater):
    captured = {}

    def fake_build(args, named_tensors, *, scale):
        captured["build"] = (args, list(named_tensors), scale)
        return {"served.lora_A.weight": torch.ones(1)}, {"r": 1}

    def fake_swap(engines, state_dict, config_dict):
        captured["swap"] = (engines, state_dict, config_dict)

    monkeypatch.setattr(uwd, "build_lora_adapter_state_dict", fake_build)
    updater._apply_lora_adapter_swap = fake_swap
    return captured


@pytest.mark.unit
def test_pp1_rank0_fastpath_skips_gloo_gather_and_keeps_hot_swap_endpoint(monkeypatch):
    local = [(_ADAPTER_NAMES[0], torch.ones(1))]
    updater = _new_updater(is_pp_src_rank=True)
    _install_update_fakes(monkeypatch, updater, rank=0, pp_size=1, local=local)
    captured = _capture_build_and_swap(monkeypatch, updater)
    monkeypatch.setattr(uwd.dist, "get_world_size", lambda _group: pytest.fail("PP1 queried world size"))
    monkeypatch.setattr(uwd.dist, "gather_object", lambda *args, **kwargs: pytest.fail("PP1 used gather_object"))

    updater._update_weights_lora_adapter()

    assert captured["build"][1] == local
    assert captured["build"][2] == 2.0
    assert captured["swap"][0] == updater.rollout_engines
    assert "served.lora_A.weight" in captured["swap"][1]


@pytest.mark.unit
def test_pp1_nonzero_rank_skips_gloo_gather_and_returns(monkeypatch):
    updater = _new_updater(is_pp_src_rank=False)
    _install_update_fakes(monkeypatch, updater, rank=3, pp_size=1, local=[])
    monkeypatch.setattr(uwd.dist, "get_world_size", lambda _group: pytest.fail("PP1 queried world size"))
    monkeypatch.setattr(uwd.dist, "gather_object", lambda *args, **kwargs: pytest.fail("PP1 used gather_object"))
    monkeypatch.setattr(
        uwd,
        "build_lora_adapter_state_dict",
        lambda *args, **kwargs: pytest.fail("nonzero rank built adapter"),
    )

    updater._update_weights_lora_adapter()


@pytest.mark.unit
@pytest.mark.parametrize("rank,is_source", [(0, False), (2, True)])
def test_pp1_fastpath_rejects_noncanonical_source_layout(monkeypatch, rank, is_source):
    updater = _new_updater(is_pp_src_rank=is_source)
    _install_update_fakes(monkeypatch, updater, rank=rank, pp_size=1, local=[])
    monkeypatch.setattr(uwd.dist, "gather_object", lambda *args, **kwargs: pytest.fail("PP1 used gather_object"))

    with pytest.raises(RuntimeError, match="global rank 0 to be the sole canonical PP source"):
        updater._update_weights_lora_adapter()


@pytest.mark.unit
def test_pp2_rank0_gathers_only_stage_source_payloads_then_uses_hot_swap_endpoint(monkeypatch):
    stage0 = [(_ADAPTER_NAMES[0], torch.tensor([0.0]))]
    stage1 = [(_ADAPTER_NAMES[1], torch.tensor([1.0]))]
    updater = _new_updater(is_pp_src_rank=True)
    _install_update_fakes(monkeypatch, updater, rank=0, pp_size=2, local=stage0)
    captured = _capture_build_and_swap(monkeypatch, updater)
    monkeypatch.setattr(uwd.dist, "get_world_size", lambda group: 4 if group == "gloo" else 0)
    gather_calls = []

    def fake_gather(obj, *, object_gather_list, dst, group):
        gather_calls.append((obj, dst, group))
        object_gather_list[:] = [stage0, [], stage1, []]

    monkeypatch.setattr(uwd.dist, "gather_object", fake_gather)

    updater._update_weights_lora_adapter()

    assert gather_calls == [(stage0, 0, "gloo")]
    assert [name for name, _tensor in captured["build"][1]] == list(_ADAPTER_NAMES)
    assert captured["swap"][0] == updater.rollout_engines


@pytest.mark.unit
@pytest.mark.parametrize(
    "rank,is_source,local",
    [
        (1, False, []),
        (2, True, [(_ADAPTER_NAMES[1], torch.tensor([1.0]))]),
    ],
)
def test_pp2_every_nonzero_rank_joins_gloo_gather_with_ownership_payload(monkeypatch, rank, is_source, local):
    updater = _new_updater(is_pp_src_rank=is_source)
    _install_update_fakes(monkeypatch, updater, rank=rank, pp_size=2, local=local)
    monkeypatch.setattr(uwd.dist, "get_world_size", lambda _group: 4)
    gather_calls = []

    def fake_gather(obj, *, object_gather_list, dst, group):
        gather_calls.append((obj, object_gather_list, dst, group))

    monkeypatch.setattr(uwd.dist, "gather_object", fake_gather)
    monkeypatch.setattr(
        uwd,
        "build_lora_adapter_state_dict",
        lambda *args, **kwargs: pytest.fail("nonzero rank built adapter"),
    )

    updater._update_weights_lora_adapter()

    assert gather_calls == [(local, None, 0, "gloo")]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
