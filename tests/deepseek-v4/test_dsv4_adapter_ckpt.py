"""Unit tests for adapter-only (LoRA) checkpoint filtering.

V4 uses megatron.bridge LinearAdapter (an nn.Linear subclass) whose LoRA params
are named linear_in.weight / linear_out.weight (base stays at .weight, frozen) —
NO .adapter. marker — so the save/load filter must be keyed on requires_grad, not
naming. A wrong filter would write an empty or full (corrupt) checkpoint, so the
save path also asserts a small non-empty subset.
"""

import copy
import json
import multiprocessing
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.dsv4.tools.assemble_pp1_adapter_resume import assemble_pp1_resume

from slime.backends.megatron_utils.adapter_ckpt import (
    _is_adapter_key,
    _norm,
    _plan_distcp_replication,
    adapter_checkpoint_staging,
    adapter_checkpoint_staging_dir,
    adapter_only_ckpt_enabled,
    adapter_only_model_save,
    commit_adapter_checkpoint_staging,
    finalize_adapter_checkpoint,
    prepare_adapter_checkpoint_staging,
    register_adapter_async_finalize,
    replicate_ckpt_metadata_per_node,
    replicate_distcp_shards_per_node,
    trainable_param_keys,
    validate_adapter_checkpoint_components,
    validate_adapter_checkpoint_size,
    validate_loaded_lora_optimizer_state,
    validate_lora_optimizer_state,
    write_adapter_checkpoint_marker,
    write_latest_marker_per_node,
)


def test_adapter_only_checkpointing_follows_lora_training_not_provider():
    assert adapter_only_ckpt_enabled(SimpleNamespace(lora_dim=16)) is True
    assert adapter_only_ckpt_enabled(SimpleNamespace(lora_dim=0)) is False
    assert adapter_only_ckpt_enabled(
        SimpleNamespace(custom_model_provider_path="some_package.model_provider", lora_dim=8)
    )
    assert adapter_only_ckpt_enabled(SimpleNamespace()) is False


class _Chunk(nn.Module):
    """Mimics a V4 LinearAdapter-wrapped module: frozen base + trainable lin_in/out."""

    def __init__(self, adapter=True):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        if adapter:
            self.lin_in = nn.Linear(4, 2, bias=False)
            self.lin_out = nn.Linear(2, 4, bias=False)
        self._adapter = adapter

    def named_parameters(self, *a, **k):
        out = [("module.decoder.q_a_proj.weight", self.base)]
        if self._adapter:
            out += [
                ("module.decoder.q_a_proj.linear_in.weight", self.lin_in.weight),
                ("module.decoder.q_a_proj.linear_out.weight", self.lin_out.weight),
            ]
        return out

    def sharded_state_dict(self, *a, **k):
        sd = {
            "decoder.q_a_proj.weight": "BASE",
            "decoder.q_a_proj._extra_state": "EXTRA",
        }
        if self._adapter:
            sd["decoder.q_a_proj.linear_in.weight"] = "ADAPT_IN"
            sd["decoder.q_a_proj.linear_out.weight"] = "ADAPT_OUT"
        return sd


def test_norm_strips_wrapper_prefixes():
    assert _norm("module.x.y") == "x.y"
    assert _norm("module.module.x.y") == "x.y"
    assert _norm(("module.x.y", object())) == "x.y"  # tuple keys (dist ckpt)


def test_trainable_keys_are_adapter_only():
    tk = trainable_param_keys([_Chunk()])
    assert tk == {
        "decoder.q_a_proj.linear_in.weight",
        "decoder.q_a_proj.linear_out.weight",
    }
    assert "decoder.q_a_proj.weight" not in tk  # frozen base excluded


def test_is_adapter_key():
    tk = {"decoder.q_a_proj.linear_in.weight"}
    assert _is_adapter_key("decoder.q_a_proj.linear_in.weight", tk)
    assert not _is_adapter_key("decoder.q_a_proj.weight", tk)  # base
    assert not _is_adapter_key("decoder.q_a_proj._extra_state", tk)  # extra state
    assert _is_adapter_key("x.adapter.linear_in.weight", set())  # AdapterWrapper variant


def test_save_filter_keeps_adapters_drops_base_and_restores():
    m = [_Chunk()]
    with adapter_only_model_save(m):
        filtered = m[0].sharded_state_dict()
    assert set(filtered) == {
        "decoder.q_a_proj.linear_in.weight",
        "decoder.q_a_proj.linear_out.weight",
    }
    # original restored after the context
    assert "decoder.q_a_proj.weight" in m[0].sharded_state_dict()


def test_save_filter_refuses_when_nothing_trainable():
    # No adapters -> no trainable params -> must raise (would write empty ckpt).
    with pytest.raises(RuntimeError, match="no trainable params"):
        with adapter_only_model_save([_Chunk(adapter=False)]):
            pass


def test_save_filter_refuses_corrupt_all_or_nothing():
    # A key space where the filter keeps everything (no base to drop) must fail
    # the "strictly smaller" guard rather than silently save a full checkpoint.
    class AllAdapter(_Chunk):
        def sharded_state_dict(self, *a, **k):
            return {
                "decoder.q_a_proj.linear_in.weight": "IN",
                "decoder.q_a_proj.linear_out.weight": "OUT",
            }

    m = [AllAdapter()]
    with pytest.raises(AssertionError, match="kept"):
        with adapter_only_model_save(m):
            m[0].sharded_state_dict()


def test_chained_optimizer_synchronize_steps_tolerates_stub():
    """Muon+LoRA builds a ChainedOptimizer with a stub sub-optimizer
    (optimizer.optimizer is None). Upstream _synchronize_steps crashed on it
    ('NoneType' has no attribute 'param_groups') during any optimizer save/load;
    slime patches it stub-safe (checkpoint.py). Guards the patch stays applied."""
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    import slime.backends.megatron_utils.checkpoint  # noqa: F401 (applies the patch on import)

    class _Stub:
        optimizer = None

    class _Real:
        class _Inner:
            param_groups = [{"params": [1], "step": 5}]

        optimizer = _Inner()

    co = ChainedOptimizer.__new__(ChainedOptimizer)
    co.chained_optimizers = [_Real(), _Stub()]
    # Must not raise on the stub, and must resolve the single real step.
    assert ChainedOptimizer._synchronize_steps(co) == 5


class _FakeFloat16Leaf:
    is_stub_optimizer = False

    def __init__(self, params):
        self.float16_groups = [list(params)]
        self.fp32_from_fp32_groups = [[]]
        self.fp32_from_float16_groups = [[nn.Parameter(param.detach().float().clone()) for param in params]]
        self.optimizer = SimpleNamespace(
            param_groups=[{"params": self.fp32_from_float16_groups[0]}],
            state={},
        )


class _FakeStubLeaf:
    is_stub_optimizer = True


class _FakeChain:
    def __init__(self, *children):
        self.chained_optimizers = list(children)


def test_lora_optimizer_validation_accepts_exact_live_adapter_set():
    chunk = _Chunk()
    optimizer = _FakeChain(
        _FakeFloat16Leaf([chunk.lin_in.weight, chunk.lin_out.weight]),
        _FakeStubLeaf(),
    )
    stats = validate_lora_optimizer_state([chunk], optimizer)
    assert stats["optimizer_param_tensors"] == 2
    assert stats["optimizer_param_numel"] == 16
    assert stats["fp32_master_numel"] == 16


def test_lora_optimizer_validation_rejects_frozen_base_owned_by_optimizer():
    chunk = _Chunk()
    optimizer = _FakeFloat16Leaf([chunk.lin_in.weight, chunk.lin_out.weight, chunk.base])
    with pytest.raises(RuntimeError, match="unexpected_optimizer_params=1"):
        validate_lora_optimizer_state([chunk], optimizer)


def test_lora_optimizer_validation_rejects_missing_adapter_param():
    chunk = _Chunk()
    optimizer = _FakeFloat16Leaf([chunk.lin_in.weight])
    with pytest.raises(RuntimeError, match="missing_lora_params=1"):
        validate_lora_optimizer_state([chunk], optimizer)


def test_lora_optimizer_validation_rejects_non_lora_trainable_param():
    chunk = _Chunk()
    chunk.base.requires_grad_(True)
    optimizer = _FakeFloat16Leaf([chunk.lin_in.weight, chunk.lin_out.weight, chunk.base])
    with pytest.raises(RuntimeError, match="non-LoRA trainable parameters"):
        validate_lora_optimizer_state([chunk], optimizer)


def test_lora_optimizer_validation_rejects_foreign_optimizer_state():
    chunk = _Chunk()
    optimizer = _FakeFloat16Leaf([chunk.lin_in.weight, chunk.lin_out.weight])
    foreign_master = nn.Parameter(torch.zeros(4, 4))
    optimizer.optimizer.state[foreign_master] = {"momentum_buffer": torch.zeros_like(foreign_master)}
    with pytest.raises(RuntimeError, match="state belongs to an untracked parameter"):
        validate_lora_optimizer_state([chunk], optimizer)


def test_loaded_lora_optimizer_state_matches_checkpoint_marker():
    chunk = _Chunk()
    optimizer = _FakeFloat16Leaf([chunk.lin_in.weight, chunk.lin_out.weight])
    for master in optimizer.fp32_from_float16_groups[0]:
        optimizer.optimizer.state[master] = {"momentum_buffer": torch.zeros_like(master)}
    stats = validate_lora_optimizer_state([chunk], optimizer)
    marker = {"saved_optimizer": True, **stats}
    assert validate_loaded_lora_optimizer_state([chunk], optimizer, marker) == stats


def test_loaded_lora_optimizer_state_rejects_silent_empty_restore():
    chunk = _Chunk()
    optimizer = _FakeFloat16Leaf([chunk.lin_in.weight, chunk.lin_out.weight])
    stats = validate_lora_optimizer_state([chunk], optimizer)
    marker = {"saved_optimizer": True, **stats, "optimizer_state_tensors": 2}
    with pytest.raises(RuntimeError, match="optimizer_state_tensors"):
        validate_loaded_lora_optimizer_state([chunk], optimizer, marker)


def test_adapter_component_marker_roundtrip_and_missing_component_failures(tmp_path):
    chunk = _Chunk()
    write_adapter_checkpoint_marker(
        str(tmp_path),
        4,
        [chunk],
        optimizer_stats={
            "optimizer_param_tensors": 2,
            "optimizer_param_numel": 16,
            "fp32_master_tensors": 2,
            "fp32_master_numel": 16,
        },
        saved_optimizer=True,
        saved_rng=True,
    )
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("4")
    marker = validate_adapter_checkpoint_components(str(tmp_path), load_optimizer=True, load_rng=True)
    assert marker["saved_optimizer"] is True
    assert marker["saved_rng"] is True
    assert marker["optimizer_param_numel"] == marker["lora_param_numel"] == 16

    marker["saved_optimizer"] = False
    marker_path = tmp_path / "iter_0000004" / "v4_adapter_checkpoint.json"
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="weights-only"):
        validate_adapter_checkpoint_components(str(tmp_path), load_optimizer=True, load_rng=False)


def test_adapter_component_marker_required_for_stateful_legacy_resume(tmp_path):
    (tmp_path / "iter_0000004").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("4")
    assert validate_adapter_checkpoint_components(str(tmp_path), load_optimizer=False, load_rng=False) is None
    with pytest.raises(RuntimeError, match="cannot prove"):
        validate_adapter_checkpoint_components(str(tmp_path), load_optimizer=False, load_rng=True)


def test_stub_optimizer_state_dict_roundtrip_is_empty_and_loadable():
    from megatron.core.optimizer.optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params

    import slime.backends.megatron_utils.checkpoint  # noqa: F401 (applies patches)

    class _Real:
        def __init__(self):
            self.loaded = None

        def state_dict(self):
            return {"state": "real"}

        def load_state_dict(self, state):
            self.loaded = state

    real = _Real()
    stub = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    stub.optimizer = None
    stub.is_stub_optimizer = True
    chain = ChainedOptimizer.__new__(ChainedOptimizer)
    chain.chained_optimizers = [real, stub]

    saved = chain.state_dict()
    assert saved == [{"state": "real"}, {}]
    chain.load_state_dict(saved)
    assert real.loaded == {"state": "real"}
    with pytest.raises(RuntimeError, match="must be empty"):
        stub.load_state_dict({"unexpected": 1})


def _make_real_float16_optimizer(model_params):
    """Build the real Megatron bf16-wrapper serialization path on CPU."""
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    masters = [nn.Parameter(param.detach().float().clone()) for param in model_params]
    param_group = {
        "params": masters,
        "wd_mult": 1.0,
        "lr_mult": 1.0,
        "is_expert_parallel": False,
        "is_decoupled_lr": False,
    }
    inner = torch.optim.SGD([param_group], lr=0.1, momentum=0.9)
    wrapper = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    wrapper.float16_groups = [list(model_params)]
    wrapper.fp32_from_float16_groups = [masters]
    wrapper.fp32_from_fp32_groups = [[]]
    wrapper.optimizer = inner
    wrapper.grad_scaler = None
    wrapper.config = SimpleNamespace(fp16=False)
    wrapper.init_state_fn = lambda *_args, **_kwargs: None
    return wrapper, masters


def _step_real_float16_optimizer(wrapper):
    for master in wrapper.fp32_from_float16_groups[0]:
        master.grad = torch.full_like(master, 0.25)
    wrapper.optimizer.step()


def test_real_float16_optimizer_and_stub_restore_masters_and_momentum():
    from megatron.core.optimizer.optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params

    import slime.backends.megatron_utils.checkpoint  # noqa: F401 (applies patches)

    source_chunk = _Chunk().to(dtype=torch.bfloat16)
    source, source_masters = _make_real_float16_optimizer([source_chunk.lin_in.weight, source_chunk.lin_out.weight])
    _step_real_float16_optimizer(source)
    source_stub = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    source_stub.optimizer = None
    source_stub.is_stub_optimizer = True
    source_chain = ChainedOptimizer.__new__(ChainedOptimizer)
    source_chain.chained_optimizers = [source, source_stub]
    saved = copy.deepcopy(source_chain.state_dict())

    target_chunk = _Chunk().to(dtype=torch.bfloat16)
    target, target_masters = _make_real_float16_optimizer([target_chunk.lin_in.weight, target_chunk.lin_out.weight])
    for master in target_masters:
        master.data.fill_(99.0)
    target_stub = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    target_stub.optimizer = None
    target_stub.is_stub_optimizer = True
    target_chain = ChainedOptimizer.__new__(ChainedOptimizer)
    target_chain.chained_optimizers = [target, target_stub]
    target_chain.load_state_dict(copy.deepcopy(saved))

    for source_master, target_master in zip(source_masters, target_masters, strict=True):
        torch.testing.assert_close(target_master, source_master, rtol=0, atol=0)
        torch.testing.assert_close(
            target.optimizer.state[target_master]["momentum_buffer"],
            source.optimizer.state[source_master]["momentum_buffer"],
            rtol=0,
            atol=0,
        )

    stats = validate_lora_optimizer_state([target_chunk], target_chain)
    assert stats["optimizer_param_numel"] == 16
    assert stats["fp32_master_numel"] == 16
    assert stats["optimizer_state_numel"] == 16


def test_real_float16_sharded_state_keys_are_lora_only():
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    import slime.backends.megatron_utils.checkpoint  # noqa: F401 (applies patches)

    chunk = _Chunk().to(dtype=torch.bfloat16)
    model_params = [chunk.lin_in.weight, chunk.lin_out.weight]
    optimizer, _masters = _make_real_float16_optimizer(model_params)
    _step_real_float16_optimizer(optimizer)
    names = (
        "decoder.q_a_proj.linear_in.weight",
        "decoder.q_a_proj.linear_out.weight",
    )
    model_sharded_state = {
        name: ShardedTensor.from_rank_offsets(name, param, (0, 0, 1))
        for name, param in zip(names, model_params, strict=True)
    }
    state = optimizer.sharded_state_dict(model_sharded_state)

    def _values(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from _values(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from _values(child)
        else:
            yield value

    keys = [item.key for item in _values(state) if isinstance(item, ShardedTensor)]
    assert len(keys) == 4
    assert any(key.startswith("optimizer.state.fp32_param.") for key in keys)
    assert any(key.startswith("optimizer.state.momentum_buffer.") for key in keys)
    assert all(key.endswith((".linear_in.weight", ".linear_out.weight")) for key in keys)
    assert all(
        "q_a_proj.weight" not in key.removesuffix(".linear_in.weight").removesuffix(".linear_out.weight")
        for key in keys
    )


def test_adapter_checkpoint_node_size_guard(tmp_path):
    iteration_dir = tmp_path / "iter_0000004"
    iteration_dir.mkdir()
    (iteration_dir / "rank0.distcp").write_bytes(b"1234")
    assert validate_adapter_checkpoint_size(str(tmp_path), 4, max_node_bytes=4) == 4
    with pytest.raises(RuntimeError, match="exceeded the node-local safety limit"):
        validate_adapter_checkpoint_size(str(tmp_path), 4, max_node_bytes=3)


def test_adapter_checkpoint_staging_supports_async_and_restores_args(tmp_path):
    args = SimpleNamespace(
        save=str(tmp_path / "save"),
        async_save=True,
        non_persistent_ckpt_type="old-type",
        non_persistent_global_ckpt_dir="old-dir",
    )

    with adapter_checkpoint_staging(args, 9) as staging_dir:
        assert staging_dir == str(tmp_path / "save" / ".pending_adapter_iter_0000009")
        assert args.non_persistent_ckpt_type == "global"
        assert args.non_persistent_global_ckpt_dir == staging_dir

    assert args.non_persistent_ckpt_type == "old-type"
    assert args.non_persistent_global_ckpt_dir == "old-dir"


def test_adapter_async_finalize_is_appended_before_request_freeze(monkeypatch):
    from megatron.training import checkpointing

    events = []

    class Request:
        def __init__(self):
            self.finalize_fns = [lambda: events.append("upstream")]

        def add_finalize_fn(self, fn):
            self.finalize_fns.append(fn)

    request = Request()

    def schedule(request_to_schedule):
        assert request_to_schedule is request
        events.append("scheduled")

    monkeypatch.setattr(checkpointing, "schedule_async_save", schedule)

    with register_adapter_async_finalize(lambda: events.append("adapter")):
        checkpointing.schedule_async_save(request)

    assert checkpointing.schedule_async_save is schedule
    assert events == ["scheduled"]
    for finalize_fn in request.finalize_fns:
        finalize_fn()
    assert events == ["scheduled", "upstream", "adapter"]


def test_adapter_async_finalize_requires_exactly_one_request(monkeypatch):
    from megatron.training import checkpointing

    monkeypatch.setattr(checkpointing, "schedule_async_save", lambda _request: None)
    with pytest.raises(RuntimeError, match="did not schedule exactly one"):
        with register_adapter_async_finalize(lambda: None):
            pass


def test_finalize_adapter_checkpoint_publishes_only_after_validation(monkeypatch):
    import slime.backends.megatron_utils.adapter_ckpt as adapter_ckpt

    events = []
    monkeypatch.setattr(
        adapter_ckpt,
        "replicate_ckpt_metadata_per_node",
        lambda *_args: events.append("metadata"),
    )
    monkeypatch.setattr(
        adapter_ckpt,
        "replicate_distcp_shards_per_node",
        lambda *_args: events.append("shards"),
    )
    monkeypatch.setattr(
        adapter_ckpt,
        "write_adapter_scaling_marker",
        lambda *_args, **_kwargs: events.append("scaling"),
    )
    monkeypatch.setattr(
        adapter_ckpt,
        "write_adapter_checkpoint_marker",
        lambda *_args, **_kwargs: events.append("components"),
    )
    monkeypatch.setattr(
        adapter_ckpt,
        "validate_adapter_checkpoint_size",
        lambda *_args: events.append("validate"),
    )
    monkeypatch.setattr(
        adapter_ckpt,
        "commit_adapter_checkpoint_staging",
        lambda *_args: events.append("commit"),
    )
    monkeypatch.setattr(
        adapter_ckpt,
        "write_latest_marker_per_node",
        lambda *_args: events.append("latest"),
    )

    finalize_adapter_checkpoint(
        "/final",
        "/staging",
        9,
        object(),
        optimizer_stats={},
        saved_optimizer=True,
        saved_rng=True,
    )

    assert events == [
        "metadata",
        "shards",
        "scaling",
        "components",
        "validate",
        "commit",
        "latest",
    ]


def test_model_save_defers_adapter_publication_in_async_mode(tmp_path, monkeypatch):
    import slime.backends.megatron_utils.adapter_ckpt as adapter_ckpt
    import slime.backends.megatron_utils.model as model_utils

    args = SimpleNamespace(
        async_save=True,
        no_save_optim=True,
        no_save_rng=False,
        save=str(tmp_path / "save"),
        lora_checkpoint_max_node_bytes=1234,
        lora_rslora=True,
    )
    events = []
    captured = {}

    @contextmanager
    def staging(_args, iteration):
        assert iteration == 9
        events.append("staging-enter")
        yield str(tmp_path / "save" / ".pending_adapter_iter_0000009")
        events.append("staging-exit")

    @contextmanager
    def model_filter(_model):
        events.append("filter-enter")
        yield
        events.append("filter-exit")

    @contextmanager
    def register(finalize_fn):
        captured["finalize"] = finalize_fn
        events.append("registered")
        yield

    def finalize(*_args, **kwargs):
        assert kwargs["saved_optimizer"] is False
        assert kwargs["saved_rng"] is True
        assert kwargs["max_node_bytes"] == 1234
        assert kwargs["rslora"] is True
        events.append("published")

    monkeypatch.setattr(model_utils, "get_args", lambda: args)
    monkeypatch.setattr(model_utils, "should_disable_forward_pre_hook", lambda _args: False)
    monkeypatch.setattr(
        model_utils,
        "save_checkpoint",
        lambda *_args, **_kwargs: events.append("writer-scheduled"),
    )
    monkeypatch.setattr(adapter_ckpt, "adapter_only_ckpt_enabled", lambda _args: True)
    monkeypatch.setattr(adapter_ckpt, "adapter_checkpoint_staging", staging)
    monkeypatch.setattr(adapter_ckpt, "adapter_only_model_save", model_filter)
    monkeypatch.setattr(adapter_ckpt, "register_adapter_async_finalize", register)
    monkeypatch.setattr(adapter_ckpt, "finalize_adapter_checkpoint", finalize)

    model_utils.save(9, [object()], object(), object())

    assert events == [
        "staging-enter",
        "filter-enter",
        "registered",
        "writer-scheduled",
        "filter-exit",
        "staging-exit",
    ]
    captured["finalize"]()
    assert events[-1] == "published"


def test_distcp_replication_plan_rejects_same_name_hash_conflict():
    expected = {"__0_0.distcp": 4}
    manifests = [
        {"__0_0.distcp": {"size": 4, "sha256": "aaaa"}},
        {"__0_0.distcp": {"size": 4, "sha256": "bbbb"}},
    ]
    with pytest.raises(RuntimeError, match="same-name distcp shard hash conflict"):
        _plan_distcp_replication(expected, manifests, (0, 1))


def test_distcp_replication_plan_requires_exact_metadata_union():
    expected = {"__0_0.distcp": 4, "__1_0.distcp": 5}
    manifests = [{"__0_0.distcp": {"size": 4, "sha256": "aaaa"}}, {}]
    with pytest.raises(RuntimeError, match="union does not equal .metadata"):
        _plan_distcp_replication(expected, manifests, (0, 1))


def _cpu_distcp_replication_worker(rank, init_file, save_dirs):
    """Model four ranks on two node-local filesystems (leader + nonleader each)."""
    import socket

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
    )
    # All subprocesses run on the same CPU host. Give the checkpoint topology
    # detector distinct identities so this exercises the node-leader transfer.
    node_index = rank // 2
    local_rank = rank % 2
    socket.gethostname = lambda: f"test-node-{node_index}"
    try:
        save_dir = save_dirs[node_index]
        staging_dir = prepare_adapter_checkpoint_staging(save_dir, 7)
        assert staging_dir == adapter_checkpoint_staging_dir(save_dir, 7)
        iteration_dir = Path(staging_dir) / "iter_0000007"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        contents = (b"left-shard", b"right-shard-data")
        if local_rank == 0:
            shard_name = f"__{node_index}_0.distcp"
            (iteration_dir / shard_name).write_bytes(contents[node_index])
        if rank == 0:
            metadata = SimpleNamespace(
                storage_data={
                    index: SimpleNamespace(
                        relative_path=f"__{index}_0.distcp",
                        offset=0,
                        length=len(contents[index]),
                    )
                    for index in range(2)
                }
            )
            (iteration_dir / ".metadata").write_bytes(pickle.dumps(metadata))
            (iteration_dir / "common.pt").write_bytes(b"common")
            (iteration_dir / "metadata.json").write_text('{"sharded_backend":"torch_dist"}')
        dist.barrier()

        replicate_ckpt_metadata_per_node(staging_dir, 7)
        replicate_distcp_shards_per_node(staging_dir, 7, chunk_bytes=3)
        validate_adapter_checkpoint_size(staging_dir, 7)
        commit_adapter_checkpoint_staging(save_dir, staging_dir, 7)
        write_latest_marker_per_node(save_dir, 7)
    finally:
        dist.destroy_process_group()


def test_cpu_gloo_replication_and_atomic_staging_commit(tmp_path):
    save_dirs = [str(tmp_path / "node0"), str(tmp_path / "node1")]
    init_file = str(tmp_path / "gloo_init")
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_cpu_distcp_replication_worker,
            args=(rank, init_file, save_dirs),
        )
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            pytest.fail("CPU/Gloo checkpoint replication worker hung")
        assert process.exitcode == 0

    for save_dir in map(Path, save_dirs):
        iteration_dir = save_dir / "iter_0000007"
        assert (iteration_dir / "__0_0.distcp").read_bytes() == b"left-shard"
        assert (iteration_dir / "__1_0.distcp").read_bytes() == b"right-shard-data"
        assert (save_dir / "latest_checkpointed_iteration.txt").read_text() == "7\n"
        assert not (save_dir / ".pending_adapter_iter_0000007").exists()


def _cpu_distcp_missing_shard_worker(rank, init_file, save_dirs):
    import socket

    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    socket.gethostname = lambda: f"missing-test-node-{rank}"
    try:
        iteration_dir = Path(save_dirs[rank]) / "iter_0000003"
        iteration_dir.mkdir(parents=True)
        if rank == 0:
            (iteration_dir / "__0_0.distcp").write_bytes(b"only-shard")
            metadata = SimpleNamespace(
                storage_data={
                    0: SimpleNamespace(relative_path="__0_0.distcp", offset=0, length=10),
                    1: SimpleNamespace(relative_path="__1_0.distcp", offset=0, length=7),
                }
            )
            (iteration_dir / ".metadata").write_bytes(pickle.dumps(metadata))
            (iteration_dir / "common.pt").write_bytes(b"common")
            (iteration_dir / "metadata.json").write_text("{}")
        dist.barrier()
        replicate_ckpt_metadata_per_node(save_dirs[rank], 3)
        try:
            replicate_distcp_shards_per_node(save_dirs[rank], 3, chunk_bytes=3)
        except RuntimeError as exc:
            assert "union does not equal .metadata" in str(exc)
            (Path(save_dirs[rank]) / "saw_global_error").write_text("yes")
        else:
            raise AssertionError("missing global shard unexpectedly passed replication")
    finally:
        dist.destroy_process_group()


def test_missing_shard_fails_all_ranks_without_collective_hang(tmp_path):
    save_dirs = [str(tmp_path / "missing-node0"), str(tmp_path / "missing-node1")]
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_cpu_distcp_missing_shard_worker,
            args=(rank, str(tmp_path / "missing_gloo_init"), save_dirs),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            pytest.fail("missing-shard failure left a rank hung in a collective")
        assert process.exitcode == 0
    assert all((Path(save_dir) / "saw_global_error").is_file() for save_dir in save_dirs)


def _write_node_local_pp_source(path, rank, *, lora_tensors, lora_numel):
    path.mkdir()
    shard_name = f"__{rank}_0.distcp"
    (path / shard_name).write_bytes(f"rank-{rank}".encode())
    metadata = SimpleNamespace(
        storage_data={
            rank: SimpleNamespace(relative_path=name) for rank, name in enumerate(("__0_0.distcp", "__1_0.distcp"))
        },
        state_dict_metadata={
            **{f"layers.{i}": SimpleNamespace(size=(8,)) for i in range(5)},
            **{f"optimizer.state.fp32_param.layers.{i}": SimpleNamespace(size=(8,)) for i in range(5)},
            **{f"optimizer.state.momentum_buffer.layers.{i}": SimpleNamespace(size=(8,)) for i in range(5)},
            "rng_state/shard_0": SimpleNamespace(),
        },
    )
    (path / ".metadata").write_bytes(pickle.dumps(metadata))
    (path / "common.pt").write_bytes(b"common")
    (path / "metadata.json").write_text('{"sharded_backend":"torch_dist"}')
    (path / "v4_lora_scaling.json").write_text('{"dim":32,"alpha":32.0,"scale":5.65685424949238}')
    marker = {
        "schema_version": 2,
        "saved_optimizer": True,
        "saved_rng": True,
        "lora_param_tensors": lora_tensors,
        "lora_param_numel": lora_numel,
        "optimizer_param_tensors": lora_tensors,
        "optimizer_param_numel": lora_numel,
        "fp32_master_tensors": lora_tensors,
        "fp32_master_numel": lora_numel,
        "fp32_master_bytes": lora_numel * 4,
        "optimizer_state_tensors": lora_tensors,
        "optimizer_state_numel": lora_numel,
        "optimizer_state_bytes": lora_numel * 4,
    }
    (path / "v4_adapter_checkpoint.json").write_text(json.dumps(marker))


def test_assemble_pp1_adapter_resume_unions_shards_and_markers(tmp_path):
    left = tmp_path / "left" / "iter_0000059"
    right = tmp_path / "right" / "iter_0000059"
    left.parent.mkdir()
    right.parent.mkdir()
    output = tmp_path / "assembled"
    _write_node_local_pp_source(left, 0, lora_tensors=2, lora_numel=16)
    _write_node_local_pp_source(right, 1, lora_tensors=3, lora_numel=24)

    target = assemble_pp1_resume([left, right], output, 59)

    assert {path.name for path in target.glob("*.distcp")} == {
        "__0_0.distcp",
        "__1_0.distcp",
    }
    marker = json.loads((target / "v4_adapter_checkpoint.json").read_text())
    assert marker["lora_param_tensors"] == 5
    assert marker["lora_param_numel"] == 40
    assert marker["optimizer_param_numel"] == 40
    assert marker["fp32_master_numel"] == 40
    assert (output / "latest_checkpointed_iteration.txt").read_text() == "59\n"
    assert left.exists() and right.exists(), "assembly must not modify either source"


def test_assemble_pp1_adapter_resume_rejects_shared_file_mismatch(tmp_path):
    left = tmp_path / "left" / "iter_0000059"
    right = tmp_path / "right" / "iter_0000059"
    left.parent.mkdir()
    right.parent.mkdir()
    _write_node_local_pp_source(left, 0, lora_tensors=2, lora_numel=16)
    _write_node_local_pp_source(right, 1, lora_tensors=3, lora_numel=24)
    (right / "common.pt").write_bytes(b"different")

    with pytest.raises(RuntimeError, match="disagree on shared file common.pt"):
        assemble_pp1_resume([left, right], tmp_path / "assembled", 59)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
