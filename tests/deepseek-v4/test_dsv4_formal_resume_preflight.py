"""CPU tests for the formal DS-V4 migration/native resume preflight."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.dsv4.formal_resume_preflight import (
    EXPECTED_LORA_NUMEL,
    EXPECTED_LORA_TENSORS,
    PreflightError,
    TargetSnapshot,
    inspect_target,
    probe_checkpoint,
    resolve_resume_plan,
)


def _snapshot(
    *,
    latest_present: bool = False,
    latest_text: str | None = None,
    iteration_dirs: tuple[str, ...] = (),
    root: str = "/target",
) -> TargetSnapshot:
    return TargetSnapshot(
        root=root,
        root_exists=True,
        latest_present=latest_present,
        latest_text=latest_text,
        iter_dir_names=iteration_dirs,
    )


def test_inspect_target_ignores_rollout_files_but_reports_every_iter_directory(tmp_path):
    target = tmp_path / "out"
    (target / "rollout").mkdir(parents=True)
    (target / "notes.txt").write_text("not a checkpoint")
    (target / "iter_partial").mkdir()

    snapshot = inspect_target(target)

    assert snapshot.latest_present is False
    assert snapshot.latest_text is None
    assert snapshot.iter_dir_names == ("iter_partial",)


def test_missing_target_root_is_an_empty_snapshot(tmp_path):
    snapshot = inspect_target(tmp_path / "does-not-exist")

    assert snapshot.root_exists is False
    assert snapshot.latest_present is False
    assert snapshot.iter_dir_names == ()


def test_inspect_target_rejects_a_file_where_the_save_root_should_be(tmp_path):
    target = tmp_path / "out"
    target.write_text("not a directory")

    with pytest.raises(PreflightError, match="exists but is not a directory"):
        inspect_target(target)


def test_resolve_empty_targets_selects_iter59_migration_with_rng_drop():
    plan = resolve_resume_plan(_snapshot(), _snapshot(root="/remote-target"))

    assert plan.mode == "migration"
    assert plan.checkpoint_iteration == 59
    assert plan.start_rollout_id == 60
    assert plan.load_rng == 0


def test_resolve_target_with_only_rollout_directory_selects_migration(tmp_path):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    (local / "rollout").mkdir(parents=True)
    (remote / "rollout").mkdir(parents=True)

    plan = resolve_resume_plan(inspect_target(local), inspect_target(remote))

    assert plan.mode == "migration"


def test_resolve_matching_native_targets_advances_iteration_and_loads_rng():
    native = _snapshot(
        latest_present=True,
        latest_text="64",
        iteration_dirs=("iter_0000059", "iter_0000064"),
    )

    plan = resolve_resume_plan(native, native)

    assert plan.mode == "native"
    assert plan.checkpoint_iteration == 64
    assert plan.start_rollout_id == 65
    assert plan.load_rng == 1


def test_zero_based_native_lineage_resumes_first_checkpoint():
    native = _snapshot(
        latest_present=True,
        latest_text="4",
        iteration_dirs=("iter_0000004",),
    )

    plan = resolve_resume_plan(
        native,
        native,
        allow_migration=False,
        minimum_native_iteration=0,
    )

    assert plan.mode == "native"
    assert plan.checkpoint_iteration == 4
    assert plan.start_rollout_id == 5
    assert plan.load_rng == 1


def test_zero_based_native_lineage_refuses_empty_root_without_explicit_fresh():
    with pytest.raises(PreflightError, match="explicit fresh-start"):
        resolve_resume_plan(
            _snapshot(),
            _snapshot(root="/remote-target"),
            allow_migration=False,
            minimum_native_iteration=0,
        )


@pytest.mark.parametrize(
    ("local", "remote", "message"),
    [
        (
            _snapshot(),
            _snapshot(latest_present=True, latest_text="64", iteration_dirs=("iter_0000064",)),
            "latest markers disagree",
        ),
        (
            _snapshot(latest_present=True, latest_text="bad", iteration_dirs=("iter_0000064",)),
            _snapshot(latest_present=True, latest_text="bad", iteration_dirs=("iter_0000064",)),
            "not numeric",
        ),
        (
            _snapshot(latest_present=True, latest_text="64", iteration_dirs=("iter_0000064",)),
            _snapshot(latest_present=True, latest_text="69", iteration_dirs=("iter_0000069",)),
            "latest iterations disagree",
        ),
        (
            _snapshot(latest_present=True, latest_text="59", iteration_dirs=("iter_0000059",)),
            _snapshot(latest_present=True, latest_text="59", iteration_dirs=("iter_0000059",)),
            "must be >= 60",
        ),
        (
            _snapshot(latest_present=True, latest_text="64", iteration_dirs=("iter_0000064",)),
            _snapshot(latest_present=True, latest_text="64"),
            "no iter_0000064 directory",
        ),
        (
            _snapshot(iteration_dirs=("iter_0000059",)),
            _snapshot(iteration_dirs=("iter_0000059",)),
            "iteration directories exist without latest markers",
        ),
    ],
)
def test_resolve_rejects_every_partial_or_inconsistent_target(local, remote, message):
    with pytest.raises(PreflightError, match=message):
        resolve_resume_plan(local, remote)


def _marker() -> dict[str, object]:
    return {
        "schema_version": 2,
        "saved_optimizer": True,
        "saved_rng": True,
        "lora_param_tensors": EXPECTED_LORA_TENSORS,
        "lora_param_numel": EXPECTED_LORA_NUMEL,
        "optimizer_param_tensors": EXPECTED_LORA_TENSORS,
        "optimizer_param_numel": EXPECTED_LORA_NUMEL,
        "fp32_master_tensors": EXPECTED_LORA_TENSORS,
        "fp32_master_numel": EXPECTED_LORA_NUMEL,
        "fp32_master_bytes": EXPECTED_LORA_NUMEL * 4,
        "optimizer_state_tensors": EXPECTED_LORA_TENSORS,
        "optimizer_state_numel": EXPECTED_LORA_NUMEL,
        "optimizer_state_bytes": EXPECTED_LORA_NUMEL * 4,
    }


def _write_checkpoint(
    root: Path,
    *,
    iteration: int,
    distcp_count: int,
    topology: dict[str, object] | None = None,
) -> Path:
    iteration_dir = root / f"iter_{iteration:07d}"
    iteration_dir.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")
    (iteration_dir / "metadata.json").write_text('{"sharded_backend":"torch_dist"}')
    (iteration_dir / "v4_lora_scaling.json").write_text('{"dim":32}')
    (iteration_dir / "v4_adapter_checkpoint.json").write_text(json.dumps(_marker()))
    for rank in range(distcp_count):
        (iteration_dir / f"__{rank}_0.distcp").write_bytes(f"rank-{rank}".encode())
    metadata = SimpleNamespace(
        storage_data={
            rank: SimpleNamespace(
                relative_path=f"__{rank}_0.distcp",
                offset=0,
                length=len(f"rank-{rank}".encode()),
            )
            for rank in range(distcp_count)
        }
    )
    (iteration_dir / ".metadata").write_bytes(pickle.dumps(metadata))

    args = SimpleNamespace(
        **(
            topology
            or {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 2,
                "expert_model_parallel_size": 8,
                "world_size": 16,
                "cp_partition_mode": "contiguous",
            }
        )
    )
    torch.save(
        {"iteration": iteration, "args": args},
        iteration_dir / "common.pt",
    )
    return iteration_dir


def test_probe_migration_accepts_iter59_old_topology_and_32_distcp(tmp_path):
    root = tmp_path / "migration"
    _write_checkpoint(
        root,
        iteration=59,
        distcp_count=32,
        topology={
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 8,
            "world_size": 16,
            "cp_partition_mode": "zigzag",
        },
    )

    result = probe_checkpoint(root, 59, mode="migration")

    assert result["status"] == "PASS"
    assert result["distcp_count"] == 32
    assert result["topology"] == {}


def test_probe_native_requires_pp1_cp2_ep8_world16_contiguous_and_32_distcp(tmp_path):
    root = tmp_path / "native"
    _write_checkpoint(root, iteration=64, distcp_count=32)

    result = probe_checkpoint(root, 64, mode="native")

    assert result["status"] == "PASS"
    assert result["distcp_count"] == 32
    assert result["topology"]["pipeline_model_parallel_size"] == 1
    assert result["topology"]["context_parallel_size"] == 2
    assert result["topology"]["cp_partition_mode"] == "contiguous"


def test_probe_zero_based_native_accepts_first_iter4_checkpoint(tmp_path):
    root = tmp_path / "zero-based-native"
    _write_checkpoint(root, iteration=4, distcp_count=32)

    result = probe_checkpoint(
        root,
        4,
        mode="native",
        minimum_native_iteration=0,
    )

    assert result["status"] == "PASS"
    assert result["iteration"] == 4


def test_probe_zero_based_native_still_fails_without_explicit_minimum(tmp_path):
    with pytest.raises(PreflightError, match="must be >= 60"):
        probe_checkpoint(tmp_path / "unused", 4, mode="native")


@pytest.mark.parametrize(
    ("mode", "iteration", "expected_distcp", "message"),
    [
        ("migration", 58, None, "migration checkpoint iteration must be 59"),
        ("native", 59, None, "native checkpoint iteration must be >= 60"),
        ("migration", 59, 16, "migration checkpoint must have 32"),
        ("native", 64, 16, "native checkpoint must have 32"),
    ],
)
def test_probe_rejects_mode_iteration_or_shard_contract_overrides(tmp_path, mode, iteration, expected_distcp, message):
    with pytest.raises(PreflightError, match=message):
        probe_checkpoint(
            tmp_path / "unused",
            iteration,
            mode=mode,
            expected_distcp=expected_distcp,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("latest", "latest iteration is 63"),
        ("shared", "shared files missing or empty"),
        ("distcp", "expected 32 distcp files, found 31"),
        ("empty_distcp", "empty distcp files"),
        ("marker_tensors", "lora_param_tensors must be 766"),
        ("marker_numel", "optimizer_state_numel must be 114933760"),
        ("optimizer", "saved_optimizer must be true"),
        ("rng", "saved_rng must be true"),
        ("common_iteration", "common checkpoint iteration must be 64"),
    ],
)
def test_probe_rejects_incomplete_or_wrong_checkpoint_components(tmp_path, mutation, message):
    root = tmp_path / mutation
    iteration_dir = _write_checkpoint(root, iteration=64, distcp_count=32)
    if mutation == "latest":
        (root / "latest_checkpointed_iteration.txt").write_text("63\n")
    elif mutation == "shared":
        (iteration_dir / "metadata.json").unlink()
    elif mutation == "distcp":
        (iteration_dir / "__31_0.distcp").unlink()
    elif mutation == "empty_distcp":
        (iteration_dir / "__31_0.distcp").write_bytes(b"")
    elif mutation in {"marker_tensors", "marker_numel", "optimizer", "rng"}:
        marker_path = iteration_dir / "v4_adapter_checkpoint.json"
        marker = json.loads(marker_path.read_text())
        field_and_value = {
            "marker_tensors": ("lora_param_tensors", 765),
            "marker_numel": ("optimizer_state_numel", EXPECTED_LORA_NUMEL - 1),
            "optimizer": ("saved_optimizer", False),
            "rng": ("saved_rng", False),
        }[mutation]
        marker[field_and_value[0]] = field_and_value[1]
        marker_path.write_text(json.dumps(marker))
    else:
        torch.save(
            {"iteration": 63, "args": torch.load(iteration_dir / "common.pt", weights_only=False)["args"]},
            iteration_dir / "common.pt",
        )

    with pytest.raises(PreflightError, match=message):
        probe_checkpoint(root, 64, mode="native")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_name", "metadata/shard set mismatch"),
        ("truncated", "out-of-bounds storage extents"),
        ("parent_path", "invalid storage paths"),
    ],
)
def test_probe_rejects_unreadable_metadata_storage_contract(tmp_path, mutation, message):
    root = tmp_path / mutation
    iteration_dir = _write_checkpoint(root, iteration=64, distcp_count=32)
    metadata_path = iteration_dir / ".metadata"
    metadata = pickle.loads(metadata_path.read_bytes())
    first = metadata.storage_data[0]
    if mutation == "wrong_name":
        first.relative_path = "__missing_0.distcp"
    elif mutation == "truncated":
        first.length += 1
    else:
        first.relative_path = "../__0_0.distcp"
    metadata_path.write_bytes(pickle.dumps(metadata))

    with pytest.raises(PreflightError, match=message):
        probe_checkpoint(root, 64, mode="native")


def test_probe_rejects_truncated_metadata_with_clean_preflight_error(tmp_path):
    root = tmp_path / "corrupt_metadata"
    iteration_dir = _write_checkpoint(root, iteration=64, distcp_count=32)
    (iteration_dir / ".metadata").write_bytes(b"\x80")

    with pytest.raises(PreflightError, match="cannot read torch_dist storage metadata"):
        probe_checkpoint(root, 64, mode="native")


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("tensor_model_parallel_size", 2),
        ("pipeline_model_parallel_size", 2),
        ("context_parallel_size", 1),
        ("expert_model_parallel_size", 4),
        ("world_size", 8),
        ("cp_partition_mode", "zigzag"),
    ],
)
def test_probe_native_rejects_each_topology_mismatch(tmp_path, field, wrong):
    topology = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 2,
        "expert_model_parallel_size": 8,
        "world_size": 16,
        "cp_partition_mode": "contiguous",
    }
    topology[field] = wrong
    root = tmp_path / field
    _write_checkpoint(root, iteration=64, distcp_count=32, topology=topology)

    with pytest.raises(PreflightError, match=field):
        probe_checkpoint(root, 64, mode="native")


def test_cli_snapshot_and_resolve_emit_machine_readable_json(tmp_path):
    script = REPO / "scripts/dsv4/formal_resume_preflight.py"
    local = subprocess.run(
        [sys.executable, str(script), "snapshot", "--root", str(tmp_path / "local")],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = subprocess.run(
        [sys.executable, str(script), "snapshot", "--root", str(tmp_path / "remote")],
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = subprocess.run(
        [
            sys.executable,
            str(script),
            "resolve",
            "--local-snapshot",
            local.stdout,
            "--remote-snapshot",
            remote.stdout,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(resolved.stdout) == {
        "checkpoint_iteration": 59,
        "load_rng": 0,
        "mode": "migration",
        "start_rollout_id": 60,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
