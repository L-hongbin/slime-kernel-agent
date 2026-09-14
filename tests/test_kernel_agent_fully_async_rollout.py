from __future__ import annotations

import asyncio
import copy
import logging
import queue
import random
import sys
import threading
import time
from argparse import ArgumentParser, Namespace
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_path = str(REPO_ROOT)
if repo_root_path in sys.path:
    sys.path.remove(repo_root_path)
sys.path.insert(0, repo_root_path)
repo_examples_path = str(REPO_ROOT / "examples")
if (examples_package := sys.modules.get("examples")) is not None and hasattr(examples_package, "__path__"):
    # Megatron also ships a top-level ``examples`` package. If it was imported
    # first in a combined test process, include this repo's package path before
    # resolving ``examples.kernel_agent``.
    examples_package.__path__ = [
        repo_examples_path,
        *(path for path in examples_package.__path__ if path != repo_examples_path),
    ]

from examples.kernel_agent import fully_async_rollout, generate_with_cuda_agent, kernel_agent_data_source
from slime.utils import arguments as slime_arguments
from slime.utils import http_utils
from slime.utils.types import Sample

pytestmark = pytest.mark.unit
NUM_GPUS = 0


def _make_rollout_args(**overrides):
    values = dict(
        rollout_num_engines=None,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=4,
        sglang_server_concurrency=512,
        sglang_max_running_requests=32,
        n_samples_per_prompt=16,
        use_distributed_post=False,
        wandb_always_use_train_step=False,
        rollout_batch_size=16,
        global_batch_size=256,
        gen_weight_version=7,
    )
    values.update(overrides)
    return Namespace(**values)


def _make_group(index: int) -> list[Sample]:
    sample = Sample(index=index, group_index=index, prompt=f"p{index}")
    sample.status = Sample.Status.COMPLETED
    sample.reward = 0.0
    sample.response = "ok"
    sample.response_length = 1
    return [sample]


def _make_verify_candidate(index: int, source_group: int, failed_count: int, difficulty: float) -> Sample:
    assert failed_count > 0
    return Sample(
        index=index,
        group_index=source_group,
        response="### CUDA_KERNELS\n```cpp\nkernel code\n```",
        metadata={
            "role": "kernel",
            "source_group_index": source_group,
            "trajectory_states": ["failed"] * failed_count,
            "group_difficulty": difficulty,
        },
    )


def _make_schedulable_verify_candidate(
    index: int, source_group: int, version: int = 7, difficulty: float = 1.0
) -> Sample:
    sample = _make_verify_candidate(index, source_group=source_group, failed_count=1, difficulty=difficulty)
    sample.prompt = [{"role": "user", "content": "Implement the requested operator."}]
    sample.status = Sample.Status.COMPLETED
    sample.metadata.update(
        {
            "gen_weight_version": version,
            "env_result": {
                "env_state": {
                    "status": "completed",
                    "correctness": False,
                    "error": "output mismatch",
                }
            },
        }
    )
    return sample


def _make_verify_data_source_args(**overrides):
    values = dict(
        rollout_global_dataset=False,
        use_multi_turn=True,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        rollout_seed=7,
        gen_weight_version=8,
        verify_prompt_config_path=str(
            REPO_ROOT / "examples/kernel_agent/prompt_config/verify_prompt/tvm_ffi_correctness_v1.jinja"
        ),
        verify_rollout_ratio=1.0,
        capture_verify_data=True,
        verify_samples_per_group=1,
        verify_version_lag=2,
        verify_data_limit=float("inf"),
        load_verify_data=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_verify_sampling_weight_combines_version_gap_failure_count_and_group_difficulty():
    candidate = _make_verify_candidate(0, source_group=3, failed_count=2, difficulty=0.75)
    entry = kernel_agent_data_source.VerifyCandidateEntry(
        sample=candidate,
        source_weight_version=5,
        group_difficulty=0.75,
        difficulty_weight=2.25,
        insertion_sequence=0,
        capture_rollout_id=None,
        loaded=False,
    )

    assert entry.sampling_weight(current_weight_version=7) == pytest.approx(0.75)


def test_verify_capture_validation_enables_default_data_source_and_capture_hook():
    args = Namespace(
        capture_verify_data=True,
        verify_rollout_ratio=0.0,
        save_verify_data="/tmp/verify_{rollout_id}.pt",
        save_debug_rollout_data=None,
        data_source_path="slime.rollout.data_source.RolloutDataSourceWithBuffer",
        rollout_function_path="slime.rollout.sglang_rollout.generate_rollout",
        rollout_all_samples_process_path=None,
    )

    slime_arguments._validate_verify_capture_args(args)

    assert args.data_source_path == "examples.kernel_agent.kernel_agent_data_source.KernelAgentDataSource"
    assert (
        args.rollout_all_samples_process_path
        == "examples.kernel_agent.kernel_agent_data_source.capture_verify_candidates"
    )


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_verify_capture_validation_rejects_save_path_when_capture_is_disabled(ratio):
    args = Namespace(
        capture_verify_data=False,
        verify_rollout_ratio=ratio,
        use_multi_turn=True,
        save_verify_data="/tmp/verify_{rollout_id}.pt",
    )

    with pytest.raises(ValueError, match="requires online capture"):
        slime_arguments._validate_verify_capture_args(args)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_verify_rollout_without_capture_warns_and_keeps_data_source_wiring(caplog, capture, fixed):
    args = _make_verify_data_source_args(
        capture_verify_data=capture,
        load_verify_data=["/tmp/fixed_verify.pt"] if fixed else None,
        rollout_function_path="examples.kernel_agent.fully_async_rollout.generate_rollout_fully_async",
    )
    with caplog.at_level(logging.WARNING, logger=slime_arguments.logger.name):
        slime_arguments._validate_verify_capture_args(args)
    assert ("online failed-kernel capture is disabled" in caplog.text) is (not capture)
    assert args.data_source_path == "examples.kernel_agent.kernel_agent_data_source.KernelAgentDataSource"


def test_verify_capture_validation_requires_rollout_id_in_save_path():
    args = Namespace(
        capture_verify_data=True,
        verify_rollout_ratio=0.0,
        save_verify_data="/tmp/verify.pt",
        save_debug_rollout_data=None,
        data_source_path="examples.kernel_agent.kernel_agent_data_source.KernelAgentDataSource",
    )

    with pytest.raises(ValueError, match=r"must contain the \{rollout_id\} placeholder"):
        slime_arguments._validate_verify_capture_args(args)


def test_verify_capture_validation_requires_positive_ratio_for_fixed_data():
    args = Namespace(
        capture_verify_data=False,
        verify_rollout_ratio=0.0,
        load_verify_data=["/tmp/fixed_verify.pt"],
        save_verify_data=None,
    )

    with pytest.raises(ValueError, match="--load-verify-data requires a positive"):
        slime_arguments._validate_verify_capture_args(args)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"kernel_verify_max_turns": 0}, "kernel-verify-max-turns"),
        ({"kernel_verify_max_turns": -2}, "kernel-verify-max-turns"),
        ({"kernel_verify_max_turns": 1}, "kernel-verify-max-turns"),
        ({"kernel_verify_max_turns": 3}, "kernel-verify-max-turns"),
        ({"kernel_verify_max_turns": 4, "use_multi_turn": False}, "requires --use-multi-turn"),
        ({"kernel_verify_max_turns": 2, "use_multi_turn": False}, "requires --use-multi-turn"),
        ({"verify_advantage_baseline": "anchor", "verify_rollout_ratio": 0.0}, "verify-advantage-baseline"),
        ({"verify_advantage_baseline": "anchor", "group_rm": True}, "does not support --group-rm"),
        ({"verify_advantage_baseline": "history", "verify_rollout_ratio": 0.0}, "requires a positive"),
        ({"verify_advantage_baseline": "invalid"}, "must be group, history, or anchor"),
    ],
)
def test_verify_training_argument_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        slime_arguments._validate_verify_capture_args(_make_verify_data_source_args(**overrides))


def test_verify_advantage_baseline_cli_replaces_anchor_flag():
    parser = slime_arguments.get_slime_extra_args_provider()(ArgumentParser())
    defaults = parser.parse_args(["--rollout-batch-size", "1"])
    assert defaults.verify_advantage_baseline == "group"
    assert not hasattr(defaults, "verify_use_anchor")
    for mode in ("group", "history", "anchor"):
        args = parser.parse_args(["--rollout-batch-size", "1", "--verify-advantage-baseline", mode])
        assert args.verify_advantage_baseline == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--rollout-batch-size", "1", "--verify-use-anchor"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--rollout-batch-size", "1", "--verify-advantage-baseline", "source"])


def test_verify_sampling_cli_uses_short_names():
    parser = slime_arguments.get_slime_extra_args_provider()(ArgumentParser())
    defaults = parser.parse_args(["--rollout-batch-size", "1"])
    assert defaults.verify_samples_per_group == 1
    assert defaults.verify_version_lag == 2
    args = parser.parse_args(
        ["--rollout-batch-size", "1", "--verify-samples-per-group", "3", "--verify-version-lag", "4"]
    )
    assert args.verify_samples_per_group == 3
    assert args.verify_version_lag == 4


@pytest.mark.parametrize("limit", [-1, 1.5])
def test_verify_capture_validation_rejects_invalid_data_limit(limit):
    args = Namespace(
        capture_verify_data=False,
        verify_rollout_ratio=0.0,
        save_verify_data=None,
        verify_data_limit=limit,
    )

    with pytest.raises(ValueError, match="--verify-data-limit must be a non-negative integer or inf"):
        slime_arguments._validate_verify_capture_args(args)


def test_verify_data_paths_support_files_directories_and_globs(tmp_path):
    first = tmp_path / "verify_1.pt"
    second = tmp_path / "verify_2.pt"
    first.touch()
    second.touch()
    (tmp_path / "ignore.json").touch()

    from_directory = kernel_agent_data_source._resolve_verify_data_paths([str(tmp_path)])
    from_glob = kernel_agent_data_source._resolve_verify_data_paths([str(tmp_path / "verify_*.pt")])
    deduplicated = kernel_agent_data_source._resolve_verify_data_paths([str(first), str(tmp_path)])

    assert from_directory == [first.resolve(), second.resolve()]
    assert from_glob == from_directory
    assert deduplicated == from_directory


def test_select_verify_entries_is_without_replacement_and_caps_each_source_group():
    candidates = [
        _make_verify_candidate(index, source_group=0, failed_count=index + 1, difficulty=1.0) for index in range(4)
    ] + [
        _make_verify_candidate(index + 4, source_group=1, failed_count=index + 1, difficulty=0.5) for index in range(4)
    ]
    entries = []
    for sequence, candidate in enumerate(candidates):
        difficulty, difficulty_weight = kernel_agent_data_source._verify_difficulty_weight(candidate)
        entries.append(
            kernel_agent_data_source.VerifyCandidateEntry(
                sample=candidate,
                source_weight_version=7,
                group_difficulty=difficulty,
                difficulty_weight=difficulty_weight,
                insertion_sequence=sequence,
                capture_rollout_id=None,
                loaded=False,
            )
        )

    selected = kernel_agent_data_source.select_verify_entries(
        entries,
        target_size=5,
        max_samples_per_group=2,
        current_weight_version=8,
        rng=random.Random(7),
    )

    assert len(selected) == 4
    assert len({entry.sample.index for entry in selected}) == 4
    selected_per_group = Counter(entry.sample.metadata["source_group_index"] for entry in selected)
    assert selected_per_group == {0: 2, 1: 2}


def test_kernel_agent_data_source_versions_prunes_and_pops_verify_candidates():
    args = Namespace(
        rollout_global_dataset=False,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        rollout_seed=7,
        capture_verify_data=True,
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    candidates = [
        _make_verify_candidate(0, source_group=0, failed_count=2, difficulty=1.0),
        _make_verify_candidate(1, source_group=0, failed_count=1, difficulty=0.5),
        _make_verify_candidate(2, source_group=1, failed_count=3, difficulty=0.75),
    ]
    for candidate, version in zip(candidates, [7, 8, 9], strict=True):
        candidate.metadata["gen_weight_version"] = version

    assert data_source.add_verify_candidates(candidates) == 3
    assert data_source.get_verify_buffer_length() == 3
    assert "source_kernel_weight_version" not in candidates[0].metadata
    assert [sample.metadata["source_kernel_weight_version"] for sample in data_source.verify_buffer] == [7, 8, 9]

    assert data_source.prune_verify_candidates(current_weight_version=10, max_version_lag=2) == 1
    selected = data_source.pop_verify_candidates(
        target_size=2,
        max_samples_per_group=1,
        current_weight_version=10,
        max_version_lag=2,
    )

    assert len(selected) == 2
    assert {sample.metadata["source_group_index"] for sample in selected} == {0, 1}
    assert data_source.get_verify_buffer_length() == 0


def test_kernel_agent_data_source_keeps_retry_and_verify_buffers_independent():
    args = Namespace(
        rollout_global_dataset=False,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        rollout_seed=7,
        capture_verify_data=True,
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    retry_group = [Sample(index=0), Sample(index=1)]
    candidate = _make_verify_candidate(2, source_group=1, failed_count=1, difficulty=1.0)
    candidate.metadata["gen_weight_version"] = 7

    data_source.add_samples([retry_group])
    data_source.add_verify_candidates([candidate])

    assert data_source.get_buffer_length() == 1
    assert data_source.get_verify_buffer_length() == 1


def test_kernel_agent_data_source_without_verify_preserves_normal_data_flow():
    args = Namespace(
        rollout_global_dataset=False,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        rollout_seed=7,
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)

    groups = data_source.get_samples(2)

    assert len(groups) == 2
    assert [len(group) for group in groups] == [2, 2]
    assert [[sample.group_index for sample in group] for group in groups] == [[0, 0], [1, 1]]
    assert [[sample.index for sample in group] for group in groups] == [[0, 1], [2, 3]]
    assert data_source.get_verify_buffer_length() == 0

    retry_group = [Sample(index=10), Sample(index=11)]
    data_source.add_samples([retry_group])

    assert data_source.get_samples(1) == [retry_group]
    assert data_source.get_verify_buffer_length() == 0


def test_kernel_agent_data_source_mixes_verify_group_when_enabled():
    data_source = kernel_agent_data_source.KernelAgentDataSource(_make_verify_data_source_args())
    source = _make_schedulable_verify_candidate(index=42, source_group=9)
    assert data_source.add_verify_candidates([source]) == 1

    groups = data_source.get_samples(1)

    assert len(groups) == 1
    assert {sample.metadata["role"] for sample in groups[0]} == {"verify"}
    assert {sample.metadata["source_kernel_index"] for sample in groups[0]} == {42}
    assert data_source.get_verify_buffer_length() == 0


def test_verify_anchor_configuration_and_reference_metadata_are_copied_per_sample():
    data_source = kernel_agent_data_source.KernelAgentDataSource(
        _make_verify_data_source_args(verify_advantage_baseline="anchor")
    )
    source = _make_schedulable_verify_candidate(index=42, source_group=9)
    source.metadata.update(ground_truth="reference", entry_point="CustomModel", precision="bf16")
    data_source.add_verify_candidates([source])
    first, second = data_source.get_samples(1)[0]
    assert first.prompt == second.prompt
    key = first.metadata["verify_anchor_key"]
    assert key == second.metadata["verify_anchor_key"]
    assert len(data_source.anchor_kv) == 1
    anchor = data_source.anchor_kv[key]["sample"]
    assert anchor.index not in {first.index, second.index}
    assert anchor.group_index == first.group_index == second.group_index
    assert anchor.prompt == first.prompt[:-1]
    assert anchor.metadata["verify_source_env_result"] == source.metadata["env_result"]
    assert anchor.metadata["verify_source_env_result"] is not first.metadata["verify_source_env_result"]
    assert first.metadata["verify_source_env_result"] == source.metadata["env_result"]
    assert first.metadata["verify_source_env_result"] is not source.metadata["env_result"]
    assert first.metadata["ground_truth"] == "reference"
    assert first.metadata["entry_point"] == "CustomModel"
    assert first.metadata["precision"] == "bf16"
    assert "verify_anchor_key" not in source.metadata


@pytest.mark.parametrize("baseline", ["group", "history"])
def test_source_reward_is_carried_without_creating_anchor(baseline):
    args = _make_verify_data_source_args(verify_advantage_baseline=baseline)
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    source = _make_schedulable_verify_candidate(42, 9)
    source.reward = -0.25
    source.metadata["multi_turn_reward"] = 7.0
    data_source.add_verify_candidates([source])
    group = data_source.get_samples(1)[0]
    assert not data_source.anchor_kv
    assert data_source.sample_index == args.n_samples_per_prompt
    for sample in group:
        assert sample.reward is None
        assert "verify_anchor_key" not in sample.metadata
        if baseline == "history":
            assert sample.metadata["verify_source_reward"] == -0.25
        else:
            assert "verify_source_reward" not in sample.metadata
    assert "verify_source_reward" not in source.metadata


@pytest.mark.parametrize("reward", [None, float("nan"), float("inf"), {}, "invalid"])
def test_history_baseline_rejects_missing_or_invalid_source_reward(reward):
    data_source = kernel_agent_data_source.KernelAgentDataSource(
        _make_verify_data_source_args(verify_advantage_baseline="history")
    )
    source = _make_schedulable_verify_candidate(42, 9)
    source.reward = reward
    with pytest.raises(ValueError, match="finite source kernel reward"):
        data_source.add_verify_candidates([source])


@pytest.mark.parametrize("reward", [None, -0.25])
def test_history_baseline_loads_single_turn_reward_from_fixed_data(tmp_path, reward):
    source = _make_schedulable_verify_candidate(42, 9)
    source.reward = reward
    source.metadata["multi_turn_reward"] = 7.0
    fixed_path = tmp_path / "source.pt"
    torch.save({"rollout_id": 3, "samples": [source.to_dict()]}, fixed_path)
    args = _make_verify_data_source_args(verify_advantage_baseline="history", load_verify_data=[str(fixed_path)])
    if reward is None:
        with pytest.raises(ValueError, match="finite source kernel reward"):
            kernel_agent_data_source.KernelAgentDataSource(args)
    else:
        data_source = kernel_agent_data_source.KernelAgentDataSource(args)
        group = data_source.get_samples(1)[0]
        assert all(sample.metadata["verify_source_reward"] == reward for sample in group)
        assert not data_source.anchor_kv


def test_kernel_agent_data_source_falls_back_to_kernel_prompt_when_verify_buffer_is_empty():
    data_source = kernel_agent_data_source.KernelAgentDataSource(_make_verify_data_source_args())

    groups = data_source.get_samples(1)

    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert all(sample.metadata.get("role") is None for sample in groups[0])


def test_shared_anchor_claim_read_and_retry_lifecycle():
    data_source = kernel_agent_data_source.KernelAgentDataSource(
        _make_verify_data_source_args(verify_advantage_baseline="anchor")
    )
    data_source.add_verify_candidates([_make_schedulable_verify_candidate(42, 9)])
    group = data_source.get_samples(1)[0]
    key = group[0].metadata["verify_anchor_key"]
    claims = []
    threads = [threading.Thread(target=lambda: claims.append(data_source.claim_anchor(key))) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(claim is not None for claim in claims) == 1
    with pytest.raises(ValueError, match="not ready"):
        data_source.get_anchor_result(key)
    data_source.complete_anchor(key, {"reward": 0.3, "weight_versions": [8]})
    for _ in range(4):
        result = data_source.get_anchor_result(key)
        assert result == {"reward": 0.3, "weight_versions": [8]}
        result["weight_versions"].append(99)
    data_source.release_anchor(key)
    data_source.add_samples([copy.deepcopy(group)])
    retry = data_source.get_samples(1)[0]  # Even a verify-only mixture must drain retries.
    assert [sample.index for sample in retry] == [sample.index for sample in group]
    new_key = data_source.prepare_anchor(retry)
    assert new_key != key
    assert {sample.metadata["verify_anchor_key"] for sample in retry} == {new_key}
    assert data_source.claim_anchor(new_key).index == group[0].metadata["verify_anchor_index"]
    data_source.release_anchor(new_key)
    assert not data_source.anchor_kv


@pytest.mark.parametrize("anchor_version", [8, 9])
@pytest.mark.parametrize("capacity", [1, 3])
@pytest.mark.parametrize("invalid_first", [False, True])
def test_shared_anchor_once_for_all_candidates_and_turns(monkeypatch, anchor_version, capacity, invalid_first):
    args = _make_verify_data_source_args(
        verify_advantage_baseline="anchor", sglang_enable_deterministic_inference=True
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    data_source.add_verify_candidates([_make_schedulable_verify_candidate(42, 9)])
    group = data_source.get_samples(1)[0]
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker.__new__(
        fully_async_rollout.KernelAgentAsyncRolloutWorker
    )
    worker.args, worker.data_buffer = args, data_source
    worker.state = Namespace(group_sampling_seeds=[7, 8])
    calls = []

    async def run():
        gate = asyncio.Semaphore(capacity)
        candidates_started = asyncio.Event()

        async def fake_generate(args, sample, params, evaluation):
            async with gate:
                is_anchor = sample.metadata.get("verify_scoring_branch") == "anchor"
                calls.append((sample.index, params["sampling_seed"], is_anchor))
                if is_anchor:
                    if capacity > 1:
                        await candidates_started.wait()  # Candidates must not await the anchor.
                    return [Sample(response="direct", reward=0.3, weight_versions=[anchor_version], metadata={})]
                candidates_started.set()
                if invalid_first and sample.index == group[0].index:
                    sample.remove_sample = True
                    sample.reward = 0.0
                    sample.loss_mask = [0]
                    sample.metadata["turn_idx"] = 0
                    return [sample]
                turns = []
                for turn_idx in range(4):
                    turn = copy.deepcopy(sample)
                    turn.status = Sample.Status.COMPLETED
                    turn.reward = 0.8
                    turn.response_length = 1
                    turn.loss_mask = [1]
                    turn.weight_versions = [8]
                    turn.metadata.update(
                        role="verify" if turn_idx % 2 == 0 else "kernel",
                        turn_idx=turn_idx,
                        verify_reward_mode="pending_anchor",
                        verify_kernel_reward=0.8,
                        verify_scoring_weight_versions=["8"],
                        verify_scoring_versions_complete=True,
                        multi_turn_reward=0.8,
                    )
                    turns.append(turn)
                return turns

        monkeypatch.setattr(fully_async_rollout, "generate_and_rm", fake_generate)
        return await asyncio.wait_for(worker._generate_group(group, {}), timeout=2)

    output = asyncio.run(run())
    assert [call[2] for call in calls] == [True, False, False]
    assert [call[1] for call in calls] == [9, 7, 8]
    assert len({call[0] for call in calls}) == 3
    assert len(output) == 4
    assert [len(turn_group) for turn_group in output] == ([2, 1, 1, 1] if invalid_first else [2, 2, 2, 2])
    for turn_idx, turn_group in enumerate(output):
        for sample in turn_group:
            assert sample.index in {candidate.index for candidate in group}
            if invalid_first and sample.index == group[0].index:
                assert sample.remove_sample and sample.reward == 0.0
                continue
            if anchor_version != 8:
                assert sample.remove_sample and sample.loss_mask == [0]
                assert sample.reward == 0.0
            else:
                assert sample.reward == pytest.approx(0.5 if turn_idx % 2 == 0 else 0.8)
                assert sample.metadata["multi_turn_reward"] == sample.reward
            if turn_idx % 2 == 0:
                assert sample.metadata["verify_reward_mode"] == "anchor"
                assert sample.metadata["verify_anchor_reward"] == 0.3
                assert sample.metadata["verify_anchor_baseline"] == "fixed_source"
    assert not data_source.anchor_kv


@pytest.mark.parametrize("failure", ["anchor_error", "anchor_aborted", "anchor_nonfinite", "cancel", "all_invalid"])
def test_shared_anchor_group_failure_and_cancellation_cleanup(monkeypatch, failure):
    args = _make_verify_data_source_args(verify_advantage_baseline="anchor", advantage_estimator="trloo")
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    data_source.add_verify_candidates([_make_schedulable_verify_candidate(42, 9)])
    group = data_source.get_samples(1)[0]
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker.__new__(
        fully_async_rollout.KernelAgentAsyncRolloutWorker
    )
    worker.args, worker.data_buffer = args, data_source
    finished = []

    async def run():
        all_started = asyncio.Event()
        started = []

        async def fake_generate(args, sample, params, evaluation):
            started.append(sample.index)
            if len(started) == 3:
                all_started.set()
            try:
                await all_started.wait()
                if sample.metadata.get("verify_scoring_branch") == "anchor":
                    if failure == "anchor_error":
                        raise RuntimeError("anchor transport failed")
                    if failure == "anchor_aborted":
                        return [Sample(status=Sample.Status.ABORTED)]
                    if failure == "anchor_nonfinite":
                        return [Sample(reward=float("nan"))]
                elif failure == "all_invalid":
                    sample.remove_sample = True
                    sample.metadata["turn_idx"] = 0
                    return [sample]
                await asyncio.Future()
            finally:
                finished.append(sample.index)

        monkeypatch.setattr(fully_async_rollout, "generate_and_rm", fake_generate)
        task = asyncio.create_task(worker._generate_group(group, {}))
        await asyncio.wait_for(all_started.wait(), timeout=2)
        if failure == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return None
        return await asyncio.wait_for(task, timeout=2)

    output = asyncio.run(run())
    assert len(set(finished)) == 3
    assert not data_source.anchor_kv
    if failure not in {"cancel", "all_invalid"}:
        assert all(sample.status == Sample.Status.ABORTED for turn_group in output for sample in turn_group)
    elif failure == "all_invalid":
        assert all(sample.remove_sample for turn_group in output for sample in turn_group)


def test_shared_anchor_counts_toward_client_capacity():
    args = _make_rollout_args(verify_advantage_baseline="anchor", n_samples_per_prompt=2)
    assert fully_async_rollout._get_group_concurrency(args, 12) == 4
    assert fully_async_rollout._get_group_concurrency(args, 1) == 1


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_kernel_agent_data_source_does_not_record_verify_buffer_when_disabled(ratio):
    data_source = kernel_agent_data_source.KernelAgentDataSource(
        _make_verify_data_source_args(verify_rollout_ratio=ratio, capture_verify_data=False)
    )
    source = _make_schedulable_verify_candidate(index=42, source_group=9)
    assert data_source.add_verify_candidates([source]) == 0

    groups = data_source.get_samples(1)

    assert all(sample.metadata.get("role") is None for sample in groups[0])
    assert data_source.get_verify_buffer_length() == 0


def test_kernel_agent_verify_source_group_cap_persists_until_weight_version_changes():
    args = _make_verify_data_source_args()
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    candidates = [
        _make_schedulable_verify_candidate(index=40, source_group=9),
        _make_schedulable_verify_candidate(index=41, source_group=9),
    ]
    assert data_source.add_verify_candidates(candidates) == 2

    first = data_source.get_samples(1)
    capped_fallback = data_source.get_samples(1)

    assert first[0][0].metadata["role"] == "verify"
    assert capped_fallback[0][0].metadata.get("role") is None
    assert data_source.get_verify_buffer_length() == 1

    args.gen_weight_version = 9
    after_version_change = data_source.get_samples(1)

    assert after_version_change[0][0].metadata["role"] == "verify"
    assert data_source.get_verify_buffer_length() == 0


def test_kernel_agent_verify_capture_is_saved_even_after_candidate_is_consumed(tmp_path):
    save_path = str(tmp_path / "verify_{rollout_id}.pt")
    args = _make_verify_data_source_args(
        verify_rollout_ratio=0.0,
        capture_verify_data=True,
        save_verify_data=save_path,
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    source = _make_schedulable_verify_candidate(index=42, source_group=9)

    assert data_source.add_verify_candidates([source], rollout_id=3) == 1
    consumed = data_source.pop_verify_candidates(
        target_size=1,
        max_samples_per_group=1,
        current_weight_version=8,
        max_version_lag=2,
    )
    assert [sample.index for sample in consumed] == [42]
    assert data_source.get_verify_buffer_length() == 0
    assert len(data_source.verify_data) == 1
    assert data_source.verify_data[0].sample is consumed[0]
    assert not data_source.verify_data[0].available

    assert data_source.save_captured_verify_data(3) == 1

    saved = torch.load(tmp_path / "verify_3.pt", weights_only=False)
    assert saved["rollout_id"] == 3
    assert len(saved["samples"]) == 1
    assert saved["samples"][0]["index"] == 42
    assert saved["samples"][0]["metadata"]["source_kernel_weight_version"] == 7
    assert saved["samples"][0]["metadata"]["verify_capture_rollout_id"] == 3
    assert data_source.save_captured_verify_data(3) == 0
    assert len(torch.load(tmp_path / "verify_3.pt", weights_only=False)["samples"]) == 1

    data_source.begin_verify_capture(4)
    assert data_source.save_captured_verify_data(4) == 0
    assert torch.load(tmp_path / "verify_4.pt", weights_only=False)["samples"] == []


def test_kernel_agent_verify_data_limit_keeps_newer_and_harder_candidates_in_buffer_and_saved_data(tmp_path):
    save_path = str(tmp_path / "verify_{rollout_id}.pt")
    args = _make_verify_data_source_args(
        verify_rollout_ratio=0.0,
        capture_verify_data=True,
        save_verify_data=save_path,
        verify_data_limit=2,
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)

    data_source.begin_verify_capture(3)
    assert (
        data_source.add_verify_candidates(
            [
                _make_schedulable_verify_candidate(index=40, source_group=9, difficulty=0.2),
                _make_schedulable_verify_candidate(index=41, source_group=10, difficulty=0.8),
            ],
            rollout_id=3,
        )
        == 2
    )
    assert data_source.save_captured_verify_data(3) == 2

    data_source.begin_verify_capture(4)
    assert (
        data_source.add_verify_candidates(
            [_make_schedulable_verify_candidate(index=42, source_group=11, difficulty=0.5)], rollout_id=4
        )
        == 1
    )
    assert data_source.save_captured_verify_data(4) == 1

    data_source.begin_verify_capture(5)
    assert (
        data_source.add_verify_candidates(
            [_make_schedulable_verify_candidate(index=43, source_group=12, version=8, difficulty=0.1)], rollout_id=5
        )
        == 1
    )
    assert data_source.save_captured_verify_data(5) == 1

    assert data_source.get_verify_buffer_length() == 2
    assert {sample.index for sample in data_source.verify_buffer} == {41, 43}
    assert [sample["index"] for sample in torch.load(tmp_path / "verify_3.pt", weights_only=False)["samples"]] == [41]
    assert torch.load(tmp_path / "verify_4.pt", weights_only=False)["samples"] == []
    assert [sample["index"] for sample in torch.load(tmp_path / "verify_5.pt", weights_only=False)["samples"]] == [43]


def test_kernel_agent_verify_data_is_ordered_by_version_difficulty_and_insertion_time():
    data_source = kernel_agent_data_source.KernelAgentDataSource(
        _make_verify_data_source_args(verify_rollout_ratio=0.0, capture_verify_data=True)
    )

    data_source.add_verify_candidates(
        [
            _make_schedulable_verify_candidate(index=40, source_group=9, version=7, difficulty=0.5),
            _make_schedulable_verify_candidate(index=41, source_group=10, version=8, difficulty=0.1),
            _make_schedulable_verify_candidate(index=42, source_group=11, version=7, difficulty=0.5),
        ]
    )

    assert [entry.sample.index for entry in data_source.verify_data] == [40, 42, 41]


def test_kernel_agent_fixed_verify_data_replays_each_weight_version_without_online_capture(tmp_path):
    fixed_path = tmp_path / "verify_3.pt"
    source = _make_schedulable_verify_candidate(index=42, source_group=9, version=0)
    torch.save({"rollout_id": 3, "samples": [source.to_dict()]}, fixed_path)
    args = _make_verify_data_source_args(
        capture_verify_data=False,
        load_verify_data=[str(fixed_path)],
        gen_weight_version=100,
        verify_version_lag=1,
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)

    first = data_source.get_samples(1)

    assert first[0][0].metadata["role"] == "verify"
    assert first[0][0].metadata["source_kernel_index"] == 42
    assert first[0][0].metadata["source_kernel_weight_version"] == 0
    assert first[0][0].metadata["verify_data_origin"] == "loaded"
    assert first[0][0].metadata["verify_data_path"] == str(fixed_path.resolve())
    assert data_source.get_verify_buffer_length() == 0

    online = _make_schedulable_verify_candidate(index=43, source_group=10, version=100)
    assert data_source.add_verify_candidates([online], rollout_id=4) == 0

    args.gen_weight_version = 101
    replayed = data_source.get_samples(1)

    assert replayed[0][0].metadata["role"] == "verify"
    assert replayed[0][0].metadata["source_kernel_index"] == 42


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_ordinary_rollout_capture_hook_does_nothing_without_explicit_flag(ratio):
    args = Namespace(verify_rollout_ratio=ratio)
    failed = _make_schedulable_verify_candidate(42, 9)
    before = copy.deepcopy(failed.metadata)
    # Without capture enabled, the hook must not even require capture APIs or a rollout id.
    kernel_agent_data_source.capture_verify_candidates(args, [[failed]], object())
    assert failed.metadata == before


def test_ordinary_rollout_capture_hook_annotates_and_buffers_failed_kernel():
    args = _make_verify_data_source_args(
        verify_rollout_ratio=0.0,
        capture_verify_data=True,
        save_verify_data=None,
        gen_weight_version=None,
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    failed = _make_schedulable_verify_candidate(index=40, source_group=9)
    failed.metadata.pop("gen_weight_version")
    failed.metadata["env_extra_info"] = {"correctness": False}
    correct = _make_schedulable_verify_candidate(index=41, source_group=9)
    correct.metadata.pop("gen_weight_version")
    correct.metadata["trajectory_states"] = ["successed"]
    correct.metadata["env_extra_info"] = {"correctness": True}

    kernel_agent_data_source.capture_verify_candidates(
        args,
        [[failed, correct]],
        data_source,
        rollout_id=4,
    )

    assert failed.metadata["group_correct_rate"] == 0.5
    assert failed.metadata["group_difficulty"] == 0.5
    assert data_source.get_verify_buffer_length() == 1
    assert data_source.verify_buffer[0].index == 40
    assert data_source.verify_buffer[0].metadata["source_kernel_weight_version"] == 4


def test_kernel_agent_data_source_builds_fresh_verify_group_from_failed_kernel():
    args = Namespace(
        rollout_global_dataset=False,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        rollout_seed=7,
        capture_verify_data=True,
        verify_prompt_config_path=str(
            REPO_ROOT / "examples/kernel_agent/prompt_config/verify_prompt/tvm_ffi_correctness_v1.jinja"
        ),
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    source = Sample(
        index=42,
        group_index=9,
        prompt=[
            {"role": "system", "content": "Write CUDA kernels."},
            {"role": "user", "content": "Implement vector addition."},
        ],
        tokens=[1, 2, 3],
        response=(
            "<think>guess that indexing is wrong</think>\n"
            "Here is an attempted implementation.\n"
            "### CUDA_KERNELS\n```cpp\ninvalid kernel\n```\n"
            "This probably needs more work."
        ),
        response_length=3,
        label="reference",
        reward=-1.0,
        weight_versions=["7"],
        status=Sample.Status.COMPLETED,
        metadata={
            "role": "kernel",
            "turn_idx": 2,
            "gen_weight_version": 7,
            "trajectory_states": ["failed", "failed", "failed"],
            "group_num_correct": 1,
            "group_num_valid": 4,
            "group_correct_rate": 0.25,
            "group_difficulty": 0.75,
            "env_result": {
                "env_state": {
                    "status": "runtime_error",
                    "error": "illegal memory access",
                    "correctness": False,
                }
            },
        },
    )

    assert data_source.add_verify_candidates([source]) == 1
    groups = data_source.get_verify_samples(
        1,
        max_samples_per_group=1,
        current_weight_version=8,
        max_version_lag=1,
    )

    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert [sample.index for sample in groups[0]] == [0, 1]
    assert {sample.group_index for sample in groups[0]} == {0}
    for verify_sample in groups[0]:
        assert [message["role"] for message in verify_sample.prompt] == ["system", "user", "assistant", "user"]
        assert verify_sample.prompt[-2]["content"] == "### CUDA_KERNELS\n```cpp\ninvalid kernel\n```"
        assert "guess that indexing is wrong" not in verify_sample.prompt[-2]["content"]
        assert "attempted implementation" not in verify_sample.prompt[-2]["content"]
        assert "illegal memory access" in verify_sample.prompt[-1]["content"]
        assert "Do not generate kernel code yet." in verify_sample.prompt[-1]["content"]
        assert "TVM-FFI interface logic" in verify_sample.prompt[-1]["content"]
        assert verify_sample.response == ""
        assert verify_sample.tokens == []
        assert verify_sample.reward is None
        assert verify_sample.weight_versions == []
        assert verify_sample.status == Sample.Status.PENDING
        assert verify_sample.metadata == {
            "role": "verify",
            "source_kernel_index": 42,
            "source_group_index": 9,
            "source_turn_idx": 2,
            "source_kernel_weight_version": 7,
            "verify_source_env_result": source.metadata["env_result"],
            "trajectory_states": ["failed", "failed", "failed"],
            "group_num_correct": 1,
            "group_num_valid": 4,
            "group_correct_rate": 0.25,
            "group_difficulty": 0.75,
        }
    assert groups[0][0].prompt is not groups[0][1].prompt
    assert source.metadata["gen_weight_version"] == 7
    assert "source_kernel_weight_version" not in source.metadata
    assert data_source.get_verify_buffer_length() == 0


def test_kernel_agent_data_source_does_not_buffer_successful_kernel_for_verify():
    args = Namespace(
        rollout_global_dataset=False,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        rollout_seed=7,
        capture_verify_data=True,
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    source = _make_verify_candidate(1, source_group=0, failed_count=1, difficulty=0.5)
    source.metadata["trajectory_states"] = ["failed", "successed"]
    source.metadata["gen_weight_version"] = 7

    assert data_source.add_verify_candidates([source]) == 0
    assert data_source.get_verify_buffer_length() == 0


def test_kernel_agent_data_source_does_not_buffer_failed_response_without_kernel_sections():
    args = Namespace(
        rollout_global_dataset=False,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        rollout_seed=7,
        capture_verify_data=True,
    )
    data_source = kernel_agent_data_source.KernelAgentDataSource(args)
    source = _make_verify_candidate(1, source_group=0, failed_count=1, difficulty=0.5)
    source.response = "<think>I could not produce a kernel.</think> Sorry."
    source.metadata["gen_weight_version"] = 7

    assert data_source.add_verify_candidates([source]) == 0
    assert data_source.get_verify_buffer_length() == 0


def test_kernel_agent_group_concurrency_matches_client_capacity():
    args = _make_rollout_args()

    client_concurrency = http_utils.get_sglang_client_concurrency(args)
    group_concurrency = fully_async_rollout._get_group_concurrency(args, client_concurrency)

    assert client_concurrency == 64
    assert group_concurrency == 4


def test_kernel_agent_logs_cuda_agent_config_once_per_process(monkeypatch, caplog):
    class FakeThread:
        @staticmethod
        def is_alive():
            return True

    class FakeWorker:
        def __init__(self, args, data_buffer, concurrency):
            self.worker_thread = None

        def set_generation_context(self, rollout_id):
            pass

        def start(self):
            self.worker_thread = FakeThread()

    monkeypatch.setattr(fully_async_rollout, "_global_worker", None)
    monkeypatch.setattr(fully_async_rollout, "_config_logged", False)
    monkeypatch.setattr(fully_async_rollout, "KernelAgentAsyncRolloutWorker", FakeWorker)
    monkeypatch.setattr(fully_async_rollout, "get_sglang_client_concurrency", lambda args: 4)
    caplog.set_level(logging.INFO, logger=fully_async_rollout.logger.name)

    args = _make_rollout_args()
    fully_async_rollout._get_global_worker(args, data_buffer=None, rollout_id=0)
    fully_async_rollout._get_global_worker(args, data_buffer=None, rollout_id=1)

    assert caplog.text.count("CUDA_AGENT_CONFIGS=") == 1


def test_kernel_agent_rollout_leaves_surplus_completed_groups_queued(monkeypatch):
    class FakeGenerateState:
        def __init__(self, args):
            self.sampling_params = {}

    monkeypatch.setattr(fully_async_rollout, "GenerateState", FakeGenerateState)
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker(
        _make_rollout_args(),
        data_buffer=None,
        concurrency=4,
    )
    for gid in range(10):
        worker.output_queue.put((gid, _make_group(gid)))
    monkeypatch.setattr(fully_async_rollout, "_get_global_worker", lambda args, data_buffer, rollout_id: worker)

    args = _make_rollout_args(
        rollout_global_dataset=True,
        rollout_batch_size=4,
        dynamic_sampling_filter_path=None,
        use_multi_turn=False,
    )
    output = asyncio.run(fully_async_rollout._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert [group[0].index for group in output.samples] == [0, 1, 2, 3]
    assert worker.queue_size() == 6
    assert [gid for gid, _ in worker.get_completed_groups()] == [4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_kernel_agent_collector_adds_failed_candidates_after_difficulty_annotation(monkeypatch, capture, ratio):
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker.__new__(
        fully_async_rollout.KernelAgentAsyncRolloutWorker
    )
    worker.output_queue = queue.Queue()
    failed = _make_schedulable_verify_candidate(index=42, source_group=9)
    failed.metadata["env_extra_info"] = {"correctness": False}
    failed.reward = 0.0
    failed.response_length = 1
    worker.output_queue.put((0, [failed]))

    class VerifyBufferSpy:
        def __init__(self):
            self.candidates = []
            self.events = []

        def add_verify_candidates(self, samples, *, rollout_id=None):
            self.events.append(("add", rollout_id))
            self.candidates.extend(samples)
            return len(samples)

        def begin_verify_capture(self, rollout_id):
            self.events.append(("begin", rollout_id))

        def save_captured_verify_data(self, rollout_id):
            self.events.append(("save", rollout_id))
            return len(self.candidates)

    data_buffer = VerifyBufferSpy()
    monkeypatch.setattr(fully_async_rollout, "_get_global_worker", lambda args, data_buffer, rollout_id: worker)
    args = _make_rollout_args(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        dynamic_sampling_filter_path=None,
        use_multi_turn=False,
        verify_rollout_ratio=ratio,
        capture_verify_data=capture,
        save_verify_data="/tmp/verify_{rollout_id}.pt" if capture else None,
    )

    output = asyncio.run(fully_async_rollout._generate_rollout_async(args, rollout_id=0, data_buffer=data_buffer))

    assert output.samples == [[failed]]
    assert data_buffer.candidates == ([failed] if capture else [])
    assert failed.metadata["group_num_valid"] == 1
    assert failed.metadata["group_correct_rate"] == 0.0
    assert failed.metadata["group_difficulty"] == 1.0
    assert output.metrics["verify_candidates_added"] == int(capture)
    assert output.metrics["verify_candidates_saved"] == int(capture)
    assert data_buffer.events == ([("begin", 0), ("add", 0), ("save", 0)] if capture else [])


@pytest.mark.parametrize("joint_training", [False, True])
@pytest.mark.parametrize("filter_by_last_turn", [False, True])
def test_kernel_agent_collector_does_not_apply_kernel_filter_to_verify_group(
    monkeypatch, joint_training, filter_by_last_turn
):
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker.__new__(
        fully_async_rollout.KernelAgentAsyncRolloutWorker
    )
    worker.output_queue = queue.Queue()
    verify_group = _make_group(42)
    verify_group[0].metadata = {"role": "verify", "turn_idx": 0}
    groups = [verify_group]
    if joint_training:
        kernel_group = _make_group(42)
        kernel_group[0].metadata = {"role": "kernel", "turn_idx": 1, "verify_trajectory": True}
        groups.append(kernel_group)
    worker.output_queue.put((0, groups))

    monkeypatch.setattr(fully_async_rollout, "_get_global_worker", lambda args, data_buffer, rollout_id: worker)
    monkeypatch.setattr(fully_async_rollout, "load_function", lambda path: object())

    def reject_filter_call(*args, **kwargs):
        raise AssertionError("kernel dynamic filter must not receive verify groups")

    monkeypatch.setattr(fully_async_rollout, "call_dynamic_filter", reject_filter_call)
    args = _make_rollout_args(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        dynamic_sampling_filter_path="fake.kernel.filter",
        use_multi_turn=True,
        max_turns=2,
        filter_by_last_turn=filter_by_last_turn,
        verify_rollout_ratio=0.0,
    )

    output = asyncio.run(fully_async_rollout._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert output.samples == groups


def test_verify_trajectory_kernels_do_not_recursively_enter_capture_buffer():
    data_source = kernel_agent_data_source.KernelAgentDataSource(_make_verify_data_source_args())
    candidate = _make_schedulable_verify_candidate(1, source_group=3)
    candidate.metadata["verify_trajectory"] = True
    assert data_source.add_verify_candidates([candidate], rollout_id=7) == 0
    candidate.metadata.pop("verify_trajectory")
    assert data_source.add_verify_candidates([candidate], rollout_id=7) == 1


def test_kernel_agent_done_callback_never_blocks_on_full_queue(monkeypatch):
    class FakeGenerateState:
        def __init__(self, args):
            self.sampling_params = {}

    monkeypatch.setattr(fully_async_rollout, "GenerateState", FakeGenerateState)
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker(
        _make_rollout_args(),
        data_buffer=None,
        concurrency=4,
    )

    class DoneTask:
        def __init__(self, gid):
            self._result = _make_group(gid)

        def result(self):
            return self._result

    def push_all():
        for gid in range(1001):
            original_group = _make_group(gid)
            worker._make_done_cb(gid, original_group)(DoneTask(gid))

    pusher = threading.Thread(target=push_all, daemon=True)
    pusher.start()
    pusher.join(timeout=10)

    assert not pusher.is_alive(), "done callback blocked on the output queue"
    assert worker.queue_size() == 1001


def test_kernel_agent_worker_applies_queue_backpressure(monkeypatch):
    class FakeGenerateState:
        def __init__(self, args):
            self.sampling_params = {}

    class FakeDataBuffer:
        def __init__(self):
            self.groups = [_make_group(index) for index in range(60)]

        def get_samples(self, count):
            out = self.groups[:count]
            self.groups = self.groups[count:]
            return out

    async def instant_generate(args, group, sampling_params, evaluation):
        return group

    monkeypatch.setattr(fully_async_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", instant_generate)
    concurrency = 3
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker(
        _make_rollout_args(),
        FakeDataBuffer(),
        concurrency=concurrency,
    )
    worker.poll_interval = 0.01
    worker.set_generation_context(rollout_id=0)

    worker.start()
    try:
        deadline = time.monotonic() + 1.0
        max_seen = 0
        while time.monotonic() < deadline:
            max_seen = max(max_seen, worker.queue_size())
            if max_seen > 2 * concurrency:
                break
            time.sleep(0.02)
    finally:
        worker.stop()

    assert 0 < max_seen <= 2 * concurrency


def test_http_client_is_scoped_to_current_event_loop():
    old_client = http_utils._http_client
    old_clients_by_loop = dict(http_utils._http_clients_by_loop)
    old_client_concurrency = http_utils._client_concurrency

    http_utils._http_client = None
    http_utils._http_clients_by_loop = {}
    try:
        http_utils.init_http_client(_make_rollout_args())
        assert http_utils._http_client is None

        barrier = threading.Barrier(2)
        worker_result: list[tuple[int, int]] = []

        async def get_loop_and_client_ids():
            barrier.wait(timeout=5)
            client = http_utils.get_http_client()
            result = (id(asyncio.get_running_loop()), id(client))
            await client.aclose()
            return result

        def run_worker_loop():
            worker_result.append(asyncio.run(get_loop_and_client_ids()))

        thread = threading.Thread(target=run_worker_loop)
        thread.start()
        main_result = asyncio.run(get_loop_and_client_ids())
        thread.join(timeout=5)

        assert len(worker_result) == 1
        assert main_result[0] != worker_result[0][0]
        assert main_result[1] != worker_result[0][1]
    finally:
        http_utils._http_client = old_client
        http_utils._http_clients_by_loop = old_clients_by_loop
        http_utils._client_concurrency = old_client_concurrency


def test_kernel_agent_worker_does_not_exceed_group_concurrency(monkeypatch):
    class FakeGenerateState:
        def __init__(self, args):
            self.sampling_params = {}

    class FakeDataBuffer:
        def __init__(self):
            self.groups = [
                [Sample(index=group_idx * 2 + sample_idx, group_index=group_idx) for sample_idx in range(2)]
                for group_idx in range(6)
            ]

        def get_samples(self, count):
            out = self.groups[:count]
            self.groups = self.groups[count:]
            return out

        def add_samples(self, groups):
            self.groups.extend(groups)

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(0.05)
            for sample in group:
                sample.status = Sample.Status.COMPLETED
                sample.reward = 0.0
                sample.response = "ok"
                sample.response_length = 1
            return group
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr(fully_async_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)

    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker(
        _make_rollout_args(),
        FakeDataBuffer(),
        concurrency=2,
    )
    worker.set_generation_context(rollout_id=3)
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        completed = []
        while time.monotonic() < deadline and len(completed) < 4:
            completed.extend(worker.get_completed_groups())
            time.sleep(0.02)
    finally:
        worker.stop()

    assert len(completed) >= 4
    assert max_in_flight <= 2
    assert all(
        sample.metadata["rollout_step"] == 3 and sample.metadata["gen_weight_version"] == 7
        for _gid, group in completed
        for sample in group
    )


def test_kernel_agent_worker_prefers_engine_generation_weight_version(monkeypatch):
    class FakeGenerateState:
        def __init__(self, args):
            self.sampling_params = {}

    class FakeDataBuffer:
        def __init__(self):
            self.groups = [[Sample(index=0, group_index=0)]]

        def get_samples(self, count):
            out = self.groups[:count]
            self.groups = self.groups[count:]
            return out

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation):
        sample = group[0]
        # Simulate a request submitted under the v7 snapshot but admitted by
        # SGLang only after a pause/update/continue advanced the engine to v8.
        sample.weight_versions.append("8")
        sample.status = Sample.Status.COMPLETED
        sample.reward = 0.0
        return group

    monkeypatch.setattr(fully_async_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)

    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker(
        _make_rollout_args(gen_weight_version=7),
        FakeDataBuffer(),
        concurrency=1,
    )
    worker.poll_interval = 0.01
    worker.set_generation_context(rollout_id=4)
    worker.start()
    try:
        deadline = time.monotonic() + 2.0
        completed = []
        while time.monotonic() < deadline and not completed:
            completed = worker.get_completed_groups()
            time.sleep(0.01)
    finally:
        worker.stop()

    assert len(completed) == 1
    sample = completed[0][1][0]
    assert sample.metadata["rollout_step"] == 4
    assert sample.metadata["gen_weight_version"] == 8
    assert sample.metadata["engine_weight_version_span"] is False
    assert sample.metadata["engine_weight_version_mismatch"] is True


def test_kernel_agent_worker_restamps_a_fresh_retry_after_abort():
    args = _make_rollout_args(gen_weight_version=7, log_exp_metrics=True)
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker.__new__(
        fully_async_rollout.KernelAgentAsyncRolloutWorker
    )
    worker.args = args
    worker._generation_context_lock = threading.Lock()
    worker._rollout_step = None
    worker._gen_weight_version = None
    sample = Sample(index=1, metadata={"rollout_step": 3, "gen_weight_version": 6})

    worker.set_generation_context(rollout_id=4)
    worker._stamp_group_for_submission([sample])
    assert sample.metadata["rollout_step"] == 4
    assert sample.metadata["gen_weight_version"] == 7
    first_submit_time = sample.metadata["gen_submit_time"]

    # The aborted input is regenerated from its prompt, rather than resumed.
    # Its next attempt must describe the new generation policy, not retain v7.
    args.gen_weight_version = 8
    worker.set_generation_context(rollout_id=5)
    worker._stamp_group_for_submission([sample])
    assert sample.metadata["rollout_step"] == 5
    assert sample.metadata["gen_weight_version"] == 8
    assert sample.metadata["gen_submit_time"] >= first_submit_time


def test_kernel_agent_turn_preserves_engine_weight_version():
    base_sample = Sample(index=1, metadata={"rollout_step": 5, "gen_weight_version": 7})
    turn_sample = generate_with_cuda_agent._sample_for_turn(
        base_sample,
        prompt_ids=[1, 2],
        response="ok",
        response_ids=[3],
        log_probs=[-0.1],
        reward=0.0,
        status=Sample.Status.COMPLETED,
        turn_idx=0,
        env_result={"env_extra_info": {}},
        args=Namespace(sglang_speculative_algorithm=None, use_rollout_routing_replay=False),
        meta_info={"weight_version": "8"},
    )

    assert turn_sample.weight_versions == ["8"]


def test_kernel_agent_turn_preserves_top_p_replay_metadata():
    base_sample = Sample(index=1)
    turn_sample = generate_with_cuda_agent._sample_for_turn(
        base_sample,
        prompt_ids=[1, 2],
        response="ok",
        response_ids=[3, 4],
        log_probs=[-0.1, -0.2],
        reward=0.0,
        status=Sample.Status.COMPLETED,
        turn_idx=0,
        env_result={"env_extra_info": {}},
        args=Namespace(sglang_speculative_algorithm=None, use_rollout_routing_replay=False),
        meta_info={
            "top_p_token_ids": [3, 7, 4],
            "top_p_token_offsets": [0, 2, 3],
        },
    )

    assert turn_sample.rollout_top_p_token_ids.tolist() == [3, 7, 4]
    assert turn_sample.rollout_top_p_token_offsets.tolist() == [0, 2, 3]


def test_kernel_agent_turn_requires_sglang_top_p_metadata_for_real_tokens():
    with pytest.raises(ValueError, match="SGLang did not return top-p replay metadata"):
        generate_with_cuda_agent._sample_for_turn(
            Sample(index=1),
            prompt_ids=[1, 2],
            response="ok",
            response_ids=[3],
            log_probs=[-0.1],
            reward=0.0,
            status=Sample.Status.COMPLETED,
            turn_idx=0,
            env_result={"env_extra_info": {}},
            args=Namespace(
                rollout_top_p=0.95,
                sglang_speculative_algorithm=None,
                use_rollout_routing_replay=False,
            ),
            meta_info={"finish_reason": {"type": "stop"}},
        )


def test_kernel_agent_top_p_request_is_forced_for_each_turn():
    adjusted = generate_with_cuda_agent._sampling_params_for_prompt_context(
        Namespace(rollout_top_p=0.95, rollout_max_context_len=None),
        {"max_new_tokens": 10},
        prompt_token_count=3,
    )

    assert adjusted["custom_params"] == {"return_top_p_token_ids": True}


def test_kernel_agent_synthetic_samples_have_singleton_top_p_replay():
    base_sample = Sample(index=1)
    real_turn = Sample(metadata={"turn_idx": 0, "trajectory_states": ["failed", "failed"]})
    padded = generate_with_cuda_agent._pad_turn_samples(
        [real_turn],
        base_sample,
        max_turns=2,
        pad_token_id=42,
        pad_token="<pad>",
        use_top_p_replay=True,
    )
    aborted = generate_with_cuda_agent._abort_result(
        Namespace(
            use_multi_turn=False,
            dppo_predictive_top_k=0,
            rollout_top_p=0.95,
            max_turns=1,
        ),
        base_sample,
        "test_abort",
        1.0,
    )

    assert padded[1].rollout_top_p_token_ids == [42]
    assert padded[1].rollout_top_p_token_offsets == [0, 1]
    assert type(padded[1]) is Sample
    assert padded[1].metadata["role"] == "pad"
    assert padded[1].metadata["is_pad_turn"] is True
    assert padded[1].metadata["trajectory_states"] == ["failed", "failed"]
    assert aborted.rollout_top_p_token_ids == [0]
    assert aborted.rollout_top_p_token_offsets == [0, 1]


def test_kernel_agent_rollout_stats_only_keeps_surplus_completed_groups_queued(monkeypatch, caplog):
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker.__new__(
        fully_async_rollout.KernelAgentAsyncRolloutWorker
    )
    worker.output_queue = queue.Queue()
    for gid in range(5):
        sample = Sample(index=gid, group_index=gid, prompt=f"secret-prompt-{gid}")
        sample.status = Sample.Status.COMPLETED
        sample.reward = float(gid)
        sample.response = f"secret-response-{gid}"
        worker.output_queue.put((gid, [sample]))

    monkeypatch.setattr(fully_async_rollout, "_get_global_worker", lambda args, data_buffer, rollout_id: worker)
    monkeypatch.setitem(fully_async_rollout.CUDA_AGENT_CONFIGS, "log_rollout_stats_only", True)
    args = Namespace(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        dynamic_sampling_filter_path=None,
        use_multi_turn=False,
    )

    caplog.set_level(logging.INFO, logger=fully_async_rollout.logger.name)
    output = asyncio.run(fully_async_rollout._generate_rollout_async(args, rollout_id=7, data_buffer=None))

    assert [group[0].index for group in output.samples] == [0, 1]
    assert worker.queue_size() == 3
    assert [gid for gid, _ in worker.get_completed_groups()] == [2, 3, 4]
    assert "kernel-agent fully-async rollout 7: done" in caplog.text
    assert "accepted_groups=2" in caplog.text
    assert "secret-prompt" not in caplog.text
    assert "secret-response" not in caplog.text


def test_kernel_agent_rollout_logs_sample_bodies_when_stats_only_is_disabled(monkeypatch, caplog):
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker.__new__(
        fully_async_rollout.KernelAgentAsyncRolloutWorker
    )
    worker.output_queue = queue.Queue()
    for gid in range(2):
        sample = Sample(index=gid, group_index=gid, prompt=f"visible-prompt-{gid}")
        sample.status = Sample.Status.COMPLETED
        sample.reward = float(gid)
        sample.response = f"visible-response-{gid}"
        worker.output_queue.put((gid, [sample]))

    monkeypatch.setattr(fully_async_rollout, "_get_global_worker", lambda args, data_buffer, rollout_id: worker)
    monkeypatch.setitem(fully_async_rollout.CUDA_AGENT_CONFIGS, "log_rollout_stats_only", False)
    args = Namespace(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        dynamic_sampling_filter_path=None,
        use_multi_turn=False,
    )

    caplog.set_level(logging.INFO, logger=fully_async_rollout.logger.name)
    asyncio.run(fully_async_rollout._generate_rollout_async(args, rollout_id=8, data_buffer=None))

    assert "First kernel-agent fully-async rollout sample" in caplog.text
    assert "visible-prompt-0visible-response-0" in caplog.text
    assert "kernel-agent fully-async rollout 8: done" in caplog.text
    assert "visible-prompt-1visible-response-1" in caplog.text


def test_kernel_agent_completed_group_drain_honors_limit():
    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker.__new__(
        fully_async_rollout.KernelAgentAsyncRolloutWorker
    )
    worker.output_queue = queue.Queue()
    for gid in range(4):
        worker.output_queue.put((gid, [Sample(index=gid)]))

    assert [gid for gid, _ in worker.get_completed_groups(limit=2)] == [0, 1]
    assert [gid for gid, _ in worker.get_completed_groups()] == [2, 3]


def test_kernel_agent_worker_cancels_inflight_tasks_on_stop(monkeypatch):
    class FakeGenerateState:
        def __init__(self, args):
            self.sampling_params = {}

    class FakeDataBuffer:
        def __init__(self):
            self.groups = [[Sample(index=group_idx, group_index=group_idx)] for group_idx in range(2)]

        def get_samples(self, count):
            out = self.groups[:count]
            self.groups = self.groups[count:]
            return out

    started = threading.Event()
    cancelled = threading.Event()

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(fully_async_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)

    worker = fully_async_rollout.KernelAgentAsyncRolloutWorker(
        _make_rollout_args(),
        FakeDataBuffer(),
        concurrency=1,
    )
    worker.set_generation_context(rollout_id=0)
    worker.start()
    try:
        assert started.wait(timeout=2.0)
    finally:
        worker.stop()

    assert cancelled.wait(timeout=2.0)
    assert worker.worker_thread is not None
    assert not worker.worker_thread.is_alive()
    assert worker.exception_count == 0
    assert worker.active_count == 0


def test_kernel_agent_http_client_does_not_cancel_slow_posts():
    class SlowJsonHandler(BaseHTTPRequestHandler):
        broken_pipes = 0
        seen = 0
        lock = threading.Lock()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            self.rfile.read(length)
            with self.lock:
                type(self).seen += 1

            time.sleep(0.15)
            payload = b'{"ok":true}'
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except BrokenPipeError:
                with self.lock:
                    type(self).broken_pipes += 1

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowJsonHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/generate"

    async def run_posts():
        async with httpx.AsyncClient(
            limits=httpx.Limits(max_connections=8),
            timeout=httpx.Timeout(None),
            trust_env=False,
        ) as client:
            return await asyncio.gather(
                *[http_utils._post(client, url, {"idx": idx}, max_retries=1) for idx in range(8)]
            )

    try:
        results = asyncio.run(run_posts())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert results == [{"ok": True}] * 8
    assert SlowJsonHandler.seen == 8
    assert SlowJsonHandler.broken_pipes == 0


def test_full_async_kernel_agent_script_guards_critical_config():
    script = (REPO_ROOT / "examples/kernel_agent/run.t1.qwen3.6.27B.fasync.sh").read_text()

    # Router retries/circuit-breaker must stay enabled (matches reference runs).
    assert "--router-disable-retries" not in script
    assert "--router-disable-circuit-breaker" not in script
    # In-cluster HTTP must bypass the egress proxy in BOTH spellings.
    assert '"no_proxy": "${NO_PROXY_LIST}"' in script
    assert '"NO_PROXY": "${NO_PROXY_LIST}"' in script
    # Async checkpointing requires the persistent worker or Megatron disables it.
    assert "--async-save" in script
    assert "--use-persistent-ckpt-worker" in script
    assert "--save-interval" in script
    assert "--save " in script or "--save $" in script
    # Qwen thinking must stay on for prompt_tvm_v2 data.
    assert "--apply-chat-template-kwargs '{\"enable_thinking\":true}'" in script
    assert "MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-${MAX_CONTEXT_LEN}}" in script
    assert '"${ROLLOUT_ARGS[@]}"' in script


def test_cuda_agent_sglang_post_is_fail_fast_by_default():
    source = (REPO_ROOT / "examples/kernel_agent/generate_with_cuda_agent.py").read_text()

    assert 'rollout_request_max_retries = int(CUDA_AGENT_CONFIGS.get("rollout_request_max_retries", 60))' in source
    assert "post(url, payload, max_retries=rollout_request_max_retries)" in source
    assert "_generate_max_retries" not in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
