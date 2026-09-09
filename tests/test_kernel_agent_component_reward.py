"""CPU contracts for opt-in best-answer component credit and packed TRLOO."""

from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The installed Megatron package also owns a top-level examples package.
import examples

examples.__path__ = [str(ROOT / "examples"), *examples.__path__]

import test_trloo_trajectory_packing as packing_tests
from examples.kernel_agent.component_reward import (
    GRAPH_SCHEMA,
    WIRE_SCHEMA,
    ComponentRewardContractError,
    attribute_best_components,
    compute_component_reward_metrics,
    filter_component_reward_group,
)
from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.kernel_reward import reward_post_process_by_group
from examples.kernel_agent.utils import postprocess_turn_samples
from test_trloo_trajectory_packing import TinyCausalModel, _args, _loss_and_grad, _manager, _samples

from slime.utils.arguments import _validate_component_reward_args
from slime.utils.types import Sample

cpu_mpu = packing_tests.cpu_mpu


def args(**changes):
    return SimpleNamespace(
        **{
            "component_reward": True,
            "runtime_graph_timeout": 60.0,
            "use_multi_turn": True,
            "advantage_estimator": "trloo",
            "multi_turn_gamma": 1.0,
            "reward_key": None,
            "custom_reward_post_process_path": "examples.kernel_agent.kernel_reward.reward_post_process_by_group",
            "dynamic_sampling_filter_path": "examples.kernel_agent.component_reward.filter_component_reward_group",
            "filter_by_last_turn": True,
            "finalize_mode": "none",
            "use_coverage_rs": False,
            "overlong_penalty": False,
            "grpo_std_normalization": False,
            "use_conditional_truncation_mask": False,
            "n_samples_per_prompt": 2,
            "target_group_size": 2,
            "min_group_size": 2,
            "reward_std_threshold": 1e-3,
            **changes,
        }
    )


def observation(implementations, *, candidate="candidate", unknown=(), unused=None):
    nodes, edges, buffers = (
        [],
        [],
        [
            {"id": "input", "roles": ["input:0"], "bytes": 4},
            {"id": "output", "roles": ["output:0"], "bytes": 4},
        ],
    )
    for i, implementation in enumerate(implementations):
        buf, key = f"temp{i}", f"call{i}"
        buffers.append({"id": buf, "roles": [f"slot:{i}"], "bytes": 4})
        nodes.append(
            {
                "id": key,
                "kind": "kernel",
                "implementation": implementation,
                "configuration": {"tile": 4},
                "footprint_complete": i not in unknown,
                "unknowns": ["unsupported_memory_instruction"] if i in unknown else [],
                "reads": [{"buffer": "input", "regions": [[0, 4]], "port": "in"}],
                "writes": [{"buffer": buf, "regions": [[0, 4]], "port": "out"}],
            }
        )
        edges.append(
            {
                "source": key,
                "target": "join",
                "buffer": buf,
                "regions": [[0, 4]],
                "certainty": "proven_region_dependency",
            }
        )
    # Synthetic output join represents observation bookkeeping, not a copy call.
    nodes.append(
        {
            "id": "join",
            "kind": "output_join",
            "footprint_complete": True,
            "unknowns": [],
            "reads": [],
            "writes": [{"buffer": "output", "regions": [[0, 4]]}],
        }
    )
    if unused:
        nodes.append(
            {
                "id": "unused",
                "kind": "kernel",
                "implementation": unused,
                "configuration": {"tile": 4},
                "footprint_complete": True,
                "unknowns": [],
                "reads": [],
                "writes": [],
            }
        )
    return {
        "schema": WIRE_SCHEMA,
        "status": "partial" if unknown else "ok",
        "alignment": {"scored_control": True, "control_trace": True},
        "identity": {
            "task_sha256": "task",
            "input_signature": "input",
            "state_signature": "state",
            "environment_signature": "env",
            "collector_sha256": "collector",
            "candidate_source_sha256": candidate,
        },
        "graph": {
            "schema": GRAPH_SCHEMA,
            "nodes": nodes,
            "edges": edges,
            "buffers": buffers,
            "outputs": [
                {
                    "source": "join",
                    "target": "forward_output",
                    "buffer": "output",
                    "regions": [[0, 4]],
                    "certainty": "proven_region_dependency",
                }
            ],
            "output_unknowns": [],
            "coverage": {
                "summary_complete": True,
                "trace_process_complete": True,
                "kernel_launches": len(implementations),
                "completed_kernel_launches": len(implementations),
            },
        },
    }


def trajectory(programs, scores, *, index=0):
    samples = []
    prefix = [1, 2]
    for turn, (program, score) in enumerate(zip(programs, scores, strict=True)):
        response = [10 + turn, 20 + turn]
        sample = Sample(
            index=index,
            group_index=0,
            group_id=index,
            tokens=prefix + response,
            response_length=2,
            response=str(program),
            reward=score,
            loss_mask=[1, 1],
            rollout_log_probs=[-1.0, -1.0],
            status=Sample.Status.COMPLETED,
            metadata={"turn_idx": turn, "env_extra_info": {"correctness": score >= 0.5, "decoy_kernel": False}},
        )
        if program is not None:
            sample.metadata["runtime_graph"] = observation(program, candidate=str(turn))
        samples.append(sample)
        prefix += response + [3]
    return samples


def test_mass_conserving_origin_example_and_no_future_fold():
    samples = trajectory([["a", "b", "x"], ["a", "b", "y"], ["a", "b", "c"]], [0.0, 0.1, 1.2])
    postprocess_turn_samples(args(), samples, "max_turns")
    assert [s.reward for s in samples] == pytest.approx([0.8, 0.0, 0.4])
    assert [s.metadata["multi_turn_reward"] for s in samples] == pytest.approx([0.8, 0.0, 0.4])
    assert math.fsum(s.reward for s in samples) == pytest.approx(1.2)
    assert samples[0].metadata["component_reward"]["resolved_cross_turn_units"] == 2


def test_reappearance_uses_earliest_observed_turn():
    samples = trajectory([["a"], ["b"], ["a"]], [0.0, 0.0, 1.0])
    result = attribute_best_components(samples, [0.0, 0.0, 1.0])
    assert result["credits"] == [1.0, 0.0, 0.0]
    assert result["units"][0]["status"] == "reappeared"
    assert result["units"][0]["origin_turn"] == 0


def test_best_is_earliest_tie_not_terminal():
    samples = trajectory([["a"], ["b"], ["c"]], [1.0, 1.0, 0.2])
    result = attribute_best_components(samples, [1.0, 1.0, 0.2])
    assert result["best_turn"] == 0
    assert result["credits"] == [1.0, 0.0, 0.0]


def test_missing_best_graph_is_explicit_best_not_terminal_fallback():
    samples = trajectory([None, ["b"], ["c"]], [1.2, 0.3, 0.1])
    result = attribute_best_components(samples, [1.2, 0.3, 0.1])
    assert result["credits"] == [1.2, 0.0, 0.0]
    assert result["status"] == "unavailable_best_turn_fallback"
    assert result["residual_fraction"] == 1.0
    assert result["unknowns"] == ["missing_runtime_graph"]


def test_partial_best_unit_mass_is_not_discarded():
    samples = trajectory([["a", "x"], ["a", "b"]], [0.0, 1.0])
    samples[1].metadata["runtime_graph"] = observation(["a", "b"], unknown=[1])
    result = attribute_best_components(samples, [0.0, 1.0])
    assert result["credits"] == [0.5, 0.5]
    assert result["residual_fraction"] == 0.5
    assert result["status"] == "partial"


@pytest.mark.parametrize("complete", [False, None])
def test_capture_completion_does_not_authorize_truncated_http_summary(complete):
    samples = trajectory([["a"], ["a"]], [0, 1])
    coverage = samples[1].metadata["runtime_graph"]["graph"]["coverage"]
    if complete is None:
        coverage.pop("summary_complete")
    else:
        coverage["summary_complete"] = complete
    assert coverage["trace_process_complete"]
    assert coverage["kernel_launches"] == coverage["completed_kernel_launches"]
    result = attribute_best_components(samples, [0, 1])
    assert result["credits"] == [0, 1]
    assert result["status"] == "unavailable_best_turn_fallback"
    assert result["unknowns"] == ["incomplete_graph_summary"]


def test_missing_launch_counts_cannot_compare_equal_as_none():
    samples = trajectory([["a"]], [1])
    coverage = samples[0].metadata["runtime_graph"]["graph"]["coverage"]
    coverage.pop("kernel_launches")
    coverage.pop("completed_kernel_launches")
    result = attribute_best_components(samples, [1])
    assert result["status"] == "unavailable_best_turn_fallback"
    assert result["unknowns"] == ["unknown_launch_completion_counts"]


def test_unknown_output_closure_does_not_normalize_observed_prefix():
    samples = trajectory([["a"], ["a"]], [0.0, 1.0])
    samples[1].metadata["runtime_graph"]["graph"]["output_unknowns"] = ["lost_output_dependency"]
    result = attribute_best_components(samples, [0.0, 1.0])
    graph = samples[1].metadata["runtime_graph"]["graph"]
    uncertain = copy.deepcopy(graph["nodes"][0])
    uncertain.update(id="possibly_used", implementation="unobserved_dependency")
    graph["nodes"].append(uncertain)
    result = attribute_best_components(samples, [0.0, 1.0])
    assert result["credits"] == [0.5, 0.5]
    assert result["status"] == "partial"
    assert len(result["units"]) == 2


def test_dead_and_no_data_effect_units_do_not_inflate_budget():
    samples = trajectory([["a"], ["a"]], [0.0, 1.0])
    samples[1].metadata["runtime_graph"] = observation(["a"], unused="dead")
    result = attribute_best_components(samples, [0.0, 1.0])
    assert len(result["units"]) == 1
    assert result["credits"] == [1.0, 0.0]


@pytest.mark.parametrize("field", ["input_signature", "environment_signature", "collector_sha256"])
def test_valid_context_difference_is_unresolved_not_transport_error(field):
    samples = trajectory([["a"], ["a"]], [0.0, 1.0])
    samples[0].metadata["runtime_graph"]["identity"][field] = "different"
    result = attribute_best_components(samples, [0.0, 1.0])
    assert result["credits"] == [0.0, 1.0]
    assert result["status"] == "partial"
    assert f"incomparable_context:{field}" in result["unknowns"]


def test_wrong_request_binding_is_hard_error():
    samples = trajectory([["a"]], [1.0])
    samples[0].metadata["runtime_graph_expected_identity"] = {"candidate_source_sha256": "other-request"}
    with pytest.raises(ComponentRewardContractError, match="transport identity mismatch"):
        attribute_best_components(samples, [1.0])


def test_unrelated_candidate_state_change_preserves_observed_kernel_membership():
    samples = trajectory([["a"], ["a"]], [0, 1])
    payload = samples[0].metadata["runtime_graph"]
    payload["identity"]["state_signature"] = "different_candidate_state"
    payload["graph"]["buffers"].append({"id": "unrelated_state", "roles": ["parameter:extra"], "bytes": 4})
    result = attribute_best_components(samples, [0, 1])
    assert result["credits"] == [1, 0]
    assert result["resolved_cross_turn_units"] == 1
    assert result["cross_turn_state_changes"] == [0]


def test_state_identity_remains_required_for_same_turn_diagnostic():
    samples = trajectory([["a"]], [1])
    samples[0].metadata["runtime_graph"]["identity"].pop("state_signature")
    with pytest.raises(ComponentRewardContractError, match="identity missing state_signature"):
        attribute_best_components(samples, [1])


def test_schema_mismatch_is_not_empty_success():
    samples = trajectory([["a"]], [1.0])
    samples[0].metadata["runtime_graph"]["schema"] = "wrong"
    with pytest.raises(ComponentRewardContractError, match="schema"):
        attribute_best_components(samples, [1.0])


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_invalid_graph_is_contract_error_even_without_quality_budget(score):
    samples = trajectory([["a"]], [score])
    samples[0].metadata["runtime_graph"]["status"] = "invalid"
    with pytest.raises(ComponentRewardContractError, match="invalid runtime_graph"):
        attribute_best_components(samples, [score])


@pytest.mark.parametrize("field", ["scored_control", "control_trace"])
@pytest.mark.parametrize("value", [None, False])
def test_unverified_execution_alignment_is_not_partial_attribution(field, value):
    samples = trajectory([["a"], ["a"]], [0, 1])
    payload = samples[1].metadata["runtime_graph"]
    payload["status"] = "partial"
    payload["alignment"][field] = value
    result = attribute_best_components(samples, [0, 1])
    assert result["credits"] == [0, 1]
    assert result["status"] == "unavailable_best_turn_fallback"
    assert f"unverified_alignment:{field}" in result["unknowns"]


@pytest.mark.parametrize("mode", ["positive", "improve"])
def test_wrong_contributing_turn_survives_soft_finalize(mode):
    samples = trajectory([["x"], ["a"], ["a"]], [0.5, 0.0, 1.0])
    postprocess_turn_samples(args(finalize_mode=mode), samples, "max_turns")
    assert samples[1].reward == 1.0
    assert not samples[1].remove_sample
    assert samples[1].metadata["component_reward_soft_finalize_protected"]


@pytest.mark.parametrize("reason", ["pad", "abort", "removed", "zero_tokens", "zero_mask", "decoy"])
def test_hard_or_decoy_turn_cannot_receive_positive_credit(reason):
    samples = trajectory([["a"], ["a"]], [2.0, 1.0])
    if reason == "pad":
        samples[0].metadata["is_pad_turn"] = True
    if reason == "abort":
        samples[0].status = Sample.Status.ABORTED
    if reason == "removed":
        samples[0].remove_sample = True
    if reason == "zero_tokens":
        samples[0].response_length = 0
    if reason == "zero_mask":
        samples[0].loss_mask = [0, 0]
    if reason == "decoy":
        samples[0].metadata["env_extra_info"]["decoy_kernel"] = True
    result = attribute_best_components(samples, [2.0, 1.0])
    assert result["best_turn"] == 1
    assert result["credits"][0] == 0.0


def test_negative_reward_preserved_without_future_fold():
    samples = trajectory([None, None], [-0.2, 0.25])
    postprocess_turn_samples(args(), samples, "max_turns")
    assert [s.reward for s in samples] == [-0.2, 0.25]
    assert samples[1].metadata["component_reward"]["status"] == "unavailable_best_turn_fallback"


def test_all_zero_is_no_budget_not_successful_attribution():
    samples = trajectory([None, None], [0.0, 0.0])
    postprocess_turn_samples(args(), samples, "max_turns")
    assert samples[0].metadata["component_reward"]["status"] == "no_positive_budget"


def test_turn_loo_uses_direct_targets():
    a = trajectory([["a"], ["a"]], [0.0, 1.0], index=0)
    b = trajectory([["x"], ["b"]], [0.0, 1.0], index=1)
    for group in [a, b]:
        postprocess_turn_samples(args(), group, "max_turns")
    raw, advantages = reward_post_process_by_group(args(), [a[0], b[0], a[1], b[1]])
    assert raw == [1.0, 0.0, 0.0, 1.0]
    assert advantages == [1.0, -1.0, -1.0, 1.0]


def test_vector_filter_keeps_equal_best_scores_with_different_earlier_origins(monkeypatch):
    monkeypatch.setitem(
        CUDA_AGENT_CONFIGS, "filter", {"reject_low_variance_groups": True, "reject_small_groups": True}
    )
    a = trajectory([["a"], ["x"], ["a"]], [0, 0, 1], index=0)
    b = trajectory([["x"], ["a"], ["a"]], [0, 0, 1], index=1)
    for group in [a, b]:
        postprocess_turn_samples(args(), group, "max_turns")
    assert a[-1].reward == b[-1].reward == 0
    assert filter_component_reward_group(args(), [a[-1], b[-1]]).keep


def test_filter_uses_any_trainable_turn_when_last_turn_removed(monkeypatch):
    monkeypatch.setitem(
        CUDA_AGENT_CONFIGS, "filter", {"reject_low_variance_groups": True, "reject_small_groups": True}
    )
    a = trajectory([["a"], None], [1, 0], index=0)
    b = trajectory([["b"], None], [2, 0], index=1)
    for group in [a, b]:
        postprocess_turn_samples(args(finalize_mode="positive"), group, "max_turns")
    assert a[-1].remove_sample and b[-1].remove_sample
    assert filter_component_reward_group(args(), [a[-1], b[-1]]).keep


def test_default_off_matches_existing_rewards_metadata_and_rng():
    original = trajectory([None, None, None], [0.5, 0.0, 1.0])
    absent, disabled = copy.deepcopy(original), copy.deepcopy(original)
    base = args(component_reward=False, finalize_mode="positive")
    no_flag = copy.copy(base)
    del no_flag.component_reward
    random.seed(31)
    before = random.getstate()
    postprocess_turn_samples(no_flag, absent, "max_turns")
    rng_a = random.getstate()
    random.setstate(before)
    postprocess_turn_samples(base, disabled, "max_turns")
    rng_b = random.getstate()
    assert absent == disabled
    assert rng_a == rng_b
    assert all("component_reward" not in s.metadata for s in disabled)
    assert [s.metadata["multi_turn_reward"] for s in disabled] == [1.5, 1.0, 1.0]
    assert reward_post_process_by_group(no_flag, absent) == reward_post_process_by_group(base, disabled)


@pytest.mark.parametrize(
    "override",
    [
        {"use_multi_turn": False},
        {"advantage_estimator": "grpo"},
        {"runtime_graph_timeout": 0},
        {"runtime_graph_timeout": float("inf")},
        {"runtime_graph_timeout": 121},
        {"custom_reward_post_process_path": None},
        {"dynamic_sampling_filter_path": "examples.kernel_agent.kernel_filter.filter_cuda_kernel_group"},
        {"filter_by_last_turn": False},
    ],
)
def test_incompatible_args_rejected(override):
    with pytest.raises(ValueError):
        _validate_component_reward_args(args(**override))


def test_valid_args_and_off_mode_are_supported():
    _validate_component_reward_args(args())
    _validate_component_reward_args(SimpleNamespace(component_reward=False))


def test_later_ambiguous_repetition_does_not_erase_earliest_unique_observation():
    samples = trajectory([["a"], ["a"], ["a"]], [0, 0, 1])
    graph = samples[1].metadata["runtime_graph"]["graph"]
    duplicate = copy.deepcopy(graph["nodes"][0])
    duplicate["id"] = "duplicate"
    graph["nodes"].append(duplicate)
    result = attribute_best_components(samples, [0, 0, 1])
    assert result["credits"] == [1, 0, 0]
    assert "ambiguous_origin_turn:1" in result["units"][0]["unknowns"]


def test_overlong_penalty_applied_once_after_quality_selection():
    samples = trajectory([["a"], ["a"]], [0, 1])
    postprocess_turn_samples(
        args(overlong_penalty=True, overlong_buffer_len=2, rollout_max_response_len=2, overlong_penalty_factor=0.2),
        samples,
        "max_turns",
    )
    assert [s.reward for s in samples] == pytest.approx([0.8, -0.2])
    assert samples[0].metadata["component_reward"]["best_turn"] == 1
    assert samples[0].metadata["component_reward"]["length_penalties"] == pytest.approx([0.2, 0.2])


def test_model_abort_never_restores_quality_budget():
    samples = trajectory([["a"], ["a"]], [0, 1])
    postprocess_turn_samples(args(), samples, "model_abort")
    assert all(s.remove_sample and s.reward == 0 for s in samples)
    assert samples[0].metadata["component_reward"]["quality_budget"] == 0


@pytest.mark.parametrize("kind", ["copy", "fill", "clone"])
def test_output_memory_transform_is_a_unit(kind):
    samples = trajectory([["a"], ["a"]], [0, 1])
    for sample in samples:
        node = sample.metadata["runtime_graph"]["graph"]["nodes"][0]
        node["kind"] = kind
    result = attribute_best_components(samples, [0, 1])
    assert len(result["units"]) == 1
    assert result["credits"] == [1, 0]


def test_library_internal_kernel_is_not_double_counted():
    samples = trajectory([["gemm"], ["gemm"]], [0, 1])
    for sample in samples:
        graph = sample.metadata["runtime_graph"]["graph"]
        library = graph["nodes"][0]
        library["kind"] = "library"
        child = copy.deepcopy(library)
        child.update(id="child", kind="kernel", parent_library_id=library["id"])
        graph["nodes"].append(child)
        graph["edges"][0]["source"] = "child"
    result = attribute_best_components(samples, [0, 1])
    assert len(result["units"]) == 1
    assert result["credits"] == [1, 0]


def test_bound_pointer_renaming_and_process_addresses_are_not_scalar_changes():
    samples = trajectory([["a"], ["a"]], [0, 1])
    for turn, sample in enumerate(samples):
        graph = sample.metadata["runtime_graph"]["graph"]
        node = graph["nodes"][0]
        node["configuration"]["arguments"] = [
            {"kind": "address_binding", "buffer": "input", "offset": 0},
            {"kind": "argument_bits", "bytes": 8, "bits": 42},
        ]
        for buffer in graph["buffers"]:
            buffer["base"] = 100000 * turn + len(buffer["id"])
    result = attribute_best_components(samples, [0, 1])
    assert result["credits"] == [1, 0]
    # Eight-byte values without binding evidence stay real observed parameters.
    samples[1].metadata["runtime_graph"]["graph"]["nodes"][0]["configuration"]["arguments"][1]["bits"] = 43
    result = attribute_best_components(samples, [0, 1])
    assert result["credits"] == [0, 1]


def test_same_bytes_with_different_tensor_layout_do_not_match():
    samples = trajectory([["a"], ["a"]], [0, 1])
    for turn, sample in enumerate(samples):
        sample.metadata["runtime_graph"]["graph"]["buffers"][0]["views"] = [
            {"shape": [2, 2], "stride": [1, 2] if turn else [2, 1], "dtype": "torch.float16"}
        ]
    result = attribute_best_components(samples, [0, 1])
    assert result["credits"] == [0, 1]


def test_partial_data_edge_retains_unknown_producer_mass_without_erasing_known_path():
    samples = trajectory([["a", "b"], ["a", "b"]], [0, 1])
    samples[1].metadata["runtime_graph"]["graph"]["edges"][1]["certainty"] = "possible_dependency"
    result = attribute_best_components(samples, [0, 1])
    assert result["credits"] == [0.5, 0.5]
    assert result["residual_fraction"] == 0.5
    assert result["status"] == "partial"


def test_metrics_deduplicate_allocations_and_count_costs_per_turn():
    a = trajectory([["x"], ["a"], ["a"]], [0.5, 0, 1], index=0)
    b = trajectory([None, None, None], [0, 1, 0], index=1)
    a[0].metadata["runtime_graph"]["cost"] = {"wall_seconds": 2.5}
    for group in (a, b):
        postprocess_turn_samples(args(finalize_mode="positive"), group, "max_turns")
    metrics = compute_component_reward_metrics(args(), a + b + [a[0]])
    assert metrics["component_reward/trajectories"] == 2
    assert metrics["component_reward/observed_turns"] == 6
    assert metrics["component_reward/status/attributed/count"] == 1
    assert metrics["component_reward/status/unavailable_best_turn_fallback/count"] == 1
    assert metrics["component_reward/residual_budget_fraction"] == 0.5
    assert metrics["component_reward/resolved_cross_turn_fraction"] == 0.5
    assert metrics["component_reward/soft_finalize_protected_turns"] == 1
    assert metrics["component_reward/runtime_graph/wall_seconds_sum"] == 2.5
    assert metrics["component_reward/best_turn/2/count"] == 1
    assert metrics["component_reward/best_turn/1/count"] == 1
    assert metrics["component_reward/best_unit_count/observations"] == 1
    assert metrics["component_reward/best_unit_count/mean"] == 1
    assert metrics["component_reward/best_unit_count/max"] == 1
    assert compute_component_reward_metrics(args(component_reward=False), a) == {}


def test_best_unit_metrics_measure_counts_without_counting_allocation_copies():
    a = trajectory([["a"]], [1], index=0)
    b = trajectory([["a", "b", "c"]], [1], index=1)
    for group in (a, b):
        postprocess_turn_samples(args(), group, "max_turns")
    metrics = compute_component_reward_metrics(args(), a + b + a)
    assert metrics["component_reward/best_turn/0/count"] == 2
    assert metrics["component_reward/positive_best_turn/0/count"] == 2
    assert metrics["component_reward/best_unit_count/observations"] == 2
    assert metrics["component_reward/best_unit_count/mean"] == 2
    assert metrics["component_reward/best_unit_count/max"] == 3


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("mode", ["ppo", "dppo_topk_kl_predictive"])
@pytest.mark.parametrize("finalize", ["none", "positive"])
def test_component_targets_pack_into_same_nonzero_loss_and_gradient(cpu_mpu, cp_size, mode, finalize):
    torch.manual_seed(421)
    model = TinyCausalModel()
    samples = _samples(model, repair=True)
    config = _args(**vars(args(finalize_mode=finalize)), policy_loss_mode=mode)
    for index in range(2):
        group = samples[index * 3 : index * 3 + 3]
        programs = ["a", "x", "a"] if index == 0 else ["x", "a", "a"]
        for turn, sample in enumerate(group):
            sample.reward = [0.5, 0.0, 1.0][turn]
            sample.metadata["runtime_graph"] = observation([programs[turn]], candidate=f"{index}:{turn}")
        postprocess_turn_samples(config, group, "max_turns")
    raw, advantages = reward_post_process_by_group(config, samples)
    assert raw == [1, 0, 0, 0, 1, 0]
    # The unchanged normalizer excludes removed turns. In positive mode turn1
    # has one surviving contributor, so its original singleton-LOO target is0.
    assert advantages == ([1, 0, 0, -1, 0, 0] if finalize == "positive" else [1, -1, 0, -1, 1, 0])
    if finalize == "positive":
        assert samples[1].remove_sample
        assert not samples[4].remove_sample
        assert samples[4].metadata["component_reward_soft_finalize_protected"]
    split = _manager(config)._convert_samples_to_train_data(copy.deepcopy(samples))
    config.pack_multi_turn_trajectories = True
    packed = _manager(config)._convert_samples_to_train_data(copy.deepcopy(samples))
    assert len(packed["tokens"]) == 2
    assert {
        r
        for rewards, mask in zip(packed["token_rewards"], packed["loss_masks"], strict=True)
        for r, m in zip(rewards, mask, strict=True)
        if m
    } == {-1, 0, 1}
    with torch.no_grad():
        model.head.weight.add_(torch.randn_like(model.head.weight) * 0.2)
    old = _loss_and_grad(model, split, config, cpu_mpu, cp_size)
    new = _loss_and_grad(model, packed, config, cpu_mpu, cp_size)
    assert old[0] == pytest.approx(new[0], abs=2e-7, rel=3e-6)
    assert old[1].norm() > 1e-5
    torch.testing.assert_close(old[1], new[1], atol=3e-7, rtol=3e-5)
    assert old[2] == pytest.approx(new[2], abs=3e-5, rel=2e-5)
    assert old[3] == pytest.approx(new[3], abs=2e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
