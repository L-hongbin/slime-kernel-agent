"""CPU contracts for source-based credit, request identity and TRLOO finalization."""

from __future__ import annotations

import asyncio
import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.kernel_agent import generate_with_cuda_agent as agent
from examples.kernel_agent import kernel_response
from examples.kernel_agent.component_reward import ComponentRewardContractError, compute_component_reward_metrics
from examples.kernel_agent.source_component_reward import attribute_best_source_components, source_request_identity
from examples.kernel_agent.utils import extract_cuda_agent_kernel_code, postprocess_turn_samples
from test_kernel_agent_component_reward import args as graph_args

from slime.utils.types import Sample

REFERENCE = "class Model:\n    pass\n"
EVAL_CONFIG = {"num_correct_trials": 5, "num_perf_trials": 100, "kernel_eval_task_timeout": 300}


def response(a=2, b=3, *, bad_export=False, dead=False):
    cuda = (
        f"__global__ void a(float*x){{int i=threadIdx.x;x[i]*={a};}}\n"
        f"__global__ void b(float*x){{int i=threadIdx.x;x[i]*={b};}}\n"
        'extern "C" void launch(float*x){a<<<1,32>>>(x);b<<<1,32>>>(x);}\n'
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,launch);"
    )
    if dead:
        cuda += "\n__global__ void unused(float*x){x[0]=1;}"
    model = "import tvm_ffi_extension\nclass ModelNew:\n def forward(self,x): return tvm_ffi_extension."
    model += "typo(x)" if bad_export else "run(x)"
    return "\n\n".join(
        [
            f"### CUDA_KERNELS\n```cpp\n{cuda}\n```",
            "### APPLY_BINDINGS\n```cpp\nvoid binding(){}\n```",
            f"### MODEL_NEW\n```python\n{model}\n```",
        ]
    )


def sample(turn, code, reward, correct, *, removed=False):
    return Sample(
        index=0,
        group_index=0,
        group_id="g",
        tokens=list(range(16)),
        response=code,
        response_length=8,
        rollout_log_probs=[-0.1] * 8,
        loss_mask=[1] * 8,
        reward=reward,
        status=Sample.Status.COMPLETED,
        remove_sample=removed,
        label={"ground_truth": REFERENCE},
        metadata={
            "turn_idx": turn,
            "env_extra_info": {"correctness": correct, "compiled": True, "decoy_kernel": False},
            "source_component_identity": source_request_identity(
                REFERENCE,
                extract_cuda_agent_kernel_code(code),
                entry_point="Model",
                precision="fp32",
                submitted="typo" not in code,
            ),
        },
    )


def test_failed_proposal_receives_best_component_credit():
    samples = [sample(0, response(b=99, bad_export=True), 0, False), sample(1, response(), 1.2, True)]
    allocation = attribute_best_source_components(samples, [0, 1.2])
    assert allocation["credits"] == pytest.approx([0.6, 0.6])
    assert len(allocation["units"]) == 2
    assert allocation["resolved_cross_turn_units"] == 1
    assert math.fsum(allocation["credits"]) == pytest.approx(1.2)
    assert allocation["future_fold_applied"] is False


def test_reappearance_finds_earliest_source_not_latest_copy():
    samples = [
        sample(0, response(b=99), 0, False),
        sample(1, response(a=8, b=99), 0, False),
        sample(2, response(), 1.2, True),
    ]
    result = attribute_best_source_components(samples, [0, 0, 1.2])
    assert result["credits"] == pytest.approx([0.6, 0, 0.6])
    assert any(u["status"] == "reappeared" and u["origin_turn"] == 0 for u in result["units"])


def test_unobserved_best_unit_keeps_residual_in_denominator():
    samples = [sample(0, response(dead=True), 0, False), sample(1, response(dead=True), 1.2, True)]
    result = attribute_best_source_components(samples, [0, 1.2])
    assert len(result["units"]) == 3
    assert result["credits"] == pytest.approx([0.8, 0.4])
    assert result["residual_fraction"] == pytest.approx(1 / 3)


def test_all_wrong_partial_reward_has_no_component_attribution():
    samples = [sample(0, response(), 0.25, False), sample(1, response(), 0.25, False)]
    result = attribute_best_source_components(samples, [0.25, 0.25])
    assert result["best_turn"] == 0
    assert result["credits"] == [0.25, 0]
    assert result["unknowns"] == ["no_correct_anchor"]


def test_identity_mismatch_fails_closed():
    s = sample(0, response(), 1.2, True)
    s.response = response(a=88)
    with pytest.raises(ComponentRewardContractError, match="evaluated candidate"):
        attribute_best_source_components([s], [1.2])


def test_hard_removal_and_different_task_are_not_earlier_origins():
    first, last = sample(0, response(), 0, False, removed=True), sample(1, response(), 1.2, True)
    assert attribute_best_source_components([first, last], [0, 1.2])["credits"] == [0, 1.2]
    first.remove_sample = False
    first.metadata["source_component_identity"]["task_sha256"] = "another-task"
    result = attribute_best_source_components([first, last], [0, 1.2])
    assert result["credits"] == [0, 1.2]
    assert "incomparable_source_context:task_sha256" in result["unknowns"]


def test_producer_schema_drift_fails_closed(monkeypatch):
    from examples.kernel_agent import source_components

    monkeypatch.setattr(
        source_components, "analyze_source_components", lambda *a, **k: {"schema": "wrong", "units": []}
    )
    with pytest.raises(ComponentRewardContractError, match="observation"):
        attribute_best_source_components([sample(0, response(), 1.2, True)], [1.2])


def test_source_payload_omits_runtime_graph_and_keeps_eval_parameters():
    args = SimpleNamespace(component_reward=True, component_reward_backend="source", runtime_graph_timeout=45)
    payload = {"kernel_code": "candidate", "ground_truth": REFERENCE, "entry_point": "Model", "backend": "tvm_ffi"}
    result = kernel_response._build_kernel_eval_payload(args, payload, EVAL_CONFIG)
    assert "runtime_graph" not in result
    assert result["timeout"] == 300
    assert result["num_correct_trials"] == 5


def test_source_custom_evaluator_gets_identity_without_graph(monkeypatch):
    args = SimpleNamespace(
        component_reward=True, component_reward_backend="source", kernel_eval_function_path="custom"
    )
    s = Sample(metadata={})
    p = {"kernel_code": "candidate", "ground_truth": REFERENCE, "entry_point": "Model", "backend": "tvm_ffi"}

    async def custom(_args, actual_sample, actual_payload):
        assert actual_sample is s
        assert "runtime_graph" not in actual_payload
        assert actual_payload["kernel_code"] == "candidate"
        assert s.metadata["source_component_identity"]["submitted"]
        return {"correctness": True, "compiled": True}

    monkeypatch.setattr(kernel_response, "load_function", lambda _: custom)
    result = asyncio.run(kernel_response.run_kernel_eval(args, s, p, EVAL_CONFIG))
    assert result["env_state"]["correctness"]


def test_precheck_failure_has_current_source_identity(monkeypatch):
    args = SimpleNamespace(
        component_reward=True, component_reward_backend="source", kernel_backend="tvm_ffi", do_precheck=True
    )
    s = Sample(label={"ground_truth": REFERENCE}, metadata={})
    monkeypatch.setattr(
        agent,
        "precheck_response",
        lambda *a: (
            False,
            {"compiled": False, "correctness": False, "speedup": 0.0, "metadata": {}, "error_message": "precheck"},
        ),
    )
    result = asyncio.run(agent.cuda_kernel_env(args, s, response(), 0))
    assert "runtime_graph" not in result
    assert not result["source_component_identity"]["submitted"]
    assert (
        result["source_component_identity"]["candidate_source_sha256"]
        == source_request_identity(
            REFERENCE, extract_cuda_agent_kernel_code(response()), entry_point="Model", precision="fp32"
        )["candidate_source_sha256"]
    )


def test_positive_source_credit_survives_soft_finalize_and_is_logged():
    args = graph_args(component_reward_backend="source", finalize_mode="positive")
    samples = [
        sample(0, response(a=5, b=99), 1.0, True),
        sample(1, response(b=99), 0, False),
        sample(2, response(), 1.2, True),
    ]
    finalized = postprocess_turn_samples(args, copy.deepcopy(samples), finish_reason="max_turns")
    assert finalized[1].reward == pytest.approx(0.6)
    assert not finalized[1].remove_sample
    assert finalized[1].metadata["component_reward_soft_finalize_protected"]
    assert sum(s.reward for s in finalized) == pytest.approx(1.2)
    assert [s.metadata["task_reward"] for s in finalized] == [1, 0, 1.2]
    metrics = compute_component_reward_metrics(args, finalized)
    assert metrics["component_reward/source/trajectories"] == 1
    assert metrics["component_reward/source/analysis_wall_seconds_sum"] >= 0
    assert not any("runtime_graph" in key for key in metrics)


@pytest.mark.parametrize("profiling", [None, {}, {"kernels": None}, {"kernels": []}])
def test_optional_failed_evaluation_profiling_does_not_break_source_transport(monkeypatch, profiling):
    args = SimpleNamespace(
        component_reward=True,
        component_reward_backend="source",
        kernel_backend="tvm_ffi",
        do_precheck=False,
        kernel_eval_function_path="custom",
    )
    s = Sample(label={"ground_truth": REFERENCE}, metadata={"source_component_profiles": [{"name": "stale"}]})

    async def custom(_args, actual_sample, actual_payload):
        assert "runtime_graph" not in actual_payload
        assert actual_payload["kernel_code"] == extract_cuda_agent_kernel_code(response())
        return {
            "compiled": False,
            "correctness": False,
            "speedup": 0.0,
            "metadata": {"profiling": profiling},
            "error_message": "compile failure",
        }

    monkeypatch.setattr(kernel_response, "load_function", lambda _: custom)
    result = asyncio.run(agent.cuda_kernel_env(args, s, response(), 0))
    assert result["source_component_identity"]["submitted"]
    assert not result["source_component_profiles"]
    assert "source_component_profiles" not in s.metadata
    assert "runtime_graph" not in result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
