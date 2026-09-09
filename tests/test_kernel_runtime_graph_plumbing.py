"""Keep runtime attribution evidence outside the model's evaluation feedback."""

import asyncio
import hashlib
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.kernel_agent import generate_with_cuda_agent as agent
from examples.kernel_agent import kernel_response
from examples.kernel_agent.kernel_response import _build_kernel_eval_payload

from slime.utils.types import Sample

CONFIG = {"num_correct_trials": 5, "num_perf_trials": 100, "kernel_eval_task_timeout": 300}


def payload():
    return {"kernel_code": "candidate", "ground_truth": "reference", "entry_point": "Model", "backend": "tvm_ffi"}


def test_default_payload_has_no_runtime_graph():
    assert "runtime_graph" not in _build_kernel_eval_payload(SimpleNamespace(), payload(), CONFIG)


def test_component_reward_enables_only_separate_diagnostic_budget():
    args = SimpleNamespace(component_reward=True, runtime_graph_timeout=45.0)
    result = _build_kernel_eval_payload(args, payload(), CONFIG)
    assert result["runtime_graph"] == {"enabled": True, "timeout_s": 45.0}
    assert result["timeout"] == 300
    assert result["num_correct_trials"] == 5
    assert result["num_perf_trials"] == 100


def test_explicit_runtime_graph_request_is_preserved():
    result = _build_kernel_eval_payload(SimpleNamespace(), {**payload(), "runtime_graph": {"enabled": False}}, CONFIG)
    assert result["runtime_graph"] == {"enabled": False}


def test_requested_source_identity_is_bound_before_receiving_graph(monkeypatch):
    submitted = []

    def submit(task_payload, **_kwargs):
        submitted.append(task_payload)
        return object()

    async def wait(*_args, **_kwargs):
        return {"status": "completed"}

    worker = SimpleNamespace(submit_and_poll=SimpleNamespace(remote=submit))
    monkeypatch.setattr(kernel_response, "_get_kernel_eval_worker", lambda *_: worker)
    monkeypatch.setattr(kernel_response, "_wait_kernel_eval_result", wait)
    sample = Sample(metadata={"runtime_graph_expected_identity": {"task_sha256": "stale"}})
    config = {
        **CONFIG,
        "kernel_eval_client_timeout": 2400,
        "kernel_eval_max_retries": 3,
        "kernel_eval_poll_interval": 1,
        "kernel_eval_rate_limit": 10,
    }
    asyncio.run(kernel_response.run_kernel_eval(SimpleNamespace(component_reward=True), sample, payload(), config))
    assert sample.metadata["runtime_graph_expected_identity"] == {
        "task_sha256": hashlib.sha256(b"reference").hexdigest(),
        "candidate_source_sha256": hashlib.sha256(b"candidate").hexdigest(),
    }
    assert len(submitted) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_custom_evaluator_receives_equivalent_opt_in_contract(monkeypatch, enabled):
    original = payload()
    original["custom_option"] = "preserved"
    sample = Sample(metadata={})

    async def custom(_args, actual_sample, actual_payload):
        assert actual_sample is sample
        assert actual_payload["custom_option"] == "preserved"
        if enabled:
            assert actual_payload["runtime_graph"] == {"enabled": True, "timeout_s": 60.0}
            assert actual_payload["reference_code"] == "reference"
            assert actual_payload["kernel_code"] == "candidate"
            assert (
                sample.metadata["runtime_graph_expected_identity"]["candidate_source_sha256"]
                == hashlib.sha256(b"candidate").hexdigest()
            )
        else:
            assert actual_payload is original
            assert sample.metadata == {}
        return {"compiled": True, "correctness": True}

    monkeypatch.setattr(kernel_response, "load_function", lambda _: custom)
    args = SimpleNamespace(component_reward=enabled, kernel_eval_function_path="custom.evaluator")
    result = asyncio.run(kernel_response.run_kernel_eval(args, sample, original, CONFIG))
    assert result == {"env_state": {"compiled": True, "correctness": True}}
    assert "runtime_graph" not in original


def run_env(monkeypatch, raw, *, enabled=True):
    async def evaluate(*_args):
        return {"env_state": {"compiled": True, "speedup": 0.0, "metadata": {}, **raw}}

    monkeypatch.setattr(agent, "run_kernel_eval", evaluate)
    args = SimpleNamespace(do_precheck=False, kernel_backend="tvm_ffi", component_reward=enabled)
    sample = Sample(label={"ground_truth": "reference"}, metadata={})
    return asyncio.run(agent.cuda_kernel_env(args, sample, "response", 0))


def test_graph_does_not_change_feedback_or_scalar_reward_inputs(monkeypatch):
    baseline = {
        "status": "completed",
        "compiled": True,
        "correctness": True,
        "speedup": 1.2,
        "metadata": {"some_existing_fact": "preserved"},
    }
    graph = {"schema": "test", "status": "complete", "components": [{"id": "never_in_feedback"}]}
    raw = {**baseline, "metadata": {**baseline["metadata"], "runtime_graph": graph}}
    original = deepcopy(raw)
    actual = run_env(monkeypatch, raw)
    expected = run_env(monkeypatch, baseline, enabled=False)
    assert actual["runtime_graph"] == graph
    assert {key: value for key, value in actual.items() if key != "runtime_graph"} == expected
    assert agent.build_model_feedback(actual) == agent.build_model_feedback(expected)
    assert raw == original


@pytest.mark.parametrize(
    "graph,reason", [(None, "response_runtime_graph_not_object"), ([], "response_runtime_graph_not_object")]
)
def test_invalid_graph_is_explicit_without_overriding_correctness(monkeypatch, graph, reason):
    actual = run_env(monkeypatch, {"correctness": True, "metadata": {"runtime_graph": graph}})
    assert actual["runtime_graph"] == {
        "schema": "kernelgym-runtime-graph/v1",
        "status": "invalid",
        "unknowns": [reason],
    }
    assert actual["env_state"]["correctness"] is True


def test_missing_server_support_is_explicit(monkeypatch):
    actual = run_env(monkeypatch, {"correctness": True})
    assert actual["runtime_graph"] == {
        "schema": "kernelgym-runtime-graph/v1",
        "status": "unavailable",
        "unknowns": ["missing_server_graph"],
    }


def test_client_precheck_failure_has_no_fake_graph(monkeypatch):
    # Exercise the real precheck's complete failure schema.
    args = SimpleNamespace(do_precheck=True, kernel_backend="tvm_ffi", component_reward=True)
    sample = Sample(label={"ground_truth": "reference"}, metadata={})
    result = asyncio.run(agent.cuda_kernel_env(args, sample, "response", 0))
    assert result["runtime_graph"] == {
        "schema": "kernelgym-runtime-graph/v1",
        "status": "unavailable",
        "unknowns": ["client_precheck"],
    }


def test_turn_sample_uses_current_graph_not_inherited_graph():
    base = Sample(index=0, metadata={"runtime_graph": {"id": "stale"}})
    kwargs = dict(
        prompt_ids=[1],
        response="x",
        response_ids=[2],
        log_probs=[-0.2],
        reward=1.0,
        status=Sample.Status.COMPLETED,
        turn_idx=0,
    )
    graph = {"status": "complete", "id": "current"}
    with_graph = agent._sample_for_turn(base, **kwargs, env_result={"runtime_graph": graph, "env_extra_info": {}})
    without_graph = agent._sample_for_turn(base, **kwargs, env_result={"env_extra_info": {}})
    assert with_graph.metadata["runtime_graph"] == graph
    assert "runtime_graph" not in without_graph.metadata
    assert base.metadata["runtime_graph"]["id"] == "stale"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
