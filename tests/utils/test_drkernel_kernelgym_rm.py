import asyncio
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import slime_plugins.drkernel.kernelgym_rm as krm
from slime_plugins.drkernel.extract import MISSING_COMPLETE_SUBMISSION_ERROR
from slime_plugins.drkernel.kernelgym_rm import (
    KERNELGYM_CLIENT_TIMEOUT_S,
    KERNELGYM_DETECT_DECOY_KERNEL,
    KERNELGYM_ENABLE_PROFILING,
    KERNELGYM_NUM_CORRECT_TRIALS,
    KERNELGYM_NUM_PERF_TRIALS,
    KERNELGYM_NUM_WARMUP,
    KERNELGYM_PERF_TRIM_COUNT,
    KERNELGYM_REFERENCE_BACKEND,
    KERNELGYM_TASK_TIMEOUT_S,
    KERNELGYM_USE_REFERENCE_CACHE,
    KERNELGYM_VERBOSE_ERRORS,
    KERNELGYM_WORKFLOW,
    KernelGymClient,
    KernelGymRequestError,
    _get_rm_url,
    _get_shared_client,
    build_evaluation_request,
    custom_rm,
    evaluate_sample,
    kernelgym_result_to_reward,
)


class _Sample:
    def __init__(self):
        self.index = 7
        self.response = ""
        self.label = "import torch\nclass Model:\n    pass"
        self.metadata = {
            "raw_problem": "import torch\nclass Model:\n    pass",
            "uuid": "sample-uuid",
            "entry_point": "Model",
        }


@pytest.mark.unit
def test_build_evaluation_request_uses_sample_label_as_reference_code():
    request = build_evaluation_request(
        SimpleNamespace(),
        _Sample(),
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert re.fullmatch(r"parallel_task_\d{6}_[0-9a-f]{8}", request["task_id"])
    assert request["reference_code"] == "import torch\nclass Model:\n    pass"
    assert request["backend"] == "auto"
    assert request["entry_point"] == "Model"
    assert request["workflow"] == KERNELGYM_WORKFLOW
    assert request["use_reference_cache"] is KERNELGYM_USE_REFERENCE_CACHE
    assert request["timeout"] == KERNELGYM_TASK_TIMEOUT_S
    assert request["verbose_errors"] is KERNELGYM_VERBOSE_ERRORS
    assert request["enable_profiling"] is KERNELGYM_ENABLE_PROFILING
    assert request["detect_decoy_kernel"] is KERNELGYM_DETECT_DECOY_KERNEL
    assert request["num_correct_trials"] == KERNELGYM_NUM_CORRECT_TRIALS
    assert request["num_perf_trials"] == KERNELGYM_NUM_PERF_TRIALS
    assert request["num_warmup"] == KERNELGYM_NUM_WARMUP
    assert request["perf_trim_count"] == KERNELGYM_PERF_TRIM_COUNT
    assert request["reference_backend"] == KERNELGYM_REFERENCE_BACKEND
    assert request["is_valid"] is False
    assert request["uuid"] == "sample-uuid"


@pytest.mark.unit
def test_build_evaluation_request_does_not_fallback_to_metadata_reference_code():
    sample = _Sample()
    sample.label = None

    with pytest.raises(KeyError, match="--label-key ground_truth"):
        build_evaluation_request(
            SimpleNamespace(),
            sample,
            kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
        )


@pytest.mark.unit
def test_build_evaluation_request_requires_string_label():
    sample = _Sample()
    sample.label = {"code": "class Model: pass"}

    with pytest.raises(TypeError, match="reference code must be a string"):
        build_evaluation_request(
            SimpleNamespace(),
            sample,
            kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
        )


@pytest.mark.unit
def test_build_evaluation_request_requires_uuid():
    sample = _Sample()
    del sample.metadata["uuid"]

    with pytest.raises(KeyError, match="missing uuid"):
        build_evaluation_request(
            SimpleNamespace(),
            sample,
            kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
        )


@pytest.mark.unit
def test_build_evaluation_request_uses_kernelbench_eval_metadata_defaults():
    sample = _Sample()
    del sample.metadata["uuid"]
    del sample.metadata["entry_point"]
    sample.metadata["problem_id"] = 1

    request = build_evaluation_request(
        SimpleNamespace(),
        sample,
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["entry_point"] == "Model"
    assert request["uuid"] == "1"


@pytest.mark.unit
def test_build_evaluation_request_ignores_metadata_task_id_override():
    sample = _Sample()
    sample.metadata["task_id"] = "custom-task-id"

    request = build_evaluation_request(
        SimpleNamespace(),
        sample,
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["task_id"] != "custom-task-id"
    assert re.fullmatch(r"parallel_task_\d{6}_[0-9a-f]{8}", request["task_id"])


@pytest.mark.unit
def test_build_evaluation_request_ignores_metadata_overrides_and_unknown_args():
    sample = _Sample()
    for key in (
        "toolkit",
        "backend_adapter",
        "workflow",
        "use_reference_cache",
        "timeout",
        "verbose_errors",
        "enable_profiling",
        "detect_decoy_kernel",
        "num_correct_trials",
        "num_perf_trials",
        "num_warmup",
        "perf_trim_count",
        "priority",
        "device_preference",
        "force_refresh",
        "reference_backend",
        "measure_performance",
        "run_correctness",
        "run_performance",
        "resources",
    ):
        sample.metadata[key] = "metadata-override"

    args = SimpleNamespace(
        kernelgym_toolkit="arg-override",
        kernelgym_backend_adapter="arg-override",
        kernelgym_workflow="arg-override",
        kernelgym_use_reference_cache=False,
        kernelgym_timeout="arg-override",
        kernelgym_verbose_errors=False,
        kernelgym_enable_profiling=False,
        kernelgym_detect_decoy_kernel=False,
        kernelgym_priority="arg-override",
        kernelgym_device_preference="arg-override",
        kernelgym_force_refresh="arg-override",
        kernelgym_measure_performance="arg-override",
        kernelgym_run_correctness="arg-override",
        kernelgym_run_performance="arg-override",
        kernelgym_resources="arg-override",
    )

    request = build_evaluation_request(
        args,
        sample,
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["workflow"] == KERNELGYM_WORKFLOW
    assert request["use_reference_cache"] is KERNELGYM_USE_REFERENCE_CACHE
    assert request["timeout"] == KERNELGYM_TASK_TIMEOUT_S
    assert request["verbose_errors"] is KERNELGYM_VERBOSE_ERRORS
    assert request["enable_profiling"] is KERNELGYM_ENABLE_PROFILING
    assert request["detect_decoy_kernel"] is KERNELGYM_DETECT_DECOY_KERNEL
    assert request["num_correct_trials"] == KERNELGYM_NUM_CORRECT_TRIALS
    assert request["num_perf_trials"] == KERNELGYM_NUM_PERF_TRIALS
    assert request["num_warmup"] == KERNELGYM_NUM_WARMUP
    assert request["perf_trim_count"] == KERNELGYM_PERF_TRIM_COUNT
    assert request["reference_backend"] == KERNELGYM_REFERENCE_BACKEND
    for key in (
        "toolkit",
        "backend_adapter",
        "priority",
        "device_preference",
        "force_refresh",
        "measure_performance",
        "run_correctness",
        "run_performance",
        "resources",
    ):
        assert key not in request


@pytest.mark.unit
def test_build_evaluation_request_accepts_explicit_tuning_args():
    request = build_evaluation_request(
        SimpleNamespace(
            kernelgym_num_correct_trials="7",
            kernelgym_num_perf_trials="80",
            kernelgym_num_warmup="11",
            kernelgym_perf_trim_count="2",
            kernelgym_reference_backend="pytorch",
        ),
        _Sample(),
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["num_correct_trials"] == 7
    assert request["num_perf_trials"] == 80
    assert request["num_warmup"] == 11
    assert request["perf_trim_count"] == 2
    assert request["reference_backend"] == "pytorch"


@pytest.mark.unit
def test_build_evaluation_request_keeps_is_valid_configurable():
    sample = _Sample()
    sample.metadata["is_valid"] = True

    request = build_evaluation_request(
        SimpleNamespace(),
        sample,
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["is_valid"] is True


@pytest.mark.unit
def test_rm_url_is_required():
    assert _get_rm_url(SimpleNamespace(rm_url="http://kernelgym")) == "http://kernelgym"
    with pytest.raises(ValueError):
        _get_rm_url(SimpleNamespace(rm_url=None))


@pytest.mark.unit
def test_kernelgym_client_default_timeout_is_hardcoded():
    assert KernelGymClient("http://kernelgym").timeout_s == KERNELGYM_CLIENT_TIMEOUT_S
    # Pinned to 10min: a wedged/saturated server must fail fast + retry, not pin a
    # request for 30min (see gbs=128 run where tail /evaluate hung the full 1800s).
    assert KERNELGYM_CLIENT_TIMEOUT_S == 600.0


@pytest.mark.unit
def test_kernelgym_client_check_health_rejects_unhealthy_status():
    async def run() -> None:
        async def fake_request_json(self, method, path, **kwargs):
            assert method == "GET"
            assert path == "/health"
            return {"status": "degraded", "gpu_status": {}}

        with patch.object(KernelGymClient, "_request_json", fake_request_json):
            client = KernelGymClient("http://kernelgym")
            with pytest.raises(KernelGymRequestError, match="is not healthy"):
                await client._check_health()

    asyncio.run(run())


@pytest.mark.unit
def test_kernelgym_client_create_runs_health_check():
    async def run() -> None:
        async def fake_request_json(self, method, path, **kwargs):
            assert method == "GET"
            assert path == "/health"
            return {"status": "healthy", "gpu_status": {"cuda:0": {"available": True}}}

        with patch.object(KernelGymClient, "_request_json", fake_request_json):
            client = await KernelGymClient.create("http://kernelgym")
            assert client.base_url == "http://kernelgym"

    asyncio.run(run())


@pytest.mark.unit
def test_kernelgym_result_to_reward_requires_compile_correct_and_not_decoy():
    assert kernelgym_result_to_reward({"compiled": True, "correctness": True, "decoy_kernel": False}) == 1.0
    assert kernelgym_result_to_reward({"compiled": False, "correctness": True, "decoy_kernel": False}) == 0.0
    assert kernelgym_result_to_reward({"compiled": True, "correctness": False, "decoy_kernel": False}) == 0.0
    assert kernelgym_result_to_reward({"compiled": True, "correctness": True, "decoy_kernel": True}) == 0.0


class _CompleteSubmission:
    """Stand-in for a fully extracted kernel submission (kernels + model)."""

    code = "kernel code"
    backend = "cuda"
    sections = {"CUDA_KERNELS", "MODEL"}


@pytest.fixture
def reset_shared_client():
    """The shared client is process-global; isolate each test from the others.

    Also swap in a fresh ``asyncio.Lock`` so a lock bound to one test's
    ``asyncio.run`` loop can't leak loop state into the next test.
    """

    krm._shared_client = None
    krm._shared_client_loop = None
    krm._shared_client_lock = asyncio.Lock()
    yield
    krm._shared_client = None
    krm._shared_client_loop = None
    krm._shared_client_lock = asyncio.Lock()


@pytest.mark.unit
def test_shared_client_health_checked_once_across_many_rm_calls(reset_shared_client):
    """30 RM calls (3 turns x 10 samples) must probe /health exactly once.

    This is the whole point of the process-wide shared client: before it, every
    ``custom_rm`` call created+closed a fresh client and re-ran ``/health``, so a
    single training step fired hundreds of redundant health probes.
    """

    health = {"n": 0}
    evals = {"n": 0}

    async def fake_check_health(self):
        health["n"] += 1

    async def fake_evaluate(self, request):
        evals["n"] += 1
        return {"compiled": True, "correctness": True, "decoy_kernel": False}

    async def run():
        args = SimpleNamespace(rm_url="http://kernelgym")
        for _turn in range(3):
            for _ in range(10):
                await custom_rm(args, _Sample())

    with (
        patch.object(KernelGymClient, "_check_health", fake_check_health),
        patch.object(KernelGymClient, "evaluate", fake_evaluate),
        patch.object(krm, "extract_kernel_submission", lambda response: _CompleteSubmission()),
    ):
        asyncio.run(run())

    assert health["n"] == 1
    assert evals["n"] == 30
    assert krm._shared_client is not None


@pytest.mark.unit
def test_shared_client_built_once_under_concurrent_first_callers(reset_shared_client):
    """Concurrent first callers (as in generate_and_rm_group) build one client.

    The lazy init is guarded by an asyncio.Lock with a double-check; without it,
    a burst of concurrent ``custom_rm`` calls would each build a client and probe
    ``/health`` before any of them populated the cache.
    """

    health = {"n": 0}

    async def fake_check_health(self):
        await asyncio.sleep(0.01)  # widen the race window so the lock is exercised
        health["n"] += 1

    async def fake_evaluate(self, request):
        return {"compiled": True, "correctness": True, "decoy_kernel": False}

    async def run():
        args = SimpleNamespace(rm_url="http://kernelgym")
        await asyncio.gather(*(custom_rm(args, _Sample()) for _ in range(50)))

    with (
        patch.object(KernelGymClient, "_check_health", fake_check_health),
        patch.object(KernelGymClient, "evaluate", fake_evaluate),
        patch.object(krm, "extract_kernel_submission", lambda response: _CompleteSubmission()),
    ):
        asyncio.run(run())

    assert health["n"] == 1


@pytest.mark.unit
def test_shared_client_not_cached_when_health_check_fails(reset_shared_client):
    """A failed first health check must fail-fast and NOT cache a broken client.

    ``_get_shared_client`` only stores the client after ``_check_health`` returns,
    so the next RM call retries the probe instead of reusing a dead client.
    """

    calls = {"n": 0}

    async def flaky_check_health(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KernelGymRequestError("health down")

    async def fake_evaluate(self, request):
        return {"compiled": True, "correctness": True, "decoy_kernel": False}

    async def run():
        args = SimpleNamespace(rm_url="http://kernelgym")
        with pytest.raises(KernelGymRequestError, match="health down"):
            await custom_rm(args, _Sample())
        assert krm._shared_client is None  # broken client not cached

        await custom_rm(args, _Sample())  # retries the probe, now succeeds
        assert krm._shared_client is not None

    with (
        patch.object(KernelGymClient, "_check_health", flaky_check_health),
        patch.object(KernelGymClient, "evaluate", fake_evaluate),
        patch.object(krm, "extract_kernel_submission", lambda response: _CompleteSubmission()),
    ):
        asyncio.run(run())

    assert calls["n"] == 2


@pytest.mark.unit
def test_shared_client_closes_session_when_health_check_fails(reset_shared_client):
    """A failed first health probe must close the aiohttp session it opened.

    ``_check_health`` lazily creates the session before issuing the request; if the
    probe then fails, ``_get_shared_client`` must close that session so a flapping
    server doesn't leak one connector per (caught) failure.
    """

    closed = {"n": 0}
    real_close = KernelGymClient.close

    async def counting_close(self):
        closed["n"] += 1
        await real_close(self)

    async def failing_check_health(self):
        # Exercise the real session creation path, then fail like a 5xx /health.
        self._get_session()
        raise KernelGymRequestError("health down")

    async def run():
        args = SimpleNamespace(rm_url="http://kernelgym")
        with pytest.raises(KernelGymRequestError, match="health down"):
            await _get_shared_client(args)

    with (
        patch.object(KernelGymClient, "_check_health", failing_check_health),
        patch.object(KernelGymClient, "close", counting_close),
    ):
        asyncio.run(run())

    assert closed["n"] == 1
    assert krm._shared_client is None


@pytest.mark.unit
def test_evaluate_sample_returns_zero_reward_for_incomplete_submission():
    sample = _Sample()
    sample.response = "### CUDA_KERNELS\n```cpp\nonly kernels\n```"

    output = asyncio.run(evaluate_sample(SimpleNamespace(), sample))

    assert output == {"reward": 0.0, "extract_error": MISSING_COMPLETE_SUBMISSION_ERROR}
    assert sample.metadata["kernelgym"] == {
        "extract_error": MISSING_COMPLETE_SUBMISSION_ERROR,
        "reward": 0.0,
    }
    assert "kernel_submission" not in sample.metadata
