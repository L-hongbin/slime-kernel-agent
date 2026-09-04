from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

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

from examples.kernel_agent import fully_async_rollout, generate_with_cuda_agent
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


def test_kernel_agent_group_concurrency_matches_client_capacity():
    args = _make_rollout_args()

    client_concurrency = http_utils.get_sglang_client_concurrency(args)
    group_concurrency = fully_async_rollout._get_group_concurrency(args, client_concurrency)

    assert client_concurrency == 64
    assert group_concurrency == 4


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
    monkeypatch.setattr(fully_async_rollout, "_get_global_worker", lambda args, data_buffer: worker)

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


def test_kernel_agent_worker_restamps_a_fresh_retry_after_abort():
    args = _make_rollout_args(gen_weight_version=7)
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

    # The aborted input is regenerated from its prompt, rather than resumed.
    # Its next attempt must describe the new generation policy, not retain v7.
    args.gen_weight_version = 8
    worker.set_generation_context(rollout_id=5)
    worker._stamp_group_for_submission([sample])
    assert sample.metadata["rollout_step"] == 5
    assert sample.metadata["gen_weight_version"] == 8


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
    padded = generate_with_cuda_agent._pad_turn_samples(
        [],
        base_sample,
        max_turns=1,
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

    assert padded[0].rollout_top_p_token_ids == [42]
    assert padded[0].rollout_top_p_token_offsets == [0, 1]
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
