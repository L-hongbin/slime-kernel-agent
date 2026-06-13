from __future__ import annotations

import asyncio
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from examples.kernel_agent import fully_async_rollout
from slime.utils import http_utils
from slime.utils.types import Sample


pytestmark = pytest.mark.unit


def _make_rollout_args(**overrides):
    values = dict(
        rollout_num_engines=None,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=4,
        sglang_server_concurrency=512,
        sglang_max_running_requests=32,
        n_samples_per_prompt=16,
        use_distributed_post=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_kernel_agent_group_concurrency_matches_client_capacity():
    args = _make_rollout_args()

    client_concurrency = http_utils.get_sglang_client_concurrency(args)
    group_concurrency = fully_async_rollout._get_group_concurrency(args, client_concurrency)

    assert client_concurrency == 64
    assert group_concurrency == 4


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
    script = Path("examples/kernel_agent/run.t1.qwen3.6.27B.full-async.sh").read_text()

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
    source = Path("examples/kernel_agent/generate_with_cuda_agent.py").read_text()

    assert (
        'KERNEL_AGENT_GENERATE_MAX_RETRIES = max(1, int(os.environ.get("KERNEL_AGENT_GENERATE_MAX_RETRIES", "60") or 60))'
        in source
    )
    assert "post(url, payload, max_retries=KERNEL_AGENT_GENERATE_MAX_RETRIES)" in source
