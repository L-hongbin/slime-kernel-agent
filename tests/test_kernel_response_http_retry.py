"""HTTP transport retry coverage for the KernelGym client."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_AGENT_ROOT = REPO_ROOT / "examples" / "kernel_agent"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(KERNEL_AGENT_ROOT))

import kernel_response

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self) -> None:
        self.calls = 0

    def remote(self):
        self.calls += 1
        return object()


class _FakeLimiter:
    def __init__(self) -> None:
        self.acquire = _RemoteMethod()
        self.release = _RemoteMethod()


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "request failed",
                request=httpx.Request("POST", "http://kernelgym/evaluate"),
                response=httpx.Response(self.status_code),
            )


class _DisconnectOnceClient:
    def __init__(self) -> None:
        self.post_task_ids: list[str] = []

    def post(self, _url: str, *, json: dict, timeout: httpx.Timeout) -> _FakeResponse:
        del timeout
        self.post_task_ids.append(json["task_id"])
        if len(self.post_task_ids) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")
        return _FakeResponse(200)

    def get(self, url: str) -> _FakeResponse:
        if "/status/" in url:
            return _FakeResponse(200, {"status": "completed"})
        if "/results/" in url:
            return _FakeResponse(200, {"status": "completed", "compiled": True})
        raise AssertionError(f"unexpected URL: {url}")


def test_submit_retries_remote_protocol_disconnect_with_same_task_id(monkeypatch):
    # Ray wraps the implementation class at import time; use its original base
    # directly so this CPU-only unit test does not start a Ray cluster.
    worker_impl = kernel_response._HybridHttpWorker.__ray_metadata__.modified_class.__mro__[1]
    worker = object.__new__(worker_impl)
    worker.server_url = "http://kernelgym"
    worker.default_timeout = 300
    worker.acquire_timeout = 5
    worker._client = _DisconnectOnceClient()
    worker._rate_limit_worker = _FakeLimiter()
    worker._task_status = {}
    monkeypatch.setattr(kernel_response.ray, "wait", lambda refs, timeout: (refs, []))
    monkeypatch.setattr(kernel_response.time, "sleep", lambda _seconds: None)

    task = {"task_id": "stable-task-id"}
    result = worker.submit_and_poll(task, client_timeout=30, max_retries=3, poll_interval=0.01)

    assert result["status"] == "completed"
    assert result["compiled"] is True
    assert worker._client.post_task_ids == ["stable-task-id", "stable-task-id"]
    assert worker._rate_limit_worker.acquire.calls == 2
    assert worker._rate_limit_worker.release.calls == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
