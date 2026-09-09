"""Result delivery must not depend on heartbeat RPCs or executor availability."""

import asyncio
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "kernel_agent"))

import kernel_response
from kernel_response import _wait_kernel_eval_result

NUM_GPUS = 0
PAYLOAD = {"task_id": "test-task", "entry_point": "Model", "backend": "tvm_ffi"}
RESULT = {"status": "completed", "compiled": True, "correctness": True, "metadata": {"runtime": 1.25}}


class _ObjectRef:
    def __init__(self):
        self.value = Future()

    def future(self):
        return self.value


class _RemoteMethod:
    def __init__(self):
        self.ref = _ObjectRef()
        self.called = asyncio.Event()
        self.calls = 0

    def remote(self, *args):
        self.calls += 1
        self.called.set()
        return self.ref


def _worker():
    return SimpleNamespace(get_task_status=_RemoteMethod(), get_token_in_use=_RemoteMethod())


def _start_wait(ref, worker, interval=0.01):
    return asyncio.create_task(_wait_kernel_eval_result(ref, worker, PAYLOAD, interval, 32))


def _assert_no_background_tasks():
    assert asyncio.all_tasks() == {asyncio.current_task()}


@pytest.mark.parametrize("blocked_method", ["get_task_status", "get_token_in_use"])
def test_result_returns_while_heartbeat_rpc_is_blocked(blocked_method, capsys):
    async def run():
        ref, worker = _ObjectRef(), _worker()
        if blocked_method == "get_token_in_use":
            worker.get_task_status.ref.value.set_result({"status": "processing"})
        waiting = _start_wait(ref, worker, interval=1.0)
        await asyncio.wait_for(getattr(worker, blocked_method).called.wait(), 3)
        ref.value.set_result(RESULT)
        # The RPC remains unresolved; neither its timeout nor a reply is needed.
        assert await asyncio.wait_for(waiting, 0.5) == RESULT
        _assert_no_background_tasks()

    asyncio.run(run())
    assert "pending=1" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["timeout", "exception"])
def test_unavailable_heartbeat_does_not_fail_result_or_accumulate_queries(failure, capsys):
    async def run():
        ref, worker = _ObjectRef(), _worker()
        waiting = _start_wait(ref, worker)
        await asyncio.wait_for(worker.get_task_status.called.wait(), 1)
        if failure == "exception":
            worker.get_task_status.ref.value.set_exception(RuntimeError("status RPC unavailable"))
        await asyncio.sleep(0.08)  # Several heartbeat periods, including the RPC timeout.
        assert not waiting.done()
        assert not ref.value.done()
        assert worker.get_task_status.calls == 1
        assert worker.get_token_in_use.calls == 0
        ref.value.set_result(RESULT)
        assert await asyncio.wait_for(waiting, 0.5) == RESULT
        _assert_no_background_tasks()

    asyncio.run(run())
    assert "heartbeat stopped" in capsys.readouterr().out


def test_cancellation_cleans_up_blocked_heartbeat():
    async def run():
        ref, worker = _ObjectRef(), _worker()
        waiting = _start_wait(ref, worker)
        await asyncio.wait_for(worker.get_task_status.called.wait(), 1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiting, 0.5)
        _assert_no_background_tasks()

    asyncio.run(run())


def test_cancellation_cleanup_when_heartbeat_wait_for_swallows_cancel(monkeypatch):
    async def run():
        ref, worker = _ObjectRef(), _worker()
        worker.get_task_status.ref.value.set_result({"status": "processing"})
        entered, swallowed = asyncio.Event(), asyncio.Event()
        original_wait_for = asyncio.wait_for

        async def legacy_wait_for(future, timeout):
            # Model Python 3.10's wait_for race: its internal waiter is cancelled
            # after the inner Future completes, so it returns the RPC value.
            await asyncio.sleep(0)
            assert future.done()
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                swallowed.set()
                return future.result()

        monkeypatch.setattr(kernel_response.asyncio, "wait_for", legacy_wait_for)
        waiting = _start_wait(ref, worker)
        await original_wait_for(entered.wait(), 1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await original_wait_for(waiting, 0.5)
        assert swallowed.is_set()
        # Cancelling the caller cancels the directly awaited result wrapper.
        # Its done() guard therefore terminates the child even in this race.
        assert ref.value.cancelled()
        assert worker.get_token_in_use.calls == 0
        _assert_no_background_tasks()

    asyncio.run(run())


def test_result_exception_propagates_and_cleans_up_heartbeat():
    async def run():
        ref, worker = _ObjectRef(), _worker()
        waiting = _start_wait(ref, worker)
        await asyncio.wait_for(worker.get_task_status.called.wait(), 1)
        ref.value.set_exception(RuntimeError("evaluation worker died"))
        with pytest.raises(RuntimeError, match="evaluation worker died"):
            await asyncio.wait_for(waiting, 0.5)
        _assert_no_background_tasks()

    asyncio.run(run())


@pytest.mark.parametrize("status", ["failed", "timeout", "cancelled"])
def test_evaluation_failure_preserves_metadata(status):
    async def run():
        ref, worker = _ObjectRef(), _worker()
        ref.value.set_result({"status": status, "error_message": "test failure", "metadata": {"compile_s": 2.0}})
        result = await _wait_kernel_eval_result(ref, worker, PAYLOAD, 0, 32)
        assert result["status"] == status
        assert result["error_message"] == "test failure"
        assert result["metadata"]["compile_s"] == 2.0
        assert result["metadata"]["kernel_eval_failure"] is True
        assert result["metadata"]["task_id"] == PAYLOAD["task_id"]
        assert worker.get_task_status.calls == 0
        _assert_no_background_tasks()

    asyncio.run(run())


@pytest.mark.parametrize("interval", [0, -1, 0.01])
def test_result_delivery_with_saturated_default_executor(interval):
    async def run():
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        release, entered = threading.Event(), asyncio.Event()

        def occupy_executor():
            loop.call_soon_threadsafe(entered.set)
            release.wait()

        blocked = loop.run_in_executor(None, occupy_executor)
        await asyncio.wait_for(entered.wait(), 1)
        try:
            ref, worker = _ObjectRef(), _worker()
            waiting = _start_wait(ref, worker, interval)
            ref.value.set_result(RESULT)
            assert await asyncio.wait_for(waiting, 0.5) == RESULT
            assert worker.get_task_status.calls == 0
            _assert_no_background_tasks()
        finally:
            release.set()
            await blocked

    asyncio.run(run())


def test_healthy_heartbeat_reports_pending_then_stops_after_result(capsys):
    async def run():
        ref, worker = _ObjectRef(), _worker()
        worker.get_task_status.ref.value.set_result({"status": "processing"})
        worker.get_token_in_use.ref.value.set_result(7)
        waiting = _start_wait(ref, worker)
        await asyncio.sleep(0.05)
        assert not waiting.done()
        assert worker.get_token_in_use.calls > 0
        ref.value.set_result(RESULT)
        assert await asyncio.wait_for(waiting, 0.5) == RESULT
        _assert_no_background_tasks()

    asyncio.run(run())
    output = capsys.readouterr().out
    assert "pending=1" in output
    assert "status_last_seen=processing" in output
    assert "tokens_in_use=7/32" in output


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
