import asyncio
import inspect
import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import uuid4
from weakref import WeakKeyDictionary

import httpx
import ray

from slime.utils.misc import load_function
from slime.utils.types import Sample

_TASK_COUNTER = 0
_WORKERS: dict[tuple[str, int, int, int, int], Any] = {}


@dataclass
class _EvalCall:
    task_id: str
    deadline: float
    invalidated: bool = False
    worker: Any = None
    object_ref: Any = None


KERNEL_EVAL_DEADLINE: ContextVar[float | None] = ContextVar("kernel_eval_deadline", default=None)
_ACTIVE_EVALS: dict[tuple[str, str], _EvalCall] = {}
_CANCEL_SLOTS: WeakKeyDictionary = WeakKeyDictionary()
_CLEANUP_TASKS: set[asyncio.Task] = set()
logger = logging.getLogger(__name__)
_CONTROL_TIMEOUT_S = 2.0
_HEARTBEAT_STARTED_AT = time.time()
# Read timeout for the submit POST /evaluate, bounded separately from the (long)
# per-task run timeout: if the submit response is slow we fall through to polling
# /status rather than blocking for the full run. Kept generous (not a few seconds)
# because an overloaded /evaluate can enqueue the task quickly yet be slow to send
# the HTTP response -- a too-short value here just triggers redundant polling.
_SUBMIT_READ_TIMEOUT_S = 60


def next_kernel_task_id(prefix: str = "parallel_task") -> str:
    global _TASK_COUNTER
    try:
        _TASK_COUNTER += 1
    except Exception:
        _TASK_COUNTER = int(time.time() * 1000) % 1000000
    return f"{prefix}_{_TASK_COUNTER:06d}_{uuid4().hex[:8]}"


async def _await_ray(object_ref, timeout: float, *, cancel_on_error: bool = True):
    """Bound the wait; discard stale queries but preserve safety RPCs when requested."""
    try:
        async with asyncio.timeout(max(0.0, timeout)):
            if cancel_on_error:
                return await object_ref
            return await asyncio.shield(object_ref)
    except BaseException:
        if cancel_on_error:
            try:
                ray.cancel(object_ref, force=False, recursive=False)
            except Exception:
                logger.debug("Failed to cancel kernel eval RPC", exc_info=True)
        raise


async def _shielded_cleanup(coro):
    # Cleanup coroutines have their own deadline. Retain them even if the caller
    # is cancelled a second time (e.g. generate guard followed by worker stop).
    task = asyncio.create_task(coro)
    _CLEANUP_TASKS.add(task)

    def finished(task):
        _CLEANUP_TASKS.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("Kernel eval cleanup failed: %s", task.exception())

    task.add_done_callback(finished)
    return await asyncio.shield(task)


async def _delete_server_task(server_url: str, task_id: str, timeout: float) -> bool:
    # A POST may already be in flight. A first DELETE returning 404 does not
    # establish cancellation: retry within a separate, bounded cleanup budget.
    logger.info("Kernel eval event=cancel_requested scope=server task_id=%s", task_id)
    slots = _CANCEL_SLOTS.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(8))
    try:
        async with asyncio.timeout(timeout):
            async with slots, httpx.AsyncClient(
                timeout=httpx.Timeout(min(_CONTROL_TIMEOUT_S, timeout)),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            ) as client:
                while True:
                    try:
                        response = await client.delete(f"{server_url.rstrip('/')}/tasks/{task_id}")
                        if response.status_code in (200, 202, 204):
                            logger.info("Kernel eval event=cancel_acknowledged scope=server task_id=%s", task_id)
                            return True
                        if response.status_code == 404:
                            logger.warning("Kernel eval event=cancel_not_found scope=server task_id=%s", task_id)
                        elif response.status_code not in (429, 500, 502, 503, 504):
                            logger.error(
                                "Kernel eval event=cancel_failed scope=server task_id=%s http_status=%s",
                                task_id,
                                response.status_code,
                            )
                            return False
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.2)
    except TimeoutError:
        logger.warning("Kernel eval event=cancel_failed scope=server task_id=%s reason=cleanup_deadline", task_id)
        return False


@ray.remote
class _TokenBucketWorker:
    """Short, serial RPCs with idempotent leases; no blocking acquire tasks."""

    def __init__(self, rate_limit: int) -> None:
        self.rate_limit = max(1, int(rate_limit))
        self._leases: dict[str, float] = {}
        self._released: dict[str, float] = {}

    def acquire(self, lease_id: str, expires_at: float) -> bool:
        now = time.time()
        self._leases = {key: expiry for key, expiry in self._leases.items() if expiry > now}
        self._released = {key: expiry for key, expiry in self._released.items() if expiry > now}
        if expires_at <= now or lease_id in self._released:
            return False
        if lease_id not in self._leases and len(self._leases) >= self.rate_limit:
            return False
        self._leases[lease_id] = expires_at
        return True

    def release(self, lease_id: str, expires_at: float) -> None:
        self._leases.pop(lease_id, None)
        # Fence a late acquire whose caller timed out before observing its reply.
        self._released[lease_id] = expires_at

    def get_current_count(self) -> int:
        now = time.time()
        return sum(expiry > now for expiry in self._leases.values())


@ray.remote(concurrency_groups={"heartbeat": 4, "cancellation": 4})
class _HybridHttpWorker:
    def __init__(self, server_url: str, rate_limit: int, default_timeout: int, acquire_timeout: int) -> None:
        self.server_url = server_url.rstrip("/")
        self.default_timeout = int(default_timeout)
        self.acquire_timeout = int(acquire_timeout)
        self._limits = httpx.Limits(max_keepalive_connections=64, max_connections=128, keepalive_expiry=30.0)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=self.default_timeout, write=10.0, pool=5.0),
            limits=self._limits,
            headers={"Content-Type": "application/json"},
        )
        # Polling must not queue behind long synchronous POST responses.
        self._control_client = httpx.AsyncClient(
            timeout=_CONTROL_TIMEOUT_S,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
        )
        self._rate_limit_worker = _TokenBucketWorker.options(
            name="kernel-eval-rate-limiter-v2", get_if_exists=True
        ).remote(rate_limit)
        self._task_status: dict[str, dict[str, Any]] = {}
        # Ray concurrency groups run on distinct event loops/threads.
        self._lock = threading.Lock()
        self._running: dict[str, tuple[Any, asyncio.Task]] = {}
        self._invalidated: dict[str, float] = {}

    def _backoff(self, attempt: int, base: int = 2, cap: int = 30) -> float:
        return min(base**attempt, cap)

    @staticmethod
    def _is_duplicate_response(response: Any) -> bool:
        """Best-effort detection of an 'already accepted' response (e.g. HTTP 400/422
        with a duplicate message) so a resubmit of an already-enqueued task_id is
        polled rather than treated as a hard failure. HTTP 409 is handled directly."""
        if getattr(response, "status_code", None) not in (400, 409, 422):
            return False
        try:
            body = response.text.lower()
        except Exception:
            return False
        return any(m in body for m in ("already exists", "already submitted", "already running", "duplicate"))

    @ray.method(concurrency_group="heartbeat")
    async def get_token_in_use(self) -> int:
        try:
            return await _await_ray(self._rate_limit_worker.get_current_count.remote(), _CONTROL_TIMEOUT_S)
        except Exception:
            return -1

    @ray.method(concurrency_group="heartbeat")
    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._task_status.get(task_id, {}))

    @ray.method(concurrency_group="cancellation")
    async def invalidate(self, task_id: str, deadline: float) -> bool:
        """Fence queued calls and interrupt a running request on its owning loop."""
        with self._lock:
            now = time.time()
            self._invalidated = {key: expiry for key, expiry in self._invalidated.items() if expiry > now}
            self._invalidated[task_id] = max(deadline, now + _CONTROL_TIMEOUT_S)
            running = self._running.get(task_id)
            status = self._task_status.get(task_id, {})
            needs_delete = bool(status.get("submitted") and not status.get("terminal"))
            if running is not None:
                loop, task = running
                loop.call_soon_threadsafe(task.cancel)
        return needs_delete

    def _check_live(self, task_id: str, deadline: float) -> None:
        with self._lock:
            if task_id in self._invalidated:
                raise asyncio.CancelledError
        if time.time() >= deadline:
            raise TimeoutError

    async def submit_and_poll(
        self,
        task_data: dict[str, Any],
        client_timeout: float,
        max_retries: int,
        poll_interval: float,
        deadline: float | None = None,
        cancel_timeout: float = 5.0,
    ) -> dict[str, Any]:
        task_id = task_data["task_id"]
        # Wall-clock deadline travels across nodes; the local asyncio timeout
        # uses a monotonic clock. The caller also enforces its original budget.
        deadline = time.time() + client_timeout if deadline is None else deadline
        with self._lock:
            if task_id in self._invalidated:
                return {"status": "cancelled", "error_message": "Kernel eval call invalidated before execution"}
            if time.time() >= deadline:
                return {"status": "timeout", "error_message": "Kernel eval deadline expired in Ray queue"}
            self._running[task_id] = (asyncio.get_running_loop(), asyncio.current_task())
            self._task_status[task_id] = {"status": "submitting", "seen_at": time.time(), "submitted": False}
        try:
            async with asyncio.timeout(max(0.0, deadline - time.time())):
                return await self._submit_and_poll(task_data, max_retries, poll_interval, deadline)
        except TimeoutError:
            return {"status": "timeout", "error_message": f"Task timeout after {client_timeout}s (client-side)"}
        except Exception as exc:
            return {"status": "failed", "error_message": str(exc)}
        finally:
            with self._lock:
                state = self._task_status[task_id]
                needs_delete = state.get("submitted") and not state.get("terminal")
            try:
                if needs_delete:
                    confirmed = await _shielded_cleanup(_delete_server_task(self.server_url, task_id, cancel_timeout))
                    if not confirmed:
                        logger.warning("Kernel eval server cancellation unconfirmed: task_id=%s", task_id)
            finally:
                with self._lock:
                    self._running.pop(task_id, None)

    async def _submit_and_poll(self, task_data, max_retries, poll_interval, deadline):
        task_id = task_data["task_id"]
        submitted_payload = None
        attempt = 0
        unlimited = max_retries is None or max_retries == -1
        while unlimited or attempt < max(1, max_retries):
            self._check_live(task_id, deadline)
            lease_id = uuid4().hex
            try:
                # Each acquire is nonblocking. Releasing by ID is safe even if
                # cancellation races with the grant or its reply.
                try:
                    async with asyncio.timeout(min(self.acquire_timeout, max(0.0, deadline - time.time()))):
                        while True:
                            self._check_live(task_id, deadline)
                            acquired = await _await_ray(
                                self._rate_limit_worker.acquire.remote(lease_id, deadline),
                                min(_CONTROL_TIMEOUT_S, max(0.0, deadline - time.time())),
                            )
                            if acquired:
                                break
                            await asyncio.sleep(0.05)
                except TimeoutError:
                    return {"status": "timeout", "error_message": "rate limiter acquire timeout"}
                self._check_live(task_id, deadline)
                with self._lock:
                    # Set before the first await in POST: cancellation must cover
                    # transport errors too, since the server may have accepted it.
                    if task_id in self._invalidated:
                        raise asyncio.CancelledError
                    self._task_status[task_id]["submitted"] = True
                print(f"[HybridWorker] POST /evaluate task_id={task_id} url={self.server_url}")
                submit_timeout = min(self.default_timeout, _SUBMIT_READ_TIMEOUT_S, max(0.0, deadline - time.time()))
                self._check_live(task_id, deadline)
                if submitted_payload is None:
                    # Keep this value stable across POST retries (request hash).
                    remaining = deadline - time.time()
                    submitted_payload = {
                        **task_data,
                        "workflow_timeout": min(float(task_data.get("workflow_timeout", remaining)), remaining),
                    }
                async with asyncio.timeout(submit_timeout):
                    response = await self._client.post(
                        f"{self.server_url}/evaluate", json=submitted_payload, timeout=submit_timeout
                    )
                print(f"[HybridWorker] POST /evaluate resp={response.status_code} task_id={task_id}")
            except (httpx.TimeoutException, httpx.ConnectError, TimeoutError) as exc:
                if unlimited or attempt < max(1, max_retries) - 1:
                    response = None
                else:
                    # A submit timeout can still mean accepted. Preserve polling
                    # and retrieval of the server's real compilation/error result.
                    print(f"[HybridWorker] submit transport error task_id={task_id} err={exc!r}; polling")
                    break
            finally:
                try:
                    await _shielded_cleanup(
                        _await_ray(
                            self._rate_limit_worker.release.remote(lease_id, deadline),
                            _CONTROL_TIMEOUT_S,
                            cancel_on_error=False,
                        )
                    )
                except Exception:
                    logger.exception("Failed to release kernel eval lease %s", lease_id)

            self._check_live(task_id, deadline)
            if response is not None:
                if response.status_code == 409 and "cancelled" in response.text.lower():
                    return {"status": "cancelled", "error_message": "Server rejected cancelled task ID"}
                if response.status_code in (200, 409) or self._is_duplicate_response(response):
                    break
                if response.status_code not in (429, 503):
                    response.raise_for_status()
            base = 5 if response is not None and response.status_code == 503 else 2
            await asyncio.sleep(min(self._backoff(attempt, base=base), max(0.0, deadline - time.time())))
            attempt += 1

        last_status = None
        while True:
            self._check_live(task_id, deadline)
            try:
                request_timeout = min(_CONTROL_TIMEOUT_S, max(0.0, deadline - time.time()))
                async with asyncio.timeout(request_timeout):
                    status_response = await self._control_client.get(
                        f"{self.server_url}/status/{task_id}", timeout=request_timeout
                    )
                if status_response.status_code == 200:
                    status_payload = status_response.json()
                    status = status_payload.get("status", "unknown")
                    terminal = status in ("completed", "failed", "timeout", "cancelled")
                    with self._lock:
                        self._task_status[task_id].update(status=status, seen_at=time.time(), terminal=terminal)
                    if status != last_status:
                        last_status = status
                        print(f"[HybridWorker] STATUS task_id={task_id} -> {status}")
                    if terminal:
                        error_message = status_payload.get("error_message", f"Task {status}")
                        # Fetch results for all terminal states, retaining compile
                        # diagnostics on server timeout/cancellation as before.
                        try:
                            request_timeout = min(_CONTROL_TIMEOUT_S, max(0.0, deadline - time.time()))
                            async with asyncio.timeout(request_timeout):
                                result_response = await self._control_client.get(
                                    f"{self.server_url}/results/{task_id}", timeout=request_timeout
                                )
                            if result_response.status_code == 200:
                                result = result_response.json()
                                result["status"] = status
                                if status != "completed":
                                    result["error_message"] = result.get("error_message") or error_message
                                return result
                        except (httpx.HTTPError, TimeoutError, ValueError):
                            pass
                        return {"status": status, "error_message": error_message}
                # 404 is not evidence of completion or successful cancellation.
            except (httpx.HTTPError, TimeoutError, ValueError):
                pass
            await asyncio.sleep(min(poll_interval, max(0.0, deadline - time.time())))


def _ensure_ray_initialized() -> None:
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)


_MISSING = object()


def _kernel_eval_param(args, config: dict[str, Any], name: str, default: Any = _MISSING) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    if name in config:
        return config[name]
    if default is not _MISSING:
        return default
    raise KeyError(f"Missing kernel eval parameter: {name}")


async def cancel_kernel_eval(args, task_id: str | None, config: dict[str, Any]) -> bool:
    if not task_id:
        return False

    config = dict(config)
    server_url = _kernel_eval_param(args, config, "kernel_env_url", None)
    if not server_url:
        return False
    server_url = str(server_url).rstrip("/")
    timeout = float(_kernel_eval_param(args, config, "kernel_eval_cancel_timeout", 5.0))
    active = _ACTIVE_EVALS.get((server_url, task_id))
    if active is not None:
        active.invalidated = True
        return await _shielded_cleanup(
            _cancel_eval_call(active.worker, active.object_ref, server_url, task_id, active.deadline, timeout)
        )
    return await _shielded_cleanup(_delete_server_task(server_url, task_id, timeout))


async def _cancel_eval_call(worker, object_ref, server_url, task_id, deadline, timeout):
    active = _ACTIVE_EVALS.get((server_url, task_id))
    if active is not None:
        active.invalidated = True
    logger.info("Kernel eval event=cancel_requested scope=call task_id=%s", task_id)

    async def invalidate_ray_call():
        if worker is None:
            return True  # Locally invalidated before a Ray call was created.
        invalidation = None
        try:
            invalidation = worker.invalidate.remote(task_id, deadline)
        except Exception:
            logger.exception("Kernel eval event=cancel_failed scope=invalidation task_id=%s", task_id)
        try:
            if object_ref is not None:
                ray.cancel(object_ref, force=False, recursive=False)
        except Exception:
            logger.warning("Kernel eval event=cancel_failed scope=ray_cancel task_id=%s", task_id, exc_info=True)
        if invalidation is None:
            return False
        try:
            # Preserve a late invalidation RPC: it must still fence queued work.
            await _await_ray(invalidation, min(_CONTROL_TIMEOUT_S, timeout), cancel_on_error=False)
            logger.info("Kernel eval event=cancel_acknowledged scope=invalidation task_id=%s", task_id)
            return True
        except Exception:
            logger.exception("Kernel eval event=cancel_failed scope=invalidation task_id=%s", task_id)
            return False

    # Even a queued/unsubmitted call needs the server tombstone. Neither a Ray
    # cancellation error nor a slow control RPC may delay that independent fence.
    invalidated, confirmed = await asyncio.gather(
        invalidate_ray_call(), _delete_server_task(server_url, task_id, timeout)
    )
    return invalidated and confirmed


def _get_kernel_eval_worker(args, config: dict[str, Any]):
    _ensure_ray_initialized()
    server_url = _kernel_eval_param(args, config, "kernel_env_url")
    if not server_url:
        raise RuntimeError("kernel_env_url is required for kernel eval.")
    task_timeout = int(_kernel_eval_param(args, config, "kernel_eval_task_timeout"))
    worker_max_concurrency = int(_kernel_eval_param(args, config, "kernel_eval_worker_max_concurrency"))
    rate_limit = int(_kernel_eval_param(args, config, "kernel_eval_rate_limit"))
    acquire_timeout = int(_kernel_eval_param(args, config, "kernel_eval_acquire_timeout"))
    worker_key = (
        server_url,
        task_timeout,
        worker_max_concurrency,
        rate_limit,
        acquire_timeout,
    )
    if worker_key not in _WORKERS:
        _WORKERS[worker_key] = _HybridHttpWorker.options(max_concurrency=worker_max_concurrency).remote(
            server_url,
            rate_limit,
            task_timeout,
            acquire_timeout,
        )
    return _WORKERS[worker_key]


def _format_env_return(result: dict[str, Any], task_payload: dict[str, Any]) -> dict[str, Any]:
    status = result.get("status") or "failed"
    if status == "completed":
        return result

    error_message = result.get("error_message") or result.get("error") or f"Kernel eval task {status}"
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "kernel_eval_failure": True,
            "task_id": task_payload.get("task_id"),
            "entry_point": task_payload.get("entry_point"),
            "backend": task_payload.get("backend"),
        }
    )
    env_state = {
        **result,
        "status": status,
        "success": result.get("success"),
        "correctness": result.get("correctness"),
        "compiled": result.get("compiled"),
        "speedup": 0.0,
        "error": error_message,
        "error_message": error_message,
        "metadata": metadata,
    }
    return env_state


async def _wait_kernel_eval_result(
    object_ref,
    worker,
    task_payload: dict[str, Any],
    heartbeat_interval: float,
    rate_limit: int,
    timeout: float,
) -> dict[str, Any]:
    start_time = time.monotonic()

    async def heartbeat():
        while True:
            await asyncio.sleep(heartbeat_interval)
            task_status, tokens_in_use = {}, -1
            try:
                task_status = await _await_ray(
                    worker.get_task_status.remote(task_payload["task_id"]), _CONTROL_TIMEOUT_S
                )
                tokens_in_use = await _await_ray(worker.get_token_in_use.remote(), _CONTROL_TIMEOUT_S)
            except Exception:
                # Observability must never gate result delivery or cancellation.
                logger.debug("Kernel eval heartbeat unavailable", exc_info=True)
            seen_at = task_status.get("seen_at")
            status_age = time.time() - seen_at if seen_at else None
            print(
                "[BatchHeartbeat] kernel_eval: "
                f"completed=0/1, pending=1, pending_duration={time.monotonic() - start_time:.1f}s, "
                f"total_elapsed={time.time() - _HEARTBEAT_STARTED_AT:.1f}s "
                f"status_last_seen={task_status.get('status', 'unknown')}, "
                f"status_age={(f'{status_age:.1f}s' if status_age is not None else 'N/A')}, "
                f"tokens_in_use={tokens_in_use}/{rate_limit}"
            )
            print(
                "[BatchHeartbeat] pending_tasks: "
                f"task_id={task_payload.get('task_id')} entry={task_payload.get('entry_point')} "
                f"uuid={(task_payload.get('uuid') or 'N/A')[:8]}"
            )

    heartbeat_task = asyncio.create_task(heartbeat()) if heartbeat_interval > 0 else None
    try:
        async with asyncio.timeout(max(0.0, timeout)):
            result = await object_ref
        return _format_env_return(result, task_payload)
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)


async def run_kernel_eval(args, sample: Sample, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    eval_func_path = _kernel_eval_param(args, config, "kernel_eval_function_path", None)
    if eval_func_path:
        eval_func = load_function(eval_func_path)
        result = eval_func(args, sample, payload)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict) and isinstance(result.get("env_state"), dict):
            return result
        if isinstance(result, dict):
            return {"env_state": result}
        return result

    client_timeout = float(_kernel_eval_param(args, config, "kernel_eval_client_timeout"))
    cancel_timeout = float(_kernel_eval_param(args, config, "kernel_eval_cancel_timeout", 5.0))
    started = time.monotonic()
    deadline = time.time() + client_timeout
    trajectory_deadline = KERNEL_EVAL_DEADLINE.get()
    if trajectory_deadline is not None:
        deadline = min(deadline, trajectory_deadline)
    server_url = str(_kernel_eval_param(args, config, "kernel_env_url")).rstrip("/")
    task_id = payload["task_id"]
    call = _EvalCall(task_id, deadline)
    _ACTIVE_EVALS[(server_url, task_id)] = call
    try:
        if time.time() >= deadline:
            raise TimeoutError
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        call.worker = worker = _get_kernel_eval_worker(args, config)
        # Worker lookup may itself take time. Never create expired queued work.
        if call.invalidated or asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        if time.time() >= deadline:
            raise TimeoutError
        call.object_ref = object_ref = worker.submit_and_poll.remote(
            payload,
            client_timeout=client_timeout,
            max_retries=int(_kernel_eval_param(args, config, "kernel_eval_max_retries")),
            poll_interval=float(_kernel_eval_param(args, config, "kernel_eval_poll_interval")),
            deadline=deadline,
            cancel_timeout=cancel_timeout,
        )
        result = await _wait_kernel_eval_result(
            object_ref,
            worker,
            payload,
            heartbeat_interval=float(_kernel_eval_param(args, config, "kernel_eval_heartbeat_interval", 120.0)),
            rate_limit=int(_kernel_eval_param(args, config, "kernel_eval_rate_limit")),
            timeout=min(client_timeout - (time.monotonic() - started), deadline - time.time()),
        )
        return {"env_state": result}
    except BaseException as exc:
        call.invalidated = True
        try:
            await _shielded_cleanup(
                _cancel_eval_call(call.worker, call.object_ref, server_url, task_id, deadline, cancel_timeout)
            )
        except Exception:
            logger.exception("Kernel eval cleanup failed: task_id=%s", task_id)
        if isinstance(exc, TimeoutError):
            return {
                "env_state": _format_env_return(
                    {"status": "timeout", "error_message": f"Task timeout after {client_timeout}s (client-side)"},
                    payload,
                )
            }
        raise
    finally:
        _ACTIVE_EVALS.pop((server_url, task_id), None)
