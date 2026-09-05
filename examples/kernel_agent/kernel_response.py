import asyncio
import inspect
import threading
import time
from typing import Any
from uuid import uuid4

import httpx
import ray

from slime.utils.misc import load_function
from slime.utils.types import Sample

try:
    from .utils import extract_cuda_agent_kernel_code
except ImportError:
    from utils import extract_cuda_agent_kernel_code


_TASK_COUNTER = 0
_WORKERS: dict[tuple[str, int, int, int, int], Any] = {}
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


@ray.remote(concurrency_groups={"acquire": 1000, "release": 1000})
class _TokenBucketWorker:
    def __init__(self, rate_limit: int) -> None:
        self.rate_limit = max(1, int(rate_limit))
        self.current_count = 0
        self._semaphore = threading.Semaphore(self.rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self) -> bool:
        self._semaphore.acquire()
        self.current_count += 1
        return True

    @ray.method(concurrency_group="release")
    def release(self) -> None:
        self._semaphore.release()
        self.current_count = max(0, self.current_count - 1)

    def get_current_count(self) -> int:
        return self.current_count


@ray.remote
class _HybridHttpWorker:
    def __init__(self, server_url: str, rate_limit: int, default_timeout: int, acquire_timeout: int) -> None:
        self.server_url = server_url.rstrip("/")
        self.default_timeout = int(default_timeout)
        self.acquire_timeout = int(acquire_timeout)
        self._limits = httpx.Limits(max_keepalive_connections=64, max_connections=128, keepalive_expiry=30.0)
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=self.default_timeout, write=10.0, pool=5.0),
            limits=self._limits,
            headers={"Content-Type": "application/json"},
        )
        self._rate_limit_worker = _TokenBucketWorker.options(
            name="kernel-eval-rate-limiter", get_if_exists=True
        ).remote(rate_limit)
        self._task_status: dict[str, dict[str, Any]] = {}

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

    def get_token_in_use(self) -> int:
        try:
            return ray.get(self._rate_limit_worker.get_current_count.remote())
        except Exception:
            return -1

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        return self._task_status.get(task_id, {})

    def submit_and_poll(
        self,
        task_data: dict[str, Any],
        client_timeout: int,
        max_retries: int,
        poll_interval: float,
    ) -> dict[str, Any]:
        start_time = time.time()
        attempt = 0
        unlimited = max_retries is None or max_retries == -1
        submit_read_timeout = min(self.default_timeout, _SUBMIT_READ_TIMEOUT_S)
        submitted = False  # POST /evaluate confirmed (HTTP 200)
        submit_error: str | None = None  # last transport error when submit was NOT confirmed

        while unlimited or attempt < max(1, max_retries):
            try:
                acquire_ref = self._rate_limit_worker.acquire.remote()
                ready, _ = ray.wait([acquire_ref], timeout=self.acquire_timeout)
                if not ready:
                    current = self.get_token_in_use()
                    print(f"[HybridWorker] acquire timeout tokens_in_use={current}")
                    return {"status": "failed", "error_message": "rate limiter acquire timeout"}

                if attempt == 0:
                    print(
                        f"[HybridWorker] POST /evaluate task_id={task_data.get('task_id', '')} url={self.server_url}"
                    )
                response = self._client.post(
                    f"{self.server_url}/evaluate",
                    json=task_data,
                    timeout=httpx.Timeout(connect=10.0, read=submit_read_timeout, write=10.0, pool=5.0),
                )
                try:
                    print(
                        f"[HybridWorker] POST /evaluate resp={response.status_code} "
                        f"task_id={task_data.get('task_id', '')}"
                    )
                except Exception:
                    pass

                try:
                    self._rate_limit_worker.release.remote()
                except Exception:
                    pass

                if response.status_code == 200:
                    submitted = True
                    break
                if response.status_code == 409 or self._is_duplicate_response(response):
                    # The task_id was already accepted server-side -- typically a prior
                    # POST that transport-timed-out on the client still enqueued it, and
                    # this retry is a duplicate. Treat it as submitted and poll the
                    # existing task instead of raising it as a hard failure.
                    print(
                        f"[HybridWorker] POST /evaluate duplicate/conflict ({response.status_code}) "
                        f"task_id={task_data.get('task_id', '')}; polling existing task"
                    )
                    submitted = True
                    break
                if response.status_code in (429, 503):
                    time.sleep(self._backoff(attempt, base=2 if response.status_code == 429 else 5))
                    attempt += 1
                    continue
                response.raise_for_status()
            # A reused keep-alive connection can be closed by the server or an
            # intervening tunnel before it sends response headers.  httpx raises
            # RemoteProtocolError for that case; it is just as retryable as a
            # connect/read timeout.  Reusing the caller-provided task_id keeps
            # retries idempotent when the first POST actually reached KernelGym.
            except httpx.TransportError as exc:
                try:
                    self._rate_limit_worker.release.remote()
                except Exception:
                    pass
                submit_error = str(exc)
                if unlimited or attempt < max(1, max_retries) - 1:
                    time.sleep(self._backoff(attempt))
                    attempt += 1
                    continue
                # Retries exhausted on a *transport* timeout: the submit response was
                # not confirmed, but the task (client-provided task_id) may well have
                # been enqueued server-side. Fall through to polling /results instead
                # of giving up -- otherwise a slow/overloaded submit is reported as a
                # bare "timed out" and the server's real result (incl. compiled state)
                # is never fetched.
                print(
                    f"[HybridWorker] submit transport error task_id={task_data.get('task_id', '')} "
                    f"err={submit_error!r}; polling /results in case it was enqueued"
                )
                break
            except Exception as exc:
                try:
                    self._rate_limit_worker.release.remote()
                except Exception:
                    pass
                return {"status": "failed", "error_message": str(exc)}

        task_id = task_data.get("task_id", "")
        last_status = None
        _ = (submitted, submit_error)  # captured for logging above; not used to gate polling
        # IMPORTANT: do NOT treat a 404 from /status as "task not enqueued". For split
        # (compile+execute) workflows the parent task_id returns 404 the entire time its
        # sub-tasks run -- the parent result only materializes when the workflow finishes
        # (which for a hung kernel is at the full task timeout). Bailing on a 404 abandons
        # a task that is actually running, so we poll until the task resolves to a terminal
        # status or the client timeout elapses.
        while time.time() - start_time < client_timeout:
            try:
                status_response = self._client.get(f"{self.server_url}/status/{task_id}")
                if status_response.status_code == 200:
                    status_payload = status_response.json()
                    status = status_payload.get("status", "unknown")
                    if status != last_status:
                        last_status = status
                        self._task_status[task_id] = {
                            "status": status,
                            "seen_at": time.time(),
                        }
                        try:
                            print(f"[HybridWorker] STATUS task_id={task_id} -> {status}")
                        except Exception:
                            pass
                    if status in ("completed", "failed", "timeout", "cancelled"):
                        error_message = status_payload.get("error_message", f"Task {status}")
                        # Fetch /results for ANY terminal status, not just completed/failed:
                        # a server-side timeout (or cancel) still writes a result carrying
                        # the real compiled state + stage metadata (e.g. kg_kernel_backend_
                        # compile_s), which must be surfaced rather than replaced by a bare
                        # status. Fall back to the bare status only if no result exists.
                        try:
                            result_response = self._client.get(f"{self.server_url}/results/{task_id}")
                            if result_response.status_code == 200:
                                result = result_response.json()
                                result["status"] = status
                                if status != "completed":
                                    result["error_message"] = result.get("error_message") or error_message
                                return result
                        except Exception:
                            pass
                        return {"status": status, "error_message": error_message}
            except Exception:
                pass
            time.sleep(poll_interval)

        return {"status": "timeout", "error_message": f"Task timeout after {client_timeout}s (client-side)"}


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
    timeout = float(_kernel_eval_param(args, config, "kernel_eval_cancel_timeout", 5.0))

    def _delete() -> bool:
        timeout_config = httpx.Timeout(connect=2.0, read=timeout, write=2.0, pool=2.0)
        with httpx.Client(timeout=timeout_config) as client:
            response = client.delete(f"{str(server_url).rstrip('/')}/tasks/{task_id}")
            return response.status_code in (200, 202, 204, 404)

    return await asyncio.to_thread(_delete)


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


def _build_kernel_eval_payload(args, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    task_payload = {
        "task_id": payload.get("task_id") or next_kernel_task_id(),
        "reference_code": payload.get("reference_code", payload.get("ground_truth")),
        "kernel_code": payload.get("kernel_code") or extract_cuda_agent_kernel_code(payload["response"]),
        "backend": payload.get("backend", payload.get("kernel_backend")),
        "entry_point": payload["entry_point"],
        # KernelGym's CUDA-Agent/TVM-FFI static precheck uses this to avoid
        # classifying an intentional FP16/BF16 task as an FP32 downgrade.
        "precision": payload.get("precision", "fp32"),
        "num_correct_trials": payload.get(
            "num_correct_trials", _kernel_eval_param(args, config, "num_correct_trials")
        ),
        "num_perf_trials": payload.get("num_perf_trials", _kernel_eval_param(args, config, "num_perf_trials")),
        # Warmup iterations before the timed perf trials. Server-side field
        # exists (api/models.py num_warmup, 0-100); without sending it the
        # service silently used its default of 3.
        "num_warmup": payload.get("num_warmup", _kernel_eval_param(args, config, "num_warmup", 3)),
        "perf_trim_count": payload.get("perf_trim_count", _kernel_eval_param(args, config, "perf_trim_count", 0)),
        "adaptive_perf_trials": payload.get(
            "adaptive_perf_trials", _kernel_eval_param(args, config, "adaptive_perf_trials", False)
        ),
        "perf_min_trials": payload.get("perf_min_trials", _kernel_eval_param(args, config, "perf_min_trials", 20)),
        "perf_cv_threshold": payload.get(
            "perf_cv_threshold", _kernel_eval_param(args, config, "perf_cv_threshold", 0.05)
        ),
        "timeout": payload.get("timeout", _kernel_eval_param(args, config, "kernel_eval_task_timeout")),
        "priority": payload.get("priority", _kernel_eval_param(args, config, "kernel_eval_priority", "normal")),
        "is_valid": payload.get("is_valid", False),
        "verbose_errors": payload.get("verbose_errors", _kernel_eval_param(args, config, "verbose_errors", True)),
        "enable_profiling": payload.get(
            "enable_profiling", _kernel_eval_param(args, config, "enable_profiling", True)
        ),
        "detect_decoy_kernel": payload.get(
            "detect_decoy_kernel", _kernel_eval_param(args, config, "detect_decoy_kernel", True)
        ),
        "reference_backend": payload.get("reference_backend"),
        "uuid": payload.get("uuid"),
    }
    if payload.get("uuid"):
        task_payload["uuid"] = payload["uuid"]
    # Reference-timing cache (KernelGym use_reference_cache) is keyed by uuid, so
    # only request it when a stable uuid is present, otherwise KernelGym cannot
    # share the cached baseline across attempts of the same problem.
    if task_payload.get("uuid") and payload.get(
        "use_reference_cache", _kernel_eval_param(args, config, "use_reference_cache", False)
    ):
        task_payload["use_reference_cache"] = True
    if payload.get("split_compile_and_execute", _kernel_eval_param(args, config, "split_compile_and_execute", True)):
        task_payload["split_compile_and_execute"] = True
    if payload.get(
        "enable_compile_artifact_cache", _kernel_eval_param(args, config, "enable_compile_artifact_cache", True)
    ):
        task_payload["enable_compile_artifact_cache"] = True
    # Optional separate reference perf-trial count; omit when unset so the server
    # falls back to num_perf_trials.
    refer_num_perf_trials = payload.get(
        "refer_num_perf_trials", _kernel_eval_param(args, config, "refer_num_perf_trials", None)
    )
    if refer_num_perf_trials is not None:
        task_payload["refer_num_perf_trials"] = refer_num_perf_trials
    # Correctness-stage timeout overrides: only send when set so an unset value
    # leaves the server's config/formula in effect.
    correctness_timeout = payload.get(
        "correctness_timeout", _kernel_eval_param(args, config, "correctness_timeout", None)
    )
    if correctness_timeout is not None:
        task_payload["correctness_timeout"] = correctness_timeout
    correctness_timeout_enabled = payload.get(
        "correctness_timeout_enabled", _kernel_eval_param(args, config, "correctness_timeout_enabled", None)
    )
    if correctness_timeout_enabled is not None:
        task_payload["correctness_timeout_enabled"] = correctness_timeout_enabled
    return task_payload


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
) -> dict[str, Any]:
    start_time = time.time()
    pending = [object_ref]
    if heartbeat_interval <= 0:
        result = await asyncio.to_thread(ray.get, object_ref)
        return _format_env_return(result, task_payload)

    while pending:
        done, pending = await asyncio.to_thread(ray.wait, pending, num_returns=1, timeout=heartbeat_interval)
        if done:
            result = await asyncio.to_thread(ray.get, done[0])
            return _format_env_return(result, task_payload)

        elapsed = time.time() - start_time
        total_elapsed = time.time() - _HEARTBEAT_STARTED_AT
        task_status = await asyncio.to_thread(ray.get, worker.get_task_status.remote(task_payload.get("task_id")))
        status_last_seen = task_status.get("status", "unknown")
        status_seen_at = task_status.get("seen_at")
        status_age = time.time() - status_seen_at if status_seen_at else None
        tokens_in_use = await asyncio.to_thread(ray.get, worker.get_token_in_use.remote())
        print(
            "[BatchHeartbeat] kernel_eval: "
            f"completed=0/1, pending=1, pending_duration={elapsed:.1f}s, total_elapsed={total_elapsed:.1f}s "
            f"status_last_seen={status_last_seen}, "
            f"status_age={(f'{status_age:.1f}s' if status_age is not None else 'N/A')}, "
            f"tokens_in_use={tokens_in_use}/{rate_limit}"
        )
        print(
            "[BatchHeartbeat] pending_tasks: "
            f"task_id={task_payload.get('task_id')} entry={task_payload.get('entry_point')} "
            f"uuid={(task_payload.get('uuid') or 'N/A')[:8]}"
        )

    return _format_env_return(
        {"status": "failed", "error_message": "Kernel eval task disappeared before completion"},
        task_payload,
    )


async def run_kernel_eval(args, sample: Sample, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    eval_func_path = _kernel_eval_param(args, config, "kernel_eval_function_path", None)
    if eval_func_path:
        task_payload = {
            "task_id": payload.get("task_id"),
            "entry_point": payload.get("entry_point"),
            "backend": payload.get("backend", payload.get("kernel_backend")),
            "uuid": payload.get("uuid"),
        }
        eval_func = load_function(eval_func_path)
        result = eval_func(args, sample, payload)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict) and isinstance(result.get("env_state"), dict):
            return result
        if isinstance(result, dict):
            return {"env_state": result}
        return result

    worker = _get_kernel_eval_worker(args, config)
    task_payload = _build_kernel_eval_payload(args, payload, config)
    object_ref = worker.submit_and_poll.remote(
        task_payload,
        client_timeout=int(_kernel_eval_param(args, config, "kernel_eval_client_timeout")),
        max_retries=int(_kernel_eval_param(args, config, "kernel_eval_max_retries")),
        poll_interval=float(_kernel_eval_param(args, config, "kernel_eval_poll_interval")),
    )
    result = await _wait_kernel_eval_result(
        object_ref,
        worker,
        task_payload,
        heartbeat_interval=float(_kernel_eval_param(args, config, "kernel_eval_heartbeat_interval", 60.0)),
        rate_limit=int(_kernel_eval_param(args, config, "kernel_eval_rate_limit")),
    )
    return {"env_state": result}
