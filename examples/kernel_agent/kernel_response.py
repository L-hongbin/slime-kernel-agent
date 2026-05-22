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
        self._rate_limit_worker = _TokenBucketWorker.options(name="kernel-eval-rate-limiter", get_if_exists=True).remote(
            rate_limit
        )
        self._task_status: dict[str, dict[str, Any]] = {}

    def _backoff(self, attempt: int, base: int = 2, cap: int = 30) -> float:
        return min(base**attempt, cap)

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

        while unlimited or attempt < max(1, max_retries):
            try:
                acquire_ref = self._rate_limit_worker.acquire.remote()
                ready, _ = ray.wait([acquire_ref], timeout=self.acquire_timeout)
                if not ready:
                    current = self.get_token_in_use()
                    print(f"[HybridWorker] acquire timeout tokens_in_use={current}")
                    return {"status": "failed", "error_message": "rate limiter acquire timeout"}

                if attempt == 0:
                    print(f"[HybridWorker] POST /evaluate task_id={task_data.get('task_id', '')} url={self.server_url}")
                response = self._client.post(f"{self.server_url}/evaluate", json=task_data)
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
                    break
                if response.status_code in (429, 503):
                    time.sleep(self._backoff(attempt, base=2 if response.status_code == 429 else 5))
                    attempt += 1
                    continue
                response.raise_for_status()
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                try:
                    self._rate_limit_worker.release.remote()
                except Exception:
                    pass
                if unlimited or attempt < max(1, max_retries) - 1:
                    time.sleep(self._backoff(attempt))
                    attempt += 1
                    continue
                return {"status": "failed", "error_message": str(exc)}
            except Exception as exc:
                try:
                    self._rate_limit_worker.release.remote()
                except Exception:
                    pass
                return {"status": "failed", "error_message": str(exc)}

        task_id = task_data.get("task_id", "")
        last_status = None
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
                        if status in ("completed", "failed"):
                            result_response = self._client.get(f"{self.server_url}/results/{task_id}")
                            if result_response.status_code == 200:
                                result = result_response.json()
                                result["status"] = status
                                if status == "failed":
                                    result["error_message"] = result.get("error_message", error_message)
                                return result
                            return {
                                "status": status,
                                "error_message": f"Failed to fetch results: HTTP {result_response.status_code}",
                            }
                        return {"status": status, "error_message": error_message}
            except Exception:
                pass
            time.sleep(poll_interval)

        return {"status": "timeout", "error_message": f"Task timeout after {client_timeout}s (client-side)"}


def _ensure_ray_initialized() -> None:
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)


def _get_kernel_eval_worker(config: dict[str, Any]):
    _ensure_ray_initialized()
    server_url = config["kernel_env_url"]
    if not server_url:
        raise RuntimeError("kernel_env_url is required for kernel eval.")
    worker_key = (
        server_url,
        int(config["kernel_eval_task_timeout"]),
        int(config["kernel_eval_worker_max_concurrency"]),
        int(config["kernel_eval_rate_limit"]),
        int(config["kernel_eval_acquire_timeout"]),
    )
    if worker_key not in _WORKERS:
        _WORKERS[worker_key] = _HybridHttpWorker.options(
            max_concurrency=int(config["kernel_eval_worker_max_concurrency"])
        ).remote(
            server_url,
            int(config["kernel_eval_rate_limit"]),
            int(config["kernel_eval_task_timeout"]),
            int(config["kernel_eval_acquire_timeout"]),
        )
    return _WORKERS[worker_key]


def _build_kernel_eval_payload(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    task_payload = {
        "task_id": payload.get("task_id") or next_kernel_task_id(),
        "reference_code": payload.get("reference_code", payload.get("ground_truth")),
        "kernel_code": payload.get("kernel_code") or extract_cuda_agent_kernel_code(payload["response"]),
        "backend": payload.get("backend", payload.get("kernel_backend")),
        "entry_point": payload["entry_point"],
        "num_correct_trials": payload.get("num_correct_trials", config["num_correct_trials"]),
        "num_perf_trials": payload.get("num_perf_trials", config["num_perf_trials"]),
        "timeout": payload.get("timeout", config["kernel_eval_task_timeout"]),
        "priority": payload.get("priority", "normal"),
        "is_valid": payload.get("is_valid", False),
        "verbose_errors": payload.get("verbose_errors", config["verbose_errors"]),
        "enable_profiling": payload.get("enable_profiling", config["enable_profiling"]),
        "detect_decoy_kernel": payload.get("detect_decoy_kernel", config["detect_decoy_kernel"]),
        "reference_backend": payload.get("reference_backend"),
    }
    if payload.get("uuid"):
        task_payload["uuid"] = payload["uuid"]
    if payload.get("split_compile_and_execute", config["split_compile_and_execute"]):
        task_payload["split_compile_and_execute"] = True
    if payload.get("enable_compile_artifact_cache", config["enable_compile_artifact_cache"]):
        task_payload["enable_compile_artifact_cache"] = True
    return task_payload


def _normalize_env_failure(result: dict[str, Any], task_payload: dict[str, Any]) -> dict[str, Any]:
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
        "success": False,
        "correctness": False,
        "compiled": False,
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
    while pending:
        done, pending = await asyncio.to_thread(ray.wait, pending, num_returns=1, timeout=heartbeat_interval)
        if done:
            result = await asyncio.to_thread(ray.get, done[0])
            return _normalize_env_failure(result, task_payload)

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

    return _normalize_env_failure(
        {"status": "failed", "error_message": "Kernel eval task disappeared before completion"},
        task_payload,
    )


async def run_kernel_eval(args, sample: Sample, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    eval_func_path = config["kernel_eval_function_path"]
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
            return {"env_state": result, "reward_extra_info": result}
        return result

    worker = _get_kernel_eval_worker(config)
    task_payload = _build_kernel_eval_payload(payload, config)
    object_ref = worker.submit_and_poll.remote(
        task_payload,
        client_timeout=int(config["kernel_eval_client_timeout"]),
        max_retries=int(config["kernel_eval_max_retries"]),
        poll_interval=float(config["kernel_eval_poll_interval"]),
    )
    result = await _wait_kernel_eval_result(
        object_ref,
        worker,
        task_payload,
        heartbeat_interval=float(config.get("kernel_eval_heartbeat_interval", 60.0)),
        rate_limit=int(config["kernel_eval_rate_limit"]),
    )
    return {"env_state": result, "reward_extra_info": result}
