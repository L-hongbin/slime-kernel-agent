"""Async KernelGym HTTP client and reward helpers for DrKernel."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import aiohttp

from .extract import CUDA_KERNEL_BACKEND, MISSING_COMPLETE_SUBMISSION_ERROR, extract_kernel_submission

if TYPE_CHECKING:
    from slime.utils.types import Sample

logger = logging.getLogger(__name__)
_TASK_COUNTER = count(1)
KERNELGYM_WORKFLOW = "kernelbench"
KERNELGYM_USE_REFERENCE_CACHE = True
KERNELGYM_CLIENT_TIMEOUT_S = 1800.0
KERNELGYM_TASK_TIMEOUT_S = 90
KERNELGYM_VERBOSE_ERRORS = True
KERNELGYM_ENABLE_PROFILING = True
KERNELGYM_DETECT_DECOY_KERNEL = True
KERNELGYM_NUM_CORRECT_TRIALS = 5
KERNELGYM_NUM_PERF_TRIALS = 50
KERNELGYM_NUM_WARMUP = 30
KERNELGYM_PERF_TRIM_COUNT = 5
KERNELGYM_REFERENCE_BACKEND = "pytorch"

__all__ = [
    "KernelGymClient",
    "KernelGymError",
    "KernelGymRequestError",
    "build_evaluation_request",
    "custom_rm",
    "evaluate_sample",
    "kernelgym_result_to_reward",
]


class KernelGymError(RuntimeError):
    """Base KernelGym client error."""


class KernelGymRequestError(KernelGymError):
    """Raised when KernelGym returns a non-2xx response or invalid payload."""


@dataclass(frozen=True)
class KernelGymClient:
    """Small async client for KernelGym's HTTP API."""

    base_url: str
    timeout_s: float = KERNELGYM_CLIENT_TIMEOUT_S
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    session: aiohttp.ClientSession | None = None
    _owns_session: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        base_url = self.base_url.rstrip("/")
        if not base_url:
            raise ValueError("KernelGym base_url must not be empty")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "_owns_session", self.session is None)

    @classmethod
    async def create(cls, base_url: str, **kwargs: Any) -> KernelGymClient:
        """Create a client and verify the KernelGym `/health` endpoint is reachable."""

        client = cls(base_url, **kwargs)
        await client._check_health()
        return client

    async def _check_health(self) -> None:
        try:
            payload = await self._request_json(
                "GET",
                "/health",
                timeout_s=5,
                max_retries=3,
            )
        except KernelGymRequestError as exc:
            raise KernelGymRequestError(f"KernelGym health check failed for {self.base_url}: {exc}") from exc

        status = payload.get("status")
        if status != "healthy":
            raise KernelGymRequestError(f"KernelGym at {self.base_url} is not healthy (status={status!r}): {payload}")

    async def __aenter__(self) -> KernelGymClient:
        await self._check_health()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _get_session(self) -> aiohttp.ClientSession:
        session = self.session
        if session is not None:
            return session

        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        connector = aiohttp.TCPConnector(limit=64, enable_cleanup_closed=True)
        session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        object.__setattr__(self, "session", session)
        return session

    async def close(self) -> None:
        session = self.session
        if session is not None and self._owns_session and not session.closed:
            await session.close()

    async def health(self) -> dict[str, Any]:
        return await self._request_json("GET", "/health")

    async def workers_status(self) -> dict[str, Any]:
        return await self._request_json("GET", "/workers/status")

    async def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._request_json("POST", "/evaluate", json=request)

    async def evaluate_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        if not requests:
            raise ValueError("KernelGym batch requests must not be empty")
        payload = {
            "batch_id": batch_id or _make_stable_id("drkernel_batch", str(len(requests))),
            "tasks": requests,
        }
        return await self._request_json("POST", "/evaluate/batch", json=payload)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        session = self._get_session()
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        request_timeout = aiohttp.ClientTimeout(total=timeout_s or self.timeout_s)
        retry_limit = self.max_retries if max_retries is None else max_retries

        for attempt in range(retry_limit + 1):
            try:
                async with session.request(method, url, json=json, timeout=request_timeout) as response:
                    text = await response.text()
                    if response.status >= 400:
                        raise KernelGymRequestError(
                            f"KernelGym {method} {path} failed with status {response.status}: {text[:1000]}"
                        )
                    try:
                        payload = await response.json()
                    except Exception as exc:
                        raise KernelGymRequestError(
                            f"KernelGym {method} {path} returned non-JSON payload: {text[:1000]}"
                        ) from exc
                    if not isinstance(payload, dict):
                        raise KernelGymRequestError(
                            f"KernelGym {method} {path} returned {type(payload).__name__}, expected object"
                        )
                    return payload
            except (aiohttp.ClientError, asyncio.TimeoutError, KernelGymRequestError) as exc:
                last_error = exc
                if attempt >= retry_limit:
                    break
                delay = self.retry_base_delay_s * (2**attempt)
                logger.info(
                    "KernelGym %s %s failed with %s; retrying in %.1fs (%d/%d)",
                    method,
                    path,
                    type(exc).__name__,
                    delay,
                    attempt + 1,
                    retry_limit,
                )
                await asyncio.sleep(delay)

        raise KernelGymRequestError(f"KernelGym {method} {path} failed after retries: {last_error}") from last_error


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _get_rm_url(args: Any) -> str:
    url = _get_arg(args, "rm_url")
    if not url:
        raise ValueError("KernelGym URL is required; set slime --rm-url")
    return str(url)


def _get_reference_code(sample: Sample) -> str:
    value = sample.label
    if value is None:
        raise KeyError("missing reference code; set --label-key ground_truth")
    if not isinstance(value, str):
        raise TypeError("reference code must be a string")
    return value


def _get_uuid(sample: Sample) -> str:
    # Match DrKernel rollout behavior for KernelBench eval data: converted
    # training data has uuid, while validation parquet commonly only has
    # problem_id/name.
    value = sample.metadata.get("uuid") or sample.metadata.get("problem_id") or sample.metadata.get("name")
    if value is None:
        raise KeyError("missing uuid; expected sample.metadata['uuid'], ['problem_id'], or ['name']")
    return str(value)


def _get_entry_point(sample: Sample) -> str:
    return str(sample.metadata.get("entry_point") or "Model")


def _make_stable_id(prefix: str, value: Any) -> str:
    text = str(value)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _get_task_id() -> str:
    return f"parallel_task_{next(_TASK_COUNTER):06d}_{uuid4().hex[:8]}"


def _metadata_or_arg(metadata: dict[str, Any], args: Any, metadata_key: str, arg_key: str, default: Any = None) -> Any:
    if metadata_key in metadata and metadata[metadata_key] is not None:
        return metadata[metadata_key]
    return _get_arg(args, arg_key, default)


def _get_int_arg(args: Any, name: str, default: int) -> int:
    return int(_get_arg(args, name, default))


def _kernelgym_tuning_fields(args: Any) -> dict[str, Any]:
    return {
        "num_correct_trials": _get_int_arg(args, "kernelgym_num_correct_trials", KERNELGYM_NUM_CORRECT_TRIALS),
        "num_perf_trials": _get_int_arg(args, "kernelgym_num_perf_trials", KERNELGYM_NUM_PERF_TRIALS),
        "num_warmup": _get_int_arg(args, "kernelgym_num_warmup", KERNELGYM_NUM_WARMUP),
        "perf_trim_count": _get_int_arg(args, "kernelgym_perf_trim_count", KERNELGYM_PERF_TRIM_COUNT),
        "reference_backend": _get_arg(args, "kernelgym_reference_backend", KERNELGYM_REFERENCE_BACKEND),
    }


def build_evaluation_request(
    args: Any,
    sample: Sample,
    *,
    kernel_code: str,
) -> dict[str, Any]:
    """Build a KernelGym EvaluationRequest-compatible CUDA-Agent payload."""

    metadata = sample.metadata
    reference_code = _get_reference_code(sample)
    entry_point = _get_entry_point(sample)
    uuid = _get_uuid(sample)

    request = {
        "task_id": _get_task_id(),
        "reference_code": reference_code,
        "kernel_code": kernel_code,
        "backend": CUDA_KERNEL_BACKEND,
        "entry_point": entry_point,
        "workflow": KERNELGYM_WORKFLOW,
        "use_reference_cache": KERNELGYM_USE_REFERENCE_CACHE,
        # DrKernel reward_model.task_timeout equivalent; KernelGym names the
        # request field `timeout`.
        "timeout": KERNELGYM_TASK_TIMEOUT_S,
        # KernelGym's API field is verbose_errors; this is the DrKernel
        # "verbose" behavior for returning detailed failure information.
        "verbose_errors": KERNELGYM_VERBOSE_ERRORS,
        "enable_profiling": KERNELGYM_ENABLE_PROFILING,
        "detect_decoy_kernel": KERNELGYM_DETECT_DECOY_KERNEL,
        # `is_valid` selects the KernelGym reference-runtime cache namespace:
        # False -> train cache, True -> validation cache. Train rollout uses
        # False; eval rollout can set this to True later.
        "is_valid": _metadata_or_arg(metadata, args, "is_valid", "kernelgym_is_valid", False),
        "uuid": uuid,
    }
    request.update(_kernelgym_tuning_fields(args))
    return request


def kernelgym_result_to_reward(result: dict[str, Any]) -> float:
    """Map a KernelGym evaluation result to the first DrKernel scalar reward."""

    compiled = bool(result.get("compiled"))
    correctness = bool(result.get("correctness"))
    decoy_kernel = bool(result.get("decoy_kernel")) if result.get("decoy_kernel") is not None else False
    return 1.0 if compiled and correctness and not decoy_kernel else 0.0


# Process-wide shared client. Created lazily on the first RM call so ``/health``
# is probed exactly once per process instead of once per sample/turn.
#
# INVARIANT: the RM runs inside a single Ray actor (``RolloutManager``) on one
# persistent asyncio loop, so the cached client and its aiohttp session always run
# on the loop they were created on. Full loop-swap support is intentionally NOT
# provided: both the cached session AND ``_shared_client_lock`` are bound to the
# loop they were first used on, and the old session can't be closed from a new loop
# (its loop is gone). The loop key only lets us notice a swap; a rebuild would still
# leak the old session and may even fail acquiring the stale lock. All of this is
# unreachable under the single-loop invariant above.
_shared_client: KernelGymClient | None = None
_shared_client_loop: asyncio.AbstractEventLoop | None = None
_shared_client_lock = asyncio.Lock()


async def _get_shared_client(args: Any) -> KernelGymClient:
    """Return the process-wide :class:`KernelGymClient`, probing ``/health`` once.

    The client (and its connection pool) is reused across every RM call in this
    process; the one-time health check fails fast with a clear error if KernelGym
    is unreachable on first use. The shared client is intentionally never closed —
    it lives for the lifetime of the process.
    """

    global _shared_client, _shared_client_loop
    loop = asyncio.get_running_loop()
    if _shared_client is not None and _shared_client_loop is loop:
        return _shared_client

    async with _shared_client_lock:
        # Re-check under the lock so concurrent first-callers build only one client.
        if _shared_client is not None and _shared_client_loop is loop:
            return _shared_client

        client = KernelGymClient(
            _get_rm_url(args),
            max_retries=int(_get_arg(args, "kernelgym_max_retries", 2)),
        )
        # ``_check_health`` opens the aiohttp session; if it fails, close that
        # session before propagating so a flapping server (whose error the
        # multi-turn loop catches per-sample) doesn't leak a connector each retry.
        try:
            await client._check_health()
        except BaseException:
            await client.close()
            raise
        _shared_client = client
        _shared_client_loop = loop
        logger.info("KernelGym shared client created and health-checked once for %s", client.base_url)
        return _shared_client


async def evaluate_sample(
    args: Any,
    sample: Sample,
    *,
    kernel_code: str | None = None,
    client: KernelGymClient | None = None,
) -> dict[str, Any]:
    """Extract/build/evaluate one sample and store review metadata on the sample."""

    metadata = sample.metadata

    if kernel_code is None:
        submission = extract_kernel_submission(sample.response)
        if submission is None:
            metadata["kernelgym"] = {
                "extract_error": MISSING_COMPLETE_SUBMISSION_ERROR,
                "reward": 0.0,
            }
            return {"reward": 0.0, "extract_error": MISSING_COMPLETE_SUBMISSION_ERROR}
        kernel_code = submission.code
        metadata["kernel_submission"] = {
            "backend": submission.backend,
            "sections": sorted(submission.sections),
        }

    request = build_evaluation_request(args, sample, kernel_code=kernel_code)

    # Reuse the process-wide shared client (health-checked once) unless a caller
    # passes its own; never close it here — it is owned by the process.
    if client is None:
        client = await _get_shared_client(args)

    result = await client.evaluate(request)

    reward = kernelgym_result_to_reward(result)
    metadata["kernelgym"] = {
        "request": request,
        "response": result,
        "reward": reward,
    }
    return {"reward": reward, "request": request, "response": result}


async def custom_rm(args: Any, sample_or_samples: Sample | list[Sample], **_: Any) -> float | list[float]:
    """Optional single-turn slime custom RM wrapper for KernelGym smoke tests."""

    samples = sample_or_samples if isinstance(sample_or_samples, list) else [sample_or_samples]
    client = await _get_shared_client(args)

    outputs = await asyncio.gather(*(evaluate_sample(args, sample, client=client) for sample in samples))

    rewards = [float(output["reward"]) for output in outputs]
    if isinstance(sample_or_samples, list):
        return rewards
    return rewards[0]
