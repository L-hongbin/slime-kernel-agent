"""Kernel-agent fully async rollout.

This keeps the official fully-async worker model, but restores the rollout-level
dynamic filtering path used by the regular sglang rollout.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import logging
import queue
import threading
import time
from collections import Counter
from collections.abc import Iterable
from itertools import islice
from typing import Any

from examples.kernel_agent.config import CUDA_AGENT_CONFIGS

from slime.observability.metric_utils import compute_rollout_step
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_sglang_client_concurrency
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger("examples.kernel_agent.fully_async_rollout")

_global_worker: KernelAgentAsyncRolloutWorker | None = None
_worker_lock = threading.Lock()
_config_logged = False


RolloutGroup = list[Sample]
RolloutTaskResult = list[Sample] | list[list[Sample]]


def _iter_samples(node: Any) -> Iterable[Sample]:
    if isinstance(node, Sample):
        yield node
        return
    if isinstance(node, list):
        for item in node:
            yield from _iter_samples(item)


def _has_aborted_sample(node: Any) -> bool:
    return any(sample.status == Sample.Status.ABORTED for sample in _iter_samples(node))


def _as_sample_groups(task_group: RolloutTaskResult) -> list[RolloutGroup]:
    if not task_group:
        return []
    if isinstance(task_group[0], list):
        groups = task_group
    else:
        groups = [task_group]
    for group in groups:
        assert group and all(
            isinstance(sample, Sample) for sample in group
        ), f"Rollout group must be list[Sample], got {type(group).__name__}"
    return groups


def _sort_key(task_group: RolloutTaskResult) -> int:
    for sample in _iter_samples(task_group):
        if sample.index is not None:
            return int(sample.index)
    return 0


def _get_last_non_pad_turn_group(groups: list[RolloutGroup]) -> RolloutGroup:
    last_turn_group: list[Sample | None] = [None] * len(groups[0])
    remaining = len(last_turn_group)

    for group in reversed(groups):
        for i, sample in enumerate(group):
            if last_turn_group[i] is not None:
                continue
            if isinstance(sample.metadata, dict) and sample.metadata.get("is_pad_turn"):
                continue
            last_turn_group[i] = sample
            remaining -= 1
        if remaining == 0:
            break

    return [sample if sample is not None else groups[-1][i] for i, sample in enumerate(last_turn_group)]


def _get_group_concurrency(args, client_concurrency: int) -> int:
    n_samples_per_prompt = max(1, int(getattr(args, "n_samples_per_prompt", 1) or 1))
    client_concurrency = max(1, int(client_concurrency))
    return max(1, client_concurrency // n_samples_per_prompt)


def _get_global_worker(args, data_buffer, rollout_id: int) -> KernelAgentAsyncRolloutWorker:
    global _config_logged, _global_worker
    with _worker_lock:
        if _global_worker is not None and getattr(_global_worker, "_failure_reason", None):
            raise RuntimeError(_global_worker._failure_reason)
        if not _config_logged:
            logger.info("CUDA_AGENT_CONFIGS=%s", CUDA_AGENT_CONFIGS)
            _config_logged = True
        if (
            _global_worker is None
            or _global_worker.worker_thread is None
            or not _global_worker.worker_thread.is_alive()
        ):
            logger.info("starting kernel-agent fully-async rollout worker")
            client_concurrency = get_sglang_client_concurrency(args)
            group_concurrency = _get_group_concurrency(args, client_concurrency)
            logger.info(
                "kernel-agent fully-async concurrency: client=%d, n_samples_per_prompt=%d, groups=%d",
                client_concurrency,
                max(1, int(getattr(args, "n_samples_per_prompt", 1) or 1)),
                group_concurrency,
            )
            _global_worker = KernelAgentAsyncRolloutWorker(
                args,
                data_buffer,
                concurrency=group_concurrency,
            )
            # Install the first context before the thread can consume a prompt.
            _global_worker.set_generation_context(rollout_id)
            _global_worker.start()
        else:
            _global_worker.set_generation_context(rollout_id)
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class KernelAgentAsyncRolloutWorker:
    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self._failure_reason: str | None = None
        self._cancel_on_stop = False
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self._inflight_lock = threading.Lock()
        self._inflight: dict[int, tuple[float, RolloutGroup]] = {}
        # The done callback runs on the event-loop thread, so put() must never
        # block. Backpressure is enforced in _loop before new prompts are read.
        self.output_queue: queue.Queue[tuple[int, RolloutTaskResult]] = queue.Queue()
        self.poll_interval = 1.0
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        self.active_count = 0
        self.submitted_count = 0
        self.completed_count = 0
        self.aborted_count = 0
        self.retried_count = 0
        self.retry_exhausted_count = 0
        self.rollout_max_retries = int(getattr(args, "rollout_max_retries", 10))
        if self.rollout_max_retries < 0:
            raise ValueError("rollout_max_retries must be >= 0")
        self.exception_count = 0
        # The collector thread advances this context at each rollout boundary,
        # while the background event loop snapshots it for each fresh request.
        # Keep rollout_step and generation weight version as an atomic pair.
        self._generation_context_lock = threading.Lock()
        self._rollout_step: int | None = None
        self._gen_weight_version: int | None = None

    def set_generation_context(self, rollout_id: int) -> None:
        rollout_step = compute_rollout_step(self.args, rollout_id)
        gen_weight_version = getattr(self.args, "gen_weight_version", None)
        with self._generation_context_lock:
            self._rollout_step = rollout_step
            self._gen_weight_version = gen_weight_version

    def _stamp_group_for_submission(self, group: RolloutGroup) -> None:
        with self._generation_context_lock:
            rollout_step = self._rollout_step
            gen_weight_version = self._gen_weight_version
        if rollout_step is None:
            raise RuntimeError("kernel-agent fully-async generation context was not initialized")

        for sample in group:
            metadata = dict(sample.metadata or {})
            metadata["rollout_step"] = rollout_step
            if gen_weight_version is None:
                metadata.pop("gen_weight_version", None)
            else:
                # An aborted fully-async request is regenerated from scratch,
                # so a requeued prompt must be restamped for the new attempt.
                metadata["gen_weight_version"] = gen_weight_version
            if getattr(self.args, "log_exp_metrics", False):
                metadata["gen_submit_time"] = time.time()
            sample.metadata = metadata

    @staticmethod
    def _reconcile_engine_weight_versions(task_group: RolloutTaskResult) -> None:
        """Prefer the engine-reported version over the submission snapshot.

        Weight updates pause the engines, but the persistent worker can already
        have a request waiting when generation resumes. SGLang's response-side
        weight_version is therefore the authoritative version for completed
        output; the submission stamp remains a fallback for custom generators
        that do not preserve that field.
        """
        for sample in _iter_samples(task_group):
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["engine_weight_version_span"] = False
            sample.metadata["engine_weight_version_mismatch"] = False
            if not sample.weight_versions:
                continue
            try:
                versions = {int(version) for version in sample.weight_versions}
            except (TypeError, ValueError):
                continue
            if len(versions) != 1:
                sample.metadata["engine_weight_version_span"] = True
                logger.warning(
                    "kernel-agent fully-async sample %s spans engine weight versions %s; " "keeping submission stamp",
                    sample.index,
                    sorted(versions),
                )
                continue
            engine_version = versions.pop()
            submitted_version = sample.metadata.get("gen_weight_version")
            sample.metadata["engine_weight_version_mismatch"] = (
                submitted_version is not None and int(submitted_version) != engine_version
            )
            sample.metadata["gen_weight_version"] = engine_version

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self._thread_main,
                name="kernel-agent-fully-async-rollout",
                daemon=True,
            )
            self.worker_thread.start()

    def stop(self, *, cancel: bool = False) -> bool:
        self.running = False
        if cancel:
            self._cancel_on_stop = True
            loop = self._event_loop
            if loop is not None and not loop.is_closed():

                def cancel_active():
                    for task in self._active_tasks:
                        if not task.done() and not task.cancelling():
                            task.cancel()

                try:
                    loop.call_soon_threadsafe(cancel_active)
                except RuntimeError:
                    pass  # loop finished between is_closed() and scheduling
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=15)
            if self.worker_thread.is_alive():
                logger.warning("kernel-agent fully-async: worker thread did not stop within timeout")
                return False
        return True

    def get_completed_groups(self, limit: int | None = None) -> list[tuple[int, RolloutTaskResult]]:
        """Pop at most ``limit`` completed prompt groups, or all when unset."""
        completed: list[tuple[int, RolloutTaskResult]] = []
        while limit is None or len(completed) < limit:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def stats(self) -> dict[str, int]:
        return {
            "active_groups": getattr(self, "active_count", 0),
            "submitted_groups": getattr(self, "submitted_count", 0),
            "completed_groups": getattr(self, "completed_count", 0),
            "aborted_groups": getattr(self, "aborted_count", 0),
            "retried_groups": getattr(self, "retried_count", 0),
            "retry_exhausted_groups": getattr(self, "retry_exhausted_count", 0),
            "exception_groups": getattr(self, "exception_count", 0),
            "queued_groups": self.queue_size(),
        }

    def snapshot(self, limit: int = 8) -> dict[str, Any]:
        """Bounded local metadata only: never query Ray/HTTP or copy sample bodies."""
        now = time.monotonic()

        def summarize(gid, group):
            return {
                "gid": gid,
                "samples": [
                    {
                        "index": sample.index,
                        "group_index": sample.group_index,
                        "role": (sample.metadata or {}).get("role", "kernel"),
                        "task_id": (sample.metadata or {}).get("task_id"),
                        "version": (sample.metadata or {}).get("gen_weight_version"),
                    }
                    for sample in islice(_iter_samples(group), limit)
                ],
            }

        with self._inflight_lock:
            active = [
                {**summarize(gid, group), "age_seconds": max(0.0, now - started)}
                for gid, (started, group) in islice(self._inflight.items(), limit)
            ]
        with self.output_queue.mutex:
            queued = [summarize(gid, group) for gid, group in islice(self.output_queue.queue, limit)]
        return {
            **self.stats(),
            "running": self.running,
            "worker_thread_alive": self.worker_thread is not None and self.worker_thread.is_alive(),
            "active_head": active,
            "queued_head": queued,
        }

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        active_tasks = self._active_tasks
        gid_counter = 0

        while self.running:
            try:
                if active_tasks:
                    done = {task for task in active_tasks if task.done()}
                    for task in done:
                        try:
                            task.result()
                        except asyncio.CancelledError:
                            if self.running:
                                self.exception_count += 1
                                logger.warning("kernel-agent fully-async task was cancelled while running")
                            else:
                                logger.info("kernel-agent fully-async task cancelled during shutdown")
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("kernel-agent fully-async task crashed: %r", exc)
                    active_tasks -= done
                    self.active_count = len(active_tasks)

                # Keep at most one completed pool queued. This preserves a warm
                # queue without consuming the whole data buffer while training
                # is slower than rollout generation.
                while (
                    len(active_tasks) < self.concurrency
                    and self.output_queue.qsize() < self.concurrency
                    and self.running
                ):
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        if not self.running:
                            break
                        gid = gid_counter
                        gid_counter += 1
                        self._stamp_group_for_submission(group)
                        original_group = copy.deepcopy(group)
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        with self._inflight_lock:
                            self._inflight[gid] = (time.monotonic(), group)
                        task.add_done_callback(self._make_done_cb(gid, original_group))
                        active_tasks.add(task)
                        self.submitted_count += 1
                        self.active_count = len(active_tasks)

                self.active_count = len(active_tasks)
                await asyncio.sleep(self.poll_interval)
            except Exception as exc:  # noqa: BLE001
                logger.exception("kernel-agent fully-async loop iteration error: %s", exc)
                await asyncio.sleep(self.poll_interval)

        if active_tasks:
            logger.info("kernel-agent fully-async: waiting for %d in-flight tasks to drain", len(active_tasks))
            done, pending = await asyncio.wait(active_tasks, timeout=0 if self._cancel_on_stop else 10)
            if pending:
                logger.warning(
                    "kernel-agent fully-async: cancelling %d in-flight tasks after drain timeout", len(pending)
                )
                for task in pending:
                    if not task.cancelling():
                        task.cancel()
                finished, pending = await asyncio.wait(pending, timeout=10)
                done |= finished
                if pending:
                    logger.error(
                        "kernel-agent fully-async: %d tasks did not finish cancellation within 10s", len(pending)
                    )
            for task in done:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001
                    logger.warning("kernel-agent fully-async: in-flight task finished with error during stop: %r", exc)
            active_tasks.difference_update(done)
            self.active_count = len(active_tasks)

    def _make_done_cb(self, gid: int, original_group: RolloutGroup):
        def _cb(done_task: asyncio.Task) -> None:
            with self._inflight_lock:
                self._inflight.pop(gid, None)
            try:
                result = done_task.result()
            except asyncio.CancelledError:
                if self.running:
                    self.exception_count += 1
                    logger.warning("kernel-agent fully-async: process task was cancelled while running")
                    self._retry_failed_group(gid, original_group, "cancelled")
                else:
                    logger.info("kernel-agent fully-async: process task cancelled during shutdown")
                return
            except Exception as exc:  # noqa: BLE001
                self.exception_count += 1
                logger.exception("kernel-agent fully-async: process task raised")
                self._retry_failed_group(gid, original_group, f"exception:{type(exc).__name__}")
                return
            if not isinstance(result, list) or not result:
                logger.warning(
                    "kernel-agent fully-async: generate_and_rm_group returned %r, expected list",
                    type(result).__name__,
                )
                self._retry_failed_group(gid, original_group, "invalid_group_result")
                return
            if _has_aborted_sample(result):
                self.aborted_count += 1
                self._retry_failed_group(gid, original_group, "aborted")
                return
            self._reconcile_engine_weight_versions(result)
            self.output_queue.put_nowait((gid, result))
            self.completed_count += 1

        return _cb

    def _retry_failed_group(self, gid: int, original_group: RolloutGroup, reason: str) -> None:
        if not self.running:
            return  # Shutdown cancellation must not manufacture new work.
        retries = max(int((sample.metadata or {}).get("kernel_agent_group_retry", 0)) for sample in original_group)
        if retries < self.rollout_max_retries:
            try:
                retry_group = copy.deepcopy(original_group)
                for sample in retry_group:
                    sample.metadata = {**(sample.metadata or {}), "kernel_agent_group_retry": retries + 1}
                self.data_buffer.add_samples([retry_group])
                self.retried_count += 1
                logger.warning(
                    "kernel-agent group_retry gid=%s retry=%s/%s reason=%s",
                    gid,
                    retries + 1,
                    self.rollout_max_retries,
                    reason,
                )
                return
            except Exception:
                logger.exception("kernel-agent fully-async: failed to requeue input group")
                reason = "retry_buffer_write_failed"
        self.retry_exhausted_count += 1
        self.running = False
        self._cancel_on_stop = True
        self._failure_reason = (
            f"Kernel-agent group {gid} cannot retry ({retries + 1} failed attempts, "
            f"max_retries={self.rollout_max_retries}, reason={reason}); submissions stopped. "
            "Inspect evaluation cancellation/infrastructure failures before restarting."
        )
        logger.error("%s snapshot=%s", self._failure_reason, self.snapshot())


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    use_multi_turn = getattr(args, "use_multi_turn", False)
    if use_multi_turn:
        max_turns = getattr(args, "max_turns", None)
        assert max_turns is not None, "--max-turns must be set when --use-multi-turn is enabled"
        assert int(max_turns) >= 1, "--max-turns must be >= 1"
    filter_by_last_turn = use_multi_turn and getattr(args, "filter_by_last_turn", False)

    worker = _get_global_worker(args, data_buffer, rollout_id)
    worker_stats_start = worker.stats()
    target = args.rollout_batch_size
    logger.info(
        "kernel-agent fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[RolloutGroup]] = {}
    started = time.monotonic()
    last_log = started
    last_progress = started
    last_warning = started
    warn_seconds = float(getattr(args, "rollout_no_progress_warn_seconds", 900.0))
    timeout_seconds = float(getattr(args, "rollout_no_progress_timeout_seconds", 7200.0))
    no_progress_warnings = 0
    max_no_progress_seconds = 0.0
    log_every = 30.0
    log_sample_bodies = not bool(CUDA_AGENT_CONFIGS.get("log_rollout_stats_only", True))
    do_print = log_sample_bodies
    drop_reason_counts: Counter[str] = Counter()
    examined_task_groups = 0

    def _record_dynamic_filter_drop(reason: str | None, count: int = 1) -> None:
        metric_gatherer.on_dynamic_filter_drop(reason=reason)
        drop_reason_counts[reason or "unknown"] += count

    while len(collected) < target:
        if getattr(worker, "_failure_reason", None):
            worker.stop(cancel=True)
            raise RuntimeError(worker._failure_reason)
        previous_count = len(collected)
        drained = 0
        # A dynamically filtered task may contribute nothing, so this loop can
        # drain again on the next iteration. It must never pop more accepted
        # prompt groups than the current rollout can consume: surplus completed
        # work stays warm for the next rollout.
        for gid, task_group in worker.get_completed_groups(limit=target - len(collected)):
            drained += 1
            examined_task_groups += 1
            groups = _as_sample_groups(task_group)
            if not groups:
                continue
            if do_print:
                sample = groups[0][0]
                logger.info(
                    "First kernel-agent fully-async rollout sample: %s, label: %s, reward: %s",
                    [str(sample.prompt) + sample.response],
                    str(sample.label)[:100],
                    sample.reward,
                )
                do_print = False

            if filter_by_last_turn:
                last_turn_group = _get_last_non_pad_turn_group(groups)
                dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, last_turn_group)
                if dynamic_filter_output.keep:
                    collected[gid] = groups
                else:
                    _record_dynamic_filter_drop(dynamic_filter_output.reason)
            else:
                for group in groups:
                    dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
                    if dynamic_filter_output.keep:
                        collected.setdefault(gid, []).append(group)
                    else:
                        _record_dynamic_filter_drop(dynamic_filter_output.reason)

        if not drained:
            await asyncio.sleep(0.05)

        now = time.monotonic()
        idle = now - last_progress
        max_no_progress_seconds = max(max_no_progress_seconds, idle)
        if len(collected) > previous_count:
            last_progress = last_warning = now
            idle = 0.0
        if len(collected) < target and (
            (timeout_seconds > 0 and idle >= timeout_seconds)
            or (warn_seconds > 0 and idle >= warn_seconds and now - last_warning >= warn_seconds)
        ):
            timed_out = timeout_seconds > 0 and idle >= timeout_seconds
            if timed_out:
                # Fence submissions before diagnostics or bounded shutdown. Do
                # not return an undersized batch or silently restart this worker.
                worker.running = False
                worker._failure_reason = (
                    f"Kernel-agent rollout {rollout_id} made no accepted-group progress for {idle:.1f}s "
                    f"(collected={len(collected)}/{target}, timeout={timeout_seconds}s). "
                    "Rollout submissions stopped; inspect the queue snapshot before restarting the rollout process."
                )
            snapshot = {
                "rollout_id": rollout_id,
                "accepted_groups": len(collected),
                "target_groups": target,
                "no_progress_seconds": idle,
                "examined_task_groups": examined_task_groups,
                "drop_reason_counts": dict(drop_reason_counts),
                "worker": worker.snapshot(),
            }
            if timed_out:
                logger.error("kernel-agent rollout no-progress timeout: snapshot=%s", snapshot)
                stopped = worker.stop(cancel=True)
                logger.error("kernel-agent rollout stalled: worker_shutdown_complete=%s", stopped)
                raise RuntimeError(worker._failure_reason)
            logger.warning("kernel-agent rollout no-progress warning: snapshot=%s", snapshot)
            last_warning = now
            no_progress_warnings += 1
        if now - last_log > log_every:
            logger.info(
                "kernel-agent fully-async rollout %d: collected %d/%d, worker=%s, elapsed=%.1fs, "
                "drop_count=%d, drop_reason_count=%s",
                rollout_id,
                len(collected),
                target,
                worker.stats(),
                now - started,
                sum(drop_reason_counts.values()),
                dict(drop_reason_counts),
            )
            last_log = now

    collect_time = time.monotonic() - started
    data = [
        group for _gid, groups in sorted(collected.items(), key=lambda item: _sort_key(item[1])) for group in groups
    ]
    if log_sample_bodies:
        sample = data[-1][0]
        logger.info(
            "kernel-agent fully-async rollout %d: done in %.1fs, queue_left=%d, %s, label: %s, reward: %s",
            rollout_id,
            collect_time,
            worker.queue_size(),
            [str(sample.prompt) + sample.response],
            str(sample.label)[:100],
            sample.reward,
        )
    else:
        logger.info(
            "kernel-agent fully-async rollout %d: done in %.1fs, queue_left=%d, accepted_groups=%d",
            rollout_id,
            collect_time,
            worker.queue_size(),
            len(data),
        )
    metrics = metric_gatherer.collect()
    metrics["fully_async_collect_time"] = collect_time
    metrics["fully_async_no_progress_warnings"] = no_progress_warnings
    metrics["fully_async_max_no_progress_seconds"] = max_no_progress_seconds
    if getattr(args, "log_exp_metrics", False):
        worker_stats_end = worker.stats()
        metrics["exp/rollout/async/collect_time_seconds"] = collect_time
        metrics["exp/rollout/async/active_groups"] = worker_stats_end["active_groups"]
        metrics["exp/rollout/async/queued_groups"] = worker_stats_end["queued_groups"]
        for key in (
            "submitted_groups",
            "completed_groups",
            "aborted_groups",
            "exception_groups",
            "retried_groups",
            "retry_exhausted_groups",
        ):
            metrics[f"exp/rollout/async/{key}_delta"] = worker_stats_end[key] - worker_stats_start[key]
        metrics["exp/rollout/async/accepted_groups"] = len(collected)
        metrics["exp/rollout/async/examined_task_groups"] = examined_task_groups
        metrics["exp/rollout/async/dropped_group_candidates"] = sum(drop_reason_counts.values())
        metrics["exp/rollout/async/acceptance_per_examined"] = (
            len(collected) / examined_task_groups if examined_task_groups > 0 else 0.0
        )
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    if evaluation:
        raise ValueError("kernel-agent fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
