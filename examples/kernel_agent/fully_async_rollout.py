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
from typing import Any

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


def _get_global_worker(args, data_buffer) -> KernelAgentAsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
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
            _global_worker.start()
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
        self.output_queue: queue.Queue[tuple[int, RolloutTaskResult]] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        self.active_count = 0
        self.submitted_count = 0
        self.completed_count = 0
        self.aborted_count = 0
        self.exception_count = 0

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self._thread_main,
                name="kernel-agent-fully-async-rollout",
                daemon=True,
            )
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=15)
            if self.worker_thread.is_alive():
                logger.warning("kernel-agent fully-async: worker thread did not stop within timeout")

    def get_completed_groups(self) -> list[tuple[int, RolloutTaskResult]]:
        completed: list[tuple[int, RolloutTaskResult]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def stats(self) -> dict[str, int]:
        return {
            "active_groups": self.active_count,
            "submitted_groups": self.submitted_count,
            "completed_groups": self.completed_count,
            "aborted_groups": self.aborted_count,
            "exception_groups": self.exception_count,
            "queued_groups": self.queue_size(),
        }

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        active_tasks: set[asyncio.Task] = set()
        gid_counter = 0

        while self.running:
            try:
                if active_tasks:
                    done = {task for task in active_tasks if task.done()}
                    for task in done:
                        try:
                            task.result()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("kernel-agent fully-async task crashed: %r", exc)
                    active_tasks -= done
                    self.active_count = len(active_tasks)

                while len(active_tasks) < self.concurrency and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        original_group = copy.deepcopy(group)
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid, original_group))
                        active_tasks.add(task)
                        self.submitted_count += 1
                        self.active_count = len(active_tasks)

                self.active_count = len(active_tasks)
                await asyncio.sleep(1)
            except Exception as exc:  # noqa: BLE001
                logger.exception("kernel-agent fully-async loop iteration error: %s", exc)
                await asyncio.sleep(1)

        if active_tasks:
            logger.info("kernel-agent fully-async: waiting for %d in-flight tasks to drain", len(active_tasks))
            done, pending = await asyncio.wait(active_tasks, timeout=10)
            if pending:
                logger.warning(
                    "kernel-agent fully-async: cancelling %d in-flight tasks after drain timeout", len(pending)
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                try:
                    task.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("kernel-agent fully-async: in-flight task finished with error during stop: %r", exc)
            self.active_count = 0

    def _make_done_cb(self, gid: int, original_group: RolloutGroup):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except asyncio.CancelledError:
                if self.running:
                    self.exception_count += 1
                    logger.warning("kernel-agent fully-async: process task was cancelled while running")
                else:
                    logger.info("kernel-agent fully-async: process task cancelled during shutdown")
                return
            except Exception:  # noqa: BLE001
                self.exception_count += 1
                logger.exception("kernel-agent fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "kernel-agent fully-async: generate_and_rm_group returned %r, expected list",
                    type(result).__name__,
                )
                return
            if _has_aborted_sample(result):
                self.aborted_count += 1
                try:
                    self.data_buffer.add_samples([copy.deepcopy(original_group)])
                except Exception:  # noqa: BLE001
                    logger.exception("kernel-agent fully-async: failed to requeue aborted input group")
                return
            self.output_queue.put((gid, result))
            self.completed_count += 1

        return _cb


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

    worker = _get_global_worker(args, data_buffer)
    target = args.rollout_batch_size
    logger.info(
        "kernel-agent fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[RolloutGroup]] = {}
    started = time.time()
    last_log = started
    log_every = 30.0
    do_print = True
    drop_reason_counts: Counter[str] = Counter()

    def _record_dynamic_filter_drop(reason: str | None, count: int = 1) -> None:
        metric_gatherer.on_dynamic_filter_drop(reason=reason)
        drop_reason_counts[reason or "unknown"] += count

    while len(collected) < target:
        drained = 0
        for gid, task_group in worker.get_completed_groups():
            drained += 1
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
                dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, _get_last_non_pad_turn_group(groups))
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

        now = time.time()
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

    collect_time = time.time() - started
    data = [
        group
        for _gid, groups in sorted(collected.items(), key=lambda item: _sort_key(item[1]))[:target]
        for group in groups
    ]
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
    metrics = metric_gatherer.collect()
    metrics["fully_async_collect_time"] = collect_time
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    if evaluation:
        raise ValueError("kernel-agent fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
