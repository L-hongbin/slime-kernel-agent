"""Kernel-agent fully async rollout.

This keeps the official fully-async worker model, but restores the rollout-level
dynamic filtering path used by the regular sglang rollout.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import logging
import math
import queue
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from typing import Any

from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.kernel_reward import annotate_group_difficulty

from slime.observability.metric_utils import compute_rollout_step
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import (
    GenerateState,
    _split_turns_as_sample_groups,
    generate_and_rm,
    generate_and_rm_group,
)
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


def _is_verify_trajectory_group(group: RolloutGroup) -> bool:
    return bool(group) and all(
        isinstance(sample.metadata, dict)
        and (sample.metadata.get("role") == "verify" or sample.metadata.get("verify_trajectory", False))
        for sample in group
    )


def _get_group_concurrency(args, client_concurrency: int) -> int:
    n_samples_per_prompt = max(1, int(getattr(args, "n_samples_per_prompt", 1) or 1))
    if getattr(args, "verify_advantage_baseline", "group") == "anchor":
        n_samples_per_prompt += 1
    client_concurrency = max(1, int(client_concurrency))
    return max(1, client_concurrency // n_samples_per_prompt)


def _get_global_worker(args, data_buffer, rollout_id: int) -> KernelAgentAsyncRolloutWorker:
    global _config_logged, _global_worker
    with _worker_lock:
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

    async def _generate_group(self, group: RolloutGroup, sampling_params: dict[str, Any]) -> RolloutTaskResult:
        if not (group[0].metadata or {}).get("verify_anchor_key"):
            return await generate_and_rm_group(self.args, group, sampling_params, evaluation=False)

        key = self.data_buffer.prepare_anchor(group)
        anchor = self.data_buffer.claim_anchor(key)
        if anchor is None:
            raise ValueError("shared anchor has already been claimed")
        anchor.metadata["verify_anchor_key"] = key
        for name in ("gen_weight_version", "rollout_step", "gen_submit_time"):
            if name in group[0].metadata:
                anchor.metadata[name] = group[0].metadata[name]
        tasks = []
        try:
            # Anchor has its own index/seed and consumes no slot in the N candidate seed array.
            for idx, sample in enumerate([anchor, *group]):
                if sample.session_id is None:
                    sample.session_id = str(uuid.uuid4())
                params = sampling_params.copy()
                if getattr(self.args, "sglang_enable_deterministic_inference", False):
                    params["sampling_seed"] = (
                        self.args.rollout_seed + len(group) if idx == 0 else self.state.group_sampling_seeds[idx - 1]
                    )
                tasks.append(asyncio.create_task(generate_and_rm(self.args, sample, params, evaluation=False)))
            anchor_task, *candidate_tasks = tasks
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    outputs = task.result()
                    if not isinstance(outputs, list) or not outputs or _has_aborted_sample(outputs):
                        raise ValueError("shared-anchor group rollout did not finish")
                    if task is anchor_task:
                        if len(outputs) != 1 or outputs[0].remove_sample:
                            raise ValueError("shared anchor requires one valid evaluated kernel")
                        turn = outputs[0]
                        reward = float(turn.reward)
                        if not math.isfinite(reward):
                            raise ValueError("shared anchor reward must be finite")
                        self.data_buffer.complete_anchor(
                            key,
                            {
                                "response": turn.response,
                                "reward": reward,
                                "env_result": copy.deepcopy(turn.metadata.get("env_result")),
                                "weight_versions": list(turn.weight_versions),
                            },
                        )
                if all(task.done() for task in candidate_tasks):
                    trajectories = [task.result() for task in candidate_tasks]
                    if not any(
                        turn.metadata.get("role") == "verify" and not turn.remove_sample
                        for trajectory in trajectories
                        for turn in trajectory
                    ):
                        return _split_turns_as_sample_groups(trajectories)

            trajectories = [task.result() for task in candidate_tasks]
            for trajectory in trajectories:
                # Reads are non-consuming: every candidate and every turn uses the same fixed baseline.
                baseline = self.data_buffer.get_anchor_result(key)
                by_turn = {turn.metadata["turn_idx"]: turn for turn in trajectory}
                for verify in trajectory:
                    if verify.metadata.get("role") != "verify":
                        continue
                    verify.metadata.update(
                        verify_anchor_key=key,
                        verify_anchor_rollout=[copy.deepcopy(baseline)],
                        verify_anchor_reward=baseline["reward"],
                        verify_reward_mode="anchor",
                        verify_anchor_baseline="fixed_source",
                    )
                    if verify.remove_sample:
                        continue
                    kernel = by_turn[verify.metadata["turn_idx"] + 1]
                    versions = set(verify.metadata["verify_scoring_weight_versions"])
                    versions.update(str(version) for version in baseline["weight_versions"])
                    verify.metadata.update(
                        verify_scoring_weight_versions=sorted(versions),
                        verify_scoring_versions_complete=bool(
                            verify.metadata["verify_scoring_versions_complete"] and baseline["weight_versions"]
                        ),
                        verify_scoring_version_mismatch=len(versions) > 1,
                    )
                    if len(versions) > 1:
                        for turn in (verify, kernel):
                            turn.remove_sample = True
                            turn.loss_mask = [0] * turn.response_length
                            turn.reward = 0.0
                            turn.metadata.update(
                                multi_turn_reward=0.0, remove_reason="verify_scoring_version_mismatch"
                            )
                    else:
                        verify.reward = verify.metadata["verify_kernel_reward"] - baseline["reward"]
                        verify.metadata["multi_turn_reward"] = verify.reward
            return _split_turns_as_sample_groups(trajectories)
        except Exception:
            logger.exception("Shared anchor group %s failed; returning aborted candidates for retry", key)
            from examples.kernel_agent.generate_with_cuda_agent import _abort_result

            abort_args = copy.copy(self.args)
            abort_args.max_turns = 1
            abort_args.use_multi_turn = True
            abort_args.padding_turns = False
            return _split_turns_as_sample_groups(
                [_abort_result(abort_args, sample, "shared_anchor_group_failed", 0.0) for sample in group]
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self.data_buffer.release_anchor(key)

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

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=15)
            if self.worker_thread.is_alive():
                logger.warning("kernel-agent fully-async: worker thread did not stop within timeout")

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
            "exception_groups": getattr(self, "exception_count", 0),
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
                        gid = gid_counter
                        gid_counter += 1
                        self._stamp_group_for_submission(group)
                        original_group = copy.deepcopy(group)
                        task = asyncio.create_task(
                            self._generate_group(
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                            )
                        )
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
            self._reconcile_engine_weight_versions(result)
            self.output_queue.put_nowait((gid, result))
            self.completed_count += 1

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset

    verify_capture_enabled = bool(getattr(args, "capture_verify_data", False))
    if verify_capture_enabled and not callable(getattr(data_buffer, "add_verify_candidates", None)):
        raise TypeError(
            "Verify capture requires a data source that implements add_verify_candidates; "
            "set --data-source-path to examples.kernel_agent.kernel_agent_data_source.KernelAgentDataSource"
        )
    if getattr(args, "save_verify_data", None) is not None and not callable(
        getattr(data_buffer, "save_captured_verify_data", None)
    ):
        raise TypeError("--save-verify-data requires KernelAgentDataSource.save_captured_verify_data")
    if getattr(args, "save_verify_data", None) is not None:
        begin_verify_capture = getattr(data_buffer, "begin_verify_capture", None)
        if not callable(begin_verify_capture):
            raise TypeError("--save-verify-data requires KernelAgentDataSource.begin_verify_capture")
        begin_verify_capture(rollout_id)

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
    started = time.time()
    last_log = started
    log_every = 30.0
    log_sample_bodies = not bool(CUDA_AGENT_CONFIGS.get("log_rollout_stats_only", True))
    do_print = log_sample_bodies
    drop_reason_counts: Counter[str] = Counter()
    examined_task_groups = 0
    verify_candidates_added = 0

    def _record_dynamic_filter_drop(reason: str | None, count: int = 1) -> None:
        metric_gatherer.on_dynamic_filter_drop(reason=reason)
        drop_reason_counts[reason or "unknown"] += count

    while len(collected) < target:
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
            for group in groups:
                annotate_group_difficulty(group)
                if verify_capture_enabled:
                    verify_candidates_added += data_buffer.add_verify_candidates(group, rollout_id=rollout_id)

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
                if _is_verify_trajectory_group(last_turn_group):
                    collected[gid] = groups
                    continue
                dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, last_turn_group)
                if dynamic_filter_output.keep:
                    collected[gid] = groups
                else:
                    _record_dynamic_filter_drop(dynamic_filter_output.reason)
            else:
                for group in groups:
                    if _is_verify_trajectory_group(group):
                        collected.setdefault(gid, []).append(group)
                        continue
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
        group for _gid, groups in sorted(collected.items(), key=lambda item: _sort_key(item[1])) for group in groups
    ]
    verify_candidates_saved = (
        data_buffer.save_captured_verify_data(rollout_id) if getattr(args, "save_verify_data", None) is not None else 0
    )
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
    metrics["verify_candidates_added"] = verify_candidates_added
    metrics["verify_candidates_saved"] = verify_candidates_saved
    if getattr(args, "log_exp_metrics", False):
        worker_stats_end = worker.stats()
        metrics["exp/rollout/async/collect_time_seconds"] = collect_time
        metrics["exp/rollout/async/active_groups"] = worker_stats_end["active_groups"]
        metrics["exp/rollout/async/queued_groups"] = worker_stats_end["queued_groups"]
        for key in ("submitted_groups", "completed_groups", "aborted_groups", "exception_groups"):
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
