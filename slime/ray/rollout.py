import dataclasses
import itertools
import logging
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.lora_utils import use_lora_weight_sync
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.misc import Box, group_by, load_function
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .rollout_validation import validate_server_group_gpu_indices
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

SGLANG_ENGINE_ENV_DEFAULTS = {
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "true",
    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "true",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
    "SLIME_ENABLE_PROFILING": "true",
}


def _sanitize_args_for_engine(args):
    """Deep-copy-free sanitized Namespace for SGLangEngine actors.

    The trainer-side argparse Namespace carries megatron-typed values (enums,
    dataclasses) that force `import megatron` when ray unpickles the actor
    args — rollout containers must not need megatron (the tar-copy of the
    fleet Megatron tree was a workaround for exactly this). Replace any
    attribute whose type lives in a megatron module with its repr string;
    the engine only consumes plain paths/ints/bools/sglang_* fields.
    """
    import argparse as _argparse

    def _is_megatron_obj(v):
        mod = getattr(type(v), "__module__", "") or ""
        return mod.split(".")[0] == "megatron"

    clean = _argparse.Namespace()
    for k, v in vars(args).items():
        if _is_megatron_obj(v):
            v = repr(v)
        elif isinstance(v, (list, tuple)) and any(_is_megatron_obj(x) for x in v):
            v = [repr(x) if _is_megatron_obj(x) else x for x in v]
        elif isinstance(v, dict) and any(_is_megatron_obj(x) for x in v.values()):
            v = {kk: (repr(x) if _is_megatron_obj(x) else x) for kk, x in v.items()}
        setattr(clean, k, v)
    return clean


@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", "decode", or "placeholder"
    rank_offset: int = 0  # cumulative engine count before this group
    gpu_offset: int = 0  # cumulative GPU count before this group
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False  # True when this group's GPUs overlap with megatron
    model_path: str | None = None  # checkpoint path for update_weights_from_disk
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[str, int] | None = None) -> tuple[list, dict[str, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps host IP → next free port.
        The caller should ``ray.get()`` on the handles to block until the
        engines are healthy, and pass *port_cursors* to the next server group
        so that different groups on the same node don't race for ports.

        Placeholder groups (worker_type="placeholder") skip engine creation entirely.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
        validate_server_group_gpu_indices(
            worker_type=self.worker_type,
            gpu_offset=self.gpu_offset,
            num_gpus_per_engine=self.num_gpus_per_engine,
            num_gpu_per_engine=num_gpu_per_engine,
            num_engines=len(self.all_engines),
            num_available_gpus=len(reordered_gpu_ids),
            rollout_num_gpus=self.args.rollout_num_gpus,
            rollout_num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
        )

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group using gpu_offset.
            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val) for key, default_val in SGLANG_ENGINE_ENV_DEFAULTS.items()
            }
            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                _sanitize_args_for_engine(self.args),
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        if self.args.rollout_external:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=rollout_engines
            )
        else:
            # Compute base_port from the maximum cursor across all nodes that
            # this group's engines may land on (conservative: just use global max).
            base_port = max(port_cursors.values()) if port_cursors else 15000
            addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
                args=self.args,
                rollout_engines=rollout_engines,
                worker_type=self.worker_type,
                num_gpus_per_engine=self.num_gpus_per_engine,
                rank_offset=self.rank_offset,
                base_port=base_port,
            )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors

    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def onload_weights_from_disk(self):
        """Reload weights from ``model_path`` for non-updatable groups.

        Used instead of ``resume_memory_occupation(tags=[WEIGHTS])`` so that
        CPU memory is not consumed by offloaded weight copies.
        """
        if not self.needs_offload or not self.model_path:
            return []
        return [
            engine.update_weights_from_disk.remote(self.model_path) for engine in self.engines if engine is not None
        ]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more server groups.

    Each RolloutServer represents one model deployed behind a single router.
    A server may contain multiple ServerGroups with different
    ``num_gpus_per_engine`` (e.g. prefill TP=2, decode TP=4).
    """

    server_groups: list[ServerGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def engines(self):
        """All node-0 engines across all groups (placeholder groups contribute nothing)."""
        return [e for g in self.server_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.server_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.server_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [g.num_gpus_per_engine for g in self.server_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        """Per-engine GPU offset for all node-0 engines, parallel to ``engines``.

        Accounts for placeholder groups that occupy GPU slots without creating engines.
        """
        offsets = []
        for g in self.server_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        """Nodes per engine.  Only valid when all active groups share the same value."""
        values = {g.nodes_per_engine for g in self.server_groups if g.worker_type != "placeholder"}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        # Record dead indices per group before starting.
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        # Start all groups concurrently.
        all_handles = []
        port_cursors: dict[str, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            # Resume GPU memory for all engines that need offload.
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
            if all_resume_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []

    def onload_weights(self):
        """Restore weights for offloaded groups.

        All groups resume from CPU cache via ``resume_memory_occupation``.
        For updatable servers, weights will be overwritten by
        ``update_weights`` shortly after.  For non-updatable servers the
        CPU backup already contains the correct (unchanged) weights.
        """
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        """Resume KV cache and CUDA graphs for offloaded groups."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))
        return ray.get(handles) if handles else []


_PREDICTIVE_SUPPORT_FIELDS = (
    "rollout_topk_token_ids",
    "rollout_topk_log_probs",
    "rollout_topk_valid_mask",
)


def _validate_predictive_support_sample(sample: Sample, top_k: int, sample_position: int) -> None:
    """Validate/normalize one Sample at the rollout -> Ray train boundary.

    This is deliberately strict for nonzero-loss tokens: silently training with
    a partial behavior support changes the predictive KL and its directional
    derivative.  Fully masked padding/abort samples may omit the fields and are
    normalized to compact all-invalid numpy rows.
    """
    response_length = int(sample.response_length)
    width = top_k + 1
    expected_shape = (response_length, width)
    loss_mask = sample.loss_mask
    if loss_mask is None or len(loss_mask) != response_length:
        raise ValueError(
            f"sample[{sample_position}] loss_mask must have response_length={response_length} entries "
            "before predictive-support validation"
        )

    values = [getattr(sample, field) for field in _PREDICTIVE_SUPPORT_FIELDS]
    if all(value is None for value in values):
        if any(loss_mask):
            raise ValueError(
                f"sample[{sample_position}] has trainable response tokens but predictive support is missing"
            )
        values = [
            np.zeros(expected_shape, dtype=np.int32),
            np.zeros(expected_shape, dtype=np.float32),
            np.zeros(expected_shape, dtype=np.bool_),
        ]
        for field, value in zip(_PREDICTIVE_SUPPORT_FIELDS, values, strict=True):
            setattr(sample, field, value)
    elif any(value is None for value in values):
        missing = [field for field, value in zip(_PREDICTIVE_SUPPORT_FIELDS, values, strict=True) if value is None]
        raise ValueError(f"sample[{sample_position}] has incomplete predictive support; missing {missing}")

    expected_dtypes = (np.dtype(np.int32), np.dtype(np.float32), np.dtype(np.bool_))
    for field, value, expected_dtype in zip(_PREDICTIVE_SUPPORT_FIELDS, values, expected_dtypes, strict=True):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"sample[{sample_position}].{field} must stay a numpy.ndarray through Ray, "
                f"got {type(value).__name__}"
            )
        if value.shape != expected_shape:
            raise ValueError(f"sample[{sample_position}].{field} shape must be {expected_shape}, got {value.shape}")
        if value.dtype != expected_dtype:
            raise TypeError(f"sample[{sample_position}].{field} dtype must be {expected_dtype}, got {value.dtype}")

    token_ids, behavior_log_probs, valid_mask = values
    if response_length and len(sample.tokens) < response_length:
        raise ValueError(
            f"sample[{sample_position}] has response_length={response_length} but only {len(sample.tokens)} tokens"
        )
    if not response_length:
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        elif len(sample.rollout_log_probs) != 0:
            raise ValueError(f"sample[{sample_position}] zero-length response has non-empty rollout_log_probs")
        return

    # Keep the 16k-token validation path entirely in NumPy.  A Python loop plus
    # np.unique per token would execute ~4.2M tiny loops for a 256-sample batch.
    def first_bad_index(bad_rows: np.ndarray) -> int:
        return int(np.flatnonzero(bad_rows)[0])

    nonfinite_rows = np.any(valid_mask & ~np.isfinite(behavior_log_probs), axis=1)
    if nonfinite_rows.any():
        token_index = first_bad_index(nonfinite_rows)
        raise ValueError(
            f"sample[{sample_position}] predictive support token {token_index} contains non-finite logprobs"
        )

    negative_id_rows = np.any(valid_mask & (token_ids < 0), axis=1)
    if negative_id_rows.any():
        token_index = first_bad_index(negative_id_rows)
        raise ValueError(
            f"sample[{sample_position}] predictive support token {token_index} contains a negative token id"
        )

    # Invalid slots become -1, which is outside the validated vocabulary-id
    # domain.  Sorting once per row makes a duplicate an adjacent equal pair;
    # repeated invalid -1 slots are explicitly excluded.
    sorted_ids = np.sort(np.where(valid_mask, token_ids, -1), axis=1)
    duplicate_rows = np.any((sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0), axis=1)
    if duplicate_rows.any():
        token_index = first_bad_index(duplicate_rows)
        raise ValueError(
            f"sample[{sample_position}] predictive support token {token_index} contains duplicate token ids"
        )

    trainable = np.asarray(loss_mask, dtype=np.bool_)
    valid_counts = valid_mask.sum(axis=1)
    bad_count_rows = trainable & (valid_counts != top_k) & (valid_counts != top_k + 1)
    if bad_count_rows.any():
        token_index = first_bad_index(bad_count_rows)
        raise ValueError(
            f"sample[{sample_position}] trainable token {token_index} has {valid_counts[token_index]} valid "
            f"support entries; expected {top_k} or {top_k + 1}"
        )

    # Padding/aborted/partial-off-policy tokens may intentionally carry an
    # all-invalid row.  The remaining invariants only apply to trainable rows.
    if not trainable.any():
        # The legacy transfer path keys off samples[0].rollout_log_probs.  Keep
        # even an all-masked first sample non-None so a later trainable sample
        # cannot cause the whole field to be omitted from train_data.
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = [0.0] * response_length
        elif len(sample.rollout_log_probs) != response_length:
            raise ValueError(
                f"sample[{sample_position}] masked rollout_log_probs must have "
                f"response_length={response_length} entries"
            )
        elif not np.isfinite(np.asarray(sample.rollout_log_probs, dtype=np.float64)).all():
            raise ValueError(f"sample[{sample_position}] masked rollout_log_probs contains non-finite values")
        return

    sampled_token_ids = np.asarray(sample.tokens[-response_length:], dtype=np.int64)
    sampled_matches = valid_mask & (token_ids.astype(np.int64, copy=False) == sampled_token_ids[:, None])
    sampled_match_counts = sampled_matches.sum(axis=1)
    bad_match_rows = trainable & (sampled_match_counts != 1)
    if bad_match_rows.any():
        token_index = first_bad_index(bad_match_rows)
        raise ValueError(
            f"sample[{sample_position}] trainable token {token_index} must contain sampled token id "
            f"{sampled_token_ids[token_index]} exactly once; found {sampled_match_counts[token_index]}"
        )

    sampled_log_probs = sample.rollout_log_probs
    if sampled_log_probs is None or len(sampled_log_probs) != response_length:
        raise ValueError(
            f"sample[{sample_position}] rollout_log_probs must have response_length={response_length} entries"
        )
    sampled_log_probs_array = np.asarray(sampled_log_probs, dtype=np.float64)
    nonfinite_sampled_rows = trainable & ~np.isfinite(sampled_log_probs_array)
    if nonfinite_sampled_rows.any():
        token_index = first_bad_index(nonfinite_sampled_rows)
        raise ValueError(f"sample[{sample_position}] sampled token {token_index} has non-finite rollout logprob")

    sampled_slots = np.argmax(sampled_matches, axis=1)
    support_sampled_log_probs = np.take_along_axis(behavior_log_probs, sampled_slots[:, None], axis=1)[:, 0]
    mismatch_rows = trainable & ~np.isclose(
        support_sampled_log_probs,
        sampled_log_probs_array,
        rtol=1e-5,
        atol=1e-6,
    )
    if mismatch_rows.any():
        token_index = first_bad_index(mismatch_rows)
        raise ValueError(
            f"sample[{sample_position}] sampled-token logprob mismatch at token {token_index}: "
            f"support={support_sampled_log_probs[token_index]}, "
            f"rollout_log_probs={sampled_log_probs_array[token_index]}"
        )


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        elif getattr(self.args, "use_multi_turn", False):
            logger.warning(
                "--use-multi-turn is enabled without --custom-reward-post-process-path. "
                "The default _post_process_rewards will include padded turns or remove_sample entries "
                "when computing reward mean/std, which may bias advantage normalization."
            )
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)

        init_tracking(args, primary=False)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _get_metrics_router_addr(self) -> str | None:
        """Return the router address for scraping SGLang engine metrics.

        The sglang_router gateway exposes ``/engine_metrics`` on its main port,
        which aggregates Prometheus metrics from all backend sglang servers.
        Returns ``http://{ip}:{port}`` for the first server, or ``None`` when
        metrics are disabled or no servers are running.
        """
        srv = self.server
        if srv is None or srv.router_ip is None:
            return None
        return f"http://{srv.router_ip}:{srv.router_port}"

    def get_metrics_router_addr(self) -> str | None:
        """Public wrapper for remote calls from the driver process."""
        return self._get_metrics_router_addr()

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.server and self.server.server_groups[0].all_engines and self.server.server_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.server_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        self._stop_rollout_background_workers()
        for monitor in self._health_monitors:
            monitor.stop()
        logging_utils.finish_tracking(self.args)

    def _stop_rollout_background_workers(self) -> None:
        stopped_modules: set[str] = set()
        for fn in (self.generate_rollout, self.eval_generate_rollout):
            module_name = getattr(fn, "__module__", None)
            if not module_name or module_name in stopped_modules:
                continue
            module = sys.modules.get(module_name)
            stop_fn = getattr(module, "_stop_global_worker", None) if module is not None else None
            if not callable(stop_fn):
                continue
            logger.info("Stopping rollout background worker from %s", module_name)
            try:
                stop_fn()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to stop rollout background worker from %s", module_name)
            stopped_modules.add(module_name)

    @property
    def server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_updatable_server(self) -> RolloutServer | None:
        """Return the server with ``update_weights=True``.

        When multiple updatable servers exist, returns the first one
        (multi-model weight update is not yet supported).
        """
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates.

        Returns engines from the first model that has
        ``update_weights=True``.  Frozen models (reference, reward,
        etc.) are automatically excluded.
        """
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def _refresh_active_lora_name(self) -> None:
        """Pull the engine's currently-served LoRA adapter name into the rollout's
        GenerateState so every /generate payload routes to the live (alternating)
        adapter. No-op unless the LoRA-adapter sync path is on. The engine is the
        source of truth: the weight sync (in the training actor process) loads the
        new name onto the engines before this rollout runs, so a single query here
        picks it up. ``None`` (no adapter loaded yet) leaves lora_path unset ->
        base-only serving, which is correct for the zero-init initial adapter."""
        if self.args.debug_train_only or not use_lora_weight_sync(self.args):
            return
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        if not engines:
            return
        try:
            active_name = ray.get(engines[0].get_active_lora_name.remote())
        except Exception as e:  # never let a name refresh break the rollout
            logger.warning(f"Failed to refresh active LoRA adapter name: {e}")
            return
        # GenerateState is a process-global singleton shared with generate_rollout
        # (which runs in this same process). Accessing it here constructs it if
        # needed; generate_rollout then reuses the same instance.
        from slime.rollout.sglang_rollout import GenerateState

        GenerateState(self.args).active_lora_name = active_name

    def generate(self, rollout_id, weight_version=None):
        start_time = time.time()
        self.rollout_id = rollout_id
        # Weight version this generation samples under (stamped into each
        # sample's metadata by _set_rollout_step_metadata). Under the async
        # loop the batch is TRAINED under weight_version+1, so per-sample
        # weight staleness = (this batch's version + 1) - sample's stamped
        # version: exactly 1 for fresh samples, >1 for buffer carry-overs.
        self.gen_weight_version = weight_version
        self.args.gen_weight_version = weight_version
        self.health_monitoring_resume()
        self._refresh_active_lora_name()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data)

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()
        self._refresh_active_lora_name()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self):
        for srv in self.servers.values():
            srv.onload_kv()

    def recover_updatable_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            srv.num_new_engines,
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # Enforce the group_id contract before flattening: any list[Sample]
            # encountered in the nested output must have group_id set on every
            # element. Default rollouts land at depth 1 and skip this validation;
            # compact / subagent paths that split one rollout into N samples must
            # set the same group_id on every sibling so the loss reducer counts
            # the group once instead of N times. Legacy rollout_id is accepted.
            _validate_group_id_annotated(data)
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

        return data, metrics

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline", "rloo"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            expected_reward_count = self.args.n_samples_per_prompt * self.args.rollout_batch_size
            # if getattr(self.args, "use_multi_turn", False):
            #     max_turns = getattr(self.args, "max_turns", None)
            #     assert max_turns is not None, "--max-turns must be set when --use-multi-turn is enabled"
            #     expected_reward_count *= int(max_turns)

            if rewards.shape[-1] == expected_reward_count:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)
            if self.args.advantage_estimator in ["rloo"]:
                # Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
                # Each contiguous group of ``n_samples_per_prompt`` samples is treated as one
                # prompt group. The leave-one-out baseline for a sample is the mean reward of
                # the other samples in that group. For singleton groups, no baseline is used.
                group_len = rewards.shape[-1]
                if group_len == 1:
                    rewards = torch.zeros_like(rewards)  # zero normalization if only one sample in the group
                else:
                    rewards = rewards * group_len / (group_len - 1)
            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _filter_missing_routing_replay_samples(self, samples: list[Sample]) -> list[Sample]:
        if not getattr(self.args, "use_rollout_routing_replay", False):
            return samples

        missing_positions = [
            i for i, sample in enumerate(samples) if getattr(sample, "rollout_routed_experts", None) is None
        ]
        if not missing_positions:
            return samples

        drop_positions = set(missing_positions)
        filter_mode = "sample"
        if getattr(self.args, "enable_turns_dp_partitions", False):
            missing_indices = {
                samples[i].index for i in missing_positions if getattr(samples[i], "index", None) is not None
            }
            if missing_indices:
                drop_positions = {
                    i
                    for i, sample in enumerate(samples)
                    if getattr(sample, "index", None) in missing_indices or i in drop_positions
                }
                filter_mode = "trajectory"

        kept_samples = [sample for i, sample in enumerate(samples) if i not in drop_positions]
        logger.warning(
            "Dropped %d/%d rollout samples before train-data sharding because %d sample(s) "
            "were missing rollout_routed_experts while routing replay is enabled "
            "(filter_mode=%s).",
            len(drop_positions),
            len(samples),
            len(missing_positions),
            filter_mode,
        )
        if not kept_samples:
            raise ValueError(
                "All rollout samples were missing rollout_routed_experts while "
                "--use-rollout-routing-replay is enabled."
            )
        return kept_samples

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        samples = self._filter_missing_routing_replay_samples(samples)

        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        # Group id (one per training aggregation unit). Default rollouts emit
        # one sample per group, so we fall back to the unique sample index.
        # Compact / subagent paths that emit multiple training samples per
        # group set ``Sample.group_id`` explicitly so all siblings share a
        # value; assigning legacy ``Sample.rollout_id`` still forwards here.
        group_ids = [sample.group_id if sample.group_id is not None else sample.index for sample in samples]

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "group_ids": group_ids,
        }
        if any(sample.metadata and "turn_idx" in sample.metadata for sample in samples):
            train_data["turn_indices"] = [
                sample.metadata["turn_idx"] if sample.metadata and "turn_idx" in sample.metadata else None
                for sample in samples
            ]

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        predictive_top_k = int(getattr(self.args, "dppo_predictive_top_k", 0) or 0)
        if predictive_top_k:
            for sample_position, sample in enumerate(samples):
                _validate_predictive_support_sample(sample, predictive_top_k, sample_position)
            for field in _PREDICTIVE_SUPPORT_FIELDS:
                train_data[field] = [getattr(sample, field) for sample in samples]

        # Per-group aggregate, precomputed at the step level (where we can
        # see every sample of every group) and broadcast per-sample so the
        # per-mb loss reducer uses the correct whole-group denominator even
        # when a group's samples land in different micro-batches (first-fit
        # packing can split a group across mbs):
        #
        #   ``group_mask_sums[i]`` — sum of loss-mask totals over every
        #   sample in sample i's group. Used as the reducer's denominator
        #   so summing partial contributions across mbs yields one
        #   token-weighted mean per group.
        group_id_list = train_data["group_ids"]
        mask_sums_per_sample = [sum(m) for m in loss_masks]
        group_total_mask: dict[int, int] = {}
        for group_id, ms in zip(group_id_list, mask_sums_per_sample, strict=True):
            group_total_mask[group_id] = group_total_mask.get(group_id, 0) + ms
        group_mask_sums = [group_total_mask[group_id] for group_id in group_id_list]
        train_data["group_mask_sums"] = group_mask_sums

        # Overwrite raw_reward when available. Mixed-source batches may only
        # populate this field for a subset of samples (e.g. SWE but not code).
        if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
            train_data["raw_reward"] = [
                sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
                for sample in samples
            ]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        # Per-sample generation weight version (stamped once at first submission in
        # _set_rollout_step_metadata). The LoRA old-actor asserts every scored
        # sample shares the snapshot's behavioral version; a buffered carry-over
        # keeps its original (older) stamp and would trip that assert loudly.
        if any(s.metadata and "gen_weight_version" in s.metadata for s in samples):
            train_data["gen_weight_versions"] = [
                (s.metadata.get("gen_weight_version") if isinstance(getattr(s, "metadata", None), dict) else None)
                for s in samples
            ]

        if getattr(self.args, "use_rollout_routing_replay", False):
            missing_routing = [
                i for i, sample in enumerate(samples) if getattr(sample, "rollout_routed_experts", None) is None
            ]
            if missing_routing:
                raise ValueError(
                    "rollout_routed_experts is required on every remaining sample when "
                    f"--use-rollout-routing-replay is enabled; missing positions: {missing_routing[:8]}"
                )
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
        elif all(sample.rollout_routed_experts is not None for sample in samples):
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _get_turns_dp_partitions(self, data, total_lengths, dp_size):
        if "sample_indices" not in data or "turn_indices" not in data:
            raise ValueError(
                "--enable-turns-dp-partitions requires train data to include sample_indices and turn_indices."
            )

        sample_indices = data["sample_indices"]
        turn_indices = data["turn_indices"]

        trajectory_turns = {}
        for sample_index, turn_idx in zip(sample_indices, turn_indices, strict=True):
            if sample_index is None or turn_idx is None:
                raise ValueError(
                    "--enable-turns-dp-partitions requires every sample to have non-None sample_index and turn_idx."
                )
            sample_index = int(sample_index)
            turn_idx = int(turn_idx)
            if turn_idx in trajectory_turns.setdefault(sample_index, set()):
                raise ValueError(
                    "--enable-turns-dp-partitions found duplicate turn data for "
                    f"sample_index={sample_index}, turn_idx={turn_idx}."
                )
            trajectory_turns[sample_index].add(turn_idx)

        turns_per_trajectory = len(next(iter(trajectory_turns.values())))
        if any(len(turns) != turns_per_trajectory for turns in trajectory_turns.values()):
            raise ValueError(
                "--enable-turns-dp-partitions requires every trajectory to have the same number of turns. "
                "For turns_geometric, enable padding turns before training data is split by DP."
            )
        if len(trajectory_turns) < dp_size:
            raise ValueError(
                "--enable-turns-dp-partitions requires at least one trajectory per DP rank: "
                f"num_trajectories={len(trajectory_turns)}, dp_size={dp_size}."
            )

        return build_dp_schedule(
            self.args,
            self.train_parallel_config,
            total_lengths,
            global_batch_size=self.args.global_batch_size,
            group_indices=sample_indices,
            pack_group_atomic=True,
            group_sample_sort_keys=turn_indices,
        )

    def _split_train_data_by_dp(self, data):
        """Compute the DP/mbs schedule and package each rank's rollout_data
        into a Ray Box. The schedule itself is computed by
        :func:`build_dp_schedule` so it stays unit-testable without Ray/sglang.

        Step split is by group id (``samples[i].group_id``, falling back to
        ``samples[i].index``). With ``--enable-turns-dp-partitions``, the
        scheduling group is ``sample_indices`` so every trajectory's turns stay
        on the same DP rank and in trajectory-major order. Each step holds
        exactly ``args.global_batch_size`` scheduling groups so the training
        step count is fixed at ``num_groups // global_batch_size``.
        """
        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if getattr(self.args, "enable_turns_dp_partitions", False):
            partitions, micro_batch_indices, num_microbatches, global_batch_sizes = self._get_turns_dp_partitions(
                data, total_lengths, dp_size
            )
        else:
            partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
                self.args,
                self.train_parallel_config,
                total_lengths,
                global_batch_size=self.args.global_batch_size,
                group_indices=data["group_ids"],
            )

        # Package per-rank rollout_data
        rollout_data_refs = []
        for r in range(dp_size):
            partition = partitions[r]
            rollout_data = {"partition": partition}
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "turn_indices",
                "group_ids",
                "group_mask_sums",
                "rollout_log_probs",
                "rollout_topk_token_ids",
                "rollout_topk_log_probs",
                "rollout_topk_valid_mask",
                "gen_weight_versions",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = [data[key][j] for j in partition]
            # keys that need to be splited at train side
            for key in ["raw_reward", "total_lengths"]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            rollout_data["global_batch_sizes"] = global_batch_sizes
            rollout_data["num_microbatches"] = num_microbatches
            rollout_data["micro_batch_indices"] = micro_batch_indices[r]
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def _validate_group_id_annotated(node, depth=0):
    """Walk the rollout function's nested output and validate ``group_id`` only
    when a compact / subagent pattern is detected.

    "Compact" = the rollout function wraps multiple training samples from one
    rollout execution into a ``list[Sample]``. In slime's convention the
    default rollout shape is ``list[list[Sample]]`` (depth-2: prompt × rollout)
    so its leaf ``list[Sample]`` lands at depth 1 and we skip validation,
    preserving backward compatibility. A compact rollout adds a third level:
    ``list[list[list[Sample]]]`` (prompt × rollout × samples-from-one-group),
    so the leaf ``list[Sample]`` lands at depth ≥ 2. At that point we require
    every sibling to carry a non-None ``group_id`` (or legacy ``rollout_id``)
    and to share the same value, so the loss reducer counts the group once
    instead of N times.
    """
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            group_ids = [s.group_id for s in node]
            missing = [i for i, group_id in enumerate(group_ids) if group_id is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but group_id is unset on "
                f"positions {missing}. Set Sample.group_id on every sibling so the loss "
                "reducer can aggregate them as one group instead of N."
            )
            assert (
                len(set(group_ids)) == 1
            ), f"Sibling samples from one compact rollout must share group_id; got {group_ids}."
        return
    for item in node:
        _validate_group_id_annotated(item, depth + 1)


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = {}
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports[rank] = dict(
            dist_init_addr=addr,
            nccl_port=None,
            host=host,
            port=int(port),
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    addr_and_ports: dict[int, dict] = {}

    # Track per-node port cursors so that different server groups (called
    # sequentially) never race for the same ports on a given node.
    node_port_cursor: dict[str, int] = {}
    engines_by_rank = dict(rollout_engines)

    # Placement groups may intentionally spread fewer-than-node-sized engines
    # across hosts (for example one TP4 engine on each 8-GPU node).  Rank
    # arithmetic cannot recover that physical placement.  Query every actor and
    # allocate its server/NCCL ports on the actor's actual host.
    hosts_by_rank = {
        rank: ray.get(engine._get_current_node_ip_and_free_port.remote())[0] for rank, engine in rollout_engines
    }

    def get_port(rank: int, consecutive: int = 1) -> int:
        host = hosts_by_rank[rank]
        start_port = node_port_cursor.get(host, base_port)
        returned_host, selected_port = ray.get(
            engines_by_rank[rank]._get_current_node_ip_and_free_port.remote(
                start_port=start_port,
                consecutive=consecutive,
            )
        )
        if returned_host != host:
            raise RuntimeError(f"Rollout engine {rank} moved nodes during port allocation: {host} -> {returned_host}")
        node_port_cursor[host] = selected_port + consecutive
        return selected_port

    ranks = sorted(engines_by_rank)
    for rank in ranks:
        addr_and_ports[rank] = {
            "host": hosts_by_rank[rank],
            "port": get_port(rank),
            "nccl_port": get_port(rank),
        }
        if worker_type == "prefill":
            addr_and_ports[rank]["disaggregation_bootstrap_port"] = get_port(rank)

    if _gpus_per_engine > args.num_gpus_per_node:
        num_nodes_per_engine = _gpus_per_engine // args.num_gpus_per_node
        if len(ranks) % num_nodes_per_engine != 0:
            raise ValueError(
                f"Rollout engine ranks ({len(ranks)}) are not divisible by nodes per engine "
                f"({num_nodes_per_engine})."
            )
        for group_start in range(0, len(ranks), num_nodes_per_engine):
            group_ranks = ranks[group_start : group_start + num_nodes_per_engine]
            root_rank = group_ranks[0]
            dist_init_addr = f"{hosts_by_rank[root_rank]}:{get_port(root_rank, 30 + args.sglang_dp_size)}"
            for rank in group_ranks:
                addr_and_ports[rank]["dist_init_addr"] = dist_init_addr
    else:
        for rank in ranks:
            addr_and_ports[rank][
                "dist_init_addr"
            ] = f"{hosts_by_rank[rank]}:{get_port(rank, 30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sglang_router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set (e.g. by the user) and
    ``force_new`` is False, skip launching and return the existing values.
    When ``force_new`` is True (multi-model), always allocate a fresh port.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    from sglang_router.launch_router import RouterArgs

    from slime.utils.http_utils import run_router

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True
        # Disable circuit breaker to prevent RDMA transfer timeouts from
        # marking decode workers as dead. Timeouts are transient (PCIe
        # contention under high load) and do not indicate a dead server.
        router_args.disable_circuit_breaker = True

    # We will not use the health check from router.
    router_args.disable_health_check = True

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}, Prometheus port: {router_args.prometheus_port}")
    return router_ip, router_port


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if args.debug_rollout_only:
        return 0
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    """Start rollout servers: one per model, each with its own router.

    Each model defined in the sglang config gets its own router and set
    of server groups.  Server groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns a dict mapping model name → ``RolloutServer``.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    # Compute megatron GPU range for per-group offload decisions.
    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        # Write back for backward compat (first model only).
        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_groups: list[ServerGroup] = []
        port_cursors: dict[str, int] = {}

        has_epd = model_cfg.has_encoder_disaggregation

        def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
            nonlocal engine_offset, gpu_offset
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            overrides = dict(group_cfg.overrides)
            if overrides_extra:
                for k, v in overrides_extra.items():
                    overrides.setdefault(k, v)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                f"(abs={group_abs_start}): needs_offload={needs_offload}"
            )

            group = ServerGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=overrides,
                needs_offload=needs_offload,
                model_path=overrides.get(
                    "model_path",
                    getattr(args, "rollout_model_path", None) or args.hf_checkpoint,
                ),
                router_ip=router_ip,
                router_port=router_port,
            )
            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus
            return group

        if has_epd:
            # --- Phase 1: start encoder groups, wait, collect URLs ---
            encoder_urls: list[str] = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type != "encoder":
                    continue
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                if handles:
                    ray.get(handles)
                urls = ray.get([e.get_url.remote() for e in group.engines])
                encoder_urls.extend(u for u in urls if u is not None)
                server_groups.append(group)

            logger.info(f"EPD phase 1 done: collected {len(encoder_urls)} encoder URLs: {encoder_urls}")

            # --- Phase 2: start non-encoder groups, injecting encoder URLs into
            # language-only LLM workers. Prefill groups use this for full EPD,
            # while regular groups allow encoder/LLM split without PD.
            non_encoder_handles: list = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type == "encoder":
                    continue
                overrides_extra = {}
                if encoder_urls and group_cfg.worker_type in ("prefill", "regular"):
                    overrides_extra["language_only"] = True
                    overrides_extra["encoder_urls"] = encoder_urls
                group = _make_group(group_cfg, router_ip, router_port, overrides_extra=overrides_extra)
                handles, port_cursors = group.start_engines(port_cursors)
                non_encoder_handles.extend(handles)
                server_groups.append(group)

            if non_encoder_handles:
                ray.get(non_encoder_handles)
        else:
            # No EPD — start all groups in one pass (original path).
            all_init_handles: list = []
            for group_cfg in model_cfg.server_groups:
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                all_init_handles.extend(handles)
                server_groups.append(group)

            if all_init_handles:
                ray.get(all_init_handles)

        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    # Expose per-model router info for custom rollout functions.
    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        # Validate total GPUs match.
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    # Default: single regular group.
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    if args.wandb_always_use_train_step:
        log_dict["train/step"] = step
        log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    step = compute_rollout_step(args, rollout_id)
    # Sample staleness: optimizer steps between when a sample was GENERATED
    # (metadata["rollout_step"], stamped at submit time) and when it is TRAINED
    # (this rollout's train-step base). ~0-1 for fresh sync-loop samples; larger
    # for buffered / over-sampled / partial-rollout carry-overs. Computed BEFORE
    # the perf log line so it is visible in the TEXT log, not only wandb (it was
    # silently wandb-only, which read as "metric not implemented" 2026-07-11).
    staleness = [
        step - int(s.metadata["rollout_step"])
        for group in samples
        for s in (group if isinstance(group, list) else [group])
        if isinstance(getattr(s, "metadata", None), dict) and "rollout_step" in s.metadata
    ]
    if staleness:
        log_dict["rollout/sample_staleness_steps/mean"] = sum(staleness) / len(staleness)
        log_dict["rollout/sample_staleness_steps/max"] = max(staleness)
        log_dict["rollout/sample_staleness_steps/stale_frac"] = sum(1 for s in staleness if s > 1) / len(staleness)
    else:
        # No sample carried a rollout_step stamp — surface that loudly instead of
        # silently omitting the metric (a lost-metadata bug looks identical to
        # "no staleness" otherwise).
        log_dict["rollout/sample_staleness_steps/unstamped"] = 1.0

    # WEIGHT-VERSION staleness (user-requested definition 2026-07-12): the diff
    # between the weight version the sample will be TRAINED under and the one it
    # was GENERATED under. Under the async drain loop this batch trains under
    # gen_weight_version + 1, so fresh samples read exactly 1; buffered prompt
    # carry-overs (stamped once at first submission) read >1.
    gen_v = getattr(args, "gen_weight_version", None)
    if gen_v is not None:
        w_staleness = [
            (gen_v + 1) - int(s.metadata["gen_weight_version"])
            for group in samples
            for s in (group if isinstance(group, list) else [group])
            if isinstance(getattr(s, "metadata", None), dict) and "gen_weight_version" in s.metadata
        ]
        if w_staleness:
            log_dict["rollout/weight_staleness_steps/mean"] = sum(w_staleness) / len(w_staleness)
            log_dict["rollout/weight_staleness_steps/max"] = max(w_staleness)
            log_dict["rollout/weight_staleness_steps/stale_frac"] = sum(1 for s in w_staleness if s > 1) / len(
                w_staleness
            )
    logger.info(f"perf {rollout_id}: {log_dict}")
    log_dict["rollout/step"] = step
    if args.wandb_always_use_train_step:
        log_dict["train/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_kernel_agent_metrics(samples)
    # Kernel-agent samples keep a turn index even in true single-turn mode.  Emit
    # per-turn metrics whenever that provenance is present; only the trajectory
    # metrics inside _compute_kernel_multi_turn_metrics require max_turns > 1.
    has_turn_metadata = any(
        isinstance(getattr(sample, "metadata", None), dict) and "turn_idx" in sample.metadata for sample in samples
    )
    if getattr(args, "use_multi_turn", False) or has_turn_metadata:
        log_dict |= _compute_kernel_multi_turn_metrics(args, samples)
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    if getattr(args, "log_response_diversity", False):
        log_dict |= _compute_response_diversity(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def _iter_response_diversity_groups(args, samples):
    if any(sample.group_id is not None or sample.group_index is not None for sample in samples):
        groups = {}
        for sample in samples:
            if sample.group_id is not None:
                group_key = sample.group_id
            elif sample.group_index is not None:
                group_key = sample.group_index
            else:
                group_key = sample.index
            groups.setdefault(group_key, []).append(sample)
        return groups.values()

    group_size = max(int(getattr(args, "n_samples_per_prompt", 1) or 1), 1)
    return (samples[i : i + group_size] for i in range(0, len(samples), group_size))


def _compute_response_diversity(args, samples) -> dict[str, float]:
    token_bits = 32
    max_token = (1 << token_bits) - 1
    tail_mask = (1 << (token_bits * 3)) - 1
    diversities = []

    for group in _iter_response_diversity_groups(args, samples):
        total = 0
        unique = set()
        for sample in group:
            response_length = int(sample.response_length or 0)
            if response_length < 4:
                continue
            tokens = sample.tokens[-response_length:]
            if len(tokens) < 4:
                continue
            total += len(tokens) - 3

            a, b, c, d = tokens[0], tokens[1], tokens[2], tokens[3]
            if (a | b | c | d) > max_token:
                raise ValueError(f"token id exceeds response diversity token_bits={token_bits}")
            key = (((a << token_bits) | b) << token_bits | c) << token_bits | d
            unique.add(key)

            for token in tokens[4:]:
                if token > max_token:
                    raise ValueError(f"token id {token} exceeds response diversity token_bits={token_bits}")
                key = ((key & tail_mask) << token_bits) | token
                unique.add(key)

        if total > 0:
            diversities.append(len(unique) / total)

    return {"response_diversity": float(np.mean(diversities).item()) if diversities else 0.0}


FAST_THRESHOLDS = (1.0, 1.2, 1.5, 2.0, 3.0)


def _compute_kernel_multi_turn_metrics(args, samples):
    values_by_turn = {}
    sample_trajectory = {}
    for sample in samples:
        metadata = sample.metadata or {}
        if "turn_idx" not in metadata:
            continue
        env_extra_info = metadata.get("env_extra_info")
        if not isinstance(env_extra_info, dict):
            continue

        try:
            turn_idx = int(metadata["turn_idx"])
        except (TypeError, ValueError):
            continue

        correctness = bool(env_extra_info.get("correctness")) and not bool(env_extra_info.get("decoy_kernel"))
        compilation = bool(env_extra_info.get("compilation"))
        speedup = env_extra_info.get("speedup")
        if isinstance(speedup, bool) or not isinstance(speedup, (int, float)):
            continue

        if sample.index is not None:
            sample_trajectory.setdefault(sample.index, {})[turn_idx] = {
                "correctness": correctness,
                "compilation": compilation,
                "speedup": float(speedup),
            }

        turn_values = values_by_turn.setdefault(
            turn_idx,
            {
                "correctness": [],
                "compilation": [],
                "speedup": [],
                "fast": {threshold: [] for threshold in FAST_THRESHOLDS},
            },
        )
        turn_values["correctness"].append(float(correctness))
        turn_values["compilation"].append(float(compilation))
        turn_values["speedup"].append(float(speedup))
        for threshold in FAST_THRESHOLDS:
            turn_values["fast"][threshold].append(float(correctness and speedup >= threshold))

    log_dict = {}
    for turn_idx in sorted(values_by_turn):
        turn_values = values_by_turn[turn_idx]
        prefix = f"kernel/turn{turn_idx}"
        for key in ("correctness", "compilation", "speedup"):
            values = turn_values[key]
            if values:
                log_dict[f"{prefix}/{key}"] = np.mean(values).item()
        for threshold, values in turn_values["fast"].items():
            if values:
                log_dict[f"{prefix}/fast@{threshold:g}"] = np.mean(values).item()
    log_dict |= _compute_kernel_trajectory_metrics(args, sample_trajectory)
    return log_dict


def _compute_kernel_trajectory_metrics(args, sample_trajectory):
    # Cross-turn trajectory metrics (best_by_turn_N, first-vs-last correctness,
    # improvement counts) only mean something with more than one turn. In single-turn
    # training every series is degenerate -- first == last, improved == regressed == 0,
    # and best_by_turn_2/3 just duplicate best_by_turn_1 -- so skip the whole block
    # instead of polluting W&B with redundant keys.
    max_turns = int(getattr(args, "max_turns", 1) or 1)
    if max_turns <= 1:
        return {}

    first_turn_correct = []
    last_turn_correct = []
    improved_samples = 0
    regressed_samples = 0
    # One best_by_turn_{k} cutoff per configured turn, k = 1..max_turns, so the set of
    # logged keys tracks the actual turn budget instead of a hardcoded 1/2/3.
    best_by_turn = {
        turn_count: {"correctness": [], "compilation": [], "speedup": []} for turn_count in range(1, max_turns + 1)
    }

    for trajectory in sample_trajectory.values():
        if not trajectory:
            continue
        sorted_turns = [trajectory[turn_idx] for turn_idx in sorted(trajectory)]
        first_correct = bool(sorted_turns[0]["correctness"])
        last_correct = bool(sorted_turns[-1]["correctness"])
        first_turn_correct.append(float(first_correct))
        last_turn_correct.append(float(last_correct))
        if not first_correct and last_correct:
            improved_samples += 1
        elif first_correct and not last_correct:
            regressed_samples += 1

        for turn_count, metrics in best_by_turn.items():
            visible_turns = sorted_turns[:turn_count]
            if not visible_turns:
                continue
            metrics["correctness"].append(float(any(turn["correctness"] for turn in visible_turns)))
            metrics["compilation"].append(float(any(turn["compilation"] for turn in visible_turns)))
            metrics["speedup"].append(max(float(turn["speedup"]) for turn in visible_turns))

    if not first_turn_correct:
        return {}

    log_dict = {
        "kernel/correct_improvement/first_turn_correct_rate": np.mean(first_turn_correct).item(),
        "kernel/correct_improvement/last_turn_correct_rate": np.mean(last_turn_correct).item(),
        "kernel/correct_improvement/improved_samples": improved_samples,
        "kernel/correct_improvement/regressed_samples": regressed_samples,
        "kernel/correct_improvement/net_improvement": improved_samples - regressed_samples,
    }
    for turn_count, metrics in best_by_turn.items():
        prefix = f"kernel/trajectory/best_by_turn_{turn_count}"
        for key, values in metrics.items():
            if values:
                log_dict[f"{prefix}/{key}"] = np.mean(values).item()
    return log_dict


def _compute_kernel_agent_metrics(samples):
    bool_keys = {
        "correctness",
        "compilation",
        "decoy_kernel",
        "correctness_candidate_forward_completed",
        "correctness_output_mismatch",
    }
    coverage_keys = {"time_coverage", "num_coverage"}
    values_by_key = {}
    decoy_reason_count = {}
    incorrect_backend_probe_skip_reason_count = {}
    overlong_penalty_values = []
    time_values = {
        "model_time": [],
        "env_time": [],
        "detail_env_time/kernel_runtime": [],
        "detail_env_time/profile_time": [],
        "detail_env_time/refer_runtime": [],
    }
    total_count = len(samples)
    coverage_rs_masked_count = 0
    conditional_truncation_masked_count = 0
    correct_count = 0
    coverage_rs_correct_masked_count = 0
    precheck_count = 0
    precheck_passed_count = 0
    env_status_count = 0
    env_timeout_count = 0
    kernel_eval_client_timeout_count = 0
    non_pad_count = 0
    generate_guard_timeout_count = 0
    incorrect_backend_probe_attempted_count = 0
    incorrect_backend_probe_valid_count = 0
    incorrect_backend_probe_custom_kernel_observed_count = 0
    incorrect_backend_probe_decoy_detected_count = 0
    partial_credit_applied_count = 0
    partial_credit_rejected_decoy_count = 0
    partial_credit_rejected_runtime_error_count = 0
    partial_credit_rejected_timeout_count = 0

    def record_reason(counter: dict[str, int], value) -> None:
        if isinstance(value, str) and value:
            key = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
            if key:
                counter[key] = counter.get(key, 0) + 1

    for sample in samples:
        metadata = sample.metadata or {}
        partial_reason = metadata.get("partial_credit_output_mismatch_reason")
        if metadata.get("partial_credit_output_mismatch") is True:
            partial_credit_applied_count += 1
        elif partial_reason == "decoy":
            partial_credit_rejected_decoy_count += 1
        elif partial_reason == "runtime_error":
            partial_credit_rejected_runtime_error_count += 1
        elif partial_reason == "timeout":
            partial_credit_rejected_timeout_count += 1
        overlong_penalty = metadata.get("overlong_penalty")
        if not isinstance(overlong_penalty, bool) and isinstance(overlong_penalty, (int, float)):
            overlong_penalty_values.append(float(overlong_penalty))
        is_coverage_rs_masked = sample.remove_sample and metadata.get("remove_reason") == "coverage_rs"
        is_conditional_truncation_masked = bool(metadata.get("conditional_truncation_masked"))
        if is_coverage_rs_masked:
            coverage_rs_masked_count += 1
        if is_conditional_truncation_masked:
            conditional_truncation_masked_count += 1

        if not metadata.get("is_pad_turn"):
            non_pad_count += 1
            if sample.status == Sample.Status.ABORTED and metadata.get("abort_reason") == "wall_clock_timeout":
                generate_guard_timeout_count += 1

        model_time = metadata.get("model_time")
        if not isinstance(model_time, bool) and isinstance(model_time, (int, float)):
            time_values["model_time"].append(float(model_time))

        env_extra_info = metadata.get("env_extra_info")
        if not isinstance(env_extra_info, dict):
            continue

        env_time = metadata.get("env_time")
        if (
            env_extra_info.get("precheck") != "failed"
            and not isinstance(env_time, bool)
            and isinstance(env_time, (int, float))
        ):
            time_values["env_time"].append(float(env_time))

            detail_env_time = env_extra_info.get("detail_env_time")
            if isinstance(detail_env_time, dict):
                for key in ("kernel_runtime", "profile_time", "refer_runtime"):
                    value = detail_env_time.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    time_values[f"detail_env_time/{key}"].append(float(value))

        env_result = metadata.get("env_result")
        env_state = env_result.get("env_state") if isinstance(env_result, dict) else None
        if not isinstance(env_state, dict):
            env_state = {}

        status = env_state.get("status")
        if status is not None:
            env_status_count += 1
            if status == "timeout":
                env_timeout_count += 1
                error_message = str(env_state.get("error_message") or env_state.get("error") or "")
                if "client-side" in error_message:
                    kernel_eval_client_timeout_count += 1

        precheck = env_extra_info.get("precheck")
        if precheck in ("passed", "failed"):
            precheck_count += 1
            if precheck == "passed":
                precheck_passed_count += 1

        record_reason(decoy_reason_count, env_extra_info.get("decoy_reason"))
        record_reason(
            incorrect_backend_probe_skip_reason_count,
            env_extra_info.get("incorrect_backend_probe_skip_reason"),
        )
        if env_extra_info.get("incorrect_backend_probe_attempted") is True:
            incorrect_backend_probe_attempted_count += 1
        if env_extra_info.get("incorrect_backend_probe_valid") is True:
            incorrect_backend_probe_valid_count += 1
        if env_extra_info.get("incorrect_backend_probe_custom_kernel_observed") is True:
            incorrect_backend_probe_custom_kernel_observed_count += 1
        if env_extra_info.get("incorrect_backend_probe_decoy_detected") is True:
            incorrect_backend_probe_decoy_detected_count += 1

        is_correct = bool(env_extra_info.get("correctness")) and not bool(env_extra_info.get("decoy_kernel"))
        if is_correct:
            correct_count += 1
            if is_coverage_rs_masked:
                coverage_rs_correct_masked_count += 1

        for key, value in env_extra_info.items():
            if key in bool_keys:
                if isinstance(value, bool):
                    values_by_key.setdefault(key, []).append(float(value))
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                values_by_key.setdefault(key, []).append(float(value))

    log_dict = {}
    for key, values in values_by_key.items():
        if not values:
            continue
        prefix = "coverage" if key in coverage_keys else "env_extra_info"
        if key in bool_keys:
            log_dict[f"{prefix}/{key}/mean"] = np.mean(values).item()
        else:
            stats = compute_statistics(values)
            for stat_key in ("min", "max", "mean"):
                log_dict[f"{prefix}/{key}/{stat_key}"] = stats[stat_key]
    if total_count > 0:
        log_dict["sample_mask/coverage_rs_masked_fraction"] = coverage_rs_masked_count / total_count
        log_dict["sample_mask/conditional_truncation_masked_fraction"] = (
            conditional_truncation_masked_count / total_count
        )
    if correct_count > 0:
        log_dict["sample_mask/coverage_rs_correct_masked_fraction"] = coverage_rs_correct_masked_count / correct_count
    if precheck_count > 0:
        log_dict["kernel/precheck_pass_rate"] = precheck_passed_count / precheck_count
    if env_status_count > 0:
        log_dict["kernel/eval_timeout_count"] = env_timeout_count
        log_dict["kernel/eval_timeout_ratio"] = env_timeout_count / env_status_count
        log_dict["kernel/eval_client_timeout_count"] = kernel_eval_client_timeout_count
        log_dict["kernel/eval_client_timeout_ratio"] = kernel_eval_client_timeout_count / env_status_count
    if non_pad_count > 0:
        log_dict["kernel/generate_guard_timeout_count"] = generate_guard_timeout_count
        log_dict["kernel/generate_guard_timeout_ratio"] = generate_guard_timeout_count / non_pad_count
        log_dict["kernel/partial_credit/applied_rate"] = partial_credit_applied_count / non_pad_count
        log_dict["kernel/partial_credit/rejected_decoy_count"] = partial_credit_rejected_decoy_count
        log_dict["kernel/partial_credit/rejected_runtime_error_count"] = partial_credit_rejected_runtime_error_count
        log_dict["kernel/partial_credit/rejected_timeout_count"] = partial_credit_rejected_timeout_count
        for reason, count in decoy_reason_count.items():
            log_dict[f"kernel/decoy_reason/{reason}_count"] = count
        for reason, count in incorrect_backend_probe_skip_reason_count.items():
            log_dict[f"kernel/incorrect_backend_probe/skip_{reason}_count"] = count
        log_dict["kernel/incorrect_backend_probe/attempted_count"] = incorrect_backend_probe_attempted_count
        log_dict["kernel/incorrect_backend_probe/attempted_ratio"] = (
            incorrect_backend_probe_attempted_count / non_pad_count
        )
        if incorrect_backend_probe_attempted_count > 0:
            log_dict["kernel/incorrect_backend_probe/valid_count"] = incorrect_backend_probe_valid_count
            log_dict["kernel/incorrect_backend_probe/valid_ratio_of_attempted"] = (
                incorrect_backend_probe_valid_count / incorrect_backend_probe_attempted_count
            )
        if incorrect_backend_probe_valid_count > 0:
            log_dict["kernel/incorrect_backend_probe/custom_kernel_observed_ratio_of_valid"] = (
                incorrect_backend_probe_custom_kernel_observed_count / incorrect_backend_probe_valid_count
            )
            log_dict["kernel/incorrect_backend_probe/decoy_detected_ratio_of_valid"] = (
                incorrect_backend_probe_decoy_detected_count / incorrect_backend_probe_valid_count
            )
    if overlong_penalty_values:
        log_dict["kernel/overlong_penalty/mean"] = np.mean(overlong_penalty_values).item()
    for key, values in time_values.items():
        if values:
            log_dict[f"kernel/time/{key}/mean"] = np.mean(values).item()
            log_dict[f"kernel/time/{key}/sum"] = np.sum(values).item()
            log_dict[f"kernel/time/{key}/count"] = len(values)
            log_dict[f"kernel/time/{key}/p50"] = np.percentile(values, 50).item()
            log_dict[f"kernel/time/{key}/p90"] = np.percentile(values, 90).item()
            log_dict[f"kernel/time/{key}/p95"] = np.percentile(values, 95).item()
            log_dict[f"kernel/time/{key}/max"] = np.max(values).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
