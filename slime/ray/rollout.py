import itertools
import logging
import sys
import time
from typing import Any

import numpy as np
import ray
import torch

from slime.backends.sglang_utils.deployment import start_rollout_servers
from slime.observability import logging_utils
from slime.observability.logging_utils import configure_logger, init_tracking
from slime.observability.rollout_data_utils import (
    load_debug_rollout_data,
    save_debug_rollout_data,
    tensorize_rollout_data_for_training,
    validate_rollout_id_annotated,
    validate_rollout_routed_experts_for_replay,
)
from slime.observability.rollout_metrics import log_eval_rollout_data, log_rollout_data
from slime.rollout.base_types import call_rollout_fn
from slime.rollout.sample_hooks import set_current_rollout_id
from slime.utils.data import get_source
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import init_http_client
from slime.utils.lora_utils import use_lora_weight_sync
from slime.utils.misc import Box, load_function
from slime.utils.types import Sample

from .utils import Lock, add_default_ray_env_vars

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


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

        rollout_init_handles: list[Any] = []
        if self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            init_http_client(args)
            self.servers, rollout_init_handles = start_rollout_servers(args, pg)

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
                "The default reward normalization includes padded or removed turns and may be biased."
            )
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if rollout_init_handles:
            ray.get(rollout_init_handles)

        init_tracking(args, primary=False)
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={"env_vars": add_default_ray_env_vars()},
        ).remote()
        self.rollout_id = -1

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if (
            self.server
            and self.server.server_groups
            and self.server.server_groups[0].all_engines
            and self.server.server_groups[0].all_engines[0]
        ):
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
    def server(self) -> Any | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_metrics_router_addr(self) -> str | None:
        """Return the default router address used to scrape SGLang metrics."""
        srv = self.server
        if srv is None or srv.router_ip is None or srv.router_port is None:
            return None
        return f"http://{srv.router_ip}:{srv.router_port}"

    def get_metrics_router_addr(self) -> str | None:
        """Expose the SGLang metrics router address to the driver."""
        return self._get_metrics_router_addr()

    def _get_updatable_server(self) -> Any | None:
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
        parallel_configs = srv.engine_parallel_configs if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets, parallel_configs

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
        set_current_rollout_id(rollout_id)
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
        save_debug_rollout_data(
            self.args.save_debug_rollout_data,
            data,
            rollout_id=rollout_id,
            evaluation=False,
        )
        log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data)

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        set_current_rollout_id(rollout_id)
        self.health_monitoring_resume()
        self._refresh_active_lora_name()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        save_debug_rollout_data(
            self.args.save_debug_rollout_data,
            data,
            rollout_id=rollout_id,
            evaluation=True,
        )
        log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

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
        """Restart dead updatable rollout engines before the next weight update.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            return

        srv.recover()

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
            data = load_debug_rollout_data(
                self.args.load_debug_rollout_data,
                rollout_id=rollout_id,
                subsample_ratio=self.args.load_debug_rollout_data_subsample,
            )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # Enforce the rollout_id contract before flattening: any list[Sample]
            # encountered in the nested output must have rollout_id set on every
            # element. Default rollouts inherit it from the data source; compact /
            # subagent paths that split one rollout into N training samples must
            # set the same rollout_id on every sibling so the loss reducer counts
            # the rollout once instead of N times.
            validate_rollout_id_annotated(data)
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

        return data, metrics

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline", "rloo"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo", "cispo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)
            if self.args.advantage_estimator == "rloo":
                group_len = rewards.shape[-1]
                if group_len == 1:
                    rewards = torch.zeros_like(rewards)
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

        rollout_ids = [sample.rollout_id for sample in samples]
        existed_rollout_id_values = set(rid for rid in rollout_ids if rid is not None)
        tmp_id = 0
        for i in range(len(rollout_ids)):
            if rollout_ids[i] is None:
                while tmp_id in existed_rollout_id_values:
                    tmp_id += 1
                rollout_ids[i] = tmp_id
                existed_rollout_id_values.add(tmp_id)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "rollout_ids": rollout_ids,
        }
        if any(sample.metadata and "turn_idx" in sample.metadata for sample in samples):
            train_data["turn_indices"] = [
                sample.metadata.get("turn_idx") if sample.metadata else None for sample in samples
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

        # Per-rollout aggregate, precomputed at the step level (where we can
        # see every sample of every rollout) and broadcast per-sample so the
        # per-mb loss reducer uses the correct whole-rollout denominator even
        # when a rollout's samples land in different micro-batches (first-fit
        # packing can split a rollout across mbs):
        #
        #   ``rollout_mask_sums[i]`` — sum of loss-mask totals over every
        #   sample in sample i's rollout. Used as the reducer's denominator
        #   so summing partial contributions across mbs yields one
        #   token-weighted mean per rollout.
        rollout_id_list = train_data["rollout_ids"]
        mask_sums_per_sample = [sum(m) for m in loss_masks]
        rollout_total_mask: dict[int, int] = {}
        for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
            rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
        train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]

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

        if getattr(self.args, "rollout_top_p", 1.0) != 1.0:
            for sample in samples:
                assert sample.rollout_top_p_token_ids is not None
                assert sample.rollout_top_p_token_offsets is not None
                assert len(sample.rollout_top_p_token_offsets) == sample.response_length + 1, (
                    f"top-p token offsets length {len(sample.rollout_top_p_token_offsets)} "
                    f"!= response length + 1 {sample.response_length + 1}"
                )
                offset_end = int(sample.rollout_top_p_token_offsets[-1])
                assert offset_end == len(sample.rollout_top_p_token_ids), (
                    f"top-p token offsets[-1] {offset_end} "
                    f"!= token ids length {len(sample.rollout_top_p_token_ids)}"
                )
            train_data["rollout_top_p_token_ids"] = [sample.rollout_top_p_token_ids for sample in samples]
            train_data["rollout_top_p_token_offsets"] = [sample.rollout_top_p_token_offsets for sample in samples]

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
            routed_experts = [torch.as_tensor(sample.rollout_routed_experts) for sample in samples]
            validate_rollout_routed_experts_for_replay(routed_experts, self.args)
            train_data["rollout_routed_experts"] = routed_experts
        elif all(sample.rollout_routed_experts is not None for sample in samples):
            train_data["rollout_routed_experts"] = [
                torch.as_tensor(sample.rollout_routed_experts) for sample in samples
            ]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        if samples[0].metadata is not None:
            train_data["source_names"] = [get_source(sample) for sample in samples]

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
        trajectory_turns: dict[int, set[int]] = {}
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
                "Enable padded turns before splitting the training data."
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
            rollout_indices=sample_indices,
            pack_group_atomic=True,
            group_sample_sort_keys=turn_indices,
        )

    def _split_train_data_by_dp(self, data):
        """Compute the DP/mbs schedule and package each rank's rollout_data
        into a Ray Box. The schedule itself is computed by
        :func:`build_dp_schedule` so it stays unit-testable without Ray/sglang.

        Step split is by rollout id (``samples[i].rollout_id``, falling back
        to ``samples[i].index``); each step holds exactly
        ``args.global_batch_size`` rollouts so the training-step count per
        rollout is fixed at ``rollout_batch_size * n_samples_per_prompt //
        global_batch_size`` regardless of how many training samples each
        rollout produced.
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
                rollout_indices=data["rollout_ids"],
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
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_log_probs",
                "rollout_top_p_token_ids",
                "rollout_top_p_token_offsets",
                "rollout_topk_token_ids",
                "rollout_topk_log_probs",
                "rollout_topk_valid_mask",
                "gen_weight_versions",
                "rollout_routed_experts",
                "source_names",
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
            tensorize_rollout_data_for_training(rollout_data)
            transport = getattr(self.args, "rollout_data_transport", "object-store")
            if transport == "nixl":
                rollout_data_refs.append(Box(ray.put(rollout_data, _tensor_transport="nixl")))
            elif transport == "object-store":
                rollout_data_refs.append(Box(ray.put(rollout_data)))
            else:
                raise ValueError(f"Unsupported rollout data transport: {transport!r}")
        return rollout_data_refs
