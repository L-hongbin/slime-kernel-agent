from __future__ import annotations

import copy
import glob
import logging
import math
import random
import threading
import uuid
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from examples.kernel_agent.kernel_reward import annotate_group_difficulty, get_verify_history_baseline
from examples.kernel_agent.prompt_utils import as_messages, format_feedback
from examples.kernel_agent.utils import CUDA_SECTIONS, extract_cuda_agent_kernel_code

from slime.observability.rollout_data_utils import load_debug_rollout_data, save_debug_rollout_data
from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.rollout.sglang_rollout import PromptTemplate
from slime.utils.types import Sample

logger = logging.getLogger("examples.kernel_agent.kernel_agent_data_source")

_VERIFY_SOURCE_METADATA_KEYS = (
    "history_baseline",
    "trajectory_states",
    "verify_capture_rollout_id",
    "verify_data_origin",
    "verify_data_path",
    "group_num_correct",
    "group_num_valid",
    "group_correct_rate",
    "group_difficulty",
)


def _resolve_verify_data_paths(path_specs: list[str]) -> list[Path]:
    resolved_paths = []
    seen_paths = set()
    for path_spec in path_specs:
        path = Path(path_spec).expanduser()
        if path.is_file():
            matches = [path]
        elif path.is_dir():
            matches = sorted(path.rglob("*.pt"))
        else:
            matches = [Path(match) for match in sorted(glob.glob(str(path), recursive=True)) if Path(match).is_file()]
        if not matches:
            raise FileNotFoundError(f"No verify data files matched {path_spec!r}")
        for match in matches:
            resolved = match.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                resolved_paths.append(resolved)
    return resolved_paths


def _verify_difficulty_weight(sample: Sample) -> tuple[float, float]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    trajectory_states = metadata.get("trajectory_states")
    if not isinstance(trajectory_states, list) or not trajectory_states:
        raise ValueError("verify sampling requires non-empty sample.metadata['trajectory_states']")
    if any(state not in {"failed", "successed"} for state in trajectory_states):
        raise ValueError("trajectory_states entries must be 'failed' or 'successed'")
    if "group_difficulty" not in metadata:
        raise ValueError("verify sampling requires sample.metadata['group_difficulty']")

    failed_count = trajectory_states.count("failed")
    difficulty = float(metadata["group_difficulty"])
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"group_difficulty must be in [0, 1], got {difficulty}")
    return difficulty, (1.0 + failed_count) * max(difficulty, 1e-6)


@dataclass
class VerifyCandidateEntry:
    sample: Sample
    source_weight_version: int
    group_difficulty: float
    difficulty_weight: float
    insertion_sequence: int
    capture_rollout_id: int | None
    loaded: bool
    available: bool = True

    @property
    def retention_key(self) -> tuple[int, float, int]:
        return self.source_weight_version, self.group_difficulty, self.insertion_sequence

    def sampling_weight(self, current_weight_version: int) -> float:
        version_gap = max(0, int(current_weight_version) - self.source_weight_version)
        return self.difficulty_weight / (1.0 + version_gap)


def select_verify_entries(
    candidates: list[VerifyCandidateEntry],
    *,
    target_size: int,
    max_samples_per_group: int,
    current_weight_version: int,
    rng: random.Random,
    selected_per_group: Counter[object] | None = None,
) -> list[VerifyCandidateEntry]:
    """Weighted sample without replacement under a per-source-group cap."""

    if target_size < 0:
        raise ValueError(f"target_size must be non-negative, got {target_size}")
    if max_samples_per_group <= 0:
        raise ValueError(f"max_samples_per_group must be positive, got {max_samples_per_group}")

    remaining = [candidate for candidate in candidates if candidate.available]
    selected: list[VerifyCandidateEntry] = []
    if selected_per_group is None:
        selected_per_group = Counter()
    while remaining and len(selected) < target_size:
        eligible = []
        weights = []
        for remaining_index, candidate in enumerate(remaining):
            sample = candidate.sample
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            source_group = metadata.get(
                "verify_source_group_key",
                metadata.get("source_group_index", sample.group_index),
            )
            if source_group is None:
                raise ValueError("verify sampling requires source_group_index or sample.group_index")
            if selected_per_group[source_group] >= max_samples_per_group:
                continue
            eligible.append((remaining_index, candidate, source_group))
            weights.append(candidate.sampling_weight(current_weight_version))

        if not eligible:
            break
        selected_index = rng.choices(range(len(eligible)), weights=weights, k=1)[0]
        remaining_index, candidate, source_group = eligible[selected_index]
        selected.append(candidate)
        selected_per_group[source_group] += 1
        remaining.pop(remaining_index)

    return selected


class KernelAgentDataSource(RolloutDataSourceWithBuffer):
    """Kernel prompt source with ordered, versioned verify-candidate data."""

    def __init__(self, args):
        super().__init__(args)
        self.verify_rollout_ratio = float(getattr(args, "verify_rollout_ratio", 0.0))
        self.capture_verify_data = bool(getattr(args, "capture_verify_data", False))
        self.save_verify_data = getattr(args, "save_verify_data", None)
        self.load_verify_data = getattr(args, "load_verify_data", None) or []
        if isinstance(self.load_verify_data, str):
            self.load_verify_data = [self.load_verify_data]
        self.verify_capture_enabled = self.capture_verify_data
        self.verify_samples_per_group = int(getattr(args, "verify_samples_per_group", 1))
        self.verify_version_lag = int(getattr(args, "verify_version_lag", 2))
        self.verify_data_limit = getattr(args, "verify_data_limit", math.inf)
        if not 0.0 <= self.verify_rollout_ratio <= 1.0:
            raise ValueError(f"verify_rollout_ratio must be in [0, 1], got {self.verify_rollout_ratio}")
        if self.verify_samples_per_group <= 0:
            raise ValueError(f"verify_samples_per_group must be positive, got {self.verify_samples_per_group}")
        if self.verify_version_lag < 0:
            raise ValueError(f"verify_version_lag must be non-negative, got {self.verify_version_lag}")
        if not math.isinf(self.verify_data_limit) and (
            self.verify_data_limit < 0 or int(self.verify_data_limit) != self.verify_data_limit
        ):
            raise ValueError(f"verify_data_limit must be a non-negative integer or inf, got {self.verify_data_limit}")
        if self.save_verify_data is not None and not self.verify_capture_enabled:
            raise ValueError("save_verify_data requires online verify capture")
        if self.save_verify_data is not None and "{rollout_id}" not in self.save_verify_data:
            raise ValueError("save_verify_data must contain the {rollout_id} placeholder")

        self.verify_data: list[VerifyCandidateEntry] = []
        self._verify_insertion_sequence = 0
        self._dirty_capture_rollout_ids: set[int] = set()
        self._verify_buffer_lock = threading.Lock()
        self._sample_allocation_lock = threading.Lock()
        self.anchor_kv: dict[str, dict] = {}
        self._anchor_lock = threading.Lock()
        self._verify_rng = random.Random(args.rollout_seed)
        self._verify_mix_rng = random.Random(args.rollout_seed + 1)
        self._verify_selection_weight_version: int | None = None
        self._verify_selected_per_group: Counter[object] = Counter()
        verify_prompt_config_path = getattr(args, "verify_prompt_config_path", None)
        self.verify_prompt_template = PromptTemplate.from_path(
            verify_prompt_config_path,
            prompt_name="verify_response",
        )
        if self.verify_rollout_ratio > 0.0 and self.verify_prompt_template is None:
            raise ValueError(
                "verify rollout requires --verify-prompt-config-path when --verify-rollout-ratio is positive"
            )
        self._load_verify_candidates()

    @property
    def verify_buffer(self) -> list[Sample]:
        """Compatibility view of candidates currently available for verify rollout."""

        with self._verify_buffer_lock:
            return [entry.sample for entry in self.verify_data if entry.available]

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Read kernel and verify prompts through one probabilistically mixed source."""

        with self._sample_allocation_lock:
            if self.verify_rollout_ratio == 0.0:
                return super().get_samples(num_samples)

            current_weight_version = getattr(self.args, "gen_weight_version", None)
            if current_weight_version is not None:
                self.prune_verify_candidates(
                    current_weight_version=int(current_weight_version),
                    max_version_lag=self.verify_version_lag,
                )

            groups = self._get_samples_from_buffer(num_samples)
            for _ in range(num_samples - len(groups)):
                verify_groups = []
                if self._verify_mix_rng.random() < self.verify_rollout_ratio:
                    if current_weight_version is not None:
                        verify_groups = self._get_verify_samples_locked(
                            1,
                            max_samples_per_group=self.verify_samples_per_group,
                            current_weight_version=int(current_weight_version),
                            max_version_lag=self.verify_version_lag,
                        )
                if verify_groups:
                    groups.extend(verify_groups)
                else:
                    groups.extend(super().get_samples(1))
            return groups

    @staticmethod
    def _source_kernel_weight_version(sample: Sample, *, fallback_version: int | None = None) -> int:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        source_version = metadata.get("source_kernel_weight_version", metadata.get("gen_weight_version"))
        if source_version is None:
            if fallback_version is None:
                raise ValueError("verify candidate requires metadata['gen_weight_version'] or a rollout version")
            source_version = fallback_version
        try:
            return int(source_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid verify candidate source kernel version: {source_version!r}") from exc

    def _prepare_verify_candidate(self, sample: Sample, *, rollout_id: int | None = None) -> Sample | None:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        trajectory_states = metadata.get("trajectory_states")
        if (
            metadata.get("role", "kernel") != "kernel"
            or metadata.get("verify_trajectory", False)
            or sample.status == Sample.Status.ABORTED
            or not isinstance(trajectory_states, list)
            or not trajectory_states
            or trajectory_states[-1] != "failed"
        ):
            return None
        kernel_submission = extract_cuda_agent_kernel_code(sample.response)
        if not any(f"### {section_name}\n```" in kernel_submission for section_name in CUDA_SECTIONS):
            return None

        candidate = copy.deepcopy(sample)
        candidate.metadata = dict(candidate.metadata or {})
        candidate.metadata["source_kernel_weight_version"] = self._source_kernel_weight_version(
            candidate,
            fallback_version=rollout_id,
        )
        candidate.metadata["source_kernel_submission"] = kernel_submission
        candidate.metadata.setdefault("source_group_index", candidate.group_index)
        if rollout_id is not None:
            candidate.metadata["verify_capture_rollout_id"] = int(rollout_id)
        if candidate.metadata["source_group_index"] is None:
            raise ValueError("verify candidate requires sample.group_index or metadata['source_group_index']")
        return candidate

    def _load_verify_candidates(self) -> None:
        if not self.load_verify_data:
            return

        paths = _resolve_verify_data_paths(self.load_verify_data)
        num_eligible = 0
        for path in paths:
            for sample in load_debug_rollout_data(str(path), rollout_id=0):
                candidate = self._prepare_verify_candidate(sample)
                if candidate is None:
                    continue
                # Saved verify files contain failures only: never reconstruct a
                # group mean from that selected subset, even if raw scores exist.
                candidate.metadata["history_baseline"] = get_verify_history_baseline(candidate.metadata)
                num_eligible += 1
                candidate.metadata["verify_data_origin"] = "loaded"
                candidate.metadata["verify_data_path"] = str(path)
                candidate.metadata["verify_source_group_key"] = (
                    str(path),
                    candidate.metadata["source_group_index"],
                )
                self._insert_verify_candidate(candidate, capture_rollout_id=None, loaded=True)

        if not num_eligible:
            raise ValueError("--load-verify-data did not contain any eligible failed kernel samples")
        logger.info(
            "Loaded %d/%d fixed verify candidates from %d file(s)",
            len(self.verify_data),
            num_eligible,
            len(paths),
        )

    def _insert_verify_candidate(
        self,
        candidate: Sample,
        *,
        capture_rollout_id: int | None,
        loaded: bool,
    ) -> VerifyCandidateEntry | None:
        group_difficulty, difficulty_weight = _verify_difficulty_weight(candidate)
        entry = VerifyCandidateEntry(
            sample=candidate,
            source_weight_version=self._source_kernel_weight_version(candidate),
            group_difficulty=group_difficulty,
            difficulty_weight=difficulty_weight,
            insertion_sequence=self._verify_insertion_sequence,
            capture_rollout_id=capture_rollout_id,
            loaded=loaded,
        )
        self._verify_insertion_sequence += 1
        insertion_index = bisect_right(
            self.verify_data,
            entry.retention_key,
            key=lambda existing: existing.retention_key,
        )
        self.verify_data.insert(insertion_index, entry)

        if math.isinf(self.verify_data_limit) or len(self.verify_data) <= int(self.verify_data_limit):
            return entry

        evicted = self.verify_data.pop(0)
        if evicted.capture_rollout_id is not None:
            self._dirty_capture_rollout_ids.add(evicted.capture_rollout_id)
        return entry if evicted is not entry else None

    def add_verify_candidates(self, samples: list[Sample], *, rollout_id: int | None = None) -> int:
        if not self.verify_capture_enabled:
            return 0
        if self.save_verify_data is not None and rollout_id is None:
            raise ValueError("rollout_id is required when --save-verify-data is configured")

        # Compute from full ordinary prompt/turn groups before selecting failures.
        raw_group_rewards: dict[tuple[object, object], list[object]] = {}
        for sample in samples:
            metadata = sample.metadata or {}
            if (
                metadata.get("role", "kernel") != "kernel"
                or metadata.get("verify_trajectory")
                or metadata.get("is_pad_turn")
                or sample.remove_sample
                or sample.status == Sample.Status.ABORTED
            ):
                continue
            key = sample.group_index, metadata.get("turn_idx")
            raw_group_rewards.setdefault(key, []).append(metadata.get("raw_task_reward"))
        group_baselines = {}
        for key, values in raw_group_rewards.items():
            if any(value is None for value in values):
                # Legacy/mixed groups must not average only the available subset.
                continue
            rewards = [float(value) for value in values]
            if not all(math.isfinite(reward) for reward in rewards):
                raise ValueError("verify capture requires finite raw_task_reward values")
            group_baselines[key] = math.fsum(reward / len(rewards) for reward in rewards)

        candidates = []
        for sample in samples:
            candidate = self._prepare_verify_candidate(sample, rollout_id=rollout_id)
            if candidate is not None:
                key = sample.group_index, candidate.metadata.get("turn_idx")
                if "history_baseline" not in candidate.metadata and key in group_baselines:
                    candidate.metadata["history_baseline"] = group_baselines[key]
                candidate.metadata["history_baseline"] = get_verify_history_baseline(candidate.metadata)
                candidates.append(candidate)

        with self._verify_buffer_lock:
            inserted = [
                self._insert_verify_candidate(
                    candidate,
                    capture_rollout_id=(
                        int(rollout_id) if self.save_verify_data is not None and rollout_id is not None else None
                    ),
                    loaded=False,
                )
                for candidate in candidates
            ]
            retained_entry_ids = {id(entry) for entry in self.verify_data}
            num_added = sum(entry is not None and id(entry) in retained_entry_ids for entry in inserted)
            current_weight_version = getattr(self.args, "gen_weight_version", None)
            if current_weight_version is None:
                current_weight_version = rollout_id
            if current_weight_version is not None:
                self._prune_verify_candidates_locked(
                    int(current_weight_version),
                    self.verify_version_lag,
                )
            if self.save_verify_data is not None:
                assert rollout_id is not None
                self._dirty_capture_rollout_ids.add(int(rollout_id))
        return num_added

    def save_captured_verify_data(self, rollout_id: int) -> int:
        """Persist one rollout and rewrite older shards changed by bounded retention."""

        if self.save_verify_data is None:
            return 0
        rollout_id = int(rollout_id)
        with self._verify_buffer_lock:
            if rollout_id not in self._dirty_capture_rollout_ids:
                return 0
            snapshots: dict[int, list[Sample]] = {}
            for entry in self.verify_data:
                if entry.loaded or entry.capture_rollout_id is None:
                    continue
                snapshots.setdefault(entry.capture_rollout_id, []).append(entry.sample)
            rollout_ids_to_save = sorted(self._dirty_capture_rollout_ids)
            for source_rollout_id in rollout_ids_to_save:
                save_debug_rollout_data(
                    self.save_verify_data,
                    snapshots.get(source_rollout_id, []),
                    rollout_id=source_rollout_id,
                    evaluation=False,
                )
            self._dirty_capture_rollout_ids.difference_update(rollout_ids_to_save)
            return len(snapshots.get(rollout_id, []))

    def begin_verify_capture(self, rollout_id: int) -> None:
        if self.save_verify_data is None:
            return
        with self._verify_buffer_lock:
            rollout_id = int(rollout_id)
            self._dirty_capture_rollout_ids.add(rollout_id)

    def _build_verify_sample(self, source: Sample, *, group_index: int, sample_index: int) -> Sample:
        if self.verify_prompt_template is None:
            raise ValueError(
                "verify rollout requires --verify-prompt-config-path to point to a Jinja template or a YAML "
                "config containing a named 'verify_response' template"
            )

        source_metadata = source.metadata if isinstance(source.metadata, dict) else {}
        env_result = source_metadata.get("env_result")
        if not isinstance(env_result, dict):
            raise ValueError("verify source requires sample.metadata['env_result']")
        if source.index is None:
            raise ValueError("verify source requires sample.index")
        kernel_submission = source_metadata.get("source_kernel_submission")
        if not isinstance(kernel_submission, str) or not kernel_submission:
            raise ValueError("verify source requires a normalized kernel submission")

        prompt = as_messages(source.prompt)
        prompt.extend(
            [
                {"role": "assistant", "content": kernel_submission},
                {"role": "user", "content": format_feedback(env_result, self.verify_prompt_template)},
            ]
        )
        metadata = {
            key: copy.deepcopy(source_metadata[key]) for key in _VERIFY_SOURCE_METADATA_KEYS if key in source_metadata
        }
        metadata.update(
            {
                "role": "verify",
                "history_baseline": get_verify_history_baseline(source_metadata),
                "source_kernel_index": source.index,
                "source_group_index": source_metadata.get("source_group_index", source.group_index),
                "source_turn_idx": source_metadata.get("turn_idx"),
                "source_kernel_weight_version": self._source_kernel_weight_version(source),
                "verify_source_env_result": copy.deepcopy(env_result),
            }
        )
        for key in ("precision", "augmentation", "ground_truth", "entry_point"):
            if key in source_metadata:
                metadata[key] = copy.deepcopy(source_metadata[key])
        return Sample(
            group_index=group_index,
            index=sample_index,
            prompt=prompt,
            label=copy.deepcopy(source.label),
            apply_chat_template_kwargs=copy.deepcopy(source.apply_chat_template_kwargs),
            metadata=metadata,
        )

    def _prune_verify_candidates_locked(self, current_weight_version: int, max_version_lag: int) -> int:
        minimum_source_version = int(current_weight_version) - max_version_lag
        num_pruned = 0
        for entry in self.verify_data:
            if entry.available and not entry.loaded and entry.source_weight_version < minimum_source_version:
                entry.available = False
                num_pruned += 1
        return num_pruned

    def prune_verify_candidates(self, *, current_weight_version: int, max_version_lag: int) -> int:
        if max_version_lag < 0:
            raise ValueError(f"max_version_lag must be non-negative, got {max_version_lag}")
        with self._verify_buffer_lock:
            return self._prune_verify_candidates_locked(current_weight_version, max_version_lag)

    def pop_verify_candidates(
        self,
        *,
        target_size: int,
        max_samples_per_group: int,
        current_weight_version: int,
        max_version_lag: int,
    ) -> list[Sample]:
        if max_version_lag < 0:
            raise ValueError(f"max_version_lag must be non-negative, got {max_version_lag}")
        with self._verify_buffer_lock:
            current_weight_version = int(current_weight_version)
            if self._verify_selection_weight_version != current_weight_version:
                self._verify_selection_weight_version = current_weight_version
                self._verify_selected_per_group.clear()
                for entry in self.verify_data:
                    if entry.loaded:
                        entry.available = True
            self._prune_verify_candidates_locked(current_weight_version, max_version_lag)
            selected = select_verify_entries(
                self.verify_data,
                target_size=target_size,
                max_samples_per_group=max_samples_per_group,
                current_weight_version=current_weight_version,
                rng=self._verify_rng,
                selected_per_group=self._verify_selected_per_group,
            )
            for entry in selected:
                entry.available = False
        return [entry.sample for entry in selected]

    def get_verify_samples(
        self,
        num_samples: int,
        *,
        max_samples_per_group: int,
        current_weight_version: int,
        max_version_lag: int,
    ) -> list[list[Sample]]:
        """Consume failed kernel turns and create fresh verify rollout groups."""

        if self.verify_prompt_template is None:
            raise ValueError(
                "verify rollout requires --verify-prompt-config-path to point to a Jinja template or a YAML "
                "config containing a named 'verify_response' template"
            )
        with self._sample_allocation_lock:
            return self._get_verify_samples_locked(
                num_samples,
                max_samples_per_group=max_samples_per_group,
                current_weight_version=current_weight_version,
                max_version_lag=max_version_lag,
            )

    def _get_verify_samples_locked(
        self,
        num_samples: int,
        *,
        max_samples_per_group: int,
        current_weight_version: int,
        max_version_lag: int,
    ) -> list[list[Sample]]:
        candidates = self.pop_verify_candidates(
            target_size=num_samples,
            max_samples_per_group=max_samples_per_group,
            current_weight_version=current_weight_version,
            max_version_lag=max_version_lag,
        )

        groups = []
        for source in candidates:
            group_index = self.sample_group_index
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                group.append(
                    self._build_verify_sample(
                        source,
                        group_index=group_index,
                        sample_index=self.sample_index,
                    )
                )
                self.sample_index += 1
            if getattr(self.args, "verify_advantage_baseline", "group") in {"anchor", "greedy-anchor"}:
                for sample in group:
                    sample.metadata["verify_anchor_index"] = self.sample_index
                self.sample_index += 1
                self.prepare_anchor(group)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def prepare_anchor(self, group: list[Sample]) -> str:
        """Register one direct prompt per group attempt, or reuse its unclaimed record."""
        first = group[0]
        with self._anchor_lock:
            key = first.metadata.get("verify_anchor_key")
            if any(sample.metadata.get("verify_anchor_key") != key for sample in group):
                raise ValueError("verify candidates must share one anchor key")
            if key in self.anchor_kv:
                if self.anchor_kv[key]["state"] != "pending":
                    raise ValueError("anchor attempt is already claimed")
                return key
            key = f"{first.group_index}:{uuid.uuid4().hex}"
            anchor = Sample(
                index=first.metadata["verify_anchor_index"],
                group_index=first.group_index,
                prompt=as_messages(first.prompt)[:-1],
                label=copy.deepcopy(first.label),
                apply_chat_template_kwargs=copy.deepcopy(first.apply_chat_template_kwargs),
                generate_function_path="examples.kernel_agent.generate_with_cuda_agent.generate_anchor",
                metadata={
                    name: copy.deepcopy(value)
                    for name, value in first.metadata.items()
                    if name in {"precision", "augmentation", "ground_truth", "entry_point", "verify_source_env_result"}
                },
            )
            anchor.metadata.update(role="kernel", verify_trajectory=True, verify_scoring_branch="anchor")
            self.anchor_kv[key] = {"state": "pending", "sample": anchor, "result": None}
            for sample in group:
                sample.metadata["verify_anchor_key"] = key
            return key

    def claim_anchor(self, key: str) -> Sample | None:
        with self._anchor_lock:
            record = self.anchor_kv[key]
            if record["state"] != "pending":
                return None
            record["state"] = "running"
            return record.pop("sample")

    def complete_anchor(self, key: str, result: dict) -> None:
        with self._anchor_lock:
            record = self.anchor_kv[key]
            if record["state"] != "running":
                raise ValueError("only a running anchor can publish its result")
            record.update(state="ready", result=copy.deepcopy(result))

    def get_anchor_result(self, key: str) -> dict:
        with self._anchor_lock:
            record = self.anchor_kv[key]
            if record["state"] != "ready":
                raise ValueError("anchor result is not ready")
            return copy.deepcopy(record["result"])

    def release_anchor(self, key: str) -> None:
        with self._anchor_lock:
            self.anchor_kv.pop(key, None)

    def get_verify_buffer_length(self) -> int:
        with self._verify_buffer_lock:
            return sum(entry.available for entry in self.verify_data)


def capture_verify_candidates(
    args, all_groups: list[list[Sample]], data_source, *, rollout_id: int | None = None
) -> None:
    """Capture all generated kernel failures at the ordinary rollout boundary."""

    capture_enabled = bool(getattr(args, "capture_verify_data", False))
    if not capture_enabled:
        return
    if rollout_id is None:
        raise ValueError("verify capture requires the current rollout_id")
    # The standard SGLang collector receives the bound get_samples callback.
    data_source = getattr(data_source, "__self__", data_source)
    add_candidates = getattr(data_source, "add_verify_candidates", None)
    save_candidates = getattr(data_source, "save_captured_verify_data", None)
    begin_capture = getattr(data_source, "begin_verify_capture", None)
    if not callable(add_candidates) or not callable(save_candidates) or not callable(begin_capture):
        raise TypeError("verify capture requires KernelAgentDataSource")

    begin_capture(rollout_id)
    for group in all_groups:
        annotate_group_difficulty(group)
        add_candidates(group, rollout_id=rollout_id)
    save_candidates(rollout_id)
