"""LASER-D budgets from pre-filter training rollouts (not separate monitor rollouts).

Method: https://arxiv.org/abs/2505.15612, Section 5 / Table 2.
The official implementation uses response-mask lengths and a correctness-gated
step bonus. We use the paper's minimum attainable correct counts, not the
official code's optional floor approximation. See docs for the monitoring change.
"""

from __future__ import annotations

import copy
import logging
import random
from bisect import bisect_right

from slime.utils.types import Sample

logger = logging.getLogger(__name__)
LASER_D_BUCKETS = ("hard", "medium", "easy")


def laser_d_bucket(correct: int, size: int, thresholds: list[float]) -> int:
    # Use the same inclusive lower bounds for monitoring and reward assignment.
    return bisect_right(thresholds, correct / size)


def select_laser_d_budgets(records, thresholds, lower: int, upper: int, interval: int) -> list[int]:
    """Find the smallest grid budget with ECR >= 1; empty buckets use the cap.

    For fixed group size K this is C_min * CDF(length). For variable-size
    kernel turn groups, average C_min(K) * coverage per prompt/turn instead of
    pretending every group still has the configured K samples.
    """
    candidates = list(range(lower, upper, interval)) + [upper]
    budgets = [upper] * len(LASER_D_BUCKETS)
    for bucket in range(len(LASER_D_BUCKETS)):
        bucket_records = [record for record in records if record["bucket"] == bucket]
        if not bucket_records:
            continue
        observations = []
        for record in bucket_records:
            lengths = record["lengths"]
            size = len(lengths)
            minimum_correct = next(
                count for count in range(1, size + 1) if laser_d_bucket(count, size, thresholds) == bucket
            )
            observations.append((minimum_correct, lengths))
        for budget in candidates:
            ecr = sum(
                minimum * sum(length <= budget for length in lengths) / len(lengths)
                for minimum, lengths in observations
            ) / len(observations)
            if ecr >= 1.0:
                budgets[bucket] = budget
                break
    return budgets


class LaserDBudgetController:
    """One collector-owned rollout transaction; checkpoint only compact state.

    A fixed budget snapshot scores this rollout. A bounded reservoir covers all
    completed groups, including dynamic-filter rejects. Only successful rollout
    completion commits monitoring/RNG state and new budgets to data_source.metadata.
    Background generation never reads or mutates this controller.
    """

    def __init__(self, args, data_source, rollout_id: int):
        if not isinstance(getattr(data_source, "metadata", None), dict):
            raise TypeError("LASER-D requires a checkpointable data source with dictionary metadata")
        self.args = args
        self.data_source = data_source
        self.rollout_id = int(rollout_id)
        self.thresholds = list(getattr(args, "difficulty_thresholds", [1 / 3, 2 / 3]))
        self.lower = int(getattr(args, "laser_d_min_length", 1024))
        self.upper = int(args.rollout_max_response_len)
        self.interval = int(getattr(args, "laser_d_length_interval", 1024))
        self.update_interval = int(getattr(args, "laser_d_update_interval", 20))
        self.monitor_groups = int(getattr(args, "laser_d_monitor_groups", 500))
        signature = [self.thresholds, self.lower, self.upper, self.interval, self.update_interval, self.monitor_groups]
        self.rng = random.Random(getattr(args, "rollout_seed", 0))
        self.state = copy.deepcopy(data_source.metadata.get("laser_d"))
        if self.state is None:
            self.state = {
                "format_version": 1,
                "signature": signature,
                "budgets": [self.lower] * len(LASER_D_BUCKETS),
                "records": [],
                "seen_groups": 0,
                "last_update_rollout_id": None,
                "last_completed_rollout_id": None,
            }
        elif self.state.get("format_version") != 1 or self.state.get("signature") != signature:
            raise ValueError("LASER-D checkpoint configuration differs from current budget/monitor parameters")
        last_completed = self.state["last_completed_rollout_id"]
        if last_completed is not None and self.rollout_id <= last_completed:
            raise ValueError("LASER-D rollout IDs must advance beyond the last checkpointed rollout")
        if "rng_state" in self.state:
            self.rng.setstate(self.state["rng_state"])
        self.budgets = list(self.state["budgets"])
        self.observed_groups = 0

    def observe(self, group: list[Sample]) -> None:
        from .kernel_reward import annotate_group_difficulty

        samples = [
            sample
            for sample in group
            if not sample.remove_sample
            and sample.status in {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}
            and (sample.metadata or {}).get("role", "kernel") == "kernel"
            and not (sample.metadata or {}).get("is_pad_turn")
        ]
        if not samples:
            return
        correct, size = annotate_group_difficulty(samples)
        bucket = laser_d_bucket(correct, size, self.thresholds)
        lengths = [int(sample.response_length) for sample in samples]
        if any(length < 0 for length in lengths):
            raise ValueError("LASER-D response lengths must be non-negative")
        for sample in samples:
            sample.metadata["laser_d"] = {
                "budget": self.budgets[bucket],
                "bucket": LASER_D_BUCKETS[bucket],
                "rollout_id": self.rollout_id,
                "budget_update_rollout_id": self.state["last_update_rollout_id"],
            }
        self.observed_groups += 1
        # With size=1, the hard bucket has no attainable positive correct count.
        # Such all-failed singleton groups cannot estimate a hard-bucket ECR.
        if not any(laser_d_bucket(count, size, self.thresholds) == bucket for count in range(1, size + 1)):
            return
        self.state["seen_groups"] += 1
        record = {"bucket": bucket, "lengths": lengths}
        records = self.state["records"]
        if len(records) < self.monitor_groups:
            records.append(record)
        else:
            index = self.rng.randrange(self.state["seen_groups"])
            if index < self.monitor_groups:
                records[index] = record

    def finish(self) -> dict[str, float]:
        last_update = self.state["last_update_rollout_id"]
        updated = bool(self.state["records"]) and (
            last_update is None or self.rollout_id - last_update >= self.update_interval
        )
        retained = len(self.state["records"])
        seen = self.state["seen_groups"]
        if updated:
            self.state["budgets"] = select_laser_d_budgets(
                self.state["records"], self.thresholds, self.lower, self.upper, self.interval
            )
            self.state["last_update_rollout_id"] = self.rollout_id
            self.state["records"] = []
            self.state["seen_groups"] = 0
            logger.info(
                "LASER-D rollout %s: used_budgets=%s next_budgets=%s monitor_groups=%s/%s",
                self.rollout_id,
                self.budgets,
                self.state["budgets"],
                retained,
                seen,
            )
        self.state["last_completed_rollout_id"] = self.rollout_id
        self.state["rng_state"] = self.rng.getstate()
        self.data_source.metadata["laser_d"] = copy.deepcopy(self.state)
        metrics = {
            "laser_d/budget_updated": int(updated),
            "laser_d/observed_groups": self.observed_groups,
            "laser_d/monitor_groups_retained": retained,
            "laser_d/monitor_groups_seen": seen,
        }
        for index, bucket in enumerate(LASER_D_BUCKETS):
            metrics[f"laser_d/{bucket}/budget_used"] = self.budgets[index]
            metrics[f"laser_d/{bucket}/budget_next"] = self.state["budgets"][index]
        return metrics
