import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None = None


def call_dynamic_filter(fn, *args, **kwargs):
    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(*args, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        # oversampling accounting: quantify how much generation a rollout actually
        # paid for vs the rollout_batch_size it kept. "submitted" groups are issued
        # in chunks of over_sampling_batch_size; "finished" groups completed and were
        # either kept or dropped by the dynamic filter; "aborted" groups were still
        # in-flight when enough valid groups had been collected and got drained by
        # abort() -> pure wasted decode. response tokens of aborted groups quantify
        # that wasted decode volume.
        self._oversampling_rounds = 0
        self._groups_submitted = 0
        self._groups_finished = 0
        self._aborted_groups = 0
        self._aborted_response_tokens = 0

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def on_oversampling_submit(self, num_groups: int):
        """One inner-loop submission of `num_groups` (== over_sampling_batch_size) groups."""
        self._oversampling_rounds += 1
        self._groups_submitted += num_groups

    def on_group_finished(self, num_groups: int = 1):
        self._groups_finished += num_groups

    def on_abort(self, num_groups: int, response_tokens: int):
        self._aborted_groups += num_groups
        self._aborted_response_tokens += response_tokens

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        metrics["rollout/oversampling/rounds"] = self._oversampling_rounds
        metrics["rollout/oversampling/groups_submitted"] = self._groups_submitted
        metrics["rollout/oversampling/groups_finished"] = self._groups_finished
        metrics["rollout/oversampling/aborted_groups"] = self._aborted_groups
        metrics["rollout/oversampling/aborted_response_tokens"] = self._aborted_response_tokens
        return metrics


async def drain_pending_groups(pendings, *, partial_rollout, rollout_id, aborted_samples):
    """Drain the in-flight generate-group tasks left after an abort and account the
    wasted oversampling decode.

    Every drained group counts as one aborted group -- even if its task raised, in which
    case it contributes 0 response tokens. Counting failed tasks too keeps the invariant
    ``groups_submitted == groups_finished + aborted_groups`` intact. Response tokens are
    summed only from groups that returned successfully. When ``partial_rollout`` is set,
    the partial groups are also collected into ``aborted_samples`` for re-queueing.

    Returns ``(num_aborted_groups, aborted_response_tokens)``.
    """
    num_groups = 0
    response_tokens = 0
    partial_count = 0
    while pendings:
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            # Count the drained group before touching its result so a task that raised
            # mid-generation is still attributed to aborted (wasted) decode.
            num_groups += 1
            try:
                group = task.result()
            except Exception as e:  # noqa: BLE001 - drained task may have failed; still counted
                logger.warning(f"aborted generate task raised during drain: {e}")
                continue
            response_tokens += sum((getattr(sample, "response_length", 0) or 0) for sample in group)

            if not partial_rollout:
                continue

            # for partial rollout, collect the partial samples into the data buffer
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            partial_count += len(group)

    if partial_rollout:
        logger.info(f"Collected {partial_count} partial samples into the data buffer")

    return num_groups, response_tokens
