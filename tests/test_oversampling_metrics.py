# CPU-only unit test for rollout oversampling accounting (MetricGatherer + abort drain).
NUM_GPUS = 0

import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout.filter_hub.base_types import MetricGatherer, drain_pending_groups


def test_collect_emits_oversampling_keys_with_zero_defaults():
    m = MetricGatherer()
    out = m.collect()
    # The five oversampling counters are always present, even with no activity,
    # so downstream perf logging / wandb has stable keys from the first rollout.
    assert out["rollout/oversampling/rounds"] == 0
    assert out["rollout/oversampling/groups_submitted"] == 0
    assert out["rollout/oversampling/groups_finished"] == 0
    assert out["rollout/oversampling/aborted_groups"] == 0
    assert out["rollout/oversampling/aborted_response_tokens"] == 0
    # No drops -> no drop keys (unchanged legacy behavior).
    assert not any(k.startswith("rollout/dynamic_filter/drop_") for k in out)


def test_oversampling_submit_counts_rounds_and_groups():
    m = MetricGatherer()
    # Two submission rounds of over_sampling_batch_size = 32 each.
    m.on_oversampling_submit(32)
    m.on_oversampling_submit(32)
    out = m.collect()
    assert out["rollout/oversampling/rounds"] == 2
    assert out["rollout/oversampling/groups_submitted"] == 64


def test_group_finished_and_abort_accounting():
    m = MetricGatherer()
    m.on_oversampling_submit(32)
    # 28 groups finished (16 kept + 12 filter-dropped); the remaining 4 were
    # still in flight when enough valid groups were collected, then drained.
    for _ in range(28):
        m.on_group_finished()
    m.on_abort(num_groups=4, response_tokens=5000)
    out = m.collect()
    assert out["rollout/oversampling/groups_finished"] == 28
    assert out["rollout/oversampling/aborted_groups"] == 4
    assert out["rollout/oversampling/aborted_response_tokens"] == 5000
    # Accounting invariant: submitted == finished + aborted-in-flight.
    assert (
        out["rollout/oversampling/groups_submitted"]
        == out["rollout/oversampling/groups_finished"] + out["rollout/oversampling/aborted_groups"]
    )


def test_drop_metrics_coexist_with_oversampling_metrics():
    m = MetricGatherer()
    m.on_dynamic_filter_drop("zero_std_0.0")
    m.on_dynamic_filter_drop("zero_std_0.0")
    m.on_dynamic_filter_drop("zero_std_1.0")
    m.on_group_finished(num_groups=18)
    out = m.collect()
    assert out["rollout/dynamic_filter/drop_zero_std_0.0"] == 2
    assert out["rollout/dynamic_filter/drop_zero_std_1.0"] == 1
    assert out["rollout/oversampling/groups_finished"] == 18


def _mk_sample(response_length, response="resp", metadata=None):
    return SimpleNamespace(
        response_length=response_length,
        response=response,
        metadata={} if metadata is None else metadata,
    )


def test_drain_counts_every_group_even_when_task_raises():
    # A group that finishes returns its samples; a group whose generate task raised
    # during abort must STILL be counted as one aborted group (with 0 tokens), so the
    # invariant groups_submitted == groups_finished + aborted_groups holds.
    async def ok():
        return [_mk_sample(10), _mk_sample(20)]

    async def boom():
        raise RuntimeError("aborted mid-generation")

    async def run():
        tasks = {asyncio.ensure_future(ok()), asyncio.ensure_future(boom())}
        return await drain_pending_groups(
            tasks, partial_rollout=False, rollout_id=0, aborted_samples=[]
        )

    groups, tokens = asyncio.run(run())
    assert groups == 2  # both drained groups counted, including the failed one
    assert tokens == 30  # only the successful group contributes response tokens


def test_drain_partial_rollout_collects_and_tags_samples():
    async def ok():
        return [_mk_sample(5, metadata={})]

    async def run():
        collected = []
        groups, tokens = await drain_pending_groups(
            {asyncio.ensure_future(ok())},
            partial_rollout=True,
            rollout_id=7,
            aborted_samples=collected,
        )
        return groups, tokens, collected

    groups, tokens, collected = asyncio.run(run())
    assert groups == 1
    assert tokens == 5
    assert len(collected) == 1
    assert collected[0][0].metadata["start_rollout_id"] == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
