"""CPU regression tests for V4 BSHD actual-length microbatch padding."""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.backends.megatron_utils import cp_utils
from slime.backends.megatron_utils import data as data_module
from slime.backends.megatron_utils.cp_utils import CP_PARTITION_CONTIGUOUS, CP_PARTITION_ZIGZAG
from slime.backends.megatron_utils.data import (
    DataIterator,
    compute_bshd_max_seq_lens,
    get_batch,
    summarize_bshd_padding,
)

NUM_GPUS = 0


@pytest.mark.parametrize(
    ("total_length", "expected_width"),
    [
        (5706, 6144),
        (16382, 16384),
    ],
)
def test_cp2_contiguous_uses_production_1024_token_buckets(total_length, expected_width):
    assert compute_bshd_max_seq_lens(
        [total_length],
        [[0]],
        pad_size=1024,
        cp_size=2,
        cp_partition_mode=CP_PARTITION_CONTIGUOUS,
        pipeline_model_parallel_size=1,
    ) == [expected_width]


def test_pp1_uses_each_microbatch_real_max_with_cp2_alignment():
    # CP2 contiguous requires global S % (2 * 128) == 0. Samples sharing an
    # MBS share its maximum; independent MBSes no longer inherit rollout max.
    assert compute_bshd_max_seq_lens(
        [257, 500, 15, 513],
        [[0, 1], [2], [3]],
        pad_size=128,
        cp_size=2,
        cp_partition_mode=CP_PARTITION_CONTIGUOUS,
        pipeline_model_parallel_size=1,
    ) == [512, 512, 256, 768]


def test_padding_follows_reordered_schedule_across_training_steps():
    # This is the actor's flattened local schedule for two training steps with
    # num_microbatches=[2, 1]. Dynamic/balanced scheduling may reorder samples;
    # widths are written back by local sample index so later loss/CP offsets stay
    # aligned even across the optimizer-step boundary.
    assert compute_bshd_max_seq_lens(
        [600, 100, 300, 17],
        [[2], [0], [3, 1]],
        pad_size=128,
        cp_size=2,
        cp_partition_mode=CP_PARTITION_CONTIGUOUS,
        pipeline_model_parallel_size=1,
    ) == [768, 256, 512, 256]


def test_pp_greater_than_one_keeps_safe_rollout_wide_width():
    # V4's PP adapter allocates one communication shape per pipeline schedule;
    # per-MBS widths are unsafe until schedules are split by width.
    assert compute_bshd_max_seq_lens(
        [257, 500, 15, 513],
        [[0, 1], [2], [3]],
        pad_size=128,
        cp_size=2,
        cp_partition_mode=CP_PARTITION_CONTIGUOUS,
        pipeline_model_parallel_size=2,
    ) == [768, 768, 768, 768]


def test_zigzag_keeps_base_padding_granularity():
    assert compute_bshd_max_seq_lens(
        [129, 257],
        [[0], [1]],
        pad_size=128,
        cp_size=2,
        cp_partition_mode=CP_PARTITION_ZIGZAG,
    ) == [256, 384]


def test_get_batch_consumes_the_current_microbatch_width_on_cpu(monkeypatch):
    total_lengths = [257, 500, 15]
    schedule = [[0, 1], [2]]
    max_seq_lens = compute_bshd_max_seq_lens(
        total_lengths,
        schedule,
        pad_size=128,
        cp_size=2,
        cp_partition_mode=CP_PARTITION_CONTIGUOUS,
    )
    rollout_data = {
        "tokens": [torch.arange(length) for length in total_lengths],
        "loss_masks": [torch.ones(length - 1, dtype=torch.int) for length in total_lengths],
        "total_lengths": total_lengths,
        "response_lengths": [length - 1 for length in total_lengths],
        "max_seq_lens": max_seq_lens,
    }
    iterator = DataIterator(rollout_data, schedule)

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0)
    old_mode = cp_utils.get_cp_partition_mode()
    cp_utils.set_cp_partition_mode(CP_PARTITION_CONTIGUOUS)
    try:
        keys = ["tokens", "loss_masks", "total_lengths", "response_lengths", "max_seq_lens"]
        first = get_batch(iterator, keys, pad_multiplier=128, qkv_format="bshd")
        second = get_batch(iterator, keys, pad_multiplier=128, qkv_format="bshd")
    finally:
        cp_utils.set_cp_partition_mode(old_mode)

    assert first["max_seq_lens"] == [512, 512]
    assert first["tokens"].shape == first["full_loss_masks"].shape == (2, 256)
    assert second["max_seq_lens"] == [256]
    assert second["tokens"].shape == second["full_loss_masks"].shape == (1, 128)


def test_padding_summary_reports_widths_and_saved_slots():
    summary = summarize_bshd_padding(
        [5706, 16382, 6000],
        [6144, 16384, 6144],
    )

    assert summary["samples"] == 3
    assert summary["unique_widths"] == 2
    assert summary["width_hist"] == "6144:2,16384:1"
    assert summary["raw_slots"] == 28088
    assert summary["padded_slots"] == 28672
    assert summary["rollout_wide_slots"] == 49152
    assert summary["padding_overhead_pct"] == pytest.approx(100 * 584 / 28088)
    assert summary["saved_vs_rollout_wide_pct"] == pytest.approx(41.6667, rel=1e-4)


@pytest.mark.parametrize(
    ("schedule", "error", "match"),
    [
        ([[0], [0, 1]], ValueError, "more than once"),
        ([[0]], ValueError, "missing sample indices"),
        ([[0], [2]], IndexError, "outside"),
        ([[0], []], ValueError, "is empty"),
    ],
)
def test_schedule_must_be_an_exact_partition(schedule, error, match):
    with pytest.raises(error, match=match):
        compute_bshd_max_seq_lens(
            [128, 256],
            schedule,
            pad_size=128,
            cp_size=2,
            cp_partition_mode=CP_PARTITION_CONTIGUOUS,
        )


def test_actor_wires_dynamic_widths_from_the_real_microbatch_schedule():
    actor_source = (
        Path(__file__).resolve().parents[2] / "slime" / "backends" / "megatron_utils" / "actor.py"
    ).read_text()
    assert "compute_bshd_max_seq_lens(" in actor_source
    assert 'rollout_data["micro_batch_indices"]' in actor_source
    assert "pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size()" in actor_source
    assert "V4_ACTUAL_LENGTH_PADDING" in actor_source
    assert "microbatch_width_order" in actor_source
    assert "microbatch_widths_descending" in actor_source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
