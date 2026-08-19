"""CPU contracts for adaptive rollout over-sampling refills."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from slime.rollout.sglang_rollout import _get_over_sampling_fetch_size

FORMAL_LAUNCHER = REPO / "scripts" / "dsv4" / "run.deepseek_v4_flash.fp4.formal.rl.sh"
FULL_LOOP = REPO / "scripts" / "dsv4" / "_dsv4_launch_core.sh"


@pytest.mark.parametrize(
    ("accepted_groups", "expected_fetch_groups"),
    [
        (0, 32),
        (14, 8),
        (15, 4),
    ],
)
def test_adaptive_refill_fetches_configured_multiple_of_missing_prompt_groups(
    accepted_groups: int,
    expected_fetch_groups: int,
) -> None:
    args = Namespace(over_sampling_batch_size=32, over_sampling_refill_factor=4)

    assert _get_over_sampling_fetch_size(args, target_data_size=16, accepted_count=accepted_groups) == (
        expected_fetch_groups
    )


def test_adaptive_refill_is_capped_by_over_sampling_batch_size() -> None:
    args = Namespace(over_sampling_batch_size=20, over_sampling_refill_factor=4)

    assert _get_over_sampling_fetch_size(args, target_data_size=16, accepted_count=0) == 20


def test_missing_adaptive_factor_preserves_fixed_legacy_refill() -> None:
    args = Namespace(over_sampling_batch_size=32)

    assert _get_over_sampling_fetch_size(args, target_data_size=16, accepted_count=15) == 32


def test_explicit_none_adaptive_factor_preserves_fixed_legacy_refill() -> None:
    args = Namespace(over_sampling_batch_size=32, over_sampling_refill_factor=None)

    assert _get_over_sampling_fetch_size(args, target_data_size=16, accepted_count=14) == 32


def test_formal_launcher_wires_adaptive_refill_factor_two() -> None:
    formal_text = FORMAL_LAUNCHER.read_text()
    full_loop_text = FULL_LOOP.read_text()

    assert "export OVER_SAMPLING_REFILL_FACTOR=2" in formal_text
    assert '--over-sampling-refill-factor "${OVER_SAMPLING_REFILL_FACTOR}"' in full_loop_text


def test_formal_launcher_wires_static_dspark_block_size_three() -> None:
    formal_text = FORMAL_LAUNCHER.read_text()
    full_loop_text = FULL_LOOP.read_text()

    assert "export SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE=3" in formal_text
    assert '--sglang-speculative-dspark-block-size "${SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE}"' in full_loop_text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
