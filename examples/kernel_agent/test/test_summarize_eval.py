import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.kernel_agent.summarize_eval import summarize_with_best


def _sample(group_id, index, turn_idx, env_extra_info=None, is_pad_turn=False):
    metadata = {"turn_idx": turn_idx}
    if is_pad_turn:
        metadata.update({"is_pad_turn": True, "remove_reason": "pad_turn"})
    if env_extra_info is not None:
        metadata["env_extra_info"] = env_extra_info
        metadata["env_result"] = {
            "env_state": {
                "compiled": env_extra_info.get("compilation"),
                "correctness": env_extra_info.get("correctness"),
                "decoy_kernel": env_extra_info.get("decoy_kernel", False),
                "speedup": env_extra_info.get("speedup"),
            }
        }
    return {
        "group_id": group_id,
        "index": index,
        "remove_sample": is_pad_turn,
        "metadata": metadata,
    }


def test_best_metrics_group_by_trajectory_and_ignore_pad_turns():
    samples = [
        _sample(0, 0, 0, {"compilation": True, "correctness": False, "decoy_kernel": False, "speedup": 0.8}),
        _sample(0, 0, 1, {"compilation": True, "correctness": True, "decoy_kernel": False, "speedup": 1.3}),
        _sample(0, 0, 2, is_pad_turn=True),
        _sample(1, 1, 0, {"compilation": False, "correctness": False, "decoy_kernel": False, "speedup": None}),
        _sample(1, 1, 2, {"compilation": True, "correctness": True, "decoy_kernel": True, "speedup": 2.0}),
        _sample(2, 2, 0, is_pad_turn=True),
    ]

    with pytest.warns(RuntimeWarning, match="No Best\\* metrics found"):
        res = summarize_with_best(samples, (1.0, 1.2), max_turns=3)

    assert res["total"] == 6
    assert res["missing_env_result"] == 2
    assert res["correct_count"] == 1
    assert res["Correct"] == 1 / 6
    assert res["Fast@1"] == 1 / 6
    assert res["Fast@1.2"] == 1 / 6

    assert res["trajectory_total"] == 3
    assert res["trajectory_missing_env_result"] == 1
    assert res["best_compile_count"] == 2
    assert res["BestCompile"] == 2 / 3
    assert res["best_correct_count"] == 1
    assert res["BestCorrect"] == 1 / 3
    assert res["BestFast@1"] == 1 / 3
    assert res["BestFast@1.2"] == 1 / 3

    assert res["per_turn"][1]["Compile"] == 1 / 3
    assert res["per_turn"][1]["Correct"] == 0
    assert res["per_turn"][2]["Compile"] == 1 / 3
    assert res["per_turn"][2]["Correct"] == 1 / 3
    assert res["per_turn"][3]["Compile"] == 1 / 3
    assert res["per_turn"][3]["Correct"] == 0
    assert res["best_source"] == "computed_group"


def test_metadata_group_id_takes_precedence_over_stale_top_level_value():
    samples = [
        _sample(0, 0, 0, {"compilation": True, "correctness": False, "speedup": 0.8}),
        _sample(0, 0, 1, {"compilation": True, "correctness": True, "speedup": 1.3}),
        _sample(0, 0, 0, {"compilation": True, "correctness": True, "speedup": 1.1}),
        _sample(0, 0, 1, {"compilation": True, "correctness": False, "speedup": 0.9}),
    ]
    for sample, normalized_group_id in zip(samples, (0, 0, 1, 1), strict=True):
        sample["metadata"]["group_id"] = normalized_group_id

    with pytest.warns(RuntimeWarning, match="No Best\\* metrics found"):
        res = summarize_with_best(samples, (1.0, 1.2), max_turns=2)

    assert res["trajectory_total"] == 2
    assert res["per_turn"][1]["Correct"] == 1 / 2
    assert res["per_turn"][2]["Correct"] == 1 / 2
    assert res["BestCorrect"] == 1.0


def test_best_metrics_are_skipped_for_single_turn():
    samples = [
        _sample(0, 0, 0, {"compilation": True, "correctness": True, "decoy_kernel": False, "speedup": 1.3}),
    ]

    res = summarize_with_best(samples, (1.0, 1.2), max_turns=1)

    assert res["best_source"] == "skipped_single_turn"
    assert "BestCorrect" not in res


def test_best_metrics_are_skipped_without_group_id():
    samples = [
        _sample(None, 0, 0, {"compilation": True, "correctness": False, "decoy_kernel": False, "speedup": 0.8}),
        _sample(None, 0, 1, {"compilation": True, "correctness": True, "decoy_kernel": False, "speedup": 1.3}),
    ]

    with pytest.warns(RuntimeWarning) as records:
        res = summarize_with_best(samples, (1.0, 1.2), max_turns=2)

    warning_text = "\n".join(str(record.message) for record in records)
    assert "No Best* metrics found" in warning_text
    assert "No group_id found" in warning_text
    assert res["best_source"] == "skipped_no_group"
    assert "BestCorrect" not in res


def test_dump_best_metrics_take_precedence():
    samples = [
        _sample(0, 0, 0, {"compilation": True, "correctness": False, "decoy_kernel": False, "speedup": 0.8}),
        _sample(0, 0, 1, {"compilation": True, "correctness": True, "decoy_kernel": False, "speedup": 1.3}),
    ]

    res = summarize_with_best(
        samples,
        (1.0, 1.2),
        max_turns=2,
        dump_best_metrics={"eval/kb/kernel/trajectory/best_by_turn_2/correctness": 0.25},
    )

    assert res["best_source"] == "dump"
    assert res["dump_best_metrics"] == {"eval/kb/kernel/trajectory/best_by_turn_2/correctness": 0.25}
    assert "BestCorrect" not in res
