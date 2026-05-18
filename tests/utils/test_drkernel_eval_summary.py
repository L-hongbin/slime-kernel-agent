import json
from pathlib import Path

from slime_plugins.drkernel.eval_summary import load_records, summarize_records


def test_summarize_kernelgym_eval_rates_for_nested_and_flat_records():
    records = [
        {"metadata": {"kernelgym": {"response": {"compiled": True, "correctness": True, "speedup": 1.3}}}},
        {"kernelgym": {"response": {"compiled": True, "correctness": True, "speedup": 1.1}}},
        {"response": {"compiled": True, "correctness": False, "speedup": 2.0}},
        {"compiled": False, "correctness": False, "speedup": 0.0},
        {
            "metadata": {
                "kernelgym": {
                    "response": {"compiled": True, "correctness": True, "decoy_kernel": True, "speedup": 3.0}
                }
            }
        },
        {"metadata": {}},
    ]

    summary = summarize_records(records)

    assert summary["total"] == 6
    assert summary["evaluated"] == 5
    assert summary["missing_response"] == 1
    assert summary["compiled_count"] == 4
    assert summary["compile_rate"] == 4 / 6
    assert summary["raw_correct_count"] == 3
    assert summary["raw_correctness_rate"] == 3 / 6
    assert summary["correct_count"] == 2
    assert summary["correctness_rate"] == 2 / 6
    assert summary["decoy_count"] == 1
    assert summary["fast@1.0_count"] == 2
    assert summary["fast@1.0_rate"] == 2 / 6
    assert summary["fast@1.0_correct_rate"] == 1.0
    assert summary["fast@1.2_count"] == 1
    assert summary["fast@1.2_rate"] == 1 / 6
    assert summary["fast@1.2_correct_rate"] == 1 / 2


def test_load_records_accepts_jsonl_kernelgym_responses(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    rows = [
        {"metadata": {"kernelgym": {"response": {"compiled": True, "correctness": True, "speedup": "1.25"}}}},
        {"response": {"compiled": True, "correctness": True, "is_decoy_kernel": True, "performance": 2.0}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_records(load_records(path))

    assert summary["total"] == 2
    assert summary["compile_rate"] == 1.0
    assert summary["correctness_rate"] == 0.5
    assert summary["fast@1.2_count"] == 1
