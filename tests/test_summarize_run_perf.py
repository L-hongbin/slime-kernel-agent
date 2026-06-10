import importlib.util
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "summarize_run_perf.py"
_SPEC = importlib.util.spec_from_file_location("summarize_run_perf", _SCRIPT_PATH)
summarize_run_perf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summarize_run_perf)


@pytest.mark.unit
def test_extracts_perf_metrics_and_formats_ascii_table(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\x1b[36m(actor)\x1b[0m perf 0: {'perf/step_time': 10.0, "
        "'perf/actor_train_tflops': 30.0}\n"
        "(rollout) perf 0: {'perf/rollout_time': 4.0}\n"
        "(actor) perf 1: {'perf/step_time': 14.0, 'perf/actor_train_tflops': 36.0}\n"
        "(rollout) perf 1: {'perf/rollout_time': 6.0}\n",
        encoding="utf-8",
    )

    values = summarize_run_perf.extract_metrics([log_path])
    rows = summarize_run_perf.summarize(values)
    table = summarize_run_perf.format_table(rows, precision=2)

    assert values["perf/step_time"] == [10.0, 14.0]
    assert values["perf/actor_train_tflops"] == [30.0, 36.0]
    assert values["perf/rollout_time"] == [4.0, 6.0]
    assert "| perf/step_time          | 2     | 12.00   | 12.00  | 14.00   |" in table
    assert "| perf/actor_train_tflops | 2     | 33.00   | 33.00  | 36.00   |" in table
    assert table.startswith("+")
