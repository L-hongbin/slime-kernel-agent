#!/usr/bin/env python3
"""Build a raw KernelBench level1 parquet from a directory of reference .py files.

Used to produce the OLD (pre-2025-07-02 scale-up) small-size KernelBench level1
raw source, schema-identical to Data/kernelbench-level1-validation/train.parquet
(columns: extra_info struct + ground_truth string). Render into the MusaCoder
load_inline prompt afterwards with tools/convert_prompt_with_template.py.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA_SOURCE = "kernelbench_level1_validation"
ABILITY = "kernel_optimization"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level1-dir", required=True, type=Path, help="Dir of NN_*.py reference files.")
    p.add_argument("--target-data", required=True, type=Path, help="Output raw parquet.")
    p.add_argument("--data-source", default=DATA_SOURCE, help="extra_info.data_source label.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(
        args.level1_dir.glob("*.py"),
        key=lambda f: int(re.match(r"(\d+)_", f.name).group(1)),
    )
    if not files:
        raise SystemExit(f"no .py files under {args.level1_dir}")

    extra_info, ground_truth = [], []
    for f in files:
        pid = int(re.match(r"(\d+)_", f.name).group(1))
        name = f.stem  # filename without .py, e.g. "45_Average_Pooling_2D"
        extra_info.append(
            {
                "ability": ABILITY,
                "data_source": args.data_source,
                "difficulty": None,
                "name": name,
                "problem_id": pid,
            }
        )
        ground_truth.append(f.read_text(encoding="utf-8"))

    extra_info_type = pa.struct(
        [
            pa.field("ability", pa.string()),
            pa.field("data_source", pa.string()),
            pa.field("difficulty", pa.null()),
            pa.field("name", pa.string()),
            pa.field("problem_id", pa.int64()),
        ]
    )
    table = pa.table(
        {
            "extra_info": pa.array(extra_info, type=extra_info_type),
            "ground_truth": pa.array(ground_truth, type=pa.string()),
        }
    )
    args.target_data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.target_data)
    print(f"wrote {args.target_data} rows={table.num_rows}")
    print(f"problem_ids: {[e['problem_id'] for e in extra_info[:5]]} ... {[e['problem_id'] for e in extra_info[-3:]]}")


if __name__ == "__main__":
    main()
