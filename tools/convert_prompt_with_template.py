#!/usr/bin/env python3
"""Convert parquet rows into Slime prompt/reward parquet using a Jinja template."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True, type=Path, help="Input parquet file.")
    parser.add_argument("--target-data", required=True, type=Path, help="Output parquet file.")
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="Input field rendered into the template as {{ problem }}.",
    )
    parser.add_argument("--template-path", required=True, type=Path, help="Jinja template path.")
    parser.add_argument(
        "--template-var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra scalar Jinja variable. Can be repeated.",
    )
    parser.add_argument(
        "--template-var-file",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Extra Jinja variable loaded from a text file. Can be repeated.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=None,
        help="Optional text file for printable samples. Defaults to TARGET_DATA with .sample.txt suffix.",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to write.")
    return parser.parse_args()


def parse_key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected KEY=VALUE, got {value!r}")
    key, parsed_value = value.split("=", 1)
    if not key:
        raise ValueError(f"Template variable key cannot be empty: {value!r}")
    return key, parsed_value


def parse_template_vars(args: argparse.Namespace) -> dict[str, str]:
    template_vars = dict(parse_key_value(item) for item in args.template_var)
    for item in args.template_var_file:
        key, path = parse_key_value(item)
        template_vars[key] = Path(path).read_text(encoding="utf-8")
    return template_vars


def get_nested(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def require_field(row: dict[str, Any], key: str) -> Any:
    value = get_nested(row, key)
    if value is None:
        available = ", ".join(row.keys())
        raise KeyError(f"Missing field {key!r}; available top-level fields: {available}")
    return value


def normalize_extra_info(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = dict(row.get("extra_info") or {})
    extra_info.pop("ability", None)
    extra_info.pop("data_source", None)
    return extra_info


def to_output_rows(
    raw_rows: list[dict[str, Any]], prompt_field: str, template: Template, template_vars: dict[str, str]
) -> list[dict[str, Any]]:
    output_rows = []
    warned_fallback = False
    for row in raw_rows:
        problem = require_field(row, prompt_field)
        if not isinstance(problem, str):
            raise TypeError(f"Field {prompt_field!r} must be a string, got {type(problem).__name__}")

        extra_info = row.get("extra_info") or {}
        data_source = row.get("data_source") or extra_info.get("data_source")
        ability = row.get("ability") or extra_info.get("ability")
        reward_model = row.get("reward_model") or {}
        ground_truth = row.get("ground_truth") or reward_model.get("ground_truth")
        if ground_truth is None:
            # No real reference: reward_model.ground_truth would silently become the
            # prompt-field text, which is only correct when --prompt-field already IS
            # the reference (e.g. ground_truth). Warn once so a bad source schema is
            # not masked.
            if not warned_fallback:
                print(
                    f"warning: rows lack 'ground_truth'/'reward_model.ground_truth'; "
                    f"falling back to --prompt-field {prompt_field!r} as reward_model.ground_truth.",
                    file=sys.stderr,
                )
                warned_fallback = True
            ground_truth = problem

        template_context = {
            "problem": problem,
            "reference_pytorch_code": problem,
            **template_vars,
        }
        output_rows.append(
            {
                "data_source": data_source,
                "prompt": [{"content": template.render(**template_context), "role": "user"}],
                "reward_model": {"ground_truth": ground_truth, "style": reward_model.get("style") or "rule"},
                "ability": ability,
                "extra_info": normalize_extra_info(row),
            }
        )
    return output_rows


def to_arrow_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.table(
        {
            "data_source": pa.array([row["data_source"] for row in rows], type=pa.large_string()),
            "prompt": pa.array(
                [row["prompt"] for row in rows],
                type=pa.list_(
                    pa.struct(
                        [
                            pa.field("content", pa.string()),
                            pa.field("role", pa.string()),
                        ]
                    )
                ),
            ),
            "reward_model": pa.array(
                [row["reward_model"] for row in rows],
                type=pa.struct(
                    [
                        pa.field("ground_truth", pa.string()),
                        pa.field("style", pa.string()),
                    ]
                ),
            ),
            "ability": pa.array([row["ability"] for row in rows], type=pa.large_string()),
            "extra_info": pa.array([row["extra_info"] for row in rows]),
        }
    )


def write_samples(rows: list[dict[str, Any]], sample_output: Path, num_samples: int) -> None:
    # Plain-text, human-reviewable dump: render the prompt content with real
    # newlines (not a JSON-escaped one-liner) so it can be read directly.
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    with sample_output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows[:num_samples]):
            extra_info = row.get("extra_info") or {}
            ident = ", ".join(f"{k}={extra_info[k]}" for k in ("problem_id", "name") if k in extra_info)
            f.write("=" * 72 + "\n")
            f.write(f"sample {index}" + (f"  [{ident}]" if ident else "") + "\n")
            f.write(f"data_source={row.get('data_source')}  ability={row.get('ability')}\n")
            f.write("=" * 72 + "\n")
            for msg in row.get("prompt") or []:
                f.write(f"---- prompt (role={msg.get('role')}) ----\n")
                f.write((msg.get("content") or "").rstrip() + "\n")
            ground_truth = (row.get("reward_model") or {}).get("ground_truth")
            if ground_truth is not None:
                f.write("---- reward_model.ground_truth ----\n")
                f.write(ground_truth.rstrip() + "\n")
            f.write("\n\n")


def main() -> None:
    args = parse_args()
    raw_rows = pq.read_table(args.raw_data).to_pylist()
    template = Template(args.template_path.read_text(encoding="utf-8"))
    template_vars = parse_template_vars(args)
    output_rows = to_output_rows(raw_rows, args.prompt_field, template, template_vars)
    output_table = to_arrow_table(output_rows)

    args.target_data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table, args.target_data)

    sample_output = args.sample_output or args.target_data.with_suffix(".sample.txt")
    write_samples(output_rows, sample_output, args.num_samples)

    print(f"wrote parquet: {args.target_data}")
    print(f"wrote samples: {sample_output}")
    print(f"rows: {output_table.num_rows}")
    print(f"columns: {', '.join(output_table.column_names)}")


if __name__ == "__main__":
    main()
