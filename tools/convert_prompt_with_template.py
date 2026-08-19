#!/usr/bin/env python3
"""Render Jinja prompts over parquet/jsonl data, with legacy Slime-schema output support."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from jinja2 import Template

DEFAULT_PROMPT_FIELD = "prompt"
DEFAULT_ID_FIELD = "uuid"
DEFAULT_TEMPLATE_NAME = "first_turn"
JINJA_SUFFIXES = {".jinja", ".j2"}
YAML_SUFFIXES = {".yaml", ".yml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True, type=Path, help="Input .parquet or .jsonl dataset.")
    parser.add_argument("--target-data", required=True, type=Path, help="Output .parquet or .jsonl dataset.")
    parser.add_argument(
        "--prompt-field",
        default=DEFAULT_PROMPT_FIELD,
        help=(
            "Legacy mode: source field rendered as {{ problem }}. Generic mode (--problem-field set): "
            "destination field replaced with the rendered prompt."
        ),
    )
    parser.add_argument(
        "--problem-field",
        default=None,
        help=(
            "Enable generic in-place conversion and read the template input from this dot path "
            "(for main-branch behavior, use reward_model.ground_truth)."
        ),
    )
    parser.add_argument("--id-field", default=DEFAULT_ID_FIELD, help="Dot path used for a synthetic extra_info.uuid.")
    parser.add_argument("--template-path", required=True, type=Path, help="A .jinja/.j2 template or prompt YAML.")
    parser.add_argument(
        "--template-name",
        default=DEFAULT_TEMPLATE_NAME,
        help="Name selected from YAML per_turn_prompts. Default: first_turn.",
    )
    parser.add_argument(
        "--string-prompt",
        action="store_true",
        help="Generic mode: write a plain string instead of a one-message chat prompt.",
    )
    parser.add_argument(
        "--template-var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra scalar Jinja variable; repeat as needed.",
    )
    parser.add_argument(
        "--template-var-file",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Extra Jinja variable loaded from a text file; repeat as needed.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=None,
        help="Reviewable text samples; defaults to TARGET_DATA with .sample.txt suffix.",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of reviewable samples to write.")
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


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f"Unsupported input format: {path}. Expected .parquet or .jsonl.")


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        pq.write_table(pa.Table.from_pylist(records), path)
        return
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    raise ValueError(f"Unsupported output format: {path}. Expected .parquet or .jsonl.")


def get_nested_value(record: dict[str, Any], field_path: str) -> Any:
    value: Any = record
    current_path = []
    for part in field_path.split("."):
        current_path.append(part)
        if isinstance(value, dict):
            if part not in value:
                raise KeyError(
                    f"Field path {field_path!r} is missing at {'.'.join(current_path)!r}; "
                    f"available fields: {sorted(value)}"
                )
            value = value[part]
        elif isinstance(value, list) and part.isdigit():
            index = int(part)
            try:
                value = value[index]
            except IndexError as exc:
                raise IndexError(f"Field path {field_path!r} index {index} is out of range.") from exc
        else:
            raise TypeError(
                f"Cannot descend into {'.'.join(current_path)!r} while resolving {field_path!r}; "
                f"current value has type {type(value).__name__}."
            )
    return value


def get_nested_or_default(record: dict[str, Any], field_path: str, default: Any = None) -> Any:
    try:
        return get_nested_value(record, field_path)
    except (KeyError, IndexError, TypeError):
        return default


def stringify_problem(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in value)
    return str(value)


def iter_yaml_templates(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    per_turn_prompts = config.get("per_turn_prompts", []) or []
    if not isinstance(per_turn_prompts, list):
        raise ValueError("template YAML field 'per_turn_prompts' must be a list.")
    return per_turn_prompts


def load_template(template_path: Path, template_name: str = DEFAULT_TEMPLATE_NAME) -> Template:
    suffix = template_path.suffix.lower()
    if suffix in JINJA_SUFFIXES:
        return Template(template_path.read_text(encoding="utf-8"))
    if suffix in YAML_SUFFIXES:
        with template_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        for item in iter_yaml_templates(config):
            if item.get("name") == template_name:
                template = item.get("template")
                if not template:
                    raise ValueError(f"Template {template_name!r} in {template_path} is empty.")
                return Template(str(template))
        raise KeyError(f"Template {template_name!r} not found in {template_path}.")
    raise ValueError(f"Unsupported template suffix {suffix!r}; expected Jinja or YAML.")


def normalize_extra_info(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = dict(row.get("extra_info") or {})
    extra_info.pop("ability", None)
    extra_info.pop("data_source", None)
    return extra_info


def to_slime_output_rows(
    raw_rows: list[dict[str, Any]], source_field: str, template: Template, template_vars: dict[str, str]
) -> list[dict[str, Any]]:
    output_rows = []
    warned_fallback = False
    for row in raw_rows:
        problem = get_nested_value(row, source_field)
        if not isinstance(problem, str):
            raise TypeError(f"Field {source_field!r} must be a string, got {type(problem).__name__}")

        extra_info = row.get("extra_info") or {}
        reward_model = row.get("reward_model") or {}
        ground_truth = row.get("ground_truth") or reward_model.get("ground_truth")
        if ground_truth is None:
            if not warned_fallback:
                print(
                    "warning: rows lack ground_truth/reward_model.ground_truth; "
                    f"falling back to source field {source_field!r}.",
                    file=sys.stderr,
                )
                warned_fallback = True
            ground_truth = problem

        context = {"problem": problem, "reference_pytorch_code": problem, **template_vars}
        output_rows.append(
            {
                "data_source": row.get("data_source") or extra_info.get("data_source"),
                "prompt": [{"content": template.render(**context), "role": "user"}],
                "reward_model": {"ground_truth": ground_truth, "style": reward_model.get("style") or "rule"},
                "ability": row.get("ability") or extra_info.get("ability"),
                "extra_info": normalize_extra_info(row),
            }
        )
    return output_rows


def convert_records(
    records: list[dict[str, Any]],
    *,
    prompt_field: str,
    problem_field: str,
    template: Template,
    template_vars: dict[str, str],
    string_prompt: bool,
    id_field: str,
) -> list[dict[str, Any]]:
    converted = []
    for index, record in enumerate(records):
        row = dict(record)
        problem = stringify_problem(get_nested_value(row, problem_field))
        rendered = template.render(problem=problem, reference_pytorch_code=problem, **template_vars)
        row[prompt_field] = rendered if string_prompt else [{"role": "user", "content": rendered}]
        row.setdefault("reward_model", {"ground_truth": problem})
        row.setdefault("extra_info", {"uuid": get_nested_or_default(row, id_field, f"sample_{index}")})
        converted.append(row)
    return converted


def to_arrow_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.table(
        {
            "data_source": pa.array([row["data_source"] for row in rows], type=pa.large_string()),
            "prompt": pa.array(
                [row["prompt"] for row in rows],
                type=pa.list_(pa.struct([pa.field("content", pa.string()), pa.field("role", pa.string())])),
            ),
            "reward_model": pa.array(
                [row["reward_model"] for row in rows],
                type=pa.struct([pa.field("ground_truth", pa.string()), pa.field("style", pa.string())]),
            ),
            "ability": pa.array([row["ability"] for row in rows], type=pa.large_string()),
            "extra_info": pa.array([row["extra_info"] for row in rows]),
        }
    )


def write_samples(rows: list[dict[str, Any]], sample_output: Path, num_samples: int) -> None:
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    with sample_output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows[:num_samples]):
            f.write("=" * 72 + "\n")
            f.write(f"sample {index}\n")
            f.write("=" * 72 + "\n")
            prompt = row.get("prompt")
            if isinstance(prompt, list):
                for message in prompt:
                    f.write(f"---- prompt (role={message.get('role')}) ----\n")
                    f.write(str(message.get("content") or "").rstrip() + "\n")
            else:
                f.write("---- prompt ----\n" + str(prompt or "").rstrip() + "\n")
            ground_truth = (row.get("reward_model") or {}).get("ground_truth")
            if ground_truth is not None:
                f.write("---- reward_model.ground_truth ----\n" + str(ground_truth).rstrip() + "\n")
            f.write("\n")


def main() -> None:
    args = parse_args()
    records = read_records(args.raw_data)
    template = load_template(args.template_path, args.template_name)
    template_vars = parse_template_vars(args)

    if args.problem_field is None:
        converted = to_slime_output_rows(records, args.prompt_field, template, template_vars)
        args.target_data.parent.mkdir(parents=True, exist_ok=True)
        if args.target_data.suffix == ".parquet":
            pq.write_table(to_arrow_table(converted), args.target_data)
        else:
            write_records(args.target_data, converted)
    else:
        converted = convert_records(
            records,
            prompt_field=args.prompt_field,
            problem_field=args.problem_field,
            template=template,
            template_vars=template_vars,
            string_prompt=args.string_prompt,
            id_field=args.id_field,
        )
        write_records(args.target_data, converted)

    sample_output = args.sample_output or args.target_data.with_suffix(".sample.txt")
    write_samples(converted, sample_output, args.num_samples)
    print(f"Converted {len(converted)} records: {args.raw_data} -> {args.target_data}")
    print(f"Template path: {args.template_path}")
    print(f"Review samples: {sample_output}")


if __name__ == "__main__":
    main()
