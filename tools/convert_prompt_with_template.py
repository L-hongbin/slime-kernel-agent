#!/usr/bin/env python3
"""Convert a prompt dataset by rendering a Jinja template."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from jinja2 import Template

DEFAULT_PROMPT_FIELD = "prompt"
DEFAULT_PROBLEM_FIELD = "reward_model.ground_truth"
DEFAULT_ID_FIELD = "uuid"
DEFAULT_TEMPLATE_NAME = "first_turn"
DEFAULT_TEMPLATE_VARIABLE = "problem"
JINJA_SUFFIXES = {".jinja", ".j2"}
YAML_SUFFIXES = {".yaml", ".yml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Jinja prompt template over a parquet/jsonl dataset and write a converted parquet/jsonl file. "
            "Pass --template-path with a .jinja/.j2 file to read Jinja directly, or a .yaml/.yml file to read a "
            "slime multi-turn prompt config."
        )
    )
    parser.add_argument("--raw-data", required=True, help="Input dataset path. Supports .parquet and .jsonl.")
    parser.add_argument("--target-data", required=True, help="Output dataset path. Supports .parquet and .jsonl.")
    parser.add_argument(
        "--prompt-field",
        default=DEFAULT_PROMPT_FIELD,
        help=f"Dataset field to replace with the rendered prompt. Default: {DEFAULT_PROMPT_FIELD}.",
    )
    parser.add_argument(
        "--problem-field",
        default=DEFAULT_PROBLEM_FIELD,
        help=(
            "Dot path for the source value passed to the Jinja template variable. "
            f"Default: {DEFAULT_PROBLEM_FIELD}."
        ),
    )
    parser.add_argument(
        "--id-field",
        default=DEFAULT_ID_FIELD,
        help=(
            "Dot path for the source value used to populate the 'uuid' field in 'extra_info'. "
            "If missing, a synthetic uuid is generated. Default: 'id'."
        ),
    )
    parser.add_argument(
        "--template-path",
        required=True,
        help="Template path. .jinja/.j2 files are read directly; .yaml/.yml files are read as slime prompt YAML.",
    )
    parser.add_argument(
        "--string-prompt",
        action="store_true",
        help="Write the rendered prompt as a plain string instead of [{'role': 'user', 'content': text}].",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f"Unsupported input format: {path}. Supported formats are .parquet and .jsonl.")


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
    raise ValueError(f"Unsupported output format: {path}. Supported formats are .parquet and .jsonl.")


def get_nested_value(record: dict[str, Any], field_path: str) -> Any:
    value: Any = record
    current_path = []
    for part in field_path.split("."):
        current_path.append(part)
        if isinstance(value, dict):
            if part not in value:
                raise KeyError(
                    f"Field path {field_path!r} is missing at {'.'.join(current_path)!r}. "
                    f"Available fields: {sorted(value)}"
                )
            value = value[part]
            continue
        if isinstance(value, list) and part.isdigit():
            index = int(part)
            try:
                value = value[index]
            except IndexError as e:
                raise IndexError(f"Field path {field_path!r} index {index} is out of range.") from e
            continue
        raise TypeError(
            f"Cannot descend into {'.'.join(current_path)!r} while resolving {field_path!r}; "
            f"current value has type {type(value).__name__}."
        )
    return value


def stringify_problem(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(value)


def iter_yaml_templates(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    per_turn_prompts = config.get("per_turn_prompts", []) or []
    if not isinstance(per_turn_prompts, list):
        raise ValueError("template YAML field 'per_turn_prompts' must be a list.")
    return per_turn_prompts


def load_template_from_yaml(yaml_path: Path, template_name: str) -> Template:
    with yaml_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for item in iter_yaml_templates(config):
        if item.get("name") == template_name:
            template = item.get("template")
            if not template:
                raise ValueError(
                    f"Template {template_name!r} exists in {yaml_path}, but its 'template' field is empty."
                )
            return Template(str(template))

    raise KeyError(f"Template {template_name!r} not found in {yaml_path}.")


def load_template(template_path: Path) -> Template:
    suffix = template_path.suffix.lower()
    if suffix in YAML_SUFFIXES:
        return load_template_from_yaml(template_path, DEFAULT_TEMPLATE_NAME)
    if suffix in JINJA_SUFFIXES:
        return Template(template_path.read_text(encoding="utf-8"))
    raise ValueError(
        f"Unsupported template format: {template_path}. Supported suffixes are "
        f"{sorted(YAML_SUFFIXES | JINJA_SUFFIXES)}."
    )


def convert_records(
    records: list[dict[str, Any]],
    prompt_field: str,
    problem_field: str,
    template: Template,
    string_prompt: bool,
    id_field: str,
) -> list[dict[str, Any]]:
    converted = []
    for i, record in enumerate(records):
        row = dict(record)
        problem = stringify_problem(get_nested_value(row, problem_field))
        rendered = template.render(**{DEFAULT_TEMPLATE_VARIABLE: problem})
        row[prompt_field] = rendered if string_prompt else [{"role": "user", "content": rendered}]
        if "reward_model" not in row:
            row["reward_model"] = {"ground_truth": problem}
        if "extra_info" not in row:
            row["extra_info"] = {"uuid": row.get(id_field, f"sample_{i}")}
        converted.append(row)
    return converted


def main() -> None:
    args = parse_args()
    raw_data = Path(args.raw_data)
    target_data = Path(args.target_data)
    template_path = Path(args.template_path)

    template = load_template(template_path)
    records = read_records(raw_data)
    converted = convert_records(
        records,
        prompt_field=args.prompt_field,
        problem_field=args.problem_field,
        template=template,
        string_prompt=args.string_prompt,
        id_field=args.id_field,
    )
    write_records(target_data, converted)
    print(f"Converted {len(converted)} records: {raw_data} -> {target_data}")
    print(f"Template path: {template_path}")
    print("Example converted prompt:", converted[0][args.prompt_field])


if __name__ == "__main__":
    main()
