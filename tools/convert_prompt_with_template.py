#!/usr/bin/env python3
"""Convert prompt records into the standard Slime prompt/reward schema."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from jinja2 import Template

DEFAULT_PROBLEM_FIELD = "reward_model.ground_truth"
DEFAULT_TEMPLATE_NAME = "first_turn"
JINJA_SUFFIXES = {".jinja", ".j2"}
YAML_SUFFIXES = {".yaml", ".yml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True, type=Path, help="Input .parquet or .jsonl file.")
    parser.add_argument("--target-data", required=True, type=Path, help="Output .parquet or .jsonl file.")
    parser.add_argument(
        "--problem-field",
        default=DEFAULT_PROBLEM_FIELD,
        help=(
            "Dot path of the input value rendered as {{ problem }} and {{ reference_pytorch_code }}. "
            f"Default: {DEFAULT_PROBLEM_FIELD}."
        ),
    )
    parser.add_argument(
        "--template-path",
        required=True,
        type=Path,
        help="Jinja template or Slime multi-turn prompt YAML containing a first_turn template.",
    )
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
        metavar="KEY=PATH_OR_URL",
        help=(
            "Extra Jinja variable loaded from a local text file or an HTTP(S) URL response body. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--template-var-url-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each HTTP(S) --template-var-file source. Default: 30.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=None,
        help="Optional human-readable sample file. Defaults to TARGET_DATA with .sample.txt suffix.",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of human-readable samples to write.")
    return parser.parse_args()


def parse_key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected KEY=VALUE, got {value!r}")
    key, parsed_value = value.split("=", 1)
    if not key:
        raise ValueError(f"Template variable key cannot be empty: {value!r}")
    if not parsed_value:
        raise ValueError(f"Template variable value cannot be empty: {value!r}")
    return key, parsed_value


def read_template_var_source(source: str, *, url_timeout: float) -> str:
    scheme = urlparse(source).scheme.lower()
    if scheme in {"http", "https"}:
        request = Request(
            source,
            headers={
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "slime-prompt-converter/1.0",
            },
        )
        try:
            with urlopen(request, timeout=url_timeout) as response:
                content = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
        except HTTPError as exc:
            raise RuntimeError(f"Failed to load template variable URL {source!r}: HTTP {exc.code}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Failed to load template variable URL {source!r}: {exc}") from exc
        return content.decode(charset)

    if scheme:
        raise ValueError(
            f"Unsupported --template-var-file source scheme {scheme!r}; use a local path or an HTTP(S) URL."
        )
    return Path(source).read_text(encoding="utf-8")


def parse_template_vars(args: argparse.Namespace) -> dict[str, str]:
    template_vars = dict(parse_key_value(item) for item in args.template_var)
    for item in args.template_var_file:
        key, source = parse_key_value(item)
        template_vars[key] = read_template_var_source(source, url_timeout=args.template_var_url_timeout)
    return template_vars


def iter_yaml_templates(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    per_turn_prompts = config.get("per_turn_prompts", []) or []
    if not isinstance(per_turn_prompts, list):
        raise ValueError("Template YAML field 'per_turn_prompts' must be a list.")
    return per_turn_prompts


def load_template(template_path: Path) -> Template:
    suffix = template_path.suffix.lower()
    if suffix in JINJA_SUFFIXES:
        return Template(template_path.read_text(encoding="utf-8"))
    if suffix in YAML_SUFFIXES:
        with template_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        for item in iter_yaml_templates(config):
            if item.get("name") == DEFAULT_TEMPLATE_NAME:
                template = item.get("template")
                if not template:
                    raise ValueError(
                        f"Template {DEFAULT_TEMPLATE_NAME!r} exists in {template_path}, but it is empty."
                    )
                return Template(str(template))
        raise KeyError(f"Template {DEFAULT_TEMPLATE_NAME!r} not found in {template_path}.")
    raise ValueError(
        f"Unsupported template format: {template_path}. Supported suffixes: "
        f"{sorted(JINJA_SUFFIXES | YAML_SUFFIXES)}."
    )


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f"Unsupported input format: {path}. Supported formats are .parquet and .jsonl.")


def get_nested_value(row: dict[str, Any], field_path: str) -> Any:
    value: Any = row
    current_path = []
    for part in field_path.split("."):
        current_path.append(part)
        if isinstance(value, dict) and part in value:
            value = value[part]
            continue
        if isinstance(value, list) and part.isdigit():
            index = int(part)
            try:
                value = value[index]
            except IndexError as exc:
                raise IndexError(f"Field path {field_path!r} index {index} is out of range.") from exc
            continue
        available = sorted(value) if isinstance(value, dict) else []
        raise KeyError(
            f"Field path {field_path!r} is missing at {'.'.join(current_path)!r}; "
            f"available fields: {available}"
        )
    return value


def stringify_problem(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(
            str(item["content"]) if isinstance(item, dict) and "content" in item else str(item) for item in value
        )
    return str(value)


def normalize_extra_info(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = dict(row.get("extra_info") or {})
    extra_info.pop("ability", None)
    extra_info.pop("data_source", None)
    return extra_info


def to_output_rows(
    raw_rows: list[dict[str, Any]],
    problem_field: str,
    template: Template,
    template_vars: dict[str, str],
) -> list[dict[str, Any]]:
    output_rows = []
    warned_fallback = False
    for row in raw_rows:
        problem = stringify_problem(get_nested_value(row, problem_field))
        extra_info = row.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            raise TypeError(f"Field 'extra_info' must be a dict, got {type(extra_info).__name__}")

        data_source = row.get("data_source") or extra_info.get("data_source")
        ability = row.get("ability") or extra_info.get("ability")
        reward_model = row.get("reward_model") or {}
        if not isinstance(reward_model, dict):
            raise TypeError(f"Field 'reward_model' must be a dict, got {type(reward_model).__name__}")

        ground_truth = row.get("ground_truth") or reward_model.get("ground_truth")
        if ground_truth is None:
            if not warned_fallback:
                print(
                    "warning: rows lack 'ground_truth'/'reward_model.ground_truth'; "
                    f"falling back to --problem-field {problem_field!r} as reward_model.ground_truth.",
                    file=sys.stderr,
                )
                warned_fallback = True
            ground_truth = problem
        ground_truth = stringify_problem(ground_truth)

        template_context = {
            "problem": problem,
            "reference_pytorch_code": problem,
            **template_vars,
        }
        output_rows.append(
            {
                "data_source": data_source,
                "prompt": [{"content": template.render(**template_context), "role": "user"}],
                "reward_model": {
                    "ground_truth": ground_truth,
                    "style": reward_model.get("style") or "rule",
                },
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


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        pq.write_table(to_arrow_table(rows), path)
        return
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    raise ValueError(f"Unsupported output format: {path}. Supported formats are .parquet and .jsonl.")


def write_samples(rows: list[dict[str, Any]], sample_output: Path, num_samples: int) -> None:
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    with sample_output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows[:num_samples]):
            extra_info = row.get("extra_info") or {}
            ident = ", ".join(f"{key}={extra_info[key]}" for key in ("problem_id", "name") if key in extra_info)
            f.write("=" * 72 + "\n")
            f.write(f"sample {index}" + (f"  [{ident}]" if ident else "") + "\n")
            f.write(f"data_source={row.get('data_source')}  ability={row.get('ability')}\n")
            f.write("=" * 72 + "\n")
            for message in row.get("prompt") or []:
                f.write(f"---- prompt (role={message.get('role')}) ----\n")
                f.write((message.get("content") or "").rstrip() + "\n")
            ground_truth = (row.get("reward_model") or {}).get("ground_truth")
            if ground_truth is not None:
                f.write("---- reward_model.ground_truth ----\n")
                f.write(str(ground_truth).rstrip() + "\n")
            f.write("\n\n")


def main() -> None:
    args = parse_args()
    template = load_template(args.template_path)
    template_vars = parse_template_vars(args)
    output_rows = to_output_rows(read_records(args.raw_data), args.problem_field, template, template_vars)
    write_records(args.target_data, output_rows)

    sample_output = args.sample_output or args.target_data.with_suffix(".sample.txt")
    write_samples(output_rows, sample_output, args.num_samples)

    print(f"wrote data: {args.target_data}")
    print(f"wrote samples: {sample_output}")
    print(f"rows: {len(output_rows)}")
    print("columns: data_source, prompt, reward_model, ability, extra_info")


if __name__ == "__main__":
    main()
