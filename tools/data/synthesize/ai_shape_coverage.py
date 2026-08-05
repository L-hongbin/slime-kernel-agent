#!/usr/bin/env python3
"""Select, generate, statically gate, and materialize AI shape siblings.

The source parquet is never rewritten.  ``select`` creates a deterministic,
coverage-first parent pilot.  ``generate`` records complete HTTP requests and
responses from one or more OpenAI-compatible endpoints.  ``materialize``
accepts only reference programs that preserve the computation skeleton while
increasing statically proven returned-input storage within the memory policy.
Runtime acceptance remains a separate parent-child paired H20 gate.
"""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.augment_prompt_tasks import (
    _SHAPE_FACTORIES,
    _normalized_ast_sha256,
    _replace_reference,
    _shape_nodes,
    analyze_code,
)

CONTRACT_VERSION = "ai_shape_parameter_sibling_v3"
GENERATION_CONTRACT_VERSION = "ai_shape_markdown_low_effort_random_targets_32k_v8"
STRICT_TARGET_GENERATION_CONTRACT_VERSIONS = frozenset(
    {
        "ai_shape_markdown_low_effort_random_targets_v7",
        GENERATION_CONTRACT_VERSION,
    }
)
LEGACY_GENERATION_CONTRACT_VERSIONS = frozenset({"ai_shape_markdown_high_effort_medium_large_4g_v6"})
MODEL_NAME = "deepseek-v4-flash-0731"
MAX_OUTPUT_TOKENS = 32 * 1024
REQUEST_TIMEOUT_SECONDS = 2700.0
GENERATION_CONCURRENCY_PER_ENDPOINT = 64
MIN_INPUT_SCALE = 2.0
MEDIUM_INPUT_MIN_BYTES = 64 * 1024**2
MEDIUM_INPUT_MAX_BYTES = 256 * 1024**2
LARGE_INPUT_MAX_BYTES = 4 * 1024**3
TARGET_TOLERANCE_FRACTION = 0.25
TARGET_SALT = "prompt_tvm_v4_ai_shape_random_targets_v1"
DEFAULT_ALREADY_LARGE_INPUT_BYTES = 128 * 1024**2
DEFAULT_SALT = "prompt_tvm_v4_ai_shape_coverage_v1"
TARGET_PARENT_COVERAGE = 0.8
VARIANT_ORDER = {"medium": 0, "large": 1}
VARIANT_ALIASES = {"median": "medium", "medium": "medium", "large": "large", "variant1": "medium", "variant2": "large"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

USER_TEMPLATE = """Create two shape variants of the PyTorch reference below.

Keep the Model class exactly unchanged. Elsewhere, preserve the exact program except for
existing numeric input-shape values and directly linked numeric shape values. Do not add,
remove, reorder, or rewrite statements, variables, imports, calls, operators, tensor
factories, dtypes, control flow, or tensor ranks.
These preservation rules take priority over target proximity: never compensate for an
infeasible target by changing a non-shape number or any program structure.

Measure input storage as the total bytes of all tensors returned by get_inputs(). Exact
targets may be infeasible for discrete shapes, so get within 25% of each target while
remaining in its band and at least 2x larger than the original input storage:
- Medium target: {medium_target_mib} MiB; band: 64 MiB to 256 MiB inclusive.
- Large target: {large_target_mib} MiB; band: greater than 256 MiB and at most 4096 MiB.
Both variants must be complete runnable references and practical to execute on an H20 without OOM.

Return exactly these two Markdown sections, with no JSON and no prose outside them:

## Medium
```python
<full reference>
```

## Large
```python
<full reference>
```

PyTorch reference:
{reference}
"""


@dataclass(frozen=True)
class RunPaths:
    """Internal artifact layout derived from the one public run directory."""

    selected: Path  # select: coverage-first parent pilot parquet
    selection_manifest: Path  # select: selection counts, salt, and parent UUIDs
    generation: Path  # generate: primary LLM request/response JSONL
    feedback_generation: Path  # generate --feedback: retry JSONL for static failures
    children: Path  # materialize: statically accepted shape-sibling children
    paired: Path  # materialize: each accepted parent followed by its children
    static_manifest: Path  # materialize: per-variant static-gate decisions
    accepted: Path  # finalize-runtime: children that passed parent+child H20
    uncovered: Path  # finalize-runtime: parents with no runtime-accepted child
    retryable: Path  # uncovered parents eligible for regenerate/feedback retry
    inconclusive: Path  # uncovered parents whose unchanged parent failed runtime
    runtime_summary: Path  # finalize-runtime: coverage counts and decisions


def _run_paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        selected=run_dir / "selected.parquet",
        selection_manifest=run_dir / "selection.json",
        generation=run_dir / "generation.jsonl",
        feedback_generation=run_dir / "generation-feedback.jsonl",
        children=run_dir / "static" / "children.parquet",
        paired=run_dir / "static" / "paired.parquet",
        static_manifest=run_dir / "static" / "manifest.json",
        accepted=run_dir / "runtime" / "accepted.parquet",
        uncovered=run_dir / "runtime" / "uncovered.parquet",
        retryable=run_dir / "runtime" / "retryable.parquet",
        inconclusive=run_dir / "runtime" / "inconclusive.parquet",
        runtime_summary=run_dir / "runtime" / "summary.json",
    )


def _existing_generation_paths(paths: RunPaths) -> list[Path]:
    generation_paths = [paths.generation]
    if paths.feedback_generation.is_file():
        generation_paths.append(paths.feedback_generation)
    return generation_paths


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _operator_family(ops: Any) -> str:
    if isinstance(ops, str):
        try:
            ops = json.loads(ops)
        except json.JSONDecodeError:
            ops = [ops]
    text = " ".join(str(item) for item in (ops or [])).lower()
    if "conv" in text:
        return "convolution"
    if any(token in text for token in ("matmul", "linear", "bmm", "einsum", "mm")):
        return "matrix"
    if any(token in text for token in ("norm", "softmax", "mean", "sum", "max", "min")):
        return "reduction_normalization"
    if any(token in text for token in ("gather", "scatter", "index", "embedding", "sort", "topk")):
        return "indexing"
    if any(token in text for token in ("reshape", "view", "permute", "transpose", "cat", "stack")):
        return "shape_layout"
    return "pointwise_other"


def _iter_rows(path: Path, columns: Sequence[str] | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    parquet = pq.ParquetFile(path)
    index = 0
    for batch in parquet.iter_batches(batch_size=256, columns=columns):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def _take_rows(path: Path, indices: Sequence[int]) -> pa.Table:
    wanted = set(indices)
    by_index = {index: row for index, row in _iter_rows(path) if index in wanted}
    if len(by_index) != len(wanted):
        raise ValueError("selected row index is outside the source parquet")
    ordered = [by_index[index] for index in indices]
    return pa.Table.from_pylist(ordered, schema=pq.ParquetFile(path).schema_arrow)


def select_parents(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    count: int,
    salt: str,
    already_large_input_bytes: int = DEFAULT_ALREADY_LARGE_INPUT_BYTES,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if already_large_input_bytes <= 0:
        raise ValueError("already_large_input_bytes must be positive")
    candidates: list[dict[str, Any]] = []
    already_large: collections.Counter[str] = collections.Counter()
    columns = ["reward_model", "extra_info"]
    for row_index, row in _iter_rows(input_path, columns):
        code = _nested(row, "reward_model.ground_truth")
        uuid = _nested(row, "extra_info.uuid")
        entry_point = _nested(row, "extra_info.entry_point", "Model")
        if not isinstance(code, str) or not isinstance(uuid, str):
            continue
        try:
            analysis = analyze_code(code, entry_point)
        except (SyntaxError, ValueError):
            continue
        source = str(_nested(row, "extra_info.v4.source_family", "unknown"))
        if analysis.input_bytes >= already_large_input_bytes:
            already_large[source] += 1
            continue
        if (
            analysis.factory_count == 0
            or analysis.input_bytes <= 0
        ):
            continue
        mode = str(_nested(row, "extra_info.v4.mode_class", "unknown"))
        op_bucket = str(_nested(row, "extra_info.v4.operator_bucket", "unknown"))
        family = _operator_family(_nested(row, "extra_info.ops", ""))
        tie = _sha256_bytes(f"{salt}:{uuid}".encode())
        candidates.append(
            {
                "row_index": row_index,
                "uuid": uuid,
                "reference_sha256": _sha256_bytes(code.encode()),
                "input_bytes": analysis.input_bytes,
                "source_family": source,
                "mode_class": mode,
                "operator_bucket": op_bucket,
                "operator_family": family,
                "tie": tie,
            }
        )
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} statically measurable parents for requested count {count}")

    selected: list[dict[str, Any]] = []
    covered: set[tuple[str, str]] = set()
    remaining = {item["row_index"]: item for item in candidates}
    source_counts: collections.Counter[str] = collections.Counter()
    while len(selected) < count:
        def score(item: Mapping[str, Any]) -> tuple[int, int, str]:
            labels = {
                ("source_family", item["source_family"]),
                ("mode_class", item["mode_class"]),
                ("operator_bucket", item["operator_bucket"]),
                ("operator_family", item["operator_family"]),
                ("source_x_operator", f"{item['source_family']}|{item['operator_family']}"),
            }
            return (len(labels - covered), -source_counts[str(item["source_family"])], str(item["tie"]))

        best = max(remaining.values(), key=score)
        if score(best)[0] == 0:
            break
        selected.append(best)
        remaining.pop(best["row_index"])
        source_counts[str(best["source_family"])] += 1
        covered.update(
            {
                ("source_family", best["source_family"]),
                ("mode_class", best["mode_class"]),
                ("operator_bucket", best["operator_bucket"]),
                ("operator_family", best["operator_family"]),
                ("source_x_operator", f"{best['source_family']}|{best['operator_family']}"),
            }
        )
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in remaining.values():
        by_source[str(item["source_family"])].append(item)
    for items in by_source.values():
        items.sort(key=lambda item: str(item["tie"]))
    while len(selected) < count:
        available = [source for source, items in by_source.items() if items]
        if not available:
            raise RuntimeError("candidate pool unexpectedly exhausted")
        source = min(available, key=lambda value: (source_counts[value], value))
        best = by_source[source].pop()
        selected.append(best)
        source_counts[source] += 1
    indices = [item["row_index"] for item in selected]
    table = _take_rows(input_path, indices)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "selection_salt": salt,
        "source_path": str(input_path.resolve()),
        "source_sha256": _sha256_file(input_path),
        "selected_path": str(output_path.resolve()),
        "selected_count": count,
        "already_large_input_bytes": already_large_input_bytes,
        "already_large_parent_count": sum(already_large.values()),
        "already_large_parent_count_by_source": dict(sorted(already_large.items())),
        "rows": selected,
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _sample_target_mib(uuid: str, variant: str, lower_mib: int, upper_mib: int) -> int:
    if lower_mib > upper_mib:
        raise ValueError(f"no_{variant}_target_available:{lower_mib}:{upper_mib}")
    digest = _sha256_bytes(f"{TARGET_SALT}:{uuid}:{variant}".encode())
    return lower_mib + int(digest[:16], 16) % (upper_mib - lower_mib + 1)


def _target_input_bytes(row: Mapping[str, Any]) -> dict[str, int]:
    uuid = _nested(row, "extra_info.uuid")
    code = _nested(row, "reward_model.ground_truth")
    entry_point = str(_nested(row, "extra_info.entry_point", "Model"))
    if not isinstance(uuid, str) or not isinstance(code, str):
        raise ValueError("target_sampling_requires_parent_uuid_and_reference")
    parent_bytes = int(analyze_code(code, entry_point).input_bytes)
    mib = 1024**2
    minimum_growth_mib = (int(MIN_INPUT_SCALE * parent_bytes) + mib - 1) // mib
    medium_lower_mib = max(MEDIUM_INPUT_MIN_BYTES // mib, minimum_growth_mib)
    large_lower_mib = max(MEDIUM_INPUT_MAX_BYTES // mib + 1, minimum_growth_mib)
    return {
        "medium": _sample_target_mib(uuid, "medium", medium_lower_mib, MEDIUM_INPUT_MAX_BYTES // mib) * mib,
        "large": _sample_target_mib(uuid, "large", large_lower_mib, LARGE_INPUT_MAX_BYTES // mib) * mib,
    }


def _request_payload(
    row: Mapping[str, Any],
    *,
    max_tokens: int,
    feedback: str | None = None,
    target_input_bytes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    code = _nested(row, "reward_model.ground_truth")
    targets = dict(target_input_bytes or _target_input_bytes(row))
    prompt = USER_TEMPLATE.format(
        reference=code,
        medium_target_mib=int(targets["medium"]) // 1024**2,
        large_target_mib=int(targets["large"]) // 1024**2,
    )
    if feedback:
        prompt += "\nYour previous proposal failed these checks. Correct it once:\n" + feedback
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": True},
        "reasoning_effort": "low",
    }


def _post(endpoint: str, payload: Mapping[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Cluster-private endpoints must not be sent through the host HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return exc.code, {"error": body}


def generate(
    input_path: Path,
    output_path: Path,
    endpoints: Sequence[str],
    max_tokens: int,
    timeout: float,
    concurrency_per_endpoint: int,
    feedback_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    if concurrency_per_endpoint <= 0:
        raise ValueError("concurrency_per_endpoint must be positive")
    all_rows = [row for _, row in _iter_rows(input_path)]
    rows_by_uuid = {str(_nested(row, "extra_info.uuid")): row for row in all_rows}
    if len(rows_by_uuid) != len(all_rows):
        raise ValueError("input parent UUIDs must be unique")
    rows = all_rows
    if feedback_by_uuid is not None:
        rows = [row for row in rows if str(_nested(row, "extra_info.uuid")) in feedback_by_uuid]
    completed: dict[str, Mapping[str, Any]] = {}
    if output_path.is_file():
        with output_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid resume JSONL line {line_number}: {exc}") from exc
                if record.get("generation_contract_version") != GENERATION_CONTRACT_VERSION:
                    raise ValueError(
                        f"resume JSONL line {line_number} uses an incompatible generation contract; "
                        "write the low-effort random-target run to a new output path"
                    )
                uuid = str(record.get("parent_uuid"))
                row = rows_by_uuid.get(uuid)
                if row is None:
                    raise ValueError(f"resume JSONL line {line_number} has unknown parent UUID: {uuid}")
                expected_reference_sha256 = _sha256_bytes(str(_nested(row, "reward_model.ground_truth")).encode())
                if record.get("parent_reference_sha256") != expected_reference_sha256:
                    raise ValueError(f"resume JSONL line {line_number} parent reference hash mismatch: {uuid}")
                if record.get("target_input_bytes") != _target_input_bytes(row):
                    raise ValueError(f"resume JSONL line {line_number} target mismatch: {uuid}")
                if uuid in completed:
                    raise ValueError(f"resume JSONL contains duplicate parent UUID: {uuid}")
                completed[uuid] = record
    pending = [(index, row) for index, row in enumerate(rows) if str(_nested(row, "extra_info.uuid")) not in completed]

    def one(index: int, row: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        endpoint = endpoints[index % len(endpoints)]
        uuid = str(_nested(row, "extra_info.uuid"))
        targets = _target_input_bytes(row)
        payload = _request_payload(
            row,
            max_tokens=max_tokens,
            feedback=(feedback_by_uuid or {}).get(uuid),
            target_input_bytes=targets,
        )
        started = time.time()
        try:
            status, response = _post(endpoint, payload, timeout)
            error = None
        except Exception as exc:  # network failures are evidence, not silent skips
            status, response, error = 0, {}, f"{type(exc).__name__}: {exc}"
        return index, {
            "contract_version": CONTRACT_VERSION,
            "generation_contract_version": GENERATION_CONTRACT_VERSION,
            "parent_uuid": uuid,
            "parent_reference_sha256": _sha256_bytes(str(_nested(row, "reward_model.ground_truth")).encode()),
            "endpoint": endpoint,
            "model": MODEL_NAME,
            "request": payload,
            "target_input_bytes": targets,
            "target_tolerance_fraction": TARGET_TOLERANCE_FRACTION,
            "http_status": status,
            "response": response,
            "error": error,
            "elapsed_seconds": round(time.time() - started, 3),
            "feedback_round": bool((feedback_by_uuid or {}).get(uuid)),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints) * concurrency_per_endpoint) as executor:
            futures = [executor.submit(one, index, row) for index, row in pending]
            for future in concurrent.futures.as_completed(futures):
                _, record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed[str(record["parent_uuid"])] = record
    return {
        "parents": len(rows),
        "resumed": len(rows) - len(pending),
        "executed": len(pending),
        "http_200": sum(record.get("http_status") == 200 for record in completed.values()),
        "output": str(output_path.resolve()),
    }


def _extract_json_object(text: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and "variants" in value:
            return value
    raise ValueError("response_contains_no_variants_json_object")


_MARKDOWN_VARIANT_RE = re.compile(
    r"^[ \t]*#{1,6}[ \t]+(median|medium|large|variant[ \t]*[12])[ \t]*\n"
    r"[ \t]*```(?:python)?[ \t]*\n(.*?)^[ \t]*```[ \t]*$",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

_CURRENT_MARKDOWN_RE = re.compile(
    r"\A[ \t\r\n]*## Medium[ \t]*\n```python[ \t]*\n(.*?)^[ \t]*```[ \t]*\n"
    r"[ \t\r\n]*## Large[ \t]*\n```python[ \t]*\n(.*?)^[ \t]*```[ \t\r\n]*\Z",
    flags=re.MULTILINE | re.DOTALL,
)


def _extract_variants(text: str, *, strict_current: bool = False) -> list[Mapping[str, Any]]:
    """Read strict current Markdown or the explicitly requested legacy formats."""
    if strict_current:
        match = _CURRENT_MARKDOWN_RE.fullmatch(text)
        if match is None:
            raise ValueError("response_does_not_exactly_match_current_markdown_contract")
        return [
            {"variant": "medium", "reference": match.group(1).strip()},
            {"variant": "large", "reference": match.group(2).strip()},
        ]

    markdown: dict[str, Mapping[str, Any]] = {}
    for match in _MARKDOWN_VARIANT_RE.finditer(text):
        heading = re.sub(r"[ \t]+", "", match.group(1).lower())
        variant = VARIANT_ALIASES[heading]
        if variant in markdown:
            raise ValueError(f"duplicate_markdown_{variant}_section")
        markdown[variant] = {"variant": variant, "reference": match.group(2).strip()}
    if markdown:
        missing = set(VARIANT_ORDER) - set(markdown)
        if missing:
            raise ValueError("markdown_missing_" + "_and_".join(sorted(missing)) + "_section")
        return [markdown[variant] for variant in VARIANT_ORDER]
    parsed = _extract_json_object(text)
    variants = parsed.get("variants")
    if not isinstance(variants, list):
        raise ValueError("variants_is_not_a_list")
    normalized = []
    for variant in variants:
        if isinstance(variant, Mapping) and variant.get("size") in VARIANT_ALIASES:
            variant = {**variant, "variant": VARIANT_ALIASES[str(variant["size"])]}
        normalized.append(variant)
    return normalized


def _response_content(record: Mapping[str, Any]) -> str:
    response = record.get("response")
    try:
        message = response["choices"][0]["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError):
        raise ValueError("response_missing_choice_message") from None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response_content_empty")
    return content


class _StructureNormalizer(ast.NodeTransformer):
    def __init__(self, mutable_numeric_node_ids: set[int]) -> None:
        self.mutable_numeric_node_ids = mutable_numeric_node_ids

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if id(node) in self.mutable_numeric_node_ids:
            return ast.copy_location(ast.Constant(value=0), node)
        return node


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            if isinstance(node, ast.AsyncFunctionDef):
                raise ValueError(f"{name}_must_not_be_async")
            return node
    raise ValueError(f"missing_{name}")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ValueError(f"missing_entry_point_{name}")


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for item in target.elts for name in _assigned_names(item)}
    return set()


def _numeric_constants(node: ast.AST) -> list[ast.Constant]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Constant)
        and isinstance(item.value, (int, float))
        and not isinstance(item.value, bool)
    ]


def _shape_numeric_node_ids(tree: ast.Module) -> set[int]:
    """Return numeric literals proved to feed input-factory shapes directly."""
    get_inputs = _function(tree, "get_inputs")
    mutable: set[int] = set()
    linked_names: set[str] = set()
    for node in ast.walk(get_inputs):
        if not isinstance(node, ast.Call) or _call_name(node) not in _SHAPE_FACTORIES:
            continue
        for shape_node in _shape_nodes(node, _call_name(node)):
            mutable.update(id(item) for item in _numeric_constants(shape_node))
            linked_names.update(item.id for item in ast.walk(shape_node) if isinstance(item, ast.Name))

    assignments: list[tuple[set[str], ast.AST]] = []
    for statement in [*tree.body, *ast.walk(get_inputs)]:
        if isinstance(statement, ast.Assign):
            names = {name for target in statement.targets for name in _assigned_names(target)}
            assignments.append((names, statement.value))
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            assignments.append((_assigned_names(statement.target), statement.value))

    changed = True
    while changed:
        changed = False
        for names, value in assignments:
            if not names.intersection(linked_names):
                continue
            before = len(linked_names)
            mutable.update(id(item) for item in _numeric_constants(value))
            linked_names.update(item.id for item in ast.walk(value) if isinstance(item, ast.Name))
            changed |= len(linked_names) != before

    return mutable


def _shape_normalized_dumps(parent_tree: ast.Module, child_tree: ast.Module) -> tuple[str, str]:
    """Normalize proven shape literals and consistently linked init literals.

    A changed literal in ``get_init_inputs`` is linked only when the same
    before/after value mapping occurs in a statically proven input-shape
    expression.  Comparing mappings across the pair avoids treating unrelated
    values such as dimension indices 2 or 3 as shape parameters merely because
    they happen to equal one of the tensor dimensions.
    """
    parent = copy.deepcopy(parent_tree)
    child = copy.deepcopy(child_tree)
    parent_mutable = _shape_numeric_node_ids(parent)
    child_mutable = _shape_numeric_node_ids(child)

    parent_shape_constants = [
        item for item in ast.walk(parent) if isinstance(item, ast.Constant) and id(item) in parent_mutable
    ]
    child_shape_constants = [
        item for item in ast.walk(child) if isinstance(item, ast.Constant) and id(item) in child_mutable
    ]
    shape_change_pairs: set[tuple[int | float, int | float]] = set()
    if len(parent_shape_constants) == len(child_shape_constants):
        shape_change_pairs = {
            (before.value, after.value)
            for before, after in zip(parent_shape_constants, child_shape_constants, strict=True)
            if before.value != after.value
        }

    parent_init_constants = _numeric_constants(_function(parent, "get_init_inputs"))
    child_init_constants = _numeric_constants(_function(child, "get_init_inputs"))
    if len(parent_init_constants) == len(child_init_constants):
        for before, after in zip(parent_init_constants, child_init_constants, strict=True):
            if before.value != after.value and (before.value, after.value) in shape_change_pairs:
                parent_mutable.add(id(before))
                child_mutable.add(id(after))

    parent = _StructureNormalizer(parent_mutable).visit(parent)
    child = _StructureNormalizer(child_mutable).visit(child)
    ast.fix_missing_locations(parent)
    ast.fix_missing_locations(child)
    return (
        ast.dump(parent, annotate_fields=True, include_attributes=False),
        ast.dump(child, annotate_fields=True, include_attributes=False),
    )


def _call_name(call: ast.Call) -> str:
    parts = []
    current: ast.AST = call.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _factory_names(function: ast.FunctionDef) -> list[str]:
    return [
        name
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and (name := _call_name(node)).startswith("torch.")
    ]


def static_gate(parent_code: str, child_code: str, entry_point: str = "Model") -> dict[str, Any]:
    parent_tree, child_tree = ast.parse(parent_code), ast.parse(child_code)
    parent_class, child_class = _class(parent_tree, entry_point), _class(child_tree, entry_point)
    if ast.dump(parent_class, annotate_fields=True, include_attributes=False) != ast.dump(
        child_class, annotate_fields=True, include_attributes=False
    ):
        raise ValueError("model_changed")
    parent_normalized, child_normalized = _shape_normalized_dumps(parent_tree, child_tree)
    if parent_normalized != child_normalized:
        raise ValueError("reference_changed_beyond_proven_input_shape_numbers")
    parent_inputs, child_inputs = _function(parent_tree, "get_inputs"), _function(child_tree, "get_inputs")
    _function(parent_tree, "get_init_inputs")
    _function(child_tree, "get_init_inputs")
    if _factory_names(parent_inputs) != _factory_names(child_inputs):
        raise ValueError("torch_factory_or_random_method_sequence_changed")
    parent_analysis = analyze_code(parent_code, entry_point)
    child_analysis = analyze_code(child_code, entry_point)
    if parent_analysis.factory_count != child_analysis.factory_count:
        raise ValueError("input_factory_count_changed")
    if child_analysis.input_bytes <= parent_analysis.input_bytes:
        raise ValueError("returned_input_storage_did_not_increase")
    scale = child_analysis.input_bytes / parent_analysis.input_bytes
    if scale < MIN_INPUT_SCALE:
        raise ValueError(f"input_storage_scale_below_minimum:{scale:.4f}")
    return {
        "input_bytes_before": parent_analysis.input_bytes,
        "input_bytes_after": child_analysis.input_bytes,
        "input_scale": scale,
        "parent_reference_sha256": _sha256_bytes(parent_code.encode()),
        "child_reference_sha256": _sha256_bytes(child_code.encode()),
        "child_normalized_ast_sha256": _normalized_ast_sha256(child_code),
    }


def _validate_variant_storage(variant: str, input_bytes: int) -> None:
    if variant == "medium" and not MEDIUM_INPUT_MIN_BYTES <= input_bytes <= MEDIUM_INPUT_MAX_BYTES:
        raise ValueError(f"medium_input_storage_out_of_range:{input_bytes}")
    if variant == "large" and not MEDIUM_INPUT_MAX_BYTES < input_bytes <= LARGE_INPUT_MAX_BYTES:
        raise ValueError(f"large_input_storage_out_of_range:{input_bytes}")


def _validate_target_proximity(input_bytes: int, target_input_bytes: int) -> float:
    if target_input_bytes <= 0:
        raise ValueError("target_input_storage_must_be_positive")
    relative_error = abs(input_bytes - target_input_bytes) / target_input_bytes
    if relative_error > TARGET_TOLERANCE_FRACTION:
        raise ValueError(
            f"target_input_storage_outside_tolerance:{input_bytes}:{target_input_bytes}:{relative_error:.6f}"
        )
    return relative_error


def _load_generation_records(paths: Sequence[Path]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                result[str(record["parent_uuid"])] = record
    return result


def _make_child(parent: Mapping[str, Any], child_code: str, static: Mapping[str, Any]) -> dict[str, Any]:
    child = copy.deepcopy(dict(parent))
    parent_code = str(_nested(parent, "reward_model.ground_truth"))
    parent_uuid = str(_nested(parent, "extra_info.uuid"))
    identity = f"{parent_uuid}:{static['child_reference_sha256']}"
    child_uuid = f"shapeai_{_sha256_bytes(identity.encode())[:24]}"
    child["reward_model"]["ground_truth"] = child_code
    child["prompt"] = _replace_reference(child.get("prompt"), parent_code, child_code, required=True)
    extra = child["extra_info"]
    if extra.get("original_prompt") is not None:
        extra["original_prompt"] = _replace_reference(extra.get("original_prompt"), parent_code, child_code, required=False)
    extra["uuid"] = child_uuid
    v4 = dict(extra.get("v4") or {})
    v4["parent_uuid"] = parent_uuid
    v4["reference_sha256"] = static["child_reference_sha256"]
    v4["normalized_ast_sha256"] = static["child_normalized_ast_sha256"]
    v4["included_in_review_train"] = False
    v4["runtime_validation_status"] = "ai_shape_static_pass_runtime_validation_required"
    v4["governance_status"] = "ai_shape_review_only"
    extra["v4"] = v4
    return child


def _rejected_variant_decisions(parent_uuid: str, reason: str) -> list[dict[str, Any]]:
    """Count a parent-wide failure once for each requested output variant."""
    return [
        {
            "parent_uuid": parent_uuid,
            "variant_index": variant_index,
            "requested_variant": variant,
            "accepted": False,
            "reason": reason,
        }
        for variant, variant_index in VARIANT_ORDER.items()
    ]


def materialize(
    selected_path: Path,
    generation_paths: Sequence[Path],
    children_path: Path,
    paired_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    records = _load_generation_records(generation_paths)
    parents = [row for _, row in _iter_rows(selected_path)]
    children: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for parent in parents:
        uuid = str(_nested(parent, "extra_info.uuid"))
        code = str(_nested(parent, "reward_model.ground_truth"))
        entry_point = str(_nested(parent, "extra_info.entry_point", "Model"))
        record = records.get(uuid)
        parent_children: list[dict[str, Any]] = []
        if record is None:
            decisions.extend(_rejected_variant_decisions(uuid, "missing_generation_record"))
        else:
            generation_contract = record.get("generation_contract_version")
            if generation_contract not in {
                *STRICT_TARGET_GENERATION_CONTRACT_VERSIONS,
                *LEGACY_GENERATION_CONTRACT_VERSIONS,
            }:
                decisions.extend(
                    _rejected_variant_decisions(
                        uuid,
                        f"unsupported_generation_contract:{generation_contract}",
                    )
                )
                continue
            if record.get("parent_reference_sha256") != _sha256_bytes(code.encode()):
                decisions.extend(
                    _rejected_variant_decisions(uuid, "generation_parent_reference_hash_mismatch")
                )
                continue
            target_input_bytes = record.get("target_input_bytes")
            if generation_contract in STRICT_TARGET_GENERATION_CONTRACT_VERSIONS:
                expected_targets = _target_input_bytes(parent)
                if target_input_bytes != expected_targets:
                    decisions.extend(_rejected_variant_decisions(uuid, "generation_target_mismatch"))
                    continue
            try:
                variants = _extract_variants(
                    _response_content(record),
                    strict_current=generation_contract in STRICT_TARGET_GENERATION_CONTRACT_VERSIONS,
                )
            except (TypeError, ValueError) as exc:
                decisions.extend(_rejected_variant_decisions(uuid, str(exc)))
                variants = []
            seen_hashes: set[str] = set()
            for variant_index, variant in enumerate(variants):
                requested_variant = variant.get("variant") if isinstance(variant, Mapping) else None
                child_code = variant.get("reference") if isinstance(variant, Mapping) else None
                decision: dict[str, Any] = {
                    "parent_uuid": uuid,
                    "variant_index": variant_index,
                    "requested_variant": requested_variant,
                    "accepted": False,
                }
                target_bytes = (
                    target_input_bytes.get(str(requested_variant))
                    if isinstance(target_input_bytes, Mapping)
                    else None
                )
                if target_bytes is not None:
                    decision["target_input_bytes"] = int(target_bytes)
                try:
                    if requested_variant not in VARIANT_ORDER:
                        raise ValueError("variant_must_be_medium_or_large")
                    if not isinstance(child_code, str) or not child_code.strip():
                        raise ValueError("reference_must_be_non_empty_text")
                    static = static_gate(code, child_code, entry_point)
                    decision.update(
                        {
                            "input_scale": static["input_scale"],
                            "input_bytes_before": static["input_bytes_before"],
                            "input_bytes_after": static["input_bytes_after"],
                        }
                    )
                    if generation_contract in STRICT_TARGET_GENERATION_CONTRACT_VERSIONS:
                        if target_bytes is None:
                            raise ValueError(f"missing_{requested_variant}_target_input_storage")
                        target_delta_bytes = int(static["input_bytes_after"]) - int(target_bytes)
                        target_relative_error = abs(target_delta_bytes) / int(target_bytes)
                        decision.update(
                            {
                                "target_delta_bytes": target_delta_bytes,
                                "target_direction": "exact" if target_delta_bytes == 0 else "above" if target_delta_bytes > 0 else "below",
                                "target_relative_error": target_relative_error,
                                "target_within_tolerance": target_relative_error <= TARGET_TOLERANCE_FRACTION,
                            }
                        )
                    else:
                        target_relative_error = None
                    _validate_variant_storage(str(requested_variant), int(static["input_bytes_after"]))
                    if generation_contract in STRICT_TARGET_GENERATION_CONTRACT_VERSIONS:
                        _validate_target_proximity(int(static["input_bytes_after"]), int(target_bytes))
                    digest = str(static["child_reference_sha256"])
                    if digest in seen_hashes:
                        raise ValueError("duplicate_child_reference")
                    seen_hashes.add(digest)
                    child = _make_child(parent, child_code.rstrip() + "\n", static)
                    parent_children.append(child)
                    decision.update(
                        {
                            "accepted": True,
                            "child_uuid": _nested(child, "extra_info.uuid"),
                            "target_relative_error": target_relative_error,
                            "child_reference_sha256": digest,
                            "change_summary": variant.get("change_summary"),
                            "constraints": variant.get("constraints"),
                        }
                    )
                except (SyntaxError, TypeError, ValueError) as exc:
                    decision["reason"] = f"{type(exc).__name__}:{exc}"
                decisions.append(decision)
        if parent_children:
            paired.append(copy.deepcopy(parent))
            paired.extend(parent_children)
            children.extend(parent_children)

    schema = pq.ParquetFile(selected_path).schema_arrow
    children_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(children, schema=schema), children_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(paired, schema=schema), paired_path, compression="zstd")
    accepted_parents = len({item["parent_uuid"] for item in decisions if item.get("accepted")})
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "selected_parent_count": len(parents),
        "static_accepted_parent_count": accepted_parents,
        "static_parent_coverage": accepted_parents / len(parents) if parents else 0.0,
        "static_accepted_child_count": len(children),
        "children_path": str(children_path.resolve()),
        "paired_path": str(paired_path.resolve()),
        "generation_paths": [str(path.resolve()) for path in generation_paths],
        "decisions": decisions,
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n")
    return manifest


def feedback_map(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    accepted: set[str] = set()
    for decision in manifest.get("decisions", []):
        uuid = str(decision["parent_uuid"])
        if decision.get("accepted"):
            accepted.add(uuid)
        elif decision.get("reason"):
            grouped[uuid].append(str(decision["reason"]))
    return {
        uuid: "; ".join(dict.fromkeys(reasons))
        for uuid, reasons in grouped.items()
        if uuid not in accepted
    }


def finalize_runtime(
    selected_path: Path,
    children_path: Path,
    static_manifest_path: Path,
    runtime_dir: Path,
    accepted_path: Path,
    uncovered_path: Path,
    retryable_path: Path,
    inconclusive_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    static_manifest = json.loads(static_manifest_path.read_text())
    runtime: dict[str, Mapping[str, Any]] = {}
    runtime_files = sorted(runtime_dir.glob("shard-*.jsonl"))
    if not runtime_files:
        raise ValueError(f"no runtime shard JSONL files under {runtime_dir}")
    for path in runtime_files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                uuid = record.get("uuid")
                if not isinstance(uuid, str) or not uuid:
                    raise ValueError(f"runtime record without UUID in {path}")
                if uuid in runtime:
                    raise ValueError(f"duplicate runtime UUID: {uuid}")
                runtime[uuid] = record

    selected = [row for _, row in _iter_rows(selected_path)]
    static_children = [row for _, row in _iter_rows(children_path)]
    child_by_uuid = {str(_nested(row, "extra_info.uuid")): row for row in static_children}
    accepted_child_to_parent: dict[str, str] = {}
    static_reasons: dict[str, list[str]] = collections.defaultdict(list)
    for decision in static_manifest.get("decisions", []):
        parent_uuid = str(decision["parent_uuid"])
        if decision.get("accepted"):
            child_uuid = str(decision["child_uuid"])
            accepted_child_to_parent[child_uuid] = parent_uuid
        elif decision.get("reason"):
            static_reasons[parent_uuid].append(str(decision["reason"]))

    expected_runtime = set(accepted_child_to_parent) | set(accepted_child_to_parent.values())
    missing_runtime = sorted(expected_runtime - set(runtime))
    if missing_runtime:
        raise ValueError(f"runtime evidence missing for {len(missing_runtime)} paired rows")

    accepted_children: list[dict[str, Any]] = []
    accepted_by_parent: dict[str, list[str]] = collections.defaultdict(list)
    runtime_reasons: dict[str, list[str]] = collections.defaultdict(list)

    def runtime_detail(record: Mapping[str, Any]) -> str:
        detail = {
            "status": record.get("status"),
            "failure_reasons": record.get("failure_reasons"),
            "error_type": record.get("error_type"),
            "error": record.get("error"),
        }
        return json.dumps(detail, sort_keys=True, ensure_ascii=False)[:2000]

    for child_uuid, parent_uuid in accepted_child_to_parent.items():
        parent_passed = bool(runtime[parent_uuid].get("passed"))
        child_passed = bool(runtime[child_uuid].get("passed"))
        if parent_passed and child_passed:
            child = copy.deepcopy(child_by_uuid[child_uuid])
            child["extra_info"]["v4"]["runtime_validation_status"] = "ai_shape_parent_child_h20_passed"
            accepted_children.append(child)
            accepted_by_parent[parent_uuid].append(child_uuid)
        elif not parent_passed:
            runtime_reasons[parent_uuid].append(
                "unchanged_parent_runtime_failed_or_inconclusive:" + runtime_detail(runtime[parent_uuid])
            )
        else:
            runtime_reasons[parent_uuid].append(
                f"child_runtime_failed:{child_uuid}:{runtime_detail(runtime[child_uuid])}"
            )

    parent_decisions = []
    uncovered_rows = []
    retryable_rows = []
    inconclusive_rows = []
    for parent in selected:
        parent_uuid = str(_nested(parent, "extra_info.uuid"))
        accepted = bool(accepted_by_parent.get(parent_uuid))
        reasons = [*static_reasons.get(parent_uuid, []), *runtime_reasons.get(parent_uuid, [])]
        if not accepted:
            uncovered_rows.append(parent)
            if not reasons:
                reasons = ["no_static_accepted_variant"]
            if any(reason.startswith("unchanged_parent_runtime_failed_or_inconclusive:") for reason in reasons):
                inconclusive_rows.append(parent)
            else:
                retryable_rows.append(parent)
        parent_decisions.append(
            {
                "parent_uuid": parent_uuid,
                "accepted": accepted,
                "accepted_child_uuids": accepted_by_parent.get(parent_uuid, []),
                "reason": "; ".join(reasons) if reasons else None,
            }
        )

    schema = pq.ParquetFile(selected_path).schema_arrow
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(accepted_children, schema=schema), accepted_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(uncovered_rows, schema=schema), uncovered_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(retryable_rows, schema=schema), retryable_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(inconclusive_rows, schema=schema), inconclusive_path, compression="zstd")
    accepted_parent_count = len(accepted_by_parent)
    summary = {
        "contract_version": CONTRACT_VERSION,
        "selected_parent_count": len(selected),
        "runtime_accepted_parent_count": accepted_parent_count,
        "runtime_parent_coverage": accepted_parent_count / len(selected) if selected else 0.0,
        "runtime_accepted_child_count": len(accepted_children),
        "uncovered_parent_count": len(uncovered_rows),
        "retryable_parent_count": len(retryable_rows),
        "inconclusive_parent_count": len(inconclusive_rows),
        "accepted_path": str(accepted_path.resolve()),
        "uncovered_path": str(uncovered_path.resolve()),
        "retryable_path": str(retryable_path.resolve()),
        "inconclusive_path": str(inconclusive_path.resolve()),
        "runtime_files": [str(path.resolve()) for path in runtime_files],
        "runtime_record_count": len(runtime),
        "decisions": parent_decisions,
    }
    _atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def combine_accepted(
    selected_path: Path,
    accepted_paths: Sequence[Path],
    output_path: Path,
    uncovered_path: Path,
    summary_path: Path,
    target_coverage: float,
    static_manifest_paths: Sequence[Path] = (),
    audit_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target coverage must be in (0, 1]")
    selected = [row for _, row in _iter_rows(selected_path)]
    selected_by_uuid = {str(_nested(row, "extra_info.uuid")): row for row in selected}
    if len(selected_by_uuid) != len(selected):
        raise ValueError("selected parent UUIDs are not unique")

    children: list[dict[str, Any]] = []
    child_uuids: set[str] = set()
    accepted_parent_uuids: set[str] = set()
    accepted_counts: dict[str, int] = {}
    audit_rows: list[dict[str, Any]] = []
    if static_manifest_paths and len(static_manifest_paths) != len(accepted_paths):
        raise ValueError("accepted paths and static manifests must have the same length")
    for path_index, path in enumerate(accepted_paths):
        static_by_child: dict[str, Mapping[str, Any]] = {}
        static_manifest_path = None
        if static_manifest_paths:
            static_manifest_path = static_manifest_paths[path_index]
            static_manifest = json.loads(static_manifest_path.read_text())
            static_by_child = {
                str(item["child_uuid"]): item
                for item in static_manifest.get("decisions", [])
                if item.get("accepted") and item.get("child_uuid")
            }
        path_count = 0
        for _, child in _iter_rows(path):
            child_uuid = str(_nested(child, "extra_info.uuid"))
            parent_uuid = str(_nested(child, "extra_info.v4.parent_uuid"))
            if child_uuid in child_uuids:
                raise ValueError(f"duplicate accepted child UUID: {child_uuid}")
            if parent_uuid not in selected_by_uuid:
                raise ValueError(f"accepted child has unknown selected parent: {parent_uuid}")
            if _nested(child, "extra_info.v4.runtime_validation_status") != "ai_shape_parent_child_h20_passed":
                raise ValueError(f"accepted child lacks paired H20 evidence: {child_uuid}")
            child_uuids.add(child_uuid)
            accepted_parent_uuids.add(parent_uuid)
            children.append(child)
            if static_manifest_paths:
                if child_uuid not in static_by_child:
                    raise ValueError(f"accepted child missing from static manifest: {child_uuid}")
                decision = static_by_child[child_uuid]
                audit_rows.append(
                    {
                        "child_uuid": child_uuid,
                        "parent_uuid": parent_uuid,
                        "input_scale": decision.get("input_scale"),
                        "input_bytes_before": decision.get("input_bytes_before"),
                        "input_bytes_after": decision.get("input_bytes_after"),
                        "change_summary": decision.get("change_summary"),
                        "constraints": decision.get("constraints"),
                        "child_reference_sha256": decision.get("child_reference_sha256"),
                        "runtime_validation_status": "ai_shape_parent_child_h20_passed",
                        "accepted_partition": str(path.resolve()),
                        "static_manifest": str(static_manifest_path.resolve()),
                    }
                )
            path_count += 1
        accepted_counts[str(path.resolve())] = path_count

    uncovered = [
        row for row in selected
        if str(_nested(row, "extra_info.uuid")) not in accepted_parent_uuids
    ]
    schema = pq.ParquetFile(selected_path).schema_arrow
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(children, schema=schema), output_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(uncovered, schema=schema), uncovered_path, compression="zstd")
    if audit_manifest_path is not None:
        if not static_manifest_paths:
            raise ValueError("audit manifest output requires static manifests")
        _atomic_text(
            audit_manifest_path,
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in audit_rows),
        )
    coverage = len(accepted_parent_uuids) / len(selected) if selected else 0.0
    summary = {
        "contract_version": CONTRACT_VERSION,
        "selected_parent_count": len(selected),
        "runtime_accepted_parent_count": len(accepted_parent_uuids),
        "runtime_parent_coverage": coverage,
        "target_coverage": target_coverage,
        "target_met": coverage >= target_coverage,
        "runtime_accepted_child_count": len(children),
        "uncovered_parent_count": len(uncovered),
        "accepted_inputs": accepted_counts,
        "accepted_path": str(output_path.resolve()),
        "accepted_sha256": _sha256_file(output_path),
        "uncovered_path": str(uncovered_path.resolve()),
        "uncovered_sha256": _sha256_file(uncovered_path),
        "selected_path": str(selected_path.resolve()),
        "selected_sha256": _sha256_file(selected_path),
    }
    if audit_manifest_path is not None:
        summary["audit_manifest_path"] = str(audit_manifest_path.resolve())
        summary["audit_manifest_sha256"] = _sha256_file(audit_manifest_path)
    _atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select", help="select eligible parents into a run directory")
    select.add_argument("source", type=Path)
    select.add_argument("run_dir", type=Path)
    select.add_argument("--count", type=int, default=16)
    generate_parser = subparsers.add_parser("generate", help="generate or resume model responses")
    generate_parser.add_argument("run_dir", type=Path)
    generate_parser.add_argument("--endpoint", action="append", required=True)
    generate_parser.add_argument(
        "--feedback",
        action="store_true",
        help="retry only parents rejected by the latest static materialization",
    )
    materialize_parser = subparsers.add_parser("materialize", help="apply static gates to generated responses")
    materialize_parser.add_argument("run_dir", type=Path)
    finalize_parser = subparsers.add_parser("finalize", help="classify paired H20 validation results")
    finalize_parser.add_argument("run_dir", type=Path)
    finalize_parser.add_argument("runtime_dir", type=Path)
    combine_parser = subparsers.add_parser("combine", help="publish accepted children from one or more runs")
    combine_parser.add_argument("output_dir", type=Path)
    combine_parser.add_argument("run_dir", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "select":
        paths = _run_paths(args.run_dir)
        result = select_parents(
            args.source,
            paths.selected,
            paths.selection_manifest,
            args.count,
            DEFAULT_SALT,
            DEFAULT_ALREADY_LARGE_INPUT_BYTES,
        )
    elif args.command == "generate":
        paths = _run_paths(args.run_dir)
        feedback = feedback_map(paths.static_manifest) if args.feedback else None
        result = generate(
            paths.selected,
            paths.feedback_generation if args.feedback else paths.generation,
            args.endpoint,
            MAX_OUTPUT_TOKENS,
            REQUEST_TIMEOUT_SECONDS,
            GENERATION_CONCURRENCY_PER_ENDPOINT,
            feedback,
        )
    elif args.command == "materialize":
        paths = _run_paths(args.run_dir)
        result = materialize(
            paths.selected,
            _existing_generation_paths(paths),
            paths.children,
            paths.paired,
            paths.static_manifest,
        )
    elif args.command == "finalize":
        paths = _run_paths(args.run_dir)
        result = finalize_runtime(
            paths.selected,
            paths.children,
            paths.static_manifest,
            args.runtime_dir,
            paths.accepted,
            paths.uncovered,
            paths.retryable,
            paths.inconclusive,
            paths.runtime_summary,
        )
    else:
        run_paths = [_run_paths(path) for path in args.run_dir]
        result = combine_accepted(
            run_paths[0].selected,
            [paths.accepted for paths in run_paths],
            args.output_dir / "accepted.parquet",
            args.output_dir / "uncovered.parquet",
            args.output_dir / "summary.json",
            TARGET_PARENT_COVERAGE,
            [paths.static_manifest for paths in run_paths],
            args.output_dir / "audit.jsonl",
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
