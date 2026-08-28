#!/usr/bin/env python3
"""Convert executable open-source kernel tasks to the DrKernel parquet schema.

The converter intentionally handles only sources that provide an executable
PyTorch reference, ``get_inputs()``, and an entry-point class.  Description-to-
Triton corpora such as TritonBench train_crawl/train_synth are not silently
promoted to RL tasks because they do not provide that contract.

Each output gets a JSONL provenance sidecar and a JSON summary.  Conversion is
lossless with respect to task behavior; KernelBook entry-point classes are AST-
renamed to ``Model`` so they fit the current DrKernel prompt contract.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .static_analysis import _call_name


class _EntryPointRenamer(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        if node.name == self.old:
            node.name = self.new
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.old:
            node.id = self.new
        return node


def normalize_entry_point(code: str, entry_point: str) -> str:
    """Return code whose task class is named ``Model``.

    AST unparsing is deliberate: it updates explicit ``super(Old, self)`` and
    other references together with the class declaration.  A pre-existing,
    different ``Model`` is rejected rather than overwritten ambiguously.
    """

    if not isinstance(code, str) or not code.strip():
        raise ValueError("empty Python reference")
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError(f"invalid entry point: {entry_point!r}")
    tree = ast.parse(code)
    top_classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    if entry_point not in top_classes:
        raise ValueError(f"missing top-level entry-point class {entry_point!r}")
    if entry_point != "Model" and "Model" in top_classes:
        raise ValueError("reference already defines a different top-level Model")
    if entry_point == "Model":
        return code.rstrip() + "\n"
    renamed = _EntryPointRenamer(entry_point, "Model").visit(tree)
    ast.fix_missing_locations(renamed)
    return ast.unparse(renamed).rstrip() + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _ops(code: str) -> str:
    tree = ast.parse(code)
    calls = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in [_call_name(node.func)]
        if name and name not in {"super"} and not name.startswith("self.")
    }
    return json.dumps(sorted(calls), ensure_ascii=False)


def _template(template_parquet: Path) -> tuple[pa.Schema, str]:
    parquet = pq.ParquetFile(template_parquet)
    batch = next(parquet.iter_batches(batch_size=1, columns=["prompt", "reward_model.ground_truth"]))
    row = batch.to_pylist()[0]
    code = row["reward_model"]["ground_truth"]
    messages = row["prompt"]
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("template must have one user prompt")
    content = messages[0].get("content")
    if not isinstance(content, str) or not isinstance(code, str) or not content.endswith(code):
        raise ValueError("template prompt must end with its exact ground truth")
    return parquet.schema_arrow, content[: -len(code)]


def _drkernel_row(
    *,
    code: str,
    prompt_prefix: str,
    data_source: str,
    level: str,
    repo_name: str,
    task_type: str,
    uuid: str,
) -> dict[str, Any]:
    content = prompt_prefix + code
    prompt = [{"content": content, "role": "user"}]
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "kernel_optimization",
        "reward_model": {"ground_truth": code, "style": "rule"},
        "extra_info": {
            "entry_point": "Model",
            "level": level,
            "module_name": "Model",
            "ops": _ops(code),
            "original_prompt": prompt,
            "repo_name": repo_name,
            "type": task_type,
            "uuid": uuid,
        },
    }


def _kernelbook_rows(
    path: Path, prompt_prefix: str, revision: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = pq.read_table(path)
    required = {
        "entry_point",
        "python_code",
        "repo_name",
        "uuid",
        "licenses",
        "sha",
        "repo_link",
        "synthetic",
    }
    missing = required - set(source.column_names)
    if missing:
        raise ValueError(f"KernelBook is missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    conversion_errors: dict[str, int] = {}
    conversion_rejected: list[dict[str, Any]] = []
    for source_index, item in enumerate(source.to_pylist()):
        try:
            code = normalize_entry_point(item["python_code"], item["entry_point"])
        except (SyntaxError, ValueError) as exc:
            key = type(exc).__name__
            conversion_errors[key] = conversion_errors.get(key, 0) + 1
            conversion_rejected.append(
                {
                    "source_row_index": source_index,
                    "source_uuid": item["uuid"],
                    "repo_name": item["repo_name"],
                    "entry_point": item["entry_point"],
                    "error": f"{key}: {exc}",
                }
            )
            continue
        code_hash = _sha256_bytes(code.encode())
        stable_uuid = f"kernelbook_{item['uuid']}_{code_hash[:16]}"
        rows.append(
            _drkernel_row(
                code=code,
                prompt_prefix=prompt_prefix,
                data_source="cuda_llm_external_kernelbook",
                level="kernelbook",
                repo_name=item["repo_name"] or "unknown",
                task_type="external_kernelbook",
                uuid=stable_uuid,
            )
        )
        provenance.append(
            {
                "converted_row_index": len(rows) - 1,
                "converted_uuid": stable_uuid,
                "source_row_index": source_index,
                "source": "GPUMODE/KernelBook",
                "source_revision": revision,
                "source_uuid": item["uuid"],
                "source_entry_point": item["entry_point"],
                "repo_name": item["repo_name"],
                "repo_sha": item["sha"],
                "repo_link": item["repo_link"],
                "licenses": item["licenses"] or [],
                "synthetic": item["synthetic"],
                "normalized_code_sha256": code_hash,
            }
        )
    return (
        rows,
        provenance,
        {
            "source_rows": source.num_rows,
            "conversion_errors": conversion_errors,
            "conversion_rejected": conversion_rejected,
            "source_sha256": _sha256_file(path),
        },
    )


def _multikernelbench_rows(
    root: Path,
    prompt_prefix: str,
    revision: str,
    *,
    include_npu: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reference_root = root / "reference"
    paths = sorted(reference_root.glob("*/*.py"))
    if not paths:
        raise ValueError(f"no reference Python files under {reference_root}")
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    excluded_npu = 0
    conversion_errors: dict[str, int] = {}
    conversion_rejected: list[dict[str, Any]] = []
    for path in paths:
        if path.name.startswith("_"):
            continue
        category = path.parent.name
        if category.startswith("npukernelbench_") and not include_npu:
            excluded_npu += 1
            continue
        try:
            code = normalize_entry_point(path.read_text(encoding="utf-8"), "Model")
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            key = type(exc).__name__
            conversion_errors[key] = conversion_errors.get(key, 0) + 1
            conversion_rejected.append(
                {
                    "source_path": path.relative_to(root).as_posix(),
                    "error": f"{key}: {exc}",
                }
            )
            continue
        relative_path = path.relative_to(root).as_posix()
        code_hash = _sha256_bytes(code.encode())
        stable_uuid = f"multikernelbench_{_sha256_bytes((revision + ':' + relative_path).encode())[:20]}"
        rows.append(
            _drkernel_row(
                code=code,
                prompt_prefix=prompt_prefix,
                data_source="cuda_llm_external_multikernelbench",
                level=category,
                repo_name="wzzll123/MultiKernelBench",
                task_type="external_multikernelbench",
                uuid=stable_uuid,
            )
        )
        provenance.append(
            {
                "converted_row_index": len(rows) - 1,
                "converted_uuid": stable_uuid,
                "source": "wzzll123/MultiKernelBench",
                "source_revision": revision,
                "source_path": relative_path,
                "category": category,
                "licenses": ["MIT"],
                "normalized_code_sha256": code_hash,
            }
        )
    return (
        rows,
        provenance,
        {
            "source_rows": len(paths),
            "excluded_npu_rows": excluded_npu,
            "conversion_errors": conversion_errors,
            "conversion_rejected": conversion_rejected,
            "source_tree_sha256": _tree_hash(paths, root),
        },
    )


def _cuda_agent_ops_rows(
    path: Path, prompt_prefix: str, revision: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Convert the CC-BY-4.0 CUDA-Agent-Ops-6K release.

    The upstream file already uses a top-level ``Model`` plus executable
    ``get_inputs``/``get_init_inputs`` helpers.  Conversion therefore preserves
    code verbatim apart from a trailing newline and adds DrKernel metadata and
    row-level attribution.
    """

    source = pq.read_table(path)
    required = {"ops", "data_source", "code"}
    missing = required - set(source.column_names)
    if missing:
        raise ValueError(f"CUDA-Agent-Ops-6K is missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    conversion_errors: dict[str, int] = {}
    conversion_rejected: list[dict[str, Any]] = []
    for source_index, item in enumerate(source.to_pylist()):
        try:
            code = normalize_entry_point(item["code"], "Model")
            tree = ast.parse(code)
            functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            if "get_inputs" not in functions:
                raise ValueError("missing top-level get_inputs")
        except (SyntaxError, TypeError, ValueError) as exc:
            key = type(exc).__name__
            conversion_errors[key] = conversion_errors.get(key, 0) + 1
            conversion_rejected.append(
                {
                    "source_row_index": source_index,
                    "error": f"{key}: {exc}",
                }
            )
            continue
        code_hash = _sha256_bytes(code.encode())
        stable_uuid = f"cuda_agent_ops_{_sha256_bytes(f'{revision}:{source_index}:{code_hash}'.encode())[:20]}"
        rows.append(
            _drkernel_row(
                code=code,
                prompt_prefix=prompt_prefix,
                data_source="cuda_llm_external_cuda_agent_ops_6k",
                level=str(item["data_source"]),
                repo_name="BytedTsinghua-SIA/CUDA-Agent-Ops-6K",
                task_type="external_cuda_agent_ops_6k",
                uuid=stable_uuid,
            )
        )
        provenance.append(
            {
                "converted_row_index": len(rows) - 1,
                "converted_uuid": stable_uuid,
                "source_row_index": source_index,
                "source": "BytedTsinghua-SIA/CUDA-Agent-Ops-6K",
                "source_revision": revision,
                "source_data_source": item["data_source"],
                "source_ops": item["ops"],
                "licenses": ["CC-BY-4.0"],
                "normalized_code_sha256": code_hash,
            }
        )
    return (
        rows,
        provenance,
        {
            "source_rows": source.num_rows,
            "conversion_errors": conversion_errors,
            "conversion_rejected": conversion_rejected,
            "source_sha256": _sha256_file(path),
        },
    )


def _accepted_jsonl_rows(
    path: Path, prompt_prefix: str, revision: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Convert locally generated accepted PyTorch candidates.

    ``accepted.jsonl`` contains executable references plus generator/filter
    metadata, but no independently verifiable upstream revision or license.
    Preserve that metadata in the provenance sidecar without presenting the
    delivery as the separately published CUDA-Agent-Ops-6K corpus.
    """

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    conversion_errors: dict[str, int] = {}
    conversion_rejected: list[dict[str, Any]] = []
    source_rows = 0
    with path.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            source_rows += 1
            source_uuid: str | None = None
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError("JSONL row must be an object")
                code = normalize_entry_point(item.get("code"), "Model")
                tree = ast.parse(code)
                functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
                if "get_inputs" not in functions:
                    raise ValueError("missing top-level get_inputs")
                code_hash = _sha256_bytes(code.encode())
                source_uuid = f"accepted_ops_{_sha256_bytes(f'{revision}:{source_index}:{code_hash}'.encode())[:20]}"
            except (json.JSONDecodeError, SyntaxError, TypeError, ValueError) as exc:
                key = type(exc).__name__
                conversion_errors[key] = conversion_errors.get(key, 0) + 1
                conversion_rejected.append(
                    {
                        "source_row_index": source_index,
                        "source_uuid": source_uuid,
                        "error": f"{key}: {exc}",
                    }
                )
                continue

            original_source = str(item.get("data_source", "unknown"))
            rows.append(
                _drkernel_row(
                    code=code,
                    prompt_prefix=prompt_prefix,
                    data_source="cuda_llm_internal_accepted_ops",
                    level=original_source,
                    repo_name="local/accepted.jsonl",
                    task_type="internal_accepted_ops_candidate",
                    uuid=source_uuid,
                )
            )
            provenance.append(
                {
                    "converted_row_index": len(rows) - 1,
                    "converted_uuid": source_uuid,
                    "source_row_index": source_index,
                    "source": "local accepted.jsonl delivery",
                    "source_revision": revision,
                    "source_data_source": original_source,
                    "source_attempt": item.get("attempt"),
                    "source_repair_round": item.get("repair_round"),
                    "source_ops": item.get("ops"),
                    "source_requested_ops": item.get("requested_ops"),
                    "source_filter": item.get("filter"),
                    "licenses": [],
                    "provenance_status": ("generation prompt, model revision, and license absent from delivery"),
                    "normalized_code_sha256": code_hash,
                }
            )
    return (
        rows,
        provenance,
        {
            "source_rows": source_rows,
            "conversion_errors": conversion_errors,
            "conversion_rejected": conversion_rejected,
            "source_sha256": _sha256_file(path),
            "provenance_status": ("incomplete: generation prompt, model revision, and license absent from delivery"),
        },
    )


def _kernelbench_rows(
    root: Path,
    prompt_prefix: str,
    revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Convert official KernelBench levels 1--3 into a decontamination baseline.

    Level 4 is intentionally excluded: the current training/evaluation question
    is defined against the canonical level 1/2/3 benchmark domains.
    """

    benchmark_root = root / "KernelBench"
    paths = [
        path for level in ("level1", "level2", "level3") for path in sorted((benchmark_root / level).glob("*.py"))
    ]
    if not paths:
        raise ValueError(f"no KernelBench level1/2/3 references under {benchmark_root}")
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    conversion_errors: dict[str, int] = {}
    conversion_rejected: list[dict[str, Any]] = []
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        try:
            code = normalize_entry_point(path.read_text(encoding="utf-8"), "Model")
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            key = type(exc).__name__
            conversion_errors[key] = conversion_errors.get(key, 0) + 1
            conversion_rejected.append({"source_path": relative_path, "error": f"{key}: {exc}"})
            continue
        level = path.parent.name
        code_hash = _sha256_bytes(code.encode())
        stable_uuid = f"kernelbench_{_sha256_bytes(f'{revision}:{relative_path}'.encode())[:20]}"
        rows.append(
            _drkernel_row(
                code=code,
                prompt_prefix=prompt_prefix,
                data_source="cuda_llm_benchmark_kernelbench",
                level=level,
                repo_name="ScalingIntelligence/KernelBench",
                task_type="benchmark_decontamination_reference",
                uuid=stable_uuid,
            )
        )
        provenance.append(
            {
                "converted_row_index": len(rows) - 1,
                "converted_uuid": stable_uuid,
                "source": "ScalingIntelligence/KernelBench",
                "source_revision": revision,
                "source_path": relative_path,
                "level": level,
                "licenses": ["MIT"],
                "normalized_code_sha256": code_hash,
            }
        )
    return (
        rows,
        provenance,
        {
            "source_rows": len(paths),
            "included_levels": ["level1", "level2", "level3"],
            "conversion_errors": conversion_errors,
            "conversion_rejected": conversion_rejected,
            "source_tree_sha256": _tree_hash(paths, root),
        },
    )


def _write_collected_torch_ops_streaming(
    *,
    path: Path,
    output: Path,
    schema: pa.Schema,
    prompt_prefix: str,
    revision: str,
    compression: str,
) -> dict[str, Any]:
    """Stream the large internal delivery without retaining three copies in RAM."""

    parquet = pq.ParquetFile(path)
    required = {"code", "source", "uuid", "ops_valid"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"collected torch ops is missing columns: {sorted(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = output.with_name(f"{output.stem}.provenance.jsonl")
    output_tmp = output.with_name(f".{output.name}.tmp")
    provenance_tmp = provenance_path.with_name(f".{provenance_path.name}.tmp")
    conversion_errors: dict[str, int] = {}
    conversion_rejected: list[dict[str, Any]] = []
    output_rows = 0
    source_index = 0
    writer = pq.ParquetWriter(output_tmp, schema, compression=compression)
    try:
        with provenance_tmp.open("w", encoding="utf-8") as provenance_handle:
            for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
                converted_batch: list[dict[str, Any]] = []
                for item in batch.to_pylist():
                    try:
                        code = normalize_entry_point(item["code"], "Model")
                        tree = ast.parse(code)
                        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
                        if "get_inputs" not in functions:
                            raise ValueError("missing top-level get_inputs")
                    except (SyntaxError, TypeError, ValueError) as exc:
                        key = type(exc).__name__
                        conversion_errors[key] = conversion_errors.get(key, 0) + 1
                        conversion_rejected.append(
                            {
                                "source_row_index": source_index,
                                "source_uuid": item.get("uuid"),
                                "error": f"{key}: {exc}",
                            }
                        )
                        source_index += 1
                        continue
                    code_hash = _sha256_bytes(code.encode())
                    stable_uuid = f"collected87k_{item['uuid']}"
                    source_name = str(item["source"])
                    converted_batch.append(
                        _drkernel_row(
                            code=code,
                            prompt_prefix=prompt_prefix,
                            data_source="cuda_llm_colleague_collected_torch_ops",
                            level=source_name,
                            repo_name=f"colleague_collection/{source_name}",
                            task_type="colleague_collected_torch_ops",
                            uuid=stable_uuid,
                        )
                    )
                    provenance = {
                        "converted_row_index": output_rows,
                        "converted_uuid": stable_uuid,
                        "source_row_index": source_index,
                        "source": "internal_colleague_delivery",
                        "source_revision": revision,
                        "source_label": source_name,
                        "source_uuid": item["uuid"],
                        "source_kernel": item.get("kernel"),
                        "source_ops_valid": item["ops_valid"],
                        "source_difficulty_score": item.get("difficulty_score"),
                        "source_op_count": item.get("op_count"),
                        "source_missing_ops": item.get("missing_ops"),
                        "licenses": [],
                        "provenance_status": "upstream URL, revision, and license absent from delivery",
                        "normalized_code_sha256": code_hash,
                    }
                    provenance_handle.write(json.dumps(provenance, ensure_ascii=False, sort_keys=True) + "\n")
                    output_rows += 1
                    source_index += 1
                if converted_batch:
                    writer.write_table(pa.Table.from_pylist(converted_batch, schema=schema))
    except BaseException:
        writer.close()
        output_tmp.unlink(missing_ok=True)
        provenance_tmp.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(output_tmp, output)
    os.replace(provenance_tmp, provenance_path)
    summary = {
        "contract_version": 1,
        "source_format": "collected-torch-ops",
        "source": str(path.resolve()),
        "source_revision": revision,
        "output": str(output.resolve()),
        "output_rows": output_rows,
        "output_sha256": _sha256_file(output),
        "provenance_jsonl": str(provenance_path.resolve()),
        "provenance_sha256": _sha256_file(provenance_path),
        "source_rows": parquet.metadata.num_rows,
        "conversion_errors": conversion_errors,
        "conversion_rejected": conversion_rejected,
        "source_sha256": _sha256_file(path),
        "provenance_status": "incomplete: coarse source labels only",
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-format",
        required=True,
        choices=(
            "kernelbook",
            "multikernelbench",
            "cuda-agent-ops",
            "accepted-jsonl",
            "kernelbench",
            "collected-torch-ops",
        ),
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--template-parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--include-npu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {args.output}")
    schema, prompt_prefix = _template(args.template_parquet)
    if args.source_format == "collected-torch-ops":
        summary = _write_collected_torch_ops_streaming(
            path=args.input,
            output=args.output,
            schema=schema,
            prompt_prefix=prompt_prefix,
            revision=args.source_revision,
            compression=args.compression,
        )
        summary.update(
            {
                "template_parquet": str(args.template_parquet.resolve()),
                "template_sha256": _sha256_file(args.template_parquet),
            }
        )
        summary_path = args.output.with_name(f"{args.output.stem}.conversion_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.source_format == "kernelbook":
        rows, provenance, source_summary = _kernelbook_rows(args.input, prompt_prefix, args.source_revision)
    elif args.source_format == "multikernelbench":
        rows, provenance, source_summary = _multikernelbench_rows(
            args.input,
            prompt_prefix,
            args.source_revision,
            include_npu=args.include_npu,
        )
    elif args.source_format == "cuda-agent-ops":
        rows, provenance, source_summary = _cuda_agent_ops_rows(
            args.input,
            prompt_prefix,
            args.source_revision,
        )
    elif args.source_format == "accepted-jsonl":
        rows, provenance, source_summary = _accepted_jsonl_rows(
            args.input,
            prompt_prefix,
            args.source_revision,
        )
    elif args.source_format == "kernelbench":
        rows, provenance, source_summary = _kernelbench_rows(
            args.input,
            prompt_prefix,
            args.source_revision,
        )
    else:  # pragma: no cover - argparse choices make this unreachable
        raise AssertionError(args.source_format)
    if not rows:
        raise ValueError("conversion produced zero rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output.with_name(f"{args.output.stem}.provenance.jsonl")
    summary_path = args.output.with_name(f"{args.output.stem}.conversion_summary.json")
    output_tmp = args.output.with_name(f".{args.output.name}.tmp")
    provenance_tmp = provenance_path.with_name(f".{provenance_path.name}.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output_tmp, compression=args.compression)
    with provenance_tmp.open("w", encoding="utf-8") as handle:
        for record in provenance:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(output_tmp, args.output)
    os.replace(provenance_tmp, provenance_path)
    summary = {
        "contract_version": 1,
        "source_format": args.source_format,
        "source": str(args.input.resolve()),
        "source_revision": args.source_revision,
        "template_parquet": str(args.template_parquet.resolve()),
        "template_sha256": _sha256_file(args.template_parquet),
        "output": str(args.output.resolve()),
        "output_rows": len(rows),
        "output_sha256": _sha256_file(args.output),
        "provenance_jsonl": str(provenance_path.resolve()),
        "provenance_sha256": _sha256_file(provenance_path),
        **source_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
