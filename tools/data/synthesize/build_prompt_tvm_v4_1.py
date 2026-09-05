#!/usr/bin/env python3
"""Derive prompt_tvm_v4_1 by adding only the FP32 reference TF32 notice."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "prompt_tvm_v4_1_tf32_notice"
RELEASE_STATUS = "review_candidate_prompt_tvm_v4_1"
EXPECTED_PARENT_ROWS = 39_636
EXPECTED_PARENT_PARQUET_SHA256 = "189da56dca3acbb03b5532b360ec1eb9adde75c173952149a8515dcebfb08f79"
EXPECTED_PARENT_MANIFEST_SHA256 = "7e8d6b9bcfd25395dc25611f781f0954c335e0d8d4139250b1351a431f8e3385"
EXPECTED_L3_ROWS = 50
EXPECTED_L3_PARQUET_SHA256 = "6b3c85f2e57f307036b8c38ff26891e8b88ac44afb702a7c7f539d2b8307755e"

PROMPT_ANCHOR = "You are a PyTorch and CUDA expert. Optimize the given PyTorch `Model` by returning a CUDA extension implementation that is faster than the original while preserving correctness."
ENVIRONMENT_NOTICE = """Evaluation environment:
  For FP32 evaluation, TF32 is enabled in the reference implementation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rewrite_prompt(prompt: Any, row_index: int) -> tuple[list[dict[str, Any]], str, str]:
    if not isinstance(prompt, list) or not prompt:
        raise TypeError(f"row {row_index}: prompt must be a non-empty conversation list")
    rewritten = [dict(message) for message in prompt]
    first = rewritten[0]
    content = first.get("content")
    if first.get("role") != "user" or not isinstance(content, str):
        raise TypeError(f"row {row_index}: first prompt message must be user text")
    if not content.startswith(PROMPT_ANCHOR + "\n\n"):
        raise ValueError(f"row {row_index}: missing expected v4 prompt introduction")
    if ENVIRONMENT_NOTICE in content:
        raise ValueError(f"row {row_index}: TF32 notice already present")
    updated = PROMPT_ANCHOR + "\n\n" + ENVIRONMENT_NOTICE + content[len(PROMPT_ANCHOR) :]
    first["content"] = updated
    return rewritten, sha256_text(content), sha256_text(updated)


def _validate_source_template(repo_root: Path) -> dict[str, str]:
    template_path = repo_root / "examples/kernel_agent/prompt_config/initial_prompt/first_turn_tvm.yaml"
    template = template_path.read_text()
    if not all(line in template for line in ENVIRONMENT_NOTICE.splitlines()):
        raise ValueError("first-turn template is missing the exact TF32 notice")
    return {"path": str(template_path.relative_to(repo_root)), "sha256": sha256_file(template_path)}


def _metadata(
    parent: pa.Schema, parent_parquet: Path, parent_manifest: Path, template: dict[str, str]
) -> dict[bytes, bytes]:
    metadata = dict(parent.metadata or {})
    metadata.update(
        {
            b"release.contract": CONTRACT.encode(),
            b"release.status": RELEASE_STATUS.encode(),
            b"release.review_only": b"true",
            b"release.training_approved": b"false",
            b"release.parent_parquet_sha256": sha256_file(parent_parquet).encode(),
            b"release.parent_manifest_sha256": sha256_file(parent_manifest).encode(),
            b"release.prompt_block_sha256": sha256_text(ENVIRONMENT_NOTICE).encode(),
            b"release.prompt_template_sha256": template["sha256"].encode(),
        }
    )
    return metadata


def _rewrite_table(table: pa.Table) -> tuple[pa.Table, list[tuple[str, str]]]:
    prompt_index = table.schema.get_field_index("prompt")
    if prompt_index < 0:
        raise ValueError("parent parquet has no prompt column")
    child_prompts: list[list[dict[str, Any]]] = []
    prompt_hashes: list[tuple[str, str]] = []
    for row_index, prompt in enumerate(table.column(prompt_index).to_pylist()):
        rewritten, parent_hash, child_hash = rewrite_prompt(prompt, row_index)
        child_prompts.append(rewritten)
        prompt_hashes.append((parent_hash, child_hash))
    if len({child for _, child in prompt_hashes}) != table.num_rows:
        raise ValueError("prompt rewrite produced duplicate prompt hashes")
    child_prompt_array = pa.array(child_prompts, type=table.schema.field(prompt_index).type)
    child = table.set_column(prompt_index, table.schema.field(prompt_index), child_prompt_array)
    if not child.drop(["prompt"]).equals(table.drop(["prompt"]), check_metadata=True):
        raise ValueError("non-prompt data changed")
    return child, prompt_hashes


def verify_written(parent: pa.Table, child_path: Path) -> None:
    child = pq.read_table(child_path)
    if not child.drop(["prompt"]).equals(parent.drop(["prompt"]), check_metadata=False):
        raise ValueError("written non-prompt columns differ from the parent")
    for row_index, (before, after) in enumerate(
        zip(parent.column("prompt").to_pylist(), child.column("prompt").to_pylist(), strict=True)
    ):
        restored = [dict(message) for message in after]
        text = restored[0]["content"]
        inserted = "\n\n" + ENVIRONMENT_NOTICE
        if text.count(inserted) != 1:
            raise ValueError(f"row {row_index}: notice must occur exactly once")
        restored[0]["content"] = text.replace(inserted, "", 1)
        if restored != before:
            raise ValueError(f"row {row_index}: prompt changed beyond the TF32 insertion")


def write_review_examples(parent: pa.Table, child: pa.Table, destination: Path) -> None:
    destination.mkdir()
    for row_index in (0, parent.num_rows // 2, parent.num_rows - 1):
        before = parent.column("prompt")[row_index].as_py()[0]["content"]
        after = child.column("prompt")[row_index].as_py()[0]["content"]
        (destination / f"row_{row_index}_v4.txt").write_text(before)
        (destination / f"row_{row_index}_v4_1.txt").write_text(after)
        diff = difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="v4", tofile="v4_1"
        )
        (destination / f"row_{row_index}.diff").write_text("".join(diff))


def build(repo_root: Path, parent_release: Path, output_release: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    parent_release = parent_release.resolve()
    output_release = output_release.resolve()
    if output_release.exists():
        raise FileExistsError(output_release)

    parent_parquet = parent_release / "train.parquet"
    parent_manifest = parent_release / "manifest.jsonl"
    parent_summary = parent_release / "summary.json"
    actual_parent_hash = sha256_file(parent_parquet)
    actual_manifest_hash = sha256_file(parent_manifest)
    if actual_parent_hash != EXPECTED_PARENT_PARQUET_SHA256:
        raise ValueError(f"unexpected parent parquet SHA-256: {actual_parent_hash}")
    if actual_manifest_hash != EXPECTED_PARENT_MANIFEST_SHA256:
        raise ValueError(f"unexpected parent manifest SHA-256: {actual_manifest_hash}")

    template = _validate_source_template(repo_root)
    table = pq.read_table(parent_parquet)
    if table.num_rows != EXPECTED_PARENT_ROWS:
        raise ValueError(f"unexpected parent rows: {table.num_rows}")

    child, prompt_hashes = _rewrite_table(table)
    child = child.replace_schema_metadata(_metadata(table.schema, parent_parquet, parent_manifest, template))

    output_release.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_release.parent.name}.build.", dir=output_release.parent) as tmp:
        staging = Path(tmp) / "release"
        staging.mkdir()
        child_parquet = staging / "train.parquet"
        child_manifest = staging / "manifest.jsonl"
        pq.write_table(child, child_parquet, compression="zstd", row_group_size=table.num_rows)
        verify_written(table, child_parquet)
        write_review_examples(table, child, staging / "review_samples")

        with parent_manifest.open() as source, child_manifest.open("w") as destination:
            for row_index, (line, hashes) in enumerate(zip(source, prompt_hashes, strict=True)):
                parent_record = json.loads(line)
                if parent_record.get("output_position") != row_index:
                    raise ValueError(f"manifest output_position mismatch at row {row_index}")
                record = dict(parent_record)
                record.update(
                    {
                        "contract": CONTRACT,
                        "release_status": RELEASE_STATUS,
                        "parent_contract": parent_record.get("contract"),
                        "parent_release_status": parent_record.get("release_status"),
                        "parent_manifest_row_sha256": sha256_text(line.rstrip("\n")),
                        "parent_prompt_sha256": hashes[0],
                        "prompt_sha256": hashes[1],
                        "prompt_contract": CONTRACT,
                        "training_approved": False,
                    }
                )
                destination.write(canonical_json(record) + "\n")

        summary = {
            "contract": CONTRACT,
            "release_status": RELEASE_STATUS,
            "review_only": True,
            "training_approved": False,
            "rows": table.num_rows,
            "transformation": "prompt-only insertion of the exact TF32 environment notice",
            "invariants": {
                "row_count_unchanged": True,
                "row_order_unchanged": True,
                "non_prompt_columns_unchanged": True,
                "notice_occurrences_per_row": 1,
                "removing_notice_recovers_parent_prompt_exactly": True,
                "unique_child_prompt_sha256": len({child_hash for _, child_hash in prompt_hashes}),
            },
            "prompt_contract": {
                "id": CONTRACT,
                "insertion_anchor": PROMPT_ANCHOR,
                "inserted_text": ENVIRONMENT_NOTICE,
                "inserted_text_sha256": sha256_text(ENVIRONMENT_NOTICE),
                "source_template": template,
            },
            "source_binding": {
                "parent_parquet": {
                    "path": str(parent_parquet),
                    "sha256": actual_parent_hash,
                },
                "parent_manifest": {
                    "path": str(parent_manifest),
                    "sha256": actual_manifest_hash,
                },
                "parent_summary": {
                    "path": str(parent_summary),
                    "sha256": sha256_file(parent_summary),
                },
                "builder": {
                    "path": str(Path(__file__).resolve().relative_to(repo_root)),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
            "artifacts": {
                "train_parquet": {
                    "path": str(output_release / "train.parquet"),
                    "rows": table.num_rows,
                    "sha256": sha256_file(child_parquet),
                },
                "manifest": {
                    "path": str(output_release / "manifest.jsonl"),
                    "rows": table.num_rows,
                    "sha256": sha256_file(child_manifest),
                },
            },
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (staging / "README.md").write_text(
            "# prompt_tvm_v4_1 review candidate\n\n"
            "This is a prompt-only derivative of `Data/prompt_tvm_v4/release/train.parquet`. "
            "The only prompt change is the TF32 environment notice below; all original v4 "
            "instructions, task order, references, labels, and other columns are preserved.\n\n"
            f"```text\n{ENVIRONMENT_NOTICE}\n```\n\n"
            "See `summary.json`, `manifest.jsonl`, and `review_samples/` for provenance and exact diffs.\n"
        )
        os.replace(staging, output_release)
    return summary


def build_l3(repo_root: Path, parent_parquet: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    parent_parquet = parent_parquet.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    actual_parent_hash = sha256_file(parent_parquet)
    if actual_parent_hash != EXPECTED_L3_PARQUET_SHA256:
        raise ValueError(f"unexpected L3 parent SHA-256: {actual_parent_hash}")
    template = _validate_source_template(repo_root)
    table = pq.read_table(parent_parquet)
    if table.num_rows != EXPECTED_L3_ROWS:
        raise ValueError(f"unexpected L3 parent rows: {table.num_rows}")
    child, prompt_hashes = _rewrite_table(table)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"eval.prompt_contract": CONTRACT.encode(),
            b"eval.parent_parquet_sha256": actual_parent_hash.encode(),
            b"eval.prompt_block_sha256": sha256_text(ENVIRONMENT_NOTICE).encode(),
            b"eval.prompt_template_sha256": template["sha256"].encode(),
        }
    )
    child = child.replace_schema_metadata(metadata)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.build.", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / output_dir.name
        staging.mkdir()
        child_parquet = staging / "train.parquet"
        pq.write_table(child, child_parquet, compression="zstd", row_group_size=table.num_rows)
        verify_written(table, child_parquet)
        write_review_examples(table, child, staging / "review_samples")
        summary = {
            "contract": CONTRACT,
            "dataset": "kernelbench_level3_validation_tvm_v4_1",
            "rows": table.num_rows,
            "transformation": "prompt-only insertion of the exact TF32 environment notice",
            "invariants": {
                "row_count_unchanged": True,
                "row_order_unchanged": True,
                "non_prompt_columns_unchanged": True,
                "notice_occurrences_per_row": 1,
                "removing_notice_recovers_parent_prompt_exactly": True,
                "unique_child_prompt_sha256": len({child_hash for _, child_hash in prompt_hashes}),
            },
            "prompt_contract": {
                "id": CONTRACT,
                "insertion_anchor": PROMPT_ANCHOR,
                "inserted_text": ENVIRONMENT_NOTICE,
                "inserted_text_sha256": sha256_text(ENVIRONMENT_NOTICE),
                "source_template": template,
            },
            "source_binding": {
                "parent_parquet": {"path": str(parent_parquet), "sha256": actual_parent_hash},
                "builder": {
                    "path": str(Path(__file__).resolve().relative_to(repo_root)),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
            "artifact": {
                "path": str(output_dir / "train.parquet"),
                "rows": table.num_rows,
                "sha256": sha256_file(child_parquet),
            },
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (staging / "README.md").write_text(
            "# KernelBench Level 3 validation prompt v4.1\n\n"
            "This 50-row evaluation set is a prompt-only derivative of "
            "`Data/kernelbench-level3-validation-tvm-v2/train.parquet`. The task rows, order, "
            "references, labels, and metadata are unchanged. Results produced from this dataset "
            "belong to the prompt-v4.1 protocol and are not a prompt-controlled comparison with "
            "older evaluations.\n"
        )
        os.replace(staging, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--parent-release", type=Path, default=repo_root / "Data/prompt_tvm_v4/release")
    parser.add_argument("--output-release", type=Path, default=repo_root / "Data/prompt_tvm_v4_1/release")
    parser.add_argument(
        "--l3-parent",
        type=Path,
        default=repo_root / "Data/kernelbench-level3-validation-tvm-v2/train.parquet",
    )
    parser.add_argument(
        "--l3-output-dir",
        type=Path,
        default=repo_root / "Data/kernelbench-level3-validation-tvm-v4_1",
    )
    parser.add_argument("--scope", choices=("all", "train", "l3"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {}
    if args.scope in {"all", "train"}:
        result["train"] = build(args.repo_root, args.parent_release, args.output_release)["artifacts"]
    if args.scope in {"all", "l3"}:
        result["l3"] = build_l3(args.repo_root, args.l3_parent, args.l3_output_dir)["artifact"]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
