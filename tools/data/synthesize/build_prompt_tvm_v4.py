#!/usr/bin/env python3
"""Build the first prompt_tvm_v4 review candidate from audited v3 leaves.

The builder is intentionally an assembly step.  It does not rerun cleanup or
invent a new deduplication policy.  It preserves the published v3 source/mode
partitions, injects the current KernelGym execution-mode contract into every
training prompt, and materializes row-level lineage that v3 only encoded in
file paths and audit ledgers.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tools.data.cleaning.complexity import extract_operator_signature

CONTRACT_VERSION = "prompt_tvm_v4_union_review_v1"
MODE_CONTRACT_ID = "kernelgym_default_train_persistent_state_v1"
MODE_CONTRACT_MARKER = "Execution-mode contract for this task (prompt_tvm_v4):"
MODE_CONTRACT_TEXT = """Execution-mode contract for this task (prompt_tvm_v4):
- Correctness evaluates both the reference `Model` and `ModelNew` in PyTorch's default training mode (`training=True`); `.eval()` is not called.
- Model instances persist across correctness trials. The Torch RNG seed is reset before each paired reference and candidate forward.
- Preserve training-mode outputs, RNG-dependent behavior, and mutable parameter or buffer state."""


@dataclasses.dataclass(frozen=True)
class PartitionSpec:
    name: str
    relative_path: str
    source_family: str
    mode_class: str
    expected_rows: int
    expected_sha256: str
    audit_family: str
    include_in_review_train: bool = True


PARTITIONS = (
    PartitionSpec(
        "drkernel_mode_same_base",
        "drkernel_rl_thinking.parquet",
        "drkernel",
        "mode_same_observed",
        40_307,
        "9e9ffca46022e74c0616f5e272871e76dfd000b6e7937b01685cd0bb11d521e5",
        "drkernel",
    ),
    PartitionSpec(
        "drkernel_mode_same_recovered",
        "recovered_timeout_h200_20260731/drkernel/mode_same_observed.parquet",
        "drkernel",
        "mode_same_observed",
        120,
        "e460fc178145bf02de91ea4262bc3e0a7d67ffaf10e97c1476825db43e0bdb4c",
        "recovered",
    ),
    PartitionSpec(
        "drkernel_mode_variant_base",
        "analysis/drkernel_rl_thinking.mode_variant.parquet",
        "drkernel",
        "mode_variant",
        5_039,
        "c1e992f11646a1a15aea3e00434dc6066a1d92f0dbeab4435896e77ddbd7f787",
        "drkernel",
    ),
    PartitionSpec(
        "drkernel_mode_variant_recovered",
        "recovered_timeout_h200_20260731/drkernel/mode_variant.parquet",
        "drkernel",
        "mode_variant",
        39,
        "f0fffc8bc3c69ba93efce648834af814103bcbfb316d9ff194b16d524cc19d21",
        "recovered",
    ),
    PartitionSpec(
        "cuda_agent_mode_same_base",
        "external/cuda_agent_ops_6k/analysis/mode_same_observed.parquet",
        "cuda_agent",
        "mode_same_observed",
        4_498,
        "384a26720c9e7c6665e8f4c5177627197a41b23b81fe090cc7857b61feef6c76",
        "cuda_agent",
    ),
    PartitionSpec(
        "cuda_agent_mode_same_recovered",
        "recovered_timeout_h200_20260731/external/cuda_agent/mode_same_observed.parquet",
        "cuda_agent",
        "mode_same_observed",
        53,
        "822455fcc0b0ceaf49e0073d917c19a92fd630680b78c5acd3dc9fe405a3ff53",
        "recovered",
    ),
    PartitionSpec(
        "cuda_agent_mode_variant_base",
        "external/cuda_agent_ops_6k/analysis/mode_variant.parquet",
        "cuda_agent",
        "mode_variant",
        439,
        "0055255dce4ae3f4fa241d1deace781753658fa978e931782b0d12f92f71d4d0",
        "cuda_agent",
    ),
    PartitionSpec(
        "cuda_agent_mode_variant_recovered",
        "recovered_timeout_h200_20260731/external/cuda_agent/mode_variant.parquet",
        "cuda_agent",
        "mode_variant",
        5,
        "8fa4c67438897321e2fbcda563f359c7bc37802daa79e6b3279eafff052ed6a0",
        "recovered",
    ),
    PartitionSpec(
        "kernelbook_mode_same_base",
        "external/kernelbook/analysis/mode_same_observed.parquet",
        "kernelbook",
        "mode_same_observed",
        10_893,
        "14e9d7114bd0f093df9b94242c32335bc8f7c211559cdeb041bbd13533397a00",
        "kernelbook",
    ),
    PartitionSpec(
        "kernelbook_mode_same_recovered",
        "recovered_timeout_h200_20260731/external/kernelbook/mode_same_observed.parquet",
        "kernelbook",
        "mode_same_observed",
        4,
        "3221e538bc7eb8d8e51d0275c692673ad3986fc83649f69b4f209a62d0cc7144",
        "recovered",
    ),
    PartitionSpec(
        "kernelbook_mode_variant_base",
        "external/kernelbook/analysis/mode_variant.parquet",
        "kernelbook",
        "mode_variant",
        917,
        "06ebfabfa519e2b48d6585f491c2c2209d178f795f434f48fc3a071988197a76",
        "kernelbook",
    ),
    PartitionSpec(
        "oubo_mode_same_deduplicated",
        "oubo_generate_accepted/training/mode_same_observed.parquet",
        "oubo_generated",
        "mode_same_observed",
        1_641,
        "b590256457d408f673aff3bc560e2606b1f144f66eb4063d28a1a877875900d3",
        "oubo",
    ),
    PartitionSpec(
        "oubo_mode_variant_base",
        "oubo_generate_accepted/analysis/mode_variant.parquet",
        "oubo_generated",
        "mode_variant",
        360,
        "e1a9dea145d00e56a97eec8973c7316bbe7ebe8e44b395479c7b521c33b8e8b9",
        "oubo",
    ),
    PartitionSpec(
        "kernelbook_mode_inconclusive",
        "external/kernelbook/analysis/mode_inconclusive.parquet",
        "kernelbook",
        "mode_inconclusive",
        6,
        "93bdc4425cbfbe8066376ad35f5199cdb2fee249c23878ad000773e0557aef6b",
        "kernelbook",
        include_in_review_train=False,
    ),
)

AUDIT_PATHS = {
    "drkernel": "local_artifacts/data_handoffs/drkernel_gpu_v5_full_20260730.audit.jsonl",
    "cuda_agent": "local_artifacts/external_gpu_v5_20260730/cuda_agent/full.audit.jsonl",
    "kernelbook": "local_artifacts/external_gpu_v5_20260730/kernelbook/full.audit.jsonl",
    "oubo": "local_artifacts/data_handoffs/accepted_gpu_v5_full_20260729.audit.jsonl",
    "recovered": "local_artifacts/h200_timeout_recheck_20260731/results300/all_timeouts.audit.jsonl",
}

PROVENANCE_PATHS = {
    "cuda_agent": "external/cuda_agent_ops_6k/source/provenance.jsonl",
    "kernelbook": "external/kernelbook/source/provenance.jsonl",
    "oubo_generated": "oubo_generate_accepted/source/provenance.jsonl",
}

RECOVERED_PASSED_PATHS = (
    "recovered_timeout_h200_20260731/drkernel/passed.parquet",
    "recovered_timeout_h200_20260731/external/cuda_agent/passed.parquet",
    "recovered_timeout_h200_20260731/external/kernelbook/passed.parquet",
    "recovered_timeout_h200_20260731/oubo_generate_accepted/passed.parquet",
)

SOURCE_DEFAULTS = {
    "drkernel": {
        "source_url": "https://huggingface.co/datasets/hkust-nlp/drkernel-rl-data",
        "source_revision": "prompt_tvm_v3",
        "licenses": [],
        "provenance_status": "inherited from the existing active v3 lineage; not reassessed by this assembly",
        "governance_status": "existing_active_or_gpu_audited",
    },
    "cuda_agent": {
        "source_url": "https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K",
        "source_revision": "44a734c78c947bfcba5189cbfd13f57a6d29a698",
        "licenses": ["CC-BY-4.0"],
        "provenance_status": "row-level source attribution available",
        "governance_status": "gpu_audited_pending_mixture_approval",
    },
    "kernelbook": {
        "source_url": "https://huggingface.co/datasets/GPUMODE/KernelBook",
        "source_revision": "b76504d85f7f14ef4b1fad81f136f638f2ce625b",
        "licenses": ["June 9 Researcher Reciprocity License 1.0", "per-row source license"],
        "provenance_status": "row-level repository revision and license available",
        "governance_status": "review_only_pending_license_approval_and_repository_group_split",
    },
    "oubo_generated": {
        "source_url": "local accepted.jsonl delivery",
        "source_revision": "accepted-delivery-20260729-bc85b5022654",
        "licenses": [],
        "provenance_status": "generation prompt, model revision, upstream identity, and license absent from delivery",
        "governance_status": "review_only_missing_upstream_provenance_and_license",
    },
}

MODE_OUTPUTS = {
    "mode_same_observed": "analysis/mode_same_observed.parquet",
    "mode_variant": "analysis/mode_variant.review.parquet",
    "mode_inconclusive": "analysis/mode_inconclusive.review.parquet",
}

V4_METADATA_TYPE = pa.struct(
    [
        pa.field("contract_version", pa.string()),
        pa.field("mode_contract", pa.string()),
        pa.field("row_index", pa.int64()),
        pa.field("source_family", pa.string()),
        pa.field("source_partition", pa.string()),
        pa.field("source_partition_row_index", pa.int64()),
        pa.field("mode_class", pa.string()),
        pa.field("mode_flags", pa.list_(pa.string())),
        pa.field("recovered_timeout", pa.bool_()),
        pa.field("gpu_audit_budget_seconds", pa.int32()),
        pa.field("included_in_review_train", pa.bool_()),
        pa.field("runtime_validation_status", pa.string()),
        pa.field("governance_status", pa.string()),
        pa.field("parent_uuid", pa.string()),
        pa.field("reference_sha256", pa.string()),
        pa.field("normalized_ast_sha256", pa.string()),
        pa.field("operator_count", pa.int32()),
        pa.field("operator_bucket", pa.string()),
        pa.field("source_group_id", pa.string()),
        pa.field("source_url", pa.string()),
        pa.field("source_revision", pa.string()),
        pa.field("source_row_index", pa.int64()),
        pa.field("source_uuid", pa.string()),
        pa.field("repo_link", pa.string()),
        pa.field("repo_revision", pa.string()),
        pa.field("licenses", pa.list_(pa.string())),
        pa.field("provenance_status", pa.string()),
        pa.field("prompt_contract_injected", pa.bool_()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def _load_kept_audit(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl(path):
        if not record.get("keep"):
            continue
        repairs = record.get("repairs") if isinstance(record.get("repairs"), dict) else {}
        effective_uuid = repairs.get("uuid") or record.get("uuid")
        if not isinstance(effective_uuid, str) or not effective_uuid:
            raise ValueError(f"kept audit record has no effective UUID: {path}")
        if effective_uuid in records:
            raise ValueError(f"duplicate kept effective UUID {effective_uuid!r}: {path}")
        records[effective_uuid] = record
    return records


def _load_provenance(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl(path):
        uuid = record.get("converted_uuid")
        if not isinstance(uuid, str) or not uuid:
            raise ValueError(f"provenance record has no converted_uuid: {path}")
        if uuid in records:
            raise ValueError(f"duplicate provenance UUID {uuid!r}: {path}")
        records[uuid] = record
    return records


def _load_recovered_uuids(input_root: Path) -> set[str]:
    recovered: set[str] = set()
    for relative_path in RECOVERED_PASSED_PATHS:
        path = input_root / relative_path
        for batch in pq.ParquetFile(path).iter_batches(columns=["extra_info"], batch_size=1024):
            for extra_info in batch.column(0).to_pylist():
                uuid = extra_info.get("uuid") if isinstance(extra_info, dict) else None
                if not isinstance(uuid, str) or not uuid:
                    raise ValueError(f"recovered row has no UUID: {path}")
                if uuid in recovered:
                    raise ValueError(f"duplicate recovered UUID {uuid!r}")
                recovered.add(uuid)
    return recovered


def inject_mode_contract(prompt: Any) -> Any:
    """Return a prompt copy with the v4 train-mode contract after paragraph 1."""

    if not isinstance(prompt, list) or not prompt:
        raise TypeError("prompt must be a non-empty conversation list")
    copied = [dict(message) for message in prompt]
    content = copied[0].get("content")
    if copied[0].get("role") != "user" or not isinstance(content, str):
        raise TypeError("the first prompt message must be a text user message")
    if MODE_CONTRACT_MARKER in content:
        raise ValueError("prompt already contains the prompt_tvm_v4 mode contract")
    boundary = content.find("\n\n")
    if boundary < 0:
        copied[0]["content"] = f"{content}\n\n{MODE_CONTRACT_TEXT}"
    else:
        copied[0]["content"] = f"{content[:boundary]}\n\n{MODE_CONTRACT_TEXT}{content[boundary:]}"
    return copied


def _extend_schema(schema: pa.Schema) -> pa.Schema:
    extra_index = schema.get_field_index("extra_info")
    if extra_index < 0:
        raise ValueError("input schema has no extra_info field")
    extra_field = schema.field(extra_index)
    if not pa.types.is_struct(extra_field.type):
        raise TypeError("extra_info must be a struct")
    if extra_field.type.get_field_index("v4") >= 0:
        raise ValueError("input extra_info already contains a v4 field")
    extended_extra = pa.struct([*extra_field.type, pa.field("v4", V4_METADATA_TYPE)])
    fields = list(schema)
    fields[extra_index] = pa.field(extra_field.name, extended_extra, nullable=extra_field.nullable)
    return pa.schema(fields, metadata=schema.metadata)


def _operator_bucket(count: int) -> str:
    if count <= 1:
        return "op_count_le_1"
    if count <= 5:
        return "op_count_2_to_5"
    return "op_count_ge_6"


def _normalized_ast_sha256(code: str) -> str:
    normalized = ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _validate_mode_flags(mode_class: str, flags: list[str], uuid: str) -> None:
    flag_set = set(flags)
    if mode_class == "mode_same_observed":
        valid = flag_set == {"train_eval_same"}
    elif mode_class == "mode_variant":
        valid = bool(
            flag_set
            & {
                "train_eval_output_diff",
                "train_eval_state_diff",
                "train_only_stochastic",
                "eval_only_failure",
                "train_only_failure",
            }
        ) and not flag_set.intersection({"train_eval_same", "train_eval_inconclusive"})
    elif mode_class == "mode_inconclusive":
        valid = "train_eval_inconclusive" in flag_set
    else:
        raise ValueError(f"unknown mode class {mode_class!r}")
    if not valid:
        raise ValueError(f"audit flags {flags!r} do not match {mode_class} for {uuid}")


def _provenance_for_row(
    source_family: str,
    effective_uuid: str,
    audit_record: Mapping[str, Any],
    provenance_maps: Mapping[str, Mapping[str, dict[str, Any]]],
    extra_info: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = dict(SOURCE_DEFAULTS[source_family])
    parent_uuid = audit_record.get("uuid")
    lookup_uuid = parent_uuid if isinstance(parent_uuid, str) else effective_uuid
    provenance = dict(provenance_maps.get(source_family, {}).get(lookup_uuid, {}))
    if source_family != "drkernel" and not provenance:
        raise ValueError(f"missing provenance for {effective_uuid} (lookup {lookup_uuid})")

    licenses = provenance.get("licenses") or defaults["licenses"]
    repo_name = extra_info.get("repo_name")
    repo_revision = provenance.get("repo_sha")
    source_group_id = repo_name or source_family
    if source_family == "kernelbook" and repo_revision:
        source_group_id = f"{repo_name}@{repo_revision}"
    elif provenance.get("source_data_source"):
        source_group_id = f"{source_family}:{provenance['source_data_source']}"

    return {
        "parent_uuid": lookup_uuid,
        "source_group_id": str(source_group_id),
        "source_url": defaults["source_url"],
        "source_revision": str(provenance.get("source_revision") or defaults["source_revision"]),
        "source_row_index": provenance.get("source_row_index"),
        "source_uuid": None if provenance.get("source_uuid") is None else str(provenance.get("source_uuid")),
        "repo_link": provenance.get("repo_link"),
        "repo_revision": repo_revision,
        "licenses": [str(value) for value in licenses],
        "provenance_status": str(provenance.get("provenance_status") or defaults["provenance_status"]),
        "governance_status": str(defaults["governance_status"]),
    }


def _runtime_validation_status(mode_class: str, recovered: bool) -> str:
    budget = "300s_recovered" if recovered else "base_budget"
    if mode_class == "mode_same_observed":
        return f"gpu_audited_eval_and_train_eval_same_observed:{budget}"
    if mode_class == "mode_variant":
        return f"gpu_audited_eval_train_contract_validation_pending:{budget}"
    return f"gpu_audited_eval_mode_audit_inconclusive:{budget}"


def _writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(path, schema=schema, compression="zstd", use_dictionary=True)


def _prepare_inputs(input_root: Path) -> tuple[pa.Schema, list[dict[str, Any]]]:
    schema: pa.Schema | None = None
    inputs = []
    for spec in PARTITIONS:
        path = input_root / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        actual_rows = parquet.metadata.num_rows
        actual_hash = sha256_file(path)
        if actual_rows != spec.expected_rows:
            raise ValueError(f"{spec.name}: expected {spec.expected_rows} rows, found {actual_rows}")
        if actual_hash != spec.expected_sha256:
            raise ValueError(f"{spec.name}: expected SHA-256 {spec.expected_sha256}, found {actual_hash}")
        current_schema = parquet.schema_arrow
        if schema is None:
            schema = current_schema
        elif not current_schema.equals(schema, check_metadata=False):
            raise ValueError(f"schema mismatch in {path}")
        inputs.append(
            {
                "name": spec.name,
                "path": str(path.resolve()),
                "relative_path": spec.relative_path,
                "rows": actual_rows,
                "sha256": actual_hash,
                "source_family": spec.source_family,
                "mode_class": spec.mode_class,
                "include_in_review_train": spec.include_in_review_train,
            }
        )
    assert schema is not None
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    if set(schema.names) != required:
        raise ValueError(f"unexpected input columns: {schema.names}")
    return schema, inputs


def _build_into(
    repo_root: Path,
    input_root: Path,
    output_dir: Path,
    *,
    logical_output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    input_schema, input_records = _prepare_inputs(input_root)
    output_schema = _extend_schema(input_schema)
    audit_maps = {name: _load_kept_audit(repo_root / relative_path) for name, relative_path in AUDIT_PATHS.items()}
    provenance_maps = {
        name: _load_provenance(input_root / relative_path) for name, relative_path in PROVENANCE_PATHS.items()
    }
    recovered_uuids = _load_recovered_uuids(input_root)

    train_path = output_dir / "train.review.parquet"
    train_writer = _writer(train_path, output_schema)
    mode_writers = {
        mode: _writer(output_dir / relative_path, output_schema) for mode, relative_path in MODE_OUTPUTS.items()
    }

    manifest_rows: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    reference_hashes: set[str] = set()
    seen_asts: set[str] = set()
    train_row_index = 0

    try:
        for spec in PARTITIONS:
            path = input_root / spec.relative_path
            partition_row_index = 0
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size):
                transformed_rows: list[dict[str, Any]] = []
                for row in batch.to_pylist():
                    extra_info = row.get("extra_info")
                    reward_model = row.get("reward_model")
                    if not isinstance(extra_info, dict) or not isinstance(reward_model, dict):
                        raise TypeError(f"invalid nested row in {path}:{partition_row_index}")
                    uuid = extra_info.get("uuid")
                    code = reward_model.get("ground_truth")
                    entry_point = extra_info.get("entry_point") or "Model"
                    if not isinstance(uuid, str) or not uuid:
                        raise ValueError(f"row has no UUID: {path}:{partition_row_index}")
                    if not isinstance(code, str) or not code:
                        raise ValueError(f"row has no ground_truth: {path}:{partition_row_index}")
                    if code not in row["prompt"][0]["content"]:
                        raise ValueError(f"prompt does not contain ground_truth for {uuid}")
                    if uuid in seen_uuids:
                        raise ValueError(f"duplicate UUID in v4 union: {uuid}")
                    reference_hash = hashlib.sha256(code.encode()).hexdigest()
                    normalized_ast_hash = _normalized_ast_sha256(code)
                    if normalized_ast_hash in seen_asts:
                        raise ValueError(f"duplicate normalized AST in v4 union: {uuid}")
                    seen_uuids.add(uuid)
                    reference_hashes.add(reference_hash)
                    seen_asts.add(normalized_ast_hash)

                    recovered = uuid in recovered_uuids
                    audit_family = "recovered" if recovered else spec.audit_family
                    audit_record = audit_maps[audit_family].get(uuid)
                    if audit_record is None:
                        raise ValueError(f"missing {audit_family} audit record for {uuid}")
                    mode_flags = sorted(set(audit_record.get("mode_flags") or []))
                    _validate_mode_flags(spec.mode_class, mode_flags, uuid)
                    provenance = _provenance_for_row(
                        spec.source_family,
                        uuid,
                        audit_record,
                        provenance_maps,
                        extra_info,
                    )
                    signature = extract_operator_signature(code, entry_point)
                    op_count = len(signature)
                    op_bucket = _operator_bucket(op_count)
                    included = spec.include_in_review_train
                    current_train_index = train_row_index if included else None
                    if included:
                        train_row_index += 1

                    v4 = {
                        "contract_version": CONTRACT_VERSION,
                        "mode_contract": MODE_CONTRACT_ID,
                        "row_index": current_train_index,
                        "source_family": spec.source_family,
                        "source_partition": spec.relative_path,
                        "source_partition_row_index": partition_row_index,
                        "mode_class": spec.mode_class,
                        "mode_flags": mode_flags,
                        "recovered_timeout": recovered,
                        "gpu_audit_budget_seconds": 300 if recovered else 60,
                        "included_in_review_train": included,
                        "runtime_validation_status": _runtime_validation_status(spec.mode_class, recovered),
                        "governance_status": provenance["governance_status"],
                        "parent_uuid": provenance["parent_uuid"],
                        "reference_sha256": reference_hash,
                        "normalized_ast_sha256": normalized_ast_hash,
                        "operator_count": op_count,
                        "operator_bucket": op_bucket,
                        "source_group_id": provenance["source_group_id"],
                        "source_url": provenance["source_url"],
                        "source_revision": provenance["source_revision"],
                        "source_row_index": provenance["source_row_index"],
                        "source_uuid": provenance["source_uuid"],
                        "repo_link": provenance["repo_link"],
                        "repo_revision": provenance["repo_revision"],
                        "licenses": provenance["licenses"],
                        "provenance_status": provenance["provenance_status"],
                        "prompt_contract_injected": True,
                    }
                    extended_extra = dict(extra_info)
                    extended_extra["v4"] = v4
                    transformed = dict(row)
                    transformed["prompt"] = inject_mode_contract(row["prompt"])
                    transformed["extra_info"] = extended_extra
                    transformed_rows.append(transformed)

                    row_manifest = {
                        "v4_row_index": current_train_index,
                        "uuid": uuid,
                        **v4,
                    }
                    manifest_rows.append(row_manifest)
                    partition_row_index += 1

                output_batch = pa.RecordBatch.from_pylist(transformed_rows, schema=output_schema)
                mode_writers[spec.mode_class].write_batch(output_batch)
                if spec.include_in_review_train:
                    train_writer.write_batch(output_batch)
    finally:
        train_writer.close()
        for writer in mode_writers.values():
            writer.close()

    row_manifest_path = output_dir / "row_manifest.parquet"
    pq.write_table(pa.Table.from_pylist(manifest_rows), row_manifest_path, compression="zstd", use_dictionary=True)

    licenses_dir = output_dir / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_root / "external/kernelbook/source/LICENSE", licenses_dir / "KernelBook-LICENSE")

    output_relatives = [
        "train.review.parquet",
        "analysis/mode_same_observed.parquet",
        "analysis/mode_variant.review.parquet",
        "analysis/mode_inconclusive.review.parquet",
        "row_manifest.parquet",
        "licenses/KernelBook-LICENSE",
    ]
    output_hashes = {relative: sha256_file(output_dir / relative) for relative in output_relatives}

    audit_inputs = {
        name: {
            "path": relative_path,
            "sha256": sha256_file(repo_root / relative_path),
        }
        for name, relative_path in AUDIT_PATHS.items()
    }
    provenance_inputs = {
        name: {
            "path": relative_path,
            "sha256": sha256_file(input_root / relative_path),
        }
        for name, relative_path in PROVENANCE_PATHS.items()
    }
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "input_root": str(input_root.resolve()),
        "output_root": str(logical_output_dir.resolve()),
        "input_partitions": input_records,
        "audit_inputs": audit_inputs,
        "provenance_inputs": provenance_inputs,
        "outputs": {
            relative: {
                "path": relative,
                "sha256": digest,
                **(
                    {"rows": pq.ParquetFile(output_dir / relative).metadata.num_rows}
                    if relative.endswith(".parquet")
                    else {}
                ),
            }
            for relative, digest in sorted(output_hashes.items())
        },
        "invariants": {
            "review_train_rows": train_row_index,
            "all_review_rows": len(manifest_rows),
            "unique_uuid_count": len(seen_uuids),
            "unique_reference_sha256_count": len(reference_hashes),
            "unique_normalized_ast_sha256_count": len(seen_asts),
            "recovered_uuid_count": len(recovered_uuids),
            "prompt_contract": MODE_CONTRACT_ID,
            "prompt_contract_injected_rows": len(manifest_rows),
            "ground_truth_code_changed": False,
            "launcher_changed": False,
        },
        "review_boundaries": [
            "Mode-variant rows are included but still need full formal persistent train-mode validation.",
            "Oubo-generated rows remain blocked on upstream provenance and license.",
            "KernelBook remains blocked on license approval and repository-grouped split policy.",
            "Semantic and input-intervention coverage expansion remains a separate evidence-gated partition.",
        ],
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_prompt_tvm_v4(
    *,
    repo_root: Path,
    input_root: Path,
    output_dir: Path,
    batch_size: int = 512,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.build.", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        manifest = _build_into(
            repo_root,
            input_root,
            staging,
            logical_output_dir=output_dir,
            batch_size=batch_size,
        )
        os.replace(staging, output_dir)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--input-root", type=Path, default=repo_root / "Data/prompt_tvm_v3")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "Data/prompt_tvm_v4")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    manifest = build_prompt_tvm_v4(
        repo_root=args.repo_root,
        input_root=args.input_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest["invariants"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
