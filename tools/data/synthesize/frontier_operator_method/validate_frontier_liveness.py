#!/usr/bin/env python3
"""CUDA liveness proof for the exact-1k frontier-operator canary.

This is intentionally a thin, source-bound wrapper around the semantic
operator validator.  It retains that validator's subprocess isolation,
train-mode/state checks, single dense Tensor return rule, and fail-closed
tainted-dispatch provenance propagation, while replacing only the registry
contract and adding the frontier ATen dispatch registry.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# This is deliberately a literal, reviewed binding rather than a dynamically
# discovered compatibility claim.  Any semantic-validator edit requires an
# explicit frontier-wrapper review/update before it can execute a canary.
SEMANTIC_VALIDATOR_SOURCE_SHA256 = "b0177d19377f71ae32aac13698079f858942ad9f95f866a2192bfa50f933b7ff"
_SEMANTIC_VALIDATOR_PATH = _REPO_ROOT / "tools/data/synthesize/semantic_operator_method/validate_semantic_liveness.py"

from tools.data.synthesize.frontier_operator_method import generate_frontier_operator as generator  # noqa: E402
from tools.data.synthesize.semantic_operator_method import validate_semantic_liveness as _semantic  # noqa: E402

CONTRACT_VERSION = "frontier_operator_runtime_liveness_v1"
RUN_BINDING_VERSION = "frontier_operator_runtime_binding_v1"
RESULT_MARKER = "__FRONTIER_OPERATOR_RESULT__="
MAX_AUTHORIZED_CANDIDATES = 1_000
MIN_LIVENESS_TRIALS = 3
REQUIRED_LIVENESS_SEED = 17

# These identities are value-propagating only after their inputs have been
# tainted by a declared operator.  All other tainted dispatches still fail
# closed in the underlying validator.  The list includes the A800-observed
# spellings for sparse/FFT/geometry/linalg plus public ATen variants used by
# the closed frontier registry.
_FRONTIER_VALUE_SCHEMAS = frozenset(
    {
        # Sparse construction, conversion, metadata extraction, and compute.
        "aten::_cslt_compress",
        "aten::_coalesce",
        "aten::_embedding_bag_forward_only",
        "aten::_embedding_bag",
        "aten::_fused_rms_norm",
        # Public F.rms_norm may remain visible rather than lowering to the
        # fused spelling on a supported runtime; it is reviewed as a
        # value-preserving normalization before the declared fused projection.
        "aten::rms_norm",
        "aten::_safe_softmax",
        "aten::_sparse_addmm",
        "aten::_sparse_coo_tensor_with_dims_and_tensors",
        "aten::_sparse_mm",
        "aten::_sparse_softmax",
        "aten::_sparse_sum",
        "aten::_to_dense",
        "aten::_to_copy",
        "aten::_unsafe_view",
        "aten::_to_sparse_semi_structured",
        "aten::_sparse_semi_structured_addmm",
        "aten::_sparse_semi_structured_mm",
        "aten::coalesce",
        "aten::col_indices",
        "aten::crow_indices",
        "aten::indices",
        "aten::sparse_compressed_tensor",
        "aten::sparse_coo_tensor",
        "aten::sparse_sampled_addmm",
        "aten::to_sparse",
        "aten::to_sparse_bsc",
        "aten::to_sparse_bsr",
        "aten::to_sparse_csc",
        "aten::to_sparse_csr",
        "aten::values",
        # Ragged, graph, retrieval, and data-dependent routing.
        "aten::bincount",
        "aten::embedding_bag",
        "aten::index",
        "aten::index_add",
        "aten::index_put",
        "aten::repeat_interleave",
        "aten::scatter_add",
        "aten::scatter_reduce",
        "aten::segment_reduce",
        "aten::sort",
        "aten::topk",
        "aten::unique",
        "aten::unique_consecutive",
        # Quantization and explicit Q/DQ/fake-quant paths.
        "aten::dequantize",
        "aten::fake_quantize_per_channel_affine",
        "aten::fake_quantize_per_tensor_affine",
        "aten::fake_quantize_per_tensor_affine_cachemask",
        "aten::quantize_per_channel",
        "aten::quantize_per_tensor",
        "aten::round",
        # Complex/spectral paths (A800: _fft_r2c / _fft_c2r).
        "aten::_fft_c2c",
        "aten::_fft_c2r",
        "aten::_fft_r2c",
        "aten::abs",
        "aten::complex",
        "aten::conj_physical",
        "aten::imag",
        "aten::real",
        "aten::stft",
        "aten::view_as_complex",
        "aten::view_as_real",
        # Numerical linear algebra (A800: linalg_cholesky_ex/cholesky_solve).
        "aten::cholesky_solve",
        "aten::linalg_cholesky_ex",
        "aten::linalg_eigh",
        "aten::linalg_inv",
        "aten::linalg_inv_ex",
        "aten::linalg_matrix_norm",
        "aten::linalg_qr",
        "aten::linalg_solve",
        "aten::linalg_solve_ex",
        "aten::linalg_svd",
        "aten::linalg_vector_norm",
        "aten::triangular_solve",
        # Vision geometry and common modern-block glue that is not covered by
        # the dense semantic registry.
        "aten::affine_grid_generator",
        "aten::bitwise_left_shift",
        "aten::bitwise_right_shift",
        "aten::bitwise_and",
        "aten::bitwise_or",
        "aten::div",
        "aten::floor_divide",
        "aten::grid_sampler_2d",
        "aten::cudnn_grid_sampler",
        "aten::pixel_shuffle",
        "aten::pixel_unshuffle",
        "aten::pow",
        "aten::remainder",
        "aten::roll",
        "aten::rsqrt",
        "aten::sin",
        "aten::cos",
        "aten::softmax",
        "aten::stack",
        "aten::sub",
        "aten::where",
    }
)
_SPARSE_ANCHORS = frozenset(
    {
        "aten::_cslt_compress",
        "aten::_sparse_coo_tensor_with_dims_and_tensors",
        "aten::_sparse_addmm",
        "aten::_sparse_mm",
        "aten::_sparse_softmax",
        "aten::_to_sparse_semi_structured",
        "aten::_sparse_semi_structured_addmm",
        "aten::_sparse_semi_structured_mm",
        "aten::sparse_compressed_tensor",
        "aten::sparse_coo_tensor",
        "aten::sparse_sampled_addmm",
        "aten::to_sparse_bsc",
        "aten::to_sparse_bsr",
        "aten::to_sparse_csc",
        "aten::to_sparse_csr",
    }
)
_RAGGED_ANCHORS = frozenset(
    {
        "aten::embedding_bag",
        "aten::_embedding_bag",
        "aten::_embedding_bag_forward_only",
        "aten::index_add",
        "aten::scatter_add",
        "aten::scatter_reduce",
        "aten::segment_reduce",
    }
)
_QUANT_ANCHORS = frozenset(
    {
        "aten::dequantize",
        "aten::fake_quantize_per_channel_affine",
        "aten::fake_quantize_per_tensor_affine",
        "aten::quantize_per_channel",
        "aten::quantize_per_tensor",
        "aten::round",
        "aten::_to_copy",
    }
)
_SPECTRAL_LINALG_ANCHORS = frozenset(
    {
        "aten::_fft_c2c",
        "aten::_fft_c2r",
        "aten::_fft_r2c",
        "aten::cholesky_solve",
        "aten::linalg_cholesky_ex",
        "aten::linalg_eigh",
        "aten::linalg_solve",
        "aten::linalg_solve_ex",
        "aten::stft",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _verify_semantic_source() -> None:
    if not _SEMANTIC_VALIDATOR_PATH.is_file():
        raise RuntimeError(f"semantic validator missing:{_SEMANTIC_VALIDATOR_PATH}")
    observed = _sha256_file(_SEMANTIC_VALIDATOR_PATH)
    if observed != SEMANTIC_VALIDATOR_SOURCE_SHA256:
        raise RuntimeError(
            "semantic validator source binding mismatch:"
            f"expected={SEMANTIC_VALIDATOR_SOURCE_SHA256}:observed={observed}"
        )


_ORIGINAL_TO_DEVICE = _semantic._to_device
_ORIGINAL_NORMALIZE_FORWARD_INPUTS = _semantic._normalize_forward_inputs
_ORIGINAL_SNAPSHOT_OUTPUT = _semantic._snapshot_output


def _require_dense_external_values(value: Any, device: str) -> Any:
    """Reject sparse/quantized external inputs; sparse exists only in forward."""
    result = _ORIGINAL_TO_DEVICE(value, device)
    try:
        import torch
    except ImportError:  # pragma: no cover - real validation requires torch
        return result
    for tensor in _semantic._flatten_tensors(result, torch):
        if tensor.layout != torch.strided or tensor.is_sparse or tensor.is_quantized:
            raise _semantic.UnsupportedCase(f"external_input_not_dense_strided:{tensor.layout}")
    return result


def _normalize_dense_forward_inputs(value: Any, device: str) -> Any:
    """Defend both normalization paths against non-local sparse/quant input."""
    result = _ORIGINAL_NORMALIZE_FORWARD_INPUTS(value, device)
    try:
        import torch
    except ImportError:  # pragma: no cover - real validation requires torch
        return result
    for tensor in _semantic._flatten_tensors(result, torch):
        if tensor.layout != torch.strided or tensor.is_sparse or tensor.is_quantized:
            raise _semantic.UnsupportedCase(f"external_input_not_dense_strided:{tensor.layout}")
    return result


def _snapshot_dense_output(value: Any) -> Any:
    result = _ORIGINAL_SNAPSHOT_OUTPUT(value)
    try:
        import torch
    except ImportError:  # pragma: no cover - real validation requires torch
        return result
    if (
        not isinstance(result, torch.Tensor)
        or result.layout != torch.strided
        or result.is_sparse
        or result.is_quantized
    ):
        raise _semantic.UnsupportedCase(
            f"final_output_not_dense_strided:{getattr(result, 'layout', type(result).__name__)}"
        )
    if result.device.type != "cuda":
        raise _semantic.UnsupportedCase(f"final_output_not_cuda:{result.device}")
    return result


def _configure_semantic_wrapper() -> None:
    """Patch the imported module before either driver or spawned worker runs."""
    _verify_semantic_source()
    _semantic.__file__ = str(Path(__file__).resolve())
    _semantic.generator = generator
    _semantic.CONTRACT_VERSION = CONTRACT_VERSION
    _semantic.RUN_BINDING_VERSION = RUN_BINDING_VERSION
    _semantic.RESULT_MARKER = RESULT_MARKER
    _semantic.MAX_AUTHORIZED_CANDIDATES = MAX_AUTHORIZED_CANDIDATES
    _semantic.MIN_LIVENESS_TRIALS = MIN_LIVENESS_TRIALS
    _semantic.REQUIRED_LIVENESS_SEED = REQUIRED_LIVENESS_SEED
    _semantic._EXPLICIT_VALUE_SCHEMAS = frozenset(_semantic._EXPLICIT_VALUE_SCHEMAS | _FRONTIER_VALUE_SCHEMAS)
    _semantic._METADATA_SCHEMAS = frozenset(
        _semantic._METADATA_SCHEMAS | {"aten::_linalg_check_errors", "aten::is_coalesced"}
    )
    _semantic._to_device = _require_dense_external_values
    _semantic._normalize_forward_inputs = _normalize_dense_forward_inputs
    _semantic._snapshot_output = _snapshot_dense_output


def _identity_schemas(declared_ops: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(identity["schema"]) for declared in declared_ops for identity in declared["runtime_identities"]}


def _require_anchor(family: str, schemas: set[str], anchors: frozenset[str], label: str, index: int) -> None:
    if not schemas & anchors:
        raise ValueError(f"{label}_anchor_missing:{index}:{family}")


def _validate_solver(manifest: Mapping[str, Any], index: int) -> None:
    solver = manifest.get("constraint_solver")
    if not isinstance(solver, Mapping) or set(solver) != {
        "contract_version",
        "backend",
        "template_id",
        "variant",
        "requested_coordinates",
        "realized_coordinates",
        "invariants",
        "invariant_payload_sha256",
    }:
        raise ValueError(f"constraint_solver schema mismatch:{index}")
    if (
        solver["contract_version"] != "frontier_constraint_cells_v1"
        or solver["backend"] != "deterministic_constructive_arithmetic"
    ):
        raise ValueError(f"constraint_solver contract mismatch:{index}")
    if not isinstance(solver["invariants"], list) or not solver["invariants"]:
        raise ValueError(f"constraint_solver invariants missing:{index}")
    if any(not isinstance(item, Mapping) or item.get("passed") is not True for item in solver["invariants"]):
        raise ValueError(f"constraint_solver invariant failure:{index}")
    if _canonical_sha256(solver["invariants"]) != solver["invariant_payload_sha256"]:
        raise ValueError(f"constraint_solver invariant hash mismatch:{index}")


def _tasks(candidates_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    parquet = pq.ParquetFile(candidates_path)
    if parquet.metadata.num_rows != MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"frontier canary must contain exactly 1000 rows:{parquet.metadata.num_rows}")
    if (
        generator.EXACT_CANARY_ROWS != MAX_AUTHORIZED_CANDIDATES
        or sum(generator.FAMILY_QUOTAS.values()) != MAX_AUTHORIZED_CANDIDATES
    ):
        raise ValueError("generator exact-1k quota constants are not bound")
    candidates = pq.read_table(candidates_path).to_pylist()
    manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(candidates) != MAX_AUTHORIZED_CANDIDATES or len(manifests) != MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"candidate/manifest exact count mismatch:{len(candidates)}:{len(manifests)}")
    current_generator_sha = _sha256_file(Path(generator.__file__).resolve())
    families: collections.Counter[str] = collections.Counter()
    templates: collections.Counter[str] = collections.Counter()
    tasks: list[dict[str, Any]] = []
    for index, (row, manifest) in enumerate(zip(candidates, manifests, strict=True)):
        uuid = _nested(row, "extra_info.uuid")
        code = _nested(row, "reward_model.ground_truth")
        entry_point = _nested(row, "extra_info.entry_point")
        declared_ops = manifest.get("declared_ops")
        if manifest.get("candidate_row_index") != index or manifest.get("uuid") != uuid:
            raise ValueError(f"manifest identity mismatch:{index}")
        if (
            manifest.get("manifest_contract_version") != generator.MANIFEST_VERSION
            or manifest.get("generator_contract_version") != generator.CONTRACT_VERSION
            or manifest.get("runtime_contract_version") != generator.RUNTIME_CONTRACT_VERSION
            or manifest.get("primary_intervention") != "frontier_operator_scenario"
        ):
            raise ValueError(f"frontier manifest contract mismatch:{index}")
        if manifest.get("generator_source_sha256") != current_generator_sha:
            raise ValueError(f"generator source binding mismatch:{index}")
        if manifest.get("parent_uuid") is not None or manifest.get("lineage_kind") != "standalone_frontier_synthetic":
            raise ValueError(f"frontier parentless lineage mismatch:{index}")
        if manifest.get("training_approved") is not False or manifest.get("structured_output_deferred") is not True:
            raise ValueError(f"review-only/single-Tensor decision mismatch:{index}")
        if manifest.get("final_output_contract") != {"kind": "single_dense_tensor", "finite_required": True}:
            raise ValueError(f"final output contract mismatch:{index}")
        source = manifest.get("scenario_source")
        if (
            not isinstance(source, Mapping)
            or set(source)
            != {
                "source_url",
                "source_kind",
                "implementation_note",
                "source_registry_version",
            }
            or not all(isinstance(source[key], str) and source[key] for key in source)
        ):
            raise ValueError(f"scenario source provenance mismatch:{index}")
        _validate_solver(manifest, index)
        if not all(isinstance(value, str) and value for value in (uuid, code, entry_point)):
            raise ValueError(f"invalid candidate code identity:{index}")
        if _sha256_bytes(code.encode()) != manifest.get("reference_sha256"):
            raise ValueError(f"reference hash mismatch:{index}")
        if not isinstance(declared_ops, list) or not declared_ops:
            raise ValueError(f"declared ops missing:{index}")
        seen_ids: set[str] = set()
        for declared in declared_ops:
            if not isinstance(declared, Mapping) or set(declared) != {
                "op_id",
                "source_calls",
                "runtime_identities",
                "min_calls_per_trial",
                "must_reach_returned_output",
            }:
                raise ValueError(f"declared op schema mismatch:{index}")
            op_id = declared.get("op_id")
            if not isinstance(op_id, str) or not op_id or op_id in seen_ids:
                raise ValueError(f"declared op id invalid:{index}:{op_id!r}")
            seen_ids.add(op_id)
            if type(declared.get("min_calls_per_trial")) is not int or declared["min_calls_per_trial"] <= 0:
                raise ValueError(f"declared op minimum invalid:{index}:{op_id}")
            if declared.get("must_reach_returned_output") is not True:
                raise ValueError(f"declared op output contract invalid:{index}:{op_id}")
            identities = declared.get("runtime_identities")
            if not isinstance(identities, list) or not identities:
                raise ValueError(f"declared op identities absent:{index}:{op_id}")
            for identity in identities:
                if (
                    not isinstance(identity, Mapping)
                    or set(identity) != {"schema", "overload"}
                    or not isinstance(identity.get("schema"), str)
                    or not identity["schema"].startswith("aten::")
                    or not isinstance(identity.get("overload"), str)
                ):
                    raise ValueError(f"declared ATen identity invalid:{index}:{op_id}:{identity!r}")
        family = manifest.get("primary_family")
        template_id = manifest.get("template_id")
        if (
            family not in generator.FAMILY_QUOTAS
            or not isinstance(template_id, str)
            or template_id not in {item.template_id for item in generator.TEMPLATES}
        ):
            raise ValueError(f"unregistered family/template:{index}")
        schemas = _identity_schemas(declared_ops)
        if family == "sparse_storage_compute":
            _require_anchor(family, schemas, _SPARSE_ANCHORS, "sparse", index)
        elif family == "ragged_graph_segment" or (family == "retrieval_recommender" and template_id == "RR01"):
            if template_id == "RG01":
                if not ({"aten::cumsum", "aten::index_select"} <= schemas):
                    raise ValueError(f"packed_ragged_prefix_offset_anchor_missing:{index}")
            else:
                _require_anchor(family, schemas, _RAGGED_ANCHORS, "ragged", index)
        elif family == "quantization_qdq":
            _require_anchor(family, schemas, _QUANT_ANCHORS, "quantization", index)
            if template_id == "QD02" and not ({"aten::bitwise_left_shift", "aten::bitwise_or"} <= schemas):
                raise ValueError(f"packed_int4_bitwise_shift_anchor_missing:{index}")
        elif family == "spectral_scientific":
            _require_anchor(family, schemas, _SPECTRAL_LINALG_ANCHORS, "spectral_or_linalg", index)
        families[str(family)] += 1
        templates[template_id] += 1
        tasks.append(
            {
                "candidate_row_index": index,
                "uuid": uuid,
                "reference_code": code,
                "entry_point": entry_point,
                "template_id": template_id,
                "primary_family": family,
                "mode_behavior": manifest.get("mode_behavior"),
                "declared_ops": declared_ops,
            }
        )
    if dict(families) != dict(generator.FAMILY_QUOTAS):
        raise ValueError(f"family quota mismatch:{dict(families)}")
    expected_templates = {item.template_id: item.variants for item in generator.TEMPLATES}
    if dict(templates) != expected_templates:
        raise ValueError(f"template quota mismatch:{dict(templates)}")
    return tasks


def _load_allowlist(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("UUID allowlist must be non-empty and unique")
    return set(values)


def _append(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--uuid-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=MIN_LIVENESS_TRIALS)
    parser.add_argument("--seed", type=int, default=REQUIRED_LIVENESS_SEED)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _failure_signature(record: Mapping[str, Any]) -> str | None:
    reason = record.get("reason")
    return (
        _canonical_sha256(
            {
                "status": record.get("status"),
                "reason": reason,
                "error_type": record.get("error_type"),
                "error": record.get("error"),
            }
        )
        if reason
        else None
    )


def _driver_main(args: argparse.Namespace) -> int:
    if args.trials != MIN_LIVENESS_TRIALS or args.seed != REQUIRED_LIVENESS_SEED or args.timeout_seconds <= 0:
        raise ValueError("require trials exactly 3, seed exactly 17, and positive timeout")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    if len(args.launcher_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.launcher_sha256
    ):
        raise ValueError("launcher-sha256 must be 64 lowercase hexadecimal characters")
    tasks = _tasks(args.candidates, args.manifest)
    allowlist = _load_allowlist(args.uuid_file)
    known = {str(task["uuid"]) for task in tasks}
    if missing := allowlist - known:
        raise ValueError(f"allowlisted UUIDs not found:{sorted(missing)[:10]}")
    selected = [task for task in tasks if str(task["uuid"]) in allowlist]
    selected = [task for task in selected if int(task["candidate_row_index"]) % args.shard_count == args.shard_index]
    if not selected:
        raise ValueError("shard selection produced no frontier validation tasks")
    wrapper_sha = _sha256_file(Path(__file__).resolve())
    generator_sha = _sha256_file(Path(generator.__file__).resolve())
    evidence = {
        "contract_version": CONTRACT_VERSION,
        "binding_version": RUN_BINDING_VERSION,
        "validator_source_sha256": wrapper_sha,
        "semantic_validator_source_sha256": SEMANTIC_VALIDATOR_SOURCE_SHA256,
        "semantic_validator_path": str(_SEMANTIC_VALIDATOR_PATH),
        "generator_source_sha256": generator_sha,
        "candidates_sha256": _sha256_file(args.candidates),
        "manifest_sha256": _sha256_file(args.manifest),
        "allowlist_sha256": _sha256_file(args.uuid_file),
        "launcher_source_sha256": args.launcher_sha256,
        "validation_config": {
            "device": args.device,
            "trials": args.trials,
            "seed": args.seed,
            "timeout_seconds": args.timeout_seconds,
            "max_device_memory_gib": _semantic.MAX_DEVICE_MEMORY_GIB,
            "persistent_train_mode_models": True,
            "single_dense_tensor_final_output": True,
            "external_sparse_quant_inputs": "forbidden",
            "tainted_dispatch": "fail_closed_except_reviewed_frontier_registry",
            "execution_controls": {
                "control_trace_comparison": "exact",
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
        },
        "allowlist_path": str(args.uuid_file.resolve()),
    }
    binding = _canonical_sha256(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter[str] = collections.Counter()
    executed = resumed = 0
    with args.output.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns output:{args.output}") from exc
        handle.seek(0)
        prior: dict[str, Mapping[str, Any]] = {}
        selected_by_uuid = {str(task["uuid"]): task for task in selected}
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            uuid = str(record.get("uuid"))
            if (
                uuid not in selected_by_uuid
                or uuid in prior
                or record.get("contract_binding_sha256") != binding
                or record.get("validator_sha256") != wrapper_sha
                or record.get("semantic_validator_source_sha256") != SEMANTIC_VALIDATOR_SOURCE_SHA256
                or record.get("launcher_sha256") != args.launcher_sha256
                or type(record.get("passed")) is not bool
            ):
                raise ValueError(f"invalid resume record:{args.output}:{line_number}")
            prior[uuid] = record
        handle.seek(0, os.SEEK_END)
        manifest_sha = evidence["manifest_sha256"]
        for task in selected:
            uuid = str(task["uuid"])
            if uuid in prior:
                resumed += 1
                counts["passed" if prior[uuid]["passed"] else "failed"] += 1
                continue
            payload = {
                **task,
                "device": args.device,
                "trials": args.trials,
                "seed": args.seed,
                "max_device_memory_gib": _semantic.MAX_DEVICE_MEMORY_GIB,
            }
            result = _semantic._run_subprocess(payload, args.timeout_seconds)
            raw_status = str(result.get("status", "failed"))
            if raw_status == "worker_protocol_error":
                result["status"] = "protocol_error"
            elif raw_status not in {
                "passed",
                "reference_failed",
                "unsupported",
                "failed",
                "timeout",
                "protocol_error",
            }:
                result["status"] = "failed"
            passed = result.get("passed") is True
            record = {
                **result,
                "status": result.get("status", "failed"),
                "passed": passed,
                "reason": result.get("reason"),
                "source_sha256": evidence["candidates_sha256"],
                "reference_sha256": _sha256_bytes(str(task["reference_code"]).encode()),
                "row_index": int(task["candidate_row_index"]),
                "family": str(task["primary_family"]),
                "manifest_sha256": manifest_sha,
                "validator_sha256": wrapper_sha,
                "semantic_validator_source_sha256": SEMANTIC_VALIDATOR_SOURCE_SHA256,
                "launcher_sha256": args.launcher_sha256,
                "contract_binding_sha256": binding,
                # Keep the semantic-liveness spelling as a compatibility
                # alias for generic analyzers; both names bind the same
                # immutable closure below.
                "validation_binding_sha256": binding,
                "failure_stage": "liveness" if not passed else None,
                "failure_signature": None,
                "raw_status": raw_status,
                "evidence": evidence,
                "binding_evidence": evidence,
                "candidates_sha256": evidence["candidates_sha256"],
                "generator_source_sha256": generator_sha,
            }
            record["failure_signature"] = _failure_signature(record)
            _append(handle, record)
            executed += 1
            counts["passed" if passed else "failed"] += 1
    summary = {
        "contract_version": CONTRACT_VERSION,
        "contract_binding_sha256": binding,
        "launcher_source_sha256": args.launcher_sha256,
        "validator_source_sha256": wrapper_sha,
        "semantic_validator_source_sha256": SEMANTIC_VALIDATOR_SOURCE_SHA256,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected": len(selected),
        "executed": executed,
        "resumed": resumed,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _configure_semantic_wrapper()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv == ["--_worker"]:
        with contextlib.redirect_stdout(sys.stderr):
            result = _semantic._worker(json.loads(sys.stdin.read()))
        print(RESULT_MARKER + json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    return _driver_main(_parser().parse_args(raw_argv))


if __name__ == "__main__":
    raise SystemExit(main())
