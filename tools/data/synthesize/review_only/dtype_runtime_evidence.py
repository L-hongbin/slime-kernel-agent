"""Pure checks for serialized CSP-DAG dtype runtime evidence."""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

EXPANSION_CONTRACT = "csp_dag_shape_dtype_layout_expansion_v1"
REQUIRED_TRIALS = 3
CONTRACTS = {
    "parameter_free": ("dtype_parameter_free_runtime_liveness_v6", "dtype_parameter_free_runtime_binding_v6"),
    "module_state": ("dtype_module_state_runtime_liveness_v5", "dtype_module_state_runtime_binding_v5"),
}
RESULT_BINDING_VERSION = "dtype_runtime_result_payload_binding_v1"
FLOAT_COMPARATOR_CONTRACT = "dtype_cast_equivalent_torch_allclose_v1"
EXACT_COMPARATOR_CONTRACT = "dtype_nonfloating_torch_equal_v1"
NONFLOATING_DTYPES = frozenset(
    {
        "torch.bool",
        "torch.uint8",
        "torch.uint16",
        "torch.uint32",
        "torch.uint64",
        "torch.int8",
        "torch.int16",
        "torch.int32",
        "torch.int64",
    }
)
BINDING_FIELDS = (
    "contract_version",
    "binding_version",
    "coherence_class",
    "validator_source_sha256",
    "launcher_source_sha256",
    "runtime_dependencies",
    "parents_sha256",
    "children_sha256",
    "manifest_sha256",
    "allowlist_sha256",
    "validation_config",
    "execution_context",
)
# ``unsupported`` is a completed runtime result with the same provenance and
# resource evidence as a pass, but it has never executed a successful dtype
# proof.  Keep its top-level schema deliberately closed: accepting arbitrary
# fields here would allow a caller to smuggle passed-only proof material (or
# an unrelated semantic claim) into a record after recomputing its payload
# digest.
UNSUPPORTED_REASON_PREFIX = "UnsupportedCase:"
UNSUPPORTED_ALLOWED_FIELDS = frozenset(
    {
        "allowlist_sha256",
        "assigned_dtype",
        "binding_version",
        "candidate_row_index",
        "children_sha256",
        "child_uuid",
        "coherence_class",
        "contract_version",
        "duration_seconds",
        "execution_context",
        "launcher_source_sha256",
        "manifest_sha256",
        "memory_guard",
        "parents_sha256",
        "parent_uuid",
        "passed",
        "reason",
        "result_binding_version",
        "result_payload_sha256",
        "runtime_dependencies",
        "status",
        "transformed_factory_count",
        "validation_binding_sha256",
        "validation_config",
        "validator_source_sha256",
        # The validator may emit this diagnostic after a successful worker
        # protocol exchange.  It is the sole optional runtime diagnostic.
        "worker_stderr_tail",
    }
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def payload_sha256(value: Mapping[str, Any], digest_field: str) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != digest_field})


def unsupported_record_errors(record: Mapping[str, Any]) -> list[str]:
    """Return exact-schema/status errors for one dtype unsupported result."""
    errors: list[str] = []
    extra = sorted(set(record) - UNSUPPORTED_ALLOWED_FIELDS)
    if extra:
        errors.append("unsupported record contains unexpected top-level fields:" + ",".join(extra))
    reason = record.get("reason")
    if (
        not isinstance(reason, str)
        or not reason.startswith(UNSUPPORTED_REASON_PREFIX)
        or not reason.removeprefix(UNSUPPORTED_REASON_PREFIX).strip()
    ):
        errors.append("unsupported record reason must have nonempty UnsupportedCase detail")
    if "worker_stderr_tail" in record and (
        not isinstance(record["worker_stderr_tail"], str) or not record["worker_stderr_tail"].strip()
    ):
        errors.append("unsupported worker_stderr_tail must be a nonempty string")
    return errors


def ast_sha256(code: str) -> str:
    return hashlib.sha256(
        ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False).encode()
    ).hexdigest()


def add(findings: list[dict[str, str]], scope: str, message: str, severity: str = "P1") -> None:
    if len(findings) < 200:
        findings.append({"severity": severity, "scope": scope, "message": message})


def row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    extra, reward = row.get("extra_info"), row.get("reward_model")
    uuid = extra.get("uuid") if isinstance(extra, Mapping) else None
    code = reward.get("ground_truth") if isinstance(reward, Mapping) else None
    if not isinstance(uuid, str) or not uuid or not isinstance(code, str) or not code:
        raise ValueError("row identity malformed")
    return uuid, code


def manifest_errors(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    manifest: Mapping[str, Any],
    index: int,
    findings: list[dict[str, str]],
) -> tuple[str, str, str, int]:
    scope = f"candidate:{index}"
    parent_uuid, parent_code = row_identity(parent)
    child_uuid, child_code = row_identity(child)
    coherence = manifest.get("coherence_class")
    if (
        manifest.get("manifest_contract_version") != "dtype_lane_manifest_v1"
        or manifest.get("materialization_status") != "review_only"
        or manifest.get("training_approved") is not False
    ):
        add(findings, scope, "manifest contract/governance mismatch")
    if (
        coherence not in CONTRACTS
        or manifest.get("candidate_row_index") != index
        or manifest.get("parent_uuid") != parent_uuid
        or manifest.get("child_uuid") != child_uuid
    ):
        add(findings, scope, "manifest coherence/index/parent/child mismatch")
    target = manifest.get("assigned_target")
    if target not in {"float16", "bfloat16"} or manifest.get("primary_intervention") != "dtype":
        add(findings, scope, "manifest target/intervention invalid")
    if (
        manifest.get("parent_reference_sha256") != hashlib.sha256(parent_code.encode()).hexdigest()
        or manifest.get("child_reference_sha256") != hashlib.sha256(child_code.encode()).hexdigest()
        or manifest.get("parent_normalized_ast_sha256") != ast_sha256(parent_code)
        or manifest.get("child_normalized_ast_sha256") != ast_sha256(child_code)
    ):
        add(findings, scope, "manifest source/reference/AST SHA mismatch")
    count = manifest.get("transformed_factory_count")
    if not isinstance(count, int) or count < 1 or manifest.get("factory_count") != count:
        add(findings, scope, "manifest transformed factory count invalid")
    realized = manifest.get("realized_intervention")
    expected_realized = {
        "primary_intervention": "dtype",
        "coherence_class": coherence,
        "dtype_before": "float32",
        "dtype_after": target,
        "factory_count": count,
        "model_parameter_policy": "registered_fp32_parent_exact_cast",
        "model_buffer_policy": "registered_state_exact_cast_or_equal",
        "explicit_cast_policy": "input storage cast before view; one base Module conversion",
        "logical_forward_dtype_policy": "all_floating_vN_and_final_outputs_match_assigned_dtype_v1",
        "internal_aten_non_target_floating_outputs": "diagnostic_only",
        "internal_aten_complex_outputs": "reject",
    }
    if realized != expected_realized:
        add(findings, scope, "manifest realized_intervention is not the exact dtype intervention")
    shape = manifest.get("logical_shape")
    shape_numel = (
        math.prod(shape)
        if isinstance(shape, list) and all(type(value) is int and value > 0 for value in shape)
        else None
    )
    expected_before = shape_numel * 4 * count if shape_numel is not None and isinstance(count, int) else None
    expected_after = shape_numel * 2 * count if shape_numel is not None and isinstance(count, int) else None
    if (
        shape_numel is None
        or manifest.get("input_bytes_before") != expected_before
        or manifest.get("input_bytes_after") != expected_after
        or manifest.get("estimated_peak_bytes") != expected_after
        or manifest.get("static_memory_budget_bytes") != 64 * 1024**3
        or manifest.get("runtime_memory_budget_bytes") != 64 * 1024**3
    ):
        add(findings, scope, "manifest dtype shape/byte/budget proof mismatch")
    semantic_gate = manifest.get("semantic_gate")
    if (
        not isinstance(semantic_gate, Mapping)
        or semantic_gate.get("status") != "passed"
        or semantic_gate.get("reasons") != []
    ):
        add(findings, scope, "manifest semantic gate is not an eligible passed verdict")
    intervention_sha = canonical_sha256(realized) if isinstance(realized, Mapping) else None
    expected_augmentation = {
        "contract_version": EXPANSION_CONTRACT,
        "generator_version": EXPANSION_CONTRACT,
        "parent_uuid": parent_uuid,
        "child_uuid": child_uuid,
        "source_artifact_sha256": manifest.get("source_artifact_sha256"),
        "source_row_index": manifest.get("source_row_index"),
        "parent_reference_sha256": manifest.get("parent_reference_sha256"),
        "parent_normalized_ast_sha256": manifest.get("parent_normalized_ast_sha256"),
        "child_reference_sha256": manifest.get("child_reference_sha256"),
        "child_normalized_ast_sha256": manifest.get("child_normalized_ast_sha256"),
        "intervention_id": f"dtype_{str(intervention_sha)[:20]}",
        "intervention_sha256": intervention_sha,
        "intervention_kind": "dtype",
        "coverage_cell": f"dtype_{coherence}_{target}",
        "shape_scale": None,
        "value_family_before": None,
        "value_family_after": None,
        "dtype_before": "float32",
        "dtype_after": target,
        "layout_before": None,
        "layout_after": None,
        "shape_dimension_name": None,
        "shape_dimension_before": None,
        "shape_dimension_after": None,
        "factory_count": count,
        "input_bytes_before": expected_before,
        "input_bytes_after": expected_after,
        "estimated_peak_bytes": expected_after,
        "memory_budget_bytes": 64 * 1024**3,
        "working_set_multiplier": 1.0,
        "memory_estimator_version": "csp_dag_exact_input_bytes_v1",
        "shape_proof": "shape inherited from bound CSP-DAG shape child",
        "compatibility_proof": "CSP-DAG controlled lowering plus paired target-GPU validation",
        "validation_status": "dtype_paired_h20_pending",
    }
    augmentation = (
        child.get("extra_info", {}).get("augmentation") if isinstance(child.get("extra_info"), Mapping) else None
    )
    if augmentation != expected_augmentation:
        add(findings, scope, "child augmentation is not the exact manifest-bound dtype intervention")
    return parent_uuid, child_uuid, str(coherence), int(count or 0)


def _histogram(value: Any, *, count: Any, allowed: frozenset[str]) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(count, int)
        and count >= 0
        and set(value).issubset(allowed)
        and all(type(item) is int and item > 0 for item in value.values())
        and sum(value.values()) == count
    )


def trace_errors(trace: Any, target: str, nodes: int, scope: str, name: str, findings: list[dict[str, str]]) -> None:
    if not isinstance(trace, Mapping):
        add(findings, scope, f"{name} trace missing")
        return
    logical, final = trace.get("logical_forward"), trace.get("final_output")
    if (
        not isinstance(logical, Mapping)
        or logical.get("target_dtype") != f"torch.{target}"
        or logical.get("forward_return_count") != 1
        or logical.get("logical_value_count") != nodes
        or logical.get("logical_value_indices_sha256") != canonical_sha256(list(range(nodes)))
        or logical.get("logical_value_names_sha256") != canonical_sha256([f"v{i}" for i in range(nodes)])
        or not isinstance(logical.get("floating_logical_value_count"), int)
        or not isinstance(logical.get("nonfloating_logical_value_count"), int)
        or logical.get("floating_logical_value_count", 0) <= 0
        or logical["floating_logical_value_count"] + logical["nonfloating_logical_value_count"] != nodes
        or logical.get("floating_logical_dtypes") != {f"torch.{target}": logical.get("floating_logical_value_count")}
        or not _histogram(
            logical.get("nonfloating_logical_dtypes"),
            count=logical.get("nonfloating_logical_value_count"),
            allowed=NONFLOATING_DTYPES,
        )
    ):
        add(findings, scope, f"{name} logical-vN dtype proof invalid")
    if (
        not isinstance(final, Mapping)
        or final.get("target_dtype") != f"torch.{target}"
        or final.get("floating_final_output_count", 0) <= 0
        or final.get("floating_final_output_dtypes") != {f"torch.{target}": final.get("floating_final_output_count")}
        or not _histogram(
            final.get("nonfloating_final_output_dtypes"),
            count=final.get("nonfloating_final_output_count"),
            allowed=NONFLOATING_DTYPES,
        )
    ):
        add(findings, scope, f"{name} final output dtype proof invalid")
    if (
        trace.get("complex_output_calls") != 0
        or not isinstance(trace.get("internal_non_target_floating_output_calls"), int)
        or trace.get("internal_non_target_floating_output_calls", -1) < 0
        or not isinstance(trace.get("target_consuming_calls"), int)
        or trace.get("target_consuming_calls", 0) <= 0
        or not isinstance(trace.get("total_dispatch_calls"), int)
        or trace.get("total_dispatch_calls", 0) < trace.get("target_consuming_calls", 0)
    ):
        add(findings, scope, f"{name} dispatcher evidence invalid")


def memory_guard_errors(value: Any, scope: str, findings: list[dict[str, str]]) -> None:
    limit = 64 * 1024**3
    if not isinstance(value, Mapping):
        add(findings, scope, "CUDA memory guard evidence missing")
        return
    integer_fields = (
        "limit_bytes",
        "total_device_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "current_allocated_bytes",
        "current_reserved_bytes",
    )
    if (
        value.get("enabled") is not True
        or value.get("configured") is not True
        or value.get("synchronized") is not True
        or value.get("max_device_memory_gib") != 64.0
        or any(type(value.get(key)) is not int or value[key] < 0 for key in integer_fields)
        or value.get("limit_bytes") != limit
        or value.get("total_device_bytes", 0) < limit
        or not isinstance(value.get("allocator_fraction"), (int, float))
        or isinstance(value.get("allocator_fraction"), bool)
        or not math.isfinite(float(value["allocator_fraction"]))
        or not math.isclose(
            value["allocator_fraction"], limit / value["total_device_bytes"], rel_tol=0.0, abs_tol=1e-12
        )
        or value.get("peak_allocated_bytes", limit + 1) > limit
        or value.get("peak_reserved_bytes", limit + 1) > limit
        or value.get("over_limit") is not False
        or value.get("within_limit") is not True
        or "telemetry_errors" in value
        or "synchronize_error" in value
    ):
        add(findings, scope, "CUDA memory guard evidence invalid")


def trial_errors(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    coherence: str,
    factories: int,
    findings: list[dict[str, str]],
    scope: str,
) -> None:
    trials = record.get("trials")
    if not isinstance(trials, list) or len(trials) != REQUIRED_TRIALS:
        add(findings, scope, "not exactly three dtype trials")
        return
    nodes = manifest.get("source_graph_node_count")
    if not isinstance(nodes, int) or nodes < 1:
        add(findings, scope, "manifest source_graph_node_count invalid")
        return
    target = str(manifest.get("assigned_target"))
    expected_tolerance = {"float16": 0.01, "bfloat16": 0.02}.get(target)
    tolerance = record.get("tolerance")
    if (
        expected_tolerance is None
        or not isinstance(tolerance, Mapping)
        or any(
            not isinstance(tolerance.get(key), (int, float))
            or isinstance(tolerance.get(key), bool)
            or not math.isfinite(float(tolerance[key]))
            or tolerance.get(key) != expected_tolerance
            for key in ("rtol", "atol")
        )
    ):
        add(findings, scope, "record dtype tolerance does not match formal target policy")
    for ordinal, trial in enumerate(trials):
        trial_scope = f"{scope}:trial:{ordinal}"
        if (
            not isinstance(trial, Mapping)
            or trial.get("trial") != ordinal
            or trial.get("seed") != 17 + 10_007 * ordinal
        ):
            add(findings, trial_scope, "trial ordinal/seed mismatch")
            continue
        if trial.get("trial_payload_sha256") != payload_sha256(trial, "trial_payload_sha256"):
            add(findings, trial_scope, "trial payload SHA mismatch")
        inputs = trial.get("inputs")
        if (
            trial.get("changed_dtype_tensors") != factories
            or not isinstance(inputs, list)
            or len(inputs) != factories
            or len({item.get("path") for item in inputs if isinstance(item, Mapping)}) != factories
            or any(
                not isinstance(item, Mapping)
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or not isinstance(item.get("shape"), list)
                or not all(type(value) is int and value >= 0 for value in item["shape"])
                or item.get("numel") != math.prod(item["shape"])
                or item.get("parent_dtype") != "torch.float32"
                or item.get("child_dtype") != f"torch.{target}"
                or item.get("factory_values_equal_to_parent_cast") is not True
                or item.get("maximum_factory_cast_difference") != 0.0
                or item.get("numel", 0) < 1
                or not isinstance(item.get("stride"), list)
                or len(item["stride"]) != len(item["shape"])
                or not all(type(value) is int and value >= 0 for value in item["stride"])
                or type(item.get("storage_offset")) is not int
                or item["storage_offset"] < 0
                or type(item.get("is_pinned")) is not bool
                or not isinstance(item.get("minimum"), (int, float))
                or isinstance(item.get("minimum"), bool)
                or not math.isfinite(float(item["minimum"]))
                or not isinstance(item.get("maximum"), (int, float))
                or isinstance(item.get("maximum"), bool)
                or not math.isfinite(float(item["maximum"]))
                or item["minimum"] > item["maximum"]
                for item in inputs
            )
        ):
            add(findings, trial_scope, "input exact-parent-cast proof invalid")
        state = trial.get("model_state")
        inventory_fields = (
            "parameter_entries",
            "buffer_entries",
            "floating_entries",
            "nonfloating_entries",
            "state_bytes_parent",
            "state_bytes_child",
            "object_alias_groups",
            "storage_alias_groups",
        )
        inventory_valid = (
            isinstance(state, Mapping)
            and all(type(state.get(key)) is int and state[key] >= 0 for key in inventory_fields)
            and state.get("parameter_entries", 0) + state.get("buffer_entries", 0)
            >= state.get("floating_entries", 0) + state.get("nonfloating_entries", 0)
            and state.get("object_alias_groups", 0)
            <= state.get("parameter_entries", 0) + state.get("buffer_entries", 0)
            and state.get("storage_alias_groups", 0)
            <= state.get("parameter_entries", 0) + state.get("buffer_entries", 0)
            and state.get("state_bytes_child", 0) <= state.get("state_bytes_parent", 0)
            and state.get("state_bytes_parent", 0) + state.get("state_bytes_child", 0) <= 64 * 1024**3
            and isinstance(state.get("state_schema_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", state["state_schema_sha256"]) is not None
        )
        shared_state = (
            inventory_valid
            and state.get("coherence_class") == coherence
            and state.get("construction_rng_equal") is True
            and state.get("values_equal_to_parent_cast") is True
            and state.get("nonfloat_values_equal") is True
            and state.get("unregistered_tensor_count") == 0
        )
        if not shared_state:
            add(findings, trial_scope, "model-state coherence/cast/RNG proof invalid")
        elif coherence == "parameter_free" and any(
            state.get(key) != 0
            for key in (
                "parameter_entries",
                "buffer_entries",
                "floating_entries",
                "nonfloating_entries",
                "state_bytes_parent",
                "state_bytes_child",
                "object_alias_groups",
                "storage_alias_groups",
            )
        ):
            add(findings, trial_scope, "parameter-free model-state proof invalid")
        elif coherence == "module_state" and (
            state.get("floating_entries", 0) < 1
            or state.get("state_bytes_parent", 0) <= 0
            or state.get("state_bytes_child", 0) <= 0
        ):
            add(findings, trial_scope, "module-state inventory proof invalid")
        cast = trial.get("cast_equivalent_outputs")
        cast_invalid = not isinstance(cast, list) or not cast
        output_histogram: collections.Counter[str] = collections.Counter()
        if isinstance(cast, list):
            if len({item.get("path") for item in cast if isinstance(item, Mapping)}) != len(cast):
                cast_invalid = True
            for item in cast:
                if not isinstance(item, Mapping):
                    cast_invalid = True
                    continue
                parent_dtype, child_dtype = item.get("parent_dtype"), item.get("child_dtype")
                floating = parent_dtype == "torch.float32" and child_dtype == f"torch.{target}"
                nonfloating = parent_dtype == child_dtype and parent_dtype in NONFLOATING_DTYPES
                shape = item.get("shape")
                maximum, mean = item.get("maximum_absolute_difference"), item.get("mean_absolute_difference")
                common_invalid = (
                    not isinstance(item.get("path"), str)
                    or not item["path"]
                    or not isinstance(shape, list)
                    or not all(type(value) is int and value >= 0 for value in shape)
                    or item.get("elements") != math.prod(shape)
                    or item.get("elements", 0) < 1
                    or not isinstance(maximum, (int, float))
                    or isinstance(maximum, bool)
                    or not math.isfinite(float(maximum))
                    or maximum < 0
                    or not isinstance(mean, (int, float))
                    or isinstance(mean, bool)
                    or not math.isfinite(float(mean))
                    or mean < 0
                    or mean > maximum
                    or item.get("comparison_payload_sha256") != payload_sha256(item, "comparison_payload_sha256")
                )
                if common_invalid or not (floating or nonfloating):
                    cast_invalid = True
                elif floating:
                    maximum_ratio = item.get("maximum_tolerance_ratio")
                    maximum_excess = item.get("maximum_tolerance_excess")
                    maximum_reference = item.get("maximum_reference_absolute_value")
                    maximum_allowed = item.get("maximum_allowed_absolute_difference")
                    expected_comparator = {
                        "contract": FLOAT_COMPARATOR_CONTRACT,
                        "implementation": "torch.allclose",
                        "rtol": expected_tolerance,
                        "atol": expected_tolerance,
                        "equal_nan": False,
                        "relative_reference_operand": "child",
                        "elementwise_bound": "abs(parent-child) <= atol + rtol * abs(child)",
                    }
                    if (
                        item.get("rtol") != expected_tolerance
                        or item.get("atol") != expected_tolerance
                        or item.get("comparator") != expected_comparator
                        or item.get("within_tolerance") is not True
                        or item.get("allclose_passed") is not True
                        or item.get("violating_elements") != 0
                        or not isinstance(maximum_ratio, (int, float))
                        or isinstance(maximum_ratio, bool)
                        or not math.isfinite(float(maximum_ratio))
                        or not 0.0 <= maximum_ratio <= 1.0
                        or not isinstance(maximum_excess, (int, float))
                        or isinstance(maximum_excess, bool)
                        or not math.isfinite(float(maximum_excess))
                        or maximum_excess > 0.0
                        or not isinstance(maximum_reference, (int, float))
                        or isinstance(maximum_reference, bool)
                        or not math.isfinite(float(maximum_reference))
                        or maximum_reference < 0.0
                        or not math.isclose(
                            float(maximum_allowed),
                            expected_tolerance + expected_tolerance * maximum_reference,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        or maximum > maximum_allowed
                    ):
                        cast_invalid = True
                elif nonfloating:
                    expected_comparator = {
                        "contract": EXACT_COMPARATOR_CONTRACT,
                        "implementation": "torch.equal",
                        "rtol": 0.0,
                        "atol": 0.0,
                        "equal_nan": False,
                        "relative_reference_operand": None,
                        "elementwise_bound": "exact equality",
                    }
                    if (
                        maximum != 0.0
                        or mean != 0.0
                        or item.get("maximum_tolerance_ratio") != 0.0
                        or item.get("maximum_tolerance_excess") != 0.0
                        or item.get("maximum_reference_absolute_value") != 0.0
                        or item.get("maximum_allowed_absolute_difference") != 0.0
                        or item.get("violating_elements") != 0
                        or item.get("within_tolerance") is not True
                        or item.get("allclose_passed") is not True
                        or item.get("rtol") != 0.0
                        or item.get("atol") != 0.0
                        or item.get("comparator") != expected_comparator
                    ):
                        cast_invalid = True
                if isinstance(child_dtype, str):
                    output_histogram[child_dtype] += 1
        if cast_invalid:
            add(findings, trial_scope, "cast-equivalent output proof invalid")
        semantic, realized = trial.get("semantic_dispatch"), trial.get("realized_dispatch")
        trace_errors(semantic, target, nodes, trial_scope, "semantic", findings)
        trace_errors(realized, target, nodes, trial_scope, "realized", findings)
        expected_output_histogram = dict(sorted(output_histogram.items()))
        for name, trace in (("semantic", semantic), ("realized", realized)):
            final = trace.get("final_output") if isinstance(trace, Mapping) else None
            declared = {}
            if isinstance(final, Mapping):
                declared.update(final.get("floating_final_output_dtypes", {}))
                declared.update(final.get("nonfloating_final_output_dtypes", {}))
            if declared != expected_output_histogram:
                add(findings, trial_scope, f"{name} final output histogram does not bind cast-equivalent leaves")
