import ast
import logging
import random
import re
from typing import Any

from slime.utils.types import Sample

try:
    pass
except ImportError:
    pass

logger = logging.getLogger(__name__)


VALIDATION_ERROR = "VALIDATION_ERROR"
SYNTAX_ERROR = "SYNTAX_ERROR"
IMPORT_ERROR = "IMPORT_ERROR"
PRECHECK_ERROR = "PRECHECK_ERROR"
COMPILATION_ERROR = "COMPILATION_ERROR"
DECOY_KERNEL_DETECTED = "DECOY_KERNEL_DETECTED"
KERNEL_EVAL_FAILED = "KERNEL_EVAL_FAILED"
CORRECTNESS_ERROR = "CORRECTNESS_ERROR"
KERNEL_EVAL_TIMEOUT = "KERNEL_EVAL_TIMEOUT"

CUDA_SECTIONS = ("CUDA_KERNELS", "APPLY_BINDINGS", "MODEL_NEW")

METADATA_POP_KEYS = (
    "compile_only",
    "device",
    "required_resource",
    "task_id",
    "inline_gpu_execute_completed",
    "inline_compile_worker_id",
    "inline_compile_worker_device",
    "compile_timing",
    "build_backend",
    "compilation_error",
    "compile_artifact_cache_enabled",
    "compile_artifact_cache_hit",
    "correctness_early_stop_enabled",
    "correctness_inplace_compare_enabled",
    "correctness_reference_cache_poison_enabled",
    "correctness_reference_alias_clone_trials",
    "correctness_tolerance_source",
    "correctness_current_trial",
    "correctness_current_substage",
    "correctness_early_stopped",
    "correctness_issue_name",
    "kernel_task_id",
    "coverage_backend",
    "reference_task_id",
    "split_compile_and_execute",
    "cpu_worker_run_s",
    "runtime_error",
)

COMPILE_ARTIFACT_POP_KEYS = (
    "precheck",
    "compiled",
    "compile_mode",
    "source_mode",
    "backend",
    "compile_timing",
    "source_files",
    "compile_artifact_cache_enabled",
    "compile_artifact_cache_hit",
    "compile_artifact_cache_dir",
    "compile_cache_hit",
    "module_name",
    "profiling_hints",
    "artifact_node_id",
    "artifact_hostname",
    "target_gpu_worker_id",
    "target_gpu_selection_strategy",
    "device",
    "precompiled_artifact_used",
)

NCU_POP_KEYS = (
    "profile_version",
    "tool_version",
    "requested_metrics",
    "csv_source",
    "wall_time_s",
)

NCU_METRIC_SHORT_FIELDS = {
    "gpu__time_duration.sum": "duration",
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed": "dram_cycles_active",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput",
    "l1tex__throughput.avg.pct_of_peak_sustained_active": "l1_throughput",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "compute_memory_throughput",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput",
    "sm__issue_active.avg.pct_of_peak_sustained_elapsed": "sm_issue_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "sm_warps_active",
    "launch__occupancy_per_block_size": "occupancy_per_block_size",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block": "shared_mem_per_block",
}


def _format_compilation_error_message(env_state: dict[str, Any]) -> str:
    metadata = env_state.get("metadata") if isinstance(env_state.get("metadata"), dict) else {}
    compile_artifact = metadata.get("compile_artifact")

    compile_error = None
    if isinstance(compile_artifact, dict):
        compile_error = compile_artifact.pop("error", None)
        if "compilation_error" in compile_artifact:
            compile_error = compile_artifact.pop("compilation_error", None)
    if compile_error:
        return f"Compilation failed. Compiler output:\n{compile_error}"
    return str(env_state.get("error_message") or "Compilation failed.")


def _extract_env_precheck_error_message(env_state: dict[str, Any]) -> str | None:
    metadata = env_state.get("metadata") if isinstance(env_state.get("metadata"), dict) else {}
    candidates = (
        metadata.get("compilation_error"),
        env_state.get("error_message"),
        env_state.get("error"),
    )
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        marker = "Precheck failed:"
        marker_index = candidate.find(marker)
        if marker_index >= 0:
            detail = candidate[marker_index + len(marker) :]
            return f"Code precheck failed:{detail}"
    return None


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _round_numeric(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(float(value), digits)


def _format_ncu_metric_value(value: Any, unit: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        formatted_value = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        formatted_value = str(value)
    return " ".join(part for part in (formatted_value, str(unit or "").strip()) if part)


def _normalize_ncu_metadata(ncu_metadata: Any) -> Any:
    if not isinstance(ncu_metadata, dict):
        return ncu_metadata

    normalized_ncu = dict(ncu_metadata)
    for key in NCU_POP_KEYS:
        normalized_ncu.pop(key, None)

    kernels = normalized_ncu.get("kernels")
    if not isinstance(kernels, list):
        return normalized_ncu

    normalized_kernels = []
    for kernel in kernels:
        if not isinstance(kernel, dict):
            normalized_kernels.append(kernel)
            continue
        normalized_kernel = dict(kernel)
        metrics = normalized_kernel.pop("metrics", None)
        if isinstance(metrics, dict):
            for metric_name, metric_payload in metrics.items():
                field = NCU_METRIC_SHORT_FIELDS.get(metric_name, metric_name)
                if isinstance(metric_payload, dict):
                    value = metric_payload.get("value")
                    unit = metric_payload.get("unit")
                else:
                    value = metric_payload
                    unit = None
                normalized_kernel[field] = _format_ncu_metric_value(value, unit)
        normalized_kernels.append(normalized_kernel)
    normalized_ncu["kernels"] = normalized_kernels
    return normalized_ncu


def _compact_aten_ops(operators: Any) -> Any:
    if not isinstance(operators, list):
        return operators

    compacted = []
    for operator in operators:
        if isinstance(operator, dict):
            operator = {
                key: value
                for key, value in operator.items()
                if key not in ("cpu_time_us", "allowed", "normalized_name")
            }
        compacted.append(operator)
    return compacted


def _get_env_metadata(env_state: dict[str, Any]) -> dict[str, Any]:
    metadata = env_state.get("metadata") if isinstance(env_state, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _coefficient_of_variation(metadata: dict[str, Any], mean_key: str, std_key: str) -> float | None:
    mean = _as_float_or_none(metadata.get(mean_key))
    std = _as_float_or_none(metadata.get(std_key))
    if mean is None or std is None or mean == 0.0:
        return None
    return std / mean


def _extract_detail_env_time(env_state: dict[str, Any]) -> dict[str, float]:
    metadata = _get_env_metadata(env_state)

    detail_env_time: dict[str, float] = {}
    compile_time = _as_float_or_none(metadata.get("kg_kernel_backend_compile_s"))
    if compile_time is not None:
        detail_env_time["compile_time"] = compile_time

    warmup_time = _as_float_or_none(metadata.get("kg_kernel_perf_warmup_s"))
    measure_time = _as_float_or_none(metadata.get("kg_kernel_perf_measure_wall_s"))
    if warmup_time is not None or measure_time is not None:
        detail_env_time["kernel_runtime"] = (warmup_time or 0.0) + (measure_time or 0.0)

    profile_time = _as_float_or_none(metadata.get("kg_kernel_perf_profile_s"))
    if profile_time is not None:
        detail_env_time["profile_time"] = profile_time

    ncu_metadata = metadata.get("ncu")
    ncu_wall_time = _as_float_or_none(ncu_metadata.get("wall_time_s")) if isinstance(ncu_metadata, dict) else None
    if ncu_wall_time is not None and ncu_wall_time > 0.0:
        detail_env_time["ncu_profile_time_s"] = round(ncu_wall_time, 4)

    reference_warmup_time = _as_float_or_none(metadata.get("kg_reference_perf_warmup_s"))
    reference_measure_time = _as_float_or_none(metadata.get("kg_reference_perf_measure_wall_s"))
    if reference_warmup_time is not None or reference_measure_time is not None:
        detail_env_time["refer_runtime"] = (reference_warmup_time or 0.0) + (reference_measure_time or 0.0)

    return detail_env_time


def _require_env_value(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise KeyError(f"{path} missing required field for env_extra_info: {key}")
    return mapping[key]


def _extract_env_extra_info(env_state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(env_state, dict):
        raise TypeError("env_state must be a dict for env_extra_info extraction.")

    metadata = _require_env_value(env_state, "metadata", "env_state")
    if not isinstance(metadata, dict):
        raise TypeError("env_state.metadata must be a dict for env_extra_info extraction.")

    num_custom_kernels = metadata.get("num_custom_kernels", 0)
    num_total_kernels = metadata.get("num_total_kernels", 0)
    custom_kernel_time = metadata.get(
        "custom_kernel_cuda_time_in_profiling_us",
        metadata.get("custom_kernel_time_in_profiling_us", 0),
    )
    total_kernel_time = metadata.get("total_kernel_run_time_in_profiling_us", 0)
    num_coverage = 0.0
    if float(num_total_kernels) > 0:
        num_coverage = float(num_custom_kernels) / float(num_total_kernels)
    time_coverage = 0.0
    if float(total_kernel_time) > 0:
        time_coverage = float(custom_kernel_time) / float(total_kernel_time)

    detail_env_time = _extract_detail_env_time(env_state)
    kernel_perf_cv = _coefficient_of_variation(metadata, "kg_kernel_perf_mean_ms", "kg_kernel_perf_std_ms")
    refer_perf_cv = _coefficient_of_variation(metadata, "kg_reference_perf_mean_ms", "kg_reference_perf_std_ms")

    env_extra_info = {
        "time_coverage": float(f"{time_coverage:.2f}"),
        "num_coverage": float(f"{num_coverage:.2f}"),
        "correctness": _require_env_value(env_state, "correctness", "env_state"),
        "compilation": _require_env_value(env_state, "compiled", "env_state"),
        "speedup": _require_env_value(env_state, "speedup", "env_state"),
        "decoy_kernel": bool(env_state.get("decoy_kernel", False)),
        "precheck": env_state.get("precheck"),
        "detail_env_time": detail_env_time,
    }
    if isinstance(kernel_perf_cv, (int, float)) and not isinstance(kernel_perf_cv, bool):
        env_extra_info["kernel_perf_cv"] = float(kernel_perf_cv)
    if isinstance(refer_perf_cv, (int, float)) and not isinstance(refer_perf_cv, bool):
        env_extra_info["refer_perf_cv"] = float(refer_perf_cv)
    return env_extra_info


def _normalize_env_feedback_fields(env_state: dict[str, Any]) -> dict[str, Any]:
    env_state = dict(env_state or {})
    env_state.setdefault("decoy_kernel", False)
    env_state.setdefault("precheck", "passed")

    metadata = env_state.get("metadata") if isinstance(env_state.get("metadata"), dict) else {}
    runtime_error = metadata.get("runtime_error")
    metadata_error = metadata.get("error")
    correctness_issue = metadata.get("correctness_issue")
    error_message = env_state.get("error_message") or env_state.get("error")
    env_precheck_error_message = _extract_env_precheck_error_message(env_state)

    if env_state.get("decoy_kernel", False):
        default_error_message = (
            "Reward hacking: Decoy kernel detected. The submitted code appears to bypass the intended custom "
            "CUDA implementation, for example by calling framework/library compute operators, returning cached or "
            "shortcut results, or defining kernels that are not actually used for the model computation. Please "
            "remove shortcut operators and implement the full computation through real custom CUDA code."
        )
        error_message = error_message or default_error_message
        env_state.update(
            {
                "speedup": env_state.get("speedup", 0.0),
                "error": DECOY_KERNEL_DETECTED,
                "error_message": error_message,
            }
        )

    elif env_precheck_error_message is not None:
        env_state.update(
            {
                "status": "failed",
                "precheck": "failed",
                "success": False,
                "correctness": None,
                "compiled": None,
                "speedup": None,
                "error": PRECHECK_ERROR,
                "error_message": env_precheck_error_message,
            }
        )
    elif env_state.get("compiled") is False or (
        isinstance(error_message, str) and "compilation failed" in error_message.lower()
    ):
        env_state.update(
            {
                "success": False,
                "correctness": None,
                "compiled": False,
                "error": COMPILATION_ERROR,
                "error_message": _format_compilation_error_message(env_state),
            }
        )
    elif env_state.get("status") == "timeout":
        error_message = error_message or "Task failed: Kernel evaluation timed out."
        env_state.update(
            {
                "success": False,
                "error": KERNEL_EVAL_TIMEOUT,
                "error_message": error_message,
            }
        )
    elif env_state.get("error_code") == CORRECTNESS_ERROR:
        correctness_error_message = str(error_message or "Kernel produced incorrect results")
        if correctness_issue not in (None, ""):
            correctness_detail = f"correctness error ：{correctness_issue}"
            if correctness_detail not in correctness_error_message:
                correctness_error_message = f"{correctness_error_message}\n{correctness_detail}"
        env_state.update(
            {
                "success": False,
                "error": CORRECTNESS_ERROR,
                "error_message": correctness_error_message,
            }
        )
    elif env_state.get("status") == "failed" or error_message is not None:
        env_state.update(
            {
                "success": False,
                "error": KERNEL_EVAL_FAILED,
                "error_message": error_message or "Task failed: Kernel evaluation error.",
            }
        )

    if runtime_error:
        runtime_error = str(runtime_error)
        current_error_message = str(env_state.get("error_message") or "")
        if runtime_error == current_error_message:
            env_state["error_message"] = f"runtime error: \n{runtime_error}"
        elif runtime_error not in current_error_message:
            env_state["error_message"] = "\n".join(
                part for part in (current_error_message, "other error message:", runtime_error) if part
            )
    if metadata_error:
        metadata_error = str(metadata_error)
        current_error_message = str(env_state.get("error_message") or "")
        if metadata_error not in current_error_message:
            env_state["error_message"] = "\n".join(
                part for part in (current_error_message, "other error message:", metadata_error) if part
            )
    if "speedup" in env_state:
        env_state["speedup"] = _round_numeric(env_state["speedup"])
    return env_state


def _strip_env_feedback_fields(env_state: dict[str, Any]) -> dict[str, Any]:
    env_state = dict(env_state or {})
    for key in ("submitted_at", "completed_at", "error_code"):
        env_state.pop(key, None)
    metadata = env_state.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        for key in ("aten_detection_trials", "aten_ops"):
            metadata.pop(key, None)
        for key in ("allowed_aten_ops", "forbidden_aten_ops"):
            if key in metadata:
                metadata[key] = _compact_aten_ops(metadata[key])
        if "device_info" in metadata:
            for key in ("gpu_name", "hardware"):
                metadata.pop(key, None)
        for key in METADATA_POP_KEYS:
            metadata.pop(key, None)
        for key in list(metadata):
            if key.startswith(("kg_stage_", "kg_reference_", "wg_", "tm_", "correctness_budget_")) or (
                key.startswith("correctness_") and key.endswith("_trial_s")
            ):
                metadata.pop(key, None)
            elif key.startswith("kg_kernel_") and key != "kg_kernel_total_s":
                metadata.pop(key, None)
        if "kg_kernel_total_s" in metadata:
            metadata["kg_kernel_total_s"] = _round_numeric(metadata["kg_kernel_total_s"])
        if "entry_point" in metadata:
            metadata["refer_entry_point"] = metadata.pop("entry_point")
        if "ncu" in metadata:
            metadata["ncu"] = _normalize_ncu_metadata(metadata["ncu"])
        compile_artifact = metadata.get("compile_artifact")
        if isinstance(compile_artifact, dict):
            compile_artifact = dict(compile_artifact)
            for key in COMPILE_ARTIFACT_POP_KEYS:
                compile_artifact.pop(key, None)
            if "entry_point" in compile_artifact:
                metadata["kernel_entry_point"] = compile_artifact.pop("entry_point")
            metadata["compile_artifact"] = compile_artifact
        env_state["metadata"] = metadata
    return env_state


def normalize_env_feedback(env_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_env_state = _normalize_env_feedback_fields(env_state)
    env_extra_info = _extract_env_extra_info(normalized_env_state)
    return _strip_env_feedback_fields(normalized_env_state), env_extra_info


def split_think_response(response: str) -> tuple[str | None, str]:
    start_tag = "<think>"
    end_tag = "</think>"
    start = response.find(start_tag)
    end = response.find(end_tag, start + len(start_tag)) if start >= 0 else -1
    if start >= 0 and end >= 0:
        response_think = response[start + len(start_tag) : end].strip("\n")
        response_content = (response[:start] + response[end + len(end_tag) :]).lstrip("\n")
        return response_think, response_content

    end = response.rfind(end_tag)
    if end >= 0:
        response_think = response[:end].strip("\n")
        response_content = response[end + len(end_tag) :].lstrip("\n")
        return response_think, response_content

    return None, response


def validate_code(code: str | None, entry_point: str = "Model") -> tuple[bool, str]:
    if not code:
        return False, f"MODEL_NEW validation error: Python code is required and must contain a '{entry_point}' class"
    if f"class {entry_point}" not in code:
        return False, f"MODEL_NEW validation error: Python code must contain a '{entry_point}' class"
    return True, ""


def _find_register_binding_semicolon_issue(source_map: dict[str, str]) -> tuple[str, int] | None:
    marker = "REGISTER_BINDING("

    for file_name, content in source_map.items():
        if not file_name.endswith(".cpp") or marker not in content:
            continue

        search_start = 0
        while True:
            marker_index = content.find(marker, search_start)
            if marker_index == -1:
                break

            paren_depth = 0
            closing_index = None
            i = marker_index + len("REGISTER_BINDING")
            while i < len(content):
                char = content[i]
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        closing_index = i
                        break
                i += 1

            if closing_index is None:
                line_no = content.count("\n", 0, marker_index) + 1
                return file_name, line_no

            next_index = closing_index + 1
            while next_index < len(content) and content[next_index].isspace():
                next_index += 1

            if next_index >= len(content) or content[next_index] != ";":
                line_no = content.count("\n", 0, marker_index) + 1
                return file_name, line_no

            search_start = closing_index + 1

    return None


class _ExtensionCallVisitor(ast.NodeVisitor):
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.module_aliases: set[str] = {module_name}
        self.from_import_aliases: dict[str, str] = {}
        self.detected_calls: set[str] = set()
        self.imported = False

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            if alias.name == self.module_name:
                self.module_aliases.add(alias.asname or alias.name)
                self.imported = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.module != self.module_name:
            self.generic_visit(node)
            return
        self.imported = True
        for alias in node.names:
            if alias.name == "*":
                continue
            self.from_import_aliases[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in self.module_aliases
        ):
            self.detected_calls.add(func.attr)
        elif isinstance(func, ast.Name) and func.id in self.from_import_aliases:
            self.detected_calls.add(self.from_import_aliases[func.id])
        self.generic_visit(node)


def _detect_extension_calls(model_code: str, module_name: str) -> tuple[bool, list[str]]:
    tree = ast.parse(model_code)
    visitor = _ExtensionCallVisitor(module_name)
    visitor.visit(tree)
    return visitor.imported or module_name in model_code, sorted(visitor.detected_calls)


def _extract_tvm_ffi_exports(source_map: dict[str, str]) -> list[str]:
    export_pattern = re.compile(
        r"\bTVM_FFI_DLL_EXPORT_TYPED_FUNC\s*\(\s*([A-Za-z_]\w*)\s*,",
        re.MULTILINE,
    )
    exports: set[str] = set()
    for name, content in source_map.items():
        if not name.lower().endswith((".cpp", ".cc", ".cxx")):
            continue
        exports.update(export_pattern.findall(str(content)))
    return sorted(exports)


def precheck_cuda_agent_code(
    model_code: str | None,
    cuda_sources: dict[str, str],
    *,
    entry_point: str = "ModelNew",
) -> tuple[str, str | None, str]:
    try:
        source_map = cuda_sources or {}

        def fail(message: str, error: str | None) -> tuple[str, str | None, str]:
            formatted_message = f"Code precheck failed: {message}"
            return formatted_message, error, "failed"

        is_valid, error_msg = validate_code(model_code, entry_point)
        if not is_valid:
            return fail(error_msg, VALIDATION_ERROR)

        model_code = model_code or ""
        try:
            compile(model_code, "<string>", "exec")
        except SyntaxError as exc:
            return fail(f"Syntax error in model code: {exc}", SYNTAX_ERROR)

        if not source_map:
            return fail("CUDA sources are required for CUDA-Agent compilation", VALIDATION_ERROR)

        cu_files = [name for name in source_map if name.endswith(".cu")]
        if not cu_files:
            return fail("CUDA-Agent sources must include at least one .cu file", VALIDATION_ERROR)

        combined_sources = "\n".join(str(content) for content in source_map.values())
        combined_cpp = "\n".join(str(content) for name, content in source_map.items() if name.endswith(".cpp"))

        python_ext_markers = [
            "REGISTER_BINDING(",
            "pybind11::module",
            "m.def(",
            '#include "../binding_registry.h"',
            "#include <torch/types.h>",
            "torch::Tensor",
        ]
        tvm_ffi_markers = [
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(",
            "TVMFFIEnvGetStream(",
            "#include <tvm/ffi/tvm_ffi.h>",
            "#include <tvm/ffi/extra/c_env_api.h>",
            "#include <tvm/ffi/function.h>",
            "#include <tvm/ffi/container/tensor.h>",
            "tvm::ffi::Tensor",
            "tvm::ffi::TensorView",
        ]

        has_python_ext_markers = any(marker in combined_sources for marker in python_ext_markers)
        has_tvm_ffi_markers = any(marker in combined_sources for marker in tvm_ffi_markers)

        if has_python_ext_markers and has_tvm_ffi_markers:
            binding_api = "mixed"
        elif has_tvm_ffi_markers:
            binding_api = "tvm_ffi"
        elif has_python_ext_markers:
            binding_api = "pybind"
        else:
            binding_api = "unknown"

        try:
            imported_extension, detected_calls = _detect_extension_calls(model_code, "cuda_extension")
        except SyntaxError as exc:
            return fail(f"Syntax error in model code: {exc}", SYNTAX_ERROR)
        if not imported_extension:
            return fail("model_new.py must import or reference cuda_extension", IMPORT_ERROR)
        if not detected_calls:
            return fail("model_new.py must call at least one cuda_extension function", VALIDATION_ERROR)

        required_marker = '#include "../binding_registry.h"'
        if required_marker not in combined_cpp:
            return fail("binding source must include ../binding_registry.h", VALIDATION_ERROR)

        if binding_api == "tvm_ffi":
            return fail(
                "binding source must use pybind/binding_registry style bindings, but TVM-FFI exports were detected",
                VALIDATION_ERROR,
            )

        if binding_api == "unknown":
            return fail("No supported Python extension binding pattern detected in CUDA sources", VALIDATION_ERROR)

        register_binding_issue = _find_register_binding_semicolon_issue(source_map)
        if register_binding_issue is not None:
            issue_file, issue_line = register_binding_issue
            return fail(f"{issue_file}:{issue_line} has REGISTER_BINDING(...) without a trailing ';'", SYNTAX_ERROR)

        return "", None, "passed"
    except Exception as exc:
        message = f"Code precheck failed: internal validation error: {exc}"
        return message, VALIDATION_ERROR, "failed"


def precheck_cuda_tvm_code(
    model_code: str | None,
    cuda_sources: dict[str, str],
    *,
    entry_point: str = "ModelNew",
) -> tuple[str, str | None, str]:
    try:
        source_map = cuda_sources or {}

        def fail(message: str, error: str | None) -> tuple[str, str | None, str]:
            formatted_message = f"Code precheck failed: {message}"
            return formatted_message, error, "failed"

        is_valid, error_msg = validate_code(model_code, entry_point)
        if not is_valid:
            return fail(error_msg, VALIDATION_ERROR)

        model_code = model_code or ""
        try:
            compile(model_code, "<string>", "exec")
        except SyntaxError as exc:
            return fail(f"Syntax error in model code: {exc}", SYNTAX_ERROR)

        try:
            imported_extension, detected_calls = _detect_extension_calls(model_code, "tvm_ffi_extension")
        except SyntaxError as exc:
            return fail(f"Syntax error in model code: {exc}", SYNTAX_ERROR)

        if not imported_extension:
            return fail("model_new.py must import or reference tvm_ffi_extension", IMPORT_ERROR)
        if not detected_calls:
            return fail("model_new.py must call at least one tvm_ffi_extension function", VALIDATION_ERROR)

        if not source_map:
            return fail("CUDA sources are required for TVM-FFI compilation", VALIDATION_ERROR)

        cu_files = [name for name in source_map if name.lower().endswith(".cu")]
        cpp_files = [name for name in source_map if name.lower().endswith((".cpp", ".cc", ".cxx"))]
        if not cu_files:
            return fail("TVM-FFI sources must include at least one .cu file", VALIDATION_ERROR)
        if not cpp_files:
            return fail("TVM-FFI sources must include at least one .cpp binding file", VALIDATION_ERROR)

        binding_candidates = [name for name in cpp_files if "binding" in name.lower() or "bind" in name.lower()]
        if not binding_candidates:
            return fail("TVM-FFI sources must include a binding .cpp file", VALIDATION_ERROR)

        combined_cpp = "\n".join(str(source_map[name]) for name in cpp_files)
        forbidden_markers = (
            "PYBIND11_MODULE",
            "REGISTER_BINDING(",
            "binding_registry.h",
        )
        for marker in forbidden_markers:
            if marker in combined_cpp:
                return fail(f"TVM-FFI binding source must not use pybind11 marker {marker}", VALIDATION_ERROR)

        host_cuda_runtime_markers = (
            "#include <cuda_runtime.h>",
            "#include <cuda.h>",
            "cudaStream_t",
        )
        for marker in host_cuda_runtime_markers:
            if marker in combined_cpp:
                return fail(
                    "TVM-FFI host binding source must keep CUDA runtime headers/types out of "
                    f"binding .cpp files; use an opaque void* stream handle instead of {marker}",
                    VALIDATION_ERROR,
                )

        tvm_header_markers = (
            "#include <tvm/ffi/tvm_ffi.h>",
            "#include <tvm/ffi/function.h>",
            "#include <tvm/ffi/container/tensor.h>",
        )
        if not any(marker in combined_cpp for marker in tvm_header_markers):
            return fail("TVM-FFI binding source must include a tvm/ffi header", VALIDATION_ERROR)

        exported_functions = _extract_tvm_ffi_exports(source_map)
        if not exported_functions:
            return fail(
                "TVM-FFI binding source must export functions with TVM_FFI_DLL_EXPORT_TYPED_FUNC(...)",
                VALIDATION_ERROR,
            )

        missing_exports = sorted(set(detected_calls) - set(exported_functions))
        if missing_exports:
            return fail(
                "TVM-FFI model calls are not exported: " + ", ".join(missing_exports),
                VALIDATION_ERROR,
            )

        return "", None, "passed"
    except Exception as exc:
        message = f"Code precheck failed: internal validation error: {exc}"
        return message, VALIDATION_ERROR, "failed"


def _strip_think_blocks(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL | re.IGNORECASE)


def _section_block_pattern() -> re.Pattern[str]:
    language_pattern = r"(?:cpp|c\+\+|cxx|cuda|cu|python|py)?"
    return re.compile(
        rf"###\s*({'|'.join(CUDA_SECTIONS)})\s*```{language_pattern}\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )


def _find_last_complete_section_group(response: str) -> dict[str, str]:
    response = _strip_think_blocks(response)
    complete_groups: list[dict[str, str]] = []
    current_group: dict[str, str] = {}
    expected_index = 0

    for match in _section_block_pattern().finditer(response):
        section_name = match.group(1).upper()
        section_body = match.group(2).strip()

        if section_name == "CUDA_KERNELS":
            current_group = {section_name: section_body}
            expected_index = 1
            continue

        if not current_group:
            continue

        expected_section = CUDA_SECTIONS[expected_index]
        if section_name != expected_section:
            continue

        current_group[section_name] = section_body
        expected_index += 1
        if expected_index == len(CUDA_SECTIONS):
            complete_groups.append(current_group)
            current_group = {}
            expected_index = 0

    return complete_groups[-1] if complete_groups else {}


def _find_last_sections(response: str) -> dict[str, str]:
    response = _strip_think_blocks(response)
    sections: dict[str, str] = {}
    for match in _section_block_pattern().finditer(response):
        sections[match.group(1).upper()] = match.group(2).strip()
    return sections


def parse_cuda_agent_response(response: str) -> tuple[dict[str, str], str | None]:
    cuda_sources: dict[str, str] = {}
    model_new_code: str | None = None
    section_group = _find_last_complete_section_group(response) or _find_last_sections(response)

    if "CUDA_KERNELS" in section_group:
        cuda_sources["kernels/generated.cu"] = section_group["CUDA_KERNELS"]
    if "APPLY_BINDINGS" in section_group:
        cuda_sources["kernels/generated_binding.cpp"] = section_group["APPLY_BINDINGS"]
    if "MODEL_NEW" in section_group:
        model_new_code = section_group["MODEL_NEW"]

    return cuda_sources, model_new_code


def extract_cuda_agent_kernel_code(response: str) -> str:
    section_group = _find_last_complete_section_group(response) or _find_last_sections(response)
    section_lang = {
        "CUDA_KERNELS": "cpp",
        "APPLY_BINDINGS": "cpp",
        "MODEL_NEW": "python",
    }

    ordered_sections = []
    for section_name in CUDA_SECTIONS:
        section_body = section_group.get(section_name)
        if section_body:
            ordered_sections.append(f"### {section_name}\n```{section_lang[section_name]}\n{section_body}\n```")

    return "\n\n".join(ordered_sections) if ordered_sections else response


def precheck_response(
    response: str,
    entry_point: str,
    backend: str,
) -> tuple[bool, dict[str, Any] | None]:
    if backend == "cuda_agent":
        precheck_func = precheck_cuda_agent_code
    elif backend == "tvm_ffi":
        precheck_func = precheck_cuda_tvm_code
    else:
        return True, None

    cuda_sources, model_new_code = parse_cuda_agent_response(response)
    error_message, error, precheck = precheck_func(
        model_new_code,
        cuda_sources,
        entry_point=entry_point,
    )
    if precheck == "passed":
        return True, None
    else:
        precheck_state = {
            "status": "failed",
            "precheck": precheck,
            "success": False,
            "correctness": None,
            "compiled": None,
            "speedup": None,
            "decoy_kernel": False,
            "error": error,
            "error_message": error_message,
            "metadata": {
                "kernel_eval_failure": True,
            },
        }
        return False, precheck_state


def _mark_remove_sample(sample: Sample, reason: str) -> None:
    sample.remove_sample = True
    sample.reward = 0.0
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["remove_reason"] = reason


def _set_multi_turn_rewards(args, output_samples: list[Sample], finish_reason: str) -> None:
    gamma = float(getattr(args, "multi_turn_gamma", 1.0))
    if args.advantage_estimator not in ["trloo"] or gamma == 0.0:
        return

    multi_turn_reward = 0.0
    for reverse_idx, sample in enumerate(reversed(output_samples)):
        turn_idx = sample.metadata.get("turn_idx") if isinstance(sample.metadata, dict) else None
        expected_turn_idx = len(output_samples) - 1 - reverse_idx
        assert int(turn_idx) == expected_turn_idx, f"turn_idx mismatch: {turn_idx=} {expected_turn_idx=}"
        turn_reward = 0.0 if sample.remove_sample else float(sample.reward)
        multi_turn_reward = turn_reward + gamma * multi_turn_reward
        sample.metadata = dict(sample.metadata or {})
        sample.metadata.update(
            {
                "multi_turn_reward": multi_turn_reward,
                "trajectory_finish_reason": finish_reason,
            }
        )


def _apply_coverage_rs(args, output_samples: list[Sample]) -> None:
    if not getattr(args, "use_coverage_rs", False):
        return

    coverage_key = getattr(args, "coverage_rs_key", "time_coverage")
    threshold = float(getattr(args, "coverage_rs_threshold", 0.3))
    factor = getattr(args, "coverage_rs_factor", 0.1)
    factor = None if factor is None else float(factor)

    for sample in output_samples:
        if (
            sample.remove_sample
            or sample.status == Sample.Status.ABORTED
            or (sample.loss_mask is not None and sum(sample.loss_mask) == 0)
        ):
            continue
        env_extra_info = sample.metadata.get("env_extra_info") if isinstance(sample.metadata, dict) else None
        if not isinstance(env_extra_info, dict):
            raise ValueError("--use-coverage-rs requires sample.metadata['env_extra_info'].")
        if coverage_key not in env_extra_info:
            raise KeyError(f"env_extra_info missing coverage key: {coverage_key}")

        is_correct = bool(env_extra_info["correctness"]) and not bool(env_extra_info["decoy_kernel"])
        if not is_correct:
            continue

        coverage = float(env_extra_info[coverage_key])
        if factor is None or factor == 0.0:
            keep_prob = 1.0 if coverage >= threshold else 0.0
        else:
            keep_prob = min(max((coverage - threshold) / factor, 0.0), 1.0)
        if random.random() > keep_prob:
            _mark_remove_sample(sample, "coverage_rs")


def _apply_rollout_progress_metadata(output_samples: list[Sample], finish_reason: str) -> None:
    total_model_time = 0.0
    total_env_time = 0.0
    num_turns_completed = 0
    abort_reason = None
    existing_total_model_time = 0.0
    existing_total_env_time = 0.0
    existing_num_turns_completed = 0

    for sample in output_samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        if abort_reason is None and metadata.get("abort_reason") is not None:
            abort_reason = metadata.get("abort_reason")
        existing_total_model_time = max(existing_total_model_time, float(metadata.get("total_model_time", 0.0) or 0.0))
        existing_total_env_time = max(existing_total_env_time, float(metadata.get("total_env_time", 0.0) or 0.0))
        existing_num_turns_completed = max(
            existing_num_turns_completed, int(metadata.get("num_turns_completed", 0) or 0)
        )
        if metadata.get("is_pad_turn"):
            continue
        total_model_time += float(metadata.get("model_time", 0.0) or 0.0)
        total_env_time += float(metadata.get("env_time", 0.0) or 0.0)
        if sample.status != Sample.Status.ABORTED:
            num_turns_completed += 1

    total_model_time = max(total_model_time, existing_total_model_time)
    total_env_time = max(total_env_time, existing_total_env_time)
    num_turns_completed = max(num_turns_completed, existing_num_turns_completed)

    total_request_time = total_model_time + total_env_time
    for sample in output_samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        total_request_time = max(total_request_time, float(metadata.get("total_request_time", 0.0) or 0.0))

    progress = {
        "finish_reason": finish_reason,
        "total_request_time": total_request_time,
        "num_turns_completed": num_turns_completed,
        "total_model_time": total_model_time,
        "total_env_time": total_env_time,
    }
    if abort_reason is not None:
        progress["abort_reason"] = abort_reason
    elif finish_reason == "model_abort":
        progress["abort_reason"] = "model_abort"
    elif finish_reason == "aborted":
        progress["abort_reason"] = "aborted"

    for sample in output_samples:
        sample.metadata = {**(sample.metadata or {}), **progress}


def postprocess_turn_samples(args, output_samples: list[Sample], finish_reason: str) -> list[Sample]:
    """Postprocess turn samples.

    finalize_mode:
    - None: keep all generated turn samples.
    - "positive": after max_reward > 0, mark turns with reward <= 0 as remove_sample.
    - "improve": after max_reward > 0, mark turns with reward <= max_reward as remove_sample.
    """

    if not output_samples:
        return []

    _apply_rollout_progress_metadata(output_samples, finish_reason)

    for sample in output_samples:
        if sample.remove_sample:
            sample.metadata = dict(sample.metadata or {})
            sample.metadata.setdefault("remove_reason", "pre_removed")
            sample.reward = 0.0

    if finish_reason == "model_abort":
        for sample in output_samples:
            reason = "pad_turn" if sample.metadata.get("is_pad_turn") else "model_abort"
            _mark_remove_sample(sample, reason)
        _set_multi_turn_rewards(args, output_samples, finish_reason)
        return output_samples

    _apply_coverage_rs(args, output_samples)

    finalize_mode = getattr(args, "finalize_mode", "positive")
    if finalize_mode == "none":
        finalize_mode = None
    if finalize_mode is None:
        _set_multi_turn_rewards(args, output_samples, finish_reason)
        return output_samples

    def is_meaningful_turn(sample: Sample, max_reward: float) -> bool:
        if max_reward > 0.0:
            reward = float(sample.reward)
            if finalize_mode == "improve":
                return reward > max_reward
            return reward > 0.0
        return True

    max_reward = float(output_samples[0].reward)
    for sample in output_samples[1:]:
        if sample.remove_sample:
            continue
        reward = float(sample.reward)
        if not is_meaningful_turn(sample, max_reward):
            _mark_remove_sample(sample, f"finalize_{finalize_mode}")
        max_reward = max(max_reward, reward)

    _set_multi_turn_rewards(args, output_samples, finish_reason)
    return output_samples
