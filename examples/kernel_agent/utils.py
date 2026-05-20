import logging
import re
from typing import Any

from slime.utils.types import Sample

try:
    from .config import CUDA_AGENT_CONFIGS
except ImportError:
    from config import CUDA_AGENT_CONFIGS

logger = logging.getLogger(__name__)
_MULTI_TURN_GAMMA_WARNING_LOGGED = False


VALIDATION_ERROR = "VALIDATION_ERROR"
SYNTAX_ERROR = "SYNTAX_ERROR"
IMPORT_ERROR = "IMPORT_ERROR"
PRECHECK_ERROR = "PRECHECK_ERROR"
COMPILATION_ERROR = "COMPILATION_ERROR"
DECOY_KERNEL_DETECTED = "DECOY_KERNEL_DETECTED"
KERNEL_EVAL_ERROR = "KERNEL_EVAL_ERROR"
KERNEL_EVAL_TIMEOUT = "KERNEL_EVAL_TIMEOUT"


def _format_compilation_error_message(env_state: dict[str, Any]) -> str:
    metadata = env_state.get("metadata") if isinstance(env_state.get("metadata"), dict) else {}
    compile_artifact = metadata.get("compile_artifact")

    compile_error = None
    if isinstance(compile_artifact, dict):
        compile_error = compile_artifact.pop("error", None)
    if compile_error:
        return f"Compilation failed. Compiler output:\n{compile_error}"
    return str(env_state.get("error_message") or "Compilation failed.")


def _normalize_failed_error(env_state: dict[str, Any]) -> tuple[str, str]:
    error_message = env_state.get("error_message") or env_state.get("error") or "Task failed"
    error_message = str(error_message)
    lower_error_message = error_message.lower()
    if "kernel compilation failed" in lower_error_message:
        return COMPILATION_ERROR, _format_compilation_error_message(env_state)
    if env_state.get("status") == "timeout":
        return KERNEL_EVAL_TIMEOUT, error_message
    return KERNEL_EVAL_ERROR, error_message


def normalize_env_feedback(env_state: dict[str, Any]) -> dict[str, Any]:
    env_state = dict(env_state or {})
    env_state.pop("error_code", None)
    for key in ("submitted_at", "completed_at"):
        env_state.pop(key, None)
    metadata = env_state.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        for key in (
            "compile_only",
            "required_resource",
            "task_id",
            "inline_gpu_execute_completed",
            "inline_compile_worker_id",
            "inline_compile_worker_device",
        ):
            metadata.pop(key, None)
        if "entry_point" in metadata:
            metadata["refer_entry_point"] = metadata.pop("entry_point")
        compile_artifact = metadata.get("compile_artifact")
        if isinstance(compile_artifact, dict):
            compile_artifact = dict(compile_artifact)
            for key in ("precheck", "compiled", "compile_mode", "source_mode", "backend"):
                compile_artifact.pop(key, None)
            if "entry_point" in compile_artifact:
                metadata["kernel_entry_point"] = compile_artifact.pop("entry_point")
            metadata["compile_artifact"] = compile_artifact
        env_state["metadata"] = metadata

    if env_state.get("decoy_kernel", False):
        default_error_message = (
            "Reward hacking: Decoy kernel detected. Please remove shortcut operators and implement the computation "
            "with custom CUDA code."
        )
        error_message = env_state.get("error_message") or default_error_message
        env_state.update(
            {
                "speedup": env_state.get("speedup", 0.0),
                "decoy_kernel": True,
                "error": DECOY_KERNEL_DETECTED,
                "error_message": error_message,
            }
        )
        return env_state

    if env_state.get("compiled") is False:
        env_state.update(
            {
                "success": False,
                "correctness": None,
                "compiled": False,
                "decoy_kernel": None,
                "reference_runtime": None,
                "kernel_runtime": None,
                "speedup": None,
                "error": COMPILATION_ERROR,
                "error_message": _format_compilation_error_message(env_state),
            }
        )
        return env_state

    if env_state.get("status") != "completed":
        error, error_message = _normalize_failed_error(env_state)
        env_state.update(
            {
                "success": False,
                "correctness": False,
                "compiled": False,
                "speedup": 0.0,
                "error": error,
                "error_message": error_message,
            }
        )
    return env_state


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


def precheck_cuda_agent_code(
    model_code: str | None,
    cuda_sources: dict[str, str],
    *,
    entry_point: str = "ModelNew",
    source_mode: str = "files",
) -> tuple[str, str | None, str]:
    try:
        normalized_mode = str(source_mode or "files").strip().lower()
        source_map = cuda_sources or {}

        def fail(message: str, error: str | None) -> tuple[str, str | None, str]:
            formatted_message = f"CUDA-Agent precheck failed: {message}"
            return formatted_message, error, "failed"

        is_valid, error_msg = validate_code(model_code, entry_point)
        if not is_valid:
            return fail(error_msg, VALIDATION_ERROR)

        try:
            compile(model_code, "<string>", "exec")
        except SyntaxError as exc:
            return fail(f"Syntax error in model code: {exc}", SYNTAX_ERROR)

        if "cuda_extension" not in model_code:
            return fail("model_new.py must use cuda_extension", IMPORT_ERROR)

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
            binding_api = "python_extension"
        else:
            binding_api = "unknown"

        if normalized_mode in {"files", "inline"}:
            required_marker = '#include "../binding_registry.h"'
            if required_marker not in combined_cpp:
                return fail("binding source must include ../binding_registry.h", VALIDATION_ERROR)

            if binding_api == "tvm_ffi":
                return fail(
                    "source_mode files/inline expects a pybind/binding_registry style binding, but TVM-FFI exports were detected",
                    VALIDATION_ERROR,
                )

            if binding_api == "unknown":
                return fail("No supported Python extension binding pattern detected in CUDA sources", VALIDATION_ERROR)

            register_binding_issue = _find_register_binding_semicolon_issue(source_map)
            if register_binding_issue is not None:
                issue_file, issue_line = register_binding_issue
                return fail(f"{issue_file}:{issue_line} has REGISTER_BINDING(...) without a trailing ';'", SYNTAX_ERROR)
        elif normalized_mode == "tvm_ffi":
            if binding_api == "python_extension":
                return fail(
                    "source_mode='tvm_ffi' requires TVM-FFI exports instead of REGISTER_BINDING/pybind bindings",
                    VALIDATION_ERROR,
                )
            if binding_api == "unknown":
                return fail(
                    "source_mode='tvm_ffi' requires TVM_FFI_DLL_EXPORT_TYPED_FUNC(...) or tvm::ffi Tensor bindings",
                    VALIDATION_ERROR,
                )

        return "", None, "passed"
    except Exception as exc:
        message = f"CUDA-Agent precheck failed: internal validation error: {exc}"
        return message, VALIDATION_ERROR, "failed"


def parse_cuda_agent_response(response: str) -> tuple[dict[str, str], str | None]:
    cuda_sources: dict[str, str] = {}
    model_new_code: str | None = None
    cuda_kernels_match = re.search(
        r"###\s*CUDA_KERNELS\s*```(?:cpp|c\+\+)?\s*\n(.*?)```",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    apply_bindings_match = re.search(
        r"###\s*APPLY_BINDINGS\s*```(?:cpp|c\+\+)?\s*\n(.*?)```",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    model_new_match = re.search(
        r"###\s*MODEL_NEW\s*```python\s*\n(.*?)```",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if cuda_kernels_match:
        cuda_sources["kernels/generated.cu"] = cuda_kernels_match.group(1).strip()
    if apply_bindings_match:
        cuda_sources["kernels/generated_binding.cpp"] = apply_bindings_match.group(1).strip()
    if model_new_match:
        model_new_code = model_new_match.group(1).strip()

    return cuda_sources, model_new_code


def extract_cuda_agent_kernel_code(response: str) -> str:
    section_patterns = {
        "CUDA_KERNELS": r"###\s*CUDA_KERNELS\s*```(?:cpp|c\+\+)?\s*\n(.*?)```",
        "APPLY_BINDINGS": r"###\s*APPLY_BINDINGS\s*```(?:cpp|c\+\+)?\s*\n(.*?)```",
        "MODEL_NEW": r"###\s*MODEL_NEW\s*```python\s*\n(.*?)```",
    }
    section_matches = {}
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, response or "", re.DOTALL | re.IGNORECASE)
        if match:
            section_matches[section_name] = match.group(1).strip()

    ordered_sections = []
    for section_name, lang in (
        ("CUDA_KERNELS", "cpp"),
        ("APPLY_BINDINGS", "cpp"),
        ("MODEL_NEW", "python"),
    ):
        section_body = section_matches.get(section_name)
        if section_body:
            ordered_sections.append(f"### {section_name}\n```{lang}\n{section_body}\n```")

    return "\n\n".join(ordered_sections) if ordered_sections else response


def precheck_cuda_agent_response(response: str, entry_point: str) -> dict[str, Any] | None:
    cuda_sources, model_new_code = parse_cuda_agent_response(response)
    error_message, error, precheck = precheck_cuda_agent_code(
        model_new_code,
        cuda_sources,
        entry_point=entry_point,
        source_mode="files",
    )
    if precheck == "passed":
        return None

    return {
        "status": "failed",
        "precheck": precheck,
        "success": False,
        "correctness": None,
        "compiled": None,
        "speedup": None,
        "error": error,
        "error_message": error_message,
        "metadata": {
            "kernel_eval_failure": True,
        },
    }


def precheck_response(
    response: str,
    entry_point: str,
    backend: str,
) -> dict[str, Any] | None:
    if backend == "cuda_agent":
        return precheck_cuda_agent_response(response, entry_point)
    return None


def _set_multi_turn_rewards(output_samples: list[Sample], finish_reason: str, args=None) -> None:
    global _MULTI_TURN_GAMMA_WARNING_LOGGED

    gamma = getattr(args, "multi_turn_gamma", None) if args is not None else None
    if gamma is None:
        gamma = CUDA_AGENT_CONFIGS["reward"].get("multi_turn_gamma", 1.0)
    elif not _MULTI_TURN_GAMMA_WARNING_LOGGED:
        logger.warning(
            "Using args.multi_turn_gamma=%s for multi-turn reward folding; "
            "CUDA_AGENT_CONFIGS['reward']['multi_turn_gamma'] will be ignored.",
            gamma,
        )
        _MULTI_TURN_GAMMA_WARNING_LOGGED = True
    gamma = float(gamma)
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


def postprocess_turn_samples(output_samples: list[Sample], finish_reason: str, args=None) -> list[Sample]:
    """Postprocess turn samples.

    finalize_mode:
    - None: keep all generated turn samples.
    - "positive": after max_reward > 0, mark turns with reward <= 0 as remove_sample.
    - "improve": after max_reward > 0, mark turns with reward <= max_reward as remove_sample.
    """

    if not output_samples:
        return []

    if finish_reason == "model_abort":
        for sample in output_samples:
            sample.remove_sample = True
            sample.reward = 0.0
        _set_multi_turn_rewards(output_samples, finish_reason, args)
        return output_samples

    finalize_mode = CUDA_AGENT_CONFIGS.get("finalize_mode", "positive")
    if finalize_mode is None:
        _set_multi_turn_rewards(output_samples, finish_reason, args)
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
        reward = float(sample.reward)
        if not is_meaningful_turn(sample, max_reward):
            sample.remove_sample = True
            sample.reward = 0.0
        max_reward = max(max_reward, reward)

    _set_multi_turn_rewards(output_samples, finish_reason, args)
    return output_samples
