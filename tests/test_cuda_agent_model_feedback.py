import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.kernel_agent import generate_with_cuda_agent
from examples.kernel_agent.generate_with_cuda_agent import (
    _apply_feedback_template,
    _truncate_middle,
    build_model_feedback,
    compact_compiler_diagnostics,
)
from examples.kernel_agent.utils import normalize_env_feedback

from slime.rollout.sglang_rollout import PromptTemplate

NUM_GPUS = 0


def _rich_env_result():
    return {
        "env_state": {
            "status": "completed",
            "error": "DECOY_KERNEL_DETECTED",
            "error_message": "Use real custom kernels.",
            "precheck": "passed",
            "compiled": True,
            "correctness": False,
            "decoy_kernel": True,
            "speedup": 1.25,
            "reference_runtime": 2.0,
            "kernel_runtime": 1.6,
            "success": False,
            "task_id": "system-only-task-id",
            "processing_time": None,
            "metadata": {
                "aten_detection_trials": [{"large_raw_trial": "DROP_ATEN_TRIAL" * 100}],
                "aten_ops": [{"name": "DROP_ALL_ATEN_OPS"}],
                "allowed_aten_ops": [{"name": "DROP_ALLOWED_ATEN_OPS"}],
                "forbidden_aten_ops": [
                    {
                        "name": "aten::matmul",
                        "normalized_name": "aten::matmul",
                        "count": 3,
                        "cpu_time_us": 999.0,
                        "allowed": False,
                    }
                ],
                "aten_detection_valid": True,
                "policy_violation_reason": "native operator used",
                "decoy_reason": "native operator used",
                "suspected_decoy_reason": "LOW_CUSTOM_KERNEL_TIME_COVERAGE",
                "correctness_issue": "output mismatch",
                "correctness_issue_name": "value_mismatch",
                "max_difference": [0.5],
                "avg_difference": [0.1],
                "custom_kernel_names": ["real_kernel"],
                "custom_kernel_coverage": "1/2 (50%)",
                "custom_kernel_cuda_time_coverage": "1/2 (50%)",
                "num_custom_kernels": 1,
                "num_total_kernels": 2,
                "incorrect_backend_usage_probe": {
                    "attempted": True,
                    "valid": True,
                    "decoy_detected": True,
                    "matched_kernel_names": ["real_kernel"],
                    "missing_kernel_names": ["missing_kernel"],
                    "profiling": {"kernels": ["DROP_PROBE_PROFILE"]},
                },
                "profiling": {"kernels": ["DROP_FULL_PROFILE"], "memory_stats": {"reserved_mb": 1}},
                "device_info": {"gpu": "DROP_DEVICE_INFO"},
                "compile_artifact": {"path": "DROP_COMPILE_ARTIFACT"},
                "correctness_trial_s": [1.0, 2.0, 3.0],
                "compile_hostname": "DROP_HOSTNAME",
                "cpu_worker_id": "DROP_WORKER",
            },
        }
    }


def test_build_model_feedback_keeps_actions_and_drops_machine_metadata():
    env_result = _rich_env_result()
    original = deepcopy(env_result)

    feedback = build_model_feedback(env_result)
    rendered = str(feedback)

    assert feedback["status"] == "completed"
    assert feedback["error"] == "DECOY_KERNEL_DETECTED"
    assert feedback["performance"] == {
        "speedup": 1.25,
        "kernel_runtime": 1.6,
        "reference_runtime": 2.0,
    }
    assert feedback["correctness_details"]["max_difference"] == [0.5]
    assert feedback["policy_details"]["decoy_reasons"] == ["native operator used"]
    assert feedback["policy_details"]["policy_warnings"] == ["LOW_CUSTOM_KERNEL_TIME_COVERAGE"]
    assert feedback["policy_details"]["forbidden_aten_ops"] == [{"name": "aten::matmul", "count": 3}]
    assert feedback["policy_details"]["incorrect_backend_usage_probe"]["missing_kernel_names"] == ["missing_kernel"]
    assert feedback["profiling_summary"]["custom_kernel_names"] == ["real_kernel"]

    for sentinel in (
        "system-only-task-id",
        "DROP_ATEN_TRIAL",
        "DROP_ALL_ATEN_OPS",
        "DROP_ALLOWED_ATEN_OPS",
        "DROP_PROBE_PROFILE",
        "DROP_FULL_PROFILE",
        "DROP_DEVICE_INFO",
        "DROP_COMPILE_ARTIFACT",
        "DROP_HOSTNAME",
        "DROP_WORKER",
        "cpu_time_us",
    ):
        assert sentinel not in rendered
    assert env_result == original


def test_build_model_feedback_keeps_flat_factual_precheck_diagnostic():
    env_result = {
        "env_state": {
            "status": "failed",
            "error": "VALIDATION_ERROR",
            "error_message": "Code precheck failed: unresolved extension call",
            "precheck": "failed",
            "compiled": None,
            "correctness": None,
            "decoy_kernel": False,
            "metadata": {
                "precheck_diagnostic": {
                    "code": "TVM_FFI_UNRESOLVED_CALL",
                    "phase": "binding_contract",
                    "evidence": [
                        {
                            "kind": "extension_call",
                            "value": "ml_forward_ops",
                            "section": "MODEL_NEW",
                            "line": 29,
                        },
                        {
                            "kind": "exported_symbol",
                            "value": "mlp_forward_ops",
                            "section": "APPLY_BINDINGS",
                            "line": 77,
                        },
                    ],
                }
            },
        }
    }

    feedback = build_model_feedback(env_result)

    assert feedback["precheck_diagnostic"] == env_result["env_state"]["metadata"]["precheck_diagnostic"]

    def container_depth(value):
        if isinstance(value, dict):
            return 1 + max((container_depth(item) for item in value.values()), default=0)
        if isinstance(value, list):
            return 1 + max((container_depth(item) for item in value), default=0)
        return 0

    assert container_depth(feedback) == 4
    serialized = json.dumps(feedback)
    assert "nearest_export" not in serialized
    assert "suggested_edit" not in serialized
    assert "repair_scope" not in serialized


def test_compact_compiler_diagnostics_preserves_each_actionable_error_block():
    compiler_output = """Compilation failed. Compiler output:
ninja exited with status 1
stdout:
[1/3] /usr/local/cuda/bin/nvcc -I/very/long/include -c /dev/shm/kernelgym/compile_cache/hash/pkg/kernels/generated.cu -o cuda_0.o
FAILED: generated_binding.cpp.o
/usr/bin/c++ -I/very/long/include -I/another/long/include -c /dev/shm/kernelgym/compile_cache/hash/pkg/kernels/generated_binding.cpp -o cpp_0.o
/dev/shm/kernelgym/compile_cache/hash/pkg/kernels/generated_binding.cpp:42:5: error: first invalid call
    bad_call(x);
    ^~~~~~~~
/opt/tvm/include/tvm/ffi/error.h:459:9: note: in expansion of macro CHECK
    CHECK(x);
    ^~~~~
/dev/shm/kernelgym/compile_cache/hash/pkg/kernels/generated_binding.cpp:51:7: error: second invalid call
      other_bad_call(y);
      ^~~~~~~~~~~~~~
[3/3] : && /usr/bin/c++ -shared -L/very/long/lib cpp_0.o cuda_0.o -o kernelgym_hash.so && :
/usr/bin/c++ -shared -L/very/long/lib cpp_0.o cuda_0.o -o kernelgym_hash.so
generated_binding.cpp:(.text+0x10): undefined reference to `real_kernel_launcher'
collect2: error: ld returned 1 exit status
ninja: build stopped: subcommand failed.
"""

    compacted = compact_compiler_diagnostics(compiler_output, max_chars=6000)

    assert "[1/3] compile generated.cu -> cuda_0.o" in compacted
    assert "[3/3] link kernelgym_hash.so" in compacted
    assert ".../kernels/generated_binding.cpp:42:5: error: first invalid call" in compacted
    assert ".../kernels/generated_binding.cpp:51:7: error: second invalid call" in compacted
    assert "^~~~~~~~" in compacted
    assert "note: in expansion of macro CHECK" in compacted
    assert "undefined reference to `real_kernel_launcher'" in compacted
    assert "collect2: error" in compacted
    assert "-I/very/long/include" not in compacted
    assert "/dev/shm/kernelgym/compile_cache/hash" not in compacted
    assert "ninja exited" not in compacted
    assert "ninja: build stopped" not in compacted


def test_compiler_command_detection_never_consumes_inline_error_or_long_note():
    inline_error = "[2/2] c++: error: unrecognized command-line option '-fbad-flag'"
    long_note = "note: " + "c++ template context " * 40

    compacted = compact_compiler_diagnostics(f"{inline_error}\n{long_note}", max_chars=6000)

    assert inline_error in compacted
    assert long_note.rstrip() in compacted


def test_oversized_compiler_diagnostics_keeps_deduplicated_candidate_rejection_pairs():
    candidate_a = "ffi.h:10:3: note: candidate: bool convert(Tensor, int64_t)"
    rejection_a = "ffi.h:10:3: note: no known conversion for argument 1 from ShapeView to Tensor"
    candidate_b = "ffi.h:20:3: note: candidate: bool convert(String, int64_t)"
    rejection_b = "ffi.h:20:3: note: no known conversion for argument 1 from ShapeView to String"
    candidate_c = "ffi.h:30:3: note: candidate: bool convert(DLDataType, int64_t)"
    rejection_c = "ffi.h:30:3: note: no known conversion for argument 1 from ShapeView to DLDataType"
    macro_note = "ffi.h:40:3: note: in expansion of macro TVM_FFI_CHECK"
    repeated_notes = []
    for index in range(40):
        repeated_notes.extend(
            [
                f"template context {index} " + "x" * 80,
                macro_note,
                candidate_a,
                rejection_a,
                candidate_b,
                rejection_b,
                candidate_c,
                rejection_c,
            ]
        )
    compiler_output = "\n".join(
        [
            "generated_binding.cpp:7:5: error: no match for operator==",
            "  if (tensor.shape() == dtype.code) {}",
            "      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~",
            *repeated_notes,
        ]
    )

    compacted = compact_compiler_diagnostics(compiler_output, max_chars=6000)

    assert len(compacted) <= 6000
    assert "error: no match for operator==" in compacted
    assert "tensor.shape() == dtype.code" in compacted
    assert compacted.count(candidate_a) == 1
    assert compacted.count(rejection_a) == 1
    assert compacted.count(candidate_b) == 1
    assert compacted.count(rejection_b) == 1
    assert candidate_c not in compacted
    assert rejection_c not in compacted
    assert macro_note not in compacted
    assert "[omitted 2 additional unique actionable diagnostic notes]" in compacted


def test_oversized_compiler_diagnostics_always_keeps_terminal_exception():
    macro_noise = "\n".join(
        f"ffi.h:{index}:3: note: in definition of macro TVM_FFI_CHECK\n  " + "x" * 120 for index in range(80)
    )
    terminal_exception = "RuntimeError: TVM-FFI compilation worker exited before producing an artifact"
    compiler_output = "\n".join(
        [
            "generated_binding.cpp:7:5: error: invalid conversion",
            "  bind(invalid_value);",
            "       ^~~~~~~~~~~~~",
            macro_noise,
            terminal_exception,
        ]
    )

    compacted = compact_compiler_diagnostics(compiler_output, max_chars=1000)

    assert len(compacted) <= 1000
    assert "error: invalid conversion" in compacted
    assert terminal_exception in compacted
    assert "in definition of macro TVM_FFI_CHECK" not in compacted


def test_sampled_primary_blocks_keep_adjacent_terminal_exception():
    terminal_exception = "RuntimeError: compiler subprocess terminated unexpectedly"
    compiler_output = "\n".join(
        [
            *(f"generated_binding.cpp:{index}:5: error: E{index} " + "e" * 120 for index in range(70)),
            terminal_exception,
        ]
    )

    compacted = compact_compiler_diagnostics(compiler_output, max_chars=600)

    assert len(compacted) <= 600
    assert terminal_exception in compacted
    assert compacted.count(terminal_exception) == 1
    assert "primary diagnostic blocks" in compacted


def test_actionable_notes_never_displace_primary_errors_when_source_context_is_oversized():
    candidate = "ffi.h:10:3: note: candidate: bool operator==(DLDataType, DLDataType)"
    rejection = "ffi.h:10:3: note: no known conversion for argument 2 from DLDataTypeCode"
    blocks = []
    for index in range(25):
        blocks.extend(
            [
                f"generated_binding.cpp:{index}:5: error: E{index} invalid binding expression",
                f"  invalid_binding_expression_{index}(tensor); " + "x" * 140,
                "  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
                f"ffi.h:{index}:3: note: in expansion of macro TVM_FFI_CHECK",
            ]
        )
    compiler_output = "\n".join([*blocks, candidate, rejection])

    compacted = compact_compiler_diagnostics(compiler_output, max_chars=6000)

    assert len(compacted) <= 6000
    for index in range(25):
        assert f"error: E{index} invalid binding expression" in compacted
    assert candidate in compacted
    assert rejection in compacted
    assert "invalid_binding_expression_0(tensor)" in compacted
    assert "^~~~~~~~~~~~~~~~~~~~~~~~~~~~~" in compacted


def test_compiler_diagnostics_unknown_format_uses_strict_conservative_fallback():
    unknown = "UNKNOWN-BEGIN-" + "测🙂x" * 100 + "-UNKNOWN-END"

    compacted = compact_compiler_diagnostics(unknown, max_chars=80)

    assert len(compacted) == 80
    assert compacted.startswith("UNKNOWN-BEGIN-")
    assert compacted.endswith("-UNKNOWN-END")
    assert "...(truncated)..." in compacted


def test_compiler_diagnostics_reports_omitted_primary_blocks_instead_of_silent_middle_loss():
    compiler_output = "\n".join(
        f"generated_binding.cpp:{index}:5: error: E{index}\n  bad_{index}();\n  ^~~~~" for index in range(70)
    )

    compacted = compact_compiler_diagnostics(compiler_output, max_chars=600)

    assert len(compacted) <= 600
    assert "primary diagnostic blocks" in compacted
    assert "omitted" in compacted
    assert "error: E0" in compacted
    assert "error: E69" in compacted


@pytest.mark.parametrize("max_chars", [1, 8, 17, 18, 31, 64])
def test_truncate_middle_counts_marker_and_unicode_inside_strict_budget(max_chars):
    text = "开始🙂" + "abc测🙂" * 40 + "结束"

    truncated = _truncate_middle(text, max_chars)

    assert len(truncated) == max_chars
    assert truncated == text[:max_chars] if max_chars <= len("...(truncated)...") else "...(truncated)..." in truncated
    assert _truncate_middle(text, 0) == text


def test_feedback_template_uses_compacted_text_and_reports_stats(monkeypatch):
    env_result = _rich_env_result()
    original = deepcopy(env_result)
    template = PromptTemplate("TEXT={feedback}", "format", "test")
    stats = {}
    monkeypatch.setitem(generate_with_cuda_agent.CUDA_AGENT_CONFIGS, "max_feedback_chars", 8000)

    rendered = _apply_feedback_template(env_result, template, feedback_stats=stats)

    assert "DECOY_KERNEL_DETECTED" in rendered
    assert "missing_kernel" in rendered
    assert "DROP_ATEN_TRIAL" not in rendered
    assert "DROP_FULL_PROFILE" not in rendered
    serialized = rendered.removeprefix("TEXT=")
    assert serialized == json.dumps(json.loads(serialized), ensure_ascii=False, default=str)
    assert "\n" not in serialized
    assert stats["original_chars"] > stats["compacted_chars"] == stats["final_chars"]
    assert stats["final_truncated"] is False
    assert env_result == original


@pytest.mark.parametrize(
    ("template", "render_mode"),
    [
        ("{feedback_dict}", "format"),
        ("{{ feedback_dict }}", "jinja"),
        ("{feedback}{feedback}", "format"),
        ("{feedback:>500}", "format"),
        ("{{ feedback * 3 }}", "jinja"),
    ],
)
def test_feedback_template_rejects_feedback_dict_budget_bypass(monkeypatch, template, render_mode):
    monkeypatch.setitem(generate_with_cuda_agent.CUDA_AGENT_CONFIGS, "max_feedback_chars", 80)
    response_template = PromptTemplate(template, render_mode, "test")

    with pytest.raises(ValueError, match="feedback"):
        _apply_feedback_template(_rich_env_result(), response_template)


def test_feedback_template_allows_one_plain_jinja_feedback_placeholder(monkeypatch):
    monkeypatch.setitem(generate_with_cuda_agent.CUDA_AGENT_CONFIGS, "max_feedback_chars", 8000)
    template = PromptTemplate("Server feedback: {{ feedback }} after", "jinja", "test")

    rendered = _apply_feedback_template(_rich_env_result(), template)

    assert rendered.startswith("Server feedback: {")
    assert rendered.endswith("} after")


def test_tvm_ffi_feedback_template_rejects_pybind_binding_instructions():
    incompatible = PromptTemplate.from_path(
        str(REPO_ROOT / "examples/kernel_agent/prompt_config/initial_prompt/multi_turn_cuda_kernel.yaml")
    )
    state = SimpleNamespace(
        args=SimpleNamespace(kernel_backend="tvm_ffi"),
        multi_turn_template=incompatible,
    )

    with pytest.raises(ValueError, match="incompatible pybind binding instructions"):
        generate_with_cuda_agent._get_tool_response_template(state)


def test_tvm_ffi_feedback_template_accepts_repository_tvm_ffi_prompt():
    template = PromptTemplate.from_path(
        str(REPO_ROOT / "examples/kernel_agent/prompt_config/multi_turn_tvm_ffi_short.yaml")
    )
    state = SimpleNamespace(
        args=SimpleNamespace(kernel_backend="tvm_ffi"),
        multi_turn_template=template,
    )

    assert generate_with_cuda_agent._get_tool_response_template(state) is template


def test_feedback_template_strictly_applies_small_final_budget(monkeypatch):
    env_result = {
        "env_state": {
            "status": "failed",
            "error": "VALIDATION_ERROR",
            "error_message": "测🙂" * 200,
        }
    }
    template = PromptTemplate("{feedback}", "format", "test")
    stats = {}
    monkeypatch.setitem(generate_with_cuda_agent.CUDA_AGENT_CONFIGS, "max_feedback_chars", 80)

    rendered = _apply_feedback_template(env_result, template, feedback_stats=stats)

    parsed = json.loads(rendered)
    assert len(rendered) <= 80
    assert stats["final_chars"] == len(rendered)
    assert stats["final_truncated"] is True
    assert parsed["_omitted"] > 0


def test_runtime_error_keeps_frames_but_normalizes_machine_paths():
    env_result = {
        "env_state": {
            "status": "completed",
            "error": "KERNEL_EVAL_FAILED",
            "error_message": (
                'File "/dev/shm/kernelgym/compile_cache/hash/pkg/model_new.py", line 44, in forward\n'
                "  bad_call()\nRuntimeError: output must be 3D"
            ),
        }
    }

    feedback = build_model_feedback(env_result)

    message = feedback["error_message"]
    assert 'File ".../generated/model_new.py", line 44, in forward' in message
    assert "bad_call()" in message
    assert "RuntimeError: output must be 3D" in message
    assert "/dev/shm/kernelgym/compile_cache/hash" not in message


def test_normalize_env_feedback_does_not_consume_raw_compile_error():
    raw = {
        "status": "failed",
        "compiled": False,
        "correctness": None,
        "speedup": None,
        "error_message": "Compilation failed.",
        "metadata": {
            "compile_artifact": {
                "error": "generated_binding.cpp:9: error: missing symbol",
                "entry_point": "forward",
            }
        },
    }
    original = deepcopy(raw)

    normalized, _ = normalize_env_feedback(raw)

    assert raw == original
    assert raw["metadata"]["compile_artifact"]["error"] == "generated_binding.cpp:9: error: missing symbol"
    assert "missing symbol" in normalized["error_message"]


def test_build_model_feedback_handles_empty_and_non_json_values(monkeypatch):
    assert build_model_feedback({"env_state": None}) == {}

    env_result = {"status": "failed", "novel_error_detail": {"values": {"a", "b"}}}
    template = PromptTemplate("{feedback}", "format", "test")
    monkeypatch.setitem(generate_with_cuda_agent.CUDA_AGENT_CONFIGS, "max_feedback_chars", 8000)

    rendered = _apply_feedback_template(env_result, template)

    assert "novel_error_detail" not in rendered
    assert "values" not in rendered


def test_build_model_feedback_marks_cyclic_actionable_metadata_without_mutation():
    cyclic = {}
    cyclic["self"] = cyclic
    env_result = {"env_state": {"status": "completed", "metadata": {"correctness_issue": cyclic}}}

    feedback = build_model_feedback(env_result)

    assert feedback["correctness_details"]["correctness_issue"]["self"] == "...(cyclic reference omitted)..."
    assert cyclic["self"] is cyclic


def test_feedback_stats_and_depth_bound_handle_cyclic_deep_metadata(monkeypatch):
    cyclic = {}
    cyclic["self"] = cyclic
    deep = "leaf"
    for _ in range(20):
        deep = {"child": deep}
    env_result = {
        "env_state": {
            "status": "completed",
            "metadata": {"correctness_issue": cyclic, "max_difference": deep},
        }
    }
    template = PromptTemplate("{feedback}", "format", "test")
    stats = {}
    monkeypatch.setitem(generate_with_cuda_agent.CUDA_AGENT_CONFIGS, "max_feedback_chars", 8000)

    rendered = _apply_feedback_template(env_result, template, feedback_stats=stats)

    assert "cyclic reference omitted" in rendered
    assert "nested detail omitted" in rendered
    assert stats["original_chars"] > 0


def test_feedback_budget_reduces_structure_without_producing_invalid_json(monkeypatch):
    env_result = _rich_env_result()
    metadata = env_result["env_state"]["metadata"]
    metadata["custom_kernel_names"] = [f"kernel_{index}_" + "测🙂" * 2000 for index in range(100)]
    metadata["suspected_decoy_reasons"] = [f"warning_{index}_" + "测🙂" * 2000 for index in range(100)]
    template = PromptTemplate("{feedback}", "format", "test")
    stats = {}
    monkeypatch.setitem(generate_with_cuda_agent.CUDA_AGENT_CONFIGS, "max_feedback_chars", 8000)

    rendered = _apply_feedback_template(env_result, template, feedback_stats=stats)
    parsed = json.loads(rendered)

    assert len(rendered) <= 8000
    assert parsed["status"] == "completed"
    assert parsed["error"] == "DECOY_KERNEL_DETECTED"
    assert stats["structured_reduced"] is True
    assert stats["final_truncated"] is True


def test_model_feedback_bounds_large_actionable_lists_with_visible_omission_marker():
    env_result = _rich_env_result()
    metadata = env_result["env_state"]["metadata"]
    metadata["forbidden_aten_ops"] = [{"name": f"aten::forbidden_{index}", "count": index + 1} for index in range(100)]
    metadata["custom_kernel_names"] = [f"kernel_{index}" for index in range(100)]

    feedback = build_model_feedback(env_result)

    forbidden = feedback["policy_details"]["forbidden_aten_ops"]
    custom_kernels = feedback["profiling_summary"]["custom_kernel_names"]
    assert len(forbidden) == 33
    assert forbidden[-1] == {"omitted_items": 68}
    assert len(custom_kernels) == 33
    assert custom_kernels[-1] == "...(68 items omitted)..."


def test_model_feedback_marks_omitted_policy_warnings():
    env_result = _rich_env_result()
    env_result["env_state"]["metadata"]["suspected_decoy_reasons"] = [f"warning_{index}" for index in range(100)]

    feedback = build_model_feedback(env_result)

    warnings = feedback["policy_details"]["policy_warnings"]
    assert len(warnings) == 32
    assert warnings[-1] == "...(70 items omitted)..."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
