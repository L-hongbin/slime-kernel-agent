from types import SimpleNamespace

import pytest

pytest.importorskip("yaml")
pytest.importorskip("jinja2")
pytest.importorskip("torch")

from slime_plugins.drkernel.rollout import (
    DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS,
    _dedupe_preserve_order,
    _truncate_text,
    build_prompt_feedback_payload,
    format_kernelgym_feedback,
    summarize_diagnostic_text,
)


# ---------------------------------------------------------------------------
# _truncate_text / _dedupe_preserve_order
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_truncate_text_passes_short_strings_through():
    assert _truncate_text("hello", limit=400) == "hello"


@pytest.mark.unit
def test_truncate_text_appends_marker_when_cut():
    text = "x" * 500
    out = _truncate_text(text, limit=100)
    assert out.endswith("...<truncated>")
    assert out.startswith("x" * 100)


@pytest.mark.unit
def test_truncate_text_returns_non_strings_unchanged():
    assert _truncate_text(None) is None
    assert _truncate_text(123) == 123


@pytest.mark.unit
def test_dedupe_preserve_order_keeps_first_occurrence():
    assert _dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# summarize_diagnostic_text priority chain
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_summarize_none_input():
    assert summarize_diagnostic_text(None) is None


@pytest.mark.unit
def test_summarize_blank_input():
    assert summarize_diagnostic_text("   \n\n  ") == ""


@pytest.mark.unit
def test_summarize_picks_precheck_line_first():
    text = (
        "Noise line\n"
        "Precheck failed: TVM-FFI host binding source must keep CUDA runtime headers/types out\n"
        "More noise\n"
    )
    assert summarize_diagnostic_text(text) == (
        "Precheck failed: TVM-FFI host binding source must keep CUDA runtime headers/types out"
    )


@pytest.mark.unit
def test_summarize_picks_syntax_error_line():
    text = "Preamble\nSyntax error in model code: unexpected indent (<string>, line 2)\njunk"
    assert summarize_diagnostic_text(text) == ("Syntax error in model code: unexpected indent (<string>, line 2)")


@pytest.mark.unit
def test_summarize_picks_timeout_line():
    text = "lots of noise\nTask exec timeout after 90s on worker 3\ntrailing"
    assert summarize_diagnostic_text(text) == "Task exec timeout after 90s on worker 3"


@pytest.mark.unit
def test_summarize_picks_attribute_error_phrase():
    text = "Traceback ...\n'NoneType' object has no attribute 'forward' here\n..."
    assert summarize_diagnostic_text(text) == "'NoneType' object has no attribute 'forward'"


@pytest.mark.unit
def test_summarize_picks_cuda_oom_line():
    text = (
        "Traceback (most recent call last):\n"
        "  File 'x.py', line 1\n"
        "CUDA out of memory. Tried to allocate 8.00 GiB. GPU 5 has a total capacity of 23.52 "
        "GiB of which 6.62 GiB is free. Including non-PyTorch memory, this process has 16.44 "
        "GiB memory in use. Process 2165460 has 448.00 MiB memory in use.\n"
        "See documentation for Memory Management ...\n"
        "...lots of noisy follow-up...\n"
    )
    out = summarize_diagnostic_text(text)
    # First OOM line lifted out; "See documentation" trailing noise dropped
    assert out.startswith("CUDA out of memory.")
    assert "8.00 GiB" in out
    assert "See documentation" not in out
    assert "Memory Management" not in out


@pytest.mark.unit
def test_summarize_extracts_first_four_compile_errors_dedup_in_order():
    text = "\n".join(
        [
            "nvcc command line ...",
            "/file.cu(10): error: undeclared identifier `foo`",
            "/file.cu(11): error: undeclared identifier `bar`",
            "/file.cu(10): error: undeclared identifier `foo`",  # dup of first
            "/file.cu(12): error: undeclared identifier `baz`",
            "/file.cu(13): error: undeclared identifier `qux`",
            "/file.cu(14): error: undeclared identifier `quux`",  # 5th distinct => dropped
            "nvcc warning: ignored",
        ]
    )
    out = summarize_diagnostic_text(text, limit=2000)
    assert "undeclared identifier `foo`" in out
    assert "undeclared identifier `bar`" in out
    assert "undeclared identifier `baz`" in out
    assert "undeclared identifier `qux`" in out
    # only 4 lines kept; the 5th distinct error is excluded
    assert "undeclared identifier `quux`" not in out
    assert out.count("undeclared identifier `foo`") == 1  # deduped


@pytest.mark.unit
def test_summarize_extracts_last_three_exception_lines():
    text = "\n".join(
        [
            "Traceback (most recent call last):",
            "  File 'a.py', line 1, ...",
            "ValueError: too early",
            "  File 'b.py', line 1, ...",
            "RuntimeError: middle one",
            "AssertionError: last one",
        ]
    )
    out = summarize_diagnostic_text(text)
    assert "ValueError: too early" in out
    assert "RuntimeError: middle one" in out
    assert "AssertionError: last one" in out


@pytest.mark.unit
def test_summarize_caps_at_limit_when_truncating():
    huge = "\n".join(f"/file.cu({i}): error: foo {i}" for i in range(500))
    out = summarize_diagnostic_text(huge, limit=200)
    assert len(out) <= 200 + len("...<truncated>")


# ---------------------------------------------------------------------------
# build_prompt_feedback_payload field whitelist
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_payload_filters_top_level_whitelist():
    env_state = {
        "task_id": "parallel_task_xxx",
        "status": "completed",
        "compiled": True,
        "correctness": True,
        "decoy_kernel": False,
        "reference_runtime": 5.0,
        "kernel_runtime": 2.5,
        "speedup": 2.0,
        "reward": 1.0,
        "success": True,
        "error_code": None,
    }
    payload = build_prompt_feedback_payload(env_state)
    # Whitelist hits
    for k in (
        "status",
        "compiled",
        "correctness",
        "decoy_kernel",
        "reference_runtime",
        "kernel_runtime",
        "speedup",
    ):
        assert k in payload, f"missing whitelist field {k}"
    # Explicit drops
    for k in ("task_id", "reward", "success", "error_code"):
        assert k not in payload, f"should have dropped {k}"


@pytest.mark.unit
def test_build_payload_metrics_only_when_compiled_and_correct():
    # Field names match real KernelGym /evaluate response schema (plural
    # num_custom_kernels and custom_kernel_cuda_time_coverage).
    metadata = {
        "custom_kernel_cuda_time_coverage": "Custom kernel CUDA time: 218103.85us / Total: 218103.85us, Coverage: 100.00%",
        "num_custom_kernels": 3,
        "num_total_kernels": 12,
        "custom_kernel_cuda_time_in_profiling_us": 1000.0,
        "total_kernel_cuda_time_in_profiling_us": 1050.0,
    }
    # compiled + correct => metrics present, all 5 fields kept
    out_ok = build_prompt_feedback_payload({"compiled": True, "correctness": True, "metadata": metadata})
    assert out_ok.get("metrics") == metadata

    # compile fails => no metrics
    out_fail = build_prompt_feedback_payload({"compiled": False, "correctness": False, "metadata": metadata})
    assert "metrics" not in out_fail

    # compiles but incorrect => no metrics
    out_incorrect = build_prompt_feedback_payload({"compiled": True, "correctness": False, "metadata": metadata})
    assert "metrics" not in out_incorrect


@pytest.mark.unit
def test_build_payload_drops_internal_kg_metadata_fields_and_keeps_correctness_diagnostics():
    env_state = {
        "compiled": True,
        "correctness": False,
        "metadata": {
            "hardware": "NVIDIA RTX 4090",
            "gpu_name": "NVIDIA RTX 4090",  # dropped (redundant)
            "device": "cuda:0",  # dropped
            "backend": "tvm_ffi",  # dropped
            "compilation_error_name": "compile_error",  # kept
            "correctness_issue_name": "Output mismatch",  # kept (diagnostic)
            "max_difference": 0.12,  # kept (diagnostic)
            "avg_difference": 0.003,  # kept (diagnostic)
            "kg_stage_completed_s": {"phase_a": 0.1},  # dropped
            "tm_enter_monotonic_ns": 12345,  # dropped
        },
    }
    out = build_prompt_feedback_payload(env_state)
    md = out.get("metadata", {})
    assert md.get("hardware") == "NVIDIA RTX 4090"
    assert md.get("compilation_error_name") == "compile_error"
    assert md.get("correctness_issue_name") == "Output mismatch"
    assert md.get("max_difference") == 0.12
    assert md.get("avg_difference") == 0.003
    for dropped in ("gpu_name", "device", "backend", "kg_stage_completed_s", "tm_enter_monotonic_ns"):
        assert dropped not in md


@pytest.mark.unit
def test_build_payload_surfaces_perf_oom_via_error_during_performance():
    # Kernel passed correctness but perf-stage OOM'd: env_state has compiled+correct=True,
    # reference_runtime=-1, speedup=0, and the actual OOM string in metadata.error_during_performance.
    env_state = {
        "status": "completed",
        "compiled": True,
        "correctness": True,
        "decoy_kernel": False,
        "reference_runtime": -1.0,
        "kernel_runtime": 599.0,
        "speedup": 0.0,
        "metadata": {
            "hardware": "NVIDIA RTX 4090",
            "error_during_performance": (
                "CUDA out of memory. Tried to allocate 8.00 GiB. GPU 5 has a total capacity of "
                "23.52 GiB of which 6.62 GiB is free. ... See documentation for Memory Management ..."
            ),
        },
    }
    out = build_prompt_feedback_payload(env_state)
    # Summarized OOM ends up in error_message
    assert "CUDA out of memory" in out["error_message"]
    assert "8.00 GiB" in out["error_message"]
    assert "See documentation" not in out["error_message"]
    # The raw stderr field never leaks into payload metadata
    assert "error_during_performance" not in out.get("metadata", {})
    # Other signals preserved
    assert out["compiled"] is True
    assert out["correctness"] is True
    assert out["speedup"] == 0.0
    assert out["reference_runtime"] == -1.0


@pytest.mark.unit
def test_build_payload_metadata_error_is_summarized_as_last_resort():
    # When only metadata.error is present (no compilation_error/runtime_error/
    # correctness_issue/error_message), the raw stderr at metadata.error should
    # still flow through summarize_diagnostic_text into payload.error_message.
    long_stderr = "\n".join(
        [
            "nvcc warning: ignored",
            "/build/foo.cu(10): error: bar",
            "/build/foo.cu(20): error: baz",
        ]
        + ["random noise"] * 50
    )
    env_state = {
        "compiled": False,
        "correctness": False,
        "metadata": {"error": long_stderr},
    }
    out = build_prompt_feedback_payload(env_state, error_summary_chars=1600)
    assert "error" not in out.get("metadata", {})
    assert "/build/foo.cu(10): error: bar" in out["error_message"]
    assert "/build/foo.cu(20): error: baz" in out["error_message"]
    assert "random noise" not in out["error_message"]
    assert "nvcc warning" not in out["error_message"]


@pytest.mark.unit
def test_build_payload_summarizes_metadata_compilation_error_into_error_message():
    long_stderr = "\n".join(
        [
            "/build/foo.cu(10): error: bar",
            "/build/foo.cu(20): error: baz",
        ]
        + ["random noise"] * 50
    )
    env_state = {
        "compiled": False,
        "correctness": False,
        "metadata": {"compilation_error": long_stderr},
    }
    out = build_prompt_feedback_payload(env_state, error_summary_chars=1600)
    # raw nvcc text never lands in payload.metadata
    assert "compilation_error" not in out.get("metadata", {})
    assert "/build/foo.cu(10): error: bar" in out["error_message"]
    assert "/build/foo.cu(20): error: baz" in out["error_message"]
    # noise lines pruned by the compile-error extractor
    assert "random noise" not in out["error_message"]


@pytest.mark.unit
def test_build_payload_returns_state_when_no_whitelisted_field_matches():
    out = build_prompt_feedback_payload({})
    assert out == {}


@pytest.mark.unit
def test_build_payload_honors_error_summary_chars_argument():
    env_state = {
        "metadata": {"compilation_error": "/x.cu(1): error: " + ("zzz" * 200)},
    }
    out = build_prompt_feedback_payload(env_state, error_summary_chars=80)
    assert len(out["error_message"]) <= 80 + len("...<truncated>")


# ---------------------------------------------------------------------------
# format_kernelgym_feedback integration
# ---------------------------------------------------------------------------


def _make_sample(kernelgym_meta):
    return SimpleNamespace(metadata={"kernelgym": kernelgym_meta})


@pytest.mark.unit
def test_format_extract_error_branch_unchanged():
    sample = _make_sample({"extract_error": "missing 3 sections", "reward": 0.0})
    out = format_kernelgym_feedback(sample)
    assert '"status": "extract_error"' in out
    assert "missing 3 sections" in out


@pytest.mark.unit
def test_format_top_level_env_state_shape_summarizes():
    # Production KernelGym /evaluate returns the env_state at the top level of
    # ``response``, not nested under ``response["env_state"]``. The format
    # function must recognize this shape via _looks_like_env_state and route it
    # through build_prompt_feedback_payload.
    response = {
        "task_id": "parallel_task_xxx",
        "status": "failed",
        "compiled": False,
        "correctness": False,
        "decoy_kernel": False,
        "reference_runtime": -1.0,
        "kernel_runtime": -1.0,
        "speedup": 0.0,
        "error_code": "COMPILATION_ERROR",
        "error_message": "Task processing failed: RuntimeError ...",
        "metadata": {
            "error": (
                "nvcc warning: blah\n"
                "/dev/shm/work/x.cu(5): error: undefined identifier `foo`\n"
                "/dev/shm/work/x.cu(7): error: undefined identifier `bar`\n" + ("garbage line\n" * 200)
            ),
            "compilation_error_name": "compile_error",
            "hardware": "NVIDIA RTX 4090",
            "gpu_name": "NVIDIA RTX 4090",
            "backend": "tvm_ffi",
            "kg_stage_completed_s": {"phase": 0.01},
            "tm_enter_monotonic_ns": 12345,
        },
    }
    sample = _make_sample({"response": response})
    fb = format_kernelgym_feedback(sample, error_summary_chars=DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS)
    # summarized error lines came through
    assert "undefined identifier `foo`" in fb
    assert "undefined identifier `bar`" in fb
    # noise + internal kg fields scrubbed
    assert "kg_stage_completed_s" not in fb
    assert "tm_enter_monotonic_ns" not in fb
    assert "garbage line" not in fb
    assert "nvcc warning" not in fb
    # explicit field drops still applied
    assert '"task_id"' not in fb
    assert '"gpu_name"' not in fb
    assert '"backend"' not in fb  # metadata.backend dropped
    # whitelist preserved
    assert '"compiled": false' in fb
    assert '"compilation_error_name": "compile_error"' in fb
    # total size compact
    assert len(fb) < 4000


@pytest.mark.unit
def test_format_nested_env_state_shape_also_summarizes():
    # Some KernelGym variants wrap env_state under response["env_state"]. Both
    # shapes must be supported. This was the only shape my original code
    # recognized; we keep a regression case to ensure backwards compatibility.
    response = {
        "env_state": {
            "compiled": False,
            "correctness": False,
            "decoy_kernel": False,
            "metadata": {
                "error": "/path/x.cu(1): error: undefined identifier `foo`\n",
                "compilation_error_name": "compile_error",
                "hardware": "NVIDIA RTX 4090",
            },
        }
    }
    sample = _make_sample({"response": response})
    fb = format_kernelgym_feedback(sample, error_summary_chars=DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS)
    assert "undefined identifier `foo`" in fb
    # noise + internal kg fields scrubbed
    assert "kg_stage_completed_s" not in fb
    assert "nvcc warning" not in fb
    assert "garbage line" not in fb
    # whitelist + summarized error keep the feedback small
    assert len(fb) < 4000


@pytest.mark.unit
def test_format_full_payload_with_no_env_state_passes_through():
    response = {"reward_extra_info": {"speedup": 1.5, "compiled": True}}
    sample = _make_sample({"response": response})
    fb = format_kernelgym_feedback(sample)
    assert '"speedup": 1.5' in fb
    assert '"compiled": true' in fb


@pytest.mark.unit
def test_format_max_chars_outer_cap_still_applies():
    response = {
        "env_state": {
            "compiled": False,
            "correctness": False,
            "metadata": {"error": "x" * 5000, "hardware": "GPU"},
        }
    }
    sample = _make_sample({"response": response})
    fb = format_kernelgym_feedback(sample, max_chars=200, error_summary_chars=4096)
    assert len(fb) <= 200 + len("...(truncated)...")
