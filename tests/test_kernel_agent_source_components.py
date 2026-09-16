"""CPU contracts for one-turn source component descriptions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.kernel_agent.source_components import SCHEMA, analyze_source_components


def response(cuda: str, native: str | None = None, model: str | None = None) -> str:
    native = native or 'extern "C" void run(float*x){k<<<1,32>>>(x);} TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,run);'
    model = model or "import tvm_ffi_extension\nclass ModelNew:\n def forward(self,x): return tvm_ffi_extension.run(x)"
    return "\n\n".join(
        [
            f"### CUDA_KERNELS\n```cpp\n{cuda}\n{native}\n```",
            "### APPLY_BINDINGS\n```cpp\nvoid binding(){}\n```",
            f"### MODEL_NEW\n```python\n{model}\n```",
        ]
    )


def kernels(result):
    return [unit for unit in result["units"] if unit["kind"] == "kernel"]


def test_reachable_kernel_is_one_multi_feature_unit_and_dead_kernel_is_residual():
    result = analyze_source_components(
        response(
            "__global__ void k(float*x){x[0]=__expf(x[0])+__shfl_down_sync(0xffffffff,x[0],1);}\n"
            "__global__ void dead(float*x){x[0]=1;}"
        )
    )
    assert result["schema"] == SCHEMA
    assert result["wall_seconds"] >= 0
    units = {unit["evidence"]["qualified_name"]: unit for unit in kernels(result)}
    assert len(units) == 2
    assert {"fast_math", "warp_shuffle"} <= set(units["k"]["features"])
    assert units["k"]["use_observed"] is True
    assert units["dead"]["use_observed"] is False
    assert units["dead"]["unknowns"] == ["use_not_observed"]
    assert units["dead"]["signature"] is not None


def test_signature_preserves_rename_constants_helpers_and_fusion_boundaries():
    base = analyze_source_components(
        response(
            "constexpr float SCALE=1.0f; __device__ float h(float x){return x+1;} "
            "__global__ void k(float*x){x[0]=h(x[0])*SCALE;}"
        )
    )
    renamed = analyze_source_components(
        response(
            "__device__ float h(float x){return x+1;} __global__ void renamed(float*x){x[0]=h(x[0]);}",
            native='extern "C" void run(float*x){renamed<<<1,32>>>(x);} TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,run);',
        )
    )
    changed_helper = analyze_source_components(
        response(
            "constexpr float SCALE=1.0f; __device__ float h(float x){return x+2;} "
            "__global__ void k(float*x){x[0]=h(x[0])*SCALE;}"
        )
    )
    changed_constant = analyze_source_components(
        response(
            "constexpr float SCALE=2.0f; __device__ float h(float x){return x+1;} "
            "__global__ void k(float*x){x[0]=h(x[0])*SCALE;}"
        )
    )
    macro_one = analyze_source_components(response("#define SCALE 1\n__global__ void k(float*x){x[0]=x[0]*SCALE;}"))
    macro_two = analyze_source_components(response("#define SCALE 2\n__global__ void k(float*x){x[0]=x[0]*SCALE;}"))
    fused = analyze_source_components(response("__global__ void k(float*x){x[0]=x[0]+1;x[1]=x[1]*2;}"))
    split = analyze_source_components(
        response(
            "__global__ void k(float*x){x[0]=x[0]+1;} __global__ void k2(float*x){x[1]=x[1]*2;}",
            native='extern "C" void run(float*x){k<<<1,32>>>(x);k2<<<1,32>>>(x);} TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,run);',
        )
    )
    assert kernels(base)[0]["signature"] != kernels(renamed)[0]["signature"]
    assert kernels(base)[0]["signature"] != kernels(changed_helper)[0]["signature"]
    assert kernels(base)[0]["signature"] != kernels(changed_constant)[0]["signature"]
    assert kernels(macro_one)[0]["signature"] != kernels(macro_two)[0]["signature"]
    assert len(kernels(fused)) == 1
    assert len(kernels(split)) == 2
    assert kernels(fused)[0]["signature"] not in {unit["signature"] for unit in kernels(split)}


def test_library_compute_call_is_one_unit_and_setter_is_not_a_unit():
    result = analyze_source_components(
        response(
            "",
            native=(
                'extern "C" void run(float*x){cublasSetMathMode(h,CUBLAS_TF32_TENSOR_OP_MATH);'
                "cublasLtMatmulDescSetAttribute(d,CUBLASLT_MATMUL_DESC_EPILOGUE,e,4);"
                "cublasGemmEx(h,a,b,c,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT);}"
                "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,run);"
            ),
        )
    )
    libraries = [unit for unit in result["units"] if unit["kind"] == "library"]
    assert len(libraries) == 1
    assert libraries[0]["use_observed"] is True
    assert "library_offload" in libraries[0]["features"]
    assert "library_math_mode" in libraries[0]["features"]
    assert libraries[0]["evidence"]["target"] == "cublasGemmEx"


def test_profile_marks_unreachable_kernel_used_without_confirming_others():
    result = analyze_source_components(
        response("__global__ void k(float*x){x[0]=1;} __global__ void dead(float*x){x[0]=2;}"),
        profiles=[{"name": "dead(float*)"}],
    )
    units = {unit["evidence"]["qualified_name"]: unit for unit in kernels(result)}
    assert units["k"]["use_observed"] is True  # Source call path.
    assert units["dead"]["use_observed"] is True  # Existing profile metadata only.
    assert units["dead"]["evidence"]["source_path"] is None


def test_invalid_model_does_not_drop_independently_parseable_native_components():
    result = analyze_source_components(
        response(
            "__global__ void k(float*x){x[0]=1;} __global__ void other(float*x){x[1]=2;}",
            model="class ModelNew:\n def forward(",
        )
    )
    assert len(kernels(result)) == 2
    assert "section_syntax_invalid:MODEL_NEW" in result["unknowns"]
    assert all(unit["signature"] is not None and not unit["use_observed"] for unit in kernels(result))


def test_bare_cuda_source_is_retained_only_as_unobserved_residual():
    result = analyze_source_components("__global__ void k(float*x){x[0]=1;}")
    assert len(kernels(result)) == 1
    assert kernels(result)[0]["signature"] is not None
    assert "raw_cuda_source_without_ffi_path" in result["unknowns"]


def test_invalid_kernel_translation_unit_keeps_residual_but_withholds_signature():
    result = analyze_source_components(
        response("__global__ void k(float*x){x[0]=1;} void incomplete("),
    )
    unit = kernels(result)[0]
    assert unit["use_observed"] is False
    assert unit["signature"] is None
    assert "component_compile_context_syntax_invalid" in unit["unknowns"]


def test_ambiguous_local_helper_withholds_signature():
    result = analyze_source_components(
        response(
            "__device__ float h(float x){return x+1;} __device__ int h(int x){return x+1;} "
            "__global__ void k(float*x){x[0]=h(x[0]);}"
        )
    )
    unit = kernels(result)[0]
    assert unit["use_observed"] is True
    assert unit["signature"] is None
    assert "local_helper_resolution_ambiguous_or_uncertain" in unit["unknowns"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
