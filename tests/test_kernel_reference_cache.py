"""Kernel-eval request metadata plumbing tests.

``--use-reference-cache`` is forwarded only when a stable uuid is present,
because KernelGym keys its reference-timing cache by uuid.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_AGENT_ROOT = REPO_ROOT / "examples" / "kernel_agent"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(KERNEL_AGENT_ROOT))

import generate_with_cuda_agent as cuda_agent
from generate_with_cuda_agent import _reference_cache_uuid, _resolve_task_precision
from kernel_response import _build_kernel_eval_payload
from slime.utils.types import Sample

NUM_GPUS = 0

CONFIG = {"num_correct_trials": 5, "num_perf_trials": 100, "kernel_eval_task_timeout": 300}


def _payload(uuid=None):
    return {
        "response": "### CUDA_KERNELS\n```cpp\n```\n",
        "ground_truth": "import torch\n",
        "kernel_code": "x",
        "kernel_backend": "tvm_ffi",
        "entry_point": "Model",
        "uuid": uuid,
    }


def test_reference_cache_on_with_uuid():
    args = SimpleNamespace(use_reference_cache=True)
    tp = _build_kernel_eval_payload(args, _payload(uuid="problem_1"), CONFIG)
    assert tp["uuid"] == "problem_1"
    assert tp.get("use_reference_cache") is True


def test_reference_cache_skipped_without_uuid():
    args = SimpleNamespace(use_reference_cache=True)
    tp = _build_kernel_eval_payload(args, _payload(uuid=None), CONFIG)
    # No uuid -> cannot key the reference cache; must not request it.
    assert "use_reference_cache" not in tp


def test_reference_cache_off_by_default():
    args = SimpleNamespace(use_reference_cache=False)
    tp = _build_kernel_eval_payload(args, _payload(uuid="problem_1"), CONFIG)
    assert "use_reference_cache" not in tp


def test_uuid_is_reference_derived_and_collision_safe():
    # Key depends ONLY on the reference identity: same reference -> same key
    # (correct sharing), different reference -> different key (no false-share).
    a = _reference_cache_uuid("import torch\nclass Model: ...", "Model")
    a2 = _reference_cache_uuid("import torch\nclass Model: ...", "Model")
    b = _reference_cache_uuid("import torch\nclass Other: ...", "Model")
    assert a == a2 and a.startswith("ref_")
    assert a != b
    # entry_point is part of the identity.
    assert _reference_cache_uuid("import torch\nclass Model: ...", "Mish") != a


def test_uuid_ignores_dataset_id():
    # No dataset-supplied id can override the reference-derived key, so two
    # datasets that reuse "1" for different references cannot collide.
    k1 = _reference_cache_uuid("reference A", "Model")
    k2 = _reference_cache_uuid("reference B", "Model")
    assert k1 != k2 and k1.startswith("ref_") and k2.startswith("ref_")


def test_uuid_none_without_reference():
    assert _reference_cache_uuid(None, "Model") is None
    assert _reference_cache_uuid("", "Model") is None


def test_uuid_handles_non_str_reference():
    assert _reference_cache_uuid({"code": "x"}, "Model").startswith("ref_")


@pytest.mark.parametrize(
    ("dtype_after", "expected"),
    [("float16", "fp16"), ("bfloat16", "bf16"), ("torch.float32", "fp32")],
)
def test_task_precision_uses_dtype_augmentation_metadata(dtype_after, expected):
    sample = Sample(metadata={"augmentation": {"dtype_after": dtype_after}})
    assert _resolve_task_precision(sample, "def get_inputs():\n    return []\n") == expected


@pytest.mark.parametrize(
    ("torch_dtype", "expected"),
    [("float16", "fp16"), ("bfloat16", "bf16")],
)
def test_task_precision_recovers_dtype_for_serial_layout_child(torch_dtype, expected):
    sample = Sample(metadata={"augmentation": {"intervention_kind": "layout", "dtype_after": None}})
    reference = f"""
import torch

def get_inputs():
    return [torch.randn(8, 8, dtype=torch.{torch_dtype})]
"""
    assert _resolve_task_precision(sample, reference) == expected


def test_task_precision_does_not_infer_from_model_internal_cast():
    sample = Sample(metadata={})
    reference = """
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.float16)

def get_inputs():
    return [torch.randn(8, 8)]
"""
    assert _resolve_task_precision(sample, reference) == "fp32"


def test_kernel_eval_payload_includes_precision():
    payload = _payload(uuid="problem_1")
    payload["precision"] = "bf16"
    task_payload = _build_kernel_eval_payload(SimpleNamespace(), payload, CONFIG)
    assert task_payload["precision"] == "bf16"


def test_cuda_kernel_env_sends_resolved_precision(monkeypatch):
    reference = """
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return x

def get_inputs():
    return [torch.randn(8, dtype=torch.bfloat16)]
"""
    sample = Sample(
        label={"ground_truth": reference},
        metadata={"augmentation": {"intervention_kind": "layout", "dtype_after": None}},
    )

    class PayloadCaptured(Exception):
        pass

    async def capture_payload(_args, _sample, payload, _config):
        assert payload["precision"] == "bf16"
        raise PayloadCaptured

    monkeypatch.setattr(cuda_agent, "run_kernel_eval", capture_payload)
    args = SimpleNamespace(do_precheck=False, kernel_backend="tvm_ffi", reference_backend="torch")

    with pytest.raises(PayloadCaptured):
        asyncio.run(cuda_agent.cuda_kernel_env(args, sample, "response", 0))


if __name__ == "__main__":
    pytest.main([__file__])
