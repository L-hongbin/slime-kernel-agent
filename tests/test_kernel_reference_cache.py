"""--use-reference-cache plumbs KernelGym's use_reference_cache, but only when a
stable uuid is present (KernelGym keys the reference-timing cache by uuid)."""

from types import SimpleNamespace

from examples.kernel_agent.generate_with_cuda_agent import _reference_cache_uuid
from examples.kernel_agent.kernel_response import _build_kernel_eval_payload

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
