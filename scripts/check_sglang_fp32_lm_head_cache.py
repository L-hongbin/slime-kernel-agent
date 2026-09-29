"""Manual preflight for the version-patched SGLang FP32 LM head cache.

Requires the patched SGLang runtime, so this is not a generic CI test. CPU
checks exercise the actual projection and weight updater. --device cuda also
verifies that an already captured graph sees new weights without recapture.
"""

import argparse
import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers import logits_processor as logits_processor_module
from sglang.srt.layers.logits_processor import LogitsProcessor

try:
    from sglang.srt.layers.fp32_lm_head_cache import get_fp32_lm_head_weight, refresh_fp32_lm_head_cache
except ImportError:
    get_fp32_lm_head_weight = None
    refresh_fp32_lm_head_cache = None

try:
    from sglang.srt.model_executor.model_runner_components import weight_updater
except ImportError:
    from sglang.srt.model_executor import model_runner as weight_updater


class TinyModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lm_head = torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16, device=device)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, value in weights:
            params[name].data.copy_(value)


@torch.no_grad()
def _check_legacy_cache(device):
    torch.manual_seed(17)
    model = TinyModel(device)
    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.use_fp32_lm_head = True
    processor.rl_on_policy_target = None
    hidden = torch.randn(8, 32, dtype=torch.bfloat16, device=device)

    def project():
        return processor._compute_lm_head(hidden, model.lm_head)

    def reference():
        return hidden.float() @ model.lm_head.weight.float().T

    # SGLang warmup can create the cache under inference_mode; the updater
    # must still be able to write it outside that context.
    with torch.inference_mode():
        torch.testing.assert_close(project(), reference(), rtol=0, atol=0)
    cache = get_fp32_lm_head_weight(model.lm_head)
    pointer = cache.data_ptr()
    assert get_fp32_lm_head_weight(model.lm_head) is cache
    assert set(model.state_dict()) == {"lm_head.weight"}
    assert model.lm_head.weight.dtype == torch.bfloat16
    # Target and MTP draft share the same head module in Qwen3.8.
    draft = torch.nn.Module()
    draft.lm_head = model.lm_head
    assert get_fp32_lm_head_weight(draft.lm_head).data_ptr() == pointer

    if hasattr(weight_updater, "WeightUpdater"):
        updater = weight_updater.WeightUpdater(
            tp_rank=0,
            device=device,
            gpu_id=0,
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: SimpleNamespace(model=model),
        )
    else:
        updater = weight_updater.ModelRunner.__new__(weight_updater.ModelRunner)
        updater.model = model
        updater.tp_rank = 0
        updater.device = device
        updater.server_args = SimpleNamespace(custom_weight_loader=[])
        updater._model_update_group = {}
    # Exercise both the standard loader and the direct parameter-copy path.
    for load_format in (None, "direct"):
        new_weight = torch.randn_like(model.lm_head.weight)
        with patch.object(weight_updater, "monkey_patch_torch_reductions"):
            success, message = updater.update_weights_from_tensor([("lm_head.weight", new_weight)], load_format)
        assert success, message
        assert cache.data_ptr() == pointer
        torch.testing.assert_close(cache, new_weight.float(), rtol=0, atol=0)
        torch.testing.assert_close(project(), reference(), rtol=0, atol=0)

    new_weight = torch.randn_like(model.lm_head.weight)
    bucket = weight_updater.FlattenedTensorBucket(named_tensors=[("lm_head.weight", new_weight)])
    with patch.object(weight_updater, "monkey_patch_torch_reductions"):
        success, message = updater.update_weights_from_tensor(
            {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            },
            "flattened_bucket",
        )
    assert success, message
    torch.testing.assert_close(cache, new_weight.float(), rtol=0, atol=0)

    # Keep the real distributed update/load path, replacing only transport.
    updater._model_update_group["check"] = object()
    for load_format in (None, "flattened_bucket"):
        new_weight = torch.randn_like(model.lm_head.weight)

        def broadcast(destination, source_weight=new_weight, **kwargs):
            source = source_weight.flatten().view(torch.uint8) if destination.dtype == torch.uint8 else source_weight
            destination.copy_(source)
            return SimpleNamespace(wait=lambda: None)

        with patch.object(torch.distributed, "broadcast", side_effect=broadcast):
            success, message = updater.update_weights_from_distributed(
                ["lm_head.weight"],
                [new_weight.dtype],
                [new_weight.shape],
                "check",
                load_format,
            )
        assert success, message
        assert cache.data_ptr() == pointer
        torch.testing.assert_close(cache, new_weight.float(), rtol=0, atol=0)
        torch.testing.assert_close(project(), reference(), rtol=0, atol=0)

    # Replacing the Parameter must not replace the captured cache allocation.
    model.lm_head.weight = torch.nn.Parameter(torch.randn_like(model.lm_head.weight))
    refresh_fp32_lm_head_cache(model)
    assert get_fp32_lm_head_weight(model.lm_head).data_ptr() == pointer
    torch.testing.assert_close(project(), reference(), rtol=0, atol=0)

    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                project()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = project()
        graph.replay()
        torch.testing.assert_close(graph_output, reference(), rtol=0, atol=0)
        for _ in range(3):
            # .data updates bypass the Parameter version counter.
            version = model.lm_head.weight._version
            model.lm_head.weight.data.copy_(torch.randn_like(model.lm_head.weight))
            assert model.lm_head.weight._version == version
            refresh_fp32_lm_head_cache(model)
            graph.replay()
            torch.testing.assert_close(graph_output, reference(), rtol=0, atol=0)
            assert cache.data_ptr() == pointer
        torch.cuda.synchronize()

    processor.use_fp32_lm_head = False
    with patch.object(
        logits_processor_module,
        "get_global_server_args",
        return_value=SimpleNamespace(rl_on_policy_target=None),
        create=True,
    ):
        torch.testing.assert_close(project(), hidden @ model.lm_head.weight.T, rtol=0, atol=0)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight[:32].clone())
    try:
        refresh_fp32_lm_head_cache(model)
    except RuntimeError as error:
        assert "layout changed" in str(error)
    else:
        raise AssertionError("Changing the cache layout must fail instead of leaving a stale CUDA graph")
    return {
        "device": device,
        "projection_and_updates": "passed",
        "cuda_graph": "passed" if device == "cuda" else "not run",
    }


@torch.no_grad()
def _check_native_out_dtype(device):
    source = inspect.getsource(LogitsProcessor._compute_lm_head)
    assert "out_dtype=torch.float32" in source.replace(" ", ""), "Missing native FP32-output LM head matmul"

    torch.manual_seed(17)
    model = TinyModel(device)
    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.use_fp32_lm_head = True
    processor.rl_on_policy_target = None
    hidden = torch.randn(8, 32, dtype=torch.bfloat16, device=device)

    def project():
        return processor._compute_lm_head(hidden, model.lm_head)

    output = project()
    reference = hidden.float() @ model.lm_head.weight.float().T
    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)
    assert output.dtype == torch.float32

    new_weight = torch.randn_like(model.lm_head.weight)
    model.lm_head.weight.data.copy_(new_weight)
    output = project()
    reference = hidden.float() @ new_weight.float().T
    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)

    if device == "cuda":
        for _ in range(3):
            project()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = project()
        graph.replay()
        torch.testing.assert_close(graph_output, reference, rtol=2e-3, atol=2e-3)
        model.lm_head.weight.data.copy_(torch.randn_like(model.lm_head.weight))
        graph.replay()
        reference = hidden.float() @ model.lm_head.weight.float().T
        torch.testing.assert_close(graph_output, reference, rtol=2e-3, atol=2e-3)
        torch.cuda.synchronize()

    return {
        "device": device,
        "native_out_dtype_fp32": "passed",
        "weight_update_visibility": "passed",
        "cuda_graph": "passed" if device == "cuda" else "not run",
    }


def check(device):
    if get_fp32_lm_head_weight is None:
        return _check_native_out_dtype(device)
    return _check_legacy_cache(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(check(args.device), indent=2))
