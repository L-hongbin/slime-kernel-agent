"""Minimal SGLang Engine load/generate smoke test for quantized checkpoints."""

from __future__ import annotations

import argparse
import importlib.machinery
import sys
import types


def _module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


def _raise_vision_unavailable(*_args, **_kwargs):
    raise RuntimeError("torchvision is unavailable in this text-only smoke environment")


def patch_broken_torchvision_import(*, force_stub: bool = False) -> None:
    """Let text-only smoke run when system torchvision mismatches venv torch."""

    if not force_stub:
        try:
            import torchvision.transforms  # noqa: F401
            from torchvision.io import decode_jpeg  # noqa: F401
            from torchvision.transforms.functional import InterpolationMode  # noqa: F401

            return
        except Exception:
            for name in list(sys.modules):
                if name == "torchvision" or name.startswith("torchvision."):
                    sys.modules.pop(name, None)

    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils

        import_utils.is_torchvision_available = lambda: False
        transformers_utils.is_torchvision_available = lambda: False
    except Exception:
        pass

    torchvision = _module("torchvision")
    torchvision.__path__ = []

    torchvision_io = _module("torchvision.io")
    torchvision_transforms = _module("torchvision.transforms")
    torchvision_functional = _module("torchvision.transforms.functional")
    torchvision_v2 = _module("torchvision.transforms.v2")
    torchvision_v2_functional = _module("torchvision.transforms.v2.functional")

    class InterpolationMode:
        NEAREST_EXACT = "nearest_exact"
        BOX = "box"
        BILINEAR = "bilinear"
        HAMMING = "hamming"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    class ImageReadMode:
        UNCHANGED = "unchanged"
        GRAY = "gray"
        RGB = "rgb"
        RGB_ALPHA = "rgb_alpha"

    class _UnavailableTransform:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, *_args, **_kwargs):
            return _raise_vision_unavailable()

    for name in ("Compose", "Lambda", "Resize", "ToTensor", "Normalize"):
        setattr(torchvision_transforms, name, _UnavailableTransform)
    torchvision_transforms.InterpolationMode = InterpolationMode

    torchvision_functional.InterpolationMode = InterpolationMode
    torchvision_functional.resize = _raise_vision_unavailable
    torchvision_functional.pil_to_tensor = _raise_vision_unavailable
    torchvision_functional.to_pil_image = _raise_vision_unavailable

    torchvision_io.ImageReadMode = ImageReadMode
    torchvision_io.decode_image = _raise_vision_unavailable
    torchvision_io.decode_jpeg = _raise_vision_unavailable

    torchvision_v2.functional = torchvision_v2_functional
    torchvision_v2_functional.resize = _raise_vision_unavailable
    torchvision_v2_functional.to_image = _raise_vision_unavailable
    torchvision_v2_functional.to_dtype = _raise_vision_unavailable

    torchvision.transforms = torchvision_transforms
    torchvision.io = torchvision_io
    torchvision_transforms.functional = torchvision_functional
    torchvision_transforms.v2 = torchvision_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--prompt", default="Write a one-line CUDA kernel name:")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eagle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_broken_torchvision_import()
    import sglang as sgl

    engine_kwargs = {
        "model_path": args.model_path,
        "trust_remote_code": True,
        "tp_size": args.tp_size,
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction_static,
        "disable_cuda_graph": True,
        "disable_custom_all_reduce": True,
        "mamba_scheduler_strategy": "extra_buffer",
    }
    if args.eagle:
        engine_kwargs.update(
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 3,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 4,
            }
        )

    print(f"[sgl-smoke] loading {args.model_path}", flush=True)
    llm = sgl.Engine(**engine_kwargs)
    try:
        outputs = llm.generate(
            [args.prompt],
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
            },
        )
        print(f"[sgl-smoke] generated: {outputs[0]['text']}", flush=True)
    finally:
        llm.shutdown()
        print("[sgl-smoke] shutdown", flush=True)


if __name__ == "__main__":
    main()
