"""AWQ W4A16 producer for Qwen3.5/3.6 hybrid checkpoints.

Initial scope is intentionally narrow: text-side SwiGLU MLP only. Self-attn,
linear_attn/Mamba, embeddings, lm_head, vision/audio towers, and MTP are left
BF16. MTP tensors are restored after the HF save so EAGLE can still load, but
the compressed-tensors config ignores them because the public HF model class
does not instantiate the extra ``mtp.*`` modules during AWQ calibration.

Example:
    .venv_llmcompressor/bin/python scripts/quantize/producers/awq_w4a16.py \\
        --model-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B \\
        --calibration-path /tmp/calib_ultrachat_200k_train_sft_256_qwen36.jsonl \\
        --output-path checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-mlp \\
        --multimodal
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASE_IGNORE_PATTERNS = [
    "lm_head",
    "re:.*\\.visual\\..*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*",
    "re:.*\\.audio_tower\\..*",
]

MLP_TARGET_RE = r"re:.*\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)$"
AWQ_PRESERVED_SHARD_NAME = "model-awq-preserved.safetensors"
AWQ_DEFAULT_DUO_SCALING = "both"
AWQ_DEFAULT_N_GRID = 40
MLP_AWQ_MAPPINGS = [
    (
        r"re:model\.language_model\.layers\.\d+\.post_attention_layernorm$",
        [
            r"re:model\.language_model\.layers\.\d+\.mlp\.gate_proj$",
            r"re:model\.language_model\.layers\.\d+\.mlp\.up_proj$",
        ],
    ),
    (
        r"re:model\.language_model\.layers\.\d+\.mlp\.up_proj$",
        [r"re:model\.language_model\.layers\.\d+\.mlp\.down_proj$"],
    ),
]


def awq_targets_and_ignore() -> tuple[list[str], list[str]]:
    """Return SGLang-safe QuantizationModifier target/ignore lists.

    ``gate_up_proj`` is not a saved HF tensor name, but SGLang fuses
    gate/up into a runtime module with that name and checks it against
    ``targets`` while loading compressed-tensors checkpoints.
    """

    ignore = list(BASE_IGNORE_PATTERNS)
    ignore.extend(
        [
            r"re:.*\.self_attn\..*",
            r"re:.*\.linear_attn\..*",
            r"re:.*mtp\..*",
        ]
    )
    return [MLP_TARGET_RE], ignore


def build_awq_mappings_for_target() -> list[tuple[str, list[str]]]:
    """Return raw mapping tuples without importing llmcompressor."""

    return list(MLP_AWQ_MAPPINGS)


def awq_modifier_scope() -> tuple[list[str], list[str]]:
    """Return AWQModifier's quantization fields.

    Do not pass the final MLP-only target/ignore list to ``AWQModifier`` with
    ``scheme=None``. In llmcompressor, that is treated as a partial quantization
    config and attaches ``quantization_scheme.weights=None`` before AWQ's
    duo-scaling validation. The smoothing scope is controlled by
    ``MLP_AWQ_MAPPINGS``; the persisted compressed-tensors scope is controlled by
    the following ``QuantizationModifier``.
    """

    return ["Linear"], []


def should_restore_awq_preserved_tensor(name: str) -> bool:
    """Whether an existing output tensor must stay byte-equivalent to source.

    AWQ MLP smoothing is allowed to change ``post_attention_layernorm`` because
    those scales are paired with the quantized MLP weights. Everything else
    that remains BF16 should stay identical to the source checkpoint. This
    catches llmcompressor side effects such as offset-norm round trips on
    attention norms or input layernorms.
    """

    if name.startswith("mtp."):
        return False
    if ".mlp." in name:
        return False
    if name.endswith(".post_attention_layernorm.weight"):
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, type=Path, help="Source BF16 HF checkpoint directory")
    parser.add_argument("--calibration-path", required=True, type=Path, help="JSONL rows with a 'text' field")
    parser.add_argument("--output-path", required=True, type=Path, help="Destination W4A16 AWQ checkpoint")
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--scheme",
        default="W4A16",
        choices=["W4A16", "W4A16_ASYM"],
        help="LLM Compressor quantization scheme. Current docs use W4A16; older AWQ examples use W4A16_ASYM.",
    )
    parser.add_argument("--force", action="store_true", help="Remove an existing output directory first")
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Load via AutoModelForImageTextToText so the architecture tag stays compatible with SGLang.",
    )
    parser.add_argument(
        "--no-restore-mtp",
        action="store_true",
        help="Do not restore source mtp.* tensors after save. Only use for no-EAGLE load tests.",
    )
    parser.add_argument(
        "--awq-duo-scaling",
        default=AWQ_DEFAULT_DUO_SCALING,
        choices=["false", "true", "both"],
        help=(
            "AWQ scale search mode. 'both' keeps the activation-only grid point "
            "with scale=1 while also testing duo-scaling."
        ),
    )
    parser.add_argument(
        "--awq-n-grid",
        type=int,
        default=AWQ_DEFAULT_N_GRID,
        help=(
            "AWQ grid points. With --awq-duo-scaling=both, llmcompressor splits "
            "this between activation-only and duo-scaling searches."
        ),
    )
    return parser.parse_args()


def load_model(model_path: Path, multimodal: bool):
    patch_transformers_broken_torchvision()
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if multimodal:
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[awq] AutoModelForImageTextToText failed ({e!r}); falling back to AutoModel", flush=True)
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    return model, tokenizer


def load_calibration(path: Path, limit: int) -> list[str]:
    texts: list[str] = []
    domain_markers = 0
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if "problem_id" in row or row.get("source") in {"drkernel", "kernelbench"}:
                domain_markers += 1
            texts.append(row["text"])
            if len(texts) >= limit:
                break
    if not texts:
        raise ValueError(f"no calibration rows found in {path}")
    if domain_markers:
        print(
            "[awq] WARNING: calibration appears to come from DrKernel/KernelBench eval dumps; "
            "do not use it for general-quality claims.",
            flush=True,
        )
    return texts


def patch_llmcompressor_transformers5() -> None:
    import torch
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "TORCH_INIT_FUNCTIONS"):
        return
    names = (
        "uniform_",
        "normal_",
        "xavier_uniform_",
        "xavier_normal_",
        "kaiming_uniform_",
        "kaiming_normal_",
        "orthogonal_",
    )
    modeling_utils.TORCH_INIT_FUNCTIONS = {
        name: getattr(torch.nn.init, name) for name in names if hasattr(torch.nn.init, name)
    }


def patch_transformers5_no_split_modules(model) -> None:
    if hasattr(model, "_get_no_split_modules"):
        return

    def _get_no_split_modules(device_map=None):
        return list(getattr(model, "_no_split_modules", []) or [])

    model._get_no_split_modules = _get_no_split_modules


def patch_transformers_broken_torchvision() -> None:
    """Avoid importing a system torchvision build mismatched with venv torch.

    On .16, torchvision is visible via system site-packages but its custom
    ops do not match the venv torch build, so importing transformers'
    vision helpers raises ``operator torchvision::nms does not exist``.
    AWQ here is text-only, so treating torchvision as unavailable is the
    least invasive fix.
    """

    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    import_utils.is_torchvision_available = lambda: False
    transformers_utils.is_torchvision_available = lambda: False


def _import_awq_classes():
    patch_transformers_broken_torchvision()
    from llmcompressor.modifiers.quantization import QuantizationModifier

    try:
        from llmcompressor.modifiers.transform.awq import AWQModifier
        from llmcompressor.modifiers.transform.awq.mappings import AWQMapping
    except Exception:
        from llmcompressor.modifiers.awq import AWQModifier
        from llmcompressor.modifiers.awq.mappings import AWQMapping

    from scripts.quantize.patches.llmcompressor_qwen3_5_awq import patch_awq_qwen3_5_rmsnorm

    patch_awq_qwen3_5_rmsnorm()
    return AWQModifier, AWQMapping, QuantizationModifier


def parse_awq_duo_scaling(value: str) -> bool | str:
    if value == "false":
        return False
    if value == "true":
        return True
    if value == "both":
        return "both"
    raise ValueError(f"unsupported awq duo scaling mode: {value!r}")


def awq_search_includes_identity(duo_scaling: bool | str) -> bool:
    """Whether the grid includes the exact no-smoothing scale vector.

    llmcompressor's duo-scaling=True formula uses ``1 / w_mean`` at ratio=0,
    so that mode does not test scale=1. Activation-only mode does.
    """

    return duo_scaling is False or duo_scaling == "both"


def build_recipe(
    scheme: str,
    *,
    duo_scaling: bool | str = AWQ_DEFAULT_DUO_SCALING,
    n_grid: int = AWQ_DEFAULT_N_GRID,
):
    AWQModifier, AWQMapping, QuantizationModifier = _import_awq_classes()
    targets, ignore = awq_targets_and_ignore()
    mappings = [AWQMapping(smooth, balance) for smooth, balance in build_awq_mappings_for_target()]
    if not awq_search_includes_identity(duo_scaling):
        print(
            "[awq] WARNING: duo_scaling=True does not test the identity/no-smoothing "
            "baseline; prefer --awq-duo-scaling=both for production checkpoints.",
            flush=True,
        )
    return [
        AWQModifier(
            mappings=mappings,
            duo_scaling=duo_scaling,
            n_grid=n_grid,
        ),
        QuantizationModifier(targets=targets, scheme=scheme, ignore=ignore),
    ]


def patch_quantization_config_for_sglang(output_path: Path) -> None:
    config_path = output_path / "config.json"
    cfg = json.loads(config_path.read_text())
    qcfg: dict[str, Any] | None = cfg.get("quantization_config")
    if not qcfg:
        raise AssertionError(f"{config_path} has no quantization_config after AWQ save")

    targets, ignore = awq_targets_and_ignore()
    groups = qcfg.get("config_groups") or {}
    if not groups:
        raise AssertionError(f"{config_path} quantization_config has no config_groups")
    for group in groups.values():
        group["targets"] = targets

    merged_ignore = list(qcfg.get("ignore") or [])
    for item in ignore:
        if item not in merged_ignore:
            merged_ignore.append(item)
    qcfg["ignore"] = merged_ignore
    qcfg["version"] = qcfg.get("version") or "local-awq-w4a16"
    cfg["quantization_config"] = qcfg
    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def copy_aux_files(model_path: Path, output_path: Path) -> None:
    for fname in (
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)


def assert_w4a16_config(output_path: Path) -> None:
    cfg = json.loads((output_path / "config.json").read_text())
    qcfg = cfg.get("quantization_config") or {}
    text = json.dumps(qcfg, sort_keys=True)
    if "4" not in text or "compressed-tensors" not in text:
        raise AssertionError(f"{output_path} does not look like a compressed-tensors W4 checkpoint")
    _, ignore = awq_targets_and_ignore()
    missing = [item for item in ignore if item not in qcfg.get("ignore", [])]
    if missing:
        raise AssertionError(f"{output_path} quantization_config missing ignore entries: {missing}")


def restore_awq_preserved_tensors(
    output_path: Path,
    source_path: Path,
    max_elements: int = 50_000_000,
) -> int:
    """Restore BF16 tensors outside the AWQ MLP transform from the source ckpt."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from scripts.quantize.utils.mtp_checkpoint import _load_weight_map, _recalculate_total_size

    source_map = _load_weight_map(source_path)
    output_map = _load_weight_map(output_path)
    tensors: dict[str, Any] = {}
    for name in sorted(output_map):
        if name not in source_map or not should_restore_awq_preserved_tensor(name):
            continue
        with safe_open(output_path / output_map[name], framework="pt", device="cpu") as output_handle:
            tensor_slice = output_handle.get_slice(name)
            numel = 1
            for dim in tensor_slice.get_shape():
                numel *= dim
            if numel > max_elements:
                continue
            current = output_handle.get_tensor(name)
        with safe_open(source_path / source_map[name], framework="pt", device="cpu") as source_handle:
            source = source_handle.get_tensor(name)
        if not torch.equal(current, source):
            tensors[name] = source.contiguous()

    if not tensors:
        return 0

    save_file(tensors, output_path / AWQ_PRESERVED_SHARD_NAME)
    output_map = {name: shard for name, shard in output_map.items() if name not in tensors}
    for name in tensors:
        output_map[name] = AWQ_PRESERVED_SHARD_NAME

    total_size = _recalculate_total_size(output_path, output_map)
    (output_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": output_map}, indent=2, sort_keys=True) + "\n"
    )
    return len(tensors)


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_path} already exists; pass --force to overwrite")
        shutil.rmtree(args.output_path)

    print(f"[awq] loading model from {args.model_path} (multimodal={args.multimodal})", flush=True)
    model, tokenizer = load_model(args.model_path, args.multimodal)
    patch_transformers5_no_split_modules(model)
    print(f"[awq] model loaded, root dtype={next(model.parameters()).dtype}", flush=True)

    calibration_texts = load_calibration(args.calibration_path, args.num_calibration_samples)
    print(
        f"[awq] using {len(calibration_texts)} calibration prompts, max_seq={args.max_seq_length}, scheme={args.scheme}",
        flush=True,
    )

    from datasets import Dataset

    calibration_ds = Dataset.from_dict({"text": calibration_texts})
    patch_llmcompressor_transformers5()

    from llmcompressor import oneshot

    duo_scaling = parse_awq_duo_scaling(args.awq_duo_scaling)
    recipe = build_recipe(args.scheme, duo_scaling=duo_scaling, n_grid=args.awq_n_grid)
    print(f"[awq] recipe={recipe}", flush=True)
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=calibration_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calibration_texts),
        output_dir=None,
    )

    args.output_path.mkdir(parents=True, exist_ok=True)
    print(f"[awq] saving compressed checkpoint to {args.output_path}", flush=True)
    model.save_pretrained(args.output_path, save_compressed=True)
    tokenizer.save_pretrained(args.output_path)
    copy_aux_files(args.model_path, args.output_path)
    patch_quantization_config_for_sglang(args.output_path)
    restored_preserved = restore_awq_preserved_tensors(args.output_path, args.model_path)
    print(f"[awq] restored {restored_preserved} non-target BF16 tensor(s) from source", flush=True)

    if not args.no_restore_mtp:
        from scripts.quantize.utils.mtp_checkpoint import inject_mtp_tensors

        injected = inject_mtp_tensors(args.output_path, args.model_path)
        print(f"[awq] restored {injected} BF16 MTP tensor(s); MTP is ignored by W4A16 config", flush=True)

    assert_w4a16_config(args.output_path)
    print(f"[awq] done. W4A16 AWQ checkpoint at {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
