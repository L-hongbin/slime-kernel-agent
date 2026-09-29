"""Validate the approved B300 inputs and isolated training dependencies."""

import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as parquet


def _rollout_package_version(name: str) -> str:
    normalized_name = name.lower().replace("_", "-")
    rollout_pythonpath = os.environ.get("SLIME_ROLLOUT_PYTHONPATH", "")
    for entry in rollout_pythonpath.split(os.pathsep):
        if not entry:
            continue
        for distribution in importlib.metadata.distributions(path=[entry]):
            distribution_name = distribution.metadata["Name"].lower().replace("_", "-")
            if distribution_name == normalized_name:
                return distribution.version
    return importlib.metadata.version(name)


def _check_sglang_version() -> str:
    version = _rollout_package_version("sglang")
    if version not in {"0.5.15.post1", "0.5.20"}:
        raise RuntimeError(f"Expected SGLang 0.5.15.post1 or 0.5.20, got {version}")
    return version


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_nccl_runtime_alignment() -> None:
    train_torch = (Path(os.environ["SLIME_TRAIN_PACKAGES"]) / "torch").resolve()
    rollout_torch = None
    for entry in os.environ.get("SLIME_ROLLOUT_PYTHONPATH", "").split(os.pathsep):
        candidate = Path(entry) / "torch"
        if entry and candidate.exists():
            rollout_torch = candidate.resolve()
            break
    if rollout_torch is None:
        raise RuntimeError("SLIME_ROLLOUT_PYTHONPATH does not contain a Torch runtime")

    relative_nccl = Path("nvidia/nccl/lib/libnccl.so.2")
    train_nccl = train_torch.parent / relative_nccl
    rollout_nccl = rollout_torch.parent / relative_nccl
    train_digest = _sha256_file(train_nccl)
    rollout_digest = _sha256_file(rollout_nccl)
    if train_digest != rollout_digest:
        raise RuntimeError(
            "Training and rollout NCCL libraries must be byte-identical for online "
            f"weight synchronization: train={train_nccl}, rollout={rollout_nccl}"
        )
    print(f"Verified shared NCCL runtime for online weight sync: {train_digest}")


def main():
    sglang_version = _check_sglang_version()
    if sglang_version == "0.5.20":
        _check_nccl_runtime_alignment()
    root = Path(os.environ["HF_MODEL_PATH"])
    config = json.loads((root / "config.json").read_text())
    text_config = config.get("text_config", config)
    assert not config.get("quantization_config") and not text_config.get("quantization_config")
    assert text_config.get("dtype", text_config.get("torch_dtype")) == "bfloat16"
    assert text_config["mtp_num_hidden_layers"] == 1
    assert text_config["num_attention_heads"] % 4 == 0
    assert text_config["num_key_value_heads"] % 4 == 0
    assert text_config["linear_num_key_heads"] % 4 == 0
    assert text_config["linear_num_key_heads"] // 4 >= 1
    index = json.loads((root / "model.safetensors.index.json").read_text())
    assert all((root / name).is_file() for name in set(index["weight_map"].values()))
    assert any("mtp" in name for name in index["weight_map"])
    data_path = Path(os.environ["RL_DATA"])
    expected_hash = "47c5ad75d127ee3647d99b485202bf7624a8d366c24a184d28f4dd1215357217"
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == expected_hash, "Unexpected approved dataset checksum"
    dataset = parquet.ParquetFile(data_path)
    assert {"prompt", "reward_model", "extra_info"}.issubset(dataset.schema_arrow.names)
    assert dataset.metadata.num_rows == 27525
    load_path = Path(os.environ["TRAIN_LOAD_PATH"]).resolve()
    archive_path = (Path(os.environ["EXP_ROOT"]) / "checkpoints").resolve()
    output_path = Path(os.environ.get("TRAIN_SAVE_PATH", str(archive_path))).resolve()
    iteration = None
    if load_path != root.resolve():
        iteration = int((load_path / "latest_checkpointed_iteration.txt").read_text().strip())
        assert (load_path / f"iter_{iteration:07d}" / ".metadata").is_file()
        print(f"Explicit resume from checkpoint iteration {iteration}: {load_path}")
    # Either local or archived latest is a valid resume source. Never overwrite a
    # newer local/unuploaded checkpoint or an already published NFS checkpoint.
    for checkpoint_root in {output_path, archive_path}:
        tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
        if tracker.exists():
            existing_iteration = int(tracker.read_text().strip())
            if load_path not in {output_path, archive_path} or iteration is None or iteration < existing_iteration:
                raise RuntimeError(
                    "Output already contains checkpoints; select a fresh EXP_ROOT "
                    "or explicitly resume from the newest local/NFS checkpoint"
                )
    megatron = Path(os.environ["SLIME_MEGATRON_LM_PATH"])
    gpt_source = (megatron / "megatron/core/models/gpt/gpt_model.py").read_text()
    assert "mtp_output_weight = mtp_output_weight.detach()" in gpt_source
    assert "weight=mtp_output_weight" in gpt_source
    subprocess.run(
        [
            sys.executable,
            "scripts/patch_megatron_mtp_hidden_detach.py",
            "--check",
            "--path",
            str(megatron / "megatron/core/transformer/multi_token_prediction.py"),
        ],
        check=True,
    )
    training_env = os.environ.copy()
    subprocess.run(
        [
            sys.executable,
            "scripts/patch_megatron_mtp_empty_mask.py",
            "--check",
            "--path",
            str(megatron / "megatron/core/models/gpt/gpt_model.py"),
        ],
        check=True,
    )
    training_env["PYTHONPATH"] = os.environ["SLIME_TRAIN_PACKAGES"] + ":" + os.environ["PYTHONPATH"]
    training_env["LD_LIBRARY_PATH"] = os.environ["SLIME_TRAIN_LD_LIBRARY_PATH"]
    subprocess.run(
        [
            sys.executable,
            "scripts/check_qwen38_b300_attention.py",
            "--check-only",
            "--backend",
            os.environ.get("TRAIN_ATTENTION_BACKEND", "flash"),
        ],
        env=training_env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/patch_flashqla_b300.py",
            "--check",
            "--path",
            str(Path(os.environ["SLIME_TRAIN_PACKAGES"]) / "flash_qla"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import flash_qla, tilelang, tvm_ffi; "
            "assert flash_qla.__version__ == '0.1.2', flash_qla.__version__; "
            "assert tilelang.__version__ == '0.1.9', tilelang.__version__; "
            "assert tvm_ffi.__version__ == '0.1.9', tvm_ffi.__version__; "
            "print('Training FlashQLA:', flash_qla.__file__, 'TileLang:', tilelang.__version__)",
        ],
        env=training_env,
        check=True,
    )
    expected_rollout_tilelang = "0.1.12" if sglang_version == "0.5.20" else "0.1.11"
    assert (
        _rollout_package_version("tilelang") == expected_rollout_tilelang
    ), f"Rollout TileLang must be {expected_rollout_tilelang} for SGLang {sglang_version}"
    subprocess.run(
        [
            sys.executable,
            "scripts/patch_flashqla_cpu_metadata.py",
            "--check",
            "--path",
            str(Path(os.environ["SLIME_TRAIN_PACKAGES"]) / "flash_qla"),
        ],
        check=True,
    )
    assert _rollout_package_version("apache-tvm-ffi") == "0.1.11", "Rollout TVM-FFI must remain unchanged"
    print(f"Verified BF16 model, {dataset.metadata.num_rows} approved prompts, MTP isolation, and separate runtimes")


if __name__ == "__main__":
    main()
