import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from slime.backends.megatron_utils.hf_checkpoint_saver import (
    _clear_existing_hf_weights,
    _copy_hf_assets,
    _finalize_local_shards,
    _SafetensorShardWriter,
    _write_pending_chunk,
    save_hf_model_to_path,
)

NUM_GPUS = 0


def test_copy_hf_assets_keeps_quantized_config_and_skips_weights(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    config = {"model_type": "tiny", "quantization_config": {"quant_method": "fp8"}}
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (src / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    (src / "pytorch_model.bin").write_bytes(b"weight")

    _copy_hf_assets(str(src), dst)

    assert json.loads((dst / "config.json").read_text(encoding="utf-8")) == config
    assert (dst / "tokenizer.json").exists()
    assert not (dst / "model.safetensors.index.json").exists()
    assert not (dst / "model-00001-of-00001.safetensors").exists()
    assert not (dst / "pytorch_model.bin").exists()


def test_clear_existing_hf_weights_removes_old_weight_files_only(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weight")

    _clear_existing_hf_weights(tmp_path)

    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "model.safetensors.index.json").exists()
    assert not (tmp_path / "model-00001-of-00001.safetensors").exists()
    assert not (tmp_path / "pytorch_model.bin").exists()


def test_save_hf_model_to_path_rejects_origin_checkpoint(tmp_path: Path):
    args = SimpleNamespace(hf_checkpoint=str(tmp_path))

    with pytest.raises(ValueError, match="same directory as --hf-checkpoint"):
        save_hf_model_to_path(args, tmp_path, model=None)


def test_prepare_hf_directory_on_nonzero_node_writer(tmp_path, monkeypatch):
    import torch.distributed as dist
    from slime.backends.megatron_utils import hf_checkpoint_saver as saver

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    output = tmp_path / "node70" / "hf" / "rollout_9"
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 8)
    monkeypatch.setattr(dist, "get_world_size", lambda: 16)
    broadcasts = []
    monkeypatch.setattr(dist, "broadcast_object_list", lambda payload, src: broadcasts.append(src))
    args = SimpleNamespace(hf_checkpoint=str(source), actor_num_nodes=2, actor_num_gpus_per_node=8)
    saver._prepare_hf_output_directory(args, output)
    assert broadcasts == [0, 8]
    assert (output / "config.json").is_file()
    writer = saver._SafetensorShardWriter(output, enabled=True)
    writer.write([("weight", torch.ones(2, 2))], shard_idx=1)
    assert len(list(output.glob("*.safetensors"))) == 1


def test_prepare_hf_directory_broadcasts_remote_writer_failure(tmp_path, monkeypatch):
    import torch.distributed as dist
    from slime.backends.megatron_utils import hf_checkpoint_saver as saver

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 16)

    def broadcast(payload, src):
        if src == 8:
            payload[0] = "PermissionError: writer path is not writable"

    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    args = SimpleNamespace(hf_checkpoint=str(tmp_path), actor_num_nodes=2, actor_num_gpus_per_node=8)
    with pytest.raises(RuntimeError, match="writer rank 8.*PermissionError"):
        saver._prepare_hf_output_directory(args, tmp_path / "output")


def test_safetensor_shard_writer_writes_hf_index(tmp_path: Path):
    writer = _SafetensorShardWriter(tmp_path, enabled=True)
    writer.write([("layers.0.weight", torch.ones(2, 2)), ("layers.0.weight_scale", torch.ones(1))], shard_idx=0)
    writer.write([("layers.1.weight", torch.zeros(2, 2))], shard_idx=1)
    writer.finalize()

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == 36
    assert index["weight_map"] == {
        "layers.0.weight": "model-00001-of-00002.safetensors",
        "layers.0.weight_scale": "model-00001-of-00002.safetensors",
        "layers.1.weight": "model-00002-of-00002.safetensors",
    }

    shard0 = load_file(tmp_path / "model-00001-of-00002.safetensors")
    shard1 = load_file(tmp_path / "model-00002-of-00002.safetensors")
    assert torch.equal(shard0["layers.0.weight"], torch.ones(2, 2))
    assert torch.equal(shard1["layers.1.weight"], torch.zeros(2, 2))


def test_finalize_shard_files_merges_node_writer_states(tmp_path: Path):
    writer0 = _SafetensorShardWriter(tmp_path, enabled=True)
    writer1 = _SafetensorShardWriter(tmp_path, enabled=True)

    writer0.write([("layers.0.weight", torch.ones(2, 2))], shard_idx=0)
    writer1.write([("layers.1.weight", torch.zeros(2, 2))], shard_idx=1)

    # each rank renames its own files off the shared plan; rank 0 writes the index
    states = [writer0.state(), writer1.state()]
    for rank, state in enumerate(states):
        _finalize_local_shards(tmp_path, state, states, write_index=rank == 0)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == 32
    assert index["weight_map"] == {
        "layers.0.weight": "model-00001-of-00002.safetensors",
        "layers.1.weight": "model-00002-of-00002.safetensors",
    }
    assert not (tmp_path / "model-00001.safetensors").exists()
    assert not (tmp_path / "model-00002.safetensors").exists()

    shard0 = load_file(tmp_path / "model-00001-of-00002.safetensors")
    shard1 = load_file(tmp_path / "model-00002-of-00002.safetensors")
    assert torch.equal(shard0["layers.0.weight"], torch.ones(2, 2))
    assert torch.equal(shard1["layers.1.weight"], torch.zeros(2, 2))


def test_pending_chunk_write_flushes_incomplete_node_group(tmp_path: Path):
    num_nodes = 3
    writers = [_SafetensorShardWriter(tmp_path, enabled=True) for _ in range(num_nodes)]
    pending_writes = [None] * num_nodes

    for chunk_idx in range(5):
        node_rank = chunk_idx % num_nodes
        pending_writes[node_rank] = (
            chunk_idx,
            [(f"layers.{chunk_idx}.weight", torch.full((1,), chunk_idx, dtype=torch.float32))],
        )

        if (chunk_idx + 1) % num_nodes == 0:
            for i, writer in enumerate(writers):
                pending_writes[i] = _write_pending_chunk(writer, pending_writes[i])

    for i, writer in enumerate(writers):
        pending_writes[i] = _write_pending_chunk(writer, pending_writes[i])

    states = [writer.state() for writer in writers]
    for rank, state in enumerate(states):
        _finalize_local_shards(tmp_path, state, states, write_index=rank == 0)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"] == {f"layers.{i}.weight": f"model-{i + 1:05d}-of-00005.safetensors" for i in range(5)}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
