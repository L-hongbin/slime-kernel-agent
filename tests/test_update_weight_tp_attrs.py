import importlib.util
from types import SimpleNamespace
from pathlib import Path

import pytest

NUM_GPUS = 0


def load_tp_attrs_module():
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/update_weight/tp_attrs.py"
    spec = importlib.util.spec_from_file_location("tp_attrs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_tensor_model_parallel_attrs = load_tp_attrs_module().get_tensor_model_parallel_attrs


@pytest.mark.unit
def test_tensor_model_parallel_attrs_default_to_non_parallel():
    attrs = get_tensor_model_parallel_attrs(SimpleNamespace())

    assert attrs == {
        "tensor_model_parallel": False,
        "partition_dim": -1,
        "partition_stride": 1,
        "parallel_mode": None,
    }


@pytest.mark.unit
def test_tensor_model_parallel_attrs_read_legacy_megatron_attrs():
    param = SimpleNamespace(
        tensor_model_parallel=True,
        partition_dim=0,
        partition_stride=1,
        parallel_mode=None,
    )

    attrs = get_tensor_model_parallel_attrs(param)

    assert attrs == {
        "tensor_model_parallel": True,
        "partition_dim": 0,
        "partition_stride": 1,
        "parallel_mode": None,
    }


@pytest.mark.unit
def test_tensor_model_parallel_attrs_read_mcore_attrs():
    param = SimpleNamespace(_mcore_tp=True, _tp_partition_dim=1)

    attrs = get_tensor_model_parallel_attrs(param)

    assert attrs == {
        "tensor_model_parallel": True,
        "partition_dim": 1,
        "partition_stride": 1,
        "parallel_mode": None,
    }


@pytest.mark.unit
def test_tensor_model_parallel_attrs_prefer_mcore_partition_when_legacy_dim_is_default():
    param = SimpleNamespace(_mcore_tp=True, _tp_partition_dim=0, partition_dim=-1)

    attrs = get_tensor_model_parallel_attrs(param)

    assert attrs["tensor_model_parallel"] is True
    assert attrs["partition_dim"] == 0


@pytest.mark.unit
def test_tensor_model_parallel_attrs_map_mcore_duplicate_to_parallel_mode():
    param = SimpleNamespace(_mcore_tp=True, _tp_duplicated=True)

    attrs = get_tensor_model_parallel_attrs(param)

    assert attrs["tensor_model_parallel"] is True
    assert attrs["parallel_mode"] == "duplicated"


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "module.module.embedding.word_embeddings.weight",
        "module.module.output_layer.weight",
        "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
        "module.module.decoder.layers.0.self_attention.linear_qkv.bias",
        "module.module.decoder.layers.0.mlp.linear_fc1.weight",
        "module.module.decoder.layers.0.mlp.linear_fc1.bias",
    ],
)
def test_tensor_model_parallel_attrs_infer_column_parallel_from_megatron_name(name):
    attrs = get_tensor_model_parallel_attrs(SimpleNamespace(), name=name)

    assert attrs == {
        "tensor_model_parallel": True,
        "partition_dim": 0,
        "partition_stride": 1,
        "parallel_mode": None,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "module.module.decoder.layers.0.self_attention.linear_proj.weight",
        "module.module.decoder.layers.0.mlp.linear_fc2.weight",
    ],
)
def test_tensor_model_parallel_attrs_infer_row_parallel_from_megatron_name(name):
    attrs = get_tensor_model_parallel_attrs(SimpleNamespace(), name=name)

    assert attrs == {
        "tensor_model_parallel": True,
        "partition_dim": 1,
        "partition_stride": 1,
        "parallel_mode": None,
    }


@pytest.mark.unit
def test_tensor_model_parallel_attrs_do_not_infer_layernorm_params_from_megatron_name():
    attrs = get_tensor_model_parallel_attrs(
        SimpleNamespace(),
        name="module.module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
    )

    assert attrs == {
        "tensor_model_parallel": False,
        "partition_dim": -1,
        "partition_stride": 1,
        "parallel_mode": None,
    }


@pytest.mark.unit
def test_tensor_model_parallel_attrs_use_name_to_fill_missing_mcore_partition_dim():
    param = SimpleNamespace(_mcore_tp=True)

    attrs = get_tensor_model_parallel_attrs(
        param,
        name="module.module.decoder.layers.0.self_attention.linear_qkv.weight",
    )

    assert attrs["tensor_model_parallel"] is True
    assert attrs["partition_dim"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
