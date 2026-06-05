import importlib.util
from types import SimpleNamespace
from pathlib import Path

import pytest


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
