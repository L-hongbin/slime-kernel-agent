_COLUMN_PARALLEL_PARAM_SUFFIXES = (
    "embedding.word_embeddings.weight",
    "output_layer.weight",
    "self_attention.linear_qkv.weight",
    "self_attention.linear_qkv.bias",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc1.bias",
    # MTP (Multi-Token Prediction) eh_proj: ColumnParallelLinear sharded on the
    # output dim (e.g. mtp.layers.0.eh_proj.weight -> SGLang mtp.fc.weight).
    "eh_proj.weight",
)

_ROW_PARALLEL_PARAM_SUFFIXES = (
    "self_attention.linear_proj.weight",
    "mlp.linear_fc2.weight",
)


def _infer_tensor_model_parallel_attrs_from_name(name: str | None) -> dict | None:
    if name is None:
        return None

    if name.endswith(_COLUMN_PARALLEL_PARAM_SUFFIXES):
        return {
            "tensor_model_parallel": True,
            "partition_dim": 0,
            "partition_stride": 1,
            "parallel_mode": None,
        }

    if name.endswith(_ROW_PARALLEL_PARAM_SUFFIXES):
        return {
            "tensor_model_parallel": True,
            "partition_dim": 1,
            "partition_stride": 1,
            "parallel_mode": None,
        }

    return None


def get_tensor_model_parallel_attrs(param, name: str | None = None) -> dict:
    inferred_attrs = _infer_tensor_model_parallel_attrs_from_name(name)

    tensor_model_parallel = getattr(param, "tensor_model_parallel", False) or getattr(param, "_mcore_tp", False)
    if not tensor_model_parallel and inferred_attrs is not None:
        tensor_model_parallel = inferred_attrs["tensor_model_parallel"]

    partition_dim = getattr(param, "partition_dim", getattr(param, "_tp_partition_dim", -1))
    if partition_dim == -1 and hasattr(param, "_tp_partition_dim"):
        partition_dim = getattr(param, "_tp_partition_dim")
    if partition_dim == -1 and inferred_attrs is not None:
        partition_dim = inferred_attrs["partition_dim"]

    parallel_mode = getattr(param, "parallel_mode", None)
    if parallel_mode is None and getattr(param, "_tp_duplicated", False):
        parallel_mode = "duplicated"

    return {
        "tensor_model_parallel": tensor_model_parallel,
        "partition_dim": partition_dim,
        "partition_stride": getattr(
            param,
            "partition_stride",
            inferred_attrs["partition_stride"] if inferred_attrs is not None else 1,
        ),
        "parallel_mode": parallel_mode,
    }
