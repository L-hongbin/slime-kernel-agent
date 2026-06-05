def get_tensor_model_parallel_attrs(param) -> dict:
    tensor_model_parallel = getattr(param, "tensor_model_parallel", False) or getattr(param, "_mcore_tp", False)

    partition_dim = getattr(param, "partition_dim", getattr(param, "_tp_partition_dim", -1))
    if partition_dim == -1 and hasattr(param, "_tp_partition_dim"):
        partition_dim = getattr(param, "_tp_partition_dim")

    parallel_mode = getattr(param, "parallel_mode", None)
    if parallel_mode is None and getattr(param, "_tp_duplicated", False):
        parallel_mode = "duplicated"

    return {
        "tensor_model_parallel": tensor_model_parallel,
        "partition_dim": partition_dim,
        "partition_stride": getattr(param, "partition_stride", 1),
        "parallel_mode": parallel_mode,
    }
