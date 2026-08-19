"""Map operator-signature tokens to low-level coverage families."""

from __future__ import annotations

ACTIVATION = {
    "elu",
    "gelu",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "leaky_relu",
    "leakyrelu",
    "mish",
    "relu",
    "selu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "tanh",
}
MATMUL = {"addmm", "bmm", "dot", "einsum", "linear", "matmul", "mm"}
REDUCTION = {
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "cumprod",
    "cumsum",
    "logsumexp",
    "max",
    "mean",
    "min",
    "norm",
    "prod",
    "sum",
}
NORMALIZATION = {
    "batch_norm",
    "batchnorm1d",
    "batchnorm2d",
    "batchnorm3d",
    "group_norm",
    "groupnorm",
    "instance_norm",
    "instancenorm2d",
    "instancenorm3d",
    "layer_norm",
    "layernorm",
    "normalize",
    "rms_norm",
}
INDEXING = {
    "embedding",
    "embedding_bag",
    "gather",
    "index_add",
    "index_select",
    "masked_fill",
    "narrow",
    "scatter",
    "select",
    "take",
    "take_along_dim",
    "topk",
    "tril",
    "triu",
    "where",
}
SHAPE_LAYOUT = {
    "cat",
    "chunk",
    "contiguous",
    "expand",
    "expand_as",
    "flatten",
    "flip",
    "movedim",
    "permute",
    "repeat",
    "repeat_interleave",
    "reshape",
    "reshape_as",
    "split",
    "squeeze",
    "stack",
    "transpose",
    "unfold",
    "unsqueeze",
    "view",
}


def token_family_cells(token: str) -> list[str]:
    basename = token.lower().rsplit(".", 1)[-1]
    cells: set[str] = set()
    if "conv" in basename:
        cells.add("conv")
    if basename in MATMUL:
        cells.add("matmul_linear")
    if basename in REDUCTION:
        cells.add("reduction")
    if basename in NORMALIZATION:
        cells.add("normalization")
    if basename in ACTIVATION:
        cells.add("activation")
    if "pool" in basename:
        cells.add("pooling")
    if basename in INDEXING or any(value in basename for value in ("gather", "index", "scatter")):
        cells.add("indexing_scatter")
    if basename in SHAPE_LAYOUT:
        cells.add("shape_layout")
    if "loss" in basename or basename in {"cross_entropy", "nll"}:
        cells.add("loss_distance")
    if token == "torch.nn.functional.scaled_dot_product_attention":
        cells.add("scaled_dot_product_attention")
    return sorted(cells)
