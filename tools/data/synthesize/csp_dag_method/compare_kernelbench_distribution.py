#!/usr/bin/env python3
"""Compare CSP-DAG coverage and distributions with official KernelBench tasks.

This audit intentionally measures coverage, never distribution matching.  It
uses compute tokens resolved from the effective ``Model.forward`` AST, then an
explicit basename vocabulary for the ten low-level coverage families.  In
particular, it has no substring rule such as ``"conv" in name`` or
``"pool" in name``: an unknown token is reported as unmapped rather than
silently assigned to a family.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_complexity_features, extract_operator_signature, feature_dict
from tools.data.synthesize.profile_prompt_tvm_distribution import profile_dataset

DEFAULT_KERNELBENCH = (
    _REPO_ROOT / "Data/kernelbench-level1-validation-tvm-v2/train.parquet",
    _REPO_ROOT / "Data/kernelbench-level2-validation-tvm-v2/train.parquet",
    _REPO_ROOT / "Data/kernelbench-level3-validation-tvm-v2/train.parquet",
)
FAMILIES = (
    "activation",
    "conv",
    "indexing_scatter",
    "loss_distance",
    "matmul_linear",
    "normalization",
    "pooling",
    "reduction",
    "shape_layout",
    "scaled_dot_product_attention",
)

# All membership is an equality check on an AST-extracted normalized call
# basename.  New APIs therefore stay visible in ``unmapped_tokens`` for manual
# classification instead of being accepted by an accidental spelling match.
_EXACT = {
    "activation": frozenset(
        {
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
    ),
    "conv": frozenset(
        {
            "conv1d",
            "conv2d",
            "conv3d",
            "convtranspose1d",
            "convtranspose2d",
            "convtranspose3d",
        }
    ),
    "indexing_scatter": frozenset(
        {
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
    ),
    "loss_distance": frozenset(
        {
            "cross_entropy",
            "l1_loss",
            "mse_loss",
            "nll",
            "nll_loss",
            "smooth_l1_loss",
            "tripletmarginloss",
            "triplet_margin_loss",
            "pairwise_distance",
            "cosine_similarity",
        }
    ),
    "matmul_linear": frozenset({"addmm", "bmm", "dot", "einsum", "linear", "matmul", "mm"}),
    "normalization": frozenset(
        {
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
    ),
    "pooling": frozenset(
        {
            "adaptiveavgpool2d",
            "adaptiveavgpool3d",
            "adaptive_avg_pool1d",
            "adaptive_avg_pool2d",
            "adaptive_avg_pool3d",
            "avgpool1d",
            "avgpool2d",
            "avgpool3d",
            "avg_pool1d",
            "avg_pool2d",
            "avg_pool3d",
            "maxpool1d",
            "maxpool2d",
            "maxpool3d",
            "max_pool1d",
            "max_pool2d",
            "max_pool3d",
        }
    ),
    "reduction": frozenset(
        {
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
    ),
    "shape_layout": frozenset(
        {
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
    ),
}
_SDPA = "torch.nn.functional.scaled_dot_product_attention"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "rows": pq.ParquetFile(resolved).metadata.num_rows,
        "sha256": _sha256(resolved),
    }


def _basename(token: str) -> str:
    return token.lower().rsplit(".", 1)[-1]


def token_families(token: str) -> tuple[str, ...]:
    if token == _SDPA:
        return ("scaled_dot_product_attention",)
    # A method on an architecture-owned ``self`` object is not an API spelling
    # claim.  For example, ``self.cls_token.expand`` is positional-embedding
    # plumbing, while ``tensor.expand`` is an actual tensor layout operation.
    # The normalized signature makes that distinction available without
    # guessing from a substring.
    if not token.startswith(("nn.", "tensor.", "torch.")):
        return ()
    basename = _basename(token)
    return tuple(family for family in FAMILIES if basename in _EXACT.get(family, ()))


def _bucket(values: Iterable[int], specs: Sequence[tuple[str, int, int | None]]) -> dict[str, int]:
    values = list(values)
    return {
        name: sum(low <= value and (high is None or value <= high) for value in values) for name, low, high in specs
    }


def _occupied(values: dict[str, int]) -> list[str]:
    return [name for name, count in values.items() if count]


def _rows(paths: Sequence[Path]) -> Iterable[str]:
    for path in paths:
        for item in pq.read_table(path, columns=["reward_model"]).column(0).to_pylist():
            yield item["ground_truth"]


def _profile(paths: Sequence[Path]) -> dict[str, Any]:
    family_rows: collections.Counter[str] = collections.Counter()
    token_rows: collections.Counter[str] = collections.Counter()
    unmapped: collections.Counter[str] = collections.Counter()
    operator_count: list[int] = []
    source_lines: list[int] = []
    code_only_lines: list[int] = []
    forward_calls: list[int] = []
    ast_depth: list[int] = []
    module_count: list[int] = []
    multi_class = 0
    rows = 0

    for code in _rows(paths):
        rows += 1
        signature = extract_operator_signature(code)
        present: set[str] = set()
        for token in signature:
            families = token_families(token)
            if families:
                token_rows[token] += 1
                present.update(families)
            else:
                unmapped[token] += 1
        family_rows.update(present)
        operator_count.append(len(signature))
        source_lines.append(len(code.splitlines()))
        code_only_lines.append(
            sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in code.splitlines())
        )
        features = feature_dict(extract_complexity_features(code))
        forward_calls.append(int(features["forward_call_count"]))
        ast_depth.append(int(features["forward_ast_depth"]))
        module_count.append(int(features["init_nn_constructor_count"]))
        multi_class += int(features["top_level_class_count"] > 1)

    bins = {
        "operator_signature_count": _bucket(
            operator_count, (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10-15", 10, 15), (">=16", 16, None))
        ),
        "code_only_lines": _bucket(
            code_only_lines,
            (("<20", 0, 19), ("20-34", 20, 34), ("35-49", 35, 49), ("50-74", 50, 74), (">=75", 75, None)),
        ),
        "source_lines": _bucket(
            source_lines,
            (("<20", 0, 19), ("20-34", 20, 34), ("35-49", 35, 49), ("50-74", 50, 74), (">=75", 75, None)),
        ),
        "forward_call_count": _bucket(
            forward_calls,
            (("0", 0, 0), ("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10-15", 10, 15), (">=16", 16, None)),
        ),
        "forward_ast_depth": _bucket(ast_depth, (("<=5", 0, 5), ("6-9", 6, 9), ("10-14", 10, 14), (">=15", 15, None))),
        "registered_module_count": _bucket(
            module_count, (("0", 0, 0), ("1-4", 1, 4), ("5-9", 5, 9), (">=10", 10, None))
        ),
    }
    return {
        "rows": rows,
        "family_row_presence": dict(sorted(family_rows.items())),
        "mapped_token_call_count": dict(sorted(token_rows.items())),
        "unmapped_token_call_count": dict(sorted(unmapped.items())),
        "bins": bins,
        "multiple_top_level_classes": multi_class,
    }


def _coverage(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    required_families = sorted(reference["family_row_presence"])
    missing_families = [family for family in required_families if not candidate["family_row_presence"].get(family, 0)]
    # ``forward_call_count`` counts only AST Call nodes.  A valid low-level
    # operator written with Python syntax, such as ``A * s``, can therefore
    # have zero calls while still appearing in ``operator_signature_count``.
    # Keep this implementation detail visible as descriptive evidence, but do
    # not use it as a second, semantically incomplete operator-coverage gate.
    descriptive_bin_axes = ("forward_call_count",)
    gate_bin_axes = tuple(name for name in reference["bins"] if name not in descriptive_bin_axes)
    required_bins = {name: _occupied(reference["bins"][name]) for name in gate_bin_axes}
    missing_bins = {
        name: [bucket for bucket in buckets if not candidate["bins"][name].get(bucket, 0)]
        for name, buckets in required_bins.items()
    }
    missing_bins = {name: buckets for name, buckets in missing_bins.items() if buckets}
    return {
        "criterion": "each occupied KernelBench low-level family and each gated form/complexity bin has at least one CSP base row; counts are descriptive only and are not targets for distribution matching",
        "required_families": required_families,
        "missing_families": missing_families,
        "required_occupied_bins": required_bins,
        "missing_occupied_bins": missing_bins,
        "descriptive_only_bins": {
            name: {
                "kernelbench": reference["bins"][name],
                "csp_base": candidate["bins"][name],
                "reason": "AST Call count excludes syntax operators already covered by operator_signature_count",
            }
            for name in descriptive_bin_axes
        },
        "multi_class_required": False,
        "multi_class_covered": candidate["multiple_top_level_classes"] > 0,
        "multi_class_scope": (
            "descriptive only: all occupied KernelBench multi-class references are Level-3 "
            "architecture/helper compositions, outside this low-level generator; an identity "
            "wrapper is forbidden and does not satisfy semantic multi-class coverage"
        ),
        "passes": not missing_families and not missing_bins,
    }


def _numel_profile(label: str, paths: Sequence[Path]) -> dict[str, Any]:
    profile = profile_dataset(label, paths)["resolved_tensor_numel"]
    return {
        "occurrences": profile["positive_numel_occurrences"],
        "p50": profile["percentiles"]["p50"],
        "p90": profile["percentiles"]["p90"],
        "p99": profile["percentiles"]["p99"],
        "ge_1000000": profile["tails"]["1000000"]["count"],
        "ge_10000000": profile["tails"]["10000000"]["count"],
        "ge_100000000": profile["tails"]["100000000"]["count"],
    }


def build_report(kernelbench: Sequence[Path], base: Path, shape: Path) -> dict[str, Any]:
    kb = _profile(kernelbench)
    csp = _profile((base,))
    script = Path(__file__).resolve()
    complexity = _REPO_ROOT / "tools/data/cleaning/complexity.py"
    return {
        "schema_version": "csp_dag_kernelbench_distribution_comparison_v3",
        "measurement_contract": {
            "intent": "coverage only, not distribution fitting",
            "kernelbench_rows": "official Level 1/2/3 references, 100/100/50 rows",
            "csp_rows": "strict-near-dedup retained base selected.parquet",
            "operator_signature": "input-dependent compute multiset resolved from the effective Model.forward AST",
            "family_classifier": "explicit normalized-token basename vocabulary; no substring or regex membership",
            "code_only_lines": "nonblank physical lines excluding comment-only lines",
            "family_counts": "row presence, multi-label",
            "forward_call_count": "descriptive AST Call-node count; zero-call syntax operators remain visible and this axis is not a coverage gate",
            "multi_class": "descriptive only; semantic helper classes are deferred and identity wrappers are forbidden",
        },
        "source_binding": {
            **{f"kernelbench_level{index}": _source(path) for index, path in enumerate(kernelbench, 1)},
            "csp_base": _source(base),
            "csp_shape": _source(shape),
            "audit_script": {"path": str(script), "sha256": _sha256(script)},
            "complexity_extractor": {"path": str(complexity), "sha256": _sha256(complexity)},
        },
        "kernelbench": kb,
        "csp_base": csp,
        "coverage": _coverage(kb, csp),
        "exact_mapped_token_overlap": len(set(kb["mapped_token_call_count"]) & set(csp["mapped_token_call_count"])),
        "positive_tensor_factory_numel": {
            "kernelbench": _numel_profile("KernelBench", kernelbench),
            "csp_base": _numel_profile("CSP base", (base,)),
            "csp_shape": _numel_profile("CSP shape", (shape,)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernelbench", type=Path, nargs=3, default=DEFAULT_KERNELBENCH)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--shape", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.kernelbench, args.base, args.shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": _sha256(args.output),
                "coverage_passes": report["coverage"]["passes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
