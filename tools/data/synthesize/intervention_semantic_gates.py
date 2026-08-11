#!/usr/bin/env python3
"""Static semantic promotion gates shared by random, dtype, and layout lanes.

These checks are intentionally narrow.  They reject only patterns exposed by
the semantic audit that can be proved from the reference and lane manifest;
warnings remain review evidence and do not replace GPU validation.
"""

from __future__ import annotations

import ast
import collections
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

POLICY_VERSION = "intervention_semantic_promotion_gate_v5"

_NONNEGATIVE_FAMILIES = frozenset({"uniform_01", "poisson_counts", "multinomial_categories"})
_SIGN_SENSITIVE_CALLS = frozenset({"abs", "absolute", "hardtanh", "leakyrelu", "leaky_relu", "relu", "relu_"})
_NORMALIZATION_CALLS = frozenset(
    {
        "batchnorm1d",
        "batchnorm2d",
        "batchnorm3d",
        "batch_norm",
        "groupnorm",
        "group_norm",
        "instancenorm1d",
        "instancenorm2d",
        "instancenorm3d",
        "instance_norm",
        "layernorm",
        "layer_norm",
        "normalize",
        "rmsnorm",
    }
)
_EMBEDDING_SEQUENCE_CALLS = frozenset(
    {
        "attention",
        "gru",
        "grucell",
        "lstm",
        "lstmcell",
        "multi_head_attention_forward",
        "multiheadattention",
        "rnn",
        "rnncell",
        "scaled_dot_product_attention",
    }
)
_CATEGORY_STRUCTURAL_CALLS = frozenset(
    {
        "argsort",
        "cat",
        "chunk",
        "clone",
        "contiguous",
        "dim",
        "expand",
        "expand_as",
        "flatten",
        "flip",
        "fliplr",
        "flipud",
        "gather",
        "index_select",
        "len",
        "masked_select",
        "movedim",
        "narrow",
        "numel",
        "permute",
        "range",
        "repeat",
        "repeat_interleave",
        "reshape",
        "roll",
        "rot90",
        "select",
        "size",
        "split",
        "squeeze",
        "stack",
        "swapaxes",
        "swapdims",
        "topk",
        "transpose",
        "unbind",
        "unique",
        "unsqueeze",
        "view",
        "where",
    }
)
_LAYOUT_VIEW_PRESERVING_CALLS = frozenset(
    {
        "chunk",
        "dim",
        "expand",
        "expand_as",
        "movedim",
        "narrow",
        "numel",
        "permute",
        "select",
        "size",
        "split",
        "squeeze",
        "swapaxes",
        "swapdims",
        "transpose",
        "unbind",
        "unsqueeze",
    }
)
_LAYOUT_MATERIALIZERS = frozenset(
    {
        "cat",
        "clone",
        "contiguous",
        "repeat",
        "repeat_interleave",
        "stack",
    }
)
_SEQUENCE_MODULES = frozenset({"gru", "lstm", "multiheadattention", "rnn"})
_BATCH_NAMES = frozenset({"b", "batch", "batch_size", "bs"})
_SEQUENCE_NAMES = frozenset({"l", "seq", "seq_len", "seq_length", "sequence", "sequence_length"})
_FLOAT_EXACT_INTEGER_MAX = {"bfloat16": 2**8, "float16": 2**11}
_FLOAT_MAX = {"bfloat16": 3.3895313892515355e38, "float16": 65504.0}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _leaf_call_names(node: ast.AST) -> set[str]:
    return {
        _call_name(call.func).rsplit(".", 1)[-1].lower()
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and _call_name(call.func)
    }


def _top_level(
    tree: ast.Module,
    kind: type[ast.AST] | tuple[type[ast.AST], ...],
    name: str,
) -> ast.AST | None:
    return next(
        (node for node in tree.body if isinstance(node, kind) and getattr(node, "name", None) == name),
        None,
    )


def _model_forward(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    model = _top_level(tree, ast.ClassDef, "Model")
    if not isinstance(model, ast.ClassDef):
        return None
    return next(
        (
            node
            for node in model.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
        ),
        None,
    )


def _literal_bool(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return node.value
    return None


def _batch_first(call: ast.Call, module: str) -> bool:
    values = [keyword.value for keyword in call.keywords if keyword.arg == "batch_first"]
    if values:
        return _literal_bool(values[-1]) is True
    positional_index = 8 if module == "multiheadattention" else 4
    return len(call.args) > positional_index and _literal_bool(call.args[positional_index]) is True


def _factory_looks_batch_first(get_inputs: ast.AST) -> bool:
    for call in (node for node in ast.walk(get_inputs) if isinstance(node, ast.Call)):
        name = _call_name(call.func).rsplit(".", 1)[-1].lower()
        if name not in {
            "empty",
            "full",
            "ones",
            "rand",
            "randint",
            "randn",
            "tensor",
            "zeros",
        }:
            continue
        dimensions: Sequence[ast.AST]
        if len(call.args) == 1 and isinstance(call.args[0], (ast.List, ast.Tuple)):
            dimensions = call.args[0].elts
        else:
            dimensions = call.args
        if len(dimensions) < 2:
            continue
        first = dimensions[0].id.lower() if isinstance(dimensions[0], ast.Name) else None
        second = dimensions[1].id.lower() if isinstance(dimensions[1], ast.Name) else None
        if first in _BATCH_NAMES and second in _SEQUENCE_NAMES:
            return True
    return False


def _sequence_axis_reasons(tree: ast.Module) -> list[str]:
    model = _top_level(tree, ast.ClassDef, "Model")
    get_inputs = _top_level(tree, (ast.FunctionDef, ast.AsyncFunctionDef), "get_inputs")
    if not isinstance(model, ast.ClassDef) or not isinstance(get_inputs, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    if not _factory_looks_batch_first(get_inputs):
        return []
    reasons: list[str] = []
    for call in (node for node in ast.walk(model) if isinstance(node, ast.Call)):
        module = _call_name(call.func).rsplit(".", 1)[-1].lower()
        if module in _SEQUENCE_MODULES and not _batch_first(call, module):
            reasons.append(f"parent_batch_sequence_axis_mismatch:{module}")
    return sorted(set(reasons))


class _FirstConsumerAnalyzer:
    """Find the first value-semantic consumer of every forward input.

    Shape/view/copy operations preserve an open path.  Once a value-semantic
    operator consumes that path, its output is treated as a new signal.  This
    avoids claiming that a late ReLU erases input signs when an earlier Linear
    or convolution has already mixed the values.
    """

    def __init__(
        self,
        forward: ast.FunctionDef | ast.AsyncFunctionDef,
        structural_calls: frozenset[str],
    ) -> None:
        self.open_names = {argument.arg for argument in forward.args.args if argument.arg != "self"}
        self.consumers: set[str] = set()
        self.structural_calls = structural_calls

    def expression_is_open(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.open_names
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.Attribute):
            return self.expression_is_open(node.value)
        if isinstance(node, ast.Subscript):
            return self.expression_is_open(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return any(self.expression_is_open(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return any(self.expression_is_open(item) for item in [*node.keys, *node.values] if item is not None)
        if isinstance(node, ast.Call):
            function_receiver_open = isinstance(node.func, ast.Attribute) and self.expression_is_open(node.func.value)
            argument_open = any(self.expression_is_open(item) for item in node.args) or any(
                self.expression_is_open(item.value) for item in node.keywords
            )
            if not function_receiver_open and not argument_open:
                return False
            name = _call_name(node.func).rsplit(".", 1)[-1].lower()
            if name in self.structural_calls:
                return True
            self.consumers.add(name or "unknown_call")
            return False
        if isinstance(
            node,
            (
                ast.BinOp,
                ast.BoolOp,
                ast.Compare,
                ast.IfExp,
                ast.UnaryOp,
            ),
        ):
            if any(self.expression_is_open(child) for child in ast.iter_child_nodes(node)):
                self.consumers.add("input_arithmetic_or_comparison")
            return False
        return any(self.expression_is_open(child) for child in ast.iter_child_nodes(node))

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            is_open = value is not None and self.expression_is_open(value)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if not isinstance(child, ast.Name):
                        continue
                    if is_open:
                        self.open_names.add(child.id)
                    else:
                        self.open_names.discard(child.id)
            return
        if isinstance(node, ast.AugAssign):
            if self.expression_is_open(node.target) or self.expression_is_open(node.value):
                self.consumers.add("input_arithmetic_or_comparison")
            return
        if isinstance(node, ast.If):
            self.expression_is_open(node.test)
            for child in [*node.body, *node.orelse]:
                self.statement(child)
            return
        if isinstance(node, (ast.For, ast.While, ast.With, ast.Try)):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    self.statement(child)
                elif isinstance(child, ast.AST):
                    self.expression_is_open(child)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self.statement(child)
            else:
                self.expression_is_open(child)


def _first_input_consumers(
    forward: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    structural_calls: frozenset[str] = _CATEGORY_STRUCTURAL_CALLS,
) -> set[str]:
    analyzer = _FirstConsumerAnalyzer(forward, structural_calls)
    for statement in forward.body:
        analyzer.statement(statement)
    return analyzer.consumers


def _layout_warnings(tree: ast.Module) -> list[str]:
    forward = _model_forward(tree)
    if forward is None:
        return []
    first_consumers = _first_input_consumers(forward, structural_calls=_LAYOUT_VIEW_PRESERVING_CALLS)
    materializers = sorted(first_consumers & _LAYOUT_MATERIALIZERS)
    non_materializers = first_consumers - _LAYOUT_MATERIALIZERS
    if materializers and not non_materializers:
        return [f"layout_only_reaches_immediate_materializer:{materializers[0]}"]
    if materializers:
        return [f"some_layout_inputs_reach_immediate_materializer:{materializers[0]}"]
    return []


def _random_reasons(tree: ast.Module, family: str) -> tuple[list[str], list[str]]:
    forward = _model_forward(tree)
    if forward is None:
        return ["model_forward_missing"], []
    first_consumers = _first_input_consumers(forward)
    reasons: list[str] = []
    warnings: list[str] = []
    if family == "multinomial_categories":
        # Category values may stay live through containers such as split(),
        # unbind(), list comprehensions, and Python collections.  The local
        # first-consumer dataflow above deliberately stops at those boundaries
        # and is therefore insufficient for promotion: a later tan(), softmax,
        # convolution, or arbitrary module call would still reinterpret the
        # category IDs as continuous values.  Require the complete forward to
        # contain only the small, explicitly structural allowlist.  This is
        # conservative by design; uncertain category use-sites must fallback.
        all_call_consumers: set[str] = set()
        for call in (node for node in ast.walk(forward) if isinstance(node, ast.Call)):
            qualified = _call_name(call.func)
            leaf = qualified.rsplit(".", 1)[-1].lower() if qualified else "unknown_call"
            # A Model attribute can be an arbitrary module even when its
            # attribute name happens to be `stack`, `flatten`, or another
            # structural torch spelling.  Subscripted/dynamic callees are
            # equally unprovable, so neither gets the structural exemption.
            if not qualified or qualified.startswith("self."):
                all_call_consumers.add(f"module_or_dynamic_call:{leaf}")
            elif leaf not in _CATEGORY_STRUCTURAL_CALLS:
                all_call_consumers.add(leaf)
        unsupported = sorted(first_consumers | all_call_consumers)
        if unsupported:
            detail = unsupported[0]
            reasons.append(f"multinomial_category_has_continuous_numeric_consumer:{detail}")
    if family in _NONNEGATIVE_FAMILIES:
        inactive = sorted(first_consumers & _SIGN_SENSITIVE_CALLS)
        if inactive:
            reasons.append(f"nonnegative_support_erases_sign_sensitive_operator:{inactive[0]}")
    normalization = sorted(first_consumers & _NORMALIZATION_CALLS)
    if normalization:
        warnings.append(f"normalization_may_weaken_value_intervention:{normalization[0]}")
    sequence = sorted(first_consumers & _EMBEDDING_SEQUENCE_CALLS)
    if family in _NONNEGATIVE_FAMILIES and sequence:
        warnings.append(f"nonnegative_support_narrows_embedding_or_sequence_domain:{sequence[0]}")
    return sorted(set(reasons)), sorted(set(warnings))


def _scalar_environment(tree: ast.Module) -> dict[str, float]:
    environment: dict[str, float] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _scalar_value(statement.value, environment)
        if value is not None:
            environment[target.id] = value
    return environment


def _scalar_value(node: ast.AST, environment: Mapping[str, float]) -> float | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.Name):
        return environment.get(node.id)
    if isinstance(node, ast.UnaryOp):
        value = _scalar_value(node.operand, environment)
        if value is None:
            return None
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        return None
    if isinstance(node, ast.BinOp):
        left = _scalar_value(node.left, environment)
        right = _scalar_value(node.right, environment)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Pow) and abs(right) <= 64:
                return left**right
        except (ArithmeticError, OverflowError):
            return None
    return None


def _round_bfloat16(value: float) -> float:
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    upper = bits >> 16
    lower = bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return struct.unpack(">f", struct.pack(">I", (upper & 0xFFFF) << 16))[0]


def _round_target(value: float, target: str) -> float:
    if target == "float16":
        try:
            return float(struct.unpack("e", struct.pack("e", value))[0])
        except OverflowError:
            return math.copysign(math.inf, value)
    if target == "bfloat16":
        return _round_bfloat16(value)
    raise ValueError(f"unsupported low-precision target:{target}")


def _call_bound(
    call: ast.Call,
    *,
    keyword: str,
    positional_index: int,
    environment: Mapping[str, float],
) -> float | None:
    matches = [item.value for item in call.keywords if item.arg == keyword]
    if matches:
        return _scalar_value(matches[-1], environment)
    if len(call.args) > positional_index:
        return _scalar_value(call.args[positional_index], environment)
    return None


def _dtype_reasons(tree: ast.Module, manifest: Mapping[str, Any], target: str) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    get_inputs = _top_level(tree, (ast.FunctionDef, ast.AsyncFunctionDef), "get_inputs")
    # Serial random children keep the multinomial draw in a module-level
    # helper and call that helper from get_inputs().  Looking only inside the
    # get_inputs() body therefore misses exactly the composed random -> dtype
    # case this gate is meant to protect.  Search the complete source module;
    # the serial source contract and static replay already prove that this is
    # the bound parent reference rather than an unrelated imported helper.
    if isinstance(get_inputs, (ast.FunctionDef, ast.AsyncFunctionDef)) and "multinomial" in _leaf_call_names(tree):
        exact_max = _FLOAT_EXACT_INTEGER_MAX[target]
        specs = manifest.get("factory_specs_before")
        if isinstance(specs, list):
            category_sizes = [
                item["shape"][-1]
                for item in specs
                if isinstance(item, Mapping)
                and isinstance(item.get("shape"), list)
                and item["shape"]
                and type(item["shape"][-1]) is int
            ]
            if any(size > exact_max + 1 for size in category_sizes):
                reasons.append(f"upstream_multinomial_cardinality_exceeds_{target}_exact_integer_range")

    forward = _model_forward(tree)
    if forward is None:
        return sorted(set([*reasons, "model_forward_missing"])), warnings
    calls = _leaf_call_names(forward)
    if calls & {"argmax", "argmin"}:
        reasons.append("low_precision_extremum_ties_can_change_index_output")
    environment = _scalar_environment(tree)
    domain_sensitive = bool(calls & {"acos", "asin"})
    for call in (node for node in ast.walk(forward) if isinstance(node, ast.Call)):
        name = _call_name(call.func).rsplit(".", 1)[-1].lower()
        minimum: float | None = None
        maximum: float | None = None
        if name in {"clamp", "clip"}:
            minimum = _call_bound(call, keyword="min", positional_index=1, environment=environment)
            maximum = _call_bound(call, keyword="max", positional_index=2, environment=environment)
        elif name == "clamp_min":
            minimum = _call_bound(call, keyword="min", positional_index=1, environment=environment)
        elif name == "clamp_max":
            maximum = _call_bound(call, keyword="max", positional_index=1, environment=environment)
        if maximum is not None and abs(maximum) > _FLOAT_MAX[target]:
            reasons.append(f"clamp_bound_not_representable_in_{target}")
        if domain_sensitive and maximum is not None and 0.0 < maximum < 1.0:
            if _round_target(maximum, target) >= 1.0:
                reasons.append(f"inverse_trig_open_interval_margin_collapses_in_{target}")
        if domain_sensitive and minimum is not None and -1.0 < minimum < 0.0:
            if _round_target(minimum, target) <= -1.0:
                reasons.append(f"inverse_trig_open_interval_margin_collapses_in_{target}")
    if calls & {"cumprod", "prod"}:
        warnings.append("long_product_requires_runtime_cast_equivalence")
    return sorted(set(reasons)), sorted(set(warnings))


def evaluate_semantic_gate(
    method: str,
    *,
    parent_code: str,
    child_code: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one deterministic fail-closed semantic promotion verdict."""

    if method not in {"random", "dtype", "layout"}:
        raise ValueError(f"unsupported intervention method:{method}")
    try:
        parent_tree = ast.parse(parent_code)
        ast.parse(child_code)
    except SyntaxError as exc:
        reasons = [f"semantic_gate_ast_parse_failed:{type(exc).__name__}"]
        warnings: list[str] = []
    else:
        reasons = []
        warnings = _sequence_axis_reasons(parent_tree)
        if method == "random":
            family = str(manifest.get("assigned_target"))
            random_reasons, random_warnings = _random_reasons(parent_tree, family)
            reasons.extend(random_reasons)
            warnings.extend(random_warnings)
        elif method == "dtype":
            target = str(manifest.get("assigned_target"))
            if target not in _FLOAT_EXACT_INTEGER_MAX:
                reasons.append(f"unsupported_dtype_target:{target}")
            else:
                dtype_reasons, dtype_warnings = _dtype_reasons(parent_tree, manifest, target)
                reasons.extend(dtype_reasons)
                warnings.extend(dtype_warnings)
        else:
            warnings.extend(_layout_warnings(parent_tree))
    reasons = sorted(set(reasons))
    warnings = sorted(set(warnings))
    return {
        "policy_version": POLICY_VERSION,
        "method": method,
        "status": "rejected" if reasons else "passed",
        "reasons": reasons,
        "warnings": warnings,
    }


def filter_promotable_candidates(
    method: str,
    *,
    parents: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
    runtime_passed: set[str],
) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Apply the semantic gate only to candidates already passed by GPU checks."""

    if not (len(parents) == len(children) == len(manifests)):
        raise ValueError("semantic gate inputs are not row-aligned")

    def reference(row: Mapping[str, Any]) -> str:
        reward = row.get("reward_model")
        value = reward.get("ground_truth") if isinstance(reward, Mapping) else None
        if not isinstance(value, str):
            raise ValueError("semantic gate row lacks reward_model.ground_truth")
        return value

    verdicts: dict[str, dict[str, Any]] = {}
    promoted: set[str] = set()
    reasons: collections.Counter[str] = collections.Counter()
    warnings: collections.Counter[str] = collections.Counter()
    for parent, child, manifest in zip(parents, children, manifests, strict=True):
        uuid = manifest.get("child_uuid")
        if not isinstance(uuid, str) or uuid not in runtime_passed:
            continue
        verdict = evaluate_semantic_gate(
            method,
            parent_code=reference(parent),
            child_code=reference(child),
            manifest=manifest,
        )
        verdicts[uuid] = verdict
        reasons.update(verdict["reasons"])
        warnings.update(verdict["warnings"])
        if verdict["status"] == "passed":
            promoted.add(uuid)
    if set(verdicts) != runtime_passed:
        raise ValueError("semantic gate did not resolve every runtime-passed child")
    summary = {
        "policy_version": POLICY_VERSION,
        "method": method,
        "runtime_passed_rows": len(runtime_passed),
        "promoted_rows": len(promoted),
        "semantic_rejected_rows": len(runtime_passed - promoted),
        "rejection_reasons": dict(sorted(reasons.items())),
        "accepted_warnings": dict(sorted(warnings.items())),
    }
    return promoted, verdicts, summary
