"""Static contract analysis and lossless reference canonicalization."""

from __future__ import annotations

import ast
import collections
import copy
import hashlib
import re
from dataclasses import dataclass
from typing import Any

_RANDOM_CALL_RE = re.compile(
    r"(?:^|\.)(?:rand|randn|randint|randperm|rand_like|randn_like|normal|"
    r"bernoulli|multinomial|poisson|gumbel_softmax)(?:$|\.)",
    re.IGNORECASE,
)
_UNSAFE_IMPORTS = frozenset({"requests", "shutil", "socket", "subprocess", "urllib"})
_UNSAFE_CALLS = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "exit",
        "open",
        "os._exit",
        "os.execl",
        "os.execv",
        "os.fork",
        "os.popen",
        "os.remove",
        "os.rename",
        "os.system",
        "os.unlink",
        "quit",
        "shutil.rmtree",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.run",
        "tempfile.mkstemp",
        "tempfile.NamedTemporaryFile",
        "tempfile.SpooledTemporaryFile",
        "tempfile.TemporaryDirectory",
        "tempfile.TemporaryFile",
    }
)
_TORCH_SERIALIZATION_CALLS = frozenset({"torch.load", "torch.save", "torch.jit.load", "torch.jit.save"})
_EXECUTION_MODE_CALLS = frozenset(
    {
        "torch.compiler.is_compiling",
        "torch._dynamo.is_compiling",
        "torch.jit.is_scripting",
        "torch.jit.is_tracing",
    }
)


@dataclass(frozen=True)
class StaticAnalysis:
    fatal_reasons: tuple[str, ...]
    flags: tuple[str, ...]
    semantic_hash: str | None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _filesystem_serialization_calls(tree: ast.AST) -> list[tuple[str, int | None]]:
    """Find Torch serialization calls that may access a real filesystem.

    ``torch.save``/``torch.load`` against ``io.BytesIO`` are useful in-memory
    tensor transformations and do not escape the worker. Any other file
    operand is treated as filesystem I/O because references run as untrusted
    programs and CUDA optimization tasks should be self-contained.
    """

    memory_buffers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _call_name(value.func) not in {"BytesIO", "io.BytesIO"}:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        memory_buffers.update(name for target in targets for name in [_call_name(target)] if name)

    findings: list[tuple[str, int | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name not in _TORCH_SERIALIZATION_CALLS:
            continue
        file_position = 1 if name.endswith("save") else 0
        file_arg = (
            node.args[file_position]
            if len(node.args) > file_position
            else next(
                (keyword.value for keyword in node.keywords if keyword.arg in {"f", "file"}),
                None,
            )
        )
        is_memory = (
            isinstance(file_arg, (ast.Name, ast.Attribute))
            and _call_name(file_arg) in memory_buffers
            or isinstance(file_arg, ast.Call)
            and _call_name(file_arg.func) in {"BytesIO", "io.BytesIO"}
        )
        if not is_memory:
            findings.append((name, getattr(node, "lineno", None)))
    return findings


def _is_distribution_constructor(name: str) -> bool:
    """Return whether a random-looking call merely constructs a distribution."""

    short = name.rsplit(".", 1)[-1]
    return name.startswith("torch.distributions.") or (
        bool(short) and short[0].isupper() and short.lower() in {"bernoulli", "multinomial", "normal", "poisson"}
    )


def _forward_rng_context(
    forward: ast.FunctionDef | ast.AsyncFunctionDef,
    self_literals: dict[str, Any],
) -> tuple[dict[str, int], list[int], set[str]]:
    """Collect explicit seed sites and local distribution bindings."""

    seeded_generators: dict[str, int] = {}
    global_seed_lines: list[int] = []
    distributions: set[str] = set()

    def record_seed(call: ast.Call) -> None:
        name = _call_name(call.func)
        if name in {"torch.manual_seed", "random.seed", "np.random.seed", "numpy.random.seed"}:
            global_seed_lines.append(getattr(call, "lineno", -1))
        elif isinstance(call.func, ast.Attribute) and call.func.attr == "manual_seed":
            owner = _call_name(call.func.value)
            if owner:
                seeded_generators[owner] = getattr(call, "lineno", -1)

    # Only unconditional top-level seed statements are trusted by default.
    # A nested branch may not execute for the delivered constructor inputs.
    for statement in forward.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            record_seed(statement.value)
            continue
        if not isinstance(statement, ast.If) or not isinstance(statement.test, ast.Compare):
            continue
        test = statement.test
        if (
            len(test.ops) != 1
            or not isinstance(test.ops[0], ast.IsNot)
            or len(test.comparators) != 1
            or not isinstance(test.comparators[0], ast.Constant)
            or test.comparators[0].value is not None
            or _resolved_literal(test.left, self_literals) is None
        ):
            continue
        for nested in statement.body:
            if isinstance(nested, ast.Expr) and isinstance(nested.value, ast.Call):
                record_seed(nested.value)

    for node in ast.walk(forward):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not _is_distribution_constructor(_call_name(value.func)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        distributions.update(_call_name(target) for target in targets if _call_name(target))
    return seeded_generators, global_seed_lines, distributions


def _forward_random_call_kind(
    node: ast.Call,
    name: str,
    *,
    seeded_generators: dict[str, int],
    global_seed_lines: list[int],
    distributions: set[str],
) -> str | None:
    """Classify a forward RNG call as ``random`` or explicitly ``seeded``."""

    if _is_distribution_constructor(name):
        return None
    receiver = _call_name(node.func.value) if isinstance(node.func, ast.Attribute) else ""
    receiver_is_distribution = receiver in distributions or (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and _is_distribution_constructor(_call_name(node.func.value.func))
    )
    is_distribution_sample = (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"sample", "rsample", "sample_n"}
        and receiver_is_distribution
    )
    if not (_RANDOM_CALL_RE.search(name) or is_distribution_sample):
        return None

    line = getattr(node, "lineno", -1)
    generator = next((keyword.value for keyword in node.keywords if keyword.arg == "generator"), None)
    generator_name = _call_name(generator) if generator is not None else ""
    if generator_name and seeded_generators.get(generator_name, line + 1) < line:
        return "seeded"
    if any(seed_line < line for seed_line in global_seed_lines):
        return "seeded"
    return "random"


def _fractional_pool_uses_rng(
    node: ast.Call,
    name: str,
    bindings: dict[str, tuple[str, ast.AST | None]],
) -> bool:
    """Return whether a fractional max-pool call must create random samples."""

    direct = name.rsplit(".", 1)[-1] in {
        "fractional_max_pool2d",
        "fractional_max_pool2d_with_indices",
        "fractional_max_pool3d",
        "fractional_max_pool3d_with_indices",
    }
    constructor: ast.Call | None = None
    if name.startswith("self."):
        attribute = name.split(".", 2)[1]
        binding_name, binding_node = bindings.get(attribute, ("", None))
        if binding_name.rsplit(".", 1)[-1] in {"FractionalMaxPool2d", "FractionalMaxPool3d"}:
            constructor = binding_node if isinstance(binding_node, ast.Call) else None
    if not direct and constructor is None:
        return False

    owner = node if direct else constructor
    position = 5 if direct else 4
    random_samples = next(
        (keyword.value for keyword in owner.keywords if keyword.arg == "_random_samples"),
        owner.args[position] if len(owner.args) > position else None,
    )
    return random_samples is None or (isinstance(random_samples, ast.Constant) and random_samples.value is None)


def _target_mutates_name(target: ast.AST, name: str) -> bool:
    # ``x = f(x)`` merely rebinds the local and is the dominant pattern in
    # reference models.  Only writes through an attribute/subscript can mutate
    # the caller-owned object.
    return isinstance(target, (ast.Attribute, ast.Subscript)) and any(
        isinstance(node, ast.Name) and node.id == name for node in ast.walk(target)
    )


def _uses_random_init_scalar(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Find random calls that choose task structure rather than tensor values."""

    calls: list[str] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name.startswith("random."):
            calls.append(name)
            continue
        if name in {"np.random.choice", "numpy.random.choice"}:
            if len(node.args) < 2 and not any(keyword.arg == "size" for keyword in node.keywords):
                calls.append(name)
            continue
        if name in {"np.random.randint", "numpy.random.randint"}:
            if len(node.args) < 3 and not any(keyword.arg == "size" for keyword in node.keywords):
                calls.append(name)
            continue
        # ``torch`` random tensors are valid model parameters.  Converting one
        # immediately to a Python scalar instead chooses a different operation,
        # dimension, or shape on each execution of the same task text.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "item":
            source = node.func.value
            if isinstance(source, ast.Call):
                source_name = _call_name(source.func)
                if _RANDOM_CALL_RE.search(source_name):
                    calls.append(source_name)
    return sorted(set(calls))


def _deterministic_random_scalar(node: ast.Call) -> tuple[ast.AST, str] | None:
    """Return a stable representative for a supported random scalar expression."""

    name = _call_name(node.func)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "item":
        source = node.func.value
        if not isinstance(source, ast.Call):
            return None
        source_name = _call_name(source.func)
        if not _RANDOM_CALL_RE.search(source_name):
            return None
        if source_name.endswith("randint"):
            low = next((keyword.value for keyword in source.keywords if keyword.arg == "low"), None)
            if low is None and len(source.args) >= 3:
                low = source.args[0]
            replacement = copy.deepcopy(low) if low is not None else ast.Constant(value=0)
        elif source_name.endswith("rand"):
            replacement = ast.Constant(value=0.5)
        else:
            # A nonzero normal representative avoids turning multipliers and
            # scales into degenerate constant-output tasks.
            replacement = ast.Constant(value=1.0)
        return ast.copy_location(replacement, node), source_name

    if name == "random.choice" and node.args:
        choices = node.args[0]
        if isinstance(choices, (ast.List, ast.Tuple)) and choices.elts:
            return ast.copy_location(copy.deepcopy(choices.elts[0]), node), name
        if isinstance(choices, ast.Call) and _call_name(choices.func) == "range":
            start = choices.args[0] if len(choices.args) > 1 else ast.Constant(value=0)
            return ast.copy_location(copy.deepcopy(start), node), name

    if name in {"np.random.choice", "numpy.random.choice"}:
        # Current corpus uses a symmetric dimension range, for which zero is a
        # valid and stable representative.  Unsupported forms remain random and
        # are rejected by the normal static rule.
        if node.args and isinstance(node.args[0], ast.Call) and _call_name(node.args[0].func) == "range":
            return ast.copy_location(ast.Constant(value=0), node), name

    if name in {"np.random.randint", "numpy.random.randint"}:
        low = next((keyword.value for keyword in node.keywords if keyword.arg == "low"), None)
        if low is None and len(node.args) >= 2:
            low = node.args[0]
        replacement = copy.deepcopy(low) if low is not None else ast.Constant(value=0)
        return ast.copy_location(replacement, node), name
    return None


class _RandomInitScalarCanonicalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802 - ast API
        node = self.generic_visit(node)
        replacement = _deterministic_random_scalar(node)
        if replacement is None:
            return node
        value, name = replacement
        self.calls.append(name)
        return value


def canonicalize_reference(code: Any, entry_point: Any) -> tuple[Any, tuple[str, ...]]:
    """Apply lossless Python shadowing and stable scalar repairs before audit.

    Python binds the last top-level class/function and the last class method.
    Removing shadowed contract definitions makes that effective program
    explicit.  Supported random constructor scalars are materialized as fixed
    representatives so one task text has one constructor contract.
    """

    if not isinstance(code, str) or not isinstance(entry_point, str):
        return code, ()
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code, ()

    flags: list[str] = []
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry_point]
    if len(classes) > 1:
        effective = classes[-1]
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.ClassDef) and node.name == entry_point and node is not effective)
        ]
        flags.append("canonicalized_duplicate_entry_point_definition")
    elif classes:
        effective = classes[0]
    else:
        effective = None

    for function_name in ("get_inputs", "get_init_inputs"):
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
        ]
        if len(functions) > 1:
            last = functions[-1]
            tree.body = [
                node
                for node in tree.body
                if not (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == function_name
                    and node is not last
                )
            ]
            flags.append(f"canonicalized_duplicate_{function_name}_definition")

    if effective is not None:
        forwards = [
            node
            for node in effective.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
        ]
        if len(forwards) > 1:
            last = forwards[-1]
            effective.body = [
                node
                for node in effective.body
                if not (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "forward"
                    and node is not last
                )
            ]
            flags.append("canonicalized_duplicate_forward_definition")

    init_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_init_inputs"
    ]
    if init_functions:
        scalar_canonicalizer = _RandomInitScalarCanonicalizer()
        scalar_canonicalizer.visit(init_functions[-1])
        flags.extend(f"canonicalized_random_init_scalar:{name}" for name in scalar_canonicalizer.calls)

    if not flags:
        return code, ()
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).rstrip() + "\n", tuple(sorted(set(flags)))


def _expr_value_dependency(node: ast.AST | None, values: dict[str, bool]) -> bool | None:
    """Conservatively track whether an expression depends on input values."""

    if node is None or isinstance(node, ast.Constant):
        return False
    if isinstance(node, ast.Name):
        return values.get(node.id, False)
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return values.get(f"self.{node.attr}", False)
        if node.attr in {"device", "dtype", "layout"}:
            return False
        return _expr_value_dependency(node.value, values)
    if isinstance(node, ast.Subscript):
        dependencies = [
            _expr_value_dependency(node.value, values),
            _expr_value_dependency(node.slice, values),
        ]
        if any(value is True for value in dependencies):
            return True
        return None if any(value is None for value in dependencies) else False
    if isinstance(node, ast.Starred):
        return _expr_value_dependency(node.value, values)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        dependencies = [_expr_value_dependency(item, values) for item in node.elts]
    elif isinstance(node, ast.Dict):
        dependencies = [
            _expr_value_dependency(item, values) for item in [*node.keys, *node.values] if item is not None
        ]
    elif isinstance(node, (ast.BinOp, ast.Compare)):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and ast.dump(node.left) == ast.dump(node.right)
        ):
            return False
        if (
            isinstance(node, ast.Compare)
            and len(node.comparators) == 1
            and ast.dump(node.left) == ast.dump(node.comparators[0])
        ):
            return False
        dependencies = [_expr_value_dependency(child, values) for child in ast.iter_child_nodes(node)]
    elif isinstance(node, (ast.BoolOp, ast.UnaryOp, ast.IfExp, ast.Slice)):
        dependencies = [_expr_value_dependency(child, values) for child in ast.iter_child_nodes(node)]
    elif isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name in {"len"}:
            return _expr_value_dependency(node.args[0], values) if node.args else False
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"dim", "numel", "size", "stride"}:
            return _expr_value_dependency(node.func.value, values)
        if name in {"torch.ones_like", "torch.zeros_like", "torch.empty_like"}:
            return False
        if name == "torch.full_like":
            fill = (
                node.args[1]
                if len(node.args) > 1
                else next((keyword.value for keyword in node.keywords if keyword.arg == "fill_value"), None)
            )
            return _expr_value_dependency(fill, values)
        if (
            name in {"torch.equal", "torch.dist"}
            and len(node.args) >= 2
            and ast.dump(node.args[0]) == ast.dump(node.args[1])
        ):
            return False
        dependencies = [_expr_value_dependency(argument, values) for argument in node.args]
        dependencies.extend(_expr_value_dependency(keyword.value, values) for keyword in node.keywords)
        if isinstance(node.func, ast.Attribute) and not (
            isinstance(node.func.value, ast.Name) and node.func.value.id in {"F", "nn", "torch"}
        ):
            dependencies.append(_expr_value_dependency(node.func.value, values))
    elif isinstance(
        node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda, ast.Await, ast.Yield)
    ):
        return None
    else:
        dependencies = [_expr_value_dependency(child, values) for child in ast.iter_child_nodes(node)]
    if any(value is True for value in dependencies):
        return True
    if any(value is None for value in dependencies):
        return None
    return False


def _assign_dependency(target: ast.AST, dependency: bool | None, values: dict[str, bool]) -> bool:
    if dependency is None:
        return False
    if isinstance(target, ast.Name):
        values[target.id] = dependency
        return True
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
        values[f"self.{target.attr}"] = dependency
        return True
    if isinstance(target, (ast.Tuple, ast.List)):
        return all(_assign_dependency(item, dependency, values) for item in target.elts)
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        index_dependency = _expr_value_dependency(target.slice, values)
        if index_dependency is None:
            return False
        values[target.value.id] = bool(values.get(target.value.id, False) or dependency or index_dependency)
        return True
    # Other attribute/subscript writes do not rebind a tracked local.
    return isinstance(target, (ast.Attribute, ast.Subscript))


def _forward_output_dependency(forward: ast.FunctionDef | ast.AsyncFunctionDef, named_args: list[str]) -> str:
    """Return ``dependent``, ``independent``, or ``unknown`` for straight-line code."""

    values = {name: True for name in named_args}
    returns: list[bool | None] = []
    for statement in forward.body:
        if isinstance(statement, ast.Assign):
            dependency = _expr_value_dependency(statement.value, values)
            if not all(_assign_dependency(target, dependency, values) for target in statement.targets):
                return "unknown"
        elif isinstance(statement, ast.AnnAssign):
            if not _assign_dependency(statement.target, _expr_value_dependency(statement.value, values), values):
                return "unknown"
        elif isinstance(statement, ast.AugAssign):
            old = _expr_value_dependency(statement.target, values)
            new = _expr_value_dependency(statement.value, values)
            dependency = True if True in {old, new} else None if None in {old, new} else False
            if not _assign_dependency(statement.target, dependency, values):
                return "unknown"
        elif isinstance(statement, ast.Return):
            returns.append(_expr_value_dependency(statement.value, values))
        elif isinstance(statement, ast.Expr):
            # Track common in-place writes such as
            # ``self.buffer.fill_(x.sum().item())``.
            call = statement.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Attribute)
                and isinstance(call.func.value.value, ast.Name)
                and call.func.value.value.id == "self"
                and call.func.attr.endswith("_")
            ):
                dependency = _expr_value_dependency(call, values)
                if dependency is None:
                    return "unknown"
                values[f"self.{call.func.value.attr}"] = dependency
            elif (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr.endswith("_")
                and isinstance(call.func.value, ast.Name)
            ):
                dependency = _expr_value_dependency(call, values)
                if dependency is None:
                    return "unknown"
                values[call.func.value.id] = dependency
        elif isinstance(statement, (ast.Pass, ast.Assert)):
            continue
        else:
            return "unknown"
    if not returns or any(value is None for value in returns):
        return "unknown"
    return "dependent" if any(returns) else "independent"


def _loaded_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {part.id for part in ast.walk(node) if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load)}


def _assigned_local_names(target: ast.AST) -> set[str] | None:
    """Return local bindings for a side-effect-free assignment target.

    Attribute and subscript stores may mutate externally visible state, so the
    dead-computation check deliberately declines to classify them.
    """

    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            item_names = _assigned_local_names(item)
            if item_names is None:
                return None
            names.update(item_names)
        return names
    if isinstance(target, ast.Starred):
        return _assigned_local_names(target.value)
    return None


def _self_bindings(model: ast.ClassDef) -> dict[str, tuple[str, ast.AST | None]]:
    """Map explicit ``self.name = value`` bindings in the effective initializer."""

    initializers = [
        node
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    ]
    initializer = initializers[-1] if initializers else None
    if initializer is None:
        return {}
    bindings: dict[str, tuple[str, ast.AST | None]] = {}
    for statement in ast.walk(initializer):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value = statement.value
        if isinstance(value, ast.Call):
            binding = _call_name(value.func)
        else:
            binding = _call_name(value) if value is not None else ""
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                bindings[target.attr] = (binding, value)
    return bindings


def _dead_call_effects(model: ast.ClassDef, calls: tuple[str, ...]) -> tuple[str, ...]:
    """Classify observable effects that a return-value slice cannot model."""

    bindings = _self_bindings(model)
    methods = {node.name for node in model.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    effects: set[str] = set()
    for call in calls:
        short = call.split(".")[-1]
        if _RANDOM_CALL_RE.search(call) or short in {
            "dropout",
            "dropout1d",
            "dropout2d",
            "dropout3d",
            "alpha_dropout",
            "feature_alpha_dropout",
        }:
            effects.add("rng")
        if not call.startswith("self."):
            continue
        attribute = call.split(".", 2)[1]
        binding, binding_node = bindings.get(attribute, ("", None))
        binding_short = binding.split(".")[-1]
        if "BatchNorm" in binding_short or binding_short == "SyncBatchNorm":
            effects.add("stateful")
        elif "Dropout" in binding_short:
            effects.add("rng")
        elif binding_short in {"RNN", "LSTM", "GRU"} and isinstance(binding_node, ast.Call):
            layers = _call_argument(binding_node, 2, "num_layers", 1)
            dropout = _call_argument(binding_node, 5, "dropout", 0.0)
            if dropout is None:
                if layers != 1:
                    effects.add("unknown")
            elif isinstance(dropout, (int, float)) and dropout > 0:
                if layers is None:
                    effects.add("unknown")
                elif layers > 1:
                    effects.add("rng")
        elif binding_short == "MultiheadAttention" and isinstance(binding_node, ast.Call):
            dropout = _call_argument(binding_node, 2, "dropout", 0.0)
            if dropout is None:
                effects.add("unknown")
            elif isinstance(dropout, (int, float)) and dropout > 0:
                effects.add("rng")
        elif binding_short in {"InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"} and isinstance(
            binding_node, ast.Call
        ):
            tracking = _call_argument(binding_node, 4, "track_running_stats", False)
            if tracking is None:
                effects.add("unknown")
            elif tracking is True:
                effects.add("stateful")
        elif binding_short.startswith("Transformer"):
            # Transformer containers normally contain dropout-bearing layers;
            # constructor signatures vary enough that unresolved parameters are
            # deliberately quarantined rather than guessed pure.
            effects.add("unknown")
        elif attribute in methods and attribute not in bindings:
            effects.add("unknown")
    return tuple(sorted(effects))


def _dead_forward_calls(
    model: ast.ClassDef,
    forward: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[int, tuple[str, ...], tuple[str, ...]]]:
    """Find straight-line call results overwritten before reaching a return.

    This is a backward local-liveness slice, not a claim that arbitrary Python
    calls are pure.  It only reports assignments to local names in a
    straight-line ``forward``.  Attribute/subscript stores and control flow are
    left unclassified because they may carry side effects or path-sensitive
    uses.  When an assignment is dead, its RHS dependencies are intentionally
    not propagated: this lets the slice expose an entire abandoned chain.
    """

    body: list[ast.stmt] = list(forward.body)
    terminal_condition_names: set[str] = set()
    if body and isinstance(body[-1], ast.If):
        terminal = body[-1]
        if (
            len(terminal.body) == 1
            and len(terminal.orelse) == 1
            and isinstance(terminal.body[0], ast.Return)
            and isinstance(terminal.orelse[0], ast.Return)
            and ast.dump(terminal.body[0].value, include_attributes=False)
            == ast.dump(terminal.orelse[0].value, include_attributes=False)
        ):
            # Both branches expose the same result.  Keep condition dependencies
            # live, but collapse the terminal return so earlier abandoned call
            # chains remain provably dead without general path analysis.
            terminal_condition_names = _loaded_names(terminal.test)
            body[-1] = terminal.body[0]

    if any(
        not isinstance(
            statement,
            (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Expr, ast.Pass, ast.Assert),
        )
        for statement in body
    ):
        return []

    live: set[str] = set(terminal_condition_names)
    dead: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
    for statement in reversed(body):
        if isinstance(statement, ast.Return):
            live.update(_loaded_names(statement.value))
            continue
        if isinstance(statement, (ast.Pass, ast.Assert)):
            if isinstance(statement, ast.Assert):
                live.update(_loaded_names(statement))
            continue
        if isinstance(statement, ast.Expr):
            # A standalone call may exist specifically for its side effect.
            live.update(_loaded_names(statement))
            continue

        if isinstance(statement, ast.Assign):
            target_sets = [_assigned_local_names(target) for target in statement.targets]
            if any(names is None for names in target_sets):
                live.update(_loaded_names(statement))
                continue
            targets = set().union(*(names or set() for names in target_sets))
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = _assigned_local_names(statement.target)
            if targets is None:
                live.update(_loaded_names(statement))
                continue
            value = statement.value
        else:  # AugAssign reads and writes its target.
            targets = _assigned_local_names(statement.target)
            if targets is None:
                live.update(_loaded_names(statement))
                continue
            value = statement.value

        calls = tuple(
            sorted(
                {
                    name
                    for node in (ast.walk(value) if value is not None else ())
                    if isinstance(node, ast.Call)
                    for name in [_call_name(node.func)]
                    if name
                }
            )
        )
        has_observable_side_effect = any(
            (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.endswith("_")
                and not node.func.attr.startswith("__")
            )
            or any(keyword.arg == "out" for keyword in node.keywords)
            or _call_name(node.func) == "setattr"
            or (isinstance(node.func, ast.Attribute) and node.func.attr in {"append", "extend", "insert", "update"})
            for node in (ast.walk(value) if value is not None else ())
            if isinstance(node, ast.Call)
        )
        if targets.isdisjoint(live):
            if calls and not has_observable_side_effect:
                dead.append((getattr(statement, "lineno", 0), calls, _dead_call_effects(model, calls)))
            elif has_observable_side_effect:
                live.update(_loaded_names(value))
            continue

        live.difference_update(targets)
        live.update(_loaded_names(value))
        if isinstance(statement, ast.AugAssign):
            live.update(targets)
    return sorted(dead)


def _literal_value(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _call_argument(call: ast.Call, position: int, keyword: str, default: Any) -> Any:
    if len(call.args) > position:
        return _literal_value(call.args[position])
    for item in call.keywords:
        if item.arg == keyword:
            return _literal_value(item.value)
    return default


_FUNCTIONAL_DROPOUT_MEMBERS = frozenset(
    {
        "dropout",
        "dropout1d",
        "dropout2d",
        "dropout3d",
        "alpha_dropout",
        "feature_alpha_dropout",
    }
)


def _functional_dropout_names(tree: ast.Module) -> frozenset[str]:
    """Return qualified/aliased names that denote PyTorch functional dropout."""

    prefixes = {"F", "torch.nn.functional"}
    direct: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch.nn.functional":
                    prefixes.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torch.nn":
                for alias in node.names:
                    if alias.name == "functional":
                        prefixes.add(alias.asname or alias.name)
            elif node.module == "torch.nn.functional":
                for alias in node.names:
                    if alias.name in _FUNCTIONAL_DROPOUT_MEMBERS:
                        direct.add(alias.asname or alias.name)
    return frozenset(direct | {f"{prefix}.{member}" for prefix in prefixes for member in _FUNCTIONAL_DROPOUT_MEMBERS})


def _resolved_literal(node: ast.AST | None, values: dict[str, Any]) -> Any:
    """Resolve a small, side-effect-free literal expression used by contracts."""

    if node is None:
        return None
    literal = _literal_value(node)
    if literal is not None:
        return literal
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return values.get(f"self.{node.attr}")
    return None


def _simple_assignment_literals(statements: list[ast.stmt], inherited: dict[str, Any]) -> dict[str, Any]:
    values = dict(inherited)
    for statement in statements:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value = _resolved_literal(statement.value, values)
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
                for item, item_value in zip(target.elts, value, strict=False):
                    if isinstance(item, ast.Name):
                        values[item.id] = item_value
    return values


def _constructor_self_literals(
    tree: ast.Module,
    model: ast.ClassDef,
    get_init_inputs: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> dict[str, Any]:
    """Resolve common ``get_init_inputs -> __init__ -> self`` literal flows."""

    module_values = _simple_assignment_literals(tree.body, {})
    init_values: list[Any] = []
    if get_init_inputs is not None:
        local_values = _simple_assignment_literals(get_init_inputs.body, module_values)
        returns = [node for node in ast.walk(get_init_inputs) if isinstance(node, ast.Return)]
        if returns and isinstance(returns[-1].value, (ast.List, ast.Tuple)):
            init_values = [_resolved_literal(item, local_values) for item in returns[-1].value.elts]

    initializers = [
        node
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    ]
    if not initializers:
        return {}
    initializer = initializers[-1]
    positional = list(initializer.args.posonlyargs) + list(initializer.args.args)
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    constructor_values = dict(module_values)
    constructor_values.update(
        {argument.arg: value for argument, value in zip(positional, init_values, strict=False) if value is not None}
    )
    self_values: dict[str, Any] = {}
    for statement in initializer.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value = _resolved_literal(statement.value, constructor_values | self_values)
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                self_values[f"self.{target.attr}"] = value
    return self_values


def _unconditional_functional_dropout(
    call: ast.Call,
    self_literals: dict[str, Any],
) -> tuple[bool, str]:
    """Classify functional dropout whose train flag cannot be disabled by mode.

    ``torch.nn.functional.dropout`` defaults to ``training=True``.  Such a call
    is random whenever ``0 < p < 1``.  A dynamically resolved probability is
    conservatively random unless the contract proves the degenerate ``p=0`` or
    ``p=1`` case.
    """

    training_node = next((item.value for item in call.keywords if item.arg == "training"), None)
    if training_node is None and len(call.args) > 2:
        training_node = call.args[2]
    if (
        isinstance(training_node, ast.Attribute)
        and isinstance(training_node.value, ast.Name)
        and training_node.value.id == "self"
        and training_node.attr == "training"
    ):
        # ``Module.eval()``/``train()`` rewrites this conventional flag even
        # when a constructor happened to initialize it from a literal.
        return False, "dynamic_training"
    training = True if training_node is None else _resolved_literal(training_node, self_literals)
    if training is not True:
        return False, "disabled" if training is False else "dynamic_training"

    probability_node = next((item.value for item in call.keywords if item.arg == "p"), None)
    if probability_node is None and len(call.args) > 1:
        probability_node = call.args[1]
    probability = 0.5 if probability_node is None else _resolved_literal(probability_node, self_literals)
    if isinstance(probability, (int, float)) and not isinstance(probability, bool):
        if probability in {0, 1}:
            return False, f"degenerate_p={probability}"
        return True, f"p={probability}"
    return True, "unresolved_p"


def _train_eval_mode_candidates(
    model: ast.ClassDef,
    forward: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    """Return static reasons to schedule a train/eval runtime comparison."""

    methods = {node.name: node for node in model.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    reachable = {forward.name}
    pending = [forward.name]
    while pending:
        method = methods[pending.pop()]
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in methods
                and node.func.attr not in reachable
            ):
                reachable.add(node.func.attr)
                pending.append(node.func.attr)

    candidates: set[str] = set()
    for name in reachable:
        if any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "training"
            for node in ast.walk(methods[name])
        ):
            candidates.add("explicit_self_training")

    initializer = methods.get("__init__")
    if initializer is None:
        return tuple(sorted(candidates))
    for node in ast.walk(initializer):
        if not isinstance(node, ast.Call):
            continue
        constructor = _call_name(node.func)
        short = constructor.split(".")[-1]
        if short in {
            "Dropout",
            "Dropout1d",
            "Dropout2d",
            "Dropout3d",
            "AlphaDropout",
            "FeatureAlphaDropout",
        }:
            probability = _call_argument(node, 0, "p", 0.5)
            if probability is None or (isinstance(probability, (int, float)) and probability > 0):
                candidates.add("dropout")
        elif short.startswith("BatchNorm") or short.startswith("LazyBatchNorm") or short == "SyncBatchNorm":
            if _call_argument(node, 99, "track_running_stats", True) is not False:
                candidates.add("batch_norm")
        elif short.startswith("InstanceNorm") or short.startswith("LazyInstanceNorm"):
            tracking = _call_argument(node, 99, "track_running_stats", False)
            if tracking is True or tracking is None:
                candidates.add("instance_norm")
        elif short in {"RNN", "LSTM", "GRU"}:
            layers = _call_argument(node, 2, "num_layers", 1)
            dropout = _call_argument(node, 5, "dropout", 0.0)
            if (dropout is None and layers != 1) or (
                isinstance(dropout, (int, float)) and dropout > 0 and (layers is None or layers > 1)
            ):
                candidates.add("recurrent_dropout")
        elif short == "MultiheadAttention":
            dropout = _call_argument(node, 2, "dropout", 0.0)
            if dropout is None or (isinstance(dropout, (int, float)) and dropout > 0):
                candidates.add("attention_dropout")
        elif short.startswith("Transformer"):
            dropout = _call_argument(node, 99, "dropout", 0.1)
            if dropout is None or (isinstance(dropout, (int, float)) and dropout > 0):
                candidates.add("transformer_dropout")
    return tuple(sorted(candidates))


def _unused_module_category(constructor: str) -> str:
    short = constructor.split(".")[-1]
    if short in {"ParameterDict", "ParameterList"}:
        return "parameter_container"
    if short == "Parameter":
        return "parameter"
    if short in {"ModuleDict", "ModuleList", "Sequential"}:
        return "module_container"
    if short.endswith("Loss") or short in {"CosineSimilarity", "TripletMarginWithDistanceLoss"}:
        return "loss_module"
    return "compute_module"


def _unused_forward_modules(
    model: ast.ClassDef, forward: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[tuple[str, str]]:
    """Return initialized modules/parameters unreachable from ``forward``.

    Reachability includes class helper methods called through ``self``.  The
    rule is intentionally limited to explicit ``self.name = nn.*(...)`` or
    ``torch.nn.*(...)`` assignments; plain configuration attributes are not
    treated as executable model state.
    """

    methods = {node.name: node for node in model.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    initializer = methods.get("__init__")
    if initializer is None:
        return []

    initialized: dict[str, str] = {}
    for statement in ast.walk(initializer):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value = statement.value
        if not isinstance(value, ast.Call):
            continue
        constructor = _call_name(value.func)
        if not (constructor.startswith("nn.") or constructor.startswith("torch.nn.")):
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                initialized[target.attr] = constructor

    if not initialized:
        return []

    # Modules can be composed during initialization and then reached through
    # only their container (for example ``self.seq = nn.Sequential(self.a,
    # self.b)``).  A child loaded anywhere in ``__init__`` is therefore not an
    # unused-forward false positive; if the container itself is unreachable it
    # will still be reported.
    composed_children = {
        node.attr
        for node in ast.walk(initializer)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Load)
    }

    reachable = {forward.name}
    pending = [forward.name]
    while pending:
        method = methods[pending.pop()]
        for node in ast.walk(method):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                continue
            called = node.func.attr
            if called in methods and called not in initialized and called not in reachable:
                reachable.add(called)
                pending.append(called)

    used: set[str] = set()
    dynamic_self_access = False
    for name in reachable:
        for node in ast.walk(methods[name]):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Load)
            ):
                used.add(node.attr)
                if node.attr in {"__dict__", "_modules", "_parameters"}:
                    dynamic_self_access = True
            elif isinstance(node, ast.Call) and _call_name(node.func) in {"getattr", "hasattr", "vars"}:
                if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                    dynamic_self_access = True
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr
                in {
                    "children",
                    "modules",
                    "named_children",
                    "named_modules",
                    "named_parameters",
                    "parameters",
                    "state_dict",
                }
            ):
                dynamic_self_access = True
    if dynamic_self_access:
        return []
    used.update(composed_children)
    return sorted((name, constructor) for name, constructor in initialized.items() if name not in used)


def _global_reads_shadowing_init_arguments(
    model: ast.ClassDef, forward: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[str]:
    """Find ignored constructor arguments read as same-named globals later."""

    methods = {node.name: node for node in model.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    initializer = methods.get("__init__")
    if initializer is None:
        return []
    init_positional = list(initializer.args.posonlyargs) + list(initializer.args.args)
    if init_positional and init_positional[0].arg in {"self", "cls"}:
        init_positional = init_positional[1:]
    init_args = {arg.arg for arg in init_positional + list(initializer.args.kwonlyargs)}
    if initializer.args.vararg is not None:
        init_args.add(initializer.args.vararg.arg)
    if initializer.args.kwarg is not None:
        init_args.add(initializer.args.kwarg.arg)
    candidates = init_args - _loaded_names(initializer)
    if not candidates:
        return []

    reachable = {forward.name}
    pending = [forward.name]
    while pending:
        method = methods[pending.pop()]
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in methods
                and node.func.attr not in reachable
            ):
                reachable.add(node.func.attr)
                pending.append(node.func.attr)

    leaked: set[str] = set()
    for name in reachable:
        method = methods[name]
        positional = list(method.args.posonlyargs) + list(method.args.args)
        local_names = {arg.arg for arg in positional + list(method.args.kwonlyargs)}
        if method.args.vararg is not None:
            local_names.add(method.args.vararg.arg)
        if method.args.kwarg is not None:
            local_names.add(method.args.kwarg.arg)
        local_names.update(
            node.id for node in ast.walk(method) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        local_names.difference_update(
            declared for node in ast.walk(method) if isinstance(node, ast.Global) for declared in node.names
        )
        leaked.update(
            node.id
            for node in ast.walk(method)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in candidates
            and node.id not in local_names
        )
    return sorted(leaked)


def analyze_reference(
    code: Any,
    entry_point: Any,
    *,
    reject_unused_forward_args: bool = True,
    reject_random_forward: bool = True,
) -> StaticAnalysis:
    """Return high-confidence rejection reasons and nonfatal audit flags."""

    fatal: list[str] = []
    flags: list[str] = []
    if not isinstance(code, str) or not code.strip():
        return StaticAnalysis(("empty_reference",), (), None)
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        return StaticAnalysis(("invalid_entry_point",), (), None)
    if "```" in code:
        fatal.append("reference_contains_code_fence")

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        flags.append(f"syntax_detail:{type(exc).__name__}:{getattr(exc, 'lineno', None)}")
        return StaticAnalysis(tuple(fatal + ["syntax_error"]), tuple(flags), None)

    semantic_hash = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry_point]
    function_lists: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_lists[node.name].append(node)
    top_functions = {name: nodes[-1] for name, nodes in function_lists.items()}
    if len(classes) > 1:
        fatal.append("duplicate_entry_point_definition")
    if len(function_lists["get_inputs"]) > 1:
        fatal.append("duplicate_get_inputs_definition")
    if len(function_lists["get_init_inputs"]) > 1:
        fatal.append("duplicate_get_init_inputs_definition")
    if not classes:
        fatal.append("missing_entry_point_class")
    if "get_inputs" not in top_functions:
        fatal.append("missing_get_inputs")
    if "get_init_inputs" not in top_functions:
        flags.append("missing_get_init_inputs")
    else:
        random_init_calls = _uses_random_init_scalar(top_functions["get_init_inputs"])
        if random_init_calls:
            fatal.append("random_init_scalar")
            flags.extend(f"random_init_scalar_call:{name}" for name in random_init_calls)

    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in _UNSAFE_CALLS or name.startswith("subprocess."):
                fatal.append("unsafe_reference_code")
                flags.append(f"unsafe_call:{name}:line={getattr(node, 'lineno', None)}")
    for name, line in _filesystem_serialization_calls(tree):
        fatal.append("unsafe_reference_code")
        flags.append(f"unsafe_filesystem_serialization:{name}:line={line}")
    if imported_roots & _UNSAFE_IMPORTS:
        fatal.append("unsafe_reference_code")
    foreign_imports = sorted(imported_roots - {"collections", "functools", "math", "torch", "typing"})
    flags.extend(f"nonstandard_import:{name}" for name in foreign_imports)

    functional_dropout_names = _functional_dropout_names(tree)

    if classes:
        self_literals = _constructor_self_literals(tree, classes[0], top_functions.get("get_init_inputs"))
        forwards = [
            node
            for node in classes[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
        ]
        if not forwards:
            fatal.append("missing_forward")
        else:
            if len(forwards) > 1:
                fatal.append("duplicate_forward_definition")
            # Canonicalized references have one forward.  Taking the last one
            # also matches Python semantics when analyze_reference is called
            # directly on an uncanonicalized snippet.
            forward = forwards[-1]
            positional = list(forward.args.posonlyargs) + list(forward.args.args)
            if positional and positional[0].arg in {"self", "cls"}:
                positional = positional[1:]
            named_args = [arg.arg for arg in positional + list(forward.args.kwonlyargs)]
            input_names = list(named_args)
            if forward.args.vararg is not None:
                input_names.append(forward.args.vararg.arg)
            if forward.args.kwarg is not None:
                input_names.append(forward.args.kwarg.arg)
            executable_body = [
                node
                for node in forward.body
                if not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )
            ]
            if (
                len(executable_body) == 1
                and isinstance(executable_body[0], ast.Return)
                and isinstance(executable_body[0].value, ast.Name)
                and executable_body[0].value.id in input_names
            ):
                flags.append(f"identity_forward_arg:{executable_body[0].value.id}")
                fatal.append("identity_forward")
            used_names = {
                node.id for node in ast.walk(forward) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            unused = [name for name in input_names if name not in used_names]
            if unused:
                flags.extend(f"unused_forward_arg:{name}" for name in unused)
                if reject_unused_forward_args:
                    fatal.append("unused_forward_argument")
            if not input_names:
                flags.append("forward_has_no_inputs")
                fatal.append("forward_has_no_inputs")
            if not any(isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) for node in ast.walk(forward)):
                fatal.append("forward_has_no_return")

            output_dependency = _forward_output_dependency(forward, input_names)
            flags.append(f"forward_output_dependency:{output_dependency}")
            if output_dependency == "independent" and input_names:
                fatal.append("forward_output_independent")

            dead_calls = _dead_forward_calls(classes[0], forward)
            for line, calls, effects in dead_calls:
                call_text = ",".join(calls)
                if not effects:
                    fatal.append("dead_forward_computation")
                    flags.append(f"dead_forward_computation:line={line}:calls={call_text}")
                    continue
                for effect in effects:
                    reason = f"dead_forward_{effect}_effect"
                    fatal.append(reason)
                    flags.append(f"{reason}:line={line}:calls={call_text}")

            unused_modules = _unused_forward_modules(classes[0], forward)
            if unused_modules:
                fatal.append("unused_forward_module")
                for name, constructor in unused_modules:
                    flags.append(f"unused_forward_module:{name}:{constructor}")
                    flags.append(f"unused_forward_{_unused_module_category(constructor)}:{name}:{constructor}")

            flags.extend(
                f"train_eval_static_candidate:{reason}" for reason in _train_eval_mode_candidates(classes[0], forward)
            )

            global_init_reads = _global_reads_shadowing_init_arguments(classes[0], forward)
            if global_init_reads:
                fatal.append("forward_uses_global_instead_of_init_argument")
                flags.extend(f"forward_uses_global_instead_of_init_argument:{name}" for name in global_init_reads)

            seeded_generators, global_seed_lines, distributions = _forward_rng_context(forward, self_literals)
            self_bindings = _self_bindings(classes[0])
            for node in ast.walk(forward):
                if isinstance(node, ast.Call):
                    name = _call_name(node.func)
                    random_kind = _forward_random_call_kind(
                        node,
                        name,
                        seeded_generators=seeded_generators,
                        global_seed_lines=global_seed_lines,
                        distributions=distributions,
                    )
                    if random_kind is None and _fractional_pool_uses_rng(node, name, self_bindings):
                        line = getattr(node, "lineno", -1)
                        random_kind = (
                            "seeded" if any(seed_line < line for seed_line in global_seed_lines) else "random"
                        )
                    if random_kind == "random":
                        flags.append(f"random_forward_call:{name}")
                        if reject_random_forward:
                            fatal.append("random_forward")
                    elif random_kind == "seeded":
                        flags.append(f"deterministic_seeded_forward_rng:{name}")
                    if name in _TORCH_SERIALIZATION_CALLS:
                        flags.append(f"non_kernel_forward_serialization:{name}")
                        fatal.append("non_kernel_forward_serialization")
                    if name in _EXECUTION_MODE_CALLS:
                        flags.append(f"execution_mode_dependent_forward:{name}")
                        fatal.append("execution_mode_dependent_forward")
                    if name in functional_dropout_names:
                        is_random, detail = _unconditional_functional_dropout(node, self_literals)
                        flags.append(f"functional_dropout:{name}:{detail}")
                        if is_random:
                            if any(seed_line < getattr(node, "lineno", -1) for seed_line in global_seed_lines):
                                flags.append(f"deterministic_seeded_forward_rng:{name}")
                            else:
                                flags.append(f"random_forward_call:{name}")
                                if reject_random_forward:
                                    fatal.append("random_forward")
                    if isinstance(node.func, ast.Attribute) and node.func.attr.endswith("_"):
                        for name in input_names:
                            if any(
                                isinstance(part, ast.Name) and part.id == name for part in ast.walk(node.func.value)
                            ):
                                flags.append(f"forward_may_mutate_input:{name}")
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(
                        isinstance(part, ast.Attribute)
                        and isinstance(part.value, ast.Name)
                        and part.value.id == "self"
                        for target in targets
                        for part in ast.walk(target)
                    ):
                        flags.append("forward_mutates_model_state")
                    for name in input_names:
                        if any(_target_mutates_name(target, name) for target in targets):
                            flags.append(f"forward_may_mutate_input:{name}")

    # Stable ordering makes the audit byte-reproducible.
    return StaticAnalysis(tuple(sorted(set(fatal))), tuple(sorted(set(flags))), semantic_hash)
