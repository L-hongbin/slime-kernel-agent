"""Exact AST view of frozen random/value wrappers for serial interventions."""

from __future__ import annotations

import ast
import copy
import dataclasses

from tools.data.synthesize.augment_prompt_tasks import _SHAPE_FACTORIES, _call_name, _top_level_function
from tools.data.synthesize.random_method import solve_value_coverage as random_solver


@dataclasses.dataclass(frozen=True)
class RandomWrapper:
    family: str
    root: ast.Call
    base: ast.Call
    local_seed: int


@dataclasses.dataclass(frozen=True)
class LogicalFactoryView:
    tree: ast.Module
    original_roots: tuple[ast.Call, ...]
    random_wrapper_count: int


def _manual_seed(call: ast.Call) -> int | None:
    seeds: list[int] = []
    for node in ast.walk(call):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "manual_seed" or len(node.args) != 1 or node.keywords:
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and type(value.value) is int:
            seeds.append(value.value)
    if not seeds:
        return None
    if len(seeds) != 1:
        return -1
    return seeds[0]


def recognize_random_wrapper(call: ast.Call) -> RandomWrapper | None:
    if not isinstance(call.func, ast.Lambda) or len(call.args) != 1 or call.keywords:
        return None
    function = call.func
    if (
        function.args.posonlyargs
        or len(function.args.args) != 1
        or function.args.args[0].arg != "_value_draw"
        or function.args.vararg is not None
        or function.args.kwonlyargs
        or function.args.kw_defaults
        or function.args.kwarg is not None
        or function.args.defaults
    ):
        return None
    base = call.args[0]
    if not isinstance(base, ast.Call) or _call_name(base.func) != "torch.randn":
        return None
    observed_seed = _manual_seed(call)
    if observed_seed == -1:
        return None
    for family in random_solver.VALUE_FAMILIES:
        if family in {"poisson_counts", "multinomial_categories"}:
            if observed_seed is None or not 0 < observed_seed < random_solver.LOCAL_GENERATOR_MAX_SEED:
                continue
            seed = observed_seed
        else:
            if observed_seed is not None:
                continue
            seed = 1
        expected = random_solver._value_wrapper(copy.deepcopy(base), family, seed)
        if ast.dump(expected, include_attributes=False) == ast.dump(call, include_attributes=False):
            return RandomWrapper(family=family, root=call, base=base, local_seed=seed)
    return None


class _DewrapRandom(ast.NodeTransformer):
    def __init__(self) -> None:
        self.count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        wrapper = recognize_random_wrapper(node)
        if wrapper is not None:
            self.count += 1
            return ast.copy_location(copy.deepcopy(wrapper.base), node)
        return self.generic_visit(node)


class _LogicalRootCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.roots: list[ast.Call] = []
        self.random_wrapper_count = 0

    def visit_Call(self, node: ast.Call) -> None:
        wrapper = recognize_random_wrapper(node)
        if wrapper is not None:
            self.roots.append(node)
            self.random_wrapper_count += 1
            return
        if _call_name(node.func) in _SHAPE_FACTORIES:
            self.roots.append(node)
            return
        self.generic_visit(node)


def logical_factory_view(tree: ast.Module) -> LogicalFactoryView:
    get_inputs = _top_level_function(tree, "get_inputs")
    collector = _LogicalRootCollector()
    collector.visit(get_inputs)
    collector.roots.sort(
        key=lambda node: (node.lineno, node.col_offset, node.end_lineno or -1, node.end_col_offset or -1)
    )

    virtual = copy.deepcopy(tree)
    transformer = _DewrapRandom()
    transformed = transformer.visit(virtual)
    if not isinstance(transformed, ast.Module):
        raise ValueError("random wrapper dewrap did not return a module")
    ast.fix_missing_locations(transformed)
    if transformer.count != collector.random_wrapper_count:
        raise ValueError("random wrapper count differs between logical and virtual views")
    return LogicalFactoryView(
        tree=transformed,
        original_roots=tuple(collector.roots),
        random_wrapper_count=collector.random_wrapper_count,
    )


def root_base_call(root: ast.Call) -> ast.Call:
    wrapper = recognize_random_wrapper(root)
    return wrapper.base if wrapper is not None else root
