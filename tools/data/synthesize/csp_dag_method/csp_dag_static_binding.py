"""Pure-static CSP-DAG manifest-to-source binding checks.

This module deliberately has no Torch, parquet, graph-runtime, or execution
dependencies.  Both runtime validation and static quality auditing can use the
same fail-closed graph/source reconstruction without importing an executable
validator.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from typing import Any

from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator


_SERIAL_EXPANSION_CONTRACT = "csp_dag_shape_dtype_layout_expansion_v1"


def uses_single_model_policy(manifest: Mapping[str, Any]) -> bool:
    """Return whether a current base/expansion manifest forbids fake helpers."""

    return (
        manifest.get("contract") in {generator.CONTRACT, _SERIAL_EXPANSION_CONTRACT}
        and "requested_multiclass" not in manifest
        and manifest.get("top_level_class_policy") == "single_model_only"
        and manifest.get("semantic_multiclass_status") == "deferred_until_real_subgraph_helper_lowering"
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def ast_key(node: ast.AST) -> str:
    """Return an AST comparison key which ignores source locations/parentheses."""

    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _expected_function(name: str, arguments: str, body_lines: list[str]) -> ast.FunctionDef:
    source = "\n".join([f"def {name}({arguments}):", *(f"    {line}" for line in body_lines)])
    return ast.parse(source).body[0]  # type: ignore[return-value]


def expected_expression(
    node: generator.Node,
    sources: list[str],
    rank: int,
) -> ast.AST:
    """Reconstruct one node's source lowering."""

    expression = generator._operation_expression(node, sources, rank)
    return ast.parse(expression, mode="eval").body


def typed_graph_context(
    manifest: dict[str, Any],
) -> tuple[dict[str, bool], list[generator.Node] | None, int | None, tuple[int, ...] | None]:
    """Validate a typed graph before it is used to reconstruct source.

    ``edges`` must exactly equal the ordered predecessor expansion, retaining
    duplicates so a tampered edge list cannot hide behind a set comparison.
    """

    checks = {
        "typed_graph_present": False,
        "typed_graph_hash_bound": False,
        "typed_graph_nodes_well_formed": False,
        "typed_graph_edges_match_predecessors": False,
        "typed_graph_counts_bound": False,
        "typed_graph_input_witness_bound": False,
    }
    try:
        graph_data = manifest["graph"]
        typed = graph_data["typed_graph"]
        if not isinstance(graph_data, dict) or not isinstance(typed, dict):
            return checks, None, None, None
        checks["typed_graph_present"] = True
        declared_hash = graph_data.get("typed_graph_sha256")
        checks["typed_graph_hash_bound"] = isinstance(declared_hash, str) and declared_hash == _sha256_text(
            generator._canonical_json(typed)
        )
        raw_nodes = typed.get("nodes")
        rank = typed.get("input_rank")
        raw_shape = typed.get("input_shape_witness")
        if (
            not isinstance(raw_nodes, list)
            or not raw_nodes
            or type(rank) is not int
            or rank < 1
            or not isinstance(raw_shape, (list, tuple))
            or len(raw_shape) != rank
            or not all(type(value) is int and value > 0 for value in raw_shape)
        ):
            return checks, None, None, None

        nodes: list[generator.Node] = []
        for expected_index, raw in enumerate(raw_nodes):
            if not isinstance(raw, dict):
                return checks, None, None, None
            index = raw.get("index")
            rule_name = raw.get("rule")
            family = raw.get("family")
            variant = raw.get("variant")
            predecessors = raw.get("predecessors")
            module_name = raw.get("module_name")
            if (
                type(index) is not int
                or index != expected_index
                or not isinstance(rule_name, str)
                or rule_name not in generator.RULE_BY_NAME
                or not isinstance(variant, str)
                or not isinstance(predecessors, (list, tuple))
                or not all(type(predecessor) is int for predecessor in predecessors)
            ):
                return checks, None, None, None
            rule = generator.RULE_BY_NAME[rule_name]
            normalized_predecessors = tuple(predecessors)
            expected_module_name = f"op_{index:03d}" if rule.module else None
            if (
                family != rule.family
                or variant not in rule.variants
                or rank not in rule.ranks
                or len(normalized_predecessors) != generator._variant_arity(rule_name, variant)
                or any(predecessor < -1 or predecessor >= index for predecessor in normalized_predecessors)
                or module_name != expected_module_name
            ):
                return checks, None, None, None
            nodes.append(
                generator.Node(
                    index=index,
                    rule=rule_name,
                    family=family,
                    variant=variant,
                    predecessors=normalized_predecessors,
                    module_name=module_name,
                )
            )
        checks["typed_graph_nodes_well_formed"] = True

        raw_edges = typed.get("edges")
        if not isinstance(raw_edges, list):
            return checks, nodes, rank, tuple(raw_shape)
        normalized_edges: list[tuple[int, int]] = []
        for edge in raw_edges:
            if (
                not isinstance(edge, (list, tuple))
                or len(edge) != 2
                or type(edge[0]) is not int
                or type(edge[1]) is not int
            ):
                return checks, nodes, rank, tuple(raw_shape)
            normalized_edges.append((edge[0], edge[1]))
        expected_edges = [(predecessor, node.index) for node in nodes for predecessor in node.predecessors]
        checks["typed_graph_edges_match_predecessors"] = normalized_edges == expected_edges
        checks["typed_graph_counts_bound"] = graph_data.get("node_count") == len(nodes) and graph_data.get(
            "edge_count"
        ) == len(expected_edges)
        shape_csp = graph_data.get("shape_csp")
        checks["typed_graph_input_witness_bound"] = isinstance(shape_csp, dict) and shape_csp.get("witness") == list(
            raw_shape
        )
        return checks, nodes, rank, tuple(raw_shape)
    except (KeyError, TypeError, ValueError, SyntaxError):
        return checks, None, None, None


def source_lowering_checks(
    code: str,
    manifest: dict[str, Any],
    nodes: list[generator.Node] | None,
    rank: int | None,
    shape: tuple[int, ...] | None,
) -> dict[str, bool]:
    """Bind source to its typed graph and manifest metrics, fail closed."""

    checks = {
        "source_top_level_structure_bound": False,
        "model_class_definition_bound": False,
        "model_init_lowering_bound": False,
        "forward_argument_binding": False,
        "forward_v_assignment_count_bound": False,
        "forward_v_assignment_order_bound": False,
        "forward_expression_lowering_bound": False,
        "forward_return_last_v_bound": False,
        "multiclass_wrapper_bound": False,
        "semantic_multiclass_policy_bound": False,
        "get_inputs_lowering_bound": False,
        "code_metrics_bound": False,
    }
    if nodes is None or rank is None or shape is None:
        return checks
    try:
        tree = ast.parse(code)
        if not uses_single_model_policy(manifest):
            return checks
        checks["semantic_multiclass_policy_bound"] = True

        expected_imports = ast.parse("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n").body
        expected_class_names = ["Model"]
        import_count = len(expected_imports)
        class_end = import_count + len(expected_class_names)
        expected_body_count = class_end + 1
        class_nodes = tree.body[import_count:class_end]
        final_node = tree.body[class_end] if len(tree.body) == expected_body_count else None
        checks["source_top_level_structure_bound"] = (
            len(tree.body) == expected_body_count
            and [ast_key(node) for node in tree.body[:import_count]] == [ast_key(node) for node in expected_imports]
            and all(isinstance(node, ast.ClassDef) for node in class_nodes)
            and [node.name for node in class_nodes] == expected_class_names
            and isinstance(final_node, ast.FunctionDef)
            and final_node.name == "get_inputs"
        )

        model = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"),
            None,
        )
        get_inputs = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"),
            None,
        )
        if model is None or get_inputs is None:
            return checks

        expected_model_header = ast.parse("class Model(nn.Module):\n    pass\n").body[0]
        checks["model_class_definition_bound"] = (
            model.name == expected_model_header.name
            and [ast_key(base) for base in model.bases] == [ast_key(base) for base in expected_model_header.bases]
            and [ast_key(keyword) for keyword in model.keywords]
            == [ast_key(keyword) for keyword in expected_model_header.keywords]
            and not model.decorator_list
            and [node.name for node in model.body if isinstance(node, ast.FunctionDef)] == ["__init__", "forward"]
            and len(model.body) == 2
        )
        initializer = next(
            (node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
            None,
        )
        forward = next(
            (node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward"),
            None,
        )
        if initializer is None or forward is None:
            return checks

        init_body = ["super().__init__()"]
        module_lines = [line for node in nodes if (line := generator._module_line(node, shape)) is not None]
        init_body.extend(module_lines or ["self.register_buffer('_anchor', torch.tensor(0.0), persistent=False)"])
        expected_initializer = _expected_function("__init__", "self", init_body)
        checks["model_init_lowering_bound"] = (
            ast_key(initializer.args) == ast_key(expected_initializer.args)
            and [ast_key(statement) for statement in initializer.body]
            == [ast_key(statement) for statement in expected_initializer.body]
            and not initializer.decorator_list
        )

        checks["multiclass_wrapper_bound"] = (
            all(not isinstance(node, ast.ClassDef) or node.name != "TensorBridge" for node in tree.body)
            and checks["semantic_multiclass_policy_bound"]
        )

        has_loss = any(node.rule == "loss" for node in nodes)
        expected_arguments = _expected_function("forward", "self, x, target" if has_loss else "self, x", ["pass"])
        checks["forward_argument_binding"] = (
            ast_key(forward.args) == ast_key(expected_arguments.args) and not forward.decorator_list
        )
        assignments = forward.body[:-1] if forward.body else []
        checks["forward_v_assignment_count_bound"] = len(assignments) == len(nodes)
        target_names: list[str] = []
        for statement in assignments:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                target_names.append(statement.targets[0].id)
            else:
                target_names.append("")
        checks["forward_v_assignment_order_bound"] = target_names == [f"v{node.index}" for node in nodes]
        expected_expressions = [
            expected_expression(
                node,
                ["x" if predecessor == -1 else f"v{predecessor}" for predecessor in node.predecessors],
                rank,
            )
            for node in nodes
        ]
        checks["forward_expression_lowering_bound"] = len(assignments) == len(expected_expressions) and all(
            isinstance(statement, ast.Assign) and ast_key(statement.value) == ast_key(expected)
            for statement, expected in zip(assignments, expected_expressions, strict=True)
        )
        expected_return = ast.parse(f"return v{nodes[-1].index}").body[0]
        checks["forward_return_last_v_bound"] = bool(forward.body) and ast_key(forward.body[-1]) == ast_key(
            expected_return
        )

        expected_input_body = generator._input_expression(shape, manifest["input_mode"])
        if has_loss:
            expected_input_body.extend(["target = torch.randn_like(x)", "return (x, target)"])
        else:
            expected_input_body.append("return (x,)")
        expected_get_inputs = _expected_function("get_inputs", "", expected_input_body)
        checks["get_inputs_lowering_bound"] = (
            ast_key(get_inputs.args) == ast_key(expected_get_inputs.args)
            and [ast_key(statement) for statement in get_inputs.body]
            == [ast_key(statement) for statement in expected_get_inputs.body]
            and not get_inputs.decorator_list
        )

        actual_physical = len(code.splitlines())
        actual_code_only = generator._code_only_physical_line_count(code)
        actual_comment_only = sum(bool(line.strip()) and line.lstrip().startswith("#") for line in code.splitlines())
        physical_bound = manifest.get("complexity", {}).get("source_line_count") == actual_physical
        metrics = manifest.get("code_metrics")
        unformatted_lines = generator._render_code_lines(
            nodes=nodes,
            shape=shape,
            input_mode=manifest["input_mode"],
        )
        unformatted_code_only = generator._code_only_physical_line_count("\n".join(unformatted_lines))
        expected_metrics = {
            "physical_line_count": actual_physical,
            "code_only_physical_line_count": actual_code_only,
            "comment_only_physical_line_count": actual_comment_only,
            "source_format_expansion_lines": actual_code_only - unformatted_code_only,
        }
        checks["code_metrics_bound"] = isinstance(metrics, dict) and metrics == expected_metrics and physical_bound
    except (KeyError, TypeError, ValueError, SyntaxError, AssertionError):
        return checks
    return checks
