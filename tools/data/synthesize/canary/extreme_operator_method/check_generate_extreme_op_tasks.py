#!/usr/bin/env python3
"""Offline contract check for the extreme-operator canary generator."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.canary.extreme_operator_method.generate_extreme_op_tasks import (  # noqa: E402
    DEFAULT_MAX_INPUT_NUMEL,
    generate_tasks,
)


def main() -> None:
    tasks = generate_tasks(atomic_count=64, long_count=64, seed=17)
    repeated = generate_tasks(atomic_count=64, long_count=64, seed=17)
    assert [task.uuid for task in tasks] == [task.uuid for task in repeated]
    assert len({task.uuid for task in tasks}) == 128
    assert len({task.reference_sha256 for task in tasks}) == 128
    assert len({task.normalized_ast_sha256 for task in tasks}) == 128
    assert all(task.input_numel_upper_bound <= DEFAULT_MAX_INPUT_NUMEL for task in tasks)
    assert all(len(task.operator_signature) == 1 for task in tasks[:64])
    assert all(len(task.operator_signature) >= 6 for task in tasks[64:])

    topologies = {task.config["topology"] for task in tasks[64:]}
    assert topologies == {"branch_merge", "chain", "random_dag", "residual"}
    for task in tasks:
        tree = ast.parse(task.code)
        names = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        assert {"Model", "get_inputs", "get_init_inputs"}.issubset(names)
        if task.config.get("topology") != "random_dag":
            continue
        forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward")
        assigned = {
            target.id
            for node in ast.walk(forward)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id.startswith("v")
        }
        loaded = {
            node.id for node in ast.walk(forward) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert assigned <= loaded, task.uuid

    print(json.dumps({"rows": len(tasks), "topologies": sorted(topologies), "status": "passed"}))


if __name__ == "__main__":
    main()
