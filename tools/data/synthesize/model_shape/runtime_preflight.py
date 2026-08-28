#!/usr/bin/env python3
"""Fail-closed source checks for the pinned DSV4/DSPARK SGLang runtime."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any


def _assignments(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            result[statement.targets[0].id] = statement.value
    return result


def _literal(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_literal(item, names) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        return {
            _literal(key, names): _literal(value, names) for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal(node.left, names) + _literal(node.right, names)
    raise ValueError(f"unsupported runtime literal: {ast.dump(node, include_attributes=False)}")


def _effort_mapping(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    assignments = _assignments(tree)
    resolved: dict[str, Any] = {}
    for name in ("REASONING_EFFORT_HIGH", "REASONING_EFFORT_MAX"):
        if name not in assignments:
            raise ValueError(f"missing {name} in {path}")
        resolved[name] = _literal(assignments[name], resolved)
    raw = assignments.get("REASONING_EFFORT_PROMPTS")
    if raw is None:
        raise ValueError(f"missing REASONING_EFFORT_PROMPTS in {path}")
    mapping = _literal(raw, resolved)
    if not isinstance(mapping, dict) or set(mapping) != {"low", "high", "max"}:
        raise ValueError(f"invalid reasoning effort keys: {mapping!r}")
    if mapping["low"] != "":
        raise ValueError("low effort must be the no-prefix 0731 default")
    if "Absolute maximum" not in mapping["high"]:
        raise ValueError("high effort does not use the 0731 high prefix")
    if "Beyond maximum" not in mapping["max"] or mapping["max"] == mapping["high"]:
        raise ValueError("max effort is not distinct from high")
    required_fragments = (
        'reasoning_effort = reasoning_effort or "low"',
        "REASONING_EFFORT_PROMPTS[reasoning_effort]",
        'thinking_mode == "thinking"',
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise ValueError(f"effort mapping is defined but not applied: {missing}")
    return {key: str(value) for key, value in mapping.items()}


def _verify_dspark_swa(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    eviction = source.find("batch.maybe_evict_swa()")
    plan = source.find("plan_stream, plan_stream_ctx", eviction + 1)
    reserve = source.find("req.kv_allocated_len = max", plan + 1)
    advance = source.find("req.decode_batch_idx += 1", reserve + 1)
    if min(eviction, plan, reserve, advance) < 0:
        raise ValueError("DSPARK spec-v2 SWA repair is incomplete")
    if not eviction < plan < reserve < advance:
        raise ValueError("DSPARK SWA eviction/decode-index repair is in the wrong order")


def verify_source(sglang_root: Path) -> None:
    effort_path = sglang_root / "python/sglang/srt/entrypoints/openai/encoding_dsv4.py"
    dspark_path = sglang_root / "python/sglang/srt/speculative/dflash_info_v2.py"
    mapping = _effort_mapping(effort_path)
    _verify_dspark_swa(dspark_path)
    print(f"PASS reasoning effort mapping: low={mapping['low']!r}, high/max distinct")
    print("PASS DSPARK spec-v2 SWA eviction and decode-index ordering")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-root", type=Path, default=Path("/sgl-workspace/sglang"))
    return parser


if __name__ == "__main__":
    verify_source(_parser().parse_args().sglang_root)
