#!/usr/bin/env python3
"""Audit and filter a KernelBench/DrKernel prompt parquet.

The cleaner is deliberately strict and reproducible:

* schema, prompt/reference consistency, AST/interface, unsafe-code, semantic
  duplicate (including optional reference corpora), unused-input, and
  stochastic-forward checks run first;
* references that pass static checks run in one disposable forked process each;
* a per-row JSONL audit, aggregate JSON summary, and review samples explain every
  removal; and
* the filtered parquet retains the input Arrow schema and row order.

``clean`` and ``recover`` always use GPU runtime validation under the current
production policy. ``static`` performs only the global source-level checks
needed before sharded GPU execution.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import json
import multiprocessing as mp
import os
import re
import signal
import sys
import time
import tokenize
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .ast_similarity import best_ast_similarity_match, read_ast_similarity_baselines
from .runtime_validation import audit_train_eval_modes_detailed, validate_ops_text_detailed
from .similarity import best_token_jaccard_match, read_token_jaccard_baselines
from .static_analysis import _call_name, analyze_reference, canonicalize_reference


@dataclass(frozen=True)
class RuntimeTask:
    row_index: int
    code: str
    entry_point: str
    same_input_repeats: int = 3
    fresh_input_trials: int = 5
    fixed_input_repeats: int = 20
    audit_train_eval: bool = False
    mode_only: bool = False
    validation_seeds: int = 1
    min_natural_output_change_fraction: float = 0.0
    require_each_forward_input_sensitive: bool = False


@dataclass(frozen=True)
class CleanupPolicy:
    """The single production acceptance contract.

    These values affect which rows are retained, so callers cannot override
    them from the command line. Every summary records the resolved policy and
    its hash.
    """

    contract_version: int = 5
    code_key: str = "reward_model.ground_truth"
    entry_point_key: str = "extra_info.entry_point"
    seed: int = 0
    rtol: float = 1e-4
    atol: float = 1e-5
    same_input_repeats: int = 3
    fresh_input_trials: int = 20
    fixed_input_repeats: int = 40
    validation_seeds: int = 3
    min_natural_output_change_fraction: float = 1e-4
    require_each_forward_input_sensitive: bool = True
    audit_train_eval: bool = True
    audit_train_eval_all: bool = True
    reject_unused_forward_args: bool = True
    reject_random_forward: bool = True
    normalize_ops_metadata: bool = True
    token_jaccard_threshold: float = 0.8
    ast_similarity_threshold: float = 0.9
    batch_size: int = 4096
    torch_threads: int = 1
    compression: str = "zstd"
    quarantine_runtime_verdicts: tuple[str, ...] = (
        "forward_argument_sensitivity_inconclusive",
        "natural_output_low_activity",
        "input_sensitivity_inconclusive",
        "synthetic_sensitivity_only",
    )
    quarantine_static_reasons: tuple[str, ...] = (
        "dead_forward_stateful_effect",
        "dead_forward_rng_effect",
        "dead_forward_unknown_effect",
    )
    denied_uuids: tuple[str, ...] = (
        "cuda_llm_163523",
        "cuda_llm_258008",
        "cuda_llm_367573",
    )


CURRENT_POLICY = CleanupPolicy()


@dataclass(frozen=True)
class CleanupConfig:
    command: str
    input: Path
    output: Path
    workers: int = 4
    timeout: float = 60.0
    overwrite: bool = False
    dedup_against: tuple[Path, ...] = ()
    prior_audit: Path | None = None
    rerun_indices_path: Path | None = None
    # Internal test/diagnostic overrides are deliberately absent from the CLI.
    device: str = "cuda"
    max_rows: int | None = None
    policy: CleanupPolicy = CURRENT_POLICY


_REFERENCE_CODE_REPAIR = "__reference_code__"


def _nested(row: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        return ""
    return "\n".join(
        message.get("content", "")
        for message in prompt
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )


def infer_effective_entry_point(code: Any, row: dict[str, Any], declared: Any) -> tuple[Any, dict[str, str]]:
    """Repair helper-class metadata when the actual task contract is ``Model``.

    The source corpus has rows whose metadata points at a helper class (for
    example ``Mish``) even though the prompt requests ``ModelNew`` and the
    reference defines a top-level ``Model``.  KernelGym derives the expected
    response class from this metadata, so leaving it unchanged makes every
    response fail precheck.  This is a lossless metadata repair, not a filter.
    """

    if declared == "Model" or not isinstance(code, str):
        return declared, {}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return declared, {}
    top_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    prompt = _prompt_text(row)
    if "Model" in top_classes and "ModelNew" in prompt and f"{declared}New" not in prompt:
        return "Model", {"entry_point": "Model", "module_name": "Model"}
    return declared, {}


def inspect_row_schema(row: dict[str, Any], code: Any) -> tuple[list[str], list[str]]:
    fatal: list[str] = []
    flags: list[str] = []
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        fatal.append("missing_prompt")
    else:
        contents: list[str] = []
        for message in prompt:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                fatal.append("invalid_prompt_message")
                continue
            contents.append(message["content"])
            if message.get("role") not in {"user", "system"}:
                flags.append(f"unexpected_prompt_role:{message.get('role')}")
        if isinstance(code, str) and code not in "\n".join(contents):
            fatal.append("prompt_reference_mismatch")
    if not isinstance(row.get("ability"), str) or not row.get("ability"):
        fatal.append("missing_ability")
    if _nested(row, "reward_model.style") != "rule":
        flags.append("unexpected_reward_style")
    ops = _nested(row, "extra_info.ops")
    if ops is not None:
        try:
            parsed_ops = json.loads(ops) if isinstance(ops, str) else ops
            if not isinstance(parsed_ops, list) or not all(isinstance(item, str) for item in parsed_ops):
                flags.append("invalid_ops_metadata")
        except (json.JSONDecodeError, TypeError):
            flags.append("invalid_ops_metadata")
    return sorted(set(fatal)), sorted(set(flags))


def _runtime_child(
    sender: Any,
    task: RuntimeTask,
    *,
    device: str,
    seed: int,
    rtol: float,
    atol: float,
    torch_threads: int,
) -> None:
    try:
        # Reference snippets can emit thousands of deprecation warnings.  The
        # structured verdict carries actionable errors, so keep the parent log
        # readable without losing failure diagnostics.
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
        import torch

        torch.set_num_threads(torch_threads)
        if task.mode_only:
            result = audit_train_eval_modes_detailed(
                task.code,
                entry_point=task.entry_point,
                device=device,
                seed=seed,
                rtol=rtol,
                atol=atol,
            )
        else:
            result = validate_ops_text_detailed(
                task.code,
                entry_point=task.entry_point,
                device=device,
                seed=seed,
                rtol=rtol,
                atol=atol,
                same_input_repeats=task.same_input_repeats,
                fresh_input_trials=task.fresh_input_trials,
                fixed_input_repeats=task.fixed_input_repeats,
                validation_seeds=task.validation_seeds,
                min_natural_output_change_fraction=task.min_natural_output_change_fraction,
                require_each_forward_input_sensitive=task.require_each_forward_input_sensitive,
                audit_train_eval=task.audit_train_eval,
            )
        sender.send(result)
    except BaseException as exc:  # noqa: BLE001 - child must report rather than poison its siblings
        detail = " ".join(str(exc).split())
        try:
            sender.send({"verdict": "Failed", "detail": f"{type(exc).__name__}: {detail}"[:1000]})
        except BaseException:
            pass
    finally:
        sender.close()


def run_runtime_tasks(
    tasks: list[RuntimeTask],
    *,
    workers: int,
    timeout: float,
    device: str,
    seed: int,
    rtol: float,
    atol: float,
    torch_threads: int,
) -> dict[int, dict[str, Any]]:
    """Run each reference in its own fork and terminate it at the deadline."""

    if not tasks:
        return {}
    ctx = mp.get_context("fork")
    queue = collections.deque(tasks)
    active: dict[int, tuple[Any, Any, RuntimeTask, float]] = {}
    results: dict[int, dict[str, Any]] = {}
    while queue or active:
        while queue and len(active) < workers:
            task = queue.popleft()
            receiver, sender = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=_runtime_child,
                args=(sender, task),
                kwargs={
                    "device": device,
                    "seed": seed,
                    "rtol": rtol,
                    "atol": atol,
                    "torch_threads": torch_threads,
                },
                daemon=True,
            )
            process.start()
            sender.close()
            active[process.pid] = (process, receiver, task, time.monotonic())

        for pid, (process, receiver, task, started) in list(active.items()):
            result: dict[str, Any] | None = None
            if receiver.poll():
                try:
                    result = receiver.recv()
                except (EOFError, OSError):
                    result = {"verdict": "Failed", "detail": "worker_pipe_closed"}
            elif not process.is_alive():
                result = {
                    "verdict": "Failed",
                    "detail": f"worker_exit:{process.exitcode}",
                }
            elif time.monotonic() - started > timeout:
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                result = {"verdict": "Failed", "detail": f"timeout_after_{timeout:g}s"}
            if result is None:
                continue
            process.join(timeout=2)
            receiver.close()
            results[task.row_index] = result
            del active[pid]
        if active:
            time.sleep(0.02)
    return results


def _normalize_mode_audit_result(result: dict[str, Any]) -> tuple[list[str], str]:
    """Convert a standalone mode-task outcome into non-filtering evidence."""

    if result.get("verdict") == "mode_audit_complete":
        return sorted(set(result.get("mode_flags", ()))), str(result.get("mode_detail", ""))
    detail = str(result.get("detail", "mode audit produced no detail"))
    return ["train_eval_audit_failed", "train_eval_inconclusive"], detail


def _runtime_failure_category(detail: str) -> str:
    """Recover the underlying error from a multi-seed validation prefix."""

    normalized = re.sub(r"^validation seed \d+ failed: ", "", detail)
    if normalized.startswith("timeout_after_"):
        return "timeout"
    return normalized.split(":", 1)[0] or "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _default_sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}.{suffix}")


def _add_common_arguments(parser: argparse.ArgumentParser, *, timeout: float, runtime: bool) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--dedup-against",
        action="append",
        default=[],
        type=Path,
        help="Apply exact, token-Jaccard, and AST similarity checks against this parquet (repeatable).",
    )
    if runtime:
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--timeout", type=float, default=timeout)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    clean_parser = commands.add_parser("clean", help="Run the current complete GPU cleanup contract.")
    _add_common_arguments(clean_parser, timeout=60.0, runtime=True)

    static_parser = commands.add_parser(
        "static",
        help="Run global static checks and write candidates for sharded GPU validation.",
    )
    _add_common_arguments(static_parser, timeout=60.0, runtime=False)

    recover_parser = commands.add_parser(
        "recover",
        help="Reuse a complete audit and rerun selected source rows on GPU.",
    )
    _add_common_arguments(recover_parser, timeout=300.0, runtime=True)
    recover_parser.add_argument("--prior-audit", required=True, type=Path)
    recover_parser.add_argument(
        "--indices",
        type=Path,
        help="Text file containing source row indices to rerun; omitted rows reuse the prior audit.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> CleanupConfig:
    args = build_parser().parse_args(argv)
    return CleanupConfig(
        command=args.command,
        input=args.input,
        output=args.output,
        workers=getattr(args, "workers", 1),
        timeout=getattr(args, "timeout", 60.0),
        overwrite=args.overwrite,
        dedup_against=tuple(args.dedup_against),
        prior_audit=getattr(args, "prior_audit", None),
        rerun_indices_path=getattr(args, "indices", None),
    )


def _read_row_indices(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    indices: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            index = int(text)
        except ValueError as exc:
            raise ValueError(f"invalid row index at {path}:{line_number}: {text!r}") from exc
        if index < 0:
            raise ValueError(f"negative row index at {path}:{line_number}")
        indices.add(index)
    return indices


def _read_runtime_audit(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in runtime audit at line {line_number}") from exc
            row_index = record.get("row_index")
            if not isinstance(row_index, int) or row_index < 0:
                raise ValueError(f"invalid row_index in runtime audit at line {line_number}")
            if row_index in records:
                raise ValueError(f"duplicate row_index {row_index} in runtime audit")
            verdict = record.get("runtime_verdict")
            detail = record.get("runtime_detail")
            if not isinstance(verdict, str) or not isinstance(detail, str):
                raise ValueError(f"invalid runtime verdict at line {line_number}")
            records[row_index] = {
                "uuid": record.get("uuid"),
                "entry_point": record.get("entry_point"),
                "semantic_hash": record.get("semantic_hash"),
                "verdict": verdict,
                "detail": detail,
                "flags": tuple(record.get("flags", [])),
                "mode_flags": tuple(record.get("mode_flags", [])),
                "mode_detail": record.get("mode_detail", ""),
                # Preserve provenance when a corrected audit reuses a prior v2
                # audit.  v1 records predate this field.
                "runtime_source": record.get("runtime_source", "reused_v1"),
                "keep": record.get("keep"),
            }
    return records


def _read_semantic_baselines(
    paths: list[Path],
    *,
    code_key: str,
    entry_point_key: str,
) -> set[tuple[str, str]]:
    """Read exact-dedup keys under the same repairs as candidate rows.

    Raw corpora can declare a helper such as ``Mish`` even when the prompt and
    executable task use ``Model``.  They can also contain shadowed contracts
    that v4 canonicalizes on the candidate side.  Applying only raw metadata
    and raw AST hashing to baselines makes exact duplicates survive the filter.
    """

    keys: set[tuple[str, str]] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        columns = {code_key.split(".", 1)[0], entry_point_key.split(".", 1)[0]}
        if "prompt" in parquet.schema_arrow.names:
            columns.add("prompt")
        for batch in parquet.iter_batches(batch_size=256, columns=sorted(columns), use_threads=False):
            for row in batch.to_pylist():
                code = _nested(row, code_key)
                entry_point = _nested(row, entry_point_key)
                if not isinstance(entry_point, str) or not entry_point:
                    entry_point = "Model"
                if not isinstance(code, str):
                    continue
                try:
                    entry_point, _ = infer_effective_entry_point(code, row, entry_point)
                    code, _ = canonicalize_reference(code, entry_point)
                    tree = ast.parse(code)
                except (SyntaxError, ValueError):
                    continue
                semantic_hash = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
                keys.add((entry_point, semantic_hash))
    return keys


def _ops_metadata(code: str) -> str:
    tree = ast.parse(code)
    calls = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in [_call_name(node.func)]
        if name and name != "super" and not name.startswith("self.")
    }
    return json.dumps(sorted(calls), ensure_ascii=False)


def _model_ops_metadata(code: str, entry_point: str) -> str:
    """Return a deterministic operation inventory for the effective model.

    Source ``extra_info.ops`` labels are generated metadata and can disagree
    with the executable reference (for example ``MaxPool1d`` while ``forward``
    reaches ``MaxPool2d``).  This inventory follows ``forward`` and reachable
    helper methods, maps ``self.module(...)`` calls back to their constructors,
    and excludes top-level input-generation calls.
    """

    tree = ast.parse(code)
    model = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry_point),
        None,
    )
    if model is None:
        return json.dumps([], ensure_ascii=False)
    methods = {node.name: node for node in model.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    forward = methods.get("forward")
    if forward is None:
        return json.dumps([], ensure_ascii=False)

    bindings: dict[str, str] = {}
    operations: set[str] = set()
    initializer = methods.get("__init__")
    if initializer is not None:
        for node in ast.walk(initializer):
            if not isinstance(node, ast.Call):
                continue
            constructor = _call_name(node.func)
            if constructor.startswith(("nn.", "torch.nn.")) and constructor not in {
                "nn.Parameter",
                "torch.nn.Parameter",
            }:
                operations.add(constructor)
        for statement in ast.walk(initializer):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            value = statement.value
            if isinstance(value, ast.Call):
                binding = _call_name(value.func)
            else:
                binding = _call_name(value) if value is not None else ""
            if not binding.startswith(("nn.", "torch.nn.", "torch.", "F.")):
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    bindings[target.attr] = binding
            if binding.startswith(("nn.", "torch.nn.")) and binding not in {
                "nn.Parameter",
                "torch.nn.Parameter",
            }:
                operations.add(binding)

    reachable = {"forward"}
    pending = ["forward"]
    while pending:
        method_name = pending.pop()
        method = methods[method_name]
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name.startswith("self."):
                parts = name.split(".")
                attribute = parts[1] if len(parts) > 1 else ""
                if attribute in methods and attribute not in bindings:
                    if attribute not in reachable:
                        reachable.add(attribute)
                        pending.append(attribute)
                    continue
                binding = bindings.get(attribute)
                if binding and binding not in {"nn.Parameter", "torch.nn.Parameter"}:
                    operations.add(binding)
                if len(parts) > 2 and parts[-1] not in {"dim", "item", "numel", "size", "stride"}:
                    operations.add(f"Tensor.{parts[-1]}")
                continue
            if name.startswith(("torch.", "F.", "nn.", "torch.nn.")):
                operations.add(name)
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr not in {
                "dim",
                "item",
                "numel",
                "size",
                "stride",
            }:
                operations.add(f"Tensor.{node.func.attr}")
    return json.dumps(sorted(operations), ensure_ascii=False)


def _replace_code_in_messages(messages: Any, old_code: str, new_code: str) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            continue
        content = message["content"]
        if old_code in content:
            message["content"] = content.replace(old_code, new_code)


def _apply_reference_code_repair(row: dict[str, Any], new_code: str) -> None:
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict) or not isinstance(reward_model.get("ground_truth"), str):
        raise ValueError("cannot repair missing reward_model.ground_truth")
    old_code = reward_model["ground_truth"]
    reward_model["ground_truth"] = new_code
    _replace_code_in_messages(row.get("prompt"), old_code, new_code)
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        _replace_code_in_messages(extra_info.get("original_prompt"), old_code, new_code)
        extra_info["ops"] = _ops_metadata(new_code)


def _write_filtered_parquet(
    input_path: Path,
    output_path: Path,
    decisions: list[bool],
    repairs: dict[int, dict[str, str]],
    *,
    batch_size: int,
    compression: str,
) -> None:
    parquet = pq.ParquetFile(input_path)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    writer = pq.ParquetWriter(temporary, parquet.schema_arrow, compression=compression)
    offset = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
            mask = decisions[offset : offset + batch.num_rows]
            if len(mask) != batch.num_rows:
                batch = batch.slice(0, len(mask))
            filtered = pc.filter(batch, pa.array(mask, type=pa.bool_()))
            if filtered.num_rows:
                kept_indices = [offset + index for index, keep in enumerate(mask) if keep]
                if any(index in repairs for index in kept_indices):
                    rows = filtered.to_pylist()
                    for source_index, row in zip(kept_indices, rows, strict=True):
                        row_repairs = repairs.get(source_index)
                        if not row_repairs:
                            continue
                        reference_code = row_repairs.get(_REFERENCE_CODE_REPAIR)
                        if reference_code is not None:
                            _apply_reference_code_repair(row, reference_code)
                        extra_info = row.get("extra_info")
                        if not isinstance(extra_info, dict):
                            continue
                        for key, value in row_repairs.items():
                            if key == _REFERENCE_CODE_REPAIR:
                                continue
                            extra_info[key] = value
                    writer.write_table(pa.Table.from_pylist(rows, schema=parquet.schema_arrow))
                else:
                    writer.write_table(pa.Table.from_batches([filtered], schema=parquet.schema_arrow))
            offset += len(mask)
            if offset >= len(decisions):
                break
    finally:
        writer.close()
    os.replace(temporary, output_path)


def run_cleanup(config: CleanupConfig) -> None:
    policy = config.policy
    runtime_enabled = config.command != "static"
    if config.command not in {"clean", "static", "recover"}:
        raise ValueError(f"unknown cleanup command: {config.command!r}")
    if config.command == "recover" and config.prior_audit is None:
        raise ValueError("recover requires a prior audit")
    if config.command != "recover" and (config.prior_audit is not None or config.rerun_indices_path is not None):
        raise ValueError("prior audit and rerun indices are only valid for recover")
    if (
        config.workers < 1
        or policy.torch_threads < 1
        or policy.batch_size < 1
        or config.timeout <= 0
        or policy.same_input_repeats < 1
        or policy.fresh_input_trials < 1
        or policy.fixed_input_repeats < policy.fresh_input_trials + 1
        or policy.validation_seeds < 1
    ):
        raise ValueError(
            "workers, torch-threads, batch-size, timeout, same-input-repeats, and "
            "fresh-input-trials and validation-seeds must be positive; "
            "fixed-input-repeats must cover the initial and fresh calls"
        )
    if not 0.0 <= policy.min_natural_output_change_fraction <= 1.0:
        raise ValueError("min-natural-output-change-fraction must be in [0, 1]")
    if not 0 <= policy.token_jaccard_threshold < 1:
        raise ValueError("token-jaccard-threshold must be in [0, 1)")
    if not 0 <= policy.ast_similarity_threshold < 1:
        raise ValueError("ast-similarity-threshold must be in [0, 1)")
    if policy.contract_version < 1:
        raise ValueError("contract-version must be positive")
    if not config.input.is_file():
        raise FileNotFoundError(config.input)
    if config.output.suffix != ".parquet":
        raise ValueError(f"output must end in .parquet: {config.output}")
    output_paths = {
        config.output,
        _default_sidecar(config.output, "audit.jsonl"),
        _default_sidecar(config.output, "summary.json"),
        _default_sidecar(config.output, "rejected_samples.txt"),
        _default_sidecar(config.output, "quarantine.parquet"),
    }
    protected_inputs = {config.input.resolve()}
    protected_inputs.update(path.resolve() for path in config.dedup_against)
    if config.prior_audit is not None:
        protected_inputs.add(config.prior_audit.resolve())
    collisions = sorted(path for path in output_paths if path.resolve() in protected_inputs)
    if collisions:
        raise ValueError(f"outputs must not overwrite inputs or prior evidence: {collisions}")
    existing = sorted(path for path in output_paths if path.exists())
    if existing and not config.overwrite:
        raise FileExistsError(f"outputs exist (pass --overwrite): {existing[:3]}")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = _default_sidecar(config.output, "audit.jsonl")
    summary_path = _default_sidecar(config.output, "summary.json")
    samples_path = _default_sidecar(config.output, "rejected_samples.txt")
    quarantine_path = _default_sidecar(config.output, "quarantine.parquet")
    for path in (audit_path, summary_path, samples_path, quarantine_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    reused_runtime = _read_runtime_audit(config.prior_audit) if config.prior_audit else None
    rerun_runtime_indices = _read_row_indices(config.rerun_indices_path) if config.rerun_indices_path else set()
    if any(index < 0 for index in rerun_runtime_indices):
        raise ValueError("runtime rerun indices must be nonnegative")
    semantic_baselines = _read_semantic_baselines(
        list(config.dedup_against),
        code_key=policy.code_key,
        entry_point_key=policy.entry_point_key,
    )
    token_jaccard_baselines = read_token_jaccard_baselines(
        list(config.dedup_against),
        code_key=policy.code_key,
    )
    ast_similarity_baselines = read_ast_similarity_baselines(
        list(config.dedup_against),
        code_key=policy.code_key,
        entry_point_key=policy.entry_point_key,
    )
    if config.dedup_against and not semantic_baselines:
        raise ValueError("dedup-against paths produced zero valid semantic baselines")
    if config.dedup_against and not token_jaccard_baselines:
        raise ValueError("dedup-against paths produced zero valid token baselines")
    if config.dedup_against and not ast_similarity_baselines:
        raise ValueError("dedup-against paths produced zero valid AST baselines")
    denied_uuids = set(policy.denied_uuids)

    if runtime_enabled and (reused_runtime is None or rerun_runtime_indices) and config.device == "cpu":
        # Hide busy training GPUs before torch is imported in the fork parent.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        torch.set_num_threads(policy.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    parquet = pq.ParquetFile(config.input)
    columns = [
        "prompt",
        "ability",
        "reward_model.ground_truth",
        "reward_model.style",
        "extra_info.entry_point",
        "extra_info.level",
        "extra_info.module_name",
        "extra_info.ops",
        "extra_info.uuid",
    ]
    decisions: list[bool] = []
    quarantine_decisions: list[bool] = []
    reason_counts: collections.Counter[str] = collections.Counter()
    flag_counts: collections.Counter[str] = collections.Counter()
    repair_counts: collections.Counter[str] = collections.Counter()
    attempted_repair_counts: collections.Counter[str] = collections.Counter()
    runtime_counts: collections.Counter[str] = collections.Counter()
    runtime_failure_categories: collections.Counter[str] = collections.Counter()
    runtime_sensitivity_probe_counts: collections.Counter[str] = collections.Counter()
    train_eval_counts: collections.Counter[str] = collections.Counter()
    executed_runtime_rows = 0
    train_eval_runtime_rows = 0
    primary_counts: collections.Counter[str] = collections.Counter()
    seen_semantic: dict[tuple[str, str], int] = {}
    seen_uuid: dict[str, int] = {}
    rejected_examples: dict[str, list[tuple[int, str, str]]] = collections.defaultdict(list)
    row_repairs: dict[int, dict[str, str]] = {}
    audit_temporary = audit_path.with_name(f".{audit_path.name}.tmp")
    start = time.monotonic()
    last_progress = start
    row_index = 0

    with audit_temporary.open("w", encoding="utf-8") as audit_handle:
        for batch in parquet.iter_batches(batch_size=policy.batch_size, columns=columns, use_threads=False):
            batch_rows = batch.to_pylist()
            if config.max_rows is not None:
                remaining = config.max_rows - row_index
                if remaining <= 0:
                    break
                batch_rows = batch_rows[:remaining]

            staged: list[dict[str, Any]] = []
            runtime_tasks: list[RuntimeTask] = []
            mode_tasks: list[RuntimeTask] = []
            for row in batch_rows:
                source_code = _nested(row, policy.code_key)
                declared_entry_point = _nested(row, policy.entry_point_key, "Model")
                entry_point, repairs = infer_effective_entry_point(source_code, row, declared_entry_point)
                code, canonical_flags = canonicalize_reference(source_code, entry_point)
                validation_row = row
                if code != source_code:
                    validation_row = copy.deepcopy(row)
                    _apply_reference_code_repair(validation_row, code)
                    repairs[_REFERENCE_CODE_REPAIR] = code
                    identity = f"{entry_point}\n{code}".encode()
                    repairs["uuid"] = f"clean_{hashlib.sha256(identity).hexdigest()[:20]}"
                    if reused_runtime is not None:
                        rerun_runtime_indices.add(row_index)
                schema_fatal, schema_flags = inspect_row_schema(validation_row, code)
                static = analyze_reference(
                    code,
                    entry_point,
                    reject_unused_forward_args=policy.reject_unused_forward_args,
                    reject_random_forward=policy.reject_random_forward,
                )
                fatal = list(schema_fatal) + list(static.fatal_reasons)
                flags = list(schema_flags) + list(static.flags) + list(canonical_flags)
                if policy.normalize_ops_metadata and isinstance(code, str) and isinstance(entry_point, str):
                    normalized_ops = _model_ops_metadata(code, entry_point)
                    if normalized_ops != _nested(validation_row, "extra_info.ops"):
                        repairs["ops"] = normalized_ops
                        flags.append("normalized_ops_metadata")
                uuid = _nested(row, "extra_info.uuid")
                effective_uuid = repairs.get("uuid", uuid)
                if isinstance(effective_uuid, str) and effective_uuid:
                    if effective_uuid in seen_uuid:
                        flags.append("duplicate_uuid")
                        identity = f"{entry_point}\n{code}".encode()
                        repaired_uuid = f"clean_{hashlib.sha256(identity).hexdigest()[:20]}"
                        suffix = 1
                        candidate = repaired_uuid
                        while candidate in seen_uuid:
                            suffix += 1
                            candidate = f"{repaired_uuid}_{suffix}"
                        repairs["uuid"] = candidate
                        seen_uuid[candidate] = row_index
                    else:
                        seen_uuid[effective_uuid] = row_index
                else:
                    flags.append("missing_uuid")
                    identity = f"{entry_point}\n{code}".encode()
                    repaired_uuid = f"clean_{hashlib.sha256(identity).hexdigest()[:20]}"
                    repairs["uuid"] = repaired_uuid
                    seen_uuid[repaired_uuid] = row_index
                if entry_point != declared_entry_point:
                    flags.append("repaired_entry_point")
                if static.semantic_hash is not None and isinstance(entry_point, str):
                    semantic_key = (entry_point, static.semantic_hash)
                    if semantic_key in semantic_baselines:
                        fatal.append("semantic_duplicate_against")
                        flags.append("duplicate_against_reference_dataset")
                    elif semantic_key in seen_semantic:
                        fatal.append("semantic_duplicate")
                        flags.append(f"duplicate_of_row:{seen_semantic[semantic_key]}")
                    else:
                        seen_semantic[semantic_key] = row_index
                if not fatal and isinstance(code, str) and token_jaccard_baselines:
                    try:
                        near_match = best_token_jaccard_match(
                            code,
                            token_jaccard_baselines,
                            threshold=policy.token_jaccard_threshold,
                        )
                    except (IndentationError, SyntaxError, tokenize.TokenError):
                        near_match = None
                    if near_match is not None:
                        fatal.append("token_jaccard_duplicate_against")
                        flags.append(
                            "token_jaccard_duplicate_against:"
                            f"{near_match.dataset_index}:{near_match.row_index}:"
                            f"{near_match.similarity:.6f}"
                        )
                if not fatal and isinstance(code, str) and isinstance(entry_point, str) and ast_similarity_baselines:
                    try:
                        ast_match = best_ast_similarity_match(
                            code,
                            entry_point,
                            ast_similarity_baselines,
                            threshold=policy.ast_similarity_threshold,
                        )
                    except (SyntaxError, ValueError, RecursionError):
                        ast_match = None
                    if ast_match is not None:
                        fatal.append("ast_similarity_duplicate_against")
                        flags.append(
                            "ast_similarity_duplicate_against:"
                            f"{ast_match.dataset_index}:{ast_match.row_index}:"
                            f"{ast_match.similarity:.6f}"
                        )
                effective_uuid = repairs.get("uuid", uuid)
                if effective_uuid in denied_uuids:
                    fatal.append("explicit_uuid_denylist")
                    flags.append(f"explicit_uuid_denylist:{effective_uuid}")
                static_train_eval_candidate = any(flag.startswith("train_eval_static_candidate:") for flag in flags)
                runtime_rerun = row_index in rerun_runtime_indices
                prior_retained = bool(reused_runtime is not None and reused_runtime.get(row_index, {}).get("keep"))
                train_eval_candidate = static_train_eval_candidate or bool(
                    policy.audit_train_eval_all and (reused_runtime is None or prior_retained or runtime_rerun)
                )
                mode_rerun = bool(
                    runtime_enabled and policy.audit_train_eval and train_eval_candidate and reused_runtime is None
                )
                if runtime_rerun:
                    flags.append("runtime_rerun_requested_by_prior_audit")
                if reused_runtime is not None and not fatal:
                    prior_runtime = reused_runtime.get(row_index)
                    if prior_runtime is None:
                        raise ValueError(f"runtime audit is missing row {row_index}")
                    if prior_runtime["verdict"] in {"skipped_by_option", "skipped_static_reject"}:
                        runtime_rerun = True
                        flags.append("runtime_rerun_newly_admissible")
                    if policy.audit_train_eval_all and runtime_rerun:
                        train_eval_candidate = True
                    if (
                        runtime_enabled
                        and policy.audit_train_eval
                        and train_eval_candidate
                        and (
                            runtime_rerun
                            or not prior_runtime.get("mode_flags")
                            or code != source_code
                            or (
                                prior_retained
                                and set(prior_runtime.get("mode_flags", ()))
                                & {"train_eval_inconclusive", "train_eval_audit_failed"}
                            )
                        )
                    ):
                        mode_rerun = True
                        flags.append("runtime_rerun_train_eval_candidate")
                staged.append(
                    {
                        "row_index": row_index,
                        "uuid": uuid,
                        "entry_point": entry_point,
                        "declared_entry_point": declared_entry_point,
                        "code": code,
                        "fatal": sorted(set(fatal)),
                        "flags": sorted(set(flags)),
                        "semantic_hash": static.semantic_hash,
                        "repairs": repairs,
                        "runtime_rerun": runtime_rerun,
                        "mode_rerun": mode_rerun,
                        "train_eval_candidate": train_eval_candidate,
                        "code_repaired": code != source_code,
                    }
                )
                if not fatal and runtime_enabled and (reused_runtime is None or runtime_rerun):
                    runtime_tasks.append(
                        RuntimeTask(
                            row_index,
                            code,
                            entry_point,
                            policy.same_input_repeats,
                            policy.fresh_input_trials,
                            policy.fixed_input_repeats,
                            False,
                            False,
                            policy.validation_seeds,
                            policy.min_natural_output_change_fraction,
                            policy.require_each_forward_input_sensitive,
                        )
                    )
                    executed_runtime_rows += 1
                if not fatal and runtime_enabled and mode_rerun:
                    mode_tasks.append(
                        RuntimeTask(
                            row_index,
                            code,
                            entry_point,
                            policy.same_input_repeats,
                            policy.fresh_input_trials,
                            policy.fixed_input_repeats,
                            False,
                            True,
                            1,
                            0.0,
                            False,
                        )
                    )
                    train_eval_runtime_rows += 1
                row_index += 1

            executed_results = run_runtime_tasks(
                runtime_tasks,
                workers=config.workers,
                timeout=config.timeout,
                device=config.device,
                seed=policy.seed,
                rtol=policy.rtol,
                atol=policy.atol,
                torch_threads=policy.torch_threads,
            )
            executed_mode_results = run_runtime_tasks(
                mode_tasks,
                workers=config.workers,
                timeout=config.timeout,
                device=config.device,
                seed=policy.seed,
                rtol=policy.rtol,
                atol=policy.atol,
                torch_threads=policy.torch_threads,
            )
            if reused_runtime is None:
                runtime_results = executed_results
            else:
                runtime_results = dict(executed_results)
                for record in staged:
                    if record["runtime_rerun"]:
                        continue
                    source = reused_runtime.get(record["row_index"])
                    if source is None:
                        raise ValueError(f"runtime audit is missing row {record['row_index']}")
                    identity = (record["uuid"], record["entry_point"], record["semantic_hash"])
                    source_identity = (source["uuid"], source["entry_point"], source["semantic_hash"])
                    if identity != source_identity:
                        raise ValueError(
                            f"runtime audit identity mismatch at row {record['row_index']}: "
                            f"current={identity!r} audit={source_identity!r}"
                        )
                    if not record["fatal"] and source["verdict"] in {
                        "skipped_by_option",
                        "skipped_static_reject",
                    }:
                        raise ValueError(
                            f"runtime audit has no reusable verdict for passing row {record['row_index']}"
                        )
                    runtime_results[record["row_index"]] = {
                        "verdict": source["verdict"],
                        "detail": source["detail"],
                        "mode_flags": list(source.get("mode_flags", ())),
                        "mode_detail": source.get("mode_detail", ""),
                    }
            for record in staged:
                runtime = runtime_results.get(record["row_index"])
                if runtime is None:
                    runtime = {
                        "verdict": "skipped_static_reject" if record["fatal"] else "skipped_by_option",
                        "detail": "",
                        "mode_flags": [],
                        "mode_detail": "",
                    }
                if record["mode_rerun"]:
                    mode_result = executed_mode_results.get(
                        record["row_index"],
                        {"verdict": "Failed", "detail": "mode_audit_result_missing"},
                    )
                    mode_flags, mode_detail = _normalize_mode_audit_result(mode_result)
                elif reused_runtime is not None:
                    prior_mode = reused_runtime[record["row_index"]]
                    mode_flags = list(prior_mode.get("mode_flags", ()))
                    mode_detail = prior_mode.get("mode_detail", "")
                else:
                    mode_flags, mode_detail = [], ""
                runtime = {
                    "verdict": runtime["verdict"],
                    "detail": runtime["detail"],
                    "mode_flags": mode_flags,
                    "mode_detail": mode_detail,
                }
                if (
                    (record["runtime_rerun"] or reused_runtime is None)
                    and runtime["verdict"] == "different_input_not_changed"
                    and "forward_output_dependency:dependent" in record["flags"]
                ):
                    runtime = {
                        "verdict": "input_sensitivity_inconclusive",
                        "detail": "all multi-probe outputs matched; AST confirms input-value data flow",
                        "mode_flags": list(runtime.get("mode_flags", ())),
                        "mode_detail": runtime.get("mode_detail", ""),
                    }
                    record["flags"].append("runtime_input_sensitivity_inconclusive")
                mode_flags = sorted(set(runtime.get("mode_flags", ())))
                record["flags"].extend(mode_flags)
                train_eval_counts.update(mode_flags)
                runtime_quarantined = runtime["verdict"] in set(policy.quarantine_runtime_verdicts)
                if runtime_quarantined:
                    record["fatal"].append(f"quarantined_runtime_verdict:{runtime['verdict']}")
                    record["flags"].append("runtime_verdict_quarantined")
                static_quarantine_reasons = sorted(set(record["fatal"]) & set(policy.quarantine_static_reasons))
                static_quarantined = bool(static_quarantine_reasons)
                if static_quarantined:
                    record["flags"].extend(
                        f"static_reason_quarantined:{reason}" for reason in static_quarantine_reasons
                    )
                quarantined = runtime_quarantined or static_quarantined
                runtime_counts[runtime["verdict"]] += 1
                if runtime["verdict"] == "Failed":
                    runtime_failure_categories[_runtime_failure_category(runtime["detail"])] += 1
                elif runtime["verdict"] == "synthetic_sensitivity_only":
                    match = re.search(r"synthetic probe ([^ ]+)", runtime["detail"])
                    category = match.group(1) if match else "per_argument"
                    runtime_sensitivity_probe_counts[category] += 1
                if runtime["verdict"] not in {
                    "passed",
                    "input_sensitivity_inconclusive",
                    "synthetic_sensitivity_only",
                    "skipped_by_option",
                    "skipped_static_reject",
                }:
                    record["fatal"].append(f"runtime:{runtime['verdict']}")
                record["fatal"] = sorted(set(record["fatal"]))
                record["flags"] = sorted(set(record["flags"]))
                keep = not record["fatal"]
                decisions.append(keep)
                quarantine_decisions.append(quarantined)
                if keep and record["repairs"]:
                    row_repairs[record["row_index"]] = record["repairs"]
                    repair_counts.update(
                        "reference_code" if key == _REFERENCE_CODE_REPAIR else key for key in record["repairs"]
                    )
                attempted_repair_counts.update(
                    "reference_code" if key == _REFERENCE_CODE_REPAIR else key for key in record["repairs"]
                )
                for reason in record["fatal"]:
                    reason_counts[reason] += 1
                for flag in record["flags"]:
                    flag_counts[flag.split(":", 1)[0]] += 1
                primary = record["fatal"][0] if record["fatal"] else "kept"
                primary_counts[primary] += 1
                if not keep and len(rejected_examples[primary]) < 5:
                    rejected_examples[primary].append(
                        (record["row_index"], str(record["uuid"]), str(record["code"])[:4000])
                    )
                audit_record = {
                    "row_index": record["row_index"],
                    "uuid": record["uuid"],
                    "entry_point": record["entry_point"],
                    "declared_entry_point": record["declared_entry_point"],
                    "keep": keep,
                    "quarantined": quarantined,
                    "reasons": record["fatal"],
                    "flags": record["flags"],
                    "semantic_hash": record["semantic_hash"],
                    "runtime_verdict": runtime["verdict"],
                    "runtime_detail": runtime["detail"],
                    "mode_flags": mode_flags,
                    "mode_detail": runtime.get("mode_detail", ""),
                    "repairs": {
                        ("reference_code" if key == _REFERENCE_CODE_REPAIR else key): (
                            "canonicalized" if key == _REFERENCE_CODE_REPAIR else value
                        )
                        for key, value in record["repairs"].items()
                    },
                    "runtime_source": (
                        f"fresh_v{policy.contract_version}"
                        if reused_runtime is None
                        else (
                            f"rerun_v{policy.contract_version}"
                            if record["runtime_rerun"]
                            else reused_runtime[record["row_index"]]["runtime_source"]
                        )
                    ),
                }
                audit_handle.write(json.dumps(audit_record, ensure_ascii=False, sort_keys=True) + "\n")

            now = time.monotonic()
            if now - last_progress >= 30:
                print(
                    f"audited {len(decisions)}/"
                    f"{min(parquet.metadata.num_rows, config.max_rows or parquet.metadata.num_rows)} "
                    f"rows; kept={sum(decisions)} elapsed={now - start:.1f}s",
                    flush=True,
                )
                last_progress = now
            if config.max_rows is not None and row_index >= config.max_rows:
                break
    if reused_runtime is not None and config.max_rows is None and len(reused_runtime) != len(decisions):
        raise ValueError(
            f"runtime audit row count {len(reused_runtime)} does not match input row count {len(decisions)}"
        )
    os.replace(audit_temporary, audit_path)

    _write_filtered_parquet(
        config.input,
        config.output,
        decisions,
        row_repairs,
        batch_size=max(policy.batch_size, 512),
        compression=policy.compression,
    )
    _write_filtered_parquet(
        config.input,
        quarantine_path,
        quarantine_decisions,
        {},
        batch_size=max(policy.batch_size, 512),
        compression=policy.compression,
    )
    with samples_path.open("w", encoding="utf-8") as handle:
        for primary in sorted(rejected_examples):
            for index, uuid, code in rejected_examples[primary]:
                handle.write("=" * 88 + "\n")
                handle.write(f"primary_reason={primary} row_index={index} uuid={uuid}\n")
                handle.write("-" * 88 + "\n")
                handle.write(code.rstrip() + "\n\n")

    output_rows = pq.ParquetFile(config.output).metadata.num_rows
    resolved_policy = asdict(policy)
    policy_sha256 = hashlib.sha256(
        json.dumps(resolved_policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "contract_version": policy.contract_version,
        "contract_sha256": policy_sha256,
        "input": str(config.input.resolve()),
        "input_sha256": _sha256_file(config.input),
        "output": str(config.output.resolve()),
        "output_sha256": _sha256_file(config.output),
        "input_rows_considered": len(decisions),
        "output_rows": output_rows,
        "rejected_rows": len(decisions) - output_rows,
        "quarantined_rows": sum(quarantine_decisions),
        "retention_rate": output_rows / len(decisions) if decisions else 0.0,
        "reason_counts_nonexclusive": dict(sorted(reason_counts.items())),
        "primary_reason_counts": dict(sorted(primary_counts.items())),
        "flag_counts_nonexclusive": dict(sorted(flag_counts.items())),
        "repair_counts": dict(sorted(repair_counts.items())),
        "attempted_repair_counts": dict(sorted(attempted_repair_counts.items())),
        "runtime_verdict_counts": dict(sorted(runtime_counts.items())),
        "runtime_failure_categories": dict(sorted(runtime_failure_categories.items())),
        "runtime_sensitivity_probe_counts": dict(sorted(runtime_sensitivity_probe_counts.items())),
        "train_eval_counts_nonexclusive": dict(sorted(train_eval_counts.items())),
        "configuration": {
            "command": config.command,
            "policy": resolved_policy,
            "policy_sha256": policy_sha256,
            "runtime_enabled": runtime_enabled,
            "runtime_reused_from": str(config.prior_audit.resolve()) if config.prior_audit else None,
            "runtime_reused_audit_sha256": _sha256_file(config.prior_audit) if config.prior_audit else None,
            "runtime_rerun_indices": (str(config.rerun_indices_path.resolve()) if config.rerun_indices_path else None),
            "runtime_rerun_indices_sha256": (
                _sha256_file(config.rerun_indices_path) if config.rerun_indices_path else None
            ),
            "runtime_rerun_rows_requested": len(rerun_runtime_indices),
            "runtime_rows_executed": executed_runtime_rows,
            "train_eval_rows_executed": train_eval_runtime_rows,
            "dedup_against": [
                {"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in config.dedup_against
            ],
            "token_jaccard_baseline_rows": len(token_jaccard_baselines),
            "semantic_baseline_keys": len(semantic_baselines),
            "token_jaccard_threshold_strictly_greater_than": policy.token_jaccard_threshold,
            "ast_similarity_baseline_rows": len(ast_similarity_baselines),
            "ast_similarity_threshold_strictly_greater_than": policy.ast_similarity_threshold,
            "device": config.device if runtime_enabled else None,
            "workers": config.workers if runtime_enabled else None,
            "timeout_seconds": config.timeout if runtime_enabled else None,
            "max_rows": config.max_rows,
            "semantic_deduplication": True,
            "canonicalize_shadowed_contracts": True,
            "canonicalize_random_init_scalars": True,
            "sensitivity_policy": (
                f"{policy.validation_seeds}_seeds_x_{policy.fresh_input_trials}_natural_fresh_"
                f"min_leaf_change_{policy.min_natural_output_change_fraction:g}_then_verified_"
                "scale_joint_range_zero_nonfinite_and_per_argument_multi_probe"
                + ("_require_each_forward_input" if policy.require_each_forward_input_sensitive else "")
            ),
            "denied_uuids": sorted(denied_uuids),
        },
        "artifacts": {
            "audit_jsonl": str(audit_path.resolve()),
            "rejected_samples": str(samples_path.resolve()),
            "quarantine_parquet": str(quarantine_path.resolve()),
            "quarantine_parquet_sha256": _sha256_file(quarantine_path),
        },
        "elapsed_seconds": time.monotonic() - start,
    }
    temporary_summary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_summary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    run_cleanup(parse_args(argv))


if __name__ == "__main__":
    # Make Ctrl-C terminate the parent; children are daemonic and cannot outlive it.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
