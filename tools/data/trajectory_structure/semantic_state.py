"""Reference-aligned, evidence-bearing state cards for a manually reviewed pilot.

No generated code is executed. An operation mapping is supplied by a reviewed
task adapter, never inferred from a function's name. Address checks concern
individual GEMM descriptors, not numerical correctness of the whole operator.
"""

from __future__ import annotations

import ast
import itertools
import re


def integer_expr(expression, values):
    expression = re.sub(r"\((?:int64_t|int|long long|long|size_t)\)", "", expression).strip()
    root = ast.parse(expression, mode="eval").body

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, (ast.Div, ast.FloorDiv)) and b and a % b == 0:
                return a // b
        raise ValueError(f"unsupported dimension expression: {expression}")

    return visit(root)


def integer_locals_before_call(owner, call, arguments):
    """Resolve simple integer declarations in the reviewed source prefix.

    This helper is restricted to the pilot's straight-line declarations. It is
    not a C++ interpreter or a resolver for conditional assignments/shadowing.
    """
    values = dict(arguments)
    prefix = "\n".join(owner["source"].splitlines()[: call["line"] - owner["line"]])
    pattern = r"(?m)^\s*(?:const\s+)?(?:int64_t|int|long long|long|size_t)\s+(\w+)\s*=\s*([^;]+);"
    for name, expression in re.findall(pattern, prefix):
        try:
            values[name] = integer_expr(expression, values)
        except (ValueError, KeyError, SyntaxError, ZeroDivisionError):
            values.pop(name, None)
    return values


def bind_scalar_arguments(owner, call, positions, enclosing_values):
    values = integer_locals_before_call(owner, call, enclosing_values)
    bound = dict(values)
    for formal, index in positions.items():
        try:
            bound[formal] = integer_expr(actual_arguments(call)[index], values)
        except (ValueError, KeyError, SyntaxError, IndexError):
            bound.pop(formal, None)
    return bound


def check_mask_stride(expression, values, reference_stride):
    try:
        origin = integer_expr(expression, values | {"i": 0, "j": 0})
        row = integer_expr(expression, values | {"i": 1, "j": 0}) - origin
        column = integer_expr(expression, values | {"i": 0, "j": 1}) - origin
    except (ValueError, KeyError, SyntaxError):
        return {"status": "unknown", "observed_index_expression": expression, "observed_row_stride": None}
    return {
        "status": "checked_points_consistent" if (origin, row, column) == (0, reference_stride, 1) else "conflict",
        "observed_index_expression": expression,
        "observed_row_stride": row,
        "expected_row_stride": reference_stride,
        "scope": "mask_address_contract_only",
    }


def component(turn, section, name):
    matches = [
        c
        for c in turn["components"]
        if c["section"] == section
        and c["name"] == name
        and c["kind"] in {"function", "kernel", "python_function", "method"}
    ]
    # Prototypes have no body_hash; never use one as evidence of implementation.
    matches = [c for c in matches if c.get("body_hash") or "{" in c["source"] or "def " in c["source"]]
    if len(matches) != 1:
        raise ValueError(f"expected one implementation: {section}:{name}, got {len(matches)}")
    return matches[0]


def call_sites(owner, target):
    return [c | {"arguments": actual_arguments(c)} for c in owner["calls"] if c["target"] == target]


def actual_arguments(call):
    # Historical tree-sitter artifacts may retain comment nodes as named children.
    return [a for a in call["arguments"] if not a.lstrip().startswith(("//", "/*"))]


def evidence(owner, call=None):
    return {
        "section": owner["section"],
        "component": owner["qualified_name"],
        "line": owner["line"] if call is None else call["line"],
        "end_line": owner["end_line"],
        "component_syntax_sha256": owner["syntax_hash"],
        "source": (
            owner["source"]
            if call is None
            else {
                "target": call["target"],
                "arguments": actual_arguments(call),
                "launch_config": call.get("launch_config"),
            }
        ),
    }


def gemm_descriptor(call, values, buffer_roles):
    args = actual_arguments(call)
    batched = call["target"] == "cublasSgemmStridedBatched"
    if call["target"] not in {"cublasSgemm", "cublasSgemmStridedBatched"}:
        return {"status": "unknown", "reason": "unsupported_api"}
    indices = {
        "A": 7,
        "lda": 8,
        "B": 10 if batched else 9,
        "ldb": 11 if batched else 10,
        "C": 14 if batched else 12,
        "ldc": 15 if batched else 13,
        "m": 3,
        "n": 4,
        "k": 5,
    }
    try:
        result = {k: integer_expr(args[v], values) for k, v in indices.items() if k not in {"A", "B", "C"}}
        result.update({k: buffer_roles[args[indices[k]].strip()] for k in ("A", "B", "C")})
        result.update(transa=args[1], transb=args[2], status="resolved", batched=batched)
        if batched:
            result.update(
                strideA=integer_expr(args[9], values),
                strideB=integer_expr(args[12], values),
                strideC=integer_expr(args[16], values),
                batch_count=integer_expr(args[17], values),
            )
        return result
    except (KeyError, ValueError, SyntaxError, IndexError) as exc:
        return {"status": "unknown", "reason": str(exc), "raw_arguments": args}


def check_gemm_indices(descriptor, operation, M, N, K, expected_batch=None):
    """Check row-major reference operands against column-major API address formulas.

    Exhaustive on small dimensions; otherwise checks corners and an interior point.
    A mismatch is an explicit address-contract counterexample. A pass only certifies
    these descriptor/address checks, NOT math mode, alpha/beta, bias or runtime.
    """
    d = descriptor
    if d["status"] != "resolved":
        return {"status": "unknown", "reason": "unresolved_descriptor"}
    if d["transa"] not in {"CUBLAS_OP_T", "CUBLAS_OP_N"} or d["transb"] not in {"CUBLAS_OP_T", "CUBLAS_OP_N"}:
        return {"status": "unknown", "reason": "unsupported_transpose"}
    if d.get("batched") and (
        d["batch_count"] <= 0 or expected_batch is not None and d["batch_count"] != expected_batch
    ):
        return {
            "status": "conflict",
            "reason": "batch_count",
            "observed": d["batch_count"],
            "expected": expected_batch,
        }
    if min(d[x] for x in ("m", "n", "k", "lda", "ldb", "ldc")) <= 0:
        return {"status": "conflict", "reason": "nonpositive_extent"}
    required = {
        "lda": d["m"] if d["transa"] == "CUBLAS_OP_N" else d["k"],
        "ldb": d["k"] if d["transb"] == "CUBLAS_OP_N" else d["n"],
        "ldc": d["m"],
    }
    for field, minimum in required.items():
        if d[field] < minimum:
            return {
                "status": "conflict",
                "reason": "leading_dimension",
                "field": field,
                "observed": d[field],
                "required_minimum": minimum,
            }
    if d["k"] != K or d["m"] * d["n"] != M * N:
        return {"status": "conflict", "reason": "logical_extent"}
    expected_C = "Y" if operation == "linear" else "P" if operation == "qk" else "O"
    if d["C"] != expected_C:
        return {"status": "conflict", "reason": "output_role", "expected": expected_C, "observed": d["C"]}

    def points(n):
        return list(range(n)) if n <= 5 else sorted({0, 1, n // 2, n - 1})

    checked = 0
    for i, j, k in itertools.product(points(M), points(N), points(K)):
        address = i * N + j
        r, c = address % d["ldc"], address // d["ldc"]
        if r >= d["m"] or c >= d["n"]:
            return {"status": "conflict", "reason": "output_address_unwritten", "index": [i, j]}
        ai = r + k * d["lda"] if d["transa"] == "CUBLAS_OP_N" else k + r * d["lda"]
        bi = k + c * d["ldb"] if d["transb"] == "CUBLAS_OP_N" else c + k * d["ldb"]
        actual = sorted([(d["A"], ai), (d["B"], bi)])
        if operation == "linear":
            expected = sorted([("X", i * K + k), ("W", j * K + k)])
        elif operation == "qk":
            expected = sorted([("Q", i * K + k), ("K", j * K + k)])
        elif operation == "pv":
            expected = sorted([("P", i * K + k), ("V", k * N + j)])
        else:
            raise ValueError(operation)
        if actual != expected:
            return {
                "status": "conflict",
                "reason": "operand_address",
                "index": [i, j, k],
                "expected_factors": expected,
                "actual_factors": actual,
            }
        checked += 1
    if d.get("batched"):
        expected_sizes = {"Q": M * K, "K": N * K, "P": M * K if operation == "pv" else M * N, "V": K * N, "O": M * N}
        for role in ("A", "B", "C"):
            if d[f"stride{role}"] != expected_sizes[d[role]]:
                return {
                    "status": "conflict",
                    "reason": "batch_stride",
                    "operand": role,
                    "observed": d[f"stride{role}"],
                    "expected": expected_sizes[d[role]],
                }
    return {
        "status": "checked_points_consistent",
        "checked_factor_addresses": checked,
        "scope": "address_contract_only_not_operator_correctness",
    }


def normalize_descriptor(d):
    if d["status"] != "resolved":
        return {"status": "unknown"}
    return d


def node(slot, method, inputs, output, sources, **details):
    return {
        "slot": slot,
        "implementation": method,
        "reference_inputs": inputs,
        "reference_output": output,
        "evidence": sources,
        "local_correctness": "unknown_not_tested_separately",
        **details,
    }


def profile(card):
    """Exclude code text/names, locations, hashes and evaluation outcome from shape grouping."""
    return [
        {
            k: n.get(k)
            for k in (
                "slot",
                "implementation",
                "candidate_activation_inputs",
                "candidate_parameter_inputs",
                "reference_output",
            )
        }
        for n in card["nodes"]
    ]


def validate_graph(nodes, external_inputs):
    """Validate reference-slot graph consistency, not implementation correctness."""
    available = set(external_inputs)
    slots = set()
    for n in nodes:
        if n["slot"] in slots:
            raise ValueError("duplicate semantic slot")
        slots.add(n["slot"])
        missing = set(n["reference_inputs"]) - available
        if missing:
            raise ValueError(f"unconnected semantic inputs at {n['slot']}: {sorted(missing)}")
        if not n["evidence"]:
            raise ValueError("semantic node without source evidence")
        output = n["reference_output"]
        available.update([output] if isinstance(output, str) else output)
    if "output" not in available:
        raise ValueError("graph does not reach output")


def propagate_contract_warnings(nodes):
    """Propagate observed static conflicts along mapped activation dependencies.

    This marks conditional upstream reliability, not local numeric correctness.
    Producer roles represent mapped intent; successful runtime writes are unknown.
    """
    values = {}
    for n in nodes:
        inputs = n.get("candidate_activation_inputs")
        inherited = set().union(*(values.get(r, set()) for r in inputs)) if inputs else set()
        n["upstream_contract_conflicts"] = sorted(inherited)
        n["dependency_evidence"] = "unknown_wiring" if inputs is None else "mapped_dependency_only"
        own = any(
            n.get(field, {}).get("status") == "conflict" for field in ("index_contract", "mask_storage_contract")
        )
        output = n["reference_output"]
        for role in [output] if isinstance(output, str) else output:
            values[role] = inherited | ({n["slot"]} if own else set())


def semantic_value(value):
    if isinstance(value, dict):
        return {
            k: semantic_value(v)
            for k, v in value.items()
            if k not in {"evidence", "source", "actual_input", "actual_output", "main_output", "bias_input"}
        }
    if isinstance(value, list):
        return [semantic_value(v) for v in value]
    return value


def compare_cards(a, b):
    if a["reference_sha256"] != b["reference_sha256"]:
        return {"relation": "different_task"}
    if not a["mapping_complete"] or not b["mapping_complete"]:
        return {"relation": "insufficient_mapping", "differences": []}
    differences, shared_address_slots = [], []
    left, right = {n["slot"]: n for n in a["nodes"]}, {n["slot"]: n for n in b["nodes"]}
    for slot in sorted(left.keys() | right.keys()):
        x, y = left.get(slot), right.get(slot)
        if x is None or y is None:
            differences.append({"slot": slot, "field": "operation_presence"})
            continue
        if (
            x.get("gemm_descriptor") == y.get("gemm_descriptor")
            and x.get("gemm_descriptor")
            and x.get("index_contract", {}).get("status") == "checked_points_consistent"
            and y.get("index_contract", {}).get("status") == "checked_points_consistent"
        ):
            shared_address_slots.append(slot)
        for field in (
            "implementation",
            "reference_inputs",
            "reference_output",
            "candidate_activation_inputs",
            "candidate_parameter_inputs",
            "gemm_descriptor",
            "index_contract",
            "storage_contract",
            "mask_storage_contract",
            "bias_contract",
            "normalization_placement",
        ):
            if semantic_value(x.get(field)) != semantic_value(y.get(field)):
                differences.append({"slot": slot, "field": field, "before": x.get(field), "after": y.get(field)})
    for field in (
        "orchestration",
        "allocation",
        "math_mode_requests",
        "interface",
        "evaluation_evidence",
        "observed_blockers",
    ):
        if a.get(field) != b.get(field):
            differences.append({"slot": "program", "field": field, "before": a.get(field), "after": b.get(field)})
    wiring_known = all(
        n.get("candidate_activation_inputs") is not None
        and n.get("candidate_parameter_inputs") is not None
        and "unknown_parameter" not in n["candidate_parameter_inputs"]
        for c in (a, b)
        for n in c["nodes"]
    )
    same_plan = wiring_known and profile(a) == profile(b)
    return {
        "relation": (
            "insufficient_wiring"
            if not wiring_known
            else "same_computation_plan" if same_plan else "different_computation_plan"
        ),
        "differences": differences,
        "same_recorded_state": same_plan and not differences,
        "shared_checked_address_slots": shared_address_slots,
        "training_equivalence": "unestablished",
        "unknowns": sorted(set(a["unknowns"] + b["unknowns"])),
    }
