"""Bounded source dataflow candidates for local revisions and fusion scope.

No generated execution, stage-count inference, equivalence proof or reward.
Statements form symbolic value stages; loops stay structured opaque regions.
Unrecognized calls, aliasing and control flow remain explicit limitations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .optimization_pilot import digest, inspect_kernel, strip_comments
from .optimization_strategies import BASELINE_RULES
from .structure import PARSERS, callable_name, native_tokens, text, walk

PURE = {
    "fmaxf",
    "fminf",
    "fmax",
    "fmin",
    "tanhf",
    "tanh",
    "expf",
    "exp",
    "__expf",
    "sqrtf",
    "sqrt",
    "rsqrtf",
    "fabsf",
    "fabs",
    "fmaf",
    "fma",
    "__ldg",
    "__shfl_sync",
    "__shfl_down_sync",
    "__shfl_up_sync",
    "__shfl_xor_sync",
    "max",
    "min",
    "__syncthreads",
}
LOOPS = {"for_statement", "while_statement", "do_statement"}


@dataclass(frozen=True)
class Value:
    key: str
    operations: int
    lines: frozenset
    unsafe: bool


class Regions:
    def __init__(self, component, context_hash):
        self.component = component
        self.context_hash = context_hash
        self.values = {}
        self.nodes = {}
        self.stages = []
        self.outputs = []
        self.output_ports = set()
        self.unknowns = set()
        self.env = {}
        self.guards = []
        self.memory = self.make("memory", "entry")
        self.parameter_names = set()
        self.source = strip_comments(component["source"])
        self.pragmas = re.findall(r"(?m)^\s*#pragma[^\n]*", self.source)
        masked = re.sub(r"(?m)^\s*#pragma[^\n]*", lambda m: " " * len(m[0]), self.source)
        root = PARSERS["CUDA_KERNELS"].parse(masked.encode()).root_node
        self.fn = next((n for n in walk(root) if n.type == "function_definition"), None)
        self.valid = self.fn is not None and not self.fn.has_error
        if not self.valid:
            self.unknowns.add("unparseable_function")
            return
        params = next(n for n in walk(self.fn) if n.type == "parameter_list")
        for index, p in enumerate(params.named_children):
            name = callable_name(p.child_by_field_name("declarator"))
            if name is None:
                continue
            name = text(name)
            type_tokens = [t for t in native_tokens(p) if t != name]
            self.env[name] = self.make("parameter", [index, type_tokens])
            self.parameter_names.add(name)
        self.parameters = dict(self.env)

    def make(self, kind, tag, inputs=(), node=None, operations=0, unsafe=False):
        inputs = tuple(inputs)
        key = digest([kind, tag, [v.key for v in inputs]])
        lines = set().union(*(v.lines for v in inputs)) if inputs else set()
        if node is not None:
            lines.update(range(node.start_point.row + 1, node.end_point.row + 2))
        value = Value(
            key,
            operations + sum(v.operations for v in inputs),
            frozenset(lines),
            unsafe or any(v.unsafe for v in inputs),
        )
        self.nodes[key] = {"kind": kind, "tag": tag, "inputs": [v.key for v in inputs]}
        self.values[key] = value
        return value

    def canon(self, node, names=None):
        """Structured opaque region with bound external values and local binders."""
        names = dict(names or {})
        for d in walk(node):
            if d.type in {"declaration", "parameter_declaration"}:
                for decl in d.children_by_field_name("declarator"):
                    name = callable_name(decl)
                    if name is not None:
                        names.setdefault(text(name), f"region_local_{len(names)}")

        def visit(n):
            if n.type == "comment":
                return None
            if not n.children:
                val = text(n)
                if n.type == "identifier":
                    if val in names:
                        val = names[val]
                    elif val in self.env:
                        val = ["bound", self.env[val].key]
                return [n.type, val]
            children = [v for c in n.children if (v := visit(c)) is not None]
            # No reassociation, cast elimination or multiplication strength reduction.
            if n.type == "binary_expression":
                op = text(n.child_by_field_name("operator"))
                if op in {"+", "*"} and len(children) == 3:
                    a, b = sorted([children[0], children[2]], key=digest)
                    children = [a, children[1], b]
            return [n.type, children]

        return visit(node)

    def expression(self, node):
        if node is None:
            return self.make("missing", None, unsafe=True)
        kind = node.type
        if kind == "identifier":
            name = text(node)
            return self.env.get(name, self.make("external", name))
        if kind in {"number_literal", "true", "false", "null", "string_literal"}:
            return self.make(kind, text(node), node=node)
        if kind == "parenthesized_expression" and len(node.named_children) == 1:
            return self.expression(node.named_children[0])
        if kind == "field_expression":
            base = self.expression(node.child_by_field_name("argument"))
            field = text(node.child_by_field_name("field"))
            full = text(node)
            return self.env.get(full, self.make("field", field, [base], node))
        if kind == "binary_expression":
            op = text(node.child_by_field_name("operator"))
            inputs = [
                self.expression(node.child_by_field_name("left")),
                self.expression(node.child_by_field_name("right")),
            ]
            if op in {"+", "*"} and not any(v.unsafe for v in inputs):
                inputs.sort(key=lambda v: v.key)
            return self.make("binary", op, inputs, node, operations=1)
        if kind in {"unary_expression", "pointer_expression"}:
            arg = self.expression(node.child_by_field_name("argument"))
            op = text(node.child_by_field_name("operator")) or text(node)[:1]
            inputs = [arg, self.memory] if op == "*" else [arg]
            return self.make("unary", op, inputs, node, operations=int(op == "*"))
        if kind == "subscript_expression":
            base = self.expression(node.child_by_field_name("argument"))
            indices = node.child_by_field_name("indices") or node.child_by_field_name("index")
            args = [self.expression(n) for n in indices.named_children] if indices is not None else []
            return self.make("load", "indexed", [base, *args, self.memory], node, operations=1)
        if kind == "call_expression":
            target = text(node.child_by_field_name("function"))
            args = node.child_by_field_name("arguments")
            inputs = [self.expression(a) for a in args.named_children if a.type != "comment"] if args else []
            cast = target.startswith(("reinterpret_cast<", "static_cast<", "const_cast<"))
            unsafe = target not in PURE and not cast
            if unsafe:
                self.unknowns.add("unresolved_call:" + target)
            if target == "__ldg":
                inputs.append(self.memory)
            return self.make("call", target, inputs, node, operations=int(not cast), unsafe=unsafe)
        if kind == "cast_expression":
            return self.make(
                "cast",
                text(node.child_by_field_name("type")),
                [self.expression(node.child_by_field_name("value"))],
                node,
            )
        if kind == "conditional_expression":
            return self.make("select", None, [self.expression(c) for c in node.named_children], node, operations=1)
        if kind == "sizeof_expression":
            return self.make("sizeof", self.canon(node), node=node)
        if kind in {"initializer_list", "argument_list"}:
            return self.make(kind, None, [self.expression(c) for c in node.named_children], node)
        self.unknowns.add("opaque_expression:" + kind)
        return self.make("opaque", self.canon(node), node=node, unsafe=True)

    def emit(self, value, node, kind):
        if value.operations < 3 or value.unsafe:
            return
        guard_values = tuple(self.guards)
        self.stages.append(
            {
                "value": value.key,
                "guard_keys": [g.key for g in guard_values],
                "region_key": digest([self.context_hash, self.pragmas, [g.key for g in guard_values], value.key]),
                "kind": kind,
                "line": self.component["line"] + node.start_point.row,
                "source": text(node),
                "operations": value.operations,
                "evidence_lines": sorted(value.lines),
                "unsafe_guard": any(g.unsafe for g in guard_values),
            }
        )

    def assign(self, left, value, node):
        if left.type == "identifier":
            name = text(left)
            for key in list(self.env):
                if key.startswith(name + "."):
                    del self.env[key]
            self.env[name] = value
            self.emit(value, node, "value_stage")
            return
        if left.type == "field_expression":
            base_name = text(left.child_by_field_name("argument"))
            if base_name not in self.env or "->" in text(left):
                self.unknowns.add("unresolved_field_write")
                return
            base = self.env[base_name]
            self.env[base_name] = self.make(
                "field_update", text(left.child_by_field_name("field")), [base, value], node
            )
            self.env[text(left)] = value
            self.emit(value, node, "value_stage")
            self.emit(self.env[base_name], node, "aggregate_stage")
            return
        if left.type in {"subscript_expression", "pointer_expression"}:
            target = self.expression(left)
            self.memory = self.make("store", None, [self.memory, target, value], node)
            port = self.output_port(left)
            if port is not None:
                self.outputs.append(value.key)
                self.output_ports.add(port)
            self.emit(value, node, "stored_stage")
            return
        self.unknowns.add("unsupported_assignment")

    def output_port(self, left):
        base = left.child_by_field_name("argument")
        if base is None:
            return None
        value = self.expression(base)
        pending, visited, pointers = [value.key], set(), set()
        while pending:
            key = pending.pop()
            if key in visited:
                continue
            visited.add(key)
            item = self.nodes[key]
            if item["kind"] == "parameter" and "*" in item["tag"][1]:
                pointers.add(key)
            elif item["kind"] not in {"memory", "store", "load", "loop_memory", "memory_phi"}:
                pending.extend(item["inputs"])
        if not pointers:
            return None
        # Preserve formal pointer role and indexing; don't include output values.
        return digest([sorted(pointers), self.canon(left)])

    def loop(self, node):
        # Summarize the entire region. Do not unroll or fabricate path execution.
        before = dict(self.env)
        body = node.child_by_field_name("body")
        assignments = [n for n in walk(body) if n.type == "assignment_expression"] if body else []
        updates = {
            text(a.child_by_field_name("left"))
            for a in assignments
            if a.child_by_field_name("left") is not None and a.child_by_field_name("left").type == "identifier"
        }
        modified = sorted(updates.intersection(before))
        field_updates = {}
        for assignment in assignments:
            left = assignment.child_by_field_name("left")
            if left is not None and left.type == "field_expression":
                if text(left.child_by_field_name("argument")) in before and "->" not in text(left):
                    field_updates[text(left)] = left
                else:
                    self.unknowns.add("unresolved_loop_field_write")
        for update in walk(node):
            if update.type == "update_expression":
                arg = update.child_by_field_name("argument")
                if arg is None or arg.type != "identifier":
                    self.unknowns.add("unresolved_loop_update")
                elif text(arg) in before:
                    modified.append(text(arg))
        modified = sorted(set(modified))
        calls = [text(n.child_by_field_name("function")) for n in walk(node) if n.type == "call_expression"]
        unsafe = any(
            c not in PURE and not c.startswith(("reinterpret_cast<", "static_cast<", "const_cast<")) for c in calls
        )
        if unsafe:
            self.unknowns.add("loop_with_unresolved_call")
        used = sorted({text(n) for n in walk(node) if n.type == "identifier"}.intersection(before))
        inputs = [before[name] for name in used]
        region = self.canon(node)
        operations = sum(
            n.type in {"binary_expression", "call_expression", "assignment_expression"} for n in walk(node)
        )
        value = self.make("loop", region, [*inputs, self.memory], node, operations=operations, unsafe=unsafe)
        for index, name in enumerate(modified):
            self.env[name] = self.make("loop_result", index, [value], node)
            self.emit(self.env[name], node, "loop_result")
        for index, left in enumerate(field_updates.values(), len(modified)):
            self.assign(left, self.make("loop_result", index, [value], node), node)
        memory_writes = any(
            a.child_by_field_name("left").type in {"subscript_expression", "pointer_expression"}
            for a in assignments
            if a.child_by_field_name("left") is not None
        )
        if memory_writes:
            self.memory = self.make("loop_memory", None, [self.memory, value], node)
            ports = [
                self.output_port(a.child_by_field_name("left"))
                for a in assignments
                if a.child_by_field_name("left") is not None
                and a.child_by_field_name("left").type in {"subscript_expression", "pointer_expression"}
            ]
            self.output_ports.update(p for p in ports if p is not None)
            if any(p is not None for p in ports):
                self.outputs.append(value.key)
        self.emit(value, node, "loop_region")

    def execute(self, node):
        if node is None:
            return
        kind = node.type
        if kind == "compound_statement":
            for child in node.named_children:
                if self.execute(child):
                    return True
        elif kind == "declaration":
            for decl in node.children_by_field_name("declarator"):
                name_node = callable_name(decl)
                if name_node is None:
                    continue
                name = text(name_node)
                value_node = decl.child_by_field_name("value")
                if value_node is None:
                    self.env[name] = self.make(
                        "allocation_or_uninitialized", [text(node.child_by_field_name("type")), self.canon(decl)]
                    )
                else:
                    value = self.expression(value_node)
                    # Keep explicit declared type, including vector width and precision.
                    value = self.make("declared", text(node.child_by_field_name("type")), [value], node)
                    self.env[name] = value
                    self.emit(value, node, "value_stage")
        elif kind == "expression_statement":
            for expr in node.named_children:
                if expr.type != "assignment_expression":
                    if expr.type == "call_expression":
                        value = self.expression(expr)
                        self.memory = self.make("call_effect", None, [self.memory, value], expr, unsafe=value.unsafe)
                    elif expr.type != "comment":
                        self.unknowns.add("unsupported_expression_statement:" + expr.type)
                    continue
                left, right = expr.child_by_field_name("left"), expr.child_by_field_name("right")
                op = text(expr.child_by_field_name("operator"))
                value = self.expression(right)
                if op != "=":
                    args = [self.expression(left), value]
                    if op in {"+=", "*="}:
                        args.sort(key=lambda v: v.key)
                    value = self.make("binary", op[:-1], args, expr, operations=1)
                self.assign(left, value, expr)
        elif kind in LOOPS:
            self.loop(node)
        elif kind == "if_statement":
            # tree-sitter CUDA wraps the condition in condition_clause.
            clause = node.child_by_field_name("condition")
            if clause is not None and clause.type == "condition_clause":
                condition = self.expression(clause.child_by_field_name("value") or clause.named_children[-1])
            else:
                condition = self.expression(clause)
            then = node.child_by_field_name("consequence")
            alternative = node.child_by_field_name("alternative")
            single_return = then is not None and (
                then.type == "return_statement"
                or (
                    then.type == "compound_statement"
                    and len(then.named_children) == 1
                    and then.named_children[0].type == "return_statement"
                )
            )
            if single_return and alternative is None:
                self.guards.append(self.make("not", None, [condition]))
                return
            original_env, original_memory, original_guards = dict(self.env), self.memory, list(self.guards)
            self.guards.append(condition)
            if self.execute(then):
                self.unknowns.add("branch_termination")
            then_env, then_memory = dict(self.env), self.memory
            self.env, self.memory, self.guards = dict(original_env), original_memory, list(original_guards)
            self.guards.append(self.make("not", None, [condition]))
            if alternative is not None:
                if self.execute(alternative):
                    self.unknowns.add("branch_termination")
            for name in original_env:
                a, b = then_env.get(name, original_env[name]), self.env.get(name, original_env[name])
                self.env[name] = a if a.key == b.key else self.make("phi", None, [condition, a, b])
            self.env = {name: value for name, value in self.env.items() if name in original_env}
            self.memory = self.make("memory_phi", None, [condition, then_memory, self.memory])
            self.guards = original_guards
        elif kind == "else_clause":
            for child in node.named_children:
                if self.execute(child):
                    return True
        elif kind == "return_statement":
            return True
        elif kind not in {"comment", "preproc_call"}:
            self.unknowns.add("unsupported_statement:" + kind)
            self.memory = self.make("unsupported_effect", kind, [self.memory], node, unsafe=True)

    def result(self):
        parsed = inspect_kernel(self.component)
        if not self.valid or parsed is None:
            return {"stages": [], "unknowns": sorted(self.unknowns)}
        body = self.fn.child_by_field_name("body")
        self.execute(body)
        declarations = []
        for n in walk(body):
            if n.type == "declaration":
                for decl in n.children_by_field_name("declarator"):
                    name = callable_name(decl)
                    if name is not None:
                        declarations.append(text(name))
        if len(set(declarations)) != len(declarations):
            self.unknowns.add("repeated_local_declarations")
        reachable = set()
        pending = list(self.outputs)
        while pending:
            key = pending.pop()
            if key in reachable:
                continue
            reachable.add(key)
            pending.extend(self.nodes[key]["inputs"])
        anchors = {
            label: {e["line"] - self.component["line"] + 1 for e in evidence}
            for label, evidence in parsed["features"].items()
            if label in BASELINE_RULES - {"reduction_epilogue"}
        }
        stages = []
        seen = set()
        for stage in self.stages:
            if stage["value"] not in reachable or stage["unsafe_guard"]:
                continue
            labels = sorted(label for label, lines in anchors.items() if lines.intersection(stage["evidence_lines"]))
            if not labels:
                continue
            stage = stage | {
                "features": labels,
                "region_key": digest([stage["region_key"], labels]),
                "is_output_value": stage["value"] in self.outputs,
            }
            if stage["region_key"] not in seen:
                stages.append(stage)
                seen.add(stage["region_key"])
        canonical = self.canon(body, {name: f"parameter_{i}" for i, name in enumerate(self.parameters)})
        if self.unknowns:
            stages = []
        return {
            "stages": stages,
            "unknowns": sorted(self.unknowns),
            "output_ports": sorted(self.output_ports),
            "whole_normalized": (
                None
                if self.unknowns
                else digest(
                    [
                        self.context_hash,
                        canonical,
                        self.component.get("templates", []),
                        [self.nodes[v.key]["tag"] for v in self.parameters.values()],
                        re.findall(r"(?m)^\s*#pragma[^\n]*", self.source),
                    ]
                )
            ),
        }


def compare(before, after):
    """Return candidate types, never infer causality or safe training credit."""
    if before.get("whole_normalized") and before["whole_normalized"] == after.get("whole_normalized"):
        return {"kind": "normalized_whole", "matched_regions": []}
    a = {s["region_key"]: s for s in before["stages"]}
    compatible_output = bool(set(before.get("output_ports", [])).intersection(after.get("output_ports", [])))
    shared = [s for s in after["stages"] if s["region_key"] in a and s["is_output_value"] and compatible_output]
    # Only report maximal shared regions; a sequence of scalar assignments is
    # evidence for one implementation fragment, not many reward units.
    shared.sort(key=lambda s: (s["operations"], len(s["source"])), reverse=True)
    return {"kind": "partial_region" if shared else "no_match", "matched_regions": shared[:1]}
