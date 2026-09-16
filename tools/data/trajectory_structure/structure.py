"""Syntactic components and conservative cross-turn provenance, never causal credit."""

from __future__ import annotations

import ast
import collections
import copy
import difflib
import hashlib
import json
import re
from typing import Any

import tree_sitter_cpp
import tree_sitter_cuda
from examples.kernel_agent.utils import _find_last_complete_section_group, parse_cuda_agent_response
from tree_sitter import Language, Parser

SCHEMA_VERSION = 1
PARSERS = {
    "CUDA_KERNELS": Parser(Language(tree_sitter_cuda.language())),
    "APPLY_BINDINGS": Parser(Language(tree_sitter_cpp.language())),
}
ATOMIC = {"string_literal", "raw_string_literal", "char_literal", "system_lib_string", "preproc_arg"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def walk(node, *, all_children=False):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children if all_children else current.named_children))


def text(node) -> str:
    return node.text.decode("utf-8") if node is not None else ""


def native_tokens(node, pragmas=(), rename_range=None) -> list[str]:
    result = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "comment":
            continue
        if n.type in ATOMIC or not n.children:
            if text(n).strip():
                if rename_range is None or not (rename_range[0] <= n.start_byte and n.end_byte <= rename_range[1]):
                    result.append((n.start_byte, text(n)))
        else:
            stack.extend(reversed(n.children))
    for pragma in pragmas:
        if node.start_byte <= pragma["start"] and pragma["end"] <= node.end_byte:
            result.append((pragma["start"], pragma["text"].strip()))
    if rename_range is not None:
        result.append((rename_range[0], "__component_name__"))
    return [value for _, value in sorted(result, key=lambda x: x[0])]


def first(node, kind):
    return next((n for n in walk(node) if n.type == kind), None)


def callable_name(declarator):
    n = declarator
    while n is not None and n.type not in {
        "identifier",
        "field_identifier",
        "qualified_identifier",
        "operator_name",
        "destructor_name",
    }:
        n = (
            n.named_children[0]
            if n.type == "parenthesized_declarator" and len(n.named_children) == 1
            else n.child_by_field_name("declarator")
        )
    return n


def ancestor_context(node):
    scopes, templates, linkage = [], [], None
    n = node.parent
    while n is not None:
        if n.type in {"namespace_definition", "class_specifier", "struct_specifier"}:
            name = text(n.child_by_field_name("name"))
            scopes.append(name or "<anonymous>")
        elif n.type == "template_declaration":
            templates.append(text(n.child_by_field_name("parameters")))
        elif n.type == "linkage_specification":
            linkage = text(n.child_by_field_name("value"))
        n = n.parent
    return list(reversed(scopes)), list(reversed(templates)), linkage


def native_calls(node, parameters=None):
    calls = []
    bindings = [(name, node.start_byte, node.end_byte) for name in native_local_bindings(parameters, None)]
    scopes = {
        "compound_statement",
        "for_statement",
        "for_range_loop",
        "if_statement",
        "while_statement",
        "switch_statement",
        "catch_clause",
    }
    for declaration in walk(node):
        if declaration.type != "declaration":
            continue
        scope = declaration.parent
        while scope is not None and scope.type not in scopes:
            scope = scope.parent
        if scope is None:
            continue
        for declarator in declaration.children_by_field_name("declarator"):
            name = callable_name(declarator)
            if name is not None:
                bindings.append((text(name), declaration.start_byte, scope.end_byte))
    for n in walk(node):
        if n.type != "call_expression":
            continue
        fn = n.child_by_field_name("function")
        if fn is None:
            continue
        base = fn.child_by_field_name("name") if fn.type == "template_function" else fn
        args = n.child_by_field_name("arguments")
        launch = next((x for x in n.named_children if x.type == "kernel_call_syntax"), None)
        name = text(base)
        kind = "kernel_launch" if launch else "direct_call"
        if re.match(r"^(cublas|cudnn|cutlass::)", name):
            kind = "library_call"
        if base.type not in {"identifier", "qualified_identifier"}:
            kind = "unresolved_expression_call"
        calls.append(
            {
                "target": name,
                "expression": text(fn),
                "kind": kind,
                "line": n.start_point.row + 1,
                "arguments": [text(a) for a in args.named_children] if args else [],
                "launch_config": [text(a) for a in launch.named_children] if launch else None,
                "local_binding": any(
                    symbol == name and start <= n.start_byte < end for symbol, start, end in bindings
                ),
            }
        )
    return calls


def native_local_bindings(parameters, body):
    names = set()
    for root in (parameters, body):
        if root is None:
            continue
        for node in walk(root):
            if node.type not in {"parameter_declaration", "declaration"}:
                continue
            for declarator in node.children_by_field_name("declarator"):
                name = callable_name(declarator)
                if name is not None:
                    names.add(text(name))
    return sorted(names)


def make_component(section, name, scope, kind, source, line, end_line, tokens, *, valid=True, **fields):
    qualified = "::".join([*scope, name])
    # A local locator. Cross-turn identity is assigned by align_components, not this ID.
    cid = f"{section}:{qualified}:{kind}:{digest([tokens[:24], line])[:10]}"
    return {
        "id": cid,
        "section": section,
        "name": name,
        "scope": scope,
        "qualified_name": qualified,
        "kind": kind,
        "line": line,
        "end_line": end_line,
        "source": source,
        "valid_syntax": valid,
        "syntax_hash": digest(tokens),
        "_tokens": tokens,
        **fields,
    }


def analyze_native(section, source):
    encoded = source.encode("utf-8")
    tree = PARSERS[section].parse(encoded)
    # Upstream grammars reject pragmas between a for-header and an unbraced body.
    # Recognize only actual preprocessor AST nodes (never text inside a string).
    # Mask for parsing with byte/line offsets preserved; retain directives in hashes and diffs.
    pragmas = [
        {"start": n.start_byte, "end": n.end_byte, "line": n.start_point.row + 1, "text": text(n)}
        for n in walk(tree.root_node)
        if n.type == "preproc_call" and text(n.child_by_field_name("directive")) == "#pragma"
    ]
    if pragmas:
        parsed = bytearray(encoded)
        for p in pragmas:
            for i in range(p["start"], p["end"]):
                if parsed[i] not in (10, 13):
                    parsed[i] = 32
        tree = PARSERS[section].parse(bytes(parsed))
    root = tree.root_node
    errors = [
        {
            "kind": n.type if not n.is_missing else "missing_token",
            "line": n.start_point.row + 1,
            "column": n.start_point.column + 1,
            "end_line": n.end_point.row + 1,
            "text": text(n),
        }
        for n in walk(root, all_children=True)
        if n.type == "ERROR" or n.is_missing
    ]
    components, exports, covered, warnings = [], [], [], []
    for node in walk(root):
        kind = None
        if node.type == "function_definition":
            kind = "kernel" if "__global__" in [c.type for c in node.children] else "function"
        elif node.type == "declaration":
            # Only direct function declarators; do not mistake a callback parameter for a prototype.
            declarators = [c for c in node.named_children if c.type == "function_declarator"]
            if declarators and len(node.children_by_field_name("declarator")) > 1:
                warnings.append(
                    {
                        "kind": "multi_declarator_kept_in_file_context",
                        "line": node.start_point.row + 1,
                        "source": text(node),
                    }
                )
                continue
            if declarators and declarators[0].child_by_field_name("declarator").type == "parenthesized_declarator":
                warnings.append(
                    {
                        "kind": "complex_declarator_kept_in_file_context",
                        "line": node.start_point.row + 1,
                        "source": text(node),
                    }
                )
                continue
            if declarators:
                kind = "prototype"
        elif node.type in {"preproc_def", "preproc_function_def"}:
            kind = "macro"
        if kind is None:
            continue
        if any(a <= node.start_byte and node.end_byte <= b for a, b in covered):
            continue
        scope, templates, linkage = ancestor_context(node)
        if kind == "macro":
            name_node = node.child_by_field_name("name")
            parameters = node.child_by_field_name("parameters")
        else:
            declaration = first(node, "function_declarator")
            if declaration is None:
                continue
            name_node = callable_name(declaration.child_by_field_name("declarator"))
            parameters = declaration.child_by_field_name("parameters")
        if name_node is None:
            continue
        name = text(name_node)
        body = node.child_by_field_name("body")
        tokens = native_tokens(node, pragmas)
        context = ["@template", *templates, "@linkage", linkage or ""]
        tokens = context + tokens
        # Retain all syntax except this declaration's own name for exact-body rename candidates.
        calls = native_calls(body, parameters) if body is not None else []
        components.append(
            make_component(
                section,
                name,
                scope,
                kind,
                encoded[node.start_byte : node.end_byte].decode(),
                node.start_point.row + 1,
                node.end_point.row + 1,
                tokens,
                valid=not node.has_error,
                signature=text(parameters),
                signature_hash=digest(native_tokens(parameters)) if parameters is not None else None,
                parameter_count=len(parameters.named_children) if parameters is not None else None,
                templates=templates,
                linkage=linkage,
                calls=calls,
                local_bindings=native_local_bindings(parameters, body),
                body_hash=digest(native_tokens(body, pragmas)) if body is not None else None,
                rename_hash=digest(context + native_tokens(node, pragmas, (name_node.start_byte, name_node.end_byte))),
            )
        )
        covered.append((node.start_byte, node.end_byte))
    for node in walk(root):
        if (
            node.type != "call_expression"
            or text(node.child_by_field_name("function")) != "TVM_FFI_DLL_EXPORT_TYPED_FUNC"
        ):
            continue
        args = node.child_by_field_name("arguments")
        if args and len(args.named_children) == 2:
            exports.append(
                {
                    "name": text(args.named_children[0]),
                    "target": text(args.named_children[1]),
                    "line": node.start_point.row + 1,
                    "valid_syntax": not node.has_error,
                }
            )
    # Includes, globals, structs and export declarations remain observable when functions are stable.
    masked = bytearray(encoded)
    for start, end in covered:
        for i in range(start, end):
            if masked[i] not in (10, 13):
                masked[i] = 32
    context_root = PARSERS[section].parse(bytes(masked)).root_node
    context_tokens = native_tokens(context_root)
    components.append(
        make_component(
            section,
            "<file-context>",
            [],
            "file_context",
            bytes(masked).decode(),
            1,
            source.count("\n") + 1,
            context_tokens,
            valid=not root.has_error,
            calls=[],
            rename_hash=None,
        )
    )
    return {
        "syntax_valid": not root.has_error,
        "errors": errors,
        "components": components,
        "exports": exports,
        "coverage_warnings": warnings,
        "parser_adaptations": [{"kind": "pragma_lexical_mask", "line": p["line"], "text": p["text"]} for p in pragmas],
    }


def python_calls(fn):
    calls = []
    # Nested function bodies are separate components, not calls made by their definition.
    stack = list(reversed(fn.body))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(n, ast.Call):
            target = ast.unparse(n.func)
            kind = "extension_call" if re.fullmatch(r"tvm_ffi_extension\.\w+", target) else "python_call"
            calls.append(
                {
                    "target": target,
                    "kind": kind,
                    "line": n.lineno,
                    "arguments": [ast.unparse(a) for a in n.args],
                    "keyword_arguments": [k.arg for k in n.keywords],
                }
            )
        stack.extend(reversed(list(ast.iter_child_nodes(n))))
    return calls


def python_tokens(node):
    # AST fields encode scope/indentation, operators and literal values; comments are excluded.
    return re.findall(
        r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|[A-Za-z_]\w*|\d+|\S", ast.dump(node, include_attributes=False)
    )


def analyze_python(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {
            "syntax_valid": False,
            "errors": [{"kind": "SyntaxError", "line": e.lineno, "column": e.offset, "text": e.msg}],
            "components": [],
            "exports": [],
        }
    components = []
    lines = source.splitlines(keepends=True)

    def visit(statements, scope):
        for node in statements:
            if isinstance(node, ast.ClassDef):
                visit(node.body, [*scope, node.name])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([node.lineno, *[x.lineno for x in node.decorator_list]])
                normalized = copy.deepcopy(node)
                normalized.name = "__component_name__"
                components.append(
                    make_component(
                        "MODEL_NEW",
                        node.name,
                        scope,
                        "python_function",
                        "".join(lines[start - 1 : node.end_lineno]),
                        start,
                        node.end_lineno,
                        python_tokens(node),
                        signature=ast.unparse(node.args),
                        calls=python_calls(node),
                        local_bindings=sorted(
                            {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                            | {n.arg for n in ast.walk(node.args) if isinstance(n, ast.arg)}
                        ),
                        body_hash=digest([ast.dump(x) for x in node.body]),
                        rename_hash=digest(python_tokens(normalized)),
                    )
                )
                visit(node.body, [*scope, node.name])
            else:
                # Include definitions in exception handlers and match cases as well.
                visit(list(ast.iter_child_nodes(node)), scope)

    visit(tree.body, [])

    class StripFunctions(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return ast.copy_location(ast.Pass(), node)

        visit_AsyncFunctionDef = visit_FunctionDef

    context = ast.fix_missing_locations(StripFunctions().visit(copy.deepcopy(tree)))
    components.append(
        make_component(
            "MODEL_NEW",
            "<file-context>",
            [],
            "file_context",
            ast.unparse(context),
            1,
            len(lines),
            python_tokens(context),
            calls=[],
            rename_hash=None,
            source_view="AST context with function bodies replaced by pass",
        )
    )
    return {
        "syntax_valid": True,
        "errors": [],
        "components": components,
        "exports": [],
        "has_modelnew": any(isinstance(n, ast.ClassDef) and n.name == "ModelNew" for n in tree.body),
    }


def call_graph(components, exports):
    definitions = [c for c in components if c["kind"] in {"function", "kernel", "python_function"}]
    by_symbol = collections.defaultdict(list)
    for comp in definitions:
        by_symbol[(comp["section"] == "MODEL_NEW", comp["qualified_name"])].append(comp)
    export_map = collections.defaultdict(list)
    for export in exports:
        if export["valid_syntax"]:
            export_map[export["name"]].append(export["target"])
    edges = []
    for comp in components:
        for call in comp.get("calls", []):
            name = call["target"]
            names = []
            py = comp["section"] == "MODEL_NEW"
            local_binding = call.get("local_binding", False) or (
                py
                and (
                    name in comp.get("local_bindings", [])
                    or (call["kind"] == "extension_call" and name.split(".")[0] in comp.get("local_bindings", []))
                )
            )
            if local_binding:
                names = []
            elif call["kind"] == "extension_call":
                names = export_map.get(name.split(".")[-1], [])
                py = False
            elif call["kind"] == "unresolved_expression_call":
                names = []
            elif py and name.startswith("self.") and name.count(".") == 1:
                names = ["::".join([*comp["scope"], name[5:]])]
            elif re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", name):
                for depth in range(len(comp["scope"]), -1, -1):
                    candidate = "::".join([*comp["scope"][:depth], name])
                    if by_symbol.get((py, candidate)):
                        names = [candidate]
                        break
            targets = [d for n in names for d in by_symbol.get((py, n), [])]
            # Static symbol candidates only. No type resolution, reachability or side-effect proof.
            resolution = (
                "unique_name_candidate" if len(targets) == 1 else "ambiguous" if targets else "unresolved_or_external"
            )
            if local_binding:
                resolution = "local_binding_unresolved"
            if not comp["valid_syntax"] or any(not t["valid_syntax"] for t in targets):
                resolution = "syntax_uncertain"
            edges.append(
                {
                    "from": comp["id"],
                    **call,
                    "candidate_targets": [c["id"] for c in targets],
                    "resolution": resolution,
                    "resolution_basis": "name_only_not_binding_or_type_checked",
                }
            )
    return edges


def analyze_response(response):
    native, model = parse_cuda_agent_response(response)
    sources = {}
    for file, section in [
        ("kernels/generated.cu", "CUDA_KERNELS"),
        ("kernels/generated_binding.cpp", "APPLY_BINDINGS"),
    ]:
        if file in native:
            sources[section] = native[file]
    if model is not None:
        sources["MODEL_NEW"] = model
    complete_group = bool(_find_last_complete_section_group(response))
    sections = {
        name: (analyze_python(body) if name == "MODEL_NEW" else analyze_native(name, body))
        for name, body in sources.items()
    }
    components = [c for s in sections.values() for c in s["components"]]
    counts = collections.Counter(c["id"] for c in components)
    ordinals = collections.Counter()
    for c in components:
        if counts[c["id"]] > 1:
            original = c["id"]
            c["id"] += f":occurrence={ordinals[original]}"
            ordinals[original] += 1
    exports = [e for s in sections.values() for e in s["exports"]]
    return {
        "selection_mode": "last_complete_group" if complete_group else "partial_fallback" if sources else "empty",
        "complete_sections": len(sources) == 3,
        "valid_modelnew": sections.get("MODEL_NEW", {}).get("has_modelnew", False),
        "sections": {
            k: {"source": sources[k], **{a: b for a, b in v.items() if a not in ["components", "exports"]}}
            for k, v in sections.items()
        },
        "components": components,
        "exports": exports,
        "calls": call_graph(components, exports),
    }


def identity_key(component):
    return (
        component["section"],
        component["qualified_name"],
        component["kind"],
        component.get("signature_hash", component.get("signature")),
    )


def snapshot_observed(snapshot, section):
    return (
        snapshot["complete_sections"]
        and snapshot["valid_modelnew"]
        and snapshot["sections"].get(section, {}).get("syntax_valid", False)
    )


def align_components(before, after):
    matches = []
    remaining_left = {c["id"]: c for c in before}
    remaining_right = {c["id"]: c for c in after}

    def pair(key, reason):
        left = collections.defaultdict(list)
        right = collections.defaultdict(list)
        for c in remaining_left.values():
            if key(c) is not None:
                left[key(c)].append(c)
        for c in remaining_right.values():
            if key(c) is not None:
                right[key(c)].append(c)
        for k in sorted(left.keys() & right.keys(), key=str):
            if len(left[k]) == len(right[k]) == 1:
                a, b = left[k][0], right[k][0]
                matches.append((a, b, reason))
                remaining_left.pop(a["id"])
                remaining_right.pop(b["id"])

    pair(identity_key, "exact_overload_signature")
    pair(
        lambda c: (
            (c["section"], tuple(c["scope"]), c["kind"], c.get("rename_hash"))
            if c.get("rename_hash") and c["valid_syntax"]
            else None
        ),
        "unique_exact_syntax_rename_candidate",
    )
    return matches, list(remaining_left.values()), list(remaining_right.values())


def token_similarity(a, b):
    # Token-trigram multiset Dice preserves local ordering. Not semantic equivalence.
    width = min(3, len(a), len(b))
    if not width:
        return 1.0 if not a and not b else 0.0
    ca = collections.Counter(tuple(a[i : i + width]) for i in range(len(a) - width + 1))
    cb = collections.Counter(tuple(b[i : i + width]) for i in range(len(b) - width + 1))
    total = sum(ca.values()) + sum(cb.values())
    return 2 * sum((ca & cb).values()) / total if total else 1.0


def edit_hunks(before, after, context=5):
    """Anchored local syntax changes. Unmatched anchors are unknown, not proof of deletion."""
    a, b = before["_tokens"], after["_tokens"]
    # Large components still receive a full source diff; avoid quadratic token alignment.
    if max(len(a), len(b)) > 12000:
        return [], "component_too_large_for_local_hunks"
    hunks = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, i, j, k, new_end in matcher.get_opcodes():
        if op == "equal":
            continue
        left = a[max(0, i - context) : i]
        right = a[j : j + context]
        # An anchor is usable only when its unchanged context also exists beside the new edit.
        if b[max(0, k - len(left)) : k] != left or b[new_end : new_end + len(right)] != right:
            continue
        hunks.append(
            {
                "operation": op,
                "before": a[i:j],
                "after": b[k:new_end],
                "left_context": left,
                "right_context": right,
                "at_start": not left,
                "at_end": not right,
                "before_token_span": [i, j],
                "after_token_span": [k, new_end],
            }
        )
    return hunks, None


def occurrences(tokens, pattern):
    if not pattern:
        return []
    return [i for i in range(len(tokens) - len(pattern) + 1) if tokens[i : i + len(pattern)] == pattern]


def hunk_state(hunk, component):
    tokens = component["_tokens"]

    def count(side):
        pattern = hunk["left_context"] + hunk[side] + hunk["right_context"]
        return sum(
            (not hunk["at_start"] or i == 0) and (not hunk["at_end"] or i + len(pattern) == len(tokens))
            for i in occurrences(tokens, pattern)
        )

    old, new = count("before"), count("after")
    if new == 1 and old == 0:
        return "present"
    if old == 1 and new == 0:
        return "reverted"
    return "unknown"


def analyze_trajectory(turns):
    """Input turns have turn_idx + response; preserve chronological missingness explicitly."""
    turns = sorted(turns, key=lambda x: x["turn_idx"])
    indices = [t["turn_idx"] for t in turns]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate turn_idx within trajectory")
    analyzed = []
    transitions = []
    events = []
    last_components = []
    lineages = {}
    next_lineage = 0
    catalog = {}
    event_by_id = {}
    for item in turns:
        s = analyze_response(item["response"])
        s["turn_idx"] = item["turn_idx"]
        s["observation"] = {k: v for k, v in item.items() if k != "response"}
        consecutive = bool(analyzed) and item["turn_idx"] == analyzed[-1]["turn_idx"] + 1
        pairs, missing, added = (
            align_components(last_components, s["components"]) if consecutive else ([], [], s["components"])
        )
        rename_pairs = [p for p in pairs if p[2] == "unique_exact_syntax_rename_candidate"]
        pairs = [p for p in pairs if p[2] != "unique_exact_syntax_rename_candidate"]
        missing.extend(a for a, b, _ in rename_pairs)
        added.extend(b for a, b, _ in rename_pairs)
        rename_targets = {b["id"] for _, b, _ in rename_pairs}
        new_lineages = {}
        changes = []
        for a, b, reason in pairs:
            lid = lineages[a["id"]]
            new_lineages[b["id"]] = lid
            b["lineage"] = lid
            status = "unchanged" if a["syntax_hash"] == b["syntax_hash"] else "changed"
            if not (a["valid_syntax"] and b["valid_syntax"]):
                status = "syntax_uncertain"
            hunks, skip = edit_hunks(a, b) if status == "changed" else ([], None)
            change = {
                "lineage": lid,
                "before": a["id"],
                "after": b["id"],
                "alignment": reason,
                "status": status,
                "token_similarity": token_similarity(a["_tokens"], b["_tokens"]),
                "hunks": hunks,
                "hunks_skip_reason": skip,
                "source_diff": (
                    "".join(
                        difflib.unified_diff(
                            a["source"].splitlines(True),
                            b["source"].splitlines(True),
                            fromfile=a["qualified_name"],
                            tofile=b["qualified_name"],
                        )
                    )
                    if status == "changed"
                    else ""
                ),
            }
            changes.append(change)
            for h in hunks:
                if not all(snapshot_observed(t, b["section"]) for t in [analyzed[-1], s]):
                    continue
                if hunk_state(h, a) != "reverted" or hunk_state(h, b) != "present":
                    continue
                eid = digest([lid, {k: v for k, v in h.items() if not k.endswith("_token_span")}])[:20]
                if eid in event_by_id:
                    event_by_id[eid]["applied_turns"].append(item["turn_idx"])
                    continue
                event = {
                    "event_id": eid,
                    "introduced_turn": item["turn_idx"],
                    "applied_turns": [item["turn_idx"]],
                    "lineage": lid,
                    "component": b["qualified_name"],
                    "section": b["section"],
                    "event_type": "anchored_edit",
                    "hunk": h,
                    "observations": [],
                }
                events.append(event)
                event_by_id[eid] = event
        for b in added:
            old = [
                (lid, c)
                for lid, c in catalog.items()
                if lid not in new_lineages.values() and identity_key(c) == identity_key(b)
            ]
            current = [c for c in added if identity_key(c) == identity_key(b)]
            if len(old) == len(current) == 1:
                lid = old[0][0]
                b["identity_reconnected"] = True
            else:
                lid = f"c{next_lineage:04}"
                next_lineage += 1
                if (
                    not old
                    and consecutive
                    and analyzed[-1]["complete_sections"]
                    and all(snapshot_observed(t, b["section"]) for t in [analyzed[-1], s])
                    and b["id"] not in rename_targets
                    and b["valid_syntax"]
                    and b["kind"] != "file_context"
                ):
                    events.append(
                        {
                            "event_id": digest([item["turn_idx"], lid, b["syntax_hash"]])[:20],
                            "introduced_turn": item["turn_idx"],
                            "lineage": lid,
                            "component": b["qualified_name"],
                            "section": b["section"],
                            "event_type": "component_added",
                            "syntax_hash": b["syntax_hash"],
                            "observations": [],
                        }
                    )
            new_lineages[b["id"]] = lid
            b["lineage"] = lid
        catalog.update({c["lineage"]: c for c in s["components"]})
        s["components_by_lineage"] = {c["lineage"]: c for c in s["components"]}
        for e in events:
            if item["turn_idx"] < e["introduced_turn"]:
                continue
            comp = s["components_by_lineage"].get(e["lineage"])
            section = s["sections"].get(e["section"])
            if not s["complete_sections"] or not s["valid_modelnew"] or not section or not section["syntax_valid"]:
                state = "unobserved"
            elif comp is None:
                old = catalog[e["lineage"]]
                ambiguous = any(
                    identity_key(c) == identity_key(old)
                    or (
                        old.get("rename_hash")
                        and c.get("rename_hash") == old["rename_hash"]
                        and c["section"] == old["section"]
                        and c["kind"] == old["kind"]
                        and c["scope"] == old["scope"]
                    )
                    for c in s["components"]
                )
                state = "unknown_component_identity" if ambiguous else "component_absent"
            elif not comp["valid_syntax"]:
                state = "unobserved"
            elif e["event_type"] == "component_added":
                state = "present" if comp["syntax_hash"] == e["syntax_hash"] else "modified_version"
            else:
                state = hunk_state(e["hunk"], comp)
            e["observations"].append({"turn_idx": item["turn_idx"], "state": state})
        if analyzed:
            sections = {}
            for name in sorted(s["sections"].keys() | analyzed[-1]["sections"].keys()):
                left = [c for c in last_components if c["section"] == name]
                right = [c for c in s["components"] if c["section"] == name]
                valid = (
                    consecutive
                    and s["complete_sections"]
                    and analyzed[-1]["complete_sections"]
                    and all(x.get("sections", {}).get(name, {}).get("syntax_valid", False) for x in [s, analyzed[-1]])
                )
                sections[name] = {
                    "comparable": valid,
                    "token_similarity": (
                        token_similarity(
                            [x for c in left for x in c["_tokens"]], [x for c in right for x in c["_tokens"]]
                        )
                        if valid
                        else None
                    ),
                }
            transitions.append(
                {
                    "from_turn": analyzed[-1]["turn_idx"],
                    "to_turn": item["turn_idx"],
                    "consecutive": consecutive,
                    "sections": sections,
                    "changes": changes,
                    "rename_candidates": [
                        {"before": a["id"], "after": b["id"], "reason": reason} for a, b, reason in rename_pairs
                    ],
                    "unmatched_before": [c["id"] for c in missing],
                    "unmatched_after": [c["id"] for c in added if not c.get("identity_reconnected")],
                    "reconnected_after": [c["id"] for c in added if c.get("identity_reconnected")],
                }
            )
        analyzed.append(s)
        last_components = s["components"]
        lineages = new_lineages
    for e in events:
        obs = e["observations"]
        e["nonadjacent_retained_turns"] = [
            o["turn_idx"] for o in obs if o["state"] == "present" and o["turn_idx"] >= e["introduced_turn"] + 2
        ]
        reverted = False
        e["reintroduced_turns"] = []
        e["reintroductions"] = []
        for o in obs:
            if o["state"] in {"reverted", "component_absent"}:
                reverted = o["state"]
            elif o["state"] == "present" and reverted:
                e["reintroduced_turns"].append(o["turn_idx"])
                e["reintroductions"].append({"turn_idx": o["turn_idx"], "after_state": reverted})
                reverted = False
    versions = collections.defaultdict(list)
    for s in analyzed:
        for c in s["components"]:
            if c["valid_syntax"]:
                versions[c["lineage"]].append(
                    {
                        "turn_idx": s["turn_idx"],
                        "hash": c["syntax_hash"],
                        "component": c["qualified_name"],
                        "section": c["section"],
                    }
                )
        s.pop("components_by_lineage", None)
    recurrences = []
    for lid, rows in versions.items():
        seen = {}
        last_seen = {}
        previous = None
        for row in rows:
            h = row["hash"]
            absent_between = h in seen and any(
                last_seen[h] < turn["turn_idx"] < row["turn_idx"]
                and turn["complete_sections"]
                and turn["sections"].get(row["section"], {}).get("syntax_valid")
                and not any(c["lineage"] == lid for c in turn["components"])
                for turn in analyzed
            )
            if h in seen and (previous != h or absent_between):
                recurrences.append(
                    {
                        "lineage": lid,
                        "component": row["component"],
                        "section": row["section"],
                        "first_observed_turn": seen[h],
                        "reappeared_turn": row["turn_idx"],
                        "syntax_hash": h,
                        "intervening_unmatched_component": absent_between,
                    }
                )
            seen.setdefault(h, row["turn_idx"])
            last_seen[h] = row["turn_idx"]
            previous = h
    return {
        "schema_version": SCHEMA_VERSION,
        "turns": analyzed,
        "transitions": transitions,
        "edit_events": events,
        "component_version_recurrences": recurrences,
        "missing_turns": [i for i in range(min(indices), max(indices) + 1) if i not in indices] if indices else [],
        "interpretation": "Syntactic retention and call-site evidence only; no causal credit or semantic equivalence.",
    }


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value
