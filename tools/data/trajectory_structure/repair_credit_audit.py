"""Offline first-correct repair-diff audit; never changes rewards or executes code.

Reuse the canonical response parser and generic syntax edit tracker, not the
FastCredit feature catalog or allocation rule. A surviving edit is a candidate,
not a correctness contribution. All turn numbers in saved structure are zero-based.
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

from .analyze import file_sha, normalize, trajectory_markdown
from .structure import analyze_trajectory, edit_hunks, hunk_state, identity_key, public


def correct(row):
    return (
        row.get("correctness") is True
        and row.get("compiled") is True
        and not row.get("decoy")
        and row.get("status", "").lower().split(".")[-1] == "completed"
    )


def first_correct(rows):
    ordered = sorted(rows, key=lambda r: r["turn_idx"])
    indices = [r["turn_idx"] for r in ordered]
    if indices != list(range(len(indices))):
        raise ValueError("repair attribution needs contiguous, unique turns starting at zero")
    return next((r["turn_idx"] for r in ordered if correct(r)), None)


def phase(row):
    if correct(row):
        return "correct"
    error = str(row.get("environment_error_message") or "")
    if "WorkerProcessCrashed" in error:
        return "native_crash_unresolved"
    if "truncated" in row.get("status", "").lower():
        return "budget_truncated"
    if "precheck" in error.lower():
        return "precheck"
    if "Compilation failed" in error or row.get("compiled") is False:
        return "compile"
    if row.get("decoy"):
        return "policy_rejection"
    if "incorrect results" in error.lower() or "mismatch" in error.lower():
        return "output_mismatch"
    return "other_failure"


def error_tolerant_native_candidates(structure):
    """Keep lexical evidence for complete native functions with syntax errors.

    This does not call malformed code valid. Exact function identity and unique
    old/new token contexts are still required. Python AST failures and incomplete
    submitted programs remain unknown. No numerical reward is assigned.
    """
    turns = {t["turn_idx"]: t for t in structure["turns"]}
    anchor = max(turns)
    terminal = turns[anchor]
    result = []
    for transition in structure["transitions"]:
        before, after = turns[transition["from_turn"]], turns[transition["to_turn"]]
        if not all(t["complete_sections"] and t["valid_modelnew"] for t in [before, after, terminal]):
            continue
        left = {c["id"]: c for c in before["components"]}
        right = {c["id"]: c for c in after["components"]}
        for change in transition["changes"]:
            if change["status"] != "syntax_uncertain":
                continue
            a, b = left[change["before"]], right[change["after"]]
            if a["section"] == "MODEL_NEW" or a["kind"] == "file_context":
                continue
            ends = [c for c in terminal["components"] if identity_key(c) == identity_key(b)]
            if len(ends) != 1 or not ends[0]["valid_syntax"]:
                continue
            hunks, skip = edit_hunks(a, b)
            if skip:
                continue
            for h in hunks:
                if hunk_state(h, a) != "reverted" or hunk_state(h, b) != "present":
                    continue
                result.append(
                    {
                        "basis": "error_tolerant_native_tokens",
                        "introduced_turn": after["turn_idx"],
                        "section": b["section"],
                        "component": b["qualified_name"],
                        "hunk": h,
                        "before_syntax_valid": a["valid_syntax"],
                        "after_syntax_valid": b["valid_syntax"],
                        "anchor_state": hunk_state(h, ends[0]),
                        "causal_contribution": "not_evaluated",
                    }
                )
    return result


def inspect_case(rows):
    anchor = first_correct(rows)
    if anchor is None or anchor == 0:
        raise ValueError("case must be initially incorrect and later correct")
    prefix = sorted(rows, key=lambda r: r["turn_idx"])[: anchor + 1]
    structure = analyze_trajectory(prefix)
    history, terminal, uncertain = [], [], []
    for event in structure["edit_events"]:
        state = next((v["state"] for v in event["observations"] if v["turn_idx"] == anchor), "unobserved")
        record = {**event, "anchor_state": state, "causal_contribution": "not_evaluated"}
        if event["introduced_turn"] == anchor:
            terminal.append(record)
        elif state == "present":
            history.append(record)
        else:
            uncertain.append(record)
    gaps = collections.Counter()
    for turn in structure["turns"]:
        if not turn["complete_sections"]:
            gaps["incomplete_selected_program"] += 1
        if not turn["valid_modelnew"]:
            gaps["missing_or_unparseable_ModelNew"] += 1
        for section in turn["sections"].values():
            if not section["syntax_valid"]:
                gaps["section_syntax_uncertain"] += 1
    changed, changed_with_hunks = 0, 0
    for transition in structure["transitions"]:
        gaps["unmatched_before_components"] += len(transition["unmatched_before"])
        gaps["unmatched_after_components"] += len(transition["unmatched_after"])
        gaps["rename_candidates_not_aligned"] += len(transition["rename_candidates"])
        for change in transition["changes"]:
            if change["status"] == "changed":
                changed += 1
                changed_with_hunks += bool(change["hunks"])
                if not change["hunks"]:
                    gaps[change["hunks_skip_reason"] or "local_anchors_unavailable"] += 1
            if change["status"] == "syntax_uncertain":
                gaps["component_syntax_uncertain"] += 1
    history_local = [e for e in history if e["event_type"] == "anchored_edit"]
    history_added = [e for e in history if e["event_type"] == "component_added"]
    terminal_local = [e for e in terminal if e["event_type"] == "anchored_edit"]
    complete = all(t["complete_sections"] and t["valid_modelnew"] for t in structure["turns"])
    syntax = complete and all(s["syntax_valid"] for t in structure["turns"] for s in t["sections"].values())
    if history_local:
        category = "earlier_local_edit_retained"
    elif history_added:
        category = "earlier_whole_component_only"
    elif anchor >= 2:
        category = "intermediate_turn_without_retained_edit"
    elif terminal_local:
        category = "first_correct_local_edit_only"
    else:
        category = "first_correct_without_local_alignment"
    return {
        "anchor_turn": anchor,
        "category": category,
        "complete_program_prefix": complete,
        "syntax_valid_prefix": syntax,
        "phase_path": [phase(r) for r in prefix],
        "history_local_count": len(history_local),
        "history_added_count": len(history_added),
        "terminal_local_count": len(terminal_local),
        "changed_components": changed,
        "changed_components_with_hunks": changed_with_hunks,
        "gaps": {k: v for k, v in gaps.items() if v},
        "retained_history_candidates": history,
        "terminal_candidates": terminal,
        "nonretained_or_unknown_history": uncertain,
        "syntax_fallback_candidates": error_tolerant_native_candidates(structure),
        "allocation": None,
        "interpretation": "First-correct retrospective syntax retention only; no causal label or reward assigned.",
        "structure": structure,
    }


def provenance():
    root = Path(__file__).resolve().parents[3]
    paths = [
        Path(__file__),
        Path(__file__).with_name("structure.py"),
        Path(__file__).with_name("analyze.py"),
        root / "examples/kernel_agent/utils.py",
    ]
    return {str(p.relative_to(root)): file_sha(p) for p in paths}


def assert_parser_matches(snapshot_root):
    """The selected-code parser must match the code used to evaluate the dump."""
    import ast

    current = Path(__file__).resolve().parents[3] / "examples/kernel_agent/utils.py"
    frozen = snapshot_root / "examples/kernel_agent/utils.py"
    names = {
        "_strip_think_blocks",
        "_section_block_pattern",
        "_find_last_complete_section_group",
        "_find_last_sections",
        "parse_cuda_agent_response",
        "extract_cuda_agent_kernel_code",
    }

    def material(path):
        tree = ast.parse(path.read_text())
        functions = {
            n.name: ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
        }
        constants = [
            ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CUDA_SECTIONS" for t in n.targets)
        ]
        assert set(functions) == names
        return functions, constants

    if material(current) != material(frozen):
        raise ValueError("current selected-code parser differs from the evaluated snapshot")
    return {
        "current_sha256": file_sha(current),
        "evaluated_sha256": file_sha(frozen),
        "selected_code_functions_ast_identical": True,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def supplement(args):
    """Add lexical observations without overwriting a completed strict audit."""
    root = args.existing_audit.resolve()
    source_before = provenance()
    original = json.loads((root / "manifest.json").read_text())
    for path, digest in original["implementation"].items():
        if not path.endswith("repair_credit_audit.py") and source_before[path] != digest:
            raise ValueError(f"shared parser differs from original audit: {path}")
    if args.output.exists():
        raise FileExistsError(args.output)
    index_path = root / "trajectory_index.json"
    index_sha = file_sha(index_path)
    index = json.loads(index_path.read_text())
    results = []
    for item in index:
        if not item["gaps"].get("component_syntax_uncertain"):
            continue
        path = root / "cases" / f"{item['case_id']}.json"
        digest = file_sha(path)
        raw = json.loads(path.read_text())
        prefix = [r for r in raw["raw_turns"] if r["turn_idx"] <= raw["anchor_turn"]]
        events = error_tolerant_native_candidates(analyze_trajectory(prefix))
        if file_sha(path) != digest:
            raise RuntimeError(f"case changed during analysis: {path}")
        results.append(
            {
                "case_id": item["case_id"],
                "case_sha256": digest,
                "anchor_turn": item["anchor_turn"],
                "strict_category": item["category"],
                "events": events,
            }
        )
    new_history = [
        r["case_id"]
        for r in results
        if r["strict_category"] != "earlier_local_edit_retained"
        and any(e["introduced_turn"] < r["anchor_turn"] and e["anchor_state"] == "present" for e in r["events"])
    ]
    terminal = [
        r["case_id"]
        for r in results
        if any(e["introduced_turn"] == r["anchor_turn"] and e["anchor_state"] == "present" for e in r["events"])
    ]
    stable = index_sha == file_sha(index_path) and source_before == provenance()
    result = {
        "audited_syntax_gap_cases": len(results),
        "cases_with_additional_events": sum(bool(r["events"]) for r in results),
        "new_historical_local_cases": new_history,
        "cases_with_terminal_fallback": terminal,
        "results": results,
        "implementation": source_before,
        "index_sha256": index_sha,
        "implementation_and_input_stable": stable,
        "interpretation": "Supplement to strict audit; lexical candidate retention only, no reward or causal labels.",
    }
    if not stable:
        raise RuntimeError("inputs or code changed; no supplement saved")
    with args.output.open("x") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)


def run(args):
    import torch

    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    started = time.monotonic()
    code_before = provenance()
    parser_provenance = assert_parser_matches(args.runtime_root)
    source_before = file_sha(args.input)
    saved = torch.load(args.input, map_location="cpu", weights_only=False)
    groups = collections.defaultdict(list)
    references = {}
    for raw in saved["samples"]:
        row = normalize(raw)
        key = (row["dataset"], row["group_id"])
        groups[key].append(row)
        label = raw.get("label") or {}
        if isinstance(label, dict) and label.get("ground_truth"):
            references[key] = label["ground_truth"]
    del saved
    index, all_outcomes = [], []
    for key, rows in sorted(groups.items(), key=lambda pair: (int(pair[0][0]), int(pair[0][1]))):
        assert len(rows) == 3, (key, len(rows))
        assert len({(r["problem_id"], r["reference_sha256"]) for r in rows}) == 1, key
        anchor = first_correct(rows)
        identity = {
            "level": int(key[0]),
            "group_id": int(key[1]),
            "problem_id": rows[0]["problem_id"],
            "reference_sha256": rows[0]["reference_sha256"],
        }
        all_outcomes.append(
            {
                **identity,
                "anchor_turn": anchor,
                "phase_path": [phase(r) for r in sorted(rows, key=lambda r: r["turn_idx"])],
            }
        )
        if anchor is None or anchor == 0:
            continue
        record = inspect_case(rows)
        case_id = f"l{key[0]}_g{int(key[1]):04d}"
        record.update(identity=identity, case_id=case_id, reference_code=references.get(key), raw_turns=rows)
        record["structure"]["identity"] = ["TRLOO_step100", *key]
        write_json(output / "cases" / (case_id + ".json"), public(record))
        compact = {
            k: v
            for k, v in record.items()
            if k
            not in {
                "structure",
                "raw_turns",
                "reference_code",
                "retained_history_candidates",
                "terminal_candidates",
                "nonretained_or_unknown_history",
                "syntax_fallback_candidates",
            }
        }
        index.append(compact)
        if len(index) % 25 == 0:
            print(json.dumps({"analyzed_candidates": len(index), "last": case_id}), flush=True)
    assert len(all_outcomes) == 2000, len(all_outcomes)
    assert collections.Counter(r["level"] for r in all_outcomes) == {1: 800, 2: 800, 3: 400}
    hist = collections.Counter("none" if r["anchor_turn"] is None else str(r["anchor_turn"] + 1) for r in all_outcomes)
    summary = {
        "trajectory_total": len(all_outcomes),
        "first_correct_turn_counts": dict(hist),
        "initial_wrong_later_correct": len(index),
        "candidate_fraction_percent": 100 * len(index) / len(all_outcomes),
        "category_counts": dict(collections.Counter(r["category"] for r in index)),
        "complete_program_prefix": sum(r["complete_program_prefix"] for r in index),
        "syntax_valid_prefix": sum(r["syntax_valid_prefix"] for r in index),
        "history_local_event_count": sum(r["history_local_count"] for r in index),
        "history_added_event_count": sum(r["history_added_count"] for r in index),
        "terminal_local_event_count": sum(r["terminal_local_count"] for r in index),
        "gaps": dict(sum((collections.Counter(r["gaps"]) for r in index), collections.Counter())),
        "levels": {},
        "training_rewards_modified": False,
        "candidate_code_executed": False,
    }
    for level in [1, 2, 3]:
        population = [r for r in all_outcomes if r["level"] == level]
        cases = [r for r in index if r["identity"]["level"] == level]
        summary["levels"][str(level)] = {
            "trajectories": len(population),
            "candidates": len(cases),
            "categories": dict(collections.Counter(r["category"] for r in cases)),
            "first_correct_turn_counts": dict(
                collections.Counter(
                    "none" if r["anchor_turn"] is None else str(r["anchor_turn"] + 1) for r in population
                )
            ),
        }
    rng = random.Random(args.review_seed)
    review = []
    for category in sorted(summary["category_counts"]):
        pool = sorted([r["case_id"] for r in index if r["category"] == category])
        selected = sorted(rng.sample(pool, min(args.review_per_category, len(pool))))
        review.extend(
            {"case_id": c, "stratum": category, "stratum_size": len(pool), "reason": "seeded_stratified_sample"}
            for c in selected
        )
    for case_id in ["l3_g0006"]:
        if (output / "cases" / (case_id + ".json")).exists() and not any(r["case_id"] == case_id for r in review):
            review.append({"case_id": case_id, "reason": "previously_manually_observed_dtype_fix"})
    for selected in review:
        record = json.loads((output / "cases" / (selected["case_id"] + ".json")).read_text())
        (output / "cases" / (selected["case_id"] + ".md")).write_text(trajectory_markdown(record["structure"]))
    write_json(output / "trajectory_index.json", index)
    write_json(output / "outcomes.json", all_outcomes)
    write_json(output / "summary.json", summary)
    write_json(
        output / "review_selection.json",
        {
            "seed": args.review_seed,
            "selected": review,
            "interpretation": "Stratified seeded sample plus separately marked purposeful controls; not an unweighted population precision estimate.",
        },
    )
    stable = source_before == file_sha(args.input) and code_before == provenance()
    write_json(
        output / "manifest.json",
        {
            "input": str(args.input.resolve()),
            "input_sha256": source_before,
            "evaluated_runtime_root": str(args.runtime_root.resolve()),
            "parser": parser_provenance,
            "implementation": code_before,
            "implementation_and_input_stable": stable,
            "packages": {
                n: importlib.metadata.version(n)
                for n in ["torch", "tree-sitter", "tree-sitter-cpp", "tree-sitter-cuda"]
            },
            "elapsed_seconds": time.monotonic() - started,
            "argv": sys.argv,
        },
    )
    if not stable:
        raise RuntimeError("input or implementation changed during analysis; results invalid")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--existing-audit", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-seed", type=int, default=20260914)
    parser.add_argument("--review-per-category", type=int, default=8)
    args = parser.parse_args()
    if args.existing_audit:
        supplement(args)
    else:
        if args.runtime_root is None:
            parser.error("--runtime-root is required with --input")
        run(args)


if __name__ == "__main__":
    main()
