"""Analyze trusted rollout .pt / JSONL / extracted g*.json files without executing kernels."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .structure import analyze_trajectory, digest, public


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def implementation_provenance():
    return {
        "implementation": {p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")},
        "response_parser": file_sha(Path(__file__).resolve().parents[3] / "examples/kernel_agent/utils.py"),
    }


def input_provenance(name, path):
    source = {"name": name, "path": str(path)}
    if path.is_file():
        source["sha256"] = file_sha(path)
    else:
        source["files"] = {p.name: file_sha(p) for p in sorted(path.glob("g*.json"))}
    return source


def input_rows(path):
    if path.is_dir():
        paths = sorted(path.glob("g*.json"))
        if not paths:
            raise ValueError(f"no g*.json under {path}")
        for p in paths:
            yield from input_rows(p)
    elif path.suffix == ".pt":
        import torch

        saved = torch.load(path, map_location="cpu", weights_only=False)
        yield from (saved["samples"] if isinstance(saved, dict) else saved)
    elif path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        saved = json.loads(path.read_text())
        yield from (saved["samples"] if isinstance(saved, dict) else saved)


def normalize(row):
    md = row.get("metadata") or {}
    env = row.get("env_result") or md.get("env_result") or {}
    state = env.get("env_state") or env
    label = row.get("label") or {}
    reference = label.get("ground_truth") if isinstance(label, dict) else None
    if reference is None:
        reference = (row.get("reward_model") or {}).get("ground_truth")
    group = row.get("group_id", row.get("trajectory_id"))
    turn = row.get("turn_idx", md.get("turn_idx"))
    if group is None or turn is None:
        raise ValueError("every row must have group_id and turn_idx")
    if not isinstance(row.get("response"), str):
        raise ValueError("full response string is required")
    tokens = row.get("tokens")
    response_length = row.get("response_length", row.get("response_tokens"))
    return {
        "group_id": str(group),
        "turn_idx": int(turn),
        "dataset": str(row.get("dataset", md.get("eval_level", md.get("difficulty", "unspecified")))),
        "problem_id": row.get("problem_id", md.get("problem_id")),
        "problem_family": row.get("problem_family", md.get("problem_family")),
        "problem_name": row.get("problem_name", md.get("problem_name")),
        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest() if reference else None,
        "response": row["response"],
        "response_sha256": hashlib.sha256(row["response"].encode()).hexdigest(),
        "correctness": state.get("correctness", row.get("correct")),
        "compiled": state.get("compiled", row.get("compiled")),
        "decoy": state.get("decoy_kernel", row.get("decoy")),
        "speedup": state.get("speedup", row.get("speedup")),
        "reward_observed": row.get("reward"),
        "error_code": state.get("error_code", state.get("error")),
        "status": str(row.get("status", row.get("sample_status", "unknown"))),
        "response_tokens": response_length,
        "prompt_tokens": (
            len(tokens) - response_length
            if tokens is not None and response_length is not None
            else row.get("prompt_tokens")
        ),
        "saved_model_feedback_available": bool(md.get("model_feedback") or row.get("model_feedback")),
        "saved_environment_feedback_available": bool(env),
        "environment_error_message": state.get("error_message"),
    }


def grouped(name, path):
    groups = collections.defaultdict(list)
    for raw in input_rows(path):
        row = normalize(raw)
        # Reference identity prevents group IDs reused across evaluation levels from colliding.
        key = (name, row["dataset"], row["reference_sha256"], str(row["problem_id"]), row["group_id"])
        groups[key].append(row)
    return groups


def trajectory_markdown(result):
    lines = [
        "# Trajectory structure review",
        "",
        f"Identity: `{result['identity']}`",
        "",
        "Only syntactic evidence is reported. No semantic equivalence or causal credit is inferred.",
        "",
        "| Turn (1-based) | Section selection | ModelNew valid | Native/Python syntax errors | Components |",
        "|---|---|---|---:|---:|",
    ]
    for t in result["turns"]:
        lines.append(
            f"| {t['turn_idx']+1} | {t['selection_mode']} | {t['valid_modelnew']} | {sum(len(s['errors']) for s in t['sections'].values())} | {len(t['components'])} |"
        )
    lines += ["", "## Component changes", ""]
    for tr in result["transitions"]:
        lines += [f"### Turn {tr['from_turn']+1} → {tr['to_turn']+1}", ""]
        for name, s in tr["sections"].items():
            lines.append(f"- {name}: comparable={s['comparable']}, token similarity={s['token_similarity']}")
        lines.append("")
        for change in tr["changes"]:
            if change["status"] == "unchanged":
                continue
            lines += [f"`{change['before']}` → `{change['after']}` ({change['status']}; {change['alignment']})", ""]
            if change["source_diff"]:
                lines += ["````diff", change["source_diff"], "````", ""]
        if tr["unmatched_before"]:
            lines += [f"Unmatched before: `{tr['unmatched_before']}`", ""]
        if tr["unmatched_after"]:
            lines += [f"Unmatched after: `{tr['unmatched_after']}`", ""]
        if tr["reconnected_after"]:
            lines += [f"Reconnected historical components: `{tr['reconnected_after']}`", ""]
        if tr["rename_candidates"]:
            lines += [
                "Rename candidates (not accepted as lineage identity):",
                "",
                "```json",
                json.dumps(tr["rename_candidates"], indent=2),
                "```",
                "",
            ]
    lines += [
        "## Modification retention",
        "",
        "| Component | Introduced turn | Subsequent observations |",
        "|---|---:|---|",
    ]
    for event in result["edit_events"]:
        observations = ", ".join(f"T{o['turn_idx']+1}:{o['state']}" for o in event["observations"])
        lines.append(f"| {event['section']}::{event['component']} | {event['introduced_turn']+1} | {observations} |")
    lines += [
        "",
        "## Component-version recurrence",
        "",
        "```json",
        json.dumps(result["component_version_recurrences"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def run(args):
    provenance = implementation_provenance()
    started_at = datetime.now(timezone.utc).isoformat()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    started = time.monotonic()
    counters = collections.Counter()
    modes = collections.Counter()
    states = collections.Counter()
    syntax = collections.Counter()
    sources = []
    index = []
    for spec in args.input:
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path).resolve()
        source = input_provenance(name, path)
        sources.append(source)
        groups = grouped(name, path)
        for key, turns in sorted(groups.items(), key=lambda x: str(x[0])):
            if args.group_id and key[-1] not in args.group_id:
                continue
            if args.max_trajectories is not None and counters["trajectories"] >= args.max_trajectories:
                break
            result = analyze_trajectory(turns)
            result["identity"] = list(key)
            file = digest(key)[:20]
            destination = output / "trajectories" / file
            if destination.with_suffix(".json").exists():
                raise ValueError(f"duplicate trajectory identity across inputs: {key}; use distinct input names")
            destination.with_suffix(".json").write_text(json.dumps(public(result), ensure_ascii=False, indent=2))
            if key[-1] in args.review_group:
                destination.with_suffix(".md").write_text(trajectory_markdown(result))
            stats = collections.Counter()
            for turn in result["turns"]:
                modes[turn["selection_mode"]] += 1
                stats["turns"] += 1
                stats["components"] += len(turn["components"])
                stats["calls"] += len(turn["calls"])
                stats["complete_protocol_turns"] += int(
                    turn["selection_mode"] == "last_complete_group" and turn["valid_modelnew"]
                )
                stats["complete_selected_turns"] += int(turn["complete_sections"] and turn["valid_modelnew"])
                for section, analysis in turn["sections"].items():
                    syntax[f"{section}:valid" if analysis["syntax_valid"] else f"{section}:errors"] += 1
            stats["edit_events"] = len(result["edit_events"])
            stats["nonadjacent_retention_events"] = sum(
                bool(e["nonadjacent_retained_turns"]) for e in result["edit_events"]
            )
            stats["reintroduction_events"] = sum(bool(e["reintroduced_turns"]) for e in result["edit_events"])
            stats["version_recurrences"] = len(result["component_version_recurrences"])
            for e in result["edit_events"]:
                states.update(o["state"] for o in e["observations"])
            counters.update(stats)
            counters["trajectories"] += 1
            index.append(
                {
                    "identity": list(key),
                    "path": str(destination.with_suffix(".json").relative_to(output)),
                    **dict(stats),
                }
            )
            if counters["trajectories"] % 20 == 0:
                print(
                    f"Analyzed {counters['trajectories']} trajectories in {time.monotonic()-started:.1f}s", flush=True
                )
    if not index:
        raise ValueError("no matching trajectories")
    (output / "trajectory_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    summary = {
        "counts": dict(counters),
        "section_selection": dict(modes),
        "syntax_coverage": dict(syntax),
        "edit_observation_states": dict(states),
        "elapsed_s": time.monotonic() - started,
        "selection": {"group_id": args.group_id, "max_trajectories": args.max_trajectories},
        "interpretation": "Development-pool structural diagnostics, not natural E1.1 prevalence or causal contribution.",
    }
    stable = provenance == implementation_provenance() and all(
        source == input_provenance(source["name"], Path(source["path"])) for source in sources
    )
    manifest = {
        "schema_version": 1,
        "inputs": sources,
        "summary": summary,
        "started_at_utc": started_at,
        "python": sys.version,
        "executable": sys.executable,
        "code_and_inputs_unchanged": stable,
        "dependencies": {
            p: importlib.metadata.version(p) for p in ["tree-sitter", "tree-sitter-cpp", "tree-sitter-cuda"]
        },
        **provenance,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if not stable:
        raise RuntimeError("Code or inputs changed during analysis; this output is invalid and must be rerun")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", action="append", required=True, help="NAME=trusted .pt, .jsonl, .json, or directory of g*.json"
    )
    p.add_argument(
        "--output-dir", type=Path, required=True, help="New output directory; existing evidence is not overwritten"
    )
    p.add_argument(
        "--group-id", action="append", default=[], help="Optional exact group ID selection (applies to each input)"
    )
    p.add_argument(
        "--review-group", action="append", default=[], help="Also render full source diffs as Markdown for these IDs"
    )
    p.add_argument("--max-trajectories", type=int)
    args = p.parse_args()
    if args.max_trajectories is not None and args.max_trajectories < 1:
        p.error("--max-trajectories must be positive")
    run(args)


if __name__ == "__main__":
    main()
