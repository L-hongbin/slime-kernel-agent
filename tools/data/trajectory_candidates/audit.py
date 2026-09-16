"""Validate and render explicit human source-review annotations, without inferring labels."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from .screen import sha


def raw_timing_evidence(screened, structure, source_manifest):
    model, group = screened["identity"][0], screened["identity"][-1]
    source = next((s for s in source_manifest["inputs"] if s["name"] == model), None)
    if source is None or "files" not in source or not group.isdigit():
        return {"available": False}
    name = f"g{int(group):03}.json"
    expected = source["files"].get(name)
    if expected is None:
        return {"available": False}
    path = Path(source["path"]) / name
    if sha(path) != expected:
        raise ValueError(f"raw input changed: {path}")
    by_turn = {t["turn_idx"]: t for t in structure["turns"]}
    seen = set()
    timings = []
    for row in json.loads(path.read_text()):
        idx = row.get("turn_idx", (row.get("metadata") or {}).get("turn_idx"))
        if idx not in by_turn:
            continue
        if idx in seen:
            raise ValueError("duplicate raw turn")
        seen.add(idx)
        response_sha = hashlib.sha256(row["response"].encode()).hexdigest()
        if response_sha != by_turn[idx]["observation"]["response_sha256"]:
            raise ValueError("raw/structure response mismatch")
        env = row.get("env_result") or (row.get("metadata") or {}).get("env_result") or {}
        state = env.get("env_state", env)
        metadata = state.get("metadata") or {}
        extra = env.get("env_extra_info") or {}
        ref = state.get("reference_runtime")
        kernel = state.get("kernel_runtime")
        speed = state.get("speedup")
        admissible = next(t for t in screened["timeline"] if t["turn_idx"] == idx)["state"] == "correct_timed"
        consistent = None
        if admissible and all(
            isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (ref, kernel, speed)
        ):
            consistent = math.isclose(ref / kernel, speed, rel_tol=1e-6)
            if not consistent:
                raise ValueError(f"inconsistent timing ratio: {path} turn {idx}")
        timings.append(
            {
                "turn_idx": idx,
                "reference_runtime": ref,
                "kernel_runtime": kernel,
                "speedup": speed,
                "admissible_timing": admissible,
                "ratio_consistent": consistent,
                "num_perf_trials": metadata.get("num_perf_trials"),
                "kernel_perf_cv": extra.get("kernel_perf_cv"),
                "profiling_kernels": (metadata.get("profiling") or {}).get("kernels", []),
                "profiling_total_cuda_time_us": (metadata.get("profiling") or {}).get("total_cuda_time_us"),
            }
        )
    if seen != set(by_turn):
        raise ValueError("raw/structure turn coverage mismatch")
    refs = [t["reference_runtime"] for t in timings if t["ratio_consistent"]]
    return {
        "available": True,
        "path": str(path),
        "sha256": expected,
        "timings": timings,
        "reference_runtime_span_ratio": max(refs) / min(refs) - 1 if refs else None,
    }


def run(root, annotations_path, output):
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["code_and_inputs_unchanged"]:
        raise ValueError("input screening run was not stable")
    output.mkdir(parents=True, exist_ok=False)
    annotation_sha = sha(annotations_path)
    annotation = json.loads(annotations_path.read_text())
    cases = annotation["cases"]
    selection = json.loads((root / "review_selection.json").read_text())

    def key(r):
        return r["model"], str(r["group"])

    by_key = {key(r): r for r in cases}
    if len(by_key) != len(cases):
        raise ValueError("duplicate manual annotation")
    if set(by_key) != {(s["identity"][0], s["identity"][-1]) for s in selection}:
        raise ValueError("incomplete audit annotation coverage")
    structure_root = Path(manifest["input_dir"])
    structure_manifest_path = structure_root / "manifest.json"
    if sha(structure_manifest_path) != manifest["input_manifest_sha256"]:
        raise ValueError("structure manifest changed since screening")
    structure_manifest = json.loads(structure_manifest_path.read_text())
    results = []
    verdicts = collections.Counter()
    decisions = collections.Counter()
    priorities = collections.Counter()
    for selected in selection:
        row = json.loads((root / "trajectories" / f"{selected['trajectory_id']}.json").read_text())
        if row["identity"] != selected["identity"] or row["trajectory_id"] != selected["trajectory_id"]:
            raise ValueError("audit selection identity mismatch")
        human = by_key[(row["identity"][0], row["identity"][-1])]
        if (
            human["verdict"] not in {"candidate", "control", "indeterminate"}
            or not human["note"]
            or not human["evidence"]
        ):
            raise ValueError("invalid manual annotation")
        source_path = Path(row["source_path"])
        source_key = source_path.relative_to(structure_root).as_posix()
        if sha(source_path) != manifest["input_files"][source_key]:
            raise ValueError("structure input changed since screening")
        source = json.loads(source_path.read_text())
        evidence = raw_timing_evidence(row, source, structure_manifest)
        record = human | {
            "identity": row["identity"],
            "trajectory_id": row["trajectory_id"],
            "source_path": row["source_path"],
            "candidate_path": str((root / "trajectories" / f"{row['trajectory_id']}.json").resolve()),
            "initial_flags": selected.get("initial_flags", selected["flags"]),
            "current_flags": row["flags"],
            "current_uncertain_windows": row["uncertain_windows"],
            "stratum": selected["stratum"],
            "raw_timing": evidence,
            "causal_contribution": "not_tested",
        }
        results.append(record)
        verdicts[human["verdict"]] += 1
        priorities[human["priority"]] += 1
        if any(record["initial_flags"].values()):
            decisions[human["decision_class"]] += 1
    lines = [
        "# E1.1 人工源代码审计",
        "",
        annotation["scope"],
        "",
        "此表由明确写下的人工标注渲染，不由候选 flag 自动生成 verdict；不是盲审或因果验证",
        "",
    ]
    for model in sorted({r["model"] for r in results}):
        lines += [f"# {model}", "", "| Group | 筛选审计 | 决策类型 | 优先级 | 源码核对结论 |", "|---|---|---|---|---|"]
        for r in [r for r in results if r["model"] == model]:
            lines.append(
                f"| [{r['group']}]({root.resolve()}/review/{r['trajectory_id']}.md) | {r['verdict']} | {r['decision_class']} | {r['priority']} | {r['note']} |"
            )
        for r in [r for r in results if r["model"] == model]:
            lines += [
                "",
                f"## Group {r['group']}",
                "",
                r["note"],
                "",
                f"人工定位：{r['evidence']}",
                "",
                f"[完整源码目录]({root.resolve()}/review_sources/{r['trajectory_id']}) · [结构 JSON]({r['source_path']})",
                "",
                "```json",
                json.dumps(
                    {
                        "initial_flags": r["initial_flags"],
                        "current_flags": r["current_flags"],
                        "raw_timing": r["raw_timing"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
    summary = {
        "reviewed": len(results),
        "unique_reference_hashes": len({r["identity"][2] for r in results}),
        "manual_verdicts": dict(verdicts),
        "initial_candidate_decision_classes": dict(decisions),
        "priorities": dict(priorities),
        "all_raw_responses_matched": all(r["raw_timing"]["available"] for r in results),
        "input_manifest_sha256": sha(root / "manifest.json"),
        "annotations_sha256": annotation_sha,
        "audit_implementation_sha256": sha(Path(__file__)),
        "claim_boundary": "Non-blind, task-diverse diagnostic review; no population precision/recall or causal estimate",
    }
    if annotation_sha != sha(annotations_path):
        raise RuntimeError("annotations changed during audit")
    (output / "cases.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (output / "review.md").write_text("\n".join(lines))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.input_dir, a.annotations, a.output_dir)


if __name__ == "__main__":
    main()
