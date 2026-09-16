"""Screen previously extracted trajectories; never execute generated code."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Protocol:
    gain_margin: float = 0.05
    rewrite_similarity: float = 0.50
    review_per_model: int = 20
    seed: str = "e11-development-audit-20260908"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def structure_observed(turn):
    return (
        turn["complete_sections"]
        and turn["valid_modelnew"]
        and all(s["syntax_valid"] for s in turn["sections"].values())
    )


def version_hash(turn):
    if not structure_observed(turn):
        return None
    return stable_hash(
        sorted((c["section"], c["qualified_name"], c["kind"], c["syntax_hash"]) for c in turn["components"])
    )


def observation_state(turn):
    o = turn["observation"]
    error = o.get("error_code")
    if str(o.get("status", "")).lower() in {"aborted", "truncated", "cancelled"}:
        return "unobserved", None
    if not turn["complete_sections"] and error in {None, "VALIDATION_ERROR"}:
        return "unobserved", None
    if error and "TIMEOUT" in str(error).upper():
        return "execution_timeout_unattributed", None
    message = str(o.get("environment_error_message") or "")
    if "WorkerProcessCrashed" in message or "result channel reached EOF" in message:
        return "worker_crash_unattributed", None
    if error and any(word in str(error).upper() for word in ("RESOURCE", "SERVER", "CONNECTION", "INFRA")):
        return "environment_uncertain", None
    if o.get("decoy") is True or error == "DECOY_KERNEL_DETECTED":
        return "evaluation_rejected", None
    if o.get("correctness") is True:
        speed = o.get("speedup")
        if o.get("compiled") is not True or o.get("decoy") is not False or error:
            return "inconsistent_result", None
        if isinstance(speed, (int, float)) and not isinstance(speed, bool) and math.isfinite(speed) and speed > 0:
            return "correct_timed", float(speed)
        return "correct_untimed", None
    if o.get("correctness") is False or o.get("compiled") is False or error:
        return "observed_failure", None
    return "unobserved", None


def actual_arguments(arguments):
    # Historical tree-sitter extraction included comment nodes among named children.
    return [a for a in (arguments or []) if not a.lstrip().startswith(("//", "/*"))]


def decision_facts(turn):
    """Literal call-site facts and named constants, not inferred semantic equivalence."""
    facts = []
    components = {c["id"]: c for c in turn["components"]}
    for call in turn["calls"]:
        c = components[call["from"]]
        target = call["target"]
        args = actual_arguments(call.get("arguments", []))
        call_facts = []
        if target in {"cublasSetMathMode", "cublasLtMatmulDescCreate", "cudnnSetConvolutionMathType"}:
            call_facts.append(("math_mode", args[1:2]))
        if target.startswith("cublas") and any("CUBLAS_COMPUTE_" in a for a in args):
            if target != "cublasLtMatmulDescCreate":
                call_facts.append(("math_mode", [a for a in args if "CUBLAS_COMPUTE_" in a]))
        if call["kind"] == "kernel_launch":
            call_facts.append(("launch_config", actual_arguments(call.get("launch_config"))))
        if target in {
            "cublasSgemm",
            "cublasDgemm",
            "cublasHgemm",
            "cublasSgemmEx",
            "cublasGemmEx",
            "cublasGemmStridedBatchedEx",
        }:
            call_facts.append(("transpose_layout", args[1:3]))
        if (
            target == "cublasLtMatmulDescSetAttribute"
            and len(args) > 2
            and any(s in args[1] for s in ("TRANSA", "TRANSB", "EPILOGUE"))
        ):
            call_facts.append(("transpose_layout", args[1:3]))
        for kind, values in call_facts:
            facts.append(
                {
                    "kind": kind,
                    "section": c["section"],
                    "component": c["qualified_name"],
                    "line": call["line"],
                    "target": target,
                    "values": values,
                    "source_syntax_valid": c["valid_syntax"] and turn["sections"][c["section"]]["syntax_valid"],
                }
            )
    for c in turn["components"]:
        if c["kind"] == "macro" and any(x in c["name"].upper() for x in ("TILE", "BLOCK", "WARP", "VEC", "CHUNK")):
            facts.append(
                {
                    "kind": "named_tiling_constant",
                    "section": c["section"],
                    "component": c["qualified_name"],
                    "line": c["line"],
                    "source": c["source"],
                }
            )
    return facts


def reachable_name_candidates(turn):
    reached = {
        c["id"]
        for c in turn["components"]
        if c["section"] == "MODEL_NEW" and c["qualified_name"] == "ModelNew::forward"
    }
    edges = collections.defaultdict(set)
    for call in turn["calls"]:
        if call["resolution"] == "unique_name_candidate":
            edges[call["from"]].update(call["candidate_targets"])
    pending = list(reached)
    while pending:
        for target in edges[pending.pop()] - reached:
            reached.add(target)
            pending.append(target)
    return reached


def cues(event):
    h = event.get("hunk", {})
    changed = " ".join(h.get("before", []) + h.get("after", []))
    context = " ".join(h.get("left_context", []) + h.get("right_context", []))
    code = (changed + " " + context + " " + event["component"]).upper()
    found = []
    for kind, terms in [
        (
            "math_mode",
            ("TF32", "MATH_MODE", "SETMATHMODE", "CUBLAS_COMPUTE_", "SETCONVOLUTIONMATHTYPE", "CUDNN_TENSOR_OP_MATH"),
        ),
        ("tiling_launch", ("TILE", "BLOCK", "WARP", "<<<", "THREADIDX", "BLOCKDIM")),
        ("transpose_layout", ("CUBLAS_OP_", "TRANSPOSE", "LEADING", "LAYOUT")),
        ("validation_interface", ("ICHECK", "CHECK_", "DTYPE", "DATA_PTR", "SHAPE", "CAST")),
    ]:
        if any(term in code for term in terms):
            found.append(kind)
    return found or ["unclassified_syntax"]


def screen_trajectory(result, protocol):
    turns = sorted(result["turns"], key=lambda t: t["turn_idx"])
    by_turn = {t["turn_idx"]: t for t in turns}
    timeline = []
    best = None
    best_turn = None
    seen_versions = collections.defaultdict(list)
    for t in turns:
        idx = t["turn_idx"]
        state, speed = observation_state(t)
        version = version_hash(t)
        old_versions = list(seen_versions.get(version, [])) if version else []
        raw_gain = speed / best - 1 if speed is not None and best is not None else None
        label = state
        if speed is not None:
            if best is None:
                label = "first_correct_timed"
            elif speed > best * (1 + protocol.gain_margin):
                label = "measured_new_best"
            elif raw_gain > 0:
                label = "small_new_best"
            elif speed < best * (1 - protocol.gain_margin):
                label = "below_historical_best"
            else:
                label = "within_best_margin"
        qualifies = label == "measured_new_best" and version is not None and not old_versions
        warnings = []
        if label == "measured_new_best" and old_versions:
            warnings.append("earlier_code_version_measured_gain")
        if speed is not None and version is None:
            warnings.append("correct_result_but_structure_unavailable")
        if timeline and idx != timeline[-1]["turn_idx"] + 1:
            warnings.append("missing_turn_indices")
        timeline.append(
            {
                "turn_idx": idx,
                "state": state,
                "label": label,
                "speedup": speed,
                "best_before": best,
                "best_turn_before": best_turn,
                "relative_gain": raw_gain,
                "breakthrough_candidate": qualifies,
                "version_hash": version,
                "same_version_earlier_turns": old_versions,
                "warnings": warnings,
                "error_code": t["observation"].get("error_code"),
                "structure_observed": structure_observed(t),
                "decision_facts": decision_facts(t),
            }
        )
        if speed is not None and (best is None or speed > best):
            best = speed
            best_turn = idx
        if version:
            seen_versions[version].append(idx)
    times = {t["turn_idx"]: t for t in timeline}
    reachable = {t["turn_idx"]: reachable_name_candidates(t) for t in turns}

    def observed_window(start, end):
        return all(
            i in times and times[i]["state"] in {"correct_timed", "observed_failure", "evaluation_rejected"}
            for i in range(start, end + 1)
        )

    decline = []
    rewrite = []
    reuse = []
    uncertain = []
    recoveries = []
    for t in timeline:
        idx = t["turn_idx"]
        baseline = t["best_before"]
        bt = t["best_turn_before"]
        if baseline is None:
            continue
        regressed = t["state"] in {"observed_failure", "evaluation_rejected"} or (
            t["speedup"] is not None and t["speedup"] < baseline * (1 - protocol.gain_margin)
        )
        if regressed:
            future = next(
                (
                    u
                    for u in timeline
                    if u["turn_idx"] > idx
                    and u["breakthrough_candidate"]
                    and u["speedup"] > baseline * (1 + protocol.gain_margin)
                ),
                None,
            )
            if future:
                row = {
                    "baseline_turn": bt,
                    "regression_turn": idx,
                    "outcome_turn": future["turn_idx"],
                    "baseline_speedup": baseline,
                    "outcome_speedup": future["speedup"],
                    "gain_over_baseline": future["speedup"] / baseline - 1,
                }
                if observed_window(bt, future["turn_idx"]):
                    decline.append(row)
                else:
                    uncertain.append({"kind": "decline_then_breakthrough", "reason": "unobserved_window", **row})
        if (
            t["state"] == "correct_timed"
            and idx - 1 in times
            and times[idx - 1]["state"] in {"observed_failure", "evaluation_rejected"}
        ):
            recoveries.append(
                {
                    "turn_idx": idx,
                    "label": t["label"],
                    "gain_over_best": t["relative_gain"],
                    "same_version_earlier_turns": t["same_version_earlier_turns"],
                }
            )
    for transition in result["transitions"]:
        t = transition["to_turn"]
        before = transition["from_turn"]
        s = transition["sections"].get("CUDA_KERNELS", {})
        similarity = s.get("token_similarity")
        if not s.get("comparable") or similarity is None or similarity >= protocol.rewrite_similarity:
            continue
        if not (structure_observed(by_turn[t]) and structure_observed(by_turn[before])):
            continue
        baseline = times[t]["best_before"]
        if baseline is None:
            continue
        u = next(
            (
                u
                for u in timeline
                if u["turn_idx"] > t
                and u["breakthrough_candidate"]
                and u["speedup"] > baseline * (1 + protocol.gain_margin)
            ),
            None,
        )
        if u:
            row = {
                "from_turn": before,
                "rewrite_turn": t,
                "outcome_turn": u["turn_idx"],
                "cuda_similarity": similarity,
                "baseline_turn": times[t]["best_turn_before"],
                "gain_over_baseline": u["speedup"] / baseline - 1,
                "rewrite_turn_speedup": times[t]["speedup"],
                "rewrite_already_improved": times[t]["label"] == "measured_new_best",
                "gain_after_rewrite_best": u["speedup"] / max(baseline, times[t]["speedup"] or 0) - 1,
                "evidence_level": "temporal_association_not_rewrite_credit",
            }
            if observed_window(row["baseline_turn"], u["turn_idx"]):
                rewrite.append(row)
            else:
                uncertain.append({"kind": "rewrite_then_breakthrough", "reason": "unobserved_window", **row})
    for event in result["edit_events"]:
        intro = event["introduced_turn"]
        for obs in event["observations"]:
            u = obs["turn_idx"]
            if obs["state"] != "present" or u < intro + 2 or not times[u]["breakthrough_candidate"]:
                continue
            row = {
                "event_id": event["event_id"],
                "introduced_turn": intro,
                "outcome_turn": u,
                "section": event["section"],
                "component": event["component"],
                "lineage": event["lineage"],
                "event_type": event["event_type"],
                "decision_cues": cues(event),
                "observations": [o for o in event["observations"] if o["turn_idx"] <= u],
            }
            comp = next((c for c in by_turn[u]["components"] if c["lineage"] == event["lineage"]), None)
            row["name_path_from_forward"] = bool(comp and comp["id"] in reachable[u])
            if observed_window(intro, u):
                reuse.append(row)
            else:
                uncertain.append({"kind": "nonadjacent_reuse", "reason": "unobserved_window", **row})
    # A missing adjacent response must remain an explicit screening blind spot, even
    # when it destroys all adjacent rewrite comparisons and all tracked hunk matches.
    for u in timeline:
        idx = u["turn_idx"]
        if not u["breakthrough_candidate"] or (idx - 1 in by_turn and structure_observed(by_turn[idx - 1])):
            continue
        previous = next((t for t in reversed(timeline) if t["turn_idx"] < idx and t["structure_observed"]), None)
        if previous:
            uncertain.append(
                {
                    "kind": "unobserved_transition_before_gain",
                    "reason": "missing_adjacent_structure",
                    "last_observed_structural_turn": previous["turn_idx"],
                    "outcome_turn": idx,
                    "cannot_locate_edit_turn": True,
                }
            )
    # A unit groups token edits in one component and introduction/outcome pair, not a proven optimization decision.
    units = {}
    for e in reuse:
        key = (e["introduced_turn"], e["outcome_turn"], e["section"], e["lineage"])
        if key not in units:
            units[key] = {k: v for k, v in e.items() if k not in {"event_id", "decision_cues", "event_type"}} | {
                "event_ids": [],
                "decision_cues": set(),
            }
        units[key]["event_ids"].append(e["event_id"])
        units[key]["decision_cues"].update(e["decision_cues"])
    for u in units.values():
        u["decision_cues"] = sorted(u["decision_cues"])
    flags = {
        "decline_then_breakthrough": bool(decline),
        "nonadjacent_reuse": bool(units),
        "rewrite_then_breakthrough": bool(rewrite),
    }
    return {
        "identity": result["identity"],
        "timeline": timeline,
        "flags": flags,
        "decline_episodes": decline,
        "reuse_units": list(units.values()),
        "rewrite_episodes": rewrite,
        "uncertain_windows": uncertain,
        "recoveries": recoveries,
        "any_candidate": any(flags.values()),
        "evidence_level": "syntactic_temporal_only_not_optimization_strategy",
        "measurement_boundary": "Observed single-run gains only; no significance or causal contribution is established.",
    }


def review_selection(rows, protocol):
    selected = []
    for model in sorted({r["identity"][0] for r in rows}):
        population = [r for r in rows if r["identity"][0] == model]
        ranking = sorted(population, key=lambda r: stable_hash([protocol.seed, r["identity"]]))
        used = set()
        tasks = set()

        def take(pool, n, stratum, *, ranking=ranking, used=used, model=model, tasks=tasks):
            pool_ids = {r["trajectory_id"] for r in pool}
            eligible = [r for r in ranking if r["trajectory_id"] in pool_ids and r["trajectory_id"] not in used]
            for _ in range(n):
                if sum(s["identity"][0] == model for s in selected) >= protocol.review_per_model:
                    break
                if not eligible:
                    break
                row = next((r for r in eligible if r["identity"][2] not in tasks), eligible[0])
                eligible.remove(row)
                used.add(row["trajectory_id"])
                tasks.add(row["identity"][2])
                selected.append(
                    {
                        "trajectory_id": row["trajectory_id"],
                        "identity": row["identity"],
                        "stratum": stratum,
                        "source_path": row["source_path"],
                        "flags": row["flags"],
                    }
                )

        # 10 candidates + 10 controls per model. Candidate slots rotate through the three labels.
        labels = list(population[0]["flags"])
        target = protocol.review_per_model // 2
        for slot in range(target):
            label = labels[slot % len(labels)]
            pool = [r for r in population if r["flags"][label] and r["trajectory_id"] not in used]
            take(pool or [r for r in population if r["any_candidate"]], 1, "candidate:" + label)
        negatives = [r for r in population if not r["any_candidate"]]
        take([r for r in negatives if r["recoveries"]], 4, "control:recovery")
        take(
            [
                r
                for r in negatives
                if any(t["label"] in {"small_new_best", "measured_new_best"} for t in r["timeline"])
            ],
            3,
            "control:other_gain",
        )
        take(negatives, protocol.review_per_model - sum(s["identity"][0] == model for s in selected), "control:other")
    return selected


def review_card(row, result):
    lines = [
        f"# {row['identity'][0]} group {row['identity'][-1]}",
        "",
        f"Source: [full structure JSON]({row['source_path']})",
        "",
        f"Flags: `{row['flags']}`",
        "",
        "| Turn | State | Outcome | Speedup | Best before | Gain (%) | Earlier same code |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for t in row["timeline"]:

        def f(x):
            return "—" if x is None else f"{x:.6g}"

        lines.append(
            f"| {t['turn_idx']+1} | {t['state']} | {t['label']} | {f(t['speedup'])} | {f(t['best_before'])} | {f(100*t['relative_gain']) if t['relative_gain'] is not None else '—'} | {[i+1 for i in t['same_version_earlier_turns']]} |"
        )
    lines += [
        "",
        "## Candidate evidence",
        "",
        "```json",
        json.dumps(
            {k: row[k] for k in ["decline_episodes", "rewrite_episodes", "recoveries", "uncertain_windows"]}, indent=2
        ),
        "```",
        "",
    ]
    events = {e["event_id"]: e for e in result["edit_events"]}
    units = sorted(
        row["reuse_units"],
        key=lambda u: (
            "unclassified_syntax" in u["decision_cues"],
            u["section"] != "CUDA_KERNELS",
            u["introduced_turn"],
        ),
    )
    lines += ["## Reuse units", "", "```json", json.dumps(units, indent=2), "```", ""]
    for unit in units:
        lines += [
            f"### {unit['section']} {unit['component']} T{unit['introduced_turn']+1} → T{unit['outcome_turn']+1}",
            "",
        ]
        for eid in unit["event_ids"]:
            e = events[eid]
            if "hunk" in e:
                h = e["hunk"]
                lines += [
                    "```json",
                    json.dumps(
                        {
                            "event_id": eid,
                            "before": h["before"],
                            "after": h["after"],
                            "left": h["left_context"],
                            "right": h["right_context"],
                        },
                        indent=2,
                    ),
                    "```",
                    "",
                ]
    lines += ["## Decision call-site facts", ""]
    for t in row["timeline"]:
        lines += [f"### Turn {t['turn_idx']+1}", "", "```json", json.dumps(t["decision_facts"], indent=2), "```", ""]
    lines += ["## Source changes and original errors", ""]
    for tr in result["transitions"]:
        lines += [
            f"### Turn {tr['from_turn']+1} → {tr['to_turn']+1}",
            "",
            f"Sections: `{tr['sections']}`",
            "",
            f"Rename candidates: `{tr['rename_candidates']}`",
            "",
        ]
        for c in tr["changes"]:
            if c["status"] == "changed":
                lines += ["````diff", c["source_diff"], "````", ""]
        lines += [f"Unmatched: `{tr['unmatched_before']}` → `{tr['unmatched_after']}`", ""]
    for t in result["turns"]:
        error = t["observation"].get("environment_error_message")
        if error:
            lines += [
                f"### Turn {t['turn_idx']+1} error (first 2000 chars)",
                "",
                "````text",
                str(error)[:2000],
                "````",
                "",
            ]
    return "\n".join(lines)


def run(input_dir, output, protocol, selection_path=None):
    if not 0 <= protocol.gain_margin < 1 or not 0 < protocol.rewrite_similarity < 1:
        raise ValueError("invalid thresholds")
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    (output / "review").mkdir()
    source_manifest = input_dir / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    if not manifest.get("code_and_inputs_unchanged"):
        raise ValueError("input structure run was not stable")
    source_hash = sha(source_manifest)
    implementation = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
    # Freeze diagnostic rules before scanning outcomes. These defaults are not optimized on KernelBench.
    (output / "protocol.json").write_text(
        json.dumps(
            asdict(protocol)
            | {
                "utility": "positive raw speedup on explicitly correct, compiled, non-decoy, error-free results; NOT reconstructed training reward",
                "rewrite_outcome": "strictly later turn exceeding pre-rewrite historical best",
                "incomplete_window": "indeterminate, never an observed regression; timeout and worker-crash origin remain unattributed",
                "audit": "deterministic task-diverse stratified diagnostic sample, not prevalence/recall estimation",
                "data_role": "existing KernelBench historical pool for tooling diagnostics only; not the formal E1.1 natural pool",
            },
            indent=2,
        )
    )
    index_path = input_dir / "trajectory_index.json"
    index_hash = sha(index_path)
    index = json.loads(index_path.read_text())
    rows = []
    hashes = {}
    sensitivity = collections.Counter()
    for entry in index:
        path = input_dir / entry["path"]
        hashes[entry["path"]] = sha(path)
        result = json.loads(path.read_text())
        screened = screen_trajectory(result, protocol)
        tid = path.stem
        screened.update(trajectory_id=tid, source_path=str(path.resolve()))
        rows.append(screened)
        (output / "trajectories" / f"{tid}.json").write_text(json.dumps(screened, ensure_ascii=False, indent=2))
        for margin in (0.0, 0.05, 0.10):
            for threshold in (0.35, 0.50, 0.65):
                sr = screen_trajectory(result, Protocol(margin, threshold, protocol.review_per_model, protocol.seed))
                for flag, value in sr["flags"].items():
                    sensitivity[f"{margin}:{threshold}:{result['identity'][0]}:{flag}"] += value
        if len(rows) % 100 == 0:
            print(f"Screened {len(rows)} trajectories", flush=True)
    by_id = {r["trajectory_id"]: r for r in rows}
    selection_sha = sha(selection_path) if selection_path else None
    if selection_path:
        selected = json.loads(selection_path.read_text())
        if len({s["trajectory_id"] for s in selected}) != len(selected):
            raise ValueError("duplicate audit selection")
        for item in selected:
            row = by_id[item["trajectory_id"]]
            if item["identity"] != row["identity"]:
                raise ValueError("audit identity mismatch")
            item["initial_flags"] = item.get("initial_flags", item["flags"])
            item["flags"] = row["flags"]
    else:
        selected = review_selection(rows, protocol)
    for item in selected:
        row = by_id[item["trajectory_id"]]
        result = json.loads(Path(row["source_path"]).read_text())
        (output / "review" / f"{row['trajectory_id']}.md").write_text(review_card(row, result))
        source_dir = output / "review_sources" / row["trajectory_id"]
        source_dir.mkdir(parents=True)
        for turn in result["turns"]:
            for section, source in turn["sections"].items():
                suffix = {"CUDA_KERNELS": "cu", "APPLY_BINDINGS": "cpp", "MODEL_NEW": "py"}[section]
                (source_dir / f"T{turn['turn_idx']+1}.{suffix}").write_text(source["source"])
    counts = {}
    for model in sorted({r["identity"][0] for r in rows}):
        subset = [r for r in rows if r["identity"][0] == model]
        counts[model] = {
            "trajectories": len(subset),
            "any_candidate": sum(r["any_candidate"] for r in subset),
            "flags": {k: sum(r["flags"][k] for r in subset) for k in subset[0]["flags"]},
            "with_indeterminate_windows": sum(bool(r["uncertain_windows"]) for r in subset),
        }
    stable = (
        source_hash == sha(source_manifest)
        and index_hash == sha(index_path)
        and all(sha(input_dir / p) == h for p, h in hashes.items())
        and implementation == {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
        and (selection_path is None or selection_sha == sha(selection_path))
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "models": counts,
                "sensitivity": dict(sensitivity),
                "audit_selected": len(selected),
                "interpretation": "diagnostic counts only; no causal or natural-prevalence claim",
            },
            indent=2,
        )
    )
    (output / "review_selection.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    (output / "candidate_index.json").write_text(
        json.dumps(
            [{k: r[k] for k in ["trajectory_id", "identity", "flags", "any_candidate", "source_path"]} for r in rows],
            indent=2,
        )
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir.resolve()),
                "input_manifest_sha256": source_hash,
                "input_index_sha256": index_hash,
                "input_files": hashes,
                "implementation": implementation,
                "code_and_inputs_unchanged": stable,
                "protocol": asdict(protocol),
                "review_selection_source": str(selection_path.resolve()) if selection_path else None,
                "review_selection_source_sha256": selection_sha,
            },
            indent=2,
        )
    )
    if not stable:
        raise RuntimeError("code or input changed; output invalid")
    print(json.dumps({"models": counts, "audit_selected": len(selected)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gain-margin", type=float, default=0.05)
    p.add_argument("--rewrite-similarity", type=float, default=0.50)
    p.add_argument(
        "--review-selection", type=Path, help="Keep the preselected audit cohort fixed after implementation fixes"
    )
    a = p.parse_args()
    run(a.input_dir, a.output_dir, Protocol(a.gain_margin, a.rewrite_similarity), a.review_selection)


if __name__ == "__main__":
    main()
