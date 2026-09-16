"""Single-pass optimization-strategy coverage on saved source, including failed turns."""

from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
import time
from pathlib import Path

from .analyze import file_sha, input_rows, normalize
from .optimization_pilot import inspect_kernel, load_timing, profile_matches
from .optimization_strategies import BASELINE_RULES, STRATEGIES, inspect_program, inspect_transition
from .structure import analyze_response


def scan_snapshot(snapshot):
    observations = []
    parsed, rejected = 0, []
    for c in snapshot["components"]:
        if c["section"] != "CUDA_KERNELS" or not (
            c["kind"] == "kernel" or (c["kind"] == "function" and "__device__" in c["source"])
        ):
            continue
        features = inspect_kernel(c)
        if features is None:
            rejected.append(dict(component=c["qualified_name"], line=c["line"]))
            continue
        parsed += 1
        for label, evidence in features["features"].items():
            observations.extend(
                dict(feature=label, component=c["qualified_name"], section=c["section"], **e) for e in evidence
            )
    observations.extend(inspect_program(snapshot))
    return observations, parsed, rejected


def bundle(observations):
    grouped = {}
    for o in observations:
        key = (o["feature"], o["section"], o["component"])
        entry = grouped.setdefault(
            key, dict(feature=key[0], section=key[1], component=key[2], evidence_count=0, evidence=[])
        )
        entry["evidence_count"] += 1
        if len(entry["evidence"]) < 4:
            entry["evidence"].append({k: v for k, v in o.items() if k not in {"feature", "section", "component"}})
    return sorted(grouped.values(), key=lambda o: (o["feature"], o["section"], o["component"]))


def scan_records(records, identity):
    turns = []
    previous = None
    pads = 0
    for raw in sorted(records, key=lambda r: r.get("turn_idx", (r.get("metadata") or {}).get("turn_idx"))):
        if raw.get("is_pad_turn"):
            pads += 1
            previous = None
            continue
        observation = normalize(raw)
        snapshot = analyze_response(raw["response"])
        timing, state = load_timing(
            dict(observation=observation, complete_sections=snapshot["complete_sections"]), raw
        )
        observations, parsed, rejected = scan_snapshot(snapshot)
        transition = []
        # Only adjacent observed turns, never bridge gaps/padding as a source change.
        if previous is not None and previous[0] + 1 == observation["turn_idx"]:
            transition = inspect_transition(previous[1], snapshot)
        env = raw.get("env_result") or (raw.get("metadata") or {}).get("env_result") or {}
        profiles = env.get("env_state") or env
        profiles = (profiles.get("metadata") or {}).get("profiling", {}).get("kernels", [])
        for entry in observations:
            entry["profiling_observed"] = bool(profile_matches(entry["component"].split("::")[-1], profiles))
        turns.append(
            dict(
                turn=observation["turn_idx"] + 1,
                evaluation_state=state,
                timing=timing,
                correctness=observation["correctness"],
                compiled=observation["compiled"],
                response_sha256=observation["response_sha256"],
                source_selection=snapshot["selection_mode"],
                complete_sections=snapshot["complete_sections"],
                parsed_functions=parsed,
                rejected_functions=rejected,
                observations=bundle(observations),
                changes=bundle(transition),
                source_sections={name: section["source"] for name, section in snapshot["sections"].items()},
            )
        )
        previous = (observation["turn_idx"], snapshot)
    return dict(identity=identity, turns=turns, padding_rows=pads)


def summarize(results):
    turns = [(r, t) for r in results for t in r["turns"]]

    def labels(t):
        return {o["feature"] for o in t["observations"]}

    def count(predicate):
        selected = [(r, t) for r, t in turns if predicate(t)]
        return dict(
            turns=len(selected),
            trajectories=len({tuple(r["identity"]) for r, _ in selected}),
            tasks=len({(r["identity"][0], r["identity"][2]) for r, _ in selected}),
        )

    stats = {}
    for label, definition in STRATEGIES.items():
        field = "changes" if definition["scope"] == "transition" else "observations"

        def hit(t, label=label, field=field):
            return any(o["feature"] == label for o in t[field])

        stats[label] = count(hit) | dict(
            failed_turns=sum(hit(t) and t["evaluation_state"] == "observed_failure" for _, t in turns),
            other_non_timed_turns=sum(
                hit(t) and t["evaluation_state"] not in {"correct_timed", "observed_failure"} for _, t in turns
            ),
            evidence=definition["evidence"],
            scope=definition["scope"],
        )
    families = {
        f: count(lambda t, f=f: any(STRATEGIES[o["feature"]]["family"] == f for o in t["observations"] + t["changes"]))
        for f in sorted({d["family"] for d in STRATEGIES.values()})
    }
    return dict(
        trajectories=len(results),
        tasks=len({(r["identity"][0], r["identity"][2]) for r in results}),
        turns=len(turns),
        padding_rows=sum(r["padding_rows"] for r in results),
        evaluation_states=dict(collections.Counter(t["evaluation_state"] for _, t in turns)),
        any_source_strategy=count(lambda t: bool(t["observations"])),
        explicit_strategy=count(lambda t: any(STRATEGIES[label]["evidence"] == "explicit" for label in labels(t))),
        baseline_four_source=count(lambda t: bool(labels(t) & (BASELINE_RULES - {"reduction_epilogue"}))),
        new_source_only=count(
            lambda t: bool(labels(t)) and not bool(labels(t) & (BASELINE_RULES - {"reduction_epilogue"}))
        ),
        additional_trajectories_beyond_four=sum(
            any(t["observations"] for t in r["turns"])
            and not any(labels(t) & (BASELINE_RULES - {"reduction_epilogue"}) for t in r["turns"])
            for r in results
        ),
        failed_source_strategy=count(
            lambda t: t["evaluation_state"] == "observed_failure" and bool(t["observations"])
        ),
        correct_timed_source_strategy=count(
            lambda t: t["evaluation_state"] == "correct_timed" and bool(t["observations"])
        ),
        source_parseable=count(lambda t: bool(t["parsed_functions"])),
        transition_strategy=count(lambda t: bool(t["changes"])),
        registry_size=len(STRATEGIES),
        observed_strategies=sum(s["turns"] > 0 for s in stats.values()),
        per_strategy=stats,
        families=families,
    )


def run(inputs, output):
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    # Include transitive local implementation and actual response parser.
    code = [
        *Path(__file__).parent.glob("*.py"),
        Path("examples/kernel_agent/utils.py"),
        Path("tools/data/trajectory_candidates/screen.py"),
    ]
    hashes = {str(p.resolve()): file_sha(p) for p in code}
    results, seen = [], set()
    for item in inputs:
        name, path = item.split("=", 1)
        source = Path(path)
        paths = sorted([*source.glob("g*.json"), *source.glob("trajectory_*.json")]) if source.is_dir() else [source]
        if not paths:
            raise ValueError(f"No trajectory JSON files: {source}")
        for p in paths:
            hashes[str(p.resolve())] = file_sha(p)
            groups = collections.defaultdict(list)
            for raw in input_rows(p):
                obs = normalize(raw)
                key = (
                    name,
                    obs["dataset"],
                    obs["reference_sha256"] or f"{p.name}:{obs['problem_id']}",
                    obs["group_id"],
                )
                groups[key].append(raw)
            for identity, records in groups.items():
                if identity in seen:
                    raise ValueError(f"Duplicate trajectory identity: {identity}")
                seen.add(identity)
                result = scan_records(records, identity)
                artifact = f"trajectory_{len(results):04}.json"
                (output / "trajectories" / artifact).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                result["artifact"] = artifact
                results.append(result)
            if len(results) % 32 == 0:
                print(f"scanned {len(results)} trajectories", flush=True)
    summary = summarize(results)
    summary["by_input"] = {
        name: summarize([r for r in results if r["identity"][0] == name])
        for name in sorted({r["identity"][0] for r in results})
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output / "catalog.json").write_text(json.dumps(STRATEGIES, ensure_ascii=False, indent=2))
    examples = {}
    for label in STRATEGIES:
        options = []
        for r in results:
            for t in r["turns"]:
                for o in t["observations"] + t["changes"]:
                    if o["feature"] == label:
                        options.append(
                            dict(
                                artifact=r["artifact"],
                                identity=r["identity"],
                                turn=t["turn"],
                                state=t["evaluation_state"],
                                observation=o,
                            )
                        )
                        break
        # Retain both failed and successful examples where available.
        examples[label] = {
            state: next((e for e in options if e["state"] == state), None)
            for state in ("correct_timed", "observed_failure")
        }
        examples[label]["first_observed"] = options[0] if options else None
    (output / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    (output / "trajectory_index.json").write_text(
        json.dumps(
            [dict(identity=r["identity"], artifact=r["artifact"]) for r in results], ensure_ascii=False, indent=2
        )
    )
    changed = [p for p, h in hashes.items() if file_sha(Path(p)) != h]
    manifest = dict(
        inputs=inputs,
        code_and_input_hashes=hashes,
        code_and_inputs_unchanged=not changed,
        changed_paths=changed,
        parser_versions={
            n: importlib.metadata.version(n) for n in ("tree-sitter", "tree-sitter-cuda", "tree-sitter-cpp")
        },
        protocol="All nonpadding saved source; no Correct/timing/profiling gate; single-pass rules; adjacent source-site changes",
        evidence_scope="Source observations, not execution coverage, precision/recall, causal speedup or reward",
        elapsed_seconds=time.monotonic() - start,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if changed:
        raise RuntimeError(f"Inputs/code changed: {changed}")
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"by_input", "per_strategy", "families"}}, indent=2)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", action="append", required=True, help="name=directory of trajectory JSON, JSONL or trusted local .pt"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.input, a.output)


if __name__ == "__main__":
    main()
