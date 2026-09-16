"""Frozen-rule validation on exported training rollouts, no generated execution."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import time
from pathlib import Path

from .analyze import normalize
from .implementation_match import Regions, compare
from .optimization_pilot import digest, extract_instances, load_timing
from .optimization_strategies import BASELINE_RULES
from .structure import analyze_response


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_keys(source):
    raw = hashlib.sha256(source.encode()).hexdigest()
    try:
        canonical = ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError:
        canonical = source
    return {"raw:" + raw, "ast:" + hashlib.sha256(canonical.encode()).hexdigest()}


def collect_exclusions(legacy_manifest, previous_pool):
    manifest = json.loads(legacy_manifest.read_text())
    excluded = set()
    files = []
    for item in manifest["inputs"]:
        for name in item["files"]:
            p = Path(item["path"]) / name
            files.append(p)
    files.extend(sorted(previous_pool.glob("trajectory_*.json")))
    provenance = {}
    for p in files:
        provenance[str(p)] = sha(p)
        rows = json.loads(p.read_text())
        for row in rows:
            ref = (row.get("label") or {}).get("ground_truth")
            if ref:
                excluded.update(reference_keys(ref))
                break
    return excluded, provenance


def analyze_records(records, identity):
    timings = []
    ignored = collections.Counter()
    stages_unknown = collections.Counter()
    for raw in sorted(records, key=lambda r: r["turn_idx"]):
        if raw.get("is_pad_turn"):
            ignored["pad"] += 1
            continue
        observation = normalize(raw)
        minimal = {"observation": observation, "complete_sections": bool(raw.get("response"))}
        timing, reason = load_timing(minimal, raw)
        if timing is None:
            ignored[reason] += 1
            continue
        snapshot = analyze_response(raw["response"])
        snapshot.update(turn_idx=raw["turn_idx"], observation=observation)
        old_instances = [
            u for u in extract_instances(snapshot, timing) if u["feature"] in BASELINE_RULES - {"reduction_epilogue"}
        ]
        context = digest(
            sorted(
                c["syntax_hash"]
                for c in snapshot["components"]
                if c["section"] == "CUDA_KERNELS" and c["kind"] in {"macro", "file_context"}
            )
        )
        kernels = []
        for c in snapshot["components"]:
            units = [u for u in old_instances if u["kernel"] == c["qualified_name"] and u["line"] == c["line"]]
            if not units:
                continue
            parsed = Regions(c, context).result()
            stages_unknown.update(parsed["unknowns"])
            kernels.append(
                {
                    "name": c["qualified_name"],
                    "line": c["line"],
                    "source": c["source"],
                    "features": sorted({u["feature"] for u in units}),
                    "old_instance_hashes": [u["instance_hash"] for u in units],
                    "structure": parsed,
                    "profile_names": units[0]["profile_names"],
                }
            )
        timings.append(
            {
                "turn": raw["turn_idx"] + 1,
                "timing": timing,
                "kernels": kernels,
                "response_sha256": observation["response_sha256"],
                "removed_in_original_training": bool(raw.get("remove_sample")),
                "source_sections": {k: v["source"] for k, v in snapshot["sections"].items()},
            }
        )
    best = max(timings, key=lambda t: (t["timing"]["speedup"], -t["turn"])) if timings else None
    matches = []
    if best:
        for earlier in timings:
            if earlier["turn"] >= best["turn"]:
                continue
            proposals = []
            for target in best["kernels"]:
                old_key_set = set(target["old_instance_hashes"])
                for source in earlier["kernels"]:
                    if not set(source["features"]).intersection(target["features"]):
                        continue
                    old = bool(old_key_set.intersection(source["old_instance_hashes"]))
                    result = compare(source["structure"], target["structure"])
                    if old or result["kind"] != "no_match":
                        proposals.append(
                            {
                                "from": earlier["turn"],
                                "to": best["turn"],
                                "source_kernel": source["name"],
                                "target_kernel": target["name"],
                                "old_exact": old,
                                "new_kind": result["kind"],
                                "regions": result["matched_regions"],
                                "speedup_before": earlier["timing"]["speedup"],
                                "speedup_best": best["timing"]["speedup"],
                            }
                        )
            # Resolve old/new ambiguity independently, so new proposals cannot
            # silently change the reported old-method baseline.
            accepted = {}
            for mode in ("old", "new"):
                subset = (
                    [p for p in proposals if p["old_exact"]]
                    if mode == "old"
                    else [p for p in proposals if p["new_kind"] != "no_match"]
                )
                sources = collections.Counter(p["source_kernel"] for p in subset)
                targets = collections.Counter(p["target_kernel"] for p in subset)
                for proposal in subset:
                    if sources[proposal["source_kernel"]] != 1 or targets[proposal["target_kernel"]] != 1:
                        continue
                    key = (proposal["source_kernel"], proposal["target_kernel"])
                    record = accepted.setdefault(
                        key, proposal | {"old_exact": False, "new_kind": "no_match", "regions": []}
                    )
                    if mode == "old":
                        record["old_exact"] = True
                    else:
                        record["new_kind"] = proposal["new_kind"]
                        record["regions"] = proposal["regions"]
            matches.extend(accepted.values())
    return {
        "identity": identity,
        "turns": timings,
        "raw_turns": len(records),
        "ignored": dict(ignored),
        "unknowns": dict(stages_unknown),
        "best_turn": best["turn"] if best else None,
        "matches": matches,
    }


def summarize(results):
    ts = [t for r in results for t in r["turns"]]
    cases = [r for r in results if any(not m["old_exact"] and m["new_kind"] != "no_match" for m in r["matches"])]
    count = {
        "tasks": len({r["identity"][1] for r in results}),
        "trajectories": len(results),
        "raw_turns": sum(r["raw_turns"] for r in results),
        "correct_timed_turns": len(ts),
        "correct_timed_trajectories": sum(bool(r["turns"]) for r in results),
        "at_least_two_correct": sum(len(r["turns"]) >= 2 for r in results),
        "four_rule_turns": sum(bool(t["kernels"]) for t in ts),
        "four_rule_trajectories": sum(any(t["kernels"] for t in r["turns"]) for r in results),
        "new_parseable_turns": sum(any(k["structure"].get("whole_normalized") for k in t["kernels"]) for t in ts),
        "old_retention_trajectories": sum(any(m["old_exact"] for m in r["matches"]) for r in results),
        "new_only_retention_trajectories": len(cases),
        "union_retention_trajectories": sum(bool(r["matches"]) for r in results),
        "normalized_whole_trajectories": sum(
            any(not m["old_exact"] and m["new_kind"] == "normalized_whole" for m in r["matches"]) for r in results
        ),
        "partial_region_trajectories": sum(
            any(not m["old_exact"] and m["new_kind"] == "partial_region" for m in r["matches"]) for r in results
        ),
        "unresolved_reasons": dict(sum((collections.Counter(r["unknowns"]) for r in results), collections.Counter())),
    }
    return count


def run(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "trajectories").mkdir()
    frozen = json.loads(args.freeze.read_text())
    for path, expected in frozen["code_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("Code changed since freeze: " + path)
    excluded, previous_files = collect_exclusions(args.legacy_manifest, args.previous_pool)
    selection = []
    input_hashes = {}
    # Select all non-overlapping tasks before analyzing any candidate code.
    for p in sorted(args.input.glob("trajectory_*.json")):
        input_hashes[str(p)] = sha(p)
        records = json.loads(p.read_text())
        reference = records[0]["label"]["ground_truth"]
        keys = reference_keys(reference)
        selection.append(
            {
                "file": str(p),
                "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                "excluded_seen_reference": bool(keys.intersection(excluded)),
            }
        )
    (args.output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2))
    results = []
    for selected in selection:
        if selected["excluded_seen_reference"]:
            continue
        p = Path(selected["file"])
        result = analyze_records(json.loads(p.read_text()), ["train_rollout121", selected["reference_sha256"], p.stem])
        (args.output / "trajectories" / p.name).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        results.append(result)
        if len(results) % 32 == 0:
            print("processed", len(results), "trajectories", flush=True)
    summary = summarize(results)
    summary["excluded_trajectories"] = sum(s["excluded_seen_reference"] for s in selection)
    summary["elapsed_seconds"] = time.monotonic() - started
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    review = [
        {"identity": r["identity"], "matches": r["matches"]}
        for r in results
        if any(not m["old_exact"] and m["new_kind"] != "no_match" for m in r["matches"])
    ]
    (args.output / "new_matches.json").write_text(json.dumps(review, ensure_ascii=False, indent=2))
    unchanged = all(sha(p) == h for p, h in (input_hashes | previous_files | frozen["code_sha256"]).items())
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "freeze": frozen,
                "input_hashes": input_hashes,
                "excluded_reference_keys": sorted(excluded),
                "previous_pool_hashes": previous_files,
                "code_and_inputs_unchanged": unchanged,
                "selection": "all raw/AST-reference-disjoint tasks; no outcome sampling",
                "scope": "Single previously collected training-run rollout; not fresh model generation, family-disjoint validation or trained reward",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not unchanged:
        raise RuntimeError("Input or implementation changed during validation")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--previous-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
