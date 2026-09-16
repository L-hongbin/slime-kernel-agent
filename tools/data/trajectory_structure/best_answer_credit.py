"""Offline best-answer membership and conservative syntactic token provenance.

This implements the explicit heuristic that retained parts deserve credit. It is
not causal attribution and does not modify training rewards. Existing raw speedup
is only a diagnostic winner selector; formal training must supply its real q(K).
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import math
import time
from pathlib import Path

from tools.data.trajectory_candidates.screen import observation_state, structure_observed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_span(sequence, block):
    if not block:
        return False
    matches = 0
    for i, value in enumerate(sequence):
        if value == block[0] and sequence[i : i + len(block)] == block:
            matches += 1
            if matches > 1:
                return False
    return matches == 1


def trace_tokens(tokens, history, turn, *, gap, min_block=8, token_limit=12000):
    """First observed content origins, including exact and partial historical reverts.

    Whole identical versions retain provenance. Partial blocks must be sufficiently
    long and unique in both components. Short/repeated old text remains unknown;
    it is never confidently awarded to the latest author just for being unmatched.
    """
    for old in history:
        if tokens == old["tokens"]:
            return list(old["owners"]), {"method": "exact_historical_version", "matched_turn": old["turn"]}
    if not history:
        return [None if gap else turn] * len(tokens), {"method": "first_observed_component", "gap": gap}
    if len(tokens) > token_limit or any(len(h["tokens"]) > token_limit for h in history):
        return [None] * len(tokens), {"method": "token_limit"}
    matches = collections.defaultdict(set)
    uncertain = set()
    old_vocabulary = set()
    for old in history:
        before = old["tokens"]
        old_vocabulary.update(before)
        for block in difflib.SequenceMatcher(a=before, b=tokens, autojunk=False).get_matching_blocks():
            if block.size == 0:
                continue
            snippet = tokens[block.b : block.b + block.size]
            accepted = block.size >= min_block and unique_span(before, snippet) and unique_span(tokens, snippet)
            for offset in range(block.size):
                j = block.b + offset
                owner = old["owners"][block.a + offset]
                if accepted and owner is not None:
                    matches[j].add(owner)
                else:
                    uncertain.add(j)
    owners = []
    for i, token in enumerate(tokens):
        if matches[i]:
            owners.append(min(matches[i]))
        elif not gap and i not in uncertain and token not in old_vocabulary:
            owners.append(turn)
        else:
            owners.append(None)
    return owners, {
        "method": "unique_historical_blocks",
        "min_block": min_block,
        "gap": gap,
        "new_token_policy": "new_vocabulary_only",
    }


def select_winner(turns, *, scores=None):
    """Earliest exact maximum among consistent correct evaluations; no tolerance tie."""
    candidates = []
    for turn in turns:
        # Winner selection concerns the recorded evaluation, not our extraction
        # coverage. Missing best source must yield unknown, not a worse winner.
        state, speed = observation_state({**turn, "complete_sections": True})
        if state != "correct_timed":
            continue
        value = speed if scores is None else scores.get(turn["turn_idx"])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        candidates.append((float(value), -turn["turn_idx"], turn))
    return max(candidates, key=lambda row: row[:2])[2] if candidates else None


def parsed_tokens(turn):
    from .structure import analyze_native, analyze_python

    tokens = {}
    for section, info in turn["sections"].items():
        parsed = analyze_python(info["source"]) if section == "MODEL_NEW" else analyze_native(section, info["source"])
        for comp in parsed["components"]:
            tokens[(section, comp["id"])] = comp
    result = {}
    for comp in turn["components"]:
        reparsed = tokens.get((comp["section"], comp["id"]))
        if reparsed is None or reparsed["syntax_hash"] != comp["syntax_hash"]:
            raise ValueError(f"saved/parser component mismatch at T{turn['turn_idx']}: {comp['id']}")
        result[comp["id"]] = reparsed["_tokens"]
    return result


def component_observed(turn, component, coverage):
    if coverage == "program":
        return structure_observed(turn)
    section = turn["sections"].get(component["section"], {})
    return bool(
        component.get("valid_syntax")
        and section.get("source")
        and (component["kind"] != "file_context" or section.get("syntax_valid"))
    )


def summarize_membership(structure, *, scores=None, min_block=8, coverage="component"):
    turns = sorted(structure["turns"], key=lambda t: t["turn_idx"])
    indices = [t["turn_idx"] for t in turns]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate turn indices")
    best = select_winner(turns, scores=scores)
    result = {
        "identity": structure["identity"],
        "winner_turn": None if best is None else best["turn_idx"],
        "score_source": "diagnostic_raw_speedup" if scores is None else "provided_q",
        "intermediate_coverage": coverage,
        "components": [],
        "turn_membership": [],
        "semantics": "retained_content_heuristic_not_causal_credit",
    }
    if best is None:
        result["status"] = "no_valid_correct_winner"
        return result
    result["winner_score"] = best["observation"]["speedup"] if scores is None else scores[best["turn_idx"]]
    if not structure_observed(best):
        result["status"] = "winner_structure_unavailable"
        return result
    result["status"] = "analyzed"
    history = collections.defaultdict(list)
    section_gaps = collections.defaultdict(set)
    all_tokens = {}
    previous_idx = -1
    for turn in turns:
        idx = turn["turn_idx"]
        if idx > best["turn_idx"]:
            break
        sections = {"CUDA_KERNELS", "APPLY_BINDINGS", "MODEL_NEW"}
        if idx != previous_idx + 1:
            for section in sections:
                section_gaps[section].update(range(previous_idx + 1, idx))
        previous_idx = idx
        observed = [c for c in turn["components"] if component_observed(turn, c, coverage)]
        all_tokens[idx] = parsed_tokens(turn) if observed else {}
        for comp in observed:
            tokens = all_tokens[idx][comp["id"]]
            past = history[comp["lineage"]]
            gaps = section_gaps[comp["section"]]
            owners, method = trace_tokens(tokens, past, idx, gap=bool(gaps), min_block=min_block)
            past.append({"turn": idx, "tokens": tokens, "owners": owners, "syntax_hash": comp["syntax_hash"]})
            if idx == best["turn_idx"]:
                counts = collections.Counter(o for o in owners if o is not None)
                exact_first = next(h["turn"] for h in past if h["syntax_hash"] == comp["syntax_hash"])
                seen_hashes = set()
                possible_origins = set(gaps)
                for h in past:
                    if h["syntax_hash"] not in seen_hashes:
                        possible_origins.add(h["turn"])
                        seen_hashes.add(h["syntax_hash"])
                result["components"].append(
                    {
                        k: comp[k]
                        for k in [
                            "id",
                            "lineage",
                            "section",
                            "qualified_name",
                            "kind",
                            "syntax_hash",
                            "line",
                            "end_line",
                        ]
                    }
                    | {
                        "token_count": len(tokens),
                        "origin_token_counts": dict(sorted(counts.items())),
                        "unknown_tokens": owners.count(None),
                        "exact_version_first_seen_turn": exact_first,
                        "provenance": method,
                        "possible_unknown_origin_turns": sorted(possible_origins) if None in owners else [],
                        "spans": compact_spans(tokens, owners),
                    }
                )
        # Incomplete ModelNew must not hide a valid CUDA function, but missing or
        # unparsable source in that SAME section remains an origin uncertainty.
        # Preserve node-local valid components even when another node is malformed.
        for section in sections:
            info = turn["sections"].get(section, {})
            observed_section = (
                structure_observed(turn)
                if coverage == "program"
                else bool(info.get("source") and info.get("syntax_valid"))
            )
            if not observed_section:
                section_gaps[section].add(idx)
    for turn in turns:
        idx = turn["turn_idx"]
        memberships = [c["id"] for c in result["components"] if c["origin_token_counts"].get(idx, 0) > 0]
        unknown_here = any(idx in c["possible_unknown_origin_turns"] for c in result["components"])
        # Presence gives positive evidence; absence is negative only under complete
        # provenance. A winner itself still earns its immediate score increment.
        mask = 1 if memberships else (None if unknown_here and idx <= best["turn_idx"] else 0)
        result["turn_membership"].append(
            {
                "turn_idx": idx,
                "retained_component_ids": memberships,
                "membership": mask,
                "is_winner": idx == best["turn_idx"],
                "structure_observed": structure_observed(turn),
            }
        )
    return result


def compact_spans(tokens, owners):
    spans = []
    for i, owner in enumerate(owners):
        if spans and spans[-1]["origin_turn"] == owner:
            spans[-1]["end"] = i + 1
        else:
            spans.append({"start": i, "end": i + 1, "origin_turn": owner})
    for span in spans:
        text = tokens[span["start"] : span["end"]]
        span["token_sha256"] = hashlib.sha256(json.dumps(text, ensure_ascii=False).encode()).hexdigest()
        span["preview"] = text[:12]
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-block", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--coverage", choices=["component", "program"], default="component")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output directory must be new")
    args.output.joinpath("trajectories").mkdir(parents=True)
    start = time.perf_counter()
    sources = sorted((args.structure_root / "trajectories").glob("*.json"))
    if args.limit is not None:
        sources = sources[: args.limit]
    source_hashes = {str(p): sha(p) for p in sources}
    implementation = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("structure.py"),
            Path(__file__).parents[1] / "trajectory_candidates/screen.py",
            Path(__file__).parents[3] / "examples/kernel_agent/utils.py",
        ]
    }
    summaries = collections.defaultdict(lambda: collections.Counter())
    index = []
    for i, path in enumerate(sources):
        structure = json.loads(path.read_text())
        result = summarize_membership(structure, min_block=args.min_block, coverage=args.coverage)
        (args.output / "trajectories" / path.name).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        summary = summaries[structure["identity"][0]]
        summary["trajectories"] += 1
        summary[result["status"]] += 1
        if result["status"] == "analyzed":
            summary["winner_t" + str(result["winner_turn"] + 1)] += 1
            components = result["components"]
            summary["components"] += len(components)
            summary["normalized_tokens"] += sum(c["token_count"] for c in components)
            summary["unknown_tokens"] += sum(c["unknown_tokens"] for c in components)
            summary["earlier_origin_retained"] += any(
                t["membership"] == 1 and t["turn_idx"] < result["winner_turn"] for t in result["turn_membership"]
            )
            summary["positive_member_turns"] += sum(t["membership"] == 1 for t in result["turn_membership"])
            summary["unknown_member_turns"] += sum(t["membership"] is None for t in result["turn_membership"])
            summary["post_winner_turns"] += sum(
                t["turn_idx"] > result["winner_turn"] for t in result["turn_membership"]
            )
            summary["winner_has_partial_component_origins"] += any(
                len(c["origin_token_counts"]) > 1 for c in components
            )
        index.append(
            {
                "trajectory_id": path.stem,
                "identity": structure["identity"],
                "status": result["status"],
                "winner_turn": result["winner_turn"],
            }
        )
        if (i + 1) % 50 == 0:
            print(json.dumps({"processed": i + 1, "elapsed_seconds": time.perf_counter() - start}), flush=True)
    manifest = {
        "schema_version": 1,
        "inputs": source_hashes,
        "implementation": implementation,
        "protocol": {
            "score_source": "diagnostic_raw_speedup",
            "min_block": args.min_block,
            "intermediate_coverage": args.coverage,
            "weighting": "none_report_membership_only",
            "ties": "earliest_exact_max",
        },
        "source_stable": all(sha(Path(p)) == h for p, h in source_hashes.items()),
        "code_stable": all(sha(Path(p)) == h for p, h in implementation.items()),
        "wall_seconds": time.perf_counter() - start,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "summary.json").write_text(json.dumps(dict(summaries), indent=2))
    (args.output / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(dict(summaries), indent=2))


if __name__ == "__main__":
    main()
