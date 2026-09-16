"""Single-stage source-implementation credit over the best observed answer."""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from .component_reward import METHOD, ComponentRewardContractError, _decoy, _finite, _metadata, _trainable
from .utils import extract_cuda_agent_kernel_code

BACKEND = "source-strategies/v1"
IDENTITY_SCHEMA = "source-component-request/v1"


def source_request_identity(reference_code, kernel_code, *, entry_point, precision, submitted=True):
    return {
        "schema": IDENTITY_SCHEMA,
        "task_sha256": hashlib.sha256(reference_code.encode()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(kernel_code.encode()).hexdigest(),
        "entry_point": entry_point,
        "precision": precision,
        "submitted": bool(submitted),
    }


def _correct(sample):
    meta = _metadata(sample)
    extra = meta.get("env_extra_info") or {}
    env = meta.get("env_result") or {}
    state = env.get("env_state") or env
    return extra.get("correctness", state.get("correctness")) is True


def _observed_speedup(sample):
    """Return a finite measured speedup; missing/invalid observations cannot qualify."""
    meta = _metadata(sample)
    extra = meta.get("env_extra_info") or {}
    env = meta.get("env_result") or {}
    state = env.get("env_state") or env
    value = extra.get("speedup", state.get("speedup"))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _source_observation(sample):
    # Imported only in this opt-in backend; a missing parser dependency must fail
    # visibly instead of silently turning an entire training run into fallback.
    from .source_components import SCHEMA, analyze_source_components

    meta = _metadata(sample)
    identity = meta.get("source_component_identity")
    if identity is None:
        return None, ["missing_source_request_identity"]
    if not isinstance(identity, dict) or identity.get("schema") != IDENTITY_SCHEMA:
        raise ComponentRewardContractError("unsupported source request identity")
    code = extract_cuda_agent_kernel_code(sample.response or "")
    if hashlib.sha256(code.encode()).hexdigest() != identity.get("candidate_source_sha256"):
        raise ComponentRewardContractError("source credit does not match the evaluated candidate")
    for key in ("task_sha256", "entry_point", "precision"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ComponentRewardContractError(f"source identity missing {key}")
    result = analyze_source_components(code, profiles=meta.get("source_component_profiles") or [])
    if not isinstance(result, dict) or result.get("schema") != SCHEMA or not isinstance(result.get("units"), list):
        raise ComponentRewardContractError("invalid source component observation")
    for unit in result["units"]:
        if (
            not isinstance(unit, dict)
            or not isinstance(unit.get("id"), str)
            or unit.get("kind") not in {"kernel", "library"}
            or not isinstance(unit.get("features"), list)
            or any(not isinstance(feature, str) for feature in unit["features"])
            or type(unit.get("use_observed")) is not bool
            or (unit.get("signature") is not None and not isinstance(unit["signature"], str))
        ):
            raise ComponentRewardContractError("invalid source component unit")
    ids = [unit["id"] for unit in result["units"]]
    if len(set(ids)) != len(ids):
        raise ComponentRewardContractError("duplicate source component IDs")
    return {"identity": identity, **result}, list(result.get("unknowns", []))


def attribute_best_source_components(samples, base_scores, *, min_speedup=None):
    """Conserve one positive quality budget; failures may be implementation origins."""
    if len(samples) != len(base_scores):
        raise ComponentRewardContractError("score/turn count mismatch")
    scores = [_finite(q, "task_reward") for q in base_scores]
    eligible = [_trainable(s) and not _decoy(s) for s in samples]
    candidates = [t for t, valid in enumerate(eligible) if valid]
    gate = {}
    if min_speedup is not None:
        min_speedup = _finite(min_speedup, "component minimum anchor speedup")
        if min_speedup < 0:
            raise ComponentRewardContractError("component minimum anchor speedup must be nonnegative")
        correct = [_correct(s) for s in samples]
        speedups = [_observed_speedup(s) for s in samples]
        candidates = [t for t in candidates if correct[t] and speedups[t] is not None and speedups[t] >= min_speedup]
        gate = dict(
            anchor_min_speedup=min_speedup,
            anchor_correct=correct,
            anchor_speedups=speedups,
            anchor_eligible_turns=candidates,
        )
    best = max(candidates, key=lambda t: (scores[t], -t)) if candidates else None
    budget = max(scores[best], 0.0) if best is not None else 0.0
    allocation = dict(
        method=METHOD,
        backend=BACKEND,
        best_turn=best,
        base_scores=scores,
        quality_budget=budget,
        credits=[0.0] * len(samples),
        units=[],
        residual_fraction=0.0,
        resolved_cross_turn_units=0,
        status="no_positive_budget",
        unknowns=[],
        future_fold_applied=False,
        source_analysis_wall_seconds=0.0,
        **gate,
    )
    if min_speedup is not None and best is None:
        allocation["status"] = "no_qualifying_anchor"
        return allocation
    if budget == 0:
        return allocation
    if not _correct(samples[best]):
        allocation.update(
            status="unavailable_best_turn_fallback", residual_fraction=1.0, unknowns=["no_correct_anchor"]
        )
        allocation["credits"][best] = budget
        return allocation
    observations, gaps = {}, {}
    for turn in range(best + 1):
        if not eligible[turn]:
            gaps[turn] = ["turn_ineligible_for_positive_credit"]
            continue
        obs, reasons = _source_observation(samples[turn])
        gaps[turn] = reasons
        if obs is not None:
            observations[turn] = obs
            allocation["source_analysis_wall_seconds"] += _finite(obs.get("wall_seconds", 0.0), "source analysis time")
    if best not in observations or not observations[best]["units"]:
        allocation.update(
            status="unavailable_best_turn_fallback",
            residual_fraction=1.0,
            unknowns=gaps.get(best, []) + ["no_best_source_units"],
        )
        allocation["credits"][best] = budget
        return allocation
    expected = observations[best]["identity"]
    for turn, obs in list(observations.items()):
        different = [
            key for key in ("task_sha256", "entry_point", "precision") if obs["identity"][key] != expected[key]
        ]
        if different:
            gaps[turn].extend(f"incomparable_source_context:{key}" for key in different)
            del observations[turn]
    units = observations[best]["units"]
    counts = {
        turn: Counter(u["signature"] for u in obs["units"] if u.get("signature")) for turn, obs in observations.items()
    }
    origin_counts = [0] * len(samples)
    residual = 0
    for unit in units:
        signature = unit.get("signature")
        item = dict(
            best_unit=unit["id"],
            kind=unit["kind"],
            features=sorted(unit.get("features", [])),
            weight=1.0 / len(units),
            origin_turn=None,
            status="unresolved",
            unknowns=list(unit.get("unknowns", [])),
            matches=[],
        )
        if not unit.get("use_observed"):
            item["unknowns"].append("best_source_use_unobserved")
        elif not unit.get("features"):
            item["unknowns"].append("no_recognized_strategy")
        elif signature and counts[best][signature] == 1:
            for turn, obs in sorted(observations.items()):
                matched = [u for u in obs["units"] if u.get("signature") == signature]
                if len(matched) == 1:
                    item["matches"].append({"turn": turn, "unit": matched[0]["id"]})
            if item["matches"]:
                origin = item["matches"][0]["turn"]
                observed_turns = {m["turn"] for m in item["matches"]}
                item.update(
                    origin_turn=origin,
                    status="continuous" if all(t in observed_turns for t in range(origin, best + 1)) else "reappeared",
                    observation_gaps={str(t): gaps.get(t, []) for t in range(origin, best) if gaps.get(t)},
                )
        else:
            item["unknowns"].append("unknown_or_ambiguous_source_implementation")
        if item["origin_turn"] is None:
            residual += 1
        else:
            origin_counts[item["origin_turn"]] += 1
        allocation["units"].append(item)
    origin_counts[best] += residual
    allocation["credits"] = [budget * count / len(units) for count in origin_counts]
    receiver = best if origin_counts[best] else max(t for t, count in enumerate(origin_counts) if count)
    allocation["credits"][receiver] += budget - math.fsum(allocation["credits"])
    allocation.update(
        residual_fraction=residual / len(units),
        status="partial" if residual else "attributed",
        resolved_cross_turn_units=sum(
            u["origin_turn"] is not None and u["origin_turn"] < best for u in allocation["units"]
        ),
        unknowns=sorted({reason for reasons in gaps.values() for reason in reasons}),
        source_unit_counts={str(t): len(o["units"]) for t, o in observations.items()},
    )
    return allocation
