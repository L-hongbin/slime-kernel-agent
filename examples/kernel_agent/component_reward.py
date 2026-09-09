"""Same-trajectory, best-answer component reward.

This is a retention heuristic over self-contained runtime facts, not a causal
claim or state merge. KernelGym produces observations; this module independently
selects units, matches their interfaces, and distributes one quality budget.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)
METHOD = "best-observed-components/v1"
WIRE_SCHEMA = "kernelgym-runtime-graph/v1"
GRAPH_SCHEMA = "coarse-component-graph/v1"
MATCH_CONTEXT = ("task_sha256", "input_signature", "environment_signature", "collector_sha256")
PROVEN = {"proven_region_dependency", "initial_value_read"}
SEMANTIC_ONLY = {"kernel_body_opaque", "opaque_kernel_semantics"}
UNIT_KINDS = {"kernel", "library", "copy", "fill", "clone", "memcpy", "memset"}


class ComponentRewardContractError(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ComponentRewardContractError(f"{field} must be a finite number")
    return float(value)


def _metadata(sample):
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _decoy(sample):
    meta = _metadata(sample)
    extra = meta.get("env_extra_info", {})
    env = meta.get("env_result", {}).get("env_state", {})
    return bool(extra.get("decoy_kernel") or env.get("decoy_kernel"))


def _trainable(sample):
    return (
        not sample.remove_sample
        and not _metadata(sample).get("is_pad_turn", False)
        and sample.status != sample.Status.ABORTED
        and int(sample.response_length or 0) > 0
        and (sample.loss_mask is None or any(sample.loss_mask))
    )


def _graph_payload(sample):
    meta = _metadata(sample)
    payload = meta.get("runtime_graph")
    if payload is None:
        return None, ["missing_runtime_graph"]
    if not isinstance(payload, dict) or payload.get("schema") != WIRE_SCHEMA:
        raise ComponentRewardContractError("unsupported runtime_graph schema")
    status = payload.get("status")
    if status not in {"ok", "partial", "unavailable", "invalid"}:
        raise ComponentRewardContractError("invalid runtime_graph status")
    if status == "invalid":
        raise ComponentRewardContractError(f"invalid runtime_graph: {payload.get('unknowns', [])}")
    identity = payload.get("identity")
    expected = meta.get("runtime_graph_expected_identity", {})
    if isinstance(identity, dict):
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ComponentRewardContractError(f"runtime_graph transport identity mismatch: {key}")
    if status == "unavailable":
        return None, list(payload.get("unknowns", [])) or ["runtime_graph_unavailable"]
    if not isinstance(identity, dict):
        raise ComponentRewardContractError("runtime_graph identity is missing")
    for key in (*MATCH_CONTEXT, "candidate_source_sha256", "state_signature"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ComponentRewardContractError(f"runtime_graph identity missing {key}")
    graph = payload.get("graph")
    if not isinstance(graph, dict) or graph.get("schema") != GRAPH_SCHEMA:
        raise ComponentRewardContractError("unsupported component graph schema")
    for key in ("nodes", "edges", "buffers", "outputs"):
        if not isinstance(graph.get(key), list):
            raise ComponentRewardContractError(f"component graph missing list {key}")
    # These are same-turn execution alignment attestations, separate from
    # cross-turn membership. A partial graph cannot override a failed replay.
    alignment = payload.get("alignment", {})
    gaps = [
        f"unverified_alignment:{key}" for key in ("scored_control", "control_trace") if alignment.get(key) is not True
    ]
    if gaps:
        return None, list(payload.get("unknowns", [])) + gaps
    return {"identity": identity, "graph": graph, "status": status}, list(payload.get("unknowns", []))


def _units(observation, *, output_only=True):
    """Recover output closure and unit signatures without a producer matcher."""
    graph = observation["graph"]
    nodes = {}
    for node in graph["nodes"]:
        if not isinstance(node.get("id"), str) or node["id"] in nodes:
            raise ComponentRewardContractError("component node IDs must be unique strings")
        nodes[node["id"]] = node
    coverage = graph.get("coverage", {})
    # Capture completion does not prove that the transmitted summary retained
    # every call. A clipped denominator must never redistribute the full budget.
    if coverage.get("summary_complete") is not True:
        return [], ["incomplete_graph_summary"]
    if coverage.get("trace_process_complete") is not True:
        return [], ["incomplete_launch_enumeration"]
    if any(
        type(coverage.get(key)) is not int or coverage[key] < 0
        for key in ("kernel_launches", "completed_kernel_launches")
    ):
        return [], ["unknown_launch_completion_counts"]
    if coverage.get("kernel_launches") != coverage.get("completed_kernel_launches"):
        return [], ["incomplete_launch_completion"]

    def parent(key):
        node = nodes[key]
        owner = node.get("parent_library_id")
        if owner is not None:
            if owner not in nodes or nodes[owner]["kind"] != "library":
                raise ComponentRewardContractError("invalid library parent")
            return owner
        return key

    upstream = defaultdict(set)
    uncertain = set()
    for edge in graph["edges"]:
        source, target = edge.get("source"), edge.get("target")
        if target not in nodes:
            raise ComponentRewardContractError("dangling dependency target")
        if source not in nodes:
            if isinstance(source, str) and source.startswith("initial:"):
                continue
            raise ComponentRewardContractError("dangling dependency source")
        if not edge.get("regions") or not edge.get("buffer"):
            continue  # Ordering/containment alone is not a data path.
        source, target = parent(source), parent(target)
        if source == target:
            continue
        if edge.get("certainty") not in PROVEN:
            uncertain.add(target)
        else:
            upstream[target].add(source)
    reached = set()
    closure_unknown = bool(graph.get("output_unknowns")) if output_only else False
    pending = []
    for edge in graph["outputs"] if output_only else []:
        source = edge.get("source")
        if source not in nodes or not edge.get("regions") or edge.get("certainty") not in PROVEN:
            closure_unknown = True
            continue
        pending.append(parent(source))
    if not output_only:
        pending = [parent(key) for key in nodes]
    if not pending and not closure_unknown:
        return [], ["no_observed_output_units"]
    while pending:
        key = pending.pop()
        if key in reached:
            continue
        reached.add(key)
        if output_only and key in uncertain:
            closure_unknown = True
        # Only proven paths certify membership; uncertain paths retain budget.
        pending.extend(upstream[key])

    buffers = {}
    for buffer in graph["buffers"]:
        if not isinstance(buffer.get("id"), str) or buffer["id"] in buffers:
            raise ComponentRewardContractError("component buffer IDs must be unique strings")
        buffers[buffer["id"]] = buffer
    units = []
    candidates = {parent(key) for key in nodes} if closure_unknown else reached
    for key in sorted(candidates):
        node = nodes[key]
        if node["kind"] not in UNIT_KINDS or parent(key) != key:
            continue
        reads, writes = node.get("reads", []), node.get("writes", [])
        if not reads and not writes and node.get("footprint_complete") is True:
            continue
        gaps = _node_gaps(node)
        if key not in reached:
            gaps.append("output_membership_unresolved")
        units.append({"id": key, "signature": None if gaps else _signature(node, buffers), "unknowns": gaps})
    return units, [] if units else ["no_eligible_output_units"]


def _node_gaps(node):
    gaps = set(node.get("unknowns", [])) | set(node.get("configuration_unknowns", []))
    gaps -= SEMANTIC_ONLY
    if node.get("footprint_complete") is not True:
        gaps.add("incomplete_footprint")
    if not node.get("implementation") or set(str(node["implementation"])) == {"0"}:
        gaps.add("implementation_unavailable")
    if not isinstance(node.get("configuration"), dict) or not node["configuration"]:
        gaps.add("configuration_unavailable")
    return sorted(gaps)


def _signature(node, buffers):
    if _node_gaps(node):
        return None
    aliases = {}

    def binding(key):
        if key not in buffers:
            raise ComponentRewardContractError("unknown buffer in component interface")
        roles = tuple(sorted(buffers[key].get("roles", [])))
        if roles:
            return ["roles", list(roles)]
        if key not in aliases:
            aliases[key] = len(aliases)
        return ["local_buffer", aliases[key]]

    def normalize(value):
        if isinstance(value, dict):
            return {k: binding(v) if k in ("buffer", "buffer_id") else normalize(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value

    # ABI order gives unnamed temporary buffers stable interface-local aliases;
    # repeated aliases remain shared. It never creates read/write evidence.
    config = normalize(node["configuration"])
    effects = []
    for mode, items in (("read", node.get("reads", [])), ("write", node.get("writes", []))):
        for effect in items:
            key = effect.get("buffer")
            role = binding(key)
            regions = effect.get("regions", [])
            if not isinstance(regions, list) or any(
                not isinstance(r, list)
                or len(r) != 2
                or type(r[0]) is not int
                or type(r[1]) is not int
                or r[0] < 0
                or r[1] <= r[0]
                for r in regions
            ):
                raise ComponentRewardContractError("invalid component byte ranges")
            descriptor = {
                "bytes": buffers[key].get("bytes"),
                "views": sorted(buffers[key].get("views", []), key=_canonical),
                "role_views": buffers[key].get("role_views", {}),
            }
            effects.append([mode, role, sorted(regions), effect.get("port"), descriptor])
    return _canonical([node["kind"], node["implementation"], config, sorted(effects, key=_canonical)])


def attribute_best_components(samples, base_scores):
    """Distribute max positive quality once; returns an auditable allocation."""
    if len(samples) != len(base_scores):
        raise ComponentRewardContractError("score/turn count mismatch")
    scores = [_finite(q, "task_reward") for q in base_scores]
    eligible = [_trainable(s) and not _decoy(s) for s in samples]
    candidates = [t for t, valid in enumerate(eligible) if valid]
    best = max(candidates, key=lambda t: (scores[t], -t)) if candidates else None
    budget = max(scores[best], 0.0) if best is not None else 0.0
    result = {
        "method": METHOD,
        "best_turn": best,
        "base_scores": scores,
        "quality_budget": budget,
        "credits": [0.0] * len(samples),
        "units": [],
        "residual_fraction": 0.0,
        "status": "no_positive_budget",
        "unknowns": [],
        "future_fold_applied": False,
    }
    observations, gaps = {}, {}
    for turn in range(len(samples)):
        if not eligible[turn]:
            gaps[turn] = ["turn_ineligible_for_positive_credit"]
            continue
        obs, reasons = _graph_payload(samples[turn])
        gaps[turn] = reasons
        if obs is not None:
            observations[turn] = obs
    if budget == 0:
        result["unknowns"] = sorted({reason for reasons in gaps.values() for reason in reasons})
        return result
    observations = {turn: obs for turn, obs in observations.items() if turn <= best}
    if best not in observations:
        result.update(status="unavailable_best_turn_fallback", residual_fraction=1.0, unknowns=gaps[best])
        result["credits"][best] = budget
        return result
    identity = observations[best]["identity"]
    # State changes are same-turn audit facts, not a global membership veto.
    # Each unit still needs matching implementation, config and buffer interface.
    result["cross_turn_state_changes"] = [
        turn for turn, obs in observations.items() if obs["identity"]["state_signature"] != identity["state_signature"]
    ]
    for turn, obs in list(observations.items()):
        mismatches = [key for key in MATCH_CONTEXT if obs["identity"][key] != identity[key]]
        if mismatches:
            # Valid requests may change inputs or collection context.
            # This is incomparability, unlike a graph bound to the wrong request.
            gaps[turn].extend(f"incomparable_context:{key}" for key in mismatches)
            del observations[turn]
    units, closure_gaps = _units(observations[best])
    if closure_gaps:
        result.update(status="unavailable_best_turn_fallback", residual_fraction=1.0, unknowns=closure_gaps)
        result["credits"][best] = budget
        return result
    by_turn = {}
    for turn, obs in observations.items():
        prior, reasons = _units(obs, output_only=(turn == best))
        gaps[turn] += reasons
        by_turn[turn] = prior
    # Match any earlier OBSERVED unit, including A->B->A reappearance. Repeated
    # identical interfaces are ambiguous, not arbitrarily assigned node IDs.
    best_counts = Counter(unit["signature"] for unit in units if unit["signature"] is not None)
    residual_units = 0
    origin_counts = [0] * len(samples)
    for unit in units:
        signature = unit["signature"]
        item = {
            "best_unit": unit["id"],
            "weight": 1.0 / len(units),
            "origin_turn": None,
            "status": "unresolved",
            "unknowns": list(unit["unknowns"]),
            "matches": [],
        }
        if signature is not None:
            item["signature_sha256"] = hashlib.sha256(signature.encode()).hexdigest()
        if signature is None or best_counts[signature] != 1:
            item["unknowns"].append("unresolved_or_ambiguous_best_unit")
        else:
            ambiguous = False
            for turn in range(best + 1):
                matches = [u for u in by_turn.get(turn, []) if u["signature"] == signature]
                if len(matches) > 1:
                    item["unknowns"].append(f"ambiguous_origin_turn:{turn}")
                    if not item["matches"]:
                        ambiguous = True
                    continue
                if len(matches) == 1:
                    item["matches"].append({"turn": turn, "unit": matches[0]["id"]})
            if item["matches"] and not ambiguous:
                origin = item["matches"][0]["turn"]
                if origin == best and any(
                    gaps.get(t) or t not in by_turn or any(u["signature"] is None for u in by_turn[t])
                    for t in range(best)
                ):
                    item["unknowns"].append("earlier_observation_unresolved")
                else:
                    observed = {x["turn"] for x in item["matches"]}
                    missing = [t for t in range(origin, best + 1) if t not in observed]
                    item.update(origin_turn=origin, status="reappeared" if missing else "continuous")
                    item["observation_gaps"] = {str(t): gaps.get(t, []) for t in range(best) if gaps.get(t)}
        if item["origin_turn"] is None:
            residual_units += 1
        else:
            origin_counts[item["origin_turn"]] += 1
        result["units"].append(item)
    result["residual_fraction"] = residual_units / len(units)
    origin_counts[best] += residual_units
    result["credits"] = [budget * count / len(units) for count in origin_counts]
    # Unresolved mass goes to BEST. Correct roundoff on a receiving turn only,
    # so a zero-credit best turn never acquires a tiny negative allocation.
    receiver = best if origin_counts[best] else max(t for t, count in enumerate(origin_counts) if count)
    result["credits"][receiver] += budget - math.fsum(result["credits"])
    result["status"] = "partial" if residual_units else "attributed"
    result["resolved_cross_turn_units"] = sum(
        u["origin_turn"] is not None and u["origin_turn"] < best for u in result["units"]
    )
    result["unknowns"] = sorted({reason for reasons in gaps.values() for reason in reasons})
    return result


def postprocess_component_turns(args, samples, finish_reason):
    """Opt-in finalization; baseline postprocess never calls this when off."""
    from .utils import (
        _apply_coverage_rs,
        _apply_overlong_penalty,
        _apply_rollout_progress_metadata,
        _mark_remove_sample,
    )

    if any("component_reward" in _metadata(s) for s in samples):
        raise ComponentRewardContractError("component turn samples already finalized")
    if [int(_metadata(s).get("turn_idx", -1)) for s in samples] != list(range(len(samples))):
        raise ComponentRewardContractError("component reward requires ordered unique turn indices")
    base = [_finite(s.get_reward_value(args), "task_reward") for s in samples]
    _apply_rollout_progress_metadata(samples, finish_reason)
    for sample in samples:
        if not _trainable(sample) or finish_reason == "model_abort":
            _mark_remove_sample(
                sample,
                (
                    "model_abort"
                    if finish_reason == "model_abort"
                    else _metadata(sample).get("remove_reason", "component_hard_removed")
                ),
            )
    _apply_coverage_rs(args, samples)
    _apply_overlong_penalty(args, samples)
    shaped = [float(s.reward) for s in samples]
    penalties = [max(0.0, q - shaped[t]) if _trainable(samples[t]) else 0.0 for t, q in enumerate(base)]
    allocation = attribute_best_components(samples, base)
    targets = [allocation["credits"][t] + min(q, 0.0) - penalties[t] for t, q in enumerate(base)]
    mode = getattr(args, "finalize_mode", "positive")
    maximum = shaped[0]
    for turn, sample in enumerate(samples):
        if sample.remove_sample:
            targets[turn] = 0.0
            continue
        rejected = (
            turn > 0
            and maximum > 0
            and ((mode == "improve" and shaped[turn] <= maximum) or (mode == "positive" and shaped[turn] <= 0))
        )
        protected = rejected and allocation["credits"][turn] > 0
        if rejected and not protected:
            _mark_remove_sample(sample, f"finalize_{mode}")
            targets[turn] = 0.0
        maximum = max(maximum, shaped[turn])
        sample.metadata["component_reward_soft_finalize_protected"] = bool(protected)
    active = [_trainable(s) for s in samples]
    for turn, sample in enumerate(samples):
        sample.reward = targets[turn]
        sample.metadata.update(
            {
                "task_reward": base[turn],
                "multi_turn_reward": targets[turn],
                "trajectory_finish_reason": finish_reason,
                "component_reward": {
                    **allocation,
                    "turn_credit": allocation["credits"][turn],
                    "turn_target": targets[turn],
                    "targets": targets,
                    "active_turns": active,
                    "length_penalties": penalties,
                },
            }
        )
    logger.info(
        "[component_reward] trajectory=%s best=%s status=%s budget=%s credits=%s residual=%s",
        samples[0].index,
        allocation["best_turn"],
        allocation["status"],
        allocation["quality_budget"],
        allocation["credits"],
        allocation["residual_fraction"],
    )
    return samples


def filter_component_reward_group(args, samples, **kwargs):
    """Same filter thresholds over turn target vectors, even when last is zero."""
    if not getattr(args, "component_reward", False):
        from .kernel_filter import filter_cuda_kernel_group

        return filter_cuda_kernel_group(args, samples, **kwargs)
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    from .config import CUDA_AGENT_CONFIGS

    config = CUDA_AGENT_CONFIGS.get("filter", {})
    target = getattr(args, "target_group_size", None) or config.get("target_group_size") or args.n_samples_per_prompt
    minimum = getattr(args, "min_group_size", None)
    minimum = config.get("min_group_size") if minimum is None else minimum
    minimum = target // 2 + 1 if minimum is None else minimum
    threshold = getattr(args, "reward_std_threshold", None)
    threshold = float(config.get("reward_std_threshold", 1e-3) if threshold is None else threshold)
    if config.get("reject_small_groups", True) and minimum <= target // 2:
        raise ComponentRewardContractError("component filter min_group_size must exceed half target group")
    trajectories = []
    seen = set()
    for sample in samples:
        key = (sample.group_index, sample.index)
        if key in seen:
            raise ComponentRewardContractError("component filter requires one representative per trajectory")
        seen.add(key)
        record = _metadata(sample).get("component_reward")
        if not isinstance(record, dict) or record.get("method") != METHOD:
            raise ComponentRewardContractError("component filter missing finalized trajectory targets")
        if len(record["targets"]) != len(record["active_turns"]):
            raise ComponentRewardContractError("component filter target/mask mismatch")
        if any(record["active_turns"]):
            trajectories.append(record)
    if config.get("reject_small_groups", True) and len(trajectories) < minimum:
        return DynamicFilterOutput(False, f"group_size_lt_min_{minimum}")
    variances = []
    width = max((len(r["targets"]) for r in trajectories), default=0)
    for turn in range(width):
        values = [
            _finite(r["targets"][turn], "component target")
            for r in trajectories
            if turn < len(r["targets"]) and r["active_turns"][turn]
        ]
        mean = math.fsum(values) / len(values) if values else 0.0
        variances.append(math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values)) if values else 0.0)
    if config.get("reject_low_variance_groups", True) and max(variances, default=0.0) < threshold:
        logger.info(
            "[component_reward_filter] drop group=%s std_by_turn=%s",
            samples[0].group_index if samples else None,
            variances,
        )
        return DynamicFilterOutput(False, f"component_reward_std_lt_{threshold:g}")
    return DynamicFilterOutput(True)


def compute_component_reward_metrics(args, samples):
    """Flat metrics with explicit denominators; allocation copies count once."""
    if not getattr(args, "component_reward", False):
        return {}
    trajectories, turns = {}, {}
    for sample in samples:
        key = (sample.group_index, sample.index)
        record = _metadata(sample).get("component_reward")
        if record is not None:
            if record.get("method") != METHOD:
                raise ComponentRewardContractError("metrics received unknown allocation method")
            trajectories.setdefault(key, record)
        turns.setdefault((*key, _metadata(sample).get("turn_idx")), sample)
    records = list(trajectories.values())
    count = len(records)
    prefix = "component_reward/"
    metrics = {"trajectories": count, "observed_turns": len(turns)}
    best_turns = Counter(r["best_turn"] for r in records)
    positive_best_turns = Counter(r["best_turn"] for r in records if r["quality_budget"] > 0)
    for turn, n in best_turns.items():
        label = "none" if turn is None else str(turn)
        metrics[f"best_turn/{label}/count"] = n
        metrics[f"positive_best_turn/{label}/count"] = positive_best_turns[turn]
    # Missing graph fallback has an unknown unit count, not a measured zero.
    unit_counts = [len(r["units"]) for r in records if r["status"] in {"attributed", "partial"} and r["units"]]
    metrics["best_unit_count/observations"] = len(unit_counts)
    metrics["best_unit_count/mean"] = math.fsum(unit_counts) / len(unit_counts) if unit_counts else 0.0
    metrics["best_unit_count/max"] = max(unit_counts, default=0)
    for status in ("attributed", "partial", "unavailable_best_turn_fallback", "no_positive_budget"):
        n = sum(r["status"] == status for r in records)
        metrics[f"status/{status}/count"] = n
        metrics[f"status/{status}/fraction"] = n / count if count else 0.0
    total_budget = math.fsum(r["quality_budget"] for r in records)
    residual_mass = math.fsum(r["quality_budget"] * r["residual_fraction"] for r in records)
    cross_mass = math.fsum(
        c for r in records for t, c in enumerate(r["credits"]) if r["best_turn"] is not None and t < r["best_turn"]
    )
    cross_trajectories = sum(r.get("resolved_cross_turn_units", 0) > 0 for r in records)
    metrics.update(
        {
            "quality_budget_sum": total_budget,
            "residual_mass_sum": residual_mass,
            "residual_fraction_mean": math.fsum(r["residual_fraction"] for r in records) / count if count else 0.0,
            "residual_budget_fraction": residual_mass / total_budget if total_budget else 0.0,
            "resolved_cross_turn_mass_sum": cross_mass,
            "resolved_cross_turn_budget_fraction": cross_mass / total_budget if total_budget else 0.0,
            "resolved_cross_turn_trajectories": cross_trajectories,
            "resolved_cross_turn_fraction": cross_trajectories / count if count else 0.0,
            "soft_finalize_protected_turns": sum(
                bool(_metadata(s).get("component_reward_soft_finalize_protected")) for s in turns.values()
            ),
        }
    )
    width = max((len(r["credits"]) for r in records), default=0)
    for turn in range(width):
        metrics[f"credit_by_turn/{turn}/sum"] = math.fsum(
            r["credits"][turn] for r in records if turn < len(r["credits"])
        )
    statuses = Counter()
    costs = []
    for sample in turns.values():
        payload = _metadata(sample).get("runtime_graph")
        statuses[payload.get("status", "invalid") if isinstance(payload, dict) else "missing"] += 1
        cost = payload.get("cost", {}).get("wall_seconds") if isinstance(payload, dict) else None
        if cost is not None:
            costs.append(_finite(cost, "runtime graph wall_seconds"))
    for status, n in statuses.items():
        metrics[f"runtime_graph/status/{status}/count"] = n
    metrics["runtime_graph/cost_observations"] = len(costs)
    metrics["runtime_graph/wall_seconds_sum"] = math.fsum(costs)
    return {prefix + key: value for key, value in metrics.items()}
