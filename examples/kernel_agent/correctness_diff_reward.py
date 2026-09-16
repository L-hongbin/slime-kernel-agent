"""Correctness-only repair credit added once to existing TRLOO returns.

Earlier completed answers receive credit when their three code sections are
close to the first implementation accepted by the original evaluation. Distinct qualifying versions
share one extra budget; original returns, masks and LOO stay unchanged. This is
a code-proximity heuristic, not a claim that every retained line was necessary.
Whole-turn counterfactual helpers remain available for separate offline audits.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import io
import json
import logging
import math
import random
import re
import tokenize
from collections import defaultdict

from .utils import parse_cuda_agent_response

logger = logging.getLogger(__name__)
SECTIONS = ("CUDA_KERNELS", "APPLY_BINDINGS", "MODEL_NEW")
SCHEMA = "correctness-diff-credit/v3"

_CPP_TOKENS = re.compile(
    r'(?P<comment>/\*[\s\S]*?\*/|//[^\n]*)|(?:"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')'
    r"|[A-Za-z_]\w*|\d+(?:\.\d+)?|::|->|\+\+|--|<<|>>|&&|\|\||[+\-*/%&|^!=<>]=|[^\s]"
)


def code_tokens(source, section):
    """Strip formatting/comments, preserving Python floor division and strings."""
    if section != "MODEL_NEW":
        return [m.group(0) for m in _CPP_TOKENS.finditer(source) if m.lastgroup != "comment"]
    ignored = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
    try:
        structural = {tokenize.INDENT: "<INDENT>", tokenize.DEDENT: "<DEDENT>", tokenize.NEWLINE: "<NEWLINE>"}
        return [
            structural.get(t.type, t.string)
            for t in tokenize.generate_tokens(io.StringIO(source).readline)
            if t.type not in ignored
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None


def program_similarity(previous, correct_response):
    before, final = selected_sections(previous), selected_sections(correct_response)
    if before is None or final is None:
        return {"known": False, "reason": "incomplete_selected_sections"}
    details, changed, canonical = {}, False, {}
    for section in SECTIONS:
        old, new = code_tokens(before[section], section), code_tokens(final[section], section)
        if not old or not new:
            return {"known": False, "reason": "unavailable_code_tokens"}
        canonical[section] = old
        matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        changed |= old != new
        details[section] = {
            "edit_ratio": 1 - matched / max(len(old), len(new)),
            "old_tokens": len(old),
            "final_tokens": len(new),
            "matched_tokens": matched,
        }
    return {
        "known": True,
        "changed": changed,
        "sections": details,
        "max_section_edit_ratio": max(d["edit_ratio"] for d in details.values()),
        "previous_version": hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest(),
    }


def selected_sections(response):
    native, python = parse_cuda_agent_response(response)
    sections = {
        "CUDA_KERNELS": native.get("kernels/generated.cu", ""),
        "APPLY_BINDINGS": native.get("kernels/generated_binding.cpp", ""),
        "MODEL_NEW": python or "",
    }
    return sections if all(sections.values()) else None


def exact_transplant(base, after, target, context=2):
    """Transport all contextual line edits exactly and uniquely, or abstain."""
    a, b = base.splitlines(keepends=True), after.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    edits, located = [], []
    for group in matcher.get_grouped_opcodes(context):
        old = "".join(a[group[0][1] : group[-1][2]])
        new = "".join(b[group[0][3] : group[-1][4]])
        if not old or target.count(old) != 1:
            return None, [], "missing_or_ambiguous_exact_context"
        pos = target.index(old)
        if any(pos < end and start < pos + len(old) for start, end, _ in located):
            return None, [], "overlapping_transport_blocks"
        edits.append({"old": old, "new": new})
        located.append((pos, pos + len(old), new))
    output = target
    for start, end, new in sorted(located, reverse=True):
        output = output[:start] + new + output[end:]
    return output, edits, None


def whole_turn_bundle(responses):
    if len(responses) != 3:
        raise ValueError("this first-correct-T3 pilot requires three responses")
    programs = [selected_sections(r) for r in responses]
    if not all(programs):
        return {"state": "unknown", "reason": "incomplete_selected_sections"}
    if programs[0] == programs[1]:
        return {"state": "unknown", "reason": "middle_has_no_source_change"}
    if programs[1] == programs[2]:
        return {"state": "unknown", "reason": "correct_anchor_has_no_source_change"}
    edits, failures, result = [], [], {}
    for name in SECTIONS:
        base, final, target = programs[1][name], programs[2][name], programs[0][name]
        on_base, _, error = exact_transplant(base, final, base)
        if error or on_base != final:
            failures.append({"section": name, "reason": error or "base_reconstruction_mismatch"})
            continue
        merged, changes, error = exact_transplant(base, final, target)
        if error:
            failures.append({"section": name, "reason": error})
        else:
            result[name] = merged
            edits.extend({"section": name, **e} for e in changes)
    if failures:
        return {"state": "unknown", "reason": "B_cannot_be_transported_exactly", "failures": failures}
    sequential = dict(programs[0])
    for e in edits:
        text = sequential[e["section"]]
        if text.count(e["old"]) != 1:
            return {"state": "unknown", "reason": "sequential_transport_ambiguous"}
        sequential[e["section"]] = text.replace(e["old"], e["new"])
    if sequential != result:
        return {"state": "unknown", "reason": "sequential_transport_mismatch"}
    return {
        "state": "transportable_candidate",
        "reason": None,
        "edits_for_01": edits,
        "transport_block_count": len(edits),
        "unit": "entire_T1_to_T2_source_change",
        "contribution": "not_evaluated",
        "source_provenance_is_not_causality": True,
    }


def render_sections(sections):
    return "\n\n".join(
        f"### {name}\n```{lang}\n{sections[name]}\n```"
        for name, lang in zip(SECTIONS, ("cpp", "cpp", "python"), strict=True)
    )


def source_hash(response):
    sections = selected_sections(response)
    return hashlib.sha256(json.dumps(sections, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def env_state(sample):
    return (sample.metadata.get("env_result") or {}).get("env_state") or {}


def completed(sample):
    return getattr(sample.status, "value", sample.status) == "completed"


def correct(sample):
    state = env_state(sample)
    return (
        completed(sample)
        and not sample.metadata.get("is_pad_turn")
        and state.get("status") == "completed"
        and state.get("compiled") is True
        and state.get("correctness") is True
        and state.get("decoy_kernel") is False
    )


def eligible(sample):
    return (
        completed(sample)
        and not sample.remove_sample
        and not sample.metadata.get("is_pad_turn")
        and not env_state(sample).get("decoy_kernel")
        and sample.response_length > 0
        and (sample.loss_mask is None or sum(sample.loss_mask) > 0)
        and selected_sections(sample.response) is not None
    )


def validate_args(args):
    mode = getattr(args, "correctness_diff_mode", "off")
    if mode == "off":
        return
    if mode not in {"baseline", "diff", "shuffled"}:
        raise ValueError("invalid correctness diff mode")
    if getattr(args, "component_reward", False):
        raise ValueError("correctness diff credit and FastCredit are mutually exclusive")
    if args.advantage_estimator != "trloo" or not args.use_multi_turn or args.max_turns != 3:
        raise ValueError("correctness diff pilot requires three-turn TRLOO")
    scale = float(getattr(args, "correctness_diff_scale", 0.25))
    maximum = float(getattr(args, "correctness_diff_max_edit_ratio", 0.1))
    if not math.isfinite(maximum) or not 0 < maximum < 1:
        raise ValueError("maximum code edit ratio must be finite and strictly between zero and one")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("credit scale must be finite/nonnegative")


def task_hash(sample):
    identity = {
        "label": sample.label,
        "entry_point": sample.metadata.get("entry_point"),
        "precision": sample.metadata.get("precision"),
        "augmentation": sample.metadata.get("augmentation"),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


async def annotate_trajectory(args, samples):
    """Use the original evaluator verdict; annotation submits no environment requests.

    Stability replays belong to offline audits, not the training credit gate.
    """
    if getattr(args, "correctness_diff_mode", "off") == "off" or not isinstance(samples, list):
        return samples
    validate_args(args)
    ordered = sorted(samples, key=lambda s: int(s.metadata["turn_idx"]))
    if [int(s.metadata["turn_idx"]) for s in ordered] != list(range(len(ordered))):
        raise ValueError("broken turn ordering")
    anchor = next((i for i, s in enumerate(ordered) if correct(s)), None)
    maximum = float(getattr(args, "correctness_diff_max_edit_ratio", 0.1))
    records = [
        {
            "schema": SCHEMA,
            "method": "near_first_correct_code",
            "response_hash": source_hash(s.response),
            "task_hash": task_hash(s),
            "anchor_turn": anchor,
            "eligible": False,
            "qualifies": False,
            "credit_weight": 0.0,
            "reason": "no_correct_anchor" if anchor is None else "not_before_anchor",
        }
        for s in ordered
    ]
    if anchor is not None:
        anchor_sample = ordered[anchor]
        for record in records:
            record.update(
                anchor_evidence_source="original_env_result",
                anchor_task_id=anchor_sample.metadata.get("task_id"),
                anchor_response_hash=records[anchor]["response_hash"],
            )
    candidates, seen_versions = [], set()
    if anchor is not None and anchor > 0:
        if any(task_hash(s) != task_hash(ordered[anchor]) for s in ordered):
            raise ValueError("mixed task identity within trajectory")
        for t, sample in enumerate(ordered[:anchor]):
            if not eligible(sample):
                records[t]["reason"] = "ineligible_earlier_turn"
                continue
            records[t]["eligible"] = True
            similarity = await asyncio.to_thread(program_similarity, sample.response, ordered[anchor].response)
            records[t]["similarity"] = similarity
            if not similarity["known"]:
                records[t]["reason"] = similarity["reason"]
            elif not similarity["changed"]:
                records[t]["reason"] = "same_code_as_correct_anchor"
            elif similarity["max_section_edit_ratio"] > maximum + 1e-12:
                records[t]["reason"] = "code_difference_too_large"
            elif similarity["previous_version"] in seen_versions:
                records[t]["reason"] = "duplicate_earlier_version"
            else:
                seen_versions.add(similarity["previous_version"])
                records[t]["qualifies"] = True
                records[t]["reason"] = "near_correct_candidate"
                candidates.append(t)

        for t in candidates:
            records[t]["credit_weight"] = 1.0 / len(candidates)
            records[t]["reason"] = "near_correct_earlier_version"

    for sample, record in zip(ordered, records, strict=True):
        sample.metadata = {**sample.metadata, "correctness_diff": record}
    return samples


def credited_returns(args, samples, baseline_returns):
    """Add bounded extra budgets before the unchanged TRLOO leave-one-out."""
    validate_args(args)
    mode = getattr(args, "correctness_diff_mode", "off")
    if mode == "off":
        return baseline_returns
    # A client deadline means no evaluator verdict was received. Its synthetic
    # zero must not train either this sample or its same-turn LOO peers. Abort
    # the batch even when the affected sample was masked: cumulative returns
    # may already have propagated the unresolved score to other turns.
    for sample in samples:
        state = env_state(sample)
        message = str(state.get("error_message") or "").lower()
        if state.get("status") == "timeout" and "(client-side)" in message:
            raise RuntimeError(
                "Unresolved KernelGym client-side evaluation timeout; refuse correctness experiment update: "
                f"group_index={sample.group_index} group_id={sample.group_id} "
                f"turn={sample.metadata.get('turn_idx')} task_id={sample.metadata.get('task_id')}"
            )
    scale = float(getattr(args, "correctness_diff_scale", 0.25))
    trajectories = defaultdict(dict)
    for i, sample in enumerate(samples):
        record = sample.metadata.get("correctness_diff")
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            raise ValueError("missing correctness-diff evidence; refuse partial code synchronization")
        if record["response_hash"] != source_hash(sample.response) or record["task_hash"] != task_hash(sample):
            raise ValueError("correctness-diff source/task binding mismatch")
        turn = int(sample.metadata["turn_idx"])
        key = (sample.group_index, sample.group_id)
        if turn in trajectories[key]:
            raise ValueError("duplicate trajectory turn")
        trajectories[key][turn] = i
        weight = record["credit_weight"]
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("invalid correctness credit weight")
        if weight and (
            not record["eligible"]
            or not record["qualifies"]
            or not eligible(sample)
            or record.get("anchor_evidence_source") != "original_env_result"
        ):
            raise ValueError("credit on an ineligible turn")
    vectors, buckets = {}, defaultdict(list)
    for key, mapping in trajectories.items():
        if set(mapping) != set(range(args.max_turns)):
            raise ValueError("incomplete trajectory for correctness credit")
        vector = {t: samples[i].metadata["correctness_diff"]["credit_weight"] for t, i in mapping.items()}
        if sum(vector.values()) > 1 + 1e-8:
            raise ValueError("extra correctness budget exceeds one trajectory budget")
        vectors[key] = vector
        anchors = {samples[i].metadata["correctness_diff"]["anchor_turn"] for i in mapping.values()}
        if len(anchors) != 1:
            raise ValueError("inconsistent correct anchor")
        anchor = next(iter(anchors))
        if anchor is not None or any(vector.values()):
            if anchor not in mapping or not correct(samples[mapping[anchor]]):
                raise ValueError("missing or invalid correct anchor")
            anchor_sample = samples[mapping[anchor]]
            anchor_hash = source_hash(anchor_sample.response)
            if any(
                samples[i].metadata["correctness_diff"].get("anchor_response_hash") != anchor_hash
                for i in mapping.values()
            ):
                raise ValueError("correct anchor source binding mismatch")
            if any(
                samples[i].metadata["correctness_diff"].get("anchor_evidence_source") != "original_env_result"
                or samples[i].metadata["correctness_diff"].get("anchor_task_id")
                != anchor_sample.metadata.get("task_id")
                for i in mapping.values()
            ):
                raise ValueError("correct anchor evaluation binding mismatch")
        mask = tuple(
            sorted(
                t
                for t, i in mapping.items()
                if samples[i].metadata["correctness_diff"]["eligible"] and eligible(samples[i])
            )
        )
        # Permute entire vectors only across compatible trajectories, preserving
        # per-trajectory budget, per-turn group totals and recipient counts.
        buckets[(key[0], next(iter(anchors)), mask)].append(key)
    assigned = {key: {t: 0.0 for t in mapping} for key, mapping in trajectories.items()}
    if mode == "diff":
        assigned = vectors
    elif mode == "shuffled":
        for bucket, keys in buckets.items():
            keys = sorted(keys, key=lambda key: str(key[1]))
            donors = list(keys)
            seed = hashlib.sha256(f'{getattr(args,"correctness_diff_seed",42)}:{bucket}'.encode()).digest()
            random.Random(seed).shuffle(donors)
            for recipient, donor in zip(keys, donors, strict=True):
                assigned[recipient] = vectors[donor]
    targets = [float(x) for x in baseline_returns]
    for key, mapping in trajectories.items():
        for t, i in mapping.items():
            weight = assigned[key].get(t, 0.0)
            sample = samples[i]
            if weight and (not sample.metadata["correctness_diff"]["eligible"] or not eligible(sample)):
                raise ValueError("shuffled credit on ineligible turn")
            targets[i] += scale * weight
            sample.metadata["correctness_diff_target"] = {
                "mode": mode,
                "baseline_return": float(baseline_returns[i]),
                "bonus": scale * weight,
                "target": targets[i],
                "recipient": weight > 0,
            }
    logger.info(
        "[correctness_diff] mode=%s trajectories=%s diff_recipients=%s bonus_sum=%.6f",
        mode,
        len(trajectories),
        sum(v > 0 for d in vectors.values() for v in d.values()),
        sum(target - float(base) for target, base in zip(targets, baseline_returns, strict=True)),
    )
    return targets


def batch_metrics(samples):
    trajectories = defaultdict(list)
    for s in samples:
        trajectories[(s.group_index, s.group_id)].append(s)
    records = [s.metadata["correctness_diff"] for s in samples]
    targets = [s.metadata["correctness_diff_target"] for s in samples]
    covered = sum(
        any(s.metadata["correctness_diff"]["credit_weight"] > 0 for s in rows) for rows in trajectories.values()
    )
    return {
        "correctness_diff/trajectories": len(trajectories),
        "correctness_diff/eligible_turns": sum(r["eligible"] for r in records),
        "correctness_diff/selected_turns": sum(r["credit_weight"] > 0 for r in records),
        "correctness_diff/covered_trajectories": covered,
        "correctness_diff/coverage": covered / len(trajectories) if trajectories else 0.0,
        "correctness_diff/credited_count": sum(t["recipient"] for t in targets),
        "correctness_diff/bonus_sum": sum(t["bonus"] for t in targets),
        "correctness_diff/shuffle_recipient_changed_count": sum(
            t["mode"] == "shuffled" and bool(r["credit_weight"]) != t["recipient"]
            for r, t in zip(records, targets, strict=True)
        ),
    }
