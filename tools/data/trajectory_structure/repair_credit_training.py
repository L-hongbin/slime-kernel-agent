"""Whole-turn repair bundles and offline shadow targets on retained TRLOO batches.

Never modifies training samples, deploys a service, or launches training. Native
code is only executed by the separately authorized explicit replay runner.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

from examples.kernel_agent.correctness_diff_reward import selected_sections, whole_turn_bundle
from examples.kernel_agent.utils import extract_cuda_agent_kernel_code

from .analyze import file_sha, normalize
from .repair_credit_audit import assert_parser_matches, correct, first_correct, phase
from .repair_credit_replay import build_variant

SECTIONS = ("CUDA_KERNELS", "APPLY_BINDINGS", "MODEL_NEW")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def training_phase(row):
    error = str(row.get("environment_error_message") or "")
    if row.get("correctness") is not True and (
        "Kernel execution failed:" in error or "CudaFinalSyncError" in error or "an illegal memory access" in error
    ):
        return "runtime_error"
    return phase(row)


def load_frozen_postprocess(runtime):
    """Compile only the archived scalar postprocess, without importing its runtime."""
    import torch

    path = runtime / "examples/kernel_agent/kernel_reward.py"
    tree = ast.parse(path.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reward_post_process_by_group"
    )
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    args = SimpleNamespace(
        advantage_estimator="trloo",
        use_multi_turn=True,
        grpo_std_normalization=False,
        use_conditional_truncation_mask=False,
    )

    def compute(rows, bonuses=None):
        bonuses = bonuses or {}
        samples = [
            SimpleNamespace(
                group_index=r["group_index"],
                remove_sample=r["remove_sample"],
                metadata={
                    "turn_idx": r["turn_idx"],
                    "multi_turn_reward": r["baseline_return"] + bonuses.get(r["row_id"], 0.0),
                },
            )
            for r in rows
        ]
        return scope["reward_post_process_by_group"](args, samples)

    return compute


def independent_loo(rows, bonuses=None):
    bonuses = bonuses or {}
    groups = collections.defaultdict(list)
    for r in rows:
        if not r["remove_sample"]:
            groups[(r["group_index"], r["turn_idx"])].append(r)
    result = {}
    for values in groups.values():
        rewards = [r["baseline_return"] + bonuses.get(r["row_id"], 0.0) for r in values]
        for i, r in enumerate(values):
            result[r["row_id"]] = (
                (rewards[i] - sum(rewards[:i] + rewards[i + 1 :]) / (len(values) - 1)) if len(values) > 1 else 0.0
            )
    return result


def shadow_rows(rows, compute, credited_ids, coefficient):
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    ids = {r["row_id"] for r in rows}
    if not set(credited_ids) <= ids:
        raise ValueError("unknown credited row")
    for r in rows:
        if r["row_id"] in credited_ids and not r["bonus_eligible"]:
            raise ValueError("ineligible row cannot receive a bonus")
    bonuses = {k: coefficient for k in credited_ids}
    _, baseline = compute(rows)
    targets, shadow = compute(rows, bonuses)
    independent = independent_loo(rows, bonuses)
    result = []
    for row, base, target, value in zip(rows, baseline, targets, shadow, strict=True):
        if not row["remove_sample"] and abs(value - independent[row["row_id"]]) > 2e-6:
            raise ValueError("archived postprocess disagrees with independent LOO")
        # Removed/zero-mask raw scalars are not effective training advantages.
        active = row["active_tokens"] > 0
        result.append(
            {
                **row,
                "bonus": bonuses.get(row["row_id"], 0.0),
                "shadow_return": target,
                "baseline_advantage": base,
                "shadow_advantage": value,
                "effective_delta": value - base if active else 0.0,
            }
        )
    return result


def summarize_shadow(rows):
    groups = {
        "active": lambda r: r["active_tokens"] > 0,
        "credited": lambda r: r["bonus"] > 0,
        "correct": lambda r: r["active_tokens"] > 0 and correct(r),
        "first_correct_anchor": lambda r: r["active_tokens"] > 0 and r["turn_idx"] == r["anchor_turn"],
        "truncated": lambda r: r["active_tokens"] > 0 and r["status"] == "truncated",
        "long_incorrect": lambda r: r["active_tokens"] > 0 and r["response_tokens"] >= 8192 and not correct(r),
        "no_correct_trajectory": lambda r: r["active_tokens"] > 0 and r["anchor_turn"] is None,
    }
    out = {}
    for name, predicate in groups.items():
        values = [r for r in rows if predicate(r)]
        n = len(values)
        out[name] = {
            "count": n,
            "increased": sum(r["effective_delta"] > 1e-6 for r in values),
            "decreased": sum(r["effective_delta"] < -1e-6 for r in values),
            "positive_to_nonpositive": sum(
                r["baseline_advantage"] > 1e-6 and r["shadow_advantage"] <= 1e-6 for r in values
            ),
            "negative_to_positive": sum(
                r["baseline_advantage"] < -1e-6 and r["shadow_advantage"] > 1e-6 for r in values
            ),
            "mean_delta": sum(r["effective_delta"] for r in values) / n if n else None,
            "mean_baseline_advantage": sum(r["baseline_advantage"] for r in values) / n if n else None,
            "active_tokens": sum(r["active_tokens"] for r in values),
        }
    return out


def prepare(args):
    import torch

    torch.set_num_threads(1)
    root = args.plan.parent.resolve()
    plan = json.loads(args.plan.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    sync = {Path(r["local_path"]).name: r["sha256"] for r in json.loads((root / "input_sync.json").read_text())}
    all_rows, cases, audits = [], [], []
    code_sha = file_sha(Path(__file__))
    for batch in plan["batches"]:
        name = batch["name"]
        source = root / "inputs" / (name + ".pt")
        if file_sha(source) != sync[source.name]:
            raise ValueError("input differs from synchronized remote file")
        runtime = Path(batch["runtime"]).resolve()
        source_manifest = json.loads((runtime.parent / "source_manifest.json").read_text())
        archived_files = [
            "examples/kernel_agent/utils.py",
            "examples/kernel_agent/kernel_reward.py",
            "slime/ray/rollout.py",
        ]
        for rel in archived_files:
            if file_sha(runtime / rel) != source_manifest[rel]:
                raise ValueError("frozen runtime no longer matches its original manifest: " + rel)
        parser = assert_parser_matches(runtime)
        saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        samples = saved["samples"]
        groups = collections.defaultdict(list)
        ledger = []
        for raw in samples:
            md = raw["metadata"]
            if any(k in md for k in ["component_reward", "source_component_reward"]):
                raise ValueError("FastCredit metadata in baseline input")
            normal = normalize(raw)
            response_length = int(raw["response_length"])
            mask = raw["loss_mask"]
            if mask is not None and (len(mask) != response_length or any(x not in (0, 1) for x in mask)):
                raise ValueError("bad loss mask")
            active = 0 if raw["remove_sample"] else response_length if mask is None else sum(mask)
            gid = raw["group_id"]
            row = {
                **normal,
                "batch": name,
                "row_id": f"{name}_g{gid}_t{normal['turn_idx']}",
                "group_index": raw["group_index"],
                "raw_index": raw["index"],
                "baseline_return": float(md["multi_turn_reward"]),
                "remove_sample": bool(raw["remove_sample"]),
                "active_tokens": int(active),
                "remove_reason": md.get("remove_reason"),
                "is_pad_turn": bool(md.get("is_pad_turn")),
                "repo_name": md.get("repo_name"),
                "source_level": md.get("level"),
                "sampled_rollout_step": md.get("rollout_step"),
                "gen_weight_version": md.get("gen_weight_version"),
            }
            if row["is_pad_turn"] and row["active_tokens"]:
                raise ValueError("padding has active tokens")
            groups[gid].append((row, raw))
            ledger.append(row)
        if len(groups) != 256 or len(ledger) != 768:
            raise ValueError("pilot expects a full 256-trajectory, three-turn training batch")
        prompt_groups = collections.defaultdict(set)
        references = collections.defaultdict(set)
        for gid, values in groups.items():
            values.sort(key=lambda v: v[0]["turn_idx"])
            rows = [v[0] for v in values]
            if len(rows) != 3 or len({r["reference_sha256"] for r in rows}) != 1:
                raise ValueError("broken trajectory identity")
            anchor = first_correct(rows)
            acc = 0.0
            for row in reversed(rows):
                acc += 0.0 if row["remove_sample"] else float(row["reward_observed"])
                if abs(row["baseline_return"] - acc) > 1e-8:
                    raise ValueError("saved return differs from baseline gamma=1 recurrence")
                row["anchor_turn"] = anchor
                row["phase"] = training_phase(row)
                row["bonus_eligible"] = bool(
                    anchor == 2
                    and row["turn_idx"] == 1
                    and not row["remove_sample"]
                    and row["active_tokens"] > 0
                    and row["status"] == "completed"
                    and not row["is_pad_turn"]
                    and selected_sections(row["response"])
                )
                prompt_groups[row["group_index"]].add(gid)
                references[row["group_index"]].add(row["reference_sha256"])
            if anchor == 2:
                bundle = whole_turn_bundle([r["response"] for r in rows])
                cid = f"{name}_g{gid}"
                observed = [
                    v[1]["metadata"].get("env_result", {}).get("env_state", {}).get("metadata", {}).get("precision")
                    for v in values
                ]
                precision = observed[anchor]
                if precision not in {"fp32", "fp16", "bf16"} or any(
                    p is not None and p != precision for p in observed
                ):
                    raise ValueError("missing/inconsistent recorded per-task precision")
                entry = (
                    values[0][1]["label"].get("entry_point") or values[0][1]["metadata"].get("entry_point") or "Model"
                )
                case = {
                    "evaluation_overrides": {"precision": precision, "entry_point": entry},
                    "case_id": cid,
                    "batch": name,
                    "group_id": str(gid),
                    "group_index": rows[0]["group_index"],
                    "anchor_turn": anchor,
                    "phase_path": [r["phase"] for r in rows],
                    "eligible": rows[1]["bonus_eligible"],
                    "bundle": bundle,
                    "reference_code": values[0][1]["label"]["ground_truth"],
                    "raw_turns": rows,
                    "structure": {
                        "turns": [
                            {
                                "turn_idx": r["turn_idx"],
                                "sections": {
                                    k: {"source": v} for k, v in (selected_sections(r["response"]) or {}).items()
                                },
                            }
                            for r in rows
                        ]
                    },
                }
                if bundle["state"] == "transportable_candidate":
                    for turn in range(3):
                        code, _ = build_variant(case, {"base_turn": turn})
                        if code != extract_cuda_agent_kernel_code(rows[turn]["response"]):
                            raise ValueError("replay representation differs from executor-selected source")
                    generated, _ = build_variant(case, {"base_turn": 1, "edits": bundle["edits_for_01"]})
                    final, _ = build_variant(case, {"base_turn": 2})
                    if generated != final:
                        raise ValueError("B applied to T2 did not exactly reconstruct T3")
                write_json(output / "cases" / (cid + ".json"), case)
                cases.append({k: v for k, v in case.items() if k not in {"reference_code", "raw_turns", "structure"}})
        if any(len(v) != 16 for v in prompt_groups.values()) or any(len(v) != 1 for v in references.values()):
            raise ValueError("incomplete or mixed prompt groups")
        compute = load_frozen_postprocess(runtime)
        baseline = shadow_rows(ledger, compute, set(), 0.0)
        for row, validated in zip(ledger, baseline, strict=True):
            row["baseline_advantage"] = validated["baseline_advantage"]
        # Keep reviewable source only in the selected case records, not the compact ledger.
        all_rows.extend(
            {k: v for k, v in r.items() if k not in {"response", "environment_error_message"}} for r in ledger
        )
        audits.append(
            {
                "batch": name,
                "rollout_id": saved["rollout_id"],
                "input_sha256": sync[source.name],
                "runtime": str(runtime),
                "parser": parser,
                "frozen_files": {p: file_sha(runtime / p) for p in archived_files},
                "trajectory_count": len(groups),
                "prompt_group_count": len(prompt_groups),
                "first_correct": dict(collections.Counter(str(v[0][0]["anchor_turn"]) for v in groups.values())),
                "removed_turns": sum(r["remove_sample"] for r in ledger),
                "truncated_turns": sum(r["status"] == "truncated" for r in ledger),
                "baseline_recurrence_verified": True,
                "independent_loo_verified": True,
            }
        )
        del samples, saved, groups, ledger
        print(json.dumps(audits[-1]), flush=True)
    write_json(output / "ledger.json", all_rows)
    write_json(output / "case_index.json", cases)
    eligible = [c for c in cases if c["eligible"] and c["bundle"]["state"] == "transportable_candidate"]
    rng = random.Random(args.seed)
    selection = []
    for batch in plan["batches"]:
        pool = sorted([c["case_id"] for c in eligible if c["batch"] == batch["name"]])
        selection.extend(rng.sample(pool, min(args.review_per_batch, len(pool))))
    summary = {
        "batches": audits,
        "trajectory_count": len(all_rows) // 3,
        "first_T3_count": len(cases),
        "transportable_count": sum(c["bundle"]["state"] == "transportable_candidate" for c in cases),
        "eligible_transportable_count": len(eligible),
        "unknown_reasons": dict(
            collections.Counter(c["bundle"]["reason"] for c in cases if c["bundle"]["state"] == "unknown")
        ),
        "selected_replays": selection,
        "selection_seed": args.seed,
        "selection_method": "seeded within each retained batch, restricted to eligible exact-transplant candidates",
        "population_is_representative_training_sample": False,
        "schema": "whole_turn_bundle_v1",
        "implementation_sha256": code_sha,
        "training_modified": False,
    }
    if file_sha(Path(__file__)) != code_sha:
        raise RuntimeError("implementation changed during audit")
    write_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "batches"}), flush=True)


def shadow(args):
    root = args.audit.resolve()
    rows = json.loads((root / "ledger.json").read_text())
    cases = json.loads((root / "case_index.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    labels = json.loads(args.labels.read_text()) if args.labels else {}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidates = {
        f"{c['case_id']}_t1" for c in cases if c["eligible"] and c["bundle"]["state"] == "transportable_candidate"
    }
    verified = {f"{cid}_t1" for cid, label in labels.items() if label["verdict"] == "A_required_for_tested_pass"}
    if not verified <= candidates:
        raise ValueError("verified allocation outside eligible candidates")
    strategies = {"transportable_candidates": candidates, "verified_subset_only": verified}
    output = {}
    for name, chosen in strategies.items():
        output[name] = {}
        for coefficient in [0.0, 0.1, 0.25]:
            changed = []
            for batch in summary["batches"]:
                batch_rows = [r for r in rows if r["batch"] == batch["batch"]]
                ids = chosen & {r["row_id"] for r in batch_rows}
                compute = load_frozen_postprocess(Path(batch["runtime"]))
                changed.extend(shadow_rows(batch_rows, compute, ids, coefficient))
            output[name][str(coefficient)] = summarize_shadow(changed)
            write_json(out / f"{name}_{coefficient:.2f}.json", changed)
    write_json(
        out / "summary.json",
        {
            "strategies": output,
            "labels": labels,
            "formula": "Y_t = saved_G_t + lambda for chosen middle turns, otherwise saved_G_t",
            "no_speedup_gate_or_speedup_scaled_bonus": True,
            "baseline_filter_and_mask_unchanged": True,
            "training_modified": False,
            "unreviewed_is_not_a_negative_label": True,
            "implementation_sha256": file_sha(Path(__file__)),
        },
    )
    print(json.dumps(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--plan", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=20260914)
    prep.add_argument("--review-per-batch", type=int, default=4)
    sim = sub.add_parser("shadow")
    sim.add_argument("--audit", type=Path, required=True)
    sim.add_argument("--labels", type=Path)
    sim.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else shadow)(args)


if __name__ == "__main__":
    main()
