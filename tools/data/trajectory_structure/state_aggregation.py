"""Build a reference-aligned structural-state pilot from reviewed real trajectories.

Produce operation graphs with source evidence, address-contract checks,
structural transitions and pairwise comparisons. Never train or score kernels.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

from .semantic_state import actual_arguments, compare_cards, propagate_contract_warnings, validate_graph
from .structural_state_pilot import PILOT, attention_nodes, attention_wire_trace, mlp_nodes


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_constants(source):
    result = {}
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign):
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = value
    return result


def interface_state(turn):
    source = turn["sections"].get("MODEL_NEW", {}).get("source", "")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"status": "unknown_python_unparseable"}, []
    methods = [
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "ModelNew"
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name == "forward"
    ]
    calls = [
        n.func.attr
        for method in methods
        for n in ast.walk(method)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "tvm_ffi_extension"
    ]
    exports = {e["name"] for e in turn["exports"]}
    missing = sorted(set(calls) - exports)
    status = "missing_export" if missing else "direct_names_resolve" if calls else "unknown_dynamic_dispatch"
    # Names are evidence, not the equivalence criterion.
    return {"status": status, "scope": "literal_forward_calls_only"}, [
        {"calls": calls, "exports": sorted(exports), "missing": missing}
    ]


def make_card(structure, raw, turn_index):
    model, _, reference, task, trajectory = structure["identity"]
    turn = structure["turns"][turn_index]
    row = raw[turn_index]
    if turn["turn_idx"] != turn_index or row["turn_idx"] != turn_index:
        raise ValueError("misordered snapshots")
    if hashlib.sha256(row["response"].encode()).hexdigest() != turn["observation"]["response_sha256"]:
        raise ValueError("response hash mismatch")
    if hashlib.sha256(row["label"]["ground_truth"].encode()).hexdigest() != reference:
        raise ValueError("reference hash mismatch")
    constants = reference_constants(row["label"]["ground_truth"])
    is_mlp = task == "1"
    expected = (
        {"batch_size": 128, "input_size": 16384, "layer_sizes": [16384, 16384], "output_size": 8192}
        if is_mlp
        else {
            "batch_size": 128,
            "seq_len": 512,
            "n_embd": 768,
            "n_head": 8,
            "max_seqlen": 1024,
            "attn_pdrop": 0.0,
            "resid_pdrop": 0.0,
        }
    )
    if any(constants.get(k) != v for k, v in expected.items()):
        raise ValueError("pilot adapter/reference dimensions mismatch")
    card = {
        "id": f"{model}_g{trajectory}_t{turn_index+1}",
        "model": model,
        "trajectory": trajectory,
        "task": task,
        "reference_sha256": reference,
        "snapshot_turn": turn_index,
        "decision_turn": turn_index + 1,
        "horizon": 5,
        "response_sha256": turn["observation"]["response_sha256"],
        "mapping_origin": "human_reviewed_task_adapter_not_automatic_semantic_inference",
        "mapping_complete": False,
        "nodes": [],
        "unknowns": [
            "operator_local_correctness",
            "all_latent_errors",
            "performance_equivalence",
            "full_history_policy_equivalence",
            "runtime_execution_of_each_mapped_path",
        ],
        "evaluation_evidence": {"outcome": row["outcome"], "compiled": row["compiled"], "correct": row["correct"]},
        "diagnostic_evidence": row["compiler_diagnostics"],
        "observed_blockers": {
            "reported_categories": sorted(
                set((row.get("compiler_categories") or []) + (row.get("runtime_categories") or []))
            ),
            "undefined_symbols": sorted(
                set(re.findall(r'identifier "([^"]+)" is undefined', "\n".join(row["compiler_diagnostics"])))
            ),
            "coverage": "reported_only_not_all_latent_errors",
        },
        "history_outcomes": [r["outcome"] for r in raw[: turn_index + 1]],
    }
    card["interface"], card["interface_evidence"] = interface_state(turn)
    mode_fields = {"cublasSetMathMode": 1, "cublasGemmEx": 17, "cublasLtMatmulDescCreate": 1}
    mode_calls = [c for c in turn["calls"] if c["target"] in mode_fields]
    card["math_mode_requests"] = sorted(
        {c["target"] + ":" + str(actual_arguments(c)[mode_fields[c["target"]]]) for c in mode_calls}
    ) or ["no_explicit_request_observed"]
    if is_mlp:
        card["orchestration"] = "native_three_layers" if model == "dsv4" and trajectory == "0" else "python_per_layer"
        card["allocation"] = (
            "native_hidden_buffers" if card["orchestration"] == "native_three_layers" else "python_layer_outputs"
        )
    else:
        card["orchestration"] = (
            "binding_level_stage_sequence" if model == "qwen38" and trajectory == "348" else "native_stage_sequence"
        )
        card["allocation"] = (
            "python_separate_intermediates"
            if model == "qwen38" and trajectory == "348"
            else "python_single_slab_with_native_views" if model == "qwen38" else "native_separate_intermediates"
        )
    if not turn["complete_sections"] or not turn["valid_modelnew"]:
        card["mapping_issue"] = "selected_source_incomplete"
        return card
    try:
        card["nodes"] = (
            mlp_nodes(turn, model, int(trajectory)) if is_mlp else attention_nodes(turn, model, int(trajectory))
        )
        if not is_mlp:
            attention_wire_trace(card["nodes"], model, int(trajectory))
        if is_mlp:
            externals = ["input"] + [f"{p}{i}" for p in ("weight", "bias") for i in range(3)]
        else:
            externals = [
                "input",
                "ln1_weight",
                "ln1_bias",
                "qkv_weight",
                "qkv_bias",
                "attention_weight",
                "attention_bias",
                "ln2_weight",
                "ln2_bias",
                "fc_weight",
                "fc_bias",
                "projection_weight",
                "projection_bias",
            ]
        validate_graph(card["nodes"], externals)
        propagate_contract_warnings(card["nodes"])
        card["mapping_complete"] = True
    except (ValueError, IndexError, KeyError) as exc:
        card["mapping_issue"] = str(exc)
        card["nodes"] = []
    return card


def comparison_summary(cards):
    pairs, transitions = [], []
    groups = collections.defaultdict(list)
    histories = collections.defaultdict(list)
    for c in cards:
        groups[c["model"], c["task"], c["decision_turn"]].append(c)
        histories[c["model"], c["trajectory"]].append(c)
    for group in groups.values():
        if len(group) != 2:
            raise ValueError("pilot requires two trajectories per model/task/turn")
        a, b = group
        pairs.append({"left": a["id"], "right": b["id"], **compare_cards(a, b)})
    for history in histories.values():
        ordered = sorted(history, key=lambda c: c["snapshot_turn"])
        for a, b in zip(ordered, ordered[1:], strict=False):
            transitions.append({"before": a["id"], "after": b["id"], **compare_cards(a, b)})
    summary = {
        "cards": len(cards),
        "complete_mappings": sum(c["mapping_complete"] for c in cards),
        "models": {},
        "same_turn_pairs": len(pairs),
        "pair_relations": dict(collections.Counter(p["relation"] for p in pairs)),
    }
    for model in PILOT:
        selected = [c for c in cards if c["model"] == model]
        pair_ids = {p[k] for p in pairs if p["relation"] == "same_computation_plan" for k in ("left", "right")}
        exact_ids = {p[k] for p in pairs if p.get("same_recorded_state") for k in ("left", "right")}
        summary["models"][model] = {
            "cards": len(selected),
            "complete_mapping_pct": 100 * sum(c["mapping_complete"] for c in selected) / len(selected),
            "same_plan_candidate_coverage_pct": 100 * sum(c["id"] in pair_ids for c in selected) / len(selected),
            "same_recorded_state_coverage_pct": 100 * sum(c["id"] in exact_ids for c in selected) / len(selected),
            "training_equivalence": "not_established",
        }
    return summary, pairs, transitions


def card_markdown(c):
    lines = [
        f"# {c['id']}：第 {c['snapshot_turn']+1} 版程序状态",
        "",
        f"用于进入第 {c['decision_turn']+1} 轮前的状态描述；人工审核适配器提取，非自动语义证明",
        "",
        "| 参考计算环节 | 候选实现 | 候选激活输入角色 | 候选参数角色 | 地址检查 | 上游已见冲突 |",
        "|---|---|---|---|---|---|",
    ]
    for n in c["nodes"]:
        lines.append(
            f"| {n['slot']} | {n['implementation']} | {n.get('candidate_activation_inputs')} | "
            f"{n.get('candidate_parameter_inputs')} | {n.get('index_contract',{}).get('status','unknown')} | "
            f"{n.get('upstream_contract_conflicts')} |"
        )
    lines += [
        "",
        "## 已观测接口和问题",
        "",
        "```json",
        json.dumps(
            {
                k: c.get(k)
                for k in [
                    "interface",
                    "interface_evidence",
                    "mapping_issue",
                    "evaluation_evidence",
                    "diagnostic_evidence",
                    "unknowns",
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]
    for n in c["nodes"]:
        lines += ["", f"## {n['slot']}：证据", "", "```json", json.dumps(n, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cards").mkdir()
    root = args.structure_root
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["code_and_inputs_unchanged"]:
        raise ValueError("unverified source archive")
    sources = {s["name"]: s for s in manifest["inputs"]}
    index = json.loads((root / "trajectory_index.json").read_text())
    inputs = {str(root / name): file_sha(root / name) for name in ("manifest.json", "trajectory_index.json")}
    implementations = {
        str(p): file_sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("semantic_state.py"),
            Path(__file__).with_name("structural_state_pilot.py"),
            Path(__file__).with_name("check_state_aggregation.py"),
        ]
    }
    cards = []
    for model, groups in PILOT.items():
        for group in groups:
            entry = next(i for i in index if i["identity"][0] == model and i["identity"][-1] == str(group))
            path = root / entry["path"]
            inputs[str(path)] = file_sha(path)
            structure = json.loads(path.read_text())
            raw_path = Path(sources[model]["path"]) / f"g{group:03d}.json"
            inputs[str(raw_path)] = file_sha(raw_path)
            if inputs[str(raw_path)] != sources[model]["files"][raw_path.name]:
                raise ValueError("raw archive has changed")
            raw = json.loads(raw_path.read_text())
            for t in range(4):
                card = make_card(structure, raw, t)
                card.update(structure_path=str(path), raw_path=str(raw_path))
                cards.append(card)
                (args.output / "cards" / f"{card['id']}.json").write_text(
                    json.dumps(card, ensure_ascii=False, indent=2) + "\n"
                )
                (args.output / "cards" / f"{card['id']}.md").write_text(card_markdown(card))
    summary, pairs, transitions = comparison_summary(cards)
    for name, value in [("summary", summary), ("pairs", pairs), ("transitions", transitions)]:
        (args.output / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    stable = all(file_sha(p) == h for p, h in (inputs | implementations).items())
    provenance = {
        "inputs": inputs,
        "implementation": implementations,
        "code_and_inputs_unchanged": stable,
        "argv": sys.argv,
        "pilot": PILOT,
        "mapping": "human-reviewed fixed-task adapters",
        "execution": "CPU source/address checks only; no candidate CUDA execution; no RL",
        "selection": "purposive small pilot, NOT natural-pool prevalence",
        "information_cutoff": "snapshot t and earlier; decision at t+1; no future results",
        "unknown_policy": "never mark components correct from whole-program compile/correct flags",
    }
    (args.output / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    if not stable:
        raise RuntimeError("sources changed; invalidate this run")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
