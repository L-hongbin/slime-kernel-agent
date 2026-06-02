#!/usr/bin/env python3
"""Analyze W4A16 AWQ vs W8A8 DrKernel full-eval gap.

The goal is to connect the low W4 score to observable rollout behavior, not to
another local MLP-loss proxy.  The script reads saved eval_0.pt dumps and writes
turn-level structure/quality summaries plus paired W4-vs-W8 transition counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODELS = {
    "BF16": Path(
        "checkpoints/Qwen3.6-27B/"
        "20260529_132454_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
    "W8A8_RTN_NonLA": Path(
        "checkpoints/Qwen3.6-27B-W8A8-RTN-nonla-mtp/"
        "20260530_073700_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
    "W8A8_SQRTN_MLP": Path(
        "checkpoints/Qwen3.6-27B-SQ-W8A8-RTN-mlp-mtp-a0p5-ultrachat/"
        "20260530_165501_smooth.mlp_mtp.100x8.eagle_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
    "W4A16_AWQ_ASYM": Path(
        "checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/"
        "20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
}

SECTION_MARKERS = ("### CUDA_KERNELS", "### APPLY_BINDINGS", "### MODEL_NEW")
CODE_HINTS = ("```", "#include", "import torch", "from torch", "class Model", "def forward")
NARRATIVE_HINTS = (
    "we need",
    "we can",
    "i will",
    "let's",
    "the task",
    "the user",
    "analysis",
    "approach",
    "solution",
)


def kg_error_text(kg: dict[str, Any]) -> str:
    pieces = [str(kg.get("error_message") or ""), str(kg.get("error_code") or "")]
    metadata = kg.get("metadata") or {}
    compile_artifact = metadata.get("compile_artifact") or {}
    pieces.append(str(compile_artifact.get("error") or ""))
    pieces.append(str(kg.get("stderr") or ""))
    pieces.append(str(kg.get("stdout") or ""))
    return "\n".join(piece for piece in pieces if piece)


def classify_kernelgym_error(kg: dict[str, Any]) -> str:
    """Coarse categories for comparing failed generated kernels."""
    if bool(kg.get("correctness")):
        return "correct"
    if bool(kg.get("compiled")):
        return "compiled_wrong"

    text = kg_error_text(kg).lower()
    if "ktvmfloat" in text:
        return "kTVMFloat"
    if "no member named" in text or "has no member" in text:
        return "no_member"
    if "not declared in this scope" in text or "was not declared" in text:
        return "not_declared"
    if "no matching function" in text or "no matching" in text:
        return "no_matching"
    if "expected" in text or "syntax error" in text:
        return "syntax_or_expected"
    if "ninja exited" in text or "compilation failed" in text or "error:" in text:
        return "compile_other"
    if str(kg.get("status") or "").lower() == "failed":
        return "failed_other"
    return "other_wrong"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/quantized/analysis/w4_awq_eval_gap_20260601"),
    )
    parser.add_argument(
        "--models-json",
        type=Path,
        help="Optional JSON mapping model names to eval_0.pt paths. Defaults to current known runs.",
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def first_pos(text: str, needles: tuple[str, ...]) -> int | None:
    positions = [pos for needle in needles if (pos := text.find(needle)) >= 0]
    return min(positions) if positions else None


def extract_spans(sample: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for event in (sample.get("trace") or {}).get("events", []):
        if event.get("name") != "sglang_generate":
            continue
        if event.get("type") == "span_start":
            current = {"start_ts": event.get("ts"), "attrs": event.get("attrs") or {}}
        elif event.get("type") == "span_end":
            attrs = event.get("attrs") or {}
            row = {
                "end_ts": event.get("ts"),
                "prompt_tokens": attrs.get("prompt_tokens"),
                "completion_tokens": attrs.get("completion_tokens"),
                "cached_tokens": attrs.get("cached_tokens"),
                "finish_reason": attrs.get("finish_reason"),
            }
            if current and current.get("start_ts") is not None:
                row["duration_s"] = event.get("ts") - current["start_ts"]
            spans.append(row)
            current = None
    return spans


def span_value(spans: list[dict[str, Any]], turn_idx: int, key: str) -> Any:
    if turn_idx >= len(spans):
        return None
    return spans[turn_idx].get(key)


def text_features(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    lower_head = stripped[:1500].lower()
    section_positions = {marker: text.find(marker) for marker in SECTION_MARKERS}
    present_sections = [marker for marker, pos in section_positions.items() if pos >= 0]
    first_section = first_pos(text, SECTION_MARKERS)
    first_code = first_pos(text, CODE_HINTS)
    all_sections = len(present_sections) == len(SECTION_MARKERS)
    starts_with_section = any(stripped.startswith(marker) for marker in SECTION_MARKERS)
    starts_with_code = starts_with_section or any(stripped.startswith(hint) for hint in CODE_HINTS)
    narrative_hits = sum(lower_head.count(hint) for hint in NARRATIVE_HINTS)
    return {
        "char_len": len(text),
        "empty_response": len(stripped) == 0,
        "all_sections": all_sections,
        "present_section_count": len(present_sections),
        "missing_sections": ",".join(marker for marker, pos in section_positions.items() if pos < 0),
        "first_section_pos": first_section,
        "first_code_pos": first_code,
        "all_sections_by_4k": all_sections and max(section_positions.values()) <= 4000,
        "first_section_by_2k": first_section is not None and first_section <= 2000,
        "first_section_by_4k": first_section is not None and first_section <= 4000,
        "no_section_by_4k": first_section is None or first_section > 4000,
        "starts_with_section": starts_with_section,
        "starts_with_code": starts_with_code,
        "starts_with_narrative": (not starts_with_code) and narrative_hits > 0,
        "narrative_hits_head": narrative_hits,
        "cuda_marker_count": text.count("### CUDA_KERNELS"),
        "fence_count": text.count("```"),
    }


def kernelgym(turn: dict[str, Any]) -> dict[str, Any]:
    return turn.get("kernelgym") or {}


def load_rows(model_name: str, path: Path) -> list[dict[str, Any]]:
    print(f"[gap] loading {model_name}: {path}", flush=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows: list[dict[str, Any]] = []
    for sample in payload["samples"]:
        sample_idx = int(sample["index"])
        meta = sample.get("metadata") or {}
        spans = extract_spans(sample)
        for turn in meta.get("turns") or []:
            turn_idx = int(turn.get("turn_idx", len(rows)))
            response = turn.get("response") or ""
            kg = kernelgym(turn)
            speedup = float(kg.get("speedup") or 0.0)
            error_text = kg_error_text(kg)
            row = {
                "model": model_name,
                "sample_idx": sample_idx,
                "turn_idx": turn_idx,
                "problem_id": meta.get("problem_id"),
                "name": meta.get("name"),
                "response_length": int(turn.get("response_length") or 0),
                "turn_status": turn.get("status"),
                "turn_reward": float(turn.get("reward") or 0.0),
                "compiled": bool(kg.get("compiled")),
                "correct": bool(kg.get("correctness")),
                "speedup": speedup,
                "fast_1_0": speedup >= 1.0,
                "fast_1_2": speedup >= 1.2,
                "kernelgym_status": kg.get("status"),
                "error_category": classify_kernelgym_error(kg),
                "error_snippet": " ".join(error_text.split())[:300],
                "finish_reason": span_value(spans, turn_idx, "finish_reason"),
                "completion_tokens": int(span_value(spans, turn_idx, "completion_tokens") or 0),
                "prompt_tokens": span_value(spans, turn_idx, "prompt_tokens"),
            }
            row.update(text_features(response))
            rows.append(row)
    return rows


def bool_rate(rows: list[dict[str, Any]], key: str) -> float:
    return 100.0 * sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0


def summarize_turn(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["response_length"]) for row in rows]
    first_section_positions = [
        int(row["first_section_pos"]) for row in rows if row.get("first_section_pos") is not None
    ]
    return {
        "n": len(rows),
        "compiled_pct": bool_rate(rows, "compiled"),
        "correct_pct": bool_rate(rows, "correct"),
        "fast1_pct": bool_rate(rows, "fast_1_0"),
        "fast12_pct": bool_rate(rows, "fast_1_2"),
        "resp_mean": statistics.mean(lengths) if lengths else 0.0,
        "resp_median": statistics.median(lengths) if lengths else 0.0,
        "resp_p90": percentile(lengths, 0.90),
        "resp_p99": percentile(lengths, 0.99),
        "resp_max": max(lengths) if lengths else 0,
        "gt_32k": sum(value > 32768 for value in lengths),
        "gt_60k": sum(value > 60000 for value in lengths),
        "finish_length": sum(row.get("finish_reason") == "length" for row in rows),
        "empty_response": sum(bool(row.get("empty_response")) for row in rows),
        "all_sections_pct": bool_rate(rows, "all_sections"),
        "all_sections_by4k_pct": bool_rate(rows, "all_sections_by_4k"),
        "first_section_by2k_pct": bool_rate(rows, "first_section_by_2k"),
        "first_section_by4k_pct": bool_rate(rows, "first_section_by_4k"),
        "no_section_by4k_pct": bool_rate(rows, "no_section_by_4k"),
        "starts_narrative_pct": bool_rate(rows, "starts_with_narrative"),
        "first_section_p50": statistics.median(first_section_positions) if first_section_positions else -1,
        "first_section_p90": percentile(first_section_positions, 0.90) if first_section_positions else -1,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "model",
        "sample_idx",
        "turn_idx",
        "problem_id",
        "name",
        "response_length",
        "finish_reason",
        "compiled",
        "correct",
        "speedup",
        "fast_1_0",
        "fast_1_2",
        "kernelgym_status",
        "error_category",
        "error_snippet",
        "all_sections",
        "present_section_count",
        "missing_sections",
        "first_section_pos",
        "all_sections_by_4k",
        "first_section_by_2k",
        "first_section_by_4k",
        "no_section_by_4k",
        "starts_with_narrative",
        "narrative_hits_head",
        "cuda_marker_count",
        "fence_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_turn_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["model"]), int(row["sample_idx"]), int(row["turn_idx"])


def transition_rows(rows: list[dict[str, Any]], base: str, target: str) -> list[dict[str, Any]]:
    lookup = {model_turn_key(row): row for row in rows}
    out = []
    for (_model, sample_idx, turn_idx), tgt in sorted(lookup.items()):
        if _model != target:
            continue
        src = lookup.get((base, sample_idx, turn_idx))
        if src is None:
            continue
        out.append(
            {
                "base": base,
                "target": target,
                "sample_idx": sample_idx,
                "turn_idx": turn_idx,
                "base_correct": src["correct"],
                "target_correct": tgt["correct"],
                "base_compiled": src["compiled"],
                "target_compiled": tgt["compiled"],
                "base_all_sections": src["all_sections"],
                "target_all_sections": tgt["all_sections"],
                "base_resp_len": src["response_length"],
                "target_resp_len": tgt["response_length"],
                "delta_resp_len": int(tgt["response_length"]) - int(src["response_length"]),
                "target_missing_sections": tgt["missing_sections"],
                "target_first_section_pos": tgt["first_section_pos"],
                "target_starts_narrative": tgt["starts_with_narrative"],
                "target_finish_reason": tgt["finish_reason"],
                "target_kernelgym_status": tgt["kernelgym_status"],
            }
        )
    return out


def summarize_transition(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [int(row["delta_resp_len"]) for row in pairs]
    return {
        "n": len(pairs),
        "base_correct_target_wrong": sum(row["base_correct"] and not row["target_correct"] for row in pairs),
        "base_wrong_target_correct": sum((not row["base_correct"]) and row["target_correct"] for row in pairs),
        "base_compiled_target_not": sum(row["base_compiled"] and not row["target_compiled"] for row in pairs),
        "base_not_compiled_target_compiled": sum(
            (not row["base_compiled"]) and row["target_compiled"] for row in pairs
        ),
        "base_sections_target_missing": sum(
            row["base_all_sections"] and not row["target_all_sections"] for row in pairs
        ),
        "base_missing_target_sections": sum(
            (not row["base_all_sections"]) and row["target_all_sections"] for row in pairs
        ),
        "delta_len_mean": statistics.mean(deltas) if deltas else 0.0,
        "delta_len_p90": percentile(deltas, 0.90),
        "delta_len_p10": percentile(deltas, 0.10),
    }


def error_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        for turn_idx in (0, 1, 2):
            scoped = [
                row for row in rows if row["model"] == model and row["turn_idx"] == turn_idx and not row["correct"]
            ]
            total = len(scoped)
            counts = Counter(str(row.get("error_category") or "unknown") for row in scoped)
            for category, count in counts.most_common():
                out.append(
                    {
                        "model": model,
                        "turn_idx": turn_idx,
                        "category": category,
                        "count": count,
                        "pct_wrong_or_fail": (100.0 * count / total) if total else 0.0,
                    }
                )
    return out


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = DEFAULT_MODELS
    if args.models_json:
        models = {key: Path(value) for key, value in json.loads(args.models_json.read_text()).items()}

    all_rows: list[dict[str, Any]] = []
    for model, path in models.items():
        if not path.exists():
            raise FileNotFoundError(f"{model}: {path}")
        all_rows.extend(load_rows(model, path))

    write_csv(args.output_dir / "turn_structure_features.csv", all_rows)

    summary_rows: list[dict[str, Any]] = []
    for model in models:
        for turn_idx in (0, 1, 2):
            rows = [row for row in all_rows if row["model"] == model and row["turn_idx"] == turn_idx]
            row = {"model": model, "turn_idx": turn_idx}
            row.update(summarize_turn(rows))
            summary_rows.append(row)

    with (args.output_dir / "turn_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    transitions: list[dict[str, Any]] = []
    for base in ("BF16", "W8A8_RTN_NonLA", "W8A8_SQRTN_MLP"):
        transitions.extend(transition_rows(all_rows, base, "W4A16_AWQ_ASYM"))

    with (args.output_dir / "paired_transitions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transitions[0].keys()))
        writer.writeheader()
        writer.writerows(transitions)

    transition_summary = []
    for base in ("BF16", "W8A8_RTN_NonLA", "W8A8_SQRTN_MLP"):
        for turn_idx in (0, 1, 2):
            pairs = [row for row in transitions if row["base"] == base and row["turn_idx"] == turn_idx]
            row = {"base": base, "target": "W4A16_AWQ_ASYM", "turn_idx": turn_idx}
            row.update(summarize_transition(pairs))
            transition_summary.append(row)

    with (args.output_dir / "paired_transition_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transition_summary[0].keys()))
        writer.writeheader()
        writer.writerows(transition_summary)

    error_rows = error_summary_rows(all_rows)
    with (args.output_dir / "error_category_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(error_rows[0].keys()))
        writer.writeheader()
        writer.writerows(error_rows)

    md_lines = ["# W4 AWQ Eval Gap Analysis", ""]
    md_lines.append("## Turn Summary")
    md_lines.append("")
    md_lines.append(
        "| model | turn | correct | compile | fast@1.0 | resp mean/median/p90 | >32k/>60k | all sections | all sections <=4k | no section <=4k | starts narrative |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        md_lines.append(
            "| {model} | T{turn} | {correct} | {compiled} | {fast} | {mean:.1f}/{median:.1f}/{p90:.1f} | {gt32}/{gt60} | {sections} | {sections4k} | {no4k} | {narr} |".format(
                model=row["model"],
                turn=int(row["turn_idx"]) + 1,
                correct=fmt_pct(float(row["correct_pct"])),
                compiled=fmt_pct(float(row["compiled_pct"])),
                fast=fmt_pct(float(row["fast1_pct"])),
                mean=float(row["resp_mean"]),
                median=float(row["resp_median"]),
                p90=float(row["resp_p90"]),
                gt32=int(row["gt_32k"]),
                gt60=int(row["gt_60k"]),
                sections=fmt_pct(float(row["all_sections_pct"])),
                sections4k=fmt_pct(float(row["all_sections_by4k_pct"])),
                no4k=fmt_pct(float(row["no_section_by4k_pct"])),
                narr=fmt_pct(float(row["starts_narrative_pct"])),
            )
        )

    md_lines.extend(["", "## Paired W4 Transitions", ""])
    md_lines.append(
        "| base | turn | base correct -> W4 wrong | W4 wrong -> correct | base compiled -> W4 not | base sections -> W4 missing | delta len mean/p90 |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in transition_summary:
        md_lines.append(
            "| {base} | T{turn} | {bcw} | {wcb} | {bcn} | {bsm} | {dmean:.1f}/{dp90:.1f} |".format(
                base=row["base"],
                turn=int(row["turn_idx"]) + 1,
                bcw=int(row["base_correct_target_wrong"]),
                wcb=int(row["base_wrong_target_correct"]),
                bcn=int(row["base_compiled_target_not"]),
                bsm=int(row["base_sections_target_missing"]),
                dmean=float(row["delta_len_mean"]),
                dp90=float(row["delta_len_p90"]),
            )
        )

    md_lines.extend(["", "## Error Category Summary", ""])
    md_lines.append("| model | turn | top wrong/fail categories |")
    md_lines.append("|---|---:|---|")
    for model in models:
        for turn_idx in (0, 1, 2):
            scoped = [row for row in error_rows if row["model"] == model and row["turn_idx"] == turn_idx]
            top = ", ".join(f"{row['category']}={int(row['count'])}" for row in scoped[:5])
            md_lines.append(f"| {model} | T{turn_idx + 1} | {top} |")

    # Top examples where W8 MLP is correct but W4 is not, to support manual review.
    md_lines.extend(["", "## Examples: W8A8_SQRTN_MLP correct but W4 wrong", ""])
    examples = [
        row
        for row in transitions
        if row["base"] == "W8A8_SQRTN_MLP" and row["base_correct"] and not row["target_correct"]
    ]
    examples.sort(key=lambda row: (row["turn_idx"], -abs(int(row["delta_resp_len"]))))
    for row in examples[:20]:
        md_lines.append(
            "- sample {sample} T{turn}: delta_len={delta}, W4 compiled={compiled}, W4 all_sections={sections}, "
            "missing={missing}, first_section_pos={first}, finish={finish}, kg_status={status}".format(
                sample=row["sample_idx"],
                turn=int(row["turn_idx"]) + 1,
                delta=row["delta_resp_len"],
                compiled=row["target_compiled"],
                sections=row["target_all_sections"],
                missing=row["target_missing_sections"],
                first=row["target_first_section_pos"],
                finish=row["target_finish_reason"],
                status=row["target_kernelgym_status"],
            )
        )

    (args.output_dir / "SUMMARY.md").write_text("\n".join(md_lines) + "\n")
    print(f"[gap] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
