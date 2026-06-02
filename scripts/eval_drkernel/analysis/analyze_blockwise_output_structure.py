#!/usr/bin/env python3
"""Analyze why blockwise W8A8 evals produce longer outputs.

This diagnostic compares saved eval_0.pt dumps from BF16, per-channel W8A8,
B128 blockwise W8A8, and B64 blockwise W8A8. It focuses on output structure:
whether responses enter the required CUDA/code format early, whether they start
with natural-language analysis, and which paired samples create the extra
turn-1 long tail.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch


OUT_DIR = Path("checkpoints/Qwen3.6-27B/blockwise_length_logic_20260601")

MODELS = {
    "bf16": Path(
        "checkpoints/Qwen3.6-27B/"
        "20260529_132454_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
    "per_channel": Path(
        "checkpoints/Qwen3.6-27B-W8A8-RTN-nonla-mtp/"
        "20260530_073700_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
    "b128": Path(
        "checkpoints/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/"
        "20260531_150558_w8a8.g128.nonla_mtp.sglcfg.mem82.cp4096.100x8.eagle_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
    "b64": Path(
        "checkpoints/Qwen3.6-27B-W8A8-G64-RTN-nonla-mtp/"
        "20260601_032830_rtn.g64.nonla_mtp.sglcfg64.mem82.cp4096.100x8.eagle.rm16_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
}

NARRATIVE_PATTERNS = [
    "let's",
    "we need",
    "we can",
    "i need",
    "i will",
    "the user",
    "previous",
    "analyze",
    "analysis",
    "wait",
    "looking at",
    "to solve",
    "the code",
    "implementation",
    "error",
    "timeout",
]

CODE_PATTERNS = [
    "### CUDA_KERNELS",
    "```",
    "import torch",
    "from torch",
    "#include",
    "class Model",
    "def forward",
    "torch.utils.cpp_extension",
    "load_inline",
]


def pct(vals: list[int], q: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def pos_or_none(text: str, needle: str) -> int | None:
    pos = text.find(needle)
    return pos if pos >= 0 else None


def min_pos(text: str, needles: list[str]) -> int | None:
    positions = [pos for needle in needles if (pos := pos_or_none(text, needle)) is not None]
    return min(positions) if positions else None


def extract_spans(sample: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for event in sample.get("trace", {}).get("events", []):
        if event.get("name") != "sglang_generate":
            continue
        if event.get("type") == "span_start":
            current = {
                "start_ts": event.get("ts"),
                "max_new_tokens": (event.get("attrs") or {}).get("max_new_tokens"),
            }
        elif event.get("type") == "span_end":
            attrs = event.get("attrs") or {}
            row = dict(current or {})
            row.update(
                {
                    "end_ts": event.get("ts"),
                    "prompt_tokens": attrs.get("prompt_tokens"),
                    "completion_tokens": attrs.get("completion_tokens"),
                    "cached_tokens": attrs.get("cached_tokens"),
                    "finish_reason": attrs.get("finish_reason"),
                }
            )
            if row.get("start_ts") is not None and row.get("end_ts") is not None:
                row["duration_s"] = row["end_ts"] - row["start_ts"]
            spans.append(row)
            current = None
    return spans


def text_features(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    head = stripped[:1000].lower()
    marker_pos = pos_or_none(text, "### CUDA_KERNELS")
    fence_pos = pos_or_none(text, "```")
    code_pos = min_pos(text, ["import torch", "from torch", "#include", "class Model", "def forward"])
    first_format_pos = min_pos(text, CODE_PATTERNS)
    starts_code = bool(
        re.match(
            r"^(### CUDA_KERNELS|```|import\s|from\s|#include|class\s+Model|def\s+forward)",
            stripped,
        )
    )
    narrative_hits = sum(head.count(pattern) for pattern in NARRATIVE_PATTERNS)
    return {
        "char_len": len(text),
        "has_cuda_marker": marker_pos is not None,
        "cuda_marker_pos": marker_pos,
        "cuda_marker_after_30k": marker_pos is None or marker_pos > 30000,
        "cuda_marker_after_60k": marker_pos is None or marker_pos > 60000,
        "has_fence": fence_pos is not None,
        "fence_pos": fence_pos,
        "has_code_like": code_pos is not None,
        "code_like_pos": code_pos,
        "first_format_pos": first_format_pos,
        "entered_format_2k": first_format_pos is not None and first_format_pos <= 2000,
        "entered_format_4k": first_format_pos is not None and first_format_pos <= 4000,
        "no_early_format_4k": first_format_pos is None or first_format_pos > 4000,
        "starts_code": starts_code,
        "starts_narrative": (not starts_code) and narrative_hits > 0,
        "narrative_hits_head": narrative_hits,
        "cuda_marker_count": text.count("### CUDA_KERNELS"),
        "fence_count": text.count("```"),
        "head": stripped[:900],
        "tail": stripped[-700:],
    }


def span_val(spans: list[dict[str, Any]], turn_idx: int, key: str) -> Any:
    if turn_idx < len(spans):
        return spans[turn_idx].get(key)
    return None


def load_model(name: str, path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    print(f"loading {name}: {path}", flush=True)
    payload = torch.load(path, map_location="cpu")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for sample in payload["samples"]:
        sample_idx = int(sample["index"])
        meta = sample.get("metadata") or {}
        spans = extract_spans(sample)
        turns = meta.get("turns") or []
        for turn in turns:
            turn_idx = int(turn.get("turn_idx", len(rows)))
            text = turn.get("response") or ""
            features = text_features(text)
            row = {
                "model": name,
                "sample_idx": sample_idx,
                "turn_idx": turn_idx,
                "name": meta.get("name"),
                "problem_id": meta.get("problem_id"),
                "turn_reward": turn.get("reward"),
                "turn_status": turn.get("status"),
                "turn_response_length": int(turn.get("response_length") or 0),
                "span_completion_tokens": int(span_val(spans, turn_idx, "completion_tokens") or 0),
                "span_finish_reason": span_val(spans, turn_idx, "finish_reason"),
                "span_prompt_tokens": span_val(spans, turn_idx, "prompt_tokens"),
                "span_cached_tokens": span_val(spans, turn_idx, "cached_tokens"),
                "kernelgym_status": (turn.get("kernelgym") or {}).get("status"),
                "kernelgym_compiled": (turn.get("kernelgym") or {}).get("compiled"),
                "kernelgym_correctness": (turn.get("kernelgym") or {}).get("correctness"),
            }
            row.update(features)
            rows[(sample_idx, turn_idx)] = row
    return rows


def bool_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(bool(row.get(key)) for row in rows) / len(rows)


def stat_line(name: str, rows: list[dict[str, Any]]) -> str:
    vals = [int(row["turn_response_length"]) for row in rows]
    finish = Counter(row.get("span_finish_reason") for row in rows)
    format_positions = [int(row["first_format_pos"]) for row in rows if row.get("first_format_pos") is not None]
    marker_positions = [int(row["cuda_marker_pos"]) for row in rows if row.get("cuda_marker_pos") is not None]
    fmt_p50 = statistics.median(format_positions) if format_positions else -1
    fmt_p90 = pct(format_positions, 0.9) if format_positions else -1
    marker_p50 = statistics.median(marker_positions) if marker_positions else -1
    marker_p90 = pct(marker_positions, 0.9) if marker_positions else -1
    return (
        f"{name}: n={len(rows)} sum={sum(vals)} mean={statistics.mean(vals):.1f} "
        f"med={statistics.median(vals):.1f} p90={pct(vals, .9):.1f} "
        f"p99={pct(vals, .99):.1f} max={max(vals) if vals else 0} "
        f">32k={sum(v > 32768 for v in vals)} >60k={sum(v > 60000 for v in vals)} "
        f"finish={dict(finish)} "
        f"starts_narr={bool_rate(rows, 'starts_narrative'):.1f}% "
        f"no_early_fmt4k={bool_rate(rows, 'no_early_format_4k'):.1f}% "
        f"missing_marker={100.0 - bool_rate(rows, 'has_cuda_marker'):.1f}% "
        f"marker_after_30k={bool_rate(rows, 'cuda_marker_after_30k'):.1f}% "
        f"marker_after_60k={bool_rate(rows, 'cuda_marker_after_60k'):.1f}% "
        f"fmt_pos_p50/p90={fmt_p50:.0f}/{fmt_p90:.0f} "
        f"marker_pos_p50/p90={marker_p50:.0f}/{marker_p90:.0f}"
    )


def marker_pos_stats(rows: list[dict[str, Any]]) -> str:
    vals = [int(row["cuda_marker_pos"]) for row in rows if row.get("cuda_marker_pos") is not None]
    if not vals:
        return "marker_pos_p50/p90=-/-"
    return f"marker_pos_p50/p90={statistics.median(vals):.0f}/{pct(vals, .9):.0f}"


def summarize_model(name: str, rows: dict[tuple[int, int], dict[str, Any]]) -> list[str]:
    lines = [f"== {name}"]
    for turn_idx in (0, 1, 2):
        turn_rows = [row for key, row in rows.items() if key[1] == turn_idx]
        long_rows = [row for row in turn_rows if row["turn_response_length"] > 32768]
        very_long_rows = [row for row in turn_rows if row["turn_response_length"] > 60000]
        lines.append(stat_line(f"turn{turn_idx + 1} all", turn_rows))
        lines.append(
            stat_line(f"turn{turn_idx + 1} >32k", long_rows) if long_rows else f"turn{turn_idx + 1} >32k: n=0"
        )
        lines.append(
            stat_line(f"turn{turn_idx + 1} >60k", very_long_rows)
            if very_long_rows
            else f"turn{turn_idx + 1} >60k: n=0"
        )
    return lines


def paired_block(
    rows_by_model: dict[str, dict[tuple[int, int], dict[str, Any]]],
    target: str,
    base: str = "per_channel",
) -> tuple[list[str], list[dict[str, Any]]]:
    lines = [f"== paired {target} vs {base} turn1"]
    examples: list[dict[str, Any]] = []
    target_rows = rows_by_model[target]
    base_rows = rows_by_model[base]
    keys = sorted(set(target_rows) & set(base_rows))
    t1_keys = [key for key in keys if key[1] == 0]
    deltas = [target_rows[key]["turn_response_length"] - base_rows[key]["turn_response_length"] for key in t1_keys]
    lines.append(
        f"common={len(t1_keys)} delta_sum={sum(deltas)} mean={statistics.mean(deltas):.1f} "
        f"median={statistics.median(deltas):.1f} p90={pct(deltas, .9):.1f} "
        f"p99={pct(deltas, .99):.1f} positive={sum(delta > 0 for delta in deltas)}"
    )

    buckets = {
        "target_long_base_not_long": [],
        "both_long": [],
        "base_long_target_not_long": [],
        "neither_long": [],
    }
    for key in t1_keys:
        target_len = target_rows[key]["turn_response_length"]
        base_len = base_rows[key]["turn_response_length"]
        target_long = target_len > 32768
        base_long = base_len > 32768
        if target_long and not base_long:
            bucket = "target_long_base_not_long"
        elif target_long and base_long:
            bucket = "both_long"
        elif base_long and not target_long:
            bucket = "base_long_target_not_long"
        else:
            bucket = "neither_long"
        buckets[bucket].append((key, target_len - base_len))

    for bucket_name, pairs in buckets.items():
        bucket_rows = [target_rows[key] for key, _ in pairs]
        bucket_deltas = [delta for _, delta in pairs]
        if not pairs:
            lines.append(f"{bucket_name}: n=0")
            continue
        lines.append(
            f"{bucket_name}: n={len(pairs)} delta_sum={sum(bucket_deltas)} "
            f"mean_delta={statistics.mean(bucket_deltas):.1f} "
            f"median_delta={statistics.median(bucket_deltas):.1f} "
            f"starts_narr={bool_rate(bucket_rows, 'starts_narrative'):.1f}% "
            f"no_early_fmt4k={bool_rate(bucket_rows, 'no_early_format_4k'):.1f}% "
            f"missing_marker={100.0 - bool_rate(bucket_rows, 'has_cuda_marker'):.1f}% "
            f"marker_after_30k={bool_rate(bucket_rows, 'cuda_marker_after_30k'):.1f}% "
            f"marker_after_60k={bool_rate(bucket_rows, 'cuda_marker_after_60k'):.1f}% "
            f"finish_length={100.0 * sum(row.get('span_finish_reason') == 'length' for row in bucket_rows) / len(bucket_rows):.1f}% "
            f"{marker_pos_stats(bucket_rows)}"
        )

    exclusive = sorted(
        buckets["target_long_base_not_long"],
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    for key, delta in exclusive:
        trow = target_rows[key]
        brow = base_rows[key]
        examples.append(
            {
                "target": target,
                "sample_idx": key[0],
                "problem_id": trow.get("problem_id"),
                "name": trow.get("name"),
                "delta": delta,
                "target_len": trow["turn_response_length"],
                "target_finish": trow.get("span_finish_reason"),
                "target_first_format_pos": trow.get("first_format_pos"),
                "target_cuda_marker_pos": trow.get("cuda_marker_pos"),
                "target_starts_narrative": trow.get("starts_narrative"),
                "target_no_early_format_4k": trow.get("no_early_format_4k"),
                "target_head": trow.get("head"),
                "target_tail": trow.get("tail"),
                "base_len": brow["turn_response_length"],
                "base_finish": brow.get("span_finish_reason"),
                "base_first_format_pos": brow.get("first_format_pos"),
                "base_cuda_marker_pos": brow.get("cuda_marker_pos"),
                "base_starts_narrative": brow.get("starts_narrative"),
                "base_head": brow.get("head"),
            }
        )
    return lines, examples


def safe_fence(text: str | None) -> str:
    return (text or "").replace("```", "` ` `")


def write_examples(examples: list[dict[str, Any]]) -> None:
    lines = ["# Blockwise-exclusive long turn1 examples", ""]
    for ex in examples:
        lines.append(
            f"## {ex['target']} sample {ex['sample_idx']} delta={ex['delta']} "
            f"target_len={ex['target_len']} base_len={ex['base_len']}"
        )
        lines.append(
            f"problem_id={ex.get('problem_id')} name={ex.get('name')} "
            f"target_finish={ex.get('target_finish')} target_fmt_pos={ex.get('target_first_format_pos')} "
            f"target_marker_pos={ex.get('target_cuda_marker_pos')} "
            f"target_starts_narr={ex.get('target_starts_narrative')} "
            f"target_no_early_fmt4k={ex.get('target_no_early_format_4k')}"
        )
        lines.append("")
        lines.append("target head:")
        lines.append("```text")
        lines.append(safe_fence(ex.get("target_head")))
        lines.append("```")
        lines.append("target tail:")
        lines.append("```text")
        lines.append(safe_fence(ex.get("target_tail")))
        lines.append("```")
        lines.append(
            f"per_channel len={ex.get('base_len')} finish={ex.get('base_finish')} "
            f"fmt_pos={ex.get('base_first_format_pos')} marker_pos={ex.get('base_cuda_marker_pos')} "
            f"starts_narr={ex.get('base_starts_narrative')}"
        )
        lines.append("per_channel head:")
        lines.append("```text")
        lines.append(safe_fence(ex.get("base_head")))
        lines.append("```")
        lines.append("")
    (OUT_DIR / "exclusive_long_examples.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_model: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for name, path in MODELS.items():
        rows_by_model[name] = load_model(name, path)

    all_rows = []
    for model_rows in rows_by_model.values():
        all_rows.extend(model_rows.values())
    fieldnames = [
        "model",
        "sample_idx",
        "turn_idx",
        "problem_id",
        "name",
        "turn_response_length",
        "span_completion_tokens",
        "span_finish_reason",
        "turn_reward",
        "kernelgym_status",
        "kernelgym_compiled",
        "kernelgym_correctness",
        "char_len",
        "has_cuda_marker",
        "cuda_marker_pos",
        "cuda_marker_after_30k",
        "cuda_marker_after_60k",
        "has_fence",
        "fence_pos",
        "has_code_like",
        "code_like_pos",
        "first_format_pos",
        "entered_format_2k",
        "entered_format_4k",
        "no_early_format_4k",
        "starts_code",
        "starts_narrative",
        "narrative_hits_head",
        "cuda_marker_count",
        "fence_count",
    ]
    with (OUT_DIR / "turn_structure_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(all_rows, key=lambda r: (r["model"], r["sample_idx"], r["turn_idx"])):
            writer.writerow({key: row.get(key) for key in fieldnames})

    summary_lines = [
        "# Blockwise output-length logic analysis",
        "",
        "Run names b64/b128 refer to SGLang blockwise_int8 weight_block_size=[64,64]/[128,128].",
        "Turn numbering below is 1-based; internal CSV turn_idx is 0-based.",
        "",
    ]
    for name in ("bf16", "per_channel", "b64", "b128"):
        summary_lines.extend(summarize_model(name, rows_by_model[name]))
        summary_lines.append("")

    examples: list[dict[str, Any]] = []
    for target in ("b64", "b128"):
        lines, target_examples = paired_block(rows_by_model, target=target)
        summary_lines.extend(lines)
        summary_lines.append("")
        examples.extend(target_examples)

    # Feature lift: compare structural rates in the blockwise-exclusive long tail
    # against per-channel long and all turn-1 samples.
    summary_lines.append("== feature lift on turn1")
    for target in ("b64", "b128"):
        target_rows = rows_by_model[target]
        base_rows = rows_by_model["per_channel"]
        keys = [key for key in sorted(set(target_rows) & set(base_rows)) if key[1] == 0]
        exclusive_keys = [
            key
            for key in keys
            if target_rows[key]["turn_response_length"] > 32768 and base_rows[key]["turn_response_length"] <= 32768
        ]
        target_exclusive = [target_rows[key] for key in exclusive_keys]
        base_same = [base_rows[key] for key in exclusive_keys]
        target_all = [target_rows[key] for key in keys]
        base_all = [base_rows[key] for key in keys]
        for label, rows in (
            (f"{target} exclusive-long", target_exclusive),
            ("same samples per_channel", base_same),
            (f"{target} all", target_all),
            ("per_channel all", base_all),
        ):
            if not rows:
                continue
            summary_lines.append(
                f"{label}: n={len(rows)} "
                f"mean_len={statistics.mean(row['turn_response_length'] for row in rows):.1f} "
                f"starts_narr={bool_rate(rows, 'starts_narrative'):.1f}% "
                f"no_early_fmt4k={bool_rate(rows, 'no_early_format_4k'):.1f}% "
                f"missing_marker={100.0 - bool_rate(rows, 'has_cuda_marker'):.1f}% "
                f"marker_after_30k={bool_rate(rows, 'cuda_marker_after_30k'):.1f}% "
                f"marker_after_60k={bool_rate(rows, 'cuda_marker_after_60k'):.1f}% "
                f"finish_length={100.0 * sum(row.get('span_finish_reason') == 'length' for row in rows) / len(rows):.1f}% "
                f"{marker_pos_stats(rows)}"
            )

    (OUT_DIR / "structure_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    write_examples(examples)

    manifest = {
        "out_dir": str(OUT_DIR),
        "models": {name: str(path) for name, path in MODELS.items()},
        "files": [
            "structure_summary.txt",
            "turn_structure_features.csv",
            "exclusive_long_examples.md",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
