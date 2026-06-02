#!/usr/bin/env python3
"""Compare BF16, per-channel W8A8, and G128 W8A8 eval token lengths.

This is a one-off diagnostic script for the 2026-06-01 G128 rollout audit.
It reads eval_0.pt dumps and writes paired per-sample evidence files.
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


ROOT = Path("checkpoints/Qwen3.6-27B/g128_length_analysis_20260601")
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
    "g128": Path(
        "checkpoints/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/"
        "20260531_150558_w8a8.g128.nonla_mtp.sglcfg.mem82.cp4096.100x8.eagle_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
}


def pct(vals: list[int], q: float) -> float | None:
    if not vals:
        return None
    ordered = sorted(vals)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def extract_messages(prompt: str) -> list[tuple[str, str]]:
    parts = re.split(r"<\|im_start\|>(\w+)\n", prompt or "")
    msgs: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        role = parts[i]
        content = parts[i + 1].split("<|im_end|>")[0]
        msgs.append((role, content))
    return msgs


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


def load_model(name: str, path: Path) -> dict[int, dict[str, Any]]:
    print(f"loading {name}: {path}", flush=True)
    obj = torch.load(path, map_location="cpu")
    out: dict[int, dict[str, Any]] = {}
    for sample in obj["samples"]:
        idx = int(sample["index"])
        spans = extract_spans(sample)
        messages = extract_messages(sample.get("prompt", ""))
        assistant_hist = [content for role, content in messages if role == "assistant"]
        if assistant_hist and len(assistant_hist[-1].strip()) < 32:
            assistant_hist = assistant_hist[:-1]
        metadata = sample.get("metadata") or {}
        response = sample.get("response") or ""
        out[idx] = {
            "index": idx,
            "name": metadata.get("name"),
            "problem_id": metadata.get("problem_id"),
            "reward": sample.get("reward"),
            "status": sample.get("status"),
            "final_response_length": sample.get("response_length"),
            "trace_total_completion": sum((sp.get("completion_tokens") or 0) for sp in spans),
            "spec_completion_total": (sample.get("spec_info") or {}).get("completion_token_num"),
            "spans": spans,
            "finish_reasons": [sp.get("finish_reason") for sp in spans],
            "prompt_chars": len(sample.get("prompt") or ""),
            "response_chars": len(response),
            "history_assistant_chars": [len(x) for x in assistant_hist],
            "response_head": response[:1800],
            "response_tail": response[-1200:],
            "history_heads": [x[:800] for x in assistant_hist[:2]],
            "metadata_raw_problem_head": (metadata.get("raw_problem") or "")[:600],
        }
    return out


def span_tok(sample: dict[str, Any], turn: int) -> int:
    spans = sample["spans"]
    if len(spans) >= turn:
        return int(spans[turn - 1].get("completion_tokens") or 0)
    return 0


def span_finish(sample: dict[str, Any], turn: int) -> str | None:
    spans = sample["spans"]
    if len(spans) >= turn:
        return spans[turn - 1].get("finish_reason")
    return None


def summary_for(name: str, data: dict[int, dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"== {name}")
    lines.append(f"samples {len(data)} total_spans {sum(len(s['spans']) for s in data.values())}")
    total_vals = [s["trace_total_completion"] for s in data.values()]
    lines.append(
        "total_completion "
        f"sum={sum(total_vals)} mean={statistics.mean(total_vals):.2f} "
        f"median={statistics.median(total_vals):.1f} p90={pct(total_vals, .9):.1f} "
        f"p99={pct(total_vals, .99):.1f} max={max(total_vals)}"
    )
    lines.append(
        f"final_reward_sum={sum(1 for s in data.values() if s['reward'] == 1.0)} "
        f"final_zero_resp={sum(1 for s in data.values() if (s['final_response_length'] or 0) == 0)}"
    )
    for turn in (1, 2, 3):
        vals = [span_tok(s, turn) for s in data.values()]
        reasons = Counter(span_finish(s, turn) for s in data.values())
        lines.append(
            f"turn{turn} sum={sum(vals)} mean={statistics.mean(vals):.2f} "
            f"median={statistics.median(vals):.1f} p90={pct(vals, .9):.1f} "
            f"p99={pct(vals, .99):.1f} max={max(vals)} zeros={sum(v == 0 for v in vals)} "
            f">32768={sum(v > 32768 for v in vals)} >60000={sum(v > 60000 for v in vals)} "
            f"finish={dict(reasons)}"
        )
    return "\n".join(lines)


def safe_fence(text: str) -> str:
    return text.replace("```", "` ` `")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    models = {name: load_model(name, path) for name, path in MODELS.items()}
    common = sorted(set.intersection(*(set(v) for v in models.values())))

    summary_lines: list[str] = []
    for name in ("bf16", "per_channel", "g128"):
        summary_lines.append(summary_for(name, models[name]))

    rows: list[dict[str, Any]] = []
    for idx in common:
        row: dict[str, Any] = {
            "index": idx,
            "name": models["g128"][idx]["name"],
            "problem_id": models["g128"][idx]["problem_id"],
        }
        for name in ("bf16", "per_channel", "g128"):
            sample = models[name][idx]
            row[f"{name}_reward"] = sample["reward"]
            row[f"{name}_total"] = sample["trace_total_completion"]
            row[f"{name}_final"] = sample["final_response_length"]
            for turn in (1, 2, 3):
                row[f"{name}_t{turn}"] = span_tok(sample, turn)
                row[f"{name}_finish_t{turn}"] = span_finish(sample, turn)
        row["g128_minus_bf16_total"] = row["g128_total"] - row["bf16_total"]
        row["g128_minus_pc_total"] = row["g128_total"] - row["per_channel_total"]
        for turn in (1, 2, 3):
            row[f"g128_minus_bf16_t{turn}"] = row[f"g128_t{turn}"] - row[f"bf16_t{turn}"]
            row[f"g128_minus_pc_t{turn}"] = row[f"g128_t{turn}"] - row[f"per_channel_t{turn}"]
        rows.append(row)

    with open(ROOT / "paired_turn_lengths.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_lines.append("\n== paired deltas on common samples")
    summary_lines.append(f"common_samples {len(common)}")
    for _base, label, keypart in (
        ("bf16", "BF16", "bf16"),
        ("per_channel", "per-channel", "pc"),
    ):
        vals = [r[f"g128_minus_{keypart}_total"] for r in rows]
        summary_lines.append(
            f"g128 - {label} total: sum={sum(vals)} mean={statistics.mean(vals):.2f} "
            f"median={statistics.median(vals):.1f} p90={pct(vals, .9):.1f} "
            f"p99={pct(vals, .99):.1f} min={min(vals)} max={max(vals)} "
            f"positive={sum(v > 0 for v in vals)} negative={sum(v < 0 for v in vals)}"
        )
        for turn in (1, 2, 3):
            vals = [r[f"g128_minus_{keypart}_t{turn}"] for r in rows]
            summary_lines.append(
                f"  turn{turn}: sum={sum(vals)} mean={statistics.mean(vals):.2f} "
                f"median={statistics.median(vals):.1f} p90={pct(vals, .9):.1f} "
                f"p99={pct(vals, .99):.1f} positive={sum(v > 0 for v in vals)}"
            )

    summary_lines.append("\n== long-tail attribution")
    for threshold in (20000, 32768, 45000, 60000):
        g_t1 = {idx for idx in common if span_tok(models["g128"][idx], 1) > threshold}
        pc_t1 = {idx for idx in common if span_tok(models["per_channel"][idx], 1) > threshold}
        bf_t1 = {idx for idx in common if span_tok(models["bf16"][idx], 1) > threshold}
        summary_lines.append(
            f"turn1 > {threshold}: g128={len(g_t1)} per_channel={len(pc_t1)} "
            f"bf16={len(bf_t1)} g128_only_vs_pc={len(g_t1 - pc_t1)} "
            f"overlap_pc={len(g_t1 & pc_t1)}"
        )

    summary_lines.append("\n== paired delta by final reward category")
    categories = {
        "g128_correct": lambda r: r["g128_reward"] == 1.0,
        "g128_wrong": lambda r: r["g128_reward"] != 1.0,
        "pc_correct_g128_wrong": lambda r: r["per_channel_reward"] == 1.0 and r["g128_reward"] != 1.0,
        "g128_correct_pc_wrong": lambda r: r["g128_reward"] == 1.0 and r["per_channel_reward"] != 1.0,
    }
    for name, pred in categories.items():
        sub = [r for r in rows if pred(r)]
        if not sub:
            continue
        vals = [r["g128_minus_pc_total"] for r in sub]
        t1_vals = [r["g128_minus_pc_t1"] for r in sub]
        summary_lines.append(
            f"{name}: n={len(sub)} total_delta_sum={sum(vals)} "
            f"mean={statistics.mean(vals):.1f} median={statistics.median(vals):.1f}; "
            f"t1_delta_sum={sum(t1_vals)} mean={statistics.mean(t1_vals):.1f} "
            f"median={statistics.median(t1_vals):.1f}"
        )

    (ROOT / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    examples = sorted(
        rows,
        key=lambda r: r["g128_t1"] - max(r["bf16_t1"], r["per_channel_t1"]),
        reverse=True,
    )[:12]
    with open(ROOT / "top_g128_turn1_longer_examples.md", "w") as handle:
        handle.write("# Top G128 Turn-1 Length Blowups\n\n")
        for row in examples:
            idx = row["index"]
            handle.write(f"## sample {idx} problem_id={row['problem_id']} name={row['name']}\n")
            handle.write(
                f"- rewards: bf16={row['bf16_reward']} per_channel={row['per_channel_reward']} "
                f"g128={row['g128_reward']}\n"
            )
            handle.write(
                f"- totals: bf16={row['bf16_total']} per_channel={row['per_channel_total']} "
                f"g128={row['g128_total']}; g128_minus_pc={row['g128_minus_pc_total']}\n"
            )
            handle.write(
                "- t1/t2/t3: "
                f"bf16=({row['bf16_t1']},{row['bf16_t2']},{row['bf16_t3']}) "
                f"per_channel=({row['per_channel_t1']},{row['per_channel_t2']},{row['per_channel_t3']}) "
                f"g128=({row['g128_t1']},{row['g128_t2']},{row['g128_t3']})\n"
            )
            handle.write(
                "- finish: "
                f"bf16=({row['bf16_finish_t1']},{row['bf16_finish_t2']},{row['bf16_finish_t3']}) "
                f"per_channel=({row['per_channel_finish_t1']},{row['per_channel_finish_t2']},{row['per_channel_finish_t3']}) "
                f"g128=({row['g128_finish_t1']},{row['g128_finish_t2']},{row['g128_finish_t3']})\n\n"
            )
            handle.write("### Final response heads\n\n")
            for model_name in ("bf16", "per_channel", "g128"):
                handle.write(f"#### {model_name}\n\n```text\n")
                handle.write(safe_fence(models[model_name][idx]["response_head"]))
                handle.write("\n```\n\n")
            handle.write("### G128 final response tail\n\n```text\n")
            handle.write(safe_fence(models["g128"][idx]["response_tail"]))
            handle.write("\n```\n\n")
            handle.write("### Prior assistant history heads from final prompt\n\n")
            for model_name in ("bf16", "per_channel", "g128"):
                handle.write(f"#### {model_name}\n\n")
                if not models[model_name][idx]["history_heads"]:
                    handle.write("(none)\n\n")
                    continue
                handle.write(f"history assistant chars={models[model_name][idx]['history_assistant_chars']}\n\n")
                for hist_idx, text in enumerate(models[model_name][idx]["history_heads"], 1):
                    handle.write(f"##### prior assistant {hist_idx}\n\n```text\n")
                    handle.write(safe_fence(text))
                    handle.write("\n```\n\n")

    with open(ROOT / "paired_summary.json", "w") as handle:
        json.dump(
            {
                "summary_txt": str(ROOT / "summary.txt"),
                "paired_turn_lengths_csv": str(ROOT / "paired_turn_lengths.csv"),
                "top_examples_md": str(ROOT / "top_g128_turn1_longer_examples.md"),
                "common_samples": len(common),
                "top_examples": examples,
            },
            handle,
            indent=2,
        )

    print((ROOT / "summary.txt").read_text())
    print(f"wrote {ROOT}")


if __name__ == "__main__":
    main()
