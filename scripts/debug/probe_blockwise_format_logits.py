#!/usr/bin/env python3
"""Probe token-level logits for blockwise-exclusive long-output prompts.

This is a small SGLang Engine diagnostic for the W8A8-B64/B128 length audit.
It selects first-turn prompts where a blockwise checkpoint is long while the
per-channel checkpoint is not, then compares next-token logprobs for format and
narrative marker tokens.

Run this on the SGLang node, for example:

    python3 scripts/debug/probe_blockwise_format_logits.py --models per_channel,b128 --samples-per-target 4
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path("checkpoints/Qwen3.6-27B/blockwise_length_logic_20260601")
FEATURE_CSV = ROOT / "turn_structure_features.csv"

EVAL_PTS = {
    "per_channel": Path(
        "checkpoints/Qwen3.6-27B-W8A8-RTN-nonla-mtp/"
        "20260530_073700_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
    "b64": Path(
        "checkpoints/Qwen3.6-27B-W8A8-G64-RTN-nonla-mtp/"
        "20260601_032830_rtn.g64.nonla_mtp.sglcfg64.mem82.cp4096.100x8.eagle.rm16_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
    "b128": Path(
        "checkpoints/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/"
        "20260531_150558_w8a8.g128.nonla_mtp.sglcfg.mem82.cp4096.100x8.eagle_ctx65536_n8_summ1600/"
        "dumps/rollout_data/eval_0.pt"
    ),
}

MODEL_PATHS = {
    "bf16": Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B"),
    "per_channel": Path("checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-RTN-nonla-mtp"),
    "b64": Path("checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-G64-RTN-nonla-mtp"),
    "b128": Path("checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp"),
}

PROBE_STRINGS = [
    "###",
    "\n###",
    "```",
    "\n```",
    "The",
    "We",
    "I",
    "Here",
    "Let's",
    "To",
    "Wait",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="per_channel,b64,b128", help="Comma-separated model keys")
    parser.add_argument("--samples-per-target", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--output-json", type=Path, default=ROOT / "format_logprob_probe.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "format_logprob_probe.md")
    parser.add_argument("--include-bf16", action="store_true")
    return parser.parse_args()


def read_feature_rows() -> dict[tuple[str, int, int], dict[str, str]]:
    rows: dict[tuple[str, int, int], dict[str, str]] = {}
    with FEATURE_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[(row["model"], int(row["sample_idx"]), int(row["turn_idx"]))] = row
    return rows


def select_samples(rows: dict[tuple[str, int, int], dict[str, str]], samples_per_target: int) -> list[int]:
    selected: list[int] = []
    base = "per_channel"
    for target in ("b64", "b128"):
        pairs: list[tuple[int, int]] = []
        for (model, sample_idx, turn_idx), row in rows.items():
            if model != target or turn_idx != 0:
                continue
            base_row = rows.get((base, sample_idx, 0))
            if base_row is None:
                continue
            target_len = int(row["turn_response_length"])
            base_len = int(base_row["turn_response_length"])
            if target_len > 32768 and base_len <= 32768:
                pairs.append((target_len - base_len, sample_idx))
        for _, sample_idx in sorted(pairs, reverse=True)[:samples_per_target]:
            if sample_idx not in selected:
                selected.append(sample_idx)
    return selected


def load_prompts(sample_ids: list[int]) -> dict[int, dict[str, Any]]:
    payload = torch.load(EVAL_PTS["per_channel"], map_location="cpu")
    wanted = set(sample_ids)
    out: dict[int, dict[str, Any]] = {}
    for sample in payload["samples"]:
        sample_idx = int(sample["index"])
        if sample_idx not in wanted:
            continue
        meta = sample.get("metadata") or {}
        turns = meta.get("turns") or []
        if not turns:
            continue
        out[sample_idx] = {
            "prompt": turns[0].get("prompt_snapshot") or sample.get("prompt") or "",
            "name": meta.get("name"),
            "problem_id": meta.get("problem_id"),
        }
    missing = sorted(wanted - set(out))
    if missing:
        raise RuntimeError(f"missing prompts for samples: {missing}")
    return out


def first_token_ids_from_eval(tokenizer: Any, sample_ids: list[int]) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for model, path in EVAL_PTS.items():
        payload = torch.load(path, map_location="cpu")
        by_sample: dict[int, int] = {}
        for sample in payload["samples"]:
            sample_idx = int(sample["index"])
            if sample_idx not in sample_ids:
                continue
            turns = (sample.get("metadata") or {}).get("turns") or []
            text = (turns[0].get("response") if turns else sample.get("response")) or ""
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                by_sample[sample_idx] = int(ids[0])
        result[model] = by_sample
    return result


def build_probe_ids(tokenizer: Any, sample_ids: list[int]) -> tuple[list[int], dict[int, str]]:
    label_by_id: dict[int, str] = {}
    for text in PROBE_STRINGS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            label_by_id.setdefault(int(ids[0]), repr(text))
    if tokenizer.eos_token_id is not None:
        label_by_id.setdefault(int(tokenizer.eos_token_id), "eos")

    for model, sample_to_id in first_token_ids_from_eval(tokenizer, sample_ids).items():
        for _sample_idx, token_id in sample_to_id.items():
            decoded = tokenizer.decode([token_id])
            label_by_id.setdefault(int(token_id), f"{model}_actual_first:{decoded!r}")

    return sorted(label_by_id), label_by_id


def logprob_to_prob(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return 0.0
    return float(math.exp(value))


def normalize_token_id_logprobs(raw: Any, probe_ids: list[int]) -> dict[int, float]:
    if raw is None:
        return {}
    # SGLang returns one block per generated token. Each block can be either a
    # list aligned to token_ids_logprob or a list of (logprob, token_id, text).
    block = raw[0] if raw and isinstance(raw, list) else raw
    out: dict[int, float] = {}
    if isinstance(block, list):
        if block and isinstance(block[0], (int, float)):
            for token_id, value in zip(probe_ids, block, strict=False):
                out[int(token_id)] = float(value)
        else:
            for item in block:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out[int(item[1])] = float(item[0])
    return out


def run_model(
    model_key: str,
    model_path: Path,
    prompts: dict[int, dict[str, Any]],
    probe_ids: list[int],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    import sglang as sgl

    print(f"[probe] loading {model_key}: {model_path}", flush=True)
    engine = sgl.Engine(
        model_path=str(model_path),
        tp_size=ARGS.tp_size,
        context_length=ARGS.context_length,
        mem_fraction_static=ARGS.mem_fraction_static,
        trust_remote_code=True,
        log_level="warning",
        skip_server_warmup=True,
        disable_cuda_graph=True,
    )
    try:
        prompt_items = sorted(prompts.items())
        outputs = engine.generate(
            prompt=[item["prompt"] for _, item in prompt_items],
            sampling_params={
                "temperature": ARGS.temperature,
                "top_p": ARGS.top_p,
                "top_k": ARGS.top_k,
                "max_new_tokens": 1,
            },
            return_logprob=True,
            top_logprobs_num=20,
            token_ids_logprob=probe_ids,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        rows: list[dict[str, Any]] = []
        for (sample_idx, item), output in zip(prompt_items, outputs, strict=False):
            meta = output.get("meta_info") or {}
            out_token_logprobs = meta.get("output_token_logprobs") or []
            first = out_token_logprobs[0] if out_token_logprobs else None
            first_logprob = first[0] if isinstance(first, (list, tuple)) and first else None
            first_token_id = first[1] if isinstance(first, (list, tuple)) and len(first) > 1 else None
            token_id_lps = normalize_token_id_logprobs(meta.get("output_token_ids_logprobs"), probe_ids)
            top_lps = meta.get("output_top_logprobs") or []
            top_block = top_lps[0] if top_lps else []
            top1 = top_block[0] if top_block and isinstance(top_block[0], (list, tuple)) else None
            top1_logprob = top1[0] if top1 is not None and len(top1) > 0 else None
            top1_token_id = top1[1] if top1 is not None and len(top1) > 1 else None
            rows.append(
                {
                    "model": model_key,
                    "sample_idx": sample_idx,
                    "name": item.get("name"),
                    "problem_id": item.get("problem_id"),
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "first_token_id": first_token_id,
                    "first_token": tokenizer.decode([first_token_id]) if first_token_id is not None else None,
                    "first_logprob": first_logprob,
                    "first_prob": logprob_to_prob(first_logprob),
                    "top1_token_id": top1_token_id,
                    "top1_token": tokenizer.decode([top1_token_id]) if top1_token_id is not None else None,
                    "top1_logprob": top1_logprob,
                    "top1_prob": logprob_to_prob(top1_logprob),
                    "probe_logprobs": {str(k): v for k, v in token_id_lps.items()},
                    "probe_probs": {str(k): logprob_to_prob(v) for k, v in token_id_lps.items()},
                    "top_logprobs": top_block,
                }
            )
        return rows
    finally:
        print(f"[probe] shutting down {model_key}", flush=True)
        engine.shutdown()
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_markdown(rows: list[dict[str, Any]], label_by_id: dict[int, str], path: Path) -> None:
    lines = ["# Blockwise Format Logprob Probe", ""]
    lines.append(
        "sampling: " f"temperature={ARGS.temperature}, top_p={ARGS.top_p}, top_k={ARGS.top_k}, max_new_tokens=1"
    )
    lines.append("")
    lines.append("Probe token ids:")
    for token_id, label in sorted(label_by_id.items()):
        lines.append(f"- `{token_id}`: {label}")
    lines.append("")
    lines.append(
        "| sample | model | sampled token | p(sampled) | top1 token | p(top1) | "
        "p(###) | p(\\n###) | p(The) | p(We) | p(Here) |"
    )
    lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|")

    def prob_for(row: dict[str, Any], label: str) -> float | None:
        for token_id, token_label in label_by_id.items():
            if token_label == label:
                value = row.get("probe_probs", {}).get(str(token_id))
                return value
        return None

    for row in rows:
        vals = [
            row["sample_idx"],
            row["model"],
            repr(row.get("first_token")),
            f"{row.get('first_prob'):.3e}" if row.get("first_prob") is not None else "-",
            repr(row.get("top1_token")),
            f"{row.get('top1_prob'):.3e}" if row.get("top1_prob") is not None else "-",
            f"{(prob_for(row, repr('###')) or 0):.3e}",
            f"{(prob_for(row, repr(chr(10) + '###')) or 0):.3e}",
            f"{(prob_for(row, repr('The')) or 0):.3e}",
            f"{(prob_for(row, repr('We')) or 0):.3e}",
            f"{(prob_for(row, repr('Here')) or 0):.3e}",
        ]
        lines.append("| " + " | ".join(map(str, vals)) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global ARGS
    ARGS = parse_args()
    model_keys = [x.strip() for x in ARGS.models.split(",") if x.strip()]
    if ARGS.include_bf16 and "bf16" not in model_keys:
        model_keys.insert(0, "bf16")
    unknown = sorted(set(model_keys) - set(MODEL_PATHS))
    if unknown:
        raise ValueError(f"unknown model keys: {unknown}")

    ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_feature_rows()
    sample_ids = select_samples(rows, ARGS.samples_per_target)
    prompts = load_prompts(sample_ids)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["bf16"], trust_remote_code=True)
    probe_ids, label_by_id = build_probe_ids(tokenizer, sample_ids)

    all_rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        all_rows.extend(run_model(model_key, MODEL_PATHS[model_key], prompts, probe_ids, tokenizer))

    output = {
        "sample_ids": sample_ids,
        "model_keys": model_keys,
        "sampling_params": {
            "temperature": ARGS.temperature,
            "top_p": ARGS.top_p,
            "top_k": ARGS.top_k,
            "max_new_tokens": 1,
        },
        "probe_ids": probe_ids,
        "label_by_id": {str(k): v for k, v in label_by_id.items()},
        "rows": all_rows,
    }
    ARGS.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(all_rows, label_by_id, ARGS.output_md)
    print(f"[probe] wrote {ARGS.output_json} and {ARGS.output_md}", flush=True)


if __name__ == "__main__":
    ARGS: argparse.Namespace
    main()
