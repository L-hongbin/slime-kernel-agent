#!/usr/bin/env python3
"""Pair DSpark generation logits with one-pass full-prefix logits.

The server must run ``make_logits_dump_hook`` and the debug
``SGLANG_DEBUG_RETURN_INPUT_FULL_LOGITS=1`` instrumentation.  Each sampled
request is pinned to one DP rank.  The script maps committed verifier rows to
returned tokens, scores the exact completed sequence in one prefill pass, and
compares the full vocabulary distributions at every response position.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch


PROMPTS = [
    "Write a CUDA kernel for a numerically stable row-wise softmax. Explain the memory access pattern and include a test.",
    "Implement an optimized CUDA reduction that sums a large float array. Discuss occupancy, synchronization, and numerical error.",
    "Given an existing PyTorch operation, explain how to replace it with a custom CUDA extension and how to validate correctness.",
]


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise TypeError(f"expected object from {url}, got {type(value).__name__}")
    return value


def flush_cache(base_url: str, timeout: int) -> None:
    for _ in range(40):
        response = requests.post(f"{base_url}/flush_cache", json={}, timeout=timeout)
        if response.ok:
            time.sleep(0.1)
            return
        if response.status_code != 400:
            response.raise_for_status()
        time.sleep(0.25)
    raise RuntimeError("flush_cache did not succeed")


def dump_snapshot(root: Path) -> set[Path]:
    return set(root.glob("pid*/*.pt"))


def new_nonempty_dumps(root: Path, before: set[Path]) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in sorted(set(root.glob("pid*/*.pt")) - before):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        next_logits = payload.get("next_token_logits")
        full_logits = payload.get("full_logits")
        has_next = isinstance(next_logits, torch.Tensor) and next_logits.shape[0] > 0
        has_full = isinstance(full_logits, torch.Tensor) and full_logits.shape[0] > 0
        if has_next or has_full:
            rows.append((path, payload))
    return rows


def parse_output_logprobs(meta: dict[str, Any]) -> tuple[list[int], list[float]]:
    pairs = meta.get("output_token_logprobs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("missing output_token_logprobs")
    return [int(row[1]) for row in pairs], [float(row[0]) for row in pairs]


def request_records(base_url: str, rid: str, timeout: int) -> list[dict[str, Any]]:
    info = requests.get(f"{base_url}/server_info", timeout=timeout).json()
    rows = []
    for state in info.get("internal_states", []):
        dump = state.get("dspark_info_record") or {}
        for record in dump.get("records", []):
            reqs = [req for req in record.get("reqs") or [] if req.get("rid") == rid]
            if reqs:
                if len(reqs) != 1:
                    raise ValueError(f"rid {rid} appears multiple times in one record")
                rows.append({"forward_ct": int(record["forward_ct"]), "req": reqs[0]})
    rows.sort(key=lambda row: row["forward_ct"])
    return rows


def clear_request_records(base_url: str, timeout: int) -> None:
    response = requests.post(
        f"{base_url}/set_internal_state",
        json={"server_args": {"dspark_clear_info_records": True}},
        timeout=timeout,
    )
    response.raise_for_status()


def choose_active_generation_dumps(
    dumps: list[tuple[Path, dict[str, Any]]], verify_width: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    by_pid: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
    for path, payload in dumps:
        by_pid.setdefault(int(payload["pid"]), []).append((path, payload))
    candidates = []
    for pid, rows in by_pid.items():
        rows.sort(key=lambda row: int(row[1]["call_index"]))
        one = [row for row in rows if row[1]["next_token_logits"].shape[0] == 1]
        verify = [row for row in rows if row[1]["next_token_logits"].shape[0] == verify_width]
        if one and verify:
            candidates.append((pid, one[0][1], [row[1] for row in verify], [str(row[0]) for row in rows]))
    if len(candidates) != 1:
        detail = {pid: [tuple(row[1]["next_token_logits"].shape) for row in rows] for pid, rows in by_pid.items()}
        raise ValueError(f"expected one active generation pid, got {detail}")
    _, prefill, verify, paths = candidates[0]
    return prefill, verify, paths


def choose_active_w1_generation_dumps(
    dumps: list[tuple[Path, dict[str, Any]]], token_count: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Select ordinary prefill + one-row decode logits from one active DP rank."""

    by_pid: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
    for path, payload in dumps:
        by_pid.setdefault(int(payload["pid"]), []).append((path, payload))
    candidates = []
    for pid, rows in by_pid.items():
        rows.sort(key=lambda row: int(row[1]["call_index"]))
        one = [row for row in rows if row[1]["next_token_logits"].shape[0] == 1]
        if len(one) >= token_count:
            candidates.append((pid, one[:token_count]))
    if len(candidates) != 1:
        detail = {pid: [tuple(row[1]["next_token_logits"].shape) for row in rows] for pid, rows in by_pid.items()}
        raise ValueError(f"expected one active W1 generation pid, got {detail}")
    _, selected = candidates[0]
    return selected[0][1], [row[1] for row in selected], [str(row[0]) for row in selected]


def generate_with_logits(
    base_url: str,
    hook_dir: Path,
    prompt: str,
    *,
    max_new_tokens: int,
    seed: int,
    dp_rank: int,
    verify_width: int,
    api_logprob_atol: float,
    capture_routed_experts: bool,
    route_num_layers: int,
    route_topk: int,
    timeout: int,
) -> dict[str, Any]:
    # Start generation from a freshly reconstructed prompt.  Otherwise a
    # preceding full-prefix score can leave a longer radix-cache branch whose
    # prompt states were built under a different prefill shape, confounding
    # generation-vs-scorer with cache-construction history.
    flush_cache(base_url, timeout)
    if verify_width > 1:
        clear_request_records(base_url, timeout)
    before = dump_snapshot(hook_dir)
    request_payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "sampling_seed": seed,
            "ignore_eos": True,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "routed_dp_rank": dp_rank,
    }
    if capture_routed_experts:
        request_payload["return_routed_experts"] = True
    result = post_json(
        f"{base_url}/generate",
        request_payload,
        timeout,
    )
    meta = result["meta_info"]
    token_ids, api_logprobs = parse_output_logprobs(meta)
    if len(token_ids) != max_new_tokens:
        raise ValueError(f"expected {max_new_tokens} tokens, got {len(token_ids)}")
    dumps = new_nonempty_dumps(hook_dir, before)
    if verify_width == 1:
        prefill_dump, token_dumps, dump_paths = choose_active_w1_generation_dumps(dumps, max_new_tokens)
        prompt_ids = [int(value) for value in prefill_dump["input_ids"].tolist()]
        logits = [payload["next_token_logits"][0].float() for payload in token_dumps]
        step_index = [-1, *range(max_new_tokens - 1)]
        row_index = [0] * max_new_tokens
        commit_lens = [1] * max_new_tokens
    else:
        prefill_dump, verify_dumps, dump_paths = choose_active_generation_dumps(dumps, verify_width)
        records = request_records(base_url, str(meta["id"]), timeout)
        if len(records) != len(verify_dumps):
            raise ValueError(f"verify record/dump count differs: {len(records)} vs {len(verify_dumps)}")

        prompt_ids = [int(value) for value in prefill_dump["input_ids"].tolist()]
        logits = [prefill_dump["next_token_logits"][0].float()]
        step_index = [-1]
        row_index = [0]
        commit_lens = [1]
        for step, (record, payload) in enumerate(zip(records, verify_dumps, strict=True)):
            commit_len = int(record["req"]["acc_len"])
            matrix = payload["next_token_logits"].float()
            if matrix.shape[0] != verify_width:
                raise ValueError(f"unexpected verify shape {tuple(matrix.shape)}")
            remaining = max_new_tokens - len(logits)
            take = min(commit_len, remaining)
            for row in range(take):
                logits.append(matrix[row])
                step_index.append(step)
                row_index.append(row)
                commit_lens.append(commit_len)
            if len(logits) == max_new_tokens:
                break
    if len(logits) != max_new_tokens:
        raise ValueError(f"mapped {len(logits)} logits for {max_new_tokens} tokens")

    generation_logits = torch.stack(logits)
    gathered = torch.log_softmax(generation_logits, dim=-1)[torch.arange(max_new_tokens), torch.tensor(token_ids)]
    max_error = float((gathered - torch.tensor(api_logprobs)).abs().max())
    if max_error > api_logprob_atol:
        raise ValueError(f"generation hook/API logprob mismatch {max_error}")
    routed_experts = None
    if capture_routed_experts:
        encoded = meta.get("routed_experts")
        if not isinstance(encoded, str):
            raise ValueError("server did not return routed_experts")
        routed_experts = np.frombuffer(base64.b64decode(encoded), dtype=np.int32).copy()
        expected_shape = (len(prompt_ids) + len(token_ids) - 1, route_num_layers, route_topk)
        if routed_experts.size != math.prod(expected_shape):
            raise ValueError(
                f"routed_experts has {routed_experts.size} values, expected "
                f"{math.prod(expected_shape)} for shape {expected_shape}"
            )
        routed_experts = routed_experts.reshape(expected_shape)
    return {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "response_ids": token_ids,
        "api_generation_logprobs": api_logprobs,
        "generation_logits": generation_logits,
        "step_index": step_index,
        "row_index": row_index,
        "commit_lens": commit_lens,
        "rid": str(meta["id"]),
        "hook_paths": dump_paths,
        "hook_api_max_abs_logprob_error": max_error,
        "routed_experts": routed_experts,
    }


def full_prefix_score(
    base_url: str,
    hook_dir: Path,
    all_ids: list[int],
    response_ids: list[int],
    *,
    dp_rank: int,
    api_logprob_atol: float,
    timeout: int,
) -> dict[str, Any]:
    flush_cache(base_url, timeout)
    before = dump_snapshot(hook_dir)
    result = post_json(
        f"{base_url}/generate",
        {
            "input_ids": all_ids,
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0.0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "routed_dp_rank": dp_rank,
        },
        timeout,
    )
    dumps = new_nonempty_dumps(hook_dir, before)
    candidates = [
        (path, payload)
        for path, payload in dumps
        if isinstance(payload.get("full_logits"), torch.Tensor) and payload["full_logits"].shape[0] > 0
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one full-prefix matrix, got {[(str(p), tuple(x['full_logits'].shape)) for p, x in candidates]}"
        )
    path, payload = candidates[0]
    full_logits = payload["full_logits"].float()
    prompt_len = len(all_ids) - len(response_ids)
    start = prompt_len - 1
    scorer_logits = full_logits[start : start + len(response_ids)]
    if scorer_logits.shape[0] != len(response_ids):
        raise ValueError(f"full-prefix slice has {scorer_logits.shape[0]} rows for {len(response_ids)} tokens")

    pairs = result["meta_info"].get("input_token_logprobs")
    if not isinstance(pairs, list) or len(pairs) < len(response_ids):
        raise ValueError("missing full-prefix input_token_logprobs")
    tail = pairs[-len(response_ids) :]
    scored_ids = [int(row[1]) for row in tail]
    if scored_ids != response_ids:
        raise ValueError("full-prefix scorer token IDs differ from response")
    api_logprobs = [float(row[0]) for row in tail]
    gathered = torch.log_softmax(scorer_logits, dim=-1)[torch.arange(len(response_ids)), torch.tensor(response_ids)]
    max_error = float((gathered - torch.tensor(api_logprobs)).abs().max())
    if max_error > api_logprob_atol:
        raise ValueError(f"full-prefix hook/API logprob mismatch {max_error}")
    return {
        "logits": scorer_logits,
        "api_logprobs": api_logprobs,
        "hook_path": str(path),
        "hook_api_max_abs_logprob_error": max_error,
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"p{int(q * 100):02d}": float(np.quantile(values, q))
        for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    }


def signed_summary(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    pos, neg, zero = x[x > 0], x[x < 0], x[x == 0]
    result: dict[str, Any] = {
        "n": int(x.size),
        "positive": int(pos.size),
        "negative": int(neg.size),
        "zero": int(zero.size),
        "positive_fraction": float(pos.size / x.size),
        "negative_fraction": float(neg.size / x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "mean_abs": float(np.abs(x).mean()),
        **quantiles(x),
    }
    if pos.size:
        result.update(
            positive_mean=float(pos.mean()),
            positive_median=float(np.median(pos)),
            positive_sum=float(pos.sum()),
        )
    if neg.size:
        result.update(
            negative_mean=float(neg.mean()),
            negative_median=float(np.median(neg)),
            negative_sum=float(neg.sum()),
        )
    if pos.size and neg.size:
        result["positive_share_abs_mass"] = float(pos.sum() / (pos.sum() - neg.sum()))
    if x.sum() != 0:
        order = np.argsort(np.abs(x))[::-1]
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, math.ceil(fraction * x.size))
            result[f"top_abs_{int(fraction * 100)}pct_signed_sum_share"] = float(x[order[:count]].sum() / x.sum())
    return result


def distribution_row(
    generation_logits: torch.Tensor,
    scorer_logits: torch.Tensor,
    token_id: int,
) -> dict[str, float | int | bool]:
    # Use float64 for the distribution diagnostics.  The API agreement check
    # above intentionally stays in float32 because that is the returned-logprob
    # path, while entropy/KL should not inherit float32 softmax underflow.
    lg = generation_logits.double()
    ls = scorer_logits.double()
    lpg = torch.log_softmax(lg, dim=0)
    lps = torch.log_softmax(ls, dim=0)
    pg, ps = lpg.exp(), lps.exp()
    log_mix = torch.logaddexp(lpg, lps) - math.log(2.0)
    entropy_g = float(-(pg * lpg).sum())
    entropy_s = float(-(ps * lps).sum())
    log_ratio = lpg - lps
    kl_gs = float((pg * log_ratio).sum())
    kl_sg = float((-ps * log_ratio).sum())
    probability_delta = pg - ps
    expected_probability_delta = float((pg * probability_delta).sum())
    js = 0.5 * float((pg * (lpg - log_mix)).sum()) + 0.5 * float((ps * (lps - log_mix)).sum())
    expected_variance = float((pg * (log_ratio - kl_gs).square()).sum())
    expected_probability_delta_variance = float((pg * (probability_delta - expected_probability_delta).square()).sum())
    generation_top = int(torch.argmax(pg))
    scorer_top = int(torch.argmax(ps))
    topk = min(20, pg.numel())
    topk_g = set(torch.topk(pg, topk).indices.tolist())
    topk_s = set(torch.topk(ps, topk).indices.tolist())
    return {
        "sampled_logprob_delta": float(log_ratio[token_id]),
        "sampled_probability_delta": float(probability_delta[token_id]),
        "sampled_probability_ratio": float(torch.exp(log_ratio[token_id])),
        "sampled_rank_generation": int((pg > pg[token_id]).sum()) + 1,
        "sampled_rank_scorer": int((ps > ps[token_id]).sum()) + 1,
        "entropy_generation": entropy_g,
        "entropy_scorer": entropy_s,
        "entropy_delta": entropy_g - entropy_s,
        "kl_generation_scorer": kl_gs,
        "kl_scorer_generation": kl_sg,
        "conditional_logratio_variance": expected_variance,
        "expected_sampled_probability_delta": expected_probability_delta,
        "conditional_probability_delta_variance": expected_probability_delta_variance,
        "js": js,
        "tv": float(0.5 * (pg - ps).abs().sum()),
        "top1_probability_generation": float(pg[generation_top]),
        "top1_probability_scorer": float(ps[scorer_top]),
        "top1_probability_delta": float(pg[generation_top] - ps[scorer_top]),
        "top1_same": generation_top == scorer_top,
        "top20_jaccard": len(topk_g & topk_s) / len(topk_g | topk_s),
        "centered_logit_std_generation": float((lg - lg.mean()).std(unbiased=False)),
        "centered_logit_std_scorer": float((ls - ls.mean()).std(unbiased=False)),
    }


def correlation(x: list[float], y: list[float]) -> float:
    a, b = np.asarray(x), np.asarray(y)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for record in records for row in record["rows"]]
    dlog = [float(row["sampled_logprob_delta"]) for row in rows]
    dp = [float(row["sampled_probability_delta"]) for row in rows]
    api_dlog = [float(row["api_sampled_logprob_delta"]) for row in rows if "api_sampled_logprob_delta" in row]
    api_dp = [float(row["api_sampled_probability_delta"]) for row in rows if "api_sampled_probability_delta" in row]
    dh = [float(row["entropy_delta"]) for row in rows]
    kl = [float(row["kl_generation_scorer"]) for row in rows]
    conditional_var = [float(row["conditional_logratio_variance"]) for row in rows]
    expected_probability_delta = [float(row["expected_sampled_probability_delta"]) for row in rows]
    conditional_probability_var = [float(row["conditional_probability_delta_variance"]) for row in rows]
    n = len(rows)
    expected_mean = statistics.fmean(kl)
    observed_mean = statistics.fmean(dlog)
    conditional_se = math.sqrt(sum(conditional_var)) / n
    observed_probability_mean = statistics.fmean(dp)
    expected_probability_mean = statistics.fmean(expected_probability_delta)
    conditional_probability_se = math.sqrt(sum(conditional_probability_var)) / n
    result = {
        "sampled_logprob_delta": signed_summary(dlog),
        "sampled_probability_delta": signed_summary(dp),
        "entropy_delta": signed_summary(dh),
        "kl_generation_scorer": signed_summary(kl),
        "js": signed_summary([float(row["js"]) for row in rows]),
        "tv": signed_summary([float(row["tv"]) for row in rows]),
        "top1_probability_delta": signed_summary([float(row["top1_probability_delta"]) for row in rows]),
        "centered_logit_std_delta": signed_summary(
            [float(row["centered_logit_std_generation"]) - float(row["centered_logit_std_scorer"]) for row in rows]
        ),
        "selection_identity": {
            "logprob": {
                "observed_sampled_mean": observed_mean,
                "full_vocab_expected_mean": expected_mean,
                "observed_minus_expected": observed_mean - expected_mean,
                "conditional_standard_error": conditional_se,
                "z_score": ((observed_mean - expected_mean) / conditional_se if conditional_se > 0 else None),
            },
            "raw_probability": {
                "observed_sampled_mean": observed_probability_mean,
                "full_vocab_expected_mean": expected_probability_mean,
                "observed_minus_expected": observed_probability_mean - expected_probability_mean,
                "conditional_standard_error": conditional_probability_se,
                "z_score": (
                    (observed_probability_mean - expected_probability_mean) / conditional_probability_se
                    if conditional_probability_se > 0
                    else None
                ),
            },
        },
        "correlations": {
            "sampled_logprob_vs_entropy_delta": correlation(dlog, dh),
            "sampled_logprob_vs_kl": correlation(dlog, kl),
            "entropy_delta_vs_centered_logit_std_delta": correlation(
                dh,
                [
                    float(row["centered_logit_std_generation"]) - float(row["centered_logit_std_scorer"])
                    for row in rows
                ],
            ),
        },
        "top1_same_fraction": statistics.fmean(float(row["top1_same"]) for row in rows),
        "top20_jaccard_mean": statistics.fmean(float(row["top20_jaccard"]) for row in rows),
        "per_prompt": [
            {
                "prompt_index": record["prompt_index"],
                "sampled_logprob_delta": signed_summary(
                    [float(row["sampled_logprob_delta"]) for row in record["rows"]]
                ),
                "entropy_delta": signed_summary([float(row["entropy_delta"]) for row in record["rows"]]),
                "kl_generation_scorer_mean": statistics.fmean(
                    float(row["kl_generation_scorer"]) for row in record["rows"]
                ),
            }
            for record in records
        ],
    }
    if len(api_dlog) == len(rows) and len(api_dp) == len(rows):
        result["api_sampled_logprob_delta"] = signed_summary(api_dlog)
        result["api_sampled_probability_delta"] = signed_summary(api_dp)
    return result


def scorer_repeat_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [row for record in records for row in record.get("repeat_rows", [])]
    if not rows:
        return None
    return {
        "entropy_repeat_delta": signed_summary([float(row["entropy_repeat_delta"]) for row in rows]),
        "js_repeat": signed_summary([float(row["js_repeat"]) for row in rows]),
        "tv_repeat": signed_summary([float(row["tv_repeat"]) for row in rows]),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31003")
    parser.add_argument("--hook-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-pt", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument("--scorer-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--verify-width", type=int, default=5)
    parser.add_argument("--api-logprob-atol", type=float, default=2e-5)
    parser.add_argument("--capture-routed-experts", action="store_true")
    parser.add_argument("--route-num-layers", type=int, default=43)
    parser.add_argument("--route-topk", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if not args.hook_dir.is_dir():
        raise FileNotFoundError(args.hook_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)

    raw_records = []
    json_records = []
    for prompt_index, prompt in enumerate(PROMPTS[: args.prompt_count]):
        generated = generate_with_logits(
            args.base_url,
            args.hook_dir,
            prompt,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + prompt_index,
            dp_rank=args.dp_rank,
            verify_width=args.verify_width,
            api_logprob_atol=args.api_logprob_atol,
            capture_routed_experts=args.capture_routed_experts,
            route_num_layers=args.route_num_layers,
            route_topk=args.route_topk,
            timeout=args.timeout,
        )
        all_ids = generated["prompt_ids"] + generated["response_ids"]
        scores = [
            full_prefix_score(
                args.base_url,
                args.hook_dir,
                all_ids,
                generated["response_ids"],
                dp_rank=args.dp_rank,
                api_logprob_atol=args.api_logprob_atol,
                timeout=args.timeout,
            )
            for _ in range(args.scorer_repeats)
        ]

        rows = []
        for position, token_id in enumerate(generated["response_ids"]):
            api_generation_logprob = float(generated["api_generation_logprobs"][position])
            api_scorer_logprob = float(scores[0]["api_logprobs"][position])
            row = {
                "prompt_index": prompt_index,
                "position": position,
                "token_id": token_id,
                "step_index": generated["step_index"][position],
                "verify_row": generated["row_index"][position],
                "commit_len": generated["commit_lens"][position],
                "api_generation_logprob": api_generation_logprob,
                "api_scorer_logprob": api_scorer_logprob,
                "api_sampled_logprob_delta": api_generation_logprob - api_scorer_logprob,
                "api_sampled_probability_delta": math.exp(api_generation_logprob) - math.exp(api_scorer_logprob),
                **distribution_row(
                    generated["generation_logits"][position],
                    scores[0]["logits"][position],
                    token_id,
                ),
            }
            rows.append(row)

        repeat_rows = []
        if len(scores) >= 2:
            for position in range(args.max_new_tokens):
                repeat = distribution_row(
                    scores[0]["logits"][position],
                    scores[1]["logits"][position],
                    generated["response_ids"][position],
                )
                repeat_rows.append(
                    {
                        "position": position,
                        "entropy_repeat_delta": repeat["entropy_delta"],
                        "js_repeat": repeat["js"],
                        "tv_repeat": repeat["tv"],
                    }
                )

        raw_record = {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "prompt_ids": generated["prompt_ids"],
            "response_ids": generated["response_ids"],
            "generation_logits": generated["generation_logits"],
            "scorer_logits": torch.stack([score["logits"] for score in scores]),
        }
        if generated["routed_experts"] is not None:
            raw_record["routed_experts"] = torch.from_numpy(generated["routed_experts"])
        raw_records.append(raw_record)
        json_records.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "prompt_length": len(generated["prompt_ids"]),
                "response_length": len(generated["response_ids"]),
                "rid": generated["rid"],
                "generation_hook_paths": generated["hook_paths"],
                "generation_hook_api_max_abs_logprob_error": generated["hook_api_max_abs_logprob_error"],
                "scorer_hook_paths": [score["hook_path"] for score in scores],
                "scorer_hook_api_max_abs_logprob_error": [score["hook_api_max_abs_logprob_error"] for score in scores],
                "rows": rows,
                "repeat_rows": repeat_rows,
                "routed_experts_shape": (
                    list(generated["routed_experts"].shape) if generated["routed_experts"] is not None else None
                ),
            }
        )
        print(f"completed prompt {prompt_index}: {len(rows)} paired distributions", flush=True)

    torch.save({"records": raw_records}, args.output_pt)
    evidence = {
        "base_url": args.base_url,
        "hook_dir": str(args.hook_dir),
        "max_new_tokens": args.max_new_tokens,
        "prompt_count": len(json_records),
        "scorer_repeats": args.scorer_repeats,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "dp_rank": args.dp_rank,
        "verify_width": args.verify_width,
        "raw_pt": str(args.output_pt),
        "raw_pt_sha256": sha256(args.output_pt),
        "summary": analyze(json_records),
        "scorer_repeatability": scorer_repeat_summary(json_records),
        "records": json_records,
    }
    args.output_json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
