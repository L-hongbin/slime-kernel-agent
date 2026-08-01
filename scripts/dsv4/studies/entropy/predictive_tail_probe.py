#!/usr/bin/env python3
"""Fixed-context decomposition of predictive-DPPO's Top-K KL tail.

The probe has five deliberately separate stages:

1. ``prepare-prompts`` selects real DrKernel prompts and renders the exact chat
   template used by the formal launcher.
2. ``generate`` samples fixed context sets from a named SGLang LoRA adapter.
3. ``capture-generation`` preserves the online DSPARK denominator, while
   ``rescore`` evaluates either context set under either SGLang adapter using
   a full-prefix forward and captures Top-K support plus rollout expert IDs.
4. ``train-score`` replays those expert IDs in a frozen Megatron forward with
   the matching adapter and computes the exact predictive-DPPO coarse KL.
5. ``analyze-grid`` reports paired prompt-cluster estimates and keeps the
   generation-versus-rescore control next to the 2x2 result.

Running the 2x2 grid (iter74/iter94 policy state x iter74/iter94 generated
contexts) separates a policy-state-dependent cross-engine effect from a token
and context-distribution effect only when the diagonal generation/rescore
controls pass.  Direct generation diagonals retain the actual online endpoint
when those controls fail.  No optimizer step is performed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault(
    "TILELANG_CACHE_DIR",
    f"/tmp/tilelang_tail_probe_rank{os.environ.get('LOCAL_RANK', '0')}",
)

import numpy as np
import requests
import torch

FORMAT_PROMPTS = "predictive-tail-prompts-v1"
FORMAT_CONTEXTS = "predictive-tail-contexts-v1"
FORMAT_CAPTURE = "predictive-tail-capture-v1"
FORMAT_SCORE = "predictive-tail-train-score-v1"


# The relative checkpoint comparison must not silently fall back to the
# standalone probe's BF16 defaults.  These values reproduce the trainer-side
# arithmetic switches printed by the active r21 launcher.  CP remains 1 in
# this eight-GPU diagnostic (the formal run is CP2/EP8 on 16 GPUs), so the
# absolute endpoint is production-like rather than production-identical.
DSV4_TRAIN_ENV = {
    "V4_FP4_FROZEN_EXPERTS": "1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a dict, got {type(value).__name__}")
    return value


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise TypeError(f"{url} returned {type(value).__name__}, expected dict")
    return value


def flush_cache(base_url: str, timeout: int) -> None:
    for _ in range(80):
        response = requests.post(f"{base_url}/flush_cache", json={}, timeout=timeout)
        if response.ok:
            return
        if response.status_code != 400:
            response.raise_for_status()
        time.sleep(0.25)
    raise RuntimeError("SGLang flush_cache did not succeed")


def parse_logprob_item(item: Any, location: str) -> tuple[float, int]:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        raise ValueError(f"{location} is not a (logprob, token_id, ...) tuple: {item!r}")
    log_prob, token_id = item[:2]
    if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
        raise ValueError(f"{location} has invalid token id {token_id!r}")
    log_prob = float(log_prob)
    if not math.isfinite(log_prob):
        raise ValueError(f"{location} has non-finite logprob {log_prob}")
    return log_prob, int(token_id)


def support_row(
    sampled_item: Any,
    top_row: Any,
    *,
    sampled_token_id: int,
    top_k: int,
    location: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    sampled_log_prob, returned_token_id = parse_logprob_item(sampled_item, location + ".sampled")
    if returned_token_id != sampled_token_id:
        raise ValueError(f"{location} sampled token mismatch: response={sampled_token_id} api={returned_token_id}")
    if not isinstance(top_row, (list, tuple)) or len(top_row) != top_k:
        actual = len(top_row) if isinstance(top_row, (list, tuple)) else type(top_row).__name__
        raise ValueError(f"{location} top-k width is {actual}, expected {top_k}")

    ids = np.full(top_k + 1, -1, dtype=np.int32)
    log_probs = np.zeros(top_k + 1, dtype=np.float32)
    valid = np.zeros(top_k + 1, dtype=np.bool_)
    slot_by_id: dict[int, int] = {}
    for slot, item in enumerate(top_row):
        log_prob, token_id = parse_logprob_item(item, f"{location}.top[{slot}]")
        if token_id in slot_by_id:
            raise ValueError(f"{location} contains duplicate top-k token {token_id}")
        slot_by_id[token_id] = slot
        ids[slot] = token_id
        log_probs[slot] = log_prob
        valid[slot] = True
    sampled_slot = slot_by_id.get(sampled_token_id, top_k)
    ids[sampled_slot] = sampled_token_id
    log_probs[sampled_slot] = sampled_log_prob
    valid[sampled_slot] = True
    return sampled_log_prob, ids, log_probs, valid


def decode_routes(meta: dict[str, Any], *, token_count: int, num_layers: int, routing_top_k: int) -> torch.Tensor:
    encoded = meta.get("routed_experts")
    if not isinstance(encoded, str):
        raise ValueError("SGLang response is missing routed_experts")
    values = np.frombuffer(base64.b64decode(encoded), dtype=np.int32).copy()
    expected = token_count * num_layers * routing_top_k
    if values.size != expected:
        raise ValueError(
            f"routed_experts has {values.size} values, expected {expected} "
            f"for ({token_count}, {num_layers}, {routing_top_k})"
        )
    return torch.from_numpy(values.reshape(token_count, num_layers, routing_top_k))


def write_torch(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha256(output)}))


def configure_train_environment(profile: str) -> dict[str, str]:
    if profile != "r21":
        raise ValueError(f"unsupported train environment profile {profile!r}")
    os.environ.update(DSV4_TRAIN_ENV)
    return {key: os.environ[key] for key in DSV4_TRAIN_ENV}


def review_preview(content: str, width: int = 1200) -> str:
    """Show the task body instead of only the dataset's shared preamble."""

    marker = content.rfind("class Model(")
    if marker < 0:
        marker = content.find("def forward")
    if marker < 0:
        return content[:width]
    start = max(0, marker - 120)
    return content[start : start + width]


def prepare_prompts(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    from slime.utils.processing_utils import load_tokenizer

    table = pq.read_table(args.parquet, columns=[args.prompt_key, args.metadata_key])
    tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    rng = random.Random(args.seed)
    indices = list(range(table.num_rows))
    rng.shuffle(indices)
    records: list[dict[str, Any]] = []
    for row_index in indices:
        messages = table[args.prompt_key][row_index].as_py()
        metadata = table[args.metadata_key][row_index].as_py()
        # Match examples/kernel_agent/generate_with_cuda_agent.py exactly: the
        # V4 tokenizer's chat-template call renders text even when tokenize=True,
        # and the formal rollout explicitly performs these two operations.
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(input_ids) > args.max_prompt_tokens:
            continue
        content = "\n".join(str(item.get("content", "")) for item in messages)
        records.append(
            {
                "row_index": row_index,
                "level": str(metadata.get("level", "")),
                "ops": str(metadata.get("ops", "")),
                "input_ids": torch.tensor(input_ids, dtype=torch.int32),
                "prompt_tokens": len(input_ids),
                "preview": review_preview(content),
            }
        )
        if len(records) == args.count:
            break
    if len(records) != args.count:
        raise RuntimeError(f"found only {len(records)} eligible prompts, expected {args.count}")
    payload = {
        "format": FORMAT_PROMPTS,
        "parquet": str(args.parquet),
        "model": str(args.model),
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "records": records,
    }
    write_torch(payload, args.output)
    review = args.output.with_suffix(".txt")
    lines = [
        f"prompt_artifact={args.output}",
        f"prompt_sha256={sha256(args.output)}",
        f"selection_seed={args.seed}",
        "",
    ]
    for slot, record in enumerate(records):
        lines.extend(
            [
                f"[{slot}] row={record['row_index']} level={record['level']} "
                f"tokens={record['prompt_tokens']} ops={record['ops']}",
                record["preview"].replace("\n", " "),
                "",
            ]
        )
    review.write_text("\n".join(lines) + "\n")
    print(f"review={review}")


def generate_contexts(args: argparse.Namespace) -> None:
    prompts = load_torch(args.prompts)
    if prompts.get("format") != FORMAT_PROMPTS:
        raise ValueError(f"unexpected prompt format {prompts.get('format')!r}")
    base_url = args.base_url.rstrip("/")
    records: list[dict[str, Any]] = []
    for slot, prompt in enumerate(prompts["records"]):
        flush_cache(base_url, args.timeout)
        prompt_ids = prompt["input_ids"].long().tolist()
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "sampling_seed": args.seed + slot,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "top_logprobs_num": args.top_k,
            "return_routed_experts": True,
            "lora_path": args.adapter_name,
            "routed_dp_rank": slot % args.dp_size,
        }
        response = post_json(f"{base_url}/generate", payload, args.timeout)
        meta = response["meta_info"]
        sampled = meta.get("output_token_logprobs")
        top_rows = meta.get("output_top_logprobs")
        if not isinstance(sampled, list) or not isinstance(top_rows, list) or len(sampled) != len(top_rows):
            raise ValueError(f"record {slot} has malformed output logprobs")
        response_ids = [parse_logprob_item(item, f"record[{slot}].sampled")[1] for item in sampled]
        if not response_ids:
            raise ValueError(f"record {slot} generated no response tokens")
        sampled_log_probs = []
        support_ids = []
        support_log_probs = []
        support_valid = []
        for response_offset, (sampled_item, top_row, token_id) in enumerate(
            zip(sampled, top_rows, response_ids, strict=True)
        ):
            sampled_lp, ids, logs, valid = support_row(
                sampled_item,
                top_row,
                sampled_token_id=token_id,
                top_k=args.top_k,
                location=f"record[{slot}].output[{response_offset}]",
            )
            sampled_log_probs.append(sampled_lp)
            support_ids.append(ids)
            support_log_probs.append(logs)
            support_valid.append(valid)
        routes = decode_routes(
            meta,
            token_count=len(prompt_ids) + len(response_ids) - 1,
            num_layers=args.num_layers,
            routing_top_k=args.routing_top_k,
        )
        records.append(
            {
                "slot": slot,
                "row_index": int(prompt["row_index"]),
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.int32),
                "response_ids": torch.tensor(response_ids, dtype=torch.int32),
                # Preserve the online decode result so the same-policy
                # full-prefix rescore can be verified before its captures are
                # used as the behavior distribution for cross-engine scoring.
                "generation_sampled_log_probs": torch.tensor(sampled_log_probs, dtype=torch.float32),
                "generation_support_ids": torch.from_numpy(np.stack(support_ids)),
                "generation_support_log_probs": torch.from_numpy(np.stack(support_log_probs)),
                "generation_support_valid": torch.from_numpy(np.stack(support_valid)),
                "generation_routes": routes,
                "finish_reason": meta.get("finish_reason"),
                "seed": args.seed + slot,
            }
        )
        print(
            f"generated slot={slot} row={prompt['row_index']} prompt={len(prompt_ids)} "
            f"response={len(response_ids)}",
            flush=True,
        )
    write_torch(
        {
            "format": FORMAT_CONTEXTS,
            "source_prompts": str(args.prompts),
            "source_prompts_sha256": sha256(args.prompts),
            "generator_policy": args.policy_label,
            "adapter_name": args.adapter_name,
            "server_base_url": base_url,
            "sampling_seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
            "dp_size": args.dp_size,
            "num_layers": args.num_layers,
            "routing_top_k": args.routing_top_k,
            "records": records,
        },
        args.output,
    )


def capture_generation(args: argparse.Namespace) -> None:
    """Convert online generation observations into a trainer-score capture.

    Unlike ``rescore``, this path preserves the exact DSPARK generation
    endpoint.  The diagonal generation captures are required to determine
    whether a full-prefix 2x2 grid represents the online DPPO denominator.
    """

    contexts = load_torch(args.contexts)
    if contexts.get("format") != FORMAT_CONTEXTS:
        raise ValueError(f"unexpected context format {contexts.get('format')!r}")
    records: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for record in contexts["records"]:
        required = (
            "generation_sampled_log_probs",
            "generation_support_ids",
            "generation_support_log_probs",
            "generation_support_valid",
            "generation_routes",
        )
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"generation record is missing fields: {missing}")
        prompt_ids = record["prompt_ids"].int().contiguous()
        response_ids = record["response_ids"].int().contiguous()
        response_tokens = int(response_ids.numel())
        if record["generation_sampled_log_probs"].numel() != response_tokens:
            raise ValueError(f"record {record['slot']} generation logprob length mismatch")
        expected_route_rows = int(prompt_ids.numel()) + response_tokens - 1
        routes = record["generation_routes"].int().contiguous()
        if routes.shape[0] != expected_route_rows:
            raise ValueError(
                f"record {record['slot']} has {routes.shape[0]} route rows, " f"expected {expected_route_rows}"
            )
        records.append(
            {
                "slot": int(record["slot"]),
                "row_index": int(record["row_index"]),
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "behavior_sampled_log_probs": record["generation_sampled_log_probs"].float(),
                "behavior_support_ids": record["generation_support_ids"].int(),
                "behavior_support_log_probs": record["generation_support_log_probs"].float(),
                "behavior_support_valid": record["generation_support_valid"].bool(),
                "routes": routes,
            }
        )
        controls.append(
            {
                "slot": int(record["slot"]),
                "valid_equal": True,
                "ids_equal": True,
                "routes_equal": True,
                "sampled_logprob_mean_abs": 0.0,
                "sampled_logprob_max_abs": 0.0,
                "support_logprob_mean_abs": 0.0,
                "support_logprob_max_abs": 0.0,
                "passed": True,
            }
        )
    write_torch(
        {
            "format": FORMAT_CAPTURE,
            "capture_mode": "generation",
            "source_contexts": str(args.contexts),
            "source_contexts_sha256": sha256(args.contexts),
            "context_policy": contexts["generator_policy"],
            "behavior_policy": contexts["generator_policy"],
            "adapter_name": contexts["adapter_name"],
            "top_k": contexts["top_k"],
            "same_policy_controls": controls,
            "records": records,
        },
        args.output,
    )


def rescore_contexts(args: argparse.Namespace) -> None:
    contexts = load_torch(args.contexts)
    if contexts.get("format") != FORMAT_CONTEXTS:
        raise ValueError(f"unexpected context format {contexts.get('format')!r}")
    base_url = args.base_url.rstrip("/")
    output_records: list[dict[str, Any]] = []
    same_policy_controls: list[dict[str, Any]] = []
    for record in contexts["records"]:
        flush_cache(base_url, args.timeout)
        prompt_ids = record["prompt_ids"].long().tolist()
        response_ids = record["response_ids"].long().tolist()
        all_ids = prompt_ids + response_ids
        payload = {
            "input_ids": all_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": args.top_k,
            "return_routed_experts": True,
            "lora_path": args.adapter_name,
            "routed_dp_rank": int(record["slot"]) % args.dp_size,
        }
        response = post_json(f"{base_url}/generate", payload, args.timeout)
        meta = response["meta_info"]
        sampled_rows = meta.get("input_token_logprobs")
        top_rows = meta.get("input_top_logprobs")
        if not isinstance(sampled_rows, list) or not isinstance(top_rows, list):
            raise ValueError("prefill response is missing input token/top logprobs")
        if len(sampled_rows) != len(all_ids) or len(top_rows) != len(all_ids):
            raise ValueError(f"prefill rows {len(sampled_rows)}/{len(top_rows)} do not match input {len(all_ids)}")

        sampled_log_probs = []
        support_ids = []
        support_log_probs = []
        support_valid = []
        for response_offset, token_id in enumerate(response_ids):
            input_position = len(prompt_ids) + response_offset
            sampled_lp, ids, logs, valid = support_row(
                sampled_rows[input_position],
                top_rows[input_position],
                sampled_token_id=token_id,
                top_k=args.top_k,
                location=f"record[{record['slot']}].input[{input_position}]",
            )
            sampled_log_probs.append(sampled_lp)
            support_ids.append(ids)
            support_log_probs.append(logs)
            support_valid.append(valid)

        # One generated token makes routed_experts cover every input position.
        # Trainer scoring consumes all but the final response token, hence N-1 rows.
        routes = decode_routes(
            meta,
            token_count=len(all_ids),
            num_layers=args.num_layers,
            routing_top_k=args.routing_top_k,
        )[:-1]
        if routes.shape[0] != len(all_ids) - 1:
            raise RuntimeError("rescore route slice is not aligned to trainer input")
        behavior_sampled_tensor = torch.tensor(sampled_log_probs, dtype=torch.float32)
        behavior_support_ids = torch.from_numpy(np.stack(support_ids))
        behavior_support_logs = torch.from_numpy(np.stack(support_log_probs))
        behavior_support_valid = torch.from_numpy(np.stack(support_valid))

        if contexts["generator_policy"] == args.policy_label:
            required = (
                "generation_sampled_log_probs",
                "generation_support_ids",
                "generation_support_log_probs",
                "generation_support_valid",
                "generation_routes",
            )
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(f"same-policy control is missing generation fields: {missing}")
            generation_valid = record["generation_support_valid"].bool()
            valid_equal = torch.equal(generation_valid, behavior_support_valid)
            ids_equal = torch.equal(record["generation_support_ids"], behavior_support_ids)
            routes_equal = torch.equal(record["generation_routes"], routes)
            sampled_abs = (record["generation_sampled_log_probs"].float() - behavior_sampled_tensor).abs()
            sampled_mean_abs = float(sampled_abs.mean())
            sampled_max_abs = float(sampled_abs.max())
            if valid_equal and ids_equal:
                support_mask = generation_valid
                support_abs = (record["generation_support_log_probs"].float() - behavior_support_logs)[
                    support_mask
                ].abs()
                support_mean_abs: float | None = float(support_abs.mean())
                support_max_abs: float | None = float(support_abs.max())
            else:
                support_mean_abs = None
                support_max_abs = None
            passed = bool(
                valid_equal
                and ids_equal
                and routes_equal
                and sampled_max_abs <= args.same_policy_atol
                and support_max_abs is not None
                and support_max_abs <= args.same_policy_atol
            )
            control = {
                "slot": int(record["slot"]),
                "valid_equal": valid_equal,
                "ids_equal": ids_equal,
                "routes_equal": routes_equal,
                "sampled_logprob_mean_abs": sampled_mean_abs,
                "sampled_logprob_max_abs": sampled_max_abs,
                "support_logprob_mean_abs": support_mean_abs,
                "support_logprob_max_abs": support_max_abs,
                "passed": passed,
            }
            same_policy_controls.append(control)
            if args.require_same_policy_control and not passed:
                raise RuntimeError(f"same-policy generation/rescore control failed: {control}")
        output_records.append(
            {
                "slot": int(record["slot"]),
                "row_index": int(record["row_index"]),
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.int32),
                "response_ids": torch.tensor(response_ids, dtype=torch.int32),
                "behavior_sampled_log_probs": behavior_sampled_tensor,
                "behavior_support_ids": behavior_support_ids,
                "behavior_support_log_probs": behavior_support_logs,
                "behavior_support_valid": behavior_support_valid,
                "routes": routes.int().contiguous(),
            }
        )
        print(
            f"rescored policy={args.policy_label} context={contexts['generator_policy']} "
            f"slot={record['slot']} response={len(response_ids)}",
            flush=True,
        )
    write_torch(
        {
            "format": FORMAT_CAPTURE,
            "capture_mode": "full_prefix_rescore",
            "source_contexts": str(args.contexts),
            "source_contexts_sha256": sha256(args.contexts),
            "context_policy": contexts["generator_policy"],
            "behavior_policy": args.policy_label,
            "adapter_name": args.adapter_name,
            "server_base_url": base_url,
            "top_k": args.top_k,
            "dp_size": args.dp_size,
            "num_layers": args.num_layers,
            "routing_top_k": args.routing_top_k,
            "same_policy_controls": same_policy_controls,
            "records": output_records,
        },
        args.output,
    )


def load_adapter_into_model(model: torch.nn.Module, adapter_dir: Path) -> dict[str, Any]:
    from custom_kernels.deepseek_v4.megatron.lora import apply_v4_lora, audit_lora
    from safetensors.torch import load_file

    from slime.backends.megatron_utils.update_weight.lora_adapter_sync import megatron_adapter_name_to_peft

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank = int(config["r"])
    effective_alpha = float(config["lora_alpha"])
    # Exported adapters encode rsLoRA's effective scale in lora_alpha, so the
    # standalone wrapper applies ordinary alpha/r scaling exactly once.
    model = apply_v4_lora(
        model,
        dim=rank,
        alpha=effective_alpha,
        dropout=0.0,
        rslora=False,
        shared_expert=True,
    ).eval()
    state = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    loaded: set[str] = set()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            peft_name = megatron_adapter_name_to_peft(SimpleNamespace(), name, parameter)
            if peft_name is None:
                continue
            if peft_name not in state:
                raise KeyError(f"adapter is missing {peft_name} for model parameter {name}")
            source = state[peft_name]
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(f"adapter shape mismatch for {peft_name}: {source.shape} vs {parameter.shape}")
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(peft_name)
    extra = sorted(set(state) - loaded)
    if extra or len(loaded) != len(state):
        raise RuntimeError(f"adapter load mismatch: loaded={len(loaded)} state={len(state)} extra={extra[:5]}")
    return {
        "model": model,
        "adapter_tensors": len(loaded),
        "adapter_sha256": sha256(adapter_dir / "adapter_model.safetensors"),
        "adapter_config": config,
        "audit": audit_lora(model),
    }


def predictive_rows(
    behavior_logs: torch.Tensor,
    current_logs: torch.Tensor,
    valid: torch.Tensor,
    sampled_behavior_logs: torch.Tensor,
    sampled_current_logs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    behavior = torch.where(valid, behavior_logs.double(), 0.0)
    current = torch.where(valid, current_logs.double(), 0.0)
    behavior_prob = behavior.exp() * valid
    current_prob = current.exp() * valid
    behavior_mass = behavior_prob.sum(dim=-1)
    current_mass = current_prob.sum(dim=-1)
    behavior_tail = (1.0 - behavior_mass).clamp_min(0.0)
    current_tail = (1.0 - current_mass).clamp_min(0.0)
    tiny = torch.finfo(torch.float64).tiny
    retained_kl = (behavior_prob * (behavior - current) * valid).sum(dim=-1)
    tail_kl = torch.where(
        behavior_tail > 0,
        behavior_tail * (behavior_tail.clamp_min(tiny).log() - current_tail.clamp_min(tiny).log()),
        0.0,
    )
    topk_kl = (retained_kl + tail_kl).clamp_min(0.0)

    count = valid.sum(dim=-1).clamp_min(1).double()
    behavior_mean = (behavior * valid).sum(dim=-1) / count
    current_mean = (current * valid).sum(dim=-1) / count
    centered_abs = (((behavior - behavior_mean[:, None]) - (current - current_mean[:, None])).abs() * valid).sum(
        dim=-1
    ) / count
    sampled_behavior_prob = sampled_behavior_logs.double().exp()
    sampled_current_prob = sampled_current_logs.double().exp()
    local_term = sampled_current_prob - sampled_behavior_prob
    retained_term = (current_prob * (behavior_prob - current_prob) * valid).sum(dim=-1)
    predictive_tail_term = current_tail * (behavior_tail - current_tail)
    predictive_dot = local_term + retained_term + predictive_tail_term
    return {
        "topk_kl": topk_kl,
        "retained_kl": retained_kl,
        "tail_kl": tail_kl,
        "behavior_tail_mass": behavior_tail,
        "current_tail_mass": current_tail,
        "centered_support_logit_abs_diff": centered_abs,
        "sampled_logprob_delta": sampled_current_logs.double() - sampled_behavior_logs.double(),
        "predictive_dot": predictive_dot,
        "predictive_tail_term": predictive_tail_term,
    }


def numeric_summary(values: torch.Tensor) -> dict[str, float | int]:
    x = values.detach().double().cpu().numpy()
    result: dict[str, float | int] = {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "mean_abs": float(np.abs(x).mean()),
        "max": float(x.max()),
    }
    for q in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 1.0):
        result[f"p{q * 100:g}"] = float(np.quantile(x, q))
    return result


def score_capture(
    *,
    model: torch.nn.Module,
    config: Any,
    capture: dict[str, Any],
    capture_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    rank: int,
    adapter_info: dict[str, Any],
    load_stats: dict[str, Any],
    train_environment: dict[str, str],
) -> None:
    import torch.distributed as dist

    all_metrics: dict[str, list[torch.Tensor]] = {}
    output_records: list[dict[str, Any]] = []
    for record in capture["records"]:
        prompt_ids = record["prompt_ids"].long().tolist()
        response_ids = record["response_ids"].long().tolist()
        model_ids = (prompt_ids + response_ids)[:-1]
        routes = record["routes"].int()
        if tuple(routes.shape) != (len(model_ids), config.num_hidden_layers, 6):
            raise ValueError(f"route shape {tuple(routes.shape)} is not aligned to {len(model_ids)} inputs")
        from slime.utils.routing_replay import RoutingReplay, record_rollout_routing_replay_for_layer

        RoutingReplay.clear_all()
        offset = 0
        for layer_id in range(config.num_hidden_layers):
            offset = record_rollout_routing_replay_for_layer(
                model,
                layer_id,
                routes[:, layer_id],
                offset,
            )
        input_ids = torch.tensor(model_ids, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids)
        RoutingReplay.check_fully_consumed(context=f"slot {record['slot']}", model_modules=model)
        start = len(prompt_ids) - 1
        train_rows = logits[0, start : start + len(response_ids)].float()
        support_ids = record["behavior_support_ids"].long().cuda()
        valid = record["behavior_support_valid"].bool().cuda()
        train_log_probs = torch.log_softmax(train_rows, dim=-1)
        current_logs = train_log_probs.gather(-1, support_ids.clamp_min(0))
        response_tensor = torch.tensor(response_ids, dtype=torch.long, device="cuda")
        current_sampled = train_log_probs.gather(-1, response_tensor[:, None])[:, 0]
        rows = predictive_rows(
            record["behavior_support_log_probs"].cuda(),
            current_logs,
            valid,
            record["behavior_sampled_log_probs"].cuda(),
            current_sampled,
        )
        if rank == 0:
            cpu_rows = {key: value.detach().cpu() for key, value in rows.items()}
            output_records.append(
                {
                    "slot": int(record["slot"]),
                    "row_index": int(record["row_index"]),
                    "response_tokens": len(response_ids),
                    "metrics": cpu_rows,
                }
            )
            for key, value in cpu_rows.items():
                all_metrics.setdefault(key, []).append(value)
        del logits, train_rows, train_log_probs, input_ids, current_logs, current_sampled
        dist.barrier()

    if rank == 0:
        concatenated = {key: torch.cat(values) for key, values in all_metrics.items()}
        summary = {key: numeric_summary(value) for key, value in concatenated.items()}
        for delta in args.deltas:
            outside = concatenated["topk_kl"] > delta
            positive_direction = outside & (concatenated["predictive_dot"] > 0)
            negative_direction = outside & (concatenated["predictive_dot"] < 0)
            summary[f"outside@{delta:g}"] = {
                "count": int(outside.sum()),
                "fraction": float(outside.double().mean()),
            }
            summary[f"would_clip_positive@{delta:g}"] = {
                "count": int(positive_direction.sum()),
                "fraction": float(positive_direction.double().mean()),
            }
            summary[f"would_clip_negative@{delta:g}"] = {
                "count": int(negative_direction.sum()),
                "fraction": float(negative_direction.double().mean()),
            }
        per_record = []
        for record in output_records:
            topk = record["metrics"]["topk_kl"]
            predictive_dot = record["metrics"]["predictive_dot"]
            for delta in args.deltas:
                outside = topk > delta
                record["metrics"][f"outside@{delta:g}"] = outside.double()
                record["metrics"][f"would_clip_positive@{delta:g}"] = (outside & (predictive_dot > 0)).double()
                record["metrics"][f"would_clip_negative@{delta:g}"] = (outside & (predictive_dot < 0)).double()
            per_record.append(
                {
                    "slot": record["slot"],
                    "row_index": record["row_index"],
                    "response_tokens": record["response_tokens"],
                    "topk_kl_mean": float(topk.mean()),
                    **{f"outside@{delta:g}": float((topk > delta).double().mean()) for delta in args.deltas},
                }
            )
        payload = {
            "format": FORMAT_SCORE,
            "capture": str(capture_path),
            "capture_sha256": sha256(capture_path),
            "capture_mode": capture.get("capture_mode", "unknown"),
            "context_policy": capture["context_policy"],
            "behavior_policy": capture["behavior_policy"],
            "current_policy": args.policy_label,
            "checkpoint": str(args.checkpoint),
            "adapter": str(args.adapter),
            "adapter_info": adapter_info,
            "load_stats": load_stats,
            "train_environment_profile": args.train_environment_profile,
            "train_environment": train_environment,
            "topology_boundary": "standalone CP1/EP8; formal r21 is CP2/EP8",
            "same_policy_controls": capture.get("same_policy_controls", []),
            "summary": summary,
            "per_record": per_record,
            "records": output_records,
        }
        write_torch(payload, output_path)
        json_output = output_path.with_suffix(".json")
        json_output.write_text(
            json.dumps(
                {key: value for key, value in payload.items() if key != "records"},
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    torch.cuda.empty_cache()


def train_score(args: argparse.Namespace) -> None:
    if len(args.capture) != len(args.output):
        raise ValueError(f"received {len(args.capture)} captures but {len(args.output)} outputs")
    jobs = []
    for capture_path, output_path in zip(args.capture, args.output, strict=True):
        capture = load_torch(capture_path)
        if capture.get("format") != FORMAT_CAPTURE:
            raise ValueError(f"{capture_path} has unexpected capture format {capture.get('format')!r}")
        if capture.get("behavior_policy") != args.policy_label:
            raise ValueError(
                f"{capture_path} behavior policy {capture.get('behavior_policy')!r} "
                f"does not match --policy-label {args.policy_label!r}"
            )
        jobs.append((capture_path, capture, output_path))

    train_environment = configure_train_environment(args.train_environment_profile)
    # Routing replay is selected while the model is constructed; recording the
    # rollout expert IDs alone is insufficient if these gates are absent.
    os.environ["ENABLE_ROUTING_REPLAY"] = "1"
    os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

    import torch.distributed as dist
    from custom_kernels.deepseek_v4.megatron.model_provider import build_v4_mcore_model
    from custom_kernels.deepseek_v4.megatron.native_checkpoint import load_native_checkpoint_into_mcore_model
    from custom_kernels.deepseek_v4.megatron.slice_torch_dist import init_dist_from_env
    from transformers import AutoConfig

    rank, world_size, local_rank = init_dist_from_env(
        args.master_port,
        pp_size=1,
        ep_size=args.ep_size,
        order="tp-cp-ep-dp-pp",
    )
    if world_size != args.ep_size:
        raise ValueError(f"WORLD_SIZE={world_size} != ep_size={args.ep_size}")
    torch.cuda.set_device(local_rank)
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = (
        build_v4_mcore_model(
            config,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=args.ep_size,
            expert_model_parallel_rank=rank,
            moe_token_dispatcher_type=args.moe_dispatcher,
            moe_flex_dispatcher_backend=args.moe_flex_backend,
            moe_router_dtype="fp32",
            moe_deepep_num_sms=args.moe_deepep_num_sms,
        )
        .cuda()
        .bfloat16()
        .eval()
    )
    load_stats = load_native_checkpoint_into_mcore_model(
        model,
        str(args.checkpoint),
        layer_map={index: index for index in range(config.num_hidden_layers)},
        strict=True,
    )
    adapter_info = load_adapter_into_model(model, args.adapter)
    model = adapter_info.pop("model")
    dist.barrier()

    for capture_path, capture, output_path in jobs:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "scoring_capture": str(capture_path),
                        "output": str(output_path),
                        "capture_mode": capture.get("capture_mode"),
                    }
                ),
                flush=True,
            )
        score_capture(
            model=model,
            config=config,
            capture=capture,
            capture_path=capture_path,
            output_path=output_path,
            args=args,
            rank=rank,
            adapter_info=adapter_info,
            load_stats=load_stats,
            train_environment=train_environment,
        )


def analyze_grid(args: argparse.Namespace) -> None:
    paths = {
        "p74_c74": args.policy74_context74,
        "p74_c94": args.policy74_context94,
        "p94_c74": args.policy94_context74,
        "p94_c94": args.policy94_context94,
    }
    cells = {name: load_torch(path) for name, path in paths.items()}
    for name, cell in cells.items():
        if cell.get("format") != FORMAT_SCORE:
            raise ValueError(f"{name} has unexpected format {cell.get('format')!r}")
        if cell.get("capture_mode") != "full_prefix_rescore":
            raise ValueError(f"{name} capture mode is {cell.get('capture_mode')!r}, " "expected full_prefix_rescore")
        if cell.get("behavior_policy") != cell.get("current_policy"):
            raise ValueError(
                f"{name} is not lag-0: behavior={cell.get('behavior_policy')!r} "
                f"current={cell.get('current_policy')!r}"
            )
    if cells["p74_c74"]["current_policy"] != cells["p74_c94"]["current_policy"]:
        raise ValueError("policy-74 row does not use one current policy")
    if cells["p94_c74"]["current_policy"] != cells["p94_c94"]["current_policy"]:
        raise ValueError("policy-94 row does not use one current policy")
    if cells["p74_c74"]["context_policy"] != cells["p94_c74"]["context_policy"]:
        raise ValueError("context-74 column does not use one context set")
    if cells["p74_c94"]["context_policy"] != cells["p94_c94"]["context_policy"]:
        raise ValueError("context-94 column does not use one context set")

    slots = sorted(int(record["slot"]) for record in cells["p74_c74"]["records"])
    if not slots:
        raise ValueError("grid contains no records")
    by_cell: dict[str, dict[int, dict[str, Any]]] = {}
    for name, cell in cells.items():
        mapping = {int(record["slot"]): record for record in cell["records"]}
        if sorted(mapping) != slots:
            raise ValueError(f"{name} slots {sorted(mapping)} do not match {slots}")
        by_cell[name] = mapping

    metric_specs: list[tuple[str, str, str]] = [
        ("topk_kl_mean", "topk_kl", "mean"),
        ("retained_kl_mean", "retained_kl", "mean"),
        ("tail_kl_mean", "tail_kl", "mean"),
        ("behavior_tail_mass_mean", "behavior_tail_mass", "mean"),
        ("current_tail_mass_mean", "current_tail_mass", "mean"),
        (
            "centered_support_logit_abs_diff_mean",
            "centered_support_logit_abs_diff",
            "mean",
        ),
        ("sampled_logprob_delta_mean_abs", "sampled_logprob_delta", "mean_abs"),
        ("predictive_dot_mean", "predictive_dot", "mean"),
        ("predictive_dot_mean_abs", "predictive_dot", "mean_abs"),
        ("predictive_tail_term_mean", "predictive_tail_term", "mean"),
    ]
    for delta in args.deltas:
        metric_specs.extend(
            (
                (f"outside@{delta:g}", f"outside@{delta:g}", "mean"),
                (
                    f"would_clip_positive@{delta:g}",
                    f"would_clip_positive@{delta:g}",
                    "mean",
                ),
                (
                    f"would_clip_negative@{delta:g}",
                    f"would_clip_negative@{delta:g}",
                    "mean",
                ),
            )
        )

    # Each row is one prompt cluster. Resampling the same slots across all four
    # cells retains the fixed-prompt pairing while continuations differ between
    # the two generated-context columns.
    sums: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, np.ndarray] = {}
    for cell_name, mapping in by_cell.items():
        counts[cell_name] = np.asarray([int(mapping[slot]["response_tokens"]) for slot in slots], dtype=np.float64)
        sums[cell_name] = {}
        for output_name, tensor_name, transform in metric_specs:
            values = []
            for slot in slots:
                tensor = mapping[slot]["metrics"][tensor_name].double().numpy()
                if transform == "mean_abs":
                    tensor = np.abs(tensor)
                values.append(float(np.asarray(tensor, dtype=np.float64).sum()))
            sums[cell_name][output_name] = np.asarray(values, dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    sampled_slots = rng.integers(0, len(slots), size=(args.bootstrap_reps, len(slots)))
    estimates: dict[str, dict[str, dict[str, float]]] = {}
    bootstrap: dict[str, dict[str, np.ndarray]] = {}
    for metric_name, _, _ in metric_specs:
        estimates[metric_name] = {}
        bootstrap[metric_name] = {}
        for cell_name in paths:
            point = sums[cell_name][metric_name].sum() / counts[cell_name].sum()
            draws = sums[cell_name][metric_name][sampled_slots].sum(axis=1) / counts[cell_name][sampled_slots].sum(
                axis=1
            )
            estimates[metric_name][cell_name] = {"estimate": float(point)}
            bootstrap[metric_name][cell_name] = draws

    comparisons = {
        "policy_effect_on_c74": ("p94_c74", "p74_c74"),
        "policy_effect_on_c94": ("p94_c94", "p74_c94"),
        "context_effect_at_p74": ("p74_c94", "p74_c74"),
        "context_effect_at_p94": ("p94_c94", "p94_c74"),
    }
    output_metrics: dict[str, Any] = {}
    for metric_name, _, _ in metric_specs:
        metric_output: dict[str, Any] = {"cells": estimates[metric_name]}
        for comparison, (left, right) in comparisons.items():
            point = estimates[metric_name][left]["estimate"] - estimates[metric_name][right]["estimate"]
            draws = bootstrap[metric_name][left] - bootstrap[metric_name][right]
            metric_output[comparison] = {
                "estimate": float(point),
                "cluster_bootstrap_ci95": [
                    float(np.quantile(draws, 0.025)),
                    float(np.quantile(draws, 0.975)),
                ],
            }
        did_point = (
            estimates[metric_name]["p94_c94"]["estimate"]
            - estimates[metric_name]["p74_c94"]["estimate"]
            - estimates[metric_name]["p94_c74"]["estimate"]
            + estimates[metric_name]["p74_c74"]["estimate"]
        )
        did_draws = (
            bootstrap[metric_name]["p94_c94"]
            - bootstrap[metric_name]["p74_c94"]
            - bootstrap[metric_name]["p94_c74"]
            + bootstrap[metric_name]["p74_c74"]
        )
        metric_output["policy_context_interaction"] = {
            "estimate": float(did_point),
            "cluster_bootstrap_ci95": [
                float(np.quantile(did_draws, 0.025)),
                float(np.quantile(did_draws, 0.975)),
            ],
        }
        output_metrics[metric_name] = metric_output

    generation_paths: dict[str, Path] = {}
    generation_cells: dict[str, dict[str, Any]] = {}
    if (args.policy74_generation is None) != (args.policy94_generation is None):
        raise ValueError("provide both generation diagonal scores or neither")
    if args.policy74_generation is not None:
        generation_paths = {
            "p74_generation": args.policy74_generation,
            "p94_generation": args.policy94_generation,
        }
        generation_cells = {name: load_torch(path) for name, path in generation_paths.items()}
        expected = {
            "p74_generation": cells["p74_c74"],
            "p94_generation": cells["p94_c94"],
        }
        generation_sums: dict[str, dict[str, np.ndarray]] = {}
        generation_counts: dict[str, np.ndarray] = {}
        for name, cell in generation_cells.items():
            if cell.get("format") != FORMAT_SCORE or cell.get("capture_mode") != "generation":
                raise ValueError(f"{name} is not a generation train-score artifact")
            if cell.get("behavior_policy") != cell.get("current_policy"):
                raise ValueError(f"{name} is not lag-0")
            if cell.get("current_policy") != expected[name].get("current_policy"):
                raise ValueError(f"{name} policy does not match its grid diagonal")
            mapping = {int(record["slot"]): record for record in cell["records"]}
            if sorted(mapping) != slots:
                raise ValueError(f"{name} slots {sorted(mapping)} do not match {slots}")
            diagonal_name = "p74_c74" if name == "p74_generation" else "p94_c94"
            for slot in slots:
                if int(mapping[slot]["response_tokens"]) != int(by_cell[diagonal_name][slot]["response_tokens"]):
                    raise ValueError(f"{name} response length differs at slot {slot}")
            generation_counts[name] = np.asarray(
                [int(mapping[slot]["response_tokens"]) for slot in slots],
                dtype=np.float64,
            )
            generation_sums[name] = {}
            for output_name, tensor_name, transform in metric_specs:
                values = []
                for slot in slots:
                    tensor = mapping[slot]["metrics"][tensor_name].double().numpy()
                    if transform == "mean_abs":
                        tensor = np.abs(tensor)
                    values.append(float(np.asarray(tensor, dtype=np.float64).sum()))
                generation_sums[name][output_name] = np.asarray(values, dtype=np.float64)

        for metric_name, _, _ in metric_specs:
            gen_estimates: dict[str, float] = {}
            gen_draws: dict[str, np.ndarray] = {}
            for name in generation_paths:
                gen_estimates[name] = float(generation_sums[name][metric_name].sum() / generation_counts[name].sum())
                gen_draws[name] = generation_sums[name][metric_name][sampled_slots].sum(axis=1) / generation_counts[
                    name
                ][sampled_slots].sum(axis=1)
            metric_output = output_metrics[metric_name]
            metric_output["generation_diagonal"] = gen_estimates
            endpoint_comparisons = {
                "rescore_minus_generation_at_p74": (
                    estimates[metric_name]["p74_c74"]["estimate"],
                    bootstrap[metric_name]["p74_c74"],
                    gen_estimates["p74_generation"],
                    gen_draws["p74_generation"],
                ),
                "rescore_minus_generation_at_p94": (
                    estimates[metric_name]["p94_c94"]["estimate"],
                    bootstrap[metric_name]["p94_c94"],
                    gen_estimates["p94_generation"],
                    gen_draws["p94_generation"],
                ),
                "generation_diagonal_change_p94_minus_p74": (
                    gen_estimates["p94_generation"],
                    gen_draws["p94_generation"],
                    gen_estimates["p74_generation"],
                    gen_draws["p74_generation"],
                ),
            }
            for comparison, (left, left_draws, right, right_draws) in endpoint_comparisons.items():
                draws = left_draws - right_draws
                metric_output[comparison] = {
                    "estimate": float(left - right),
                    "cluster_bootstrap_ci95": [
                        float(np.quantile(draws, 0.025)),
                        float(np.quantile(draws, 0.975)),
                    ],
                }

    controls = {}
    for cell_name in ("p74_c74", "p94_c94"):
        rows = cells[cell_name].get("same_policy_controls", [])
        if len(rows) != len(slots):
            raise ValueError(f"{cell_name} has {len(rows)} same-policy controls for {len(slots)} slots")
        support_means = [
            float(row["support_logprob_mean_abs"]) for row in rows if row.get("support_logprob_mean_abs") is not None
        ]
        support_maxima = [
            float(row["support_logprob_max_abs"]) for row in rows if row.get("support_logprob_max_abs") is not None
        ]
        controls[cell_name] = {
            "records": len(rows),
            "all_passed": all(bool(row.get("passed", False)) for row in rows),
            "all_valid_equal": all(bool(row["valid_equal"]) for row in rows),
            "all_ids_equal": all(bool(row["ids_equal"]) for row in rows),
            "all_routes_equal": all(bool(row["routes_equal"]) for row in rows),
            "sampled_logprob_mean_abs": float(np.mean([float(row["sampled_logprob_mean_abs"]) for row in rows])),
            "sampled_logprob_max_abs": max(float(row["sampled_logprob_max_abs"]) for row in rows),
            "support_logprob_mean_abs": float(np.mean(support_means)) if support_means else None,
            "support_logprob_max_abs": max(support_maxima) if support_maxima else None,
        }

    payload = {
        "format": "predictive-tail-grid-analysis-v1",
        "scope": (
            "Lag-0 cross-engine mismatch decomposed by checkpoint state and " "fixed-prompt generated context set."
        ),
        "paths": {key: str(value) for key, value in paths.items()},
        "path_sha256": {key: sha256(value) for key, value in paths.items()},
        "generation_paths": {key: str(value) for key, value in generation_paths.items()},
        "generation_path_sha256": {key: sha256(value) for key, value in generation_paths.items()},
        "policy74": cells["p74_c74"]["current_policy"],
        "policy94": cells["p94_c74"]["current_policy"],
        "context74": cells["p74_c74"]["context_policy"],
        "context94": cells["p74_c94"]["context_policy"],
        "prompt_clusters": len(slots),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.seed,
        "same_policy_controls": controls,
        "full_prefix_grid_represents_generation_endpoint": all(bool(row["all_passed"]) for row in controls.values()),
        "metrics": output_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}))

    review = args.output.with_suffix(".txt")
    lines = [
        payload["scope"],
        f"prompt_clusters={len(slots)} bootstrap_reps={args.bootstrap_reps} seed={args.seed}",
        f"same_policy_controls={json.dumps(controls, sort_keys=True)}",
        "",
    ]
    review_names = [
        name for name in ("outside@0.15", "topk_kl_mean", "sampled_logprob_delta_mean_abs") if name in output_metrics
    ]
    for metric_name in review_names:
        metric = output_metrics[metric_name]
        lines.append(f"[{metric_name}]")
        lines.append("cells " + " ".join(f"{name}={metric['cells'][name]['estimate']:.10g}" for name in paths))
        for comparison in (*comparisons, "policy_context_interaction"):
            row = metric[comparison]
            lines.append(
                f"{comparison}={row['estimate']:.10g} "
                f"ci95=[{row['cluster_bootstrap_ci95'][0]:.10g},"
                f"{row['cluster_bootstrap_ci95'][1]:.10g}]"
            )
        if "generation_diagonal" in metric:
            lines.append(
                "generation_diagonal "
                + " ".join(f"{name}={value:.10g}" for name, value in metric["generation_diagonal"].items())
            )
            for comparison in (
                "rescore_minus_generation_at_p74",
                "rescore_minus_generation_at_p94",
                "generation_diagonal_change_p94_minus_p74",
            ):
                row = metric[comparison]
                lines.append(
                    f"{comparison}={row['estimate']:.10g} "
                    f"ci95=[{row['cluster_bootstrap_ci95'][0]:.10g},"
                    f"{row['cluster_bootstrap_ci95'][1]:.10g}]"
                )
        lines.append("")
    review.write_text("\n".join(lines) + "\n")
    print(f"review={review}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-prompts")
    prepare.add_argument("--parquet", type=Path, required=True)
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--prompt-key", default="prompt")
    prepare.add_argument("--metadata-key", default="extra_info")
    prepare.add_argument("--count", type=int, default=16)
    prepare.add_argument("--seed", type=int, default=20260723)
    prepare.add_argument("--max-prompt-tokens", type=int, default=6000)
    prepare.set_defaults(function=prepare_prompts)

    generate = sub.add_parser("generate")
    generate.add_argument("--prompts", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--base-url", default="http://127.0.0.1:31003")
    generate.add_argument("--adapter-name", required=True)
    generate.add_argument("--policy-label", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=1024)
    generate.add_argument("--seed", type=int, default=20260723)
    generate.add_argument("--top-k", type=int, default=20)
    generate.add_argument("--dp-size", type=int, default=8)
    generate.add_argument("--num-layers", type=int, default=43)
    generate.add_argument("--routing-top-k", type=int, default=6)
    generate.add_argument("--timeout", type=int, default=2400)
    generate.set_defaults(function=generate_contexts)

    generation_capture = sub.add_parser("capture-generation")
    generation_capture.add_argument("--contexts", type=Path, required=True)
    generation_capture.add_argument("--output", type=Path, required=True)
    generation_capture.set_defaults(function=capture_generation)

    rescore = sub.add_parser("rescore")
    rescore.add_argument("--contexts", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    rescore.add_argument("--base-url", default="http://127.0.0.1:31003")
    rescore.add_argument("--adapter-name", required=True)
    rescore.add_argument("--policy-label", required=True)
    rescore.add_argument("--top-k", type=int, default=20)
    rescore.add_argument("--dp-size", type=int, default=8)
    rescore.add_argument("--num-layers", type=int, default=43)
    rescore.add_argument("--routing-top-k", type=int, default=6)
    rescore.add_argument("--same-policy-atol", type=float, default=1e-6)
    rescore.add_argument("--require-same-policy-control", action="store_true")
    rescore.add_argument("--timeout", type=int, default=2400)
    rescore.set_defaults(function=rescore_contexts)

    score = sub.add_parser("train-score")
    score.add_argument("--capture", type=Path, nargs="+", required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--adapter", type=Path, required=True)
    score.add_argument("--policy-label", required=True)
    score.add_argument("--output", type=Path, nargs="+", required=True)
    score.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
    score.add_argument("--ep-size", type=int, default=8)
    score.add_argument("--moe-dispatcher", choices=("flex", "alltoall", "allgather"), default="flex")
    score.add_argument("--moe-flex-backend", default="deepep")
    score.add_argument("--moe-deepep-num-sms", type=int, default=20)
    score.add_argument("--master-port", type=int, default=29731)
    score.add_argument("--train-environment-profile", choices=("r21",), default="r21")
    score.set_defaults(function=train_score)

    analyze = sub.add_parser("analyze-grid")
    analyze.add_argument("--policy74-context74", type=Path, required=True)
    analyze.add_argument("--policy74-context94", type=Path, required=True)
    analyze.add_argument("--policy94-context74", type=Path, required=True)
    analyze.add_argument("--policy94-context94", type=Path, required=True)
    analyze.add_argument("--policy74-generation", type=Path)
    analyze.add_argument("--policy94-generation", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
    analyze.add_argument("--bootstrap-reps", type=int, default=10000)
    analyze.add_argument("--seed", type=int, default=20260723)
    analyze.set_defaults(function=analyze_grid)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.function(args)
    finally:
        if args.command == "train-score":
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
