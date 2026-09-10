import asyncio
import base64 as pybase64
import copy
import inspect
import json
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from jinja2 import Template, TemplateError
except ImportError:
    Template = None
    TemplateError = Exception
from tqdm import tqdm

from slime.backends.sglang_utils.server_control import abort_servers_until_idle
from slime.observability.metric_utils import compute_rollout_step
from slime.observability.trace_utils import build_sglang_meta_trace_attrs, trace_function, trace_span
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter, should_drop_dynamic_filter_output
from slime.rollout.sample_hooks import apply_rollout_sample_hooks
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, get_sglang_client_concurrency, post
from slime.utils.lora_utils import rollout_lora_path as _rollout_lora_path
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)

_PROCESSOR_PROMPT_KEYS = {"input_ids", "attention_mask"}

_PREDICTIVE_SUPPORT_FIELDS = (
    "rollout_topk_token_ids",
    "rollout_topk_log_probs",
    "rollout_topk_valid_mask",
)


def _empty_predictive_support(num_tokens: int, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return compact all-invalid predictive-support rows.

    All-invalid rows are only suitable for tokens whose loss mask is zero
    (padding/aborted samples).  Real training tokens are validated later on the
    rollout-manager boundary before they enter Ray's train-data object.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    shape = (num_tokens, top_k + 1)
    return (
        np.zeros(shape, dtype=np.int32),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.bool_),
    )


def _parse_sglang_logprob_item(item: Any, *, field: str, token_index: int, support_index: int | None = None):
    location = f"{field}[{token_index}]"
    if support_index is not None:
        location += f"[{support_index}]"
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        raise ValueError(f"{location} must be a (logprob, token_id, ...) sequence, got {item!r}")

    log_prob, token_id = item[0], item[1]
    if isinstance(token_id, (bool, np.bool_)) or not isinstance(token_id, (int, np.integer)):
        raise ValueError(f"{location} has a non-integer token id: {token_id!r}")
    token_id = int(token_id)
    if token_id < 0 or token_id > np.iinfo(np.int32).max:
        raise ValueError(f"{location} token id is outside int32 range: {token_id}")

    try:
        log_prob = float(log_prob)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} has a non-numeric log probability: {log_prob!r}") from exc
    if not np.isfinite(log_prob):
        raise ValueError(f"{location} has a non-finite log probability: {log_prob!r}")
    return log_prob, token_id


def _extract_predictive_support(
    meta_info: dict[str, Any], top_k: int
) -> tuple[list[int], list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Parse one SGLang response into a unique fixed-width behavior support.

    SGLang returns ``output_top_logprobs`` independently from the sampled-token
    logprob and does not promise that the sample is in top-k.  We therefore use
    the top-k order as the first K slots, overwrite its entry with the sampled
    logprob when present, or append the sampled token in slot K otherwise.  At
    non-unit temperature these values must already be log-probabilities from
    SGLang's temperature-scaled sampling distribution; do not re-temperature a
    sparse Top-K result because its missing tail prevents exact normalization.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    sampled_items = meta_info.get("output_token_logprobs", [])
    top_rows = meta_info.get("output_top_logprobs", [])
    if sampled_items is None:
        sampled_items = []
    if top_rows is None:
        top_rows = []
    if not isinstance(sampled_items, (list, tuple)):
        raise ValueError("meta_info.output_token_logprobs must be a sequence")
    if not isinstance(top_rows, (list, tuple)):
        raise ValueError("meta_info.output_top_logprobs must be a sequence")
    if len(top_rows) != len(sampled_items):
        raise ValueError(
            "predictive support token-count mismatch: "
            f"len(output_top_logprobs)={len(top_rows)} != "
            f"len(output_token_logprobs)={len(sampled_items)}"
        )

    token_count = len(sampled_items)
    token_ids, log_probs, valid_mask = _empty_predictive_support(token_count, top_k)
    sampled_token_ids: list[int] = []
    sampled_log_probs: list[float] = []

    for token_index, (sampled_item, top_row) in enumerate(zip(sampled_items, top_rows, strict=True)):
        sampled_log_prob, sampled_token_id = _parse_sglang_logprob_item(
            sampled_item,
            field="output_token_logprobs",
            token_index=token_index,
        )
        if not isinstance(top_row, (list, tuple)) or len(top_row) != top_k:
            actual = len(top_row) if isinstance(top_row, (list, tuple)) else type(top_row).__name__
            raise ValueError(f"output_top_logprobs[{token_index}] must contain exactly {top_k} entries, got {actual}")

        support_slot_by_id: dict[int, int] = {}
        for support_index, top_item in enumerate(top_row):
            top_log_prob, top_token_id = _parse_sglang_logprob_item(
                top_item,
                field="output_top_logprobs",
                token_index=token_index,
                support_index=support_index,
            )
            if top_token_id in support_slot_by_id:
                raise ValueError(f"output_top_logprobs[{token_index}] contains duplicate token id {top_token_id}")
            support_slot_by_id[top_token_id] = support_index
            token_ids[token_index, support_index] = top_token_id
            log_probs[token_index, support_index] = top_log_prob
            valid_mask[token_index, support_index] = True

        sampled_slot = support_slot_by_id.get(sampled_token_id, top_k)
        token_ids[token_index, sampled_slot] = sampled_token_id
        # This must be the independently-returned sampled logprob, even when a
        # rounded copy of the token is already present in output_top_logprobs.
        log_probs[token_index, sampled_slot] = sampled_log_prob
        valid_mask[token_index, sampled_slot] = True
        sampled_token_ids.append(sampled_token_id)
        sampled_log_probs.append(sampled_log_prob)

    return sampled_token_ids, sampled_log_probs, token_ids, log_probs, valid_mask


def _append_predictive_support(
    sample: Sample,
    support: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    previous_response_length: int,
    top_k: int,
) -> None:
    """Append a partial-generation support chunk without Python-list expansion."""
    expected_width = top_k + 1
    expected_dtypes = (np.dtype(np.int32), np.dtype(np.float32), np.dtype(np.bool_))
    chunk_rows = support[0].shape[0]
    for field, value, expected_dtype in zip(_PREDICTIVE_SUPPORT_FIELDS, support, expected_dtypes, strict=True):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{field} chunk must be a numpy.ndarray, got {type(value).__name__}")
        if value.shape != (chunk_rows, expected_width):
            raise ValueError(f"{field} chunk shape must be {(chunk_rows, expected_width)}, got {value.shape}")
        if value.dtype != expected_dtype:
            raise TypeError(f"{field} chunk dtype must be {expected_dtype}, got {value.dtype}")

        existing = getattr(sample, field)
        if existing is None:
            if previous_response_length:
                previous_mask = sample.loss_mask
                if (
                    previous_mask is None
                    or len(previous_mask) < previous_response_length
                    or any(previous_mask[:previous_response_length])
                ):
                    raise ValueError(
                        f"cannot enable predictive support after {previous_response_length} existing trainable "
                        f"response tokens: {field} is missing"
                    )
                existing = np.zeros((previous_response_length, expected_width), dtype=expected_dtype)
            else:
                existing = np.zeros((0, expected_width), dtype=expected_dtype)
        if not isinstance(existing, np.ndarray):
            raise TypeError(f"existing {field} must be a numpy.ndarray, got {type(existing).__name__}")
        if existing.shape != (previous_response_length, expected_width):
            raise ValueError(
                f"existing {field} shape must be {(previous_response_length, expected_width)}, got {existing.shape}"
            )
        if existing.dtype != expected_dtype:
            raise TypeError(f"existing {field} dtype must be {expected_dtype}, got {existing.dtype}")
        setattr(sample, field, np.concatenate((existing, value), axis=0))


def _set_rollout_step_metadata(args: Namespace, rollout_id: int, samples: list[list[Sample]]) -> None:
    rollout_step = compute_rollout_step(args, rollout_id)
    gen_weight_version = getattr(args, "gen_weight_version", None)
    for group in samples:
        for sample in group:
            sample.metadata = sample.metadata or {}
            sample.metadata["rollout_step"] = rollout_step
            # Stamp ONCE at first submission: a buffered prompt re-submitted in
            # a later cycle keeps its original version, which is what makes the
            # weight-staleness metric detect carry-overs.
            if gen_weight_version is not None and "gen_weight_version" not in sample.metadata:
                sample.metadata["gen_weight_version"] = gen_weight_version


def _prepare_prompt_ids(sample: Sample, tokenizer, processor: Any) -> list[int]:
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal_inputs = any(value is not None for value in raw_multimodal_inputs.values())
    reuse_existing_input_ids = bool(sample.tokens) and (
        sample.multimodal_train_inputs is not None or not has_multimodal_inputs
    )

    if processor and has_multimodal_inputs and not reuse_existing_input_ids:
        processor_output = processor(text=sample.prompt, **build_processor_kwargs(raw_multimodal_inputs))
        prompt_ids = processor_output["input_ids"][0]
        if sample.multimodal_train_inputs is None:
            sample.multimodal_train_inputs = {
                k: v for k, v in processor_output.items() if k not in _PROCESSOR_PROMPT_KEYS
            } or None
        return prompt_ids

    if reuse_existing_input_ids:
        return sample.tokens

    return tokenizer.encode(sample.prompt, add_special_tokens=False)


def _decode_routed_experts(
    meta_info: dict[str, Any],
    *,
    token_count: int,
    num_layers: int,
    expected_topk: int | None,
) -> np.ndarray:
    raw = np.frombuffer(
        pybase64.b64decode(meta_info["routed_experts"].encode("ascii")),
        dtype=np.int32,
    )
    denom = token_count * num_layers
    if denom <= 0:
        raise ValueError(
            "cannot decode routed_experts with non-positive shape: "
            f"token_count={token_count}, num_layers={num_layers}"
        )
    if raw.size % denom != 0:
        raise ValueError(
            "routed_experts payload size is not divisible by token_count*num_layers: "
            f"size={raw.size}, token_count={token_count}, num_layers={num_layers}, "
            f"expected_topk={expected_topk}"
        )
    actual_topk = raw.size // denom
    if expected_topk is not None and actual_topk != expected_topk:
        logger.warning(
            "routed_experts payload topk=%s differs from args.moe_router_topk=%s; using payload shape",
            actual_topk,
            expected_topk,
        )
    return raw.reshape(token_count, num_layers, actual_topk)


class PromptTemplate:
    FORMAT_SUFFIXES = {".yaml", ".yml"}
    JINJA_SUFFIXES = {".jinja", ".j2"}

    def __init__(self, template: str, render_mode: str, source: str | None = None) -> None:
        self.template = template
        self.render_mode = render_mode
        self.source = source

    @classmethod
    def from_path(cls, config_path: str | None, *, prompt_name: str = "tool_response") -> "PromptTemplate | None":
        if config_path is None:
            return None

        config_file = Path(config_path)
        suffix = config_file.suffix.lower()
        if suffix in cls.JINJA_SUFFIXES:
            return cls(config_file.read_text(encoding="utf-8"), "jinja", str(config_file))
        if suffix not in cls.FORMAT_SUFFIXES:
            raise ValueError(
                f"Unsupported multi_turn_prompt_config_path suffix: {config_file.suffix}. "
                "Expected .yaml/.yml or .jinja/.j2."
            )

        with config_file.open(encoding="utf-8") as f:
            prompt_cfg = yaml.safe_load(f) or {}

        for item in prompt_cfg.get("per_turn_prompts", []) or []:
            if str(item.get("name")) == prompt_name and item.get("template"):
                return cls(str(item["template"]), "format", str(config_file))
        return None

    def format(self, feedback: str, feedback_dict: dict[str, Any]) -> str:
        if self.render_mode == "format":
            return self.template.format(feedback=feedback, feedback_dict=feedback_dict)
        if self.render_mode == "jinja":
            if Template is None:
                raise RuntimeError("Jinja multi-turn prompt template requires jinja2 to be installed.")
            try:
                return Template(self.template).render(feedback=feedback, feedback_dict=feedback_dict)
            except TemplateError:
                raise
            except Exception as exc:
                raise TemplateError(
                    f"failed to render Jinja multi-turn prompt template: {type(exc).__name__}: {exc}"
                ) from exc
        raise ValueError(f"Unknown multi-turn prompt render mode: {self.render_mode}")


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, trust_remote_code=True, **getattr(args, "tokenizer_load_kwargs", {})
        )
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        self.apply_chat_template_kwargs = self._get_apply_chat_template_kwargs()
        logger.info("GenerateState apply_chat_template_kwargs=%s", self.apply_chat_template_kwargs)
        self._warn_history_thinking_template()
        self.multi_turn_template = PromptTemplate.from_path(getattr(args, "multi_turn_prompt_config_path", None))

        self.semaphore = asyncio.Semaphore(get_sglang_client_concurrency(args))
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )
        if args.rollout_top_p != 1.0:
            self.sampling_params["custom_params"] = {"return_top_p_token_ids": True}

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        # Name of the LoRA adapter the engines currently serve (alternating sync
        # path). Refreshed each rollout step from the engine (the source of truth)
        # by the RolloutManager; ``None`` until the first adapter is loaded. Every
        # /generate payload routes to this exact name.
        self.active_lora_name: str | None = None

        self.reset()

    def _warn_history_thinking_template(self) -> None:
        if not bool(getattr(self.args, "preserve_history_thinking", False)):
            return

        if self._is_qwen3_5_model():
            logger.warning(
                "args.preserve_history_thinking=True with a Qwen3.5/Qwen3.6-series tokenizer. "
                "Pass preserve_thinking=True to tokenizer.apply_chat_template when rendering multi-turn prompts."
            )
            return

        logger.warning(
            "args.preserve_history_thinking=True. The tokenizer.apply_chat_template behavior is model-specific "
            "and may not preserve previous assistant <think> blocks; please verify the rendered multi-turn prompt "
            "or add model-specific handling."
        )

    def _get_apply_chat_template_kwargs(self) -> dict[str, Any]:
        kwargs = dict(getattr(self.args, "apply_chat_template_kwargs", None) or {})
        if bool(getattr(self.args, "preserve_history_thinking", False)) and self._is_qwen3_5_model():
            kwargs["preserve_thinking"] = True
        return kwargs

    def _is_qwen3_5_model(self) -> bool:
        hf_checkpoint = getattr(self.args, "hf_checkpoint", None)
        if not hf_checkpoint:
            return False
        config_path = Path(hf_checkpoint) / "config.json"
        if not config_path.exists():
            return False
        try:
            with config_path.open(encoding="utf-8") as f:
                model_type = str((json.load(f) or {}).get("model_type", "")).lower()
        except (OSError, json.JSONDecodeError):
            return False
        return model_type in {"qwen3_5", "qwen3.5"} or model_type.startswith(("qwen3_5", "qwen3.5"))

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)

    sampling_params["max_new_tokens"] -= sample.response_length

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    predictive_top_k = int(getattr(args, "dppo_predictive_top_k", 0) or 0)
    if predictive_top_k < 0:
        raise ValueError(f"dppo_predictive_top_k must be non-negative, got {predictive_top_k}")
    if predictive_top_k:
        payload["top_logprobs_num"] = predictive_top_k

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    lora_path = _rollout_lora_path(args, state.active_lora_name)
    if lora_path is not None:
        payload["lora_path"] = lora_path

    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
    if images:
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]
        # For single-turn multimodal requests, send text so SGLang expands the
        # image placeholders with its own processor rules.
        payload["text"] = sample.prompt
    else:
        payload["input_ids"] = prompt_ids

    if not sample.tokens:
        sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing (SGLang Model Gateway)
    headers = None
    if sample.session_id:
        if getattr(args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": sample.session_id}

    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}) as span:
        output = await post(url, payload, headers=headers)
        span.update(build_sglang_meta_trace_attrs(output["meta_info"]))

    predictive_support = None
    if predictive_top_k:
        (
            new_response_tokens,
            new_response_log_probs,
            support_token_ids,
            support_log_probs,
            support_valid_mask,
        ) = _extract_predictive_support(output["meta_info"], predictive_top_k)
        predictive_support = (support_token_ids, support_log_probs, support_valid_mask)
        if output.get("text") and not new_response_tokens:
            raise ValueError(
                "SGLang returned non-empty response text without output_token_logprobs "
                "while predictive-mask support is enabled"
            )
    elif "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens, new_response_log_probs = [], []

    previous_response_length = sample.response_length
    if predictive_support is not None:
        _append_predictive_support(
            sample,
            predictive_support,
            previous_response_length=previous_response_length,
            top_k=predictive_top_k,
        )

    sample.append_response_tokens(
        args,
        tokens=new_response_tokens,
        log_probs=new_response_log_probs,
        trainable=True,
        meta_info=output["meta_info"],
        text=output["text"],
    )

    return sample


@trace_function("generate_and_rm", target="sample")
async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            # Multi-turn contract: every element of the gathered group must be
            # list[Sample] with per-sample turn_idx (_split_turns_as_sample_groups
            # asserts both). This semaphore-abort path bypasses the custom
            # generate's _abort_result (which builds a proper aborted turn list),
            # so a bare Sample here crashed rollout 2 of formal r7e (2026-07-10)
            # when the dynamic-sampling fill aborted queued tasks.
            if getattr(args, "use_multi_turn", False):
                sample.metadata = {**(sample.metadata or {}), "turn_idx": 0}
                return [sample]
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    sample = await apply_rollout_sample_hooks(args, sample, evaluation=evaluation)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    if isinstance(sample, list):
        samples = sample
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            return samples

        samples_need_reward = [sample for sample in samples if sample.reward is None]
        with trace_span(samples_need_reward, "reward_model"):
            rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # Some custom generate paths may have already filled the reward.
        if sample.reward is None:
            with trace_span(sample, "reward_model"):
                sample.reward = await async_rm(args, sample)

    return sample


@trace_function(
    "generate_and_rm_group",
    target="group",
    attrs_getter=lambda args, group, sampling_params, evaluation=False: {"group_size": len(group)},
)
async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample] | list[list[Sample]]:
    # ``generate_and_rm`` may return either a ``Sample`` or a ``list[Sample]``
    # depending on whether the ``--custom-generate-function-path`` callable
    # emits one trainable sample or several (e.g. multi-turn agent rollouts
    # that fan out into multiple prefix-chained samples). The asyncio.gather
    # below preserves whichever shape each task produced, so the group is
    # ``list[Sample]`` for plain rollouts and ``list[list[Sample]]`` for
    # the fan-out case.
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    if getattr(args, "use_multi_turn", False):
        group = _split_turns_as_sample_groups(group)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        assert group and all(
            isinstance(sample, Sample) for sample in group
        ), "Group RM requires all samples to be valid"
        with trace_span(group, "group_reward_model"):
            rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


def _split_turns_as_sample_groups(group: list[Sample] | list[list[Sample]]) -> list[list[Sample]]:
    assert group and all(isinstance(samples, list) for samples in group), (
        "--use-multi-turn requires custom generate to return list[Sample] for each original sample, "
        f"got group={type(group).__name__}"
    )

    turn_groups: dict[int, list[Sample]] = {}
    for sample_outputs in group:
        assert sample_outputs and all(
            isinstance(sample, Sample) for sample in sample_outputs
        ), "--use-multi-turn expects list[list[Sample]] after generate_and_rm_group"
        for sample in sample_outputs:
            turn_idx = sample.metadata.get("turn_idx") if sample.metadata is not None else None
            assert turn_idx is not None, "--use-multi-turn requires sample.metadata['turn_idx']"
            turn_groups.setdefault(int(turn_idx), []).append(sample)

    sorted_turn_indices = sorted(turn_groups)
    return [turn_groups[turn_idx] for turn_idx in sorted_turn_indices]


def _groups_missing_routing_replay(args: Namespace, groups: list[list[Sample]]) -> bool:
    """True if routing replay is on and any sample lacks its routed-experts record.

    Failed/aborted generations come back without ``routed_experts`` in meta_info;
    such samples would crash the replay consumer downstream (``torch.from_numpy(None)``)
    and post-hoc dropping underfills the fixed global batch (num_steps floors to 0).
    Rejecting the group HERE lets over-sampling refill it, keeping the batch exact.
    Pad turns are fine: they carry an EMPTY (not None) routed array.
    """
    if not getattr(args, "use_rollout_routing_replay", False):
        return False
    return any(
        getattr(sample, "rollout_routed_experts", None) is None
        for group in groups
        for sample in (group if isinstance(group, list) else [group])
    )


def _get_last_non_pad_turn_group(groups: list[list[Sample]]) -> list[Sample]:
    """Return each trajectory's last real turn sample, excluding padded turns."""
    last_turn_group: list[Sample | None] = [None] * len(groups[0])
    remaining = len(last_turn_group)

    for group in reversed(groups):
        for i, sample in enumerate(group):
            if last_turn_group[i] is not None:
                continue
            if isinstance(sample.metadata, dict) and sample.metadata.get("is_pad_turn"):
                continue
            last_turn_group[i] = sample
            remaining -= 1
        if remaining == 0:
            break

    return [sample if sample is not None else groups[-1][i] for i, sample in enumerate(last_turn_group)]


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
    urls = [worker["url"] for worker in response["workers"]]

    await abort_servers_until_idle(urls)

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


def _get_over_sampling_fetch_size(args: Namespace, target_data_size: int, accepted_count: int) -> int:
    """Return the number of *prompt groups* to submit on the next refill.

    ``over_sampling_batch_size`` remains the hard cap and, unless an adaptive
    factor is configured, preserves slime's historical fixed-granularity
    behavior.  With ``over_sampling_refill_factor=2``, a rollout that still
    needs ``k`` accepted prompt groups submits at most ``2 * k`` candidates.
    Each candidate is expanded separately to ``n_samples_per_prompt``
    completions by the data source.
    """
    max_prompt_groups = int(args.over_sampling_batch_size)
    refill_factor = getattr(args, "over_sampling_refill_factor", None)
    if refill_factor is None:
        return max_prompt_groups

    missing_groups = target_data_size - accepted_count
    if missing_groups <= 0:
        return 0
    return min(max_prompt_groups, int(refill_factor) * missing_groups)


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    use_multi_turn = getattr(args, "use_multi_turn", False)
    max_turns = getattr(args, "max_turns", None)
    filter_by_last_turn = use_multi_turn and getattr(args, "filter_by_last_turn", False)

    if use_multi_turn:
        assert max_turns is not None, "--max-turns must be set when --use-multi-turn is enabled"
        max_turns = int(max_turns)
        assert max_turns >= 1, "--max-turns must be >= 1"

    # target_data_size is the number of original prompt groups to accept.
    # In multi-turn mode, one accepted prompt group can contribute multiple turn groups to data.
    target_data_size = args.rollout_batch_size

    accepted_count = 0
    data = []
    all_data = []
    do_print = True
    # Routing-replay reject-and-refill bookkeeping (see the check inside the loop).
    missing_routing_reject_count = 0
    missing_routing_reject_limit = max(4 * target_data_size, 64)
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while accepted_count < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            # If no adaptive factor is configured this keeps slime's fixed
            # over_sampling_batch_size behavior. The formal DS-V4 path uses factor=2,
            # so a tail deficit of k accepted groups refills only 2*k prompt
            # groups instead of launching another full 32-prompt wave.
            fetch_prompt_groups = _get_over_sampling_fetch_size(args, target_data_size, accepted_count)
            assert fetch_prompt_groups > 0
            logger.info(
                "Rollout candidate refill: accepted_groups=%s missing_groups=%s "
                "candidate_groups_remaining=%s fetch_prompt_groups=%s "
                "samples_per_prompt=%s completion_requests=%s max_prompt_groups=%s refill_factor=%s",
                accepted_count,
                target_data_size - accepted_count,
                state.remaining_batch_size,
                fetch_prompt_groups,
                args.n_samples_per_prompt,
                fetch_prompt_groups * args.n_samples_per_prompt,
                args.over_sampling_batch_size,
                getattr(args, "over_sampling_refill_factor", None),
            )
            samples = data_source(fetch_prompt_groups)
            _set_rollout_step_metadata(args, rollout_id, samples)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task_group = task.result()
            if not task_group:
                state.remaining_batch_size -= 1
                assert state.remaining_batch_size >= 0
                continue
            groups: list[list[Sample]] = task_group if isinstance(task_group[0], list) else [task_group]
            is_filtered = True

            last_turn_dynamic_filter_output = None
            if filter_by_last_turn:
                last_turn_group = _get_last_non_pad_turn_group(groups)
                last_turn_dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, last_turn_group)
            all_data.extend(groups)

            # Routing-replay reject-and-refill: a missing routed-experts record ANYWHERE
            # in the task_group rejects the WHOLE trajectory atomically (never a subset —
            # partially accepting sibling turn-groups would ship an incomplete trajectory
            # and the post-hoc drop would underfill the fixed global batch). is_filtered
            # stays True so remaining_batch_size is decremented and over-sampling refills.
            # NOTE: the groups stay in all_data on purpose — all_data holds every generated
            # sample including dynamic-filter-dropped ones; consumers of
            # --rollout-all-samples-process-path must not assume routed_experts is present.
            if _groups_missing_routing_replay(args, groups):
                for _ in groups:
                    metric_gatherer.on_dynamic_filter_drop(reason="missing_routing_replay")
                missing_routing_reject_count += 1
                # Livelock guard: a persistent stream of routing-less groups means the
                # engine isn't returning routed_experts at all — fail loudly instead of
                # resampling forever.
                if missing_routing_reject_count >= missing_routing_reject_limit:
                    raise RuntimeError(
                        f"{missing_routing_reject_count} rollout groups were missing "
                        "rollout_routed_experts while --use-rollout-routing-replay is enabled "
                        f"(accepted {accepted_count}/{target_data_size}). The rollout engine is "
                        "likely not configured to return routed experts (check SGLANG "
                        "routed-experts capture)."
                    )
                state.remaining_batch_size -= 1
                assert state.remaining_batch_size >= 0
                continue

            for group in groups:
                assert group and all(
                    isinstance(sample, Sample) for sample in group
                ), f"Rollout group must be list[Sample], got {type(group).__name__}"

                if do_print:
                    sample = group[0]
                    logger.info(
                        f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                    )
                    do_print = False

                if last_turn_dynamic_filter_output is not None:
                    # Use the last turn's filter result to keep or drop the whole multi-turn trajectory.
                    if should_drop_dynamic_filter_output(
                        last_turn_dynamic_filter_output,
                        remaining_batch_size=state.remaining_batch_size,
                        target_data_size=target_data_size,
                    ):
                        for _ in groups:
                            metric_gatherer.on_dynamic_filter_drop(reason=last_turn_dynamic_filter_output.reason)
                    elif accepted_count < target_data_size:
                        data.extend(groups)
                        is_filtered = False
                    break
                else:
                    dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)

                    if should_drop_dynamic_filter_output(
                        dynamic_filter_output,
                        remaining_batch_size=state.remaining_batch_size,
                        target_data_size=target_data_size,
                    ):
                        metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                        continue
                    # add the samples to the data
                    # NOTE: here we have not stored all the unused samples back to the data buffer.
                    if accepted_count < target_data_size:
                        data.append(group)
                        is_filtered = False

            if is_filtered:
                state.remaining_batch_size -= 1
                assert state.remaining_batch_size >= 0
            else:
                accepted_count += 1
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert accepted_count == target_data_size, f"Got {accepted_count} samples, expected {target_data_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    eval_multimodal_keys = (
        dataset_cfg.multimodal_keys if dataset_cfg.multimodal_keys is not None else args.multimodal_keys
    )
    eval_apply_chat_template = (
        dataset_cfg.apply_chat_template if dataset_cfg.apply_chat_template is not None else args.apply_chat_template
    )
    eval_apply_chat_template_kwargs = (
        dataset_cfg.apply_chat_template_kwargs
        if dataset_cfg.apply_chat_template_kwargs is not None
        else args.apply_chat_template_kwargs
    )

    cache_key = dataset_cfg.cache_key + (
        args.hf_checkpoint,
        eval_apply_chat_template,
        json.dumps(eval_multimodal_keys, sort_keys=True) if eval_multimodal_keys is not None else None,
        (
            json.dumps(eval_apply_chat_template_kwargs, sort_keys=True)
            if eval_apply_chat_template_kwargs is not None
            else None
        ),
    )
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=eval_multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=eval_apply_chat_template,
            apply_chat_template_kwargs=eval_apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=dataset_cfg.stop if dataset_cfg.stop is not None else args.rollout_stop,
        stop_token_ids=(
            dataset_cfg.stop_token_ids if dataset_cfg.stop_token_ids is not None else args.rollout_stop_token_ids
        ),
        skip_special_tokens=(
            dataset_cfg.skip_special_tokens
            if dataset_cfg.skip_special_tokens is not None
            else args.rollout_skip_special_tokens
        ),
        no_stop_trim=dataset_cfg.no_stop_trim if dataset_cfg.no_stop_trim is not None else True,
        spaces_between_special_tokens=False,
    )
    if dataset_cfg.repetition_penalty is not None:
        base_sampling_params["repetition_penalty"] = dataset_cfg.repetition_penalty
    min_new_tokens = dataset_cfg.min_new_tokens
    if min_new_tokens is None:
        min_new_tokens = getattr(args, "eval_min_new_tokens", None)
    if min_new_tokens is not None:
        base_sampling_params["min_new_tokens"] = min_new_tokens

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.custom_rm_path = dataset_cfg.custom_rm_path
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logged_sample = sample[0] if isinstance(sample, list) else sample
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(logged_sample.prompt) + logged_sample.response]} "
                f"reward={logged_sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
