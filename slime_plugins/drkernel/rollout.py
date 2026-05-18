from __future__ import annotations

import asyncio
import copy
import logging
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment
from tqdm import tqdm
from transformers import AutoTokenizer

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, abort, generate_and_rm
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

__all__ = [
    "DrKernelPromptRenderer",
    "PromptRenderResult",
    "generate_rollout",
    "generate_rollout_async",
]


# Fixed composable prompt pack for this rollout plugin (not CLI-configurable).
_PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "prompt_templates" / "single_turn_v1.yaml"


@dataclass(frozen=True)
class PromptRenderResult:
    prompt: str
    chosen: dict[str, Any]


def _get_arg(args: Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def ensure_no_pre_chat_template(args: Namespace) -> None:
    if args.apply_chat_template:
        raise ValueError(
            "DrKernel rollout applies tokenizer chat template after dynamic prompt rendering; remove --apply-chat-template from the run args."
        )


class DrKernelPromptRenderer:
    """Render DrKernel first-turn prompts from the composable prompt YAML.

    The runtime YAML intentionally describes a composable search space:
    - role is selected independently;
    - first_turn_template selects the backend/rule body;
    - compiler/GPU info is injected from args/metadata as plain strings.

    Legacy equivalence is covered in tests by fixing specific role/backend pairs.
    """

    def __init__(self, hf_checkpoint: str) -> None:
        self.template_root = _PROMPT_CONFIG_PATH.parent
        self.config: dict = yaml.safe_load(_PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.profile_name = "drkernel_single_turn_v1"
        profiles = self.config.get("profiles", {})
        if self.profile_name not in profiles:
            raise KeyError(f"Unknown DrKernel prompt profile: {self.profile_name}")
        self.profile: dict = profiles[self.profile_name]

        self.jinja_env = Environment(keep_trailing_newline=False)
        self._text_cache: dict[Path, str] = {}
        self.tokenizer: AutoTokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)

    def render_sample(self, args: Namespace, sample: Any, rollout_id: int) -> PromptRenderResult:
        metadata = sample.metadata

        problem = sample.prompt
        if not isinstance(problem, str):
            raise TypeError(
                "DrKernel prompt renderer expects a raw string problem. Do not apply chat template before custom rollout."
            )

        role_candidate = self._select_candidate(
            rollout_id=rollout_id,
            sample=sample,
            slot_name="role",
        )
        backend_candidate = self._select_candidate(
            rollout_id=rollout_id,
            sample=sample,
            slot_name="first_turn_template",
        )

        role = self._load_fragment(role_candidate["text_path"])
        backend = self._load_fragment(backend_candidate["backend_text_path"])
        layout = self._load_fragment(self.profile["layout"])

        compiler_name = metadata.get("compiler_name") or _get_arg(args, "drkernel_compiler_name")
        gpu_name = metadata.get("gpu_name") or _get_arg(args, "drkernel_gpu_name")
        extra_environment = metadata.get("extra_environment") or _get_arg(args, "drkernel_extra_environment")

        prompt = self._render_template(
            layout,
            role=role,
            backend=backend,
            problem=problem,
            compiler_name=compiler_name,
            gpu_name=gpu_name,
            extra_environment=extra_environment,
        )

        chosen = {
            "profile": self.profile_name,
            "role": role_candidate["id"],
            "first_turn_template": backend_candidate["id"],
        }
        if compiler_name:
            chosen["compiler_name"] = compiler_name
        if gpu_name:
            chosen["gpu_name"] = gpu_name
        if extra_environment:
            chosen["extra_environment"] = extra_environment
        return PromptRenderResult(prompt=prompt, chosen=chosen)

    def apply_to_sample(self, args: Namespace, sample: Any, rollout_id: int) -> Any:
        result = self.render_sample(args, sample, rollout_id)
        sample.metadata["raw_problem"] = sample.prompt
        sample.metadata["chosen_prompt_slots"] = result.chosen
        sample.metadata["drkernel_user_prompt"] = result.prompt

        messages = [{"role": "user", "content": result.prompt}]
        tools = sample.metadata.get("tools")
        sample.prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **(_get_arg(args, "apply_chat_template_kwargs", {}) or {}),
        )
        if not isinstance(sample.prompt, str):
            raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return a string.")
        if hasattr(sample, "tokens"):
            sample.tokens = []
        return sample

    def _load_fragment(self, relative_path: str) -> str:
        path = (self.template_root / relative_path).resolve()
        if path not in self._text_cache:
            self._text_cache[path] = path.read_text(encoding="utf-8")
        return self._text_cache[path]

    def _render_template(self, text: str, **kwargs: Any) -> str:
        return self.jinja_env.from_string(text).render(**kwargs)

    def _select_candidate(
        self,
        *,
        rollout_id: int,
        sample: Any,
        slot_name: str,
    ) -> dict[str, Any]:
        slot_cfg: dict = self.profile[slot_name]
        candidates = list(slot_cfg["candidates"])

        allowed_ids = sample.metadata.get("template_allowed", {}).get(slot_name)
        if allowed_ids is not None:
            if not allowed_ids:
                raise ValueError(f"DrKernel prompt allowed list for {slot_name} is empty")
            allowed = set(allowed_ids)
            known = {candidate["id"] for candidate in candidates}
            unknown = sorted(allowed - known)
            if unknown:
                raise ValueError(
                    f"Unknown DrKernel prompt candidates for {slot_name}: {unknown}. Known: {sorted(known)}"
                )
            candidates = [candidate for candidate in candidates if candidate["id"] in allowed]

        select_mode = slot_cfg.get("select", "fixed")
        if select_mode == "fixed":
            return candidates[0]
        if select_mode == "cycle":
            basis = sample.index if sample.index is not None else rollout_id
            return candidates[int(basis) % len(candidates)]

        raise ValueError(f"Unsupported DrKernel prompt select mode for {slot_name}: {select_mode}")


@lru_cache(maxsize=1)
def _get_prompt_renderer(hf_checkpoint: str) -> DrKernelPromptRenderer:
    """Return a cached renderer with tokenizer loaded during construction.

    Importing this plugin remains side-effect-light, while rollout/eval paths
    pay tokenizer initialization once before samples are rendered.
    """
    return DrKernelPromptRenderer(hf_checkpoint)


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Any]]]
) -> Any:
    """Minimal single-turn DrKernel rollout example.

    This shows how dynamic prompts plug into slime without rewriting SGLang
    generation. The generation/reward path is still the default
    `generate_and_rm_group`.
    """
    assert args.rollout_global_dataset
    ensure_no_pre_chat_template(args)

    renderer = _get_prompt_renderer(args.hf_checkpoint)
    state = GenerateState(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            for group in samples:
                for sample in group:
                    renderer.apply_to_sample(args, sample, rollout_id)

            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list = task.result()

            if do_print:
                sample: Sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
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
    ensure_no_pre_chat_template(args)

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

    renderer = _get_prompt_renderer(args.hf_checkpoint)
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        processor = (
            load_processor(args.hf_checkpoint, trust_remote_code=True) if args.multimodal_keys is not None else None
        )
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=renderer.tokenizer,
            processor=processor,
            max_length=dataset_cfg.max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        _slime_max_context_len=dataset_cfg.max_context_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

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
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params.copy()
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params["sampling_seed"] = args.rollout_seed + j
            renderer.apply_to_sample(args, sample, rollout_id)
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
                f"eval_rollout_single_dataset example data: {[str(logged_sample.prompt) + logged_sample.response]} reward={logged_sample.reward}"
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


def generate_rollout(args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    assert args.rollout_global_dataset
    ensure_no_pre_chat_template(args)
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
