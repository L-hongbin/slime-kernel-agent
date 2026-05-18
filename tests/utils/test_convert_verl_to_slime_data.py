import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")
pytest.importorskip("pyarrow")
pytest.importorskip("ray")

from datasets import Dataset as HfDataset

import slime.rollout.data_source as data_source_module
from slime.utils.data import Dataset as SlimeDataset


# Import the script by path because scripts/ is not a Python package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONVERTER_PATH = _REPO_ROOT / "scripts" / "data" / "convert_verl_to_slime.py"
_SPEC = importlib.util.spec_from_file_location("convert_verl_to_slime", _CONVERTER_PATH)
convert_verl_to_slime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(convert_verl_to_slime)


class DebugTokenizer:
    """Minimal tokenizer stub for the `--apply-chat-template` path in scripts/debug.sh."""

    def __call__(self, prompts, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [[ord(char) for char in prompt] for prompt in prompts]}

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        assert tools is None
        assert tokenize is False
        rendered = "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages)
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


class TextOnlyShouldNotUseProcessor:
    def __call__(self, *args, **kwargs):
        raise AssertionError("text-only DrKernel samples should not call the VLM processor")


def _verl_sample():
    # Representative VERL-style input: the converter extracts reward_model.ground_truth
    # and folds top-level provenance into extra_info.
    ground_truth = "class Model:\n    pass\n"
    return {
        "data_source": "cuda_llm",
        "ability": "kernel_optimization",
        "prompt": [
            {
                "role": "user",
                "content": f"Optimize the following kernel:\n{ground_truth}",
            }
        ],
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {
            "original_prompt": "removed during conversion",
            "entry_point": "Model",
            "uuid": "cuda_llm_unit_test",
        },
    }


def _convert_sample(sample):
    sample = convert_verl_to_slime.get_ground_truth(copy.deepcopy(sample))
    return convert_verl_to_slime.merge_into_extra_info(sample)


@pytest.mark.unit
def test_verl_converter_emits_debug_script_schema():
    # What this tests:
    # - `reward_model.ground_truth` becomes the top-level `ground_truth` field.
    # - `data_source` and `ability` move into `extra_info`.
    # - VERL-only fields that scripts/debug.sh does not consume are removed.
    converted = _convert_sample(_verl_sample())

    assert set(converted) == {"ground_truth", "extra_info"}
    assert converted["ground_truth"] == "class Model:\n    pass\n"
    assert converted["extra_info"] == {
        "ability": "kernel_optimization",
        "data_source": "cuda_llm",
        "entry_point": "Model",
        "uuid": "cuda_llm_unit_test",
    }


@pytest.mark.unit
def test_slime_dataset_loads_converted_parquet_with_debug_script_keys(tmp_path):
    # What this tests:
    # - the converted record can be serialized to parquet and read by slime.utils.data.Dataset;
    # - scripts/debug.sh keys are honored: `--input-key ground_truth`, `--label-key ground_truth`,
    #   and `--metadata-key extra_info`;
    # - `--apply-chat-template` turns a string ground_truth into a user message plus assistant prompt.
    converted = _convert_sample(_verl_sample())
    parquet_path = tmp_path / "converted.parquet"
    HfDataset.from_list([converted]).to_parquet(str(parquet_path))

    dataset = SlimeDataset(
        str(parquet_path),
        tokenizer=DebugTokenizer(),
        processor=None,
        max_length=None,
        prompt_key="ground_truth",
        label_key="ground_truth",
        metadata_key="extra_info",
        apply_chat_template=True,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.label == "class Model:\n    pass\n"
    assert sample.metadata["data_source"] == "cuda_llm"
    assert sample.metadata["ability"] == "kernel_optimization"
    assert sample.metadata["entry_point"] == "Model"
    assert sample.metadata["uuid"] == "cuda_llm_unit_test"
    assert sample.prompt == "<|im_start|>user\nclass Model:\n    pass\n<|im_end|>\n<|im_start|>assistant\n"


@pytest.mark.unit
def test_rollout_data_source_does_not_load_processor_for_text_only_data(tmp_path, monkeypatch):
    # Qwen3.5 checkpoints expose a Qwen3VLProcessor, but DrKernel's converted
    # ground_truth field is still a raw text problem. RolloutDataSource should
    # avoid passing a processor into Dataset unless multimodal keys are set.
    converted = _convert_sample(_verl_sample())
    parquet_path = tmp_path / "converted.parquet"
    HfDataset.from_list([converted]).to_parquet(str(parquet_path))

    monkeypatch.setattr(data_source_module, "load_tokenizer", lambda *args, **kwargs: DebugTokenizer())
    monkeypatch.setattr(data_source_module, "load_processor", TextOnlyShouldNotUseProcessor())

    data_source = data_source_module.RolloutDataSource(
        SimpleNamespace(
            rollout_global_dataset=True,
            prompt_data=str(parquet_path),
            hf_checkpoint="qwen3.5-debug",
            multimodal_keys=None,
            dump_details=None,
            rollout_max_prompt_len=128,
            input_key="ground_truth",
            label_key="ground_truth",
            metadata_key="extra_info",
            tool_key=None,
            apply_chat_template=False,
            apply_chat_template_kwargs=None,
            rollout_seed=42,
            rollout_shuffle=False,
        )
    )

    dataset = data_source.dataset
    assert dataset is not None
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.prompt == "class Model:\n    pass\n"
    assert sample.label == "class Model:\n    pass\n"
    assert sample.multimodal_inputs is None
