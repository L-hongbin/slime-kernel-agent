import json

from slime.utils.data import Dataset


class DummyTokenizer:
    def __call__(self, prompts, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [[1] * len(prompt.split()) for prompt in prompts]}


class RaisingProcessor:
    def __call__(self, *args, **kwargs):
        raise AssertionError("text-only prompts should not call the processor")


def test_text_only_dataset_ignores_processor(tmp_path):
    data_path = tmp_path / "data.jsonl"
    row = {
        "ground_truth": "write a cuda kernel",
        "extra_info": {"id": 0},
    }
    data_path.write_text(json.dumps(row) + "\n")

    dataset = Dataset(
        data_path.as_posix(),
        tokenizer=DummyTokenizer(),
        processor=RaisingProcessor(),
        max_length=32,
        prompt_key="ground_truth",
        label_key="ground_truth",
        metadata_key="extra_info",
    )

    assert len(dataset) == 1
    assert dataset[0].prompt == "write a cuda kernel"
    assert dataset[0].multimodal_inputs is None
