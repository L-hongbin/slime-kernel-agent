#!/usr/bin/env python3
# test_slime_data_loading_curriculum.py

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def make_demo_jsonl(path: Path):
    rows = [
        {
            "prompt": [{"role": "user", "content": "计算 1+1"}],
            "label": "2",
            "difficulty_score": 1.0,
            "difficulty_level": "L0",
        },
        {
            "prompt": [{"role": "user", "content": "证明 sqrt(2) 是无理数"}],
            "label": "proof",
            "difficulty_score": 4.5,
            "difficulty_level": "L2",
        },
        {
            "prompt": [{"role": "user", "content": "写一个高性能 CUDA softmax kernel"}],
            "label": "cuda",
            "difficulty_score": 8.0,
            "difficulty_level": "L4",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_slime_args(prompt_data: str, hf_checkpoint: str):
    """
    最小化构造 RolloutDataSource 需要的 args。
    字段名尽量对齐 slime 的 arguments.py / data_source.py。
    """
    return SimpleNamespace(
        # dataset
        prompt_data=prompt_data,
        hf_checkpoint=hf_checkpoint,
        input_key="prompt",
        label_key="label",
        metadata_key="metadata",
        tool_key="tools",
        multimodal_keys=None,

        # chat template
        apply_chat_template=True,
        apply_chat_template_kwargs={},

        # rollout dataset behavior
        rollout_global_dataset=True,
        rollout_shuffle=False,
        rollout_seed=42,
        rollout_max_prompt_len=4096,

        # group sampling
        n_samples_per_prompt=4,

        # buffer
        buffer_filter_path=None,

        # checkpoint state
        save="./tmp_slime_test_save",
        load=None,

        # debug dump
        dump_details=None,
    )


def test_real_slime_data_source(prompt_data: str, hf_checkpoint: str, num_prompts: int):
    """
    真实调用 slime 的数据源。
    需要你在 slime 仓库环境里运行，并且 hf_checkpoint 是本地 Qwen3-4B tokenizer 路径。
    """
    from slime.rollout.data_source import RolloutDataSourceWithBuffer

    args = build_slime_args(prompt_data, hf_checkpoint)
    ds = RolloutDataSourceWithBuffer(args)

    print(f"[OK] dataset length = {len(ds)}")

    groups = ds.get_samples(num_prompts)
    print(f"[OK] fetched prompt groups = {len(groups)}")
    print(f"[OK] samples per prompt = {[len(g) for g in groups]}")

    for group_id, group in enumerate(groups):
        print(f"\n===== group {group_id} =====")
        for sample in group:
            print(
                {
                    "group_index": getattr(sample, "group_index", None),
                    "index": getattr(sample, "index", None),
                    "label": getattr(sample, "label", None),
                    "prompt_len": len(getattr(sample, "tokens", []) or []),
                    "prompt": str(getattr(sample, "prompt", ""))[:120],
                }
            )

    assert len(groups) == num_prompts
    assert all(len(g) == args.n_samples_per_prompt for g in groups)
    print("\n[PASS] slime data loading test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-data", type=str, default="./tmp_slime_test/demo.jsonl")
    parser.add_argument("--hf-checkpoint", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--make-demo", action="store_true")
    parser.add_argument("--test-slime", action="store_true")

    args = parser.parse_args()

    prompt_data = Path(args.prompt_data)

    if args.make_demo:
        make_demo_jsonl(prompt_data)
        print(f"[OK] wrote demo jsonl: {prompt_data}")


    if args.test_slime:
        test_real_slime_data_source(
            prompt_data=str(prompt_data),
            hf_checkpoint=args.hf_checkpoint,
            num_prompts=args.num_prompts,
        )

"""
python3 test_dataloader.py \
  --hf-checkpoint /nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B \
  --make-demo \
  --test-slime
"""


if __name__ == "__main__":
    main()