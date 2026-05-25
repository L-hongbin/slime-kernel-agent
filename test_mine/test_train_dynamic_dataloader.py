#!/usr/bin/env python3

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from slime.rollout.curriculum_data_source import DynamicCurriculumWrapper
from slime.utils.misc import load_function


DATA_PATH = Path(
    "/nfs/FM/gongoubo/new_project/github/slime-kernel-agent/data/drkernel-cuda-rl-data-curriculum.parquet"
)
HF_CHECKPOINT = "/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B"


def build_train_like_args():
    return SimpleNamespace(
        # train.py -> create_rollout_manager -> RolloutManager data source loading
        data_source_path="slime.rollout.data_source.RolloutDataSourceWithBuffer",
        use_dynamic_curriculum=True,
        # dataset
        rollout_global_dataset=True,
        prompt_data=str(DATA_PATH),
        hf_checkpoint=HF_CHECKPOINT,
        input_key="prompt",
        label_key=None,
        metadata_key=None,
        tool_key="tools",
        multimodal_keys=None,
        # tokenizer / processor / chat template
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_max_prompt_len=4096,
        # rollout sampling
        n_samples_per_prompt=4,
        rollout_seed=42,
        rollout_shuffle=False,
        # buffer
        buffer_filter_path=None,
        # save/load
        save="./tmp_curriculum/save",
        load=None,
        dump_details=None,
        # curriculum
        difficulty_level_key="difficulty_level",
        difficulty_score_key="difficulty_score",
    )


def build_data_source_like_rollout_manager(args):
    data_source_cls = load_function(args.data_source_path)
    data_source = data_source_cls(args)

    if getattr(args, "use_dynamic_curriculum", False):
        data_source = DynamicCurriculumWrapper(
            args=args,
            base_data_source=data_source,
        )

    return data_source


def assert_groups(args, groups, allowed_levels):
    assert len(groups) == 8
    assert all(len(group) == args.n_samples_per_prompt for group in groups)

    levels = []
    for group in groups:
        sample = group[0]
        metadata = sample.metadata or {}
        level = metadata.get(args.difficulty_level_key)
        score = metadata.get(args.difficulty_score_key)
        levels.append(level)

        assert level is not None
        assert score is not None
        assert level in allowed_levels
        assert all(sample.metadata == item.metadata for item in group)

    return levels


def main():
    assert DATA_PATH.exists(), f"Dataset does not exist: {DATA_PATH}"

    args = build_train_like_args()
    data_source = build_data_source_like_rollout_manager(args)

    assert isinstance(data_source, DynamicCurriculumWrapper)
    assert len(data_source) > 0

    bucket_sizes = {level: len(rows) for level, rows in sorted(data_source.curriculum_buckets.items())}
    print("[OK] dataset length =", len(data_source))
    print("[OK] curriculum bucket sizes =", bucket_sizes)
    assert {"L0", "L1", "L2", "L3", "L4", "L5"}.issubset(bucket_sizes)

    stage_checks = [
        (0, {"L0", "L1"}),
        (100, {"L0", "L1", "L2", "L3"}),
        (200, {"L0", "L1", "L2", "L3", "L4", "L5"}),
    ]

    for rollout_id, allowed_levels in stage_checks:
        groups = data_source.get_samples(num_samples=8, rollout_id=rollout_id)
        levels = assert_groups(args, groups, allowed_levels)
        print(f"[OK] rollout_id={rollout_id} levels={levels}")

    print("\n[PASS] train-like dynamic dataloader test passed.")


if __name__ == "__main__":
    main()
