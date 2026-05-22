import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from slime.rollout.curriculum_data_source import DynamicCurriculumDataSource


DATA_PATH = Path(
    "/nfs/FM/gongoubo/new_project/github/slime-kernel-agent/data/drkernel-cuda-rl-data-curriculum.parquet"
)


def build_args(path: Path):
    return SimpleNamespace(
        # dataset
        rollout_global_dataset=True,
        prompt_data=str(path),
        hf_checkpoint="/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B",

        # jsonl keys
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


def main():
    path = DATA_PATH
    assert path.exists(), f"Dataset does not exist: {path}"

    args = build_args(path)
    ds = DynamicCurriculumDataSource(args)
    assert len(ds) > 0

    for rollout_id in [0, 50, 100, 200, 300, 500]:
        print("\n==============================")
        print("rollout_id =", rollout_id)

        groups = ds.get_samples(
            num_samples=8,
            rollout_id=rollout_id,
        )

        for i, group in enumerate(groups):
            sample = group[0]
            metadata = sample.metadata or {}
            prompt = str(sample.prompt)
            prompt_preview = prompt[:200] + ("..." if len(prompt) > 200 else "")

            print(
                i,
                "level=",
                metadata.get("difficulty_level"),
                "score=",
                metadata.get("difficulty_score"),
                "prompt_preview=",
                prompt_preview,
                "label=",
                sample.label,
                "group_index=",
                sample.group_index,
                "sample_index=",
                sample.index,
                "copies=",
                len(group),
            )

        assert len(groups) == 8
        assert all(len(group) == args.n_samples_per_prompt for group in groups)
        assert all(hasattr(group[0], "prompt") for group in groups)
        assert all(hasattr(group[0], "metadata") for group in groups)
        assert all(group[0].metadata.get("difficulty_level") is not None for group in groups)
        assert all(group[0].metadata.get("difficulty_score") is not None for group in groups)

    print("\n[PASS] DynamicCurriculumDataSource test passed.")


if __name__ == "__main__":
    main()
