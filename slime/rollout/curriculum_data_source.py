import copy
import logging
import random
from collections import defaultdict

from slime.rollout.data_source import RolloutDataSourceWithBuffer

logger = logging.getLogger(__name__)


CURRICULUM_SCHEDULE = [
    {
        "until": 100,
        "levels": ["L0", "L1"],
        "weights": {"L0": 0.7, "L1": 0.3},
    },
    {
        "until": 200,
        "levels": ["L0", "L1", "L2", "L3"],
        "weights": {"L0": 0.2, "L1": 0.3, "L2": 0.3, "L3": 0.2},
    },
    {
        "until": None,
        "levels": ["L0", "L1", "L2", "L3", "L4", "L5"],
        "weights": {
            "L0": 0.05,
            "L1": 0.1,
            "L2": 0.2,
            "L3": 0.25,
            "L4": 0.25,
            "L5": 0.15,
        },
    },
]


class DynamicCurriculumDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        super().__init__(args)

        self.level_key = getattr(args, "difficulty_level_key", "difficulty_level")
        self.score_key = getattr(args, "difficulty_score_key", "difficulty_score")
        self.rng = random.Random(getattr(args, "rollout_seed", 42))

        self.curriculum_bucket_offsets = defaultdict(int)
        self.curriculum_buckets = self._build_curriculum_buckets()

    def _get_metadata(self, sample):
        return sample.metadata or {}

    def _infer_level(self, sample):
        metadata = self._get_metadata(sample)

        if self.level_key in metadata:
            return metadata[self.level_key]

        score = metadata.get(self.score_key, None)
        if score is None:
            return "L0"

        score = float(score)
        if score < 2.5:
            return "L0"
        if score < 4.0:
            return "L1"
        if score < 5.5:
            return "L2"
        if score < 7.5:
            return "L3"
        if score < 10.5:
            return "L4"
        return "L5"

    def _build_curriculum_buckets(self):
        buckets = defaultdict(list)

        if self.dataset is None:
            return buckets

        for sample in self.dataset.samples:
            level = self._infer_level(sample)
            buckets[level].append(sample)

        for level in buckets:
            self.rng.shuffle(buckets[level])

        for level, rows in sorted(buckets.items()):
            logger.info(f"[DynamicCurriculum] {level}: {len(rows)} samples")

        return buckets

    def _get_stage(self, rollout_id):
        for stage in CURRICULUM_SCHEDULE:
            if stage["until"] is None or rollout_id < stage["until"]:
                return stage
        return CURRICULUM_SCHEDULE[-1]

    def _sample_level(self, stage):
        available_levels = [
            level
            for level in stage["levels"]
            if len(self.curriculum_buckets.get(level, [])) > 0
        ]

        if not available_levels:
            raise RuntimeError(f"No available curriculum buckets for stage={stage}")

        weights = [stage["weights"].get(level, 1.0) for level in available_levels]
        return self.rng.choices(available_levels, weights=weights, k=1)[0]

    def _sample_one_from_level(self, level):
        bucket = self.curriculum_buckets[level]
        offset = self.curriculum_bucket_offsets[level]

        if offset >= len(bucket):
            self.rng.shuffle(bucket)
            offset = 0

        sample = bucket[offset]
        self.curriculum_bucket_offsets[level] = offset + 1
        return sample

    def _repeat_prompt_sample(self, prompt_sample):
        group = []
        for _ in range(self.args.n_samples_per_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.group_index = self.sample_group_index
            sample.index = self.sample_index
            self.sample_index += 1
            group.append(sample)

        self.sample_group_index += 1
        return group

    def get_samples(self, num_samples: int, rollout_id=None):
        # 先消费 partial rollout buffer，保持原始逻辑
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        rollout_id = 0 if rollout_id is None else rollout_id
        stage = self._get_stage(rollout_id)

        level_counter = defaultdict(int)

        for _ in range(num_samples):
            level = self._sample_level(stage)
            prompt_sample = self._sample_one_from_level(level)
            group = self._repeat_prompt_sample(prompt_sample)
            samples.append(group)
            level_counter[level] += 1

        logger.info(
            f"[DynamicCurriculum] rollout_id={rollout_id}, "
            f"levels={stage['levels']}, sampled={dict(level_counter)}"
        )

        return samples


class DynamicCurriculumWrapper:
    def __init__(self, args, base_data_source):
        self.args = args
        self.base_data_source = base_data_source
        self.dataset = base_data_source.dataset
        self.buffer = base_data_source.buffer
        self.level_key = getattr(args, "difficulty_level_key", "difficulty_level")
        self.score_key = getattr(args, "difficulty_score_key", "difficulty_score")
        self.rng = random.Random(getattr(args, "rollout_seed", 42))

        self.sample_group_index = base_data_source.sample_group_index
        self.sample_index = base_data_source.sample_index
        self.curriculum_bucket_offsets = defaultdict(int)
        self.curriculum_buckets = self._build_curriculum_buckets()

    def _get_metadata(self, sample):
        return sample.metadata or {}

    def _infer_level(self, sample):
        metadata = self._get_metadata(sample)

        if self.level_key in metadata:
            return metadata[self.level_key]

        score = metadata.get(self.score_key, None)
        if score is None:
            return "L0"

        score = float(score)
        if score < 2.5:
            return "L0"
        if score < 4.0:
            return "L1"
        if score < 5.5:
            return "L2"
        if score < 7.5:
            return "L3"
        if score < 10.5:
            return "L4"
        return "L5"

    def _build_curriculum_buckets(self):
        buckets = defaultdict(list)

        if self.dataset is None:
            return buckets

        for sample in self.dataset.samples:
            level = self._infer_level(sample)
            buckets[level].append(sample)

        for level in buckets:
            self.rng.shuffle(buckets[level])

        for level, rows in sorted(buckets.items()):
            logger.info(f"[DynamicCurriculum] {level}: {len(rows)} samples")

        return buckets

    def _get_stage(self, rollout_id):
        for stage in CURRICULUM_SCHEDULE:
            if stage["until"] is None or rollout_id < stage["until"]:
                return stage
        return CURRICULUM_SCHEDULE[-1]

    def _sample_level(self, stage):
        available_levels = [
            level
            for level in stage["levels"]
            if len(self.curriculum_buckets.get(level, [])) > 0
        ]

        if not available_levels:
            raise RuntimeError(f"No available curriculum buckets for stage={stage}")

        weights = [stage["weights"].get(level, 1.0) for level in available_levels]
        return self.rng.choices(available_levels, weights=weights, k=1)[0]

    def _sample_one_from_level(self, level):
        bucket = self.curriculum_buckets[level]
        offset = self.curriculum_bucket_offsets[level]

        if offset >= len(bucket):
            self.rng.shuffle(bucket)
            offset = 0

        sample = bucket[offset]
        self.curriculum_bucket_offsets[level] = offset + 1
        return sample

    def _repeat_prompt_sample(self, prompt_sample):
        group = []
        for _ in range(self.args.n_samples_per_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.group_index = self.sample_group_index
            sample.index = self.sample_index
            self.sample_index += 1
            group.append(sample)

        self.sample_group_index += 1
        return group

    def get_samples(self, num_samples: int, rollout_id=None):
        samples = self.base_data_source._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        rollout_id = 0 if rollout_id is None else rollout_id
        stage = self._get_stage(rollout_id)

        level_counter = defaultdict(int)

        for _ in range(num_samples):
            level = self._sample_level(stage)
            prompt_sample = self._sample_one_from_level(level)
            group = self._repeat_prompt_sample(prompt_sample)
            samples.append(group)
            level_counter[level] += 1

        logger.info(
            f"[DynamicCurriculum] rollout_id={rollout_id}, "
            f"levels={stage['levels']}, sampled={dict(level_counter)}"
        )

        self.base_data_source.sample_group_index = self.sample_group_index
        self.base_data_source.sample_index = self.sample_index

        return samples

    def add_samples(self, samples):
        return self.base_data_source.add_samples(samples)

    def save(self, rollout_id):
        return self.base_data_source.save(rollout_id)

    def load(self, rollout_id=None):
        result = self.base_data_source.load(rollout_id)
        self.sample_group_index = self.base_data_source.sample_group_index
        self.sample_index = self.base_data_source.sample_index
        return result

    def __len__(self):
        return len(self.base_data_source)

    def __getattr__(self, name):
        return getattr(self.base_data_source, name)
