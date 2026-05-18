import re
from argparse import Namespace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")
pytest.importorskip("jinja2")
pytest.importorskip("torch")
pytest.importorskip("yaml")

import yaml
from datasets import Dataset as HfDataset
from jinja2 import Environment

from slime_plugins.drkernel.rollout import DrKernelPromptRenderer


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_ROOT = _REPO_ROOT / "slime_plugins" / "drkernel" / "prompt_templates"
_REAL_REVIEW_DATA_PATH = _REPO_ROOT / "data" / "drkernel-rl-data-0513" / "train.parquet"
_PROMPT_REVIEW_PATH = _REPO_ROOT / "checkpoints" / "drkernel_prompt_debug" / "formatted_prompt_examples.txt"
_PROMPT_REVIEW_SEPARATOR = "#" * 100

_LEGACY_FIRST_TURN = {
    "csl_cuda_agent": _TEMPLATE_ROOT / "legacy" / "first_turn" / "csl_cuda_agent.jinja",
    "lhb_v3": _TEMPLATE_ROOT / "legacy" / "first_turn" / "lhb_v3.jinja",
    "lhb_v4": _TEMPLATE_ROOT / "legacy" / "first_turn" / "lhb_v4.jinja",
    "pybind11_module": _TEMPLATE_ROOT / "legacy" / "first_turn" / "pybind11_module.jinja",
    "tvm_ffi_module": _TEMPLATE_ROOT / "legacy" / "first_turn_tvm_ffi" / "tvm_ffi_module.jinja",
}
_ACTIVE_FIRST_TURN = {key: value for key, value in _LEGACY_FIRST_TURN.items() if key != "lhb_v4"}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _render_template(text: str, **kwargs) -> str:
    return Environment(keep_trailing_newline=False).from_string(text).render(**kwargs)


def _wrap_problem_for_prompt(problem: str) -> str:
    problem_body = problem.strip("\n")
    return f"You are given the following PyTorch model:\n```python\n{problem_body}\n```"


def _load_single_turn_profile():
    config = yaml.safe_load((_TEMPLATE_ROOT / "single_turn_v1.yaml").read_text(encoding="utf-8"))
    return config["profiles"]["drkernel_single_turn_v1"]


@lru_cache(maxsize=1)
def _load_real_review_records(num_records: int):
    if not _REAL_REVIEW_DATA_PATH.exists():
        pytest.skip(f"real DrKernel review data not found: {_REAL_REVIEW_DATA_PATH}")

    dataset = HfDataset.from_parquet(str(_REAL_REVIEW_DATA_PATH))
    if len(dataset) < num_records:
        pytest.skip(f"need at least {num_records} DrKernel review samples, found {len(dataset)}")
    return [dataset[i] for i in range(num_records)]


def _new_prompt_renderer() -> DrKernelPromptRenderer:
    return DrKernelPromptRenderer("/fake/qwen")


class _FakeChatTemplateTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        suffix = "\n<assistant>" if add_generation_prompt else ""
        return f"<chat>\n{messages[0]['content']}{suffix}"


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    tokenizer = _FakeChatTemplateTokenizer()
    monkeypatch.setattr("slime_plugins.drkernel.rollout.load_tokenizer", lambda *_, **__: tokenizer)
    return tokenizer


@pytest.fixture(scope="module")
def prompt_review_file():
    _PROMPT_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROMPT_REVIEW_PATH.write_text("", encoding="utf-8")
    return _PROMPT_REVIEW_PATH


def _format_prompt_for_review(
    title: str,
    chosen: dict,
    prompt: str,
    source_metadata: dict | None = None,
    max_lines: int | None = None,
) -> str:
    lines = prompt.splitlines()
    if max_lines is None:
        body = "\n".join(lines)
    else:
        body = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            body += f"\n... <truncated {len(lines) - max_lines} lines>"
    chosen_text = yaml.safe_dump(chosen, allow_unicode=True, sort_keys=True).rstrip()
    source_text = ""
    if source_metadata:
        source_dump = yaml.safe_dump(source_metadata, allow_unicode=True, sort_keys=True).rstrip()
        source_text = f"source metadata:\n{source_dump}\n\n"
    return (
        f"{_PROMPT_REVIEW_SEPARATOR}\n"
        f"# {title}\n"
        f"{_PROMPT_REVIEW_SEPARATOR}\n"
        f"chosen:\n{chosen_text}\n\n"
        f"{source_text}"
        f"formatted prompt:\n{body}\n"
    )


def _render_review_sample(
    role_id: str,
    template_id: str,
    sample_index: int,
    source_record: dict,
    compiler_name: str | None = "nvcc_12_4",
    gpu_name: str | None = "H100",
):
    args = Namespace(
        rollout_seed=123,
        drkernel_compiler_name=compiler_name,
        drkernel_gpu_name=gpu_name,
        drkernel_extra_environment=None,
    )
    metadata = dict(source_record["extra_info"], review_data_index=sample_index - 1)
    metadata["template_allowed"] = {
        "role": [role_id],
        "first_turn_template": [template_id],
    }
    sample = SimpleNamespace(
        group_index=0,
        index=sample_index,
        prompt=source_record["ground_truth"],
        metadata=metadata,
    )

    renderer = _new_prompt_renderer()
    result = renderer.render_sample(args, sample, rollout_id=0)
    return result.chosen, result.prompt, sample.metadata


@pytest.mark.unit
def test_drkernel_single_turn_yaml_is_composable():
    profile = _load_single_turn_profile()

    assert profile["layout"] == "layouts/single_turn.jinja"
    assert profile["role"]["select"] == "cycle"
    assert profile["first_turn_template"]["select"] == "cycle"

    role_ids = {candidate["id"] for candidate in profile["role"]["candidates"]}
    assert role_ids == {"accelerate_best_perf", "optimize_correctness"}

    backend_ids = {candidate["id"] for candidate in profile["first_turn_template"]["candidates"]}
    assert backend_ids == set(_ACTIVE_FIRST_TURN)

    for candidate in profile["role"]["candidates"]:
        assert (_TEMPLATE_ROOT / candidate["text_path"]).exists()
    for candidate in profile["first_turn_template"]["candidates"]:
        assert (_TEMPLATE_ROOT / candidate["backend_text_path"]).exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "template_id,role_id",
    [
        ("csl_cuda_agent", "optimize_correctness"),
        ("pybind11_module", "optimize_correctness"),
        ("tvm_ffi_module", "optimize_correctness"),
    ],
)
def test_drkernel_legacy_equivalence_is_covered_by_tests(template_id, role_id):
    # The runtime YAML is intentionally composable. Fixed legacy combinations
    # should recover old role/backend content while applying the new shared
    # problem wrapper, ignoring whitespace differences and without env injection.
    # lhb_v3 remains a valid backend choice, but it does not have an exact role
    # string among the two runtime role variants.
    profile = _load_single_turn_profile()
    layout = (_TEMPLATE_ROOT / profile["layout"]).read_text(encoding="utf-8")

    role_candidate = next(candidate for candidate in profile["role"]["candidates"] if candidate["id"] == role_id)
    backend_candidate = next(
        candidate for candidate in profile["first_turn_template"]["candidates"] if candidate["id"] == template_id
    )

    role = (_TEMPLATE_ROOT / role_candidate["text_path"]).read_text(encoding="utf-8")
    backend = (_TEMPLATE_ROOT / backend_candidate["backend_text_path"]).read_text(encoding="utf-8")
    problem = "def example_problem(x):\n    return x + 1\n"

    rendered = _render_template(
        layout,
        role=role,
        backend=backend,
        problem=problem,
        compiler_name=None,
        gpu_name=None,
        extra_environment=None,
    )
    legacy_path = _LEGACY_FIRST_TURN[template_id]
    legacy_rendered = _render_template(
        legacy_path.read_text(encoding="utf-8"),
        problem=_wrap_problem_for_prompt(problem),
    )

    assert _normalize_whitespace(rendered) == _normalize_whitespace(legacy_rendered)


@pytest.mark.unit
def test_drkernel_prompt_renderer_applies_template_allowed_and_records_metadata():
    args = Namespace(
        hf_checkpoint="/fake/qwen",
        rollout_seed=123,
        drkernel_compiler_name="nvcc_12_4",
        drkernel_gpu_name="H100",
        drkernel_extra_environment=None,
        apply_chat_template_kwargs={"enable_thinking": False},
    )
    sample = SimpleNamespace(
        group_index=1,
        index=2,
        prompt="class Model:\n    pass\n",
        metadata={
            "template_allowed": {
                "role": ["accelerate_best_perf"],
                "first_turn_template": ["lhb_v3"],
            }
        },
    )

    renderer = _new_prompt_renderer()
    renderer.apply_to_sample(args, sample, rollout_id=3)

    assert sample.metadata["raw_problem"] == "class Model:\n    pass\n"
    assert sample.metadata["chosen_prompt_slots"] == {
        "profile": "drkernel_single_turn_v1",
        "role": "accelerate_best_perf",
        "first_turn_template": "lhb_v3",
        "compiler_name": "nvcc_12_4",
        "gpu_name": "H100",
    }
    user_prompt = sample.metadata["drkernel_user_prompt"]
    assert user_prompt.startswith("You are a PyTorch and CUDA expert. Accelerate")
    assert "Target environment:" in user_prompt
    assert "- NVCC: nvcc_12_4" in user_prompt
    assert "- GPU: H100" in user_prompt
    assert "KEY RULES" in user_prompt
    assert "baseline.\n\n\nTarget environment:" not in user_prompt
    assert "- GPU: H100\n\n\n========================" not in user_prompt
    assert sample.prompt.startswith("<chat>\nYou are a PyTorch and CUDA expert. Accelerate")
    assert sample.prompt.endswith("\n<assistant>")


@pytest.mark.unit
def test_drkernel_prompt_renderer_cycles_candidates_without_allowed_list():
    args = Namespace(
        rollout_seed=123,
        drkernel_compiler_name=None,
        drkernel_gpu_name=None,
        drkernel_extra_environment=None,
    )
    sample = SimpleNamespace(
        group_index=1,
        index=7,
        prompt="class Model:\n    pass\n",
        metadata={},
    )

    renderer = _new_prompt_renderer()
    result = renderer.render_sample(args, sample, rollout_id=3)

    assert result.chosen == {
        "profile": "drkernel_single_turn_v1",
        "role": "optimize_correctness",
        "first_turn_template": "tvm_ffi_module",
    }


@pytest.mark.unit
def test_drkernel_prompt_renderer_prints_tvm_ffi_review_sample():
    args = Namespace(
        rollout_seed=123,
        drkernel_compiler_name="nvcc_12_4",
        drkernel_gpu_name="H100",
        drkernel_extra_environment=None,
    )
    sample = SimpleNamespace(
        group_index=1,
        index=3,
        prompt="class Model:\n    pass\n",
        metadata={
            "template_allowed": {
                "role": ["optimize_correctness"],
                "first_turn_template": ["tvm_ffi_module"],
            }
        },
    )

    renderer = _new_prompt_renderer()
    result = renderer.render_sample(args, sample, rollout_id=3)

    assert result.chosen["first_turn_template"] == "tvm_ffi_module"
    assert "TVM-FFI" in result.prompt
    assert "tvm_ffi_extension" in result.prompt
    assert "correctness.\n\n\nTarget environment:" not in result.prompt
    assert "- GPU: H100\n\n\nDo not use inline CUDA" not in result.prompt


@pytest.mark.unit
def test_drkernel_apply_to_sample_applies_chat_template_after_prompt_rendering(fake_tokenizer):
    args = Namespace(
        hf_checkpoint="/fake/qwen",
        rollout_seed=123,
        drkernel_compiler_name="nvcc_12_4",
        drkernel_gpu_name="H100",
        drkernel_extra_environment=None,
        apply_chat_template_kwargs={"enable_thinking": False},
    )
    sample = SimpleNamespace(
        group_index=1,
        index=2,
        prompt="class Model:\n    pass\n",
        metadata={
            "template_allowed": {
                "role": ["accelerate_best_perf"],
                "first_turn_template": ["lhb_v3"],
            },
            "tools": [{"name": "compile"}],
        },
        tokens=[1, 2, 3],
    )

    renderer = _new_prompt_renderer()
    renderer.apply_to_sample(args, sample, rollout_id=3)
    rendered_user_prompt = sample.metadata["drkernel_user_prompt"]

    assert sample.metadata["drkernel_user_prompt"] == rendered_user_prompt
    assert sample.prompt.startswith("<chat>\nYou are a PyTorch and CUDA expert. Accelerate")
    assert sample.prompt.endswith("\n<assistant>")
    assert sample.tokens == []

    call = fake_tokenizer.calls[0]
    assert call["messages"] == [{"role": "user", "content": rendered_user_prompt}]
    assert call["tools"] == [{"name": "compile"}]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert call["kwargs"] == {"enable_thinking": False}


@pytest.mark.unit
def test_drkernel_prompt_renderer_saves_formatted_review_samples(prompt_review_file):
    profile = _load_single_turn_profile()
    role_ids = [candidate["id"] for candidate in profile["role"]["candidates"]]
    template_ids = [candidate["id"] for candidate in profile["first_turn_template"]["candidates"]]
    combo_count = len(role_ids) * len(template_ids)
    total = combo_count + 1
    review_records = _load_real_review_records(total)

    blocks = []
    sample_index = 0
    for role_id in role_ids:
        for template_id in template_ids:
            sample_index += 1
            source_record = review_records[sample_index - 1]
            chosen, prompt, metadata = _render_review_sample(role_id, template_id, sample_index, source_record)
            title = f"DRKERNEL FORMATTED PROMPT SAMPLE {sample_index:02d}/{total}: {role_id} + {template_id}"
            source_metadata = {
                key: metadata[key] for key in ("review_data_index", "uuid", "ops", "repo_name") if key in metadata
            }
            blocks.append(_format_prompt_for_review(title, chosen, prompt, source_metadata))
            assert source_record["ground_truth"].strip("\n") in prompt
            assert "Target environment:\n- NVCC: nvcc_12_4\n- GPU: H100\n\n\n" not in prompt

    sample_index += 1
    source_record = review_records[sample_index - 1]
    chosen, prompt, metadata = _render_review_sample(
        "accelerate_best_perf",
        "lhb_v3",
        sample_index,
        source_record,
        compiler_name=None,
        gpu_name=None,
    )
    title = (
        f"DRKERNEL FORMATTED PROMPT SAMPLE {sample_index:02d}/{total}: "
        "accelerate_best_perf + lhb_v3 + NVCC/GPU NOT SPECIFIED"
    )
    source_metadata = {
        key: metadata[key] for key in ("review_data_index", "uuid", "ops", "repo_name") if key in metadata
    }
    blocks.append(_format_prompt_for_review(title, chosen, prompt, source_metadata))
    assert source_record["ground_truth"].strip("\n") in prompt
    assert "Target environment:" not in prompt
    assert "compiler_name" not in chosen
    assert "gpu_name" not in chosen

    prompt_review_file.write_text("\n".join(blocks), encoding="utf-8")
    review_text = prompt_review_file.read_text(encoding="utf-8")
    print(f"\nsaved {total} formatted prompt samples: {prompt_review_file}\n")
    first_preview = "\n".join(blocks[0].splitlines()[:40])
    print(f"{first_preview}\n... <truncated review file preview>\n")

    assert total == 9
    assert review_text.count("DRKERNEL FORMATTED PROMPT SAMPLE") == 9
    assert "NVCC/GPU NOT SPECIFIED" in review_text
    assert review_text.count(_PROMPT_REVIEW_SEPARATOR) >= total * 2
    for role_id in role_ids:
        assert f"role: {role_id}" in review_text
    for template_id in template_ids:
        assert f"first_turn_template: {template_id}" in review_text
