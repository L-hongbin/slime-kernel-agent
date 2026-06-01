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
_LEGACY_TVM_FFI_V2_3 = _TEMPLATE_ROOT / "legacy" / "first_turn_tvm_ffi" / "tvm_ffi_module_v2_3.jinja"
_ACTIVE_FIRST_TURN = {key: value for key, value in _LEGACY_FIRST_TURN.items() if key != "lhb_v4"}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _render_template(text: str, **kwargs) -> str:
    return Environment(keep_trailing_newline=False).from_string(text).render(**kwargs)


def _wrap_problem_for_prompt(problem: str) -> str:
    problem_body = problem.strip("\n")
    return f"You are given the following PyTorch model:\n```python\n{problem_body}\n```"


def _load_drkernel_v1_profile():
    config = yaml.safe_load((_TEMPLATE_ROOT / "prompts_v1.yaml").read_text(encoding="utf-8"))
    return config["profiles"]["drkernel_v1"]


def _load_drkernel_v1_tvm_ffi_profile():
    config = yaml.safe_load((_TEMPLATE_ROOT / "prompts_v1.yaml").read_text(encoding="utf-8"))
    return config["profiles"]["drkernel_v1_tvm_ffi"]


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
        "backend": [template_id],
    }
    sample = SimpleNamespace(
        group_index=0,
        index=sample_index,
        prompt=source_record["ground_truth"],
        metadata=metadata,
    )

    renderer = _new_prompt_renderer_with_profile("drkernel_v1")
    result = renderer.render_sample(args, sample, rollout_id=0)
    return result.chosen, result.prompt, sample.metadata


@pytest.mark.unit
def test_drkernel_prompts_yaml_is_composable():
    profile = _load_drkernel_v1_profile()

    assert profile["layout"] == "layouts/first_turn.jinja"
    assert profile["role"]["select"] == "cycle"
    assert profile["backend"]["select"] == "cycle"

    role_ids = {candidate["id"] for candidate in profile["role"]["candidates"]}
    assert role_ids == {"accelerate_best_perf", "optimize_correctness"}

    backend_ids = {candidate["id"] for candidate in profile["backend"]["candidates"]}
    assert backend_ids == set(_ACTIVE_FIRST_TURN)

    for candidate in profile["role"]["candidates"]:
        assert (_TEMPLATE_ROOT / candidate["text_path"]).exists()
    for candidate in profile["backend"]["candidates"]:
        assert (_TEMPLATE_ROOT / candidate["first_turn_text_path"]).exists()
        # Multi-turn coupling: every backend candidate must declare a paired tool_response template
        # so that turn-≥1 output format stays consistent with turn-0 backend.
        assert (
            "tool_response_text_path" in candidate
        ), f"backend candidate {candidate['id']!r} is missing tool_response_text_path"
        assert (_TEMPLATE_ROOT / candidate["tool_response_text_path"]).exists()


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
    profile = _load_drkernel_v1_profile()
    layout = (_TEMPLATE_ROOT / profile["layout"]).read_text(encoding="utf-8")

    role_candidate = next(candidate for candidate in profile["role"]["candidates"] if candidate["id"] == role_id)
    backend_candidate = next(
        candidate for candidate in profile["backend"]["candidates"] if candidate["id"] == template_id
    )

    role = (_TEMPLATE_ROOT / role_candidate["text_path"]).read_text(encoding="utf-8")
    backend = (_TEMPLATE_ROOT / backend_candidate["first_turn_text_path"]).read_text(encoding="utf-8")
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
def test_drkernel_tvm_ffi_v2_3_legacy_matches_active_profile():
    profile = _load_drkernel_v1_tvm_ffi_profile()
    layout = (_TEMPLATE_ROOT / profile["layout"]).read_text(encoding="utf-8")

    role_candidate = next(
        candidate for candidate in profile["role"]["candidates"] if candidate["id"] == "optimize_correctness"
    )
    backend_candidate = profile["backend"]["candidates"][0]

    assert backend_candidate["id"] == "tvm_ffi_module"
    assert backend_candidate["first_turn_text_path"] == "backends/tvm_ffi_module_v2_3.jinja"

    role = (_TEMPLATE_ROOT / role_candidate["text_path"]).read_text(encoding="utf-8")
    backend = (_TEMPLATE_ROOT / backend_candidate["first_turn_text_path"]).read_text(encoding="utf-8")
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
    legacy_rendered = _render_template(
        _LEGACY_TVM_FFI_V2_3.read_text(encoding="utf-8"),
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
                "backend": ["lhb_v3"],
            }
        },
    )

    renderer = _new_prompt_renderer_with_profile("drkernel_v1")
    renderer.apply_to_sample(args, sample, rollout_id=3)

    assert sample.metadata["raw_problem"] == "class Model:\n    pass\n"
    assert sample.metadata["chosen_prompt_slots"] == {
        "profile": "drkernel_v1",
        "role": "accelerate_best_perf",
        "backend": "lhb_v3",
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

    renderer = _new_prompt_renderer_with_profile("drkernel_v1")
    result = renderer.render_sample(args, sample, rollout_id=3)

    assert result.chosen == {
        "profile": "drkernel_v1",
        "role": "optimize_correctness",
        "backend": "tvm_ffi_module",
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
                "backend": ["tvm_ffi_module"],
            }
        },
    )

    renderer = _new_prompt_renderer()
    result = renderer.render_sample(args, sample, rollout_id=3)

    assert result.chosen["backend"] == "tvm_ffi_module"
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
                "backend": ["lhb_v3"],
            },
            "tools": [{"name": "compile"}],
        },
        tokens=[1, 2, 3],
    )

    renderer = _new_prompt_renderer_with_profile("drkernel_v1")
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
    profile = _load_drkernel_v1_profile()
    role_ids = [candidate["id"] for candidate in profile["role"]["candidates"]]
    template_ids = [candidate["id"] for candidate in profile["backend"]["candidates"]]
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
        assert f"backend: {template_id}" in review_text


# ---------------------------------------------------------------------------
# Multi-turn renderer surface: first_turn_messages / tool_response / materialize
# ---------------------------------------------------------------------------


def _new_prompt_renderer_with_profile(profile_name: str) -> DrKernelPromptRenderer:
    renderer = DrKernelPromptRenderer("/fake/qwen")
    renderer.profile_name = profile_name
    renderer.profile = renderer.config["profiles"][profile_name]
    return renderer


@pytest.mark.unit
def test_render_first_turn_messages_seeds_metadata_and_messages_list():
    args = Namespace(
        rollout_seed=0,
        drkernel_compiler_name="nvcc_12_4",
        drkernel_gpu_name="H100",
        drkernel_extra_environment=None,
    )
    sample = SimpleNamespace(
        group_index=0,
        index=0,
        prompt="class Model:\n    pass\n",
        metadata={},
    )

    renderer = _new_prompt_renderer()
    messages = renderer.render_first_turn_messages(args, sample, rollout_id=0)

    # Returned list is exactly one user turn and is stored on the sample.
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert sample.metadata["messages"] is messages

    # Audit metadata is populated for downstream multi-turn driver and dump-details.
    chosen = sample.metadata["chosen_prompt_slots"]
    assert chosen["profile"] == renderer.profile_name
    assert chosen["backend"] == "tvm_ffi_module"
    assert chosen["role"] in {"accelerate_best_perf", "optimize_correctness"}
    assert sample.metadata["raw_problem"] == "class Model:\n    pass\n"
    assert sample.metadata["drkernel_user_prompt"] == messages[0]["content"]


@pytest.mark.unit
def test_render_tool_response_message_uses_tvm_ffi_paired_template():
    args = Namespace(
        rollout_seed=0,
        drkernel_compiler_name=None,
        drkernel_gpu_name=None,
        drkernel_extra_environment=None,
    )
    sample = SimpleNamespace(
        group_index=0,
        index=0,
        prompt="class Model:\n    pass\n",
        metadata={},
    )

    # Default renderer's active profile is drkernel_v1_tvm_ffi, so turn-0 backend is tvm_ffi_module.
    renderer = _new_prompt_renderer()
    renderer.render_first_turn_messages(args, sample, rollout_id=0)

    feedback = "compile error: undefined symbol my_kernel_launcher in apply_bindings.cpp:24"
    tool_msg = renderer.render_tool_response_message(args, sample, feedback=feedback)

    assert tool_msg["role"] == "user"
    assert feedback in tool_msg["content"]
    # tvm_ffi tool_response template namedrops TVM-FFI; pybind variant would not.
    assert "TVM-FFI" in tool_msg["content"]
    assert "tvm_ffi_extension" in tool_msg["content"]


@pytest.mark.unit
def test_render_tool_response_message_uses_pybind_paired_template():
    args = Namespace(
        rollout_seed=0,
        drkernel_compiler_name=None,
        drkernel_gpu_name=None,
        drkernel_extra_environment=None,
    )
    sample = SimpleNamespace(
        group_index=0,
        index=0,
        prompt="class Model:\n    pass\n",
        metadata={"template_allowed": {"backend": ["pybind11_module"]}},
    )

    # Switch to the multi-backend profile so we can force the pybind path.
    renderer = _new_prompt_renderer_with_profile("drkernel_v1")
    renderer.render_first_turn_messages(args, sample, rollout_id=0)
    assert sample.metadata["chosen_prompt_slots"]["backend"] == "pybind11_module"

    tool_msg = renderer.render_tool_response_message(args, sample, feedback="undefined symbol my_kernel")

    assert "undefined symbol my_kernel" in tool_msg["content"]
    assert "CUDA_KERNELS" in tool_msg["content"]
    # The pybind variant must not pull in tvm_ffi-specific verbiage.
    assert "TVM-FFI" not in tool_msg["content"]
    assert "tvm_ffi_extension" not in tool_msg["content"]


@pytest.mark.unit
def test_render_tool_response_message_requires_first_turn_to_have_run():
    args = Namespace(
        rollout_seed=0,
        drkernel_compiler_name=None,
        drkernel_gpu_name=None,
        drkernel_extra_environment=None,
    )
    sample = SimpleNamespace(
        group_index=0,
        index=0,
        prompt="class Model:\n    pass\n",
        metadata={},  # no chosen_prompt_slots
    )

    renderer = _new_prompt_renderer()
    with pytest.raises(RuntimeError, match="render_first_turn_messages"):
        renderer.render_tool_response_message(args, sample, feedback="any")


@pytest.mark.unit
def test_render_tool_response_message_errors_when_candidate_lacks_tool_path():
    args = Namespace(
        rollout_seed=0,
        drkernel_compiler_name=None,
        drkernel_gpu_name=None,
        drkernel_extra_environment=None,
    )
    sample = SimpleNamespace(
        group_index=0,
        index=0,
        prompt="class Model:\n    pass\n",
        metadata={},
    )

    renderer = _new_prompt_renderer()
    renderer.render_first_turn_messages(args, sample, rollout_id=0)

    # Drop the tool_response_text_path field from the chosen backend candidate to simulate a misconfigured profile.
    chosen_backend = sample.metadata["chosen_prompt_slots"]["backend"]
    for candidate in renderer.profile["backend"]["candidates"]:
        if candidate["id"] == chosen_backend:
            del candidate["tool_response_text_path"]

    with pytest.raises(KeyError, match="tool_response_text_path"):
        renderer.render_tool_response_message(args, sample, feedback="any")


@pytest.mark.unit
def test_materialize_prompt_passes_full_messages_list_to_tokenizer(fake_tokenizer):
    args = Namespace(
        rollout_seed=0,
        drkernel_compiler_name=None,
        drkernel_gpu_name=None,
        drkernel_extra_environment=None,
        apply_chat_template_kwargs={"enable_thinking": False},
    )
    sample = SimpleNamespace(
        group_index=0,
        index=0,
        prompt="class Model:\n    pass\n",
        metadata={},
        tokens=[7, 8, 9],
    )

    renderer = _new_prompt_renderer()
    messages = renderer.render_first_turn_messages(args, sample, rollout_id=0)
    messages.append({"role": "assistant", "content": "<think>plan</think>\nfirst draft answer"})
    messages.append(renderer.render_tool_response_message(args, sample, feedback="compile_error"))

    renderer.materialize_prompt(args, sample, messages)

    last_call = fake_tokenizer.calls[-1]
    assert [m["role"] for m in last_call["messages"]] == ["user", "assistant", "user"]
    assert last_call["add_generation_prompt"] is True
    assert last_call["tokenize"] is False
    assert last_call["kwargs"] == {"enable_thinking": False}
    # Materialize clears stale rollout tokens so they don't leak into the next-turn submit.
    assert sample.tokens == []


@pytest.mark.unit
def test_tool_response_text_path_pairs_each_backend_with_matching_output_format():
    """Sanity guard: every backend's tool_response template references the same output sections."""
    config = yaml.safe_load((_TEMPLATE_ROOT / "prompts_v1.yaml").read_text(encoding="utf-8"))
    pybind_backends = {"csl_cuda_agent", "lhb_v3", "pybind11_module"}
    tvm_ffi_backends = {"tvm_ffi_module"}

    for profile_name, profile in config["profiles"].items():
        for candidate in profile["backend"]["candidates"]:
            tool_path = candidate["tool_response_text_path"]
            body = (_TEMPLATE_ROOT / tool_path).read_text(encoding="utf-8")
            # Both families ask the model to return the three-section response.
            assert "CUDA_KERNELS" in body
            assert "APPLY_BINDINGS" in body
            assert "MODEL_NEW" in body
            assert "{{ feedback }}" in body

            if candidate["id"] in tvm_ffi_backends:
                assert "TVM-FFI" in body, f"{profile_name}/{candidate['id']} should namedrop TVM-FFI in tool_response"
            elif candidate["id"] in pybind_backends:
                assert (
                    "TVM-FFI" not in body
                ), f"{profile_name}/{candidate['id']} tool_response leaked TVM-FFI verbiage into pybind family"
