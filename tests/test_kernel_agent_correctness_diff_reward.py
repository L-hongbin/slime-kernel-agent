"""CPU contracts for correctness-only code-diff reward and its matched control."""

import asyncio
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples

examples.__path__ = [str(ROOT / "examples"), *examples.__path__]
from examples.kernel_agent import correctness_diff_reward as credit
from examples.kernel_agent.kernel_reward import reward_post_process_by_group
from slime.utils.types import Sample


def args(**kw):
    return SimpleNamespace(
        **{
            "correctness_diff_mode": "diff",
            "correctness_diff_scale": 0.25,
            "correctness_diff_seed": 42,
            "correctness_diff_max_edit_ratio": 0.1,
            "component_reward": False,
            "advantage_estimator": "trloo",
            "use_multi_turn": True,
            "max_turns": 3,
            "grpo_std_normalization": False,
            **kw,
        }
    )


def response(a=0, b=0):
    return credit.render_sections(
        {
            "CUDA_KERNELS": f"int kernel(int x) {{ int y=x*2; int z=y+3; return z+{a}; }}",
            "APPLY_BINDINGS": "void bind() {}",
            "MODEL_NEW": f"class ModelNew:\n    def forward(self,x):\n        y=x+3\n        return y+{b}",
        }
    )


def far_response():
    return credit.render_sections(
        {
            "CUDA_KERNELS": "int unrelated; " * 80,
            "APPLY_BINDINGS": "void other(int y) {}",
            "MODEL_NEW": "class ModelNew:\n    pass",
        }
    )


def trajectory(gid=0, anchor=1):
    return [
        Sample(
            group_index=0,
            group_id=gid,
            index=gid,
            status=Sample.Status.COMPLETED,
            response=response(int(t > 0), int(t > 1)),
            response_length=3,
            loss_mask=[1, 1, 1],
            tokens=[1, 2, 3],
            label={"ground_truth": "reference"},
            reward=float(t == anchor),
            metadata={
                "turn_idx": t,
                "multi_turn_reward": 1.0,
                "task_id": f"original-{t}",
                "env_result": {
                    "env_state": {
                        "status": "completed",
                        "compiled": True,
                        "correctness": t == anchor,
                        "decoy_kernel": False,
                        "speedup": 0.1,
                    }
                },
            },
        )
        for t in range(3)
    ]


def annotate(rows, **kw):
    return asyncio.run(credit.annotate_trajectory(args(**kw), rows))


def test_first_correct_T2_credits_T1_once_without_mutating_baseline():
    rows = trajectory()
    before = copy.deepcopy(rows)
    assert annotate(rows) is rows
    assert [s.metadata["correctness_diff"]["credit_weight"] for s in rows] == [1, 0, 0]
    for x, y in zip(rows, before, strict=True):
        assert x.reward == y.reward and x.tokens == y.tokens and x.loss_mask == y.loss_mask
        assert {k: v for k, v in x.metadata.items() if k != "correctness_diff"} == y.metadata
        record = x.metadata["correctness_diff"]
        assert record["anchor_evidence_source"] == "original_env_result"
        assert record["anchor_task_id"] == "original-1"
        assert record["anchor_response_hash"] == credit.source_hash(rows[1].response)
        assert "anchor_probe" not in record and "anchor_verified" not in record
    assert credit.credited_returns(args(), rows, [1, 1, 1]) == [1.25, 1, 1]
    assert credit.credited_returns(args(), rows, [1, 1, 1]) == [1.25, 1, 1]
    assert [s.metadata["multi_turn_reward"] for s in rows] == [1, 1, 1]


def test_distinct_versions_share_one_budget():
    rows = trajectory(anchor=2)
    annotate(rows)
    assert [s.metadata["correctness_diff"]["credit_weight"] for s in rows] == [0.5, 0.5, 0]
    assert credit.credited_returns(args(), rows, [1, 1, 1]) == [1.125, 1.125, 1]


def test_duplicate_versions_only_credit_first_eligible_origin():
    rows = trajectory(anchor=2)
    rows[1].response = rows[0].response
    annotate(rows)
    assert [s.metadata["correctness_diff"]["credit_weight"] for s in rows] == [1, 0, 0]
    assert rows[1].metadata["correctness_diff"]["reason"] == "duplicate_earlier_version"


def test_each_section_must_be_near_so_boilerplate_cannot_hide_a_rewrite():
    rows = trajectory()
    sections = credit.selected_sections(rows[1].response)
    sections["MODEL_NEW"] = "class ModelNew:\n    def forward(self,a,b,c,d):\n        return a*b+c-d"
    rows[1].response = credit.render_sections(sections)
    annotate(rows)
    assert credit.credited_returns(args(), rows, [1, 1, 1]) == [1, 1, 1]


@pytest.mark.parametrize(
    "change",
    [
        "all_wrong",
        "early_correct",
        "truncated_source",
        "removed_source",
        "zero_mask",
        "truncated_anchor",
        "decoy_anchor",
        "decoy_source",
        "padding_source",
        "padding_anchor",
    ],
)
def test_ineligible_turns_get_no_bonus(change):
    rows = trajectory()
    if change == "all_wrong":
        rows[1].metadata["env_result"]["env_state"]["correctness"] = False
    if change == "early_correct":
        rows[0].metadata["env_result"]["env_state"]["correctness"] = True
    if change == "truncated_source":
        rows[0].status = Sample.Status.TRUNCATED
    if change == "removed_source":
        rows[0].remove_sample = True
    if change == "zero_mask":
        rows[0].loss_mask = [0, 0, 0]
    if change == "truncated_anchor":
        rows[1].status = Sample.Status.TRUNCATED
    if change == "decoy_anchor":
        rows[1].metadata["env_result"]["env_state"]["decoy_kernel"] = True
    if change == "decoy_source":
        rows[0].metadata["env_result"]["env_state"]["decoy_kernel"] = True
    if change == "padding_source":
        rows[0].metadata["is_pad_turn"] = True
    if change == "padding_anchor":
        rows[1].metadata["is_pad_turn"] = True
    annotate(rows)
    assert credit.credited_returns(args(), rows, [1, 1, 1]) == [1, 1, 1]


def test_comment_only_changes_do_not_receive_bonus():
    rows = trajectory()
    rows[1].response = rows[0].response.replace("int kernel", "/* note */ int kernel")
    annotate(rows)
    assert credit.credited_returns(args(), rows, [1, 1, 1]) == [1, 1, 1]


def test_python_indentation_floor_division_and_comments():
    a = "def f(x):\n    if x:\n        x //= 2\n    return x\n"
    b = "def f(x):\n    if x:\n        x //= 2\n        return x\n"
    assert credit.code_tokens(a, "MODEL_NEW") != credit.code_tokens(b, "MODEL_NEW")
    assert "//=" in credit.code_tokens(a, "MODEL_NEW")
    assert "//" in credit.code_tokens("x = a // b # comment", "MODEL_NEW")
    assert credit.code_tokens(a, "MODEL_NEW") == credit.code_tokens(a.replace("    ", "  "), "MODEL_NEW")


def test_cpp_strings_operators_and_comments():
    a = credit.code_tokens('x >> 2; // comment\nconst char* s="http://a";', "CUDA_KERNELS")
    assert ">>" in a and '"http://a"' in a and "comment" not in a
    assert a != credit.code_tokens('x > > 2; const char* s="http://a";', "CUDA_KERNELS")


@pytest.mark.parametrize("mode", ["baseline", "diff", "shuffled"])
@pytest.mark.parametrize("anchor", [1, 2])
def test_training_reuses_original_results_without_environment_requests(monkeypatch, mode, anchor):
    from examples.kernel_agent import generate_with_cuda_agent as gen

    rows = trajectory(anchor=anchor)
    before = copy.deepcopy(rows)

    async def generated(a, s, p):
        return rows

    async def forbidden(*a, **kw):
        pytest.fail("credit annotation must not submit or cancel environment tasks")

    monkeypatch.setattr(gen, "_guarded_generate", generated)
    monkeypatch.setattr(gen, "cuda_kernel_env", forbidden)
    monkeypatch.setattr(gen, "cancel_kernel_eval", forbidden)
    config = args(correctness_diff_mode=mode)
    out = asyncio.run(gen.generate(config, rows[0], {}, evaluation=False))
    assert out is rows
    for sample, original in zip(rows, before, strict=True):
        assert sample.reward == original.reward
        assert sample.loss_mask == original.loss_mask
        assert {k: v for k, v in sample.metadata.items() if k != "correctness_diff"} == original.metadata
    targets = credit.credited_returns(config, rows, [1, 1, 1])
    expected = [1, 1, 1] if mode == "baseline" else ([1.25, 1, 1] if anchor == 1 else [1.125, 1.125, 1])
    assert targets == expected


def test_offline_probe_records_do_not_gate_training_credit():
    rows = trajectory()
    rows[1].metadata["correctness_diff"] = {
        "schema": "correctness-diff-credit/v2",
        "anchor_probe": {"verdict": "unknown", "exception_type": "TimeoutError"},
        "anchor_verified": False,
    }
    annotate(rows)
    assert "anchor_probe" not in rows[1].metadata["correctness_diff"]
    assert credit.credited_returns(args(), rows, [1, 1, 1]) == [1.25, 1, 1]


def test_shuffled_vectors_preserve_per_turn_and_trajectory_budgets():
    rows = []
    for gid in range(16):
        ss = trajectory(gid, anchor=2)
        if gid == 1:
            ss[1].response = far_response()
        elif gid == 2:
            ss[0].response = far_response()
        elif gid >= 3:
            ss[0].response = far_response()
            ss[1].response = far_response()
        annotate(ss)
        rows.extend(ss)
    base = [1.0] * len(rows)
    diff = credit.credited_returns(args(), rows, base)
    control = credit.credited_returns(args(correctness_diff_mode="shuffled"), rows, base)
    assert diff != control
    assert sum(diff) == sum(control) and sum(v > 1 for v in diff) == sum(v > 1 for v in control) == 4
    for t in range(3):
        assert sum(diff[t::3]) == sum(control[t::3])
    assert max(sum(control[i : i + 3]) - 3 for i in range(0, len(rows), 3)) <= 0.25
    assert control == credit.credited_returns(args(correctness_diff_mode="shuffled"), list(reversed(rows)), base)[::-1]
    assert credit.credited_returns(args(correctness_diff_mode="baseline"), rows, base) == base
    assert credit.credited_returns(args(correctness_diff_scale=0), rows, base) == base


def test_loo_and_zero_scale_identity():
    x, y = trajectory(0), trajectory(1)
    y[0].response = far_response()
    annotate(x)
    annotate(y)
    rows = x + y
    targets, adv = reward_post_process_by_group(args(), rows)
    assert targets == [1.25, 1, 1, 1, 1, 1] and adv == pytest.approx([0.25, 0, 0, -0.25, 0, 0])
    assert reward_post_process_by_group(args(correctness_diff_scale=0), rows) == reward_post_process_by_group(
        args(correctness_diff_mode="off"), rows
    )


@pytest.mark.parametrize("mode", ["baseline", "diff", "shuffled"])
@pytest.mark.parametrize("removed", [False, True])
def test_unresolved_client_timeout_aborts_all_experiment_arms_before_loo(mode, removed):
    rows = trajectory()
    annotate(rows)
    rows[2].metadata["env_result"]["env_state"] = {
        "status": "timeout",
        "error_message": "Task timeout after 2400s (client-side)",
    }
    rows[2].remove_sample = removed
    if removed:
        rows[2].loss_mask = [0, 0, 0]
    before = copy.deepcopy(rows)
    with pytest.raises(RuntimeError, match="Unresolved KernelGym.*group_id=0 turn=2"):
        reward_post_process_by_group(args(correctness_diff_mode=mode), rows)
    assert [s.metadata for s in rows] == [s.metadata for s in before]
    assert [s.reward for s in rows] == [s.reward for s in before]
    assert [s.loss_mask for s in rows] == [s.loss_mask for s in before]


def test_client_timeout_abort_is_not_disabled_by_zero_scale_and_off_is_unchanged():
    rows = trajectory()
    annotate(rows)
    rows[2].metadata["env_result"]["env_state"] = {
        "status": "timeout",
        "error_message": "Task timeout after 2400s (client-side)",
    }
    with pytest.raises(RuntimeError, match="Unresolved KernelGym"):
        credit.credited_returns(args(correctness_diff_scale=0), rows, [1, 1, 1])
    original = [1, 1, 1]
    assert credit.credited_returns(args(correctness_diff_mode="off"), rows, original) is original


@pytest.mark.parametrize("auxiliary", [False, True])
def test_candidate_or_sanitizer_timeout_keeps_existing_reward_policy(auxiliary):
    rows = trajectory()
    annotate(rows)
    state = rows[2].metadata["env_result"]["env_state"]
    if auxiliary:
        state["runtime_sanitizer"] = {"status": "error", "error": "memcheck timed out after 60s"}
    else:
        state.update(status="timeout", error_message="Task processing failed: kernel timeout after 300s")
    assert credit.credited_returns(args(), rows, [1, 1, 0]) == [1.25, 1, 0]


def test_changed_source_task_missing_evidence_and_duplicate_rejected():
    rows = trajectory()
    with pytest.raises(ValueError, match="missing"):
        credit.credited_returns(args(), rows, [1, 1, 1])
    annotate(rows)
    rows[0].response = far_response()
    with pytest.raises(ValueError, match="binding"):
        credit.credited_returns(args(), rows, [1, 1, 1])
    rows = trajectory()
    annotate(rows)
    rows[0].label = {"ground_truth": "changed"}
    with pytest.raises(ValueError, match="binding"):
        credit.credited_returns(args(), rows, [1, 1, 1])
    rows = trajectory()
    annotate(rows)
    with pytest.raises(ValueError, match="duplicate"):
        credit.credited_returns(args(), rows + [rows[0]], [1] * 4)


@pytest.mark.parametrize(
    "kw",
    [
        {"component_reward": True},
        {"correctness_diff_scale": float("nan")},
        {"correctness_diff_scale": -0.1},
        {"max_turns": 2},
        {"correctness_diff_max_edit_ratio": 0},
        {"correctness_diff_max_edit_ratio": float("nan")},
    ],
)
def test_invalid_protocol(kw):
    with pytest.raises(ValueError):
        credit.validate_args(args(**kw))


def test_evaluation_bypasses_credit(monkeypatch):
    from examples.kernel_agent import generate_with_cuda_agent as gen

    rows = trajectory()

    async def guarded(a, s, p):
        return rows

    monkeypatch.setattr(gen, "_guarded_generate", guarded)
    out = asyncio.run(gen.generate(args(), rows[0], {}, evaluation=True))
    assert out is rows and all("correctness_diff" not in s.metadata for s in rows)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "failed"},
        {"status": "timeout"},
        {"status": "pending"},
        {"compiled": False},
        {"compiled": None},
        {"correctness": False},
        {"correctness": None},
        {"decoy_kernel": None},
        {"decoy_kernel": True},
    ],
)
def test_incomplete_original_success_state_is_not_an_anchor(change):
    state = {"status": "completed", "compiled": True, "correctness": True, "decoy_kernel": False, **change}
    rows = trajectory()
    rows[1].metadata["env_result"]["env_state"] = state
    assert not credit.correct(rows[1])
    annotate(rows)
    assert credit.credited_returns(args(), rows, [1, 1, 1]) == [1, 1, 1]


def test_missing_original_result_does_not_trigger_fallback_or_credit():
    rows = trajectory()
    rows[1].metadata.pop("env_result")
    annotate(rows)
    assert credit.credited_returns(args(), rows, [1, 1, 1]) == [1, 1, 1]


@pytest.mark.parametrize("mode", ["baseline", "diff", "shuffled"])
@pytest.mark.parametrize("selected", [False, True])
def test_changed_original_anchor_verdict_is_rejected_before_credit(mode, selected):
    rows = trajectory()
    if not selected:
        rows[0].response = far_response()
    annotate(rows)
    rows[1].metadata["env_result"]["env_state"]["correctness"] = False
    with pytest.raises(ValueError, match="invalid correct anchor"):
        credit.credited_returns(args(correctness_diff_mode=mode), rows, [1, 1, 1])


def test_original_anchor_task_binding_is_checked():
    rows = trajectory()
    annotate(rows)
    rows[1].metadata["task_id"] = "different-evaluation"
    with pytest.raises(ValueError, match="evaluation binding mismatch"):
        credit.credited_returns(args(), rows, [1, 1, 1])


def test_old_recheck_schema_is_rejected_until_reannotated():
    rows = trajectory()
    annotate(rows)
    rows[0].metadata["correctness_diff"]["schema"] = "correctness-diff-credit/v2"
    with pytest.raises(ValueError, match="missing correctness-diff evidence"):
        credit.credited_returns(args(), rows, [1, 1, 1])


def test_metrics_do_not_report_obsolete_recheck_failures():
    rows = trajectory()
    annotate(rows)
    credit.credited_returns(args(), rows, [1, 1, 1])
    metrics = credit.batch_metrics(rows)
    assert metrics["correctness_diff/covered_trajectories"] == 1
    assert metrics["correctness_diff/bonus_sum"] == 0.25
    assert "correctness_diff/anchor_recheck_failed_trajectories" not in metrics


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
