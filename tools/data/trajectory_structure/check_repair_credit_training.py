"""CPU workflow checks for whole-turn interventions and shadow TRLOO targets."""

import copy
import unittest
from types import SimpleNamespace

from examples.kernel_agent.correctness_diff_reward import exact_transplant, whole_turn_bundle
from examples.kernel_agent.kernel_reward import reward_post_process_by_group

from .check_structure import response
from .repair_credit_replay import planned_precheck, sha, task_evaluation
from .repair_credit_training import independent_loo, shadow_rows, training_phase

NUM_GPUS = 0


def compute(rows, bonuses=None):
    bonuses = bonuses or {}
    samples = [
        SimpleNamespace(
            group_index=r["group_index"],
            remove_sample=r["remove_sample"],
            metadata={
                "turn_idx": r["turn_idx"],
                "multi_turn_reward": r["baseline_return"] + bonuses.get(r["row_id"], 0),
            },
        )
        for r in rows
    ]
    args = SimpleNamespace(
        advantage_estimator="trloo",
        use_multi_turn=True,
        component_reward=False,
        grpo_std_normalization=False,
        use_conditional_truncation_mask=False,
    )
    return reward_post_process_by_group(args, samples)


def row(name, value, turn=1, removed=False, eligible=False, group=1):
    return {
        "row_id": name,
        "baseline_return": value,
        "group_index": group,
        "turn_idx": turn,
        "remove_sample": removed,
        "active_tokens": 0 if removed else 100,
        "bonus_eligible": eligible,
    }


class WholeTurnChecks(unittest.TestCase):
    def test_recorded_training_precision_cannot_silently_use_global_fp32(self):
        plan = {"evaluation": {"precision": "fp32"}}
        case = {"evaluation_overrides": {"precision": "fp16", "entry_point": "Model"}}
        self.assertEqual(task_evaluation(plan, {}, case)["precision"], "fp16")
        with self.assertRaises(ValueError):
            task_evaluation(plan, {"evaluation_overrides": {"precision": "fp32"}}, case)

    def test_task_precision_override_keeps_other_protocol_fields(self):
        base = {"evaluation": {"precision": "fp32", "entry_point": "Model", "num_correct_trials": 5}}
        result = task_evaluation(base, {"evaluation_overrides": {"precision": "fp16"}})
        self.assertEqual(result["precision"], "fp16")
        self.assertEqual(result["num_correct_trials"], 5)
        self.assertEqual(base["evaluation"]["precision"], "fp32")
        with self.assertRaises(ValueError):
            task_evaluation(base, {"evaluation_overrides": {"num_correct_trials": 1}})

    def test_shape_assert_mismatch_is_runtime_not_output_mismatch(self):
        value = {"correctness": False, "environment_error_message": "Kernel execution failed: flattened dim mismatch"}
        self.assertEqual(training_phase(value), "runtime_error")

    def test_precheck_is_bound_to_exact_candidate(self):
        plan = {
            "client_precheck_results": {
                "case": {"candidate_sha256": sha("code"), "passed": False, "state": {"correctness": False}}
            }
        }
        self.assertFalse(planned_precheck(plan, "case", "code")["passed"])
        with self.assertRaises(ValueError):
            planned_precheck(plan, "case", "different")

    def test_precheck_without_feedback_is_rejected(self):
        plan = {"client_precheck_results": {"case": {"candidate_sha256": sha("code"), "passed": False}}}
        with self.assertRaises(ValueError):
            planned_precheck(plan, "case", "code")

    def test_complete_patch_reconstructs_actual_after(self):
        base = "header\nleft\nold\nright\nfooter\n"
        after = base.replace("old", "new")
        result, edits, reason = exact_transplant(base, after, base)
        self.assertIsNone(reason)
        self.assertEqual(result, after)
        self.assertEqual(len(edits), 1)

    def test_disjoint_changes_compose(self):
        original = "A0\nx\ny\nz\na\nb\nc\nd\nB0\ne\nf\n"
        middle = original.replace("A0", "A1")
        final = middle.replace("B0", "B1")
        result, _, reason = exact_transplant(middle, final, original)
        self.assertIsNone(reason)
        self.assertEqual(result, original.replace("B0", "B1"))

    def test_conflicting_changes_abstain(self):
        result, edits, reason = exact_transplant("int a=2;\n", "int a=3;\n", "int a=1;\n")
        self.assertIsNone(result)
        self.assertFalse(edits)
        self.assertTrue(reason)

    def test_duplicate_target_abstains(self):
        result, _, reason = exact_transplant("a\nb\nc\n", "a\nx\nc\n", "a\nb\nc\na\nb\nc\n")
        self.assertIsNone(result)
        self.assertTrue(reason)

    def test_deletion_is_a_change(self):
        before = "start\nold\nend\n"
        after = "start\nend\n"
        self.assertEqual(exact_transplant(before, after, before)[0], after)

    def test_one_failed_block_does_not_produce_partial_variant(self):
        before = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\n"
        after = before.replace("b\n", "B\n").replace("j\n", "J\n")
        result, edits, _ = exact_transplant(before, after, before.replace("k\n", "other\n"))
        self.assertIsNone(result)
        self.assertFalse(edits)

    def test_missing_section_is_unknown(self):
        program = response("void f(){}")
        case = whole_turn_bundle([program.split("### MODEL_NEW")[0], program, program])
        self.assertEqual(case["reason"], "incomplete_selected_sections")

    def test_unchanged_source_outcome_flip_is_not_repair(self):
        program = response("void f(){}")
        case = whole_turn_bundle([program, program, program])
        self.assertEqual(case["reason"], "middle_has_no_source_change")


class ShadowChecks(unittest.TestCase):
    def test_zero_coefficient_is_identity(self):
        rows = [row("a", 1, eligible=True), row("b", 2)]
        result = shadow_rows(rows, compute, {"a"}, 0)
        self.assertTrue(all(r["effective_delta"] == 0 for r in result))

    def test_same_turn_peer_decreases_but_terminal_turn_unchanged(self):
        rows = [
            row("repair", 1, eligible=True),
            row("already_correct", 2),
            row("final", 1, turn=2),
            row("final_peer", 2, turn=2),
        ]
        result = {r["row_id"]: r for r in shadow_rows(rows, compute, {"repair"}, 0.25)}
        self.assertAlmostEqual(result["repair"]["effective_delta"], 0.25)
        self.assertAlmostEqual(result["already_correct"]["effective_delta"], -0.25)
        self.assertEqual(result["final"]["effective_delta"], 0)
        self.assertEqual(result["final_peer"]["effective_delta"], 0)
        self.assertEqual(result["already_correct"]["shadow_return"], 2)

    def test_removed_turn_stays_removed_and_outside_centering(self):
        rows = [row("a", 1, eligible=True), row("b", 2), row("removed", 100, removed=True)]
        result = shadow_rows(rows, compute, {"a"}, 0.25)
        self.assertAlmostEqual(result[0]["shadow_advantage"], -0.75)
        self.assertEqual(result[2]["effective_delta"], 0)

    def test_singleton_group_cannot_gain_advantage(self):
        result = shadow_rows([row("a", 1, eligible=True)], compute, {"a"}, 0.25)
        self.assertEqual(result[0]["shadow_advantage"], 0)

    def test_other_question_is_unchanged(self):
        rows = [row("a", 1, eligible=True), row("b", 2), row("c", 1, group=2), row("d", 2, group=2)]
        result = shadow_rows(rows, compute, {"a"}, 0.25)
        self.assertEqual(result[2]["effective_delta"], 0)
        self.assertEqual(result[3]["effective_delta"], 0)

    def test_ineligible_reward_and_unknown_identity_rejected(self):
        for ids in [{"a"}, {"missing"}]:
            with self.assertRaises(ValueError):
                shadow_rows([row("a", 1)], compute, ids, 0.25)

    def test_original_inputs_unchanged(self):
        rows = [row("a", 1, eligible=True), row("b", 2)]
        original = copy.deepcopy(rows)
        shadow_rows(rows, compute, {"a"}, 0.25)
        self.assertEqual(rows, original)

    def test_independent_reference_excludes_self(self):
        result = independent_loo([row("a", 1), row("b", 2), row("c", 3)])
        self.assertEqual(result, {"a": -1.5, "b": 0.0, "c": 1.5})


if __name__ == "__main__":
    unittest.main()
