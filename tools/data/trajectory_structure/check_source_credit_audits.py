"""CPU contracts for reconstructed source audits and serialized rollout checks."""

import copy
import unittest

import numpy as np

from examples.kernel_agent.component_reward import ADDITIVE_METHOD, ADDITIVE_MODE, METHOD
from examples.kernel_agent.source_component_reward import source_request_identity
from examples.kernel_agent.utils import extract_cuda_agent_kernel_code
from slime.utils.types import Sample

from .check_structure import response
from .source_credit_pilot import audit_rows, finite_reward, reconstruct_sample, sample_status
from .source_credit_rollout_audit import audit_predictive, audit_sample, group_checks

NUM_GPUS = 0


def saved_sample(turn=0, additive=False):
    code = response("__global__ void kernel(float*x){x[threadIdx.x]*=2;}")
    record = {
        "method": METHOD,
        "credits": [0.5, 0.5],
        "targets": [0.5, 0.5],
        "quality_budget": 1.0,
        "best_turn": 1,
        "status": "attributed",
        "base_scores": [0.0, 1.0],
        "turn_credit": 0.5,
        "turn_target": 0.5,
        "turn_idx": turn,
    }
    if additive:
        record.update(
            method=ADDITIVE_METHOD,
            mode=ADDITIVE_MODE,
            allocation_method=METHOD,
            backend="source-strategies/v1",
            scale=0.25,
            anchor_min_speedup=1.0,
            baseline_rewards=[0.0, 1.0],
            baseline_returns=[1.0, 1.0],
            active_turns=[True, True],
            anchor_correct=[False, True],
            anchor_speedups=[None, 2.0],
            anchor_eligible_turns=[1],
            source_turns=[0],
            scaled_credits=[0.125, 0.0],
            target_deltas=[0.125, 0.0],
            targets=[1.125, 1.0],
            turn_target=[1.125, 1.0][turn],
        )
    return {
        "group_id": 7,
        "index": 7,
        "response": code,
        "response_length": 2,
        "tokens": [9, 10, 11],
        "loss_mask": [1, 1],
        "rollout_log_probs": [-0.1, -0.2],
        "rollout_topk_token_ids": np.array([[10, 12], [11, 12]], dtype=np.int32),
        "rollout_topk_log_probs": np.array([[-0.1, -0.3], [-0.2, -0.4]], dtype=np.float32),
        "rollout_topk_valid_mask": np.ones((2, 2), dtype=np.bool_),
        "status": "completed",
        "remove_sample": False,
        "label": {"ground_truth": "reference"},
        "reward": [0.0, 1.0][turn] if additive else 0.5,
        "metadata": {
            "turn_idx": turn,
            "task_reward": [0.0, 1.0][turn],
            "multi_turn_reward": 1.0 if additive else 0.5,
            "component_reward": record,
            "source_component_identity": source_request_identity(
                "reference", extract_cuda_agent_kernel_code(code), entry_point="Model", precision="fp32"
            ),
        },
    }


class RolloutAuditChecks(unittest.TestCase):
    def test_replace_and_additive_preserve_their_distinct_rewards(self):
        for additive in [False, True]:
            with self.subTest(additive=additive):
                samples = [saved_sample(t, additive) for t in range(2)]
                audits = [audit_sample(s, i) for i, s in enumerate(samples)]
                self.assertEqual([a["errors"] for a in audits], [[], []])
                self.assertEqual(group_checks(samples, audits), [])
                self.assertEqual(samples[0]["reward"], 0.0 if additive else 0.5)

    def test_additive_rejects_replaced_baseline(self):
        sample = saved_sample(additive=True)
        sample["reward"] = 1.125
        sample["metadata"]["multi_turn_reward"] = 1.125
        errors = audit_sample(sample, 0)["errors"]
        self.assertIn("sample_reward_not_saved_baseline_reward", errors)
        self.assertIn("multi_turn_reward_not_saved_baseline_return", errors)

    def test_additive_rejects_corrupted_target_formula(self):
        sample = saved_sample(additive=True)
        sample["metadata"]["component_reward"]["targets"][0] = 2.0
        self.assertTrue(
            any(e.startswith("invalid_additive_component_record:") for e in audit_sample(sample, 0)["errors"])
        )

    def test_response_identity_mismatch_is_reported(self):
        sample = saved_sample()
        sample["response"] = response("void changed(){}")
        self.assertIn("candidate_source_sha256_mismatch", audit_sample(sample, 0)["errors"])

    def test_excluded_missing_response_does_not_crash(self):
        sample = {"status": Sample.Status.ABORTED, "response": None, "loss_mask": [], "metadata": {"turn_idx": 0}}
        audit = audit_sample(sample, 0)
        self.assertTrue(audit["hard_excluded"])
        self.assertEqual(audit["errors"], [])

    def test_excluded_trainable_mask_is_reported(self):
        errors = []
        audit_predictive({"loss_mask": [1]}, errors, strict=False)
        self.assertIn("excluded_turn_has_trainable_loss_mask", errors)

    def test_missing_predictive_support_is_not_reconstructed(self):
        sample = saved_sample()
        sample["rollout_topk_token_ids"] = None
        self.assertIn("missing_or_partial_predictive_topk_support", audit_sample(sample, 0)["errors"])
        self.assertIsNone(sample["rollout_topk_token_ids"])

    def test_predictive_dtype_and_empty_width_fail_without_crashing(self):
        sample = saved_sample()
        sample["rollout_topk_valid_mask"] = np.ones((2, 2), dtype=np.float32)
        self.assertIn("predictive_topk_valid_mask_not_bool", audit_sample(sample, 0)["errors"])
        for name in ["rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_topk_valid_mask"]:
            sample[name] = sample[name][:, :0]
        self.assertIn("predictive_topk_empty_support_width", audit_sample(sample, 0)["errors"])

    def test_duplicate_and_missing_sampled_token_are_reported(self):
        sample = saved_sample()
        sample["rollout_topk_token_ids"][0] = [12, 12]
        errors = audit_sample(sample, 0)["errors"]
        self.assertIn("predictive_topk_duplicate_valid_token_id", errors)
        self.assertIn("sampled_trainable_response_token_not_exactly_once_in_predictive_support", errors)

    def test_sampled_log_probability_must_match(self):
        sample = saved_sample()
        sample["rollout_log_probs"][0] = -0.9
        self.assertIn("sampled_trainable_logprob_mismatch_predictive_support", audit_sample(sample, 0)["errors"])

    def test_nonfinite_group_credit_is_reported(self):
        sample = saved_sample()
        sample["metadata"]["component_reward"]["credits"] = [None, 1.0]
        self.assertIn("component_budget_or_credits_missing", group_checks([sample], [audit_sample(sample, 0)]))

    def test_missing_turn_index_is_reported(self):
        samples = [saved_sample(t) for t in range(2)]
        samples[0]["metadata"].pop("turn_idx")
        self.assertEqual(
            group_checks(samples, [audit_sample(s, i) for i, s in enumerate(samples)]),
            ["turn_indices_not_ordered_contiguous"],
        )

    def test_inconsistent_group_allocation_is_reported(self):
        samples = [saved_sample(t) for t in range(2)]
        samples[1]["metadata"]["component_reward"]["targets"][0] = 2.0
        self.assertIn(
            "component_allocation_not_identical_within_group",
            group_checks(samples, [audit_sample(s, i) for i, s in enumerate(samples)]),
        )


class HistoricalPilotChecks(unittest.TestCase):
    def record(self):
        return {
            "turn_idx": 0,
            "status": "completed",
            "response": response("void f(){}"),
            "task_reward": 0.0,
            "label": {"ground_truth": "reference"},
        }

    def test_reconstruction_is_marked_and_input_unchanged(self):
        record = self.record()
        record.update(remove_sample=True, remove_reason="finalize_negative")
        before = copy.deepcopy(record)
        sample, audit = reconstruct_sample(record)
        self.assertEqual(record, before)
        self.assertFalse(sample.remove_sample)
        self.assertTrue(audit["soft_finalize_removal_restored"])
        self.assertTrue(sample.metadata["offline_source_credit_reconstruction"]["synthetic_nonempty_loss_mask"])

    def test_hard_removal_is_preserved(self):
        record = self.record()
        record.update(remove_sample=True, remove_reason="generation_failed", task_reward=1.0)
        result = audit_rows("case", [record], "fixture")
        self.assertEqual(result["allocation"]["credits"], [0.0])
        self.assertEqual(result["checks"]["hard_removed_credit"], 0.0)

    def test_status_enum_and_string_agree(self):
        self.assertEqual(sample_status(Sample.Status.COMPLETED), sample_status("completed"))

    def test_task_reward_precedes_cumulative_reward(self):
        self.assertEqual(finite_reward({"task_reward": 0.25, "reward": 1.0}), 0.25)
        self.assertEqual(finite_reward({"task_reward": None, "reward": 1.0}), 1.0)
        for invalid in [True, float("nan"), float("inf")]:
            self.assertEqual(finite_reward({"task_reward": invalid}), 0.0)

    def test_duplicate_or_missing_turns_rejected(self):
        for rows in [[self.record(), self.record()], [{**self.record(), "turn_idx": 1}]]:
            with self.assertRaisesRegex(ValueError, "contiguous, unique"):
                audit_rows("case", rows, "fixture")


if __name__ == "__main__":
    unittest.main()
