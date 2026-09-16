"""CPU-only contracts for offline first-correct repair attribution."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from .check_structure import response
from .repair_credit_audit import error_tolerant_native_candidates, first_correct, inspect_case
from .repair_credit_replay import build_variant, run
from .structure import analyze_response

NUM_GPUS = 0


def row(turn, code, correct=False, **kw):
    return {
        "turn_idx": turn,
        "response": response(code),
        "correctness": correct,
        "compiled": correct,
        "decoy": False,
        "status": "completed",
        **kw,
    }


ORIGINAL = "void f(){begin(); int a=1; separator(); int b=2; end();}"
FIRST = ORIGINAL.replace("a=1", "a=3")
FINAL = FIRST.replace("b=2", "b=4")


class RepairAuditChecks(unittest.TestCase):
    def test_first_correct_is_not_fastest(self):
        self.assertEqual(first_correct([row(0, ORIGINAL, True, speedup=0.1), row(1, FINAL, True, speedup=2.0)]), 0)

    def test_failed_trajectory_has_no_anchor(self):
        self.assertIsNone(first_correct([row(0, ORIGINAL), row(1, FIRST)]))

    def test_decoy_is_not_anchor(self):
        self.assertIsNone(first_correct([row(0, ORIGINAL, True, decoy=True)]))

    def test_truncated_correct_record_is_not_anchor(self):
        self.assertIsNone(first_correct([row(0, ORIGINAL, True, status="truncated")]))

    def test_noncompiled_correct_record_is_not_anchor(self):
        self.assertIsNone(first_correct([row(0, ORIGINAL, True, compiled=False)]))

    def test_missing_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            first_correct([row(0, ORIGINAL), row(2, FINAL, True)])

    def test_duplicate_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            first_correct([row(0, ORIGINAL), row(0, FINAL, True)])

    def test_retained_intermediate_edit(self):
        r = inspect_case([row(0, ORIGINAL), row(1, FIRST), row(2, FINAL, True)])
        self.assertGreater(r["history_local_count"], 0)
        self.assertGreater(r["terminal_local_count"], 0)
        self.assertEqual(r["category"], "earlier_local_edit_retained")
        self.assertTrue(all(x["causal_contribution"] == "not_evaluated" for x in r["retained_history_candidates"]))

    def test_terminal_small_fix_not_removed(self):
        r = inspect_case([row(0, ORIGINAL), row(1, FIRST, True)])
        self.assertEqual(r["history_local_count"], 0)
        self.assertGreater(r["terminal_local_count"], 0)

    def test_reversion_is_not_retention(self):
        r = inspect_case([row(0, ORIGINAL), row(1, FIRST), row(2, ORIGINAL, True)])
        self.assertEqual(r["history_local_count"], 0)
        self.assertTrue(any(x["anchor_state"] == "reverted" for x in r["nonretained_or_unknown_history"]))

    def test_deletion_can_survive(self):
        before = "void f(){begin(); first(); obsolete(); last(); end();}"
        middle = before.replace(" obsolete();", "")
        last = middle + "\nvoid unrelated(){}"
        r = inspect_case([row(0, before), row(1, middle), row(2, last, True)])
        self.assertTrue(any(x.get("hunk", {}).get("operation") == "delete" for x in r["retained_history_candidates"]))

    def test_context_drift_is_unknown(self):
        r = inspect_case([row(0, ORIGINAL), row(1, FIRST), row(2, "void f(){other(); return;}", True)])
        self.assertEqual(r["history_local_count"], 0)
        self.assertTrue(r["nonretained_or_unknown_history"])

    def test_comments_are_not_edits(self):
        r = inspect_case([row(0, ORIGINAL), row(1, "// note\n" + ORIGINAL, True)])
        self.assertEqual(r["terminal_local_count"], 0)

    def test_no_reward_assigned_or_inputs_mutated(self):
        rows = [row(0, ORIGINAL, reward_observed=0), row(1, FIRST), row(2, FINAL, True, reward_observed=1)]
        saved = copy.deepcopy(rows)
        r = inspect_case(rows)
        self.assertEqual(rows, saved)
        self.assertIsNone(r["allocation"])

    def test_later_correct_performance_revision_is_not_used(self):
        r = inspect_case([row(0, ORIGINAL), row(1, FIRST, True), row(2, FINAL, True)])
        self.assertEqual(r["anchor_turn"], 1)
        self.assertEqual(len(r["structure"]["turns"]), 2)

    def test_partial_program_is_exposed(self):
        broken = row(0, ORIGINAL)
        broken["response"] = broken["response"].split("### MODEL_NEW")[0]
        r = inspect_case([broken, row(1, FIRST, True)])
        self.assertFalse(r["complete_program_prefix"])
        self.assertIn("incomplete_selected_program", r["gaps"])

    def test_replay_edit_requires_exact_unique_source(self):
        case = inspect_case([row(0, ORIGINAL), row(1, FIRST, True)])
        with self.assertRaises(ValueError):
            build_variant(case, {"base_turn": 0, "edits": [{"section": "CUDA_KERNELS", "old": "missing", "new": "x"}]})
        with self.assertRaises(ValueError):
            build_variant(case, {"base_turn": 0, "edits": [{"section": "CUDA_KERNELS", "old": ";", "new": "x"}]})

    def test_replay_edit_does_not_modify_saved_source(self):
        case = inspect_case([row(0, ORIGINAL), row(1, FIRST, True)])
        before = copy.deepcopy(case)
        code, sections = build_variant(
            case, {"base_turn": 0, "edits": [{"section": "CUDA_KERNELS", "old": "a=1", "new": "a=3"}]}
        )
        self.assertEqual(sections["CUDA_KERNELS"], FIRST)
        self.assertEqual(case, before)
        self.assertIn("### MODEL_NEW", code)

    def test_native_syntax_repair_is_not_lost(self):
        a = "void f(float* x){start(); float v=x[0); finish(v);}"
        b = a.replace("x[0)", "x[0]")
        case = inspect_case([row(0, a), row(1, b), row(2, b + "\nvoid other(){}", True)])
        candidates = error_tolerant_native_candidates(case["structure"])
        self.assertTrue(candidates)
        self.assertTrue(any(e["anchor_state"] == "present" and e["introduced_turn"] == 1 for e in candidates))
        self.assertTrue(all(e["causal_contribution"] == "not_evaluated" for e in candidates))

    def test_native_syntax_repair_cannot_fill_missing_program(self):
        a = row(0, "void f(float*x){float v=x[0);}")
        a["response"] = a["response"].split("### MODEL_NEW")[0]
        r = inspect_case([a, row(1, "void f(float*x){float v=x[0];}", True)])
        self.assertEqual(error_tolerant_native_candidates(r["structure"]), [])

    def test_error_tree_keeps_existing_identifiers_in_rewritten_expressions(self):
        before = """void f(float* input, float* weight) {
            float input_val = input[((n*C+ci)*D+d)*H+h)*W+w];
            float weight_val = weight[((ci*O+co)*KD+kd)*KH+kh)*KW+kw];
            acc += input_val * weight_val;
        }"""
        after = """void f(float* input, float* weight) {
            int input_idx = (((n*C+ci)*D+d)*H+h)*W+w;
            int weight_idx = (((ci*O+co)*KD+kd)*KH+kh)*KW+kw;
            float input_val = input[input_idx];
            float weight_val = weight[weight_idx];
            acc += input_val * weight_val;
        }"""
        for source, valid in [(before, False), (after, True)]:
            parsed = analyze_response(response(source))
            component = next(c for c in parsed["components"] if c["qualified_name"] == "f")
            self.assertEqual(component["valid_syntax"], valid)
            # A diff insertion is not proof that a variable/calculation is new.
            self.assertEqual(component["_tokens"].count("input_val"), 2)
            self.assertEqual(component["_tokens"].count("weight_val"), 2)
        case = inspect_case([row(0, before), row(1, after, True)])
        self.assertIsNone(case["allocation"])
        self.assertTrue(case["syntax_fallback_candidates"])
        self.assertTrue(all(e["causal_contribution"] == "not_evaluated" for e in case["syntax_fallback_candidates"]))


class ReplayChecks(unittest.TestCase):
    def run_plan(self, root, submit, opener):
        case = inspect_case([row(0, ORIGINAL), row(1, FIRST, True)])
        case["reference_code"] = "reference fixture"
        # Replay only needs saved sections, not the parser's internal syntax trees.
        case["structure"] = {
            "turns": [{"turn_idx": t["turn_idx"], "sections": t["sections"]} for t in case["structure"]["turns"]]
        }
        (root / "case.json").write_text(json.dumps(case, default=str))
        task = {"name": "control", "case_path": "case.json", "base_turn": 1, "require_correct": True}
        plan = {"evaluation": {"precision": "fp32"}, "tasks": [task, {**task, "name": "later"}]}
        (root / "plan.json").write_text(json.dumps(plan))
        with patch("urllib.request.build_opener", return_value=opener):
            run(
                SimpleNamespace(
                    plan=root / "plan.json",
                    output=root / "out",
                    url="http://fixture.invalid",
                    submit=submit,
                    wait_timeout=1,
                )
            )

    def test_prepare_does_not_contact_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opener = Mock()
            self.run_plan(root, False, opener)
            opener.open.assert_not_called()
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertTrue(manifest["completed"])
            self.assertFalse(manifest["submitted"])
            self.assertEqual(len(manifest["requests"]), 2)

    def test_ambiguous_submission_is_persisted_and_not_retried(self):
        opener = Mock()
        opener.open.side_effect = [io.BytesIO(b'{"status":"healthy"}'), io.BytesIO(b"{}"), TimeoutError()]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(TimeoutError):
                self.run_plan(root, True, opener)
            self.assertEqual(opener.open.call_count, 3)
            manifest = json.loads((root / "out/manifest.json").read_text())
            request = json.loads((root / "out/request_control.json").read_text())
            self.assertEqual(manifest["requests"][0]["task_id"], request["task_id"])
            self.assertFalse((root / "out/request_later.json").exists())

    def test_failed_control_stops_later_requests(self):
        opener = Mock()
        opener.open.side_effect = [
            io.BytesIO(json.dumps(value).encode())
            for value in [
                {"status": "healthy"},
                {},
                {},
                {"status": "completed"},
                {"env_state": {"correctness": False, "decoy_kernel": False}},
            ]
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "correct control did not reproduce"):
                self.run_plan(root, True, opener)
            self.assertEqual(opener.open.call_count, 5)
            self.assertFalse((root / "out/request_later.json").exists())


if __name__ == "__main__":
    unittest.main()
