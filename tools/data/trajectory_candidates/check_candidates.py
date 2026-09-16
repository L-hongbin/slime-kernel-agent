"""CPU-only candidate-screening behavioral checks."""

import copy
import unittest

from tools.data.trajectory_structure.check_structure import response
from tools.data.trajectory_structure.structure import analyze_trajectory

from .screen import Protocol, cues, decision_facts, observation_state, review_selection, screen_trajectory


def fixture(speeds, codes=None):
    rows = []
    for i, speed in enumerate(speeds):
        code = codes[i] if codes else f"void f(){{int x={i};}}"
        raw = "incomplete" if speed == "gap" else response(code)
        rows.append(
            {
                "turn_idx": i,
                "response": raw,
                "correctness": True if isinstance(speed, (int, float)) else False,
                "compiled": isinstance(speed, (int, float)),
                "decoy": False,
                "speedup": speed if isinstance(speed, (int, float)) else 0,
                "error_code": "VALIDATION_ERROR" if speed == "gap" else "COMPILATION_ERROR" if speed is None else None,
                "status": "completed",
            }
        )
    result = analyze_trajectory(rows)
    result["identity"] = ["model", "development", "task", "1", "0"]
    return result


class CandidateChecks(unittest.TestCase):
    def test_decline_then_observed_new_best(self):
        r = screen_trajectory(fixture([1.0, None, 0.8, 1.2]), Protocol())
        self.assertTrue(r["flags"]["decline_then_breakthrough"])
        self.assertEqual(r["decline_episodes"][0]["baseline_turn"], 0)
        self.assertEqual(r["decline_episodes"][0]["outcome_turn"], 3)

    def test_recovery_is_not_breakthrough(self):
        r = screen_trajectory(fixture([1.0, None, 0.8]), Protocol())
        self.assertFalse(r["any_candidate"])
        self.assertEqual(r["recoveries"][0]["label"], "below_historical_best")

    def test_first_correct_is_not_performance_breakthrough(self):
        r = screen_trajectory(fixture([None, None, 2.0]), Protocol())
        self.assertFalse(r["any_candidate"])
        self.assertEqual(r["timeline"][-1]["label"], "first_correct_timed")

    def test_exact_threshold_not_above_margin(self):
        r = screen_trajectory(fixture([1.0, 1.05]), Protocol())
        self.assertFalse(r["timeline"][-1]["breakthrough_candidate"])

    def test_missing_middle_makes_episode_indeterminate(self):
        r = screen_trajectory(fixture([1.0, None, "gap", 1.2]), Protocol())
        self.assertFalse(r["flags"]["decline_then_breakthrough"])
        self.assertTrue(any(x["kind"] == "decline_then_breakthrough" for x in r["uncertain_windows"]))

    def test_same_code_gain_requires_remeasurement(self):
        r = screen_trajectory(fixture([1.0, None, 1.3], ["void f(){}", "void g(){}", "void f(){}"]), Protocol())
        self.assertEqual(r["timeline"][-1]["label"], "measured_new_best")
        self.assertFalse(r["timeline"][-1]["breakthrough_candidate"])
        self.assertIn("earlier_code_version_measured_gain", r["timeline"][-1]["warnings"])

    def test_nonadjacent_reuse_needs_later_gain(self):
        codes = [
            "void f(){int x=32;}",
            "void f(){int x=64;}",
            "void f(){int x=64;} void g(){int y=1;}",
            "void f(){int x=64;} void g(){int y=2;}",
        ]
        r = screen_trajectory(fixture([1.0, 0.8, 0.9, 1.2], codes), Protocol())
        self.assertTrue(r["flags"]["nonadjacent_reuse"])
        self.assertTrue(
            any(
                u["component"] == "f" and u["introduced_turn"] == 1 and u["outcome_turn"] == 3
                for u in r["reuse_units"]
            )
        )
        self.assertFalse(
            screen_trajectory(fixture([1.0, 0.8, 0.9, 0.95], codes), Protocol())["flags"]["nonadjacent_reuse"]
        )

    def test_rewrite_gain_must_be_strictly_later(self):
        codes = [
            "void f(){int x=1;}",
            "void g(){cublasSgemm(a,b,c,d,e,f,g,h,i,j,k,l,m);}",
            "void g(){cublasSgemm(a,b,c,d,e,f,g,h,i,j,k,l,n);}",
        ]
        r = screen_trajectory(fixture([1.0, 1.2, 1.1], codes), Protocol())
        self.assertFalse(r["flags"]["rewrite_then_breakthrough"])
        r = screen_trajectory(fixture([1.0, 0.8, 1.3], codes), Protocol())
        self.assertTrue(r["flags"]["rewrite_then_breakthrough"])

    def test_resource_error_not_model_regression(self):
        r = fixture([1.0, None, 1.2])
        r["turns"][1]["observation"]["error_code"] = "RESOURCE_ERROR"
        out = screen_trajectory(r, Protocol())
        self.assertFalse(out["flags"]["decline_then_breakthrough"])
        self.assertEqual(out["timeline"][1]["state"], "environment_uncertain")

    def test_invalid_measurements_do_not_enter_best(self):
        for value in [0.0, float("nan"), float("inf")]:
            r = fixture([1.0])
            r["turns"][0]["observation"]["speedup"] = value
            self.assertEqual(observation_state(r["turns"][0])[0], "correct_untimed")
        r = fixture([1.0])
        r["turns"][0]["observation"]["decoy"] = True
        self.assertEqual(observation_state(r["turns"][0])[0], "evaluation_rejected")

    def test_comment_nodes_are_not_library_arguments(self):
        r = fixture([1.0], ["void f(){cublasSgemm(h,CUBLAS_OP_T, // opA comment\n CUBLAS_OP_N,x);}"])
        self.assertEqual(decision_facts(r["turns"][0])[0]["values"], ["CUBLAS_OP_T", "CUBLAS_OP_N"])

    def test_gemm_ex_can_have_both_math_and_layout_facts(self):
        r = fixture([1.0], ["void f(){cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,CUBLAS_COMPUTE_32F_FAST_TF32);}"])
        self.assertEqual({f["kind"] for f in decision_facts(r["turns"][0])}, {"math_mode", "transpose_layout"})

    def test_binding_stride_check_is_not_layout_change(self):
        self.assertNotIn(
            "transpose_layout",
            cues(
                {
                    "component": "wrapper",
                    "hunk": {
                        "before": ["x", ".", "strides", "(", ")"],
                        "after": [],
                        "left_context": ["TVM_FFI_ICHECK"],
                        "right_context": [],
                    },
                }
            ),
        )

    def test_gap_before_gain_never_silently_becomes_negative(self):
        r = screen_trajectory(fixture([1.0, "gap", 1.3]), Protocol())
        self.assertFalse(r["any_candidate"])
        self.assertTrue(any(w["kind"] == "unobserved_transition_before_gain" for w in r["uncertain_windows"]))

    def test_worker_crash_and_timeout_have_unattributed_states(self):
        r = fixture([None])
        t = r["turns"][0]
        t["observation"]["error_code"] = "KERNEL_EVAL_FAILED"
        t["observation"]["environment_error_message"] = "WorkerProcessCrashed: channel EOF"
        self.assertEqual(observation_state(t)[0], "worker_crash_unattributed")
        t["observation"]["error_code"] = "KERNEL_EVAL_TIMEOUT"
        self.assertEqual(observation_state(t)[0], "execution_timeout_unattributed")

    def test_rewrite_immediate_and_later_gains_are_separate(self):
        r = screen_trajectory(
            fixture(
                [1.0, 2.0, 3.0],
                [
                    "void f(){int x=1;}",
                    "void g(){cublasSgemm(a,b,c,d,e,f,g,h,i,j,k,l,m);}",
                    "void g(){cublasSgemm(a,b,c,d,e,f,g,h,i,j,k,l,n);}",
                ],
            ),
            Protocol(),
        )
        e = r["rewrite_episodes"][0]
        self.assertTrue(e["rewrite_already_improved"])
        self.assertEqual(e["gain_over_baseline"], 2.0)
        self.assertEqual(e["gain_after_rewrite_best"], 0.5)

    def test_math_modes_remain_distinct_api_facts(self):
        r = fixture(
            [1.0, 1.1],
            [
                "void f(){cublasSetMathMode(h,CUBLAS_TF32_TENSOR_OP_MATH);}",
                "void g(){cublasLtMatmulDescCreate(d,CUBLAS_COMPUTE_32F_FAST_TF32,CUDA_R_32F);}",
            ],
        )
        a = decision_facts(r["turns"][0])
        b = decision_facts(r["turns"][1])
        self.assertEqual(a[0]["kind"], b[0]["kind"])
        self.assertNotEqual(a[0]["values"], b[0]["values"])

    def test_selection_is_unique_deterministic_and_includes_controls(self):
        base = screen_trajectory(fixture([1.0, None, 1.2]), Protocol())
        rows = []
        for i in range(40):
            row = copy.deepcopy(base)
            row["trajectory_id"] = str(i)
            row["identity"] = ["model", "dev", str(i), "1", str(i)]
            row["source_path"] = "fixture"
            if i >= 20:
                row["flags"] = {k: False for k in row["flags"]}
                row["any_candidate"] = False
            rows.append(row)
        selection = review_selection(rows, Protocol())
        self.assertEqual(selection, review_selection(list(reversed(rows)), Protocol()))
        self.assertEqual(len(selection), 20)
        self.assertEqual(len({s["trajectory_id"] for s in selection}), 20)
        self.assertEqual(sum(s["stratum"].startswith("control") for s in selection), 10)


if __name__ == "__main__":
    unittest.main()
