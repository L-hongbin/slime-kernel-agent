"""CPU checks for reference-aligned state graphs and GEMM address contracts."""

import hashlib
import unittest

from .semantic_state import (
    actual_arguments,
    bind_scalar_arguments,
    check_gemm_indices,
    check_mask_stride,
    compare_cards,
    gemm_descriptor,
    integer_expr,
    integer_locals_before_call,
    node,
    propagate_contract_warnings,
    validate_graph,
)
from .state_aggregation import interface_state, make_card

NUM_GPUS = 0


def descriptor(**changes):
    return {
        "status": "resolved",
        "transa": "CUBLAS_OP_T",
        "transb": "CUBLAS_OP_N",
        "m": 4,
        "n": 2,
        "k": 3,
        "lda": 3,
        "ldb": 3,
        "ldc": 4,
        "A": "W",
        "B": "X",
        "C": "Y",
        "batched": False,
    } | changes


def graph():
    return [
        node(
            "linear",
            "cublas",
            ["input", "weight"],
            "output",
            [{"line": 1}],
            candidate_activation_inputs=["input"],
            candidate_parameter_inputs=["weight"],
        )
    ]


def card(nodes=None):
    return {
        "reference_sha256": "ref",
        "mapping_complete": True,
        "nodes": nodes or graph(),
        "unknowns": ["local_correctness"],
        "orchestration": "native",
        "allocation": "separate",
        "math_mode_requests": [],
        "interface": {"status": "direct_names_resolve"},
        "evaluation_evidence": {"correct": False},
    }


class StructuralStateChecks(unittest.TestCase):
    def test_safe_integer_arithmetic(self):
        self.assertEqual(integer_expr("(long)M * 3 * C", {"M": 2, "C": 4}), 24)
        self.assertEqual(integer_expr("C / nh", {"C": 768, "nh": 8}), 96)

    def test_does_not_execute_expression(self):
        with self.assertRaises(ValueError):
            integer_expr("__import__('os').system('false')", {})

    def test_unknown_dimension_fails(self):
        with self.assertRaises(KeyError):
            integer_expr("unknown + 1", {})

    def test_stride_is_read_from_source_not_expected_value(self):
        owner = {"line": 10, "source": "void helper() {\n long long strideA = T * hs + 1;\n call();\n}"}
        values = integer_locals_before_call(owner, {"line": 12}, {"T": 512, "hs": 96})
        self.assertEqual(values["strideA"], 49153)

    def test_unresolved_local_does_not_keep_guessed_value(self):
        owner = {"line": 10, "source": "void helper() {\n int strideA = dynamic_call();\n call();\n}"}
        values = integer_locals_before_call(owner, {"line": 12}, {"strideA": 123})
        self.assertNotIn("strideA", values)

    def test_helper_dimensions_come_from_driver_actual_arguments(self):
        owner = {"line": 1, "source": "void driver() {\n int M = B * T;\n call();\n}"}
        call = {"line": 3, "arguments": ["x", "w", "b", "y", "M", "4*C", "C"]}
        bound = bind_scalar_arguments(owner, call, {"M": 4, "Nf": 5, "K": 6}, {"B": 2, "T": 3, "C": 4, "Nf": 12})
        self.assertEqual((bound["M"], bound["Nf"], bound["K"]), (6, 16, 4))

    def test_unknown_driver_dimension_is_not_reference_default(self):
        owner = {"line": 1, "source": "void driver() {\n call();\n}"}
        bound = bind_scalar_arguments(owner, {"line": 2, "arguments": ["dynamic()"]}, {"Nf": 0}, {"Nf": 12})
        self.assertNotIn("Nf", bound)

    def test_mask_fix_is_recorded_as_fix(self):
        values = {"T": 512, "max_seqlen": 1024}
        self.assertEqual(check_mask_stride("i*T+j", values, 1024)["status"], "conflict")
        r = check_mask_stride("i*max_seqlen+j", values, 1024)
        self.assertEqual((r["status"], r["observed_row_stride"]), ("checked_points_consistent", 1024))

    def test_comment_nodes_are_not_function_arguments(self):
        c = {"arguments": ["x", "// comment", "W", "/* note */", "out"]}
        self.assertEqual(actual_arguments(c), ["x", "W", "out"])

    def test_local_pass_keeps_upstream_conflict(self):
        a = node(
            "qk",
            "matmul",
            ["input"],
            "scores",
            [{}],
            candidate_activation_inputs=["input"],
            index_contract={"status": "conflict"},
        )
        b = node("softmax", "softmax", ["scores"], "probs", [{}], candidate_activation_inputs=["scores"])
        c = node(
            "pv",
            "matmul",
            ["probs"],
            "output",
            [{}],
            candidate_activation_inputs=["probs"],
            index_contract={"status": "checked_points_consistent"},
        )
        propagate_contract_warnings([a, b, c])
        self.assertEqual(c["upstream_contract_conflicts"], ["qk"])
        self.assertEqual(c["local_correctness"], "unknown_not_tested_separately")

    def test_unknown_wire_does_not_invent_dependency(self):
        n = node("unknown", "unknown", ["input"], "output", [{}], candidate_activation_inputs=None)
        propagate_contract_warnings([n])
        self.assertEqual(n["dependency_evidence"], "unknown_wiring")

    def test_unknown_mask_does_not_invent_stride(self):
        r = check_mask_stride("i*dynamic+j", {}, 1024)
        self.assertEqual(r["status"], "unknown")
        self.assertIsNone(r["observed_row_stride"])

    def test_correct_linear_descriptor(self):
        r = check_gemm_indices(descriptor(), "linear", 2, 4, 3)
        self.assertEqual(r["status"], "checked_points_consistent")
        self.assertEqual(r["checked_factor_addresses"], 24)

    def test_wrong_weight_transpose_detected(self):
        r = check_gemm_indices(descriptor(transa="CUBLAS_OP_N", lda=4), "linear", 2, 4, 3)
        self.assertEqual(r["status"], "conflict")

    def test_swapped_row_column_interpretation_detected(self):
        d = descriptor(transa="CUBLAS_OP_N", transb="CUBLAS_OP_T", A="X", B="W", m=2, n=4, k=3, lda=2, ldb=4, ldc=2)
        self.assertEqual(check_gemm_indices(d, "linear", 2, 4, 3)["reason"], "operand_address")

    def test_qk_orientation(self):
        d = descriptor(A="K", B="Q", C="P", m=4, n=4, k=3, lda=3, ldb=3, ldc=4)
        self.assertEqual(check_gemm_indices(d, "qk", 4, 4, 3)["status"], "checked_points_consistent")
        d.update(A="Q", B="K")
        self.assertEqual(check_gemm_indices(d, "qk", 4, 4, 3)["status"], "conflict")

    def test_pv_orientation(self):
        d = descriptor(transa="CUBLAS_OP_N", A="V", B="P", C="O", m=3, n=4, k=4, lda=3, ldb=4, ldc=3)
        self.assertEqual(check_gemm_indices(d, "pv", 4, 3, 4)["status"], "checked_points_consistent")
        d["transb"] = "CUBLAS_OP_T"
        self.assertEqual(check_gemm_indices(d, "pv", 4, 3, 4)["status"], "conflict")

    def test_wrong_leading_dimension(self):
        self.assertEqual(check_gemm_indices(descriptor(lda=1), "linear", 2, 4, 3)["reason"], "leading_dimension")

    def test_wrong_batch_stride(self):
        d = descriptor(
            A="K",
            B="Q",
            C="P",
            m=4,
            n=4,
            k=3,
            lda=3,
            ldb=3,
            ldc=4,
            batched=True,
            strideA=12,
            strideB=12,
            strideC=15,
            batch_count=2,
        )
        self.assertEqual(check_gemm_indices(d, "qk", 4, 4, 3)["reason"], "batch_stride")

    def test_wrong_batch_count(self):
        d = descriptor(
            A="K",
            B="Q",
            C="P",
            m=4,
            n=4,
            k=3,
            lda=3,
            ldb=3,
            ldc=4,
            batched=True,
            strideA=12,
            strideB=12,
            strideC=16,
            batch_count=2,
        )
        self.assertEqual(check_gemm_indices(d, "qk", 4, 4, 3, expected_batch=3)["reason"], "batch_count")

    def test_unsupported_stays_unknown(self):
        self.assertEqual(check_gemm_indices({"status": "unknown"}, "linear", 2, 4, 3)["status"], "unknown")

    def test_descriptor_reads_roles_not_variable_names(self):
        call = {
            "target": "cublasSgemm",
            "arguments": [
                "h",
                "CUBLAS_OP_T",
                "CUBLAS_OP_N",
                "N",
                "M",
                "K",
                "&a",
                "renamed_weight",
                "K",
                "renamed_input",
                "K",
                "&b",
                "renamed_output",
                "N",
            ],
        }
        d = gemm_descriptor(
            call, {"M": 2, "N": 4, "K": 3}, {"renamed_weight": "W", "renamed_input": "X", "renamed_output": "Y"}
        )
        self.assertEqual(check_gemm_indices(d, "linear", 2, 4, 3)["status"], "checked_points_consistent")

    def test_graph_reaches_output(self):
        validate_graph(graph(), ["input", "weight"])

    def test_missing_dependency_rejected(self):
        with self.assertRaisesRegex(ValueError, "unconnected"):
            validate_graph(graph(), ["input"])

    def test_duplicate_slot_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_graph(graph() * 2, ["input", "weight"])

    def test_no_evidence_rejected(self):
        g = graph()
        g[0]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_graph(g, ["input", "weight"])

    def test_whole_program_correct_does_not_mark_node_correct(self):
        c = card()
        c["evaluation_evidence"]["correct"] = True
        self.assertEqual(c["nodes"][0]["local_correctness"], "unknown_not_tested_separately")

    def test_source_names_not_state_identity(self):
        a, b = card(), card()
        a["nodes"][0]["evidence"] = [{"component": "first", "line": 1}]
        b["nodes"][0]["evidence"] = [{"component": "renamed", "line": 200}]
        self.assertTrue(compare_cards(a, b)["same_recorded_state"])

    def test_same_plan_different_contract_not_same_state(self):
        a, b = card(), card()
        a["nodes"][0]["index_contract"] = {"status": "conflict"}
        b["nodes"][0]["index_contract"] = {"status": "checked_points_consistent"}
        p = compare_cards(a, b)
        self.assertEqual(p["relation"], "same_computation_plan")
        self.assertFalse(p["same_recorded_state"])

    def test_missing_mapping_not_merged(self):
        a, b = card(), card()
        a["mapping_complete"] = False
        self.assertEqual(compare_cards(a, b)["relation"], "insufficient_mapping")

    def test_unknown_wiring_is_not_equality(self):
        a, b = card(), card()
        a["nodes"][0]["candidate_activation_inputs"] = None
        b["nodes"][0]["candidate_activation_inputs"] = None
        self.assertEqual(compare_cards(a, b)["relation"], "insufficient_wiring")

    def test_same_reference_graph_different_actual_input(self):
        a, b = card(), card()
        b["nodes"][0]["candidate_activation_inputs"] = ["wrong_intermediate"]
        self.assertEqual(compare_cards(a, b)["relation"], "different_computation_plan")

    def test_weight_transpose_is_not_dropped(self):
        a, b = card(), card()
        b["nodes"][0]["candidate_parameter_inputs"] = ["weight_transposed"]
        self.assertEqual(compare_cards(a, b)["relation"], "different_computation_plan")

    def test_scale_fusion_placement_is_state_difference(self):
        a, b = card(), card()
        a["nodes"][0]["normalization_placement"] = {"scale": "qk_alpha"}
        b["nodes"][0]["normalization_placement"] = {"scale": "softmax_parameter"}
        self.assertFalse(compare_cards(a, b)["same_recorded_state"])

    def test_different_compile_obligations_not_same_state(self):
        a, b = card(), card()
        a["observed_blockers"] = {"undefined_symbols": ["handle_type"]}
        b["observed_blockers"] = {"undefined_symbols": ["printf"]}
        self.assertFalse(compare_cards(a, b)["same_recorded_state"])

    def test_distinct_tasks_not_merged(self):
        a, b = card(), card()
        b["reference_sha256"] = "another"
        self.assertEqual(compare_cards(a, b)["relation"], "different_task")

    def test_mask_stride_difference_is_state_difference(self):
        a, b = card(), card()
        a["nodes"][0]["mask_storage_contract"] = {"status": "conflict"}
        self.assertFalse(compare_cards(a, b)["same_recorded_state"])

    def test_export_mismatch_localized(self):
        t = {
            "sections": {
                "MODEL_NEW": {
                    "source": "class ModelNew:\n def forward(self,x):\n  return tvm_ffi_extension.missing(x)\n"
                }
            },
            "exports": [{"name": "present"}],
        }
        state, ev = interface_state(t)
        self.assertEqual(state["status"], "missing_export")
        self.assertEqual(ev[0]["missing"], ["missing"])

    def test_dynamic_dispatch_is_unknown(self):
        t = {
            "sections": {
                "MODEL_NEW": {
                    "source": "class ModelNew:\n def forward(self,x):\n  return getattr(tvm_ffi_extension, self.name)(x)\n"
                }
            },
            "exports": [],
        }
        self.assertEqual(interface_state(t)[0]["status"], "unknown_dynamic_dispatch")

    def test_future_rows_cannot_change_current_program_state(self):
        reference = "batch_size=128\ninput_size=16384\nlayer_sizes=[16384,16384]\noutput_size=8192\n"
        digest = hashlib.sha256(reference.encode()).hexdigest()
        turn = {
            "turn_idx": 0,
            "sections": {},
            "exports": [],
            "calls": [],
            "complete_sections": False,
            "valid_modelnew": False,
            "observation": {"response_sha256": hashlib.sha256(b"current").hexdigest()},
        }
        structure = {"identity": ["qwen38", "dataset", digest, "1", "0"], "turns": [turn]}
        raw = [
            {
                "turn_idx": 0,
                "response": "current",
                "label": {"ground_truth": reference},
                "outcome": "precheck_failure",
                "compiled": False,
                "correct": False,
                "compiler_diagnostics": [],
            },
            {"outcome": "correct"},
        ]
        before = make_card(structure, raw, 0)
        raw[1] = {"outcome": "runtime_failure", "response": "future changed"}
        self.assertEqual(before, make_card(structure, raw, 0))


if __name__ == "__main__":
    unittest.main()
