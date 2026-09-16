"""CPU checks of explicitly scoped cuBLAS/cuDNN host-state transition rules."""

import unittest
from .native_state import interpret as interpret_partial

NUM_GPUS = 0


def interpret(events):
    return interpret_partial(events, coverage=["cublasSetWorkspace_v2", "cublasSetStream_v2"])


def event(seq, api, *, obj="h", resource="p", size=128, value="", status=0):
    return {
        "sequence": seq,
        "api": api,
        "object": obj,
        "resource": resource,
        "bytes": size,
        "value_name": value,
        "value": -1,
        "status": status,
        "call_id": 0,
    }


class NativeStateChecks(unittest.TestCase):
    def test_write_errors_make_workspace_unknown(self):
        result = interpret_partial(
            [event(0, "cublasCreate_v2"), event(1, "cublasGemmEx")],
            coverage=["cublasSetWorkspace_v2", "cublasSetStream_v2"],
            write_errors=1,
        )
        self.assertEqual(
            result["gemm_state"][0]["observed_handle_state"]["workspace"]["kind"],
            "unknown_due_to_probe_coverage",
        )

    def test_missing_hook_coverage_does_not_claim_effective_workspace(self):
        result = interpret_partial([event(0, "cublasCreate_v2"), event(1, "cublasGemmEx")])
        self.assertEqual(
            result["gemm_state"][0]["observed_handle_state"]["workspace"]["kind"], "unknown_due_to_probe_coverage"
        )

    def test_dropped_events_make_workspace_unknown(self):
        result = interpret_partial(
            [event(0, "cublasCreate_v2"), event(1, "cublasGemmEx")],
            coverage=["cublasSetWorkspace_v2", "cublasSetStream_v2"],
            dropped_events=1,
        )
        self.assertEqual(
            result["gemm_state"][0]["observed_handle_state"]["workspace"]["kind"], "unknown_due_to_probe_coverage"
        )

    def test_workspace_binding_is_reset_by_even_same_stream_set(self):
        result = interpret(
            [
                event(0, "cublasCreate_v2"),
                event(1, "cublasSetWorkspace_v2"),
                event(2, "cublasSetStream_v2"),
                event(3, "cublasGemmEx", value="FAST_TF32"),
            ]
        )
        self.assertEqual(
            result["gemm_state"][0]["observed_handle_state"]["workspace"]["kind"], "default_after_SetStream"
        )
        self.assertEqual(len(result["workspace_resets"]), 1)

    def test_workspace_set_after_stream_remains_bound(self):
        result = interpret(
            [
                event(0, "cublasCreate_v2"),
                event(1, "cublasSetStream_v2"),
                event(2, "cublasSetWorkspace_v2"),
                event(3, "cublasGemmEx"),
            ]
        )
        self.assertEqual(result["gemm_state"][0]["observed_handle_state"]["workspace"]["kind"], "user_bound")

    def test_failed_api_does_not_change_observed_state(self):
        result = interpret(
            [
                event(0, "cublasCreate_v2"),
                event(1, "cublasSetWorkspace_v2"),
                event(2, "cublasSetStream_v2", status=1),
                event(3, "cublasGemmEx"),
            ]
        )
        self.assertEqual(len(result["workspace_resets"]), 0)
        self.assertEqual(result["gemm_state"][0]["observed_handle_state"]["workspace"]["kind"], "user_bound")

    def test_descriptor_pointer_reuse_creates_distinct_lifetimes(self):
        result = interpret(
            [
                event(0, "cudnnCreateConvolutionDescriptor"),
                event(1, "cudnnDestroyConvolutionDescriptor"),
                event(2, "cudnnCreateConvolutionDescriptor"),
            ]
        )
        self.assertNotEqual(result["lifetimes"][0]["id"], result["lifetimes"][1]["id"])
        self.assertEqual(result["lifetimes"][0]["destroyed_sequence"], 1)

    def test_late_capture_keeps_prior_state_unknown(self):
        result = interpret([event(0, "cublasGemmEx")])
        self.assertEqual(result["gemm_state"][0]["observed_handle_state"]["workspace"]["kind"], "unknown")

    def test_descriptor_state_snapshot_not_mutated_by_future_set(self):
        result = interpret(
            [
                event(0, "cublasLtMatmulDescCreate", value="FAST_TF32"),
                event(1, "cublasLtMatmulDescSetAttribute", value="BIAS", resource="p1"),
                event(2, "cublasLtMatmul"),
                event(3, "cublasLtMatmulDescSetAttribute", value="BIAS", resource="p2"),
                event(4, "cublasLtMatmul"),
            ]
        )
        self.assertEqual(result["gemm_state"][0]["observed_descriptor_state"]["attributes"]["BIAS"]["pointer"], "p1")
        self.assertEqual(result["gemm_state"][1]["observed_descriptor_state"]["attributes"]["BIAS"]["pointer"], "p2")


if __name__ == "__main__":
    unittest.main()
