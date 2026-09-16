"""CPU contracts for bounded whole-forward ordered retry, not graph splicing."""

import copy
import unittest

from .check_runtime_trace import graph, node
from .service_replay import _same_capture_interfaces, ordered_retry_plan


class OrderedRetry(unittest.TestCase):
    def setUp(self):
        self.graph = graph(
            [
                node(
                    "kernel:1",
                    1,
                    [("a", 0, 16)],
                    [("a", 0, 16)],
                    evidence={"launch_id": 0},
                    unknowns=["inplace_order_requires_detailed_capture"],
                )
            ]
        )
        self.graph["coverage"] = {"total_launches_reported": 1, "kernel_launches": 1, "completed_kernel_launches": 1}
        self.result = {
            "status": "ok",
            "alignment": {"scored_control": True, "control_trace": True},
            "graph": self.graph,
        }

    def test_selects_complete_inplace_launch(self):
        self.assertEqual(ordered_retry_plan(self.result), [0])

    def test_atomic_order_cannot_be_resolved_by_thread_runs(self):
        self.graph["nodes"][0]["writes"][0]["atomic_unknown"] = True
        self.assertEqual(ordered_retry_plan(self.result), [])

    def test_never_retries_unaligned_or_incomplete_capture(self):
        for key in ["scored_control", "control_trace"]:
            for value in [None, False]:
                result = copy.deepcopy(self.result)
                result["alignment"][key] = value
                self.assertEqual(ordered_retry_plan(result), [])
        self.graph["nodes"][0]["footprint_complete"] = False
        self.assertEqual(ordered_retry_plan(self.result), [])
        self.graph["nodes"][0]["footprint_complete"] = True
        self.graph["coverage"]["completed_kernel_launches"] = 0
        self.assertEqual(ordered_retry_plan(self.result), [])

    def test_order_evidence_may_improve_but_interface_must_match(self):
        detailed = copy.deepcopy(self.graph)
        detailed["nodes"][0]["unknowns"] = []
        detailed["nodes"][0]["reads"][0]["entry_regions"] = [[0, 16]]
        detailed["buffers"][0]["base"] += 4096
        detailed["nodes"][0]["stream"] = 500
        self.assertTrue(_same_capture_interfaces(self.graph, detailed))
        detailed["nodes"][0]["implementation"] = "different"
        self.assertFalse(_same_capture_interfaces(self.graph, detailed))

    def test_changed_alias_or_region_rejected(self):
        for change in ["buffer", "regions"]:
            detailed = copy.deepcopy(self.graph)
            detailed["nodes"][0]["reads"][0][change] = "b" if change == "buffer" else [[0, 8]]
            self.assertFalse(_same_capture_interfaces(self.graph, detailed))


if __name__ == "__main__":
    unittest.main()
