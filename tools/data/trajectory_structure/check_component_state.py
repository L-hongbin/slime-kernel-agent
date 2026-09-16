"""Focused CPU checks for source-state facts and runtime/source inventory joins."""

import unittest

from .check_structure import response
from .component_state import runtime_inventory, state_facts
from .structure import analyze_response

NUM_GPUS = 0


class StateChecks(unittest.TestCase):
    def test_globals_static_locals_and_object_fields_not_automatic_temporaries(self):
        turn = analyze_response(
            response(
                cuda="""
            static float* global_buffer;
            struct Plan { int capacity; float* workspace; };
            void f() { static int initialized = 0; int temporary = 1; }
        """
            )
        )
        facts = [f for f in state_facts(turn)["state_facts"] if f["kind"] == "native_state_declaration"]
        by_name = {f["name"]: f for f in facts}
        self.assertEqual(by_name["initialized"]["storage"], "function_static")
        self.assertEqual(by_name["global_buffer"]["storage"], "file_scope")
        self.assertEqual(by_name["workspace"]["storage"], "object_field")
        self.assertNotIn("temporary", by_name)

    def test_prototypes_are_not_storage(self):
        turn = analyze_response(response(cuda="void helper(int x); extern float* buffer;"))
        names = [f["name"] for f in state_facts(turn)["state_facts"] if f["kind"] == "native_state_declaration"]
        self.assertEqual(names, ["buffer"])

    def test_private_python_state_and_init_only_owner(self):
        turn = analyze_response(
            response(
                model="""
class ModelNew:
    def __init__(self):
        self._cache = None
    def forward(self, x):
        self._cache = x
        return x
"""
            )
        )
        facts = [f for f in state_facts(turn)["state_facts"] if f["kind"] == "python_instance_assignment"]
        self.assertEqual(len(facts), 2)
        self.assertNotEqual(facts[0]["owner_component"], facts[1]["owner_component"])
        self.assertTrue(all(f["execution_status"] == "static_candidate" for f in facts))

    def test_native_library_descriptor_creation_is_resource_operation(self):
        turn = analyze_response(
            response(cuda="void f(){cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);}")
        )
        facts = state_facts(turn)
        self.assertTrue(any(f.get("target") == "cublasLtMatmulDescCreate" for f in facts["state_facts"]))
        self.assertTrue(any(f["kind"] == "math_mode" for f in facts["configuration_facts"]))

    def test_allocation_arguments_remain_expressions_not_claimed_sizes(self):
        turn = analyze_response(response(cuda="void f(){cudaMalloc(&ptr, batch * width * sizeof(float));}"))
        fact = next(f for f in state_facts(turn)["state_facts"] if f["kind"] == "native_resource_operation")
        self.assertIn("batch * width", fact["arguments"][1])
        self.assertNotIn("bytes", fact)

    def test_runtime_shared_storage_is_presence_not_read_write(self):
        turn = analyze_response(response())
        obs = {
            "calls": [
                {
                    "call_id": 0,
                    "export": "run",
                    "phase": "forward_0",
                    "caller": {"file": "/tmp/model_new.py", "line": 4},
                    "args_before": {"leaves": [{"kind": "tensor", "storage_id": "s1"}]},
                }
            ],
            "cuda_activities": [],
            "snapshots": [
                {
                    "phase": "forward_0",
                    "label": "after",
                    "model_state": {
                        "at_item_limit": False,
                        "leaves": [{"path": ["", "attributes", "_buf"], "kind": "tensor", "storage_id": "s1"}],
                    },
                }
            ],
        }
        joined = runtime_inventory(turn, obs)
        self.assertEqual(joined["python_tensor_state"][0]["passed_as_ffi_argument_calls"], [0])
        self.assertEqual(joined["python_tensor_state"][0]["read_write_or_necessity"], "unknown")

    def test_export_only_does_not_invent_tensor_edges(self):
        turn = analyze_response(response())
        obs = {
            "calls": [
                {"call_id": 0, "export": "run", "phase": "f", "caller": {"file": "/tmp/model_new.py", "line": 4}}
            ],
            "cuda_activities": [],
            "snapshots": [],
        }
        self.assertEqual(runtime_inventory(turn, obs)["python_tensor_state"], [])


if __name__ == "__main__":
    unittest.main()
