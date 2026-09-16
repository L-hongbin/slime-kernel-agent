"""CPU-only behavioral checks for the offline structure-analysis workflow."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from .analyze import grouped, normalize, run
from .structure import align_components, analyze_response, analyze_trajectory


def response(cuda=None, binding=None, model=None):
    cuda = (
        cuda
        if cuda is not None
        else '__global__ void k(float*x){x[threadIdx.x] *= 32;}\nextern "C" void launch(float*x){k<<<1,32>>>(x);}'
    )
    binding = (
        binding
        if binding is not None
        else "void launch(float*);\nvoid wrapper(float*x){launch(x);}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(run,wrapper);"
    )
    model = (
        model
        if model is not None
        else "import tvm_ffi_extension\nclass ModelNew:\n    def forward(self,x):\n        return tvm_ffi_extension.run(x)\n"
    )
    return "\n\n".join(
        f"### {name}\n```{lang}\n{body}\n```"
        for name, lang, body in [
            ("CUDA_KERNELS", "cpp", cuda),
            ("APPLY_BINDINGS", "cpp", binding),
            ("MODEL_NEW", "python", model),
        ]
    )


def traj(*responses):
    return analyze_trajectory([{"turn_idx": i, "response": r} for i, r in enumerate(responses)])


class StructureChecks(unittest.TestCase):
    def test_shadowed_parameter_is_not_resolved_to_global_function(self):
        r = analyze_response(
            response(
                "void f(){} void g(void (*f)()){f();}",
                model="def f():\n    pass\nclass ModelNew:\n    def forward(self,f):\n        return f()",
            )
        )
        edges = [e for e in r["calls"] if e["target"] == "f"]
        self.assertEqual(len(edges), 2)
        self.assertTrue(
            all(e["resolution"] == "local_binding_unresolved" and not e["candidate_targets"] for e in edges)
        )

    def test_cuda_launch_and_python_export_chain(self):
        r = analyze_response(response())
        self.assertTrue(all(s["syntax_valid"] for s in r["sections"].values()))
        edges = r["calls"]
        self.assertTrue(
            any(
                e["kind"] == "kernel_launch" and e["launch_config"] == ["1", "32"] and len(e["candidate_targets"]) == 1
                for e in edges
            )
        )
        self.assertTrue(
            any(e["target"] == "tvm_ffi_extension.run" and len(e["candidate_targets"]) == 1 for e in edges)
        )

    def test_actual_parser_prefers_complete_trio_over_trailing_partial(self):
        raw = response() + "\n### CUDA_KERNELS\n```cpp\nvoid broken("
        r = analyze_response(raw)
        self.assertEqual(r["selection_mode"], "last_complete_group")
        self.assertTrue(r["valid_modelnew"])
        self.assertNotIn("broken", r["sections"]["CUDA_KERNELS"]["source"])

    def test_complete_thinking_block_is_not_executed_source(self):
        r = analyze_response("<think>" + response("void misleading(){}") + "</think>" + response())
        self.assertNotIn("misleading", [c["name"] for c in r["components"]])

    def test_incomplete_response_is_not_low_similarity_rewrite(self):
        r = traj(response(), response()[:-3])
        self.assertFalse(r["turns"][1]["complete_sections"])
        self.assertTrue(all(v["token_similarity"] is None for v in r["transitions"][0]["sections"].values()))

    def test_comments_and_formatting_do_not_create_edits(self):
        before = response("void f(){int x=32;}")
        after = response("// void fake(){}\nvoid f() { /* note */ int x = 32; }")
        r = traj(before, after)
        self.assertFalse(r["edit_events"])
        self.assertFalse(any(c["name"] == "fake" for c in r["turns"][1]["components"]))

    def test_cpp_string_whitespace_and_include_are_significant(self):
        r = traj(
            response('#include <a.h>\nvoid f(){printf("a b");}'), response('#include <b.h>\nvoid f(){printf("ab");}')
        )
        changed = [c for c in r["transitions"][0]["changes"] if c["status"] == "changed"]
        self.assertEqual(len(changed), 2)

    def test_token_permutation_is_not_reported_identical(self):
        r = traj(response("void f(){x=a-b;}"), response("void f(){x=b-a;}"))
        self.assertLess(r["transitions"][0]["sections"]["CUDA_KERNELS"]["token_similarity"], 1)

    def test_nested_pragma_is_preserved_without_false_syntax_error(self):
        a = "void f(){\n#pragma unroll\nfor(int i=0;i<4;i++)\n#pragma unroll 4\nfor(int j=0;j<4;j++)x[i]=0;\n}"
        b = a.replace("#pragma unroll 4", "#pragma unroll 2")
        r = traj(response(a), response(b))
        for t in r["turns"]:
            self.assertTrue(t["sections"]["CUDA_KERNELS"]["syntax_valid"])
        self.assertTrue(any(e["component"] == "f" for e in r["edit_events"]))
        self.assertIn("#pragma unroll 4", r["turns"][0]["components"][0]["source"])

    def test_pragma_inside_raw_string_is_not_masked(self):
        r = analyze_response(response('void f(){auto s=R"tag(\n#pragma unroll\n)tag";}'))
        self.assertFalse(r["sections"]["CUDA_KERNELS"]["parser_adaptations"])

    def test_python_indentation_is_significant(self):
        a = "class ModelNew:\n    def forward(self,x):\n        if x:\n            y=1\n        return y"
        b = "class ModelNew:\n    def forward(self,x):\n        if x:\n            y=1\n            return y"
        r = traj(response(model=a), response(model=b))
        self.assertTrue(any(c["status"] == "changed" for c in r["transitions"][0]["changes"]))

    def test_literal_and_macro_modification_retention(self):
        versions = [response(f"#define TILE {v}\nvoid f(){{int x=TILE;}}") for v in [32, 64, 32, 64]]
        r = traj(*versions)
        e = next(e for e in r["edit_events"] if e["component"] == "TILE" and e["introduced_turn"] == 1)
        self.assertEqual([o["state"] for o in e["observations"]], ["present", "reverted", "present"])
        self.assertEqual(e["reintroduced_turns"], [3])
        self.assertEqual(e["nonadjacent_retained_turns"], [3])
        self.assertEqual(e["applied_turns"], [1, 3])

    def test_context_drift_is_unknown_not_removed(self):
        r = traj(response("void f(){a=32;b=1;}"), response("void f(){a=64;b=1;}"), response("void f(){different();}"))
        e = next(e for e in r["edit_events"] if e["introduced_turn"] == 1 and e["component"] == "f")
        self.assertEqual(e["observations"][-1]["state"], "unknown")

    def test_incomplete_intermediate_does_not_erase_identity(self):
        a = response("void f(){int x=32;}")
        b = response("void f(){int x=64;}")
        r = traj(a, b, "incomplete text", b)
        e = next(e for e in r["edit_events"] if e["component"] == "f" and e["introduced_turn"] == 1)
        self.assertEqual([o["state"] for o in e["observations"]], ["present", "unobserved", "present"])
        self.assertFalse(e["reintroduced_turns"])

    def test_missing_turn_is_explicit(self):
        r = analyze_trajectory([{"turn_idx": 0, "response": response()}, {"turn_idx": 2, "response": response()}])
        self.assertEqual(r["missing_turns"], [1])
        self.assertFalse(r["transitions"][0]["consecutive"])

    def test_duplicate_turn_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_trajectory([{"turn_idx": 0, "response": response()}] * 2)

    def test_namespace_and_overload_are_not_merged(self):
        code = "namespace a {void f(int x){}} namespace b {void f(int x){}}\nvoid g(int x){} void g(float x){}"
        r = analyze_response(response(code))
        f = [c for c in r["components"] if c["kind"] == "function" and c["section"] == "CUDA_KERNELS"]
        self.assertIn("a::f", [c["qualified_name"] for c in f])
        self.assertIn("b::f", [c["qualified_name"] for c in f])
        pairs, _, _ = align_components(f, f)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(sum(reason == "exact_overload_signature" for _, _, reason in pairs), 4)

    def test_exact_rename_is_only_a_candidate(self):
        r = traj(response("void a(){int x=1;}"), response("void b(){int x=1;}"))
        self.assertTrue(r["transitions"][0]["rename_candidates"])
        a = next(c for c in r["turns"][0]["components"] if c["name"] == "a")
        b = next(c for c in r["turns"][1]["components"] if c["name"] == "b")
        self.assertNotEqual(a["lineage"], b["lineage"])

    def test_ambiguous_rename_is_not_forced(self):
        r = traj(response("void a(){} void b(){}"), response("void c(){} void d(){}"))
        self.assertFalse(
            any(c["alignment"] == "unique_exact_syntax_rename_candidate" for c in r["transitions"][0]["changes"])
        )

    def test_new_function_can_be_tracked(self):
        a = response("void f(){}")
        b = response("void f(){} void g(){int x=1;}")
        r = traj(a, b, b, b)
        e = next(e for e in r["edit_events"] if e["event_type"] == "component_added" and e["component"] == "g")
        self.assertEqual(e["nonadjacent_retained_turns"], [3])

    def test_component_disappearance_and_reappearance(self):
        a = response("void f(){int x=1;}")
        b = response("void g(){int x=2;}")
        r = traj(a, b, a, a)
        self.assertTrue(
            any(
                e["component"] == "f" and e["intervening_unmatched_component"]
                for e in r["component_version_recurrences"]
            )
        )
        self.assertEqual(sum(e["component"] == "f" for e in r["component_version_recurrences"]), 1)

    def test_repeated_prototypes_do_not_collide(self):
        r = analyze_response(response(binding="void launch(float*); void launch(float*);"))
        ids = [c["id"] for c in r["components"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_multi_declaration_is_explicitly_unsegmented(self):
        r = analyze_response(response("void f(), g();"))
        self.assertTrue(r["sections"]["CUDA_KERNELS"]["coverage_warnings"])
        context = next(c for c in r["components"] if c["section"] == "CUDA_KERNELS" and c["kind"] == "file_context")
        self.assertIn("g", context["source"])
        self.assertFalse(any(c["kind"] == "prototype" and c["section"] == "CUDA_KERNELS" for c in r["components"]))

    def test_function_pointer_is_not_prototype(self):
        r = analyze_response(response("void (*cb)();"))
        self.assertFalse(any(c["section"] == "CUDA_KERNELS" and c["kind"] == "prototype" for c in r["components"]))
        context = next(c for c in r["components"] if c["section"] == "CUDA_KERNELS" and c["kind"] == "file_context")
        self.assertIn("cb", context["source"])

    def test_python_definition_in_exception_handler(self):
        r = analyze_response(
            response(
                model="try:\n    pass\nexcept Exception:\n    def helper():\n        return 1\nclass ModelNew:\n    def forward(self,x):\n        return helper()"
            )
        )
        self.assertTrue(any(c["qualified_name"] == "helper" for c in r["components"]))

    def test_invalid_python_and_native_are_observable(self):
        r = analyze_response(response("void f(){", model="class ModelNew:\n    def forward("))
        self.assertFalse(r["sections"]["CUDA_KERNELS"]["syntax_valid"])
        self.assertFalse(r["sections"]["MODEL_NEW"]["syntax_valid"])
        self.assertTrue(r["sections"]["CUDA_KERNELS"]["errors"])

    def test_different_cuda_algorithm_not_hidden_by_unchanged_wrappers(self):
        a = response("void launch(float*x){for(int i=0;i<32;i++)x[i]*=2;}")
        b = response("void launch(float*x){cublasSgemm(x);}")
        r = traj(a, b)
        c = r["transitions"][0]["sections"]
        self.assertLess(c["CUDA_KERNELS"]["token_similarity"], 1)
        self.assertEqual(c["MODEL_NEW"]["token_similarity"], 1)
        self.assertTrue(any(e["kind"] == "library_call" for e in r["turns"][1]["calls"]))

    def test_end_to_end_jsonl_and_cross_dataset_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "input.jsonl"
            rows = [
                {"group_id": 0, "metadata": {"turn_idx": t}, "label": {"ground_truth": ref}, "response": response()}
                for ref in ["task A", "task B"]
                for t in range(2)
            ]
            src.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(len(grouped("model", src)), 2)
            run(
                SimpleNamespace(
                    input=[f"model={src}"],
                    output_dir=root / "out",
                    group_id=[],
                    review_group=["0"],
                    max_trajectories=None,
                )
            )
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(summary["counts"]["trajectories"], 2)
            self.assertEqual(summary["counts"]["turns"], 4)
            self.assertEqual(len(list((root / "out/trajectories").glob("*.md"))), 2)
            self.assertTrue(json.loads((root / "out/manifest.json").read_text())["code_and_inputs_unchanged"])

    def test_trusted_pt_input(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "eval.pt"
            torch.save(
                {"samples": [{"group_id": 4, "metadata": {"turn_idx": 0}, "response": response(), "reward": 0.25}]},
                src,
            )
            rows = next(iter(grouped("model", src).values()))
            self.assertEqual(rows[0]["reward_observed"], 0.25)
            self.assertEqual(rows[0]["turn_idx"], 0)

    def test_delete_restore_reintroduces_edit_with_basis(self):
        r = traj(
            response("void f(){int x=32;}"),
            response("void f(){int x=64;}"),
            response("void g(){int y=1;}"),
            response("void f(){int x=64;}"),
        )
        e = next(e for e in r["edit_events"] if e["component"] == "f" and e["introduced_turn"] == 1)
        self.assertEqual([o["state"] for o in e["observations"]], ["present", "component_absent", "present"])
        self.assertEqual(e["reintroduced_turns"], [3])
        self.assertEqual(e["reintroductions"][0]["after_state"], "component_absent")
        restored = next(c for c in r["turns"][3]["components"] if c["name"] == "f")
        self.assertIn(restored["id"], r["transitions"][2]["reconnected_after"])
        self.assertNotIn(restored["id"], r["transitions"][2]["unmatched_after"])

    def test_signature_replacement_does_not_share_lineage(self):
        r = traj(response("void g(int x){x=1;}"), response("void g(float x){x=2;}"))
        gs = [next(c for c in t["components"] if c["name"] == "g") for t in r["turns"]]
        self.assertNotEqual(gs[0]["lineage"], gs[1]["lineage"])

    def test_native_scope_and_declaration_order(self):
        cuda = "void k(){} void g(){ k(); {int k=1;} for(int k=0;k<2;k++){} k(); int k=0; k(); }"
        r = analyze_response(response(cuda))
        calls = [c for c in r["calls"] if c["target"] == "k"]
        self.assertEqual(
            [c["resolution"] for c in calls],
            ["unique_name_candidate", "unique_name_candidate", "local_binding_unresolved"],
        )

    def test_invalid_sibling_does_not_create_unobservable_event(self):
        r = traj(response("void f(){int x=32;} void g(){}"), response("void f(){int x=64;} void g(){"))
        self.assertFalse(r["edit_events"])
        self.assertTrue(any(c["status"] == "changed" for c in r["transitions"][0]["changes"]))

    def test_environment_feedback_is_not_claimed_as_rendered_feedback(self):
        n = normalize(
            {
                "group_id": 0,
                "turn_idx": 0,
                "response": response(),
                "problem_family": "network",
                "env_result": {"env_state": {"error_message": "compiler failure"}},
            }
        )
        self.assertTrue(n["saved_environment_feedback_available"])
        self.assertFalse(n["saved_model_feedback_available"])
        self.assertEqual(n["environment_error_message"], "compiler failure")
        self.assertEqual(n["problem_family"], "network")

    def test_fallback_matches_executor_but_is_explicit(self):
        sections = response().split("\n\n### ")
        raw = "### " + sections[2] + "\n\n" + sections[0] + "\n\n### " + sections[1]
        r = analyze_response(raw)
        self.assertEqual(r["selection_mode"], "partial_fallback")
        self.assertTrue(r["complete_sections"])

    def test_quoted_macro_export_is_not_normalized_to_identifier(self):
        r = analyze_response(
            response(binding='void wrapper(float*x){}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC("run",wrapper);')
        )
        self.assertFalse(next(c for c in r["calls"] if c["kind"] == "extension_call")["candidate_targets"])


if __name__ == "__main__":
    unittest.main()
